---
title: "Insider trading, Reg FD, and what is actually illegal"
date: "2026-06-19"
publishDate: "2026-06-19"
description: "The precise legal line between a legal information edge and a federal crime, why Reg FD reshaped how news reaches price, and how to build an edge that survives a subpoena."
tags: ["regulation", "insider-trading", "reg-fd", "securities-law", "compliance", "mosaic-theory", "expert-networks", "market-integrity", "enforcement", "trading-edge"]
category: "trading"
subcategory: "Law & Geopolitics"
author: "Hiep Tran"
featured: true
readTime: 41
---

> [!important]
> **TL;DR** — Trading on private information is *not* automatically illegal; it becomes a federal crime only when the information is material, non-public, and obtained or passed on through a *breach of a duty of trust*. That single distinction separates legal research from prison.
>
> - The crime is **fraud on a duty**, not "knowing something others don't." Reading filings, building a mosaic from public bits, and doing scrubbed expert calls are all legal because no breached duty supplies the edge.
> - **Regulation FD (2000)** killed the selective whisper to favored analysts: material news must now reach the whole market at once, which is *why* the modern edge is analysis, not access.
> - For a tip to be illegal, a **personal benefit** must pass to the insider (Dirks/Salman) and the tippee must know of the breach — the test that briefly created, then closed, the *Newman* loophole.
> - **The one number to remember:** the SEC can claw back your gain *and* add a penalty of up to **three times** that gain, before the DOJ adds prison — so the expected value of trading on an illegal tip is sharply negative.

On October 16, 2009, federal agents arrested Raj Rajaratnam, the billionaire founder of the Galleon Group hedge fund, in his Manhattan apartment. It was the first time wiretaps — the tool of mob and drug cases — had been used at scale in an insider-trading prosecution. Over the next two years, prosecutors played the jury recordings of Rajaratnam being fed earnings, deal news, and board decisions before they were public. In 2011 he was convicted on fourteen counts and sentenced to eleven years in prison, then the longest insider-trading sentence in US history. The government calculated his illegal gains at roughly \$60 million; the penalties, disgorgement, and forfeiture stacked far higher.

Here is the part that confuses almost everyone. In the same years Galleon was running its wiretapped tips, thousands of analysts and portfolio managers were *also* trading on information the rest of the market did not have — channel checks, satellite imagery of parking lots, scrubbed calls with industry experts, painstaking reads of obscure filings. None of them went to prison. Some of them ran the most respected research shops on Wall Street. They had an information edge too. The difference between them and Rajaratnam was not the *existence* of an edge, nor even its *size*. It was a legal question with a precise answer: did a **breach of a duty** stand behind the information?

That line — between an information edge the law rewards and one it punishes with prison — is the single most misunderstood concept in markets. People who have never read a securities statute will tell you confidently that "trading on inside information is illegal." That sentence is wrong in a way that matters, both as a compliance fact and as a lens on how information legally diffuses into price. This post draws the line exactly where the law draws it.

![Decision tree showing three questions that decide whether an information edge is legal or a crime](/imgs/blogs/insider-trading-reg-fd-and-what-is-actually-illegal-1.png)

## Foundations: the building blocks of insider-trading law

Let me build every term from zero, because the popular version of this law is a folk myth and the real version is a chain of specific legal tests.

### What "insider trading" actually means

Start with the everyday intuition, then correct it. Most people picture insider trading as "someone who knows a secret about a company buys the stock before everyone else and makes money." That picture is half right and dangerously incomplete. The law does not punish *knowing a secret*. It punishes *cheating someone you owed honesty to*. Insider trading, in US law, is a species of **fraud** — specifically, securities fraud under Section 10(b) of the Securities Exchange Act of 1934 and the SEC's implementing rule, **Rule 10b-5**.

Rule 10b-5 is breathtakingly broad on its face. It makes it unlawful "to employ any device, scheme, or artifice to defraud" or "to engage in any act, practice, or course of business which operates as a fraud or deceit upon any person" in connection with buying or selling a security. Notice what it does *not* say: it never uses the words "insider trading." There is no statute in the United States that says "thou shalt not trade on inside information." The entire offense was built by courts, case by case, out of that one anti-fraud rule. That is why the doctrine is so subtle — and why so many confident statements about it are wrong.

### Material non-public information (MNPI)

The first building block is **MNPI** — material non-public information. Both adjectives are load-bearing.

**Material** means there is a substantial likelihood that a reasonable investor would consider the information important in deciding whether to buy or sell, *and* that it would significantly alter the "total mix" of information available. The Supreme Court set this standard in *TSC Industries v. Northway* (1976) and applied it to forward-looking events in *Basic v. Levinson* (1988). A pending merger, an earnings number that will miss badly, an FDA approval or rejection, the loss of a top customer — these are material. The color of the CEO's tie is not. The boundary is genuinely fuzzy in the middle: a single executive departure may or may not be material depending on the firm.

**Non-public** means the information has not been disseminated to the market in a way that gives investors a fair chance to act on it. A fact buried on page 80 of a 10-K filed at midnight is *public* — anyone could read it. A fact whispered to you by the CFO is *non-public* even if the CFO is loud about it. The test is dissemination, not obscurity.

Only when information is *both* material *and* non-public does the rest of the machinery even engage. If the information is immaterial, or already public, you are free to trade — that is just research.

The materiality boundary deserves a second look because it is where real cases are won and lost. *Basic v. Levinson* gave us the test for *contingent* events — like a merger that might or might not happen — through a **probability-times-magnitude** formula: an event that is unlikely but enormous (a transformative acquisition) can be material even at low probability, while an event that is near-certain but trivial is not. A reasonable investor weighs *how likely* against *how big*. This is why "preliminary merger talks" can be material long before any deal is signed: the magnitude is so large that even a modest probability clears the bar. It is also why the SEC published Staff Accounting Bulletin No. 99 to warn companies *against* a purely numerical materiality threshold (the old "5% of earnings" rule of thumb) — a small number can be material if it tips a company from a profit to a loss, lets it just barely "make the number," or masks an illegal act. For a practitioner, the operational lesson is that materiality is a *judgment about investor behavior*, not a line on a spreadsheet, and the safest assumption when sitting on borderline information is to treat it as material.

### The two theories: classical and misappropriation

The second building block is the question the entire offense turns on: **whose duty did you breach?** US law recognizes two answers, and you need a "yes" to one of them for the trade to be a crime.

The **classical theory** covers the obvious case: a corporate *insider* — an officer, director, or employee — trades in their *own company's* stock using MNPI. They owe a fiduciary duty directly to the shareholders they are trading against. Trading on the secret while staying silent is a deception of those shareholders. The Supreme Court grounded this in *Chiarella v. United States* (1980), where it reversed a conviction precisely because the defendant — a print-shop worker who figured out merger targets from documents — owed *no duty* to the companies whose stock he traded. No duty, no fraud, no crime. *Chiarella* is the case that proves the point of this whole post: an information edge alone is not illegal.

The **misappropriation theory** plugs the gap *Chiarella* left. It covers an *outsider* who steals confidential information from the source it was entrusted to, in breach of a duty owed to *that source*. The classic case is *United States v. O'Hagan* (1997): a lawyer at a firm advising an acquirer traded in the *target's* stock. He owed no duty to the target's shareholders (the classical theory failed), but he *did* owe a duty to his law firm and its client, and he defrauded them by secretly using their information. The Supreme Court upheld the conviction. Misappropriation is why a printer, a lawyer, a banker, a government employee, or even a spouse who steals a secret can be liable even though they are total strangers to the company whose stock moved.

So the unifying principle is simple to state and easy to forget: **insider trading is trading on MNPI in breach of a duty.** Strip out the duty and you have legal research. Strip out the materiality or the non-publicness and you have legal research. All three must be present.

There is a useful way to remember how the two theories combine, called the **"disclose or abstain" rule**. If you possess MNPI *and* you owe a relevant duty, the law gives you exactly two lawful choices: publicly *disclose* the information before trading (which insiders obviously cannot do — it is the company's secret), or *abstain* from trading entirely until the information is public. There is no third option. The crime is choosing a secret *third* path — trading while staying silent — because that silence, in the presence of a duty, is itself the deception that Rule 10b-5 forbids. This is why insider trading is sometimes called a "fraud by silence": there is no lie spoken, only a duty to speak that was breached by trading instead. The breadth of "who owes a duty" is wider than most people guess. It reaches the obvious insiders (officers, directors, employees), the **temporary** or **constructive** insiders the company brings inside its circle of trust (its lawyers, bankers, accountants, consultants, even a major investor given confidential access in a deal), and — under misappropriation — anyone who steals the information from a source they owed confidentiality, including family members. The US Supreme Court and the SEC's Rule 10b5-2 make clear that a duty can arise from a *family or personal relationship of trust*, not just an employment contract — which is how a brother-in-law who trades on what he overheard at Thanksgiving ends up liable.

### Tipper and tippee liability: the personal-benefit test

The third building block is the hardest, and it is where most of the modern case law lives. What happens when the person who trades is not the insider, but someone the insider *told* — a **tippee**?

The rule comes from *Dirks v. SEC* (1983). A tippee inherits the insider's duty — and therefore can be liable — only if **two conditions** hold. First, the **tipper** (the insider) breached their duty by disclosing the information *for a personal benefit*. Second, the **tippee knew or should have known** of that breach. If the insider didn't get a benefit, there was no breach to inherit, and the tippee is clean even if they traded on obvious MNPI.

"Personal benefit" is broad. It can be cash. It can be a kickback. The Supreme Court held in *Salman v. United States* (2016) that it can be the simple act of gifting valuable information to a trading *relative or friend* — the benefit is the gift itself, no money required. But for a few years, the benefit test was the eye of a legal storm, because of a case called *United States v. Newman* (Second Circuit, 2014). There, the court overturned the convictions of two hedge-fund analysts who were *remote* tippees — third- and fourth-hand recipients down a long chain. The court held the government had to prove the tippees knew of a personal benefit, and that the benefit had to be something "of a pecuniary or similarly valuable nature," not mere friendship. *Newman* briefly made it far harder to prosecute remote tippees and led the government to drop charges against several people. *Salman* then partly reversed *Newman*'s narrowing two years later, and the 2018 *Martoma* re-decision further loosened it. The takeaway for a practitioner: the personal-benefit test is real, it is the crux of tippee cases, and its exact contours have shifted — which is why surveillance-flagged trades down a tipping chain are so heavily litigated.

![Tipper to tippee chain showing liability attaches only with a personal benefit and knowledge of the breach](/imgs/blogs/insider-trading-reg-fd-and-what-is-actually-illegal-3.png)

### Regulation FD: no more selective disclosure

The fourth building block changed the *plumbing* of information itself. Before the year 2000, a company could legally tell its favored sell-side analysts about a coming earnings miss *before* telling the public. The analysts (and their best clients) would trade on the head start. This was not classical insider trading — the company chose to disclose, so there was arguably no "deception" — but it was deeply unfair, and it concentrated the information edge in a small club.

**Regulation Fair Disclosure (Reg FD)**, adopted by the SEC in August 2000 and effective that October, banned it. The rule is simple: when a public company (or someone acting on its behalf) discloses material non-public information to securities professionals or shareholders likely to trade on it, it must disclose that same information to the *public* — simultaneously if the disclosure was intentional, or promptly (within 24 hours) if it slipped out by accident. The mechanism is usually an 8-K filing, a press release, or a webcast open to anyone.

Reg FD did not change what counts as insider trading. It attacked a *different* problem — the legal-but-unfair head start — and in doing so it reshaped how news reaches price. We will see the market consequences below; they are large.

![Information flow before and after Regulation FD comparing selective whisper to a simultaneous public broadcast](/imgs/blogs/insider-trading-reg-fd-and-what-is-actually-illegal-2.png)

### The mosaic theory: the legal counterweight

The fifth building block is the one that keeps research legal: the **mosaic theory**. An analyst is allowed to combine many pieces of *non-material or public* information — a 10-K footnote, a supplier's shipping data, a count of cars in a parking lot, a scrubbed expert call about an industry trend — into a *material* conclusion that the market has not yet reached. The conclusion can be hugely valuable and completely non-public, and trading on it is *legal*, because no single input was material-and-non-public-and-obtained-through-a-breach. The edge was *manufactured* by analysis, not *stolen* from a duty.

The mosaic theory is not a loophole; it is the affirmative legal basis of the entire research industry. The SEC and the courts explicitly recognize it. The catch is that it is a factual line, not a magic incantation — if one of your "mosaic tiles" is in fact a tip of MNPI from a breaching insider, saying the word "mosaic" does not launder it.

### Rule 10b5-1 plans and Section 16

The last two building blocks are the tools insiders use to trade *legally* in their own stock — because executives must be able to sell shares to pay taxes and diversify without being accused of trading on every quarter's MNPI.

A **Rule 10b5-1 plan** is a pre-arranged, written trading plan adopted *while the insider does not possess MNPI*. It specifies the amounts, prices, and dates of future trades (or a formula for them). Once the plan runs on autopilot, trades executed under it have an affirmative defense to an insider-trading charge — the insider couldn't have been trading *on* information they didn't have when they set it. After years of abuse, the SEC tightened these rules in 2023, most importantly by adding a **mandatory cooling-off period** of (for officers and directors) 90 days between adopting a plan and the first trade. We will work an example of why that matters.

**Section 16** of the 1934 Act is a separate, bright-line rule for the most senior insiders (officers, directors, and 10%-plus owners). It does two things: it forces them to *report* their trades publicly (the Form 4 filings traders watch), and its "short-swing profit" rule lets the company recover *any* profit they make from buying and selling within a six-month window — automatically, with no need to prove they used MNPI. It is a prophylactic rule: it removes the temptation by removing the profit.

That is the full toolkit. With those building blocks in place, we can do what this series exists to do: trace how each one moves prices, and how a practitioner reads or trades the rule. For the foundational disclosure regime these rules sit on top of, see [securities law 101: the '33 and '34 Acts and the SEC](/blog/trading/law-and-geopolitics/securities-law-101-the-33-and-34-acts-and-the-sec); for the general mechanism of how any rule gets discounted into price, see [how law moves markets: the transmission chain](/blog/trading/law-and-geopolitics/how-law-moves-markets-the-transmission-chain).

## Why insider-trading law exists at all

It is worth pausing on *why* the law draws this line, because the rationale is also the mechanism through which the law moves prices.

The standard justification has two pillars: **fairness** and **market integrity**. The fairness argument is intuitive — a market in which insiders can systematically pick the pockets of ordinary investors is a rigged game, and people will not play a rigged game. But the integrity argument is the one that shows up in asset prices, and it runs through **the cost of capital**.

Here is the chain. Investors are not naive; if they believe a market is full of better-informed insiders trading against them, they demand to be compensated for that adverse-selection risk. They compensate themselves by paying *less* for shares — i.e., by demanding a *higher return*, which is the same thing as a higher cost of capital for the companies raising money there. A market with credible insider-trading enforcement has *narrower bid-ask spreads* (market makers fear being picked off less), *deeper liquidity*, and *lower cost of capital*. The empirical literature — most famously Bhattacharya and Daouk's 2002 study of 103 countries — found that the cost of equity dropped meaningfully in countries *after* they first *enforced* insider-trading laws (mere existence of a law on the books did nothing; enforcement was what mattered). Estimates of the effect cluster around a several-percentage-point reduction in the cost of equity, though the exact magnitude is debated.

That is the spine of this whole series in miniature: a *legal regime* (enforced insider law) changes a *market microstructure variable* (adverse selection, spreads) which changes a *price* (cost of capital, valuations). Let me make it concrete.

There is a subtler, more contested side to this story worth naming honestly, because a thoughtful investor should know the counter-argument exists. A minority of economists — going back to Henry Manne's 1966 book *Insider Trading and the Stock Market* — argue that insider trading actually makes prices *more accurate* by pushing private information into the tape faster, and that it is a cheap, harmless way to compensate entrepreneurs. The mainstream rejection of that view rests precisely on the adverse-selection mechanism above: yes, insider trading speeds *some* information into price, but it does so by *taxing* every uninformed investor, and that tax is paid back in wider spreads and a higher cost of capital for everyone. The empirical weight — cross-country cost-of-equity studies, the observed narrowing of spreads around enforcement regimes — sits firmly with the mainstream. The reason this debate matters to a practitioner is that it clarifies *what the law is really protecting*: not the feelings of investors who "missed out," but the structural willingness of capital to show up at all. A market where outsiders believe the game is rigged is a market where the marginal saver keeps their money in a bank, and the companies that needed that capital pay more for it.

#### Worked example: the cost-of-capital effect of strong vs weak insider law

Take a company raising \$1 billion of equity. In a market with credible enforcement, suppose investors price the shares to a cost of equity of 8.00%. Now compare an otherwise identical market where insider trading is rampant and unpunished; the adverse-selection premium pushes the required return up by, say, 30 basis points to 8.30%.

Cost of capital is simply the price companies pay for money. The extra 0.30% on a \$1 billion raise is an incremental annual cost of:

\$1,000,000,000 × 0.0030 = \$3,000,000 per year.

Capitalized as a perpetuity at the 8% base rate, the present-value hit to firm value from that wider risk premium is roughly:

\$3,000,000 ÷ 0.08 = \$37,500,000.

So a *few basis points* of extra adverse-selection premium — invisible on any given trade — translates into tens of millions of permanent value destruction on a single large raise, and across an entire market it scales into the trillions. The intuition: insider-trading enforcement is not a moral nicety bolted onto markets; it is a direct input into every discount rate. For the mechanics of how a discount rate sets value, this connects to [equity valuation and the discounted-cash-flow model](/blog/trading/equity-research/discounted-cash-flow-dcf-valuation-from-first-principles).

## How Reg FD reshaped the flow of information into price

Reg FD is the cleanest example in all of securities law of a rule *changing the microstructure of information itself*. Before October 2000, the path of material news to price looked like a relay race with a head start for a favored few. After Reg FD, it looked like a starting gun fired for everyone at once.

The before-state worked like this. A company about to miss earnings would "guide down" its favored sell-side analysts in quiet calls. Those analysts cut their estimates; their best institutional clients heard first and traded; the stock began drifting *before* any public announcement. By the time a retail investor read the news, the move was substantially done. The information edge was real, valuable, and concentrated — and entirely legal, because the company *chose* to disclose. This was the world of the "analyst whisper number."

Reg FD detonated that model. Now, the instant a company tells *anyone* outside the firm something material, the clock demands it tell *everyone*. The relay race became a broadcast. Three consequences followed, all of them visible in market data:

First, **single-stock reactions to news became faster and sharper**. Instead of a multi-day drift as the favored few traded ahead, the price now gaps at the moment of the public 8-K or webcast and then largely stops. The information is priced in one jump rather than leaking over days. Empirically, studies after Reg FD found *reduced* selective-disclosure drift and a *concentration* of the price reaction into the public-announcement window.

Second, **the edge migrated from access to analysis**. If everyone gets the press release at the same instant, you cannot win by being on the call list. You can only win by understanding the release faster and better, or by having built a mosaic that *anticipated* it from public and non-material parts. This is the single most important structural reason the modern buy-side invests so heavily in primary research, alternative data, and expert networks: Reg FD closed the access door and forced everyone through the analysis door.

Third — and this is the unintended consequence the SEC argued about for years — **some companies talk less**. Faced with the risk that any slip becomes a Reg FD violation, some managers reduced the granularity of their guidance. The research on whether Reg FD increased or decreased overall information quality is genuinely mixed; what is *not* mixed is that it democratized *access* to whatever the company chooses to say.

#### Worked example: a Reg FD violation and the selective-disclosure repricing

Suppose a mid-cap company's stock trades at \$50.00. On a Monday, its IR head privately tells three favored analysts that the quarter will miss; those analysts and their clients sell, and the stock drifts to \$47.00 over two days on no public news — a 6% move with no headline to explain it. On Wednesday the company files the public 8-K, and the stock settles at \$46.00.

Two things happened, and the law treats them very differently. The drift from \$50.00 to \$47.00 *before* the public filing is the *selective-disclosure* damage — favored clients captured \$3.00 per share of the \$4.00 total move while the public was in the dark. That pre-announcement drift on no public news is exactly the fingerprint of a Reg FD violation. The final \$1.00 from \$47.00 to \$46.00 is the legitimate public repricing once everyone could act.

For an investor, the practical signal is the \$3.00 of *unexplained pre-announcement drift*: a stock sliding hard with no public catalyst, then "confirming" with bad news days later, is a pattern worth flagging — both as a possible Reg FD problem and as a reason to distrust the cleanliness of that name's disclosure. The intuition: Reg FD's whole purpose is to compress that \$3.00 of head-start drift to zero by making the news public the instant it leaves the building.

## Expert networks and the legal guardrails

If Reg FD pushed the edge toward analysis, **expert networks** are the industry that grew up to supply it — and the industry where the legal line gets walked closest. An expert network is a matchmaking firm that connects investors with paid consultants: a former executive, a doctor running drug trials, a supplier's salesperson, a channel distributor. The investor pays for the consultant's *general industry knowledge and judgment* — which is legal mosaic-building — and the danger is that the consultant hands over their *current employer's specific MNPI*, which is a tip.

The line is precise. A call in which a retired semiconductor executive explains *how* foundry yields behave, what the historical lead times look like, and what publicly disclosed capacity expansions imply — that is research. A call in which a *currently employed* finance manager tells you *this quarter's* bookings before they are announced — that is a tip, and trading on it is a crime, with the expert as the tipper and you as the knowing tippee.

This is exactly the seam the SAC Capital cases ran through (see the case studies below). After those prosecutions, compliant funds built heavy guardrails around expert calls: pre-clearing the expert and the topic, banning consultants who are *current* employees of public companies in the fund's universe, recording or chaperoning calls, maintaining "restricted lists" of names the fund cannot discuss, and training analysts to terminate a call the moment a consultant starts volunteering specific non-public numbers. The guardrails are not bureaucratic theater; they are the operational difference between a legal mosaic and a chargeable conspiracy.

![Matrix comparing six research methods on whether they use MNPI, breach a duty, and the legal verdict](/imgs/blogs/insider-trading-reg-fd-and-what-is-actually-illegal-8.png)

## The 10b5-1 reform: closing the cooling-off gap

Rule 10b5-1 plans are how insiders sell legally, but for two decades they were quietly gamed. The original rule had no waiting period, so an executive could *adopt* a plan while sitting on bad news and schedule a sale to execute *the next day* — capturing the same illegal head start the plan was supposed to prevent, but with a paper defense attached. Academic studies (notably work by Larcker and colleagues at Stanford) found that trades under 10b5-1 plans systematically *beat* the market right after adoption and *avoided* losses right before bad news — a statistical fingerprint of opportunistic timing.

The SEC's 2022 amendments, effective in 2023, closed the gap. The headline fix is the **mandatory cooling-off period**: officers and directors must wait the *later of* 90 days after adopting (or modifying) a plan *or* two business days after the next earnings release before the first trade can execute. The reform also bans most *overlapping* plans, limits single-trade plans to one per year, requires a good-faith certification, and mandates public disclosure of plan adoptions and terminations. The logic is structural: if you must wait three months, the MNPI you held when you set the plan is almost certainly stale or public by the time you trade — so the plan cannot be a vehicle for the very edge it was designed to neutralize.

![Timeline of a 10b5-1 plan with the 2023 90-day cooling-off period before the first trade](/imgs/blogs/insider-trading-reg-fd-and-what-is-actually-illegal-5.png)

#### Worked example: the cooling-off period's bite

Suppose a CEO holds MNPI on a coming disappointment. Under the *old* rule, she adopts a 10b5-1 plan today to sell 100,000 shares tomorrow at the current \$60.00. Tomorrow the news has not yet broken; the plan executes at \$60.00 for proceeds of \$6,000,000. A week later the news drops and the stock falls to \$48.00. She legally sold 100,000 shares for \$1,200,000 (that is \$12.00 per share) more than she would have *after* the news — a head start wearing a legal costume.

Under the *2023 rule*, the same plan cannot trade for 90 days. By the time the first sale is allowed, the bad news is long public and the stock already sits at \$48.00. The plan executes at \$48.00 for \$4,800,000 — the *post-news* price. The cooling-off period stripped out the entire \$1,200,000 of timing advantage. The intuition: a 90-day wall between *deciding to sell* and *selling* makes it nearly impossible for today's secret to still be a secret when the trade prints.

## How enforcement actually works

People assume insider-trading cases start with a tip to the SEC. Mostly they start with a *machine*. The exchanges (FINRA's surveillance arm and the exchanges themselves) and the SEC run automated systems that scan for **anomalous trading ahead of market-moving events** — an account that bought call options three days before a merger announcement, a cluster of first-time traders in an obscure name that then doubled, trading patterns that correlate with someone's social or professional network. The SEC's Market Abuse Unit built an "Analysis and Detection Center" specifically to mine this trade-blotter data for suspicious patterns across the whole tape.

A flagged pattern is not a case; it is a lead. From there the work is old-fashioned: subpoena the brokerage records ("blue sheets") to learn exactly who traded and when, map the relationships among the traders (who called whom, who roomed together in college, who sat on which board), pull phone and email records, and — in the strongest cases — get a cooperator to flip. The Galleon case industrialized two of these: large-scale *wiretaps* and a chain of *cooperators* who recorded their friends. Once the network is mapped and the breach proven, the case splits into two parallel tracks.

The **SEC** brings a *civil* action. Its remedies are **disgorgement** (give back the ill-gotten gain), a *civil penalty* of up to **three times** the gain (the "treble" penalty, under the Insider Trading Sanctions Act of 1984), officer-and-director bars, and injunctions. The standard of proof is the civil "preponderance of the evidence." The **DOJ** brings a *criminal* action for *willful* violations: the maximum is **20 years in prison per count** plus criminal fines (up to \$5 million for an individual under the 1934 Act), proven to the higher "beyond a reasonable doubt" standard. The two run in parallel, which is why a single insider-trading scheme can yield both a multi-million-dollar civil judgment *and* a prison sentence.

![Pipeline of an insider-trading case from surveillance flag to parallel SEC civil and DOJ criminal actions](/imgs/blogs/insider-trading-reg-fd-and-what-is-actually-illegal-4.png)

#### Worked example: the expected value of trading on an illegal tip

Here is the calculation every would-be tippee should do and almost none do. Suppose a tip lets you buy ahead of a deal and book a clean \$1,000,000 gain. Now price in the *full* consequence stack if caught.

If caught, the SEC takes the gain back (disgorgement of \$1,000,000) *and* adds a penalty of up to three times the gain (\$3,000,000). That is \$4,000,000 of civil exposure on a \$1,000,000 profit. Then add the DOJ: criminal fines and, far more costly, prison — career over, future earnings gone. Even ignoring prison and legal fees, the monetary outcome if caught is roughly:

+\$1,000,000 (the gain) − \$1,000,000 (disgorgement) − \$3,000,000 (treble penalty) = **−\$3,000,000**.

Now weight by the probability of getting caught. Surveillance is good and improving; suppose, generously to the tipster, the chance of detection on a well-timed pre-deal trade is just 30%. The expected value is:

(0.70 × +\$1,000,000) + (0.30 × −\$3,000,000) = \$700,000 − \$900,000 = **−\$200,000**.

And that already-negative number *excludes* legal fees (easily seven figures), the destruction of future earnings, and prison. At any realistic detection probability above ~25%, the expected value is negative; once you add prison and career loss, it is catastrophic at *any* nonzero probability. The intuition: the treble-penalty structure is deliberately engineered so that the math of cheating never works — the law converts a \$1,000,000 temptation into a multi-million-dollar negative-EV bet.

![Expected value of an illegal tip showing a 1,000,000 gain becomes a 4,000,000 loss once caught](/imgs/blogs/insider-trading-reg-fd-and-what-is-actually-illegal-7.png)

## The expanding frontier: shadow trading and global regimes

The line described so far is the *settled* US doctrine. But the law is not frozen, and two developments are worth a practitioner's attention because they show the boundary *moving outward*.

The first is **shadow trading**. In *SEC v. Panuwat* (the case settled into law over 2021–2024), the SEC charged — and a jury agreed — that an executive at one biopharma company who learned his *own* firm was about to be acquired, and then bought options in a *different, comparable* company (reasoning that the deal would lift the whole peer group), had committed insider trading. He did not trade his own company's stock or his acquirer's; he traded an *economically linked third party*. The misappropriation theory stretched to cover it: he used his employer's confidential information, in breach of his duty to his employer, to trade *somewhere*. The implication is significant — MNPI about Company A can taint a trade in Company B if the link is close enough. For analysts who routinely trade "the read-through" (if the leader misses, short the laggard), *Panuwat* is a warning that a read-through built on *stolen* MNPI is just as illegal as trading the source name itself.

The second is the **patchwork of global regimes**. The United States built insider-trading law out of an anti-fraud rule, so its offense centers on *breach of duty*. The European Union, by contrast, built its regime — the **Market Abuse Regulation (MAR)** — on a different theory: a flat *parity-of-information* prohibition. Under MAR, anyone who *possesses* inside information about a security is barred from trading it, regardless of whether they breached any duty to get it. That is a meaningfully *broader* prohibition than the US standard — the EU does not require the personal-benefit gymnastics of *Dirks* and *Newman*. The United Kingdom, Japan, Hong Kong, and most developed markets sit somewhere on this spectrum. The practical consequence for a cross-border investor is that **the same trade can be legal in New York and illegal in Frankfurt**, and a global fund must comply with the *strictest* regime that touches the trade. This is the same enforcement-credibility story from the cost-of-capital section, playing out across borders: jurisdictions that enforce a broad, credible insider regime tend to attract more foreign portfolio investment, because outside investors fear being picked off less.

#### Worked example: the read-through trade, legal vs shadow trading

Suppose Company A and Company B are two comparable specialty lenders, and a new regulation is expected to hurt the whole sub-sector. You build a *legal* mosaic: you read both companies' public filings, you talk to their public-facing investor-relations teams, you analyze the proposed rule's text, and you conclude B is more exposed than the market realizes. You short B. The edge came entirely from public and non-material parts — legal, and a textbook read-through.

Now change one fact. You are an executive *at Company A*, and you learn — confidentially, from inside A — that A is about to report a disastrous regulatory charge that will surely drag the whole sub-sector down. You short B to profit from the coming sector-wide hit. Same short, same \$200,000 of profit if B falls 4% on a \$5,000,000 position. But this trade is *shadow trading*: you misappropriated A's confidential MNPI, in breach of your duty to A, and used it to trade B. After *Panuwat*, that is insider trading even though you never touched A's or its acquirer's stock. The intuition: the read-through itself is fine — what poisons it is *where the conviction came from*. Public analysis is legal; a stolen secret routed through a peer name is not.

## How it shows up in real markets: three case studies

### Galleon and Raj Rajaratnam: the wiretap era

The Galleon case (2009–2011) is the modern template for how a *network* of tips gets dismantled. Rajaratnam did not have a single source; he ran a web — a McKinsey partner, a Goldman Sachs board member (Rajat Gupta, later convicted for tipping Berkshire Hathaway's investment and Goldman's quarterly results), corporate insiders at tech firms — who fed him MNPI that he traded across Galleon's funds. The government proved gains it pegged near \$60 million. What made it a watershed was *method*: for the first time, prosecutors used Title III wiretaps to capture the tips in real time, and they turned cooperators who recorded their co-conspirators. The eleven-year sentence and the wiretap precedent put the entire hedge-fund industry on notice that the old "it's just color from my network" defense was over. The structural lesson for markets: enforcement technology (wiretaps, cooperators, then trade-pattern algorithms) is what makes insider law *credible* — and credibility, per the cost-of-capital argument, is what actually narrows spreads.

The market-microstructure footprint of these cases is worth tracing because it is how the law reaches into price discovery. When a fund is built on a steady diet of tips, its trades carry *information* the rest of the market lacks — which means the market makers and other participants on the other side are systematically *adversely selected*, losing a little on every fill. They protect themselves the only way they can: by quoting wider and pulling liquidity around events. The Galleon prosecution, by removing one large informed-and-cheating participant and signaling that others would follow, marginally *reduced* that adverse selection in the affected names. This is the cost-of-capital channel running in reverse and in miniature: less cheating, less adverse selection, tighter spreads, cheaper capital. It is also why enforcement actions cluster the way they do — the SEC gets far more deterrence per dollar from a handful of *high-profile* convictions (which reset everyone's perceived probability of getting caught) than from a large number of small ones.

### SAC Capital and Mathew Martoma: the expert-network seam

If Galleon was about a network of tips, the SAC Capital cases were about the *expert-network* seam specifically. Mathew Martoma, a portfolio manager at SAC's CR Intrinsic unit, cultivated a doctor — Sidney Gilman — who chaired the safety-monitoring committee for a clinical trial of an Alzheimer's drug being developed by Elan and Wyeth. Through an expert network, Gilman gave Martoma the *negative* trial results before they were public. SAC reversed a large long position into a short, and the government calculated that the firm *avoided losses and made gains* of roughly \$275 million on the switch — at the time the most profitable single insider-trading scheme ever charged. Martoma was convicted in 2014 and sentenced to nine years; SAC Capital itself pleaded guilty in 2013, paid a \$1.8 billion penalty, and was barred from managing outside money, effectively ending it as a hedge fund. The case is *the* cautionary tale for expert-network compliance: the consultant was a *current* insider to the specific trial, the information was specific and material and non-public, and a personal benefit flowed — every element of an illegal tip, dressed up as a legitimate expert call. For how funds manage this kind of catastrophic compliance and key-person risk, see [operational risk and compliance at a hedge fund](/blog/trading/hedge-funds/operational-due-diligence-and-fund-compliance).

### The Newman reversal: when the line moved

The third case study is different — it is a case the *government lost*, and the loss reshaped the market for a few years. In *United States v. Newman* (2014), the Second Circuit overturned the convictions of two hedge-fund analysts who were *remote* tippees — they received Dell and NVIDIA earnings information passed down a chain of intermediaries. The court held the government had failed to prove the analysts knew the original insiders received a *personal benefit*, and tightened what counts as a benefit (not mere friendship). The ruling forced the Manhattan US Attorney to drop charges against roughly a dozen people and vacate several guilty pleas. For about two years, *Newman* materially raised the bar for prosecuting remote tippees — visible in a dip in new insider cases. Then *Salman* (2016) and the re-decided *Martoma* (2018) walked the benefit test back toward the government's position. The market lesson is subtle but important: **the insider-trading line is not a fixed wall; it moves with the case law**, and a single appellate ruling can re-price the legal risk on an entire category of trading strategies overnight.

## Common misconceptions

The folk version of this law is wrong in specific, costly ways. Here are the three that matter most, each corrected with a number or a case.

### Misconception 1: "All trading on private information is illegal"

This is the big one, and it is simply false. The crime requires *MNPI obtained or passed through a breach of a duty*. Strip out the breach and it is legal — which is precisely why *Chiarella v. United States* (1980) was *reversed*: the print-shop worker traded on genuinely private merger information he deduced from documents, made money, and the Supreme Court still threw out his conviction because he owed *no duty* to anyone. Quantitatively: in the worked example above, the *exact same* \$1,000,000 of profit is a felony if it came from a breaching tip and perfectly legal if it came from a mosaic of public and non-material parts. The dollar figure is identical; the *provenance of the information* is the entire offense. People who believe "any edge is illegal" either trade timidly and leave legal alpha on the table, or — worse — assume the line is hopeless and stop watching it at all.

### Misconception 2: "10b5-1 plans are bulletproof"

A 10b5-1 plan is an *affirmative defense*, not an automatic shield, and the 2023 reforms made that explicit. Before the reform, plans adopted in possession of MNPI and executed within days were a documented abuse — recall the worked example where the old rule let an executive capture \$1,200,000 of timing advantage with a plan adopted the day before selling. After 2023, the 90-day cooling-off period and the overlapping-plan ban strip that out, and the SEC has prosecuted plans that were obvious bad-faith timing devices (a plan adopted while holding MNPI is not protected at all, cooling-off or not). The number to remember: the defense survives only if the plan was adopted in *good faith* and *without* MNPI — and the burden of showing that is on the insider.

### Misconception 3: "The mosaic theory is a loophole"

The mosaic theory is not a loophole; it is the *affirmative legal basis* of the entire research industry, explicitly blessed by the SEC and the courts. The misconception cuts both ways and both ways are dangerous. People who think it is a "loophole" assume it is a flimsy excuse that won't hold up — and so they avoid perfectly legal primary research. Other people think it is a magic word that *launders* any input — and so they fold a genuine MNPI tip into a "mosaic" and assume the label protects them. It does not: if even *one* tile in your mosaic is material non-public information from a breaching insider, the whole trade is tainted regardless of how much legitimate public research surrounds it. The SAC/Martoma case is the proof — Martoma had plenty of legitimate research; the *one* tile that was a tip of specific trial data is what put him in prison for nine years. The mosaic theory protects information you *built*, never information you *stole*.

## How to trade it: building a legal information edge

Everything above lands here. The practitioner's job is to build the *largest possible legal edge* while staying provably on the right side of the line — and to read the rule itself as a signal about how information will diffuse into price.

### The three pillars of a legal edge

**Primary research and the mosaic.** The whole point of Reg FD is that it pushed the edge from access to analysis — so build the analysis. Read the filings nobody reads (the footnotes, the 8-K exhibits, the proxy). Combine public and non-material data into a material *conclusion*: shipping manifests, satellite parking-lot counts, app-download data, hiring trends, supplier checks. Each tile must be either public or non-material; the *combination* can be hugely valuable and entirely yours. This is legal because you *manufactured* the edge from parts the law does not protect.

**Channel checks, done right.** Call the suppliers, distributors, and customers around a company to gauge demand — but stay in the lane of *the channel's own* business and *industry-level* color, never the target company's specific non-public numbers. "Are you ordering more chips this quarter?" to a distributor is research; "what is your customer's unannounced bookings number?" is a tip.

**Expert calls, with guardrails.** Use expert networks for judgment and history, never for current MNPI. Pre-clear the expert and topic, exclude current employees of names in your universe, chaperone or record the calls, and train yourself to hang up the instant a consultant volunteers specific non-public figures. The guardrails *are* the edge — they are what let you use the channel at all.

### The compliance bright lines

Three bright lines keep you safe. First, **if information is material, non-public, *and* came to you through someone's breach of duty, you cannot trade — full stop**, no matter how it reached you. Second, **when in doubt, the duty question is the one to ask**: "did anyone breach a duty of trust to get me this?" If yes, stop. Third, **document your mosaic**: keep the trail of public and non-material sources that built your thesis, so that if surveillance flags your well-timed trade, you can *prove* the edge was manufactured, not stolen. The single most protective habit in this entire field is a research file that shows your work.

Two operational practices turn those bright lines into daily habits. The first is **wall-crossing** — the formal process by which a fund's analyst is *deliberately* brought "over the wall" and given a company's MNPI (for example, to evaluate a private placement), with the explicit consequence that the analyst and often the whole firm are then **restricted** from trading that name until the information is public. The discipline is that crossing the wall is a one-way door: you trade access for a trading freeze, knowingly and on the record. The second is the **discipline of accidental receipt**. If MNPI lands in your inbox by mistake — a misdirected email, a slip on a call, a document left in a printer (the very fact pattern of *Chiarella*) — the safe move is not to quietly use it; it is to stop, not trade the name, and document that you received it inadvertently. You are generally clean if you owed no duty and obtained it through no breach of *anyone's* duty, but the moment you *know* it leaked through someone else's breach, the misappropriation theory can attach. The cheap, protective reflex is the same one a good poker player has about a flashed card: pretend you never saw it, and do not act on it.

### What a fund's surveillance flags

It helps to know what the *other* side is watching, because compliant funds run the same surveillance the SEC does. The flags: trading in a name *right before* a catalyst with no documented research trail; options activity clustered ahead of an announcement (cheap, leveraged, and a classic insider tell); trading in a name on a *restricted list* or right after an expert call about that exact name; and abnormal, well-timed profits that correlate with someone's network. A steep, perfectly-timed run-up into news — exactly the GameStop-style spike below — is the visual signature surveillance algorithms are built to catch. The point of knowing the flags is not to evade them; it is to *avoid tripping them legitimately*, by keeping a research trail and steering clear of the seams.

![GameStop daily close in January 2021 rising from 17 dollars to a 347 dollar peak](/imgs/blogs/insider-trading-reg-fd-and-what-is-actually-illegal-6.png)

The chart above is not an insider-trading case — the GameStop squeeze was a public, crowd-driven event. It is here to show the *shape* surveillance hunts for: a steep, well-timed run-up that begs the question "who knew, and when?" When that shape appears *without* a public catalyst, it is exactly what flags a trade for review.

#### Worked example: sizing a legal mosaic edge as an information ratio

Finally, the payoff calculation: how do you size a *legal* edge once you have it? Suppose your mosaic research gives you a genuine, repeatable predictive edge that produces an expected excess return (alpha) of 4.0% per year on a basket of positions, with a tracking error (volatility of that excess return) of 8.0%. The **information ratio** — the cleanest measure of skill per unit of risk — is:

IR = alpha ÷ tracking error = 4.0% ÷ 8.0% = **0.50**.

An information ratio of 0.50 is a genuinely good, durable, *legal* edge — many respected funds run on less. Contrast that with the illegal tip from the earlier example: a one-off \$1,000,000 with a *negative* expected value once you weight in detection and penalties. A repeatable IR of 0.50, compounded across hundreds of positions and many years, dwarfs a single felonious score that on a risk-and-penalty-adjusted basis is worth *less than zero*. For the full machinery of sizing positions from an information ratio and combining many such signals, see [the information ratio and combining alpha signals](/blog/trading/hedge-funds/the-information-ratio-and-portfolio-construction). The intuition: the legal edge is *smaller per trade* but *positive, repeatable, and uncapped* — which is exactly why the best investors never need to cross the line.

### The catalysts and what invalidates the view

Reading insider-trading law as a *market signal* means watching three things. First, **the case law**: an appellate ruling that moves the personal-benefit test (like *Newman*, then *Salman*) re-prices the legal risk on whole strategies — when the line tightens, expert-network and remote-tip-adjacent strategies get safer; when it loosens, they get riskier. Second, **the enforcement posture**: an SEC that staffs up its Market Abuse Unit and brings high-profile cases makes the credibility argument stronger, which (per the cost-of-capital channel) is a marginal tailwind for market valuations and a marginal headwind for anyone running close to the line. Third, **the disclosure plumbing**: a Reg FD enforcement action against a company is a flashing sign to *distrust that name's disclosure discipline* and to watch its stock for unexplained pre-announcement drift.

What **invalidates** the playbook is simple and worth stating plainly: the discovery that any tile in your "legal mosaic" is in fact MNPI from a breaching source. The moment that is true, the edge is not small-but-legal; it is a felony, and the entire expected-value calculation flips from a positive, compounding 0.50 information ratio to a multi-million-dollar negative-EV bet with prison attached. The whole discipline of this field is keeping that one fact — *no breached duty supplied my edge* — provably true. Said another way, the line that decides everything is not whether you knew something the market did not; it is *how you came to know it*. Build the edge from parts the law leaves open — filings, channels, scrubbed expert judgment, the patient assembly of a mosaic — and you can know a great deal that the price has not yet caught up to, perfectly legally. Take a single shortcut through a breached duty and the same knowledge becomes a felony. The best investors are not the ones who get the closest to the line; they are the ones who build an edge large enough that they never need to. For the broader toolkit of reading the regulatory and legal calendar as a source of catalysts, see [the regulatory calendar: trading the rulemaking clock](/blog/trading/law-and-geopolitics/the-regulatory-calendar-trading-the-rulemaking-clock), and for the master model of how any rule diffuses into price, [how a rule becomes a price: expectations, drift, and repricing](/blog/trading/law-and-geopolitics/how-a-rule-becomes-a-price-expectations-drift-and-repricing).

## Further reading & cross-links

**Within this series:**
- [Securities law 101: the '33 and '34 Acts and the SEC](/blog/trading/law-and-geopolitics/securities-law-101-the-33-and-34-acts-and-the-sec) — the disclosure regime and the definition of materiality these rules sit on top of.
- [How law moves markets: the transmission chain](/blog/trading/law-and-geopolitics/how-law-moves-markets-the-transmission-chain) — the master model: law → policy → macro → price → the trade.
- [How a rule becomes a price: expectations, drift, and repricing](/blog/trading/law-and-geopolitics/how-a-rule-becomes-a-price-expectations-drift-and-repricing) — the event-study toolkit applied to legal events.
- [The regulatory calendar: trading the rulemaking clock](/blog/trading/law-and-geopolitics/the-regulatory-calendar-trading-the-rulemaking-clock) — turning enforcement and rulemaking dates into a catalyst calendar.

**Cross-asset and practitioner context:**
- [Operational due diligence and fund compliance](/blog/trading/hedge-funds/operational-due-diligence-and-fund-compliance) — how funds build the expert-call and MNPI guardrails in practice.
- [The information ratio and portfolio construction](/blog/trading/hedge-funds/the-information-ratio-and-portfolio-construction) — sizing a legal edge from its skill-per-risk.
- [Discounted-cash-flow valuation from first principles](/blog/trading/equity-research/discounted-cash-flow-dcf-valuation-from-first-principles) — how the cost-of-capital channel turns insider-law credibility into firm value.
