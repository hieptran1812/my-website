---
title: "Disclosure: The Prospectus, the Filings, and Insider Trading"
date: "2026-06-21"
publishDate: "2026-06-21"
description: "How a public company is forced to tell the truth on a schedule, and why trading on its secrets is a crime that exists to protect that schedule."
tags: ["capital-markets", "disclosure", "prospectus", "insider-trading", "regulation-fd", "edgar", "materiality", "rule-10b-5", "securities-law", "market-integrity"]
category: "trading"
subcategory: "Capital Markets"
author: "Hiep Tran"
featured: true
readTime: 40
---

> [!important]
> **TL;DR** — A public market only works if a stranger will buy the share you want to sell at a price you both trust; mandatory disclosure plus the ban on trading around it is the machine that manufactures that trust.
>
> - A company "speaks now" in the S-1 prospectus when it goes public, then never stops: 10-K once a year, 10-Q each quarter, 8-K within four business days of a surprise, a proxy before each shareholder vote — all on EDGAR.
> - Information is **material** if a reasonable investor would consider it important; that single fuzzy word does enormous legal work, and most enforcement is a fight over it.
> - **Regulation FD** bans selective leaking to favored analysts; **insider trading** law (Rule 10b-5) bans trading on material non-public information when doing so breaches a duty.
> - The one idea to remember: insider-trading enforcement is not about fairness for its own sake — it exists to protect the disclosure compact, because if insiders can trade on secrets, outsiders stop trusting the price and the market goes thin.

On the morning a company rings the opening bell, something quietly remarkable has already happened weeks before. A few hundred pages have been filed with the U.S. Securities and Exchange Commission describing, in exhausting detail, everything that might go wrong with the business: that it has never made a profit, that one customer is 40% of revenue, that the founder controls the votes, that a patent dispute could vaporize the product line. Nobody is forced to invest. But everybody who does has, in principle, been told.

That document — the registration statement, the prospectus, the S-1 — is the opening move in a game that never ends. Going public is not a one-time confession; it is the start of a lifelong obligation to keep talking. And wrapped around all that talking is a body of law with a fearsome reputation: insider trading. The two things look unrelated — boring paperwork on one side, perp walks and wiretaps on the other. They are the same thing. Disclosure forces the truth out into the open on a schedule; insider-trading law makes sure nobody profits by jumping the schedule. Together they are the trust compact that makes a liquid market possible.

This post is about that compact. We will walk the disclosure calendar from the prospectus through the continuous filings, pin down the slippery word *material*, see why Regulation FD exists, and then understand insider trading not as a morality play but as the enforcement arm that keeps the whole disclosure regime honest.

![Disclosure lifecycle from S-1 to the next 8-K filing](/imgs/blogs/disclosure-the-prospectus-filings-and-insider-trading-1.png)

## Foundations: what disclosure is and why a market needs it

Start with a problem you already understand. You want to sell a used car. You know it has 90,000 miles, a small oil leak, and a brand-new transmission. The buyer knows none of this. They have to guess. And because they have to guess, they will assume the worst and offer you a low price — the price of an *average* car of that age, discounted for the risk that yours is a lemon. Your honest, well-maintained car gets punished for the sins of every bad car the buyer has ever heard about. Economists call this the **lemons problem**, and its end state is grim: good sellers withdraw, only bad cars trade, and eventually the market thins out or collapses.

A stock is a far worse lemons problem than a used car. You cannot kick its tires. Its value depends on facts — revenue, debt, lawsuits, a drug trial's result — that live inside the company and that the insiders know and you do not. Left alone, that information gap would make outsiders demand a steep discount or simply refuse to play, and companies would find it ruinously expensive to raise money from the public.

**Disclosure** is the regulatory answer: force the company to publish the facts, on a schedule, in a standard format, so the buyer no longer has to guess and the honest company no longer gets punished for the dishonest one. A **security** here just means a tradable financial claim — a share of stock, a bond — and a **public company** is one whose securities trade on a market open to ordinary investors. The deal society offers such a company is blunt: you may raise money from the public and enjoy a liquid market for your shares, but in exchange you give up financial privacy forever. That is the bargain at the heart of the [securities acts](/blog/trading/capital-markets/why-markets-are-regulated-disclosure-and-the-securities-acts).

Notice the spine of the whole capital-markets machine running underneath this. A capital market turns savings into long-term investment, and it does so because the secondary market is liquid — because you can sell tomorrow the claim you bought today. But liquidity depends on trust: a stranger will only stand ready to buy from you if they believe the price reflects the real facts and that you do not know something they don't. Disclosure builds that belief. It is the reason [a price can be made](/blog/trading/capital-markets/how-a-price-is-made-discovery-arbitrage-and-efficiency) that both sides will accept.

![What disclosure has to protect, the stock of public capital by market](/imgs/blogs/disclosure-the-prospectus-filings-and-insider-trading-4.png)

The stakes are not theoretical. The global equity market is worth on the order of \$115 trillion and the bond market more than \$140 trillion. Every dollar of that depends on holders believing the disclosed facts are roughly the real facts. That belief is the asset the disclosure regime protects.

## The prospectus: the "speak now" moment of going public

When a private company decides to sell shares to the public for the first time, it files a **registration statement** with the SEC. For most U.S. companies the relevant form is the **S-1**, and the most important part of it — the part actually handed to investors — is the **prospectus**. The mechanics of pricing and allocating that deal belong to [the IPO process end to end](/blog/trading/capital-markets/the-ipo-process-end-to-end-from-mandate-to-first-trade); here we care about the *informational* content, because the prospectus is the single richest act of disclosure a company ever performs.

What must it contain? Put simply, everything a reasonable buyer would want to know before handing over money:

- **The business**: what the company actually does, its markets, its competitors, its customers.
- **Risk factors**: a long, deliberately uncomfortable list of everything that could go wrong — competition, key-person dependence, regulation, litigation, the fact that it may never be profitable. Companies write these defensively, because a risk disclosed cannot later be claimed to have been hidden.
- **MD&A** — Management's Discussion and Analysis: the narrative where management explains the numbers in their own words. Not just *what* revenue was, but *why* it moved, and what trends and uncertainties management sees ahead. This is where a careful reader learns the most.
- **Audited financial statements**: income statement, balance sheet, cash flows, signed off by an independent auditor.
- **Use of proceeds, ownership, dilution, executive pay**: who owns what, what the money is for, and how much the public buyer's stake gets diluted.

The phrase that captures the prospectus is "speak now." This is the company's chance — and obligation — to put every material fact on the record before the first share is sold. The legal teeth come from the Securities Act of 1933: if the registration statement contains a material misstatement or omission, the company *and* its directors, *and* the underwriting banks, can be liable to buyers. That is why underwriters run **due diligence**: they are personally exposed if the document lies.

It is worth dwelling on *who* is on the hook, because that is what makes the prospectus believable rather than just a brochure. Under Section 11 of the 1933 Act, liability for a defective registration statement reaches the issuer, every director who signed it, the principal executive and financial officers, the underwriters, and even the accountants who certified the financials. A buyer who lost money does not have to prove the company *intended* to deceive — for most of these defendants the standard is closer to strict liability, with a "due diligence" defense available only to those who can show they reasonably investigated and believed the statements were true. That is a remarkable allocation of risk: it means the bank that brought the company to market has its own money and reputation riding on the accuracy of the document, so the bank polices the issuer on the investor's behalf. The prospectus is trustworthy not because the company is honest, but because a chain of well-resourced parties each face ruin if it isn't.

There is also a quiet-period discipline around the prospectus that beginners find counterintuitive. In the weeks before and during the offering, the company is sharply limited in what it can say *outside* the document — no hyping the stock in interviews, no selectively briefing favored buyers. Everything material must flow through the registered prospectus, where it is on the record and subject to liability. The famous near-derailment of Google's 2004 IPO, when its founders gave a *Playboy* interview during the quiet period, is the canonical cautionary tale: the SEC required the interview be appended to the prospectus rather than letting it sit as unregulated marketing. The principle is the same one that runs through the whole regime — material information about a public security belongs in the official channel, available to all, not whispered through side doors.

#### Worked example: how a borderline fact gets into the prospectus

A biotech is filing its S-1. Its lead drug is in a Phase 3 trial. Two weeks before filing, an internal analysis suggests the trial may be trending toward a weaker result than hoped, though it is not conclusive. Does this go in the prospectus?

Run the materiality test (defined fully below): would a reasonable investor buying into a one-drug biotech consider a hint of trial weakness important? Almost certainly yes — the entire \$800M valuation rests on that drug. So counsel will insist on a risk factor: "Our lead candidate is in Phase 3; interim data may not predict the final result, and a failure would materially harm our business." They will *not* disclose the specific internal number if it is preliminary and could mislead, but they cannot stay silent on the risk. The cost of over-disclosing is a slightly scarier-looking prospectus; the cost of under-disclosing is a securities-fraud lawsuit the day the trial fails. Honest companies disclose the risk and let the price adjust — which is the whole point.

The IPO market itself swings violently, and disclosure quality matters most exactly when the market is hottest and buyers are least careful.

![US IPO proceeds by year from 2014 to 2024](/imgs/blogs/disclosure-the-prospectus-filings-and-insider-trading-2.png)

In 2021, U.S. companies raised roughly \$142 billion in traditional IPOs across nearly 400 deals; in 2022 that collapsed to about \$8 billion across 71 deals as rates rose and risk appetite vanished. In a boom year the prospectus is read carelessly and over-optimistically; that is precisely when the mandatory risk-factor section is doing its quiet, unglamorous job of forcing the downside onto the page.

## Continuous disclosure: the filings that never stop

Here is the idea most people miss. The prospectus is the *beginning*, not the end. If disclosure happened only at the IPO, the information would be stale within a quarter and the lemons problem would return. So the Securities Exchange Act of 1934 imposes **continuous disclosure**: a public company must keep filing, forever, on a fixed cadence.

![The continuous disclosure filings every public company owes](/imgs/blogs/disclosure-the-prospectus-filings-and-insider-trading-3.png)

The four core filings, each answering a different question on a different clock:

- **10-K** — the annual report. Audited full-year financials, a refreshed business description, updated risk factors, and MD&A. Due roughly 60–90 days after the fiscal year-end depending on company size. This is the company's once-a-year complete statement of itself.
- **10-Q** — the quarterly report. Unaudited financials for the quarter, due about 40–45 days after quarter-end. Three of these plus the 10-K cover the year.
- **8-K** — the "current report." Filed within **four business days** of a material event. This is the one with the short fuse, because it handles surprises that cannot wait for the next quarter.
- **DEF 14A** — the proxy statement. Sent before the annual shareholder meeting, it discloses executive pay, board nominees, and the items shareholders will vote on.

All of these land in **EDGAR** — the SEC's Electronic Data Gathering, Analysis, and Retrieval system — a free public database where anyone can read any filing minutes after it posts. EDGAR is the great equalizer: the same 10-K a billion-dollar fund reads is available to a retail investor for nothing. That equality of *access to filed information* is itself a deliberate piece of market design. Before EDGAR went fully electronic in the 1990s, getting a company's filings meant requesting paper from the SEC's reading room or paying a service — a friction that quietly favored the well-resourced. Putting everything online for free was a profound act of leveling: it did not give everyone the same *analysis*, but it gave everyone the same *raw material*, which is exactly what the disclosure compact promises.

A subtlety worth internalizing is the difference between the *form* and the *content* of a filing. The 10-K is not just a data dump; it is a structured argument the company makes about itself, and the most valuable parts are often the least quantitative. The risk-factor section, refreshed every year, is where a careful reader watches for new threats creeping in — a new competitor named for the first time, a regulatory inquiry that appeared this year and not last. The MD&A is where management is forced to explain the *why* behind the numbers, and where evasive or deteriorating language is itself a signal. Footnotes to the financial statements hide the real story of off-balance-sheet obligations, related-party deals, and accounting choices. Disclosure is not satisfied by filing on time; it is satisfied by filing *truthfully and completely*, and the gap between a technically-compliant filing and an actually-informative one is where a lot of securities litigation lives.

The cadence also creates a rhythm the whole market organizes itself around. Earnings season — the cluster of weeks each quarter when most companies file their 10-Qs and report — is the heartbeat of the equity market, the moment when the largest single batch of new material information hits all at once. Analysts pre-position their models, options markets price in the expected move, and the stock gaps to a new level as the filing reconciles expectations with reality. None of that machinery would exist without the mandatory quarterly clock; it is the regulation that creates the rhythm, and the rhythm that disciplines the price.

The 8-K deserves special attention because it is where continuous disclosure meets real time. Events that trigger an 8-K include a change of CEO or CFO, a major acquisition, bankruptcy, the departure of the auditor, a material agreement, or a material impairment. The four-business-day clock means the market should never go long without learning a company-shaking fact.

#### Worked example: an 8-K-worthy event and its timing

On a Tuesday afternoon, the CEO of a public company resigns abruptly amid a disagreement with the board. Is this material? A reasonable investor in a company whose strategy is identified with its CEO would clearly care — leadership change can move a stock 5–15% in a day. So it is an 8-K item (departure of a principal officer).

The clock starts. The company has four business days — through the following Monday — to file an 8-K describing the resignation. In practice, because the company also cannot let *some* people know before others (see Reg FD below), it will usually issue a press release and file the 8-K close to simultaneously, often before the next market open, to control the timing and prevent leaks. Suppose the stock would fall 10%. A \$10 billion company would lose \$1 billion in market value on the news. The disclosure rule does not stop the fall — it ensures the fall happens *for everyone at the same instant*, rather than for insiders on Tuesday and outsiders on Friday. That simultaneity is the entire game.

The deeper point: ongoing disclosure, not the IPO prospectus, is where the regime spends most of its effort. The IPO is one loud day; the 10-Ks, 10-Qs, and 8-Ks are the thousands of quiet days that keep the price honest for the decades a company stays public. This is also what underpins the secondary-market liquidity that, in turn, made the IPO fundable in the first place — the spine again: nobody funds a long-lived public company unless they can trust and trade its shares every day afterward.

## Materiality: the one word that does all the work

Every rule above hinges on a single question: is the fact **material**? Disclose every material fact; do not trade on material non-public information; do not selectively leak material information. So what counts?

The U.S. standard comes from the Supreme Court cases *TSC Industries v. Northway* (1976) and *Basic v. Levinson* (1988). The test: information is material if there is **a substantial likelihood that a reasonable investor would consider it important** in deciding how to vote or trade — or, put differently, if it would significantly alter the "total mix" of information available. *Basic* added a crucial nuance for uncertain events: for something that might or might not happen (a merger under negotiation, a pending trial result), materiality depends on the **probability** the event occurs times the **magnitude** of its effect. A low-probability event can still be material if it is huge; a high-probability event can be immaterial if it is trivial.

That is a wonderfully reasonable standard and a nightmare to apply. There is no dollar threshold, no checklist. It is a judgment call, made in advance, under uncertainty, by people who will be second-guessed by regulators and plaintiffs' lawyers with the benefit of hindsight. Most insider-trading and disclosure litigation is, at bottom, an argument about whether a particular fact was material.

The probability-times-magnitude rule from *Basic* is the hardest part to apply and the most important to understand. Consider a merger negotiation. Early on, when two CEOs have merely had a friendly dinner, the probability of a deal is low — but the magnitude, a 40% takeover premium, is enormous. As the talks progress through letters of intent, due diligence, and board approval, the probability climbs while the magnitude stays huge, and at some point the expected effect crosses the materiality line. The company famously cannot pin the line to a single event like "signing," because *Basic* itself rejected a bright-line rule that said merger talks were immaterial until a definitive agreement. The practical consequence is that companies issue "we do not comment on rumors or speculation" precisely because saying anything specific about a maybe-deal risks either tipping the market or making a misleading statement. Materiality forces a company to manage a moving target: a fact can be immaterial on Monday and material by Friday without any single dramatic event flipping the switch.

There is one more wrinkle the law adds: even a true statement can be actionable if it is *misleading by omission*. A company that voluntarily touts strong demand while sitting on internal data showing demand collapsing has not lied about the strong part — but by omitting the collapse, it has made the whole statement misleading. This is why disclosure is not just about the mandatory filings: the moment a company chooses to speak on a topic, it takes on a duty to speak completely and not leave out the material part that changes the picture. Silence is often safer than a half-truth.

#### Worked example: judging materiality on a borderline fact

A retailer's internal sales data shows same-store sales running 0.4% below the prior quarter — a tiny miss, well within normal noise. Is it material? Probably not on its own: a reasonable investor would not change their decision over 0.4% of noise, so disclosure is not compelled and trading on it is not clearly illegal.

Now change one number. The same retailer's data shows same-store sales running **8% below** expectations — far outside normal range, in a company the market prices for steady growth. Apply *TSC*: would a reasonable investor consider an 8% sales collapse important? Unquestionably. The fact has flipped from immaterial noise to material information. An executive who quietly sold stock on that 8% figure before it was public would be trading on material non-public information; the same executive selling on the 0.4% figure would have a strong defense that the fact was immaterial. The law lives in that gap between 0.4% and 8% — and where exactly the line sits is the judgment call that keeps securities lawyers employed.

## Regulation FD: everyone gets the news at once

Before the year 2000, there was a polite, corrosive practice: companies would tell favored sell-side analysts and big institutional holders about a coming earnings miss or a slowing order book in a quiet call, days before the public found out. The analysts would adjust, the big funds would trade, and the retail investor — reading the same EDGAR filings — was a step behind by design.

**Regulation FD** (Fair Disclosure), adopted by the SEC in 2000, ended that. The rule is simple to state: when a public company discloses material non-public information to certain market professionals (analysts, big shareholders), it must disclose that same information to **everyone** at the same time — typically through a press release or an 8-K. No more selective leaking to the favored few. Reg FD is not anti-fraud law; it is a fairness-of-*access* rule. It says material information, when released, must be released to the whole market simultaneously.

#### Worked example: a Reg FD violation

An investor-relations officer takes a call from a major hedge-fund analyst the week before earnings. The analyst probes; the IR officer, trying to be helpful, says, "I wouldn't get too excited about this quarter — orders softened in the last month." That sentence is material (it foreshadows a miss) and non-public. The hedge fund sells 2 million shares the next morning. When earnings come out and the stock drops 9%, the fund has saved, say, \$5 per share on 2 million shares — \$10 million it would not have had if it learned the news with everyone else.

That is a Reg FD violation. The SEC can bring an enforcement action and a settlement against the company and the IR officer — historically in the hundreds of thousands to low millions of dollars, plus the reputational damage. Crucially, Reg FD here catches the *selective disclosure* even before you get to whether the trading was insider trading. The cure the rule wants is dull and correct: the IR officer should have said nothing non-public, and any real guidance change should have gone out to the whole market in a filing. Reg FD's whole purpose is to keep the playing field that disclosure creates from being quietly tilted in private.

## Insider trading: the crime that protects the compact

Now the dramatic part — and the part most people misunderstand. Insider trading is illegal in the United States, but **not** because trading on an information advantage is inherently banned. Markets are full of legal information advantages: a brilliant analyst who builds a better model, a fund that counts cars in a parking lot, a trader with faster data. None of that is illegal. The law targets something narrower and more specific: trading on **material non-public information in breach of a duty**.

That last clause is everything. The U.S. has no general "level playing field" statute. Instead, liability is built on the anti-fraud provision **Rule 10b-5** (under Section 10(b) of the 1934 Act), and the courts have read it to require a *breach of a fiduciary-like duty*. Two theories define when that breach exists.

![Two theories of why insider trading is illegal](/imgs/blogs/disclosure-the-prospectus-filings-and-insider-trading-5.png)

- **Classical theory.** A corporate insider — an officer, director, or employee — owes a duty to the company's shareholders. When that insider trades on material non-public company information without disclosing it, they breach that duty. This covers the CFO who buys before good earnings, the executive who dumps stock before a disaster.
- **Misappropriation theory.** What about an outsider — a lawyer, a banker, a printer of merger documents — who has no duty to the *target's* shareholders? The Supreme Court closed this gap in *United States v. O'Hagan* (1997): a person who **misappropriates** confidential information in breach of a duty owed to the *source* of the information (their law firm, their client) and trades on it commits fraud "in connection with" a securities transaction. The lawyer who trades on a client's secret merger breaches a duty to the client, not the target's shareholders — but it is still illegal.

Then there is **tipper-tippee** liability, from *Dirks v. SEC* (1983) and refined in *Salman v. United States* (2016). An insider (the **tipper**) who passes a tip to someone else (the **tippee**) is liable if the tipper breached a duty by tipping — which generally requires the tipper to receive a **personal benefit** (money, a favor, or simply the benefit of gifting valuable information to a friend or relative). The tippee is liable if they knew or should have known the tip came from a breach. This is how the law reaches the chain of people who never set foot in the company but trade on its leaked secrets.

#### Worked example: insider-trading profit and the penalty math

An associate at a law firm sees that her firm is advising on a \$2 billion all-cash acquisition of TargetCo at \$50 per share, while TargetCo trades at \$35. Before the deal is announced, she buys 40,000 shares at \$35 for \$1.4 million. When the deal is announced, TargetCo jumps to \$48 (near the offer). She sells: 40,000 × (\$48 − \$35) = **\$520,000** profit.

Now the penalty. Under the Insider Trading Sanctions Act, the SEC can seek **disgorgement** of the \$520,000 gain *plus* a civil penalty of up to **three times** that gain — here up to \$1.56 million — so she can owe roughly \$2.08 million on a \$520,000 profit. That is before criminal charges: insider trading can carry prison time (up to 20 years per count) and criminal fines. The expected value of the trade, once you weight in the probability of getting caught (high, because the SEC's market-surveillance algorithms flag well-timed pre-announcement buying) and the 4× downside, is sharply negative. The penalty structure is deliberately punitive precisely so that the math never works — because a market where it sometimes works is a market outsiders flee.

It is worth naming what is *not* a breach, because the boundary is where the doctrine gets interesting. A geologist who infers from public satellite data that a mining company has struck ore, and buys the stock, has no duty to anyone and trades legally — that is research, not theft. A trader who legally buys an entire block of stock and then, knowing their own intent to keep buying, anticipates the price impact, is acting on their own plans, not on misappropriated secrets. Even an employee who trades on a *suspicion* assembled from scattered public hints — the "mosaic theory" — is generally safe, because the pieces were public even if the synthesis was private. The law does not punish knowing more; it punishes betraying a trust to get the knowledge. That distinction is what keeps insider-trading law from accidentally banning the very analysis that makes prices informative in the first place.

The famous cases are worth knowing because they map onto these theories. **Ivan Boesky** in the 1980s ran an arbitrage operation fed by tips from investment bankers — classic tipper-tippee on merger information, and the case that made "greed is good" infamous. **Raj Rajaratnam** and the Galleon Group hedge fund were convicted in 2011 in a sweeping case built on wiretaps, with a network of tippers inside public companies; Rajaratnam got 11 years. The broader **SAC Capital / Steven Cohen** crackdown of the early 2010s pursued a hedge fund whose edge prosecutors alleged came partly from an "expert network" pipeline of inside information; SAC pleaded guilty and paid \$1.8 billion. **Martha Stewart** is the instructive oddity: she was *not* convicted of insider trading itself but of obstruction and lying to investigators about why she sold ImClone stock on a tip — a reminder that the cover-up is often what sinks the defendant.

## Why the ban exists: protecting disclosure, not punishing cleverness

Step back and ask the question that unlocks everything: *why* go to all this trouble? Why wiretaps and 20-year sentences over what is, mechanically, just buying a stock at the wrong time?

The answer is not that insider trading is "unfair" in some free-floating moral sense. It is that insider trading **destroys the disclosure compact**. The whole regime rests on a promise to the outsider: *the price you see reflects the publicly known facts, and the person on the other side of your trade does not have a secret you cannot get.* Disclosure makes the first half of that promise true. The insider-trading ban makes the second half true. Break it, and the outsider learns a brutal lesson — that the well-connected are systematically front-running the news — and does the rational thing: demands a wider spread to compensate for the risk of trading against someone who knows more, or stops trading altogether.

![The trust compact that keeps the secondary market liquid](/imgs/blogs/disclosure-the-prospectus-filings-and-insider-trading-7.png)

This is the **adverse-selection** problem that market-makers live with, and it shows up directly in the bid-ask spread. A market-maker quoting a stock does not know whether the next seller is a retiree rebalancing or an insider dumping ahead of bad news. To survive trading against the occasional informed insider, the maker widens the spread on *everyone*. So insider trading is not a victimless transfer from the company to the insider — it is a tax on every ordinary investor in the form of worse prices. (The formal models of how informed trading widens spreads live in the [order-book](/blog/trading/quantitative-finance/order-book-simulator-quant-research) and [market-making](/blog/trading/quantitative-finance/market-making-simulator-quant-research) literature; here we keep the narrative.)

![Bid-ask spread by liquidity tier](/imgs/blogs/disclosure-the-prospectus-filings-and-insider-trading-6.png)

You can see the cost of information risk in the spread structure across liquidity tiers. A mega-cap like Apple — heavily covered, intensely disclosed, hard to have an information edge on — trades at roughly a 1 basis point spread. A micro-cap, where information is scarce and the next trader might genuinely know something you don't, can carry an 80 basis point spread. Part of that 80× gap is liquidity, but part is exactly the adverse-selection premium: the less you trust that the price is clean, the more it costs you to trade. Insider-trading enforcement is, in effect, a program to keep that premium down across the whole market — to keep spreads tight by keeping the playing field trustworthy. That is why it belongs alongside the rest of [market-integrity machinery](/blog/trading/capital-markets/market-integrity-manipulation-spoofing-and-circuit-breakers).

## Common misconceptions

**"Disclosure means the company has to tell investors everything, all the time."** No. It must disclose *material* information on the prescribed schedule and must not make *misleading* statements. It can keep trade secrets, strategy, and ordinary business detail private. The duty is to not lie and to file the required reports — not to narrate the business in real time. Silence on an immaterial fact is fine; silence that makes a prior statement misleading is not.

**"Any trading on information the public doesn't have is insider trading."** This is the biggest myth. Legal edges are everywhere: superior analysis, mosaic-theory research that assembles public scraps into an insight, faster data feeds. What is banned is trading on *material non-public information obtained through a breach of duty*. An analyst who deduces a bad quarter from satellite images of empty parking lots has done nothing wrong; an executive who knows the quarter from the internal numbers and sells has.

**"Insider trading laws exist to make markets fair."** Fairness is the rhetoric; the function is liquidity. The ban exists because *unpoliced* insider trading makes outsiders distrust the price, widen spreads, and withdraw — strangling the secondary-market liquidity that the entire capital-raising machine depends on. It protects the disclosure compact, not fairness for its own sake.

**"Reg FD bans companies from talking to analysts."** No. Companies talk to analysts constantly. Reg FD only bans *selectively* disclosing *material non-public* information to them ahead of the public. A company can discuss public facts, color, and context all day; it just cannot hand the favored few a market-moving fact before everyone else gets it in a filing.

**"If I overheard a tip and traded, I'm fine because I'm not an insider."** Dangerous. Under tipper-tippee and misappropriation doctrine, a tippee who knew or should have known the information came from a breach of duty can be liable even with no connection to the company. The chain reaches the friend-of-a-friend who traded on the merger leak.

## How it shows up in real markets

The disclosure-and-enforcement machine is most visible in the cases where it bit hardest. The **Galleon** prosecution (2009–2011) was a turning point: for the first time, the government used Title III wiretaps — the tools of organized-crime cases — against a hedge fund, capturing Raj Rajaratnam on tape soliciting and trading on tips. The message to the industry was that insider trading would be prosecuted like racketeering, and the chilling effect on "expert network" channels was immediate and lasting.

The **SAC Capital** resolution in 2013 showed the regime reaching an entire firm: SAC pleaded guilty to securities fraud and paid \$1.8 billion, and Steven Cohen was barred from managing outside money for two years. The case crystallized a structural worry — that some funds' consistent edge came not from genius but from a pipeline of material non-public information — and pushed the whole industry to build compliance walls around how research touched company insiders.

On the disclosure side, the steady drumbeat of SEC enforcement against late or misleading filings, and against Reg FD violations, is less dramatic but more constant. Companies that fumble an 8-K timeline, or whose executives chat too freely with a favored analyst, settle quietly and pay. The market mostly never hears about it — which is the point. The regime works best when it is invisible, when the price is clean and the spread is tight because everyone simply assumes the disclosed facts are the real facts.

![US equity market cap year-end 2014 to 2024](/imgs/blogs/disclosure-the-prospectus-filings-and-insider-trading-8.png)

And the prize for getting this right is enormous. U.S. equity market capitalization grew from about \$26 trillion in 2014 to roughly \$58 trillion in 2024. That is tens of trillions of dollars of savings funneled into productive companies — capital that flows precisely because investors trust the disclosed numbers and trust that nobody is systematically trading against them on secrets. The disclosure regime and its insider-trading enforcement arm are, quietly, part of the infrastructure that makes those trillions willing to show up at all. The same compact underwrites the [exchanges and clearinghouses](/blog/trading/finance/stock-exchanges-and-clearinghouses) where those trillions change hands.

## The takeaway: disclosure plus the ban is the trust you're buying

The thing to carry away is that the prospectus, the 10-K, Regulation FD, and the insider-trading laws are not four separate rules. They are one machine with one job: to make a stranger willing to be the buyer.

Disclosure puts the facts on the table on a schedule — the prospectus to open, then the relentless 10-K / 10-Q / 8-K / proxy cadence forever after. Materiality decides what has to go on that table. Reg FD makes sure the facts reach everyone at the same instant. And the insider-trading ban — with its theories, its tipper-tippee chains, its punitive 3× penalties — exists to guarantee that nobody pockets the gap between when an insider knows and when you do. Strip out any one piece and the others weaken; strip out the enforcement and the disclosures become a fiction that insiders front-run.

So the next time you read about a perp walk over a well-timed trade, do not file it under "Wall Street greed" and move on. File it under *infrastructure maintenance*. The regulator is not avenging a moral wrong; it is protecting the single belief that lets the whole capital market function — that the price you see is the price the facts justify, and that the person on the other side of your trade does not know a secret you can't get. That belief is what regulation manufactures. It is the trust that makes markets liquid, and liquidity is what makes issuance — the funding of real companies and real projects — possible at all.

## Further reading & cross-links

- [Why markets are regulated: disclosure and the securities acts](/blog/trading/capital-markets/why-markets-are-regulated-disclosure-and-the-securities-acts) — the statutory foundation this post builds on.
- [Market integrity: manipulation, spoofing, and circuit breakers](/blog/trading/capital-markets/market-integrity-manipulation-spoofing-and-circuit-breakers) — the other half of keeping the market honest.
- [The IPO process end to end: from mandate to first trade](/blog/trading/capital-markets/the-ipo-process-end-to-end-from-mandate-to-first-trade) — the deal mechanics behind the S-1.
- [How a price is made: discovery, arbitrage, and efficiency](/blog/trading/capital-markets/how-a-price-is-made-discovery-arbitrage-and-efficiency) — why disclosed information ends up in the price.
- [Stock exchanges and clearinghouses](/blog/trading/finance/stock-exchanges-and-clearinghouses) — the venues where disclosed, trusted securities trade.
- [Order-book simulator](/blog/trading/quantitative-finance/order-book-simulator-quant-research) and [market-making simulator](/blog/trading/quantitative-finance/market-making-simulator-quant-research) — the microstructure models of how informed trading widens spreads.
