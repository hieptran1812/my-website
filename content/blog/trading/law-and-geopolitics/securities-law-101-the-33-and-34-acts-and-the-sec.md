---
title: "Securities law 101: the 1933 and 1934 Acts and the SEC"
date: "2026-06-19"
publishDate: "2026-06-19"
description: "Public markets run on mandatory disclosure. This is how the 1933 and 1934 Acts and the SEC force information into the open, and how that timing creates tradeable events."
tags: ["regulation", "securities-law", "sec", "disclosure", "ipo", "10b-5", "equities", "event-trading", "market-structure"]
category: "trading"
subcategory: "Law & Geopolitics"
author: "Hiep Tran"
featured: true
readTime: 42
---

> [!important]
> **TL;DR** — Public markets are not built on trust; they are built on *mandatory disclosure*, and two Depression-era statutes wrote the rules. The Securities Act of 1933 governs the one-time *sale* of new securities (the IPO "truth in securities" law). The Securities Exchange Act of 1934 created the SEC and governs everything *after* — ongoing reporting and anti-fraud.
>
> - Disclosure is what makes price discovery possible: a stock can only be priced on the facts that are legally forced into the open.
> - Filings cluster information into a knowable **calendar** — 10-K once a year, 10-Q three times, 8-K within four business days of a material event. Each filing is a scheduled (or unscheduled) information event the market must reprice.
> - **Rule 10b-5** is the anti-fraud backbone: it is illegal to make a materially false statement, or to omit a material fact, in connection with buying or selling a security.
> - The one number to remember: an 8-K for a truly material event must be filed within **four business days**. The faster you can read and price that filing, the larger your edge over everyone still digesting it.

On the morning of February 22, 2024, a company called Super Micro Computer was a \$45 billion AI-server darling that had quintupled in a year. By the close on October 30, 2024, its auditor, Ernst & Young, had resigned, citing an unwillingness "to be associated with the financial statements prepared by management," and the stock had been cut roughly in half. The trigger was not a product flaw or a lost customer. It was a *disclosure event*: a short-seller's report alleging accounting manipulation, a delayed 10-K annual report, and finally an 8-K announcing the auditor's exit. No new factory burned down. What changed was what the market was *legally entitled to know* — and could no longer trust.

That is the whole game. A public company is, in the eyes of the law, a machine for producing disclosure. The product you trade — the share price — is a running estimate of value built almost entirely out of information that securities law *forces* the company to publish. When the law changes what must be disclosed, or a company files something new, the information set changes, and the price moves to the new estimate. Understanding the disclosure regime is not a compliance chore for traders. It is a map of *where* and *when* new information enters the market, and therefore where repricing happens.

This post builds that map from zero. We will define what a "security" even is, walk through the two foundational Acts, decode the alphabet soup of filings (10-K, 10-Q, 8-K, S-1), explain what "material" means, and show how the disclosure calendar functions as the market's information clock. Then we get to the payoff: how to read the disclosure stream for an edge, with five worked dollar examples — an 8-K repricing, an IPO pop, a restatement, the value of reading a filing first, and the cost of a disclosure failure.

![Disclosure regime as the engine of price discovery, from law to filing to repricing](/imgs/blogs/securities-law-101-the-33-and-34-acts-and-the-sec-1.png)

## Foundations: what the law forces into the open

Start with a simple question: why does any of this exist? Before the 1933 and 1934 Acts, a company could sell stock to the public on the strength of a pretty brochure and a salesman's pitch. There was no requirement to publish audited financials, disclose insider ownership, or even tell the truth in any enforceable way. The 1929 crash and the Depression that followed were widely blamed, in part, on a market where the small investor was flying blind against insiders who knew the real numbers. Congress's answer was not to guarantee that investments would be *good*. It was to guarantee that investments would be *disclosed* — that every buyer would have access to the same material facts as the insiders. The legal philosophy is called **disclosure-based regulation**, and the United States chose it over the alternative ("merit regulation," where a government body decides whether an investment is worthy). The SEC does not vet whether a company is a good buy. It vets whether the company has told you the truth, fully, on time.

### What is a "security"?

A **security** is a tradeable financial instrument representing some financial value — most commonly a *share* of ownership in a company (stock), a *debt* claim (a bond), or a derivative of either. But the legal definition is famously broad, and the test for whether something counts comes from a 1946 Supreme Court case, *SEC v. W.J. Howey Co.* The **Howey test** says an arrangement is an "investment contract" (and therefore a security) if it has four features:

1. an **investment of money**,
2. in a **common enterprise**,
3. with an **expectation of profit**,
4. derived **from the efforts of others**.

That fourth prong is the heart of it. If you buy a thing expecting to profit mostly from *your own* work (say, a franchise you operate), it is probably not a security. If you buy a thing expecting to profit from *someone else's* work (the company's management running the business), it is. This is why the SEC argues many crypto tokens are securities — buyers send money to a common project expecting the founders' efforts to make the token rise — and why that fight is so consequential. (We unpack that turf war in the companion post on crypto regulation.) For the rest of this post, "security" means stocks and bonds of public companies, where the rules are settled.

Why does the *definition* matter to a trader rather than only to a lawyer? Because the label "security" is what *switches on the entire disclosure machine*. If an instrument is a security, the issuer owes you registration, a prospectus, periodic reports, and anti-fraud liability — a thick, reliable information stream. If it is not (a collectible, a commodity, a pure currency), none of that applies, and you are back to the pre-1933 world of caveat emptor. Whole asset classes live or die on which side of the Howey line they fall: a token that is ruled a security must either register (expensive, slow) or trade only under an exemption (restricted), while a token ruled a commodity escapes the SEC's disclosure regime entirely and falls instead to the CFTC's lighter touch. When a court reclassifies an instrument, it is not a technicality — it changes what information must legally exist about that instrument, and therefore how confidently it can be priced.

It is worth naming the two great categories of security, because the disclosure regime treats them alike but the *cash-flow rights* differ sharply. **Equity** (common and preferred stock) is an ownership claim: you own a residual slice of the company, you can vote, and you are paid last — dividends are discretionary and in a bankruptcy you stand behind every creditor. **Debt** (bonds, notes) is a contractual claim: the issuer owes you fixed coupons and your principal back, you generally cannot vote, and you are paid *before* equity. Both are securities; both trigger disclosure; but a 10-K means something different to a bondholder (who cares most about the balance sheet and whether the company can pay) than to a shareholder (who cares most about growth and the income statement). The fixed-income series treats the bondholder's lens in depth; here, keep in mind that the *same filing* feeds two audiences asking different questions.

### The 1933 Act: truth in the sale

The **Securities Act of 1933** governs the **primary market** — the moment securities are *created and sold for the first time*, most visibly an Initial Public Offering (IPO) or a follow-on offering. Its core command is simple: before you can sell a new security to the public, you must **register** it with the SEC by filing a detailed disclosure document, and you must give every buyer a **prospectus** distilling that document. The slogan at the time was "truth in securities."

The registration document is called the **S-1** (for a US company's IPO). It is a remarkable artifact: a private business's first complete public confession. It contains audited financial statements, a description of the business and its strategy, the use of proceeds (what the company will do with the money), a long list of **risk factors**, details on executive compensation, related-party transactions, and the ownership stakes of insiders. For an investor, the S-1 is often the single richest information event in a company's life, because it reveals numbers that were previously secret.

The S-1 is the *registration statement*; the **prospectus** is the part of it that must be delivered to every buyer. In practice the working version circulated during the roadshow is the **preliminary prospectus**, nicknamed the "**red herring**" because of the red-ink legend on its cover warning that the registration is not yet effective and the price is not yet set. The **final prospectus** carries the actual offering price and is filed once the SEC declares the registration "effective." The legal fiction worth internalizing is that the prospectus, not the company's PR, is the *official* basis on which you are deemed to have decided to buy — which is why everything a company says around an offering is measured against it.

The 1933 Act splits the offering into three regulated periods, and the rules differ in each:

- The **pre-filing period** — before the S-1 is filed. The company is in "registration" and may not "condition the market" by stirring up interest in the coming offering. Routine business communications are fine; offering hype is not.
- The **waiting period** — between filing and effectiveness, while the SEC reviews. Here the company may make *oral* offers and circulate the preliminary (red-herring) prospectus, but it cannot sell or take binding orders. This is when the roadshow happens.
- The **post-effective period** — after the SEC declares the registration effective. Now sales can close, and every buyer must get the final prospectus.

This structure is why the 1933 Act imposes a **quiet period** and a set of **gun-jumping rules**. "Gun-jumping" means promoting the offering before the registration is effective — like a sprinter starting before the gun. From the moment a company decides to go public, its ability to hype the stock is sharply curtailed: it can essentially only speak through the official prospectus. This is why you do not see CEOs doing splashy interviews in the weeks before an IPO, and why an offhand comment can force a company to "cool off" and delay. The law wants the *document*, not the marketing, to be the basis of your decision.

The enforcement teeth behind all this is **Section 11** of the 1933 Act, which creates near-strict liability for material misstatements or omissions in a registration statement. If the S-1 contains a material falsehood, investors who bought in the offering can sue not only the company but the directors, the officers who signed, the underwriters, and the auditors — and the defendants must prove *due diligence* to escape. This is why underwriters spend weeks combing through a company's records before an IPO: their own liability rides on the accuracy of the disclosure. For a trader, Section 11 is the reason an S-1's numbers are unusually trustworthy: a lot of well-capitalized parties are personally on the hook if they are wrong.

### The 1934 Act: the SEC and the rest of the company's life

The **Securities Exchange Act of 1934** does three big things. First, it created the **Securities and Exchange Commission (SEC)** — the federal agency that writes and enforces the rules. Second, it governs the **secondary market**, where already-issued securities trade between investors (the stock exchange you actually use). Third, and most importantly for traders, it imposes **continuous disclosure**: a public company must keep telling the truth, forever, on a schedule, through periodic reports.

The periodic reports are the famous filings:

- **10-K** — the annual report. Audited financials, a full Management's Discussion and Analysis (MD&A), updated risk factors, the works. Large companies must file it within ~60 days of fiscal year-end.
- **10-Q** — the quarterly report, filed three times a year (the fourth quarter is folded into the 10-K). Unaudited but still detailed; due within ~40 days of quarter-end for large filers.
- **8-K** — the *current* report. Filed on demand, **within four business days**, whenever a *material event* occurs between scheduled reports: an acquisition, a CEO departure, a bankruptcy, the loss of a major customer, a change of auditor, results of a shareholder vote. The 8-K is the market's real-time newswire of legally mandated facts.

The 1934 Act also gives us **proxy rules** (Regulation 14A), which govern how companies solicit shareholder votes and require a detailed **proxy statement** (the "DEF 14A") before the annual meeting, disclosing executive pay and board nominees. And it gives us **Section 16**, which requires corporate insiders — officers, directors, and 10%+ owners — to publicly report their own trades in the company's stock (on **Forms 3, 4, and 5**), usually within two business days. Insider buying and selling is therefore *itself* a disclosed, tradeable signal. Section 16 even has a "short-swing profit" rule that disgorges any profit an insider makes buying and selling within six months — a blunt deterrent against insiders front-running their own company's news.

Three more 1934-Act disclosure channels deserve a name, because each is a distinct information feed a trader can read:

- **Schedule 13D / 13G** — anyone who acquires more than **5%** of a public company's voting stock must disclose it. A 13D (the "activist" version) is filed when the buyer intends to influence control; a 13G (the "passive" version) when they do not. A fresh 13D from a known activist is, by itself, a tradeable event: it announces that a sophisticated investor has taken a big stake and may push for change.
- **Form 13F** — large institutional managers (over \$100 million in US equities) must disclose their holdings each quarter, 45 days after quarter-end. This is how the market reconstructs "smart money" positioning, albeit on a lag.
- **Tender-offer rules (Regulation 14D/14E)** — govern how a buyer makes a public offer to purchase shares, including in a takeover. These rules set the disclosure and timing that make merger-arbitrage trading possible (the subject of a dedicated post on merger arb and regulatory deal risk).

The unifying idea: the 1934 Act does not just make the *company* disclose; it makes *significant participants* disclose their positions and intentions. The market's information set therefore includes not only "what is the company doing" but "who owns it, who is buying, and who wants control" — and each of those is a legally-mandated, periodically-updated signal.

![Matrix comparing the 1933 Act and 1934 Act across coverage, timing, filings, and enforcement](/imgs/blogs/securities-law-101-the-33-and-34-acts-and-the-sec-2.png)

### What "material" means — the load-bearing word

Almost every rule above hinges on one word: **material**. The legal standard, from *TSC Industries v. Northway* (1976) and *Basic v. Levinson* (1988), is that a fact is material if there is "a substantial likelihood that a reasonable investor would consider it important" in making an investment decision — equivalently, that disclosing it would have "significantly altered the total mix of information available."

Notice what this does *not* say. It does not say "anything that moves the stock 5%." Materiality is a legal standard about *importance to a reasonable investor*, judged in context, not a fixed price threshold. A 2% revenue miss might be immaterial for a stable utility and extremely material for a hyper-growth name priced for perfection. For events that are *contingent* (a pending merger, a lawsuit), *Basic v. Levinson* says materiality depends on the **probability** of the event times the **magnitude** of its effect — a probability-weighted standard that, conveniently, is exactly how a trader thinks about an uncertain catalyst.

The reason materiality matters so much: it defines the *trigger* for mandatory disclosure (you must file an 8-K for a *material* event) and the *boundary* of fraud (it is only illegal to lie about, or omit, *material* facts). The entire regime is calibrated to "information a reasonable investor would care about" — which is another way of saying "information that should move the price."

The probability-times-magnitude rule from *Basic v. Levinson* is worth dwelling on, because it is the bridge between legal materiality and how an option or an arbitrageur actually prices an uncertain event. The court was deciding when a company must disclose a *contingent* event — there, merger talks. Its answer: a contingency is material when the probability that it happens, weighted by the magnitude of its effect on the company, is large enough that a reasonable investor would care. That is precisely the expected-value framework a trader uses to price a binary catalyst.

#### Worked example: pricing a binary catalyst from probability times magnitude

Suppose "Helix Therapeutics" trades at \$30. It has one drug awaiting an FDA decision (a "PDUFA date") in 60 days. The market believes the drug has a 50% chance of approval. If approved, the company is worth \$50 a share; if rejected, it is worth \$14 (the value of its cash and pipeline minus the wasted spend).

The probability-weighted fair value, ignoring the time value of money over two months, is:

(0.50 × \$50) + (0.50 × \$14) = \$25 + \$7 = **\$32**.

But the stock trades at \$30, implying the market's *real* probability of approval is lower than 50%. Solve for the implied probability p: p × \$50 + (1 − p) × \$14 = \$30 → \$36p = \$16 → p ≈ **44%**. The intuition: the disclosure that *resolves* this (an 8-K announcing FDA action) will snap the price from \$30 to either \$50 (a +67% move) or \$14 (a −53% move) the instant it is filed — and the legal materiality standard is screaming "disclose immediately," because probability (44%) times magnitude (a \$36 swing) is enormous. Legal materiality and trade-able expected value are the same calculation in different vocabularies.

## The disclosure calendar is the market's information clock

Put the periodic filings on a timeline and a structure appears. For a company with a December fiscal year-end, the year looks like this: the 10-K lands in late February, three 10-Qs land in May, August, and November, the proxy and annual meeting cluster in spring, and 8-Ks fire whenever something material happens in between. The *scheduled* filings convert the company's continuous reality into discrete, pre-announced information events. The *unscheduled* 8-K is the wildcard.

![Annual disclosure calendar timeline showing 10-K, 10-Q, proxy, and 8-K cadence](/imgs/blogs/securities-law-101-the-33-and-34-acts-and-the-sec-3.png)

This calendar is why "earnings season" exists and why it matters so much to traders. Four times a year, in a compressed window, thousands of companies release fresh, audited-or-reviewed numbers under threat of fraud liability if they lie. Implied volatility in options reliably rises into an earnings date and collapses after (the "vol crush"), precisely because everyone knows a large, scheduled information event is coming. We dig into the mechanics of trading these scheduled releases in the event-trading series; here the point is structural: *the law schedules the information, and the schedule is public, so the market pre-positions around it.*

The 8-K is the more interesting beast for an edge-seeker, because it is *not* scheduled. When a CEO abruptly resigns, when a drug fails a trial, when a customer representing 30% of revenue walks, the company has a hard legal deadline — four business days — to file an 8-K describing the event. That filing is often the first *authoritative* version of the news, distinct from rumor. The trader who is set up to ingest 8-Ks the instant they hit the SEC's EDGAR system, parse them, and act, is reading the market's information feed at the source.

It is worth being concrete about **EDGAR** — the Electronic Data Gathering, Analysis, and Retrieval system — because it is the literal pipe through which mandated disclosure reaches the public. Every registered filing in the United States goes into EDGAR, and the system timestamps and publishes filings essentially in real time during the day (with a brief processing lag). The feed is *free* and *machine-readable*, which is the great democratizing fact of US securities law: a retail trader and a hedge fund pull from the same source at the same moment. The difference between them is not access but *speed of ingestion and depth of reading*. Quant funds run programs that watch the EDGAR feed, classify each new 8-K by item number, extract the numbers, and act within seconds — turning a legal disclosure obligation into a low-latency data signal. The disclosure regime did not just make information *exist*; it made it *structured, timestamped, and free*, which is exactly what an algorithm needs.

Two subtleties about the 8-K's four-day clock matter for trading. First, the clock runs from when the event *occurs* (or the company *learns* of it), and "four business days" can stretch a real-world week across a weekend — so the authoritative filing sometimes trails the rumor by days, during which the stock trades on speculation. Second, certain 8-K items (notably Item 2.02, earnings results) are technically "furnished" rather than "filed," a legal distinction that changes the liability standard but not the practical reality that the market reads them the instant they post. The point for a trader: the 8-K is the *truth*, but it is not always the *first* version of the news — the gap between rumor and the authoritative filing is itself a window where the price can be unstable and wrong.

#### Worked example: an 8-K material-event repricing

Suppose mid-cap "Northwind Logistics" trades at \$40.00 on 50 million shares, a \$2.0 billion market cap. Analysts model it at 16× forward earnings of \$2.50, and a key assumption is that its single largest customer, accounting for 25% of revenue, renews a contract. After the close, Northwind files an 8-K under Item 1.02 ("Termination of a Material Definitive Agreement"): the customer is *not* renewing.

How does the market reprice? Walk the chain. If 25% of revenue vanishes and the business has high fixed costs, analysts might cut forward earnings from \$2.50 to \$1.70 — a 32% earnings hit, amplified by operating leverage. Worse, losing your biggest customer is a *signal* about competitiveness, so the market may also compress the multiple from 16× to 13×. New fair value:

\$1.70 EPS × 13 = **\$22.10**, versus the prior \$40.00.

That is a **−44.8%** repricing, or about \$890 million of market cap erased, all triggered by a single filing. Note both levers moved: \$0.80 of lost earnings and ~3 turns of lost multiple. The intuition: an 8-K does not just update a number; it can update the market's *opinion of the business*, which is why material events so often overshoot a naive earnings-only estimate.

## Rule 10b-5: the anti-fraud backbone

If disclosure is the engine, **Rule 10b-5** is the law that keeps the engine honest. Promulgated by the SEC under Section 10(b) of the 1934 Act, it is short enough to quote in spirit: it is unlawful, in connection with the purchase or sale of any security, to (a) employ any device or scheme to defraud, (b) make any untrue statement of a *material* fact, or omit a material fact necessary to make statements not misleading, or (c) engage in any act that operates as a fraud.

Three things make 10b-5 the workhorse of securities enforcement:

1. **It covers omissions, not just lies.** If a company has said something that is now misleading because it left out a material fact, staying silent can itself be fraud. This is the "duty to update / duty to correct" territory.
2. **It is the basis for insider-trading cases.** Trading on material non-public information (MNPI) in breach of a duty is prosecuted as 10b-5 fraud (the "deception" being the breach of trust). The full mechanics — what is actually illegal, and what Regulation FD requires — are the subject of the companion post on insider trading and Reg FD.
3. **It supports private lawsuits.** Defrauded investors can sue under 10b-5, which is why a stock that drops after a revelation of prior misstatements often attracts a securities class action — itself a disclosable, financially material event.

For a trader, 10b-5 is the reason disclosed information can be *trusted enough to price on*. A 10-K is not just a document; it is a document the executives signed under penalty of fraud liability (and, post-Sarbanes-Oxley, personal CEO/CFO certification — see the disclosure-and-accounting-law post). That credibility is what lets the market treat a filing as a reliable input rather than marketing.

A handful of doctrines fill out how 10b-5 actually works, and each has a market consequence:

- **Scienter.** A 10b-5 violation generally requires *scienter* — an intent to deceive, or at least reckless disregard for the truth. An honest mistake corrected in good faith is not fraud. This matters because it tells you which corporate stumbles become legal cases (and second, multiplier hits to the stock) versus which are merely operational misses.
- **Reliance and "fraud-on-the-market."** In a private class action, each investor would normally have to prove they personally relied on the lie. *Basic v. Levinson* solved this with the **fraud-on-the-market** theory: in an efficient market, the stock price already reflects all public information, so a public lie that distorts the price defrauds *everyone* who traded at that price — reliance is presumed. This is the legal embodiment of the efficient-market idea, and it is why a single material misstatement can support a class action covering every shareholder during the "class period."
- **Loss causation.** Plaintiffs must show the *truth coming out* is what caused their loss — typically a "corrective disclosure" (a later 8-K, a short report, a restatement) followed by a price drop. This is why the *date the truth is revealed* is so financially loaded: it both triggers the repricing and starts the legal clock.

There is also a crucial *safe harbor* a trader must know: the **Private Securities Litigation Reform Act (PSLRA) forward-looking-statement safe harbor**. Forward-looking statements — guidance, projections — are protected from 10b-5 liability if accompanied by "meaningful cautionary language." This is why every earnings call opens with a "safe harbor statement" and why guidance is hedged with caveats. The practical effect: companies are far more exposed for lying about *historical facts* (last quarter's revenue) than for being wrong about the *future* (next year's guidance). A trader weighing the litigation risk in a stock should weight stated historical numbers as near-sacred and guidance as legally softer — which is itself a clue about where management has room to spin.

#### Worked example: the value of an information advantage from reading a filing first

Imagine "Cascade Semiconductor" reports a clean headline beat at 4:01 p.m.: EPS \$1.20 vs \$1.10 expected. The stock jumps 4% in the after-hours print, to \$104 from \$100, as algorithms react to the headline number. But the full 10-Q, posted to EDGAR at 4:06 p.m., contains a footnote: gross margin fell 300 basis points because of a one-time inventory write-down, *and* management's guidance language for next quarter quietly shifted from "expect growth" to "expect flat-to-down."

A trader who reads the actual filing in the five minutes before the rest of the market digests it sees that the "beat" is low-quality and guidance is deteriorating. Suppose the correct reaction, once everyone reads the footnotes, is a *fall* to \$94 (−6%), not a rise to \$104.

The edge: you can short at the temporary \$104 print and cover near \$94 as the market re-reads. On 10,000 shares:

(104 − 94) × 10,000 = **\$100,000** of edge, captured purely from reading the disclosure faster and more carefully than the headline-trading crowd.

The intuition: the law guarantees the information is *published* and *equal* — it does not guarantee everyone *reads* it at the same speed or depth. Disclosure equality is an opportunity, not a leveler, for whoever does the work. (This is legal: you are trading on *public* information, just faster. Trading on *non-public* information is the crime — that line is the whole subject of the insider-trading post.)

## The IPO pipeline: where a business first reveals itself

The 1933 Act's registration process is best understood as a pipeline with legal checkpoints. A private company decides to go public; it files the S-1; the SEC reviews it and sends **comment letters** (questions the company must answer and re-file, often several rounds); the company goes on a **roadshow** to pitch institutional investors (constrained by the quiet period); the underwriters **build a book** of demand; the night before trading, the deal is **priced**; and the next morning, the stock **opens** for public trading — often at a price well above the IPO price, the famous "**IPO pop**."

![IPO registration pipeline from S-1 filing through SEC review, roadshow, pricing, and first trade](/imgs/blogs/securities-law-101-the-33-and-34-acts-and-the-sec-4.png)

That pop is a direct consequence of how the pipeline allocates information and shares. The IPO price is set by the company and underwriters the night before, sold to favored institutional clients. The opening price is set by the *whole market* the next morning. When the open is far above the IPO price, value was transferred from the issuing company (which sold low) to the IPO allocants (who flipped high). This is the classic "money left on the table."

#### Worked example: IPO pricing versus first-trade pop and the dollars left on the table

Suppose "Lumen AI" sells 20 million shares in its IPO at \$30.00, raising 20,000,000 × \$30 = **\$600 million**. The next morning, the stock opens and trades at \$48.00 — a 60% first-day pop.

The market is telling us the shares were "worth" \$48 at the open. The company sold them for \$30. The gap is value the company *could* have raised but didn't:

(48 − 30) × 20,000,000 = **\$360 million** left on the table.

Said differently: the same 20 million shares were worth \$960 million at the open, but the company collected only \$600 million. The \$360 million difference accrued to whoever received IPO allocations at \$30 and could sell at \$48. The intuition: a big IPO pop is not unambiguous good news for the *company* — it is a signal the deal was underpriced, and a measure of wealth handed from the issuer to allocated insiders. As an investor, a screaming pop should make you ask *who got the cheap shares*, not just *how exciting is this story*.

The S-1 itself is where you do the homework that the pop's excitement papers over. When a company files its S-1, you can read — for the first time — its real revenue growth, its margins, its customer concentration, its stock-based compensation, and the candid risk factors its lawyers insisted on. Many a hyped IPO has been deflated by a careful read of the S-1 revealing decelerating growth or a path to profitability that requires heroics. The filing is the antidote to the roadshow.

## Registered versus exempt: the private-markets escape hatch

Not every securities sale goes through the full S-1 registration. Registration is expensive and slow, so the law provides **exemptions** — ways to raise money without registering, in exchange for restrictions on *who* can buy and whether they can *resell*. This is the legal foundation of private markets, and it explains why so much value creation now happens *before* a company ever files an S-1.

The big exemptions:

- **Regulation D (Reg D), Rule 506** — by far the most-used. Lets a company raise *unlimited* money from **accredited investors** (roughly, individuals with \$1 million+ net worth excluding their home, or \$200k+ income) with light disclosure and no SEC review. The catch: shares are **restricted** — generally not freely resellable for a holding period (the Rule 144 framework). This is how virtually all venture-backed startups raise capital.
- **Rule 144A** — lets large institutions (**Qualified Institutional Buyers**, or QIBs, generally managing \$100 million+) trade unregistered securities among themselves. The backbone of the private high-yield bond market.
- **Regulation A+ (Reg A+)** — a "mini-IPO" allowing public-ish raises up to \$75 million a year with a lighter disclosure document, available to retail investors with caps.

![Matrix comparing registered IPO with Reg D, Rule 144A, and Reg A+ exempt offerings](/imgs/blogs/securities-law-101-the-33-and-34-acts-and-the-sec-6.png)

The trade-off is always the same: **registration buys you the broadest possible investor base (anyone, including retail) at the cost of full, ongoing disclosure; an exemption frees you from the paperwork but locks out most investors.** Understanding this is essential for a market participant because it explains a structural shift: companies now stay private far longer (funded by Reg D rounds from venture and growth investors) and IPO later and larger, which means the early, high-growth phase is increasingly closed to public-market investors. The information you *can* see in public filings is, increasingly, the information about more mature businesses.

The **Rule 144** holding-period framework is the hinge between these two worlds and worth a concrete grasp, because it governs how — and when — privately-issued restricted stock can ever reach the public market. When you buy shares in a Reg D round, they carry a restrictive legend; you generally cannot resell them to the public for a **six-month** holding period (for a reporting company; one year if it does not file reports), and even then only subject to volume limits and a current public-information requirement if you are an affiliate. This is the legal reason an IPO is followed by a **lock-up** expiry: pre-IPO holders and insiders agree (and Rule 144 reinforces) not to dump shares for typically 90–180 days. The expiry of a lock-up is a *scheduled, disclosed* supply event — a known date on which a flood of previously-restricted shares becomes sellable, which frequently pressures the stock. A trader who maps lock-up expiries onto the calendar is reading a supply-side disclosure signal that the offering documents told them about months earlier.

#### Worked example: a lock-up expiry as a scheduled supply shock

Suppose "Orbital Robotics" IPO'd 25 million shares to the public, but pre-IPO insiders and venture funds hold another 175 million restricted shares under a 180-day lock-up. The stock trades at \$40 on the 25 million-share public float — a thin float that helped the price run up. On the lock-up expiry date, the 175 million restricted shares become eligible to sell.

The float is about to grow eightfold: from 25 million to potentially 200 million shares. Even if only 20% of the newly-freed shares hit the market in the first weeks — 0.20 × 175,000,000 = 35 million shares — that is more than the entire prior public float arriving as supply. With demand roughly fixed in the short run, a supply shock of that size routinely drives a high-single-digit to double-digit decline into and around the expiry. If the price gives back, say, 12%, that is \$4.80 a share, or \$960 million of value across the 200 million shares, repriced by a *known, disclosed* date rather than any change in the business. The intuition: the disclosure regime told you the share count, the holders, and the lock-up term in the S-1 — the lock-up expiry is one of the most legible, calendar-able supply events in markets, hiding in plain sight in the offering documents.

### How the JOBS Act changed the on-ramp

The **Jumpstart Our Business Startups (JOBS) Act of 2012** rewired the IPO process for smaller companies, creating the category of **Emerging Growth Company (EGC)** — generally a firm with under ~\$1.235 billion in revenue. An EGC gets an "IPO on-ramp" with reduced burdens: it can **file its S-1 confidentially** (so the SEC reviews it privately before the company commits to a public timeline), provide only two years of audited financials instead of three, and "**test the waters**" with institutional investors before the formal process. The JOBS Act also created the Reg A+ mini-IPO and (in Title III) **equity crowdfunding** under Regulation Crowdfunding (Reg CF), letting startups raise small amounts from ordinary investors.

For a trader, the JOBS Act's most visible effect is the **confidential S-1**: a company can be deep into the IPO process before the public sees a single number. When a confidentially-filed S-1 is finally made public (required at least 15 days before the roadshow), it is a genuine new-information event — the first public look at a business that has been preparing in secret.

## Common misconceptions

Five myths trip up newcomers, each correctable with a number or a case.

**Myth 1: "The SEC vets investments, so a registered IPO is safe."** No. The SEC reviews *disclosure*, not *merit*. The agency's own rule is explicit, and the prospectus even says so. Plenty of fully-registered, SEC-reviewed IPOs have been disasters: WeWork's 2019 S-1 was complete and compliant — it disclosed enormous losses and bizarre related-party deals — and the IPO collapsed *because* careful readers digested those disclosures. The S-1 worked exactly as designed: it told the truth, and the truth was unappealing. The SEC's job was done when the facts were on the table; pricing the facts is the market's job, and yours.

**Myth 2: "All material information moves the stock the same way the news sounds."** No — the *quality* of a beat or miss matters more than the headline. In a study of earnings reactions, a meaningful share of companies that "beat" on headline EPS still fall, because guidance, margins, or revenue mix disappoint. Recall the Cascade Semiconductor example: a \$1.20 vs \$1.10 "beat" that should send the stock *down* 6% once the footnotes are read. The headline number is the bait; the filing is the meal.

**Myth 3: "If a company restates earnings, it is just an accounting technicality."** No — restatements are among the most violent repricings in markets, because they attack the *credibility of all the company's numbers at once*. A classic study (the GAO's restatement research) found that companies announcing restatements lost, on average, roughly **10%** of market value in the days around the announcement — and severe restatements (those touching revenue or alleging fraud) far more. The point is not the single corrected number; it is that if last year's earnings were wrong, the market must now discount *every* figure the company reports, which compresses both earnings *and* the multiple.

**Myth 4: "Reading a company's SEC filings before the rest of the market is insider trading."** No — and this confusion costs people real edge. EDGAR is public; anything a company files is, by definition, *public* information the moment it posts. Reading a 10-Q's footnotes faster and more carefully than a headline-trading algorithm is not insider trading; it is *research*. Insider trading is trading on material *non-public* information in breach of a duty — say, a CFO tipping a friend before the 8-K is filed. The line is whether the information is public yet. The entire premise of the worked example where you capture \$100,000 by reading a filing first is *legal* precisely because the filing was public; you were simply faster. (Where exactly that line sits — and how Regulation FD forces companies to disclose to everyone at once — is the subject of the insider-trading post.)

**Myth 5: "A big company can sit on bad news as long as it wants until the next quarterly report."** No — that is the whole point of the 8-K and the four-business-day clock. A genuinely material event cannot wait for the next 10-Q; it must be disclosed promptly. And even where there is no specific 8-K trigger, the anti-fraud rules create a *duty to correct* a prior statement that has become misleading and a *duty to update* certain forward-looking assurances. The combination is why a company that "knew for weeks" and stayed silent faces 10b-5 exposure, and why the gap between when management *learns* something and when it *files* is a legally policed window — not an open-ended one. The deadline is the reason the disclosure stream is timely enough to trade.

## How it shows up in real markets

The abstractions above are visible in concrete events. Three illustrate the regime in action.

**GameStop, January 2021 — disclosure of positioning as the fuel.** GameStop's squeeze is usually told as a meme-stock story, but it is also a *disclosure* story. Two regulatory disclosures lit the fuse. First, **short-interest reporting**: exchanges publish aggregate short interest twice a month, and GameStop's was disclosed at *over 100% of its float* — a legally visible, extreme number that told everyone the stock was a tinderbox. Second, **13F filings** (which require large institutions to disclose their holdings quarterly) and **Form 4** insider filings let the crowd track who was positioned where. The stock ran from \$17.25 on January 4 to a \$347.51 close on January 27, then collapsed. The repricing was not driven by any change in the *business* — GameStop filed no transformative 8-K — but by disclosed *positioning* data colliding with a coordinated buy-in. It is a reminder that mandated disclosure includes not just company facts but *market structure* facts, and those can move prices just as hard.

![GameStop January 2021 daily close repricing during the public short squeeze](/imgs/blogs/securities-law-101-the-33-and-34-acts-and-the-sec-7.png)

**The general principle: information events drive volatility.** Step back and look at how violently markets can reprice when a shock hits. The VIX — the market's "fear gauge," measuring expected 30-day volatility — spikes precisely when a large, surprising information event overwhelms the prevailing price consensus. A calm market sits near 20; a genuine shock can triple that in a day. The magnitudes below are the macro-scale version of what an 8-K does to a single stock: a sudden change in the information set forces a violent re-estimate of value, and the speed of that re-estimate *is* volatility.

![VIX closing level at selected stress events showing how violently shocks reprice volatility](/imgs/blogs/securities-law-101-the-33-and-34-acts-and-the-sec-8.png)

**An S-1 revealing a business — the WeWork tell.** When WeWork filed its S-1 in August 2019, the document — required by the 1933 Act — revealed losses scaling *with* revenue (it lost about \$1.61 for every \$1 of revenue at one point), governance arrangements that enriched the founder, and a "Community Adjusted EBITDA" metric that stripped out most real costs. None of this was hidden; the S-1 disclosed it all, exactly as the law demands. The market read the filing and the deal imploded within weeks, the IPO was pulled, and the implied valuation fell from \$47 billion to a fraction of that. This is the 1933 Act functioning perfectly: the *truth* was disclosed, and disclosure of the truth is what killed the deal.

The thread connecting all three cases is that the *information event* — not the underlying business reality, which changed slowly or not at all — drove the repricing. GameStop's stores were the same the day before and the day after the squeeze; what moved was the collision of disclosed positioning data with a crowd. WeWork's losses on the day the IPO was pulled were identical to the day it filed; what moved was that the market finally *read* the disclosure. The VIX spikes mark moments when a surprise overwhelmed the prior consensus faster than orderly trading could absorb. In each case, value was repriced when the *information set* changed — which is exactly the mechanism the disclosure regime is built to schedule and police. The trader's job is to be early and accurate about *when the information set changes*, and the filings calendar is the closest thing to a published schedule of those moments.

#### Worked example: a restatement's hit to EPS and multiple

Take "Beacon Software," trading at \$60 on reported EPS of \$3.00 — a 20× multiple. The company files an 8-K under Item 4.02 ("Non-Reliance on Previously Issued Financial Statements"): it must restate the last two years because it recognized revenue too early. After the restatement, true EPS was \$2.40, not \$3.00 — a 20% earnings cut.

If only earnings changed, fair value would be \$2.40 × 20 = \$48 (−20%). But a restatement also breaks *trust*, so the market demands a lower multiple for the now-suspect numbers — say it compresses from 20× to 15×:

\$2.40 × 15 = **\$36.00**, a **−40%** move from \$60.

The earnings cut alone explains 20 points of the drop; the multiple compression — the "credibility tax" — explains the other 20. The intuition: restatements are double-barreled, hitting the numerator (earnings) *and* the denominator-of-trust (the multiple) at once, which is why they so reliably produce outsized declines. This is the mechanism behind Myth 3's ~10% average — and far worse for the severe cases.

## Enforcement: the teeth behind the rules

Disclosure rules would be theater without enforcement, and the SEC has a graduated toolkit. For civil violations it can seek **disgorgement** (clawing back ill-gotten gains), **civil monetary penalties** (fines, often a multiple of the gain), and **injunctions**. Administratively, it can impose **officer-and-director bars** (barring an individual from serving as an executive of a public company) and **suspend or revoke** the registrations of regulated firms. For willful fraud, the SEC refers cases to the Department of Justice for **criminal prosecution**, which can mean prison.

It helps to understand *how* the SEC actually proceeds, because the procedure produces disclosable milestones the market trades on. An investigation usually begins quietly (a "Matter Under Inquiry," then a "Formal Order of Investigation" that grants subpoena power). If the staff concludes a violation occurred, it sends the target a **Wells Notice** and the target submits a **Wells Submission** arguing against charges. The SEC then either files a **civil complaint in federal court** (where it can win injunctions, disgorgement, and penalties) or brings an **administrative proceeding** before its own in-house judges (where bars and suspensions are common). Most cases **settle** — typically "neither admitting nor denying" the allegations — with a consent decree, a fine, and undertakings. Each of these steps can be a material, disclosable event: the company often must reveal a Wells Notice, a settlement is announced, and a litigated loss reprices the stock anew.

The practical lesson for a trader is that enforcement creates a *sequence* of repricing moments, not one. The market reprices on the *risk* of charges (when an investigation or short report surfaces), again on the *fact* of charges (the complaint or Wells disclosure), and a third time on the *resolution* (settlement size, admissions, executive bars). A name under active SEC scrutiny is therefore an unusually event-dense stock — and the events are, by law, disclosed as they cross the materiality threshold. The careful reader who tracks the docket and the 8-K stream knows the rhythm of these catalysts before the casual observer registers that anything is wrong.

![SEC enforcement toolkit tree from civil remedies and administrative actions to criminal referral](/imgs/blogs/securities-law-101-the-33-and-34-acts-and-the-sec-5.png)

The reason this matters to a trader is that an *enforcement action is itself a disclosable, material event*. A **Wells Notice** (the SEC's formal warning that it intends to recommend charges) often must be disclosed and can crater a stock on its own. A settlement that includes a large fine and a CEO bar reprices the company twice: once for the cash cost, and again for the governance and reputational damage that compresses the multiple. And under the SEC's **whistleblower program** (created by Dodd-Frank), insiders are paid 10–30% of sanctions over \$1 million for tips — which means disclosure failures are increasingly likely to be reported from the inside.

#### Worked example: the cost of a disclosure failure (fine plus de-rating)

Suppose "Atlas Materials" is found to have materially overstated reserves in its filings for three years. The SEC settles for **\$200 million** (disgorgement plus penalty) and the CEO accepts a five-year officer-and-director bar. Atlas trades at \$25 on 400 million shares — a \$10 billion market cap — at a 14× multiple on \$1.79 EPS.

Tally the damage. The \$200 million cash fine is a direct hit: 200,000,000 / 400,000,000 = **\$0.50 per share**, a 2% mechanical decline. But the larger cost is the *de-rating*: a company caught lying in its filings loses the market's trust, so the multiple compresses from 14× to 11×. On \$1.79 EPS:

\$1.79 × 11 = \$19.69, then subtract the \$0.50 fine impact → roughly **\$19.19**, versus \$25.00.

That is a **−23%** move, of which only ~2 points are the fine itself and ~21 points are the credibility de-rating. The intuition: the headline fine is almost a rounding error next to the market's repricing of *everything the company says* once its disclosures are proven false. The punishment markets mete out for a disclosure failure dwarfs the punishment the SEC writes down.

## How to read the disclosure stream for an edge — the playbook

Everything above converges on a practical discipline. Disclosure is the market's information supply chain; an edge comes from reading it faster, deeper, or earlier than the consensus. Here is the playbook.

**Know the calendar and pre-position around scheduled events.** Earnings dates, the 10-K/10-Q deadlines, and the proxy/annual-meeting season are public. The information *will* arrive; the only question is your reaction function. Before a known release, decide in advance what each outcome means for your position and what would invalidate your view. Options implied volatility tells you what move the market is pricing — if you have a differentiated read on the *direction* of a disclosure, the scheduled event is your catalyst. (The event-trading series covers the vol-crush mechanics in depth.)

**Set up to ingest 8-Ks at the source.** The SEC's **EDGAR** system publishes filings the instant they are accepted, and the feed is free and machine-readable. The single most replicable disclosure edge is parsing 8-Ks faster than the market — especially the high-signal items: Item 1.01/1.02 (material agreements signed or terminated), Item 2.02 (results), Item 4.01/4.02 (auditor change / restatement), Item 5.02 (executive departure). A surprise in any of these is, by construction, a *material* event the issuer was legally required to disclose promptly.

**Read the filing, not the headline.** The recurring lesson of the worked examples: the headline number is the bait, the footnotes are the meal. Gross margin, guidance language, segment detail, related-party transactions, the wording of risk factors, and changes from the prior filing's language are where the real information hides. A "beat" with deteriorating quality is a short; a "miss" with improving guidance can be a buy. Diffing this quarter's filing against last quarter's — what language changed, what risk factor was added or dropped — is one of the highest-yield, lowest-tech edges available.

**Track Section 16 insider trades and 13F positioning.** Insiders must disclose their own buys and sells on Form 4 within two business days. Cluster buying by multiple insiders, especially after a selloff, is a meaningful signal — insiders rarely buy as a group unless they see value. Conversely, watch *who else* is in the trade via 13F filings and short-interest reports; crowded positioning is itself disclosed information that shapes how the next catalyst will reprice the stock (the GameStop lesson).

**Map the supply calendar, not just the news calendar.** The offering documents tell you the share count, the holders, and the lock-up terms. Lock-up expiries, secondary offerings, and large 13D/13G changes are *supply* events that move price independently of any business news, and they are all disclosed and calendar-able in advance. The Orbital Robotics example showed an eightfold float increase arriving on a known date. A complete disclosure read tracks both the *information* the company will release and the *supply* of stock the documents schedule.

**Weight historical facts above guidance.** The PSLRA safe harbor protects forward-looking statements; historical numbers carry near-strict liability. Practically, treat a stated past-quarter revenue figure as close to ground truth and treat guidance as the legally softest, most spin-prone part of a filing. When a story depends heavily on rosy guidance rather than demonstrated results, you are leaning on the part of the disclosure the law polices least — and management knows it.

**The catalysts that matter most:** an 8-K under a high-signal item; an S-1 made public (first look at a business); a restatement (Item 4.02); a Wells Notice or enforcement settlement; a guidance change buried in a 10-Q; cluster insider buying on Form 4; a short-interest reading at an extreme.

**What invalidates the view.** A disclosure edge evaporates when (a) the information is already priced — if the stock barely moves on a genuinely material 8-K, the market knew or expected it, and you are late; (b) your read of materiality is wrong — what looks material to you is immaterial to the reasonable-investor standard the market is applying; or (c) the timing window closes — the edge in reading a filing first lasts minutes, not days, so size and speed must match. The discipline is to act on the disclosure, then *immediately re-check whether the price already reflects it.* If it does, there is no trade, only a lesson about what the market already knew.

The deepest version of this skill is to stop seeing filings as compliance documents and start seeing them as the legally-mandated heartbeat of the market's information supply. The 1933 Act forces the truth out at the birth of a security; the 1934 Act forces it out for the rest of the security's life; Rule 10b-5 makes that truth trustworthy enough to price on. Every filing is a moment when private reality becomes public information, and every such moment is a chance for the price to be wrong for a few minutes before it is right. Reading the disclosure stream is, quite literally, reading the source code of price discovery.

## Further reading & cross-links

- [Who writes the rules: legislatures, regulators, central banks, courts](/blog/trading/law-and-geopolitics/who-writes-the-rules-legislatures-regulators-central-banks-courts) — where the SEC sits in the rule-making machine and how its rules get made and challenged.
- [Insider trading, Reg FD, and what is actually illegal](/blog/trading/law-and-geopolitics/insider-trading-reg-fd-and-what-is-actually-illegal) — the line between legally reading a filing fast and illegally trading on non-public information, and how Rule 10b-5 draws it.
- [Disclosure and accounting law: SOX, IFRS vs GAAP](/blog/trading/law-and-geopolitics/disclosure-and-accounting-law-sox-ifrs-vs-gaap) — Sarbanes-Oxley's CEO/CFO certification and why the *quality* of disclosure depends on the accounting rules behind it.
- [Crypto regulation: securities vs commodities and the turf war](/blog/trading/law-and-geopolitics/crypto-regulation-securities-vs-commodities-and-the-turf-war) — the Howey test applied to tokens, and what happens when "is it a security?" is genuinely unsettled.
- [Market-structure law: Reg NMS, PFOF, and short-selling rules](/blog/trading/law-and-geopolitics/market-structure-law-reg-nms-pfof-and-short-selling-rules) — the 1934 Act's other job: the plumbing of how trades actually clear and how short interest is disclosed.
- [How law moves markets: the transmission chain](/blog/trading/law-and-geopolitics/how-law-moves-markets-the-transmission-chain) — the series spine that places the disclosure regime in the full law-to-price pipeline.
- [Equity research](/blog/trading/equity-research) — how to turn the numbers a filing discloses into a valuation.
- [Event trading](/blog/trading/event-trading) — the reaction mechanics of trading scheduled and surprise information events.
