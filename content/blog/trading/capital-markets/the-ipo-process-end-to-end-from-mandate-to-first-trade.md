---
title: "The IPO Process, End to End: From Mandate to First Trade"
date: "2026-06-21"
publishDate: "2026-06-21"
description: "A full, step-by-step walkthrough of how a private company becomes a public one — the go-public decision, the bank bake-off, the S-1, SEC review, the roadshow, building the book, pricing night, allocation, the opening auction, and the lockup."
tags: ["capital-markets", "ipo", "primary-market", "underwriting", "bookbuilding", "equity-issuance", "investment-banking", "going-public", "stock-exchange", "prospectus"]
category: "trading"
subcategory: "Capital Markets"
author: "Hiep Tran"
featured: true
readTime: 42
---

> [!important]
> **TL;DR** — An IPO is a 4-to-6 month relay that converts a private company's accumulated value into a publicly tradable security, and the whole machine only works because a deep secondary market stands ready to buy what the primary market creates.
>
> - The sequence is fixed: decide to go public → bake-off and mandate → kickoff and due diligence → draft the S-1 → SEC review → red-herring and price range → roadshow → build the book → pricing night → allocation → listing day → lockup. Each stage de-risks the one after it.
> - The lead bookrunners get paid a **gross spread**, conventionally **7% on a US deal** — on a \$500M raise that is **\$35M**, the single biggest cost of going public.
> - The "IPO pop" is not free money: a stock priced at \$20 that opens at \$26 on 25M shares means **\$150M** of value transferred from the company to the day-one buyers — money "left on the table".
> - The one fact to remember: **the price you set on pricing night is a deliberate compromise between the issuer (wants it high) and the buyers (want a pop)** — and the bank, sitting in the middle, books both as wins.

On the morning of December 10, 2020, the floor of the New York Stock Exchange was unusually quiet for a blockbuster listing — the pandemic had thinned the crowd — but the order imbalance scrolling across the screens was anything but calm. DoorDash had priced its IPO the night before at \$102 a share, already above its raised range. When the stock finally opened for trading after a delayed opening auction, the first print was \$182. The company had sold roughly 33 million shares the night before at \$102. By the time the opening cross resolved at \$182, the market was telling everyone, loudly, that those shares had been worth \$80 more apiece than the company collected for them.

That gap — between the price the company sold at and the price the market immediately paid — is the most visible artifact of an IPO, and the most misunderstood. To a casual observer it looks like a triumph: "the stock popped 86%!" To the CFO who just watched roughly \$2.6 billion of value walk out the door to a handful of favored institutional accounts, it looks like something else entirely. Both readings are about the same number. Understanding why both are true, and who engineered it, is what this post is about.

An IPO — an *initial public offering* — is the headline event of the **primary market**, the engine of a capital market that *creates* securities to raise money. (Its sibling, the secondary market, *trades* those securities afterward; for the big picture see [what a capital market is and how money finds its best use](/blog/trading/capital-markets/what-is-a-capital-market-how-money-finds-its-best-use).) But "an IPO" is not one event. It is a four-to-six-month industrial process with a fixed running order, a named cast, a regulator looking over everyone's shoulder, and a final night where months of work collapse into a single number. We are going to walk that process from the boardroom decision all the way to the opening trade, and we are going to put dollar figures on the parts that matter.

![IPO process timeline from decide to go public through lockup](/imgs/blogs/the-ipo-process-end-to-end-from-mandate-to-first-trade-1.png)

## Foundations: what an IPO actually is, and why companies bother

Strip away the jargon and an IPO is a sale. A company that has so far been owned by a small group — founders, employees with stock options, venture capitalists, maybe a few private-equity funds — sells a slice of itself to the public for the first time, and in doing so lists its shares on a stock exchange where anyone can buy and sell them from then on.

Two things happen at once, and beginners constantly conflate them. First, the company can raise **new capital**: it prints brand-new shares and sells them, and the cash goes into the company's bank account to fund growth. This is the *primary* transaction — new securities created, money raised. Second, the company's existing shares become **liquid**: founders and early investors who held illiquid private stock can now (after a waiting period) sell into a public market at a quoted price. That liquidity is delivered by the *secondary* market — all the trading that happens after the IPO.

Here is the spine of this whole series, and it is worth saying slowly because the IPO is its cleanest illustration: **secondary-market liquidity is what makes primary issuance possible.** Nobody would buy a freshly minted share at the IPO if they could never sell it again. The promise of a deep, continuous secondary market — thousands of buyers and sellers every day, a tight bid-ask spread, the ability to exit a \$50M position in an afternoon — is precisely what lets the company sell that share in the first place. The primary market raises the money; the secondary market is the reason anyone shows up to fund it. (For how that trading venue itself works, see [stock exchanges and clearinghouses](/blog/trading/finance/stock-exchanges-and-clearinghouses).)

### Why go public at all — the trade-off

Going public is expensive, slow, and permanently changes how a company lives. So why do it? The honest answer is a trade-off with real costs on both sides.

On the **benefit** side:

- **Capital.** A primary IPO raises a large slug of money at once — often hundreds of millions or billions of dollars — at a lower cost than debt for a fast-growing, unprofitable company that banks won't lend to. This is the highest rung of the [financing ladder](/blog/trading/capital-markets/the-financing-ladder-from-bootstrap-to-public-markets).
- **Liquidity.** Employees and early investors get a path to convert paper wealth into cash. This matters enormously for hiring and for keeping venture investors happy after a 10-year hold.
- **Acquisition currency.** Public stock is a liquid, valued currency the company can use to buy other companies without spending cash. A private company's shares are nearly useless for M&A; a public company's are a checkbook.
- **Profile and discipline.** Being public raises a company's profile with customers and lenders, and the discipline of quarterly reporting can sharpen operations.

On the **cost** side, all of it is real:

- **Direct cost.** The bankers' gross spread alone runs to tens of millions of dollars; legal, accounting, printing, and exchange fees add several million more.
- **Disclosure.** A public company must file audited financials, risk factors, executive pay, and material events with the regulator — forever. Competitors read all of it.
- **Short-term pressure.** The market reprices the company every 90 days against analyst estimates. A single missed quarter can erase years of goodwill. Long-horizon decisions get harder when the stock reacts to the next three months.
- **Loss of control.** Founders dilute their ownership and answer to a board, activist investors, and a shareholder base they did not choose.

#### Worked example: is the gross spread worth it?

Suppose a company raises **\$500M** of new capital in its IPO and the lead banks charge the conventional US **gross spread of 7%**. The fee is `0.07 × \$500M = \$35M`. For that \$35M the company gets a syndicate that drafts the deal, navigates the SEC, organizes a roadshow, lines up institutional demand, sets a price, and commits to *buy any unsold shares itself* (more on that risk below). Is \$35M worth it? Against a \$500M raise it is 7 cents on the dollar — steep, but the alternative is doing it alone with no distribution and no price support. The intuition: the spread is the price of certainty — the bank converts a "we hope this sells" into a "this is sold", and certainty on a half-billion-dollar transaction is worth a lot.

The deep mechanics of *why* underwriters charge that spread and how they price the risk of being left holding unsold stock belong to a sibling post — [underwriting and the syndicate: who takes the risk](/blog/trading/capital-markets/underwriting-and-the-syndicate-who-takes-the-risk) — so we will name the cost here and link out for the derivation. Our job is the *process*: how the deal moves from a board decision to a first trade.

## The cast: who is in the room

Before the timeline, meet the working group. An IPO is run by a surprisingly large set of parties, each with a defined job and, crucially, different incentives.

![IPO working group showing issuer banks lawyers auditors exchange](/imgs/blogs/the-ipo-process-end-to-end-from-mandate-to-first-trade-3.png)

- **The issuer** — the company going public, represented day-to-day by its CFO, CEO, and general counsel. The issuer wants the highest price and the cleanest process. It is the customer.
- **The lead underwriters / bookrunners** — the investment banks that run the deal. The *lead-left* bookrunner (its name printed leftmost on the prospectus cover) is the quarterback: it builds the book, controls the syndicate, and usually runs the stabilization. On a big deal there may be several joint bookrunners. For how a bank earns its keep across all its businesses, see [inside an investment bank: how they make money](/blog/trading/finance/inside-an-investment-bank-how-they-make-money).
- **The syndicate** — the broader group of banks (co-managers and selling-group members) underneath the bookrunners. They lend their distribution (their own client lists) and share the underwriting risk and the spread. A wide syndicate means wider reach into different pockets of investor demand.
- **Issuer's counsel and underwriters' counsel** — two separate sets of securities lawyers. Issuer's counsel drafts and defends the company's disclosures; underwriters' counsel protects the banks and runs the due-diligence process. They negotiate every risk factor and every word of the prospectus.
- **The auditors** — an independent accounting firm that audits the financial statements and issues a "comfort letter" to the underwriters confirming the numbers in the prospectus tie out. No clean audit, no IPO.
- **The SEC** — in the US, the Securities and Exchange Commission, the disclosure regulator. It does not bless the company as a "good investment"; it reviews the registration statement to ensure the disclosure is complete and not misleading, and it must declare the registration *effective* before any shares can be sold.
- **The exchange** — NYSE or Nasdaq in the US. It lists the stock, runs the opening auction on listing day, and (on the NYSE) provides a Designated Market Maker who helps open the stock in an orderly way.

The reason to learn the cast first is that the *incentive conflicts* between them drive the entire process. The issuer wants a high price; the bank's institutional buy-side clients want a low price and a pop; the bank itself sits in the middle, paid by the issuer but dependent on keeping its buy-side clients happy for the next deal. Hold that tension in mind — it explains pricing night.

## Stage 1: deciding to go public

The decision is usually the slow part. A company typically spends one to two years getting "IPO-ready": cleaning up its accounting to GAAP standards, building the financial-reporting and internal-control machinery a public company needs, recruiting an independent board and an audit committee, and often hiring a CFO who has done this before. The board weighs the trade-off above and picks a rough window.

A key fork in the road: a company can go public to *raise primary capital* (sell new shares, money to the company), to give existing holders a *secondary* exit (sell old shares, money to the sellers), or both. Most IPOs are mostly primary, because a deal that's dominated by insiders cashing out sends a bad signal. The split is negotiated and disclosed.

#### Worked example: how much of the company gets sold?

A company valued at **\$5 billion** decides to raise **\$500M** of new primary capital and let early investors sell **\$100M** of existing stock, for a **\$600M** total deal. New shares to raise \$500M at, say, a \$20 price = `\$500M / \$20 = 25M new shares`. If the company had 225M shares before, it now has 250M, so the public float from new issuance is `25M / 250M ≈ 10%`. Add the \$100M of secondary shares (5M more shares changing hands, not newly created) and the day-one tradable float is `30M / 250M ≈ 12%` of the company. The intuition: most IPOs float only a slice — often 10–20% — so the early stock price is set by a relatively thin slice of supply meeting concentrated institutional demand, which is exactly why the price can be so jumpy.

There is also a quieter strategic question lurking in the go-public decision: *should we stay private longer?* For two decades the answer has trended toward "yes". The rise of deep late-stage private capital — growth funds, sovereign wealth, crossover investors who buy both private and public — means a company can now raise hundreds of millions privately without the disclosure and quarterly-pressure costs of being public. Many companies that would have IPO'd at a \$1bn valuation in 2000 now stay private until \$10bn or more. The cost of waiting is that early employees and seed investors stay illiquid longer; the benefit is that the company controls its narrative and avoids the public market's short-term reflexes until it is genuinely ready. The IPO has shifted from "the way a growing company raises money" to "a late-stage liquidity-and-currency event for a company that is already large" — a move up the [financing ladder](/blog/trading/capital-markets/the-financing-ladder-from-bootstrap-to-public-markets) that happens later in a company's life than it used to.

The timing of the *window*, finally, is its own discipline layered on top of all of this. Even a company that has decided it is ready cannot pick its date freely: IPOs cluster violently in good markets and vanish in bad ones, because the deal needs a receptive secondary market to land. A board can vote to go public in January and find the window slammed shut by March. We will see exactly how violently in a moment.

## Stage 2: the bake-off and the mandate

Once the board commits, the company runs a **bake-off** (also called a "beauty contest"): it invites several investment banks to pitch for the lead roles. Each bank sends a team that presents a thick deck arguing why it should run the deal. The pitches cluster around a few themes:

- **Valuation.** Each bank shows where it thinks the company will price. (Beware: banks sometimes "buy the mandate" by pitching an optimistic valuation, then walk it back later. Sophisticated issuers discount the highest number.)
- **Distribution.** Which investors can the bank reach? A bank with deep relationships among the long-only mutual funds that anchor IPO books is worth more than one that only knows hedge funds.
- **Research.** Which analyst will cover the stock after the IPO, and how respected is that analyst's voice in the sector? Post-IPO research coverage is part of the package.
- **Aftermarket support and league-table standing.** Has the bank's recent deals traded well? Banks live and die by their ranking in the IPO "league tables".

The issuer picks one or more bookrunners and assigns roles. The bank named *lead-left* runs the book and earns the largest economics. Others are joint bookrunners or co-managers. The chosen banks sign an **engagement letter**; the formal **underwriting agreement** (which fixes the spread and the firm-commitment terms) is not signed until pricing night, because nobody commits to a price until the book is built.

#### Worked example: how the \$35M spread is carved up

On our \$500M deal with a \$35M gross spread, the spread is conventionally split into three pieces: the **management fee** (~20%, to the bookrunners for running the deal), the **underwriting fee** (~20%, compensation for bearing the risk of unsold shares, split across the syndicate by commitment), and the **selling concession** (~60%, paid to whichever syndicate member actually placed each share with a buyer). So of the \$35M: management `≈ \$7M`, underwriting `≈ \$7M`, selling concession `≈ \$21M`. The intuition: most of the fee follows *distribution* — the bank that finds the buyer earns the most — which is why distribution reach dominates the bake-off pitch.

## Stage 3: kickoff and due diligence

With the banks mandated, the working group holds an **all-hands kickoff** meeting and sets a calendar that counts backward from a target pricing date. Then begins **due diligence**: the underwriters and their lawyers interrogate the company in detail, because under US securities law they have a "due diligence defense" — they are liable for material misstatements in the prospectus *unless* they conducted a reasonable investigation. Diligence is therefore not a formality; it is the banks protecting themselves from getting sued.

Diligence covers business diligence (management presentations, customer calls, market sizing), financial diligence (the auditors' comfort letter, quality-of-earnings review), and legal diligence (contracts, litigation, IP, regulatory exposure). Anything material that surfaces here must be disclosed in the prospectus. This is where a company's skeletons — a key-customer concentration, a pending lawsuit, an accounting weakness — get dragged into the light and written up as risk factors.

The reason diligence is so adversarial traces back to one feature of how a standard IPO is structured: it is a **firm-commitment** underwriting. In a firm commitment, the banks do not merely *broker* the sale of shares; they legally *buy* the entire offering from the company at the offer price (net of the spread) and then resell it to investors. That single fact rewires everyone's incentives. Because the banks own the shares for a financial instant — between buying from the issuer and reselling to the book — they bear the risk that the shares don't sell. If demand evaporates after they've signed, *they* are stuck with the inventory. That is why they conduct exhaustive diligence, why they don't sign the underwriting agreement until the book is built on pricing night, and why the gross spread is large: it is, in part, the price of warehousing risk on a half-billion-dollar block, even if only for an afternoon. (The contrast with a "best-efforts" deal, where the bank just tries its best and bears no inventory risk, is drawn out fully in the [underwriting post](/blog/trading/capital-markets/underwriting-and-the-syndicate-who-takes-the-risk).)

A second, subtler product of the kickoff is the **calendar itself**. The working group fixes a target pricing week, then works backward: roadshow start, red-herring print, expected SEC clearance, S-1 filing, and the diligence deadlines feeding it. This calendar is fragile. An SEC comment round that drags, a quarter-end that forces a fresh set of audited numbers into the filing, or a market wobble that closes the window — any of these can blow up the schedule and push pricing out by weeks or kill it outright. The single biggest source of slippage is the SEC review, which is why teams build slack around it.

## Stage 4: drafting the S-1 (the prospectus)

The central document of a US IPO is the **registration statement on Form S-1**, whose core is the **prospectus** — the offering document that tells investors everything material about the company. Drafting it is a multi-week grind of all-hands sessions where the working group argues over every sentence. The S-1 has a standard anatomy:

- **The prospectus summary and "the offering"** — the elevator pitch and the mechanics (how many shares, expected range, use of proceeds).
- **Risk factors** — pages of everything that could go wrong, written defensively by lawyers. A reader learns more from a careful read of the risk factors than from any pitch.
- **Use of proceeds** — what the company will do with the money.
- **MD&A** (Management's Discussion and Analysis) — management's narrative explanation of the financials and trends.
- **Business** — the detailed description of what the company does and its market.
- **Audited financial statements** — typically the last three years, the heart of the document.

The S-1 is filed with the SEC. Historically this filing was public the moment it hit; since the 2012 JOBS Act, an **emerging growth company** (one below roughly \$1.2bn in revenue) can file **confidentially** first and only make the document public a few weeks before the roadshow. This matters more than it sounds. Before confidential filing, a company that filed an S-1 exposed three years of financials, its margins, its customer concentration, and its strategy to every competitor on day one — and then, if the market turned and the deal got pulled, it had bared all of that for nothing. Confidential filing lets a company go through the entire SEC comment cycle in private and only commit to a public debut once the path is reasonably clear. The same regime also lets EGCs "test the waters" — hold pre-marketing conversations with qualified institutional investors before the public filing to gauge appetite. Together these reforms made the front half of the IPO far less risky to start, which is one reason the modern IPO pipeline can build up quietly and then release in a rush when a window opens.

We will return to the disclosure machinery, because it is the through-line of the entire capital-markets system — see also [the life of a security, from idea to delisting](/blog/trading/capital-markets/the-life-of-a-security-from-idea-to-delisting) for where the S-1 sits in a security's whole lifecycle.

## Stage 5: SEC review and comment letters

Once the S-1 is filed, the SEC's Division of Corporation Finance reviews it. The reviewer is not deciding whether the company is a good investment — that is a critical and constantly misunderstood point. The SEC operates a **disclosure-based** regime: its job is to make sure the company has told investors everything material and told it accurately, so that *investors* can decide. It is emphatically not a merit regulator that stamps deals "safe".

The review produces **comment letters** — written lists of questions and demands for clarification or additional disclosure. The company responds, often with a revised S-1 (an amendment), and the cycle repeats. A clean deal might go two or three rounds; a messy one many more. Comment letters and responses become public, so analysts pore over them for what the SEC pushed on. This back-and-forth typically takes one to two months and is the main source of schedule risk in the middle of the process.

A typical comment letter is more pointed than outsiders expect. The reviewer will ask the company to quantify a vaguely-worded claim ("you say you are the market leader — provide the basis"), to reconcile a non-GAAP metric the company is fond of with the audited GAAP figures, to expand a risk factor that reads as boilerplate, or to explain an unusual accounting treatment. None of this is a verdict on the business; all of it is the regulator forcing the disclosure to be specific, comparable, and honest. The discipline this imposes is one of the underrated benefits of going public: a company emerges from SEC review with financials and disclosures that have been stress-tested by a professional skeptic.

The key milestone at the end of this stage: the SEC declares the registration statement **effective**. Until that moment, *no shares can be sold*. Effectiveness usually happens on the evening of pricing, right before the deal prices — the working group will have a near-final S-1 ready, request acceleration of effectiveness, get it granted, sign the underwriting agreement, and price, all within the same few hours.

It is worth dwelling on what "disclosure-based regulation" really means, because it is the philosophical core of the entire US capital market and the through-line of this series. The alternative philosophy is **merit regulation**, where a regulator decides whether a security is good enough to be sold to the public — and some jurisdictions have historically used it. The US chose disclosure instead: the regulator's job is to guarantee that investors get the *information*, and the market's job is to set the *price*. The bet is that a well-informed market prices risk better than any committee of bureaucrats could. The IPO is where this bet is placed most visibly — the SEC clears the disclosure, then steps back and lets the roadshow and the book decide what the company is worth.

## Stage 6: the red herring and the price range

When the SEC is nearly done, the company prints the **preliminary prospectus** — universally called the **"red herring"** because of the red-ink legend on the cover warning that the registration is not yet effective and the securities cannot yet be sold. The red herring is the document the sales force hands to investors during the roadshow.

Crucially, the red herring carries a **price range** — for example, "\$18.00 to \$20.00 per share" — and a number of shares. This range is the banks' opening bid in the price negotiation. It is set after preliminary conversations with a few big "anchor" investors (sometimes via formal "testing the waters" meetings) so the bankers have a sense of where demand lives. The range is deliberately a *range*, not a point: it signals seriousness while leaving room for the book to push the final price up or down.

#### Worked example: turning a range into a deal size

Take a range of **\$18–\$20** and an offering of **25M shares**. At the bottom of the range the deal raises `25M × \$18 = \$450M`; at the top, `25M × \$20 = \$500M`; and if demand is strong enough to price *above* the range — say \$22 — it raises `25M × \$22 = \$550M`. So the price range alone spans a \$100M swing in proceeds before you even consider exercising the over-allotment option. The intuition: the range is where the issuer and the market agree to disagree, and the roadshow's whole purpose is to resolve that \$100M question in the issuer's favor.

## Stage 7: the roadshow and investor education

With the red herring in hand, management and the bankers hit the road — historically a literal two-week tour of New York, Boston, San Francisco, London, and other money centers; since 2020 often a hybrid of in-person and video meetings. This is the **roadshow**, and its purpose is **investor education**: get the story in front of the institutional investors — mutual funds, pension funds, sovereign wealth funds, hedge funds — who will anchor the book.

A roadshow day is a brutal schedule of back-to-back meetings: large group lunches, small group breakfasts, and the prized **one-on-ones** with the biggest potential buyers. Management gives the pitch; investors grill them on the numbers and the risks. Quiet-period rules constrain what can be said — everything must be consistent with the prospectus, and forward-looking promises are dangerous — so the art is in the delivery, not new information.

While management talks, the **sales force** at the banks works the phones, gauging interest and collecting **indications of interest (IOIs)** — non-binding signals of how many shares an account might want and at what price. Those IOIs are the raw material of the book. The roadshow is simultaneously a sales pitch and a price-discovery exercise: every conversation is also a data point about demand.

## Stage 8: building the book

As IOIs come in, the bookrunners assemble them into **the book** — the running ledger of who wants how many shares, at what price, and with what quality of demand. "Bookbuilding" is the heart of how an IPO price gets set, and the mechanics of it — limit vs market orders, price sensitivity, the bookrunner's allocation discretion — are deep enough that they get their own sibling post: [bookbuilding and price discovery: how the IPO price is set](/blog/trading/capital-markets/bookbuilding-and-price-discovery-how-the-ipo-price-is-set). Here we cover what the book *is* and how it feeds pricing night.

The book is not a simple auction. The bookrunner is reading two things at once:

- **Coverage** — how many times oversubscribed the book is. A book covered 10× at the top of the range (orders for 250M shares against 25M on offer) is a screaming-hot deal; a book that struggles to cover 1× is a troubled one.
- **Quality** — *who* is in the book and *how price-sensitive* they are. A book full of long-only mutual funds who say "I want 2 million shares and I'll pay up to \$22" is far better than a book of momentum hedge funds who want to flip on day one. The bookrunner wants holders who will still own the stock in six months, because they stabilize the aftermarket.

The book also reveals **price sensitivity**: which orders survive if the price moves up, and which fall away. That ladder of demand-versus-price is exactly what the bookrunner needs on pricing night.

## Stage 9: pricing night

This is the night everything collapses into a number. After the market closes on the day before listing, the working group gathers (these days, on a call) for the **pricing meeting**. The bookrunner presents the final book; the issuer and the banks negotiate the final offering price and the final deal size; the SEC declares the registration effective; the underwriting agreement is signed; and the shares are formally "priced."

The decision is whether to price **below**, **within**, or **above** the range — and that decision is driven almost entirely by the book.

![Pricing night decision weak book versus hot book](/imgs/blogs/the-ipo-process-end-to-end-from-mandate-to-first-trade-5.png)

- **Hot book (heavily oversubscribed).** If the deal is covered many times over with high-quality, price-insensitive demand, the bankers can price at the top of the range or above it and even increase the share count. The issuer raises more money. But — and this is the central tension — the bankers will almost always *deliberately leave a little on the table*, pricing slightly below where the book says the clearing price is, to engineer a first-day **pop**. A pop rewards the institutional buyers (who can sell into a higher price), reflects well on the bank's league-table standing, and is sold to the issuer as "a successful debut".
- **Within range.** A normally-covered book prices somewhere in the middle. Routine.
- **Weak book (undersubscribed).** If demand is thin, the bankers must cut the price below the range, cut the deal size, or — in the worst case — *pull the deal* entirely and try again later. A priced-then-broken IPO (one that trades below its offer price on day one) is a black eye for everyone.

The conflict here is structural and worth stating plainly: **the issuer is paying the bank, but the bank's repeat customers are the institutional buyers.** A bank that consistently prices deals too high — capturing every dollar for the issuer — will find its buy-side clients won't show up for the next deal. So the bank has a standing incentive to under-price just enough to keep the buy-side fed. The "pop" is the visible residue of that incentive.

#### Worked example: the pop, and money left on the table

Suppose the book clears comfortably at \$22 but the bankers price the deal at **\$20** to engineer a pop, on **25M shares**. The company collects `25M × \$20 = \$500M`. The next morning the stock opens at **\$26** and the market is clearly willing to pay it. The "money left on the table" is the gap between the first-trade price and the offer price, times the shares sold: `(\$26 − \$20) × 25M = \$150M`. That \$150M is value that transferred from the company's existing owners to the day-one buyers who got allocations at \$20. The intuition: a big pop is celebrated in the press as success, but to the issuer it is a \$150M discount it handed to the bank's favorite clients — which is why some founders are furious about pops, not delighted.

This is exactly the DoorDash situation from the opening: a \$102 price against a \$182 first trade is an enormous transfer. Whether that transfer is "the cost of a smooth debut" or "the bank shortchanging the issuer" is a genuine debate, and it is why a handful of companies (Google in 2004 most famously) have tried auction-style IPOs to capture the pop for themselves. Those remain rare; the conventional bookbuilt, banker-set price dominates, for reasons the [bookbuilding post](/blog/trading/capital-markets/bookbuilding-and-price-discovery-how-the-ipo-price-is-set) unpacks.

## Stage 10: allocation

Pricing sets the price; **allocation** decides who actually gets the shares. In a hot deal the book is many times oversubscribed, so the bookrunner must *ration* — and here the bookrunner's **discretion** is enormous and largely unregulated. Unlike a stock exchange's strict price-time priority, IPO allocation is a judgment call.

The bookrunner allocates to reward the holders it wants:

- **Long-only, high-conviction funds** that will hold the stock get generous allocations, because they stabilize the aftermarket.
- **Anchor investors** who committed early and large get priority.
- **Accounts that will flip** (sell immediately into the pop) get cut back, because flipping pressures the price.
- **Retail** historically got a small or zero allocation in traditional IPOs, though platforms and "directed share programs" have changed this at the margin.

Allocation is also where the bank's relationships get monetized — the favors it does for clients across all its deals get repaid here. This discretion has been abused historically (the "spinning" and "laddering" scandals of the dot-com era, where allocations were traded for favors), which is one reason allocation practices draw regulatory scrutiny. But the core principle stands: **allocation is a tool to engineer a stable shareholder base**, not a fair lottery.

#### Worked example: rationing an oversubscribed book

A deal offers **25M shares** and the book holds orders for **250M shares** — 10× oversubscribed. A long-only fund that ordered 5M shares and signaled it will hold might receive 2M (a 40% fill). A momentum account that ordered 5M and is known to flip might receive 500K (a 10% fill) or zero. The math: the bookrunner is allocating `25M` shares across `250M` of demand, so the *average* fill is `25M / 250M = 10%`, but the bookrunner skews fills toward holders and away from flippers. The intuition: in a hot IPO, *getting an allocation at all is the prize* — the 10× book is why retail rarely gets meaningful size, and why an allocation is effectively a gift of the expected pop.

## Stage 11: listing day and the first trade

The morning after pricing, the stock lists. Here is the subtlety beginners miss: **the IPO price is not a trading price.** The \$20 the company priced at last night was set by the bankers in a meeting. The first *trading* price is set by the exchange's **opening auction**, where all the buy and sell interest that has accumulated overnight gets matched at a single clearing price.

On Nasdaq this is a fully electronic opening cross. On the NYSE, a human **Designated Market Maker (DMM)** works with the floor to find the price at which the largest number of shares can trade, publishing indicative prices and order imbalances as buyers and sellers react — which is why NYSE IPOs sometimes take an hour or more to open. The first print is the number the press reports as "the IPO popped X%", and it is genuinely the first time the open market — not a banker — sets the price.

This is the cleanest demonstration of the series spine. The primary transaction (the company selling 25M shares at \$20) is *complete* before the stock ever trades. The secondary market — the opening auction and every trade after — exists to give those shares a continuously discoverable price and the liquidity to exit. Without the promise of that secondary market, the primary sale at \$20 would never have happened. (For the auction mechanics and order matching, the [order-book simulator](/blog/trading/quantitative-finance/order-book-simulator-quant-research) goes deep on how a clearing price forms.)

## Stage 12: the lockup, quiet periods, and stabilization

The deal isn't over when the stock opens. Three mechanisms govern the weeks and months after.

**The lockup.** Insiders — founders, employees, pre-IPO investors — agree contractually not to sell their shares for a set period, conventionally **180 days**, occasionally 90 or staggered. The lockup exists so the market isn't flooded with insider selling right after the IPO, which would crush the price and signal no confidence. But the lockup *expiry* is a known future event: on the day it lapses, a large block of previously-locked shares becomes sellable, and the stock often dips in anticipation — a **supply overhang**.

#### Worked example: the lockup-expiry supply overhang

A company has **250M** total shares but floated only **30M** (12%) at the IPO; the other **220M** are locked up for 180 days. When the lockup expires, the *potential* tradable supply jumps from 30M to 250M — more than **8×**. Even if only a fraction of insiders sell, the daily trading volume can't absorb a flood: if the stock trades, say, 3M shares a day and insiders want to sell 20M, that's nearly **7 days of average volume** hitting the market. The intuition: the overhang is why IPO stocks frequently sag in the days around their 180-day mark — savvy traders front-run the expiry, and the price discounts the coming supply before it even arrives.

**Quiet periods.** Securities law imposes "quiet" windows that limit what the company and the *underwriters' research analysts* can say. Most importantly, the banks' analysts cannot publish research recommendations until a quiet period (currently 10 days for IPOs under US rules, historically longer) after the offering — to keep the bank's sales pitch separate from its "independent" research. The first wave of analyst initiations after the quiet period lifts is itself a market event.

**Stabilization and the greenshoe.** This is the most elegant piece of the machine. The underwriting agreement gives the bookrunner a **greenshoe** (formally an *over-allotment option*), conventionally the right to sell up to an extra **15%** of shares. Here's the trick: the bookrunner *deliberately oversells* the deal — it sells 115% of the base shares, leaving itself **short** 15%. Then:

- If the stock trades **down** after listing, the bookrunner buys shares **in the open market** to cover its short. That buying supports the price (this is the "stabilization" the law explicitly permits), and the bookrunner covers its short cheaply.
- If the stock trades **up**, the bookrunner can't buy cheaply, so instead it **exercises the greenshoe** — it buys the extra 15% from the company at the IPO price to cover the short. The company sells more shares and raises more money.

Either way the bookrunner ends square, and either outcome is benign: a weak stock gets price support, a strong stock gets a bigger deal. It is a beautifully self-hedging structure.

![Greenshoe over-allotment option stabilisation mechanics](/imgs/blogs/the-ipo-process-end-to-end-from-mandate-to-first-trade-7.png)

#### Worked example: sizing the greenshoe

On a **25M-share** base deal at **\$20**, a 15% greenshoe is `0.15 × 25M = 3.75M` extra shares. The bookrunner sells `25M + 3.75M = 28.75M` shares to investors, collecting `28.75M × \$20 = \$575M`, but the company has only agreed to issue 25M (`\$500M`) so far — the bank is short 3.75M shares. If the stock sinks to \$18, the bank buys 3.75M in the market for `3.75M × \$18 = \$67.5M`, covering its `3.75M × \$20 = \$75M` short and pocketing the `\$7.5M` difference *while supporting the price*. If instead the stock rips to \$26, the bank exercises the shoe, buying 3.75M from the company at \$20 (`\$75M`), so the company raises the full `\$575M`. The intuition: the greenshoe makes the bank's stabilization *costless to itself* — it's short either way, and covering the short is exactly the act that steadies the stock.

## How the whole machine breathes: the IPO market is a window

Everything above describes one deal. Zoom out and the striking fact about IPOs is how brutally *cyclical* they are. The primary market does not run at a steady rate; it gorges in good years and starves in bad ones, because a deal needs a receptive secondary market to land. When stocks are rising and investors are hungry, the IPO "window" is open and deals pour out; when volatility spikes and prices fall, the window slams shut and even ready companies wait.

![US IPO proceeds by year 2014 to 2024 showing 2021 boom and 2022 freeze](/imgs/blogs/the-ipo-process-end-to-end-from-mandate-to-first-trade-2.png)

The chart above is the clearest picture of this in modern memory. US traditional-IPO proceeds ran in a \$30–\$85bn band for most of the 2010s, then exploded to **\$142bn in 2021** — the great post-pandemic, zero-rates, everything-rally boom. Then in 2022 the Fed started hiking, risk appetite evaporated, and IPO proceeds collapsed to **\$8bn** — a 94% drop, an effective *freeze*. Companies that had been planning 2022 IPOs simply pulled their deals and waited. The window had shut.

The deal *count* tells the same story even more starkly.

![Number of US IPOs by year 2014 to 2024](/imgs/blogs/the-ipo-process-end-to-end-from-mandate-to-first-trade-4.png)

The US went from **397 IPOs in 2021** to **71 in 2022** — fewer than one-fifth as many companies got out the door. This is the single most important practical fact about going public: **you do not fully control your own timing.** A company can be IPO-ready, with a clean S-1 and a great story, and still be unable to price a deal because the market won't have it. The window is a feature of the secondary market's mood, and the primary market is its hostage.

The freeze wasn't only American. Global IPO proceeds followed the same arc.

![Global IPO proceeds by year 2019 to 2024](/imgs/blogs/the-ipo-process-end-to-end-from-mandate-to-first-trade-8.png)

Worldwide proceeds peaked at **\$459bn in 2021** and then fell for three straight years to **\$121bn in 2024** as higher rates kept the window stubbornly narrow. The lesson for the series: the primary market is not an independent faucet you can turn on at will. It is downstream of the secondary market's appetite — which is, once again, the spine of this whole series.

## Alternatives to the traditional IPO

The bookbuilt, banker-led, firm-commitment IPO we have walked through is the dominant path to public markets, but it is not the only one. Three alternatives exist precisely because of the frustrations the standard process creates — chiefly the underpricing pop and the cost of the spread. Understanding why each is rare teaches you why the standard IPO survives.

**The auction IPO.** The most direct attack on underpricing is to skip the bankers' price-setting and run a true auction: collect bids, find the price that clears the offered shares, and sell to the highest bidders at that clearing price. Google did exactly this in 2004 with a modified Dutch auction, explicitly to capture the pop for itself rather than hand it to favored institutions. The result was instructive: the deal priced at \$85 (below the original range), the auction was messy, and the stock still rose afterward. Auctions remain rare because they sacrifice the things the bookbuilt process is good at — the bookrunner's ability to *curate* a quality, long-only shareholder base and to manage the aftermarket. An auction sells to whoever bids highest, including flippers, so the resulting shareholder register can be unstable. The market, it turns out, often *values* the banker's discretion enough to tolerate the pop.

**The direct listing.** Pioneered at scale by Spotify (2018) and Slack (2019), a direct listing lists existing shares on the exchange *without* a primary capital raise and *without* underwriters buying and reselling stock. There is no bookbuilding and no offer price; the stock simply opens via the exchange's auction at whatever price supply and demand produce. The appeal: no underpricing pop to give away, no spread on a primary raise (because there is no primary raise in the classic version), and no lockup mechanics in the same way. The catch: a direct listing historically raised *no new money* — it was a liquidity event for existing holders, suited to a company that was already cash-rich and famous enough not to need a roadshow. The SEC has since allowed primary capital raises in direct listings, narrowing the gap, but the format still suits only a small set of well-known, well-funded companies.

**The SPAC.** A special-purpose acquisition company is a shell with no operations that IPOs itself to raise a pool of cash, then goes hunting for a private company to merge with — taking that target public through the back door. SPACs exploded in 2020–2021 (they are the reason "ex-SPAC" qualifiers litter IPO statistics) precisely when the traditional window was hottest, because they offered a faster, lighter-disclosure path with a negotiated price rather than a market-set one. The boom ended badly: many SPAC mergers traded far below their \$10 reference price, redemptions soared, and regulators tightened the rules. The episode is a clean lesson in why the slow, adversarial, disclosure-heavy traditional IPO exists — the shortcuts that make a SPAC fast are the same shortcuts that let weaker companies reach public investors with thinner scrutiny.

#### Worked example: why a company tolerates the pop instead of an auction

Return to our \$20-priced deal that opened at \$26, leaving \$150M on the table. An auction might have captured most of that \$150M for the company. So why don't more issuers auction? Suppose the curated bookbuilt deal gives the company a *stable* register — holders who don't dump — and the stock holds \$26 and drifts to \$30 over the next quarter on a clean float, versus an auctioned deal that prices at \$25 but is full of flippers who sell into the open, pushing it to \$21 by week two. On the company's retained `225M` insider shares, a stock at \$30 vs \$21 is a `(\$30 − \$21) × 225M = \$2.025bn` difference in the value of what insiders still own — dwarfing the \$150M "saved" at pricing. The intuition: the pop is a visible, one-time cost; aftermarket stability is a larger, ongoing benefit, and that trade-off is why the banker-curated IPO survives despite the obvious giveaway.

## Common misconceptions

**"The IPO price is the market price."** No. The IPO price is set by bankers in a pricing meeting; the *market* price is set the next morning by the opening auction. The gap between them — the pop or the break — is precisely the thing the bankers were negotiating. On a hot deal the bankers deliberately price *below* the expected clearing price.

**"A big first-day pop means the IPO was a success."** It means the deal was *underpriced*. A 50% pop on a \$500M raise (priced at \$20, opens at \$30, on 25M shares) is `(\$30 − \$20) × 25M = \$250M` the company *didn't* collect. The press calls it success; the CFO may call it a quarter-billion-dollar gift to the bank's clients. The "right" pop is small and positive — enough to reward holders, not so big it screams mispricing.

**"The SEC approves the company as a good investment."** Flatly false. The SEC runs a disclosure regime — it checks that you've told investors everything material and told it accurately. It expresses *no view* on whether the stock is worth buying. "SEC effective" means "the disclosure is adequate", not "this is safe".

**"The underwriting fee is just a commission for selling shares."** It's that *plus* the price of risk transfer. In a firm-commitment IPO (the standard), the banks *buy* the shares from the company and resell them — so if the deal flops, the *banks* eat the unsold inventory, not the company. The 7% spread compensates for bearing that risk. (The firm-commitment-vs-best-efforts distinction is the subject of the [underwriting post](/blog/trading/capital-markets/underwriting-and-the-syndicate-who-takes-the-risk).)

**"Once you're public you can raise money whenever you want."** The window applies to follow-on offerings too, and a public company that misses estimates can see its stock — and its ability to raise — close just as hard as a private one waiting for an IPO window.

## How it shows up in real markets

**DoorDash, December 2020.** Priced at \$102 (above range), first trade \$182, an ~78% pop. On roughly 33M shares sold, the money left on the table was on the order of `(\$182 − \$102) × 33M ≈ \$2.6bn`. A textbook case of a hot-market deal where the engineered pop became a stampede — and a textbook example of the issuer-versus-buyside conflict at its most extreme.

**Airbnb, the next day (December 2020).** Priced at \$68, opened at \$146 — a 115% pop. Two enormous, deliberately-discounted deals in 48 hours, both in the white-hot late-2020 window. The clustering is not coincidence: when the window is open this wide, every banker rushes their deals through it before it closes.

**The 2021 → 2022 cliff.** As the charts show, US proceeds went from \$142bn to \$8bn and deal count from 397 to 71 in a single year, when the Fed pivoted to rate hikes. Dozens of companies that had confidentially filed S-1s in late 2021 quietly let them sit. This is the window mechanism in its purest form: the deals were ready; the market wasn't.

**The 2024 thaw — but only a thaw.** US proceeds recovered to ~\$30bn in 2024 and a handful of marquee names tested the water, but the count (~150) stayed far below 2021. A reminder that windows reopen gradually and selectively — the biggest, cleanest names go first, and the long tail of smaller companies waits for a fuller recovery.

#### Worked example: why the pop scales with the window

The median first-day pop swings with the market's mood. In a cold year it might be ~12%; in the hottest part of 2020 it ran above 40% on the median deal. On a \$500M raise (25M shares at \$20), a 12% pop is `0.12 × \$500M = \$60M` left on the table; a 42% pop is `0.42 × \$500M = \$210M`. Same company, same deal size — the *only* variable is how hot the window is. The intuition: bankers underprice *more* in hot markets because demand is so frothy they can't be sure where it clears, so they discount further to guarantee a clean debut — which is exactly why the worst pops (biggest giveaways) happen in the best markets.

That swing in the median pop is itself a measurable series.

![Median first day IPO pop by year US](/imgs/blogs/the-ipo-process-end-to-end-from-mandate-to-first-trade-6.png)

The median pop ran ~14–18% in normal years, spiked to **42% in 2020** and **32% in 2021** as the window blew wide open, then settled back to ~12–16% as conditions normalized. Read it alongside the proceeds chart and the picture is complete: the hotter the window, the more money companies leave on the table — the pop is the *price of issuing into euphoria*.

## The takeaway: an IPO is a negotiated handoff, not a coronation

The thing to carry away is that an IPO is not the moment a company "becomes valuable" — its value was built over years in private. The IPO is a carefully staged **handoff**: a private claim on that value gets converted into a public, tradable security, and the conversion is negotiated, every step, between parties with conflicting interests.

The issuer wants the highest price. The buy-side wants a discount and a pop. The bank, paid by the issuer but dependent on the buy-side, brokers a compromise — and the visible residue of that compromise is the first-day pop, a transfer of hundreds of millions of dollars from the company's old owners to the deal's favored buyers. None of this is hidden; it is the *design* of the bookbuilt IPO, and once you see the incentive triangle, every confusing feature — the deliberate underpricing, the discretionary allocation, the lockup, the greenshoe — snaps into place as a tool for managing that handoff.

And the deepest point is the series spine, which the IPO illustrates better than any other event: **the primary transaction is finished before the stock ever trades, yet it could never have happened without the secondary market that follows.** The company sold its shares at \$20 last night; the market sets \$26 this morning. The \$20 sale was only possible because everyone — issuer, banks, buyers — knew that \$26 market would be there. A capital market turns savings into long-term investment, and the IPO is the precise instant where a long-term, illiquid private claim becomes a liquid public one. The whole apparatus — bake-off, S-1, SEC review, roadshow, book, pricing night, allocation, opening auction, lockup, greenshoe — exists to make that one instant safe enough that buyers will fund it and sellers will release it.

When the window is open, the machine roars; when it shuts, even ready companies wait. That is not a flaw in the system. It is the system doing its job — refusing to let new long-term capital be created unless a deep, willing secondary market stands ready to keep it liquid tomorrow morning.

## Further reading and cross-links

- [The financing ladder: from bootstrap to public markets](/blog/trading/capital-markets/the-financing-ladder-from-bootstrap-to-public-markets) — where the IPO sits as the top rung of a company's funding journey.
- [The life of a security, from idea to delisting](/blog/trading/capital-markets/the-life-of-a-security-from-idea-to-delisting) — the full lifecycle map; the IPO is the "issuance and listing" chapter.
- [Underwriting and the syndicate: who takes the risk](/blog/trading/capital-markets/underwriting-and-the-syndicate-who-takes-the-risk) — firm-commitment vs best-efforts, the gross spread, and how banks price the risk of unsold shares.
- [Bookbuilding and price discovery: how the IPO price is set](/blog/trading/capital-markets/bookbuilding-and-price-discovery-how-the-ipo-price-is-set) — the deep mechanics of turning demand into a price, indications of interest, and allocation discretion.
- [Inside an investment bank: how they make money](/blog/trading/finance/inside-an-investment-bank-how-they-make-money) — the economics of the banks running your deal.
- [Stock exchanges and clearinghouses](/blog/trading/finance/stock-exchanges-and-clearinghouses) — the venue that lists the stock and runs the opening auction on day one.
- [Order-book simulator](/blog/trading/quantitative-finance/order-book-simulator-quant-research) — how a clearing price forms in the opening auction and every trade after.
