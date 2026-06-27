---
title: "Global Market Structure: Reg NMS, MiFID, and the Cross-Border Race for Capital"
date: "2026-06-21"
publishDate: "2026-06-21"
description: "How the US and EU wire their markets with opposite rulebooks, how IOSCO sets the global baseline, and how companies and capital shop jurisdictions for the best trust-versus-cost trade-off."
tags: ["capital-markets", "reg-nms", "mifid-ii", "market-structure", "iosco", "adr", "cross-border", "listings", "best-execution", "regulation"]
category: "trading"
subcategory: "Capital Markets"
author: "Hiep Tran"
featured: true
readTime: 42
---

> [!important]
> **TL;DR** — Every market-structure rulebook on Earth is trying to manufacture the same thing — enough *trust* that strangers will trade, which creates *liquidity*, which makes *issuance* possible — but the US and the EU reach for opposite default tools to do it.
>
> - **US Reg NMS (2005)** hard-wires a single best price: the Order Protection Rule bans "trading through" a better quote posted anywhere, anchored to the **NBBO** computed from a **consolidated tape**. It manufactured both fierce venue competition *and* the fragmentation, dark pools, and payment-for-order-flow debates we live with today.
> - **EU MiFID II (2018)** legislates *conduct and transparency* instead of a hard price rule: a **best-execution** duty (a process, not a single number), trading obligations, transparency waivers, systematic internalisers, and the famous **research-unbundling** rule.
> - **IOSCO** is the global standard-setter that pushes both toward a common baseline; the **PFMI** governs the market plumbing (CCPs, depositories) worldwide.
> - **ADRs/GDRs and dual listings** let capital and companies cross borders; exchanges in New York, London, and Hong Kong compete for listings, and firms route activity to the lightest acceptable regime — **regulatory arbitrage**.
> - The one number to remember: after Reg NMS, roughly **45% of US equity volume** trades *off* the lit exchanges it was designed to wire together.

## A tale of two rulebooks

On a single ordinary morning, the same share of the same global company can change hands under two completely different legal philosophies. Buy 100 shares of a US-listed stock through a US broker and a federal rule physically forbids your order from executing at \$50.05 if someone, anywhere in the national market system, is publicly offering it at \$50.00 — the machine *must* route you to the better price or it has broken the law. Buy the same economic exposure in Frankfurt or Paris, and no such single mandated price exists; instead your broker owes you a documented *duty* to have taken "all sufficient steps" to get you a good outcome, judged on process across price, cost, speed, and likelihood of execution.

Neither approach is obviously right. The American rule is mechanical, auditable, and brutal in its simplicity — and it accidentally spawned a dozen competing venues, dark pools, and the payment-for-order-flow industry. The European rule is principles-based, flexible, and harder to game on price alone — and it gave the world the research-unbundling earthquake that gutted sell-side analyst budgets overnight. Both are answers to the exact same question every market on Earth must answer: *how do you make strangers trust the price enough to trade size, so that companies can raise capital tomorrow?*

That question — and the wildly different tools jurisdictions reach for to answer it — is this post. We will take apart US Reg NMS and EU MiFID II side by side, meet IOSCO (the quiet global referee), follow a foreign company into New York via an ADR, watch exchanges fight over listings, and end on the uncomfortable truth that capital flows to wherever the trust-versus-cost trade-off is best — rules be damned.

![US Reg NMS versus EU MiFID II matrix comparing core mechanism best price transparency off-exchange and research](/imgs/blogs/global-market-structure-reg-nms-mifid-and-cross-border-1.png)

## Foundations: what "market structure" even means

Before the alphabet soup, the plain idea. **Market structure** is the set of rules and wiring that decides *where* a trade can happen, *what price* it must get, *who gets to see* the order, and *who reports* it afterward. It is the difference between a single village market square and a sprawl of competing stalls connected by a shared price board.

Three terms recur, so let us define them from zero:

- A **security** is a tradable claim — a share of ownership (equity) or a promise to repay with interest (a bond). The capital market exists to *create* these (the primary market, an IPO or bond sale) and to *trade* them afterward (the secondary market). For how a security gets *created and priced* at issuance, see the deal-mechanics posts; here we own the *trading-rules layer*.
- A **venue** is any place a trade can execute: a public "lit" **exchange** (NYSE, Nasdaq, the London Stock Exchange) that displays quotes, or an "off-exchange" venue (a dark pool, or a wholesaler/internaliser that fills your order from its own book). For the lit-versus-dark mechanics in depth, see [the fragmented tape](/blog/trading/capital-markets/lit-markets-dark-pools-and-the-fragmented-tape).
- **Best execution** is the obligation a broker has to get you a good deal — but, crucially, *what "good" means is defined differently in the US and the EU*, and that single difference cascades into two entire market structures.

Why does any of this need *rules* at all? Because of the series' spine: **regulation manufactures trust, trust creates liquidity, and liquidity makes issuance possible.** Nobody funds a 30-year project by buying a share unless they believe they can sell that share tomorrow morning at a fair, visible price to a stranger who trusts the same machine. Market-structure rules are the trust-manufacturing layer of the secondary market. Get them right and capital pools deeply and cheaply; get them wrong and issuers flee to a venue that got them right.

And the numbers at stake are enormous. The global capital markets are a roughly \$255 trillion stock of claims — equity plus bonds — split across jurisdictions that each wire their slice differently.

![Global capital markets by size showing global equity global bond US equity and US bond segments in trillions of dollars](/imgs/blogs/global-market-structure-reg-nms-mifid-and-cross-border-2.png)

The US alone is roughly \$55 trillion of equity and a similar bond market; how the US chooses to wire that \$55 trillion is the story of Reg NMS.

One more foundation, because it is the hinge of the whole post: the difference between **rules-based** and **principles-based** regulation. A rules-based regime writes down a bright-line, mechanically testable command — *do not execute at a price worse than the protected best quote* — and a machine or an auditor can check compliance with a stopwatch and a tape. A principles-based regime writes down an *objective* — *take all sufficient steps to obtain the best result for the client* — and judges compliance by whether your process was reasonable, documented, and defensible after the fact. The US leans rules-based (Reg NMS is the archetype); the EU's MiFID II leans principles-based (best execution is the archetype). Neither is strictly better: bright-line rules are gameable at the edges and create perverse side effects (you optimize the metric, not the outcome), while principles are flexible and broad but harder to enforce and slower to litigate. Almost every difference between US and EU market structure traces back to this one fork in regulatory philosophy.

## US Reg NMS (2005): wiring a fragmented market into one price

In the early 2000s the US equity market had a problem of plenty. Stocks traded on the NYSE, on Nasdaq, on regional exchanges, and on a growing swarm of electronic communication networks (ECNs). Prices on these venues could drift apart for the same stock at the same instant. An investor could get filled at a worse price on one venue while a better price sat untouched on another — a **trade-through**. That is exactly the kind of unfairness that *erodes trust* and shrinks liquidity.

The SEC's answer, **Regulation National Market System (Reg NMS)**, adopted 2005 and phased in through 2007, did not abolish the competing venues. Instead it *wired them together* into a single virtual market governed by four interlocking rules.

**1. The Order Protection Rule (Rule 611) — the trade-through ban.** This is the heart of Reg NMS. A venue may not execute a trade at a price inferior to a *protected quotation* — an automated, immediately accessible best bid or offer — displayed on another venue. In plain terms: if a better price is publicly showing somewhere in the national market system, the machine must honor it (or route your order to it). The trade-through ceases to be a thing that *can* happen to a marketable order.

**2. The NBBO (National Best Bid and Offer).** To enforce rule 611 you need a single, authoritative answer to "what *is* the best price right now?" That is the **NBBO**: the highest bid and lowest offer across all venues, recomputed continuously. The NBBO is the reference price the whole system snaps to.

**3. The SIP and the consolidated tape.** The NBBO has to come from somewhere. Every venue is required to report its quotes and trades to a **Securities Information Processor (SIP)**, which consolidates them into a single public feed — the **consolidated tape**. This is the shared price board that lets a fragmented market behave like one. (The SIP is also famously slower than the direct exchange feeds the fastest traders pay for — a tension we will return to.)

**4. The Access Fee Cap (Rule 610).** Venues compete partly by paying you to post liquidity (a rebate) and charging you to take it (an access fee). Reg NMS capped that take fee at \$0.30 per 100 shares (0.3¢/share) so that venues couldn't make their displayed quote a mirage by burying a huge fee behind it. This birthed the **maker-taker** pricing model and, eventually, the entire economics of where orders get routed.

![The Order Protection Rule in action showing a banned trade-through versus required Reg NMS routing to the best displayed quote](/imgs/blogs/global-market-structure-reg-nms-mifid-and-cross-border-3.png)

#### Worked example: a trade-through Reg NMS prevents

You send a marketable order to **buy 1,000 shares**. Venue A, where your broker has a relationship, is offering at **\$50.05**. But the NBBO offer — sitting on Venue B — is **\$50.00**. Without the Order Protection Rule, a lazy or conflicted router could simply fill you on Venue A:

- Fill at \$50.05 × 1,000 = **\$50,050**.
- Fill at the NBBO \$50.00 × 1,000 = **\$50,000**.
- The trade-through would have cost you **\$50** on a single small order.

Reg NMS makes that fill *illegal*: the system must route to Venue B's \$50.00 (or A must match it). Scale \$50 of protection across the roughly 10–12 billion shares that trade in US equities on a busy day and the rule is quietly redistributing serious money toward investors and away from venues that would otherwise pocket the spread. **The intuition: the trade-through ban turns a swarm of competing venues into one fair price board — at the cost of forcing everyone to watch, and trust, the same tape.**

There is a subtlety worth pausing on, because it explains a lot of modern US market structure. Rule 611 only protects *automated, immediately accessible, top-of-book* quotations. It does not protect manual quotes, and it does not protect *depth* — the second-best, third-best, and deeper prices sitting behind the top of each venue's book. So a large order that needs to "walk the book" can legally execute at progressively worse prices once it exhausts the protected top-of-book quote; protection is a thin film over the very best price, not a guarantee across the whole order. This single design detail is why so much institutional energy goes into *smart order routers* that slice a big parent order into hundreds of small child orders, each one small enough to rest at protected prices, drip-fed across venues to avoid signaling size. The rule protects the small marketable order beautifully and protects the large order barely at all — which is precisely why the large order hides in dark pools, and why the off-exchange slab grew the way it did.

A second subtlety: the access-fee cap created the **maker-taker** economy. A venue that wants to attract displayed liquidity pays a *rebate* (say 0.2¢/share) to whoever posts a resting quote (the "maker") and charges a *fee* (capped at 0.3¢/share) to whoever crosses the spread to take it (the "taker"). Brokers route to maximize their own rebate-versus-fee economics, which can pull an order toward a venue for reasons that have nothing to do with the client's fill quality. Reg NMS thus didn't just wire the price together — it created a parallel economy of rebates and fees that quietly shapes *where your order goes* underneath the price guarantee. Critics call this a conflict of interest baked into the plumbing; defenders call it the subsidy that keeps lit quotes tight. Either way, it is a direct, second-order consequence of capping the access fee instead of banning it.

### What Reg NMS created — competition *and* complexity

Here is the irony that defines US market structure. By guaranteeing that *any* venue offering the best price would receive the order, Reg NMS made it viable to launch a new venue with almost no incumbent advantage — just post a better quote and the rules deliver you flow. Competition exploded: dozens of exchanges and ECNs, then dark pools, then wholesalers. The same rule that *unified* the price *fragmented* the trading.

Today, roughly **45% of US equity volume executes off the lit exchanges** — in dark pools and through wholesalers who internalise retail orders.

![US equity volume by venue after Reg NMS showing lit exchanges versus off-exchange dark and wholesale share](/imgs/blogs/global-market-structure-reg-nms-mifid-and-cross-border-5.png)

That off-exchange slab is where **payment for order flow (PFOF)** lives: a retail broker routes your order to a wholesaler, the wholesaler fills you at or just inside the NBBO and pays the broker for the privilege. The Order Protection Rule guarantees you the NBBO — but the NBBO itself is built from the lit quotes, and if more and more volume hides off-lit, the price board everyone trusts gets thinner. That feedback loop — the rule that wired the market also hollowed out the very quotes it references — is the central, unresolved critique of Reg NMS, and the live subject of the SEC's ongoing market-structure reforms. We dig into the lit-versus-dark mechanics and PFOF in [the fragmented tape](/blog/trading/capital-markets/lit-markets-dark-pools-and-the-fragmented-tape).

## EU MiFID/MiFID II (2018): the same job, the opposite tools

Cross the Atlantic and the philosophy flips. The EU's **Markets in Financial Instruments Directive** — MiFID (2007), then the far more sweeping **MiFID II** (in force January 2018) — does *not* impose a hard, mechanical price-protection rule like rule 611. Europe never built a single mandated NBBO or a consolidated tape the way the US did. Instead, MiFID II manufactures trust through *conduct obligations* and *transparency*, policed after the fact.

**Best execution as a process, not a number.** A US broker can largely discharge its duty by pointing at the NBBO. A MiFID II firm must take "all sufficient steps" to obtain the best *result* for the client across price, cost, speed, likelihood of execution and settlement, size, and nature — and must publish reports on its execution venues and quality. Best execution in Europe is a documented *process you can be audited on*, not a single price the machine enforces.

**The trading obligation.** MiFID II pushes standardized instruments onto regulated venues (regulated markets, multilateral trading facilities, organized trading facilities) rather than letting them trade purely bilaterally — Europe's way of pulling activity into transparent light.

**Transparency, with waivers.** MiFID II demands pre- and post-trade transparency — but then grants **waivers** (large-in-scale, reference-price, negotiated-trade) that let big or sensitive orders avoid lighting up the screen. The European answer to "should big orders hide?" is *yes, but only under named, supervised exceptions* — versus the US answer of *dark pools, freely, as long as you report the print.*

**Systematic internalisers (SIs).** When a firm deals on its own account off-venue in a frequent, organized way, MiFID II makes it register as a **systematic internaliser** with its own quoting and transparency obligations. This is Europe's structured counterpart to the US wholesaler/internaliser — same economic function (filling client orders from your own book), but wrapped in a defined regulatory category rather than left to evolve.

**Research unbundling — the rule that shook the sell side.** Before MiFID II, banks bundled investment research into trading commissions: a fund "paid" for analyst reports by routing trades. MiFID II banned the bundle for European managers — research had to be *priced and paid for separately*, so clients could see exactly what they were buying. Budgets collapsed, coverage of small-caps thinned, and the analyst's business model was rewired overnight. (For what sell-side research *is* and why ratings exist, see [the analyst, the rating, and the wall](/blog/trading/capital-markets/sell-side-research-the-analyst-the-rating-and-the-wall).)

#### Worked example: MiFID II unbundling cuts a fund's research bill

A European equity fund runs **\$5 billion** and historically traded enough to generate **\$8 million/year** in commissions, of which the bank informally attributed perhaps **\$3 million** to "research" baked into the spread — invisible to the fund's own investors.

Post-MiFID II, the fund must pay for research from an explicit, disclosed budget. After reviewing what it actually reads, it sets a hard research budget of **\$1.2 million/year** and pays execution-only commissions on the rest:

- Old implicit research cost: ≈ **\$3,000,000/yr**.
- New explicit research budget: **\$1,200,000/yr**.
- Saving passed toward investors: ≈ **\$1,800,000/yr**, a **60% cut** in the research bill.

**The intuition: when you force a hidden cost into the daylight and make someone write a check for it, they buy a lot less of it — which is exactly why MiFID II unbundling shrank the research industry.**

The deep contrast: the US *engineers the price* and lets the conduct sort itself out; the EU *engineers the conduct and the disclosure* and lets the price form across venues. Same spine — trust → liquidity → issuance — opposite levers.

### Why Europe never built a consolidated tape

It is worth dwelling on the single biggest structural difference: the US has a consolidated tape and a mandated NBBO; the EU, for most of MiFID II's life, did not. After MiFID II fragmented European trading across dozens of venues in many countries and currencies, there was no single feed telling you the best price in, say, a French stock across all the venues it traded on. A buy-side trader in Europe had to stitch together venue feeds (often paid, often expensive) to approximate what a US trader gets for almost nothing from the SIP. The result: European equity market data became balkanized and costly, and the absence of a cheap consolidated view arguably *reduced* the very transparency MiFID II was trying to create. This is why the EU has been building a **consolidated tape provider** regime as part of its later reforms — an explicit admission that the US got the shared-price-board piece right, even as Europe still rejects the hard trade-through rule. The two regimes are slowly borrowing from each other: the EU adding a tape, the US debating whether its tape is fast and complete enough.

This convergence-by-borrowing is itself a lesson. There is no permanent winner between the mechanical and the principles-based approach; each regime keeps importing the other's good ideas as its own side effects bite. The US's price guarantee bred fragmentation, so it now studies Europe's venue-conduct rules; Europe's conduct regime bred opaque, costly data, so it now builds America's tape.

### Who pays, and who bears the risk, in each regime

Follow the money in both systems and you see who actually carries the cost of trust. Under Reg NMS, the *venues* bear the cost of connectivity and reporting to the SIP, *fast traders* pay for the direct feeds that beat the SIP, and *investors* receive trade-through protection roughly for free — but pay implicitly through a slightly thinner lit book as volume migrates off-exchange. Under MiFID II, the *firms* bear a heavy compliance cost (best-execution reporting, transaction reporting to regulators, SI obligations), *asset managers* now pay explicitly for research instead of hiding it in commissions, and *small-cap issuers* bear an indirect cost as research coverage thins and their cost of capital rises. In both cases the cost of manufacturing trust does not vanish — it is reallocated, and a careful reader can always ask "who is paying for this rule, and who is bearing the risk it leaves uncovered?"

## IOSCO and the global baseline

If every jurisdiction invents its own rulebook, how does a German pension fund trust a Brazilian exchange, or a US clearinghouse face a Japanese bank? Enter **IOSCO** — the International Organization of Securities Commissions, the global club of national market regulators (the SEC, the UK's FCA, the EU's ESMA, and ~130 others).

IOSCO does not *make* binding law. It is a **standard-setter**: it publishes principles — the IOSCO **Objectives and Principles of Securities Regulation** (protect investors; ensure fair, efficient, transparent markets; reduce systemic risk) — and members commit to implement them at home. Think of it as the shared grammar that lets 130 different national rulebooks remain mutually intelligible. Cross-border consistency lowers the cost of capital flowing between markets, because an investor in one jurisdiction can assume a baseline of disclosure and conduct in another.

The most consequential piece of IOSCO's work for *market structure* is the **PFMI — Principles for Financial Market Infrastructures** (IOSCO with the BIS's CPMI, 2012). The PFMI sets global standards for the *plumbing*: central counterparties (CCPs), central securities depositories, payment systems, and trade repositories — the institutions that clear and settle trades so that when you buy, you actually receive your shares and the seller actually gets paid. After 2008, when the failure of a single derivatives counterparty nearly cascaded through the system, the world agreed that the plumbing needed common, robust standards. For how CCPs and depositories actually work, see [stock exchanges and clearinghouses](/blog/trading/finance/stock-exchanges-and-clearinghouses).

IOSCO also runs the machinery that makes cross-border *enforcement* possible. Its **Multilateral Memorandum of Understanding (MMoU)** commits members to share information and assist each other's investigations — so that a fraudster in one country cannot simply hide behind a border. Before the MMoU, a regulator chasing manipulation that touched a foreign account often hit a wall; afterward, the SEC could ask the FCA (and ~120 others) for trading records and expect cooperation. This is unglamorous plumbing, but it is exactly what makes the *trust* in a security portable: an investor in one jurisdiction can believe that misconduct touching another jurisdiction will not vanish into a legal black hole.

The takeaway: above the national rulebooks sits a thin global layer whose entire job is to make the *trust* manufactured in one country portable to another — so capital can cross borders at all. Without it, every border would be a wall that capital could not safely cross, and the global \$255 trillion pool would shatter back into disconnected national puddles, each with a higher cost of capital because each would have to manufacture its trust alone.

## Listing regimes and cross-border access

Now the most visible cross-border choice: *where does a company list its shares, and how can foreigners buy them?*

A **primary listing** is a company's main home exchange — where its shares are principally regulated and traded. A **secondary listing** is an additional venue where the same shares trade under a lighter, recognition-based regime. A **dual listing** (in the strict sense) is two separate listed entities sharing one economic business (historically used in Anglo-Australian and Anglo-Dutch mergers), but colloquially people use "dual-listed" for any company listed in two places.

The cross-border workhorse is the **depositary receipt**. A US investor often cannot easily buy a stock listed only in Tokyo, São Paulo, or Mumbai — different currency, settlement system, and custody. A **depositary bank** solves this: it buys the foreign shares, holds them via a local **custodian**, and issues dollar-denominated receipts that trade on a US exchange or over-the-counter. In the US these are **ADRs (American Depositary Receipts)**; the global version traded on multiple markets is a **GDR (Global Depositary Receipt)**. Each ADR represents a fixed **ratio** of underlying shares (e.g. 1 ADR = 4 ordinary shares, or 1 ADR = ½ a share), chosen so the ADR's dollar price lands in a comfortable trading range.

![How an ADR brings a foreign stock to New York from a Tokyo-listed firm through a custodian and depositary bank to a US investor](/imgs/blogs/global-market-structure-reg-nms-mifid-and-cross-border-4.png)

#### Worked example: an ADR lets a US investor buy a foreign firm

A Japanese firm trades in Tokyo at **¥6,000** per ordinary share. A depositary bank sets up an ADR program with a ratio of **1 ADR = 5 ordinary shares**, and the exchange rate is **¥150 = \$1**.

- Underlying value per ADR = 5 × ¥6,000 = **¥30,000**.
- Convert: ¥30,000 ÷ 150 = **\$200 per ADR**.
- A US investor buys **50 ADRs** for 50 × \$200 = **\$10,000**, settling in dollars on a US exchange, with the depositary bank passing through dividends (converted to USD, net of a small fee) and the custodian in Tokyo holding the 250 underlying shares.

If the yen later weakens to ¥160 = \$1 with the Tokyo price unchanged, the ADR is worth ¥30,000 ÷ 160 = **\$187.50** — the US investor bears the currency move even though the local share price never budged. **The intuition: an ADR is a dollar wrapper around foreign shares held in custody at a fixed ratio — convenient access, but you inherit the FX risk for free.**

ADRs come in tiers that map directly onto how much US disclosure the foreign firm is willing to accept — another instance of the trust-versus-cost trade. A **Level I** ADR trades over-the-counter with minimal SEC reporting: cheap for the issuer, but it cannot raise new capital in the US and trades thinly. A **Level II** ADR lists on a US exchange (NYSE/Nasdaq) and must reconcile to US disclosure standards: more costly, but visible and liquid. A **Level III** ADR is the full package — a US-exchange listing *plus* a public capital raise, requiring the heaviest registration and disclosure. A foreign firm climbing from Level I to Level III is literally buying more US-investor trust with more disclosure, and getting deeper liquidity and the ability to issue in return. The depositary-receipt ladder is the cross-border listing decision in miniature.

Sponsored versus unsponsored matters too. A **sponsored** ADR is set up with the foreign company's cooperation (one depositary bank, dividends and voting passed through cleanly); an **unsponsored** ADR can be created by a depositary bank *without* the company's involvement, sometimes with several competing programs and messier shareholder rights. The receipt you buy is only as good as the program behind it — which is why the tier and sponsorship are part of due diligence, not trivia.

### The race for listings

Exchanges compete fiercely to attract listings, because listings bring trading fees, prestige, and an ecosystem of bankers, lawyers, and analysts. New York (NYSE + Nasdaq) offers the deepest pools of capital and the highest valuations for growth and tech — but also the strictest disclosure and the costs of Sarbanes-Oxley. London historically sold flexibility and a global investor base. Hong Kong sells itself as the gateway to Chinese capital. This competition is the **"race for listings,"** and it is where market-structure design meets corporate strategy.

![The race for listings showing IOSCO baseline feeding New York London and Hong Kong venues into a valuation versus cost trade-off and the issuer choice](/imgs/blogs/global-market-structure-reg-nms-mifid-and-cross-border-6.png)

The global IPO market that these venues fight over is itself volatile — it boomed to roughly \$459 billion in proceeds in 2021 and then collapsed by more than half.

![Global IPO proceeds by year showing the 2021 boom and the subsequent collapse in dollars billions](/imgs/blogs/global-market-structure-reg-nms-mifid-and-cross-border-7.png)

#### Worked example: a company choosing New York vs London

A profitable European software company expects roughly **\$200 million** of annual revenue and is choosing where to IPO.

- **New York:** comparable software firms trade at ~**8× revenue**, implying a ~**\$1.6 billion** valuation. But ongoing US compliance (SOX, US GAAP reconciliation, US litigation exposure, listing fees) runs perhaps **\$5 million/year**, plus a richer underwriting fee on a bigger raise.
- **London:** comparable listings trade at ~**5× revenue**, implying a ~**\$1.0 billion** valuation, with ongoing compliance closer to **\$2 million/year**.

The New York listing is worth **\$600 million more** in headline valuation but costs an extra **\$3 million/year** to maintain. For a company raising serious primary capital and whose founders care about valuation, the higher multiple swamps the running cost — capital flows to the venue with the deepest, most trusting buyer base. **The intuition: a listing decision is a trust-versus-cost trade — you pay more to access a pool of investors who will pay more, because they trust the disclosure regime more.**

This is the spine, made concrete at the level of a single company: deeper trust (stricter, more credible disclosure) → deeper liquidity → a higher price for the same cash flows → easier issuance. The race for listings is just hundreds of firms making that calculation, and exchanges tuning their rulebooks to win it.

## Regulatory arbitrage and cross-border fragmentation

The flip side of the race for listings is **regulatory arbitrage**: firms routing activity not to the *best* regime but to the *lightest acceptable* one. A trading desk might book derivatives through a subsidiary in a jurisdiction with looser margin rules; an issuer might choose a venue with thinner disclosure to avoid scrutiny; a fund might domicile where reporting is laxest. Each actor is doing exactly what the trust-versus-cost trade-off predicts — minimizing cost for a given level of market access.

This creates a permanent tension. National regulators want robust rules to protect their investors and contain systemic risk; global capital wants to flow to the cheapest compatible venue. Push the rules too hard in one place and activity migrates elsewhere — the regulator "wins" on paper but loses the market. Push too soft and you import risk and erode trust. The post-2008 reforms (mandatory CCP clearing, the PFMI, MiFID II's transparency push) were partly an attempt to *coordinate* enough that arbitrage couldn't simply route around safety — which is exactly why IOSCO's baseline matters.

Regulatory arbitrage is not always villainous, and that nuance matters. Some of it is healthy *regulatory competition*: jurisdictions experiment with lighter or smarter rules, the good experiments attract activity, and the bad ones lose it — a feedback loop that, over time, can push the whole system toward better rulebooks. London's post-Brexit attempts to streamline its listing rules, or Singapore's and Dubai's bids to attract fund domiciles, are competition as much as arbitrage. The danger is only when the migration is *toward genuinely under-priced risk* — derivatives booked where margin is too thin, leverage hidden where reporting is too weak — because that risk does not stay local. A counterparty failure in a lightly regulated corner can cascade back into the well-regulated core, which is the precise lesson of 2008. So the policy goal is not to eliminate jurisdictional choice (impossible, and partly beneficial) but to set a *floor* — via IOSCO and the PFMI — below which no venue can compete. Competition above the floor is healthy; a race below it is how systemic crises are seeded.

There is also a subtler, slower form of arbitrage: *listing* arbitrage, where the choice of home venue is itself a bet on which regulator will be friendliest to the company's structure over its life. A founder who wants outsized voting control will avoid an exchange that bans dual-class shares; a company in a sensitive industry will avoid a venue with aggressive disclosure of state ties. Each such choice nudges exchanges to adapt — which is exactly why Hong Kong rewrote its rules after losing Alibaba, and why the "race for listings" is less a one-time contest than a permanent negotiation between issuers and venues over the terms of trust.

The US equity market cap has roughly doubled over a decade, in part because deep, trusted, well-wired markets attract capital and listings from everywhere.

![US equity market cap over time from 2014 to 2024 in trillions of dollars](/imgs/blogs/global-market-structure-reg-nms-mifid-and-cross-border-8.png)

That gravitational pull — capital concentrating where trust is deepest — is the same force an emerging market must fight against when it tries to graduate. A market like Vietnam that wants foreign capital must first manufacture the trust (settlement reliability, foreign-ownership clarity, disclosure quality) that lets a global investor treat it as safe — the upgrade story we cover in [what Vietnam must fix](/blog/trading/capital-markets/the-emerging-market-upgrade-what-vietnam-must-fix).

## Common misconceptions

**"There's a single global stock-market rulebook."** There isn't, and there never has been. The US runs a hard-wired price-protection regime (Reg NMS); the EU runs a conduct-and-transparency regime (MiFID II); Asia's venues differ again. IOSCO supplies a *baseline*, not a binding world law. The same share trades under different philosophies depending on where you stand.

**"Reg NMS guarantees you the best possible price."** It guarantees you won't be *traded through* the best *displayed* price — the NBBO from the lit quotes. It does not guarantee price improvement, and as ~45% of volume migrated off-lit, the NBBO it references is built from a thinner and thinner slice of true liquidity. Protection against trade-throughs is not the same as the best economically achievable fill.

**"Europe's best-execution rule is weaker because it isn't mechanical."** Different, not weaker. A documented "all sufficient steps" process audited across price, cost, speed, and likelihood can catch abuses a single price check misses — and MiFID II's research-unbundling rule reshaped an entire industry in a way no US rule did. Principles-based regulation trades enforceability-by-machine for breadth-of-coverage.

**"An ADR is a different company's stock."** An ADR is a *receipt* for the very same shares, held in custody abroad at a fixed ratio. Economically you own the foreign firm — including its currency risk — through a dollar wrapper. It is access plumbing, not a separate security with separate fundamentals.

**"Companies list where the rules are lightest."** Sometimes (regulatory arbitrage is real), but more often they list where *valuations* are highest — which usually means where disclosure is *strictest*, because strict, credible disclosure is exactly what makes investors trust the price enough to pay up. The race for listings is frequently a race *toward* tougher rules, not away from them.

**"Fragmentation is just inefficiency to be eliminated."** Fragmentation has a genuine cost — thinner books per venue, harder price discovery, expensive data — but it is also the *output of competition* that Reg NMS deliberately enabled. Forcing everything back onto one monopoly exchange would re-concentrate pricing power and dull the incentive to innovate on speed, fees, and order types. The honest framing is a trade-off, not a defect: fragmentation buys competitive pressure at the price of complexity, and reasonable people disagree on whether the current 45%-off-lit split has tipped too far.

## How it shows up in real markets

**Reg NMS and the speed wars.** Because rule 611 references the SIP-computed NBBO but the fastest traders buy direct exchange feeds, a structural latency gap opened between the "public" price and the "real" price. This is the world Michael Lewis's *Flash Boys* dramatized and the SEC has spent years trying to narrow — a direct, decade-long consequence of one design choice in 2005.

**MiFID II's research cliff (2018).** Within a year of unbundling taking effect, surveys showed European asset managers slashing research budgets by double-digit percentages, banks cutting analyst headcount, and small- and mid-cap coverage thinning — fewer analysts means less visibility means a higher cost of capital for smaller listed firms. A conduct rule aimed at *transparency* rippled all the way out to *issuance* for small companies, the spine in action.

**The Alibaba listing saga.** Alibaba chose **New York** for its record ~\$25 billion IPO in 2014 after Hong Kong's rules at the time wouldn't accommodate its governance structure — a textbook race-for-listings loss for Hong Kong. HK later reformed its listing rules (weighted voting rights, 2018) explicitly to stop losing China's tech giants to New York, and Alibaba secondary-listed in Hong Kong in 2019. Exchanges *change their rulebooks* to win listings.

**The Saudi Aramco choice (2019).** The world's most valuable IPO ultimately listed primarily on the **Tadawul** (Riyadh) rather than New York or London, despite years of global courtship — a sovereign issuer weighing valuation, control, and political-disclosure cost and deciding the home venue's trust-cost trade-off won. Cross-border listing is as much politics and control as it is finance.

**Post-2008 CCP clearing under the PFMI.** The G20's mandate to clear standardized OTC derivatives through CCPs — implemented via the PFMI baseline and national rules (Dodd-Frank in the US, EMIR in the EU) — concentrated risk into a handful of clearinghouses worldwide. It is the clearest case of IOSCO-style coordination reshaping global market structure to contain the kind of counterparty cascade that nearly broke the system.

**The London-versus-Amsterdam share-trading shift (2021).** When the UK left the EU's single market, EU rules barred EU firms from trading EU shares on UK venues, and within days roughly €6 billion of daily euro-denominated share trading migrated from London to Amsterdam and Paris. Nobody changed the *companies* or the *prices* — only the *rules* about which venue was permissible — and an enormous, decades-old liquidity pool relocated almost overnight. It is the most vivid recent proof that market-structure rules, not fundamentals, decide where trading lives.

**The Didi delisting (2021–2022).** The Chinese ride-hailing firm raised ~\$4.4 billion in a New York IPO in mid-2021, then was forced by Beijing's data-security regulators to delist barely a year later, retreating from US markets entirely. It is the dark side of the race for listings: a company can win the deepest, highest-valuation venue and still be yanked out by its *home* regulator's cross-border priorities. Cross-border listing is never purely the issuer's choice.

## The takeaway: every regime answers the same question

Step back and the whole zoo of acronyms collapses into one idea. Reg NMS, MiFID II, IOSCO's principles, the PFMI, ADR programs, listing rules — they are all tools for manufacturing **trust** so that **liquidity** can form so that **issuance** is possible. The US reached for a hard, mechanical price guarantee and got fierce competition plus fragmentation as a side effect. The EU reached for conduct and transparency and got documented best execution plus the research-unbundling shock. Neither solved the problem permanently; both made trade-offs that show up as the texture of their markets today.

For a reader trying to understand the machine, three durable lessons:

1. **The rule shapes the market more than the market shapes the rule.** Reg NMS *created* the venue fragmentation it now struggles with; MiFID II *created* the research industry's collapse. Market structure is downstream of a handful of design choices.
2. **There is no free trust.** Strict disclosure costs issuers money but buys them higher valuations; light regimes are cheaper but command lower prices. Capital prices that trade-off continuously, and it flows to wherever the trade-off is best — which is why exchanges keep rewriting their rulebooks.
3. **Coordination is the only check on arbitrage.** Without a global baseline like IOSCO's, capital would simply route around every national rule. The thin layer of international standards is what keeps the manufactured trust *portable* across borders.

The deepest insight is the one the series keeps returning to: secondary-market liquidity is what makes primary issuance possible, and *market-structure rules are the machinery that turns trust into liquidity*. Get them right and you become the place the world brings its capital and its companies. Get them wrong and you watch both leave for a venue that did.

## Further reading & cross-links

- [Lit markets, dark pools, and the fragmented tape](/blog/trading/capital-markets/lit-markets-dark-pools-and-the-fragmented-tape) — the lit-versus-dark mechanics and PFOF that Reg NMS set loose.
- [Sell-side research: the analyst, the rating, and the wall](/blog/trading/capital-markets/sell-side-research-the-analyst-the-rating-and-the-wall) — what MiFID II unbundling reshaped.
- [Why markets are regulated: disclosure and the securities acts](/blog/trading/capital-markets/why-markets-are-regulated-disclosure-and-the-securities-acts) — the disclosure foundation under every regime here.
- [Market integrity: manipulation, spoofing, and circuit breakers](/blog/trading/capital-markets/market-integrity-manipulation-spoofing-and-circuit-breakers) — the conduct rules that complement structure.
- [The emerging-market upgrade: what Vietnam must fix](/blog/trading/capital-markets/the-emerging-market-upgrade-what-vietnam-must-fix) — manufacturing trust to attract cross-border capital.
- [Stock exchanges and clearinghouses](/blog/trading/finance/stock-exchanges-and-clearinghouses) — the venues and CCPs the PFMI governs.
