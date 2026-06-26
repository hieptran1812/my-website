---
title: "Inside an Investment Bank: ECM, DCM, M&A, and Trading"
date: "2026-06-21"
publishDate: "2026-06-21"
description: "A division-by-division product map of the sell-side: what each desk does, how it earns, and how it connects issuers to investors."
tags: ["capital-markets", "investment-bank", "ecm", "dcm", "mergers-acquisitions", "sales-and-trading", "market-making", "prime-brokerage", "league-tables", "sell-side"]
category: "trading"
subcategory: "Capital Markets"
author: "Hiep Tran"
featured: true
readTime: 40
---

> [!important]
> **TL;DR** — An investment bank is three businesses under one roof: a deal factory (IBD) that helps companies raise money and merge, a trading floor (Markets) that makes prices in the securities those deals create, and an asset-management/research wing that glues the first two to investors.
>
> - **IBD** earns event fees: an underwriting *spread* on a stock or bond it sells, and an advisory *fee* (roughly 1% of deal value) on M&A. No deal, no fee.
> - **Markets** earns a tiny *spread* on enormous *volume* — it buys at the bid, sells at the ask, and warehouses risk in between as a principal.
> - The two sides must be walled off: the deal team knows secrets the trading desk must never trade on. Compliance polices the wall.
> - The one number to remember: on a \$500M IPO at a 7% gross spread, the syndicate splits \$35M — and about 60% of it pays the salesforce that actually places the stock.

## The morning a deal becomes a price

At 9:29 a.m. on the day of a big IPO, two halves of the same bank are doing opposite jobs. Upstairs, in a glass-walled room on the *private side*, the equity-capital-markets bankers who spent six months preparing the company are watching the order book they built — a list of which institutions agreed to buy how many shares, and at what price. They set the offer at, say, \$27 a share last night. Their job is essentially over the moment the bell rings.

Downstairs, on the *public side*, a trading desk is about to start making a two-sided market in a stock that has never traded before. They will quote a price to buy and a price to sell, absorb the first frantic minutes of demand, and try not to lose money doing it. When the stock opens at \$34 — a 26% "pop" — the bankers upstairs and the traders downstairs will have completely different feelings about the same number. The bankers left \$7 a share on the table; the traders just had a wild, profitable morning.

There is a reason both teams care so intensely about that opening print. For the bankers, the offer price is the number they negotiated with the company and defended to investors for weeks; a huge pop means they mispriced it and the company financed itself too cheaply. For the traders, the first minutes of a new stock are pure, high-volume chaos — exactly the conditions in which a market-maker either earns a fortune on spread or gets run over by a one-sided flood of orders. One number, the opening price, simultaneously grades the work of the primary market and sets the stage for the secondary market. That is the bank in miniature.

That single morning contains the whole anatomy of an investment bank. One firm both *created* the security (primary market) and *traded* it (secondary market), and the only thing keeping those two activities from contaminating each other is a set of rules and a literal wall. This post is the division-by-division map of that firm: what each desk does, how it gets paid, and how — running underneath all of it — every desk is in the business of connecting people who need capital to people who have it. (For the high-level overview of how a bank makes money, see [Inside an Investment Bank: How They Make Money](/blog/trading/finance/inside-an-investment-bank-how-they-make-money); here we go desk by desk and fee pool by fee pool.)

![Investment bank product map showing three divisions and the desks under each](/imgs/blogs/inside-an-investment-bank-ecm-dcm-ma-and-trading-1.png)

## Foundations: what a sell-side bank actually is

Start with the everyday version. Suppose your neighbour wants to build an apartment block and needs \$10 million he doesn't have. Across town, a retired teacher has \$50,000 in savings earning nothing. Neither knows the other exists. The job of a capital market is to introduce them — to move the teacher's savings into the builder's project and give the teacher a claim (a share, a bond) she can sell to someone else tomorrow if she changes her mind. An investment bank is the professional matchmaker, dealmaker, and market-maker that makes that introduction at industrial scale.

The word "sell-side" is the key. In market jargon, the firms that *create and sell* securities and services — banks, brokers, dealers — are the **sell-side**. The firms that *buy and hold* them — pension funds, mutual funds, hedge funds, insurers — are the **buy-side**. (We dig into the buyers in [The Buy-Side: Who Actually Owns the Market](/blog/trading/capital-markets/the-buy-side-who-actually-owns-the-market).) An investment bank sits squarely on the sell-side: it manufactures securities for issuers and distributes them to the buy-side, then stands in the middle of the trading that follows.

A modern "bulge-bracket" investment bank — Goldman Sachs, Morgan Stanley, JPMorgan, and a handful of European and boutique rivals — is organised into three broad divisions:

1. **Investment Banking Division (IBD)** — the advisory and capital-raising business. This is where deals are born: IPOs, bond offerings, mergers. It earns *event-driven* fees.
2. **Markets, or Sales & Trading** — the secondary-market engine. It makes prices, provides liquidity, and finances clients. It earns *spread and volume*.
3. **Asset Management and other** (research, wealth management) — the wing that manages money for clients and produces the research that informs trading and deals. It earns *recurring fees*.

The crucial thing to hold in your head — the spine of this whole series — is that these three are not independent businesses that happen to share a logo. They are a single machine for turning savings into long-term investment. IBD *creates* the securities; Markets makes them *liquid*; research and distribution connect both to the savers. And the secret that makes it all hang together is liquidity: nobody buys a freshly minted 30-year bond or a newly listed stock unless they believe they can sell it tomorrow morning. The trading floor's willingness to make a market is what lets the deal floor sell the deal. Secondary-market liquidity is what makes primary issuance possible.

### Primary and secondary, and why a bank straddles both

Two more terms close the loop, because they are the geography the whole bank is built on. The **primary market** is where securities are *created* — a company sells shares it never sold before (an IPO), or a government auctions a brand-new bond. Money flows from investors to the issuer; a new claim comes into existence. The **secondary market** is where those already-existing securities *change hands* between investors — the stock exchange, the bond dealer network. No new money reaches the issuer in the secondary market; ownership simply rotates.

A retail saver almost only ever touches the secondary market: when you buy a share through a brokerage app, you are buying it from another investor, not from the company. The company got its money years ago, at the IPO. So why does the secondary market matter to the issuer at all, if it sees none of that money? Because the *price* and *liquidity* of the secondary market are what set the terms of the *next* primary deal. A company whose shares trade actively at a high multiple can raise more equity, more cheaply, in a follow-on. A government whose bonds trade in a deep, liquid market borrows at a lower yield. The secondary market is the issuer's permanent credit reference.

An investment bank is the rare institution that operates on *both* sides of that line at once. IBD works the primary market — manufacturing the security. Markets works the secondary — trading it forever after. The bank is the hinge between creation and circulation. That dual role is exactly what generates both its power and its conflicts: the same firm that priced the IPO last night is quoting the stock this morning, and it must not let what it knows on one side leak to the other.

Let's walk each division and meet each desk.

## IBD: the deal factory and its three product lines

The Investment Banking Division is the part most people picture when they hear "investment banker": pitch decks, all-nighters, and very large fees attached to discrete events. IBD has two jobs — *raising capital* for clients and *advising* them on transactions — split across three product lines: **ECM**, **DCM**, and **M&A**.

The unifying logic: IBD does not earn a spread on daily trading. It earns a fee when something *happens* — a company goes public, issues a bond, or buys a competitor. The fee is large but lumpy, which is why IBD revenue swings violently with the deal cycle.

A useful everyday parallel: IBD is the part of the bank that behaves like a real-estate agent or a wedding planner — it is paid a large one-time fee for orchestrating a complex, infrequent, high-stakes event. The trading floor, by contrast, behaves like a currency-exchange kiosk at an airport — it makes a tiny margin on each of thousands of transactions a day. Same building, opposite business models. The IBD banker's year is a handful of deals that each pay enormously; the trader's year is millions of small spreads that each pay almost nothing. Understanding that one earns on *events* and the other on *flow* explains almost everything else about how the two sides behave, hire, and get paid.

### ECM — Equity Capital Markets

ECM raises money for companies by selling *ownership* — equity. Its flagship product is the **IPO** (initial public offering), the first sale of a company's shares to the public. ECM also runs **follow-on offerings** (a public company selling more shares after it is already listed), **rights issues**, and **convertible bonds** (debt that can turn into equity). We cover the deal mechanics in depth in [The IPO Process, End to End](/blog/trading/capital-markets/the-ipo-process-end-to-end-from-mandate-to-first-trade); here the point is how the desk earns.

It is worth noting that the IPO is only the *first* equity event in a public company's life, and often not the most frequent one for the bank. After listing, the same ECM desk runs **follow-on offerings** (selling more shares to fund growth), **block trades** (helping an early investor sell a large stake quickly and discreetly), and **convertible** issuance. These post-IPO products are usually faster, lower-risk, and lower-fee than the IPO, but they recur — a successful listing becomes an ongoing ECM relationship. The full menu of post-IPO equity raising is covered in [Beyond the IPO: Follow-Ons, Rights Issues, and Private Placements](/blog/trading/capital-markets/beyond-the-ipo-follow-ons-rights-issues-and-private-placements). The IPO is the headline; the follow-on flow is the annuity.

ECM gets paid an **underwriting spread** (also called the *gross spread*): the difference between the price the bank pays the company for the shares and the price at which it sells them to investors. For a US IPO the convention is famously sticky — roughly **7%** of the money raised. The bank guarantees the company a price, takes the shares onto its own book, and resells them; the 7% compensates it for the risk, the distribution effort, and the advice.

That 7% is split three ways, and understanding the split is the key to understanding why banks form *syndicates*. The gross spread divides into a **management fee** (for running the deal, ~20%), an **underwriting fee** (for bearing the risk and covering costs, ~20%), and a **selling concession** (for actually placing the shares with investors, ~60%). The largest slice goes to whoever sells the stock — which is why a deal is shared among many banks, each bringing its own roster of buyers.

![How a 7% IPO gross spread splits into management, underwriting, and selling fees](/imgs/blogs/inside-an-investment-bank-ecm-dcm-ma-and-trading-3.png)

#### Worked example: the 7% spread on a \$500M IPO

A company raises \$500M in an IPO at a 7% gross spread.

- Gross spread = 7% × \$500M = **\$35M** total to the syndicate.
- Management fee (~20%) = **\$7M**, mostly to the lead bookrunner who ran the process.
- Underwriting fee (~20%) = **\$7M**, split by underwritten allocation, paying for risk and expenses.
- Selling concession (~60%) = **\$21M**, paid to whichever banks' salesforces actually placed the shares.

If the lead bookrunner ran the books *and* placed 40% of the stock, it might keep \$7M (management) + a share of the \$7M underwriting + \$8.4M (40% of \$21M) ≈ **\$18M** of the \$35M. The intuition: the fee follows the work, and the biggest reward goes to the bank that can find the buyers — distribution, not advice, is what the spread mostly pays for.

There is a subtle but important detail about the 7% spread: the bank is also choosing *how much risk* to take in the deal structure. In a **firm-commitment** underwriting — the standard for a large US IPO — the bank actually buys all the shares from the company at the agreed price and resells them, so if demand evaporates overnight the bank eats the unsold inventory. That is real principal risk, and it is why the underwriting fee slice exists. In a **best-efforts** deal — more common for small or risky issuers — the bank only promises to *try* to sell the shares and takes no inventory risk; the issuer bears the shortfall. The 7% convention assumes firm commitment. The deeper mechanics of who shoulders that risk and how the syndicate spreads it live in [Underwriting and the Syndicate: Who Takes the Risk](/blog/trading/capital-markets/underwriting-and-the-syndicate-who-takes-the-risk).

There is also the matter of the **greenshoe**, or over-allotment option, which lets the syndicate sell up to ~15% more shares than the base deal and either buy them back in the market (supporting a weak open) or exercise the option (capturing the extra in a strong one). It is a stabilisation tool baked into nearly every IPO, and it quietly adds to or subtracts from the spread the bank ultimately earns. Price discovery itself — how the \$27 in our opening got set — is its own craft, covered in [Bookbuilding and Price Discovery](/blog/trading/capital-markets/bookbuilding-and-price-discovery-how-the-ipo-price-is-set).

The ECM fee pool rises and falls with the IPO window. The chart below — US IPO proceeds by year — shows why ECM bankers' bonuses are so volatile: 2021 was a \$142bn feast; 2022 collapsed to \$8bn when rates spiked and the window slammed shut. A 7% slice of \$142bn is roughly \$10bn of fees in a good year and a fraction of that in a bad one. The window is not a metaphor: it is the set of weeks when valuations, volatility, and investor appetite all line up to let a company list at a price it will accept. When rates spike or markets wobble, the window shuts, deals get pulled mid-roadshow, and an entire desk's pipeline freezes at once.

![US IPO proceeds by year from 2014 to 2024 showing the 2021 peak and 2022 collapse](/imgs/blogs/inside-an-investment-bank-ecm-dcm-ma-and-trading-2.png)

### DCM — Debt Capital Markets

DCM raises money for companies and governments by selling *debt* — bonds. A bond is a promise to pay back a fixed sum on a fixed date with interest in between; selling one is borrowing. DCM helps an issuer decide how much to borrow, for how long, at what coupon, and then sells the bonds to investors. (How a bond deal is built and syndicated is its own post: [Underwriting and the Syndicate: Who Takes the Risk](/blog/trading/capital-markets/underwriting-and-the-syndicate-who-takes-the-risk); the *pricing* of bonds — duration, the yield curve — lives in fixed-income, e.g. [The Yield Curve Explained](/blog/trading/fixed-income/the-yield-curve-explained-the-most-important-chart-in-finance).)

DCM earns a fee too, but it is far thinner than ECM's 7% — typically **0.3% to 0.9%** of the amount raised for a corporate bond, and a few basis points for a government deal. Why so much thinner? Because a bond is a far easier product to sell than an equity stake. Its payoff is contractual and ratable; the price is anchored to a known yield curve; the buyers are repeat institutional investors who model credit for a living. There is less to advise, less to discover, and less risk in placing it — so the fee compresses.

But DCM makes up in *volume* what it lacks in margin. The debt markets dwarf the equity markets in issuance. The chart below shows 2023 US bond issuance by type (with the enormous Treasury number excluded so the corporate and mortgage bars are readable). Corporate issuance alone — the bread and butter of a DCM desk — runs well over a trillion dollars a year. A 0.5% fee on \$1.4tn of corporate issuance is a \$7bn fee pool, on par with ECM in a good year and far steadier across the cycle.

![US bond issuance by type in 2023 excluding Treasury showing corporate and mortgage bars](/imgs/blogs/inside-an-investment-bank-ecm-dcm-ma-and-trading-4.png)

#### Worked example: the DCM fee on a \$1B bond

A blue-chip company issues a \$1B 10-year bond. DCM charges a 0.40% fee.

- Fee = 0.40% × \$1,000,000,000 = **\$4,000,000**.
- The deal might price in a single afternoon, with the desk lining up demand from pension funds and insurers in the morning.
- Compare ECM: 7% × \$1B would be \$70M — about 17× more for the same headline size.

The intuition: a bond is cheap to underwrite because it is easy to sell — the issuer is borrowing against a contract, not selling a slice of an uncertain future. The DCM desk wins by doing many of these, fast, for repeat issuers.

A few things make DCM a fundamentally different business from ECM even though both raise capital. First, **issuers are repeat customers**: a large company taps the bond market several times a year, refinancing maturing debt and funding new spending, so the DCM banker is managing an ongoing relationship rather than a once-in-a-lifetime IPO. Second, the product is **ratable** — a credit-rating agency assigns a letter grade (AAA down to junk) that lets investors price the bond off the yield curve plus a credit spread, so there is far less to "discover." Third, DCM deals can move at extraordinary speed: a well-known issuer can announce a benchmark bond in the morning, build a book of orders by lunch, and price it the same afternoon, because the buyers are standing institutions who already know the name. The desk's skill is reading the market window — picking the day and the maturity when demand is deepest and the spread to Treasuries is tightest. The full deal choreography — auction versus syndication, the order book, allocation — is in [How a Bond Is Issued](/blog/trading/capital-markets/how-a-bond-is-issued-auctions-syndication-and-the-deal); the *pricing* of the resulting instrument stays in fixed-income, where it belongs.

DCM also quietly does some of the most consequential advisory in the bank, even though it is wrapped in a thin fee. A company's choice of *how much* debt to carry, at *what maturities*, and in *which currencies* shapes its survival in a downturn — too much short-term debt and a refinancing market that freezes can sink an otherwise healthy firm. The DCM banker advises on that capital structure continuously, not just on issuance day, which is why the relationship is sticky and why the desk's real product is judgment about the debt markets rather than the mechanical act of selling a bond. The choice between raising equity or debt in the first place — the most basic capital-structure decision — is its own topic in [Debt vs Equity: The Two Ways to Raise Capital](/blog/trading/capital-markets/debt-vs-equity-the-two-ways-to-raise-capital).

The relationship between DCM and Treasury issuance is worth pausing on. The chart above excludes Treasuries precisely because they are so enormous they would flatten everything else — the US government issues many trillions a year. But government bonds are mostly sold by *auction*, not by an underwriting syndicate, so banks earn far thinner fees on them (a few basis points, as primary dealers) than on corporate deals. The fat-margin DCM business is corporate and structured debt, where the bank's distribution and advice actually add value.

### M&A — Mergers & Acquisitions advisory

M&A is pure advice: the bank counsels a company that is buying, selling, or merging, and earns a **success fee** — typically around **1% of the deal value**, with the percentage shrinking on mega-deals and rising on small ones. Crucially, M&A involves no underwriting risk and no balance sheet; the bank sells *judgment* — valuation, negotiation, structuring, and the credibility of a fairness opinion. It is the highest-margin business in IBD because it consumes almost no capital.

M&A is where the famous secrecy lives. The advisory team knows a takeover is coming before the world does; that knowledge is **material non-public information**, and trading on it is insider trading. This is the single biggest reason banks erect walls between IBD and the trading floor — more on that below.

#### Worked example: the advisory fee on a \$10B merger

A bank advises the seller in a \$10B acquisition at a 1% fee.

- Advisory fee = 1% × \$10,000,000,000 = **\$100,000,000**.
- On a mega-deal the percentage usually slides — a \$10B deal might pay closer to 0.6–0.8%, so call it \$60M–\$80M; on a \$200M deal the fee could be 1.5–2%.
- The bank deploys *no capital*: it does not buy the company or guarantee a price. It is paid for advice and execution.

The intuition: M&A is the closest a bank gets to selling pure expertise. The fee is a fraction of a number so large that even a fraction is enormous — and because no balance sheet is at risk, almost all of it is profit.

M&A advisory is also where the most senior bankers spend their careers, because the product being sold is *trust*. A board deciding to sell a company it spent decades building is making the biggest decision of its corporate life, and it pays for a banker whose judgment and relationships it believes in. That is why M&A is a "people business" in a way trading is not: the fee follows the senior banker who owns the client relationship, not a desk or an algorithm. It is also why **fairness opinions** — the bank's formal written judgment that a deal price is fair to shareholders — carry weight and legal liability: the board uses them as cover, and the bank's reputation is the collateral. Two flavors recur: **sell-side advisory** (running an auction to get the best price for a company being sold) and **buy-side advisory** (helping an acquirer identify, value, and negotiate a target). Often a single mega-merger employs banks on both sides, plus financing banks arranging the debt to pay for it — so one transaction can generate fees across ECM, DCM, *and* M&A at once. The arithmetic of a leveraged acquisition is where the bank's product lines visibly converge.

## Markets: the secondary-market engine

Cross to the other side of the building and the rhythm changes completely. Where IBD lives in months-long deal cycles, the **Markets** division — Sales & Trading — lives in milliseconds. This is the secondary-market engine: the desks that make prices in securities after they have been issued, providing the liquidity that, as we keep insisting, is what made the primary issuance possible in the first place.

The core activity is **market-making**. A market-maker continuously quotes two prices: a **bid** (what it will pay to buy) and an **ask** or offer (what it will charge to sell). The gap between them is the **bid-ask spread**, and capturing that spread, over and over, is how the desk earns. The desk is a *principal* — it trades for its own account, taking securities onto its own balance sheet — rather than a pure *agent* that merely routes a client's order to an exchange. (The agency-vs-principal distinction is its own subject: [The Broker-Dealer: Agency, Principal, and Prime Brokerage](/blog/trading/capital-markets/the-broker-dealer-agency-principal-and-prime-brokerage).)

Here is the mechanism. A client wants to sell a \$50M block of a bond *now*. There may be no buyer at this instant. The desk buys it anyway — at a price slightly *below* the mid-market — and holds it as inventory, bearing the risk that the price moves against it before it finds a buyer. Later it sells to another client slightly *above* the mid. The difference is the spread, and the willingness to stand in the gap and warehouse risk is **facilitation**, or **principal risk-taking**. The client gets immediacy; the desk gets paid for providing it and for the risk it ran.

![How a trading desk earns the spread by buying at the bid and selling at the ask](/imgs/blogs/inside-an-investment-bank-ecm-dcm-ma-and-trading-5.png)

#### Worked example: spread capture on a \$50M facilitation

A pension fund needs to sell a \$50M block of a corporate bond. Mid-market is \$100.00 per \$100 face.

- The desk buys the block at the **bid**, \$99.90 — paying the fund \$49,950,000.
- It warehouses the bonds, then sells to an insurer at the **ask**, \$100.10 — receiving \$50,050,000.
- Gross spread captured = \$0.20 per \$100 × \$50M / \$100 = **\$100,000**.
- If the market moves against the desk while it holds inventory, that \$100,000 can shrink or flip to a loss — that is the principal risk.

The intuition: the desk is paid to *carry the gap in time* between a seller and a buyer. The spread is the fee for immediacy plus the compensation for the risk of holding the security in between. Many small spreads on huge volume add up to a large, steady business.

What sets the spread the desk can charge is not greed but **adverse selection** — the fear that the client selling to it knows something it does not. If a fund dumps a \$50M block, the desk has to wonder: is this a routine rebalancing, or does this seller know bad news that hasn't hit the tape yet? The wider the spread, the more the desk protects itself against trading with someone better-informed. This is why illiquid, hard-to-value securities carry fat spreads and liquid mega-caps carry razor-thin ones — and why the formal models of market-making and the order book are a discipline of their own (see [Order Book Simulator](/blog/trading/quantitative-finance/order-book-simulator-quant-research) and [Market-Making Simulator](/blog/trading/quantitative-finance/market-making-simulator-quant-research)). The desk also has to manage **inventory risk**: every block it buys to facilitate a client sits on its book exposed to market moves until it can lay it off, so a good desk continuously hedges its net position rather than betting on direction.

The role connects straight back to primary issuance. When IBD prices a new bond, the buyers ask one question before committing: *will someone make a market in this afterward so I can sell if I need to?* The answer is usually "yes, our trading desk will." That promise of secondary liquidity is what lets the primary deal clear. A desk that quotes the bonds it underwrote is, in effect, standing behind its own product — which is both a service to investors and, occasionally, a temptation to support a price that should be allowed to fall.

### FICC vs Equities — the two trading floors

Sales & Trading splits into two broad halves by asset class:

- **Equities** — stocks, equity derivatives, ETFs. Spreads are tiny on liquid names because competition (including electronic market-makers) is fierce.
- **FICC** — Fixed Income, Currencies, and Commodities: bonds, rates, credit, FX, commodities. FICC is larger and more opaque; many products trade over-the-counter (dealer-to-client) rather than on a lit exchange, so spreads are wider and relationships matter more.

The two halves earn their spreads in structurally different arenas. Equities increasingly trade on *lit* exchanges and through electronic market-makers, where the spread is competed down to almost nothing on liquid names — so the equities desk's edge is increasingly in execution technology, block facilitation, and derivatives rather than vanilla share quoting. FICC, by contrast, is largely *over-the-counter*: a corporate bond does not trade on a central exchange but is quoted by dealers to clients who phone or message in. That opacity is the desk's friend — wider, less-observable spreads — and its risk, because pricing a bond nobody has traded today is genuinely hard. The fragmentation of where equities actually trade (lit exchanges versus dark pools and wholesalers) is its own subject in [Lit Markets, Dark Pools, and the Fragmented Tape](/blog/trading/capital-markets/lit-markets-dark-pools-and-the-fragmented-tape).

The size of a spread is not arbitrary — it tracks **liquidity**. A mega-cap stock that trades billions of dollars a day has a spread of a basis point or two; a micro-cap that barely trades has a spread of dozens of basis points, because the market-maker bears far more risk holding something it may not be able to offload. The chart makes the relationship vivid.

![Bid-ask spread by liquidity tier in basis points from mega-cap to micro-cap](/imgs/blogs/inside-an-investment-bank-ecm-dcm-ma-and-trading-6.png)

A second force shapes the equities desk: it has been **automated** more aggressively than any other part of the bank. Most liquid stock now trades electronically, and specialist electronic market-makers (Citadel Securities, Virtu) compete the spread to a fraction of a cent. The bank's equities desk has responded by moving up the value chain — into block trades too large to work electronically, into derivatives and structured products, and into selling *algorithms* that slice a client's big order into the market without moving the price. The human trader who once shouted quotes now mostly supervises systems and handles the trades the machines cannot. FICC has automated more slowly precisely because its products are less standardised, which is part of why FICC spreads stayed wider for longer.

The whole Markets business sits on top of a vast pool of tradable securities — the secondary market it serves. US equity market cap has roughly doubled over the past decade, and a bigger, more active market means more flow to intermediate. That growing pool is the engine's fuel: the larger the stock of securities outstanding and the more they turn over, the more spread there is to capture. It is also the clearest picture of why the secondary market is the issuer's silent partner: every trillion dollars of market cap that the trading floor keeps liquid is a trillion dollars of capital that companies could, in principle, tap again through a follow-on — because investors know they can get out. A liquid secondary market is a standing invitation to issue.

![US equity market cap from 2014 to 2024 roughly doubling over the decade](/imgs/blogs/inside-an-investment-bank-ecm-dcm-ma-and-trading-8.png)

### Prime brokerage — the desk that banks the buy-side

A third desk inside Markets deserves its own mention: **prime brokerage**. Prime brokerage is the package of services a bank sells to hedge funds and other large traders — custody of their assets, financing (lending them money to buy more, i.e. *leverage*), securities lending (lending them shares to *short*), trade clearing, and reporting. It is, in effect, the bank acting as the operational backbone for the buy-side.

Prime brokerage does not earn a trading spread; it earns **financing fees and lending spreads** — the interest on the margin loans it extends and the fees on the securities it lends out. It is a balance-sheet-intensive, relationship-driven, recurring-revenue business, which makes it prized: it is steadier than trading and stickier than deals. (How prime brokerage connects to securities lending and repo is covered in [Securities Lending and Repo](/blog/trading/capital-markets/securities-lending-and-repo-the-financing-plumbing).)

The reason prime brokerage is so sticky is operational: once a hedge fund custodies its assets, clears its trades, and finances its book through one prime broker, switching is a months-long migration. So the bank earns a recurring stream — financing interest, stock-loan fees, and a share of the fund's trading commissions — for as long as the relationship lasts. The downside is concentration: a prime broker is, in effect, *lending against* its clients' portfolios, and if a large leveraged client blows up faster than the bank can liquidate the collateral, the loss lands on the bank. That is the fat tail we'll meet in the Archegos case below.

#### Worked example: a prime brokerage financing fee

A hedge fund puts up \$100M of its own equity and borrows \$150M from its prime broker to hold a \$250M portfolio (1.5× leverage). The financing rate is 5% a year.

- Margin loan = \$150M; annual interest = 5% × \$150M = **\$7.5M** to the prime broker.
- Add securities-lending fees on shares the fund borrows to short — say \$1M a year.
- Add execution commissions routed to the bank — say \$2M a year.
- Total recurring prime revenue from this one client ≈ **\$10.5M a year**, with no deal event required.

The intuition: prime brokerage earns like a bank, not like a dealer — a steady interest-and-fee stream on balances, repeating every year the client stays. The tradeoff is that the bank is on the hook if that \$150M loan ever exceeds the value of the collateral behind it.

### Flow vs prop — and why Volcker drew a line

Two ways a desk can use its book:

- **Flow trading** — trading to *facilitate clients*: the desk takes positions to fill customer orders and make markets, capturing the spread. The position is a by-product of serving the client.
- **Proprietary ("prop") trading** — the desk trading the bank's *own* capital to make a directional bet, with no client on the other side. The position is the whole point.

After the 2008 crisis, the US **Volcker Rule** (part of the 2010 Dodd-Frank Act) curbed prop trading at deposit-taking banks. The logic: a bank funded partly by insured deposits and backstopped by taxpayers should not gamble that money on directional bets. Market-making and hedging — flow — remained allowed, because they serve clients and provide liquidity; pure prop, where the bank is just speculating, was pushed out (to hedge funds, in large part). The line between "facilitating a client" and "betting" is genuinely blurry, which is why Volcker compliance is a small industry in itself. The practical effect: the modern bank trading floor is overwhelmingly a *flow* business — paid to intermediate, not to gamble.

## Research: the glue between the two sides

Sitting alongside the trading floor is **equity (and credit) research** — analysts who publish ratings, price targets, and earnings forecasts on the companies the bank covers. Research generates no direct fee. So why does the bank pay for it? Because research is the **glue** that connects every other desk to investors. (The analyst, the rating, and the wall get a full treatment in [Sell-Side Research: The Analyst, the Rating, and the Wall](/blog/trading/capital-markets/sell-side-research-the-analyst-the-rating-and-the-wall).)

Research does three jobs. It gives the *sales* force something to talk to clients about, which drives trading commissions to the desk. It lends *credibility* to ECM deals — a company is easier to sell at IPO if respected analysts will cover it afterward. And it gives the buy-side a reason to route business to the bank. Historically research was paid for by bundled trading commissions; in Europe, MiFID II forced it to be priced and paid for separately, shrinking research budgets and the analyst headcount across the industry.

The economics of research are worth dwelling on because they are so counterintuitive: it is a cost centre that the whole bank depends on. An analyst publishing a "buy" on a stock generates no invoice. The payment is indirect — the buy-side fund that values the analyst's work routes its trades through the bank's desk, and the resulting commissions cover the research cost several times over. This bundling worked for decades but hid the true price of research, which is exactly why MiFID II unbundled it: forcing funds to write an explicit cheque for research revealed how little a lot of it was worth, and budgets collapsed. The survivors are the analysts whose calls genuinely move the buy-side's decisions — a brutal market test that the old bundled model masked.

There is one more role research plays that ties straight back to the deal floor: an analyst who covers a company well makes that company easier to take public and easier to keep raising capital. A company contemplating an IPO will look at whether a bank's analyst is respected in its sector, because post-listing coverage is part of what it is buying. This is exactly the link that the post-2003 rules had to police — because the temptation was for IBD to promise glowing coverage to win the mandate, corrupting the analyst's independence. Research is the most conflicted seat in the building, which brings us to the walls.

## The conflicts — and the walls that contain them

A bank that both advises an issuer *and* serves investors has a structural problem: its interests point in two directions at once. It wants to sell the issuer's stock at a high price (good for the issuer client and the fee) and it wants its investor clients to buy at a price that will go up (good for them). It knows secrets on the deal side that would be enormously valuable on the trading side. Left unmanaged, these conflicts would let the bank front-run its own clients, prop up the companies it underwrites with rosy research, and trade on inside information.

The containment mechanism is the **Chinese wall** (or "information barrier"): a set of rules, physical separations, and access controls that split the firm into a **private side** (IBD — people who hold material non-public information) and a **public side** (Sales, Trading, Research — people who deal with public markets). Information does not cross the wall except through a controlled, logged process called **wall-crossing**, where a public-side person is brought "over the wall," made an insider, and restricted from trading until the information is public.

![The wall separating the private deal side from the public trading and research side](/imgs/blogs/inside-an-investment-bank-ecm-dcm-ma-and-trading-7.png)

It helps to enumerate the specific conflicts the wall is built to contain, because "conflict of interest" is vague until you make it concrete:

- **Front-running.** If the trading desk learned that IBD was about to announce a takeover, it could buy the target's stock first and profit when the news broke. That is insider trading, full stop.
- **Tainted research.** If the analyst knew banking wanted the issuer's business, a "buy" rating becomes a sales pitch rather than a judgment. The 2003 settlement exists because this happened at scale.
- **Allocation favoritism.** In a hot IPO, the bank decides which investors get shares. It could steer allocations to clients who reward it elsewhere (a practice called *spinning* when it favors executives), shortchanging the issuer's pricing.
- **Dual representation.** Advising both a buyer and a seller, or financing the same deal it advises on, pits the bank's duties against each other.

The walls are not merely cultural; they are policed by **compliance**, recorded, and enforced by regulators. After the dot-com era, the 2003 **Global Analyst Research Settlement** forced banks to separate research from investment banking — because analysts had been publishing "buy" ratings on companies their IBD colleagues were trying to win deals from. Research analysts can no longer be paid based on banking revenue, and banking cannot promise favorable coverage. The wall is the institutional answer to the question: how can one firm sit on both sides of the market without cheating the people on each side?

The honest takeaway is that the walls *manage* the conflict rather than eliminate it. The conflict is inherent to the business model — it is the price of being a full-service intermediary. The reader's job is to know which hat the bank is wearing in any given interaction: is it advising you, or selling to you?

## The fee pools and the league tables

How do the desks' earnings stack up across the industry? The global investment-banking *fee pool* — ECM + DCM + M&A advisory — runs on the order of **\$100bn a year**, swinging with the deal cycle. Sales & Trading revenue across the big banks is larger and steadier, often several times the IBD pool, because it lives on flow rather than events.

Within IBD, banks obsess over **league tables** — published rankings of which bank did the most deal volume, by product and region, over a period. League tables are the industry's scoreboard. They matter because they are self-reinforcing: a company choosing a bookrunner for its IPO wants a bank that is *seen* to be a leader, so a high ranking wins more mandates, which raises the ranking. The "bulge bracket" — the handful of banks that top these tables globally — earns an outsized share of the fee pool precisely because reputation compounds. (The deeper economics of how banks make money across these lines is in [Inside an Investment Bank: How They Make Money](/blog/trading/finance/inside-an-investment-bank-how-they-make-money).)

The league-table game has a quirk that reveals how the industry thinks: banks will sometimes take a low-fee or even loss-leading role in a marquee deal just to claim *credit* in the rankings, because the ranking itself wins future, more profitable mandates. A bank listed as a bookrunner on the year's biggest IPO advertises that fact to every other company considering a listing. This is why deal "credit" is fought over in negotiations almost as hard as the fee itself, and why the same transaction can list a long roster of banks — each wants its name on the tombstone. The scoreboard is not vanity; it is the marketing engine of a business where reputation is the main barrier to entry.

It is also worth separating the **fee pool** (the revenue) from **profitability** (what's left after costs and capital). M&A looks small in headline revenue but is hugely profitable because it consumes almost no capital and few people relative to the fee. Sales & Trading generates enormous revenue but eats balance sheet and technology spend, so its return on the capital it ties up can be modest. When banks reorganise — shrinking a FICC desk, building out advisory — they are usually chasing *return on capital*, not raw revenue. The post-crisis decade has been one long migration toward the capital-light end of that spectrum.

The economics differ sharply by line:

- **M&A**: highest margin (almost no capital), most volatile (pure event fees), reputation-driven.
- **ECM**: high fee (7% spread), brutally cyclical (the IPO window opens and shuts).
- **DCM**: thin fee (sub-1%), high volume, steadiest across the cycle.
- **Sales & Trading**: tiny spreads, vast volume, capital-intensive, the most balance-sheet-heavy line.

This mix is why a diversified bank is more stable than the sum of its parts. The four lines do not peak together: M&A and ECM thrive in calm, confident, rising markets, while Sales & Trading often does best in *volatile* markets when clients scramble to reposition and spreads widen. So a year that starves the deal floor can feed the trading floor, and vice versa. A bank that is strong across all four lines smooths its earnings across the cycle in a way a pure M&A boutique or a pure trading shop cannot — which is the strategic logic behind the full-service "universal bank" model in the first place.

There is also a regulatory layer shaping the economics. Because Sales & Trading and prime brokerage consume balance sheet, they are governed by **capital requirements** — rules forcing the bank to hold equity against the risk it warehouses. Post-2008 rules (Basel III, the Volcker Rule, leverage ratios) raised the capital cost of trading inventory and prime-brokerage lending, which nudged banks toward *capital-light* businesses: advisory, where no balance sheet is at risk, and agency execution, where the bank routes rather than warehouses. That regulatory gravity has reshaped the modern bank as surely as any market trend — the cheapest revenue to earn, in regulatory-capital terms, is the M&A fee that needs no capital at all.

## Common misconceptions

**"Investment banks mostly trade their own money to get rich."** Not since Volcker. The modern floor is overwhelmingly a *flow* business — making markets and facilitating clients, not directional betting with the bank's own capital. Pure prop trading migrated largely to hedge funds after 2010. The bank's edge is volume and the spread, not a crystal ball.

**"The 7% IPO fee is the bank's profit."** No — it is the *gross* spread, split across a syndicate and consumed largely by the cost of distribution (the ~60% selling concession). The lead bank's net take after sharing, expenses, and the risk of an underwritten deal is far less than 7%.

**"Research is free, objective advice for investors."** Research generates no direct fee and is structurally conflicted: the bank that rates a stock may also want its issuer's banking business. The post-2003 walls and pay rules exist precisely because that conflict was once abused. Read sell-side research as informed but interested.

**"DCM is a small sideline next to the glamorous IPO business."** The opposite by volume. Bond issuance dwarfs equity issuance; a sub-1% fee on trillions of dollars of debt is a fee pool that rivals ECM and is far steadier. DCM is the quiet workhorse.

**"Bigger deals pay bigger percentage fees."** Generally the reverse. Fee *percentages* shrink as deal size grows — a \$10B M&A deal pays well under 1%, while a \$200M deal might pay 1.5–2%. The absolute fee rises with size; the rate falls.

**"The bank's job in an IPO is to get the highest price for the company."** Only partly. The bank serves two masters: the issuer (who wants a high price) and the investors it sells to repeatedly (who want a price that will rise). A modest first-day "pop" keeps investors happy and the bank's allocation valuable — which is why critics argue underwriters systematically underprice IPOs at the issuer's expense. The 26% pop in our opening was \$7 a share the company arguably left on the table, and a happy day for the buyers the bank must keep coming back.

**"A high league-table ranking means a bank is the most profitable."** Not necessarily. Rankings measure *deal volume credit*, not profit, and banks sometimes buy ranking with low-fee roles. A boutique far down the volume table can earn higher margins per banker than a bulge-bracket name at the top, because it carries no capital-hungry trading floor.

## How it shows up in real markets

**The 2021 IPO boom and 2022 bust.** In 2021, US IPOs raised about \$142bn — a feast for ECM desks, whose 7% spreads turned into roughly \$10bn of fees. Then rates rose, valuations cracked, and 2022 IPO proceeds collapsed to about \$8bn — a ~94% drop. ECM bankers who had been the heroes of 2021 were idle in 2022; the same teams pivoted to whatever was still issuing. This is the defining feature of IBD revenue: it is a *cyclical, event-driven* business, and the cycle can turn in a single quarter.

**A blockbuster IPO's split in practice.** When a large company lists with a 7% spread, the lead bookrunners — the banks running the order book — capture the management fee and the largest selling concessions, while a long tail of co-managers takes smaller slices for bringing incremental buyers. The league-table credit, often the most coveted prize, goes disproportionately to the bookrunners. This is why banks fight so hard for the bookrunner role and why the order book — the list of who will buy — is the deal's most guarded asset.

**FICC's good years.** Trading revenue is counter-cyclical to deals in an important way: volatility is good for market-makers. In turbulent years — 2020's pandemic shock, 2022's rate repricing — FICC desks posted bumper results even as the IPO window shut, because wide spreads and frantic client hedging meant more flow to intermediate. A bank with both a strong IBD and a strong Markets division is naturally hedged across the cycle: when deals dry up, trading often picks up.

**The Archegos blowup (2021).** Prime brokerage's risk showed up starkly when the family office Archegos defaulted on margin loans, costing its prime brokers — Credit Suisse most of all, over \$5bn — enormous losses. It was a reminder that the steady, recurring prime-brokerage business carries a fat tail: when a leveraged client implodes, the financing desk is left holding collateral worth less than the loan. The desk that quietly banks the buy-side can take the loudest losses. Several banks had each extended Archegos enormous leverage without seeing the others' exposures, so when the positions unwound they raced to liquidate the same concentrated stocks at once — the banks that sold first lost least. It was a textbook illustration of why prime-brokerage risk management, not the financing margin, is the business's true core competence.

**Why deals and trading sit under one roof at all.** A reasonable reader might ask: if the conflicts are so severe, why not just split the bank? Some firms effectively are split — pure-play M&A boutiques (Evercore, Lazard) carry no trading floor and no balance sheet, and they compete fiercely for advisory mandates precisely by being conflict-free. The counter-argument for the universal model is that the businesses feed each other: a trading desk's read on investor appetite informs how to price a deal; an underwriting relationship opens the door to the issuer's trading and treasury business; research that the buy-side respects drives both commissions and deal credibility. The full-service bank is a bet that the synergies outweigh the conflicts — and the walls are the price of taking that bet without breaking the law.

**Vietnam and emerging markets.** The same desk structure appears in smaller markets, scaled down and often less specialised. A Vietnamese securities firm running an IPO on the Ho Chi Minh exchange does ECM, DCM, and brokerage under one roof, but the secondary-market liquidity that anchors everything is thinner and more sentiment-driven — foreign flows in and out of the market can swing the index hard, which in turn opens or shuts the local issuance window. The principle is identical to the US: the depth of the secondary market sets the terms for primary deals. The local nuance — how foreign flows, ETFs, and index inclusion move a frontier market — is covered in [Foreign Flows, ETFs, and the Index Effect in Vietnam](/blog/trading/vietnam-stocks/foreign-flows-etfs-and-the-index-effect-vietnam).

## The takeaway: one machine, many meters

The cleanest way to hold an investment bank in your head is this: it is a single machine for moving savings into investment, with a different *meter* on each desk measuring how it charges for its slice of that flow.

- IBD's meter is the **deal event** — a spread on what it sells, a fee on what it advises.
- Markets' meter is the **spread on volume** — a sliver of every trade it stands in the middle of.
- Prime brokerage's meter is **financing** — interest and lending fees for banking the buy-side.
- Asset management's meter is **AUM** — a recurring fee on the money it runs.

Every one of those meters is charging for the same underlying service: connecting someone who has capital to someone who needs it, and standing in the gap to make the connection safe and immediate. The deal floor connects them in the *primary* market, by creating the security. The trading floor connects them in the *secondary* market, by making it liquid. And the second of those is what makes the first possible — no investor funds a new issue she cannot exit, so the bank's willingness to make a market tomorrow is what lets it sell the deal today.

This also explains why the industry keeps reorganising rather than settling into a fixed shape. The four meters earn at different times in the cycle and cost different amounts of regulatory capital, so the optimal *mix* of desks shifts as markets and rules change. After 2008, capital rules made trading inventory expensive, so banks leaned toward advisory; when rates are high, DCM hums while ECM freezes; when volatility spikes, FICC saves a year that the deal floor lost. A bank is therefore best understood not as a fixed object but as a portfolio of these meters, continuously rebalanced to earn the steadiest return on the capital it must hold. The boutiques that carry only the M&A meter accept more volatile revenue in exchange for zero capital drag and zero conflicts; the universal banks carry all four and accept the conflicts (and the walls) as the cost of the diversification.

The walls in the middle are not bureaucratic clutter; they are the load-bearing structure that lets one firm sit on both sides of the market without cheating either. When you read about a bank, ask which meter is running and which side of the wall you are standing on. That single question turns the bank from an opaque monolith into what it actually is: a stack of distinct, very different businesses, each charging in its own currency for the same elemental act of matchmaking.

## Further reading & cross-links

- [Inside an Investment Bank: How They Make Money](/blog/trading/finance/inside-an-investment-bank-how-they-make-money) — the high-level overview this post maps in detail.
- [The Broker-Dealer: Agency, Principal, and Prime Brokerage](/blog/trading/capital-markets/the-broker-dealer-agency-principal-and-prime-brokerage) — agency vs principal, and the prime desk up close.
- [The Buy-Side: Who Actually Owns the Market](/blog/trading/capital-markets/the-buy-side-who-actually-owns-the-market) — the investors the sell-side serves.
- [Sell-Side Research: The Analyst, the Rating, and the Wall](/blog/trading/capital-markets/sell-side-research-the-analyst-the-rating-and-the-wall) — research as the glue, and its conflicts.
- [The IPO Process, End to End: From Mandate to First Trade](/blog/trading/capital-markets/the-ipo-process-end-to-end-from-mandate-to-first-trade) — the ECM flagship deal, step by step.
- [Underwriting and the Syndicate: Who Takes the Risk](/blog/trading/capital-markets/underwriting-and-the-syndicate-who-takes-the-risk) — how the spread and the risk are shared.
- [The Yield Curve Explained](/blog/trading/fixed-income/the-yield-curve-explained-the-most-important-chart-in-finance) — the pricing backdrop for DCM bond deals.
