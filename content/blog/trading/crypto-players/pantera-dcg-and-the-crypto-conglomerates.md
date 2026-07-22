---
title: "Pantera and DCG: The Crypto Conglomerates Where Concentration Became Contagion"
date: "2026-07-22"
publishDate: "2026-07-22"
description: "How two crypto empires — Pantera's fund and DCG's holding company of a trust, a lender, and an OTC desk — show why owning the whole value chain turns an ordinary downturn into a solvency crisis."
tags: ["crypto", "digital-currency-group", "grayscale", "gbtc", "genesis", "pantera-capital", "contagion", "closed-end-fund", "related-party", "crypto-lending", "market-structure", "gemini-earn"]
category: "trading"
subcategory: "Crypto Players"
author: "Hiep Tran"
featured: true
readTime: 40
---

> [!important]
> **TL;DR** — When one owner controls the fund, the trust, the lender, *and* the OTC desk, money can move between those pockets in ways an outside investor never sees — and a loss in one pocket can drain all the others.
>
> - **Pantera Capital** is the *fund* model: one of the earliest US crypto funds (its first Bitcoin fund launched in 2013), running venture, liquid-token, and passive-Bitcoin strategies. A loss there mostly stays inside the fund's own net asset value.
> - **Digital Currency Group (DCG)** is the *conglomerate* model: one holding company that owned Grayscale (which ran the GBTC trust), Genesis (a lender and OTC desk), CoinDesk (media), and more — valued at roughly \$10 billion in November 2021.
> - The signature machine was the **GBTC premium trade**: GBTC traded as much as ~40% *above* the bitcoin it held in 2020–2021, then flipped to a discount of nearly **50% by December 2022** — a swing that turned a "free money" trade into forced selling and margin calls.
> - When Three Arrows Capital (3AC) defaulted in mid-2022, Genesis reportedly had **\$2.36 billion** lent to it, and DCG stepped in with a **\$1.1 billion promissory note** — an intercompany IOU that papered over the hole. Genesis still owed roughly **\$900 million** to about **230,000+ retail savers** through the Gemini Earn program.
> - Genesis filed for bankruptcy in **January 2023**. In 2024, Gemini agreed to return **at least \$1.1 billion** to Earn users, Genesis reached a **\$2 billion** settlement with the New York Attorney General, and GBTC finally converted to an ETF — closing the discount. The related-party allegations against DCG were *reported and alleged*; the numbers below are sourced or dated.

Here is a question worth sitting with: what happens when the same person owns the fund that buys a token, the trust that holds it, the desk that trades it, and the lender that lends against it?

In traditional finance, walls exist between those jobs on purpose — a broker is not supposed to also be your lender, your fund manager, *and* the market maker quoting your price, because each of those roles can be tempted to help the others at your expense. Crypto, in its first decade, tore most of those walls down. Nowhere was that clearer than at **Digital Currency Group** — the holding company Barry Silbert built into a genuine crypto conglomerate — and nowhere was the contrast starker than with **Pantera Capital**, a pioneer that mostly stayed in its lane as a fund.

This post builds the whole thing from zero. We will define every term — *holding company*, *closed-end trust*, *net asset value*, *premium and discount*, *crypto lender*, *OTC desk*, *related-party transaction*, *intercompany loan*, *contagion* — and then we will watch, with real (sourced) numbers, how a group that owned every link in the chain turned an ordinary bear market into a solvency crisis that stranded hundreds of thousands of ordinary savers.

![The DCG conglomerate: one holding company sat atop a trust, a lender, an OTC desk and a media outlet at once.](/imgs/blogs/pantera-dcg-and-the-crypto-conglomerates-1.webp)

The diagram above is the mental model for the whole article. At the top sits DCG, the holding company. Below it are the businesses it owned: **Grayscale**, which ran the GBTC trust; **Genesis**, the lender and OTC desk; and quieter arms like CoinDesk (media) and Foundry (mining). The bottom row is where the trouble lived — a trust earning a fat fee, a lender with loans out to funds like 3AC and Alameda, and a roughly \$900 million debt owed to retail savers. Every arrow in this picture is a place where money, or a loss, could travel between siblings. The rest of the post walks each one.

This is educational writing about market structure and a real, litigated episode — not investment advice, and not an accusation. Where a claim is contested, it is labeled *reported*, *alleged*, or *per the complaint*, and the outcomes are stated as they were reported.

## Foundations: the building blocks

Before we can see why concentration became contagion, we need a shared vocabulary. If you already know what a closed-end fund's NAV discount is, skim. If not, read every definition — the rest of the article leans on all of them.

### What a holding company (and a conglomerate) actually is

A **holding company** is a company whose main asset is *other companies*. It doesn't necessarily sell a product itself; it owns controlling stakes in businesses that do. Berkshire Hathaway is the famous example — it owns insurers, a railroad, a candy company, and stock in dozens of firms.

A **conglomerate** is a holding company whose businesses span very different industries. The appeal is diversification and shared resources: if one arm has a bad year, another can carry it, and cash can be shuffled internally to wherever it's most useful.

That last sentence is also the danger. "Cash can be shuffled internally" means a parent can move money *from* a healthy subsidiary *to* a sick one — or borrow from one to prop up another. When those moves happen between companies under the same roof, they are **related-party transactions** (defined below), and they are exactly the moves outside investors and customers cannot see.

DCG was a crypto conglomerate: under one parent sat an asset manager, a lender, a broker/OTC desk, a media company, and a mining operation. **Pantera** was not a conglomerate — it was an asset-management firm running funds. That structural difference is the entire story.

### What a closed-end trust is, and what "NAV" means

Most people meet investment funds through **open-end funds** (like a typical index mutual fund or a modern ETF). "Open-end" means the fund can create and destroy its own shares on demand: if you put in \$100, the fund mints \$100 of new shares and buys \$100 of assets; if you cash out, it sells \$100 of assets and burns your shares. Because shares are freely created and redeemed, the market price of a share stays glued to the value of the assets behind it.

The value of those underlying assets, per share, is the **net asset value (NAV)** — literally, (assets − liabilities) ÷ number of shares. NAV is the "fair value" of one share: what the stuff inside is worth right now.

A **closed-end fund** (or **closed-end trust**) is different. It issues a fixed pile of shares and then — crucially — those shares trade on their own. There is usually *no easy redemption*: you generally cannot hand a share back to the trust for its NAV in cash. If you want out, you sell your share to another investor on a market, at whatever price that market offers. That price can be higher or lower than NAV, and there is no automatic mechanism forcing it back in line.

**Grayscale's Bitcoin Trust (GBTC)** was, for most of its life, a closed-end trust holding bitcoin. Each share represented a slice of a big pile of BTC. But GBTC shares traded on the over-the-counter market (OTCQX) at a price set by supply and demand for the *shares*, which could drift far from the value of the *bitcoin* inside.

### Premium and discount to NAV

This gap has a name.

- When a share trades **above** its NAV, it trades at a **premium**. Buyers are paying more for the share than the assets inside are worth. (Why would anyone? Usually because the share is the only convenient way to get exposure — more on that shortly.)
- When a share trades **below** its NAV, it trades at a **discount**. The share is cheaper than the assets inside.

> A premium is the market paying extra for convenience or scarcity; a discount is the market charging you for being trapped. In a closed-end fund with no redemption, both can persist for years, because nothing forces the price back to NAV.

Hold onto that: *nothing forces the price back to NAV*. In an ETF, arbitrageurs instantly redeem shares for the underlying whenever a gap opens, so gaps vanish. In a closed-end trust with a one-way door — bitcoin goes *in* but shares can't be redeemed *out* — the gap can grow enormous. GBTC's did, in both directions.

### What a crypto lender is

A **crypto lender** takes in crypto (or cash) from people who want to earn interest, and lends it out to people who want to borrow, keeping the spread. Genesis was one of the largest. On one side, it borrowed assets — including, through the **Gemini Earn** program, crypto from ordinary retail savers who were promised interest. On the other side, it lent those assets to trading firms and funds (like 3AC and Alameda) who paid to borrow.

This is the same maturity-and-credit transformation a bank does: borrow from many, lend to a few, profit from the spread, and pray your borrowers pay you back. The difference is that a crypto lender in 2022 had no deposit insurance, thin capital, little disclosure, and a handful of enormous, correlated borrowers. If one big borrower blew up, the lender could blow up — and everyone who lent *to* the lender was suddenly a creditor in a bankruptcy.

### What an OTC desk is

**OTC** stands for **over-the-counter** — trading that happens *off* a public exchange order book, negotiated directly between two parties. An **OTC desk** helps large players buy or sell big blocks of crypto without moving the public price. If a fund wants to buy \$50 million of a token, dumping that into a thin exchange order book would spike the price against them; an OTC desk instead finds a counterparty and prices the block privately. Genesis ran a large OTC/trading operation alongside its lending. Owning both a lender and an OTC desk means one firm sees both the borrowing *and* the trading flow of its clients — an information advantage, and a conflict.

### Why finance builds walls (and crypto didn't)

Traditional finance is full of deliberate separations. After the 1929 crash, the US **Glass-Steagall** law split commercial banking (taking deposits) from investment banking (underwriting and trading securities) for decades — the logic being that a bank shouldn't gamble with the deposits it's supposed to safeguard. Brokers keep customer assets *segregated* from their own. Fund managers face rules against self-dealing. Research analysts are separated from bankers by **information barriers** (often called "Chinese walls"). Auditors are supposed to be independent of the companies they audit.

Every one of these walls costs efficiency — it would be *cheaper* to let one firm do everything. They exist anyway, because the alternative repeatedly produced disasters: the conflicted party helping itself at the customer's expense, or one blown-up division dragging down the safe one next door. The walls are scar tissue from past failures.

Crypto, being new and lightly regulated, mostly skipped the scar tissue. It was normal — even celebrated as "vertical integration" — for one group to run a fund, a trust, a lender, an exchange, and a market maker at once. DCG was among the most integrated. The efficiency was real. So was the missing wall. When you read the rest of this post, notice how many of the specific failures map exactly onto a wall that TradFi builds and DCG's structure did not have.

### Related-party transactions and intercompany loans

A **related-party transaction** is a deal between two entities that are connected — same parent, same owner, overlapping management. A loan from a lender to *its own parent company* is the textbook case. These aren't automatically wrong; they happen constantly inside corporate groups. But they get special scrutiny because the usual check — an arm's-length counterparty who will say "no" if the terms are bad — is missing. When the borrower and lender answer to the same boss, "no" is hard to say.

An **intercompany loan** is exactly that: one company in a group lends to another. A **promissory note** is a written IOU — a promise to repay a stated sum by a stated date. When DCG gave Genesis a \$1.1 billion promissory note, it was writing an IOU to a company it controlled, to fill a hole in that company's books.

### Contagion

Finally, **contagion**: when trouble at one entity spreads to others through the connections between them — a loan that won't be repaid, a shared owner, a collateral asset they both hold. The word is borrowed from epidemiology on purpose. In a tightly connected group, one sick node can infect its neighbors fast.

#### Worked example: how a discount is different from a loss (the simplest case)

Let's ground NAV with the friendliest possible numbers before any real ones.

Suppose a closed-end trust holds exactly **1 bitcoin**, and there are **100 shares** outstanding. Bitcoin is worth **\$100,000**. Then:

$$\text{NAV per share} = \frac{\$100{,}000}{100} = \$1{,}000.$$

Now the market. If demand for shares is hot, a share might trade at **\$1,400** — a **40% premium**. You paid \$1,400 for something whose fair value is \$1,000. If the mood sours and everyone wants out, the same share might trade at **\$550** — a **45% discount**. The bitcoin inside hasn't changed; the *share price* has detached from it.

Here is the subtle part beginners miss: at a 45% discount, you have *not* necessarily lost 45% of your money. If bitcoin *rose* while your discount widened, your NAV went up even as your share price fell. You can hold an asset whose underlying value climbed and still be underwater, purely because the wrapper you bought it through got cheaper. That wedge — between what you own and what you can sell it for — is the seed of everything that follows.

The one-sentence intuition: **in a closed-end trust, you are exposed to two prices at once — the assets inside, and the market's mood about the wrapper — and they can move against each other.**

## The two models: a fund versus a conglomerate

Pantera and DCG are both "crypto giants," but they are built completely differently, and the difference decides how far a loss can travel.

![Two conglomerate models, two conflict surfaces: a fund's loss stays in its NAV; a conglomerate's loss can travel trust to lender to parent.](/imgs/blogs/pantera-dcg-and-the-crypto-conglomerates-8.webp)

### Pantera: the fund model

**Pantera Capital** was founded in 2003 by Dan Morehead as a global-macro hedge fund, then pivoted hard into crypto. In 2013 it launched what it describes as the first institutional Bitcoin fund in the United States, when bitcoin traded around \$65 (Pantera). That early bet became legendary: Pantera reported its Bitcoin Fund had returned **131,165%** net of fees over its lifetime — roughly a 1,000x — as of November 2024 (Pantera; CoinDesk, 2024-11-26).

Pantera grew into one of the larger digital-asset managers, with assets under management that climbed from about \$400 million to over \$5 billion and stood around \$4.8 billion in blockchain assets (Pantera, 2024). It runs several distinct strategies — a passive **Bitcoin fund**, a **venture fund** (equity stakes in crypto companies), a **liquid-token fund** (publicly traded tokens), and early-stage token funds — but the key structural fact is simple: *Pantera manages money*. Its clients are limited partners whose capital sits in funds. When a Pantera bet goes to zero, the loss shows up in that fund's NAV, and the LPs in that fund bear it. There is no lender full of retail deposits sitting next door, no trust whose fee revenue depends on the same assets, no OTC desk trading against the flow. The blast radius of a bad trade is, structurally, the fund itself.

That is not a moral virtue — it is a structural one. A fund can still lose you money. But it has *fewer pipes* through which one loss can drain other pockets.

### DCG: the conglomerate model

**Digital Currency Group** was launched in 2015 by Barry Silbert, who had just sold his previous company, SecondMarket, to Nasdaq. DCG was built as a holding company, and it acquired or founded an entire stack of crypto businesses:

- **Grayscale Investments** — an asset manager that ran a family of closed-end crypto trusts, the largest being **GBTC**. Grayscale charged a **2% annual management fee** on the assets it held, which on a multi-billion-dollar trust meant hundreds of millions of dollars a year in revenue.
- **Genesis** — a prime brokerage, **lender**, and **OTC/trading desk** rolled into one. Genesis borrowed from retail (via Gemini Earn) and institutions, and lent to trading firms.
- **CoinDesk** — a leading crypto media and events business (it ran the influential Consensus conference; DCG later sold it).
- **Foundry** — a bitcoin mining and staking operation.
- Plus stakes in **175+** companies across 35+ countries.

In November 2021, at the top of the bull market, DCG raised **\$700 million** in a secondary share sale led by SoftBank, with Alphabet's CapitalG and Ribbit Capital participating, valuing the group at about **\$10 billion** (CoinDesk; CNBC, 2021-11-01).

Look at what one owner now controlled: the **trust** that held the bitcoin, the **fee** that trust generated, the **lender** that lent against crypto (including against GBTC shares), and the **OTC desk** that traded it. That is four links of the same chain. In good times, that integration was a profit machine — each arm fed the others. In bad times, it was a set of open pipes.

| | **Pantera** | **DCG** |
|---|---|---|
| Core business | Funds (venture + liquid + passive BTC) | Holding company over 6+ operating arms |
| Owns a lender that takes retail deposits? | No | Yes — Genesis, funded partly via Gemini Earn |
| Owns a fee-earning trust? | No | Yes — Grayscale / GBTC (2% fee) |
| Related-party surface | Low | High (trust ↔ lender ↔ parent) |
| How far a single loss travels | Mostly the fund's own NAV | Trust → lender → parent → retail creditors |

The rest of this post is about that last row.

### The fourth hat: why owning the OTC desk mattered too

We'll spend most of the post on the trust and the lender, but don't forget the desk. Genesis wasn't only a lender — it was also a large **OTC/trading** operation. Owning a desk alongside a lender gives one firm a rare vantage point: it can see who is *borrowing*, who is *selling*, and who is *desperate*, all at once. In TradFi, that combination sits behind an information barrier precisely because it's so exploitable — a desk that knows a client is a forced seller can price accordingly.

For DCG's group, the desk also meant that when 3AC's collateral had to be liquidated, the group had its own machinery to do it — and its own view of how bad the flow was long before outsiders did. None of this requires anyone to have behaved badly; the point is structural. A saver lending through Gemini Earn was, without knowing it, upstream of a lender, a trust, *and* a desk that all shared an owner and a view of the same falling market. Four hats, one head. That is the maximal version of the conflict this whole series maps — and it's why the same downturn that merely bruised a fund could be existential for a conglomerate.

## The signature machine: the GBTC premium trade

To understand how DCG's pipes carried a loss, you first have to understand the single most profitable — and most dangerous — machine in the group: the **GBTC premium trade**.

### Why GBTC existed and why it traded at a premium

For years, most institutions and many retail brokerage accounts could not easily hold bitcoin directly — custody was hard, compliance was harder, and a spot bitcoin ETF did not yet exist in the US. GBTC solved that: it was a security, with a ticker, that you could buy in a normal brokerage account, and it held bitcoin for you. Convenience had a captive audience.

Because GBTC was a closed-end trust, its share price was set by demand for the *shares*, not by redemption. And the way *new* shares got created had a delay built in. Accredited investors (and institutions) could do an **in-kind creation**: hand bitcoin (or cash) to Grayscale and receive newly minted GBTC shares at NAV. But those freshly minted shares were **locked up for six months** before they could be sold on the public market.

That six-month delay is the whole trick. New supply of tradable shares always lagged demand. So on the public market, buyers bid GBTC *above* NAV — a **premium**. And that premium created a seemingly magical arbitrage.

![The GBTC premium machine: deposit bitcoin, wait out the six-month lockup, sell the shares above NAV — a money machine that only ran while the premium held.](/imgs/blogs/pantera-dcg-and-the-crypto-conglomerates-3.webp)

Follow the pipeline above. A firm would **borrow bitcoin** (often with leverage), **deposit** it with Grayscale, receive GBTC shares **at NAV**, wait out the **six-month lockup**, then **sell** the shares on OTCQX at a **premium**. If the premium held, the difference was profit — on top of any move in bitcoin itself.

#### Worked example 1: the premium trade, and the moment it flips

Let's run it with round numbers, then let the trade turn against us.

**The good years (2020–2021).** Suppose you deposit **\$1,000,000** of bitcoin with Grayscale and receive GBTC shares worth \$1,000,000 at NAV. Six months later, GBTC trades at a **40% premium**. You sell:

$$\$1{,}000{,}000 \times (1 + 0.40) = \$1{,}400{,}000.$$

Your gross gain from the premium alone is **\$400,000** — a 40% return in six months, *before* counting whatever bitcoin did in the meantime. This is why the trade was so popular that it visibly changed GBTC's size: the amount of bitcoin locked inside the trust ballooned as everyone rushed to feed the machine. GBTC's real premium in this era ranged roughly from 10% up to the 40s of percent; it reached as high as ~40% in the 2020–2021 bull run (multiple market-data sources).

**The flip (late February 2021).** Now the machine jams. So much new supply had been minted that, once all those locked shares hit the market — and once competing products (bitcoin futures ETFs, cheaper trusts) arrived — demand for GBTC shares dried up. In **late February 2021**, the premium crossed zero and GBTC slid into a **discount** (CoinDesk). Suddenly the "arbitrage" runs in reverse.

Redo the trade at a **35% discount**. You deposited \$1,000,000 of bitcoin, but the shares you're now allowed to sell fetch:

$$\$1{,}000{,}000 \times (1 - 0.35) = \$650{,}000.$$

Even if bitcoin's price was flat, the *wrapper* just cost you **\$350,000**. And remember: many players did this trade with **borrowed** bitcoin. They owe back the full bitcoin they borrowed, but the shares they're stuck holding are worth far less than the bitcoin's value. That is not a paper annoyance — it is a margin call.

The one-sentence intuition: **the GBTC premium trade was only "arbitrage" as long as the premium existed; the instant it flipped to a discount, the same structure became a leveraged loss with a lockup you couldn't escape.**

### The discount as a force of its own

Once GBTC was in a discount, the discount itself became a market event. Here is the arc, with dated levels.

![GBTC's share price swung from roughly a 40% premium to a nearly 50% discount, then back to par when it converted to an ETF.](/imgs/blogs/pantera-dcg-and-the-crypto-conglomerates-2.webp)

- GBTC traded at a **premium** (10%–~40%) through late 2020 and into early 2021.
- The premium **flipped to a discount in late February 2021** (CoinDesk).
- The discount widened through 2022: about **−36% on 30 September 2022** (CoinDesk), then to a record of **nearly −50% in December 2022**, after the FTX collapse (multiple sources).
- It stayed deeply negative until **11 January 2024**, when GBTC converted into a spot bitcoin ETF, its fee dropped from 2% to 1.5%, and the discount closed to roughly **0% for the first time since February 2021** (CoinDesk; Bloomberg).

Why did the discount matter so much? Because so many big players held GBTC as an *asset* and had borrowed against it. When GBTC is at NAV, a pile of it is worth its full NAV. When GBTC is at a 45% discount, the *same pile* is worth barely half — and if you have to sell it into a thin OTC market, you get even less. That collapse in the value of a widely-held collateral asset is a transmission mechanism all by itself. Hold that thought; it connects directly to Three Arrows Capital.

## The lender in the middle: how Genesis turned a market move into a solvency hole

Grayscale earned a fee no matter what. The part of DCG that could actually *break* was the lender.

### How the lending book worked

Genesis sat between savers and borrowers. On the funding side, its most retail-facing pipe was **Gemini Earn**: a program run with the Gemini exchange where ordinary customers lent their crypto to Genesis in exchange for interest — advertised up to around 8% (NYAG). Genesis pooled that crypto with institutional funding and lent it out to trading firms and funds.

The problem with this model is *concentration and correlation*. A prudent bank spreads its loans across thousands of unrelated borrowers whose fortunes don't move together. Genesis's biggest loans went to a small number of crypto trading firms whose fortunes moved together violently — because they were all long the same asset class, often with the same collateral (including GBTC), in the same cycle. When crypto fell in 2022, they all got in trouble at once.

#### Worked example: how a lender earns its spread — and how thin the cushion is

The economics of a lender are seductive on the way up and brutal on the way down. Let's size both.

Say Genesis borrows **\$1,000,000,000** of crypto from savers (via Earn) at **5%** interest and lends it to trading firms at **9%**. The **spread** — the gap it keeps — is 4 percentage points:

$$\$1{,}000{,}000{,}000 \times (0.09 - 0.05) = \$40{,}000{,}000 \text{ per year}.$$

Forty million dollars a year for standing in the middle. That looks like a wonderful business — until you ask what cushion protects the savers if a borrower doesn't pay. A lender's own capital (its equity) is that cushion. If Genesis held, say, **\$100 million** of equity against a **\$1 billion** book, its cushion is **10%**. Now recall the 3AC loan: reportedly **\$2.36 billion** to a *single* borrower. A default there isn't a 10% dent — it's multiples of the entire cushion. One borrower could vaporize the equity and eat into the savers' money.

That is the arithmetic of thin capital plus concentration: the spread income accrues slowly, in small annual slices, while the loss from one blown-up borrower arrives all at once and is larger than every year of profit combined. A lender can earn its spread flawlessly for years and still be one bad loan from insolvency.

The one-sentence intuition: **a lender's profit is a thin spread collected slowly; its risk is a fat loss delivered all at once — and if a few borrowers are correlated, "diversified" lending isn't.**

The single most important borrower was **Three Arrows Capital (3AC)** — a hedge fund that had leveraged itself across the market, including a giant, once-profitable position built around the GBTC premium trade. (3AC was, at one point, among the largest holders of GBTC.) When the GBTC premium flipped to a discount and the broader market fell — including the Terra/LUNA collapse in May 2022 — 3AC's leverage detonated. We cover 3AC's own story in depth in [Three Arrows Capital and crypto-lender contagion](/blog/trading/crypto/three-arrows-capital-and-crypto-lender-contagion); here we care about what it did to Genesis.

Reporting based on court filings put Genesis's exposure to 3AC at about **\$2.36 billion** (The Block, July 2022), backed partly by collateral including roughly **17.4 million GBTC shares**, plus Grayscale Ethereum Trust shares and AVAX and NEAR tokens — all of which Genesis liquidated. When 3AC couldn't meet a margin call, Genesis sent a notice of default and later filed a claim of about **\$1.2 billion** against 3AC's estate (CoinDesk, 2022-07-18). 3AC filed for bankruptcy on 1 July 2022.

### The intercompany loan that papered over the hole

Here is the moment where the conglomerate structure did something a standalone fund could never do.

Genesis had a roughly billion-dollar hole where the 3AC loan used to be. A hole that size at a lender that owes money to retail savers is an existential threat. So the parent stepped in. In June 2022, **DCG assumed a chunk of Genesis's liability via a \$1.1 billion promissory note** — an IOU from the parent to its subsidiary, due in 2032 — intended to cover the estimated shortfall from 3AC's default (The Block; DCG investor letter). DCG separately had a **\$575 million** loan from Genesis (due 2023). Together, DCG owed Genesis on the order of **\$1.6–1.7 billion** in intercompany obligations.

![How one borrower's default climbed up the stack: retail money flowed down into over-levered borrowers; when they defaulted, the loss climbed back up to the parent.](/imgs/blogs/pantera-dcg-and-the-crypto-conglomerates-4.webp)

Trace the money in the figure. Retail savers lent to Genesis. Genesis lent to 3AC (and Alameda and others). 3AC defaulted. That left a hole in Genesis. And the parent, DCG, filled the hole with a promise to pay — a note. On paper, Genesis's balance sheet now looked whole: where a defaulted \$2.36 billion loan had been, there was now a \$1.1 billion IOU from a \$10-billion-valued parent.

But swap out *what kind of asset* Genesis was holding. Before, it (nominally) held a loan to a third party. After, it held a claim on *its own parent*. That is a related-party transaction at the center of the balance sheet. And whether that IOU was worth its face value depended entirely on the parent's ability to pay — a judgment the retail savers on the other side of Genesis's book were in no position to make, because they couldn't see it.

#### Worked example 2: the intercompany-loan chain, step by step

Let's make the chain explicit with the reported figures. (These are real reported magnitudes; the arithmetic is here to show the mechanism, not to audit anyone's books.)

1. **Retail in.** ~230,000+ Gemini Earn users had lent Genesis crypto worth about **\$900 million** (NYAG count of users; SEC put the assets at ~\$900M from ~340,000 Earn investors). Genesis owes this — it is a real liability to real people.
2. **Institutional out.** Genesis had lent **\$2.36 billion** to 3AC (The Block).
3. **The default.** 3AC collapses. Genesis's claim is ~**\$1.2 billion** after liquidating collateral (CoinDesk). Call the residual hole **~\$1.1 billion**.
4. **The patch.** DCG issues a **\$1.1 billion** promissory note to Genesis, due **2032**. The hole is now "filled" — with an IOU from the parent.
5. **The catch.** Genesis's ability to repay its ~\$900 million to retail now depends on collecting a related-party note due nearly a decade later, plus whatever else it can recover. A ten-year IOU does not pay a saver who wants their coins back *this month*.

The one-sentence intuition: **an intercompany note can make a balance sheet balance without producing a single dollar of cash — and cash is exactly what a lender needs when its depositors ask for their money back.**

### The run, and the halt

That last point is the whole ballgame. In November 2022, when FTX collapsed and panic swept crypto, Gemini Earn users (and other Genesis lenders) tried to withdraw. Genesis couldn't meet the requests — it lacked the liquid assets, because its assets were impaired loans and a long-dated related-party note, not cash. On **16 November 2022**, Genesis halted withdrawals. About **\$900 million** of Gemini Earn customers' assets were frozen (SEC; press reporting).

#### Worked example 3: how one entity's loss propagates to a sister entity

Now watch the loss jump sideways, from the lender to the savers, and see why the conglomerate structure made it worse.

Imagine Genesis's simplified book at the moment of the freeze:

- **It owes:** ~\$900 million to Gemini Earn retail (due basically on demand), plus other institutional creditors.
- **It is owed:** a defaulted/impaired 3AC loan; a ~\$1.1 billion related-party note from DCG due 2032; and other loans of varying quality.

![Genesis's book: it owed real cash to retail savers while its biggest claims sat with a defaulted fund and its own parent.](/imgs/blogs/pantera-dcg-and-the-crypto-conglomerates-6.webp)

The figure shows the mismatch. On the right — a *real, near-term, cash* obligation to retail. On the left — the biggest claims are (a) impaired and (b) a long-dated IOU from the parent. A saver asking for \$5,000 back cannot be paid with "a claim on DCG due in 2032." The assets and the liabilities are mismatched not just in *amount* but in *time* and *quality*.

Now the sister-entity twist. In a standalone lender, a hole is a hole and the market prices it. In a conglomerate, the parent has two conflicting instincts: protect the retail-facing subsidiary (good) and protect the parent's own balance sheet and equity value (self-interested). The New York Attorney General's later complaint *alleged* that DCG and its executives concealed Genesis's true financial condition and misled Earn investors about the risks — framing the \$1.1 billion note in ways that made Genesis look healthier than it was. Those are **allegations**, contested by DCG; we'll cover the outcomes below. The structural point stands regardless of intent: when the lender and the entity that filled its hole answer to the same owner, the people who can least afford a loss — retail savers — are the least able to see it coming.

## Mark versus liquidity: why "we're covered" wasn't true

There's one more concept that turns a bad quarter into insolvency, and it applies to every locked or oversized position in this saga: the gap between what an asset is *marked* at and what it can actually be *sold* for.

### The mark is not the money

When you hold an asset, your books carry it at a **mark** — an accounting value, usually based on a reference price like NAV or the last quote. But if you actually need to *sell* — especially a large or illiquid position, into a thin market, under pressure — the price you get can be dramatically lower. Two forces open the gap:

1. **The discount / market price is below the mark.** A pile of GBTC "worth" its NAV is only sellable at the discounted share price — nearly 50% below NAV at the December 2022 lows.
2. **Slippage from size and thin liquidity.** Dumping a huge block into a shallow order book (or even an OTC market) pushes the price down as you sell. The more you must sell at once, the worse your average price.

![Mark vs. liquidity: on a locked or oversized position, the value on the books and the cash you can actually raise are two different numbers.](/imgs/blogs/pantera-dcg-and-the-crypto-conglomerates-5.webp)

#### Worked example 4: the mark-vs-liquidity gap on a locked position

Suppose you (or a lender's risk model) carry **\$1,000,000** of GBTC on the books at NAV. That's the blue bar — the mark. Now reality intrudes:

- GBTC trades at a **−45% discount**, so the most you can get for it *right now* is about **\$550,000** (the amber bar). The mark just lost nearly half its value with no change in the bitcoin inside.
- You're not the only forced seller, and the OTC market for a big block is thin, so a rushed sale realizes maybe **\$480,000** (the red bar) after slippage.

Your \$1,000,000 "asset" raises \$480,000 in cash when you actually need it. The **\$520,000 gap** is *paper wealth* — real on a spreadsheet, gone at the moment of truth.

This is exactly the trap 3AC's GBTC position became, and exactly why Genesis's collateral didn't cover the loans it backed. When Gemini later sued Genesis, it did so over roughly **30.9 million GBTC shares** — collateral it valued at about **\$1.6 billion** (CoinDesk, 2023-10) — precisely because the mark on that collateral and the cash it could raise had diverged. Everyone in the chain was "covered" at the mark and underwater at the exit.

The one-sentence intuition: **a mark is a promise the market makes in calm weather; liquidity is what it actually pays you in a storm — and leverage is settled in the storm.**

## How it shows up in price

Everything above eventually lands in the price *you* see on a screen. Here is the transmission, made concrete.

### The GBTC discount as a market force

A deep, persistent discount does three things to the wider market:

- **It signals stress and forces behavior.** A 45% discount screams that big holders are trapped and the wrapper is unwanted. Funds marking GBTC at NAV look solvent; funds marking it at market look wounded — and the ones who *must* sell (to meet margin or redemptions) crystallize the loss and push the discount wider still. It is reflexive: selling widens the discount, which forces more selling.
- **It sets up a coiled spring.** Because the discount could only close if GBTC ever became redeemable (i.e., converted to an ETF), the discount was also a bet on a regulatory outcome. When that outcome arrived in January 2024, the discount snapped shut — and the flip from "trapped at a discount" to "redeemable at NAV" produced its own large flows as arbitrageurs and trapped holders finally got out.
- **It moves bitcoin itself.** After conversion, holders who had been stuck for years finally could exit at NAV, and many did — GBTC saw heavy outflows, and because redemptions meant selling the underlying bitcoin, that selling pressure showed up in bitcoin's spot price. A structural feature of one wrapper became a real bid/offer in the whole market. We trace that bridge in [Bitcoin ETFs and the TradFi bridge](/blog/trading/crypto/bitcoin-etfs-and-the-tradfi-bridge).

### Forced selling and the slippage tax

When a lender defaults on its obligations, its collateral gets liquidated — and liquidation is *forced* selling, the worst kind. Genesis liquidated 3AC's collateral (GBTC, ETHE, AVAX, NEAR). Selling millions of shares and millions of tokens in a falling, thin market is a slippage tax paid straight into the price: each sale prints lower, dragging the quote down for everyone, including holders who had nothing to do with the loan.

Picture the order book for a moment. Suppose the market shows buyers for 100,000 tokens at \$20, another 100,000 at \$19, another at \$18, and so on down. If you must sell 500,000 tokens *now*, you don't get \$20 for all of them — you "walk the book" down through each level, and your *average* fill might be \$18, not the \$20 you saw quoted. The larger and more urgent your sale, the deeper you walk, and the worse your average. A forced seller of a defaulted borrower's collateral is the textbook worst case: maximum size, maximum urgency, minimum choice. This is the mechanism by which *one fund's leverage* becomes *every holder's lower price*. If you were long AVAX or NEAR in mid-2022 and wondered why the floor kept dropping, part of the answer was a lender dumping a defaulted borrower's collateral into a book that couldn't absorb it.

### The retail-defense takeaway

If you are on the other side of these trades — the saver, the small holder — the defensive lessons are concrete:

- **A yield is a loan.** "Earn 8%" means *you are lending your coins to someone who lends them to someone riskier*. Ask who the ultimate borrower is and what happens if they don't pay. If you can't find out, that opacity is the answer.
- **Redeemability is everything.** Prefer wrappers you can redeem at NAV (open-end ETFs) over closed-end trusts where you're at the mercy of a discount. The GBTC discount cost trapped holders real money for three years.
- **Follow the ownership map.** When one group owns your yield product, the trust, the lender, and the desk, the walls that are supposed to protect you may not exist. [Cui bono — the incentive map of crypto](/blog/trading/crypto-players/cui-bono-the-incentive-map-of-crypto) is the tool for reading who profits at each step. And the broader map of who moves crypto prices lives in [crypto VCs and market makers](/blog/trading/crypto/crypto-vc-and-market-makers).

## Common misconceptions

**"A discount means the fund is a bargain — just buy it below NAV."** Sometimes, but not automatically. In a closed-end trust with no redemption, a discount can persist or *widen* indefinitely — there's no mechanism forcing convergence. You only reliably capture a discount if a catalyst (like ETF conversion) forces it to close. GBTC's discount lasted from February 2021 to January 2024; a "bargain" buyer in mid-2022 could have watched it deepen toward −50% first.

**"Grayscale/GBTC was the thing that blew up."** No. GBTC itself never missed a beat — it always held its bitcoin, and Grayscale kept collecting its 2% fee. What broke was **Genesis**, the *lender*. The connection is that GBTC was widely used as *collateral* and as the engine of the leveraged premium trade, so its discount hurt the borrowers, not the trust. Owning the trust was the safe, profitable part; owning the lender was the dangerous part.

**"The \$1.1 billion note means DCG paid Genesis \$1.1 billion."** A promissory note is a *promise* to pay, not a payment. It filled a hole on the balance sheet without moving cash. That distinction — accounting solvency versus actual liquidity — is precisely what left Genesis unable to honor withdrawals.

**"Retail savers were speculating and knew the risk."** Many Gemini Earn users understood it as a simple, high-yield savings product. The regulators' central *allegation* was that the risks — especially Genesis's true condition after 3AC — were concealed or downplayed. Whatever the ultimate legal findings, the structural reality is that a retail saver could not see the intercompany plumbing that determined whether they'd be repaid.

**"A conglomerate is safer because it's diversified."** Diversification helps only if the parts are *uncorrelated*. DCG's arms were all levered to the same crypto cycle and often the same assets (GBTC). Correlated diversification isn't diversification — it's concentration wearing a costume, with extra pipes for a loss to travel through.

**"Pantera avoided all this because it was smarter."** Pantera avoided the *specific* contagion mostly because of *structure*, not superior foresight — it didn't run a retail-funded lender whose hole could sink savers. A fund can still lose enormous sums (many crypto funds did in 2022). The point isn't that funds are safe; it's that a fund has fewer sideways pipes than a conglomerate.

## How it shows up in real markets

Below are the concrete episodes, with dates and sourced figures, where these mechanics played out.

### 1. The GBTC premium machine and its reversal (2020–2021)

Through 2020 and into early 2021, the GBTC premium — as high as ~40% — turned depositing bitcoin into GBTC into a lucrative, popular trade, ballooning the bitcoin locked in the trust. Big funds, including 3AC, piled in, often with leverage. Then, in **late February 2021**, the premium flipped to a discount (CoinDesk). The trade that had minted money became a leveraged loss no one could exit for six months at a time. The premium machine didn't just enrich its users on the way up; it *created* the trapped, levered positions that would detonate on the way down. It was, quite literally, the setup for the collapse.

### 2. Terra, then 3AC (May–July 2022)

The Terra/LUNA collapse in May 2022 vaporized tens of billions and cracked the leverage across crypto. 3AC — long the GBTC trade and much else, all levered — couldn't survive. Genesis, reportedly its lender to the tune of **\$2.36 billion** (The Block), issued a margin call, then a default notice, then liquidated 3AC's collateral (17.4 million GBTC shares and more) and filed a **~\$1.2 billion** claim (CoinDesk, 2022-07-18). 3AC filed for bankruptcy on **1 July 2022**. The lender at the center of DCG's group now had a billion-dollar-plus hole.

### 3. The \$1.1 billion note (June 2022)

Rather than let Genesis show the hole, **DCG assumed liability via a \$1.1 billion promissory note due 2032** (The Block; DCG's own investor letter later disclosed roughly \$2 billion of intercompany obligations, including this note and a \$575 million loan). This is the pivotal related-party move: the parent used its own credit to keep the subsidiary's balance sheet whole on paper. It bought time. It did not produce cash.

### 4. FTX, the run, and the halt (November 2022)

When FTX collapsed in **November 2022**, confidence evaporated and Genesis's lenders — including Gemini Earn's retail users — rushed for the exits. Genesis couldn't pay: its assets were impaired loans and a long-dated related-party note, not liquid cash. On **16 November 2022**, it halted withdrawals, freezing about **\$900 million** of Gemini Earn customers' assets (SEC; press reporting). At the same moment, GBTC's discount hit its record low of nearly **−50% (December 2022)**, marking down the very collateral the whole structure leaned on.

### 5. Bankruptcy and the regulators (January–October 2023)

Genesis filed for **Chapter 11 bankruptcy on 19 January 2023** in the Southern District of New York, listing more than 100,000 creditors and liabilities it estimated between **\$1.2 billion and \$11 billion**. Days earlier, on **12 January 2023**, the **SEC charged Genesis and Gemini** with the unregistered offer and sale of securities through Gemini Earn. Then, on **19 October 2023**, **New York Attorney General Letitia James sued Gemini, Genesis, and DCG** (and executives including Barry Silbert), *alleging* they defrauded more than **230,000 investors** of over **\$1 billion** by concealing Genesis's condition; the suit was later expanded against DCG and Silbert in 2024. DCG has **denied wrongdoing** and contested the allegations. These are the *allegations and filings* — not established findings against DCG.

### 6. The settlements and the ETF (2024)

2024 brought resolutions, reported as follows:

- **Gemini** agreed (settlement with the NY Department of Financial Services, **28 February 2024**) to return **at least \$1.1 billion** to Earn customers and pay a **\$37 million** fine, with Earn users to receive their assets back in kind.
- **Genesis** settled with the NY Attorney General for **\$2 billion** (announced **May 2024**) to compensate defrauded victims — the largest such settlement against a crypto firm in the state's history — and separately agreed to a **\$21 million** SEC penalty (SEC, 2024).
- **Genesis emerged from bankruptcy in August 2024**, ultimately distributing on the order of **\$4 billion** to creditors (helped enormously by crypto prices recovering).
- **GBTC converted to a spot ETF on 11 January 2024**, cut its fee from 2% to 1.5%, and its discount closed to ~0% for the first time since February 2021 (CoinDesk; Bloomberg). The one-way door finally opened both ways — and the discount that had defined the crisis simply evaporated.

### 7. The counterfactual: Pantera's contained losses

Pantera was not immune to 2022 — crypto funds broadly took heavy marks as the market fell. But there was no Pantera "Genesis moment," no retail savings program frozen by a Pantera entity, no intercompany note holding up a Pantera lender. The losses that happened were fund losses, borne by fund investors, inside the funds. Same terrible market, far smaller blast radius — because the structure had fewer pipes. That is the entire thesis of this post, delivered by contrast: a bad bet hurts a fund's investors; a bad bet *plus* a conglomerate's plumbing can hurt everyone downstream.

![From pioneer fund to contagion: the same plumbing that built the conglomerate — trust and lender — is what transmitted the 2022 shock.](/imgs/blogs/pantera-dcg-and-the-crypto-conglomerates-7.webp)

The timeline ties it together: a decade that began with Pantera pioneering a simple bitcoin fund in 2013 ended, on the DCG side, with the exact machine that built the empire — the trust and the lender — becoming the channel that transmitted a market shock into a solvency crisis and a regulatory reckoning.

## When this matters to you

You may never lend to a Genesis or buy a GBTC at a premium. But the pattern is everywhere in crypto and beyond, and recognizing it protects you.

- **When someone offers you yield, map the chain.** A savings-like product paying well above risk-free rates is lending your money to a borrower you can't see. Ask: who is the ultimate borrower, what's the collateral, and what happens in a default? A firm that won't answer is telling you something.
- **Prefer redeemable wrappers.** A discount you can't escape is a tax on being trapped. Structures with a real redemption mechanism (ETFs) keep price glued to value; closed-end trusts don't. The GBTC discount was, for three years, the difference between "worth its NAV" and "sellable for half."
- **Watch for one owner holding many hats.** The single most useful question about any crypto (or finance) group is: *does the same entity own the fund, the trust, the lender, and the desk?* Every additional hat is a wall that isn't there — a place a loss can travel and a conflict that has no arm's-length check. That's the map [Cui bono](/blog/trading/crypto-players/cui-bono-the-incentive-map-of-crypto) teaches you to read.
- **Distinguish accounting solvency from liquidity.** "The balance sheet balances" is not the same as "there's cash to pay you." Intercompany notes, marked-not-realized collateral, and long-dated IOUs can all make a balance sheet look whole while leaving nothing to hand a depositor who wants out. In a run, only cash counts.

The deeper lesson is old and not crypto-specific: financial systems put walls between roles for a reason, and the reason is that concentration creates conflicts and pipes that only reveal themselves in a crisis. DCG's story is a clean, recent, fully-documented case of what happens when those walls come down. Pantera's quieter story is the control group. Neither is a villain-and-hero tale — it's a structure-and-consequence one, and structure is something you can actually check before you trust anyone with your money.

## Sources & further reading

Primary sources and reporting behind the headline figures (with as-of context):

- **SEC**, *Genesis Agrees to Pay \$21 Million Penalty to Settle SEC Charges* (2024) and the SEC's January 12, 2023 charges re: Gemini Earn — the ~\$900 million from ~340,000 Earn investors, and the unregistered-securities claim. [sec.gov](https://www.sec.gov/newsroom/press-releases/2024-37)
- **New York Attorney General**, *AG James Sues Gemini, Genesis, and DCG* (October 19, 2023) and *AG James Secures Settlement Worth \$2 Billion from Genesis* (May 2024) — the 230,000+ investors / \$1B+ fraud allegations and the settlement. [ag.ny.gov](https://ag.ny.gov/press-release/2024/attorney-general-james-secures-settlement-worth-2-billion-crypto-firm-genesis)
- **NYDFS / Gemini settlement** (February 28, 2024) — Gemini to return at least \$1.1 billion to Earn users, pay a \$37 million fine (Decrypt; The Block; Banking Dive).
- **The Block**, *Crypto lender Genesis lent \$2.36 billion to Three Arrows Capital* (July 2022) and *DCG is debt-free, minus its \$1.1 billion promissory note to Genesis* — the intercompany note and 3AC exposure.
- **CoinDesk**, *Genesis Files \$1.2B Claim Against Three Arrows Capital* (2022-07-18); *Grayscale's GBTC Discount Closes to Zero for First Time Since February 2021* (2024-01-11); *Gemini Sues Bankrupt Lender Genesis Over \$1.6B Worth of GBTC* (2023-10-27); GBTC discount levels through 2022.
- **CNBC / CoinDesk**, *Digital Currency Group tops \$10 billion valuation* (November 1, 2021) — the SoftBank/Alphabet secondary round.
- **Pantera Capital** — firm and fund pages; *Pantera Bitcoin Fund Hits 1,000x* (November 2024) for the 131,165% lifetime figure and AUM history. [panteracapital.com](https://panteracapital.com/)
- **Wikipedia** — *Digital Currency Group* and *Pantera Capital* for corporate structure and founding dates (cross-checked against the primary sources above).

Related deep dives on this blog:

- [Crypto VCs and market makers](/blog/trading/crypto/crypto-vc-and-market-makers) — the series hub: who actually moves crypto prices.
- [Three Arrows Capital and crypto-lender contagion](/blog/trading/crypto/three-arrows-capital-and-crypto-lender-contagion) — the borrower whose default started this chain.
- [Cui bono — the incentive map of crypto](/blog/trading/crypto-players/cui-bono-the-incentive-map-of-crypto) — how to read who profits at each step of the stack.
- [Bitcoin ETFs and the TradFi bridge](/blog/trading/crypto/bitcoin-etfs-and-the-tradfi-bridge) — what happened when GBTC finally became redeemable.

*Educational content on market structure and a documented, litigated episode — not investment advice, and not an allegation of wrongdoing beyond what the cited filings state.*
