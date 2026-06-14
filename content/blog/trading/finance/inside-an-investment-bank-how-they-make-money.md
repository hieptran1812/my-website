---
title: "Inside an Investment Bank: How Goldman Sachs and JPMorgan Actually Make Money"
date: "2026-06-15"
publishDate: "2026-06-15"
description: "A plain-English tour of the four divisions inside an investment bank, exactly how each one earns its money, and why thirty-to-one leverage turned 2008 into a catastrophe."
tags: ["investment-banking", "goldman-sachs", "jpmorgan", "ipo", "underwriting", "market-making", "leverage", "financial-crisis", "volcker-rule", "financial-institutions"]
category: "trading"
subcategory: "Finance"
author: "Hiep Tran"
featured: true
readTime: 41
---

> [!important]
> **TL;DR** — An investment bank is not one business; it is four distinct money-making engines under one roof, each earning a different way: advisory earns fees, underwriting earns a spread, trading earns the bid-ask gap, and asset management earns a slice of the money it runs.
>
> - The four engines are the Investment Banking Division (advice on mergers and on raising money), Global Markets (buying and selling securities all day), Research (rating stocks and bonds), and Asset & Wealth Management (running other people's money for a fee).
> - There are exactly three ways the whole place makes money: a *fee* for advice, a *spread* for handling a transaction, and a *return* on the bank's own capital when it takes risk. Almost everything else is a variation on these three.
> - A \$1 billion IPO at a 4% gross spread hands the underwriters \$40 million. A market maker quoting a two-cent spread on 10,000 shares pockets \$200 a round-trip. A 1% advisory fee on a \$5 billion merger is \$50 million.
> - The thing that turned 2008 into a catastrophe was *leverage*: at 30-to-1, the bank holds \$1 of its own money for every \$30 of assets, so a 3.3% fall in those assets wipes the firm out. Bear Stearns and Lehman were not unlucky — they were structurally fragile.
> - The one number to remember: about 30-to-1. That ratio is the difference between a profitable middleman and a smoking crater.

Why does Goldman Sachs exist? Most people have a vague sense that it is rich, powerful, and somehow always at the center of things, but if you asked them to explain *what it actually does to earn money*, the answer dissolves into a fog of "investments" and "Wall Street." That fog is the problem this post clears. An investment bank is not a mysterious money machine and it is not, despite the cartoon, a casino where bankers gamble with your checking account. It is a collection of very different businesses that happen to share a name, a balance sheet, and a culture — and once you can name the businesses, the profits stop being mysterious and the 2008 collapse stops being a freak accident and becomes something almost inevitable in hindsight.

The diagram above is the mental model for the whole post: one firm, four engines, three ways to earn. Hold that picture in your head and everything that follows is just detail poured into it.

![Tree of an investment bank split into four divisions](/imgs/blogs/inside-an-investment-bank-how-they-make-money-1.png)

We will build this from absolute zero. By the end you will be able to read a bank's earnings report and know which engine produced the number, explain to a friend exactly how a company "goes public" and what the bank skims off the top, and understand why a regulator spends so much energy on a single ratio. None of this is financial advice — it is a map of how a corner of the financial world is built so that the news makes sense.

## The basics: what an investment bank is, and what it is not

Let us start by clearing up the single most common confusion, because almost everything else depends on it.

### Investment bank vs commercial bank

A *commercial bank* — sometimes called a retail bank — is the one you already know. It takes *deposits* (the money you park in checking and savings) and makes *loans* (mortgages, car loans, business credit lines). It earns the difference between the interest it charges borrowers and the lower interest it pays depositors. That gap is its *net interest margin*. When you think "Chase branch on the corner," "FDIC insurance on my savings," or "swipe my debit card," you are thinking of a commercial bank.

An *investment bank* does almost none of that. Historically it had no branches, took no insured deposits, and made few ordinary loans. Instead it sits between *companies and governments that need money or advice* on one side, and *large investors* on the other, and it earns by being the indispensable middleman in the biggest, most complex transactions in the economy: a company buying another company, a startup selling shares to the public for the first time, a government issuing billions of dollars of bonds, a pension fund trying to buy or sell a huge block of stock without moving the price against itself.

Here is the wrinkle that trips everyone up. After 2008, the two surviving pure investment banks — Goldman Sachs and Morgan Stanley — legally converted into *bank holding companies*, and the giant commercial banks — JPMorgan, Bank of America, Citigroup — already owned enormous investment-banking arms. So today's reality is that most household names are *universal banks*: they have a commercial bank and an investment bank under one corporate roof. JPMorgan takes your deposits *and* advises on multibillion-dollar mergers. When this post says "investment bank," it means the investment-banking *activity* — the four engines — regardless of which legal shell it lives inside.

### Sell-side vs buy-side

The whole financial world divides into two camps, and the divide is more useful than any other.

The *sell-side* is the banks and dealers. They create financial products, bring new securities to market, make markets in them, and sell research and execution services. Goldman, JPMorgan, Morgan Stanley — sell-side. They are called the sell-side because, historically, they were *selling* — selling new stock issues to investors, selling research, selling trade execution.

The *buy-side* is the institutions that manage pools of money and *buy* those products: pension funds, mutual funds, hedge funds, insurance companies, sovereign wealth funds, university endowments. They are the customers. When a hedge fund wants to buy a billion dollars of bonds, it calls the sell-side to get it done. (We cover the buy-side's biggest players in detail in [the field guide to financial institutions](/blog/trading/finance/field-guide-to-financial-institutions) and [how hedge funds work](/blog/trading/finance/how-hedge-funds-work-leverage-2-and-20).)

An investment bank is the archetypal sell-side firm. Keep that fixed: the bank is the shop, the buy-side funds are the shoppers, and most of the bank's revenue is some form of payment for letting the shoppers do something they could not easily do alone.

### Primary market vs secondary market

This distinction is the hinge of the entire business, so we will be slow about it.

The *primary market* is where securities are *created* — sold for the first time, from the issuer (the company or government) to investors. When Airbnb sold shares to the public in December 2020, that first sale was the primary market: the cash flowed *from investors into Airbnb*. The company actually got money it could spend.

The *secondary market* is where those already-existing securities trade *between investors* — investor A sells to investor B. When you buy an Airbnb share today on your brokerage app, Airbnb the company gets *nothing*; you are buying from some other investor who wants to sell. The cash flows sideways, investor to investor. The stock exchange is a secondary market.

![Primary market issuing versus secondary market trading](/imgs/blogs/inside-an-investment-bank-how-they-make-money-3.png)

Why does this matter so much? Because the investment bank earns *differently* in each. In the primary market it earns by *underwriting* — managing the new issue and taking a cut of the money raised. In the secondary market it earns by *making markets* — standing ready to buy and sell, and pocketing the small gap between its buy price and its sell price. Two markets, two completely different money engines. We will return to both.

### The four words you must own

Before the deep part, let us define four terms inline, because every section leans on them.

*Underwriting* is when a bank takes on the job — and often the risk — of selling a new issue of securities. In the strongest form, called a *firm-commitment* underwriting, the bank actually *buys the whole issue from the company* at an agreed price and then resells it to investors. If it cannot sell it all, the bank is stuck holding the rest. The bank's profit is the gap between what it pays the issuer and what it sells to investors for — the *underwriting spread*, also called the *gross spread*.

The *bid-ask spread* (or bid-offer spread) is the gap between the price at which someone will *buy* from you (the *bid*) and the price at which they will *sell* to you (the *ask* or offer). If a dealer is willing to buy a stock at \$49.99 and sell it at \$50.01, the spread is \$0.02. Whoever stands in the middle quoting both prices captures that two cents every time a buyer and a seller pass through.

A *market maker* is exactly that someone-in-the-middle: a dealer who continuously quotes both a bid and an ask in a security, promising to trade at those prices, and earns the spread for providing *liquidity* — the ability for others to trade instantly without waiting for a matching counterparty to show up.

*Leverage* is using borrowed money to control more assets than your own capital alone could. If you have \$1 and borrow \$30 to control \$31 of assets, your leverage is 31-to-1 (often rounded to "about 30-to-1"). Leverage multiplies both gains and losses on your own money. This single concept is the villain of the 2008 section.

One more pairing. When a bank acts as a *principal*, it is using its *own* money and taking the risk onto its *own* balance sheet — it owns the security, so it gains or loses if the price moves. When it acts as an *agent* (or *broker*), it is merely arranging a transaction *for someone else* and collecting a fee or commission, never owning the security or bearing its price risk. Advisory work is pure agency: the bank gives advice and bills a fee, full stop. Trading can be either — an *agency* trade just matches a client to the market, while a *principal* trade means the bank itself is the counterparty, putting its capital at risk. The mix of agency and principal across the bank is, more than anything, what determines how much can go wrong.

## The four divisions, one at a time

Now the engines. A modern investment bank reports its results in roughly four segments. The names differ slightly between firms — Goldman calls them by one set of labels, JPMorgan another — but the functions are the same everywhere. We will use generic names and note the variations.

### Division 1: the Investment Banking Division (IBD)

This is the part most people picture when they hear "investment banker," and confusingly it shares its name with the whole firm. The Investment Banking Division — IBD — is the *advisory and capital-raising* business. It does two big things.

First, **mergers and acquisitions advisory (M&A)**. When one company wants to buy another, or a company wants to sell itself, or a board needs to know whether a takeover offer is fair, it hires bankers. The bank's M&A team analyzes the businesses, builds the financial models, helps negotiate the price and structure, manages the months-long process, and — crucially — provides a *fairness opinion*, a formal letter saying the price is reasonable, which protects the board from being sued. For this the bank charges an *advisory fee*, usually a percentage of the deal size. The bank takes essentially no principal risk here. It is pure agency: brains and relationships rented out for a fee.

Second, **capital raising**, which means helping a company or government sell new securities to raise money. This splits into two desks:

- **Equity Capital Markets (ECM)** handles selling *stock* — initial public offerings (IPOs, the first time a company sells shares to the public), follow-on offerings (selling more shares after already being public), and convertible securities. ECM is where the IPO machinery lives.
- **Debt Capital Markets (DCM)** handles selling *bonds* — corporate bonds, government bonds, and other debt. When Apple wants to borrow \$10 billion from investors by issuing bonds rather than going to a bank for a loan, DCM bankers structure, price, and sell that bond.

In both ECM and DCM the bank *underwrites* the issue and earns the gross spread — the cut described above. So IBD has two revenue styles: a *fee* for advice (M&A), and a *spread* for underwriting (ECM/DCM). Both are fee-like and capital-light; the bank is mostly selling expertise, relationships, and the credibility of its name.

It is worth pausing on why a company would even hire a bank to sell its bonds rather than just selling them itself. Three reasons. First, the bank already has a *distribution network* — a Rolodex of pension funds, insurers, and bond funds that buy new issues — so it can place a large issue in a single afternoon, where the company alone might take weeks and never reach the right buyers. Second, the bank provides *price discovery*: it knows, from constant contact with those buyers, what yield investors will demand today, so it can set the coupon (the bond's interest rate) at exactly the level that clears the issue without overpaying. Third, the bank lends its *credibility*: a deal led by a top-tier name signals to cautious investors that the disclosure has been vetted. For all of this the bank earns the underwriting spread, which on investment-grade bonds is far smaller than on equity — usually a fraction of a percent rather than the 4–7% of an IPO, because bonds are less risky to place and the buyers are repeat institutions.

#### Worked example: the underwriting fee on a \$500 million bond deal

A solid company wants to borrow \$500,000,000 by issuing 10-year corporate bonds, and it hires a bank's DCM desk to underwrite the deal. Investment-grade bond underwriting fees are small — assume a gross spread (here called the *underwriting fee* or *concession*) of 0.5%.

- Underwriting fee: 0.5% of \$500,000,000 = \$2,500,000.
- The company receives roughly \$500,000,000 - \$2,500,000 = \$497,500,000 in proceeds (before its own legal and rating-agency costs).
- The bank earned \$2,500,000 for structuring, pricing, and placing the bonds with its institutional buyers, typically over a few days of marketing.

Notice how much *thinner* this spread is than an IPO's 4%: a 0.5% bond fee versus a 4% equity fee on the same nominal size would be \$2.5 million versus \$40 million. Intuition: underwriting fees scale with how hard and risky the security is to place — a routine investment-grade bond sold to repeat buyers is cheap to underwrite, while a first-time equity offering to a fickle public commands many times the spread.

This is also why DCM is a high-volume, lower-margin business and ECM is a lower-volume, higher-margin one. A bank's DCM desk might underwrite hundreds of bond deals a year at half a percent each; its ECM desk does far fewer IPOs but earns a much fatter percentage on each. Both are *spread* income, capital-light, and both depend on the same underlying asset: the bank's relationships with the buy-side institutions that actually take down the paper.

### Division 2: Global Markets (sales and trading)

If IBD is the brains-for-hire business, Global Markets is the *buying-and-selling-all-day* business. This is the trading floor — rows of screens, the place movies love. Its job is to *make markets*: to stand ready to buy or sell securities for the bank's buy-side clients, instantly, at quoted prices, and to earn the bid-ask spread for providing that liquidity. It is conventionally split in two.

- **FICC** stands for **Fixed Income, Currencies, and Commodities**. "Fixed income" means bonds and interest-rate products; "currencies" means foreign exchange (FX); "commodities" means oil, gold, gas, and the like, plus the derivatives on all of these. FICC is usually the larger of the two halves at most banks, because the bond, rate, and FX markets are enormous and trade in huge size.
- **Equities** is the stock side: trading shares and equity derivatives (options, futures, swaps on stocks and indices), plus *prime brokerage* — the package of lending, custody, and trade-execution services banks sell to hedge funds.

Global Markets has two flavors of revenue. The dominant, sustainable one is the **spread**: quoting both sides and capturing the gap, multiplied by gigantic volume. The other is **principal risk** — the bank using its own capital to take positions, hoping prices move its way. Some of that is unavoidable: a market maker that just bought a million bonds from a client *owns* those bonds for a while ("*inventory*") and bears their price risk until it can sell them on. Some of it used to be deliberate *proprietary trading* ("*prop*"): the bank betting its own money purely for its own profit, like an in-house hedge fund. As we will see, regulation later forced most of the deliberate prop trading out. The spread is the heart of the business; the principal risk is the part that occasionally blows up.

The division has two kinds of people who are easy to confuse. *Sales* are the relationship people who talk to the buy-side clients all day — they know which funds want what, they pitch trade ideas, and they bring the orders in. *Trading* are the people who actually quote the prices and manage the bank's inventory and risk. A salesperson takes a hedge fund's call and says "we can do that block"; a trader decides at what price the bank is willing to be the other side, and how to hedge the resulting position. The two work as a pair: sales owns the relationship and the order flow, trading owns the price and the risk. Their combined output is *commissions* (on agency trades, where the bank just executes for the client and charges a flat fee) and *spread* (on principal trades, where the bank itself is the counterparty). In the modern, electronic, razor-thin-spread world, both have been squeezed hard, which is why this division's revenue swings violently with market volatility — a calm year is lean, a volatile year (more trading, wider spreads) can be a bonanza.

One more concept belongs here because it explains a lot of the headlines: *derivatives*. A derivative is a contract whose value derives from something else — a stock, a rate, a currency, a commodity. Options, futures, and swaps are all derivatives. Banks make markets in derivatives too, and they are especially profitable because they are complex, customized, and harder for a client to price independently — which means the bank can earn a wider spread. They are also where principal risk concentrates, because a derivative can carry enormous *notional* exposure (the face amount the contract references) for a small upfront cost, so a position that looks modest can hide a very large bet. The London Whale episode we will cover later was a derivatives position that grew monstrous; the 2008 meltdown ran through mortgage derivatives. Derivatives are the high-octane fuel of Global Markets: lucrative, useful for hedging, and the place where the principal-risk engine most often catches fire.

### Division 3: Research

Research is the smallest revenue line and the most misunderstood. A bank's research analysts publish reports on companies, industries, bonds, currencies, and the economy, with ratings like "buy," "hold," or "sell" and price targets. Buy-side clients read this research, and it helps them decide what to trade — which they then do *through the bank's trading desk*, generating spread revenue. So research does not usually charge directly; it is a magnet that pulls trading business toward the bank.

The reason research matters out of proportion to its size is *conflict of interest*. The same firm that publishes a glowing report on a company might also be earning fat underwriting fees from that company. After the dot-com bust, regulators forced a legal *wall* between research and banking — analysts can no longer be paid based on banking deals, and the two sides are supposed to be insulated. We will revisit this; for now, file research under "information service that supports the markets engine, fenced off from the banking engine."

### Division 4: Asset and Wealth Management (AWM)

The fourth engine is the calmest and, increasingly, the most prized. *Asset management* means running investment funds for institutions — mutual funds, ETFs, pension mandates. *Wealth management* means managing the money of rich individuals and families (the "private bank"). In both cases the bank invests *other people's money* and charges a *management fee*, usually a small annual percentage of the *assets under management* (AUM) — the total pile it runs. Sometimes it also earns a *performance fee* on the gains.

Why is this engine prized? Because the revenue is *recurring and stable*. A management fee on a few hundred billion dollars of AUM arrives every year, in good markets and bad, without the bank having to win a new deal or take a trading risk. Compared to the lumpy, feast-or-famine fees of M&A and the white-knuckle risk of trading, a steady stream of management fees is the boring revenue every bank wishes it had more of. This is the same fee model that powers the giant index managers; we explore it at scale in [the piece on BlackRock, Vanguard, and State Street](/blog/trading/finance/big-three-blackrock-vanguard-state-street).

#### Worked example: the management fee on a \$3 billion wealth book

A bank's wealth-management arm runs \$3,000,000,000 of client money across portfolios, charging an all-in annual fee of 1% of assets.

- Annual management fee: 1% of \$3,000,000,000 = \$30,000,000.
- That \$30,000,000 arrives *every year* the money stays, with no deal to win and no trading bet to place. If markets rise 10% and the book grows to \$3.3 billion, next year's fee rises to \$33,000,000 automatically.
- If markets fall 20% and the book drops to \$2.4 billion, the fee falls to \$24,000,000 — still substantial, still recurring. The revenue shrinks with the market but rarely vanishes.

Compare this to M&A: a banker might earn \$30 million on one merger this year and zero from that client next year. Intuition: AWM trades a lower fee *rate* for a far more *durable* fee *stream*, which is exactly why a dollar of management-fee revenue is valued more highly by investors than a dollar of lumpy trading revenue — predictability is worth a premium.

The risk in AWM is gentler than elsewhere but real: clients can *withdraw* their money (called *outflows*), and a stretch of poor performance or a scandal can trigger a stampede that shrinks the fee base. There is also *fee compression* — the relentless decline in what investors will pay, driven by cheap index funds — which squeezes the rate even as the assets grow. But no AWM business has ever failed in a weekend the way Lehman did. That is the whole appeal.

## How each engine actually makes money: fee vs spread vs principal

Step back and the whole sprawling firm collapses into three revenue types. This is the single most useful simplification in the post.

![Matrix of divisions by revenue type and main risk](/imgs/blogs/inside-an-investment-bank-how-they-make-money-4.png)

1. **Fees** — payment for *advice* or for *running money*. M&A advisory fees and AWM management fees are both fees. The bank takes little or no principal risk; it is selling expertise or stewardship. Risk to the bank: the deal collapses (no fee) or clients withdraw their money (less AUM to charge on). Low risk, capital-light, high margin.

2. **Spread** — payment for *handling a transaction*. The underwriting gross spread (primary market) and the bid-ask spread (secondary market) are both spreads. The bank captures a gap. Risk: in underwriting, being stuck with unsold securities; in market-making, being stuck with inventory whose price moves against you before you can offload it. Medium risk, because the bank briefly touches the security.

3. **Principal risk** — payment for *taking risk with the bank's own capital*. Proprietary positions, big inventory bets, holding loans on the balance sheet. The bank gains if prices move its way and loses if they don't. Highest risk, highest variance, and the source of every famous blow-up.

Hold these three up against the four divisions and the bank becomes legible. IBD = mostly fees and underwriting spreads. Global Markets = mostly bid-ask spreads with a tail of principal risk. AWM = pure management fees. Research = an indirect support for the spread engine. The art of running a bank is loading up on the fee and spread businesses (steady, capital-light) while *carefully limiting* the principal-risk business (lucrative but lethal). When a bank forgets that art, you get 2008.

#### Worked example: the three revenue types on one \$5 billion deal

Suppose MegaCorp buys SmallCo for \$5 billion, paying with newly raised money. One bank can touch this deal three ways and earn three different kinds of revenue.

- **As M&A advisor to MegaCorp**, the bank charges an advisory *fee* of, say, 1% of the deal: 1% of \$5,000,000,000 = \$50,000,000. Pure agency. Risk: if the deal dies, the bank may collect only a small retainer.
- **As underwriter of the \$2 billion of new bonds** MegaCorp issues to help pay for the purchase, the bank's DCM desk earns a *spread*. At a 0.5% underwriting fee on \$2,000,000,000, that is \$10,000,000. The bank briefly owns the bonds before reselling them — a touch of risk.
- **As market maker** in MegaCorp's stock afterward, the Global Markets desk quotes a bid and an ask and earns the *spread* on every share that trades through it, plus or minus whatever its *inventory* does — a sliver of *principal* risk.

One transaction, three revenue types, three risk levels. Intuition: the same firm can be paid as an advisor, a placement agent, and a risk-taker — and the wise bank wants as much of the first two and as little of the last as it can manage.

## The IPO machine: how a company goes public

The IPO — *initial public offering* — is the most visible thing investment banks do, so it is worth walking through end to end. An IPO is the moment a private company sells shares to the public for the first time, turning founders' and early investors' paper holdings into tradable stock and raising fresh cash for the company. The bank that runs it is the *underwriter* (usually several banks share the work; the senior one is the *lead* or *bookrunner*).

![Pipeline of the IPO process from mandate to stabilization](/imgs/blogs/inside-an-investment-bank-how-they-make-money-2.png)

Here is the assembly line, stage by stage.

1. **Win the mandate.** The company interviews several banks in a "bake-off." Each pitches its valuation, its plan, and its fee. The company picks one or a few to lead. Relationships and the prestige of the bank's name matter enormously here.

2. **Due diligence and filing.** The bank, lawyers, and accountants comb through the company's finances and write the prospectus — in the U.S., the *S-1* registration statement filed with the Securities and Exchange Commission (SEC). This document is the company's full disclosure to prospective investors: risks, financials, ownership, everything.

3. **The roadshow.** Management and the bankers travel (or video-call) to pitch the company to big buy-side investors — the funds that might buy large blocks of shares. The goal is to gauge and build demand.

4. **Bookbuilding.** As the roadshow runs, the bank collects *indications of interest*: how many shares each fund would buy and at what price. This "book" of orders tells the bank the true demand curve. This is the bank's core skill — it knows the buyers, and it can read whether demand is hot or tepid.

5. **Pricing.** The night before trading begins, the bank and company set the final offer price and the number of shares. Set it too low and the company "leaves money on the table"; too high and the stock sinks on day one and everyone is unhappy. The bank threads this needle using the book.

6. **Listing.** Shares begin trading on an exchange (NYSE or Nasdaq). The opening trade is the first time the public secondary market prices the stock.

7. **Stabilization.** For a period after listing, the lead underwriter is allowed to support the price — typically via the *greenshoe* (over-allotment) option, which lets it sell extra shares it can buy back to soak up selling pressure if the price sags. This is legal price support, disclosed in advance.

The bank's pay for all of this is the **gross spread**: it buys the shares from the company at a discount to the offer price and sells them to investors at the full offer price, keeping the difference. In the U.S., the gross spread on a typical IPO has long clustered around **7% for smaller deals**, falling toward **3–4% for the very large ones** (the percentage shrinks as the deal grows, but the dollars grow). The spread is split among the underwriting banks and the brokers who help place the shares.

#### Worked example: the gross spread on a \$1 billion IPO

A company does an IPO that raises \$1,000,000,000 at a 4% gross spread (a reasonable rate for a deal that size).

- Money the company is trying to raise (offer value): \$1,000,000,000.
- Gross spread: 4% x \$1,000,000,000 = \$40,000,000.
- The company nets roughly \$1,000,000,000 - \$40,000,000 = \$960,000,000 (before legal and accounting costs).
- The \$40,000,000 is split among the underwriting syndicate. The lead bookrunner typically takes the largest share.

So the underwriters collectively earn \$40 million for a few months of work pricing and placing the shares — and they earned it whether the stock later went up or down, because they sold it on to investors. Intuition: the bank's IPO profit is a *spread*, locked in at pricing, not a bet on where the stock goes next.

There is a real risk hiding in "firm commitment," though. If the bank agreed to buy the whole \$1 billion issue and demand collapses overnight, the bank can be stuck owning shares it must dump below the offer price — that is the spread's flip side, the principal risk of underwriting. It is rare, but it is why the gross spread is not free money: the bank is being paid partly to *bear the placement risk*.

### Why the day-one "pop" is not a free lunch for the bank

You have surely seen the headline: "Stock soars 80% on first day of trading!" People assume the bank deliberately underpriced the IPO to do its buy-side friends a favor. The truth is more interesting and we save the full debunk for the misconceptions section — but note here that a big pop means the *company* raised less than it could have. The bank's spread is a percentage of the offer price, so a pop does not directly enrich the bank; it mostly enriches the investors who got allocations at the offer price. The incentives are genuinely tangled, which is exactly why this is so often misunderstood.

## Market-making: how the trading floor earns its keep

Now the secondary-market engine. A *market maker* quotes two prices at once: a *bid* (what it will pay to buy) and an *ask* (what it will charge to sell), and it stands ready to trade with anyone at those prices. The gap between them is the *bid-ask spread*, and capturing that gap, over and over, on enormous volume, is the business.

![Graph of a stock trade routing from you to broker to market maker](/imgs/blogs/inside-an-investment-bank-how-they-make-money-5.png)

The figure traces the path of a single order. You tap "buy" in an app. Your *broker* (the app, or a registered investment advisor) routes the order — sometimes to an exchange, sometimes to a wholesale *market maker* in exchange for a small payment called *payment for order flow* (PFOF), which we will discuss. The market maker fills you, captures the spread, and the trade is then *cleared and settled* through the clearinghouse (in U.S. equities, the DTCC, on a "T+1" timetable — settlement one business day after the trade). At each hop someone takes a sliver.

The market maker's profit per round-trip is the spread times the size, but its *risk* is *inventory*. The moment a market maker buys from a seller, it owns that security and is exposed to its price until a buyer shows up. If the price drops while the maker is holding, the inventory loss can swallow many spreads' worth of profit. Managing inventory — staying roughly flat, hedging, adjusting quotes to attract the side it needs — is the whole craft. A good market maker earns the spread thousands of times and rarely gets caught long into a falling market.

#### Worked example: capturing a two-cent spread on 10,000 shares

A dealer quotes a stock at \$49.99 bid / \$50.01 ask. The spread is \$50.01 - \$49.99 = \$0.02 per share.

- A seller hits the dealer's bid: the dealer *buys* 10,000 shares at \$49.99, paying 10,000 x \$49.99 = \$499,900.
- Moments later a buyer lifts the dealer's ask: the dealer *sells* those 10,000 shares at \$50.01, receiving 10,000 x \$50.01 = \$500,100.
- Profit on the round-trip: \$500,100 - \$499,900 = \$200. Equivalently, \$0.02 x 10,000 = \$200.

Two hundred dollars sounds tiny. But a large dealer does this on billions of shares a day across thousands of names. Two cents on ten thousand shares, repeated a hundred thousand times, is real money — and it arrives without the dealer ever "betting" on the stock's direction, *as long as the two sides arrive close together in time*. Intuition: market-making is a volume business that earns a tollbooth fee on liquidity, and its enemy is not being wrong about direction but being *stuck holding inventory* when the matching trade is slow to come.

The reason spreads have collapsed over the decades — many liquid stocks now trade at a one-cent or sub-penny spread — is competition and electronic trading. Tighter spreads are great for you (cheaper to trade) and brutal for market makers (less per trade), which is why the survivors are giant, automated, and run on razor-thin margins times colossal volume.

## Leverage and the 2008 lesson: why a 3% loss can kill a bank

We now reach the part that turns a tour of revenue lines into a story about catastrophe. Everything above describes how a bank earns in good times. Leverage describes how it dies.

Recall the definition: *leverage* is controlling more assets than your own capital by borrowing the difference. Banks are leveraged by nature — borrowing to fund assets is what they do. The question is *how much*. In the mid-2000s, the big U.S. investment banks ran leverage of roughly **30-to-1** or higher. That number deserves to be felt, not just read.

![Stack of a bank balance sheet showing thin equity under borrowed assets](/imgs/blogs/inside-an-investment-bank-how-they-make-money-6.png)

The figure is a balance sheet drawn to scale. Almost the entire column is *assets funded by borrowed money*. At the very top sits a thin sliver: the bank's own *equity* — the shareholders' money, the cushion that absorbs losses. At 30-to-1, that cushion is about 3.2% of the total. Everything below it is somebody else's money that must be repaid, much of it borrowed *overnight* in the *repo* (repurchase agreement) market — short-term loans that lenders can refuse to roll over the moment they get nervous.

#### Worked example: how 30-to-1 leverage gets wiped out by a 3.3% loss

A bank holds \$31 of assets funded by \$1 of its own equity and \$30 of borrowed money. Leverage = \$31 / \$1 = 31-to-1 (call it "about 30-to-1").

- Suppose the assets fall in value by 3.3%: loss = 3.3% x \$31 ~ \$1.02.
- The bank still owes the \$30 of debt in full — debt does not shrink when assets fall.
- New equity = \$1 (original) - \$1.02 (loss) ~ -\$0.02. **The equity is gone.**

A mere 3.3% drop in the value of the assets has erased 100% of the shareholders' money. Scale it up: the same math means a bank can be *insolvent* — assets worth less than what it owes — after a downturn that, in any other context, would look mild. Intuition: leverage does not just magnify your returns, it sets a precise *death threshold* — at 30-to-1, that threshold is a roughly 3.3% loss, and in a panic, assets can fall that much in a single bad week.

Now add the second killer: *funding runs*. Because so much of the borrowing is short-term, a leveraged bank depends on lenders agreeing, every single day, to keep lending. The instant those lenders doubt the bank can repay — even if the doubt is wrong — they refuse to roll the loans. The bank must then sell assets fast to raise cash, those *fire sales* push prices down, which deepens the losses, which scares lenders more. This *liquidity spiral* is how an investment bank dies: not slowly, over quarters, but in *days*, when its overnight funding evaporates.

That is precisely what happened in 2008.

### Bear Stearns and Lehman Brothers

**Bear Stearns**, the smallest of the big U.S. investment banks, had loaded up on mortgage-related securities at high leverage. As the housing market turned in 2007–2008, those assets fell and lenders lost confidence. In March 2008, over a single weekend, Bear's overnight funding vanished, and it collapsed from a going concern into a fire-sale rescue: JPMorgan bought it for \$2 a share (later raised to \$10), with the Federal Reserve backstopping \$30 billion of the dodgiest assets. A storied 85-year-old firm gone in a weekend, killed by leverage plus a funding run.

**Lehman Brothers** was the same disease, fatal. Bigger than Bear, similarly leveraged into mortgages, Lehman could not find a buyer or a government rescue. On September 15, 2008, it filed for bankruptcy — at roughly \$639 billion in assets, still the largest bankruptcy in U.S. history. Its failure froze global credit markets and turned a housing downturn into the worst financial crisis since the 1930s. The lesson regulators drew was blunt: investment banks running 30-to-1 leverage on illiquid assets funded by overnight money are *structurally* fragile, and their failure can take the whole system with them.

### The regulatory response

Two responses matter for our story.

First, **Goldman Sachs and Morgan Stanley became bank holding companies** in September 2008. By converting, they gained permanent access to the Federal Reserve's emergency lending and the stability of insured deposits — at the cost of much tougher regulation and far *lower allowed leverage*. The era of the freestanding 30-to-1 investment bank ended that month. Today's big banks run leverage closer to 10-to-1 to 15-to-1, with regulators watching capital ratios constantly.

Second, the **Volcker Rule** (part of the 2010 Dodd-Frank Act, named for former Fed chair Paul Volcker) banned banks that take insured deposits from engaging in most *proprietary trading* — betting the firm's own money purely for its own profit. Market-making for clients is still allowed (you cannot have markets without it), but the in-house hedge-fund-style prop desks were largely shut down or spun off. The intent: keep the principal-risk engine — the one that blows up — away from the deposits the government insures.

![Timeline from Glass-Steagall 1933 to the Volcker Rule 2010](/imgs/blogs/inside-an-investment-bank-how-they-make-money-7.png)

The timeline puts it in the long arc. In 1933, after the Great Depression, the **Glass-Steagall Act** built a wall between commercial banking (deposits and loans) and investment banking (securities), on the theory that mixing them had helped cause the crash. That wall stood for over sixty years. In 1999, the **Gramm-Leach-Bliley Act** repealed it, letting the two recombine into universal banks — which is how Citigroup and others got so large. Then came 2008, and the **Volcker Rule** in 2010 rebuilt a narrower wall: not separating the businesses, but fencing proprietary trading away from insured deposits. The history rhymes: a crisis builds a wall, a boom tears it down, the next crisis builds a smaller one.

## A note on compensation and culture

It would be incomplete to describe how the bank earns without noting where a striking share of the money goes: the people. Investment banking is famous for paying its employees a large fraction of revenue. For decades, the big trading-and-banking firms paid out something like 40–50% of *net revenue* as *compensation and benefits* — base salaries plus the all-important year-end *bonus*, which can dwarf the salary for senior producers. The logic is that the firm's main asset walks out the door every evening: the relationships, the deal flow, and the trading skill live in people, and competitors will hire them away if the pay is not there.

This shapes the culture. The hours are punishing, especially for junior analysts in IBD, who routinely work very long weeks building models and pitch books. Pay is sharply tied to the revenue you personally bring in or the deals you touch, which concentrates rewards on a relatively small number of senior "producers" and rainmakers. It also creates the incentive problems regulators worry about: if your bonus depends on this year's profit and the firm eats the loss if a risky bet sours later, you are tempted to take risks whose downside lands on the firm and the system rather than on you. Much of post-2008 reform — deferring bonuses, paying in stock that vests over years, clawback provisions — is an attempt to align the trader's incentives with the firm's survival. Culture is not a soft topic here; it is the mechanism that translates the revenue engines into either prudent stewardship or reckless risk.

## Common misconceptions

**"Investment banks just gamble with your deposits."** This conflates two different businesses. The classic investment bank had *no* insured deposits to gamble with — it funded itself in the wholesale markets. Even at today's universal banks, the Volcker Rule specifically bars using insured deposits for proprietary trading. The real 2008 problem was not "gambling with grandma's savings"; it was *leverage and short-term funding* on the bank's own (uninsured) balance sheet. The cartoon points at the wrong villain — it was the borrowing structure, not the deposit base.

**"A big IPO 'pop' means the bank deliberately mispriced it to help its friends."** A first-day surge does mean the offer price was below what the market would bear, but the story is muddier than a simple favor. Demand is genuinely hard to gauge in advance; pricing a hair low reduces the risk of a failed deal and rewards the long-term investors the company wants on its register. Critically, the *bank's* fee is a percentage of the offer price, so a pop does not directly fatten the bank — it is the *company* that "left money on the table," and the *allocated investors* who gained. There are real conflicts (banks like happy buy-side clients who steer trading business their way), but "the bank pocketed the pop" is simply wrong about the mechanics.

**"Banks always bet against their own clients."** Sometimes a bank is the counterparty to a client's trade — that is what market-making *is*, and the two sides naturally have opposite positions in that single trade. But "the bank profits when you lose" is not the business model: a market maker earns the *spread* and tries to stay *flat* (no directional position), not to win a bet against you. The genuine scandals — like the 2010 SEC case against Goldman over the "Abacus" mortgage product — were about *failing to disclose* a conflict, not about the mere existence of opposite sides to a trade. The distinction between an undisclosed conflict and ordinary two-sided market-making is the whole ballgame.

**"Research is independent of the banking side."** It is *supposed* to be, and since the early-2000s reforms there is a legal wall: analysts cannot be paid based on banking deals, and communications are restricted. But the firm as a whole still benefits when its bankers win an issuer's business, and history (the dot-com-era analyst scandals) shows the wall can be porous when incentives press on it. Treat sell-side research as useful information with a known directional bias — banks publish far more "buy" ratings than "sell," partly because a "sell" can poison the firm's banking relationship with that company.

**"Making markets is risk-free arbitrage."** Capturing the bid-ask spread *looks* risk-free, but a market maker is constantly exposed to *inventory risk* — owning a security whose price can fall before it finds a buyer — and to *adverse selection*, the danger that the person trading against it knows something it doesn't (the seller dumping right before bad news). The spread is partly compensation for exactly these risks. In a calm market it is a tollbooth; in a crash, market makers can take large losses or simply stop quoting, which is why liquidity "disappears" precisely when you most want it.

**"Investment banking and 'the trading floor' are the same job."** They share a building and a brand, but the Investment Banking Division (advisory and capital raising) and Global Markets (sales and trading) are different worlds — different skills, different hours, different revenue (fees and underwriting spreads vs. bid-ask spreads and principal risk), and an information *wall* between them, because the bankers know confidential deal information that traders are forbidden to act on. Lumping them together hides the most important structural fact about the firm: it is several businesses, not one.

## How it shows up in real markets

Abstractions become real when you watch them play out. Here are concrete episodes where the mechanisms above did the work.

### A landmark advisory mandate: Microsoft buys Activision Blizzard (2022–2023)

In January 2022, Microsoft agreed to buy the video-game maker Activision Blizzard for about \$68.7 billion in cash — at the time one of the largest technology acquisitions ever. The deal is a textbook IBD M&A engagement: each side hired bankers to advise, value the businesses, structure the terms, and provide fairness opinions. The advisory *fees* on a deal this size run into the tens of millions per advisor, paid for advice and process management, with essentially no principal risk to the banks. The deal also dragged on for nearly two years through antitrust review on three continents, which illustrates the M&A fee's risk: much of the fee is contingent on *closing*, so bankers earn the big number only if the regulators ultimately let it through (they did, in October 2023). The mechanism from this post — advisory as pure agency, paid a percentage fee, at risk to deal completion — is exactly what played out.

### A mega-IPO and its gross spread: Alibaba (2014) and the modern wave

In September 2014, Alibaba raised about \$25 billion in what was then the largest IPO in history, listing on the NYSE. The underwriting syndicate — led by a group of major banks — earned the *gross spread* on that enormous offering; even at a low percentage rate (large deals price well under the 7% small-deal norm), the dollar fees were vast, reportedly around 1% of the deal, or roughly \$250 million split among the banks. The Alibaba deal shows the gross-spread mechanism at industrial scale and confirms the pattern from our worked example: the percentage falls as the deal grows, but the dollars climb. It also showcased bookbuilding and stabilization — the underwriters exercised the greenshoe over-allotment to manage early trading.

### The collapse: Lehman Brothers, September 2008

We covered the mechanics above; as a case study, Lehman is the purest demonstration of the leverage-plus-funding-run failure mode. Lehman was leveraged roughly 30-to-1 into mortgage assets that were falling in value and hard to sell. As confidence eroded through 2008, its short-term lenders and trading counterparties pulled back; unable to fund itself or find a rescuer, it filed for bankruptcy on September 15, 2008, with about \$639 billion in assets. The immediate aftermath — frozen money markets, a global credit seizure, emergency interventions — is why this single firm's failure reset the rules for the entire industry. Every element from the leverage section is visible: the thin equity cushion, the overnight funding dependence, the fire-sale spiral, the 3%-loss death threshold made horribly literal.

### The market-making and PFOF debate: GameStop, January 2021

In January 2021, a crowd of retail traders, coordinating online, drove the stock of GameStop from a few dollars to a peak above \$480, squeezing hedge funds that had bet against it. The episode put a spotlight on the plumbing in our trade-path figure. Commission-free brokerages route much of their order flow to wholesale *market makers* in exchange for *payment for order flow* (PFOF), and one such market maker handled a large share of the retail volume. Critics argued PFOF creates a conflict — the broker is paid by the firm filling your trade — while defenders argued it funds zero-commission trading and that retail orders often get *price improvement* (filled slightly better than the public quote). The clearinghouse mechanics also bit: amid the volatility, the DTCC demanded far larger *collateral* from brokers to cover settlement risk, which is why some brokers abruptly restricted buying. Every actor in the trade-path figure — broker, market maker, exchange, clearinghouse — was suddenly front-page news, and the spread/PFOF business model became a Congressional hearing topic.

### A trading-loss scandal: the "London Whale," JPMorgan 2012

In 2012, a trader in JPMorgan's London-based Chief Investment Office built enormous positions in credit derivatives — so large the trader was nicknamed the "London Whale" — and the positions soured spectacularly, costing the bank more than \$6 billion. This is the principal-risk engine misfiring. The CIO was nominally *hedging* the bank's balance sheet, but the positions grew into something that looked far more like a proprietary bet, exactly the kind of activity the just-passed Volcker Rule aimed to curb. The episode showed that even a famously well-run bank can let a "hedging" desk drift into outsized principal risk, that risk controls can fail to flag a position until it is enormous, and that the line between legitimate market-making/hedging and banned prop trading is genuinely hard to police. JPMorgan paid over \$900 million in regulatory penalties on top of the trading loss.

### The undisclosed-conflict scandal: Goldman's "Abacus," 2010

In April 2010, the SEC charged Goldman Sachs over a mortgage-linked product called Abacus 2007-AC1. Goldman had created a security tied to a basket of mortgage bonds and sold it to investors, while a hedge fund that had helped *select* the underlying bonds was simultaneously betting *against* the very product — a bet the buyers were not told about. Goldman settled for \$550 million. This case is the precise illustration of the "do banks bet against clients?" misconception. The problem was *not* that there were two sides to the trade — there always are — but that a material conflict was not disclosed to the buyers. It clarified the rule that matters: market-making with opposite positions is normal; *hiding* who is on the other side and why is the offense.

### The steady engine in the spotlight: the rise of asset and wealth management

Less dramatic but more important to modern bank strategy is the deliberate pivot toward AWM. After 2008 made trading revenue volatile and capital-expensive, banks leaned harder into the *recurring management fee* engine. Morgan Stanley's multi-year acquisition spree in wealth management (buying the brokerage Smith Barney and later the platform E\*Trade and the asset manager Eaton Vance) transformed it from a trading-heavy firm into one where stable wealth-and-asset-management fees are a dominant, prized revenue source. The mechanism from this post — a small annual percentage on a vast pile of client assets, arriving in good markets and bad — is exactly why every big bank now wants more AWM and less reliance on the lumpy, risky engines. The boring engine became the strategic one.

## When this matters to you, and where to read next

You may never work at a bank, but its engines touch your life constantly. When you buy a stock in your retirement account, a market maker on the other side captured a sliver of spread — smaller than it used to be, thanks to the competition that crushed spreads to pennies, which is money back in your pocket. When a company you use goes public, the gross spread you now understand was carved out of the money it raised. When the news says a bank "took a trading loss" or "advised on a merger" or "grew its wealth-management fees," you can place the number in the right engine and judge how risky it was. And when the next crisis comes — there is always a next one — you will know to look first at *leverage and funding*, not at the headlines about greed, because the structural number is what determines whether a wobble becomes a collapse.

The one mental model to carry away: a bank is *fees plus spreads plus carefully limited principal risk*, and the whole regulatory apparatus exists to keep that last term small relative to a thin equity cushion. About 30-to-1 was too much; that is why the rules changed.

To go deeper into the ecosystem this firm lives in, three companion pieces fit directly alongside this one. Start with [the field guide to financial institutions](/blog/trading/finance/field-guide-to-financial-institutions) for the wide-angle map of every player, sell-side and buy-side. Then read [how hedge funds work and the "2-and-20" fee model](/blog/trading/finance/how-hedge-funds-work-leverage-2-and-20) to see the buy-side's most aggressive users of leverage — the same force that felled Lehman, deployed by the bank's clients. Finally, [the piece on BlackRock, Vanguard, and State Street](/blog/trading/finance/big-three-blackrock-vanguard-state-street) shows the management-fee engine — the calm fourth division here — scaled up to trillions and turned into the dominant force in modern markets.

This is educational, not advice. Nothing here is a recommendation to buy, sell, or invest in anything — it is a map of how a corner of finance is built, so that the next time you read about Goldman or JPMorgan, you know exactly which engine produced the number on the page.
