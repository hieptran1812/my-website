---
title: "The Players: Savers, Issuers, and the Middlemen in Between"
date: "2026-06-21"
publishDate: "2026-06-21"
description: "A map of everyone in a capital market — who supplies the money, who needs it, and exactly what each middleman in between gets paid for."
tags: ["capital-markets", "buy-side", "sell-side", "intermediaries", "investment-banks", "asset-management", "market-structure", "fees", "underwriting", "exchanges"]
category: "trading"
subcategory: "Capital Markets"
author: "Hiep Tran"
featured: true
readTime: 42
---

> [!important]
> **TL;DR** — A capital market is a relay race in which household savings are passed, hand to hand, until they reach a company or government that turns them into something real; every hand that touches the baton charges a toll, and learning the names of those tolls is how you learn who actually runs finance.
>
> - There are only **three kinds of player**: capital **providers** (savers and the funds that pool their money), capital **users** (issuers — companies, governments, and the vehicles they create), and **intermediaries** (the middlemen who connect the two and take a cut).
> - Two things flow in opposite directions: **money** goes from provider to issuer, and a **security** — the legal claim on that money — comes back the other way. Every intermediary sits on one of those two arrows and is paid for moving it.
> - Each middleman sells exactly **one service** and has a **named fee** for it: underwriting spread, commission, bid-ask spread, management fee on AUM, listing and data fees, rating fees, index license fees. Once you can name the fee, you understand the business.
> - The one number to anchor on: a fund charging **"2-and-20"** on a \$100M portfolio collects **\$2M every year before earning a single dollar of profit**, then takes a fifth of the gains on top.

## A morning when the whole machine showed up at once

On the morning of December 14, 2020, Airbnb's shares were supposed to start trading on the Nasdaq at \$68. They opened at \$146 — more than double — and the company's founders watched, live, as the market decided their nine-year-old startup was worth around \$100 billion. The financial press called it a triumph. A few investors called it a robbery: every dollar above \$68 was money that *could* have gone into Airbnb's bank account and instead went to the lucky funds who got allocated shares at the offer price and flipped them by lunchtime.

Pause on that morning, because almost every player in a capital market had a hand in it. Households had, over years, sent their savings to mutual funds and pension plans. Those funds had hired asset managers to invest the money. The asset managers placed orders through broker-dealers. An investment bank — actually a syndicate of them, led by Morgan Stanley and Goldman Sachs — had spent months preparing Airbnb to sell shares, set the \$68 price, and pocketed a fee for the whole production. The Nasdaq provided the venue and charged Airbnb to list there. A clearinghouse stood behind every trade so that no buyer worried the seller would vanish. And index providers were already calculating whether Airbnb was big enough to join their benchmarks, which would force still more funds to buy it.

That single IPO is the whole series in miniature: a **primary market** that *created* a new security to raise capital, instantly handed off to a **secondary market** where that security would trade billions of times over the coming years, joined by **plumbing** that settled the cash and shares, and run end-to-end by **intermediaries** who each took a fee. This post is the cast list. Before we trace how issuance works, how trading works, or how settlement works in later posts, we need the map of *who is on the field and what each of them is paid for*. Here is that map.

![Map of capital providers, intermediaries, and issuers with money and securities flowing](/imgs/blogs/the-players-savers-issuers-and-the-middlemen-in-between-1.png)

Read the map left to right and it answers the only question that matters in this whole machine: *whose money is it, where does it end up, and who skims it on the way?* On the left are the people whose money it actually is. On the right are the people who put it to work. In the middle is a thick band of firms — none of which owns the money or uses it — who exist purely to move it across the gap and who, collectively, earn hundreds of billions of dollars a year for doing so.

## Foundations: the three roles and the two arrows

Before any names, fix two ideas. They are the load-bearing beams of everything below.

**A security is a promise written down so it can be sold.** When you lend a company \$1,000, you have a claim on that company — it owes you. If that claim lives only in a private contract, it is hard to sell to someone else. A *security* is that same claim standardized, documented, and made transferable: a bond, a share of stock, a unit in a fund. The whole magic of a capital market is that it turns a relationship ("I lent you money") into a tradable object ("here is a thing you can buy from me"). That object is the baton in our relay race.

**Money and securities flow in opposite directions.** This is the single most useful sentence in the post, so say it slowly. When a saver funds an issuer, *money* moves from saver to issuer. In exchange, a *security* — the claim on the issuer — moves from issuer back to saver. Cash one way, paper the other. Every player we describe sits on one of those two arrows and gets paid for helping it move. If you ever get lost, ask: *which arrow is this firm on, and what do they charge to move it?*

With those two beams in place, here are the three roles. They are exhaustive: every participant in a capital market is one of these, and many large institutions are several at once.

1. **Capital providers (the buy-side, the savers).** They have money and want a return. They *buy* securities, so they are the "buy-side." Households, mutual funds, ETFs, pension funds, insurers, sovereign wealth funds, endowments, bank treasuries, and foreign investors all live here. They are the source of the money and the final owners of the securities.

2. **Capital users (the issuers).** They need money to do something real — build a factory, fund a deficit, pave a road, scale a startup — and they raise it by *creating and selling* securities. Corporations, sovereign governments, municipalities, agencies, special-purpose vehicles, and startups live here. They are the destination of the money and the original source of the securities.

3. **Intermediaries (the middlemen, much of it the "sell-side").** They neither own the money nor use it. They stand between providers and issuers, moving money one way and securities the other, and they charge a fee for each service. Investment banks, broker-dealers, exchanges, clearinghouses, custodians, asset managers, market makers, rating agencies, and index providers all live here. The "sell-side" is the slice of them that *sells* securities and services to the buy-side — investment banks and broker-dealers especially.

The rest of this post walks through the three groups, and for each intermediary it asks the one question the brochures bury: *what, exactly, are you paid?*

Here is why this map is not just trivia. The series' thesis is that **secondary-market liquidity is what makes primary issuance possible** — nobody hands a company \$1,000 for thirty years unless they believe they can sell that claim tomorrow morning to somebody else. The buy-side supplies the patience; the sell-side and the exchanges supply the liquidity; the issuers supply the projects. Take away any one group and the machine seizes. So let us meet them.

## The capital providers: whose money is it, really?

Start where the money is born: with people who earn more than they spend. A capital market exists because of a basic mismatch. Households accumulate savings they do not need today and will need in thirty years (retirement) or ten (a house) or five (college). Companies and governments need money *today* for projects that will only pay off over those same long horizons. The capital market is the institution that matches the saver's surplus to the user's deficit — and crucially, lets the saver get the money back early by selling the claim to the next saver.

But almost no household does this directly. The striking fact about the modern market is how little of it ordinary people own *in their own name*. Look at who actually holds US stocks.

![Pie chart of US stock market ownership by holder type](/imgs/blogs/the-players-savers-issuers-and-the-middlemen-in-between-2.png)

Even the "households (direct)" slice overstates direct ownership, because it lumps in the very wealthy who hold large individual portfolios. For a typical person, their exposure to the stock market arrives *through* a fund: a 401(k) invested in an index ETF, a pension that owns thousands of companies, an insurance policy whose reserves are parked in bonds. The household is the ultimate owner, but a chain of pooled vehicles sits in between. That chain is the buy-side, and it is worth drawing.

![Tree of capital provider types from households to funds to official money](/imgs/blogs/the-players-savers-issuers-and-the-middlemen-in-between-3.png)

Let's meet each branch, because their *motives* and *time horizons* differ enormously — and those differences are why the market has so many different kinds of securities.

**Households.** The ultimate source. A household saves to smooth consumption across a lifetime: save in your earning years, spend in retirement. A household's time horizon is its life, but its *patience* is fragile — fear and greed move it. Most households reach the market through a tax-advantaged retirement account or a brokerage account, and most of *that* money sits in funds rather than individual securities. The household's role is to keep feeding savings into the top of the funnel.

**Mutual funds and ETFs.** A mutual fund pools many savers' money and buys a diversified basket of securities, so a person with \$5,000 can own a sliver of five hundred companies. An ETF (exchange-traded fund) does the same but its own units trade on an exchange all day like a stock. Both exist to solve a problem households cannot solve alone: diversification and professional selection at low cost. They now own roughly **28%** of the US stock market — the single largest organized owner — and a growing share of that is *passive* index funds that simply buy the whole market in proportion. Their horizon matches their investors': long, but subject to redemptions when investors get scared.

**Pension funds.** A pension fund holds money to pay retirees decades from now. That gives it the longest, most patient horizon of any large investor — a pension promising payments in 2055 can happily own a 30-year bond or an illiquid private asset, because it does not need the cash soon. Pensions are the natural buyers of long-duration assets precisely because their *liabilities* are long-dated. This matching of asset life to liability life is the deepest idea in institutional investing, and it is why pensions and insurers behave so differently from a day-trading household.

**Insurers.** A life insurer collects premiums today and pays claims over decades; a property insurer collects premiums and pays claims after disasters. Either way, an insurer is sitting on a pile of other people's money (the "float") that it must invest to *match* the timing and size of future claims. Insurers are enormous, conservative buyers of high-grade bonds, because a bond's fixed payments can be lined up against expected claims. They are the bedrock of the corporate and government bond markets.

**Sovereign wealth funds, central-bank reserves, endowments.** These are the "official" and ultra-long pools. A sovereign wealth fund (think Norway's, or Singapore's) invests a nation's surplus — often from oil or trade — for future generations, with a horizon measured in centuries. A central bank holds foreign-exchange reserves, much of it parked in US Treasury bonds for safety and liquidity. A university endowment invests gifts to fund the institution forever. All of them share an unusually long horizon and a tolerance for illiquidity that lets them buy assets households and even pensions cannot.

**Bank treasuries.** A commercial bank is mostly a lender, but its treasury department holds a portfolio of high-quality liquid securities — largely government bonds — as a buffer it can sell or pledge instantly if depositors withdraw. Banks are therefore huge holders of safe paper. (How a bank's balance sheet and lending business works is a different machine — see the banking series; here we only note the treasury as a capital provider.)

**Foreign investors.** Roughly **17%** of the US stock market is owned by foreigners — overseas pensions, sovereign funds, and households reaching for the returns and depth of the world's largest market. Cross-border flows are some of the most powerful and most fickle forces in any market; when foreigners pull money out of a country, prices and the currency can fall together. In smaller markets this dominates — we will see it vividly in Vietnam below.

#### Worked example: who actually owns "your" 401(k) share of Apple

Suppose you have \$30,000 in a target-date retirement fund, and that fund holds a total-market index. Apple is about **7%** of the US market by weight, so roughly **\$2,100** of your money tracks Apple. But you do not own a single Apple share in your name. The chain is: you → the target-date fund → its underlying index fund → the actual Apple shares, which are themselves held in "street name" by a custodian on the fund's behalf. Four layers separate you from the security. *The intuition: the modern saver almost never touches the security they own — they own a claim on a fund that owns a claim on a custodian that holds the share.* Every one of those layers, as we will see, charges a fee.

That worked example contains a phrase we should illustrate now, because it names a player most savers have never heard of and yet who holds nearly everything: the custodian. We will meet it among the intermediaries. First, let us see how *big* the pools these providers fund actually are.

![Bar chart of global equity and bond market size in trillions of dollars](/imgs/blogs/the-players-savers-issuers-and-the-middlemen-in-between-4.png)

Two takeaways from that chart shape everything else. First, the **bond market is bigger than the stock market** — globally about \$140 trillion of debt versus \$115 trillion of equity. Most people picture "the market" as the stock market, but the larger machine is debt: governments and companies borrowing. Second, the **US is a huge slice of both** — roughly half of world equity and a large share of world bonds — which is why US market structure, settlement rules, and regulators set the global standard. When the US moved to one-day settlement in 2024, the rest of the world had to scramble to keep up.

There is a pattern hiding in that list of providers, and it is the key to why the market can fund thirty-year projects at all. Read the buy-side from top to bottom and you are reading a **gradient of patience**. A household's patience is shallow and emotional — it can panic-sell in a week. A mutual fund's patience is its investors' patience, slightly buffered. A pension's patience is decades, because its obligations are decades away. An insurer's patience is shaped like its claims. A sovereign wealth fund's patience is generational. The capital market works by *sorting* securities to the providers whose patience matches the security's life: short, liquid assets to the impatient; long, illiquid assets to the patient. A 30-year infrastructure bond ends up with a pension or insurer not by accident but because they are the only buyers who can hold it without flinching. This sorting — long liabilities buying long assets — is the quiet machinery that lets the economy finance things that take a lifetime to pay off. And it only works because, at every level, the impatient saver knows they can sell to a more patient one tomorrow. Patience and liquidity are the same idea viewed from two ends.

So that is the buy-side: a funnel from household savings, through pooled vehicles with ever-longer horizons, into a \$255-trillion stock of global claims. Now, who is on the other end of those claims?

## The capital users: why anyone issues a security in the first place

An issuer is anyone who *creates* a security to raise money. The motive is always the same shape — "I have a use for capital today and a way to pay for it later" — but the specifics differ enough that each type of issuer gets its own market.

**Corporations.** A company raises capital in two fundamentally different ways, and the choice defines its whole financial life. It can sell **equity** — shares, a permanent slice of ownership that never has to be repaid but dilutes existing owners and hands over a piece of every future profit. Or it can sell **debt** — bonds, a promise to repay a fixed sum with interest, which must be honored but leaves ownership untouched. A young company short on profits and full of risk leans on equity; a stable, cash-generating company leans on cheap debt. The deep tradeoff between the two is its own subject — we cover it in [Debt vs. Equity: the two ways to raise capital](/blog/trading/capital-markets/debt-vs-equity-the-two-ways-to-raise-capital) — but for the cast list, the point is that corporations are the most visible issuers and the reason both the stock and corporate-bond markets exist.

**Sovereign governments.** A national government is the largest single class of issuer on earth. The US Treasury alone issues *trillions* of dollars of bonds every year — note the \$23 trillion of Treasury issuance in our debt-issuance data, dwarfing every other category — to fund deficits and roll over maturing debt. Government bonds are special: they are usually the *safest* security in their currency (a government can tax and, in its own currency, print), so they anchor the price of all other borrowing. The yield on a government bond is the baseline "risk-free rate" off which everything else is priced. (How that yield curve is built and what it means lives in the fixed-income series — see [The yield curve explained](/blog/trading/fixed-income/the-yield-curve-explained-the-most-important-chart-in-finance); here, the government is simply our biggest issuer.)

**Municipalities.** State and local governments issue bonds — "munis" in the US — to fund schools, roads, water systems, and stadiums. They are smaller, often tax-advantaged, and bought heavily by households and insurers in high tax brackets. Their existence is a reminder that the capital market funds the physical world: the road you drove on this morning was very likely financed by a bond.

**Agencies and government-sponsored enterprises.** Bodies like Fannie Mae and Freddie Mac in the US issue bonds to fund mortgages, sitting in a gray zone between government and corporate credit. They are huge issuers — note the agency and mortgage-backed slices in the issuance data — and they exist to channel capital-market money into a specific public goal (here, home ownership).

**Special-purpose vehicles.** This is the issuer most people have never heard of and the one that caused the most trouble. An SPV (or "special-purpose entity") is a shell company created for one transaction: it buys a pool of assets — mortgages, car loans, credit-card receivables — and issues securities backed by the cash those assets throw off. This is **securitization**, and it lets a bank turn illiquid loans into tradable bonds. Done well, it spreads risk and frees up bank capital. Done badly, it concentrated and hid risk on the road to 2008. (The mechanics of slicing a loan pool into tranches are covered in [Securitization: how banks turn loans into securities](/blog/trading/banking/securitization-how-banks-turn-loans-into-securities); here, the SPV is simply a manufactured issuer.)

It is worth pausing on *why* these issuers fall into separate markets at all, rather than one big pool. The answer is **credit risk and rules**. A US Treasury bond and a junk-rated corporate bond are both "debt," but they are bought by different providers, priced off different risk, and governed by different disclosure. An insurer restricted to investment-grade paper cannot touch the junk bond; a hedge fund hunting yield lives there. The issuer's identity — sovereign, municipal, agency, corporate, SPV — is shorthand for *how likely am I to be repaid and who is allowed to lend to me*. That is also why the rating agencies, whom we meet shortly, wield such power: they convert "who is this issuer" into a letter grade that decides which providers may buy. The map of issuers is, underneath, a map of risk tiers — and the whole capital market is an apparatus for matching each tier of risk to the providers willing and permitted to bear it.

**Startups.** A startup is an issuer in slow motion. It sells equity privately — to friends, angels, venture funds — in a series of rounds, each at a higher price, with the dream of one day selling shares to the public in an IPO. The startup matters to our story because it is where the *primary market* begins for the most exciting companies, and because the gap between the private price and the public price is exactly the "pop" that made Airbnb's morning so dramatic.

#### Worked example: the same project, debt or equity

A company wants \$100M to build a factory expected to earn \$15M a year. **Debt route:** issue \$100M of bonds at a 6% coupon. It pays \$6M a year in interest, keeps the other \$9M, and owns the factory outright once the bonds are repaid — but if earnings fall below \$6M, it risks default. **Equity route:** sell \$100M of new shares. It pays no fixed cost, so a bad year just means lower profits, not default — but the new shareholders permanently own a slice of all \$15M a year, forever, not just until a loan is repaid. *The intuition: debt is cheaper and temporary but unforgiving; equity is expensive and permanent but safe. The issuer's job is to pick the mix that survives a bad year without giving away too much of a good one.*

Notice what *both* routes require: somebody on the buy-side willing to hand over \$100M. And notice what neither the issuer nor the saver can do alone: find each other, agree a price, document the security, and settle the trade. That is the entire reason the middle of our map is so crowded. Let us finally meet the middlemen — and, for each, name the fee.

## The intermediaries: one service each, one fee each

Here is the band in the middle of the map. The trick to understanding it — and the trick the industry's job titles work hard to obscure — is that every intermediary sells exactly **one service** and charges a **specific named fee** for it. Memorize the fees and the org chart of global finance falls into place.

![Pipeline of intermediaries and the named fee each one charges](/imgs/blogs/the-players-savers-issuers-and-the-middlemen-in-between-5.png)

Walk that pipeline once, then we will take each box in turn.

### Investment banks (the sell-side): the underwriting spread

An investment bank's flagship service is **underwriting** — managing the creation and sale of a new security in the primary market. When a company goes public or a government sells bonds, the bank does the work: valuing the issuer, drafting the disclosure documents, lining up buyers, setting the price, and often *guaranteeing* the proceeds by buying the whole issue itself and reselling it. For this it earns the **underwriting spread** (also called the gross spread): the difference between what investors pay and what the issuer receives.

We are not going to re-derive how an investment bank makes money across all its businesses — that is its own deep dive in [Inside an investment bank: how they make money](/blog/trading/finance/inside-an-investment-bank-how-they-make-money). For the cast list, fix the one fee that defines its primary-market role: the spread on a deal.

#### Worked example: a 3% underwriting spread on a \$500M IPO

A company sells \$500M of stock in an IPO. The underwriting syndicate charges a gross spread of **3%** — at the small-deal end this can reach 7%, at the mega-deal end it falls toward 1–2%, but 3% is a fair round number. The bank's fee is therefore:

- Gross spread = 3% × \$500M = **\$15M**.
- The issuer nets \$500M − \$15M = **\$485M** in its bank account.
- That \$15M is split across the syndicate — the lead bank takes the largest share, co-managers less — and within each bank between the "management fee," the "underwriting fee," and the "selling concession" paid to whoever actually placed the shares.

*The intuition: the underwriting spread is the price of certainty and distribution — the issuer pays \$15M so it does not have to find five hundred buyers itself and does not have to worry whether the deal will sell.* And note what this fee does *not* capture: if the stock then pops from \$68 to \$146, that gain accrues to the investors who got allocated, not to the bank's spread and not to the issuer. The spread is the visible fee; the pop is the hidden one.

### Broker-dealers: the commission (and payment for order flow)

A **broker** acts as your agent: it takes your order to buy or sell and routes it to a market on your behalf. A **dealer** acts as a principal: it buys from you or sells to you out of its own inventory. Most firms do both, hence "broker-dealer." When you tap "buy" in a trading app, a broker-dealer is what stands between you and the market.

Historically a broker charged a **commission** — a flat fee or a per-share fee for executing your order. In the US, retail stock commissions have collapsed to *zero* at the big apps, which raises the obvious question: how do they get paid? Largely through **payment for order flow (PFOF)** — the broker sends your order to a wholesale market maker, who pays the broker a fraction of a cent per share for the right to fill it. You do not see this fee, but it is there, embedded in the price you get. Institutional brokers still charge explicit commissions, often a few cents or basis points per share, plus fees for research and access.

### Market makers: the bid-ask spread

A **market maker** is a dealer who continuously quotes two prices: a **bid** (what it will pay to buy) and an **ask** (what it will charge to sell), slightly higher. The gap between them is the **bid-ask spread**, and it is the market maker's fee for one priceless service: *immediacy.* Because a market maker is always willing to trade, you never have to wait for a matching counterparty to show up. You pay for that convenience by buying a hair above fair value and selling a hair below it.

The size of the spread is not arbitrary — it scales with how hard the stock is to trade. A mega-cap like Apple trades millions of shares a second, so its spread is microscopic; a tiny micro-cap might cost you a full percent just to round-trip.

![Horizontal bar chart of bid-ask spread by stock liquidity tier in basis points](/imgs/blogs/the-players-savers-issuers-and-the-middlemen-in-between-6.png)

#### Worked example: who earns what on a \$1,000 retail trade

You buy \$1,000 of a large-cap stock through a zero-commission app. Follow every cent:

- **Commission you see: \$0.** The app advertises free trades.
- **Bid-ask spread:** at a ~3 bps (0.03%) spread, the round-trip cost embedded in the price is about 3 bps on the way in, so roughly **\$0.15–\$0.30** of your trade is the market maker's edge. On a single \$1,000 trade it is pennies.
- **Payment for order flow:** the wholesaler pays the app a fraction of a cent per share; on a \$1,000 trade that might be **\$0.05–\$0.15** to the broker.
- **Exchange / venue fee:** a small per-share fee or rebate, fractions of a cent.
- **Clearing & settlement:** a tiny per-trade fee to the clearinghouse and custodian, well under a cent for retail size.

*The intuition: on a small retail trade the fees are almost invisible — pennies — which is exactly why "free" trading is viable; the providers make it up on volume across millions of orders. The fee you cannot see (the spread) is larger than the fee you can see (the commission), which is now zero.* Multiply those pennies by the tens of billions of shares traded daily and you have the entire revenue base of the trading industry.

### Exchanges: listing fees and (the real money) data fees

An **exchange** is the organized venue where buyers and sellers meet and a public price is formed — the NYSE, the Nasdaq, the London Stock Exchange, Vietnam's HOSE. People assume exchanges make their money from trading fees. They make *some*, but two other revenue lines are larger and more durable.

First, **listing fees**: a company pays an exchange an initial fee to list its shares and an annual fee to stay listed — typically tens to a few hundred thousand dollars a year for a large company. In return the company gets a venue, a ticker, and the prestige and liquidity of being on a major exchange.

Second, and this is the part outsiders miss, **market-data fees**: the exchange sells the real-time stream of its quotes and trades to banks, funds, data vendors, and trading firms, who cannot operate without it. For the modern exchange, *selling data about trades is often more profitable than hosting the trades.* The trades create the data; the data is the product. (The mechanics of how an exchange matches orders and connects to a clearinghouse are covered in [Stock exchanges and clearinghouses](/blog/trading/finance/stock-exchanges-and-clearinghouses) — here we only note what they charge.)

#### Worked example: an exchange's per-trade and data fees

Take a mid-size trading firm running 10 million shares a day across an exchange.

- **Per-trade / per-share fee:** at roughly \$0.0003 per share (and exchanges often *rebate* liquidity providers, charging only liquidity takers), 10M shares might net the exchange on the order of **\$1,500–\$3,000 a day** from this one firm — real, but thin.
- **Market-data subscription:** the same firm pays for the exchange's full real-time data feed — \$10,000+ per month is routine for direct feeds, far more for the lowest-latency colocated access — call it **\$120,000+ a year**, every year, regardless of whether it trades a lot or a little.

*The intuition: the per-trade fee scales with activity and gets competed toward zero; the data fee is a near-fixed subscription the firm cannot do without, so it is stickier and, in aggregate, the larger and steadier business.* That is why exchanges fight so hard over data: it is their annuity.

### Clearinghouses (CCPs) and custodians: the plumbing fee

Two players sit *behind* every trade, invisible to the saver, and they are why the modern market can trade trillions of dollars without participants worrying about each other.

A **central counterparty (CCP)**, or clearinghouse, steps into the middle of every trade through a process called **novation**: after you and a stranger agree a trade, the CCP becomes the buyer to the seller and the seller to the buyer, so neither of you bears the other's risk — you both face the CCP. It collects **margin** (a cash deposit) from members to cover the risk that one defaults, and it nets everyone's trades down so only the small net difference actually has to be settled. (The full machinery of clearing, settlement, and the default waterfall is a later post in this series; here, the CCP is a player who charges a basis-point clearing fee for guaranteeing the trade.)

A **central securities depository / custodian** is where securities actually *live.* When you "own" a share, you almost never hold a paper certificate; the share is recorded electronically in a depository (in the US, the DTCC's nominee holds the master record) and your custodian bank keeps the sub-record that says the share is yours. Custodians safekeep trillions of dollars of assets, settle trades, collect dividends and coupons, and handle corporate actions — for a small fee measured in basis points per year on assets held.

These plumbing fees are tiny per transaction, but the players are enormous because *everything* flows through them. They are the toll booths on the only road into the market.

### Asset managers: the management fee on AUM (and, for some, "2-and-20")

We met asset managers on the buy-side, because they invest on behalf of savers. But in the fee map they are also intermediaries: a household does not buy a thousand stocks itself, it pays a manager to do it. The manager's fee is a percentage of **assets under management (AUM)** — charged every year, win or lose. For a cheap index fund this is a few **basis points** (0.03–0.20%); for an active mutual fund it is often 0.5–1%; for a hedge fund or private-equity fund it is the famous **"2-and-20"**: a 2% annual management fee *plus* 20% of the profits.

#### Worked example: the hedge fund's "2-and-20" on \$100M

A hedge fund manages \$100M and has a good year, returning 15% (a \$15M gain) before fees.

- **Management fee:** 2% × \$100M = **\$2M**. The fund collects this every single year, whether it makes money or loses it. On the first day of the year, before any profit, the manager is already owed \$2M.
- **Performance fee:** 20% × \$15M = **\$3M** of the gains.
- **Total to the manager:** **\$5M**, leaving investors with \$15M − \$5M = \$10M, a net return of **10%** on their \$100M.

Now run the *bad* year: the fund loses 10%, so investors are down \$10M. The manager still collects the **\$2M** management fee. Investors are out \$12M; the manager is up \$2M. *The intuition: "2-and-20" pays the manager handsomely for showing up and spectacularly for winning, while the downside is borne almost entirely by the investor. The management fee is the part that never sleeps — it is why the single most important question to ask any fund is "what do you charge whether or not you make me money?"* (The internals of how a hedge fund is actually run are the hedge-fund series' subject; here, the fund is a capital provider that charges a fee on AUM.)

### Rating agencies: the rating fee

A **credit rating agency** — Moody's, S&P, Fitch — assesses how likely an issuer is to repay its debt and assigns a letter grade (AAA down to junk). This solves a real problem: a saver cannot personally analyze the creditworthiness of every government and company on earth, so the rating is a shared shorthand. Crucially, in the dominant **"issuer-pays"** model, *the issuer being rated pays the agency for the rating.* A company about to sell \$500M of bonds pays the agency a fee — often a fraction of a basis point on the issue size, scaling into the hundreds of thousands of dollars — to be rated, because most large buyers (pensions, insurers) are restricted to rated paper.

That fee structure contains an obvious conflict — the agency is paid by the entity it judges — and it sat at the center of 2008, when mortgage-backed securities that agencies stamped AAA turned out to be anything but. The rating fee is small, but the rating itself moves enormous flows of capital, because it determines who is *allowed* to buy a bond.

### Index providers: the license fee

The newest and quietly most powerful intermediary is the **index provider** — S&P Dow Jones, MSCI, FTSE Russell. An index is just a defined list of securities and a rule for weighting them (the S&P 500 is "the 500 biggest US companies, weighted by market value"). The index provider does two things and charges for both. It **licenses** the index to fund companies who build products tracking it — an ETF that tracks the S&P 500 pays S&P a license fee, typically a few basis points of the fund's assets. And it sells **data and membership decisions** that move markets: when a provider adds a company to a major index, every passive fund tracking that index *must* buy it, creating forced demand. Airbnb's eventual index inclusion forced billions of dollars of mechanical buying.

*The index provider sells the definition of "the market" itself* — and in an era where passive investing dominates, deciding what is in the index is deciding where a large slice of the world's savings flows. For a fee of a few basis points, that is extraordinary leverage.

### Prime brokers and securities lending: the financing and lending fee

There is one more intermediary that retail savers never see but that quietly oils the whole secondary market: the **prime broker**, almost always an arm of a large investment bank. A hedge fund does not open a hundred separate accounts at a hundred venues; it routes everything through a prime broker, which provides three bundled services and charges for each. It **finances** the fund's positions by lending it money to buy more than its own capital would allow (margin lending, charged as an interest spread). It **clears and custodies** the fund's trades in one place (a basis-point fee). And it runs the plumbing of **securities lending** — the mechanism that makes short selling possible.

Securities lending is worth understanding because it is invisible and enormous. When a fund wants to *short* a stock — sell a share it does not own, betting the price falls — it must first *borrow* that share from someone who owns it. Who owns idle shares sitting in custody? Exactly the long-horizon providers from earlier: pension funds, index funds, insurers, holding millions of shares they have no intention of selling soon. Through their custodian, they lend those shares out and earn a small fee for doing so, turning a dormant asset into a trickle of income. The prime broker stands in the middle, borrowing from the long holders and lending to the short sellers, and takes a cut of the lending fee.

#### Worked example: who earns what when a fund borrows \$10M of stock to short it

A hedge fund borrows \$10M worth of a moderately hard-to-borrow stock to short it for a year.

- **Borrow fee:** at a 3% annual "borrow rate" (easy-to-borrow names cost almost nothing; hot shorts can cost 20%+), the fund pays 3% × \$10M = **\$300,000** for the year to borrow the shares.
- **The lender's cut:** the pension or index fund that owned those shares might receive, say, 60% of that — about **\$180,000** — as pure incremental income on shares it was holding anyway.
- **The prime broker's cut:** the remaining **\$120,000** is split between the prime broker and the lending agent for arranging and guaranteeing the loan.

*The intuition: the long-horizon saver's dormant shares quietly earn a fee by enabling someone else's bet against the company they own — a perfect picture of how every idle asset in the market gets put to work, and how the middleman in between takes a slice for matching the two sides.* It also explains a strange fact: a pension fund can be long a stock and, by lending it out, be paid to facilitate the short sellers betting against it. The roles on our map overlap more than the boxes suggest.

## The two markets, and why the issuer only gets paid once

We have met every player. Now connect them back to the spine, because there is a distinction that confuses almost everyone the first time and is the key to the whole machine: the difference between the **primary** and **secondary** markets, and the fact that *the issuer gets the money only once.*

![Before and after comparison of money and security flows in primary versus secondary markets](/imgs/blogs/the-players-savers-issuers-and-the-middlemen-in-between-7.png)

In the **primary market**, a security is *created and sold for the first time.* A saver pays cash; an investment bank takes its underwriting spread; the issuer receives the proceeds. *This is the only moment the issuer's bank account fills.* The money flows from saver to issuer exactly once, when the security is born.

In the **secondary market**, that same security trades between investors, over and over, for the rest of its life. A buyer pays a seller; a broker takes a commission and a market maker takes the spread; the issuer **gets nothing.** When you buy Apple stock today, your money goes to the investor selling it, not to Apple. Apple was paid once, decades ago, when those shares were first issued.

This is the point most beginners miss, so make it concrete. The vast majority of all trading volume — essentially all of it — is *secondary*: investors swapping existing claims among themselves, with the issuer a bystander. So why does it matter to the issuer at all? Because of the spine of this entire series:

> Secondary-market liquidity is what makes primary issuance possible.

No saver would have paid \$68 for an Airbnb share in the primary market if they did not believe a vibrant secondary market would let them sell it later. The deep, liquid secondary market is the *reason* the primary market can sell anything. The intermediaries who run the secondary market — exchanges, market makers, brokers, clearinghouses — are not parasites on the "real" business of raising capital; they are the precondition for it. A company can only raise money today because thousands of strangers can trade that claim tomorrow. Take away tomorrow's liquidity and today's issuance dies.

#### Worked example: the issuer gets paid once, the middlemen forever

Trace \$500M of stock from birth.

- **Day 0, primary market:** the company sells \$500M of new shares. The underwriters take a 3% spread (**\$15M**), the issuer nets **\$485M**. The issuer's account fills, once.
- **Year 1 onward, secondary market:** suppose those shares turn over once a year — \$500M of trading. Every year, brokers earn commissions, market makers earn the spread, the exchange earns trading and data fees, the clearinghouse and custodian earn basis-point plumbing fees, and the funds that hold the stock charge their investors a management fee on it. Across all those hands, the *annual* toll on a \$500M position might run to several million dollars — and it recurs every year, forever, while the issuer never sees another cent.

*The intuition: issuing is a one-time event for the company but a perpetual annuity for the intermediaries. The company pays a big visible fee once; the savers pay a stream of small fees for as long as they hold and trade. That recurring stream is precisely why the secondary market is enormous and why its players are so profitable — and, paradoxically, why the issuer benefits, because that profitable liquidity is what let it raise the money in the first place.*

## Common misconceptions

**"The stock market is where companies get their money."** Almost never. Companies get money in the *primary* market — IPOs and follow-on offerings — which is a tiny fraction of activity. The "stock market" you watch on the news is the *secondary* market, where investors trade existing shares and the company is a spectator. US IPOs raised only about \$30 billion in 2024; secondary trading was tens of trillions. The company's connection to all that trading is indirect but vital: liquidity is what let it issue.

**"The intermediaries are just rent-seeking middlemen."** Some fees are too high and some conflicts are real (issuer-pays ratings, hidden order-flow payments). But each intermediary sells a genuine service that the saver and issuer cannot cheaply provide themselves: distribution (the bank), immediacy (the market maker), a meeting place and a price (the exchange), counterparty safety (the CCP), safekeeping (the custodian), diversification (the fund), credit shorthand (the rater), and a definition of the market (the index). Remove them and the cost of raising and deploying capital would *rise*, not fall. The right critique is "are these fees competitive?", not "should these players exist?".

**"Free trading means trading is free."** Zero commission moved the fee from a line item you see to a spread and an order-flow payment you don't. On a \$1,000 retail trade the hidden cost is pennies — genuinely cheap — but it is not zero, and for large or illiquid orders the spread can dwarf any old commission. The fee did not disappear; it changed clothes.

**"A fund manager makes money only when I make money."** Only the *performance* slice works that way. The *management fee on AUM* is charged every year regardless of results. A hedge fund losing 10% still collects its 2%. Over a long horizon, the steady management fee — not the headline performance fee — is what compounds against the saver's returns. A 1% annual fee on a long-horizon portfolio can quietly eat a quarter or more of the final wealth.

**"The credit rating is an objective, independent verdict."** In the dominant model the rated issuer *pays* for the rating, which creates a structural conflict that contributed to the 2008 crisis, when mortgage securities rated AAA collapsed. Ratings are useful shorthand, not gospel; a sophisticated buyer treats them as one input, not the answer.

## How it shows up in real markets

### The Airbnb pop: the fee you don't see is the biggest one

Return to that December 2020 morning. Airbnb's underwriters set the IPO price at \$68 and earned their gross spread on the roughly \$3.5 billion raised. But the stock opened at \$146. On the shares sold in the offering, the gap between \$68 and the first trade — well over \$3 billion of value — went to the investors who were *allocated* shares at \$68 and could sell at \$146, not to Airbnb and not to the underwriters' explicit fee. Critics call this "leaving money on the table," and it reignited an old debate: did the bank misprice the deal, or is a pop the unavoidable cost of guaranteeing the sale of a hard-to-value company? Either way, it is a vivid lesson in our fee map: the *visible* underwriting spread was modest; the *invisible* transfer — the pop — was larger, and it flowed to the buy-side, not the middlemen. Naming who earns what is rarely as simple as reading the fee schedule.

### When the T+1 switch reorganized the plumbing

On May 28, 2024, US stock settlement moved from "trade date plus two business days" (T+2) to "T+1" — trades now settle one day after they happen. This is invisible to a retail saver but a massive operation for the plumbing players: custodians, clearinghouses, and especially *foreign* investors who now had one fewer day to source US dollars and confirm trades across time zones. The change cut the clearinghouse's risk window in half (less time for a counterparty to default before settlement) and reduced the margin members must post, but it forced enormous spending on faster back-office systems. It is a clean example of how the plumbing players — the ones savers never see — quietly carry the operational weight of the whole market, and why a US rule change ripples worldwide given the US's share of global markets.

### The boom-and-bust of issuance: 2021 versus 2022

The primary market is wildly cyclical, because issuance only happens when issuers like the price and savers have appetite. Watch the swing.

![Bar chart of US IPO proceeds by year highlighting the 2021 boom and 2022 collapse](/imgs/blogs/the-players-savers-issuers-and-the-middlemen-in-between-8.png)

In 2021, with cheap money and euphoric markets, US companies raised about **\$142 billion** in traditional IPOs — the buy-side's appetite was bottomless and the underwriters' spread machine ran hot. In 2022, as the Federal Reserve raised rates and markets fell, IPO proceeds collapsed to roughly **\$8 billion** — a more than 90% drop in a single year. The pipeline of intermediaries did not disappear; the *issuers* simply stopped issuing, because no company wants to sell shares cheaply into a falling market. This is the spine again: when secondary-market conditions sour, primary issuance freezes. The middlemen's underwriting revenue is hostage to the buy-side's mood, which is exactly why investment banks diversify into the perpetual annuities of trading, data, and asset management.

### Vietnam: when foreign providers are the whole story

In a large market like the US, foreign investors are about 17% of equity ownership — significant but not dominant. In a smaller, faster-growing market like Vietnam, the *foreign* capital provider can dominate sentiment far beyond its share, because local institutions are still young and households trade emotionally. Vietnamese foreign net flows on the Ho Chi Minh exchange swung from a net *buy* of about 27 trillion VND in 2022 to a net *sell* of roughly 90 trillion VND in 2024 (≈ \$3.6 billion of net selling). When a single class of capital provider reverses that hard, it moves the whole index regardless of domestic fundamentals. It is the buy-side lesson at its starkest: *who provides the capital, and how patient they are, can matter more than what the issuers are actually worth* — a dynamic explored in [Foreign flows, ETFs and the index effect in Vietnam](/blog/trading/vietnam-stocks/foreign-flows-etfs-and-the-index-effect-vietnam). And it shows the index provider's hidden power again: a decision to upgrade Vietnam from "frontier" to "emerging" in a major index would force passive funds to buy, flooding in foreign capital by rule, not by conviction.

### LTCM: when an intermediary forgets it is a capital provider's agent

In 1998, the hedge fund Long-Term Capital Management — staffed by Nobel laureates — blew up so spectacularly that the Federal Reserve organized a rescue to prevent its failure from cascading through the banks that had lent it money. LTCM was a capital provider (it invested its partners' and clients' money) and a fee-charger (2-and-20), but it had borrowed so heavily from the sell-side that its collapse threatened everyone it traded with. It is a reminder that the neat boxes on our map blur under leverage: a buy-side fund big enough and levered enough becomes a systemic node, and the intermediaries who financed it discover they were bearing the risk all along. The detailed anatomy lives in [LTCM 1998: when genius failed](/blog/trading/finance/ltcm-1998-when-genius-failed); for the cast list, it is the case that proves the roles are real but never airtight.

## The takeaway: read any deal by asking "whose money, which arrow, what fee?"

If you remember one habit from this post, make it this three-part question, which you can now ask of any transaction in any capital market:

1. **Whose money is it?** Trace it back up the buy-side funnel — past the fund, past the asset manager, past the custodian — to the household, pension, insurer, or foreign saver whose savings are actually at risk. The money is almost never the middleman's; they are handling someone else's.

2. **Which arrow is this?** Is money flowing *to an issuer* (a primary transaction — capital is being raised, something real gets funded) or merely *between investors* (a secondary transaction — claims are changing hands, the issuer is a bystander)? The news blurs these constantly; you no longer have to.

3. **What is each middleman paid, and for what service?** Underwriting spread for distribution and certainty; commission and order-flow payment for access; bid-ask spread for immediacy; listing and data fees for the venue and its information; basis-point fees for clearing and custody; a management fee on AUM for diversification and selection, plus a performance cut for the brave; a rating fee for credit shorthand; a license fee for the definition of "the market" itself.

Once you can answer those three for any situation, the financial press stops being a wall of jargon and becomes a story with a clear cast. You will see that the issuer is paid once and the intermediaries are paid forever; that the fee you see is often smaller than the fee you don't; that the buy-side's patience and the sell-side's liquidity are two halves of the same machine; and that this whole crowded apparatus exists for one purpose — to move a household's savings into a factory, a road, or a company that turns it into something more, and to give that household a way out the moment it wants its money back.

That is the genius hiding under all the fees: a relay race in which the baton — a tradable claim — can be passed to a fresh runner at any moment, which is the only reason the first runner ever agreed to carry it. Every player on the map is there to keep that baton moving.

## Further reading & cross-links

- [What is a capital market: how money finds its best use](/blog/trading/capital-markets/what-is-a-capital-market-how-money-finds-its-best-use) — the series opener: the savings-into-investment machine these players run.
- [Debt vs. equity: the two ways to raise capital](/blog/trading/capital-markets/debt-vs-equity-the-two-ways-to-raise-capital) — the two securities our corporate issuers create, and how they choose.
- [Inside an investment bank: how they make money](/blog/trading/finance/inside-an-investment-bank-how-they-make-money) — the underwriting spread and every other revenue line of the sell-side, in depth.
- [Stock exchanges and clearinghouses](/blog/trading/finance/stock-exchanges-and-clearinghouses) — how the venue and the CCP we met here actually match orders and guarantee trades.
- [The yield curve explained](/blog/trading/fixed-income/the-yield-curve-explained-the-most-important-chart-in-finance) — how the government issuer's bonds set the price of all other capital.
- [Securitization: how banks turn loans into securities](/blog/trading/banking/securitization-how-banks-turn-loans-into-securities) — the SPV issuer in full, including how the 2008 version went wrong.
- [Foreign flows, ETFs and the index effect in Vietnam](/blog/trading/vietnam-stocks/foreign-flows-etfs-and-the-index-effect-vietnam) — the foreign capital provider and the index provider's power in a smaller market.
- [LTCM 1998: when genius failed](/blog/trading/finance/ltcm-1998-when-genius-failed) — when the lines between provider, intermediary, and systemic risk dissolved.
