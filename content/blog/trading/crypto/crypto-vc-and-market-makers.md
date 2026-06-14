---
title: "Crypto VC and Market Makers: The Real Power Structure Behind the Tokens"
date: "2026-06-15"
publishDate: "2026-06-15"
description: "How a small set of venture funds and trading firms fund the projects and supply the liquidity that actually move crypto tokens, and why their interests rarely line up with retail buyers."
tags: ["crypto", "venture-capital", "market-makers", "tokenomics", "liquidity", "defi", "wintermute", "alameda"]
category: "trading"
subcategory: "Crypto"
author: "Hiep Tran"
featured: true
readTime: 40
---

> [!important]
> **TL;DR** — Behind crypto's loud, retail-facing surface sits a small group of venture funds (a16z crypto, Paradigm, Multicoin) and trading firms (Wintermute, Jump, GSR, and the late Alameda) that actually fund the projects and supply the liquidity, and their incentives often run against the people buying on launch day.
>
> - A token is not a share. Insiders buy it years early and far cheaper, then sell into a public market with no IPO gate, no lockup enforced by a stock exchange, and no quarterly disclosure.
> - Venture funds buy tokens at private prices like \$0.02 or \$0.10 and the token may launch publicly at \$2, a 20x to 100x on paper before retail can even click "buy".
> - Market makers borrow a project's tokens, quote both sides of the order book to make it tradeable, and keep call options that pay off if the price rises, so they help launch the very tokens they later profit from.
> - The "VC coin" critique is about float and FDV: a token can show a \$10 billion fully diluted valuation while only 5% of supply trades, so a tiny pool of liquidity props up a huge headline number.
> - The two famous blow-ups, Jump's entanglement with Terra and Alameda's with FTX, were not bugs in this structure, they were the structure taken to its extreme.

The diagram above is the mental model: in crypto, the price you see on launch day is the *last* price in a chain that started years earlier, in private rooms, at numbers retail never gets. By the time a token is on a public exchange, venture funds already own a cheap chunk of the supply, a market maker is already quoting both sides of the book, and an exchange has already decided to list it. The froth you scroll past on social media is the visible tip; the load-bearing structure is a handful of funds and trading firms whose names most buyers never learn.

![Pipeline of a token from seed to unlocks](/imgs/blogs/crypto-vc-and-market-makers-1.png)

This is not a conspiracy and it is not (mostly) illegal. It is a *plumbing* story. Equity markets have venture capitalists and they have market makers too; what makes crypto different is that the same plumbing runs faster, with weaker disclosure, and with the early buyers able to exit on a public market years sooner than an equity investor ever could. That speed and that opacity are what make the conflicts sharp. This piece builds the whole structure from zero, defines every term, walks through the arithmetic with real dollar figures, and then shows you where it has already played out in real markets. No predictions, no advice, no shilling. Just the machinery.

## First principles: the words this story turns on

Before we name a single fund, we need a shared vocabulary. Crypto borrows half its words from finance and invents the other half, and the whole power story collapses into confusion if any of these terms is fuzzy. Read this section even if you think you know it; the conflicts later hinge on the *precise* meaning of "token", "float", and "FDV".

**Blockchain.** A shared, append-only ledger that many independent computers keep copies of and agree on, so that no single party controls the record. When you hear "on-chain", it means the action (a transfer, a trade, a loan) is recorded on this public ledger for anyone to inspect. "Off-chain" means it happened in a private system, like a company's internal database, where you have to trust the company.

**Token.** A unit of value recorded on a blockchain. Some tokens are money-like (a stablecoin pegged to a dollar). Some are "governance" or "utility" tokens that represent a stake in, or access to, a project. Crucially, a token is *not* a share of a company in the legal sense. Owning a project's token usually gives you no claim on its profits, no board vote enforced by company law, and no dividend. It gives you a price that goes up and down, and sometimes a vote inside the project's own software.

**Wallet.** Software (or hardware) that holds the cryptographic keys controlling your tokens. "Self-custody" means you hold the keys; "custody" means someone else (an exchange, a custodian) holds them for you, and you trust them to give the tokens back. This distinction is the whole plot of [the FTX collapse](/blog/trading/crypto/ftx-collapse-sam-bankman-fried): customers thought their tokens were in custody and safe, while the firm was lending them out.

**Smart contract.** A program that runs on the blockchain and executes automatically when conditions are met, with no human in the loop. A token's rules, its supply, its vesting schedule, its trading pool, are often enforced by smart contracts. This matters because "the code locks the founders' tokens for two years" is a verifiable claim, while "the founders promised not to sell" is not.

**Exchange.** A venue where tokens are bought and sold. A *centralized* exchange (Binance, Coinbase) is a company that holds your tokens and matches your orders internally, like a brokerage; we go deep on these in [centralized crypto exchanges](/blog/trading/crypto/centralized-crypto-exchanges-binance-coinbase). A *decentralized* exchange (DEX) is a smart contract that lets you swap tokens against a pooled stockpile of assets, with no company in the middle. Both need *liquidity* to function.

**Liquidity.** How easily you can buy or sell something without moving its price. A liquid market has lots of resting orders close to the current price, so a \$10,000 trade barely nudges it. An illiquid market is thin: a single \$10,000 sell order might crash the price 30%, because there simply isn't enough on the other side. Liquidity is not a property of a token; it is *provided* by someone, and that someone is usually a market maker.

**Two-sided quoting.** The core job of a market maker: at the same time, post an order to *buy* slightly below the current price (the "bid") and an order to *sell* slightly above it (the "ask"). The gap between them is the *spread*, and capturing that spread, again and again across thousands of trades, is how the market maker earns its keep. By always standing ready on both sides, it lets a retail buyer find a seller and vice versa, even when no natural counterparty is around at that exact second.

**Stablecoin.** A token designed to hold a fixed value, almost always \$1. A *collateralized* stablecoin (like USDC) is backed by real dollars or bonds held in reserve. An *algorithmic* stablecoin (like the doomed UST) tries to hold its peg with a software mechanism and a paired token instead of real reserves, a design that, as we'll see with Terra, can unravel in a spiral. *Collateral* just means the asset pledged to back a loan or a token; if the borrower fails, the collateral is seized.

**Oracle.** A service that feeds outside-world data (prices, especially) into a blockchain, since a smart contract can't natively see the price of anything off its own chain. Oracles matter here because a token's on-chain price, and decisions made from it (like whether a loan is under-collateralized), depend on the oracle being accurate; a manipulated or thin price feeds straight into the contract's logic.

**TVL (total value locked).** The dollar value of all assets deposited into a protocol's smart contracts, the headline "size" metric for a DeFi project. Like FDV, TVL is a number that can be inflated by counting tokens at a price the market couldn't actually sustain, so it should be read with the same skepticism.

**Venture capital (VC).** Funds that invest in early-stage companies (or projects) in exchange for a stake, betting that a few big winners pay for the many that fail. A VC raises a pool of money from "limited partners" (pension funds, endowments, rich individuals), then deploys it over several years, and aims to return multiples of that pool. In equity, the VC waits for an IPO or acquisition to cash out. In crypto, as we'll see, the wait can be far shorter.

**Token sale / SAFT.** How a project raises money by selling its future tokens. A *SAFT* (Simple Agreement for Future Tokens) is a contract where an investor pays now for tokens that don't exist yet and will be delivered when the network launches. A "private round" or "seed round" is a token sale to a small set of insiders at a low price, long before the public can buy. A "public sale" or "launch" is when the token first trades on an open market.

**Vesting and unlock schedules.** Insiders' tokens are usually *locked* for a period so they can't dump everything on day one. A "cliff" is a date before which nothing unlocks; after the cliff, tokens "vest" (become sellable) gradually, often monthly, over a year or more. An *unlock* is a scheduled date when a chunk of previously locked tokens becomes tradeable, increasing the *circulating* supply, and these dates are publicly known in advance.

**Circulating supply vs total supply.** *Circulating supply* is the number of tokens currently free to trade. *Total supply* (or *max supply*) is every token that will ever exist, including the locked insider tokens that haven't unlocked yet. The gap between them is the overhang: supply that *will* hit the market on a known schedule.

**Fully diluted valuation (FDV).** The market price of one token multiplied by the *total* supply, as if every token, locked or not, were trading today. *Market capitalization* uses only the circulating supply. When circulating supply is small, FDV can be wildly larger than market cap, and that gap is the single most important number in the "VC coin" critique we'll dissect later.

**The conflict, in one sentence.** Insiders (VCs and the team) buy cheap and early, with tokens that unlock on a schedule, into a market whose liquidity is supplied by a firm that often also holds an upside stake. Retail buys at launch, at the top of that chain, into a token whose supply is about to grow. The structure does not *force* anyone to behave badly, but it lines up the incentives so that, when insiders sell, retail is frequently the buyer on the other side. That phrase, "retail as exit liquidity", is the dark heart of this whole story.

With those nine terms pinned down, we can build the power structure piece by piece.

## What crypto VC actually is, and why a token is not a share

Venture capital in tech is an old, well-understood machine. A fund raises, say, \$500 million from limited partners, then over four or five years writes checks into startups in exchange for *equity*, meaning actual ownership shares. The fund's bet is asymmetric: most startups return zero, but one or two might return 50x or 100x, and those winners carry the whole fund. The catch is *time and illiquidity*. Equity in a private startup can't be sold on any public market. The VC's money is locked up, on paper, until the startup either goes public (an IPO) or gets acquired, which historically takes seven to ten years. The VC can't just decide one morning to sell.

Crypto VC works the same way at the front end and very differently at the back end. The funds still raise pools from limited partners, still write early checks, still bet on a few winners. a16z crypto, the dedicated crypto arm of Andreessen Horowitz, raised a \$350 million first fund in 2018, then a \$2.2 billion fund in 2021, and a roughly \$4.5 billion fund in 2022, by far the largest dedicated crypto venture vehicle to that point. Paradigm, founded by Coinbase co-founder Fred Ehrsam and former Sequoia partner Matt Huang, raised around \$2.5 billion across its funds and built a reputation as the sharpest technical investor in the space. Multicoin Capital, smaller but influential, made concentrated early bets on networks like Solana. These are real institutional funds with real diligence, real legal teams, and real losses (a16z and Paradigm both took heavy paper markdowns in the 2022 downturn).

The difference is *what they buy and how they exit*.

![Equity VC versus token VC economics](/imgs/blogs/crypto-vc-and-market-makers-4.png)

In equity VC, the asset is a share, illiquid for years. In token VC, the asset is increasingly a *token* (or a SAFT, a right to future tokens), and a token can trade on a public market within months of launch. The way this works is that the fund buys tokens in a private round at a deep discount, agrees to a vesting schedule, and then, once those tokens unlock and the project is live on exchanges, the fund can *sell on the open market*. There is no IPO gatekeeper, no underwriter, no lockup that a stock exchange enforces. The only thing standing between a token VC and a sale is its own vesting contract, and once that vests, the exit is a liquid public order book, not a years-away IPO. Conceptually, tokenizing the cap table collapses the VC's exit timeline from a decade to, sometimes, a single year.

That single structural change, *liquid early exit*, reshapes every incentive downstream. An equity VC that wants a return must make the company genuinely valuable enough to go public, a process that forces years of building and a public audit of the books. A token VC can profit the moment there is a liquid market and a price above its entry, regardless of whether the project ever ships anything durable. The market it sells *into* is the retail demand at launch. This is not inherently fraudulent, plenty of token projects are real, but it changes what "success" means for the earliest, cheapest holders.

#### Worked example: the 20x on paper

Suppose a fund participates in a project's seed round and buys 10 million tokens at \$0.10 each. The check is:

```
tokens bought      = 10,000,000
price per token    = $0.10
amount invested    = 10,000,000 x $0.10 = $1,000,000
```

The fund has put in \$1 million. The token is not yet public; there is no market price.

A year later the project launches on exchanges and the token opens trading at \$2.00. On paper, the fund's stake is now:

```
tokens held        = 10,000,000
public price       = $2.00
paper value        = 10,000,000 x $2.00 = $20,000,000
paper gain         = $20,000,000 - $1,000,000 = $19,000,000
multiple           = $20,000,000 / $1,000,000 = 20x
```

A 20x on paper, before retail could click "buy" even once. Now the word "paper" is doing real work here. The fund cannot instantly realize \$20 million: its tokens are vesting, and if it tried to dump all 10 million into the order book at once, the price would collapse long before it sold the last token. But the *entry price gap* is the entire point. The fund's break-even is \$0.10; the retail buyer's break-even is \$2.00. The fund can sell, in pieces, anywhere above \$0.10 and still profit, while a retail buyer at \$2.00 needs the price to *stay* at \$2.00 just to break even. **The intuition: the earliest, cheapest entry is what turns a token launch from a coin flip into a near-guaranteed win for insiders and a coin flip for everyone else.**

This entry-price asymmetry is why the *price chart you can see* tells you almost nothing about who is winning. A token down 50% from its launch is a disaster for the retail buyer at \$2.00 and still a 10x for the seed fund at \$0.10.

### How a crypto VC fund actually makes money

It helps to see *whose* money the fund is playing with, because that shapes its behavior. A crypto VC fund is structured like any venture or hedge fund: a small team of "general partners" (the GPs, the people you read about) raises a pool from "limited partners" (the LPs, who put up almost all the actual capital but make no investment decisions). The fund is then compensated with the same template that the rest of the money-management industry uses, the model we unpack in [how hedge funds work](/blog/trading/finance/how-hedge-funds-work-leverage-2-and-20): a *management fee* (often around 2% of the fund per year, to pay salaries and keep the lights on) and *carried interest* or "carry" (often around 20% of the profits, the GPs' real upside). So a \$4 billion fund might collect roughly \$80 million a year in management fees alone, before any investment has paid off, and then keep a fifth of every dollar of gain on top.

This matters because it changes the *time pressure*. An LP commits money for a fixed fund life (often ten years) and expects the capital back, with a multiple, inside that window. The GPs are under pressure to *show returns* and *return capital*, and in crypto, the fastest path to returning capital is a liquid token, not a decade-away IPO. The fee-and-carry structure therefore rewards the fund for getting projects to a tradeable token quickly and for marking its remaining bags at a high valuation in the meantime, because carry is calculated on gains, and an inflated FDV makes the gains look larger on paper. The same fee model that built TradFi asset management thus imports a specific incentive into crypto: launch tokens, mark them high, and find liquidity to exit through. None of that is unique to crypto, but the liquid-token exit makes it act faster and with sharper edges.

#### Worked example: management fee versus carry on a crypto fund

Take a \$2 billion crypto venture fund charging "2 and 20". The management fee, charged on the committed capital each year, is:

```
fund size          = $2,000,000,000
management fee rate = 2%
annual fee         = $2,000,000,000 x 0.02 = $40,000,000
```

\$40 million a year, every year, regardless of performance, just for managing the pool. Over a ten-year fund life that is up to \$400 million in fees alone. Now suppose the fund's investments eventually return \$5 billion in total value (a 2.5x on the \$2 billion). The profit, and the carry on it, is:

```
total returned     = $5,000,000,000
capital invested   = $2,000,000,000
profit             = $5,000,000,000 - $2,000,000,000 = $3,000,000,000
carry rate         = 20%
GP carry           = $3,000,000,000 x 0.20 = $600,000,000
```

The GPs keep \$600 million of carry, on top of the \$400 million of fees, while the LPs take the rest of the gain and bear most of the loss if the fund disappoints. **The intuition: the fund earns whether or not retail does well, and its biggest payday, carry, is computed on gains it is incentivized to realize fast and mark high, which is exactly why "liquid token exit" reshapes a crypto VC's behavior toward early launches.**

## Token economics: cliffs, unlocks, and the supply nobody sees yet

The second pillar of the structure is *supply mechanics*. In equities, the share count is relatively stable and any big issuance is disclosed and slow. In tokens, the future supply is baked into a vesting schedule from day one, and that schedule is a clock counting down to selling pressure.

When a project launches, only a fraction of its tokens are *circulating*, free to trade. The rest, the insider allocations to VCs, the team, the treasury, are locked under vesting contracts. The standard shape is a cliff (often one year, during which nothing unlocks at all) followed by linear vesting (often two to three more years, where tokens drip out monthly). So a token might launch with 15% of supply circulating, hit its one-year cliff, and then begin releasing several more percent of total supply *every single month* for years.

![Token supply ownership stack](/imgs/blogs/crypto-vc-and-market-makers-5.png)

Look at who owns the stack. A typical allocation might be 35% to investors (the VCs), 20% to the team and founders, 25% to a treasury or foundation, and only 20% as genuine "community float", the tokens that anyone can buy and that actually trade freely at launch. (Allocations vary widely; these are illustrative, in the range many 2021 to 2024 launches used.) The visible market, the order book retail trades against, is built on that thin 20% slice. The other 80% is overhang, locked but scheduled to arrive.

Why does an unlock pressure the price? Because the people whose tokens just unlocked are frequently sellers. A VC sitting on a 20x wants to lock in *some* of that gain; an employee who has been working for tokens wants to pay rent. None of them has to sell, but at the margin, an unlock pours new sellable supply into a market whose *buying* demand has not grown to match. More sellers, same buyers, lower clearing price. Unlock dates are public, so sophisticated traders often position *ahead* of them, shorting into the unlock, which can push the price down before a single insider token is even sold.

#### Worked example: an unlock pressures the price

A token trades at \$1.00 with 100 million tokens circulating, so its market cap is \$100 million. The total supply is 1 billion tokens; the other 900 million are locked. On a known unlock date, 50 million previously-locked VC tokens vest and become sellable.

Before the unlock:

```
circulating supply = 100,000,000
price              = $1.00
market cap         = 100,000,000 x $1.00 = $100,000,000
```

The unlock adds 50 million tokens to the sellable pool, a 50% jump in circulating supply overnight:

```
new circulating    = 100,000,000 + 50,000,000 = 150,000,000
```

Now suppose the VCs holding those 50 million tokens decide to sell just one fifth of them, 10 million tokens, into the market over the following week. The market has to *absorb* \$10 million of selling (at the old price) without an equivalent jump in buying. Thin order books rarely absorb that cleanly. If the selling pushes the price down to \$0.70:

```
new price          = $0.70
loss for a holder who bought at $1.00 = ($1.00 - $0.70) / $1.00 = 30%
```

A retail holder who bought at \$1.00 is down 30%, purely because scheduled supply arrived and some insiders took profit, with no change in the project's actual technology or adoption. **The intuition: in tokens, the supply curve is a calendar, and the biggest scheduled events in that calendar are insider unlocks that retail almost never prices in.**

This is why "tokenomics" is not a buzzword; it is the single most predictive thing about whether a token launch is structurally tilted against new buyers. A low circulating float at launch with heavy near-term unlocks is the classic setup we'll name later as "low-float, high-FDV".

## What a crypto market maker does, and the token-loan deal that powers launches

Now the second class of insider: the trading firm. A *market maker* (MM) is a firm that stands ready to buy and sell a token continuously, posting a bid and an ask, so that anyone wanting to trade can. Without a market maker, a freshly launched token's order book is a desert: maybe a few scattered orders, huge gaps, wild price swings on tiny trades. A market maker fills that order book with deep, tight quotes, and in doing so it *creates* the liquidity that makes the token feel like a real, tradeable asset.

In traditional markets, market making is a well-established business; we cover its mechanics in [market makers and high-frequency trading](/blog/trading/finance/market-makers-and-high-frequency-trading). The crypto version, dominated by firms like Wintermute, Jump Crypto (the crypto arm of Chicago trading giant Jump Trading), GSR, and, until its 2022 implosion, Alameda Research, runs on the same principle, capture the spread across enormous volume, but with two twists that make it central to the power structure.

The first twist is *scale and reach*. Wintermute alone has reported quoting markets in thousands of trading pairs across dozens of venues, both centralized exchanges and on-chain DEXs, handling enormous daily volumes. These firms are not passive; they are the connective tissue between every venue, arbitraging price differences and keeping markets aligned. A crypto market maker also has to bridge two very different kinds of venue. On a *centralized* exchange it posts limit orders into a classic order book, exactly like an equity market maker. On a *decentralized* exchange it interacts with an **automated market maker** (AMM): a smart contract that holds a *liquidity pool*, a stockpile of two tokens, and prices swaps by a fixed formula rather than by matching buyers to sellers. Anyone can deposit into a liquidity pool to earn a share of the trading fees, but a professional market maker manages inventory across both worlds at once, hedging a position it took on a DEX by trading the opposite way on a centralized venue. The depth of a token's order book, the dollar value resting near the current price, is largely *its* depth, which is why the firm's choice to quote tightly or to step back is, in practice, the token's liquidity.

The spread it captures sounds tiny per trade but compounds violently across volume. Suppose a market maker quotes a token with a 0.2% spread (bid at \$0.999, ask at \$1.001 on a \$1.00 token) and trades \$50 million of volume in a day, capturing roughly half the spread on average. That is `$50,000,000 x 0.001 = $50,000` of gross spread capture in a single day on a single token, before any option upside, and a firm like Wintermute does this across thousands of pairs simultaneously. The spread is the wage for standing in the middle; the token-loan option, below, is the bonus.

The second twist, the one that ties them directly to the VC story, is the **token-loan market-making deal**. Here is how it works. A new project has a problem: its token needs deep liquidity on day one, but the project itself doesn't want to (or can't) post millions in inventory to quote the book. So the project *lends* the market maker a large batch of its own tokens. The market maker uses those tokens as inventory to quote both sides of the market, providing the liquidity. In exchange, the deal is typically structured as a *loan plus a call option*: the market maker borrows, say, 10 million tokens, agrees to return them (or their value) at the end of the term, and is granted call options to buy a slice of those tokens at a fixed "strike" price. If the token price rises above the strike, the market maker exercises the options and keeps the upside. If it falls, the options expire worthless and the firm just returns the borrowed tokens.

![Graph of how VC market maker exchange and project intertwine](/imgs/blogs/crypto-vc-and-market-makers-3.png)

Trace the graph. The VC funds the project. The VC, having done dozens of these, often *introduces* the project to a market maker it trusts. The project lends tokens to the market maker. The market maker quotes the order book on the exchange. The project lists on the exchange. The exchange sells to retail. And the price that emerges is set, in large part, by the few parties at the top of that chain, not by the retail crowd at the bottom. The same small network of funds and trading firms recurs across launch after launch, which is exactly why "a handful of firms" is not hyperbole.

This structure is not sinister on its face. New tokens genuinely need liquidity, and someone has to provide it; the option structure compensates the market maker for the risk of holding a volatile, possibly-worthless new token. But notice the conflict it bakes in. The market maker now *holds upside in the very token it is making a market in*. It profits more if the price goes up. And it controls a large chunk of the visible liquidity, so its decision to quote tightly (or to pull its quotes) directly moves the price. The firm that makes the market also benefits from where the market goes.

#### Worked example: a token-loan market-making deal

A project lends a market maker 10 million tokens to provide liquidity for one year. The token launches at \$1.00. The deal grants the market maker call options to buy those 10 million tokens at a strike of \$1.50 each, in return for tight quoting and the liquidity service. The market maker also captures the bid-ask spread on everything it trades in the meantime.

Case A, the token rises to \$3.00 by the end of the term:

```
strike price       = $1.50
market price       = $3.00
profit per token   = $3.00 - $1.50 = $1.50
tokens under option= 10,000,000
option profit      = 10,000,000 x $1.50 = $15,000,000
```

The market maker exercises, buys 10 million tokens at \$1.50, and they're worth \$3.00, a \$15 million gain on the option alone, on top of a year of spread capture. It returns the borrowed tokens (or settles the difference per the contract) and keeps the rest.

Case B, the token falls to \$0.40:

```
strike price       = $1.50
market price       = $0.40
```

The option is far out of the money (why buy at \$1.50 what's worth \$0.40?), so it expires worthless. The market maker simply returns the borrowed tokens. Its downside on the option leg is zero; it only loses if its day-to-day inventory bled money, which tight risk management is designed to prevent.

So the structure is: heads, the market maker makes \$15 million; tails, it walks away near flat. **The intuition: a token-loan deal hands the market maker a free option on the token's success, which is exactly why these firms are not neutral bystanders, they are stakeholders in the launches they make markets for.**

This is the legitimate version. The illegitimate version, dumping the borrowed tokens to crash the price and then buying back cheap, or coordinating with the project to paint a misleadingly active market, is a real risk that has surfaced in lawsuits and disputes, and it is precisely *because* the market maker holds the inventory and the options that the temptation exists.

## FDV vs circulating supply: the number that hides the overhang

We now have enough to dissect the single most weaponized number in crypto: *fully diluted valuation*. Recall the definitions. Market cap is price times *circulating* supply, the tokens trading today. FDV is price times *total* supply, every token that will ever exist, as if all of it were already on the market. When the circulating supply is a small fraction of the total, FDV towers over market cap, and the headline FDV becomes a marketing number with very little liquidity beneath it.

#### Worked example: \$10 billion FDV on 5% float

A token launches with a total supply of 10 billion tokens. At launch, only 5% of them, 500 million tokens, are circulating; the other 95% are locked under VC, team, and treasury vesting. The opening price is \$2.00.

The circulating market cap:

```
circulating supply = 500,000,000
price              = $2.00
market cap         = 500,000,000 x $2.00 = $1,000,000,000  ($1B)
```

The fully diluted valuation:

```
total supply       = 10,000,000,000
price              = $2.00
FDV                = 10,000,000,000 x $2.00 = $20,000,000,000  ($20B)
```

Headlines scream a \$20 billion token. But the *actual* money supporting that price is the buying against 500 million circulating tokens, a real market cap of \$1 billion. The ratio of FDV to market cap is 20 to 1. To hold the price at \$2.00 as the other 9.5 billion tokens unlock over the next few years, the market would need to absorb up to \$19 billion of *new* sellable supply at today's price, an amount of fresh buying demand that almost never materializes.

Now run the dilution forward. Suppose a year in, circulating supply has grown to 2 billion tokens through unlocks, a 4x increase in float, while genuine buying demand has only doubled the dollars in the market. The price that clears is roughly:

```
dollars in market (doubled) ~ $2,000,000,000
new circulating supply       = 2,000,000,000
implied price ~ $2,000,000,000 / 2,000,000,000 = $1.00
```

The price has halved, not because the project failed, but because supply outran demand on the unlock calendar. **The intuition: FDV is the price of the float projected onto the whole supply, so a giant FDV on a tiny float is a promise that demand will quadruple just to keep the price flat, a promise the unlock schedule almost guarantees will be broken.**

This is the mechanical core of the "low-float, high-FDV" critique. A project can engineer a sky-high headline valuation by listing with a tiny float, the VCs and team get to mark their locked bags at that inflated FDV, and the retail buyers provide the demand that holds the price up, right up until the unlocks start and the supply they didn't see coming arrives.

## The "VC coin" critique and the influence over what even launches

Put the pieces together and you get the term of art that crypto natives use, half as analysis and half as insult: the *VC coin*. It describes a token that launches with a low circulating float, a high FDV, heavy insider allocations, and a near-term unlock cliff, in other words, a token engineered so that the insiders' paper gains are enormous and the structural pressure on the price, once unlocks begin, is downward. The critique is not that these tokens are fake; many have real teams and real software. The critique is that the *cap table is the product*: the token's price discovery is dominated by insiders who entered far cheaper and who will be selling on a schedule, and retail is structurally positioned as the buyer they sell to.

There is a sharper, second layer to the power story: the funds and firms don't just profit from launches, they *decide which projects get to launch at all*. A founder with an idea needs capital; the largest checks come from a small set of funds. Those funds then open doors: introductions to the market makers who will provide launch liquidity, to the exchanges that will list the token, to the other funds that will fill out the round. A project the big funds *won't* back has a much harder path to the liquidity and listings it needs to reach retail. So the same names recur, the same funds, the same market makers, the same exchanges, not because of a formal cartel, but because the network effects of capital, liquidity, and listings concentrate power in whoever already has all three.

The listing step deserves its own look, because it is where the three powers meet. A centralized exchange does not list a token out of charity; historically, getting a major exchange to list a token has involved some combination of a listing fee, a marketing commitment, and, critically, a requirement that the token arrive with a designated market maker already lined up to quote the book. The exchange doesn't want a token that trades like a desert on its venue, that's bad for its own reputation and its taker fees, so it effectively *requires* the market-maker relationship before it will list. That requirement closes the loop: the VC introduces the project to the market maker, the market maker's presence helps satisfy the exchange's listing bar, and the exchange opens the gate to retail. Each insider needs the others, and a founder outside that network has to assemble all three from scratch. This is why the criticism is structural rather than moral: no one has to break a rule for the same dozen firms to sit at every important table.

The conflict can also run *through* the exchange itself. Several large exchanges run their own venture arms, Coinbase Ventures, Binance Labs (later rebranded), which invest in the very projects their exchanges may go on to list. An exchange investing in a token it then lists is yet another role-combination: the gatekeeper holds a cheap early stake in what it admits. As with the market maker holding options, this is not automatically abusive, but it stacks one more interested party on the insider side of the table, and it is part of why "who is on the cap table" is a question that, in crypto, reaches all the way up to the exchange.

![Tree of the crypto market power structure](/imgs/blogs/crypto-vc-and-market-makers-7.png)

The tree is the summary. Above retail sit two insider classes. On one branch, *capital*: the venture funds, a16z and Paradigm with their multi-billion-dollar vehicles, Multicoin and a long tail of smaller seed funds. On the other branch, *liquidity*: the market makers, Wintermute and GSR quoting thousands of pairs, and the two that blew up, Jump and Alameda. Both branches sit *above* the retail buyer, who interacts only with the visible exchange price, the very last and highest number in the chain. The structure is not hidden, the funds publish their portfolios, the unlock schedules are on-chain, but the *implications* of the structure are rarely spelled out to the people buying at the top.

![Matrix of who funds and who quotes](/imgs/blogs/crypto-vc-and-market-makers-2.png)

The matrix sharpens the division of labor. a16z crypto and Paradigm fund projects (a16z's dedicated crypto funds total roughly \$7.6 billion across vintages) and do not, as their main business, quote order books or run token-loan desks. Wintermute and GSR are the inverse: their business *is* quoting markets, Wintermute across thousands of pairs, with token loans as a standard service to projects. Jump and Alameda did all of it, fund, quote, and lend, and in Alameda's case, that all-of-it model with no firewall between the trading firm and the exchange is exactly what detonated. The lesson encoded in the matrix is that the *separation* of these roles is a safety feature, and the firms that collapsed are the ones that erased it.

## When the structure collapses: Jump and Terra, Alameda and FTX

The two defining blow-ups of 2022 are not exceptions to this structure. They are the structure with its safety separations removed, and they show exactly how the conflicts turn into catastrophe.

![Timeline of crypto VC and the 2022 collapses](/imgs/blogs/crypto-vc-and-market-makers-6.png)

The timeline frames it. The mega-funds raised at the top of the cycle, a16z's \$4.5 billion fund closed in 2022, the very year the firms that supplied liquidity to the market came apart. First Terra. Then FTX and Alameda. The capital came in at the peak; the liquidity backbone broke right after.

**Jump and Terra.** Terra was a blockchain whose flagship product was UST, an "algorithmic" stablecoin meant to hold \$1 through a mint-and-burn mechanism with a sister token, LUNA, rather than through dollar reserves in a bank. To bootstrap the peg early on, the system leaned on market makers to keep UST trading at \$1. In May 2021, UST had briefly de-pegged, falling under \$1, and recovered, and it later emerged through litigation that Jump (specifically Jump's crypto trading arm) had stepped in to buy UST and restore the peg, while, according to the allegations, holding agreements that gave it the right to buy LUNA at a steep discount, an arrangement that could be worth hundreds of millions if confidence held. The point is not to litigate who did what; the point is the *shape*. The market maker propping the peg also held a large discounted stake in the system whose survival it was propping. When UST de-pegged for good in May 2022, the mint-and-burn mechanism spiraled, and roughly \$40 billion of combined UST and LUNA value evaporated in days. The firm that had supplied the confidence-restoring liquidity was a stakeholder in the thing it was stabilizing.

**Alameda and FTX.** This is the structure taken to its logical extreme, and we dissect it fully in [the FTX collapse](/blog/trading/crypto/ftx-collapse-sam-bankman-fried). FTX was a centralized exchange. Alameda Research was a crypto trading firm and market maker. Both were founded and controlled by the same person, Sam Bankman-Fried. In a healthy structure, an exchange (which holds customer funds in custody) and a trading firm (which takes risk in the market) are walled off from each other, the exchange must not lend customer money to the trading firm. FTX and Alameda erased that wall. Alameda was permitted to run a massive negative balance on FTX, effectively borrowing customer deposits, and much of its "assets" were FTX's own token, FTT, a token whose value FTX itself controlled. When a public report questioned Alameda's balance sheet in November 2022, FTT crashed, customers rushed to withdraw, and FTX could not return their funds because Alameda had been using them. The hole was estimated at around \$8 billion.

#### Worked example: Alameda's circular collateral

The fatal mechanism was *marking a position against a token you control*. Suppose Alameda holds a large block of FTT, FTX's exchange token, and uses it as collateral to borrow against. Say it holds 100 million FTT, and FTT is "worth" \$25 on FTX.

```
FTT held           = 100,000,000
quoted price       = $25
"collateral value" = 100,000,000 x $25 = $2,500,000,000  ($2.5B)
```

On the books, that's \$2.5 billion of collateral, against which Alameda could borrow, including, it turned out, against FTX customer deposits. But the price of FTT was thin and substantially supported by FTX and Alameda themselves; there was nowhere near \$2.5 billion of real buying that would let anyone actually *sell* 100 million FTT at \$25. When confidence cracked and FTT fell from \$25 toward \$2:

```
new price          = $2
real collateral value = 100,000,000 x $2 = $200,000,000  ($0.2B)
collateral wiped out  = $2,500,000,000 - $200,000,000 = $2,300,000,000
```

Roughly \$2.3 billion of "collateral" vanished, because it was never real money, it was a token whose price the firm itself had been holding up. The loans made against it, including those funded by customer deposits, could not be repaid. **The intuition: when a firm marks its solvency against a token it controls and supplies the liquidity for, it isn't measuring its wealth, it's measuring its own marketing, and the moment outsiders stop believing, the number, and the firm, goes to near zero.**

Both blow-ups share a DNA: a firm that was supposed to *serve* the market (provide liquidity, hold custody) was instead a *stakeholder* in the asset it served, with no separation between the two roles. That is the conflict this entire piece is about, scaled until it broke.

## Common misconceptions

**"Crypto is decentralized, so there are no gatekeepers."** The technology can be decentralized while the *economics* are concentrated. A blockchain may have thousands of independent validators, but the *funding* of the projects built on it, the *liquidity* that makes their tokens tradeable, and the *listings* that reach retail flow through a small number of funds, market makers, and exchanges. Decentralized rails, centralized money. The two are not the same axis.

**"If a token is publicly traded, retail and insiders are on a level playing field."** They bought at different prices, on different timelines, with different information. The seed fund's break-even might be \$0.10 while the retail buyer's is \$2.00. The insider knows the unlock schedule cold and sizes positions around it; retail usually doesn't even know it exists. A shared *current* price is not a shared *cost basis*, and cost basis is what determines who can afford to sell into whom.

**"Market makers are neutral, they just provide liquidity."** Sometimes, but a market maker on a token-loan deal *holds an upside stake* in that token via call options, and it controls a large share of the visible liquidity. It is structurally not neutral: it profits if the price rises and it can move the price by quoting tighter or pulling its quotes. A neutral utility doesn't hold options on the thing it's a utility for.

**"A high FDV means the project is valuable."** FDV is just price times *total* supply, including all the locked tokens that aren't trading. A \$10 billion FDV can sit on top of a \$500 million real market with 95% of supply locked. The FDV is what insiders use to mark their bags; it is not a measure of money anyone could actually realize. A high FDV on a low float is closer to a warning label than a badge.

**"VCs and market makers make money the same way retail does, by the price going up."** Partly, but the *asymmetry* is the whole point. The VC's deep discount means it profits across a huge range of outcomes where retail loses; the market maker's option structure means it profits on the upside and walks away near-flat on the downside. Retail, buying at the top of the chain with no discount and no option, has neither cushion. Same direction, wildly different risk.

**"Terra and FTX were one-off frauds, the structure itself is fine."** The frauds were enabled *by* the structure: a firm being both the market maker and a stakeholder (Jump and Terra), and a firm being both the exchange-custodian and the trading firm (Alameda and FTX). The lesson regulators and the industry drew was about *separation of roles*, precisely because the conflicts are built into the structure and only stay latent as long as someone chooses not to exploit them.

## How it shows up in real markets

These are named, public episodes. Figures are approximate and as-of the events described.

**a16z crypto's 2021 to 2022 fund cycle.** Andreessen Horowitz built the largest dedicated crypto venture practice, scaling from a \$350 million fund in 2018 to a \$2.2 billion fund in 2021 and roughly \$4.5 billion in 2022, more than \$7 billion across vintages. The 2022 fund closed *after* the market had already turned, and the firm took substantial paper markdowns as token and equity valuations fell. It is the clearest example of crypto VC operating at institutional scale, with all the entry-price and exit-timeline advantages over retail that this piece describes, and also of the fact that even the biggest, smartest funds lose heavily when the cycle breaks.

**Paradigm and Multicoin's concentrated bets.** Paradigm raised around \$2.5 billion and made it a point of pride to underwrite deep technical diligence; Multicoin made early, concentrated bets on networks like Solana that returned enormous multiples on the way up and gave back large paper gains in 2022. Both illustrate the token-VC model: early private entry, vesting, and a liquid public market to exit into, on a timeline an equity VC could never match.

**Jump and the Terra / wstETH involvement.** Beyond the 2021 UST peg restoration, Jump's crypto arm was a major liquidity provider across the ecosystem, including in staked-ETH markets (wstETH, the wrapped form of staked Ether). Its central role meant that when Terra collapsed in May 2022, taking roughly \$40 billion of UST and LUNA value with it, the shock rippled straight through the market-making and lending firms that had been intertwined with it, contributing to the contagion that took down lenders and funds in the months after.

**Alameda's role inside FTX.** The defining demonstration that erasing the wall between an exchange and a trading firm is fatal. Alameda borrowed against FTX customer deposits, collateralized substantially by FTX's own FTT token, and when FTT collapsed in November 2022 the roughly \$8 billion hole in customer funds could not be filled. The event reset the entire industry's thinking on custody, segregation of funds, and the conflict of a single party controlling both the venue and a major trader on it.

**The 2024 low-float, high-FDV launches.** A wave of 2024 token launches drew sustained criticism for listing with very low circulating floats and very high fully diluted valuations, sometimes single-digit-percent floats supporting multi-billion-dollar FDVs, followed by steep declines as unlocks began. Commentators (including some VCs publicly) debated whether the model had become extractive, with insiders marking enormous paper gains at FDVs that the thin float and heavy unlock calendar made unsustainable. It is the "VC coin" critique playing out in real time, and the reason "float and FDV" became the first thing skeptics check on a new listing.

**Wintermute's 2022 on-chain exploit.** In September 2022, Wintermute, one of the largest crypto market makers, lost roughly \$160 million from its decentralized-finance operations to a smart-contract exploit traced to a flawed vanity wallet address. The firm stayed solvent and kept quoting, but the episode is a reminder that a market maker is also an enormous, concentrated holder of inventory and code-controlled funds, so its *operational* failure (a hack, a bug, a key compromise) is a single point of failure for the liquidity of everything it quotes. The same concentration that makes these firms efficient makes their accidents systemic.

**A market maker pulling liquidity in a crash.** Because market makers *provide* the liquidity, they can also *withdraw* it. In sharp crashes, market makers facing their own risk limits widen their spreads or pull their quotes entirely, and the order book that looked deep evaporates exactly when sellers need it most. This happened repeatedly during the 2022 deleveraging: tokens that traded smoothly in calm markets gapped down violently when the firms quoting them stepped back to protect their own books. It is the dark side of liquidity-as-a-service, the provider's first duty is to its own survival, not to the market's continuity.

**Regulators circling the structure.** After 2022, the conflicts described here stopped being purely an industry critique and became a regulatory one. The SEC and other agencies brought actions touching token sales (whether a SAFT-style sale is an unregistered securities offering), exchange-and-trading-firm combinations, and market-maker conduct, while the FTX prosecution put the exchange-trader conflict on trial directly. The unresolved legal question underneath all of it is the same one this piece opened with, *is a token a security?* If it is, the early private sales, the disclosure gaps, and the insider selling fall under decades of securities law; if it isn't, much of the structure sits in a gray zone. That ambiguity is itself part of why the structure has been able to operate as fast and as opaquely as it has.

## When this matters to you, and further reading

If you ever look at a crypto token, the structure in this piece is the lens that makes the chart legible. The questions it tells you to ask are concrete and mostly answerable from public data:

- **What is the circulating float versus the total supply, and what is the FDV?** A small float under a huge FDV means the price you see is propped on a thin slice of supply, with a lot more scheduled to arrive.
- **What is the unlock schedule?** Unlock dates are public and are the biggest predictable selling events in a token's life. A cliff in the next few months is a calendar of supply you'll be competing with.
- **Who funded it, and at what price?** If well-known funds entered at a deep discount, their cost basis is far below yours, which means they can profit selling into prices where you lose.
- **Who makes the market, and do they hold an upside stake?** A token-loan market maker is not a neutral utility; it is a stakeholder, and it controls liquidity it can pull.
- **Are any roles dangerously combined?** The lesson of Terra and FTX is that a single party acting as exchange, trader, and stakeholder at once is the configuration that breaks. Separation of roles is a safety feature; its absence is a red flag.

There is one more habit worth building: read every "size" number, FDV, TVL, market cap, daily volume, as a claim about *liquidity that may not exist*. A \$20 billion FDV, a \$5 billion TVL, a \$2.5 billion collateral balance, each is a price multiplied by a quantity, and the price is only real to the extent that someone would actually pay it for the *whole* quantity. The recurring failure in this piece, from the low-float VC coin to Alameda's FTT collateral, is the same arithmetic mistake made on purpose: treating a price that holds for a thin slice as if it held for the entire stack. Once you see that, the headline numbers stop being facts and become marketing you can choose to discount.

None of this is investment advice, and none of it says the structure is illegitimate. Venture funding builds real things; market making provides genuinely necessary liquidity; most projects are not frauds. The point is narrower and more durable: the *visible* crypto market, the price, the social froth, the launch hype, sits on top of an *invisible* one, the funds that supplied the capital and the firms that supplied the liquidity, and those insiders entered earlier, cheaper, and with better information than anyone buying at the top. Understanding who sits where in that chain is the difference between reading a price as a signal and reading it as someone else's exit.

To go deeper on the surrounding machinery: [the FTX collapse](/blog/trading/crypto/ftx-collapse-sam-bankman-fried) dissects the exchange-and-trading-firm conflict in full; [centralized crypto exchanges](/blog/trading/crypto/centralized-crypto-exchanges-binance-coinbase) explains the listing and custody layer the tokens flow through; [how hedge funds work](/blog/trading/finance/how-hedge-funds-work-leverage-2-and-20) shows the fee-and-leverage model these crypto funds and trading firms inherited from TradFi; and [market makers and high-frequency trading](/blog/trading/finance/market-makers-and-high-frequency-trading) explains the liquidity-provision business that crypto market makers run, faster and with weaker rules, on tokens.
