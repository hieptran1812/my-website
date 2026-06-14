---
title: "Stablecoins: Tether, Circle, and the Shadow Dollar System Inside Crypto"
date: "2026-06-15"
publishDate: "2026-06-15"
description: "How dollar-pegged crypto tokens work, why Tether and Circle quietly became a hundreds-of-billions shadow banking system, how the peg actually holds, and what makes a stablecoin one bad weekend away from a run."
tags: ["stablecoins", "tether", "circle", "usdt", "usdc", "crypto", "shadow-banking", "treasury-bills", "de-peg", "stablecoin-regulation", "defi"]
category: "trading"
subcategory: "Crypto"
author: "Hiep Tran"
featured: true
readTime: 41
---

> [!important]
> **TL;DR** — Stablecoins are crypto tokens engineered to be worth one dollar each, and two issuers, Tether and Circle, have quietly built a shadow dollar-banking system that holds hundreds of billions in reserves, is systemically important to all of crypto, is only lightly regulated, and is always one loss of confidence away from a run.
>
> - A fiat-backed stablecoin works like a coat-check: you hand the issuer \$1, it mints one token, and the token is supposed to be redeemable for \$1 forever. The peg holds because anyone can mint or redeem at par, and arbitrage closes any penny-sized gap.
> - Tether's USDT (~\$110 billion outstanding, as of 2024) and Circle's USDC (~\$35 billion) together hold well over \$120 billion in reserves, the bulk of it in US Treasury bills - which makes these crypto firms among the larger holders of short-term US government debt on Earth.
> - Tether earns billions a year in float income on those reserves while paying token-holders nothing; that is the same business as a money-market fund, and structurally close to "narrow banking."
> - The model fails the way any bank fails: by a run. USDC briefly fell to about \$0.88 over the March 2023 SVB weekend when \$3.3 billion of its cash was stuck in a failing bank; the algorithmic stablecoin UST went from ~\$18 billion to near zero in days in May 2022.
> - Regulators have noticed: the US is moving toward a stablecoin framework (the GENIUS Act direction) and Europe's MiCA already sets reserve and licensing rules. The core risk - lightly supervised entities issuing private dollars at scale - is exactly the one that bank regulation was invented to contain.

In 2014 a small company issued a crypto token and promised that each one would always be worth exactly one US dollar. A decade later, the descendants of that idea move more value across the world's blockchains than Visa moves across its card rails, and the two largest issuers sit on a pile of US Treasury bills big enough to rank them alongside entire countries on the list of America's creditors. They are not banks. They are barely regulated. Almost no ordinary person has heard their balance-sheet details. And the entire \$2-trillion crypto economy quietly assumes that, whatever else breaks, a stablecoin will always be worth a dollar.

The diagram above is the mental model: you put one real dollar into an issuer, it holds that dollar (or a Treasury bill bought with it) in reserve, it mints one token on a blockchain, the token circulates as money inside crypto, and at any time you can send the token back and get your dollar out. That single promise - redeemable at par, on demand - is the whole magic trick. When the promise is believed, the token trades at \$1.00 and nobody thinks about the reserves. When the promise is doubted, the token can fall, and the doubt itself becomes the thing that breaks it. This is a bank run wearing a hoodie.

![Pipeline showing a dollar deposited, a token minted, circulated, and redeemed](/imgs/blogs/stablecoins-tether-circle-shadow-dollar-1.png)

This post takes that machine apart slowly. We will build, from zero, every idea the story turns on: what a blockchain token even is, what "the peg" means, the three different ways a token can try to stay worth a dollar, what "reserves" really are, how minting and redemption work mechanically, and why a stablecoin is structurally almost identical to a money-market fund and to an old idea called narrow banking. Then we meet the two giants - Tether and Circle - and the controversies that dog them; we see exactly how arbitrage holds the peg with real arithmetic; we follow the reserves into the US Treasury market; we walk through the real de-peg episodes (USDC's scary weekend, Terra's total collapse); we explain why algorithmic stablecoins are built to fail; we look at how the US and Europe are starting to regulate this; and we close on the uncomfortable framing that these firms have become a shadow dollar system that almost nobody supervises. By the end you should be able to explain to anyone - precisely, with numbers - what a stablecoin is, why it usually works, and exactly how it can stop.

## Foundations: how a dollar token actually works

Before the rest of the post makes sense, eight ideas have to be solid. None of them is hard. Take your time here; everything later leans on this section.

### A blockchain, a token, and a wallet

A **blockchain** is a shared public ledger - a giant spreadsheet of who owns what - that is maintained not by one company but by thousands of computers around the world that agree, every few seconds, on the next set of entries. Nobody can quietly edit a past row, and anyone can read the whole thing. The two blockchains that matter most for stablecoins are **Ethereum** and **Tron**.

A **token** is just an entry on that ledger that represents a unit of something. When you "hold 1,000 USDT," there is no coin anywhere; there is a line in a smart contract's table that says your address owns 1,000 units. A **smart contract** is a small program that lives on the blockchain and enforces rules automatically - in this case, the rules for who can create tokens (mint), destroy them (burn), and transfer them. A **wallet** is simply the pair of keys that lets you sign instructions to move tokens you control. The key point: a stablecoin token is a database row whose value comes entirely from a promise made off the blockchain, by a real company holding real assets.

### What "pegged to the dollar" means

A **stablecoin** is a token designed to hold a stable value against some reference - almost always the US dollar, at a one-to-one ratio. "Pegged to \$1" means the issuer and the market both try to keep one token trading at one dollar. This is unusual: most crypto assets (Bitcoin, Ether) float freely and swing wildly. A stablecoin's entire purpose is to *not* move. It is the cash register of crypto - the thing you sell volatile coins into, the unit prices are quoted in, the asset that sits in lending pools.

The **peg** is not a law of nature. It is an equilibrium held in place by two forces: the issuer's promise to redeem at par, and arbitrageurs who profit whenever the market price drifts from \$1. Remove either force - freeze redemptions, or remove the reserves that make redemption credible - and the peg is just a number on a screen.

### Why crypto needed a dollar in the first place

Crypto was born wanting to escape the dollar, so why does it run on dollar tokens? Because volatility makes a currency useless for commerce. If you are a trader who just sold Bitcoin at a profit, you want to lock in dollars without wiring money back to a slow, gatekept bank - especially since many crypto exchanges have no bank relationship at all. A stablecoin lets you hold "dollars" that live on the same rails as your other tokens, move in seconds, settle 24/7, and never ask permission. It is the bridge between the volatile crypto world and the stable dollar world, and it never leaves the blockchain. That convenience is why stablecoins became the dominant settlement asset of the entire ecosystem.

### The three families: how a token tries to stay at one dollar

There are three fundamentally different designs, and the difference is the single most important thing in this whole post, because **how a stablecoin earns its peg determines how it breaks.**

- **Fiat-backed (reserve-backed):** For every token, the issuer holds about \$1 of real-world assets - cash, bank deposits, US Treasury bills. This is USDT and USDC. The peg rests on actual redeemable reserves. The risk is whether the reserves are real, safe, and reachable.
- **Crypto-collateralized (over-collateralized):** The token is backed not by dollars but by *other crypto* locked in a smart contract, and because crypto is volatile, you must lock up *more* than \$1 of crypto for every \$1 of stablecoin minted. This is **DAI**. The peg rests on having a fat cushion of collateral and automatic liquidation when it shrinks. The risk is a crash so fast the cushion evaporates.
- **Algorithmic:** The token is backed by *nothing solid* - instead, code and a sister token try to expand and contract supply to push the price back to \$1. This was **UST (TerraUSD)**. The peg rests entirely on confidence and a feedback loop. The risk is that the same loop runs in reverse and the whole thing implodes. It did.

![Tree of stablecoin types branching into fiat-backed, crypto-backed, and algorithmic](/imgs/blogs/stablecoins-tether-circle-shadow-dollar-2.png)

Hold this tree in your head. Most of crypto's stable value lives on the leftmost branch (fiat-backed). The most spectacular failures live on the rightmost (algorithmic). The middle branch is the cleverest engineering and the smallest share.

### Reserves: what actually backs the token

For a fiat-backed stablecoin, **reserves** are the pool of assets the issuer holds so it can honor redemptions. Not all reserves are equal. The safest, most liquid reserve is cash in a bank and very-short-term US **Treasury bills** (T-bills) - IOUs from the US government that mature in days to a year, are essentially default-free, and can be sold instantly for close to face value. Riskier reserves include longer bonds (which can lose value if rates rise - the exact trap that killed Silicon Valley Bank), corporate IOUs (**commercial paper**), loans to other companies, or even other crypto. The quality of the reserves is the quality of the peg. A token "backed" by illiquid loans is a token that may not be redeemable in a hurry.

### Minting and redemption: the door in and the door out

**Minting** is creating new tokens: an approved customer (usually a large institution, not a retail user) wires real dollars to the issuer, and the issuer's smart contract creates that many new tokens and sends them to the customer. **Redemption** (or burning) is the reverse: the customer sends tokens back to the issuer, the tokens are destroyed, and the issuer wires back real dollars. This door - open, two-way, at \$1 per token - is what anchors the price. In practice the door is often only open to large, vetted clients and may carry fees and minimums, which matters enormously when a panic hits, because retail holders cannot redeem directly; they can only sell on an exchange, where the price can fall below \$1.

### A de-peg, in one sentence

A **de-peg** is when the market price of a stablecoin drifts meaningfully away from \$1 - usually below, to \$0.98, \$0.95, or in a crisis far lower. A small, brief de-peg is normal market noise that arbitrage erases in minutes. A large or persistent de-peg signals that the market doubts the reserves or the redemption mechanism. The whole danger of stablecoins lives in the question: is this a passing wobble that snaps back, or the start of a run that does not?

### The close analogy: a money-market fund and "narrow banking"

Here is the frame that makes a practitioner respect the topic. A fiat-backed stablecoin is, economically, almost exactly a **money-market fund** (MMF) - a fund that takes your cash, buys safe short-term assets like T-bills, earns the interest, and lets you redeem at a stable \$1 per share. MMFs are a multi-trillion-dollar industry, and crucially, they are *heavily regulated* precisely because they promise a stable \$1 while holding assets that could, in a crisis, be worth slightly less - the dreaded "breaking the buck." A stablecoin makes the same promise with far less regulation.

It is also close to an old reform idea called **narrow banking**: a bank that takes deposits and holds *only* the safest assets (cash and government bonds), making no risky loans, so it can always pay everyone back. A narrow bank is supposed to be run-proof. A fully T-bill-backed stablecoin is the closest thing crypto has built to a narrow bank - which is reassuring right up until you ask whether the reserves are *really* all T-bills, *really* segregated, and *really* redeemable on the worst day. With those eight ideas in hand, the story tells itself.

## Tether (USDT): the giant nobody fully audited

Tether is the original and by far the largest stablecoin. Launched in 2014 (originally as "Realcoin"), USDT had grown to roughly **\$110 billion** in tokens outstanding by 2024 and has continued to climb - the most-used dollar token in crypto, dominant especially on exchanges and in emerging markets. If you trade on most non-US crypto exchanges, you are quoting prices in USDT whether you think about it or not.

### What Tether does and how it makes money

Tether's business is breathtakingly simple. Customers give it dollars; it mints USDT; it invests the dollars in (mostly) T-bills and other assets; it keeps the interest. Token-holders get a stable dollar token and *zero* yield. Tether keeps the entire return on the reserves. When short-term US rates are around 5%, a reserve pile of \$100 billion throws off on the order of \$5 billion a year in interest, against a tiny staff. Tether has reported some of the largest profits-per-employee in corporate history. We will put exact numbers on this float income in a worked example below.

This is the money-market-fund business with one giant twist: an MMF passes the interest to its investors and charges a small fee; Tether keeps *all* the interest and pays investors nothing. The "fee" is effectively 100% of the yield. That is an extraordinarily profitable spread, and it exists because the token's value to holders is not yield - it is dollar stability plus the ability to move on crypto rails.

### The transparency controversy and the settlements

Here is the asterisk that has followed Tether for its entire life: for years it claimed every USDT was "fully backed" by dollars, while declining to produce a full, independent audit of its reserves. Investigators found the reality was messier. The two landmark outcomes, both in 2021:

- The **New York Attorney General (NYAG)** concluded after a multi-year investigation that Tether had, at times, *not* been fully backed - that during certain periods reserves fell short and that funds had been commingled with its affiliated exchange, Bitfinex, to cover a hole. Tether and Bitfinex settled, paid an **\$18.5 million** penalty, admitted no wrongdoing, agreed to stop serving New York, and agreed to publish periodic reserve breakdowns.
- The **Commodity Futures Trading Commission (CFTC)** separately found that Tether's "fully backed" claims were misleading - that for a substantial portion of the relevant period, the reserves were not held entirely in dollars as implied. Tether paid a **\$41 million** penalty to settle.

Since then, Tether publishes quarterly **attestations** - a snapshot of reserve categories signed by an accounting firm - but, importantly, *not a full audit*. An attestation says "on this date, the assets existed and totaled this much"; a full audit examines controls, valuation, and existence over time and gives an opinion. Tether has long promised a full audit and, as of this writing, has not delivered one from a top-tier firm. Its published reserve breakdowns now show the large majority in US Treasury bills and cash equivalents, with smaller slices in repo, secured loans, Bitcoin, and gold. Whether you trust those numbers is, in the end, a question of confidence - which is precisely the variable that runs are made of.

### Why Tether's size is itself the risk

Tether is so large that it has become load-bearing for crypto. A huge share of crypto trading volume is denominated in USDT; if USDT were ever to seriously de-peg, the price of nearly everything quoted against it would lurch, lending pools holding USDT would take losses, and the contagion would be system-wide. This is the textbook definition of **systemic importance** - an institution whose failure would damage the whole system - applied to a lightly regulated private company that most of its users have never scrutinized.

## Circle (USDC): the transparent challenger

If Tether is the opaque giant, **Circle** is the institution that built USDC as the answer to "what would a stablecoin look like if it tried to be trusted by regulators?" Launched in 2018 (originally via a consortium with Coinbase called Centre), **USDC** grew to a peak above \$50 billion and sits around **\$35 billion** as of 2024. Circle is US-based, has pursued licensing and regulatory engagement aggressively, and went public as a listed company.

USDC's pitch is transparency. Circle publishes **monthly attestations** of its reserves and has steadily simplified them toward the safest possible mix: cash at banks plus a dedicated government money-market fund (the Circle Reserve Fund, managed by BlackRock) holding short-dated Treasuries and overnight repo. The reserves are meant to be held in segregated accounts for the benefit of token-holders, not on Circle's own balance sheet to spend. In design terms, USDC is the closest large stablecoin to the "narrow bank / pure T-bill MMF" ideal.

But - and this is the central lesson of 2023 - *more transparent is not the same as run-proof.* The very transparency that told everyone USDC's reserves were partly cash *in banks* is what triggered its scariest moment, because one of those banks was Silicon Valley Bank. We will walk through that weekend in detail. The takeaway to hold now: the safest, most honest stablecoin in the market still de-pegged hard, because reserve *quality* is not the only risk - reserve *accessibility* on a specific bad day matters just as much.

## DAI: the crypto-collateralized middle path

USDT and USDC are both fiat-backed, and both ultimately depend on a real company holding real dollars in real banks. That dependence is exactly what crypto purists distrust: a fiat stablecoin can be frozen, censored, or seized by the authorities who control the banking system the reserves sit in. The middle branch of our tree - **crypto-collateralized** stablecoins, of which **DAI** (issued by the MakerDAO protocol) is the flagship - is an attempt to build a dollar token that needs no bank and no trusted company at all.

The idea is to back the dollar not with dollars but with *other crypto, locked in a public smart contract.* To mint DAI, you deposit volatile collateral - say Ether - into a smart contract called a vault, and the contract lets you borrow DAI against it. The catch is **over-collateralization**: because the collateral can crash, the protocol forces you to lock up *more* than \$1 of crypto for every \$1 of DAI you mint, often a collateralization ratio of 150% or higher. If your collateral's value falls and your ratio drops toward the minimum, the protocol automatically *liquidates* you - it sells your collateral to buy back and burn your DAI, protecting the peg. There is no company keeping the interest; the rules live entirely in code on Ethereum, and the reserves are publicly visible on-chain in real time. An **oracle** - a service that feeds external price data onto the blockchain - tells the contract what the collateral is worth so it knows when to liquidate.

This is genuinely clever, and it has held up through multiple crypto crashes (with wobbles). But it has its own distinct failure modes. First, it is *capital-inefficient*: locking \$150 of Ether to create \$100 of DAI is an expensive way to make a dollar, which caps how big a purely crypto-backed stablecoin can grow - DAI is far smaller than USDT or USDC. Second, in a fast crash, liquidations can fail: if the price falls faster than the protocol can auction collateral (as nearly happened to MakerDAO during the March 2020 "Black Thursday" crypto crash, when network congestion let some collateral get auctioned for almost nothing), the system can end up under-collateralized and the peg can wobble. Third - and this is the irony - to stay stable and scale up, DAI has over time taken on substantial *fiat-backed stablecoin* (USDC) as part of its own collateral. A token built to avoid trusting Circle ended up partly backed by Circle, which means a USDC de-peg can bleed straight into DAI. Purity is hard.

#### Worked example: over-collateralization and a liquidation

You want \$10,000 of DAI without selling your Ether. The protocol requires a minimum 150% collateralization ratio.

- You deposit \$15,000 of Ether into a vault and mint \$10,000 of DAI. Your ratio is \$15,000 / \$10,000 = 150% - right at the edge, so in practice you would post more, but take 150% for the arithmetic.
- Ether now falls 20%. Your collateral is worth \$15,000 x 0.80 = **\$12,000**, against \$10,000 of DAI debt - a ratio of 120%, below the 150% minimum.
- The protocol liquidates: it auctions your \$12,000 of Ether, uses the proceeds to buy back and burn \$10,000 of DAI (plus a liquidation penalty, say \$1,300), and returns the remainder. You are left with roughly \$700 of value instead of your original \$5,000 cushion - a brutal but *peg-preserving* outcome.
- Crucially, the \$10,000 of DAI in circulation is always matched by collateral the protocol can sell. As long as liquidations work faster than prices fall, every DAI stays backed by more than \$1 of assets.

The intuition: **a crypto-backed stablecoin survives volatility by demanding a fat cushion up front and seizing it the moment it thins - the peg is defended by liquidating users, not by a company's reserves.** It removes the trusted issuer at the cost of efficiency and of a new danger: a crash too fast for the liquidation machine to keep up.

## How the peg actually holds: mint/redeem arbitrage

We keep saying "arbitrage holds the peg." Let us make that mechanical and exact, because it is the beating heart of the whole system and it is genuinely elegant.

Conceptually, the peg is a rubber band anchored at \$1 by the redemption door. Whenever the market price strays, a trader can pocket the gap and, in doing so, drag the price back. There are two directions.

![Pipeline showing mint at par when above peg and redeem at par when below peg](/imgs/blogs/stablecoins-tether-circle-shadow-dollar-1.png)

**When the token trades below \$1** (say \$0.99), it is cheap relative to the dollar it can be redeemed for. An authorized trader buys tokens on the open market at \$0.99 and redeems them at the issuer for \$1.00, pocketing the difference. This buying pushes the market price up toward \$1, and the redemption shrinks the token supply. **When the token trades above \$1** (say \$1.01), it is expensive: a trader wires \$1.00 to the issuer, mints a fresh token, and sells it on the market for \$1.01, pocketing the gap. This selling pushes the price down toward \$1, and the minting expands supply. Either way, the profit motive squeezes the price back to par.

#### Worked example: arbitrage holding the one-dollar peg

Suppose USDC slips to \$0.99 on an exchange after a wave of selling - a 1% gap. You are an authorized participant who can redeem directly with Circle at par.

- You buy 10,000,000 USDC on the exchange at \$0.99 each. Cost: 10,000,000 x \$0.99 = **\$9,900,000**.
- You send those 10,000,000 USDC to Circle and redeem them. Circle burns the tokens and wires you 10,000,000 x \$1.00 = **\$10,000,000**.
- Your gross profit: \$10,000,000 - \$9,900,000 = **\$100,000** on a near-instant, near-riskless trade (minus any redemption fee and transfer cost).

Now notice the *second-order* effect, which is the part that matters. Your buying of 10 million tokens on the exchange pushed the price up - say from \$0.99 to \$0.995 - and you took 10 million tokens out of circulation by redeeming them. Other arbitrageurs see the same gap and pile in until the price is back at \$1.00 and the easy profit is gone. The peg was restored not by a central authority but by self-interested traders, each grabbing a sliver of a penny. **Intuition: the peg holds because a gap from \$1 is literally free money for anyone with redemption access - so the gap gets eaten almost as fast as it appears.**

The critical fragility hides in the phrase "redemption access." This arbitrage *only works if redemption actually works* - if the door is open, the reserves are there, and the wire arrives. Remove confidence in any of those, and the arbitrageur will not buy the cheap token, because they are no longer sure they can redeem it for \$1. At that instant the rubber band snaps, and a 1% gap can widen into a 12% gap with nobody willing to step in. That is the difference between a wobble and a run.

## The reserves are mostly Treasury bills - so the issuer is a major Treasury holder

Here is where the crypto story collides with the plumbing of the actual US government's finances. When Tether and Circle take in tens of billions of real dollars, they do not leave it as idle cash. They buy the safest interest-bearing asset on Earth: short-term US Treasury bills. As of 2024, Tether's published reserves alone reportedly included on the order of \$80-100 billion in US Treasuries and Treasury-backed repo, and the company has stated it ranks among the larger holders of US government debt - in the same conversation as mid-sized sovereign nations.

![Stack of stablecoin reserve composition with Treasury bills as the dominant layer](/imgs/blogs/stablecoins-tether-circle-shadow-dollar-4.png)

This stack is the reserve composition of a well-run fiat stablecoin, top to bottom by safety and liquidity: a thick base of T-bills, a layer of overnight repo (lending cash overnight against Treasury collateral), a cash-and-deposit buffer for instant redemptions, and a thin top slice of riskier "other" (gold, Bitcoin, secured loans) that the most conservative issuers avoid entirely. The closer a stablecoin is to "all T-bills and cash," the closer it is to that run-proof narrow bank we discussed.

Two consequences follow, and they cut in opposite directions. First, it makes the stablecoin *safer*: T-bills are about as good as reserves get, and a token genuinely backed by them is hard to break by reserve quality alone. Second, it makes the stablecoin *systemically entangled with TradFi*: these issuers are now real participants in the US Treasury and repo markets, and a forced fire-sale of tens of billions in T-bills during a stablecoin run could move those markets and ripple into the broader financial system. The link to traditional finance, which crypto set out to escape, has become a thick cable.

#### Worked example: Tether's reserve float income

Let us size the money machine. Assume Tether holds \$100 billion in reserves, of which \$90 billion is in T-bills and repo yielding roughly 5% (a realistic short-rate environment in 2023-2024), and \$10 billion in non-yielding cash and other assets.

- Interest on the yielding portion: \$90,000,000,000 x 0.05 = **\$4,500,000,000 per year** - four and a half billion dollars.
- Tether pays USDT holders **\$0** of this. The float income is the company's to keep, minus operating costs that are tiny relative to the haul.
- With a famously small headcount, that works out to profit-per-employee figures in the hundreds of millions of dollars - among the highest ever recorded for any company.

The intuition: **a fiat stablecoin is a machine that converts other people's desire for a stable dollar token into a giant, interest-free loan to the issuer, who invests it and keeps the yield.** This is exactly why every bank, fintech, and now sovereign is eyeing the business - and exactly why regulators want a say in who gets to run a private money printer this profitable.

Note the asymmetry of incentives this creates. The issuer earns more by holding *higher-yielding, slightly riskier* reserves (longer bonds, commercial paper, loans) rather than ultra-safe overnight instruments. The history of money-market funds is littered with funds that reached for an extra fraction of a percent and then "broke the buck" when those reach-for-yield assets soured. A stablecoin issuer faces the identical temptation, with far less oversight forcing it to resist.

## How stablecoins wire into DeFi and TradFi

A stablecoin's importance is not just its size; it is its *role.* Inside crypto, stablecoins are the connective tissue of **DeFi** (decentralized finance - financial services like lending, trading, and derivatives run by smart contracts instead of firms). Three uses dominate, and each one multiplies the damage a de-peg would do.

First, stablecoins are the dominant **unit of account and settlement asset** on exchanges. Most trading pairs are quoted against USDT or USDC rather than against actual dollars, because moving a stablecoin between accounts is instant and permissionless while moving real dollars through banks is slow and gatekept. If the quote currency itself wobbled, every price on the exchange would move at once - not because the underlying assets changed, but because the ruler did.

Second, stablecoins are the workhorse collateral and liquidity of **lending protocols and liquidity pools.** In an automated market maker (**AMM**) - a smart contract that lets people swap one token for another against a shared pool of reserves - a huge fraction of pools pair a volatile token against a stablecoin, and a deep one against USDC. Lending protocols let users borrow against crypto and receive stablecoins, or deposit stablecoins to earn yield. The protocols' accounting assumes the stablecoin is worth exactly \$1; if it de-pegs, loans that looked over-collateralized are suddenly under-collateralized, liquidations fire, and the stress propagates through every protocol holding the affected token. The total value locked (**TVL**) in these systems - tens of billions of dollars - rests on the stable-at-a-dollar assumption.

Third, in the other direction, stablecoins are crypto's **payment and remittance rail**, increasingly used outside trading entirely: a worker sending money home, a business settling an invoice, a person in a high-inflation country preserving savings. Here the stablecoin competes directly with banks and money-transfer firms, settling in seconds at near-zero cost, any day of the week.

Each of these uses also explains how stablecoins reach *into* traditional finance, not just away from it. The reserves are real Treasuries held at real custodians and bought through real dealers; the redemptions move real dollars through real banks; and the issuers are now large enough that the Treasury and repo markets feel their flows. A stablecoin is therefore a two-way bridge: crypto risk can flow into the Treasury market through forced reserve sales during a run, and traditional-banking risk can flow into crypto through the reserves, exactly as it did when SVB's failure broke USDC. The wall between crypto and the regulated financial system - the wall crypto was supposed to build - has become a busy doorway, and stablecoins are the door.

## The de-peg episodes: when the peg actually broke

Theory is comforting. Now look at what has actually happened, because the de-pegs are where you learn what is real.

### USDC and the Silicon Valley Bank weekend, March 2023

This is the cleanest case study of how even a "good" stablecoin can break - and recover. In early March 2023, Silicon Valley Bank failed in a textbook bank run (the full story is its own saga; see the cross-link below). Circle had parked part of USDC's *cash* reserves - about **\$3.3 billion** - in deposits at SVB. When SVB was seized on Friday, March 10, that cash was suddenly of uncertain accessibility: under FDIC rules only \$250,000 per account is insured, and the fate of the rest was unknown over the weekend.

The market did the math in real time. Circle disclosed the \$3.3 billion exposure on Friday night. With the question "is part of USDC's backing trapped in a failed bank?" hanging open, holders rushed to sell USDC and to redeem it where they could. Redemptions via Circle effectively paused over the weekend because the banking system was closed. With the redemption door jammed shut and \$3.3 billion of backing in limbo, arbitrage could not function - and USDC's price on exchanges fell to around **\$0.88** at the trough on Saturday, March 11. A 12% de-peg on the most transparent, most "narrow-bank-like" stablecoin in the world.

![Before-after of the peg holding at par versus de-pegging below par](/imgs/blogs/stablecoins-tether-circle-shadow-dollar-3.png)

Then it recovered, and the recovery is as instructive as the break. Over the weekend, US authorities announced that *all* SVB depositors - insured and uninsured alike - would be made whole. The instant that backstop was credible, the \$3.3 billion was no longer in doubt, redemption was about to reopen at par, and the arbitrage logic snapped back to life: USDC at \$0.88 was now obviously worth buying, because it would soon redeem at \$1.00. By Monday, March 13, USDC had climbed back to roughly \$1.00. The de-peg lasted about 48 hours.

#### Worked example: buying USDC below par during the SVB weekend

Put yourself in the trade, with the benefit (and risk) of believing the backing was money-good.

- On Saturday you buy 1,000,000 USDC at \$0.88. Cost: 1,000,000 x \$0.88 = **\$880,000**.
- You are betting that the \$3.3 billion is recoverable and the peg will return. This is *not* riskless: had the US let SVB's uninsured depositors take a haircut, USDC's backing would have had a real hole, and \$0.88 might have been generous.
- The backstop is announced; the peg returns to \$1.00. You redeem or sell your 1,000,000 USDC for **\$1,000,000**.
- Profit: \$1,000,000 - \$880,000 = **\$120,000**, a 13.6% return over a weekend - the reward for taking the confidence risk when others were fleeing.

The intuition: **a de-peg is the market pricing the probability that redemption fails; if you have better information than the panic, the gap is an opportunity, and if you do not, it is a trap.** The same \$0.88 print that minted profits for those who correctly judged the backstop would have meant a permanent loss had the judgment been wrong.

The deeper lesson: USDC did everything "right" - safe reserves, full transparency, monthly attestations - and still de-pegged, because **transparency revealed a real exposure** and the redemption door slammed shut at the worst moment. The fix was not crypto-native at all. It was a US government bailout of a traditional bank. The most decentralized-sounding asset in the room was saved by the most centralized authority there is.

### The algorithmic collapse: UST and Terra, May 2022

The SVB episode was a 48-hour scare with a happy ending. The Terra/UST collapse was the opposite: a near-total, permanent wipeout, and the defining example of *why algorithmic stablecoins are fragile by design.*

UST (TerraUSD) was an algorithmic stablecoin with no real reserves. Its peg relied on a sister token, LUNA, and a mint/burn loop: you could always burn \$1 of UST to mint \$1 worth of LUNA, and vice versa, regardless of UST's market price. In theory, if UST fell to \$0.98, you could buy it cheap and burn it for \$1 of LUNA, profiting and supporting the price. To pull deposits in, the ecosystem offered a roughly **20% APY** on UST through a protocol called Anchor - a yield with no sustainable source behind it, paid largely from subsidies. At its peak, UST had roughly \$18 billion outstanding and the Terra ecosystem (UST + LUNA) was worth on the order of \$40 billion.

In May 2022, large UST sales knocked it slightly below \$1. The arbitrage that was supposed to save it required minting fresh LUNA - which *increased LUNA supply and drove LUNA's price down*. As LUNA fell, the market realized that the thing supposedly backing UST was evaporating, so more holders dumped UST, which forced more LUNA minting, which crashed LUNA further. This is a **reflexive death spiral**: the stabilizing mechanism became the destruction mechanism. We will dissect the run dynamic in a moment; for the full forensic account, see the dedicated case study linked below.

#### Worked example: a fiat-backed vs algorithmic stability comparison

Compare what happens to each design when \$1 billion of tokens are dumped on the market in a panic.

- **Fiat-backed (USDC):** Holders sell \$1 billion of USDC, pushing the price to, say, \$0.97. Arbitrageurs buy the cheap tokens and redeem them with Circle for \$1.00 each, pulling \$1 billion of cash out of *real reserves that exist*. Supply shrinks, the selling is absorbed by hard assets, and the price returns to \$1.00. The reserves act as a shock absorber: \$1 billion of fear meets \$1 billion of T-bills, and the T-bills win.
- **Algorithmic (UST):** Holders sell \$1 billion of UST, pushing it to \$0.97. The "arbitrage" requires burning UST to mint \$1 of LUNA each - but there are *no reserves*, only freshly printed LUNA. Minting \$1 billion of new LUNA into a falling market crushes LUNA's price, which destroys the perceived backing, which triggers more UST selling. The shock absorber is made of the same material as the shock. The price does not return to \$1; it heads toward \$0.

The intuition: **fiat backing absorbs a panic with assets that exist; algorithmic "backing" tries to absorb a panic by printing a token whose value depends on the panic not happening.** One design dampens runs; the other amplifies them. That is the entire difference, and it is why UST went to zero while USDC came back.

In the days that followed May 2022, UST fell to a few cents and never recovered; LUNA went from over \$80 to a fraction of a cent; roughly \$40 billion of combined value evaporated; and the contagion helped topple crypto lenders and a major hedge fund in the months after. It is the largest stablecoin failure in history and the clearest proof that "stable" is a property of the design, not of the name.

## How a run on a stablecoin works

Strip away the specifics and the failure mode of *any* stablecoin - fiat or algorithmic - is the same shape as a bank run, because a stablecoin is a promise to pay \$1 on demand backed by assets that cannot all be turned into cash at once at full value.

![Graph of a stablecoin run from doubt to selling and redemption to drained reserves](/imgs/blogs/stablecoins-tether-circle-shadow-dollar-6.png)

The dynamic, as a forward chain of cause and effect: doubt about the reserves appears (a bank exposure, a settlement, a rumor). That doubt splits into two simultaneous pressures - holders dump tokens on exchanges (driving the *market* price below \$1) and authorized participants rush to redeem directly with the issuer (draining *reserves*). The falling exchange price feeds more redemptions ("get out at \$1 while the door is still open"). To meet redemptions, the issuer must convert reserves to cash - selling T-bills, which is fast and clean for the first few billion but slower and more price-moving as the wave grows. If outflows exceed the issuer's *liquid* reserves, the door jams: redemptions slow or pause, the peg breaks hard, and the only question left is whether the *total* reserves (eventually liquidated) still cover everyone, or whether there is a genuine hole.

The decisive variable is **liquidity, not solvency.** An issuer can be perfectly solvent - reserves worth 100 cents on the dollar - and still fail to meet a run if those reserves are not all *liquid right now.* This is exactly what happened to USDC: its reserves were money-good, but \$3.3 billion of cash was frozen in a closed bank over a weekend, and that timing gap alone produced a 12% de-peg.

#### Worked example: a run where redemptions exceed liquid reserves

Take a hypothetical fiat stablecoin, "PegCoin," with \$50 billion outstanding and the following reserves: \$5 billion cash, \$10 billion in overnight repo and T-bills maturing this week (call this "instantly liquid"), and \$35 billion in T-bills maturing over the next several months (liquid, but only by *selling* into the market at whatever price clears).

- A scandal breaks on a Friday. Over the weekend, holders request redemption of \$20 billion - 40% of all tokens.
- The issuer pays the first \$15 billion from cash plus instantly-liquid reserves. Fine so far; the door is open and the peg holds at \$1.00.
- The next \$5 billion requires *selling* term T-bills before maturity. In a calm market that is trivial. But it is a weekend, settlement is slow, and a fire-sale of \$5 billion at once nicks the price - say the issuer realizes \$0.995 on the dollar. A small realized loss appears, and the sales take days to settle.
- Because the cash cannot be wired until the T-bill sales settle, redemptions *pause*. Retail holders who cannot redeem directly dump PegCoin on exchanges, and with the door visibly jammed, the price craters to **\$0.90** even though the reserves, fully liquidated, would cover every token.

The intuition: **a stablecoin does not need to be insolvent to break; it only needs its redemptions to outrun its instantly-available cash, because a frozen door turns a liquidity gap into a confidence collapse.** This is why the *composition* of reserves (how much is cash vs. term bonds) matters as much as the *total*, and why "we are fully backed" is a true statement that can coexist with a brutal de-peg.

## How stablecoins are regulated (and why it is finally happening)

For most of their history, stablecoins occupied a legal grey zone: they looked like deposits, acted like money-market funds, but were issued by entities that were neither banks nor regulated funds. That is changing fast, because regulators recognize the shape of the risk - private money at systemic scale - and it is the oldest risk in finance.

### The US direction: the GENIUS Act framework

The United States has moved toward a dedicated federal framework for payment stablecoins, advancing under the banner of the **GENIUS Act** (and related proposals). The thrust of this direction is to treat large stablecoin issuers more like regulated money-issuers: requiring that tokens be backed **one-to-one by high-quality liquid assets** (cash and short-term Treasuries), that reserves be **segregated** and not lent out or rehypothecated, that issuers publish regular disclosures and submit to **audits**, that there be a clear federal or state licensing regime, and that token-holders have priority claims on reserves if the issuer fails. In spirit, it is an attempt to force stablecoins to actually *be* the narrow bank they pretend to be - all T-bills and cash, fully transparent, run-resistant - rather than relying on the issuer's promise.

### Europe's MiCA

The European Union moved first with **MiCA** (Markets in Crypto-Assets), which became applicable to stablecoins in 2024. MiCA imposes licensing, reserve, and redemption requirements on "e-money tokens" and "asset-referenced tokens," caps the scale of large non-euro stablecoins used heavily for payments in the EU, and mandates that issuers hold safe, segregated reserves and honor redemption at par. The practical effect has already been visible: some exchanges restricted USDT for EU users pending compliance, illustrating that regulation can reshape which stablecoins dominate in which regions.

### The through-line

Notice what every framework converges on: *high-quality liquid reserves, segregation, transparency, and redemption rights.* These are not arbitrary - they are precisely the features that make the difference between a stablecoin that survives a run and one that does not, the same features that bank regulation and money-market-fund reform landed on after their own crises. Regulation here is not crypto-skeptics meddling; it is the financial system's immune response recognizing a familiar pathogen. The risk it targets is real, and named: a lightly supervised entity issuing hundreds of billions in private dollars, systemically entangled with both crypto and the Treasury market, whose failure mode is a classic run.

## The shadow dollar system

Step back and the full picture is striking. Tether and Circle - two private companies, neither a chartered bank - have issued well over **\$140 billion** in dollar-denominated claims that circulate globally as money, hold over \$120 billion in reserves dominated by US Treasuries, earn billions in float, and serve hundreds of millions of users, many in countries where the local currency is unstable and access to actual dollars is restricted. They are, functionally, a **shadow dollar-banking system**: institutions performing the core economic function of banks (issuing dollar-denominated money and intermediating it into safe assets) while sitting largely outside the regulatory perimeter built for banks.

The phrase "shadow banking" was coined before 2008 to describe money-market funds, structured-investment vehicles, and other entities that did bank-like things without bank-like supervision - and that turned out to be the fault lines along which that crisis spread. Stablecoins are the newest member of that family. They bring real benefits: cheap, fast, borderless dollar payments; a lifeline for people in high-inflation economies; the settlement rails for an entire new financial system. And they carry the matching risk: systemic importance without systemic safeguards, a peg that is only as strong as confidence in reserves nobody fully audits, and a failure mode - the run - that finance has spent a century learning to fear and has not yet fully tamed even in the regulated banking it understands best.

## Common misconceptions

**"Stablecoins are backed by Bitcoin."** No - this conflates the families. The big fiat stablecoins (USDT, USDC) are backed by dollars, T-bills, and cash, not crypto. Some over-collateralized stablecoins like DAI hold crypto as collateral, but they hold *more than \$1* of crypto per \$1 issued precisely because crypto is too volatile to be a 1:1 backing. And algorithmic stablecoins like UST were backed by essentially nothing solid. "Backed by Bitcoin" describes almost none of the major stablecoins.

**"A stablecoin is just a digital dollar / it's the same as money in a bank."** It is a *private claim* on a *private company's* reserves, not a government liability and not an insured bank deposit. There is no FDIC insurance on a stablecoin. If the issuer's reserves are short, frozen, or fraudulent, holders can lose money - as UST holders did completely. A bank deposit up to \$250,000 is guaranteed by the US government; a stablecoin is guaranteed only by the issuer's solvency and honesty.

**"If it's pegged to \$1, it can't lose value."** The peg is an aspiration enforced by arbitrage and redemption, not a guarantee. USDC traded at \$0.88; UST went to near zero. A peg is exactly as strong as the market's confidence that redemption works, and that confidence can evaporate in hours. "Pegged" is a design goal, not a property of the asset.

**"Tether is obviously a fraud / Tether is obviously fine."** Both confident takes overreach. Tether has paid real penalties for past misleading claims about backing and has *not* produced a full top-tier audit - genuine reasons for caution. But it has also published quarterly attestations showing large, mostly-Treasury reserves, redeemed billions on demand through multiple stress events, and held its peg when UST did not. The honest position is uncertainty about a systemically important entity, which is itself the problem.

**"Algorithmic stablecoins just need better code."** The failure of UST was not a bug; it was the *design.* Any stablecoin "backed" by a token whose value depends on the stablecoin staying pegged contains a reflexive loop that runs in reverse under stress. Better incentives can delay the spiral, but the absence of hard, external reserves means there is nothing to absorb a true panic. This is a structural property, not an implementation detail.

**"USDC de-pegging proves transparency is pointless."** The opposite. Transparency is *why* the market could price the risk so fast - everyone knew about the \$3.3 billion at SVB because Circle disclosed it. The de-peg was caused by a real exposure plus a frozen redemption door over a weekend, not by transparency. And transparency is what let confident buyers step in at \$0.88, because they could verify the backing was likely money-good. Opaque reserves would have made the panic worse, not better.

## How it shows up in real markets

**USDC's SVB-weekend de-peg (March 2023).** The canonical example of a high-quality stablecoin breaking on a *liquidity/access* problem, not a *quality* one. \$3.3 billion of cash stuck in a failed bank over a closed weekend drove USDC to ~\$0.88, and a government backstop of a traditional bank - not any crypto mechanism - restored the peg within ~48 hours. The lesson: reserve composition and the timing of redemption access matter as much as total backing.

**Tether's CFTC and NYAG settlements (2021).** Two regulators independently found Tether's "fully backed" claims had been misleading during certain periods, with reserve shortfalls and commingling with affiliate Bitfinex. Combined penalties of roughly \$60 million and an agreement to publish reserve breakdowns. The episode is why "Tether is fully backed" remains a statement to verify rather than assume, and why a full audit is the recurring demand.

**The UST/Terra collapse (May 2022).** The largest stablecoin failure ever: ~\$18 billion of UST and a ~\$40 billion ecosystem reduced to near-zero in days when the algorithmic peg's reflexive loop ran in reverse. It triggered a cascade that helped sink crypto lenders and a major hedge fund, and it permanently discredited reserve-free algorithmic designs. The full forensic story is the linked case study.

**Stablecoins as the dollar's reach into emerging markets.** In countries with high inflation or capital controls - Argentina, Turkey, Nigeria, Lebanon and others - USDT in particular has become a grassroots dollar substitute: a way to hold and move value pegged to the world's reserve currency without a US bank account. This is one of the clearest real-world utilities of stablecoins and a major driver of their growth, and it extends US dollar dominance into places traditional dollar banking never reached - a geopolitical fact regulators on both sides have begun to weigh. To see why the dollar's creation and global plumbing matter here, see [how money is created](/blog/trading/finance/how-money-is-created-banks-central-banks-money-multiplier).

**The depeg-and-recover pattern.** Beyond the headline cases, fiat stablecoins routinely make brief excursions to \$0.998 or \$1.002 and snap back within minutes as arbitrage works - the *normal* heartbeat of a healthy peg. The skill is distinguishing this benign noise from the start of a real run: noise is small, symmetric, and self-correcting; a run is one-directional, accompanied by reserve doubt or a frozen door, and feeds on itself. The same de-peg number can be a buying opportunity or a warning, depending entirely on whether redemption still works. For the broader family of institutions whose failures take this run shape, see the [field guide to financial institutions](/blog/trading/finance/field-guide-to-financial-institutions).

The four families and their stress records, side by side:

![Matrix comparing USDT, USDC, DAI, and an algorithmic stablecoin](/imgs/blogs/stablecoins-tether-circle-shadow-dollar-5.png)

And the era at a glance - the milestones and the de-pegs that punctuate it:

![Timeline of stablecoin milestones and de-pegs from 2014 to 2025](/imgs/blogs/stablecoins-tether-circle-shadow-dollar-7.png)

## When this matters to you / further reading

If you ever touch crypto, you are touching stablecoins, even if you never buy one knowingly - because the prices you see are quoted in them and the exchanges you use settle in them. The practical takeaways are simple and worth internalizing.

First, **a stablecoin is a credit instrument, not cash.** Holding USDT or USDC is lending the issuer money interest-free and trusting their reserves. Treat it with the scrutiny you would give any IOU: who is the issuer, what backs it, can you redeem, and what happens on the worst day. Do not assume "\$1" is a fact; it is a promise.

Second, **reserve composition and redemption access are the whole game.** The safest stablecoin is the one closest to all-cash-and-T-bills, fully transparent, with a redemption door that genuinely stays open under stress. Total backing is necessary but not sufficient; liquidity and access on a bad weekend are what actually save you.

Third, **algorithmic stablecoins are a fundamentally different and more dangerous animal.** A "stable" coin yielding 20% with no hard reserves is not a clever innovation; it is a structure that pays you to ignore a reflexive loop that can take it to zero. The yield is the warning, not the reward.

Fourth, **the systemic risk is real and largely unsupervised - for now.** These are bank-like institutions of systemic scale operating mostly outside bank regulation, which is exactly why the US and EU are racing to bring them inside it. Watch the regulation: a framework that forces true narrow-bank backing would make stablecoins much safer, while a failure to regulate leaves a multi-hundred-billion-dollar run waiting to happen.

To go deeper on the mechanisms behind this story, three companion pieces are the natural next steps. The Terra collapse - the definitive algorithmic-stablecoin failure - is dissected in full in [the Terra/Luna 2022 collapse](/blog/trading/crypto/terra-luna-2022-collapse). The bank run that nearly broke USDC is told from the banking side in [SVB and Credit Suisse 2023](/blog/trading/finance/svb-credit-suisse-2023-bank-runs), which is the best companion for understanding why reserve *access* can fail even when reserve *quality* is fine. And to place stablecoin issuers in the broader landscape of entities that issue, hold, and intermediate money, the [field guide to financial institutions](/blog/trading/finance/field-guide-to-financial-institutions) is the map. Read together, they make the same point from four angles: the oldest risk in finance - a promise to pay on demand that cannot be honored all at once - has simply put on new clothes, and the clothes are made of code.
