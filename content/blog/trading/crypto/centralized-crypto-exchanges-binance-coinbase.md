---
title: "Centralized Crypto Exchanges: How Binance and Coinbase Became Broker, Exchange, and Bank in One"
date: "2026-06-15"
publishDate: "2026-06-15"
description: "A plain-English tour of how centralized crypto exchanges work, why bundling broker, exchange, and custodian into one company creates deep conflicts of interest, and what FTX's collapse taught everyone who skipped that lesson."
tags: ["crypto", "centralized-exchange", "binance", "coinbase", "custody", "proof-of-reserves", "kyc", "ftx", "cex", "self-custody"]
category: "trading"
subcategory: "Crypto"
author: "Hiep Tran"
featured: true
readTime: 41
---

> [!important]
> **TL;DR** — A centralized crypto exchange like Binance or Coinbase is the single place most people touch crypto, and it earns its convenience by doing three jobs that traditional finance keeps deliberately separate: it is your broker, the exchange, and the bank that holds your coins, all at once.
>
> - In traditional finance a *broker* takes your order, an *exchange* matches it, and a separate *custodian* holds the asset — three different firms, on purpose, so no single one can quietly spend what is yours. A crypto exchange merges all three into one balance sheet.
> - The moment you deposit, your coins leave your control. They land in the exchange's pooled wallets and you are left holding a database entry — an IOU — not the keys. "Not your keys, not your coins" is the whole warning in five words.
> - Exchanges make money many ways at once: a trading fee of roughly 0.1% to 0.5% per trade, the bid-ask spread, listing fees worth millions, a cut of staking and lending yield, and in Binance's case a house token, BNB, worth tens of billions.
> - The bundling is convenient and it is dangerous. FTX showed what happens when the firm holding your coins is also free to spend them; Binance's roughly \$4.3 billion 2023 settlement with US authorities showed what happens when the biggest exchange runs for years without real anti-money-laundering controls.
> - The one habit to remember: an exchange is a place to *trade*, not a place to *store*. Proof of reserves helps but does not prove solvency, and only coins in a wallet whose keys you alone hold are truly yours. All dollar and market figures here are approximate and as of early 2026.

Almost everyone who has ever bought a single dollar of Bitcoin did it the same way: they downloaded an app, typed in their card number or linked their bank, and tapped "buy." That app was almost certainly a *centralized exchange* — Coinbase, Binance, Kraken, or one of a handful of others. It felt as ordinary as buying a stock on a brokerage app or a song on iTunes. And that ordinariness is exactly the thing worth slowing down to examine, because under the friendly interface sits an arrangement that traditional finance spent a century learning to *avoid*: one company acting as your broker, the marketplace, and the bank holding your money, with very little forcing those roles apart.

The diagram above is the mental model for the whole post: four big exchanges, each making a different set of trade-offs between scale, regulation, and how much of crypto's machinery they bundle into one app. Hold that picture; everything else is detail poured into it.

![Matrix comparing four centralized crypto exchanges](/imgs/blogs/centralized-crypto-exchanges-binance-coinbase-1.png)

We will build this from absolute zero. You do not need to know what a blockchain is, what a private key looks like, or what "custody" means — by the end you will, and you will be able to read a headline about an exchange settlement or a "proof-of-reserves" report and know exactly what it does and does not promise. None of this is investment advice. It is a map of how a particular corner of the financial world is wired, so the news stops being noise.

## The basics: blockchains, keys, and what "custody" really means

Before we can say anything sharp about exchanges, we need a small pile of vocabulary. We will define each term the first time it appears and never assume you already have it.

### A blockchain, a coin, and a wallet

A *blockchain* is a shared public ledger — a giant spreadsheet of who owns what — that is copied across thousands of computers around the world, with no single company in charge of it. Bitcoin's blockchain records every Bitcoin balance; Ethereum's records balances of *ether* (its native coin) and thousands of other *tokens*. A *token* is just an entry on a blockchain that represents some unit of value — it can be a coin like Bitcoin, a *stablecoin* (a token designed to stay worth about \$1, like USDT or USDC), or a share in some project. When people say something is *on-chain*, they mean it is recorded on one of these public ledgers, visible to anyone.

Ownership on a blockchain is controlled by a *private key* — a long secret number, in practice a string of words or characters, that mathematically proves you control a particular balance. Whoever holds the private key can move the coins; whoever does not, cannot. There is no customer-service line, no password reset, no bank to call. The key *is* the ownership. A *wallet* is just software that stores and uses your keys to send and receive coins. This is the single most important idea in all of crypto, and the entire rest of this post hangs off it: **on a blockchain, control of the key is control of the coins, full stop.**

### Custody, and "not your keys, not your coins"

*Custody* means who holds the keys — who has the practical power to move the coins. There are two modes, and the gap between them is the heart of this article.

In *self-custody*, you hold your own private keys, in your own wallet, on your own device. You alone can move your coins. You also alone are responsible: lose the key, and the coins are gone forever, with no recourse. There were famous cases early on of people who threw away hard drives holding the keys to thousands of Bitcoin and could never get them back.

In *custodial* arrangements, someone else holds the keys for you. When you keep coins on Coinbase or Binance, the exchange holds the keys; you hold only a *claim* — a number in their database saying you are owed that much. The crypto community compressed this into a slogan you will see everywhere: **"not your keys, not your coins."** It means that if you do not control the private key, you do not really own the coin — you own a promise from whoever does.

![Pipeline of depositing and trading on a centralized exchange](/imgs/blogs/centralized-crypto-exchanges-binance-coinbase-2.png)

### CEX vs DEX

A *centralized exchange* — a **CEX**, which is what this post is about — is a company that runs a crypto marketplace and holds your coins for you while you trade. Binance, Coinbase, Kraken, and OKX are CEXs. You create an account, deposit money, and trade against an order book the company operates. The company is in custody of everything.

A *decentralized exchange* — a **DEX** — is the opposite: it is not a company but a *smart contract*, a small program that lives on a blockchain and executes automatically. (Uniswap and Aave are the famous ones; we cover them in [a separate deep dive on DeFi protocols](/blog/trading/crypto/defi-protocols-uniswap-aave-makerdao).) On a DEX you connect your own wallet, keep your own keys, and trade directly from it — the code holds nothing of yours; it just swaps one token for another and hands the result straight back to your wallet. A DEX cannot run off with your coins because it never holds them. It also cannot reset your password, reverse a mistaken trade, or stop a hacker who got your key. The two are genuine opposites: a CEX trades convenience and support for custody; a DEX trades custody for self-reliance.

![Tree splitting trading venues into CEX and DEX](/imgs/blogs/centralized-crypto-exchanges-binance-coinbase-7.png)

The figure above is the taxonomy we will keep returning to. Every venue you can trade on is, at root, either *a company that holds your keys* or *code that holds nothing*. That single fork decides who you are trusting.

### What an order book is

An *order book* is the engine inside an exchange that matches buyers and sellers. It is a live list of *bids* (the prices at which people are willing to buy) and *asks* (the prices at which people are willing to sell), sorted from best to worst. The highest bid and the lowest ask sit at the top, facing each other. When a buyer's bid meets a seller's ask at the same price, a trade happens and both orders are removed. The order book is exactly the same machinery a stock exchange uses; crypto did not invent it. What is different is that on a CEX the *same company* that runs the order book also holds everyone's coins and can see every order — a concentration of knowledge and power no stock exchange has.

The gap between the best bid and the best ask is the *bid-ask spread* — say someone will buy Bitcoin at \$99,990 and someone else will sell at \$100,010, a \$20 spread. If you want to buy *right now* you pay the ask; if you want to sell *right now* you take the bid; so you lose roughly the spread every time you cross it. We will compute exactly what that costs you later.

Two order types matter for the rest of this post, because they decide which fee you pay. A *market order* says "fill me right now at the best available price"; it crosses the spread immediately and is charged the higher *taker* fee, because it *takes* liquidity off the book. A *limit order* says "only fill me at this price or better"; it usually rests on the book waiting for someone to trade against it, and is charged the lower *maker* fee, because it *makes* liquidity by adding a resting order others can hit. Beginners almost always use market orders — the one-tap "buy" button — so they pay both the wider taker fee and the full spread, which is why a casual buy costs more than the headline rate suggests. The exchange, sitting in the middle, collects on both: the spread from the order book and the fee from the trade.

One more wrinkle the order book hides: *slippage*. On a deep, liquid market like Bitcoin, a \$10,000 order barely moves the price. But on a thin market, a single large order can eat through several layers of the book — the first \$2,000 fills at \$1.00, the next \$3,000 at \$1.02, the rest at \$1.05 — so the *average* price you pay is worse than the price you saw when you clicked. Slippage is not a fee the exchange charges; it is the cost of demanding more liquidity than is sitting at the best price. It compounds the spread, and like the spread it is worst exactly where liquidity is thinnest.

### Hot wallets and cold wallets

An exchange holds enormous amounts of crypto for its customers, and it splits that hoard into two kinds of storage. A *hot wallet* is a wallet whose keys are kept on internet-connected servers, ready to process withdrawals instantly. It is convenient and it is exposed — if hackers breach the servers, they can drain a hot wallet. A *cold wallet* is storage whose keys are kept *offline* — on hardware never connected to the internet, sometimes literally in a vault. Cold storage is far safer but slow to access. A well-run exchange keeps the large majority of customer coins (often 90–98%) in cold storage and only a small float in hot wallets for daily withdrawals — the same way a bank keeps most cash in the vault and only a little in the teller's drawer.

### Commingling of funds

Here is the term that turns up at the scene of every exchange disaster. *Commingling* means mixing different people's money together in one pool instead of keeping each person's funds segregated. When you deposit Bitcoin to an exchange, it does not put *your specific coins* in a labeled box. It pools your coins with everyone else's in its hot and cold wallets and simply records, in its own database, that *you* are owed that amount. This is efficient — but it means there is no on-chain link between "your" coins and you. You own a *fraction of a pool*, recorded only on the company's internal books. If those books are honest and the pool is full, fine. If the company quietly lends the pool out, or spends it, or it is stolen, your database entry still says you own coins that may no longer be there.

![Graph of where deposited coins sit inside a centralized exchange](/imgs/blogs/centralized-crypto-exchanges-binance-coinbase-4.png)

The figure traces the path. Your deposit flows into the exchange's hot and cold wallets, joins a commingled pool, and what you are left holding is the lavender box on the right: an IOU on a number. If the pool is ever lent out or rehypothecated (the red box) — that is, pledged or re-used as if it were the exchange's own — your claim does not change, but the coins behind it might vanish.

### Proof of reserves

After enough disasters, exchanges began publishing *proof of reserves* — a cryptographic attestation meant to show they actually hold the coins they owe. The honest version has two halves. First, the exchange proves its *assets*: it shows it controls wallets holding, say, 600,000 Bitcoin, by signing a message from those wallets. Second, it proves its *liabilities*: it builds a structure (a *Merkle tree*, a way of hashing every customer balance into one summary number) so each customer can check their own balance is included, and the total of all balances is published. Reserves cover liabilities only if assets ≥ liabilities. The catch — which we will hammer later — is that proof of reserves shows what the exchange *has* and what it *owes to depositors* at one instant, but says nothing about money it owes to *other* lenders, nothing about whether the assets are borrowed for the snapshot, and nothing about tomorrow. It is a photograph, not a guarantee.

### KYC and AML

*KYC* stands for *Know Your Customer*: the legal requirement that a financial firm verify who its customers really are — collect your ID, your address, sometimes a selfie — before letting you trade in size. *AML* stands for *Anti-Money-Laundering*: the broader set of rules requiring firms to monitor for and report suspicious activity, like funds linked to sanctioned countries or criminal proceeds. Banks have done this for decades. Crypto exchanges, which move value across borders instantly and pseudonymously, are exactly the kind of business regulators worry about — and, as we will see, Binance's failure to run real AML controls is what produced its enormous settlement.

There is a tension baked into KYC that is worth naming, because it explains a lot of crypto's culture. The original promise of Bitcoin was *permissionless* money — value you could hold and send without asking anyone's permission or proving who you are. A centralized exchange, by demanding your ID before you can trade, reintroduces exactly the gatekeeper crypto was built to remove. That is the price of being a regulated on-ramp: to plug crypto into the banking system — to let you fund an account from your bank and cash out to it — an exchange must follow the same rules a bank follows, which means knowing who you are and policing what you do. The pseudonymous, permissionless world still exists on-chain and on DEXs; the moment you step onto a major CEX, you have stepped back into the identified, permissioned one. The convenience of the on-ramp and the surrender of pseudonymity are, again, the same fact.

### Fiat rails: how dollars actually get in and out

A detail people skip: an exchange does not magically hold dollars. To let you fund an account with a bank transfer or a card, the exchange needs *banking partners* — real banks willing to hold its customers' cash and process the transfers. This is harder than it sounds, because many banks have been wary of crypto firms, and an exchange that loses its banking partners can suddenly be unable to take deposits or pay out withdrawals in dollars even if it is perfectly solvent in crypto. This is also why exchanges lean so heavily on *stablecoins* — tokens like USDT and USDC engineered to stay worth about \$1 — as a stand-in for dollars that lives entirely on-chain and never needs a bank to move. When you "deposit \$10,000" to many exchanges, what you often end up holding is \$10,000 of a stablecoin, an IOU on a dollar issued by a third party, sitting inside an IOU on a balance issued by the exchange. The dollar you think you hold is two promises deep.

### The TradFi contrast: why three firms, not one

Now the contrast that gives this whole post its spine. In traditional finance — *TradFi*, as crypto people call the old world — the three jobs an exchange bundles are kept in *separate hands on purpose*.

When you buy a share of Apple through a brokerage app, three different entities act. Your *broker* (the app) takes your order and routes it. An *exchange* (Nasdaq, say) matches your order against a seller. And a *custodian* — in the US, ultimately a utility called the *Depository Trust & Clearing Corporation*, the DTCC, plus the broker's own segregated accounts — actually holds the shares, legally separated from the broker's own money. On top of that sits *SIPC* insurance, which protects customer assets up to a limit if a broker fails. The whole architecture exists so that *no single firm can quietly spend what belongs to customers*, and so that if your broker goes bankrupt, your shares are still yours because they were never mixed with the broker's money. (If you want the full tour of how those institutions divide the work, see [the field guide to financial institutions](/blog/trading/finance/field-guide-to-financial-institutions) and [inside an investment bank](/blog/trading/finance/inside-an-investment-bank-how-they-make-money).)

![Before-after of separated TradFi roles versus bundled crypto roles](/imgs/blogs/centralized-crypto-exchanges-binance-coinbase-3.png)

The figure says it in one glance. On the left, TradFi: broker, exchange, custodian — three boxes, three firms, deliberately apart. On the right, a crypto CEX: the *same firm* is your broker, *the same firm* runs the order book, *the same firm* holds your coins. The separation that protects you in the old world simply is not there. That is not an accident of crypto being young; it is the design. And it is the source of every conflict of interest we are about to walk through.

## What a centralized exchange actually is, and why it exists

A centralized exchange is, mechanically, three businesses fused together. It is a *broker* (you place orders through it), an *exchange* or *trading venue* (it runs the order book that matches those orders), and a *custodian or bank* (it holds your coins and your cash). Bolted onto those are usually a *payment processor* (it takes your card or bank transfer to fund the account), a *lender* (it lends out deposits to earn yield), and sometimes an *issuer* (it mints its own token). One app, six hats.

Why did crypto end up this way when TradFi went the other direction? Because at the start, the alternative was almost unusable. In Bitcoin's early years, to own crypto you had to run wallet software, safeguard a private key, and send transactions by hand — and one typo or one lost key meant permanent loss. The vast majority of people could not and would not do that. An exchange that let you sign up with an email, buy with a credit card, and never touch a private key removed every hard part at once. The bundling *is* the product. People did not flock to Coinbase despite the fact that it holds their keys; they flocked to it *because* it does, so they do not have to.

So a CEX exists to be the on-ramp and the comfortable middle. It converts ordinary money (*fiat*, meaning government-issued currency like dollars) into crypto and back; it gives you a familiar order-book interface; it remembers your balance; it resets your password; it has a support line. For that convenience you hand it custody — and custody is where all the danger lives.

It helps to see the exchange as a kind of *fractional-reserve bank* with none of a bank's safeguards. A normal bank also does not keep all your deposits in a vault — it lends most of them out and holds only a fraction in reserve, which is why a *bank run*, where everyone withdraws at once, can break even a solvent bank. (We unpack this in [how money is created](/blog/trading/finance/how-money-is-created-banks-central-banks-money-multiplier).) But a bank operates under capital rules, a central-bank backstop, and deposit insurance that pays you back if it fails. A crypto exchange can run the same fractional game — taking deposits, lending them out, holding only part in reserve — with *none* of those backstops: no required capital ratio, no lender of last resort, no insurance. So the exchange combines the fragility of a fractional-reserve bank with the absence of every protection that makes fractional-reserve banking survivable. That combination is the structural reason exchange failures are so total when they come.

The deepest reason the bundling is dangerous is that the three roles have *conflicting* duties. A custodian's only job is to keep your asset safe and untouched. A broker's job is to get you a good execution. An exchange's job is to run a fair, neutral marketplace. When one firm wears all three hats, those duties collide: the custodian half is supposed to leave your coins alone, while the lending and trading halves are tempted to put them to work. In TradFi, separating the firms separates the incentives. In crypto, the same balance sheet must somehow serve all three, and the temptation to dip into the custody pool to fund the trading book is exactly the temptation that has felled exchange after exchange.

## The big four: Binance, Coinbase, Kraken, OKX

Let us walk the named players, because their differences are the whole story of how crypto exchanges trade off scale against regulation.

### Binance

*Binance* is the largest crypto exchange in the world by trading volume, often handling well over \$10 billion of trades in a day across hundreds of tokens. It was founded in 2017 by Changpeng Zhao, universally known as *CZ*, and it grew at a speed TradFi has no comparison for: within months of launch it was among the biggest venues on earth. Its strategy was reach and speed — list almost everything, operate everywhere, and stay deliberately stateless, with no single declared headquarters, moving operations across jurisdictions to stay ahead of regulators.

That same statelessness produced the reckoning. In November 2023, Binance reached a settlement with the US Department of Justice and other agencies in which it agreed to pay roughly **\$4.3 billion** — one of the largest corporate penalties in US history — over charges that it had operated for years without an effective anti-money-laundering program and had willfully violated US sanctions, allowing transactions tied to sanctioned jurisdictions and illicit actors to flow through the platform. As part of the deal, CZ pleaded guilty, stepped down as CEO, paid a personal fine, and served a short prison sentence; a former colleague, Richard Teng, took over as CEO. Crucially, the charges Binance settled were about *money-laundering and sanctions controls* — not, as in FTX's case, about stealing customer money. The company kept operating. But the settlement was the moment the industry's largest player was forced under the umbrella of US law enforcement, and the era of "move faster than the regulators" visibly ended.

Binance also issues its own token, *BNB*, originally created to give fee discounts to people who paid trading fees in it. BNB grew into one of the largest tokens by market value — tens of billions of dollars — and that intertwining of an exchange's fortunes with a token it created is precisely the structure that, in FTX's case, turned fatal. More on that conflict shortly.

It is worth being precise about *why* Binance grew so fast, because the reasons are also the risks. First, it listed tokens aggressively — hundreds of them, including many that more cautious venues refused — which drew traders who wanted access to everything in one place. Second, it offered *derivatives*: not just spot buying of coins but *futures* and *perpetual swaps* with high *leverage* (borrowed money that multiplies both gains and losses), letting traders put on a \$100,000 position with \$5,000 of their own. Leveraged derivatives are where the volume — and the fees — are largest, and Binance became the dominant derivatives venue. Third, it operated globally and lightly regulated for years, so it could move faster than competitors anchored in one jurisdiction. Each of these drove growth, and each is also a hazard: aggressive listings include scams and worthless tokens, high leverage manufactures cascades of forced liquidations in a crash, and light regulation is exactly what produced the \$4.3 billion bill. The growth and the danger were, once again, the same engine.

### Coinbase

*Coinbase* is the mirror image: the regulated, US-based, publicly listed alternative. Founded in 2012 and headquartered in the US, it built its brand on doing the boring, compliant thing — heavy KYC, careful listings, working *with* US regulators rather than around them. In April 2021 it became the first major crypto exchange to *go public*, listing its shares directly on the Nasdaq stock exchange (ticker COIN). Going public means a company sells shares to the public and must thereafter file audited financials and answer to shareholders and securities regulators — exactly the scrutiny Binance avoided for years. Coinbase's volumes are smaller than Binance's (often a few billion dollars a day), its fees are higher, and it lists fewer tokens, but its proposition is the opposite of Binance's: trade a little less reach and pay a little more for a venue that lives inside the US legal system, holds customer assets under a US public company's disclosures, and is not going to vanish overnight to a different jurisdiction.

It is not risk-free. In June 2023 the US Securities and Exchange Commission (the SEC, the main US markets regulator) sued Coinbase, alleging it operated as an unregistered securities exchange — the central front in the regulatory war we will get to. But "being sued by the SEC over how tokens are classified" is a categorically different danger than "the company spent your deposits." Coinbase is the example of a CEX trying to be a TradFi-grade institution.

### Kraken, OKX, Bybit

Three more fill in the landscape. *Kraken*, founded in the US in 2011, is one of the oldest surviving exchanges, known for a security-first reputation and a strong presence in fiat trading pairs; it too was sued by the SEC in 2023 along similar lines to Coinbase. *OKX*, based in the Seychelles, is a very large global venue — frequently among the top by volume — with lighter oversight than the US firms, and it also runs its own token, OKB. *Bybit*, a large derivatives-focused exchange, earned a grim place in this list in February 2025 when it suffered what was reported as the largest crypto theft ever — roughly **\$1.5 billion** drained from one of its wallets in a single attack widely attributed to North Korea-linked hackers. (Bybit, notably, covered customer losses from its own capital and kept operating — the difference between a hack a solvent firm absorbs and a hole that sinks the firm.) Together these venues show the spectrum: from the heavily regulated US trio to the lighter-touch global giants, every one of them still bundling broker, exchange, and custodian into a single company.

## How exchanges make money

An exchange is a business, and understanding *how* it earns reveals where its incentives point — including where they point against you. There are five main revenue lines, and they stack.

![Stack of a centralized exchange's revenue lines](/imgs/blogs/centralized-crypto-exchanges-binance-coinbase-5.png)

The base layer, and the one everyone notices, is the **trading fee**: a percentage skimmed off every trade. These are typically in the range of 0.1% to 0.5% per trade, often split into a *maker* fee (cheaper, charged when your order adds liquidity to the book by resting on it) and a *taker* fee (pricier, charged when your order removes liquidity by hitting an existing order). It sounds tiny, but multiplied by billions of dollars of daily volume it is the engine of the whole business. We will compute one in a moment.

On top sits the **spread and the exchange's own market-making**. Because the exchange runs the order book and can act on its own account, it can also earn the bid-ask spread by quoting both sides — buying low, selling high — the same way a stock-market dealer does. This is a quieter, more conflicted source of income, because the house both runs the table and plays at it.

Third are **listing fees**. To get a new token added to a major exchange — which can mint or destroy that token's fortunes overnight — projects have at times paid the exchange large sums, reportedly running into the millions of dollars for a top venue. This is a genuine conflict: the exchange is paid by the very projects whose tokens it then offers to its users as if neutrally curated.

Fourth is a **cut of staking and lending yield**. *Staking* means locking up certain coins to help secure a blockchain in exchange for a reward; exchanges offer to do this for you and keep a slice of the reward. They also lend out deposited coins to borrowers (margin traders, institutions) and keep part of the interest. Both turn your idle balance into the exchange's income — and quietly increase the risk that the coins backing your balance are off doing something else.

Fifth, and most distinctive, is the **house token**. Binance has BNB; OKX has OKB; FTX had FTT. The exchange creates a token, gives it utility (fee discounts, special access), and benefits enormously as the token's value rises — it usually holds a large share itself. This braids the exchange's solvency together with the price of a token it controls, and that braid is the exact mechanism that detonated FTX.

#### Worked example: the trading fee on a \$10,000 Bitcoin buy

Suppose you decide to buy \$10,000 of Bitcoin on a typical exchange that charges a 0.1% taker fee (Binance's standard rate; some venues charge more).

- You place a market order to buy \$10,000 of Bitcoin.
- The taker fee is 0.1% of \$10,000 = **\$10**.
- So you spend \$10,000 and receive about \$9,990 worth of Bitcoin; the exchange keeps \$10.

Now do it on a higher-fee venue at 0.5% — closer to Coinbase's retail pricing for small trades:

- The fee is 0.5% of \$10,000 = **\$50**. You receive about \$9,950 of Bitcoin.

The difference between a 0.1% and a 0.5% venue on this single trade is \$40 — five times as much fee for the same purchase. Intuition: the fee is small per trade but it is charged *every* time you trade, in both directions, so an active trader pays it again and again; the headline rate, multiplied by your turnover, is the real cost of using a convenient venue.

#### Worked example: the bid-ask spread you actually pay

Fees are visible; the spread is the cost most people never see. Suppose Bitcoin's order book shows a best bid of \$99,950 and a best ask of \$100,050 — a \$100 spread, or 0.1% of the price.

- You buy *now* at the ask: \$100,050 per coin.
- A moment later you change your mind and sell *now* at the bid: \$99,950 per coin.
- You did not move the market and the price did not change — yet you are down \$100 per coin, the full spread, just for crossing in and back out.

On a \$10,000 position (about 0.1 coin at this price), that round-trip spread costs roughly \$10 — *on top of* the \$10 or \$50 in explicit fees. On a thinly traded *altcoin* (any coin other than Bitcoin) the spread can be 1% or worse, so a \$10,000 round trip can quietly cost \$100 or more before any named fee. Intuition: the headline fee is only half your trading cost; the spread is the other half, and it is widest exactly where liquidity is thinnest, which is where beginners often trade.

## The custody conflict: your coins are on their books, not in your wallet

We have laid every piece; now we assemble the central problem. When you hold crypto on a CEX, *your coins are an entry on the company's books, not coins in a wallet you control.* The exchange holds the keys to the pooled wallets; you hold a claim. Re-read the figure of where the coins sit if you need to. This single fact generates a stack of conflicts that the TradFi separation of roles was specifically built to prevent.

The first conflict is the most basic: **the firm holding your assets can use them.** Because deposits are commingled in pools the exchange controls, and because nothing on-chain ties any coin to any customer, the exchange is physically able to lend those coins out, post them as collateral, trade with them, or — in the worst case — spend them, and your database balance will not flicker. In a well-run, honest exchange this never happens; the pool stays full. But "we promise not to" is the only thing standing between you and the abuse, where in TradFi a wall of law, segregated accounts, and a separate custodian stand there instead.

The second is **the house token loop.** When an exchange's own balance sheet is propped up by a token it issued, its solvency becomes circular: it looks rich because its token is valuable, and the token is valuable partly because the exchange supports it. If confidence cracks, the token falls, the balance sheet shrinks, and the fall feeds on itself. This is not hypothetical — it is precisely the mechanism that destroyed FTX, whose balance sheet was stuffed with its own FTT token.

The third is **information asymmetry.** The exchange sees every resting order, every stop-loss, every customer's full position. A venue that also trades on its own account is sitting on a view of the market no outside trader can match. Whether any given exchange abuses that is a question of conduct; that the structure *permits* it is a question of design, and the design is the point.

The fourth is **withdrawal control.** Because the exchange holds the keys, it controls the exit. In a panic — a *bank run*, where everyone tries to withdraw at once — the exchange can simply pause withdrawals, and a customer holding only a claim has no on-chain way to force the coins out. Several failing exchanges froze withdrawals in their final days; the freeze is usually the first public sign that the pool is not full.

A fifth, subtler conflict lives in the *liquidation engine* on exchanges that offer leverage. When you trade with borrowed money and the price moves against you past a threshold, the exchange *liquidates* your position — force-sells it to repay the loan before your losses exceed your collateral. The exchange both sets the rules for when this happens and, often, runs the engine that does the selling, and it earns liquidation fees when it triggers. In a sharp crash, thousands of leveraged positions liquidate at once, the forced selling drives the price down further, which triggers still more liquidations — a cascade. The exchange profits from the fees on every one of them. There is no suggestion that the major venues deliberately hunt customer liquidations, but once more the structure permits a conflict that the separation of roles in TradFi — where the lender, the exchange, and the broker are different firms — does not.

#### Worked example: self-custody vs exchange-custody, priced as a tradeoff

Imagine you hold \$10,000 of Bitcoin and weigh two ways to keep it.

- **On the exchange:** You can trade in one tap, the password is recoverable, support exists, and you might earn, say, 3% a year if you let the exchange stake or lend it — about \$300 a year. The cost is *counterparty risk*: if the exchange is insolvent or hacked and cannot cover the loss, your \$10,000 claim can be worth a fraction of that, or nothing. Call the chance of a total loss small in any given year — but it is not zero, and FTX's depositors learned it can be 1.
- **In self-custody:** You move the \$10,000 to a wallet whose keys only you hold. Now no exchange can lend, freeze, or lose it; counterparty risk drops to essentially zero. The cost shifts onto *you*: lose the key and it is gone forever, fall for a scam that tricks you into signing it away and there is no reversal, and you earn no yield while it sits there.

The decision is not "which is safer" but "which risk do you prefer to carry" — the risk that a company fails, or the risk that you fail. Intuition: an exchange converts the terrifying, irreversible risk of losing your own key into the more familiar risk of trusting an institution, and charges you, in yield and in fees, for the conversion; the FTX lesson is that the institution's risk is realer than its convenience makes it feel.

## Proof of reserves after FTX — and its limits

After FTX collapsed in late 2022 (the full anatomy is in [the FTX collapse deep dive](/blog/trading/crypto/ftx-collapse-sam-bankman-fried)), depositors everywhere asked the obvious question of every other exchange: *do you actually have my coins?* The industry's answer was a wave of *proof of reserves* publications, and Binance, OKX, Kraken, and others raced to show wallet attestations. It was real progress. But it is essential to understand what it proves and, more importantly, what it does not.

Proof of reserves, done properly, demonstrates two things at one instant: that the exchange controls wallets holding a certain amount of crypto (the *assets* side), and, via a Merkle tree of customer balances, that it owes a certain total to depositors (the *liabilities* side). If assets ≥ liabilities at that snapshot, depositors are at least covered *for the coins that were measured*. That is genuinely useful.

Now the holes, because they are large. **First, it is a snapshot.** It proves nothing about the moment before or after; an exchange could borrow coins to pass the audit and return them the next day — a trick known to have happened. **Second, it only measures liabilities to depositors, not all liabilities.** An exchange can hold \$10 billion of assets against \$10 billion of customer deposits and look fully reserved, while *also* owing \$5 billion to outside lenders that the proof never shows — so it is actually deeply insolvent. **Third, it usually ignores asset quality and the house token.** If a third of the "assets" are the exchange's own illiquid token, the reserve evaporates the instant anyone tries to sell. **Fourth, it requires a trustworthy auditor and complete liability disclosure**; a partial liability list makes the assets look more than sufficient. Proof of reserves is a real improvement over the FTX-era darkness, but it is a photograph of one side of the ledger at one instant — not an audited statement of solvency, and certainly not insurance.

There is also a more technical hole worth naming, because it shows how a number can be true and useless at once. Suppose two exchanges share a relationship and one lends the other a billion dollars of Bitcoin for a single day. On that day, *both* can run a proof of reserves and *both* can truthfully show they control the coins — because the coins moved into the borrower's wallet for the snapshot. The same billion dollars is counted twice, once at each venue, and neither attestation is technically lying. The only fix is a *liabilities-inclusive, auditor-verified, point-in-time* attestation that also rules out borrowed assets — and almost no public proof of reserves does all three. What the industry calls "proof of reserves" is usually closer to "proof we touched these wallets recently," which is to a real solvency audit roughly what a selfie in front of a bank is to a bank statement.

#### Worked example: a proof-of-reserves shortfall

Picture an exchange, "ExampleX," that publishes a proof of reserves.

- It attests on-chain control of wallets holding **\$8 billion** of crypto — the assets side, verifiable.
- Its Merkle tree of customer balances totals **\$10 billion** — what it owes depositors.

Right there, assets of \$8 billion against \$10 billion of customer liabilities is a **\$2 billion shortfall**: it holds only 80 cents for every dollar it owes. If every customer tried to withdraw, the last \$2 billion of claims could not be paid, and a run would expose it instantly.

Now suppose, worse, that ExampleX's "proof" *omits* the \$10 billion liability tree and only trumpets the \$8 billion of assets, while it *also* owes \$3 billion to outside lenders that never appear anywhere. A casual reader sees "\$8 billion in reserves!" and feels safe; the real picture is \$8 billion of assets against \$13 billion of total obligations — a \$5 billion hole hidden behind a true-but-incomplete number. Intuition: a reserve figure means nothing until you can see the *liabilities* it is supposed to cover, including the ones the exchange would rather you never tally; a number with only one side of the ledger is marketing, not proof.

## The SEC regulatory war

The deepest fight over centralized exchanges in the US is not about fraud at all — it is about *classification*, and it determines which rulebook exchanges must live under. The SEC's position, pressed hard from 2023 onward, is that many crypto tokens are *securities* — investment contracts, the same legal category as a stock — and that a platform listing them is therefore an unregistered *securities exchange*, *broker*, and *clearing agency* all rolled into one, operating illegally because it never registered as any of them. On that theory the SEC sued Coinbase and Kraken in June 2023 (and had earlier targeted Binance on overlapping grounds).

The exchanges' counter is that most tokens are *not* securities — they are commodities, or a new category entirely — and that the SEC is trying to regulate by lawsuit rather than writing clear rules, leaving firms unable to comply even if they wanted to. The stakes are concrete: if tokens are securities, exchanges must register, segregate customer assets under securities rules, separate the broker and exchange functions, and submit to disclosure — in other words, the SEC's endgame would *force the TradFi separation onto crypto*. That is why the classification fight is really the custody fight in legal clothing. The state of this war shifts with each court ruling and each change in administration, so treat any specific status as a snapshot; the durable point is that the entire legal battle is about whether one company may keep being broker, exchange, and custodian at once.

#### Worked example: Binance's \$4.3 billion settlement in context

The \$4.3 billion Binance agreed to pay in 2023 is hard to feel without comparisons, so let us scale it.

- It is among the **largest corporate resolutions in US history** — in the same league as penalties once reserved for the biggest banks and carmakers, now levied on a crypto exchange barely six years old.
- Set it beside fees: at a 0.1% trading fee, an exchange must process **\$4.3 trillion** of trading volume to earn \$4.3 billion in fee revenue. Binance's scale meant the fine, though enormous, was survivable — a measure of just how large the business had grown.
- Contrast the *nature* of it with FTX. Binance's \$4.3 billion was a penalty for *failing to police money laundering and sanctions* — a compliance failure, paid by a company that remained solvent and kept its customers' coins. FTX's roughly **\$8 billion** hole was *missing customer money* — the firm had spent it. One number is a fine a living business absorbs; the other is the grave of a dead one.

Intuition: the dollar figures are similar in size but opposite in kind, and the difference is the whole moral of centralized exchanges — a fine for bad controls is recoverable, but a hole where the customer deposits used to be is the catastrophe that bundling makes possible.

## FTX: the cautionary tale of what bundling enables

Everything in this post points at one event. *FTX* was, in 2022, one of the largest crypto exchanges in the world, run by Sam Bankman-Fried, with a sister trading firm called *Alameda Research*. The two were supposed to be separate; they were not. FTX took customer deposits and, through a back door in its own code, funneled billions of dollars of them to Alameda, which used the money for trading bets, venture investments, and propping up FTX's own house token, FTT. When a report in November 2022 revealed how much of Alameda's "assets" were just FTT — a token FTX itself created — confidence cracked, FTT fell, and depositors rushed to withdraw. The pool was not full. Within days FTX froze withdrawals and filed for bankruptcy, with a hole of roughly \$8 billion in customer funds. Bankman-Fried was later convicted of fraud.

![Timeline of CEX milestones, the FTX collapse, and the Binance settlement](/imgs/blogs/centralized-crypto-exchanges-binance-coinbase-6.png)

Read the figure as the recurring lesson it is. FTX is not a story about crypto being uniquely evil. It is the textbook demonstration of *what the bundling makes possible*: because one company was broker, exchange, custodian, lender, and token issuer with no separation and no real oversight, it could take customer coins out of the pool, lend them to its own trading arm, and back the whole edifice with a token it printed — and no outside custodian, no segregation rule, no separate exchange stood in the way. Every conflict of interest we cataloged in the custody section was present at once, and nothing structural stopped any of them. The full anatomy — the timeline, the on-chain mechanics, the aftermath — is in [the FTX collapse deep dive](/blog/trading/crypto/ftx-collapse-sam-bankman-fried). For our purposes the takeaway is narrow and sharp: FTX is what the architecture *permits* when the promises are the only safeguard, and the promises fail.

## Common misconceptions

**"My coins are sitting safely in my account on the exchange."** No — your coins are in the *exchange's* pooled wallets, and what is "in your account" is a database entry recording a claim. If the pool is full and honest, the claim is good. If it is not, the entry does not protect you. Your account is an IOU, not a vault.

**"Proof of reserves means the exchange is solvent and safe."** No — proof of reserves, at best, shows assets ≥ depositor liabilities at one instant. It does not show debts to other lenders, does not prove the assets were not borrowed for the snapshot, does not judge whether the assets are a worthless house token, and says nothing about tomorrow. A passing proof of reserves is necessary, not sufficient.

**"Binance was shut down / Binance stole customer money like FTX."** No — Binance's \$4.3 billion settlement was about anti-money-laundering and sanctions failures, not about spending customer deposits. CZ stepped down and the firm continued operating with its customers' coins intact. Conflating the Binance settlement with the FTX fraud blurs the single most important distinction in this whole subject: a compliance fine versus a missing-funds catastrophe.

**"A US-listed, regulated exchange like Coinbase can't fail or freeze my funds."** Being public and regulated reduces certain risks — disclosure, audits, US legal recourse — but it is still a custodial company holding your coins on its books, still bundling the three roles, and still exposed to hacks, legal action, and market stress. Regulation lowers the odds; it does not change the structure.

**"A decentralized exchange is just a safer version of a centralized one."** No — a DEX is a different thing, not a safer version. It removes custody risk because it never holds your coins, but it adds the full weight of self-custody (lose your key, lose everything; sign a malicious transaction, no reversal) plus *smart-contract risk* (a bug in the code can drain it). It trades one set of dangers for another; it does not make the dangers disappear.

**"If an exchange gets hacked, I'm automatically covered."** Sometimes, not always. Some large, solvent exchanges have absorbed hacks from their own capital (Bybit covered its roughly \$1.5 billion 2025 loss; Bitfinex famously did not fully cover its 2016 hack at the time and instead socialized the loss across all customers). Whether you are made whole depends entirely on the specific exchange's solvency and policy — there is no industry-wide deposit insurance like a bank's.

## How it shows up in real markets

**Mt. Gox (2014).** The original cautionary tale. Mt. Gox, based in Japan, once handled the majority of all Bitcoin trades, and then collapsed after roughly **850,000 Bitcoin** belonging to customers went missing over years of undetected theft and mismanagement. It established, a decade before FTX, that an exchange holding your coins is a single point of failure — and creditors are *still* being repaid, more than a decade later.

**The Coinbase IPO (April 2021).** When Coinbase listed on Nasdaq, it was a landmark: a crypto exchange entering the most established market in the world, subjecting itself to public-company disclosure and valued at tens of billions of dollars at its debut. It marked the moment a CEX could be a mainstream, audited, regulated institution — the opposite pole from Binance's statelessness.

**The FTX collapse (November 2022).** The roughly \$8 billion hole, the frozen withdrawals, the FTT death spiral, the fraud conviction — the defining demonstration of what bundling broker, exchange, custodian, lender, and token issuer into one unsupervised company makes possible. Covered above and in its own [deep dive](/blog/trading/crypto/ftx-collapse-sam-bankman-fried).

**The Binance settlement (November 2023).** The world's largest exchange paying roughly \$4.3 billion and its founder stepping down and serving time — the moment the era of out-running regulators ended and the biggest player came under US law enforcement, while, crucially, keeping customer funds intact.

**Proof-of-reserves season (late 2022 onward).** In FTX's wake, exchange after exchange rushed out wallet attestations, and the public got a fast education in what those numbers do and do not prove — including the discovery that some attestations omitted liabilities or used borrowed coins, which is why the practice, while an improvement, never became a guarantee of solvency.

**Exchange hacks: Bitfinex (2016) and Bybit (2025).** Two bookends on hot-wallet theft. Bitfinex lost roughly 120,000 Bitcoin in 2016 and spread the loss across all customers with a token-based recovery scheme; Bybit lost roughly \$1.5 billion in 2025 — the largest crypto theft on record — and covered it from its own capital. Same root cause (keys on internet-connected servers), opposite outcomes for depositors, decided entirely by the exchange's solvency and choices.

**The SEC suits against Coinbase and Kraken (2023).** The opening of the regulatory war over whether tokens are securities and whether exchanges must therefore unbundle and register — the legal expression of this entire post's thesis, still unresolved and shifting with each ruling.

## When this matters to you, and further reading

If you ever buy crypto, almost everything above touches you directly, and it reduces to a few habits. Treat an exchange as a place to *trade*, not a place to *store*: the longer coins sit on a CEX, the longer you carry counterparty risk for no extra return beyond modest yield. If you hold a meaningful amount, learn self-custody and move it to a wallet whose keys you control — accepting that the responsibility moves to you. Read a "proof of reserves" headline for what it omits, especially the liabilities side. Watch withdrawal behavior: a venue that suddenly "pauses withdrawals for maintenance" in a stressed market is showing you the first symptom of an empty pool. And keep the Binance-versus-FTX distinction sharp — a compliance fine is survivable; a missing-funds hole is fatal — because the headlines will keep blurring them.

The deeper point is the one the TradFi contrast keeps making. Traditional finance separated broker, exchange, and custodian after its own century of disasters taught it why; crypto re-bundled them for convenience and is relearning the same lessons in fast-forward. Centralized exchanges are extraordinarily useful — they are the door through which nearly everyone enters crypto — and that usefulness is inseparable from the conflicts that make them dangerous. The convenience and the danger are the same fact seen from two sides.

To go deeper: [the FTX collapse, step by step](/blog/trading/crypto/ftx-collapse-sam-bankman-fried), shows what the bundling enables when the promises fail; [DeFi protocols like Uniswap and Aave](/blog/trading/crypto/defi-protocols-uniswap-aave-makerdao) show the non-custodial alternative and its own different risks; [inside an investment bank](/blog/trading/finance/inside-an-investment-bank-how-they-make-money) and [the field guide to financial institutions](/blog/trading/finance/field-guide-to-financial-institutions) show how and why the traditional world keeps these roles in separate hands. Read together, they make the same argument from four directions: who holds the keys, and who is allowed to spend what is yours, is the only question that finally matters.
