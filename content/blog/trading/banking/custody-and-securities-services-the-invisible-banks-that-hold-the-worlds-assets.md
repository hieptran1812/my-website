---
title: "Custody and Securities Services: The Invisible Banks That Hold the World's Assets"
date: "2026-06-21"
publishDate: "2026-06-21"
description: "How custodian banks like BNY and State Street safekeep, settle, and service tens of trillions of dollars of assets they do not own — and why a fraction of a basis point on that mountain is one of banking's stickiest businesses."
tags: ["banking", "custody", "securities-services", "custodian-banks", "settlement", "fund-administration", "securities-lending", "asset-servicing", "bny-mellon", "state-street", "central-securities-depository"]
category: "trading"
subcategory: "Banking"
author: "Hiep Tran"
featured: true
readTime: 44
---

> [!important]
> **TL;DR** — A custodian bank is the vault and the back office for everyone else's investments: it safekeeps the assets, settles the trades, and services the holdings, but it never owns a single share. That is a capital-light, low-margin, enormous-scale, very sticky fee-and-deposit business that quietly underpins the entire asset-management industry.
>
> - **Custody is not management.** A custodian *holds* tens of trillions; an asset manager *decides* what to buy. Assets under custody (AUC) and assets under management (AUM) are two completely different numbers measuring two completely different jobs.
> - **The fee is tiny, the base is gigantic.** All-in custody fees run on the order of **1 basis point** (0.01%) of assets held. On \$50 trillion that is still about **\$5 billion a year** of recurring revenue — plus the client cash that sits as cheap deposits.
> - **The plumbing is invisible until it breaks.** Settlement, corporate actions, fund administration, and securities lending all run through the custodian and the central securities depository. When that machinery fails — as securities-lending cash reinvestment did in 2008 — the loss lands on the bank, not the held assets.
> - **The number to remember:** the largest custodian, **BNY**, holds on the order of **\$50 trillion** of assets it does not own. The whole point of the business is that you have never had to think about who is holding your shares.

A few firms you have probably never opened an account with are holding almost everything you own.

If you have a pension, an index fund, a 401(k), or a brokerage account, the shares and bonds behind it are not sitting in your broker's desk drawer. They are not even, in any literal sense, sitting at the company whose stock you "own." They are recorded, deep in a chain of intermediaries, by a custodian bank — a firm like **The Bank of New York Mellon** (BNY), **State Street**, **JPMorgan**, or **Citi** — and at the bottom of that chain, by a **central securities depository (CSD)**. BNY alone safekeeps on the order of **\$50 trillion** of assets. To put that in scale: that is roughly half of all the financial wealth held by households in the United States, sitting inside the ledgers of one bank that owns none of it.

These are the invisible banks. They do not run flashy ad campaigns, they do not chase your checking account, and most people who work in finance could not tell you what State Street actually *does* day to day. What they do is hold the world's assets, move them when they are traded, collect the dividends and votes and interest that flow off them, keep the books for the funds that own them, and lend them out for a fee. It is unglamorous, it is operationally relentless, and it is one of the most durable franchises in all of banking.

![Custody chain showing investor to asset manager to global custodian to sub-custodian to central securities depository to issuer](/imgs/blogs/custody-and-securities-services-the-invisible-banks-that-hold-the-worlds-assets-1.png)

The diagram above is the mental model for this entire post: you, the true owner, sit at the top; below you a chain of agents holds your assets on your behalf; and the custodian bank sits in the middle of that chain, touching everything, owning nothing. By the end you will understand every box — what each one does, why it exists, where the money is, and where the risk hides. And you will see how this fits the spine that runs through this whole series: a bank is a leveraged, confidence-funded machine that lives or dies on trust. The custody business is the purest expression of that idea, because here the *only* product is trust. A custodian sells you the promise that your assets are safe, recorded correctly, and will be there when you ask for them. Everything else is just the machinery that keeps that promise.

## Foundations: the words you need before we go deep

Custody has a vocabulary problem. The terms sound interchangeable — custody, safekeeping, settlement, servicing, administration — and they are not. Let us define every one of them from zero, with a single running analogy, before we touch a number.

Here is the analogy. Picture a **coat-check and back office for everyone else's valuables**. People bring their coats (their assets) and hand them over. The coat-check does four distinct jobs. First, it *holds* the coats safely and keeps a numbered ticket proving whose is whose — that is **safekeeping**. Second, when one guest sells their coat to another, it physically hands the coat to the buyer and takes the payment, making sure neither side gets cheated — that is **settlement**. Third, while it has the coat, it brushes off the lint, sews on the button that came loose, and collects the dry-cleaning voucher that came with it — that is **asset servicing**. Fourth, it keeps a tidy ledger of who owns what and what each coat is worth today — that is **administration**. The coat-check never owns a single coat. It just charges a small fee per coat and earns the trust of every guest. That is a custodian.

Now the precise definitions.

A **custodian** (or *custodian bank*) is a financial institution that holds clients' securities — stocks, bonds, funds — for safekeeping, so that those assets are not lost or stolen, and provides the services that keep those holdings in good order. The client owns the assets; the custodian merely holds and tends them. The largest are **global custodians** (BNY, State Street, JPMorgan, Citi, BNP Paribas) that can hold assets across dozens of markets at once.

**Safekeeping** is the core job: keeping the assets secure and keeping an accurate record of who owns them. In the modern world almost nothing is a paper certificate any more; "holding" a share means holding an electronic *book entry* — a line in a ledger — that says this many shares belong to this account. Safekeeping is the integrity of that record.

**Settlement** is the process of completing a trade: actually transferring the security from seller to buyer and the cash from buyer to seller, so the trade is *final*. A trade is *agreed* in seconds on an exchange; it is *settled* a day or two later, when the shares and the money truly change hands. The custodian is the agent that makes settlement happen for its clients.

**Asset servicing** is everything the custodian does to the assets while it holds them. The big three are **corporate actions** (handling stock splits, mergers, rights issues, and the like — events the issuing company initiates that change your holding), **income collection** (collecting the dividends on shares and the coupons on bonds and crediting them to the owner), and **proxy voting** (passing along the right to vote at company meetings, or voting per the owner's instructions). If your fund owns 4,000 different securities across 30 countries, somebody has to track every dividend, every split, every vote, in every currency — that somebody is the custodian.

**Fund administration** is bookkeeping for an investment fund: independently calculating the fund's **net asset value (NAV)** — the per-share value of the fund, which is the total value of what it holds minus its liabilities, divided by the shares outstanding — keeping the fund's official books, and handling investor subscriptions and redemptions. It is what lets an investor trust that a fund's stated price is real and independently checked.

**Securities lending** is lending out the held securities — temporarily transferring shares to a borrower (usually someone who wants to sell them short) in exchange for collateral and a fee. The owner keeps the economic upside; the custodian arranges the loan and splits the fee. It is the one place in the custody business where the held assets actually generate extra income.

A **central securities depository (CSD)** is the master record-keeper for a whole market: the single ledger where the definitive ownership of a country's securities lives. In the United States it is the **DTC** (Depository Trust Company, part of DTCC). In Europe there are many, plus two giant *international* CSDs, Euroclear and Clearstream. Custodians are members of CSDs; the CSD is where the buck finally stops on "who owns this share."

A **sub-custodian** is a local custodian that a global custodian appoints in a foreign market it cannot reach directly. To hold a Japanese stock for a U.S. client, BNY does not become a member of the Japanese depository itself; it hires a bank in Tokyo — the sub-custodian — that *is* a member, to hold the asset locally on its behalf. This is why the chain has more than one link.

Two more terms you must keep separate, because confusing them is the single most common mistake about this business:

- **Assets under custody (AUC)** — the value of all the assets a custodian *holds and safekeeps*. It is a measure of how much stuff is in the vault. The custodian makes no investment decisions about any of it.
- **Assets under management (AUM)** — the value of all the assets a manager *makes decisions about*: what to buy, what to sell, how to weight. That is a different job done by a different firm (or a different division).

AUC is almost always a much bigger number than AUM, because one custodian holds the assets of *many* managers. Hold these two apart and most of this post will make sense.

These services do not stand alone — they nest. Custody is the foundation, and everything else is bolted on top of the same held assets. That bundling is the strategy: once a client's assets are in the vault, the custodian is uniquely placed to settle their trades, service their holdings, keep their fund's books, and lend their stock, and every one of those is a fresh recurring fee on the same pile.

![Matrix of the securities-services product suite: custody, settlement, asset servicing, fund administration, securities lending, with what each does and how it earns](/imgs/blogs/custody-and-securities-services-the-invisible-banks-that-hold-the-worlds-assets-4.png)

The matrix above lays out the five product lines we will work through and how each one earns its money. **Custody** safekeeps the assets and keeps the ownership record, for a basis-point fee on assets held. **Settlement** moves shares against cash when a trade is done, for a fee per transaction. **Asset servicing** collects dividends, votes proxies, and handles corporate actions, for a bundled fee plus the foreign-exchange margin on income. **Fund administration** strikes the NAV and keeps the fund's books, for a fee per fund and per share class. **Securities lending** lends the held shares to short sellers, splitting the lending spread with the owner. Notice the pattern: every line charges a recurring fee on assets the custodian does not own. That is the whole business in one picture — and we will now take each line in turn.

With the vocabulary set, we can go deep. We will walk the chain top to bottom: what custody actually is, how a trade settles, what servicing involves, how fund administration works, how securities lending makes money, what the economics really are, and where it all goes wrong.

## What custody actually is: holding what you do not own

Strip away the jargon and custody is a single, ancient idea: I have something valuable, I do not want to guard it myself, so I pay someone trustworthy to hold it for me. The Medici held other people's money; a hotel safe holds your passport; a custodian holds your shares. The reason this is a *bank* job, and a very large one, is that holding modern financial assets is not passive at all. It is a continuous, high-stakes, record-keeping operation.

Start with what "holding a share" even means today. There is no certificate. When you buy 100 shares of Apple, what changes is a number in a chain of ledgers. At the bottom, the **DTC** records that a certain custodian holds a big pile of Apple shares. The custodian's own books record that, within that pile, 100 of them belong to your asset manager's account. The asset manager's books record that those 100 belong to the fund you invested in. The fund's books record your slice. This is called holding **in street name** or through *omnibus* accounts — your shares are pooled with everyone else's at each level, and your ownership is established by the chain of records, not by a piece of paper with your name on it.

This is enormously efficient — billions of shares can change hands a day without a single certificate moving — but it means the *integrity of the records* is everything. If the custodian's ledger is wrong, your ownership is in doubt. So the custodian's first and deepest obligation is **segregation**: keeping client assets legally and operationally separate from the bank's own assets, and from each other. Client securities held in custody are *not* the bank's property and do not sit on the bank's balance sheet as the bank's assets. If the custodian itself failed, those securities are supposed to be ring-fenced and returned to clients — they are not available to the custodian's creditors. This is the legal bedrock of the whole business and the reason a custodian can hold \$50 trillion on a balance sheet that is a small fraction of that size.

Compare this to the rest of the bank. In [what a bank actually does](/blog/trading/banking/the-payments-business-how-money-actually-moves-between-banks), the deposits a bank takes *are* its liabilities, and the loans it makes *are* its assets — the assets and liabilities are the bank's own, and the thin equity cushion sits between them. In custody, the assets are the *clients'*, sitting off to the side. The custodian's balance sheet is mostly the *cash* clients leave with it (which does become the bank's deposits — we will come back to this) and its own modest operations. That structural difference is why custody is **capital-light**: you can run an enormous franchise on relatively little of your own capital, because you are not taking the credit risk of owning the assets.

#### Worked example: a custody fee in basis points on AUC

Let us put the first number on the page. A *basis point* (bp) is one hundredth of one percent — 0.01%, or 0.0001 as a decimal. Custody is priced in basis points of assets held, and the all-in rate for large institutional clients is very low — often **around 1 basis point** or even less for the biggest, simplest portfolios, rising for complex multi-market mandates with lots of servicing.

Say a pension fund hands a custodian \$10 billion of assets to safekeep, at an all-in fee of 1 basis point per year.

$$\text{Annual fee} = \$10{,}000{,}000{,}000 \times 0.0001 = \$1{,}000{,}000$$

So the custodian earns **\$1 million a year** to hold \$10 billion. That is one ten-thousandth of the assets. It sounds almost insultingly small — and from the client's side it is a bargain, which is exactly why they do not shop it around much. The intuition: custody is the cheapest service in finance measured against the assets it touches, and that cheapness is *the moat* — no client switches custodian to save a single basis point, because the operational pain of moving \$10 billion of holdings dwarfs the saving.

Hold onto that \$1-per-\$10,000 ratio. The entire economics of the business is the tension between that tiny rate and the staggering base it is applied to.

## Settlement and the CSD: where a trade actually finishes

A trade has two lives. There is the moment it is *agreed* — two parties match on price and size, in milliseconds, on an exchange or between dealers. And there is the moment it is *settled* — when the security and the cash truly change hands and the trade becomes final and irreversible. Those two moments are not the same, and the gap between them is where custodians and CSDs earn their keep.

Why is there a gap at all? Because matching a trade is easy and *moving* assets safely is hard. You have to make sure the seller actually has the shares, the buyer actually has the cash, the records are updated everywhere in the chain, and — crucially — that neither side hands over their half without getting the other half. In the United States and many markets the standard settlement cycle is now **T+1**: trade date plus one business day. (It used to be T+3, then T+2; the U.S. moved to T+1 in May 2024.) That one-day delay is the operational window in which all the plumbing runs.

The single most important idea in settlement is **delivery-versus-payment (DvP)**: the rule that the delivery of the security and the payment of the cash happen *simultaneously and conditionally* — one cannot occur without the other. This is the mechanism that removes **principal risk**: the risk that you pay for shares that never arrive, or deliver shares for money that never comes. Before DvP was universal, this risk was real and occasionally catastrophic; the failure of the German bank Herstatt in 1974, caught mid-settlement across time zones, is the textbook case that pushed the whole system toward simultaneous, conditional settlement.

![Pipeline of a delivery-versus-payment settlement: trade agreed, custodians instruct, CSD matches, DvP swap, books updated, settled at T plus one](/imgs/blogs/custody-and-securities-services-the-invisible-banks-that-hold-the-worlds-assets-3.png)

Walk the pipeline above. A trade is agreed. Each side's custodian sends a *settlement instruction* to the CSD describing the trade — the security, the quantity, the price, the date, the counterparty. The CSD **matches** the two instructions; if the buyer's and seller's instructions agree on every field, the trade is ready to settle. On settlement day the CSD performs the DvP swap: in one linked, atomic step it debits the seller's securities account and credits the buyer's, and at the same time moves the cash the other way. Neither leg happens unless both can. The books are updated everywhere, and the trade is *final* — done, irreversible, settled.

The custodian's role here is to be the client's hands inside this system. The asset manager does not talk to the CSD directly; it tells its custodian "I bought 100,000 shares from this broker, settling Tuesday," and the custodian instructs the CSD, monitors the match, makes sure the cash is in place, and confirms settlement. When something goes wrong — the counterparty does not deliver, a number is mismatched, the security is short — the custodian manages the **failed trade**, chasing it, sometimes funding it, and reporting it. Multiply this by millions of trades a day across dozens of markets and you see why a custodian is, at its core, an industrial-scale settlement factory.

The sub-custodian chain matters most here, when the trade is in a foreign market. To settle that 100,000-share trade in, say, Tokyo, BNY does not connect to the Japanese depository itself — it instructs its **sub-custodian** in Tokyo, a local bank that *is* a member of the local CSD, to settle on its behalf. The instruction therefore travels global custodian → sub-custodian → local CSD, and the confirmation travels back up the same chain. Each link adds a small fee, a small delay, and a small risk, which is why a global custodian's *network* of sub-custodians across 100-plus markets is itself a competitive asset: the more markets you can reach reliably and cheaply, the more attractive you are to a globally diversified fund. Building and maintaining that network is one of the real barriers to entry in global custody — you cannot conjure trusted local agents in a hundred jurisdictions overnight.

#### Worked example: settling a trade delivery-versus-payment

Suppose a fund managed by your asset manager buys **100,000 shares** of a company at **\$50.00** a share, settling T+1, through its custodian.

The trade value is:

$$100{,}000 \times \$50.00 = \$5{,}000{,}000$$

On trade date, the deal is agreed but nothing has moved. The custodian receives the instruction: receive 100,000 shares, pay \$5,000,000, on settlement day, versus the seller's broker. The next business day, at the CSD, the DvP swap fires:

- The seller's securities account is debited 100,000 shares; the buyer's (your fund's, via its custodian) is credited 100,000 shares.
- *At the same instant*, \$5,000,000 of cash moves from the buyer's account to the seller's.

If the buyer's cash were not there, the securities leg would not fire — and vice versa. Neither side is ever exposed to having given up their half without receiving the other. The intuition: DvP turns a two-sided promise into a single all-or-nothing event, which is the only reason it is safe to settle \$5 million on the word of a counterparty you may never have met.

Now scale that. Across all U.S. equities, the DTCC settles trades worth *trillions* of dollars on a typical day. The custodian is the connective tissue between the millions of investors and the one ledger where it all nets down.

## Asset servicing: the unglamorous work of tending the holdings

Holding a share and settling its trade is only the start. While the custodian holds your assets — which could be years — those assets are *alive*. Companies pay dividends, split their stock, merge, spin off divisions, issue rights, change names, and call meetings. Bonds pay coupons, get called, mature, and default. Every one of these events is something that must be *handled correctly for every holder*, in every currency, on the right date, or the owner loses money or rights. This is **asset servicing**, and it is the labor-intensive heart of the business.

There are three pillars.

**Corporate actions.** A *corporate action* is any event initiated by the issuer that affects its securities. They come in two flavors. *Mandatory* actions happen to you whether you do anything or not — a stock split (your 100 shares become 200, each worth half as much), a cash dividend, a merger where your shares convert to the acquirer's. *Voluntary* (or elective) actions require a decision — a rights issue where you may buy more shares at a discount, a tender offer where you may sell at a premium, a dividend you may take in cash or in stock. For voluntary actions the custodian must notify the owner, collect the election by a hard deadline, and execute it correctly. Get the deadline wrong on a rights issue and the client's right simply lapses, worthless. There is no second chance. This is why corporate-actions teams are some of the most carefully run operations in any bank — the errors are immediate, dated, and expensive.

**Income collection.** Every dividend and every coupon that the held assets throw off has to be *collected* from the issuer (often via the sub-custodian and CSD), converted to the owner's currency if needed, taxed correctly under the right treaty (a U.S. fund holding a French stock may be entitled to a reduced French withholding-tax rate — the custodian files for it), and credited to the owner's account on the right day. A global fund might receive thousands of income events a year across dozens of tax regimes. The custodian runs that as a factory: predict the income, collect it, reconcile it, and chase whatever does not arrive on time.

**Proxy voting.** As the holder of record (in street name), the custodian receives the meeting notices and the right to vote. It passes the proxy to the beneficial owner, collects voting instructions, and casts the votes on their behalf. For a big index fund holding thousands of companies, this is the machinery behind shareholder democracy — and it has become politically charged, because how the giant managers vote their shares matters enormously to corporate governance.

The "so what" of asset servicing is that it is where custody stops being a commodity. Anyone can hold a book entry. *Servicing* 4,000 holdings across 30 countries flawlessly, every day, with full tax reclaim and zero missed deadlines, is genuinely hard, and it is sticky: once a client's whole operating workflow is wired into a custodian's servicing, ripping it out is a multi-year project. The complexity is the moat.

#### Worked example: a missed corporate-action deadline

Suppose a fund holds **2,000,000 shares** of a company that announces a rights issue: holders may buy **1 new share for every 10 held**, at **\$8.00**, when the stock trades at **\$10.00**. The rights are worth roughly the \$2.00 discount per new share.

The fund is entitled to buy:

$$\frac{2{,}000{,}000}{10} = 200{,}000 \text{ new shares at } \$8.00$$

Each new share is worth about \$2.00 more than its cost, so the rights are worth roughly:

$$200{,}000 \times \$2.00 = \$400{,}000$$

If the custodian notifies the client and the election is made on time, the fund captures (or sells) that \$400,000 of value. If the custodian *misses the deadline* — fails to notify, or fails to process the election — the rights lapse worthless, and the client is out \$400,000. In practice the custodian, having failed in its servicing duty, makes the client whole and eats the loss itself. The intuition: in asset servicing the held assets are never the custodian's risk, but the *operational error* of mishandling them absolutely is — a single missed date can cost more than a year of the custody fee on that account.

## Fund administration: keeping the books for the funds

The next service up the stack is **fund administration** — and it is where the custodian becomes the independent scorekeeper for the funds whose assets it holds. When you buy a mutual fund or an ETF, you trust that its stated price is real: that the \$24.17 "net asset value" is genuinely what one share is worth, calculated honestly, not whatever the manager wishes it were. The fund administrator is the party that makes that trust well-founded.

The central task is **striking the NAV**. The net asset value of a fund is:

$$\text{NAV per share} = \frac{\text{Total value of holdings} + \text{cash} - \text{liabilities}}{\text{shares outstanding}}$$

To strike it, the administrator independently values every holding (using market prices, or for illiquid assets, an agreed valuation method), adds the fund's cash, subtracts its fees and other liabilities, and divides by the number of shares outstanding. For a daily-priced fund this happens *every business day*, after the markets close, on a tight deadline, because the next morning investors will buy and sell at that price. An administrator running thousands of funds is striking thousands of NAVs a night.

Around the NAV sits the rest of fund administration: keeping the fund's general ledger and official books and records; processing **subscriptions and redemptions** (new money coming in, investors cashing out) and issuing or cancelling the corresponding shares; calculating and accruing the management and performance fees; producing the financial statements and the regulatory reports; and increasingly, providing the data the manager needs to run the fund. The administrator is independent of the manager — that independence is the whole point. A manager marking its own homework is a recipe for the kind of fraud that fund administration exists to prevent.

Why does the custodian do this? Because it already holds the assets, so it already knows, authoritatively, what the fund owns. Bolting administration on top of custody is natural: the same firm that safekeeps the holdings is best placed to value them and keep the books. This is why "securities services" arms of the big banks bundle custody, administration, and middle-office services together — they are selling the whole back office of an asset manager as a service.

#### Worked example: striking a fund's NAV

Suppose a fund holds a portfolio currently worth **\$980 million**, plus **\$25 million** of cash, and owes **\$5 million** in accrued fees and expenses. It has **40 million shares** outstanding.

Total net assets:

$$\$980{,}000{,}000 + \$25{,}000{,}000 - \$5{,}000{,}000 = \$1{,}000{,}000{,}000$$

NAV per share:

$$\frac{\$1{,}000{,}000{,}000}{40{,}000{,}000} = \$25.00$$

So tonight the fund's NAV is **\$25.00 per share**, and tomorrow investors buy and redeem at that price. If the portfolio rises 2% to \$999.6 million the next day (cash and fees roughly unchanged), net assets become about \$1,019.6 million and NAV per share rises to about \$25.49. The intuition: NAV is just the fund's net worth divided into shares, struck fresh every day — and the administrator's job is to make sure that single number is correct, independent, and on time, because every investor's buy and sell price depends on it.

## Securities lending: making the held assets earn

So far the custodian holds assets and gets paid a tiny fee. But the assets are just *sitting* there. Securities lending is the business of making them work — and it is where the custody franchise turns into a meaningfully larger revenue line and, occasionally, a source of real loss.

The idea is simple. Someone wants to *borrow* shares — most often a hedge fund or trader who wants to **short-sell** them (sell shares they do not own, betting the price falls, then buy them back cheaper and return them). To short a stock you must first borrow it. Where do the shares to borrow come from? From the giant pools of long-term holders — pension funds, index funds, sovereign wealth funds — who own the stock and are happy to lend it out for extra income, since they were not going to sell it anyway. The custodian, sitting on those holdings, acts as the **lending agent**: it lends the owner's shares to the borrower, takes **collateral** (cash or other securities, worth more than the loaned shares), charges the borrower a fee, and splits that fee with the owner.

Crucially, the owner keeps the economics of the share. While it is on loan, the owner still receives the equivalent of any dividends (the borrower passes them through) and keeps the upside if the price rises. What the owner gives up is the *vote* during the loan, and takes on the risk that the borrower fails to return the shares — which is what the collateral is for. If the borrower defaults, the agent sells the collateral and buys the shares back in the market. As long as the collateral is sufficient and liquid, the owner is protected.

There are two ways the fee is generated, and the distinction matters for understanding 2008. For a *general collateral* stock (easy to borrow, lots of supply), the fee is small and the real money comes from **reinvesting the cash collateral**: the borrower posts cash, the agent invests that cash in short-term instruments, earns a yield, and shares it. For a *special* or *hard-to-borrow* stock (scarce, heavily shorted), the borrowing fee itself is high — sometimes tens of percent annualized — and that fee is the prize. The first model embeds an investment risk (the cash has to be reinvested *somewhere*); the second does not. Remember that, because the first model is exactly what blew up.

![Bar chart splitting one hundred dollars of securities-lending revenue between the asset owner, the agent custodian, and program costs](/imgs/blogs/custody-and-securities-services-the-invisible-banks-that-hold-the-worlds-assets-7.png)

The chart shows how the gross revenue is divided. Of every \$100 of lending revenue generated, the **asset owner** — whose shares these are — typically keeps the lion's share, on the order of **\$75**; the **agent custodian** takes a cut for arranging and managing the program, around **\$15**; and the rest, roughly **\$10**, covers the program's costs, including the buffer against reinvestment and collateral risk. The exact split is negotiated and varies; the principle is that the owner keeps most, because it is the owner's asset doing the earning, and the agent earns a slice for running the machine and standing behind the indemnification it usually provides.

#### Worked example: splitting securities-lending revenue

Suppose a custodian's lending program, across all its lendable assets, generates **\$200 million** of gross lending revenue in a year, split 75 / 15 / 10 as in the chart.

- Asset owners keep: \$200m × 75% = **\$150 million**
- The agent custodian earns: \$200m × 15% = **\$30 million**
- Program costs absorb: \$200m × 10% = **\$20 million**

Now look at it from one owner's side. A pension fund with \$20 billion of lendable assets that earns, say, 4 basis points of net lending income on those assets makes:

$$\$20{,}000{,}000{,}000 \times 0.0004 = \$8{,}000{,}000$$

That \$8 million a year is found money on a portfolio that was going to hold those shares anyway — and for many big holders, securities-lending income covers a meaningful chunk of their entire custody bill. The intuition: securities lending turns a static pile of long-term holdings into a small but real yield, which is why it is one of the few parts of custody where everyone in the chain — owner, agent, borrower — has a reason to keep the machine running.

## The economics: tiny fee, gigantic base, sticky deposits

Now we can see the whole business model, and it is not what a newcomer expects. Custody is **low-margin per dollar, enormous in absolute scale, and remarkably sticky** — three properties that together make it one of banking's best franchises. There are three engines of revenue, and they reinforce each other.

**Engine one: fees on assets and activity.** This is the visible business — basis points on AUC, fees per trade settled, bundled fees for servicing and administration, and the agent's share of securities lending. Each individual fee is small. The power is in the base.

![Bar chart showing assets under custody of the biggest custodians: BNY, State Street, JPMorgan, Citi, BNP Paribas](/imgs/blogs/custody-and-securities-services-the-invisible-banks-that-hold-the-worlds-assets-2.png)

The chart shows the scale that makes the tiny fees add up. **BNY** holds on the order of **\$50 trillion** of assets under custody and administration; **State Street** is close behind in the high \$40-trillions; **JPMorgan's** securities-services arm and **Citi's** are each in the tens of trillions; **BNP Paribas** is the largest European player. (These figures come from company disclosures and industry reporting and are rounded — different firms define "assets under custody/administration" slightly differently, so do not over-read small gaps.) The point is the order of magnitude: a handful of firms hold *hundreds of trillions of dollars* of the world's assets between them. The global pool of assets under custody runs into the hundreds of trillions; a few banks dominate it.

#### Worked example: why a tiny basis-point fee on trillions is a huge business

Take a single custodian holding **\$50 trillion** of assets, and an all-in average fee of just **1 basis point** (0.01%) across custody, settlement, and servicing.

$$\$50{,}000{,}000{,}000{,}000 \times 0.0001 = \$5{,}000{,}000{,}000$$

That is **\$5 billion a year** of recurring revenue from a fee so small that no individual client ever bothers to negotiate it hard.

![Bar chart of annual revenue at different custody fee rates on fifty trillion dollars of assets](/imgs/blogs/custody-and-securities-services-the-invisible-banks-that-hold-the-worlds-assets-5.png)

The chart makes the leverage of scale concrete: on \$50 trillion of assets, **half a basis point** is about \$2.5 billion a year, **1 basis point** is \$5 billion, **2 basis points** is \$10 billion, and **3 basis points** is \$15 billion. Move the average fee by a single basis point and you move the revenue by \$5 billion. This is why custodians fight over scale rather than over price: winning a \$2 trillion mandate adds far more revenue than nudging the fee rate up, and the whole industry is a race to spread enormous fixed-cost processing platforms over an even more enormous asset base. The intuition: in custody, *size is the strategy* — the business has huge fixed costs (the technology, the global network, the operations staff) and a wafer-thin margin per dollar, so the only way to make real money is to be very, very big.

**Engine two: deposits — the hidden profit center.** Here is the part outsiders miss. Custody clients constantly hold *cash* with the custodian: cash waiting to settle a trade, dividends and coupons just collected, cash collateral from securities lending, cash between investment decisions. That cash sits at the custodian as **deposits** — and deposits are the raw material of the core banking machine described across this series. The custodian can invest that cash (in safe, short-term assets) and earn the spread between what it pays the client on the cash and what it earns on the investment. When interest rates are high, this **net interest income** on custody-related deposits can rival or exceed the fee income. State Street and BNY are, in this sense, deposit-funded banks whose deposits happen to come from the custody business. This is the same maturity-and-spread engine that drives [net interest margin](/blog/trading/banking/net-interest-margin-and-the-spread-business-explained) everywhere in banking — it just arrives through the back door of custody.

**Engine three: the rest of securities services.** Foreign-exchange conversion on cross-border income and trades, collateral management, middle-office outsourcing, data and analytics, and increasingly technology platforms sold to asset managers. Each is a fee, each leans on the same held-asset base, and each makes the client stickier.

And sticky is the word. Why does a client almost never leave?

- **Switching is brutal.** Moving \$50 billion of holdings across 30 markets to a new custodian is a project measured in *quarters*, with real operational risk during the transition. The pain vastly exceeds any fee saving.
- **The custodian is wired into everything.** The client's reporting, its compliance, its NAV, its fund accounting, its tax — all run through the custodian. Ripping it out means re-plumbing the whole operation.
- **The fee is already tiny.** When you are paying 1 basis point, there is no fat to squeeze by switching. The price is not the pain point.

The result is a franchise with very long client lifetimes and very predictable revenue — exactly the profile that financial markets prize. The trade-off is that growth is slow and grinding (you grow with the asset markets and by winning the occasional big mandate), and margins are forever thin. It is the opposite of investment banking's feast-or-famine; it is a utility. For a fuller picture of where this sits inside a universal bank's revenue mix, see [inside an investment bank: how they make money](/blog/trading/finance/inside-an-investment-bank-how-they-make-money) — securities services is the steady, boring counterweight to the volatile trading and deal-making lines.

## The risks: capital-light does not mean risk-free

It is tempting to conclude that a business which does not own the assets has no risk. That is exactly wrong. The held assets are off the custodian's risk; the *plumbing around them* is squarely on it. A custodian's risks are different in kind from a lender's — they are mostly **operational** rather than credit — but they are real, and one of them caused billions of dollars of losses in 2008.

![Graph of custodian risks: operational risk leading to a processing failure and payout, and securities lending leading to cash reinvestment and the 2008 loss](/imgs/blogs/custody-and-securities-services-the-invisible-banks-that-hold-the-worlds-assets-8.png)

The diagram traces the two main paths to a loss. Start with **operational risk** — the risk of loss from failed processes, systems, people, or external events. In custody this is the dominant risk, because the entire product is flawless, high-volume processing. A missed corporate-action deadline (as in our earlier example), a botched settlement, a pricing error in a struck NAV, a system outage that stops a day's settlements, a cyber breach of the records — any of these can force the custodian to *make the client whole out of its own pocket*. The held assets are fine; the bank's P&L takes the hit. This is why custodians are obsessive about controls, reconciliation, and redundancy: their margin is so thin that a single large error can erase the profit on a big relationship. (For the full anatomy of this risk class, see [operational risk: fraud, cyber, and the loss events](/blog/trading/banking/operational-risk-fraud-cyber-and-the-loss-events).)

The second path is the dangerous one: **securities-lending cash-collateral reinvestment**. Recall that in the general-collateral model the borrower posts *cash*, and the agent reinvests that cash to earn a yield. Here is the trap. The cash collateral is short-term and can be recalled the moment the borrower returns the shares — effectively overnight money. But to earn a meaningful yield, the temptation is to invest it in instruments that are *slightly* longer-dated or *slightly* riskier than truly overnight, safe paper. That is a maturity-and-credit mismatch hiding inside what is sold as a low-risk program. In normal times it earns a few extra basis points. In a crisis, the value of those reinvestment instruments falls, the borrowers want their cash collateral back, and the pool is worth *less than what is owed*. The gap is a loss — and it lands on the program.

This is precisely what happened in **2008**. Several large securities-lending programs, including those run by major custodians and managers, had reinvested cash collateral in instruments — including some mortgage-related and longer-dated paper — that cratered when credit markets froze. When borrowers returned shares and demanded their cash collateral back, the reinvestment pools could not cover it at par. The result was billions of dollars of losses spread across asset owners and the agents who ran the programs; some agents shared or absorbed losses to preserve client relationships, and lawsuits followed for years. The lesson was burned into the industry: the safe-sounding part of securities lending — reinvesting "overnight" cash — had quietly become a leveraged bet on credit and liquidity. It is the same maturity-transformation trap that defines banking itself, dressed up as a custody service.

There are other risks worth naming. **Sub-custodian risk** is the risk that a local agent in a foreign market fails, is expropriated, or freezes assets — your Russian or other emerging-market holdings are only as safe as the local sub-custodian and the local rule of law. **Concentration and systemic risk** is the flip side of the industry's beauty: because a handful of firms hold so much, the failure or major outage of one would be a systemic event — these custodians are designated systemically important and supervised accordingly. And **conflict-of-interest risk** runs through a universal bank that is custodian, lending agent, fund administrator, *and* a trading counterparty all at once: the firewalls between those roles have to be real.

#### Worked example: the operational error that eats the margin

Recall a custody relationship earning **1 basis point** on **\$10 billion** of assets — **\$1 million a year** of revenue, from the first worked example. Now suppose the custodian misses one voluntary corporate action on that account and has to compensate the client **\$400,000** (the rights value from the servicing example).

$$\text{Net result for the year} = \$1{,}000{,}000 - \$400{,}000 = \$600{,}000$$

A single operational error wiped out **40% of a year's revenue** on that relationship. In a business with a 1-basis-point fee, there is no room for mistakes — one bad day on one account can cost most of a year's earnings on it. The intuition: custody's risk is not that the assets vanish; it is that the razor-thin margin leaves no cushion for operational failure, so the whole business is an exercise in not making errors at scale.

## Common misconceptions

**"A custodian owns or controls the assets it holds."** No — and this is the foundational error. The custodian *holds* the assets; the client *owns* them and (via the manager) *controls* them. Client securities are legally segregated and ring-fenced; they are not the custodian's property and are not available to the custodian's creditors if it fails. The whole legal architecture exists to keep "held" and "owned" completely separate. A custodian with \$50 trillion of AUC could fail without those \$50 trillion being at risk to its creditors.

**"Assets under custody and assets under management are basically the same big number."** They measure entirely different jobs. AUC is what a custodian *holds* and makes no decisions about; AUM is what a manager *decides* on. AUC is typically far larger because one custodian holds many managers' assets. A custodian might report \$50 trillion of AUC while the world's largest *manager* runs around \$10–12 trillion of AUM — and the fee on AUC (about 1 bp) is a tiny fraction of the fee on AUM (20–50 bp or more). Confusing the two makes the whole business model unintelligible.

![Before and after comparison of assets under custody versus assets under management, showing held but not owned versus managed with decisions](/imgs/blogs/custody-and-securities-services-the-invisible-banks-that-hold-the-worlds-assets-6.png)

The figure makes the contrast literal. On the left, **AUC**: a huge number — say \$50 trillion — held but not owned, no decisions made, earning about 1 basis point. On the right, **AUM**: a much smaller number — say \$3 trillion — actively managed, every buy-sell-weight decision made, earning 20 to 50 basis points. Same word "assets," opposite jobs, opposite fee scales. A custodian and a manager are as different as a warehouse operator and the company whose goods it stores.

**"Custody is risk-free because the bank does not own anything."** The held assets are off-risk; the *operations* are not. Operational errors (missed deadlines, settlement fails, pricing mistakes, outages) force the custodian to compensate clients out of its own thin margin, and securities-lending cash reinvestment lost billions in 2008. Capital-light is not risk-light; it just shifts the risk from credit to operations and to the lending program.

**"Settlement is instant — you buy a stock and it is yours immediately."** You *agree* the trade instantly; you do not *settle* it until T+1 (one business day later in the U.S. since May 2024, T+2 in many markets). In that gap the custodian and CSD do the real work of moving securities against cash under delivery-versus-payment. The "instant" feeling in your brokerage app is your broker fronting you the position; the actual transfer of ownership happens later, in the plumbing.

**"This is a backwater — the real banking money is in lending and trading."** In dollar terms securities services is smaller than a giant bank's lending or markets business, but its *quality* is exceptional: recurring, fee-and-deposit, capital-light, low-credit-risk, and extraordinarily sticky. In years when trading blows up or credit losses spike, the custody line keeps grinding out steady earnings. Markets often value a dollar of stable custody revenue more highly than a dollar of volatile trading revenue, precisely because it is predictable.

## How it shows up in real banks

**BNY: the world's largest custodian.** The Bank of New York traces to 1784 and Alexander Hamilton; today, as BNY, it is the largest custodian on earth, with assets under custody and administration on the order of \$50 trillion — a figure that dwarfs its own balance sheet. BNY is the clearest example of the model: a bank whose product is safekeeping and servicing, whose revenue is a blend of basis-point fees and net interest income on the cash its custody clients leave with it, and whose moat is the sheer operational scale and stickiness of holding a meaningful slice of the world's assets. When you read that one bank holds "tens of trillions," BNY is usually the bank in question.

**State Street and the invention of the index-fund back office.** State Street, founded in Boston in 1792, is both a top-two global custodian (high \$40-trillions of AUC/A) and, through State Street Global Advisors, a giant asset *manager* and the issuer of the original ETF, SPY. It is the textbook case of why AUC and AUM live in the same building but are different businesses: the custody arm safekeeps and services trillions for *other* managers, while the asset-management arm runs its own trillions of AUM. The servicing of the index-fund era — striking NAVs for thousands of funds, processing creations and redemptions for ETFs, lending the underlying securities — is in large part a State Street and BNY story.

**The 2008 securities-lending losses.** The cleanest illustration of where the risk really sits. Going into the crisis, securities-lending programs run by custodians and large managers had reinvested clients' *cash collateral* in instruments that were supposed to be safe and short-term but in fact carried credit and liquidity risk — including some mortgage-linked and longer-dated paper. When markets froze in 2008, those reinvestment pools fell below the value of the cash owed back to borrowers. As borrowers returned shares and demanded their collateral, the gap crystallized into losses measured in the billions across the industry. Some agent lenders shared or absorbed losses to protect client relationships; litigation ran for years. The episode rewired how the industry runs cash-collateral reinvestment — tighter guidelines, shorter maturities, more transparency — and stands as the proof that the "safe" plumbing of custody contains the same maturity-transformation risk as the rest of banking. It is the same lesson as the [interest-rate mismatch that sank the savings and loans](/blog/trading/banking/the-savings-and-loan-crisis-interest-rate-mismatch-and-a-thousand-failures): borrowing short and investing long is dangerous wherever it hides.

**Why AUC is not AUM — the BlackRock vs BNY contrast.** BlackRock, the world's largest asset manager, runs on the order of \$10–12 trillion of AUM — it *decides* what that money buys. BNY, the largest custodian, holds around \$50 trillion of AUC — it *holds* assets and decides nothing. BlackRock is, in fact, a *client* of custodians for much of the operational holding of its funds' assets. Two of the biggest numbers in finance, attached to two of the biggest firms, measuring two completely different roles. A reader who keeps them straight understands the division of labor at the center of modern asset management: managers make the calls, custodians hold the assets and run the back office, and the fee scales are an order of magnitude apart.

**The deposit twist in a high-rate world.** In 2022–2023, as central banks raised rates sharply, the custodians' deposit engine roared. The cash that custody clients leave on deposit — settlement balances, collected income, lending collateral — suddenly earned a real return, and net interest income at firms like BNY and State Street jumped. It was a vivid reminder that a custodian is also a deposit-funded bank: its fee business is steady, but its earnings still flex with interest rates through the cash its custody franchise gathers. This is the same engine analyzed in [cash management and transaction banking for corporates](/blog/trading/banking/cash-management-and-transaction-banking-for-corporates) — operational cash balances, gathered cheaply, are a quietly enormous source of bank profit.

**The other side of the trading book.** When a bank's [trading desk makes markets](/blog/trading/banking/the-trading-book-market-making-flow-vs-prop-and-the-volcker-rule), every trade it does has to settle, and the securities have to be held somewhere. The custody and securities-services arm is the unglamorous counterpart to the trading floor: one side takes the price risk and earns the spread; the other side holds, settles, and services the positions for a fee, taking almost no price risk at all. In a universal bank the two sit under one roof, and the steady custody fees help smooth the lumpy trading revenue.

## The takeaway: how to read the invisible banks

Once you see custody, you see the skeleton of the whole asset-management world. Every fund, every pension, every ETF you have ever owned sits on top of a custodian you never chose and never think about. That invisibility is the product working perfectly: a custodian's job is to be so reliable that you never have to wonder who is holding your shares.

Here is how to use this lens.

**When you read a bank's results, separate the AUC line from everything else.** A custody-heavy bank like BNY or State Street should be read as a fee-and-deposit utility, not a lender. Ask three questions: how much AUC, growing at what rate (it tracks the markets plus mandate wins)? What is the fee rate doing (it grinds down over time — pricing power is limited)? And how much is net interest income on custody deposits, which means the "fee business" actually flexes with interest rates? A custodian that looks cheap in a high-rate year may simply be earning a temporary deposit windfall.

**Treat capital-light as a feature *and* a warning.** The franchise is wonderful precisely because it does not need much equity — the assets are the clients'. But that same thinness means there is no cushion for operational failure. The right way to judge a custodian is not by its loan losses (there are almost none) but by its operational track record: missed corporate actions, settlement fails, system outages, and — above all — how it runs securities-lending cash-collateral reinvestment. That last item is where a sleepy custody business can hide a credit-and-liquidity bet that only shows up in a crisis.

**Remember the spine of this series.** A bank is a leveraged, confidence-funded maturity-transformation machine, and it survives only on trust. The custody business looks like the exception — capital-light, not obviously leveraged, no big loan book — but it is in fact the rule in its purest form. The product is *literally* trust: the promise that your assets are safe and correctly recorded. The deposits it gathers are funded short and invested for spread, exactly like any bank's. And the one place it lost real money — 2008 securities-lending reinvestment — was the moment that short-funded, long-invested trade became visible. The invisible banks are not outside the fragile trade at the heart of banking. They are it, stripped down to its essence: hold what others own, earn a sliver, and never, ever break the trust that lets you hold it.

The next time you check your retirement balance, remember that the number is real because somewhere, a custodian struck a NAV last night, settled yesterday's trades against cash, collected this quarter's dividends, and kept the record straight — for a fee so small you will never see it, on a pile of assets so large you can barely picture it.

## Further reading & cross-links

- [The payments business: how money actually moves between banks](/blog/trading/banking/the-payments-business-how-money-actually-moves-between-banks) — the cash side of the settlement plumbing that custody rides on.
- [Cash management and transaction banking for corporates](/blog/trading/banking/cash-management-and-transaction-banking-for-corporates) — the other capital-light, deposit-gathering operational franchise, and a close cousin of securities services.
- [The trading book: market-making, flow vs prop, and the Volcker Rule](/blog/trading/banking/the-trading-book-market-making-flow-vs-prop-and-the-volcker-rule) — the price-taking trading floor whose trades the custody arm holds and settles.
- [Operational risk: fraud, cyber, and the loss events](/blog/trading/banking/operational-risk-fraud-cyber-and-the-loss-events) — the dominant risk class in a custody business, where a processing error eats the margin.
- [Net interest margin and the spread business explained](/blog/trading/banking/net-interest-margin-and-the-spread-business-explained) — the deposit engine that quietly drives custodian earnings.
- [Inside an investment bank: how they make money](/blog/trading/finance/inside-an-investment-bank-how-they-make-money) — where securities services sits in the revenue mix of a universal bank.

*This article is educational, not investment advice.*
