---
title: "Settlement and Custody: Who Actually Holds Your Shares?"
date: "2026-06-21"
publishDate: "2026-06-21"
description: "When your app says you own 1,000 shares, where do those shares physically live, who is the legal owner, and what happens to your claim if every intermediary in the chain fails?"
tags: ["capital-markets", "settlement", "custody", "dvp", "central-securities-depository", "dtc", "euroclear", "post-trade", "beneficial-ownership", "clearing", "market-plumbing"]
category: "trading"
subcategory: "Capital Markets"
author: "Hiep Tran"
featured: true
readTime: 42
---

> [!important]
> **TL;DR** — When your brokerage app says "you own 1,000 shares," you do not hold a certificate and you are almost never the name on the company's register. You hold a *claim on a claim on a claim* — a chain that runs from you, through your broker, through a global custodian and a sub-custodian, down to a central securities depository (CSD) that holds the single master record for the whole market.
>
> - **Settlement finality** is the legal moment ownership becomes irrevocable. Delivery-versus-payment (DvP) ties the share leg and the cash leg together so neither side can pay and get nothing back.
> - **The CSD** (DTC in the US, Euroclear/Clearstream in Europe) immobilised the paper certificates decades ago; today shares are pure *book entries*. The CSD is the legal owner of record; everyone above it owns an entitlement.
> - **Custody is a chain of intermediaries**, and segregation of client assets is the only thing standing between you and a broker's creditors if the broker fails — the Lehman 2008 custody mess is the cautionary tale.
> - **The one number to remember**: US equity settlement went from **T+5 in 1975 to T+1 in May 2024** — a 5× compression of the window in which a trade can fail.

In May 2024, the United States quietly rewired a pipe that almost no investor has ever seen. On the 28th, the standard settlement cycle for US stocks shortened from two business days after the trade (T+2) to one (T+1). There was no ticker-tape celebration, no opening bell. For the retail investor watching a balance update on a phone, nothing visibly changed. But behind that screen, thousands of back-office staff at brokers, custodians, and the central depository spent two years and hundreds of millions of dollars rebuilding processes so that the legal transfer of ownership — the moment "your" shares actually become yours — could happen a full day sooner.

Why would the industry spend that kind of money to shave one day off something invisible? Because the gap between *trade* and *settlement* is the gap in which things go wrong. In that window, your counterparty can default, a broker can fail, a payment can bounce, and the chain of ownership that connects you to the company you "own" can snap. The shorter the window, the smaller the pile of unsettled trades exposed to a shock. The 2021 meme-stock episode — when brokers faced sudden, enormous collateral calls because their clients' trades were still settling two days later — is exactly what T+1 was designed to shrink.

This post is about what happens *at and after settlement*: where your shares physically live, who the law considers the owner, who actually does the unglamorous work of collecting your dividends and counting your proxy votes, and what happens to your claim when an intermediary in the chain goes bankrupt. This is the **plumbing layer** of the capital-markets machine — the part that joins the primary market (which *creates* securities) to the secondary market (which *trades* them). And the punchline of the whole series lives right here: nobody funds a 30-year project unless they can sell their claim on it tomorrow morning, and nobody buys that claim unless they trust that, once they pay, the shares will actually, irrevocably, become theirs. **Trustworthy custody is part of what makes a claim safely sellable.**

![The custody chain from the investor down to the central securities depository](/imgs/blogs/settlement-and-custody-who-actually-holds-your-shares-1.png)

## Foundations: trade, clearing, settlement, and custody

Let us build the vocabulary from zero, because these four words are constantly confused, and the confusion is exactly where people lose money.

Start with an everyday analogy. You agree to buy a used car from a stranger you met online. **Agreeing the price and the car** is the *trade*. **Working out who owes whom what, and meeting at a notary** is *clearing*. **Handing over the cash and the keys at the same instant** is *settlement*. And **where the car sits in your garage afterward, and who keeps the title deed** is *custody*. Each is a distinct step, and the whole point of a market's plumbing is to make the dangerous middle steps boringly safe.

Now the precise definitions:

- A **security** is a tradable claim: a share is a claim on a slice of a company's residual value; a bond is a claim on a stream of payments. (We do *not* re-derive how to value these — fixed-income owns bond pricing, equity-research owns valuing a stock. We own how the claim *moves and is held*.)
- A **trade** is the agreement: at 10:31 a.m. you agreed to buy 1,000 shares of a company at \$100. At that instant you have a *contract*, not the shares. The trade is "executed" but not yet "settled."
- **Clearing** is everything between the trade and settlement: confirming both sides agree on the terms, calculating the net amounts each party owes, and — for most markets — interposing a **central counterparty (CCP)** that becomes the buyer to every seller and the seller to every buyer so that no participant is exposed to any other participant's default. (The CCP mechanism is its own deep dive — see the sibling post on the clearinghouse.)
- **Settlement** is the actual exchange: the cash moves one way, the securities move the other, and ownership legally transfers. The moment that exchange becomes irrevocable is **settlement finality**.
- **Custody** is the holding afterward: where the security sits, who maintains the record that says you own it, and who performs the housekeeping (collecting dividends, processing splits, voting) on your behalf.

The **central securities depository (CSD)** is the institution at the bottom of the whole stack. It maintains *the* master record of who owns each security in a market. In the United States that is the **Depository Trust Company (DTC)**, a subsidiary of DTCC. In Europe the giants are **Euroclear** and **Clearstream**. Decades ago these institutions performed a quiet revolution: they **immobilised** the paper certificates — locked the physical certificates in a vault and stopped moving them — and then **dematerialised** them entirely, so that a "share" became nothing more than an electronic *book entry* in the CSD's ledger. Transferring ownership stopped being a courier carrying paper across Wall Street and became a debit to one account and a credit to another inside a single database.

This is the foundation everything else rests on. Once you understand that the share is a book entry at the CSD, and that you are several layers above that entry, the rest of this post is just tracing the layers and asking, at each one, "what happens if this layer fails?"

## Settlement finality: the legal moment ownership becomes irrevocable

Settlement sounds simple — swap cash for shares — until you ask the awkward question: *who moves first?* If the seller delivers the shares first and then waits for payment, the buyer might never pay. If the buyer pays first and then waits for delivery, the seller might never deliver. This is the oldest problem in trade, and in a market clearing trillions of dollars a day it is not a hypothetical.

The classic disaster is a **free delivery** (also called *free of payment*): one leg of the swap happens without the other being guaranteed. The defining historical scar is the 1990 collapse of Drexel Burnham Lambert and, more broadly, the recognition through the 1974 failure of Germany's Herstatt Bank that you could deliver one leg of a currency trade and never receive the other — "Herstatt risk" entered the language precisely because the two legs were not linked.

The fix is **delivery-versus-payment (DvP)**: the system is engineered so that the transfer of securities happens *if and only if* the transfer of cash happens, simultaneously and atomically. Neither leg can complete alone. Either both legs settle or neither does. In a book-entry CSD this is mechanically elegant: the depository debits the seller's securities and credits the buyer's securities in the same operation that debits the buyer's cash and credits the seller's cash. There is no instant in which one party has given up something and received nothing.

![Delivery-versus-payment links the cash and securities legs so neither settles alone](/imgs/blogs/settlement-and-custody-who-actually-holds-your-shares-3.png)

**Settlement finality** is the precise legal moment at which the transfer can no longer be reversed — not by the parties, not by a court, not even by the bankruptcy administrator of a failed participant. This matters enormously: if finality were fuzzy, a bankrupt firm's administrator could "claw back" settled trades, and every counterparty would have to price in the risk that a completed trade might be undone. Legal frameworks like the EU Settlement Finality Directive exist specifically to make this moment hard and bright, so that once the CSD's book entry flips, it stays flipped.

#### Worked example: DvP preventing a free-delivery loss on a \$1,000,000 trade

You are a pension fund selling 10,000 shares at \$100 each — a \$1,000,000 trade — to a hedge fund. Settlement is T+1.

- **Without DvP (free delivery)**: On settlement day you instruct your custodian to deliver the 10,000 shares to the buyer's account, trusting the cash will follow. The shares move. Then the hedge fund's bank, citing a "technical issue," does not send the \$1,000,000. You have delivered \$1,000,000 of stock and received \$0. You are now an unsecured creditor of a hedge fund, hoping to recover cents on the dollar in a possible bankruptcy. Your loss exposure: **up to \$1,000,000**.
- **With DvP**: Your shares are locked at the CSD pending the matched instruction. The depository will only debit your 10,000 shares *at the same instant* it credits your account with \$1,000,000 from the buyer. If the buyer's cash does not arrive, the transfer simply does not happen — you still hold your shares. Your loss exposure: **\$0** (the trade fails, you re-sell, and you bear only the market move, not the full notional).

The intuition: DvP converts "principal risk" — the risk of losing the entire \$1,000,000 — into mere "replacement-cost risk," the much smaller risk that the price has moved against you while you re-arrange the trade.

### What happens when settlement fails

Settlement does not always succeed. A **settlement fail** happens when one party cannot deliver — the seller does not have the shares (perhaps a short seller who could not borrow them), or the buyer does not have the cash. In a DvP system a fail is *safe* in the sense that nobody loses principal — the trade simply does not complete — but it is not free. The non-failing party is left without the asset or cash it expected, may incur funding costs, and in some markets faces **buy-in** procedures (the exchange forcibly buys the shares in the open market and charges the failing party) or **fail penalties** (Europe's CSDR introduced cash penalties for fails precisely to discipline the plumbing).

This is the deep reason the industry compressed the settlement cycle. Look at the history:

![US equity settlement cycle shortened from five days to one day between 1975 and 2024](/imgs/blogs/settlement-and-custody-who-actually-holds-your-shares-2.png)

In 1975 US equities settled at T+5 — a full week of trades sitting unsettled, exposed. By 1995 it was T+3, by 2017 T+2, and in May 2024 T+1. Every compression shrinks the *stock* of open, unsettled trades, which shrinks the amount of collateral the system must post and the size of the pile exposed if a participant fails mid-cycle. The 2021 GameStop episode crystallised this: brokers faced multibillion-dollar intraday collateral calls from the clearinghouse precisely because volatile trades took two days to settle. Cut the cycle to one day and you roughly halve that exposure.

## The CSD: where the master record actually lives

So if the share is a book entry, *whose* book is it? The CSD's. The central securities depository sits at the bottom of the chain and holds the single, authoritative ledger of ownership for an entire market.

Here is the mind-bending part. In the United States, the overwhelming majority of shares are registered not in your name, not even in your broker's name, but in the name of **Cede & Co.** — a nominee partnership of DTC. When a company like Apple looks at its official shareholder register, it does not see millions of individual investors. It sees, for the bulk of its float, a single line: *Cede & Co., holder of X billion shares*. DTC, through Cede & Co., is the **registered legal owner** of most US stock. You — and your broker, and the custodian — hold *beneficial* entitlements that sit on top of that single legal entry.

This is the endpoint of immobilisation and dematerialisation. Rather than re-registering a certificate every time a share changes hands (slow, expensive, error-prone), the certificate (or its electronic equivalent) never moves at all. It stays parked at the CSD under the nominee name, and trades are settled by adjusting book entries in the accounts the CSD's participants (brokers and custodians) hold *at* the CSD. A trade of a billion shares between two brokers is a debit to one broker's DTC account and a credit to the other's — the underlying registration with the issuer does not change.

### The CSD is the system's single most concentrated point

Because the CSD holds the master record for the whole market, it is the ultimate systemic node. DTC settles trades for essentially the entire US equity and corporate-bond market. Euroclear and Clearstream do the same for vast pools of European and international securities. The 2008-era frozen markets and the operational stresses of the pandemic both underscored that if a CSD's processing were ever disrupted, the entire market's transfer of ownership would freeze. This is why CSDs are among the most heavily regulated, redundant, and conservatively run institutions in finance — they are deliberately *boring*, because the cost of excitement is catastrophic.

#### Worked example: beneficial vs legal ownership of 1,000 shares in street name

You buy 1,000 shares of a company at \$50, a \$50,000 position, through a typical online broker.

- **Legal owner of record**: Cede & Co. (DTC's nominee). The company's register shows those 1,000 shares as part of a giant Cede & Co. block, not under your name.
- **The broker's books**: Your broker holds an entitlement to those 1,000 shares within its omnibus position at DTC, and the broker's records show *you* as the beneficial owner of 1,000 of them.
- **What you actually own**: a *beneficial interest* — a contractual and statutory claim on 1,000 shares' worth of economic rights (price, dividends, voting) — enforced through your broker, which enforces it through DTC. You are the **beneficial owner**; Cede & Co. is the **legal owner**. This arrangement is called holding in **street name**.

The intuition: when you "own" a US stock, you almost certainly do not appear on the company's register at all — you own a chain of enforceable claims that bottoms out at a nominee holding one big pooled certificate. It works fine 99.99% of the time, and the 0.01% is what the rest of this post is about.

### Street name vs direct registration

There is an alternative. The **Direct Registration System (DRS)** lets you be recorded directly on the issuer's books (via its transfer agent) as the registered owner of your shares — no broker nominee, no street name. You hold the shares "directly." The 2021 meme-stock crowd rediscovered DRS and pushed shares out of brokers and onto transfer-agent books partly out of distrust of the intermediary chain.

DRS is real and it does change *who holds your record*. But it is not a magic exit from the system: the shares are still dematerialised book entries (just at the transfer agent rather than in street name), DRS holdings are clunky to trade quickly, and they still rely on the issuer's transfer agent being solvent and competent. DRS trades convenience and liquidity for a shorter ownership chain. For most investors the liquidity of street name wins; for those who prize a direct claim, DRS is the lever.

## The custody chain: a claim on a claim on a claim

Now we assemble the full stack, top to bottom, and this is the single most important diagram in the post (fig 1, above). When you hold a foreign stock through a US broker, the chain typically looks like this:

1. **You** — the *beneficial owner*. You bear the economic risk and reward and you hold a claim.
2. **Your broker** — holds your position in **street name**, in an account on its own books showing you as beneficial owner. The broker itself holds an entitlement one layer down.
3. **The broker's custodian** — often a **global custodian** (BNY, State Street, Citi) that holds securities on behalf of the broker and many other clients.
4. **A sub-custodian** — in each local market, the global custodian appoints a local agent (a **sub-custodian**) that has direct access to that country's CSD. (For a US stock held through a US chain, this layer may collapse; for a Vietnamese or Brazilian stock held by a US investor, it is essential.)
5. **The CSD** — DTC, Euroclear, Clearstream, or the local depository — holds the master book entry. Through its nominee it is the legal owner of record.

Each link is a *claim on the link below*. You have a claim on your broker; your broker has a claim on its custodian; the custodian has a claim on the sub-custodian; the sub-custodian has a claim at the CSD. This is the literal meaning of "you own a claim on a claim on a claim." The actual security — the book entry — exists once, at the bottom. Everything above is contractual entitlement.

### Omnibus vs segregated accounts

At each layer, the intermediary can hold your assets in one of two ways, and the difference is the difference between sleeping soundly and losing your shares.

- An **omnibus account** pools many clients' holdings into a single account at the next layer down. The custodian's account at the CSD might say "Broker X: 5,000,000 shares" without breaking out which client owns which share. Internally, the broker's own records say who owns what — but the layer below sees one big commingled pool.
- A **segregated account** holds your assets separately, identifiable as *yours* all the way down, ring-fenced from other clients and (crucially) from the intermediary's own assets.

Omnibus accounts are cheaper and operationally simpler — that is why they dominate. Segregation costs more but buys protection. The regulatory baseline almost everywhere is that client assets must at least be segregated from the *firm's own* assets (so the firm cannot treat your shares as its own collateral), even if clients are commingled with each other in an omnibus pool.

![Omnibus pooling vs segregated accounts when a broker fails](/imgs/blogs/settlement-and-custody-who-actually-holds-your-shares-5.png)

#### Worked example: an omnibus account commingling 50 clients' shares, and why segregation matters

A broker holds 50 clients in one omnibus account at its custodian. Each client believes they own 1,000 shares of the same stock, so the records *should* show 50,000 shares. Client positions are worth \$50 each, so each client has a \$50,000 stake and the pool should be \$2,500,000.

- **The clean case**: The omnibus pool at the custodian holds exactly 50,000 shares. The broker fails. An administrator reconciles the broker's internal client records against the 50,000 shares in the pool, finds they match, and returns 1,000 shares to each of the 50 clients. Everyone is whole.
- **The shortfall case**: Through error or fraud, the omnibus pool actually holds only 45,000 shares — a 5,000-share, \$250,000 shortfall — while the broker's records still claim 50,000. The broker fails. Now there are 50 clients with claims to 50,000 shares but only 45,000 shares exist. Under commingled treatment, the shortfall is shared **pro-rata**: each client receives 45,000 ÷ 50,000 = 90% of their shares, i.e. **900 shares worth \$45,000**, and becomes an unsecured creditor for the missing \$5,000. Every client eats part of the loss, even clients whose own shares were always present.
- **The segregated case**: If each client's 1,000 shares had been individually segregated and identifiable, the administrator could return each *present* client's shares in full and isolate the shortfall to whichever sub-account it actually came from. Clients whose shares were intact get **1,000 shares each, \$50,000, whole**.

The intuition: in an omnibus pool, one client's (or the broker's) shortfall becomes *everyone's* shortfall; segregation converts a shared, socialised loss into an isolated, identifiable one. You pay a little more for segregation to avoid subsidising a stranger's — or a fraudster's — hole.

## What custodians actually do (it is not just storage)

If you think a custodian is a high-tech vault that just *holds* your shares, you are missing 90% of the job and most of the fee. Holding a book entry is trivial. The real work — what global custodians like **BNY** (the world's largest, with tens of trillions under custody), **State Street**, and **Citi** are paid for — is **asset servicing**: keeping your claim alive and correct as the world acts on the underlying security.

![What a global custodian does: safekeeping, corporate actions, income, and voting](/imgs/blogs/settlement-and-custody-who-actually-holds-your-shares-7.png)

The custodian's job breaks into four buckets:

1. **Safekeeping** — holding the securities at the CSD and maintaining the records that connect them to you. This is the smallest part.
2. **Corporate actions** — when the issuer does something to the security, the custodian must process it correctly across every client. Stock splits, reverse splits, mergers, spin-offs, rights issues, tender offers, name changes. A 4-for-1 split must turn your 1,000 shares into 4,000 across the entire chain, accurately, on the right date. Mandatory actions happen automatically; *elective* actions (do you want cash or stock in this merger? do you exercise these rights?) require the custodian to collect your instruction and act on it before the deadline. Botch a corporate action and clients lose real money.
3. **Income collection** — dividends and coupons. When the company pays a dividend, the cash flows down the chain, and the custodian must collect it from the issuer (via the CSD), apply any withholding tax correctly, and credit the right amount to the right beneficial owner. Cross-border, this includes reclaiming over-withheld foreign tax under treaty — a genuinely valuable, fiddly service.
4. **Proxy voting and reporting** — your shares carry votes. Since you are not on the register (Cede & Co. is), your right to vote must be passed up the chain: the CSD assigns voting entitlement to participants, who pass it to custodians, who pass voting materials to brokers, who pass them to you, and your vote travels back down. The custodian also produces the statements, tax documents, and regulatory reports that let you (and auditors and tax authorities) see what you hold.

#### Worked example: a custodian collecting a \$0.50 dividend on 10,000 shares

You hold 10,000 shares through a custodian. The company declares a dividend of **\$0.50 per share**.

- **Gross dividend**: 10,000 × \$0.50 = **\$5,000**.
- **The mechanics**: On the payment date, the issuer pays the total dividend to the CSD, which credits the participants in proportion to their holdings. Your custodian receives its slice, identifies that 10,000 of those shares belong to you, and credits your account.
- **Withholding (cross-border case)**: Suppose this is a foreign stock with a 15% treaty withholding rate. The custodian withholds 15% (\$750) for the foreign tax authority and credits you the **net \$4,250**, while documenting the \$750 so you can claim a foreign-tax credit at home. If your treaty rate were lower than the default statutory rate, the custodian would file to reclaim the difference — money you would simply lose without a competent custodian.
- **What you see**: a single line in your app — "Dividend: \$4,250" (or \$5,000 domestically) — with none of the four-layer collection visible.

The intuition: the custodian's value is that a \$5,000 dividend on a foreign holding arrives in your account, correctly net of tax, on the right day, without you ever knowing which sub-custodian in which country collected it from which paying agent. That invisibility *is* the product.

## Why the chain matters: failure, segregation, and the cost of long chains

Every link in the custody chain is a counterparty whose failure you must survive. The reason the system is engineered around segregation, finality, and regulation is that intermediaries *do* fail, and the difference between a frightening week and a permanent loss is whether the plumbing held.

### The Lehman 2008 custody mess

The most instructive case is **Lehman Brothers**, September 2008. Lehman's broker-dealer held vast quantities of client assets — and for many prime-brokerage hedge-fund clients, those assets had been **rehypothecated**: Lehman was contractually allowed to re-use client securities as collateral for its own borrowing. When Lehman's UK entity (LBIE) collapsed, clients who thought they owned identifiable, segregated securities discovered their assets had been pledged onward and commingled. Untangling who owned what took the administrators (PwC) *years* of litigation — some claims were not resolved until well into the 2010s. Clients whose assets were properly segregated and not rehypothecated got them back relatively quickly; clients whose assets had been re-used joined the queue of unsecured creditors.

The lesson the entire industry took from Lehman: **segregation is not a formality, it is the whole point.** The legal distinction between "my shares are sitting in an identifiable account that is bankruptcy-remote from the broker" and "my shares are part of a commingled pool the broker could pledge" is the difference between recovery in weeks and recovery in years — or never.

### The cost of long chains

Long custody chains are not free. Each layer adds:

- **Counterparty risk** — one more institution whose failure or fraud you depend on surviving.
- **Operational risk** — one more place a corporate action can be misprocessed, a dividend mis-credited, a vote dropped.
- **Cost** — each layer takes a fee, and cross-border chains stack sub-custodian fees in every market.
- **Opacity** — the deeper the chain, the harder it is for you, a regulator, or an administrator to see what is actually held where. Opacity is exactly what made Lehman's unwind a multi-year forensic exercise.

This is also why netting matters so much in the layer just above custody. The clearinghouse nets down the gross obligations before they ever reach settlement, so the volume that must actually move through the custody chain is a tiny fraction of gross trading:

![Netting collapses gross trade obligations into a small net settlement amount](/imgs/blogs/settlement-and-custody-who-actually-holds-your-shares-4.png)

DTCC reports that multilateral netting eliminates roughly **98%** of the value that would otherwise have to settle: trillions of dollars of gross daily trade obligations net down to tens of billions of actual settlement movement. Less to settle means less exposure sitting in the pipes, fewer movements through the custody chain, and a smaller pile at risk if a participant fails. Netting and short settlement cycles are two levers pulling in the same direction: shrink the open exposure.

### The appeal (and hype) of direct and tokenised holding

If long chains are risky, opaque, and costly, why not hold the asset *directly*? This is the pitch behind **DRS** (covered above) and, more loudly, behind **tokenisation** — putting securities on a blockchain so that ownership is recorded on a shared ledger and transfers settle atomically, peer-to-peer, without the layered intermediary chain.

The appeal is real: atomic DvP on a shared ledger could deliver near-instant settlement finality and a single source of truth, collapsing several custody layers. Projects from major exchanges, the DTCC itself, and central banks are seriously exploring it.

But the hype runs ahead of reality, and it is worth being clear-eyed:

- A blockchain replaces the *technology* of the master record, not the *need* for one. Someone still has to be the authoritative registry — and that someone starts to look a lot like a CSD.
- Custody of the private keys becomes the new custody problem. "Not your keys, not your coins" is just "beneficial vs legal ownership" in new clothing — and most retail holders will hand their keys to a custodian, recreating the chain.
- Legal settlement finality — the hard, court-proof irrevocability that the existing system spent decades building into law — does not automatically attach to a ledger entry. Code is not yet law.
- Asset servicing (corporate actions, tax reclaim, voting) still has to be done by *someone*. A token does not file your treaty reclaim.

The honest framing: tokenisation may genuinely compress the chain and speed finality, and the largest depositories are building it rather than fighting it. But it will not abolish custody — it will move the custody problem from "who holds the certificate" to "who holds the key and runs the registry." The need for a trusted holder of the master record is structural, not technological.

This is the chart that frames the stakes: the sheer *stock* of value sitting in custody at any moment.

![US equity market capitalisation held in custody from 2014 to 2024](/imgs/blogs/settlement-and-custody-who-actually-holds-your-shares-6.png)

US equity market cap roughly doubled from about \$26tn in 2014 to about \$58tn in 2024 — and essentially all of it sits as book entries in the custody chain, the overwhelming majority registered to Cede & Co. The plumbing is holding tens of trillions of dollars of claims on behalf of people who have never thought about it once. Globally the number is far larger still.

![The global stock of securities sitting in custody by market](/imgs/blogs/settlement-and-custody-who-actually-holds-your-shares-8.png)

Global equity market cap of roughly \$115tn and a global bond market of around \$140tn — a quarter of a quadrillion dollars of claims — all of it parked, somewhere, in a CSD, under a nominee, serviced by a custodian, beneath layers of brokers, on behalf of beneficial owners. That is the scale of the thing the plumbing quietly holds.

## Common misconceptions

**"I own the actual shares — they're in my account."** No. In street name (the default for almost everyone), the *legal* owner of record is Cede & Co. (DTC's nominee). You own a *beneficial entitlement* — a claim enforced through your broker. Your "account" is a record on your broker's books, not a vault with your name on certificates. This is fine — it is how the whole market works — but it means your protection comes from segregation and regulation, not from physical possession.

**"A settlement fail means I lost my money."** Usually not. Under DvP, a fail means the trade *didn't complete* — nobody handed over value and got nothing back. You still hold your shares or your cash. A fail costs funding, time, and possibly a penalty or buy-in, but DvP specifically prevents the catastrophic principal loss of a free delivery. The danger is in *non-DvP* settings, not in DvP fails.

**"My broker holds my shares, so my broker can spend them."** Only if the rules are broken. Client-asset segregation rules require your securities to be held separately from the broker's own assets, so they are not available to the broker's general creditors if it fails. The Lehman lesson is that *rehypothecation* (re-use by agreement, common in prime brokerage) and *commingling* are where this protection erodes — read your custody agreement to know whether your assets can be re-used.

**"Tokenisation will get rid of custodians and middlemen."** It will likely *compress* the chain and speed settlement, but it cannot abolish the need for an authoritative registry and a trusted holder of keys. The custody problem moves; it does not vanish. "Not your keys, not your coins" is the same beneficial-vs-legal distinction in new packaging — and most people will still delegate to a custodian.

**"T+1 means my trade settles instantly."** No. T+1 means settlement happens *one business day* after the trade, not the same second. It is faster than T+2, which shrinks risk, but it is still a delay during which a fail or a default can occur. True instant ("atomic") settlement is what tokenisation aspires to and what the existing system has deliberately *not* done — because instant settlement removes the netting window that eliminates 98% of settlement volume.

## How it shows up in real markets

**The May 2024 US move to T+1.** On 28 May 2024 the US (with Canada and Mexico) moved equities, corporate bonds, and many other instruments to T+1. The driver was risk reduction — the 2021 meme-stock collateral spikes had shown how dangerous a two-day pile of unsettled volatile trades could be. The cost was operational: every broker, custodian, and the CSD had to compress affirmation, allocation, and funding into hours instead of a day, and cross-border investors faced a funding mismatch because FX still settled at T+2. The episode is a perfect illustration of the spine of this series — the plumbing exists to make claims safely transferable, and the industry will spend hundreds of millions to make that transfer marginally safer.

**Lehman Brothers, September 2008.** As above: the collapse turned the abstract question "is my custody segregated and un-rehypothecated?" into a multi-year, multibillion-dollar forensic recovery. It is the reference case for why segregation and bankruptcy-remoteness are load-bearing, not box-ticking.

**MF Global, 2011.** A brokerage that improperly used segregated *customer* funds to cover its own proprietary bets, leaving a roughly \$1.6bn shortfall in customer accounts. Customers eventually recovered most of it after litigation, but the episode is the clearest modern example of segregation rules being *violated*: the protection only works if the firm actually keeps client assets separate, and supervision is what enforces that.

**Cross-border custody in emerging markets.** When a US fund buys Vietnamese or Brazilian stock, the chain lengthens: US broker → global custodian → local sub-custodian → local CSD. Each market's rules on foreign ownership, settlement cycle, and account structure differ, and the sub-custodian's competence becomes a real source of return and risk. (For how foreign flows actually move a single market, see the Vietnam foreign-flows post linked below.)

## The takeaway: custody is the trust that makes a claim sellable

Step back and look at what the plumbing is *for*. The whole capital-markets machine rests on a promise: that when you buy a claim, you really will own it, irrevocably, and you will be able to sell it tomorrow to someone who trusts the same promise. Settlement finality makes the *moment of ownership* hard and bright. The CSD makes the *record of ownership* singular and authoritative. Custody and segregation make the *holding of ownership* survive the failure of any single intermediary. Asset servicing keeps the claim *alive* — dividends collected, votes counted, corporate actions applied — so that the thing you own keeps being worth owning.

Strip any of these out and the secondary market seizes. If settlement might be reversed, no one trusts a completed trade. If the master record is ambiguous, no one knows what they own. If custody is not segregated, every broker failure becomes a clients' loss and people stop holding securities through brokers. And — here is the spine of the series — if the secondary market seizes, the primary market dies with it, because **nobody funds a 30-year project unless they can sell their claim on it tomorrow morning, safely.** The unglamorous, invisible plumbing of settlement and custody is precisely what makes "sell it tomorrow, safely" true.

So the next time your app shows "1,000 shares," read it correctly: you hold a beneficial claim, enforced through a broker, against a custodian, against a sub-custodian, against a book entry at a depository, registered to a nominee. It is a long chain — and the genius of the system is that, almost all of the time, you never have to think about a single link in it. The day you *do* have to think about it is the day a link breaks, and on that day the only thing standing between you and your shares is whether the plumbing was built right.

## Further reading & cross-links

- [What happens after the trade: the post-trade lifecycle](/blog/trading/capital-markets/what-happens-after-the-trade-the-post-trade-lifecycle) — the full trade → clear → settle sequence this post picks up at settlement.
- [The clearinghouse: how a CCP removes counterparty risk](/blog/trading/capital-markets/the-clearinghouse-how-a-ccp-removes-counterparty-risk) — the layer above custody that nets obligations and novates trades.
- [Securities lending and repo: the financing plumbing](/blog/trading/capital-markets/securities-lending-and-repo-the-financing-plumbing) — where rehypothecation and the re-use of custodied assets live.
- [Stock exchanges and clearinghouses](/blog/trading/finance/stock-exchanges-and-clearinghouses) — how the trading venue and the clearing layer fit together.
- [Inside an investment bank: how they make money](/blog/trading/finance/inside-an-investment-bank-how-they-make-money) — the intermediaries that sit in this chain and how they are paid.
- [Foreign flows, ETFs, and the index effect in Vietnam](/blog/trading/vietnam-stocks/foreign-flows-etfs-and-the-index-effect-vietnam) — what cross-border custody chains look like at the far end.
