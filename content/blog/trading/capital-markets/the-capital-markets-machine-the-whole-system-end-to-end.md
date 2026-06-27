---
title: "The Capital-Markets Machine: The Whole System, End to End"
date: "2026-06-27"
publishDate: "2026-06-27"
description: "Follow one dollar of household saving through issuance, trading, clearing, settlement, custody, financing, and regulation — and see why the whole machine runs on one thing: a claim you can sell tomorrow."
tags: ["capital-markets", "primary-market", "secondary-market", "clearing-settlement", "liquidity", "cost-of-capital", "securitization", "regulation", "vietnam", "market-structure"]
category: "trading"
subcategory: "Capital Markets"
author: "Hiep Tran"
featured: true
readTime: 42
---

> [!important]
> **TL;DR** — A capital market is a machine that turns idle savings into long-term investment, and the whole machine runs on one quiet promise: that the claim you buy today, you can sell tomorrow.
>
> - A dollar of saving travels through a fund, an underwriter, an exchange, a clearinghouse, a custodian, and a regulator — and still arrives at a company that builds something real, leaving you holding a tradable claim.
> - The secret of the entire system is a feedback loop: deep secondary-market trading is what lets the primary market issue cheaply, because nobody funds a 30-year project unless they can exit by lunchtime.
> - Every famous market disaster — 2008, a clearing-member default, a liquidity freeze, a fraud — is the *same* failure wearing different clothes: the loss of trust that a claim stays sellable.
> - The one number to keep: a liquid secondary market can shave **2–3 percentage points** off an issuer's cost of capital. That gap is the entire reason these markets exist.

On the morning of 19 September 2024, the US equity market quietly did something it had not done since the 1920s: it began settling trades in a single business day. The change — moving from "T+2" to "T+1," from two days to one — sounds like a plumbing footnote. It was. Trillions of dollars of stock changed hands that day exactly as they had the day before. No headline crash, no parade. And yet that boring switch is the perfect doorway into the most important idea in all of finance, because it is a story about **the time between a handshake and a claim becoming real money** — and that gap, that interval of trust, is the thing the entire capital-markets machine is built to manage.

We have spent forty-one posts taking that machine apart. We watched a company climb [the financing ladder](/blog/trading/capital-markets/the-financing-ladder-from-bootstrap-to-public-markets) and then go public through [an end-to-end IPO](/blog/trading/capital-markets/the-ipo-process-end-to-end-from-mandate-to-first-trade). We sat inside [an exchange's matching engine](/blog/trading/capital-markets/inside-an-exchange-the-matching-engine-and-the-order-book), followed [a trade through clearing and settlement](/blog/trading/capital-markets/what-happens-after-the-trade-the-post-trade-lifecycle), and watched [a clearinghouse absorb a default](/blog/trading/capital-markets/margin-and-the-default-waterfall-how-a-ccp-survives-a-blowup). We took apart [securitization](/blog/trading/capital-markets/securitization-from-first-principles-turning-loans-into-bonds) and watched [the machine break in 2008](/blog/trading/capital-markets/2008-when-the-securitization-machine-broke-case-study). This post does the opposite of all of those. It puts the machine back together and runs it — by following one dollar, from a household savings account all the way out the far side and back again.

![The capital-markets machine from savings to primary market to secondary market and back via the liquidity loop](/imgs/blogs/the-capital-markets-machine-the-whole-system-end-to-end-1.png)

This is the master diagram of the whole series. Read it once now and it will probably look like a tangle; by the end of this post every box and arrow should feel obvious. The trick to reading it is to notice that it is not a line — it is a *loop*. Money flows out from savers into new investment, and a signal flows back the other way (price and liquidity) that tells the next round of capital where to go and how much it should cost.

Why does any of this need to exist? Because the economy has a timing problem. The people with spare money (savers) want it back soon, safely, and with the option to change their minds. The people who need money (companies building factories, governments building roads) need it for decades and cannot promise to return it on demand. Those two desires are flatly incompatible — and yet the long-term projects are exactly what make a society richer. The capital market is the institution that resolves the contradiction. It lets the saver keep her short-term flexibility (she can sell her claim any morning) while the company keeps the money for thirty years (because when she sells, she sells to someone *else*, not back to the company). The claim changes hands; the capital stays put. That sleight of hand — long-term funding built out of short-term-minded savers — is the deepest trick in all of finance, and the secondary market is the stage on which it is performed.

## Foundations: the five jobs every capital market must do

Before we follow the dollar, let's define the machine from zero, because the whole rest of the post leans on five jobs that any capital market — New York, London, or Ho Chi Minh City — has to perform. If you forget every other piece of jargon in this series, keep these five.

**Job one: create a claim.** A company or a government needs money for longer than any single saver wants to lend it. The market's first job is to manufacture a *security* — a standardised, transferable, legally enforceable claim on future cash flows. A share is a claim on a slice of profits forever; a bond is a claim on fixed interest then your money back. The genius is not the claim itself but its *standardisation*: every share of one company is identical and interchangeable, so it can trade without anyone re-reading the contract. We pulled this apart in [what a security actually is](/blog/trading/capital-markets/what-a-security-actually-is-claims-you-can-sell). Creating and selling brand-new claims is the **primary market**.

**Job two: let the claim trade.** Once a claim exists, its owner must be able to sell it to someone else without bothering the company that issued it. That second-hand market — where existing securities change hands and no new money reaches the issuer — is the **secondary market**. It feels secondary, hence the name, but you will see it is the engine that powers everything.

**Job three: settle the trade.** When two strangers agree a price, somebody has to make sure the buyer's cash actually arrives and the seller's shares actually move, even though they have never met and may never trust each other. That is the job of the *plumbing*: clearing, settlement, and custody, run through a [central counterparty (CCP)](/blog/trading/capital-markets/the-clearinghouse-how-a-ccp-removes-counterparty-risk) and a depository.

**Job four: connect the savers to the issuers.** Households do not call companies directly. A chain of **intermediaries** — funds, brokers, investment banks, exchanges — sits in the middle, each doing one specialised thing and each taking a small toll. We mapped them in [the players](/blog/trading/capital-markets/the-players-savers-issuers-and-the-middlemen-in-between).

**Job five: keep it honest.** None of the first four jobs work if people lie about what a claim is worth. So the whole thing is wrapped in **disclosure-based regulation** — force issuers to tell the truth, punish those who don't, and let buyers decide. That is the entire philosophy behind [the securities acts](/blog/trading/capital-markets/why-markets-are-regulated-disclosure-and-the-securities-acts).

Five jobs: create, trade, settle, connect, police. Notice that the first two are about *what the market produces* (claims, and a place to trade them), the next two are about *who does the work* (the plumbing and the people), and the last is about *what keeps it all standing* (trust). A market that does only the first job — creating claims nobody can resell — is not a capital market at all; it is a graveyard of paper. A market that does the first two but botches settlement is a casino where you can win the bet and still not get paid. The five jobs are a checklist, and a market is only as strong as its weakest one.

There is one more thing to fix in your head before we start. People often assume the bank is the center of this story, because banks are where most of us keep our money. But a capital market is precisely the system that lets savers *bypass* the bank and lend or invest directly into the economy through tradable claims. A bank takes your deposit and decides where it goes; a capital market hands you a menu of claims and lets the price — set by millions of other savers — decide. That difference, between a queue at a bank and an open auction of claims, is why developed economies run most of their long-term financing through markets rather than bank loans. We contrasted the two channels in [debt vs equity](/blog/trading/capital-markets/debt-vs-equity-the-two-ways-to-raise-capital): a bank loan and a bond are cousins, but a bond is a *claim you can sell*, and that single property changes everything downstream.

Now watch a single dollar pass through all five jobs.

## Following one dollar: from a savings account to a steel mill

Meet our saver. Call her Mai. She earns a salary, spends most of it, and at the end of the month has \$1,000 left over. She does not want it sitting in a checking account losing value to inflation, and she has no idea how to evaluate a company's balance sheet. So she does what most savers do: she hands the problem to someone whose job is to solve it.

### Step 1 — the dollar enters a buy-side fund

Mai puts her \$1,000 into a low-cost index fund inside her retirement account. She is now, whether she thinks about it this way or not, a **capital provider**. The fund is part of the **buy-side** — the asset managers, pension funds, insurers, and sovereign wealth funds who actually own the market, which we covered in [the buy-side](/blog/trading/capital-markets/the-buy-side-who-actually-owns-the-market). Households like Mai still directly or indirectly own the largest single slice of US equities.

The fund pools Mai's \$1,000 with millions of other savers' dollars. That pooling is the first piece of magic: alone, Mai could never buy a diversified basket of 500 companies or get an allocation in a new bond deal. Pooled, she effectively can. The fund charges her a small management fee for the service — for a cheap index fund, around 0.05% a year.

Why does Mai need the fund at all? Three reasons, each one a job the fund does that she cannot do alone. First, **scale**: a single \$1,000 cannot be diversified across hundreds of names, but \$50 billion can, and the fund slices that diversification back to her in proportion to her stake. Second, **access**: when Truong Steel sells a new bond, the underwriters allocate it to a short list of large, known buyers — they will not take a \$1,000 order from a stranger, but they will take a \$500M order from a pension fund. Third, **operational machinery**: somebody has to actually hold the securities, collect the coupons and dividends, vote the shares, and handle the settlement of every trade. The fund has the custody relationships and back office to do that; Mai does not. These are not glamorous services, but they are the difference between savings sitting dead in a checking account and savings working in the economy.

The buy-side is enormous and varied, and the *type* of saver behind the dollar shapes how it behaves in the market. A pension fund investing on a 30-year horizon is a patient, stabilising holder; a hedge fund running leveraged positions is a fast, sometimes destabilising one; an insurer matching long-dated liabilities is a natural buyer of long bonds. When you read that "the market sold off," what actually happened is that some mix of these holders changed their minds at once. Mai, through her index fund, is one of the patient ones — and patient capital is exactly what lets companies fund 30-year projects.

![One saver's dollar passing through fund, underwriter, and exchange fees to reach a company](/imgs/blogs/the-capital-markets-machine-the-whole-system-end-to-end-2.png)

The figure tracks what each layer of the machine actually takes out of Mai's dollar. The striking thing is how *little* each toll is in a competitive market — and how the bulk of the dollar still arrives at a company that does something real with it. Let's make that concrete.

#### Worked example: what the fund layer costs Mai

Mai invests \$1,000 in an index fund charging 0.05% per year. The annual fee is:

\$1,000 × 0.0005 = \$0.50 per year.

Over a 30-year retirement horizon, if her money roughly triples to \$3,000, she pays on the order of \$0.50 to \$1.50 per year, averaging perhaps \$25 in total fees across three decades. Compare that to an old-style actively managed fund charging 1.5%: \$15 in year one alone, and well over \$1,000 cumulatively. **The intermediary layer is cheap only because the secondary market made indexing possible — and that competition is itself a product of the machine we're describing.**

### Step 2 — the dollar buys a new issue in the primary market

Now suppose a real company — call it Truong Steel — wants to build a \$500 million mill. It cannot borrow that from a bank cheaply for twenty years, so it taps the **primary market**. It hires an investment bank's [equity or debt capital markets desk](/blog/trading/capital-markets/inside-an-investment-bank-ecm-dcm-ma-and-trading) to run an issue — say a bond, sold through [a syndicated deal](/blog/trading/capital-markets/how-a-bond-is-issued-auctions-syndication-and-the-deal). The bank's underwriters build a book of buyers, and Mai's index fund — needing bonds to balance its portfolio — puts in an order. A sliver of Mai's \$1,000 ends up funding that order.

This is the moment the machine does its actual job. Mai's idle savings, pooled and channeled, become *new capital* in a company's bank account. Truong Steel takes the cash and builds the mill: steel beams, wages, a real productive asset. In exchange, the company issues a **claim** — a bond — that flows back to Mai's fund. We covered the mechanics of who takes the risk in this step in [underwriting and the syndicate](/blog/trading/capital-markets/underwriting-and-the-syndicate-who-takes-the-risk).

It is worth dwelling on what the underwriter actually does for its fee, because beginners often see the bank as a pure toll-collector. In a *firm-commitment* deal — the standard for a serious IPO or bond — the underwriting bank does not merely introduce buyers; it *buys the whole issue itself* at an agreed price and then resells it. For a few hours or days, the bank owns \$500M of Truong Steel and bears the risk that demand evaporates before it can place the paper. The spread is the price of that risk transfer plus the distribution muscle: a syndicate of banks with relationships to hundreds of institutional buyers, a research analyst who will cover the stock, and a trading desk that will make a market in it afterward. The fee looks large in dollars and small in basis points, and both are true. What the company is really buying is *certainty*: it walks away with a known amount of cash, and the bank keeps the risk that the market turned.

There is a second subtlety here that connects directly to the feedback loop we'll reach shortly. The underwriter does not price the new issue in a vacuum. It looks at where *comparable* securities trade in the secondary market — other steel companies' bonds, the broader credit market, the yield curve — and prices the new deal at a small concession to those levels so it will clear. In other words, the primary market price is *anchored to* secondary-market prices. Without an active secondary market in similar claims, the underwriter would be pricing blind, and a blind price is an expensive price. The two engines are bolted together at exactly this joint.

#### Worked example: one dollar inside a \$500M IPO

Suppose instead Truong Steel does a \$500M equity IPO. The bank charges a gross underwriting spread of, say, 4% — typical for a mid-size IPO. Of the \$500M raised from investors:

- Underwriting spread: \$500M × 0.04 = **\$20M** to the syndicate.
- Net to the company: \$500M − \$20M = **\$480M** reaches Truong Steel.

If Mai's fund bought \$1,000 of the deal, \$40 of her money paid the bankers and \$960 reached the company. Now scale it down to her single dollar: about **\$0.04** went to the underwriter and **\$0.96** built the mill. **The primary market is the one place in the whole machine where savings literally turn into bricks — everything else just keeps that claim tradable.**

### Step 3 — the claim trades in the secondary market

Here is the part that confuses most beginners. Once Mai's fund owns the bond (or share), and Truong Steel has its cash, *the company is done*. If the fund later sells that bond to a hedge fund, Truong Steel gets nothing — not a cent. That sale happens in the **secondary market**, and no new money reaches the issuer.

So why does it matter? Because the *only reason Mai's fund was willing to buy the new issue in the first place* is that it knew it could sell later. That is the entire secret of the machine, and it deserves its own figure.

![Why a liquid secondary market lowers the issuer's cost of capital](/imgs/blogs/the-capital-markets-machine-the-whole-system-end-to-end-4.png)

The claim now lives inside the secondary market's machinery: [the exchange's order book](/blog/trading/capital-markets/inside-an-exchange-the-matching-engine-and-the-order-book), where buyers and sellers post bids and offers; [market makers](/blog/trading/capital-markets/market-makers-and-the-spread-who-provides-liquidity) who quote both sides and earn the spread; [the fragmented tape](/blog/trading/capital-markets/lit-markets-dark-pools-and-the-fragmented-tape) of lit exchanges and dark pools where trades actually print. Every time the claim trades, [a price is discovered](/blog/trading/capital-markets/how-a-price-is-made-discovery-arbitrage-and-efficiency) — and that price is a signal that flows back to the primary market, telling the *next* steel company whether building a mill is worth it.

What does "liquidity" actually mean here, concretely? It is the ability to convert a claim back into cash quickly, in size, without moving the price against yourself. A mega-cap stock like Apple has a bid-ask spread of about one basis point — you can sell millions of dollars of it and lose almost nothing to the spread. A micro-cap might have a spread of 80 basis points, and trying to sell a large position would push the price down further still. That gradient — from a one-cent toll on a giant to a punishing haircut on a tiny name — is the visible price of liquidity, and it is set entirely in the secondary market. Market makers are the ones quoting those spreads; they earn the spread as their pay and bear the risk that they get picked off by someone who knows more than they do (the "adverse selection" problem that the [quant order-book and market-making models](/blog/trading/quantitative-finance/market-making-simulator-quant-research) formalise). The spread is not a tax — it is the wage of the people who guarantee that a buyer is always there.

Here is the mental flip that makes the whole machine click. A saver does not actually care about owning a claim forever. What she cares about is being *able* to sell it whenever she wants. The secondary market provides that option to everyone simultaneously — and like any option, it has value even when it is never exercised. Mai may hold her bond to maturity and never sell a share of it, yet she still benefits every single day from the fact that she *could*. That standing option to exit is what she is really buying, and it is what makes her willing to fund a 30-year mill in the first place.

#### Worked example: the cost-of-capital prize of liquidity

Take two identical companies issuing identical 10-year bonds. The only difference: company A's bonds will trade in a deep, liquid secondary market; company B's will be almost impossible to resell.

Buyers of B's bond know they are stuck. To compensate, they demand an extra **liquidity premium** — say 2.5 percentage points. So:

- Company A issues at a yield of **5.0%**. On \$500M of debt, annual interest = \$500M × 0.050 = **\$25M/year**.
- Company B issues at a yield of **7.5%**. Annual interest = \$500M × 0.075 = **\$37.5M/year**.

The difference is **\$12.5M every year**, purely because A's buyers can sell tomorrow and B's cannot. Over the 10-year life of the bond, that is \$125M — a quarter of the entire amount raised — saved by nothing more than secondary-market liquidity. **This is why the secondary market is not "secondary" at all: it is the thing that makes the primary market affordable.**

### Step 4 — every trade clears, settles, and is custodied

When Mai's fund eventually sells the bond, a chain of invisible events fires. The trade is captured, matched, and sent for **clearing**, where [a central counterparty steps between buyer and seller](/blog/trading/capital-markets/the-clearinghouse-how-a-ccp-removes-counterparty-risk) so neither can default on the other. Then it **settles**: cash moves one way, the security moves the other, simultaneously, through a [central securities depository and custodian](/blog/trading/capital-markets/settlement-and-custody-who-actually-holds-your-shares) — which is where the T+1 switch we opened with lives.

The plumbing's quiet superpower is **netting**. The CCP does not move every gross trade; it nets everyone's buys against their sells and moves only the small leftover.

![CCP netting compresses gross trade obligations down to a tiny net settlement amount](/imgs/blogs/the-capital-markets-machine-the-whole-system-end-to-end-6.png)

#### Worked example: netting compresses the day's trades

Suppose in one day across the market there are \$2,000bn of gross trade obligations — every buy and sell added up. A clearinghouse like the NSCC nets these multilaterally. DTCC reports netting efficiency of roughly **98%**. So:

- Gross obligations: **\$2,000bn**.
- Settled after netting: \$2,000bn × (1 − 0.98) = **\$40bn**.

Only \$40bn of cash and securities actually has to move to settle \$2,000bn of trading. **Netting is why the financial system can settle a tidal wave of trades with a trickle of actual money movement — and why a settlement failure, when it happens, is so dangerous: the trickle assumes everyone is good for the rest.**

The CCP performs a second trick that is even more important than netting: **novation**. The instant a trade is cleared, the CCP legally inserts itself between the buyer and the seller, becoming the buyer to every seller and the seller to every buyer. After novation, Mai's fund no longer faces the hedge fund that bought its bond — it faces the clearinghouse. This means it no longer has to ask "is the hedge fund good for the money?"; it only has to trust the CCP. That is a profound simplification. Instead of every participant assessing the creditworthiness of every counterparty (an impossible web), everyone trusts one central, heavily-capitalised, heavily-regulated entity. The CCP collects margin from both sides and stands behind a [default waterfall](/blog/trading/capital-markets/margin-and-the-default-waterfall-how-a-ccp-survives-a-blowup) of resources so that even if a big member fails, the trades still settle. The whole arrangement converts a tangle of bilateral trust relationships into a single hub-and-spoke one — and that conversion is one of the great risk-management inventions of modern finance.

And the shorter that settlement window, the less time anything can go wrong inside it. That is the whole point of the long march from T+5 to T+1. Each shortening of the cycle is a deliberate reduction in *settlement risk* — the risk that between agreeing a trade and completing it, one side fails and leaves the other holding a broken half-transaction. A five-day window in 1975 meant a week of exposure on every trade; a one-day window in 2024 means a single overnight. Multiply that across the trillions in daily volume and the margin the CCP must hold against that risk falls accordingly, freeing capital across the whole system. The cost of the shorter cycle lands on anyone who needs time to arrange funding — notably foreign investors who must convert currency and move cash across time zones before they can pay. That tension, between shrinking risk and squeezing foreign liquidity, is exactly the knife-edge a young market like Vietnam has to walk, and we will see it bite below.

![The US settlement cycle shortening from five days to one day over fifty years](/imgs/blogs/the-capital-markets-machine-the-whole-system-end-to-end-5.png)

### Step 5 — the claim is financed in repo and securities lending

Mai's bond does not just sit in custody. The fund can lend it out for a fee through the **securities-lending** market, or pledge it as collateral to borrow cash overnight in the **repo** market — the hidden funding layer we covered in [securities lending and repo](/blog/trading/capital-markets/securities-lending-and-repo-the-financing-plumbing). This is how short sellers borrow stock and how dealers finance their inventory. It squeezes extra return out of an otherwise dormant claim — and, as we'll see, it is also where liquidity freezes start.

This layer is invisible to most savers but it is enormous, and it is where the machine's leverage lives. A dealer who wants to make markets in Truong Steel's bond does not tie up its own cash to hold inventory; it buys the bond and immediately repos it out — pledges it overnight in exchange for cash, agreeing to buy it back tomorrow at a tiny premium. The cash funds the next purchase. In effect the same claim supports a chain of financing, and that chain is what lets dealers carry the inventory that *makes* the secondary market liquid. The plumbing of financing and the liquidity of trading are the same phenomenon viewed from two angles. The catch — and it is the catch behind half the crises in this series — is that this funding is *overnight*. It has to be rolled over every single day, and the day lenders refuse to roll it, the whole chain unwinds at once.

#### Worked example: how repo finances a market maker's inventory

A dealer buys \$100M of bonds to make markets. Instead of using \$100M of its own capital, it repos the bonds out at a 2% "haircut":

- Cash borrowed against the bonds: \$100M × (1 − 0.02) = **\$98M**.
- Dealer's own capital tied up: \$100M − \$98M = **\$2M**.

With \$2M of capital the dealer carries \$100M of inventory — 50× leverage on that position. **This is why the secondary market is so liquid in good times and so fragile in bad ones: the same repo plumbing that lets a dealer hold 50× its capital in inventory forces it to dump that inventory the instant lenders raise the haircut.**

### Step 6 — all of it is policed by disclosure and integrity rules

Wrapping every step is the regulatory layer. Truong Steel had to file a prospectus and ongoing disclosures so Mai's fund could value the claim honestly — [the disclosure regime](/blog/trading/capital-markets/disclosure-the-prospectus-filings-and-insider-trading). The secondary market is surveilled for [manipulation and spoofing](/blog/trading/capital-markets/market-integrity-manipulation-spoofing-and-circuit-breakers), with circuit breakers to halt a panic. None of this makes anyone money directly. It does something more important: it keeps the *trust* that makes a claim worth buying in the first place.

There is a deeper point hiding in the regulation step. Every other layer of the machine handles a *physical* problem — moving cash, moving securities, matching orders. Regulation handles an *informational* one. A claim is only worth what its future cash flows are worth, and those cash flows are in the future, which means nobody can verify them today. The buyer is always, fundamentally, trusting the issuer's word about what it owns and owes. Disclosure law is the machinery that makes that word *credible* to a stranger: standardised filings, audited numbers, criminal penalties for lying, and a regulator with subpoena power. Where that machinery is strong, strangers will buy claims from strangers at a fair price. Where it is weak, every transaction carries a "maybe they're lying" discount, and that discount is just the cost of capital climbing back up. Disclosure is not separate from the cost-of-capital story — it *is* the cost-of-capital story, told from the side of information instead of liquidity.

#### Worked example: where Mai's whole dollar ends up

Tally the tolls on Mai's \$1.00 across the full machine, using the figures above:

- Fund fee: about **\$0.005 per year** (0.5 cent), and she keeps the rest invested.
- Underwriter spread on the new issue: **\$0.04** once, at issuance.
- Exchange and clearing fees per trade: a fraction of a cent, call it **\$0.0001**.
- Net reaching the company to build something real: roughly **\$0.955** of every dollar.

So of every dollar Mai saves, about **95.5 cents** funds an actual productive asset, and the machine's entire army of intermediaries splits roughly **4.5 cents** for making that possible — and keeping her claim liquid, settled, and honest for as long as she holds it. **A machine that delivers 95 cents of every saved dollar to real investment, while guaranteeing the saver can exit any morning, is not overhead — it is one of the most efficient pieces of social technology ever built.**

That is the dollar's full journey: in through a fund, out into a real asset, back as a tradable claim, kept liquid by the secondary market, settled by the plumbing, financed in repo, and policed by disclosure. Now let's zoom out to the loop itself.

## The feedback loop: how liquidity loops back to cheapen capital

Most people describe a capital market as a one-way pipe: savers' money flows to companies, the end. The truth is a loop, and the loop is the whole reason the system is so powerful.

Trace it in two directions. **Forward**, money flows from savers → funds → new issues → companies → real investment. **Backward**, two signals flow the other way:

1. **Liquidity flows back as a discount.** Because the secondary market lets any buyer exit, buyers of new issues demand a smaller premium. That lowers the issuer's cost of capital — the \$12.5M/year we computed above — which makes more projects worth financing, which creates more issuance, which creates more securities to trade, which deepens the secondary market further. Each turn of the loop makes the next turn cheaper.

2. **Price discovery flows back as a signal.** The secondary-market price of Truong Steel's bonds tells the *next* steel company exactly what the market thinks of steel mills right now. If the price is high (yields low), capital is cheap and mills get built. If the price collapses, capital dries up and the projects don't happen. The secondary market is a giant, continuous referendum on where society's savings should flow next — this is the deep meaning of [price discovery and efficiency](/blog/trading/capital-markets/how-a-price-is-made-discovery-arbitrage-and-efficiency).

This is also exactly the mechanism behind index inclusion and ETFs, which quietly stitch the secondary market back to the primary one through [the creation/redemption process](/blog/trading/capital-markets/indices-etfs-and-the-bridge-back-to-the-primary-market). When an ETF needs more shares, authorized participants assemble baskets and deliver them in exchange for new ETF units — a primary-market act triggered by secondary-market demand.

The loop also explains a pattern that puzzles newcomers: why issuance is so wildly cyclical. Companies do not issue steadily; they cram years of financing into a few good quarters and then go quiet for years. The reason is the loop. When the secondary market is buoyant — prices high, spreads tight, buyers hungry — the cost of capital is low and the issuance window is wide open, so everyone rushes through it at once. When the secondary market turns, the window slams shut, often within weeks, even though the underlying companies have not changed at all. The primary market is not driven by companies' financing needs; it is driven by the secondary market's appetite. That is the loop dictating the rhythm of real investment in the economy.

And the loop has a dark mirror. Run it forward in good times and it is a virtuous circle: liquidity lowers cost of capital, which funds more projects, which creates more claims, which deepens liquidity. Run it backward in a panic and it is a doom loop: a fall in prices widens spreads, which raises the cost of capital, which kills issuance, which thins the market, which widens spreads further. The same feedback that makes the machine so powerful on the way up makes it so vicious on the way down. Every section that follows — what breaks the machine — is really just a description of the loop running in reverse.

#### Worked example: the loop compounding cost-of-capital

Take a young market where the liquidity premium starts at 3% and falls by 0.5% each year as trading deepens. An issuer rolling \$500M of debt annually pays:

- Year 1: \$500M × 0.03 = **\$15M** in liquidity premium.
- Year 4: \$500M × 0.015 = **\$7.5M**.
- Year 7: \$500M × 0.00 ≈ **\$0**.

Over seven years the issuer's *extra* cost falls from \$15M to nothing — not because the company changed, but because the market around it got deeper. **A maturing secondary market is a compounding subsidy to every issuer in the economy. That is the prize a country wins by building real capital markets — and it is precisely what Vietnam is chasing.**

## What breaks it: every crisis is the same crisis

Here is the unifying claim of the whole series, the thing forty-one posts were quietly building toward: **every famous capital-markets disaster is the same failure wearing a different costume.** In each one, the trust that a claim stays *sellable tomorrow* evaporates, and the moment that happens, the loop runs in reverse and the machine seizes.

![How a securitization blowup, a CCP default, and a fraud all converge on a liquidity freeze](/imgs/blogs/the-capital-markets-machine-the-whole-system-end-to-end-9.png)

Why is this worth stating so insistently? Because if you believe each crisis is its own special story — subprime mortgages here, a rogue trader there, a fraud somewhere else — you will spend your life fighting the last war, fixing the specific gadget that broke and missing the next break entirely. But if you see that they are all the same failure of sellability, you gain a single lens that works on crises that have not happened yet. Whenever you find a market where a claim *looks* liquid but its liquidity secretly depends on a fragile belief — that ratings are right, that a counterparty is solid, that overnight funding will always roll, that the disclosures are true — you are looking at the next crisis in embryo. The specifics will be novel; the mechanism never is.

Look at the four big failure modes the series covered and notice they all funnel into one place.

**2008 — the securitization machine.** Mortgages were pooled into bonds, the bonds were tranched, the senior tranches were stamped AAA, and everyone trusted the stamp. When the underlying mortgages went bad, nobody could tell which bonds were poisoned, so buyers refused to bid on *any* of them. The claims stopped being sellable. We told this story in full in [2008 when the machine broke](/blog/trading/capital-markets/2008-when-the-securitization-machine-broke-case-study).

![US securitization issuance collapsing in 2008 then slowly recovering](/imgs/blogs/the-capital-markets-machine-the-whole-system-end-to-end-7.png)

The chart shows the collapse in one line: securitization issuance fell off a cliff in 2008–2009 and took years to recover. That cliff is not a number — it is the sound of trust disappearing.

#### Worked example: the 2008 cascade in one arithmetic

Take a simplified \$100 mortgage pool, tranched per a standard structure: \$80 senior (AAA), \$15 mezzanine (BBB), \$5 equity (first-loss). The AAA buyers believed losses could never exceed 20%.

- Subprime losses come in at **25%** of the pool: \$25 of losses.
- Equity wiped out (−\$5), mezzanine wiped out (−\$15), leaving \$5 of losses hitting the "safe" AAA tranche.
- AAA recovery: (\$80 − \$5) / \$80 = **94 cents on the dollar** — but only *if you can value it*.

The real damage was not the 6-cent loss. It was that nobody could tell *which* AAA bonds had taken the hit, so the market priced *all* of them as if they were the bad ones — many traded at 50 cents or didn't trade at all. **A 6% real loss became a 50% price collapse because the missing ingredient was not money, it was trust. That gap between real losses and market panic is the signature of every liquidity crisis.**

**A clearing-member default.** If a big member of a CCP fails, the CCP must absorb the loss without itself failing — that is what [the default waterfall](/blog/trading/capital-markets/margin-and-the-default-waterfall-how-a-ccp-survives-a-blowup) is for. If the waterfall is too thin, the failure spreads to surviving members, and suddenly *everyone* doubts whether their settled trades will actually settle. Same root: trust that the claim becomes real money tomorrow.

**A liquidity freeze.** In a repo run — the 2008 ABCP freeze, or any sudden dash for cash — lenders refuse to roll over short-term funding. Holders of perfectly good assets are forced to dump them because they can't finance them, prices crash, which makes lenders even more nervous. The assets were fine; the *financing* of the claim vanished.

**A disclosure or integrity breach.** When a major fraud surfaces — cooked books, a rigged benchmark, insider rings — buyers suddenly cannot trust *any* of an issuer's disclosures. They can no longer value the claim, so they stop bidding. This is exactly why [market-integrity rules](/blog/trading/capital-markets/market-integrity-manipulation-spoofing-and-circuit-breakers) and disclosure law are not bureaucratic overhead — they are the maintenance crew for the trust the whole machine runs on. One Enron, left unpunished, teaches every buyer that filings might be fiction — and that lesson raises the cost of capital for every honest company too.

Notice what these four have in common at the mechanical level. In each, a holder discovers that the claim they own cannot be valued or sold at anything like the price they assumed. The instant that happens, two things follow automatically: they try to sell (adding supply), and other holders, seeing the selling, stop buying (removing demand). Supply up, demand gone — the price gaps down, often with no trades in between, because there is simply no bid. That airless gap, where the next price is far below the last and there is nothing in the middle, *is* a liquidity crisis. It is not caused by a shortage of money in the world; it is caused by a shortage of *willingness to be the buyer*, which is just another name for trust.

Four disasters, one disease. The cure in every case is the same: restore the belief that a claim can be sold tomorrow. That is why central banks become "buyers of last resort" in a crisis — not to bail out the greedy, but to put a floor under sellability so the loop can restart. When a central bank announces it will buy a class of assets "in whatever quantities are needed," it is not really providing money — it is providing the *promise of a bid*, which is the one thing a frozen market is missing. The moment holders believe a buyer will always be there, they no longer need to dump, and the panic drains away. The lender of last resort is, more precisely, the *buyer of last resort* — and what it restores is sellability itself.

## Common misconceptions

**"The secondary market is just gambling — it doesn't help the real economy."** This is the most expensive misconception in finance. The secondary market never sends a cent to a company, true. But it is the reason the *primary* market can send cents to companies at all. Strip out secondary liquidity and the worked example above shows issuers paying 2–3% more for capital — a permanent tax on every productive project in the economy. Trading is not a sideshow to investment; it is investment's precondition.

**"The intermediaries are just middlemen skimming a cut."** Each layer takes a toll, but the tolls are tiny *because* competition driven by the machine grinds them down. Mai paid \$0.04 of her dollar to the underwriter and a fraction of a cent to the exchange — and in return she got diversification, liquidity, settlement guarantees, and legal protection she could never assemble alone. The right question is not "why is there a fee?" but "how cheap did this market make a service that used to be impossible?"

**"Settlement is a back-office detail."** Settlement is where ownership becomes real. The entire fifty-year march from T+5 to T+1 exists to shrink the window in which a counterparty can fail before the claim becomes yours. Ask anyone who lived through a settlement fail whether the plumbing is a detail.

**"Regulation just slows everything down."** Disclosure-based regulation is what lets a stranger buy a claim from another stranger without doing months of due diligence. It is the substitute for trust between people who will never meet. Markets with weak disclosure don't move faster — they barely move at all, because nobody dares buy.

**"A bigger market is always a better market."** Size without depth is fragile. What matters is whether the claim stays sellable under stress. A market can be huge and still freeze the instant trust wobbles — which is precisely what 2008 proved. The US mortgage-bond market was the largest, most sophisticated debt market on earth in 2007, and it seized solid in 2008. Depth, not size, is the measure of a market's health: how much you can sell, how fast, under stress, without crashing the price.

**"Prices are set by supply and demand, full stop."** True but incomplete, and the incompleteness matters. In a capital market, the *willingness to be a buyer or seller at all* depends on trust in the claim and confidence that you can re-trade it later. When that trust holds, supply and demand interact smoothly and prices move in orderly steps. When it breaks, the demand curve does not shift — it *disappears*, and price stops being a continuous function of supply and demand and becomes a discontinuous jump. Most of finance lives in the smooth regime; every crisis is the system falling into the discontinuous one. Understanding markets means understanding both regimes and what flips between them: trust in sellability.

## How it shows up in real markets

**The 2021–2022 IPO whiplash.** The primary market is brutally cyclical because it runs on the secondary market's mood. When secondary prices are high and liquidity is abundant, companies rush to issue; when prices crater, the issuance window slams shut.

![US IPO proceeds by year showing the 2021 boom and 2022 collapse](/imgs/blogs/the-capital-markets-machine-the-whole-system-end-to-end-3.png)

The chart tells the story in two bars: 2021 saw a record wave of US IPO proceeds, and 2022 saw it collapse to almost nothing — not because companies got worse, but because the secondary market turned and the feedback loop ran in reverse. This is the loop made visible: secondary conditions setting the primary market's heartbeat. (For how the price itself gets set on deal night, see [bookbuilding and price discovery](/blog/trading/capital-markets/bookbuilding-and-price-discovery-how-the-ipo-price-is-set).)

The same companies that could have raised capital at a premium in 2021 found, a year later, that no one would fund them at any reasonable price. Nothing about their factories or products changed in twelve months; what changed was the secondary market's mood, and the mood is what sets the cost of capital. This is the single most counterintuitive consequence of the whole machine: whether a real, physical investment gets made depends less on the merits of that investment than on the liquidity conditions in a trading venue thousands of miles away. The loop is that powerful — and that is exactly why a deep, stable secondary market is a national asset, not a casino.

#### Worked example: the IPO "pop" is a transfer, not free money

Truong Steel prices its IPO at \$20 a share and sells 25 million shares to raise \$500M. On the first day of secondary trading the stock opens at \$26 — a 30% "pop."

- Money the company received: 25M × \$20 = **\$500M**.
- Market value at the first trade: 25M × \$26 = **\$650M**.
- The \$150M gap went to the investors who got the IPO allocation, not to the company.

That \$150M is money Truong Steel left on the table — it could have priced higher and raised more. The pop is celebrated as a "successful IPO," but it is really a transfer from the issuing company to the favored buyers who got the allocation. **The pop is where the primary and secondary markets disagree on price out loud, and the company pays for the gap — which is why bookbuilding, the process of guessing the secondary clearing price in advance, is worth so much.**

**The 19 September 2024 T+1 switch.** The US, Canada, and Mexico moved to one-day settlement. The benefit is a smaller window of counterparty risk and less margin tied up in the CCP. The cost lands on foreign investors and FX desks who now have less time to fund trades — a friction that matters enormously for emerging markets, as we'll see with Vietnam.

**Vietnam — the same machine, half-built.** Everything above describes a mature market. Now look at a young one. Vietnam has all five jobs in some form: [HOSE, HNX, and UPCoM exchanges](/blog/trading/capital-markets/the-life-of-a-security-from-idea-to-delisting) as the trading layer, a depository running settlement, the State Securities Commission policing disclosure. But the machine is only partly assembled — and the missing parts are exactly the ones that build trust.

![Vietnam stock market capitalization as a share of GDP rising but still below mature levels](/imgs/blogs/the-capital-markets-machine-the-whole-system-end-to-end-8.png)

The chart shows the depth gap: Vietnam's market cap has climbed toward 60–90% of GDP, but mature markets routinely run well above 100%, and Vietnam still lacks a true central counterparty and still requires pre-funding of trades. Those are not technicalities — they are exactly the trust mechanisms (a CCP that guarantees the trade; settlement that doesn't tie up your cash for days) that let foreign capital flow in cheaply.

Walk the five jobs through Vietnam and the missing pieces light up. **Create a claim?** Yes — companies list on HOSE and the government issues bonds. **Trade it?** Yes — the exchanges run order books, and a wave of new retail investors has pushed trading-account numbers up several-fold in just a few years, deepening volumes. **Settle it?** Here is the first gap: Vietnam settles on a T+2 cycle but, crucially, *without a central counterparty* guaranteeing the trade, and with a pre-funding requirement that forces a foreign buyer to park cash in the country before it is even allowed to trade. That pre-funding is a direct tax on foreign liquidity: a large global fund must tie up money idle, bearing currency risk, just for the privilege of placing an order. **Connect savers to issuers?** Partly — the domestic intermediary base is young and the foreign-access channel is narrow. **Police it?** The State Securities Commission exists and is strengthening, but the disclosure and enforcement track record is still building the long memory that makes strangers trust filings.

The single biggest blocker is the **foreign ownership limit** — "the room." Many Vietnamese companies cap how much of their shares foreigners may own, and once that room is full, a foreign buyer simply cannot buy at any price, or must pay a premium through awkward workarounds. From the machine's point of view this is catastrophic: it means the claim is *not freely sellable to the largest pool of global buyers*, which is precisely the property that lowers the cost of capital. An emerging-market upgrade from FTSE Russell or MSCI is, in plumbing terms, a verdict that these gaps have been fixed enough that global index money can flow in reliably. That is why the upgrade is worth so much: it is not a vanity badge, it is a certification that Vietnam's claims have become reliably sellable to the world.

#### Worked example: the cost-of-capital prize of a Vietnam upgrade

Suppose an emerging-market upgrade (FTSE Russell or MSCI) draws in foreign capital that deepens Vietnam's secondary market enough to cut the country-wide liquidity/risk premium by, say, 1.5 percentage points. Across an economy issuing the equivalent of, very roughly, **\$40bn** of new equity and bonds a year:

\$40bn × 0.015 = **\$600M per year** in lower financing costs, every year, for the whole economy.

That is the prize: not a one-time pop in the index, but a permanent reduction in the cost of building every factory, road, and company in the country. **An emerging-market upgrade is, at bottom, a certificate that says "your claims are now reliably sellable tomorrow" — and that certificate is worth hundreds of millions a year.** Foreign flows, which dominate the VN-Index's swings, are the market voting on exactly that question.

## The takeaway: how to read any market through the machine

You now have the whole machine. Here is the mental model to carry out of this series — a way to read *any* market, anywhere, by asking where each of the five jobs is being done and how well.

When you look at a market, ask:

1. **Where are the claims created, and how cheaply?** That is the primary market — IPOs, bond auctions, follow-ons. A thriving primary market is a sign the loop is healthy.
2. **How liquid is the secondary market?** Look at spreads, volumes, how easily you can exit. This is the single best gauge of the system's health, because it sets the cost of everything upstream.
3. **How short and safe is settlement?** T+1 vs T+2, a real CCP vs none, pre-funding vs delivery-versus-payment. This is the trust infrastructure.
4. **How thin are the intermediary tolls?** Competitive fees mean a working machine; fat fees mean a captured one.
5. **How honest is the disclosure?** Strong, enforced disclosure is what lets strangers trade. Weak disclosure is why some "markets" never get off the ground.

There is a sixth, silent question underneath all five, and it is the one this whole series has really been about: **how well does this market keep its claims sellable tomorrow?** Every one of the five jobs is, ultimately, in service of that. Creating standardised claims makes them sellable to anyone. Running a deep secondary market makes them sellable instantly. Settling safely makes the sale final. Cheap intermediaries make selling affordable. Honest disclosure makes buyers willing to be on the other side. Sellability-tomorrow is not one feature of the machine; it is the output the entire machine is built to produce. Grade a market on that single axis and you will rarely be far wrong.

You can use this as a working lens on the news. A central bank launching a bond-buying program: it is restoring sellability. A regulator tightening disclosure after a scandal: rebuilding the trust that makes buyers show up. An exchange shortening the settlement cycle: shrinking the window where sellability can fail. A country chasing an index upgrade: certifying its claims to the world's buyers. A start-up's IPO "popping" 30% on day one: the primary price was set below where the secondary market clears, a transfer from the company to the first buyers. In every case the same question — *can the holder sell tomorrow, at a fair price, to a buyer who trusts the claim?* — cuts to the heart of the story faster than any headline.

And above all, hold onto the one idea that ties every post in this series together: **a capital market is a machine for turning savings into investment, and it can only do that because the secondary market makes a long-term claim sellable today.** Liquidity is not the froth on top of the market — it is the foundation under it. Every crisis is a moment that foundation cracks; every reform is an attempt to shore it up; every basis point of cost-of-capital saved is the dividend it pays.

The next time you read that a market "crashed," or that a country got an "emerging-market upgrade," or that settlement moved to T+1, you will know what you are really looking at: the machine for moving savings into the future, working a little better — or a little worse — at the one thing it exists to do.

We began with a boring Thursday in September 2024, when US markets quietly shaved a day off settlement. You can now see why that footnote was worth the whole story. Cutting settlement from two days to one shrinks the window in which a counterparty can fail before a claim becomes truly yours — which is another way of saying it makes claims a little more reliably sellable, a little more trustworthy, a little cheaper to fund. No headline, no parade, just the machine tightening one more bolt on the trust that holds it together. Forty-two posts later, that is the whole series in a sentence: the capital market is a machine for turning savings into investment, it runs on the trust that a claim stays sellable tomorrow, and everything else — the IPOs, the order books, the clearinghouses, the disclosure filings, the foreign-ownership rooms — is just the elaborate, beautiful, occasionally fragile machinery for keeping that trust alive. Read any market through that lens and it will rarely surprise you.

## Further reading & cross-links

Within this series — the full machine, track by track:

- Foundations: [what a capital market is](/blog/trading/capital-markets/what-is-a-capital-market-how-money-finds-its-best-use), [the players](/blog/trading/capital-markets/the-players-savers-issuers-and-the-middlemen-in-between), [what a security actually is](/blog/trading/capital-markets/what-a-security-actually-is-claims-you-can-sell), [debt vs equity](/blog/trading/capital-markets/debt-vs-equity-the-two-ways-to-raise-capital)
- Primary market: [the financing ladder](/blog/trading/capital-markets/the-financing-ladder-from-bootstrap-to-public-markets), [the IPO process](/blog/trading/capital-markets/the-ipo-process-end-to-end-from-mandate-to-first-trade), [underwriting and the syndicate](/blog/trading/capital-markets/underwriting-and-the-syndicate-who-takes-the-risk), [how a bond is issued](/blog/trading/capital-markets/how-a-bond-is-issued-auctions-syndication-and-the-deal)
- Secondary market: [inside an exchange](/blog/trading/capital-markets/inside-an-exchange-the-matching-engine-and-the-order-book), [market makers and the spread](/blog/trading/capital-markets/market-makers-and-the-spread-who-provides-liquidity), [lit markets and dark pools](/blog/trading/capital-markets/lit-markets-dark-pools-and-the-fragmented-tape), [how a price is made](/blog/trading/capital-markets/how-a-price-is-made-discovery-arbitrage-and-efficiency), [indices, ETFs and the bridge back](/blog/trading/capital-markets/indices-etfs-and-the-bridge-back-to-the-primary-market)
- Plumbing: [the post-trade lifecycle](/blog/trading/capital-markets/what-happens-after-the-trade-the-post-trade-lifecycle), [the clearinghouse](/blog/trading/capital-markets/the-clearinghouse-how-a-ccp-removes-counterparty-risk), [the default waterfall](/blog/trading/capital-markets/margin-and-the-default-waterfall-how-a-ccp-survives-a-blowup), [settlement and custody](/blog/trading/capital-markets/settlement-and-custody-who-actually-holds-your-shares), [securities lending and repo](/blog/trading/capital-markets/securities-lending-and-repo-the-financing-plumbing)
- Intermediaries: [inside an investment bank](/blog/trading/capital-markets/inside-an-investment-bank-ecm-dcm-ma-and-trading), [the buy-side](/blog/trading/capital-markets/the-buy-side-who-actually-owns-the-market)
- Securitization & regulation: [securitization from first principles](/blog/trading/capital-markets/securitization-from-first-principles-turning-loans-into-bonds), [2008 when the machine broke](/blog/trading/capital-markets/2008-when-the-securitization-machine-broke-case-study), [why markets are regulated](/blog/trading/capital-markets/why-markets-are-regulated-disclosure-and-the-securities-acts), [disclosure and insider trading](/blog/trading/capital-markets/disclosure-the-prospectus-filings-and-insider-trading), [market integrity](/blog/trading/capital-markets/market-integrity-manipulation-spoofing-and-circuit-breakers)

Build-on and sibling-series posts:

- [Inside an investment bank: how they make money](/blog/trading/finance/inside-an-investment-bank-how-they-make-money)
- [Stock exchanges and clearinghouses](/blog/trading/finance/stock-exchanges-and-clearinghouses)
- [The yield curve explained](/blog/trading/fixed-income/the-yield-curve-explained-the-most-important-chart-in-finance) (how the price of money itself is set)
- [Securitization: how banks turn loans into securities](/blog/trading/banking/securitization-how-banks-turn-loans-into-securities)
- [LTCM 1998: when genius failed](/blog/trading/finance/ltcm-1998-when-genius-failed) — a liquidity freeze in one fund
