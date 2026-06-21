---
title: "What a Security Actually Is: Claims You Can Sell"
date: "2026-06-21"
publishDate: "2026-06-21"
description: "A security is a standardised, transferable, tradable claim on cash flows or ownership — and those two inventions, fungibility and transferability, are the whole reason capital markets exist."
tags: ["capital-markets", "securities", "stocks", "bonds", "fungibility", "transferability", "howey-test", "dematerialisation", "isin", "tokenisation"]
category: "trading"
subcategory: "Capital Markets"
author: "Hiep Tran"
featured: true
readTime: 42
---

> [!important]
> **TL;DR** — A security is a *standardised, transferable, tradable claim* on cash flows or ownership; the two inventions that matter are fungibility (one unit is identical to the next, so they pool into deep liquidity) and transferability (you can sell your claim to a stranger, which is what makes a long-term claim financeable).
>
> - There are five families — equity, debt, hybrids, pooled vehicles, and derivatives — and they differ only in *what* they claim and *where they sit in line* when cash is paid out.
> - A claim becomes a tradable object through dematerialisation: paper certificates became electronic book-entry records held by a central depository, identified by an ISIN, and transferred by a ledger update in seconds.
> - The legal definition (the Howey test: an investment of money in a common enterprise expecting profit from others' efforts) decides what gets regulated — which is why tokenised "digital securities" are the same idea in new wrapping.
> - The one fact to remember: the global stock of securities is roughly **\$115tn of equity and \$140tn of bonds** — and none of it could be funded if you couldn't sell your slice tomorrow morning.

A bank loan and a corporate bond do almost the same economic thing: a company gets cash today and promises to pay it back with interest. Yet one of them you cannot get rid of without the bank's permission, and the other you can sell to a stranger in Tokyo before lunch. That single difference — *can I sell my claim?* — is the entire subject of this post, and it is the hinge on which the whole capital-markets machine turns.

In this series we keep coming back to one idea: a capital market is a machine that turns savings into long-term investment, and it runs on two engines — a **primary market** that *creates* securities to raise capital, and a **secondary market** that *trades* them to provide liquidity. The secret that makes it all work is that secondary-market liquidity is what makes primary issuance possible. Nobody funds a 30-year railway, a 10-year drug trial, or a 100-year cathedral with money they can never get back. They fund it with money they can get back *by selling their claim to someone else*. The thing they sell is a security. So before we can talk about how securities are issued (Track B), traded (Track C), cleared (Track D), or regulated (Track G), we have to answer the deceptively simple question this whole edifice rests on: what *is* a security, actually?

The answer is not "a stock" or "a bond" — those are examples. The answer is a *design pattern*: take an ordinary economic promise, do two specific things to it, and you transform it from a private favour you're stuck with into a liquid object the whole world can price. This post is about those two things.

![How a private promise becomes a security through standardisation and transferability](/imgs/blogs/what-a-security-actually-is-claims-you-can-sell-1.png)

## Foundations: a security is a claim you made fungible and transferable

Start with money you already understand. Suppose your neighbour borrows \$1,000 from you and writes "IOU \$1,000, repaid in one year with \$60 interest" on a napkin. You now hold a *claim* — a legal right to a future cash flow. It is real, it is enforceable, and it has value. But it has two crippling problems.

First, it is **bespoke**. The terms are specific to your neighbour, the napkin, and the handshake. If a second neighbour borrows \$1,000 on slightly different terms, you now hold two different claims that can't be compared at a glance, can't be added together cleanly, and can't be priced against each other. Every IOU is a snowflake.

Second, it is **stuck to you**. If you suddenly need your \$1,000 back before the year is up, you can't simply hand the napkin to a stranger and walk away with cash. The stranger doesn't know your neighbour, doesn't trust the napkin, can't verify the terms, and has no way to collect. Your claim is *illiquid*: locked to the original lender until maturity.

A security is what you get when you fix both problems on purpose.

You fix the first problem with **standardisation** — and the powerful version of standardisation is **fungibility**: every unit of the security is legally and economically identical to every other unit. One share of Apple is interchangeable with any other share of Apple. One \$1,000 face-value bond of a given issue is interchangeable with any other bond of that same issue (same coupon, same maturity, same rank). Because the units are identical, they *pool*. A million identical shares aren't a million snowflakes; they are one deep, homogeneous mass that a market can quote a single price for. Fungibility is what lets ten thousand buyers and ten thousand sellers meet at one number.

You fix the second problem with **transferability** — the legal and operational ability to sell your claim to someone you've never met, with the buyer receiving exactly the rights you held, and the issuer obligated to honour them to whoever now holds the security. Transferability is what turns "locked to you until maturity" into "exit any time the market is open."

Here is the punchline, and it is the spine of this entire series: **transferability is what makes a long-term claim financeable**. A company that wants to build something that pays off over 30 years cannot find a single investor willing to wait 30 years. But it *can* find an investor willing to hold the claim for 30 *months* — because that investor knows they can sell to the next holder, who sells to the next, and so on, a relay race of holders that collectively spans the 30 years even though no single runner goes the whole distance. The long-term project gets long-term funding from a chain of short-term holders, and the baton they pass is the security. Fungibility makes the baton identical so the next runner will take it; transferability makes the handoff legal and instant. Together they are the whole invention.

> [!note]
> **The one-line definition.** A security is a *standardised (ideally fungible), transferable, tradable claim on cash flows or ownership.* Strip away any one of those words and it stops being a security: an un-standardised claim won't pool, an un-transferable claim can't be sold, and a claim on nothing isn't worth anything.

It is worth pausing on *how recent and how unobvious* this invention is. For most of human history, lending money meant lending it to someone you knew, on terms you negotiated, with no expectation of ever passing the loan to a third party — the claim and the relationship were the same thing. The leap that built the modern world was realising that a claim could be *severed from the relationship*: I can hold a piece of a company or a government's debt without ever meeting the company, the government, or the person I eventually sell to. The Dutch East India Company, often called the first joint-stock company, is famous not merely because it sold shares in 1602 but because those shares were transferable on the Amsterdam exchange — a holder could exit without dissolving the company or finding a replacement partner. That separation of *ownership* from *the act of running the enterprise*, made durable by transferability, is arguably the single most consequential financial idea ever, because it let strangers pool their savings into ventures none of them could fund alone and still keep the option to leave. Every IPO, every bond auction, every ETF is a descendant of that one move.

The deeper reason transferability is so powerful is that it changes the *time horizon mismatch* at the heart of all long-term investment. Savers, as a rule, want their money back soon and on demand — they are saving for a house, a child's education, a rainy day, retirement, all of which can arrive at unpredictable times. Real investment, by contrast, is overwhelmingly long-term and irreversible: a factory, a pipeline, a fab, a vaccine pipeline, a railway. These two facts seem irreconcilable. How do you fund a 30-year asset with money that its owners might want back next Tuesday? The answer is the security. Each saver holds a claim they can sell at will, so *from their point of view* the investment is liquid and reversible — they can be out by Tuesday. But because the claim simply changes hands rather than being redeemed by the issuer, *from the project's point of view* the funding is permanent and patient. The security is a piece of financial alchemy that makes the same pile of capital simultaneously short-term to its holders and long-term to its user. Banks perform a version of this trick on their balance sheets (borrowing short via deposits, lending long), but securities let the *market* perform it at vastly greater scale, and without a single intermediary bearing all the risk.

Why does this matter at the scale of an economy? Because the alternative to securities is a **bank queue** — you save money, the bank lends it out, and the bank holds the loan. That works, but the bank is a single, capacity-constrained, risk-averse allocator, and the loans it holds are illiquid. Securities replace the queue with a *market*: savings flow directly into standardised claims that millions of holders can price and trade continuously. The result is a vastly larger, deeper pool of long-term capital. We argue this case in full in [what is a capital market and how money finds its best use](/blog/trading/capital-markets/what-is-a-capital-market-how-money-finds-its-best-use); here the point is narrower — *the security is the unit that makes the market possible.* And it is enormous.

### Why a bank loan is not a security (yet)

It helps to nail the definition by looking at the closest thing to a security that *isn't* one: an ordinary bank loan. Economically, a loan and a bond are near-twins — both are a promise to repay borrowed money with interest. So why is one a security and the other not?

Take the napkin IOU again, but make it a real \$10,000,000 corporate loan from a single bank. It fails the security test on both inventions. It is not **standardised**: the loan agreement is a bespoke document, dozens of pages of covenants, collateral terms, and conditions negotiated for this one borrower and this one lender. No two loans are alike, so they don't pool — you can't quote a single price for "a loan," only for *this* loan after reading its specific contract. And it is only weakly **transferable**: the bank *can* sell the loan, but doing so usually requires the borrower's consent, a fresh round of due diligence by the buyer, a negotiated assignment, and often a haircut for the trouble. There is no deep, continuous market of buyers standing ready at a posted price. The loan is sticky.

Now watch the same economic promise cross the line into security-hood. If, instead of borrowing from one bank, the company issues a \$10,000,000 *bond* — standard \$1,000 denominations, a single set of public terms, a prospectus everyone reads once, an ISIN, and free transfer to any buyer — the identical underlying promise becomes a fungible, sellable object. Nothing changed about the cash the company owes; everything changed about whether you can get out. **The difference between a loan and a bond is not the economics of the debt; it is whether the claim was engineered to be standardised and transferable.** That is the whole boundary of the word "security," and it is why the same company can fund itself either way and choose based on how badly its investors will want an exit.

This also explains a deep structural fact about modern finance: the relentless drive to *turn loans into securities*. A bank holding a portfolio of illiquid mortgages can pool them, standardise the claims against the pool, and issue tradable securities backed by the cash flows — securitization, the subject of Track F. The motive is exactly the inventions above: convert sticky, bespoke loans into fungible, transferable claims so they can be sold, freeing the bank's capital to lend again. The security is the *output* of that transformation, and the entire \$255tn market is, at bottom, the world's economic promises run through the standardise-and-transfer machine.

![The global stock of securities split into equity and bonds, globally and for the US](/imgs/blogs/what-a-security-actually-is-claims-you-can-sell-2.png)

Roughly \$115 trillion of equity and \$140 trillion of bonds sit outstanding worldwide — about a quarter-quadrillion dollars of standardised, transferable claims, of which the US alone accounts for about \$55tn of equity and \$55tn of bonds. Every dollar of that is a claim that someone, somewhere, can sell tomorrow. That sellability is not a nice-to-have feature bolted onto the side; it is the load-bearing wall. Take it away and the \$255tn collapses back into a tangle of bespoke IOUs that nobody could fund.

#### Worked example: a share as a 1/N fractional claim

Suppose a company is worth \$2,000,000,000 (\$2bn) in total equity value, and it has divided its ownership into **200 million shares**. Each share is then a claim on:

$$\frac{\$2{,}000{,}000{,}000}{200{,}000{,}000 \text{ shares}} = \$10 \text{ of company value per share.}$$

Buy one share for \$10 and you own exactly 1/200,000,000th of everything the company owns and earns: a 0.0000005% slice of its factories, its brand, its future profits, and its votes. Buy 2 million shares for \$20,000,000 and you own 1% of the company. The key property is the *identical-ness*: your share grants exactly the same rights as the founder's share and the pension fund's share. That sameness is what lets all 200 million shares trade at one quoted price — and what lets you sell your slice without renegotiating anything. **A share is fungibility applied to ownership: it chops a whole company into identical, sellable pieces.**

## The taxonomy: five families of claim

Every security in the world is a claim on either *cash flows* (money the issuer promises to pay) or *ownership* (a residual stake in an enterprise) — or some blend of the two. From that single seed grow five families. They differ in only two dimensions: **what** you're promised, and **where you sit in line** when the issuer pays out or fails.

![The five families of securities: equity, debt, hybrids, pooled vehicles, and derivatives](/imgs/blogs/what-a-security-actually-is-claims-you-can-sell-3.png)

**Equity (shares).** A share is a claim on *ownership* — specifically the *residual* claim. After a company pays its suppliers, its workers, its lenders, and the tax authority, whatever is left belongs to the shareholders. That "whatever is left" is the residual, and it is the most powerful and most dangerous position in finance: unlimited upside (if the company grows, the residual grows with no ceiling) but last in line (if the company fails, equity is wiped out before debt loses a cent). Shares typically also carry *votes* — a say in who runs the company — and *dividends* — a discretionary share of profits the board chooses to pay out. The reason a stock can be valued at all is a separate craft; we don't re-derive DCF or multiples here — see the equity-research side of the site for that. Our concern is what a share *is*: a fungible, transferable slice of residual ownership.

**Debt (bonds, notes, bills).** A bond is a claim on *cash flows* you are contractually owed: fixed (or formula-driven) interest payments called **coupons**, plus the return of the **principal** (face value, par) at **maturity**. Debt sits *ahead of* equity in line — bondholders get paid before shareholders see anything, in both good times (interest is a legal obligation, not a discretionary dividend) and bad (in bankruptcy, bondholders are creditors with first claim on assets). The trade-off is symmetric: debt is safer because it's senior, but its upside is capped — the most a normal bond can pay you is its promised coupons and your principal back, no more. We treat the *issuance and deal mechanics* of bonds later in this series; the *pricing* of bonds (yields, duration, the curve) lives in the fixed-income track — see [the yield curve explained](/blog/trading/fixed-income/the-yield-curve-explained-the-most-important-chart-in-finance). Here, again, the point is the *claim*: a bond is a fungible, transferable slice of a standardised loan. The deeper equity-vs-debt fork — sell ownership or borrow — gets its own treatment in [debt vs equity: the two ways to raise capital](/blog/trading/capital-markets/debt-vs-equity-the-two-ways-to-raise-capital).

**Hybrids (preferred shares, convertibles).** These blend the two. **Preferred stock** pays a fixed dividend like a bond but ranks below debt and above common equity — debt-like income with an equity-like position in the stack. A **convertible bond** is a bond that the holder can convert into a fixed number of shares: you collect coupons like a lender, but if the stock soars you convert and capture the upside like an owner. Hybrids exist precisely because the equity/debt line is a spectrum, not a wall, and issuers and investors want to sit at custom points along it.

**Pooled vehicles (fund units, ETF shares).** Instead of buying one company's shares, you buy *one unit of a fund that owns hundreds of securities*. A mutual-fund unit or an **ETF share** is a claim on a *pro-rata slice of a basket*. If a fund holds \$1bn of assets and has issued 100 million units, each unit is a \$10 claim on 1/100,000,000th of the whole basket. The genius here is *recursion*: the fund's units are themselves standardised, transferable, tradable claims — securities built out of other securities — so you get instant diversification in a single fungible object. ETFs add a second twist (continuous on-exchange trading plus a creation/redemption mechanism that keeps the share price glued to the basket value), but the core claim is simple: **a slice of a basket.**

**Derivatives (claims on claims).** A derivative's value is *derived* from an underlying security, rate, or price — an option on a stock, a future on an index, a swap on an interest rate. It is a claim whose payoff is defined by reference to something else. Derivatives are the deepest end of the pool and we deliberately do not trade them here; the mechanics, payoffs, and Greeks live in the options-and-volatility track — see [the order-book simulator](/blog/trading/quantitative-finance/order-book-simulator-quant-research) for how such claims actually change hands. For our purposes, note only the pattern: once you can standardise and transfer a claim, you can build standardised, transferable claims *on top of it* — claims on claims, all the way up.

The taxonomy isn't trivia. It tells you, for any security you encounter, the two things that matter most: *what cash flow or ownership am I claiming, and who gets paid before me?* Everything else — the ticker, the exchange, the colour of the certificate — is packaging.

#### Worked example: a bond as a stream of \$30 coupons plus \$1,000 par

Take a plain two-year corporate bond with a **\$1,000 face value** and a **6% annual coupon paid semiannually**. "6% on \$1,000" is \$60 a year, paid in two instalments, so each coupon is:

$$\text{coupon} = \frac{6\% \times \$1{,}000}{2} = \frac{\$60}{2} = \$30 \text{ every six months.}$$

Buy the bond at par for \$1,000 today and you have purchased a precisely dated stream of claims: \$30 at six months, \$30 at twelve, \$30 at eighteen, and at twenty-four months a final \$30 coupon *plus* your \$1,000 principal back — a final cash flow of \$1,030. Total cash received over the life: \$30 × 3 + \$1,030 = \$1,120 against \$1,000 paid, a \$120 gain if held to maturity and the issuer never defaults. **A bond is just a calendar of fixed, claimable cash flows — and because every \$1,000 bond of this issue is identical, you can sell yours at month seven to a stranger who simply collects the rest of the calendar.**

![A bond as a timeline of fixed cash flows: a payment out at purchase and four claimable inflows](/imgs/blogs/what-a-security-actually-is-claims-you-can-sell-5.png)

Notice what transferability bought the issuer here. The company needed money for two years, but it didn't have to find one investor willing to lock up \$1,000 for two years. It found an investor willing to hold for seven months, who sold to the next, and so on. The *issuer's* funding was long-term; each *holder's* commitment was short-term and reversible. That mismatch — long-term funding from short-term holders — is impossible without a security, and it is the engine of the whole bond market.

#### Worked example: a fund unit as a claim on a basket

A fund holds a basket: \$400m of equities, \$500m of bonds, and \$100m of cash, for **\$1,000,000,000 of total assets**. It has issued **50,000,000 units**. Each unit's net asset value (NAV) is:

$$\text{NAV per unit} = \frac{\$1{,}000{,}000{,}000}{50{,}000{,}000} = \$20 \text{ per unit.}$$

Buy 1,000 units for \$20,000 and you hold a claim on 1/50,000th of the entire basket — proportionally \$8,000 of equities, \$10,000 of bonds, and \$2,000 of cash, without buying any of those underlying securities directly. If the basket's value rises 5% to \$1.05bn, your NAV per unit rises to \$21 and your 1,000 units are worth \$21,000. **A fund unit is a security that re-packages many securities into one fungible, transferable claim — diversification you can buy and sell in a single click.**

## What a claim actually bundles: cash-flow rights, control rights, and a place in line

"A claim on cash flows or ownership" is a tidy phrase, but a real security bundles a *set* of distinct rights, and knowing which ones you hold is the difference between understanding a security and just owning a ticker. There are two broad bundles — **economic rights** and **control rights** — and a third property that overrides both when things go wrong: your **priority in line**.

**Economic rights** are the rights to money: a share's right to dividends and to the residual value of the company; a bond's right to coupons and principal; a fund unit's right to a slice of the basket's value and its distributions. These are the rights people usually mean when they say "I own a security." They are the cash flows the claim points at.

**Control rights** are the rights to *decide*: a common share's right to vote on directors and major corporate actions; a bond's covenants (contractual promises the issuer makes — limits on extra borrowing, requirements to maintain certain ratios) and the bondholders' rights if those covenants are breached. Control rights are why two securities with similar cash flows can be worth very different amounts: a share with ten votes and a share with one vote may both claim the same dividend, but they are not the same security, because one carries far more say over the enterprise. Many companies issue *dual-class* shares precisely to split economic rights from control — founders keep the votes, the public buys the cash flows.

The single most important property, though, is **priority** — where your claim sits in line when cash is distributed, and especially when the issuer runs out of money. This is the *capital stack*, and it is the spine of how risk is sliced in finance. From safest to riskiest:

- **Senior debt** is paid first. In a wind-down, senior creditors are made whole before anyone below them gets a cent.
- **Subordinated (junior) debt** is paid after senior debt but before equity.
- **Preferred shares** sit below all debt but above common shares — they get their fixed dividend and their liquidation value before common holders.
- **Common equity** is the residual: last in line, paid only after everyone above is fully satisfied — which in a bankruptcy is usually *never*, but in a success is *everything left over, with no ceiling*.

Priority is the hidden variable behind the whole risk-return trade-off. The reason equity can multiply your money and also zero it out is that it sits at the bottom of the stack: it absorbs the first losses and captures the last gains. The reason a senior bond is "safe" is not that the issuer is virtuous but that the bond is *structurally* protected by everything junior to it — the equity and subordinated debt below it form a cushion that must be wiped out before the senior bond takes a loss. When you read that a tranche is "AAA," what you are really reading is "there is so much loss-absorbing capital beneath this claim that it would take a catastrophe to reach it." We see this exact logic, quantified, when we look at securitization tranches later.

#### Worked example: who gets paid when the money runs out

A company fails and its assets are sold for **\$70,000,000**. Its capital stack, from senior down, is: **\$50,000,000 of senior debt, \$20,000,000 of subordinated debt, \$10,000,000 of preferred stock, and common equity**. Walk the waterfall:

1. **Senior debt (\$50m):** paid in full first → takes \$50,000,000. Remaining: \$70m − \$50m = \$20,000,000.
2. **Subordinated debt (\$20m):** paid next → takes the remaining \$20,000,000 in full. Remaining: \$0.
3. **Preferred stock (\$10m):** there is nothing left → receives **\$0**.
4. **Common equity:** receives **\$0** and is wiped out.

The senior lenders got 100 cents on the dollar; the subordinated lenders got lucky and were also made whole; preferred and common got nothing. Now suppose instead the assets had fetched only \$40m: senior debt would take \$40m and recover 80%, and *everyone* below — sub debt, preferred, common — would get zero. And had they fetched \$90m, senior and sub debt would be made whole (\$70m), preferred would take its \$10m, and the remaining \$10m would flow to common equity — the only rung whose payoff has no ceiling on the way up and no floor on the way down. **The same dollar of company value is worth wildly different amounts depending on which rung of the stack your security sits on; "what do I claim?" is incomplete without "and who gets paid before me?"** This priority ordering is the reason the five families exist at all: each family is simply a different deal about where you stand in this queue, and the price of every security in the world is, at bottom, the market's estimate of how often the money will run out before it reaches your rung.

## How standardisation manufactures liquidity

It's worth dwelling on *why* fungibility is so powerful, because it's the least intuitive of the two inventions. Transferability is obviously useful — of course you want to be able to sell. But fungibility seems like a technicality. It isn't. Fungibility is what *manufactures* liquidity out of thin air.

Liquidity is the ability to buy or sell quickly without moving the price much. It comes from having *many* buyers and *many* sellers willing to trade *the same thing*. If every claim were a snowflake, no two buyers would ever want the identical object, and there'd be no crowd at any single price — every trade would be a one-off negotiation, like selling a used house. Fungibility collapses ten thousand slightly-different claims into one identical claim that ten thousand people are happy to buy or sell, and *that* crowd is the liquidity. The depth of the buyer pool is literally a function of how standardised the claim is.

![Who owns US equity, by type of holder, showing a deep and varied buyer pool](/imgs/blogs/what-a-security-actually-is-claims-you-can-sell-4.png)

Look at *who* holds US equity and you see the buyer pool that fungibility built. Households directly own about 38%, mutual funds and ETFs about 28%, foreign investors about 17%, pensions about 10%. That is an extraordinarily diverse crowd — a retiree in Ohio, a sovereign-wealth fund in Singapore, an index fund in Boston — and the *only* reason they can all hold and trade "a share of Apple" is that every share of Apple is identical. The retiree's share and the sovereign fund's share are the same object, so they meet at one price on one exchange. Standardisation didn't just organise the market; it *summoned* the participants. A security with a thin, narrow holder base is illiquid almost by definition — see how spread and depth collapse for small issues in the secondary-market track of this series.

There's a second-order effect here that most beginners miss: fungibility is what makes *price discovery* possible at all. Price discovery is the process by which a market figures out what something is worth, and it works by aggregating the opinions of thousands of buyers and sellers into a single number. But you can only aggregate opinions about the *same thing*. If every claim were unique, each trade would reveal the price of one snowflake and tell you nothing about the next — there would be no "market price," only a scatter of unrelated one-off deals, the way there is no single "price of a house" but only the price of *that* house on *that* day. Fungibility gives a million units one shared identity, so a million separate trades all become evidence about one price. The bid-ask spread you see quoted — the tiny gap between the best price a buyer will pay and the best a seller will accept — is the visible output of that aggregation, and it tightens as fungibility deepens the crowd.

How tight? The spread is essentially the *toll* you pay for instant liquidity, and it scales inversely with the depth of the fungible pool. A mega-cap like Apple, with millions of identical shares and a planet-wide buyer base, trades at a spread of about a single basis point — one hundredth of one percent. A thinly-held micro-cap stock, with the same legal fungibility but a tiny crowd, can show a spread of 80 basis points or more — eighty times wider — because there simply aren't enough willing counterparties standing by at any given instant. Same security *type*, wildly different liquidity, and the difference is purely the size of the pool. This is why standardisation alone isn't sufficient for liquidity; it's *necessary* — it's the precondition that lets a crowd form — but the crowd still has to show up. We trace exactly how spread and depth behave across the size spectrum in the secondary-market track of this series.

This is also why the *secondary* market (where existing securities trade) and the *primary* market (where new ones are created) are joined at the hip. A company issuing shares for the first time is implicitly promising every buyer: "these will be liquid — you'll be able to sell them on a deep secondary market." If that promise is credible, investors pay a high price at issuance, and the company raises a lot of capital. If it isn't — if the secondary market for these shares will be thin and the buyer pool shallow — investors demand a steep *illiquidity discount*, and the company raises far less. **Secondary-market liquidity prices itself straight into the primary market.** The deeper the resale market, the cheaper the capital. That is the spine of this series stated as a market mechanism.

The size of that illiquidity discount is not a rounding error. Empirically, claims that are otherwise identical but cannot be freely resold — restricted stock, private-company shares, thinly-traded bonds — routinely change hands at discounts of 20% to 30% relative to their freely-tradable twins. That gap *is* the market pricing the word "transferable." Flip it around and you have stated the entire commercial case for building a deep secondary market: every percentage point you can shave off the illiquidity discount is a percentage point cheaper that every issuer in the economy can raise capital. A government that wants its companies to fund themselves cheaply doesn't lecture them about productivity; it builds deep, fungible, well-regulated securities markets, because that is what compresses the discount and lowers the cost of capital for everyone at once.

## How a claim becomes a tradable object

So far we've talked about the *idea* of a security — a standardised, transferable claim. But ideas don't trade; *objects* do. For a claim to change hands ten thousand times a second, it has to be represented as something a settlement system can move cheaply, instantly, and without error. The history of how claims became tradable objects is the history of dematerialisation.

![From bearer certificate to registered to dematerialised book-entry record at a depository](/imgs/blogs/what-a-security-actually-is-claims-you-can-sell-6.png)

**Bearer securities** came first. A bearer certificate is a physical document where *whoever holds the paper owns the claim* — like cash. No name on it, no register; possession is everything. Bearer instruments are gloriously simple to transfer (just hand over the paper) and gloriously dangerous (lose it and you've lost everything; steal it and you own it; and they're a money-laundering and tax-evasion dream, which is why most jurisdictions have largely killed them off).

**Registered securities** fixed the danger. Here the issuer (or its agent) keeps a **share register** — an official list of who owns what. Your ownership is established by your name on that register, not by possession of a certificate. To transfer the claim, you don't hand over paper; you instruct the registrar to *update the list* — strike your name, write the buyer's. Registration makes ownership robust (lose the certificate and you still own the claim; the register is the truth) but historically slow (every transfer meant physically updating books and moving certificates).

**Dematerialisation** is the leap that made modern markets possible. Instead of printing certificates at all, the security exists only as an *electronic entry* in a system. The paper is destroyed (or never created); the record *is* the security. This sounds mundane and it is the single most important piece of plumbing in the entire capital market. When a claim is a pure book-entry record, transferring it is just editing a database row — settlement that once took five business days now takes one (the US moved to **T+1** in May 2024), and could in principle be instant.

The records live at a **central securities depository** — DTC in the United States, VSDC in Vietnam, Euroclear and Clearstream in Europe. The depository is the master ledger: it holds (or "immobilises") the securities centrally and tracks ownership through a chain of custodians and brokers down to you. This sets up the entire custody-and-settlement story we tell in Track D; for now the point is that *the depository is where the object actually lives*. You don't have a certificate in a drawer; you have a claim recorded in your broker's account, which is recorded in the broker's account at the depository, which is the authoritative ledger. Custody is a chain of book-entries.

This produces a feature most investors never notice: you almost certainly do not hold your securities in your own name. Modern markets run on an **indirect holding system** (in the US, the colloquial term is holding in "street name"). The depository records that the securities are held by a small number of large participants — banks and brokers. Your broker, in turn, records on *its* books that you are the **beneficial owner** of your slice. So the legal chain is: the issuer's register shows the depository's nominee as the holder of record; the depository's books show your broker; your broker's books show you. You get all the economic and control rights — dividends are passed down the chain to you, votes are solicited from you — but the *registered* holder is a nominee several layers up.

Why build it this way? Because it makes transfer almost free. When you sell your shares to someone whose broker also uses the same depository, *nothing moves at the depository level at all* — the security stays in the same nominee account, and the change of ownership happens entirely as book-entry updates between brokers. Only net positions between depository participants ever need to settle. This is what collapses millions of daily trades into a handful of net movements, and it is the foundation of the netting and clearing efficiency we explore in Track D. The cost is a subtle one: because you're a beneficial owner several layers removed from the register, your rights run *through* the chain, and the integrity of every intermediary in it matters — a theme that returns when we discuss custody risk and what happens when a broker fails.

Finally, for a global market to trade these objects, every security needs an **identifier** — a unique, unambiguous name that any system anywhere can resolve. Three matter:

- **CUSIP** — a 9-character code identifying North American securities (assigned by CUSIP Global Services).
- **ISIN** — the *International Securities Identification Number*, a 12-character global standard that wraps a national identifier with a country prefix and a check digit. This is the universal key.
- **Ticker** — the short, human-friendly trading symbol on a given exchange (AAPL, MSFT). Tickers are *not* unique globally — the same ticker can mean different companies on different exchanges — which is exactly why the back office relies on ISIN/CUSIP, not tickers.

#### Worked example: reading an ISIN

Take the ISIN **US0378331005** (Apple's common stock). It decodes in three parts:

- **`US`** — the two-letter country code (ISO 3166) of the issuing jurisdiction. Here, the United States.
- **`037833100`** — a 9-character national identifier. For US securities this *is* the CUSIP. So Apple's CUSIP is `037833100`, embedded right inside its ISIN.
- **`5`** — a single check digit, computed from the preceding 11 characters by the Luhn algorithm. If someone mistypes the ISIN, the check digit almost always fails to match, and the system rejects the bad code before it can settle a trade against the wrong security.

So one 12-character string tells a machine in Frankfurt, instantly and unambiguously: *US-issued, CUSIP 037833100, and here's a checksum proving you typed it right.* **An ISIN is how a fungible claim gets a globally unique name — the barcode that lets the world's settlement systems agree on exactly which security just changed hands.**

That barcode is not a detail. A market of \$255tn in standardised claims, traded across dozens of countries and hundreds of venues, can only net, clear, and settle if every participant agrees on *which security* a trade refers to. The identifier is the agreement.

## What legally counts as a security

We've defined a security economically (a standardised, transferable claim) and operationally (a dematerialised book-entry record with an identifier). But there's a third definition that decides something crucial: *whether the law treats it as a security at all* — and therefore whether the issuer must register it, disclose its risks, and submit to the rules. This is not academic. Get classified as a security and you face a mountain of disclosure obligations; escape the classification and you're (mostly) free of them. Billions of dollars and many criminal cases turn on which side of the line a thing falls.

The American answer — and the one most of the world echoes — comes from a 1946 Supreme Court case about orange groves, *SEC v. W.J. Howey Co.* Howey sold tracts of a Florida orange grove to buyers, then offered to manage the groves and split the profits. Buyers weren't farmers; they were investors hoping Howey's management would make them money. The Court asked: is this *thing* — a land sale plus a service contract — actually a security in disguise? It built a four-part test, and the test is the closest thing finance has to a definition of "investment" itself.

![The Howey test: four prongs that together decide whether an arrangement is a regulated security](/imgs/blogs/what-a-security-actually-is-claims-you-can-sell-9.png)

An arrangement is an **investment contract** (and therefore a security) if there is:

1. **an investment of money** — you put in capital (or something of value);
2. **in a common enterprise** — your fortunes are pooled with other investors' and/or the promoter's;
3. **with an expectation of profit** — you're in it for a financial return, not to consume or use the thing; and
4. **to be derived from the efforts of others** — the return depends on the work of a promoter or third party, not your own labour.

All four prongs must be met. The orange-grove deal met all four (you bought a tract, your returns were pooled and depended entirely on Howey's farming), so it was a security despite being dressed as a real-estate sale. The test's brilliance is that it looks at *economic reality, not labels*: you cannot dodge securities law by calling your security a "membership," a "token," a "yield product," or a "fractional NFT" if the substance is an investment of money in a common enterprise expecting profit from others' efforts.

Why does this definition matter so much? Because being a security triggers **disclosure-based regulation** — the requirement to tell investors the truth about what they're buying before they buy it. The whole edifice of the 1933 and 1934 Securities Acts, the birth of the SEC, and the principle that "sunlight is the best disinfectant" hangs off this classification. We unpack that regime in [why markets are regulated: disclosure and the securities acts](/blog/trading/capital-markets/why-markets-are-regulated-disclosure-and-the-securities-acts). The Howey test is the gate; everything on the regulated side of the gate has to disclose.

#### Worked example: is a token a security?

A startup sells **10,000,000 tokens at \$2 each**, raising **\$20,000,000**, and tells buyers the team will use the proceeds to build a platform that will make the tokens more valuable. Run the prongs:

1. *Investment of money?* Yes — buyers paid \$2 × 10,000,000 = \$20,000,000.
2. *Common enterprise?* Yes — all buyers' fortunes ride on the same platform and the same team's success.
3. *Expectation of profit?* Yes — buyers bought because they expect the token's price to rise, not to *use* it today.
4. *From the efforts of others?* Yes — the value depends on the team building and promoting the platform.

Four for four: under Howey this token sale is almost certainly a **securities offering**, regardless of the word "token," and selling \$20,000,000 of it without registration or a valid exemption is selling unregistered securities. **The label is wrapping; the four prongs are the substance — and the substance is what the law reads.** This is the exact logic behind the major crypto enforcement actions of the 2020s, and it's why "is it a security?" is the first question any token's lawyers ask.

## The modern edge: tokenisation is the same idea in new wrapping

If a security is a standardised, transferable claim represented as a book-entry record, then a *tokenised* security is the same claim represented as an entry on a blockchain instead of a depository's database. That's the whole idea. The breathless language around "digital securities" and "real-world asset tokenisation" can obscure how *conservative* the concept is: it is dematerialisation, again, with a different ledger technology underneath.

The genuine improvements tokenisation can offer are exactly the things that make any security better — finer standardisation and easier transfer. **Fractionalisation**: if a claim can be split into a billion tiny identical units cheaply, you can let someone own \$50 of a building or a Treasury bond, deepening the buyer pool (more fungibility → more liquidity). **Faster settlement**: a blockchain can in principle move a claim and its payment atomically in seconds rather than T+1 (more efficient transfer). **Programmability**: coupons, dividends, and corporate actions can be encoded so they execute automatically.

But the *legal* nature is unchanged. A tokenised bond is still a bond — a claim on coupons and principal — and if it passes the Howey test (it almost always does), it is still a security subject to the same disclosure rules. The technology changes the *plumbing*, not the *thing*. This is the most important takeaway about the entire crypto-meets-securities collision: tokenisation is not a way around the definition of a security; it is a new way to *implement* one. Standardise a claim, make it transferable, give it an identifier, record it on a ledger — whether that ledger is DTC's mainframe or an Ethereum smart contract, you've built a security, and the 1946 orange-grove test still applies.

## Common misconceptions

**"A stock certificate is the security."** No — the *claim* is the security; the certificate (or the book-entry record) is just its representation. We proved this by dematerialising the certificate out of existence entirely. Today the security is an electronic entry at a depository, and nothing was lost when the paper disappeared. Confusing the wrapper for the claim is the single most common beginner error, and it's exactly the error tokenisation hype repeats — the blockchain entry is a new wrapper, not a new kind of thing.

**"Stocks and bonds are completely different animals."** They're the same animal — a standardised, transferable claim — pointed at different cash flows and sitting at different places in line. A bond claims fixed cash flows and gets paid first; a share claims the residual and gets paid last. That's the entire difference. Hybrids exist precisely because the line between them is a dial, not a switch. Seeing both as members of one family (the security) is the conceptual unlock of this whole post.

**"Liquidity is just a property of big, popular securities."** Liquidity is *manufactured* by standardisation, then *amplified* by a deep buyer pool — it isn't luck, and it isn't only for mega-caps. A small, obscure bond issue can be perfectly liquid if it's fungible and there's a market-maker quoting it; a "famous" asset can be illiquid if every unit is a snowflake (try selling a specific apartment quickly at a fair price). Fungibility comes first; popularity follows.

**"If you call it a token / membership / NFT, it isn't a security."** The Howey test reads economic substance, not labels. An "investment of money in a common enterprise with profit expected from others' efforts" is a security whatever you name it. The 2020s crypto enforcement wave is one long demonstration of this — courts kept finding that re-labelled securities were still securities. The name is wrapping; the four prongs are the law.

**"Owning a fund unit means you own the underlying shares."** You own a claim on a *pro-rata slice of the fund's basket*, not the underlying securities directly — the fund (or its custodian) holds those. Your unit is itself a security, one layer of claim removed from the assets. This recursion (claims on baskets of claims) is a feature, not a bug: it's how diversification gets packaged into a single tradable object.

## How it shows up in real markets

**The primary market manufactures new securities — in booms and freezes.** The clearest evidence that "a security is a claim you can sell" comes from watching what happens to issuance when the *can-you-sell-it* promise wobbles. US IPO proceeds hit about \$142bn in 2021, then collapsed to roughly \$8bn in 2022 — a near-total freeze. Nothing changed about the *legal* nature of shares between those years. What changed was the *secondary*-market backdrop: in the 2022 bear market, investors feared they couldn't resell new shares at a good price, so the implicit liquidity promise lost credibility, and the primary engine stalled. Issuers create securities only when buyers believe the secondary market will be there to absorb them later.

![US IPO proceeds by year, highlighting the 2021 boom and the 2022 freeze](/imgs/blogs/what-a-security-actually-is-claims-you-can-sell-7.png)

This is the spine made visible: the primary market (which *creates* securities) is hostage to the secondary market (which *trades* them). When secondary liquidity is deep and prices are high, primary issuance floods in; when secondary markets seize, primary issuance dries up overnight, even though every company that wanted capital in 2022 still wanted it. The two engines are one machine.

**The debt universe is mostly one issuer's securities.** Look at US debt-security issuance by type and one bar dwarfs everything: the US Treasury issues an order of magnitude more debt securities than corporates, municipals, mortgage-backed securities, or asset-backed securities combined — roughly \$23 trillion of gross Treasury issuance in a year versus around \$1.4tn corporate and \$1.5tn in mortgage MBS. (Treasury's number is huge partly because short-term bills are constantly rolled over.) The Treasury is the most prolific manufacturer of standardised, transferable claims on Earth, and the deep fungible market in its securities is exactly why they're the global benchmark for "safe."

![US debt-security issuance by type in 2023 on a log scale, dominated by Treasury](/imgs/blogs/what-a-security-actually-is-claims-you-can-sell-8.png)

The log scale is doing honest work here — without it, every non-Treasury bar would be a sliver. But the families are all the same *kind of thing*: a Treasury bill, a corporate bond, a muni, an MBS, and an ABS are all standardised, transferable claims on cash flows, differing only in who promises the cash and where you sit in line. The securitization track of this series shows how an MBS or ABS is *built* — pooling thousands of loans into new tradable claims; here, note simply that the entire debt market is the security pattern repeated at every scale.

**T+1 went live and nothing dramatic happened — which was the point.** On 28 May 2024, the US shortened its equity settlement cycle from two business days to one (T+1). This is purely a dematerialisation story: because securities are book-entry records, the transfer is a ledger update, and the question of *how fast* is a question of operational design, not physics. Cutting a day out of settlement reduces the time the system carries counterparty risk and frees up collateral. The fact that the switch was a non-event for retail investors is the deepest proof that the security long ago stopped being a piece of paper — you can't move paper certificates around the country in one day, but you can edit a database in a millisecond. The whole apparatus of clearing and settlement that made this possible is Track D of this series.

**Vietnam shows the same pattern building in fast-forward.** Vietnam's depository (VSDC) holds securities in book-entry form just like DTC; the VN-Index trades dematerialised claims; and the number of securities trading accounts has gone from about 2.2 million in 2018 to over 9 million in 2024 — a buyer pool deepening in real time as more savers convert cash into standardised, transferable claims. The same two inventions, fungibility and transferability, that built the \$255tn global market are building Vietnam's, with the same plumbing (a central depository, ISINs, book-entry transfer) underneath. The pattern is universal because the *problem* it solves — turning bespoke, stuck claims into fungible, sellable ones — is universal.

## The takeaway: the unit is the whole story

If you remember one thing from this post, make it this: **a security is not a kind of asset; it is a kind of engineering applied to a claim.** Take any economic promise — a slice of a company, a loan, a share of a basket — and do two things to it. Standardise it until every unit is identical and they pool into one fungible mass. Make it transferable so any holder can sell to any stranger and pass on the exact same rights. The moment both are true, you've manufactured liquidity out of a bespoke favour, and a long-term project can be funded by a relay of short-term holders who each know they can hand off the baton.

Everything else in this series is built on that unit. The primary market (Track B) is the factory that *manufactures* securities. The secondary market (Track C) is where the fungible units trade and liquidity gets priced. The plumbing (Track D) is what lets a book-entry claim move from one holder to another in a day. The intermediaries (Track E) are the people who run the factory and the trading floor. Securitization (Track F) is the security pattern applied recursively to pools of loans. Regulation (Track G) is the disclosure regime that the Howey test decides applies. And the whole machine exists for one reason: to turn savings into long-term investment, using a unit you can always sell.

So the next time you see a stock ticker, a bond quote, an ETF, or a tokenised Treasury, look past the wrapper and ask the two questions that define the thing: *what cash flow or ownership am I claiming, and can I sell it to a stranger tomorrow?* If the answer to the second is yes, you're holding a security — and the reason it has any value at all is that the answer is yes. The whole quarter-quadrillion-dollar edifice of global capital markets rests on that single, quietly radical word: *transferable*. To see how one such unit travels from a financing idea all the way to delisting, follow [the life of a security from idea to delisting](/blog/trading/capital-markets/the-life-of-a-security-from-idea-to-delisting).

## Further reading & cross-links

- [What is a capital market and how money finds its best use](/blog/trading/capital-markets/what-is-a-capital-market-how-money-finds-its-best-use) — the series intro: the savings-to-investment machine and why a market beats a bank queue.
- [Debt vs equity: the two ways to raise capital](/blog/trading/capital-markets/debt-vs-equity-the-two-ways-to-raise-capital) — the fundamental fork between the two biggest families of claim.
- [The life of a security from idea to delisting](/blog/trading/capital-markets/the-life-of-a-security-from-idea-to-delisting) — the full lifecycle map: issuance, listing, secondary trading, corporate actions, maturity.
- [Why markets are regulated: disclosure and the securities acts](/blog/trading/capital-markets/why-markets-are-regulated-disclosure-and-the-securities-acts) — what the Howey gate opens onto: the 1933/1934 Acts and disclosure-based regulation.
- [Stock exchanges and clearinghouses](/blog/trading/finance/stock-exchanges-and-clearinghouses) — where standardised claims actually trade and settle.
- [The yield curve explained](/blog/trading/fixed-income/the-yield-curve-explained-the-most-important-chart-in-finance) — how the bond claims in this post get *priced*.
