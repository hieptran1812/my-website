---
title: "Su Zhu, Three Arrows Capital, and the Leverage That Broke the Lenders"
date: "2026-07-29"
publishDate: "2026-07-29"
description: "How Three Arrows Capital actually made money, why every crypto lender handed it billions with no collateral, and how one fund's margin call turned into the bankruptcy of the firms holding ordinary people's deposits."
tags: ["crypto", "three-arrows-capital", "leverage", "counterparty-risk", "crypto-lending", "rehypothecation", "margin-call", "contagion", "credit-risk", "crypto-players"]
category: "trading"
subcategory: "Crypto Players"
author: "Hiep Tran"
featured: true
readTime: 50
---

> [!important]
> **TL;DR** — Three Arrows Capital was not brought down by a bad trade. It was brought down by funding long-dated, unsellable assets with short-dated debt that lenders could recall in days, and by being lent that money on reputation rather than collateral.
>
> - The fund's original edge was **spread capture**: buying a thing in one place and selling the identical thing somewhere else for more. Those trades were real, but they were capacity-limited, and they compressed sharply through 2021.
> - What replaced them was **directional conviction funded like arbitrage** — the same aggressive borrowing, now backing positions that could go to zero rather than positions that had to converge.
> - Crypto's lenders extended credit **uncollateralized or thinly collateralized**, with no credit bureau, no consolidated view of the borrower's total leverage, and no way to know how many other lenders existed. Reputation was the collateral.
> - **Rehypothecation** — pledging the same collateral onward down a chain — meant one dollar of real assets backed several dollars of loans. That is what turns one fund's failure into a system event.
> - The number to hold onto: a retail depositor earning 9% on an app was, without being told, an **unsecured creditor of a leveraged hedge fund**. The 9% was not a savings rate. It was a credit spread on a loan they did not know they had made.

There is a version of this story that gets told as a morality tale about hubris, and it is not very useful. Two traders got arrogant, made a huge bet, and lost. The end.

That version explains nothing. Funds lose money constantly; it is an occupational hazard, and the industry is built to absorb it. When a hedge fund blows up in equities or rates, the prime broker seizes the collateral, sells it, takes its loss, and files the incident under Tuesday. The fund's investors are wiped out and almost nobody else notices.

That is emphatically not what happened here. When Three Arrows Capital failed in June 2022, the damage did not stop at the fund. It went straight through to the firms that had lent to it, and then through *those* firms to hundreds of thousands of ordinary people who had deposited money into an app because it advertised a yield. Several of those firms filed for bankruptcy within weeks.

So the question worth asking is not "why did the fund lose money." It is: **why was a single hedge fund's loss able to reach a retail depositor's balance at all?** Answering that requires understanding two separate machines — how the fund made money, and how it was funded — and seeing why the second one was far more dangerous than the first.

This post is the player profile and the operating model. For the general narrative of the contagion as it unfolded across the market, see [Three Arrows Capital and the crypto lender contagion](/blog/trading/crypto/three-arrows-capital-and-crypto-lender-contagion); this one goes underneath it, into the trades and the funding structure that made the contagion possible.

## The machine, in one picture

![A two-column balance sheet showing 3AC's illiquid, long-dated assets on the left and its short-dated recallable liabilities on the right](/imgs/blogs/su-zhu-3ac-and-the-leverage-that-broke-the-lenders-1.webp)

The diagram above is the mental model, and almost everything else in this post is an elaboration of it.

On the left are the things the fund owned. Read down the column and notice that the further you go, the longer it takes to turn any of it into cash — from spot bitcoin you can sell in seconds, down through positions with contractual lockups measured in months, to venture equity measured in years.

On the right are the things the fund owed. Read down *that* column and notice that the longest-dated item is a few weeks.

That is the entire fragility. A firm in this shape is solvent right up until someone asks for their money back, and then it is not — not because the assets are worthless, but because they cannot be converted fast enough to matter. It is the oldest failure mode in finance, and it has a name: a **maturity mismatch**. Banks run one deliberately, which is why they have deposit insurance, capital requirements, and a central bank that will lend against good collateral in an emergency. A crypto hedge fund in 2022 ran the same mismatch with none of those three things.

Everything that follows is about how a firm ends up in that shape while believing it is running an arbitrage book.

## Foundations: leverage, collateral, and what a margin call actually is

If you already know this material, skim to the next section. If you do not, none of the rest will land properly without it, so let us build it from nothing.

### Borrowing against something you own

Start with the simplest possible arrangement. You own an asset — a house, a share, a bitcoin. You want cash but you do not want to sell. So you borrow against it: a lender gives you money, and if you fail to pay it back, the lender takes the asset.

The asset you pledge is **collateral**. The arrangement is a **secured loan** or **collateralized loan** — "secured" meaning the lender has a specific asset to seize, rather than merely a promise.

Two numbers describe every such loan.

The **loan-to-value ratio (LTV)** is what you borrowed divided by what the collateral is worth. Borrow \$15,000 against collateral worth \$30,000 and your LTV is 50%. Lower is safer for the lender.

The **haircut** is the same idea from the other direction: the discount a lender applies to collateral before deciding how much to lend against it. A 50% haircut on \$30,000 of bitcoin means the lender treats it as \$15,000 of lending capacity. Volatile, hard-to-sell assets get bigger haircuts, which is the lender's way of saying *I do not trust that this price will still be there when I need to sell*.

### Leverage

**Leverage** is using borrowed money to control a position larger than your own capital. If you have \$1,000 and borrow \$4,000 to buy \$5,000 of something, you are levered 5-to-1.

Leverage is symmetric in a way people consistently underrate. It multiplies gains and losses by the same factor — a 10% rise on that \$5,000 position is \$500, which is a 50% return on your \$1,000. A 10% fall is a \$500 loss, which is 50% of your capital gone. A 20% fall wipes you out entirely.

But it is asymmetric in a way that matters more: **the losses arrive faster than you can react to them, and someone else decides when you exit.** That is the part beginners miss, and it is the whole story of 2022.

### The margin call

A lender who takes collateral does not simply wait to see whether you repay. The loan agreement contains thresholds — expressed as an LTV, a coverage ratio, or a collateral value — and if the collateral falls through one, specific things happen automatically.

At the first threshold you get a **margin call**: a demand to post more collateral or repay part of the loan, restoring the cushion. At the second, the lender stops asking and simply **liquidates** — sells your collateral, takes what it is owed, and hands you whatever is left.

The animation below is the mechanic that people find most counter-intuitive, so it is worth watching for a moment.

<figure class="blog-anim">
<svg viewBox="0 0 720 420" role="img" aria-label="A collateral bar worth $30,000 falls as bitcoin's price drops; it crosses the $20,000 margin-call line and then the $16,667 liquidation line, while the $15,000 of debt it secures never moves" style="width:100%;height:auto;max-width:820px">
<title>A margin call is a line the collateral falls through, not a decision anyone makes</title>
<style>
.mcH{font:600 16px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937)}
.mcS{font:500 12.5px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280)}
.mcT{font:600 12.5px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937)}
.mcAx{stroke:var(--border,#d1d5db);stroke-width:1.5}
.mcRef{stroke:var(--text-secondary,#6b7280);stroke-width:1.5;stroke-dasharray:6 4}
.mcDebt{stroke:var(--text-primary,#1f2937);stroke-width:2.5}
.mcBar{fill:var(--accent,#6b7280);opacity:.85;transform-box:fill-box;transform-origin:50% 100%}
.mcZone{fill:#ffc9c9;opacity:.34}
.mcWarn{font:700 13px ui-sans-serif,system-ui;fill:#ef4444}
@keyframes mcFall{0%,10%{transform:scaleY(1)}60%,90%{transform:scaleY(.5326)}100%{transform:scaleY(1)}}
@keyframes mcHi{0%,10%{opacity:1}30%,90%{opacity:0}100%{opacity:1}}
@keyframes mcLo{0%,50%{opacity:0}62%,90%{opacity:1}100%{opacity:0}}
@keyframes mcCall{0%,44%{opacity:0}50%,90%{opacity:1}100%{opacity:0}}
@keyframes mcLiq{0%,56%{opacity:0}62%,90%{opacity:1}100%{opacity:0}}
.aFall{animation:mcFall 8s ease-in infinite}
.aHi{animation:mcHi 8s ease-in infinite}
.aLo{animation:mcLo 8s ease-in infinite}
.aCall{animation:mcCall 8s ease-in infinite}
.aLiq{animation:mcLiq 8s ease-in infinite}
@media (prefers-reduced-motion:reduce){.aFall,.aHi,.aLo,.aCall,.aLiq{animation:none}.aLo,.aCall,.aLiq{opacity:1}}
</style>
<text class="mcH" x="20" y="26">You pledged 1 BTC at $30,000 and borrowed $15,000 against it</text>
<text class="mcS" x="20" y="46">The debt is a flat line. Only the collateral moves - and the call fires when the two get too close.</text>
<rect class="mcZone" x="112" y="156" width="470" height="194"/>
<line class="mcAx" x1="112" y1="60" x2="112" y2="350"/>
<line class="mcAx" x1="112" y1="350" x2="700" y2="350"/>
<line class="mcRef" x1="112" y1="59" x2="582" y2="59"/>
<text class="mcT" x="592" y="63">$30,000 - where you started</text>
<line class="mcRef" x1="112" y1="156" x2="582" y2="156"/>
<text class="mcT" x="592" y="160">$20,000 - margin call (75% LTV)</text>
<line class="mcRef" x1="112" y1="188" x2="582" y2="188"/>
<text class="mcT" x="592" y="192">$16,667 - liquidation (90% LTV)</text>
<line class="mcDebt" x1="112" y1="205" x2="582" y2="205"/>
<text class="mcT" x="592" y="209">$15,000 - what you owe</text>
<rect class="mcBar aFall" x="170" y="59" width="120" height="291" rx="4"/>
<text class="mcT aHi" x="300" y="76">collateral: $30,000</text>
<text class="mcT aLo" x="300" y="212">collateral: $16,000</text>
<text class="mcWarn aCall" x="300" y="146">MARGIN CALL - post more or be sold</text>
<text class="mcWarn aLiq" x="300" y="248">LIQUIDATED - the lender sells it for you</text>
<text class="mcS" x="112" y="380">Collateral value of the 1 BTC pledged ($)</text>
<text class="mcS" x="112" y="400">Nobody chose to sell here. The price crossed a number written into the loan agreement, and the sale happened automatically.</text>
</svg>
<figcaption>A margin call is not a phone call and not a judgement. The debt is fixed, the collateral floats, and when the gap between them closes past a threshold written into the contract, the lender sells - whether or not you think the position is right.</figcaption>
</figure>

#### Worked example: where exactly does the call fire?

Let us put numbers on it. All the figures in this walkthrough are illustrative round numbers chosen to be easy to follow.

You own 1 bitcoin. It is worth \$30,000. You pledge it and borrow \$15,000.

- **Starting LTV:** \$15,000 / \$30,000 = **50%**. Comfortable.
- **The margin-call threshold** in your loan agreement is 75% LTV. Your debt is fixed at \$15,000, so the call fires when \$15,000 / collateral = 0.75, which means collateral = \$15,000 / 0.75 = **\$20,000**. Bitcoin falls 33% and you get the call.
- **The liquidation threshold** is 90% LTV. That is collateral = \$15,000 / 0.90 = **\$16,667**. Bitcoin falls 44% and the position is sold out from under you.

Now notice three things that are not obvious.

First, **you were never asked whether you agreed.** There is no negotiation at \$16,667. The sale is mechanical.

Second, **the gap between "uncomfortable" and "gone" is small.** From the call at \$20,000 to liquidation at \$16,667 is a 17% move. In an asset that routinely moves 10% in a day, that is a few hours of margin.

Third — and this is the one that matters for everything below — **your ability to survive the call depends entirely on whether you have something liquid to post.** If you have spare cash, the call is an annoyance. If everything you own is locked up in things you cannot sell today, the call is fatal. The position is not what kills you. The *shape of the rest of your balance sheet* is what kills you.

*The intuition: a margin call converts a paper loss into a forced sale at the worst possible moment, and the only defence against it is liquidity you have not already spent.*

### Lending with no collateral at all

Everything above assumed the lender holds collateral. Now delete that assumption.

An **unsecured** or **uncollateralized** loan is a loan backed by nothing but a promise to repay. If the borrower fails, the lender does not seize an asset — it joins a queue of creditors in a court process and eventually receives some fraction of what it was owed, possibly years later, possibly nothing.

In traditional finance, unsecured lending to a hedge fund is close to unheard of. Hedge funds borrow through a **prime broker** — a bank that lends against the fund's own portfolio, holds that portfolio in custody, marks it daily, and can seize it instantly. The prime broker can do this because it can *see* the collateral; it is sitting in an account the broker controls.

Crypto in 2021 had no equivalent. There was no prime broker holding the fund's whole book. There were many separate lenders, each seeing only its own slice, each unable to verify what the others were doing.

**Counterparty risk** is the risk that the person on the other side of your agreement does not perform. In a collateralized world you manage it with collateral. In an uncollateralized world you manage it with judgement about the borrower — which is to say, with reputation. And reputation, unlike collateral, does not get marked to market daily. It goes from "excellent" to "worthless" in a single afternoon.

### Rehypothecation

One more term, and it is the one that turns an individual failure into a systemic one.

**Rehypothecation** is when a lender who has received collateral pledges that same collateral onward to secure *its own* borrowing. The bitcoin you posted to Lender A does not sit in a vault; A pledges it to B to fund itself, and B may pledge it to C.

This is not inherently fraudulent — it is standard practice in traditional markets, disclosed in the agreements, and it is a large part of why funding markets are cheap. But it has a specific and unavoidable consequence: **the same asset now supports several loans at once.** In good conditions nobody notices. In bad conditions every link in the chain calls simultaneously, on an asset that can only be sold once.

We will do the arithmetic on that in Part 2, because it is the single most important number in this story.

### The vocabulary of the trades

Finally, four terms you need for Part 1.

**Arbitrage**, strictly speaking, is buying and selling the *same* thing in two places at once for a locked-in difference. Genuine arbitrage carries no market risk — you are not betting on direction, you are collecting a spread that must close. In practice most "arbitrage" is *relative value*: two things that are nearly the same, which should converge, but which can diverge further first. The difference between those two sentences is where fortunes disappear.

**Net asset value (NAV)** is what a fund or trust's underlying holdings are actually worth per share. If a trust holds \$100 million of bitcoin and has 10 million shares, its NAV is \$10 a share.

A **premium** or **discount** is the gap between the market price of those shares and NAV. Trading at \$12 against a \$10 NAV is a 20% premium; at \$8 it is a 20% discount. In a well-functioning structure, arbitrageurs create and redeem shares to close that gap. When redemption is impossible, the gap can persist — and widen — indefinitely.

The **basis** is the difference between the futures price of something and its spot (immediate) price. When futures trade above spot, the market is in **contango**, and that gap is a yield you can harvest if you are willing to hold both legs to expiry.

That is the toolkit. Now the business.

## Part 1 — The operating model: how the fund actually made money

Three Arrows Capital was founded by Su Zhu and Kyle Davies, who had met as students and worked as traders before starting the firm; it began well outside crypto, in traditional currency and rates arbitrage, and moved into digital assets later. The fund was incorporated in the British Virgin Islands and run from Singapore. (Details of the founders' biographies come from press profiles written during and after the firm's collapse rather than from any filing, and are best treated as such.)

That trajectory matters because of what it implies about the firm's instincts. This was not a crypto-native operation that discovered leverage; it was a pair of relative-value traders who brought a relative-value playbook — high leverage against convergent positions — into a market that would eventually stop offering convergent positions.

The most common misconception about Three Arrows is that it was always a directional bull-market fund that got lucky and then got unlucky. It was not. For most of its life it ran genuine, unglamorous spread-capture strategies, and it ran them well. Understanding what those trades were is essential, because the reason the fund died is that **the trades stopped working while the funding structure built for them stayed in place.**

### The trust-premium trade

The single most important trade in the 2020–2021 crypto fund landscape was an arbitrage against a closed-end trust structure, and it deserves careful explanation because its mechanics are what made it lethal.

Picture a trust that holds bitcoin and issues shares representing that bitcoin. Now give it three specific structural features:

1. **Shares are created only through a private placement at NAV.** Accredited investors deliver bitcoin (or cash) to the trust and receive newly issued shares worth exactly what they delivered.
2. **Those newly created shares carry a holding period** before they can be sold in the public market — six months under the relevant securities rule.
3. **There is no redemption program.** You cannot hand shares back to the trust and receive the underlying bitcoin. Ever.

Feature 3 is the one that breaks the machine. In a normal exchange-traded fund, if the shares trade above the value of the underlying, an arbitrageur creates new shares, sells them, and pockets the difference — which pushes the price back to NAV. If they trade below, the arbitrageur buys shares, redeems them for the underlying, and sells that. **Two-way creation and redemption is what pins an ETF to its NAV.**

Remove redemption and you have a one-way valve. The structure can absorb demand (by creating shares) but it cannot absorb supply. So when retail demand for bitcoin exposure inside a brokerage account was intense, the shares traded at a large premium to the bitcoin they represented. And when that demand disappeared, nothing existed to close the gap, and the shares fell to a large *discount* — and stayed there.

![The trust-premium round trip: deliver bitcoin at NAV, receive shares, wait out a six-month lockup, then sell into whatever the market offers](/imgs/blogs/su-zhu-3ac-and-the-leverage-that-broke-the-lenders-2.webp)

#### Worked example: the premium round trip and the six-month door

Here is the trade, with illustrative round numbers.

You have \$10,000,000 of bitcoin. You put it into the trust's private placement and receive shares worth \$10,000,000 at NAV.

Now you wait six months. You cannot sell. You cannot redeem. You cannot hedge cheaply, because the thing you are exposed to is not the bitcoin price — you are still long bitcoin either way — it is the *premium*, and there is no instrument that lets you short the premium directly.

Six months later, the lockup lifts:

- **If the shares trade at a 20% premium:** you sell for \$12,000,000. Gross profit \$2,000,000, less the trust's management fee of roughly 1% over six months (about \$100,000) — call it **\$1,900,000 on \$10,000,000 in six months**, or roughly 38% annualized, on a position that felt like arbitrage.
- **If the shares trade at a 20% discount:** you sell for \$8,000,000. You have lost **\$2,000,000**, plus the fee, and you have lost it on a trade you entered because it looked like free money.

And here is the part that makes it dangerous rather than merely risky. The natural way to run this trade is to make it look market-neutral: borrow bitcoin, deliver the borrowed bitcoin into the trust, and short bitcoin futures against the position so you are not exposed to the bitcoin price at all. Do that and your profit is purely the premium — beautiful, until you notice what you have actually built. You have a **six-month non-cancellable commitment**, funded with **borrowed bitcoin that has to be returned**, whose profit depends on a spread that **has no mechanism forcing it to close**.

When the premium flipped to a discount in early 2021, every fund holding this trade discovered the same three facts at once: the position could not be exited, the loss grew with the discount, and the bitcoin they had borrowed to enter it still had to be repaid.

*The intuition: an arbitrage that cannot be closed at will is not an arbitrage. It is a leveraged bet on someone else's demand, with a lockup.*

### The basis trade

The second staple was the cash-and-carry, and it is a genuinely good trade that killed people for a reason worth understanding precisely.

Crypto futures traded persistently above spot through the bull market, because leveraged buyers wanted long exposure and were willing to pay for it. That gap is free money in the following sense: a futures contract must converge to the spot price at expiry. So if you buy the asset and simultaneously sell the future, you have locked in the gap, and you do not care what the price does in between.

![A cash-flow timeline of a levered basis trade showing the locked-in $250,000 profit and the $2.25 million variation-margin call that can arrive first](/imgs/blogs/su-zhu-3ac-and-the-leverage-that-broke-the-lenders-3.webp)

#### Worked example: a levered basis trade and the call that has nothing to do with being wrong

Illustrative numbers again.

Bitcoin is at \$30,000. The three-month future trades at \$30,750 — a 2.5% premium over three months, about 10% annualized.

- You buy **333.3 BTC** of spot for **\$10,000,000**.
- You sell **333.3 BTC** of the three-month future at **\$30,750**.
- You post **\$2,500,000** of margin at the futures venue.

At expiry, whatever the price is, the future converges to spot. Your two legs offset except for the \$750 per bitcoin you locked in: 333.3 × \$750 = **\$250,000**. Against \$2,500,000 of margin that is **10% in three months**, roughly 40% annualized, with no view on direction whatsoever.

The total coming back to you at expiry is worth stating explicitly, because it is what the timeline above shows: 333.3 × \$30,750 = \$10,250,000 from the two legs combined, plus the \$2,500,000 of margin returned, equals **\$12,750,000** against the \$12,500,000 you put in. The \$250,000 difference is the whole trade.

Now break it.

Bitcoin rallies 25% to \$37,500 in week six. Your spot is up \$2,500,000 — excellent. Your short future is down 333.3 × \$6,750 = **\$2,250,000** — a mark-to-market loss on which the venue wants cash **now**, in the form of *variation margin*, the daily settlement of a derivative's gains and losses. Your \$2,500,000 of posted margin is nearly consumed.

The trade is fine. Your net position has not lost a cent; the two legs still offset. But **the gain is in the wrong pocket.** The profit sits in spot bitcoin, possibly at a different exchange, possibly in cold storage, possibly already pledged to somebody else. The loss is a cash demand at the futures venue, due today.

If you cannot move liquidity to the venue in time, you are liquidated on the short leg — at which point you are no longer hedged. You are simply long \$10,000,000 of bitcoin, at the top of a 25% rally, having been forced into that position by a trade designed to have no direction at all.

*The intuition: delta-neutral is not margin-neutral. A hedged book can still die of a liquidity problem, and the liquidity problem always arrives at the venue where you are losing.*

### Cross-venue and cross-border spreads

The third family was geography. The same asset trades at different prices in different jurisdictions when capital cannot move freely between them — the best-known example being the persistent premium on Korean exchanges, where domestic demand was strong and capital controls made it awkward for foreigners to arbitrage the gap away.

These trades are attractive and fundamentally limited, and the table shows why:

| What makes the spread exist | What makes it hard to capture |
| --- | --- |
| Capital controls or banking restrictions between jurisdictions | You need local bank accounts, local entities, and local counsel |
| Fiat on-ramps that are slow or restricted to residents | Settlement takes days, during which the spread can move against you |
| Genuine local demand imbalance | Capacity is capped by how much fiat you can legally move |
| Fragmented venues with no consolidated order book | Each leg has its own withdrawal limits and outage risk |

The important property is the last row of the right column: **these trades do not scale.** A spread that yields handsomely on \$20 million may be entirely uncapturable at \$500 million, because the frictions that create the spread are the same frictions that stop you moving size through it.

Which brings us to the actual mechanism of failure.

### The pivot: when the spreads ran out

By 2021 something structural had happened. Capital had flooded into exactly these trades. The trust premium had inverted to a discount. The futures basis had compressed as more balance sheet chased it. The cross-border spreads had narrowed as the infrastructure matured.

A fund that had grown large on spread capture now faced a problem with no good answer: **the strategies that justified the size no longer had the capacity to support it.**

There are three honest responses. Return capital to investors. Accept much lower returns. Or shrink.

There is a fourth response, and it is the one that recurs in every version of this story across every asset class: **keep the leverage, keep the return target, and change the trades.** Replace convergent positions — where two prices *must* meet, so time is your ally — with directional positions, where you are simply right or wrong and time is neutral at best.

This is the hinge of the entire post, so let me be precise about what changes:

| | Spread capture | Directional conviction |
| --- | --- | --- |
| **Source of profit** | A gap that must close | Being right about the future |
| **Effect of time** | Works for you — convergence is contractual | Neutral or against you |
| **Worst case** | The spread widens temporarily, then converges | The thesis is wrong and the position goes to zero |
| **Appropriate leverage** | High — the position self-corrects | Low — nothing rescues you |
| **Right funding** | Short-term is fine | Long-term, locked-up capital |

Read the bottom two rows. Spread capture *can* justify aggressive short-term borrowing, because the position converges on a known date. Directional conviction cannot, because there is no date and no convergence.

What happened in 2021–2022 across the industry was that the top row changed while the bottom two rows did not. Funds moved into concentrated directional positions — including illiquid ones with contractual lockups and staked assets that could not be redeemed at all — while continuing to fund those positions with the short-dated, recallable, uncollateralized borrowing that made sense for arbitrage.

Look again at the first figure. That is what the left column becoming long-dated while the right column stayed short-dated looks like.

## Part 2 — The funding side: why everyone lent without collateral

Here is the genuinely strange part of the story, and the part with the most transferable lesson: the fund's trades were only half the problem. The other half is that a dozen firms handed it enormous sums with little or no collateral, and none of them knew about the others.

### Credit without a credit system

![Two panels contrasting what lenders believed they held against what they actually held](/imgs/blogs/su-zhu-3ac-and-the-leverage-that-broke-the-lenders-4.webp)

To see how this happened, compare what a traditional prime broker does with what a crypto lender in 2021 actually did.

| Function | Traditional prime broker | Crypto lender, 2021 |
| --- | --- | --- |
| **Custody of the collateral** | Holds the fund's portfolio itself | Often held nothing at all |
| **Marking positions** | Daily, on its own systems | Relied on borrower-reported figures |
| **View of total leverage** | Sees the whole book it finances | Sees only its own loan |
| **Cross-lender visibility** | Regulatory reporting, credit bureaus, ISDA infrastructure | None |
| **Seizing collateral** | Instant, contractual, operationally routine | A claim in an offshore court |
| **Competitive pressure** | Priced on risk | Priced on winning the client |

Every row is bad, but the last two are the ones that turn bad into catastrophic.

Consider the position of a credit officer at a crypto lender in 2021. Deposits are pouring in from retail customers who have been promised a yield. That yield is a contractual obligation — the money must be put to work or the business loses money on every dollar it takes in. The pool of borrowers who can absorb hundreds of millions of dollars is tiny. And the largest, most respected borrower in that pool is asking for a loan and mentioning that a competitor has offered better terms.

Demanding full collateral means losing the loan. Losing the loan means holding idle deposits that cost you 8% a year. **The competitive dynamics of the industry actively selected against prudence** — the lender with the strictest standards did the least business and looked worst to its own investors, right up until the moment it looked best.

And because no lender could see the others, each one was evaluating a borrower whose *total* leverage was unknowable. A firm might reasonably conclude it was comfortable lending \$200 million to a counterparty it believed was worth billions. Ten firms reaching that same reasonable conclusion independently produce a borrower with \$2 billion of unsecured debt and no one who knows it.

This is the mechanism, and it does not require anyone to behave badly. **Ten individually defensible credit decisions can compose into an indefensible aggregate**, and in a market with no shared credit infrastructure, nothing exists to catch it.

### How one dollar of collateral becomes three loans

Now add rehypothecation, and the picture gets materially worse.

![A chain showing one bitcoin worth $30,000 pledged onward twice, ultimately backing $54,000 of loans across three lenders](/imgs/blogs/su-zhu-3ac-and-the-leverage-that-broke-the-lenders-5.webp)

#### Worked example: the collateral chain

Illustrative numbers, one asset, three lenders.

1. The fund pledges **1 BTC worth \$30,000** to Lender L1 and borrows **\$20,000** against it.
2. L1 does not vault that bitcoin. L1 has its own funding needs, so it **repledges the same bitcoin** to Lender L2, borrowing **\$18,000**.
3. L2 does the same thing, pledging it onward to L3 for **\$16,000**.

Total credit extended against one bitcoin: \$20,000 + \$18,000 + \$16,000 = **\$54,000**.

Real collateral backing it: **\$30,000**.

That is **56 cents of asset per dollar of credit**, and note carefully — *before the price has moved at all*. This is not a stress scenario. This is the calm, everything-is-fine state of the system.

Now let bitcoin halve to \$15,000. The single asset securing \$54,000 of loans is worth \$15,000. All three lenders' thresholds breach in the same hour. All three issue calls. All three have a contractual right to the same bitcoin, which can be sold exactly once.

Two consequences follow, and both showed up in 2022.

**First, losses exceed the visible exposure.** A lender that believed it had \$20,000 of secured exposure discovers it has an unsecured claim and a legal argument about who owns the collateral.

**Second, and worse: the calls are synchronized.** Rehypothecation does not merely amplify leverage — it *correlates* the timing of every margin call in the chain. There is no staggering, no lender who calls first and gets out cleanly while others wait. Everyone finds out simultaneously, and everyone tries to sell the same asset into the same order book in the same hour.

*The intuition: rehypothecation converts one borrower's default into several lenders' defaults, all arriving at the same moment, backed by an asset that can only be sold once.*

### The yield had to come from somewhere

There is one more piece of the funding structure, and it is where ordinary people enter the story.

The lenders were not deploying their own capital. They were deploying customer deposits, gathered by advertising yields — 8%, 9%, sometimes more — on assets that pay nothing on their own. Bitcoin has no coupon. A dollar stablecoin has no dividend. Every point of that yield had to be *manufactured* by lending the deposits to someone who would pay for them.

So the chain ran: depositor to platform, platform to fund, fund into positions. And the yield the depositor received was, definitionally, a share of the interest a leveraged hedge fund was paying to borrow.

This is not a scandal in itself. It is how banks work. The difference is that a bank tells you it is a bank, is capitalized to absorb losses, is supervised, and — in most jurisdictions — insures your deposit up to a limit. The apps offering 9% did none of those four things, and many of their customers believed they were using a savings account.

We will come back to what that depositor was actually holding, because it is the most important practical lesson in this post. But first, the sequence of events that turned the structure into a set of bankruptcies.

## Part 3 — The cascade

A note on sourcing before we start, because it matters more here than anywhere else in this post.

The mechanisms described above are structural — they are how leverage, collateral and rehypothecation work, and the numbers used to illustrate them are deliberately round and hypothetical. What follows is different: it concerns real firms, real defaults and real bankruptcies. So the standard changes. Below, figures are given **only where a primary document supports them**, and where the most widely repeated numbers in circulation could not be traced to a filing, I say so rather than repeat them. A great deal of what "everyone knows" about this episode traces back to a handful of contemporaneous news reports citing unnamed sources, and some of it does not survive contact with the court record.

### The trade that had no exit: staked ETH

Of all the positions on the left column of that first figure, one deserves its own walkthrough, because it demonstrates the maturity mismatch more cleanly than anything else — and because the arithmetic is genuinely surprising.

When you stake ether to help secure the Ethereum network, your ether is locked. A **liquid staking token** solves the inconvenience: you deposit ether with a staking protocol and receive a token — stETH being the largest — that represents your staked position and can be traded freely. You get the staking yield *and* something tradeable.

Now the crucial structural fact. In June 2022, that token could not be redeemed for the underlying ether. Not by anyone, at any price, for any reason. Withdrawals from the Ethereum staking contract simply did not exist yet; the ability to withdraw staked ether arrived only with the Shanghai/Capella upgrade in 2023, the better part of a year later.

So stETH had exactly the property we met in Part 1 with the trust shares: **a one-way valve**. You could always create it by staking; you could not redeem it. Its price was therefore set entirely by whoever was willing to buy it on the secondary market — and the arbitrage that would normally pin it to ether, buying the discounted token and redeeming it for the real thing, was structurally impossible.

For most of its life this did not matter, and the token traded within a whisker of ether. Then it became collateral. Deposit stETH into a lending protocol, borrow ether against it, buy more stETH, deposit that too — a loop that manufactures leveraged staking yield and is enormously popular right up until the discount moves.

#### Worked example: the staked-ETH squeeze

Illustrative round numbers, chosen to make the arithmetic legible.

You hold **\$100,000,000** of stETH. You have borrowed **\$70,000,000** against it — a 70% LTV. Your lender liquidates at 100% LTV, meaning the moment the collateral is worth less than the debt.

![A column chart showing collateral value under four scenarios crossing below the $70 million debt line](/imgs/blogs/su-zhu-3ac-and-the-leverage-that-broke-the-lenders-6.webp)

Watch what two simultaneous moves do:

1. **Start:** no discount, ether flat. Collateral \$100,000,000 against \$70,000,000 of debt. Comfortable.
2. **A 3% discount opens, ether flat.** Collateral = \$100,000,000 × 0.97 = **\$97,000,000**. Still fine. Nothing has happened to ether at all — this is purely the market repricing the token's illiquidity.
3. **The discount widens to 7% and ether falls 20%.** Collateral = \$100,000,000 × 0.93 × 0.80 = **\$74,400,000**. Now you are at 94% LTV and the calls are arriving.
4. **Ether falls 25% instead of 20%, same 7% discount.** Collateral = \$100,000,000 × 0.93 × 0.75 = **\$69,750,000** — below the \$70,000,000 you owe. **Liquidation, and the position is already underwater.**

The lesson is in the multiplication. Two moves that each look survivable — a 7% discount and a 25% price fall — compose into a 30.25% collateral loss, because they multiply rather than add. Leverage of 70% LTV has no room for that.

And now the part that makes it a trap rather than merely a loss. What do you do when the call arrives? You cannot redeem the stETH for ether — the withdrawal mechanism does not exist. Your only exit is to *sell stETH into the secondary market*, which is the very market whose thin bid created the discount in the first place. Every unit you sell widens the discount, which lowers the collateral value of every remaining unit, which brings the next liquidation closer.

This is the loop from the previous figure, in one asset: **the act of meeting the margin call makes the margin call worse.** And because every large leveraged holder of the same token faces the identical arithmetic at the identical moment, they all reach for the same exit together.

*The intuition: an asset that cannot be redeemed is only worth what today's buyer will pay, and if you are levered against it, the discount alone can liquidate you without the underlying price doing anything unusual at all.*

### One borrower, many lenders, one hour

<figure class="blog-anim">
<svg viewBox="0 0 720 400" role="img" aria-label="A single defaulting fund sends a shock outward to four different kinds of lender at once, and each of those lenders passes it on to its own depositors" style="width:100%;height:auto;max-width:820px">
<title>One borrower, many lenders: how a single default arrives everywhere at the same time</title>
<style>
.ctH{font:600 15.5px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937)}
.ctS{font:500 12px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280)}
.ctL{font:600 11.5px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937);text-anchor:middle}
.ctBox{fill:var(--surface,#f3f4f6);stroke:var(--border,#d1d5db);stroke-width:1.5}
.ctFund{fill:#a5d8ff;stroke:var(--border,#d1d5db);stroke-width:1.5}
.ctEdge{stroke:var(--border,#d1d5db);stroke-width:1.5;stroke-dasharray:5 4}
.ctDot{fill:#ef4444}
@keyframes ctA{0%{transform:translate(0,0);opacity:0}3%{opacity:1}11%{transform:translate(-265px,68px);opacity:1}14%{opacity:0}100%{opacity:0}}
@keyframes ctB{0%{transform:translate(0,0);opacity:0}3%{opacity:1}11%{transform:translate(-85px,68px);opacity:1}14%{opacity:0}100%{opacity:0}}
@keyframes ctC{0%{transform:translate(0,0);opacity:0}3%{opacity:1}11%{transform:translate(95px,68px);opacity:1}14%{opacity:0}100%{opacity:0}}
@keyframes ctD{0%{transform:translate(0,0);opacity:0}3%{opacity:1}11%{transform:translate(275px,68px);opacity:1}14%{opacity:0}100%{opacity:0}}
@keyframes ctDrop{0%{transform:translate(0,0);opacity:0}3%{opacity:1}10%{transform:translate(0,76px);opacity:1}13%{opacity:0}100%{opacity:0}}
@keyframes ctHit{0%,9%{fill:var(--surface,#f3f4f6)}14%,92%{fill:#ffc9c9}100%{fill:var(--surface,#f3f4f6)}}
.aA{animation:ctA 8s linear infinite}
.aB{animation:ctB 8s linear infinite;animation-delay:.4s}
.aC{animation:ctC 8s linear infinite;animation-delay:.8s}
.aD{animation:ctD 8s linear infinite;animation-delay:1.2s}
.aDrop{animation:ctDrop 8s linear infinite;animation-delay:2.6s}
.aDrop2{animation-delay:2.9s}
.aDrop3{animation-delay:3.2s}
.aDrop4{animation-delay:3.5s}
.aHit{animation:ctHit 8s linear infinite}
.aHit2{animation-delay:.4s}
.aHit3{animation-delay:.8s}
.aHit4{animation-delay:1.2s}
.aHitD{animation:ctHit 8s linear infinite;animation-delay:2.6s}
.aHitD2{animation-delay:2.9s}
.aHitD3{animation-delay:3.2s}
.aHitD4{animation-delay:3.5s}
@media (prefers-reduced-motion:reduce){.aA,.aB,.aC,.aD,.aDrop,.aHit,.aHitD{animation:none}.aA,.aB,.aC,.aD,.aDrop{opacity:0}}
</style>
<text class="ctH" x="20" y="22">One borrower defaults. Four lenders find out in the same hour.</text>
<rect class="ctFund" x="280" y="34" width="160" height="48" rx="8"/>
<text class="ctL" x="360" y="55">The fund</text>
<text class="ctL" x="360" y="72">misses its margin calls</text>
<line class="ctEdge" x1="360" y1="82" x2="95" y2="150"/>
<line class="ctEdge" x1="360" y1="82" x2="275" y2="150"/>
<line class="ctEdge" x1="360" y1="82" x2="455" y2="150"/>
<line class="ctEdge" x1="360" y1="82" x2="635" y2="150"/>
<rect class="ctBox aHit" x="20" y="150" width="150" height="54" rx="8"/>
<text class="ctL" x="95" y="172">Retail yield platform</text>
<text class="ctL" x="95" y="190">lent depositors' coins</text>
<rect class="ctBox aHit aHit2" x="200" y="150" width="150" height="54" rx="8"/>
<text class="ctL" x="275" y="172">Institutional lender</text>
<text class="ctL" x="275" y="190">lent unsecured</text>
<rect class="ctBox aHit aHit3" x="380" y="150" width="150" height="54" rx="8"/>
<text class="ctL" x="455" y="172">Exchange</text>
<text class="ctL" x="455" y="190">extended margin</text>
<rect class="ctBox aHit aHit4" x="560" y="150" width="150" height="54" rx="8"/>
<text class="ctL" x="635" y="172">Another fund</text>
<text class="ctL" x="635" y="190">held its paper</text>
<line class="ctEdge" x1="95" y1="204" x2="95" y2="280"/>
<line class="ctEdge" x1="275" y1="204" x2="275" y2="280"/>
<line class="ctEdge" x1="455" y1="204" x2="455" y2="280"/>
<line class="ctEdge" x1="635" y1="204" x2="635" y2="280"/>
<rect class="ctBox aHitD" x="20" y="280" width="150" height="46" rx="8"/>
<text class="ctL" x="95" y="308">Depositors</text>
<rect class="ctBox aHitD aHitD2" x="200" y="280" width="150" height="46" rx="8"/>
<text class="ctL" x="275" y="308">Depositors</text>
<rect class="ctBox aHitD aHitD3" x="380" y="280" width="150" height="46" rx="8"/>
<text class="ctL" x="455" y="308">Traders on margin</text>
<rect class="ctBox aHitD aHitD4" x="560" y="280" width="150" height="46" rx="8"/>
<text class="ctL" x="635" y="308">Its own investors</text>
<circle class="ctDot aA" cx="360" cy="82" r="6"/>
<circle class="ctDot aB" cx="360" cy="82" r="6"/>
<circle class="ctDot aC" cx="360" cy="82" r="6"/>
<circle class="ctDot aD" cx="360" cy="82" r="6"/>
<circle class="ctDot aDrop" cx="95" cy="204" r="5"/>
<circle class="ctDot aDrop aDrop2" cx="275" cy="204" r="5"/>
<circle class="ctDot aDrop aDrop3" cx="455" cy="204" r="5"/>
<circle class="ctDot aDrop aDrop4" cx="635" cy="204" r="5"/>
<text class="ctS" x="20" y="356">Nobody in the bottom row lent to the fund. They lent to someone who did, which is the same exposure with an extra step hiding it.</text>
<text class="ctS" x="20" y="376">The second wave is the one that matters: the shock does not stop at the lender, it stops at whoever was funding the lender.</text>
</svg>
<figcaption>A concentrated borrower turns four separate lending decisions into one position. When it defaults, the losses do not arrive in sequence with time to react between them - they arrive everywhere at once, and then pass straight through to the depositors who never knew they were exposed.</figcaption>
</figure>

The animation above is the structural point, and it is why this failure was different in kind from an ordinary fund blow-up.

Because the borrowing was unsecured and spread across many lenders who could not see one another, **there was no sequence.** No lender got a warning from watching another lender's problem. There was no first mover who seized collateral and exited cleanly while the rest queued. Every counterparty discovered its exposure in the same window, and each then had to answer the same question at the same time: *do I have enough capital to absorb this?*

For several of them the answer was no. And because those firms were themselves funded by customer deposits rather than by shareholders' capital, their answer became their depositors' problem — the second wave in the figure, and the one that turned a hedge fund's failure into a consumer event.

### The case study: what the Genesis filings actually establish

The best-documented single link in the chain runs from the fund to **Genesis Global Capital**, and it is worth walking through carefully, because it demonstrates every mechanism in this post at once — and because the court record says something more precise, and more interesting, than the headline numbers that circulated at the time.

According to the New York Attorney General's amended complaint in its action against Gemini Trust Company, Genesis Global Capital and Digital Currency Group, **Three Arrows Capital defaulted on 13 June 2022** on loans from Genesis Asia Pacific running to billions of dollars.

Three details in that record do more work than any headline figure.

**First, the collateral was illiquid, and Genesis knew it.** The complaint states that Genesis "accepted from Three Arrows illiquid collateral to secure more than \$500 million in loans." Read that against Part 1 of this post. This is the maturity mismatch arriving at the lender rather than the borrower: the security Genesis held was precisely the kind of asset that cannot be sold quickly at a fair price, which means it provided protection in exactly the scenarios where protection was not needed, and none in the scenario that actually happened.

**Second, the specific asset named is the one from our worked example.** An internal communication quoted in the complaint states that "\$500 [million] of the collateral we absorbed to offset the [Three Arrows] losses was GBTC which isn't liquid." That is the trust structure from Part 1 — the one with the six-month lockup and no redemption program — showing up as collateral on a lender's balance sheet. Recall what the discount does in that structure: it is set by whoever will buy today, and there is no redemption mechanism to close it. A lender seizing that collateral in June 2022 was seizing an asset whose price was being set by other forced sellers.

**Third, notice what the collateral did not do.** It did not prevent the loss. It converted an unsecured exposure into a partly secured one whose security could not be monetised at anything like its marked value. This is the practical content of the phrase "thinly collateralized", and it is why the distinction between *collateralized* and *usefully collateralized* is the one that matters.

Now the part that requires discipline. The figure most often quoted for Genesis's total exposure to Three Arrows is **\$2.36 billion**. That number could not be traced to a primary filing for this post — the amended complaint's own language is "billions", without the decimal. It is repeated widely in contemporaneous press coverage, and it may well be right. But the honest formulation is that the complaint establishes *billions of dollars* and specifically identifies *more than \$500 million* secured by illiquid collateral, and that anyone citing \$2.36 billion is citing reporting rather than a document. The same caution applies to several other numbers attached to this episode — total creditor claims against the fund, the size of each lender's individual loss, and the fund's assets under management at its peak. Those figures circulate with a confidence the underlying documentation does not always support.

That last one deserves a sentence of its own, because it is the most-repeated number in the whole story. The widely cited **"\$10 billion under management"** is a claim that was made *about* the fund during the bull market, not a figure from an audited statement or a court filing. It should always be read as "reported at the time as", never as an established fact — and the gap between what a fund was said to manage in 2021 and what its liquidators actually found in 2022 is itself one of the lessons of the episode.

### Two jurisdictions, two different proceedings

One structural point that is easy to garble, and often is. The insolvency proceeding and the personal court order against the founders happened in **different jurisdictions**, and they are different things:

- The **liquidation of the fund** was a British Virgin Islands proceeding — the fund was a BVI-incorporated entity, and that is where the winding-up and the appointment of liquidators sat, with recognition sought in the United States so the liquidators could pursue assets there.
- The **committal order against Su Zhu** was a **Singapore** matter, and it concerned **non-cooperation with the liquidators** — a contempt finding about failing to comply with court orders to assist the insolvency process. It was not a finding of fraud, and it should never be described as one.

That distinction is not pedantry. "Sentenced" invites readers to assume a fraud conviction, and the record does not support that reading. A contempt sanction for failing to cooperate with a liquidation is a serious thing and a different thing.

### The pattern across the lenders

Rather than assign each firm a loss figure this post cannot source, here is the structural pattern the bankruptcies shared — which is the transferable part anyway:

| What each lender had in common | Why it mattered |
| --- | --- |
| Lending to the same concentrated borrower | Their exposures were one position, not many |
| Little or no visibility into that borrower's other lenders | Each sized its loan against a leverage number it could not see |
| Collateral that was illiquid, correlated, or absent | Security that fails precisely in the scenario it exists for |
| Funding from customer deposits, not shareholder capital | Losses passed through to depositors instead of stopping at equity |
| Retail-facing yields that had to be paid regardless | Pressure to keep lending on weakening terms |

Read down the right-hand column and the outcome stops looking like a series of separate corporate failures and starts looking like what it was: **one exposure, held by several firms, funded by the public.**


## How it shows up in price

A forced seller is a different animal from a willing seller, and the difference is visible in the tape if you know what to look for.

A willing seller has a reservation price. If the bid is too low, they wait. That patience is what makes normal markets orderly — it means supply withdraws as prices fall, which is the mechanism that stops falls.

A forced seller has no reservation price. The contract says sell, so they sell, at whatever the book will pay, immediately. Supply *increases* as prices fall, because lower prices trigger more liquidations. That is the opposite of the stabilizing mechanism, and it is why the loop below is self-feeding rather than self-correcting.

![A closed feedback loop showing margin calls driving liquid asset sales, price falls in thin books, lower collateral marks, and further margin calls](/imgs/blogs/su-zhu-3ac-and-the-leverage-that-broke-the-lenders-7.webp)

Four fingerprints in particular are worth learning to recognize, because they recur in every deleveraging event, not just this one.

**The good assets go first.** This is the most counter-intuitive signature of a credit event, and it confuses people every time. When a levered book needs cash urgently, it cannot sell the locked, illiquid, or impaired positions — those are precisely the ones with no bid. So it sells the *liquid* things: the major assets, the ones with deep order books, the positions the manager likes most. Which means that in the first hours of a forced unwind, **the assets that fall hardest are often the ones with the least wrong with them.** If you are trying to infer what a distressed seller owns from what is falling, you will reliably get the answer backwards.

**Discounts on redemption-blocked assets widen far beyond fundamentals.** When an asset cannot be redeemed for the thing it represents, its price is set purely by who is willing to buy it today. Put a forced seller into that market and the discount is bounded only by the depth of the bid. This is the same one-way-valve problem from Part 1, seen from the other end: the mechanism that let the premium persist is the mechanism that lets the discount blow out.

**Liquidity evaporates precisely when it is needed.** Market makers widen spreads and pull size when volatility spikes, because their inventory risk rises — see [what a crypto market maker actually does](/blog/trading/crypto-players/what-a-crypto-market-maker-actually-does) for why this is rational rather than treacherous. The practical effect is that the order book is thinnest at the exact moment the largest forced orders arrive. Summer, when desks are short-staffed, makes it worse.

**Correlations go to one.** In calm markets, different assets have different stories. In a deleveraging, everything owned by the same levered holders falls together, because the thing being sold is not any particular asset — it is *leverage itself*. Diversification measured on calm-market data disappears exactly when you were relying on it.

The general version of this, and how it applies beyond credit events, is in [how crypto prices actually move](/blog/trading/crypto-players/how-crypto-prices-actually-move).

## How retail ended up short a hedge fund's leverage

Now we can answer the question this post opened with: why did a hedge fund's loss reach a retail depositor at all?

![A chain from a $10,000 retail deposit through a lending platform to an uncollateralized hedge fund loan, with an unsecured claim returning to the depositor](/imgs/blogs/su-zhu-3ac-and-the-leverage-that-broke-the-lenders-8.webp)

Follow the chain in the figure. You deposit \$10,000 into an app quoting 9%. The platform lends it onward at 12–13%, keeping the spread as its revenue. The borrower posts no collateral. The borrower deploys the money into positions that cannot be sold quickly.

Now ask the question that matters: **what do you actually own?**

Not bitcoin. Not dollars. You own an *unsecured claim against the platform*, whose principal asset is an *unsecured claim against a hedge fund*, whose assets are locked positions in a falling market. You are three steps removed from anything real, and at every step you rank behind somebody holding actual collateral.

This is the crucial distinction, and it is genuinely not obvious from the user interface:

| | A bank deposit | A crypto yield account |
| --- | --- | --- |
| **What you hold** | A claim on a supervised, capitalized institution | A claim on an operating company |
| **Where the yield comes from** | Disclosed, regulated lending, with capital held against it | Undisclosed borrowers, leverage unknown to you |
| **If the borrower defaults** | The bank's capital absorbs it | Your principal absorbs it |
| **If the institution fails** | Deposit insurance up to a limit | You are an unsecured creditor in bankruptcy |
| **Who can see the risk** | Supervisors, auditors, published capital ratios | Nobody outside the firm |

Both rows look like "money that earns interest" in an app. Only one of them is.

#### Worked example: what a 9% yield is actually pricing

This is the single most useful calculation in this post, because it works on any yield, in any market, forever. It requires no inside information at all.

Interest rates decompose into two parts: what you can earn essentially risk-free, and the extra you demand for the chance of not being repaid. That second part is the **credit spread**, and it is not a bonus. It is the market's price for the probability that you lose your money.

Both numbers here are illustrative. Take an advertised yield of 9%, and suppose the risk-free rate is 2% — short-dated government bills were in the low single digits during this period, and the exact figure does not change the shape of the answer. The credit spread is then **7 percentage points**.

Now invert it. In a fair market, the spread compensates for expected loss, so:

**spread ≈ probability of default × loss given default**

- **If a default means you lose everything** (loss given default = 100%), then a 7% spread implies a **7% annual default probability**. Roughly a 1-in-14 chance, per year, that your deposit is gone.
- **If you would recover 40%** in a bankruptcy, then 7% = p × 0.60, so p = **11.7%** — about a 1-in-9 chance per year.

Sit with that for a second. The platform is not hiding the risk. It is *advertising* it, in the only honest number on the page. A yield of 9% against a 2% risk-free rate is the market saying: **there is roughly a one-in-ten chance per year that this counterparty does not return your money.**

The mistake was never failing to spot a conspiracy. It was reading a credit spread as an interest rate.

And note the asymmetry, because it is brutal. Being right earns you 7 extra points. Being wrong costs you 60–100 points. You need to be right about nine to fourteen times in a row to break even on a single failure — and the years when everyone offers 9% are precisely the years when the failures correlate.

*The intuition: a yield is a price, and what it prices is the chance you do not get paid. If you cannot name the borrower and the trade, the yield is telling you something you have decided not to hear.*

## The red flags that were visible from the outside

None of the following required access to anyone's books. All of it was in public terms of service, marketing pages, and the arithmetic above.

![A six-row matrix of observable red flags, what each actually meant, and the question to ask](/imgs/blogs/su-zhu-3ac-and-the-leverage-that-broke-the-lenders-9.webp)

**A yield with no nameable source.** Run the calculation above on any advertised rate. Then ask what trade generates it. If the answer is a generic phrase — "our institutional lending desk", "market-neutral strategies" — that is not an answer, and the absence of a specific answer is itself the finding. Every real yield has a payer and a trade.

**Undisclosed borrowers.** You cannot assess concentration you cannot see. The question is not "who are your borrowers" — you will not get names — but "what percentage of the loan book goes to your largest single borrower?" A firm that will not answer that in a range has told you what you need to know.

**No proof of collateral.** Ask what share of the loan book is collateralized, and at what LTV. This is a number every lender knows and can disclose without naming anyone. Silence here is the single strongest signal in the list, because it is the cheapest question to answer honestly.

**Terms that permit lending your assets.** This is usually disclosed, in the terms of service, in plain language, and almost nobody reads it. Language granting the platform the right to pledge, lend, or rehypothecate customer assets converts you from a custodial client into a general creditor. That is not a subtle distinction — it determines whether your coins are yours in a bankruptcy or part of the estate.

**Yield paid in the platform's own token.** This creates a circular dependency: the payout depends on the price of an asset the platform controls and whose value depends on the platform's health. When the firm is stressed, the token falls, so the effective yield falls, so depositors leave, so the firm is more stressed. It is a reflexive loop that fails exactly when you need it not to.

**Withdrawal frictions described as maintenance.** Unplanned withdrawal delays, new limits, "network upgrades", or processing times that lengthen without notice are the earliest observable symptom of a liquidity gap. By the time a firm announces a problem, the problem is usually weeks old.

One structural point ties these together. Notice that every item above is about **the funding side, not the trading side**. Retail depositors spent 2022 studying the assets — is bitcoin going up, is this protocol safe — when the thing that actually determined whether they got their money back was the capital structure of the intermediary holding it. Analysing the asset is the wrong analysis when your claim is on the intermediary.

## Common misconceptions

**"They lost money on a bad trade."** Funds lose money on bad trades constantly and it rarely matters to anyone else. What made this different was the funding structure: unsecured, short-dated, spread across lenders who could not see each other, backing assets that could not be sold. A fund with the same positions and locked-up investor capital would have suffered a terrible year and survived it.

**"The lenders were reckless."** Some were. But the deeper problem is that the industry's competitive structure actively punished caution — the lender demanding full collateral lost the business to the one that did not, and had to explain the lower returns to its own investors. Individual prudence was not enough, because the risk was created by the *aggregate* of many separately reasonable decisions. That is a structural failure, not a character failure, and it is why it repeats.

**"Collateral makes a loan safe."** Collateral makes a loan safer, which is not the same thing. Collateral is only worth what you can sell it for, in the size you hold, at the moment everyone else is selling the identical thing. Collateral that is correlated with the borrower's solvency — crypto collateral against a crypto borrower — provides the least protection exactly when it is needed most. And rehypothecated collateral may be backing several loans at once, so "we hold collateral" and "we hold *exclusive* collateral" are very different claims.

**"Delta-neutral means safe."** A hedged position can be arithmetically riskless and still fail, because being right about the price is not the same as having cash at the venue that is demanding it today. The basis-trade example above loses nothing on paper and can still be liquidated. Nearly every large leverage failure is a liquidity failure wearing a solvency costume.

**"This was a crypto problem."** The mechanism is universal — it is a maturity mismatch, funded unsecured, amplified by rehypothecation, with no consolidated view of the borrower's leverage. The same structure caused the 1998 failure of Long-Term Capital Management, whose lenders also each saw only their own slice. What was distinctive about 2022 was not the mechanism but the absence of the shock absorbers traditional markets had built after learning this lesson repeatedly: no central clearing, no capital requirements, no lender of last resort, and depositors who did not know they were creditors.

**"You needed inside information to see it coming."** The yield calculation above requires a published rate and a Treasury yield. The terms permitting rehypothecation were in the terms of service. The question about collateralized share of the loan book required only asking. What was genuinely invisible from outside was the *aggregate* leverage of any single borrower — but you did not need that number to know the structure was fragile.

## When this matters to you, and further reading

The transferable lesson has nothing to do with crypto and everything to do with a habit: **when someone offers you a return, work out who is paying it and what has to be true for them to keep paying.**

That habit generalises. It applies to a corporate bond fund yielding four points above governments, to a private credit vehicle promising equity-like returns with bond-like volatility, to a stablecoin paying yield on a dollar, and to any product whose marketing emphasises the return and describes the source in the passive voice. In every case the same three questions do most of the work:

1. **Who is the borrower, and what trade pays this?** If nobody will say, the yield is compensation for a risk you have not been shown.
2. **Am I a customer or a creditor?** Read what happens to your assets in an insolvency. This is written down, usually clearly, and it changes everything.
3. **What is the maturity of their funding versus the maturity of their assets?** Anyone borrowing short to hold long is one confidence shock away from failure, no matter how good the assets are.

For the market-wide narrative of how the 2022 contagion spread, see [Three Arrows Capital and the crypto lender contagion](/blog/trading/crypto/three-arrows-capital-and-crypto-lender-contagion). For the collapse that triggered the sequence, [the Terra-Luna collapse](/blog/trading/crypto/terra-luna-2022-collapse) covers the algorithmic-stablecoin mechanism in detail. For the failure that followed later the same year and rhymed with it, [FTX and Sam Bankman-Fried](/blog/trading/crypto/ftx-collapse-sam-bankman-fried) and [Alameda Research, the cautionary tale](/blog/trading/crypto-players/alameda-research-the-cautionary-tale) cover what happens when the borrower and the venue are the same firm. And for the wider map of who extends credit and takes risk in this market, [the hidden power structure of crypto](/blog/trading/crypto-players/the-hidden-power-structure-of-crypto) and [Pantera, DCG and the crypto conglomerates](/blog/trading/crypto-players/pantera-dcg-and-the-crypto-conglomerates) are the natural next reads.

*This post is educational, not investment advice. It describes mechanisms and history; it does not recommend any position, platform, or asset.*

## Sources & further reading

**On sourcing standards in this post.** Every worked example above uses deliberately round, hypothetical numbers and is labelled as such — those are illustrative arithmetic, not claims about what any firm actually held. Claims about real firms and real events are made only where a primary document supports them. Where a widely circulated figure could not be traced to a filing, the post says so explicitly rather than repeating it; the \$2.36 billion Genesis exposure figure and the "\$10 billion under management" figure are both handled that way in Part 3.

**Primary documents**

- **New York Attorney General, amended complaint** in the action against Gemini Trust Company, LLC, Genesis Global Capital, LLC and Digital Currency Group. This is the source for the 13 June 2022 default by Three Arrows on loans from Genesis Asia Pacific, for the statement that Genesis "accepted from Three Arrows illiquid collateral to secure more than \$500 million in loans", and for the internal communication that "\$500 [million] of the collateral we absorbed to offset the [Three Arrows] losses was GBTC which isn't liquid."
- **British Virgin Islands liquidation of Three Arrows Capital Ltd** — the winding-up proceeding and the appointment of joint liquidators, together with the recognition proceedings brought in the United States to allow the liquidators to pursue assets there. The liquidators' reports to creditors are the authoritative source for claim totals and recoveries; treat any claim figure not drawn from them as reporting.
- **Singapore court committal proceedings against Su Zhu** — a contempt matter concerning non-cooperation with the liquidators. Note again that this is distinct from the BVI insolvency proceeding and is not a fraud finding.
- **Monetary Authority of Singapore**, media releases concerning Three Arrows Capital Pte Ltd (the reprimand) and the subsequent prohibition orders against the founders. MAS publishes these at `mas.gov.sg/news`.
- **US Chapter 11 dockets** for the lenders that failed in 2022–23 — Voyager Digital, Celsius Network, BlockFi and Genesis Global Capital. First-day declarations are the most useful single document in each case, and the dockets are hosted publicly by the claims agents (Stretto for Voyager and Celsius; Kroll for BlockFi and Genesis).
- **Ethereum protocol upgrade documentation** — for the Shanghai/Capella upgrade that first enabled withdrawals of staked ether, establishing that stETH could not be redeemed for ether during the June 2022 episode described in Part 3.
- **SEC EDGAR** — for the structure of the Grayscale Bitcoin Trust described in Part 1 (creation by private placement at net asset value, the Rule 144 holding period applying to privately placed shares, and the absence of a redemption program), the trust's own registration statements and periodic reports are authoritative, as are any Schedule 13G filings disclosing large holders.

**Context and reporting**

Contemporaneous coverage from Bloomberg, the Financial Times, Reuters, The Block and CoinDesk carries most of the widely quoted figures for this episode. That reporting is valuable for chronology and for quotes, but a good deal of it rests on unnamed sources, and several of its numbers have not been confirmed by any filing. Where this post and that reporting differ, the difference is deliberate.

**Related posts on this site**

- [Three Arrows Capital and the crypto lender contagion](/blog/trading/crypto/three-arrows-capital-and-crypto-lender-contagion) — the market-wide narrative of how the failure spread
- [The Terra-Luna collapse](/blog/trading/crypto/terra-luna-2022-collapse) — the algorithmic-stablecoin failure that opened the sequence
- [FTX and Sam Bankman-Fried](/blog/trading/crypto/ftx-collapse-sam-bankman-fried) and [Alameda Research, the cautionary tale](/blog/trading/crypto-players/alameda-research-the-cautionary-tale) — what happens when the borrower and the venue are the same firm
- [Pantera, DCG and the crypto conglomerates](/blog/trading/crypto-players/pantera-dcg-and-the-crypto-conglomerates) — the conglomerate structure behind Genesis
- [The hidden power structure of crypto](/blog/trading/crypto-players/the-hidden-power-structure-of-crypto) — who actually extends credit and takes risk in this market

