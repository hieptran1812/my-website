---
title: "MiFID II and the EU Market-Structure Regime: Research Unbundling and Transparency"
date: "2026-06-19"
publishDate: "2026-06-19"
description: "How Europe's MiFID II rewrote the economics of an entire industry: by forcing research to be unbundled from execution and mandating pre- and post-trade transparency, it gutted sell-side research budgets, reshaped where liquidity trades, and opened a US-versus-EU regulatory-divergence trade."
tags: ["regulation", "market-structure", "mifid-ii", "research-unbundling", "transparency", "best-execution", "dark-pools", "systematic-internalisers", "soft-dollars", "small-caps", "europe", "trading"]
category: "trading"
subcategory: "Law & Geopolitics"
author: "Hiep Tran"
featured: true
readTime: 44
---

> [!important]
> **TL;DR** — MiFID II, in force across the European Union from January 2018, is the cleanest case study of how one market-structure rule can change the economics of an entire industry: by forcing investment research to be paid for separately from trade execution, and by mandating pre- and post-trade transparency, it shrank sell-side research budgets, thinned out coverage of smaller companies, pushed trading into new venue types, and split the world into two regulatory regimes you can trade against each other.
>
> - **Unbundling** banned the old practice of paying for research invisibly inside the trading commission. Once the price of research became a visible, explicit budget line, asset managers cut it — industry estimates put the drop in sell-side research spend at roughly **a third** over the following years.
> - The cut fell hardest on **small and mid caps**: analysts dropped the low-commission names first, so coverage of smaller companies collapsed while mega-cap coverage barely moved. Less coverage meant **wider spreads and thinner liquidity** — but also a wider **mispricing gap** for whoever does the work.
> - The **transparency** rules and the **double volume cap** on dark trading did not make Europe more "lit." Flow migrated into **systematic internalisers** and **periodic auctions** that the cap does not touch.
> - The one number to remember: a US asset manager can still bundle research into commissions under **Section 28(e)** soft-dollar protection, while an EU manager cannot — a hard regulatory divergence that any global firm has to staff and pay for twice.

On 3 January 2018, after a delay that the entire European financial industry had begged for, the second Markets in Financial Instruments Directive — universally called **MiFID II** — finally took effect across the European Union. Most retail investors never noticed. There was no crash, no headline number, no single dramatic price move to point at. And yet within eighteen months, thousands of sell-side analysts had lost their jobs, hundreds of small European companies had lost all analyst coverage, the way a billion-euro fund paid for the ideas it traded on had been rebuilt from scratch, and a quiet but permanent gap had opened between how the United States and Europe regulate the plumbing of their stock markets.

This is what makes MiFID II such a perfect teaching case. It is the rare rule whose *whole purpose* was to change the economics of an industry — not to ban a product or punish a fraud, but to re-price a service that had been mispriced for decades by hiding its cost. When you force a hidden cost into the daylight, people look at it, question it, and cut it. That single mechanism — *make the invisible visible, and watch the budget shrink* — is the spine of this entire post, and it is one of the most reliable patterns in all of financial regulation.

Our series spine runs **law and geopolitics → economic policy → macro and flows → asset prices → the trade**. MiFID II is the law. The "policy" it encodes is a belief that bundled pricing was a conflict of interest that hurt the end investor. The "flows" it reshaped are research budgets, analyst headcount, and where liquidity actually trades. The "prices" it moved are small-cap spreads and the valuation gap on under-covered names. And "the trade" — the payoff every post in this series builds toward — is twofold: harvesting the mispricing that lost coverage leaves behind, and positioning around the US-versus-EU regulatory divergence. Let's build it from zero.

![Before and after columns showing bundled research cost hidden in commission versus unbundled research paid from an explicit budget](/imgs/blogs/mifid-ii-and-the-eu-market-structure-regime-1.png)

## Foundations: bundling, soft dollars, and what MiFID II actually changed

Before we can trade the consequences, we need five building blocks, each defined from zero: what "bundling" was, what "soft dollars" meant, what MiFID II's unbundling mandate said, what "transparency" means in a market-structure context, and the handful of new venue types the rules created. None of this requires a finance background — just patience to build each idea on the last.

### What "bundling" was: paying for research inside the commission

Start with the everyday version. Imagine you hire a stockbroker to buy and sell shares for you. Every time they execute a trade, they charge a **commission** — a small fee, historically a few cents per share. That fee is the price of *execution*: the act of routing your order to an exchange, getting it filled, and confirming it.

Now here is the trick that defined the industry for fifty years. The big investment banks — the **sell side**, so called because they sell research and execution services to investors — did not only execute trades. They also employed armies of **analysts**: people who studied companies, built financial models, published research reports, hosted calls with company management, and arranged investor meetings. That research is genuinely valuable to a fund manager deciding what to buy. But the banks did not send a separate invoice for it. Instead, they **bundled** the cost of research *into* the trading commission.

So when a fund manager traded through a bank and paid, say, 5 cents per share in commission, perhaps 1.5 cents of that paid for the actual execution and 3.5 cents was, in effect, payment for the research the bank provided. The research looked free. It was not free. It was being paid for out of the *fund's* trading commissions — which is to say, out of the *end investor's* money, because trading costs are deducted from the fund's returns. The investor was paying for research and could not see how much.

The technical name for this arrangement is **"soft dollars"** (or "soft commissions"). A "hard dollar" is money a manager pays out of its own pocket — its own profit-and-loss account. A "soft dollar" is a benefit the manager receives that is paid for indirectly through client trading commissions. Research bought with soft dollars cost the manager nothing directly; it cost the client invisibly.

The arrangement was not some shady back-room deal; it was the *normal, accepted, decades-old* operating model of the entire equity business. It grew up in an era of *fixed* commissions, when brokers competed on service rather than price and "free" research was simply how they differentiated themselves. When fixed commissions were abolished and execution costs collapsed toward near-zero, the research subsidy buried in the commission became, proportionally, an ever-larger share of a shrinking fee — and ever harder to justify on its own merits. By the 2010s, the buy side was paying billions of euros a year in soft commissions for research, and almost no one could say with any precision how much, for what, or whether it was worth it. Regulators looked at this and saw the textbook definition of a problem: a large, recurring payment that nobody priced, nobody itemised, and the ultimate payer could not see.

European regulators framed the issue with a specific legal word: **inducement**. An inducement is any benefit a firm receives that could bias it away from acting in the client's best interest. Free research, in the regulators' view, was an inducement — it gave the manager something of value (research it would otherwise have to buy) in a way that was tied to where it sent its trades, creating exactly the conflict that fiduciary duty is supposed to prevent. MiFID II's solution was blunt: research is an inducement *unless* it is paid for in one of the two clean ways (manager's own P&L, or a disclosed Research Payment Account). Anything else is banned. That single reclassification — from "normal service" to "banned inducement" — is the legal hinge on which the whole industry turned.

### Why bundling was a conflict of interest

Spend a moment on *why* regulators came to hate this. Three problems compound:

- **The cost was hidden.** The end investor could not see, and therefore could not challenge, how much of their money was being spent on research. A cost you cannot see is a cost you cannot control.
- **It created an incentive to over-trade.** Because research was paid for *per trade*, a manager who wanted more research had a reason to trade *more* — generating more commissions to "pay" for the research, even when trading more was bad for the client. The phrase for churning a portfolio to generate commissions is *over-trading*, and bundling quietly rewarded it.
- **It distorted the price of research.** Nobody knew what research was actually worth, because nobody paid a clean price for it. Banks gave research away to win trading business; managers consumed far more than they would if each report carried a sticker price.

The fix, in the abstract, is simple: separate the two. Make the manager pay one clean price for execution and a separate, explicit price for research. That separation is **unbundling**, and it is the heart of MiFID II.

### The unbundling mandate

MiFID II — a *directive*, meaning a piece of EU law that each member state writes into its own national rulebook — required, from January 2018, that investment research be paid for either:

1. **directly by the asset manager out of its own P&L** (true hard dollars — the manager eats the cost), or
2. **from a dedicated, ring-fenced Research Payment Account (RPA)**, a pot of money agreed with the client in advance, with a budget set, disclosed, and reconciled, so the client can see exactly what is being spent on research.

What the rule *banned* was the old default: research paid for as an undifferentiated lump inside the execution commission. Crucially, the manager now had to assign research a *price* and decide, line by line, whether each piece was worth it. The mechanism we flagged at the top — make the invisible visible, watch the budget shrink — was now law.

The figure above shows the before-and-after money flow. On the left (bundled, pre-2018), the asset owner's money flows to the manager, the manager pays one combined commission to the broker, and the research desk is funded out of that commission — the cost is hidden the whole way down. On the right (unbundled), the manager must split the payment into two cheques: a thin execution-only commission and an explicit research budget that now sits as a visible line in the profit-and-loss account, where it gets negotiated down.

### Pre-trade and post-trade transparency

Unbundling is the famous half of MiFID II, but the directive's other half is just as consequential: a sweeping expansion of **transparency**.

In market-structure language, transparency comes in two flavours:

- **Pre-trade transparency** means the public can see *orders before they execute*: the bids and offers sitting in the order book, with prices and sizes. A fully "lit" exchange shows you the whole queue.
- **Post-trade transparency** means the public can see *trades after they execute*: price, size, and time, reported promptly to a public tape.

Why does this matter? Because transparency is the raw material of **price discovery** — the process by which a market figures out what something is worth. If everyone can see the orders and the trades, prices reflect more information and reflect it faster. MiFID II vastly widened the universe of instruments and venues that had to publish pre- and post-trade data, dragging into the light a great deal of trading — especially in bonds and derivatives — that had previously happened in the dark.

### The double volume cap on dark trading

A **dark pool** is a trading venue with *no* pre-trade transparency: you can send an order and it can match, but the order book is invisible. Dark pools exist for a good reason — a big institution trying to sell a million shares does not want to broadcast its intent, because front-runners would push the price down before it finished. Dark trading lets large orders execute without moving the market against themselves.

MiFID II did not ban dark pools, but it leashed them with the **double volume cap (DVC)**: for any given stock, no more than **4%** of total trading could happen in a single dark venue, and no more than **8%** across *all* dark venues combined, on a rolling basis. Breach the cap and dark trading in that stock gets suspended for six months. The intent was to push trading back into the light. As we will see, it mostly pushed trading *sideways* into venues the cap did not cover.

### Systematic internalisers and best execution

Two more terms complete the toolkit.

A **systematic internaliser (SI)** is a bank or firm that, instead of sending your order to a public exchange, fills it *against its own book* — it takes the other side itself, "internalising" the trade. SIs are bilateral (you versus the bank), not on-exchange, but under MiFID II they must publish quotes and meet certain transparency duties. They became, as we will see, the principal escape valve when the dark cap bit.

**Best execution** is the legal duty a broker owes you to get the *best possible result* on your order — not just the best price, but the best combination of price, cost, speed, and likelihood of execution, and the size and nature of the order. The principle predates MiFID II, but the directive sharpened it in a way that interacts directly with unbundling. Once research could no longer be paid for inside the commission, the *only* thing the commission was now buying was execution — so the manager had no excuse for routing a trade anywhere except where execution was genuinely best. Under bundling, a manager could quietly justify routing trades to a bank with great research even if its execution was mediocre, because the research was "part of the deal." Unbundling severed that justification: execution must now stand on its own, judged purely on execution quality. Best execution and unbundling are therefore two halves of the same idea — strip out the cross-subsidy so each service is bought, and judged, on its own merits.

To make best execution auditable, MiFID II (in its original form) forced firms to publish two sets of reports: **RTS 27**, in which trading venues published granular execution-quality statistics, and **RTS 28**, in which investment firms disclosed their top five execution venues per asset class and the quality obtained. The theory was that sunlight would discipline routing. In practice, the reports were enormous, almost nobody read them, and the data was nearly impossible to compare across venues. They became a byword for compliance box-ticking — and a later review eased or scrapped them, a first concrete instance of the rollback debate we will reach at the end. The lesson is one worth filing away for every market-structure rule: a transparency mandate only works if someone can actually *use* the data, and a report nobody reads imposes cost without producing the discipline it promised.

### The US link: Section 28(e) and the soft-dollar safe harbour

One reason MiFID II matters far beyond Europe is what it did *not* match on the other side of the Atlantic. In the United States, paying for research with soft dollars is explicitly *protected*. **Section 28(e)** of the US Securities Exchange Act of 1934 is a "safe harbour": it says a manager does *not* breach its fiduciary duty by paying more than the lowest commission, *provided* the extra pays for genuine "brokerage and research services." (For the foundations of the 1934 Act and the SEC's authority, see [Securities Law 101](/blog/trading/law-and-geopolitics/securities-law-101-the-33-and-34-acts-and-the-sec).)

So in 2018, the world split. An EU manager *must* unbundle. A US manager *may* still bundle, protected by 28(e). For a while the SEC issued **no-action relief** letting US brokers accept hard-dollar research payments from EU-facing clients without being treated as investment advisers — a bridge so global firms could comply with MiFID II without tripping US rules. That relief was time-limited, and its eventual lapse forced firms to confront the divergence head-on. The gap between the two regimes is not a bug to be ironed out; it is a permanent feature, and a tradeable one.

## How unbundling collapsed research budgets

With the foundations in place, follow the money. The mechanism is almost mechanical: the moment research had to carry an explicit price, the people paying for it asked a question they had never been forced to ask before — *is this worth it?* — and for most of what they were consuming, the honest answer was *no, not at this price.*

![Bar chart showing sell-side equity research budgets indexed to 100 in 2017 falling to about 65 by 2022](/imgs/blogs/mifid-ii-and-the-eu-market-structure-regime-6.png)

The chart above tells the headline story with illustrative figures consistent with industry estimates (Frost Consulting, Coalition Greenwich): global sell-side equity research spending fell by roughly a third in the years after MiFID II. The exact number is impossible to pin precisely — the whole point of the old regime is that nobody measured it cleanly — but every serious estimate lands in the same zone: a structural decline of 20% to 40% in what the buy side pays the sell side for research.

The decline came through three channels at once:

- **Headcount.** Banks and independent research houses cut analysts. When clients will pay for, say, your three best analysts on a sector but not your tenth-ranked, the marginal analysts go.
- **Price compression.** The price the buy side was willing to pay per analyst, per report, per management meeting, collapsed once it was negotiated explicitly. Some large funds, faced with a price list, simply paid for far less than they used to consume for "free."
- **Absorption into P&L.** Critically, most large asset managers chose to pay for research out of *their own* pocket (hard dollars) rather than set up Research Payment Accounts and bill clients. Once it was *their* money, not the client's, they spent far less. A cost on someone else's tab is generous; a cost on your own is scrutinised.

The result is the single cleanest demonstration of the make-it-visible-and-it-shrinks principle in modern finance. Nothing about the *value* of research changed on 3 January 2018. Companies were no harder to analyse; the work was no less useful. The only thing that changed was that the price became visible — and a visible price is a price you cut.

### The pricing problem nobody could solve

A subtle but important consequence is that nobody actually knew how to *price* research once they had to. Under bundling, the price was whatever the commission happened to be; unbundling demanded an explicit number, and the industry discovered there was no agreed way to set one. How much is a single analyst report worth? A phone call with an analyst? A meeting with a company's chief executive arranged by the bank? An invitation to a conference? Each of these is "research" under MiFID II, and each had to be priced and paid for.

The early scramble produced wildly different answers. Some banks set headline subscription prices for "premium" research access in the tens of thousands of euros; a few floated six- and seven-figure numbers for top-tier macro research, partly to discover what the market would bear. The buy side, suddenly writing real cheques, pushed back hard — and prices fell fast. Within a couple of years a kind of equilibrium emerged in which a large fund might pay a bank a modest five- or six-figure annual sum for a research relationship that, under the old regime, had implicitly cost it far more. The mismatch between the banks' opening prices and what clients would actually pay is itself the proof that research had been radically over-consumed when it looked free.

### How the sell side restructured

Faced with collapsing research revenue, the industry reshaped itself in three directions:

- **Full-service banks cut and concentrated.** Rather than cover everything, banks narrowed to the sectors and names where they could charge — typically large caps with deep institutional interest — and shed analysts on the long tail. Research became a cost centre to be managed, not a loss leader to be lavished.
- **Independent research boutiques gained ground.** Firms that did nothing but sell research, with no execution arm and therefore no bundling to unwind, could offer a clean, priced product — exactly what the new regime demanded. Some specialist independents, particularly in small-cap and niche-sector coverage, found that unbundling *levelled the playing field* against the bulge-bracket banks they had always competed with at a disadvantage.
- **Research marketplaces and issuer-sponsored research emerged.** New platforms sprang up to price, distribute, and meter research consumption à la carte. And a controversial workaround appeared: **issuer-sponsored research**, in which the *covered company itself* pays a provider to publish coverage — solving the coverage gap, but introducing an obvious conflict, since the company funding the research has every incentive to want it positive. For an investor, sponsored research is a signal to read with care, not at face value.

The through-line: an industry that had run for half a century on a hidden cross-subsidy had to learn, almost overnight, to charge an honest price — and the act of charging an honest price shrank it by a third.

#### Worked example: a fund's research budget before and after unbundling

Take a mid-sized European equity fund running €5 billion. Before MiFID II, it traded its portfolio with an annual turnover that generated, say, €15 million in trading commissions. Of that, suppose 60% — €9 million — was effectively the research portion bundled into the commission, and 40% — €6 million — was true execution cost. The research looked free, because it never appeared as a line item; it was buried in the €15 million commission deducted from fund returns.

After unbundling, the manager must split this. Clean execution-only commissions for the same trading volume might run €6 million — call it unchanged, since the execution work is the same. Now the research must be paid separately. Faced with a price list for the first time, the manager decides it does not need €9 million of research; it needs the top providers only, and chooses to pay them out of its own P&L. It sets a research budget of €3.5 million.

The arithmetic:

```
Before: total drag on the fund  = 6.0  (execution) + 9.0  (research, hidden)  = 15.0  EUR mn
After:  total drag on the fund  = 6.0  (execution) + 0.0  (research now hard $) =  6.0  EUR mn
        cost moved to manager P and L                                          =  3.5  EUR mn
Research spend cut: from 9.0 (client-funded) to 3.5 (manager-funded) = a 61 percent fall
Cost to the END INVESTOR: 15.0 -> 6.0 = a 60 percent reduction in trading drag
```

The end investor's trading costs dropped from €15 million to €6 million (roughly \$16.2 million to \$6.5 million at about 1.08 USD per EUR). The manager's own profit took a €3.5 million — about \$3.8 million — hit it never used to bear. And the sell side lost €9 million of revenue from this one client, replaced by €3.5 million paid by the manager directly — a 61% cut. The intuition: unbundling did not destroy €9 million of value; it revealed that, at an honest price, the buyer wanted far less than it consumed for free.

## The unintended consequence: coverage and liquidity for small caps

Here is where MiFID II's neat theory met messy reality. The reformers wanted lower costs and cleaner incentives — and on costs, they largely succeeded. But cutting research budgets is not surgical. When a bank cuts its analyst team, *which* companies lose coverage? Not the giant, heavily traded mega caps — those generate huge commissions and every fund wants research on them, so they keep their analysts. The companies that lose coverage are the *small and mid caps*: the names that trade thinly, generate little commission, and were only ever covered because research was "free" to give away.

![Causal chain showing visible research cost leading to budget cuts that drop small-cap coverage, widening spreads and draining liquidity while opening a mispricing gap](/imgs/blogs/mifid-ii-and-the-eu-market-structure-regime-2.png)

The figure traces the chain. Visible research cost forces budget cuts; the cuts fall on small caps first because they were the least commercially viable to cover; lost coverage widens bid-ask spreads (fewer analysts means less attention and slower price discovery) and drains liquidity (fewer investors even *know* the name exists); and — the one bright spot — the very same coverage gap opens a *mispricing* opportunity for anyone willing to do the analysis themselves.

![Grouped bar chart comparing average analyst coverage by company size before and after MiFID II, with the loss concentrated in small and micro caps](/imgs/blogs/mifid-ii-and-the-eu-market-structure-regime-7.png)

The second chart makes the concentration explicit, again with illustrative figures consistent with the CFA Institute's 2019 member survey and subsequent coverage studies. Large caps barely moved — they had, and kept, twenty-plus analysts each. Mid caps slipped. But small and micro caps were gutted: a small cap that had five or six analysts before might have three or four after, and a micro cap that had two might drop to one or zero. Hundreds of the smallest listed European companies became **"orphans"** — listed on a public exchange, but followed by nobody.

This matters far beyond the egos of the companies involved, because **coverage feeds liquidity**, and the link is causal, not coincidental:

- An analyst publishing on a stock *creates demand for information* about it — investors read the note, build a view, and trade. No analyst, no note, no flow.
- Coverage *legitimises* a name for institutional buyers, many of whom have rules against holding stocks with no sell-side coverage.
- Coverage *narrows the bid-ask spread* by reducing the uncertainty a market maker faces. A market maker quoting an uncovered stock demands a wider spread to compensate for the risk of trading against someone who knows more than they do (the "adverse selection" problem). Less public research means more information asymmetry means wider spreads.

So the chain completes: less research → less coverage → wider spreads → less liquidity → and, in a vicious loop, even less reason for any analyst to cover the name. For small caps, MiFID II's cost-saving triumph was, simultaneously, a liquidity catastrophe.

It is worth dwelling on the **adverse-selection** mechanism, because it is the precise reason coverage and spreads are mechanically linked rather than vaguely associated. A market maker — the firm that stands ready to buy and sell a stock, quoting a bid and an offer — makes money on the spread but loses money when it trades against someone better informed. If a hedge fund knows the company is about to miss earnings and sells to the market maker at the bid, the market maker is left holding stock that is about to fall. To protect itself against this, the market maker *widens the spread*: the less it knows, the bigger the cushion it demands. Public analyst research is precisely what *narrows* the information gap between the market maker and the rest of the market — it puts a shared, public estimate of fair value on the table, so the market maker is less afraid of being picked off. Remove the research, and the market maker is trading blind; it widens the spread to compensate, and every investor in the name pays that wider spread on every trade. This is why the coverage-to-liquidity link is not a soft correlation but a hard, modellable consequence of information economics — and why MiFID II's small-cap coverage cut translated so directly into wider small-cap spreads.

#### Worked example: the liquidity cost of lost coverage

A European small cap, "NordTech," trades at €20 a share. Before MiFID II it had four covering analysts and a typical bid-ask spread of 30 basis points — that is, 0.30% of price, or about 6 cents (bid €19.97, offer €20.03). After unbundling, three of the four analysts drop coverage. With one analyst left and far less institutional attention, the spread widens to 90 basis points — 0.90%, about 18 cents (bid €19.91, offer €20.09).

Now price the cost to an investor who wants to build a €2 million position and later sell it. The spread is a *round-trip* cost: you buy at the offer and eventually sell at the bid, so you pay roughly the full spread once on the way in and once on the way out — but the standard way to quote it is the half-spread each way, which sums to one full spread on a round trip.

```
Position size                          = 2,000,000 EUR
Before: round-trip spread cost = 0.30% x 2,000,000  =  6,000 EUR
After:  round-trip spread cost = 0.90% x 2,000,000  = 18,000 EUR
Extra cost caused purely by lost coverage           = 12,000 EUR per round trip
```

Building and exiting the position now costs €12,000 — about \$13,000 — more than it would have — money that vanishes into the spread, paid to market makers for bearing the extra uncertainty of an uncovered name. The intuition: lost coverage is not free even for the patient long-term holder; it shows up as a permanent tax every time you trade the stock.

## The transparency rules and the migration to new venues

Now to MiFID II's other half. The transparency push and the double volume cap were meant to drag trading into the light. What actually happened is a textbook lesson in how markets route around a rule: when you constrain one channel, the flow does not disappear — it finds the nearest channel you did *not* constrain.

![Five stacked tiers from fully lit regulated market down through systematic internaliser and periodic auction to dark pool under the volume cap](/imgs/blogs/mifid-ii-and-the-eu-market-structure-regime-4.png)

The figure ranks the venue types by transparency. At the top, the fully **lit** regulated markets and multilateral trading facilities (MTFs) publish their entire order book — they are the reference price everyone else leans on. Below them sit **systematic internalisers**, which must quote but trade bilaterally against their own book. Below those, **periodic auction books** run very short lit auctions — a few milliseconds — that achieve near-dark execution while technically staying inside the transparency rules. Then **large-in-scale (LIS) waivers** let genuinely big block trades skip pre-trade quotes (with no cap, because a block is exactly the order dark trading exists to protect). And at the bottom, true **dark pools**, now squeezed by the 4%/8% double volume cap.

When the dark cap began suspending dark trading in capped stocks in 2018, the obvious prediction was that flow would climb back up to the lit books at the top. It did not. Instead it slid *sideways and down* the stack into the venue types the cap does not touch.

![Flow diagram showing dark pool volume hitting the double volume cap and migrating into systematic internalisers, periodic auctions and large-in-scale blocks rather than back to lit books](/imgs/blogs/mifid-ii-and-the-eu-market-structure-regime-5.png)

This is the migration in one picture. Dark flow hits the cap; rather than returning to the public order book, it reroutes into **systematic internalisers** (which boomed after 2018 precisely because they sit outside the dark cap), into **periodic auctions** (which grew from a curiosity to a meaningful slice of volume), and into **large-in-scale blocks** (which are exempt by design). The net effect: European trading did *not* become dramatically more lit. A large share of volume simply moved from one form of low-transparency execution to another. The rule changed the *plumbing* far more than it changed the *outcome* — a humbling result for the reformers, and a crucial one for any trader trying to find liquidity, because you now have to know which venue type your flow is hiding in.

#### Worked example: the value of being the marginal analyst on an under-covered name

This is the alpha side of the coverage collapse, and it is the heart of the playbook. Return to a small cap that has lost most of its coverage. With few or no analysts watching, the market's estimate of fair value can drift far from reality, because there is nobody publishing the work that would correct it. If *you* do that work, you can be the marginal informed buyer — and capture the gap when the price eventually catches up.

Take "Adriatic Foods," a small cap trading at €12, which lost all coverage after 2018. Doing the analysis yourself, you build a model that says fair value is €16 — a 33% discount, sustained precisely *because* no analyst is publishing a target to anchor the market. You buy €1 million of stock at €12.

Two years later, the company's growth becomes undeniable, a broker re-initiates coverage with a €16 target, institutional buyers who screen for "covered names only" can now hold it, and the price converges to €16.

```
Entry: 1,000,000 EUR at 12 EUR  ->  83,333 shares
Exit:  83,333 shares at 16 EUR  =  1,333,333 EUR
Gross gain                       =    333,333 EUR  = +33%
Less wider-spread trading cost (about 0.9% round trip) ~ 9,000 EUR
Net gain                         ~    324,000 EUR  ~ +32%
```

A net gain of about €324,000 — roughly \$350,000 on the \$1.08 million you put in — a 32% return, sourced not from a market call but from a *structural* feature of the post-MiFID II landscape: the coverage gap that unbundling created is, for the diligent, a standing supply of mispriced names. The intuition: regulation that drove analysts *out* of small caps handed an edge to investors willing to do the analyst's job themselves, because their information now has scarcity value.

## The US-versus-EU divergence and the cross-border compliance headache

Step back to the biggest-picture consequence, the one that turns a European market-structure rule into a global trade. After 2018, two of the world's largest capital markets ran on incompatible research-payment regimes, and that incompatibility is permanent enough to position around.

![Matrix comparing the US and EU regimes across research payment, transparency, dark trading, best execution and coverage effect](/imgs/blogs/mifid-ii-and-the-eu-market-structure-regime-3.png)

The matrix lays the two regimes side by side. On **research payment**, the US allows bundling under Section 28(e) soft-dollar protection; the EU requires unbundling from an explicit budget. On **transparency**, the US runs Reg NMS quotes plus a consolidated post-trade tape; the EU mandates broad pre- and post-trade transparency across many venue types. On **dark trading**, US dark pools face no hard volume cap; the EU imposes the 4%/8% double volume cap. On **best execution**, both impose a duty, but the EU layered on the RTS 27/28 reporting (since eased). And on **coverage**, the US kept research broadly intact because soft dollars still fund it, while EU small-cap coverage fell sharply. (For the US market-structure rules in detail — Reg NMS, payment for order flow, short-selling rules — see [Market-Structure Law](/blog/trading/law-and-geopolitics/market-structure-law-reg-nms-pfof-and-short-selling-rules).)

For a *global* asset manager — one running money for both US and European clients out of overlapping desks — this divergence is not abstract. It is a daily operational problem:

- The same analyst's research might be paid for by soft dollars for the US book and by an explicit RPA or hard dollars for the EU book — *for the same report*.
- The firm must build systems to track which client, in which jurisdiction, consumed which research, and bill it the right way.
- A US broker accepting hard-dollar payments from an EU manager risked being reclassified as an investment adviser under US law — the exact problem the SEC's no-action relief was meant to solve. When that relief lapsed, firms had to restructure how cross-border research was paid for, or stop providing it across the line.
- Compliance, legal, and operations headcount all rose to manage the seam between two regimes that will never align.

#### Worked example: the compliance cost of running US and EU books

A global asset manager runs a US book and an EU book off the same research and trading infrastructure. Pre-MiFID II, it paid for all research with soft dollars under a single, simple regime. Post-2018, it has to operate two regimes in parallel.

Estimate the incremental annual cost of the divergence itself — not the research, but the *machinery* to comply with two incompatible rulebooks:

```
Dedicated MiFID II / research-payment compliance staff: 6 FTE x 180,000 USD = 1,080,000 USD
Legal + advisory (cross-border structuring, no-action analysis)            =   400,000 USD
Systems build + maintenance (RPA tracking, research consumption ledger)    =   600,000 USD
EU research now paid hard-dollar from own P and L (was client soft dollars) = 3,000,000 USD
                                                                  Total      = 5,080,000 USD per year
```

Roughly \$5 million a year, every year, purely to sit astride the divergence — most of it the research the firm now eats rather than passes to clients, plus the compliance scaffolding to keep the two books legal. The intuition: regulatory divergence is not free even for the firms big enough to absorb it; it is a permanent fixed cost that favours scale and quietly squeezes smaller managers who cannot spread €5 million across enough assets.

## The partial rollback and the rebundling debate

Here is the twist that makes MiFID II a *living* case study rather than a closed historical one: by the early 2020s, the regulators who built it were quietly admitting parts of it had overshot, and pieces of the regime began to be unwound. For a practitioner, the rollback is not a footnote — it is an active, datable catalyst.

The trigger was the small-cap coverage collapse. Policymakers across Europe came to see thin small-cap coverage as a problem for *capital formation* itself: if small and growing companies cannot attract analyst coverage, they struggle to attract investors, struggle to raise equity, and may decide a public listing is not worth the trouble. A market-structure rule meant to protect investors had, as a side effect, made it harder for the next generation of companies to use public markets at all. That is a serious enough unintended consequence that it forced a rethink.

The unwinding came in stages, and the geography matters:

- **The UK moved first and furthest.** Freed by Brexit to diverge from EU rules, the UK reviewed MiFID II and chose to *re-permit* bundled payments for research: managers may once again pay for research through trading commissions if they and their clients choose, restoring optionality rather than mandating a return to the old world. The explicit goal was to revive research coverage of smaller UK companies and keep London competitive as a listing venue.
- **The EU eased at the edges.** The bloc relaxed the unbundling rules for research on *smaller* companies (those below a market-capitalisation threshold), where bundling could resume, and scrapped or pared back the much-criticised RTS 27/28 best-execution reports. These are targeted easings rather than a wholesale reversal, but they signal a clear direction of travel: *more flexibility, less rigidity.*
- **The double volume cap was revisited too.** Later reform replaced the twin 4%/8% caps with a single, simpler cap, an acknowledgement that the original double cap was complex to administer and had mostly redirected flow rather than illuminated it.

Why does this matter for the trade? Because each of these steps is a *concrete, datable rule change* that the affected names should re-rate against — and, as this whole series argues, markets price the *expected* rule before it bites. If re-bundling spreads and revives small-cap coverage, the orphaned names that lost their analysts stand to re-rate as coverage returns and liquidity improves. But there is a genuine open question, and it is the crux of the debate: *can a rule change actually conjure coverage back?* The analysts were laid off; the research habits atrophied; the budgets were absorbed into manager P&L and are unlikely to balloon again just because bundling is permitted. It is entirely possible that re-permitting bundling proves a weak lever — that you cannot un-ring the bell, and small-cap coverage stays thin regardless. That uncertainty is exactly what makes it tradeable: the market has to price a probability, and probabilities are where edges live.

## Common misconceptions

MiFID II is wrapped in tidy slogans that fall apart on contact with the data. Three deserve correcting with numbers.

**Misconception 1: "More transparency always means more liquidity."** This is the reformers' core assumption, and it is, at best, half true. Transparency aids price discovery, but for *large* orders it can *destroy* liquidity: a big institution that must reveal its order before it executes will see the market move against it, so it either trades in smaller, costlier pieces or does not trade at all. That is precisely why the large-in-scale waiver and dark pools exist — and why, when the double volume cap squeezed dark trading, volume fled to SIs and periodic auctions rather than to the lit book. The cap was supposed to push perhaps 8%-plus of dark volume into the light; instead, systematic-internaliser volume *rose* and periodic-auction volume grew from near-zero to a meaningful share. More mandated transparency produced more *low-transparency-elsewhere* trading, not more lit liquidity. Transparency and liquidity are not the same thing, and for blocks they actively trade off.

**Misconception 2: "Unbundling helped investors."** On *execution cost*, yes — separating research from commissions cut the visible trading drag substantially, and that is a real win for end investors (recall the worked example: a fund's trading drag falling from €15 million to €6 million). But the verdict is genuinely *mixed*, because the same reform gutted small-cap coverage. When hundreds of small companies lose all analyst coverage, the investors in *those* companies — and the companies themselves, who now struggle to raise capital and attract buyers — are worse off. Spreads on uncovered small caps widened (in our example, from 30 to 90 basis points), which is a direct cost to anyone holding them. The honest scorecard: lower explicit costs and cleaner incentives for large-cap-focused investors, but thinner coverage, wider spreads, and worse liquidity for the small-cap end of the market. "Helped investors" is true for some and false for others — exactly the kind of distributional nuance that gets flattened in the slogan.

**Misconception 3: "MiFID II only matters in Europe."** This one is simply false, and missing it is expensive. MiFID II reshaped the *global* research industry, because the big banks run global research desks: when European clients stopped paying soft dollars, the banks cut analysts who covered companies everywhere, not just in Europe. It forced the SEC into a years-long dance of no-action relief to keep US firms compliant. It created a permanent two-regime world that every global manager must staff and pay for (the €5 million-a-year worked example above). And it set a template — explicit, unbundled, transparent — that regulators elsewhere study when they reform their own markets. A rule written in Brussels became a fixed cost in New York and a precedent in Singapore. Treating it as a parochial European matter is the surest way to be blindsided by the next market-structure rule that "only" applies somewhere else.

**Misconception 4: "Unbundling killed sell-side research."** Overstated. Research budgets fell by roughly a third — a severe contraction, but a contraction, not an extinction. Large-cap coverage survived almost intact, because the largest, most-traded companies still generate enough commission and command enough institutional interest to be worth covering at an honest price. What unbundling killed was the *long tail*: the marginal analyst on the marginal small cap, the coverage that only ever existed because it was free to give away. The industry that emerged is smaller, more concentrated on large caps, more populated by priced independents, and far more honest about what research costs — but it is very much alive. The accurate statement is narrower and more useful: unbundling did not kill research; it killed the *cross-subsidy* that let research be lavished on names that could never pay for it.

## How it shows up in real markets

Pull the threads together into the observable footprint MiFID II left on real markets — the patterns a practitioner can actually see and use.

**Small-cap coverage and liquidity decline.** The most measurable effect. In the years after 2018, the average number of analysts per small-cap European company fell sharply, hundreds of names dropped to zero or one analyst, and bid-ask spreads on the smallest names widened materially. Exchanges and policymakers grew alarmed enough that fixing small-cap coverage became an explicit goal of later reform — a tacit admission that the original rule overshot. If you screen European small caps today, you are screening a universe that is structurally *less covered* than its US equivalent, and the gap is a MiFID II fingerprint.

**Research-budget compression.** The sell-side research industry shrank by roughly a third in spend and shed thousands of analyst jobs. Independent research boutiques that could offer a clean, priced product fared relatively better than full-service banks that had relied on bundling. The survivors learned to charge explicitly — and the buy side learned to consume far less. This is visible in bank earnings, in analyst headcount data, and in the proliferation of "research marketplaces" that sprang up to price and distribute research à la carte.

**The migration of liquidity.** Where European equities actually trade changed shape after 2018: systematic-internaliser volume rose, periodic auctions went mainstream, and lit-venue market share stopped climbing the way the cap's designers expected. For anyone executing size in European stocks, *knowing which venue type holds the liquidity* became a real edge — the flow you are hunting may be sitting in an SI or a periodic auction rather than on the exchange you are watching.

**The divergence as a standing condition.** Most concretely, the US-EU split is simply *there*, every day, as a fixed cost and a structural difference. US small caps remain better covered than EU small caps. Global managers carry duplicate compliance machinery. And the *rollback* of pieces of MiFID II is itself a recurring market-structure catalyst that European financial stocks and exchanges react to.

**The listings-competitiveness anxiety.** A quieter but increasingly important footprint: European policymakers now worry openly that the combination of thin coverage, fragmented liquidity, and heavy regulation makes EU exchanges *less attractive places to list* than US ones — a concern sharpened every time a European company chooses New York for its initial public offering. MiFID II is only one strand of that worry, but it is a visible one, and it is a large part of *why* the rebundling rollback gathered political momentum. For an investor, the signal is that market-structure reform in Europe is now being driven as much by the goal of *reviving the public market itself* as by investor protection — which tilts the direction of future rule changes toward easing, and makes the rollback trade a multi-year structural theme rather than a one-off.

## How to trade it: the playbook

Every post in this series ends here — not on a summary, but on *so how do you read or trade this?* MiFID II gives you three distinct, durable angles.

### The coverage-gap alpha trade

This is the cleanest and most repeatable edge. MiFID II created a *standing inventory* of under-covered European small and mid caps — names that lost their analysts and, with them, the public research that anchors fair value. Your job is to be the marginal informed investor on those names.

- **The signal.** Screen for European small/mid caps with *zero or one* analyst, decent fundamentals, and a wide bid-ask spread (the spread itself confirms the coverage gap). The fewer eyes on a name with real underlying quality, the larger the potential mispricing.
- **The position.** Do the analyst's work yourself — build the model, talk to management, form a target. Buy where your fair value sits well above the market price (the worked example's 33% discount). Size for *illiquidity*: small caps move on small flows, so a meaningful position can take weeks to build without pushing the price, and the wide spread is a real entry cost to budget for.
- **The catalyst.** The gap closes when *coverage returns* — a broker re-initiates, the company graduates into an index, or fundamentals become undeniable enough that institutional screens pick it up. Re-initiation of coverage is the single most reliable re-rating trigger for an orphaned small cap.
- **What invalidates it.** The thesis is wrong if the company is uncovered *for a reason* — declining business, governance problems, a structural reason institutions avoid it. An uncovered name can stay cheap forever, or get cheaper. The edge is in distinguishing *neglected-but-good* from *neglected-and-deservedly-so*; the wide spread cuts both ways, so a value trap here is dearer to exit than a covered one.

### The divergence trade

Position around the structural US-EU split rather than any single name.

- **The signal.** US small caps are structurally better covered and more liquid than EU small caps. That feeds into relative valuations, relative liquidity premia, and the relative cost of running each book. Watch, too, the *firms* on the seam — exchanges, research boutiques, and trading venues whose revenue depends on which regime dominates.
- **The position.** At the portfolio level, the divergence argues for a *coverage-and-liquidity premium* in how you weight EU small caps versus US small caps: demand more discount per unit of fundamental quality in Europe, because you will pay more in spread and wait longer for coverage to find the name. At the single-stock level, European venue operators and independent-research firms are direct plays on the regime.
- **The catalyst.** *Convergence or further divergence.* If the EU rolls back unbundling (below) and re-bundling spreads, EU coverage could partially recover — bullish for orphaned small caps and for the venues that trade them. If the US ever moved toward MiFID-style unbundling, US coverage would face the same compression. Each shift is a datable, tradeable event.
- **What invalidates it.** A surprise harmonisation that erases the gap, or a discovery that the coverage premium is already fully priced into EU small-cap valuations (in which case there is no edge left to harvest).

A practical refinement on the divergence trade: the cleanest expression is often *relative* rather than directional. Rather than betting EU small caps will outperform or underperform outright — which loads you with broad market risk you may not want — you can pair them against a comparable US small-cap basket, isolating the *coverage-and-liquidity differential* itself as the thing you are long or short. If your thesis is that the EU coverage gap is too wide and will narrow as rebundling revives research, you go long the under-covered EU names and short the well-covered US comparables, harvesting the convergence while netting out the direction of the broad equity market. The risk in any such pair is that the two legs are not truly comparable — different sectors, different macro exposures — so the spread can move for reasons that have nothing to do with coverage. Construct the pair carefully, sector by sector, or the divergence you think you are trading gets swamped by noise.

### The rebundling / rollback trade

The newest angle, and a live catalyst. By the early 2020s, the consensus — even among regulators — was that MiFID II's unbundling had *overshot* on small caps. The UK (post-Brexit, free to diverge) moved to *allow* re-bundling, letting managers once again pay for research through commissions if they choose; the EU debated and began easing parts of the regime, including the burdensome RTS 27/28 reports and the rigidity of the research rules. This is the **rebundling debate**, and it is a recurring market-structure catalyst.

- **The signal.** Track the legislative and regulatory calendar for unbundling rollback — consultation papers, draft directives, effective dates. (Trading the rulemaking clock is its own discipline; see [The Regulatory Calendar](/blog/trading/law-and-geopolitics/the-regulatory-calendar-trading-the-rulemaking-clock).) The *direction of travel* is toward more flexibility, but the timing and detail are the tradeable variables.
- **The position.** Re-bundling, if it spreads, is *bullish for small-cap coverage and liquidity* — and therefore for orphaned small caps and for the exchanges and research firms that benefit from revived research flow. It is a slow-burn re-rating thesis, not a one-day event, because coverage rebuilds over quarters, not minutes.
- **The catalyst.** Each concrete easing — the UK's rule change, an EU directive amendment, an effective date — is a step that the relevant names should re-rate against. Position *ahead* of the confirmed rule, because, as this whole series argues, the market prices the *expected* rule before it bites. (On how a rule becomes a price through the run-up and the drift, see [How a Rule Becomes a Price](/blog/trading/law-and-geopolitics/how-a-rule-becomes-a-price-expectations-drift-and-repricing).)
- **What invalidates it.** A reversal of the rollback (regulators re-tightening), or evidence that re-bundling does *not* in fact revive small-cap coverage — perhaps because the analysts and the habit are gone for good, and a rule change cannot conjure them back.

The thread tying all three trades together is the spine of this series: a market-structure *rule* changed the *economics* of research and execution; that reshaped the *flows* of coverage and liquidity; that moved the *prices* of small caps and the *relative attractiveness* of two regimes; and the practitioner's edge is reading the rule — and its coming rollback — early enough to position before the repricing is complete. MiFID II is the cleanest proof in the series that a single line of market-structure law can re-price an entire industry. Make the invisible visible, and watch what happens next.

## Further reading & cross-links

Within this series:

- [Market-Structure Law: Reg NMS, PFOF, and Short-Selling Rules](/blog/trading/law-and-geopolitics/market-structure-law-reg-nms-pfof-and-short-selling-rules) — the US counterpart regime that MiFID II diverged from: how American market plumbing is wired.
- [Securities Law 101: The '33 and '34 Acts and the SEC](/blog/trading/law-and-geopolitics/securities-law-101-the-33-and-34-acts-and-the-sec) — the foundation of US securities regulation, including the Section 28(e) safe harbour that lets US managers still bundle.
- [How a Rule Becomes a Price: Expectations, the Drift, and the Repricing](/blog/trading/law-and-geopolitics/how-a-rule-becomes-a-price-expectations-drift-and-repricing) — the event-study toolkit for trading the rebundling rollback before it lands.
- [The Regulatory Calendar: Trading the Rulemaking Clock](/blog/trading/law-and-geopolitics/the-regulatory-calendar-trading-the-rulemaking-clock) — how to read the legislative and agency calendar that governs when an unbundling rollback actually takes effect.
- [How Law Moves Markets: The Transmission Chain](/blog/trading/law-and-geopolitics/how-law-moves-markets-the-transmission-chain) — the master spine this post hangs on: rule → policy → flows → prices → the trade.

Cross-links out, for the mechanisms this post builds on:

- [Equity Research: How to Value a Company](/blog/trading/equity-research/how-to-value-a-company) — the analyst's work itself, which the coverage-gap alpha trade requires you to do yourself when the sell side has walked away.
- [Hedge Funds: Generating Alpha from Market Inefficiencies](/blog/trading/hedge-funds/generating-alpha-from-market-inefficiencies) — the broader discipline of harvesting structural mispricings, of which the under-covered small cap is one of the most durable examples.
