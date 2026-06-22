---
title: "Lit Markets, Dark Pools, and the Fragmented Tape: Where Trading Actually Happens"
date: "2026-06-22"
publishDate: "2026-06-22"
description: "A plain-English tour of the modern US stock market — lit exchanges, dark pools, wholesalers, payment for order flow, and the consolidated tape that tries to stitch it all back together."
tags: ["capital-markets", "secondary-market", "dark-pools", "market-structure", "payment-for-order-flow", "nbbo", "fragmentation", "liquidity", "high-frequency-trading", "best-execution"]
category: "trading"
subcategory: "Capital Markets"
author: "Hiep Tran"
featured: true
readTime: 42
---

> [!important]
> **TL;DR** — When you buy a share of Apple, your order does not go to "the stock market." It is sprayed by a router across roughly sixteen lit exchanges, thirty-plus dark pools, and a handful of giant wholesalers, and a separate machine called the consolidated tape stitches the prices back into one official quote a few microseconds late.
>
> - **Lit venues** display their quotes before a trade; **dark pools** hide orders and match them at the lit midpoint so a pension fund can sell 2M shares without telegraphing it.
> - **Fragmentation** is the price of competition: dozens of venues compete on fees and speed, but the same stock now trades in dozens of places at once.
> - **Payment for order flow (PFOF)** is why your trading app is "free" — a wholesaler pays your broker ~\$0.0020 a share to fill your order off-exchange, usually a hair better than the public quote.
> - The one number to remember: **about 45% of US equity volume now trades off-exchange** (dark pools + wholesalers), and that share has climbed steadily for fifteen years.

## A morning in the life of one share

At 9:47 a.m. on an ordinary Tuesday, a schoolteacher in Ohio taps "Buy 1,000 shares" of a mid-cap stock on her phone. At the very same instant, a pension fund in Boston needs to sell two million shares of the same company before the close, quietly, without crashing the price. Both of them think they are sending an order to "the market." Neither of them is.

The teacher's order will almost certainly never touch a stock exchange at all. It will be sold — yes, *sold* — by her broker to a trading firm you have probably never heard of, which will fill it in its own internal book a fraction of a cent better than the public price, and pay her broker for the privilege. The pension fund's order will be sliced into thousands of child orders and dribbled into hidden pools where no quote is ever shown, precisely so that nobody — not the teacher, not a high-frequency trader, not a journalist watching the tape — can see the elephant moving through the room.

This is the modern secondary market: not a single trading floor with a bell, but a sprawling, fragmented archipelago of venues, each with its own rules, fees, and clientele, all racing each other in microseconds. It is one of the least understood corners of finance, and one of the most consequential, because **the secondary market's job is to make trading so cheap and so liquid that the primary market can keep doing its job — raising capital.** Nobody buys a freshly issued share if they fear they can never sell it. This post is a map of where that selling actually happens.

![Order routed by a smart order router across lit exchanges, dark pools, and a wholesaler](/imgs/blogs/lit-markets-dark-pools-and-the-fragmented-tape-1.png)

That map looks chaotic, and it is. But every box on it exists for a reason, and by the end you will be able to trace your own order through it. Let us build the picture from the ground up.

## Foundations: what a secondary market is, and the lit/dark split

A **capital market** turns savings into long-term investment through two engines. The **primary market** *creates* securities — a company sells brand-new shares in an IPO, or a government auctions fresh bonds — and the cash flows to the issuer. The **secondary market** is everything that happens afterward: you buy that share from someone who already owns it, and no money reaches the company. The issuer is long gone; you and the seller are just trading an existing claim.

It is tempting to think the secondary market is the boring, derivative part — the issuer already got paid, so who cares? But the secondary market is the load-bearing wall of the whole structure. **Secondary-market liquidity is what makes primary issuance possible.** If you could never resell a 30-year bond, you would demand a far higher yield to lock your money up forever, and the issuer's cost of capital would soar. Liquidity is the promise that you can change your mind tomorrow morning — and that promise is priced into everything.

So the central question becomes: *where, physically and legally, does the trading happen?* Here is the first and most important division.

A **lit venue** publishes its order book before any trade occurs. Anyone can see the best bid (the highest price a buyer will pay), the best offer (the lowest price a seller will accept), and often the depth behind them. The New York Stock Exchange, Nasdaq, and the Cboe equities exchanges are lit. When you see a stock "quoted at \$50.00 bid, \$50.02 offered," that quote came from a lit venue. The gap between bid and offer — here \$0.02 — is the **bid-ask spread**, and it is the cost of immediacy: the price of trading *right now* instead of waiting.

A **dark venue** does the opposite. It accepts orders but shows nothing — no bid, no offer, no depth — and matches buyers against sellers privately, usually at the **midpoint** of the public quote (here \$50.01). The orders are invisible until after they trade. These venues are formally called **Alternative Trading Systems (ATSs)**, and the informal name "dark pool" captures the idea: a pool of liquidity you cannot see into.

Why would anyone want to trade in the dark? Because **showing your hand moves the price against you.** That is the whole game, and it is worth slowing down to understand, because it explains half of modern market structure.

#### Worked example: why the pension fund fears the light

The Boston pension fund must sell 2,000,000 shares of a stock trading at \$50.00 bid / \$50.02 offered. The lit book has only about 5,000 shares bid at \$50.00, then 4,000 at \$49.98, then 6,000 at \$49.96, and so on — the book thins fast, as real books do.

If the fund simply dumps the whole order onto the lit market, it "sweeps the book": it sells 5,000 shares at \$50.00, the next 4,000 at \$49.98, and keeps eating downward through the resting bids. Long before it has sold a tenth of its position, the visible price has cratered, and every trader watching the tape now *knows* a giant seller is in the market. They pull their bids and short ahead of the rest of the order. By the time the fund finishes, the average sale price might be \$49.25 — a **market impact** of \$0.75 a share. On 2,000,000 shares that is \$0.75 × 2,000,000 = **\$1,500,000 of value evaporated**, simply because the order was visible.

The one-sentence intuition: in a lit market, *size is information*, and information leaks straight into the price — so the biggest traders will pay almost anything to stay hidden.

That single fear — leakage — is the engine behind dark pools, behind order slicing, behind the entire off-exchange ecosystem. Let us now see the trade-off in full.

### A few more building blocks before we go deep

Three concepts recur throughout this post, so it is worth nailing them down now.

**The order book.** A lit exchange maintains, for each stock, a continuously updated list of all resting buy orders (the bids) and sell orders (the offers), sorted by price and then by time. The highest bid and lowest offer are the "top of book"; everything behind them is the "depth." When a new order arrives that crosses — a buy order priced at or above the best offer — the matching engine pairs it with the resting order and a trade prints. Our sibling post on the [matching engine and the order book](/blog/trading/capital-markets/inside-an-exchange-the-matching-engine-and-the-order-book) walks through this machinery in detail; here you only need the mental model that a lit book is a *public, price-time-ordered queue.*

**Marketable versus resting orders.** An order that crosses the spread immediately — "buy at the market" or "buy at \$50.02 when the offer is \$50.02" — is *marketable*: it takes liquidity and trades right away. An order that sits in the book waiting — "buy at \$49.98 when the offer is \$50.02" — is *resting*: it provides liquidity and waits for someone to trade against it. This distinction is load-bearing because exchanges pay and charge participants differently depending on which they do, which brings us to the third concept.

**Maker-taker pricing.** Most lit US exchanges run a "maker-taker" fee model: they pay a small **rebate** to the participant whose resting order is matched (the "maker," who supplied liquidity) and charge a slightly larger **fee** to the participant whose marketable order takes it (the "taker"). The difference is the exchange's revenue. A typical schedule might rebate \$0.0020 a share to the maker and charge \$0.0030 to the taker, netting the exchange \$0.0010. This sounds trivial, but it drives an enormous amount of routing behavior: brokers and high-frequency market makers route partly to *capture rebates*, not just to get the best price for the end customer, and a few venues run "inverted" (taker-maker) schedules to attract the opposite flow. Reg NMS lets exchanges set these fees within a cap (historically \$0.0030 a share), and the resulting rebate game is one of the subtler conflicts of interest in the whole system.

With order book, marketable-versus-resting, and maker-taker in hand, the rest of the machinery clicks into place.

## Lit versus dark: immediacy versus invisibility

![Before-after comparison of sweeping the lit book versus working an order in a dark pool](/imgs/blogs/lit-markets-dark-pools-and-the-fragmented-tape-3.png)

The figure makes the choice concrete. On the left, the naive path: sweep the lit book, eat each price level, watch the price fall, and pay roughly \$750k in slippage on a more careful execution (less than the brute-force \$1.5M, but still brutal). On the right, the patient path: route the order into dark pools, match at the midpoint, never move a single visible quote, and save most of that cost.

#### Worked example: the same 2M shares, worked in the dark

Now the fund routes its 2,000,000 shares into dark pools instead. Each fill happens at the **midpoint**, \$50.01, against hidden buyers who are themselves trying to accumulate quietly. Because no quote is displayed, the public price barely budges over the hour it takes to work the order.

Two savings stack up. First, every share sold at the \$50.01 midpoint instead of the \$50.00 bid earns an extra \$0.01 — half the spread — worth \$0.01 × 2,000,000 = **\$20,000**. Second, and far larger, the order avoids the market-impact spiral: instead of dragging the average price down by \$0.35–\$0.75, the fund sells near the prevailing price. If we conservatively say it avoids \$0.35 of impact, that is \$0.35 × 2,000,000 = **\$700,000** preserved.

The one-sentence intuition: a dark pool lets a whale trade like a minnow — it converts the *cost of being seen* into a *savings from being hidden*, which is why institutions route the vast majority of their large orders through the dark.

But the dark is not free lunch. Three real costs lurk:

- **No guarantee of a fill.** A dark order only executes if a matching counterparty happens to be in the same pool at the same time. Liquidity is invisible *and* uncertain. The fund may sit in the dark all day and fill only part of its order, forcing the rest back into the light near the close anyway.
- **Adverse selection.** If the only counterparties willing to trade with you in the dark are better-informed than you — they know something you do not — then your "good" fills are systematically the ones you will regret. Sophisticated pools police this with minimum sizes and counterparty screening; the formal models of this live in the [order-book and adverse-selection literature](/blog/trading/quantitative-finance/order-book-simulator-quant-research), which we link out to rather than re-derive here.
- **Information leakage anyway.** Some dark pools historically leaked order information to favored high-frequency clients, the subject of a string of SEC enforcement actions. Dark does not always mean private.

So the choice between lit and dark is really a choice between **certainty of execution** (lit, but you pay impact) and **minimal footprint** (dark, but you might not fill). Every large trader threads this needle thousands of times a day.

### Not all dark pools are the same

"Dark pool" is an umbrella over several quite different animals, and lumping them together causes most of the public confusion. Three broad kinds matter:

- **Agency / independent crossing networks.** Run by brokers acting purely as agents, or by independent firms, these pools cross client orders against each other at the midpoint and charge a small commission. The operator does not trade against you; it is a matchmaker. Liquidnet, the old ITG POSIT, and bank-run "block crossing" venues fit here. These are the pools institutions trust most for genuine size.
- **Broker-dealer internalization pools.** Run by big banks (the historical Sigma X, Crossfinder, MS Pool names), these match client orders but the operating bank may *also* trade against the flow with its own capital. That dual role is exactly where most of the historical conflicts and SEC fines arose — a few pools were caught steering favorable fills to proprietary or favored high-frequency clients while telling institutions the pool was "clean."
- **Wholesaler / electronic market-maker internalizers.** Citadel Securities and Virtu are not really "pools" you send a resting order into; they are principal market makers that fill *your* order out of their own inventory. Technically these are off-exchange but they behave differently from a crossing network — there is always a counterparty (the wholesaler), so a fill is far more certain, but you are trading against a professional, not against another institution.

The reason the distinctions matter: when a critic says "half the market is dark and that's dangerous," the *crossing-network* half (institutions quietly trading blocks) is doing something genuinely benign, while the *internalizer* half (retail flow siphoned to wholesalers) is the part that raises the price-discovery worry. Averaging them into one "45% off-exchange" number hides a real difference in kind.

### How a big order is actually worked: the execution algorithm

A pension fund does not literally "route 2M shares to a dark pool" in one click. It hands the order to an **execution algorithm** — software, usually provided by its broker, that slices the parent order into hundreds or thousands of child orders and decides, second by second, where to send each one. The common algo families:

- **VWAP / TWAP** — spread the order evenly across the day to match the volume-weighted or time-weighted average price, so the fund's fill tracks the day's "fair" average rather than any single moment.
- **Implementation shortfall** — front-load the order to minimize the gap between the price when the decision was made and the average fill, trading impact against timing risk.
- **Liquidity-seeking / "dark-aggregating"** — ping many dark pools simultaneously for hidden midpoint liquidity, only crossing to the lit book when the dark runs dry.

#### Worked example: slicing a 2M-share order across a day

The fund's 2,000,000-share sell order goes into a liquidity-seeking algo. The stock trades about 10,000,000 shares a day, so the fund is 20% of daily volume — far too big to show. The algo aims to be no more than ~10% of volume in any short window to stay invisible.

Over the day it might fire roughly 4,000 child orders averaging 500 shares each. Suppose 1,200,000 shares (60%) fill in dark pools at an average midpoint of \$50.01, and the remaining 800,000 shares fill on lit venues at an average of \$49.97 after some unavoidable impact. The blended average sale price is then (1,200,000 × \$50.01 + 800,000 × \$49.97) ÷ 2,000,000 = (\$60,012,000 + \$39,976,000) ÷ 2,000,000 = **\$49.994 a share**. Against an arrival price of \$50.01, that is under a penny of slippage — versus the \$0.75 disaster of dumping the whole order on the lit book.

The one-sentence intuition: a good execution algo turns one terrifying elephant of an order into a long, quiet drizzle of minnow-sized child orders, most of them hidden, so the market never realizes a whale was in the water.

## Fragmentation: why one stock trades in dozens of places

Here is where the modern market diverges sharply from the cartoon of "the stock exchange." In the United States, a single stock can trade simultaneously across:

- **~16 lit exchanges** — NYSE, NYSE Arca, NYSE American, Nasdaq, Nasdaq BX, Nasdaq PSX, the three Cboe equities exchanges (BZX, BYX, EDGA, EDGX), IEX, MEMX, MIAX, and more. They are largely fungible: an Apple share is an Apple share wherever it trades. But each charges different fees, offers different rebates, and sits in a different data center.
- **~30+ dark pools / ATSs** — operated by big banks (the old "Sigma X," "Crossfinder," "MS Pool" names), by independent firms, and by agency brokers. Each has its own clientele and matching rules.
- **A handful of wholesalers** — Citadel Securities, Virtu, and a few others — that internalize the bulk of retail order flow off-exchange entirely.

Why on earth would a market evolve into dozens of competing venues for the same product? The short answer is **regulation that mandated competition.** In the US, Regulation NMS (National Market System, 2005) and its predecessors deliberately broke the near-monopoly of the NYSE by requiring orders to be routed to whichever venue showed the best price. Once any venue's best quote had to be honored market-wide, new venues could spring up, undercut on fees, and instantly compete. The number of exchanges exploded.

![US equity volume split between lit exchanges and off-exchange venues in 2024](/imgs/blogs/lit-markets-dark-pools-and-the-fragmented-tape-2.png)

As the chart shows, the split today is roughly **55% lit / 45% off-exchange**. Off-exchange — dark pools plus wholesaler internalization — is now nearly half of all US equity volume. That is a staggering figure: it means the "price discovery" that the lit exchanges are supposed to perform is happening on a shrinking slice of the actual trading.

The trade-off of fragmentation is competition versus complexity:

- **The upside (competition):** spreads have collapsed, explicit commissions have gone to zero for retail, and venues relentlessly undercut each other on fees. A market with one monopoly exchange had no such pressure.
- **The downside (complexity):** liquidity is scattered. To find the best price you must look in sixteen places at once, which requires a **smart order router (SOR)** — software that knows where the liquidity is, splits your order, and sweeps multiple venues in milliseconds. You also need a way to *combine* all those venues' quotes into one official price, which is the consolidated tape (we get there shortly). And complexity creates seams — tiny timing and pricing gaps between venues — that the fastest players can exploit.

Fragmentation, in other words, is not a bug or a conspiracy. It is the *direct consequence* of a policy choice to favor competition over a single national exchange. Whether that choice was wise is the market-quality debate we close on.

### The glue that holds fragmentation together: the Order Protection Rule

If sixteen exchanges all quote the same stock at different prices, what stops a venue from filling your order at a worse price than another venue is openly showing? The answer is the **Order Protection Rule** (Rule 611 of Reg NMS), often called the **trade-through rule.** It says, roughly: a trade may not execute at a price *worse* than the best displayed quote on any other lit venue — you may not "trade through" a better protected quote. If Nasdaq is showing \$50.02 and another venue tries to fill a buy at \$50.04, that trade is a prohibited trade-through; the order must first be routed to take Nasdaq's \$50.02.

This single rule is what makes fragmentation tolerable. Without it, splitting the market into sixteen venues would mean sixteen different prices and no guarantee you got the best one. With it, every venue is forced to respect every other venue's best displayed price, so the *displayed* market behaves as if it were one book even though it physically is not. The SOR's job is precisely to honor Rule 611: sweep the protected quotes in price order so no trade-through occurs.

But notice the rule's two big loopholes, both central to this post. First, it only protects **displayed** quotes — dark liquidity is invisible and therefore unprotected, which is part of why dark trading is legal at all. Second, it protects the quote *as published*, and "published" means on the SIP, which is slightly stale — so a trade that respects the SIP's NBBO may nonetheless be trading through a *fresher* price visible only on a direct feed. The trade-through rule, in other words, both *enables* fragmentation (by re-aggregating displayed prices) and *creates* the latency-arb gap (by anchoring everyone to a tape that lags). It is the hinge on which the whole system turns.

#### Worked example: a trade-through and the routing that prevents it

A broker holds a 2,000-share buy order and its primary venue, Cboe BZX, shows 2,000 shares offered at \$50.04. But the SIP's protected NBBO offer is \$50.02, with 800 shares offered on Nasdaq and 1,200 on NYSE Arca. If the broker simply filled all 2,000 on BZX at \$50.04, it would trade through the \$50.02 quotes — illegal, and \$0.02 × 2,000 = **\$40** worse for the customer.

Instead the SOR routes 800 shares to Nasdaq and 1,200 to Arca at \$50.02, exhausting the protected quotes, and only then — if more were needed — moves up to \$50.04. The customer pays 2,000 × \$50.02 = **\$100,040** rather than \$100,080, saving the \$40 that the trade-through rule guaranteed.

The one-sentence intuition: the Order Protection Rule is the law that forces a fragmented market to behave like a single one, and the SOR is the robot that obeys it on every order.

#### Worked example: the smart order router splitting a 1,000-share buy

Take a 1,000-share buy order. The SOR checks the consolidated quote and sees that the best offer is \$50.02, but it is split: 300 shares offered at \$50.02 on Nasdaq, 400 shares at \$50.02 on NYSE Arca, and 300 shares at \$50.02 on Cboe BZX. No single venue has the full 1,000.

The router fires three child orders simultaneously — 300 to Nasdaq, 400 to Arca, 300 to BZX — each timed to hit within microseconds so the quotes don't move. All three fill at \$50.02. The trader sees a single, clean "bought 1,000 @ \$50.02" on the screen and never knows it touched three venues. Total cost: 1,000 × \$50.02 = **\$50,020**, the genuine best price available anywhere.

The one-sentence intuition: fragmentation only works because routing software re-aggregates the scattered liquidity into a single best price — the human sees one market even though the machine touched many.

## Wholesalers and payment for order flow: why your trades are "free"

Now we come to the part that makes headlines. When the Ohio teacher buys 1,000 shares with no commission, *somebody* is paying for the infrastructure, the matching, the regulatory overhead. Who?

The answer is the **wholesaler** — a market-making firm, Citadel Securities and Virtu being the two giants, that specializes in trading against retail order flow. Here is the arrangement, which is called **payment for order flow (PFOF)**:

1. The teacher's broker (Robinhood, Schwab, E*Trade, etc.) does not send her order to an exchange. It sends it to a wholesaler.
2. The wholesaler **pays the broker** for that order — a fraction of a cent per share, often quoted around \$0.0010–\$0.0030.
3. The wholesaler fills the order out of its own inventory, *off-exchange*, usually at a price slightly better than the public quote — this is **price improvement**.
4. The wholesaler keeps whatever spread is left over as its profit.

![Pipeline showing a retail order routed to a wholesaler who pays the broker and fills inside the public quote](/imgs/blogs/lit-markets-dark-pools-and-the-fragmented-tape-5.png)

Why does the wholesaler want retail flow so badly that it will pay for it? Because retail orders are **uninformed** in the statistical sense — a schoolteacher buying 1,000 shares for her retirement account is not trading on inside knowledge of next quarter's earnings. The wholesaler can fill her order, hedge it, and capture the spread with very low risk of being "run over" by a better-informed counterparty. Institutional flow, by contrast, often *is* informed, and trading against it is dangerous. Retail flow is the safe, predictable, profitable flow — so wholesalers compete to buy it.

#### Worked example: the economics of one PFOF order

The teacher buys 1,000 shares. The public market is \$50.00 bid / \$50.02 offered, a 2-cent spread, so the midpoint is \$50.01.

- **The customer's outcome:** instead of paying the full \$50.02 offer, the wholesaler fills her at \$50.015 — half a cent of **price improvement**. On 1,000 shares she saves \$0.005 × 1,000 = **\$5.00** versus crossing the lit spread. She also pays \$0 commission.
- **The broker's outcome:** the wholesaler pays the broker \$0.0020 a share for the order. On 1,000 shares that is \$0.0020 × 1,000 = **\$2.00** of revenue, earned without lifting a finger. Multiply by millions of orders a day and you have a real business — this is a large part of how "free" brokers make money.
- **The wholesaler's outcome:** it bought at the \$50.00 bid (or hedged near there) and sold to the teacher at \$50.015, capturing roughly \$0.015 a share of gross spread, out of which it paid the broker \$0.0020 and gave the customer \$0.005 of improvement. The residual is its margin.

The one-sentence intuition: PFOF is a three-way split of the bid-ask spread — the customer gets a sliver of price improvement, the broker gets a sliver of payment, and the wholesaler keeps the rest for bearing the inventory risk.

So is PFOF good or bad? This is one of the genuine debates in market structure, and an honest answer has two sides.

**The case for PFOF:** Retail investors today trade for *zero commission* and routinely get filled *better than the displayed quote*. Both of those are real, measurable improvements over the world of \$10 commissions and exchange-only fills that prevailed twenty years ago. Wholesalers compete on price improvement, and brokers are required to seek **best execution** for their customers.

**The case against PFOF:** It creates a **conflict of interest**. The broker is paid by the wholesaler, not the customer, which gives the broker an incentive to route to whoever pays the most rather than whoever fills best. Critics argue that the "price improvement" is measured against an NBBO that is itself slightly stale and artificially wide, so the improvement is smaller than it looks. And there is a structural worry: if nearly half of all volume — the easy, uninformed retail half — is siphoned off-exchange to wholesalers, then the lit exchanges are left with a *harder* mix of informed and institutional flow, which can *widen* lit spreads for everyone.

### What "best execution" actually requires

The legal counterweight to the PFOF conflict is the broker's **best-execution** duty. This is not a duty to get the single best price on every trade — markets move too fast for that to be a coherent standard. It is a duty to use *reasonable diligence* to obtain the most favorable terms *reasonably available*, judged across price, speed, likelihood of execution, and total cost, and to *regularly and rigorously review* the execution quality the broker is actually getting from the venues it routes to. Brokers must file public reports (the SEC's Rule 605 execution-quality and Rule 606 order-routing disclosures) showing where they sent orders and how good the fills were.

The tension is obvious: a broker paid by a wholesaler has an incentive to route to the highest payer, while best execution demands it route to the best filler — and those need not be the same venue. The SEC's position is that disclosure plus the best-execution duty *manages* the conflict; the European position, as we will see, is that the conflict cannot be managed and must be removed. The honest truth is that measuring "best execution" rigorously is genuinely hard, because the benchmark — the NBBO — is itself produced by the slightly-stale SIP, so a fill can look like "price improvement versus the NBBO" while still being worse than what a faster, fuller view of the market would have offered.

#### Worked example: price improvement that is real, and price improvement that is illusory

A retail order to buy 500 shares arrives when the SIP NBBO is \$50.00 / \$50.04 — a wide 4-cent spread. The wholesaler fills at \$50.03, reporting \$0.01 of "price improvement" versus the \$50.04 offer, worth \$0.01 × 500 = **\$5.00**. That looks like a win.

But suppose the *true* market, visible on direct feeds, was already \$50.01 / \$50.02 — the SIP quote was 300 microseconds stale and artificially wide. Against the true \$50.02 offer, the \$50.03 fill is actually \$0.01 *worse*, costing the customer \$0.01 × 500 = **\$5.00** versus what a faster router could have achieved. The reported "improvement" was measured against a stale benchmark.

The one-sentence intuition: price improvement is only as honest as the quote it is measured against, and the whole PFOF debate hinges on whether the NBBO yardstick is fresh enough to mean anything.

**The European answer:** The EU and UK looked at this conflict and effectively **banned PFOF** (phased in through the EU's MiFIR review, with a full prohibition taking effect by 2026). Their reasoning was that the conflict of interest could not be adequately managed by disclosure alone. The US, by contrast, has kept PFOF while the SEC has repeatedly proposed (and largely shelved) reforms like mandatory order-by-order auctions. The two regimes are now running a natural experiment on whether banning PFOF helps or hurts retail investors — and the early evidence is genuinely mixed.

## Common misconceptions

**"Dark pools are illegal or shady."** No. ATSs are SEC-registered, regulated venues that file detailed volume reports (FINRA publishes ATS volume by venue every week). They exist for a legitimate and important reason: letting institutions trade size without ruinous market impact, as the 2M-share example showed. Specific dark pools have been fined for *specific* abuses (leaking order data, misrepresenting who trades there), but the venue type itself is mainstream plumbing, not a back alley.

**"Payment for order flow means I get a worse price."** Usually the opposite, at the level of the individual fill: retail orders routed to wholesalers typically receive *price improvement* versus the public quote, and pay zero commission. The legitimate worry is subtler and *market-wide* — that pulling the easy flow off-exchange makes the lit quote itself worse for everyone — not that your specific 1,000-share order got skinned.

**"The NBBO is the real, true price of the stock."** The NBBO is the best *displayed* bid and offer across lit venues, assembled by the consolidated tape. But it ignores hidden dark liquidity, it can be a few microseconds stale, and the true tradable price for a large order is nothing like the NBBO once you account for impact. The NBBO is a useful reference, not a law of physics.

**"Fragmentation means the market is broken."** Fragmentation is the *cost* of a policy that mandated competition, and it bought us razor-thin spreads and zero commissions. It also created complexity and latency seams. "Broken" is the wrong word; "a deliberate trade-off with real downsides" is the right one.

**"High-frequency traders front-run my order."** Classic front-running — a broker trading ahead of its own client's order — is illegal and rare. What HFTs actually do is faster: they read public information (quote changes, the consolidated tape, order-flow patterns) microseconds before slower participants and adjust. It is a latency game played on public data, which is why the latency-arbitrage section below matters — but it is not the same as illegally trading ahead of a specific known order.

## The consolidated tape and the NBBO

With trading scattered across dozens of venues, the market needs one authoritative answer to a simple question: *what is the best price for this stock right now?* That answer is the **National Best Bid and Offer (NBBO)**, and it is produced by the **consolidated tape** — a system of **Securities Information Processors (SIPs)** that ingest every quote and every trade from every lit venue and stitch them into a single national feed.

![Off-exchange share of US equity volume rising from 2010 to 2024](/imgs/blogs/lit-markets-dark-pools-and-the-fragmented-tape-4.png)

That rising off-exchange share is exactly why the SIP and NBBO matter more, not less, over time: as more volume goes dark, the displayed lit quotes that feed the NBBO become a more precious and more contested signal of where the price "really" is.

The SIP does two essential jobs:

- **Best execution reference.** A broker is legally required to fill your order at the NBBO or better. The SIP defines what the NBBO *is*, so it is the yardstick against which "best execution" and "price improvement" are measured.
- **Trade reporting.** Every trade, including dark and off-exchange prints, must be reported to the tape (off-exchange trades print through a FINRA facility called the TRF). So even though the *quote* is dark, the *trade* eventually shows up on the tape — just after the fact, without revealing it was a hidden order.

Here is the catch, and it is a famous one. The SIP is **slow** — not slow in human terms, but slow in market terms. Consolidating quotes from sixteen geographically dispersed data centers, time-stamping them, and rebroadcasting them takes time, historically hundreds of microseconds to a few milliseconds. Meanwhile, the fastest traders pay for **direct feeds** straight from each exchange's matching engine, co-located in the same data center, and receive the same quotes *microseconds sooner* than the SIP can publish them.

![Graph of venue quotes feeding the slow SIP and fast direct feeds, opening a latency-arbitrage gap](/imgs/blogs/lit-markets-dark-pools-and-the-fragmented-tape-7.png)

That gap — the difference between when a fast trader sees a price change and when the public SIP shows it — is the **latency-arbitrage** window. It is small, but it is real, and it is the heart of the modern speed game.

#### Worked example: latency arbitrage on a stale SIP quote

Suppose the NBBO for a stock is \$50.00 bid / \$50.02 offered, and that quote is published by the SIP. At time *T*, a big buyer lifts all the \$50.02 offers on the primary exchange, and the true market instantly jumps to \$50.03 bid / \$50.05 offered. A co-located HFT reading the direct feed sees this at *T + 50 microseconds*. The SIP, however, won't publish the new quote until *T + 500 microseconds* — for 450 microseconds, the public NBBO still says the stock is offered at \$50.02.

In that window, the HFT can buy from anyone still resting an order priced off the stale \$50.02 NBBO — say a 5,000-share sell order on a venue that pegs to the SIP — and instantly resell at the true \$50.05. That is \$0.03 a share × 5,000 = **\$150 of near-riskless profit**, captured purely because it saw the price move before the public did. Repeat this thousands of times a day across thousands of names.

The one-sentence intuition: when the official price is even a few hundred microseconds stale, "the price" is briefly two different numbers at once, and whoever sees the newer one first can pick the pocket of whoever is still quoting the old one.

This is precisely the problem that the IEX exchange was built to attack, with its famous 38-microsecond "speed bump" coil of fiber that delays incoming orders just enough to neutralize the direct-feed advantage. It is also why there has been a long regulatory push (the SEC's Market Data Infrastructure Rule) to speed up and decentralize the SIP. The latency-arb gap is not a loophole someone forgot to close; it is an inherent feature of stitching a fragmented market together with a centralized tape that cannot move at the speed of light.

### Who runs the tape, and the round-lot wrinkle

The SIP is not a single computer; it is a piece of regulated market infrastructure governed by committees of the very exchanges whose data it consolidates — historically two plans, one for NYSE-listed stocks (the "CTA/CQ" tapes A and B) and one for Nasdaq-listed stocks (the "UTP" tape C). That governance structure is itself contested: the exchanges that run the SIP *also* sell the fast proprietary direct feeds that beat it, so they have a commercial interest in the SIP not being *too* good. The SEC's reforms aim to break that conflict by introducing competing consolidators and by widening what the tape must carry.

One concrete example of the tape's historical narrowness: until recent reforms, the NBBO only reflected **round lots** — orders of 100 shares or more. For a stock like a \$3,000-a-share Amazon (pre-split) or a high-priced Berkshire, almost all real orders were *odd lots* (under 100 shares), which meant the official NBBO ignored most of the actual trading interest. A retail investor could be getting filled at prices the NBBO never showed, making "price improvement versus NBBO" claims partly meaningless for exactly the high-priced names where the spread mattered most. The reforms shrank the round-lot definition for expensive stocks and added odd-lot information to the tape — a quiet but important fix that makes the public price a truer picture of where small investors actually trade.

#### Worked example: the odd-lot blind spot

A stock trades at \$2,000 a share. The official round-lot NBBO is \$1,999.00 / \$2,001.00 — a \$2.00 spread — but those quotes require 100-share (\$200,000) commitments that almost nobody posts. Hidden in odd lots, there are real orders at \$1,999.80 bid and \$2,000.20 offered, a far tighter \$0.40 spread.

A retail investor buying 10 shares (\$20,000) sees the \$2,001.00 round-lot offer on her screen and is "improved" to \$2,000.50 — a reported \$0.50 × 10 = **\$5.00** of improvement. But the true odd-lot offer was \$2,000.20, so she actually paid \$0.30 × 10 = **\$3.00** more than the best real price available. The round-lot NBBO blinded her to it.

The one-sentence intuition: a consolidated tape is only as useful as the orders it is allowed to see, and for years it was structurally blind to exactly the small orders that ordinary investors place.

## Does fragmentation and dark trading help or hurt price discovery?

This is the deep question, and it ties everything back to the series' spine. **Price discovery** is the process by which the market figures out what a security is actually worth — it is the public good that the secondary market is supposed to produce. (Our sibling post on [how a price is made](/blog/trading/capital-markets/how-a-price-is-made-discovery-arbitrage-and-efficiency) treats the mechanism in full; here we ask only how *fragmentation* affects it.)

The worry is straightforward. Price discovery happens on **lit** venues, where displayed quotes reveal supply and demand. If the easy, uninformed retail flow is siphoned off to wholesalers, and the big institutional flow is hidden in dark pools, then the lit exchanges — the place where prices are actually *formed* — see a thinner, more adversarial slice of the total. In the extreme, you could end up with a market where almost everything trades in the dark *referencing* a lit NBBO that almost nothing actually contributes to. The midpoint everyone is matching at would be a price that fewer and fewer real orders helped set — a hall of mirrors.

![Stacked bar of lit versus off-exchange share drifting over time from 2010 to 2024](/imgs/blogs/lit-markets-dark-pools-and-the-fragmented-tape-8.png)

The drift in that chart — off-exchange rising from ~30% in 2010 toward ~47% by 2024 — is what keeps regulators up at night. There is a level of dark trading beyond which the lit market becomes too thin to do its price-discovery job well. Europe responded with explicit **dark-trading caps** (the MiFID II "double volume cap" limited how much of a stock could trade dark). The US has so far relied on competition and disclosure instead of hard caps.

There is a subtler version of the worry worth spelling out, because it explains why the debate never quite resolves. Dark pools and wholesalers do not set prices; they *reference* the lit price. A midpoint cross at \$50.01 only exists because some lit venue published \$50.00 / \$50.02. So the dark market is, in a precise sense, a **free rider** on the lit market's price discovery: it consumes the price signal without contributing the displayed orders that produce it. As long as enough flow stays lit to keep that signal sharp, free-riding is harmless and even efficient — the lit market does the expensive work of discovery once, and everyone benefits. The danger is a tipping point: if so much flow free-rides that the lit market can no longer afford to produce a good signal (because the market makers who post lit quotes can no longer earn enough spread to justify the adverse-selection risk), then the signal degrades for everyone, including the dark venues that depend on it. That feedback loop — dark trading erodes the lit quotes that dark trading needs — is the real systemic concern, and it is why "what fraction is dark" is a number worth watching rather than a curiosity.

The honest counterargument: empirically, US spreads are *narrow* and price efficiency is *high* by historical and international standards. If fragmentation and dark trading were badly damaging price discovery, we would expect to see it in wide spreads and sluggish prices, and mostly we do not — yet. The system has so far been resilient. But "so far" is doing a lot of work, and the steady climb of the off-exchange share means the question is not settled; it is being tested in real time.

#### Worked example: how the spread you pay depends on liquidity tier

The cost of fragmentation is not uniform — it bites hardest in the names that need help most. Consider trading 1,000 shares across the liquidity tiers, where the displayed spread runs from about 1 basis point in a mega-cap to roughly 80 basis points in a micro-cap.

For a mega-cap like Apple at \$200, a 1-bp spread is \$200 × 0.0001 = \$0.02, so crossing it on 1,000 shares costs about \$0.02 × 1,000 = **\$20**. For a micro-cap at \$5 with an 80-bp spread, the spread is \$5 × 0.0080 = \$0.04 per share, costing \$0.04 × 1,000 = **\$40** on the same share count — but as a *fraction of value traded* (\$5,000), that \$40 is 0.8%, versus 0.01% for the Apple trade. The micro-cap is roughly **80× more expensive to trade** per dollar.

The one-sentence intuition: fragmentation and dark trading lavish competition on the liquid mega-caps where it is least needed, while the illiquid small-caps — where good price discovery matters most — are left with wide spreads and thin lit books.

![Bid-ask spread in basis points across mega-cap to micro-cap liquidity tiers](/imgs/blogs/lit-markets-dark-pools-and-the-fragmented-tape-6.png)

That tiering is the practical face of the whole debate: the headline statistics ("US spreads are tight!") are dominated by the handful of giant, hyper-liquid names, and they quietly average away the much worse experience in the long tail of small and micro-cap stocks, where fragmentation scatters already-scarce liquidity.

## Following the money: who gets paid in the fragmented market

It helps to step back and ask the question that cuts through every market-structure argument: *who is paid, by whom, for what?* Because once you can see the flows of money, the incentives — and the conflicts — become obvious.

- **The lit exchanges** are paid two ways. First, the net of maker-taker fees on every trade (a fraction of a cent per share). Second, and increasingly the bigger prize, **market-data and connectivity fees** — they sell those fast proprietary direct feeds and the co-location racks right next to the matching engine for very large sums. This second revenue stream is why exchanges have a quiet interest in the public SIP staying a step behind: the slower the free tape, the more valuable the fast paid feed.
- **The wholesalers** are paid the residual spread on the retail flow they internalize, net of the PFOF they pay brokers and the price improvement they give customers. Their edge is that retail flow is uninformed and therefore cheap to trade against.
- **The brokers** are paid PFOF by wholesalers, plus interest on customer cash and margin lending. For a "free" retail app, PFOF and cash interest are the business model — the trade itself is the loss leader.
- **The dark pools** are paid commissions (agency crossing networks) or capture spread (internalizing pools) on the institutional flow they match.
- **The high-frequency market makers** are paid the spread they earn quoting on lit venues, plus maker rebates, plus the latency-arb edge of seeing prices first.

Lay these out side by side and the central tension of the whole system snaps into focus: **almost every intermediary is paid more when the public, free, consolidated view of the market is a little worse than the private, paid, fast view.** That is not a conspiracy — no one is breaking the rules — but it is a structural gravity that pulls the market toward complexity, speed, and opacity, and it is exactly why regulators keep circling back to the SIP, to PFOF, and to dark-volume caps. Every reform in this area is, at bottom, an attempt to narrow the gap between the public view and the private one.

#### Worked example: where the spread on one retail trade ends up

Trace the 2-cent spread on the teacher's 1,000-share buy (\$50.00 / \$50.02). The wholesaler captures roughly \$0.015 a share of gross spread = \$0.015 × 1,000 = **\$15.00** of raw economics. Out of that it pays the broker \$0.0020 × 1,000 = **\$2.00** of PFOF and gives the customer \$0.005 × 1,000 = **\$5.00** of price improvement, keeping about \$8.00 as its margin before hedging costs. The lit exchange that *would have* earned the maker-taker net on this trade earns **nothing**, because the trade never reached it.

The one-sentence intuition: every "free" retail trade quietly redistributes the bid-ask spread away from the public exchange and toward the wholesaler, broker, and customer — which is great for that customer and corrosive for the lit market that sets the price everyone references.

## How it shows up in real markets

**The "Flash Boys" moment (2014).** Michael Lewis's book *Flash Boys* made latency arbitrage a household phrase and turned IEX — the exchange with the speed bump — into a cause célèbre. Whatever one thinks of the book's framing, it forced a real reckoning: the SEC opened inquiries, exchanges disclosed their data-feed latencies, and IEX won approval as a full exchange in 2016. The episode is the clearest public example of the latency-arb gap described above moving from an insider's edge to a front-page controversy.

**The GameStop saga (January 2021).** When retail traders piled into GameStop, the plumbing of PFOF was suddenly on national television. Robinhood, which earns much of its revenue from PFOF, had to restrict buying — not because of a conspiracy, but because the *clearing* layer (the topic of our [exchanges-and-clearinghouses](/blog/trading/finance/stock-exchanges-and-clearinghouses) post) demanded enormous collateral to cover the volatile trades. The episode exposed how the wholesaler/PFOF model, the fragmented routing, and the post-trade plumbing are all one connected machine — and how few retail investors understood that their "free" trades ran through Citadel Securities' internalizer.

**The off-exchange tipping point (2023–2024).** For the first time, monthly data began showing off-exchange volume occasionally *exceeding* lit-exchange volume on certain days — the ~45% average masking days above 50%. This crossed a psychological line and intensified the SEC's market-structure reform agenda, including proposals to tighten the definition of best execution and to require some retail orders into competitive auctions. As of this writing those reforms are mostly proposed, not enacted, but the direction of travel is clear: regulators are increasingly uncomfortable with how much trading happens where no quote is shown.

**The EU PFOF ban (phasing to 2026).** Europe's decision to prohibit PFOF outright is the single biggest live experiment in this whole debate. Brokers in the EU that had built free-trading apps on PFOF revenue must now find another model — typically explicit commissions or spreads. Watching whether EU retail investors end up better or worse off than their US counterparts will, over the next few years, produce the closest thing to a controlled trial that market structure ever gets.

**The Vietnam contrast.** It is worth noting how different all of this looks in a less-fragmented market. On the Ho Chi Minh exchange (HOSE), trading is far more centralized — one dominant venue, no sprawling dark-pool ecosystem, and order matching that is comparatively transparent. The trade-off runs the other way: less fragmentation and less latency arbitrage, but also less competition on fees and, historically, periods of [foreign-flow-driven volatility](/blog/trading/vietnam-stocks/foreign-flows-etfs-and-the-index-effect-vietnam) where big institutional orders had nowhere dark to hide. Market structure is a set of choices, and different markets choose differently.

## The takeaway: the market is a map, and now you can read it

Step back and the fragmented modern market resolves into something coherent. Every piece of the machinery you just toured exists to manage a single tension: **the conflict between immediacy and information.** Traders want to execute *now*, but executing now reveals *what they want*, and revealing what they want moves the price against them.

- **Lit venues** sell immediacy at the cost of visibility.
- **Dark pools** sell invisibility at the cost of certainty.
- **Wholesalers and PFOF** monetize the one flow that has *no* information to hide — uninformed retail — and use it to subsidize free trading.
- **Fragmentation** is the competitive market we got when regulators broke the monopoly exchange, and the SOR is the software that re-aggregates the scattered liquidity.
- **The consolidated tape and NBBO** are the heroic, imperfect attempt to make all of this look like one market again — imperfect because light is finite and the tape is always a few microseconds behind the fastest feed.
- **The Order Protection Rule** is the legal thread that forces all of it to behave like one market, by forbidding trades at worse-than-best displayed prices.

It is worth holding two facts in mind at once, because the temptation is to pick one and crusade. Fact one: the modern US equity market is, by almost any historical or international measure, *astonishingly* cheap and liquid for the ordinary investor — zero commissions, sub-penny effective spreads on big names, near-instant fills, routine price improvement. Fact two: that same market has grown so complex, so fast, and so opaque that nearly half its volume now trades where no quote is shown, on a structure whose every intermediary profits a little when the public view lags the private one. Both are true. A serious view of market structure refuses to collapse them into a simple "it's great" or "it's rigged," and instead asks the harder question the regulators are asking: *how much complexity and opacity is the cheapness worth, and where is the line past which price discovery quietly breaks?*

The deep point, and the one to carry back to the series spine: this entire baroque apparatus exists to make secondary trading *cheap, fast, and deep* — because deep secondary liquidity is the unglamorous foundation that lets the primary market raise capital at all. A company can sell a 30-year claim on its future only because a teacher in Ohio and a pension fund in Boston can both change their minds tomorrow morning, cheaply, in a market that has bent itself into knots to let them do so without crashing the price. The fragmentation, the darkness, the speed bumps and the latency arbs are not the market malfunctioning. They are the market *working* — paying, in complexity, for the liquidity that the whole edifice of capital formation quietly depends on.

The next time your app says "filled, \$50.015, \$0 commission," you will know it was never that simple — and now you know exactly why. Behind those three words sits a smart order router, a wholesaler, a dark pool or two, sixteen lit exchanges, a trade-through rule, and a tape racing to catch up with all of them. That is the secondary market: a fantastically elaborate machine whose entire purpose is to make trading feel boringly, reassuringly easy.

## Further reading & cross-links

- [Inside an exchange: the matching engine and the order book](/blog/trading/capital-markets/inside-an-exchange-the-matching-engine-and-the-order-book) — how a single lit venue actually matches buyers and sellers.
- [Market makers and the spread: who provides liquidity](/blog/trading/capital-markets/market-makers-and-the-spread-who-provides-liquidity) — the firms quoting the bids and offers you trade against.
- [How a price is made: discovery, arbitrage, and efficiency](/blog/trading/capital-markets/how-a-price-is-made-discovery-arbitrage-and-efficiency) — the price-discovery mechanism that fragmentation stresses.
- [Stock exchanges and clearinghouses](/blog/trading/finance/stock-exchanges-and-clearinghouses) — the venues and the post-trade plumbing behind every fill.
- [Order-book simulator (quant research)](/blog/trading/quantitative-finance/order-book-simulator-quant-research) — the formal microstructure and adverse-selection models we link out to here.
- [Foreign flows, ETFs, and the index effect in Vietnam](/blog/trading/vietnam-stocks/foreign-flows-etfs-and-the-index-effect-vietnam) — what trading looks like in a less-fragmented market.
