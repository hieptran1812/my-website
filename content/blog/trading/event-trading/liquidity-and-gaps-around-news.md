---
title: "Liquidity and Gaps: Why Prices Jump When the News Hits"
date: "2026-06-16"
publishDate: "2026-06-16"
description: "In the seconds around a macro release, market makers pull their quotes and the order book thins to a vacuum, so the print pushes price through empty space — creating gaps, slippage, and stop runs. Here is the microstructure of that vacuum and how to protect your execution."
tags: ["event-trading", "macro", "market-microstructure", "liquidity", "order-book", "slippage", "price-gap", "stop-run", "market-orders", "limit-orders", "crypto", "trading"]
category: "trading"
subcategory: "Event Trading"
author: "Hiep Tran"
featured: true
readTime: 41
---

> [!important]
> **TL;DR** — In the seconds around a macro release, **market makers cancel their quotes** to avoid being run over, so the order book thins to a near-vacuum; when the number prints, the resulting orders push price through *empty* levels, which is why you get **gaps**, **slippage**, and **stop runs** instead of a smooth move.
>
> - **What happens:** liquidity providers withdraw their resting orders just before the print to dodge adverse selection, so the book that absorbed your order at 8:29 is gone at 8:30:00 — a market order then fills *terribly*, sweeping through prices that have no resting volume behind them.
> - **The cross-asset map:** the gap hits everything at once. On Aug 5 2024 a single overnight cascade gapped the Nikkei **−12.40%**, the S&P 500 **−3.00%**, the Nasdaq 100 **−3.43%**, and Bitcoin **−15%** — but the *mechanics* differ: 24/7 markets (crypto, FX) reprice live through the news, while session markets (cash equities) are shut and gap over the close or weekend.
> - **The trade:** never feed a market order into the vacuum. Use **limit orders**, wait out the first seconds, treat stops as the market orders they really are, and size for the gap — not the average day.
> - **The one number to remember:** your stop does not guarantee your exit *price* — it guarantees an *attempt*; the fill can be 1% or more beyond the level when it triggers into a thin book.

On the morning of a US CPI release a couple of years ago, a trader I know placed what he thought was a careful trade. He was long a major index ETF, and to cap his risk he set a stop-loss order at exactly \$100.00 — a clean, round level a couple of percent below the market. His logic was textbook: if the print came in hot and the market fell, the stop would sell him out at \$100 and his loss would be capped. He went to make coffee.

The 8:30 a.m. print was hot. The index fell hard in the first second. His stop triggered — and it filled at **\$98.97**. Not \$100. He had sold a full **1% below his own stop level**, on a move that took less than two seconds. He stared at the fill, certain it was a broker error. It was not. His stop did exactly what a stop is built to do: at \$100 it turned into a *market* order, and that market order arrived into a book that, for those two seconds, was almost empty. There was nobody bidding at \$100, or \$99.80, or \$99.50. The first resting bid with any size was down near \$99. His sell order walked down the ladder, taking every thin bid it could find, until it filled — a full point lower than where it triggered.

This post is about *why* that happens, and it happens to everyone who trades around scheduled news without understanding the plumbing. The move you see on the chart looks like a clean candle. Underneath it is a violent, mechanical process: in the seconds around a release, the people whose job is to provide liquidity *take it away*, the order book hollows out into what traders call a **liquidity vacuum**, and the first orders to arrive push price through the empty space — fast, far, and with no trades in between. That is what a *gap* is. That is what *slippage* is. And that is why a stop placed in the wrong spot becomes a guaranteed bad fill. Let us build the whole thing from zero.

![Timeline of a liquidity vacuum: full book, makers pull quotes, the print hits empty space and price gaps, the book refills at a new level](/imgs/blogs/liquidity-and-gaps-around-news-1.png)

## Foundations: how liquidity actually works

Before we can explain why prices jump, we have to be precise about what "liquidity" *is*. Most beginners use the word as a vague synonym for "how much trading is going on." That is not wrong, but it is not sharp enough to explain a gap. To understand a gap you have to look at the actual machine that matches buyers and sellers — the **order book** — and watch what happens to it in the seconds around a print.

If you want the *macro* reason these releases move markets at all — why a CPI or jobs number is the thing the whole world is waiting for — the companion piece in the macro series, [the macro calendar of CPI, NFP, FOMC and PMI](/blog/trading/macro-trading/the-macro-calendar-cpi-nfp-fomc-pmi), lays out the mechanism. Here we are zoomed all the way in on the *plumbing*: what the act of trading looks like in the milliseconds when a number drops.

### The order book

Every modern exchange — for stocks, futures, FX, or crypto — matches trades through an **order book**: a live, sorted list of every resting buy order and sell order, organized by price. On the buy side sit the **bids**: standing offers to buy a certain quantity at a certain price. On the sell side sit the **asks** (also called offers): standing offers to sell. The book is sorted so the highest bid and the lowest ask sit at the top, facing each other.

Say a stock's book looks like this: the best bid is \$50.00 for 1,000 shares, with more bids stacked below at \$49.99, \$49.98, and so on; the best ask is \$50.01 for 1,200 shares, with more asks stacked above at \$50.02, \$50.03. When you place a **market order to buy**, you are saying "fill me immediately at whatever price the book offers." Your order is matched against the lowest asks first — you take the 1,200 shares at \$50.01, then if you need more you climb to \$50.02, then \$50.03. Each level you climb is a slightly worse price. The book is a staircase, and a big order walks up (or down) the stairs.

### The bid-ask spread

The gap between the best bid and the best ask is the **bid-ask spread**. In our example it is \$50.01 − \$50.00 = \$0.01, one cent — a "tight" spread, the sign of a liquid, healthy market. The spread is the cost of immediacy: if you buy at the ask (\$50.01) and immediately sell at the bid (\$50.00), you lose the spread. Market makers — the firms that quote both a bid and an ask continuously — earn that spread as their compensation for standing ready to trade. A *wide* spread (say, \$50.00 bid, \$50.50 ask) means liquidity is scarce and the cost of trading right now is high.

#### Worked example: what a blown-out spread costs to cross

Say you hold a \$20,000 position and you want out *right now*, in the seconds after a print. On a normal day the spread is one cent on a \$50 stock — crossing it costs you \$0.01 ÷ \$50 = 0.02%, or about 0.02% × \$20,000 = **−\$4**. In the vacuum the spread has blown out to \$0.50 (a \$50.00 bid against a \$50.50 ask), so crossing it costs \$0.50 ÷ \$50 = 1.0%, or 1.0% × \$20,000 = **−\$200**. The spread alone — before any further slippage from walking the staircase — got **50× more expensive** purely because makers widened their quotes around the event. The intuition: the spread is the entry fee to trade *immediately*, and around news the bouncer raises the price of immediacy by orders of magnitude, so paying for immediacy is exactly what you want to avoid.

### Market depth

**Depth** is how much size rests in the book *near* the touch (the best bid and ask). A deep book has thousands of shares stacked within a few cents of the top, so even a large order barely nudges the price. A thin book has only a handful of shares at each level, so a modest order sweeps several levels and moves the price meaningfully. Depth is the shock absorber. The deeper the book, the more an arriving order is *absorbed* rather than *amplified*. Two markets can have the identical best bid and ask yet completely different depth — and depth is precisely what evaporates around a news release.

Walk the staircase once to make depth concrete. Suppose the ask side reads: 1,200 shares at \$50.01, then 1,500 at \$50.02, then 2,000 at \$50.03, and so on. A market buy for 1,000 shares fills entirely at \$50.01 — you never even climb a step, and your average price is \$50.01, basically the touch. Now suppose a thin book: 100 shares at \$50.01, 100 at \$50.05, 100 at \$50.20, nothing until \$51.00. The *same* 1,000-share order now takes 100 at \$50.01, 100 at \$50.05, 100 at \$50.20, and 700 at \$51.00 — an average fill near \$50.86, almost a full point above the touch you saw. The order is identical; only the depth changed. That is the entire mechanism of a vacuum in miniature: the staircase loses its lower steps, so the order falls to the bottom. The reason this is so dangerous around news is that the book *looks* fine right up until the print — the screen shows a tight \$50.00/\$50.01 quote — and then, in the half-second before 8:30:00, the steps vanish and a quote that *looked* deep was actually a façade with nothing behind it.

There is a related quantity professionals track called **market impact** or the **slippage curve**: the function mapping order size to the average price you'll achieve. A deep book has a shallow, nearly flat impact curve — size barely matters. A thin book has a steep, convex impact curve — every extra unit of size costs disproportionately more. Around a release, the impact curve does not just shift; it *steepens dramatically*, so the size of order that was harmless at 8:25 is ruinous at 8:30:00. The same dollars, the same intent, a wildly different cost — because the curve under them tilted vertical for a few seconds.

![Order book before and at a release: a deep two-sided book versus a thin one-sided book](/imgs/blogs/liquidity-and-gaps-around-news-2.png)

### The liquidity vacuum

Here is the central concept of this whole post. A **liquidity vacuum** is a brief window — usually a few seconds around a scheduled release — when market makers *cancel* their resting orders, so the book that was deep a moment ago is suddenly almost empty. The bids and asks that would normally absorb an order are simply *not there*. The book does not just get thinner; it can go nearly one-sided, with a few token orders far from the last price and a wide-open hole in between. An order arriving into that hole does not get absorbed — it falls straight through.

The reason makers do this is the single most important idea in event microstructure, and we will spend a full section on it below: it is **adverse selection**. For now, just hold the picture: the book breathes. It is full and tight at 8:29:00, it hollows out at 8:29:5x as makers pull quotes, and it stays a vacuum through the print until liquidity providers feel safe enough to come back — usually within seconds, but those seconds are where the damage happens.

### Slippage

**Slippage** is the difference between the price you *expected* and the price you actually *got*. If you fire a market order expecting to buy at \$50.01 (the ask you saw on your screen) but the order fills at an average of \$50.06 because it had to climb the staircase, you slipped 5 cents — roughly 0.1%. Slippage is always against you on a market order, and it grows with two things: the *size* of your order (bigger orders climb more stairs) and the *thinness* of the book (fewer stairs to stand on). In a liquidity vacuum, both effects compound: you are sending a normal-sized order into an abnormally thin book, so the slippage can be enormous — tenths of a percent, or whole percent, instead of fractions of a cent.

### A price gap

A **gap** is a discontinuity in price: the market trades at one level, then the *next* trade prints at a meaningfully different level, with **no trades in between**. On a candlestick chart, a gap looks like a literal hole — the close of one bar and the open of the next sit at different prices with empty space between them. Gaps are the visible scar of a liquidity vacuum: price did not *travel* from \$100 to \$98.5; it *jumped*, because there was nothing to trade against in the middle. There were no fills at \$99.50 or \$99.20 because there were no resting orders there to fill against.

### A stop run

A **stop order** (or stop-loss) is an instruction that sleeps until price touches a trigger level, then *wakes up and becomes a market order*. People use stops to cap losses: "if it falls to \$100, get me out." A **stop run** (or stop sweep) is what happens when price reaches a cluster of stops, the stops all trigger into market orders at once, those market orders push price further in the same direction, which triggers *more* stops, and so on — a cascade. In a thin book, a stop run is brutal: the stops fire into a vacuum, each fill is worse than the last, and price can sweep far past the stop level in a flash before liquidity returns and it snaps back. This is the mechanism behind the \$98.97 fill in the opening story.

### Market orders versus limit orders

This distinction is the difference between protecting your execution and getting run over, so be precise about it:

- A **market order** says: *fill me now, at any price the book offers.* It guarantees execution but not price. In a vacuum, "any price" can be catastrophically bad.
- A **limit order** says: *fill me only at this price or better.* It guarantees price but not execution — if the book never trades at your limit, you simply do not get filled. A limit order *cannot* slip past your chosen price, because it refuses to.

A stop-loss is, by default, a *market* order in disguise: it rests harmlessly until triggered, then converts to a market order and inherits all of the market order's danger. This is why "I have a stop, so I am protected" is a half-truth — you are protected from *forgetting to exit*, but not from a *terrible fill*.

### 24/7 versus session markets

The last foundation is *when* markets are open, because it determines whether a gap shows up *intraday* or *over a close*.

- **24/7 markets** never close. Crypto trades every second of every day, including weekends. The major FX (foreign exchange) market runs essentially around the clock from Sunday evening to Friday evening. In these markets, a release at any hour reprices *live* — you watch the move happen in real time, vacuum and all.
- **Session markets** have fixed hours. Cash equities (ordinary stocks) trade roughly 9:30 a.m. to 4:00 p.m. in their home time zone, then close. News that breaks while the market is shut — overnight, over a weekend, on a holiday — cannot be traded. It piles up, unpriced, until the next open, when the market reopens at whatever level reflects all the accumulated news. That single jump is the **opening gap**, and it is the session-market version of the vacuum: the "vacuum" is the entire closed period.

With those nine terms defined, we can now explain the move.

## Why market makers pull their quotes before a release

The whole vacuum starts with a rational decision by the firms that provide liquidity. Understanding *why* they pull quotes is what turns "the market is weird around news" into "of course the book empties out — I would empty it too."

A market maker earns the bid-ask spread by quoting both sides continuously and capturing the small edge between them, thousands of times a day. The business works because, most of the time, the flow they trade against is *uninformed* — a buyer here, a seller there, roughly balanced, none of whom knows something the maker does not. The maker pockets the spread and stays roughly flat.

The killer for this business is **adverse selection**: trading against someone who knows more than you do. If a counterparty buys from you at your ask *right before* the price jumps up, you have just sold something cheaply that you will have to buy back expensively — you got "picked off." Adverse selection is the maker's nightmare, and a scheduled release is the moment of *maximum* adverse selection risk. At 8:30:00.000, a number drops that can move price 1–3% in a heartbeat, and the fastest participants — the firms with the lowest-latency connections to the data feed — will trade against every resting quote in the microseconds before the maker can cancel. If the maker leaves a bid resting and the number is hot, that bid gets hit by sellers an instant before price collapses; the maker is now long, into a falling market, having bought at a price that is already stale.

No rational maker accepts that bet. So in the seconds before a scheduled release, makers do the only sensible thing: they **cancel their resting orders** and stand aside until the dust settles. They would rather miss a few seconds of spread income than get picked off on a 2% move. Multiply this across every maker on the book, and the book empties. That collective, simultaneous withdrawal *is* the liquidity vacuum.

There is a second, subtler driver. Even makers who *want* to quote through the event must quote *wider* to compensate for the risk — a \$0.01 spread becomes \$0.20 or \$0.50, because they are pricing in the chance of being run over. So the book does not just get thinner; the orders that remain sit *farther* from the last price. The touch (best bid/ask) gaps out, and the depth behind it vanishes. The market goes from a tight, deep staircase to a few scattered steps with huge holes between them.

It helps to know *who* the makers are, because it explains how fast and how completely the book empties. Modern liquidity is dominated by high-frequency market-making firms whose entire edge is reacting in microseconds. They are not heroes who stick around to provide stability; they are profit-maximizing algorithms, and their risk controls *force* them to cancel ahead of known events. Many run an explicit rule: pull all quotes a fixed number of milliseconds before any scheduled release on the economic calendar, and do not requote until a configurable delay after the print. Because the calendar is *public* — everyone knows CPI is at 8:30 a.m. on its release day — every maker's cancel logic fires at essentially the same instant. The withdrawal is not gradual or staggered; it is a coordinated, calendar-driven evaporation. That synchronization is why the vacuum is so total: it is not that *some* liquidity leaves, it is that *almost all of it* leaves at the same moment, by design.

The flow that arrives during a release is also unusually **toxic** — a term of art for order flow that is informed and therefore dangerous to trade against. Around a print, the orders hitting the book are disproportionately from the fastest informed players reacting to the number, plus a wave of mechanical stops and forced liquidations. Almost none of it is the benign, two-sided retail flow makers like to capture. Makers know this, which is a third reason to stand aside: not only is the price about to move, but the people they'd be trading against in those seconds are precisely the ones most likely to be right. Re-entering early means volunteering to be the uninformed side of a one-sided, informed flush. So they wait — and the book stays hollow until the informed surge has spent itself.

#### Worked example: the maker's adverse-selection math

Say a market maker quotes a stock at \$50.00 bid / \$50.01 ask for 1,000 shares each side, earning the \$0.01 spread. On a normal trade of 1,000 shares, the maker's gross edge is 1,000 × \$0.01 = **\$10**. Now a CPI print drops and the stock instantly falls to \$49.00. If the maker had left that \$50.00 bid resting and it got hit for 1,000 shares an instant before the drop, the maker is long 1,000 shares at \$50.00 in a \$49.00 market — an unrealized loss of 1,000 × (\$50.00 − \$49.00) = **−\$1,000**. One pick-off wipes out **100 trades' worth** of spread income (\$1,000 ÷ \$10). The intuition: the maker is risking \$1,000 to earn \$10, so the only sane move into a known event is to cancel and disappear.

That math, run by every maker at once, is the entire reason the book hollows out — and the reason the *next* order, yours, fills into empty space.

## The vacuum and the gap: price jumps with no trades in between

Now connect the two. The makers have pulled their quotes; the book is a vacuum. The number prints. What happens to price?

Whatever orders *do* arrive — fast algorithms repricing the surprise, stops triggering, a few brave humans — meet a book with almost no resting volume. A buy order does not find sellers at \$50.02 and \$50.03; it finds nothing until \$51.00, so it lifts price to \$51.00 in a single, near-instant move. There were *no trades* at \$50.10, \$50.40, \$50.70 — not because nobody wanted those prices, but because there were no resting orders to trade against. Price *teleported*. That teleport is the gap.

This is why a news candle on a chart is misleading. It looks like price *swept* from \$50 to \$51, implying a continuous walk through every price in between. It did not. It *jumped*, and the "body" of the candle is mostly empty space that no trade ever occurred in. If you had a limit order resting at \$50.50, you might never have been filled — price skipped your level entirely. If you had a *stop* at \$50.50, it triggered and chased the move up to \$51.00, filling at the worst possible spot.

The size of the gap is set by how thin the book was and how big the imbalance of arriving orders is. A small surprise into a moderately thin book produces a small gap; a large surprise into a near-total vacuum produces a violent one. And crucially, the gap is *front-loaded*: the first orders, hitting the emptiest book, move price the most. By the time liquidity providers tiptoe back in a few seconds later — quoting at the *new* level, because that is now fair value — the gap is locked in. The book refills, but it refills *around the new price*, not the old one.

A common follow-up question is whether the gap "fills" — whether price comes back to trade through the empty space it skipped. Sometimes it does, sometimes it does not, and the difference is *why* the gap happened. If the move was a knee-jerk overshoot in a vacuum, with no real change in fundamentals, the gap often gets retraced once calm liquidity returns and fades the overshoot — the empty levels get traded through on the way back. The Nikkei retracing more than three-quarters of its −12.4% crash the very next day is a textbook example of a vacuum overshoot snapping back. But if the print genuinely repriced fair value — a regime-changing CPI, a hawkish surprise that resets the whole rate path — the gap is *information*, not noise, and price has no reason to revisit the old level. It simply trades on from the new one. So "do gaps fill?" has no universal answer; it depends on whether the gap was the market overshooting in thin conditions or honestly repricing new facts. The microstructure tells you the *mechanism* of the jump; only the fundamentals tell you whether it *sticks*.

One more subtlety: even after the touch gaps, your order can suffer **partial fills** that compound the pain. Say you send a market sell for 5,000 shares into a thinning book. The first 500 fill at the touch; by the time the next chunk routes, makers have pulled the rest and price has dropped, so the next 1,000 fill lower, and the remainder lower still. You did not get one clean bad price — you got a *smear* of progressively worse prices, with your own order pushing the market away from you as it consumed each thin level. Large orders are self-defeating in a vacuum precisely because they advertise their own urgency: every level you take signals more selling to come, so the remaining liquidity retreats further. This is the convex slippage curve playing out in real time, fill by fill.

![Anatomy of a gap: prior close, a jump to the new open with no trades between, and a stop swept through the empty space](/imgs/blogs/liquidity-and-gaps-around-news-4.png)

#### Worked example: a \$25,000 market order into the vacuum

Say you fire a market buy order for \$25,000 of an asset the instant a release prints. On a normal day, the book is deep and you fill within a hair of the touch — slippage of maybe 0.02%, or about −\$5. But you sent it into the vacuum. The book is thin, your order climbs several empty levels, and your average fill lands **0.4% above** the price you saw on screen. Your slippage is \$25,000 × 0.004 = **−\$100**. You paid \$25,100 worth of price for \$25,000 of intended exposure, and \$100 evaporated before the position even started working. (The 0.4% figure is illustrative — actual vacuum slippage varies with the asset and the surprise — but the direction is always the same: into a thin book, the cost balloons.) The intuition: the same order that costs you \$5 at 8:25 can cost you \$100 at 8:30:00, and the *only* thing that changed is the depth of the book.

This is the core reason event traders treat the first few seconds after a print as untouchable. The move is real, but the *execution* is a trap.

## Slippage and stop runs: why market orders and stops are dangerous at 8:30:00

We now have the two most expensive mistakes in event trading, and they are really the same mistake wearing two costumes.

### The market-order trap

A market order is a promise to take *any* price. That promise is harmless when the book is deep, because "any price" is within a cent of where you're looking. In a vacuum, "any price" is a blank check, and the book cashes it for the worst price available. The bigger your order, the deeper into the empty book it reaches, and the worse the average fill — slippage scales with size *and* with thinness, and both work against you at 8:30:00.

![Slippage cost rising with market-order size in a thin book, an illustrative example](/imgs/blogs/liquidity-and-gaps-around-news-6.png)

The chart above is illustrative, not a market quote — but the *shape* is universal. A small order (\$5,000) barely dents even a thin book and slips a little; a large order (\$100,000 or \$250,000) reaches far down the empty ladder and slips badly. The relationship is convex: doubling your size more than doubles your slippage, because each additional dollar fills at a worse level than the last. This is exactly why large institutions *never* send a single market order into a release — they slice it, or they wait, or they post limits. The retail trader who slams "market buy" the instant a number prints is paying the convex tail of that curve.

#### Worked example: slippage scaling with size (illustrative)

Take the illustrative numbers above. A \$5,000 market order slips 0.10% → cost \$5,000 × 0.001 = **−\$5**. A \$25,000 order slips 0.40% → \$25,000 × 0.004 = **−\$100**. A \$100,000 order slips 1.20% → \$100,000 × 0.012 = **−\$1,200**. A \$250,000 order slips 2.50% → \$250,000 × 0.025 = **−\$6,250**. Notice the cost per dollar of size rises the whole way: the \$5k order cost 0.1%, the \$250k order cost 2.5% — **25× worse per dollar**, for an order only 50× larger. The intuition: in a vacuum, size is not your friend; the book punishes scale, so the urgent, large market order is the single most expensive way to trade an event.

### The stop-run trap

Now recall that a stop-loss *is* a market order — it just sleeps until triggered. So everything above applies to stops, with an extra cruelty: stops *cluster*. Traders put them at obvious places — round numbers (\$100), recent lows, just below support. When price reaches that cluster, all the stops trigger at once, all become market sells at once, and they hit the thin book simultaneously. That wave of selling pushes price lower, which triggers the *next* layer of stops, which pushes price lower still. The result is a stop run: a fast, self-reinforcing flush that sweeps far past the trigger level, then often snaps back once the stops are exhausted and liquidity returns.

The trader from the opening got caught in exactly this. The stop at \$100 was a *market* order waiting to happen. When price touched \$100, it converted, joined a wave of other stops, and filled at \$98.97 — a full point of slippage — into a book that had no bids between \$100 and \$99. The "guarantee" of the stop was a guarantee to *try* to exit, not a guarantee of the *price*.

#### Worked example: a stop that fills a point below its level

You are long a \$10,000 position and set a stop at \$100, expecting to lose only the distance to your stop. The print is hot, price gaps through the cluster, and your stop fills at \$98.50 instead of \$100 — 1.5% of slippage beyond your level. On your \$10,000 position, that extra 1.5% is \$10,000 × 0.015 = **−\$150** *beyond* your plan. You budgeted to lose to \$100; you actually exited at \$98.50, so the vacuum cost you an unplanned **−\$150** on top of the loss you'd already accepted. The intuition: a stop caps the *level* at which you start trying to exit, not the *price* you achieve — and in a vacuum the achieved price can be a full percent worse, so size your risk on the *fill*, not the trigger.

The defense, which we will formalize in the playbook, is the **stop-limit** order: a stop that, when triggered, becomes a *limit* order instead of a market order, refusing to fill worse than a price you set. The trade-off is real — if price gaps clean past your limit, the stop-limit does not fill at all, and you are left holding the position. There is no free lunch: a plain stop guarantees the exit but not the price; a stop-limit guarantees the price but not the exit. Around news, you must consciously choose which risk you are willing to bear.

## 24/7 crypto and FX versus cash-equity sessions: where the gap shows up

The vacuum is universal, but *where* you see the gap depends on whether the market ever closes. This is one of the most practically important distinctions in event trading, because the same news produces a different-looking chart in crypto versus stocks.

![Two-track timeline: 24/7 markets reprice through the news while session markets gap over the close](/imgs/blogs/liquidity-and-gaps-around-news-5.png)

### 24/7 markets: the vacuum is intraday

Crypto trades every second, including weekends and holidays. FX trades essentially around the clock on weekdays. In these markets, a release reprices *live*. You watch the move happen: the vacuum opens, the print hits, price slides or rips, the book refills — all in real time, on your screen. The advantage is that there is no overnight discontinuity; you can react (or get run over) in the moment. The disadvantage is that the *thinnest* books — weekend crypto liquidity, the FX market in the gap between the New York close and the Tokyo open — are *exactly* when major news has historically broken, and a vacuum in an already-thin weekend book produces some of the most violent gaps you will ever see. A macro shock that hits crypto on a Saturday afternoon faces almost no liquidity to absorb it, so it gaps hardest precisely when you can do the least about it.

The macro series develops the idea that crypto behaves like a high-beta liquidity sponge — it amplifies the same risk-on/risk-off impulse that moves everything else — in [crypto as a macro liquidity asset](/blog/trading/macro-trading/crypto-as-a-macro-liquidity-asset). Around an event, that beta plus a thin 24/7 book is why Bitcoin's gaps so often dwarf the equity move on the same news.

### Session markets: the vacuum is the whole closed period

Cash equities close. When news breaks overnight, over a weekend, or on a holiday, there is no market to trade it — the book is not just thin, it is *gone*. The news accumulates, unpriced, until the next open. Then the market reopens at the single price that clears all the orders that built up while it was shut, and *that* is the **opening gap**. It is the session-market analog of the vacuum, except the "vacuum" lasted the entire closed period and the gap is the one print that resolves it.

This is why a US stock can close Friday at \$100 and open Monday at \$95 having "never traded" between — no trade printed at \$99, \$98, \$97, or \$96, because the market was closed. Anyone holding over the weekend could not exit during the move; their first opportunity to act is the gapped open itself. A stop resting in that range does not protect you mid-move — it triggers *at the open*, filling at the gapped price, not the stop level.

There is a partial bridge: many equities trade in *pre-market* and *after-hours* sessions, and futures on the major indices trade nearly around the clock. So a savvy participant can see (and trade) some of the reaction before the cash open. But those extended sessions are *themselves* thin — low volume, wide spreads, their own mini-vacuums — so they preview the gap rather than eliminate it. The cash open is still where most of the volume, and most of the gap, lands.

FX sits in an interesting middle ground. The currency market trades nearly around the clock on weekdays, handing off between the Sydney, Tokyo, London, and New York sessions, so a weekday release reprices live. But FX *does* close over the weekend, from the New York close on Friday to the Sydney open on Sunday evening. That creates a recurring vulnerability: a geopolitical shock or a surprise policy move over a weekend cannot be traded until the Sunday-evening reopen, and the first prints into that reopen — the thinnest, sleepiest liquidity of the entire week — can gap hard. The August 2024 yen episode is the cautionary tale: USD/JPY had peaked near 161.9 in early July and collapsed toward 141.7 by August 5, and much of that violence came in low-liquidity windows where the carry unwind met a book with almost nobody on the other side.

Vietnam's equity market shows the session-gap effect with its own twist. The HOSE exchange runs fixed sessions with explicit opening (ATO) and closing (ATC) auction phases — discrete moments designed to *aggregate* orders into a single clearing price rather than trade continuously. News that breaks while the market is closed — an overnight US move, an SBV policy announcement, a foreign-flow shift — gets absorbed into the VN-Index's *next open* as a gap, exactly like any other session market. When the State Bank of Vietnam hiked its refinancing rate from 4.0% to 6.0% over autumn 2022 to defend the dong, and foreign investors sold heavily, the VN-Index gapped lower across multiple sessions on its way from roughly 1,530 in January 2022 down to a trough of 911 by November 15 2022. The mechanism is identical to the US case: a closed market cannot price news in real time, so it discharges the accumulated move as a gap at the open, and a tight stop sitting in the gap range fills at the gapped price, not the level you set. (The macro series covers the policy side in [Vietnam's monetary policy](/blog/trading/finance/vietnam-monetary-policy-state-bank-dong-credit-ceiling).)

#### Worked example: a weekend crypto gap you cannot exit

You hold \$8,000 of an unhedged crypto position over a weekend. Saturday afternoon, a macro headline hits — and crypto, trading 24/7 into a thin weekend book, gaps **−5%** in minutes with almost no liquidity to absorb it. Your loss is \$8,000 × 0.05 = **−\$400**. The cruel part is not just the size; it is that you *could not act*. There was no deep bid to sell into during the slide, and if you'd had a stop it would have filled deep in the hole. Contrast a stock: had this been equity news on a Saturday, the market would be *closed*, you'd have lost nothing on paper until Monday — but you'd then face the gap at the open with the same inability to exit mid-move. The intuition: 24/7 markets let the gap happen *while you watch*; session markets defer it to the open — either way, the move occurs in a window where you cannot get a good fill, so the protection has to be put on *before* the event, not during it.

## How it reacted: real episodes

Theory is cheap. Here are two dated episodes where the vacuum, the gap, and the cross-asset cascade played out with real numbers.

### Aug 5 2024: one gap hit every market at once

The first week of August 2024 produced one of the cleanest demonstrations of a synchronized, cross-asset gap in modern markets. The setup was a classic carry-trade unwind: the Bank of Japan hiked on July 31, the US jobs report on August 2 came in weak (+114,000 versus ~175,000 expected, with the unemployment rate jumping to 4.3%), and the yen — which traders had borrowed cheaply to fund risk positions worldwide — began to rip higher. Leverage that had been built on a stable, cheap yen suddenly had to be unwound all at once. (The macro series traces the full mechanism in [carry-trade unwinds: when leverage breaks](/blog/trading/macro-trading/carry-trade-unwinds-1998-2008-2024-when-leverage-breaks).)

When Tokyo opened on Monday August 5, the floodgates broke. The Nikkei 225 fell **−12.40%** in a single session — its worst day since the 1987 Black Monday crash. The cascade then rolled around the globe with the sun: the S&P 500 fell **−3.00%**, the Nasdaq 100 **−3.43%**, and Bitcoin — trading 24/7, with no close to hide behind — fell about **−15%**. The VIX, the equity-market "fear gauge," spiked to an intraday **65.73**, a level seen only in genuine crises, having closed at just 23.4 the prior Friday.

![Same-day moves on August 5 2024 across the Nikkei, S&P 500, Nasdaq 100, and Bitcoin](/imgs/blogs/liquidity-and-gaps-around-news-3.png)

The microstructure point is what unites all four bars. This was not four separate, gradual sell-offs. It was one liquidity event. As the forced unwind hit, makers across every market pulled their quotes — nobody wants to stand in front of a cascade — and the books hollowed out simultaneously. Orders arriving into those vacuums gapped each market through empty space. The session markets (Nikkei, S&P, Nasdaq) gapped at their *opens*; the 24/7 market (Bitcoin) gapped *live and hardest*, because a forced global deleveraging hitting a 24/7 book with no off-switch is the purest vacuum there is. And the snap-back was equally telling: the Nikkei rebounded **+10.23%** the very next day, because once the forced sellers were exhausted, liquidity returned and price retraced much of a move that had overshot in the vacuum.

#### Worked example: the Aug 5 gap on a real position

Say you held a \$25,000 S&P 500 index position into August 5 2024. The **−3.00%** day cost you \$25,000 × 0.03 = **−\$750** — but if your exposure was via a leveraged or crypto-correlated vehicle, the math gets worse fast. The same \$25,000 in Bitcoin that day, at **−15%**, would be \$25,000 × 0.15 = **−\$3,750**. And here is the vacuum tax on top: if you'd tried to sell either position *into* the cascade with a market order, you'd have eaten gap slippage on the way out — easily another half-percent or more on a thin, panicked book, so the realized loss exceeds the headline percentage. The intuition: the headline move is the *floor* of your loss when you transact in a vacuum, not the ceiling; the gap and the slippage stack on top of the number you read in the news.

### CPI-day opening gaps: the scheduled vacuum

The August 2024 cascade was an *unscheduled* shock. But the most reliable vacuum of all is the *scheduled* one: the 8:30 a.m. US CPI release, which lands like clockwork once a month, an hour before the cash equity open. Because the print drops while the cash market is *closed*, equities cannot reprice it until 9:30 — so a hot or cool CPI produces a textbook *opening gap*.

The reaction depends entirely on the surprise and the regime (the [reaction function](/blog/trading/event-trading/the-reaction-function-why-the-same-number-moves-differently) governs the sign). In the inflation-panic regime of 2022, a hot August CPI (released September 13 2022: 8.3% versus 8.1% expected) sent the S&P 500 down **−4.32%** on the day, the Nasdaq down **−5.16%**, and Bitcoin — trading through the print with no close — down about **−9.4%**. Two months later, a *cool* October CPI (released November 10 2022: 7.7% versus 7.9% expected) did the mirror image: the S&P jumped **+5.54%**, the Nasdaq **+7.35%**, the 10-year Treasury yield fell 28 basis points, and Bitcoin rallied about **+10%**.

The microstructure is the same on both days. The futures and 24/7 markets reprice the surprise *instantly* at 8:30:00 — gapping through their vacuums in the first second — while the cash equity market, still closed, *waits* and then gaps at 9:30 to reflect the move futures already made. A trader watching only cash equities sees a single jump at the open; a trader watching futures and crypto saw the whole gap happen at 8:30:00, in a vacuum, in under a second. (For how a single print propagates across futures, FX, rates, and crypto in sequence, see [cross-asset transmission: how one print hits every market](/blog/trading/event-trading/cross-asset-transmission-how-one-print-hits-every-market).)

#### Worked example: the cool-CPI gap as a missed limit fill

Say on November 10 2022 you wanted to *buy* the S&P 500 dip and had a limit buy resting just *below* the prior close, hoping for one more flush before the print. The cool CPI hit, futures gapped up instantly, and the cash market opened **+5.54%** higher. Your limit — sitting below the prior close — was never touched; price gapped *over* it. On a \$50,000 intended position, you missed a +5.54% move = \$50,000 × 0.0554 = **+\$2,770** of upside you never captured, because the gap skipped your level entirely. The intuition: gaps cut both ways for limit orders — they protect you from bad fills on the wrong side, but they also *strand* you when price teleports past your level on the right side; around a binary event, "wait for my price" can mean "never get filled."

## Common misconceptions

A handful of beliefs cause most of the avoidable damage around news. Each is corrected with the mechanism.

### "My stop guarantees my exit price."

This is the most expensive myth in trading, and the whole post has been building to it. **A stop guarantees an *attempt* to exit, not a *price*.** When triggered, a plain stop becomes a *market* order, and a market order takes whatever the book offers. Into a vacuum, that can be a full percent or more beyond your level — as in the opening story, where a \$100 stop filled at \$98.97. If you need price certainty, use a **stop-limit** and accept the opposite risk: that a clean gap leaves you unfilled. There is no order type that gives you both guaranteed exit *and* guaranteed price across a gap; physics forbids it, because there has to be a counterparty at your price for you to fill there, and in a vacuum there isn't one.

### "The big news candle means price traded at every level inside it."

No. A news candle's body is mostly *empty space* that no trade occurred in. Price gapped from one level to another; the candle just draws a rectangle connecting them. This matters because traders treat the candle's midpoint as "support" or a "fair" re-entry, when in fact *no liquidity ever existed there*. The level is an artifact of the chart, not a real price the market agreed on.

### "More volume means more liquidity, so news days are easy to trade."

Volume and liquidity are not the same thing. News days have *enormous* volume — but the volume arrives *after* the gap, at the new level, transacted by participants who all want the same direction at once. *During* the gap, in the vacuum, there is almost no resting liquidity. High daily volume coexists with a near-empty book in the critical first seconds. You can trade a *lot* on a news day and still get a terrible fill, because the fat volume bar and the thin instantaneous book describe different moments.

### "I'll just use a market order to be sure I get out."

"Getting out" at *any* price is rarely what you actually want — you want out at a *reasonable* price. A market order into a vacuum can fill so far from your intended level that the "protection" costs more than riding the move would have, especially since vacuums often *overshoot* and snap back (the Nikkei's −12.4% day was followed by +10.23%). Selling the absolute bottom of a vacuum with a market order locks in the worst tick of an overshoot. The disciplined move is usually to size the position so you *don't need* to panic-exit into the vacuum at all.

### "24/7 markets are safer because I can always exit."

The opposite is closer to true. *Being open* is not the same as *being liquid*. A 24/7 market can be open on a Saturday with almost no depth, so "I can always trade" becomes "I can always trade into a vacuum." The August 5 2024 Bitcoin gap of −15% happened in a continuously open market — being open did not save anyone; it just meant the gap played out live instead of over a close. Continuous trading removes the *overnight* gap but replaces it with the risk of a thin-session vacuum at any hour.

## The playbook: how to protect execution around events

Everything above reduces to one rule: **do not feed a market order into the vacuum.** Here is the operational playbook, built from that rule.

![Execution playbook around a release: wait, use limits, control stops, size for the gap, then confirm](/imgs/blogs/liquidity-and-gaps-around-news-7.png)

**1. Wait out the first seconds.** The vacuum is briefest right at the print and refills within seconds-to-minutes as makers return. Unless you have a low-latency edge (you don't, if you're reading this for the explanation), the worst time to transact is 8:30:00 to roughly 8:30:30. Let the first knee-jerk pass, let liquidity come back, and act once the spread has normalized. The companion piece on [the spike, the fade, and the trend](/blog/trading/event-trading/anatomy-of-a-news-reaction-spike-fade-trend) makes the same point from the reaction side: the *first* move is the knee-jerk into the vacuum, and the *second* move — once the book refills and humans read the internals — is usually the more honest one to trade.

**2. Use limit orders, not market orders.** Name the price you are willing to pay or accept. A limit order *cannot* slip past your number, because it refuses to. Yes, you risk not getting filled if price runs away — but "not filled" is a far better outcome than "filled 1% worse." If you must get in, use a *marketable limit* (a limit a few ticks through the touch): it fills like a market order in a normal book but caps your worst price, so a vacuum can't take you for an unbounded ride.

**3. Control your stops.** A plain stop is a market order into the vacuum. Three defenses, in order of conservatism: (a) **widen** the stop so it sits beyond the likely vacuum range, accepting a larger but *known* loss instead of a smaller but *uncertain* one; (b) convert to a **stop-limit**, accepting the risk of no fill on a clean gap; or (c) **remove the stop entirely over the event** and rely on position sizing — appropriate only if your size is small enough that the worst-case gap is survivable. Never leave a tight plain stop sitting in the obvious cluster (round numbers, recent lows) into a scheduled print; that is volunteering to be the liquidity that the stop run consumes.

**4. Size for the gap, not the average day.** This is the master defense, because it makes every other one optional. If you size your position so that even a worst-case gap — a multiple of the average move — is a loss you can absorb, then you never *need* to panic-exit into the vacuum. Use the options market's **expected move** as your gauge: an at-the-money straddle priced at, say, \$60 on a \$5,000 index implies a ±1.2% move (\$60 ÷ \$5,000), and a real gap can be several times that. Size so that two or three expected moves against you is a controlled loss, not a catastrophe. (For how the options market prices that expected move and how it collapses after the event, see [the volatility surface](/blog/trading/quantitative-finance/volatility-surface).)

**5. Confirm before re-entering.** Once the vacuum has refilled — the spread is tight again, depth is back, the chart has printed a few honest bars at the new level — *then* you can trade the reaction with normal execution. The new level is real now; the gap is in the rearview. Trade what the market actually agreed on after the dust settled, not the empty space the candle drew during the vacuum.

#### Worked example: limit versus market on a \$50,000 entry

You want to enter a \$50,000 position right after a print. Option A: a market order into the vacuum, slipping 0.4% → cost \$50,000 × 0.004 = **−\$200** of slippage on entry. Option B: a patient limit order placed once the book refills, capturing a fill 0.3% better than the market-order fill → you save \$50,000 × 0.003 = **+\$150** versus the market order, and avoid the \$200 slip, for a swing of roughly **\$350** on a single \$50,000 trade. Do that 50 times a year and the difference between "limit, patient" and "market, urgent" is on the order of \$17,500 of pure execution edge — before you've made a single correct *directional* call. The intuition: around events, execution discipline is not a rounding error; it is one of the largest, most repeatable edges available to a non-latency trader, and it costs nothing but patience.

### The if-then summary

- **If** a scheduled release is imminent → pull tight plain stops out of the obvious cluster, switch pending entries to limits, and confirm your size survives a multi-expected-move gap.
- **If** the print just dropped (first ~30 seconds) → do *nothing* with market orders; let the vacuum refill.
- **If** you must exit during the vacuum → use a marketable limit to cap your worst fill, and accept you might not get out at all if it gaps clean past.
- **If** the book has refilled (tight spread, restored depth) → trade the reaction normally; the new level is real.
- **Invalidation:** if you find yourself about to send a market order in the first seconds after a print "to be safe," stop — that is the single action this entire post exists to prevent. The safety is in the sizing and the patience, never in the urgent market order.

The deepest lesson is that the move you fear and the *execution* of trading that move are two different problems. The gap is unavoidable — it is the market's honest repricing of new information through a thin book. But the bad *fill* is entirely avoidable: it is a self-inflicted wound from sending the wrong order type into a vacuum you could have simply waited out. Master the plumbing, and the news stops being a thing that happens *to* your account and becomes a thing you navigate around with deliberate, patient execution.

## Further reading and cross-links

- [Anatomy of a news reaction: the spike, the fade, and the trend](/blog/trading/event-trading/anatomy-of-a-news-reaction-spike-fade-trend) — the reaction-side companion: the first move is the knee-jerk into the vacuum; the second is the honest one.
- [Cross-asset transmission: how one print hits every market](/blog/trading/event-trading/cross-asset-transmission-how-one-print-hits-every-market) — how a single release gaps futures, FX, rates, and crypto in sequence.
- [The reaction function: why the same number moves differently](/blog/trading/event-trading/the-reaction-function-why-the-same-number-moves-differently) — what sets the *sign* of the gap, regime by regime.
- [Carry-trade unwinds: 1998, 2008, 2024 — when leverage breaks](/blog/trading/macro-trading/carry-trade-unwinds-1998-2008-2024-when-leverage-breaks) — the mechanism behind the August 5 2024 synchronized cascade.
- [Crypto as a macro liquidity asset](/blog/trading/macro-trading/crypto-as-a-macro-liquidity-asset) — why a thin 24/7 book makes crypto's gaps the most violent on the same news.
- [When correlations go to one in a crisis](/blog/trading/cross-asset/when-correlations-go-to-one-in-a-crisis) — why every market gaps together in a liquidity event, and diversification stops helping exactly when you need it.
- [The macro calendar: CPI, NFP, FOMC, PMI](/blog/trading/macro-trading/the-macro-calendar-cpi-nfp-fomc-pmi) — which scheduled prints create the vacuum, and when.
- [The volatility surface](/blog/trading/quantitative-finance/volatility-surface) — how the options market prices the expected move you size against.
