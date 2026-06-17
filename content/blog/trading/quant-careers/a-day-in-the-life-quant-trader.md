---
title: "A Day in the Life: Quant Trader"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "A concrete, hour-by-hour walkthrough of a quant trader's day at a market maker or prop firm — pre-open prep and risk review, managing quotes and flow at the open, intraday risk and a worked spike, the flatten and P&L attribution at the close, and the post-close review loop — grounded in the expected-value and risk discipline the job actually lives by."
tags: ["quant-careers", "quant-finance", "careers", "quant-trader", "market-making", "prop-trading", "risk-management", "pnl-attribution", "inventory", "expected-value", "day-in-the-life", "trading-desk"]
category: "trading"
subcategory: "Quant Careers"
author: "Hiep Tran"
featured: true
readTime: 41
---

> [!important]
> **TL;DR** — A quant trader's day is a fast, disciplined feedback loop: you quote, you get filled, your inventory and risk change, you re-price, and you do it again hundreds of times an hour — and the whole thing is governed by expected value and hard risk limits, not gut feeling.
>
> - The shape of the day is fixed: **pre-open prep and risk review → manage quotes and flow at the open → watch risk and size through the day → flatten or hedge at the close → attribute the P&L → review and tweak parameters**. Feedback arrives in *minutes*, which is the deepest difference from a researcher's months-long loop.
> - **The edge is the spread, not a market call.** A market maker earns a small, repeatable amount on each round trip and manages the inventory that piles up as a *cost*, not a bet. The skill is knowing when a one-sided flow is information and skewing your quotes to shed risk before the move runs you over.
> - **You live inside hard limits — position, daily loss, concentration, and the Greeks — and breaching one is not a judgment call, it is a forced action.** The job is to stay profitable inside that box, and the people who blow up are the ones who treated a limit as a suggestion.
> - **The one number to remember:** on a good day a junior's small book might print **about +\$4,200**, and a clean P&L attribution will show that almost all of it is durable spread capture (illustrative: +\$6,800 edge, +\$900 lucky drift, −\$2,300 adverse selection, −\$1,200 costs) — the lucky market-move term should be *small*, and if it is the whole P&L, you did not trade well, you got away with something.

It is 9:14 on a Tuesday morning, and **Maya** is sitting in front of six screens that are already alive with numbers, sixteen minutes before the US equity market opens.

She is five months into her first job as a quant trader at a listed-derivatives market maker — the kind of firm that does not bet on whether the market goes up or down, but stands ready to buy and sell all day and earns a sliver on each trade. On her main screen is a grid of the option contracts she is responsible for: each row a strike and expiry, each with a theoretical fair value her firm's model is spitting out, a bid she is willing to post, an ask she is willing to post, and a running position. To her left, a news terminal is scrolling overnight headlines. To her right, a risk dashboard shows her starting Greeks, her overnight P&L, and — in a box she checks before anything else — how much room she has left against each of her limits today. She is not predicting anything. She is getting ready to be the person everyone else trades against, profitably, for the next six and a half hours.

This is the moment before the open, and it is the calmest she will feel until the close. In a few minutes the grid will start blinking, fills will land, her position will move, and the loop will begin. Figure 1 is the rhythm of the day ahead — and notice what it is *not*: it is not six hours of staring and reacting on instinct. It is a structured loop of prep, then live management inside a box of risk limits, then a flatten, then a forensic accounting of where every dollar came from, then a review that quietly tunes tomorrow's parameters.

![A timeline of a quant trader's day showing six blocks from pre-open prep with news and risk review, through the open and managing flow, intraday risk and a spike, the close and flatten, P&L attribution into edge versus market move versus costs, and a post-close review that tweaks parameters](/imgs/blogs/a-day-in-the-life-quant-trader-1.png)

The thesis of this post is simple and, to most people who picture trading from movies, surprising: **the quant trader's job is not to be right about the market. It is to capture a small statistical edge thousands of times while ruthlessly managing the inventory and risk that capture creates.** A researcher's day, which we cover in [a day in the life of a quant researcher](/blog/trading/quant-careers/a-day-in-the-life-quant-researcher), is a slow loop of investigations with feedback in months. A trader's day is a fast loop of decisions with feedback in minutes — the market tells you, by the close, roughly whether your day worked. That tempo shapes everything: the temperament, the discipline, the way the desk is organized, and what your first real responsibility — carrying a small book of your own — actually feels like. If you want to be a quant trader, the question is not "can you call the market?" It is "can you make hundreds of small expected-value-positive decisions under pressure, manage the risk they generate, and stay inside your limits on the worst day of the year?"

## Foundations: what a quant trader actually does all day

Before any of this makes sense, you need a small, precise vocabulary. None of it is hard, but every word is load-bearing, and most outsiders' picture of the job is wrong precisely because they are missing one or two of these definitions. Assume you are brilliant but have never worked in markets — we will build each idea from zero.

**What a quant trader is.** A quant trader (often "QT") is a person who makes and manages live trading decisions, in real time, on the firm's own capital, using models and systems built by a team. At a market maker — Jane Street, Optiver, SIG, IMC, Citadel Securities, Jump, HRT and the wider ecosystem we map in [the firm archetypes](/blog/trading/quant-careers/the-firm-archetypes-prop-vs-hft-vs-pod-shop-vs-systematic-fund) — the trader's primary job is to *quote*: to stand in the market continuously offering to both buy and sell, and to make money on the difference while controlling the risk that builds up. The role lives next to three others, which we cover in [the four paths](/blog/trading/quant-careers/the-four-paths-trader-researcher-developer-engineer): the *researcher*, who builds the predictive models; the *developer*, who builds the fast, reliable software the trading runs on; and the *risk manager*, who sets and polices the limits. The trader's deliverable is a managed book that is profitable and inside its limits at the end of the day.

**Market making and the spread.** A market maker quotes a *bid* (the price at which they will buy) and an *ask*, or offer (the price at which they will sell). The gap between them is the *spread*. If you are willing to buy at \$99.98 and sell at \$100.02, your spread is 4 cents. When one counterparty sells to you at your bid and another buys from you at your ask, you have done a *round trip* and pocketed the spread — 4 cents per share, before costs. You did not predict anything. You provided liquidity to two people who wanted to trade *now*, and you charged them a tiny fee for the service. That fee, captured across enormous volume, is the business. The whole mechanic is the subject of [how quant firms actually make money](/blog/trading/quant-careers/how-quant-firms-actually-make-money), and the EV mindset behind it is exactly what the [market-making games in interviews](/blog/trading/quantitative-finance/market-making-games-quant-interviews) are testing.

**Fair value (the theoretical price, or "theo").** To quote sensibly you need an estimate of what the thing is *actually worth* right now — the *fair value*, or in desk slang the "theo." For a simple stock it might be the midpoint of the current market plus a model adjustment; for an option it comes from a pricing model fed with the underlying price, volatility, time to expiry, and rates. The researchers and developers build the machinery that computes theo continuously; the trader posts a bid a little below theo and an ask a little above it, so that *on average* every fill is at a price favorable to the firm by half the spread.

**Inventory (your position).** Every time you get filled, your *inventory* — your net position — changes. Buy 500 lots and you are *long* 500; sell 800 and you are *short* 300. Inventory is the central object of the job, and the key idea that beginners miss is this: **inventory is a risk you are carrying, not a bet you wanted to make.** As a market maker you did not choose to be long; you got filled because someone sold to you. The moment you hold inventory, you are exposed to the price moving against it. A long position loses if the price falls; a short loses if it rises. So a market maker is constantly trying to *recycle* inventory back toward flat — to find the other side and earn the spread again — while the position sits on the book as a live risk.

**P&L (profit and loss).** Your P&L is the running tally of how much money your book has made or lost. It has two components that you must keep separate in your head: *realized* P&L (locked in from round trips you have completed) and *unrealized*, or mark-to-market, P&L (the paper gain or loss on the inventory you currently hold, valued at the current market price). A market maker can show a green realized P&L from spread capture and a red unrealized P&L because the inventory they are holding has moved against them. The whole skill is keeping the first bigger than the second.

**Expected value (EV).** EV is the single most important idea in the whole job — it is the spine of this entire series. The expected value of a decision is the probability-weighted average of its outcomes. A market maker quotes because each fill is *EV-positive*: on average it earns half the spread, even though any individual fill might be followed by an adverse move. You do not need to win on every trade; you need each decision to be positive in expectation and to repeat it enough times that the law of large numbers turns a tiny edge into real money. The sizing discipline that goes with EV — how big to bet given an edge and the risk of ruin — is the [Kelly criterion](/blog/trading/quantitative-finance/kelly-criterion-sequential-betting-quant-interviews), and it lives at the core of every position decision a trader makes.

**Risk limits.** A trader does not have unlimited freedom. The firm sets *hard limits*: the maximum position you may hold in any one name, the maximum amount of money you may lose in a day before you must stop, the maximum share of your risk concentrated in one place, and — for options — caps on the *Greeks*, which measure how sensitive your book is to moves in price, volatility, and time. We will define each precisely later; for now hold the idea that the trader's freedom is the space *inside* all of these limits at once, and that breaching one is not a debate — it is a forced action. The discipline this demands is the whole subject of [risk discipline and not blowing up](/blog/trading/quant-careers/risk-discipline-and-not-blowing-up).

**P&L attribution.** At the end of the day you do not just look at the total P&L; you *decompose* it — you attribute each dollar to a cause. How much came from capturing the spread (your edge)? How much from the market happening to drift in your favor while you held inventory (luck)? How much did costs and adverse selection eat? Attribution is how a trader learns whether a green day was *earned* or *got away with*, and it is the feedback signal that tunes everything else.

**The desk.** A trader does not work alone. A small *desk* surrounds them: researchers who supply the model and signals, developers who build and own the quoting system, and risk managers who set the limits. We will spend a section on these relationships, because "what is the first real responsibility?" is answered partly by "who do you depend on, and who depends on you?"

With those words in place, the day makes sense. Hold three ideas as you read: the job is a **loop** (quote, fill, manage, re-quote), the edge is the **spread** (small, repeatable, EV-positive), and the constraint is the **risk box** (you operate inside hard limits). Everything else is detail.

## The loop the whole job runs on

Before we walk the clock, look at the loop the trader runs continuously, because every section below is just this loop at a different time of day. Figure 2 shows it: you estimate fair value, post a two-sided quote around it, get filled when someone trades, update your inventory, check that inventory against your risk limits, then re-price and re-quote — usually skewing your prices to lean against the inventory you are now carrying. Then it repeats. Hundreds of times an hour. The bottom edge of the figure is the most important one: the *loop* edge that feeds your new quote back to the top, because **each fill changes the next price you show.**

![A pipeline diagram of the live quote-and-risk loop showing estimate fair value, post a quote, get filled, update inventory, check risk limits, and re-price and skew, with a loop edge feeding the new quote back to fair value](/imgs/blogs/a-day-in-the-life-quant-trader-2.png)

The reason this is a loop and not a line is the inventory feedback. If you simply posted the same symmetric quote forever, a one-sided flow — everyone wanting to buy from you — would leave you with a growing short position and no reaction, until a move wiped out a year of spread capture in an hour. The loop closes that gap: every fill updates your inventory, and your inventory *changes how you quote*. Get short, and you raise your prices (you skew up) so you are more eager to buy and less eager to sell, pulling yourself back toward flat. This single feedback mechanism — quote, fill, *adjust*, re-quote — is the difference between a market maker and a person handing out free options. Keep the loop in your mind as we now walk the day from 7 a.m.

## Pre-open: prep, news, risk review, and system checks

The market opens at 9:30, but Maya's day starts around 7. The pre-open block is unglamorous and absolutely load-bearing — most of the disasters that happen *during* the day were set up by a prep step that got skipped. There are four jobs.

**The overnight risk review.** The first thing Maya looks at is not the news; it is her own book. Markets she trades may have moved overnight — futures trade nearly around the clock, and overseas markets have had a full session. Her starting position is whatever she carried home (ideally close to flat, but options books often carry residual risk that cannot be perfectly hedged), and that position has a fresh unrealized P&L this morning that did not exist at last night's close. She checks: where did I start, how have my Greeks shifted overnight, and is anything already close to a limit before I have even quoted? The single worst way to start a day is to discover at 9:45 that you were already near your loss limit because of an overnight move you never looked at.

**The news scan.** Next she reads. Not to predict — to *avoid being the last to know*. The job here is to find anything that changes fair value or makes the market dangerous: an earnings release in a name she quotes, a scheduled economic number (a CPI print, a Fed decision), a corporate action like a dividend or a split, a merger headline. A market maker who keeps quoting a tight spread into a known event is offering free money to anyone who knows the event is coming. The discipline is to *widen or pull* quotes around scheduled risk and to know which of today's events touch her book. This overlaps with [the markets knowledge every quant needs](/blog/trading/quant-careers/the-markets-knowledge-every-quant-needs) — you cannot manage risk around an event you do not understand.

**System checks.** A market maker is a software system as much as a person, and the trader is the human responsible for it being healthy. Maya checks that the pricing engine is computing theos, that her market-data feeds are live and not stale, that her connection to the exchange is up, that her automated quoting is configured for today's parameters, and that the *kill switch* — the button that pulls every quote instantly — works. This is where the developer relationship is most concrete: the system is built by the [quant developers](/blog/trading/quant-careers/a-day-in-the-life-quant-developer-and-low-latency-engineer), but at the open it is the trader's responsibility, and "the feed was stale and I didn't notice" is a sentence that ends careers.

**Setting the day's parameters.** Finally, Maya sets her plan: how wide to quote in each name given today's expected volatility, which contracts to be more cautious in, where her soft internal limits sit relative to the firm's hard limits, and how aggressively to lean on inventory. Some of this is automated; the trader's judgment is in the overrides. By 9:25 she has a flat-ish book, a clean news picture, a healthy system, and a plan. The screens are lit. She waits for the bell.

#### Worked example: the EV of one quote, repeated all day

Strip the job to a single decision and the whole business becomes visible. Suppose Maya quotes a contract with a fair value (theo) of exactly **\$100.00**, posting a bid of **\$99.98** and an ask of **\$100.02** — a 4-cent spread, 2 cents on each side. A customer who wants to sell hits her bid; she buys at \$99.98, which is \$0.02 *below* theo. A different customer who wants to buy lifts her ask; she sells at \$100.02, \$0.02 *above* theo. Across the round trip she earned the full **\$0.04** spread, or **\$0.02 of edge per fill** relative to fair value.

Now the EV. On most fills the counterparty is trading for reasons unrelated to short-term price — they need liquidity — so her \$0.02 edge is clean. But some fraction of the time she is trading against someone who *knows something*, and the price moves against her right after: this is *adverse selection*, and it is the cost of being a market maker. Say 80% of her fills are clean (she keeps the \$0.02) and 20% are adversely selected, costing her \$0.06 on average when they happen. The EV per fill is:

```python
ev_clean   = 0.80 * 0.02     # = +0.0160 : clean fills keep the half-spread
ev_adverse = 0.20 * (-0.06)  # = -0.0120 : informed flow moves against her
ev_per_fill = ev_clean + ev_adverse   # = +0.0040 per share, per fill
```

Her edge is **+\$0.004 per share per fill** — less than half a cent. That sounds like nothing. But she trades, say, 2,000 fills a day at an average 500 shares each: `0.004 * 500 * 2000 = 4000`, or about **\$4,000** of expected daily edge from spread capture alone. *The whole job is a sliver of edge per fill, made real by relentless volume and ruthless control of the adverse-selection tail.*

## The open and managing flow

At 9:30 the loop comes alive. Quotes start trading, fills land in a fast trickle then a stream, and Maya's position begins to move. The open is often the most volatile, highest-volume part of the day — overnight news gets repriced, orders that queued up overnight all execute at once — so it is both the most profitable and the most dangerous window. The trader's job here is not to predict where the open settles; it is to *manage the flow*: to keep quoting through the chaos while controlling the inventory it dumps on her.

The mechanics of "managing flow" come down to reading whether the trading hitting her quotes is *balanced* or *one-sided*. Balanced flow — sellers hitting her bid and buyers lifting her ask in roughly equal measure — is the dream: she recycles inventory, stays near flat, and banks spread. One-sided flow — everyone wanting to do the same thing, all selling to her or all buying from her — is the danger, because it builds a position and is often *information*. If wave after wave of sell orders is hitting her bid, the most likely explanation is not "I am getting lucky on spread"; it is "the market is about to go lower and I am the one accumulating the falling thing."

The professional response is to **skew and widen**. *Skewing* means shifting both your bid and ask in the direction that helps you shed inventory: if you are getting too long, you lower both prices so you are less eager to buy and more eager to sell. *Widening* means increasing the spread so you charge more for the liquidity you are providing into a dangerous flow. Together they slow the accumulation, make each additional unit of inventory more expensive for the counterparty, and bias your fills toward the trades that flatten you. This is the single most important skill of the open, and it is exactly what the [trading games in interviews are testing](/blog/trading/quantitative-finance/market-making-games-quant-interviews) — they hit you one-sided and watch whether you keep quoting symmetrically (the rookie error) or adjust.

Figure 4 puts the rookie and the seasoned response side by side. The rookie holds a symmetric quote, lets the one-sided flow pile up a position, treats each fill as captured edge, and gets run over when the move comes. The seasoned trader reads the one-sided flow as information, skews and widens to slow the bleed, then bids up aggressively to buy back and flatten — paying a little to get out, but surviving the move with the spread still earned.

![A before-after diagram contrasting a rookie response that keeps a symmetric quote and accumulates a dangerous position against a seasoned response that reads the flow, skews and widens the quote, and aggressively flattens the inventory](/imgs/blogs/a-day-in-the-life-quant-trader-4.png)

#### Worked example: managing inventory after a one-sided flow

Make the skew decision concrete. Maya is quoting that \$100 contract, and a one-sided wave of sellers hits her bid repeatedly. Over twenty minutes she buys **800 lots** at an average price of **\$99.95** — she is now long 800, and the market has drifted to a theo of **\$99.85**. Her inventory is now marked \$0.10 below where she bought it, and across 800 lots at \$100 of notional per lot per point of move, the position is showing a mark-to-market loss of **\$8,000**. She faces a choice.

**Option A — do nothing (the rookie).** Keep the symmetric quote. If the flow is informed and the price keeps falling another \$0.10 to \$99.75, she loses another \$8,000, and her 800-lot long becomes a \$16,000 hole. Her expected outcome is bad because the flow is *telling her* the price is more likely to fall than rise.

**Option B — skew and widen (the pro).** She immediately lowers her quotes — bidding only \$99.78 and offering \$99.86 — so she is barely buying and eagerly selling. She also lifts her own offer aggressively to *buy back* on any bounce. Suppose this lets her flatten the 800 lots over the next fifteen minutes at an average of \$99.82. Her realized loss on the round trip is `800 * (99.82 - 99.95) = ` a loss of about **\$10,400** on the inventory — *but* she has also been earning spread on every one of those flattening trades, and she has stopped the bleed. Compare the EV: Option A's expected loss, given the flow signal, is larger and has a fat tail; Option B caps the damage and keeps her inside her position limit.

The key reframe: **the \$10,400 is not a trading loss to be ashamed of — it is the cost of inventory insurance, paid to get back to flat before the move finished.** A market maker's losses are dominated not by bad spread capture but by inventory held too long into an adverse move. *The skill is not avoiding losses; it is making the loss small and bounded instead of large and open-ended.*

## Intraday: monitoring risk, sizing, and handling a spike

After the open settles, the day becomes a long middle stretch — 10 a.m. to 3 p.m. — that an outsider might find boring and a trader knows is where discipline is won or lost. The flow is calmer, the spread capture grinds along, and the job becomes *vigilance*: watching the book, watching the limits, and being ready for the moment when calm turns into a spike.

Figure 3 is an illustrative path through such a day on Maya's small book. The blue line is her running P&L; the amber line is her net position — her inventory, which *is* her risk. For most of the morning the position churns near flat while spread capture grinds the P&L steadily upward. Then, near midday, a one-sided flow event drives her position to roughly 800 lots short and dents the P&L as the mark moves against her. She skews, widens, and buys the position back; the P&L recovers as she earns spread on the flattening trades and the move stalls; and she closes flat, up about +\$4,200 on the day. The shape is the whole lesson: **steady edge, a risk event, a disciplined response, a recovery.**

![An intraday chart with two lines showing running P&L climbing steadily from spread capture, a dent during a one-sided flow event that drives the position short, and a recovery after the trader skews and flattens, closing flat at about plus four thousand two hundred dollars](/imgs/blogs/a-day-in-the-life-quant-trader-3.png)

The intraday discipline is built on the *risk limits*, which deserve their own figure because they are the box the trader lives inside. Figure 5 lays out four kinds. A **position limit** caps the maximum net long or short you may hold in any one name — it bounds how far a single move can hurt one position. A **loss limit** (a stop) caps the P&L you may lose in a day before you must flatten and stop trading — it prevents a bad day from becoming a blown-up book. A **concentration limit** caps how much of your risk can sit in one name or sector — it stops a single correlated bet from quietly *becoming* your whole book. And for options, **Greeks limits** cap your delta, gamma, vega, and theta — they bound how sensitive your book is to moves in price, volatility, and time.

![A matrix of four risk limits showing position limit, loss limit, concentration limit, and Greeks limit, with what each one caps and what failure mode it controls](/imgs/blogs/a-day-in-the-life-quant-trader-5.png)

A word on the Greeks, since they are where options trading differs most from simple stock market making. *Delta* is how much your book's value changes for a \$1 move in the underlying — your effective directional exposure. *Gamma* is how fast your delta itself changes as the underlying moves — high gamma means your directional exposure can flip on you quickly during a spike, which is exactly when it is most dangerous. *Vega* is your exposure to changes in implied volatility — if the market gets scared and option prices inflate, a short-vega book bleeds. *Theta* is the daily decay of option value as time passes — often a friend to a market maker who is net short options, a steady cost to one who is long. A trader's intraday job includes *hedging the Greeks*: if a flurry of fills has left her long delta, she sells the underlying to neutralize it, so that she is exposed to the *spread* she is paid for and not to the *direction* she has no edge in.

#### Worked example: an intraday risk-limit breach and the right response

It is 1:40 p.m. and a sudden headline spikes the underlying. In ninety seconds, fills cascade and Maya's position blows through a limit. Specifically: her **loss limit** for the day is **−\$25,000**, and the spike has driven her unrealized P&L to **−\$26,500** — she is \$1,500 *past* her hard stop. What is the right response?

The wrong responses, in order of how much they will hurt her career:

- **"It will come back."** Holding through a breach hoping for a reversal is not trading; it is gambling with the firm's capital, and it is the exact behavior every risk system exists to stop. Do not do this.
- **"I'll quietly trade my way back."** Continuing to take risk to recoup the loss is *adding* risk after a breach — the textbook way a \$26,500 loss becomes a \$120,000 loss.

The right response is mechanical and immediate: **stop adding risk and reduce the position toward flat, now.** A loss limit is not a target to manage around; it is a tripwire that converts a judgment call into a forced action. Maya pulls her aggressive quotes, hedges her delta to neutralize directional exposure, and works the inventory down. She does not try to make the \$26,500 back today. She also does the thing that distinguishes a professional: she *tells her risk manager and her lead immediately*, before they ask. A breach handled openly and mechanically is a Tuesday; a breach hidden and traded against is a firing.

The deeper point is about EV and survival, the [Kelly](/blog/trading/quantitative-finance/kelly-criterion-sequential-betting-quant-interviews) intuition made visceral: a strategy with positive EV still goes to zero if you ever bet enough to be wiped out. The limit exists because the cost of one ruinous day exceeds the benefit of many good ones. *A trader's first job is not to make money — it is to still be trading tomorrow, which means treating limits as walls, not suggestions.* This is the whole heart of [risk discipline and not blowing up](/blog/trading/quant-careers/risk-discipline-and-not-blowing-up).

## The close: flatten, hedge, and lock in the book

As 4 p.m. approaches, the day's last act begins, and its logic is different from everything before. During the day, inventory is something you recycle; at the close, inventory is something you must *decide whether to carry overnight*. Holding a position after the close exposes you to overnight risk — gaps from news while the market is shut, overseas moves, the next morning's open — with no ability to manage it in real time. So the default discipline for a market maker is to go home as flat as possible.

There are two ways to get there. *Flattening* means trading out of your inventory entirely — buying back your shorts, selling your longs — so you end the day with no position and no overnight risk. *Hedging* means, when you cannot perfectly flatten (options books often have residual exposure that cannot be unwound without giving up too much edge), neutralizing the dangerous parts: hedging your delta to zero so you have no directional bet overnight, even if you still hold some inventory. The trader's close-time judgment is which residual risks are acceptable to carry (small, well-understood, theta-positive) and which must be hedged away (directional, large, event-exposed).

The close is also when the day's P&L stops being a moving number and becomes a *fact*. The realized P&L is locked; the unrealized P&L on whatever she carries is marked at the closing price. Maya's screen now shows a single number for the day — say, about +\$4,200 — and that number is where the second half of the job begins. Because the number alone tells you almost nothing. A trader who looks only at the bottom line learns nothing and improves at nothing. The learning is in the *attribution*.

## Post-close: P&L attribution and the review loop

The market is closed, the screens are calmer, and Maya does the most undervalued work of the day: she takes apart her P&L. **P&L attribution** is the discipline of splitting the day's total into its causes, and it is the feedback signal that makes a trader better instead of just luckier.

Figure 7 is the attribution of Maya's +\$4,200 day as a waterfall. The reported number decomposes into: **+\$6,800 of spread capture** (the durable edge — the half-spread earned across all her clean fills), **+\$900 of market move** (the lucky drift — the underlying happened to nudge in favor of inventory she was holding), **−\$2,300 of adverse selection** (getting run over during the one-sided midday flow, the cost of being a liquidity provider), and **−\$1,200 of costs** (exchange and clearing fees, financing). Sum them and you get the +\$4,200.

![A waterfall chart attributing a day's plus four thousand two hundred dollar P&L into plus six thousand eight hundred of spread capture, plus nine hundred of market move, minus two thousand three hundred of adverse selection, and minus one thousand two hundred of costs](/imgs/blogs/a-day-in-the-life-quant-trader-7.png)

Why does this decomposition matter so much? Because **the durable, repeatable part of the P&L is the spread capture, and the part you should distrust is the market-move term.** If Maya's +\$4,200 had come almost entirely from a lucky market move — say +\$5,000 of market move and *negative* spread capture — she would have had a green day while trading *badly*: she got away with it, and the same behavior will lose money the day the market goes the other way. Conversely, a small *red* day that was +\$6,000 of spread capture undone by a one-off −\$8,000 adverse event might reflect genuinely good trading and a single piece of bad luck. The bottom line lies; the attribution tells the truth. This is the exact analog of the researcher's [in-sample versus out-of-sample discipline](/blog/trading/quantitative-finance/overfitting-purged-cv-deflated-sharpe-quant-research) — separating the repeatable signal from the noise that happened to help today.

The review loop closes the day. With the attribution in hand, Maya asks: was the midday adverse selection avoidable, or just the cost of providing liquidity into real information? Should her quotes have widened sooner around the headline? Were her spreads too tight in the names that bled and too wide in the names that traded clean? These answers become small *parameter tweaks* for tomorrow — a slightly wider default spread in the volatile name, a faster skew response, a tighter soft limit before a known event. This is the trader's version of the feedback loop, and its speed is the job's signature: a researcher waits months to learn if an idea worked; a trader tunes tomorrow's quotes tonight.

#### Worked example: attributing a day's P&L into edge versus market move

Make the attribution arithmetic explicit, because it is the most important calculation a trader does after the close. Maya's systems give her three measurable inputs for the day:

```python
spread_capture  =  6800   # sum over fills of (fill price - theo) on the right side
market_move     =   900   # mark-to-market change on held inventory from price drift
adverse_costs   = -2300   # realized loss from inventory held into adverse flow
fees_financing  = -1200   # exchange + clearing + financing costs

reported_pnl = spread_capture + market_move + adverse_costs + fees_financing
    #  = 6800 + 900 - 2300 - 1200 = 4200
edge_share   = spread_capture / reported_pnl   # = 1.62  -> edge is 162% of net P&L
luck_share   = market_move    / reported_pnl   # = 0.21  -> luck is only 21% of net
```

The reported P&L is **+\$4,200**, but the interpretation lives in the shares. Spread capture of \$6,800 is **162% of the net** — the entire net result, and then some, came from the durable edge, with adverse selection and costs eating it back down. The market-move (luck) term is only **\$900, about 21% of the net** and small relative to the edge. *This is what a good day looks like under the hood: the edge does the heavy lifting, luck is a minor character, and a green bottom line is earned rather than borrowed from a coin flip that happened to land right.*

## The desk: who the trader works with all day

A quant trader is the most *visible* role on the desk — the one with the live book and the P&L number — but they are the tip of a small, tightly coupled team, and understanding those relationships is most of understanding the first real responsibility. Figure 6 shows the desk: the trader (blue, in the center) sits between three others and runs the live book.

![A graph of the trading desk showing a quant researcher, a quant developer, and a risk manager all feeding into the trader at the center, who manages the live book of positions and P&L on the firm's capital](/imgs/blogs/a-day-in-the-life-quant-trader-6.png)

**The researcher (QR) supplies the model.** The fair-value engine, the volatility model, the signals that nudge the quotes — these come from the [quant researchers](/blog/trading/quant-careers/a-day-in-the-life-quant-researcher). The trader is the researcher's most important customer and critic: when a model misprices a contract and the trader gets adversely selected, that is feedback the researcher needs. The healthiest desks have a tight loop here — the trader sees the model fail in production, the researcher fixes it, the next day's quotes are sharper. A trader who cannot articulate *why* a model is wrong is just a button-presser; a trader who can is a research partner.

**The developer (QD) builds the system.** The quoting engine, the risk dashboard, the kill switch, the connection to the exchange — built and owned by the [quant developers and low-latency engineers](/blog/trading/quant-careers/a-day-in-the-life-quant-developer-and-low-latency-engineer). At a high-frequency firm the *speed* of that system is itself the edge; a faster quote gets to the top of the queue and gets the clean fills, and the slow trader is left with the adversely-selected scraps. The trader tells the developer what tools the desk needs and feels every millisecond of latency in their fill quality.

**The risk manager sets the box.** The limits in Figure 5 are not the trader's to set; they come from a risk function whose job is to ensure no single trader or bad day can threaten the firm. The relationship is sometimes adversarial in the healthy way that a good editor is adversarial to a writer: risk pushes back, asks why a position is so large, makes the trader justify the inventory. A trader who treats risk as the enemy is a liability; a trader who treats risk as the institutional memory of every blow-up that ever happened is using them correctly.

**The execution layer.** When a trader needs to move size — to flatten a large inventory, to hedge a big delta — *how* the order is worked matters, because a clumsy execution moves the market against you and gives back your edge. The algorithms that slice a large order into the market over time — VWAP, TWAP, POV — are the subject of [execution algorithms in quant research](/blog/trading/quantitative-finance/execution-algorithms-vwap-twap-pov-quant-research), and a trader who understands them flattens cheaply while one who does not pays a hidden tax on every exit.

The first real responsibility, then, is **carrying a small book** — being handed a slice of the firm's risk and being the human accountable for it. Not the whole desk's book; a small, contained set of contracts where a junior can learn the loop, make survivable mistakes, and prove they can stay inside the limits. Everything in this post is what it feels like to carry that book for the first time.

#### Worked example: Maya carries her first small book

Six weeks in, Maya's lead hands her a small book: a handful of liquid option contracts, a **position limit of 1,000 lots** per name, a **daily loss limit of \$25,000**, and a mandate to "quote tight, stay flat, don't blow your limits." It is the most consequential moment of her [first 90 days](/blog/trading/quant-careers/your-first-90-days-ramping-without-blowing-up). What does the first month look like in numbers?

The firm does not expect her to be a star; it expects her to be *safe and break-even-ish* while she learns. Suppose her edge per fill is the +\$0.004 per share from the earlier example, she trades a modest 1,200 fills a day at 400 shares each, and she trades 20 days in the month:

```python
edge_per_fill   = 0.004           # USD per share, net of adverse selection
shares_per_fill = 400
fills_per_day   = 1200
trading_days    = 20

gross_edge_day  = edge_per_fill * shares_per_fill * fills_per_day   # = 1920 / day
gross_edge_month = gross_edge_day * trading_days                    # = 38,400

    #  But a rookie makes rookie mistakes: a couple of mismanaged one-sided flows,
    #  a slow skew, an over-tight spread into an event. Say that costs her, on net,
    #  about 60% of the gross edge while she is learning the loop.
learning_drag = 0.60
net_pnl_month = gross_edge_month * (1 - learning_drag)              # = 15,360
```

Her gross edge is about **\$38,400** for the month, but a realistic learning drag of ~60% leaves her around **+\$15,000** net — a small, real, *positive* number that says she captured edge and survived. Crucially, what her lead is watching is **not the \$15,000.** It is whether she breached a limit (no), whether she handled the one-sided flows without panicking (mostly), and whether her [P&L attribution](/blog/trading/quantitative-finance/evaluating-alpha-signals-ic-sharpe-turnover-quant-research) shows edge rather than luck (yes). *A junior's first book is graded on discipline and process, not on the size of the P&L — because a trader who stays inside the box and captures real edge will compound for years, while one who makes money by gambling will eventually give it all back and more.*

## How a junior trader grows into the seat

The arc from "carrying a small book" to "running a real book" is the whole junior-to-senior path compressed onto a trading desk, and it has a recognizable shape.

**Months 0–3: learn the loop and the limits.** The first quarter is about internalizing the quote-fill-manage loop until it is reflexive, learning to read one-sided flow, and — above all — never breaching a limit. The firm measures process: are you flat at the close, do you skew when you should, do you escalate problems openly? Most of the curve here is "stop making the rookie mistakes." The full ramp playbook is [your first 90 days](/blog/trading/quant-careers/your-first-90-days-ramping-without-blowing-up).

**Months 3–12: own a small book and tighten the feedback loop.** Now you carry real (small) risk and the daily attribution becomes your teacher. You start to see which of your habits make money and which leak it, and your spreads, skews, and event-handling get sharper. Comp at this stage is mostly base — for a top-tier first-year QT, a base around **\$250k–\$300k** with a sign-on, building toward a first-year total that can reach **\$450k–\$650k** on target (reported on levels.fyi and Glassdoor, 2025; *illustrative and survivorship-biased* — a strong seat in a strong year, not the median).

**Years 1–3: bigger book, real P&L responsibility.** As you prove you can stay inside the box and capture edge, you are given more risk and your bonus — which is where the real money is — starts to track your P&L contribution. A strong mid-level QT 2–3 years in can reach into seven figures in a good seat and a good year (a reported example is base \$200k + bonus \$1.3M ≈ **\$1.5M** at ~2.5 years; this is a *strong* outcome, not typical). The number that matters: **the bonus does not repeat automatically.** A \$1.3M year can be a \$300k year if the seat or the P&L changes. Say it to yourself every time you read a headline comp number.

**Years 3+: run a book, then a desk.** The senior trader is no longer just managing a book; they are deciding *strategy* — which markets to make, how to allocate the desk's risk, how to respond to a changing market structure — and often mentoring juniors and partnering with research on the next generation of models. The seat compounds: a survivor at the five-year mark can be in the **\$800k–\$1.2M** range on standard performance, far higher in a strong seat (these are reported ranges, 2025–2026; heavily conditional on survival and the firm). The honest framing, which this whole series insists on: these numbers describe the people who *survived the filter*. Many wash out, switch firms, or land below the headline — the comp curve is a survivorship curve, and the senior quant has internalized exactly that EV-under-uncertainty thinking about their own career that they apply to markets.

## Common misconceptions

The quant trader's job is one of the most misunderstood in finance, romanticized in some directions and dismissed in others. Here are the four myths worth killing.

**Myth 1: "It's gut-feel gambling — you're betting on the market."** This is the deepest misunderstanding, and the whole post has been an argument against it. A market maker does not bet on direction; they earn a spread and manage inventory as a *cost*. Every decision is sized by expected value, bounded by hard limits, and reviewed by attribution. The discipline is closer to running an insurance business — pricing risk, collecting many small premiums, capping the tail — than to gambling. The traders who *do* gamble, who treat their book as a casino, are exactly the ones the risk limits exist to stop, and they do not last.

**Myth 2: "You predict the market."** A market maker is largely *direction-neutral* by design — they hedge their delta precisely so they are *not* exposed to which way the market goes. Their edge is in the spread and in being faster and sharper than the next quote, not in a crystal ball. The skill that looks like prediction — skewing against one-sided flow — is not "I think the market goes down"; it is "the flow is informed, so my inventory is more likely to hurt me than help me, so I should shed it." That is risk management dressed as a forecast, not a forecast.

**Myth 3: "It's all automated, so the traders just sit there."** The quoting *is* automated — no human can post and adjust thousands of quotes a second. But automation does not remove the trader; it *raises* the trader's job to the level of managing the system, setting its parameters, handling the events the automation is not trusted with, catching the model errors, and being the human accountable for the risk. The automation is a power tool; the trader is the one who decides where to point it and who hits the kill switch when it misbehaves. A flash event, a stale feed, a model that has not seen a regime like today — these are exactly the moments the "they just sit there" picture gets wrong.

**Myth 4: "The screen time is the whole job."** The visible six and a half hours of quoting are maybe half the job. The other half is the pre-open prep that prevents disasters, the post-close attribution that turns a number into learning, and the review loop that tunes tomorrow's parameters. The traders who improve fastest are the ones who treat the screen time as the *experiment* and the prep-and-review as the *science* — the part where you actually figure out what worked and why.

## How it plays out in the real world

Strip away the mystique and look at the real desks. At a market maker like **Optiver** (Amsterdam, founded 1986, ~2,100 employees) or **SIG** (Bala Cynwyd, founded 1987, 3,000+ employees), a new quant trader joins through a structured education program — SIG famously uses poker to teach EV and decision-making under uncertainty — and spends months on a simulator and a tiny book before touching real size. The day looks exactly like Maya's: a pre-open routine, hours of automated quoting that the trader supervises and overrides, hard limits enforced by a risk function, and a close-time flatten. The interview that gets you there is a [mental-math gauntlet and a trading game](/blog/trading/quantitative-finance/market-making-games-quant-interviews) precisely because those test the two things the day demands — speed-with-calibration and EV-under-pressure.

At **Jane Street** (NYC, founded 2000), the trader's day is heavy on collaborative problem-solving and the firm's famous internal tooling, with the same EV-and-risk spine. At a high-frequency firm like **Jump** (Chicago, founded 1999) or **HRT** (NYC, founded 2002), the trader's edge is fused with the speed of the system the [developers](/blog/trading/quant-careers/a-day-in-the-life-quant-developer-and-low-latency-engineer) build — the loop is the same, but the clean-fill-versus-adverse-selection battle is fought in microseconds and the "managing flow" judgment is increasingly encoded in the automation the trader configures. At **Citadel Securities**, the market-making arm (distinct from the Citadel hedge fund), the scale is enormous and the desk specialization deep, but the atom is unchanged: quote, get filled, manage inventory, stay inside the limits, attribute the P&L.

The honest part. The comp can be extraordinary — a strong mid-level QT in a strong seat reaching seven figures, top outliers far beyond — but it is **bonus-driven, P&L-linked, and does not repeat automatically** (reported ranges on levels.fyi and the 2026 quant-pay surveys, 2025–2026; illustrative and survivorship-biased). The job is also genuinely high-pressure: your performance is measured in a daily number, the up-or-out reality is real, and a single mishandled limit breach can end a seat. The people who thrive are not the cowboys; they are the disciplined ones who find the daily loop satisfying, who *enjoy* the attribution and the tuning, and who treat their own career the way they treat their book — as a positive-EV process to be run carefully, sized sensibly, and protected from the ruinous tail.

## When this matters / Further reading

This post matters most at two moments. The first is **before you interview**: if you can articulate the quote-fill-manage loop, why inventory is a cost and not a bet, and why a trader skews against one-sided flow, you will sound like someone who understands the job rather than someone who watched a movie about it — and that is exactly the signal a [trading game](/blog/trading/quantitative-finance/market-making-games-quant-interviews) is designed to extract. The second is **in your first 90 days on a desk**, when the difference between a successful ramp and a washout is almost entirely whether you internalize the risk discipline before you internalize the thrill.

The single idea to carry out of here: **the quant trader's edge is a sliver per fill, made real by volume and protected by limits — and the job is to capture that edge thousands of times while never letting the inventory or the loss run past the box.** Everything else — the screens, the speed, the comp — is downstream of that one discipline.

To go deeper:

- **The sibling days in the life.** See how the other seats differ in tempo and texture: [a day in the life of a quant researcher](/blog/trading/quant-careers/a-day-in-the-life-quant-researcher) (the slow, skeptical loop) and [a day in the life of a quant developer and low-latency engineer](/blog/trading/quant-careers/a-day-in-the-life-quant-developer-and-low-latency-engineer) (the system the trader's loop runs on).
- **Surviving the first months.** [Your first 90 days: ramping without blowing up](/blog/trading/quant-careers/your-first-90-days-ramping-without-blowing-up) is the practical ramp guide for the exact moment you are handed your first small book.
- **The discipline the job lives by.** [Risk discipline and not blowing up](/blog/trading/quant-careers/risk-discipline-and-not-blowing-up) is the full treatment of the limits-as-walls mindset that runs through this whole post.
- **The technical underpinnings.** The EV-and-quoting skill is tested by [market-making games in quant interviews](/blog/trading/quantitative-finance/market-making-games-quant-interviews); the sizing intuition is the [Kelly criterion](/blog/trading/quantitative-finance/kelly-criterion-sequential-betting-quant-interviews); and the cost of getting out of a position cleanly is the subject of [execution algorithms: VWAP, TWAP, POV](/blog/trading/quantitative-finance/execution-algorithms-vwap-twap-pov-quant-research).
