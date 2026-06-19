---
title: "Gap Risk and Overnight Exposure: The Move That Skips Your Stop"
date: "2026-06-19"
publishDate: "2026-06-19"
description: "Why a stop-loss only protects you if the price actually trades there, and how a gap jumps straight over your exit to fill you far lower."
tags: ["risk-management", "gap-risk", "overnight-risk", "stop-loss", "limit-down", "leverage", "tail-risk", "position-sizing"]
category: "trading"
subcategory: "Risk Management"
author: "Hiep Tran"
featured: true
readTime: 41
---

> [!important]
> **A stop-loss is not a guarantee — it is an instruction that only executes if the price actually trades at your level. A gap skips straight over it.**
> - A stop says "sell once the price reaches \$95." If the market closes at \$96 and reopens at \$80, the price never trades at \$95, so your stop becomes a market order that fills at \$80 — a −20% loss where you planned for −5%.
> - Prices gap because information arrives while the market is closed (earnings, a central bank, a weekend headline), because exchanges halt and reopen at a limit, and because there is simply no one bidding at your level when it gaps.
> - The danger is the mismatch: you *size* your position for the −5% stop, but the gap delivers a −20% loss. On leverage that gap-multiplier can wipe the account before any order fills.
> - Every asset gaps. Single stocks gap on earnings, leveraged ETFs gap on the reset, FX gaps over the weekend, and crypto — open 24/7 — still gaps because liquidity vanishes the instant news hits.
> - The fix is not a tighter stop. It is to **size for the gap, not the stop**: buy options for a true floor, cut overnight and event exposure, and treat the stop as a convenience for orderly markets only.

You enter a position at \$100. You are disciplined, so you place a stop-loss at \$95 — a −5% line in the sand. You have done the arithmetic: if it hits, you lose 5% of your stake, you write it off, and you live to trade the next idea. The stop is your seatbelt. You sleep fine.

Then, overnight, the company misses earnings. Or a regulator opens an investigation. Or, on a Sunday, a war starts. The market was closed the whole time, so nothing traded — but the *fair price* of your position fell off a cliff while you slept. When the market reopens the next morning, the first trade prints at \$80. Your stop at \$95? It never triggered, because the price never *touched* \$95. It jumped from \$96 (the prior close) straight to \$80 (the new open). Your seatbelt was buckled around a level the car flew right past. Instead of the −5% you planned for, you are down −20%, and there was nothing you could have done between the close and the open.

This is **gap risk**, and it is one of the most under-appreciated ways that careful traders blow up. They do everything "right" — they use stops, they keep losses small, they respect their rules — and then a single overnight move skips the entire system. The mechanic that makes this possible is simple and unforgiving: **a stop-loss is a conditional order, not a price guarantee.** It promises to *try* to sell once a trade prints at your level. It promises nothing about *where* that sale actually happens. Figure 1 shows the whole problem in one picture — the price walking calmly down to a \$96 close, then reopening at \$80, with the \$95 stop level circled in the empty space where no trade ever occurred.

![A price line that drifts down to close at ninety-six dollars then opens the next session at eighty dollars, with the ninety-five dollar stop level circled in the gap where no trade happened](/imgs/blogs/gap-risk-and-overnight-exposure-the-move-that-skips-your-stop-1.png)

This post builds gap risk from first principles. We will define what a stop actually *is* and is not. We will see exactly *why* prices gap — the three independent mechanisms. We will trace, in dollars, how a −5% stop becomes a −20% loss, and how leverage turns that −20% into a wipeout. We will rank which assets carry the most gap risk and why crypto's 24/7 market does not save you. And we will land on a playbook that treats the stop for what it is: a tool for orderly markets, not a shield against the disorderly ones. The survival thesis of this whole series — *you can only compound if you are still in the game* — has no sharper edge than this. A gap is precisely the event that can take you out of the game while your "risk management" was switched on.

## Foundations: what a stop actually is, and the words you need

Before we can see why a stop fails, we have to be precise about what it is. Most of the danger here lives in a gap (pun intended) between what people *think* a stop does and what it *mechanically* does.

**An order is an instruction to a market.** When you trade, you are not reaching into a vault and pulling out shares. You are sending an instruction to a venue — an exchange or a broker — that gets matched against someone else's opposite instruction. The two main flavors:

- A **market order** says "fill me right now at whatever price is available." You are guaranteed to trade (if anyone is there) but not guaranteed the price.
- A **limit order** says "fill me only at price X or better." You are guaranteed the price (or better) but not guaranteed to trade at all.

**A stop-loss is a conditional market order.** A stop at \$95 is dormant — it sits in the broker's system doing nothing — until the market price *trades at or through* \$95. The moment a trade prints at \$95 or below, the stop "triggers" and converts into a **market order**: sell immediately at whatever the next available price is. Read that twice. The trigger needs a *trade at your level*. The fill is at *whatever is available* once triggered. In a calm, continuous market those two prices are nearly the same — the price slid through \$95, so you sell around \$95. But if the price *gaps* past \$95 without ever trading there, two things happen at once: the trigger fires late (at the first trade *below* \$95) and the market-order fill happens wherever the book actually is — which can be far, far below.

**A continuous market visits every price on the way.** The reason a stop usually works is that intraday, a liquid market is *continuous*: to get from \$100 to \$90, the price generally trades at \$99, \$98, \$97, and so on. Every level prints. A stop at \$95 catches a trade at or near \$95 because \$95 is on the path. **Continuity is the assumption every stop quietly relies on.**

**A gap is a discontinuity — a price that skips levels.** A gap is what happens when the price moves from one level to a much lower (or higher) one *without trading in between*. This almost always happens across a *closed* market: the close was \$96, the market shut, information arrived, and the open is \$80. No trades happened at \$95, \$90, or \$85 — there were no trading hours in which they could. The price *teleported*. A stop cannot catch a teleport.

**Overnight and weekend exposure** is the umbrella term for everything you are holding while the market is closed and you cannot act. The overnight session (the hours between a market's close and its next open) and the weekend (Friday close to Monday open, plus any holidays) are windows where the world keeps moving but you are frozen. Anything that resolves during those windows lands on you as a gap.

**Liquidity** is how much you can trade without moving the price. Even within trading hours, if there are no buyers stacked up near your stop level — a thin order book — the price can *air-pocket* down: it trades at \$95, your stop fires, but the next bid is at \$88, so that is where you fill. This is a gap *inside* trading hours, caused by an empty book rather than a closed market. We cover the closed-market case mostly, but keep this cousin in mind: **a gap is any jump that skips your level, whether from a closed market or an empty one.**

With those defined, the thesis sharpens to a single sentence: **a stop protects you only across the levels the price actually trades; a gap is, by definition, the set of levels it does not.**

### The three flavors of stop order, and why none of them beats a gap

There is a common hope that a *smarter* stop order fixes this. It does not, but it is worth knowing the menu, because the differences matter in continuous markets and the *similarities* matter across gaps.

- A **stop-market** order (the default "stop-loss") triggers at your level and becomes a *market* order — fill at any price. Across a gap, it fills at the reopen. This is the order in every worked example above: it *always* fills, but the price is whatever the market is when it triggers, which after a gap is far below your level.
- A **stop-limit** order triggers at your level and becomes a *limit* order at a price you specify — "sell at \$95 or better, but never below \$93." This protects your *price* but sacrifices your *fill*: across a gap to \$80, the limit at \$93 *never executes*, so you do not sell at all and you are now riding the position down with no exit. The stop-limit converts a bad fill into *no* fill — which, in a crash, is often worse, because you are still holding while it keeps falling. **A stop-limit does not survive a gap; it just changes the failure from "filled too low" to "not filled at all."**
- A **guaranteed stop** (offered by some brokers, usually for a fee or wider spread) contractually promises a fill *at* your level regardless of gaps — the broker eats the slippage. This *does* survive a gap, but you pay for it continuously, and it is offered mainly on retail CFD-style products, not on the instruments where the biggest gaps live. When it exists, a guaranteed stop is effectively a bundled insurance product, which is why it costs money — the broker is selling you the very gap protection a plain stop lacks.

The pattern across all three: the only versions that survive a gap are the ones that *cost money up front* (a guaranteed stop, or — better and more flexible — a bought put). The free versions either fill you far too low (stop-market) or not at all (stop-limit). **There is no free order type that turns a discontinuous market into a continuous one.** Gap protection is a thing you buy, not a box you check.

A final foundational point on *direction*: gaps cut both ways, and the asymmetry matters. A gap *against* your position is unstoppable and uncapped (the price can open arbitrarily far away). A gap *in your favor* is a windfall — but you cannot rely on it, and a short seller faces *unlimited* upside-gap risk: if a stock you are short gaps up 200% on a buyout, your stop to buy back fills at the gapped price and your loss can exceed your entire stake. **For shorts, the dangerous gap is up, and it has no ceiling — which is the deeper reason naked short selling is so much riskier than being long.**

#### Worked example: the stop that fills exactly where you planned

Let's anchor the *good* case first, so the failure is unmistakable. Take the recurring **\$100,000 account**. You buy \$10,000 of a stock at \$100 — 100 shares, 10% of the account in this one name. You set a stop at \$95 (−5%).

The market is open and orderly. Bad news trickles out during the session; the price slides: \$99, \$98 … \$96, \$95. At \$95, a trade prints. Your stop fires and becomes a market order. The order book is deep — there are buyers stacked at \$94.98, \$94.95 — so you fill 100 shares at an average of about \$94.95.

- Sale proceeds: 100 × \$94.95 = \$9,495.
- Cost: 100 × \$100 = \$10,000.
- Loss: \$10,000 − \$9,495 = **−\$505**, about −5.05% on the position, or **−\$505 on the \$100,000 account (−0.5%)**.

You planned for −5% and you got −5%. The stop did its job because the price *walked* to your level. *In a continuous market, the stop level and the fill level are nearly the same number — which is exactly the assumption a gap breaks.*

## Why prices gap: three independent mechanisms

A gap is not one thing. There are three distinct mechanisms that produce a price that skips your stop, and they can stack on top of one another. Understanding which is which tells you when you are exposed.

**1. Information arrives while the market is closed.** This is the classic overnight gap. Markets are open for a fraction of the day — a US stock exchange trades roughly 6.5 hours; the other ~17.5 hours, plus weekends and holidays, it is dark. But the *world* runs 24/7. A company reports earnings after the close. A central bank surprises on a Sunday. A drug trial fails overnight. A war breaks out on a Saturday. When the market reopens, it must *reprice in a single instant* to absorb everything that happened while it was shut. The open is not a continuation of the close — it is a fresh auction that clears at whatever price balances all the new information. If the news was bad enough, that clearing price is a long way below the prior close, and there were no trading hours in between to walk the price down. **The gap is the market catching up all at once.**

**2. The exchange halts the price and reopens it at a limit.** Many markets have **circuit breakers** and **price limits** — rules that *stop trading* when a price moves too far too fast. Futures markets have daily "limit-down" levels; individual stocks get **single-stock halts** on big moves; whole indices halt on extreme days. The intent is to give the market a breather and prevent a disorderly cascade. But the side effect is a *guaranteed* gap: while the instrument is **locked limit-down** or halted, *no trades are allowed at all*. Your stop sits inside the frozen zone and cannot fill, because the exchange forbids any trade there. When the halt lifts, the instrument reopens — and it reopens at whatever price balances the new supply and demand, which after a limit-down move is *below* the limit. The exchange itself manufactured the discontinuity. We will see this in Figure 5.

**3. There is no liquidity at your level.** Even with the market open and no halt, your stop can be skipped if the order book is empty around it. Take a thinly traded small-cap, or a crypto pair at 3 a.m.: the price trades at \$95, your stop fires as a market order, but the best bid is way down at \$88 because nobody is willing to buy in between. You fill at \$88. This is a *liquidity gap* — a discontinuity caused by an air-pocket in the book rather than a closed market. It is the same failure mode and it compounds the other two: when bad news hits a closed market, the reopening book is *also* thin (everyone is hitting bids, no one is providing them), so the reopen gaps further than the "fair" news impact alone would suggest. Gaps feed on the liquidity withdrawal we cover in the [companion post on liquidity risk](/blog/trading/risk-management/liquidity-risk-you-cant-sell-what-no-one-will-buy).

The key insight across all three: **none of them is rare, and all of them are invisible to your stop.** Your stop's logic — "fire when a trade prints at \$95" — has no concept of a closed market, a halt, or an empty book. It does exactly what it was built to do, which is nothing, until a trade appears below your level. By then the damage is done.

#### Worked example: the same stop, now across a gap

Back to the **\$100,000 account** and the identical \$10,000 position — 100 shares at \$100, stop at \$95. Same trader, same discipline. The only difference: the bad news lands *overnight* instead of intraday.

The stock closes the day at \$96 — *above* your \$95 stop, so the stop never fired during the session. After the close, the company reports a disastrous quarter. Overnight, no shares change hands (the market is closed). The next morning the stock reopens — the first trade of the day, the new clearing price — at \$80.

What happens to your stop? At the open, the first trade prints at \$80, which is *below* \$95, so the stop triggers. It converts to a market order. But the market is now at \$80, so that is where you fill.

- Sale proceeds: 100 × \$80 = \$8,000.
- Cost: 100 × \$100 = \$10,000.
- Loss: \$10,000 − \$8,000 = **−\$2,000**, or **−20% on the position** — **−\$2,000 on the \$100,000 account (−2%)**.

You planned for −\$505 and you took −\$2,000. The stop did not malfunction; it executed *exactly* as designed. The price simply never visited \$95. *The −5% you "limited" your loss to was a number the market was never obligated to honor — your real risk was the size of the gap, which you never chose.*

## The stop illusion: you sized for −5%, the gap charges −20%

Here is the cruelest part. The number that hurts you is not the −20% on its own — it is the *mismatch* between what you sized for and what you got. A stop is not just an exit; it is the basis of your **position sizing**. The whole point of a risk-controlled approach is: "I am willing to lose \$X on this trade, my stop is Y% away, so I buy a position big enough that a Y% move equals exactly \$X." The stop *defines* how big the trade is. If the stop is a lie, the sizing is a lie.

Figure 2 makes this concrete across three position sizes on the \$100,000 account. The amber bars are the loss you *intended* — the −5% stop. The red bars are the loss the −20% gap actually delivers. The gap turns every bar into roughly **four times** what you planned. A trader who put 10% of the account in one name expected to risk \$500 and instead lost \$2,000; one who concentrated 50% expected to risk \$2,500 and lost \$10,000 — a tenth of everything, on a single overnight, with the seatbelt on.

![Grouped bars comparing the intended five percent stop loss against the realised twenty percent gap loss across three position sizes on a one hundred thousand dollar account](/imgs/blogs/gap-risk-and-overnight-exposure-the-move-that-skips-your-stop-2.png)

The "illusion" is that the stop's percentage tells you your risk. It does not. **Your risk per trade is not the stop distance — it is the worst realistic gap.** If a single stock can plausibly gap −20% on an earnings miss (and they do — far worse than that on biotechs and small caps), then a position you "stopped" at −5% really carries −20% of *unstoppable* risk. The stop is a comfort blanket sewn over a trapdoor.

This connects straight to the recovery asymmetry that anchors this whole series. A −5% loss needs a +5.3% gain to recover — trivial. But the realized −20% loss needs a +25% gain to get back to even. And if the gap is worse — a −50% biotech collapse — you need +100% just to break even, and a −60% gap needs +150%. The gap does not merely cost you more money; it pushes you onto the steep part of the recovery curve, where each extra percent of loss demands disproportionately more to undo. We derive that asymmetry in full in [the arithmetic of ruin](/blog/trading/risk-management/leverage-and-the-arithmetic-of-ruin); here, just note that *the gap is an engine for moving you up that curve against your will.*

#### Worked example: the same dollar risk, two very different outcomes

Suppose your rule is "risk no more than 1% of the account (\$1,000) per trade." You believe your \$95 stop is 5% away, so you size the position at **\$20,000** (because 5% of \$20,000 = \$1,000). Twenty thousand dollars is 20% of the account in one name — already aggressive, but your rule "permits" it because the stop "caps" the loss at \$1,000.

Now the −20% gap hits:

- Position: \$20,000 at \$100 → 200 shares.
- Gap reopen at \$80: 200 × \$80 = \$16,000.
- Loss: \$20,000 − \$16,000 = **−\$4,000**, or **−4% of the \$100,000 account**.

Your "1% risk per trade" rule just delivered a **4% loss** — four times your stated limit — because the rule was built on the stop distance, not the gap distance. Had you sized for the *gap* instead (assume a plausible −20% and risk 1% against *that*), you would have bought only **\$5,000** of the stock (20% of \$5,000 = \$1,000), and the same −20% gap would have cost exactly \$1,000, as intended. *Sizing against the stop quadruples your true risk; sizing against the gap restores the limit you thought you had.*

## A stop in a continuous market vs across a gap

It is worth seeing the two regimes side by side, because the difference is mechanical, not a matter of luck. Figure 4 contrasts them. On the left — the continuous market — the price visits every level on the way down, so a trade prints at \$95, the stop triggers, buyers are nearby, and you fill near \$95 with the −5% you planned for. On the right — across a gap — the last trade is \$96, then the market closes, then it opens at \$80; no trade ever printed at \$95, so the stop triggers only at the open and fills at \$80, four times the loss you sized for.

![A before and after comparison showing a stop filling near ninety-five dollars in a continuous market on the left versus filling at the eighty dollar reopen price across a gap on the right](/imgs/blogs/gap-risk-and-overnight-exposure-the-move-that-skips-your-stop-4.png)

The single thing that flips between the two columns is **continuity**. Everything else — your order, your discipline, your stop level — is identical. This is why "use a tighter stop" is not a fix for gap risk: a stop at \$98 fares no better across a gap to \$80 than a stop at \$95 does. The gap jumps over *both*. Tightening the stop only changes your fate in the *continuous* world; in the *gapped* world, every stop below the close is equally useless. **You cannot solve a discontinuity problem with a finer placement of a continuous tool.**

There is a second-order trap here too. Because tighter stops get hit more often in normal trading (random wiggles cross them), traders who have been "stopped out for nothing" learn to *widen* their stops or remove them entirely. That makes their continuous-market experience smoother — fewer annoying small losses — while doing absolutely nothing about the gap. The smoother ride lulls them into bigger size. Then the gap arrives and finds them larger than ever. *The behavior that reduces small, visible losses often increases the large, invisible ones.*

## The fat tail of overnight moves

A natural objection: "Sure, gaps happen, but they are rare — most of my risk is intraday." The data says the opposite. A large fraction of the total price movement in many markets happens *overnight*, not intraday, and — crucially — the *distribution* of overnight gaps has a **fatter tail** than the distribution of intraday moves. Big jumps are not just possible overnight; they are *concentrated* there, because that is when accumulated, unhedgeable information gets released in one shot.

Figure 3 overlays the two distributions on a log-count axis. The intraday moves (blue) cluster in a near-normal bell with thin tails — the price wiggles around, mean-reverts, and only rarely does anything dramatic within a session. The overnight gaps (red) share a similar quiet core but have a *heavy* tail: the −8%, −12%, −20% bins are far more populated than the intraday distribution would ever produce. The amber band marks the danger zone where the move that skips your stop lives — and it lives almost entirely in the overnight distribution.

![Two overlaid distributions on a log count axis showing intraday moves with thin tails versus overnight gaps with a fat tail of large jumps, with the danger zone shaded](/imgs/blogs/gap-risk-and-overnight-exposure-the-move-that-skips-your-stop-3.png)

This is the same fat-tail phenomenon that wrecks naive risk models elsewhere in this series — see [fat tails and the normal-distribution trap](/blog/trading/risk-management/fat-tails-and-the-normal-distribution-trap) for why a Gaussian model under-counts these events by orders of magnitude. The practical upshot for gaps: **your "normal" daily risk and your overnight gap risk are drawn from two different distributions, and the overnight one has the fat tail.** A risk model calibrated on intraday vol — which is most of them — will badly under-estimate how often, and how far, you get gapped. If you measured your "typical day" and sized for it, you sized for the wrong distribution.

There is also a structural reason the overnight tail is fat: information is *lumpy*. Earnings, FOMC decisions, jobs reports, FDA rulings, geopolitical shocks — these arrive at *scheduled or unscheduled discrete moments*, and many of them are timed to land outside trading hours precisely so the market can absorb them in an orderly auction at the next open. The auction is orderly for the *market*; it is a cliff for *you* if you are on the wrong side and frozen. The lumpiness of information *guarantees* a fat overnight tail. It is not a statistical accident you can diversify away within a single name — it is the structure of how news and markets interact.

A useful way to decompose a stock's risk is to split each day's return into two pieces: the **close-to-open** move (the overnight gap, which you cannot trade through) and the **open-to-close** move (the intraday session, which you can). For many individual stocks, a surprising share of the *total* variance — and almost all of the worst single-day shocks — lives in the close-to-open piece. The intraday piece is the part your stop can manage; the close-to-open piece is the part it cannot. When you carry a position overnight, you are *specifically* choosing to bear the un-manageable half of the risk, and it is the half with the fat tail. A trader who flattens by the close is not being timid — they are declining to hold the one component of risk their tools cannot touch. This is also why "buy-and-hold" investors, who *never* flatten, must instead defend themselves through *sizing and diversification* rather than stops: they have accepted permanent overnight exposure, so the stop was never their real defense in the first place.

One more subtlety: gaps are not independent of *how crowded* a position is. When everyone owns the same trade, the overnight news that turns it sour forces the *same* crowd to sell into the *same* reopen auction — so the gap is amplified by the rush for the one exit. A lightly-held name might gap −8% on a given piece of bad news; the *crowded* version of that same name gaps −20%, because the reopening book is buried under everyone's simultaneous sell orders. The gap and the crowd are multiplicative. This is the link between gap risk and the [crowded-trade exit problem](/blog/trading/risk-management/liquidity-risk-you-cant-sell-what-no-one-will-buy): the gap is *where* the crowded exit gets settled, all at once, at a price none of them chose.

## Limit-down: when the exchange manufactures the gap

The overnight gap is bad news arriving while you sleep. The limit-down gap is worse in one specific way: the exchange *prevents* you from exiting even while the market is technically "open."

Here is the mechanism. Many futures, and individual stocks under volatility rules, have **price limits** or **circuit breakers**: if the price falls by some threshold (say −7% for an index-level breaker, or a contract-specific daily limit for a future), trading **halts**. During the halt, no trades occur. For a future, it may go **locked limit-down** — quotes pile up at the limit price with no buyers, and it simply cannot trade lower until a session resets or the limit expands. Your stop is somewhere inside that frozen band, and it *cannot* fill, because the exchange has made trading at those prices illegal for the duration of the halt.

When the halt lifts, the instrument **reopens** — and it reopens via an auction that finds the price where supply meets demand. After a forced halt, that price is *below* the limit, because all the sell pressure that built up during the halt now has to clear at once. So the sequence is: slide into the limit → halt (no fills) → reopen far below. Figure 5 shows it. The price slides into the −7% down-limit at \$93. Your \$95 stop is *above* the limit, sitting inside the locked zone — it never fills. The instrument halts (the gray band), then reopens at \$84 — a −16% level it teleported to from \$93. The exchange built the discontinuity into the rules.

![A step price path that slides into a down-limit at ninety-three dollars, halts in a locked zone where the ninety-five dollar stop cannot fill, then reopens at eighty-four dollars](/imgs/blogs/gap-risk-and-overnight-exposure-the-move-that-skips-your-stop-5.png)

The irony is that circuit breakers exist to *protect* the market — to stop a cascade and let cooler heads find a clearing price. From a systemic view they often do help. But from the seat of an individual holder with a stop, **a halt is a guaranteed gap with a guaranteed inability to act**. You are not even allowed to take your medicine at the limit price; you must wait for the reopen, which is designed to be lower. Anyone who lived through the COVID crash of February–March 2020 remembers index-level halts triggering on the way down; anyone in commodities knows the terror of a contract going locked limit against them for *days*, unable to exit at any price as losses compound. The lesson: **a stop is not just useless across a halt — the halt actively traps you while the market reprices against you.**

#### Worked example: trapped through a limit-down halt

You are short volatility... no — let's keep it simple and concrete on the **\$100,000 account**. You hold a \$15,000 position in an index future (let's say 15% of the account, in notional terms ignoring margin for the moment), entered at an index level we will call 100 for round numbers, with a stop at 95 (−5%).

Bad news hits mid-session. The future slides to the −7% daily down-limit at 93 and goes **locked limit-down**. Your stop at 95 is above the limit, inside the halted zone — it cannot fill. You sit, frozen, watching offers stack up with no bids. After the halt, the contract reopens at 84 (−16%) and you finally exit there.

- Position: \$15,000 at level 100.
- Exit at level 84: value = \$15,000 × (84 / 100) = \$12,600.
- Loss: \$15,000 − \$12,600 = **−\$2,400**, or **−16% on the position** — **−\$2,400 on the account (−2.4%)**.

Your −5% stop delivered a −16% loss, and you were *legally prevented* from exiting in between. *When the exchange halts the market, your stop does not just miss — it is forbidden from working, and you take the full reopen gap whether you like it or not.*

## Gap risk by asset class: everything gaps, but not equally

Gap risk is universal, but its *size* and *timing* vary enormously by what you trade. Figure 6 ranks the major asset classes by an illustrative worst-case single-event gap, colored by how dangerous the stop-skip is.

![A horizontal bar chart ranking gap risk by asset class from single stocks on earnings and biotechs at the top down through futures and weekend foreign exchange to twenty four hour crypto](/imgs/blogs/gap-risk-and-overnight-exposure-the-move-that-skips-your-stop-6.png)

Walking the ladder:

**Single stocks on earnings** are the textbook gap. Four times a year, a company reports, and the stock can move 10–30% on the print — overnight, in the after-hours session, with thin liquidity. A −20% earnings gap is utterly routine; −40% on a guidance cut or a fraud disclosure happens regularly. **If you hold a single stock through its earnings date with a stop, your stop is decorative.** The market knows the report is coming; the *expected move* is priced into options precisely *because* everyone knows the gap is likely.

**Small-cap and biotech on binary events** are the most violent. A biotech awaiting an FDA decision or a trial readout is a coin flip with the entire company's value on the line. The stock often *halts* ahead of the news, then reopens −50%, −70%, or +200%. There is no stop that survives a binary event — the gap can be most of your capital. **These positions must be sized as if the stop does not exist, because for the event, it does not.**

**Leveraged and inverse ETFs** carry a sneaky double gap. They reset their leverage *daily*, so they amplify the underlying's overnight gap by their leverage factor *and* suffer decay over time. A 3× leveraged ETF on an index that gaps −7% opens roughly −21%, and the daily reset means the math compounds against you in volatile, gappy periods. Worse, in extreme cases the issuer can *terminate* the product — Volmageddon in February 2018 saw a short-vol ETP lose ~96% of its value after the close and get wound down. **A leveraged ETF gap is the underlying's gap times the leverage, with extra failure modes.**

**Index futures** trade nearly around the clock (~23 hours), which sounds protective — and it partly is, because there are fewer fully-closed windows. But "open" is not "liquid." Overnight futures liquidity is thin, so a news shock at 2 a.m. produces a real gap on a thin book, and futures still face the limit-down halts above. The near-24-hour session means your stop *might* fill in the overnight session — but possibly at a gapped price on an empty book.

**FX majors** are the weekend-gap asset. Spot FX trades ~24 hours on weekdays, then **closes from Friday evening to Sunday evening**. Anything that happens over the weekend — an election, a referendum, a central-bank surprise, a geopolitical shock — lands as a **Monday-open gap**. The Swiss franc's un-pegging in January 2015 (a weekday, but a similar discontinuity) moved the franc ~20–30% in *minutes* with no liquidity, blowing up brokers and traders whose stops filled tens of percent away. **FX is calm 99% of the time and then gaps the entire year's range over one weekend.**

**Crypto trades 24/7** — and many beginners assume that means *no gaps*, because the market never closes. This is the most dangerous misconception of all. Crypto absolutely gaps. It gaps because: (1) news still arrives in discrete shocks (an exchange collapse, a regulatory ban, a protocol exploit), and the *price* reprices instantly even though the market is "open"; (2) liquidity is wildly uneven — at 4 a.m. on a Sunday the order book can be paper-thin, so a large sell air-pockets the price down 20% with no halt to stop it; (3) there are no circuit breakers on most venues, so nothing arrests a cascade. The 24/7 market does not *prevent* gaps; it just *relocates* them from the open to any moment of thin liquidity plus news. **"Always open" means "always exposed," not "never gaps."**

The through-line of Figure 6: the assets with the biggest stop-skipping gaps are the *concentrated, event-driven, single-name or leveraged* ones. That is not a coincidence — it is the same lesson as [concentration and position limits](/blog/trading/risk-management/concentration-and-position-limits-the-one-trade-that-can-end-you): the gap is just the mechanism by which a concentrated, event-exposed bet collects its tail.

#### Worked example: a weekend FX gap on 30× leverage

FX deserves its own example because the leverage on offer is extreme and the gap window is predictable — every single weekend. Take a retail trader with a **\$100,000 account** who goes long a currency pair at **30× leverage**: \$3,000,000 of notional. They place a stop 1% away, reasoning that 1% of \$3,000,000 is \$30,000, a 30% account loss they consider their absolute max. They feel protected — the stop is "right there."

On Friday evening the market closes. Over the weekend, a surprise election result hits the currency. The pair reopens Sunday evening **−3% gapped** — modest by FX-crisis standards (the 2015 franc move was many times this).

- Notional: \$3,000,000.
- Gap loss: \$3,000,000 × 3% = **−\$90,000**.
- Account before: \$100,000. Account after: \$100,000 − \$90,000 = **\$10,000**.

A −3% weekend gap — small, routine, the kind that happens several times a year somewhere in FX — vaporized **90% of the account**, despite a 1% stop, because 30× leverage turned a 3% move into a 90% loss and the weekend close meant the stop could not fire until the gapped reopen. To recover the \$10,000 back to \$100,000 requires a **+900% gain**. *At high leverage the weekend is not a pause in risk — it is the single most dangerous window of the week, and the stop is asleep through all of it.*

## Leverage turns a gap into a wipeout

Everything above assumed you were trading your own cash. Add **leverage** and the gap stops being a bad day and becomes a *terminal* event. This is where gap risk and the arithmetic of ruin meet.

Leverage scales your gains and losses by a factor L. A −20% gap on an unlevered position is a −20% hit to your equity — painful, survivable. The same −20% gap on a 5× levered position is a **−100% hit to your equity** — the account is *gone*. And the gap is exactly the move your stop cannot prevent, so on leverage you do not even get the consolation of an orderly exit. The margin call arrives, but the position has already gapped through it; there is nothing left to liquidate at a level that saves you.

Figure 7 shows it in two panels. The top panel is the underlying gapping from a \$96 close to an \$80 open — the same −20% gap as Figure 1. The bottom panel is the *account equity* for three leverage levels through that gap. The 1× line (green) drops to \$80,000 — bruised but alive. The 3× line (amber) drops to \$40,000 — a −60% blow that needs a +150% gain just to recover. The 5× line (red) hits **zero** — wiped, because a −20% gap times 5 is −100% of equity, and the stop could not fill in time to prevent it.

![A two panel chart showing an underlying price gapping down twenty percent on top, and account equity for one times three times and five times leverage below, with the five times position wiped to zero](/imgs/blogs/gap-risk-and-overnight-exposure-the-move-that-skips-your-stop-7.png)

The brutal arithmetic: **with leverage L, a gap of g% costs you L × g% of your equity, and the gap is unstoppable.** A 5× book is wiped by any −20% gap; a 10× book is wiped by any −10% gap — which is a *single bad earnings print*. This is why leveraged single-stock positions held through earnings are a classic blow-up, and why FX traders at 50× or 100× leverage are destroyed by weekend gaps that an unlevered holder would shrug off. The stop you placed gives a false sense that the leverage is "controlled." It is not. **On leverage, the stop is the seatbelt and the gap is the cliff — the seatbelt does not help if the car goes off the cliff.** The full derivation of how leverage compresses your distance-to-ruin lives in [leverage and the arithmetic of ruin](/blog/trading/risk-management/leverage-and-the-arithmetic-of-ruin); the gap is the specific event that collects on that compression.

#### Worked example: the 5× book wiped overnight

Take the **\$10,000,000 book** this time. The manager wants more exposure than the cash allows, so they run **5× leverage** — \$50,000,000 of notional long in a basket that, unbeknownst to them, is concentrated in a name reporting earnings tonight. They place stops "limiting" each position to −4%, feeling prudent: "Even if everything goes wrong, I'm capped at a few percent."

Overnight, the key name misses badly; the basket gaps −20% at the open before any stop can fill.

- Notional: \$50,000,000.
- Gap loss on notional: \$50,000,000 × 20% = **−\$10,000,000**.
- Equity before: \$10,000,000.
- Equity after: \$10,000,000 − \$10,000,000 = **\$0**.

The entire \$10,000,000 book is gone in one overnight gap, despite "−4% stops" on every position. The stops were sized against the *stop distance*; the gap charged the *gap distance* times the *leverage*; the product exceeded 100% of equity. *Leverage does not just amplify the gap — it makes a single ordinary overnight move the difference between a drawdown and a death.*

## Common misconceptions

**"My stop-loss guarantees I can't lose more than 5%."** No. A stop guarantees a *trigger* once the price trades at your level and a *market-order fill* thereafter — it guarantees nothing about the *fill price* if the price gaps past your level. A −5% stop that fills on a −20% gap loses **−20%**. The only thing that *guarantees* a maximum loss is a *long option* (a put you bought), whose payoff floor is contractual and does not depend on continuous trading.

**"A tighter stop reduces gap risk."** No. A gap to \$80 jumps over a stop at \$98 just as easily as one at \$95. Tighter stops only change outcomes in the *continuous* regime (where they get hit more often, costing you small losses), and do nothing in the *gapped* regime. Tightening the stop while concentrating size is the worst of both worlds: more small losses *and* the same fat gap.

**"Crypto trades 24/7, so it doesn't gap."** Wrong, and dangerously so. Crypto gaps on news shocks (repriced instantly even though the venue is "open") and on liquidity air-pockets (thin books at odd hours), with *no* circuit breakers to arrest a cascade. A 24/7 market relocates the gap from the open to any moment of thin liquidity plus news — it does not remove it. The −20% wick at 3 a.m. is a gap by any other name.

**"I sized for 1% risk per trade, so I'm fine."** Only if you sized against the *gap*, not the *stop*. If your 1% risk was computed off a 5% stop, a −20% gap delivers **4%** — four times your stated limit. The position-sizing rule is only as honest as the loss number you feed it; feed it the stop distance and it lies, feed it the plausible gap distance and it tells the truth.

**"Holding overnight is the same risk as holding intraday."** No — overnight moves are drawn from a *fatter-tailed* distribution because lumpy information (earnings, central banks, weekends) is concentrated outside trading hours. Your intraday-calibrated risk model systematically under-counts overnight gaps. The same position is materially riskier held overnight than flat-by-close, even though the chart looks continuous.

**"A circuit breaker protects me by halting the fall."** It protects the *market's orderliness*, not your *exit*. While the instrument is halted or locked limit-down, your stop *cannot fill* — you are trapped, and the reopen is designed to clear *below* the limit. The halt converts a slide into a guaranteed gap *and* removes your ability to act during it.

## How it shows up in real markets

Gap risk is not a textbook hypothetical — it is the proximate cause of a long list of blow-ups. Two recent episodes from the series' crisis record make it vivid.

**The yen-carry unwind, 5 August 2024.** A massively crowded "carry" trade — borrow cheap yen, buy higher-yielding assets worldwide — began to unwind as the Bank of Japan shifted policy and the yen strengthened. The deleveraging was reflexive and fast. On 5 August 2024, the **Nikkei fell −12.4% in a single day — its worst since 1987's Black Monday** — and Wall Street's "fear gauge," the VIX, **spiked to an intraday peak around 65.7** (TSE/Nikkei; Cboe). For anyone holding the wrong side with a stop, the move was a *gap*: markets that had closed Friday at one level reopened into a cascade, and the speed meant fills happened far from any "−5%" line. A leveraged carry position with a stop offered no protection — the unwind skipped straight through it, exactly as Figure 7's 5× line skips to zero. The carry trade is a textbook of how a *crowded* position turns an orderly exit into a gap, because everyone is trying to sell through the same door at once.

**The COVID crash, February–March 2020.** This was gap risk at the index level. As the pandemic spread, US equities suffered the **fastest bear market on record**, an S&P 500 drawdown of about **−34% from the 19 February peak to the 23 March trough**, with the **VIX closing at a record 82.69 on 16 March 2020** (Cboe; S&P Dow Jones Indices). The mechanics included repeated *index-level circuit breakers* halting trading on the way down — the manufactured-gap mechanism of Figure 5, at scale. Many overnight sessions saw S&P futures gap to **limit-down before the US cash open**, so the stock market's open was a fresh, far-lower auction price, not a continuation. A holder with stops watched the market repeatedly halt, reopen lower, halt again — every reopen a gap, every halt a window in which no stop could fill. The trader's takeaway: in the regimes when you *most* want your stop to work — a fast, correlated crash — is exactly when the market is *most* discontinuous and your stop *least* able to fire.

A third, structural example worth naming: the **single-stock earnings gap**, which happens *thousands of times a quarter* without making headlines. Any quarter, a meaningful slice of reporting companies gap more than 10% on the print. None of these are "tail events" in the rare sense — they are the *base rate* of holding a single name through its report. The trader who is gapped on earnings did not get unlucky; they took a known, recurring, un-stoppable risk and called it "managed" because they had a stop. *The crises that make the news and the earnings gaps that do not are the same mechanism at different scales: a price that repriced while you could not act.*

## The gap-risk playbook

Gap risk cannot be eliminated — you cannot force the market to trade at every level, and you cannot prevent news from arriving while it is closed. But it can be *managed*, and the discipline is concrete. Survival here means accepting that the stop is a tool for orderly markets and building a separate defense for the disorderly ones.

- **Size for the gap, not the stop.** This is the master rule. Estimate the *plausible worst-case gap* for the asset — −20% for a liquid single stock, −40%+ for a biotech on a binary event, the weekend range for FX — and size the position so that *that gap* equals your risk budget, not so that your stop distance does. If 1% of the account is \$1,000 and the plausible gap is −20%, the position is \$5,000, not \$20,000. The position will feel "too small." That feeling is the sign you have finally sized for reality.

- **Use long options for a true floor.** A stop is a conditional order that can be skipped; a **bought put** is a *contract* that pays off no matter how far or fast the price gaps. If you must hold a position through an event, the only protection that survives a gap is one whose payoff is contractual, not execution-dependent. The trade-off is the premium you pay — see [hedging a portfolio with options](/blog/trading/options-volatility/hedging-a-portfolio-with-options-protective-puts-collars-and-tail-risk) for protective puts and collars, and weigh the [cost versus payoff of tail hedging](/blog/trading/risk-management/tail-hedging-cost-vs-payoff-paying-to-survive-the-worst-day) before assuming insurance is "too expensive." Across a real gap, the put is the difference between a planned cost and an unplanned wipeout.

- **Cut overnight and event exposure deliberately.** The cleanest way to avoid an overnight gap is to *not be there for it*. Flatten or reduce single-name positions before earnings; trim leverage into known event windows (FOMC, jobs reports, elections, FDA dates); avoid carrying concentrated, illiquid positions over weekends. You give up the overnight expected return — but you also give up the fat overnight tail, and for a survival-first trader that is a good trade. Decide explicitly which events you will hold through (with an option floor) and which you will sidestep (by being flat).

- **Cap leverage by the gap, not the margin.** Your broker will let you lever far past the point where a routine gap wipes you. Set your *own* leverage limit such that the plausible worst-case gap costs a *survivable* fraction of equity. If a −20% gap is realistic and you never want to lose more than 20% of equity to one event, your leverage cap is **1×** for that exposure. A 5× book is, mathematically, a bet that nothing you hold ever gaps −20% — a bet the market wins routinely.

- **Treat the stop as a convenience, not a guarantee.** Keep using stops for orderly-market risk control — they genuinely cap your loss when the price walks to your level. Just *price in* that they will fail across gaps, halts, and air-pockets, and never let a stop be the *reason* a position is sized as large as it is. The stop manages the continuous risk; your *size* and your *options floor* manage the discontinuous risk. They are different jobs done by different tools.

One way to operationalize all of this is a single pre-trade question: *if this position gapped to its worst plausible level overnight, would I survive it?* If the honest answer is "no," the position is too big or too levered, full stop — no stop placement fixes a "no." If the answer is "yes, but it would hurt," that is a position you can hold, optionally with an option floor for the events you choose to sit through. And if the answer is "I cannot even estimate the worst plausible gap," you are trading something whose tail you do not understand, which is its own warning. The discipline is not to predict the gap — gaps are unpredictable by construction — but to *pre-pay for survival* so that when one arrives, it cannot end you. A gap you have sized for is an annoyance; a gap you have not is a gravestone, and the only difference between the two was a decision you made *before* the news ever hit.

The whole point of this series is that you compound only if you survive, and the gap is one of the purest tests of that idea: it is the loss that arrives precisely when your defenses are switched on, in the size you never chose, at the speed you cannot answer. The traders who survive it are not the ones with the tightest stops — they are the ones who sized small enough that the gap, when it came, was a bruise and not a burial.

### Further reading

- [Leverage and the arithmetic of ruin](/blog/trading/risk-management/leverage-and-the-arithmetic-of-ruin) — why a gap on leverage maps to a wipeout, and how leverage compresses your distance to zero.
- [Liquidity risk: you can't sell what no one will buy](/blog/trading/risk-management/liquidity-risk-you-cant-sell-what-no-one-will-buy) — the intraday cousin of the gap: an air-pocket in the order book that skips your level even with the market open.
- [Tail hedging: cost vs payoff — paying to survive the worst day](/blog/trading/risk-management/tail-hedging-cost-vs-payoff-paying-to-survive-the-worst-day) — when paying an insurance premium for a true floor beats trusting a stop.
- [Hedging a portfolio with options: protective puts, collars, and tail risk](/blog/trading/options-volatility/hedging-a-portfolio-with-options-protective-puts-collars-and-tail-risk) — the mechanics of the only protection that survives a gap.
- [Position sizing and risk of ruin in options trading](/blog/trading/options-volatility/position-sizing-and-risk-of-ruin-in-options-trading) — sizing against the real loss distribution, not the comfortable one.
