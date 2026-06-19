---
title: "Circuit breakers, halts, and the legal plumbing of a crash"
date: "2026-06-19"
publishDate: "2026-06-19"
description: "A beginner-friendly deep dive into the rules that pace a market crash — market-wide circuit breakers, single-stock LULD bands, clearinghouse margin, and the reopen auction — and how to trade around them."
tags: ["regulation", "market-structure", "circuit-breakers", "luld", "volatility", "clearinghouse", "margin", "flash-crash", "risk-management", "trading-playbook"]
category: "trading"
subcategory: "Law & Geopolitics"
author: "Hiep Tran"
featured: true
readTime: 41
---

> [!important]
> **TL;DR** — When markets fall fast, a body of rules decides whether the decline is orderly or a cascade: market-wide circuit breakers, single-stock price bands, clearinghouse margin, and the reopen auction are the kill-switches, and knowing them tells you what happens at the −7%, −13%, and −20% levels and why liquidity vanishes near a band.
>
> - Market-wide circuit breakers (MWCB) halt all US trading for 15 minutes at a −7% (Level 1) or −13% (Level 2) S&P 500 decline, and close the market for the day at −20% (Level 3).
> - Single-stock Limit Up-Limit Down (LULD) bands pause one stock for 15 seconds when its quote sits at a band 5%, 10%, or 20% from a rolling reference price, depending on the stock's tier.
> - A volatility spike forces clearinghouse margin calls, which force selling, which raises volatility again — a reflexive loop that breakers are designed to interrupt, not prevent.
> - The one number to remember: the −20% Level 3 breaker has **never** fired intraday since the modern thresholds were adopted; the system is built to pace a crash, not to stop one.

On Monday, March 16, 2020, the S&P 500 opened by falling so fast that within the first minute of trading the market hit a hard stop. Trading in every US stock froze for fifteen minutes. It was the third time in eight trading sessions that a market-wide circuit breaker had tripped — a mechanism that, before that month, had not fired once in the twenty-three years since it took its modern form. To anyone watching a screen, the experience was disorienting: prices simply stopped moving, the order book went quiet, and for a quarter of an hour there was nothing to do but wait.

That pause was not an accident, a glitch, or an exchange "turning off the market" on a whim. It was the law working exactly as designed. A specific rule — written into the exchanges' rulebooks and approved by the Securities and Exchange Commission (SEC) — says that when the S&P 500 falls 7% from the prior day's close, every US equity venue must stop trading for fifteen minutes. The rule has a number, a threshold, a duration, and a long paper trail going back to a single terrifying day in October 1987.

This post is about that paper trail and the machinery it built. When a market crashes, it does not fall through empty air. It falls through a dense lattice of rules — circuit breakers, price bands, margin formulas, auction mechanics — each of which changes how, and how fast, the price can move. Understanding that lattice is not academic. It tells you precisely what will happen at −7%, what happens to a single stock that gaps, why the bid disappears just as you most want to sell, and how a margin call on someone else can force a wave of selling onto you. It is, in the most literal sense, the legal plumbing of a crash.

![Four legal kill-switches that pace a market crash, from single-stock bands to market-wide halts](/imgs/blogs/circuit-breakers-halts-and-the-legal-plumbing-of-a-crash-1.png)

This connects directly to the series' core idea — that a rule changes the *rules of the game*, and the practitioner who reads the rule early prices the consequence before it bites. We trace that idea in [how law moves markets: the transmission chain](/blog/trading/law-and-geopolitics/how-law-moves-markets-the-transmission-chain). Here, the rule and the price are almost the same thing: the circuit-breaker threshold *is* a price level, and the market trades right up to it.

## Foundations: how the kill-switches work

Before we can trade around these rules, we have to define them from zero. There are four pieces of machinery, and they operate at different scales and on different triggers. Let us build each one up carefully.

### Where the rules come from: Black Monday, 1987

Every piece of machinery in this post traces back to one date: **Monday, October 19, 1987 — "Black Monday."** On that single day the Dow Jones Industrial Average fell **22.6%**, the largest one-day percentage decline in its history, before or since. There was no triggering headline of commensurate size — no war, no bank failure, no policy bombshell that morning. The market simply fell, and as it fell it fed on itself.

The post-mortem identified a mechanism that should sound familiar by now: a reflexive feedback loop. A then-popular strategy called **portfolio insurance** instructed institutions to sell stock-index futures automatically as the market fell, to hedge their equity portfolios. As prices dropped, the program-trading systems sold futures; the futures selling dragged the cash market down; the lower cash market triggered more program selling; and the loop accelerated. Compounding it, the market-making and settlement systems of 1987 were overwhelmed — quotes went stale, the futures and cash markets disconnected, and clearinghouses faced members who could not meet their obligations. The crash nearly broke the plumbing entirely; some clearing members came within hours of failing.

The official investigation, the **Brady Commission** report (named for its chair, future Treasury Secretary Nicholas Brady), reached a conclusion that defined market structure for the next four decades. The report argued that the cash market, the futures market, and the options market are not three separate markets but **one market**, linked by arbitrage and by the clearing system — and that a panic in one will transmit instantly to the others. Its central recommendation was a set of **coordinated circuit breakers**: pre-agreed, mechanical pauses that would interrupt the feedback loop, give participants time to assess information, and let the clearing system catch up. The logic was not that a pause changes the fundamentals — it does not — but that a pause breaks the *reflexivity*, the self-reinforcing spiral of selling-begets-selling.

The first circuit breakers were adopted in 1988 under that recommendation, keyed to the Dow and set at coarse point-based thresholds. They were revised repeatedly — after each revision tied to the lesson of the previous stress — until the 2010 Flash Crash forced the rewrite that produced today's percentage-based, S&P-keyed, 7%/13%/20% system. The intuition to carry forward: circuit breakers are not a market-design preference, they are a *fire-suppression system* installed after a specific fire, and every threshold in the current rulebook is the scar tissue of a past crash.

### Who actually writes these rules

This is the part the rest of finance often skips, and it is exactly the series' lens. Circuit breakers and LULD bands are not laws passed by Congress, and they are not decreed by the Federal Reserve. They live in a specific, layered legal structure that determines how fast they can change and who can change them.

At the base is **statute**: the Securities Exchange Act of 1934, which created the SEC and gave it authority over the national market system. The Act itself says nothing about a "7% halt" — it delegates. One rung up is the **regulator**, the SEC, which has rulemaking authority and, crucially, the power to *approve or reject* the rules the exchanges propose. One rung up from that are the **self-regulatory organizations (SROs)** — the exchanges (NYSE, Nasdaq, Cboe, and the rest) and FINRA — which write the actual operational rules. A circuit breaker is, technically, an **exchange rule** (NYSE Rule 7.12 and its siblings) that the SEC approved; the LULD system is a **National Market System Plan** — a joint agreement among all the exchanges, filed with and approved by the SEC.

Why does this plumbing of *rule-making* matter to a trader? Because it tells you how a rule can change and how to see the change coming. A change to a circuit-breaker threshold does not require an act of Congress; it requires the exchanges to file a rule change and the SEC to approve it — a process that runs through a public docket with comment periods. After a major stress event, that docket lights up, and the practitioner who reads it knows the rules are about to move *before* the new thresholds take effect. That is the same docket-reading edge we develop across the series in [how law moves markets: the transmission chain](/blog/trading/law-and-geopolitics/how-law-moves-markets-the-transmission-chain). For now, the takeaway is structural: these kill-switches sit in the SRO-rule / SEC-approval layer, which is faster and more technical than statute, and is where market-structure rules genuinely get made.

### What a "halt" actually is

Start with the word everyone misuses. A **halt** is a temporary, rule-mandated pause in trading. During a halt, you cannot buy or sell the affected security (or, for a market-wide halt, any security) on the exchange. Orders already resting in the book stay there; new orders can be entered and cancelled but will not execute until trading resumes. The key fact a beginner must internalize: a halt is measured in **seconds or minutes**, not days. A single-stock LULD pause is 15 seconds. A market-wide Level 1 or Level 2 halt is 15 minutes. Only the most extreme breaker — Level 3 — closes the market for the rest of the day.

A halt is not the same as a stock being *delisted*, *suspended* by the SEC for fraud (those can last days), or *closed* for a holiday. Those are different legal events. A circuit-breaker halt is a brief, automatic, volatility-driven timeout, after which trading resumes through a structured reopening.

### Market-wide circuit breakers (MWCB)

A **market-wide circuit breaker** halts trading in *all* US-listed equities (and the related options and futures) when the broad market falls by a set percentage from the prior session's close. The benchmark is the **S&P 500 index**. There are three levels, and they are written into the exchange rules (NYSE Rule 7.12, Nasdaq Rule 4121, and parallel rules at every venue), all keyed to the same S&P 500 trigger so no single exchange can be out of step.

The three thresholds and their consequences:

- **Level 1 — S&P 500 down 7%** from the prior close: a 15-minute market-wide halt, *unless* the decline happens at or after 3:25 p.m. ET, in which case trading continues (you are too close to the close to bother).
- **Level 2 — S&P 500 down 13%**: another 15-minute halt, again with the same 3:25 p.m. carve-out. Level 2 can only trigger if Level 1 has already been breached earlier in the day, or if the market gaps straight through 7% to 13%.
- **Level 3 — S&P 500 down 20%**: trading closes for the rest of the day, at any time. There is no reopening. Everyone goes home.

![The market-wide circuit breaker ladder, with deeper declines triggering longer halts and Level 3 closing for the day](/imgs/blogs/circuit-breakers-halts-and-the-legal-plumbing-of-a-crash-2.png)

The percentages reset every day off the prior session's official S&P 500 close. That is a crucial detail for the playbook: the −7% level is not a fixed number of points — it moves with the index. We will compute it precisely in a worked example below.

#### Worked example: the S&P points to each MWCB level

Suppose the S&P 500 closed yesterday at **5,000.00**. Today's circuit-breaker triggers are computed off that number:

- **Level 1 (−7%):** 5,000 × 0.07 = 350 points down. The market halts when the S&P touches 5,000 − 350 = **4,650.00**.
- **Level 2 (−13%):** 5,000 × 0.13 = 650 points down. Trigger at 5,000 − 650 = **4,350.00**.
- **Level 3 (−20%):** 5,000 × 0.20 = 1,000 points down. Trigger at 5,000 − 1,000 = **4,000.00**, and the market closes.

Now suppose the prior close were instead **6,000.00**. The same percentages give: Level 1 at 6,000 − 420 = **5,580.00**, Level 2 at 6,000 − 780 = **5,220.00**, Level 3 at 6,000 − 1,200 = **4,800.00**. Notice the *points* needed grew with the index — a 7% drop is 350 points off 5,000 but 420 points off 6,000 — while the *percentage* trigger is fixed. The intuition: the breaker is a percentage rule, so you must recompute the exact price levels every morning from yesterday's close, because last week's "−7% number" is already wrong.

### Single-stock Limit Up-Limit Down (LULD) bands

The market-wide breakers protect the *index*. But a single stock can blow up on its own — a fat-finger order, a failed algorithm, a halt-the-world headline on one name — without the whole S&P moving. For that, there is a separate, finer-grained mechanism: **Limit Up-Limit Down**, or **LULD**.

LULD works by drawing a moving price corridor around each stock. The system computes a **reference price** (roughly the average trade price over the trailing five minutes) and places two bands around it: an **upper band** a set percentage above and a **lower band** the same percentage below. Trades may not execute *outside* those bands. The band percentage depends on the stock's **tier**:

- **Tier 1** — the most liquid names: the S&P 500 stocks, the Russell 1000 components, and certain high-volume ETFs. Band = **5%** above/below the reference price (for stocks priced above \$3).
- **Tier 2** — every other National Market System stock above \$3. Band = **10%**.
- Lower-priced stocks (\$0.75–\$3.00, and below) get *wider* percentage bands — up to 20% or more — because a few cents is a large percentage of a cheap stock.
- The bands are also *doubled* in the opening and closing periods (the first and last few minutes of the day), when prices are naturally jumpier.

The **reference price** itself is worth understanding, because it is what makes the corridor *move*. It is not yesterday's close or a fixed anchor; it is the arithmetic mean of the eligible trade prices over the trailing five minutes (and at the open, the prior close seeds it). As the stock genuinely trends, the reference price drags the whole corridor along with it: a stock that climbs steadily all morning sees its bands ratchet up too, so LULD never blocks a legitimate trend — it only catches a *sudden* move that outruns the five-minute average. This is the design's elegance and its limit. It permits a stock to fall 30% over an hour without a single pause, as long as it does so smoothly, because the reference price keeps up; it only fires when the move is faster than the corridor can follow. LULD is a brake on *velocity*, not on *distance*.

Here is the subtle part — the **limit state**. When the stock's quote (specifically, the National Best Bid or Offer) hits a band and stays there, trading does not immediately halt. Instead the stock enters a **15-second limit state**: trades can still occur at or inside the band, but not through it. If, within those 15 seconds, the market pulls back inside the band, trading continues normally — no halt at all. Only if the quote is *still* pinned at the band when the 15 seconds expire does the stock go into a formal **5-minute trading pause**, after which it reopens via an auction. That two-step grace period is why the vast majority of limit states you will see on a screen flicker and clear without ever becoming a halt — the system is biased toward letting trading continue.

![A Limit Up-Limit Down price band, with the tradable corridor and the limit state at the band](/imgs/blogs/circuit-breakers-halts-and-the-legal-plumbing-of-a-crash-3.png)

This two-stage design — a soft 15-second limit state before a hard pause — is deliberate. It gives the market a chance to find a clearing price near the band without slamming the brakes. Most limit states resolve without a pause.

#### Worked example: a LULD band width in dollars for two stocks

Take a **Tier 1** stock — say a large-cap trading at a reference price of **\$200.00**. Tier 1 band = 5%.

- Band width = 200 × 0.05 = **\$10.00** each way.
- Upper band = **\$210.00**; lower band = **\$190.00**.
- A buy order at \$211 cannot execute — it is above the upper band. If the offer sits at \$210 for 15 straight seconds, the stock pauses for 5 minutes.

Now take a **Tier 2** stock at a reference price of **\$40.00**. Tier 2 band = 10%.

- Band width = 40 × 0.10 = **\$4.00** each way.
- Upper band = **\$44.00**; lower band = **\$36.00**.

The Tier 2 stock can swing \$4 (10%) before pausing, while the \$200 Tier 1 name can only move \$10 — which is just 5%. The intuition: the corridor is a *percentage* of the stock's own price, scaled by liquidity tier, so a smaller, less-liquid stock is allowed to move a larger fraction before the band catches it. When you watch a thinly traded name "freeze," it is almost always sitting in a 15-second limit state at one of these bands.

### The clearinghouse and the margin call

The third piece of plumbing is invisible on a price screen but is often the real driver of a cascade: the **clearinghouse**. When you buy or sell a stock, the trade does not settle instantly between you and the anonymous counterparty. In the US equities market, a central entity — the **National Securities Clearing Corporation (NSCC)**, part of the **Depository Trust & Clearing Corporation (DTCC)** — steps into the middle of every trade as the buyer to every seller and the seller to every buyer. This is **central clearing**, and the clearinghouse's job is to guarantee that the trade settles even if your counterparty defaults.

To make that guarantee, the clearinghouse demands collateral from its members (the big brokers and banks) — a deposit called **margin**, sized to cover the potential loss on their positions before settlement. The clearinghouse calculates the required margin using a **Value-at-Risk (VaR)** model: roughly, "how much could this member's portfolio lose over the next day or two in a bad scenario?" The headline input to that model is **volatility**. When volatility spikes, the model's estimate of the potential loss spikes too, and the clearinghouse issues a **margin call** — a demand for more cash collateral, often due within hours.

This is the engine of the **reflexive loop** at the heart of a crash, and it deserves its own diagram.

![The margin spiral, where a volatility spike forces a margin call that forces selling that raises volatility again](/imgs/blogs/circuit-breakers-halts-and-the-legal-plumbing-of-a-crash-4.png)

The loop runs like this: prices fall → volatility spikes → the clearinghouse's VaR model demands more margin → members who are short on cash must sell assets to raise it → that selling pushes prices down further and raises volatility again → the next day's margin call is even larger. This is not a bug; it is a direct, mechanical consequence of how risk-based margin works. Circuit breakers and LULD pauses exist in part to interrupt this loop — to buy time for cash to arrive and for the market to find a clearing price before the spiral feeds on itself.

There are actually two margin layers, and conflating them is a common error. **Initial margin** is the collateral the clearinghouse holds up front against the *potential future* loss on a position — the VaR-style number that balloons with volatility. **Variation margin** is the daily (sometimes intraday) settlement of the position's *realized* mark-to-market move: if your position lost \$5 million today, you wire \$5 million in cash to the clearinghouse tonight, full stop. In a crash both fire at once: variation margin demands cash to cover today's loss, *and* initial margin jumps because tomorrow's potential loss looks bigger. A leveraged player can be solvent on paper — owning assets worth more than its debts — and still be forced to sell, simply because it cannot wire the cash by the deadline. That distinction, solvency versus liquidity, is the whole story of why forced selling happens at prices no rational owner would choose: the clearinghouse does not accept "I'm good for it"; it accepts cash, today.

This is also why the clearinghouse is drawn in lavender in our figures — it is a **counterparty/intermediary**, not the asset itself. It is legally neutral; it has no view on whether stocks are cheap. But its rulebook is the most powerful forced-selling engine in the system, because its margin demands are non-negotiable and time-stamped. When you read that a fund "blew up" in a crash, the proximate cause is almost always a margin call it could not meet, not a change in what it thought its assets were worth.

#### Worked example: a clearinghouse VaR margin call as volatility spikes

A clearing member holds a \$1,000,000,000 (one billion dollar) long equity portfolio. The clearinghouse sets margin at a multiple of the portfolio's daily VaR, which scales with volatility. Suppose:

- In calm markets, daily volatility is **1%**, and the clearinghouse requires margin equal to a 3-day, 99%-confidence move. A rough VaR figure is 2.33 (the 99% normal multiplier) × 1% × √3 × \$1bn ≈ **\$40.4 million**.
- A crash hits and daily volatility jumps to **4%** (a 4× increase). The same formula now gives 2.33 × 4% × √3 × \$1bn ≈ **\$161.4 million**.

The required margin has roughly **quadrupled, a jump of about \$121 million in cash**, demanded within hours. If the member does not have \$121 million sitting idle, it must raise it by selling — and the fastest thing to sell is liquid stock, in the very market that is already falling. The intuition: risk-based margin is *pro-cyclical* by construction — it asks for the least collateral when markets are calm and the most exactly when cash is scarcest, which is why a volatility spike can force selling that has nothing to do with anyone's view on value.

### The reopen auction

The last piece of machinery is how trading *resumes* after a pause. You do not simply flip the switch back to continuous trading — that would invite a chaotic rush. Instead, the exchange runs a **reopening auction** (also called a halt-resumption or LULD auction). For a short window, the exchange collects buy and sell orders without matching them, then computes the single price at which the most shares can trade — the price that maximizes executed volume and minimizes the imbalance between buyers and sellers. At that instant, all the collected orders cross at that one price, and continuous trading begins again from there.

The auction is where genuine price discovery happens after a halt. It is also where the gap can be largest: if sellers vastly outnumber buyers, the auction will clear at a much lower price than the last trade before the halt. The closing auction at 4:00 p.m. works the same way and is, on most days, the single largest liquidity event — which is why a forced seller racing a margin deadline often aims for the close.

To make the auction concrete: the exchange's matching engine takes every buy and sell order, and for each candidate price it computes how many shares *would* trade and how large the leftover **imbalance** (unmatched shares) *would* be. It picks the price that maximizes the matched volume; among ties, it picks the one that minimizes the imbalance and sits closest to the reference price. The exchange also publishes **imbalance information** in the moments before the auction — "2 million shares to sell, indicative price \$48" — precisely so that liquidity providers can step in on the other side and dampen the gap. After a halt in a falling market, that imbalance is almost always lopsided to the sell side, and the indicative price walks lower as the auction approaches.

#### Worked example: a reopen auction clearing on a sell imbalance

A Tier 1 stock is paused after pinning its lower LULD band at **\$190** (5% below a \$200 reference). The exchange opens the auction book and collects:

- **Buy interest:** 50,000 shares willing to pay \$188 or better, plus 100,000 shares willing to pay \$185 or better — so 150,000 total shares bid at \$185 and up.
- **Sell interest:** 400,000 shares offered at \$190 or lower (forced sellers and margin liquidations), willing to take whatever clears.

There is a **250,000-share sell imbalance**. The matching engine searches for the price that crosses the most shares. At \$190 only 50,000 buy shares can match (the \$188-or-better bids do not reach \$190 — wait, they pay *up to* \$188, so they will not buy at \$190). Walking the price *down*: at \$185, all 150,000 buy shares are willing and 400,000 sell shares are willing, so **150,000 shares cross at \$185** and 250,000 sell shares go unfilled.

The stock, last seen trading near \$200 before the band, reopens at **\$185 — a 7.5% gap down through the band** — and there is still a 250,000-share overhang pressing on the next few minutes. The intuition: an auction can only clear where buyers actually are, so when forced selling swamps the book, the reopen price is set by how *deep* the buy interest goes, not by the band level — the band paused the stock, but it did nothing to conjure buyers.

### The same crash, four different rulebooks

One more foundation, because it is where most people get blindsided: the equity circuit breakers we have described govern **US-listed stocks during regular trading hours (9:30 a.m.–4:00 p.m. ET)**. The same crash hits at least four different rulebooks, and they do not agree.

- **Equity-index futures (the E-mini S&P 500):** futures trade nearly around the clock and have their own **price limits**, not the equity breakers. Overnight, the futures can fall at most **5% (a "limit down")** from the reference price before trading is restricted to that level — they can trade at or above the limit but not below it. During the cash session, the futures align with the equity 7%/13%/20% breakers. This is why, on a bad-news evening, you will see the S&P futures "locked limit down −5%" hours before the US stock market opens: the futures rulebook caps the *overnight* fall at 5%, and all the pent-up selling waits for the cash open.
- **Single-stock options:** when the underlying stock is halted, its listed options are halted too — you cannot use options to trade around a paused stock. The hedge you were counting on is frozen alongside the thing it was hedging.
- **Foreign markets:** other countries have their own breakers at different levels. China's CSI 300 famously had a 5%/7% breaker that, when introduced in January 2016, *triggered the very panic it was meant to calm* (traders rushed to sell before the 7% close locked them out — a live demonstration of the magnet effect) and was scrapped within four days. Japan, Korea, and India each run their own thresholds. A shock that starts overseas moves through *their* breakers first.
- **Crypto and FX:** spot crypto markets have **no** circuit breakers at all — they trade 24/7 with nothing to pause a cascade — which is precisely why crypto can fall 30% in an hour on a weekend when the equity market is closed and its breakers are irrelevant.

The practical consequence: a shock that lands at 2 a.m. ET moves through futures price limits, foreign breakers, and unbounded crypto markets long before the US equity 7% breaker can engage at 9:30 a.m. By the time the cash market opens, the "news" is already in the futures, and the equity open is often a gap that the breakers then pace from there. Knowing *which* rulebook is live at the moment a shock hits is half of trading it.

## The 2010 Flash Crash: the gap that built LULD

The single-stock LULD system did not exist before 2012. It was built in direct response to one of the strangest days in market history: the **Flash Crash of May 6, 2010**.

That afternoon, beginning around 2:32 p.m. ET, a large automated sell program began dumping E-mini S&P 500 futures contracts at a pace that overwhelmed the available liquidity. As prices fell, high-frequency market makers — who provide most of the bids and offers in normal times — pulled back. With the bids gone, the selling hit air. In a matter of minutes the Dow Jones Industrial Average fell almost 1,000 points (around 9%), the deepest intraday point loss in its history to that date, and then recovered most of it within about twenty minutes.

![The 2010 flash crash timeline, from the sell algorithm through the recovery to the LULD fix](/imgs/blogs/circuit-breakers-halts-and-the-legal-plumbing-of-a-crash-5.png)

The truly bizarre part was at the single-stock level. With no LULD bands in place, individual stocks printed absurd prices. Accenture, a large, healthy company, traded as low as **\$0.01** — one cent — while Sotheby's printed as high as **\$99,999.99**. These were not real valuations; they were the result of the order book emptying out and market orders sweeping whatever stub quotes remained. Roughly 20,000 trades, across more than 300 securities, executed at prices more than 60% away from their pre-crash levels, and the exchanges later cancelled them as "clearly erroneous."

The market-wide circuit breakers that existed in 2010 were keyed off the Dow and set at thresholds (10%, 20%, 30%) that were far too coarse and too slow — they never tripped that day, even as individual stocks went to a penny. The system had a gaping hole: it could protect the index from a 10% move but had nothing to stop a single name from going to zero and back in five minutes.

The regulatory response was swift. The SEC first introduced stock-by-stock circuit breakers, then in 2012 replaced them with the LULD plan we described above — the rolling reference price, the percentage bands, the 15-second limit state. At the same time, the market-wide breakers were overhauled: the trigger was moved from the Dow to the broader S&P 500, the thresholds were tightened to 7%/13%/20%, and the reference was changed to the prior day's close. The crash of 2010 is, quite literally, the reason the modern numbers are what they are. This is the transmission chain in miniature: a market event exposed a gap in the rules, the regulator rewrote the rules, and the new rules now shape every future crash.

## March 2020: the breakers that finally fired

For twenty-three years after the 1987-era circuit breakers were first installed, the market-wide breaker in its modern form never tripped. Then came March 2020.

As the COVID-19 pandemic shut down the global economy, the S&P 500 fell so fast and so often that the Level 1 breaker — the −7% halt — fired on four separate days in two weeks:

- **March 9, 2020:** the S&P opened sharply lower as an oil-price war compounded pandemic fears; the −7% Level 1 breaker tripped minutes into the session.
- **March 12, 2020:** another Level 1 halt as the selling resumed.
- **March 16, 2020:** the third Level 1 halt, this time within the first minute of trading — the day we opened with.
- **March 18, 2020:** a fourth Level 1 halt.

The VIX — the market's "fear gauge," an index of expected 30-day S&P 500 volatility — closed at **82.7** on March 16, 2020, its highest close on record, eclipsing even the 2008 financial crisis. That number is the single best summary of how violent the repricing was.

![VIX close at major equity stress events, with March 2020 the highest at 82.7](/imgs/blogs/circuit-breakers-halts-and-the-legal-plumbing-of-a-crash-6.png)

Two lessons came out of March 2020. First, the Level 1 breaker worked as intended: each 15-minute halt gave the market a pause to absorb information, and on no day did the decline accelerate straight through to Level 2 or Level 3. Several of those mornings the breaker tripped in the **opening seconds** — on March 16 the S&P was down 7% almost immediately, the halt fired, the market sat dark for 15 minutes, reopened through an auction, and then traded the rest of the day still sharply lower. The breaker bought a pause; it did not buy a bounce. Second — and more important for the playbook — the breakers *paced* the decline; they did not prevent it. Over the full month the S&P fell about 34% from its February peak to its March 23 trough. The kill-switches slowed the fall into a series of steps rather than one continuous plunge, but the destination was the same.

It is also worth noting what the breakers did *not* have to do. The Level 2 (−13%) and Level 3 (−20%) breakers never came close to firing, even in the worst month since 2008. The deepest single-day S&P decline that March was about 12% on March 16 — which is why the −13% Level 2 was so nearly tested and the −20% Level 3 stayed comfortably out of reach. The system's deepest machinery, designed for a 1987-scale one-day collapse, sat unused even in a generational crisis. That is the empirical backbone of the misconception we correct below: the headline breakers are far rarer than people assume.

The pandemic crash also showed the margin spiral in action. As volatility exploded, clearinghouses across futures, options, and equities issued enormous margin calls. The Federal Reserve, watching short-term funding markets seize up, ultimately had to intervene massively to keep the plumbing flowing — a reminder that the legal machinery of a crash connects directly to the machinery of monetary policy. We trace that link in [how the Fed sets interest rates](/blog/trading/finance/how-the-fed-sets-interest-rates) and the lender-of-last-resort role in [shadow banking and the repo market](/blog/trading/finance/shadow-banking-and-the-repo-market).

## August 2024: the yen-carry unwind and the halt cascade

A more recent and more localized stress test came on **August 5, 2024**. Over the prior weekend the Bank of Japan had raised rates, and a soft US jobs report stoked recession fears. The combination triggered a violent unwind of the **yen carry trade** — the popular strategy of borrowing cheap yen to buy higher-yielding assets elsewhere. As the yen surged, leveraged players who had borrowed in yen faced losses and margin calls, and they sold everything liquid to cover.

The result was a global volatility shock. Japan's Nikkei 225 fell over 12% in a single session — its worst day since 1987. In the US, the VIX briefly spiked to one of its highest intraday readings ever, closing around **38.6**, and the market opened sharply lower. Crucially, the US did *not* hit a market-wide circuit breaker that morning — the S&P's opening decline, while frightening, stayed inside −7%. But the episode triggered a wave of **single-stock LULD halts** as individual names — especially volatile tech and crypto-linked stocks — gapped at the open and tripped their bands.

August 2024 is a clean case study in how the two layers divide labor. The market-wide breaker is a blunt, rare instrument; it did not fire. The single-stock LULD bands are the fine-grained, common layer; they fired many times, pausing individual names for the 5-minute auction reset while the broad index, though down, never breached the −7% line. Most "halts" you will ever see in your trading life are single-stock LULD pauses, not market-wide breakers.

This is also a textbook example of how a *foreign* policy decision transmits into US market structure — the kind of cross-border shock we map in the [event-trading playbook for central-bank surprises](/blog/trading/event-trading/trading-central-bank-rate-decisions) and the regime analysis in the [cross-asset correlations guide](/blog/trading/cross-asset/correlation-regimes-and-diversification).

## How liquidity vanishes near a band: the magnet debate

Here is the most practically important — and most debated — behavior of these rules. As a price approaches a circuit-breaker level or a LULD band, **liquidity tends to evaporate**. The bid-ask spread widens, the size available at each price thins out, and the order book becomes a ghost of its calm-market self. Why?

The core reason is **option value and adverse selection**. Imagine you are a market maker posting a bid just below the lower LULD band. If the stock is about to pause, two things can happen: either the market recovers (in which case you bought at a decent price), or the stock is genuinely worth far less and is about to gap down hard at the reopen (in which case you just caught a falling knife and will be badly underwater after the auction). The closer the price gets to the band, the more the second scenario dominates — anyone selling to you near the band is more likely to know something you do not. The rational response is to pull your bid. When every market maker reasons this way, the bids vanish exactly when sellers most need them.

There is a long-running academic and practitioner debate about whether circuit breakers and bands act as a **magnet** (also called the "gravitational pull" effect): the theory that, knowing a halt is coming at a specific level, traders rush to execute *before* the band locks them out, which accelerates the move toward the band and makes the halt self-fulfilling. The mechanism is a coordination problem. If I believe the stock is going to pause and reopen lower, my best move is to sell *now*, before the pause traps me at a stale price for five minutes; but if every seller reasons the same way, the collective rush to beat the band is exactly what drives the price into it. The band stops being a speed bump and becomes a destination.

The evidence is genuinely mixed. Some empirical studies of LULD find a measurable magnet effect — order flow tilts toward selling as the price nears the lower band, and the approach speeds up. Others find the bands do their job: they dampen short-term volatility and give the limit state time to resolve without a pause, with most limit states never becoming halts at all. The clearest natural experiment is China's January 2016 breaker, which was so poorly calibrated (a 5% halt followed by a 7% close-for-the-day, with very little distance between them) that it produced a textbook magnet: on January 7, 2016, the market hit the 5% halt, reopened, and within minutes raced to the 7% level and closed for the day after just **29 minutes** of trading. Regulators scrapped it four days after launch. The lesson built into the US design is the *spacing* between levels — 7% to 13% to 20% — chosen specifically so that hitting one level does not mechanically pull the price to the next.

What is not debated is the **liquidity hole**: whether or not the band "pulls" the price, the visible order book thins dramatically as you approach it. In the displayed depth, you can watch the size at each price level shrink and the spread widen in real time as a stock approaches its LULD band — market makers are quietly pulling quotes to avoid being the last bid before a gap. For a trader, the practical takeaway is the same under either theory: **the worst place to be a forced seller is right at a band**, because that is precisely where there is no one to sell to, and the worst place to be a *buyer* is into a band on the way down, because the seller hitting your bid is more likely to know something you do not.

This is the heart of why these rules matter to you even if you never trade through a crash. The bands and breakers do not just stop trading at a level — they reshape the liquidity *around* that level, and that reshaping is where money is lost.

## Common misconceptions

Three beliefs about circuit breakers are widespread, intuitive, and wrong. Correcting them is most of what separates a panicked retail reaction from a composed one.

### Misconception 1: "A halt means trading stopped for the day"

This is the most common and most damaging error. A single-stock LULD pause is **5 minutes**. A market-wide Level 1 or Level 2 halt is **15 minutes**. A 15-second limit state is just that — fifteen seconds. Only **Level 3** — an S&P 500 decline of −20% — closes the market for the day, and as we will see, that has *never* happened intraday under the modern rules.

The numbers matter. On March 16, 2020, the most extreme day, US trading was halted for exactly 15 minutes; the market reopened and traded for the rest of the session. A trader who saw the halt, assumed the market was "closed," and stopped watching would have missed a full day of repricing. A halt is a pause to reset the auction, not a fire alarm that ends the day.

### Misconception 2: "Circuit breakers prevent crashes"

They do not. They **pace** crashes. In March 2020, the breakers fired four times and the S&P still fell about 34% peak-to-trough. The breaker's job is to interrupt a panic for long enough that information can spread, margin cash can arrive, and the next auction can find a real clearing price — not to put a floor under the market. The destination of a crash is set by fundamentals and forced selling; the breakers only control the *path*. Believing a breaker is a floor is how people get caught buying into a pause that is about to reopen another 5% lower.

### Misconception 3: "The −20% breaker fires all the time in bad markets"

It has never fired intraday. The Level 3 (−20%) market-wide breaker has not tripped a single time since the 7%/13%/20% thresholds were adopted after 2012 — not in March 2020, not in any session since. Even the famous −7% Level 1 breaker, in its modern form, fired for the *first time ever* in March 2020 and only on four days that month. The Level 2 (−13%) breaker has *never* fired. The plumbing exists for a tail event so extreme it has not occurred under the current rules. Treating a Level 1 halt as routine, or a Level 3 close as a frequent risk, badly overstates how often the deepest machinery actually engages. (For scale: the 1987 crash that *inspired* circuit breakers was a one-day −22.6% Dow move — exactly the kind of event the −20% level is built for, and which has not recurred.)

### Misconception 4: "A halt protects me if I'm caught in the wrong position"

It does the opposite of what people hope. A halt **freezes** you in your position — you cannot exit during the pause. If you are long into a Level 1 halt, you sit there while the world's bad news keeps arriving, and the market reopens at whatever the auction clears, which is frequently *lower* than the price that triggered the halt. The halt removes your ability to act precisely when you most want it. We quantify this gap risk in a worked example below.

## How it shows up in real markets

We have already walked through the three canonical episodes; here they are side by side as the practitioner's reference set.

- **The 2010 Flash Crash (single-stock failure):** with no LULD in place, individual stocks printed from \$0.01 to \$99,999.99 and roughly 20,000 trades were later cancelled. This is the "what happens with *no* single-stock kill-switch" baseline — and the reason LULD exists.
- **March 2020 (market-wide breakers):** the Level 1 (−7%) breaker fired on four days (March 9, 12, 16, 18), the only times it has ever fired in its modern form. VIX closed at a record 82.7 on March 16. The market still fell ~34% peak-to-trough. This is the "breakers pace, not prevent" baseline.
- **August 2024 (single-stock LULD cascade):** a yen-carry unwind triggered many single-stock LULD halts at the open while the market-wide breaker never fired (S&P stayed inside −7%). VIX spiked to ~38.6. This is the "most halts are single-stock, not market-wide" baseline.

One more piece of context worth holding: these episodes happened against very different interest-rate backdrops. March 2020 hit when the Fed had just slashed rates to near zero; August 2024 came as the Fed sat near the top of a tightening cycle, about to begin cutting. The rate environment shapes how much leverage is in the system and how fast the Fed can respond — which is why the margin spiral was so much more dangerous in some episodes than others.

![Fed funds upper bound across the 2022 to 2024 tightening and easing cycle](/imgs/blogs/circuit-breakers-halts-and-the-legal-plumbing-of-a-crash-8.png)

#### Worked example: gap risk over a halt — being caught long into a Level 1

You hold a **\$100,000** long position in an S&P 500 ETF. The prior close was 5,000; bad news hits and the S&P opens sliding. You decide to sell at −5% (S&P at 4,750), but the market is moving fast and your order sits unfilled. At −7% (S&P at 4,650) the **Level 1 breaker fires** and trading halts for 15 minutes. You are frozen.

- At the moment of the halt (−7%), your position is worth 100,000 × (1 − 0.07) = **\$93,000** on paper — a \$7,000 unrealized loss. But you cannot sell; the market is closed for 15 minutes.
- During the halt, more bad news arrives. The reopen auction clears at **−10%** from the prior close (S&P at 4,500) because sell imbalances dominated the auction.
- The first price at which you can actually exit is −10%: 100,000 × (1 − 0.10) = **\$90,000**. Your realized loss is **\$10,000**, not the \$7,000 you saw when the halt hit, and far more than the \$5,000 you intended to cap it at.

![Gap risk on a long position caught across a Level 1 halt, with the reopen gapping lower](/imgs/blogs/circuit-breakers-halts-and-the-legal-plumbing-of-a-crash-7.png)

The intuition: a halt does not cap your loss — it *delays your exit* until the reopen auction, which can clear materially below the level that triggered the halt. The 15-minute pause is exactly when you most want to act and exactly when you cannot, so the gap between the halt level and the reopen is pure, untradeable risk.

## How to trade it: the playbook

Everything above lands here. You will almost never *cause* a halt, but you will frequently trade in the vicinity of one, and the rules above dictate a handful of concrete behaviors.

### 1. Never chase a price into a band

The single most important rule. As a price approaches a LULD band or a circuit-breaker level, the book thins and the spread blows out (the liquidity-hole / magnet effect). If you send a market order into that zone, you will get an atrocious fill or you will be the one who triggers the limit state. If you must trade near a band, use **limit orders inside the band**, accept that you may not get filled, and never assume the displayed liquidity is real. The corollary: do not place stop-loss orders just outside a band, because the stop will convert to a market order and execute into the liquidity hole at the worst possible price.

### 2. Respect the reopen auction; do not trade the first prints blind

When a stock or the market reopens after a halt, the first prices come out of an auction that may have cleared on a huge imbalance. The opening prints after a halt are noisy and can immediately reverse. The disciplined move is to *let the auction settle* and watch the first minute of continuous trading before acting, rather than hitting the first bid or lifting the first offer. If you are a forced seller, recognize that the reopen auction — and the 4:00 p.m. closing auction — are the deepest liquidity events of the day, and route there rather than into the thin continuous book.

### 3. Treat a halt as gap risk, not protection

As the worked example showed, being caught long into a Level 1 halt freezes you while the news keeps coming. If you carry a position into a high-risk event (a Fed decision, a major data release, a geopolitical flashpoint), size it so that an unfillable gap of 10% does not break you. The halt will not save you; only your *position size and hedges* will. This is why pre-event hedging — buying puts or reducing exposure *before* the catalyst — matters more than any intraday stop. The mechanics of pre-positioning for known events are in the [event-trading guide to scheduled catalysts](/blog/trading/event-trading/trading-scheduled-economic-data-releases).

### 4. Watch the VIX as the margin-spiral signal

The clearest early-warning indicator of the reflexive loop is **realized and implied volatility**, summarized by the VIX. When the VIX gaps from the low teens to the 30s or 40s in a session, you are watching the input that drives clearinghouse VaR models — and therefore the margin calls that force selling. A fast VIX spike means: (a) margin calls are coming, (b) forced sellers will hit the tape over the next day or two regardless of value, and (c) liquidity will stay thin near every band. A VIX of 30+ is the signal to reduce gross exposure and stop providing liquidity into the move, not to "buy the dip" on the first green candle. For the broader volatility-as-a-tradeable-object toolkit, see the [volatility and the VIX deep dive](/blog/trading/quantitative-finance/volatility-vix-and-the-fear-gauge).

### 5. Know the exact levels before the open

Because the breaker triggers reset off the prior close, the −7%/−13%/−20% price levels are different every single day. On a morning that looks risky, compute them (as in our first worked example) and write them down. Know where the index has to go to halt, and know that the closer you get to that level, the worse the liquidity. This turns an abstract rule into three specific, actionable price lines on your chart.

### 6. Know your remedies if you get a bad fill

If you are filled at a clearly absurd price during a dislocation — the 2010 "penny prints" being the extreme case — there is a rule for that too. The exchanges maintain **clearly erroneous execution (CEE)** rules and, post-2010, numeric guidelines that let a trade be busted (cancelled) if it executes more than a defined percentage away from the prevailing price. After the Flash Crash, roughly 20,000 trades were cancelled under these rules. The practical points: (a) a bust is *not* guaranteed and the thresholds are wide, so do not rely on it; (b) the window to request a review is short — typically you must flag it to the exchange within 30 minutes; and (c) once LULD is doing its job, genuinely erroneous prints are far rarer, because the band stops the trade from happening in the first place. Treat the CEE rule as a backstop of last resort, not a safety net you can lean on.

There is also a feed worth watching directly. Every halt — single-stock and market-wide — is published in real time on the consolidated tape and the exchanges' halt feeds, with a reason code (LUDP for a LULD pause, "M" codes for market-wide, "T1/T2" for news pending). A serious desk has this feed on a screen, because a single-stock halt in a name you hold, or in a bellwether, is information: it tells you a band was hit, a 5-minute auction is coming, and the reopen could gap. Reacting to the *halt itself* — rather than waiting to be surprised by the reopen — is a small, concrete edge that costs nothing but attention.

### What invalidates the framework

This playbook assumes the rules stay as described. It is *invalidated* — or at least must be re-derived — in a few cases. First, the percentages and durations are set by exchange rules approved by the SEC and **can be changed**; after any major event, expect a review (as happened after 1987, 2010, and the 2010s reforms). Second, the rules apply to **US-listed equities during regular hours**; futures, FX, crypto, and overnight/pre-market sessions have *different* (and sometimes no) circuit-breaker regimes, so a shock that hits overnight can move a long way before the cash market's breakers ever engage. Third, the framework assumes the clearinghouse and exchanges function normally; a true operational failure (an exchange outage, a clearinghouse member default) is a different and more dangerous scenario governed by separate default-management rules. Know which regime you are actually trading in before you rely on a band to protect you.

The deepest point is the one the series keeps returning to: a rule is not a footnote to the price — sometimes the rule *is* the price. The −7% line is both a legal threshold and a tradeable level, the LULD band is both a regulation and a place where liquidity dies, and the margin formula is both a risk-management tool and the mechanism that turns a dip into a cascade. Read the plumbing, and a crash stops being chaos and becomes a sequence of rules you can anticipate.

## Further reading & cross-links

**Within this series:**

- [How law moves markets: the transmission chain](/blog/trading/law-and-geopolitics/how-law-moves-markets-the-transmission-chain) — the master mental model this post is an instance of: a rule changes the game, markets price the consequence.
- [Market structure law: Reg NMS, PFOF, and short-selling rules](/blog/trading/law-and-geopolitics/market-structure-law-reg-nms-pfof-and-short-selling-rules) — the broader rulebook that governs how orders route, who provides liquidity, and how the 2021 GameStop episode tested it.
- [The law, policy, and geopolitics playbook](/blog/trading/law-and-geopolitics/the-law-policy-and-geopolitics-playbook) — the capstone that ties every rule-driven trade together.

**Cross-links out (the mechanisms, not re-derived here):**

- [Trading central-bank rate decisions](/blog/trading/event-trading/trading-central-bank-rate-decisions) and [trading scheduled economic data releases](/blog/trading/event-trading/trading-scheduled-economic-data-releases) — how to position before the catalysts that trigger the gaps and halts.
- [Correlation regimes and diversification](/blog/trading/cross-asset/correlation-regimes-and-diversification) — why a yen-carry unwind or a vol spike makes everything fall together.
- [How the Fed sets interest rates](/blog/trading/finance/how-the-fed-sets-interest-rates) and [shadow banking and the repo market](/blog/trading/finance/shadow-banking-and-the-repo-market) — the lender-of-last-resort plumbing that backstops the clearing system when the margin spiral runs.
- [Volatility, the VIX, and the fear gauge](/blog/trading/quantitative-finance/volatility-vix-and-the-fear-gauge) — the volatility toolkit behind the margin-spiral signal in the playbook.
