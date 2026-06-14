---
title: "Backtesting done right: transaction costs, slippage, and point-in-time data"
date: "2026-06-14"
publishDate: "2026-06-14"
description: "A from-scratch guide to honest backtesting for quant researcher interviews and take-home cases: why a backtest is a simulation that usually lies, how look-ahead bias inflates a Sharpe of 0.4 into a fake 2.0, how to model commissions, the half-spread, and square-root market impact in basis points and dollars, how turnover drags down net returns, how to size a strategy to its capacity, and why point-in-time data is the number-one killer -- with twelve figures and a dozen fully worked dollar examples."
tags:
  [
    "backtesting",
    "transaction-costs",
    "slippage",
    "market-impact",
    "point-in-time-data",
    "look-ahead-bias",
    "quant-interviews",
    "quantitative-research",
    "turnover",
    "strategy-capacity",
    "sharpe-ratio",
    "trading-simulation",
  ]
category: "trading"
subcategory: "Quantitative Finance"
author: "Hiep Tran"
featured: true
readTime: 44
---

> [!important]
> **TL;DR** — A backtest is a simulation of a trading strategy run over historical data, and most backtests lie; getting the timing, the transaction costs, and the point-in-time data right is what separates a real edge from a fantasy curve.
>
> - **A backtest is just a loop over time.** At each bar you compute a signal from data you already had, decide a position, execute it at a realistic price, and tally profit and loss (P&L). Every honest backtest is this loop done carefully.
> - **The single most dangerous bug is look-ahead bias** — using information you would not actually have had at the moment you traded. A one-line mistake (trade at the same close you used to make the decision) can turn a true *Sharpe ratio* of 0.4 into a fake 2.0.
> - **Costs are not a footnote; they are the result.** A round trip (buy then sell) on a typical liquid US stock costs roughly **11 basis points** — commission plus the half-spread plus market impact. On a \$100,000,000 book that is real money, and at high turnover it eats the whole edge.
> - **Market impact follows a square-root law:** the cost of an order grows with the square root of its size relative to daily volume. This is what sets a strategy's *capacity* — the dollar size past which impact eats the alpha.
> - **The number to remember:** 1 basis point (bp) = 0.01% = \$1 per \$10,000 traded. A 200%-per-month turnover strategy paying 11 bps round-trip bleeds **5.28% a year** in costs before it earns a cent.
> - **Point-in-time data is the number-one killer.** A database that quietly drops dead companies, uses revised financials, or knows index membership before it was public will hand you alpha that never existed.

Two researchers backtest the *same* trading idea on the *same* ten years of data. One reports a Sharpe ratio of 2.0 and a smooth, climbing equity curve that looks like a staircase to heaven. The other reports a Sharpe of 0.4 and a curve that wobbles upward like a drunk walking home. They did not use different data, and neither of them lied. The difference is entirely in *how they ran the simulation* — what prices they assumed they traded at, what costs they charged themselves, and what information they let the strategy peek at. One of them ran an honest simulation. The other ran a fantasy.

![The gross backtest curve ignores costs and lies, while the net curve after costs tells the truth: trading at perfect prices with no commissions, spread, or impact produces a Sharpe near 2.0, but trading at the next open with realistic costs collapses it to near 0.4.](/imgs/blogs/backtesting-done-right-quant-research-1.png)

The diagram above is the mental model for this whole post. A backtest is a *simulation*, and a simulation is only as honest as its assumptions. The left column is the fantasy: trade at perfect prices, pay nothing, move the market not at all. The right column is the truth: trade at the next available price, pay the commission and the spread and the impact, and watch the beautiful curve flatten into something you might actually be able to trade. This is the most important and least glamorous skill in quantitative research. In a *quant researcher* interview — at firms like Two Sigma, Citadel, DE Shaw, or AQR — and especially in a take-home case, nobody will be impressed that you found a signal. They will probe, relentlessly, whether your backtest was honest. This article teaches you to make it honest, and to spot when it is not.

A quick note before we start: this is an educational walkthrough of how trading simulations work, not financial advice. None of the strategies here is a recommendation to buy or sell anything. The goal is to make you fluent in the mechanics so that when you see a backtest — yours or someone else's — you can tell whether it is real.

## Foundations: what a backtest actually is

Let us build every piece from zero, because the whole post depends on getting the vocabulary exactly right.

A **backtest** is a computer simulation that takes a *trading strategy* and runs it over *historical market data* to estimate how it would have performed. You feed it prices from the past, it pretends to make trading decisions as if it were living through that history, and it reports the profit and loss it would have earned. The word "backtest" just means "test it backward, on data we already have."

A **strategy** is a rule that turns information into positions. A **position** is how much of something you hold: long 1,000 shares of Apple, short \$2,000,000 of crude oil futures, flat (holding nothing). A **signal** is the number the strategy computes that drives the position — for example, "the 10-day average price minus the 50-day average price," which is positive when a stock is trending up. The strategy maps the signal to a *target position*: maybe "be long when the signal is positive, short when it is negative."

A **bar** is one row of historical data — typically one day, but it could be one minute or one hour. A daily bar usually has an **open** (the first traded price of the day), a **high**, a **low**, and a **close** (the last traded price). When we say "trade at the close" we mean we assume our order filled at that day's closing price.

**Profit and loss (P&L)** is the money the strategy makes or loses. If you are long 100 shares and the price rises \$1, you made \$100. If you are short and it rises, you lose. **Mark-to-market** means re-valuing your position at the current price every bar, even if you have not sold, so the equity curve reflects paper gains and losses as they happen.

The **equity curve** is the running total of your account value over time — the headline picture of a backtest. A good-looking equity curve climbs steadily; a bad one is flat or falling. But as we will see, a good-looking equity curve is the *easiest thing in the world to fake*, which is exactly why interviewers do not trust it on its own.

Finally, two summary numbers you must know cold:

- **Return** is the percentage gain over a period. If \$100 grows to \$110, the return is 10%.
- The **Sharpe ratio** is the average return divided by its standard deviation, usually annualized. It measures return *per unit of risk*. A *standard deviation* is just a measure of how much the returns bounce around: high standard deviation means a wild ride, low means a smooth one. A Sharpe of 1.0 is decent; 2.0 is excellent; above 3.0 in a backtest usually means you have a bug. We will compute Sharpe explicitly later, but the intuition is: it rewards smooth, consistent gains and punishes volatile ones.

With that vocabulary, we can state the thesis precisely. A backtest estimates a strategy's true performance, and that estimate is biased upward by three things: **trading at prices you could not have gotten**, **ignoring the costs of trading**, and **using information you did not have**. Fix those three, and the backtest tells the truth. Leave them in, and it lies in your favor every time.

### The event loop

The cleanest way to run an honest backtest is an **event-driven loop**: you walk forward through time, one bar at a time, and at each bar you may only use information that had already arrived by then. This discipline is the entire game.

![The backtest event loop advances one bar at a time: a new bar arrives, you update data up to that bar only, compute a signal from data through today, decide a target position, execute it at the next bar's open plus cost, then mark P&L and advance the clock.](/imgs/blogs/backtesting-done-right-quant-research-2.png)

Read the loop in the figure carefully, because the ordering is the safeguard. A new bar arrives — say, today's close. You update your data, but *only up to today*. You compute the signal from data through today. You decide what position you want. Then — and this is the crucial step — you **execute at the next bar's open**, not at today's close, and you charge yourself the cost of that trade. Finally you mark your P&L and advance the clock to the next bar.

Why execute at the next open rather than today's close? Because by the time today's close has printed, the market is *closed*. You computed your signal using the close price; you physically cannot also have traded at that same close, because you only knew it the instant the market shut. The earliest you can act is the next morning's open. Skipping this one-bar gap is the most common way a backtest cheats, and it is the first thing a sharp interviewer will check.

Here is the loop as a few lines of Python, using `pandas`, so the abstraction is concrete:

```python
import numpy as np
import pandas as pd

def event_driven_backtest(prices: pd.DataFrame, signal_fn, cost_bps=11.0):
    """prices has columns ['open', 'close']; signal_fn(history) -> target weight in [-1, 1]."""
    cash, position, equity = 1.0, 0.0, []
    prev_weight = 0.0
    for t in range(1, len(prices)):
        history = prices.iloc[:t]            # data available *through* bar t-1
        target_w = signal_fn(history)        # signal computed only on the past
        # We decided at the close of t-1; we trade at the OPEN of bar t.
        fill_price = prices['open'].iloc[t]
        turnover = abs(target_w - prev_weight)
        cost = turnover * cost_bps / 1e4     # round-trip cost charged on the traded fraction
        # Mark to market on this bar's return, net of the cost we just paid.
        bar_ret = prices['close'].iloc[t] / prices['open'].iloc[t] - 1.0
        equity.append(target_w * bar_ret - cost)
        prev_weight = target_w
    return pd.Series(equity, index=prices.index[1:])
```

Notice three things. First, `history = prices.iloc[:t]` — the signal sees data only *up to* bar `t-1`, never bar `t`. Second, the fill happens at `prices['open'].iloc[t]` — the next bar's open. Third, every change in position pays a `cost`. Those three lines are the difference between honesty and fantasy. We will spend the rest of the article unpacking each one.

### Vectorized versus event-driven

There is a faster way to backtest, and you should know when to use it and when to distrust it. A **vectorized backtest** computes the whole thing as array math: line up the signal as one column, shift it down by one row to enforce the one-bar lag, multiply by the next period's returns, and sum. No loop. On a laptop, a vectorized backtest of a daily strategy over twenty years runs in milliseconds; an event-driven loop over the same data might take seconds. When you are screening hundreds of ideas, that speed difference matters enormously.

![Vectorized backtests are fast and great for screening many ideas, but only an event-driven loop reliably enforces correct timing and path-dependent costs like market impact.](/imgs/blogs/backtesting-done-right-quant-research-4.png)

The matrix above is the tradeoff. Vectorized is fast but *leaky*: it is dangerously easy to forget the `.shift(1)` that enforces the lag, and it cannot naturally model costs that depend on the path you took (impact that grows when you trade more, fills that depend on whether the order was big relative to that day's volume). Event-driven is slow but *strict*: the loop structure forces you to confront, at every bar, exactly what you knew and exactly what you paid. The professional workflow uses both: vectorize to screen thousands of candidate signals quickly, then re-run the handful of survivors through a careful event-driven simulation that you actually trust. If a signal looks great vectorized but dies event-driven, the event-driven number is the truth and the vectorized one had a bug.

## Look-ahead and timing: no peeking

**Look-ahead bias** is using information in your simulation that you would not have had at the moment you made the decision. It is the deadliest backtest error because it is invisible in the results — the equity curve looks fantastic, precisely *because* the strategy is cheating. There is no warning light. You have to reason about the timing yourself.

The canonical form is the one we already named: compute a signal from today's close, then trade at *today's close*. That sounds innocent, but it is impossible. The close is the last price of the day. The moment you know it, trading for the day is over. If your simulation fills your order at that same close, it is letting you trade on information that arrived at the exact instant the window slammed shut. In live trading you would have to wait until the next session.

![The only honest rule for daily signals: a signal computed from today's close can only be executed at the next bar's open, never at today's close, because by the time the close prints the market is shut.](/imgs/blogs/backtesting-done-right-quant-research-3.png)

The timeline above is the rule made visual. Day *t* closes at 16:00; prices are now final. After the close you compute your signal. The order then sits overnight — you cannot do anything until the market reopens. At 09:30 the next morning, day *t+1*, you finally get filled, at the open price, plus slippage. Your position then earns whatever return happens over day *t+1*. Every honest daily backtest respects this gap. The signal uses information through the close of *t*; the very first return you are allowed to capture is the return *from* the open of *t+1* onward.

#### Worked example: a look-ahead bug that turns a Sharpe of 0.4 into 2.0

Let us make the damage concrete with real numbers. Suppose you have a mean-reversion signal on a stock: when the stock closes down a lot on a day, you buy, betting it bounces back. The honest version says: observe today's close, buy at *tomorrow's* open, and earn tomorrow's open-to-close return. Run that, and you get a *Sharpe ratio of 0.4* — a marginal but real edge, hit rate around 52% (you are right slightly more than half the time).

Now introduce a one-line bug. Instead of buying at tomorrow's open, you buy at *today's* close — the same close you used to spot that the stock fell. Suddenly your simulation captures the move that already happened. A stock that closed down sharply often *did* the worst of its falling near the close, so "buying at the close after a big down day" mechanically catches the snap-back that already started. Your hit rate jumps to an impossible 68%, and your Sharpe inflates to **2.0**.

![A one-line look-ahead bug -- trading at the same close used to make the decision -- inflates the hit rate from a realistic 52% to an impossible 68% and the Sharpe ratio from a true 0.4 to a fake 2.0.](/imgs/blogs/backtesting-done-right-quant-research-11.png)

The before/after figure shows the inflation. The left column is the bug: signal uses `close[t]`, trade at `close[t]` too, captures the same-bar move, fake Sharpe 2.0. The right column is the fix: signal uses `close[t]`, but trade at `open[t+1]`, earn the next bar's return, true Sharpe 0.4. Same strategy. Same data. A factor-of-five difference in the headline number, entirely from one timing mistake.

Let us actually compute a Sharpe to ground the intuition. Say the honest strategy's daily returns have a mean of 0.02% (2 bps) per day and a daily standard deviation of 0.8% (80 bps). There are about 252 trading days in a year. The annualized Sharpe is:

$$\text{Sharpe} = \frac{\bar{r}}{\sigma_r}\sqrt{252} = \frac{0.0002}{0.008}\times\sqrt{252} = 0.025 \times 15.87 \approx 0.40$$

Here $\bar{r}$ is the mean daily return, $\sigma_r$ is the daily standard deviation, and $\sqrt{252}$ annualizes it (returns scale with time but standard deviation scales with the square root of time, so the ratio scales with $\sqrt{252}$). Now the buggy version, by capturing the same-bar bounce, lifts the mean daily return to about 0.10% (10 bps) while the standard deviation barely changes:

$$\text{Sharpe}_{\text{bug}} = \frac{0.0010}{0.008}\times\sqrt{252} = 0.125\times15.87 \approx 1.98 \approx 2.0$$

The single sentence to take away: **look-ahead bias does not add noise, it adds free money — and free money is the surest sign your simulation is cheating.** A backtest Sharpe above 2 or 3 should make you *more* suspicious, not more excited.

### Other flavors of peeking

Same-bar fills are the most common look-ahead bug, but they are not the only one. Watch for these too:

- **Survivorship in the universe.** If you build your tradable universe today and then run the backtest on it, you have only included companies that survived. We will return to this under point-in-time data, because it is enormous.
- **Using the full-sample mean or volatility to normalize a signal.** If you z-score a signal by subtracting its mean and dividing by its standard deviation computed over the *whole* history, you have leaked the future into the past. You must use only data available up to each bar (an *expanding* or *rolling* window).
- **Forward-filled fundamentals stamped with the wrong date.** A company reports Q1 earnings, but the report is not public until six weeks after the quarter ends. If your data stamps the earnings on the quarter-end date, you are trading on numbers nobody had yet.
- **Restated data.** Vendors often overwrite a company's old reported numbers with later, revised figures. Backtesting on revised numbers means trading on a version of reality that did not exist at the time.

The discipline is always the same question: *at the instant I am pretending to trade, did I actually have this number?* If the answer is no, or "I am not sure," you have a look-ahead bug.

## Transaction costs: what trading actually costs

Now we turn to the second source of inflation: pretending trading is free. It is not. Every time you trade, you pay, and the payment comes in three layers stacked on top of each other.

![Every round-trip trade pays a stack of three costs on top of the mid price: commission, the half-spread, and market impact -- here 0.5 + 2 + 3 = 5.5 bps per side, or 11 bps for the round trip, which is $1,100 on a $1,000,000 trade.](/imgs/blogs/backtesting-done-right-quant-research-5.png)

The stack in the figure is the anatomy of a trade's cost. Start at the bottom with the **mid price** — the fair value, halfway between the best price you can buy at and the best price you can sell at. That is the price you would *love* to trade at, and almost never do. On top of it, three costs:

**Commission.** This is the fee the broker or exchange charges. For institutional equity trading it is small and explicit — call it **0.5 basis points** per side in our example, though for a large fund trading liquid names it can be even less. A *basis point* (bp) is one hundredth of a percent: 0.01%, or \$1 for every \$10,000 you trade. Commissions are the easy part: they are a known number, charged per share or per dollar.

**The half-spread.** Quotes come in pairs: the **bid** (the highest price someone will pay you for the stock) and the **ask** or **offer** (the lowest price someone will sell it to you for). The gap between them is the **bid-ask spread**. If you want to buy *right now*, you pay the ask; if you want to sell *right now*, you receive the bid. Either way you cross half the spread away from the mid. So a market order pays the **half-spread** — here **2 bps** per side. The spread exists because the people quoting prices (market makers) need to be paid for providing liquidity and bearing risk. Crossing it is the price of immediacy.

**Market impact.** This is the subtle, expensive one. When you buy, you push the price *up*; when you sell, you push it *down*. A small order barely moves the price. A large order, relative to how much normally trades, moves it a lot, because you exhaust the people willing to sell at the best price and have to reach to worse ones. In our example we charge **3 bps** per side, but unlike the other two, impact *grows with the size of your order*. It is the cost that determines how big a strategy can get, and we will give it its own section.

#### Worked example: gross versus net P&L on a \$100,000,000 book

Let us put dollars on the stack. You run a strategy on a **\$100,000,000 book** (a "book" is just the pool of capital the strategy trades). The gross backtest — costs ignored — says the strategy earns **8% a year**, or **\$8,000,000**. Beautiful. Now let us charge the costs.

First, the per-trade cost. One *side* of a trade (just the buy, or just the sell) costs commission plus half-spread plus impact: 0.5 + 2 + 3 = **5.5 bps**. A **round trip** — buying and later selling, the full cycle of a position — costs twice that, **11 bps**. In dollars, on a single \$1,000,000 trade, a round trip costs:

$$11 \text{ bps} \times \$1{,}000{,}000 = 0.0011 \times \$1{,}000{,}000 = \$1{,}100$$

Now, how much does the strategy trade in a year? Say it has **100% annual turnover**, meaning over a year it replaces its entire \$100,000,000 of positions once. (We will define turnover carefully in the next section.) Replacing the book once is \$100,000,000 of round-trip trading. The annual cost is:

$$11 \text{ bps} \times \$100{,}000{,}000 = 0.0011 \times \$100{,}000{,}000 = \$110{,}000$$

So the net P&L is \$8,000,000 − \$110,000 = **\$7,890,000**, or 7.89% — barely dented. At 100% turnover, costs are a rounding error and the gross curve is nearly honest.

But now suppose the strategy is faster and turns over **2,000% a year** (it replaces its positions twenty times over — perfectly normal for a short-horizon stat-arb strategy). Now the annual trading is \$2,000,000,000 of round trips, and the cost is:

$$11 \text{ bps} \times \$2{,}000{,}000{,}000 = 0.0011 \times \$2{,}000{,}000{,}000 = \$2{,}200{,}000$$

Net P&L: \$8,000,000 − \$2,200,000 = **\$5,800,000**, or 5.8%. The costs just ate over a quarter of the return. And if the gross edge had been a more typical 4% (\$4,000,000), the same costs would more than halve it. This is the single most important sentence in the whole post: **the faster you trade, the more of your gross edge the costs consume, and at high enough turnover they consume all of it.**

![Same signal and same $100M book: the gross equity curve compounds to about $128M over five years while the 11-bps round-trip cost drags the net curve down to roughly $108M -- the widening gap is the cumulative cost you never see in the gross plot.](/imgs/blogs/backtesting-done-right-quant-research-10.png)

The gross-versus-net equity curve drives it home. Both curves start at \$100,000,000. The solid gross curve climbs to \$128,000,000 over five years; the dashed net curve, paying 11 bps every round trip, limps to about \$108,000,000. The gap between them *is* the cumulative cost drag, and it widens every year because the costs compound away from you just as the gains compound for you. A backtest that shows you only the gross curve is showing you the staircase to heaven and hiding the trapdoor.

## Slippage models: from a constant to the square-root law

**Slippage** is the difference between the price you assumed you would trade at and the price you actually got. It bundles together the half-spread and the impact — everything that makes your fill worse than the mid. How you model slippage in a backtest is one of the biggest levers on whether the result is honest, and there is a ladder of models from crude to realistic.

![Three slippage models in increasing realism: a fixed constant cost ignores size and liquidity, spread-proportional cost is decent for small orders, and the square-root impact model is what desks actually use for sizing -- though it needs a calibrated constant.](/imgs/blogs/backtesting-done-right-quant-research-7.png)

The matrix lays out the three rungs.

**Fixed slippage.** The crudest model: assume every trade costs a constant, say 5 bps, no matter what. It is a placeholder you use when you have nothing better. Its fatal flaw is that it ignores order size and liquidity entirely — it charges the same 5 bps whether you trade \$10,000 or \$10,000,000 of a stock, which is absurd, because the \$10,000,000 order moves the price and the \$10,000 one does not.

**Spread-proportional slippage.** A real improvement: assume your slippage is some multiple of the half-spread, say `cost = k × half-spread`. A patient algorithm that works the order over time might achieve `k ≈ 0.5` (half the half-spread); an aggressive one that crosses immediately pays `k ≈ 1.0` or more. This is decent for *small* orders, where the spread dominates and impact is negligible. It still misses the impact of large orders.

**Square-root impact.** The model desks actually use to size strategies. It says the impact cost of an order grows with the *square root* of the order's size relative to the average daily volume traded:

$$\text{impact (bps)} = a \times \sqrt{\frac{Q}{V}}$$

Here $Q$ is the size of your order (in shares or dollars), $V$ is the **average daily volume** (ADV) — how much of that stock trades on a typical day — and $a$ is a constant you calibrate from your own trading data (often somewhere around 8 to 12 bps for liquid equities when $Q/V$ is measured as a fraction). The square root is the key. It says impact is *concave* in size: doubling your order does not double the cost.

![Market impact in basis points scales with the square root of order size as a fraction of daily volume: trading 1% of daily volume costs about 10 bps, 4% costs about 20 bps, 9% costs about 30 bps -- four times the size only doubles the cost.](/imgs/blogs/backtesting-done-right-quant-research-6.png)

The square-root curve above is worth staring at. Trading **1%** of a day's volume costs about **10 bps**. Trading **4%** — four times as much — costs about **20 bps**, only *twice* as much, because $\sqrt{4} = 2$. Trading **9%** costs about **30 bps** ($\sqrt{9} = 3$). The cost rises, but it rises slower than the size, which is what makes large institutions possible at all — if impact were *linear* in size, no large fund could trade. The flip side: because the curve keeps rising, there is always a size at which the impact swamps your edge. That size is your capacity.

#### Worked example: the slippage on a real order

You manage a portfolio and want to buy **\$5,000,000** of a stock whose average daily volume is **\$200,000,000**. Your order is a fraction of the day's volume:

$$\frac{Q}{V} = \frac{\$5{,}000{,}000}{\$200{,}000{,}000} = 0.025 = 2.5\%$$

Using a square-root model calibrated with $a = 10$ bps at 1% of ADV, the impact is:

$$\text{impact} = 10 \text{ bps} \times \sqrt{\frac{2.5\%}{1\%}} = 10 \times \sqrt{2.5} = 10 \times 1.58 = 15.8 \text{ bps}$$

In dollars, that one-side impact on the \$5,000,000 order is:

$$15.8 \text{ bps} \times \$5{,}000{,}000 = 0.00158 \times \$5{,}000{,}000 = \$7{,}900$$

Add the half-spread (say 2 bps = \$1,000) and commission (0.5 bps = \$250) and the full cost of getting into the position is about **\$9,150**, or 18.3 bps. If you later sell the same position back, you pay the round-trip again. The single sentence: **the impact term dominates the cost of a large order, and because it grows with the square root of size, the only way to shrink it is to trade smaller or slower.**

Here is the square-root model as a tiny function, so you can drop it into a backtest:

```python
def slippage_bps(order_dollars, adv_dollars, a=10.0, half_spread_bps=2.0, commission_bps=0.5):
    """One-side cost in basis points for a market order, square-root impact model."""
    participation = order_dollars / adv_dollars          # fraction of average daily volume
    impact = a * (participation / 0.01) ** 0.5            # a calibrated at 1% of ADV
    return impact + half_spread_bps + commission_bps
```

A backtest that calls something like this on every fill — charging more for bigger orders in thinner names — is dramatically more honest than one that charges a flat 5 bps. And critically, it *automatically* punishes a strategy for trying to scale beyond what the market can absorb, which is exactly the behavior you want a simulation to have.

## Turnover and its cost drag

We have used the word **turnover** loosely; let us pin it down, because it is the multiplier that turns a small per-trade cost into a large annual drag. Turnover is the rate at which a strategy replaces its positions. If you hold \$100,000,000 and over a month you buy and sell \$200,000,000 of stock (in one direction — the one-way notion), your monthly turnover is **200%**: you churned through your book twice.

Why does turnover matter so much? Because **your total cost is your per-trade cost times how much you trade**, and turnover is how much you trade. A brilliant signal that requires you to flip your entire portfolio every single day can be completely destroyed by costs, while a mediocre signal you only act on once a quarter can survive easily. Turnover is the bridge between the per-trade cost (a property of the market) and the annual drag (the thing that actually shrinks your return).

![Annual cost drag equals the round-trip cost times annual turnover: at 11 bps round-trip, 25% monthly turnover drags 0.66% per year, 50% drags 1.32%, 100% drags 2.64%, and 200% monthly turnover drags 5.28% a year.](/imgs/blogs/backtesting-done-right-quant-research-8.png)

The bar chart shows the drag climbing linearly with turnover. The arithmetic behind it: if you turn over $\tau$ of your book one-way per month, then over a month you do $\tau$ worth of one-way trading, but each position you change is part of a round trip (you bought it, you will sell it), so the round-trip cost applies to $2\tau$ per month, and there are 12 months in a year. With an 11-bps round-trip cost:

$$\text{annual drag} = 11 \text{ bps} \times 12 \times (2\tau)$$

For 200% monthly turnover ($\tau = 2.0$), that is $0.11\% \times 12 \times 4 = 0.11\% \times 48 = 5.28\%$ per year. (Conventions for counting turnover and round trips vary across desks; what never varies is that drag scales linearly with how much you trade.)

#### Worked example: the annual cost drag of 200% monthly turnover

Let us compute it slowly, in dollars, on the \$100,000,000 book. The strategy turns over **200% of the book per month, one-way**. That means each month it does \$200,000,000 of buying-or-selling in one direction. Since every position it opens it eventually closes, the round-trip cost of 11 bps applies. Over a year, the one-way trading is:

$$200\% \times \$100{,}000{,}000 \times 12 \text{ months} = \$2{,}400{,}000{,}000 \text{ one-way}$$

That one-way figure already represents both legs over time when we apply the round-trip rate to it carefully, but the cleanest way to see it is: each month, \$200,000,000 of positions are *both* established and later unwound, so the round-trip cost hits \$200,000,000 each month. Over twelve months:

$$\text{annual cost} = 11 \text{ bps} \times \$200{,}000{,}000 \times 12 = 0.0011 \times \$200{,}000{,}000 \times 12 = \$2{,}640{,}000$$

So this strategy bleeds **\$2,640,000 a year**, which on a \$100,000,000 book is exactly the **5.28%** the chart shows. If the gross edge is 8%, the net is 2.72% — costs ate two-thirds of it. If the gross edge had been 5%, the net would be a negative −0.28%: **a profitable-looking strategy that actually loses money once you trade it.** The single sentence: **turnover is a tax rate on your gross edge, and at high turnover the tax can exceed 100%.**

This is why, in real research, you do not chase the highest-Sharpe gross signal. You chase the highest *net* Sharpe, and net Sharpe falls as turnover rises. A common and powerful trick is to *slow the signal down* — trade only when the signal is strong enough to be worth the cost, which mechanically cuts turnover and can raise net returns even as it lowers gross returns. We will see desks doing exactly this in the real-research section.

## Capacity: how many dollars before impact eats the alpha

We have all the pieces now to answer the question every allocator and every interviewer eventually asks: **how big can this strategy get?** This is *capacity* — the amount of money you can run before market impact eats the edge. It is the most important number for turning a backtest into a business, and it falls straight out of the square-root impact law.

The logic: your gross dollar profit grows roughly *linearly* with the assets you deploy — twice the money, twice the gross profit, because you take twice the position in every name. But your impact cost grows faster than linearly with the size of each trade, because bigger trades move the price more (the square-root law applied to ever-larger orders). At some point, the extra impact from deploying one more dollar exceeds the extra gross profit that dollar earns. Past that point, adding money *reduces* your total net profit. That turning point is your capacity.

![Net annual dollar profit rises with assets then falls as square-root impact costs overtake the gross edge: the curve peaks at a capacity of roughly $500M, beyond which deploying more money reduces total net profit.](/imgs/blogs/backtesting-done-right-quant-research-9.png)

The capacity curve above is the shape every strategy has. On the left, in the green region, each additional dollar still earns positive net alpha — impact is small relative to the edge, so growing the book grows the profit. The curve rises. At the peak — here around **\$500,000,000** of assets under management — net profit is maximized. To the right, in the red region, impact has overtaken the gross edge: each additional dollar now *costs* more in extra impact than it earns, so total net profit falls. Push the book far enough and net profit goes to zero and then negative. The strategy still "works" in the sense of having a real signal, but the market is too small to hold the money you want to put in it.

#### Worked example: sizing a strategy to its capacity

Let us size a concrete strategy. Suppose your signal predicts a gross edge of **20 bps** per round trip on each trade, before costs — that is, each time you cycle a position, the signal earns you 20 bps gross. Your fixed costs (commission plus half-spread) are 2.5 bps per round trip, leaving 17.5 bps of gross edge to pay for impact. The strategy trades names with an average daily volume of \$200,000,000 each, and it trades the full book once per round trip.

The question is: at what book size does the impact cost equal the remaining 17.5 bps of edge, wiping out the net? Using the square-root impact (round-trip, so roughly double the one-side figure), with $a = 10$ bps calibrated at 1% of ADV:

$$\text{round-trip impact} \approx 2 \times 10 \text{ bps} \times \sqrt{\frac{Q/V}{1\%}}$$

Set this equal to the 17.5 bps of available edge and solve for the participation $Q/V$:

$$20 \times \sqrt{\frac{Q/V}{0.01}} = 17.5 \;\Rightarrow\; \sqrt{\frac{Q/V}{0.01}} = 0.875 \;\Rightarrow\; \frac{Q/V}{0.01} = 0.766 \;\Rightarrow\; Q/V = 0.77\%$$

So the *break-even* trade is about 0.77% of ADV — past that, impact alone exceeds your remaining edge. But you do not want to run at break-even (zero net profit); you want to run near the *peak* of the capacity curve, which for a square-root model sits at a smaller participation. A standard result is that net profit, $\text{(edge)} \times Q - (\text{impact const}) \times Q^{1.5}$, is maximized where the derivative is zero, which lands at a participation roughly **4/9 of the break-even level** — here about 0.34% of ADV per name. If you trade, say, 50 names each with \$200,000,000 ADV, your per-name dollar position at the optimum is about $0.34\% \times \$200{,}000{,}000 \approx \$680{,}000$, and across 50 names your book is roughly:

$$50 \times \$680{,}000 \approx \$34{,}000{,}000 \text{ per round-trip cycle}$$

If the strategy cycles a few times before names refresh and you account for the full universe and holding period, the total deployable capital scales up into the hundreds of millions — which is exactly the \$500,000,000-ish peak the capacity curve illustrated. The single sentence to remember: **capacity is set by liquidity, not by how good your signal is — a brilliant signal in a tiny, illiquid market has tiny capacity, and a mediocre signal in a giant liquid market can hold billions.**

This is why, in interviews, "what is the capacity of this strategy?" is a favorite follow-up. It tests whether you understand that a backtest's *return* and a strategy's *value* are different things. A 30% backtest return that caps out at \$5,000,000 of capacity is worth far less than a 6% return that scales to \$5,000,000,000, because the second one earns \$300,000,000 a year and the first earns \$1,500,000.

## Point-in-time data: the number-one killer, again

We come to the most insidious source of fake alpha, the one that has fooled more researchers than all the others combined: **point-in-time (PIT) data**. A point-in-time database records, for every date, *what was actually known on that date* — not what we later learned, not what was revised, not the cleaned-up final version. Backtesting without PIT data is like betting on yesterday's horse race with today's newspaper.

There are three biases that creep in when your data is *not* point-in-time, and they all push your backtest's performance up.

![Three silent biases inflate any backtest that uses today's database -- survivorship drops dead firms, restatement uses revised numbers, and index timing knows membership early -- while a point-in-time database that records what was known on each date produces honest alpha.](/imgs/blogs/backtesting-done-right-quant-research-12.png)

The flow chart traces both paths. Raw vendor history feeds three biases, which feed an inflated backtest. A proper point-in-time database feeds an honest one.

**Survivorship bias.** If your stock universe is "all companies in the database today," you have silently excluded every company that went bankrupt, got delisted, or was acquired and disappeared. Those failures are *gone* from today's database. But a strategy living through history would have held some of them and taken the losses. Backtesting only on the survivors systematically overstates returns, because you removed exactly the disasters. The effect is large: studies have found survivorship can inflate measured equity returns by **1% to 4% a year** — enough to make a losing strategy look like a winner.

**Restatement / look-ahead in fundamentals.** Companies revise their reported numbers. A firm reports earnings, then months later restates them (sometimes because of an error, sometimes a routine revision). Many databases overwrite the original number with the revised one. If your backtest uses the revised earnings on the original date, you are trading on numbers nobody had. Worse, the act of restating is itself often correlated with the stock's future — so leaking restated data can manufacture predictive power out of thin air.

**Index-inclusion timing.** When a company is added to an index like the S&P 500, the announcement comes days before the change takes effect, and the stock typically jumps on the announcement. If your database records the company as a member from the *effective* date but you assume you knew on the *announcement* date — or worse, knew it would be added before the announcement — you capture a move you could never have traded. The same applies to any "membership" data: which stocks were borrowable, which were in a screening universe, which had options listed.

#### Worked example: how survivorship inflates a backtest

Make it concrete. Suppose you backtest a simple strategy — "buy the 100 cheapest stocks by price-to-book ratio and hold for a year" — over a 20-year window, using a database that contains only companies *still trading today*. Over those 20 years, some of the cheap stocks you would have picked went to zero (cheap-and-getting-cheaper is often a sign of distress). But those bankrupt companies are not in today's database, so your backtest never picks them, never takes the −100% loss.

Quantify the leak. Say in a truly point-in-time universe, 3 of your 100 picks each year go bankrupt, losing 100% each, while the other 97 average a 12% gain. The honest annual return is:

$$\frac{97 \times 12\% + 3 \times (-100\%)}{100} = \frac{1164\% - 300\%}{100} = \frac{864\%}{100} = 8.64\%$$

But the survivorship-biased backtest never sees those 3 bankruptcies — they vanished from the database — so it averages only the 97 survivors:

$$\frac{97 \times 12\%}{97} = 12.0\%$$

The bias added **12.0% − 8.64% = 3.36% a year** of fake return, every year, just by deleting the losers from history. Compounded over 20 years, that gap is the difference between a strategy you would fund and one you would fire. The single sentence: **survivorship bias does not make your strategy look a little better — it removes precisely the worst outcomes, which is the most flattering possible distortion.**

The fix is not clever code; it is *better data*. You need a database that, for any historical date, can tell you exactly which companies existed, what their then-current (unrevised) fundamentals were, and what their index memberships were — as of that date. Vendors that sell this (with full restatement history and delisting records) charge a lot for it, and the reason they can charge a lot is that it is the difference between a real backtest and a fantasy one. In a take-home case where you are handed data, your *first* question to yourself should be: is this point-in-time, and if not, which of these three biases is inflating my result?

## In the interview room and the take-home

This is where it all gets tested. Below are five fully worked problems of the kind you will face in a quant researcher interview or a take-home case. Work each one yourself before reading the solution.

#### Worked example: Problem 1 — gross versus net on a \$100M book

*"Your backtest shows a strategy earning 6% gross on a \$100,000,000 book, with 500% annual turnover. Round-trip costs are 11 bps. What is the net return, and would you trade it?"*

Annual round-trip trading is 500% of the book: $5.0 \times \$100{,}000{,}000 = \$500{,}000{,}000$. The cost:

$$11 \text{ bps} \times \$500{,}000{,}000 = 0.0011 \times \$500{,}000{,}000 = \$550{,}000$$

Gross profit is $6\% \times \$100{,}000{,}000 = \$6{,}000{,}000$. Net is $\$6{,}000{,}000 - \$550{,}000 = \$5{,}450{,}000$, or **5.45%**. The cost ate about 0.55% of the 6%, leaving a healthy edge. You would trade it, but you would immediately ask the follow-up: how sensitive is this to the cost assumption? If real costs turn out to be 22 bps instead of 11, the drag doubles to \$1,100,000 and net falls to 4.9% — still fine. The strategy is robust to cost. The single sentence: **always restate a gross number as a net number before you have any opinion about it.**

#### Worked example: Problem 2 — the look-ahead smell test

*"A candidate shows you a daily equity strategy with a backtest Sharpe of 3.5 and a 71% daily hit rate. What is your first question?"*

A Sharpe of 3.5 on a daily strategy and a 71% hit rate are both extraordinary — far beyond what real daily signals achieve (real daily hit rates hover near 51–53%). The overwhelming prior is *look-ahead bias*. Your first question: **"At what price are you executing relative to the bar that generates the signal?"** If the answer is "I trade at the same close I computed the signal from," you have found the bug — the strategy is capturing same-bar moves it could never have traded. Make them change the fill to the next open and re-run; the Sharpe will likely collapse toward something believable. The single sentence: **an implausibly high backtest Sharpe is not evidence of a great strategy, it is evidence of a bug, and the bug is usually timing.**

#### Worked example: Problem 3 — sizing to capacity

*"Your signal earns 15 bps gross per round trip. Fixed costs are 3 bps round-trip. You trade a name with \$100,000,000 ADV. How large can a single trade be before it stops being profitable, using a square-root impact model with $a = 10$ bps at 1% of ADV (one-side)?"*

After fixed costs, you have 12 bps of edge to spend on impact. Round-trip impact is about twice the one-side figure: $2 \times 10 \times \sqrt{(Q/V)/0.01}$ bps. Set equal to 12:

$$20\sqrt{\frac{Q/V}{0.01}} = 12 \;\Rightarrow\; \sqrt{\frac{Q/V}{0.01}} = 0.6 \;\Rightarrow\; \frac{Q/V}{0.01} = 0.36 \;\Rightarrow\; Q/V = 0.36\%$$

So break-even is at 0.36% of ADV, which on \$100,000,000 ADV is $0.0036 \times \$100{,}000{,}000 = \$360{,}000$ per trade. But the *profit-maximizing* size is smaller (around 4/9 of break-even participation), roughly 0.16% of ADV or about \$160,000 per trade — that is where net profit per name peaks. The single sentence: **break-even size and profit-maximizing size are different, and you should run near the profit-maximizing one, not the break-even one.**

#### Worked example: Problem 4 — the turnover tradeoff

*"Signal A is gross Sharpe 1.5 with 50x annual turnover. Signal B is gross Sharpe 1.0 with 4x annual turnover. Round-trip cost is 10 bps and both run on a \$100,000,000 book with 10% annualized volatility. Which has the higher net Sharpe?"*

The trick is to convert each gross Sharpe to a gross return, subtract the cost drag, then recompute Sharpe. With 10% volatility, gross return = Sharpe × volatility. Signal A: $1.5 \times 10\% = 15\%$ gross. Cost drag: $10 \text{ bps} \times 50 = 500 \text{ bps} = 5.0\%$. Net return: 10%. Net Sharpe: $10\% / 10\% = 1.0$. Signal B: $1.0 \times 10\% = 10\%$ gross. Cost drag: $10 \text{ bps} \times 4 = 40 \text{ bps} = 0.4\%$. Net return: 9.6%. Net Sharpe: $9.6\% / 10\% = 0.96$.

So after costs, the two are *almost identical* — A's higher gross Sharpe is nearly entirely consumed by its 12x-higher turnover. In practice you would prefer B: it has comparable net performance, far lower turnover (so lower cost sensitivity and higher capacity), and less execution risk. The single sentence: **a high gross Sharpe with high turnover is often worse than a lower gross Sharpe with low turnover, because turnover is a multiplier on costs.**

#### Worked example: Problem 5 — spotting the data bug

*"A backtest of a value strategy on US equities from 2000 to 2020 shows a 14% annual return. The candidate built the universe from the current constituents of the Russell 3000 index. What is wrong, and how big might the error be?"*

Building the universe from *current* constituents bakes in survivorship bias: every company that went bankrupt or was delisted between 2000 and 2020 is absent, so the backtest never holds the losers. For a value strategy this is especially damaging, because cheap stocks have elevated bankruptcy rates — the strategy is most exposed to exactly the names that survivorship deletes. The magnitude is typically 1–4% a year of inflation; for a distress-tilted value strategy it can be at the high end. A 14% return might really be 10–12%. The fix: rebuild the universe point-in-time, including delisted names with their delisting returns. The single sentence: **whenever someone builds a historical universe from a present-day list, suspect survivorship first and quantify it before believing any return.**

#### Worked example: Problem 6 — the cost of immediacy

*"You need to buy \$2,000,000 of a stock with a 10-bps bid-ask spread and \$50,000,000 ADV. Compare the cost of crossing the spread immediately versus working the order patiently over the day with square-root impact ($a = 8$ bps at 1% of ADV)."*

Crossing immediately, you pay the half-spread (5 bps) plus the full impact of a \$2,000,000 order at once. Participation is $\$2{,}000{,}000 / \$50{,}000{,}000 = 4\%$, so impact is $8 \times \sqrt{4\%/1\%} = 8 \times 2 = 16$ bps. Total: 5 + 16 = **21 bps**, or $0.0021 \times \$2{,}000{,}000 = \$4{,}200$. Working the order patiently — splitting it into many small child orders over the day — you avoid crossing the full spread on every share (you can post passively and capture some spread) and you reduce instantaneous impact by spreading the participation over time. A patient execution might pay roughly half: about 10 bps, or \$2,000. The savings of ~\$2,200 on one \$2,000,000 trade is why execution algorithms exist. The single sentence: **immediacy is a luxury you pay for in basis points, and patient execution is one of the cheapest sources of alpha there is.**

## Common misconceptions

**"A higher backtest Sharpe is always better."** No. A higher *gross* Sharpe can hide a strategy that costs do not survive, and an implausibly high Sharpe (above 3 on a liquid daily strategy) is more likely a bug than a discovery. Always look at the net Sharpe, the turnover, and the capacity together. A Sharpe of 1.0 that scales to billions beats a Sharpe of 3.0 that caps at a few million.

**"Costs are a small correction you can add at the end."** They are not a correction; at moderate-to-high turnover they are the dominant term. A strategy can have a beautiful gross curve and a flat or negative net curve. You must model costs *inside* the simulation, charged on every fill, not subtracted as a lump at the end — because costs interact with sizing and capacity in ways a flat subtraction misses.

**"If I use a one-bar lag, I have no look-ahead."** The next-open execution lag is necessary but not sufficient. Look-ahead also hides in full-sample normalization, in restated fundamentals stamped with the wrong date, in survivorship-biased universes, and in index membership known too early. The lag handles price timing; the other leaks need point-in-time data and careful date-stamping.

**"Market impact is linear — a 2x order costs 2x."** No. Impact follows a square-root law: a 4x order costs about 2x, a 9x order about 3x. This concavity is what allows large funds to exist, and it is what sets capacity. Modeling impact as linear will make you both over-cautious on small orders and over-optimistic on the existence of any capacity limit at all.

**"More data and a longer backtest always make the result more reliable."** Only if the data is clean and point-in-time. A longer backtest on survivorship-biased data just compounds the bias. And a longer backtest invites *overfitting* — trying enough variations of a strategy until one looks good on the history by luck. The cure is point-in-time data, honest costs, and out-of-sample discipline, not just more years.

**"The backtest return is what I will earn live."** Almost never. Live trading adds execution slippage beyond your model, capacity constraints as your size grows, regime changes the history did not contain, and the simple fact that other people may have found the same signal and are competing it away. A good researcher treats the backtest as an *upper bound* on live performance and asks how much of it will survive contact with the real market.

## How it shows up in real research

The concepts above are not academic. Here is how they bite in the day-to-day work of a quant research desk.

**The execution desk is a profit center, not a cost center.** At a large fund, a separate team of researchers works only on *execution* — how to get filled at the best price. They build the square-root impact models, calibrate the constant $a$ from the firm's own historical fills, and design algorithms that slice a large order into hundreds of small child orders timed to minimize impact. Shaving 2 bps off average execution cost on a fund that trades hundreds of billions a year is tens of millions of dollars. When a researcher proposes a new signal, the execution team's cost model is what determines whether it is fundable. The signal researcher and the execution researcher are two halves of the same coin.

**The 5%-of-ADV rule of thumb.** Desks impose hard participation limits — often a rule like "never trade more than 5% (sometimes 10%) of a name's average daily volume in a day." This is the square-root law turned into policy: beyond a few percent of ADV, impact rises steeply and fills become unreliable. The limit caps capacity directly: if you cannot trade more than 5% of \$50,000,000 ADV in a name, you cannot hold more than a few hundred thousand dollars of it at a sensible horizon, and that ceiling, summed across your universe, *is* your strategy's capacity. Every backtest that ignores this rule overstates how much money the strategy can actually run.

**The famous look-ahead disasters.** The history of quantitative finance is littered with backtests that looked spectacular and traded terribly, almost always because of a timing or data leak. A classic pattern: a researcher z-scores a signal using the full sample's mean and standard deviation, the backtest shows a Sharpe of 4, and the strategy loses money from day one live, because the normalization was quietly using the future. Another classic: a fundamental signal stamped on the quarter-end date instead of the (six-weeks-later) report date, capturing earnings information before it was public. These are not exotic; they are the bread-and-butter mistakes that experienced reviewers hunt for first.

**The quant quake of August 2007.** Over a few days in August 2007, many quantitative equity funds suffered sharp, simultaneous losses. Part of the lesson was about *crowding and capacity*: when many funds run similar signals and all try to deleverage at once, the market impact of their combined selling becomes enormous — far beyond what any single fund's backtest, which assumed it traded alone, had modeled. The impact term that a single backtest treats as a smooth square-root curve becomes violent and nonlinear when everyone hits it together. The episode is a permanent reminder that a backtest models *you* trading against a passive market, while reality is *everyone* trading against each other.

**Why funds re-run everything point-in-time.** Serious shops maintain their own point-in-time databases — sometimes spending years and millions building them — precisely because vendor data is so often contaminated. They store, for every signal and every date, exactly the data vector that was available then. When a researcher proposes a strategy, the first validation is to re-run it on the firm's point-in-time store and see how much of the edge survives. A signal that looks great on cleaned vendor data and dies on point-in-time data was never real. This single discipline kills more "discoveries" than any other.

**The take-home case as a trap.** When a firm sends you a take-home with a dataset and asks you to find and backtest a signal, the dataset is often *deliberately* imperfect — it may not be point-in-time, the costs may be unspecified, the timing conventions ambiguous. The firm is testing whether you notice. The strongest candidates do not just report a Sharpe; they report the Sharpe *with* an explicit statement of their cost assumptions, their execution timing, and the data caveats they could not resolve. Showing that you know *what could be wrong* with your own backtest is worth more than the backtest itself. As you can see in [decision-making under uncertainty](/blog/trading/quantitative-finance/decision-making-under-uncertainty-quant-interviews), the meta-skill of reasoning about what you do not know is exactly what the interview is probing.

## When this matters to you and further reading

If you are preparing for quant researcher interviews, internalize this hierarchy: **timing first, costs second, data third, signal last.** A correct signal with a broken backtest is worthless; a modest signal with an honest backtest is fundable. When you build your own projects — and you should build some, because a clean GitHub backtest with explicit cost and timing handling is a genuine signal to interviewers — make the cost model and the next-bar execution the *first* things you write, not afterthoughts.

The practical workflow that falls out of everything above: vectorize to screen many ideas fast, then take the survivors through a careful event-driven loop that charges a square-root impact cost on every fill and executes at the next bar's open. Report gross *and* net side by side, always quote the turnover, and estimate the capacity from a participation limit. State your data's point-in-time status out loud. If you do those things, your backtests will be boring and believable — which, in this field, is the highest compliment.

To go deeper on the surrounding ideas:

- [The Kelly criterion and sequential betting](/blog/trading/quantitative-finance/kelly-criterion-sequential-betting-quant-interviews) — once you have a real net edge, this is how you size it for long-run growth without blowing up. Capacity and Kelly together determine how much money the strategy should actually run.
- [Linear regression from first principles for quant interviews](/blog/trading/quantitative-finance/linear-regression-deep-quant-interviews) — most signals are built from regressions, and the same look-ahead discipline (only fit on past data) applies to estimating the coefficients.
- [Hypothesis testing and p-values for quant interviews](/blog/trading/quantitative-finance/hypothesis-testing-pvalues-quant-interviews) — the statistical companion to this post: how to tell whether a backtest's edge is real or a lucky fluctuation you overfit to.
- [Covariance, correlation, and their pitfalls](/blog/trading/quantitative-finance/covariance-correlation-pitfalls-quant-interviews) — portfolio-level backtesting depends on getting the covariance of your positions right, and the estimation pitfalls there are as treacherous as the timing pitfalls here.

The thread tying all of it together is the same one we started with: a backtest is a simulation, and a simulation is a story you tell yourself about the past. Make the story honest — pay your costs, respect the clock, use only the data you really had — and the backtest becomes a tool you can trust. Skip any of those, and it becomes the most expensive kind of lie: the one you believe.
