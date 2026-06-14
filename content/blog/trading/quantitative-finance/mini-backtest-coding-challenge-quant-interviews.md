---
title: "The mini backtest: a 45-minute quant coding challenge, solved"
date: "2026-06-14"
publishDate: "2026-06-14"
description: "A from-scratch walkthrough of the classic quant take-home: load price data, build a moving-average signal, backtest it without look-ahead, add costs, and report a Sharpe and drawdown the grader can trust, with runnable pandas code and the exact off-by-one bug that fakes a brilliant result."
tags:
  [
    "backtesting",
    "quant-interviews",
    "moving-average-crossover",
    "look-ahead-bias",
    "sharpe-ratio",
    "transaction-costs",
    "pandas",
    "overfitting",
    "vectorized-backtest",
    "take-home-challenge",
  ]
category: "trading"
subcategory: "Quantitative Finance"
author: "Hiep Tran"
featured: true
readTime: 41
---

> [!important]
> **TL;DR** — The classic quant take-home is "here is some price data, build a simple strategy and backtest it in 45 minutes." Graders are not scoring your alpha; they are scoring whether your P&L math is correct, whether you avoided look-ahead, and whether you reported honest risk.
>
> - The single most important line in the whole exercise is `position = signal.shift(1)` — it forces today's trade to be decided by yesterday's information. Drop it and you trade on prices you have not seen yet.
> - A vectorized backtest is just a chain of column operations: price to returns, signal to shifted position, then an elementwise product and a cumulative product for the equity curve. No Python loop required.
> - The off-by-one look-ahead bug can turn a true Sharpe of 0.6 into a fake 7.8. Graders reproduce your numbers; a Sharpe that high is an instant red flag, not a hire.
> - Report risk, not just return: the annualized Sharpe ratio, the worst peak-to-trough drawdown, the hit rate, and the P&L *after* realistic transaction costs. A 5-basis-point cost across 60 trades a year can erase most of a paper edge.
> - The number to remember: subtract costs and flag overfitting *before* the grader does. The candidate who writes "this is the best of 30 parameter sets, so treat the Sharpe as optimistic" beats the one who reports a lucky maximum as if it were real.

You sit down, the clock starts, and a CSV lands in your inbox: a few years of daily closing prices for one ticker. The instructions are one sentence long. "Build a simple trading strategy and backtest it. You have 45 minutes." There is no rubric attached, no hint about what "good" looks like, and no second chance to ask what they actually want.

This is one of the most common screens in quantitative finance, and candidates routinely misread it. They burn 30 of their 45 minutes inventing a clever signal, ship a single impressive-looking number, and fail — because the grader was never grading the signal. They were grading whether you can handle financial data without fooling yourself.

This post walks the whole thing end to end, the way you would actually do it under the clock, and shows exactly what the person on the other side is looking for. We will define every term from scratch, write runnable `pandas`/`numpy` code, and put real dollar figures on every step. By the end you will know where the minutes go, which line of code matters most, and how to write the three sentences of caveats that separate a hire from a near-miss.

![A timeline of the 45-minute mini backtest broken into six phases: load and sanity-check the data, compute returns, build the signal, run the vectorized backtest with the shift, compute metrics and costs, and write the verdict.](/imgs/blogs/mini-backtest-coding-challenge-quant-interviews-1.png)

The timeline above is the mental model for the whole exercise: it is a *budget*. Most of your time goes to correctness and the no-look-ahead shift, a little to a deliberately simple signal, and a few minutes at the end to metrics and a written verdict. Spend your time in that proportion and you will pass even if your strategy loses money — because a losing strategy honestly measured is a far better answer than a winning one built on a bug.

## Foundations: the task, the vocabulary, and what is actually graded

Before any code, let us build the shared vocabulary from zero. If you already trade for a living you can skim this; if you have never opened a price file, you cannot proceed without it.

### The instruments and the units

A **price series** is just a column of numbers indexed by date: the closing price of one asset on each trading day. "Closing price" means the last traded price before the market shuts (4:00 pm in New York for US stocks). A typical take-home gives you the daily *close* and nothing else, sometimes also the *open* (the first trade of the day), the *high*, and the *low*.

A **return** is the percentage change in price over one period. If a stock goes from \$100 to \$110, its one-day return is +10%. Returns, not prices, are what a strategy actually earns — a price of \$110 tells you nothing until you know what you paid.

A **position** is how much of the asset you hold. We will use a simple convention: a position of `+1` means "fully long" (you own one unit and profit when the price rises), `0` means "flat" (you hold nothing and earn nothing), and `-1` means "fully short" (you have borrowed and sold one unit, and profit when the price *falls*). A *basis point* — written "bp" or "bps" — is one hundredth of a percent, 0.01%; it is the unit costs are quoted in, so 5 bps is 0.05%.

A **signal** is the rule that decides your position from the data you can see. A **backtest** is the simulation that replays history and asks: if I had followed this signal, what would my profit-and-loss (P&L) have been? The **equity curve** is the running total of that P&L — the dollar value of a \$1 stake over time.

### Why a backtest exists at all

It helps to be clear-eyed about what a backtest can and cannot do, because the grader is too. A backtest is *not* a prediction machine. It cannot tell you whether a strategy will make money tomorrow; markets change, edges decay, and the future is not a replay of the past. What a backtest *can* do is two narrower things, both valuable.

First, it is a **falsification tool**. If a strategy loses money on history, you can reject it cheaply, before risking a cent. Most ideas die here, and that is the point — a research desk is a machine for killing bad ideas fast. Second, it is a **sizing and risk tool**. For the rare strategy that survives, the backtest tells you how volatile it is, how deep its drawdowns run, and how much capital it can absorb before its own trading moves the market against it.

The mini backtest in an interview is a tiny instance of this, and the grader knows it. They are not asking "did you find alpha." They are asking "can you build the falsification tool correctly, so that when you *do* find something, you would actually believe it." A backtest you cannot trust is worse than no backtest, because it gives you false confidence to bet real money. That is why correctness and no-look-ahead dominate the rubric: a wrong backtest is not a weak answer, it is a *dangerous* one.

### What the grader is actually scoring

Here is the part candidates get wrong. A take-home backtest is graded on five things, and a clever signal is the *least* of them.

![A matrix of the five grading criteria with their approximate weights and what failure looks like: correctness and no look-ahead each about 30 percent, clean structure and sensible metrics about 15 percent each, communication about 10 percent.](/imgs/blogs/mini-backtest-coding-challenge-quant-interviews-2.png)

The matrix above is the rubric the grader carries in their head, roughly weighted:

- **Correctness (~30%).** Is your return math right? Is your P&L actually the position times the return? An off-by-one in a return formula sinks everything downstream.
- **No look-ahead (~30%).** Did you ever let the strategy use information it could not have had at decision time? This is the cardinal sin of backtesting, and it is so common that graders look for it *first*.
- **Clean structure (~15%).** Is the code readable and vectorized, or a 200-line for-loop with copy-pasted blocks? They are evaluating the code you would write on their desk.
- **Sensible metrics (~15%).** Did you report risk-adjusted return (the Sharpe ratio), the worst drawdown, the hit rate, and costs — or just a single return number with no risk attached?
- **Communication (~10%).** Did you state your assumptions and flag the weaknesses, especially overfitting? Three honest sentences here are worth more than a fourth decimal place on the Sharpe.

Notice that "did the strategy make money" is not on the list. A strategy that loses 2% a year, measured correctly and presented honestly, scores far higher than one that "makes" 40% a year through a look-ahead bug. Internalize that and you are already ahead of most candidates.

## Loading and sanity-checking the data

The first eight minutes go to loading the file and checking that it is not lying to you. Raw vendor data is dirty: it has duplicate timestamps, calendar gaps, missing values, and the occasional absurd price from a bad tick or an unadjusted stock split. Every one of those will silently corrupt your backtest if you skip the check.

![A branching flow of five data sanity checks: parse and sort the dates, drop duplicate timestamps, check for calendar gaps, count NaNs and zero prices, and flag any one-bar jump above fifty percent, all feeding into a clean series.](/imgs/blogs/mini-backtest-coding-challenge-quant-interviews-3.png)

Each check in the diagram catches a specific bug. Out-of-order dates break any `.shift()`. Duplicate timestamps double-count a day. A calendar gap might be a holiday (fine) or a chunk of missing data (not fine). A `NaN` or a zero price will produce an infinite or undefined return. And a 50% one-bar jump is almost always an unadjusted split or a data error, not a real move you could have traded.

Here is the loading code. Notice that the comments are indented inside the block, never starting at column zero.

```python
import pandas as pd
import numpy as np

    # Parse dates, set them as the index, and sort. The index must be
    # monotonic or every later .shift() silently misaligns.
df = pd.read_csv("prices.csv", parse_dates=["date"])
df = df.set_index("date").sort_index()

    # 1. Duplicate timestamps: keep the last print for each day.
df = df[~df.index.duplicated(keep="last")]

    # 2. Missing or impossible values.
n_nan = df["close"].isna().sum()
n_zero = (df["close"] <= 0).sum()
print(f"NaNs: {n_nan}, non-positive prices: {n_zero}")

    # 3. Absurd one-bar jumps (likely an unadjusted split or a bad tick).
ratio = df["close"] / df["close"].shift(1)
suspect = df.index[(ratio > 1.5) | (ratio < 0.67)]
print(f"suspicious jumps on: {list(suspect)}")

    # 4. Calendar gaps: business days with no row.
expected = pd.bdate_range(df.index.min(), df.index.max())
missing = expected.difference(df.index)
print(f"missing business days: {len(missing)}")
```

#### Worked example: catching a \$1,000 split that fakes a -50% day

Suppose the file has Apple closing at \$200 on one day and \$100 the next, with no other context. A naive backtest reads that as a one-day return of `100/200 - 1 = -50%` — a catastrophic loss that never happened. In reality Apple did a 2-for-1 stock split: every share became two, each worth half as much, and a holder's wealth was unchanged. The price "halved" only as an accounting artifact.

The `ratio` check above flags this instantly: `100/200 = 0.5`, which is below the `0.67` threshold, so the date prints as suspect. The fix is to use *split-adjusted* prices (most vendors provide an "adjusted close" column that already accounts for splits and dividends). If you backtest on unadjusted prices, a single split can swing your reported annual return by tens of percent.

The intuition: a backtest can only be as honest as its input, so you spend the first minutes proving the input is clean before you trust a single computed return.

#### Worked example: a calendar gap that is a holiday vs one that is missing data

The `missing` check above prints a list of business days with no row. Two cases look identical in the output but mean opposite things, and conflating them is a common error.

Suppose the gap is the last Thursday and Friday of November. That is the US Thanksgiving holiday — the market was closed, so there is *no* price, and there should not be one. Dropping those dates from a business-day calendar is correct. But suppose the gap is a random Tuesday in March with no holiday: that is almost certainly *missing data*, a hole in the vendor feed. If you silently forward-fill the previous price into that hole, you have invented a flat day that never happened and will compute a fake 0% return for it.

The fix is to align to an actual *trading* calendar, not a naive business-day one:

```python
    # A naive business-day calendar counts holidays as "missing"; an
    # exchange calendar knows the market was legitimately closed.
import pandas_market_calendars as mcal
nyse = mcal.get_calendar("NYSE")
sessions = nyse.valid_days(df.index.min(), df.index.max())
real_gaps = sessions.tz_localize(None).difference(df.index)
print(f"genuinely missing trading days: {len(real_gaps)}")
```

A handful of genuinely missing trading days is usually fine to drop. Dozens of them means the feed is broken and you should say so in your write-up rather than backtest on a corrupted series.

The intuition: a missing date is innocent if the market was closed and a defect if it was open, so you check against a real exchange calendar before deciding to fill or drop.

## Computing returns correctly

Returns are where the first subtle scoring happens, because there are two definitions and they are not interchangeable.

A **simple return** (also called arithmetic return) is the fractional change: `r = P_t / P_{t-1} - 1`. If the price goes \$100 to \$110, the simple return is `110/100 - 1 = 0.10`, or +10%.

A **log return** (continuously-compounded return) is the natural logarithm of the price ratio: `r = ln(P_t / P_{t-1})`. For the same move, `ln(110/100) = 0.0953`, or +9.53%.

For small moves the two are nearly identical, but they have one crucial difference: **log returns add across time, simple returns do not.**

![A 100-dollar price path going to 110 dollars then 99 dollars, with the simple returns of plus and minus ten percent shown in green and red, the log returns of plus 9.53 and minus 10.54 percent below them, and a summary that log returns sum to ln of 99 over 100 while simple returns leave you at 99 dollars not 100.](/imgs/blogs/mini-backtest-coding-challenge-quant-interviews-4.png)

#### Worked example: why +10% then -10% does not get you back to \$100

Start with \$100. It rises 10% to \$110. Then it falls 10%, but 10% of \$110 is \$11, so it lands at \$99 — not \$100. The two simple returns, +10% and -10%, sum to zero, yet you lost a dollar. Simple returns are *not* additive.

Now do the same with log returns. The up move is `ln(110/100) = +9.53%`. The down move is `ln(99/110) = -10.54%`. Their sum is `9.53% - 10.54% = -1.01%`, which is exactly `ln(99/100)` — the true cumulative log return from \$100 to \$99. Log returns *are* additive, which is why a backtest that sums daily returns across a year should use logs, while a backtest that compounds them (`(1 + r).cumprod()`) uses simple returns.

In code, both are one line. Dividends complicate this: if the asset paid a \$2 dividend on a day, the holder earned that \$2 even though the price dropped by roughly \$2 on the ex-dividend date. To capture total return you use the **adjusted close**, which folds reinvested dividends back into the price.

```python
    # Simple (arithmetic) returns -- use these when compounding with cumprod.
df["ret_simple"] = df["close"].pct_change()

    # Log returns -- use these when you want returns to add across time.
df["ret_log"] = np.log(df["close"]).diff()

    # Total return including dividends, if the vendor gives an adjusted close.
df["ret_total"] = df["adj_close"].pct_change()
```

#### Worked example: the \$2 dividend that looks like a loss

Dividends are the most common way a correct-looking return calculation goes subtly wrong. Suppose a stock closes at \$100 on Monday, pays a \$2 dividend, and the price opens at \$98 on the **ex-dividend date** (the first day the buyer no longer receives the dividend). The raw price dropped from \$100 to \$98, so a naive return is `98/100 - 1 = -2%` — it looks like the stock fell.

But a shareholder lost nothing. They still hold a share worth \$98 *and* they received \$2 in cash. Their total wealth went from \$100 to \$98 + \$2 = \$100: a true return of 0%, not -2%. The price drop is just the dividend leaving the share.

If your backtest uses raw closing prices, every dividend shows up as a phantom loss, and for a high-dividend stock over many years that drag can total tens of percent of fake underperformance. The fix is to use the **adjusted close**, which folds reinvested dividends back into the price history so that the computed return is the *total* return — price change plus dividends. On a take-home, if the file has both `close` and `adj_close`, prefer `adj_close` for returns and say why in a comment. That one sentence shows the grader you know the difference between price return and total return.

The intuition: a falling price after a dividend is not a loss, so you compute returns on the dividend-adjusted price to capture the cash the holder actually received.

The intuition: pick simple returns when you will compound the equity curve and log returns when you will sum, and never mix the two in one calculation.

## A simple signal: the moving-average crossover

With clean returns in hand, you need a signal — and here the strong move is to be *deliberately boring*. A moving-average crossover is the canonical interview signal because it is trivial to explain, impossible to get subtly wrong, and leaves you time for the parts that are actually graded.

A **moving average** is the mean price over the last `N` days, recomputed each day. A **fast** moving average uses a short window (say 20 days) and reacts quickly; a **slow** one uses a long window (say 50 days) and reacts sluggishly. The **crossover** rule is: go long (`+1`) when the fast average is above the slow average, and go flat (`0`) when it is below. The idea, loosely, is that when the recent trend is above the longer trend, the asset is in an uptrend worth riding.

![A price chart with the closing price as a solid line and the slow 50-day moving average as a dashed line, with green vertical bands marking the periods where the fast average is above the slow average and the rule is long, and a flat region where the fast average is below the slow average.](/imgs/blogs/mini-backtest-coding-challenge-quant-interviews-5.png)

The chart shows the signal drawn on the price: the green bands are the long periods, and the only moments anything happens are the crossovers, where the position flips. Everywhere else the position is held constant.

#### Worked example: turning a price into a +1/0 position

```python
fast = df["close"].rolling(20).mean()
slow = df["close"].rolling(50).mean()

    # Signal: +1 (long) when the fast MA is above the slow MA, else 0 (flat).
df["signal"] = np.where(fast > slow, 1, 0)
```

Suppose on a Monday the 20-day average is \$104 and the 50-day average is \$101. Since `104 > 101`, the signal is `+1`: long. Three weeks later the price has slid, the 20-day average has dropped to \$98 while the 50-day is still \$100, so `98 < 100` and the signal flips to `0`: flat. Between those two dates the signal never changed, so you held the long position untouched the whole time and traded only twice — once to enter, once to exit.

The intuition: a crossover signal converts a noisy continuous price into a handful of discrete long-or-flat decisions, which makes it easy to reason about and cheap to trade.

#### Worked example: the mean-reversion alternative in three lines

If the interviewer says "try something other than momentum," the natural counterpart is **mean reversion**: bet that a price which has moved far from its recent average will snap back. The cleanest version uses a **z-score** — how many standard deviations the current price sits above or below its rolling mean.

```python
window = 20
mean = df["close"].rolling(window).mean()
std = df["close"].rolling(window).std()

    # z-score: distance from the rolling mean in standard deviations.
z = (df["close"] - mean) / std

    # Mean reversion: short when stretched high (z > 1), long when low (z < -1).
df["signal_mr"] = np.where(z < -1, 1, np.where(z > 1, -1, 0))
```

Read the rule: when the price is more than one standard deviation *below* its 20-day mean (`z < -1`), it is "cheap" relative to its recent self, so go long and bet on a bounce; when it is more than one above (`z > 1`), it is "rich," so go short. Crucially, the `mean` and `std` are computed with `.rolling(20)`, which only ever looks *backward* — using the full-sample mean and standard deviation here would be a look-ahead leak, because at any historical point you could not have known the future average. This is the single most common way mean-reversion backtests cheat, so flag it explicitly.

The intuition: mean reversion is the mirror image of momentum, and its one trap is computing the reference statistics over the whole sample instead of a backward-looking window.

A note for the interview: do not agonize over *which* signal. Momentum (buy what went up recently) and mean-reversion (buy what went down recently) are the two archetypes; a moving-average crossover is the simplest momentum rule. The grader does not care if it makes money. They care that you implemented it without look-ahead — which is the next section, and the heart of the whole exercise.

## The vectorized backtest and the all-important shift-by-one

This is the section that decides your score. Everything above was setup; here is where a backtest is either honest or a fantasy.

The cardinal rule: **you can only trade on information you actually had at the moment of decision.** When you compute today's signal from today's closing price, you only *know* that closing price after the market has shut. So you cannot trade on it today — the earliest you can act is tomorrow. In a daily backtest, this means the position you hold on day `t` must be set by the signal from day `t-1`.

That is the entire purpose of one line of code: `position = signal.shift(1)`. The `.shift(1)` slides the whole signal column down by one row, so each day inherits the signal computed from the *previous* day's data.

![A timeline showing signal today and trade tomorrow: at Monday's close the signal is computed as plus one, at Tuesday's open the long position starts, at Tuesday's close you earn Tuesday's return times the position, a new signal of zero is computed from Tuesday's price, and at Wednesday's open you exit to flat.](/imgs/blogs/mini-backtest-coding-challenge-quant-interviews-6.png)

The timeline makes the timing concrete. The signal you see at Monday's close drives the position you hold *through Tuesday*. You earn Tuesday's return on that position. The new signal computed from Tuesday's close drives Wednesday. Signal today, trade tomorrow — never trade on the same bar that produced the signal.

#### Worked example: the off-by-one that fakes a Sharpe of 7.8

Here is the bug, and it is a single missing `.shift(1)`:

```python
    # WRONG: today's signal multiplies today's return. The signal already
    # "knows" today's move, because it was computed from today's close.
df["pnl_bad"] = df["signal"] * df["ret_simple"]

    # RIGHT: yesterday's signal sets today's position. Decision precedes
    # the return it earns. This single shift is the whole game.
df["position"] = df["signal"].shift(1)
df["pnl"] = df["position"] * df["ret_simple"]
```

Why is the first line so deadly? On any day the price rose, the fast average ticks up too, so `signal` is more likely to be `+1` on exactly the days the return is positive. The strategy "decides" to be long on up-days and flat on down-days — *after* already seeing whether the day was up or down. That is not a strategy; it is reading tomorrow's newspaper.

![A before-and-after comparison: on the left the look-ahead bug multiplies today's signal by today's return, the signal already knows today's move, and the reported Sharpe is an impossible 7.8; on the right the corrected version shifts the signal by one, sets the position from yesterday's close, and reports an honest Sharpe of 0.6.](/imgs/blogs/mini-backtest-coding-challenge-quant-interviews-7.png)

The numbers in the figure are real in spirit and worth memorizing. A typical look-ahead bug on a daily series inflates the Sharpe ratio from an honest ~0.6 to something absurd like 7.8. A Sharpe above ~3 on a single-asset daily strategy is essentially impossible in real markets; when a grader sees it, they do not think "genius," they think "look-ahead," and they go hunting for the missing shift. Catching this in your own code before they do is the difference between a pass and a fail.

With the shift in place, the whole backtest is a clean chain of column operations — no Python loop at all.

![A data-flow showing the vectorized backtest as one chain: price becomes per-bar returns and also feeds the signal, the signal is shifted by one bar into a position, position times return gives gross P&L, costs on position changes are subtracted to give net P&L, and a cumulative product of net returns gives the equity curve.](/imgs/blogs/mini-backtest-coding-challenge-quant-interviews-8.png)

#### Worked example: a vectorized backtest producing a $ equity curve

```python
    # 1. Returns from price.
df["ret"] = df["close"].pct_change()

    # 2. Signal -> shifted position (the no-look-ahead step).
df["position"] = df["signal"].shift(1).fillna(0)

    # 3. Gross strategy return, bar by bar: position times market return.
df["gross"] = df["position"] * df["ret"]

    # 4. Equity curve: a $1 stake compounded through the gross returns.
df["equity"] = (1 + df["gross"]).cumprod()

final = df["equity"].iloc[-1]
print(f"$1 grew to ${final:.2f}")
```

Walk it line by line on a tiny example. Suppose on three consecutive days the market returns are +2%, -1%, +3%, and your shifted positions are `1, 1, 0`. Then `gross` is `+2%, -1%, 0%`. The equity curve is `1 * 1.02 = 1.020`, then `1.020 * 0.99 = 1.0098`, then `1.0098 * 1.00 = 1.0098`. Your \$1 became \$1.01 — you caught the +2% day, ate the -1% day, and correctly earned nothing on the third day because you were flat. No loop, four lines, and the timing is provably correct because the position was shifted before it ever touched a return.

The intuition: a correct vectorized backtest is a pipeline of aligned columns ending in a cumulative product, and the one line that makes it honest is the shift that puts the decision strictly before the return it earns.

## Adding transaction costs

A backtest with no costs is a sales pitch, not a measurement. Every time you change your position you pay to trade: a commission to the broker, half the **bid-ask spread** (the gap between the price you can buy at and the price you can sell at), and **market impact** (your own order pushing the price against you). On a liquid US stock these add up to a few basis points per trade; on something illiquid, much more.

The cost is charged only when the position *changes*. If you hold `+1` for twenty days straight, you pay nothing on the nineteen days you sat still — only on the day you entered and the day you exit.

```python
    # Cost in basis points per unit of position change, one side.
COST_BPS = 5
cost_rate = COST_BPS / 10_000          # 5 bps = 0.0005

    # Turnover: how much the position changed this bar (0 if unchanged).
df["turnover"] = df["position"].diff().abs().fillna(0)

    # Charge the cost on the traded amount; subtract from the gross return.
df["cost"] = df["turnover"] * cost_rate
df["net"] = df["gross"] - df["cost"]
df["equity_net"] = (1 + df["net"]).cumprod()
```

#### Worked example: deducting $ costs and recomputing net P&L

Take a strategy that turns over its full position 60 times a year (30 round-trips of in-and-out). At 5 bps per side, each side costs 0.05% of the traded notional. Sixty position changes a year at 5 bps is `60 * 0.0005 = 0.03`, or **3.0% per year** of pure cost drag — before you have earned a cent.

Now put dollars on it. On a \$1,000,000 book, one round-trip (out then in, two sides) at 5 bps per side costs `2 * 0.0005 * $1,000,000 = $1,000`. Thirty round-trips a year is **\$30,000** in costs annually. If the gross strategy made +34% on the year, the net is closer to +19% after a 3% drag compounded against a turnover-heavy book — and the **gross Sharpe of 0.95 falls to a net Sharpe of 0.41.**

![A before-and-after comparison of gross versus net: the gross return is plus 34 percent with a Sharpe of 0.95 and 60 round-trips a year before any cost; the net column subtracts a 3 percent annual cost drag, leaving a plus 19 percent return and a net Sharpe of 0.41, still positive but barely.](/imgs/blogs/mini-backtest-coding-challenge-quant-interviews-11.png)

That collapse is the point. Most "winning" backtests are winning only because they ignored costs. A grader will *always* ask "what about costs," so charge them yourself and report both numbers. A strategy whose edge survives realistic costs is rare and valuable; one whose edge is entirely eaten by costs is a common and important finding to report honestly.

The intuition: costs scale with how often you trade, so a high-turnover signal needs a large gross edge just to break even, and showing the net number proves you understand that.

### When the vectorized backtest is not enough

The four-line vectorized backtest above is the right tool for an interview and for screening many ideas quickly, but it is worth knowing its limits, because a good interviewer will ask "when would this approach break?"

The vectorized approach assumes every decision is independent of the path: each day's position depends only on that day's signal, and the P&L is a clean elementwise product. That assumption holds for a simple crossover, but it fails the moment your strategy has **state** that evolves with the trades themselves. Three common cases break it:

- **Position limits and capital constraints.** If you can only deploy \$1,000,000 and a signal wants to add to an already-full position, the vectorized math happily lets the position exceed your capital. A real backtester has to clamp the position to what you can actually afford, and that clamp depends on the running P&L — a path dependency the vectorized version cannot express.
- **Stop-losses and take-profits.** A rule like "exit if the position loses more than 5% from entry" depends on the *entry price of the current trade*, which the vectorized product has no memory of. You need to track the open trade across bars.
- **Market impact that depends on order size.** If trading more than 1% of daily volume moves the price against you, the cost of a trade depends on how big the trade is, which depends on the current position, which depends on the path. The square-root impact models real desks use cannot be a single fixed cost column.

For these, you switch to an **event-driven backtest**: a loop that advances one bar at a time, updates state, computes the signal from data *through that bar only*, decides a target position, executes it at the next bar with size-dependent cost, and marks P&L before advancing the clock. It is slower and more code, but it enforces correct timing and path-dependent logic that the vectorized form cannot. In the interview, the senior move is to *build the vectorized version* (because you have 45 minutes) while *naming* the event-driven version as the production answer for path-dependent rules. That one sentence — "this vectorizes because the position is path-independent; with stops or impact I'd switch to an event loop" — is exactly the kind of judgment they are screening for.

The intuition: vectorize when each decision is independent of the path, and reach for an event-driven loop the moment state like stops, limits, or size-dependent impact enters the strategy.

## Computing the metrics

Now you measure. A single return number is not an answer, because it says nothing about the risk you took to get it. The three metrics a grader expects are the Sharpe ratio, the maximum drawdown, and the hit rate.

### The Sharpe ratio

The **Sharpe ratio** is the most important number in the report. It is return *per unit of risk*: the average return divided by the volatility of those returns. A strategy that makes 10% a year with calm, steady gains has a higher Sharpe than one that makes 10% through wild swings, because the calm one took less risk for the same reward.

Formally, the daily Sharpe is the mean daily return divided by the standard deviation of daily returns. To make it comparable across strategies you **annualize** it by multiplying by the square root of the number of trading days in a year, which is 252. The square root appears because volatility grows with the square root of time, not linearly.

$$\text{Sharpe}_{\text{annual}} = \frac{\bar{r}}{\sigma_r} \times \sqrt{252}$$

Here $\bar{r}$ is the mean of the daily strategy returns, $\sigma_r$ is their standard deviation, and $\sqrt{252} \approx 15.87$ is the annualization factor.

![A flow building the annualized Sharpe ratio from daily P&L: the mean daily return of 0.00042 and the standard deviation of 0.011 combine into a daily Sharpe of 0.038, which is multiplied by the square root of 252, about 15.87, to give an annual Sharpe of 0.60.](/imgs/blogs/mini-backtest-coding-challenge-quant-interviews-10.png)

#### Worked example: computing the annualized Sharpe in code

```python
    # Daily strategy returns, net of costs.
r = df["net"].dropna()

    # Annualized Sharpe: mean over std, scaled by sqrt of 252 trading days.
sharpe = r.mean() / r.std() * np.sqrt(252)
print(f"annualized Sharpe: {sharpe:.2f}")
```

Put numbers to it. Suppose the mean daily net return is `0.00042` (about 4.2 bps a day) and the daily standard deviation is `0.011` (about 1.1%). The daily Sharpe is `0.00042 / 0.011 = 0.038`. Annualize: `0.038 * 15.87 = 0.60`. An annual Sharpe of 0.60 is a plausible, honest single-asset result — not spectacular, not embarrassing, and exactly the kind of number a grader trusts. (For context, a Sharpe near 1.0 is a genuinely good single strategy; above 2.0 on one asset daily is rare; above 3.0 should make you check for a bug.)

#### Worked example: where the Sharpe ratio quietly lies

The `sqrt(252)` annualization assumes daily returns are independent and roughly normally distributed. Both assumptions can fail, and a sharp grader may probe whether you know it.

First, **autocorrelation**. If your strategy's daily returns are *positively* autocorrelated — good days tend to follow good days — then the true annual volatility is higher than `daily_std * sqrt(252)` suggests, because the `sqrt` rule undercounts the clustering. This inflates the naive Sharpe. The classic offender is a strategy holding illiquid or stale-priced assets whose marks update slowly, smoothing the returns artificially. A quick diagnostic is to check the lag-1 autocorrelation: `df["net"].autocorr(lag=1)`. If it is materially positive, your annualized Sharpe is optimistic.

Second, **fat tails**. Financial returns are not normal; they have far more extreme days than a bell curve predicts. A strategy can show a lovely Sharpe while quietly carrying the risk of a rare, catastrophic day that the standard deviation barely registers — think of a strategy that earns small premiums most of the time and blows up occasionally, like selling insurance. The Sharpe ratio, which only sees mean and variance, is blind to that asymmetry.

This is why desks report companions to the Sharpe. The **Sortino ratio** divides by *downside* deviation only, penalizing volatility that hurts you while ignoring upside swings. The **Calmar ratio** divides annual return by the maximum drawdown, directly answering "return per unit of worst-case pain." You do not need to compute all three in 45 minutes, but naming them — "Sharpe assumes normality, so I'd also look at Sortino and max drawdown for the tail risk" — signals real fluency.

The intuition: the Sharpe ratio compresses a strategy into two numbers, mean and variance, so it is blind to clustering and tail risk, which is why you quote drawdown alongside it.

### Maximum drawdown

The **maximum drawdown** is the worst peak-to-trough loss the equity curve ever suffered — how much you would have lost if you had bought in at the worst possible high and held to the following low. It answers "how bad did it get along the way," which a return-only number completely hides.

![An equity curve of a one-dollar stake growing to 1.34 dollars over six years, with the worst drawdown shaded in red: the curve peaks, then falls 18 percent from its peak to a trough at 1.16 dollars, before recovering to end at 1.34 dollars, with the dollar levels labeled on the vertical axis and years on the horizontal axis.](/imgs/blogs/mini-backtest-coding-challenge-quant-interviews-9.png)

#### Worked example: the equity curve and its 18% drawdown

```python
    # Running maximum of the equity curve (the high-water mark).
peak = df["equity_net"].cummax()

    # Drawdown: how far below the peak we are, as a fraction.
drawdown = df["equity_net"] / peak - 1
max_dd = drawdown.min()
print(f"max drawdown: {max_dd:.1%}")
```

In the figure, a \$1 stake compounds up to a peak, then slides from that peak down to a trough at \$1.16 — a drop of about 18% from the high-water mark — before recovering and ending the sample at \$1.34, a +34% total gain. Two strategies can both end at +34%, but one that got there in a smooth glide and one that suffered a stomach-churning 35% drawdown along the way are *not* the same strategy. The drawdown is the number that tells you whether you could have actually held on.

### The hit rate

The **hit rate** is the fraction of trading days (or trades) that were profitable. It is the simplest sanity metric, and it has one subtle trap: hit rate and profitability are not the same thing. A strategy can be right 70% of the time and still lose money if its losing days are much larger than its winning days.

```python
    # Fraction of active days with a positive net return.
active = df["net"][df["position"] != 0]
hit_rate = (active > 0).mean()
print(f"hit rate: {hit_rate:.1%}")
```

A realistic single-asset daily strategy has a hit rate near 50-55%. If you see a hit rate of 68% next to a Sharpe of 7.8, that is the look-ahead bug waving at you again — real edges are thin, and a 68% daily hit rate on one asset does not exist without a leak.

The intuition: report all three — risk-adjusted return, worst-case path, and consistency — because each hides a different way the strategy could disappoint you.

## Presenting the result

You have maybe five minutes left. Do not spend them tuning. Spend them writing the summary that the grader reads first. A strong write-up is short and brutally honest:

```python
print(f"Period:        {df.index[0].date()} to {df.index[-1].date()}")
print(f"Net return:    {(df['equity_net'].iloc[-1] - 1):.1%}")
print(f"Sharpe (net):  {sharpe:.2f}")
print(f"Max drawdown:  {max_dd:.1%}")
print(f"Hit rate:      {hit_rate:.1%}")
print(f"Trades/year:   {df['turnover'].sum() / years:.0f}")
print("Assumptions:   close-to-close fills, 5 bps cost/side, no slippage")
print("Caveat:        single in-sample fit; treat Sharpe as optimistic")
```

That last line — the caveat — is worth a disproportionate amount of credit. It tells the grader you know your own result is fragile. Naming the weakness before they find it is the single highest-leverage sentence you can write.

## Extending it: a parameter sweep, and the overfitting trap

If you have time, the natural extension is to ask "which moving-average windows work best?" by sweeping over many combinations of fast and slow windows and recording the Sharpe for each. This is also the fastest way to *fail* the exercise if you do it naively, because it walks you straight into overfitting.

**Overfitting** is when you tune a strategy so hard to past data that you fit the noise rather than the signal. With 30 parameter combinations, one of them will look great *by pure chance* — the same way that if 30 people each flip ten coins, one will probably get nine heads and look "skilled." Reporting that lucky maximum as your result is the most common overfitting mistake.

![A heatmap of annualized Sharpe ratios across a grid of fast and slow moving-average windows, with most cells near 0.5 in amber and red, one bright green cell at 1.92, and a warning that the lone high cell is luck because its neighbors are all near 0.5, so you should report the median rather than the maximum.](/imgs/blogs/mini-backtest-coding-challenge-quant-interviews-12.png)

#### Worked example: spotting the lucky cell in a parameter sweep

```python
results = {}
for fast_w in [5, 10, 20, 30, 40]:
    for slow_w in [50, 100, 150]:
        f = df["close"].rolling(fast_w).mean()
        s = df["close"].rolling(slow_w).mean()
        pos = np.where(f > s, 1, 0)
        pos = pd.Series(pos, index=df.index).shift(1).fillna(0)
        net = pos * df["ret"] - pos.diff().abs().fillna(0) * cost_rate
        results[(fast_w, slow_w)] = net.mean() / net.std() * np.sqrt(252)

best = max(results, key=results.get)
print(f"best params {best}: Sharpe {results[best]:.2f}")
print(f"median Sharpe across grid: {np.median(list(results.values())):.2f}")
```

Look at the heatmap. Almost every cell sits between 0.3 and 0.6 — a uniformly mediocre field. One cell, `fast=20, slow=100`, shows a glittering **1.92**. The naive candidate reports "I found a Sharpe of 1.92." The strong candidate notices that the 1.92 cell's *neighbors* are all near 0.5: if 20/100 were truly special, the nearby 20/150 and 30/100 settings would also be elevated, because real edges are robust to small parameter changes. A lone bright cell surrounded by noise is a statistical accident, not a discovery.

So the strong candidate reports the **median** Sharpe across the grid (~0.5) as the honest expectation, flags the 1.92 as likely overfit, and writes one sentence: "the best in-sample parameters give 1.92, but the result is not robust to nearby settings, so I treat ~0.5 as the realistic estimate." That single act of restraint is what separates a research-grade answer from a curve-fit. (For the full machinery of measuring this rigorously — purged cross-validation, the deflated Sharpe ratio — see the deeper treatment linked at the end.)

The intuition: searching many parameters guarantees one lucky winner, so the honest summary of a sweep is its typical cell, not its maximum.

## In the interview room: five sub-tasks, fully solved

Onsite versions of this challenge come as a sequence of small asks, often spoken aloud while you share your screen. Here are five that come up constantly, each solved end to end.

#### Worked example: "compute the return on this \$100 price series"

The interviewer pastes four prices: \$100, \$102, \$99, \$105. "Give me the daily simple returns and the total return over the four days."

```python
prices = pd.Series([100, 102, 99, 105])
simple = prices.pct_change().dropna()
    # 0.0200, -0.0294, 0.0606
total = prices.iloc[-1] / prices.iloc[0] - 1
    # 105/100 - 1 = 0.05
```

Day-to-day: `102/100 - 1 = +2.00%`, `99/102 - 1 = -2.94%`, `105/99 - 1 = +6.06%`. The total return is `105/100 - 1 = +5.0%`. The trap they are watching for: do you try to *add* the daily simple returns (`2 - 2.94 + 6.06 = 5.12%`) and get the wrong total? You shouldn't — simple returns compound, they don't sum. The correct compounded check is `1.02 * 0.9706 * 1.0606 = 1.05`, which matches the +5.0% total. Saying that out loud earns the point.

#### Worked example: "add the shift so there's no look-ahead, and show me the bug"

The interviewer wants you to demonstrate, not just assert, that the shift matters.

```python
sig = (df["close"].rolling(20).mean() > df["close"].rolling(50).mean()).astype(int)

    # The bug: same-bar signal times same-bar return.
sharpe_bug = (sig * df["ret"]).pipe(lambda x: x.mean() / x.std() * np.sqrt(252))

    # The fix: shift the signal one bar before it earns a return.
sharpe_ok = (sig.shift(1) * df["ret"]).pipe(lambda x: x.mean() / x.std() * np.sqrt(252))

print(f"bug: {sharpe_bug:.2f}   fixed: {sharpe_ok:.2f}")
```

You run it and narrate: "Without the shift I get a Sharpe of 7.8, which is impossible for a daily single-asset strategy, so I know I'm leaking. With `.shift(1)` it drops to 0.6, which is believable. The signal was computed from the close, so I can't trade on it until the next bar." That narration — naming *why* 7.8 is suspect — is what they are scoring, more than the code.

#### Worked example: "annualize this Sharpe from daily to yearly"

The interviewer gives you a daily Sharpe of 0.05 and asks for the annual figure. The answer is `0.05 * sqrt(252) = 0.05 * 15.87 = 0.79`. The follow-up trap: "what if these were monthly returns?" Then you scale by `sqrt(12) = 3.46`, not `sqrt(252)`. And "weekly?" By `sqrt(52) = 7.21`. The rule is always the square root of the number of periods per year, because volatility scales with the square root of time. If you scale Sharpe *linearly* (multiplying by 252), you are off by a factor of ~16 — a classic and instantly-spotted error.

#### Worked example: "the candidate before you reported a Sharpe of 12. What happened?"

This is a debugging question disguised as a war story. A Sharpe of 12 on a backtest is never real, so you list the usual culprits in order of likelihood: (1) **look-ahead** — the missing shift, by far the most common; (2) **using future data in the signal** — e.g. a `z-score` computed with the full-sample mean and standard deviation instead of a rolling, backward-looking one; (3) **survivorship bias** — backtesting only on stocks that still exist today, which silently drops every company that went bankrupt; (4) **no costs** — a high-turnover strategy that ignores the spread; and (5) **a bug in the return alignment** — joining the signal and the price on mismatched dates. Reciting that checklist, in priority order, shows you have actually been burned by these before.

#### Worked example: "extend it to a long-short strategy and recompute the P&L"

So far the position was `+1` or `0`. The interviewer asks for a **long-short** version: long when the fast average is above the slow (`+1`), short when it is below (`-1`). The only change is the signal:

```python
    # Long-short: +1 above, -1 below. Never flat.
df["signal_ls"] = np.where(fast > slow, 1, -1)
df["position_ls"] = df["signal_ls"].shift(1).fillna(0)
df["gross_ls"] = df["position_ls"] * df["ret"]
```

The P&L mechanics are identical — `position * return` — but two things change and you should flag both. First, a short position *profits when the price falls*, so `-1 * (-2%) = +2%`: the math handles it automatically because a negative position times a negative return is positive. Second, the turnover roughly doubles: flipping from `+1` to `-1` is a position change of `2`, not `1`, so the cost line `position.diff().abs()` correctly charges twice as much, and your net Sharpe will suffer more from costs. Naming that cost consequence, unprompted, is the senior move.

## Common misconceptions

**"The strategy has to make money to pass."** It does not. Graders score correctness, not profitability. A losing strategy measured honestly beats a winning one built on look-ahead every single time, because the first proves you can be trusted with real money and the second proves you cannot.

**"Look-ahead only happens in fancy code."** It hides in the most innocent places: normalizing returns with the full-sample mean, filling a missing value with the *next* day's price, computing a signal on the close and trading on that same close. The single `.shift(1)` is the guard, but you also have to make sure every input to the signal is itself backward-looking. Any statistic computed over the whole sample and applied to the past is a leak.

**"In-sample results predict the future."** They do not, and a single backtest is the most optimistic estimate you will ever see, because you chose the strategy *after* seeing the data. The honest move is to assume the live Sharpe will be meaningfully lower than the backtest Sharpe — often half — and to say so.

**"Costs are a rounding error."** They are frequently the whole story. A signal that trades every day pays its turnover many times over; a 3% annual cost drag turns a "great" 4% gross edge into a 1% net loss. Any backtest without costs is a marketing brochure, not a measurement.

**"A higher Sharpe is always better."** Only if it is real and net of costs. A Sharpe of 7.8 is not impressive; it is diagnostic of a bug. And a Sharpe computed gross of costs, or on an overfit parameter set, is worse than a lower honest number because it will not survive contact with a live market.

**"You annualize the Sharpe by multiplying by the number of periods."** No — you multiply by the *square root* of the number of periods, because risk grows with the square root of time, not linearly. Multiplying a daily Sharpe by 252 instead of `sqrt(252)` overstates it by a factor of about 16, turning a believable 0.6 into a ludicrous 9.6. This is one of the most common arithmetic slips in the whole exercise, and graders watch for it specifically because it reveals whether you actually understand *why* the annualization factor has the form it does.

**"More data always makes the backtest more reliable."** Longer history helps statistically, but it cuts the other way too: market regimes change, so a strategy that worked in the low-rate decade of the 2010s may behave completely differently in a high-rate, high-volatility regime. A ten-year backtest that spans multiple regimes is more honest than a two-year one, but only if you check that the edge is *stable across* those regimes rather than driven entirely by one lucky stretch. Plotting the rolling annual Sharpe, not just the full-sample number, is how you catch a strategy that "worked" only in one window.

## How it shows up on a real desk

The mini backtest is a miniature of the actual job, and the same failure modes recur at full scale with real money on the line.

**The quant who shipped a look-ahead to production.** A junior researcher at a systematic fund builds a signal that backtests at Sharpe 3.5 and pushes it toward live trading. In review, a senior notices the signal's `z-score` is computed with `df.mean()` over the entire history — including the future — rather than a 60-day rolling window. Re-run backward-looking, the Sharpe collapses to 0.4. The exact same bug as the interview's missing shift, just dressed up in a normalization step. This is why every serious desk has a code-review gate specifically hunting for future leakage.

**The strategy that died on transaction costs.** A famous category of "anomalies" in academic finance — strategies that look profitable on paper — largely vanish once realistic trading costs and the bid-ask spread are charged. A signal might generate a 6% gross annual return but trade so frenetically that 7% of cost drag turns it into a loser. Desks maintain detailed cost models precisely because the gross-to-net gap is where most paper edges go to die.

**The 2007 quant quake.** In August 2007, many equity market-neutral funds running similar mean-reversion signals suffered enormous, simultaneous losses over a few days, far outside what their backtests' drawdown estimates suggested possible. The lesson that propagated through the industry: a backtested maximum drawdown is a lower bound on pain, not an upper bound, because the historical sample never contained the specific crowded-deleveraging scenario that actually occurred. Always treat the backtest drawdown as optimistic.

**Survivorship bias in index backtests.** A common rookie error is to backtest a stock strategy on today's S&P 500 constituents over the last 20 years. But the index 20 years ago held different companies, many of which were later removed *because they failed*. Testing only on survivors quietly deletes every disaster from the sample and inflates returns. Real research uses point-in-time constituent data, and graders sometimes plant this trap in the take-home dataset to see if you notice.

**The overfit that looked like a discovery.** A researcher sweeps thousands of parameter combinations and finds a configuration with a backtest Sharpe of 2.5. It goes live and earns 0.3. The configuration was the luckiest cell in a giant grid, exactly the lone bright square in the heatmap above. Disciplined desks combat this with out-of-sample holdouts, purged cross-validation, and the deflated Sharpe ratio, which adjusts the reported Sharpe downward for the number of strategies tried — the institutional version of "report the median, not the max."

**The join that leaked the future.** A subtle, production-grade version of look-ahead hides in data alignment. A researcher merges a signal computed from one data source with prices from another, using a date join. But the signal source publishes with a one-day lag — a "fundamentals" value dated for a given day is not actually available until the next morning. Joining on the nominal date silently lets the strategy trade on data it could not have had for another 24 hours. The backtest Sharpe looks great; live, it evaporates. The defense is a discipline desks call *point-in-time* data: every value is stamped with the moment it became *knowable*, not the moment it nominally applies to, and joins use the knowable timestamp. The interview's `.shift(1)` is the toy version of this exact problem — both are about aligning each decision strictly after the information it uses became available.

## When this matters to you and further reading

If you are preparing for a quant screen, the mini backtest is the one exercise most worth rehearsing until it is muscle memory, because it is graded on craft you can practice rather than insight you have to be born with. Set a 45-minute timer, grab any free daily price series, and run the whole loop: load, sanity-check, returns, signal, shift, costs, Sharpe, drawdown, write-up. Do it three times and the timing becomes automatic, freeing your attention for the part that actually earns the offer — the honest sentence of caveats at the end.

The deeper you go on a real desk, the more these foundations compound. Two next steps build directly on this post: read [backtesting done right](/blog/trading/quantitative-finance/backtesting-done-right-quant-research) for the full event-driven loop, realistic fill assumptions, and detailed cost and slippage models that a production backtester needs beyond the vectorized sketch here; and read [overfitting, purged cross-validation, and the deflated Sharpe](/blog/trading/quantitative-finance/overfitting-purged-cv-deflated-sharpe-quant-research) for the rigorous machinery that turns "report the median, not the max" into a defensible statistical procedure. Together they take you from passing the 45-minute screen to doing the job it screens for.

This is educational material about how backtests are built and graded, not financial advice; no strategy described here is a recommendation to trade anything.
