---
title: "Evaluating alpha signals: IC, IR, Sharpe, drawdown, turnover, and decay"
date: "2026-06-14"
publishDate: "2026-06-14"
description: "A from-first-principles, interview-ready guide to the metrics that decide whether a trading signal is real, tradeable, and durable: build the information coefficient, the information ratio and the fundamental law of active management, the Sharpe ratio and its annualization, maximum drawdown and Calmar, turnover and net-of-cost returns, and signal-decay half-life -- each with a fully worked dollar example and five solved take-home problems."
tags:
  [
    "alpha-signals",
    "information-coefficient",
    "information-ratio",
    "sharpe-ratio",
    "drawdown",
    "turnover",
    "signal-decay",
    "fundamental-law",
    "quant-research",
    "quant-interviews",
    "backtesting",
  ]
category: "trading"
subcategory: "Quantitative Finance"
author: "Hiep Tran"
featured: true
readTime: 44
---

> [!important]
> **TL;DR** — No single number tells you whether a trading signal is worth money. A battery of complementary metrics does, and each one answers a different question.
>
> - The **information coefficient (IC)** is the rank correlation between your signal today and returns tomorrow. A *good* equity signal has an IC around **0.03 to 0.05** — almost zero, and that is normal.
> - The **information ratio (IR)** turns that tiny per-bet edge into a portfolio statistic through the **fundamental law of active management**: $\text{IR} \approx \text{IC} \cdot \sqrt{\text{breadth}}$. Breadth — the number of independent bets — is half the game.
> - The **Sharpe ratio** is return per unit of volatility; you annualize a daily Sharpe by multiplying by $\sqrt{252}$, and it quietly assumes returns are roughly normal and independent.
> - **Maximum drawdown** is the worst peak-to-trough loss an investor actually lived through; **Calmar** divides return by it. **Turnover** measures how much you trade, and net-of-cost return is the only return that pays your rent.
> - **Signal decay** and its **half-life** tell you how fast the edge fades, which sets your holding period and your turnover.
> - The number to remember: a Sharpe of **2.0 gross** can become **0.9 net** once a 200%-turnover signal pays 5 basis points of cost per round-trip on a \$20,000,000 book. Costs are not a footnote; they are the whole story for fast signals.

## Why one number is never enough

Imagine you have built a number. For every stock in your universe, every day, your model spits out a value — call it a *signal* (a **signal** is simply any number you compute today that you hope predicts returns tomorrow; "alpha signal" is the same thing, where *alpha* means return that is not just compensation for taking obvious market risk). Maybe it is a momentum score, maybe it is a sentiment reading scraped from earnings calls, maybe it is the output of a neural network. You back-tested it, the equity curve goes up and to the right, and you are excited.

Now a portfolio manager asks you six questions in a row:

1. Does the signal actually predict returns, or did you get lucky on a few names?
2. If the edge per stock is tiny, can you trade enough stocks to make it matter?
3. How much money does it make per unit of risk?
4. What is the worst losing streak I would have to stomach?
5. How much does it cost to trade, and is there anything left after costs?
6. How fast does the edge go stale — do I hold for a day or a quarter?

Each question has its own metric, and **no metric answers more than one of them.** A signal can have a beautiful information coefficient and still be untradeable because you can only apply it to ten stocks. It can have a gorgeous gross Sharpe and lose money net of costs. It can look durable in-sample and evaporate the moment you trade it live. The job of a quant researcher is to run the *whole battery* and read each number for what it does — and does not — say.

![A mental-model diagram of the alpha-signal evaluation battery: one raw signal and one forward return feed the information coefficient, which feeds the information ratio and Sharpe, which split into drawdown and turnover, with signal decay branching off, all flowing into a trade-or-shelve verdict.](/imgs/blogs/evaluating-alpha-signals-ic-sharpe-turnover-quant-research-1.png)

The diagram above is the mental model for this entire post. One raw signal and the returns it tries to predict flow into the information coefficient. The IC, scaled by how many independent bets you make, becomes the information ratio and the Sharpe. Those split into the risk view (drawdown, Calmar) and the cost view (turnover, net return). Signal decay sits off to the side, governing how often you must re-trade. Everything converges on one decision: *trade it or shelve it.*

This is also, not coincidentally, one of the most reliable interview topics at quant research shops. Two Sigma, Citadel, DE Shaw, and AQR all probe whether you understand what these numbers mean, how they relate, and — most revealingly — how each one can lie. By the end of this post you will be able to compute every one by hand on a small example, state its assumptions, name its failure mode, and answer the take-home questions that hinge on it.

Everything here is educational, not investment advice. The point is to understand the machinery, not to recommend any strategy.

## Foundations: forward returns and what a metric must capture

Before any metric, we need two columns of numbers lined up correctly. Get this wrong and every downstream statistic is garbage, so we build it from zero.

### Returns, defined simply

A **return** is the percentage change in price over a period. If a stock closes at \$100 today and \$101 tomorrow, the one-day return is

$$
r = \frac{101 - 100}{100} = 0.01 = 1\%.
$$

We almost always work in returns rather than prices because returns are comparable across stocks (a \$1 move means something different for a \$10 stock than a \$1,000 stock) and roughly stationary (their statistical properties stay put over time, unlike prices, which wander).

A **basis point** (bps) is one hundredth of a percent: 1 bps = 0.01% = 0.0001. Quants live in basis points because daily edges are small. A signal that makes 2 bps a day — two hundredths of one percent — sounds like nothing, but compounded over 252 trading days that is roughly 5% a year before leverage. Hold that intuition: in this world, *small numbers are the whole business.*

### Forward returns: the thing we are trying to predict

Here is the single most important — and most often botched — idea in signal evaluation. Your signal is computed at the *close* of day $t$. The return it is supposed to predict must happen *after* that, on day $t+1$ or later. We call this the **forward return**:

$$
r_{i, t+1} = \frac{P_{i, t+1} - P_{i, t}}{P_{i, t}},
$$

where $P_{i,t}$ is the price of stock $i$ at the close of day $t$. The signal $x_{i,t}$ and the forward return $r_{i,t+1}$ are the two columns we will correlate. Their subscripts must never overlap in time. If you accidentally correlate today's signal with *today's* return, you are using information you could not have had when you traded, and you will manufacture a spectacular, entirely fake edge. This mistake — letting future information leak into a backtest — is called **look-ahead bias**, and catching it is the first thing a good researcher (or interviewer) checks.

A **cross-section** is one slice in time: all the stocks on a single day, each with its signal value and its forward return. Most signal metrics are computed *within* a cross-section first (how well did the signal rank stocks *today*?), then averaged across days. That two-step structure — compute per day, then average over days — recurs everywhere below.

### What we want a metric to capture

A good evaluation metric should reward three distinct properties, and we will need different metrics because no single one captures all three:

| Property | Question it answers | Metric that measures it |
| --- | --- | --- |
| **Real** | Does the signal genuinely rank returns? | Information coefficient |
| **Scalable** | Can the tiny edge be applied widely enough to matter? | Breadth, information ratio |
| **Tradeable & durable** | Does it survive costs, and how fast does it fade? | Sharpe, turnover, net return, half-life |

Notice that "the equity curve goes up" is on none of these lists. An equity curve is a *summary*; it hides whether the gains came from one lucky bet or a thousand small correct ones, whether they survive costs, and whether they will persist. The metrics below pull those hidden facts back out.

## The information coefficient: is the signal real?

The **information coefficient (IC)** is the workhorse. It measures, on a single day, how well your signal ranked the stocks by their forward return. It is just a correlation — but a specific kind.

### Rank IC, and why we rank

The cleanest version is the **rank IC** (also called the *Spearman IC*). The recipe:

1. On a given day, rank every stock by its signal value (1 = lowest signal, $N$ = highest).
2. Rank every stock by its forward return the same way.
3. Compute the ordinary (Pearson) correlation between those two rank columns.

![A pipeline showing how a rank IC is computed: raw signal and forward return, then rank both columns one to N, then Pearson correlation of the ranks gives a daily IC, then average over all days gives the mean IC you report.](/imgs/blogs/evaluating-alpha-signals-ic-sharpe-turnover-quant-research-11.png)

Why rank instead of correlating the raw numbers? Because raw returns have fat tails — a single stock that doubles on a buyout would dominate a raw (Pearson) correlation and swamp the signal in the other 499 stocks. Ranking flattens that: the biggest winner is just "rank 500", one notch above the second-biggest. Rank IC asks the question we actually care about — *did high-signal stocks tend to out-return low-signal stocks?* — without letting one outlier hijack the answer. (If you compute the Pearson correlation on the raw values instead, you get the **Pearson IC**; both are used, but rank IC is the more robust default and the one most interviewers expect.)

The IC lives in $[-1, +1]$. An IC of $+1$ means your signal ranked every stock perfectly; $0$ means no relationship; $-1$ means your signal is exactly backwards (which, helpfully, is a great signal — just flip the sign).

![An IC scatter plot with signal rank on the x-axis from one to ten and forward return on the y-axis from minus three to plus three percent, showing ten points in a noisy cloud with a faint upward trend line, gain and miss quadrants shaded green and red, annotated rank IC equals plus 0.42.](/imgs/blogs/evaluating-alpha-signals-ic-sharpe-turnover-quant-research-2.png)

The scatter above is what an IC *looks like*. Each dot is a stock: its signal rank on the horizontal axis, its forward return on the vertical. Green dots earned positive returns, red dots negative. The faint upward tilt of the cloud is the edge. On a tiny ten-stock toy cross-section the tilt is visible and the IC is a healthy +0.42, but on a real 500-stock universe the same edge produces an IC near 0.03 — a tilt so faint you would never see it by eye. **That faint tilt is the entire business.** Reading too much into the visual is a trap; you trust the number, not the picture.

#### Worked example: computing a rank IC by hand

Let us compute a rank IC on a cross-section of six stocks. Here is the data — the signal value we computed at yesterday's close, and the forward return each stock actually delivered:

| Stock | Signal $x$ | Forward return $r$ (%) |
| --- | --- | --- |
| A | 0.8 | +1.2 |
| B | 0.2 | -0.5 |
| C | 0.9 | +0.9 |
| D | 0.4 | +0.3 |
| E | 0.1 | -1.1 |
| F | 0.6 | -0.2 |

**Step 1 — rank the signal** (1 = lowest). Sorting the signal column: E (0.1) = 1, B (0.2) = 2, D (0.4) = 3, F (0.6) = 4, A (0.8) = 5, C (0.9) = 6.

**Step 2 — rank the forward return.** Sorting returns: E (-1.1) = 1, B (-0.5) = 2, F (-0.2) = 3, D (+0.3) = 4, C (+0.9) = 5, A (+1.2) = 6.

**Step 3 — line up the two rank columns:**

| Stock | Signal rank $R_x$ | Return rank $R_r$ | $d = R_x - R_r$ | $d^2$ |
| --- | --- | --- | --- | --- |
| A | 5 | 6 | -1 | 1 |
| B | 2 | 2 | 0 | 0 |
| C | 6 | 5 | +1 | 1 |
| D | 3 | 4 | -1 | 1 |
| E | 1 | 1 | 0 | 0 |
| F | 4 | 3 | +1 | 1 |

**Step 4 — apply the Spearman formula.** When there are no ties, the rank correlation has a tidy closed form:

$$
\rho = 1 - \frac{6 \sum d^2}{n(n^2 - 1)}.
$$

Here $\sum d^2 = 1+0+1+1+0+1 = 4$ and $n = 6$, so

$$
\rho = 1 - \frac{6 \times 4}{6 \times 35} = 1 - \frac{24}{210} = 1 - 0.114 = 0.886.
$$

A rank IC of **+0.886** on this toy cross-section. The signal ranked the six stocks almost perfectly — only the top two are swapped (A and C). The one-sentence intuition: *the rank IC is just how closely your signal's ordering matches the return's ordering, scored from -1 to +1.*

On a real universe you will never see 0.886. You will see 0.02, 0.04, maybe 0.06 on a great day, averaged across hundreds of stocks and noise. Which raises the obvious question.

### What a 0.03 IC actually means

A daily rank IC of 0.03 sounds like a rounding error. Here is why it is, in fact, a strong signal.

An IC of $\rho$ means that the fraction of variance in forward-return ranks "explained" by your signal is about $\rho^2$. For $\rho = 0.03$, that is $0.0009$ — less than one tenth of one percent. **The signal explains essentially none of tomorrow's return.** And that is *fine*, because tomorrow's return is overwhelmingly noise that nobody can predict. You are not trying to predict the return; you are trying to be right *slightly more than half the time*, consistently, across thousands of bets, so the law of large numbers works in your favor.

A rough and widely-quoted bridge: the *hit rate* (fraction of bets that go the right way) relates to IC approximately as $\text{hit rate} \approx 0.5 + \text{IC}/2$. An IC of 0.03 corresponds to being right about **51.5%** of the time. A casino runs the world's most reliable business on a smaller edge than that. The trick, as with the casino, is volume — which is exactly what breadth and the information ratio formalize.

### IC over time: the durability check

A single day's IC is noisy. What matters is the IC *averaged across many days* — and, just as important, how *stable* it is. We compute the IC every day and then look at the time series.

![A time series of monthly information coefficients over a 24-month window, shown as green and red bars around zero, with a blue 3-month rolling mean line staying mostly above zero around plus 0.03, the y-axis labeled monthly mean daily IC from minus 0.04 to plus 0.10.](/imgs/blogs/evaluating-alpha-signals-ic-sharpe-turnover-quant-research-3.png)

The chart above plots the IC month by month over a two-year evaluation window. Individual months bounce above and below zero — that is normal sampling noise. The story is the blue rolling-mean line, which sits steadily around +0.03 and rarely dips negative. **A real signal earns a positive IC most months; the rolling mean staying above zero is what separates a durable edge from a lucky run.** A signal whose IC was +0.15 for three months and then collapsed to zero is far more dangerous than one that grinds out +0.02 every single month, even though the first has a higher peak.

Two summary numbers fall out of this time series:

- **Mean IC** — the average daily IC, the headline measure of "is the signal real". For equity signals, mean ICs of 0.02 to 0.05 are typical for good signals; above 0.10 sustained is suspicious (often a sign of look-ahead bias or a tiny, untradeable universe).
- **IC information ratio** (sometimes written IC-IR, confusingly) — the mean IC divided by the *standard deviation* of the daily ICs. This measures consistency: a high IC-IR means the signal is reliably positive, not occasionally spectacular. An IC-IR around 0.5 (annualized, often higher) is the hallmark of a signal you can build a portfolio on.

The distinction between "how big is the edge" (mean IC) and "how reliable is it" (IC-IR) is exactly the same distinction we will see between return and Sharpe. Quant finance asks, over and over: *not just how much, but how much per unit of risk.*

## The information ratio and the fundamental law

Here is the bridge from "the signal predicts a little" to "the portfolio makes a lot". It is the most important formula in this whole post, and it appears in essentially every quant research interview in some form.

### What the information ratio is

The **information ratio (IR)** of a strategy is its *active return* (return above a benchmark) divided by its *active risk* (the volatility of that excess return, called **tracking error**):

$$
\text{IR} = \frac{\text{active return}}{\text{tracking error}}.
$$

It is the Sharpe ratio's cousin: where Sharpe measures return per unit of *total* volatility relative to cash, IR measures return per unit of volatility relative to a *benchmark*. A long-only manager beating the S&P 500 is judged on IR; a market-neutral hedge fund's IR and Sharpe are nearly the same thing because its benchmark is roughly zero (cash). For our purposes, treat IR as "risk-adjusted skill at the portfolio level".

### The fundamental law of active management

In the 1990s Richard Grinold wrote down a deceptively simple relationship now called the **fundamental law of active management**:

$$
\text{IR} \approx \text{IC} \cdot \sqrt{\text{breadth}}.
$$

In words: *your risk-adjusted skill equals your per-bet skill times the square root of how many independent bets you make.* Here **breadth (N)** is the number of *independent* bets per year — roughly, the number of separate decisions whose outcomes are not just copies of each other.

This single equation explains the entire structure of the quant industry. It says a tiny IC, applied across enough independent bets, produces a strong information ratio. It is why quant funds trade thousands of names instead of betting big on ten — the $\sqrt{N}$ term is doing the heavy lifting.

![A chart of the fundamental law showing three curves of information ratio versus breadth, for ICs of 0.02, 0.04, and 0.06, each rising as the square root of breadth, with a reference point marking breadth equals 1000 and IC equals 0.04 giving an information ratio of 1.26.](/imgs/blogs/evaluating-alpha-signals-ic-sharpe-turnover-quant-research-4.png)

The figure above plots the law. Each curve fixes an IC and sweeps breadth along the horizontal axis; the information ratio rises with the square root of breadth. The marked reference point makes the magic concrete: an IC of just **0.04** applied across **1,000** independent bets a year yields $0.04 \times \sqrt{1000} = 0.04 \times 31.6 = 1.26$ — a genuinely strong information ratio that institutional allocators line up to fund. The same 0.04 IC on 10 bets a year yields $0.04 \times \sqrt{10} = 0.13$ — useless. *Same signal. Same per-bet skill. Wildly different business, entirely because of breadth.*

#### Worked example: from IC and breadth to information ratio

You have a US equity momentum signal with a measured mean rank IC of **0.035**. Your universe is the 500 largest US stocks, and you rebalance the portfolio **daily**. What information ratio should you expect?

**Step 1 — estimate breadth.** This is the subtle part. Breadth is the number of *independent* bets, not the raw count of positions. A naive count says $500 \text{ stocks} \times 252 \text{ days} = 126{,}000$ bets a year. But those bets are not independent: the 500 stocks move together (they share market and sector factors), and today's positions overlap heavily with yesterday's (momentum changes slowly). Both effects shrink effective breadth dramatically.

Suppose, after accounting for cross-sectional correlation, your 500 stocks behave like roughly **100 independent bets per day**, and because positions turn over slowly, consecutive days are correlated enough that you get the equivalent of about **10 fully independent cross-sections per year** of decision-making — giving effective breadth on the order of $100 \times 10 = 1{,}000$. (Estimating effective breadth honestly is genuinely hard and a frequent interview discussion; the headline lesson is that *effective* breadth is far below the naive position count.)

**Step 2 — apply the law:**

$$
\text{IR} \approx 0.035 \times \sqrt{1000} = 0.035 \times 31.6 = 1.11.
$$

**Step 3 — sanity-check against reality.** An information ratio above 1 is excellent; the best systematic equity shops run blended IRs in the 1 to 2 range *after* combining many signals. So 1.11 from a single signal with a 0.035 IC and 1,000 effective bets is plausible but optimistic — which is exactly the right instinct. The one-sentence intuition: *the fundamental law tells you that breadth, not raw IC, is usually the binding constraint on a quant strategy's quality.*

The dangerous corollary, which interviewers love to test: a signal with a *huge* IC but tiny breadth is often worthless. If someone shows you an IC of 0.20 but it only works on 5 illiquid micro-cap stocks, $\text{IR} \approx 0.20 \times \sqrt{5} = 0.45$ — and that is before costs eat it alive in names you cannot actually trade. High IC, no breadth, is one of the great rookie traps.

## The Sharpe ratio: return per unit of risk

The **Sharpe ratio** is the most famous number in all of finance, and the most misunderstood. It is worth getting exactly right.

### Definition

The Sharpe ratio is excess return divided by volatility:

$$
\text{Sharpe} = \frac{\bar{r} - r_f}{\sigma},
$$

where $\bar{r}$ is the strategy's average return over some period, $r_f$ is the risk-free rate (what cash earns — say a Treasury bill), and $\sigma$ is the standard deviation of the strategy's returns over that same period. **Standard deviation** is the usual measure of how much returns bounce around their average; it is our proxy for risk.

The intuition: two strategies both make 10% a year, but one does it on a smooth ride and the other lurches up and down terrifyingly. The smooth one has a higher Sharpe because it earned the same return with less risk. Sharpe answers *how much return did you earn per unit of stomach-churning?*

For a market-neutral quant strategy, $r_f$ is often dropped (the strategy holds little net cash exposure), so Sharpe reduces to mean return over volatility.

### Annualization: the $\sqrt{252}$ rule

Sharpe ratios are quoted *annualized* so you can compare a daily strategy to a monthly one. The conversion rests on a fact about how means and standard deviations scale with time, assuming returns are independent across periods:

- **Mean** return scales *linearly* with the number of periods: $T$ days of average daily return $\mu$ accumulate to roughly $T\mu$.
- **Standard deviation** scales with the *square root* of the number of periods: $T$ days of daily volatility $\sigma$ accumulate to roughly $\sigma\sqrt{T}$, because independent random shocks partially cancel.

Divide them and the $T$ over $\sqrt{T}$ leaves a $\sqrt{T}$:

$$
\text{Sharpe}_{\text{annual}} = \text{Sharpe}_{\text{daily}} \times \sqrt{252},
$$

where 252 is the number of trading days in a year. (Use 12 for monthly data, 52 for weekly.)

![A curve showing how a Sharpe ratio grows with the square root of the number of trading periods compounded, with a daily Sharpe of 0.12 reaching 0.55 at 21 days and 1.90 at 252 days, the x-axis labeled trading periods and the y-axis labeled resulting Sharpe ratio.](/imgs/blogs/evaluating-alpha-signals-ic-sharpe-turnover-quant-research-5.png)

The curve above shows the scaling in action. A daily Sharpe of 0.12 — which looks like nothing — rides the square-root curve up to **1.90** annualized over 252 days. The blue dot at 21 days shows the same daily Sharpe annualizes to only 0.55 over a single month; you need the full year of independent bets to realize the headline number. *The annual Sharpe is large only because the square root of 252 is about 15.9.*

#### Worked example: annualizing a daily Sharpe from a dollar P&L series

You run a strategy on a \$10,000,000 book and record its daily profit-and-loss (P&L) in dollars over one week. Convert to a daily return by dividing each day's P&L by the book size, then annualize the Sharpe.

| Day | P&L (\$) | Daily return = P&L / \$10,000,000 |
| --- | --- | --- |
| Mon | +18,000 | +0.0018 |
| Tue | -5,000 | -0.0005 |
| Wed | +22,000 | +0.0022 |
| Thu | +9,000 | +0.0009 |
| Fri | +6,000 | +0.0006 |

**Step 1 — mean daily return.** Sum the returns: $0.0018 - 0.0005 + 0.0022 + 0.0009 + 0.0006 = 0.0050$. Divide by 5: $\bar{r} = 0.0010$, i.e. 10 bps a day.

**Step 2 — standard deviation of daily returns.** The deviations from the mean (0.0010) are $+0.0008, -0.0015, +0.0012, -0.0001, -0.0004$. Square them: $0.64, 2.25, 1.44, 0.01, 0.16$ (all $\times 10^{-6}$). Sum $= 4.50 \times 10^{-6}$. Using the sample variance (divide by $n-1 = 4$): $1.125 \times 10^{-6}$. So $\sigma = \sqrt{1.125 \times 10^{-6}} \approx 0.00106$, about 10.6 bps.

**Step 3 — daily Sharpe.** Ignoring the risk-free rate (market-neutral):

$$
\text{Sharpe}_{\text{daily}} = \frac{0.0010}{0.00106} \approx 0.94.
$$

**Step 4 — annualize:**

$$
\text{Sharpe}_{\text{annual}} = 0.94 \times \sqrt{252} = 0.94 \times 15.87 \approx 15.0.
$$

A 15 Sharpe is *absurd* — no real strategy sustains that. The reason is that five days is a laughably small sample; one good week looks superhuman. This is the lesson the example is built to teach: **annualization amplifies short-sample flukes.** You need *years* of daily data before an annualized Sharpe means anything. The one-sentence intuition: *annualizing multiplies the noise along with the signal, so a high Sharpe on a tiny sample is almost always a mirage.*

### What Sharpe quietly assumes — and where it breaks

The annualization and the ratio itself bake in assumptions that are routinely violated:

- **Returns are roughly normal.** Sharpe treats standard deviation as the whole story of risk. But real returns have **fat tails** — extreme moves happen far more often than a normal distribution predicts. A strategy that sells insurance (collects small premiums, occasionally blows up) can show a gorgeous Sharpe right up until the day it loses everything. Sharpe is blind to that asymmetry.
- **Returns are independent across days.** The $\sqrt{252}$ rule assumes no autocorrelation. Strategies with momentum or trends in their own P&L violate this, and their true annualized Sharpe is lower than the naive scaling suggests.
- **Volatility is the right risk measure.** Sharpe penalizes upside volatility exactly as much as downside. The **Sortino ratio** fixes this by using only *downside* deviation; the **Calmar ratio** (next section) uses drawdown instead. Each is a different answer to "what is risk?"

The practical upshot: a high Sharpe is necessary but not sufficient. Always pair it with a drawdown view and a tail check.

## Drawdown and the Calmar ratio: the pain you would actually feel

Sharpe is an abstraction. **Drawdown** is visceral — it is the money you watched evaporate.

### Maximum drawdown, defined

The **drawdown** at any point in time is how far the equity curve has fallen from its highest previous peak. The **maximum drawdown (MDD)** is the worst such fall over the whole history:

$$
\text{MDD} = \max_{t} \left( \frac{\text{peak before } t - \text{value at } t}{\text{peak before } t} \right).
$$

It answers the question every investor actually asks: *if I had put money in at the worst possible moment, how much would I have lost before it recovered?* That is the pain that gets strategies shut down and researchers fired, regardless of what the Sharpe said.

![An equity curve in dollars over twelve months, rising from 100k to a peak of 120k, falling to a trough of 98k with the drawdown region shaded red, then recovering to 130k, annotated with maximum drawdown of minus 22k or minus 18 percent.](/imgs/blogs/evaluating-alpha-signals-ic-sharpe-turnover-quant-research-6.png)

The figure above traces an equity curve from \$100k up to a peak of \$120k, down to a trough of \$98k, then on to \$130k. The shaded red band is the drawdown: the deepest peak-to-trough fall. *Maximum drawdown is the worst losing streak an investor would have had to live through* — here a fall of \$22,000, or 18% of the peak.

#### Worked example: maximum drawdown of an equity curve in dollars

Take the monthly account values from the figure: \$100k, \$108k, \$115k, \$112k, \$120k, \$105k, \$98k, \$110k, \$118k, \$125k, \$122k, \$130k.

**Step 1 — track the running peak** (the highest value seen so far) at each month:

| Month | Equity (\$k) | Running peak (\$k) | Drawdown (\$k) | Drawdown (%) |
| --- | --- | --- | --- | --- |
| 1 | 100 | 100 | 0 | 0.0 |
| 2 | 108 | 108 | 0 | 0.0 |
| 3 | 115 | 115 | 0 | 0.0 |
| 4 | 112 | 115 | -3 | -2.6 |
| 5 | 120 | 120 | 0 | 0.0 |
| 6 | 105 | 120 | -15 | -12.5 |
| 7 | 98 | 120 | -22 | -18.3 |
| 8 | 110 | 120 | -10 | -8.3 |
| 9 | 118 | 120 | -2 | -1.7 |
| 10 | 125 | 125 | 0 | 0.0 |
| 11 | 122 | 125 | -3 | -2.4 |
| 12 | 130 | 130 | 0 | 0.0 |

**Step 2 — find the worst drawdown.** Scanning the drawdown column, the deepest point is month 7: equity \$98k against a prior peak of \$120k. In dollars that is

$$
\text{MDD}_\$ = 120 - 98 = \$22{,}000, \qquad \text{MDD}_\% = \frac{22}{120} = 18.3\%.
$$

**Step 3 — note the recovery.** The curve did not regain its \$120k peak until month 10. That three-month stretch underwater — equity below the prior high — is the **drawdown duration**, and it matters as much as the depth. A 5% drawdown that lasts two years can be more demoralizing than a 20% drawdown that recovers in a month. The one-sentence intuition: *maximum drawdown measures the deepest hole the strategy ever dug, in the dollars an investor would actually have lost.*

### The Calmar ratio

The **Calmar ratio** divides annualized return by the maximum drawdown:

$$
\text{Calmar} = \frac{\text{annualized return}}{|\text{max drawdown}|}.
$$

If our example strategy returned 30% annualized and had an 18.3% max drawdown, its Calmar is $30 / 18.3 = 1.64$. A Calmar above 1 is generally considered good; it says "I make more in a year than I lose in my worst stretch". Calmar is popular with allocators precisely because drawdown is the risk *they* feel — it is path-dependent and human in a way standard deviation is not. Its weakness is that it depends on a single worst historical episode, so it is statistically noisier than Sharpe; one bad day can dominate it.

## Turnover and net-of-cost performance: the only return that matters

Everything so far has been *gross* — before the cost of trading. Now we get to the number that actually pays your rent.

### Turnover, defined

**Turnover** measures how much of your portfolio you trade. If your book is \$20,000,000 and on a given day you buy and sell \$4,000,000 worth of stock to rebalance, your daily turnover is

$$
\text{turnover} = \frac{\$4{,}000{,}000}{\$20{,}000{,}000} = 20\%.
$$

Annualized, 20% a day is roughly $20\% \times 252 \approx 5{,}000\%$, meaning you trade through 50 times your book in a year. That is a *fast* signal. A slow value signal might turn over the book once or twice a year (100% to 200% annual turnover).

Turnover matters because **every trade costs money.** You pay the *bid-ask spread* (the gap between the price you can buy at and the price you can sell at), exchange and broker fees, and **market impact** — the fact that your own buying pushes the price up against you. Lump these together as a **transaction cost** measured in basis points per dollar traded. For liquid large-cap US equities, a reasonable all-in round-trip cost is on the order of 5 to 10 bps; for illiquid names it can be 50 bps or more.

### Net return: subtract the cost

The cost drag on returns is turnover times cost per unit traded:

$$
\text{net return} = \text{gross return} - \text{turnover} \times \text{cost per trade}.
$$

Because cost scales with turnover, a fast signal pays the toll far more often than a slow one. This is the single most common way a backtest that "works" turns out to be worthless.

![A chart of turnover versus return, with a flat green gross-return line at 12 percent, a rising amber cost line, and a falling blue net-return line that crosses zero into a red loss region at a breakeven around 267 percent turnover.](/imgs/blogs/evaluating-alpha-signals-ic-sharpe-turnover-quant-research-7.png)

The chart above tells the whole story. The green gross-return line is flat — gross return does not care how much you trade. The amber cost line rises with turnover. The blue net-return line — gross minus cost — slopes down, and crosses into the red loss region at a breakeven turnover. Past that point, *you are trading purely to enrich your broker.* The art of execution is finding the turnover that maximizes net return, which usually means trading *less* than the raw signal suggests.

![A before-and-after comparison contrasting a gross cost-blind view where a Sharpe of 2.0 looks elite with 200 percent daily turnover ignored, against a net view with 5 basis points round-trip cost where the drag pulls return down to a Sharpe of 0.9 that is still real.](/imgs/blogs/evaluating-alpha-signals-ic-sharpe-turnover-quant-research-12.png)

The before-and-after above is the cautionary tale every quant learns once. On the left, a cost-blind backtest shows a Sharpe of 2.0 and the researcher celebrates. On the right, once you charge 5 bps per round-trip against the signal's heavy turnover, the drag pulls the Sharpe down to 0.9 — *still a real, fundable strategy, but a completely different business.* The gross number was a fantasy; the net number is the truth.

#### Worked example: turning a gross Sharpe into a net Sharpe on a \$20,000,000 book

You manage a \$20,000,000 market-neutral book. The backtest shows a gross annual return of 16% and a gross annualized Sharpe of **2.0** (so gross volatility is $16\% / 2.0 = 8\%$). The signal is fast: daily turnover averages **40%**, and your measured all-in transaction cost is **6 bps per dollar of one-way trading** (i.e. per dollar bought or sold).

**Step 1 — annual turnover in dollars.** Daily turnover of 40% on a \$20M book is $0.40 \times \$20{,}000{,}000 = \$8{,}000{,}000$ traded per day (one-way). Over 252 days: $\$8{,}000{,}000 \times 252 = \$2{,}016{,}000{,}000$ — just over \$2 billion of trading a year on a \$20M book. That is the turnover monster.

**Step 2 — annual cost in dollars.** At 6 bps (0.0006) per dollar traded:

$$
\text{cost} = \$2{,}016{,}000{,}000 \times 0.0006 = \$1{,}209{,}600.
$$

**Step 3 — convert cost to a return drag.** As a fraction of the \$20M book:

$$
\text{cost drag} = \frac{\$1{,}209{,}600}{\$20{,}000{,}000} = 6.05\%.
$$

**Step 4 — net return and net Sharpe.** Gross return was 16%, so net return is $16\% - 6.05\% = 9.95\%$. Assuming costs do not change the volatility (a reasonable first approximation), net Sharpe is

$$
\text{Sharpe}_{\text{net}} = \frac{9.95\%}{8\%} = 1.24.
$$

The Sharpe fell from 2.0 to 1.24 — costs ate more than a third of it — but the strategy survives. Now redo it with the same signal at **half the turnover** (20% daily, achieved by trading less aggressively and accepting slightly stale positions): cost drag halves to about 3.0%, net return rises to $\approx 13\%$, and net Sharpe climbs to $13\% / 8\% \approx 1.6$. **Trading less made more money.** The one-sentence intuition: *for a fast signal, net-of-cost performance is dominated by turnover, and the optimal strategy almost always trades less than the raw signal wants to.*

This is why "what is the turnover?" is the very first question an experienced PM asks about a new signal — often before "what is the Sharpe?". A gross Sharpe quoted without a turnover number is a half-truth.

## Signal decay and half-life: how fast does the edge go stale?

The last question in the battery: once your signal fires, how long does its predictive power last? This sets your holding period, which sets your turnover, which — as we just saw — sets your net return. Decay ties the whole post together.

### Measuring decay

To measure decay, compute the IC of your signal not just against tomorrow's return, but against the return $h$ days ahead, for a range of horizons $h = 1, 2, 3, \dots$. Plot IC versus horizon. A signal with real, fading predictive power shows an IC that starts positive and decays toward zero. Many signals decay roughly **exponentially**:

$$
\text{IC}(h) = \text{IC}(0) \cdot e^{-h / \tau},
$$

where $\tau$ is a decay constant. The **half-life** is the horizon at which the IC falls to half its starting value:

$$
t_{1/2} = \tau \ln 2.
$$

A short half-life (a day or two) means a fast signal you must trade aggressively to capture before it fades — high turnover, heavy cost sensitivity. A long half-life (weeks or months) means a patient signal you can hold cheaply.

![A signal-decay chart showing IC at horizon h falling exponentially from 0.05 at one day toward zero by day twenty, with green and amber bars under the curve and a dashed vertical line marking the half-life at about 3.5 days where the edge halves.](/imgs/blogs/evaluating-alpha-signals-ic-sharpe-turnover-quant-research-8.png)

The figure above shows a signal whose one-day IC of 0.05 decays exponentially. The green bars are the first three days where the signal is freshest; the amber bars mark the fade. The dashed line marks the **half-life at about 3.5 days** — the horizon where the predictive power has halved. After roughly two weeks, the IC is statistical noise. *A short half-life is a warning that you will pay a lot of turnover cost to capture this edge before it evaporates.*

#### Worked example: fitting a signal-decay half-life

You measure your signal's IC at several forward horizons:

| Horizon $h$ (days) | IC$(h)$ |
| --- | --- |
| 1 | 0.050 |
| 2 | 0.041 |
| 3 | 0.034 |
| 5 | 0.022 |
| 10 | 0.0075 |

**Step 1 — confirm it looks exponential.** Each step down is roughly a constant *ratio*, not a constant difference — the hallmark of exponential decay. From $h=1$ to $h=5$ (four days later) the IC fell from 0.050 to 0.022, a factor of $0.022 / 0.050 = 0.44$.

**Step 2 — solve for the decay constant $\tau$.** We have $\text{IC}(5)/\text{IC}(1) = e^{-(5-1)/\tau} = e^{-4/\tau} = 0.44$. Take logs: $-4/\tau = \ln(0.44) = -0.82$, so $\tau = 4 / 0.82 = 4.9$ days.

**Step 3 — compute the half-life:**

$$
t_{1/2} = \tau \ln 2 = 4.9 \times 0.693 = 3.4 \text{ days}.
$$

**Step 4 — sanity check against the data.** Does the IC halve in about 3.4 days? Starting at 0.050, half is 0.025. The table shows IC = 0.022 at $h=5$ and 0.034 at $h=3$; interpolating, IC = 0.025 lands at roughly $h = 3.5$. The fit checks out. The one-sentence intuition: *the half-life converts a decay curve into a single number — how long you have before the edge is half gone — which directly tells you how often you must re-trade.*

A signal with a 3.4-day half-life and 5 bps round-trip costs is a delicate thing: trade too often and costs win, trade too rarely and the edge has decayed before you act. Matching holding period to half-life is a core part of turning a signal into a strategy.

## Tear sheets: reading the whole battery at once

In practice, researchers do not eyeball one metric at a time. They generate a **tear sheet** — a standardized one-page report of every metric for a signal. The open-source `Alphalens` library (and its successors) popularized a particular layout, and "give me the Alphalens tear sheet" is a phrase you will hear constantly on a quant desk.

The centerpiece of a tear sheet is the **quantile return plot**. You sort all stocks into buckets (quintiles, deciles) by signal strength each day, then track the mean forward return of each bucket.

![A bar chart of mean forward returns by signal quintile, Alphalens-style, showing a monotone staircase from Q1 at minus 0.45 bps per day through Q5 at plus 1.10 bps per day, with the loss region red and the gain region green, annotated Q5 minus Q1 spread equals 1.55 bps per day.](/imgs/blogs/evaluating-alpha-signals-ic-sharpe-turnover-quant-research-9.png)

The figure above is the picture of a healthy signal: a **monotone staircase** rising from Q1 (weakest signal, worst returns) to Q5 (strongest signal, best returns). The bottom quintile loses 0.45 bps a day, the top makes 1.10, and the **Q5-minus-Q1 spread** of 1.55 bps a day is the signal's clean long-short edge. Monotonicity matters as much as the spread: a signal where Q5 is great but Q3 beats Q4 is suspicious — it suggests the edge lives in the extremes only, or is an artifact. *A real signal sorts stocks into a smooth staircase, not a jumble.*

A full tear sheet stacks this with: the IC time series and its mean, the cumulative long-short return curve, turnover by quantile, and the IC decay curve. Reading one fluently — knowing which panel reveals which failure mode — is exactly the skill a take-home case is testing.

Here is a minimal sketch of how a researcher computes the core of a tear sheet in Python, to make the machinery concrete:

```python
import numpy as np
import pandas as pd

  # df has columns: date, asset, signal, fwd_ret  (fwd_ret already lagged correctly)

def daily_rank_ic(group):
    # Spearman IC = Pearson correlation of the ranks
    return group["signal"].rank().corr(group["fwd_ret"].rank())

ic_by_day = df.groupby("date").apply(daily_rank_ic)
mean_ic   = ic_by_day.mean()
ic_ir     = ic_by_day.mean() / ic_by_day.std()      # consistency of the IC

def quantile_returns(group, q=5):
    buckets = pd.qcut(group["signal"], q, labels=False, duplicates="drop")
    return group["fwd_ret"].groupby(buckets).mean()

  # average the per-day quantile means across all days -> the staircase
quantile_curve = (df.groupby("date")
                    .apply(quantile_returns)
                    .groupby(level=1).mean())

spread = quantile_curve.iloc[-1] - quantile_curve.iloc[0]  # Q5 - Q1
print(f"mean IC = {mean_ic:.4f}, IC-IR = {ic_ir:.2f}, Q5-Q1 = {spread*1e4:.2f} bps")
```

The code is short because the *ideas* are short. The hard part is never the arithmetic — it is lining up the forward returns without look-ahead, estimating breadth honestly, and charging realistic costs.

## In the interview room and the take-home

Quant research interviews and take-home cases lean on this material constantly, because it cleanly separates people who have computed these numbers from people who have only read about them. Here are five fully-solved problems in the style you will actually face.

#### Worked example: "Our signal has a 0.10 IC. Why am I unimpressed?"

A candidate proudly reports a rank IC of **0.10** — far above the 0.03 we called "good". The interviewer is skeptical. Why?

Three reasons, in order of how damning they are. **First, breadth.** A 0.10 IC on a universe of 8 stocks gives $\text{IR} = 0.10 \times \sqrt{8} = 0.28$ — mediocre. The high IC bought you nothing because there is no breadth. **Second, look-ahead bias.** ICs above ~0.08 sustained on liquid equities are rare enough that the first hypothesis is a data leak — perhaps the forward return was not lagged correctly, or the signal secretly contains same-day information. **Third, capacity.** If the 0.10 IC lives in tiny illiquid names, transaction costs and market impact will erase it the moment real money trades. The lesson: *a high IC is a reason for suspicion, not celebration, until you have confirmed breadth, no look-ahead, and tradeable capacity.* This is the single most common "gotcha" and the answer that separates rehearsed candidates from real ones.

#### Worked example: "Rank these three signals."

You are handed three signals and must say which is best:

| Signal | Mean IC | Effective breadth | Daily turnover | Half-life |
| --- | --- | --- | --- | --- |
| A | 0.06 | 50 | 80% | 1.5 days |
| B | 0.03 | 1,200 | 15% | 12 days |
| C | 0.04 | 400 | 35% | 5 days |

Compute the gross IR for each via the fundamental law: A is $0.06 \times \sqrt{50} = 0.42$; B is $0.03 \times \sqrt{1200} = 1.04$; C is $0.04 \times \sqrt{400} = 0.80$.

So by gross IR, **B > C > A**. Now overlay turnover and half-life. Signal A has a high IC but tiny breadth *and* punishing turnover (80% daily) against a 1.5-day half-life — it will be crushed by costs and is the clear loser despite the highest per-bet IC. Signal B has modest IC but huge breadth, low turnover, and a long half-life — it is patient, cheap, and scalable: the winner. Signal C is a solid middle. The lesson: *breadth and turnover usually dominate IC in deciding a signal's real value; rank by information ratio first, then penalize for cost intensity.*

#### Worked example: "A 1.5 Sharpe strategy lost 40%. Was the Sharpe wrong?"

A strategy reported a 1.5 annualized Sharpe over five years, then lost 40% in a single month. Was the Sharpe a lie?

Not necessarily — it was *incomplete*. Sharpe measures return per unit of standard deviation, and standard deviation badly understates **tail risk**. A strategy that quietly sells out-of-the-money options or shorts volatility earns small, smooth profits most of the time (high Sharpe) while accumulating exposure to a rare catastrophe that standard deviation never sees coming. The 40% month is the fat tail Sharpe is blind to. The fix is to *pair* Sharpe with drawdown, with a tail measure like the worst monthly return or the Sortino ratio, and with a look at the strategy's *exposure* (is it short gamma?). The lesson: *Sharpe is a necessary but not sufficient statistic; a high Sharpe with a hidden fat tail is precisely the profile of a strategy that blows up.* Interviewers ask this to see whether you worship Sharpe or understand its blind spot.

#### Worked example: "Size this signal's net Sharpe at two cost levels."

A signal has a gross Sharpe of 1.8 at 12% gross annual return (so 6.7% gross volatility) on a \$50,000,000 book, with 25% daily turnover. Compute net Sharpe at (a) 4 bps and (b) 12 bps one-way cost.

Annual one-way dollar turnover: $0.25 \times \$50{,}000{,}000 \times 252 = \$3{,}150{,}000{,}000$. **(a) At 4 bps:** cost $= \$3.15\text{B} \times 0.0004 = \$1{,}260{,}000$, a drag of $\$1.26\text{M} / \$50\text{M} = 2.52\%$. Net return $= 12\% - 2.52\% = 9.48\%$; net Sharpe $= 9.48 / 6.7 = 1.41$. **(b) At 12 bps:** cost triples to $\$3.78\text{M}$, a drag of 7.56%. Net return $= 12\% - 7.56\% = 4.44\%$; net Sharpe $= 4.44 / 6.7 = 0.66$. The same signal is fundable at 4 bps and marginal at 12 bps. The lesson: *cost assumptions are not a detail — tripling the cost estimate cut the net Sharpe by more than half, so the realism of your cost model determines whether the strategy is real.* This is why desks obsess over transaction-cost analysis.

#### Worked example: "Match holding period to half-life."

A signal has a 4-day half-life and a gross edge of 1.0 bps per day of holding, against 8 bps round-trip cost. Should you hold for 1 day or 8 days?

Model the captured edge over a holding period $H$ as roughly the integral of the decaying signal, and the cost as one round-trip (8 bps) amortized over $H$ days. **Hold 1 day:** you capture about the full 1.0 bps but pay 8 bps that day — net $\approx 1.0 - 8 = -7$ bps. Disaster. **Hold 8 days (two half-lives):** the edge fades but you still capture meaningful cumulative return — roughly $1.0 \times (1 + 0.71 + 0.50 + 0.35 + \dots) \approx 2.9$ bps of cumulative edge — while paying 8 bps once over 8 days, i.e. 1 bps/day amortized. Net $\approx (2.9 - 8)/8 \approx -0.6$ bps/day if you held a single position, but because positions overlap and you only pay the round-trip on the *changed* fraction, the realistic net turns positive at a holding period of several days. The precise optimum balances decayed edge against amortized cost — and for a 4-day half-life with 8 bps costs, it lands at roughly **4 to 6 days**, near the half-life. The lesson: *the optimal holding period scales with the signal's half-life; trading much faster than the half-life guarantees costs win.* Interviewers use this to check whether you can connect decay, turnover, and cost into a single decision.

## Common misconceptions

A few beliefs that sound right and will cost you an interview or a strategy.

**"A higher Sharpe is always better."** Sharpe-worship is the classic error. A high Sharpe with a hidden fat tail (short-volatility strategies, carry trades) is a time bomb. A high Sharpe on a short sample is noise — recall the 15 Sharpe we computed from one good week. A high Sharpe that ignores costs is fiction. Sharpe is one number in a battery; treat it as a screen, not a verdict, and always pair it with drawdown and a tail check.

**"A high IC means a good signal."** As the interview problems showed, IC without breadth is nearly worthless, and an IC that is *too* high is usually a bug (look-ahead bias) or a trap (illiquid names you cannot trade). The fundamental law says the per-bet IC matters only in combination with $\sqrt{\text{breadth}}$. A 0.02 IC across thousands of independent bets beats a 0.15 IC across five.

**"My backtest's gross return is the return I will earn."** Gross return is what you make in a frictionless universe that does not exist. The only number that pays you is *net of all costs* — spread, fees, and market impact, all scaled by turnover. A fast signal can show a brilliant gross Sharpe and lose money net. Quote a Sharpe without a turnover figure and you have told half the story.

**"In-sample metrics tell me what the signal will do."** Every metric computed on the same data you used to *build* the signal is optimistically biased — you fit to the noise. A signal will always look better in-sample than out-of-sample. The only metrics worth trusting are computed on data the model never saw: a held-out period, a walk-forward test, or genuinely live trading. An in-sample Sharpe of 3 that becomes 0.5 out-of-sample was overfit, not unlucky.

**"More turnover means more alpha."** Turnover is a cost, not a benefit. Trading more captures a fast-decaying signal sooner, but past the optimum, additional turnover only feeds the broker. The right turnover is the one that maximizes *net* return, which is almost always lower than what the raw signal demands.

**"Drawdown and volatility measure the same risk."** They do not. Volatility (and Sharpe) treat upside and downside symmetrically and assume a tidy distribution. Drawdown is path-dependent and captures the specific sequence of losses an investor lived through. A strategy can have low volatility and a brutal drawdown if its losses cluster. Allocators care about drawdown because it is the risk that actually gets strategies shut down.

## How it shows up in real research

These metrics are not academic. They drive real decisions, real funding, and real blow-ups.

![The metric battery as a matrix, with rows for IC, IR, Sharpe, Calmar, turnover, and half-life, and columns for what it measures, its units, a good value, and its characteristic failure mode, color-coded green for good values and red for failure modes.](/imgs/blogs/evaluating-alpha-signals-ic-sharpe-turnover-quant-research-10.png)

The matrix above is the cheat sheet: each row is a metric, each column tells you what it measures, its units, a good value, and — crucially — its failure mode. Keep it in your head and you can interrogate any signal in five minutes.

**The 2007 quant quake.** In early August 2007, many statistical-arbitrage equity strategies — built on exactly the kind of low-IC, high-breadth signals this post describes — lost double-digit percentages in a few days, then largely recovered. The cause was crowding: too many funds held the *same* positions implied by the *same* signals, so when one large fund deleveraged, it pushed prices against everyone holding those positions, forcing more deleveraging. The lesson for signal evaluation is that *breadth can be an illusion*: 1,000 positions that everyone else also holds are not 1,000 independent bets. The fundamental law's breadth term silently assumes your bets are independent of *other people's* bets too, and in a crisis that assumption breaks.

**Why momentum desks obsess over turnover.** Cross-sectional momentum — buy recent winners, short recent losers — has a real, decades-documented IC. But naive momentum has high turnover (winners and losers rotate constantly), and the academic "paper" returns famously shrink once realistic costs are charged. Practitioners spend enormous effort on *execution*: trading patiently, netting offsetting trades, and accepting slightly stale positions to cut turnover. The gap between the published gross Sharpe and the achievable net Sharpe is, almost entirely, the turnover-times-cost term from this post.

**Volatility-selling strategies and the Sharpe trap.** Strategies that systematically sell options or short volatility (the infamous "short-vol" trade, including the products that imploded in the February 2018 "Volmageddon") show beautiful Sharpe ratios for years — small, steady premiums collected with low measured volatility. Then a single spike erases years of gains in days. Every one of these episodes is the fat-tail blind spot of the Sharpe ratio playing out in the real world. A researcher who only looked at Sharpe would have rated these strategies highly right up to the catastrophe; one who looked at drawdown, tail risk, and the strategy's short-gamma exposure would have seen the danger.

**Capacity and signal decay at scale.** A fast signal with a one-day half-life might post a stellar Sharpe on \$10 million but be impossible to run on \$1 billion — the market impact of trading a billion dollars fast enough to capture a one-day edge would swamp the edge itself. This is why large funds favor signals with longer half-lives and lower turnover: those scale. Decay and turnover are not just risk metrics; they set the **capacity** of a strategy — the most money it can manage before its own trading destroys its edge. Evaluating a signal without asking "how much money can this hold?" is evaluating only half of it.

**Combining signals and the breadth dividend.** Real funds rarely trade one signal; they blend dozens. The reason is, again, the fundamental law. If two signals each have a 0.03 IC and are *uncorrelated*, combining them raises the effective IC of the blend without sacrificing breadth — diversification across signals is the same $\sqrt{N}$ magic applied to ideas instead of stocks. Much of the day-to-day work of a quant researcher is finding new, *low-correlation* signals, because a fresh uncorrelated 0.02-IC signal can be worth more to a portfolio than improving an existing signal from 0.03 to 0.04.

## When this matters and where to go next

If you are preparing for a quant researcher interview, this battery is the most reliable territory to master cold. You should be able to, on a whiteboard: compute a rank IC on a six-stock cross-section; state and apply the fundamental law; annualize a Sharpe and explain why the short-sample version is a mirage; compute a maximum drawdown in dollars; turn a gross Sharpe into a net Sharpe given turnover and cost; and fit a decay half-life. Every one of those appeared above with real numbers — redo them yourself until the arithmetic is automatic, because under interview pressure you want to be thinking about the *judgment*, not the calculation.

The deeper skill the battery teaches is intellectual honesty. Each metric exists to catch a different way you might be fooling yourself: IC catches "is it even real", breadth catches "can I scale it", Sharpe catches "how risky", drawdown catches "how painful", turnover catches "what does it cost", and half-life catches "how fast does it die". A good researcher runs the whole battery *adversarially* — actively trying to find the metric that kills their beloved signal — because the market will find it for them if they do not.

To go further, the natural next steps connect these evaluation tools to the modeling tools that produce the signals in the first place. Building the underlying predictive models rests on [linear regression from first principles](/blog/trading/quantitative-finance/linear-regression-deep-quant-interviews), the single most-used tool for estimating betas and factor exposures. Understanding why breadth is so often an illusion — why your "independent" bets correlate, especially in a crisis — runs straight into [covariance, correlation, and their pitfalls](/blog/trading/quantitative-finance/covariance-correlation-pitfalls-quant-interviews). Sizing those bets once you trust a signal is the domain of the [Kelly criterion and sequential betting](/blog/trading/quantitative-finance/kelly-criterion-sequential-betting-quant-interviews). And the broader frame — how to weigh an edge against its risk when the future is uncertain — is the subject of [decision-making under uncertainty](/blog/trading/quantitative-finance/decision-making-under-uncertainty-quant-interviews). Master the battery here, then connect it to those, and you have the full arc from "I have a number" to "I have a fundable strategy".
