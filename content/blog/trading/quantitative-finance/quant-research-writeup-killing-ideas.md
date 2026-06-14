---
title: "The quant research write-up: presenting results and knowing when to kill an idea"
date: "2026-06-14"
publishDate: "2026-06-14"
description: "A from-scratch guide to the real output of a quant researcher: a decision to trade an idea or kill it, communicated honestly. Covers the anatomy of a strong write-up, robustness and sensitivity checks, the four kill criteria, presenting uncertainty with error bars, reproducibility, and five fully solved interview and take-home problems with real dollar figures."
tags:
  [
    "quant-research",
    "research-writeup",
    "kill-criteria",
    "backtesting",
    "robustness",
    "out-of-sample",
    "capacity",
    "reproducibility",
    "quant-interviews",
    "intellectual-honesty",
  ]
category: "trading"
subcategory: "Quantitative Finance"
author: "Hiep Tran"
featured: true
readTime: 44
---

> [!important]
> **TL;DR** — a quant researcher's real product is not a model or a backtest. It is a *decision* — trade this idea or kill it — written up so honestly that a portfolio manager can bet real money on your judgment. The write-up, the kill criteria, and the intellectual honesty are graded as heavily as the math.
>
> - **The output is a decision, not a result.** "The Sharpe was 1.3" is not an answer. "Trade it at $20M, expect $1.0M/yr net, kill it if the 6-month rolling Sharpe drops below 0.3" is an answer. A researcher who only reports numbers has not finished the job.
> - **A false positive in production is far more expensive than a false negative.** Killing a real edge costs you a few weeks you can always revisit; trading a fake edge bleeds capital, burns capacity, and erodes the trust you need to get your *next* idea funded. When in doubt, kill.
> - **Four hard gates kill most ideas:** no out-of-sample edge, fragile to parameters, no capacity, or too correlated with what the book already trades. Any one red gate is a kill, no matter how beautiful the in-sample Sharpe.
> - **Report uncertainty, not point estimates.** A Sharpe of 1.0 means nothing without its error bar. If the 95% confidence band runs from −0.1 to 2.1, you have *no* statistically significant edge yet — you have a hint.
> - **The number to remember:** a marginal signal that correlates **0.85** with momentum, your existing book's biggest sleeve, keeps only about **15%** of its standalone value after you hedge out the overlap — which is usually not enough to clear the bar.

Here is a scene that plays out in every quant interview and on every research desk. A young researcher has spent three weeks on an idea. The backtest looks gorgeous: a Sharpe ratio of 1.8, a smooth equity curve climbing up and to the right, a 12% annual return. They walk into the review, put the chart on the screen, and say, "It works." And the senior researcher leans back and asks the only question that matters: *"Would you bet your own bonus on it? And how would you know if you were wrong?"*

That gap — between a backtest that *looks* good and an idea you would actually trade — is the entire subject of this post. Almost everyone can run a backtest. What separates a quant researcher who gets hired and promoted from one who does not is the ability to take a pile of numbers and turn it into a *defensible decision*: trade it, at this size, with these risks, monitored this way — or kill it, here is exactly why, and here is what I learned. Firms like Two Sigma, Citadel, D. E. Shaw, and AQR do not hire people to produce backtests. They hire people to produce *good decisions about capital*, written up so that someone else can act on them.

![The eight sections of a quant research write-up arranged as a path from hypothesis through data, method, and results, with robustness, risks, and capacity gating the final trade-or-kill decision](/imgs/blogs/quant-research-writeup-killing-ideas-1.png)

The diagram above is the mental model for everything that follows. A research write-up walks one path: from a stated *hypothesis* (why should this edge exist?), through the *data* and *method*, to the *results*, and then — crucially — through three gates that most ideas fail: *robustness*, *risks*, and *capacity*. Only after all of that comes the one thing the reader actually wants: the *decision*, stated in plain words. We will build each box from zero, ground every concept in a worked example with real dollars, and finish with an interview-room section containing five fully solved problems of the kind you will see in a take-home or on a whiteboard.

This is educational, not financial advice. The goal is to make the reasoning a research desk uses feel obvious — not to tell you to trade anything.

## Foundations: the building blocks of a research decision

Before we can talk about *good* write-ups, we need a precise, shared vocabulary. None of it assumes prior finance knowledge — just careful definitions. Read this section top to bottom; everything later leans on it.

### What a "signal" and an "edge" actually are

A *signal* (also called an *alpha signal* or a *factor*) is a number you compute for each tradable asset at each point in time that you believe predicts its future return. If you compute, for every stock each morning, "the percentage by which its price fell over the last five days," that is a signal — and the bet that stocks which fell will bounce back is the *mean-reversion* hypothesis. A signal is just a column of numbers; whether it is *useful* is the open question.

An *edge* is a signal that genuinely predicts returns after costs — a real, repeatable reason you make money. The whole job is separating real edges from signals that only *looked* predictive in your historical sample by luck. That separation is hard precisely because, with enough tries, random noise will always throw up a few signals that fit the past beautifully and predict the future not at all.

### Backtest, in-sample, out-of-sample

A *backtest* is a simulation: you take your signal, apply your trading rules to historical data, and compute what your profit-and-loss (*PnL*) would have been. The stretch of history you used to *build and tune* the strategy is the *in-sample* (IS) period — also called the *training* set. The stretch you deliberately held back and never looked at while building is the *out-of-sample* (OOS) period — also called the *test* or *holdout* set.

This distinction is the beating heart of quant research. Any strategy can be made to look perfect in-sample, because you tuned it to fit that data. The only honest test of an edge is how it performs on data it has never seen. A researcher who reports only in-sample results is, knowingly or not, reporting a number that is almost guaranteed to overstate the truth.

### The two summary numbers: Sharpe and IC

The *Sharpe ratio* is the single most-quoted number in quant research. It measures return per unit of risk. If a strategy earns an average excess return (return above cash) of $\mu$ per year and has an annual standard deviation of returns (its *volatility*, a measure of how much the returns bounce around) of $\sigma$, then

$$\text{Sharpe} = \frac{\mu}{\sigma}.$$

A Sharpe of 1.0 means that, in an average year, your return equals one standard deviation of your year-to-year swings. Loosely, higher is better: a Sharpe of 0.5 is mediocre, 1.0 is a respectable single signal, 2.0 is excellent, and anything you compute above 3 in a backtest should make you *suspicious*, not happy — it usually means a bug or a leak, not a goldmine.

The *information coefficient* (IC) is the correlation between your signal's predictions and the actual subsequent returns, measured across all assets at each date and then averaged over time. (*Correlation* is a number between −1 and +1 measuring how well two things move together; +1 is perfect agreement, 0 is no relationship, −1 is perfect disagreement.) A daily IC of 0.03 sounds tiny but is actually a strong signal in liquid equities, because you are making thousands of small bets and a 3% edge on each compounds.

### The audience: who reads a write-up, and what they need

A write-up is never written for you. It is written for three audiences, and a strong one serves all three:

| Reader | What they care about | The question they ask |
|---|---|---|
| **Portfolio manager (PM)** | Will this make money net of costs, and how much? | "How many dollars, at what risk, and when do I cut it?" |
| **Risk** | How can this blow up, and how correlated is it with the book? | "What is the worst case, and does it add or duplicate risk?" |
| **Research peers** | Is the method sound, and is the result real? | "Did you avoid look-ahead, p-hack, or overfit? Can I reproduce it?" |

The PM wants a *sized, costed decision*. Risk wants the *downside and the correlation*. Peers want the *method and the reproducibility*. If your write-up answers only "the Sharpe was high," you have served none of them.

### Look-ahead bias and point-in-time data

*Look-ahead bias* is the cardinal sin of backtesting: using, at a simulated decision time, information that would not actually have been available then. The classic example: backtesting a strategy on today's S&P 500 membership list applied to ten years ago — but ten years ago the index contained different companies, and the ones that got *removed* were often the ones that did badly. Using today's list quietly deletes the losers from your history. This is *survivorship bias*, one flavor of look-ahead.

*Point-in-time* (PIT) data is the cure: data stamped with exactly what was known *as of* each historical date — the index membership as of that day, the earnings figure as first reported (not later revised), the universe of stocks that actually existed and traded then. A backtest on PIT data is honest about what you could have known. A backtest on "latest" data is usually flattering and wrong.

### The research lifecycle: most ideas die

Before any single write-up, it helps to see where a write-up sits in the larger flow of research. An idea is not "tested once and shipped." It funnels through a sequence of gates, and at *every* gate the most likely exit is not "advance" — it is "kill."

![The research lifecycle as a funnel where an idea passes through in-sample testing, out-of-sample testing, robustness, and research review, with a kill branch leaving at each gate and only a survivor reaching production with small monitored sizing](/imgs/blogs/quant-research-writeup-killing-ideas-2.png)

The lifecycle above shows the funnel. An idea with a reason enters, runs through an in-sample test, then an out-of-sample test, then robustness checks, then a research review with peers and risk — and only a survivor reaches production, sized small and monitored. The dashed branches all lead to the same place: a *kill*, documented and filed so the work is not wasted. On a real desk the funnel is steep — for every idea that reaches production, many more exit through a kill branch. The write-up is what you produce at each gate, whether the idea advances or dies. With this map and the vocabulary in place, we can build the real thing.

## The asymmetric cost of a false positive

Why is so much of this post about *killing* ideas rather than celebrating them? Because the two ways to be wrong are not equally expensive, and a good researcher internalizes the asymmetry deep in their bones.

There are two errors you can make. A *false positive* is trading an idea that has no real edge — you concluded "trade" when the truth was "kill." A *false negative* is killing an idea that did have a real edge — you concluded "kill" when the truth was "trade." Statisticians call these Type I and Type II errors; the [hypothesis testing post](/blog/trading/quantitative-finance/hypothesis-testing-pvalues-quant-interviews) develops that framing in depth. On a research desk, the two errors have wildly different price tags.

![A before-and-after comparison showing that a false positive shipped to production deploys capital on a zero edge and bleeds 1.5 million dollars, while an over-cautious false negative only shelves the idea with a write-up at the cost of research time](/imgs/blogs/quant-research-writeup-killing-ideas-4.png)

The figure above lays the two costs side by side. When a false positive *ships* — you trade a fake edge — real things break. You deploy capital on a strategy with zero true alpha. Trading costs and the absence of edge produce a real drawdown; the figure shows a representative −$1.5M over a year on a modestly sized book. You consume *capacity* and *risk budget* that a real idea could have used. And worst of all, you erode the PM's trust, which is the currency you need to get your next idea funded. A researcher who ships two false positives in a year may not get a third at-bat.

When a false negative happens — you over-kill, shelving an idea that would have worked — the cost is bounded and recoverable. You lose the *research time* you spent. No capital is lost; there is no drawdown. And the idea is not gone forever — it sits in your kill file with a write-up, and you can revisit it next year when you have more data, a cleaner universe, or a better way to trade it. Many strategies that desks trade today were killed once and resurrected.

#### Worked example: pricing the two errors in dollars

You manage a $20,000,000 sleeve and you are deciding whether to deploy a candidate signal whose *true* edge you are unsure of.

- **Case A — you trade a fake edge (false positive).** Suppose the signal's true alpha is zero, but trading it costs you 0.5% per year in transaction costs and slippage on the $20M, plus a typical −1.0% of bad luck in a year where you were exposed to a factor that went the wrong way. That is $0.5\% \times \$20\text{M} + 1.0\% \times \$20\text{M} = \$100{,}000 + \$200{,}000 = \$300{,}000$ of direct loss, plus the opportunity cost of the capacity and the reputational hit. Round the all-in cost to roughly **$300,000 and a dented reputation**.
- **Case B — you kill a real edge (false negative).** Suppose the idea would truly have earned a Sharpe of 0.8, worth maybe $0.8 \times 6\% \times \$20\text{M} \approx \$96{,}000$ in a year (using a 6% target volatility). You lose that *one year* of profit — but only until you reconsider. The recoverable cost is roughly **$96,000 of foregone profit, fully revisitable**, plus the three weeks you already spent.

The false positive is both larger in dollars *and* irreversible in reputation; the false negative is smaller and recoverable. **When the evidence is ambiguous, the asymmetry says kill.** That single sentence is the soul of disciplined quant research.

## The anatomy of a strong write-up

A research write-up is a document — often literally one page plus an appendix — that a reviewer can read in five minutes and act on. Its structure is not arbitrary. It mirrors the order in which a skeptical reader builds (or refuses to build) confidence.

![The one-page write-up read top to bottom as a pipeline: decision and hypothesis first, then data and point-in-time sourcing, method and assumptions, results with error bars, the robustness grid, and finally risks and capacity](/imgs/blogs/quant-research-writeup-killing-ideas-5.png)

The pipeline above is the reading order. Notice what comes *first*: the decision and the hypothesis, not the results. A reviewer should know, in the first two sentences, what you are recommending and why you ever thought it would work. Everything after that is evidence the reader uses to accept or reject your recommendation. Let us walk each section.

### 1. Hypothesis: why should this edge exist?

Every strong write-up opens with an economic or behavioral *reason* the edge should exist before showing a single number. "Stocks that just reported earnings drift in the direction of the surprise for several weeks because investors underreact to news" is a hypothesis. "I searched 500 signals and this one had the highest Sharpe" is not a hypothesis — it is a confession of data mining. A reviewer trusts a result far more when there is a story that predicted it, because a pre-registered reason is much harder to fake than a curve fit after the fact.

### 2. Data: what, from when, and point-in-time?

State your universe (which assets), your date range, the split between in-sample and out-of-sample, and — critically — whether the data is point-in-time. "US equities, top 1500 by market cap as of each date, 2010–2024, IS 2010–2019, OOS 2020–2024, point-in-time index membership and as-first-reported fundamentals" is a sentence that tells a reviewer you did the unglamorous work that separates real research from a toy.

### 3. Method: the signal, the rules, the assumptions

Describe how the signal is computed, how it becomes positions, and what you assumed about costs. The assumptions are where strategies quietly die or quietly cheat. Did you assume you can trade at the closing price (often unrealistic)? Did you include the bid-ask spread, commissions, and market impact? A 1.5 Sharpe before costs that becomes 0.4 after honest costs is a different idea — and the honest researcher reports the 0.4.

### 4. Results: in-sample *and* out-of-sample, with error bars

Report the Sharpe, the IC, the annual return, the worst drawdown — both in-sample and out-of-sample, side by side, *with* their uncertainty (we will spend a whole section on error bars). The single most informative number in the whole write-up is often the *ratio* of OOS Sharpe to IS Sharpe: if your in-sample Sharpe was 1.8 and your out-of-sample Sharpe is 0.3, the edge largely evaporated when it met fresh data, and that decay is the story.

### 5–7. Robustness, risks, capacity: the three gates

These are the sections juniors skip and seniors read first. *Robustness*: does the edge survive when you change the parameters and the time period? *Risks*: how does it lose, how badly, and in what regimes? *Capacity*: how many dollars can you actually deploy before your own trading moves the price and eats the edge? Each gets its own section below.

### 8. Decision: trade or kill, in plain words

End with the call, stated so plainly a non-technical reader gets it: *"Recommend trading at $15M initial, scaling to $40M if the live 3-month Sharpe stays above 0.5; kill immediately if the rolling 6-month Sharpe drops below 0."* Or: *"Recommend killing: the OOS Sharpe of 0.2 with a 95% band of [−0.4, 0.8] is not distinguishable from zero, and the edge does not survive the 2019–2020 subperiod."* A write-up without a clear decision is an unfinished write-up.

#### Worked example: structuring a one-page write-up from a backtest

You ran a backtest and got: IS Sharpe 1.4 (2012–2019), OOS Sharpe 0.9 (2020–2024), OOS IC 0.04, max drawdown −9%, turnover 180% per year, estimated capacity $40M. Here is the one-page write-up, in order:

1. **Decision (lead with it):** "Recommend trading at $20M, scaling to the $40M capacity if the live 6-month Sharpe holds above 0.6. Kill if rolling 6-month Sharpe falls below 0.2."
2. **Hypothesis:** "Short-term reversal: stocks that overshoot on high-volume days mean-revert over 3–5 days as liquidity providers get paid to absorb the flow."
3. **Data:** "Top 1500 US equities, point-in-time membership, 2012–2024, IS 2012–2019 / OOS 2020–2024."
4. **Method:** "Signal = negative of 3-day return, z-scored cross-sectionally; positions proportional to signal, dollar-neutral; costs = 5 bps per side including spread and impact."
5. **Results:** "IS Sharpe 1.4, OOS Sharpe 0.9 [95% band 0.4–1.4], OOS IC 0.04, max DD −9%, turnover 180%/yr."
6. **Robustness:** "Sharpe stays in 0.7–1.1 across lookbacks of 2–5 days and z-thresholds 1.0–2.0; positive in every two-year subperiod."
7. **Risks:** "Crowded trade; vulnerable to liquidity crises (March 2020 the edge briefly inverted); correlation to existing reversal sleeve = 0.45."
8. **Capacity:** "Net dollar alpha peaks near $40M; beyond that, impact costs dominate."

The one-sentence intuition: **a write-up is the decision first, then exactly the evidence a skeptic needs to either trust or break that decision.**

## Robustness and sensitivity: does the edge survive being poked?

A single backtest number is one draw from a noisy process. The question robustness answers is: *if I had made slightly different reasonable choices, would I have reached the same conclusion?* A real edge is insensitive to the dozens of small arbitrary decisions you made; a fragile, overfit one collapses the moment you nudge a parameter or change the window. Three checks do most of the work.

### Check 1 — parameter sensitivity: plateau, not spike

Every strategy has knobs: the lookback window, the entry threshold, the holding period. For a real edge, the performance surface across those knobs is a broad *plateau* — many nearby settings all work about equally well. For an overfit edge, it is a lonely *spike* — one magic setting works and its neighbors do not, which is the signature of a parameter tuned to fit noise.

![A robustness heatmap of out-of-sample Sharpe across a grid of lookback windows and z-score thresholds, showing a broad green plateau of values near 1.0 in the upper region fading to amber and red below 0.6 in the bottom-right corner](/imgs/blogs/quant-research-writeup-killing-ideas-6.png)

The heatmap above is exactly the check. Each cell is the out-of-sample Sharpe for one combination of lookback (rows) and entry threshold (columns). The wide green region — Sharpe between 0.8 and 1.2 across most of the grid — says the edge is real and not an artifact of one lucky setting. The amber-and-red bottom-right corner shows where longer lookbacks and tighter thresholds start to fail, which is information too, but the *plateau is the headline*. If instead only a single cell were green and everything around it were red, that would be a kill: you would have found a setting that fits the past, not an edge that predicts the future.

### Check 2 — subperiod stability: positive across regimes

Slice your out-of-sample period into calendar chunks — two-year windows, or by market regime — and recompute the Sharpe in each. A real edge is positive in most subperiods. An edge that earns all its money in a single window (one bull market, one crisis) is probably a property of *that regime*, not a repeatable strategy.

![Out-of-sample Sharpe sliced into five two-year calendar subperiods shown as a bar chart, with four green bars above zero ranging from plus 0.6 to plus 1.0 and one red bar at minus 0.4 for the 2019 to 2020 regime break](/imgs/blogs/quant-research-writeup-killing-ideas-11.png)

The bar chart above shows the subperiod test in action. Four of five subperiods are solidly positive (Sharpe +0.6 to +1.0), but 2019–2020 is red at −0.4: the edge inverted during a regime break. One red subperiod out of five is not automatically a kill, but it is a *flag* you must explain. Maybe the strategy is vulnerable to that specific regime and you can hedge it; maybe it reveals the edge is more fragile than the headline Sharpe suggests. What you must never do is quietly drop the bad subperiod and report the average of the good ones.

### Check 3 — universe variation: not a single-stock artifact

Recompute the result on variations of your asset universe: large-caps only, small-caps only, excluding the most volatile names, a different country. An edge that exists across many universes is structural. An edge that lives entirely in, say, the twenty smallest, least-liquid stocks — where you could never actually trade size — is a backtest artifact dressed up as alpha.

#### Worked example: building a robustness grid and reading it

You have a momentum signal with an out-of-sample Sharpe of 1.1 at your default settings (lookback 20 days, z-threshold 1.5). You build the grid in the heatmap above. Reading it:

- **Across z-thresholds at lookback 20:** Sharpe is 1.0, 1.1, 1.2, 0.9 for z = 1.0, 1.5, 2.0, 2.5. A smooth hump, all positive — **stable**.
- **Across lookbacks at z = 1.5:** Sharpe is 1.0, 1.1, 1.0, 0.7 for 10, 20, 40, 60 days. Gently declining but all positive — **stable**.
- **The corner:** the worst cell (lookback 60, z = 2.5) is 0.4 — still positive, just weaker.

Verdict: the minimum across the entire grid is 0.4 and the median is about 1.0. The edge is a plateau, not a spike — **it passes the parameter-sensitivity gate.** Now contrast a *failing* grid: default Sharpe 1.1, but every neighbor 0.1 or negative. Same headline number, opposite conclusion. The one-sentence intuition: **a real edge is a wide green plateau on the parameter grid; an overfit one is a single green cell in a sea of red.**

## The kill criteria: four ways an idea dies

Most ideas should be killed, and a disciplined researcher has explicit criteria so the decision is not a matter of mood. Four gates account for the vast majority of justified kills. Any single failed gate is enough — you do not get to average them.

![A decision tree starting from a new signal asking to trade, branching through four sequential gates — out-of-sample edge, parameter stability, capacity, and low book correlation — where each gate has a red kill branch and only an idea passing all four reaches the green trade leaf](/imgs/blogs/quant-research-writeup-killing-ideas-3.png)

The decision tree above is the kill logic. A new signal enters at the top and must pass four gates in sequence. Fail the out-of-sample test and you kill for *no edge*. Pass that but fail the parameter grid and you kill for *fragility*. Pass that but find no capacity and you kill for *unscalability*. Pass even that but discover the signal duplicates what the book already trades, and you kill for *redundancy*. Only a signal that survives all four reaches the green "trade" leaf — and even then, you size it small and watch it. Let us take each gate.

### Gate 1 — no out-of-sample edge

The most common and most decisive kill. If the in-sample Sharpe was 1.8 but the out-of-sample Sharpe is 0.2 with a confidence band straddling zero, the edge was a curve fit. The decay from IS to OOS is the tell. A useful rule of thumb: expect OOS Sharpe to be roughly half of IS Sharpe even for a *real* edge (because IS is always flattered by selection), so an IS Sharpe of 1.0 that you need to clear a 0.6 bar should worry you before you even run the OOS test.

### Gate 2 — fragile to parameters

Covered above: a spike instead of a plateau. If the result depends on a single magic setting, you have fit noise. Kill it, even if that one setting has a spectacular Sharpe.

### Gate 3 — no capacity

A strategy that works on $1M but not on $20M is, for most institutional desks, not worth the operational overhead. Capacity is the dollar amount you can deploy before your own trades move prices enough to eat the edge. We devote the next section to computing it. A 3-Sharpe strategy with $2M of capacity often loses to a 1-Sharpe strategy with $200M of capacity, because the firm makes money in *dollars*, not in Sharpe.

### Gate 4 — too correlated with the existing book

This is the gate juniors forget. A signal can have a real, robust, scalable edge *and still be a kill* if it largely duplicates a strategy the firm already trades. The firm does not get paid for the same risk twice. A new signal earns its risk budget only if it adds something the book does not already have.

![A correlation-to-book check shown as a grid where a new signal is compared against three existing book sleeves: momentum at correlation 0.85 marked red as mostly a copy keeping only 15 percent of its value, and value and carry at correlations 0.10 and 0.05 marked green as near-independent and additive, retaining 99 percent after hedging](/imgs/blogs/quant-research-writeup-killing-ideas-7.png)

The grid above shows the correlation check. The new signal correlates 0.85 with the book's momentum sleeve — they are nearly the same bet. After you hedge out the overlap (hold the new signal but subtract the part explained by momentum), only about 15% of its standalone value survives, because most of what it earned was just momentum you already owned. Against value (correlation 0.10) and carry (0.05), it is near-independent and keeps ~99% of its value after hedging. **A signal that correlates 0.85 with your biggest sleeve is usually a kill, even with a great standalone Sharpe.** The covariance and correlation traps that make this calculation subtle are developed in the [covariance and correlation pitfalls post](/blog/trading/quantitative-finance/covariance-correlation-pitfalls-quant-interviews).

#### Worked example: keep-or-kill from a metrics table

A reviewer hands you this table for a candidate signal and asks for a decision:

| Metric | Value | Threshold | Verdict |
|---|---|---|---|
| IS Sharpe | 1.6 | — | (reference only) |
| OOS Sharpe | 1.0, band [0.5, 1.5] | > 0.7 net | PASS |
| Parameter grid | plateau, min 0.6 | plateau, not spike | PASS |
| Capacity | $45M at peak net $1.2M/yr | > $20M | PASS |
| Correlation to book | 0.85 to momentum | < 0.5 | **FAIL** |

Three gates pass cleanly. But the correlation gate fails hard: at 0.85 to the momentum sleeve, this signal is mostly something the book already owns, and the marginal contribution after hedging is roughly $1.0 \times \sqrt{1 - 0.85^2} \approx 1.0 \times 0.53 \approx 0.53$ Sharpe of *new* risk-adjusted return — and that overstates it, because the part that survives hedging is also the noisiest part. **Decision: kill, on Gate 4.** Write it up honestly: "Real, robust, scalable edge — but 85% redundant with momentum; revisit only if we can find the orthogonal 15% and trade it cheaply." The one-sentence intuition: **a beautiful Sharpe does not survive a red on any single hard gate.**

## Capacity: the dollar gate that decides everything

Capacity deserves its own treatment because it is where Sharpe-obsessed juniors most often go wrong. A strategy does not have a single return number; it has a *return that shrinks as you put more money into it*, because your own buying pushes prices up and your own selling pushes them down. That self-inflicted cost is *market impact*, and it grows faster than linearly with size.

![A capacity curve plotting net dollar alpha per year on the vertical axis against capital deployed on the horizontal axis, rising from zero to a peak of 1.4 million dollars at 55 million dollars of capital then falling, with a green go zone to the left of the peak and a red no-go zone to the right](/imgs/blogs/quant-research-writeup-killing-ideas-12.png)

The capacity curve above is the whole story in one picture. Gross alpha (before impact) grows roughly in proportion to the capital you deploy — twice the money, twice the gross dollars. But market impact costs grow *faster* than the capital, because to trade twice the size you must accept worse prices. The *net* dollar alpha — gross minus impact — therefore rises, peaks, and then *falls*. The peak (here $1.4M/yr at $55M of capital) is the capacity. To the left of the peak (green), adding capital adds net dollars: GO. To the right (red), adding capital *loses* net dollars: NO-GO. Deploying past the peak makes you objectively worse off.

### Why dollars, not Sharpe, are the real objective

A firm's profit is measured in dollars, and dollars equal Sharpe times volatility times capital. A 3.0-Sharpe strategy you can only run at $2M produces about $3.0 \times 6\% \times \$2\text{M} = \$360{,}000$ a year (at 6% volatility). A 1.0-Sharpe strategy you can run at $200M produces $1.0 \times 6\% \times \$200\text{M} = \$12{,}000{,}000$ a year. The "worse" Sharpe makes 33× the money. This is why scalable mediocre edges beat brilliant tiny ones on most institutional desks, and why "what is the capacity?" is the question that ends more research meetings than any other.

#### Worked example: a capacity-based go/no-go decision

You have a strategy with these economics. Gross alpha is 4% of deployed capital per year. Market impact costs scale as $c(K) = 0.5\% \times (K / \$25\text{M})$ of capital — that is, at $25M the round-trip impact drag is 0.5% of capital, at $50M it is 1.0%, at $75M it is 1.5%, and so on (impact per dollar rises linearly with size). Net dollar alpha is:

$$\text{Net}(K) = \underbrace{4\% \times K}_{\text{gross}} - \underbrace{0.5\% \times \frac{K}{\$25\text{M}} \times K}_{\text{impact}}.$$

Let us compute net dollars at several sizes:

- **$25M:** gross $= 4\% \times 25 = \$1.0\text{M}$; impact $= 0.5\% \times 1 \times 25 = \$0.125\text{M}$; **net $= \$0.875\text{M}$.**
- **$50M:** gross $= \$2.0\text{M}$; impact $= 0.5\% \times 2 \times 50 = \$0.5\text{M}$; **net $= \$1.5\text{M}$.**
- **$75M:** gross $= \$3.0\text{M}$; impact $= 0.5\% \times 3 \times 75 = \$1.125\text{M}$; **net $= \$1.875\text{M}$.**
- **$100M:** gross $= \$4.0\text{M}$; impact $= 0.5\% \times 4 \times 100 = \$2.0\text{M}$; **net $= \$2.0\text{M}$.**
- **$125M:** gross $= \$5.0\text{M}$; impact $= 0.5\% \times 5 \times 125 = \$3.125\text{M}$; **net $= \$1.875\text{M}$.**

Net dollars rise to a peak around $100M (net $2.0M/yr) and then *fall* — at $125M you make less than at $100M. To find the exact peak, set the derivative of $\text{Net}(K)$ to zero: $0.04 - 2 \times \frac{0.005}{25} K = 0$, giving $K = 0.04 \times \frac{25}{0.01} = \$100\text{M}$. **Go/no-go: deploy up to ~$100M; do not exceed it.** Below $100M, every extra dollar adds net profit; above it, every extra dollar destroys it. The one-sentence intuition: **capacity is where the net-dollar curve peaks, and trading past it makes you poorer, not richer.**

#### Worked example: the marginal contribution to a $100,000,000 book

Now the hardest and most realistic version of the question. Your firm runs an existing $100,000,000 book with a Sharpe of 1.5. You have a *new* marginal signal with a standalone Sharpe of 1.0, and you must decide whether adding it clears the bar. The firm's rule: a new signal must add at least 0.1 to the *combined* portfolio Sharpe to be worth the operational cost and risk budget.

The combined Sharpe of two strategies depends on their correlation $\rho$. For two equally-weighted, equal-volatility strategies with Sharpes $S_1$ and $S_2$ and correlation $\rho$, the combined Sharpe is:

$$S_{\text{combined}} = \frac{S_1 + S_2}{\sqrt{2(1 + \rho)}}.$$

Let us test the new signal at two correlations to the existing book.

- **Correlation $\rho = 0.85$ (it duplicates momentum):**
$$S_{\text{combined}} = \frac{1.5 + 1.0}{\sqrt{2(1 + 0.85)}} = \frac{2.5}{\sqrt{3.7}} = \frac{2.5}{1.924} \approx 1.30.$$
Wait — that is *below* the existing 1.5. Adding a highly-correlated, lower-Sharpe signal at equal weight actually *drags the book down*. The correct comparison uses optimal weights, but even then the marginal Sharpe improvement is tiny: the new signal adds almost no independent information. **It fails the +0.1 bar — kill.**

- **Correlation $\rho = 0.10$ (it is near-independent):**
$$S_{\text{combined}} = \frac{1.5 + 1.0}{\sqrt{2(1 + 0.10)}} = \frac{2.5}{\sqrt{2.2}} = \frac{2.5}{1.483} \approx 1.69.$$
The combined Sharpe jumps from 1.5 to about 1.69 — an improvement of **+0.19**, comfortably above the +0.1 bar. **It clears — keep.**

Now translate to dollars. At 6% target volatility, a book Sharpe of 1.5 on $100M earns about $1.5 \times 6\% \times \$100\text{M} = \$9.0\text{M}$/yr. Lifting the Sharpe to 1.69 (the low-correlation case) and holding volatility at 6% raises expected profit to $1.69 \times 6\% \times \$100\text{M} \approx \$10.1\text{M}$/yr — a **marginal contribution of roughly $1.1M per year**. That clears almost any reasonable bar. The same signal at 0.85 correlation contributes *negative* marginal dollars at equal weight. **Same standalone Sharpe of 1.0, opposite decision — the correlation to the existing book is the deciding variable.** The one-sentence intuition: **a marginal signal is worth what it adds to the book, not what it earns alone, and correlation is the multiplier that sets the difference.**

## Presenting uncertainty honestly: error bars, not point estimates

A point estimate without an error bar is not evidence — it is a guess wearing a lab coat. The single most common way capable researchers mislead themselves and their reviewers is by reporting "Sharpe = 1.0" as if it were a fact, when the honest statement is "Sharpe = 1.0, but with this much data I can only say it is somewhere between −0.1 and 2.1 with 95% confidence."

![Two signals each with a point estimate Sharpe of 1.0 shown as bars with confidence bands: signal A in green whose 95% band runs from 0.5 to 1.5 clearing zero and labeled keep candidate, and signal B in red whose 95% band runs from minus 0.1 to 2.1 straddling zero and labeled not evidence yet](/imgs/blogs/quant-research-writeup-killing-ideas-10.png)

The figure above makes the point unforgettable. Two signals report the *identical* point estimate: Sharpe 1.0. But Signal A's 95% confidence band (0.5 to 1.5) sits entirely above zero — you can be confident it has a real, positive edge. Signal B's band (−0.1 to 2.1) straddles zero — for all the data can tell you, its true Sharpe might be slightly negative. Same headline number, completely different evidence. Reporting only the dot would let Signal B masquerade as Signal A.

### How big is the error bar on a Sharpe?

There is a clean approximation. The standard error of an estimated Sharpe ratio $\hat{S}$ measured over $N$ years of data (with daily or higher-frequency sampling, ignoring higher moments) is roughly:

$$\text{SE}(\hat{S}) \approx \sqrt{\frac{1 + \tfrac{1}{2}\hat{S}^2}{N}}.$$

The 95% confidence interval is then about $\hat{S} \pm 1.96 \times \text{SE}(\hat{S})$. The lesson lives in the $N$ in the denominator: you need a *lot* of data to pin down a Sharpe. This connects directly to the [estimators, bias, and variance material](/blog/trading/quantitative-finance/estimators-mle-bias-variance-quant-interviews), where the precision of an estimate is the whole game.

#### Worked example: is a Sharpe of 1.0 statistically significant?

You measured a Sharpe of 1.0 over **2 years** of data. Is it distinguishable from zero?

$$\text{SE} \approx \sqrt{\frac{1 + \tfrac{1}{2}(1.0)^2}{2}} = \sqrt{\frac{1.5}{2}} = \sqrt{0.75} \approx 0.87.$$

The 95% interval is $1.0 \pm 1.96 \times 0.87 = 1.0 \pm 1.70$, i.e. **[−0.70, 2.70]**. That band straddles zero by a mile — with only 2 years, a Sharpe of 1.0 is *not* statistically significant. You cannot rule out that the true edge is zero or even negative.

Now redo it with **10 years** of data:

$$\text{SE} \approx \sqrt{\frac{1.5}{10}} = \sqrt{0.15} \approx 0.387,$$

giving a 95% interval of $1.0 \pm 1.96 \times 0.387 = 1.0 \pm 0.76$, i.e. **[0.24, 1.76]**. Now the band clears zero — with 10 years, the same Sharpe of 1.0 *is* significant. **The point estimate did not change; the evidence did, entirely because of the sample size.** If you report only the dot, you hide exactly the information a reviewer needs to tell these two situations apart. The one-sentence intuition: **a Sharpe without its confidence band is a number, not evidence; always report the interval and check whether it clears zero.**

## Reproducibility: can a stranger re-run your result?

A result that only you can produce, on your laptop, with a script you cannot quite remember, is not a finding — it is an anecdote. Reproducibility is the discipline that turns a one-off number into evidence a desk can trust and a regulator can audit. It is also, brutally, where many beautiful backtests are revealed to be bugs.

![A reproducibility checklist arranged as a grid: fixed random seed, point-in-time data with no future leakage, pinned code and environment, frozen universe membership, logged parameters, a one-command rerun, saved outputs, and a bit-for-bit match on a clean machine](/imgs/blogs/quant-research-writeup-killing-ideas-8.png)

The checklist above is what "reproducible" concretely means. Six ingredients, one outcome:

- **Fixed random seed** — any randomness (train/test splits, bootstrap samples, model initialization) is pinned so every run produces the same split, not a different lucky one each time.
- **Point-in-time data** — as established in the foundations, no future information leaks into past decisions.
- **Pinned code and environment** — the exact git commit SHA and a locked list of library versions, so the code that produced the number still exists and still runs.
- **Frozen universe** — the asset universe is the as-of membership, not today's list.
- **Logged parameters** — every knob (lookback, threshold, cost assumption) is recorded with the result, not held in your head.
- **One-command rerun** — `make backtest` or one script regenerates everything from raw data to final Sharpe.

The outcome you are aiming for: a colleague checks out your commit on a *clean machine* and gets a **bit-for-bit identical** Sharpe, IC, and equity curve. If they cannot, the result is not yet trustworthy — and surprisingly often, the act of trying to reproduce it surfaces a look-ahead bug that was quietly inflating the backtest.

### Why reproducibility kills false positives

The discipline is not bureaucracy; it is a bug detector. The most common backtest inflators — using revised instead of as-reported data, applying today's index to yesterday, accidentally peeking at the test set during tuning, a random seed that cherry-picked a lucky split — are all caught the moment someone tries to reproduce the result point-in-time on a clean machine. A reproducible pipeline makes these mistakes *visible*, which is exactly why a desk insists on it before risking a dollar.

#### Worked example: a reproducibility audit catches a leak

A colleague reports a stunning Sharpe of 2.4 and asks you to deploy $30M. You run the reproducibility checklist. The seed is fixed (good), the code is pinned (good), but when you check the data, you find the signal uses each company's *annual revenue* dated to January 1 — even though that revenue was not *reported* until, on average, late February. The backtest "knew" each year's revenue about seven weeks before the market did.

You rebuild the signal point-in-time, lagging each fundamental to its actual first-report date, and re-run. The Sharpe drops from 2.4 to **0.3** — the entire edge was look-ahead bias. The dollars at stake: deploying $30M on the fake 2.4-Sharpe would have looked like an expected $2.4 \times 6\% \times \$30\text{M} \approx \$4.3\text{M}$/yr, but the real edge of 0.3 is worth $0.3 \times 6\% \times \$30\text{M} \approx \$0.5\text{M}$/yr — and after honest costs, likely nothing. **The reproducibility audit turned a $4.3M phantom into a justified kill, before any capital was lost.** The one-sentence intuition: **reproducibility is not paperwork; it is the cheapest way to catch the bug that would otherwise become a production drawdown.**

## In the interview room and the take-home

Quant researcher interviews at Two Sigma, Citadel, D. E. Shaw, and AQR increasingly center on *judgment*, not just math. A take-home gives you data and asks "is there an edge here, and would you trade it?" A live round hands you a metrics table and asks "keep or kill?" The graders are watching for the discipline this whole post is about: do you check OOS, do you report uncertainty, do you ask about capacity and correlation, and — above all — are you honest when the answer is "kill"? Here are five fully solved problems in that style. The broader decision-making toolkit behind them is in the [decision-making under uncertainty post](/blog/trading/quantitative-finance/decision-making-under-uncertainty-quant-interviews).

#### Worked example: "Your backtest shows Sharpe 2.5. Walk me through what you check before trusting it."

This is the most common opener, and the trap is to act *excited*. The strong answer is a checklist delivered calmly:

1. **Is it in-sample or out-of-sample?** A 2.5 in-sample means little; I want the OOS Sharpe and the IS-to-OOS decay. If OOS is 0.4, the edge is mostly overfit.
2. **What is the error bar?** Over how many years? With 2 years, even a 2.5 has a wide band. I compute $\text{SE} \approx \sqrt{(1 + 0.5 \times 2.5^2)/N}$; over 3 years that is $\sqrt{4.125/3} \approx 0.66$ wait — let me redo: $1 + 0.5 \times 6.25 = 4.125$, $\text{SE} = \sqrt{4.125/3} \approx 1.17$, so the band is $2.5 \pm 2.3$, i.e. [0.2, 4.8]. A 2.5 over only 3 years is *barely* significant — surprising, and worth saying out loud.
3. **Is it a plateau or a spike** across parameters? **Does it survive subperiods** and **universe variations**?
4. **Where did the costs go?** A 2.5 before costs that is 0.5 after honest impact is a different idea.
5. **Is it reproducible point-in-time?** A Sharpe above ~3 usually signals a leak, not a goldmine.

**Decision framing:** "I would not trust a 2.5 until I see OOS, the error bar, the robustness grid, and a point-in-time reproduction. My prior is that a 2.5 is a bug or a leak until proven otherwise." That answer — skepticism first — is what gets the offer.

#### Worked example: keep-or-kill from a scorecard

The interviewer shows you this scorecard for a candidate signal and asks for a one-word decision plus justification.

![A keep-or-kill scorecard with three gates: OOS Sharpe of 1.0 against a threshold above 0.7 marked pass in green, parameter stability showing a plateau marked pass in green, and book correlation of 0.85 against a threshold below 0.5 marked kill in red](/imgs/blogs/quant-research-writeup-killing-ideas-9.png)

The scorecard above is the figure to reason from. Two gates pass: OOS Sharpe 1.0 (above the 0.7 net bar) and a stable parameter plateau. But book correlation is 0.85 against a 0.5 threshold — a hard fail.

**Decision: KILL.** Justification: "The standalone economics are genuinely good — a robust 1.0 OOS Sharpe on a plateau. But at 0.85 correlation to our momentum sleeve, this is mostly risk we already own. After hedging out the overlap it retains roughly $\sqrt{1 - 0.85^2} \approx 53\%$ of its standalone Sharpe in the best case, and the surviving piece is the noisiest part, so the true marginal contribution is well under that. It fails our correlation gate. I would file it and revisit only if we can isolate and cheaply trade the orthogonal component." The grader is checking whether you let a pretty Sharpe override a hard gate. You did not.

#### Worked example: the take-home — "Here is 10 years of data. Is there an edge?"

A take-home hands you a CSV and freedom. The graded behaviors:

1. **Split first, before looking.** Carve out the last 3 years as OOS and *do not touch it* while building. State this in your write-up.
2. **Form a hypothesis before mining.** "I will test short-term reversal because high-volume overshoots tend to revert." A stated hypothesis beats "I tried 50 things and kept the best."
3. **Report IS and OOS side by side with error bars.** Suppose you get IS Sharpe 1.2, OOS Sharpe 0.7 [band 0.2, 1.2].
4. **Robustness grid + subperiods.** Show the plateau and the subperiod bars.
5. **Costs and capacity.** Estimate net-of-cost Sharpe and a rough capacity. Say "net Sharpe ~0.5, capacity ~$30M."
6. **End with a decision and a kill rule.** "Recommend a small live test at $10M; kill if 6-month live Sharpe < 0.2."

The single most common take-home failure is reporting a gorgeous *in-sample* Sharpe with no OOS, no error bar, no costs, and no decision. The single most impressive move is to *kill your own idea* in the write-up when the OOS evidence is weak — that is the clearest signal of a researcher who can be trusted with capital.

#### Worked example: "Two researchers, same idea, different write-ups. Who do you hire?"

Researcher X writes: *"Sharpe 1.8, 14% return, here is the equity curve. Strong signal, recommend deploying."* Researcher Y writes: *"IS Sharpe 1.8, OOS Sharpe 0.7 [0.3, 1.1]. Plateau across parameters; one negative subperiod (2020, −0.3) I cannot fully explain. Correlation to existing book 0.3. Capacity ~$25M net $0.6M/yr. Recommend a small $8M live test with a kill rule at rolling Sharpe < 0.2; the unexplained 2020 subperiod is my main reservation."*

**Hire Y, decisively.** X reported a *number*; Y reported a *decision* with its uncertainty, its weaknesses, and a plan to manage the risk. Y's headline Sharpe (0.7 OOS) is *lower* than X's (1.8 IS) — and that honesty is exactly why Y is more trustworthy. X's 1.8 is an in-sample number that will likely decay to something like Y's 0.7 the moment it meets fresh data; X just did not tell you that. The lesson: **the researcher who surfaces their idea's weaknesses is more valuable than the one who hides them, because the desk needs the truth to size the bet.**

#### Worked example: "When would you trade an idea you are not confident in?"

A subtle question probing whether you understand sizing. The answer is not "never" — it is *small, with a kill rule*. You trade a low-confidence idea when (a) the downside is bounded and observable, (b) the live data will resolve your uncertainty faster than the backtest can, and (c) the position is small enough that being wrong is cheap.

Concretely: you have a signal with OOS Sharpe 0.6 [band 0.0, 1.2] — promising but not significant. Rather than deploy $40M or kill outright, you allocate $5M as a *live experiment*. Over 6 months of live trading you gather fresh, un-overfittable data. If the live Sharpe tracks above 0.4, you scale up; if it drops below 0.1, you kill. The cost of being wrong is bounded: at $5M and 6% volatility, a full year of −0.5 Sharpe is only about $0.5 \times 6\% \times \$5\text{M} = \$15{,}000$ of expected loss — a cheap price for resolving the uncertainty. **The decision is not binary "trade or kill"; it is "size to your confidence, and let live data buy you certainty you cannot get from the backtest."** The one-sentence intuition: **uncertainty is managed by position size and kill rules, not by waiting for a certainty that backtests can never provide.**

## Common misconceptions

**"A high Sharpe ratio means a good strategy."** A high *in-sample* Sharpe means you fit the past well, which is almost free to do. A high *out-of-sample* Sharpe with a tight error bar, a parameter plateau, real capacity, and low book correlation means a good strategy. The headline number alone is the least informative thing in the write-up; the IS-to-OOS decay and the error bar carry the real information.

**"The job is to find edges and report them."** The job is to make *decisions about capital* and communicate them. An edge you cannot scale, that duplicates the book, or that you cannot reproduce point-in-time is not a deployable decision — and a researcher who reports it as a win has misunderstood the role. The deliverable is "trade this at this size with this kill rule" or "kill this, here is why," not "look what I found."

**"Killing an idea is a failure."** Killing ideas *is the job* — most ideas should die, and a researcher who kills cleanly and documents why is doing high-value work. The kill file is an asset: it stops the firm from re-testing dead ends and it preserves ideas for revival when conditions change. A researcher who never kills anything is not being productive; they are being undisciplined.

**"I should present my idea in the strongest possible light."** This is the most dangerous misconception, because it sounds like good salesmanship. P-hacking your own narrative — quietly dropping the bad subperiod, reporting the lucky parameter, showing IS as if it were OOS, burying the high book correlation in an appendix — does not make the idea better; it makes you *untrustworthy*. The moment a PM catches one buried negative, every future number you report is discounted. Intellectual honesty is not a virtue here; it is the asset that lets your work get funded at all.

**"More data is always the answer to a weak result."** Sometimes, but a weak result on a fragile parameter spike or a signal that only works in one regime will not be rescued by more data — it will just give you a more precise estimate of *zero*. More data tightens error bars; it does not manufacture an edge that was never there. Knowing when more data helps (a real-but-noisy edge) versus when it cannot (an artifact) is itself a researcher skill.

**"Reproducibility is a formality you do at the end."** Reproducibility built in from the start is a *bug detector* that runs continuously. Bolted on at the end, it is a painful audit that often reveals the result was a leak all along — after you have already presented it. The researchers who reproduce point-in-time as they go catch their own look-ahead bugs before anyone else sees the inflated number.

## How it shows up in real research

**The research review meeting.** On most quant desks, every idea that wants capital goes through a research review — a meeting where the researcher presents the write-up and peers plus risk plus a PM try to break it. The questions are exactly the gates in this post: "Is that in-sample or out-of-sample?" "What is the error bar?" "Show me the parameter grid." "What is the capacity?" "How correlated is this with the carry book?" "Can I reproduce it?" A researcher who walks in having already asked themselves these questions sails through; one who has not gets sent back. The review is institutionalized skepticism, and the write-up is your defense.

**The kill file as institutional memory.** Mature desks keep a searchable archive of killed ideas with their write-ups. When a new researcher proposes "what about earnings-drift momentum?", a senior can pull the 2021 write-up that killed it for capacity and say "here is what we found, and here is what would have to change for it to work now." This turns each kill into permanent, compounding knowledge instead of a dead end re-walked every two years. The discipline of writing up a kill — not just abandoning the idea — is what makes the archive valuable.

**The quant crisis of August 2007.** For three days in August 2007, a swath of well-known equity quant strategies — value, reversal, and momentum factors that many funds traded — suffered enormous, simultaneous losses, then partly rebounded. The mechanism was the correlation gate in this post, realized at the worst moment: a large number of funds held nearly the *same* positions (their signals were highly correlated, ~0.85 in spirit), so when one large book was forced to delever and sell, it pushed prices against everyone holding the same names, forcing more selling. Each fund's backtest had looked independent; in a crisis they were one giant crowded trade. The lesson the industry took: a signal's correlation *to what everyone else trades* is as important as its correlation to your own book, and a write-up that ignores crowding ignores a real tail risk.

**Look-ahead bias in published factors.** A recurring finding in academic and practitioner research is that many published "anomalies" shrink dramatically or vanish out-of-sample after publication — partly because they were partly data-mined, partly because trading them away is exactly what should happen to a real edge, and partly because the original backtests used subtly non-point-in-time data. The practical consequence on a desk: a signal that worked in a paper gets the *same* OOS, robustness, capacity, and reproducibility gauntlet as any in-house idea, and many published factors are killed on Gate 1 the moment they meet honest point-in-time data and realistic costs.

**Sizing to confidence at a multi-strategy firm.** Large multi-strategy platforms institutionalize the "size to your confidence" idea: a new signal rarely gets full capital on day one. It gets a small allocation as a live experiment, and capital scales as live performance confirms the backtest — or the allocation is cut on a pre-agreed kill rule. This is the worked example about trading a low-confidence idea, turned into firm-wide policy. The researcher's write-up specifies the initial size, the scale-up triggers, and the kill rule, and the firm's risk system enforces them automatically. The decision in the write-up is not a one-time call; it is a *control loop*.

**The reproducibility mandate after a blowup.** After any meaningful production loss traced to a research error, desks typically tighten their reproducibility requirements: mandatory point-in-time pipelines, pinned environments, peer reproduction on a clean machine before deployment, and logged seeds. The motivation is always the same painful lesson — that the loss came from a bug a reproducibility check would have caught for a few hours of effort. The discipline that feels like overhead in calm times is the discipline that prevents the next eight-figure mistake.

## When this matters to you and further reading

If you are preparing for a quant *researcher* interview, internalize this above all: the firms are hiring your *judgment about capital*, expressed as a clear, honest decision — not your ability to produce a high backtest number. In a take-home, split your data first, form a hypothesis before you mine, report in-sample and out-of-sample with error bars, check robustness and capacity and correlation, and *end with a decision and a kill rule*. In a live round, when handed a beautiful Sharpe, lead with skepticism and the checklist, not excitement. And whenever the evidence is ambiguous, remember the asymmetry: a false positive in production is far more expensive than a false negative you can always revisit, so when in doubt, kill — and write up exactly why.

These ideas connect to the rest of the quant toolkit. The statistics of telling a real edge from noise live in [hypothesis testing and p-values](/blog/trading/quantitative-finance/hypothesis-testing-pvalues-quant-interviews) and in [estimators, bias, and variance](/blog/trading/quantitative-finance/estimators-mle-bias-variance-quant-interviews). The correlation arithmetic behind the book-overlap gate is in [covariance and correlation pitfalls](/blog/trading/quantitative-finance/covariance-correlation-pitfalls-quant-interviews). The sizing-to-confidence logic — how much to bet when you have a real but uncertain edge — is the [Kelly criterion and sequential betting](/blog/trading/quantitative-finance/kelly-criterion-sequential-betting-quant-interviews), and the broader framework for choosing under uncertainty is in [decision-making under uncertainty](/blog/trading/quantitative-finance/decision-making-under-uncertainty-quant-interviews). For practice, take any backtest you have run — even a toy one — and write the one-page version: decision first, then data, method, results-with-error-bars, robustness, risks, capacity, and a kill rule. The discipline of writing the decision, not the number, is the single most transferable skill a quant researcher can build.
