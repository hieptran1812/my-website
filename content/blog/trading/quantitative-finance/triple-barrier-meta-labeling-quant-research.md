---
title: "Triple-barrier labeling and meta-labeling: Lopez de Prado's financial ML toolkit"
date: "2026-06-14"
publishDate: "2026-06-14"
description: "A from-scratch, interview-ready guide to the two ideas that fix financial ML labels: the triple-barrier method (which turns a price path into a clean +1/0/-1 label) and meta-labeling (which splits which way from how much to bet), with worked dollar examples, sample-weight math, bet sizing, and five fully solved take-home problems."
tags:
  [
    "triple-barrier-method",
    "meta-labeling",
    "financial-machine-learning",
    "lopez-de-prado",
    "quant-research",
    "sample-weights",
    "bet-sizing",
    "labeling",
    "quant-interviews",
    "position-sizing",
  ]
category: "trading"
subcategory: "Quantitative Finance"
author: "Hiep Tran"
featured: true
readTime: 44
---

> [!important]
> **TL;DR** — How you *label* financial data decides what a model can possibly learn; the triple-barrier method and meta-labeling fix the naive fixed-horizon label and cleanly separate "which way" from "how much to bet."
>
> - A **fixed-time-horizon label** ("did the price rise after 5 days?") throws away two things that matter most: the *path* the price took on the way, and the *volatility* of the asset. It mislabels trades that were stopped out and trades that left money on the table.
> - The **triple-barrier method** boxes every trade with three barriers — an upper profit-take, a lower stop-loss, and a vertical time limit — and labels it by whichever barrier the price touches *first*.
> - Setting the barriers as a multiple of **recent volatility** makes a 1% move meaningful in a calm market and ignorable in a wild one, so the labels are comparable across regimes.
> - Overlapping holding windows make samples **non-independent**; the fix is a **sample weight** equal to the average of 1/concurrency over each label's life.
> - **Meta-labeling** runs two models: a primary model picks the *side* (long or short) and a secondary model decides *whether to act and how big*. It raises precision, kills false trades, and gives you a probability you can size a bet from.
> - The one number to remember: in our worked confusion matrix, meta-labeling lifts precision from **60% to 80%** while giving up only 20 points of recall — and the sized strategy ends at **\$128** versus **\$112** for the raw signal.

Here is a question that has sunk more quant take-homes than any clever feature ever saved: *what, exactly, is the thing your model is trying to predict?*

Most people skip past it. They grab prices, compute "the return 5 days from now," threshold it into up or down, and hand that to a gradient-boosted tree. The model trains, the accuracy looks fine, and the backtest is a disaster. The features were not the problem. The *label* was the problem. A label that ignores the path a price took and the volatility of the asset is teaching the model to answer the wrong question — and a model can only ever be as good as the question its labels pose.

This post is about the two ideas that fix that, both from Marcos Lopez de Prado's *Advances in Financial Machine Learning*: the **triple-barrier method** for labeling, and **meta-labeling** for separating direction from sizing. They are favorite interview and take-home topics at shops like Two Sigma, Citadel, DE Shaw, and WorldQuant precisely because they reward people who think about the problem *before* the model. We will build both from zero, with dollar examples you can do in your head, the sample-weight math worked out by hand, Python you can run, and five fully solved problems of the kind you will actually be handed.

![Fixed-horizon labeling stamps the same outcome on the smooth winner, the stopped-out loser, and the wild round trip; the triple barrier reads whichever outcome happens first.](/imgs/blogs/triple-barrier-meta-labeling-quant-research-1.png)

The diagram above is the mental model for the whole post. On the left is the naive label: wait a fixed number of days and look at the sign of the return, blind to everything that happened in between. On the right is the triple-barrier label: draw three lines around the trade and let the *first one the price touches* decide the outcome. Almost everything else here is a consequence of taking that picture seriously.

## Foundations: the labeling problem, from zero

Before any of this makes sense, we need a shared vocabulary. If you already trade, skim; if you do not, read carefully, because every later section leans on these.

### What a "label" even is

In **supervised machine learning**, you show a model many examples, each a pair: a set of *features* (the inputs the model sees) and a *label* (the answer you want it to learn). A spam filter sees the words in an email (features) and a human-assigned spam / not-spam tag (label). The model learns the mapping from features to label, then predicts labels on new, unseen examples.

In trading, the features are whatever you measure at the moment you might enter a trade — momentum, order-book imbalance, a valuation ratio, a signal from another model. The **label** is what happened *after* you entered: did the trade make money or lose money? The whole art of *financial* machine learning is that this second part — "what happened after" — is far subtler than it looks.

A few terms we will use constantly, defined once:

- A **position** is a trade you are holding. A **long** position profits when the price rises; a **short** profits when it falls. The *side* of a trade is whether it is long or short.
- A **return** is the percentage change in price. If you buy at \$100 and sell at \$105, your return is +5%.
- **Volatility** is how much a price typically moves, usually measured as the standard deviation of daily returns. A *standard deviation* is just a number that says "a typical day's move is about this big." An asset with 0.5% daily volatility is calm; one with 2% daily volatility is wild.
- A **basis point** (bp) is one hundredth of a percent — 0.01%. A "40 bps move" is a 0.40% move. Traders count in bps because the moves they care about are small.
- A **drawdown** is a peak-to-trough decline in your account value. If you grow to \$130 and then fall to \$110 before recovering, you suffered a \$20 (about 15%) drawdown.

### The naive label: fixed-time-horizon

The simplest possible label is the **fixed-time-horizon** label. Pick a horizon — say 5 trading days. For each potential entry, look at the return over those 5 days and assign:

$$
y_i = \begin{cases} +1 & \text{if } r_{i,5} > \tau \\ 0 & \text{if } |r_{i,5}| \le \tau \\ -1 & \text{if } r_{i,5} < -\tau \end{cases}
$$

where $r_{i,5}$ is the 5-day return starting at observation $i$, and $\tau$ (tau) is a fixed threshold like 1%. So: up more than 1% is a buy (+1), down more than 1% is a sell (-1), anything in between is "do nothing" (0).

This is clean, easy to code, and *wrong* in two specific ways that the triple-barrier method exists to fix.

### Why fixed-horizon labels are bad

**Problem one: it ignores the path.** The label only looks at the price 5 days later. It is completely blind to what the price did in between. Consider three trades that all end at +2% after 5 days:

- **Path A** drifts smoothly up to +2%. A trade you would have happily held.
- **Path B** crashes to -9% on day 2, scares you out at a stop loss, and only *then* recovers to +2%. In real life you were stopped out at -9% and never saw the recovery — but the fixed label calls this a +1 winner.
- **Path C** rockets to +12% on day 3, then sags back to +2% by day 5. You should have taken the 12%; the fixed label calls it a +1 and pretends the round trip never happened.

![A fixed-horizon label assigns the same plus-one outcome to a smooth winner, a stopped-out loser, and a wild round trip a risk manager would treat completely differently.](/imgs/blogs/triple-barrier-meta-labeling-quant-research-12.png)

All three get the same +1 label. A risk manager would treat them as three completely different trades. The fixed-horizon label has erased exactly the information that distinguishes a good trade from a lucky one.

**Problem two: it ignores volatility.** A fixed 1% threshold means very different things in different markets. In a calm week where the asset moves 0.3% a day, a 1% move over 5 days is a real signal. In a panic where the asset moves 3% a day, a 1% move is *noise* — the asset crosses 1% several times before lunch. Using one threshold across both regimes labels noise as signal in the calm period and signal as noise in the wild period. The model learns a blurry average of two incompatible worlds.

**Problem three (the subtle one): it ignores when you would have actually exited.** Real strategies have stop losses and profit targets. A fixed-horizon label assumes you sit on every trade for exactly 5 days no matter what — which no trader does. The label describes a strategy nobody runs.

#### Worked example: how badly a fixed label misreads a stopped-out trade

Make Path B from above concrete. You buy at **\$100** with a mental stop at **\$93** (a -7% line you would never let a trade cross). The closes go: day 1 \$98, day 2 **\$91**, day 3 \$96, day 4 \$99, day 5 \$102.

The fixed-5-day label looks only at day 5: the price is \$102, the return is +2%, so the label is **+1, a winner**. But look at day 2 — the price hit \$91, *below* your \$93 stop. In reality you were stopped out on day 2 at roughly -7%, realized a \$7 loss per share, and were flat for the recovery you never participated in. The fixed label is teaching the model that this setup *makes* +2% when, traded the way you actually trade, it *lost* 7%. Feed enough of these mislabeled trades to a model and it learns to love exactly the setups that whipsaw you out before they recover — the worst possible lesson. The triple-barrier method, with a -7% lower barrier, would have correctly stamped this **-1** on day 2 and closed the window there.

The fix for all three problems is the same idea: instead of one line in the future, draw a *box* around the trade.

### Why financial data breaks the textbook assumptions

It helps to name *why* finance needs its own labeling toolkit when image and text models do fine with simple labels. Three properties of market data are to blame, and every technique in this post is a response to one of them.

First, the **signal-to-noise ratio is brutally low**. A cat photo is unambiguously a cat; a price chart is mostly noise with a faint drift hidden inside. A label that is even slightly wrong drowns the thin signal entirely, so getting the label exactly right matters far more here than in domains where the signal is loud.

Second, **the data is serially dependent**. Today's price is yesterday's price plus a small change; consecutive observations are deeply related. This is what makes overlapping labels share information and what breaks the i.i.d. assumption that ordinary cross-validation relies on. In a dataset of independent photos there is no analogous problem.

Third, **the distribution shifts under your feet**. Volatility regimes change, correlations break, strategies that worked stop working as others discover them. A fixed 1% threshold that was sensible last year is a hair-trigger this year. Anchoring everything to *current volatility* rather than absolute price levels is how you keep labels comparable as the world moves.

Hold these three in mind and the rest of the post reads as a checklist: triple-barrier labels fight the low signal-to-noise by labeling the real exit; volatility scaling fights the distribution shift; sample weights fight the serial dependence; meta-labeling squeezes more out of a thin edge by sizing it well.

## The triple-barrier method

The triple-barrier method replaces the single future point with three barriers, and labels the trade by whichever barrier the price hits *first*:

1. An **upper barrier** — a profit-take level above the entry. Touch it first and the label is **+1** (the trade hit its target).
2. A **lower barrier** — a stop-loss level below the entry. Touch it first and the label is **-1** (the trade hit its stop).
3. A **vertical barrier** — a time limit. If neither horizontal barrier is touched by then, the trade times out and the label is the **sign of the return** at that point (often coded 0 if you only care about the horizontal hits, or +1/-1 by the sign).

The name comes from the shape: two horizontal barriers and one vertical barrier box the trade in. The label is determined by *first touch*, which means the method finally respects the path — it reads the price day by day and stops the moment a barrier is crossed.

![A profit-take barrier, a stop-loss barrier, and a time barrier box in every trade so the first touch sets the label.](/imgs/blogs/triple-barrier-meta-labeling-quant-research-2.png)

#### Worked example: assigning a triple-barrier label to a price path

You enter a long position at **\$100**. You set:

- Upper barrier at **\$105** (a +5% profit target).
- Lower barrier at **\$95** (a -5% stop loss).
- Vertical barrier at **5 trading days**.

Now watch a concrete path, day by day (these are the closing prices):

| Day | Price | Touched a barrier? |
|---|---|---|
| 0 (entry) | \$100.00 | no |
| 1 | \$100.60 | no |
| 2 | \$100.10 | no |
| 3 | \$101.50 | no |
| 4 | \$105.00 | **yes — upper barrier** |

On day 4 the price reaches \$105, touching the upper barrier. **First touch wins**, so the label is **+1**, and — this is important — the trade's *holding window ends on day 4*, not day 5. We never look at days 5 onward for this label. If the price had instead fallen to \$95 on day 2 before ever reaching \$105, the label would be **-1** and the window would close on day 2.

What if the price wandered between \$95 and \$105 for all 5 days and never touched either horizontal barrier? Then the vertical barrier fires on day 5, and the label is the sign of the return at day 5. If the price closed at \$100.80, the return is +0.8%, so the label is +1 (or 0 if your scheme reserves +1/-1 for genuine barrier touches). The vertical barrier guarantees *every* trade gets labeled — no trade is left hanging forever.

**The intuition this teaches:** the triple-barrier label answers the question a trader actually faces — "given my profit target and my stop, which one would have triggered first?" — instead of the fiction "where is the price exactly N days from now."

### The first-touch algorithm, in words and code

The procedure is mechanical. For each entry at time $t_0$:

1. Set the three barriers: upper $= P_{t_0}(1 + u)$, lower $= P_{t_0}(1 - \ell)$, vertical $= t_0 + h$ bars ahead.
2. Walk forward bar by bar. At each bar, check: did the high reach the upper barrier? Did the low reach the lower barrier?
3. The first bar that touches *either* horizontal barrier ends the trade with the matching label.
4. If you reach the vertical barrier untouched, end the trade there and label by the sign of the return.

![Scanning bars after entry, the first barrier the price touches decides the label, and the vertical barrier guarantees every trade is labeled.](/imgs/blogs/triple-barrier-meta-labeling-quant-research-9.png)

Here is a compact, runnable pandas implementation. The careful detail is using intrabar highs and lows so you do not miss a barrier the price pierced and then closed back inside.

```python
import pandas as pd
import numpy as np

def triple_barrier_labels(prices, events, pt=0.05, sl=0.05, max_h=5):
    """
    prices : DataFrame with columns ['high','low','close'], indexed by bar.
    events : list of integer entry positions (bar index of t0).
    pt, sl : profit-take / stop-loss as fractions of entry price.
    max_h  : vertical barrier, in bars.
    Returns a DataFrame: entry, exit, label (+1/-1), ret.
    """
    out = []
    for t0 in events:
        entry_px = prices['close'].iloc[t0]
        upper = entry_px * (1 + pt)
        lower = entry_px * (1 - sl)
        label, exit_t, exit_px = 0, t0 + max_h, None
        # walk forward, first touch wins
        for t in range(t0 + 1, min(t0 + max_h + 1, len(prices))):
            hi, lo = prices['high'].iloc[t], prices['low'].iloc[t]
            if hi >= upper:                  # profit target hit first
                label, exit_t, exit_px = 1, t, upper
                break
            if lo <= lower:                  # stop loss hit first
                label, exit_t, exit_px = -1, t, lower
                break
        if exit_px is None:                  # vertical barrier: label by sign
            exit_px = prices['close'].iloc[min(t0 + max_h, len(prices) - 1)]
            label = int(np.sign(exit_px / entry_px - 1))
        out.append({'entry': t0, 'exit': exit_t,
                    'label': label, 'ret': exit_px / entry_px - 1})
    return pd.DataFrame(out)
```

The order of the checks matters when both barriers sit inside a single bar's high-low range; production code resolves that tie with intrabar (tick) data or a conservative assumption (assume the stop hit first, so you do not flatter the strategy).

## Setting the barriers dynamically from volatility

The fixed-threshold problem has a clean fix: stop using a fixed threshold. Instead, set the barriers as a multiple of *recent volatility*. If a typical day moves the price by $\sigma$ (sigma), then a sensible profit target is some multiple of $\sigma$ over the holding horizon. In a calm market $\sigma$ is small, so the barriers sit close to the entry; in a wild market $\sigma$ is large, so they widen out. The same *rule* adapts to the regime automatically.

The standard recipe estimates daily volatility with an **exponentially weighted moving standard deviation** of returns — "exponentially weighted" just means recent days count more than old days, decaying smoothly — then scales the barriers by it.

```python
def daily_vol(close, span=20):
    """Exponentially weighted daily-return volatility (recent days weigh more)."""
    rets = close.pct_change()
    return rets.ewm(span=span).std()

def vol_scaled_barriers(entry_px, sigma_daily, k=2.0, horizon_days=5):
    # barriers as a k-sigma move, given the vol estimate at entry
    move = k * sigma_daily * np.sqrt(horizon_days)   # scale vol to the horizon
    return entry_px * (1 + move), entry_px * (1 - move)
```

The $\sqrt{\text{horizon}}$ factor is the **square-root-of-time rule**: volatility over $h$ independent days grows like $\sigma\sqrt{h}$, not $\sigma h$, because random daily moves partly cancel. Defining it once: if each day's move is independent with standard deviation $\sigma$, the standard deviation of the *sum* of $h$ days is $\sigma\sqrt{h}$.

![A two-sigma rule gives plus or minus 1 percent barriers in a calm market and plus or minus 4 percent in a volatile one, so the same rule adapts to the regime.](/imgs/blogs/triple-barrier-meta-labeling-quant-research-3.png)

#### Worked example: volatility-scaled barriers in two regimes

Take a stock priced at **\$100**, a 2-sigma barrier rule ($k = 2$), and for simplicity a 1-day horizon so the $\sqrt{h}$ factor is just 1.

**Calm regime.** The exponentially-weighted daily volatility is **0.5%** ($\sigma = 0.005$). The barrier move is $k \sigma = 2 \times 0.5\% = 1.0\%$. So:

- Upper barrier $= \$100 \times (1 + 0.01) = \$101$.
- Lower barrier $= \$100 \times (1 - 0.01) = \$99$.

A 1% move ends the trade. In a quiet market that is a meaningful directional move, so the label is informative.

**Volatile regime.** A few weeks later the market is in turmoil and the daily volatility has risen to **2.0%** ($\sigma = 0.020$). Same rule, same $k = 2$: the barrier move is $2 \times 2.0\% = 4.0\%$. So:

- Upper barrier $= \$100 \times (1 + 0.04) = \$104$.
- Lower barrier $= \$100 \times (1 - 0.04) = \$96$.

Now it takes a **4%** move to trigger a barrier. The everyday 1-2% noise of the turbulent market no longer ends trades prematurely. The exact same code, with no hand-tuning, drew tight barriers when the market was calm and wide barriers when it was wild.

**The intuition this teaches:** a barrier should be measured in *units of risk*, not in fixed cents — a fixed \$1 stop is a hair trigger in a calm market and a meaningless rounding error in a crisis, but a 2-sigma stop means the same thing in both.

## Sample uniqueness and overlapping labels

Here is the trap that catches almost everyone the first time, and the one interviewers love because it tests whether you understand *why* standard ML assumptions break in finance.

Standard machine learning assumes your training examples are **independent and identically distributed** (i.i.d.) — drawn separately, each carrying its own fresh piece of information. Coin flips are i.i.d. Financial labels are emphatically *not*.

Think about why. With the triple-barrier method, each label's outcome depends on the returns over its *entire holding window*, which can be several days. If you generate a new label every day, then on any given day, *several labels are simultaneously alive*, and they all depend on that same day's return. Two trades whose windows overlap are partly the *same observation* counted twice.

![Three trades whose holding windows overlap on the same days share returns, so each sample is only partly unique and needs a weight of one over its concurrency.](/imgs/blogs/triple-barrier-meta-labeling-quant-research-4.png)

If you feed overlapping labels into a model as if they were independent, you commit a quiet but serious sin: you *overstate how much data you have*. A thousand heavily-overlapping labels might carry the independent information of only a couple hundred truly distinct trades. The model and your cross-validation both think they have seen far more evidence than they have, so they overfit and your out-of-sample results collapse. (This is the labeling-side cousin of the leakage problems that motivate [purged cross-validation](/blog/trading/quantitative-finance/financial-ml-pipeline-purged-cv-quant-research) and the [deflated Sharpe ratio](/blog/trading/quantitative-finance/overfitting-purged-cv-deflated-sharpe-quant-research).)

### The fix: sample weights from concurrency

The cure is to *down-weight* overlapping samples so the total information is counted once. The key quantity is **concurrency**: on each day, how many labels are simultaneously alive? A day on which three trades are open contributes only 1/3 of a unit of "uniqueness" to each of those three labels, because that day's return is shared three ways.

Formally, each label's **uniqueness weight** is the average, over its holding window, of one over the concurrency on each day:

$$
\bar u_i = \frac{1}{|T_i|}\sum_{t \in T_i} \frac{1}{c_t}
$$

where $T_i$ is the set of days label $i$ is alive, and $c_t$ is the number of labels alive on day $t$. A label that lives entirely on uncrowded days keeps a weight near 1; a label that lives on heavily-overlapped days gets a small weight. You then pass these weights to the model (most libraries accept a `sample_weight` argument) so crowded, redundant observations stop dominating the fit.

![Each label's weight is the average of one over the number of trades open on each day inside its holding window, so the most-overlapped trade gets the least weight.](/imgs/blogs/triple-barrier-meta-labeling-quant-research-11.png)

#### Worked example: computing sample weights for overlapping labels

Three trades, each held for 3 days:

- **Trade A** is alive on days 1, 2, 3.
- **Trade B** is alive on days 2, 3, 4.
- **Trade C** is alive on days 3, 4, 5.

First, count the concurrency $c_t$ on each day — how many trades are open:

| Day | Trades alive | Concurrency $c_t$ | Per-trade share $1/c_t$ |
|---|---|---|---|
| 1 | A | 1 | 1.00 |
| 2 | A, B | 2 | 0.50 |
| 3 | A, B, C | 3 | 0.33 |
| 4 | B, C | 2 | 0.50 |
| 5 | C | 1 | 1.00 |

Day 3 is the crowded one: all three trades share it, so it contributes only 0.33 to each.

Now average the shares over each trade's three days:

- **Trade A** (days 1, 2, 3): $\bar u_A = (1.00 + 0.50 + 0.33)/3 = 1.83/3 = 0.61$.
- **Trade B** (days 2, 3, 4): $\bar u_B = (0.50 + 0.33 + 0.50)/3 = 1.33/3 = 0.44$.
- **Trade C** (days 3, 4, 5): $\bar u_C = (0.33 + 0.50 + 1.00)/3 = 1.83/3 = 0.61$.

Trade B is the *most* overlapped — it sits in the middle where every other trade is also open — so it gets the smallest weight, 0.44. Trades A and C, which each get a day to themselves (day 1 and day 5), keep more weight at 0.61. In training you would normalize these so they sum to the sample count (multiply each by $3 / (0.61+0.44+0.61) = 3/1.66 = 1.81$, giving roughly 1.10, 0.80, 1.10), but the *ratios* are the point: B counts less because B is more redundant.

```python
def sample_weights(events):
    """events: DataFrame with integer 'entry' and 'exit' (inclusive) columns."""
    last = int(events['exit'].max())
    conc = pd.Series(0, index=range(0, last + 1))
    for _, e in events.iterrows():          # count concurrency per day
        conc.loc[e['entry']:e['exit']] += 1
    w = []
    for _, e in events.iterrows():          # average 1/concurrency over each life
        days = range(int(e['entry']), int(e['exit']) + 1)
        w.append(np.mean([1.0 / conc.loc[t] for t in days]))
    w = pd.Series(w, index=events.index)
    return w * len(w) / w.sum()             # normalize to sum to sample count
```

**The intuition this teaches:** overlapping labels are not extra data, they are the same data wearing different hats — and a sample weight of 1/concurrency is how you count each day's return exactly once.

## Meta-labeling: separating "which way" from "how much"

Now the second big idea, and the one that most directly improves a live strategy. The triple-barrier method gave us *good labels*. Meta-labeling changes the *architecture* of the model around them.

The insight is that a trading decision has two parts that are easier to learn separately than together:

1. **Direction** — should I go long or short? This is the *side*.
2. **Conviction and size** — given a side, should I actually take this trade, and if so, how big? This is the *bet*.

A single model forced to learn both at once is solving a harder problem than it needs to. Meta-labeling splits them:

- A **primary model** (or even a simple rule — a moving-average crossover, a human analyst, an existing signal) decides the side. It is allowed to be *aggressive*: high recall, generating lots of candidate trades, accepting that many will be wrong.
- A **secondary (meta) model** then looks at each candidate trade *together with the side the primary chose* and predicts a single thing: **will this particular trade be profitable?** Its label is binary — 1 if the trade the primary suggested would have hit its profit barrier, 0 if it would have hit its stop. The meta-model's job is to *filter and size*: veto the low-confidence trades, and turn its probability into a bet size for the rest.

![A primary model decides direction while a secondary model decides whether to act on that side and how big to bet.](/imgs/blogs/triple-barrier-meta-labeling-quant-research-5.png)

The name "meta-labeling" comes from the fact that the secondary model's labels are *labels about the primary model's labels* — they encode "was the primary right this time?"

### Why this raises precision

Two definitions, because the whole payoff is stated in these terms:

- **Precision** answers: *of the trades I took, what fraction made money?* High precision means few false trades.
- **Recall** answers: *of all the profitable opportunities that existed, what fraction did I catch?* High recall means you miss few winners.

There is a tension between them. A model that trades constantly catches every winner (high recall) but also takes every loser (low precision). A model that trades only on near-certainties has high precision but misses many winners (low recall). The metric that balances them is the **F1 score**, the harmonic mean of precision and recall.

Meta-labeling deliberately exploits this tension. You let the primary model run hot — high recall, lots of candidates, including many false positives. Then the meta-model *raises precision* by vetoing the candidates it judges low-probability. You trade fewer times, but a much higher fraction of your trades win. You give up a little recall (you skip a few real winners that looked uncertain) to buy a lot of precision (you dodge many losers).

![Meta-labeling lifts precision from sixty to eighty percent while giving up only twenty points of recall, because false trades are vetoed faster than winners are skipped.](/imgs/blogs/triple-barrier-meta-labeling-quant-research-6.png)

#### Worked example: a meta-labeling confusion-matrix improvement

Suppose the primary model generates **100 long signals**. The ground truth, from triple-barrier labels, is that **60 of them are winners** and **40 are losers**. If you trade every signal (no meta-model), your confusion matrix on the "go long" decision is:

| | Actually a winner | Actually a loser |
|---|---|---|
| **You traded** | 60 (true positives) | 40 (false positives) |
| **You skipped** | 0 | 0 |

- Precision $= 60 / (60 + 40) = 60\%$. Six of every ten trades win.
- Recall $= 60 / 60 = 100\%$. You caught every winner — because you traded everything.

Now add a meta-model that predicts, for each signal, the probability it wins, and you only trade when that probability clears a threshold. Suppose the meta-model is good: it correctly waves through **48 of the 60** true winners and correctly vetoes **28 of the 40** losers, while mistakenly vetoing 12 winners and mistakenly waving through 12 losers:

| | Actually a winner | Actually a loser |
|---|---|---|
| **You traded** | 48 (true positives) | 12 (false positives) |
| **You skipped** | 12 (false negatives) | 28 (true negatives) |

- Precision $= 48 / (48 + 12) = 80\%$. Now eight of every ten trades win.
- Recall $= 48 / 60 = 80\%$. You still catch 80% of the winners.
- F1 rises from $2 \cdot \frac{0.6 \cdot 1.0}{0.6 + 1.0} = 0.75$ to $2 \cdot \frac{0.8 \cdot 0.8}{0.8 + 0.8} = 0.80$.

![Filtering the primary signal with the meta-label cuts false positives from 40 to 12 while only costing a few true positives, lifting precision from 60 to 80 percent.](/imgs/blogs/triple-barrier-meta-labeling-quant-research-10.png)

You went from trading 100 times at 60% precision to trading 60 times at 80% precision. You cut your *false trades from 40 down to 12* — a 70% reduction in money-losing trades — at the cost of skipping 12 real winners. For most strategies, where each losing trade also costs spread and fees, that trade is hugely worth it.

To turn this into dollars, suppose each winner makes \$500 and each loser costs \$500 (an even-money strategy), and every trade also pays \$20 of round-trip spread and commission. Trading all 100 signals nets $60 \times \$500 - 40 \times \$500 - 100 \times \$20 = \$30{,}000 - \$20{,}000 - \$2{,}000 = \$8{,}000$. With the meta-filter you trade 60 times: $48 \times \$500 - 12 \times \$500 - 60 \times \$20 = \$24{,}000 - \$6{,}000 - \$1{,}200 = \$16{,}800$. The same primary signal, the same winners and losers in the world — but filtering doubled the take-home from \$8,000 to \$16,800, because every vetoed loser saved both the \$500 loss and the \$20 of friction. That is the entire economic case for meta-labeling in one line.

**The intuition this teaches:** meta-labeling does not make the primary model smarter; it adds a *second opinion* that abstains on the trades the primary is least sure about, and abstaining is itself a profitable action.

### Combining meta-labeling with an existing primary signal

A beautiful, practical property: the primary model does not have to be an ML model at all. Meta-labeling layers on top of *anything* that produces a side — a momentum rule, a mean-reversion signal, a discretionary trader's calls, a vendor signal you bought. You take that signal's directional calls as given, generate triple-barrier meta-labels for them ("did this call work?"), and train the secondary model to filter and size. This is why meta-labeling is so popular in practice: it improves a strategy you already have without forcing you to rebuild the alpha. (If you are building that primary signal from scratch, see [building an alpha signal](/blog/trading/quantitative-finance/building-an-alpha-signal-quant-research).)

The features the meta-model uses are often *different* from the primary's features. The primary asks "which way is the market going?" The meta-model asks "is *this* the kind of setup where the primary tends to be right?" — so its features include things like the current volatility regime, the time of day, the strength of the primary's own signal, recent hit rate, and market microstructure. It is learning the *conditions under which the primary can be trusted*.

Building the meta-labels themselves is mechanical once you have the primary's side and triple-barrier outcomes. The meta-label is 1 when the primary's *directional* call agreed with the realized outcome, and 0 when it did not:

```python
def make_meta_labels(primary_side, tb_labels):
    """
    primary_side : Series of +1 (long) / -1 (short), the primary's call.
    tb_labels    : Series of +1 / -1, the triple-barrier outcome.
    Returns a binary meta-label: 1 if the primary was right, else 0.
    """
    # the primary 'wins' when its side matches the barrier outcome's sign
    agreed = (primary_side * tb_labels) > 0
    return agreed.astype(int)
```

Note the asymmetry this creates: the meta-model is trained *only on the trades the primary proposed*, and its label is "was the primary correct," not "which way did the market go." That smaller, focused question is genuinely easier to learn than the full directional problem — which is the whole reason the two-model split works. A subtle but important consequence: the meta-model can only ever *reduce* the set of trades, never add new directions. If the primary never goes short, no meta-model will ever produce a short. The primary owns the opportunity set; the meta-model owns the discipline.

## Bet sizing from the meta-label's probability

The meta-model does not just output a yes/no; it outputs a **probability** $p$ that the trade will win. That probability is gold, because it tells you not just *whether* to bet but *how much*. A trade the model is 95% sure of deserves a bigger position than one it is 56% sure of. Sizing every trade the same throws away the model's confidence.

The cleanest mapping turns the probability into a position size that is **zero near a coin flip and grows with confidence**. One standard construction: compute a test statistic from $p$, push it through the normal distribution to get a bet size between -1 and 1, and scale by your maximum position. The practical shape is simple: below some threshold (say $p = 0.55$) you do not trade at all, and above it the size ramps up toward your cap as $p$ approaches 1.

![Bet size sits at zero near a coin flip and rises toward the full cap as the meta-model's confidence grows, abstaining below the threshold.](/imgs/blogs/triple-barrier-meta-labeling-quant-research-7.png)

#### Worked example: sizing a dollar bet from a meta-label probability

You have a **\$100,000** account and a self-imposed rule that no single trade risks more than **\$10,000** of capital (your position cap). The meta-model says this long trade has probability $p = 0.80$ of hitting its profit barrier.

A simple, defensible sizing rule is to treat the model's *edge* — how far its probability is above a coin flip, doubled — as the fraction of the cap to deploy:

$$
\text{edge} = 2p - 1 = 2(0.80) - 1 = 0.60.
$$

So this trade gets 60% of the cap: $0.60 \times \$10{,}000 = \$6{,}000$ of capital at risk. A trade with $p = 0.95$ would get $\text{edge} = 0.90$, or \$9,000 — much larger. A trade with $p = 0.55$ gives $\text{edge} = 0.10$, or just \$1,000, and one at $p = 0.52$ rounds down to roughly nothing, which is the model telling you to pass.

**Connecting to fractional Kelly.** This "edge = $2p - 1$" rule is exactly the **Kelly criterion** for an even-money bet (a bet that pays \$1 for every \$1 risked): the Kelly fraction is $f^* = 2p - 1$, the edge. Full Kelly maximizes long-run growth but is famously volatile — a string of losses with a full-Kelly stake produces gut-wrenching drawdowns. So desks run **fractional Kelly**, betting some fraction (often a half or a quarter) of the Kelly amount to cut volatility for a small cost in growth. At half-Kelly, our $p = 0.80$ trade would risk $0.5 \times \$6{,}000 = \$3{,}000$. The meta-model's probability flows straight into the sizing math; the [Kelly criterion and sequential betting](/blog/trading/quantitative-finance/kelly-criterion-sequential-betting-quant-interviews) post derives why $f^* = 2p - 1$ and why fractional Kelly is the sane default.

**The intuition this teaches:** the meta-model's probability is not a label to threshold and discard — it is a dial, and turning it straight into Kelly-scaled position size is most of what "alpha" looks like once you have a calibrated edge.

### Putting it together: the combined strategy P&L

When you stack the pieces — a primary signal for direction, a meta-model to filter and size — the cumulative profit-and-loss curve improves in two ways at once. The *level* rises because you skip losing trades and bet bigger on winners. The *smoothness* improves because the meta-filter is most aggressive exactly when conditions are bad, so the drawdowns shrink.

![Filtering the raw signal with the meta-label lifts the ending equity and smooths the drawdowns versus trading the primary signal alone.](/imgs/blogs/triple-barrier-meta-labeling-quant-research-8.png)

#### Worked example: primary-only versus primary-plus-meta equity

Start both strategies with **\$100**. The primary-only strategy trades every signal: it ends at **\$112**, but along the way it whipsaws — up to \$118, back to \$104, the kind of jagged ride that gets a strategy turned off. The primary-plus-meta strategy trades the same signals but skips the low-probability ones and sizes the rest by confidence. It climbs more steadily and ends at **\$128**.

The \$16 difference (\$128 versus \$112) comes from two sources you can now name precisely: the 28 losing trades the meta-filter vetoed (each one a small loss avoided, plus saved spread and fees), and the larger sizes on the high-confidence winners. Just as important, the *worst drawdown* of the meta strategy is shallower, because the filter pulls capital exactly when the primary signal is least reliable. A smoother equity curve with the same or higher return is a higher [Sharpe ratio](/blog/trading/quantitative-finance/evaluating-alpha-signals-ic-sharpe-turnover-quant-research) — which is what actually gets a strategy funded.

## In the interview room and the take-home

These five problems are the shape of what you will be handed — some are 60-second whiteboard questions, some are mini take-homes. Each is fully solved. Work them by hand before peeking; the muscle you build is the point.

#### Worked example: problem 1 — label this path

*"A stock is bought at \$50. You set an upper barrier at +4%, a lower barrier at -4%, and a vertical barrier at 4 days. The closes are: day 1 \$50.80, day 2 \$48.20, day 3 \$49.50, day 4 \$52.40. What is the triple-barrier label?"*

Compute the barriers first: upper $= \$50 \times 1.04 = \$52.00$; lower $= \$50 \times 0.96 = \$48.00$. Now walk forward and stop at first touch:

- Day 1: \$50.80 — between \$48 and \$52, no touch.
- Day 2: \$48.20 — still above the \$48 lower barrier (48.20 > 48.00), no touch. (This is the trap: \$48.20 is *close* to the stop but does not cross it.)
- Day 3: \$49.50 — no touch.
- Day 4: \$52.40 — this is above the \$52.00 upper barrier. **First touch is the upper barrier.**

The label is **+1**. The lesson the interviewer is checking: \$48.20 did not touch the \$48.00 stop, so the trade survived to ride up to the profit target. Get the inequality direction right and read the path in order — a single sloppy comparison flips the label.

#### Worked example: problem 2 — set volatility-scaled barriers

*"An asset trades at \$200. Its exponentially-weighted daily return volatility is 1.5%. Using a 2-sigma rule over a 4-day horizon, where do the barriers go?"*

Apply the square-root-of-time rule to scale the daily vol to the 4-day horizon:

$$
\text{move} = k \cdot \sigma \cdot \sqrt{h} = 2 \times 0.015 \times \sqrt{4} = 2 \times 0.015 \times 2 = 0.06.
$$

So the barrier move is **6%**. The barriers:

- Upper $= \$200 \times (1 + 0.06) = \$212$.
- Lower $= \$200 \times (1 - 0.06) = \$188$.

The follow-up they often ask: *"What if you used $\sigma \cdot h$ instead of $\sigma \cdot \sqrt{h}$?"* Then the move would be $2 \times 0.015 \times 4 = 0.12$, or 12% — barriers at \$224 and \$176, twice as wide. That is the classic error: time-scaling volatility *linearly* instead of by its square root. Daily moves partly cancel, so 4 days of 1.5% vol is about 3% of total move ($1.5\% \times 2$), not 6%. Naming the square-root-of-time rule out loud is what they are listening for.

#### Worked example: problem 3 — does meta-labeling help here?

*"A primary momentum signal has 70% recall but only 45% precision — it catches most winners but more than half its trades lose. Your boss asks whether meta-labeling can help and what it would cost. What do you say?"*

This is meta-labeling's ideal patient: **high recall, low precision** is exactly the profile it fixes. The primary is finding the winners (good recall) but is drowning them in false positives (bad precision). A meta-model trained to predict "will *this* momentum trade win?" can veto the worst candidates and lift precision substantially.

Quantify the cost honestly. Say the meta-filter raises precision from 45% to 65% but drops recall from 70% to 50%. You now trade less often and miss more winners — but each trade is far likelier to pay, and you save the spread and fees on every vetoed loser. The right answer names the tradeoff explicitly (recall for precision), flags that it only helps if the *primary already has real directional skill* (meta-labeling cannot rescue a coin-flip primary — there is nothing to filter), and proposes measuring the net effect on the after-cost Sharpe, not on accuracy.

#### Worked example: problem 4 — size the bet

*"Your meta-model outputs $p = 0.70$ that a long trade wins, paying roughly even money. Your position cap is \$20,000 and the desk runs half-Kelly. How much do you risk?"*

Step through it:

1. Kelly fraction for an even-money bet: $f^* = 2p - 1 = 2(0.70) - 1 = 0.40$.
2. Half-Kelly: $0.5 \times 0.40 = 0.20$.
3. Dollar risk: $0.20 \times \$20{,}000 = \$4{,}000$.

So you risk **\$4,000** of your \$20,000 cap. The interviewer's likely follow-up: *"Why half-Kelly and not full?"* Because full Kelly maximizes long-run *growth* but with brutal volatility, and — critically — your $p = 0.70$ is an *estimate*. If the true probability is really 0.62, full Kelly would have you systematically overbetting, which is far more punishing than underbetting (overbetting can drive growth negative even with a real edge). Fractional Kelly is your margin of safety against an overconfident model.

#### Worked example: problem 5 — sample weights and why they matter

*"You have 4 trades: A alive on days 1-2, B on days 2-3, C on days 3-4, D on day 4 only. Compute each trade's average uniqueness, and explain in one sentence why you cannot just train on these as i.i.d. examples."*

Concurrency per day:

| Day | Trades alive | $c_t$ | $1/c_t$ |
|---|---|---|---|
| 1 | A | 1 | 1.00 |
| 2 | A, B | 2 | 0.50 |
| 3 | B, C | 2 | 0.50 |
| 4 | C, D | 2 | 0.50 |

Average over each trade's life:

- **A** (days 1-2): $(1.00 + 0.50)/2 = 0.75$.
- **B** (days 2-3): $(0.50 + 0.50)/2 = 0.50$.
- **C** (days 3-4): $(0.50 + 0.50)/2 = 0.50$.
- **D** (day 4): $0.50/1 = 0.50$.

A is the most unique (0.75) because it gets day 1 to itself; B, C, and D are all 0.50 because they live entirely on shared days. The one-sentence answer: *the trades' returns overlap on the same days, so treating them as i.i.d. would count those shared days multiple times, inflating the apparent sample size and causing the model and the cross-validation to overfit.* If you can say that sentence cleanly, you have shown the interviewer you understand the deepest point in this whole topic.

#### Worked example: problem 6 — why does the meta-model's threshold matter?

*"Your meta-model is well calibrated. You can set the trading threshold anywhere from $p = 0.50$ to $p = 0.90$. Walk me through what happens to your trade count, precision, and total profit as you raise it, and how you would pick the threshold."*

Picture sweeping the threshold up from 0.50. At $p = 0.50$ you trade everything the primary suggests — maximum trades, lowest precision, you collect every winner but pay for every loser. As you raise the threshold, you progressively veto the least-confident trades first, which (because the model is calibrated) are disproportionately losers. So: **trade count falls, precision rises, recall falls.** Total profit, though, is not monotonic — it traces a hump. Early on, each unit of threshold removes mostly losers, so profit *rises*. Past some point you start vetoing genuine winners faster than losers, and profit *falls* again.

Concretely, suppose at $p = 0.55$ you make \$16,800 (our earlier number), at $p = 0.65$ you make \$18,000 by cutting a few more marginal losers, but at $p = 0.80$ you have thrown away so many winners that profit drops to \$11,000. The optimal threshold is near the *top of the profit hump*, which here is around 0.65 — not the highest-precision point. The answer the interviewer wants: you pick the threshold by maximizing *after-cost profit (or Sharpe) on a validation set*, not by maximizing precision, because precision keeps climbing right up to the point where you barely trade at all. The threshold is a business decision about the precision-recall tradeoff, and it must be chosen out-of-sample to avoid overfitting it to noise.

## Common misconceptions

**"The triple-barrier method is a trading strategy."** No — it is a *labeling* method. It tells you how to turn a price path into a clean +1/0/-1 training label. It does not tell you *when* to enter (that is your primary signal) or *how much* to bet (that is sizing). Confusing the label with the strategy is the most common beginner error; the barriers describe what *would have* happened, after the fact, to a trade your signal proposed.

**"Meta-labeling makes the primary model more accurate."** It does not touch the primary model at all. The primary's directional accuracy is exactly the same before and after. Meta-labeling adds a *second* model that decides whether to *act* on the primary's calls. The improvement comes entirely from abstaining on bad trades and sizing good ones — not from better direction.

**"A higher-recall model is always better."** Recall (catching every winner) sounds great, but a model that trades constantly to catch every winner also eats every loser and pays spread and fees on all of it. After costs, a high-recall, low-precision strategy often loses money. The point of meta-labeling is to *trade recall for precision* on purpose. More trades is not more profit.

**"Sample weights are an optional refinement."** They are not optional when your labels overlap, which they almost always do if you generate labels more often than your holding period. Skipping them makes your model think it has seen far more independent evidence than it has, so your backtest looks great and your live performance falls off a cliff. Overlap correction is load-bearing, not polish.

**"I should use a tighter stop to get cleaner labels."** Tighter barriers are touched more often by noise, so a tight stop in a volatile market labels random wiggles as -1, and your model learns to predict noise. The right barrier width is measured in *volatility*, not cents — that is the entire reason to scale barriers by sigma.

**"The vertical (time) barrier is just a technical detail."** It is the thing that guarantees *every* label resolves. Without it, a trade that never touches either horizontal barrier would never get a label, and you would silently drop exactly the trades that went nowhere — biasing your training set toward dramatic moves and teaching the model that markets always trend. The time barrier keeps the boring, sideways trades in the dataset where they belong.

## How it shows up in real research

**The take-home that screens out half the candidates.** A common quant take-home hands you raw price data and asks you to "build a model that predicts the next move." The candidates who go straight to features and XGBoost score poorly. The ones who *first* construct proper triple-barrier labels, scale the barriers by volatility, and apply sample weights — and *say why* — stand out immediately, because they have shown they understand that the label defines the problem. The modeling is often the easy part; the labeling is the test.

**Cleaning up a discretionary trading desk.** A real and common use of meta-labeling: a desk of human portfolio managers makes directional calls, and a quant team trains a meta-model on the managers' historical calls to predict which ones tend to work. The model never overrides the human's *direction* — it sizes the book, leaning into the setups where that manager has an edge and trimming the ones where they historically misfire. The humans keep the alpha; the model keeps the discipline. This is meta-labeling layered on a non-ML primary, exactly as designed.

**Why a "great" backtest dies in production.** A frequent post-mortem on a strategy that backtested with a Sharpe of 3 and lived at 0.5: the labels overlapped heavily and no sample weights were applied, so the cross-validation thought it had thousands of independent observations when it really had a few hundred. The model overfit the redundancy. Adding uniqueness weights (and purged cross-validation) before the next strategy shipped brought the backtested Sharpe down to a believable 1.2 — which then *held up* live. A lower honest number beats a higher dishonest one every time.

**Sizing as the real alpha.** Several well-known systematic shops have said publicly that their directional models are only modestly better than a coin flip — hit rates in the low-to-mid 50s. What makes them profitable is *sizing*: betting big when the calibrated probability is high and small when it is low, scaled by a fractional-Kelly rule fed by something very much like a meta-model's probability. A 53% directional model, sized well, beats a 58% model sized flat. Meta-labeling is the discipline that turns a mediocre edge into a fundable one.

**The volatility-scaling fix for a regime-blind model.** A momentum model trained with fixed 1% barriers worked beautifully in a calm year and blew up the next year when volatility doubled — because its fixed barriers were now hair-triggers, labeling every intraday wiggle as a stop-out. Re-labeling with 2-sigma volatility-scaled barriers made the labels comparable across both years, and the retrained model held up through the regime change. The fix was entirely in the labels; the features never changed.

**The interview where the answer is "I would not trade that."** A sharp variant of the take-home gives you a primary signal with genuinely zero directional edge — its long calls win exactly 50% of the time. Candidates who reflexively reach for meta-labeling fail this one. With no directional skill in the primary, there is nothing for the meta-model to filter: the "winners" and "losers" are statistically identical, so any threshold the meta-model learns is fitting noise, and it will not generalize. The correct answer is that meta-labeling *amplifies* an existing edge but cannot *create* one — if the primary is a coin flip, the only honest move is to throw it out and find a real signal. Knowing the boundary of a technique is as valuable as knowing the technique, and this is the question that tests it.

## When this matters to you and further reading

If you are prepping for quant research interviews, internalize the *order of operations*: the label defines the problem, the label needs to respect the path and the volatility, overlapping labels need weights, and direction and sizing are separate decisions. That sequence is the skeleton of a strong answer to almost any "how would you model this market?" question — and saying it out loud, before reaching for a model, is what separates candidates who have built real strategies from those who have only run notebooks.

The deeper habit worth taking away is to treat every modeling problem as a labeling problem first. Before you ever choose features or an algorithm, ask: what exactly am I predicting, does that target reflect how the position would really be held and exited, and is each training example carrying genuinely independent information? In most domains those questions have boring answers, so people learn to skip them. In finance the answers are subtle, load-bearing, and exactly where the edge — and the interview points — live. The triple-barrier method and meta-labeling are two specific, battle-tested answers, but the discipline of asking the questions at all is the transferable skill.

The canonical source is Marcos Lopez de Prado's **Advances in Financial Machine Learning** (Wiley, 2018) — chapter 3 covers the triple-barrier method and meta-labeling, chapter 4 covers sample weights and uniqueness, and chapter 10 covers bet sizing. It is dense and worth every page; the ideas in this post are the practitioner's distillation of those chapters. To go deeper on the surrounding machinery, the companion pieces here cover [the financial ML pipeline and purged cross-validation](/blog/trading/quantitative-finance/financial-ml-pipeline-purged-cv-quant-research), [overfitting, purged CV, and the deflated Sharpe ratio](/blog/trading/quantitative-finance/overfitting-purged-cv-deflated-sharpe-quant-research), and [the Kelly criterion for sequential betting](/blog/trading/quantitative-finance/kelly-criterion-sequential-betting-quant-interviews) that the sizing math here points to.

This is educational material about how a class of models is built and evaluated, not investment advice; every strategy that can make money can lose it, and the sizing rules here are about controlling that risk, not eliminating it.
