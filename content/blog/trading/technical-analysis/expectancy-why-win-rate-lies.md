---
title: "What 'Win Rate' Really Means -- and Why It Lies: Expectancy, R-Multiples, and the Math of an Edge"
date: "2026-06-15"
publishDate: "2026-06-15"
description: "Win rate is the most over-marketed and least useful single number in trading. This is the honest math of an edge: expectancy, R-multiples, the breakeven win rate 1/(1+R), why a 40% system can beat a 70% one, and how variance and risk of ruin decide whether you survive long enough to collect."
tags:
  [
    "expectancy",
    "win-rate",
    "r-multiple",
    "reward-to-risk",
    "trading-edge",
    "risk-of-ruin",
    "position-sizing",
    "variance",
    "sample-size",
    "technical-analysis",
    "trading-psychology",
    "money-management",
  ]
category: "trading"
subcategory: "Technical Analysis"
author: "Hiep Tran"
featured: true
readTime: 39
---

> [!important]
> **TL;DR** -- win rate is the most over-marketed and least useful single number in trading. The number that actually decides whether a strategy makes money is **expectancy**: how much you make *on average per trade*, counting wins and losses together.
>
> - Measure trades in **R-multiples**: 1R is the money you put at risk on the trade (entry minus stop). A win that makes twice your risk is **+2R**; a full stop-out is **-1R**. Now every trade speaks the same units regardless of position size.
> - **Expectancy** is $E = p \times \text{avg win} - (1-p) \times \text{avg loss}$. In R-multiples with $-1$R losers, it is $E[R] = p \cdot W - (1-p)$. Positive means the system makes money; negative means it bleeds, no matter how often it wins.
> - The **breakeven win rate** for a reward-to-risk ratio $R$ is $\frac{1}{1+R}$. At 2-to-1 you only need to be right **33%** of the time; at 1-to-1 you need **50%**; at a tight 1-to-2 you need **67%**.
> - A **40% win rate** with +2.5R winners (E = +0.40R) crushes a **70% win rate** with +0.5R winners (E ~ +0.05R gross, negative after costs). High win rate sells; expectancy pays.
> - Even a positive edge has **variance**. The standard error of a win rate is about $\sqrt{p(1-p)/n}$, so a 30-trade track record tells you almost nothing -- and you must size positions to survive the drawdowns (**risk of ruin**) before the edge can compound.

A trading coach sells a system with a "90% win rate." The marketing is not a lie -- the system really does close nine winners for every loser. And it loses money. Not in some unlucky stretch; it loses money *in expectation*, every month, forever. The buyers are baffled. They were told the hard part of trading is being right, and they are right almost every time. How can being right nine times out of ten lose money?

This post is the answer, and it is the honest spine that the rest of this series hangs from. The short version: *being right* and *making money* are different things, and the gap between them is the single most important idea in trading. A "90% win rate" tells you how *often* you win. It tells you nothing about how much you win when you win or how much you lose when you lose -- and that second number is where the money actually is. The coach's system wins 90 cents ninety times and loses ten dollars ten times: ninety times 0.90 is +81, ten times 10 is -100, net **-19** per hundred trades. Right nine times out of ten, and broke.

![Expectancy combines win rate and loss rate with the average win and the average loss into one signed number; win percent feeds only one of two branches, while expectancy multiplies each branch by its payoff and subtracts the loss leg.](/imgs/blogs/expectancy-why-win-rate-lies-1.png)

The diagram above is the mental model for the whole post. One trade splits into two branches -- a win with probability $p$ and a loss with probability $1-p$. Win rate is *only the split*. Expectancy weighs each branch by what it actually pays, adds the win leg, subtracts the loss leg, and collapses the whole thing into one signed number: positive means the strategy grows your account, zero means it goes nowhere, negative means it bleeds. Everything that follows -- R-multiples, the breakeven win rate, the two-system comparison, the variance and the risk of ruin -- is just this picture, made precise.

A note before we start: this is educational. It explains the mechanics and the math of a trading edge so you can read any strategy's claims honestly. It is not advice to trade anything, and it is certainly not advice to trade more. Every method that can make money can lose it, and we will be specific about how.

## Foundations: what a trade outcome is

Before we can talk about expectancy we need a precise, shared vocabulary for a single trade. We will build it from zero, one term at a time, with money.

### A trade, a win, and a loss

A **trade** is one round trip: you enter a position (buy or sell), and later you exit. The trade's **outcome** is simply your exit value minus your entry value, in dollars, after costs. If you bought a stock at \$100 and sold it at \$103, your outcome is +\$3 per share. If you sold at \$98, it is -\$2 per share. A **win** is any trade that ends positive; a **loss** is any trade that ends negative. (A trade that ends exactly flat is a *scratch*; it is rare and we will fold it into "loss" or ignore it.)

The **win rate**, written $p$, is the fraction of your trades that are wins. If 40 of your last 100 trades were wins, your win rate is $p = 0.40$, or 40%. The **loss rate** is $1 - p$ -- the rest. That is the entire content of "win rate." Notice what it does *not* contain: it says nothing about *how big* the wins or losses were. A 40% win rate is consistent with making a fortune and with going broke, and the rest of this post is about which.

### Average win and average loss

The next two numbers are where the money lives. The **average win** is the mean dollar size of your winning trades; the **average loss** is the mean dollar size of your losing trades (we will write it as a positive number and subtract it explicitly). If your winners averaged +\$300 and your losers averaged -\$200, then your average win is \$300 and your average loss is \$200.

Already you can feel the trap in win rate. A trader who wins 40% of the time but whose winners are \$300 and losers are \$200 is doing something very different from a trader who wins 40% of the time with \$50 winners and \$500 losers. Same win rate, opposite fortunes. Win rate alone cannot tell them apart; the average win and average loss can.

### The R-multiple: putting every trade in the same units

Here is the single most useful idea for thinking clearly about trades, and we will use it for the rest of the post. The problem with dollars is that they depend on how big a position you took. A +\$300 trade on a \$50,000 account is enormous; the same +\$300 on a \$5 million account is a rounding error. To compare trades and strategies cleanly we need units that strip out position size. Those units are **R-multiples**.

**1R is the amount you risked on the trade** -- the money you would lose if your protective exit (your *stop-loss*, the price at which you bail out to cap the damage) gets hit. Concretely: if you buy a stock at \$100 and place your stop at \$95, you have decided in advance that you are willing to lose \$5 per share. That \$5 is your 1R for this trade. (A **stop-loss** is just a pre-committed exit price that caps your loss; "your risk" is the distance from entry to stop, times the number of shares.)

Now we measure every outcome as a multiple of that risk:

- A trade that makes exactly what you risked is **+1R**. (Bought at \$100, stop at \$95 so 1R = \$5, sold at \$105 -- you made \$5, which is +1R.)
- A trade that makes twice your risk is **+2R**. (Sold at \$110 -- you made \$10, which is two times your \$5 risk.)
- A full **stop-out** -- price hits your stop and you exit for the planned loss -- is **-1R**. (Price fell to \$95, you lost your \$5, which is -1R by definition.)
- A trade you exit early for half the planned loss is **-0.5R**; a trade that blows past your stop in a gap and costs you \$7 is **-1.4R**.

The beauty is that R-multiples are *position-size-independent*. Whether you traded 100 shares or 10,000, a trade that hit its 2R target is +2R. This lets us compare a tiny scalp and a huge swing trade on one ruler, and -- crucially -- it lets us reason about a strategy's edge without ever mentioning dollars. Throughout the post we will assume, unless we say otherwise, that **losers are -1R** (you take your planned loss and no more) and **winners are some positive multiple $+W$R** (a trade that makes $W$ times your risk).

### The equity curve

Finally, the **equity curve** is just the running total of your account value (or your cumulative R) as the trades pile up, plotted against trade number. It starts at zero, ticks up on wins, ticks down on losses, and wanders. A profitable strategy's equity curve drifts upward over many trades; a losing one drifts down. But -- and this matters enormously, as we will see -- *the path is not a straight line*. Even a great strategy's equity curve has stretches that go sideways or down for a painfully long time. Reading an equity curve honestly means seeing both the drift and the wiggle.

With these five things -- win rate, average win, average loss, the R-multiple, and the equity curve -- we have everything we need to define the only number that actually matters.

## Expectancy: the only number that matters

We are now going to derive expectancy from scratch. It is one line of arithmetic, but it is the line that separates strategies that make money from strategies that don't.

### Deriving it from a single trade

Think about what one trade is *worth to you on average*, before you know whether it will win or lose. With probability $p$ it wins, paying your average win. With probability $1-p$ it loses, costing your average loss. The average outcome -- the **expected value** of one trade, where *expected value* means the probability-weighted average over all the ways it can turn out -- is:

$$E = p \times (\text{avg win}) - (1-p) \times (\text{avg loss})$$

That is **expectancy**: the mean profit (or loss) of one trade, counting the wins and the losses together, weighted by how often each happens. The win leg ($p \times \text{avg win}$) is the money your winners bring in per trade on average; the loss leg ($(1-p) \times \text{avg loss}$) is the money your losers take out per trade on average. Subtract, and you have your edge per trade.

If $E$ is positive, the strategy makes money on average -- every trade is, in expectation, a small deposit into your account. If $E$ is zero, you go nowhere (minus costs, which means you lose). If $E$ is negative, the strategy is a slow withdrawal, and trading it more just empties the account faster. **This single sign -- the sign of $E$ -- is what "having an edge" actually means.** Not your win rate. The sign of $E$.

### Expectancy in R-multiples

R-multiples make this even cleaner. Adopt our standing assumption: losers are $-1$R and winners are $+W$R, where $W$ is the average reward-to-risk of your winners. Then "avg loss" is just 1 (one R), and "avg win" is $W$ (R), and expectancy in R is:

$$E[R] = p \cdot W - (1-p) \cdot 1 = p\,W - (1 - p)$$

Read the symbols out loud: $p$ is the win rate, $W$ is the average size of a winner in R, and $1-p$ is the loss rate (each losing $1$R). The whole formula says: *the R you win on average from your wins, minus the R you lose on average from your losses.* The answer comes out in **R per trade**, a pure number you can compare across any strategy on Earth.

A concrete reading: suppose $E[R] = +0.3$. That means **every trade you take is worth, on average, three-tenths of your risk unit.** It does not mean every trade makes 0.3R -- most individual trades will be a clean $-1$R or a clean $+2$R or whatever. It means that if you average over many trades, you net +0.3R apiece. Over 100 trades at +0.3R each you would expect roughly **+30R** of accumulated profit. If you risked \$200 (1R = \$200) on each trade, that is about +\$6,000 over those 100 trades -- on average, with a lot of wiggle around it. A "+0.3R per trade" edge is, by trading standards, very good. Many professional strategies live between +0.1R and +0.4R per trade and make a great living off the difference, simply by taking the trade thousands of times.

#### Worked example: what +0.3R buys you over 100 trades

Let us make the "+0.3R per trade" concrete. Take a strategy that wins 50% of the time, with winners at +1.6R and losers at -1R. Its expectancy is:

$$E[R] = 0.50 \times 1.6 - 0.50 \times 1.0 = 0.80 - 0.50 = +0.30\text{R per trade}.$$

Now run 100 trades at \$200 of risk each (1R = \$200). On average you win 50 and lose 50. The 50 winners bring in $50 \times 1.6\text{R} = 80\text{R}$; the 50 losers cost $50 \times 1.0\text{R} = 50\text{R}$. Net: $80 - 50 = +30\text{R}$, which at \$200 per R is **+\$6,000**. Same answer as the per-trade view, because expectancy is *linear*: 100 trades at +0.3R is +30R, full stop. The one-sentence intuition: **expectancy is a per-trade wage, and the number of trades is your hours -- the edge only turns into money when you put in the trades.**

This is also why over-trading a *negative*-expectancy system is so destructive. If $E[R] = -0.05$, then more trades means more guaranteed bleed: 100 trades is -5R, 1,000 trades is -50R. The strategy doesn't "come good" with volume; volume just delivers the loss faster. The sign of $E$ is destiny; the number of trades is the speed.

## The breakeven win rate

We can now answer a question that demolishes most win-rate marketing: *given a reward-to-risk ratio, what win rate do you need just to break even?* The answer is a clean little formula, and it is one you should commit to memory.

### Deriving 1/(1+R)

Set expectancy to zero and solve for the win rate. With winners at $+W$R and losers at $-1$R (and we will write the reward-to-risk ratio as $R = W$, the size of a winner relative to a loser), breakeven means:

$$p \cdot R - (1 - p) \cdot 1 = 0.$$

Expand and collect the $p$ terms:

$$pR - 1 + p = 0 \quad\Longrightarrow\quad p(R + 1) = 1 \quad\Longrightarrow\quad p = \frac{1}{1 + R}.$$

So the **breakeven win rate** for a reward-to-risk of $R$ is $\frac{1}{1+R}$. Above that win rate you make money; below it you lose; exactly at it you tread water (and lose to costs). That is the whole story of how often you need to be right, and notice that it depends *entirely* on $R$ -- on how big your wins are relative to your losses -- and not at all on anything else.

![The breakeven win rate equals one over one plus the reward-to-risk ratio, so a larger payoff lets you lose more trades; at 2-to-1 you only need to win 33 percent of the time, at 1-to-1 you need 50 percent.](/imgs/blogs/expectancy-why-win-rate-lies-2.png)

The figure plots $\frac{1}{1+R}$ as $R$ climbs from 0.5 to 5. Each column shows the breakeven win rate you need at that reward-to-risk: the green part is "win rates above this, you profit," the red part is "win rates below this, you lose." The line falls fast. The more your winners dwarf your losers, the less often you need to be right. Here is the same thing as a table you can memorize:

| Reward-to-risk $R$ | Breakeven win rate $\frac{1}{1+R}$ | What it means |
| --- | --- | --- |
| 1 : 2 (win 0.5R, lose 1R) | $\frac{1}{1.5} \approx 67\%$ | tight targets: you must win two of every three |
| 1 : 1 (win 1R, lose 1R) | $\frac{1}{2} = 50\%$ | the coin-flip line; you must beat 50% |
| 2 : 1 (win 2R, lose 1R) | $\frac{1}{3} \approx 33\%$ | you can be wrong two times in three and still win |
| 3 : 1 (win 3R, lose 1R) | $\frac{1}{4} = 25\%$ | wrong three times in four and still profitable |
| 5 : 1 (win 5R, lose 1R) | $\frac{1}{6} \approx 17\%$ | wrong five times in six and still ahead |

### Why a high R-to-R lets you be wrong most of the time

Stare at the 3-to-1 row. A strategy that wins 3R when it wins and loses 1R when it loses only needs to be right **25%** of the time to break even -- it can be *wrong three times out of four* and still not lose money. This is the mechanical reason trend-following works, and we will meet it again in the real-markets section: trend-followers are wrong most of the time, take a string of small -1R losses, and ride the occasional huge winner to +5R, +10R, or more. Their win rate looks terrible. Their expectancy is excellent, because $R$ is enormous and the breakeven win rate is tiny.

Now stare at the 1-to-2 row. A strategy with *tight* targets -- it grabs 0.5R and runs, but lets losers reach the full 1R -- needs to win **67%** of the time just to break even. This is the hidden cost of the "high win rate" systems people sell. To inflate the win rate, you take profits early (small wins) and give losers room (full losses), which pushes $R$ below 1 and shoves the breakeven win rate *above* your actual win rate. You feel like a genius -- you win most of your trades! -- while quietly running a negative expectancy. We will see exactly this in the next section.

#### Worked example: breakeven win rate for 1:1, 2:1, 3:1, and 1:3

Let us compute four of these by hand so the formula is muscle memory, and attach dollars. Say 1R = \$100 throughout.

- **1:1.** $R = 1$, so breakeven $p = \frac{1}{1+1} = 50\%$. If you win exactly half: 50 wins of +\$100 = +\$5,000, 50 losses of -\$100 = -\$5,000, net **\$0**. Spot on the line.
- **2:1.** $R = 2$, so breakeven $p = \frac{1}{1+2} \approx 33.3\%$. At a 33.3% win rate over 100 trades: 33.3 wins of +\$200 = +\$6,667, 66.7 losses of -\$100 = -\$6,667, net **\$0**. You were wrong two-thirds of the time and broke even.
- **3:1.** $R = 3$, so breakeven $p = \frac{1}{1+3} = 25\%$. At 25%: 25 wins of +\$300 = +\$7,500, 75 losses of -\$100 = -\$7,500, net **\$0**. Wrong three-quarters of the time, flat.
- **1:3.** $R = \frac{1}{3}$ (you win 0.33R, lose 1R), so breakeven $p = \frac{1}{1 + 1/3} = \frac{1}{4/3} = 75\%$. You need to win **three out of four** just to tread water -- and if your real win rate is 70%, this "high win rate" strategy *loses money*.

The one-sentence intuition: **the reward-to-risk ratio sets the bar; your win rate only matters relative to that bar.** A 30% win rate clears a low bar (2:1 or 3:1) easily; a 70% win rate fails a high bar (1:3) badly.

## Why a 70% win rate can lose and a 40% can win

Now we put the two halves together and watch the marketing collapse. We will compare two systems with full arithmetic.

**System A** -- the ugly one. Win rate **40%**. Winners average **+2.5R**, losers **-1R**. It loses more often than it wins. By win rate alone it looks bad.

**System B** -- the pretty one. Win rate **70%**. Winners average **+0.5R**, losers **-1R**. It wins more than two-thirds of the time. By win rate alone it looks great, and this is exactly the kind of system that gets sold.

Compute the expectancy of each.

$$E_A[R] = 0.40 \times 2.5 - 0.60 \times 1.0 = 1.00 - 0.60 = +0.40\text{R per trade}.$$

$$E_B[R] = 0.70 \times 0.5 - 0.30 \times 1.0 = 0.35 - 0.30 = +0.05\text{R per trade}.$$

System A, with its embarrassing 40% win rate, makes **+0.40R per trade**. System B, with its beautiful 70% win rate, makes **+0.05R per trade** -- and that is *before costs*. Subtract a realistic round-trip cost of about 0.1R (the spread you cross plus commissions, expressed in risk units) and System B flips to **-0.05R per trade**: a net loser. The pretty system loses money; the ugly one is a machine.

![System A wins under half the time yet earns plus 0.40R per trade, while System B wins far more often yet nets near zero gross and goes negative after costs, because the low-win-rate system has winners five times the size of its losers.](/imgs/blogs/expectancy-why-win-rate-lies-3.png)

The table figure lays out every step side by side. Look at the two "contribution" rows. System A's winners contribute +1.00R per trade because they are large (2.5R) even though they are infrequent (40%); System B's winners contribute only +0.35R because they are tiny (0.5R) despite being frequent (70%). Meanwhile both systems lose the same per losing trade (1R), but System A loses more *often* (60%), costing 0.60R, versus System B's 0.30R. The arithmetic favors A overwhelmingly: its huge winners more than pay for its frequent losers.

### The payoff-ratio trap

What you just saw is the **payoff-ratio trap**, and it is the single most common way traders fool themselves. The payoff ratio is just $R$ -- average win over average loss. Win rate and payoff ratio trade off against each other, and *you can manipulate one at the expense of the other*. Take profits earlier and you win more often (higher $p$) but smaller (lower $R$). Let winners run and you win less often (lower $p$) but bigger (higher $R$). The win rate is the knob people show you; the payoff ratio is the knob that actually moves the money, and it moves it the *opposite* way.

So when someone advertises a 90% or 70% win rate, the very first question is: **at what reward-to-risk?** Because by $p = \frac{1}{1+R}$, a 70% win rate only beats breakeven if $R > \frac{1-0.70}{0.70} = \frac{0.30}{0.70} \approx 0.43$ -- the winners must average at least 0.43R. If the winners are 0.5R, you barely clear it gross and lose after costs (System B exactly). If the winners are smaller still, the gorgeous win rate is a money-losing machine dressed as a winner.

#### Worked example: 100 trades of System A versus System B

Project both systems over 100 trades, with 1R = \$200, and include the 0.1R cost on every trade.

**System A** (40% win, +2.5R / -1R, -0.1R cost):
- 40 winners: $40 \times 2.5\text{R} = 100\text{R}$.
- 60 losers: $60 \times 1.0\text{R} = 60\text{R}$ lost.
- 100 trades of cost: $100 \times 0.1\text{R} = 10\text{R}$.
- Net: $100 - 60 - 10 = +30\text{R}$, which at \$200 per R is **+\$6,000**.

**System B** (70% win, +0.5R / -1R, -0.1R cost):
- 70 winners: $70 \times 0.5\text{R} = 35\text{R}$.
- 30 losers: $30 \times 1.0\text{R} = 30\text{R}$ lost.
- 100 trades of cost: $100 \times 0.1\text{R} = 10\text{R}$.
- Net: $35 - 30 - 10 = -5\text{R}$, which at \$200 per R is **-\$1,000**.

Same 100 trades, same \$200 risk per trade. The 40%-win-rate system makes **+\$6,000**; the 70%-win-rate system loses **\$1,000**. The trader running System B was *right* 70 times and still went home poorer. The one-sentence intuition: **how often you win is a feeling; expectancy is the bank statement, and they routinely disagree.**

### Why tight targets inflate win rate but destroy expectancy

There is a behavioral reason the trap is so seductive, and it is worth naming. Taking a small profit feels *wonderful* -- you booked a win, you were right, the chart proved you correct. Holding for a bigger target and watching the trade come back to your entry feels *terrible* -- you "gave back" profit. So traders, chasing the good feeling and dodging the bad one, systematically cut winners short and let losers run (hoping they come back). That behavior maximizes win rate and *minimizes* expectancy. It is the emotionally natural thing to do, and it is precisely backwards. The honest version of "let your winners run and cut your losers short" is just: *push $R$ up, even though it pushes your win rate down, because $R$ is the knob connected to the money.*

## Variance, sample size, and luck

So far everything has been about averages. But you do not trade the average -- you trade one actual sequence of wins and losses, and that sequence is *random*. Two questions follow, and they are where most people's reasoning about their own track record falls apart. First: how do you tell a real edge from a lucky streak? Second: how long before the average reliably shows up? Both are answered by the same statistics.

### The equity curve is a random walk with drift

A positive-expectancy strategy's equity curve is a **random walk with positive drift**. "Random walk" means each trade adds a random step (a win or a loss); "positive drift" means the average step is positive (your +0.3R, say). Over the long run, the drift wins and the curve climbs. But over any short run, the *randomness* can completely swamp the drift. A +0.4R strategy can lose ten trades in a row. A coin-flip strategy can win fifteen in a row. The drift is the signal; the wiggle is the noise; and over small samples the noise is far louder than the signal.

This is exactly why you cannot read your edge off a short stretch of equity curve, and it connects directly to the [law of large numbers and the central limit theorem](/blog/trading/math-for-quants/law-large-numbers-central-limit-theorem-math-for-quants): the average converges to the true expectancy only as the number of trades grows, and the *speed* of that convergence is governed by the standard error.

### The standard error of a win rate

Here is the tool. If your true win rate is $p$ and you observe $n$ trades, the observed win rate $\hat{p}$ has a **standard error** -- the typical size of its random wobble around the true value -- of approximately:

$$\text{SE}(\hat{p}) \approx \sqrt{\frac{p(1-p)}{n}}.$$

The symbols: $p$ is the true win rate, $n$ is the number of trades, and the result is the typical distance between what you *observe* and what is *true*. The key feature is the $n$ in the denominator under a square root: to halve your uncertainty you need **four times** as many trades. Uncertainty shrinks slowly. A rough 95% confidence interval (the range your observed win rate will fall in about 19 times out of 20) is $\hat{p} \pm 2\,\text{SE}$.

![The 95 percent confidence band on a 55 percent win rate shrinks as the square root of sample size; at 30 trades the band straddles 50 percent so a thin edge looks like a coin, and only past 100 trades does the band clear 50 percent.](/imgs/blogs/expectancy-why-win-rate-lies-6.png)

The figure shows the funnel. Take a true win rate of 55% -- a real, tradeable edge in a 1:1 system (breakeven is 50%, so 55% is comfortably profitable). Plot the 95% confidence band as the sample grows. At $n = 10$ the band runs from roughly 24% to 87% -- you could observe almost anything. At $n = 30$ it is still about 37% to 73%, *straddling 50%*: your 30-trade results are statistically indistinguishable from a coin. Only around $n = 100$ does the lower edge of the band finally clear 50% (about 45% to 65%), and you can start to argue you have an edge rather than luck. At $n = 300$ the band tightens to about 49% to 61%, and at $n = 1{,}000$ to about 52% to 58%. The edge was always there; it just takes hundreds of trades for the data to *prove* it.

### How many trades to distinguish a 55% edge from a coin

We can make "distinguish from a coin" precise with a **t-statistic** -- the number of standard errors between your observed result and the no-edge baseline. To claim a real edge, you want your win rate to sit a couple of standard errors above 50%. The gap you are trying to detect is $0.55 - 0.50 = 0.05$. The standard error at sample size $n$ (using $p \approx 0.5$, so $p(1-p) \approx 0.25$) is $\sqrt{0.25/n} = \frac{0.5}{\sqrt{n}}$. You need:

$$\frac{0.05}{\,0.5/\sqrt{n}\,} = \frac{0.05\,\sqrt{n}}{0.5} = 0.1\sqrt{n} \geq 2 \quad\Longrightarrow\quad \sqrt{n} \geq 20 \quad\Longrightarrow\quad n \geq 400.$$

So you need on the order of **400 trades** before a genuine 55% edge clears the two-standard-error bar with confidence. (For a stronger 60% edge the gap doubles to 0.10 and the requirement drops to about $n \geq 100$; for a razor-thin 52% edge it explodes to over 2,500 trades.) The thinner the edge, the more trades you need to prove it -- quadratically more. This is the same logic the desks use when they decide whether a [backtested signal is real or overfit](/blog/trading/quantitative-finance/backtesting-done-right-quant-research): a great-looking result on a small sample is mostly noise.

#### Worked example: is 17 wins out of 30 "evidence"?

Someone shows you a track record: **17 wins in 30 trades**, a 57% win rate, and claims a clear edge. Let us check it. With \$100 of risk per trade and 1:1 payoffs, 17 wins of +\$100 and 13 losses of -\$100 nets $17 \times 100 - 13 \times 100 = +\$400$ -- a real-looking profit. But is the *win rate* statistically distinguishable from a coin?

The observed $\hat{p} = 17/30 \approx 0.567$. Under the null hypothesis of a fair coin ($p = 0.5$), the standard error over 30 trades is $\sqrt{0.5 \times 0.5 / 30} = \sqrt{0.00833} \approx 0.091$, or 9.1%. The t-statistic is:

$$t = \frac{0.567 - 0.50}{0.091} = \frac{0.067}{0.091} \approx 0.73.$$

A t-statistic of 0.73 is *nothing* -- you want roughly 2 or more to claim significance. In plain terms: a fair coin flipped 30 times produces 17-or-more heads about **29% of the time**. Almost a third of coins would "beat" this track record by luck alone. The \$400 profit is real money, but as *evidence of skill* it is worthless. Now contrast 170 wins in 300 trades (the same 56.7%): the standard error drops to $\sqrt{0.25/300} \approx 0.029$, the t-statistic rises to $0.067 / 0.029 \approx 2.3$, and *now* you have something -- a fair coin produces that result only about 1% of the time. Same win rate, ten times the sample, completely different conclusion. The one-sentence intuition: **a small sample cannot tell skill from luck, and 30 trades is a small sample.** This is the heart of [expected-value reasoning under uncertainty](/blog/trading/quantitative-finance/expected-value-techniques-quant-interviews): the point estimate is only as trustworthy as the sample behind it.

### Why 20 trades tell you almost nothing

Put a sharper point on it. The standard error at $n = 20$ for a coin is $\frac{0.5}{\sqrt{20}} \approx 0.112$ -- about 11 percentage points. So a 20-trade win rate carries a $\pm 22$-point 95% band. An observed 60% over 20 trades is consistent with a *true* win rate anywhere from about 38% to 82%. You genuinely cannot tell, from 20 trades, whether you have a strong edge, no edge, or a losing system. This is why "I tried it for a couple of weeks and it works" is meaningless, and why honest strategy evaluation is measured in hundreds of trades, not dozens. The fewer trades, the more the result is a story your luck is telling you.

## Drawdown and risk of ruin (preview)

Positive expectancy says you make money *on average over many trades*. It does not promise you survive long enough to collect. Between you and the long run sits **variance**, and variance shows up as **drawdown** -- the peak-to-trough decline in your equity. Even a great system has losing streaks, and a losing streak you cannot financially or emotionally survive ends the game before the edge can pay off.

### Losing streaks are normal, not a malfunction

A strategy that wins 40% of the time loses 60% of the time, and independent losses string together more often than intuition expects. The probability of $k$ losses in a row, when each loss has probability $1-p$, is:

$$P(k \text{ losses in a row}) = (1-p)^k.$$

For our 40%-win System A ($1-p = 0.60$): the chance of 5 straight losses on any given starting trade is $0.60^5 \approx 0.078$, about 1 in 13. Over a few hundred trades, a run of 5, 6, even 8 consecutive losses is not just possible -- it is *expected to happen*. That is a stretch of -5R to -8R with no relief, inside a system whose long-run expectancy is firmly positive. Nothing has broken. The edge is intact. You are simply living through the variance, and if you panic and quit at the bottom of the streak you convert a winning system into a realized loss.

### The path matters: same total, different drawdown

Here is a fact that surprises almost everyone. Two strategies can produce the **exact same set of trade outcomes** and the **exact same final profit**, yet one is comfortable to trade and the other nearly impossible -- because the *order* of the wins and losses differs, and order determines drawdown.

![Two orderings of the same ten R-multiples reach the same plus 2R total, but the favorable order never falls more than 2R while the losses-first order sinks to minus 6R before recovering, so the path, not just the total, decides survival.](/imgs/blogs/expectancy-why-win-rate-lies-8.png)

The figure takes ten trades -- four winners of +2R and six losers of -1R -- which sum to $4 \times 2 - 6 \times 1 = +2$R no matter how you arrange them. The green path interleaves them favorably and never draws down more than 2R; it is a calm ride to +2R. The amber path front-loads all six losses, plunging to **-6R** before the winners arrive and haul it back to the same +2R finish. Identical trades, identical total, but the amber path puts you 6R underwater -- a 6R drawdown that, if 1R is 2% of your account, is a stomach-churning 12% loss before you recover. Same expectancy, wildly different experience, and the amber path is the one that makes people quit at the worst possible moment.

#### Worked example: hand-simulating a ten-trade equity curve

Let us walk the amber path one trade at a time, with 1R = \$500 (so the account starts at, say, \$25,000 and each R is 2% of it). The R sequence is $[-1, -1, -1, -1, -1, -1, +2, +2, +2, +2]$.

| Trade | Outcome | Cumulative R | Account (\$) | Drawdown |
| --- | --- | --- | --- | --- |
| start | -- | 0 | 25,000 | -- |
| 1 | -1R | -1 | 24,500 | -1R |
| 2 | -1R | -2 | 24,000 | -2R |
| 3 | -1R | -3 | 23,500 | -3R |
| 4 | -1R | -4 | 23,000 | -4R |
| 5 | -1R | -5 | 22,500 | -5R |
| 6 | -1R | -6 | 22,000 | **-6R (trough)** |
| 7 | +2R | -4 | 23,000 | recovering |
| 8 | +2R | -2 | 24,000 | recovering |
| 9 | +2R | 0 | 25,000 | back to flat |
| 10 | +2R | +2 | 26,000 | +2R, new high |

You end at \$26,000 -- a tidy +\$1,000, exactly +2R. But you got there only after watching \$3,000 (12% of the account) evaporate by trade 6. Now re-order the *same ten trades* as $[+2, -1, +2, -1, -1, +2, -1, +2, -1, -1]$ and the cumulative path is $+2, +1, +3, +2, +1, +3, +2, +4, +3, +2$ -- the same \$26,000 finish, but the worst drawdown is only 2R (\$1,000, 4% of the account). The one-sentence intuition: **expectancy tells you where you end up; the path tells you whether you're still in the game to get there.**

### Risk of ruin: why position sizing keeps you alive

This is where **position sizing** -- how much of your account you risk per trade -- becomes the thing that actually keeps you solvent. **Risk of ruin** is the probability that a string of losses drives your account below the point where you can continue (often defined as some fixed percentage drawdown, or literally to zero). It depends on three things: your edge, your win rate, and *how much you bet per trade*. Crucially, it depends on bet size *enormously*.

![Risk of ruin climbs fast as you bet more per trade with a thin edge; the same 55 percent even-money edge ruins almost never at 2 percent risk per trade but about a quarter to two-thirds of the time at 20 to 25 percent risk per trade.](/imgs/blogs/expectancy-why-win-rate-lies-7.png)

The matrix figure shows risk of ruin for the same edges at different bet sizes. Read the 55%-win column (a real but thin edge): risking 2% of your account per trade, your probability of ruin is under 1% -- you are essentially safe. Risking 10% per trade, it climbs to about 36%. Risking 20-25% per trade, you ruin yourself **60-67% of the time** -- *with a positive edge*. The edge does not save you; the bet size kills you. And in the 50%-win column (no edge at all), every bet size leads to near-certain ruin eventually, because a zero-drift random walk hits any boundary given enough time.

The deep lesson, which the [Kelly criterion](/blog/trading/quantitative-finance/kelly-criterion-sequential-betting-quant-interviews) makes exact, is that there is an *optimal* bet size that maximizes long-run growth, and betting more than that does not grow your account faster -- it grows your risk of ruin faster while *lowering* your growth. Most professionals bet a fraction of the Kelly amount precisely because the penalty for over-betting is so severe and their estimate of the edge is uncertain. Expectancy answers "should I trade this?" Position sizing answers "how do I survive trading it?" -- and you need both, in that order, to make money. We treat sizing in full in the Kelly post; here the takeaway is just that a positive edge is *necessary but not sufficient*, and the thing standing between an edge and a blown account is how much you risk per trade.

## Common misconceptions

Six beliefs that feel true, cost money, and that the math above corrects.

**"A high win rate means a strategy is profitable."** The single biggest one, and the whole reason for this post. Win rate is one of *four* numbers (with average win, average loss, and number of trades) that determine profit, and it is the only one that says nothing about the size of outcomes. A 90% win rate with 1:10 reward-to-risk loses money; a 30% win rate with 5:1 makes a fortune. Always ask for the win rate *and the payoff ratio together* -- one without the other is marketing, not information.

**"A winning streak means the system got better."** No. A positive-expectancy system is a random walk with drift; streaks -- winning and losing -- are the *expected behavior* of randomness, not evidence of anything changing. Eight wins in a row from a 55% system is luck, exactly as eight losses in a row is. Treating a hot streak as proof of improvement (and sizing up) is how traders give back their gains; treating a cold streak as proof of breakage (and quitting) is how they lock in losses. The system did not change; your sample did.

**"I just need to win more than half my trades."** Only true at exactly 1:1 reward-to-risk. The real bar is $\frac{1}{1+R}$, which can be far below 50% (you need only 25% at 3:1) or far above it (you need 67% at 1:2). "Win more than half" is a special case people mistake for a law. The law is the breakeven formula; "more than half" is just the slice of it where $R = 1$.

**"Expectancy is enough -- ignore variance."** Expectancy tells you the *average*, but you trade one *path*, and the path can ruin you before the average arrives. A +0.3R system bet at 25% per trade can blow up before its edge ever shows. Positive expectancy is necessary; surviving the variance, through small position sizing, is what lets you *collect* it. Expectancy and risk management are a pair, never one without the other.

**"More trades is always more data."** More *independent* trades is more data. But many trades overlap -- you hold ten correlated positions in the same sector, or you take the same setup repeatedly in one trending regime -- and overlapping or regime-bound trades carry far less information than their count suggests. A hundred trades that are really one bet repeated under one market condition is closer to *one* data point than a hundred. Sample size counts independent bets, not clicks.

**"My backtest's win rate will be my live win rate."** Backtests are run on the past, often with hindsight baked in (you knew which stocks survived, you tuned the rules to fit the curve). Live trading adds costs, slippage, and the small but relentless ways reality differs from a clean historical series. Live win rates and live R-multiples are almost always a little worse than the backtest, and a strategy whose backtested edge is thin can cross into negative expectancy the moment real costs land on it. Treat the backtest as an optimistic upper bound, not a promise.

## How it shows up in real markets

Five recognizable patterns where this math plays out with real money. Named where possible, with as-of caveats, because the specifics matter and they go stale.

### Trend-following CTAs: ~35-40% win rates, wildly profitable

Managed-futures trend-followers -- the systematic CTAs (Commodity Trading Advisors) like the strategies historically associated with firms such as Winton, AHL (Man Group), and the late Bill Dunn's Dunn Capital -- are the textbook case of a *low* win rate with a *high* expectancy. Their hit rates on individual trades commonly run around 35-40%; they are wrong on the majority of their trades. They survive and thrive because of $R$: they cut losers fast (a string of small -1R stops as a market chops) and let the rare big trend run for +5R, +10R, or more. The 2008 financial crisis was a banner year for many trend-followers precisely because a few enormous trends (a collapse in equities, a spike in bonds) delivered the giant winners that the formula $p = \frac{1}{1+R}$ says you only need a quarter of the time to win. Their equity curves are choppy and spend long stretches in drawdown -- and clients who fixate on the low win rate or bail during the flat years routinely miss the payoff. As of the mid-2020s, trend-following remains a multi-hundred-billion-dollar category built on a deliberately low win rate.

### Option sellers and scalpers: 90% win rates that blow up in one tail

The mirror image is the strategy that sells insurance: writing out-of-the-money options, shorting volatility, or scalping tiny edges. These win **almost all the time** -- the option expires worthless, the scalp closes for a few ticks -- racking up a 90%+ win rate that looks like genius and prints a smooth, beautiful equity curve. The problem is the loss distribution: when the rare bad event hits, the loss is not -1R, it is -20R or -50R, enough to erase years of small wins in a single session. The pattern has a name on Wall Street: **picking up pennies in front of a steamroller.** The canonical blow-ups are real and repeated: Long-Term Capital Management in 1998 (a portfolio of high-win-rate convergence bets that a tail event detonated), the "Volmageddon" of February 5, 2018 (the short-volatility ETP known as XIV lost about 90% of its value overnight and was wound down, after years of looking like free money), and a long graveyard of option-selling funds that posted gorgeous track records right up until the month they didn't. The expectancy of these strategies can be *negative* even while the win rate is 90%+, because the rare loss is so large it overwhelms the frequent small wins -- exactly the coach's "90% win rate" system from the opening, at institutional scale.

### The signal seller with a great-looking 30-trade record

A subscription service or social-media trader posts a track record: 23 wins, 7 losses over 30 trades, a 77% win rate, screenshots of green P&L. It looks undeniable. But we did this math: 30 trades is a sample so small that even a coin produces flashy results regularly, and a *selectively reported* 30 trades (showing the good run, quietly dropping the bad one) is worse than worthless. With no audited, continuous, hundreds-of-trades record, a great-looking short track record is indistinguishable from luck or cherry-picking. The standard error over 30 trades is around 9 percentage points; the "edge" you are being sold is inside the noise. This is the most common retail trap, and the defense is simply the sample-size math: **demand hundreds of trades, audited and continuous, or assume it's variance.**

### Why casinos run a tiny edge over a huge number of bets

The cleanest real-world proof that expectancy plus volume beats win rate is the casino. On a single-zero roulette wheel the house edge is about **2.7%** -- the casino's expectancy per dollar bet is roughly +\$0.027, and the *player's* win rate on an even-money bet (red/black) is about 48.6%, just under half. The house does not win most individual bets; it wins a hair more than half, on a tiny per-bet edge. But it takes the bet *millions of times*, and by the law of large numbers the variance washes out and the 2.7% drift becomes a near-deterministic river of profit. The casino is a positive-expectancy machine running a thin edge over enormous $n$ -- the exact opposite of the gambler at the table, who has a negative edge over a small $n$ and whose occasional hot night is pure variance. A professional trading operation is built to be the casino, not the gambler: a small, real, *measured* edge, taken thousands of times, sized to survive the swings.

### The same setup in a single regime is one bet, not a hundred

A subtler real-market trap, worth its own note because it fools sophisticated people. Suppose you trade a mean-reversion setup that worked beautifully through a calm, range-bound year -- 80 winners, 20 losers, an 80% win rate over 100 trades, gorgeous expectancy. But all 100 trades happened in *one volatility regime*. They were not 100 independent draws from the strategy's true distribution; they were closer to one big bet on "the market stays range-bound," repeated. When the regime changes -- a trend starts, volatility spikes -- the mean-reversion setup can flip to a string of large losses, and the 80% win rate evaporates. This is why honest strategy evaluation insists on trades across *multiple regimes and instruments*, and why "100 trades" in one market condition is far weaker evidence than the number suggests. It connects back to the misconception above: data is *independent bets*, and a regime is a hidden way your bets are all the same bet.

## When this matters to you and further reading

This is the spine of everything else in this series, so here is where it actually touches your decisions. The next time you read *any* strategy claim -- an indicator's "win rate," a signal service's track record, a backtest, your own last month of trades -- run the same three checks. First, **win rate and payoff ratio together**: a win rate without an average win and average loss is not information, and the breakeven bar is $\frac{1}{1+R}$, not 50%. Second, **expectancy, in R**: $E[R] = p\,W - (1-p)$ -- is the sign positive, and is it still positive after realistic costs? Third, **sample size and variance**: how many *independent* trades is this built on, what is the standard error, and could you survive the drawdowns the variance will hand you? A claim that can't answer all three is marketing.

The distribution figure below is worth keeping in your head as the picture of a *real* edge -- not a smooth 90%-win equity curve, but a messy histogram of mostly small losses paid for by a few large wins.

![A profitable trend-follower's R-multiples are right-skewed: many minus 1R full stops paid for by a handful of winners running to plus 3R and beyond, so the strategy makes money despite losing most of its trades.](/imgs/blogs/expectancy-why-win-rate-lies-5.png)

This figure is what a winning low-win-rate strategy actually looks like under the hood: a tall red bar of -1R full stops -- most of the trades -- and a long green tail of winners stretching out to +3R, +6R, and a single home run beyond. There are 24 losers of -1R and 16 winners that average about +2.5R, so the win rate is 40% and the expectancy works out to $0.40 \times 2.5 - 0.60 \times 1.0 = +0.40$R per trade -- a profitable system that nonetheless *loses the majority of its trades*, because the green tail is heavy enough to pay for the red bar many times over. If your mental image of a "good system" is a high win rate and a smooth line, replace it with this picture: frequent small losses, rare large wins, and a positive number at the bottom of the page.

And here is the comparison that should stay with you longest -- two equity curves from the same forty trades, one for each system we studied.

![Two equity curves from the same forty trades show the low-win-rate positive-expectancy system climbing to about plus 15R net while the high-win-rate negative-expectancy system bleeds to about minus 2R, proving that expectancy not win rate decides the outcome.](/imgs/blogs/expectancy-why-win-rate-lies-4.png)

The green curve is System A -- 40% win rate, +2.5R winners -- choppy, full of small dips, but grinding relentlessly upward to about +15R net of costs. The amber curve is System B -- 70% win rate, +0.5R winners -- which looks fine for a while and then bleeds steadily to about -2R, undone by costs eating an edge that was never really there. The trader running the amber system won far more often and lost money; the trader running the green system was wrong most of the time and got rich. That single image is the thesis of this entire series.

Where to go next. To turn a positive expectancy into the *right* bet size -- the amount that maximizes long-run growth without courting ruin -- read [the Kelly criterion and sequential betting](/blog/trading/quantitative-finance/kelly-criterion-sequential-betting-quant-interviews). To sharpen the expected-value reasoning underneath all of this, see [expected-value techniques](/blog/trading/quantitative-finance/expected-value-techniques-quant-interviews). To understand *why* large samples converge and small ones lie, read [the law of large numbers and the central limit theorem](/blog/trading/math-for-quants/law-large-numbers-central-limit-theorem-math-for-quants). To measure an edge without fooling yourself with hindsight, see [backtesting done right](/blog/trading/quantitative-finance/backtesting-done-right-quant-research). And to ground all of this in what a chart and a "setup" even are, start from [what technical analysis really is](/blog/trading/technical-analysis/what-technical-analysis-really-is) and [why support and resistance levels exist](/blog/trading/technical-analysis/support-and-resistance-why-levels-exist) -- because no level, pattern, or indicator means anything until you can say what its expectancy is.
