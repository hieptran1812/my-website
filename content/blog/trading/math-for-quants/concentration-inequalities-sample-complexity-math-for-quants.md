---
title: "Concentration inequalities: how many trades before you believe an edge"
date: "2026-08-30"
publishDate: "2026-08-30"
description: "The central limit theorem tells you what happens eventually. Concentration inequalities tell you what is guaranteed after a finite number of trades, which turns 'is this edge real?' into a number of observations you can actually count."
tags: ["concentration-inequalities", "hoeffding", "bernstein", "sample-complexity", "multiple-testing", "quantitative-research", "backtesting", "statistics", "sharpe-ratio", "math-for-quants"]
category: "trading"
subcategory: "Quantitative Finance"
author: "Hiep Tran"
featured: false
readTime: 22
---

> [!important]
> **TL;DR:** The central limit theorem describes a strategy you will never run, the one with infinite trades. Concentration inequalities bound what can go wrong after a *finite* number, so "do I believe this edge?" becomes "have I seen enough observations?"
>
> - Markov, Chebyshev, Hoeffding and Bernstein form a ladder. Each rung demands one more assumption and pays you back with a tighter bound.
> - For a strategy whose per-trade result lives in the range [-1R, +2R], pinning the true edge to within 0.1R at 95% confidence needs **1,660 trades** by Hoeffding and only **293** by Bernstein, because Bernstein knows the variance and Hoeffding only knows the range.
> - Search 500 candidate signals and the significance bar moves from a t-statistic of 1.96 to about **3.89**. The scale of that move is $\sqrt{2\ln m}$, which for 500 is 3.53.
> - A Sharpe of 1.5 measured over 200 daily observations is a t-statistic of about **1.34**. That is not evidence. It is a rounding error with a good story attached.

## The number that should scare you

Here is a real conversation, compressed. A researcher shows a backtest: annualised Sharpe ratio of 1.5 over 200 trades. The *Sharpe ratio* is just return divided by the volatility of that return, an edge-per-unit-of-risk score, and 1.5 is genuinely good. The room asks one question: do you believe it?

The honest answer, which almost nobody gives, is a number. A Sharpe of 1.5 sustained over 200 daily observations, which is roughly 0.79 of a year, corresponds to a t-statistic of about 1.34. The conventional bar for "probably not luck" is 1.96. So the backtest is not merely weak evidence. It fails the *loosest* test anyone would apply, before you even ask how many other strategies were tried before this one got shown.

That gap, between a number that looks impressive and a number that is believable, is what this post is about. The tool that closes it is a family of results called **concentration inequalities**: statements that bound how far a sample average can stray from the truth after a specific, finite number of observations.

![Four rows showing Markov, Chebyshev, Hoeffding and Bernstein, each with the assumption it requires, the bound it delivers, and how fast its tail decays](/imgs/blogs/concentration-inequalities-sample-complexity-math-for-quants-1.webp)

The figure above is the mental model for everything that follows. Read it top to bottom as a ladder. Each rung asks you to assume something more about the world, and pays you back with a sharper guarantee. Markov asks almost nothing and tells you almost nothing. Bernstein asks you to know both the range and the variance, and repays that with a bound that can cut your data requirement by a factor of five.

## Foundations: the four things you need before any of this makes sense

You do not need a statistics background to follow this. You need four ideas.

### 1. The true edge versus the measured edge

Every strategy has a **true expected return per trade**, written $\mu$. You never observe it. What you observe is the **sample mean**, written $\bar{X}_n$: you take your n trades, add the results, divide by n. The whole discipline of quantitative research is the gap between those two symbols.

### 2. R, the unit traders actually think in

Rather than dollars, traders normalise by risk. **One R is the amount you put at risk on a single trade.** Say you size every position so that being stopped out costs you \$1,000. That \$1,000 is your R. A trade that makes \$500 returned +0.5R; one that makes \$2,000 returned +2.0R; being stopped out is -1.0R.

Working in R rather than dollars does two things. It makes trades comparable across position sizes, so a \$500 win on a \$1,000 risk and a \$5,000 win on a \$10,000 risk are the same trade. And it makes the *bounded* assumption below natural: if you always use a stop loss at 1R and always take profit at 2R, then every trade result lives in the interval [-1R, +2R], no exceptions. Throughout this post I will quote results in R and, where it helps, translate back at \$1,000 of risk per trade.

### 3. A tail bound, and what delta means

Every result in this post has the same shape:

$$
P\left(\left|\bar{X}_n - \mu\right| \ge \epsilon\right) \le \delta
$$

In words: the probability that your measured edge is off by more than $\epsilon$ is at most $\delta$. Here $\epsilon$ is your **tolerance**, how much error you are willing to live with, and $\delta$ is your **failure probability**, how often you accept being wrong by more than that. Setting $\delta$ to 0.05 is the familiar 95% confidence.

**Sample complexity** is what you get when you flip this around. Fix $\epsilon$, fix $\delta$, and solve for n. The answer is the number of trades you need. That is the number a researcher should carry in their head, and almost none do.

### 4. Why the central limit theorem is not enough

The [law of large numbers and the central limit theorem](/blog/trading/math-for-quants/law-large-numbers-central-limit-theorem-math-for-quants) say your sample mean converges to the truth and that its error becomes normally distributed. Both are *asymptotic*: they describe the limit as n goes to infinity. They are silent on n = 200.

That silence matters more than it sounds. Trade returns are skewed and fat-tailed exactly where the normal approximation is worst, in the tail you are trying to bound. The CLT gives you an approximation whose error you cannot quantify. A concentration inequality gives you an inequality that is true at every n, with no approximation at all. It is weaker in the sense that it is conservative, and stronger in the sense that it is a guarantee rather than a hope.

## The ladder, one rung at a time

### Markov: the bound that assumes nothing

If a random quantity X is never negative and has mean $\mathbb{E}[X]$, then for any threshold a:

$$
P(X \ge a) \le \frac{\mathbb{E}[X]}{a}
$$

The intuition is a budgeting argument. The average is a fixed budget. If a large value occurred too often, the average would have to be bigger than it is. That is the whole proof, and it needs no distribution, no variance, no independence.

#### Worked example: a drawdown bound from a single number

Suppose a fund's average monthly drawdown, meaning peak-to-trough loss within the month, is 5%. Drawdowns cannot be negative, so Markov applies directly. What is the chance of a 25% drawdown in a given month?

At most 5 / 25 = 20%.

That is almost certainly a terrible bound. The real probability is probably well under 1%. But notice what it cost: one number, no assumption about the shape of anything. **Markov is the floor of the ladder because it is the price of assuming nothing, and the price is a bound so loose you would not act on it.**

### Chebyshev: pay one variance, get one order of magnitude

Add the assumption that the variance $\sigma^2$ is finite, and apply Markov to the squared deviation. You get Chebyshev:

$$
P\left(|X - \mu| \ge k\sigma\right) \le \frac{1}{k^2}
$$

Being three standard deviations from the mean has probability at most 1/9, whatever the distribution. Applied to a sample mean, whose variance is $\sigma^2/n$, this becomes:

$$
P\left(\left|\bar{X}_n - \mu\right| \ge \epsilon\right) \le \frac{\sigma^2}{n\epsilon^2}
$$

Now we can compute a sample complexity. Solve for n:

$$
n \ge \frac{\sigma^2}{\delta\epsilon^2}
$$

### Hoeffding: bounded draws buy you exponential decay

Chebyshev's bound decays like 1/n. That is slow: to get ten times more confident you need ten times more data. If you additionally know each observation is confined to a fixed interval [a, b], Hoeffding's inequality (Hoeffding, 1963) upgrades that to exponential decay:

$$
P\left(\left|\bar{X}_n - \mu\right| \ge \epsilon\right) \le 2\exp\left(\frac{-2n\epsilon^2}{(b-a)^2}\right)
$$

Note the constant carefully, because it is the single most-misquoted formula in this area: the numerator carries a **2**, the denominator carries the **squared range**, and the leading **2** is there only because we asked for a two-sided bound. Drop it for a one-sided question.

What does "bounded" mean for a trading strategy? It means the P&L per trade genuinely cannot escape an interval. A stop loss at 1R and a profit cap at 2R gives you [-1R, +2R], so b - a = 3R. This assumption is not free, and it is where the inequality gets abused: a strategy that can gap through its stop is not bounded at 1R, and using Hoeffding on it is not conservative, it is wrong. The bound holds only if the interval genuinely holds.

## Worked examples: how many trades, really

Now let us do the calculation the interview question is really asking.

**The setup, used for all three examples.** A strategy with per-trade returns confined to [-1R, +2R]. You want to pin the true edge $\mu$ to within a tolerance of 0.1R, with a failure probability of 5%. So b - a = 3R, tolerance 0.1R, delta 0.05.

We will also need a concrete return distribution for the variance-aware bounds. Suppose the strategy behaves like this, which is a very common shape: mostly small losses, regular modest wins, occasional big ones.

| Outcome | Probability | Result |
| --- | --- | --- |
| Stopped out early | 60% | -0.3R |
| Target hit | 35% | +0.5R |
| Trend runs to the cap | 5% | +2.0R |

Its mean is 0.6(-0.3) + 0.35(0.5) + 0.05(2.0) = **+0.095R**, so this is a genuinely profitable strategy. Its second moment is 0.6(0.09) + 0.35(0.25) + 0.05(4.0) = 0.3415, so the variance is 0.3415 - 0.095² = **0.3325 R²** and the standard deviation is about 0.58R. The largest deviation any single trade can have from the mean is 2.0 - 0.095 = **1.905R**.

#### Worked example 1: Hoeffding says 1,660 trades

Set the right-hand side of Hoeffding equal to 0.05 and solve.

1. The exponent is -2n(0.1)²/(3)² = -2n(0.01)/9 = -n/450.
2. So the bound reads 2exp(-n/450) = 0.05, giving exp(-n/450) = 0.025.
3. Take logs: n/450 = ln(40) = 3.6889.
4. Therefore n = 450 × 3.6889 = **1,660 trades**.

Put a book behind those units so the answer stops being abstract. Run this on a desk with \$20m of capital risking 1R = \$100k per trade. Then "within 0.1R" means "within \$10k of P&L per trade", and 1,660 trades is what it costs to earn that precision. At one trade a day that is about six and a half years of live trading before Hoeffding will certify the edge. Over those 1,660 trades a true edge of 0.2R is \$332k of expected profit, and the bound is telling you that until you have paid for the whole sample, \$166k of that could be luck. *Hoeffding is honest and expensive: it charges you for the worst case your interval allows, whether or not you ever trade it.*

#### Worked example 2: Chebyshev says 665, and beats Hoeffding

Now use the variance instead of the range.

1. n = σ² / (δ ε²) = 0.3325 / (0.05 × 0.01).
2. The denominator is 0.0005.
3. So n = **665 trades**.

This should be surprising. Chebyshev, the weaker-looking bound with polynomial decay, needs *two and a half times less data* than Hoeffding here. There is no contradiction. Hoeffding throws the variance away entirely and prices the worst distribution consistent with [-1R, +2R], which is a coin flip between the two endpoints. Our strategy is nothing like that: most of its mass sits in a narrow band near zero. Chebyshev knows this and Hoeffding does not.

The reason Hoeffding still matters is the *rate*. Chebyshev's requirement scales like 1/δ, Hoeffding's like ln(1/δ). Push the confidence up and Hoeffding overtakes.

![Line chart of trades required against failure probability, with Chebyshev, Hoeffding and Bernstein curves crossing near 1.5%](/imgs/blogs/concentration-inequalities-sample-complexity-math-for-quants-2.webp)

| Failure probability δ | Chebyshev | Hoeffding | Bernstein |
| --- | --- | --- | --- |
| 5% | 665 | 1,660 | 293 |
| 1% | 3,325 | 2,384 | 420 |
| 0.1% | 33,248 | 3,420 | 602 |

The crossover sits near δ ≈ 1.5%. Above it Chebyshev is tighter; below it Hoeffding's exponential decay takes over and never gives the lead back. *Which bound is "better" is not a property of the bounds. It is a property of how certain you are trying to be.*

### Bernstein: use the range and the variance together

The obvious question is why you should have to choose. Bernstein's inequality does not make you. It keeps Hoeffding's exponential decay but drives it with the variance, using the range only as a correction on the second-order term:

$$
P\left(\left|\bar{X}_n - \mu\right| \ge \epsilon\right) \le 2\exp\left(\frac{-n\epsilon^2}{2\sigma^2 + \tfrac{2}{3}c\epsilon}\right)
$$

Here c is the largest deviation any single observation can have from the mean. Different textbooks carry slightly different constants on that second denominator term; the form above is the classical Bernstein statement, and it is the one to quote.

Read the denominator as a competition between two regimes. When $\epsilon$ is small, the ${2\sigma^2}$ term dominates and you get an essentially Gaussian bound driven by variance alone. When $\epsilon$ is large, the range term takes over and the decay degrades to something exponential rather than sub-Gaussian. This is exactly right: small deviations are governed by the bulk of the distribution, large ones by how far a single observation can reach.

#### Worked example 3: Bernstein says 293 trades

Same strategy, same tolerance of 0.1R, same 5% failure probability. Now σ² = 0.3325 and c = 1.905.

1. Denominator: 2(0.3325) + (2/3)(1.905)(0.1) = 0.6650 + 0.1270 = 0.7920.
2. The bound reads 2exp(-n(0.01)/0.7920) = 0.05.
3. So n(0.01)/0.7920 = ln(40) = 3.6889.
4. Therefore n = 0.7920 × 3.6889 / 0.01 = 292.1, round up to **293 trades**.

From 1,660 to 293. **The same data, the same confidence, the same tolerance, and a 5.7-fold reduction in what you need, purely from refusing to throw away the variance.** In calendar terms that is the difference between six and a half years and roughly fourteen months.

On the same \$20m desk at 1R = \$100k, that reduction is the whole argument for using the sharper bound. Waiting for Hoeffding means leaving the strategy at pilot size for five extra years. If a validated edge of 0.2R on one trade a day is worth roughly \$5m a year once it is sized properly, then choosing the bound that respects the variance is worth on the order of \$25m of foregone profit that you no longer forego. The mathematics did not change; only the willingness to use what you already measured.

One caveat that keeps this honest. Our strategy's true edge is +0.095R and we just bounded the error at 0.1R, so the interval still contains zero. To actually rule out "no edge" you want the tolerance meaningfully smaller than the edge. Tighten it to 0.05R and Hoeffding jumps to 6,640 trades, exactly four times as many, because it scales as 1/ε². Bernstein goes to 1,075, only 3.7 times as many, because shrinking ε also shrinks its linear range term. *The sharper bound degrades more gracefully, which is a second reason to prefer it.*

## The union bound: the tax on searching

Everything above assumes you decided what to test *before* you looked. Nobody works that way. A researcher screens hundreds of candidate signals and shows you the winner. The maximum of many noisy numbers is not a noisy number. It is a systematically large one.

The union bound is the crude, correct tool. For any events, the probability that at least one occurs is at most the sum of their probabilities. Apply it to m independent t-statistics that are pure noise, each standard normal, and use the Gaussian tail bound. The largest of them satisfies:

$$
\mathbb{E}\left[\max_{1 \le j \le m} Z_j\right] \le \sqrt{2\ln m}
$$

That square-root-of-a-log is the number to memorise. It grows agonisingly slowly, which cuts both ways: doubling your search barely moves the bar, but the bar is already far above 1.96 by the time m reaches the hundreds.

![Table of m against sqrt(2 ln m), Bonferroni threshold and implied five-year Sharpe, with the 500 row highlighted](/imgs/blogs/concentration-inequalities-sample-complexity-math-for-quants-3.webp)

#### Worked example 4: searching 500 signals

You screen 500 candidate signals over five years of daily data. None of them has any edge whatsoever. What does the best of them look like?

1. **The scale of the maximum.** $\sqrt{2\ln 500} = \sqrt{12.43} = 3.53$. That is the leading term.
2. **What the best one actually scores.** The refined expectation for the maximum of m standard normals subtracts a correction term and gives about **2.91**. So on average, the winner of a 500-way search of pure noise carries a t-statistic near 2.91.
3. **Translate to Sharpe.** For a strategy measured over T years, the t-statistic of the mean return is approximately the annualised Sharpe times $\sqrt{T}$ (Lo, 2002). Over five years, $\sqrt{5} = 2.236$, so a t of 2.91 is an annualised Sharpe of 2.91 / 2.236 = **1.30**.
4. **The threshold you actually need.** To hold the family-wise error rate at 5% across 500 tests, Bonferroni asks each test to clear a two-sided p-value of 0.05 / 500 = 0.0001, which is a t-statistic of **3.89**, an annualised Sharpe of 1.74 over the same five years.

So: **searching 500 signals over five years of daily data, pure noise delivers a best-in-class backtest Sharpe of about 1.30, and you need roughly 1.74 to claim anything.** Those two numbers, 1.30 and 1.74, are the multiple-testing tax made concrete.

Price it. On the same \$20m book run at a 10% volatility target, a Sharpe of 1.30 implies 13% of expected annual return, which is \$2.6m a year. That is the number a researcher would put in a memo, and on these assumptions every dollar of it is an artefact of having looked 500 times. The gap up to the honest threshold is worth \$880k a year on its own: Sharpe 1.74 is \$3.48m, and anything between \$2.6m and \$3.48m of claimed profit is inside the noise floor of the search that produced it. A researcher who screened 500 signals and presents a Sharpe of 1.5 has presented something *below the noise floor of their own search*.

This is the same phenomenon that the deflated Sharpe ratio corrects for more carefully, using the actual number and correlation of trials rather than a worst-case union bound. That machinery is covered properly in [overfitting, purged cross-validation and the deflated Sharpe ratio](/blog/trading/quantitative-finance/overfitting-purged-cv-deflated-sharpe-quant-research), so I will not re-derive it here. The union bound is the version you can do on a whiteboard, and it is deliberately conservative: real signals are correlated, so the effective m is smaller than the raw count. That makes 3.89 an upper bound on the honest threshold, not the honest threshold itself. Notably, Harvey, Liu and Zhu (2016) reach a comparable conclusion from an entirely different direction, arguing that a newly claimed factor in the cross-section of returns should carry a t-statistic above about 3.0.

## McDiarmid: concentration for things that are not averages

Everything so far bounds a *mean*. But you rarely report a mean. You report a Sharpe ratio, a maximum drawdown, an information coefficient, all of which are complicated functions of the whole sample. McDiarmid's bounded differences inequality (McDiarmid, 1989) handles those.

The condition is intuitive. Suppose f is any function of your n observations, and changing observation i alone, leaving everything else fixed, can move f by at most $c_i$. Then f concentrates around its own expectation:

$$
P\left(\left|f(X_1,\dots,X_n) - \mathbb{E}f\right| \ge \epsilon\right) \le 2\exp\left(\frac{-2\epsilon^2}{\sum_{i=1}^{n} c_i^2}\right)
$$

Hoeffding is the special case where f is the average and every $c_i$ equals the range divided by n. But the general version is the one that matters on a desk, because it converts a question about backtest stability into a question you can measure: **how much does any single trade move my headline number?**

#### Worked example 5: how stable is a Sharpe of 1.50?

Take a backtest of 1,000 trades reporting a Sharpe of 1.50. Run a jackknife: drop each trade in turn, recompute the Sharpe, record the largest move. Two cases.

**Concentrated book.** The largest single-trade influence is c = 0.02 Sharpe points. With all $c_i$ equal, the sum of squares is n·c², so the half-width at 95% is:

$$
\epsilon = c\sqrt{\frac{n\ln(2/\delta)}{2}} = 0.02\sqrt{\frac{1000 \times 3.6889}{2}} = 0.02 \times 42.95 = 0.86
$$

The honest interval is 1.50 ± 0.86, which runs from **0.64 to 2.36**. That backtest does not distinguish a mediocre strategy from an excellent one.

**Size-capped book.** Now impose position limits so no single trade can move the Sharpe by more than c = 0.005. The same arithmetic gives 0.005 × 42.95 = **0.21**, an interval of 1.29 to 1.71.

![Two panels comparing a concentrated book with a wide Sharpe band against a size-capped book with a narrow one](/imgs/blogs/concentration-inequalities-sample-complexity-math-for-quants-4.webp)

Quartering the per-trade influence quartered the uncertainty band, on exactly the same 1,000 trades. *The cheapest way to make a backtest believable is often not more data. It is risk management that stops any one trade from dominating the number.* That is the generalisation result hiding behind every "our Sharpe is stable out of sample" claim, and it is why position limits are an epistemic tool as much as a risk one.

## The trade-offs

| The simple view | What actually happens |
| --- | --- |
| "The CLT means 30 observations is enough" | The CLT is a limit statement with no error term at finite n. Thirty fat-tailed, skewed trade returns are nowhere near normal in the tail you care about. |
| "A tighter bound is always better" | Tighter bounds need stronger assumptions. Hoeffding on a strategy that can gap through its stop is not conservative, it is invalid. |
| "Hoeffding beats Chebyshev" | Only below δ ≈ 1.5% for this strategy. Above that, the variance-aware bound wins. |
| "More data is the answer" | Sometimes. Capping per-trade influence tightened the Sharpe band 4x with zero extra data. |
| "My t-statistic is 2.5, so it is significant" | Only if you tested one thing. Best-of-500 pure noise averages 2.91. |
| "Bounds are too conservative to use" | The conservatism is the product. A bound that holds at every n is worth more than an approximation whose error you cannot quantify. |

## Common misconceptions

**"n = 30 is enough."** This is a garbled memory of a CLT rule of thumb for estimating a mean of a well-behaved distribution when you only care about the centre. It says nothing about tails, nothing about skewed or heavy-tailed returns, and nothing about a *guarantee*. Our very ordinary strategy needed 293 observations under the sharpest bound available and 1,660 under a bound that only knows the range. Neither number is close to 30.

**"A t-statistic of 2 means I am 95% confident."** It means that only if the hypothesis was fixed before you looked at the data. If the strategy was selected as the best of a search, the reference distribution is not the t-distribution, it is the distribution of the *maximum*. For 500 candidates that maximum averages 2.91 under pure noise. Your t of 2 is not weak evidence for the strategy; it is evidence against it, because noise alone would have done better.

**"Concentration inequalities are the same thing as confidence intervals."** A confidence interval built from the CLT is an approximation that becomes exact as n grows. A concentration bound is an inequality that is exactly true at every n and is therefore wider. Use the interval when you are estimating and the bound when you are deciding, especially when the cost of being wrong is asymmetric.

**"Bernstein is strictly better than Hoeffding, so just always use Bernstein."** Bernstein needs the variance, and you have to *estimate* it from the same data you are testing. That estimate has its own error, which a careful treatment propagates through, typically by using an empirical Bernstein bound. If you plug a noisy variance estimate into Bernstein and quote the result as a guarantee, you have quietly reintroduced the problem you were trying to solve. See [estimators, bias, variance and consistency](/blog/trading/math-for-quants/estimators-bias-variance-consistency-math-for-quants) for why that estimate is not free.

**"The union bound is too crude to bother with."** It is crude, and it is also the only correction you can perform in your head in an interview. Its conservatism has a known direction: correlated signals mean the effective number of independent tests is below the raw count, so the union bound gives an upper bound on the true threshold. Knowing the direction of an error is most of what you need to use it.

## Sources and further reading

- **Boucheron, Lugosi and Massart**, *Concentration Inequalities: A Nonasymptotic Theory of Independence* (Oxford University Press, 2013). The standard reference for everything on the ladder, including the entropy method behind McDiarmid.
- **Wainwright**, *High-Dimensional Statistics: A Non-Asymptotic Viewpoint* (Cambridge University Press, 2019), chapter 2. The cleanest modern treatment of sub-Gaussian and sub-exponential tails, and the source to check when textbook Bernstein constants disagree.
- **Hoeffding (1963)**, "Probability Inequalities for Sums of Bounded Random Variables", *Journal of the American Statistical Association* 58(301). The original.
- **McDiarmid (1989)**, "On the method of bounded differences", in *Surveys in Combinatorics*.
- **Lo (2002)**, "The Statistics of Sharpe Ratios", *Financial Analysts Journal* 58(4). Where the t-statistic to Sharpe conversion used above comes from, along with the corrections for autocorrelated returns.
- **Harvey, Liu and Zhu (2016)**, "... and the Cross-Section of Expected Returns", *Review of Financial Studies* 29(1). The multiple-testing argument for a higher significance bar in asset pricing.
- **Bailey and Lopez de Prado (2014)**, "The Deflated Sharpe Ratio", *Journal of Portfolio Management* 40(5). The refined version of this post's Worked example 4.

Adjacent posts here: [hypothesis testing and p-values](/blog/trading/math-for-quants/hypothesis-testing-pvalues-math-for-quants) for the machinery this builds on, and [overfitting, purged CV and the deflated Sharpe](/blog/trading/quantitative-finance/overfitting-purged-cv-deflated-sharpe-quant-research) for the production-grade treatment of selection bias.

## In the interview room and on the desk

**How the question is actually asked.** Nobody says "state Hoeffding's inequality." They say: *"Your strategy has a Sharpe of 1.5 over 200 trades. Do you believe it?"* Or the version that is really the same question: *"How many observations would you want before you allocated capital to this?"* Both are sample-complexity questions wearing a trader's clothes.

**What a strong answer contains, in order.** First, convert to a t-statistic rather than reacting to the Sharpe. Two hundred daily observations is 0.79 years, so t = 1.5 × √0.79 ≈ 1.34, which does not clear 1.96. Say that immediately: the number fails the loosest available test. Second, give the finite-sample version rather than waving at the CLT. If per-trade results live in [-1R, +2R] and you want the edge pinned to 0.1R at 95%, Hoeffding demands 1,660 trades. Third, sharpen it: with a measured variance of about 0.33R², Bernstein brings that to 293, and say why, because Hoeffding prices the worst distribution the range allows and this one is nothing like it. Fourth, and this is what separates candidates: ask where the strategy came from. If it is the survivor of a 500-signal screen, the bar is not 1.96 but roughly 3.89, and pure noise would have produced a best-of-500 Sharpe near 1.30 over five years. At that point 1.5 is not an edge, it is the expected output of the search process.

**The follow-up and the trap.** The push is always "so how much data do you actually need?" Answer it with the calendar, not the formula: at t = 1.5 you need about 430 trading days to clear 1.96 as a single pre-registered test, and about 1,695 days, close to seven years, to clear the 500-signal Bonferroni bar. That is the real reason firms invest in higher-frequency signals and in cross-sectional breadth: both buy observations faster. The trap is quoting a t-statistic of 2 for a strategy that was chosen as the best of hundreds and calling it 95% confidence. Have the antidote memorised: the maximum of m noise draws scales like $\sqrt{2\ln m}$, which is 3.53 for 500 and 3.72 for 1,000. Saying that number out loud, unprompted, is the whole signal.

**Who weights this most.** Jane Street and Jump probe fast probabilistic reasoning, so they want the bound estimated in your head in seconds; Two Sigma and WorldQuant probe multiple-testing discipline, so they want to hear the search size before they hear the Sharpe.
