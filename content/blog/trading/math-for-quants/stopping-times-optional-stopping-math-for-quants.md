---
title: "Stopping times and optional stopping: why you cannot beat a fair game"
date: "2026-09-21"
publishDate: "2026-09-21"
description: "Quit while you are ahead feels like it should work, and provably does not. This is the measure-theoretic half of that story: what a stopping time actually is, the three conditions under which a stopped martingale keeps its expectation, and why every betting system that looks like free money is buying it with unbounded time or unbounded capital."
tags: ["stopping-time", "optional-stopping-theorem", "martingale", "gamblers-ruin", "doubling-system", "stop-loss", "filtration", "quant-interview", "probability", "math-for-quants"]
category: "trading"
subcategory: "Quantitative Finance"
author: "Hiep Tran"
featured: false
readTime: 19
---

> [!important]
> **TL;DR:** A stopping time is an exit rule whose trigger you can evaluate without seeing the future, and the optional stopping theorem says a fair game stopped at one is still fair. The theorem's three sufficient conditions are the whole content, because every strategy that appears to beat a fair game breaks exactly one of them.
>
> - The engine is not really about stopping. The stopped process is always a martingale, so ${E[X_{\tau \wedge n}] = X_0}$ for every finite ${n}$, unconditionally. The three conditions exist only to license taking ${n \to \infty}$ inside the expectation.
> - Bounded stopping time, bounded martingale, or bounded increments with finite expected stopping time. They rule out, in order: unbounded waiting, unbounded capital, and unbounded bet size.
> - Gambler's ruin with \$250,000 of capital and \$25,000 a trade, chasing \$400,000: a 62.5% chance of the target and a 37.5% chance of zero, which is \$250,000 in expectation, exactly the starting stack.
> - The doubling ladder on a \$255,000 bankroll wins \$1,000 in 255 months out of 256 and loses the whole \$255,000 in the other one. Expectation zero, to the cent. Its expected number of bets is 2, so the finite-expected-time condition holds and the candidate who stops there is wrong: it is the unbounded bet size that breaks it.
> - A stop-loss is a genuine stopping time and it changes your expectation by exactly nothing. On a \$50m book it cut the standard deviation from \$5.00m to \$4.62m, which is precisely the 3.68 days it took the position out of the market.

## The betting system that survives every backtest

Someone will eventually show you a strategy that has never had a losing month. It is not fraud and the backtest is not lying. Risk a small amount; if it loses, risk twice as much, and keep doubling until a win recovers everything plus the original stake. Every sequence terminates in a win, every terminated sequence is profitable, and the equity curve is a staircase going up and to the right.

The strategy is a martingale in the gambler's sense, it is playing a game with zero edge, and the reason it appears to manufacture money out of nothing is not subtle once you see it. What is subtle is saying precisely *which* assumption it is breaking, because "it will blow up eventually" is a feeling rather than an argument, and a good interviewer will keep pushing until you name the hypothesis.

That naming is what the optional stopping theorem is for. It says a fair game stopped at a legal moment is still a fair game, and its value to a trader is almost entirely in the fine print. Read the hypotheses as a catalogue of the ways a strategy can pretend to have an edge, and you can point at the exact line in almost any system where the money is supposedly coming from.

A companion post, [optimal stopping and the secretary problem](/blog/trading/math-for-quants/optimal-stopping-secretary-when-to-take-the-trade-math-for-quants), works the decision problem: given that you may stop, when should you? This post is the other half. It asks what stopping is allowed to mean, and what stopping can and cannot do to an expectation.

## Foundations: what you know, and when you may act

Three objects, built from zero.

**A filtration is what you know by time ${t}$.** Write $\mathcal{F}_t$ for the set of questions you can answer at time $t$. At the close on day 12 of a month, $\mathcal{F}_{12}$ contains "did the stock trade below \$45 in the first twelve days?" because you watched it, and "what is my current P&L?" because you can read it off the book. It does not contain "is day 12 the month's high?" because that depends on days 13 through 25. The collection $\mathcal{F}_0 \subseteq \mathcal{F}_1 \subseteq \dots$ is growing, because information accumulates and is never unlearned. That growing family is the filtration. The full treatment, including what goes wrong in a backtest when a rule quietly consults the wrong $\mathcal{F}_t$, is in [filtrations, adapted processes, and no look-ahead](/blog/trading/math-for-quants/filtrations-no-lookahead-math-for-quants).

**A martingale is a process with no expected drift given what you know.** Formally $E[X_{n+1} \mid \mathcal{F}_n] = X_n$: your best forecast of tomorrow's value, given everything you have seen, is today's value. A running total of fair coin flips is the canonical example, since the next flip adds ${+\$1}$ or ${-\$1}$ with equal probability and contributes nothing in expectation. The conditional expectation doing the work is a projection onto your information, developed in [conditional expectation as projection](/blog/trading/math-for-quants/conditional-expectation-projection-math-for-quants), and the reason discounted asset prices are martingales under a particular measure is in [martingales and the risk-neutral measure](/blog/trading/math-for-quants/martingales-risk-neutral-measure-math-for-quants). Here the two-line version suffices: a martingale is a fair game, and "fair" is a statement about conditional expectations, not about symmetry of the payouts.

**A stopping time is a rule whose trigger you can evaluate without seeing the future.** A random time $\tau$ is a stopping time if, for every $t$, the event $\{\tau \le t\}$ belongs to $\mathcal{F}_t$. In plain language: at every moment, you can say whether you have already stopped, using only what you have seen. You are allowed to not know when you will stop. You are not allowed to need tomorrow's data to know whether you stopped today.

![Annotated price path with a vertical line at day 12, the region to the right greyed out as not yet observed, and two rule cards contrasting a valid 45 dollar trigger with an invalid sell at the month's high](/imgs/blogs/stopping-times-optional-stopping-math-for-quants-1.webp)

That figure is the whole definition in one picture. The rule "$\tau$ = first day the price closes at or below \$45" settles from the observed half alone: on each of days 1 through 12 you compare the close to the trigger level of \$45, and you know. The rule "$\tau$ = the day the month prints its high" cannot be settled without the grey band, because declaring day 19 the high requires knowing that days 20 through 25 print lower. It is describable after the fact and executable never.

The useful corollary is that most real exit rules pass. A stop-loss, a take-profit, a time-based exit, a trailing stop that ratchets on the highest price *so far*, a margin call, a barrier-option knock-in: all stopping times, all decidable at the instant they fire. So the theorem is not an exotic constraint that real strategies dodge. It applies to nearly everything a desk does.

## The theorem, and why it is really a statement about limits

Here is Doob's optional stopping theorem in the form you should be able to state from memory.

Let $(X_n)$ be a martingale with respect to $(\mathcal{F}_n)$, and let $\tau$ be a stopping time. Then

$$
E[X_\tau] = E[X_0]
$$

provided **any one** of the following holds:

- **(a) $\tau$ is bounded**: there is an $N$ with $\tau \le N$ almost surely.
- **(b) $X$ is bounded**: there is a $K$ with $|X_n| \le K$ for all $n$, and $\tau \lt \infty$ almost surely.
- **(c) $E[\tau] \lt \infty$ and the increments are bounded**: there is a $K$ with $|X_n - X_{n-1}| \le K$ for all $n$.

Now the part that most treatments bury, and which makes the three conditions stop feeling arbitrary. The stopping is not the hard part. For **any** stopping time at all, with no conditions whatsoever, the *stopped process* $X^\tau_n := X_{\tau \wedge n}$ is itself a martingale. The reason is mechanical: stopping is the trading strategy "hold one unit until $\tau$, then hold zero", the indicator $\mathbf{1}_{\{\tau \ge n\}}$ is known at time $n-1$ because $\{\tau \le n-1\}$ is in $\mathcal{F}_{n-1}$, and a martingale bet on with a previsible position is still a martingale. So

$$
E[X_{\tau \wedge n}] = E[X_0] \quad \text{for every finite } n
$$

holds unconditionally, always, for every stopping time in existence.

What you actually want is $E[X_\tau]$, which is the $n \to \infty$ limit of the left-hand side. The entire content of the three conditions is that each one licenses moving that limit inside the expectation. Under (a) there is no limit to take, because $\tau \wedge N = \tau$ already. Under (b) the stopped values sit inside a fixed band, so bounded convergence applies. Under (c) the displacement is dominated by $K\tau$, which is integrable precisely because $E[\tau]$ is finite, so dominated convergence applies.

That is the sentence worth carrying out of this post: **optional stopping is not a theorem about stopping, it is a theorem about interchanging a limit and an expectation.** Every betting system that appears to beat a fair game is a failed interchange wearing a costume. The gambler is computing something true at every finite horizon and then asserting it at infinity, where it is false.

![Three by three matrix listing each sufficient condition of the optional stopping theorem, what it requires, and the betting system it rules out](/imgs/blogs/stopping-times-optional-stopping-math-for-quants-2.webp)

Read that figure as a catalogue of failure modes rather than a list of hypotheses. Condition (a) requires that you are out by a fixed deadline no matter what, which kills "hold until it comes back", because the wait is unbounded. Condition (b) requires that your equity path never leaves a fixed band, which kills any rule that survives its drawdown only with unlimited capital. Condition (c) requires a capped bet size alongside a finite expected exit, which kills the doubling ladder: its expected number of bets is 2, comfortably finite, but the bets themselves double without limit.

A strategy needs only **one** condition to hold for the conclusion to bite, so to escape the theorem you have to break all three at once. Since any real trader has both a finite bankroll and a finite career, conditions (a) and (b) are imposed on you whether you consent or not.

## Worked example 1: gambler's ruin, with the expectation check

A discretionary trader has \$250,000 of risk capital and takes positions that risk \$25,000 each, so ten units. Assume the trades are genuinely edgeless: each one gains or loses \$25,000 with probability one half, independently. The trader will keep going until the account either reaches \$400,000, which is sixteen units, or hits zero. These figures are illustrative arithmetic on assumed inputs, not measured from any real book.

Equity $X_n$ is a martingale, and $\tau$ is the first time it touches 0 or 400,000. Between the two walls the process is confined to $[0, 400{,}000]$, and it reaches a wall with probability one, so condition (b) applies. Optional stopping gives $E[X_\tau] = X_0 = 250{,}000$. Write $p$ for the probability the target comes first, so $X_\tau$ is \$400,000 with probability $p$ and \$0 otherwise:

$$
400{,}000\,p + 0 \cdot (1-p) = 250{,}000 \quad \Longrightarrow \quad p = \frac{250{,}000}{400{,}000} = \frac{10}{16} = 0.625
$$

![Gambler's ruin lattice showing an equity staircase between an absorbing ruin line at zero and an absorbing target line at 400k dollars, starting at 250k dollars](/imgs/blogs/stopping-times-optional-stopping-math-for-quants-3.webp)

So a 62.5% chance of reaching \$400,000 and a 37.5% chance of ruin. The expectation check is the point of the exercise: ${0.625 \times \$400{,}000 = \$250{,}000}$, and the ruin branch contributes nothing, so the expected terminal equity is \$250,000, exactly the starting stack. A 62.5% win rate, a 60% gain when it works, total loss when it does not, and not one cent of expected value created.

The theorem earns its keep a second time if you ask how long this takes. For a symmetric walk, $M_n = X_n^2 - n$ measured in units is also a martingale, and applying optional stopping to it gives $E[X_\tau^2] = E[\tau]$. Since $X_\tau$ is 16 units with probability ${10/16}$ and 0 otherwise, ${E[X_\tau^2] = 16^2 \times \frac{10}{16} = 160}$, and subtracting the starting ${10^2}$ leaves

$$
E[\tau] = 10 \times (16 - 10) = 60 \text{ trades}
$$

Sixty trades, \$1.5m of notional risked, to arrive at an expected outcome identical to doing nothing.

Now see what a real edge does to those numbers. If each trade wins with probability 0.51 instead of 0.50, an expected value of \$500 a trade, the process is no longer a martingale and optional stopping no longer forces the answer. The standard biased-walk formula with $q/p = {49/51}$ gives a probability of reaching the target of 69.7% rather than 62.5%. One percentage point of edge per trade moved the outcome by more than seven points, and no stopping rule anywhere in this post did anything comparable. **Edge comes from the game, never from the exit.**

## Worked example 2: the doubling ladder, priced to the cent

Now the strategy from the opening. A \$255,000 bankroll, a base bet of \$1,000, and the rule: on a loss, double; on a win, stop and start over. The bets go \$1,000, \$2,000, \$4,000, \$8,000, \$16,000, \$32,000, \$64,000, \$128,000, summing to exactly \$255,000, so the bankroll funds precisely eight attempts.

![Ladder of eight doubling bets from 1k to 128k dollars with a bracket for the 255,000 dollar bankroll and two terminal outcome boxes](/imgs/blogs/stopping-times-optional-stopping-math-for-quants-4.webp)

Any single win, at any rung, nets ${+\$1{,}000}$, because the winning bet returns double the losses beneath it plus the base stake. Eight consecutive losses exhaust the bankroll. On fair coin flips the probabilities are

$$
P(\text{win}) = 1 - \tfrac{1}{2^8} = \frac{255}{256} = 99.61\%, \qquad P(\text{ruin}) = \frac{1}{256} = 0.39\%
$$

and the expectation is

$$
\frac{255}{256}\times \$1{,}000 \; - \; \frac{1}{256}\times \$255{,}000 \;=\; \frac{255{,}000}{256} - \frac{255{,}000}{256} \;=\; \$0
$$

Both terms are the same exact fraction, ${255{,}000/256}$, which is \$996.09 to the cent. They cancel identically, not approximately.

Now name the violated condition, which is the part that separates a good answer from a vague one. In the **truncated** version just computed, the stopping time is bounded by 8, so condition (a) holds, the theorem applies, and zero is the guaranteed answer. The system is not beaten by bad luck; it is arithmetically incapable of having an edge.

In the **idealised** version, where you keep doubling forever until a win, the arithmetic changes: you win \$1,000 with probability 1, and $E[X_\tau] = \$1{,}000 \neq 0$. The theorem is not violated, its hypotheses simply fail. Which one? Not (a), since $\tau$ is unbounded. Not (b), since the equity path is unbounded below. And here is the trap: $\tau$ is geometric with mean 2, so $E[\tau] = 2$ is finite, and a candidate who has half-remembered condition (c) will announce that it applies. It does not, because (c) is a conjunction and the increments are not bounded: the $n$-th bet is $\$1{,}000 \times 2^{n-1}$, which exceeds any $K$ you propose. The doubling system is the textbook example of finite expected time paired with unbounded increments, and that is the entire reason it looks like free money on paper.

The dollar version of "unbounded increments" is that the expected worst drawdown before the winning flip is infinite: the loss immediately before a win on rung $k$ is $\$1{,}000 \times (2^{k-1} - 1)$, arriving with probability $2^{-k}$, and that series diverges. Any finite bankroll truncates it, and the truncation is exactly where the expectation snaps back to zero.

## Worked example 3: a stop-loss changes the variance and not the mean

The trading translation. A desk runs a \$50m position whose daily P&L is ${+\$1m}$ or ${-\$1m}$ with equal probability, held for a 25-day month. Again, illustrative arithmetic on assumed inputs. Two rules:

- **Hold to day 25.** Month-end P&L is $X_{25}$, with mean \$0 and variance 25 in units of ${\$m^2}$, so a standard deviation of \$5.00m.
- **Stop out at ${-\$5m}$.** Exit the moment cumulative P&L touches ${-\$5m}$, otherwise hold to day 25. Call that exit time $\sigma = \min(\tau_{-5}, 25)$, which is a bounded stopping time, so condition (a) applies and $E[X_\sigma] = 0$.

![Two column comparison of hold to day 25 against stopping out at minus five million, showing identical expected P&L and different standard deviations](/imgs/blogs/stopping-times-optional-stopping-math-for-quants-5.webp)

The expected month-end P&L is \$0.00m under both rules. Not approximately, not on average over enough months. Identically, by theorem. The stop truncates the left tail and pays for that truncation with foregone recoveries, and the two amounts are equal because the theorem says they are. The mechanism is visible directly: once the stop fires at ${-\$5m}$, the remaining days are still a fair walk, so the expected month-end P&L of the paths that stopped out is also exactly ${-\$5m}$.

What does change is everything else. Enumerating all ${2^{25}}$ paths exactly:

- The stop fires in **32.7%** of months.
- $E[\sigma] = 21.32$ days rather than 25.
- $\operatorname{Var}(X_\sigma) = 21.32$ as well, because $X_n^2 - n$ is a martingale and optional stopping on the bounded $\sigma$ gives $E[X_\sigma^2] = E[\sigma]$. Standard deviation \$4.617m against \$5.000m, a reduction of 7.7%, which the figure rounds to \$4.62m.
- Worst possible month goes from ${-\$25m}$ to a hard floor of ${-\$5m}$. Holding, a month ends ${\$5m}$ or worse down 21.2% of the time and ${\$9m}$ or worse down 5.4% of the time. Stopping, neither is possible.

The identity in the last two points is the elegant part. Variance fell from 25.00 to 21.32, a drop of 3.68, and the expected days out of the market is ${25 - 21.32 = 3.68}$. **A stop-loss buys you exactly the variance of the days it takes you out of the market, and nothing else.** That is a real and valuable thing to buy, since a hard floor on the worst month is what keeps a fund alive to trade the next one. It is simply not an edge, and a risk report that presents it as one is mislabelled.

## Common misconceptions

**"Optional stopping means you can never profit from timing."** It means you cannot profit from timing a *martingale*. Real assets have drift, and a process with positive drift is a submartingale, where stopping rules absolutely do move the expectation: that is what a trend-following exit is monetising. More importantly, the theorem is stated relative to a filtration. If you genuinely know something the market does not, the price is not a martingale with respect to *your* $\mathcal{F}_t$, and timing on it is worth money. The theorem says timing creates nothing where there was nothing. It says nothing against timing where there is something.

**"A stopping time is just any exit rule."** It is any exit rule that is decidable when it fires, relative to the right filtration. The failures are rarely as blatant as "sell at the high". They look like a trigger built on a centred moving average, a signal computed with a lag the live system will not have, a volatility estimate fitted on the full sample, or a universe filtered by which names survived. Each is a rule that reads $\mathcal{F}_{t+k}$ while claiming to read $\mathcal{F}_t$. A backtest built on one is not optimistic, it is describing a strategy nobody could have run.

**"Martingale betting works if you have enough capital."** More capital moves you along a curve of constant expectation, never off it. Take the same \$1,000 base bet but a bankroll of \$1,048,575,000, which funds twenty doublings. Ruin probability falls to 1 in 1,048,576, and a system that wins \$1,000 about 999,999 times out of a million looks close to riskless. The loss on the remaining path is \$1,048,575,000, and the expectation is still exactly zero. Doubling the bankroll halves the ruin probability and doubles the loss, in perfect step, forever. There is no bankroll large enough, because the quantity you are trying to change is not a function of the bankroll.

## How it shows up on the desk

Risk limits are the institutional enforcement of conditions (a) and (b). A daily loss limit, a position cap, a drawdown trigger that flattens the book: each converts an unbounded process into a bounded one, so the theorem's conclusion is guaranteed rather than hoped for. A firm that imposes them is not claiming its traders have no edge. It is refusing to let anyone's P&L rest on a hypothesis that finite capital cannot support.

The most common live symptom is a backtest that has never taken a large loss. That is usually not skill. It is a strategy that averages down, or holds through drawdowns without a stop, or sizes up after losses, all of which are the doubling ladder in different clothing. Each converts a distribution with occasional moderate losses into one with frequent small gains and a rare catastrophic loss, and a few years of history may simply not contain the rare branch. Compute the loss on the branch you have not seen and compare it to the capital. If the two are comparable, the backtest is measuring luck about a sample, not an edge. Funding is the same story from the other side: a position you must exit on a margin call has a stopping time set by your lender, not by you.

## Sources and further reading

- J. L. Doob, *Stochastic Processes*, Wiley, 1953. The original systematic treatment of martingales and optional sampling.
- David Williams, *Probability with Martingales*, Cambridge University Press, 1991. Chapter 10 states the stopped-process result and the three-condition optional stopping theorem in the form used here, and Chapter 10.12 works gambler's ruin and the ${E[\tau] = a(N-a)}$ identity.
- Steven Shreve, *Stochastic Calculus for Finance II: Continuous-Time Models*, Springer, 2004. Chapters 3 and 8 carry stopping times and optional sampling into continuous time and into American option pricing.
- Geoffrey Grimmett and David Stirzaker, *Probability and Random Processes*, 3rd edition, Oxford University Press, 2001. Chapter 12 for gambler's ruin in both the fair and biased cases.
- Continuous-time versions of everything here need the machinery in [Brownian motion from the random walk](/blog/trading/math-for-quants/brownian-motion-random-walk-math-for-quants).

All dollar figures in the worked examples are illustrative arithmetic on assumed inputs, computed exactly rather than drawn from any real book.

## In the interview room and on the desk

This arrives as a puzzle far more often than as a definition. "Here is a betting system. It has won 99% of the time for three years. What is wrong with it?" Jane Street and SIG weight this most heavily, and both will keep pushing after your first answer, because the first answer is almost always the intuition rather than the argument.

The strong answer comes in four steps, in this order. First, identify the underlying process as a martingale and say with respect to which filtration. Second, confirm that the exit rule *is* a legitimate stopping time, because it usually is, and saying so out loud shows you know that the flaw is not there. Third, name which of the three sufficient conditions fails, and say why: unbounded stopping time, unbounded martingale, or bounded expected time with unbounded increments. Fourth, and only fourth, produce the ruin arithmetic as confirmation that the expectation really is zero.

The trap is doing step four first and skipping step three. Arguing from ruin probability alone sounds quantitative and is not an answer to the question that was asked. It is also fragile: the interviewer will simply reply "suppose I have a billion dollars of bankroll", and a candidate who only has the ruin argument now has to concede that the probability really did fall, with nothing to say about why the expectation did not. The candidate who named the condition can answer in one line, that the bankroll changes which condition binds and never changes the conclusion, and then show that doubling the bankroll halves the ruin probability and doubles the loss in perfect step.

Two follow-ups are common enough to prepare. "Does a stop-loss improve your expected return?" No, on a martingale it is exactly neutral, and what it buys is a bounded worst case and slightly lower variance. "Why is gambler's ruin covered by the theorem at all?" Because between two absorbing walls the process is bounded and reaches a wall almost surely, so condition (b) holds, and the ruin probability drops out of a single expectation equation rather than a recursion. Being able to derive ${10/16}$ in one line, instead of solving a difference equation at the whiteboard, is usually the moment the conversation turns.
