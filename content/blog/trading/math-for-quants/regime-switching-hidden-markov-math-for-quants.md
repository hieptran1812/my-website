---
title: "Regime switching and hidden Markov models: when the model itself changes"
date: "2026-09-21"
publishDate: "2026-09-21"
description: "Markets are not one process. A hidden Markov model treats the regime as a state you can only infer, which is useful because the inference arrives as a probability. It also arrives late, and the lateness is the part that costs money."
tags: ["regime-switching", "hidden-markov-models", "forward-backward", "baum-welch", "viterbi", "look-ahead-bias", "volatility-regimes", "markov-chains", "quant-interview", "math-for-quants"]
category: "trading"
subcategory: "Quantitative Finance"
author: "Hiep Tran"
featured: false
readTime: 19
---

> [!important]
> **TL;DR:** A hidden Markov model does not predict a regime change. It detects one, late, with a probability attached, and the lateness is a property of the model rather than a flaw in your fit.
>
> - The regime is a **state you never observe**. You see returns. The model infers which of K distributions produced them, and returns a probability instead of a call.
> - Four questions, four algorithms. A desk can only trade the **filtered** probability, which uses data up to today. Smoothing and Viterbi both read the future.
> - In a two-state model with a calm regime at 12.70% annualised volatility and a stressed one at 30.16%, the filter crosses 50% a **median of 2 trading days** after the true switch, and **14.0%** of stressed spells end before it ever crosses.
> - On a \$500m book that cuts to 30% exposure on the signal, that lag costs **\$1,944,950** per episode on average, roughly **\$4,074,670 a year**, with a standard deviation of **\$12,865,650** that dwarfs the mean.
> - Backtesting on **smoothed** probabilities lifts the Sharpe from 0.539 to 0.689, beating an oracle that knows the true regime in real time. That \$6,530,000 a year is pure look-ahead.

Every model in a quant's first two years makes the same quiet promise: there is one data-generating process, and enough data will pin it down.

Markets break that promise in public. A period of 8% annualised volatility and a period of 40% annualised volatility are not two draws from one distribution. Fit a single normal across both and you get a mean nobody experienced and a variance that was wrong every day: too wide in the calm, far too narrow in the storm.

The honest response is to say there is more than one process, and that which one is running today is itself unknown. That is a hidden Markov model. It earns its place not because it forecasts the switch, because it does not, but because it turns a question people answer with a hunch into a number with error bars.

![Two stacked lanes over the same 120 days: an observed daily return series that widens in the middle third, and below it the hidden regime bar that explains why](/imgs/blogs/regime-switching-hidden-markov-math-for-quants-1.webp)

That figure is the mental model for the whole post. The top lane is what arrives on your screen. The bottom lane is what you want and never get.

## The building blocks: a Markov chain you cannot see

Three pieces, defined from zero.

**A state** is which regime the market is in today. Call it $S_t$, taking one of K values. In this post K is 2: state 1 is *calm*, state 2 is *stressed*. The state is not a number you can look up. It is not realised volatility, and it is not a VIX level. It is a label on the underlying process, and nobody publishes it.

**A Markov chain** governs how the state moves. The Markov property says tomorrow's state depends on today's state and nothing earlier. The chain is summarised by a **transition matrix** $P$, whose entry $p_{jk}$ is the probability of moving from state $j$ to state $k$. Rows sum to 1.

**An emission distribution** is what each state produces: given the state, the return is an ordinary draw, $r_t \mid S_t = k \sim N(\mu_k, \sigma_k^2)$. The state does not set the return. It sets the distribution the return comes from, which is why a single quiet day proves nothing.

Put together, the model used throughout this post is:

$$
P = \begin{pmatrix} 0.99 & 0.01 \\ 0.05 & 0.95 \end{pmatrix}, \qquad
r_t \mid S_t = 1 \sim N(0.05,\ 0.80^2), \qquad
r_t \mid S_t = 2 \sim N(-0.15,\ 1.90^2)
$$

with returns in percent per day. The calm regime drifts up 0.05% a day at 0.80% daily volatility, which is ${0.80 \times \sqrt{252} = 12.70\%}$ annualised. The stressed regime drifts down 0.15% a day at 1.90%, or 30.16% annualised. Every number in this post is illustrative arithmetic on these assumed parameters, not an empirical estimate from real market data.

![A two-node state graph: CALM and STRESSED, each carrying its emission distribution and annualised volatility, joined by self-loops of 0.99 and 0.95 and cross arrows of 0.01 and 0.05](/imgs/blogs/regime-switching-hidden-markov-math-for-quants-2.webp)

Two quantities fall straight out of $P$ and are worth computing before anything else.

The **stationary distribution** is the long-run share of days in each state, found by solving $\pi P = \pi$. For a two-state chain it has a closed form: ${\pi_1 = (1-p_{22}) / [(1-p_{11}) + (1-p_{22})]}$. Here that is ${0.05 / (0.01 + 0.05) = 5/6 = 83.3\%}$ calm and ${1/6 = 16.7\%}$ stressed.

The **expected spell length** in state $k$ is ${1/(1-p_{kk})}$, because leaving is a geometric coin flip each day. Calm spells run 100 days on average, stressed spells 20. Mixing two regimes in those proportions produces an unconditional volatility near 17% annualised and fat tails no single normal can. That is the same stylised fact [ARCH and GARCH](/blog/trading/math-for-quants/arch-garch-volatility-math-for-quants) capture with a smoothly evolving variance. Regime switching is the discrete alternative: not a variance that drifts, but a variance that jumps between two settings.

The closest relative in this series is the [Kalman filter](/blog/trading/math-for-quants/kalman-filter-state-space-math-for-quants), which tracks a *continuous* hidden state. Swap the continuous state for a discrete one and the Kalman recursion becomes the forward recursion below, with the same philosophy: predict, then correct with the new observation.

## Four questions, and the only one you can trade

Rabiner's 1989 tutorial organises hidden Markov models around three canonical problems. Finance needs a fourth distinction inside the second one, because that is where the money is lost.

![A four-row matrix mapping each question to its algorithm, the data it uses, and whether it is usable in real time](/imgs/blogs/regime-switching-hidden-markov-math-for-quants-3.webp)

**Likelihood.** How probable is the observed return series under this model? The **forward algorithm** answers it in $O(K^2 T)$ by carrying a vector of probabilities through time instead of enumerating $K^T$ paths. You need it to fit and compare models, and you never trade it.

**Filtering.** What is the probability of each regime *today*, given everything up to today: $P(S_t = k \mid r_1, \ldots, r_t)$? This is the forward recursion again, normalised at each step. It is the only one of the four adapted to the information you actually have, in the sense the post on [filtrations and no look-ahead](/blog/trading/math-for-quants/filtrations-no-lookahead-math-for-quants) makes precise.

**Smoothing.** What was the probability of each regime on day $t$, given the *whole* sample: $P(S_t = k \mid r_1, \ldots, r_T)$? The **forward-backward algorithm** answers this, and it is the right tool for describing history. It uses returns from after day $t$, so it cannot be computed on day $t$, and a strategy trading it is trading tomorrow's newspaper.

**Decoding.** What single sequence of states was most likely overall? **Viterbi** answers it with the same dynamic program, keeping the best path rather than summing over paths. It also reads the whole sample.

The distinction is the whole post. Three of the four algorithms use data a live trader will not have. The filtered probability is the only one a desk can act on, and it is the worst-behaved of the three, which is exactly why people quietly reach for the others.

## Worked example 1: running the filter by hand

The forward recursion has two steps a day. **Predict**: push yesterday's belief through the transition matrix. **Update**: reweight by how well each state explains today's return, then renormalise. That second step is Bayes' rule with the emission density as the likelihood, which makes this the same machinery as [Bayesian updating](/blog/trading/math-for-quants/bayesian-inference-traders-math-for-quants), run once a day forever.

With $\xi_{t|t}$ the filtered vector and $f_k(r)$ the state-$k$ density:

$$
\xi_{t|t-1} = \xi_{t-1|t-1} P, \qquad
\xi_{t|t}(k) = \frac{\xi_{t|t-1}(k)\, f_k(r_t)}{\sum_{j} \xi_{t|t-1}(j)\, f_j(r_t)}
$$

Start from the stationary distribution, $\xi_{0|0} = (5/6,\ 1/6) = (0.83333,\ 0.16667)$, and feed in three days: -0.40%, -2.10%, -2.60%. The density is the ordinary normal one, so for the first day and the calm state,

$$
f_1(-0.40) = \frac{1}{0.80\sqrt{2\pi}} \exp\!\left(-\tfrac{1}{2}\left(\tfrac{-0.40 - 0.05}{0.80}\right)^{2}\right) = 0.425709
$$

| day | predict (calm, stressed) | $f_{\text{calm}}$ | $f_{\text{stressed}}$ | ratio | numerators | sum | posterior (calm, stressed) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1, $r=-0.40$ | 0.83333, 0.16667 | 0.425709 | 0.208160 | 0.489 | 0.35475608, 0.03469403 | 0.38945011 | 0.91092, **0.08908** |
| 2, $r=-2.10$ | 0.90626, 0.09374 | 0.013472 | 0.124002 | 9.204 | 0.01220913, 0.01162395 | 0.02383308 | 0.51228, **0.48772** |
| 3, $r=-2.60$ | 0.53154, 0.46846 | 0.002066 | 0.091432 | 44.256 | 0.00109816, 0.04283223 | 0.04393039 | 0.02500, **0.97500** |

Three things in that table are worth more than the arithmetic.

**The stationary vector is a fixed point of the predict step.** Row 1's prediction equals its input, because ${0.83333 \times 0.99 + 0.16667 \times 0.05 = 0.83333}$. Start anywhere else and the predict step drags you back toward 83/17 every day. That drag is a prior fighting every piece of evidence you feed in.

**A mild down day is evidence for calm.** The ratio on day 1 is 0.489, so a -0.40% return roughly halves the odds on stress and the filter moves *down*, from 16.7% to 8.9%. Beginners expect any red day to raise the stress probability. What matters is the size relative to each regime's scale.

**A 2.7-sigma day is not enough.** Day 2's -2.10% is 2.69 standard deviations below the calm mean and carries a likelihood ratio of 9.204, yet the posterior on stress lands at 48.772%, just under the threshold. A desk running "cut when stress exceeds 50%" does not cut. It carries the full \$500m into day 3, and day 3 is -2.60%.

![Predict then update, shown with the actual day-2 numbers from the table: 0.91092 and 0.08908 predicted forward to 0.90626 and 0.09374, reweighted by the densities, normalised to 0.51228 and 0.48772](/imgs/blogs/regime-switching-hidden-markov-math-for-quants-4.webp)

That is the lag, visible in three observations. Now quantify it.

## Fitting it: Baum-Welch as EM

You will not be handed $P$, $\mu$ and $\sigma$. Baum-Welch estimates them, and it is the [EM algorithm](/blog/trading/math-for-quants/mle-method-of-moments-math-for-quants) specialised to this model. The structure, without the derivation:

1. Guess parameters.
2. **E step.** Run forward-backward to get, for every day, the smoothed probability of each state and of each state *pair* across consecutive days.
3. **M step.** Re-estimate as if those probabilities were observed counts. Each $p_{jk}$ becomes the expected number of $j \to k$ transitions divided by the expected time in $j$. Each $\mu_k$ and $\sigma_k$ becomes a probability-weighted sample mean and variance.
4. Repeat. Each iteration cannot decrease the likelihood, which is the guarantee EM buys you and the only one it buys you.

That last clause matters. Monotone ascent is not convergence to the global optimum, and the likelihood surface of a mixture model is genuinely multimodal, for the same reason [MCMC chains get stuck](/blog/trading/math-for-quants/mcmc-metropolis-gibbs-math-for-quants) on models with discrete latent states.

To see how bad it gets, simulate 2,520 days (ten years) from the model above and fit it back with 40 random starts. With the correct K of 2, **all 40 starts land on the same optimum**, log-likelihood -3671.907. With K of 3, the 40 starts produce **31 distinct optima** and only 5.0% of them find the best one. With K of 4, 38 distinct optima and 2.5%. A single run of Baum-Welch on a four-state model is, with 97.5% probability, reporting a false summit with total confidence.

There is a second and subtler problem. The K of 2 fit recovers $\hat\mu = (+0.0679, -0.1882)$ and $\hat\sigma = (0.7909, 1.9783)$, all close to truth, and $\hat p_{22} = 0.9672$ against a true 0.95. That looks like a rounding error. It is not, because the quantity you care about is the spell length ${1/(1-\hat p_{22}) = 30.5}$ days against a true 20. Ten years of daily data contains only about 21 stressed spells, and duration is a very steep function of persistence near 1. The parameter is nearly right and the number you would tell a risk committee is more than 50% too long.

Worth noting for calibration: the fitted K of 2 model scores -3671.907 against -3676.026 for the *true* parameters. The fit beats the truth by 4.12 in sample, with 7 parameters. Even the correct model overfits.

## How many states? Two, usually

With the same ten-year path, compare models by BIC, which is $-2 \log L + p \log T$ with $p$ the parameter count and ${\log(2520) = 7.8320}$:

| K | parameters | best log-likelihood | BIC | penalty for the extra states |
| --- | --- | --- | --- | --- |
| 2 | 7 | -3671.907 | 7398.638 | baseline |
| 3 | 14 | -3667.491 | 7444.630 | +45.99 |
| 4 | 23 | -3665.343 | 7510.822 | +112.18 |

Going from two states to four buys 6.56 log-likelihood points for 16 extra parameters. BIC is right to reject it, but the interesting part is *what the extra states actually are*.

The third state the K of 3 fit invents has mean +3.34%, standard deviation 0.4565%, and self-persistence 0.0446. Read that back: a "regime" that lasts one day, fires almost never, and has almost no variance. It is not a regime. It is a tight cluster wrapped around one outlier day, and it is the visible edge of a real pathology. A Gaussian mixture likelihood is unbounded: park a component on a single observation and shrink its variance toward zero and the likelihood goes to infinity. Every fit you have ever run was saved by a variance floor or by luck.

The K of 4 fit is worse in a different way. Three of its four states have self-persistence below 0.63, meaning an expected life under three days. A state that does not persist is not a regime, it is a shape the optimiser used to absorb noise, and it will be a different shape on next year's data.

Two states survive this because volatility regimes really do have two economically distinct settings, and because two states need only 7 parameters, which 2,520 observations can support. Three can be defensible when you have a genuine third mechanism in mind, such as a crisis state distinct from an ordinary drawdown, and you should specify it in advance rather than let BIC find it.

## Worked example 2: how late is late, and what it costs

This is the honest core. Simulate 4,000,000 days from the model, which contains 33,259 stressed spells with a mean length of 19.997 days, run the filter, and measure how long after each true switch the filtered stress probability first crosses 50%.

| statistic | value |
| --- | --- |
| crosses on the switch day itself | 23.7% of spells |
| median lag | 2 days |
| mean lag | 2.72 days |
| 90th percentile lag | 7 days |
| **never crosses before the spell ends** | **14.0% of spells** |
| mean length of those undetected spells | 3.43 days |
| lag on the way *out*, back below 50% | 2.56 days on average |

![The filtered stress probability climbing across the 50% threshold two days after the true regime has already switched, with the full-risk window shaded](/imgs/blogs/regime-switching-hidden-markov-math-for-quants-5.webp)

The model here is *perfectly specified*. These are not estimation errors or a bad fit. This is the irreducible delay of inferring a state from noisy draws, and no amount of data removes it. The intuition is information-theoretic: the expected log-likelihood ratio in favour of stress, per stressed day, is the Kullback-Leibler divergence between the two emission densities, 1.4866 nats. From a deep-calm prior around 2% you need roughly 3.9 nats, so between two and three typical days. Some days deliver far more and some deliver less than nothing, which is why the distribution has a 7-day tail and a 14.0% failure rate.

Now put money on it. Take a **\$500m** equity book that holds full exposure while the filter says calm and cuts to 30% when stress crosses 50%, executing the morning after the signal. The extra exposure carried through the lag is 70% of \$500m, or \$350m.

Measuring the realised return over that window across the simulated switches gives a mean of -0.5557%, a standard deviation of 3.6759%, and a 5th percentile of -5.9018%. On \$350m of unwanted exposure:

| | window return | on \$350m |
| --- | --- | --- |
| mean cost per episode | -0.5557% | -\$1,944,950 |
| standard deviation | 3.6759% | \$12,865,650 |
| 5th percentile episode | -5.9018% | -\$20,656,300 |

With 2.095 stressed spells a year, the expected annual drag is 2.095 × \$1,944,950 = \$4,074,670, or 0.81% of the book.

The mean is not the point. The standard deviation is **6.6 times the mean**, and one episode in twenty costs more than \$20m. The lag does not charge a predictable fee for insurance that arrives late. It leaves you fully exposed for a few days, at random, precisely when the distribution has its fattest left tail. A committee told "our regime model cuts risk in a crisis" is hearing something true on average and dangerously incomplete in the tail.

## Worked example 3: the look-ahead trap

Here is the most common way this model is abused, and it is rarely deliberate. You fit on ten years of history. Fitting requires forward-backward, so the smoothed probabilities are sitting in memory. You build the backtest from them. The equity curve is beautiful, and it is fiction.

Run both versions on 2,000 independent ten-year paths. Exposure is 1.00 when the stress probability is at or below 0.50 and 0.30 above it, set on day $t$ and applied to day $t+1$'s return. The only difference is which probability sets the weight.

| strategy | probability used | annual return | annual volatility | Sharpe |
| --- | --- | --- | --- | --- |
| buy and hold | none | 4.280% | 16.92% | 0.261 |
| **filtered** | $r_1 \ldots r_t$ | 7.241% | 13.46% | **0.539** |
| **smoothed** | $r_1 \ldots r_T$ | 8.547% | 12.39% | **0.689** |
| oracle, true regime known | the state itself | 8.369% | 12.42% | 0.674 |

Smoothing lifts the Sharpe by 0.150, from 0.539 to 0.689. The return gap of 8.547% - 7.241% = 1.306% a year is **\$6,530,000** on the \$500m book, \$65,300,000 over the ten years.

The damning row is the last one. The smoothed backtest at 0.689 **beats the oracle at 0.674**, a strategy handed the true regime every morning with certainty. No real signal can beat perfect knowledge of the state. Smoothing does, because it is not merely inferring the regime, it is reading returns that have not happened yet and reweighting the days within a regime accordingly.

![The filtered and smoothed probability paths on the same axes: the filtered line crosses 50% two days after the switch, the smoothed line was already above it three days before](/imgs/blogs/regime-switching-hidden-markov-math-for-quants-6.webp)

The mechanism shows up directly. On the day *before* a regime actually changes, the smoothed path is already above 50% **25.7%** of the time. The filtered path manages 0.49%. The smoother appears to predict the switch because it has already seen it, and a backtest cannot tell the difference between prescience and hindsight.

If you take one operational rule from this post: **build the strategy's signal from a separate filtering pass that is never given data past the trade date, even if the fit used the whole sample.** Better still, refit on a rolling window and filter forward only. A smoothed probability in a backtest is a bug with a plausible alibi.

## Common misconceptions

**"A regime model predicts the switch."** It detects one, and it detects late. Nothing in the model contains information about the future: the transition probability out of calm is 0.01 every single day, whatever happened yesterday. The value of the model is that it reports 48.772% instead of a yes or a no, and a probability can be sized against, which a call cannot.

**"More states fit better, so they are better."** More states always fit better in sample, by construction. Above, K of 4 bought 6.56 log-likelihood points and lost 112.18 on BIC, and the states it found lasted under three days. Worse, the multimodality gets severe exactly where you stop being able to diagnose it: 2.5% of starts found the best four-state optimum. If you cannot name the economic mechanism behind state 3 before you fit it, you are fitting noise.

**"A regime model removes the need for a stop."** It replaces a fast, dumb, certain rule with a slow, smart, probabilistic one, and the two fail in different places. The stop fires on the 14.0% of stressed spells the filter never detects; the filter handles the slow grind the stop whipsaws through. The honest architecture runs both and accepts the overlap.

**"Regimes are volatility regimes."** Volatility is where they are easiest to identify, because the emission distributions separate most cleanly on scale. Correlation, liquidity and mean reversion all switch too, and often earlier. A model reading only volatility finds the crisis after the correlation structure has already told you.

## How it shows up in real markets

The technique arrived in economics through Hamilton (1989), which applied a two-state Markov-switching model to US GNP growth and recovered contraction dates that lined up with the NBER's, from a model that was given no recession labels at all. That paper is the reason "regime switching" means what it means in finance. The real-time version of the same question, how fast such a filter can call a turning point as it happens rather than in revised data, is what Chauvet and Hamilton (2006) study, and the answer has always been "with a delay".

On a multi-strategy desk the model usually appears in three places. In **risk**, as a volatility-regime overlay that scales gross exposure. In **allocation**, in the tradition of Ang and Bekaert (2002), where the optimal portfolio differs across regimes because correlations rise in the stressed one and diversification stops working exactly when it is needed. In **execution**, as a fast two-state model over order flow deciding whether the book is in a normal or a toxic regime.

In all three the failure mode is the same and it is organisational rather than mathematical. The model is built by research on smoothed history, shown to a committee as a backtest, and then deployed with filtered probabilities. Live performance comes in far below the backtest, and the gap gets attributed to costs, or capacity, or the market changing. It was 0.150 of Sharpe that was never there.

## Sources and further reading

- Hamilton, J. D. (1989). "A New Approach to the Economic Analysis of Nonstationary Time Series and the Business Cycle." *Econometrica* 57(2), 357-384. The founding finance application.
- Rabiner, L. R. (1989). "A Tutorial on Hidden Markov Models and Selected Applications in Speech Recognition." *Proceedings of the IEEE* 77(2), 257-286. Still the clearest statement of the three canonical problems.
- Baum, L. E. and Petrie, T. (1966). "Statistical Inference for Probabilistic Functions of Finite State Markov Chains." *Annals of Mathematical Statistics* 37(6), 1554-1563.
- Dempster, A. P., Laird, N. M. and Rubin, D. B. (1977). "Maximum Likelihood from Incomplete Data via the EM Algorithm." *Journal of the Royal Statistical Society B* 39(1), 1-38. Baum-Welch is the special case.
- Viterbi, A. J. (1967). "Error Bounds for Convolutional Codes and an Asymptotically Optimum Decoding Algorithm." *IEEE Transactions on Information Theory* 13(2), 260-269.
- Kim, C.-J. (1994). "Dynamic Linear Models with Markov-Switching." *Journal of Econometrics* 60(1-2), 1-22. The smoother used for the smoothed path above.
- Ang, A. and Timmermann, A. (2012). "Regime Changes and Financial Markets." *Annual Review of Financial Economics* 4, 313-337. The survey to read first.
- Ang, A. and Bekaert, G. (2002). "International Asset Allocation With Regime Shifts." *Review of Financial Studies* 15(4), 1137-1187.
- Chauvet, M. and Hamilton, J. D. (2006). "Dating Business Cycle Turning Points." In *Nonlinear Time Series Analysis of Business Cycles*, Elsevier.
- Hamilton, J. D. (1994). *Time Series Analysis*, Princeton University Press, chapter 22.

All simulation figures in this post are illustrative arithmetic on the stated two-state model, not estimates from market data.

## In the interview room and on the desk

The question usually arrives as "how would you tell whether the market has changed regime?", and it is deliberately open. Weak candidates start listing indicators. Strong ones reframe it in the first sentence.

**Separate detection from prediction immediately.** Say that a regime model infers a hidden state from observed returns, that it returns a posterior probability rather than a call, and that it contains no information about *when* the next switch will occur, because the transition probability is constant. If the interviewer wanted prediction, the honest answer is that this is the wrong tool.

**Then volunteer the lag before you are asked.** This is the move that separates a candidate who has read the tutorial from one who has run the model. Say that filtered probabilities are most confident *after* the regime has already turned, give the order of magnitude, a couple of days at the median with a tail of a week or more, and say that a material share of short spells are never detected at all. Then name the consequence: a regime-switched allocation is systematically late, the cost is not the mean but the variance of being late, and it must be sized accordingly.

**Then name the three algorithms and which one is tradeable.** Forward gives the likelihood and the filtered probability. Forward-backward gives smoothed probabilities using the whole sample. Viterbi gives the single most likely path, also using the whole sample. Only the filtered probability is adapted to your information set.

**The trap is presenting a backtest built on smoothed probabilities.** It is the single most common way this model is abused, and it is usually accidental, because fitting by Baum-Welch produces smoothed probabilities as a by-product and they are the ones sitting in the notebook. A good interviewer will ask how the regime path in your backtest was computed, and the correct answer is a separate filtering pass with no data past the trade date. If you can add that a smoothed backtest can beat an oracle that knows the true state, and explain why that proves look-ahead rather than skill, you have answered the question better than it was asked.

Say what a two-state model can and cannot support, and why you would not reach for four. **Two Sigma** and **Citadel** weight this in research rounds as a look-ahead discipline question more than a modelling one. Any **macro or multi-strategy** seat weights it as a risk-overlay question, where the follow-up is always "and what does it cost you to be late?"
