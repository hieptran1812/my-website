---
title: "Monte Carlo variance reduction: buying accuracy without buying paths"
date: "2026-09-21"
publishDate: "2026-09-21"
description: "Monte Carlo error falls as one over the square root of the path count, so a tenfold improvement costs a hundredfold more compute. Antithetics, control variates, importance sampling, stratification and quasi-Monte Carlo are how a desk buys that accuracy by being clever instead."
tags: ["monte-carlo", "variance-reduction", "control-variates", "antithetic-variates", "importance-sampling", "quasi-monte-carlo", "sobol", "asian-options", "derivatives-pricing", "quant-interview", "math-for-quants"]
category: "trading"
subcategory: "Quantitative Finance"
author: "Hiep Tran"
featured: false
readTime: 20
---

> [!important]
> **TL;DR:** A Monte Carlo price is a sample mean, so its error falls only as one over the square root of the path count. Buying accuracy with hardware is the most expensive way to get it.
>
> - **The square-root law.** Halving the error bar costs four times the paths. Getting one more decimal digit costs a hundred times the paths. On a \$22.7m block of at-the-money calls, one million paths leaves a standard error of \$35,798.
> - **Antithetic variates** pair every draw with its mirror. They help only when the payoff is monotone in the shock. On that same call they cut the variance 1.672-fold. On a payoff that is even in the shock they are exactly twice as expensive for nothing.
> - **Control variates** subtract an error you can measure. The optimal coefficient is a regression slope, and the leftover variance is ${1-\rho^2}$. A geometric-average Asian controls an arithmetic one at ${\rho = 0.999137}$, which is a **580-fold** variance cut: an error bar of \$64,150 on a \$42.4m program becomes \$2,665.
> - **Importance sampling** moves the sampler to where the payoff lives and reweights. On a deep out-of-the-money call it cuts variance 76-fold. Push the shift too far and the weights explode: at six standard deviations it is 7.9 times *worse* than doing nothing.
> - **Report the estimate, the standard error and the path count.** A Monte Carlo price quoted without an error bar is not a price, it is a rumour with decimals.

A trading desk asks the pricing library for a number and gets 8.4555. It sounds precise. It is not. Run the same code with a different seed and you get 8.5021. The digits after the first two were never real, and nothing in the output told you where they stopped being real.

That is the whole problem. Monte Carlo is the most general pricing tool there is: give it any payoff, however path-dependent, and it returns a number. What it returns is a *sample mean*, and a sample mean carries a standard error that shrinks agonisingly slowly. The engineering answer is to rent more machines. The quant answer is that the standard error has two factors, the path count and the payoff's own variability, and only one of them is expensive to change.

![Standard error on a block quote falling as one over the square root of the Monte Carlo path count, annotated with the cost of halving the error](/imgs/blogs/monte-carlo-variance-reduction-math-for-quants-1.webp)

The dollar figures throughout are illustrative arithmetic on assumed inputs, stated spot, strike, rate and volatility, with every intermediate computed rather than quoted. The variance factors are measured on those examples, not borrowed from a textbook.

## The foundations: what a Monte Carlo price actually is

A derivative pays some amount that depends on the path of the underlying. Under the risk-neutral measure its price today is the discounted expected payoff. Monte Carlo replaces that expectation with an average over simulated paths.

Write ${Y_i = e^{-rT} f(S^{(i)})}$ for the discounted payoff along the ${i}$-th simulated path. The estimator is the sample mean:

$$
\hat{P}_n = \frac{1}{n}\sum_{i=1}^{n} Y_i .
$$

Two facts follow, and they are the only two facts in this entire subject.

**It is unbiased.** The expected value of the sample mean is the expected value of one draw, which is the price. No amount of simulation error makes it systematically too high or too low. That is why Monte Carlo is trusted at all, and it is a consequence of the [law of large numbers](/blog/trading/math-for-quants/law-large-numbers-central-limit-theorem-math-for-quants).

**Its standard error is ${\sigma_Y/\sqrt{n}}$**, where ${\sigma_Y}$ is the standard deviation of a *single* path's discounted payoff, not of the price. The central limit theorem makes the estimator approximately normal around the true price with that standard deviation, which is what lets you put a confidence interval on it.

That square root is the tyrant. To cut the error by a factor ${k}$ you need ${k^2}$ times the paths, so every doubling of precision quadruples the bill.

#### Worked example 1: the standard error on a \$22.7m block

A market maker quotes a block of 2,000,000 one-year European calls. Spot \$100, strike \$100, risk-free rate 3%, volatility 25%. Black-Scholes prices one option at \$11.3485, so the block is worth

$$
11.3485 \times 2{,}000{,}000 = \$22{,}697{,}000 .
$$

What is the standard deviation of a single path's payoff? For a European call it has a closed form, because the second moment of the payoff can be written down directly:

$$
E\!\left[\left((S_T-K)^{+}\right)^{2}\right] = S_0^2 e^{(2r+\sigma^2)T} N(d_1+\sigma\sqrt{T}) - 2KS_0 e^{rT} N(d_1) + K^2 N(d_2).
$$

With ${d_1 = 0.245}$ and ${d_2 = -0.005}$, discounting and subtracting the squared mean gives ${\sigma_Y = 17.899}$. Note that the standard deviation of one path's payoff is **larger than the price itself**, by a factor of 1.58. That is normal for options and it is the reason naive Monte Carlo is so slow.

Now price the error bar:

| Paths | Standard error per option | Standard error on the block |
| --- | --- | --- |
| 10,000 | \$0.17899 | \$357,980 |
| 100,000 | \$0.056602 | \$113,204 |
| 1,000,000 | \$0.017899 | \$35,798 |
| 4,000,000 | \$0.0089495 | \$17,899 |
| 100,000,000 | \$0.0017899 | \$3,580 |

At one million paths the quote is \$22,697,000 plus or minus \$35,798 at one standard error, so a two-standard-error band spans ${4 \times 35{,}798 = \$143{,}192}$. That is wider than the edge on most block trades. Halving it took four million paths. Getting the error down to \$3,580 took one hundred million.

The lesson is not that Monte Carlo is bad. It is that ${n}$ is the expensive lever and ${\sigma_Y}$ is the cheap one. Every technique below attacks ${\sigma_Y}$.

## Antithetic variates: pair every path with its mirror

The simplest idea in the subject. Every Gaussian shock ${Z}$ has an equally likely mirror ${-Z}$. So instead of ${n}$ independent draws, take ${n/2}$ draws and use each one twice, once as itself and once negated. Average each mirrored pair before averaging across pairs.

The variance of one pair average is

$$
\mathrm{Var}\!\left(\frac{h(Z)+h(-Z)}{2}\right) = \frac{\mathrm{Var}(h) + \mathrm{Cov}\big(h(Z),\,h(-Z)\big)}{2},
$$

while two *independent* draws give ${\mathrm{Var}(h)/2}$. Comparing the two lines gives the exact condition, and it is worth memorising because interviewers ask for it:

**Antithetic variates help if and only if ${\mathrm{Cov}(h(Z), h(-Z)) \lt 0}$.** That is guaranteed when ${h}$ is monotone, because ${h(Z)}$ and ${h(-Z)}$ are then oppositely ordered. It is guaranteed for nothing else.

![Two panels contrasting a monotone call payoff where mirrored draws cancel with an even variance-swap payoff where they coincide](/imgs/blogs/monte-carlo-variance-reduction-math-for-quants-2.webp)

Take the call from worked example 1. Its ${d_2 = -0.005}$ is negative, which means the two mirrored paths can never *both* finish in the money: one needs ${Z \gt 0.005}$ and the other needs ${Z \lt -0.005}$. The product of the two payoffs is identically zero, so the covariance is exactly minus the squared price:

$$
\mathrm{Cov}\big(h(Z), h(-Z)\big) = 0 - 11.3485^2 = -128.79 .
$$

With ${\mathrm{Var}(h) = 17.899^2 = 320.37}$ the correlation between the mirrored payoffs is ${-128.79/320.37 = -0.402}$. Per draw, antithetic sampling replaces ${\mathrm{Var}(h)}$ with ${\mathrm{Var}(h) + \mathrm{Cov}}$, the halves cancelling, so the variance ratio is

$$
\frac{320.37}{320.37 - 128.79} = \frac{320.37}{191.58} = 1.672 .
$$

The standard error falls by ${1/\sqrt{1.672} = 0.773}$, so the \$35,798 error bar on the block becomes \$27,672, a 23% improvement for one line of code. Equivalently, the same accuracy now costs 60% of the paths.

**Where it goes wrong.** A one-period variance swap pays, with zero drift, in proportion to the *squared* shock. That payoff is even: ${h(Z) = h(-Z)}$ identically. The correlation is ${+1}$, the pair average equals the single draw, and two draws bought what one would have told you, exactly twice as expensive as doing nothing. Any payoff with a symmetric kink, a straddle, a butterfly, a volatility payoff, sits somewhere on the way to that.

Even when the sign is right the size can be trivial. On the deep out-of-the-money call of worked example 3 below, ${\mathrm{Var}(h) = 2.2045^2 = 4.85982}$ and the covariance is ${-0.2190^2 = -0.047961}$, so the variance ratio is ${4.85982/4.81186 = 1.010}$. One percent. On tail options, antithetic variates are a rounding error.

## Control variates: subtract an error you can measure

This is the technique that earns its keep. Suppose each simulated path produces, alongside the payoff ${Y}$ you want, some other quantity ${X}$ whose true expectation ${\mu_X}$ you know exactly. Then you can *see* how wrong the run was on ${X}$, by comparing its simulated average against the truth. If ${X}$ and ${Y}$ move together, that visible error tells you about the invisible one.

The controlled estimator is

$$
Y(b) = Y - b\,(X - \mu_X),
$$

which is unbiased for every choice of ${b}$, since ${E[X - \mu_X] = 0}$. Its variance is a quadratic in ${b}$:

$$
\mathrm{Var}\big(Y(b)\big) = \mathrm{Var}(Y) - 2b\,\mathrm{Cov}(X,Y) + b^2\,\mathrm{Var}(X).
$$

Differentiate, set to zero, and the optimum falls out:

$$
b^{\ast} = \frac{\mathrm{Cov}(X,Y)}{\mathrm{Var}(X)}, \qquad \mathrm{Var}\big(Y(b^{\ast})\big) = \mathrm{Var}(Y)\,\big(1-\rho^2\big).
$$

Two readings of that matter. First, ${b^{\ast}}$ is exactly the slope from [regressing](/blog/trading/math-for-quants/regression-ols-gls-regularized-math-for-quants) ${Y}$ on ${X}$, and ${1-\rho^2}$ is the residual variance fraction. A control variate *is* a regression. Second, the factor ${{1}/{(1-\rho^2)}}$ is violently non-linear in ${\rho}$: at ${\rho = 0.9}$ you get 5x, at ${0.99}$ you get 50x, at ${0.999}$ you get 500x. The search for a control is a search for a *very* correlated twin, not a merely helpful one. Its level does not matter, only its correlation, because ${b^{\ast}}$ absorbs any difference in scale.

#### Worked example 2: a geometric Asian controlling an arithmetic one

An arithmetic-average Asian call pays the average of the underlying over monitoring dates, less the strike, floored at zero. It has no closed form, because a sum of lognormals is not lognormal. Its geometric cousin, which averages in log space, has one: a *product* of lognormals is lognormal.

Take spot \$100, strike \$100, rate 5%, volatility 30%, one year, ${m = 12}$ monthly observations. The log of the geometric average is exactly normal with

$$
\text{mean } \ln S_0 + \Big(r-\tfrac{\sigma^2}{2}\Big)\frac{m+1}{2m}T, \qquad \text{variance } \sigma^2 T\,\frac{(m+1)(2m+1)}{6m^2}.
$$

For ${m = 12}$ those coefficients are the exact fractions ${13/24}$ and ${325/864}$, so the effective volatility is ${0.30\sqrt{325/864} = 0.18400}$. Feeding that through the Kemna-Vorst formula gives the geometric Asian call at

$$
G = \$8.024703 .
$$

**Verifying that number independently.** Re-deriving my own algebra would only reproduce my own mistake, so the closed form was priced by two further routes that share none of the steps above: direct numerical quadrature of ${E[(e^{X}-K)^{+}]}$ against the normal density, and the plain Black-Scholes formula applied to an underlying with volatility 0.18400 and a cost of carry of 0.0196354 chosen so that the forward matches. All three agree to fourteen significant figures at 8.0247032233069. The control's known mean is therefore not merely self-consistent, it is right.

Now simulate. One million paths, twelve monthly steps, NumPy's default generator seeded at 0:

| Estimator | Price | Path standard deviation | Standard error |
| --- | --- | --- | --- |
| Plain arithmetic | \$8.4555 | 12.8298 | \$0.012830 |
| Controlled, ${b^{\ast} = 1.0489}$ | \$8.4752 | 0.533042 | \$0.000533 |

The measured correlation between the two payoffs is ${\rho = 0.999137}$, so the predicted variance factor is ${1/(1-\rho^2) = 580}$ and the predicted standard-error factor is ${\sqrt{580} = 24.1}$. The measured ratio ${0.012830/0.000533 = 24.07}$ matches.

**Now in money.** A corporate hedging client buys an average-price call on 5,000,000 shares, a \$500m notional program. At the controlled price the premium is

$$
8.4752 \times 5{,}000{,}000 = \$42{,}376{,}000 .
$$

The plain estimator's error bar on that number is ${0.012830 \times 5{,}000{,}000 = \$64{,}150}$. The controlled estimator's is ${0.000533 \times 5{,}000{,}000 = \$2{,}665}$. Same paths, same runtime, same machine. To reach \$2,665 by brute force you would need ${580 \times 1{,}000{,}000 = 580{,}000{,}000}$ paths. Read the other way: ${1{,}000{,}000 / 580 = 1{,}724}$ controlled paths already match a million plain ones.

And the error bar was not decoration. The plain run priced the block at ${8.4555 \times 5{,}000{,}000 = \$42{,}277{,}500}$, which is \$98,500 below the controlled figure, or 1.5 of its own standard errors. A 40-million-path run combining antithetics and the control puts the converged price at \$8.4743, within two standard errors of the controlled estimate and nowhere near the plain one.

![Scatter of arithmetic against geometric Asian payoffs hugging a fitted line, with the exactly known control mean marked](/imgs/blogs/monte-carlo-variance-reduction-math-for-quants-3.webp)

One practical note. Even the lazy choice ${b = 1}$, simply subtracting the control's error with no regression at all, gives a standard error of \$0.000801, a 16-fold improvement. Estimating ${b^{\ast}}$ from the same paths that produce the estimate introduces a small bias of order ${1/n}$; at a million paths it is far below the standard error, but on a short run use a pilot batch to fit ${b^{\ast}}$ and a fresh batch to price.

## Importance sampling: make the rare event common

Antithetics and control variates both leave the sampler alone. Importance sampling changes it.

The problem is structural. If an option only pays when the underlying moves three standard deviations, 99% of your paths return exactly zero and contribute nothing but runtime, while all the information sits in the 1% you barely visit. The fix is to draw from a distribution that visits the interesting region often, then undo the distortion with a weight. For any ${Q}$ whose support covers ${P}$'s,

$$
E_P\big[h(Z)\big] = E_Q\!\left[h(Z)\,\frac{dP}{dQ}(Z)\right],
$$

which is the [Radon-Nikodym derivative](/blog/trading/math-for-quants/radon-nikodym-densities-math-for-quants) doing exactly the job it does in [Girsanov's theorem](/blog/trading/math-for-quants/girsanov-change-of-measure-math-for-quants). The standard choice shifts the Gaussian: sample ${Z \sim N(\mu, 1)}$ instead of ${N(0,1)}$, and the likelihood ratio is

$$
L(z) = \exp\!\Big(-\mu z + \tfrac{\mu^2}{2}\Big).
$$

![Original and shifted sampling densities on the same axis with the payoff threshold marked](/imgs/blogs/monte-carlo-variance-reduction-math-for-quants-4.webp)

#### Worked example 3: pricing a deep out-of-the-money call

Three months, spot \$100, strike \$150, volatility 40%, rate 4%. Black-Scholes prices it at \$0.21900, and the probability of finishing in the money is ${N(d_2) = 1.889\%}$. The payoff is positive only when the shock exceeds ${-d_2 = 2.0773}$.

The naive estimator's path standard deviation is 2.2045, which is **10.1 times the price itself**. At one million paths the standard error per option is ${2.2045/1{,}000 = \$0.0022045}$, or 1.01% of the price. On a block of 5,000,000 options, worth ${0.21900 \times 5{,}000{,}000 = \$1{,}095{,}000}$, that is an error bar of \$11,023. A million paths and the third digit of a one-million-dollar position is still noise.

Now shift the sampler to sit on the strike, ${\mu = 2.0773}$, so that roughly half the draws land in the money instead of one in fifty-three. The path standard deviation falls to 0.25289. The standard error becomes ${0.25289/1{,}000 = \$0.00025289}$ per option, which is \$1,264 on the block and 0.115% of the price. The variance ratio is

$$
(2.2045/0.25289)^2 = 76 .
$$

To match that with brute force you would need 76,000,000 paths. The true optimal shift, found by minimising the variance numerically, is ${\mu = 2.648}$ and delivers a factor of 107, so aiming at the strike captures 76 of the available 107. That is the usual pattern.

**Where it blows up.** The second moment under the shifted measure is ${E_P[h^2 L]}$, and ${L}$ carries the factor ${e^{\mu^2/2}}$, which grows without limit in ${\mu}$. Push the shift too far and a handful of enormous weights dominate the average. At ${\mu = 6}$ the path standard deviation is 6.209, a variance ratio of 0.126: **7.9 times worse than plain Monte Carlo**, from the technique that was supposed to help. The estimator is still unbiased, which makes it worse rather than better, because the failure is invisible in the mean and shows up only as an error bar that refuses to shrink. Monitor the largest likelihood ratio and the effective sample size. If one path carries a tenth of the total weight, the number on your screen is one path's opinion.

## Stratification: force the sample to cover the space

Random sampling clumps, and over a million draws the clumping averages out only because you paid for it. Stratification removes it by construction: split the standard normal into ${k}$ equiprobable bins and draw one point from each. The variance left is the *within-bin* variance, which discards everything the bins already account for.

On the call from worked example 1 this is exact and dramatic. Ten strata cut the variance from 320.37 to 31.17, a factor of 10.3. One hundred strata cut it to 2.711, a factor of 118.

The catch is dimensional. Stratifying ${d}$ dimensions on a grid needs ${k^d}$ points, hopeless past a handful of axes. So you stratify the one or two directions that carry the payoff, typically the terminal value or the first component of a Brownian bridge, and sample the rest freely. Latin hypercube sampling is the pragmatic version: stratify every axis one-dimensionally and pair the strata at random.

## Quasi-Monte Carlo: stop being random

Sobol sequences fill the unit cube by construction, leaving fewer gaps and fewer clusters than random points, and the Koksma-Hlawka bound gives an error of order ${(\log n)^d/n}$ rather than ${n^{-1/2}}$.

Read that exponent carefully, because ${(\log n)^d}$ is catastrophic in ${d}$: taken literally the bound is worthless for a 250-step simulation at any path count you will ever run. Quasi-Monte Carlo works anyway because of **effective dimension**. An average over a path is dominated by a few smooth directions, and the sequence only has to be equidistributed in those.

Measured on the Asian option above, comparing scrambled Sobol against pseudorandom points over 32 randomisations:

| Dimension | Points | Sobol RMSE | Pseudorandom RMSE | Error ratio | Path-equivalent |
| --- | --- | --- | --- | --- | --- |
| 12 | 1,024 | 0.04832 | 0.45943 | 9.51x | 90x |
| 12 | 16,384 | 0.00691 | 0.08535 | 12.35x | 150x |
| 250 | 1,024 | 0.10718 | 0.41705 | 3.89x | 15x |
| 250 | 16,384 | 0.01422 | 0.08382 | 5.89x | 35x |

At twelve monitoring dates Sobol converges at roughly ${n^{-0.70}}$, against the theoretical ${n^{-0.5}}$ for random points, and 1,024 Sobol points do the work of about 90,000 random ones. Move to daily monitoring and the advantage does not vanish, but it shrinks by a factor of four to six in path-equivalent terms. That is the honest statement: quasi-Monte Carlo survives high nominal dimension only when effective dimension stays low, and how low it stays depends on how you construct the path.

One structural point that catches people out. A raw Sobol sequence is deterministic, so a quasi-Monte Carlo estimate has **no standard error at all**. You cannot put an error bar on it. Randomised quasi-Monte Carlo, with digital scrambling or random shifts, restores unbiasedness and lets you treat the independent randomisations as your sample. Always scramble.

## How to report a Monte Carlo number

![Matrix of variance reduction techniques against what each needs and the measured gain](/imgs/blogs/monte-carlo-variance-reduction-math-for-quants-5.webp)

A Monte Carlo price is three numbers, never one:

1. **The estimate.**
2. **The standard error**, and therefore the number of digits that survive it. If the standard error is \$0.013 there is no meaning in the fourth decimal.
3. **The path count**, plus the seed and the number of time steps. Without these nobody can reproduce you, including you next month.

One warning belongs on the same line. The standard error measures *sampling* error only. It says nothing about discretisation bias, model error, or whether the payoff was coded correctly. A tight error bar around the wrong number is the most dangerous output a pricing library can produce, which is the argument for reaching for a PDE where one exists: see [Fokker-Planck and the forward equation](/blog/trading/math-for-quants/fokker-planck-kolmogorov-forward-math-for-quants) for when one solve beats a million paths. For the code-level mechanics, see [Monte Carlo and simulation coding for quant interviews](/blog/trading/quantitative-finance/monte-carlo-simulation-coding-quant-interviews).

## Common misconceptions

**"More paths always fixes it."** More paths fix sampling error and nothing else. Sampling error falls as ${1/\sqrt{n}}$; discretisation bias from your time grid does not fall at all. At one hundred million paths the call in worked example 1 has a \$3,580 error bar on a \$22.7m block, and if you used twelve time steps where the payoff needed 250, that bias is exactly as large as it was at ten thousand. Grinding up the path count hides a bias behind a shrinking error bar, which is worse than seeing it.

**"Antithetic variates always help."** They help if and only if ${\mathrm{Cov}(h(Z), h(-Z)) \lt 0}$, which monotonicity guarantees and nothing else does. On the call above they give 1.672x, on an even payoff they are exactly 2x worse, and on a deep out-of-the-money call they give 1.010x. "Always on" is a defensible default only because most vanilla payoffs are monotone, not because the technique is safe.

**"Quasi-Monte Carlo is just better Monte Carlo."** It is a different object: deterministic, not unbiased in the Monte Carlo sense, and unscrambled it gives a number with no error bar. Its advantage rests on effective dimension, a property of your problem and your path construction, not of the sequence. A candidate who says "we use Sobol, it converges as ${1/n}$" has skipped the only interesting part of the question.

## Sources and further reading

- Paul Glasserman, *Monte Carlo Methods in Financial Engineering*, Springer, 2004. Chapter 4 covers variance reduction, chapter 5 quasi-Monte Carlo. The standard reference for everything above.
- Phelim Boyle, Mark Broadie and Paul Glasserman, "Monte Carlo methods for security pricing", *Journal of Economic Dynamics and Control* 21 (1997), 1267-1321. The survey that fixed the vocabulary of this field.
- A. G. Z. Kemna and A. C. F. Vorst, "A pricing method for options based on average asset values", *Journal of Banking and Finance* 14 (1990), 113-129. The geometric-average closed form and its use as a control.
- Russel Caflisch, William Morokoff and Art Owen, "Valuation of mortgage-backed securities using Brownian bridges to reduce effective dimension", *Journal of Computational Finance* 1 (1997), 27-46. Where effective dimension was named and measured.
- Art Owen, "Scrambled net variance for integrals of smooth functions", *Annals of Statistics* 25 (1997), 1541-1562. Why scrambling restores an error bar.

## In the interview room and on the desk

The question is almost never phrased as a mathematics question. It arrives as **"your Monte Carlo price is too slow, what do you do?"**, sometimes with the softer opening "how would you speed up a pricer?". The weak answer is more machines, more cores, a GPU. It is weak not because it is wrong but because it buys accuracy at the square-root rate, which the interviewer knows and is waiting for you to say.

A strong answer asks a question first: **what does the payoff look like?** That is the whole discriminator, and the order runs like this. Is the payoff monotone in the shock? Then antithetics are free, worth roughly 1.7x on a vanilla, and you take them. Is there a related instrument with a closed form that moves with the payoff? Then a control variate is the big one, and you quote the mechanism, that ${b^{\ast}}$ is a regression slope and the leftover variance is ${1-\rho^2}$, so a correlation of 0.999 is a 500-fold cut rather than a marginal one. Is the payoff concentrated in a rare region, a deep out-of-the-money option, a credit tail, a barrier rarely touched? Then importance sampling, with the caveat volunteered rather than extracted: shift too far and the likelihood ratio develops heavy tails and the estimator gets worse while still looking unbiased. Only then, if the dimension is low or the effective dimension is, mention scrambled Sobol.

The follow-up is usually the Asian option, because it is the cleanest case in the subject: the geometric average has a closed form, the arithmetic one does not, and the two are correlated above 0.999. If you can say why the geometric one is tractable, that a product of lognormals is lognormal while a sum is not, you have shown the thing they are testing.

The trap is quoting a price without a standard error. A candidate who says "the Monte Carlo gives 8.4555" has already failed, and adding decimals makes it worse. The right sentence is "8.475, standard error 0.013 at a million paths, so three digits". The same trap runs in the other direction: quoting a variance reduction factor without saying what it was measured on. Jane Street, Citadel and every derivatives pricing seat weight this heavily, because it is one of the few interview topics that maps directly onto something the desk does every single day.
