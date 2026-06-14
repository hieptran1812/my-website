---
title: "Estimators, MLE, and the bias-variance tradeoff for quant interviews"
date: "2026-06-14"
publishDate: "2026-06-14"
description: "Build estimators, bias, variance, MSE, maximum likelihood, the method of moments, the bias-variance tradeoff, and the Cramer-Rao bound from zero, then use the exact same machinery to crack the estimation questions that Jane Street, Two Sigma, Citadel, and DE Shaw ask in interviews."
tags:
  [
    "estimators",
    "maximum-likelihood",
    "bias-variance-tradeoff",
    "mean-squared-error",
    "quant-interviews",
    "statistics",
    "standard-error",
    "cramer-rao-bound",
    "shrinkage",
    "quantitative-finance"
  ]
category: "trading"
subcategory: "Quantitative Finance"
author: "Hiep Tran"
featured: true
readTime: 41
---

> [!important]
> **TL;DR** — Every number a quant reports — a win-rate, a volatility, a mean return — is an *estimate* computed from finite data, and an interviewer is really probing whether you understand how good that estimate is.
>
> - An **estimator** is a recipe that turns data into a guess about an unknown number; because the data is random, the estimate is random too, and its distribution is the *sampling distribution*.
> - The quality of an estimator splits cleanly: **MSE = bias² + variance**. Bias is how far off you are on average; variance is how much you bounce around. There is no cross term.
> - **Maximum likelihood** picks the parameter value that makes your observed data most probable. For a win-rate it gives the obvious answer — wins over trades — and hands you a standard error for free.
> - The **bias-variance tradeoff** is why deliberately *biased* estimators (shrinkage, regularization) often beat unbiased ones: a little bias can buy a large cut in variance and lower the total error.
> - The one number to remember: the standard error of a mean shrinks like **1 / √n**, so to halve your uncertainty you need **four times** the data. This is why estimating mean returns is nearly hopeless and estimating volatility is merely hard.

Here is a question a Two Sigma interviewer actually likes to ask, lightly disguised: *"Your strategy won 30 of its last 50 trades. What's its win-rate, and how confident are you in that number?"*

The trap is that "30 out of 50, so 60%" is only the first half of the answer. The 60% is an *estimate* — a single number squeezed out of a small, noisy sample. The real question is the second half: how much would that 60% wobble if you ran the same 50 trades again in a parallel universe? Get that wrong and you will size positions as if you know your edge to the percentage point when you barely know it to the nearest ten points.

![Reality has one true parameter you never see; you observe a finite sample, run an estimator, and report a number that is itself random.](/imgs/blogs/estimators-mle-bias-variance-quant-interviews-1.png)

The diagram above is the mental model for this entire post. Somewhere out there is a *true* number — call it the true win-rate, the true volatility, the true mean return. You never see it. What you see is a finite sample that the market coughed up, and you run it through an **estimator** — a formula — to produce a reported number. Because the sample is random, your reported number is random too. Everything quant interviewers ask about estimation is really a question about that last arrow: how far is the reported number likely to be from the truth, and can you do better?

This post builds the whole apparatus from zero. We will define parameter, statistic, and estimator; derive why the sample mean and sample variance look the way they do (including the famous "divide by n minus 1"); develop maximum likelihood and the method of moments as two recipes for inventing estimators; make the bias-variance tradeoff precise; and meet the Cramer-Rao bound, which sets a hard floor on how good an unbiased estimator can be. Along the way we solve the kind of problems that come up at Jane Street, Citadel, and DE Shaw, and we close by looking at where all of this bites on a real trading desk. No statistics background is assumed — if you can take an average, you can follow every step.

## Foundations: parameter, statistic, estimator, and the sampling distribution

Before any formula, four words. Statistics is sloppy about them in casual speech and interviewers are not, so let us pin them down.

A **parameter** is a fixed, usually unknown number that describes the process generating your data. The true probability that your strategy wins a trade is a parameter; call it $p$. The true average daily return of a stock is a parameter; call it $\mu$. The true volatility — the typical size of a daily move — is a parameter; call it $\sigma$. Parameters are properties of *reality*, not of your data. They do not change when you collect more observations; you just learn more about them. By deep convention parameters get Greek letters.

A **statistic** is any number you compute purely from the data, with no unknown quantities inside it. The fraction of your last 50 trades that won is a statistic — you can calculate it exactly. The average of your observed returns is a statistic. A statistic is a property of *your sample*, and it changes every time you collect a new sample.

An **estimator** is a statistic that you are using as a *guess* for a particular parameter. The distinction between "statistic" and "estimator" is one of intent: the sample fraction of wins is just a statistic until you announce "I'm using this as my guess for the true win-rate $p$," at which point it is an estimator *of* $p$. We write an estimator with a hat: $\hat{p}$ (read "p-hat") is an estimator of $p$; $\hat{\mu}$ estimates $\mu$. The hat means "our data-driven guess at the thing under the hat."

An **estimate** (no "-or") is the actual number the estimator spits out for one particular sample: $\hat{p} = 0.60$. The estimator is the recipe; the estimate is the dish.

Here is the idea that makes the whole subject tick, and the one beginners skip. Because your sample is random — a different 50 trades would have given different wins — *the estimator is itself a random variable*. Before you collect the data, $\hat{p}$ has not happened yet; it is a quantity with a probability distribution. That distribution has a name: the **sampling distribution** of the estimator. It is the distribution of estimates you would get if you re-ran your data collection over and over.

Almost everything in this post is a statement about the sampling distribution. "Is my estimator any good?" means "is its sampling distribution centered on the truth, and how tightly?" "How confident am I in 60%?" means "how wide is the sampling distribution of $\hat{p}$ around its center?" The sampling distribution is the object; bias, variance, MSE, standard error, and confidence intervals are all just descriptions of its shape.

One more piece of vocabulary you will use constantly. The **standard error** of an estimator is the standard deviation of its sampling distribution — the typical distance between your estimate and the center of the cloud of estimates you could have gotten. It is *not* the standard deviation of the data; it is the standard deviation of the *estimate*. The standard error answers "how much would my reported number jiggle if I redid the experiment?" and it is the single number you should attach to any estimate you report in an interview. An answer of "60%" is incomplete; "60%, with a standard error of about 7 percentage points" is the answer of someone who understands estimation.

## Bias, variance, and the decomposition MSE = bias² + variance

Now we can ask precisely what "a good estimator" means. There are two distinct ways an estimator can disappoint you, and a famous identity that ties them together.

The **bias** of an estimator is how far its sampling distribution sits from the truth, on average. Formally,

$$\text{Bias}(\hat{\theta}) = \mathbb{E}[\hat{\theta}] - \theta,$$

where $\theta$ is the true parameter, $\hat{\theta}$ is your estimator, and $\mathbb{E}[\hat{\theta}]$ is the *expected value* — the long-run average — of the estimate over all possible samples. If $\mathbb{E}[\hat{\theta}] = \theta$, the estimator is **unbiased**: on average, across infinitely many repetitions, it lands exactly on target. A biased estimator is one that is systematically too high or too low.

The **variance** of an estimator is how spread out its sampling distribution is:

$$\text{Var}(\hat{\theta}) = \mathbb{E}\left[(\hat{\theta} - \mathbb{E}[\hat{\theta}])^2\right].$$

Variance measures the *random* error — the noise that makes one sample's estimate differ from another's, regardless of where the center is. Its square root is the standard error.

These two are genuinely independent dials, and the cleanest way to feel that is a dartboard.

![Bias is the distance of the cluster from the bullseye and variance is the spread of the shots; they move independently.](/imgs/blogs/estimators-mle-bias-variance-quant-interviews-2.png)

Each panel is a hand throwing darts at a target whose center is the true parameter. **Bias** is how far the *cluster's center* sits from the bullseye — a consistent pull to one side. **Variance** is how *spread out* the darts are — the scatter within the cluster. The top-left thrower is the dream: low bias (centered) and low variance (tight). The bottom-right is the nightmare: a thrower who is both consistently off-target *and* all over the place. Crucially, the top-right thrower (centered but scattered) and the bottom-left thrower (tight but off-target) are *both* flawed, in different ways — and an estimator can have either flaw without the other.

Neither bias alone nor variance alone tells you how close your estimate will actually be. The number that does is the **mean squared error**:

$$\text{MSE}(\hat{\theta}) = \mathbb{E}\left[(\hat{\theta} - \theta)^2\right].$$

In words: take the squared distance from your estimate to the truth, and average it over all possible samples. It is the natural "how wrong, typically?" score, and squaring means a dart twice as far off counts four times as bad. The single most useful fact in this entire post is that MSE splits exactly:

$$\boxed{\;\text{MSE}(\hat{\theta}) = \text{Bias}(\hat{\theta})^2 + \text{Var}(\hat{\theta})\;}$$

There is no cross term. The proof is two lines of algebra — add and subtract $\mathbb{E}[\hat{\theta}]$ inside the square, expand, and the middle term vanishes because $\mathbb{E}[\hat{\theta} - \mathbb{E}[\hat{\theta}]] = 0$ — but the *consequence* is what matters. Your total expected error is your systematic miss (squared) plus your random scatter, and you can trade one against the other.

![Total error stacks systematic miss as bias-squared on top of random scatter as variance; a little bias can lower the total.](/imgs/blogs/estimators-mle-bias-variance-quant-interviews-3.png)

The figure makes the tradeoff concrete with three estimators of the same quantity, with error measured in squared dollars. The unbiased estimator on the left has zero bias but a variance of 9.0, for an MSE of 9.0. The noisy unbiased estimator on the right is *also* unbiased — zero bias — but its variance is 16.0, so its MSE is 16.0. Being unbiased did not save it. The shrunk estimator in the middle deliberately accepts a small bias (bias² = 1.0) in exchange for slashing variance to 4.0, giving an MSE of just 5.0 — the lowest of the three. This is the headline result we will keep returning to: *unbiasedness is not the goal; low total error is.* A little bias, bought cheaply, can be the best trade you make.

#### Worked example: bias and variance of a strategy's win-rate estimate

Let us put numbers on the dartboard with the interview question from the intro. You ran $n = 50$ trades; $X = 30$ won. You model each trade as an independent coin flip that comes up "win" with unknown probability $p$ — a *Bernoulli* trial, the simplest random event there is. Your estimator is the natural one:

$$\hat{p} = \frac{X}{n} = \frac{30}{50} = 0.60.$$

**Is it biased?** The number of wins $X$ in $n$ independent trials, each winning with probability $p$, follows a *binomial* distribution, whose mean is $\mathbb{E}[X] = np$. So

$$\mathbb{E}[\hat{p}] = \mathbb{E}\!\left[\frac{X}{n}\right] = \frac{np}{n} = p.$$

The expected value of $\hat{p}$ is exactly $p$. The estimator is **unbiased** — on average across many runs of 50 trades, it lands on the true win-rate. No systematic pull to either side.

**What is its variance?** A binomial $X$ has variance $\text{Var}(X) = np(1-p)$, so

$$\text{Var}(\hat{p}) = \frac{\text{Var}(X)}{n^2} = \frac{np(1-p)}{n^2} = \frac{p(1-p)}{n}.$$

We do not know the true $p$, so we plug in our estimate $\hat{p} = 0.60$:

$$\text{Var}(\hat{p}) \approx \frac{0.60 \times 0.40}{50} = \frac{0.24}{50} = 0.0048.$$

The **standard error** is the square root: $\sqrt{0.0048} \approx 0.069$, or about 6.9 percentage points. So the honest answer to the interviewer is: *"My best estimate of the win-rate is 60%, with a standard error of roughly 7 points — meaning the true rate could comfortably be anywhere from the low 50s to the high 60s."* A rough 95% confidence interval is $0.60 \pm 2 \times 0.069 = [0.46, 0.74]$. After 50 trades you cannot even rule out that this strategy is a coin flip.

> **The intuition:** the win-rate estimator is unbiased, but with only 50 trades its standard error is so large that "60%" and "50%" are statistically indistinguishable. The estimate is correct on average and nearly useless in isolation.

## Consistency: does more data eventually save you?

An estimator can be unbiased and still terrible if its variance is huge. The reassuring property we want is that the estimator *gets better* as data accumulates — that with enough observations, the estimate homes in on the truth. That property is **consistency**.

An estimator $\hat{\theta}_n$ (the subscript $n$ reminds us it depends on sample size) is **consistent** if it converges to the true value $\theta$ as $n \to \infty$. The technical phrase is "converges in probability": for any tiny tolerance you name, the probability that $\hat{\theta}_n$ lands outside that tolerance of $\theta$ goes to zero as the sample grows. A sufficient and easy-to-check condition: if an estimator's bias goes to zero *and* its variance goes to zero as $n \to \infty$, then by the MSE decomposition its MSE goes to zero, and it is consistent.

![As n grows the estimate's wandering band collapses onto the true value; consistency is about the limit, not small samples.](/imgs/blogs/estimators-mle-bias-variance-quant-interviews-11.png)

The picture shows one estimator's path as the sample size grows from $n = 10$ to $n = 1000$. Early on (left) the estimate swings wildly — small samples are noisy. The dotted lines are the $\pm 2$ standard-error band, and they form a funnel that collapses toward the dashed true value. By $n = 1000$ the band has nearly pinched shut: the estimate is locked onto the truth. That collapsing funnel *is* consistency. Note what the picture does *not* promise: it says nothing about small samples. A consistent estimator can be wildly wrong at $n = 10$; consistency is purely a statement about the limit.

The win-rate estimator $\hat{p}$ is consistent: its bias is zero and its variance $p(1-p)/n$ shrinks to zero as $n$ grows. Almost every sensible estimator you will meet is consistent — it is close to a minimum bar. The interesting questions are about the *rate* of convergence (how fast the funnel closes) and about small-sample behavior (how bad it is before the limit kicks in), which is exactly where bias and variance come back in.

## The sample mean and why its error shrinks like 1 / √n

The workhorse estimator is the **sample mean**. Given observations $x_1, x_2, \dots, x_n$, it is

$$\bar{x} = \frac{1}{n}\sum_{i=1}^{n} x_i.$$

If the data are draws from a distribution with true mean $\mu$ and true variance $\sigma^2$, the sample mean is unbiased — $\mathbb{E}[\bar{x}] = \mu$, because the expectation of an average is the average of the expectations — and its variance is

$$\text{Var}(\bar{x}) = \frac{\sigma^2}{n}.$$

That formula deserves a moment. The variance of a single observation is $\sigma^2$; the variance of the *average of $n$* observations is $\sigma^2 / n$, smaller by a factor of $n$. Taking the square root, the **standard error of the mean** is $\sigma / \sqrt{n}$. The square root is the whole story of why estimation in finance is hard.

![The estimate's spread is sigma over root-n: from n=10 to n=250 the standard error shrinks five-fold, not twenty-five-fold.](/imgs/blogs/estimators-mle-bias-variance-quant-interviews-5.png)

The figure overlays the sampling distribution of the mean for three sample sizes, all centered on the same truth of 1.0%. At $n = 10$ the distribution is wide and short — the estimate could land far from the truth. At $n = 250$ it is tall and tight. But look at the numbers: going from $n = 10$ to $n = 250$ is a 25-fold increase in data, yet the standard error only shrinks from 3.2% to 0.64% — a *five-fold* improvement, because $\sqrt{25} = 5$. To cut your uncertainty in half you need *four times* the data; to cut it by ten you need *a hundred times*. This brutal arithmetic — diminishing returns to data — is why a quant can estimate a high-frequency signal from millions of ticks but cannot pin down a stock's long-run mean return from decades of history.

#### Worked example: estimating a strategy's mean daily P&L and its precision

Suppose a strategy's daily profit-and-loss has a true daily mean of $\mu = \$200$ and a true daily standard deviation of $\sigma = \$2{,}000$ — a realistic ratio, where the daily noise dwarfs the daily edge by ten to one. You have one year of data, $n = 252$ trading days. How precisely can you estimate the mean daily P&L?

The standard error of the sample mean is

$$\text{SE}(\bar{x}) = \frac{\sigma}{\sqrt{n}} = \frac{\$2{,}000}{\sqrt{252}} = \frac{\$2{,}000}{15.87} \approx \$126.$$

So your estimate of the daily edge is $\$200 \pm \$126$ (one standard error). A rough 95% interval is $\$200 \pm 2 \times \$126 = [-\$52, \ \$452]$. After a *full year* of trading, you cannot even confidently say the strategy makes money — the interval includes zero. To shrink that $\$126$ standard error to a more comfortable $\$40$ (so the edge is clearly positive), you would need $(126/40)^2 \approx 10$ years of data, and that assumes the edge stays constant for a decade, which it never does.

> **The intuition:** because the standard error falls only as 1 / √n, distinguishing a real but small mean return from zero takes an enormous amount of data — which is precisely why mean returns are the hardest thing in finance to estimate.

## The sample variance, and the real reason you divide by n minus 1

You have surely seen the sample variance written with a strange denominator:

$$s^2 = \frac{1}{n-1}\sum_{i=1}^{n}(x_i - \bar{x})^2.$$

Why $n - 1$ and not $n$? The textbook answer — "it makes the estimator unbiased" — is correct but unsatisfying. Here is the *why*.

The thing you would love to compute is the average squared deviation from the *true* mean $\mu$: $\frac{1}{n}\sum (x_i - \mu)^2$. That would be unbiased for $\sigma^2$. But you do not know $\mu$, so you substitute the sample mean $\bar{x}$. And $\bar{x}$ is special: it is the value that *minimizes* the sum of squared deviations. By construction, the data sit closer to their own sample mean than to the true mean. So $\sum (x_i - \bar{x})^2$ is systematically *smaller* than $\sum (x_i - \mu)^2$ would have been. Dividing by $n$ therefore *underestimates* the true variance. Dividing by $n - 1$ instead inflates the result by just enough to correct the bias.

![The sample mean hugs the data, so deviations run small; dividing by n-1 instead of n fixes the downward bias on average.](/imgs/blogs/estimators-mle-bias-variance-quant-interviews-6.png)

The figure walks a tiny sample $\{2, 6, 10\}$ all the way through. The sample mean is 6; the deviations are $-4, 0, +4$; the sum of squared deviations is $16 + 0 + 16 = 32$. Divide by $n = 3$ and you get 10.67 — the biased estimate. Divide by $n - 1 = 2$ and you get 16.00 — the unbiased one. The amber box on the right names the intuition: you "spent" one degree of freedom computing the sample mean, leaving only $n - 1$ independent pieces of information about the spread. Once you fix the mean and $n - 1$ of the deviations, the last deviation is forced (they must sum to zero), so there are only $n - 1$ free deviations — and you average over that many.

#### Worked example: biased vs unbiased variance on a tiny sample

Make the bias visible. Take three daily returns, in percent: $-2\%, 0\%, +2\%$. The sample mean is $0\%$. The squared deviations are $4, 0, 4$, summing to $8$.

- **Biased (÷ n):** $s_{\text{biased}}^2 = 8 / 3 = 2.67$ (percent-squared). Volatility estimate $= \sqrt{2.67} = 1.63\%$.
- **Unbiased (÷ n−1):** $s^2 = 8 / 2 = 4.00$ (percent-squared). Volatility estimate $= \sqrt{4.00} = 2.00\%$.

The biased version reports a volatility of 1.63% versus the unbiased 2.00% — it understates the risk by nearly 20% on this tiny sample. With three data points the correction is enormous; with 252 daily returns, dividing by 251 instead of 252 barely moves the answer (a 0.2% change). The $n - 1$ correction matters precisely when data is scarce, which in finance is exactly when you are estimating something from a short window.

> **The intuition:** the sample mean hugs the data, so deviations from it are too small; dividing by n−1 inflates the estimate by just the right amount to remove the resulting downward bias.

#### Worked example: showing E[÷n variance] understates σ²

Why does dividing by $n$ give exactly $\frac{n-1}{n}\sigma^2$ rather than $\sigma^2$? Here is the clean algebra interviewers sometimes ask you to reproduce. Write the $\div n$ estimator as $\tilde{s}^2 = \frac{1}{n}\sum(x_i - \bar{x})^2$. The key identity is

$$\sum_{i=1}^n (x_i - \bar{x})^2 = \sum_{i=1}^n (x_i - \mu)^2 - n(\bar{x} - \mu)^2.$$

Take expectations of both sides. The first term: $\mathbb{E}\big[\sum (x_i - \mu)^2\big] = n\sigma^2$, since each term has expectation $\sigma^2$. The second term: $\mathbb{E}\big[n(\bar{x}-\mu)^2\big] = n \cdot \text{Var}(\bar{x}) = n \cdot \frac{\sigma^2}{n} = \sigma^2$. So

$$\mathbb{E}\Big[\sum (x_i - \bar{x})^2\Big] = n\sigma^2 - \sigma^2 = (n-1)\sigma^2.$$

Therefore $\mathbb{E}[\tilde{s}^2] = \frac{(n-1)\sigma^2}{n} = \frac{n-1}{n}\sigma^2$, which is *less* than $\sigma^2$ — a downward bias of exactly one part in $n$. Dividing instead by $n-1$ gives $\mathbb{E}[s^2] = \sigma^2$ exactly. Concretely, with $n = 2$ the $\div n$ estimator captures only half the true variance on average; with $n = 252$ it captures $251/252 = 99.6\%$.

> **The intuition:** the $\div n$ estimator is biased by the factor (n−1)/n because estimating the mean from the same data steals exactly one observation's worth of information about the spread.

## Maximum likelihood estimation: let the data vote

So far we have *guessed* good estimators (the sample mean, the sample fraction) and checked their properties. Maximum likelihood is a *machine* that manufactures estimators for you, and it is the single most important idea in this post for interviews. The principle is almost philosophically simple: **choose the parameter value that makes the data you actually observed as probable as possible.**

Here is the setup. You have a model with an unknown parameter $\theta$. For any candidate value of $\theta$, the model assigns a probability (or probability density) to your observed data. Viewed as a function of $\theta$ with the data held fixed, that quantity is the **likelihood**, written $L(\theta)$. The **maximum likelihood estimate** (MLE) is the value $\hat{\theta}$ that maximizes $L(\theta)$ — the parameter under which your data was the least surprising.

In practice we maximize the **log-likelihood** $\ell(\theta) = \log L(\theta)$ instead, for two reasons: the log turns the product over independent observations into a sum (far easier to differentiate), and it does not move the location of the maximum (log is increasing, so wherever $L$ peaks, $\ell$ peaks too). To find the peak, take the derivative — called the **score** — set it to zero, and solve.

![30 wins in 50 trades: the log-likelihood over p peaks exactly at p-hat equals 0.60, the value that makes the data most probable.](/imgs/blogs/estimators-mle-bias-variance-quant-interviews-4.png)

The figure shows the log-likelihood for the win-rate problem. Each value of $p$ on the horizontal axis gets a log-likelihood — how probable "30 wins in 50 trades" is under that $p$. The curve rises, peaks, and falls, and the peak sits exactly at $p = 0.60$. At the peak the slope is zero (the score equation), and the extreme values of $p$ near 0 or 1 are deeply implausible: if the true win-rate were 5%, seeing 30 wins would be astronomically unlikely, so the likelihood there is tiny. The MLE just reads off the top of the hill.

#### Worked example: deriving the MLE for a Bernoulli win-rate

Let us derive that peak rather than assert it. Each trade is Bernoulli$(p)$: it wins (value 1) with probability $p$, loses (value 0) with probability $1 - p$. With $X = 30$ wins out of $n = 50$ independent trades, the probability of the exact observed sequence is $p^{X}(1-p)^{n-X}$, so the likelihood is

$$L(p) = p^{X}(1-p)^{n-X} = p^{30}(1-p)^{20}.$$

Take the log:

$$\ell(p) = X\log p + (n - X)\log(1 - p) = 30\log p + 20\log(1-p).$$

Differentiate with respect to $p$ and set to zero (the score equation):

$$\ell'(p) = \frac{X}{p} - \frac{n - X}{1 - p} = 0.$$

Multiply through and solve: $X(1-p) = (n-X)p \implies X - Xp = np - Xp \implies X = np$, so

$$\hat{p} = \frac{X}{n} = \frac{30}{50} = 0.60.$$

The MLE of a win-rate is simply wins over trades — exactly the intuitive estimator, now derived from a principle rather than guessed. And there is a bonus: maximum likelihood theory says that for large $n$, the MLE's variance is approximately $1 / I(\hat{\theta})$, where $I$ is the *Fisher information* (more on this at the Cramer-Rao section). For the Bernoulli, this reproduces $\text{Var}(\hat{p}) \approx \hat{p}(1-\hat{p})/n = 0.24/50 = 0.0048$, giving the same standard error of 6.9 percentage points we found before. The MLE hands you the estimate *and* its uncertainty from one calculation.

#### Worked example: deriving the MLE for a Normal mean and variance

The other MLE every quant must be able to derive on a whiteboard is the Normal (Gaussian) distribution, the workhorse model for returns. Suppose returns $x_1, \dots, x_n$ are independent draws from a Normal with unknown mean $\mu$ and unknown variance $\sigma^2$. The density of one observation is $\frac{1}{\sqrt{2\pi\sigma^2}}\exp\!\big(-\frac{(x_i - \mu)^2}{2\sigma^2}\big)$. The log-likelihood for the whole sample is

$$\ell(\mu, \sigma^2) = -\frac{n}{2}\log(2\pi) - \frac{n}{2}\log\sigma^2 - \frac{1}{2\sigma^2}\sum_{i=1}^n (x_i - \mu)^2.$$

**Maximize over $\mu$.** Only the last term involves $\mu$. Differentiating and setting to zero: $\frac{1}{\sigma^2}\sum(x_i - \mu) = 0$, which forces $\sum x_i = n\mu$, so

$$\hat{\mu} = \frac{1}{n}\sum_{i=1}^n x_i = \bar{x}.$$

The MLE of the mean is the sample mean. **Now maximize over $\sigma^2$.** Differentiating $\ell$ with respect to $\sigma^2$ and setting to zero gives $-\frac{n}{2\sigma^2} + \frac{1}{2\sigma^4}\sum(x_i - \mu)^2 = 0$, which solves to

$$\hat{\sigma}^2 = \frac{1}{n}\sum_{i=1}^n (x_i - \hat{\mu})^2.$$

Look closely at that denominator: it is $n$, not $n - 1$. **The maximum likelihood estimate of the variance is the biased one.** This is a favorite interview "gotcha": MLE is not guaranteed to be unbiased, and here it is not — it divides by $n$ and underestimates $\sigma^2$ by the factor $(n-1)/n$ we derived earlier. The unbiased sample variance $s^2$ (with $n-1$) is a *correction* applied to the MLE, not the MLE itself. For large $n$ the difference is negligible, which is why nobody loses sleep over it in practice — but in an interview, knowing that the Normal-variance MLE uses $n$ and is biased marks you as someone who actually did the derivation.

## The method of moments: match theory to data

Maximum likelihood is powerful but sometimes the likelihood is a horror to maximize, or you do not want to commit to a full distribution. The **method of moments** is an older, simpler recipe that often gives the same answer with far less work. The idea: a distribution's *moments* (its mean, its mean-of-squares, and so on) are functions of its parameters. Write those functions, set each theoretical moment equal to the corresponding moment measured in your data, and solve the resulting equations for the parameters.

![Write each moment as a formula in the unknowns, set it equal to the same moment in the data, and solve, with no calculus needed.](/imgs/blogs/estimators-mle-bias-variance-quant-interviews-8.png)

The first moment is just the mean: $\mathbb{E}[X] = \mu$. The second moment is the mean of squares: $\mathbb{E}[X^2] = \mu^2 + \sigma^2$ (because variance equals mean-of-squares minus square-of-mean). On the data side you compute the sample mean and the sample mean-of-squares. Setting theory equal to data gives two equations in two unknowns, which you solve algebraically. No derivatives, no likelihood — just bookkeeping.

#### Worked example: method-of-moments for mean and variance of returns

Suppose you observe a batch of daily returns whose sample mean is $1.0\%$ and whose sample mean-of-squares is $4.0$ (in percent-squared). Match moments:

- **First moment:** $\hat{\mu} = $ sample mean $= 1.0\%$.
- **Second moment:** $\hat{\mu}^2 + \hat{\sigma}^2 = $ sample mean-of-squares $= 4.0$, so $\hat{\sigma}^2 = 4.0 - (1.0)^2 = 3.0$ (percent-squared), giving $\hat{\sigma} = \sqrt{3.0} = 1.73\%$.

For the Normal distribution the method of moments gives the *same* answers as maximum likelihood — the sample mean and the (÷ n) sample variance. That coincidence is special to the Normal. Where the two methods diverge, the MLE is usually more efficient (lower variance), but the method of moments is a lifesaver when the likelihood has no closed form, and it is a great first move in an interview when you are asked to "estimate the parameters" of an unfamiliar distribution: write down its mean and variance in terms of the parameters, match to the data, solve.

> **The intuition:** the method of moments turns estimation into algebra — express the distribution's moments in terms of its parameters, plug in the sample moments, and invert.

## The bias-variance tradeoff: buying accuracy with bias

We now return to the central theme and make it actionable. The MSE decomposition told us total error is bias² + variance. The **bias-variance tradeoff** is the empirical fact that these two quantities usually move in *opposite* directions as you change how flexible or aggressive your estimator is. Push for lower bias — fit the data more closely, use more parameters, trust the sample more — and variance tends to rise. Accept more bias — simplify, shrink toward a prior, regularize — and variance tends to fall. The art is finding the bottom of the resulting U-curve.

![More complexity cuts bias but raises variance; their sum bottoms out at an intermediate sweet spot, not at maximum flexibility.](/imgs/blogs/estimators-mle-bias-variance-quant-interviews-7.png)

The figure plots all three curves against model complexity. As complexity rises (moving right), bias² falls — a more flexible model can capture more structure. But variance rises — a more flexible model also chases noise, so its estimates swing more from sample to sample. The total MSE, their sum, is the blue U-curve. It does *not* bottom out at maximum flexibility; it bottoms out at an intermediate sweet spot. To the left of the sweet spot you are *underfitting* (too much bias, the model is too crude); to the right you are *overfitting* (too much variance, the model memorizes noise). Every regularization technique in statistics and machine learning is a knob for sliding along this curve toward the bottom.

The most important tool for deliberately trading bias for variance is **shrinkage** (the statistician's word) or **regularization** (the machine-learning word). The idea: pull your raw estimate toward some simpler, more stable target — zero, the grand average, a structured model — by a shrinkage factor. You introduce bias (you are no longer centered on the truth) but you cut variance (you no longer swing as wildly). If the variance reduction outweighs the bias you added, your MSE falls.

#### Worked example: comparing the MSE of an unbiased mean vs a shrunk mean, in dollars

Here is the calculation that makes the tradeoff undeniable, framed as an interviewer might. You are estimating the true mean daily edge $\mu$ of a strategy. Two estimators are on the table.

**Estimator A — the unbiased sample mean.** Suppose, from the data you have, its sampling distribution has bias $0$ and standard error $\$100$, so its variance is $\$100^2 = \$10{,}000$ (in squared dollars). Then

$$\text{MSE}_A = \text{bias}^2 + \text{variance} = 0 + 10{,}000 = \$10{,}000.$$

**Estimator B — a shrunk mean.** You suspect the true edge is small, so you shrink the sample mean halfway toward zero: $\hat{\mu}_B = 0.5\,\bar{x}$. Shrinking by half multiplies the standard error by half (variance by a quarter): the new variance is $0.25 \times 10{,}000 = \$2{,}500$. But shrinking introduces bias. If the true edge is, say, $\mu = \$60$, then $\mathbb{E}[\hat{\mu}_B] = 0.5 \times 60 = \$30$, so the bias is $30 - 60 = -\$30$ and bias² $= \$900$. Then

$$\text{MSE}_B = \text{bias}^2 + \text{variance} = 900 + 2{,}500 = \$3{,}400.$$

Estimator B has *less than half* the MSE of the unbiased Estimator A, despite being biased — because the $\$7{,}500$ it saved in variance dwarfs the $\$900$ of bias it added. The shrunk estimate is, on average, *closer to the truth* than the unbiased one. This is not a trick: when the underlying signal is weak relative to the noise (the usual case for mean returns), shrinking toward zero is almost always the better trade. The one caveat: if the true edge were large — say $\mu = \$300$ — the bias² would balloon to $(150)^2 = \$22{,}500$ and shrinking would *hurt*. Shrinkage helps exactly when the thing you are estimating is genuinely small, which is why it is so well suited to financial means.

> **The intuition:** deliberately biasing an estimator toward a sensible target lowers its total error whenever the variance you save exceeds the bias² you pay — and for weak signals buried in noise, you usually save a lot and pay a little.

## The Cramer-Rao lower bound: a floor on how good you can be

We have seen that you can lower variance by accepting bias. But what if you insist on staying unbiased — how low can the variance go? The astonishing answer is that there is a *hard floor*, set by the data itself, that no unbiased estimator can beat. That floor is the **Cramer-Rao lower bound** (CRLB).

The bound is stated in terms of the **Fisher information** $I(\theta)$, which measures how much the log-likelihood "curves" at its peak — how sharply the data distinguishes nearby parameter values. A sharply peaked likelihood (lots of curvature) means the data is very informative about $\theta$; a flat likelihood means it is not. Formally, $I(\theta)$ is the expected squared score, or equivalently minus the expected second derivative of the log-likelihood. The Cramer-Rao bound says: for *any* unbiased estimator $\hat{\theta}$,

$$\text{Var}(\hat{\theta}) \ge \frac{1}{I(\theta)}.$$

That is it. The reciprocal of the Fisher information is a wall that no unbiased estimator can get under.

![A hard floor: an unbiased estimator can sit on it as efficient or above it as wasteful, but nothing unbiased lives below.](/imgs/blogs/estimators-mle-bias-variance-quant-interviews-9.png)

The picture makes it physical. The Cramer-Rao bound is a floor. An unbiased estimator can *sit on* the floor — its variance equals the bound — in which case we call it **efficient**: it squeezes every drop of information out of the data. Or it can float *above* the floor, wasting information (for example, an estimator that ignores half your data has roughly double the variance it needs). What no unbiased estimator can ever do is reach the red forbidden region *below* the floor. The bound is why efficiency is a meaningful word: it is not "low variance in the abstract," it is "variance as low as the laws of probability permit for an unbiased estimator."

Two facts make the CRLB matter in interviews. First, the maximum likelihood estimator is **asymptotically efficient**: as $n \to \infty$, the MLE's variance converges to the Cramer-Rao floor. This is the deep reason MLE is the default — for large samples, you cannot do better with any unbiased estimator. Second, the bound gives you a quick way to *check* whether an estimator is any good: compute the Fisher information, get the floor, and see how close your estimator's variance is. If it is far above the floor, you are leaving information on the table.

#### Worked example: the Cramer-Rao floor for a win-rate estimator

Let us verify that the win-rate estimator $\hat{p}$ is efficient — that it actually sits on the floor. For a single Bernoulli$(p)$ observation, the Fisher information works out to $I_1(p) = \frac{1}{p(1-p)}$. (You get this by differentiating the log-likelihood twice and taking the expectation; the curvature of $\log p$ and $\log(1-p)$ combines to exactly that.) Fisher information adds across independent observations, so for $n$ trials $I_n(p) = \frac{n}{p(1-p)}$. The Cramer-Rao floor is therefore

$$\text{Var}(\hat{p}) \ge \frac{1}{I_n(p)} = \frac{p(1-p)}{n}.$$

But we computed earlier that $\hat{p} = X/n$ has variance *exactly* $\frac{p(1-p)}{n}$. The win-rate estimator's variance equals the Cramer-Rao bound — it is **efficient**, sitting right on the floor. No unbiased estimator of a win-rate can do better than wins-over-trades. With $p = 0.60$ and $n = 50$, the floor is $0.24/50 = 0.0048$, the same standard error of 6.9 percentage points yet again — now revealed as not just *an* answer but the *best possible* answer for an unbiased estimator.

> **The intuition:** the Cramer-Rao bound says the simple win-rate estimator is not just reasonable but optimal among unbiased estimators — its variance hits the theoretical floor, so there is no cleverer unbiased recipe to find.

## Which estimator is better? A decision procedure

Interviewers love to hand you two estimators and ask which to use. The wrong instinct — drilled in by intro stats courses — is to reach for "the unbiased one." The right instinct is to compare *total error*. Here is the decision procedure.

![Choosing an estimator means comparing total mean squared error, not just checking whether it is unbiased.](/imgs/blogs/estimators-mle-bias-variance-quant-interviews-10.png)

Walk the flow. First ask whether either estimator is biased. If *both* are unbiased, the comparison is easy: pick the one with lower variance (equivalently, the more efficient one). If *either* is biased, you cannot shortcut — you must compute the full MSE = bias² + variance for each and pick the lower total. Finally, sanity-check against the Cramer-Rao floor: if your chosen estimator's variance is near the bound, it is about as good as an unbiased estimator can get, and you can ship it. The dashed return path is the reminder that this is the criterion you loop back to whenever a new estimator is proposed.

The deepest point the flow encodes is the one beginners resist: **a biased estimator can be strictly better than every unbiased estimator.** Our shrunk-mean example had lower MSE than the unbiased mean. In fact, a celebrated result (the James-Stein phenomenon) shows that when you estimate three or more means at once, the obvious unbiased estimator is *inadmissible* — a particular shrinkage estimator beats it for every possible true value. Unbiasedness feels virtuous, but it is a constraint, and constraints cost you. The quant's job is to minimize error, not to honor a constraint that was never the goal.

## In the interview room: five fully-solved problems

The technique sections above each closed with a worked example. Here are five more, in the compressed back-and-forth style of an actual interview, to drill the reflexes: name the estimator, check bias, compute variance, attach a standard error, and — when asked — pick the lower MSE.

#### Worked example: pooling two desks' win-rate estimates

*"Desk A reports a 55% win-rate from 200 trades; Desk B reports 65% from 50 trades. What's your best combined estimate of the shared true win-rate?"*

The naive answer — average them to 60% — is wrong, because it ignores that Desk A's estimate is far more precise (four times the data). The right move is to pool the *raw counts*, which is the MLE on the combined sample. Desk A won $0.55 \times 200 = 110$; Desk B won $0.65 \times 50 = 32.5$, call it 33. Combined: $143$ wins in $250$ trades, so

$$\hat{p} = \frac{143}{250} = 0.572.$$

The combined estimate is 57.2%, pulled toward Desk A's value because A carries more information. Its standard error is $\sqrt{0.572 \times 0.428 / 250} = \sqrt{0.000979} \approx 0.031$, or 3.1 points — tighter than either desk alone. The general principle: when combining unbiased estimates, weight each by its precision (inverse variance), which for counts just means pooling the raw data. Equal-weighting throws away the fact that one estimate is sharper.

> **The intuition:** combine estimates by precision, not by simple averaging — pooling the raw counts automatically gives the more-informative sample its proper, larger weight.

#### Worked example: would you rather have a biased ruler or a noisy one?

*"Estimator A is unbiased with standard error $\$50$. Estimator B is biased by $\$20$ but has standard error $\$30$. Which has lower MSE?"*

Compute both. Estimator A: $\text{MSE}_A = 0^2 + 50^2 = \$2{,}500$. Estimator B: $\text{MSE}_B = 20^2 + 30^2 = 400 + 900 = \$1{,}300$. Estimator B wins, with barely over half the MSE, despite its bias — the $\$1{,}600$ it saved in variance (from $2{,}500$ down to $900$) swamps the $\$400$ of bias² it added. The follow-up the interviewer wants: *at what bias would B stop being worth it?* Set $b^2 + 900 = 2{,}500$, so $b^2 = 1{,}600$, $b = \$40$. As long as B's bias stays under $\$40$, the trade is favorable. This break-even calculation is the whole bias-variance tradeoff in one line.

> **The intuition:** always compute MSE before choosing; a biased estimator beats an unbiased one whenever its bias² stays below the variance it saves.

#### Worked example: how many trades to call an edge real?

*"You believe a strategy wins 55% of the time. How many trades until you can distinguish that from a coin flip with reasonable confidence?"*

"Reasonable confidence" means the gap between 55% and 50% should be at least two standard errors — so the confidence interval clears 50%. The standard error of $\hat{p}$ is $\sqrt{p(1-p)/n} \approx \sqrt{0.25/n}$ (using $p \approx 0.5$). We need $0.05 \ge 2 \times \sqrt{0.25/n}$, i.e. $0.025 \ge \sqrt{0.25/n}$, so $0.000625 \ge 0.25/n$, giving $n \ge 400$. You need roughly 400 trades to be reasonably sure a 55% edge is not luck. Halve the edge to 52.5% and the requirement quadruples to about 1,600 trades — because the needed $n$ scales as $1/(\text{edge})^2$. This is why small edges demand vast sample sizes to validate, and why a strategy with a thin but real edge can look indistinguishable from noise for a painfully long time.

> **The intuition:** the trades needed to confirm an edge scale as one over the edge squared, so halving your edge quadruples the data you need to prove it exists.

#### Worked example: estimating volatility from a sample of returns

*"Here are five daily returns: −1%, +2%, 0%, +1%, −2%. Estimate the volatility and give a feel for its precision."*

Step one, the sample mean: $(-1 + 2 + 0 + 1 - 2)/5 = 0/5 = 0\%$. Step two, squared deviations from the mean (which is 0): $1, 4, 0, 1, 4$, summing to $10$. Step three, the unbiased sample variance divides by $n - 1 = 4$: $s^2 = 10/4 = 2.5$ (percent-squared). Step four, the volatility estimate is $s = \sqrt{2.5} = 1.58\%$ per day. If you wanted an *annualized* volatility, multiply by $\sqrt{252} \approx 15.87$: $1.58\% \times 15.87 \approx 25\%$ per year. The precision caveat the interviewer is fishing for: with only five observations, this volatility estimate is itself extremely noisy — the sampling distribution of a variance estimated from five points is wide and skewed, so "25%" here might really be anywhere from the high teens to the high 30s. Volatility estimated from a handful of points is a guess dressed up as a number.

> **The intuition:** volatility from a tiny sample is a real estimate with a large standard error of its own — report the number, but never trust its last digit.

#### Worked example: the MLE of a uniform's upper bound

*"Order sizes arrive uniformly between 0 and some unknown maximum θ. You observe sizes 3, 7, 5, 8, 6. Estimate θ by maximum likelihood, and is it biased?"*

This one breaks the "set the derivative to zero" reflex, which is exactly why it is asked. The likelihood of observing $n$ points from Uniform$(0, \theta)$ is $1/\theta^n$ *if* $\theta$ is at least as large as the biggest observation, and zero otherwise (an impossible-to-observe value cannot have been drawn). Since $1/\theta^n$ *decreases* in $\theta$, the likelihood is maximized by making $\theta$ as small as it is allowed to be — which is exactly the largest observation. So

$$\hat{\theta}_{\text{MLE}} = \max(x_i) = 8.$$

**Is it biased?** Yes, and obviously so: the maximum of your sample can never exceed the true $\theta$, so $\hat{\theta} = \max(x_i) \le \theta$ always, and on average it falls short — its expectation is $\frac{n}{n+1}\theta$, a downward bias. With $n = 5$, the MLE recovers only $5/6 \approx 83\%$ of $\theta$ on average. The unbiased fix multiplies by $\frac{n+1}{n} = \frac{6}{5}$: $\hat{\theta}_{\text{unbiased}} = \frac{6}{5} \times 8 = 9.6$. The lesson: the MLE is found at a *boundary*, not by calculus, and it is biased — two facts that together make this a classic screening question.

> **The intuition:** when the parameter sets the support of the distribution, the MLE lives at the edge of the data and is biased toward the inside — calculus does not apply and unbiasedness is not free.

## Common misconceptions

**"Unbiased means accurate."** No. Unbiased means *correct on average across infinitely many samples* — it says nothing about any single estimate. A wildly noisy unbiased estimator (the top-right dartboard, or the $\$16{,}000$-MSE estimator in our decomposition figure) can be far less accurate than a slightly biased, low-variance one. Accuracy is about MSE, not bias.

**"More data always helps a lot."** More data helps, but with crushing diminishing returns: the standard error of a mean falls as $1/\sqrt{n}$, so quadrupling your data only halves your uncertainty. Worse, in finance "more data" often means going further back in time to a regime that no longer holds, so the extra data is biased even as it reduces variance. The marginal value of the next year of history can be near zero or negative.

**"The MLE is unbiased."** Frequently false. The MLE of a Normal variance divides by $n$, not $n-1$, and is biased low. The MLE is *asymptotically* unbiased (the bias vanishes as $n \to \infty$) and asymptotically efficient, but for small samples it can be meaningfully biased. Never assume "MLE" implies "unbiased."

**"Standard deviation and standard error are the same thing."** They are not, and confusing them is the fastest way to lose an interviewer's confidence. The standard *deviation* describes the spread of the *data* (how much individual returns vary). The standard *error* describes the spread of your *estimate* (how much your computed mean would vary if you redid the experiment). The standard error of the mean is the data's standard deviation divided by $\sqrt{n}$ — they differ by a factor that grows with sample size.

**"A confidence interval has a 95% chance of containing the true value."** In the standard (frequentist) framework, the true parameter is a fixed number, not random — so for any *particular* interval you computed, it either contains the truth or it does not; there is no "95% chance" about that one interval. The correct statement is about the *procedure*: if you repeated the whole experiment many times, 95% of the intervals you would construct would contain the truth. Subtle, and exactly the kind of precision quant interviewers reward.

**"Variance is divided by n−1, so I should always use n−1."** The $n-1$ correction makes the *variance* estimator unbiased, but the resulting *volatility* estimator (its square root) is still biased, because the square root of an unbiased estimate is not an unbiased estimate of the square root (a consequence of Jensen's inequality). For most purposes the bias is tiny and ignored, but "should I divide by $n$ or $n-1$?" has the honest answer "it depends on what you are optimizing — unbiasedness of the variance wants $n-1$; the MLE and minimum MSE want $n$; the volatility is biased either way."

## How it shows up on a real trading desk

**Estimating volatility.** Volatility — the typical size of returns — is the most-estimated quantity in finance, and the bias-variance tradeoff governs every choice. Use a short window (say 20 days) and your estimate is nearly unbiased for *current* volatility but extremely noisy (high variance) — it lurches around. Use a long window (say 250 days) and your estimate is stable (low variance) but biased, because it averages in stale regimes and lags real changes. Practitioners resolve this with *exponentially weighted* estimators (RiskMetrics-style), which weight recent data more heavily — a tunable knob that slides along the bias-variance curve. The same tension reappears in GARCH models and in choosing how to blend realized and implied volatility. Every volatility number on a desk is a point chosen on a U-curve.

**Why mean returns are nearly impossible to estimate.** We saw it in the worked example: with a daily Sharpe-like ratio of edge to noise around 1-to-10, even a full year of data leaves the mean indistinguishable from zero. The $1/\sqrt{n}$ law is merciless for means. This is why serious quant shops largely *give up* on estimating expected returns from history and instead lean on shrinkage (pulling all estimated means toward a common value), economic priors, or signals with vastly higher data rates (intraday, cross-sectional). The classic paper here is by Robert Merton (1980), who showed that estimating the mean requires a *long calendar span* — and no amount of high-frequency sampling within a fixed span helps the mean, though it helps the variance enormously. It is one of the most important asymmetries in finance: you can know a stock's volatility well and its mean return barely at all.

**Shrinkage in risk models.** Building a covariance matrix for a portfolio of, say, 500 stocks from 252 days of returns is a textbook bias-variance disaster. The raw *sample* covariance matrix is unbiased but catastrophically noisy — you are estimating roughly 125,000 distinct entries from far too little data, and the matrix is near-singular (more assets than days), so a portfolio optimizer fed this matrix produces absurd, noise-chasing weights.

![Shrinking the noisy sample covariance toward a structured target trades a little bias for a large drop in estimation variance.](/imgs/blogs/estimators-mle-bias-variance-quant-interviews-12.png)

The fix, due to Olivier Ledoit and Michael Wolf (2004), is exactly the shrinkage idea from this post applied to a whole matrix: pull the noisy sample covariance toward a simple, structured target (such as a constant-correlation matrix or a single-factor model). You introduce bias — the shrunk matrix is no longer centered on the true covariance — but you slash variance, and the result is always invertible, stable, and produces sane portfolio weights. The before-and-after is the bias-variance tradeoff made operational: a little bias buys a large reduction in estimation noise, and the total error falls. Ledoit-Wolf shrinkage is now standard in production risk systems precisely because the unbiased estimator, here, is unusable.

**The win-rate question, revisited as risk control.** Recall that after 50 trades, a 60% win-rate had a standard error of 7 points, with a confidence interval stretching down to the high 40s. On a desk this is not academic: if you size positions assuming a 60% edge that might really be 50%, you will over-bet and blow through your risk budget when the true edge is smaller than you thought. This is the bridge to position sizing — see [the Kelly criterion and sequential betting](/blog/trading/quantitative-finance/kelly-criterion-sequential-betting-quant-interviews), where the *uncertainty* in your estimated edge, not just the point estimate, should shrink how aggressively you bet (the practical answer is to bet a fraction of full Kelly precisely because the edge is estimated, not known).

**Backtest overfitting as variance masquerading as bias.** When a researcher tries 200 strategy variants and reports the best one's backtested Sharpe ratio, that maximum is a wildly biased estimate of the chosen strategy's true Sharpe — the selection itself is an estimation procedure with enormous upward bias. The cure is the same family of ideas: penalize complexity (regularize), hold out data, and shrink the reported performance toward a skeptical prior. The bias-variance lens explains why the strategy that looked best in-sample (lowest apparent error) so often performs worst out-of-sample (it was overfit, sitting on the high-variance right side of the U-curve).

## When this matters to you and where to go next

Estimation is the quiet foundation under every confident-sounding number in quantitative finance. The moment you internalize that a reported win-rate, volatility, or expected return is a *random* draw from a sampling distribution — not a fact — you start asking the right second question every time: *how wide is that distribution, and have I traded bias for variance wisely?* That habit is exactly what interviewers at Jane Street, Two Sigma, Citadel, and DE Shaw are listening for. They do not want "60%"; they want "60%, standard error 7 points, and here is why I might deliberately shrink it."

For the probability machinery underneath all of this, work through [expected-value techniques](/blog/trading/quantitative-finance/expected-value-techniques-quant-interviews) (expectation and linearity are the engine behind every bias and variance calculation here) and [conditional probability and Bayes](/blog/trading/quantitative-finance/conditional-probability-bayes-quant-interviews) (the Bayesian view recasts shrinkage as a prior, and makes the whole bias-variance story feel inevitable). To see estimation turn into decisions, read [decision-making under uncertainty](/blog/trading/quantitative-finance/decision-making-under-uncertainty-quant-interviews) and the [Kelly criterion](/blog/trading/quantitative-finance/kelly-criterion-sequential-betting-quant-interviews). For the back-of-envelope skill of guessing magnitudes before you can measure them, [Fermi estimation](/blog/trading/quantitative-finance/estimation-fermi-problems-quant-interviews) is the perfect complement. And when you are ready to watch volatility estimation collide with options pricing, the [volatility surface](/blog/trading/quantitative-finance/volatility-surface) is where the noisy estimates of this post become the tradable objects of a derivatives desk.

This is educational material, not investment advice. But the core lesson transfers to any number you will ever be handed: ask not just *what is the estimate*, but *how was it estimated, and how much should I trust it.*
