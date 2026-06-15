---
title: "The probability distributions that markets actually use"
date: "2026-06-15"
description: "A beginner-friendly, build-from-zero tour of the six probability distributions quants reach for every day — Normal, lognormal, Student-t, Poisson, exponential, and Bernoulli — with worked dollar examples for prices, fat tails, trade counts, and waiting times."
tags: ["probability-distributions", "normal-distribution", "lognormal", "student-t", "fat-tails", "poisson", "exponential", "quant-finance", "risk-management", "math-for-quants"]
category: "trading"
subcategory: "Quantitative Finance"
author: "Hiep Tran"
featured: true
readTime: 40
---

> [!important]
> **TL;DR** — A handful of probability distributions do almost all the work in quant trading, and knowing which one fits which job — and where each one lies to you — is half the battle.
>
> - The **Normal** (bell curve) is the building block for *returns*: simple, symmetric, and convenient, but it badly underestimates how often markets crash.
> - The **lognormal** is the distribution of *prices*, because prices compound and can never go below zero; under the standard model the expected price is $E[S_T] = S_0\,e^{\mu T}$.
> - The **Student-t** fixes the Normal's worst flaw — *fat tails* — and a single parameter (degrees of freedom) dials how heavy those tails are.
> - **Poisson** counts *events* in a window (trades, jumps, defaults) at rate $\lambda$; the **exponential** measures the *waiting time* between them and is famously memoryless.
> - The one number to remember: switching a daily-return model from Normal to Student-t can multiply the modeled odds of a 5% crash day by roughly **46×** — same volatility, wildly different tail.

Here is a question that quietly decides whether a trading firm survives a bad week: *how often does a "once-in-a-lifetime" market crash actually happen?*

If you model daily stock returns with the famous bell curve — the Normal distribution every statistics class starts with — the math says a 5% down day should occur about once every 16,700 trading days. That is once every **66 years**. And yet the S&P 500 has had dozens of days worse than that in living memory: October 1987, the 2008 financial crisis, March 2020. The bell curve is not a little bit wrong about crashes. It is wrong by orders of magnitude. The entire discipline of quant risk management exists, in large part, because the most convenient distribution in mathematics tells comforting lies about the tails.

This post is a tour of the small set of probability distributions that quants actually use — and, just as important, where each one breaks. We will build every one from zero, attach it to a concrete market quantity (a price, a return, a trade count, a waiting time), and ground it in a worked example with real dollar figures. By the end you will know which distribution to reach for, what its parameters mean, and which lie it is telling you.

![Matrix of six distributions mapped to what each one models and its key parameter](/imgs/blogs/probability-distributions-for-markets-math-for-quants-1.png)

The table above is the map for the whole post: six distributions, the one thing each one models, and the single parameter that controls it. The Normal gives you returns, the lognormal gives you prices, the Student-t gives you fat tails, Poisson counts events, the exponential times the gaps between them, and Bernoulli is the humble win-or-loss coin flip behind every hit rate. We will visit each one in turn. Let us start from absolute zero.

## Foundations: the building blocks of probability

Before we can talk about *which* distribution fits the market, we need to agree on what a distribution even is. We will define each term the first time it appears, build the simplest possible version of every idea, and only then climb toward the real machinery. A practitioner can skim; a beginner can follow every step.

### What is a "random variable"?

A **random variable** is just a number whose value is uncertain before it is revealed. Tomorrow's return on a stock is a random variable: you do not know it today, but you know it will be *some* number. The roll of a die is a random variable. The number of trades that will print in the next minute is a random variable. We write random variables with capital letters — $X$, $S$, $N$ — and a specific observed value with a lowercase letter.

That is the whole idea. The rest of probability is about describing the *pattern* of values a random variable tends to take. That pattern is its distribution.

### What is a "probability distribution"?

A **probability distribution** is the complete rulebook for a random variable: it tells you how likely each possible value (or range of values) is. There are two flavors, and the difference matters constantly in markets.

A **discrete** distribution applies to things you *count* — whole numbers, like the number of trades in a minute or the number of defaults in a portfolio this year. For a discrete variable we use a **probability mass function** (PMF), written $P(X = k)$: the literal probability that the variable equals a specific whole number $k$. All the masses add up to 1, because *something* must happen.

A **continuous** distribution applies to things you *measure* on a smooth scale — a return, a price, a waiting time. For a continuous variable, the probability of hitting any *exact* value is zero (what is the probability tomorrow's return is exactly 0.7314158…%?). So instead we use a **probability density function** (PDF), written $f(x)$. Density is not probability itself; it is probability *per unit of x*. To get an actual probability you integrate the density over a range:

$$ P(a \le X \le b) = \int_a^b f(x)\,dx. $$

Here $f(x)$ is the height of the density curve at $x$, and the integral is the *area under the curve* between $a$ and $b$. The total area under any PDF is exactly 1. The everyday analogy: density is like the population *density* of a country (people per square mile), and probability is the actual *number of people* in a region — you get it by multiplying density by area.

### What is the "cumulative distribution function"?

The **cumulative distribution function** (CDF), written $F(x)$, answers one question: *what is the probability the variable comes in at or below $x$?*

$$ F(x) = P(X \le x). $$

The CDF is the workhorse of risk math, because almost every risk question is really a tail question: "What is the probability of a loss worse than 5%?" is just $F(-0.05)$. The CDF climbs from 0 (nothing is below $-\infty$) to 1 (everything is below $+\infty$). When we ask how often a 5% crash day happens, we are reading one number off a CDF.

### What is the "expected value" and the "variance"?

Two summary numbers describe most of a distribution's shape.

The **expected value** (or **mean**), written $E[X]$ or $\mu$ ("mu"), is the long-run average — the center of gravity of the outcomes. For a discrete variable it is the sum of each outcome times its probability; for a continuous one it is $E[X] = \int x\,f(x)\,dx$.

The **variance**, written $\mathrm{Var}(X)$ or $\sigma^2$ ("sigma squared"), measures how spread out the values are around the mean:

$$ \mathrm{Var}(X) = E\big[(X - \mu)^2\big]. $$

We square the distance from the mean so that being far below hurts as much as being far above and the deviations do not cancel. The square root of variance, $\sigma$, is the **standard deviation**, and in finance the standard deviation of returns has its own name: **volatility**. When a trader says "this stock has 20% vol," they mean its annual returns have a standard deviation of 20%.

### What are "skewness" and "kurtosis"?

Mean and variance are the first two *moments* of a distribution — the first describes its location, the second its spread. But two distributions can share the same mean and variance and still look completely different in shape. Two more numbers capture that difference, and both turn out to matter enormously for markets.

**Skewness** measures *asymmetry*. A distribution with zero skew (like the Normal) is a mirror image around its mean: a move 3% up is exactly as likely as a move 3% down. A distribution with *negative* skew has a longer left tail — big down moves are bigger and more frequent than big up moves. Stock returns are negatively skewed, because crashes are violent and sudden while rallies tend to be slower grinds. The market "takes the stairs up and the elevator down." Formally, skewness is the third standardized moment, $E[(X-\mu)^3]/\sigma^3$, but the only thing you need to feel is the asymmetry it names.

**Kurtosis** measures *tail heaviness* — how much probability lives in the extremes relative to a Normal. The Normal has a kurtosis of exactly 3 (by convention we often subtract 3 and talk about *excess kurtosis*, so the Normal sits at 0). A distribution with *positive* excess kurtosis has **fat tails**: more weight in the extremes, more frequent surprises. Daily equity returns have large positive excess kurtosis — often 5, 10, or higher — which is the statistical fingerprint of the fat tails this entire post is about. Formally it is the fourth standardized moment, $E[(X-\mu)^4]/\sigma^4$, but again the feel is what counts: high kurtosis means "calm most of the time, violent more often than a bell curve allows."

These two numbers — skewness and kurtosis — are exactly the gap between "mean and variance" and "the whole story." The Normal, having zero skew and zero excess kurtosis, throws both away. That is its convenience and its blind spot. We will get there. First, the most famous distribution of all.

![Pipeline from price history through returns and fitting to a usable distribution](/imgs/blogs/probability-distributions-for-markets-math-for-quants-3.png)

Before we tour the distributions one by one, the pipeline above shows the workflow every quant runs: take a price history, turn it into returns, fit a distribution, *check the tails* (the step amateurs skip), and only then use the result to price risk. Notice the order — fitting is cheap; honestly checking the tails is the hard part, and it is the whole reason this post spends so long on the Student-t.

## A map of the territory

There are dozens of named distributions, but markets lean on a small, sturdy set. They split cleanly into two families, and keeping the split straight saves you from category errors (like trying to model a price with a distribution that can go negative).

![Tree taxonomy of market distributions branching into discrete and continuous families](/imgs/blogs/probability-distributions-for-markets-math-for-quants-4.png)

The taxonomy above is worth memorizing. On the **discrete** side: Bernoulli (a single win/loss) and Poisson (counts of events). On the **continuous** side: the Normal (returns), the lognormal (prices), the Student-t (fat-tailed returns), and the exponential (waiting times). Everything else a quant uses is a relative or a mixture of these. We will take them roughly in the order they matter for a working trader, starting with the one that anchors all the others.

## 1. The Normal distribution: the convenient building block

### The intuition before the formula

Suppose you measure the height of every adult in a city. Most people cluster near the average; fewer are very tall or very short; almost nobody is a giant or extraordinarily small. Plot the counts and you get the familiar symmetric hump — the **bell curve**. Heights do this. So do measurement errors, exam scores, and the daily wiggles of a calm market. The bell curve shows up everywhere because of a deep theorem (the Central Limit Theorem, which gets its own post in this series), but for now one plain idea is enough: *when lots of small, independent pushes add up, the total tends to be Normal.* A day's return is the sum of thousands of tiny trades, so as a first approximation it looks bell-shaped.

### The formal definition

The **Normal distribution**, written $N(\mu, \sigma^2)$, has the density

$$ f(x) = \frac{1}{\sigma\sqrt{2\pi}}\,\exp\!\left(-\frac{(x-\mu)^2}{2\sigma^2}\right). $$

The two parameters are everything you need: $\mu$ sets the center, $\sigma$ sets the width. There are no other knobs — which is both the Normal's great strength (simplicity) and its fatal weakness (it cannot make its tails heavier without also widening its whole body). The curve is perfectly symmetric around $\mu$, and the famous **68–95–99.7 rule** says about 68% of the mass falls within one $\sigma$ of the mean, 95% within two, and 99.7% within three.

### Why quants love it (and why that is dangerous)

The Normal is convenient for reasons that have nothing to do with whether it is *true*:

- It is fully described by just two numbers, $\mu$ and $\sigma$.
- The sum of Normals is Normal, so portfolios of Normal assets stay Normal — the math stays closed-form.
- It is the backbone of the Black-Scholes option-pricing model and most classical Value-at-Risk systems.

The danger is the tails. Because the density falls off like $e^{-x^2}$, the probability of extreme moves collapses *astonishingly* fast. A 3-sigma day has probability about 0.13%; a 5-sigma day has probability about 1 in 3.5 million. Markets do not respect this. They produce 5-sigma days every few years. The Normal's tails are too thin, and thin tails are how a risk model gets a firm killed.

#### Worked example: the daily move on a \$100 stock

You hold a \$100 stock with an annualized volatility of 20.6%. To get the *daily* volatility we divide by the square root of the number of trading days in a year ($\approx 252$), because variance scales with time and standard deviation scales with the square root of time:

$$ \sigma_{\text{daily}} = \frac{0.206}{\sqrt{252}} \approx 0.013 = 1.3\%. $$

So on a \$100 position, a one-standard-deviation day is a move of about \$1.30. The 68–95–99.7 rule then says: on roughly 68% of days the stock moves less than \$1.30 in either direction, on 95% of days less than \$2.60, and on 99.7% of days less than \$3.90. A move of \$5 — a 5% day — is $0.05 / 0.013 \approx 3.85$ standard deviations out. Reading the Normal CDF, the probability of a day that bad or worse is about $6.0 \times 10^{-5}$, or **1 in 16,700 trading days** — about once every 66 years.

The intuition: the Normal turns "5% down day" into a once-in-a-lifetime event, and that single understatement is the original sin of quantitative risk.

### Where the Normal earns its keep

It would be unfair to leave the impression that the Normal is useless — it is the most useful distribution in finance precisely because it is "wrong but cheap." For the calm 95% of days, a Normal fit to recent returns is genuinely accurate, and almost every classical result in quant finance leans on its closed-form convenience. The Black-Scholes formula assumes log-returns are Normal; portfolio theory assumes returns are jointly Normal so that a portfolio's risk reduces to the tidy quadratic form $w^\top \Sigma w$; the standard error of a sample mean is Normal by the Central Limit Theorem. None of these tools would have clean formulas without the Normal. The professional stance is not "never use the Normal" — it is "use the Normal for the center, and switch to a fat-tailed model the moment the tail is what you are pricing." The mistake is using one distribution for both jobs.

A second virtue worth naming: the Normal is *stable under addition*. If asset A's return is $N(\mu_A, \sigma_A^2)$ and asset B's is $N(\mu_B, \sigma_B^2)$, then a portfolio holding both has a return that is *also* Normal, with mean and variance you can compute by hand. No other distribution in this post is so well-behaved when you add things up. That single property is why so much of portfolio mathematics quietly assumes normality even when everyone in the room knows the tails are wrong — the algebra is simply too convenient to give up until the crisis forces the issue.

## 2. The lognormal distribution: why prices cannot go negative

### The intuition before the formula

There is a subtle but crucial mistake hiding in "model returns with a Normal." A Normal random variable can be any real number, including very negative ones. If you model a *return* as Normal, fine — a return can be $-30\%$. But if you naively model a *price* as Normal, the math will happily hand you a negative price, and a stock cannot cost negative dollars. Prices have a hard floor at zero.

The fix comes from how prices actually move: they compound. A 10% gain followed by a 10% loss does not return you to where you started (\$100 → \$110 → \$99). Prices multiply, they do not add. And when you take the logarithm of a product, it turns into a sum — and sums of small independent things are Normal. So the natural model is: *the log of the price is Normal.* A variable whose logarithm is Normal is called **lognormal**, and by construction it can never go below zero (you can take the log of any positive number, but never of zero or a negative). Prices are lognormal for the same reason returns are Normal: one is the exponential of the other.

![Graph showing Normal log-returns feeding GBM compounding into a lognormal price](/imgs/blogs/probability-distributions-for-markets-math-for-quants-5.png)

The diagram above traces the logic: start with Normal log-returns, compound them through time via **geometric Brownian motion** (GBM) — the standard continuous-time model where a price grows by a fixed drift plus random Normal shocks — apply the exponential, and you land on a lognormal price that stays above zero and is right-skewed (it has more room to run up than down).

### The formal definition

Under geometric Brownian motion, a price $S_t$ starting at $S_0$ evolves so that

$$ S_T = S_0 \exp\!\left(\Big(\mu - \tfrac{1}{2}\sigma^2\Big)T + \sigma\sqrt{T}\,Z\right), \quad Z \sim N(0,1). $$

Here $\mu$ is the drift (the average growth rate), $\sigma$ is the volatility, $T$ is the time horizon in years, and $Z$ is a standard Normal shock. The exponent is Normal, so $S_T$ is lognormal. The two facts you will use constantly:

$$ E[S_T] = S_0\,e^{\mu T}, \qquad \text{median}(S_T) = S_0\,e^{(\mu - \frac{1}{2}\sigma^2)T}. $$

The mean and the median are *not* equal — the mean is higher — because the lognormal is right-skewed. That gap, the $-\tfrac{1}{2}\sigma^2$ term, is one of the most important and most counterintuitive facts in all of finance. It is called **volatility drag**, and it says: the more a price bounces around, the more its typical (median) outcome lags its average (mean) outcome.

#### Worked example: a \$100 stock after one year of GBM

You buy a stock at $S_0 = \$100$. You assume a drift of $\mu = 8\%$ per year and a volatility of $\sigma = 20\%$, over a horizon of $T = 1$ year. What can you say about the price in a year?

First, the **expected price**:

$$ E[S_T] = 100 \times e^{0.08 \times 1} = 100 \times 1.0833 = \$108.33. $$

Now the **median** — the outcome with a 50/50 chance of being above or below it:

$$ \text{median} = 100 \times e^{(0.08 - 0.5 \times 0.20^2)\times 1} = 100 \times e^{0.06} = \$106.18. $$

Notice the median (\$106.18) sits *below* the mean (\$108.33). That \$2.15 gap is volatility drag in dollars: the typical investor in this stock ends up with less than the headline "expected" return suggests, because the big winners pull the average up. Finally the **one-sigma band**. On the log scale, $\ln S_T$ is Normal with mean $\ln 100 + 0.06 = 4.665$ and standard deviation $\sigma\sqrt{T} = 0.20$. One standard deviation each way gives:

$$ \text{low} = e^{4.665 - 0.20} = \$86.94, \qquad \text{high} = e^{4.665 + 0.20} = \$129.69. $$

So about 68% of the time, a year from now the stock lands between **\$86.94 and \$129.69**. Note the band is *asymmetric* around \$100 — it has more room above than below — which is exactly the right-skew the lognormal builds in.

The intuition: prices are lognormal because they compound, the floor at zero comes for free, and the mean always sits a little above the median by the amount of volatility drag.

#### Worked example: volatility drag over five years

Volatility drag compounds, so its effect grows with the horizon. Take the same stock — $S_0 = \$100$, drift $\mu = 8\%$, volatility $\sigma = 20\%$ — but now look out $T = 5$ years. The expected price is

$$ E[S_5] = 100 \times e^{0.08 \times 5} = 100 \times e^{0.40} = \$149.18, $$

while the median outcome is

$$ \text{median} = 100 \times e^{(0.08 - 0.5 \times 0.20^2) \times 5} = 100 \times e^{0.30} = \$134.99. $$

The gap between mean and median has widened from \$2.15 at one year to **\$14.19 at five years**. A naive investor who reads "8% expected return" and projects \$100 growing to \$149 will be disappointed more than half the time, because the *typical* outcome is only \$135. The mean is dragged up by a handful of spectacular paths; the median — what you are most likely to actually experience — lags behind by the accumulated drag. Crank the volatility to 40% and the five-year median falls all the way to $100 \times e^{(0.08 - 0.08)\times 5} = \$100$: a stock with a positive expected return whose typical five-year outcome is *no gain at all*. That is volatility drag at full strength, and it is exactly why leveraged funds erode in choppy markets.

The intuition: "expected return" is a statement about the average across all parallel universes, not about the universe you will actually live in — and the higher the volatility, the more those two diverge.

## 3. The Student-t distribution: fat tails and survival

### The intuition before the formula

Return to the original sin: the Normal says a 5% down day is a 66-year event, and reality says it happens every few years. The problem is the Normal's tails fall off too fast. We need a distribution that looks almost identical in the middle — the day-to-day wiggle is genuinely bell-shaped — but assigns far more probability to the extremes. We need **fat tails**.

The everyday analogy: think of two airlines with the same *average* delay of 10 minutes. Airline A is boringly consistent — almost always 8 to 12 minutes late. Airline B is usually right on time, but a few flights per year are *cancelled outright*, stranding you for a day. Same average, totally different tail. The Normal is Airline A. Markets are Airline B. The **Student-t distribution** is the math for Airline B: a bell-shaped center with heavy, slow-decaying tails.

![Before and after comparison of Normal versus Student-t odds of a 5 percent crash day](/imgs/blogs/probability-distributions-for-markets-math-for-quants-2.png)

The contrast above is the whole reason the Student-t earns its place. Fit a Normal to a stock's daily returns and a 5% down day looks like a 1-in-16,700-day event. Fit a Student-t with the *same volatility* and the same crash becomes roughly a 1-in-361-day event — about once a year. The center of the two curves is nearly indistinguishable; only the tail differs, and the tail is where the money is lost.

### The formal definition

The **Student-t distribution** has a parameter $\nu$ ("nu"), the **degrees of freedom**, that controls tail heaviness. Its density is

$$ f(x) = \frac{\Gamma\!\big(\frac{\nu+1}{2}\big)}{\sqrt{\nu\pi}\,\Gamma\!\big(\frac{\nu}{2}\big)}\left(1 + \frac{x^2}{\nu}\right)^{-\frac{\nu+1}{2}}, $$

where $\Gamma$ is the gamma function (a continuous version of the factorial — you do not need its details). The key is the term $(1 + x^2/\nu)^{-(\nu+1)/2}$: instead of dying like $e^{-x^2}$ (the Normal), the t-density dies like a *power* of $x$. Power-law decay is vastly slower than exponential decay, and that slowness is the fat tail.

The single knob $\nu$ does all the work:

- **Small $\nu$ (3, 4, 5)** → very fat tails. Extreme moves are common. This is what real daily equity returns look like.
- **Large $\nu$ (30, 100)** → tails thin out, and the t converges to the Normal. The Normal is the Student-t with $\nu = \infty$.

There is a catch worth knowing: with $\nu \le 2$ the variance is infinite, and with $\nu \le 1$ even the mean is undefined. For most equity-return work quants use $\nu$ somewhere between 3 and 6 — fat enough to capture crashes, finite enough to have a usable variance.

A useful way to read the degrees-of-freedom number: $\nu$ is roughly "how many independent observations the tail behaves like." A small $\nu$ means a few extreme observations dominate everything — a single crash day can outweigh a year of calm. A large $\nu$ means no single observation matters much, which is the Normal's comfortable world where everything averages out smoothly. The Student-t was originally invented in 1908 by William Gosset (writing under the pen name "Student" because his employer, the Guinness brewery, forbade publishing) to handle *small samples*, where a few unusual observations carry outsized weight. It is a happy accident of history that the same mathematics — designed for the small-sample problem of a brewer — turns out to describe the fat-tailed behavior of financial returns almost perfectly. The mechanism is the same in both cases: a world where the extremes refuse to be tamed by averaging.

One more practical point. The plain Student-t is *symmetric* — it makes a 5% up day exactly as likely as a 5% down day. Real markets are not symmetric; crashes are bigger than rallies (that negative skew again). So practitioners often reach for a **skewed Student-t**, which adds one more parameter to tilt the distribution and let the left tail be fatter than the right. The lesson generalizes: each distribution is a base model you bend with extra parameters until it matches the asymmetry and tail weight your data actually show.

### Why fat tails cost real money

A risk model built on the Normal will systematically *under-reserve* capital for crashes. It will set position sizes too large, set Value-at-Risk limits too loose, and price tail-risk insurance too cheap. The 2008 crisis was, in part, a fat-tail failure: models that assumed thin tails treated correlated mortgage defaults as near-impossible right up until they happened together.

![Before and after of a thin-tailed versus fat-tailed model on extreme moves](/imgs/blogs/probability-distributions-for-markets-math-for-quants-7.png)

The figure above makes the cost concrete: two models can agree perfectly on the calm center and still disagree by a factor of dozens on the crash. A thin-tailed model leaves the crash underpriced; a fat-tailed model prices it in. When you are sizing a position or buying disaster insurance, that factor is the difference between surviving a bad month and not.

#### Worked example: the probability of a -5% day, Normal vs Student-t

You have a stock with a daily volatility of 1.3% (the same \$100 stock from before). You want the probability of a day at or below $-5\%$ under two models.

**Under the Normal:** a $-5\%$ move is $0.05 / 0.013 \approx 3.85$ standard deviations out. The Normal CDF gives

$$ P_{\text{Normal}}(X \le -5\%) \approx 6.0 \times 10^{-5} = \frac{1}{16{,}700}. $$

**Under the Student-t with $\nu = 4$:** first we scale the t so its variance also equals $(1.3\%)^2$. A raw t with $\nu = 4$ has variance $\nu/(\nu-2) = 2$, so we divide by $\sqrt{2}$, giving a scale of $1.3\%/\sqrt{2} \approx 0.92\%$. A $-5\%$ move is then $0.05/0.0092 \approx 5.4$ scaled-t units out. Integrating the t-density's tail:

$$ P_{t,\,\nu=4}(X \le -5\%) \approx 2.8 \times 10^{-3} = \frac{1}{361}. $$

Same volatility. Same center. But the fat-tailed model says a 5% crash is about **46 times more likely** than the Normal claims — roughly once a year instead of once a lifetime. In dollar terms: if you sold out-of-the-money "crash insurance" on a \$1,000,000 book and priced it off the Normal, you would charge 46× too little and blow up the first time reality showed up.

The intuition: fat tails are not a rounding error — they are the dominant risk, and the degrees-of-freedom parameter is the dial that controls how scared you should be of the extremes.

## How fitting a distribution actually works

You have the distributions; how do you choose parameters from real data? This is the "fit" step in the pipeline, and it is simpler than it sounds.

The standard tool is **maximum likelihood estimation** (MLE): pick the parameter values that make the data you actually observed as probable as possible. For the Normal, MLE just hands you the obvious answer — set $\hat\mu$ to the sample average return and $\hat\sigma$ to the sample standard deviation. For the Student-t, MLE additionally searches for the degrees of freedom $\hat\nu$ that best matches how fat the observed tails are; for daily equity index returns, that search almost always lands somewhere between 3 and 6.

Here is a runnable sketch in Python with `scipy`, the kind of thing a quant runs in a few lines:

```python
import numpy as np
from scipy import stats

rets = np.array([0.004, -0.011, 0.002, 0.018, -0.053, 0.007, -0.002, 0.009])

mu, sigma = stats.norm.fit(rets)              # Normal MLE: sample mean and std
nu, loc, scale = stats.t.fit(rets)            # Student-t MLE: df, location, scale
print(f"Normal:  mu={mu:.4f}, sigma={sigma:.4f}")
print(f"Student-t: nu={nu:.2f}, loc={loc:.4f}, scale={scale:.4f}")

p_norm = stats.norm.cdf(-0.05, loc=mu, scale=sigma)        # P(<= -5%) Normal
p_t    = stats.t.cdf(-0.05, df=nu, loc=loc, scale=scale)   # P(<= -5%) Student-t
print(f"P(<=-5%): Normal={p_norm:.2e}  Student-t={p_t:.2e}")
```

The one discipline that separates good fitting from dangerous fitting is the **tail check** — comparing the model's predicted frequency of large moves against the historical count, not just eyeballing the center. A Normal and a t can both look like a perfect fit in the middle 95% of the data and disagree by 50× in the 1% that matters. Always check the tail.

## 4. Bernoulli and binomial: the math of hit rates

### The intuition before the formula

The simplest random variable in all of finance is a single bet that either wins or loses. Did the trade make money: yes or no? That is a **Bernoulli** trial — one event with two outcomes, "success" with probability $p$ and "failure" with probability $1-p$. A trader's **hit rate** (the fraction of trades that are profitable) is exactly the $p$ of a Bernoulli.

Stack many independent Bernoulli trials together and count the successes, and you get the **binomial** distribution: out of $n$ trades, how many won? The everyday analogy is flipping a biased coin $n$ times and counting heads.

### The formal definition

A Bernoulli variable has mean $p$ and variance $p(1-p)$. The binomial — the number of successes $X$ in $n$ independent trials each with success probability $p$ — has PMF

$$ P(X = k) = \binom{n}{k} p^k (1-p)^{n-k}, $$

with mean $np$ and variance $np(1-p)$. The term $\binom{n}{k}$ ("n choose k") counts the number of ways to arrange $k$ wins among $n$ trades. These two distributions are the foundation of every conversation about win rates, the Kelly criterion for bet sizing, and whether a strategy's track record could be luck.

#### Worked example: is a 55% hit rate going to feel good?

You run a strategy with a true hit rate of $p = 55\%$ and you take $n = 100$ trades, each risking \$1,000 with a symmetric \$1,000 win/loss. Your expected number of winners is $np = 55$, and the standard deviation of the count is $\sqrt{np(1-p)} = \sqrt{100 \times 0.55 \times 0.45} \approx 4.97$.

So even with a genuine edge, your win count will routinely swing by 5 either way. What is the probability you actually *lose money over the 100 trades* — that is, you win 50 or fewer? Using a Normal approximation to the binomial (valid because $n$ is large), 50 wins is about $(50.5 - 55)/4.97 \approx -0.9$ standard deviations below the mean, giving a probability of roughly **18%**. In dollar terms, with a 55% edge you still have nearly a one-in-five chance of being *down* after 100 trades — a stretch of about \$2,000–\$10,000 underwater is entirely normal noise.

The intuition: a real edge is a statistical statement about the long run, and the binomial tells you exactly how much short-run pain a genuine edge can still inflict — which is why undercapitalized traders with good systems still go broke.

## 5. Poisson and the counting of market events

### The intuition before the formula

Some market quantities are not values but *counts*: how many trades printed in the last minute, how many large jumps happened this quarter, how many bonds in a portfolio defaulted this year. These share a structure — events arrive randomly over time, at some average rate, independently of each other. The distribution for "how many events in a fixed window" is the **Poisson distribution**.

The everyday analogy: a call center gets, on average, 8 calls a minute. Some minutes bring 5, some bring 12, occasionally one brings 18 — but the average is 8. Poisson tells you how the count is spread around that average. In markets, replace "calls" with "trades," "price jumps," or "defaults."

![Stack contrasting discrete count families against continuous value families](/imgs/blogs/probability-distributions-for-markets-math-for-quants-6.png)

The stack above places Poisson where it belongs: among the discrete, count-based distributions, alongside Bernoulli and binomial, and distinct from the continuous value-based ones like the Normal and lognormal. When you are counting whole things, you are in the discrete world; when you are measuring on a smooth scale, you are in the continuous world. Picking the wrong family is a classic beginner error.

### The formal definition

The **Poisson distribution** has a single parameter $\lambda$ ("lambda"), the average number of events per window. Its PMF is

$$ P(N = k) = \frac{\lambda^k e^{-\lambda}}{k!}, \qquad k = 0, 1, 2, \dots $$

A defining quirk: the mean and the variance are *both* equal to $\lambda$. So if you measure trade arrivals and find the variance of the per-minute count is much larger than the mean, that is a signal the arrivals are *not* Poisson — they are clustered (bursty), which is exactly what happens around news events. That mean-equals-variance test is a quant's first diagnostic for whether the simple Poisson model is even appropriate.

Where does the Poisson come from? It is the limit of the binomial when you have a huge number of tiny opportunities for an event, each individually unlikely. Chop a minute into 60,000 milliseconds; in each millisecond a trade either arrives (tiny probability) or does not. That is a binomial with $n = 60{,}000$ and a small $p$, and as $n$ grows and $p$ shrinks with $np = \lambda$ held fixed, the binomial converges exactly to the Poisson. This is why Poisson is the natural distribution for *rare events with many chances to happen*: trades in a minute, defaults in a year, lightning strikes in a summer, typos in a book. The single parameter $\lambda = np$ absorbs both the number of opportunities and the per-opportunity probability into one rate.

The same lens explains a Poisson's most important market relative, the **compound Poisson** and the **jump-diffusion** model. A pure Poisson counts jumps; a jump-diffusion model says the price follows smooth (Normal) diffusion *most* of the time, but at Poisson-distributed random moments it gets a sudden discontinuous jump (a gap up or down on news). That combination — a calm Normal engine plus a Poisson-timed jump generator — is one of the cleanest mathematical pictures of why markets gap, and it gets its own treatment later in this series.

#### Worked example: the probability of a busy trading minute

A liquid stock trades, on average, $\lambda = 8$ times per minute. Your execution algorithm is tuned for normal flow but can get overwhelmed if more than 12 trades hit in a single minute. What is the probability of such a busy minute?

We need $P(N > 12) = 1 - P(N \le 12)$. Summing the Poisson PMF from $k = 0$ to $12$ with $\lambda = 8$ gives $P(N \le 12) \approx 0.9362$, so

$$ P(N > 12) = 1 - 0.9362 = 0.0638 \approx 6.4\%. $$

So roughly **1 minute in 16** is busier than your algorithm's comfort zone. If a busy minute costs you, say, \$50 in extra slippage and you trade 390 minutes a day, you would expect about $0.064 \times 390 \approx 25$ busy minutes daily, or about \$1,250 a day in avoidable cost — a real number that justifies engineering a more robust algorithm. For comparison, a truly extreme minute of 16-plus trades has probability about $0.008$, or **1 in 121 minutes** — rare, but you will see a few every day.

The intuition: Poisson turns "how often will the unusual happen" into a precise number, and because its variance equals its mean, it also gives you a free test for whether your events are arriving independently or clustering into bursts.

## 6. The exponential distribution: waiting between events

### The intuition before the formula

Poisson counts events in a window; its close cousin, the **exponential distribution**, measures the *gap* between consecutive events — the waiting time until the next one. If trades arrive as a Poisson process at rate $\lambda$, then the time between trades is exponentially distributed. The two are two views of the same coin: count the events, or time the gaps.

The exponential has one famous, almost magical property: it is **memoryless**. If the average wait for the next trade is 8 seconds, and you have *already* waited 8 seconds with nothing happening, your expected remaining wait is still 8 seconds. The distribution has no memory of how long you have been waiting. The everyday analogy: a fair coin does not "owe" you a head after five tails — and a memoryless waiting time does not "owe" you an event because you have been patient. This is deeply counterintuitive and constantly trips people up.

### The formal definition

The **exponential distribution** with rate $\lambda$ has density and survival function

$$ f(t) = \lambda e^{-\lambda t}, \qquad P(T > t) = e^{-\lambda t}, \qquad E[T] = \frac{1}{\lambda}. $$

Here $\lambda$ is the same rate as in the matching Poisson process, $t$ is the elapsed time, and the expected wait is simply the reciprocal of the rate. The survival function $P(T > t) = e^{-\lambda t}$ is the probability you are *still waiting* after time $t$, and the memoryless property is the statement that $P(T > s + t \mid T > s) = P(T > t)$ — conditioning on having already waited $s$ changes nothing.

#### Worked example: how long until the next print, and what it costs

The same stock trades at $\lambda = 8$ times per minute, which is a rate of $8/60 \approx 0.133$ trades per second. You want to buy and you have decided to wait for the next print rather than cross the spread aggressively. How long will you wait, and what does it cost?

The expected waiting time is

$$ E[T] = \frac{1}{\lambda} = \frac{60}{8} = 7.5 \text{ seconds}. $$

So on average you will wait **7.5 seconds** for the next trade. But the spread of waits is wide: the probability of waiting more than 30 seconds is

$$ P(T > 30) = e^{-0.133 \times 30} = e^{-4} \approx 0.018 = 1.8\%. $$

About 1 attempt in 55, you will sit unfilled for over half a minute. Now the dollar implication. Suppose while you wait, the price drifts against you at \$0.001 per second (a tenth of a cent — modest adverse drift). Over the expected 7.5-second wait on a 1,000-share order, your expected slippage is

$$ 0.001 \times 7.5 \times 1{,}000 = \$7.50. $$

That \$7.50 is the price of patience. If the spread you were trying to avoid crossing is only \$5, you should just cross it — the expected waiting cost exceeds the spread. This is the core tension in execution: every passive order trades a known spread cost against an uncertain, exponentially-distributed waiting cost.

The intuition: the exponential turns "how long until the next trade" into both an average and a full risk profile, and its memorylessness means waiting longer never improves your odds for the next instant — a fact that quietly governs every passive-versus-aggressive execution decision.

## 7. Mixtures and regime-switching: calm versus crisis

### The intuition before the formula

Here is the deepest idea in the post, and it reconciles everything above. Real markets are not *one* distribution — they are a **mixture** of distributions, because markets have moods. Most of the time the market is calm: returns are nearly Normal with low volatility. Occasionally it flips into crisis: volatility triples, correlations spike, and moves get violent. These are called **regimes**.

The everyday analogy: the weather where you live is a mixture. Most days are mild (one distribution of temperatures); a few days a year are extreme storms (a different distribution entirely). If you averaged all days into a single bell curve, you would both overstate how common storms are on a normal day *and* understate how bad the worst day can be. You need two distributions and a switch between them.

### The formal definition

A **mixture distribution** combines component distributions with weights that sum to 1. A simple two-regime model for returns:

$$ f(x) = (1-q)\,N(\mu_c, \sigma_c^2) + q\,N(\mu_x, \sigma_x^2), $$

where the first component is the **calm** regime (low volatility $\sigma_c$), the second is the **crisis** regime (high volatility $\sigma_x$), and $q$ is the small probability of being in crisis. Here is the beautiful part: *a mixture of two Normals automatically has fat tails.* You do not need the Student-t at all — you can manufacture fat tails by mixing a calm Normal and a crisis Normal. This is why the Student-t works so well as a model: a t-distribution is mathematically equivalent to a Normal whose variance is itself random, which is exactly what a regime-switching market produces. Fat tails are what mixing looks like from the outside.

#### Worked example: a regime mixture that produces a fat tail

Suppose on a calm day (probability $1-q = 95\%$) returns are Normal with volatility 1%, and on a crisis day (probability $q = 5\%$) returns are Normal with volatility 4%. The overall variance is the weighted average of the variances (since both means are near zero):

$$ \sigma^2 = 0.95 \times 0.01^2 + 0.05 \times 0.04^2 = 0.95 \times 0.0001 + 0.05 \times 0.0016 = 0.000175, $$

so the blended daily volatility is $\sqrt{0.000175} \approx 1.32\%$ — almost exactly the 1.3% we used earlier. But the *tail* is now far heavier than a single Normal at 1.32% vol, because a 5% move is only $1.25$ crisis-sigmas (easy on a crisis day) versus $3.85$ calm-sigmas. The probability of a $-5\%$ day is dominated by the crisis term:

$$ P(X \le -5\%) \approx 0.05 \times \Phi\!\left(\frac{-0.05}{0.04}\right) \approx 0.05 \times 0.106 \approx 0.0053 = \frac{1}{189}. $$

That is roughly **1 in 190 days** — in the same fat-tailed ballpark as the Student-t's 1 in 361, and dozens of times more than the single Normal's 1 in 16,700. The two crisis days a month are doing almost all the tail risk. In dollar terms on a \$1,000,000 book, the difference between budgeting for a 1-in-16,700 crash and a 1-in-190 crash is the difference between holding \$50,000 of reserve and holding enough to actually survive.

The intuition: you do not need an exotic distribution to get fat tails — you just need to admit that the market has more than one mood, and the moment you mix a calm regime with a crisis regime, fat tails appear for free.

## Putting the distributions side by side

It helps to see the whole toolkit in one comparison, with what each models, its parameters, and its single biggest failure mode:

| Distribution | What it models | Parameter(s) | Discrete/Continuous | Biggest failure mode |
|---|---|---|---|---|
| Normal | Returns (calm) | $\mu$, $\sigma$ | Continuous | Tails far too thin |
| Lognormal | Prices | $\mu$, $\sigma$ | Continuous | Assumes constant vol |
| Student-t | Fat-tailed returns | $\nu$, scale | Continuous | Symmetric (no skew) |
| Poisson | Event counts | $\lambda$ | Discrete | Assumes no clustering |
| Exponential | Waiting times | $\lambda$ | Continuous | Memorylessness rarely exact |
| Bernoulli/Binomial | Win/loss, hit rate | $p$ (and $n$) | Discrete | Assumes independent trials |
| Mixture | Calm + crisis | weights, components | Either | Needs regime identification |

> Every distribution is a deliberate simplification of reality. The skill is not finding the "true" one — there isn't one — it's knowing which lie is cheap and which lie will bankrupt you.

The pattern across the table: the *continuous* distributions describe values you measure (returns, prices, waits), the *discrete* ones describe things you count (wins, events), and almost every failure mode is the same failure in disguise — a model that assumes calm when the market has switched to crisis.

## Common misconceptions

**"Stock returns are Normal."** They are *approximately* Normal in the calm center and dramatically non-Normal in the tails. Daily equity returns have excess kurtosis (fatter tails) and negative skew (crashes are bigger than rallies). The Normal is a useful first sketch and a dangerous final model. Treat it as a starting point, never as truth about the tails.

**"Prices are Normal."** Prices are *lognormal*, not Normal — a Normal price model can produce negative prices, which is impossible. It is *log-returns* that are approximately Normal. Confusing the two is one of the most common beginner mistakes, and it quietly corrupts any model that mixes them up.

**"Higher volatility just means a wider bell curve."** Only within a single regime. Real fat tails come from volatility that is itself *random* — the variance changes day to day. That is why a Student-t or a regime mixture beats a fatter single Normal: they let the variance breathe. Widening one Normal makes calm days look wrong too; fattening the tail leaves the center alone.

**"The exponential distribution means events are due."** Memorylessness means the exact opposite. If trades arrive exponentially and you have waited 20 seconds, you are *not* "due" for one — your expected remaining wait is unchanged. Gambler's-fallacy thinking about waiting times leads directly to bad execution decisions.

**"A 55% hit rate guarantees profit."** Over enough trades, yes; over any finite stretch, no. The binomial says a genuine 55% edge still loses money over 100 trades about 18% of the time. Short-run variance is large, and it is what bankrupts undercapitalized traders who actually have an edge.

**"Poisson always fits event counts."** Only if events arrive independently at a constant rate. Around news, earnings, or open/close, trades *cluster* — the variance of the count exceeds its mean, violating Poisson's mean-equals-variance signature. When that happens, you need a clustered model (like a Hawkes process), not plain Poisson.

**"Volatility is a single number."** A stock's "20% vol" is a summary of a distribution, not a fixed property. Volatility itself changes over time — it clusters (calm begets calm, turbulence begets turbulence), which is precisely why a single static Normal fails and models like GARCH, which let volatility evolve, exist. When you quote one volatility number, you are reporting the average width of a distribution whose width is itself a moving target.

**"Fitting more parameters always gives a better model."** A skewed Student-t with five parameters will fit any historical sample better than a two-parameter Normal — but it can *overfit*, mistaking the quirks of one particular history for a law of nature. The tail estimate from a short sample is especially fragile: you are trying to estimate the frequency of events you have barely seen. More parameters buy flexibility at the cost of stability, and the right number is the one that holds up out of sample, not the one that hugs the past most tightly.

## How it shows up in real markets

### 1. Black Monday, October 19, 1987

The S&P 500 fell **20.5% in a single day**. Under a Normal model calibrated to 1987 volatility, that move was roughly a 20-sigma event — a probability so small (around $10^{-88}$) that it should not have happened even once in the entire history of the universe, let alone in one Monday. The crash is the canonical proof that equity returns have fat tails the Normal cannot capture. Every modern risk system that uses a Student-t, a regime mixture, or a historical-simulation tail traces its lineage to the lesson of that day: the Normal's tails are not just thin, they are fictional.

### 2. The 2008 Gaussian copula and mortgage defaults

The pricing of collateralized debt obligations leaned heavily on a Gaussian (Normal-based) model of how mortgage defaults move together. The Normal copula assigned near-zero probability to many mortgages defaulting *simultaneously*. When the housing market turned, defaults clustered exactly the way a fat-tailed, regime-switching reality produces — and the thin-tailed model had priced that scenario as essentially impossible. Hundreds of billions in losses followed. The mechanism is the same as Black Monday: a Normal assumption in the tail, where reality is anything but Normal.

### 3. The exponential clock of high-frequency trading

Market-making and execution algorithms model order and trade arrivals as Poisson processes with exponential inter-arrival times, then adapt when reality clusters. On a calm afternoon, a liquid stock's trades genuinely look Poisson at perhaps 8–20 per minute, and the exponential waiting time governs whether a passive order will fill before the price moves. The worked example's tension — \$7.50 of expected drift versus a \$5 spread — is a decision execution desks make millions of times a day. When the simple exponential breaks down is precisely the interesting moment: news hits, arrivals cluster, and the desks switch to clustered (Hawkes) models that explicitly allow each trade to make the next one more likely.

### 4. The 2020 COVID crash and regime switching

In late February and March 2020, the market flipped regimes in days. The VIX (a measure of expected volatility) leapt from the low teens to over 80; daily moves of 5–12% in either direction became routine for two weeks. A single-Normal risk model calibrated on calm-2019 data would have called those moves astronomically unlikely; a regime-switching or fat-tailed model, which budgets for a crisis component, treated them as a rare-but-expected crisis draw. The episode is a clean illustration of the mixture model: a calm regime and a crisis regime, with the market jumping between them, and all the tail risk concentrated in the crisis component.

### 5. Insurance, defaults, and the Poisson count of rare disasters

Credit-risk and insurance desks model the *number* of defaults or claims in a year as Poisson (or a clustered variant) with rate $\lambda$. If you hold 1,000 bonds each with a 1% annual default probability, the expected number of defaults is $\lambda = 10$, and Poisson tells you the probability of, say, 20 or more defaults — the scenario that eats your capital. The mean-equals-variance property is also the early-warning diagnostic: when realized default counts start showing variance far above their mean, defaults are *correlating* (a regime shift), and the independent-Poisson assumption — and the capital reserve built on it — is no longer safe.

### 6. Volatility drag in leveraged ETFs

The lognormal's $-\tfrac{1}{2}\sigma^2$ term is not academic. A 3× leveraged ETF on a volatile index can *lose* money over a year even when the underlying index ends flat, purely because of volatility drag: the daily compounding of large up-and-down moves erodes the median outcome below the mean. Investors who treated a leveraged ETF as a simple 3× bet — ignoring that prices are lognormal and that drag compounds — have repeatedly been surprised to find a flat market handed them a 20%+ loss. The gap between the mean and the median, the very thing our \$100-stock worked example computed, is the whole explanation.

## When this matters to you

If you ever build a backtest, size a position, price an option, or even just hold a leveraged fund, these distributions are silently governing your outcomes. The single most valuable habit you can build is to *distrust the Normal in the tails* — assume crashes are far more common than the bell curve claims, because they are. The second habit is to keep the families straight: model returns (continuous, roughly Normal but fat-tailed), prices (lognormal, floored at zero), counts (Poisson, discrete), and waits (exponential, memoryless) with the right tool, and never let a price model wander into negative numbers.

A closing note on honesty and risk: every distribution here is a simplification, and the dollar figures in this post are illustrative, not predictions. This is educational material, not financial advice — markets can and do behave worse than *any* of these models on their worst days, which is exactly why position sizing and survival matter more than precision. The point of knowing the distributions is not to predict the future; it is to be the person who budgeted for the crisis regime before it arrived.

For the next step, three companion posts go deeper on neighboring ground:

- The [distributions cheat sheet for quant interviews](/blog/trading/quantitative-finance/distributions-cheat-sheet-quant-interviews) — a fast-reference version of these distributions with the formulas and identities you would be asked to recall under pressure.
- [Brownian motion from the ground up](/blog/trading/quantitative-finance/brownian-motion-quant-interviews) — the continuous-time process that turns Normal log-returns into the lognormal price model we used here.
- [Classic quant probability problems](/blog/trading/quantitative-finance/classic-quant-probability-problems) — worked problems that drill the Poisson, exponential, and binomial reasoning from this post into reflexes.

Master these six distributions and their one mixture, learn where each one lies, and you will have the probabilistic backbone that every other piece of quant math — the Central Limit Theorem, stochastic calculus, option pricing, risk management — is built on top of.
