---
title: "Maximum likelihood and the method of moments: how quants fit a model to returns"
date: "2026-06-15"
description: "A beginner-friendly, build-from-zero guide to maximum likelihood and the method of moments, and how quants use them to calibrate distributions, fit GARCH volatility models, and pin down the precision of their estimates."
tags: ["maximum-likelihood", "method-of-moments", "estimation", "garch", "fisher-information", "cramer-rao", "calibration", "volatility", "quant-finance", "math-for-quants"]
category: "trading"
subcategory: "Quantitative Finance"
author: "Hiep Tran"
featured: true
readTime: 44
---

> [!important]
> **TL;DR** — Fitting a model to market data almost always comes down to one question: which parameter values make the data we actually saw the most probable? Maximum likelihood answers that question precisely; the method of moments answers a quick-and-dirty cousin of it.
>
> - The **likelihood** $L(\theta) = \prod_i f(x_i;\theta)$ is the probability of your data viewed as a function of the model's parameters; the **maximum-likelihood estimate** (MLE) $\hat\theta$ is the parameter value that makes that probability as large as possible.
> - We always maximize the **log-likelihood** $\ell(\theta) = \sum_i \log f(x_i;\theta)$ instead, because turning a product of tiny numbers into a sum makes the calculus and the computer arithmetic behave.
> - The **method of moments** sets sample moments equal to theoretical moments and solves — fast, closed-form, perfect for a starting guess, but usually less precise than MLE.
> - MLE is **consistent** (it homes in on the truth), **asymptotically normal** (its error is bell-shaped), and **efficient** (no unbiased estimator beats it for large samples); the **Cramér–Rao bound** says exactly how small its standard error can be.
> - The one number to remember: with daily returns, the standard error of a volatility estimate is roughly $\sigma/\sqrt{2n}$, so estimating volatility to within **5%** of its true value needs about **200 days** of data — and no method can do better.

Here is a question every quant faces on day one and every day after: you have a model with a knob on it, and you have a pile of data. Which way should you turn the knob?

The model might be "daily returns are normally distributed." The knob is the volatility — turn it up and the bell curve gets wider, turn it down and it gets narrower. You have 250 days of returns sitting in a spreadsheet. There is no law of nature that hands you the right volatility; you have to *infer* it from the data. The same problem shows up everywhere: fitting a fat-tailed distribution to crash-prone returns (the knob is "how fat are the tails"), calibrating a volatility-forecasting model (several knobs), pricing options off historical data, sizing a position by its estimated risk. In every case you are turning knobs to make a model agree with reality. This post is about the two ideas that tell you how far to turn each knob, and how much to trust the answer.

![Pipeline from a sample of returns through the likelihood function to the argmax to the fitted parameter](/imgs/blogs/mle-method-of-moments-math-for-quants-1.png)

The diagram above is the mental model for the whole post. Data goes in on the left. You build a *likelihood* — a number that says how probable your data is for a given setting of the knobs. You take its logarithm to make the math friendly. You find the knob setting that maximizes it. Out the right side comes $\hat\theta$, your best-fit parameter. The method of moments is a shortcut that skips the middle and matches averages instead, and we will see exactly when that shortcut is good enough. Let us build all of it from zero, and ground every formula in a dollar decision a trader actually makes.

## Foundations: the building blocks of fitting a model

Before we can maximize anything, we need to agree on a handful of words. We will define each one the first time it appears, build the simplest possible version of the idea, and only then reach for the machinery a practitioner uses every day. If you already know what a probability density is, skim; if not, you can still follow every line.

### What is a parameter?

A **parameter** is a number that pins down the exact shape of a model. Think of a model as a *family* of possible worlds, and the parameter as the address of one specific world inside that family. "Returns are normally distributed" describes a whole family — there is a thin normal, a fat normal, one centered at zero, one centered at +1%. The two parameters of a normal distribution, its **mean** $\mu$ (the center) and its **variance** $\sigma^2$ (the spread), select exactly one curve from that family. We collect all the parameters of a model into a single symbol $\theta$ (the Greek letter "theta"); for a normal, $\theta = (\mu, \sigma^2)$. Estimation is the art of choosing $\theta$ from data.

### What is a probability density?

A **probability density** $f(x;\theta)$ is a function that tells you how *concentrated* probability is around each possible value $x$, given a parameter setting $\theta$. It is tall where outcomes are likely and short where they are rare. For a continuous quantity like a daily return, you do not read off a probability directly from the density — you get an actual probability only by integrating the density over a range. But the height still has a clean meaning: a return value where the density is twice as tall is, loosely, twice as "expected." The everyday analogy: a density is like the darkness of ink sprayed onto a number line by a leaky pen — darker regions are where the pen spends more time, i.e. where outcomes pile up. The semicolon in $f(x;\theta)$ is read "the density at $x$, *for the parameter* $\theta$" — it reminds us the shape depends on the knob setting.

For the normal distribution, the density is the famous bell curve:

$$ f(x;\mu,\sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}}\, \exp\!\left(-\frac{(x-\mu)^2}{2\sigma^2}\right). $$

Here $x$ is the value (a return), $\mu$ is the center, $\sigma^2$ is the variance, $\pi$ is the usual 3.14159, and $\exp$ is the exponential function $e^{(\cdot)}$. The fraction out front normalizes the curve so its total area is exactly 1, and the exponential makes it tall near $\mu$ and tiny far away. Do not memorize it; just notice that it has a center knob and a spread knob, and that it falls off fast as $x$ moves away from $\mu$.

### What is a sample, and what does "i.i.d." mean?

A **sample** is the data you collected: $x_1, x_2, \ldots, x_n$, say the daily returns of a stock over $n$ trading days. Throughout most of this post we assume the data points are **i.i.d.** — *independent and identically distributed*. "Identically distributed" means each $x_i$ is drawn from the same density $f(x;\theta)$ with the same true $\theta$. "Independent" means knowing one return tells you nothing about the next. Real returns are not perfectly i.i.d. — volatility clusters, which is the entire reason GARCH exists — but i.i.d. is the clean case we learn on, and we will relax it later.

### What is an estimator?

An **estimator** is a recipe that turns a sample into a guess for a parameter. The sample mean $\bar x = \frac{1}{n}\sum_i x_i$ is an estimator of $\mu$. We write a guessed parameter with a hat: $\hat\theta$ is "our estimate of $\theta$." A good estimator gets close to the truth, does not systematically miss high or low (low **bias**), and does not bounce around much from sample to sample (low **variance**). We have a whole sibling post on these tradeoffs — [estimators, bias, and variance](/blog/trading/quantitative-finance/estimators-mle-bias-variance-quant-interviews) — and the present post is really about the two best general recipes for *building* an estimator in the first place.

#### Worked example: turning a knob by eye before turning it by math

You collect 5 daily returns from a stock, in percent: $+1.0,\ -0.5,\ +0.5,\ +2.0,\ -0.5$. You believe they are normal. Which volatility makes the data look most natural? Try $\sigma = 0.3\%$ first. Two of your returns ($+2.0$ and, to a lesser extent, $+1.0$) are several $\sigma$ away from the average of $+0.5\%$ — a $+2.0\%$ return is $(2.0-0.5)/0.3 = 5$ standard deviations out, which under a normal happens about once in 3.5 million draws. Seeing it in a sample of 5 should make you deeply suspicious that $\sigma = 0.3\%$ is wrong. Now try $\sigma = 1.0\%$: the $+2.0\%$ return is only 1.5 standard deviations out, perfectly ordinary. The wider curve makes your actual data look *unremarkable*, which is exactly what you want. We have not done any calculus yet, but you have already grasped the whole engine of maximum likelihood: pick the parameter that makes the data you actually saw look the least surprising. In a moment we will compute the precise best $\sigma$ — it turns out to be about \$0.93 per \$100, i.e. 0.93% — but the intuition is the entire game. The one-sentence intuition: the right parameter is the one under which your real data stops looking like a freak accident.

## 1. The likelihood function and the MLE recipe

Now we formalize the eyeball test. The key conceptual flip — and it trips up everyone at first — is that we are going to treat the *data as fixed* and the *parameter as the variable*. That is backwards from how we usually think about a density, where the parameter is fixed and the data varies. Hold that thought; it is the whole trick.

### From density to likelihood

For a single data point $x_i$, the density $f(x_i;\theta)$ measures how plausible that one observation is under parameter $\theta$. Because our data points are independent, the joint plausibility of seeing *all* of them is the product of the individual densities — independence is exactly the property that lets us multiply probabilities. We give that product a name, the **likelihood**:

$$ L(\theta) = \prod_{i=1}^{n} f(x_i;\theta). $$

The capital pi $\prod$ means "multiply these together," running $i$ from 1 to $n$. Read $L(\theta)$ out loud as: "the likelihood of the parameter $\theta$, given the data we have." It is a number between 0 and (for continuous data) potentially large or tiny, and crucially it is a *function of $\theta$* — you feed it a candidate knob setting and it returns how well that setting explains the data.

The analogy that makes this click: imagine a detective with a fixed set of clues (the data) considering several suspects (parameter values). For each suspect, the detective asks "if this person did it, how probable are exactly these clues?" The suspect under whom the clues are most probable is the prime suspect. The likelihood is that "how probable are the clues under this suspect" number, and the MLE is the prime suspect.

### The maximum-likelihood estimate

The **maximum-likelihood estimate** is the parameter value that maximizes the likelihood:

$$ \hat\theta = \arg\max_{\theta}\ L(\theta). $$

The "$\arg\max$" means "the *argument* (the $\theta$) that produces the maximum," not the maximum value itself. We do not care how big $L$ gets; we care *where* it peaks. That location is our best-fit parameter.

![Before-and-after contrast of a bad-fit narrow curve assigning low probability versus the MLE fit assigning high probability](/imgs/blogs/mle-method-of-moments-math-for-quants-2.png)

The figure above shows the move in pictures. On the left, a badly chosen curve — too narrow — leaves a lot of the data stranded out in the thin tails where the density is almost zero, so the product of densities is microscopic: low likelihood. On the right, the maximum-likelihood curve has been widened until the data clusters near the fat middle of the bell, where the density is tall, so the product is as large as it can be: high likelihood. Picture the optimizer sliding and stretching the curve until the observed dots sit as high up on it as possible. That sliding-and-stretching is literally what maximizing the likelihood does, and it is why a too-narrow risk model — one that thinks the world is calmer than it is — gets brutally penalized the moment a real return lands in its neglected tail.

### Why this is the natural thing to do

There is a deep reason maximum likelihood is *the* default. Among all the ways you could fit a fully-specified model, it is the one that is asymptotically efficient — it squeezes the most information out of every data point, as we will prove informally in section 4. But even before the theory, the principle is just honest: of all the worlds your model could be in, choose the one in which what happened was most likely to happen. Any other choice is, in a precise sense, betting that you got unlucky with your data — and you should not assume you got unlucky without a reason.

## 2. The log-likelihood and why we take logs

The likelihood as written has two practical problems, and the fix for both is the same: take a logarithm.

### The first problem: products of tiny numbers

Each density value for continuous data can be small, and we multiply $n$ of them. For 250 daily returns you are multiplying 250 numbers each well below 1, and the product underflows to a literal zero in floating-point arithmetic long before $n$ gets large. The computer cannot even represent the likelihood, let alone maximize it. Multiplying is also clumsy for calculus.

The fix is to maximize the **log-likelihood** instead:

$$ \ell(\theta) = \log L(\theta) = \log \prod_{i=1}^{n} f(x_i;\theta) = \sum_{i=1}^{n} \log f(x_i;\theta). $$

The logarithm turns the product into a sum, because $\log(ab) = \log a + \log b$. A sum of 250 moderate negative numbers is perfectly well-behaved arithmetic. And here is the key fact that makes this legal: the logarithm is **monotonic** — it always increases as its input increases — so wherever $L(\theta)$ peaks, $\ell(\theta) = \log L(\theta)$ peaks at the *exact same* $\theta$. Maximizing the log-likelihood gives you the identical $\hat\theta$ as maximizing the likelihood, with none of the numerical pain.

![Stack showing the product of densities becoming a sum of log densities and then setting the derivative to zero](/imgs/blogs/mle-method-of-moments-math-for-quants-3.png)

The stacked layers above are the recipe. Start with the product of densities at the top. Take the log, and the product collapses into a sum of log-densities — the single most useful algebraic move in all of estimation. With a sum, calculus becomes easy: to find the peak of a smooth function, take its derivative with respect to $\theta$, set that derivative to zero, and solve. The derivative-equals-zero condition is called the **likelihood equation**, and its solution is $\hat\theta$. Under the hood, this is just "find the top of the hill by finding where the slope is flat," done one parameter at a time.

### The second problem: the answer should not depend on units

A subtle bonus: because logs turn scaling into adding a constant, and adding a constant does not move the location of a maximum, the log-likelihood's *peak* is robust to a lot of harmless rescaling. This is why practitioners almost never speak of "the likelihood" in practice — they speak of "the log-likelihood," report its value at the optimum (a single negative number you will see printed by every fitting library), and compare models by differences in it.

#### Worked example: MLE of the mean and variance of normal returns

This is the canonical derivation every quant should be able to reproduce, so we will do it in full and then plug in numbers. Suppose your $n$ daily returns are i.i.d. normal with unknown mean $\mu$ and unknown variance $\sigma^2$. Write down the log-likelihood by plugging the normal density into the sum:

$$ \ell(\mu,\sigma^2) = \sum_{i=1}^{n}\left[ -\tfrac{1}{2}\log(2\pi) - \tfrac{1}{2}\log\sigma^2 - \frac{(x_i-\mu)^2}{2\sigma^2}\right]. $$

The first term is a constant that does not involve the parameters, so it cannot affect *where* the maximum is — we can ignore it. To find the best $\mu$, take the derivative of $\ell$ with respect to $\mu$ and set it to zero. The only term involving $\mu$ is the last one, and its derivative is $\sum_i (x_i - \mu)/\sigma^2$. Setting that to zero gives $\sum_i (x_i - \mu) = 0$, which rearranges to:

$$ \hat\mu = \frac{1}{n}\sum_{i=1}^{n} x_i = \bar x. $$

The maximum-likelihood estimate of the mean is just the sample average. Now take the derivative with respect to $\sigma^2$ and set it to zero; after a line of algebra you get:

$$ \hat\sigma^2 = \frac{1}{n}\sum_{i=1}^{n}(x_i - \hat\mu)^2. $$

The MLE of the variance is the average squared deviation from the sample mean. (Note the divisor is $n$, not $n-1$; the MLE of variance is slightly biased downward, a point we will return to — the $n-1$ "sample variance" is the bias-corrected cousin, covered in the [estimators post](/blog/trading/quantitative-finance/estimators-mle-bias-variance-quant-interviews).)

Now the numbers. Take the five returns from earlier, in percent: $+1.0,\ -0.5,\ +0.5,\ +2.0,\ -0.5$. The sample mean is $\hat\mu = (1.0 - 0.5 + 0.5 + 2.0 - 0.5)/5 = 2.5/5 = 0.5\%$. The squared deviations are $(0.5)^2, (-1.0)^2, (0)^2, (1.5)^2, (-1.0)^2 = 0.25, 1.0, 0, 2.25, 1.0$, summing to $4.5$. So $\hat\sigma^2 = 4.5/5 = 0.9\,(\%^2)$ and $\hat\sigma = \sqrt{0.9} \approx 0.95\%$. On a \$100,000 position, a one-day move of one estimated standard deviation is about \$950. That \$950 is the headline daily-risk number this fit produces, and it came straight out of maximizing the log-likelihood. The one-sentence intuition: for normal data, maximum likelihood quietly hands you back the two statistics you already know — the average and the average squared deviation — which is why the bell curve is the friendliest model to fit.

## 3. The method of moments: the fast shortcut

Maximum likelihood is the gold standard, but it is not always easy — sometimes the likelihood equation has no closed-form solution and you need a computer to crawl up the hill. There is an older, simpler idea that often gets you 90% of the way in one line, and it is the workhorse for *starting values* and *quick calibration*. It is called the **method of moments**.

### The idea: match averages to averages

A **moment** is an average of a power of the data. The first moment is the mean $E[X]$, the second moment is $E[X^2]$, and so on; we built these carefully in the [moments post](/blog/trading/math-for-quants/expectation-variance-moments-math-for-quants). Every parametric model has *theoretical* moments — formulas for $E[X]$, $E[X^2]$, etc. in terms of the parameters. And every sample has *empirical* moments — you just compute the averages from the data. The method of moments says: set the theoretical moments equal to the empirical ones, and solve the resulting equations for the parameters.

The plain-English version: if your model says "the mean should be $\mu$ and the variance should be $\sigma^2$," and your data has an average of 0.5% and a spread of 0.9, then *declare* $\mu = 0.5\%$ and $\sigma^2 = 0.9$ and move on. You match what the model predicts on average to what you actually observed on average. For the normal, this gives the same answer as MLE — but for most other distributions it does not, and that gap is the price of the shortcut.

### The recipe

To estimate $k$ parameters, you need $k$ equations, so you match the first $k$ moments. For the $j$-th moment:

$$ \frac{1}{n}\sum_{i=1}^{n} x_i^{\,j} = E_\theta[X^j], \qquad j = 1, 2, \ldots, k. $$

The left side is computed from data; the right side is a formula in $\theta$. Solve the system for $\theta$ and you have your method-of-moments estimate. It is almost always faster than MLE and frequently has a closed form even when the MLE does not.

![Matrix comparing method of moments against maximum likelihood across idea speed efficiency and best use](/imgs/blogs/mle-method-of-moments-math-for-quants-4.png)

The matrix above lays the two methods side by side. The method of moments matches sample moments to theoretical ones, runs fast in closed form, but is usually *less efficient* — its estimates bounce around more from sample to sample than the MLE's. Maximum likelihood maximizes the full likelihood, often needs a numerical optimizer, but achieves the smallest possible variance. The practical workflow most quants use is exactly the diagram's bottom row: compute a method-of-moments estimate first to get sensible starting values, then hand those to the optimizer that polishes them into the MLE. The shortcut and the gold standard are partners, not rivals.

#### Worked example: method of moments for a Student-t — getting the degrees of freedom from kurtosis

Real returns have **fat tails**: extreme moves happen far more often than a normal curve predicts. The standard fix is the **Student-t distribution**, a bell-shaped curve with an extra parameter $\nu$ (the Greek "nu"), the **degrees of freedom**, that controls tail fatness. Small $\nu$ means very fat tails; as $\nu \to \infty$ the t-distribution becomes the normal. We want to estimate $\nu$ from data — and the method of moments gives a famously clean answer through **kurtosis**, the fourth standardized moment that measures tail fatness (a normal has kurtosis exactly 3; anything above 3 is fat-tailed).

The theoretical excess kurtosis (kurtosis minus 3) of a Student-t with $\nu > 4$ degrees of freedom is:

$$ \text{excess kurtosis} = \frac{6}{\nu - 4}. $$

The method of moments says: compute the sample excess kurtosis from your returns, set it equal to this formula, and solve for $\nu$. Suppose you measure a sample excess kurtosis of $2.0$ in a stock's daily returns (entirely typical — equities routinely run between 1 and 6). Set $6/(\nu-4) = 2.0$, so $\nu - 4 = 6/2.0 = 3$, giving $\hat\nu = 7$. Your returns behave like a t-distribution with 7 degrees of freedom — clearly fat-tailed, but not wildly so. Why does this matter in dollars? Under a normal, a 4-standard-deviation daily loss has probability about 1 in 16,000 — call it "never." Under a t with 7 degrees of freedom, the same 4-sigma loss is roughly 10 times more likely. On a \$1,000,000 book where one sigma is \$15,000, that 4-sigma day is a \$60,000 loss, and the t-distribution says to budget for it ten times as often as the normal does. The one-sentence intuition: the method of moments lets you read a distribution's tail-fatness parameter straight off the sample kurtosis with a single division, which is why it is the first thing a risk desk reaches for.

### When the shortcut breaks

The method of moments has two well-known failure modes. First, **higher moments are noisy**: the fourth moment depends on the data raised to the fourth power, so a single outlier can swing your kurtosis estimate dramatically, making the $\nu$ estimate jittery. Second, it can produce **impossible answers** — if the sample kurtosis comes out below 3 (thinner than normal), the formula $6/(\nu-4)$ has no valid positive solution, and you are stuck. These are exactly the cases where you fall back to maximum likelihood, which uses the whole shape of the data, not just two summary numbers, and is far more stable.

How big is the precision gap between the two methods in practice? It depends entirely on the parameter. For the mean of a normal, there is no gap at all — they are the same estimator. For the degrees-of-freedom of a Student-t, the gap is large, because the method-of-moments estimate leans on the fourth moment, which is statistically the worst-behaved thing you can estimate. A useful way to think about the cost is the **relative efficiency**: the ratio of the variances of the two estimators. If the MLE has variance $V_{\text{MLE}}$ and the method of moments has variance $V_{\text{MoM}}$, then a relative efficiency of, say, $V_{\text{MLE}}/V_{\text{MoM}} = 0.5$ means the method-of-moments estimate is twice as noisy — equivalently, you would need *twice the data* to get the same precision out of it. The table below summarizes when each method is the right tool.

| Situation | Use method of moments | Use maximum likelihood |
| --- | --- | --- |
| Mean and variance of normal returns | Yes — identical to MLE, instant | Yes — same answer |
| Degrees of freedom of a Student-t | Quick starting guess only | Final estimate (much lower variance) |
| GARCH parameters $\omega, \alpha, \beta$ | Seed values for the optimizer | The real fit (no closed form exists) |
| Tail index from extremes | Rarely — too few data points | Yes (Hill estimator is an MLE) |
| You need an answer in one line of code | Yes | No |

The honest rule of thumb most desks follow: use the method of moments to get a fast, sensible number and to seed the optimizer, then let maximum likelihood deliver the estimate you actually report and trade on.

## 4. Why trust the MLE? Consistency and efficiency

We have two recipes. Why is maximum likelihood the one the textbooks crown? Because it has three properties that, taken together, make it about as good as an estimator can be. We will build intuition for each, then state the one formula that ties them together.

![Tree of MLE properties branching into consistency asymptotic normality and efficiency with their consequences](/imgs/blogs/mle-method-of-moments-math-for-quants-5.png)

The tree above maps the three guarantees. Consistency means the estimate converges to the truth as data piles up. Asymptotic normality means its error is bell-shaped, so confidence intervals are easy. Efficiency means no (unbiased) estimator does better — and the bridge to all of it is a quantity called the **Fisher information**, which appears in the middle branch. Let us take them one at a time; together, these three properties are the reason "just use MLE" is sound advice 95% of the time.

### Consistency: it homes in on the truth

An estimator is **consistent** if, as the sample size $n$ grows, the estimate converges to the true parameter. Pour in enough data and the MLE will get arbitrarily close to the right answer. This is the bare minimum we demand of any estimator — an inconsistent one is broken — and the MLE delivers it under mild conditions. The everyday analogy: a consistent estimator is a scale that, while it might be a little off on any single weighing, has its readings tighten around your true weight the more times you step on it.

### Asymptotic normality: its error is bell-shaped

For large $n$, the distribution of the MLE's *error* (estimate minus truth) is approximately normal, centered at zero. This is a gift from the central limit theorem — covered in the [law of large numbers and CLT post](/blog/trading/math-for-quants/law-large-numbers-central-limit-theorem-math-for-quants) — and it is enormously practical: it means you can attach a standard error and a confidence interval to *any* MLE using the same normal-curve machinery, no matter how exotic the model. When a fitting library prints "$\hat\nu = 7.0\ (\pm 1.3)$," that $\pm 1.3$ is the standard error from this very property.

### The Fisher information: how much a data point tells you

The middle branch of the tree is the engine. The **Fisher information** $I(\theta)$ measures how much information a single observation carries about the parameter — equivalently, how sharply *peaked* the log-likelihood is around its maximum. A sharp, narrow peak means the data strongly prefers one parameter value: lots of information, precise estimate. A broad, flat peak means many parameter values fit nearly as well: little information, imprecise estimate. Formally, the Fisher information for one observation is the expected curvature of the log-likelihood:

$$ I(\theta) = -E\!\left[\frac{\partial^2 \log f(X;\theta)}{\partial \theta^2}\right]. $$

The second derivative is the curvature (how fast the slope changes); the minus sign makes it positive at a peak (where the curve bends down); the expectation averages over the data. You do not need to evaluate this by hand often — libraries do — but the *meaning* is what matters: more curvature equals more information equals a tighter estimate. For $n$ independent observations the information simply adds up to $n\,I(\theta)$, which is the formal reason more data means more precision.

A concrete way to see the link between curvature and precision: take two log-likelihood hills, both peaking at the same $\hat\theta$. One is a sharp, narrow spike; the other is a gentle, rounded mound. For the spike, moving $\theta$ even slightly away from the peak causes the log-likelihood to plummet — the data is *adamant* about which parameter value it prefers, so your estimate is precise. For the mound, you can wander a long way from the peak before the log-likelihood drops much — the data is *indifferent* among a wide band of parameter values, so your estimate is fuzzy. The Fisher information is literally the steepness of that hill at its top, and the standard error of the MLE is, for one observation, one over the square root of that steepness. This is why "how precise is my estimate?" and "how curved is the likelihood?" are the same question wearing different clothes.

### Efficiency: no one does it better

An estimator is **efficient** if it achieves the smallest possible variance among all unbiased estimators. The MLE is *asymptotically efficient*: for large $n$, nothing unbiased beats it. That is the headline result — it says maximum likelihood is not just a reasonable choice but, in a precise large-sample sense, the *optimal* one. The method of moments, by contrast, is generally not efficient: it throws away information by summarizing the data into a few moments rather than using its full shape, so its estimates have larger variance. That extra variance is the dollar cost of the shortcut, and section 5 makes it concrete.

## 5. The Cramér–Rao bound: the speed limit on precision

We can now answer the most practical question a risk manager ever asks of an estimate: "how precise can this *possibly* be?" There is a hard floor on the variance of any unbiased estimator, and it is one of the most beautiful results in statistics.

### The bound

The **Cramér–Rao lower bound** states that for any unbiased estimator $\hat\theta$ of $\theta$ from $n$ i.i.d. observations:

$$ \mathrm{Var}(\hat\theta) \ \ge\ \frac{1}{n\,I(\theta)}. $$

In words: the variance of *any* unbiased estimator is at least one over the total Fisher information. You cannot do better, no matter how clever your method — it is a speed limit, not a suggestion. And the punchline that ties this section to the last: the MLE *achieves* this bound for large $n$. That is precisely what "asymptotically efficient" means. The MLE drives right at the speed limit; the method of moments usually putters along below it.

The analogy: the Cramér–Rao bound is like the diffraction limit on a telescope. No matter how good your lens, physics caps how sharp an image you can form from a given amount of light. Here, no matter how good your estimator, mathematics caps how precise an estimate you can form from a given amount of data. More data — a bigger aperture — is the only way to push the limit.

### Reading the bound: more data, more precision

The bound makes the value of data quantitative. Variance falls like $1/n$, so the *standard error* (the square root of variance) falls like $1/\sqrt{n}$. To halve your standard error you need four times the data; to cut it by ten you need a hundred times the data. This $\sqrt{n}$ law is the reason precision is expensive and why quants are forever data-hungry — and why a strategy that needs three years of clean data to even *measure* its edge is a hard strategy to run.

The $\sqrt{n}$ law also explains a frustrating asymmetry between estimating *return* and estimating *risk*. The expected return is a first-moment quantity, and its standard error is $\sigma/\sqrt{n}$ — which, for a strategy with a Sharpe ratio around 1, means you need on the order of a *decade* of daily data before the estimated mean return is even two standard errors away from zero. Volatility, a second-moment quantity, is estimated far more precisely from the same data, because squared returns carry more information about spread than raw returns carry about the center. This is the deep reason every quant will tell you that *risk is knowable and return is not*: the Cramér–Rao bound is simply much kinder to the variance than to the mean. You can pin down how much a strategy will swing long before you can confirm whether it makes money — which is exactly why position sizing leans on volatility estimates and why claims of a measured "edge" deserve the same skepticism you would give any number with a giant error bar.

#### Worked example: Cramér–Rao — the smallest possible standard error for a volatility estimate

Here is the question that matters on a risk desk: if I estimate volatility from $n$ days of data, how precise can that estimate be at best? For normal returns with known mean, the Fisher information about $\sigma$ works out so that the Cramér–Rao bound gives a standard error for the volatility estimate of:

$$ \mathrm{SE}(\hat\sigma)\ \approx\ \frac{\sigma}{\sqrt{2n}}. $$

This is the floor — the MLE essentially achieves it. Let us put numbers on it. Suppose the true daily volatility is $\sigma = 1\%$ and you have $n = 250$ trading days (one year). The best-possible standard error is $1\%/\sqrt{2 \times 250} = 1\%/\sqrt{500} = 1\%/22.4 \approx 0.045\%$. So your volatility estimate is good to about $\pm 0.045\%$ on a true $1\%$ — roughly a $\pm 4.5\%$ *relative* error. On a \$1,000,000 book, a $1\%$ daily vol means about \$10,000 of one-sigma daily risk, and the $\pm 4.5\%$ uncertainty in vol translates to roughly $\pm\$450$ of uncertainty in that risk number — small but not nothing, and it is the *best you can do* with a year of data.

Now turn the question around: how much data do you need to know vol to within $5\%$ relative error? Set $\sigma/\sqrt{2n} = 0.05\,\sigma$, so $\sqrt{2n} = 20$, $2n = 400$, $n = 200$ days. About 200 trading days — under a year — and no method, however sophisticated, can do it with less. The one-sentence intuition: the Cramér–Rao bound converts "how confident should I be in this volatility number?" into a precise data-budget, and the $\sqrt{n}$ in the denominator is why doubling your confidence costs you four times the history.

## 6. Calibrating GARCH by maximum likelihood

So far our examples assumed returns were i.i.d. — drawn fresh and independent each day from the same distribution. Real markets violate this in one glaring way: **volatility clusters**. Calm days follow calm days; wild days follow wild days. The model built to capture this is **GARCH**, and fitting it is the most important real-world application of maximum likelihood in all of quant risk.

### What GARCH is, in plain English

GARCH stands for Generalized Autoregressive Conditional Heteroskedasticity — a mouthful that unpacks into one simple idea: *today's variance depends on yesterday's surprise and yesterday's variance.* "Heteroskedasticity" just means "changing variance" (as opposed to constant). The everyday analogy: volatility has momentum, like a choppy sea. A big wave (a large return, up or down) tells you the next few waves will probably be big too; a flat calm tells you to expect more calm. GARCH writes that momentum down as a formula.

The most common version, GARCH(1,1), models the variance $\sigma_t^2$ on day $t$ as:

$$ \sigma_t^2 = \omega + \alpha\, r_{t-1}^2 + \beta\, \sigma_{t-1}^2. $$

Here $r_{t-1}$ is yesterday's return, $\sigma_{t-1}^2$ was yesterday's variance, and the three parameters are $\omega$ (omega, a baseline floor for variance), $\alpha$ (alpha, how much yesterday's *surprise* $r_{t-1}^2$ feeds into today), and $\beta$ (beta, how much yesterday's *variance* persists into today). The "(1,1)" means one lag of each. The whole personality of a market lives in $\alpha$ and $\beta$: a high $\beta$ means volatility is sticky and slow to fade; a high $\alpha$ means it reacts violently to shocks.

### Why MLE is the only sensible way to fit it

Notice the problem: the variance $\sigma_t^2$ is not directly observed — it is built recursively from the parameters and the past returns. You cannot just "average" your way to $\omega$, $\alpha$, $\beta$ the way the method of moments would. You need a method that scores an *entire trajectory* of conditional variances against the data, and that is exactly what the likelihood does. Assuming each day's return is normal given its conditional variance, the log-likelihood is:

$$ \ell(\omega,\alpha,\beta) = -\frac{1}{2}\sum_{t=1}^{n}\left[\log(2\pi) + \log\sigma_t^2 + \frac{r_t^2}{\sigma_t^2}\right], $$

with each $\sigma_t^2$ computed from the recursion above using the candidate parameters. There is no closed-form solution; you maximize this numerically.

![Pipeline of GARCH calibration looping from a parameter guess through the variance recursion and log-likelihood back to the optimizer until the maximum is found](/imgs/blogs/mle-method-of-moments-math-for-quants-6.png)

The pipeline above is the actual calibration loop, and it is worth tracing because it is how every GARCH library works under the hood. Feed in the return series. Make an initial guess for $(\omega, \alpha, \beta)$ — often from the method of moments, to start somewhere sensible. Run the variance recursion forward through the whole series to build every $\sigma_t^2$. Plug those into the log-likelihood and sum it up — that is your "score" for this guess. If it is not yet maximal, the numerical optimizer nudges the parameters and the loop repeats, re-running the recursion each time. When the score stops improving, you have the maximum-likelihood parameters. The loop in the diagram is the optimizer climbing the likelihood hill, one trial set of knobs at a time.

#### Worked example: fitting GARCH(1,1) and forecasting tomorrow's risk

Let us walk a realistic calibration conceptually and land on a dollar number. You feed two years (about 500 days) of daily equity returns into a GARCH(1,1) fit. The optimizer maximizes the log-likelihood and returns, say, $\hat\omega = 0.000002$, $\hat\alpha = 0.08$, $\hat\beta = 0.90$. These are typical equity values: $\alpha + \beta = 0.98$, very close to 1, meaning volatility is highly persistent (shocks fade slowly over weeks, not days).

First, the **long-run variance** the model implies — the level volatility reverts to — is $\omega/(1 - \alpha - \beta) = 0.000002/(1 - 0.98) = 0.000002/0.02 = 0.0001$, so long-run daily vol is $\sqrt{0.0001} = 0.01 = 1\%$. Now suppose yesterday was a rough day: yesterday's return was $r_{t-1} = -3\%$ (so $r_{t-1}^2 = 0.0009$) and yesterday's variance estimate was $\sigma_{t-1}^2 = 0.0004$ (a 2% vol day). Today's forecast variance is:

$$ \sigma_t^2 = 0.000002 + 0.08(0.0009) + 0.90(0.0004) = 0.000002 + 0.000072 + 0.000360 = 0.000434. $$

So today's forecast volatility is $\sqrt{0.000434} \approx 0.0208 = 2.08\%$ — elevated, because yesterday's $-3\%$ shock fed in through $\alpha$. On a \$1,000,000 position, a one-day 99% Value-at-Risk under a normal is about $2.33 \times \sigma_t \times \$1{,}000{,}000 = 2.33 \times 0.0208 \times \$1{,}000{,}000 \approx \$48{,}500$. Compare that with a calm day forecasting back near the 1% long-run vol, where the same VaR would be about \$23,300. The model has more than doubled the capital it wants you to hold against this position overnight, purely because it saw yesterday's shock — and that responsiveness, calibrated by maximum likelihood, is the entire reason desks run GARCH instead of a flat historical volatility. The one-sentence intuition: maximum likelihood is what lets a volatility model *learn* the persistence of fear from the data, so its risk forecast leans into turbulent periods instead of being caught flat-footed.

### QMLE: robustness when the normal assumption is wrong

We assumed returns were conditionally normal to write the GARCH likelihood. But returns are fat-tailed even after accounting for clustering, so that assumption is technically wrong. Remarkably, the GARCH parameters you get by maximizing the *normal* likelihood are still consistent even when the true distribution is not normal — this is called **Quasi-Maximum Likelihood Estimation (QMLE)**. You use the normal likelihood as a convenient scoring function; it gives you the right $\alpha$ and $\beta$ on average even though the normal is the wrong shape, as long as the variance recursion is correctly specified. You do have to adjust the *standard errors* (the normal understates the uncertainty), but the point estimates survive. QMLE is one of the most useful robustness results a practitioner can know: it means you do not have to get the distribution exactly right to get the volatility dynamics right.

The mechanism behind QMLE is worth a sentence, because it explains why the trick works and when it fails. The normal log-likelihood, for a fixed variance recursion, is maximized by matching the model's variance to the data's realized squared returns — and that matching depends only on the *first two moments* of the data being correctly modeled, not on its full shape. So as long as your GARCH recursion gets the conditional variance right, the fact that the true tails are fat (a property of the third and fourth moments) does not bias the variance parameters. The catch is the standard errors: the formula $1/(n I(\theta))$ assumes the normal is the true model, and when it is not, you must replace it with a "sandwich" estimator that uses the *actual* curvature observed in the data. In practice, every serious GARCH library reports these robust standard errors by default — if yours does not, the error bars it prints are too optimistic, often by 30% or more, which can make a parameter look statistically significant when it is not.

## 7. Overfitting, local maxima, and QMLE

Maximum likelihood is powerful, and like all powerful tools it has sharp edges. Three of them cut quants regularly.

### Overfitting: when more parameters make forecasts worse

The likelihood can *always* be increased by adding parameters — a richer model can hug the historical data more tightly. But hugging the past is not the goal; *forecasting the future* is. Past a point, extra parameters stop capturing real structure and start memorizing the noise in your particular sample, and an estimator that has memorized noise forecasts the future *worse*, not better. This is **overfitting**, and it is the central tension in all model fitting.

![Before-and-after contrast of an underfit model with too few parameters versus an overfit model with too many that forecasts poorly out of sample](/imgs/blogs/mle-method-of-moments-math-for-quants-7.png)

The contrast above frames the tradeoff. On the underfit side, a model with too few parameters misses real features — for instance, a plain normal model that ignores fat tails and so under-budgets for crashes. On the overfit side, a model with too many parameters chases the wiggles of one historical sample; it scores a gorgeous in-sample likelihood and then forecasts poorly the moment new data arrives. The job is to land between them. Practitioners pick the sweet spot using information criteria — the **AIC** (Akaike Information Criterion) and **BIC** (Bayesian Information Criterion) both take the maximized log-likelihood and *subtract a penalty for each parameter*, so a parameter only earns its place if it improves the fit by more than its penalty. Out-of-sample testing — fitting on one period and scoring on a later, untouched period — is the ultimate referee.

#### Worked example: when one extra parameter is not worth it

You fit two volatility models to the same 500 days. Model A (GARCH(1,1), 3 parameters) achieves a maximized log-likelihood of $1{,}820$. Model B (a fancier GARCH variant with 5 parameters) achieves $1{,}822$ — two points higher, because more parameters always fit at least as well. Is B worth it? The AIC is $-2\ell + 2k$, where $\ell$ is the log-likelihood and $k$ is the parameter count (lower AIC is better). For A: $-2(1820) + 2(3) = -3640 + 6 = -3634$. For B: $-2(1822) + 2(5) = -3644 + 10 = -3634$. A dead tie on AIC — the two extra parameters bought exactly enough fit to cover their penalty and not a basis point more. Under the stricter BIC, which penalizes parameters by $\log(n)$ each — here $\log(500) \approx 6.2$ — Model B's penalty is far heavier and Model A wins outright. The dollar consequence: if you chase Model B's two-point likelihood edge, you are very likely shipping a risk model that will misforecast on next month's data and leave you either over-reserving capital (a real cost) or under-reserving it (a worse one). The one-sentence intuition: a higher likelihood is necessary but never sufficient — a parameter has to beat its complexity penalty, or it is just an expensively dressed-up way to memorize the past.

### Local maxima: the optimizer climbing the wrong hill

Numerical optimizers find a peak by climbing uphill from a starting point. If the likelihood surface has more than one peak — a **local maximum** that is lower than the true **global maximum** — the optimizer can get stuck on a false summit and report a confident, wrong answer. GARCH likelihoods are mostly well-behaved, but more elaborate models (regime-switching, multi-asset, fat-tailed GARCH) routinely have multiple peaks. The standard defenses: start the optimizer from several different points (including a method-of-moments estimate) and keep the best result; use sensible parameter bounds (e.g. force $\alpha, \beta \ge 0$ and $\alpha + \beta < 1$ so variance stays finite and positive); and sanity-check the fitted parameters against economic intuition. An $\hat\alpha + \hat\beta$ of exactly 1.0, or a $\hat\beta$ near zero, is usually the optimizer telling you it got lost.

### Numerical conditioning

Even with one peak, the optimization can be numerically delicate. Returns are tiny numbers (a 1% day is 0.01), and squaring and summing them produces variances around $0.0001$, which strains floating-point precision. Practitioners scale returns up (e.g. work in percent, so a 1% day is 1.0) before fitting, then scale the parameters back — a mundane trick that turns a finicky optimization into a stable one. This is the unglamorous reality of MLE in production: the math is clean, but the arithmetic needs care.

## Common misconceptions

**"Maximum likelihood gives the probability that my parameter is correct."** No. $L(\theta)$ is the probability of the *data* given the parameter, not the probability of the parameter given the data. Those are different objects (flipping them requires Bayes' theorem and a prior). The MLE is the parameter under which your data is most probable — it makes no direct claim about the probability of the parameter itself. Treating a likelihood as a probability-of-the-parameter is the single most common conceptual error in the whole subject.

**"The method of moments and maximum likelihood always agree."** They agree for the normal distribution (both give the sample mean and sample variance), which is exactly why beginners assume they always do. For almost any other distribution — Student-t, gamma, GARCH — they give different answers, and the MLE is generally the more precise one. The agreement for the normal is a special case, not the rule.

**"A higher likelihood always means a better model."** Only within a fixed parameter count. Across models with different numbers of parameters, likelihood always rewards complexity, so comparing raw likelihoods between a 3-parameter and a 5-parameter model is meaningless — the bigger model wins automatically and tells you nothing. You must penalize for parameters (AIC/BIC) or test out-of-sample. Likelihood ranks models of the same size; it does not rank models of different sizes.

**"The MLE is unbiased."** Not necessarily. The MLE of the variance divides by $n$, not $n-1$, so it is biased low in small samples (it systematically under-estimates variance). The MLE's superpower is *efficiency and consistency in large samples*, not unbiasedness in small ones. For small $n$ you often apply a bias correction by hand — the $n-1$ divisor being the classic example, detailed in the [estimators post](/blog/trading/quantitative-finance/estimators-mle-bias-variance-quant-interviews).

**"If the likelihood is maximized, the model fits well."** Maximizing the likelihood finds the best parameters *within* a model family; it says nothing about whether the family itself is right. You can fit the best-possible normal to wildly fat-tailed data and still have a terrible model — the best wrong model is still wrong. Always check residuals and tail behavior after fitting, not just whether the optimizer converged.

**"More data always fixes precision problems."** The Cramér–Rao bound shrinks like $1/n$, so precision does improve with data — but only at the $1/\sqrt{n}$ rate, and only if the data is i.i.d. and the model is correctly specified. If your model is mis-specified (wrong distribution, missing regime) or your data is non-stationary (the true parameters drift over time), piling on more history can make things *worse* by averaging together periods that should not be averaged. More data helps a correct model; it cannot rescue a wrong one.

## How it shows up in real markets

### 1. Daily VaR and the regulatory capital it drives

Every bank's market-risk desk calibrates a volatility model — very often GARCH or a close relative — to its trading positions, and the fitted $\sigma_t$ feeds directly into **Value-at-Risk**, the dollar figure regulators use to set capital. In our worked example, a single $-3\%$ shock pushed a \$1,000,000 position's 99% one-day VaR from about \$23,300 to about \$48,500. Multiply that across a multi-billion-dollar book and the maximum-likelihood fit is, very literally, deciding how many tens of millions of dollars the bank must set aside overnight. When the 2020 COVID crash hit, GARCH-style models calibrated by MLE were what made bank risk numbers spike within days — the $\alpha$ term feeding the giant March return-shocks straight into the variance forecast, exactly as the recursion prescribes.

### 2. The Student-t and the 1998 LTCM blow-up

Long-Term Capital Management famously sized positions using risk models that assumed thin, normal-ish tails. When Russia defaulted in August 1998, markets produced moves the normal model rated as essentially impossible — daily losses many standard deviations out, on multiple days. A Student-t fit, with its degrees-of-freedom parameter estimated honestly from data (whether by matching sample kurtosis or by MLE), would have assigned those moves vastly higher probability and demanded far more capital. The episode is the canonical lesson that *which distribution you fit* matters as much as *how well you fit it*: maximum likelihood on the wrong family is precise about the wrong thing.

### 3. Volatility surface calibration

Options market-makers fit pricing models to the entire grid of traded option prices — the **volatility surface** we cover in [its own deep dive](/blog/trading/quantitative-finance/volatility-surface). Calibration there is a likelihood-flavored optimization: choose the model parameters (of a stochastic-volatility model like Heston, say) that make the observed market prices most consistent with the model, typically by minimizing a weighted squared pricing error — which, under Gaussian pricing noise, is exactly a log-likelihood maximization in disguise. The same local-maxima and overfitting hazards from section 7 bite hard here: a surface model with too many free parameters fits today's quotes beautifully and reprices tomorrow's badly.

### 4. Fitting jump and regime-switching models

When a single GARCH cannot explain both quiet drift and sudden crashes, quants reach for regime-switching or jump-diffusion models, which carry many more parameters. These are fit by maximum likelihood — and they are textbook cases of the local-maxima problem. Practitioners run the optimizer from dozens of starting points and lean on method-of-moments estimates to seed it, precisely because a naive single run will often climb the wrong hill and report a "calm regime" that is really just a numerical artifact. The discipline of multi-start optimization is not academic fussiness; it is what keeps a \$100,000,000 strategy from being run off a mis-fit model.

### 5. The Hill estimator and tail-index estimation

Risk desks that care specifically about the extreme tail — the once-a-decade loss — often estimate a **tail index** that governs how fast the tail decays, using the Hill estimator, which is itself a maximum-likelihood estimator applied to the largest observations. It connects directly to the [tail-risk and extreme-value-theory post](/blog/trading/math-for-quants/tail-risk-extreme-value-theory-math-for-quants). The same Cramér–Rao logic applies with a vengeance: because tail events are rare *by definition*, the effective sample size is tiny, the Fisher information is small, and the standard error on a tail estimate is enormous — which is why honest tail-risk numbers come with embarrassingly wide error bars, and why anyone quoting a precise "1-in-1000-year loss" should be met with skepticism.

### 6. Backtesting and the data-mining trap

The overfitting hazard from section 7 has a market-wide version: with thousands of analysts fitting thousands of models to the same historical data, *someone* will find a model with a spectacular in-sample likelihood purely by chance. This is **data-mining bias**, and it is why a backtest's likelihood or Sharpe ratio is almost worthless without out-of-sample confirmation. The AIC/BIC penalties and out-of-sample discipline are the desk-level defenses; the firm-level defense is treating any strategy that has not survived live, forward trading as unproven, no matter how good its fitted likelihood looks.

## When this matters to you

If you ever fit *any* model to *any* data — a volatility estimate for position sizing, a distribution for a risk report, a forecasting model for a strategy — you are doing maximum likelihood or one of its cousins, whether you call it that or not. Understanding it changes three things about how you work. First, you will stop trusting a point estimate without its standard error: the Cramér–Rao bound tells you the irreducible uncertainty, and a volatility number without an error bar is half a number. Second, you will reach for the method of moments to get fast, sane starting values before you let an optimizer loose — and you will know that for the normal it is already the final answer. Third, and most important, you will treat a high likelihood with suspicion rather than celebration: you will ask whether it survived a parameter penalty and an out-of-sample test before you bet real capital on it.

A closing caution, because this is finance and not a statistics seminar: every number in this post — the \$950 daily risk, the \$48,500 VaR, the 7 degrees of freedom — is an *estimate*, conditional on the model being right and the past resembling the future. Both assumptions fail at the worst possible moments. Maximum likelihood is the best tool we have for turning data into parameters, but it cannot tell you that the regime is about to change or that your distribution is the wrong shape. Use it to quantify what the data says; never let it convince you the data has said everything. This is educational material, not investment advice.

**Further reading.** To go deeper, the natural next steps on this blog are the [estimators, bias, and variance deep dive](/blog/trading/quantitative-finance/estimators-mle-bias-variance-quant-interviews) for the small-sample behavior of $\hat\theta$, the [distributions cheat sheet](/blog/trading/quantitative-finance/distributions-cheat-sheet-quant-interviews) for the menu of densities you might fit, the [moments deep dive](/blog/trading/math-for-quants/expectation-variance-moments-math-for-quants) for the moments the method-of-moments matches, the [law of large numbers and CLT post](/blog/trading/math-for-quants/law-large-numbers-central-limit-theorem-math-for-quants) for why the MLE's error is bell-shaped, and the [volatility surface deep dive](/blog/trading/quantitative-finance/volatility-surface) for calibration as likelihood in the wild. Off-blog, the original sources are worth the climb: R. A. Fisher's 1922 paper introducing maximum likelihood, and Tim Bollerslev's 1986 paper introducing GARCH — both are landmarks, and both are more readable than their reputations suggest.
