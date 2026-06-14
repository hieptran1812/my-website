---
title: "Hypothesis testing and p-values for quant interviews"
date: "2026-06-14"
publishDate: "2026-06-14"
description: "Build hypothesis testing from zero, learn what a p-value actually means and the many ways people misread it, and master the multiple-testing trap that makes most discovered trading strategies fake — the exact statistics top quant desks grill you on."
tags:
  [
    "hypothesis-testing",
    "p-values",
    "quant-interviews",
    "statistical-significance",
    "multiple-testing",
    "false-discovery-rate",
    "type-i-error",
    "statistical-power",
    "confidence-intervals",
    "deflated-sharpe-ratio",
    "quantitative-finance",
    "backtesting"
  ]
category: "trading"
subcategory: "Quantitative Finance"
author: "Hiep Tran"
featured: true
readTime: 44
---

> [!important]
> **TL;DR** — Deciding whether a result is real or noise is the daily question behind every backtest, and interviews test whether you understand it deeply rather than as a recipe.
>
> - **A hypothesis test is one decision loop**: assume nothing works (the *null hypothesis*), measure how surprising the data would be if that were true (the *test statistic*), and only then decide to keep or discard the edge.
> - **A p-value is the probability of data this extreme IF the null is true** — it is emphatically *not* the probability the null is true, not the chance you are wrong, and not the size of the effect. Confusing these is the single most common way candidates fail.
> - **Two errors, two costs**: a *type I error* (false positive) is deploying a strategy that does not work; a *type II error* (false negative) is killing one that does. The significance level $\alpha$ caps the first; *power* fights the second.
> - **Multiple testing is the trap that matters most on a desk**: run 20 independent backtests at $\alpha = 0.05$ and the chance of at least one false "winner" is **64%**; at 100 tests it is over **99%**. Almost every "discovered" strategy is this artifact.
> - **The fixes have names interviewers want to hear**: *Bonferroni* (divide $\alpha$ by the number of tests) controls the chance of any false positive; *Benjamini–Hochberg* controls the *false discovery rate* and keeps more real edges.
> - **The one number to remember**: a reported backtest Sharpe of $2.0$ found after trying 50 variations on two years of data can deflate to a *deflated Sharpe* near $0.6$ — honest, and usually not worth deploying.

Here is a question that has ended more quant interviews than almost any other, and it sounds harmless. You build a trading strategy. You backtest it. Over the last two years it made money on **58%** of days, with an average of **\$42** of profit per day. Is the strategy real, or did you just get lucky?

Most candidates answer with enthusiasm about the strategy. The interviewer is not asking about the strategy. They are asking whether you can tell the difference between *signal* and *noise* — because that distinction is the entire job. A market-maker who cannot tell a real edge from a random streak will confidently deploy garbage and lose money until they are fired. The firms that ask these questions — Jane Street, Two Sigma, Citadel, DE Shaw, Optiver, Jump, HRT — are not testing whether you memorized a formula. They are testing whether you can reason cleanly about *how much evidence a number actually carries*.

![Hypothesis testing is one decision loop run on every backtest: state the null, compute a statistic, get a p-value, decide](/imgs/blogs/hypothesis-testing-pvalues-quant-interviews-1.png)

The diagram above is the mental model for everything that follows. Every statistical test — no matter how fancy the name — runs the same four-step loop. You **state a null hypothesis** (the boring default: "this strategy has no edge, its true mean daily P&L is zero"). You **compute a test statistic** that measures how far your data sits from that boring default, scaled by how noisy the data is. You **turn that into a p-value** — the probability that pure chance, under the null, would produce a result at least this extreme. And you **decide**: if the data would be too surprising under "no edge," you reject the null and treat the edge as real; otherwise you shrug and call it noise.

We will build this from absolute zero. No statistics background is assumed — every term is defined the first time it appears, every concept gets a worked example with real numbers, and every interview-favorite trap is dissected. By the end you will be able to derive the answers yourself, *and* — the part that actually lands the offer — narrate your reasoning out loud the way an interviewer wants to hear it. A short note before we start: nothing here is investment advice. This is about the statistics of evidence, not a recommendation to trade anything.

## Foundations: the building blocks of a test

Before any clever puzzles, we need a small, precise vocabulary. Skip nothing here; the whole edifice rests on these five ideas. We will use one running example throughout: a strategy whose *daily profit-and-loss* — its **P&L**, the dollars you made or lost that day — we want to judge.

### The null hypothesis and the alternative

The **null hypothesis**, written $H_0$, is the skeptical default — the claim that *nothing interesting is happening*. For a trading strategy, the natural null is "the true average daily P&L is exactly \$0" — the strategy has no edge, and any profit you saw is luck. We write this as $H_0: \mu = 0$, where $\mu$ (the Greek letter "mu") is the **true** long-run mean daily P&L — the number you would converge to if you could trade the strategy forever. You never observe $\mu$ directly; you only see a finite sample of days and estimate it.

The **alternative hypothesis**, written $H_1$ (or $H_a$), is what you suspect might be true instead — "the strategy actually makes money," $H_1: \mu > 0$. The alternative can be **one-sided** ($\mu > 0$, you only care about profit) or **two-sided** ($\mu \neq 0$, the mean differs from zero in either direction). The choice matters and interviewers probe it, so hold that thought.

The asymmetry between $H_0$ and $H_1$ is the soul of the whole subject. **The null is presumed true until the data forces you to abandon it** — exactly like "innocent until proven guilty" in a courtroom. You do not set out to prove the strategy works; you set out to see whether the data is *incompatible enough* with "it doesn't work" that you are forced to conclude otherwise. This framing — guilty until proven innocent is the *wrong* one — saves you from a dozen interview mistakes.

### The test statistic

A **test statistic** is a single number that summarizes how far your data sits from what the null predicts, measured in units of noise. The generic shape is always the same:

$$\text{test statistic} = \frac{\text{observed effect} - \text{null value}}{\text{standard error of the effect}}$$

The numerator is "how big is the thing we see, relative to the boring default." The denominator — the **standard error** — is "how much would this estimate wobble from sample to sample just by chance." Dividing one by the other gives you a *signal-to-noise ratio*. A test statistic of $0.3$ says "the effect I see is small relative to the noise; unremarkable." A test statistic of $3.5$ says "the effect is three and a half standard errors away from the null; that almost never happens by chance." The whole game is turning a messy dollar number into this clean, unit-free signal-to-noise score.

### The significance level $\alpha$

Before you look at the data, you pick a threshold for how much surprise you will demand before abandoning the null. That threshold is the **significance level**, written $\alpha$ (the Greek letter "alpha"), and by deep convention it is usually $\alpha = 0.05$, i.e. 5%. It means: "I will tolerate a 5% chance of falsely rejecting the null when it is actually true." Choosing $\alpha$ is choosing how often you are willing to cry wolf.

![The null distribution clusters near zero and alpha is the shaded tail probability we agree to treat as too extreme to be noise](/imgs/blogs/hypothesis-testing-pvalues-quant-interviews-2.png)

The figure above is the picture every quant carries in their head. If the null is true, your test statistic does not land on a single value — it wanders around zero from sample to sample, tracing out the bell-shaped **null distribution** (the spread of statistics you would see across many imaginary repeats of the experiment, if $H_0$ held). Most of the time it lands near the middle. The two red tails, together covering 5% of the area, are the **rejection region**: values so far from zero that, *if the null were true*, you would almost never see them. The boundaries here sit at $t = \pm 1.96$ for a two-sided 5% test — a number worth memorizing, because it appears constantly. If your statistic lands in the blue middle (95% of the area), you **fail to reject** $H_0$. If it lands in a red tail, you **reject** $H_0$ and call the effect statistically significant.

### Putting the foundations together

Those five pieces — null, alternative, test statistic, significance level, and the null distribution with its rejection region — are the complete grammar of a hypothesis test. Everything else in this post is either (a) a specific recipe for computing the test statistic and its null distribution (the z-test, the t-test), (b) a way to read the result honestly (p-values, confidence intervals, power), or (c) what goes wrong when you run the loop many times (multiple testing). Let us start with the two ways a test can be wrong, because the costs are asymmetric and interviewers love the asymmetry.

## Type I and type II errors: the two ways to be wrong

A hypothesis test is a decision under uncertainty, and any decision under uncertainty can be wrong in exactly two ways. You can **reject a true null** — declare an edge that isn't there. Or you can **fail to reject a false null** — miss an edge that is there. These have names, costs, and probabilities you must be able to recite.

![A two-by-two of truth versus decision names the false positive and false negative that every test risks, with their probabilities](/imgs/blogs/hypothesis-testing-pvalues-quant-interviews-3.png)

The figure lays out the full landscape. The columns are the truth (which you never know): the null is true (no edge) or the null is false (real edge). The rows are your decision: reject $H_0$ (call it real) or fail to reject (call it noise).

- **Type I error (false positive):** the null is true — no edge — but you reject it and deploy the strategy anyway. The probability of this is exactly $\alpha$, the significance level you chose. On a desk, a type I error means you put real capital behind a coin flip. You *control* this probability directly by setting $\alpha$.
- **Type II error (false negative):** the null is false — there really is an edge — but you fail to reject it and walk away. The probability of this is written $\beta$ (the Greek letter "beta"). A type II error means you threw away a money-making strategy. You do *not* control $\beta$ directly; it depends on how big the real edge is and how much data you have.
- **The two correct cells**: rejecting a false null is a **true positive**, and its probability, $1 - \beta$, is the **power** of the test (the headline of a later section). Failing to reject a true null is a **true negative**, with probability $1 - \alpha$.

The asymmetry is the interview point. In most of classical statistics, type I errors are treated as worse — hence the strict $\alpha = 0.05$ — because declaring a false discovery pollutes the scientific record. But on a trading desk the cost balance shifts and is strategy-specific. Deploying a fake edge (type I) bleeds money slowly and is recoverable if you size small. Missing the one genuinely great strategy of the decade (type II) is an enormous opportunity cost. A sharp candidate says out loud: "Which error is more expensive *here* determines how I'd set $\alpha$ and how much data I'd demand."

#### Worked example: naming the error a trader just made

You backtest a momentum strategy, get a statistically significant result at $\alpha = 0.05$, and deploy $1{,}000{,}000$ of capital. Six months later it has lost money and you conclude the edge was never real. **Which error did you make, and what was its probability?**

Walk it carefully. The truth, it turns out, is that there was no edge: $H_0$ was true. Your decision was to reject $H_0$. Rejecting a true null is a **type I error** — a false positive. Its probability was capped at the $\alpha = 0.05$ you chose. So roughly 1 in 20 strategies that pass this exact bar with no real edge will sail through and lose you money. The intuition the interviewer wants: *the 5% was never a guarantee the strategy works — it was your self-imposed false-positive budget, and this strategy spent it.*

## What a p-value actually means (and what it does not)

We now reach the concept that interviewers weaponize. The p-value is simultaneously the most-used and most-misunderstood number in all of statistics. Get its definition exactly right and you will out-reason most candidates instantly.

### The exact definition

The **p-value** is the probability, *computed assuming the null hypothesis is true*, of observing a test statistic at least as extreme as the one you actually got. Read that twice. It is a conditional probability with the condition fixed at "$H_0$ is true." It answers one narrow question: **if there were truly no edge, how often would pure noise produce a result this striking or more?**

![A p-value is the tail area beyond your observed statistic under the null: how often noise alone would produce a result this extreme](/imgs/blogs/hypothesis-testing-pvalues-quant-interviews-5.png)

The picture makes it concrete. You ran your test and got a statistic of $t = 2.1$. Plant yourself in the world where the null is true, so the statistic is wandering around zero on that bell curve. The p-value is the *tail area* — the amber sliver — to the right of $2.1$ (and, for a two-sided test, the matching sliver on the left). Here that area is about $0.018$. In words: *if the strategy truly had no edge, only about 1.8% of the time would random noise hand you a result this good or better.* Because $0.018 < 0.05$, you reject the null: the data is too surprising under "no edge" to keep believing it.

A small p-value means the data is *surprising under the null*. A large p-value means the data is *unremarkable under the null* — exactly the kind of thing noise produces all the time. That is the whole interpretation.

### What the p-value is NOT — the four traps

Now the part that separates strong candidates from the pack. The p-value is **not** any of the following, and interviewers fish for each one.

1. **It is not the probability that the null hypothesis is true.** This is the cardinal error. The p-value is $P(\text{data this extreme} \mid H_0 \text{ true})$ — the data given the null. The thing people *want* it to be is $P(H_0 \text{ true} \mid \text{data})$ — the null given the data. Those two conditional probabilities are different quantities (flipping the condition is exactly Bayes' theorem), and conflating them is the same mistake as the disease-test base-rate fallacy. A p-value of $0.018$ does **not** mean "1.8% chance there's no edge." Look again at the figure: the "NOT $P(H_0$ is true$)$" annotation sits there precisely because this is the trap.

2. **It is not the probability you made a mistake.** "$p = 0.05$, so there's a 5% chance I'm wrong to reject" is false. Whether you are wrong depends on whether the null is *actually* true, which the p-value cannot tell you.

3. **It is not a measure of effect size or importance.** A microscopic, useless edge can have a tiny p-value if you have enough data; a large, valuable edge can have a big p-value if you have too little. *Significance is not magnitude.* "Statistically significant" answers "is it distinguishable from zero?" — not "is it big?" On a desk these come apart constantly: a strategy with a real but $0.3$ basis-point edge per trade is statistically detectable over millions of trades yet may not clear transaction costs.

4. **$p \geq 0.05$ does not prove the null is true.** Failing to reject is not the same as accepting. Absence of evidence is not evidence of absence — you may simply have too little data to see a real edge. We return to this under misconceptions because it is its own special trap.

#### Worked example: reading a p-value out loud, correctly

Your colleague runs a t-test on a new signal and reports $p = 0.03$. They say, "Great — there's only a 3% chance this signal is worthless, so I'm 97% confident it's real." **Find every error and state the correct interpretation.**

There are two errors packed into one sentence. First, "3% chance this signal is worthless" treats the p-value as $P(H_0 \text{ true} \mid \text{data})$ — the flipped, wrong conditional. Second, "97% confident it's real" compounds it by treating $1 - p$ as the probability of the alternative, which is also wrong. The correct reading is narrow and humble: *"If this signal truly had no edge, we'd see a result this strong only about 3% of the time. That's surprising enough to reject 'no edge' at the 5% level, so we'll treat it as a candidate edge — but the 3% says nothing directly about how probable the edge is, which also depends on how many signals we tested and how plausible an edge was to begin with."* That last clause — *how many signals we tested* — is the bridge to the multiple-testing section, and dropping it in casually is a strong interview signal.

## The z-test and the t-test: turning dollars into a decision

We have talked about test statistics in the abstract. Now we compute them. Two recipes cover the vast majority of interview questions about means: the **z-test** (when you know the noise level, or have so much data it doesn't matter) and the **t-test** (when you must estimate the noise from the same small sample — the realistic case).

### The z-test: when you know the standard deviation

Suppose you observe $n$ days of P&L with sample mean $\bar{x}$ (read "x-bar," the average of your observations) and you somehow *know* the true daily standard deviation $\sigma$ (the typical day-to-day swing). The **standard error of the mean** — how much $\bar{x}$ itself wobbles — shrinks with more data as

$$\text{SE} = \frac{\sigma}{\sqrt{n}}.$$

That $\sqrt{n}$ is one of the most important facts in all of quantitative finance: *to halve your uncertainty about the mean, you need four times as much data.* The z-statistic is then

$$z = \frac{\bar{x} - \mu_0}{\sigma / \sqrt{n}},$$

where $\mu_0$ is the null value (here $0$). Under the null, $z$ follows the standard normal distribution — the canonical bell curve with mean $0$ and standard deviation $1$. You compare $z$ to the critical values: reject a two-sided test at 5% if $|z| > 1.96$; reject a one-sided ("$\mu > 0$") test at 5% if $z > 1.645$.

#### Worked example: is the strategy's mean daily P&L greater than zero?

You have $n = 100$ trading days. The average daily P&L is $\bar{x} = 42$. Assume (for now) you know the true daily standard deviation is $\sigma = 200$. **Test $H_0: \mu = 0$ against the one-sided alternative $H_1: \mu > 0$ at $\alpha = 0.05$.**

Step 1 — standard error: $\text{SE} = \dfrac{\sigma}{\sqrt{n}} = \dfrac{200}{\sqrt{100}} = \dfrac{200}{10} = 20$. So even though daily swings are $200, the *average* over 100 days wobbles by only about $20.

Step 2 — the statistic: $z = \dfrac{\bar{x} - 0}{\text{SE}} = \dfrac{42}{20} = 2.1$. Your observed mean of \$42 sits $2.1$ standard errors above zero.

Step 3 — the decision: for a one-sided 5% test the critical value is $1.645$. Since $2.1 > 1.645$, you **reject** $H_0$. Equivalently, the one-sided p-value is the area beyond $z = 2.1$ under the standard normal, which is about $0.018$ — comfortably below $0.05$. (This is exactly the $t = 2.1$, $p = 0.018$ picture from the p-value figure; same numbers throughout.)

The intuition: *a \$42 daily average looks impressive, but the honest question is "\$42 compared to what wobble?" — and \$42 against a \$20 standard error is genuinely surprising under the no-edge null, so we treat the edge as real.*

### The t-test: when you must estimate the noise

The z-test cheats: it assumes you know $\sigma$. In reality you never do — you estimate it from the same sample, getting the **sample standard deviation** $s$. That extra uncertainty (you are now unsure of *both* the mean and the spread) means your statistic no longer follows the normal distribution. It follows **Student's t-distribution** with $n - 1$ **degrees of freedom** — a parameter, roughly "how much independent information you have to estimate the spread," equal here to your sample size minus one. The statistic looks identical except $s$ replaces $\sigma$:

$$t = \frac{\bar{x} - \mu_0}{s / \sqrt{n}}.$$

![For small samples the t-distribution has fatter tails than the normal, so it demands a bigger statistic before you may reject](/imgs/blogs/hypothesis-testing-pvalues-quant-interviews-6.png)

The figure shows why this matters. The solid curve is the normal; the dashed curve is the t-distribution with 5 degrees of freedom. The t-curve is shorter in the middle and **fatter in the tails** (the amber regions): because you are unsure of the volatility itself, the t-distribution honestly admits that extreme values are more probable than the normal would claim. The practical consequence: the critical value grows. For a two-sided 5% test the normal demands $|z| > 1.96$, but with only 5 degrees of freedom the t-test demands $|t| > 2.57$. **Small samples make you work harder to reject the null** — exactly right, because small samples carry less evidence. As $n$ grows, the t-distribution converges to the normal (by $n = 30$ or so they are nearly identical), which is why the z-test is fine for large samples.

#### Worked example: the same strategy with the noise estimated, not known

Take the same $n = 100$ days and $\bar{x} = 42$, but now you do not know $\sigma$ — you estimate it from the data and get sample standard deviation $s = 210$. **Redo the one-sided test at $\alpha = 0.05$.**

Step 1: $\text{SE} = \dfrac{s}{\sqrt{n}} = \dfrac{210}{10} = 21$.

Step 2: $t = \dfrac{42}{21} = 2.0$, with $n - 1 = 99$ degrees of freedom.

Step 3: with 99 degrees of freedom the t-distribution is already almost identical to the normal, so the one-sided critical value is about $1.66$ (versus $1.645$ for the normal). Since $2.0 > 1.66$, you **reject** $H_0$; the one-sided p-value is about $0.024$.

The intuition: *estimating the noise instead of knowing it costs you a little — the bar rose slightly and the statistic shrank a touch — but with 100 days the penalty is tiny. With only 5 or 10 days it would have been large, and that is precisely when overfit strategies fool people.* On a desk this is the warning behind every short-history backtest: the fewer days you have, the fatter the tails, and the more a flashy result is just the t-distribution's wide tails doing their job.

### Two-sample tests: comparing two strategies

Often the question is not "does this strategy beat zero?" but "does strategy A beat strategy B?" That is a **two-sample test**. If the two samples are independent, the statistic compares the *difference* of means to the standard error of that difference:

$$t = \frac{\bar{x}_A - \bar{x}_B}{\sqrt{\frac{s_A^2}{n_A} + \frac{s_B^2}{n_B}}}.$$

The denominator just combines the two wobbles. When the comparison is *paired* — the same days, two strategies — you instead test whether the per-day **difference** has mean zero, which is more powerful because it cancels out market-wide moves that hit both strategies. Interviewers like asking which design you would use; the answer is almost always "paired if I can, because it removes the common noise."

## Confidence intervals and their duality with tests

A p-value gives a yes/no verdict. A **confidence interval** gives a *range* — and it turns out to carry strictly more information while being the *same* mathematical object as a test. Understanding the duality is a frequent interview discriminator.

A **95% confidence interval** for the mean is a range computed from your data such that, *if you repeated the whole experiment many times, 95% of the intervals you construct this way would contain the true mean $\mu$*. Note the subtlety the interviewer is testing: the randomness is in the *interval*, not in $\mu$. The true mean is a fixed (unknown) number; it is your interval that jiggles from sample to sample. The standard recipe for a 95% interval on a mean is

$$\bar{x} \pm 1.96 \cdot \text{SE} \quad(\text{using } 1.96 \text{ for the normal; the matching } t\text{-critical value for small } n).$$

![A 95 percent confidence interval contains exactly the null values a two-sided 5 percent test would fail to reject](/imgs/blogs/hypothesis-testing-pvalues-quant-interviews-7.png)

Here is the duality, and the figure shows it cleanly. **The 95% confidence interval is exactly the set of null values that a two-sided 5% test would fail to reject.** A hypothesized mean *inside* the green interval is one you cannot reject (the data is compatible with it); a hypothesized mean *outside* the interval, in a red zone, is one you reject. Containment and non-rejection are the same fact viewed two ways. So a confidence interval silently runs *every possible test at once*: it tells you not just whether you reject "$\mu = 0$" but whether you'd reject "$\mu = 20$," "$\mu = 90$," and so on.

#### Worked example: build a 95% confidence interval for a strategy's mean

Same strategy: $\bar{x} = 42$, $\text{SE} = 15$ (suppose a slightly different sample gives this standard error). **Construct the 95% confidence interval and use it to test $H_0: \mu = 20$ and $H_0: \mu = 90$.**

The interval is $42 \pm 1.96 \times 15 = 42 \pm 29.4$, i.e. roughly $(12, 72)$ — the green band in the figure. Now read off two tests for free:

- $H_0: \mu = 20$. Is $20$ inside $(12, 72)$? Yes. So you **fail to reject** "$\mu = 20$" at the 5% level — the data is compatible with a true mean of \$20.
- $H_0: \mu = 90$. Is $90$ inside $(12, 72)$? No, it is in the red zone. So you **reject** "$\mu = 90$" — the data is incompatible with a mean that large.

Crucially, $0$ is *not* in $(12, 72)$ either, so you also reject "$\mu = 0$" — the strategy is significantly profitable at 5%. The intuition: *a confidence interval is a Swiss-army test. Instead of asking one yes/no question it shows you the whole range of edges the data can and cannot rule out, which is far more useful when you actually have to size a position.*

#### Worked example: a 95% confidence interval for a win-rate

A different flavor: you want a confidence interval for a *proportion*, like a win-rate. Your strategy won on $58$ of its last $100$ days, so the observed win-rate is $\hat{p} = 0.58$. **Build a 95% confidence interval for the true win-rate $p$.**

For a proportion, the standard error is $\text{SE} = \sqrt{\dfrac{\hat{p}(1 - \hat{p})}{n}}$. Plug in: $\text{SE} = \sqrt{\dfrac{0.58 \times 0.42}{100}} = \sqrt{\dfrac{0.2436}{100}} = \sqrt{0.002436} \approx 0.0494$. The 95% interval is $0.58 \pm 1.96 \times 0.0494 = 0.58 \pm 0.097$, i.e. roughly $(0.483, 0.677)$.

Now the test: is $0.50$ — the "fair coin, no edge" win-rate — inside the interval? Yes, $0.50$ sits inside $(0.483, 0.677)$. So despite winning 58% of days, you **cannot** reject "this strategy is a coin flip" at the 5% level with only 100 days. The intuition that stuns candidates: *a 58% win-rate over 100 days is not statistically distinguishable from a coin. You would need far more data — or a much higher win-rate — before the interval clears 50% and the edge becomes provable.* This is exactly why short backtests with "good-looking" win-rates are so dangerous.

## Statistical power and sample size

We have spent most of our attention on the type I error, controlled by $\alpha$. The neglected twin is the type II error, $\beta$, and its complement **power** $= 1 - \beta$, the probability of correctly detecting a real edge when one exists. Power is what separates a serious quant from someone who just runs tests until something passes.

![Power rises with effect size: at a fixed sample size a tiny true edge is nearly invisible while a large one is almost always caught](/imgs/blogs/hypothesis-testing-pvalues-quant-interviews-8.png)

Power is not one number — it is a curve. The figure plots power against the true effect size (how big the real edge is, in standardized units). When the true edge is tiny, power is barely above $\alpha$: a real-but-small edge is almost invisible and you will usually miss it (a type II error, the red underpowered zone). As the true edge grows, power climbs toward $1$: a large edge is caught nearly every time. The conventional target — the dashed line — is **power $= 0.80$**, meaning an 80% chance of detecting the effect if it is real. Below that you are "underpowered," and underpowered tests are quietly toxic: they not only miss real edges, they also make the edges they *do* flag look bigger than they are (because only the lucky-large estimates clear the bar).

Power depends on four things, locked in a tradeoff: the significance level $\alpha$, the true effect size, the noise $\sigma$, and the sample size $n$. Fix any three and the fourth is determined. The most common interview version: *given an edge I care about, how much data do I need to detect it reliably?*

The picture of why power rises is the two-overlapping-curves figure from earlier:

![Type I and type II errors live in two overlapping distributions: more separation between no-edge and real-edge means more power](/imgs/blogs/hypothesis-testing-pvalues-quant-interviews-4.png)

The left curve is the null (no edge); the right curve is the world where the edge is real. The critical value (the vertical line) is set by $\alpha$ using the *null* curve — the red area beyond it is your false-positive rate $\alpha$ (type I). The **power is the area of the right curve beyond that same line**; the amber area to its *left* is $\beta$ (type II), the real edges that fall short of the bar and get missed. Now you can see every lever at once. Slide the right curve further right (a bigger true edge), and more of it clears the line — power rises, $\beta$ falls. Shrink the curves' width (more data, smaller standard error), and the same shift in means gives cleaner separation — power rises again. *Power is just how much of the real-edge distribution sits in the rejection region.*

#### Worked example: the power to detect a \$5/day edge

Your strategy, if real, makes $\mu = 5 per day. Daily noise is $\sigma = 50. You will run a one-sided 5% test on $n = 400$ days. **Roughly what is your power to detect this edge?**

Step 1 — standard error: $\text{SE} = \dfrac{\sigma}{\sqrt{n}} = \dfrac{50}{\sqrt{400}} = \dfrac{50}{20} = 2.5$.

Step 2 — the rejection threshold, in dollars. A one-sided 5% test rejects when the statistic exceeds $1.645$, i.e. when $\bar{x} > 1.645 \times \text{SE} = 1.645 \times 2.5 \approx 4.11$. So you reject whenever the observed mean beats about \$4.11/day.

Step 3 — power is the chance the observed mean clears $4.11 *when the truth is $5*. Under $H_1$, $\bar{x}$ is centered at $5$ with standard error $2.5$, so the threshold $4.11$ sits $\dfrac{4.11 - 5}{2.5} = \dfrac{-0.89}{2.5} \approx -0.36$ standard errors *below* the true mean. The probability of landing above a point $0.36$ standard errors below center is $\Phi(0.36) \approx 0.64$. So **power $\approx 64\%$** — you have only a ~64% chance of detecting this real edge with 400 days.

Step 4 — how much data for 80% power? To reach power $0.80$ you need the true mean to sit about $0.84$ standard errors above the threshold (since $\Phi(0.84) \approx 0.80$), and the threshold itself sits $1.645$ SE above zero, so you need the effect to be $1.645 + 0.84 = 2.485$ standard errors: $\dfrac{5}{\sigma/\sqrt{n}} \geq 2.485$, giving $\sqrt{n} \geq 2.485 \times \dfrac{50}{5} = 24.85$, so $n \geq 618$ days. The intuition: *a small edge against big noise needs a lot of data — roughly two and a half years of daily data here just to have an 80% shot at seeing a real \$5/day edge. Anyone claiming significance on a few months of such a strategy is mostly seeing noise.*

## Multiple testing: why running 100 backtests guarantees fake winners

Here is the section that, more than any other, mints the difference between a junior who passed a stats class and someone a desk actually wants. Everything above assumed you run *one* test. On a real desk — and in any quant research process — you run hundreds. And the moment you run many tests, the logic of the single test quietly betrays you.

Recall what $\alpha = 0.05$ promises: a 5% chance of a false positive *on a single test*. Now run $m$ independent tests, all on strategies that genuinely have no edge. The chance that a *given* test produces no false positive is $0.95$. The chance that *all $m$* produce no false positive is $0.95^m$. So the chance of **at least one false positive** — at least one fake "winner" — is

$$P(\text{at least one false positive}) = 1 - 0.95^m.$$

![Run enough tests and a false winner is nearly guaranteed: the chance of at least one false positive grows toward one](/imgs/blogs/hypothesis-testing-pvalues-quant-interviews-9.png)

The figure plots this curve and the numbers are brutal. At $m = 1$ test, the chance of a false winner is the promised 5%. At $m = 14$ it crosses 50% — a coin flip. At $m = 20$ it is **64%**. At $m = 100$ it is **99.4%** — a near-certainty. In other words, *if you try 100 strategies that are all secretly worthless, you are virtually guaranteed to find several that pass a 5% significance test.* They are not edges. They are the statistical equivalent of buying 100 lottery tickets and being amazed that one won.

This is the **multiple comparisons problem**, and it is the single biggest reason most "discovered" trading strategies are fake. The quantity $1 - 0.95^m$ is the **family-wise error rate (FWER)** — the probability of at least one false positive across the whole family of tests. Left uncorrected, it explodes. The fix is to make the per-test bar stricter so the *family-wide* false-positive rate stays controlled.

#### Worked example: how many fake winners from 20 worthless strategies?

You and your team backtest $m = 20$ candidate strategies, each tested at $\alpha = 0.05$. Suppose, unknown to you, *all 20 are worthless* (no real edge). **How many do you expect to pass, and what's the chance at least one does?**

Expected number of false positives: each worthless strategy passes with probability $0.05$, so across 20 the expected count is $20 \times 0.05 = 1.0$. You should *expect one fake winner* even though nothing works. The chance of at least one is $1 - 0.95^{20} = 1 - 0.358 = 0.642$, about **64%**. The intuition that lands the point: *finding a strategy that passes at 5% after testing 20 is not evidence of skill — it is the single most likely outcome of testing 20 nothing-strategies. The "winner" you'd proudly present is, on the base rates, probably the expected false positive.* The right reaction is not excitement; it is to correct for how many tests you ran.

## Corrections: Bonferroni and the false discovery rate

There are two famous ways to fight multiple testing, and naming both — plus knowing when to use each — is exactly what a multiple-testing interview question is fishing for.

![Two ways to tame multiple testing: Bonferroni divides alpha by the number of tests while Benjamini-Hochberg ranks p-values against a sliding line](/imgs/blogs/hypothesis-testing-pvalues-quant-interviews-10.png)

### Bonferroni: control the chance of any false positive

The **Bonferroni correction** is brutally simple: if you run $m$ tests and want the family-wise error rate to stay at $\alpha$, test each one at the stricter level $\alpha / m$. For $m = 20$ tests and $\alpha = 0.05$, every individual test must clear $0.05 / 20 = 0.0025$ — a 40× stricter bar. The guarantee is airtight: the probability of *any* false positive across all 20 tests stays at or below 5%. (The logic is the union bound — the chance of any of several rare events is at most the sum of their individual chances.)

The cost is power. By demanding $p < 0.0025$ everywhere, Bonferroni throws out most genuinely real-but-modest edges along with the fakes. It is the right tool when *any* false positive is expensive — when you would rather miss real edges than ever deploy a fake one. It is too conservative when you can tolerate a few false leads in exchange for catching more real ones.

### Benjamini–Hochberg: control the false discovery rate

The **false discovery rate (FDR)** reframes the goal. Instead of "never have a single false positive" (Bonferroni's family-wise control), FDR controls the *expected fraction of your discoveries that are false*. If you make 10 discoveries and accept an FDR of 10%, you are tolerating about 1 false one among them — which is a far more natural budget for a research process where you will validate the survivors anyway.

The **Benjamini–Hochberg (BH) procedure** implements it elegantly. Sort your $m$ p-values from smallest to largest: $p_{(1)} \leq p_{(2)} \leq \dots \leq p_{(m)}$. Find the largest rank $k$ such that $p_{(k)} \leq \dfrac{k}{m}\,\alpha$. Reject all hypotheses with rank $1$ through $k$. That sliding threshold — strict for the very smallest p-value, more lenient as you go up the ranks — is the whole trick.

![The Benjamini-Hochberg line rises with rank while Bonferroni uses one flat strict cutoff, so BH keeps more real discoveries](/imgs/blogs/hypothesis-testing-pvalues-quant-interviews-11.png)

The figure contrasts the two. The dashed flat line is Bonferroni: every p-value, regardless of rank, faces the same harsh $0.0025$ cutoff. The rising diagonal is the BH line, $\dfrac{k}{m}\alpha$. The green squares are p-values that fall *below* the BH line — your discoveries. BH keeps strictly more of them than Bonferroni would, because its bar relaxes for the higher ranks while still controlling the false fraction. **Bonferroni asks "is this single result extreme enough to be sure it's not noise?"; BH asks "across all my results, is the slate of discoveries mostly real?"** The second question is usually the one a quant research pipeline should be asking.

#### Worked example: correcting 20 strategies with Bonferroni vs FDR

You test $m = 20$ strategies and the four smallest p-values come back as $p_{(1)} = 0.0008$, $p_{(2)} = 0.0021$, $p_{(3)} = 0.0039$, $p_{(4)} = 0.0074$ (the other 16 are all above $0.01$). Use $\alpha = 0.05$. **Which strategies survive Bonferroni, and which survive Benjamini–Hochberg?**

*Bonferroni* bar: $0.05 / 20 = 0.0025$. Compare each: $p_{(1)} = 0.0008 < 0.0025$ — survives. $p_{(2)} = 0.0021 < 0.0025$ — survives. $p_{(3)} = 0.0039 > 0.0025$ — fails. $p_{(4)}$ fails. So Bonferroni keeps **2** strategies.

*Benjamini–Hochberg*: compare $p_{(k)}$ to $\dfrac{k}{20}(0.05) = k \times 0.0025$.
- $k = 1$: $0.0008 \leq 1 \times 0.0025 = 0.0025$? Yes.
- $k = 2$: $0.0021 \leq 2 \times 0.0025 = 0.0050$? Yes.
- $k = 3$: $0.0039 \leq 3 \times 0.0025 = 0.0075$? Yes.
- $k = 4$: $0.0074 \leq 4 \times 0.0025 = 0.0100$? Yes.
- $k = 5$: the fifth p-value is above $0.01$, and $5 \times 0.0025 = 0.0125$ — suppose $p_{(5)} = 0.020 > 0.0125$, fails.

The largest passing rank is $k = 4$, so BH rejects ranks 1 through 4 — it keeps **4** strategies. The intuition: *same data, but Bonferroni's flat strict line salvages only the two most extreme winners while BH's sliding line recovers two additional real edges — at the cost of a controlled, small expected fraction of false ones. Which you prefer depends on whether a missed edge or a false lead costs you more.*

## In the interview room: five fully-solved problems

The concepts above become muscle memory only when you can solve cold problems out loud. Here are five in the exact register interviewers use — terse setups, a demand to reason, and a trap or two baked in. Solve each yourself first, then check the walkthrough.

#### Worked example: test a coin for fairness from a sample

*"I flip a coin 100 times and get 60 heads. Is the coin fair? Walk me through it."*

This is a one-sample proportion test in disguise. $H_0: p = 0.5$ (fair), $H_1: p \neq 0.5$ (two-sided — we did not pre-suspect a direction). Under the null, the number of heads has mean $np = 50$ and standard deviation $\sqrt{np(1-p)} = \sqrt{100 \times 0.5 \times 0.5} = \sqrt{25} = 5$. The observed 60 heads is $\dfrac{60 - 50}{5} = 2.0$ standard deviations above the null mean. For a two-sided 5% test the critical value is $1.96$; since $2.0 > 1.96$, you **reject** — there is (just barely) significant evidence the coin is unfair, with a two-sided p-value of about $0.046$.

The interviewer is watching for three things. First, that you used a *two-sided* test (you had no prior reason to expect more heads specifically). Second, that you can compute the standard deviation of a binomial in your head: $\sqrt{npq}$. Third — the sharp follow-up — *"how confident are you?"* The honest answer: barely. $p \approx 0.046$ squeaks under $0.05$; with 60 heads out of 100 the evidence is weak, and a single test this close to the boundary should not make you sure. If they push to "what if it were 60 heads out of 1000 flips?", the standard deviation becomes $\sqrt{1000 \times 0.25} \approx 15.8$, the count $600$ is $\dfrac{600 - 500}{15.8} \approx 6.3$ standard deviations out — overwhelming evidence. *Same proportion, vastly more data, vastly more significance: magnitude and significance are different, and data quantity drives the latter.*

#### Worked example: the strategy with a great Sharpe over six months

*"A trader shows you a strategy with an annualized Sharpe ratio of 2.5 over six months of daily data. Are you impressed?"*

The right move is to convert "impressive Sharpe" into "how many standard errors from zero," which is a hypothesis test. The **Sharpe ratio** is the mean return divided by its standard deviation (here annualized). The t-statistic for "is the mean return greater than zero" relates to the Sharpe by a clean approximation: $t \approx \text{Sharpe} \times \sqrt{T}$, where $T$ is the number of years of data (because the Sharpe is a per-year signal-to-noise and $\sqrt{T}$ accumulates the evidence). Six months is $T = 0.5$ years, so $t \approx 2.5 \times \sqrt{0.5} \approx 2.5 \times 0.707 \approx 1.77$.

A t-statistic of $1.77$ is *not* significant at the two-sided 5% level (which needs $1.96$); it barely clears a one-sided test. So despite the flashy 2.5 Sharpe, six months simply is not enough data to be confident the edge is real — the standard error on a six-month Sharpe is roughly $\sqrt{1/T} = \sqrt{2} \approx 1.4$, meaning the true Sharpe could plausibly be anywhere from near zero to about 5. The interviewer wants you to say: *"A high Sharpe over a short window is mostly an uncertainty statement, not an edge statement. I'd want at least two to three years before trusting a Sharpe of this size — and I'd ask how many strategy variants were tried to get here."* That last sentence is the multiple-testing reflex, and it is gold.

#### Worked example: p-hacking caught in the act

*"A researcher tested 40 signals, found 3 with $p < 0.05$, and reports those 3 as discoveries. What's wrong, and how would you fix it?"*

The problem is uncorrected multiple testing. With 40 worthless signals, the expected number of false positives at $\alpha = 0.05$ is $40 \times 0.05 = 2$, and the chance of at least one is $1 - 0.95^{40} \approx 0.87$. So finding 3 "significant" signals out of 40 is *almost exactly what pure noise produces* — 2 expected by chance, and 3 is within a hair of that. The researcher has discovered essentially nothing; they have rediscovered the multiple comparisons problem.

The fix, stated crisply: *"Correct for the 40 tests. Bonferroni would require $p < 0.05/40 = 0.00125$ — far stricter, and probably none of the 3 survive. If I want more power I'd use Benjamini–Hochberg to control the false discovery rate. And ideally I'd validate any survivor on fresh out-of-sample data the researcher never touched, because correction reduces but does not eliminate the risk of an overfit fluke."* If the interviewer asks how the researcher *should* have reported, the answer is: report all 40 p-values and the correction, not a cherry-picked 3. Reporting only the winners is the essence of **p-hacking**.

#### Worked example: failing to reject is not proof

*"I tested whether my strategy's mean return differs from zero, got $p = 0.40$, and concluded the strategy has no edge. Correct?"*

No — this is the "absence of evidence" trap. A large p-value ($0.40$) means the data is *consistent* with the null, not that the null is *true*. You failed to find evidence of an edge, which is very different from finding evidence of no edge. The most common reason for a big p-value on a real strategy is simply too little data: a small-but-real edge buried in noise produces exactly this result. The correct conclusion: *"I cannot reject 'no edge' at this sample size — but I also can't conclude there's no edge. The data just doesn't have the power to decide."*

The way to actually argue "no edge" is to build a confidence interval and show it is *tight around zero*: if the 95% interval for the mean is $(-0.5, +0.5)$ basis points, you can credibly say any edge is too small to matter. But if the interval is $(-30, +35)$ basis points, you have learned nothing — a large edge is perfectly compatible with your data. *The width of the interval, not the p-value, tells you whether "no meaningful edge" is a defensible claim.* Interviewers love this because it tests whether you understand that a hypothesis test is asymmetric: it can reject the null, but it can never prove it.

#### Worked example: choosing one-sided versus two-sided, and the cost of switching after

*"You ran a two-sided test and got $p = 0.06$ — not significant. Your boss says 'just use a one-sided test, then it's $0.03$ and we're good.' What do you say?"*

Mechanically the boss is right that a one-sided test has half the p-value (when the effect is in the predicted direction): $0.06 / 2 = 0.03$. But switching the test *after seeing the data, specifically to cross the threshold,* is statistical malpractice — it inflates your true type I error. The choice of one-sided versus two-sided must be made *before* looking at the data and must be justified by the science: a one-sided test is legitimate only when an effect in the opposite direction would be meaningless or impossible to act on. "We only care about profit, never loss" can justify one-sided up front; "we need it to be significant" never can.

The deeper point to voice: *"Changing the test to get the answer you want is a form of p-hacking. If we'd genuinely pre-committed to one-sided because we only care about positive edge, fine — but deciding that now, because $0.06$ disappointed us, means our real false-positive rate is higher than the 5% we're claiming. I'd rather report $p = 0.06$ honestly and either collect more data or treat it as a weak candidate."* This answer signals integrity under pressure, which is exactly what a desk wants in someone who will control real risk.

## Common misconceptions

These are the beliefs that feel right, get repeated constantly, and are wrong. Each one is a place an interviewer can catch you.

**"$p < 0.05$ means the result is true (or 95% likely true)."** No. A p-value below $0.05$ means the data would be surprising if the null were true, so you reject the null — but the *probability* the result is real depends on how plausible an edge was to begin with and how many things you tested. After 100 tests, a $p = 0.04$ winner is more likely a false positive than a real edge. Significance is a statement about the data under the null, never directly about the truth of the alternative.

**"The p-value is the probability the null hypothesis is true."** This is the flipped conditional and the most damaging error in the field. The p-value is $P(\text{data} \mid H_0)$; the thing people want is $P(H_0 \mid \text{data})$. Getting from one to the other requires Bayes' theorem and a prior — you cannot read it off the p-value. If you remember one sentence from this entire post, make it this one.

**"Failing to reject the null proves the null is true."** No — absence of evidence is not evidence of absence. A non-significant result is consistent with "no edge" *and* with "a real edge I lack the data to detect." To argue for the null you need a tight confidence interval around zero, demonstrating the effect, if any, is negligibly small. A wide interval means you simply do not know.

**"A statistically significant edge is a big, tradeable edge."** Significance and magnitude are independent. With enough data, a $0.1$ basis-point edge is wildly significant and completely useless after costs. With little data, a huge edge can be non-significant. Always ask for the effect size *and* its confidence interval, not just the p-value.

**"More data always means a better test."** More data raises power, which is good — but it also makes *trivially small* deviations significant. With ten million trades, essentially any non-zero edge becomes "significant," including ones smaller than your transaction costs. Past a point, the right question shifts from "is it significant?" to "is it big enough to matter?" — an effect-size question, not a p-value question.

**"If I correct for multiple testing, my surviving strategy is definitely real."** Correction controls the *rate* of false positives; it does not certify any individual survivor. A Bonferroni- or BH-surviving strategy is a stronger candidate, not a proven edge. The gold standard remains out-of-sample validation: test the survivor on data it has never seen, ideally a different time period or market.

## How it shows up on a real trading desk

Statistics in a textbook is tidy. On a desk it is a daily fight against your own optimism, and the multiple-testing trap is the enemy that never sleeps. Here is how these ideas actually bite.

**"Is this Sharpe real?" — the deflated Sharpe ratio.** A researcher walks in with a backtest showing a Sharpe of $2.0$ over two years. The desk-hardened response is not "great," it is "how many strategies did you try to get this one?" Because the more variations you test — different lookback windows, thresholds, universes — the higher the *best* Sharpe will look *purely by chance*, even if none has real edge. The **deflated Sharpe ratio**, developed by Bailey and López de Prado, formalizes this: it discounts a reported Sharpe for the number of trials, the length of the backtest, and the non-normality of returns, producing an honest significance.

![A raw backtest Sharpe inflates with the number of trials, so deflation adjusts it down to an honest significance before deployment](/imgs/blogs/hypothesis-testing-pvalues-quant-interviews-12.png)

The figure traces the logic. A raw Sharpe of $2.0$, found after $N = 50$ variants on only two years of data, deflates dramatically — to something like $0.6$ once you account for the fact that the *maximum of 50 random Sharpes* is naturally large. The honest call is "likely overfit, do not deploy." The lesson the desk drills into you: *the reported Sharpe is the maximum of everything you tried, and the maximum of many noisy numbers is biased upward. Deflate before you believe.*

**Data snooping and the backtest overfitting epidemic.** López de Prado's research argues that the majority of published quantitative-finance "discoveries" are false, precisely because of uncorrected multiple testing — researchers run thousands of backtests and report the winners. A famous illustration: with enough trials you can find a strategy that perfectly "predicts" the S&P 500 using, say, the historical butter production in Bangladesh — a real spurious correlation that has circulated for years. The mechanism is exactly $1 - 0.95^m$ marching to certainty. The defense is discipline: pre-register your hypotheses, count your trials honestly, correct for them, and hold out clean data you never optimize against.

**The 2010s factor-zoo reckoning.** Academic finance spent decades publishing hundreds of "factors" claimed to predict stock returns — value, momentum, and then a sprawling zoo of more exotic ones. In an influential 2016 paper, Harvey, Liu, and Zhu argued that because of multiple testing, the significance bar for a *new* factor should be a t-statistic around $3.0$, not the traditional $2.0$ — and that under a stricter bar, a large fraction of published factors fail to replicate. This is the multiple-testing correction applied to an entire research literature, and it reshaped how serious quants treat any factor claim: *a t-stat of 2 used to mean "publishable"; after correcting for how many factors the profession tested, it now means "probably noise."*

**Sizing under uncertainty, not just yes/no.** On a live desk you rarely just reject or fail to reject — you size. A strategy whose 95% confidence interval for its edge is $(0.1, 3.0)$ basis points per trade gets a small allocation that grows as more data tightens the interval. The confidence interval, not the p-value, drives the position. This is the practical face of the test/interval duality: the interval tells you both *whether* there's an edge and *how sure* you are, and capital flows in proportion to certainty.

**A/B testing the execution algorithm.** The same machinery runs far from "alpha." When a desk tweaks an execution algorithm to (hopefully) reduce slippage, it runs a paired test: route some orders the old way, some the new way, on the same instruments and times, and test whether the per-order cost difference has mean zero. Paired design cancels the market-wide noise; the t-test on the differences gives the verdict. And because the team tries many tweaks over a quarter, they correct for multiple testing before declaring any one tweak a real improvement — the FDR is the natural budget when dozens of small experiments run in parallel.

## When this matters to you and where to go next

If you are interviewing, internalize the loop in the first figure and be able to walk it out loud on any setup an interviewer throws at you: name the null, build the statistic as signal-over-noise, define the p-value with the conditional in the right direction, and — the move that separates you — *ask how many things were tested before this result was found.* That single instinct, the multiple-testing reflex, is what desks are really probing, because it is the difference between a researcher who finds real edges and one who manufactures expensive illusions.

If you are building real strategies, the practical discipline is short: pre-commit to your hypothesis and your test before you look at the data; count every variant you try and correct for them with Bonferroni or Benjamini–Hochberg; deflate every Sharpe for the number of trials; report confidence intervals, not just p-values, so you size by certainty; and validate every survivor on data it has never seen. None of this is exotic — it is the grammar of the test from the very first figure, applied honestly, many times, by someone who has internalized that *the maximum of many noisy numbers is biased upward and your own optimism is the adversary.*

To go deeper, the natural next steps connect directly to the puzzles desks ask. Hypothesis testing is built on probability, so it pairs naturally with [conditional probability and Bayes for quant interviews](/blog/trading/quantitative-finance/conditional-probability-bayes-quant-interviews) — the p-value-versus-$P(H_0)$ confusion is literally a Bayes problem. The Sharpe-significance and sizing material connects to [the Kelly criterion and sequential betting](/blog/trading/quantitative-finance/kelly-criterion-sequential-betting-quant-interviews), which turns an estimated edge into a position size, and to [classic quant probability problems](/blog/trading/quantitative-finance/classic-quant-probability-problems) and [expected value techniques](/blog/trading/quantitative-finance/expected-value-techniques-quant-interviews) for the broader interview toolkit. Master the loop, respect the multiple-testing trap, and you will reason about evidence the way a desk needs you to — calmly, quantitatively, and without fooling yourself.
