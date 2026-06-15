---
title: "Hypothesis testing and p-values, honestly: telling a real trading edge from luck"
date: "2026-06-15"
description: "A build-from-zero tour of hypothesis testing, the p-value, Type I and II errors, the t-stat of a Sharpe ratio, the multiple-testing trap, Bonferroni, FDR, and the deflated Sharpe ratio for quant trading."
tags: ["hypothesis-testing", "p-value", "statistical-significance", "type-i-error", "type-ii-error", "t-test", "sharpe-ratio", "multiple-testing", "deflated-sharpe", "p-hacking", "math-for-quants", "quant-finance"]
category: "trading"
subcategory: "Quantitative Finance"
author: "Hiep Tran"
featured: true
readTime: 46
---

> [!important]
> **TL;DR** — A hypothesis test is the machinery that answers the only question that matters about a backtest: *is this edge real, or is it luck?* It does that by asking how surprising your result would be in a world where the strategy is worthless.
>
> - **The p-value is the probability of seeing data at least this extreme if the strategy has no edge** — it is *not* the probability the strategy is worthless. Confusing those two is the single most expensive mistake in quant research.
> - The t-statistic of a strategy's mean return is, conveniently, almost exactly its **Sharpe ratio times the square root of the number of years**: $t \approx \mathrm{SR}\cdot\sqrt{n}$. A Sharpe of 1.0 over 3 years gives $t\approx 1.7$ — *not* significant at 5%.
> - **Type I error** (trading a dead strategy) costs you money directly; **Type II error** (discarding a real edge) costs you opportunity. The whole game is buying down Type I without drowning in Type II.
> - Test 100 random strategies at the 5% level and **about 5 will look significant by pure luck**. Mining strategies inflates false positives, and corrections — Bonferroni, FDR, the **deflated Sharpe ratio** — exist to claw that inflation back.
> - The one number to remember: to clear a t-stat of 2 (the rough bar for "real"), a strategy with Sharpe 1.0 needs about **4 years** of daily data. Most "amazing" backtests have nowhere near that.

There is a number that gets whispered in every quant interview, printed on every research memo, and used to wave billions of dollars into and out of trading strategies: **the p-value**. It is the gatekeeper. A strategy with `p < 0.05` gets funded; one with `p = 0.07` gets shelved. And almost everyone who uses it — including people with PhDs — misunderstands what it actually means.

Here is the uncomfortable truth that this post is built around. A backtest that shows a beautiful, smooth, upward-sloping equity curve tells you *nothing* by itself about whether the strategy works. Markets are noisy. If you flip a coin a thousand times, some runs will look like a hot streak. If you test enough strategies on enough historical data, some of them will look like genius — not because they are, but because you looked at so many. Hypothesis testing is the discipline that separates the genuine edges from the lucky streaks, and the p-value is its central, slippery, endlessly-abused tool. The diagram below is the mental model we will tour for the rest of the article: raw data goes in one end, and a keep-or-discard decision comes out the other, with a test statistic and a p-value as the two gears in the middle.

![Pipeline from backtest returns to test statistic to p-value to a trade or discard decision](/imgs/blogs/hypothesis-testing-pvalues-math-for-quants-1.png)

That pipeline is the whole machine. You start with returns from a backtest. You squeeze them into a single number — a *test statistic* — that measures how far your result sits from "no edge." You convert that into a *p-value*, which scores how surprising the result would be if the strategy were truly worthless. And then you make a decision: trade it, or throw it away. Every section below cracks open one of those gears. By the end you will be able to look at a Sharpe ratio and a backtest length and say, on the spot, whether the number deserves your respect or your suspicion.

## Foundations: the courtroom of statistics

Before any math, an analogy that will carry the entire post: **a hypothesis test is a criminal trial.**

In a courtroom, the default assumption is *innocent until proven guilty*. The prosecution does not have to prove innocence is impossible; they have to prove guilt *beyond a reasonable doubt*. If the evidence is weak, the verdict is "not guilty" — which is **not** the same as "proven innocent." It just means the evidence wasn't strong enough to overturn the default.

Statistics works exactly the same way. The default assumption — the thing we hold true until evidence forces us to abandon it — is that **your strategy has no edge**. That default is called the *null hypothesis*. The "edge is real" claim is the *alternative hypothesis*, the thing you, the prosecutor, are trying to establish. You never prove the alternative directly. You gather evidence (returns), measure how unlikely that evidence would be under the assumption of no edge, and if it's unlikely enough — beyond a reasonable doubt — you *reject* the null. If it isn't, you "fail to reject," which, like "not guilty," is not a declaration of innocence. It just means the data didn't clear the bar.

Hold that courtroom in your head. It explains nearly every confusing thing about p-values.

### The null and alternative hypotheses

Let's name the players precisely, because the whole edifice rests on them.

- The **null hypothesis**, written $H_0$, is the boring, skeptical default. For a trading strategy it is usually: *the true average return of this strategy is zero* (after costs). Written as math, $H_0: \mu = 0$, where $\mu$ (the Greek letter "mu") is the **true, unknown** average return per period — the number you would converge to if you could trade the strategy forever.
- The **alternative hypothesis**, written $H_1$ or $H_a$, is what you suspect is true and want to demonstrate: *the strategy makes money*, i.e. $H_1: \mu > 0$. (Sometimes you test $\mu \neq 0$, allowing it to lose money too — that's a *two-sided* test, more on that later.)

A *basis point* — a term you'll meet constantly — is one hundredth of a percent, 0.01%. We'll use it for small returns and fees.

The crucial asymmetry: $H_0$ is specific (it pins $\mu$ to exactly 0), while $H_1$ is vague (it just says $\mu$ is positive). That asymmetry is deliberate. We can compute exactly how the world behaves *if* $\mu = 0$ — we know the math of pure noise. We cannot compute how it behaves if "the strategy works," because that covers infinitely many possible edges. So we always do our arithmetic in the null world, the world of no edge, and ask: how weird is my data *here*?

### The test statistic: collapsing a backtest into one number

A backtest produces hundreds or thousands of daily returns. We cannot reason about all of them at once. So we **collapse them into a single number** that captures "how far from no-edge are we, measured in units of noise." That number is the *test statistic*.

The most important test statistic in all of trading is the **t-statistic** for a mean. In plain English, it answers: *how many standard errors above zero is my average return?* If your average daily return is a big positive number and the day-to-day variation is small, the t-stat is large and you have a loud signal. If your average return is barely positive and the daily swings are enormous, the t-stat is near zero and you have noise. We'll build the exact formula in the t-test section. For now, the intuition is everything: **the test statistic is a signal-to-noise ratio.**

### Significance level $\alpha$: how often you're willing to be fooled

Back to the courtroom. Some innocent people get convicted — the system makes mistakes. A society decides, in advance, how often it's willing to tolerate convicting an innocent person, and sets the "beyond a reasonable doubt" bar accordingly.

In statistics that tolerance is the **significance level**, written $\alpha$ (alpha). It is the probability you're willing to accept of *rejecting the null when the null is actually true* — of declaring an edge real when it's pure luck. The near-universal convention is $\alpha = 0.05$, meaning you accept a 5% chance of being fooled by noise on any single test. Particle physicists, who hate being wrong, use a far stricter $\alpha$ around 0.0000003 (the famous "5 sigma"). Quants, depending on how much pain a false discovery causes, often demand a t-stat of 3 rather than 2.

The decisive point, which we will hammer repeatedly: **you must choose $\alpha$ before you look at the result.** Choosing it afterward — "well, 0.06 is basically 0.05" — is moving the goalposts after the ball is kicked. We'll see the damage that does.

With those four pieces — null, alternative, test statistic, and $\alpha$ — you already have the skeleton of every hypothesis test ever run. Now we fill in the muscles.

#### Worked example: a coin that might be rigged

You suspect a coin is biased toward heads. Your null hypothesis is "fair coin," $H_0: p = 0.5$, where $p$ is the true probability of heads. You flip it 100 times and get 58 heads.

Is 58 surprising for a fair coin? A fair coin's head-count over 100 flips has an average of 50 and a standard deviation of $\sqrt{100 \times 0.5 \times 0.5} = \sqrt{25} = 5$. Your result of 58 is $(58 - 50)/5 = 1.6$ standard deviations above the expected 50. That's your test statistic: $z = 1.6$.

Now the question that defines the p-value: in the fair-coin world, how often do you get 58 *or more* heads? That tail probability, for $z = 1.6$ on one side, is about **5.5%**, so the one-sided p-value is roughly 0.055. At $\alpha = 0.05$ you would *fail to reject* — 0.055 is just over the line. The evidence is suggestive but not, by the rule you set in advance, conclusive. If you had pre-committed to $\alpha = 0.10$, you'd reject. Same data, different verdict, because the bar was set differently — and you must set it *before*, not after.

The intuition this teaches: even a result that *feels* like a hot streak (58 vs 50) can be perfectly ordinary noise, and the p-value is just the precise measure of "how ordinary."

## 1. Type I and Type II errors: the two ways a trader gets burned

Every decision under uncertainty can go wrong in two distinct ways, and confusing them is how careers end. Picture the world's true state on one axis (the strategy either has an edge or it doesn't) and your decision on the other (you either trade it or you don't). That gives a 2×2 grid, and two of the four boxes are mistakes.

![Two by two matrix of truth versus decision showing Type I and Type II errors](/imgs/blogs/hypothesis-testing-pvalues-math-for-quants-2.png)

The two-by-two grid above is the single most important figure in the entire post, so let's read every box.

- **Top-left (the strategy is real, you trade it):** correct. You caught a real edge and you're making money. Champagne.
- **Bottom-right (the strategy is noise, you discard it):** also correct. You correctly rejected a worthless idea. No harm done.
- **Bottom-left (the strategy is noise, but you trade it anyway):** this is a **Type I error**, a *false positive*, a *false discovery*. You believed a lucky backtest. This is the box that *costs you cash*: you deploy capital into a strategy with no edge, you pay the trading costs, and you bleed.
- **Top-right (the strategy is real, but you discard it):** this is a **Type II error**, a *false negative*, a *missed discovery*. The edge was there and you walked past it. This box costs you *opportunity* — the profit you never made.

The significance level $\alpha$ you met earlier is precisely the long-run rate of Type I errors *when the null is true*. Set $\alpha = 0.05$ and, across many tests of genuinely worthless strategies, you'll wrongly "discover" 5% of them. The Type II error rate gets its own Greek letter, $\beta$ (beta), and its complement, $1 - \beta$, is called the **power** of the test — the probability you *correctly* detect a real edge when there is one.

### The fundamental tension: you can't shrink both errors at once

Here is the seesaw at the heart of testing. If you make your bar for "real" stricter — demand a t-stat of 3 instead of 2 — you reject fewer worthless strategies by accident, so Type I errors fall. But you also reject more *genuine* strategies whose edge wasn't loud enough to clear the higher bar, so Type II errors rise and power falls. Tighten one and the other loosens. The only way to push *both* down at once is to gather more data — a bigger sample sharpens the test in both directions simultaneously.

A trading desk has to price the two errors against each other in dollars. A false positive (Type I) is a strategy that loses money in production. A false negative (Type II) is a foregone profit. Which hurts more depends on the shop. A market-maker running thousands of tiny strategies can tolerate some false positives because each costs little and diversification dampens them. A long-horizon fund deploying a single large allocation into one strategy lives in terror of Type I — a false discovery there is a catastrophe.

#### Worked example: the dollar cost of a false discovery

You run a single strategy through a backtest, it looks significant, you allocate **\$10 million** to it, and it is in fact a Type I error — a dead strategy that looked alive. What does that mistake cost?

Suppose the strategy, having no real edge, drifts to zero gross but you pay round-trip trading costs of 10 basis points per turn and you turn the book over twice a week, so about 100 turns a year. That's $100 \times 10\text{ bps} = 1{,}000$ bps $= 10\%$ of capital a year burned on costs alone. On \$10 million that is **\$1 million a year** vaporized, plus the opportunity cost of the \$10 million doing nothing useful. Run that strategy for two years before you admit it's dead and you've lit **\$2 million** on fire.

Now the Type II side. Suppose you *discarded* a genuine strategy with a true Sharpe of 1.0 that, on \$10 million at 8% volatility, would have made about $1.0 \times 0.08 \times \$10\text{M} = \$800{,}000$ a year. Walk past it for two years and the missed profit is **\$1.6 million**. The intuition: both errors are measured in millions, and a serious shop must put a literal price tag on each before it sets its significance bar.

## 2. The t-test for a mean return, from scratch

We've been waving at "the test statistic." Now we build the one quants use every single day: the **t-statistic for a mean**.

The plain-English idea first. You have a pile of daily returns. You compute their average — that's your estimate of the edge. But the average of a noisy sample is itself uncertain; if you'd run the strategy on a different stretch of history you'd get a slightly different average. The question is whether your average is *far enough* from zero that noise alone can't plausibly explain it. "Far enough" has to be measured in the natural unit of noise, which is the **standard error** of the average.

The standard error of a sample mean is the standard deviation of individual returns divided by the square root of how many you have:

$$\text{SE}(\bar r) = \frac{s}{\sqrt{n}}$$

Here $\bar r$ (r-bar) is the *sample average return*, $s$ is the *sample standard deviation* of the per-period returns (a measure of how much they bounce around), and $n$ is the *number of periods*. The $\sqrt{n}$ in the denominator is the engine of everything: it says that as you gather more data, the uncertainty in your average shrinks — but only as fast as the square root, so to halve your uncertainty you need *four times* the data. (This is the standard-error result from the [law of large numbers and the central limit theorem](/blog/trading/quantitative-finance/hypothesis-testing-pvalues-quant-interviews) family of ideas; we lean on it constantly.)

The t-statistic is then simply the average measured in units of its own standard error:

$$t = \frac{\bar r - 0}{\text{SE}(\bar r)} = \frac{\bar r}{s/\sqrt{n}} = \frac{\bar r}{s}\sqrt{n}$$

The "$-\,0$" is the null value $\mu = 0$ — we're measuring distance from "no edge." A t-stat of 2 means your average return sits two standard errors above zero; a t-stat of 0.5 means it's swimming in noise.

### Why a t-distribution and not a normal one

A small wrinkle that matters for honesty. If we knew the *true* standard deviation, this statistic would follow a normal (bell-curve) distribution, and a value beyond ±1.96 would be the 5% cutoff. But we don't know the true standard deviation — we *estimate* it from the same sample, with $s$. That extra layer of uncertainty fattens the tails slightly, and the right reference curve is the **Student's t-distribution** with $n - 1$ degrees of freedom. For large $n$ (more than ~100 observations, which any real backtest has) the t-distribution is indistinguishable from the normal, and the magic cutoffs are the familiar ones: **a t-stat of about 1.96 for two-sided 5%, and a t-stat of 2 as the everyday rule of thumb for "significant."** For the tiny samples in some interview puzzles the difference matters; for a multi-year daily backtest it does not.

### The headline result: the t-stat of a Sharpe ratio

Now the result that makes this whole post tractable for traders. The **Sharpe ratio** is the average return divided by the standard deviation of returns, annualized — it's the industry's signal-to-noise measure. Define the *per-period* Sharpe as $\mathrm{SR}_{\text{period}} = \bar r / s$. Look back at the t-stat formula:

$$t = \frac{\bar r}{s}\sqrt{n} = \mathrm{SR}_{\text{period}} \cdot \sqrt{n}$$

The t-statistic of a strategy's mean return is *literally its per-period Sharpe times the square root of the number of periods*. And when we measure the Sharpe in **annual** terms (the convention) and $n$ in **years**, this cleans up into the single most useful approximation in backtest evaluation:

$$\boxed{\,t \approx \mathrm{SR}_{\text{annual}} \cdot \sqrt{n_{\text{years}}}\,}$$

(The annualization of Sharpe and the annualization of $\sqrt n$ cancel exactly when returns are independent, which is why this works regardless of whether you measure in days, weeks, or years.) This one line lets you convert any reported Sharpe and backtest length into a significance verdict *in your head*. Memorize it.

![Pipeline showing data summarized to a test statistic then to a p-value then a decision](/imgs/blogs/hypothesis-testing-pvalues-math-for-quants-3.png)

The figure above lays out where the t-stat lands relative to the null distribution: under "no edge," the t-stat hovers near zero and almost never strays far; your observed t-stat sits somewhere on that curve, and the p-value is the area in the tail beyond it. The rejection region — the extreme tail past your $\alpha$ cutoff — is the only place a t-stat counts as significant.

#### Worked example: is a Sharpe 1.0 strategy over 3 years significant?

You're handed a strategy with a backtested **annual Sharpe ratio of 1.0** over **3 years** of daily data. The pitch deck calls it "consistently profitable." Should you believe it at the 5% level?

Apply the formula directly:

$$t \approx \mathrm{SR} \cdot \sqrt{n} = 1.0 \times \sqrt{3} \approx 1.0 \times 1.73 = 1.73$$

The bar for two-sided 5% significance is a t-stat of about **1.96**, and the everyday rule of thumb is **2.0**. A t-stat of **1.73 falls short of both.** The two-sided p-value for $t = 1.73$ is roughly **0.085** — about 8.5%. In courtroom terms: suggestive, but not beyond a reasonable doubt. You fail to reject the null. This "consistently profitable" strategy is statistically indistinguishable from luck.

How long *would* it need to run? To hit $t = 2$ at Sharpe 1.0, you need $\sqrt{n} = 2$, so $n = 4$ years. To hit the stricter $t = 3$ that many serious desks demand, $\sqrt{n} = 3$, so $n = 9$ years of daily data. The intuition this teaches, and it's brutal: **a Sharpe of 1.0 — which sounds great — needs about four years of data just to clear the lowest bar of statistical respectability, and most backtests you'll be shown are far shorter.** Now you know why a three-year backtest with a Sharpe of 1.0, no matter how pretty the equity curve, deserves a raised eyebrow rather than a wire transfer.

## 3. The p-value, honestly

Here is the section the whole post is named for. The p-value is the most misunderstood number in quantitative finance, and getting it exactly right is the difference between a researcher who survives and one who blows up.

**The precise definition:** the p-value is the probability of observing a test statistic *at least as extreme as the one you got*, **assuming the null hypothesis is true.** In symbols, for a one-sided test, $p = P(\text{stat} \geq \text{observed} \mid H_0)$. In English for a trader: *if my strategy truly had no edge, how often would pure luck hand me a backtest that looks this good or better?*

Read that twice. The p-value is computed **in the null world** — the world where your strategy is worthless. It measures how comfortably your result fits inside ordinary noise. A small p-value means "this result would be rare under no-edge, so the no-edge assumption is starting to strain." A large p-value means "this result is totally ordinary under no-edge, so no-edge explains it fine."

The decision rule is mechanical: **if $p < \alpha$, reject the null** (call it real); **if $p \geq \alpha$, fail to reject** (call it unproven). With $\alpha = 0.05$, the line is at 0.05.

### What a p-value is NOT (read this even if you read nothing else)

This is where fortunes are lost. The p-value is **not** the probability that the null hypothesis is true. It is **not** the probability that your strategy is worthless. It is **not** the probability you'll lose money. Let me be blunt about the trap:

$$p = P(\text{data} \mid H_0) \quad \neq \quad P(H_0 \mid \text{data})$$

The p-value is $P(\text{data given no edge})$. What you actually *want* is $P(\text{no edge given the data})$ — the probability your strategy is dead, given what you saw. Those are different quantities, and flipping a conditional probability is like saying "most people who own a yacht are rich" implies "most rich people own a yacht." False.

To get from one to the other you need **Bayes' rule**, and crucially you need a **prior** — your belief, *before* seeing the data, about how likely a randomly-cooked-up strategy is to be real. And here's the killer fact for quants: most strategies you dream up *are* worthless. The base rate of real edges is low. When the base rate of true effects is low, even a "significant" p-value can correspond to a high probability that the result is a false positive. A p-value of 0.05 does **not** mean a 5% chance the strategy is dead — depending on your prior, the real probability it's dead can be 30%, 50%, even higher. We'll make this concrete in the multiple-testing section, where the base rate becomes the whole story.

A second thing it is not: a p-value of 0.04 is *not twice as convincing* as 0.08. The p-value is not a measure of effect size or of how big your edge is. A strategy can have a tiny, useless edge and a tiny p-value if $n$ is enormous, or a huge edge and a big p-value if $n$ is tiny. **Significance is not the same as importance.** Always look at the t-stat (which encodes effect size relative to noise) and the actual size of the edge in dollars, never the p-value alone.

> A p-value answers "how surprising is my data if I'm wrong about having an edge?" — never "how likely is it that I have an edge?" The second question needs a prior, and your prior on a random backtest should be skeptical.

#### Worked example: reading a p-value correctly and incorrectly

Your colleague runs one test on one strategy and reports **p = 0.03**, declaring "there's a 97% chance this strategy works." Where's the error, and what's the right reading?

The **wrong reading** treats $p = 0.03$ as $P(\text{no edge} \mid \text{data}) = 0.03$, hence "97% it works." That's the flipped-conditional trap.

The **right reading**: *if this strategy had no edge, only 3% of the time would noise produce a backtest this good or better.* That's it. To get the probability it actually works, bring in the base rate. Suppose, being honest, that **1 in 20** of the strategies your team invents has any real edge (a generous prior). Among 20 strategies, ~1 is real. The 19 dead ones, tested at $\alpha = 0.05$, throw up about $19 \times 0.05 \approx 1$ false positive. The 1 real one, with decent power, shows up as significant maybe 0.8 of the time. So among the significant results you have roughly **0.8 true and 1.0 false** — meaning a "significant" result here is real only about $0.8 / (0.8 + 1.0) \approx 44\%$ of the time. Not 97%. **A p = 0.03 result, given a realistic prior, is closer to a coin flip than a sure thing.** That gap between 44% and the naive 97% is, in dollar terms, the most expensive misunderstanding in the business.

## 4. The multiple-testing problem: how data mining manufactures edges

Everything so far assumed you ran *one* test. The reality of quant research is that you run *thousands*. You sweep parameters, try different universes, test momentum and mean-reversion and seasonality and a hundred signals. And the moment you test many things, the p-value's promise quietly breaks.

The intuition is a lottery. Buy one lottery ticket and your odds of winning are minuscule. Buy a million tickets and someone — you — is going to win, not from skill but from sheer volume. Testing strategies is buying lottery tickets where the "prize" is a backtest that looks significant by chance. Run enough tests and some *will* clear `p < 0.05`, guaranteed, even if every single strategy is worthless.

![Before and after comparison of false positives under one test versus one hundred tests](/imgs/blogs/hypothesis-testing-pvalues-math-for-quants-4.png)

The contrast above is the crux. On the left, a single test at $\alpha = 0.05$ has a 5% chance of fooling you — usually you're fine. On the right, a hundred tests of worthless strategies are expected to produce about five false winners, and the probability of getting *at least one* false positive rockets to nearly certainty.

Here's the arithmetic. If you run $m$ independent tests, each at $\alpha = 0.05$, and every strategy is genuinely worthless, the chance that *no* test fires falsely is $(1 - 0.05)^m = 0.95^m$. So the chance that **at least one** false positive appears — the *family-wise error rate* — is:

$$P(\text{at least one false positive}) = 1 - 0.95^m$$

Plug in numbers and it's alarming. For $m = 1$: 5%. For $m = 14$: about 51% — past 14 tests you're more likely than not to have at least one false winner. For $m = 100$: $1 - 0.95^{100} \approx 99.4\%$ — virtually guaranteed. And the *expected number* of false positives is just $m \times \alpha = 100 \times 0.05 = 5$. Test a hundred random ideas and five will, on average, masquerade as discoveries.

This is the engine behind the **replication crisis** — the broad finding across science (and finance) that a huge fraction of "significant" published results don't hold up when retested. In trading it has a specific name when it goes unaccounted-for: **backtest overfitting**. You can read the deep treatment in [overfitting, purged cross-validation, and the deflated Sharpe ratio](/blog/trading/quantitative-finance/overfitting-purged-cv-deflated-sharpe-quant-research); here we focus on the testing math that makes it happen.

#### Worked example: 100 random strategies and the cost of trading a false discovery

You generate **100 strategies from pure noise** — say, 100 random trading rules with no possible edge — and backtest each one. How many will look significant at the 5% level, and what does it cost if you trade one?

Expected false positives: $100 \times 0.05 = \textbf{5 strategies}$ will show `p < 0.05` by pure luck. The probability that at least one looks significant is $1 - 0.95^{100} \approx 99.4\%$ — you are essentially *guaranteed* to find a "winner." Worse, the temptation is to pick the *best-looking* of the five, which by construction has the most extreme lucky t-stat — the most convincing-looking pure fluke in the batch.

Now the dollars. You pick the best of those five mirages, allocate **\$5 million**, and trade it for a year. With no real edge, gross PnL drifts to zero, but you pay costs. At a modest 5 bps per round trip and 200 turns a year, that's $200 \times 5 = 1{,}000$ bps $= 10\%$ in costs, so **\$500,000** of capital incinerated on a strategy that was always going to be flat — plus the slippage, the financing, and the opportunity cost of \$5 million. The intuition, and the whole reason corrections exist: **find the prettiest result in a big enough pile of noise and it will look like a discovery and trade like a disaster.**

## 5. The honest checklist: how to test many strategies without fooling yourself

If running many tests inflates false positives, the fix is to make the bar *harder* in proportion to how many tests you ran. There are two main philosophies for doing this, plus a Sharpe-specific tool built for exactly this situation. But before the corrections, the procedure itself has to be honest. The order of operations matters as much as the math.

![Vertical stack of the honest testing procedure from stating hypotheses to reporting all trials](/imgs/blogs/hypothesis-testing-pvalues-math-for-quants-5.png)

The stack above is the discipline that makes the rest trustworthy. Read top to bottom: state the null and alternative *before* you look; pick $\alpha$ *before* you look; **count every trial you run** (this is the number the corrections need); compute the statistic; adjust for multiplicity; decide; and report *all* trials, not just the winner. The single most common way researchers fool themselves is skipping the "count every trial" step — running 200 variations, reporting the one that worked, and pretending it was the only test. That's not a calculation error; it's a sin of omission, and no formula can fix data you hid.

### Bonferroni: the blunt, safe correction

The **Bonferroni correction** is the simplest and most conservative fix: if you run $m$ tests and want an overall false-positive rate of $\alpha$, require each individual test to clear $\alpha / m$ instead of $\alpha$. Divide your threshold by the number of tests.

The logic is a clean upper bound: the chance of *any* false positive across $m$ tests is at most $m$ times the per-test rate, so shrinking each per-test rate to $\alpha/m$ keeps the total at or below $\alpha$. It controls the *family-wise error rate* — the probability of even one false positive — which is the strictest thing you can ask for.

Its virtue is that it's bulletproof and needs no assumptions. Its vice is that it's *brutal*: divide by 100 and your threshold becomes 0.0005, so real-but-modest edges get crushed (Type II errors soar). Bonferroni trades away power for safety. When the cost of a single false discovery is enormous — a big concentrated allocation — that trade is right. When you're screening hundreds of small signals and can tolerate a few mistakes, it's too harsh.

#### Worked example: Bonferroni versus the raw threshold for 20 tests

You test **20 strategies simultaneously** and want to keep your *overall* chance of even one false discovery at 5%. What threshold must each strategy's p-value clear, and what does that do to the t-stat bar?

Raw (uncorrected) threshold: $\alpha = 0.05$, corresponding to a t-stat of about **2.0**. Under this, the chance of at least one false positive across 20 worthless strategies is $1 - 0.95^{20} \approx 64\%$ — you'd be fooled more often than not.

Bonferroni threshold: $\alpha / m = 0.05 / 20 = \textbf{0.0025}$. To clear a two-sided p-value of 0.0025 you need a t-stat of about **3.0** (since the 0.0025 two-sided tail sits near 3 standard deviations). So the bar jumps from $t = 2$ to $t = 3$. Translate that with our Sharpe formula: a strategy with Sharpe 1.0 needed 4 years to hit $t = 2$, but now needs $n = 3^2 = \textbf{9 years}$ to hit $t = 3$. The intuition: **testing 20 things instead of one doesn't just split your attention — it raises the evidence bar for every single one of them, from "4 years of Sharpe-1.0 data" to "9 years."** That's the price of the lottery you bought by testing twenty tickets.

### False discovery rate (FDR): the smarter, less brutal correction

Bonferroni asks "what's the chance of *even one* false positive?" — often too strict when you're happy to make a few mistakes in exchange for catching more real effects. The **false discovery rate (FDR)**, controlled by the *Benjamini–Hochberg procedure*, asks a gentler question: "of the strategies I *declare* winners, what fraction are false?" You set a tolerable false-discovery proportion — say 10% — meaning you accept that 1 in 10 of your "discoveries" will be junk, in exchange for far more power to detect the real ones.

The mechanics, briefly: sort your $m$ p-values from smallest to largest, $p_{(1)} \le p_{(2)} \le \dots \le p_{(m)}$. Find the largest rank $k$ such that $p_{(k)} \le \frac{k}{m}\,q$, where $q$ is your target FDR (e.g. 0.10). Reject everything ranked $k$ and below. The threshold *slides*: the most extreme results get an easy bar and weaker ones a stricter one, so you don't crush every signal the way Bonferroni does.

FDR is the workhorse of modern large-scale testing — genomics, A/B testing at scale, and increasingly quant signal research — precisely because it scales to thousands of tests without throwing away every real edge. The tradeoff is honest and explicit: you knowingly let some false discoveries through, in a controlled proportion, to keep your power up. Bonferroni for the high-stakes single bet; FDR for the wide signal sweep.

| Correction | Question it answers | Bar at $m=100$, $\alpha=0.05$ | Strength | When to use |
| --- | --- | --- | --- | --- |
| None (raw) | Per-test false-positive rate | 0.05 | None | One pre-specified test only |
| Bonferroni | Chance of *any* false positive | 0.0005 | Strictest | A few high-stakes tests; big single allocation |
| Benjamini–Hochberg (FDR) | Fraction of *declared winners* that are false | Sliding, ~0.001–0.05 | Moderate | Wide signal sweeps; many small bets |
| Deflated Sharpe | Is the *best* Sharpe better than expected from this many trials? | Trial-adjusted | Sharpe-specific | Backtest selection from many variants |

### The deflated Sharpe ratio: the quant-native correction

Bonferroni and FDR are general-purpose. The **deflated Sharpe ratio (DSR)**, developed by Bailey and López de Prado, is built for the exact situation a quant faces: you tried many strategy variants and you're reporting the best Sharpe you found. The DSR asks the precise question: *given that I picked the maximum of $N$ trials, how much of this Sharpe should I expect from luck alone, and is my number meaningfully above that?*

The key insight is the **expected maximum**. If you run $N$ independent backtests of strategies with *zero* true edge, the *best* one will still show a positive Sharpe just by being the luckiest of the batch — and the more trials, the higher that lucky maximum climbs. There's a clean approximation for how high. The expected maximum Sharpe from $N$ trials of pure noise grows roughly like:

$$\mathbb{E}[\max \mathrm{SR}_N] \approx \sigma_{\mathrm{SR}} \left[(1-\gamma)\,\Phi^{-1}\!\left(1 - \tfrac{1}{N}\right) + \gamma\,\Phi^{-1}\!\left(1 - \tfrac{1}{N e}\right)\right]$$

where $\sigma_{\mathrm{SR}}$ is the standard deviation of Sharpe estimates across trials, $\Phi^{-1}$ is the inverse normal (it turns a probability into the number of standard deviations that hits it), $\gamma \approx 0.577$ is the Euler–Mascheroni constant, and $e$ is the base of natural logs. You don't need to love this formula — you need the takeaway it encodes: **the benchmark you must beat is not zero; it's the expected lucky maximum, and that benchmark rises with the number of trials.** The deflated Sharpe is then the probability that your *observed* Sharpe exceeds that trial-inflated benchmark, adjusted for the length and the skew and fat tails of your returns. A raw Sharpe of 2.0 that came from 1 trial is impressive; the same 2.0 picked as the best of 1,000 trials might sit *below* the expected lucky maximum and deflate to near-zero significance.

![Before and after comparison of a raw backtest Sharpe versus a deflated Sharpe after accounting for trials](/imgs/blogs/hypothesis-testing-pvalues-math-for-quants-7.png)

The before-and-after above tells the story in two columns: the headline number is the best of many tries and looks bulletproof; after you subtract the trials benchmark, the deflated Sharpe can collapse toward zero, exposing what was probably noise all along.

Two refinements make the deflated Sharpe sharper than a blunt Bonferroni divide, and both matter in practice. First, it accounts for the **length** of the backtest, not just the count of trials: a Sharpe estimated on twenty years of data is more trustworthy than the same Sharpe on two years, because the standard error of the estimate is smaller, and the DSR folds that directly in through the $\sqrt{n}$ term we built earlier. Second, it accounts for the **shape** of the returns — their skewness and kurtosis (fat tails). A strategy whose returns are negatively skewed and fat-tailed (small steady gains punctuated by rare large losses, the classic "picking up pennies in front of a steamroller" profile) has a *less reliable* Sharpe than its raw number suggests, because the disaster that defines its true risk may simply not have happened yet in the sample. The DSR deflates such a Sharpe further. A general-purpose correction like Bonferroni knows none of this; it only sees a count of tests. The deflated Sharpe is the multiple-testing correction that speaks fluent finance, which is exactly why it has become the standard tool in serious backtest due diligence.

#### Worked example: deflating a Sharpe 2.0 backtest found across 1,000 trials

Your researcher proudly presents a strategy with a backtested **annual Sharpe of 2.0** over **5 years**. Then you ask the question that matters: *how many variations did you try before this one?* The honest answer: **1,000** (every parameter combination in a grid search). Let's deflate it.

First, the t-stat ignoring the trials looks great: $t \approx 2.0 \times \sqrt{5} \approx 4.5$, well past any single-test bar. But that's a single-test view, and this wasn't a single test.

Now the trials adjustment. With $N = 1{,}000$ trials and a typical per-trial Sharpe standard deviation of about $\sigma_{\mathrm{SR}} \approx 0.5$ (over a 5-year sample), the expected *maximum* Sharpe from pure noise is roughly $\sigma_{\mathrm{SR}} \times \Phi^{-1}(1 - 1/1000) \approx 0.5 \times 3.1 \approx \textbf{1.55}$. So even with *no real edge*, the best of 1,000 random tries is expected to post a Sharpe around 1.55 by luck alone. Your observed 2.0 is only modestly above that 1.55 benchmark — not the heroic distance above *zero* it first appeared. The "edge" above the noise benchmark is just $2.0 - 1.55 = 0.45$, and once you account for the uncertainty in the estimate, the **deflated Sharpe lands around 0.6** in significance terms and the probability it's real drops well below the comfortable 95%. The haircut: a 2.0 that screamed "fund me" becomes a ~0.6-equivalent that whispers "probably luck." The intuition this drives home, and it's the thesis of the entire post: **a Sharpe ratio without its trial count is a number with no meaning — the same 2.0 is brilliant from one try and worthless from a thousand.**

## 6. P-hacking and the replication crisis, applied to backtests

We've covered the honest failures — running many tests and not correcting. P-hacking is the *dishonest* failure, and it's so easy to do accidentally that it deserves its own section. **P-hacking** is the practice — often unconscious — of tweaking your analysis until the p-value crosses below 0.05, then reporting only the version that worked, as if it were the plan all along.

![Tree taxonomy of testing errors splitting into Type I false positives and Type II false negatives](/imgs/blogs/hypothesis-testing-pvalues-math-for-quants-6.png)

The taxonomy above is worth re-reading here: every form of p-hacking ultimately manufactures **Type I errors** — false discoveries — and in backtests it's amplified by the data mining we just covered. The branches show the two error families and how mining worsens the false-positive side specifically.

In trading, p-hacking wears a dozen costumes, and you should recognize each one:

- **Universe shopping:** the signal doesn't work on the S&P 500, so you try the Russell 2000, then European small-caps, then crypto, until one universe shows significance — then report only that one.
- **Period cherry-picking:** the strategy is flat over 2010–2024, but it's stellar over 2015–2019, so you quietly restrict the backtest to those years.
- **Parameter optimization disguised as a single test:** you sweep the lookback window from 5 to 200 days, find that 47 days works, and report "our 47-day momentum signal" without mentioning the 195 windows you tried.
- **Outlier surgery:** the strategy loses money on three crisis days, so you "clean" them as anomalies, and suddenly the Sharpe doubles.
- **Optional stopping:** you keep extending the backtest until the t-stat happens to peek above 2, then stop and report.

Every one of these inflates false positives without ever showing up in the reported p-value, because the reported p-value pretends only one test was run. The fix is the discipline from the honest-checklist stack: **pre-register your hypothesis, count every trial, and report the failures alongside the winner.** A strategy that survives being decided *before* the data is seen is worth a hundred that were reverse-engineered to fit it.

The broader phenomenon is the **replication crisis**: across psychology, medicine, and economics, large fractions of published "significant" findings fail to reproduce when retested independently. In quant terms, the analog is the strategy that's brilliant in-sample and dead out-of-sample — the backtest that doesn't *replicate* in live trading. The market is the ultimate replication experiment, and it is merciless. This is exactly why disciplined shops insist on out-of-sample tests, walk-forward analysis, and the deflated Sharpe — they're all forms of demanding replication before committing capital. The companion piece on [backtesting done right](/blog/trading/quantitative-finance/backtesting-done-right-quant-research) walks through the operational guardrails in detail.

> The market doesn't care how good your backtest looked. It only ever runs one experiment — the future — and it grades on a curve you can't see until you've already paid the tuition.

#### Worked example: the in-sample to out-of-sample collapse

You build a strategy on data from **2010–2019** (in-sample), tuning until it shows a **Sharpe of 1.8**. You set aside **2020–2024** as untouched out-of-sample data and finally run it there. The result: a Sharpe of **0.3**. What happened, and what does the gap mean in dollars?

The in-sample 1.8 was inflated by every choice you made *to fit that data* — the parameters, the universe, the period. None of that fitting carries forward, so the out-of-sample 0.3 is the honest estimate of the real edge (and even that may be optimistic). The collapse from 1.8 to 0.3 is the *overfitting tax* made visible. In dollars: if you'd sized the position for the promised Sharpe 1.8 — say targeting 15% volatility on **\$10 million** expecting $1.8 \times 0.15 \times \$10\text{M} = \$2.7\text{M}$ a year — you'd actually earn about $0.3 \times 0.15 \times \$10\text{M} = \$450{,}000$, a **\$2.25 million** annual shortfall against the plan, with risk sized for a strategy six times better than the one you have. The intuition: **the only Sharpe that ever spends is the out-of-sample one; the in-sample number is marketing.**

## Common misconceptions

**"A p-value of 0.05 means there's a 95% chance my strategy works."** No — this is the flipped-conditional error, and it's the most expensive mistake in the field. The p-value is $P(\text{data} \mid \text{no edge})$, the probability of your result *if the strategy is worthless*. The probability the strategy works given the data is a completely different quantity that requires a prior on how often your ideas are real — and since most strategy ideas are duds, even a 0.05 result often corresponds to a coin-flip or worse chance of being real.

**"A lower p-value means a bigger or better edge."** No — the p-value measures *statistical significance*, not *economic importance*. With enough data, a microscopic, untradeable edge can produce a tiny p-value, while a genuinely large edge measured on a short sample can show a fat p-value. Always read the effect size (the actual Sharpe, the dollar edge after costs) separately. A significant strategy with a 2-bp edge that costs 3 bps to trade is significant *and* unprofitable.

**"Failing to reject the null proves the strategy doesn't work."** No — like a "not guilty" verdict, failing to reject means the evidence wasn't strong enough, not that the null is true. A real edge tested on too little data (low *power*) routinely fails to reach significance. Absence of evidence is not evidence of absence; you may simply need more data, which the standard-error math quantifies exactly.

**"I ran one test, so I don't have a multiple-testing problem."** Usually false. If you tried several parameter settings, universes, or date ranges and reported the best, you ran *many* tests even if only one made it into the memo. The number of trials is the number of *things you tried*, not the number you *reported*. Honesty about the trial count is what the deflated Sharpe and Bonferroni both depend on.

**"Statistical significance means the strategy will make money."** No — significance is a statement about the past sample under the null, computed before any out-of-sample test, before slippage, before regime change, before the alpha decays as others find it. Significance is a *necessary screening filter*, not a *sufficient guarantee*. Plenty of significant in-sample strategies die instantly in production.

**"Bonferroni is always the right correction."** No — Bonferroni controls the chance of *any* false positive, which is often too strict and buries real edges (high Type II error). When you're screening many signals and can tolerate a known fraction of false discoveries, FDR gives you far more power. The right correction depends on the dollar cost of a false positive versus a missed discovery in *your* specific situation.

## How it shows up in real markets

### 1. The 2011 "300+ factors" reckoning

For decades, academic finance published hundreds of "factors" — characteristics like value, momentum, and size that supposedly predicted returns. By the early 2010s researchers had catalogued over 300 of them. In a landmark 2014 paper, Campbell Harvey, Yan Liu, and Heqing Zhu argued that, given how many factors had been tested across the literature, the conventional t-stat bar of 2 was far too lax: with hundreds of trials in the field, a factor needed a t-stat closer to **3** to be credible. The mechanism is exactly the multiple-testing inflation from this post — an entire profession had been running thousands of tests and reporting the winners, with no family-wise correction. Many celebrated factors, retested honestly, faded. The lesson: when a whole field shares a dataset, the *field's* trial count is what should set the bar, not any single paper's.

### 2. The momentum factor that mostly replicated

Not every story is a debunking. Cross-sectional momentum — buying recent winners, shorting recent losers — was documented by Jegadeesh and Titman in 1993, and unlike many factors it has *replicated* across decades, asset classes, and continents, which is the gold standard for an effect being real rather than mined. That out-of-sample, out-of-period, out-of-geography survival is precisely the replication test the p-value alone can't give you. The lesson cuts both ways: rigorous testing isn't about being a nihilist who believes nothing — it's about demanding that an edge prove itself the way momentum did, across data it was never fit to.

### 3. LTCM and the convergence trades that "couldn't" lose

Long-Term Capital Management, run by Nobel laureates, built trades on relationships that had held for years of historical data — convergence bets whose backtests showed tiny, consistent profits with stellar Sharpe ratios. In 1998 the Russian default and the ensuing flight to quality blew those relationships apart, and LTCM lost roughly **\$4.6 billion** in months. Part of the failure was a testing failure: the backtests were estimated on a calm period that didn't contain the kind of crisis that breaks convergence, an extreme form of period cherry-picking. A Sharpe estimated on data that excludes the regime that kills you is not a real Sharpe. The lesson: significance computed on a sample missing the disaster scenario is worse than useless — it's actively dangerous.

### 4. The "quant quake" of August 2007

In a few days in August 2007, many statistical-arbitrage funds running similar mean-reversion strategies suffered violent, simultaneous losses as crowded positions unwound. Strategies with gorgeous multi-year backtests and high t-stats lost double-digit percentages in days. The testing lesson is subtle: a backtest assumes *you* are the only one running the strategy, but a significant edge that's easy to find will be found by others, and the crowding it creates is a risk no in-sample p-value can measure. Significance on historical data says nothing about how many other desks are about to trade the same signal.

### 5. The deflated Sharpe in modern fund due diligence

Today, sophisticated allocators and quant shops routinely apply the deflated Sharpe ratio (or close cousins) when a manager presents a backtest. The first question in serious due diligence is no longer "what's the Sharpe?" but "**how many configurations did you test to get it?**" A 2.5 Sharpe from one pre-registered hypothesis is taken seriously; a 2.5 Sharpe that is the best of 10,000 grid-searched variants is deflated, often to statistical irrelevance. This is the multiple-testing math from this post operationalized as a business process — the difference between capital allocated and capital declined.

### 6. A/B testing at trading-adjacent tech firms

Outside pure trading, the same math governs product experimentation at scale. A large platform running thousands of simultaneous A/B tests faces exactly the family-wise inflation we computed: at 5% per test, thousands of tests guarantee hundreds of false "wins." Mature experimentation teams apply FDR control and pre-registration for the same reason quants do — to stop shipping features (or trading strategies) whose apparent lift was noise. It's a reminder that hypothesis testing isn't a finance quirk; it's the universal grammar of deciding under uncertainty, and trading is just one dialect.

## When this matters to you

If you ever build, evaluate, or invest in a quantitative strategy — your own or someone else's — the single habit from this post that will save you the most money is this: **before you believe any backtest, ask how many things were tried to produce it, and adjust your skepticism accordingly.** A pretty equity curve and a `p < 0.05` are necessary but nowhere near sufficient. The t-stat, the trial count, the out-of-sample result, and the deflated Sharpe are what separate a real edge from an expensive mirage.

Concretely, carry these four reflexes:

- **Translate any Sharpe and backtest length into a t-stat in your head** with $t \approx \mathrm{SR}\cdot\sqrt{n}$, and don't get excited until it clears 2 (and ideally 3 if the field has been mining the same data).
- **Never read a p-value as "the chance the strategy works."** It's the chance of your data under no edge. The chance it works needs a skeptical prior and an out-of-sample test.
- **Count every trial** — every parameter, universe, and period you tried — and apply Bonferroni for high-stakes single bets or FDR for wide screens.
- **Demand replication.** The market only runs one experiment, the future. An edge that survived out-of-sample, out-of-period, and out-of-asset-class data it was never fit to is worth a hundred that were tuned into existence.

This is educational material about statistical mechanics, not investment advice — no strategy here is a recommendation, and every approach that can make money can lose it. But the discipline of honest hypothesis testing is the closest thing quant trading has to a survival skill. The researchers who last are not the ones who find the most impressive backtests; they're the ones who are hardest to fool, especially by themselves.

For further reading, work through these companion pieces on this blog and the foundational sources behind them:

- [Hypothesis testing and p-values for quant interviews](/blog/trading/quantitative-finance/hypothesis-testing-pvalues-quant-interviews) — the same ideas drilled as interview problems, with the standard-error and CLT machinery worked in detail.
- [Overfitting, purged cross-validation, and the deflated Sharpe ratio](/blog/trading/quantitative-finance/overfitting-purged-cv-deflated-sharpe-quant-research) — the deeper treatment of the multiple-testing and deflation math, with cross-validation that respects time.
- [Backtesting done right](/blog/trading/quantitative-finance/backtesting-done-right-quant-research) — the operational guardrails (walk-forward, out-of-sample discipline, cost modeling) that turn this theory into a research process.

Beyond the blog, the primary sources are worth your time: Harvey, Liu, and Zhu's "…and the Cross-Section of Expected Returns" (2014) for the factor-mining reckoning; Bailey and López de Prado's work on the deflated Sharpe ratio (2014) for the trial-adjusted significance machinery; and Benjamini and Hochberg's 1995 paper for the false discovery rate. Read them with the courtroom in your head — every one of them is, at bottom, an argument about how hard the evidence should have to work before you convict a strategy of having an edge.
