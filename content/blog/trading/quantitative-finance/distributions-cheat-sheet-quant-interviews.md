---
title: "Probability distributions for quant interviews: when each one shows up"
date: "2026-06-14"
publishDate: "2026-06-14"
description: "A first-principles map of the eight workhorse probability distributions and the two limit theorems that cover almost every quant interview question, built from zero with worked dollar examples and the recognition reflexes top trading desks actually test."
tags:
  [
    "probability-distributions",
    "quant-interviews",
    "binomial-distribution",
    "poisson-process",
    "normal-distribution",
    "central-limit-theorem",
    "exponential-distribution",
    "lognormal",
    "geometric-distribution",
    "quantitative-finance"
  ]
category: "trading"
subcategory: "Quantitative Finance"
author: "Hiep Tran"
featured: true
readTime: 41
---

> [!important]
> **TL;DR** — Quant interviewers do not want you to recite the formula for a Poisson distribution. They want you to hear a situation, recognize *which* distribution it is, and reason with it out loud. Eight distributions plus two limit theorems cover almost everything they ask.
>
> - **The whole subject is a recognition reflex.** "Trades until your first fill" is Geometric. "Orders per second at the exchange" is Poisson. "Total P&L over a thousand trades" is Normal, by the Central Limit Theorem. Learn the *mechanism* behind each, and the right distribution announces itself.
> - **The eight workhorses split cleanly.** Discrete counting: Bernoulli, Binomial, Geometric, Negative binomial, Poisson. Continuous timing and magnitude: Uniform, Exponential, Normal, Lognormal.
> - **Two limit theorems glue the map together.** The *Poisson limit* says many rare chances ($n$ large, $p$ small, $np$ fixed) turn a Binomial into a Poisson. The *Central Limit Theorem* says the sum of many independent things is Normal, whatever they individually look like.
> - **The one number to remember:** the mean wait for a first success with probability $p$ is $1/p$. If your order fills 25% of the time, you expect **4 attempts** before a fill. Almost every waiting-time question reduces to this.
> - **Where the model breaks matters as much as where it holds.** Real market returns have *fat tails*: a "6-sigma" day should be a once-in-500-million-year event under the Normal, yet markets deliver them every few years. Knowing this is the difference between a junior answer and a desk-ready one.

Here is the kind of question that gets asked in the first ten minutes of a quant interview, and it sounds deceptively casual. "You're sending orders into a market and each one has a 25% chance of getting filled. How many orders do you expect to send before you get your first fill?" The candidate who freezes and tries to write out an infinite sum is already behind. The candidate who says "that's geometric, so $1/p$, four orders" and then *explains why* is the one who gets to the next round.

That is the entire game. Interviewers at the firms that ask these questions — Jane Street, Two Sigma, Citadel, DE Shaw, Optiver, SIG, Jump, Hudson River Trading — are not testing whether you memorized a probability textbook. They are testing whether you can map a messy real-world situation onto the right mathematical object *fast*, and then reason cleanly inside it. Trading is exactly this skill applied at speed: a situation arrives (an order book shifts, a number prints, a counterparty does something strange), and you have to instantly know what kind of randomness you are looking at and what it implies for a price.

![The interview skill is recognition, not recall: read the mechanism behind a situation, decide discrete or continuous, then pick the distribution and reason with it](/imgs/blogs/distributions-cheat-sheet-quant-interviews-1.png)

The diagram above is the mental model for this entire article. A situation arrives. You ask one question — *am I counting things, or timing things?* — which mostly decides discrete versus continuous. You match the situation's *mechanism* to one of eight distributions. Then you reason with it. The eight workhorses plus two limit theorems sitting at the bottom of that figure are the whole toolkit. Master the mechanism behind each one — not the formula, the *story* — and you will recognize them in disguise the way a chess player sees a fork.

We will build everything from absolute zero. No probability background is assumed. By the end you will be able to look at any of the classic interview prompts, name the distribution in one breath, compute the answer with friendly numbers, and — the part that actually lands offers — narrate the reasoning the way an interviewer wants to hear it. This is educational material about how randomness is modeled; none of it is financial advice.

## Foundations: random variables, mass, density, and the two summaries

Before any distributions, four pieces of vocabulary. Skip none of them; everything else rests here. We will keep the language plain and attach a number to every idea.

### A random variable is a number that depends on chance

A **random variable** is just a number whose value is decided by some random process. Roll a die: the number that comes up is a random variable. Send an order: whether it fills (call it $1$) or not (call it $0$) is a random variable. Watch the clock until the next order arrives: that waiting time, in seconds, is a random variable. We write random variables with capital letters like $X$ or $N$ or $T$, and a *specific value* they might take with lowercase letters like $k$ or $t$.

The single most important fork in the road is this: a random variable is either **discrete** or **continuous**.

- A **discrete** random variable can only land on separated values you could list: $0, 1, 2, 3, \dots$. Counts are always discrete. "Number of fills," "number of orders this second," "number of trades until the first profit" — you can't have 2.7 fills.
- A **continuous** random variable can land anywhere in a range, including all the decimals in between. Times, prices, returns, and magnitudes are continuous. "Seconds until the next order" could be $3$, or $3.4$, or $3.41592$.

Almost every interview question telegraphs which one it is. *Counting* → discrete. *Timing or measuring* → continuous. That single classification already cuts your eight choices in half.

### Discrete: the probability mass function (pmf)

For a discrete random variable, you describe it completely by listing the probability of each value it can take. That list is the **probability mass function**, or **pmf**. We write $P(X = k)$ — "the probability that $X$ equals $k$." A pmf has two rules: every probability is between $0$ and $1$, and they all add up to exactly $1$ (the variable has to land *somewhere*).

For one fair die, the pmf is $P(X = k) = 1/6$ for each $k$ in $\{1, 2, 3, 4, 5, 6\}$. Six bars, each of height $1/6$, summing to $1$. That picture — a row of bars whose heights are probabilities — is the visual you should attach to the word "discrete."

### Continuous: the probability density function (pdf)

Here is the part that trips up beginners, and where interviewers love to probe. For a *continuous* variable, the probability of any *exact* value is zero. The chance the next order arrives at *exactly* $3.000000\dots$ seconds is $0$, because there are infinitely many instants it could pick. So we cannot list probabilities value-by-value. Instead we use a **probability density function**, or **pdf**, written $f(x)$.

Density is not probability. Density is probability *per unit of x*. To get an actual probability you have to ask about a *range*, and the probability is the **area under the density curve** over that range. The chance the next order arrives between $3$ and $4$ seconds is the area under $f(t)$ from $t = 3$ to $t = 4$. The total area under any pdf is $1$, just as a pmf's bars sum to $1$.

> The mental swap: a pmf gives you heights you can *read off*; a pdf gives you a curve whose *area* you have to measure. Probability for a continuous variable always lives in an interval, never a point.

### The cumulative distribution function (CDF)

There is one object that works for both discrete and continuous variables, and quants reach for it constantly: the **cumulative distribution function**, or **CDF**, written $F(x) = P(X \le x)$ — "the probability that $X$ is *at most* $x$." It accumulates probability as you sweep from left to right, starting at $0$ on the far left and climbing to $1$ on the far right. The CDF is how you answer "what's the chance it's *less than* this / *more than* this," which is most of what trading risk questions actually want. $P(X > x) = 1 - F(x)$, and that complement trick — "probability of at least one" equals "one minus probability of none" — is the single most reused move in the entire interview canon.

### Two summaries: expectation and variance

You rarely need the whole distribution to answer a question. Two numbers usually suffice.

The **expectation** (or **expected value**, or **mean**), written $E[X]$, is the long-run average — the center of mass of the distribution. For a discrete variable it's the probability-weighted sum of the values:

$$E[X] = \sum_k k \cdot P(X = k)$$

Here $k$ ranges over every value $X$ can take, and $P(X = k)$ is its probability. Expectation answers "on average, what do I get?" If a bet pays $\$10$ with probability $0.3$ and $\$0$ otherwise, its expected payoff is $0.3 \times \$10 = \$3$. The most important property: expectation is **linear**. $E[X + Y] = E[X] + E[Y]$ *always*, even when $X$ and $Y$ are dependent. This is the workhorse that lets you break a scary sum into easy pieces.

The **variance**, written $\operatorname{Var}(X)$, measures spread — how far values typically land from the mean. Its square root, the **standard deviation** $\sigma$, is in the same units as $X$ and is the one you quote ("the daily move is about one standard deviation, $\$2$"). The key property for trading: for **independent** variables, variances *add*. $\operatorname{Var}(X + Y) = \operatorname{Var}(X) + \operatorname{Var}(Y)$ when $X$ and $Y$ don't influence each other. Standard deviations do *not* add — they grow with the square root of the count, which is why risk over $n$ independent trades scales like $\sqrt{n}$, not $n$. Hold onto that; it is the seed of the Central Limit Theorem and half the desk intuition in this article.

With those four ideas — random variable, pmf/pdf, CDF, mean/variance — we can build all eight distributions. Each section below follows the same rhythm: the *story* (the mechanism that produces it), the *math* (kept minimal and defined), and a *worked example* with real numbers.

## Bernoulli and Binomial: counting successes in n trials

### Bernoulli: the atom of all randomness

The **Bernoulli distribution** is the simplest random variable there is: a single yes/no trial. It takes value $1$ ("success") with probability $p$ and value $0$ ("failure") with probability $1 - p$. One coin flip. One order that either fills or doesn't. One trade that's either a winner or a loser. That's it.

Its summaries are clean. The mean is $E[X] = p$ (a success worth $1$ happens a fraction $p$ of the time). The variance is $p(1-p)$, which is largest at $p = 0.5$ — a fair coin is the most unpredictable, and a near-certain event ($p$ near $0$ or $1$) has almost no variance. Every other discrete distribution in this article is, at heart, a pile of Bernoulli trials assembled in some way. It is the atom.

### Binomial: add up n independent Bernoullis

Now run $n$ independent Bernoulli trials, each with the same success probability $p$, and count the total number of successes. That count follows the **Binomial distribution**, written $\text{Binomial}(n, p)$. Flip a coin $5$ times, count the heads. Send $5$ orders, count the fills. Make $10$ trades, count the winners.

![A Binomial is just n independent Bernoulli trials added up: one trial is a single green-or-red bar, and five trials spread into a bell of counts](/imgs/blogs/distributions-cheat-sheet-quant-interviews-2.png)

The left panel is a single Bernoulli with $p = 0.6$: just two bars, a $0.4$ chance of failure and a $0.6$ chance of success. The right panel is what happens when you run that trial five times and count successes — the probability spreads across $k = 0$ through $k = 5$, peaking near $k = 3$ (because $5 \times 0.6 = 3$ is the average number of successes). The pmf is

$$P(X = k) = \binom{n}{k} p^k (1-p)^{n-k}$$

where $\binom{n}{k} = \frac{n!}{k!(n-k)!}$ is the number of ways to choose *which* $k$ of the $n$ trials succeeded, $p^k$ is the probability those $k$ all succeed, and $(1-p)^{n-k}$ is the probability the other $n - k$ all fail. You almost never compute this by hand in an interview — you reach for the mean and variance instead, which are beautifully simple: $E[X] = np$ and $\operatorname{Var}(X) = np(1-p)$. (Both follow instantly from "expectation and variance add over the $n$ independent Bernoulli atoms.")

#### Worked example: a $ payoff under a binomial count

You run a strategy that makes exactly **10 independent trades** a day. Each trade is a winner with probability $p = 0.6$, and a winner pays you $\$50$ while a loser pays $\$0$. What is your expected daily profit, and how spread out is it?

Let $X$ be the number of winners, so $X \sim \text{Binomial}(10, 0.6)$. The expected number of winners is $E[X] = np = 10 \times 0.6 = 6$ winners. Your profit is $\$50 \times X$, and by linearity of expectation your expected daily profit is

$$E[\$50 \cdot X] = \$50 \times E[X] = \$50 \times 6 = \$300.$$

For the spread, the variance of the *count* is $np(1-p) = 10 \times 0.6 \times 0.4 = 2.4$, so the standard deviation of the count is $\sqrt{2.4} \approx 1.55$ winners. In dollars, multiplying a random variable by a constant multiplies its standard deviation by that constant, so your daily profit has standard deviation $\$50 \times 1.55 \approx \$77$. So you make about $\$300$ a day, give or take roughly $\$77$. The one-sentence intuition: **a Binomial count's mean is $np$ and its dollar payoff just scales that mean and spread by the per-success dollar amount.**

## Geometric and negative binomial: waiting for the first (or r-th) success

The Binomial fixes the number of trials and counts successes. Flip the question around: fix the number of successes you want, and count how many *trials it takes* to get there. That flip produces the waiting-time distributions, and they are interview gold because they sound hard and collapse to one line.

### Geometric: trials until the first success

Keep running independent Bernoulli trials, each succeeding with probability $p$, until the *first* success. The number of trials $N$ you needed follows the **Geometric distribution**. Its pmf is

$$P(N = k) = (1-p)^{k-1} \, p$$

read as "the first $k - 1$ trials all failed (probability $(1-p)^{k-1}$) and the $k$-th succeeded (probability $p$)." Each extra trial multiplies the probability by another factor of $(1-p)$, so the bars decay by a constant ratio — geometric decay, hence the name.

![Geometric decay: each extra failed trial multiplies the probability by the same factor, so the bars decay geometrically, with mean 1/p](/imgs/blogs/distributions-cheat-sheet-quant-interviews-4.png)

The one fact to burn into memory is the mean: $E[N] = 1/p$. If something succeeds a fraction $p$ of the time, you wait $1/p$ attempts on average for the first one. A $25\%$ fill rate means $4$ attempts on average; a $10\%$ rate means $10$. The figure shows $p = 0.25$, where the bars decay by a factor of $0.75$ each step and the mean sits at $4$ trials. The Geometric is also the *only discrete* distribution that is **memoryless**: having already failed $20$ times tells you nothing — your remaining wait is still a fresh Geometric with mean $1/p$. (We'll meet its continuous twin, the Exponential, shortly.)

#### Worked example: expected number of trades until your first fill

You are quoting passively in a market and your resting order gets filled on any given "look" with probability $p = 0.2$. Looks are independent. How many looks do you expect before your first fill, and what's the chance you're *still* unfilled after $10$ looks?

The number of looks until the first fill is $N \sim \text{Geometric}(0.2)$, so the expected wait is

$$E[N] = \frac{1}{p} = \frac{1}{0.2} = 5 \text{ looks}.$$

For "still unfilled after $10$ looks," use the complement trick. Being unfilled after $10$ looks means all $10$ failed, and each fails with probability $1 - p = 0.8$:

$$P(N > 10) = (0.8)^{10} \approx 0.107.$$

So about a $10.7\%$ chance you're still waiting after ten looks. If each fill is worth $\$30$ of edge to you, then over a long session your expected edge per resting order is just $\$30$ (you eventually fill with probability $1$), but the *timing* is geometric — sometimes instant, sometimes a long wait, with the average at $5$ looks. The intuition: **waiting for the first success is Geometric, the mean wait is $1/p$, and "still waiting after $k$" is just $(1-p)^k$.**

### Negative binomial: trials until the r-th success

Generalize once more. Instead of stopping at the first success, run until the $r$-th success and count the trials. That's the **Negative binomial distribution**. Because the waits between successive successes are independent Geometric variables, the mean just adds up: the expected number of trials to collect $r$ successes is $r/p$. If a single fill takes $5$ looks on average, then collecting $3$ fills takes $3 \times 5 = 15$ looks on average. Interviewers use the negative binomial to check whether you understand that *independent waiting times add* — the same additivity that powers the whole article. You rarely need its full pmf; you need "expected trials to $r$ successes is $r/p$, because it's $r$ stacked Geometrics."

## Poisson: rare events, the limit of the Binomial, and the process

The **Poisson distribution** is the single most useful distribution on a trading desk, because it is the natural model for *counts of events arriving over time*: orders hitting an exchange, trades printing, packets arriving, customers walking in. It counts how many of some "rare-ish but frequent-in-aggregate" event happen in a fixed window.

### The story: many chances, each unlikely

Imagine an event that has a tiny chance of happening in any given instant, but there are an enormous number of instants. Orders into a busy market are like this: in any given microsecond almost nothing happens, but across a whole second thousands of microseconds add up to a steady stream. The Poisson distribution is what you get in this limit. It has a single parameter, $\lambda$ (lambda), the **average number of events per window**. Its pmf is

$$P(X = k) = \frac{\lambda^k e^{-\lambda}}{k!}$$

where $k$ is the count, $e \approx 2.718$ is Euler's number, and $k!$ is $k$ factorial. The defining property — and the reason interviewers love it — is that the **mean and variance are both equal to $\lambda$**. If orders arrive at $\lambda = 100$ per second on average, the count in a one-second window has mean $100$ and standard deviation $\sqrt{100} = 10$.

### The Poisson limit of the Binomial

Where does the Poisson come from? It is the Binomial pushed to an extreme. Take $\text{Binomial}(n, p)$ and let the number of trials $n$ grow huge while the success probability $p$ shrinks, holding the average $np = \lambda$ fixed. In that limit the Binomial *becomes* the Poisson with parameter $\lambda$.

![The Poisson limit: hold the mean np = 3 fixed and let n grow while p shrinks, and the Binomial bars march onto the Poisson shape](/imgs/blogs/distributions-cheat-sheet-quant-interviews-3.png)

The figure makes the convergence visible. All three panels have mean $3$. The left is $\text{Binomial}(10, 0.3)$ — recognizably Binomial, a bit lumpy. The middle is $\text{Binomial}(50, 0.06)$ — more trials, smaller probability, same mean — and it already looks almost identical to the right panel, which is the true $\text{Poisson}(3)$. This is why the Poisson is the right tool for "rare events with many chances": you don't need to know $n$ and $p$ separately, just their product. A practical rule of thumb interviewers like: when $n \ge 20$ and $p \le 0.05$, the Poisson approximation to the Binomial is excellent, and it replaces an ugly $\binom{n}{k}$ computation with a clean $\lambda^k e^{-\lambda}/k!$.

### The Poisson process

Spread that idea across continuous time and you get the **Poisson process**: events landing on a timeline at a constant average rate $\lambda$ per unit time, each independent of the others. This is *the* canonical model for order flow.

![The Poisson process: marks land at rate lambda, counts in any window are Poisson and the gaps between marks are exponential, so one process produces two distributions](/imgs/blogs/distributions-cheat-sheet-quant-interviews-10.png)

The timeline above shows the two faces of one process. Count the green marks in *any* fixed window and that count is $\text{Poisson}(\lambda \times \text{window length})$ — the amber box highlights a $1.5$-second window containing $2$ orders. Measure the *gaps* between consecutive marks (the blue brackets below the axis) and those inter-arrival times are Exponentially distributed with mean $1/\lambda$. The same process gives you a discrete count distribution and a continuous waiting-time distribution at the same time. That duality is a favorite interview pivot: "you've told me orders arrive as a Poisson process at rate $\lambda$ — what's the distribution of the time between orders?" The answer is always Exponential.

#### Worked example: orders arriving as Poisson, probability of at least k in a window

Orders arrive at a venue as a Poisson process at rate $\lambda = 3$ orders per second. (a) What's the probability that *exactly* $2$ orders arrive in the next second? (b) What's the probability that *at least one* order arrives in the next second? (c) Over a $2$-second window, what's the expected count and its standard deviation?

(a) With $\lambda = 3$ for a one-second window, the count $X \sim \text{Poisson}(3)$ and

$$P(X = 2) = \frac{3^2 e^{-3}}{2!} = \frac{9 \times 0.0498}{2} \approx 0.224.$$

So about a $22.4\%$ chance of exactly two orders.

(b) For "at least one," use the complement — "at least one" is "one minus the chance of none":

$$P(X \ge 1) = 1 - P(X = 0) = 1 - \frac{3^0 e^{-3}}{0!} = 1 - e^{-3} \approx 1 - 0.0498 = 0.950.$$

A $95\%$ chance of seeing at least one order in any given second.

(c) Poisson counts add for disjoint windows, so a $2$-second window has parameter $\lambda \times 2 = 6$. The expected count is $6$ and, because a Poisson's variance equals its mean, the standard deviation is $\sqrt{6} \approx 2.45$ orders. The intuition: **counts over a window are Poisson($\lambda \times$ window), "at least one" is $1 - e^{-\lambda}$, and the mean and standard deviation are $\lambda$ and $\sqrt{\lambda}$.**

## Uniform: every value equally likely

The **Uniform distribution** is the "no information beyond the range" distribution: every value in some interval is equally likely. It comes in two flavors.

The **discrete uniform** puts equal probability on a finite set of values — a fair die is $\text{Uniform}\{1, \dots, 6\}$, each with probability $1/6$. The mean is just the midpoint of the values.

The **continuous uniform**, written $\text{Uniform}(a, b)$, spreads probability evenly across the interval from $a$ to $b$. Its density is flat: $f(x) = \frac{1}{b - a}$ for $x$ between $a$ and $b$, and zero outside. The flat-top rectangle is its signature shape. Its mean is the midpoint $\frac{a + b}{2}$, and its variance is $\frac{(b-a)^2}{12}$ — that $12$ in the denominator is worth memorizing, because it shows up constantly in interview estimation problems. The continuous uniform is also the engine behind every random-number generator and every Monte Carlo simulation: you generate a $\text{Uniform}(0, 1)$ and transform it into whatever distribution you need.

Interviewers use the uniform as a building block. "You pick a random point uniformly on a stick of length $1$ and break it — what's the expected length of the longer piece?" "Two people agree to meet between noon and 1pm, each arriving at a uniform random time — what's the chance they meet within $15$ minutes?" These *order-statistics-on-uniforms* puzzles are a whole genre, and the flat density is what makes the geometry tractable. If you want to go deeper on those, see [order statistics and uniform tricks for quant interviews](/blog/trading/quantitative-finance/order-statistics-uniform-tricks-quant-interviews).

#### Worked example: expected longer piece of a broken stick

You have a stick of length $1$. You pick a point $U$ uniformly at random along it (so $U \sim \text{Uniform}(0,1)$) and snap it there, giving two pieces of length $U$ and $1 - U$. What is the expected length of the *longer* piece?

The longer piece has length $\max(U, 1-U)$. By symmetry, whichever side the break lands on, the longer piece is always at least $0.5$. When the break is at position $U < 0.5$ the longer piece is $1 - U$; when $U > 0.5$ it is $U$. Averaging over the uniform break point, the expected longer piece works out to

$$E[\max(U, 1-U)] = \tfrac{3}{4} = 0.75.$$

You can see it without calculus: the longer piece ranges uniformly from $0.5$ (break in the exact middle) to $1$ (break at an end), and the average of a quantity uniform on $0.5$ to $1$ is the midpoint, $0.75$. So if the stick were a $\$100$ bill being split, the bigger half averages $\$75$ and the smaller half averages $\$25$. The intuition: **uniform randomness plus a symmetry argument cracks order-statistics puzzles without integrals.**

## Exponential: continuous memorylessness, the dual of the Poisson process

The **Exponential distribution** models the *waiting time* until the next event in a Poisson process. If events arrive at rate $\lambda$ per unit time, the time $T$ until the next one is $\text{Exponential}(\lambda)$, with density $f(t) = \lambda e^{-\lambda t}$ for $t \ge 0$. Its mean is $E[T] = 1/\lambda$ — if orders arrive at $3$ per second, you wait $1/3$ of a second on average for the next one. It is the continuous twin of the Geometric: the Geometric counts *discrete trials* until a success, the Exponential measures *continuous time* until an event.

### Memorylessness, the property interviewers probe

The Exponential's defining and counterintuitive property is that it is **memoryless**. Formally, $P(T > s + t \mid T > s) = P(T > t)$: given that you've *already* waited $s$ seconds with no event, the probability you wait at least $t$ more is exactly the same as if you'd just started. The wait "forgets" how long it's been going.

![Exponential memorylessness: after waiting 6 seconds, the remaining wait is a fresh exponential with the same mean, so the green tail is an identical copy of the original blue curve](/imgs/blogs/distributions-cheat-sheet-quant-interviews-5.png)

The figure shows it directly. The blue curve is the full Exponential density with mean $4$ seconds. The red dashed line marks the moment you've *already* waited $6$ seconds with nothing happening. The green curve is the distribution of your *remaining* wait from that point — and it is an exact rescaled copy of the original blue curve, just shifted to start at $6$. Your remaining wait has the same mean of $4$ seconds it had at the start. The clock does not "remember" that you've been waiting. This is why it's the right model for genuinely random arrivals: an order is no more "due" because none has come for a while.

Interviewers weaponize this. "A bus comes on average every $10$ minutes, exponentially distributed. You've already been waiting $10$ minutes. What's your expected additional wait?" The trap answer is "$0$, it's overdue." The correct answer is "$10$ more minutes — it's memoryless." (Real buses are not exponential, which is exactly why this is a good test of whether you're reasoning from the *model* or from intuition about buses.)

#### Worked example: expected wait and the memoryless trap

Trades print on a tape as a Poisson process at $\lambda = 2$ trades per second, so the gap between trades is $\text{Exponential}(2)$. (a) What's the expected time until the next trade? (b) You've been watching for $1.5$ seconds with no print. Now what's your expected additional wait? (c) What's the probability the next gap exceeds $2$ seconds?

(a) The mean of an $\text{Exponential}(\lambda)$ is $1/\lambda = 1/2 = 0.5$ seconds.

(b) By memorylessness, the $1.5$ seconds you already waited are irrelevant. Your expected *additional* wait is still $0.5$ seconds — exactly the same as at the start. This is the answer that separates candidates who memorized "memoryless" from those who actually believe it.

(c) The survival probability of an Exponential is $P(T > t) = e^{-\lambda t}$, so

$$P(T > 2) = e^{-2 \times 2} = e^{-4} \approx 0.018.$$

About a $1.8\%$ chance the next gap is longer than $2$ seconds. The intuition: **the Exponential is the Poisson process's waiting time, its mean is $1/\lambda$, and its memorylessness means a past wait never shortens the future one.**

## Normal: the Central Limit Theorem and the 68-95-99.7 rule

The **Normal distribution** (also called the **Gaussian**) is the famous bell curve, and it earns its fame because of one of the deepest facts in all of probability: it is what sums of many independent things look like. A Normal is fully described by two numbers — its mean $\mu$ (the center) and its standard deviation $\sigma$ (the spread) — and written $N(\mu, \sigma^2)$. Its density is the symmetric bell peaked at $\mu$.

### The 68-95-99.7 rule

You should know the Normal's geometry cold, because interviewers expect instant tail estimates without a z-table.

![The normal bell and the 68-95-99.7 rule: about 68 percent of the mass sits within one standard deviation, 95 percent within two, and 99.7 percent within three](/imgs/blogs/distributions-cheat-sheet-quant-interviews-6.png)

The rule, illustrated above: about **68%** of a Normal's probability lies within $\pm 1\sigma$ of the mean (the green band), about **95%** within $\pm 2\sigma$ (adding the amber bands), and about **99.7%** within $\pm 3\sigma$ (adding the red bands). The complements are the numbers that matter for risk: roughly $16\%$ of the time you're beyond $+1\sigma$ *or* below $-1\sigma$ on each side, only $2.5\%$ beyond $+2\sigma$, and a mere $0.15\%$ beyond $+3\sigma$. A move of $2$ standard deviations is a "once-a-month" event; $3$ sigma is "a couple of times a year" — *if* the world were actually Normal. (It isn't in the tails, as we'll see, and that gap is itself an interview topic.)

### The Central Limit Theorem

Here is why the Normal is everywhere. The **Central Limit Theorem (CLT)** says: if you add up many independent random variables — *whatever* their individual distributions, as long as each contributes a small piece — the sum is approximately Normal. The individual things can be wildly non-Normal. Their sum smooths into a bell anyway.

![The Central Limit Theorem: add independent uniforms and the shape marches from flat to triangle to a bell in three steps](/imgs/blogs/distributions-cheat-sheet-quant-interviews-8.png)

The figure proves it with the least bell-shaped thing imaginable. A single $\text{Uniform}(0,1)$ is dead flat (left panel) — no hint of a bell. Add *two* uniforms and the sum is a triangle (middle) — the bell starts to emerge. Add *twelve* and the sum is visually indistinguishable from a Normal (right). This is not a coincidence about uniforms; it's the universal pull toward the bell. The CLT is *the* reason the Normal models aggregate quantities: total P&L over many trades, the average of many measurements, the sum of many small independent shocks. If a quantity is a big pile of small independent pieces, reach for the Normal.

The CLT also explains why standard deviation scales like $\sqrt{n}$. If you sum $n$ independent copies of a variable with standard deviation $\sigma$, the variances add (to $n\sigma^2$), so the *standard deviation* of the sum is $\sqrt{n}\,\sigma$. The mean grows like $n$ but the spread grows only like $\sqrt{n}$ — so the sum's *relative* noise shrinks. That single fact is why running more independent trades makes your daily P&L *more* predictable in percentage terms, and it is the mathematical heart of diversification and of why a market maker wants flow, not bets. For the betting-size flip side of this — how much to wager when you do have edge — see [the Kelly criterion and sequential betting](/blog/trading/quantitative-finance/kelly-criterion-sequential-betting-quant-interviews).

#### Worked example: normal approximation of total $ P&L over many independent trades

You make **1,000 independent trades** in a day. Each trade wins $\$2$ with probability $0.6$ and loses $\$1.80$ with probability $0.4$. Approximate the distribution of your *total daily P&L* and estimate the chance you have a losing day.

First, one trade. Its expected payoff is $0.6 \times \$2 + 0.4 \times (-\$1.80) = \$1.20 - \$0.72 = \$0.48$. Its variance: the payoffs are $+\$2$ and $-\$1.80$, deviations from the $\$0.48$ mean are $+\$1.52$ and $-\$2.28$, so $\operatorname{Var} = 0.6 \times (1.52)^2 + 0.4 \times (2.28)^2 \approx 0.6 \times 2.31 + 0.4 \times 5.20 \approx 1.39 + 2.08 = 3.47$ dollars-squared, giving a per-trade standard deviation of about $\$1.86$.

Now sum $1{,}000$ independent copies. The mean total P&L is $1{,}000 \times \$0.48 = \$480$. The variance adds, so the total variance is $1{,}000 \times 3.47 = 3{,}470$, and the total standard deviation is $\sqrt{3{,}470} \approx \$59$. By the CLT, total P&L is approximately $N(\$480,\ \$59^2)$.

![On the desk: a thousand independent micro-edges add up, by the CLT, to a tidy normal daily P&L centered at plus 500 dollars with almost no chance of a losing day](/imgs/blogs/distributions-cheat-sheet-quant-interviews-12.png)

A losing day means total P&L below $\$0$. That's $\frac{0 - 480}{59} \approx 8.1$ standard deviations below the mean. Under the Normal, the chance of being $8$ sigma below the mean is astronomically small — effectively zero. (The figure rounds the example to a clean $+\$500$ mean with $\$60$ standard deviation and shows the same conclusion: the entire bell sits in the green gain region, with the $\$0$ breakeven line far off to the left.) This is the deepest lesson on the desk: **a tiny per-trade edge, repeated thousands of times independently, becomes a near-certain profit, because the CLT shrinks the relative noise as $\sqrt{n}$.** A coin-flip edge you'd never bet your lunch on becomes a printing press at scale — *provided* the trades are genuinely independent, which is the assumption that fails catastrophically in a crisis.

## Lognormal: why modeled stock prices are lognormal

A subtle and important question: if returns are Normal, what about *prices*? The answer is that modeled stock prices are **lognormal** — a random variable is lognormal if its *logarithm* is Normal. Here's the chain of reasoning, which interviewers love to walk you through.

A stock's price doesn't move by *adding* dollars; it moves by *multiplying* by growth factors. A $1\%$ up day multiplies the price by $1.01$. Over many days, the final price is the *product* of all those daily factors. Now take the logarithm: the log of a product is the *sum* of the logs. So the log-price is a *sum* of many small independent log-returns — and by the CLT, a sum of many independent things is Normal. Therefore the log-price is Normal, which means the price itself is **lognormal**. Prices multiply; logs of prices add; sums go Normal; so prices go lognormal.

![Why modeled stock prices are lognormal: the price is floored at zero but has unbounded upside, so the distribution leans right with a long tail of large gains](/imgs/blogs/distributions-cheat-sheet-quant-interviews-7.png)

The lognormal's shape, above, captures two facts that a Normal cannot. First, a price **cannot go below $\$0$** — the red region shows losses bounded at the floor of zero, because the worst a multiplicative process can do is multiply by something close to (but above) zero. Second, the **upside is unbounded** — a stock can double, triple, ten-bag — so the right tail (green) stretches out long. The result is a right-*skewed* distribution: most of the mass sits below the mean, with a long tail of large gains pulling the average up. Notice the peak (the most likely price) sits *below* today's $\$100$, even when the stock has positive expected return — a quirk of skew that trips up beginners.

#### Worked example: a stock's lognormal price after one year

A stock trades at $S_0 = \$100$ today. Under the standard model, its log-return over one year is Normal with a **drift** (expected log-return) of $\mu = 8\%$ and a **volatility** (standard deviation of log-return) of $\sigma = 20\%$. Describe the distribution of the price $S_1$ in one year, and find a rough range that should contain it about $95\%$ of the time.

The log-return $\ln(S_1 / S_0) \sim N(0.08,\ 0.20^2)$. So the one-year price is

$$S_1 = S_0 \, e^{R}, \quad R \sim N(0.08,\ 0.20^2).$$

For the $95\%$ range, recall that a Normal lands within $\pm 2\sigma$ of its mean about $95\%$ of the time. Here $R$ lands roughly between $0.08 - 2(0.20) = -0.32$ and $0.08 + 2(0.20) = +0.48$. Exponentiating those log-return bounds:

- Lower: $\$100 \times e^{-0.32} \approx \$100 \times 0.726 = \$72.6$
- Upper: $\$100 \times e^{+0.48} \approx \$100 \times 1.616 = \$161.6$

So about $95\%$ of the time the stock should land between roughly $\$73$ and $\$162$ in one year. Notice the asymmetry: the price can rise $\$62$ above today but only fall about $\$27$ below — the upside band is wider than the downside band, which is the lognormal skew showing up in dollars. The intuition: **prices are lognormal because returns compound multiplicatively, so you reason about the Normal log-return first and exponentiate at the end, and the result is bounded below at $\$0$ with a long upside tail.** This lognormal-price assumption is exactly what sits underneath the [Black-Scholes option pricing model](/blog/trading/quantitative-finance/black-scholes).

## A decision section: which distribution models what

Now the payoff. The whole article compresses into one reflex: hear the situation, ask two questions, land on the distribution. The first question is *counting or timing/measuring?* (discrete vs continuous). The second drills into the mechanism.

![A decision map: counting versus timing and discrete versus continuous splits every common interview situation into one of the eight workhorse distributions](/imgs/blogs/distributions-cheat-sheet-quant-interviews-9.png)

Here is the same map in a table you can drill until it's automatic:

| If the situation is...                                  | The distribution is... | The one fact to quote                          |
| ------------------------------------------------------- | ---------------------- | ---------------------------------------------- |
| One yes/no trial                                        | Bernoulli($p$)         | mean $p$, variance $p(1-p)$                     |
| Successes in $n$ fixed independent trials               | Binomial($n,p$)        | mean $np$, variance $np(1-p)$                   |
| Trials until the *first* success                        | Geometric($p$)         | mean $1/p$, memoryless                          |
| Trials until the *$r$-th* success                       | Negative binomial      | mean $r/p$ (it's $r$ stacked Geometrics)        |
| Count of rare events in a fixed window                  | Poisson($\lambda$)     | mean $=$ variance $= \lambda$                   |
| Every value on a range equally likely                   | Uniform($a,b$)         | mean $\frac{a+b}{2}$, variance $\frac{(b-a)^2}{12}$ |
| Continuous time *between* Poisson events                | Exponential($\lambda$) | mean $1/\lambda$, memoryless                    |
| Sum/average of many independent pieces                  | Normal (via CLT)       | 68-95-99.7; spread grows as $\sqrt{n}$          |
| A *product* of returns / a modeled price                | Lognormal              | log is Normal; bounded at $\$0$, right-skewed   |

Two limit theorems glue the map together and are themselves frequent interview targets. The **Poisson limit** connects the discrete-counting world to itself: a Binomial with many rare trials *is* a Poisson. The **Central Limit Theorem** connects everything to the Normal: any sum of many independent pieces *is* Normal. Memorize the map, and the next time an interviewer says "orders arrive at a rate of..." you've already started down the Poisson branch before they finish the sentence.

## In the interview room: fully solved problems

Now we put it all together the way an interview actually feels: a prompt, a moment to recognize the distribution, then a clean computation narrated out loud. Each of these is the kind of question that decides rounds.

#### Worked example: the first heads on a fair coin

"I flip a fair coin repeatedly. What's the expected number of flips to get my first heads? And the probability it takes more than $3$ flips?"

Recognition: "first success" → Geometric, with $p = 0.5$. The expected number of flips is $1/p = 1/0.5 = 2$ flips. For more than $3$ flips, all three must be tails: $P(N > 3) = (1-p)^3 = (0.5)^3 = 0.125$, a $12.5\%$ chance. Narrate the recognition step explicitly — "this is geometric because we're waiting for the first success" — because the interviewer is grading your *labeling*, not just your arithmetic.

#### Worked example: at least one defect

"A manufacturing process produces a defect on each unit independently with probability $p = 0.01$. In a batch of $200$ units, what's the probability of *at least one* defect?"

Recognition: counting rare events ($p$ small) over many trials ($n = 200$) → reach for the Poisson approximation with $\lambda = np = 200 \times 0.01 = 2$. "At least one" begs the complement: $P(X \ge 1) = 1 - P(X = 0) = 1 - e^{-\lambda} = 1 - e^{-2} \approx 1 - 0.135 = 0.865$. About $86.5\%$. You could grind the exact Binomial $1 - (0.99)^{200}$ and get $0.866$ — essentially identical — but the Poisson route is faster and shows you saw the rare-event structure. If each defect costs the firm $\$50$ to remediate, the expected remediation cost per batch is $\lambda \times \$50 = 2 \times \$50 = \$100$, because the expected *number* of defects is exactly $\lambda = 2$.

#### Worked example: how many orders before two fills

"Your resting order fills on each look with probability $p = 0.2$, looks independent. What's the expected number of looks until your *second* fill?"

Recognition: "trials until the $r$-th success" → Negative binomial with $r = 2$, $p = 0.2$. It's two stacked Geometrics, so the expected total is $r/p = 2/0.2 = 10$ looks. The clean way to say it: "the first fill takes $1/p = 5$ looks on average, the second fill takes another $5$ on average, so $10$ total — independent waiting times just add." That additivity narration is what the interviewer wants to hear; it shows you understand *why* rather than reciting a formula.

#### Worked example: the memoryless elevator

"An elevator arrives as a Poisson process, on average one every $4$ minutes. You arrive and find no elevator. You wait $4$ minutes — still nothing. What's your expected remaining wait?"

Recognition: time between Poisson events → Exponential, mean $1/\lambda = 4$ minutes, and the magic word is **memoryless**. Your expected remaining wait is still $4$ minutes, exactly as it was when you arrived. The four minutes you've already burned carry zero information about when the elevator comes. If you instinctively want to say "it's overdue, so less than $4$," that instinct is precisely the bias the question is built to expose — and saying "I know it *feels* overdue, but the exponential is memoryless, so it's still $4$ minutes" is a strong answer because it shows you trust the model over your gut.

#### Worked example: a $ P&L tail under the normal

"Your daily P&L is approximately Normal with mean $\$10{,}000$ and standard deviation $\$4{,}000$. What's the probability you lose money on a given day, and roughly how often do you have a day worse than $-\$2{,}000$?"

Recognition: aggregate P&L → Normal, use 68-95-99.7. Losing money means P&L below $\$0$, which is $\frac{0 - 10{,}000}{4{,}000} = -2.5$ standard deviations from the mean. Beyond $2.5\sigma$ on one side is roughly $0.6\%$ — so you lose money on about $1$ day in $160$. For a day worse than $-\$2{,}000$: that's $\frac{-2{,}000 - 10{,}000}{4{,}000} = -3$ standard deviations. The tail beyond $-3\sigma$ is about $0.15\%$ — roughly $1$ trading day in $670$, or about once every $2.7$ years. Quoting these from the 68-95-99.7 rule without a z-table is exactly the fluency interviewers are checking. (And the honest footnote: real P&L has fatter tails than the Normal, so that "once every $2.7$ years" understates how often the bad day truly comes — which is the perfect segue to the next section.)

#### Worked example: combining two Poisson streams

"Orders from venue A arrive at $\lambda_A = 4$ per second and from venue B at $\lambda_B = 6$ per second, independently. What's the distribution of the *combined* order flow, and the chance you see no orders at all in a given $0.1$-second window?"

Recognition: independent Poisson streams *superpose* into a single Poisson whose rate is the sum. Combined rate $\lambda = 4 + 6 = 10$ per second. Over a $0.1$-second window the parameter is $\lambda \times 0.1 = 1$, so the count is $\text{Poisson}(1)$, and the chance of zero is $P(X = 0) = e^{-1} \approx 0.368$. About a $37\%$ chance of a quiet $0.1$-second window. The superposition fact — "Poissons add" — is a clean one-liner that signals you know the structure, not just the formula.

## Common misconceptions

**"The probability of an exact value under a continuous distribution is small but nonzero."** No — it is *exactly* zero. The chance the next order arrives at precisely $3.0000\dots$ seconds is $0$, because a continuous variable has infinitely many possible values. Probability only accumulates over an *interval* (the area under the density). Confusing density (which can exceed $1$) with probability (which never can) is the single most common continuous-distribution error, and interviewers fish for it deliberately.

**"After a long run of failures, success is 'due.'"** This is the gambler's fallacy, and the Geometric and Exponential are memoryless precisely to formalize why it's wrong. If trials are independent, the past run is irrelevant; your remaining wait has the same distribution it always did. A coin that's landed tails ten times in a row is still a $50/50$ coin on the next flip. The *only* way the past matters is if the trials are *not* independent — and recognizing when independence fails is a deeper skill than any single distribution.

**"The mean is the most likely value."** Only for symmetric distributions like the Normal. For a right-skewed distribution like the Lognormal, the most likely value (the *mode*) sits *below* the mean, because a few large values in the long right tail drag the average up past the bulk of the mass. This is why a stock with positive expected return can still have a *most-likely* one-year price below today's — the mean and the mode are different animals, and interviewers test whether you know which one a question is asking for.

**"Binomial and Poisson are different models you pick between."** They're the same phenomenon at different scales. The Poisson *is* the Binomial in the limit of many rare trials. If you find yourself with a Binomial where $n$ is large and $p$ is small, switching to the Poisson with $\lambda = np$ isn't an approximation you're settling for — it's the natural description, and it replaces an ugly factorial with a clean exponential.

**"Real market returns are Normally distributed."** This is the assumption that has blown up more trading firms than any other, and a sharp interviewer wants to hear you flag it. Returns are *approximately* Normal in the body but have dramatically **fatter tails**: extreme moves happen far more often than the bell predicts. Treating tail risk as Gaussian is how you end up with a position that "can't lose more than $X$" right up until the day it loses $5X$.

**"Variance and standard deviation both just add up over independent trades."** Only the *variance* adds. If you run $n$ independent trades each with standard deviation $\sigma$, the total variance is $n\sigma^2$, so the total *standard deviation* is $\sqrt{n}\,\sigma$ — not $n\sigma$. Forgetting the square root is one of the most common quantitative slips in an interview, and it inverts the whole point of diversification: it is *because* spread grows only as $\sqrt{n}$ while the mean grows as $n$ that piling up independent trades makes a small edge into a near-certain profit. Quote the $\sqrt{n}$ and you signal you understand risk; quote a linear $n$ and you've just claimed that doubling your trade count doubles your risk, which is exactly backwards.

**"A higher-mean bet is always the better bet."** Expected value is only half the story. A bet with a higher mean but enormous variance can be far worse than a steady low-variance one, because real traders face *ruin* — a string of losses that wipes out the bankroll before the long-run average ever shows up. This is why expectation alone never sizes a position; the spread, the worst case, and the path all matter. Interviewers slip a high-variance "+EV" trap into market-making games precisely to see whether you reach for variance and ruin, not just the mean.

## How it shows up on a real trading desk

These distributions are not interview trivia. They are the working vocabulary of the desk. Here is where each one actually lives.

**Order arrivals as a Poisson process.** Every market-microstructure model starts by treating order flow as a Poisson process: orders, cancellations, and trades arrive at some rate $\lambda$, and the counts in any window are Poisson while the gaps are Exponential. A market maker's fill probability, the expected time to get filled at a given queue position, the likelihood of an adverse burst of flow — all are computed from Poisson and Exponential arithmetic. When a desk says "the venue is quoting $50$ updates a second," they are stating a $\lambda$, and everything downstream — staffing the matching engine, sizing buffers, estimating adverse selection — flows from it. The duality you saw in the timeline figure (counts are Poisson, gaps are Exponential) is used dozens of times a day.

**Daily P&L as a Normal, via the CLT.** A desk that does thousands of small independent trades has a daily P&L that is, to an excellent approximation, Normal — that's the worked example made real. Risk limits, "value at risk" estimates, and capital allocation all lean on that Normal approximation. The $\sqrt{n}$ scaling is why high-frequency desks chase volume: more independent trades shrink the *relative* noise, turning a microscopic per-trade edge into a steady, low-variance income stream. The same math is why a firm prefers ten uncorrelated strategies to one big bet of the same expected return — the diversification benefit *is* the variance-adds-but-std-grows-as-$\sqrt{n}$ fact.

**Prices as lognormal, and options priced on top.** The entire edifice of options pricing — Black-Scholes and everything after it — assumes the underlying price is lognormal, exactly because returns compound multiplicatively and the CLT pushes log-prices to Normal. When a trader quotes an option, the lognormal assumption is baked into the model spitting out the price. The known *failures* of that assumption — that real prices have fatter tails and that volatility isn't constant — are precisely what create the [volatility surface](/blog/trading/quantitative-finance/volatility-surface), the systematic way the market prices in the lognormal model's shortcomings. The whole multi-billion-dollar volatility-trading business exists in the gap between the lognormal model and reality.

**Fat tails and the limit of the Normal.** This is the most important thing the Normal *doesn't* model, and it's worth its own picture.

![Normal versus fat tails: market returns put far more mass in the extremes than the normal bell allows, so the red fat tails rise above the blue normal curve in the extremes](/imgs/blogs/distributions-cheat-sheet-quant-interviews-11.png)

The figure overlays a Normal (blue dashed) against a fat-tailed distribution (red solid) with the same variance. In the body they look similar, but in the extremes — beyond about $\pm 2\sigma$ — the red sits *above* the blue: there is genuinely more probability out in the tails than the bell allows. The numbers are staggering. Under the Normal, a $6$-standard-deviation daily move is about a $1$-in-$500$-million event — you'd expect to wait millions of years to see one. Yet equity markets deliver moves of that nominal size every handful of years: October 1987 (the S&P 500 fell about $20\%$ in a day, a more-than-$20\sigma$ event under any sane Normal estimate), the 2008 crisis, the 2010 flash crash, the 2020 COVID drawdown. The Normal is a superb model for the *middle* of the distribution and a dangerous one for the *edges*. Desks that priced tail risk as Gaussian — Long-Term Capital Management is the textbook case — discovered that the once-in-the-history-of-the-universe event arrives on a Tuesday. The interview-ready summary: *use the Normal for the body, respect the fat tails for the risk, and never bet the firm on a Gaussian tail estimate.*

**The waiting-time questions traders ask constantly.** "How long until my order at the back of the queue fills?" is a Geometric/Exponential question. "How many of my $20$ resting orders will fill in the next minute?" is a Binomial (or Poisson) question. "If I need three fills to complete this trade, how long should I budget?" is a Negative binomial / sum-of-Exponentials question. Execution traders run this arithmetic in their heads all day, and it's the same arithmetic the interview is rehearsing you for.

## When this matters and where to go next

If you are preparing for quant interviews, the highest-leverage thing you can do with this article is to stop memorizing formulas and start drilling *recognition*. Take any probability prompt — from a problem set, a mock interview, a forum — and before computing anything, say out loud which distribution it is and why. "Waiting for the first success, so Geometric." "Rare events over a window, so Poisson." "Sum of many independent things, so Normal." Once the label is automatic, the computation is the easy part, and the narration is what gets you hired.

The mechanism stories are what make recognition fast, so internalize them in this compressed form: Bernoulli is one trial; Binomial is $n$ trials counted; Geometric is the wait for the first success ($1/p$); Negative binomial is the wait for the $r$-th ($r/p$); Poisson is rare events counted over a window ($\lambda$); Uniform is "no information beyond the range"; Exponential is the memoryless wait between Poisson events ($1/\lambda$); Normal is what sums become; Lognormal is what products become. Two theorems tie it off: many rare chances make a Poisson, and many independent pieces make a Normal.

From here, the natural next steps deepen specific branches of this map. The waiting-time and order-statistics genre opens up in [order statistics and uniform tricks](/blog/trading/quantitative-finance/order-statistics-uniform-tricks-quant-interviews); the information-updating side — how a new data point shifts a probability — is the subject of [conditional probability and Bayes for quant interviews](/blog/trading/quantitative-finance/conditional-probability-bayes-quant-interviews); the betting-and-sizing question of what to *do* once you have an edge is [the Kelly criterion](/blog/trading/quantitative-finance/kelly-criterion-sequential-betting-quant-interviews); and the lognormal-price assumption you met here is the foundation that [Black-Scholes](/blog/trading/quantitative-finance/black-scholes) builds an entire pricing theory on. Each one reuses the eight workhorses and two theorems from this article — which is exactly the point. The map is small. The fluency is everything.
