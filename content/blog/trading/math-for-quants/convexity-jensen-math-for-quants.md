---
title: "Convexity and Jensen's inequality: why curves bend the odds"
date: "2026-06-15"
description: "How the shape of a curve decides whether averaging helps or hurts you, why volatile strategies bleed compounded returns, why option payoffs love volatility, and where the unique answer to an optimization problem comes from."
tags: ["convexity", "jensens-inequality", "volatility-drag", "gamma", "kelly-criterion", "log-utility", "convex-optimization", "quantitative-finance"]
category: "trading"
subcategory: "Quantitative Finance"
author: "Hiep Tran"
featured: true
readTime: 44
---

> [!important]
> **TL;DR** — The shape of a curve quietly decides whether averaging your luck helps you or hurts you, and that single fact runs underneath option pricing, portfolio compounding, risk aversion, and every solver a quant desk runs.
>
> - A **convex** function cups upward (its second derivative is positive); a **concave** function caps downward (second derivative negative). The chord joining two points sits *above* a convex curve and *below* a concave one.
> - **Jensen's inequality** turns that shape into a hard fact about averages: for a convex $f$, the average output beats the output of the average, $E[f(X)] \ge f(E[X])$. For concave $f$ the inequality flips.
> - Convex payoffs *love* volatility (positive gamma is long volatility), while concave growth *hates* it: a strategy with arithmetic return $\mu$ and volatility $\sigma$ compounds at only $g \approx \mu - \sigma^2/2$ — the **volatility drag**.
> - The single number to remember: **+50% then −50% on \$100 ends at \$75**, a −13.4% compounded loss even though the arithmetic average move was exactly zero.

In the casino of finance there is a magic trick that almost nobody can see, and it has nothing to do with luck. Take \$100. Have it go up 50% in year one, then down 50% in year two. The two moves are mirror images; their plain average is zero. So you should end where you started, with \$100, right? You end with \$75. A quarter of your money has vanished, and not one trade went wrong, no fee was charged, no scam occurred. The money was eaten by the *shape of compounding* — by a property mathematicians call convexity and the inequality that comes with it.

That same shape, read in the other direction, is why a stock option can be worth real money even when everyone agrees the stock will end up exactly where it is now, and why the very best portfolio-construction problems have one correct answer that a computer is guaranteed to find rather than a swamp of false bottoms it can get stuck in. Convexity is one of those rare ideas that, once you see it, you cannot un-see: it explains the option desk, the compounding desk, and the risk desk all at once. This post builds it from absolutely nothing — no calculus assumed — and walks it all the way to the math a practitioner uses on a Tuesday.

![Before and after comparison of a convex curve cupping upward and a concave curve capping downward](/imgs/blogs/convexity-jensen-math-for-quants-1.png)

The figure above is the whole post in one picture. On the left is a **convex** curve — it cups upward like the inside of a bowl. On the right is a **concave** curve — it caps downward like the outside of a dome. Everything we are about to learn is a consequence of which way the curve bends. When a curve cups up, mixing two outcomes lands you *above* the curve, and that "above" is free money the convexity hands you. When a curve caps down, mixing lands you *below*, and that shortfall is the tax that volatility quietly charges. By the end of this article that single distinction will read like a sentence, and you will spot it everywhere a price, a return, or a payoff is involved.

## Foundations: the building blocks

Before we can say anything precise, we need to agree on what a few objects are. If you have never thought hard about a curve, an average, or the word "expected", this section is for you. A practitioner can skim it; a beginner should not skip it, because the entire argument rests on these definitions being crisp.

### What a function is, simply put

A **function** is a machine: you feed it a number, it gives you back a number. Write it $f(x)$ — read "f of x" — meaning "the output the machine $f$ produces when you feed in $x$." A few functions you already know in disguise: doubling your money is $f(x) = 2x$; the area of a square plot of land of side $x$ is $f(x) = x^2$; the value of \$1 growing for one period at rate $x$ is $f(x) = 1 + x$. The thing we care about is not the formula but the *shape* of the picture you get when you plot inputs across the bottom and outputs up the side.

### What "convex" and "concave" actually mean

Here is the cleanest definition, and it needs no calculus. Pick any two points on the curve. Draw the straight line connecting them — that straight line is called a **chord** (the same word as a chord across a circle). Now ask one question: does the curve dip *below* the chord between those two points, or does it bulge *above* it?

- If the curve stays **below** the chord — if the straight shortcut always runs above the actual curve — the function is **convex**. It cups upward, like the path of a hanging chain or the inside of a salad bowl.
- If the curve bulges **above** the chord — if the straight shortcut runs below the actual curve — the function is **concave**. It caps downward, like the arch of a bridge or the outside of a hill.

A handy mnemonic: con**cave** has a "cave" you could shelter under (the curve arches over you); con**vex** is its opposite, a bowl you would fall into. The everyday function $f(x) = x^2$ is convex — plot it and it is a U, a bowl. The function $f(x) = \sqrt{x}$ is concave — it shoots up fast then flattens, an arch. The straight line $f(x) = 2x$ is the boundary case: its chord *is* the curve, so it is technically both convex and concave at once, which will matter later because it is exactly where the inequality becomes an equality.

### The second-derivative test, without the jargon

There is a faster way to check convexity than drawing chords, and it is the one quants use. The **first derivative** of a function is its *slope* — how steeply the output is rising or falling at a given point. The **second derivative** is the *slope of the slope* — how fast the steepness itself is changing. Written $f''(x)$ (read "f double-prime of x").

The test is one line:

$$ f''(x) > 0 \text{ everywhere} \implies f \text{ is convex}; \qquad f''(x) < 0 \text{ everywhere} \implies f \text{ is concave}. $$

Why does this work? A positive second derivative means the slope is *increasing* — the curve is steepening as you move right, which is exactly what bending upward into a bowl looks like. A negative second derivative means the slope is *decreasing* — the curve is leveling off and rolling over, which is what an arch looks like. For $f(x) = x^2$, the first derivative is $2x$ and the second is $2$, a positive constant, so $x^2$ is convex everywhere — matching the picture. For $f(x) = \ln(x)$ (the natural logarithm, the function that turns multiplication into addition and underpins all compounding math), the second derivative is $-1/x^2$, always negative, so the logarithm is concave everywhere. Hold on to that fact about the logarithm; it is the seed of half this article.

#### Worked example: checking the curvature of a payoff

Suppose you own a *call option* — a contract that pays you nothing if a stock finishes below a strike price $K = \$100$, and pays you (stock price − \$100) for every dollar the stock finishes above \$100. We will define options properly later, but for now just treat the payoff as a function of the final stock price $S$. Below \$100 the payoff is flat at \$0; above \$100 it rises one-for-one. Plot it and you get a hockey stick: flat, then a sharp kink up at \$100, then a straight 45-degree ramp.

Is this convex? Use the chord test. Take two stock prices, say \$80 (payoff \$0) and \$120 (payoff \$20). The chord between them is the straight line from (\$80, \$0) to (\$120, \$20). At the midpoint stock price \$100, the chord sits at \$10. The actual payoff at \$100 is \$0. Since \$0 is below \$10, the curve is below the chord — **convex**. The kink does not break convexity; a convex function is allowed to have corners as long as it never bulges above any chord. That convex kink is the entire reason an option is worth owning, and we will cash that observation out in dollars shortly.

The one-sentence intuition: convexity is just "the curve stays under every straight shortcut," and you can verify it with two points and a ruler.

### Convex combinations: the algebra behind the chord

There is a more precise way to state the chord test, and it is the version you will meet in textbooks, so it is worth seeing once. A **convex combination** of two points $x_1$ and $x_2$ is any weighted average $\lambda x_1 + (1-\lambda) x_2$ where the weight $\lambda$ runs from 0 to 1. As $\lambda$ slides from 0 to 1, that expression sweeps out every point on the straight line between $x_1$ and $x_2$ — at $\lambda = 1$ you are at $x_1$, at $\lambda = 0$ you are at $x_2$, and at $\lambda = 0.5$ you are exactly at the midpoint. The weights are non-negative and sum to one, which is precisely the shape of a portfolio (fractions of your money that add to 100%) or a probability distribution (likelihoods that add to one). That is not a coincidence; it is why convexity shows up wherever weights or probabilities do.

With that vocabulary, the formal definition of a convex function is a single line:

$$ f\big(\lambda x_1 + (1-\lambda) x_2\big) \le \lambda f(x_1) + (1-\lambda) f(x_2) \quad \text{for all } \lambda \in [0,1]. $$

The left side is "the function evaluated at the average input." The right side is "the average of the function's outputs" — which is exactly the height of the chord. The $\le$ says the curve sits at or below the chord. That is the chord test in symbols, and if you squint you will notice it is *already* Jensen's inequality for a two-point distribution: $\lambda$ and $1-\lambda$ are the probabilities, $\lambda x_1 + (1-\lambda)x_2$ is $E[X]$, and the right-hand side is $E[f(X)]$. Jensen's inequality is just this definition extended from two points to any number of points, or to a continuous spread. Everything later in the post is this one line, dressed for a different occasion.

A useful corollary: a function is convex if and only if the region *above* its graph (the set of all points sitting on or over the curve, called the **epigraph**) is a convex set. This is the bridge between curved functions and the convex sets we need for optimization in Section 4 — the two notions of convexity, the one for functions and the one for sets, are the same idea seen from two angles.

### What "expected value" means

The last building block is the word **expected**. In probability, the **expected value** of a random quantity $X$, written $E[X]$, is the long-run average you would get if you could repeat the situation thousands of times. If a fair die pays you its face value in dollars, the expected payoff is $(1+2+3+4+5+6)/6 = \$3.50$ — not because any single roll pays \$3.50 (none can), but because that is the average over many rolls. The expected value is a *weighted average of the outcomes, weighted by how likely each is*. We unpack this machinery in full in the companion post on [expectation, variance, and moments](/blog/trading/math-for-quants/expectation-variance-moments-math-for-quants); here we only need the one-line idea that $E[X]$ is the probability-weighted average.

Now we have the two ingredients — a curved machine $f$, and an average — and the whole drama of this post is what happens when you combine them in the two possible orders.

## 1. Jensen's inequality: the order of operations that costs money

Here is the question that Jensen's inequality answers, and it is a question about *order of operations*. You have a random input $X$ and a curved function $f$. There are two things you could compute, and they sound almost identical:

1. **Average first, then bend.** Take the average of the inputs, then push that single average through the function: $f(E[X])$.
2. **Bend first, then average.** Push each possible input through the function, then average the outputs: $E[f(X)]$.

If $f$ were a straight line, these two would be identical — averaging and a straight-line transformation commute, you can do them in either order. But the moment $f$ is curved, the order matters, and *which way it matters is fixed by the direction of the curvature.* That is Jensen's inequality:

$$ \boxed{\,E[f(X)] \ge f(E[X]) \quad \text{when } f \text{ is convex}\,} $$

and the inequality flips, $E[f(X)] \le f(E[X])$, when $f$ is concave. Equality holds only when $f$ is a straight line, or when $X$ has no randomness at all (it is a constant).

![Pipeline from period returns through arithmetic mean and a half-variance subtraction to the geometric mean and compounded drag](/imgs/blogs/convexity-jensen-math-for-quants-2.png)

### Why it is true — the picture, not the proof

You do not need the formal proof to trust this; you need the chord. Picture the simplest random input: a coin flip that lands on a low value $x_{\text{low}}$ or a high value $x_{\text{high}}$ with equal probability. The expected input $E[X]$ is the midpoint between them. Now:

- $f(E[X])$ is the height of the *curve* at that midpoint.
- $E[f(X)]$ is the average of the two output heights $f(x_{\text{low}})$ and $f(x_{\text{high}})$ — which is exactly the height of the *chord* at the midpoint, because the midpoint of a chord is the average of its two endpoints.

So Jensen's inequality is literally the chord test restated in the language of averages. For a convex function the chord sits above the curve, so the average output ($E[f(X)]$, the chord) is above the curve at the mean ($f(E[X])$). That is the entire content of $E[f(X)] \ge f(E[X])$. The proof for many outcomes, or a continuous distribution, is the same idea repeated: a convex curve lies above all its tangent lines, and averaging tangent-line values can only push you up. The diagram above traces this for compounding specifically: raw returns get averaged the naive way (arithmetic), then convexity forces a subtraction of half the variance to land at the true compounded rate.

### The Jensen gap is not a rounding error — it is the variance

The most useful version of Jensen's inequality is not the bare inequality but the *size* of the gap between the two sides. For a smooth function $f$ and a random variable $X$ with mean $\mu$ and variance $\sigma^2$ (variance being the average squared distance from the mean — the formal measure of "how spread out" the randomness is), a second-order approximation gives:

$$ E[f(X)] \approx f(\mu) + \tfrac{1}{2} f''(\mu)\,\sigma^2. $$

Read that slowly, because it is the workhorse of this entire post. The gap between "average then bend" and "bend then average" is approximately *half the curvature times the variance.* Three consequences fall straight out:

- If $f$ is convex ($f'' > 0$), the gap is positive — averaging the outputs beats the output of the average. The convex player *gains* from variance.
- If $f$ is concave ($f'' < 0$), the gap is negative — the concave player *loses* from variance.
- The gap scales with $\sigma^2$. No variance, no gap. Double the volatility and the squared term quadruples the gap. This is why Jensen effects are invisible in calm markets and brutal in violent ones.

![Stack showing expected payoff above the Jensen gap above the payoff at the mean for a convex function](/imgs/blogs/convexity-jensen-math-for-quants-3.png)

The stack above is the bookkeeping of a convex payoff. At the bottom sits $f(E[X])$, the payoff if you simply plugged in the average outcome. The Jensen gap sits on top of it, and the gap is fed by variance. The sum is $E[f(X)]$, the payoff you actually expect once you account for the spread of outcomes. For a convex payoff this stack is genuine extra value; for a concave one the same stack runs in reverse and the gap is a cost you pay.

#### Worked example: Jensen on a two-scenario option spread

Let us turn the inequality into dollars. You hold the call option from before: it pays \$0 if the stock ends at or below \$100, and (stock − \$100) above it. Suppose the stock is equally likely to end at \$80 or \$120. The *expected stock price* is the average of the two, $(\$80 + \$120)/2 = \$100$.

Now compute both orders:

- **Average first, then bend:** plug the expected price \$100 into the payoff. At exactly \$100 the option pays \$0. So $f(E[S]) = \$0$.
- **Bend first, then average:** the payoff at \$80 is \$0; the payoff at \$120 is \$20. Average them: $(\$0 + \$20)/2 = \$10$. So $E[f(S)] = \$10$.

The expected payoff is **\$10**, but the payoff at the expected price is **\$0**. The \$10 gap is pure Jensen — pure convexity. A trader who priced this option by saying "the stock will probably land around \$100 where the option is worthless, so it's worth nothing" would have mispriced it by \$10 per share. On a position of 10,000 shares that is a \$100,000 error from one wrong order of operations. The convex kink in the payoff converts the *spread* of possible prices into expected value, and the more spread (the more volatility) you expect, the more that option is worth.

The one-sentence intuition: for a convex payoff, uncertainty itself is the asset — the wider the outcomes, the higher the expected payoff, even when the expected price does not move at all.

## 2. Volatility drag: the tax you pay for bouncing around

Now flip the function. Compounding lives in the world of *logarithms*, and the logarithm is concave. That single fact is why volatile strategies bleed money over time, and it has a name traders mutter with respect and dread: **volatility drag**, also called *variance drain*.

### Why compounding lives in log-land

When returns *compound*, they multiply rather than add. If you earn return $r_1$ in period one and $r_2$ in period two, \$1 grows to $(1 + r_1)(1 + r_2)$. Over $n$ periods it is a long product, $\prod (1 + r_t)$. Products are hard to reason about; the logarithm rescues us because it turns a product into a sum: $\ln\!\big[\prod(1 + r_t)\big] = \sum \ln(1 + r_t)$. So the *average log return* governs long-run growth, and because we average logs — a concave function — Jensen's inequality applies with the *concave* sign. The average of the logs is *less than* the log of the average. That deficit, exponentiated back, is the drag.

### Two means, and why they disagree

There are two ways to summarize a sequence of returns, and the entire phenomenon is the gap between them.

- The **arithmetic mean** return is the plain average: add the period returns and divide by how many there are. It answers "what was my typical single-period return?"
- The **geometric mean** return is the constant rate that would have produced your actual ending wealth: take the product of the growth factors $(1 + r_t)$, take the $n$-th root, subtract 1. It answers "what did my money actually compound at?"

The geometric mean is what you eat. The arithmetic mean is what your monthly statement brags about. And Jensen guarantees the geometric mean is *always* less than or equal to the arithmetic mean, with equality only if every period's return was identical (zero volatility). The famous inequality $\text{AM} \ge \text{GM}$ — the arithmetic mean is at least the geometric mean — is itself just Jensen's inequality applied to the concave logarithm.

![Before and after comparison of plus fifty then minus fifty arithmetic view against the compounded reality ending at seventy five dollars](/imgs/blogs/convexity-jensen-math-for-quants-6.png)

#### Worked example: +50% then −50% on \$100

This is the example from the opening, now in full. You start with \$100.

- Year one: +50%. Your money becomes $\$100 \times 1.50 = \$150$.
- Year two: −50%. Your money becomes $\$150 \times 0.50 = \$75$.

You end at **\$75**, a loss of \$25, or −25% over the two years.

Now the two means:

- **Arithmetic mean** of the two returns: $(+50\% + (-50\%))/2 = 0\%$. The "average year" was flat. Your statement says you broke even on a per-year basis.
- **Geometric mean**: your money went from \$100 to \$75, a factor of 0.75 over two years. The per-year constant rate that does this is $\sqrt{0.75} - 1 = 0.8660 - 1 = -13.4\%$. Your money actually compounded at **−13.4% per year**.

The arithmetic mean said 0%; the geometric mean said −13.4%. The 13.4-percentage-point gulf is volatility drag, and it came entirely from the swings. The figure above shows both views side by side: the arithmetic column adds the percentages to zero, while the compounded column walks \$100 to \$150 to \$75 and lands at a −13.4% annual rate.

The one-sentence intuition: a 50% loss requires a 100% gain to recover, not a 50% gain — losses and gains are not symmetric in multiplicative space, and that asymmetry is the drag.

### The recovery math, made explicit

The asymmetry that drives the drag deserves its own moment, because it is the part that surprises even experienced investors. After a loss of fraction $\ell$, your money is multiplied by $(1 - \ell)$. To get back to even you need a gain $g$ such that $(1 - \ell)(1 + g) = 1$, which rearranges to $g = \ell / (1 - \ell)$. That denominator is the villain. A 10% loss needs $0.10/0.90 = 11.1\%$ to recover — close to symmetric, barely noticeable. A 50% loss needs $0.50/0.50 = 100\%$. A 90% loss needs $0.90/0.10 = 900\%$ — a tenfold gain just to break even. The required-recovery gain grows explosively as losses deepen, and that explosive growth *is* the convexity of $1/(1-\ell)$. This is why drawdowns matter so much more than their headline percentage suggests, and why risk managers obsess over capping the size of any single loss rather than chasing the size of any single gain.

The practical corollary for a track record: a strategy that makes +20% in nine years and loses −80% in the tenth has an arithmetic mean return of $(9 \times 20\% - 80\%)/10 = +10\%$ per year — a number that looks excellent on a marketing sheet. But the compounded reality is $1.20^9 \times 0.20 = 5.16 \times 0.20 = 1.03$, i.e. you turned \$1 into \$1.03 over ten years, a geometric return of about 0.3% per year. The single 80% loss erased almost a decade of 20% gains, because in multiplicative space one near-wipeout outweighs many good years. That gulf between a +10% arithmetic average and a +0.3% geometric reality is the most expensive lesson in this article.

### The drag formula every quant carries in their head

You do not have to compute geometric means from scratch. There is a beautiful approximation, again straight from the Jensen-gap formula $E[f(X)] \approx f(\mu) + \tfrac12 f''(\mu)\sigma^2$ applied to the log. If a strategy has arithmetic mean return $\mu$ per period and volatility $\sigma$ (standard deviation of returns) per period, the geometric (compounded) growth rate is approximately:

$$ \boxed{\,g \approx \mu - \tfrac{1}{2}\sigma^2\,} $$

The compounded rate is the arithmetic rate *minus half the variance*. The $-\tfrac12\sigma^2$ term is the volatility drag in symbols. It is not a fee, not a behavioral mistake, not bad timing — it is a mathematical certainty baked into the geometry of compounding. Two strategies with identical average returns will compound differently if their volatilities differ, and the more volatile one will always finish behind.

Let us sanity-check it against the worked example. There $\mu = 0$ (arithmetic mean), and the volatility of {+50%, −50%} around a mean of 0 is $\sigma = 0.50$, so $\sigma^2 = 0.25$. The formula predicts $g \approx 0 - 0.25/2 = -0.125$, i.e. −12.5%. The exact answer was −13.4%. The approximation is in the right ballpark; it is a second-order expansion, so it drifts when the swings are huge (50% is huge), but for the smaller, more realistic returns that fill a real track record it is excellent.

#### Worked example: the drag on a \$1,000,000 book

Consider a high-octane strategy with arithmetic mean return $\mu = 10\%$ per year and volatility $\sigma = 40\%$ per year — aggressive but not unheard of for a concentrated or leveraged book. What does it actually compound at, and what does the drag cost on a \$1,000,000 account?

$$ g \approx \mu - \tfrac{1}{2}\sigma^2 = 0.10 - \tfrac{1}{2}(0.40)^2 = 0.10 - \tfrac{1}{2}(0.16) = 0.10 - 0.08 = 0.02. $$

The arithmetic mean is 10%, but the compounded growth is only **2%**. The volatility ate **8 percentage points** of compounded return — eighty percent of the headline number — purely because of the swings.

In dollars on a \$1,000,000 book over one year: the naive investor expects $\$1{,}000{,}000 \times 0.10 = \$100{,}000$ of growth. The compounded reality is closer to $\$1{,}000{,}000 \times 0.02 = \$20{,}000$. The drag costs roughly **\$80,000 in the first year alone**, and because it compounds, the gap widens every year after. Run it ten years: at 10% the naive path implies $\$1{,}000{,}000 \times 1.10^{10} \approx \$2{,}593{,}742$; at the true 2% compounded rate you reach only $\$1{,}000{,}000 \times 1.02^{10} \approx \$1{,}218{,}994$. The drag has cost roughly **\$1.37 million** of imagined wealth over the decade.

The one-sentence intuition: volatility is not free even when it is symmetric — it deterministically shaves $\tfrac12\sigma^2$ off your compounded return, so two desks with the same average return but different risk end up in very different places.

## 3. Convex payoffs love volatility: gamma is long vol

We met the option payoff and saw it is convex. Now we cash that out into the single most important sentence on a derivatives desk: **positive gamma is long volatility.** It is Jensen's inequality wearing a trading hat.

### What an option is, defined from zero

An **option** is a contract that gives you the right, but not the obligation, to buy (a *call*) or sell (a *put*) something at a fixed *strike price* $K$ before a deadline. Because you can walk away when the trade is bad, your payoff is one-sided: a call pays $\max(S - K, 0)$ at expiry — you exercise only when the stock $S$ is above the strike, otherwise you let it expire worthless. That $\max(\cdot, 0)$ is the convex kink. We go much deeper into the mechanics in the companion piece on [options theory](/blog/trading/quantitative-finance/options-theory) and the [Black-Scholes model](/blog/trading/quantitative-finance/black-scholes); here we only need the shape.

### Gamma is the curvature of the payoff

Traders measure an option's sensitivities with Greek letters. **Delta** is the first derivative of the option's value with respect to the stock price — how much the option moves when the stock moves a dollar. **Gamma** is the *second* derivative — how much the delta itself changes as the stock moves. Gamma is, quite literally, the curvature of the option's value curve. And the option we hold has *positive gamma*: its value curve cups upward, it is convex in the stock price.

Here is where Jensen makes gamma pay. Suppose you hold a convex (positive-gamma) position and you stay *delta-neutral* — meaning you continuously hedge away the first-order exposure so that small up-and-down moves in the stock do not, on average, make or lose you money on the linear part. What is left is the curvature. By Jensen's inequality, a convex function of a random (jiggling) stock price has an expected value *above* the value at the average price. The jiggling itself — the realized volatility — feeds value into the convex position. The more the stock thrashes around, the more the convexity harvests. That is what "long volatility" means: you make money when realized volatility is high, regardless of direction.

![Tree from convexity branching into a unique global optimum and a positive Jensen gap with their consequences](/imgs/blogs/convexity-jensen-math-for-quants-5.png)

The tree above lays out everything convexity buys. The root is the single property — convexity. One branch leads to a *unique global optimum* (no local-minimum traps, a solver that always converges), which we take up in the next section. The other branch leads to a *positive Jensen gap* (the payoff loves variance, positive gamma is long volatility), which is the option story we are in now. Two very different-looking trading facts, one geometric cause.

#### Worked example: gamma harvesting on a delta-hedged call

You own one call option on a \$100 stock, and you have hedged it delta-neutral. The option's gamma is such that for every \$1 the stock moves, your delta changes by 0.04. Suppose over a day the stock makes a round trip: it rises \$2 to \$102, you re-hedge, then it falls \$2 back to \$100, you re-hedge again. The stock ended exactly where it started — a flat day, zero net move.

Yet you made money. Here is the mechanism. As the stock rose, your positive gamma meant your delta increased, so you were getting *longer* into the rally — you sold stock to re-hedge at the higher price. As the stock fell back, your delta decreased, so you bought stock back at the lower price. You sold high and bought low, automatically, because the convex payoff curved you into it. The profit from this round trip is approximately $\tfrac12 \times \Gamma \times (\Delta S)^2$ per move. With $\Gamma = 0.04$ and a \$2 move, that is $\tfrac12 \times 0.04 \times 2^2 = \tfrac12 \times 0.04 \times 4 = \$0.08$ per leg, and two legs give about **\$0.16** of gamma profit on a stock that did not move at all.

Notice the $(\Delta S)^2$ — the profit depends on the *square* of the move, which is variance, which is volatility. A flat-but-jumpy day pays the long-gamma holder; a flat-and-quiet day pays nothing. That is Jensen's $\tfrac12 f'' \sigma^2$ gap realized one re-hedge at a time.

The one-sentence intuition: positive gamma turns a jiggling price into profit because convexity makes you systematically sell into rallies and buy into dips, and the payout scales with the square of the moves — that is, with volatility.

### Theta is the rent you pay to be long convexity

There is no free lunch, and the option market prices the Jensen gap explicitly. The holder of a long-gamma option pays for the privilege through **theta** — the steady erosion of the option's value as time passes and the expiry deadline approaches, all else equal. Theta and gamma are two sides of one coin: the option seller charges enough theta to cover the gamma profits the buyer expects to harvest *if* realized volatility matches the volatility implied by the option's price. The whole game reduces to a single comparison. If the stock's realized volatility comes in *higher* than the implied volatility baked into the price you paid, your gamma harvest beats your theta rent and you win. If realized comes in *lower*, theta bleeds you faster than gamma feeds you and you lose. Owning convexity is therefore a bet that the world will be more uncertain than the price implies; selling it is a bet that the world will be calmer. The premium is not a gift to either side — it is the market's honest estimate of how much the Jensen gap will be worth, and you profit only by being right about volatility, not direction.

#### Worked example: gamma profit versus theta rent

Stay with the delta-hedged call from before, where a flat-but-jumpy day with two \$2 round-trip moves earned about \$0.16 of gamma profit. Suppose the option charges \$0.12 of theta per day — that is the daily time-decay rent. On this jumpy day you net $\$0.16 - \$0.12 = +\$0.04$: realized volatility was high enough that gamma beat theta. Now take a quiet day where the stock barely moves, say two \$0.50 round trips. Gamma profit is $\tfrac12 \times 0.04 \times 0.50^2 \times 2 = \tfrac12 \times 0.04 \times 0.25 \times 2 = \$0.01$. Against the same \$0.12 theta you *lose* \$0.11 on the day. Across a position of 10,000 shares, the jumpy day makes \$400 and the quiet day loses \$1,100. The break-even daily move — where gamma exactly pays the theta — is the implied volatility translated into a daily figure; beat it and you profit, miss it and you bleed.

The one-sentence intuition: being long convexity is renting volatility, theta is the rent, and you come out ahead only when the market delivers more movement than the price you paid assumed.

### The flip side: short gamma is short volatility

Whoever *sold* you that option has the mirror position: negative gamma, concave payoff, short volatility. They collected the premium up front and now bleed exactly your \$0.16 every time the stock thrashes. They are betting markets stay calm. This is the structural reason option *sellers* look brilliant in quiet markets and get carried out in crashes — they are short the convexity, short the Jensen gap, and a volatility spike turns the gap against them violently. We will see this in the real-markets section with the volatility blow-ups of 2018 and 2020.

## 4. Why a convex problem has one right answer

Convexity has a second, completely different superpower, and it is the reason the entire field of *convex optimization* exists: a convex optimization problem has a **unique global optimum** with no local-minimum traps. For a quant who builds portfolios or calibrates models, this is the difference between a solver you can trust and one that lies to you.

### Convex sets, and what a convex problem is

We need one more flavor of convexity: a **convex set**. A set of points is convex if, for any two points in it, the entire straight line between them also lies in the set. A filled disk is convex; a crescent moon is not (a chord between the two horns leaves the set). A *convex optimization problem* is one where you minimize a convex function over a convex set of allowed choices. Minimizing portfolio risk subject to "the weights add up to 100%" is convex: risk $w^\top \Sigma w$ is a convex function (its curvature, the covariance matrix, is positive semidefinite), and "weights sum to one" carves out a convex set.

### The no-traps guarantee

Imagine searching for the lowest point of a landscape in thick fog by always walking downhill. In a bumpy, non-convex landscape — many valleys — you will get stuck in whatever valley you happen to start near, a *local minimum* that may be nowhere near the true *global minimum*. You have no way to know you are stuck, because every direction is uphill. This is the nightmare of training a neural network or fitting a messy nonlinear model: the answer you get depends on where you started, and you can never be sure it is the best one.

A convex function has exactly one valley. Because the curve bows upward everywhere, there is no second bowl to fall into. Any point where the ground is flat — where the gradient (the slope in every direction at once) is zero — *must* be the single global minimum. Walking downhill always reaches it, from any starting point, and once there, you are provably done. This is why portfolio optimization, support-vector machines, LASSO regression, and a huge swath of model calibration are formulated as convex problems whenever possible: convexity converts "we found *an* answer" into "we found *the* answer." We work through the gradient mechanics of exactly this in [matrix calculus for optimization](/blog/trading/math-for-quants/matrix-calculus-optimization-math-for-quants).

![Matrix comparing convex and concave functions across second derivative, chord position, Jensen direction, and trading reading](/imgs/blogs/convexity-jensen-math-for-quants-4.png)

The matrix figure above contrasts the two curvatures property by property, and the no-traps story is the optimization reading of the convex column: positive curvature, chord above, Jensen gap positive, long volatility — and, crucially for a solver, one bowl and one bottom. The concave column flips every entry, which is why *maximizing* a concave function (like log-utility) enjoys the same one-answer guarantee — maximizing a concave function is the same as minimizing its convex negative.

#### Worked example: minimum-variance portfolio has one answer

Suppose you hold two assets and want the mix that minimizes risk. Asset A has 20% volatility, asset B has 30% volatility, and they are uncorrelated. Put fraction $w$ in A and $1 - w$ in B. The portfolio variance is a convex function of $w$:

$$ \sigma_p^2(w) = w^2 (0.20)^2 + (1-w)^2 (0.30)^2 = 0.04\,w^2 + 0.09\,(1-w)^2. $$

Because this is a convex (upward) parabola in $w$, it has exactly one bottom, and we find it by setting the derivative to zero: $0.08\,w - 0.18\,(1-w) = 0$, which gives $0.26\,w = 0.18$, so $w = 0.692$. Put **69.2%** in the lower-volatility asset A and **30.8%** in B. The resulting portfolio variance is $0.04(0.692)^2 + 0.09(0.308)^2 = 0.01916 + 0.00854 = 0.0277$, a volatility of $\sqrt{0.0277} = 16.6\%$ — *below* both assets' individual volatilities, the free lunch of diversification.

Here is the convexity payoff: because $\sigma_p^2(w)$ is convex, that 69.2% is provably *the* minimum, not a fluke of where the solver started. On a \$1,000,000 book, getting the genuine minimum rather than a nearby local trap might be the difference between 16.6% and 18% portfolio volatility — and over a year that lower risk is worth real money in reduced drawdowns and, via the drag formula, in higher compounded return.

The one-sentence intuition: convexity is the certificate that the optimum your solver returns is the real, unique, best answer — so you can build a portfolio without wondering whether a better one is hiding in a valley you never explored.

### What convexity buys a working desk, concretely

The uniqueness guarantee is not an abstraction a desk admires from afar; it changes how the code is written and trusted. Three concrete payoffs follow from formulating a problem as convex. First, **reproducibility**: a convex solver returns the same answer regardless of its starting guess, so two analysts who run the optimizer on the same data get the same portfolio — there is no "seed lottery" where the result depends on a random initialization, as there is in neural-network training. Second, **a stopping certificate**: convex solvers can produce a *duality gap*, a rigorous bound on how far the current solution could possibly be from the true optimum, so the solver can say "this answer is provably within 0.001% of the best possible" and stop with confidence rather than guessing. Third, **stability under small data changes**: because there is one bowl, nudging the inputs a little nudges the answer a little, instead of flipping it into a distant valley — the portfolio does not lurch wildly when one expected-return estimate ticks up.

The cost of convexity is *expressiveness*. Not every objective you might want is convex. A constraint like "hold at most 20 names" (a cardinality constraint) or "each position is either zero or at least 2% of the book" carves out a non-convex, disconnected set, and the clean guarantees evaporate — these become hard combinatorial problems. The art of practical optimization is keeping as much of the problem convex as possible, and isolating the genuinely non-convex pieces so they do not poison the whole. When you read that a portfolio optimizer uses *quadratic programming* or a *cone solver*, that is a desk deliberately staying inside the convex world to keep the uniqueness guarantee — the same guarantee Markowitz's variance objective handed the field for free in 1952.

## 5. Concave utility, risk aversion, and the bridge to Kelly

The last stop ties convexity to *how people feel about money*, and from there to the most famous bet-sizing rule in finance. The key object is a **utility function** — a curve that converts dollars into satisfaction.

### Why utility is concave

Ask yourself: does your hundredth dollar feel as good as your first? The first \$1,000 you ever save changes your life; the millionth \$1,000 barely registers. Each extra dollar adds *less* satisfaction than the one before — economists call this **diminishing marginal utility**. A curve whose extra-output-per-extra-input keeps shrinking is, by the second-derivative test, *concave*. The standard model is **log-utility**, $u(w) = \ln(w)$, which we already proved is concave ($u'' = -1/w^2 < 0$). It captures diminishing returns and, beautifully, it makes utility depend on *percentage* changes in wealth, which is exactly how compounding works.

### Concavity *is* risk aversion

Now run Jensen's inequality on a concave utility. For a concave $u$, $E[u(\text{wealth})] \le u(E[\text{wealth}])$. In words: the expected satisfaction from a *risky* outcome is less than the satisfaction of receiving its expected value for *certain*. A concave investor would rather have the average for sure than gamble around it. That is the precise mathematical definition of **risk aversion** — and it is nothing more than Jensen with the concave sign. The curvature of your utility curve *is* your aversion to risk; a flatter, more-curved-down utility means a more risk-averse person.

![Stack of concave utility showing the risk premium gap between the utility of the sure mean and the expected utility of a bet](/imgs/blogs/convexity-jensen-math-for-quants-7.png)

The stack above is the risk-aversion picture. At the top is $u(E[\text{wealth}])$, the satisfaction of taking the average outcome with certainty. Below it, separated by the *risk premium* gap, is $E[u(\text{wealth})]$, the satisfaction you actually expect from the gamble. The gap — the Jensen deficit for a concave function — is how much value the risk destroys for you. The bottom of the stack, the **certainty equivalent**, is the guaranteed dollar amount that would make you exactly as happy as the gamble; for a risk-averse person it sits below the gamble's expected dollar value, and the difference is what you would pay to avoid the risk.

#### Worked example: certainty equivalent of a coin-flip bet

You have \$100. Someone offers a fair coin flip: heads you win \$50 (ending at \$150), tails you lose \$50 (ending at \$50). The *expected dollar value* is $(\$150 + \$50)/2 = \$100$ — a fair bet, no edge either way. Should a log-utility investor take it?

Compute expected *utility*, not expected dollars:

$$ E[u] = \tfrac{1}{2}\ln(150) + \tfrac{1}{2}\ln(50) = \tfrac{1}{2}(5.0106) + \tfrac{1}{2}(3.9120) = 4.4613. $$

Now find the *certainty equivalent* — the sure amount $C$ whose utility equals this expected utility, $\ln(C) = 4.4613$, so $C = e^{4.4613} = \$86.60$. The risky bet, worth \$100 in expected dollars, is worth only **\$86.60** to this investor. They would happily pay up to \$13.40 to avoid the flip, even though it is mathematically fair. That \$13.40 is the risk premium — the Jensen gap on the concave log curve, in dollars.

The one-sentence intuition: for a risk-averse (concave-utility) investor, a fair gamble is a losing proposition in satisfaction terms, and the size of the loss is the Jensen gap — which is why people buy insurance and demand extra return for taking on risk.

### The bridge to the Kelly criterion

This is where the two halves of the post collide. A log-utility investor maximizes expected log-wealth. But expected log-wealth is *exactly the long-run geometric growth rate* — the very quantity that volatility drag attacks. So an investor who maximizes log-utility is automatically an investor who maximizes compounded growth and is automatically penalized by $\tfrac12\sigma^2$ of drag for taking on volatility. Out of this single objective falls the **Kelly criterion**: the bet size that maximizes log-wealth growth, which trades off the arithmetic edge $\mu$ against the variance penalty $\tfrac12\sigma^2$. Bet too small and you leave growth on the table; bet too big and the drag overwhelms the edge and your compounded growth turns negative even with a positive expected return. The full sequential-betting machinery, with the half-Kelly safety margin practitioners actually use, is in [the Kelly criterion and sequential betting](/blog/trading/quantitative-finance/kelly-criterion-sequential-betting-quant-interviews). For our purposes the headline is: Kelly is concavity (log-utility) and Jensen (the variance drag) fused into a position-sizing rule.

#### Worked example: why over-betting kills compounded growth

Take a bet with a genuine edge: arithmetic expected return $\mu = 20\%$ per round and volatility $\sigma = 50\%$ per round. The geometric growth at full exposure is $g \approx \mu - \tfrac12\sigma^2 = 0.20 - \tfrac12(0.25) = 0.20 - 0.125 = 7.5\%$. Now double your bet size (2x leverage). Returns scale linearly but volatility *also* scales linearly, so the variance quadruples: $\mu' = 40\%$, $\sigma' = 100\%$, $\sigma'^2 = 1.00$. The geometric growth becomes $g \approx 0.40 - \tfrac12(1.00) = 0.40 - 0.50 = -10\%$. You doubled your edge and turned a +7.5% compounder into a −10% bleeder, on a bet you have a real edge in. On \$1,000,000 that is the difference between growing to ~\$1,075,000 and shrinking to ~\$900,000 in one round — and the gap compounds.

The one-sentence intuition: because variance grows with the *square* of your bet size while edge grows only linearly, there is always a leverage level past which the volatility drag eats the edge, and finding that ceiling is exactly what the Kelly criterion does.

## Common misconceptions

**"The average return is the return I'll get."** No — the *arithmetic* average is not what your money compounds at. Your money compounds at the *geometric* mean, which is always lower by roughly half the variance. A fund advertising a 10% "average annual return" with high volatility may have actually compounded investors' money at far less. Always ask whether a quoted average is arithmetic or geometric; the gap is the volatility drag and it is real money.

**"Volatility is just risk, and risk is symmetric."** Volatility is symmetric in *arithmetic* space — a +5% and a −5% are equal-sized. But wealth compounds *multiplicatively*, and there the symmetry breaks: a −50% loss needs a +100% gain to recover, not +50%. Convexity (Jensen) is the precise statement of that asymmetry. Treating volatility as a neutral, two-sided wiggle hides the fact that it deterministically drags down compounded growth.

**"A convex payoff is good and a concave one is bad."** It depends entirely on which side you are on and what you are doing. Convexity is good when you *own* it (long options, long gamma — you profit from volatility) and bad when you are *short* it (you bled by volatility). Concavity of utility is not a flaw — it is the rational description of risk aversion. The shape is neutral; what matters is whether you are long or short the curvature.

**"If the expected price doesn't move, the option is worthless."** This is the exact error Jensen's inequality demolishes. A convex payoff converts the *spread* of outcomes into expected value, so an option can be worth real money even when the expected price equals the strike. Our worked example showed a \$10 expected payoff on an option that pays \$0 at the expected price. Pricing options off the expected price alone ignores the entire reason they exist.

**"Optimizers always find the best answer."** Only for convex problems. For non-convex problems (most neural-network training, many nonlinear calibrations) the solver finds *a* local minimum that depends on where it started, and there is generally no way to certify it is the global best. The reason quants reformulate problems as convex whenever they can is precisely to recover the uniqueness guarantee — convexity is what makes "the optimizer's answer" trustworthy.

**"More leverage on a winning strategy means more compounded money."** Up to a point, then it reverses. Leverage scales your edge linearly but scales your variance quadratically, so past the Kelly-optimal level the $\tfrac12\sigma^2$ drag grows faster than the edge and your compounded growth falls — eventually going negative even on a strategy with a positive expected return. The over-betting worked example turned a +7.5% compounder into a −10% bleeder by doubling the bet.

## How it shows up in real markets

### 1. Leveraged and inverse ETFs decay

Daily-rebalanced leveraged ETFs — a "3x" fund aims to deliver three times the index's *daily* return — are textbook volatility-drag machines. Because they reset their leverage every day, their variance is roughly nine times the index's, and the $\tfrac12 \sigma^2$ drag is correspondingly enormous. In choppy, sideways markets a 3x fund can lose money even when the underlying index ends flat, for exactly the +50%/−50% reason. During the volatile stretches of 2020, several leveraged products lost double-digit percentages versus their implied multiple of the index, and a few inverse-volatility products were liquidated outright. The prospectuses warn, in dense legalese, that the funds are unsuitable for holding longer than a day — that warning is Jensen's inequality translated into compliance language. The lesson: a product that delivers $3\times$ the daily return does *not* deliver $3\times$ the compounded return, and the gap is volatility drag.

### 2. The XIV implosion of February 2018

On February 5, 2018, an exchange-traded note called XIV — designed to profit from *selling* volatility, a short-gamma, short-convexity bet — lost about 96% of its value in a single after-hours session and was terminated days later, wiping out roughly \$2 billion. For years it had quietly earned the option-seller's premium in a calm market. Its holders were short the convexity, short the Jensen gap. When the VIX volatility index more than doubled in a day, the concave payoff turned violently against them. The episode is the cleanest real-world demonstration that short-volatility strategies look like free money right up until the convexity they sold comes due — the gains accrue slowly and linearly, the loss arrives convexly and all at once.

### 3. Long-Term Capital Management, 1998

LTCM ran convergence trades that were, in payoff terms, short convexity and heavily leveraged — many small gains punctuated by the risk of a large loss. Their models treated returns as roughly normal and underestimated both the variance and the tail. When Russia defaulted in August 1998, volatility exploded, correlations the models assumed were stable broke, and the leverage that magnified their tiny edge magnified the loss into a near-systemic event requiring a \$3.6 billion bailout. The volatility-drag and short-convexity lessons are both here: leverage scales variance quadratically, and a payoff that is concave in stress will hand back years of gains in days.

### 4. The 2008 "picking up nickels in front of a steamroller" trades

Many structured-credit and carry strategies before 2008 had the same shape: a steady, modest, *arithmetic* return that masked a deeply concave, short-convexity payoff. Sellers of credit protection and writers of out-of-the-money puts collected premium month after month — the slow, linear accrual. When the housing market cracked and volatility spiked, the convex losses arrived. The phrase traders use, "picking up nickels in front of a steamroller," is a folk statement of Jensen's inequality: a long string of small concave gains does not compensate for the rare convex loss, because the geometry is against you.

### 5. Volatility selling in the long bull market, 2012–2020

Through most of the 2010s, systematically selling equity-index options was a wildly profitable trade. Realized volatility kept coming in below the implied volatility priced into options, so option sellers harvested the difference. They were short gamma, short the Jensen gap, and for eight years the gap paid them. Then March 2020 arrived: the VIX hit the high 80s, realized volatility exploded, and many short-volatility funds gave back several years of returns in a few weeks. The pattern is identical to XIV, just slower-burning. The structural truth is that being short convexity earns a premium precisely *because* it loses badly in the tail — the premium is compensation for the convex risk, not free money.

### 6. Why insurers and option market-makers exist at all

The risk premium from the concave-utility worked example — the \$13.40 a risk-averse person pays to avoid a fair \$100 gamble — is the economic foundation of the entire insurance industry. People pay more than the actuarially fair price for insurance because their utility is concave; the certainty of a small premium beats the gamble of a large loss. Insurers, who pool many independent risks and so face far less variance per dollar, are effectively long the diversification that flattens their own utility curve. The same logic explains why option market-makers can charge a spread: they provide convexity (or absorb it) to customers who value certainty, and the Jensen gap is the raw material of the fee.

### 7. Diversification's free lunch is a Jensen effect

When you combine imperfectly correlated assets, portfolio variance falls *faster* than expected return, because variance is a convex (quadratic) function of the weights while return is linear. That is why the two-asset minimum-variance portfolio in our worked example had *lower* volatility (16.6%) than either of its components (20% and 30%). The "only free lunch in finance" is, mechanically, the convexity of the variance function interacting with the drag formula: lower portfolio variance means less $\tfrac12\sigma^2$ drag means higher compounded growth, for the same arithmetic return. Markowitz's whole framework is convexity put to work.

## When this matters to you

If you ever invest in anything that compounds — which is any portfolio you hold for more than one period — volatility drag is silently shaping your results. Two funds with the same advertised average return will leave you with different amounts of money, and the calmer one will usually win the compounding race. When you read a performance figure, ask whether it is arithmetic or geometric; the difference is roughly half the variance, and for a volatile strategy that can be the larger part of the headline number. When you size a position, remember that the relationship between bet size and growth is not linear — there is a ceiling (Kelly) past which more risk means *less* compounded wealth, and most blow-ups in this article are stories of people who blew through it.

If you ever buy insurance, an option, or any product with a one-sided payoff, you are trading the Jensen gap: paying a premium to convert uncertainty into certainty (insurance, on the concave side) or paying a premium to own convexity that pays off in turmoil (options, on the convex side). Knowing which side of the curvature you are on tells you whether time and calm are your friend or your enemy. And if you ever build or trust a model that optimizes anything, convexity is the property that tells you whether the answer it returns is *the* answer or merely *an* answer — a distinction worth real money when the output is the portfolio you are about to fund.

This is educational, not individualized advice; the point is the mechanism, not a recommendation to buy or sell anything. To go deeper, the natural next steps on this blog are [expectation, variance, and moments](/blog/trading/math-for-quants/expectation-variance-moments-math-for-quants) for the probability machinery behind every average in this post, [matrix calculus for optimization](/blog/trading/math-for-quants/matrix-calculus-optimization-math-for-quants) for the gradient mechanics that solve the convex problems, [the Kelly criterion](/blog/trading/quantitative-finance/kelly-criterion-sequential-betting-quant-interviews) for the bet-sizing rule that falls out of log-utility, and [options theory](/blog/trading/quantitative-finance/options-theory) for the convex payoffs that turn volatility into value. Convexity is the thread running through all four; once you can see the bend in a curve, you can see who pays and who gets paid when the world gets uncertain.
