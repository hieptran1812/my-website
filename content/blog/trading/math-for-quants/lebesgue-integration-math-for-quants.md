---
title: "Lebesgue vs Riemann integration: the picture that makes expected value finally click"
date: "2026-06-15"
description: "A beginner-friendly, build-from-zero tour of why Lebesgue integration slices the y-axis instead of the x-axis, why expected value is literally a Lebesgue integral, and why this is the one definition that prices discrete, continuous, and mixed payoffs alike."
tags: ["lebesgue-integration", "riemann-integration", "expectation", "measure-theory", "monte-carlo", "dominated-convergence", "fubini", "st-petersburg-paradox", "quant-finance", "math-for-quants"]
category: "trading"
subcategory: "Quantitative Finance"
author: "Hiep Tran"
featured: true
readTime: 43
---

> [!important]
> **TL;DR** — Lebesgue integration cuts the *y*-axis (it groups inputs by the value they produce and weighs each value by how likely it is), and that single shift is why expected value is *literally* an integral: $E[X] = \int X \, dP$.
>
> - **Riemann** slices the input axis into thin vertical strips and sums their areas; **Lebesgue** slices the output axis into horizontal layers and asks "how much probability lands at each value?"
> - Because Lebesgue weighs *values* by *probability*, one formula — $E[X]=\int X\,dP$ — computes the mean of a coin flip, a continuous stock return, **and** a mixed payoff like an option with a lump of probability sitting exactly at the strike. No separate rule for each case.
> - The **convergence theorems** (monotone and dominated convergence) are the legal permission slip to swap a limit with an expectation — which is exactly what a Monte-Carlo price or a sum of discounted cash flows relies on.
> - Some bets have **no usable expected value at all** (the St. Petersburg payout, a Cauchy-distributed PnL): the integral diverges or is undefined, and that is a precise mathematical warning that you cannot size the bet by EV.
> - The one fact to remember: $E[X]=\int X\,dP$ is a single, type-agnostic definition of "average outcome", and it only exists when the integral of the *absolute* payoff is finite.

Here is a question that sounds trivial until you try to answer it cleanly: what is the *average* payoff of a bet? For a coin that pays \$100 on heads and \$0 on tails, you would say \$50, and you would compute it as $100 \times 0.5 + 0 \times 0.5$. For a stock whose return tomorrow could be any number on a smooth bell curve, you would reach for a different tool — an integral of the return times its probability density. And for an option that pays \$0 below a strike, ramps up above it, but has a strange *lump* of probability sitting exactly at the strike price, you might not know which tool to grab at all. Discrete needs a sum. Continuous needs an integral. The lump needs... something in between?

It turns out there is one definition of "average" that handles all three with no special cases, no switching tools, no awkward seams. That definition is the **Lebesgue integral**, and the entire reason it works is a small, almost philosophical change in *how you slice the area under a curve*. The figure below is the whole idea in one picture, and the rest of this post is a slow, careful tour of it.

![Riemann slices vertical strips on the x-axis while Lebesgue slices horizontal value bands on the y-axis](/imgs/blogs/lebesgue-integration-math-for-quants-1.png)

The diagram above is the mental model for the entire post. On the left, Riemann integration chops the horizontal input axis into thin strips and adds up the area of each one — the way you were taught in your first calculus class. On the right, Lebesgue integration chops the *vertical* output axis into bands of value and asks, for each band, "how much of the input maps into here?" That second question — weighing each *value* by how much *probability* produces it — is exactly the question a probabilist asks when computing an expected value. So expectation is not *like* an integral. Expectation *is* a Lebesgue integral. Once you see that, a lot of measure-theoretic machinery stops being intimidating abstraction and becomes the most natural bookkeeping in the world.

This is the kind of topic where it is fair to ask, up front, *how much does a working trader actually need this?* The honest answer: you will rarely if ever sit down and compute a Lebesgue integral by hand. But you will constantly rely on three things this framework guarantees — that "expected value" is well-defined for the irregular payoffs you actually trade, that a Monte-Carlo average really does converge to the price you want, and that some "expected values" are a mirage you must never size a position by. Those three guarantees are worth understanding from the ground up, even if the proofs stay on the shelf.

## Foundations: the building blocks

Before we can say what a Lebesgue integral *is*, we need a small vocabulary. Every term here gets defined from zero. A practitioner can skim; a beginner should not skip.

### Outcomes, the sample space, and a measure

Start with the set of all the ways the world could turn out. We call it the **sample space** and write it $\Omega$ (the Greek letter omega). If you flip a coin, $\Omega = \{\text{heads}, \text{tails}\}$. If you watch a stock's price tomorrow, $\Omega$ is the set of all possible prices — every non-negative number. A single point $\omega \in \Omega$ is one specific outcome: "heads", or "the price closed at \$103.42".

Next we need a way to assign *size* to subsets of $\Omega$. A **measure** is exactly that: a rule $\mu$ (the Greek letter mu) that takes a set and returns a non-negative number — its "size", "length", "area", or "weight". On the real line, the natural measure is ordinary length: the interval from 2 to 5 has measure 3. This length-measure is called **Lebesgue measure**, named after Henri Lebesgue, the French mathematician who built this theory around 1902.

A **probability measure** $P$ is a measure with one extra rule: the whole space has size exactly 1, i.e. $P(\Omega) = 1$. So $P$ assigns each event a number between 0 and 1 — its probability. The pair "sample space plus a probability measure" is how we encode all the uncertainty in a market. (There is a third ingredient, the *sigma-algebra* of measurable events, which we touch on shortly; if you want that piece in full depth, it gets its own treatment in the [expectation and moments](/blog/trading/math-for-quants/expectation-variance-moments-math-for-quants) companion post.)

### A random variable is just a function

A **random variable** sounds mysterious, but it is nothing more than a function that takes an outcome and returns a number. Write it $X : \Omega \to \mathbb{R}$. For the coin, $X(\text{heads}) = 100$ and $X(\text{tails}) = 0$ — that is a bet paying \$100 on heads. For the stock, $X(\omega)$ might be the dollar value of an option given that the price ended at the value $\omega$ describes. The random variable *is* the payoff. It maps "what happened" to "how many dollars you made".

So when we eventually write $E[X] = \int X \, dP$, every symbol is now concrete: $X$ is the payoff function, $P$ is the probability weighting, and the integral sign is the instruction to add it all up the *Lebesgue* way. The pipeline below traces that exact path.

![Pipeline from states of the world through a payoff function and probability weighting to the expected value](/imgs/blogs/lebesgue-integration-math-for-quants-2.png)

Read the pipeline left to right: the states of the world feed into the payoff $X$, which turns each state into a dollar amount; the probability measure $P$ says how heavily to weight each state; and integrating the payoff against $P$ produces the expected value $E[X]$. That is the whole machine. Everything else in this post is either explaining *why* the Lebesgue way of adding up is the right way, or showing *when* the machine works and when it jams.

### The Riemann integral, briefly, so we can contrast it

The integral you already know is the **Riemann integral**, named after Bernhard Riemann. To compute $\int_a^b f(x)\,dx$ — the area under a curve $f$ between $x=a$ and $x=b$ — you cut the interval $[a,b]$ into many thin vertical strips. Each strip has some small width $\Delta x$ and a height roughly equal to $f$ somewhere in that strip. The area of one strip is height times width, $f(x_i)\,\Delta x$. You add up all the strips and then let the strips get infinitely thin:

$$
\int_a^b f(x)\,dx = \lim_{\Delta x \to 0} \sum_i f(x_i)\,\Delta x.
$$

Here $f$ is the function being integrated, $x_i$ is a sample point inside the $i$-th strip, $\Delta x$ is the strip width, and the limit takes the strips to zero width. The defining move of Riemann is: **partition the input axis** (the $x$-axis). You decide *where* to cut along the bottom, and the heights come along for the ride.

### The Lebesgue integral: partition the *output* instead

Lebesgue had a different, almost mischievous idea. Instead of cutting the input axis, cut the **output axis** (the $y$-axis). Pick value-levels $y_0 < y_1 < y_2 < \dots$ along the vertical. For each thin band of values $[y_k, y_{k+1})$, ask a single question: *how much of the input domain produces a value in this band?* Call that amount $\mu(\{x : y_k \le f(x) < y_{k+1}\})$ — the measure (the size, the total probability) of all inputs whose output lands in the band. Then the contribution of that band is its value $y_k$ times that measured amount, and you sum over all bands:

$$
\int f \, d\mu = \lim \sum_k y_k \cdot \mu\big(\{x : y_k \le f(x) < y_{k+1}\}\big).
$$

In words: **value times "how much input gives that value"**. Each symbol: $f$ is the function, $y_k$ is a value level, the curly-brace set is "all inputs producing roughly that value", $\mu$ measures the size of that set, and we sum value-times-size over all the value bands. The famous one-liner, attributed to Lebesgue himself, is the coin-counting analogy: to add up a pile of coins, the Riemann way is to go in the order you grabbed them; the Lebesgue way is to first sort them into stacks by denomination — all the dimes together, all the quarters together — and then multiply each denomination by how many coins are in its stack. Same total, smarter bookkeeping.

> Riemann asks "where am I along the bottom?"; Lebesgue asks "how much input lands at this height?" That single switch is the whole theory.

Why does it matter which axis you cut? Because when the function is well-behaved — smooth, continuous, no wild oscillation — both methods give the same answer, and Riemann is often easier to compute. But when the function is *ugly*, Riemann's insistence on cutting the input axis breaks down, while Lebesgue's "sort by value" approach sails right through. And in probability, where the natural weight is *how likely each value is*, the Lebesgue framing is not just more robust — it is the only one that makes the definitions clean. We will see all of this concretely.

### Measurable functions: the gatekeeper

There is one admission requirement to play the Lebesgue game. A function is **measurable** if, for every value band you might pick, the set of inputs that land in it is itself a "measurable set" — a set the measure $\mu$ knows how to size. This is a low bar in practice: every continuous function is measurable, every step function is measurable, every function you would ever write down for a financial payoff is measurable. Measurability is the gatekeeper, but almost everything you care about walks right through the gate.

![Tree of integration ideas from measurability through the Lebesgue integral to expectation convergence and Fubini](/imgs/blogs/lebesgue-integration-math-for-quants-5.png)

The tree above lays out the whole conceptual map so you can see where each piece sits. At the top is the task: integrate a payoff. The first branch is the measurability check — is this even a legitimate random variable? If yes, it becomes a valid random variable; if no, the integral simply does not exist. The other main branch is the Lebesgue integral itself, which yields the expectation $E[X]$ and is supported underneath by the convergence theorems we will meet in section 4. Keep this map in the corner of your mind; every section below is filling in one node of it.

With the vocabulary in place, let us do the simplest possible worked example to anchor everything.

#### Worked example: the expected value of a simple coin bet, the Lebesgue way

You take a bet that pays \$100 if a fair coin lands heads and \$0 if it lands tails. The payoff is a random variable $X$ with $X(\text{heads}) = 100$ and $X(\text{tails}) = 0$. The probability measure is $P(\text{heads}) = 0.5$ and $P(\text{tails}) = 0.5$.

The Lebesgue recipe says: sort by value. There are exactly two values the payoff takes, \$100 and \$0. For the value \$100, the set of outcomes producing it is $\{\text{heads}\}$, with probability mass $P(\{\text{heads}\}) = 0.5$. For the value \$0, the set is $\{\text{tails}\}$, with mass 0.5. Now multiply each value by its mass and add:

$$
E[X] = \int X \, dP = 100 \times 0.5 + 0 \times 0.5 = 50.
$$

So $E[X] = \$50$. Notice that this *is* the schoolbook formula "probability-weighted average" — but we got there by the Lebesgue route of grouping by value and weighing by probability. For a payoff that takes only finitely many values, the Lebesgue integral is exactly the familiar sum. The intuition this example teaches: the Lebesgue integral generalizes the everyday "value times probability" sum, so the simplest cases are unchanged and only the messy cases get the upgrade.

## 1. The two pictures, side by side

Let us slow down on the central contrast, because everything downstream depends on it. Picture a payoff drawn as a curve over the set of outcomes — height equals dollars. (The before-and-after figure that opened the post is the reference; we are now narrating it.) The two integration methods are two different ways to compute the "weighted area" under that curve.

The Riemann method walks along the bottom. It chops the outcome axis into tiny slivers, measures the payoff's height above each sliver, multiplies height by sliver-width, and sums. Its weakness is that it commits to a fixed partition of the *inputs* before it ever looks at the *outputs*. If the payoff jumps around violently as you move a hair along the input axis — if neighboring inputs produce wildly different values — then no matter how finely you chop the bottom, the height inside each sliver is ambiguous, and the sums refuse to settle on a single answer.

The Lebesgue method walks along the side. It chops the *value* axis into bands and, for each band, gathers up *all* the inputs that produce a value in that band — no matter how scattered those inputs are across the bottom. It does not care if the qualifying inputs are spread all over the place; it just needs to *measure their total size*. Then it weights that value by that size and sums. Because it never needs neighboring inputs to behave similarly, it handles violently irregular functions that Riemann chokes on.

In finance terms, the Lebesgue "band" is "the set of all market states in which my payoff is between \$40 and \$41", and the measure of that band is "the total probability of those states". Value times probability, summed over all value bands — that is an expected payoff. This is why the second figure drew the path from outcomes, through the payoff, through probability weighting, to $E[X]$: it is the Lebesgue picture relabeled in the language of money and probability.

### Why the y-axis cut is the natural one for probability

Here is the punchline of section 1, and it is worth stating sharply. In probability, the thing you have a handle on is *the distribution of values* — how likely each payoff size is. You usually do *not* have a clean handle on the underlying outcome space $\Omega$ (it might be an infinite-dimensional space of price paths). The Riemann approach wants to partition $\Omega$, the thing you cannot see well. The Lebesgue approach partitions the *values*, the thing you can see — and weighs each value by its probability, which is precisely the information a distribution gives you. The math lines up with what you actually know. That alignment is not a coincidence; it is why measure-theoretic probability is built on the Lebesgue integral and not the Riemann one.

## 2. Expectation IS a Lebesgue integral

Now we make the central claim fully precise and then beat on it until it feels obvious.

For a random variable $X$ on a probability space, the **expected value** is *defined* as the Lebesgue integral of $X$ against the probability measure $P$:

$$
E[X] = \int_\Omega X(\omega) \, dP(\omega) = \int X \, dP.
$$

Every symbol: $\Omega$ is the set of outcomes, $X(\omega)$ is the payoff in outcome $\omega$, $P$ is the probability measure, and $dP(\omega)$ is the "probability weight" attached to the neighborhood of outcome $\omega$. This single line is the definition of expectation in modern probability. There is no separate definition for discrete versus continuous; this is *the* definition, and the familiar formulas you learned earlier are special cases of it.

Let us recover those special cases, because seeing the one definition collapse into the two you already know is what makes it trustworthy.

**The discrete case.** Suppose $X$ takes values $x_1, x_2, \dots$ with probabilities $p_1, p_2, \dots$. The Lebesgue integral "sorts by value", so it groups outcomes by which $x_i$ they produce, and the measure of each group is just $p_i$. The integral becomes the sum:

$$
E[X] = \sum_i x_i \, p_i.
$$

That is the schoolbook discrete expectation — value times probability, summed.

**The continuous case.** Suppose $X$ has a probability density function $f$, meaning the probability of landing in a tiny value-band $[x, x+dx]$ is $f(x)\,dx$. The Lebesgue integral against $P$ becomes a Riemann-style integral against length, weighted by the density:

$$
E[X] = \int_{-\infty}^{\infty} x \, f(x) \, dx.
$$

That is the schoolbook continuous expectation. The reason a density even *exists* — the reason you can write $dP(x) = f(x)\,dx$ — is a deep result called the Radon-Nikodym theorem, but you do not need its machinery to use the formula. You just need to trust that "integrate value against probability" specializes to "integrate value against density" whenever a density is available.

The matrix figure below is the payoff: one definition, three payoff types, no separate rules.

![Matrix showing discrete continuous and mixed payoffs all handled by the same integral against P](/imgs/blogs/lebesgue-integration-math-for-quants-3.png)

Each row of the matrix is a payoff type, and the last column is the same machine — integrate the payoff against the probability measure — producing the mean every time. For a discrete bet, the probability mass sits in lumps at a few values and the integral becomes a sum. For a continuous payoff, the mass spreads out as a smooth density and the integral becomes the density integral. For a mixed payoff, the mass is *both* a lump and a spread, and — this is the magic — the *same* Lebesgue integral handles it without any new rule. Let us now do that mixed case explicitly, because it is the one Riemann-flavored thinking handles worst and the one that shows up most in derivatives.

#### Worked example: a digital option with a point mass at the strike

A **digital option** (also called a binary option) is a bet on whether a stock finishes above a strike price. Concretely: a cash-or-nothing digital pays \$100 if the stock closes above \$50 at expiry, and \$0 otherwise. Suppose the probability of finishing above \$50 is $p = 0.40$. The payoff $X$ takes value \$100 with probability 0.40 and \$0 with probability 0.60. As a Lebesgue integral against $P$:

$$
E[X] = 100 \times 0.40 + 0 \times 0.60 = 40.
$$

So the fair (undiscounted, risk-neutral) value of this digital is \$40. That part is just the coin bet again. Now make it genuinely *mixed*, which is where the single definition earns its keep.

Consider instead a payoff that is partly continuous and partly a lump. Say you hold a structure that pays you the stock's *excess over \$50* if it finishes above \$50 (a continuous ramp), **and** pays a fixed rebate of \$20 in the special event that the stock finishes *exactly* at \$50, an event the market assigns a non-zero probability of, say, 0.05 because of a large resting order and a pin at that price. (Pinning at a round strike on expiry is a real, observed phenomenon.) The distribution of this payoff has a *point mass* — a lump of probability 0.05 sitting at one specific value — glued onto a continuous density for the ramp.

How do you take the expectation? With Riemann thinking you would panic: a density integral cannot "see" a single point (a point has zero width, so $f(x)\,dx$ at one point contributes nothing), yet there is clearly \$20 × 0.05 = \$1 of expected value living right there. The Lebesgue integral has no such trouble. It splits the measure into its lump part and its spread part and integrates each:

$$
E[X] = \underbrace{20 \times 0.05}_{\text{point mass}} \;+\; \underbrace{\int_{50}^{\infty} (s - 50)\, f(s)\, ds}_{\text{continuous ramp}}.
$$

Suppose the continuous ramp integral works out to \$6.50. Then $E[X] = \$1.00 + \$6.50 = \$7.50$. The point mass contributed \$1.00 that a naive density integral would have silently dropped. The intuition this teaches: a single definition, $E[X]=\int X\,dP$, automatically accounts for lumps of probability that the continuous-only formula throws away — which is exactly the situation you face with options that pin at a strike, barriers that knock out at a level, or any payoff with a discontinuity sitting where probability piles up.

### What this costs and when it bites

The cost of getting this wrong is real money. If a desk prices a structure with a point mass by treating the distribution as purely continuous, it will misprice the lump — and lumps cluster exactly at the strikes, barriers, and round numbers where order flow concentrates and where the desk is most exposed. The Lebesgue framing is the bookkeeping that keeps the lump on the balance sheet.

## 3. Measurable functions and the few cases Riemann can't reach

We claimed Lebesgue can integrate functions Riemann cannot. Let us see the canonical example, understand why it breaks Riemann, and then — crucially — explain why this almost never happens to a real payoff.

#### Worked example: a function Riemann can't integrate but Lebesgue can

Define a function on the interval $[0,1]$ that is 1 at every *rational* number (every fraction, like 1/2 or 7/9) and 0 at every *irrational* number (like $\sqrt{2}/2$). This is the **Dirichlet function**. To make it a payoff: suppose, absurdly, a contract pays \$1 if a uniformly random number drawn from $[0,1]$ is rational and \$0 if it is irrational.

Try the Riemann method. Chop $[0,1]$ into tiny strips. In *every* strip, no matter how thin, there are both rational and irrational numbers (both are dense). So if you sample the height at a rational point, the strip's height is 1; if you sample at an irrational point, it is 0. The "upper" sum (always picking height 1) gives total area 1; the "lower" sum (always picking height 0) gives total area 0. They never agree, no matter how fine the strips. Riemann therefore declares the integral **does not exist**.

Now the Lebesgue method. Sort by value. The function takes only two values, 1 and 0. The set producing the value 1 is "all rationals in $[0,1]$". Here is the key fact: the rationals are **countable** (you can list them one by one), and a countable set has Lebesgue measure **zero** — its total length is 0, because you can cover the whole list with intervals of total length as small as you like. So the value 1 occurs on a set of measure 0, and the value 0 occurs on a set of measure 1 (the irrationals fill up almost all of $[0,1]$). The Lebesgue integral is therefore:

$$
\int_0^1 \mathbf{1}_{\text{rational}} \, d\mu = 1 \times \underbrace{0}_{\text{measure of rationals}} + 0 \times \underbrace{1}_{\text{measure of irrationals}} = 0.
$$

So the expected payoff of the "pay \$1 if rational" contract is \$0 — you will draw an irrational number with probability 1, so you almost surely get nothing. The intuition this teaches: Lebesgue can assign a sensible value (here, \$0) to a payoff so jagged that Riemann cannot even define it, because Lebesgue only needs to *measure the size* of each value-set, not require neighboring inputs to behave alike.

### Why your real payoffs are nice

Here is the reassuring part, and it is important not to oversell measure theory as a daily tool. The Dirichlet function is a deliberately pathological object — a payoff that flips between two values infinitely often in any interval. Real financial payoffs are nothing like it. An option payoff is *piecewise linear and continuous* (flat below the strike, a straight ramp above it). A digital is a *step function* (one jump). A bond's cash flows are a *finite list of dated lumps*. Every one of these is measurable, every one is Riemann-integrable too, and for every one of them Riemann and Lebesgue give the identical answer.

So why bother with Lebesgue if your payoffs are well-behaved? Three reasons, in increasing order of importance. First, the *mixed* payoffs from section 2 (continuous-plus-lump) are awkward for Riemann and natural for Lebesgue. Second, and more fundamentally, the *limits* of nice payoffs — what happens as you take a Monte-Carlo sample size to infinity, or refine a time grid — are governed by the convergence theorems, and those theorems are stated and proved in the Lebesgue framework and simply do not hold in general for the Riemann integral. Third, the whole edifice of stochastic calculus, risk-neutral pricing, and the Itô integral is built on Lebesgue foundations; you cannot even *state* "$E[X]$ exists" rigorously for a price path without it. The everyday payoff is nice; the *theory that guarantees your tools work* needs Lebesgue.

## 4. The convergence theorems: permission to swap limit and expectation

This is the section where Lebesgue integration stops being a curiosity and becomes the load-bearing wall under your entire numerical toolkit. The question it answers is deceptively simple but constantly assumed: **when is the limit of the expectations equal to the expectation of the limit?**

Why do you care? Because a Monte-Carlo price is a limit. You simulate $n$ paths, average the payoffs, and *assume* that as $n \to \infty$ this average converges to the true expected payoff. A discounted-cash-flow valuation is a limit and a sum: you add up infinitely many tiny or numerous cash flows and *assume* you can interchange the sum with the expectation. Every time you write "the average converges to the mean" or "I can push the expectation inside the sum", you are invoking a convergence theorem. Most of the time it is true. Sometimes — and these are the times that blow up models — it is false, and knowing the conditions is what separates a robust model from a fragile one.

Formally, the danger is that for a sequence of payoffs $X_n$ converging to a limit $X$, it is *not* automatically true that

$$
\lim_{n\to\infty} E[X_n] \overset{?}{=} E\Big[\lim_{n\to\infty} X_n\Big] = E[X].
$$

The left side is "average each one, then take the limit"; the right side is "take the limit first, then average". They can disagree, sometimes wildly (the limit of the expectations can be finite while the expectation of the limit is something else entirely). The convergence theorems are the precise conditions under which the swap is legal.

![Stack of the monotone convergence dominated convergence and Fatou theorems leading to a safe limit swap](/imgs/blogs/lebesgue-integration-math-for-quants-4.png)

The stack above lists the three workhorses, top to bottom, ending in the payoff: a safe swap of limit and expectation. Let us take them in order.

### Monotone convergence

The **Monotone Convergence Theorem (MCT)** says: if your payoffs are non-negative and they only ever *increase* toward their limit — $0 \le X_1 \le X_2 \le X_3 \le \dots$ climbing up to $X$ — then you may swap freely:

$$
\lim_{n\to\infty} E[X_n] = E[X].
$$

The condition is "non-negative and rising". Here $X_n$ is the $n$-th approximation and $X$ is its increasing limit. This is the gentlest theorem and the one that justifies building up a complicated payoff as an increasing sum of simpler ones — for example, valuing a perpetual stream of cash flows as the rising limit of its finite partial sums. As long as everything is non-negative and you are adding more each step (never subtracting), the limit of the values equals the value of the limit.

### Dominated convergence

The **Dominated Convergence Theorem (DCT)** is the one you will lean on most, because it does not require the payoffs to be non-negative or to climb monotonically — they can wobble up and down, as Monte-Carlo payoffs do. It says: if $X_n \to X$ and there exists a single fixed payoff $g$ with finite expectation that *caps* all of them — $|X_n| \le g$ for every $n$, and $E[g] < \infty$ — then the swap is legal:

$$
\lim_{n\to\infty} E[X_n] = E[X].
$$

The condition is "a fixed, integrable cap dominates the whole sequence". Here $g$ is the dominating function — a ceiling that no $X_n$ ever pokes through, and whose own expectation is finite. The dominating cap is what stops probability mass from "escaping to infinity" and corrupting the limit. We will use DCT directly to justify Monte-Carlo in the next section.

### Fatou's lemma

**Fatou's lemma** is the weakest of the three and acts as a one-sided safety net when you cannot verify the conditions for the other two. For non-negative payoffs it guarantees only an inequality:

$$
E\Big[\liminf_{n\to\infty} X_n\Big] \le \liminf_{n\to\infty} E[X_n].
$$

Here $\liminf$ is the "limit of the smallest tail values". In plain terms, Fatou says the expectation of the eventual limit can be no *bigger* than the limiting expectation — value cannot be created out of nowhere in the limit, though it *can* leak away. It is the theorem you reach for to prove a bound when equality is too much to ask. For a risk manager, the lesson encoded in Fatou is sobering: when you cannot guarantee a cap, the safest assumption is that taking limits can only *lose* you expected value, never manufacture it.

### A concrete cautionary tale: when the swap fails

To feel why the conditions matter, consider a sequence where the swap *fails*. Suppose payoff $X_n$ pays \$$n$ with probability $1/n$ and \$0 otherwise. Then $E[X_n] = n \times (1/n) = \$1$ for every $n$, so $\lim E[X_n] = \$1$. But as $n \to \infty$, the probability of the big payout, $1/n$, shrinks to 0 — so the *limiting* payoff $X$ is \$0 with probability 1, giving $E[X] = \$0$. The two sides disagree: \$1 on the left, \$0 on the right. The expected value "escaped to infinity" because there was no fixed integrable cap (the payouts grew without bound). This is exactly the failure DCT rules out by demanding the dominating function $g$. The lesson: a model that quietly assumes "average converges to mean" can be silently wrong when the payoff's tail grows faster than its probability shrinks — and that is not a hypothetical, it is the shape of every fat-tailed blow-up.

## 5. Dominated convergence: the license to Monte Carlo

Now we cash in the theory on the single most common numerical task in quant finance: pricing something by simulation. The whole method rests on dominated convergence, and seeing the connection explicitly is what makes the convergence theorems feel useful rather than ornamental.

A **Monte-Carlo price** works like this. You want the expected (discounted) payoff $E[X]$ of some derivative, but the integral is too hard to do by hand. So you draw $n$ independent random scenarios $X^{(1)}, X^{(2)}, \dots, X^{(n)}$ from the right distribution, evaluate the payoff in each, and take the sample average:

$$
\hat{X}_n = \frac{1}{n} \sum_{i=1}^{n} X^{(i)}.
$$

Here $\hat{X}_n$ is the average over $n$ simulated payoffs and each $X^{(i)}$ is the payoff in one simulated scenario. The Law of Large Numbers says this average converges to $E[X]$ as $n \to \infty$ — but the *clean* statement of that law, and the interchange that lets you treat the simulated average as an estimate of the true integral, lives in the Lebesgue convergence framework. Specifically, justifying that the *expectation of the average equals the true expectation* (so the estimator is unbiased and converges) is exactly a limit-swap of the kind DCT licenses, provided the payoff is dominated by something integrable.

![Pipeline from simulating paths through averaging payoffs and dominated convergence to the true price](/imgs/blogs/lebesgue-integration-math-for-quants-7.png)

The pipeline above is the Monte-Carlo argument in five steps: simulate many price paths, evaluate each payoff, average them, confirm dominated convergence holds, and conclude that the average lands on the true price. The fourth node is the one people skip and the one that matters: *dominated convergence holds*. If the payoff is bounded — and a digital option's payoff is bounded by \$100, a call spread's by the spread width — then there is a trivial dominating function (the constant bound), DCT applies, and the simulation provably converges to the integral you wanted. If the payoff is *unbounded* and heavy-tailed, that node fails, and the simulation can wander forever without settling. Let us make the convergence concrete with numbers.

#### Worked example: a Monte-Carlo average converging to a digital's price

Recall the digital that pays \$100 with probability $p = 0.40$ and \$0 otherwise, so the true expected payoff is $E[X] = \$40$. We do not "know" 0.40 in the simulation; we only draw scenarios. Suppose after running batches of simulations we observe these running averages of the payoff:

| Simulations $n$ | Heads-above-strike count | Sample average $\hat{X}_n$ | Error vs \$40 |
|---|---|---|---|
| 100 | 46 | \$46.00 | +\$6.00 |
| 1,000 | 421 | \$42.10 | +\$2.10 |
| 10,000 | 3,970 | \$39.70 | −\$0.30 |
| 100,000 | 40,050 | \$40.05 | +\$0.05 |
| 1,000,000 | 399,800 | \$39.98 | −\$0.02 |

At 100 paths the estimate is off by \$6.00; by a million paths it is within \$0.02 of the true \$40.00. The error shrinks roughly like $1/\sqrt{n}$ — quadruple the paths, halve the error — which is the signature convergence rate of Monte-Carlo. Crucially, the payoff here is *bounded* by \$100, so dominated convergence applies with the constant cap $g = \$100$ (and $E[g] = \$100 < \infty$), and the average is *guaranteed* to converge to the integral $E[X] = \$40$. The intuition this teaches: dominated convergence is the exact mathematical permission slip that turns "I simulated and averaged" into "my average provably approaches the true price", and it works precisely because the payoff is capped.

This connects directly to the practical mechanics of running simulations, which we cover in the dedicated [Monte-Carlo simulation walkthrough](/blog/trading/quantitative-finance/monte-carlo-simulation-coding-quant-interviews) — that post is the "how to code it" companion to this "why it converges" one.

### When the license is revoked

Flip the example. Suppose instead of a bounded digital you are simulating a payoff with a fat tail so heavy that its true expectation is infinite (we will build exactly such a payoff in the next section). Then there is *no* integrable dominating function, DCT does not apply, and the sample average does **not** converge to anything stable. You will see the running average lurch upward every time a rare giant draw appears, never settling. A practitioner who trusts the average anyway is sizing a position by a number that does not exist. That is the bridge to the most important cautionary section of this post.

## 6. Integrable vs non-integrable: when "expected value" is a lie

So far we have treated $E[X]$ as something that always exists once $X$ is measurable. That is wrong, and the wrongness is consequential. The expected value is defined only when the integral of the *absolute* payoff is finite:

$$
E[|X|] = \int |X| \, dP < \infty.
$$

If $E[|X|] = \infty$, we say $X$ is **not integrable**, and $E[X]$ either equals $+\infty$ (or $-\infty$) or is genuinely **undefined**. This is not a technicality. It is a precise warning that the very concept you want to use to make a decision — the average outcome — does not exist as a finite number, so any decision that leans on it is built on sand.

![Before and after contrast of an integrable bounded payoff with finite mean versus a heavy-tailed bet with undefined expectation](/imgs/blogs/lebesgue-integration-math-for-quants-6.png)

The before-and-after figure contrasts the two worlds. On the left, an integrable payoff: a bounded option payoff whose tails decay fast enough that the expectation is finite, so you can size the bet by its expected value. On the right, a non-integrable payoff: a St. Petersburg or Cauchy-style bet whose tails decay too slowly, so the expectation blows up and sizing by EV becomes meaningless. The whole figure is a warning label. Let us earn it with the two classic examples.

#### Worked example: the St. Petersburg paradox, a bet with infinite expected value

The **St. Petersburg game**, posed in 1713, works like this. A fair coin is flipped until it first lands heads. If the first heads is on flip $k$, you win \$$2^k$. So heads on flip 1 pays \$2, heads on flip 2 pays \$4, flip 3 pays \$8, and so on, doubling each time. The probability of first-heads-on-flip-$k$ is $(1/2)^k$. What is the expected payout?

$$
E[X] = \sum_{k=1}^{\infty} 2^k \times \left(\frac{1}{2}\right)^k = \sum_{k=1}^{\infty} 1 = 1 + 1 + 1 + \dots = \infty.
$$

Every term contributes exactly \$1 to the expectation, and there are infinitely many terms, so the expected payout is *infinite*. By naive expected-value logic, you should be willing to pay *any* finite price — \$1,000, \$1,000,000, your house — to play this game once, because the EV exceeds any price. But nobody would. The reason is that the integral defining $E[X]$ diverges: the payoff's tail (huge payouts) shrinks in probability exactly as slowly as the payouts grow, so the value never gets discounted away, and no finite "average" exists. The intuition this teaches: when the integral $\int X\,dP$ diverges, "expected value" is literally infinite and cannot guide a finite decision — the EV criterion has nothing to say, and you must size by something else entirely.

What *do* you size by when EV fails? In practice, you size by *utility* (the value of an extra dollar shrinks as you get richer, which tames the doubling) or by *growth rate* — which is exactly the domain of the [Kelly criterion](/blog/trading/quantitative-finance/kelly-criterion-sequential-betting-quant-interviews), the framework for sizing sequential bets by the geometric growth they produce rather than by a possibly-infinite arithmetic average. St. Petersburg is the canonical demonstration that arithmetic EV and good sizing are *different things*, and Kelly is the repair.

#### Worked example: a Cauchy-distributed PnL with no expected value at all

St. Petersburg has an *infinite* expectation. The **Cauchy distribution** is worse: its expectation is *undefined* — not infinite, but genuinely without a value, because the positive and negative tails are both infinite and there is no consistent way to subtract them.

The Cauchy distribution shows up in finance more often than you would like: it is what you get when you take the *ratio* of two independent normal quantities (and ratios are everywhere — relative value trades, hedge ratios estimated by dividing one noisy number by another). Its density is $f(x) = \frac{1}{\pi(1+x^2)}$, which looks like a bell curve but has tails so heavy that the integral $E[|X|] = \int |x| f(x)\,dx$ diverges. Concretely, suppose a strategy's daily PnL (in thousands of dollars) followed a Cauchy distribution. You might run it for 250 days and compute a sample average daily PnL of, say, \$1,200. Encouraged, you scale up. Another 250 days and the running average is now \$4,800 — then a single catastrophic day drags it to −\$15,000 — then it climbs back. The sample average *never converges*. It is not noisy-around-a-true-value; there *is no* true value for it to converge to, because $E[X]$ does not exist.

This is the most dangerous case precisely because the sample average *looks* like it is telling you something — it produces a number every day. But that number is a phantom. The intuition this teaches: a Cauchy-like PnL produces sample averages that wander forever without settling, so a backtest's "average daily profit" can be a meaningless artifact, and sizing a live book by it is sizing by a number that mathematics says does not exist.

### How to tell, in practice, whether your EV exists

You will not run a Lebesgue integral to check integrability. The practical heuristics are: (1) Is the payoff *bounded*? Bounded payoffs always have finite expectation (a constant dominates them), so options, spreads, and digitals are safe. (2) If unbounded, how fast does the tail decay? If the probability of a loss of size $L$ shrinks faster than $1/L^2$, the mean and variance both exist; if it shrinks like $1/L^2$ the mean exists but variance does not (this is the boundary where [extreme value theory and tail risk](/blog/trading/math-for-quants/tail-risk-extreme-value-theory-math-for-quants) take over); if it shrinks only like $1/L$ (Cauchy-like), even the mean is gone. (3) When in doubt, plot the *running* sample mean of your historical PnL: if it keeps wandering and never flattens out as the sample grows, suspect a non-integrable tail and stop trusting the average.

## 7. Fubini's theorem: swapping the order of integration

The last piece of machinery you will actually use is **Fubini's theorem**, which governs *double* integrals — the situation where you average over two sources of uncertainty at once, or over time *and* states.

The everyday version of the question: if you want the total of a grid of numbers, can you add up the rows first and then the columns, or the columns first and then the rows? For a finite grid, obviously yes — the total is the total. Fubini's theorem says the same freedom holds for *integrals* (continuous, infinite "grids"), **provided** the integral of the absolute values is finite. Formally, for a payoff $h(x, y)$ depending on two variables:

$$
\int\!\!\int h(x,y)\, dx\, dy = \int\!\Big(\int h(x,y)\, dy\Big) dx = \int\!\Big(\int h(x,y)\, dx\Big) dy,
$$

as long as $\int\!\int |h(x,y)|\, dx\, dy < \infty$. Here $h$ is the two-variable payoff, and the theorem says the order of integration does not matter — integrate over $y$ first or $x$ first, you get the same total — *but only when the absolute integral is finite*. That proviso is Fubini's twin, sometimes called Tonelli's theorem, and the finiteness condition is the same integrability we just spent a section on.

Where does this bite in finance? Three places. First, **multi-asset expectations**: the expected payoff of a basket option depends on the joint distribution of several assets, and computing it means integrating over all of them; Fubini lets you do the integrals one asset at a time in whatever order is most convenient. Second, **time and state**: the value of a stream of cash flows is an integral over *time* of an expectation over *states*, and Fubini lets you swap "discount then average" with "average then discount" — the interchange that makes [expected-value-of-a-cash-flow-stream](/blog/trading/math-for-quants/expectation-variance-moments-math-for-quants) calculations tractable. Third, it underlies the formula that the expectation of a sum is the sum of expectations even for infinitely many terms — which is exactly the cash-flow interchange a fixed-income desk relies on.

#### Worked example: a two-asset expected payoff, computed two ways

You hold a structure that pays the *sum* of the gains on two stocks, A and B, capped so the payoff is bounded (so we are safely integrable and Fubini applies). To keep the arithmetic clean, model it discretely. Stock A gains \$10 with probability 0.5 and \$0 with probability 0.5. Stock B gains \$20 with probability 0.5 and \$0 with probability 0.5. They move independently. The payoff is $X = (\text{gain on A}) + (\text{gain on B})$, and we want $E[X]$.

**Order 1 — sum over A's outcomes first, then B.** Build the full grid of four equally likely (probability 0.25 each) joint outcomes and their payoffs: (A=10, B=20) → \$30; (A=10, B=0) → \$10; (A=0, B=20) → \$20; (A=0, B=0) → \$0. Summing rows (fixing A, summing over B) then columns:

$$
E[X] = 0.25(30) + 0.25(10) + 0.25(20) + 0.25(0) = 7.5 + 2.5 + 5.0 + 0 = \$15.
$$

**Order 2 — use linearity (sum the marginals).** Fubini guarantees we can integrate one asset at a time: $E[X] = E[\text{gain A}] + E[\text{gain B}]$. Now $E[\text{gain A}] = 10 \times 0.5 = \$5$ and $E[\text{gain B}] = 20 \times 0.5 = \$10$, so $E[X] = 5 + 10 = \$15$.

Both routes give \$15. That agreement is *exactly* Fubini's theorem in action: because the payoff is bounded (hence integrable), the order in which you integrate the two assets does not change the answer, so you are free to use the easy route (sum the marginals) instead of building the whole joint grid. The intuition this teaches: Fubini is the permission to compute a multi-asset or time-and-state expectation in whatever order is easiest, and it is safe precisely as long as the payoff is integrable.

### When Fubini fails

The cautionary footnote: if the integrability condition is violated — if $\int\!\int |h| = \infty$ — swapping the order can give *different answers*, and there are textbook examples where one order gives $+1$ and the other gives $-1$. In finance this is the same disease as the Cauchy PnL: heavy tails break integrability, and once integrability is gone, the convenient interchanges you have been taking for granted stop being valid. Every "swap the order" and "push the expectation inside the sum" you do silently assumes you are in the integrable world.

## Common misconceptions

**"Lebesgue integration gives different answers than Riemann."** No — whenever the Riemann integral exists, the Lebesgue integral exists and equals it. Lebesgue does not overrule Riemann; it *extends* Riemann to functions and situations Riemann cannot reach (pathological functions like Dirichlet's, mixed distributions with point masses, and limits of sequences). For every payoff you actually trade, the two agree to the penny. The difference is not the *answer*; it is the *range of questions the framework can answer*.

**"Expected value always exists."** It does not. $E[X]$ is defined only when $E[|X|] < \infty$. St. Petersburg has an infinite expectation; a Cauchy-distributed quantity has *no* expectation at all. Treating "the average" as automatically available is the error that lets traders size positions by a number that mathematically does not exist. Always ask whether the integral converges before you lean on it.

**"A point mass doesn't matter because a single point has zero width."** That reasoning is true for a *density* (a continuous spread of probability), where a single point genuinely contributes nothing. But a *point mass* is a lump of finite probability sitting at one value, and it contributes value-times-probability just like any discrete outcome. The Lebesgue integral keeps the lump; a careless density-only calculation drops it. This is precisely why options that pin at a strike need the mixed-distribution treatment.

**"Measure theory is just abstract decoration with no practical payoff."** The practical payoff is concrete: it is the guarantee that your Monte-Carlo average converges (dominated convergence), the guarantee that you can swap the order of a multi-asset integral (Fubini), and the precise diagnostic for when "expected value" is a mirage (integrability). You will not compute Lebesgue integrals by hand, but you rely on what they certify every time you run a simulation or sum a cash-flow stream.

**"If the sample average produces a number, that number means something."** A sample average always produces *a* number — even from a Cauchy distribution with no true mean. The number being computable does not mean it is converging to anything. The honest check is to plot the *running* mean as the sample grows: if it keeps wandering rather than flattening, the underlying expectation may not exist, and the average is a phantom you must not size by.

**"The Law of Large Numbers always saves you, so just run more simulations."** The Law of Large Numbers requires a *finite* mean (and good convergence requires controlled tails). For a payoff with infinite or undefined expectation, more simulations do not help — the running average lurches with every rare giant draw and never settles. Throwing compute at a non-integrable payoff is throwing compute at a number that does not exist.

## How it shows up in real markets

### 1. Pricing a digital option that pins at the strike

On a quarterly options-expiration Friday ("quad witching"), a heavily traded single-stock name often **pins** to a round strike — the price gets pulled toward, say, \$50 by the hedging flows of dealers who are long or short large amounts of options struck there. A digital option struck at \$50 then has a meaningful *point mass* of probability sitting exactly at the strike on the expiry. A desk that prices the digital by a smooth model density alone will misprice the lump and misjudge its exposure at the most dangerous moment — right at the strike, right at expiry, where the payoff is discontinuous and the gamma is enormous. The Lebesgue framing — split the distribution into its continuous part and its point mass and integrate each — is the bookkeeping that keeps the pinned probability on the books. The lesson: discontinuities in a payoff plus probability piled up at the discontinuity is exactly the mixed-distribution case, and it is a real, recurring market event, not a textbook curiosity.

### 2. Monte-Carlo pricing of a path-dependent exotic

A bank prices an Asian option (whose payoff depends on the *average* price of an underlying over a window) or a barrier option by simulation, because no closed-form integral exists. The whole method rests on the sample average of simulated payoffs converging to the true expected payoff. For these payoffs — bounded, or at least with well-controlled tails — dominated convergence guarantees the convergence, and the desk reports a price with a confidence interval that shrinks like $1/\sqrt{n}$. A quant who *cannot* state why the average converges is flying blind about when it would *not* (a payoff with an unbounded, heavy tail). The lesson: every production Monte-Carlo pricer is an applied dominated-convergence theorem, and the theorem's condition — an integrable cap on the payoff — is exactly the assumption that fails for the payoffs that blow up.

### 3. The 1998 LTCM unwind and "averages" that didn't exist

Long-Term Capital Management ran convergence trades whose PnL came from *ratios* and *spreads* of prices — the kind of quantities whose distributions have heavy tails. In calm periods the historical "average daily PnL" looked steady and positive, and the fund sized aggressively (leverage exceeding 25-to-1 at points in 1998). When Russia defaulted in August 1998, the tails that the steady-looking average had quietly ignored arrived all at once, and the fund lost roughly \$4.6 billion in months. The deep lesson, in the language of this post: a sample average computed from a heavy-tailed, possibly non-integrable PnL is not a reliable estimate of a stable expected value, and sizing by it is sizing by a number that may not exist. The running average looked informative right up until it wasn't.

### 4. Why "expected value" of a lottery or a moonshot is the wrong sizing tool

A venture portfolio, a deep out-of-the-money options book, or a crypto moonshot bet often has a payoff distribution dominated by rare, enormous outcomes — the St. Petersburg shape in spirit. The *arithmetic* expected value can be huge (or, in the pure limit, infinite), tempting an investor to bet large. But a single-shot arithmetic EV ignores that you only get to compound your *actual* wealth, and a string of losses before the big win can ruin you first. This is why disciplined allocators size such bets by *growth rate* (Kelly) or by *utility*, not by raw EV — exactly the repair St. Petersburg demands. The lesson: when the payoff's tail makes the arithmetic mean unreliable or infinite, EV is the wrong objective, and the [Kelly framework](/blog/trading/quantitative-finance/kelly-criterion-sequential-betting-quant-interviews) for sequential betting is the right one.

### 5. Discounted cash flows and the interchange of sum and expectation

A fixed-income desk values a stream of uncertain future cash flows as the sum of each cash flow's discounted expected value. Writing "the expected value of the sum equals the sum of the expected values" — and pushing the discounting inside — is an interchange of summation/integration and expectation that is licensed by Fubini and the monotone convergence theorem, *provided* the cash flows are integrable. For ordinary bonds and swaps with bounded, well-behaved cash flows, the interchange is safe and nobody thinks twice. For exotic structures with heavy-tailed or unbounded payoffs, the interchange can fail, and a valuation that quietly assumed it can be materially wrong. The lesson: the everyday "sum the discounted expected cash flows" formula is a Fubini/MCT interchange in disguise, and it is valid exactly in the integrable world.

### 6. Fat-tailed risk models and the variance that wasn't there

Many risk systems assume returns have a finite variance so they can use volatility and Gaussian-style value-at-risk. But some return series — especially in stressed regimes or for certain illiquid instruments — have tails so heavy that the *second* moment (variance) integral diverges, and a few have tails heavy enough that even the *mean* is suspect. When the variance integral does not converge, the "volatility" your system reports is an estimate of a quantity that does not exist; it will jump around wildly as the sample grows and lull you into false precision. The lesson: integrability is not abstract — whether the mean and variance integrals converge determines whether your headline risk numbers are estimating something real or estimating a mirage, which is the entire motivation for the heavier machinery in [tail-risk and extreme value theory](/blog/trading/math-for-quants/tail-risk-extreme-value-theory-math-for-quants).

## When this matters to you

If you are learning quantitative finance, here is the honest accounting of where this topic touches your actual work. You will almost never compute a Lebesgue integral by hand — but you will constantly *rely on the three things it certifies*, and knowing the names and conditions changes how you build and trust models.

First, it gives you **one definition of expected value** that works for every payoff you will ever price — discrete bets, continuous returns, and the awkward mixed payoffs (options pinned at a strike, barriers, rebates) that are exactly where the money and the risk concentrate. When you meet a payoff with a lump of probability at a single value, you will know not to drop it.

Second, it gives you the **license to simulate**. Every Monte-Carlo price you run is an applied dominated-convergence theorem; understanding the condition (an integrable cap on the payoff) tells you precisely when the average is guaranteed to converge and when — for an unbounded, heavy-tailed payoff — it is not, and you must stop trusting the number.

Third, and most valuable for protecting capital, it gives you a **precise diagnostic for when "expected value" is a lie**. St. Petersburg (infinite EV) and Cauchy (undefined EV) are not just curiosities; they are the mathematical shape of every fat-tailed strategy whose backtested "average profit" is a phantom. The reflex this should install: before you size anything by its expected value, ask whether that expected value actually exists — and if the running sample mean keeps wandering, assume it does not.

This is educational material, not financial advice; nothing here is a recommendation to trade any instrument. The point is mechanical understanding, not a strategy.

**Further reading.** To go deeper on the pieces this post leaned on: the [expectation, variance, and higher moments](/blog/trading/math-for-quants/expectation-variance-moments-math-for-quants) post in this series builds out the four moments and where the integrals that define them converge or fail; the [Monte-Carlo simulation walkthrough](/blog/trading/quantitative-finance/monte-carlo-simulation-coding-quant-interviews) is the hands-on companion to section 5's convergence argument; the [Kelly criterion for sequential betting](/blog/trading/quantitative-finance/kelly-criterion-sequential-betting-quant-interviews) is the right way to size the bets whose arithmetic expected value (St. Petersburg, section 6) is infinite or undefined; and the [tail-risk and extreme value theory](/blog/trading/math-for-quants/tail-risk-extreme-value-theory-math-for-quants) post takes up the story exactly where integrability breaks down and the mean or variance integral stops converging. For the primary source, Henri Lebesgue's 1902 thesis founded the theory; any graduate measure-theory text (Royden, or Williams's *Probability with Martingales* for the probabilist's angle) carries the proofs this post deliberately left on the shelf.
