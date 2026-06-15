---
title: "Probability spaces and random variables, intuitively"
date: "2026-06-15"
description: "A beginner-friendly, build-from-zero tour of the sample space, events, the probability measure, sigma-algebras as information, filtrations, random variables, and distributions, all grounded in concrete trading dollars."
tags: ["probability-space", "random-variables", "sigma-algebra", "filtration", "distributions", "measure-theory", "quant-finance", "expected-value", "math-for-quants"]
category: "trading"
subcategory: "Quantitative Finance"
author: "Hiep Tran"
featured: true
readTime: 44
---

> [!important]
> **TL;DR** — A probability space is the formal stage on which every market model is acted out, and a random variable is the rule that turns an uncertain outcome into a profit-and-loss number you can actually reason about.
>
> - The triple $(\Omega, \mathcal{F}, P)$ is the whole game: $\Omega$ is every outcome that could happen, $\mathcal{F}$ is the collection of questions you are allowed to ask, and $P$ assigns each question a probability between 0 and 1.
> - A **random variable** is a function from outcomes to numbers; it is how "the market went up" becomes "you made \$500."
> - The $\sigma$-algebra $\mathcal{F}$ is really *information*, and when it grows over time it becomes a **filtration** $\mathcal{F}_t$ — the formal "what we know at time $t$," which is exactly why a backtest must never peek at the future.
> - A **distribution** is the bookkeeping of how probability is spread across a random variable's values; discrete distributions count, continuous ones integrate, and both let you price a bet.
> - The one fact to remember: a trade that pays $+\$500$ with probability $0.55$ and loses $\$450$ with probability $0.45$ has an expected value of $+\$72.50$ — and that single number is the entire reason the trade exists.

Here is a question that sounds like a riddle but is actually the foundation of an entire industry: before a trade happens, in what sense does it have a value at all?

The trade has not resolved. You do not yet know whether you will make money or lose it. And yet a quant will tell you, with a straight face and a precise number, that the trade is "worth" exactly \$72.50 to you right now. How can something uncertain have an exact value? The answer is the machinery of this post. Probability is the mathematics of reasoning carefully about things that have not happened yet, and the single most important move it makes is to separate three ideas that beginners always tangle together: the set of things that *could* happen, the *questions* you can ask about what happened, and the *weight* you assign to each answer. Get those three apart and kept apart, and almost everything else in quantitative finance — pricing, risk, backtesting, portfolio construction — becomes a story about numbers attached to outcomes. Tangle them, and you will spend years confused. This post pulls them apart slowly, from absolute zero, and ties every piece to a dollar.

![Pipeline from a raw market outcome through the sample space and a random variable to a dollar P and L number](/imgs/blogs/probability-spaces-random-variables-math-for-quants-1.png)

The diagram above is the mental model for the whole post: out in the world, something happens — a coin lands, a stock ticks, a trade resolves. That raw happening is just one of many things that *could* have happened. A random variable is the rule that reads that raw outcome and reports back a single number, usually a profit or loss. Everything we build below is either describing the left side (what can happen and how likely it is) or describing that arrow in the middle (the rule that turns happenings into numbers). Let us start at the very beginning.

## Foundations: the building blocks of chance

Before we can talk about probability spaces, we need to agree on what a handful of plain words mean. We will define each one the first time it appears, build the simplest possible version of every idea, and only then climb toward the real machinery. If you already know what a sample space is, you can skim; if you do not, you can still follow every step.

### What is an "outcome"?

An **outcome** is one complete, fully-specified way the world could turn out for the thing you are studying. The key words are *complete* and *one*. If you flip a single coin, the outcomes are "heads" and "tails" — there are exactly two, and each one tells you everything there is to know about that flip. If you place a single trade that can only win or lose, the outcomes are "win" and "lose."

The important discipline, which trips up beginners constantly, is that an outcome must be specified at the right level of detail for your question. If you are studying *two* trades, then a single outcome is not "the first trade won" — that is incomplete, because it says nothing about the second trade. A complete outcome for two trades is something like "the first won and the second lost." We will write that as $WL$, reading left to right.

### What is the "sample space"?

The **sample space**, written with the Greek capital letter $\Omega$ (omega), is simply the set of *all* possible outcomes, with nothing left out and nothing listed twice. It is the complete menu of ways the world could turn out.

For one coin: $\Omega = \{H, T\}$. For one trade that can win or lose: $\Omega = \{\text{win}, \text{lose}\}$. For two trades: $\Omega = \{WW, WL, LW, LL\}$, four outcomes. The sample space is the single most important object to get right, because every later object — events, probabilities, random variables — is built on top of it. If your sample space is wrong (too small, overlapping, or at the wrong level of detail), everything downstream is wrong too.

A useful sanity check: the outcomes in $\Omega$ must be **mutually exclusive** (no two can happen at once) and **collectively exhaustive** (together they cover every possibility). Exactly one outcome from $\Omega$ happens, every time. The everyday analogy is the faces of a die: a roll lands on exactly one face, never two, and never a face that is not there.

### What is an "event"?

Here is the first move that feels like a genuine idea rather than a definition. An **event** is any *collection of outcomes* — any subset of $\Omega$. The reason this matters is that in real life we almost never care about a single fully-detailed outcome; we care about *groups* of them that share some feature we can name.

Take two trades, $\Omega = \{WW, WL, LW, LL\}$. The statement "at least one trade won" is not a single outcome — it is the event $\{WW, WL, LW\}$, a bundle of three outcomes. The statement "the day was profitable" might be the event $\{WW, WL, LW\}$ as well, or a different bundle, depending on the dollar sizes. An event is how we translate a plain-English question ("did at least one trade win?") into a precise mathematical object (a specific subset of $\Omega$).

Two events deserve names right away. The **certain event** is all of $\Omega$ — "something happens," which always does. The **impossible event** is the empty set $\varnothing$ — "nothing in $\Omega$ happens," which never does. Everything else lives in between.

### What is a "probability"?

A **probability** is a number between 0 and 1 that we attach to an event to say how much we believe it, or how often it happens in the long run. Probability 0 means "essentially never," probability 1 means "essentially always," and 0.5 means "as likely as not." We write $P(A)$ for "the probability of event $A$."

There are two honest ways to read that number, and a working quant uses both. The **frequentist** reading: if you could repeat the situation a million times, $P(A)$ is the fraction of times $A$ happens. The **subjective** (or Bayesian) reading: $P(A)$ is the price you would consider fair for a bet that pays \$1 if $A$ happens and \$0 otherwise. For a fair coin, $P(\text{heads}) = 0.5$ under both readings. For "this specific stock closes above \$105 today," only the subjective reading really makes sense, because the day happens once — but the math is identical either way, which is the whole point of building it carefully.

With those four words — outcome, sample space, event, probability — we have the raw materials. Now we assemble them into the object every quant model is secretly built on.

## The probability triple $(\Omega, \mathcal{F}, P)$

Mathematicians package the three pieces above into a single object called a **probability space**, written as a triple $(\Omega, \mathcal{F}, P)$. It is worth slowing down here, because this triple is the formal stage on which every market model in finance is performed, and understanding its three layers separately is what separates someone who can use probability from someone who merely memorized formulas.

![Stack of the three layers Omega, the event collection F, and the probability measure P](/imgs/blogs/probability-spaces-random-variables-math-for-quants-2.png)

The figure stacks the three layers in the order you should think about them. At the bottom is $\Omega$, the raw set of outcomes. In the middle is $\mathcal{F}$, the collection of events — the questions you are allowed to ask. At the top is $P$, the rule that assigns a probability to each question. Think of it as a measuring instrument: $\Omega$ is the territory, $\mathcal{F}$ is the set of regions you have drawn boundaries around, and $P$ is the device that tells you the "size" (the probability) of each region. Let us define each layer precisely.

### $\Omega$: the set of all outcomes

We already have this one. $\Omega$ is the complete, mutually-exclusive, collectively-exhaustive list of everything that could happen. In finance, $\Omega$ might be tiny ($\{$win, lose$\}$ for one binary trade) or unimaginably vast (every possible path the entire market could take over the next year, second by second). The size does not change the structure; it only changes how we describe $P$.

### $\mathcal{F}$: the collection of events (the $\sigma$-algebra)

Now the layer that beginners skip and practitioners obsess over. $\mathcal{F}$ (script F) is the collection of all events we are allowed to assign probabilities to. When $\Omega$ is small, $\mathcal{F}$ is usually *every* possible subset — for $\Omega = \{H, T\}$, the events are $\varnothing$, $\{H\}$, $\{T\}$, and $\{H, T\}$, all four. When $\Omega$ is large or continuous, we cannot consistently assign a probability to literally every subset, so $\mathcal{F}$ is a carefully-chosen sub-collection. Either way, $\mathcal{F}$ is called a **$\sigma$-algebra** (sigma-algebra), and it must obey three rules so that probability behaves sensibly:

1. The whole space is included: $\Omega \in \mathcal{F}$. (You can always ask "did *something* happen?")
2. It is closed under complement: if $A \in \mathcal{F}$, then "not $A$" (written $A^c$) is also in $\mathcal{F}$. (If you can ask "did it rain?" you can ask "did it not rain?")
3. It is closed under countable unions: if $A_1, A_2, \dots$ are all in $\mathcal{F}$, then $A_1 \cup A_2 \cup \dots$ is in $\mathcal{F}$. (If you can ask about each of many events, you can ask "did at least one of them happen?")

Those three rules are not arbitrary bookkeeping. They are exactly the closure properties you need so that the everyday logical operations — *and*, *or*, *not* — never take you outside the set of questions you can answer. A $\sigma$-algebra is, put simply, a self-consistent universe of questions. We will return to this layer in a moment with the single most important reinterpretation in the whole post: $\mathcal{F}$ is *information*. Hold that thought.

### $P$: the probability measure

Finally, $P$ is a **measure** — a function that takes any event in $\mathcal{F}$ and returns a number in $[0, 1]$. It must obey two axioms, due to Kolmogorov, that pin down everything probability does:

1. $P(\Omega) = 1$. The certain event has probability 1; *something* happens with certainty.
2. For events that cannot overlap (disjoint events $A$ and $B$, meaning $A \cap B = \varnothing$), probabilities add: $P(A \cup B) = P(A) + P(B)$. (More precisely, this holds for countably many disjoint events at once.)

From just these two rules, every familiar fact about probability follows. The probability of "not $A$" is $P(A^c) = 1 - P(A)$, because $A$ and $A^c$ are disjoint and together make all of $\Omega$. The impossible event has $P(\varnothing) = 0$. Probabilities never exceed 1. The general addition rule for possibly-overlapping events, $P(A \cup B) = P(A) + P(B) - P(A \cap B)$, drops out with one line of algebra. The whole edifice rests on two short axioms, which is part of what makes probability so powerful: a tiny set of rules, applied relentlessly.

#### Worked example: a single trade as a random variable

Let us make the triple concrete with the example from the TL;DR, because it is the smallest model that already contains everything important. You have a single trading strategy. When it fires, history tells you it makes \$500 fifty-five percent of the time and loses \$450 the other forty-five percent of the time. We will build the full probability space and then attach a random variable.

The sample space has two outcomes: $\Omega = \{\text{win}, \text{lose}\}$. The event collection $\mathcal{F}$ is every subset: $\{\varnothing, \{\text{win}\}, \{\text{lose}\}, \Omega\}$. The measure assigns $P(\{\text{win}\}) = 0.55$ and $P(\{\text{lose}\}) = 0.45$, and you can check the axioms: $P(\Omega) = 0.55 + 0.45 = 1$, good, and the two single-outcome events are disjoint so their probabilities add, good.

Now the random variable, which we will call $X$ and which represents your profit in dollars. It is a rule from outcomes to numbers: $X(\text{win}) = +500$ and $X(\text{lose}) = -450$. That is the whole map. The **expected value** of $X$, written $E[X]$, is each value weighted by its probability and summed:

$$ E[X] = 0.55 \times (+500) + 0.45 \times (-450) = 275 - 202.50 = +72.50. $$

So the trade is "worth" $+\$72.50$ before it happens — not because you ever receive exactly \$72.50 (you receive either $+\$500$ or $-\$450$, never \$72.50), but because if you ran this trade thousands of times, your *average* result per trade would converge to \$72.50. **The one-sentence intuition: a random variable plus a probability measure is everything you need to assign a single, decision-grade number to an uncertain outcome.**

That \$72.50 is the engine of the strategy. Trade it once and the outcome is noise. Trade it ten thousand times with proper sizing and the law of large numbers grinds that \$72.50-per-trade edge into a smooth profit. Probability theory is what lets you see the edge before the noise washes it out.

## Random variables: turning outcomes into numbers

We have used the term twice now; let us define it with full care, because it is the bridge between abstract outcomes and the numbers a trader actually cares about.

A **random variable** is a function $X : \Omega \to \mathbb{R}$ that assigns a real number to every outcome. That is the entire definition. It is not "random" in the sense of being unpredictable once the outcome is known — given the outcome, $X$ is completely determined. It is random *before* the outcome is revealed, in the sense that you do not yet know which outcome will occur and therefore do not yet know which number $X$ will report.

The everyday analogy: think of $X$ as a scorekeeper standing at the end of a game. The game can end many ways (those are the outcomes in $\Omega$). The scorekeeper has a fixed rulebook that converts however the game ended into a single number on the board. The scorekeeper is not random; the game's ending is. The random variable *is* the rulebook.

![Pipeline from a raw market outcome through the sample space and a random variable to a dollar P and L number](/imgs/blogs/probability-spaces-random-variables-math-for-quants-1.png)

The figure above (the same mental-model pipeline) is worth a second look now that the word means something. The arrow labeled "apply rule" is the random variable. Notice what it accomplishes: it takes the messy, qualitative world of outcomes ("the trade won," "the stock gapped down on bad earnings") and compresses it into the one dimension a portfolio cares about — a number, almost always a dollar amount of profit or loss. This compression is the reason random variables are everywhere in finance. A price is a random variable. A return is a random variable. A daily P&L is a random variable. A portfolio's value is a random variable. The future value of a derivative is a random variable. Each is a rule that reads the state of the world and reports a number.

### The technical word "measurable," and why you can mostly ignore it

If you read a textbook, you will see a random variable defined not just as any function $\Omega \to \mathbb{R}$, but as a **measurable** function. Here is what that word means, in plain terms, and why it matters less than it looks.

For $X$ to be useful, you need to be able to ask probability questions about it — things like "what is the probability that $X$ is at most \$100?" That question only makes sense if the set of outcomes where $X \le 100$ is actually an event in $\mathcal{F}$, so that $P$ can measure it. "Measurable" is the technical guarantee that for every threshold, the set $\{\omega : X(\omega) \le x\}$ is in $\mathcal{F}$. When $\Omega$ is finite and $\mathcal{F}$ is all subsets (as in every example we will run by hand), *every* function is automatically measurable, so you never have to check. The condition only bites in continuous, infinite settings, and even there the functions you actually write down — prices, returns, payoffs — are always measurable. The way this works under the hood is that measurability is exactly the bridge that connects the random variable to the $\sigma$-algebra, which we will see is the same thing as connecting it to *information*.

![Stack of the three layers Omega, the event collection F, and the probability measure P](/imgs/blogs/probability-spaces-random-variables-math-for-quants-2.png)

The triple from before (shown again as a reminder) is where measurability lives: the random variable on top must be compatible with the event collection $\mathcal{F}$ in the middle, so that probabilities from $P$ can flow up to questions about $X$. For our hand examples, that compatibility is free. For the deep theory of continuous-time finance, it becomes the load-bearing assumption — but that is a sibling post.

## Events are subsets of $\Omega$

We defined events as subsets of $\Omega$ in the foundations; now let us see them in action, because events are how plain-English market questions become computable.

![Graph of an event drawn as a subset of outcomes inside the sample space Omega](/imgs/blogs/probability-spaces-random-variables-math-for-quants-3.png)

Picture the sample space as a region and an event as a circle drawn inside it, enclosing some of the outcomes and leaving others out. The figure shows exactly this for the two-trade example: the event "the day is profitable" is a circle enclosing the outcomes $WW$, $WL$, and $LW$, while $LL$ sits outside it. The probability of the event is just the total probability weight of the outcomes inside the circle. This picture — an event as a region, its probability as the area — is the single most useful image to carry, because it makes the addition axiom obvious: two non-overlapping circles have a combined area equal to the sum of their separate areas.

Events combine using set operations that map directly onto logic:

| Plain English | Set operation | Probability |
| --- | --- | --- |
| $A$ **and** $B$ both happen | intersection $A \cap B$ | $P(A \cap B)$ |
| $A$ **or** $B$ (or both) happen | union $A \cup B$ | $P(A) + P(B) - P(A \cap B)$ |
| $A$ does **not** happen | complement $A^c$ | $1 - P(A)$ |
| $A$ but **not** $B$ | difference $A \setminus B$ | $P(A) - P(A \cap B)$ |

That table is the entire grammar of combining events. Notice the subtraction in the *or* row: if you just added $P(A) + P(B)$ you would double-count the region where both happen, so you subtract it back out once. This is the **inclusion-exclusion principle**, and forgetting it is one of the most common probability errors, in interviews and in production risk code alike.

### Independence: when knowing one event tells you nothing about another

Two events $A$ and $B$ are **independent** when knowing that one happened does not change the probability of the other. Formally:

$$ P(A \cap B) = P(A) \times P(B). $$

The intuition: independence is the mathematical statement that two things have nothing to do with each other. Two separate fair coin flips are independent — the first landing heads tells you nothing about the second. The multiplication rule is why, for two independent fair coins, "heads then heads" has probability $0.5 \times 0.5 = 0.25$.

Independence is one of the most abused assumptions in all of finance, so it earns a warning even here in the foundations. Quants love to assume returns on different days, or different assets, are independent, because the math gets dramatically easier — variances add, probabilities multiply, the central limit theorem kicks in. But markets are full of hidden dependence. In a crash, assets that looked independent for years suddenly all fall together, because a common cause (a liquidity panic, a rate shock) drives them all at once. The assumption that two events are independent is a *modeling choice*, not a fact of nature, and when it breaks, it breaks at the worst possible time. We will come back to this in the misconceptions and real-markets sections, because it has cost real firms real fortunes.

#### Worked example: the sample space of two trades

Let us build a full sample space from scratch, the second canonical example. You run two trades in a day. To keep the arithmetic clean, suppose each trade independently wins with probability $0.55$ and loses with probability $0.45$ — the same edge as before. The dollar payoffs are deliberately chosen so the totals are easy to read: a win adds \$525, a loss subtracts \$475, but we will track the *combined* day in round numbers.

First, the sample space. With two trades each having two results, there are $2 \times 2 = 4$ complete outcomes:

$$ \Omega = \{WW, WL, LW, LL\}. $$

![Tree branching from the start of the day into four outcomes of two coin-flip trades](/imgs/blogs/probability-spaces-random-variables-math-for-quants-4.png)

The tree above shows how the day branches. From the start, trade 1 either wins (probability 0.55) or loses (0.45). From each of those, trade 2 again wins or loses. Following any path from root to leaf and *multiplying* the probabilities along the way gives that outcome's probability — multiplication is exactly the independence rule in action. The four leaves are the four outcomes, and the dollar labels are the day's total P&L for each, which we now compute.

Because the trades are independent, each outcome's probability is a product:

- $P(WW) = 0.55 \times 0.55 = 0.3025$
- $P(WL) = 0.55 \times 0.45 = 0.2475$
- $P(LW) = 0.45 \times 0.55 = 0.2475$
- $P(LL) = 0.45 \times 0.45 = 0.2025$

Sanity check the measure: $0.3025 + 0.2475 + 0.2475 + 0.2025 = 1.0000$. The axiom holds.

Now define the random variable $S$ = total P&L for the day, using round payoffs of $+\$1{,}000$ for two wins-worth of edge, and so on. To keep the figure and prose in lockstep, let us say each win contributes net and the day totals work out to: $WW \to +\$1{,}000$, $WL \to +\$50$, $LW \to +\$50$, $LL \to -\$900$. Then:

$$ E[S] = 0.3025(1000) + 0.2475(50) + 0.2475(50) + 0.2025(-900). $$

Computing term by term: $0.3025 \times 1000 = 302.50$; $0.2475 \times 50 = 12.375$, twice gives $24.75$; $0.2025 \times (-900) = -182.25$. Summing: $302.50 + 24.75 - 182.25 = +\$145.00$.

So a two-trade day is worth $+\$145.00$ in expectation. **The one-sentence intuition: a sample space plus a measure lets you turn "two coin-flip trades" into a single expected dollar value for the whole day, by enumerating every outcome and weighting it.** Notice also that the most likely *single* outcome ($WW$, at 30.25%) is the big winner, but the day can still lose \$900 about one time in five — the expected value hides the risk, which is why the next sibling concepts (variance, distributions) exist.

## Distributions: how probability is spread across values

So far we have attached probabilities to outcomes one at a time. A **distribution** is the organized, complete summary of how a random variable's probability is spread across all its possible *values*. It is the single most useful object for actually computing anything, because it lets you answer "what is the probability $X$ falls in this range?" without re-deriving the sample space each time.

![Matrix listing each of the four outcomes with its probability and its P and L value](/imgs/blogs/probability-spaces-random-variables-math-for-quants-5.png)

The matrix above is a distribution in its most honest form: every outcome paired with its probability and the dollar value the random variable assigns it. That is literally all a discrete distribution is — a table from values to probabilities. Everything fancier (the cumulative distribution function, the probability density, the named distributions like Normal or Poisson) is a more compact way of writing down the same idea when the table would be too long or infinitely long to list.

### Discrete distributions: the probability mass function

When a random variable can only take a countable list of values — like our P&L, which is one of four amounts — its distribution is described by a **probability mass function**, or **pmf**. The pmf, written $p(x)$, simply gives the probability that the variable equals each specific value:

$$ p(x) = P(X = x). $$

A pmf must satisfy two rules that come straight from the probability axioms: every value is non-negative ($p(x) \ge 0$), and the values sum to 1 ($\sum_x p(x) = 1$). For our two-trade day, the pmf is the matrix above read as $p(1000) = 0.3025$, $p(50) = 0.495$ (the two ways to get \$50 combine), and $p(-900) = 0.2025$. Note how the two outcomes that both pay \$50 get *added* — the distribution cares about values, so it merges outcomes that share a value. This is the first time the distinction between "outcomes" and "values" earns its keep.

The expected value, in pmf language, is the same weighted sum we have been computing:

$$ E[X] = \sum_x x \, p(x). $$

The intuition is unchanged: each value times how often it occurs, summed. The pmf is just the neat container that holds all the values and weights in one place.

### The cumulative distribution function

There is a second, often more convenient, way to describe any distribution — discrete or continuous — called the **cumulative distribution function**, or **cdf**, written $F(x)$. It answers a *threshold* question instead of an *exact-value* question:

$$ F(x) = P(X \le x). $$

In words: $F(x)$ is the probability that the random variable comes out at or below the level $x$. The cdf always starts at 0 far to the left (the variable is surely above $-\infty$), climbs to 1 far to the right (the variable is surely below $+\infty$), and never decreases as you move right, because adding more values to "$\le x$" can only add probability, never remove it. The cdf is the workhorse of risk management: a question like "what is the probability my portfolio loses more than \$1 million tomorrow?" is exactly a cdf question, $P(\text{loss} > 1{,}000{,}000) = 1 - F(1{,}000{,}000)$.

### Continuous distributions: the probability density function

Now the leap that separates a coin-flip bet from a real stock return. Some random variables are **continuous** — they can take any value in a range, not just a discrete list. A stock's return tomorrow could be $+1.0\%$, or $+1.01\%$, or $+1.0137\%$, with infinitely many possibilities packed into any interval.

Here is the subtlety that confuses every beginner, so we will face it head on: for a truly continuous variable, the probability of *any single exact value* is zero. The probability that tomorrow's return is *exactly* $+1.0000\dots\%$, to infinite decimal places, is 0. This is not a trick; it is forced by the math. If you have infinitely many possible values and each had positive probability, the total would exceed 1. So we stop asking "what is the probability of exactly this value?" (always 0) and start asking "what is the probability of landing in this *range*?" (a sensible positive number).

To answer range questions for a continuous variable, we use a **probability density function**, or **pdf**, written $f(x)$. The density is *not* a probability — it is a probability *per unit of $x$*, like a rate. To get an actual probability, you integrate the density over a range, which is the continuous version of summing a pmf:

$$ P(a \le X \le b) = \int_a^b f(x)\, dx. $$

The intuition: the pdf is a curve, and the *area under the curve* between two values is the probability of landing between them. A higher curve means values near there are more likely; the total area under the whole curve is 1, matching the certain event. The cdf and pdf are two views of the same thing: the cdf is the running total of area from the left, so the pdf is the slope of the cdf, $f(x) = F'(x)$. The everyday analogy: the pdf is the *speed* of probability accumulation; the cdf is the *odometer* showing how much probability you have accumulated so far.

| Feature | Discrete (pmf) | Continuous (pdf) |
| --- | --- | --- |
| Models | a coin-flip bet, a binary trade, count of fills | a return, a price, a P&L |
| Basic object | $p(x) = P(X = x)$ | $f(x)$, a density (not a probability) |
| Probability of exact value | can be positive | always 0 |
| Probability of a range | sum: $\sum_{a \le x \le b} p(x)$ | integral: $\int_a^b f(x)\,dx$ |
| Must satisfy | $p(x) \ge 0$, $\sum p = 1$ | $f(x) \ge 0$, $\int f = 1$ |
| Expected value | $\sum_x x\, p(x)$ | $\int x\, f(x)\, dx$ |

#### Worked example: the probability of a price band

Now the third canonical example, our first continuous one. You hold a stock currently trading at \$100. Over the next day, suppose its closing price $X$ is modeled as **uniformly distributed** between \$90 and \$110 — meaning every price in that \$20-wide band is equally likely, and prices outside it cannot happen. (A uniform model is unrealistic for a real stock, which clusters near the center, but it makes the integral arithmetic transparent; we will swap in the realistic Normal afterward.)

For a uniform distribution on $[90, 110]$, the density is flat. Since the total area must equal 1 and the band is 20 wide, the height is $f(x) = 1/20 = 0.05$ for every $x$ between 90 and 110, and 0 outside. Now ask: what is the probability the stock closes between \$95 and \$105?

$$ P(95 \le X \le 105) = \int_{95}^{105} 0.05 \, dx = 0.05 \times (105 - 95) = 0.05 \times 10 = 0.50. $$

So there is a 50% chance the stock lands in the central \$95-to-\$105 band. The arithmetic is just "height times width" — the area of a rectangle — because the density is flat. **The one-sentence intuition: for a continuous variable, the probability of a price band is the area under the density curve over that band, which for a flat density is simply height times width.**

Now the realistic wrinkle. Suppose instead $X$ follows a **Normal distribution** (the familiar bell curve) centered at \$100 with a standard deviation of \$5 — meaning a typical day's move is about \$5, and large moves get exponentially rarer. The band \$95 to \$105 is exactly one standard deviation on each side of the center. A well-known fact about the Normal distribution is that about **68%** of its probability lies within one standard deviation of the mean. So under the realistic model:

$$ P(95 \le X \le 105) \approx 0.68. $$

The Normal model puts *more* probability in the central band (68% versus 50%) precisely because it concentrates outcomes near the center instead of spreading them flat. The lesson is sharp: the *same question* ("will the stock stay within \$5 of \$100?") gets two very different answers — 50% versus 68% — purely because of the distribution we assumed. Choosing the right distribution is not a formality; it is most of the modeling. For the catalog of which distribution fits which situation, see the [distributions cheat sheet](/blog/trading/quantitative-finance/distributions-cheat-sheet-quant-interviews).

## The $\sigma$-algebra is information

We now arrive at the deepest idea in the post, and the one that pays the most direct dividend in real quant work. Back in the foundations, we defined the $\sigma$-algebra $\mathcal{F}$ as "the collection of events you can ask about." That definition is correct but lifeless. Here is the living interpretation: **a $\sigma$-algebra is exactly the information you currently have.**

Think about what it means to "know" something. To know the result of trade 1 is to be able to answer every yes/no question about trade 1: "did it win?", "did it lose?". The set of questions you can answer *is* your knowledge. And a set of questions you can answer, closed under *and*, *or*, and *not*, is precisely a $\sigma$-algebra. The match is not a metaphor; it is an equivalence. Your information at any moment is encodable as the $\sigma$-algebra of events whose truth you can determine.

This reframing turns a dry axiom into the most practically important concept in the post. When we said $\mathcal{F}$ must be closed under complement and union, we were really saying: if you know whether $A$ happened, you know whether *not-$A$* happened; if you know about each of several events, you know about their combination. That is just what having information means.

### Conditional probability: updating on what you know

Once $\mathcal{F}$ is information, the natural next question is: how does learning something change a probability? The answer is **conditional probability**, written $P(A \mid B)$ and read "the probability of $A$ given $B$." It is defined as:

$$ P(A \mid B) = \frac{P(A \cap B)}{P(B)}, \qquad P(B) > 0. $$

The intuition: learning that $B$ happened shrinks your world from all of $\Omega$ down to just the outcomes inside $B$. Within that smaller world, you re-weight: the probability of $A$ becomes the share of $B$'s probability that also has $A$. The numerator $P(A \cap B)$ is "both happened"; the denominator $P(B)$ rescales so the new probabilities inside $B$ sum to 1 again. Conditioning is zooming in and re-normalizing.

Notice the connection to independence. If $A$ and $B$ are independent, then $P(A \cap B) = P(A)P(B)$, so $P(A \mid B) = P(A)P(B)/P(B) = P(A)$ — learning $B$ changes nothing. Independence is exactly the case where information about $B$ is useless for predicting $A$. (Full Bayesian updating — flipping $P(A \mid B)$ into $P(B \mid A)$ — is its own large topic, covered in a sibling post; here we only need the definition.)

#### Worked example: information updates the odds

This is the fourth canonical example, and it ties the $\sigma$-algebra-as-information idea directly to dollars. Return to the two-trade day with outcomes $\{WW, WL, LW, LL\}$ and probabilities $0.3025, 0.2475, 0.2475, 0.2025$, and the P&L map $WW \to +\$1{,}000$, $WL \to +\$50$, $LW \to +\$50$, $LL \to -\$900$. At the start of the day you know nothing; your information $\mathcal{F}_0$ is trivial, and the expected day is the $+\$145.00$ we computed.

![Before-and-after of the four outcomes collapsing to two once the first trade's result is known](/imgs/blogs/probability-spaces-random-variables-math-for-quants-6.png)

Now suppose the first trade settles and it **won**. The figure shows what happens to your world: of the four outcomes, only the two consistent with "trade 1 won" survive — $WW$ and $WL$ — while $LW$ and $LL$ are ruled out entirely. Your information has grown, and growing information collapses the set of possibilities. The before column lists all four as live; the after column keeps $WW$ and $WL$ and crosses out the other two. This collapse is conditioning made visible.

Let us compute the updated expected P&L. The event $B$ = "trade 1 won" has $P(B) = P(WW) + P(WL) = 0.3025 + 0.2475 = 0.55$ (as it must, since trade 1 wins 55% of the time). Conditional on $B$, the two survivors re-weight:

$$ P(WW \mid B) = \frac{0.3025}{0.55} = 0.55, \qquad P(WL \mid B) = \frac{0.2475}{0.55} = 0.45. $$

These are just trade 2's own win/lose probabilities, which makes sense because the trades are independent — knowing trade 1 won tells you nothing about trade 2, only that the day's total is now determined by trade 2 alone. The conditional expected day is:

$$ E[S \mid B] = 0.55 \times 1000 + 0.45 \times 50 = 550 + 22.50 = +\$572.50. $$

The expected day jumped from $+\$145.00$ to $+\$572.50$ the instant you learned trade 1 won. **The one-sentence intuition: information is not free decoration — it literally moves the expected value, and the formal object that tracks "what you know" is the growing $\sigma$-algebra.** Symmetrically, had trade 1 *lost*, the surviving outcomes would be $LW$ and $LL$, and the conditional expectation would be $0.55 \times 50 + 0.45 \times (-900) = 27.50 - 405 = -\$377.50$ — a bad day in the making. Same starting model, opposite expectations, all driven by a single bit of information.

## Filtrations: how information grows over time

We have seen information as a single $\sigma$-algebra. But in markets, information is not static — it *arrives*, tick by tick, trade by trade, day by day. The mathematical object that captures growing information is a **filtration**: a family of $\sigma$-algebras $\mathcal{F}_t$, one for each time $t$, that only ever grows:

$$ \mathcal{F}_s \subseteq \mathcal{F}_t \quad \text{whenever } s \le t. $$

In words: whatever you knew at the earlier time $s$, you still know at the later time $t$, plus possibly more. Information accumulates and is never forgotten. $\mathcal{F}_t$ is the formal, precise answer to the question "what do we know at time $t$?" — and it is the backbone of all of modern continuous-time finance, from the pricing of options to the theory of optimal execution.

![Timeline of a filtration growing through a trading day from knowing nothing at open to knowing everything at close](/imgs/blogs/probability-spaces-random-variables-math-for-quants-7.png)

The timeline above traces a single filtration through one trading day. At the open, $\mathcal{F}_{\text{9:30}}$ is trivial — you know nothing about today's trades yet. When trade 1 settles at 11:00, your information grows to $\mathcal{F}_{\text{11:00}}$, which now contains the answer to "did trade 1 win?". When trade 2 settles at 14:00, it grows again. By the close, $\mathcal{F}_{\text{16:00}}$ contains the full day. Each step only adds; the timeline marches one direction. This monotone growth — information only increases — is the defining property, and it has a consequence that is the difference between a real trading strategy and a fantasy one.

### Why filtrations forbid look-ahead in backtests

Here is the payoff, and it is enormous. A trading strategy is a rule that decides what to do at time $t$. For that strategy to be *implementable in real life*, its decision at time $t$ may depend only on information available at time $t$ — that is, only on $\mathcal{F}_t$. It cannot depend on anything in $\mathcal{F}_{t+1}$ or later, because that information has not arrived yet. In the language of stochastic processes, the strategy must be **adapted** to the filtration: $\mathcal{F}_t$-measurable at each $t$.

When a strategy's decision at time $t$ secretly uses future information, it is said to have **look-ahead bias** (also called the "look-ahead" or "future leak" bug), and it is the single most common and most fatal mistake in quantitative backtesting. The way this works in practice: a researcher computes some signal using a dataset, and somewhere in the pipeline a value from the future sneaks into a calculation that is supposed to be made in the past. Examples that have ruined real backtests:

- Using a stock's *closing* price to decide a trade you claim to make at the *open* of the same day. The close is in $\mathcal{F}_{\text{16:00}}$; your decision lives in $\mathcal{F}_{\text{9:30}}$. You used information you would not have had.
- Normalizing a feature by its mean over the *entire* historical sample, including dates after the trade. The full-sample mean is not in $\mathcal{F}_t$; it depends on the future.
- Using a company's restated, as-of-today financial figures to backtest a decision from five years ago, when only the original, un-restated figures were known.
- Filling a missing data point by interpolating between a past and a *future* observation.

Every one of these is the same crime in disguise: the strategy at time $t$ peeked into a $\sigma$-algebra strictly larger than $\mathcal{F}_t$. The result is a backtest that looks brilliant on paper and loses money in production, because the live strategy genuinely does not have the future data the backtest secretly used. The filtration is the formal definition of "no peeking," and disciplined quants build their entire data pipeline so that every value used in a decision is provably in $\mathcal{F}_t$. This is also where the related concepts of **survivorship bias** (only backtesting on companies that still exist today) and **point-in-time data** (databases that record what was known on each date, not what is known now) come from — they are all the filtration discipline applied to data.

#### Worked example: the cost of one look-ahead bug

Let us put a dollar figure on look-ahead bias, since that is the voice of this series. Suppose you backtest a simple strategy on 1,000 trading days. The honest version, which only uses $\mathcal{F}_t$ at each step, earns an expected $+\$145$ per day, matching our two-trade model, for a total of $1{,}000 \times \$145 = \$145{,}000$ over the sample.

Now suppose a subtle bug lets the strategy peek one day ahead — it learns each day's winning outcome a day early and sizes up when the day will be good. In the backtest, this fantasy version might appear to earn, say, $+\$900$ per day (it loads into the $WW$ outcomes and avoids the $LL$ ones), totaling $\$900{,}000$ — a backtest that looks more than six times better. A junior researcher sees \$900,000 and ships it. In live trading, the future information simply is not there; the strategy reverts to its true edge of \$145 per day at best, and if the position sizing was tuned to the fantasy, it may lose money outright. **The one-sentence intuition: the gap between $\$900{,}000$ and $\$145{,}000$ is the look-ahead bug, and it exists entirely because the backtest used a $\sigma$-algebra larger than $\mathcal{F}_t$ — the filtration is the rule that would have caught it.** This is not a toy worry: entire books of "strategies" at real funds have evaporated when someone finally enforced point-in-time data and the look-ahead edge vanished.

## Common misconceptions

Even readers who can recite the axioms carry beliefs that quietly corrupt their reasoning. Here are the most damaging, each corrected with the *why*.

**"The expected value is what I should expect to get."** No — and the name is genuinely misleading. The expected value of our single trade is $+\$72.50$, but you will *never* receive \$72.50 on any single trade; you get $+\$500$ or $-\$450$. The expected value is the long-run *average* over many repetitions, not a prediction of any one result. Treating it as a forecast for a single event is how people convince themselves a 1-in-100 catastrophe "won't happen to me." It is a center of gravity, not a crystal ball.

**"Probability zero means impossible, and probability one means certain."** For finite sample spaces, yes. But for continuous variables, the probability of any *exact* value is zero even though that value can occur — tomorrow's return will be *some* exact number, yet each exact number had probability zero. "Probability zero" precisely means "negligible in the integral sense," not "cannot happen." Conflating the two leads to nonsense like assuming a continuous model can never produce a specific price.

**"If two assets have been uncorrelated for years, they are independent."** This is the assumption that bankrupts funds. Historical lack of correlation is weak evidence of independence, and independence can vanish exactly when it matters. In normal times, two strategies may look unrelated; in a liquidity crisis, they fall together because a common cause (forced selling) drives both. Independence is a modeling assumption with an expiration date, not a permanent property — and the tail is where it expires.

**"A great backtest means a great strategy."** A backtest is only as honest as its filtration discipline. The most dangerous backtests are the ones that look *too* good, because spectacular results are usually a symptom of look-ahead bias, overfitting, or survivorship bias rather than genuine edge. The right reflex when a backtest looks amazing is suspicion, not celebration: ask which $\sigma$-algebra each decision actually used.

**"The sample space is obvious; I can skip defining it."** The single most common source of wrong probability answers, in interviews and in code, is a misspecified sample space — outcomes that overlap, are listed at the wrong granularity, or are not equally likely when you assumed they were. The famous interview blunders (the Monty Hall problem, the boy-girl paradox) are all sample-space errors. Spending thirty seconds writing out $\Omega$ explicitly prevents most of them. For a tour of these traps, see the [classic quant probability problems](/blog/trading/quantitative-finance/classic-quant-probability-problems).

**"Conditional probability is just a smaller version of the same thing."** Conditioning genuinely changes the measure — it re-normalizes over a shrunken world. People forget the denominator $P(B)$ and report $P(A \cap B)$ when they meant $P(A \mid B)$, which can be off by a large factor. The conditional probability of a profitable day given trade 1 won is *not* the same as the probability of both — the first is 0.55, the joint pieces are smaller. Always divide.

**"More data always means a better probability estimate."** Data only helps if it is the *right* data, used at the *right* time. A million rows of returns measured at the wrong granularity, or contaminated by survivorship and look-ahead, produce a confidently wrong estimate — and confidence with the wrong sign is worse than admitted ignorance. The discipline is not "collect more"; it is "be ruthless about which $\sigma$-algebra each number belongs to." A small, clean, point-in-time dataset beats a vast, leaky one every time, because the leaky one estimates a probability under a measure that does not exist in live trading.

**"The probability density is a probability, so it must be at most 1."** A density $f(x)$ is a probability *per unit*, not a probability, and it can be far larger than 1. A Normal distribution with a tiny standard deviation has a tall, narrow peak whose height exceeds 1 — yet the total *area* under it is still exactly 1. Only areas (integrals) are probabilities, and only areas are capped at 1. Confusing the height of the curve with a probability is one of the most common slips when people first meet continuous distributions, and it leads to nonsense like "this density of 2.5 is impossible."

## How it shows up in real markets

The abstract triple $(\Omega, \mathcal{F}, P)$ is not a classroom toy; it is the load-bearing structure under named, dated, money-moving episodes. Here are concrete scenarios where each piece of this post is the protagonist.

**Option pricing and the risk-neutral measure.** The entire edifice of derivatives pricing rests on the idea that there can be *two different probability measures on the same sample space*. The real-world measure $P$ describes how prices actually move; the **risk-neutral measure** $Q$ is a re-weighted measure under which the fair price of any derivative is just its discounted expected payoff. Black-Scholes (1973) is, at heart, the statement that under $Q$, the expected discounted payoff of an option is computable in closed form. The fact that you can keep the same $\Omega$ and same $\mathcal{F}$ but swap $P$ for $Q$ — the change of measure — is the single most powerful trick in quantitative finance, and it is only coherent because we built the triple with the measure as a separate, swappable layer. A \$700-trillion-notional derivatives market runs on this distinction.

**The 1998 collapse of Long-Term Capital Management.** LTCM, run by Nobel laureates, built positions assuming various spreads were nearly independent and that extreme co-movements were astronomically unlikely under their distributional model. When Russia defaulted in August 1998, the independence assumption shattered: positions that the model treated as separate bets all moved against the fund at once, because a single common cause (a global flight to quality) drove them together. The fund lost roughly \$4.6 billion in months and required a Fed-organized bailout. The post-mortem lesson is pure probability theory: their probability *measure* assigned far too little weight to the joint tail event, and their independence assumption was a modeling choice that expired catastrophically.

**Look-ahead bias in published "anomalies."** A large fraction of academic trading anomalies fail to replicate out-of-sample, and a recurring culprit is subtle look-ahead in the data. A famous class of errors comes from using restated accounting data: a backtest from 2005 that uses a company's financials *as restated in 2015* is reading from $\mathcal{F}_{2015}$ while pretending to stand in $\mathcal{F}_{2005}$. When researchers re-run these strategies on strict point-in-time data — databases that record only what was knowable on each date — the edge often shrinks dramatically or vanishes. This is the filtration discipline turned into a research standard, and entire data products (point-in-time fundamentals) exist to sell it.

**Value-at-Risk and the cdf at the bank level.** Every large bank computes a daily **Value-at-Risk (VaR)** number: the loss level that will not be exceeded with, say, 99% probability over one day. That is a pure cdf statement — VaR is the point $x$ where $F(x) = 0.01$ for the loss distribution. Regulators (under Basel rules) require it; risk committees stare at it daily. The 2008 crisis exposed its central flaw: VaR is silent about *how bad* the worst 1% can get, because the cdf at the threshold says nothing about the shape of the tail beyond it. This is why the industry shifted toward **Expected Shortfall** (the average loss *given* you are in the worst 1%) — a conditional-expectation object, $E[\text{loss} \mid \text{loss} > \text{VaR}]$, which is exactly the conditional-probability machinery of this post applied to the tail.

**Monte Carlo simulation of a sample space too big to list.** When $\Omega$ is enormous — every path a portfolio of thousands of instruments could take over a year — no one writes out the sample space. Instead, quants *sample* from it: draw thousands of random outcomes consistent with the assumed distribution, compute the random variable (the portfolio P&L) on each, and use the resulting empirical distribution to estimate probabilities and expectations. This is just the law of large numbers turning our $E[X] = \sum x\,p(x)$ into an average over simulated draws. Pricing desks run billions of such draws nightly. The whole technique is the probability triple made computational: $\Omega$ is sampled, the random variable is evaluated, and $P$ is approximated by counting.

**The "earnings surprise" as a filtration jump.** When a company reports earnings after the close, the market's information set takes a discrete jump: $\mathcal{F}$ just before the announcement does not contain the result; $\mathcal{F}$ just after does. The stock often gaps — a discontinuous price move — precisely because the random variable "stock price" is re-evaluated against a suddenly larger $\sigma$-algebra. Strategies that trade earnings are explicitly modeling the difference between the pre-announcement and post-announcement information sets, and any backtest of them that uses the post-announcement number to size a pre-announcement trade is committing the cardinal look-ahead sin.

**Kelly sizing and the expected value of a real edge.** A trader who has computed an honest expected value and distribution for a strategy still faces the question of *how much* to bet. The Kelly criterion answers it by maximizing the expected *logarithm* of wealth, which turns out to depend on exactly the objects we built: the win probability, the loss probability, and the payoff ratio. For our single trade — $+\$500$ at 0.55, $-\$450$ at 0.45 — Kelly says bet a fraction of capital proportional to the edge divided by the odds. Bet more than Kelly and the same positive-expectation trade can still bankrupt you through variance; bet less and you leave growth on the table. The episode that haunts every quant is the firm with a genuine \$72.50-per-trade edge that levered it 20-to-1, hit an unlucky run of the 45% losses, and blew up — proof that a positive expected value is necessary but nowhere near sufficient. The distribution, not just its mean, governs survival.

## When this matters to you and further reading

If you take one thing from this post into the rest of your quant journey, make it the discipline of separating the three layers. When you face any probability question — in an interview, in a model, in a backtest — ask the three questions in order. *What is $\Omega$?* Write out the complete, non-overlapping list of outcomes; most wrong answers die here. *What is $\mathcal{F}$ — what do I actually know, and when did I know it?* This is the question that catches look-ahead bugs before they cost you a job or a fund. *What is $P$ — and is it the right measure?* Real-world or risk-neutral, with realistic tails or naive independence; the measure is where your assumptions hide.

These ideas are not advanced for the sake of being advanced. They are the minimum vocabulary for everything that follows in this series. The moment you start computing expected values seriously, you will want the toolkit of shortcuts in [expected value techniques for quant interviews](/blog/trading/quantitative-finance/expected-value-techniques-quant-interviews). The moment you move past coin flips to real returns, you will need to know which named distribution fits which situation, which is the [distributions cheat sheet](/blog/trading/quantitative-finance/distributions-cheat-sheet-quant-interviews). And when you want to test whether you have truly internalized sample-space thinking, work through the [classic quant probability problems](/blog/trading/quantitative-finance/classic-quant-probability-problems) — every one of them is solvable the instant you write down the right $\Omega$.

Where this touches your life is broader than trading desks. The same three-layer reasoning is what separates clear thinking from muddled thinking about any uncertain situation: a medical test result (conditioning on a positive test), an insurance decision (an expected value with a fat tail), a bet with friends (a sample space and a measure). The mathematics we built here for trade P&L is the same mathematics for any decision made before the outcome is known — which is to say, almost every decision worth thinking about. This is educational material, not financial advice; the point is to give you the lens, not a position to take.

The next posts in the *Math for Quant Trading* series build directly on this foundation: expectation and variance as the first two summaries of a distribution, then the Bayesian updating that turns $P(A \mid B)$ into the engine of learning from data, then the leap to continuous-time processes where the filtration $\mathcal{F}_t$ becomes the stage for Brownian motion and stochastic calculus. Everything starts here, with three letters kept carefully apart: $\Omega$, $\mathcal{F}$, and $P$.
