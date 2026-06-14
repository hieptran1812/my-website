---
title: "Expected value techniques: linearity, indicators, and symmetry"
date: "2026-06-14"
publishDate: "2026-06-14"
description: "A from-scratch guide to the four expected-value techniques -- linearity, indicators, symmetry, and first-step recursion -- that crack most quant interview problems in a line or two, with fully worked numeric examples and interview-style solutions."
tags: ["expected-value", "quant-interviews", "probability", "linearity-of-expectation", "indicator-variables", "symmetry", "recursion", "optional-stopping", "brain-teasers", "quantitative-trading"]
category: "trading"
subcategory: "Quantitative Finance"
author: "Hiep Tran"
featured: true
readTime: 40
---

> [!important]
> **TL;DR** -- you almost never compute an expected value by brute force. Four techniques crack the large majority of expected-value interview problems in a line or two.
>
> - **Linearity of expectation**: the expected value of a sum is the sum of the expected values -- and this holds *even when the pieces depend on each other*. That one surprising fact is the most powerful tool in the kit.
> - **Indicator variables**: any *count* of events is a sum of 0/1 flags, and the expected value of a 0/1 flag is just the probability it equals 1. Counting problems collapse to "add up the probabilities".
> - **Symmetry**: when outcomes play interchangeable roles, they must share the total equally -- so you can read an answer straight off the structure with no summation at all.
> - **First-step (recursive) analysis**: for a repeating process, condition on the very first move; the same unknown expectation reappears one level down, giving one equation you solve for it.
> - The number to remember: the expected number of *fixed points* of a random shuffle is exactly **1**, no matter how many items you shuffle -- and you'll prove it in one line with indicators.

Here is a question a trading-firm interviewer might open with: *you shuffle a standard deck of 52 cards; how many cards, on average, end up in the exact position they started in?* Most people reach for a sum over messy permutation probabilities and stall. The answer is **exactly 1**, and the clean solution is a single line. The gap between the messy attempt and the one-liner is not raw intelligence -- it is knowing which of a small handful of techniques to reach for.

That is what this post is about. Expected value -- the long-run average of a random quantity -- is the single most-tested concept in quantitative interviews at firms like Jane Street, Citadel, Two Sigma, Optiver, SIG, Hudson River Trading, Jump, and DE Shaw. And the interviewers are not testing whether you can grind through arithmetic. They are testing whether you recognize structure. There are essentially four levers, and once you internalize them, a large fraction of "EV brain teasers" become a line or two of reasoning.

![The four-technique toolbox for expected value: linearity, indicators, symmetry, and first-step analysis all feeding the central question, what is E[X]?](/imgs/blogs/expected-value-techniques-quant-interviews-1.png)

The diagram above is the mental model. The central question is always "what is the expected value?" The four boxes around it are the levers: **linearity**, **indicators**, **symmetry**, and **first-step recursion**. The whole game is matching the shape of the problem to the right lever. We will build each one from absolute zero, ground it in numbers you can check by hand, and then frame it explicitly for the interview room -- including a "how this shows up" angle and a problem set with full step-by-step solutions.

No probability background is assumed. We define every term the first time it appears.

## The building blocks: what expected value actually is

Before any technique, we need to be crystal clear on what we are computing.

A **random variable** is just a number whose value is determined by chance -- the result of a die roll, the payout of a bet, the number of heads in ten coin flips. We usually write it with a capital letter like $X$. A random variable is not a single number; it is a *recipe* that produces a number once the randomness resolves.

The **expected value** of a random variable, written $\mathbb{E}[X]$, is its probability-weighted average: you multiply each possible value by the probability of getting it, and add those products up. Formally, if $X$ can take values $x_1, x_2, \dots$ with probabilities $p_1, p_2, \dots$, then

$$\mathbb{E}[X] = \sum_i x_i \, p_i.$$

Here $x_i$ is one possible value of $X$, $p_i$ is the probability $X$ equals that value, and the sum runs over every value $X$ can take. The probabilities are non-negative and add up to 1, so the expected value is genuinely a weighted average -- it always lands somewhere between the smallest and largest possible outcomes.

#### Worked example: a single die

You roll one fair six-sided die. The possible values are $1, 2, 3, 4, 5, 6$, each with probability $\tfrac{1}{6}$. So

$$\mathbb{E}[X] = 1\cdot\tfrac{1}{6} + 2\cdot\tfrac{1}{6} + 3\cdot\tfrac{1}{6} + 4\cdot\tfrac{1}{6} + 5\cdot\tfrac{1}{6} + 6\cdot\tfrac{1}{6} = \frac{1+2+3+4+5+6}{6} = \frac{21}{6} = 3.5.$$

The expected value is **3.5** -- a number the die can never actually show. That is the single most important intuition in this entire post: *the expected value is a balance point, not a prediction of the most likely outcome.* The die never rolls 3.5, yet 3.5 is the average you converge to over many rolls. The intuition this teaches: the mean is where the distribution balances, not where it most often lands.

### Why the expected value is not the most likely outcome

This trips up beginners constantly, so let us make it visual. Consider a payout that is 0 half the time, 2 with probability 0.30, and 4 with probability 0.20.

![A bar chart with three outcome bars at values 0, 2, and 4; the mean E[X] = 1.4 marked by a dashed line lands between bars where no bar exists, while the most likely outcome is 0.](/imgs/blogs/expected-value-techniques-quant-interviews-2.png)

The most likely outcome is 0 -- it carries half the probability mass (the term *probability mass* just means "how much probability sits on that value"). But the expected value is

$$\mathbb{E}[X] = 0\cdot 0.50 + 2\cdot 0.30 + 4\cdot 0.20 = 0 + 0.6 + 0.8 = 1.4.$$

The mean is **1.4**, a value that *never occurs*. Think of the bars as weights on a seesaw: the expected value is the point where the seesaw balances. The tall bar at 0 pulls left, but the bars out at 2 and 4 have enough leverage to drag the balance point out to 1.4. An interviewer who asks "what is the expected payout?" wants 1.4. An interviewer who asks "what is the most likely payout?" wants 0. Hearing the difference is half the battle.

Two more facts we will lean on constantly:

- **Expected value scales and shifts linearly.** For any constants $a$ and $b$, $\mathbb{E}[aX + b] = a\,\mathbb{E}[X] + b$. If you double every payout and add 5, the average doubles and shifts by 5. This is intuitive: stretching and sliding the values stretches and slides their average.
- **The expectation of a constant is the constant.** $\mathbb{E}[7] = 7$. There is no randomness to average over.

With those in hand, we meet the first and most important technique.

## Technique 1 -- Linearity of expectation (the one that surprises everyone)

Here is the rule. For *any* two random variables $X$ and $Y$,

$$\mathbb{E}[X + Y] = \mathbb{E}[X] + \mathbb{E}[Y].$$

More generally, for any number of pieces, the expected value of the sum is the sum of the expected values:

$$\mathbb{E}[X_1 + X_2 + \dots + X_n] = \mathbb{E}[X_1] + \mathbb{E}[X_2] + \dots + \mathbb{E}[X_n].$$

That looks almost too simple to matter. The surprise -- and it genuinely surprises most people the first time -- is the fine print: **this holds even when $X$ and $Y$ are dependent.** The pieces can influence each other in any tangled way you like. They can be perfectly correlated, anti-correlated, or related through some baroque mechanism. Linearity does not care. You never need the pieces to be independent.

![Linearity of expectation contrasted with brute force: instead of enumerating every joint outcome, you split the total into simple parts, take each part's expectation, and add them back -- dependence never enters.](/imgs/blogs/expected-value-techniques-quant-interviews-3.png)

Why is this such a big deal? Because it lets you replace one *hard* expectation -- the expectation of a complicated total, which would normally force you to enumerate every joint outcome and weight by the joint probability -- with a *sum of easy* expectations, each computed on its own. The diagram contrasts the two routes: the brute-force path (red, left) enumerates every combined outcome and weights by its joint probability, a combinatorial nightmare; the linearity path (blue/green, right) chops the total into simple parts, finds each part's average separately, and adds. Crucially, the parts are allowed to overlap and interfere -- linearity still works.

Two terms to pin down. **Independent** means knowing the value of one variable tells you nothing about the other (separate dice). **Dependent** means it does (the same die read two different ways). The headline again: linearity needs *neither*. It is unconditionally true.

#### Worked example: the sum of two dice, two ways

You roll two fair dice and add them. What is the expected total?

*The brute-force way.* You could list all 36 equally likely outcomes of the pair, compute the sum for each, and average. The sums range from 2 to 12, and you would find the average is 7. Doable, but tedious -- and it gets impossible fast as the number of dice grows.

*The linearity way.* Let $X$ be the first die and $Y$ the second. The total is $X + Y$. We already know $\mathbb{E}[X] = 3.5$ and $\mathbb{E}[Y] = 3.5$. So

$$\mathbb{E}[X + Y] = \mathbb{E}[X] + \mathbb{E}[Y] = 3.5 + 3.5 = 7.$$

One line. And here is the kicker: even if the two dice were *glued together* so that they always showed the same face -- maximally dependent -- the expected total would *still* be $3.5 + 3.5 = 7$. Try it: the glued pair shows $(1,1), (2,2), \dots, (6,6)$ each with probability $\tfrac{1}{6}$, giving sums $2, 4, 6, 8, 10, 12$, whose average is $\tfrac{2+4+6+8+10+12}{6} = \tfrac{42}{6} = 7$. Same answer. The dependence changed the *spread* of the total dramatically but not its *average*. The intuition this teaches: linearity lets you ignore how the pieces interact and just add their separate averages.

### How linearity shows up in interviews

The tell is almost always the same: **the quantity you are asked about is a sum or a total.** "Expected total of the dice." "Expected number of something." "Expected length of something." The instant you can write the target as $X_1 + X_2 + \dots + X_n$, you stop thinking about the joint distribution entirely and just add up the easy marginal expectations. Interviewers love linearity precisely because the naive joint-distribution approach looks intractable, so a candidate who reaches for it signals real fluency. The next technique is how you manufacture that sum when the problem is about *counting*.

## Technique 2 -- Indicator-variable decomposition

A huge class of EV problems asks: *how many events happen, on average?* How many fixed points in a shuffle? How many people share a birthday? How many empty bins? How many record-high days? These are **counting** problems, and indicators turn every one of them into a linearity problem.

An **indicator variable** (also called a *Bernoulli* variable) is a random variable that equals 1 if some event happens and 0 if it does not. We write $I_A$ for the indicator of event $A$:

$$I_A = \begin{cases} 1 & \text{if event } A \text{ happens} \\ 0 & \text{otherwise.}\end{cases}$$

The magic is what its expected value works out to. Since $I_A$ takes only the values 0 and 1,

$$\mathbb{E}[I_A] = 1\cdot \mathbb{P}(A) + 0\cdot \mathbb{P}(\text{not } A) = \mathbb{P}(A).$$

**The expected value of an indicator is exactly the probability of the event.** That tiny identity is the hinge of the whole technique.

Now the recipe. Suppose you want the expected number of events that happen, out of $n$ possible events. Define one indicator $I_i$ for each event: $I_i = 1$ if event $i$ happens, else 0. The *total count* of events that happen is just the sum of the flags:

$$X = I_1 + I_2 + \dots + I_n.$$

Apply linearity, then the indicator identity:

$$\mathbb{E}[X] = \mathbb{E}[I_1] + \dots + \mathbb{E}[I_n] = \mathbb{P}(\text{event } 1) + \dots + \mathbb{P}(\text{event } n).$$

The expected count is simply the **sum of the individual event probabilities**. And because we used linearity, *the events are allowed to depend on each other in any way* -- we never need them to be independent.

![Indicator decomposition: a count is a sum of one 0/1 flag per event; taking the expectation of each flag gives its probability, and linearity sums them -- no independence required.](/imgs/blogs/expected-value-techniques-quant-interviews-4.png)

The figure walks the whole pipeline: five events, each with its own 0/1 flag (green when it fired, red when it did not); this round three flags are 1, so the count $X = 3$; take the expectation of the sum, and out drops $\mathbb{E}[X] = p_1 + p_2 + \dots + p_n$, the sum of probabilities. Notice we never asked whether the events were independent. That is the entire point.

#### Worked example: fixed points of a random permutation

Back to the opening question. You shuffle $n$ items uniformly at random. A **fixed point** is an item that lands back in its own starting position. What is the expected number of fixed points?

The brute-force approach -- summing over the number of permutations with exactly $k$ fixed points -- involves *derangements* (permutations with no fixed point) and is genuinely fiddly. With indicators it is a one-liner.

1. For each position $i$, define the indicator $I_i = 1$ if item $i$ ends up in position $i$ (a fixed point), else 0.
2. The number of fixed points is $X = I_1 + I_2 + \dots + I_n$.
3. By symmetry, item $i$ is equally likely to land in any of the $n$ positions, so the probability it lands in *its own* position is $\mathbb{P}(I_i = 1) = \tfrac{1}{n}$.
4. By linearity: $\mathbb{E}[X] = \sum_{i=1}^{n} \mathbb{E}[I_i] = \sum_{i=1}^{n} \tfrac{1}{n} = n \cdot \tfrac{1}{n} = 1.$

![Fixed points of a random permutation: each of the n positions is a fixed point with probability 1/n, so by linearity the expected count is n times 1/n equals exactly 1, independent of n.](/imgs/blogs/expected-value-techniques-quant-interviews-5.png)

The expected number of fixed points is **exactly 1**, for *every* $n$. Shuffle 5 cards or 52 or 5 million -- on average, one card comes home. And note how badly the indicators depend on each other: if item 1 is fixed, that changes the probabilities for everyone else (one fewer position to fill). Linearity sails right past that dependence. The intuition this teaches: to count events, sum their probabilities and let dependence take care of itself.

### How indicators show up in interviews

Whenever the question contains the words "expected **number** of ..." you should immediately think *indicators*. The skill is choosing the right per-event flag and computing one probability. Classic interview prompts that yield to this in one or two lines: expected number of people who get their own hat back from a random redistribution (= 1, identical to fixed points); expected number of adjacent equal cards in a shuffled deck; expected number of distinct coupons after $k$ draws; expected number of "ascents" in a random sequence. Each is a count, so each is a sum of indicators, so each is a sum of probabilities.

## Technique 3 -- Symmetry arguments

Sometimes you do not even need to set up indicators. If the structure of a problem is **symmetric** -- meaning some set of outcomes play perfectly interchangeable roles -- then those outcomes must have the *same* expected value, and that single fact often hands you the answer with no computation at all.

The reasoning is a clean two-step. First, argue that several quantities are interchangeable: relabel them, rotate them, or reflect the setup, and the problem looks identical. If nothing distinguishes them, they must share the same expected value. Second, you usually know what all of them *sum* to. Divide the total by the count, and each one's expected value pops out.

#### Worked example: where does the first ace fall in a shuffled deck?

Shuffle a standard 52-card deck. On average, in what position does the **first ace** appear?

Setting up indicators here is awkward. Symmetry makes it effortless. There are 4 aces and 48 non-aces. Lay the deck out and look at the four aces; they chop the 48 non-aces into **five gaps**: the cards before the first ace, between the first and second, and so on, with a final gap after the fourth ace.

![Symmetry of the four aces: the 48 non-aces fall into five gaps around the four aces, and because no gap is special, each gap has the same expected size of 48/5 = 9.6 cards.](/imgs/blogs/expected-value-techniques-quant-interviews-6.png)

Now the symmetry argument. *No gap is special.* If you relabel the aces, or reflect the whole deck end-to-end, the five gaps simply swap roles -- the problem is unchanged. So by symmetry the five gaps all have the **same expected size**. Since they together hold all 48 non-aces,

$$\mathbb{E}[\text{size of one gap}] = \frac{48}{5} = 9.6.$$

The first ace sits right after the first gap, so its expected position is

$$\mathbb{E}[\text{position of first ace}] = \mathbb{E}[\text{gap 1 size}] + 1 = 9.6 + 1 = 10.6.$$

No summation, no messy probabilities -- just "the gaps are interchangeable, so they split the total evenly." The intuition this teaches: when nothing distinguishes a set of pieces, they must each carry an equal share of whatever they collectively make up.

### How symmetry shows up in interviews

Symmetry is the technique that makes you look fast. The tell: the problem involves *positions*, *ranks*, *orderings*, or *interchangeable players*, and you sense that singling one out would be arbitrary. "Expected rank of a card." "Expected position of the first/last special item." "Expected fraction of the circle in each arc." "Which of two symmetric players is more likely to win?" In each case, resist the urge to compute -- ask first whether relabeling leaves the problem unchanged. If it does, the answer is a fraction of a total you already know.

## Technique 4 -- First-step (recursive) analysis

The previous three techniques shine when the answer is a sum, a count, or a symmetric share. The fourth handles a different shape: a **process that repeats** -- you keep flipping, drawing, or walking until something happens, and you want the expected number of steps.

The trick is to **condition on the very first step**. Write $E$ for the unknown expected number of steps from the start. Take one step (that costs you 1). With some probability you are done; with the rest you have landed back in a state that looks like the start, from which the remaining expected number of steps is *again* $E$. That self-reference gives you a single equation with $E$ on both sides, which you solve algebraically.

![First-step analysis as a recursion tree: from the start you spend one step, then with probability p you reach the goal and with probability 1-p you return to a start-like state whose expected remaining steps is the same E -- one equation you solve for E.](/imgs/blogs/expected-value-techniques-quant-interviews-7.png)

The recursion tree shows the logic. You always pay 1 for the first step. With probability $p$ you reach the goal (0 more steps). With probability $1-p$ you bounce back to a start-like state, from which you again expect $E$ steps. So

$$E = 1 + p\cdot 0 + (1-p)\cdot E.$$

Solve: $E - (1-p)E = 1$, so $pE = 1$, giving $E = \tfrac{1}{p}$. That recovers a fact you may already know -- the expected number of trials to get the first success when each trial succeeds with probability $p$ is $\tfrac{1}{p}$ (the *geometric distribution* mean). But the first-step method generalizes far beyond that simple case, because the "start-like state" can be one of *several* states, each feeding back its own expectation.

#### Worked example: the coupon collector

A cereal box contains one of $n$ equally likely toys. You buy boxes until you have collected all $n$ distinct toys. How many boxes, on average?

Break the journey into **phases** by how many *distinct* toys you already own. Suppose you currently hold $k$ distinct toys. The probability that the next box is a *new* toy is $\tfrac{n-k}{n}$ (there are $n-k$ toys you are missing out of $n$). By the geometric-mean fact above, the expected number of boxes to get one new toy from this phase is the reciprocal:

$$\mathbb{E}[\text{boxes in phase } k] = \frac{n}{n-k}.$$

You pass through phases $k = 0, 1, \dots, n-1$. By linearity, the total expected number of boxes is the sum:

$$\mathbb{E}[\text{total}] = \sum_{k=0}^{n-1} \frac{n}{n-k} = n\left(\frac{1}{n} + \frac{1}{n-1} + \dots + \frac{1}{1}\right) = n\,H_n,$$

where $H_n = 1 + \tfrac{1}{2} + \tfrac{1}{3} + \dots + \tfrac{1}{n}$ is the $n$-th **harmonic number**. For large $n$, $H_n \approx \ln n + 0.577$, so the expected number of draws grows like $n\ln n$.

![Coupon collector curve: the expected number of draws to collect all n coupons is n times the n-th harmonic number, growing like n ln n -- 29 draws for 10 coupons, 72 for 20, 225 for 50, with the last coupon alone taking about n draws.](/imgs/blogs/expected-value-techniques-quant-interviews-8.png)

The curve makes the growth concrete. For $n = 10$ coupons, $H_{10} = 2.93$, so you expect $10 \times 2.93 = 29.3$ draws. For $n = 20$, about 72. For $n = 50$, about 225. The shape is the punchline: the *first* coupon is free (any box works), but the *last* coupon -- when you are missing just one -- takes about $n$ boxes on its own, because each box has only a $\tfrac{1}{n}$ chance of being the one you need. The tail dominates the wait. The intuition this teaches: collecting "the last few" is what makes completion expensive, and the cost grows like $n\ln n$, not like $n$.

This example quietly used *three* techniques at once -- first-step thinking (the per-phase reciprocal), linearity (summing the phases), and symmetry (every missing toy is equally likely). That blending is exactly what fluency looks like.

#### Worked example: an ant walking on a cube

Here is a recursion that genuinely needs the multi-state version. An ant sits at one corner of a cube and wants to reach the diagonally opposite corner. At each step it walks along one of the 3 edges leaving its current corner, chosen uniformly at random. How many steps, on average?

A cube has 8 corners, which sounds like 8 unknowns. But **symmetry** collapses them: all that matters is the ant's *distance to the target*, measured in edges. There are four distance classes -- $D_0$ (the target itself), $D_1$ (the 3 corners adjacent to the target), $D_2$ (the 3 corners adjacent to the start), and $D_3$ (the start corner, diagonally opposite). Let $E_d$ be the expected number of steps to reach the target from distance $d$. Of course $E_0 = 0$.

![Ant on a cube: the eight corners collapse by symmetry into four distance states D0 through D3, giving three coupled first-step equations whose solution is E1 = 7, E2 = 9, E3 = 10 -- the ant needs 10 steps on average.](/imgs/blogs/expected-value-techniques-quant-interviews-9.png)

Now write a first-step equation for each state by looking at where the 3 edges lead:

- From $D_3$ (the start), all 3 neighbors are $D_2$ corners. So $E_3 = 1 + E_2$.
- From $D_2$, one neighbor is the $D_3$ corner and two neighbors are $D_1$ corners. So $E_2 = 1 + \tfrac{1}{3}E_3 + \tfrac{2}{3}E_1$.
- From $D_1$, one neighbor is the target ($D_0$, contributing 0) and two neighbors are $D_2$ corners. So $E_1 = 1 + \tfrac{1}{3}\cdot 0 + \tfrac{2}{3}E_2$.

Three equations, three unknowns. Substitute $E_3 = 1 + E_2$ and $E_1 = 1 + \tfrac{2}{3}E_2$ into the middle equation:

$$E_2 = 1 + \tfrac{1}{3}(1 + E_2) + \tfrac{2}{3}\left(1 + \tfrac{2}{3}E_2\right) = 1 + \tfrac{1}{3} + \tfrac{1}{3}E_2 + \tfrac{2}{3} + \tfrac{4}{9}E_2 = 2 + \tfrac{7}{9}E_2.$$

So $E_2 - \tfrac{7}{9}E_2 = 2$, i.e. $\tfrac{2}{9}E_2 = 2$, giving $E_2 = 9$. Then $E_1 = 1 + \tfrac{2}{3}\cdot 9 = 7$ and $E_3 = 1 + 9 = 10$.

The ant takes **10 steps on average** to cross the cube. The intuition this teaches: when a process has a few distinct "states", write one first-step equation per state and solve the small linear system -- symmetry keeps the number of states tiny.

### How first-step analysis shows up in interviews

The tell is a **memoryless, repeating process**: coin patterns, random walks, gambler's-ruin-style games, "keep drawing until ...". The skill is spotting the small set of states (often collapsed by symmetry, as on the cube) and writing the self-referential equation. When the interviewer says "expected number of *tosses/steps/draws until* X", reach for first-step analysis before anything else.

## A subtle case: waiting for a pattern (HH vs HT)

First-step analysis has a famous trap that interviewers adore, because the "obvious" answer is wrong. Flip a fair coin repeatedly. On average, how many flips until you first see the pattern **HH** (two heads in a row)? And how many until you first see **HT** (a head then a tail)? Intuition screams they should be equal -- both are two-flip patterns, each with probability $\tfrac{1}{4}$ per fixed pair of positions. They are *not* equal: $\mathbb{E}[\text{HH}] = 6$ but $\mathbb{E}[\text{HT}] = 4$.

![Why HH waits longer than HT: in the waiting automaton, after a single H a tail throws HH all the way back to the start but only nudges HT forward, so HH averages 6 tosses while HT averages 4.](/imgs/blogs/expected-value-techniques-quant-interviews-10.png)

The reason is what happens **after a partial match**, shown in the automaton above. Suppose you have just flipped an H and are "halfway" to either target.

- **Chasing HT:** the next flip is either T (you win) or H (you are *still* halfway -- you have a fresh H to build on). You never lose progress.
- **Chasing HH:** the next flip is either H (you win) or T -- and a T destroys *all* your progress. You are thrown back to the start with nothing.

That asymmetry in what a mismatch costs is the whole story. Let us nail it with first-step equations for HT. Let $E$ be the expected flips from scratch, and $E_H$ the expected additional flips once you have a single H "in hand".

$$E = 1 + \tfrac{1}{2}E_H + \tfrac{1}{2}E \quad(\text{first flip is H or T}),$$
$$E_H = 1 + \tfrac{1}{2}\cdot 0 + \tfrac{1}{2}E_H \quad(\text{from an H, a T finishes, an H keeps you at } E_H).$$

From the second equation, $\tfrac{1}{2}E_H = 1$, so $E_H = 2$. Plug into the first: $E = 1 + \tfrac{1}{2}\cdot 2 + \tfrac{1}{2}E$, i.e. $\tfrac{1}{2}E = 2$, so $\mathbf{E = 4}$ for HT.

Now HH. Again $E$ from scratch and $E_H$ after one H:

$$E = 1 + \tfrac{1}{2}E_H + \tfrac{1}{2}E,$$
$$E_H = 1 + \tfrac{1}{2}\cdot 0 + \tfrac{1}{2}E \quad(\text{from an H, an H finishes, but a T sends you back to the start, } E).$$

Substitute the second into the first: $E = 1 + \tfrac{1}{2}\left(1 + \tfrac{1}{2}E\right) + \tfrac{1}{2}E = 1 + \tfrac{1}{2} + \tfrac{1}{4}E + \tfrac{1}{2}E = \tfrac{3}{2} + \tfrac{3}{4}E$. So $\tfrac{1}{4}E = \tfrac{3}{2}$, giving $\mathbf{E = 6}$ for HH. The difference -- 6 versus 4 -- is entirely due to the T in the HH chase wiping out partial progress, while the T in the HT chase *completes* it. The intuition this teaches: the expected wait for a pattern depends on how much progress a mismatch destroys, not just on the pattern's probability.

## Optional stopping: when to keep drawing

A close cousin of these problems asks not "how long until X" but "when should I *stop*?" You are offered repeated draws from some distribution, you can stop whenever you like and keep your current value, and you want to maximize your expected payoff. This is an **optional stopping** problem, and the core intuition is short enough to carry into any interview.

![Optional stopping decision: draw again only when the expected value of the next draw exceeds the value you already hold, looping back to re-check after each draw, and stop -- cashing out -- the moment no draw beats your current value in expectation.](/imgs/blogs/expected-value-techniques-quant-interviews-11.png)

The rule, shown as a decision loop above: **draw again exactly when the expected value of the next draw exceeds the value you already hold.** Your current best is the threshold. If continuing has positive expected gain over stopping, continue; the moment it does not, stop and cash out. After each draw you simply re-check the same comparison with your updated value.

#### Worked example: roll a die, keep your last roll

You may roll a fair die up to two times. After the first roll you choose to keep it or reroll; if you reroll, you must keep the second roll. You are paid the dollar value of the face you keep. What is the optimal strategy, and what is your expected payoff?

Work backwards. The expected value of a *fresh* roll is 3.5 (computed at the very start). So after your first roll, you should **reroll exactly when your first roll is below 3.5** -- that is, when it shows 1, 2, or 3 -- because a fresh roll's expected value (3.5) beats what you hold. If you roll 4, 5, or 6, you keep it, because no reroll beats it in expectation.

Now compute the expected payoff under this rule:

- With probability $\tfrac{1}{2}$ (rolling 4, 5, or 6) you keep the first roll. The average of 4, 5, 6 is 5, so this branch contributes $\tfrac{1}{2}\cdot 5 = 2.5$.
- With probability $\tfrac{1}{2}$ (rolling 1, 2, or 3) you reroll and accept a fresh die worth 3.5 on average, contributing $\tfrac{1}{2}\cdot 3.5 = 1.75$.

$$\mathbb{E}[\text{payoff}] = 2.5 + 1.75 = 4.25.$$

The optimal two-roll strategy earns **\$4.25** on average, versus 3.5 for a single forced roll -- the option to reroll is worth 0.75. The intuition this teaches: the right stopping threshold is the expected value of continuing, so you keep anything that beats it and reroll anything that does not.

## In the interview room

Now we put it together. Below are four interview-style problems with full step-by-step solutions. For each, the first move is to *name the technique* -- that recognition is what the interviewer is really watching for.

### Problem 1 -- Expected number of distinct birthdays

*Thirty people walk into a room. Assuming birthdays are uniform over 365 days and independent, what is the expected number of **distinct** days that appear as someone's birthday?*

**Technique: indicators.** "Expected number of distinct ..." is a count, so set up one flag per day.

1. For each day $d$ (there are 365), let $I_d = 1$ if at least one person has birthday $d$, else 0. The number of distinct birthdays is $X = \sum_{d=1}^{365} I_d$.
2. By linearity, $\mathbb{E}[X] = \sum_{d=1}^{365} \mathbb{P}(\text{day } d \text{ is used})$.
3. Compute one probability. The chance a *single* person misses day $d$ is $\tfrac{364}{365}$. With 30 independent people, the chance *all* miss day $d$ is $\left(\tfrac{364}{365}\right)^{30}$. So the chance day $d$ is used is $1 - \left(\tfrac{364}{365}\right)^{30}$.
4. By symmetry every day has the same probability, so $\mathbb{E}[X] = 365\left(1 - \left(\tfrac{364}{365}\right)^{30}\right)$.

Numerically, $\left(\tfrac{364}{365}\right)^{30} \approx 0.9208$, so $\mathbb{E}[X] \approx 365 \times 0.0792 \approx 28.9$. With 30 people you expect about **28.9 distinct birthdays** -- you only lose about 1.1 to collisions. The clean move was turning a scary "distinct count" into 365 identical probabilities and adding them.

### Problem 2 -- How many people until two share a birthday?

*People enter a room one at a time. On average, how many people must enter before two of them share a birthday? (The famous "birthday problem", phrased as an expectation.)*

This one is a trap if you confuse it with the *threshold* version. The well-known "23 people" fact is the point where the *probability* of a shared birthday first exceeds 50% -- it is a median-style statement, **not** an expectation.

![The birthday collision tipping point: the probability that all birthdays stay distinct falls below 50% at just 23 people, while the expected number you must ask before a collision is about 24.6 -- the two questions have different answers.](/imgs/blogs/expected-value-techniques-quant-interviews-12.png)

The chart shows the probability that all birthdays remain distinct as people enter: it crosses 50% at exactly **23 people** (the celebrated answer to "how many people make a shared birthday more likely than not"). But the *expected* number you must ask before the first collision is a different quantity. Working it out -- by summing the probability that the streak of "all distinct" survives each additional person -- gives an expectation of about **24.6 people**. The two numbers (23 for the median-ish threshold, 24.6 for the mean) are close but answer genuinely different questions. The interview skill here is hearing *which* question was asked: "more likely than not" means the 50% threshold (23); "on average / expected" means the mean (about 24.6). Saying "23" to an expectation question is a classic miss.

### Problem 3 -- Expected longer piece of a broken stick

*You break a stick of length 1 at a uniformly random point. What is the expected length of the **longer** of the two pieces?*

**Technique: direct expectation with a symmetry shortcut.** Let the break point be $U$, uniform on $[0, 1]$. The two pieces have lengths $U$ and $1 - U$. The longer piece is $\max(U, 1-U)$.

By symmetry, the longer piece is at least $\tfrac{1}{2}$ (one piece always gets the majority), and the break point's *distance from the middle*, $|U - \tfrac{1}{2}|$, is uniform on $[0, \tfrac{1}{2}]$ with average $\tfrac{1}{4}$. The longer piece equals $\tfrac{1}{2} + |U - \tfrac{1}{2}|$, so

$$\mathbb{E}[\text{longer piece}] = \tfrac{1}{2} + \mathbb{E}\left[\,\left|U - \tfrac{1}{2}\right|\,\right] = \tfrac{1}{2} + \tfrac{1}{4} = \tfrac{3}{4}.$$

The longer piece averages **3/4** of the stick (and so the shorter piece averages 1/4). The symmetry move -- recentering on the midpoint -- turned a $\max$ into a clean uniform average. As a sanity check via direct integration: $\mathbb{E}[\max] = \int_0^1 \max(u, 1-u)\,du = 2\int_{1/2}^1 u\,du = 2\cdot\tfrac{1}{2}(1 - \tfrac{1}{4}) = \tfrac{3}{4}$. Same answer.

### Problem 4 -- The gambler who doubles or quits

*You start with \$1. Each round, a fair coin decides whether your money doubles or you lose it all. You quit the instant you decide to. If you commit in advance to playing exactly $k$ rounds (stopping early only if you have been wiped out), what is your expected wealth?*

**Technique: linearity via the multiplicative structure, then a stopping caution.** This problem is designed to expose a confusion between expected wealth and the probability of being rich.

Let $W_k$ be your wealth after $k$ rounds. Each round, wealth multiplies by 2 with probability $\tfrac{1}{2}$ (heads) and by 0 with probability $\tfrac{1}{2}$ (tails). The expected multiplier per round is $\tfrac{1}{2}\cdot 2 + \tfrac{1}{2}\cdot 0 = 1$. Because the rounds are independent multipliers, the expected wealth after $k$ rounds is the product of the per-round expected multipliers:

$$\mathbb{E}[W_k] = 1 \times 1^k = 1.$$

Your expected wealth is **\$1**, unchanged, for any $k$. But here is the catch the interviewer is fishing for: that \$1 average is wildly misleading about your *actual* fate. With probability $1 - \left(\tfrac{1}{2}\right)^k$ you have been wiped out to \$0; with the tiny probability $\left(\tfrac{1}{2}\right)^k$ you survived every round and are sitting on $2^k$. For $k = 10$: a 99.9% chance of \$0 and a 0.1% chance of \$1024 -- which averages back to exactly \$1. The expected value is dominated by a vanishingly rare jackpot. The intuition this teaches: a healthy expected value can hide near-certain ruin, which is precisely why traders care about the *whole distribution*, not just the mean. This is the seed of why position sizing and risk control exist at all.

### Problem 5 -- Expected value of the maximum of two dice

*Roll two fair dice. What is the expected value of the larger of the two faces (ties count as that value)?*

**Technique: indicators via the "tail sum" identity.** For a non-negative integer random variable $M$, there is a slick identity: $\mathbb{E}[M] = \sum_{j \geq 1} \mathbb{P}(M \geq j)$. Each term $\mathbb{P}(M \geq j)$ is the expectation of the indicator "$M$ is at least $j$", and summing them rebuilds $M$.

Let $M = \max(X, Y)$ of two dice. Then $\mathbb{P}(M \geq j) = 1 - \mathbb{P}(\text{both dice} < j) = 1 - \left(\tfrac{j-1}{6}\right)^2$ for $j = 1, \dots, 6$.

| $j$ | $\mathbb{P}(M \geq j)$ |
|---|---|
| 1 | $1 - 0 = 1$ |
| 2 | $1 - (1/6)^2 = 35/36$ |
| 3 | $1 - (2/6)^2 = 32/36$ |
| 4 | $1 - (3/6)^2 = 27/36$ |
| 5 | $1 - (4/6)^2 = 20/36$ |
| 6 | $1 - (5/6)^2 = 11/36$ |

Summing the tail probabilities:

$$\mathbb{E}[M] = 1 + \frac{35 + 32 + 27 + 20 + 11}{36} = 1 + \frac{125}{36} = \frac{36 + 125}{36} = \frac{161}{36} \approx 4.47.$$

The expected maximum is **161/36 ≈ 4.47**, comfortably above the 3.5 average of a single die -- taking the larger of two rolls pulls the average up, as it should. The tail-sum identity turned a $\max$ (awkward to enumerate directly) into six easy "at least $j$" probabilities.

#### Worked example: expected number of records (left-to-right maxima)

*You read off $n$ distinct numbers in a uniformly random order -- say the daily high temperatures of $n$ days arriving in random sequence. A **record** is a value that is larger than everything before it (the first value is automatically a record). On average, how many records do you see? Compute it for $n = 10$.*

**Technique: indicators.** "Expected number of records" is a count, so the move is forced: one 0/1 flag per position, then add up the probabilities. The only real work is computing one probability cleanly, and a small symmetry insight makes it fall out instantly.

1. For each position $k$ (from 1 to $n$), define the indicator $R_k = 1$ if the value in position $k$ is a record -- strictly larger than all $k-1$ values before it -- and $R_k = 0$ otherwise. The total number of records is $X = R_1 + R_2 + \dots + R_n$.
2. By linearity, $\mathbb{E}[X] = \sum_{k=1}^{n} \mathbb{P}(\text{position } k \text{ is a record})$. As always with indicators, we do *not* need the positions to be independent -- and they very much are not, since one record changes the bar for the next.
3. Now the key probability, and here is where symmetry does the heavy lifting. Position $k$ is a record exactly when the $k$-th value is the **largest among the first $k$ values**. But the first $k$ values arrived in a uniformly random order, so each of them is equally likely to be the largest of the group. The largest is therefore in the final slot (position $k$) with probability exactly $\tfrac{1}{k}$. Notice this does not depend at all on *which* numbers they are -- only on their relative order -- which is why the answer is so clean.
4. Therefore $\mathbb{P}(\text{position } k \text{ is a record}) = \tfrac{1}{k}$, and summing,

$$\mathbb{E}[X] = \sum_{k=1}^{n} \frac{1}{k} = 1 + \frac{1}{2} + \frac{1}{3} + \dots + \frac{1}{n} = H_n,$$

the $n$-th **harmonic number** -- the very same object that governed the coupon collector earlier. For $n = 10$,

$$\mathbb{E}[X] = H_{10} = 1 + \tfrac{1}{2} + \tfrac{1}{3} + \tfrac{1}{4} + \tfrac{1}{5} + \tfrac{1}{6} + \tfrac{1}{7} + \tfrac{1}{8} + \tfrac{1}{9} + \tfrac{1}{10} = \frac{7381}{2520} \approx 2.93.$$

So across 10 randomly ordered values you expect only about **2.93 records** -- fewer than three, even though the sequence is ten long. The growth is brutally slow: since $H_n \approx \ln n + 0.577$, doubling the data to $n = 20$ lifts the expected record count only to about $3.6$, and you would need roughly $n = 12{,}000$ values before you expected even *ten* records. Records get rare fast because beating the running maximum gets harder with every step. The intuition this teaches: when a count's per-position probability decays like $\tfrac{1}{k}$, the total grows only logarithmically -- "new highs" are inherently scarce. (This is exactly why a trading strategy hitting frequent fresh equity highs is so striking: under pure noise, records should be logarithmically rare.)

*Deeper mechanics, if the interviewer pushes.* There is a beautiful subtlety here that a sharp interviewer may probe: are the record indicators $R_1, \dots, R_n$ independent? Astonishingly, **yes** -- the events "position $k$ is a record" turn out to be mutually independent, even though each one is a statement about the same shared sequence. The reason is that $R_k$ depends only on the *relative rank* of the $k$-th value among the first $k$, and these relative-rank events for different $k$ carry non-overlapping information about the permutation. This is a genuinely rare gift: indicators in counting problems are *usually* dependent (recall how a fixed point in a shuffle changes everyone else's odds), and linearity is precisely the tool that lets us ignore that. Here, the bonus independence means we can go past the mean for free. Because independent indicators have variances that add, $\mathrm{Var}(X) = \sum_{k=1}^n \tfrac{1}{k}\left(1 - \tfrac{1}{k}\right) = H_n - H_n^{(2)}$, where $H_n^{(2)} = \sum 1/k^2$. For $n = 10$ this gives a variance of about $1.38$ and a standard deviation near $1.17$ -- so the record count clusters tightly around 2.93, almost always landing between 1 and 5. The transferable point: linearity gets you the mean with no independence at all, but on the rare occasion the indicators *are* independent, the very same decomposition hands you the variance too, for almost no extra work.

#### Worked example: expected total payout on a prize wheel

*A charity sets up a prize wheel divided into sectors. On any single spin the wheel pays \$0 with probability $\tfrac{1}{2}$, \$10 with probability $\tfrac{1}{4}$, \$50 with probability $\tfrac{1}{8}$, and \$100 with probability $\tfrac{1}{8}$. A donor buys **20 spins**. What is the expected total payout? And -- the part the interviewer is really probing -- does your answer change if the spins are rigged to be correlated (a sticky wheel that tends to repeat its last result)?*

**Technique: pure linearity of expectation.** This problem is engineered to test whether you reach for the joint distribution when you do not have to. The total is a *sum* of per-spin payouts, which is the unmistakable signal for linearity.

1. First find the expected payout of a *single* spin. It is a plain weighted average:

$$\mathbb{E}[\text{one spin}] = 0\cdot\tfrac{1}{2} + 10\cdot\tfrac{1}{4} + 50\cdot\tfrac{1}{8} + 100\cdot\tfrac{1}{8} = 0 + 2.5 + 6.25 + 12.5 = 21.25.$$

So one spin is worth **\$21.25** on average. (Sanity check: the probabilities $\tfrac{1}{2} + \tfrac{1}{4} + \tfrac{1}{8} + \tfrac{1}{8} = 1$, as they must.)

2. Let $P_1, P_2, \dots, P_{20}$ be the payouts of the 20 spins. The total is $T = P_1 + P_2 + \dots + P_{20}$. By linearity,

$$\mathbb{E}[T] = \mathbb{E}[P_1] + \mathbb{E}[P_2] + \dots + \mathbb{E}[P_{20}] = 20 \times 21.25 = 425.$$

The expected total payout is **\$425**.

3. Now the trap, sprung. Does correlation between spins change this? **No.** Linearity of expectation never required the spins to be independent -- it holds for *any* dependence whatsoever. A "sticky" wheel that tends to repeat its last sector would make the total payout far more *variable* (long runs of \$0 or long runs of \$100 become common, fattening both tails), but each spin's *marginal* distribution is still that same four-way split, so each $\mathbb{E}[P_i]$ is still \$21.25, and the sum is still \$425. The dependence reshapes the spread of $T$ completely while leaving its mean untouched -- exactly the lesson from the two glued dice, now in dollars.

The clean move was refusing to model how the spins relate to each other at all: a total's expected value is just the sum of the parts' expected values, and that is true no matter how tangled the parts are. The intuition this teaches: for an *expected total payout*, dependence is a red herring -- compute one part's average and multiply (or add) -- and only when the question turns to risk, variance, or the chance of a bad streak does the correlation finally matter.

To make the "variance changes but the mean does not" claim concrete, picture the two extremes the interviewer might describe. If the 20 spins are *independent*, the total clusters fairly tightly around \$425, and the outcome is rarely far from there. If instead the wheel is *perfectly* sticky -- the first spin's sector locks in and all 20 spins repeat it -- then the total is just $20$ copies of one draw: it is \$0 with probability $\tfrac{1}{2}$, \$200 with probability $\tfrac{1}{4}$, \$1000 with probability $\tfrac{1}{8}$, and \$2000 with probability $\tfrac{1}{8}$. That is a wildly spread-out, all-or-nothing payout. Yet its mean is $0\cdot\tfrac{1}{2} + 200\cdot\tfrac{1}{4} + 1000\cdot\tfrac{1}{8} + 2000\cdot\tfrac{1}{8} = 50 + 125 + 250 = 425$ -- identical to the independent case, to the dollar. Two payouts that could hardly look more different share the exact same expected value, which is the whole point: linearity reads only the marginals.

## Common misconceptions

**"The expected value is the most likely outcome."** No -- it is the probability-weighted average, and it routinely lands on a value that can never occur (a die's 3.5; the seesaw's 1.4). The most likely outcome is the *mode*; the average is the *mean*. Interviewers test the distinction on purpose.

**"Linearity of expectation needs the variables to be independent."** This is the single most common error, and getting it right is a strong positive signal. Linearity holds *unconditionally* -- the pieces can be arbitrarily dependent. (It is *variance* that requires independence to add simply; expectation never does.) When a candidate hesitates to use linearity because "the events aren't independent", they have misremembered the rule.

**"Two-flip patterns are all equally hard to wait for."** HH and HT both have probability $\tfrac{1}{4}$ at any fixed pair of positions, yet $\mathbb{E}[\text{HH}] = 6 \neq 4 = \mathbb{E}[\text{HT}]$. Overlap structure -- how much progress a mismatch destroys -- changes the waiting time even when the single-window probability is identical.

**"A positive expected value means you'll come out ahead."** The doubling gambler has expected wealth exactly \$1 every round, yet is almost surely wiped out. A high mean can be carried entirely by a rare jackpot. This is why expectation alone is never enough to size a bet -- you also need the variance, the tail, and the chance of ruin.

**"The birthday answer is 23, full stop."** 23 answers "how many people make a shared birthday *more likely than not*" -- a probability-threshold question. The *expected* number of people until the first collision is about 24.6. Different question, different number; the words "on average" versus "more likely than not" tell you which.

**"You should always keep drawing if the next draw could be higher."** No -- you keep drawing only if the next draw is higher *in expectation* than what you hold. With a die, you reroll a 3 (since $3 < 3.5$) but keep a 4 (since $4 > 3.5$), even though a reroll *could* land on 6. Optimal stopping compares expectations, not best cases.

**"More data should mean lots more record-highs."** It barely does. The expected number of records in $n$ random values is the harmonic number $H_n \approx \ln n$, so it grows *logarithmically*, not linearly -- ten values give about 2.93 records, a hundred give about 5.2, a thousand about 7.5. Each new observation has only a $\tfrac{1}{k}$ chance of beating the running maximum, so fresh highs become exponentially rarer to come by. Candidates who guess the count grows in proportion to $n$ are off by an entire functional form; the indicator-plus-symmetry argument pins it to $H_n$ exactly.

**"Correlated payouts must have a different expected total than independent ones."** They have the same expected total -- linearity does not see correlation at all. The prize-wheel total is \$425 whether the spins are independent or a sticky wheel that clusters its outcomes. What correlation changes is the *variance* and the tail of the total (long runs of one outcome become likely), never the mean. Conflating "the total behaves very differently" with "the total *averages* differently" is the same independence confusion that trips people up on linearity, now wearing a payout costume.

## How it shows up in real trading and research

These are not just puzzles. The same four techniques are load-bearing in how trading firms actually price, hedge, and size risk.

**Pricing a derivative is computing an expected value.** The fair price of a financial option is, under the right probability measure, the *expected* discounted payoff. The entire edifice of [Black-Scholes option pricing](/blog/trading/quantitative-finance/black-scholes) is a way of computing one expectation; firms that get the expectation slightly more right than competitors capture the edge. When the payoff is a sum of pieces -- a basket option on many stocks, say -- linearity of expectation lets a desk value the basket's expected payoff as a combination of per-name expectations without ever modeling the full joint distribution of every stock at once.

**Counting events with indicators is everywhere in risk.** "Expected number of defaults in a bond portfolio this year" is exactly the fixed-points calculation in disguise: one indicator per bond, expected count equals the sum of default probabilities -- and, just as with shuffles, the defaults are *correlated* (a recession hits everyone), yet linearity gives the expected count regardless. "Expected number of limit orders that get filled", "expected number of days the strategy loses money", "expected number of venues where our quote is best" -- all indicator sums.

**Symmetry prices fairness and detects mispricing.** Market makers constantly ask "by symmetry, what should these two related instruments be worth relative to each other?" If a call and a put that *should* be symmetric trade at asymmetric prices, that gap is a signal. The first-ace gap argument -- equal interchangeable pieces splitting a known total -- is the same reasoning that says two economically equivalent claims must carry equal value, and any deviation is an arbitrage to be hunted. See [options theory](/blog/trading/quantitative-finance/options-theory) for the put-call parity version of exactly this symmetry.

**First-step / recursive thinking underlies dynamic strategies.** Any "should I act now or wait?" decision -- exercise an American option early, accept a quote or wait for a better one, hold a position or close it -- is an optional-stopping problem solved by first-step reasoning: compare the value of acting now against the *expected* value of waiting one more step. The doubling-gambler lesson -- that a fine expected value can mask near-certain ruin -- is precisely why firms obsess over the *full distribution* and the chance of a wipeout, not just the mean. It is also why "expected value" and "risk" are taught together; a strategy with a great mean and a fat left tail is exactly the kind that blows up, as the [JPM argument for not holding 100% equities](/blog/trading/quantitative-finance/jpm-why-not-100-equities) illustrates from the portfolio side.

**Monte Carlo simulation is the brute-force fallback.** When a payoff is too tangled for any of the four techniques, desks estimate the expectation by simulating millions of random scenarios and averaging -- the definition of expected value, computed numerically. The techniques in this post are what let a quant *avoid* simulation when a closed-form answer exists, and *sanity-check* the simulation when it does not. A simulated price that disagrees with a linearity argument is a bug, almost every time.

## When this matters to you and where to go next

If you are interviewing, drill the recognition step until it is automatic: *sum or total* points to linearity, *expected number / count* points to indicators, *interchangeable positions or players* points to symmetry, *a repeating process* points to first-step recursion. Most expected-value interview problems are one of those four wearing a costume, and the candidates who get offers are the ones who undress the problem in the first ten seconds rather than grinding through the joint distribution.

The deeper payoff is conceptual. Expected value is the atom of quantitative finance -- prices are expectations, risk measures are expectations, edge is a difference of expectations. Getting fluent with these four techniques is not interview trivia; it is learning to *see* the averaging structure that the entire field is built on. From here, the natural next steps are to layer on **variance** (where independence finally does matter), the **risk-neutral measure** that makes [derivatives pricing](/blog/trading/quantitative-finance/derivatives-pricing) an expectation, and the way real desks turn these averages into positions through risk and sizing. The four levers you have here will keep showing up the whole way.

*This is educational material, not financial advice. Expected-value calculations describe averages over many repetitions; any single outcome can differ enormously from its mean, and a strong expected value never guarantees a good result on any particular bet.*

### Further reading

- [Black-Scholes option pricing](/blog/trading/quantitative-finance/black-scholes) -- the canonical example of pricing as a single expectation.
- [Options theory](/blog/trading/quantitative-finance/options-theory) -- put-call parity as a symmetry argument.
- [Derivatives pricing](/blog/trading/quantitative-finance/derivatives-pricing) -- how the risk-neutral measure turns prices into expectations.
- [Why not 100% equities](/blog/trading/quantitative-finance/jpm-why-not-100-equities) -- the portfolio-side reminder that a good mean is not the whole story.
