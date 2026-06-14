---
title: "The classic quant probability problem set: a technique-first walkthrough"
date: "2026-06-14"
publishDate: "2026-06-14"
description: "A curated, technique-organized tour of the canonical probability problems that recur in quant interviews — counting, conditioning, expected-value tricks, Markov first-step, and order statistics — each solved in full with the transferable pattern called out."
tags:
  [
    "quant-interviews",
    "probability",
    "brain-teasers",
    "expected-value",
    "bayes-theorem",
    "markov-chains",
    "combinatorics",
    "order-statistics",
    "interview-prep",
    "quantitative-finance",
  ]
category: "trading"
subcategory: "Quantitative Finance"
author: "Hiep Tran"
featured: true
readTime: 42
---

> [!important]
> **TL;DR** — Behind dozens of famous quant-interview probability problems sit only about five techniques. If you can name the technique a problem is testing, you have already half-solved it.
>
> - The whole canon (the "Green Book" by Xinfeng Zhou, "Heard on the Street," "Fifty Challenging Problems in Probability") reduces to five tools: **counting**, **conditioning / Bayes**, **expected-value tricks** (mostly linearity and optimal stopping), **Markov / first-step analysis**, and **order statistics on the continuous line**.
> - The four numbers worth memorizing as anchors: re-roll a die and the EV is **4.25**; it takes **14.7** rolls on average to see all six faces; a positive test for a 1%-prevalence disease is only about **17%** likely to be real; and a randomly broken stick forms a triangle with probability **1/4**.
> - Interviewers grade the *path*, not the final number. A spoken solution that walks **restate → enumerate → set up → compute → sanity-check** earns partial credit even when the arithmetic slips; a silent "the answer is 4.25" that is wrong earns nothing.
> - The single most common trap is reusing one symbol for two different quantities — that is exactly what makes the two-envelope "always switch" argument feel correct and be wrong.
> - Every technique here is developed in depth in its own companion post; this is the map that tells you which one a given problem belongs to.

Why does a fair, six-sided die that you may re-roll once have an expected value of exactly 4.25, and not, say, 4? Why is a person who tests positive for a rare disease — on a test that is "99% accurate" — still probably healthy? Why can 100 prisoners, each allowed to open only half the boxes in a room, escape execution about **31%** of the time when random guessing gives them a probability with thirty zeros after the decimal point?

These are not trick questions in the gotcha sense. They are the standard probability problems that recur, year after year, across the interview loops at Jane Street, Citadel, Two Sigma, Optiver, SIG, Hudson River Trading, Jump, and DE Shaw. They feel like a grab-bag of unrelated puzzles. They are not. Almost every one of them is a thin disguise over one of a handful of techniques, and the entire skill of "being good at brain teasers" is really the skill of **recognizing, fast, which technique a problem is built on**. Once you name the technique, the problem usually solves itself.

![Technique to problems map: five core techniques unlock the whole canon of quant probability brain teasers.](/imgs/blogs/classic-quant-probability-problems-1.png)

The diagram above is the mental model for this entire post. On the left are the five techniques. On the right are the famous problems. Every arrow says "this tool unlocks this problem." Memorize the *left column* and the right column stops being scary, because a new problem you have never seen before is almost always a recombination of tools you already have.

This post is a technique-first walkthrough. We will not march through problems alphabetically; we will group them by the tool they teach, solve two or three canonical examples in full for each tool, and after each one state the **transferable lesson** — the sentence you carry to the next problem. By the end you will have worked through eleven of the most-asked problems in the canon and, more importantly, you will have a checklist for classifying the twelfth.

## Foundations: how interviewers actually grade these

Before any problem, a word on what is actually being measured, because it changes how you should solve everything below.

A quant interview "brain teaser" is not a test of whether you have memorized the answer. The interviewer almost certainly knows you might have seen the problem before; many of these are famous. What they are testing is **how you think when you do not immediately know the answer** — because on a trading desk or a research team, the interesting problems are by definition the ones nobody has solved yet. So the grading rubric is mostly about your process.

Let me define the few terms we will lean on throughout, building from zero so nothing is assumed.

- **Sample space** — the complete list of everything that could happen. If you roll one die, the sample space is the six outcomes {1, 2, 3, 4, 5, 6}. Half of all probability mistakes are really mistakes about what the sample space is.
- **Event** — a subset of the sample space you care about. "The die shows an even number" is the event {2, 4, 6}.
- **Probability** — for equally likely outcomes, it is just `(number of outcomes in your event) / (total number of outcomes)`. Rolling even is 3/6 = 1/2. When outcomes are *not* equally likely, you weight each by its own probability and add.
- **Expected value (EV)** — the long-run average. You compute it by multiplying each outcome's value by its probability and summing: `EV = Σ value × probability`. For one fair die, `EV = (1+2+3+4+5+6)/6 = 3.5`. The expected value is often *not* a value you can ever actually get — you will never roll a 3.5 — and that is fine; it is an average.
- **Conditional probability** — the probability of A *given that* B already happened, written `P(A | B)`. Conditioning shrinks the sample space down to only the outcomes where B is true, then re-measures A inside that smaller world. This is where most of the famous "paradoxes" live.
- **Independence** — two events are independent if knowing one tells you nothing about the other. Coin flips are independent; drawing cards without replacement is not.

With that vocabulary, here is the spoken-solution discipline that earns marks. The interviewer is listening for five moves, in order:

1. **Restate** the problem in your own words and pin down every assumption. ("So the die is fair, and I get exactly one optional re-roll, and I have to commit to keep-or-reroll *before* I see the second roll — is that right?")
2. **Enumerate** — name the technique you think applies and lay out the sample space or the states. ("This is an optimal-stopping problem; the states are the value of the first roll.")
3. **Set up** the governing equation — linearity of expectation, a first-step recursion, Bayes' rule, a counting ratio.
4. **Compute** the arithmetic slowly, narrating each number as you write it.
5. **Sanity-check** — units, bounds (a probability must be between 0 and 1), limiting cases, and "does this beat a naive guess?"

The single highest-leverage habit is the **sanity check**, because it catches the errors that lose offers. A probability of 1.3 is impossible. An expected value below the worst outcome or above the best outcome is impossible. If your "clever" strategy does not beat random guessing, you have made an error. Train yourself to run these checks automatically and out loud — they signal exactly the kind of disciplined, self-correcting reasoning a desk wants.

Now to the techniques. Each section below develops one tool; if you want the deep theory behind a tool rather than the worked problems, follow the cross-links to its dedicated companion post.

## Technique 1 — Counting (combinatorics)

The first tool is the oldest: **carefully count the outcomes**. Most "what is the probability" questions on a finite sample space are really "count the favorable cases, count the total cases, divide." The whole art is counting *correctly* — without double-counting and without missing cases. The deep treatment of permutations, combinations, and the inclusion-exclusion principle lives in [counting and combinatorics for quant interviews](/blog/trading/quantitative-finance/counting-combinatorics-quant-interviews); here we use it on the single most famous counting problem in the canon.

### The birthday problem

> **Problem.** In a room of 23 people, what is the probability that at least two of them share a birthday? (Ignore leap years and assume birthdays are uniformly spread over 365 days.)

Almost everyone's gut says "23 people, 365 days — must be tiny, maybe a few percent." The gut is badly wrong, and *why* it is wrong is the lesson.

**Identify the technique.** This is counting, with one crucial move: **count the complement.** "At least two share" is a messy event — two people, or three people, or two separate pairs, and so on. Its complement, "everybody's birthday is different," is a single clean event you can count directly. Whenever the event you want is a tangle of cases but its opposite is simple, compute the opposite and subtract from 1.

#### Worked example: 23 people, just over a coin flip

**Solve it step by step.** Line the 23 people up and bring them in one at a time, asking each to have a birthday that collides with nobody already seated.

- Person 1 can have any birthday: probability 365/365 = 1.
- Person 2 must dodge person 1's one birthday: probability 364/365.
- Person 3 must dodge two taken birthdays: 363/365.
- ... and so on down to person 23, who must dodge 22 taken birthdays: 343/365.

Because each person's birthday is independent, the probability that *all 23 are distinct* is the product:

$$P(\text{all distinct}) = \frac{365}{365}\cdot\frac{364}{365}\cdot\frac{363}{365}\cdots\frac{343}{365} = \prod_{k=0}^{22}\frac{365-k}{365}.$$

Multiply it out and you get about **0.493**. Therefore:

$$P(\text{at least one shared birthday}) = 1 - 0.493 = 0.507 \approx 51\%.$$

**Sanity check.** Just over half — surprisingly high, but it should be high, and here is the intuition that fixes the gut. The relevant quantity is not "how many people" but "how many *pairs* of people," because a collision needs a pair. With 23 people there are `C(23, 2) = 23 × 22 / 2 = 253` pairs. Each pair has a 1/365 ≈ 0.27% chance of matching, and with 253 independent-ish chances at it, a match becomes likely. The number of pairs grows like the *square* of the headcount, which is why the answer climbs so fast.

> **Transferable lesson.** When the event you want is a union of many overlapping cases ("at least one ..."), count the complement instead, and look for the quantity that actually scales — here, *pairs*, not people. "At least one" almost always means "1 minus the probability of none."

### A counting cousin: the boy-girl seating idea

The same complement trick answers a whole family of "at least one" questions — at least one six in four die rolls, at least one ace in a five-card hand, at least one collision when you hash *n* keys into *m* buckets (the birthday problem in disguise, and a real concern in low-latency systems). The move is identical every time: `P(at least one) = 1 − P(none)`, and `P(none)` is a clean product. If an interviewer asks "what is the chance of at least one X in n tries," your hand should already be writing `1 − (1 − p)^n` before they finish the sentence.

## Technique 2 — Conditioning and Bayes

The second tool is **conditioning**: when the problem hands you information, you must shrink the sample space to only the outcomes consistent with that information, then re-measure. This is the home of the most notorious "paradoxes," all of which are really just people forgetting to shrink the sample space correctly. The full machinery — the law of total probability, Bayes' rule, the prosecutor's fallacy — is developed in [conditional probability and Bayes for quant interviews](/blog/trading/quantitative-finance/conditional-probability-bayes-quant-interviews). Here are the two problems that teach it.

### The two-children problem

> **Problem.** A family has two children. You are told that *at least one of them is a boy.* What is the probability that *both* are boys?

The seductive wrong answer is 1/2 ("the other child is equally likely a boy or a girl"). The right answer is **1/3**, and the gap between them is the entire point.

**Identify the technique.** Conditioning. The phrase "at least one is a boy" is a condition that selects which outcomes survive. Your job is to list the original equally likely outcomes, cross out the ones the condition kills, and re-measure inside what remains.

![Two-children conditioning: the wording of the condition decides which outcomes survive and sets the answer.](/imgs/blogs/classic-quant-probability-problems-9.png)

**Solve it step by step.** Order the children oldest-first. With no information, the four equally likely outcomes (each probability 1/4) are:

| Outcome | Older | Younger | Survives "at least one boy"? |
|---|---|---|---|
| BB | boy | boy | yes |
| BG | boy | girl | yes |
| GB | girl | boy | yes |
| GG | girl | girl | **no** — ruled out |

The condition "at least one boy" eliminates only GG. Three outcomes remain, each now equally likely with probability 1/3. Both-boys (BB) is one of those three:

$$P(\text{BB} \mid \text{at least one boy}) = \frac{1}{3}.$$

**The twist that proves you understand it.** Change the wording to "the *older* child is a boy." Now the surviving outcomes are only BB and BG — GB is killed too, because in GB the older child is a girl. That leaves two equally likely cases, so the answer becomes **1/2**. Same family, same "a boy is involved," completely different answer, purely because the *condition* changed which outcomes survive. The interviewer is testing whether you read the condition precisely or pattern-matched to a memorized number.

> **Transferable lesson.** Conditioning means: list the original equally likely outcomes, delete the ones the information rules out, and re-normalize over what is left. The exact phrasing of the condition is load-bearing — "at least one is a boy" and "the older is a boy" are different conditions with different answers.

### The disease-test problem (positive predictive value)

> **Problem.** A disease affects 1% of the population. A test is 99% sensitive (if you are sick, it catches it 99% of the time) and has a 5% false-positive rate (if you are healthy, it wrongly flags you 5% of the time). You test positive. What is the probability you actually have the disease?

This is the most important problem on this entire list, because the same structure governs spam filters, fraud alerts, anomaly detection on a trading system, and every "rare-event" classifier in finance. The intuitive guess — "the test is 99% accurate, so I'm 99% likely sick" — is catastrophically wrong. The real answer is about **17%**.

**Identify the technique.** Bayes' rule. The trap is that you are given `P(positive | sick)` but you want `P(sick | positive)` — and these are *not* the same number when the disease is rare. The cleanest way to handle it is not the formula but **natural frequencies**: stop thinking in percentages and count actual people.

![Disease test as natural frequencies: false positives swamp true positives when the disease is rare, so a positive means about 17%.](/imgs/blogs/classic-quant-probability-problems-10.png)

#### Worked example: the disease test in real people

Imagine a concrete population of **10,000 people** and just count.

- 1% are sick: that is **100 sick** people. The other **9,900 are healthy.**
- Of the 100 sick, the test catches 99%: **99 true positives.** (1 sick person is missed — a false negative.)
- Of the 9,900 healthy, the test wrongly flags 5%: `0.05 × 9,900 =` **495 false positives.**
- Total positives = 99 true + 495 false = **594 people** who test positive.

Now the question "given a positive, what is the chance you are sick?" is just: of the 594 positives, how many are actually sick?

$$\text{PPV} = \frac{99}{594} = 0.167 \approx 17\%.$$

So a positive test on this rare disease still leaves you about **83% likely to be healthy.** The same answer from Bayes' rule directly:

$$P(\text{sick}\mid +) = \frac{P(+\mid\text{sick})\,P(\text{sick})}{P(+)} = \frac{0.99 \times 0.01}{0.99\times 0.01 + 0.05\times 0.99} = \frac{0.0099}{0.0594} = 0.167.$$

The intuition the example teaches: when a condition is **rare**, the false positives — drawn from the huge healthy majority — easily outnumber the true positives drawn from the tiny sick minority, even with a "good" test. Base rates dominate.

> **Transferable lesson.** Never confuse `P(evidence | hypothesis)` with `P(hypothesis | evidence)`. When you are handed the first and asked for the second, reach for Bayes — and if the algebra feels slippery, drop to natural frequencies: pick a round population, count the four cells (true/false × positive/negative), and divide. This is the same reasoning a desk uses when an alert fires: a "5-sigma" anomaly on a system that runs millions of checks a day is mostly false alarms.

## Technique 3 — Expected-value tricks

The third tool is the workhorse of the whole canon: clever ways to compute an expected value without grinding through the full distribution. Two sub-tricks dominate — **linearity of expectation** (the EV of a sum is the sum of the EVs, *even when the parts are dependent*) and **optimal stopping** (when you may quit or continue, compare the value of stopping now to the value of playing on). The deeper toolbox — indicator variables, the tail-sum formula, Wald's identity — is in [expected-value techniques for quant interviews](/blog/trading/quantitative-finance/expected-value-techniques-quant-interviews). Here are the two problems that anchor it.

### The re-roll die (optimal stopping)

> **Problem.** You roll a fair six-sided die. After seeing the result you may either keep it, or re-roll once and keep the second roll (no further choice). You win the dollar value of the die you keep. Played optimally, what is the expected value?

**Identify the technique.** Optimal stopping. At the decision point you compare two quantities: the value in hand (the first roll) versus the **continuation value** — what you expect to get if you throw it away and re-roll. The rule writes itself: keep the first roll if it beats the continuation value, otherwise re-roll.

#### Worked example: the optimal one-re-roll policy

**Solve it step by step.** If you re-roll, the second roll is a plain fair die with expected value `(1+2+3+4+5+6)/6 = 3.5`. So the continuation value is **3.5**. Therefore your rule is:

- Keep the first roll if it is **strictly greater than 3.5**, i.e. a 4, 5, or 6.
- Re-roll if it is 3.5 or below, i.e. a 1, 2, or 3.

(A first roll of 4 beats 3.5, so you keep a 4. A 3 is below 3.5, so you re-roll a 3.)

![Re-roll decision threshold: keep a 4, 5, or 6; re-roll a 1, 2, or 3; the optimal EV is 4.25.](/imgs/blogs/classic-quant-probability-problems-2.png)

Now compute the EV under this policy. Each first roll happens with probability 1/6. For a 1, 2, or 3 you re-roll and collect the continuation value 3.5. For a 4, 5, or 6 you keep the face value:

$$\text{EV} = \frac{1}{6}\Big(\underbrace{3.5 + 3.5 + 3.5}_{\text{re-roll a 1, 2, 3}} + \underbrace{4 + 5 + 6}_{\text{keep a 4, 5, 6}}\Big) = \frac{3.5\times 3 + 15}{6} = \frac{10.5 + 15}{6} = \frac{25.5}{6} = 4.25.$$

The optimal expected value is **4.25.**

**Sanity check.** It must lie between 3.5 (the value of never using the option — just keeping whatever you roll) and 6 (the best possible). 4.25 sits comfortably above 3.5, which confirms the re-roll option is worth something: exactly `4.25 − 3.5 = 0.75` dollars of added value from being allowed one do-over.

**The natural follow-up.** "Now you may re-roll *twice* (three rolls total, keep the last if you get there)." Same logic, one layer up: the continuation value of having two rolls left is the 4.25 you just computed. So on the first roll you keep anything above 4.25 — that is a 5 or 6 — and otherwise play the two-roll game worth 4.25:

$$\text{EV}_3 = \frac{1}{6}\Big(4.25 + 4.25 + 4.25 + 4.25 + 5 + 6\Big) = \frac{4.25\times 4 + 11}{6} = \frac{28}{6} \approx 4.67.$$

The threshold *rises* as you get more re-rolls, because the option to play on is worth more. This recursive "the continuation value of n rolls is the EV of (n−1) rolls" pattern is the heart of optimal stopping, and it generalizes directly to American-option exercise and to the secretary problem.

> **Transferable lesson.** When you may stop or continue, the decision rule is always the same: act now if and only if the value in hand beats the value of continuing optimally. Compute the continuation value first; it becomes your threshold. More chances to continue raise the threshold.

### Coupon collector (linearity of expectation)

> **Problem.** You roll a fair die repeatedly. On average, how many rolls does it take to see *all six* faces at least once?

The brute-force approach — track which faces you have, sum over a sprawling tree of possibilities — is a nightmare. Linearity of expectation turns it into grade-school arithmetic.

**Identify the technique.** Break the journey into **stages** and add up the stages. Going from "I have collected *k* distinct faces" to "I have collected *k*+1" is its own little sub-game, and the expected length of each sub-game is easy. Then linearity of expectation lets you simply *add* the expected stage lengths to get the expected total — and crucially, this works even though the stages are not independent in the obvious sense, because expectation always adds.

The one fact you need: if each roll succeeds with probability *p*, the expected number of rolls until the first success is **1/p**. (This is a geometric distribution; intuitively, if something happens 1-in-3 of the time, you wait about 3 tries.)

![Coupon collector stages: each new face is harder to hit, and the waiting times sum as a harmonic series to 14.7.](/imgs/blogs/classic-quant-probability-problems-8.png)

**Solve it step by step.** Suppose you already have *k* distinct faces. The probability that the next roll shows a *new* face is `(6 − k)/6`, since `6 − k` of the six faces are still missing. So the expected number of rolls to collect that next new face is the reciprocal, `6/(6 − k)`:

- From 0 faces → 1st face: every roll is new, `p = 6/6`, expected `6/6 = 1.00` roll.
- From 1 → 2: `p = 5/6`, expected `6/5 = 1.20` rolls.
- From 2 → 3: `p = 4/6`, expected `6/4 = 1.50` rolls.
- From 3 → 4: `p = 3/6`, expected `6/3 = 2.00` rolls.
- From 4 → 5: `p = 2/6`, expected `6/2 = 3.00` rolls.
- From 5 → 6: `p = 1/6`, expected `6/1 = 6.00` rolls.

By linearity of expectation, the expected total is just the sum:

$$\text{E[total]} = 1 + 1.2 + 1.5 + 2 + 3 + 6 = 14.7 \text{ rolls}.$$

**The general formula.** Notice the pattern: the total is `6 × (1/6 + 1/5 + 1/4 + 1/3 + 1/2 + 1/1)`, which is `6 × H₆` where `Hₙ` is the *n*-th harmonic number. For *n* coupons:

$$\text{E[total]} = n\left(1 + \frac{1}{2} + \frac{1}{3} + \cdots + \frac{1}{n}\right) = n\,H_n \approx n\ln n + 0.577\,n.$$

The last term reveals the punchline: collecting *n* coupons takes roughly `n ln n` tries, *not* `n`. That extra `ln n` factor — the long, painful wait for the final few faces — is why loyalty-card and gacha-game "collect them all" promotions feel so much grindier than you would naively expect.

> **Transferable lesson.** When a total is a sum of pieces, expected value adds — *always*, dependence or not. Decompose into stages (or into indicator variables, one per item), find each piece's small expectation, and sum. Linearity of expectation is the single most powerful EV trick in the canon.

## Technique 4 — Markov chains and first-step analysis

The fourth tool handles processes that **wander between states** — random walks, gambling games, queues. The key idea is **first-step analysis**: write the unknown quantity (a probability or an expected time) at a state in terms of its value one step later, producing a small system of equations you solve. The full theory of states, transition matrices, hitting times, and stationary distributions is in [Markov chains and hitting times for quant interviews](/blog/trading/quantitative-finance/markov-chains-hitting-times-quant-interviews). Two problems anchor it.

### Gambler's ruin (and the drunkard near a cliff)

> **Problem.** You start with \$3. You repeatedly bet \$1 on a fair coin: win, you gain \$1; lose, you lose \$1. You stop when you either reach \$5 (you win) or hit \$0 (you are ruined). What is the probability you reach \$5 before going broke? And how many bets does it take on average?

**Identify the technique.** First-step analysis on a random walk between two absorbing barriers (\$0 and \$5 are "walls" — once you hit one, you stop). Let `P(i)` be the probability of reaching \$5 starting from \$i. The walk is symmetric, so from any interior state you go up or down with probability 1/2 each.

![Gambler's ruin: between a \$0 wall and a \$5 wall, your win probability is your fraction of the distance to the target.](/imgs/blogs/classic-quant-probability-problems-7.png)

#### Worked example: \$3 toward \$5 on a fair coin

**Solve it step by step.** The first-step equation says: from state *i*, with probability 1/2 you move to *i*+1 and with probability 1/2 to *i*−1, so

$$P(i) = \tfrac{1}{2}P(i+1) + \tfrac{1}{2}P(i-1),$$

with boundary conditions `P(0) = 0` (already ruined) and `P(5) = 1` (already won). That equation says each `P(i)` is the average of its neighbors — which forces `P` to be a **straight line** in *i*. The line from `P(0)=0` to `P(5)=1` is simply `P(i) = i/5`. Starting from \$3:

$$P(\text{reach \$5}\mid \text{start at \$3}) = \frac{3}{5} = 60\%.$$

For a fair game the general rule is beautifully simple: starting at *i* with target *N*, your probability of winning is **`i/N`** — your fraction of the distance to the target.

**Expected number of bets.** A parallel first-step equation for the expected duration `D(i)` (each step adds 1 to the count) gives `D(i) = i(N − i)`. From \$3 toward \$5: `D = 3 × (5 − 3) = 6` bets on average before the game ends.

**The drunkard near a cliff.** Now the famous limiting case. A drunk stands one step from the edge of a cliff. Each step he moves toward the cliff or away from it with probability 1/2. Will he fall? Here the "win" barrier is infinitely far away (`N → ∞`) and the "ruin" barrier (the cliff) is one step away (`i = 1`). The survival probability is `i/N = 1/N → 0`. So on an infinite line, a symmetric random walker hits any fixed point with probability **1** eventually — the drunkard falls off the cliff with certainty. (He may wander away for a very long time first; the *expected* number of steps to fall is actually infinite. He will fall — you just cannot say when.)

> **Transferable lesson.** For a process that moves between states, write the unknown at each state in terms of one step ahead, add the boundary conditions, and solve the small linear system. For a *fair* random walk between two barriers, the win probability is linear — your fraction of the distance — and you should be able to write `i/N` on sight.

### The ants on a stick

> **Problem.** Several ants are placed at random points on a 1-meter stick, each facing a random direction, all moving at the same speed of 1 cm/s. When two ants collide they instantly reverse direction. An ant that reaches an end falls off. What is the **longest** possible time before *all* ants have fallen off?

This one looks like it needs you to track a combinatorial explosion of collisions. It does not, and the trick that collapses it is one of the most beautiful in the canon.

**Identify the technique.** A change of perspective that turns a hard tracking problem into a trivial one — the "relabeling" or "pass-through" trick. It is adjacent to Markov/state reasoning in that it is about realizing the *state you were tracking does not matter.*

![Ants on a stick: a collision-and-reverse is indistinguishable from the ants passing through each other.](/imgs/blogs/classic-quant-probability-problems-4.png)

**Solve it step by step.** Here is the key observation. When two ants collide and reverse, picture instead that they **pass through each other** and keep going. From a bird's-eye view of the *set of positions*, these two scenarios are completely indistinguishable — the only difference is which name-tag is on which moving dot. Two ants bouncing off each other produce exactly the same picture of moving points as two ants walking straight through each other; you just swapped their labels at the collision instant.

But ants that pass through each other are trivial: each one walks in a straight line at constant speed until it reaches an end. The farthest any single ant can possibly have to travel is the **full length of the stick** — an ant at the very left edge facing right (or vice versa) must traverse all 1 meter. At 1 cm/s, that takes:

$$t_{\max} = \frac{\text{length}}{\text{speed}} = \frac{100 \text{ cm}}{1 \text{ cm/s}} = 100 \text{ seconds}.$$

No configuration of ants and no number of collisions can take longer than 100 seconds, because in the pass-through picture every walker is gone within length/speed, and the bouncing picture has the *same* set of exit times. The collisions are a red herring; they only relabel which ant is which, never how long the last point survives.

> **Transferable lesson.** When identical particles interact by exchanging momentum, "they collide and reverse" is the same as "they pass through each other" — you only swapped indistinguishable labels. More broadly: if you are tracking individual identities through a hard process, ask whether the identities even matter to the question. Often they do not, and the problem collapses.

## Technique 5 — Order statistics and continuous tricks

The final tool moves from dice and coins to the **continuous** world — picking real numbers uniformly from an interval, where the questions become geometric. The central skills are **order statistics** (when you drop *n* random points on [0,1], what is the expected position of the smallest, the largest, the spacing between them?) and **translating a probability into the area of a region.** The full treatment — uniform order statistics, the `k/(n+1)` spacing rule, geometric probability — is in [order statistics and uniform tricks for quant interviews](/blog/trading/quantitative-finance/order-statistics-uniform-tricks-quant-interviews). The defining problem:

### The broken stick (triangle problem)

> **Problem.** You break a stick of length 1 at two points chosen uniformly and independently along its length, producing three pieces. What is the probability the three pieces can form a triangle?

**Identify the technique.** Geometric probability. Two uniform break points `(x, y)` are a single random point in the **unit square** `[0,1] × [0,1]`. "Can form a triangle" carves out a region of that square, and the probability is just the *area* of that region. The trick is finding which region.

![Broken stick triangle region: the triangle inequality carves the unit square into a green quarter, giving probability 1/4.](/imgs/blogs/classic-quant-probability-problems-11.png)

#### Worked example: two uniform breaks, area = 1/4

**Solve it step by step.** Three lengths `a`, `b`, `c` form a triangle if and only if no single piece is longer than the other two combined — the **triangle inequality**. Since the three pieces sum to 1, "no piece exceeds the sum of the others" is exactly the same as "**no piece exceeds 1/2**." If any piece is longer than half the stick, the other two cannot reach across to meet it.

So we need all three pieces to be under 1/2. Lay the two break points as a point `(x, y)` in the unit square. Work out which `(x, y)` give all-pieces-under-half, and the allowed region turns out to be exactly the central region with **area 1/4** of the square. (Each way a piece can exceed 1/2 cuts off a corner triangle of area 1/8; the failing corners total 3/4, leaving 1/4 valid.) Therefore:

$$P(\text{three pieces form a triangle}) = \frac{1}{4} = 25\%.$$

**The variant that catches people.** A different — and very common — phrasing is: "break the stick once at random, then break the *longer* piece at random." That is a *different* random model (the second break is no longer uniform over the whole stick), and it yields a different answer (it works out to `ln 2 − 1/2 ≈ 0.193`, about 19.3%). The lesson the interviewer is probing: **the answer depends entirely on the breaking model**, so before computing you must nail down exactly how the points are chosen. Stating "I'll assume both break points are independent and uniform over the whole stick — is that right?" is precisely the kind of clarifying move that earns marks.

> **Transferable lesson.** A probability over continuous uniform choices is the *area* (or volume) of the favorable region. Translate the condition into inequalities, draw the region in the unit square, and measure it. And always confirm the sampling model — "uniform over the whole stick" versus "break the longer piece" are different problems with different answers.

## A bonus pair: state-tracking and game-theory induction

Two more canonical problems deserve a place because they teach moves that recur constantly: **state-tracking** (compressing a huge history into a tiny sufficient state) and **backward induction** (solving a multi-stage game from the end).

### The 100 prisoners and the boxes

> **Problem.** 100 prisoners are numbered 1 to 100. In a room are 100 boxes, each containing a slip with one prisoner's number, in random order. One at a time, each prisoner may open at most 50 boxes, looking for their own number, then leaves the room exactly as they found it (no communication). If *every* prisoner finds their own number, all go free; if even one fails, all are executed. They may agree on a strategy beforehand. What strategy maximizes their survival probability — and what is it?

If each prisoner opens 50 boxes at random, each succeeds with probability 1/2, and all 100 succeed with probability `(1/2)^100 ≈ 8 × 10⁻³¹` — effectively zero. Astonishingly, a clever strategy lifts this to about **31%**.

![Prisoners and boxes: follow the permutation cycle starting at your own number and you win whenever the longest cycle is short.](/imgs/blogs/classic-quant-probability-problems-3.png)

**Identify the technique.** State-tracking via permutation cycles, plus a counting argument. The boxes-and-slips arrangement is a **permutation**: box *i* contains some slip, which points to another box, and so on. A permutation decomposes into **cycles**.

**The strategy.** Each prisoner starts by opening *the box with their own number.* Whatever slip is inside points to the next box to open; they follow that chain. Because the prisoner started on their own number, they are walking the cycle that *contains* their number — and that cycle is guaranteed to loop back to their own slip. So a prisoner finds their number **if and only if the cycle containing it has length ≤ 50.** Every prisoner succeeds simultaneously exactly when the permutation has **no cycle longer than 50.**

**Why it works out to ~31%.** A random permutation of 100 elements has a cycle longer than 50 with probability `1/51 + 1/52 + ... + 1/100 ≈ ln 2 ≈ 0.693` (at most one such long cycle can exist, which is what makes the sum clean). So the survival probability is `1 − 0.693 ≈ 0.307`, about **31%** — independent of the number of prisoners, even as it grows. The strategy converts 100 *independent* coin-flip-like fates into 100 fates that all rise and fall together on a single event (the longest cycle), and correlated success is far more valuable than independent success when you need *everyone* to win.

> **Transferable lesson.** Look for the hidden structure that lets you compress the problem — here, the permutation-cycle view turns 100 separate searches into one question about the longest cycle. And when you need *all* of many events to succeed, engineer *correlation* between them: making fates rise and fall together beats making each independently likely.

### The pirates dividing gold

> **Problem.** Five rational, greedy pirates (rank them A, B, C, D, E from senior to junior) must split 100 gold coins. The most senior surviving pirate proposes a split; all surviving pirates vote; if at least half approve, it passes; otherwise the proposer is thrown overboard and the next-senior proposes. Each pirate's priorities, in order: (1) survive, (2) maximize their own gold, (3) all else equal, prefer to throw someone overboard. What does A propose?

This looks like it needs game-theoretic genius. It needs only **backward induction**: solve the simplest version (the endgame), then work upward.

![Pirates and gold: solve the two-pirate base case, then induct forward to find the senior pirate keeps 98 coins.](/imgs/blogs/classic-quant-probability-problems-12.png)

**Identify the technique.** Backward induction — collapse a multi-stage game from the *end*, where the answer is trivial, and let each earlier stage be solved using only the stage below it.

**Solve it step by step**, starting from the smallest subgame:

- **2 pirates (D, E):** D proposes. He votes for himself — 1 of 2 votes, which is "at least half," so it passes. D takes everything. Result: **D: 100, E: 0.** Crucially, **E gets nothing** if it ever comes down to two.
- **3 pirates (C, D, E):** C needs 2 of 3 votes (his own plus one more). E knows that if C is thrown overboard, the game becomes the 2-pirate case where E gets 0. So C can buy E's vote with just **1 coin** — better for E than the 0 he would get otherwise. Result: **C: 99, D: 0, E: 1.**
- **4 pirates (B, C, D, E):** B needs 2 of 4 votes (himself plus one). In the 3-pirate fallback, *D* gets 0. So B buys D with **1 coin**. Result: **B: 99, C: 0, D: 1, E: 0.**
- **5 pirates (A, B, C, D, E):** A needs 3 of 5 votes (himself plus two). In the 4-pirate fallback, *C and E* each get 0. So A buys both with **1 coin each**, securing their votes. Result: **A: 98, B: 0, C: 1, D: 0, E: 1.**

So the senior pirate A keeps **98 coins**, hands 1 each to C and E, and it passes 3 votes to 2. The counterintuitive lesson is that being senior is enormously powerful *because* the juniors can do the same backward reasoning and know exactly how little they will get if they refuse.

> **Transferable lesson.** Any finite multi-stage game collapses from the end. Solve the trivial base case, then at each earlier stage assume everyone reasons forward to the known outcomes below and acts accordingly. "Start from the end" is the universal move for sequential games — including, on a desk, anticipating how a counterparty will respond on the next round.

## The two-envelope paradox: why the naive EV is wrong

We close the problem tour with a "paradox" that is really a lesson about expected-value *bookkeeping*, because the error it teaches is the single most common one in the whole canon.

> **Problem.** Two sealed envelopes; one holds twice as much money as the other. You pick one and peek inside (or don't). A friend offers to let you switch to the other envelope. The "argument": let your envelope contain `X`. The other holds either `2X` or `X/2`, each with probability 1/2. So switching has expected value `½(2X) + ½(X/2) = 1.25X > X`. You should always switch! But by symmetry the same argument says you should switch *back*. Something is wrong. What?

![The two-envelope fallacy: the naive switch argument reuses one symbol X for two different dollar amounts.](/imgs/blogs/classic-quant-probability-problems-5.png)

**Identify the technique.** Careful expected-value setup — and the discovery of a hidden double-count. The wrong argument *feels* airtight, which is exactly why it is instructive.

**Where the argument breaks.** The fatal move is letting one symbol `X` stand for two genuinely different amounts. When the other envelope holds `2X`, your envelope is the *smaller* of the two; when the other holds `X/2`, your envelope is the *larger*. So `X` does not name a single, fixed quantity — it secretly means the small amount in one branch and the large amount in the other. You cannot factor it out of the expectation as if it were one number; doing so silently adds an amount to itself across two incompatible worlds. A random variable must mean the **same thing in every term** of an expectation, and `X` does not.

**The correct setup.** Name the *real* amounts. The pair is `(A, 2A)` for some fixed (unknown) `A`. You hold the small one `A` with probability 1/2, or the large one `2A` with probability 1/2.

$$\text{EV(switch)} = \tfrac{1}{2}(2A) + \tfrac{1}{2}(A) = \tfrac{3}{2}A, \qquad \text{EV(stay)} = \tfrac{1}{2}(A) + \tfrac{1}{2}(2A) = \tfrac{3}{2}A.$$

They are equal. There is no free lunch; switching and staying both average `1.5A`. The paradox dissolves the instant you stop reusing one symbol for two amounts.

**The tell you should have caught.** If switching were *always* worth +25%, you would switch forever and never open either envelope — an infinite loop. Any "strategy" that says "always do X again" without ever resolving is a flashing warning that the expected-value bookkeeping is broken.

> **Transferable lesson.** A random variable must denote the same quantity in every term of an expectation. The fastest way to debug a too-good-to-be-true EV is to replace the slippery symbol with the actual, fixed underlying amounts and recompute. If a strategy implies an infinite loop of "do it again," its EV is wrong.

## The think-out-loud framework

Every problem above shares a spoken-solution skeleton. Internalize it and you will perform far better than someone who knows more answers but narrates none of them, because the interviewer can only grade what you say out loud.

![The spoken-solution skeleton: interviewers grade the path — restate, enumerate, set up, compute, sanity-check — not just the final number.](/imgs/blogs/classic-quant-probability-problems-6.png)

1. **Restate.** Repeat the problem back and pin every assumption. "Fair die, one optional re-roll, I commit before seeing the second roll." This buys thinking time, surfaces ambiguities (the broken-stick model, the exact two-children wording), and shows you do not jump to conclusions — a trait that matters enormously when real money is on the line.
2. **Enumerate / name the technique.** Out loud, classify it. "This is optimal stopping." "This is Bayes with a rare base rate." "This is linearity of expectation." Naming the technique is the move this entire post is built to train, and saying it aloud lets the interviewer follow — and nudge — your reasoning.
3. **Set up.** Write the governing equation before plugging in numbers: the first-step recursion, the Bayes ratio, the sum of stage expectations, the counting fraction. A clean setup is most of the credit; the arithmetic is the easy part.
4. **Compute.** Do the algebra slowly and narrate each number. If you slip on a number, a clean setup means the interviewer sees the slip is arithmetic, not conceptual, and you keep the credit.
5. **Sanity-check.** Out loud: Is the probability in [0,1]? Is the EV between the worst and best outcomes? Does the strategy beat a naive guess? What happens in the limiting case (one prisoner, infinite barrier, zero base rate)? This final pass catches the errors that actually cost offers and, more than anything else, signals the disciplined self-correction a desk wants.

The contrast is stark and worth stating plainly: a candidate who blurts a single number and freezes when it is wrong gets no partial credit and no recovery path. A candidate who narrates the five steps earns the model even when a number slips, and gives the interviewer a place to nudge them back on track. **Think out loud — always.**

## Common misconceptions

A handful of wrong beliefs trip up most candidates. Each is worth correcting explicitly.

**"The expected value is the most likely outcome."** No — the EV is a long-run average and is frequently a value you can never actually realize. The re-roll game's EV is 4.25; you can never roll a 4.25. The EV is the center of mass of the distribution, not its peak (the mode) and not necessarily its middle (the median). Conflating these is a classic error.

**"99% accurate means a positive result is 99% reliable."** This is the base-rate fallacy, and it is the most expensive misconception on the list. When the underlying condition is rare, false positives drawn from the large healthy majority swamp true positives drawn from the tiny sick minority. A positive on a 1%-prevalence disease with this test means only ~17% chance of being sick. Always ask for the base rate before trusting a positive.

**"Independent events are 'due' to balance out."** The gambler's fallacy. A fair coin that landed heads five times in a row is *still* 50/50 on the next flip — the coin has no memory. The law of large numbers says the *proportion* converges over many flips, not that short-run streaks get "corrected." Confusing convergence-in-proportion with short-run compensation is how people lose money.

**"More information always makes a probability go up or always go down."** Conditioning can move a probability in either direction, and the *exact wording* of the information matters. "At least one child is a boy" gives 1/3; "the older child is a boy" gives 1/2. Same family, different condition, different answer. Read the condition with a lawyer's care.

**"A clever strategy can't change the odds in a symmetric setup."** The 100-prisoners problem is the rebuttal: random guessing gives `(1/2)^100`, while cycle-following gives ~31% — a factor of 10²⁹ improvement — without changing a single box. The strategy works by *correlating* the prisoners' fates so they succeed or fail together. When you need *all* of many events, engineered correlation is worth far more than independent likelihood.

**"If switching looks like +EV, switch."** Not if your EV calculation reused one symbol for two different amounts (the two-envelope trap). And not if the "strategy" implies an infinite regress of switching. A +EV that survives neither the relabeling test nor the infinite-loop test is a bookkeeping bug, not a free lunch.

## How these map to real desk reasoning

These are not parlor tricks. Each technique is a thinking habit that recurs on a real trading or research desk, which is precisely why firms screen for them.

**Bayesian updating is the entire job of a market maker.** A quote is a prior; every trade that hits it is evidence; the desk must update its estimate of fair value in real time. The disease-test reasoning — "a single positive signal against a strong prior base rate is mostly noise" — is exactly how a desk treats one anomalous print: a "5-sigma" move on a system running millions of checks a day is, by base rates, almost always a glitch or a fat finger, not a real signal. Overreacting to it is the base-rate fallacy with money attached.

**Linearity of expectation is how you price a basket without modeling correlations.** The expected payoff of a portfolio is the sum of the expected payoffs of its parts, *regardless of how the parts are correlated*. That lets a desk compute an expected P&L for a complex book by summing simple pieces — the same move that turned the coupon-collector nightmare into six lines of arithmetic. Correlations matter enormously for *risk* (the variance), but never for the *expected* total.

**Optimal stopping is the theory of when to exercise.** The re-roll threshold — "keep what you have only if it beats the value of playing on" — is literally the rule for exercising an American option early: exercise only when the immediate payoff exceeds the continuation value of holding. The same logic governs when to take a partial fill versus waiting for a better price, and when to close a position versus letting it run.

**First-step analysis is how you reason about path-dependent risk.** A position that gets liquidated if it touches a barrier, a strategy with a stop-loss, a fund facing a drawdown limit — all are random walks with absorbing barriers, and gambler's-ruin reasoning estimates the probability of ruin before the target. The blunt lesson of gambler's ruin — that a player with finite capital facing an infinitely-deep opponent is eventually ruined with certainty, even in a fair game — is the mathematical case for position-sizing and risk limits. Bet too large relative to your bankroll and the cliff finds you.

**Backward induction is how you anticipate a counterparty.** The pirate game is a toy model of any sequential negotiation or auction: to know what to do now, reason forward to how every later stage resolves, then act on those known endpoints. Desks use exactly this to model how an order will move a market across child slices, or how a counterparty will respond on the next round of a quote.

**The relabeling trick — "do the identities even matter?" — is a modeling reflex.** The ants problem teaches you to ask whether the complicated thing you are tracking actually affects the answer. On a desk this shows up as: do I need the full order book, or just the imbalance? Do I need every path, or just the terminal distribution? Throwing away irrelevant state is how hard problems become tractable.

## When this matters and further reading

If you are preparing for quant interviews, the highest-leverage thing you can do is stop memorizing answers and start memorizing the **five-tool map** at the top of this post. When a new problem lands, your first move is not to recall its solution — it is to classify it. "At least one ... → count the complement." "Given that ... → condition and shrink the sample space." "Expected number of ... → linearity, decompose into stages." "Moves between states ... → first-step analysis." "Uniform on an interval ... → area of a region." Classification is 90% of the speed, and speed under pressure is what the interview measures.

To go deeper on any single tool, follow its companion post in this series:

- [Counting and combinatorics for quant interviews](/blog/trading/quantitative-finance/counting-combinatorics-quant-interviews) — permutations, combinations, the complement trick, inclusion-exclusion, and the pairs-not-people intuition behind the birthday problem.
- [Conditional probability and Bayes for quant interviews](/blog/trading/quantitative-finance/conditional-probability-bayes-quant-interviews) — the law of total probability, Bayes' rule, natural frequencies, and the prosecutor's fallacy.
- [Expected-value techniques for quant interviews](/blog/trading/quantitative-finance/expected-value-techniques-quant-interviews) — linearity of expectation, indicator variables, the tail-sum formula, optimal stopping, and Wald's identity.
- [Markov chains and hitting times for quant interviews](/blog/trading/quantitative-finance/markov-chains-hitting-times-quant-interviews) — states, transition matrices, first-step analysis, expected hitting times, and stationary distributions.
- [Order statistics and uniform tricks for quant interviews](/blog/trading/quantitative-finance/order-statistics-uniform-tricks-quant-interviews) — uniform order statistics, the `k/(n+1)` spacing rule, geometric probability, and continuous-vs-discrete framing.

For the canonical books, the three that cover everything here and more:

- **Xinfeng Zhou, *A Practical Guide to Quantitative Finance Interviews*** (the "Green Book"). The single most-cited interview-prep book; its brain-teaser and probability chapters are the source for most problems above, with the same technique-first emphasis.
- **Timothy Falcon Crack, *Heard on the Street: Quantitative Questions from Wall Street Job Interviews***. The other classic; heavier on the spoken-reasoning and "explain your thinking" style that real interviews use.
- **Frederick Mosteller, *Fifty Challenging Problems in Probability with Solutions***. A slim, beautiful book of the original puzzles (the broken stick, the birthday problem, and many more) with elegant full solutions — the best place to build raw intuition.
- **Mark Joshi, *Quant Job Interview Questions and Answers***. Broader than just probability (it covers C++, derivatives pricing, and stochastic calculus too), but its probability section is excellent and oriented toward what desks actually ask.

The meta-lesson is the one worth carrying out of every interview and into the job itself: a hard problem you have never seen is almost always a familiar technique wearing a costume. Learn to see through the costume — name the technique — and you have already half-solved it.

*This is educational material about problem-solving technique, not financial advice.*
