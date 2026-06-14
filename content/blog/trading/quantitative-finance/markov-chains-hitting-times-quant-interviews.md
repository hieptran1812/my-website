---
title: "Markov chains and hitting times: gambler's ruin and random walks"
date: "2026-06-14"
publishDate: "2026-06-14"
description: "A from-scratch, interview-focused deep dive on Markov chains: the memoryless property, the simple random walk, gambler's ruin probabilities for fair and biased games, expected game length, absorbing chains and the fundamental matrix, expected hitting times, stationary distributions, and a full set of solved interview problems."
tags:
  [
    "markov-chains",
    "random-walk",
    "gamblers-ruin",
    "hitting-times",
    "quant-interviews",
    "first-step-analysis",
    "absorbing-states",
    "fundamental-matrix",
    "stationary-distribution",
    "expected-value",
    "probability",
    "quantitative-finance",
  ]
category: "trading"
subcategory: "Quantitative Finance"
author: "Hiep Tran"
featured: true
readTime: 40
---

> [!important]
> **TL;DR** — A huge fraction of quant-interview probability questions are secretly Markov chains: the moment you can phrase a problem as "I'm in some *state*, and where I go next depends only on where I am now," the answer is linear algebra, not cleverness.
>
> - A *Markov chain* is a system that moves between **states**, where the next state depends only on the current state, not on the whole history. That "no memory" rule is the **Markov property**.
> - Two questions cover most interviews: **"probability of reaching A before B"** and **"expected number of steps until something happens."** Both are solved by **first-step analysis** — write the unknown for each state as a probability-weighted average of the unknowns one step away, then solve the resulting linear system.
> - **Gambler's ruin** is the canonical example: a random walk between $0 (broke) and $N (target). For a **fair** game the probability of hitting $N first is just `i/N`; for a game with an **edge** `p`, ruin probability collapses *exponentially* in your bankroll.
> - The expected length of a *fair* gambler's-ruin game starting at `$i` toward `$N` is exactly **`i·(N−i)`** bets — 25 bets if you start at $5 with a $10 target.
> - Absorbing chains have a clean matrix recipe: the **fundamental matrix** `N = (I − Q)⁻¹` gives expected visits, and `N·1` gives expected steps to absorption. Memorize the setup; the linear algebra does the rest.

Here is a puzzle that gets asked, in some disguise, at almost every quant trading firm. You flip a fair coin over and over. How many flips, on average, until you see two heads in a row? Most people guess 4 (after all, the probability of HH on any given pair is 1/4, and 1/(1/4) = 4). The real answer is **6**. The off-by-50% gap between the naive guess and the truth is exactly the kind of thing interviewers at Jane Street, Citadel, Optiver, SIG, and Jump probe for: do you actually understand *dependence over time*, or are you pattern-matching to a formula?

![Define states, write what one step does, and the question becomes a linear system](/imgs/blogs/markov-chains-hitting-times-quant-interviews-1.png)

The diagram above is the mental model for this entire article. When a problem involves "doing something repeatedly and waiting for an outcome," you almost never need a clever combinatorial trick. You need four steps: (1) **list the states** you can be in, (2) **write down what one step does** — where you can go and with what probability, (3) **set up a recursion** that says "the answer for a state is the average of the answers for the states one step away," and (4) **solve the linear system**. That recipe — called *first-step analysis* — is the whole game. The coin problem, gambler's ruin, the frog hopping on lily pads, the drunk staggering home: all the same machine.

This piece builds the machine from zero. We will define what a state and a transition are, meet the *simple random walk*, derive gambler's ruin for both fair and biased games (with real numbers you can check), compute how long the game lasts, formalize the matrix recipe for absorbing chains, touch the stationary distribution for return-time problems, and then spend a long section in the *interview room* solving the kinds of problems you will actually be handed. By the end, "expected number of steps until X" and "probability of A before B" should feel less like puzzles and more like turning a crank.

A note on scope: this is educational material about probability and the math behind trading puzzles, not financial advice. Where the article touches betting or bankroll, it is describing mechanisms and how interviewers test them, not recommending you gamble.

## Foundations: states, the Markov property, and the transition matrix

Everything starts with three ideas. None of them require any background — if you can follow "I'm here, and from here I can go there" you can follow this whole section.

### What is a state?

A **state** is a complete description of the situation you are in, detailed enough that knowing it is *all* you need to predict what happens next. That last clause is the whole trick. A state is not just "where you are"; it is "everything that matters about where you are."

Some examples:

- In a gambling game, the state is **your current wealth** — say, `$7`. Nothing about how you got to $7 changes the odds of your next bet.
- For the coin problem "wait for HH," the state is **how much of the pattern you have already matched** — "no progress," "the last flip was an H," or "done." You do not need the full sequence of flips, just the relevant suffix.
- For a frog hopping between lily pads, the state is **which pad it is sitting on**.

Choosing the right state space is 80% of solving these problems, and it is the single skill interviewers are testing. Too coarse a state and the Markov property below fails; too fine and you drown in equations. The art is finding the *smallest* description that still makes the future depend only on the present.

### The Markov property (memorylessness)

A process has the **Markov property** — it is *memoryless* — if the probability of where it goes next depends **only on the current state**, not on the path that led there. Formally, if `X₀, X₁, X₂, …` is the sequence of states over time, then

$$ P(X_{n+1} = j \mid X_n = i, X_{n-1}, \dots, X_0) = P(X_{n+1} = j \mid X_n = i). $$

In plain English: *given the present, the past is irrelevant to the future.* The history can be thrown away once you know where you stand right now.

Why does this matter so much? Because it is what makes the linear system in step (3) finite and solvable. If the future depended on the entire history, the "state" would be the whole past sequence and there would be infinitely many of them. Memorylessness collapses that to a manageable handful of states.

A quick sanity check on what counts as Markov. "My wealth after the next bet" is Markov *if your bet size depends only on current wealth*. But "the temperature tomorrow given today" is only approximately Markov — real weather depends on more than one day. And "the next card from a shuffled deck" is **not** Markov if your state is just "which card I drew last," because the deck has memory (drawn cards are gone). You would have to make the state "the full set of remaining cards" to recover the Markov property. The skill is recognizing when you have packed enough into the state.

### The transition matrix

Once states are fixed, a Markov chain is completely described by its **transition probabilities**: for every pair of states `i` and `j`, the number

$$ P_{ij} = P(\text{go to } j \text{ next} \mid \text{currently in } i). $$

Stack these into a grid — rows are "from," columns are "to" — and you get the **transition matrix** `P`. It has one ironclad property: **each row sums to 1**, because from any state you must go *somewhere* (including possibly staying put). A matrix whose rows are probability distributions like this is called *stochastic*.

![Each node is a state; each arrow is a one-step move labelled with its probability; a node's outgoing probabilities sum to one](/imgs/blogs/markov-chains-hitting-times-quant-interviews-2.png)

The figure shows a three-state toy chain — a cartoon weather model with states Sunny, Cloudy, Rainy — drawn as a **transition diagram**: each state is a circle, each possible one-step move is an arrow, and the arrow's label is its probability. Reading off row S: from Sunny you stay Sunny with probability 0.7 and turn Cloudy with probability 0.3 (and never jump straight to Rainy). Those add to 1, as they must. The full matrix is

$$
P = \begin{pmatrix} 0.7 & 0.3 & 0.0 \\ 0.4 & 0.2 & 0.4 \\ 0.0 & 0.5 & 0.5 \end{pmatrix},
\qquad \text{rows} = \begin{matrix} S \\ C \\ R \end{matrix}
$$

and you can verify every row totals 1.0. The transition diagram and the matrix are the *same object* drawn two ways: the picture is better for intuition, the matrix is better for computation. Interviewers expect you to fluently move between them.

That is the entire vocabulary. State, Markov property, transition matrix. Everything else is consequences.

## The simple random walk

The most important Markov chain in all of quant interviews is the **simple random walk** on the integers. It is the skeleton underneath gambler's ruin, underneath "drunk on a sidewalk," and (in continuous time) underneath the Brownian-motion models that price options.

### Definition

You stand on an integer number line. At each step you take a single step **right (+1) with probability `p`**, or **left (−1) with probability `q = 1 − p`**, independent of everything that came before. If `p = q = 1/2` the walk is called **symmetric** or **fair**; if `p ≠ 1/2` it is **biased** and **drifts** in the direction of the larger probability.

![At each step you move +1 with probability p or -1 with probability q = 1 - p, independent of how you got here](/imgs/blogs/markov-chains-hitting-times-quant-interviews-3.png)

This is a Markov chain with states `…, −2, −1, 0, 1, 2, …` and transitions `P(i → i+1) = p`, `P(i → i−1) = q`. Notice the crucial Markov feature in the figure: the next step depends only on which integer `i` you are standing on, **not on the winding path that got you there**. Whether you arrived at position 7 by going straight up or by bouncing around for a thousand steps, your odds of the next move are identical.

### Why it is the right model for gambling

Map the number line to dollars: your position `i` is your current bankroll. Each step is one bet on a game that pays +$1 if you win (probability `p`) and costs you $1 if you lose (probability `q`). A symmetric walk is a *fair* game (a coin flip); a walk with `p < 1/2` is the realistic casino game, where the house has an edge and the walk drifts toward zero. A walk with `p > 1/2` is the rare situation where *you* have the edge — a card counter, or a market maker capturing a spread — and the walk drifts toward riches.

The simple random walk on an *unbounded* line is mostly a thought experiment. The interesting version, and the one interviewers ask, puts **walls** on it. That is gambler's ruin.

## Gambler's ruin: probability of going broke

Here is the setup that launched a thousand interview questions. You start with **`$i`**. You repeatedly make $1 bets, winning each with probability `p` and losing with probability `q = 1 − p`. You stop the instant you hit **`$0`** (you are broke — *ruined*) or **`$N`** (you have reached your target and walk away). What is the probability you go broke?

![Start with $i; each bet moves you +$1 (prob p) or -$1 (prob q); the game ends only at $0 (ruin) or $N (target)](/imgs/blogs/markov-chains-hitting-times-quant-interviews-4.png)

The figure shows the structure: a random walk pinned between two **absorbing barriers**. A state is **absorbing** if once you enter it you can never leave — `$0` and `$N` are absorbing because the game ends there. The interior states `$1, $2, …, $(N−1)` are **transient**: you visit them temporarily, and with probability 1 you eventually leave for one of the walls and never return. (That "with probability 1 you eventually get absorbed" fact is worth internalizing — a bounded random walk cannot wander forever.)

### Setting it up with first-step analysis

Let `r_i` = the probability of eventual ruin (hitting $0 before $N) when you currently have `$i`. We want `r_i`. The key move — *first-step analysis* — is to condition on what happens on the very next bet.

![Condition on the very first move: the unknown at state i equals p times the unknown at i+1 plus q times the unknown at i-1](/imgs/blogs/markov-chains-hitting-times-quant-interviews-6.png)

From `$i`, with probability `p` you win and move to `$(i+1)`, from which your ruin probability is `r_{i+1}`; with probability `q` you lose and move to `$(i−1)`, from which your ruin probability is `r_{i−1}`. So

$$ r_i = p \cdot r_{i+1} + q \cdot r_{i-1}, \qquad i = 1, 2, \dots, N-1, $$

with the **boundary conditions** `r_0 = 1` (already broke — certain ruin) and `r_N = 0` (already at target — zero ruin). That is `N−1` linear equations in `N−1` unknowns. The whole problem is now algebra.

### Solving the fair game (p = 1/2)

When `p = q = 1/2`, the recursion becomes `r_i = ½ r_{i+1} + ½ r_{i−1}`, which rearranges to

$$ r_{i+1} - r_i = r_i - r_{i-1}. $$

This says the *gaps between consecutive `r` values are all equal* — in other words, `r_i` is a **straight line** in `i`. A line through the boundary points `r_0 = 1` and `r_N = 0` is

$$ \boxed{\,r_i = \frac{N - i}{N}\,} \qquad\Longleftrightarrow\qquad P(\text{reach } \$N \text{ first}) = \frac{i}{N}. $$

Clean and memorable: in a fair game, the probability you reach your target before going broke is **just the fraction of the total money you start with**. Start with $3 aiming for $10? You win 30% of the time. The casino's mirror image: if you have $100 and the house has effectively $10,000 between you and "breaking the bank," your probability of doubling the house is only 100/10,100 ≈ 1%.

#### Worked example: fair game, $5 toward $10

You start with `$5`, target `$N = $10`, fair coin. Probability of ruin:

$$ r_5 = \frac{N - i}{N} = \frac{10 - 5}{10} = \frac{5}{10} = 0.50. $$

So you reach $10 before going broke exactly half the time — which makes sense by symmetry, since $5 is the midpoint. Now start with `$3` instead: `r_3 = 7/10 = 0.70`, a 70% chance of ruin, 30% chance of success. The single sentence to remember: **in a fair game, ruin probability falls linearly as your starting stake rises.**

### Solving the biased game (p ≠ 1/2)

When the game is not fair, the gaps between consecutive `r` values are no longer equal — they form a **geometric sequence** with ratio `q/p`. Working through the recursion (it is a standard linear difference equation) gives

$$ r_i = \frac{(q/p)^i - (q/p)^N}{1 - (q/p)^N}, \qquad P(\text{reach } \$N) = \frac{1 - (q/p)^i}{1 - (q/p)^N}. $$

Let `s = q/p` denote the **odds ratio**. If you have an *edge* (`p > 1/2`), then `s < 1` and `sⁿ` shrinks fast; if the house has the edge (`p < 1/2`), then `s > 1` and `sⁿ` blows up. This single ratio controls everything.

#### Worked example: a 0.6 edge, $5 toward $10

Now suppose `p = 0.6` (you win 60% of bets), so `q = 0.4` and `s = q/p = 0.4/0.6 = 2/3 ≈ 0.667`. Start at `$5`, target `$10`. Probability of *success* (reaching $10):

$$ P(\text{reach }\$10) = \frac{1 - s^5}{1 - s^{10}} = \frac{1 - (2/3)^5}{1 - (2/3)^{10}} = \frac{1 - 0.1317}{1 - 0.01734} = \frac{0.8683}{0.9827} \approx 0.884. $$

So ruin probability `r_5 = 1 − 0.884 ≈ 0.116`. Compare that to the fair game's `0.50`. **A modest 60/40 edge crushes your ruin probability from 50% all the way down to about 12%.** That is the entire economics of a market-making desk in one number: a small per-trade edge, repeated thousands of times, makes going bust astronomically unlikely.

![With N = $10: a fair game (p = 0.5) gives ruin probability (N-i)/N, a straight line; a 0.6 edge crushes ruin probability exponentially](/imgs/blogs/markov-chains-hitting-times-quant-interviews-5.png)

The figure plots ruin probability against starting wealth for the same `N = 10`. The fair game (solid) is a straight line from 1.0 at `i = 0` down to 0 at `i = 10`. The edge game (dashed) is a *convex curve that plunges*: by `i = 5` it is already down near 0.12, and it hugs zero for the rest of the way. The qualitative lesson worth carrying into an interview: **an edge does not just tilt the odds, it changes the entire shape from linear to exponential.** Doubling your bankroll roughly *squares* your survival odds when you have an edge.

A subtle, counterintuitive consequence interviewers love: when you have an edge, **betting smaller is strictly better.** If you have $5 of edge-money and bet $1 at a time toward $10, your survival odds are ~88%; if you bet $5 all at once ("double or nothing"), they collapse to exactly 60%. More bets give the law of large numbers more time to express your edge. The opposite is true at a casino: if you *must* play a negative-edge game, betting big and quitting early (a *bold* strategy) maximizes your slim chance, because it gives the house edge fewer bets to grind you down.

## Expected duration: how long does the game last?

"Probability of ruin" is only half the story. The other classic question is: **how many bets, on average, until the game ends** (one way or the other)?

This is again first-step analysis, with one new wrinkle. Let `D_i` = expected number of bets until absorption starting from `$i`. From `$i` you *take one bet* (that is the `+1`), then land at `$(i+1)` with probability `p` or `$(i−1)` with probability `q`:

$$ D_i = 1 + p \cdot D_{i+1} + q \cdot D_{i-1}, \qquad D_0 = D_N = 0. $$

The crucial difference from the ruin recursion is that **`+1`**: every step counts, so the equation for an *expected count* always carries a `1` for the step you just took. Forgetting it is the single most common mistake on these problems.

![Condition on the very first move: the unknown at state i equals p times the unknown at i+1 plus q times the unknown at i-1](/imgs/blogs/markov-chains-hitting-times-quant-interviews-6.png)

The same first-step picture applies — the bottom equation in that figure is exactly this expected-count recursion, with the telltale `+1`.

### The fair-game answer: i·(N − i)

For the fair game (`p = q = 1/2`), this difference equation has a beautiful closed form:

$$ \boxed{\,D_i = i \cdot (N - i)\,}. $$

You can verify it satisfies the recursion: `½[(i+1)(N−i−1)] + ½[(i−1)(N−i+1)] + 1 = i(N−i)` after expanding (try it — the cross terms cancel and the `+1` exactly fills the gap). It also nails the boundaries: `D_0 = 0·N = 0` and `D_N = N·0 = 0`.

![Fair game, target N = $10: expected duration is i(N-i) bets, peaking at 25 bets when you start at $5 and shrinking toward each wall](/imgs/blogs/markov-chains-hitting-times-quant-interviews-9.png)

The figure plots `D_i = i(10−i)` for `N = 10`. It is an inverted parabola, **symmetric and peaking in the middle**: starting at $5 you expect `5·5 = 25` bets; near either wall the game ends fast (`D_1 = 1·9 = 9`, `D_9 = 9·1 = 9`).

#### Worked example: expected length of the fair $5/$10 game

Start at `$5`, target `$N = 10`, fair coin:

$$ D_5 = 5 \cdot (10 - 5) = 5 \cdot 5 = 25 \text{ bets}. $$

A startling fact lurks here. The game is *fair* — your expected *money* gain is exactly zero on every bet, so your expected final wealth is your starting wealth. Yet you expect to play **25 rounds** before it ends. Random walks meander; they take a surprisingly long time to drift to a boundary even when there is no drift at all. (Pushing the walls apart makes it worse fast: `D` grows like the *square* of the distance between barriers. A $50/$100 game expects `50·50 = 2,500` bets.) The one-sentence intuition: **a fair game can be expected to last a long time precisely because nothing is pulling it toward either exit.**

### The biased-game duration

For `p ≠ q`, the duration has a less pretty but still mechanical formula:

$$ D_i = \frac{i}{q - p} - \frac{N}{q - p} \cdot \frac{1 - (q/p)^i}{1 - (q/p)^N}. $$

For our `p = 0.6`, `N = 10`, `i = 5` example, plugging in gives `D_5 ≈ 19.2` bets — shorter than the fair game's 25, because the drift now actively herds you toward the $10 wall instead of letting you dawdle in the middle. You do not need to memorize this formula for an interview; you need to be able to *set up the recursion* `D_i = 1 + pD_{i+1} + qD_{i−1}` and solve the small system by hand for whatever specific `N` they give you.

## Absorbing chains and the fundamental matrix

Everything so far we solved by hand, one recursion at a time. There is a general matrix recipe that handles *any* absorbing chain mechanically — and naming it correctly in an interview signals you have seen the real theory, not just the gambler's-ruin special case.

### Canonical form

An **absorbing Markov chain** is one where (a) at least one state is absorbing, and (b) from every state you can eventually reach an absorbing one. Reorder the states so the **transient** ones come first and the **absorbing** ones come last. The transition matrix then splits into four blocks:

![Order transient states first and absorbing states last; Q holds transient-to-transient moves, R holds transient-to-absorbing, and absorbing rows are the identity](/imgs/blogs/markov-chains-hitting-times-quant-interviews-7.png)

$$
P = \begin{pmatrix} Q & R \\ 0 & I \end{pmatrix}
$$

- **`Q`** (top-left, `t × t`): probabilities of moving from one transient state to another — moves *between the in-play states*.
- **`R`** (top-right, `t × a`): probabilities of moving from a transient state directly into an absorbing one — the *exits*.
- **`0`** (bottom-left): you can never go from absorbed back to transient — once it's over, it's over.
- **`I`** (bottom-right): the identity — an absorbing state transitions to itself with probability 1.

### The fundamental matrix

Here is the payoff. Define the **fundamental matrix**

$$ \boxed{\,N = (I - Q)^{-1}\,}. $$

The entry `N_{ij}` is the **expected number of times the chain visits transient state `j`, starting from transient state `i`, before being absorbed.** (Where does this come from? `N = I + Q + Q² + Q³ + …`, the sum of "you're there at step 0, step 1, step 2…" — a matrix geometric series that sums to `(I − Q)⁻¹`. That series is exactly counting expected visits.)

![N = (I - Q) inverse counts expected visits to each transient state; multiplying N by an all-ones vector gives expected steps to absorption](/imgs/blogs/markov-chains-hitting-times-quant-interviews-8.png)

From this one inverse, two quantities you actually want fall right out, shown in the figure:

- **Expected steps to absorption:** `t = N · 1` (where `1` is a column of all ones). Summing row `i` of `N` adds up the expected visits to *every* transient state, which is the total expected number of steps before you get absorbed starting from `i`.
- **Absorption probabilities:** `B = N · R`. The entry `B_{ij}` is the probability of being absorbed in absorbing state `j`, starting from transient state `i`. (Intuitively: expected visits to each transient state, times the probability of exiting from each to absorber `j`.)

#### Worked example: a 3-state absorbing chain by hand

Let's make it concrete with the smallest non-trivial gambler's ruin: `N = 3`, fair coin. The transient states are `$1` and `$2`; the absorbing states are `$0` and `$3`. From `$1`: go to `$0` w.p. ½, go to `$2` w.p. ½. From `$2`: go to `$1` w.p. ½, go to `$3` w.p. ½. So

$$
Q = \begin{pmatrix} 0 & \tfrac12 \\ \tfrac12 & 0 \end{pmatrix}\ \ (\text{rows } \$1,\$2), \qquad
R = \begin{pmatrix} \tfrac12 & 0 \\ 0 & \tfrac12 \end{pmatrix}\ \ (\text{cols } \$0,\$3).
$$

Then `I − Q = \begin{pmatrix} 1 & -½ \\ -½ & 1 \end{pmatrix}`, whose determinant is `1 − ¼ = ¾`, so

$$
N = (I-Q)^{-1} = \frac{1}{3/4}\begin{pmatrix} 1 & \tfrac12 \\ \tfrac12 & 1 \end{pmatrix} = \frac{4}{3}\begin{pmatrix} 1 & \tfrac12 \\ \tfrac12 & 1 \end{pmatrix} = \begin{pmatrix} \tfrac43 & \tfrac23 \\ \tfrac23 & \tfrac43 \end{pmatrix}.
$$

Expected steps to absorption from `$1` is the sum of row 1: `4/3 + 2/3 = 2`. And indeed the hand formula `D_1 = i(N−i) = 1·(3−1) = 2` agrees. Absorption probabilities `B = N·R`: row 1 gives ruin probability `(4/3)(½) = 2/3`, matching `r_1 = (N−i)/N = 2/3`. **The matrix machine and the by-hand recursion give the same answers — they have to, because they are the same math wearing different clothes.** Use the recursion for small problems on a whiteboard; mention the fundamental matrix to show you know it generalizes.

## Expected hitting times as a linear system

The phrase **hitting time** is the formal name for "how many steps until I first reach a target state." Everything in the duration section was a hitting-time computation; let's state the general pattern, because interviewers phrase it in endless costumes ("expected steps to reach the corner," "average time to first return," "how long until the token is absorbed").

Let `h_i` = expected number of steps to first hit some target set `T`, starting from state `i`. The recipe is always:

1. **`h_i = 0` for every state already in the target `T`.** (You're there; zero steps.)
2. For every other state, **first-step analysis with a `+1`:** `h_i = 1 + Σ_j P_{ij} h_j`, summing over the states `j` you can move to.

That is a linear system in the unknown `h_i`'s — one equation per non-target state. Solve it. For "probability of hitting A before B" the recipe is identical but **without the `+1`** and with boundaries `h_A = 1`, `h_B = 0` (you're tracking a probability, not a count, so reaching the target adds nothing per step).

The reason to drill this until it is automatic: the *setup* never changes. Identify states → mark the target states with the boundary value → write `value = (per-step cost) + average of neighbor values` for the rest → solve. Whether the chain is a gambling line, a chessboard, or a graph of cities, the mechanical procedure is the same, and that mechanical sameness is exactly what makes these problems fast once you stop treating each as a fresh puzzle.

## The stationary distribution (intuition only)

One more concept rounds out the toolkit, and it answers a *different* family of questions: not "how long to get absorbed" (those chains end), but "in a chain that runs forever, what fraction of time is spent in each state?" and "how long until I come back to where I started?"

If a chain has no absorbing traps and can get from anywhere to anywhere (it is *irreducible*) and doesn't get stuck in a rigid cycle (it is *aperiodic*), then no matter where you start, the probability distribution over states **converges to a single fixed distribution** called the **stationary distribution** `π`.

![Start anywhere; multiply by P each step and the distribution converges to the stationary vector pi, where pi = pi P and return time is one over its probability](/imgs/blogs/markov-chains-hitting-times-quant-interviews-12.png)

The figure shows the idea: start with all your probability on one state, multiply by the transition matrix `P` each step, and the distribution spreads out and **settles**. The limiting vector `π` is the one that no longer changes when you apply `P`:

$$ \pi = \pi P, \qquad \sum_i \pi_i = 1. $$

`π_i` is the long-run fraction of time the chain spends in state `i`. And here is the one fact that turns this into an interview superpower:

$$ \boxed{\,\text{Expected return time to state } i = \frac{1}{\pi_i}\,}. $$

If, in the long run, you spend 1/4 of your time at a state, then on average you come back to it every 4 steps. This is the fastest possible way to answer "expected number of steps to return to the start" — *if* you can find `π` (which is often easy by symmetry).

#### Worked example: the frog on four lily pads

A frog sits on pad A of four pads arranged in a square (A–B–C–D–A). Each second it hops to one of its two neighbors, each with probability 1/2. **What is the expected number of hops until it returns to A?**

![From each pad the frog jumps to either neighbour with probability one half; every pad is visited equally, so the expected return time is exactly four hops](/imgs/blogs/markov-chains-hitting-times-quant-interviews-11.png)

By the symmetry of the square, every pad is identical — there is no reason the frog should favor any one — so the stationary distribution is **uniform**: `π_A = π_B = π_C = π_D = 1/4`. Therefore

$$ \text{expected return time to A} = \frac{1}{\pi_A} = \frac{1}{1/4} = 4 \text{ hops}. $$

The intuition in one sentence: **the more time a chain spends in a state, the more often it returns, and "expected return time" is just the reciprocal of that long-run frequency.** (You can double-check by first-step analysis: by symmetry the frog is equally likely to be one or three steps from home after leaving, and grinding the small linear system also yields 4. The stationary-distribution shortcut just skips the algebra.)

## In the interview room

Theory is necessary but not sufficient. What firms actually do is hand you a concrete, slightly-disguised version and watch how you set it up under mild pressure. Below are six problems in the exact spirit of what gets asked, each solved end-to-end. The meta-skill on display every time is the same: **name the states, write the one-step recursion, solve.**

### Problem 1 — Expected flips to see HH (the classic)

> *Flip a fair coin repeatedly. What is the expected number of flips to first see two heads in a row (HH)?*

**Set up the states** by tracking progress toward the pattern HH:

- `S₀` — no useful progress (start, or the last flip was a T).
- `S₁` — the last flip was an H (one away from done).
- `HH` — done (absorbing).

![States track the longest suffix that matches HH so far; a tail throws you back, and the expected wait solves a tiny linear system to 6 tosses](/imgs/blogs/markov-chains-hitting-times-quant-interviews-10.png)

The figure shows the chain. The single insight that makes pattern problems Markov: **the state is the longest suffix of your flips that is a prefix of the target pattern.** A tail in `S₁` does not send you to "two flips wasted"; it sends you back to `S₀`, because the relevant progress is gone. (For HH there is no partial overlap to preserve; patterns like HTH where the prefix and suffix share an H are where this gets subtle — see Problem 2.)

**Write the recursions** (each is `1` for the flip you make, plus the average of where you land):

$$ E_0 = 1 + \tfrac12 E_1 + \tfrac12 E_0, \qquad E_1 = 1 + \tfrac12 \cdot 0 + \tfrac12 E_0. $$

In `E₀`: flip H (prob ½) → go to `S₁`; flip T (prob ½) → stay in `S₀`. In `E₁`: flip H (prob ½) → **done**, contributing 0 more flips; flip T (prob ½) → back to `S₀`.

**Solve.** From the first equation, `½E₀ = 1 + ½E₁`, so `E₀ = 2 + E₁`. Substitute into the second: `E₁ = 1 + ½E₀ = 1 + ½(2 + E₁) = 2 + ½E₁`, giving `½E₁ = 2`, so `E₁ = 4` and `E₀ = 6`.

**Answer: 6 flips.** The naive "1/(1/4) = 4" is wrong because the events "HH starting at flip 1," "HH starting at flip 2," etc., *overlap* — a stray tail resets your progress, and that resetting cost is exactly the extra 2 flips. The clean cross-check: for a fair coin, the expected wait for a *self-overlapping* pattern of length `k` like HH is `2^{k} + 2^{k-1} + … `; for HH that is `2² + 2¹ = 6`. For three-in-a-row HHH it is `2³ + 2² + 2¹ = 14`.

### Problem 2 — When the pattern overlaps: HT vs HH

> *Same fair coin. Expected flips to first see HT? Is it the same as HH?*

This problem exists to catch people who memorized "6" without understanding it. **HT and HH have different expected waits**, and seeing why is the point. States, tracking progress toward HT:

- `S₀` — no H yet (start or last relevant flip was a "useless" one).
- `S₁` — last flip was H, waiting for the T.
- `HT` — done.

Recursions:

$$ E_0 = 1 + \tfrac12 E_1 + \tfrac12 E_0, \qquad E_1 = 1 + \tfrac12 \cdot 0 + \tfrac12 E_1. $$

The asymmetry is in `E₁`: once you have an H, a T finishes you (→ 0), but **another H keeps you in `S₁`** — you don't lose progress, you're still "primed with an H, waiting for a T." Contrast HH, where the bad flip (a T) sent you all the way back to `S₀`.

**Solve.** From `E₁`: `½E₁ = 1`, so `E₁ = 2`. From `E₀`: `½E₀ = 1 + ½E₁ = 1 + 1 = 2`, so `E₀ = 4`.

**Answer: 4 flips for HT, versus 6 for HH.** The deep takeaway interviewers want: **patterns that cannot overlap themselves (HT) are reached faster than patterns that can (HH).** When you fail to complete HH, you waste the H you had; when you fail to complete HT, the failing flip is another H that *keeps you in business*. As a final flex, the overlap-aware formula (Conway's leading-numbers / correlation method) gives **HTH = 10**: states `S₀ → H → HT → HTH`, where an H out of `HT` jumps you forward to `HTH` (done) but the overlap means a failed attempt can leave you in state `H` rather than `S₀`. Set up `E_{HT} = 1 + ½·0 + ½E_0`, `E_H = 1 + ½E_H + ½E_{HT}`, `E_0 = 1 + ½E_H + ½E_0`, and grind: `E_{HT}=6, E_H=8, E_0=10`.

### Problem 3 — Gambler's ruin with an edge

> *You start with $20. You make $1 bets on a game you win with probability 0.55. You stop at $0 or at $40. What is your probability of reaching $40 before going broke? Is this game worth playing?*

**Recognize it instantly** as biased gambler's ruin: `i = 20`, `N = 40`, `p = 0.55`, `q = 0.45`. The odds ratio is `s = q/p = 0.45/0.55 = 9/11 ≈ 0.8182`.

**Apply the formula** `P(\text{reach } N) = \frac{1 - s^i}{1 - s^N}`:

$$ s^{20} = (0.8182)^{20} \approx 0.0188, \qquad s^{40} = (0.8182)^{40} \approx 0.000352. $$

$$ P(\text{reach }\$40) = \frac{1 - 0.0188}{1 - 0.000352} = \frac{0.9812}{0.99965} \approx 0.9815. $$

**Answer: about 98.2%.** Starting at the midpoint of a *fair* game you would have exactly 50%; a 55/45 edge over 20 bets of cushion launches your survival probability to ~98%. **Is it worth playing?** Each bet has positive expected value (`0.55 − 0.45 = +$0.10` per $1 risked, a 10% edge), and you almost certainly hit the target — so on the stated terms, yes, the math strongly favors you. The honest caveat an interviewer wants to hear: this assumes the edge is *real and stable*, you can survive the ~2% tail where you bust, and the bet size is small relative to your bankroll. Real trading edges are noisier and can vanish; the structure of the math is sound, but "the edge is genuinely 55%" is the load-bearing and most dangerous assumption.

### Problem 4 — Random walk on a graph (return time)

> *A knight-like token sits on a vertex of a triangle (vertices A, B, C, every pair connected). Each step it moves to one of the other two vertices, each with probability 1/2. Starting at A, what is the expected number of steps to first return to A? And the expected number of steps to first reach a specific other vertex, say C?*

**Return time first, via the stationary distribution.** The triangle is fully symmetric, so `π_A = π_B = π_C = 1/3`. Expected return time to A is `1/π_A = 3 steps`.

**Reach-C time, via first-step analysis.** Let `h_X` = expected steps to first hit C from X. Target: `h_C = 0`. By symmetry `h_A = h_B` (A and B are interchangeable with respect to "distance to C"). Call it `h`. From A: with prob ½ you go to C (done, but that took 1 step), with prob ½ you go to B (then expected `h` more):

$$ h = 1 + \tfrac12 \cdot 0 + \tfrac12 \cdot h \ \Longrightarrow\ \tfrac12 h = 1 \ \Longrightarrow\ h = 2. $$

**Answers: expected return to A is 3 steps; expected time to first reach C is 2 steps.** The instructive contrast: *return* time (3) is longer than *first-passage* time to a specific other vertex (2), because to return you must first leave and then come all the way back. A clean follow-up they might ask: do these add up sensibly? Expected steps to return to A = 1 (the forced first step to some neighbor) + expected time to get from a neighbor back to A = `1 + 2 = 3`. It checks out — and showing that consistency check unprompted is exactly the kind of thing that separates a strong answer from a correct one.

### Problem 5 — Three-state regime model (expected time to stress)

> *A market sits in one of three regimes — Calm, Normal, or Stressed — and re-evaluates once a day. From **Calm**, it stays Calm with probability 0.7 and slips to Normal with probability 0.3 (it never lurches straight to Stressed). From **Normal**, it relaxes back to Calm with probability 0.5, stays Normal with probability 0.3, and tips into Stressed with probability 0.2. Starting from Calm, what is the expected number of days until the market first becomes Stressed?*

This is the same hitting-time machine as gambler's ruin, but now the chain is not a tidy line — it is a little three-node graph, which is exactly the disguise interviewers use to check whether you really internalized the method or just memorized `i(N−i)`. The target state is **Stressed**, so we treat it as absorbing for the purpose of *this* question (we only care about the first time we touch it). The transition structure for the two non-target states is

$$
P = \begin{pmatrix} 0.7 & 0.3 & 0.0 \\ 0.5 & 0.3 & 0.2 \\ - & - & - \end{pmatrix}, \qquad \text{rows} = \begin{matrix} \text{Calm} \\ \text{Normal} \\ \text{Stressed} \end{matrix}
$$

(the Stressed row is irrelevant — once we hit it we stop counting).

**Name the unknowns.** Let `h_C` and `h_N` be the expected number of days to first reach Stressed, starting from Calm and Normal respectively. By definition `h_S = 0`.

**Write the one-step recursions** (the telltale `+1` for the day that passes on every transition):

$$ h_C = 1 + 0.7\,h_C + 0.3\,h_N, \qquad h_N = 1 + 0.5\,h_C + 0.3\,h_N + 0.2\cdot 0. $$

Read off the second equation: from Normal a day passes (`+1`); with probability 0.5 you fall back to Calm (`h_C` more days expected), with 0.3 you linger in Normal (`h_N` more), and with 0.2 you hit Stressed and stop (the `0`).

**Solve the 2×2 system.** Collect terms in the first equation: `0.3 h_C = 1 + 0.3 h_N`, so

$$ h_C = \tfrac{1}{0.3} + h_N = \tfrac{10}{3} + h_N. $$

Collect the second: `0.7 h_N = 1 + 0.5 h_C`. Substitute `h_C = 10/3 + h_N`:

$$ 0.7\,h_N = 1 + 0.5\left(\tfrac{10}{3} + h_N\right) = 1 + \tfrac{5}{3} + 0.5\,h_N = \tfrac{8}{3} + 0.5\,h_N. $$

Hence `0.2 h_N = 8/3`, giving `h_N = 40/3 ≈ 13.33` days, and back-substituting,

$$ h_C = \tfrac{10}{3} + \tfrac{40}{3} = \tfrac{50}{3} \approx 16.67 \text{ days}. $$

**Answer: about 16.7 days from Calm** (and ~13.3 from Normal). The sanity checks are quick and worth narrating out loud: `h_C > h_N`, as it must be — Calm is one structural step *further* from Stressed than Normal — and both are comfortably larger than the naive `1/0.2 = 5` you'd get by pretending each day were an independent 20% shot at stress. That naive number is wrong for the same reason the coin problem's "4" was wrong: the only door into Stressed is *through Normal*, and most days you are either sitting in Calm or being knocked back to it, so the true wait is roughly three times longer. The transferable lesson: **for a multi-state hitting time, mark the target as the zero-boundary, write one `value = 1 + Σ (transition prob)·(neighbor value)` equation per remaining state, and solve the small linear system — the shape of the graph never changes the recipe.**

*Deeper mechanics — the same answer from the fundamental matrix.* If you want to show the interviewer the general machinery rather than ad-hoc algebra, this is exactly the `t = N·1` recipe from the absorbing-chain section. The transient block (rows and columns Calm, Normal) is `Q = \begin{pmatrix} 0.7 & 0.3 \\ 0.5 & 0.3 \end{pmatrix}`, so `I − Q = \begin{pmatrix} 0.3 & -0.3 \\ -0.5 & 0.7 \end{pmatrix}` with determinant `(0.3)(0.7) − (−0.3)(−0.5) = 0.21 − 0.15 = 0.06`. Inverting a 2×2 (swap the diagonal, negate the off-diagonal, divide by the determinant) gives the fundamental matrix `N = (I−Q)^{-1} = \frac{1}{0.06}\begin{pmatrix} 0.7 & 0.3 \\ 0.5 & 0.3 \end{pmatrix}`. Summing the Calm row of expected visits, `(0.7 + 0.3)/0.06 = 1/0.06 = 50/3 ≈ 16.67`, reproduces `h_C` exactly; the Normal row gives `(0.5 + 0.3)/0.06 = 0.8/0.06 = 40/3 ≈ 13.33`. Two routes, one answer — which is the entire moral of the absorbing-chain section made concrete on a regime model.

### Problem 6 — Double-or-broke with an edge (ruin *and* duration together)

> *A trader starts a session with $5 of risk budget and makes a sequence of $1 bets on a strategy that wins each bet with probability 0.55. She quits when she has either doubled to $10 or lost everything ($0). What is her probability of going broke, and how many bets does the session last on average? Compare both to the fair-coin version.*

This is the full gambler's-ruin package — **probability and duration in one breath** — with a deliberate edge so you can show you know both the fair `i/N`, `i(N−i)` shortcuts *and* the biased formulas, and can sanity-check one against the other. Recognize the parameters instantly: `i = 5`, `N = 10`, `p = 0.55`, `q = 0.45`, odds ratio

$$ s = \frac{q}{p} = \frac{0.45}{0.55} = \frac{9}{11} \approx 0.8182. $$

**Ruin probability.** Plug into `r_i = \dfrac{s^i - s^N}{1 - s^N}`. The two powers we need are `s⁵ = (9/11)⁵ ≈ 0.3666` and `s¹⁰ ≈ 0.1344`, so

$$ r_5 = \frac{s^5 - s^{10}}{1 - s^{10}} = \frac{0.3666 - 0.1344}{1 - 0.1344} = \frac{0.2322}{0.8656} \approx 0.268. $$

So she busts about **27%** of the time and reaches $10 about **73%** of the time. Cross-check against the *fair* version: at the midpoint `i = 5` of `N = 10`, fair ruin is exactly `(N−i)/N = 5/10 = 0.50`. A 55/45 edge has already pulled her ruin probability down from 50% to 27% — the same edge-shrinks-ruin effect from earlier, here with only $5 of cushion rather than $20, so the improvement is real but not yet dramatic.

**Expected number of bets.** For the biased game the duration formula is

$$ D_i = \frac{i}{q-p} - \frac{N}{q-p}\cdot\frac{1 - s^i}{1 - s^N}. $$

Here `q − p = 0.45 − 0.55 = −0.10`, and the success factor is `\frac{1-s^5}{1-s^{10}} = \frac{0.6334}{0.8656} ≈ 0.7317`. So

$$ D_5 = \frac{5}{-0.10} - \frac{10}{-0.10}\cdot 0.7317 = -50 - (-100)(0.7317) = -50 + 73.17 \approx 23.2 \text{ bets}. $$

(The two negative signs cancel — a classic place to drop a sign under pressure, so it's worth saying "minus times minus is plus" out loud.) **Answer: ruin probability ≈ 0.27, expected session length ≈ 23.2 bets.** Now the cross-check that proves you understand the structure: the *fair* version's duration is the clean `D_5 = i(N−i) = 5·5 = 25` bets. The edge shortens the expected session only slightly — from 25 to ~23 — because a 55/45 drift is gentle; it nudges her toward the $10 wall a touch faster but does not herd her there. (Contrast the steeper `p = 0.6` example earlier, which cut a comparable game down to ~19 bets.) The transferable lesson: **ruin probability and duration come from the *same* recursion with different boundary bookkeeping — `r_i = p\,r_{i+1} + q\,r_{i-1}` carries no `+1`, while `D_i = 1 + p\,D_{i+1} + q\,D_{i-1}` does — and the fair-game answers `i/N` and `i(N−i)` are always there as a free sanity rail for the biased numbers.**

### A note on what they're really testing

Across all six problems the *content* differs but the *process* is identical, and that is the whole point of the genre. The interviewer is checking whether you can (1) compress a wordy scenario into a small state space, (2) write `value = cost + weighted average of neighbor values` without fumbling the `+1` or the boundary conditions, and (3) solve a 2×2 or 3×3 system by hand calmly. If you can narrate those three steps out loud while doing them, you will look fluent even on a problem you have never seen — because, structurally, you *have* seen it.

## Common misconceptions

**"The expected wait for a pattern of probability `q` is `1/q`."** This is right for a *single trial* (expected rolls to get a 6 on a die is `1/(1/6) = 6`) but wrong for *patterns over time*, because overlapping attempts are not independent. HH has per-pair probability 1/4 but expected wait 6, not 4. The fix is to track partial progress as states — the `1/q` rule silently assumes no overlap.

**"In a fair game, my expected number of bets is small because the game is balanced."** The opposite. A *fair* game has no drift pulling it to a boundary, so it meanders for a long time — `i(N−i)` bets, which is `25` at the midpoint of a $10 game and grows like the *square* of the table size. Balance makes games *longer*, not shorter.

**"With a positive edge, betting bigger gets me to my target faster, so it's better."** Bigger bets do end the game faster, but they *lower* your probability of reaching the target when you have an edge, because they give the law of large numbers fewer trials to express that edge. With an edge, bet *small*. (Only in a *losing* game does bold, big betting help — and only by giving the house edge less time to grind you down.)

**"Absorbing states and the stationary distribution are the same machinery."** They answer opposite questions. Absorbing chains *end* — you compute probability and time of absorption via `(I − Q)⁻¹`. Stationary distributions describe chains that *run forever* — long-run frequencies and return times via `π = πP`. A chain with an absorbing trap has no (interior) stationary distribution; a chain that runs forever has no absorption. Don't reach for the wrong tool.

**"A random walk will always eventually come back to where it started."** True for the symmetric walk in *one and two* dimensions (it is *recurrent*), but **false in three or more dimensions** — the famous result that "a drunk man finds his way home, but a drunk bird may not." In 3D a simple random walk returns to its origin only about 34% of the time. If an interviewer asks about walks in higher dimensions, this recurrence-vs-transience fact is what they're fishing for. There is a sharp follow-up trap hiding inside the recurrent case, too: the 1D walk returns with probability 1, yet the *expected* number of steps to return is **infinite** (the return-time distribution has such a heavy tail that its mean diverges). So "certain to return" and "returns quickly on average" are not the same statement — recurrence is about *whether*, expected return time is about *how long*, and a walk can be guaranteed to come home while taking, in expectation, forever to do it. That gap between probability-1 and finite-expectation is exactly the subtlety a sharp interviewer probes once you confidently say "it always returns."

**"`p` slightly above 1/2 only helps a little."** Over a single bet, yes. But ruin probability depends on `(q/p)^i`, which is *exponential* in your bankroll. A 51/49 edge held over a few hundred bets is the difference between near-certain ruin and near-certain success. Small edges compound dramatically — that is the entire business model of high-frequency market making.

## How it shows up in real trading and research

**Market-making and the survival of an edge.** A market maker earns a tiny expected profit per trade (the captured spread) but takes on inventory risk that random-walks up and down. Gambler's-ruin math is the back-of-envelope for *risk of ruin*: with a real per-trade edge and bets that are small relative to capital, the `(q/p)^i` term makes blowing up exponentially unlikely — which is precisely why desks obsess over keeping each trade small and the edge positive rather than chasing big wins. The 60/40 example collapsing ruin from 50% to 12% is this principle in miniature.

**Pairs trading and mean reversion.** A classic stat-arb strategy bets that the price *spread* between two related assets, having wandered from its average, will revert. Modeled crudely, the spread is a random walk with a pull toward the mean (an *Ornstein–Uhlenbeck* process, the continuous cousin of a biased walk toward a center). Expected hitting time — "how long until the spread reverts to zero so I can close the trade?" — is exactly a hitting-time computation, and it directly drives how long capital is tied up and how to size the position.

**Credit-rating migration.** Rating agencies publish *transition matrices*: the probability that a bond rated, say, BBB this year is rated AAA, BBB, or D (default) next year. Default is an **absorbing state** — once a bond defaults, it stays defaulted. The fundamental-matrix machinery `(I − Q)⁻¹` then yields the expected number of years to default and the cumulative default probability over any horizon, feeding directly into bond pricing and capital requirements. This is the gambler's-ruin absorbing chain applied to credit, almost verbatim.

**Algorithm and execution analysis.** When a large order is sliced into many small child orders, the realized average price random-walks around the arrival price. "Expected time for our execution to reach a target benchmark" and "probability we beat VWAP before the close" are hitting-time and absorption questions on the execution path. The same recursion that solves the coin problem prices the risk of a slow fill.

**The link to options pricing.** The reason this material is foundational, not just a brain-teaser, is that the continuous-time limit of a symmetric random walk is **Brownian motion**, and Brownian motion is the engine inside the Black–Scholes model. The discrete *binomial option-pricing tree* is *literally* a biased random walk on stock prices, and pricing a knock-out option — one that dies if the stock hits a barrier — is a two-absorbing-barrier problem, the option-market twin of gambler's ruin. If you understand absorbing walks, you already understand the skeleton of barrier-option pricing.

## When this matters and where to go next

If you are preparing for quant interviews, internalize the *process* over the formulas: most "expected number of steps" and "probability of A before B" questions are Markov chains, and the move is always name-the-states → write `value = cost + average of neighbors` → solve the linear system. Drill the coin-pattern family (HH, HT, HTH), the gambler's-ruin family (fair `i/N` and the biased `(q/p)` formula plus its `i(N−i)` duration), and the small-graph return-time family until the setup is reflexive. The few facts genuinely worth memorizing are tiny: fair-ruin probability `i/N`, fair duration `i(N−i)`, the fundamental matrix `(I − Q)⁻¹`, and return time `1/π_i`.

Beyond interviews, this is the substrate of a large amount of quantitative finance, so it pays off twice. Natural next steps from here: the continuous-time limit (Brownian motion and Itô calculus), which turns these recursions into differential equations; the [Black–Scholes model](/blog/trading/quantitative-finance/black-scholes), whose binomial-tree derivation is a random walk you can now read fluently; and the broader probability toolkit interviewers pair with this one — [conditional probability and Bayes](/blog/trading/quantitative-finance/conditional-probability-bayes-quant-interviews) for the "given that…" questions and [order statistics and uniform-distribution tricks](/blog/trading/quantitative-finance/order-statistics-uniform-tricks-quant-interviews) for the "expected minimum/maximum" questions. Markov chains are where time and probability meet; once that machine is automatic, a whole genre of hard-looking problems turns into turning a crank.
