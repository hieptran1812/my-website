---
title: "Dynamic programming and optimal execution: how to sell a million shares without moving the market"
date: "2026-06-15"
description: "How the Bellman equation turns a multi-period decision into a sequence of easy ones, and how the Almgren-Chriss model uses it to find the cheapest way to trade a large order under uncertainty."
tags: ["dynamic-programming", "bellman-equation", "optimal-execution", "almgren-chriss", "market-impact", "backward-induction", "optimal-stopping", "quantitative-finance"]
category: "trading"
subcategory: "Quantitative Finance"
author: "Hiep Tran"
featured: true
readTime: 44
---

> [!important]
> **TL;DR** — Dynamic programming is a way to make a hard sequence of decisions by solving the *last* one first and working backward, and it is exactly the tool that tells a trading desk the cheapest way to sell a giant order over a day.
>
> - The **Bellman equation** says the best value of being in a situation equals the best single action's immediate reward plus the best value of wherever that action lands you. That one recursive sentence collapses an exponential search into a short list of easy problems.
> - **Backward induction** solves it: figure out the value of the final step, then the step before, and so on back to now. Each step trusts the answer already computed for the future.
> - The **Almgren-Chriss model** applies this to trading: it minimizes *expected cost plus a penalty on the variance of cost*, balancing **market impact** (the price you push by trading fast) against **price risk** (the danger of waiting and getting a worse price). The answer is a smooth schedule that sells more early and tapers off.
> - The one number to remember: a patient TWAP plan and a risk-averse optimal plan can differ by tens of thousands of dollars on a single large order — in our worked example, about **\$36,000** on a \$5,000,000 trade.

In 2010, a single mutual-fund sell order of roughly \$4.1 billion in stock-index futures, fed into the market by an automated program that only watched volume and ignored price, helped trigger the "Flash Crash" — the Dow fell about 1,000 points in minutes and recovered most of it before lunch. The order itself was not malicious or even unusual in size. The problem was *how* it was sliced: the algorithm pushed shares into a thinning market faster than the market could absorb them, and the price caved in. That single afternoon is the most expensive advertisement ever made for one quiet question: **when you have a lot to sell, how fast should you sell it?**

That question has a precise mathematical answer, and the math behind it is one of the most beautiful ideas in all of applied mathematics: **dynamic programming**. It is the same engine that plans a chess move, routes your GPS, and aligns your DNA in a biology lab. In finance it powers *optimal execution* — the art of working a large order into the market at the lowest possible cost. By the end of this post you will be able to set up a Bellman equation by hand, solve a tiny trading problem with backward induction, derive the famous Almgren-Chriss schedule, and understand why the optimal plan almost never sells in equal slices.

![Diagram of the dynamic programming loop from state to action to reward to value to next state](/imgs/blogs/dynamic-programming-optimal-execution-math-for-quants-1.png)

## The one loop behind every multi-period decision

The diagram above is the mental model for everything that follows, so let us walk it once slowly. On the far left is a **state** — a complete description of your situation right now. In execution, the state is mostly one number: how many shares you still have left to sell. From that state you choose an **action** — how many shares to sell this minute. That action has a consequence you can measure immediately, the **reward** (in our case it is a *cost*, so a negative reward — the money you lose to fees, spread, and pushing the price). Choosing the action also moves you to a **next state**: fewer shares left. And the **value** of the whole thing is the best total reward you can collect from here to the end.

Then the loop repeats. New state, new action, new cost, new state, until you have nothing left to sell. The entire field of dynamic programming is the study of one question about this loop: *given that the future will also be played optimally, what is the best action right now?* That sounds circular — how can you choose now if the answer depends on the future, which depends on now? The resolution is the single most important trick in the subject, and it is the reason this whole article exists. We will get there in a moment. First, the vocabulary.

A reassurance before we start, because the words "dynamic programming" scare people: there is nothing dynamic and no programming. The name was chosen in the 1950s by Richard Bellman partly because it sounded impressive to a budget committee that was suspicious of mathematics. What the technique actually does is embarrassingly simple — it remembers answers so it never solves the same sub-problem twice. If you have ever solved a maze by working backward from the exit, you already understand the core idea. We are going to make that instinct precise and then point it at a five-million-dollar trade.

## Foundations: the building blocks

Before any equations, we need to agree on what the pieces are. If you have never seen a "state" or an "expected value" written down formally, this section is for you. A practitioner can skim it; a beginner should not skip it.

### What a "state" is, in trading terms

A **state** is a snapshot that contains everything you need to know to make the next decision — and nothing about how you got there. The technical name for "the past does not matter once you know the present" is the **Markov property**, and it is what makes dynamic programming tractable. In our liquidation problem the state at any time is essentially the pair (how much time is left, how many shares are left). If you have 60,000 shares left and 3 hours to go, it does not matter whether you started with 100,000 or 200,000 — your remaining problem is identical. That is the Markov property doing its job: it lets us throw away history and plan from the present.

We will write the state as $s$. In a discrete problem $s$ might just be "40,000 shares left at step 2." The set of all possible states is the *state space*. The whole difficulty of dynamic programming, as we will see, is that this space can be astronomically large.

### What an "action" and a "policy" are

An **action** $a$ is a choice available in a state — here, "sell 25,000 shares this period." The list of actions allowed in a state is the *action space*. A **policy**, written $\pi$, is a rule that picks an action for *every possible state*. "Sell one-third of whatever I have left each period" is a policy. "Sell 25,000 every period regardless" is a different policy. The goal of dynamic programming is to find the *optimal policy* $\pi^*$ — the rule that, followed everywhere, produces the best total outcome. Notice the difference between a *plan* (a fixed sequence of actions decided up front) and a *policy* (a rule that reacts to the state you actually find yourself in). When there is uncertainty, a policy beats a plan, because it can adapt.

### What a "reward" and the "value function" are

A **reward** $r(s, a)$ is the immediate payoff for taking action $a$ in state $s$. In execution the reward is negative — it is the cost of trading: the spread you cross, the fees you pay, and the price you push by demanding liquidity. We want to *minimize* total cost, which is the same as *maximizing* total (negative) reward; we will flip freely between the two and always tell you which.

The **value function** $V(s)$ is the heart of everything. It is defined as *the best total reward achievable starting from state $s$ and playing optimally forever after.* Read that twice. $V(s)$ already assumes you will make every future decision perfectly. It is not the value of one action; it is the value of *being in that situation* given perfect play ahead. If $V(\text{60,000 shares left, 3 hours to go}) = -\$8{,}200$, it means the cheapest possible way to finish liquidating from there will cost about \$8,200 in expectation. The whole game is to compute this function.

### What "expected value" means, quickly

Because prices are random, we cannot know future costs exactly — we can only know them *on average*. The **expected value** of a random quantity, written $\mathbb{E}[\cdot]$, is its probability-weighted average: each possible outcome times its probability, all summed. If a coin flip pays you \$10 on heads and \$0 on tails, the expected payoff is $0.5 \times \$10 + 0.5 \times \$0 = \$5$, even though you will never actually receive \$5 on any single flip. When the next state $s'$ is random, the value of an action must average over where you might land: $\mathbb{E}[V(s')]$. If expected value is rusty, the [law of large numbers and central limit theorem](/blog/trading/math-for-quants/law-large-numbers-central-limit-theorem-math-for-quants) post builds it from scratch.

### What "variance" and "risk aversion" mean

**Variance** measures how spread out a random quantity is — the average squared distance from its mean. A trade whose cost is "\$10,000 give or take \$500" has low variance; one that is "\$10,000 give or take \$40,000" has high variance, even if both average \$10,000. A **risk-averse** trader does not just care about the average cost; they will pay a little more on average to make the outcome more *certain*. The single parameter that encodes "how much I dislike uncertainty" is usually written $\lambda$ (lambda), and as we will see, $\lambda$ is the dial that turns the patient execution plan into an aggressive one.

#### Worked example: the simplest possible value function

You have **2 shares** to sell over **2 minutes**, and you must finish. Suppose selling 1 share costs you \$1 in impact, and selling 2 shares at once costs you \$4 (impact grows faster than linearly — pushing harder costs disproportionately more). There is no price risk in this toy; costs are certain. What is the best policy and what is $V$ at the start?

Two plans exist. Plan A: sell 1 share each minute. Cost $= \$1 + \$1 = \$2$. Plan B: sell both in minute 1 (then nothing in minute 2). Cost $= \$4 + \$0 = \$4$. The optimal policy is "sell 1 share per minute," and the value of the starting state is $V(\text{2 shares, 2 min}) = -\$2$. The lesson in one sentence: **because impact cost is convex — it grows faster than the size you trade — spreading a trade out is cheaper than dumping it, and that single fact is the entire reason execution algorithms exist.**

## The Bellman equation

Now the central idea. We have a hard problem: choose a whole sequence of actions over many periods to minimize total cost under uncertainty. The number of possible sequences explodes — with 10 periods and 10 possible trade sizes each, that is $10^{10}$ = ten billion plans. Checking them one by one is hopeless. Dynamic programming makes the explosion vanish with a single observation.

### The principle of optimality, in plain English

Bellman's **principle of optimality** says: *whatever the first action is, the rest of an optimal plan must itself be optimal for the situation that first action leaves you in.* Put another way — if your overall plan is the best possible, then starting from tomorrow's leftover position, your remaining plan must also be the best possible for that leftover position. It cannot contain a wasteful sub-plan, because you could swap in the better sub-plan and improve the whole.

Here is the everyday analogy. Suppose the cheapest driving route from San Francisco to New York happens to pass through Denver. Then the portion of that route from Denver to New York *must itself* be the cheapest route from Denver to New York. If it weren't, you could replace it with the cheaper Denver-to-New York route and get an even cheaper San Francisco-to-New York route — contradicting that you had the cheapest one. The big problem's solution is built out of its sub-problems' solutions. That is the principle of optimality, and it is almost a tautology once you see it — which is exactly why it is so powerful.

![Diagram of a decision tree branching into sell-fast and sell-slow paths from the start state](/imgs/blogs/dynamic-programming-optimal-execution-math-for-quants-3.png)

The tree above shows why this matters. At the start you face a branch: sell fast or sell slow. Each branch leads to its own consequences — selling fast means high impact cost but low remaining price risk; selling slow means low impact but a larger position exposed to a moving market. Without the principle of optimality you would have to trace every leaf of this tree, and the tree doubles in width at every step. With it, you only ever need to know the *value* of each child node, computed once, and you pick the better child. The exponential tree collapses into a linear walk.

### The equation itself

Write that principle as a formula and you get the **Bellman equation**:

$$ V(s) = \max_{a} \Big[\, r(s, a) + \mathbb{E}\big[\, V(s')\,\big] \,\Big]. $$

Let us name every symbol. $V(s)$ is the best achievable value starting from state $s$. The $\max_a$ means "choose the action $a$ that makes the bracket as large as possible." Inside, $r(s,a)$ is the *immediate* reward of taking action $a$ now. And $\mathbb{E}[V(s')]$ is the *expected value of the future*: where action $a$ lands you is the next state $s'$ (possibly random), and $V(s')$ is the best you can do from there. So the equation reads: **the value of being here equals the best you can do by adding this period's reward to the value of wherever you end up.** When we are minimizing cost instead of maximizing reward, the same equation holds with $\max$ replaced by $\min$ and $r$ a cost:

$$ V(s) = \min_{a} \Big[\, c(s, a) + \mathbb{E}\big[\, V(s')\,\big] \,\Big]. $$

The genius is that $V$ appears on *both* sides. It is defined in terms of itself one step in the future. That self-reference is not a problem — it is a recipe. It tells us that if we somehow knew $V$ for all the states one step ahead, we could compute $V$ for the current states by a simple one-period optimization. And we *do* know $V$ at the very end of the problem: at the final step there is no future, so the value is just the immediate cost of whatever we are forced to do. From that anchor we can walk backward. That walk is called backward induction, and it is the subject of the next section.

> The Bellman equation does not solve a multi-period problem. It *dissolves* it — into a chain of single-period problems, each one easy, each one trusting the next.

#### Worked example: writing the Bellman equation for a trade

You have 100,000 shares and 3 periods. The state is (period $t$, shares remaining $x$). The action is $n_t$, the number sold in period $t$. The cost of selling $n$ shares in one period is the impact function $c(n) = \eta n^2$ (a clean convex cost; $\eta$ is a small constant in dollars-per-share-squared). The next state is deterministic here: $x' = x - n$. The Bellman equation for this problem is

$$ V(t, x) = \min_{0 \le n \le x} \Big[\, \eta\, n^2 + V(t+1,\, x - n) \,\Big], $$

with the terminal condition $V(3, x) = \eta\, x^2$ — at the last period you must dump everything remaining, paying $\eta x^2$. We just turned "find the best of $10^{10}$ plans" into "solve three tiny one-variable minimizations." The lesson: **the Bellman equation is a translation rule that rewrites an impossible global search as a stack of trivial local ones, and the terminal condition is the brick the whole stack stands on.**

## Backward induction: solving the last step first

The Bellman equation says $V$ today depends on $V$ tomorrow. So we compute tomorrow first. We start at the very end of the problem — where the future is empty and the value is obvious — and we walk backward toward the present, filling in $V$ one period at a time. Each step is a one-period optimization that *reuses* the values we already computed for the future. This is **backward induction**, sometimes called *value iteration* in the finite-horizon case.

![Diagram of backward induction as a stack from the final step up to the start](/imgs/blogs/dynamic-programming-optimal-execution-math-for-quants-2.png)

The stack above shows the order of work. At the bottom is the final step: you must sell everything left, so its value is known with no thinking required. The step above it can now be solved, because it only needs to know the value of the final step — which we just computed. The step above *that* needs only the step below it, and so on up to the start. Notice the direction of computation (bottom-up, future-first) is the opposite of the direction of action (top-down, present-first). We *think* backward so we can *act* forward. That inversion is the single most common stumbling block for beginners, so it is worth saying plainly: you solve the problem end-to-start, then you execute it start-to-end.

### Why this is so much faster than brute force

Brute force checks every full plan: exponential in the number of periods. Backward induction visits each *state* once and does one cheap minimization there: it is *linear* in the number of states times the number of actions. For our 3-period, 100,000-share problem, brute force would consider on the order of billions of slicing combinations; backward induction solves it with three small optimizations. This collapse — from exponential to polynomial — is the whole reason dynamic programming earns its place in the toolbox.

#### Worked example: a 3-step backward induction by hand

Let us actually solve the problem from the previous example, with concrete numbers. You hold **9 units** (think of each unit as ~11,000 shares so the arithmetic stays friendly) and have **3 periods** to liquidate. The cost of selling $n$ units in a period is $c(n) = n^2$ dollars (we set $\eta = 1$ to keep the numbers clean). You must end with zero. Find the optimal schedule by backward induction.

**Step 3 (the last period).** Whatever you have left, $x$, you must sell. So the value is simply

$$ V_3(x) = x^2. $$

If you arrive at period 3 with 3 units, it costs $3^2 = \$9$. No choice, no optimization.

**Step 2 (second-to-last).** You hold $x$ units. You sell some amount $n$ now (cost $n^2$) and carry $x - n$ into period 3 (which will cost $(x-n)^2$). So

$$ V_2(x) = \min_{0 \le n \le x} \big[\, n^2 + (x - n)^2 \,\big]. $$

Take the derivative with respect to $n$ and set it to zero: $2n - 2(x - n) = 0 \Rightarrow n = x/2$. Selling *half* now is optimal. The cost is $V_2(x) = (x/2)^2 + (x/2)^2 = x^2/2$. So splitting the remaining position evenly across the last two periods halves the cost of dumping it all at once. Intuition check: convex cost punishes lumpiness, so you smooth it out.

**Step 1 (now).** You hold all **9 units**. You sell $n$ now (cost $n^2$) and carry $9 - n$ into period 2, which will then cost $V_2(9 - n) = (9 - n)^2 / 2$. So

$$ V_1(9) = \min_{0 \le n \le 9} \Big[\, n^2 + \tfrac{(9 - n)^2}{2} \,\Big]. $$

Differentiate: $2n - (9 - n) = 0 \Rightarrow 3n = 9 \Rightarrow n = 3$. So you sell **3 units now**. Carry 6 into period 2, where you sell half: **3 units**. Carry 3 into period 3 and sell the last **3 units**. The optimal schedule is **3, 3, 3** — perfectly even — and the total cost is $3^2 + 3^2 + 3^2 = \$27$.

Now compare to the naive "dump it" plan of selling all 9 in period 1: that costs $9^2 = \$81$. The backward-induction schedule costs \$27 — a **\$54 saving, two-thirds cheaper**, on a problem you solved with three lines of algebra. The lesson: **with a purely convex cost and no price risk, the optimal schedule is a flat line (sell equal amounts each period), and backward induction recovers it automatically without you having to guess.** Hold onto that "flat line" result — the moment we add *price risk*, it breaks, and the schedule tilts toward selling early. That tilt is the entire content of the Almgren-Chriss model.

## Value versus policy, and the curse of dimensionality

Two loose ends from the foundations deserve a full section, because they are where dynamic programming meets reality.

### Value iteration versus policy iteration

There are two classic ways to solve a Bellman equation. **Value iteration** is what we just did: repeatedly apply the Bellman update to the value function until it stops changing (in a finite-horizon problem, one backward sweep is enough). **Policy iteration** flips the order: start with any policy, evaluate its value exactly, then greedily improve the policy by picking the best action against that value, and repeat. Both converge to the same optimal pair $(V^*, \pi^*)$; policy iteration often converges in fewer big steps but each step is more expensive. For finite-horizon execution problems — a fixed trading day with a hard deadline — a single backward induction sweep is the natural and complete solution, so we will not need the iterative machinery. But it is worth knowing the names, because reinforcement learning (the data-driven cousin of dynamic programming) is built almost entirely on these two loops.

### The curse of dimensionality

Backward induction is linear in the number of states. The catch is that the number of states can be astronomically large. Add a second tradeable asset and the state becomes (shares of A left, shares of B left, time left); if each ranges over 1,000 values, that is a million states. Add a third asset and it is a billion. Track the current price level, the recent volatility, the order-book depth, and a momentum signal, and the state space explodes beyond any computer's memory. Bellman himself coined the phrase for this: the **curse of dimensionality** — the number of states grows exponentially in the number of state variables.

![Matrix showing impact cost rises with speed while price risk falls with speed](/imgs/blogs/dynamic-programming-optimal-execution-math-for-quants-4.png)

The matrix above previews where we are going and why the curse matters in practice. It shows the central trade-off of execution: trade fast and your *impact cost* (the price you push) is high while your *price risk* (exposure to a moving market) is low; trade slow and it is the reverse; the lowest *total* cost sits in the middle. A full dynamic program over a rich state — price, depth, volatility, signal — would solve this trade-off exactly, but it would drown in the curse of dimensionality. The genius of the Almgren-Chriss model, which we turn to next, is that it makes a few clever simplifying assumptions that let you write the optimal trade-off down in *closed form* — no giant state table, just a formula. It is dynamic programming's logic distilled into algebra. When the state is too big to tabulate, you either approximate the value function (this is what deep reinforcement learning does) or you find a special structure that gives a closed-form answer (this is what Almgren-Chriss does).

#### Worked example: how fast the curse bites

Suppose you discretize each state variable into just 100 buckets. With one variable (shares left) you have 100 states — trivial. With two variables, $100^2 = 10{,}000$ states — still easy. With four variables, $100^4 = 100{,}000{,}000$ states — a hundred million cells, each needing its own optimization. At, say, \$0.000001 of compute cost per cell, the one-variable problem costs a fraction of a cent and the four-variable problem costs about **\$100** *per backward sweep*, and a realistic model needs many sweeps and far finer buckets. The lesson: **adding one state variable multiplies the work by the number of buckets, so honest high-dimensional dynamic programming is usually impossible and the entire research field is about dodging the curse with approximations or special structure.**

## The execution problem: impact versus risk

Now we point all of this at the real problem. You are a portfolio manager — or the algorithm working on their behalf — and you need to sell a large block, say 100,000 shares of a stock trading around \$50, a \$5,000,000 position. You cannot just hit the market with all of it at once. Why not? Two forces pull in opposite directions, and the whole art is balancing them.

### Force one: market impact (the cost of trading fast)

When you sell, you are demanding liquidity — you need someone to buy. The buyers at the best price are limited; once you exhaust them, you have to accept lower and lower prices to keep selling. **Market impact** is the adverse price move *you yourself cause* by trading. The faster and bigger you trade, the more you push the price against yourself. There are two flavors, and the distinction is central:

- **Temporary impact** is the extra cost you pay *right now* for demanding immediate liquidity, which then fades once you stop pushing. It is a function of your *trading rate* — how aggressively you are selling at this moment. Trade twice as fast and your temporary impact roughly more than doubles (it is convex). Think of it as the premium for jumping the queue; once you stop jumping, the queue reforms and the price recovers.
- **Permanent impact** is the lasting shift in the price level that your trade leaves behind, because your selling signals information ("someone big wants out") that the market does not un-learn. It is a function of *total volume traded*, not rate. Under the standard linear assumption, permanent impact moves the price by an amount proportional to the total shares you sell — and crucially, since you have to sell the whole block no matter how you slice it, *permanent impact is the same for every schedule*. That is a subtle and important point: it means permanent impact does not affect the *shape* of the optimal schedule, only the overall baseline cost. The schedule is decided entirely by the tug-of-war between *temporary* impact and *risk*.

The takeaway: temporary impact says **slow down** — spread your trading out so you never demand too much liquidity at once.

### Force two: price risk (the cost of trading slow)

If impact were the only force, you would trade infinitely slowly to make impact vanish. But the longer you hold an unsold position, the longer you are exposed to the market simply *moving* — news, other traders, plain randomness — and moving against you. **Price risk** (also called *timing risk* or *volatility risk*) is the uncertainty in your final proceeds caused by the price wandering while you still hold inventory. It grows with the size of your leftover position and with how long you hold it. A risk-averse trader hates this uncertainty.

The takeaway: price risk says **speed up** — get the position sold before the market has time to move.

![Matrix and tree of speed versus cost showing the middle is cheapest](/imgs/blogs/dynamic-programming-optimal-execution-math-for-quants-4.png)

So we have a genuine tension. Impact says go slow; risk says go fast. The matrix above lays out the trade-off cell by cell: every row is a trading speed, every column a cost component, and the cheapest *total* lives in the middle row. Almgren and Chriss's contribution, in their celebrated 2000 paper "Optimal Execution of Portfolio Transactions," was to write this tension as a single objective and solve it exactly.

### The objective: expected cost plus a risk penalty

A risk-neutral trader would just minimize expected cost — and we will see that gives the boring "trade evenly" answer. A risk-averse trader minimizes a combination: **expected cost plus $\lambda$ times the variance of cost.** Formally, if $\mathbb{E}[C]$ is the expected total execution cost (shortfall) and $\mathrm{Var}[C]$ is its variance, the trader solves

$$ \min_{\text{schedule}} \;\; \mathbb{E}[C] \;+\; \lambda\, \mathrm{Var}[C]. $$

Every symbol: $C$ is the total cost of executing relative to the price when you started (the *implementation shortfall*); $\mathbb{E}[C]$ is its average; $\mathrm{Var}[C]$ is how uncertain that cost is; and $\lambda \ge 0$ is the **risk-aversion parameter** — the price, in units of expected dollars, that you are willing to pay to remove one unit of cost variance. When $\lambda = 0$ you only care about average cost (risk-neutral). As $\lambda$ grows you increasingly fear uncertainty and rush to finish. This is exactly the *mean-variance* trade-off from portfolio theory, applied to the cost of a trade rather than the return of a portfolio — the same machinery as the [mean-variance efficient frontier](/blog/trading/math-for-quants/mean-variance-efficient-frontier-math-for-quants) post, repurposed for execution.

## The Almgren-Chriss model, built up

Let us assemble the model from its parts, keeping the algebra honest but minimal. Split the trading day into $N$ equal periods of length $\tau$ (so the total time is $T = N\tau$). Let $x_k$ be the number of shares *still held* after period $k$, with $x_0 = X$ (the whole order) and $x_N = 0$ (you must finish). The amount you sell in period $k$ is $n_k = x_{k-1} - x_k$. The list $(x_0, x_1, \dots, x_N)$ is your **trajectory**, and the list of trades $(n_1, \dots, n_N)$ is your **schedule**.

### The two cost pieces in symbols

**Temporary impact cost.** Selling $n_k$ shares in a period of length $\tau$ means a trading rate of $n_k / \tau$. Under the standard linear temporary-impact model, each share you sell in that period suffers a price penalty proportional to the rate, so the cost of period $k$ is about $\eta\, n_k^2 / \tau$, where $\eta$ is the temporary-impact coefficient (in dollars per share, per unit rate). The square is the convexity we keep meeting: doubling the rate more than doubles the cost. Summed over the day, total temporary-impact cost is $\sum_k \eta\, n_k^2 / \tau$.

**Price risk.** While you hold $x_k$ shares through a period, the stock price wanders by a random amount with standard deviation $\sigma \sqrt{\tau}$ per share ($\sigma$ is the per-period volatility). The *variance* contributed by holding $x_k$ shares through period $k$ is $\sigma^2 \tau\, x_k^2$. Summed over the trajectory, total cost variance is $\mathrm{Var}[C] = \sigma^2 \tau \sum_k x_k^2$. The thing to notice: variance depends on the *holdings* $x_k$, so the longer you carry a big position, the more risk you accumulate.

### The objective and its solution

Put them together. Ignoring the permanent-impact term (which, recall, is the same for every schedule), the trader minimizes

$$ \underbrace{\sum_{k=1}^{N} \frac{\eta\, n_k^2}{\tau}}_{\text{expected impact cost}} \;+\; \lambda\, \underbrace{\sigma^2 \tau \sum_{k=1}^{N} x_k^2}_{\text{risk penalty}}. $$

This is a tidy quadratic optimization in the holdings $x_k$, and you can attack it two ways. The first is **dynamic programming** — exactly the backward induction we have been practicing — treating $x_k$ as the state and $n_k$ as the action; the value function turns out to be quadratic in shares-remaining, and the optimal action is linear in the state. The second is plain calculus: set the derivative with respect to each $x_k$ to zero. Both give the same answer, and it is famous.

The optimal trajectory satisfies a simple second-order relationship whose solution is an **exponential decay** of the remaining position:

$$ x_k \;=\; X\, \frac{\sinh\!\big(\kappa (T - t_k)\big)}{\sinh(\kappa T)}, \qquad \kappa \approx \sqrt{\frac{\lambda \sigma^2}{\eta}}. $$

Do not be intimidated by the $\sinh$ (hyperbolic sine); the shape it describes is intuitive. The number $\kappa$ (kappa) is the **urgency**, and it is the single most important quantity in the model. Look at what it is made of: $\kappa = \sqrt{\lambda \sigma^2 / \eta}$. Urgency goes *up* when you are more risk-averse ($\lambda$ large) or the stock is more volatile ($\sigma$ large) — both make waiting scary. Urgency goes *down* when temporary impact is large ($\eta$ large) — that makes rushing expensive. When $\kappa$ is large, the curve decays fast: you sell most of your position early. When $\kappa \to 0$ (risk-neutral, $\lambda = 0$), the curve becomes a straight line — and a straight-line trajectory means *equal-sized trades each period*, which is exactly **TWAP** (Time-Weighted Average Price). So TWAP is not a separate idea; it is the $\lambda = 0$ corner of Almgren-Chriss.

![Diagram contrasting a flat TWAP schedule with a front-loaded optimal schedule](/imgs/blogs/dynamic-programming-optimal-execution-math-for-quants-6.png)

The before-and-after above shows the two schedules side by side. TWAP sells the same quantity every period — a flat profile that minimizes impact but leaves a big position exposed to risk for most of the day. The risk-averse optimal schedule front-loads: it sells more early to shrink the position fast, then tapers off as there is less left to protect. The front-loading is precisely the exponential decay of $x_k$. If you ever wondered why a "smart" execution algorithm sells aggressively at the open and quietly at the close, this curve is the reason. (For the family of practical algorithms — VWAP, TWAP, POV — built on top of this logic, see the [execution algorithms deep-dive](/blog/trading/quantitative-finance/execution-algorithms-vwap-twap-pov-quant-research).)

#### Worked example: TWAP versus the optimal schedule, in dollars

Let us put real numbers on it. You must sell **100,000 shares** of a \$50 stock (a \$5,000,000 position) over **5 periods**. Use a temporary-impact coefficient such that selling $n$ shares in a period costs $\eta n^2$ with $\eta = \$0.000004$ per share-squared (so selling 20,000 in one period costs $0.000004 \times 20{,}000^2 = \$1{,}600$). The per-period price volatility is $\sigma = \$0.40$ per share, and you are moderately risk-averse with $\lambda$ chosen so that the urgency works out to a schedule that front-loads noticeably.

**TWAP plan.** Sell 20,000 shares each period. Impact cost per period $= 0.000004 \times 20{,}000^2 = \$1{,}600$, so total impact cost $= 5 \times \$1{,}600 = \$8{,}000$. Now the risk. The holdings after each period are 80,000; 60,000; 40,000; 20,000; 0. The cost variance is $\sigma^2 \sum x_k^2 = 0.40^2 \times (80{,}000^2 + 60{,}000^2 + 40{,}000^2 + 20{,}000^2 + 0)$. The sum of squares is $6.4{+}3.6{+}1.6{+}0.4 = 12.0$ billion, so $\mathrm{Var} = 0.16 \times 12.0\text{B} = 1.92\text{B}$ dollars², giving a cost standard deviation of $\sqrt{1.92\text{B}} \approx \$43{,}800$. That is the risk a TWAP plan leaves on the table: the cost could swing tens of thousands of dollars because so much position is held for so long.

**Risk-averse optimal plan.** With a moderate urgency, the optimal schedule front-loads — roughly 30,000; 24,000; 19,000; 15,000; 12,000 shares (these sum to 100,000 and follow the exponential-decay shape). Impact cost $= 0.000004 \times (30{,}000^2 + 24{,}000^2 + 19{,}000^2 + 15{,}000^2 + 12{,}000^2) = 0.000004 \times (900{+}576{+}361{+}225{+}144)\text{M} = 0.000004 \times 2{,}206\text{M} = \$8{,}824$. So impact cost rises by about \$824 versus TWAP — that is the price of speed. But the holdings now decay much faster: 70,000; 46,000; 27,000; 12,000; 0, with sum of squares $4.9{+}2.116{+}0.729{+}0.144 = 7.889$ billion, so $\mathrm{Var} = 0.16 \times 7.889\text{B} = 1.262\text{B}$ and the cost standard deviation drops to $\approx \$35{,}500$.

So the risk-averse trader pays about **\$824 more in expected impact** to cut the cost standard deviation from \$43,800 to \$35,500 — roughly **\$8,300 less risk** for \$824 of certain cost. Whether that is worth it depends on $\lambda$, but a moderately risk-averse desk happily takes that deal. If we value the risk reduction at the trader's own $\lambda$, the total objective improves; and in the tail, where the market moves against you, the front-loaded plan can easily save **tens of thousands of dollars** — on a single \$5,000,000 order the gap between a thoughtless and a thoughtful schedule is real money. The lesson: **TWAP minimizes the certain part of the cost (impact) but ignores the uncertain part (risk); Almgren-Chriss trades a small, certain increase in impact for a large reduction in risk, and the right balance is set by how much you fear uncertainty.**

## How urgency reshapes the whole schedule

The parameter $\lambda$ — and through it the urgency $\kappa$ — is the dial that turns one model into a whole family of behaviors. Let us walk the extremes and the middle, because this is where intuition really clicks.

![Diagram contrasting a greedy schedule with a Bellman-optimal schedule](/imgs/blogs/dynamic-programming-optimal-execution-math-for-quants-5.png)

The before-and-after above contrasts a *greedy* plan with the *Bellman-optimal* plan, and it is worth dwelling on the difference because beginners conflate them. A greedy plan minimizes the cost of *this step only* — it grabs the locally cheapest action and ignores the future. The Bellman-optimal plan minimizes the cost of the *whole path*, pricing in every future step before it acts. Greedy is myopic; Bellman is far-sighted. In execution, a purely greedy "minimize impact right now" plan would trade as slowly as possible and leave a huge position exposed — a disaster if the market moves. The Bellman plan looks ahead, sees the accumulating risk, and accepts a little more impact now to avoid a lot of risk later. That foresight is the entire value-add of dynamic programming over naive rules.

### The three regimes

**Risk-neutral ($\lambda = 0$, $\kappa = 0$): trade evenly.** With no fear of uncertainty, the only thing to minimize is impact, and (as our 3-step worked example showed) convex impact is minimized by spreading trades evenly. The trajectory is a straight line down to zero; the schedule is TWAP. This is the patient extreme.

**Infinitely risk-averse ($\lambda \to \infty$, $\kappa \to \infty$): trade immediately.** With overwhelming fear of price moves, you dump the entire position in the first instant to eliminate all risk, accepting enormous impact cost. The trajectory is a cliff to zero. This is the panicked extreme — and it is, in effect, what the Flash Crash algorithm approximated by trading on volume without regard to price impact.

**Moderate $\lambda$: front-loaded exponential decay.** In between — where every real desk lives — the trajectory is the smooth $\sinh$ curve: sell aggressively early while there is a lot to protect, taper off as the position shrinks. The half-life of the position is governed by $1/\kappa$. A higher $\kappa$ (more volatile stock, more risk-averse trader, smaller impact) means a shorter half-life and a more front-loaded plan.

![Timeline of an execution trajectory decaying from open to close](/imgs/blogs/dynamic-programming-optimal-execution-math-for-quants-7.png)

The timeline above traces a moderate-urgency trajectory across an actual trading day: 100,000 shares at the open, decaying through 62,000, 35,000, 15,000, and finally flat at the close. The position never sits still — it bleeds down continuously, fastest at the start when there is most to lose and slowest at the end. That curve *is* the optimal policy made visible. Every execution algorithm you will ever use is, internally, trying to track some version of this line while adapting to the real order book as it goes.

#### Worked example: how $\lambda$ changes the expected shortfall

Take the same 100,000-share, 5-period setup. We will compare three urgencies and report each plan's expected impact cost and its risk (cost standard deviation), then the risk-adjusted objective $\mathbb{E}[C] + \lambda \mathrm{Var}[C]$.

| Urgency | Schedule (shares per period) | Expected impact | Cost std-dev | Character |
|---|---|---|---|---|
| $\lambda = 0$ (TWAP) | 20k, 20k, 20k, 20k, 20k | \$8,000 | \$43,800 | patient, low impact, high risk |
| moderate $\lambda$ | 30k, 24k, 19k, 15k, 12k | \$8,824 | \$35,500 | balanced, front-loaded |
| high $\lambda$ | 50k, 25k, 13k, 8k, 4k | \$15,776 | \$28,200 | urgent, high impact, low risk |

Read the table top to bottom. As $\lambda$ rises, the schedule tilts harder toward the front: the first-period trade climbs from 20,000 to 50,000 shares. Expected impact cost rises (from \$8,000 to \$15,776 — almost double, because slamming 50,000 shares into the first period is expensive) while risk falls (cost std-dev drops from \$43,800 to \$28,200). The high-urgency plan pays an extra \$7,776 of certain impact to remove about \$15,600 of risk versus TWAP. Whether that trade is good depends entirely on your $\lambda$: a desk liquidating ahead of a feared earnings shock will gladly pay it; a desk with a quiet stock and no deadline will not. The lesson: **there is no single "optimal" schedule — there is an optimal schedule for each level of risk aversion, and choosing $\lambda$ is choosing where on the impact-versus-risk frontier you want to live.**

## Optimal stopping: when to act, not just how much

Execution asks "how much to trade each period." A close cousin asks "*when* to trade at all" — should I sell now, or wait one more period in hope of a better price? This is the theory of **optimal stopping**, and it is dynamic programming with a special action set: in every state you choose either *stop* (take the reward now) or *continue* (pay nothing, move to the next state, and decide again). It governs when to exercise an American option, when a market-maker pulls a quote, and when a patient seller finally lets go.

### The stopping rule is a threshold

The Bellman equation for stopping compares two values in each state: the **stopping value** (what you get if you act now) and the **continuation value** (the expected best value if you wait). You stop the moment the stopping value exceeds the continuation value. Because the continuation value typically falls as a deadline approaches (less time left means less chance of improvement), the optimal rule is almost always a **threshold**: stop as soon as the price (or some signal) crosses a level, and that level drifts over time. This is the same logic behind an American option's *early-exercise boundary* — exercise the moment the stock crosses a price that moves as expiry nears.

#### Worked example: sell now or wait one period

You hold one share, currently worth **\$100**, and you may sell now for \$100 or wait exactly one more period. If you wait, the price will be **\$110 with probability 0.5** or **\$94 with probability 0.5** — but waiting costs you a **\$1** carrying/holding cost (financing, the risk of being wrong), and after that one period you must sell at whatever the price is. Should you sell now or wait?

**Stopping value (sell now):** \$100, clean.

**Continuation value (wait one period):** the expected price next period minus the holding cost. Expected price $= 0.5 \times \$110 + 0.5 \times \$94 = \$55 + \$47 = \$102$. Subtract the \$1 holding cost: continuation value $= \$102 - \$1 = \$101$.

Since \$101 (wait) > \$100 (sell now), the Bellman rule says **wait** — the expected gain from the upside more than covers the holding cost. Now change one number: suppose the holding cost were \$3 instead of \$1. Then continuation value $= \$102 - \$3 = \$99 < \$100$, and the rule flips to **sell now**. The *threshold* holding cost where you are indifferent is exactly \$2 (the expected price improvement). The lesson: **optimal stopping reduces a hard "when do I act" question to a simple comparison — act the instant the value of acting now beats the expected value of waiting — and the break-even point is a clean, computable threshold.**

This is the same comparison an American-option holder makes every day, and the same one an execution algorithm makes when it decides whether to post a passive limit order (wait for a better price, risk non-execution) or cross the spread immediately (act now, pay the spread). Optimal stopping and optimal execution are two faces of the same Bellman coin.

## Merton's problem: dynamic programming in continuous time

Everything so far has been *discrete* — a finite list of periods. The same logic works when time flows continuously, and the bridge from discrete dynamic programming to continuous-time **stochastic control** is one of the crown jewels of mathematical finance: **Merton's portfolio problem.**

In 1969 Robert Merton asked: if you have wealth to invest and consume over a lifetime, with a risky stock and a safe bond, how much should you hold in stocks and how much should you spend, at every instant, to maximize your total happiness (utility) over your life? It is the ultimate multi-period decision under uncertainty — exactly our setting, but with infinitely many infinitesimal periods. The discrete Bellman equation, $V(s) = \max_a [r(s,a) + \mathbb{E} V(s')]$, becomes a differential equation called the **Hamilton-Jacobi-Bellman (HJB) equation** as the period length shrinks to zero. Solving it, Merton found a startlingly clean answer: the optimal fraction of wealth in the risky asset is constant — $w^* = (\mu - r)/(\gamma \sigma^2)$, the excess return over the safe rate, divided by your risk aversion times variance. (That formula is, not coincidentally, the same shape as the one-asset mean-variance weight.) The deep point for us: **continuous-time optimal control is just dynamic programming taken to the limit, and the Bellman equation is its discrete shadow.** Almgren-Chriss can itself be written in continuous time, where the optimal trajectory is exactly the $\sinh$ decay we derived, now as the solution to an HJB equation rather than a backward sweep.

## Common misconceptions

**"Dynamic programming means writing dynamic, flexible code."** No. The name is a historical accident — Bellman chose it to sound impressive to funders. Dynamic programming is a *mathematical method* for breaking a multi-stage problem into nested sub-problems and reusing their solutions. It has nothing to do with software being "dynamic," and you can do it entirely on paper, as we did with the 3-step example.

**"The optimal way to trade a big order is to spread it evenly (TWAP)."** Only if you are risk-neutral. Even slicing minimizes *impact* cost, but it leaves a large position exposed to the market for most of the trading window. The moment you care about the *variance* of your cost — and every real desk does — the optimal schedule front-loads, selling more early to shrink the risky position faster. TWAP is the $\lambda = 0$ special case, not the general answer.

**"Market impact is just the bid-ask spread."** The spread is the smallest piece. Impact is the price you *move* by demanding liquidity beyond what is resting at the top of the book — and it has a temporary part (fades when you stop) and a permanent part (a lasting repricing because your trade revealed information). For a large order, impact dwarfs the spread; a \$5,000,000 order can cost far more in impact than in spread, which is precisely why optimal execution exists as a discipline.

**"A greedy 'cheapest-now' rule is good enough."** Greedy ignores the future, and in execution that is dangerous. A greedy minimize-impact rule trades as slowly as possible and accumulates risk; a greedy minimize-risk rule dumps everything and pays enormous impact. The Bellman-optimal plan is far-sighted by construction — it prices every future consequence before acting now — which is the whole reason it beats simple rules.

**"Permanent impact changes the optimal schedule's shape."** Under the standard linear model, it does not. Because you must sell the entire block regardless of how you slice it, the *total* permanent impact is identical for every schedule. It raises the baseline cost but does not tilt the trajectory. The shape is decided purely by the tug-of-war between *temporary* impact and *risk*. (Nonlinear permanent impact, used in some research models, does change this — but the linear baseline is where intuition should start.)

**"A bigger state space always means a better model."** Adding state variables (price, depth, volatility, signals) can make a model more realistic, but it runs straight into the curse of dimensionality — the state count explodes exponentially and the problem becomes uncomputable. Sometimes a small, closed-form model like Almgren-Chriss beats a sprawling one you cannot actually solve. Parsimony is a feature, not a compromise.

## How it shows up in real markets

### 1. The 2010 Flash Crash

On May 6, 2010, a large mutual-fund complex used an automated algorithm to sell about \$4.1 billion of E-mini S&P 500 futures, and the algorithm was configured to target a percentage of volume *without* regard to price or time. As prices fell and volume spiked, the algorithm — chasing the rising volume — sold *faster*, feeding the decline. The Dow dropped roughly 1,000 points in minutes. The official SEC-CFTC report singled out the execution logic: a schedule that ignored impact and price, the exact failure mode our model is built to prevent. An Almgren-Chriss-style controller, aware of impact and risk, would have throttled selling as the market thinned. The lesson the whole industry took: *how* you slice an order is a risk decision, not a clerical one.

### 2. Why institutional desks run VWAP and Implementation-Shortfall algos

Almost every large asset manager today routes big orders through execution algorithms named VWAP, TWAP, POV (percentage-of-volume), and "Implementation Shortfall" (IS). The IS algorithm is Almgren-Chriss in production: it explicitly minimizes expected shortfall plus a risk penalty, with a trader-set "urgency" slider that is exactly our $\lambda$. Brokers publish the urgency dial as low/medium/high; behind it sits $\kappa = \sqrt{\lambda\sigma^2/\eta}$ choosing the exponential-decay rate. A 2012-era study of large-cap US equity orders found implementation-shortfall algos saved tens of basis points versus naive scheduling on big orders — on a \$100,000,000 order, 10 basis points is \$100,000. The math in this post is, quite literally, a line item on institutional P&L.

### 3. Index-rebalance "trading at the close"

When an index like the S&P 500 adds or drops a name, index funds must trade enormous quantities at the closing auction to match the index. These are deadline-constrained liquidations — finish by the close, no exceptions — which is the pure Almgren-Chriss setting with a hard terminal time. Desks model the closing-auction impact, the pre-close volatility, and their tracking-error tolerance (their $\lambda$) to decide how much to pre-trade in the days and hours before versus how much to leave for the auction. Getting the schedule wrong shows up directly as tracking error, the metric index funds are judged on. The September and December "quad-witching" rebalances move billions and are choreographed with exactly this math.

### 4. Crypto liquidations and on-chain slippage

In decentralized finance, market impact is brutally visible: an automated market maker prices each trade along a curve, so a large swap moves the price mechanically and the "impact" is literally computed by the formula. When a large leveraged position is liquidated on-chain, the liquidation bot faces the same impact-versus-speed trade-off — dump it fast and eat huge slippage, or work it slowly and risk the collateral falling further. The May 2021 and November 2022 crypto crashes featured cascading liquidations where forced fast selling crushed prices, which triggered more liquidations — a feedback loop that is the Flash Crash mechanism with the impact function made explicit by the AMM. The order-book and impact intuition transfers directly; see the [order book simulator deep-dive](/blog/trading/quantitative-finance/order-book-simulator-quant-research) for how depth and impact interact tick by tick.

### 5. Market makers and the optimal-stopping side

A market maker quoting both sides of the book is constantly solving an optimal-stopping problem: hold the current quote (continue) or pull and re-price (stop), given inventory risk and adverse-selection risk. The Avellaneda-Stoikov market-making model, an industry reference, is a continuous-time stochastic-control problem in the exact Merton/HJB family — it solves for optimal bid and ask offsets that depend on inventory and time-to-close, the same state variables we have used. When a maker's inventory grows lopsided, the model skews quotes to encourage trades that flatten it — dynamic programming managing risk in real time. See the [market-making simulator deep-dive](/blog/trading/quantitative-finance/market-making-simulator-quant-research) for how this plays out in a live book.

### 6. American option early exercise

Every American-style option embeds an optimal-stopping problem: at each moment, exercise now (stopping value = intrinsic value) or wait (continuation value = expected discounted future payoff). The early-exercise boundary — the stock price at which it becomes optimal to exercise — is computed by backward induction on a binomial tree or by solving the associated free-boundary PDE. For a deep in-the-money American put, exercising early to capture interest on the strike can be worth real money, and the boundary tells you exactly when. This is dynamic programming pricing roughly trillions of dollars of listed options, every day, on the same Bellman logic we used to sell 9 toy units.

### 7. Algorithmic execution during the 2020 COVID volatility spike

In March 2020, equity volatility ($\sigma$) spiked to multi-year highs. Recall urgency $\kappa = \sqrt{\lambda\sigma^2/\eta}$ scales with $\sigma$: when volatility triples, urgency rises sharply, and the optimal schedule front-loads much harder. Desks that ran fixed-schedule TWAP/VWAP during those weeks left far too much position exposed to enormous intraday swings and ate large timing losses; desks whose IS algos adapted their urgency to the live volatility finished faster and bled less. It was a real-time demonstration that $\lambda$ and $\sigma$ belong inside the schedule, not bolted on after — the optimal plan *must* respond to volatility, and the ones that did measurably outperformed.

## When this matters to you

If you ever place a large order — even a "large for you" order in an illiquid small-cap or a thinly traded crypto token — you are facing a tiny version of this exact problem. Selling your whole position with one market order pays the full impact; working it in pieces over a few minutes or hours can save real money, and the right speed depends on how much you fear the price moving while you wait. That intuition — *slice it, front-load if you're nervous, and never dump a big order into a thin market* — is the practical residue of everything above, and it costs nothing to apply. (This is educational, not financial advice; the right action depends on your situation.)

More broadly, dynamic programming is the most transferable idea in this whole series. The same Bellman equation that schedules a trade also plans a robot's path, allocates a budget over a lifetime, sequences a course of medical treatment, and trains every reinforcement-learning agent that plays a game. Once you can recognize the loop — state, action, reward, next state — and the backward-induction trick that solves it, you will see it everywhere.

Where to go next from here:

- **The original source:** Robert Almgren and Neil Chriss, "Optimal Execution of Portfolio Transactions" (Journal of Risk, 2000) — the paper that started the field; surprisingly readable, and the $\sinh$ trajectory we derived is its centerpiece.
- **The execution-algorithm family in practice:** the [VWAP, TWAP, and POV deep-dive](/blog/trading/quantitative-finance/execution-algorithms-vwap-twap-pov-quant-research) shows how the schedules in this post become production order routers.
- **The microstructure beneath impact:** the [order book simulator](/blog/trading/quantitative-finance/order-book-simulator-quant-research) and the [market-making simulator](/blog/trading/quantitative-finance/market-making-simulator-quant-research) show, tick by tick, where market impact actually comes from and how the optimal-stopping side of the same coin plays out for a liquidity provider.
- **The optimization toolkit:** the mean-variance objective at the heart of Almgren-Chriss is the same one in the [mean-variance efficient frontier](/blog/trading/math-for-quants/mean-variance-efficient-frontier-math-for-quants) post — execution is portfolio theory pointed at the cost of a single trade.
- **For the continuous-time leap:** read about Merton's portfolio problem and the Hamilton-Jacobi-Bellman equation, the bridge from the discrete backward induction here to the full machinery of stochastic optimal control.
