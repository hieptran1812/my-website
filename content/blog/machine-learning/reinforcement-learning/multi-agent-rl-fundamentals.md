---
title: "Multi-Agent RL: When Multiple Agents Share the World"
date: "2026-06-22"
publishDate: "2026-06-22"
description: "A from-scratch, code-first guide to multi-agent reinforcement learning: why single-agent methods break, the Markov-game formalism, non-stationarity, and the CTDE paradigm that actually works in production."
tags:
  [
    "reinforcement-learning",
    "deep-learning",
    "multi-agent",
    "markov-decision-process",
    "machine-learning",
    "pytorch",
    "game-theory",
    "rllib",
  ]
category: "machine-learning"
subcategory: "Reinforcement Learning"
author: "Hiep Tran"
featured: true
readTime: 55
image: "/imgs/blogs/multi-agent-rl-fundamentals-1.png"
---

The first time I shipped two RL agents into the same environment, I expected twice the trouble. What I got was a wholly different category of failure. Each agent, trained alone, had converged beautifully: a market-making bot that learned to quote tight spreads against a fixed simulated counterparty, and a hedging bot that learned to lay off inventory against a fixed price process. Put them in the same simulated order book — let each one's actions become the other's environment — and both diverged within a few thousand steps. The market-maker would widen spreads to avoid getting picked off; the hedger, seeing wider spreads, would trade more aggressively to beat the next widening; the market-maker would react by widening more. Two policies that were each individually optimal chased each other into a corner where neither made money. Nothing in single-agent RL had prepared me for this. The bug was not in either agent. The bug was that I had assumed the environment was stationary when it was, in fact, learning back.

That is the entire story of multi-agent reinforcement learning (MARL) in one paragraph. The moment more than one learning agent shares a world, the ground shifts under each of them. The reward an agent sees for a given action depends on what every other agent is doing, and every other agent is changing what it does as it learns. The convergence proofs you trusted in single-agent RL — the ones that guarantee Q-learning finds the optimal value function, that policy gradient ascends a fixed objective — quietly stop applying, because they all assumed a fixed environment. MARL is what you do once you accept that the environment is not fixed, that it is, in a precise sense, an adversary or a teammate or an ambivalent crowd, and that you must learn anyway.

This post builds MARL from first principles. We will start from the single-agent Markov Decision Process you already know, generalize it to the **Markov game** (also called the stochastic game), and watch exactly where and why the single-agent guarantees break. We will formalize the three interaction regimes — cooperative, competitive, and mixed — and the **Dec-POMDP**, the partially observable formalism that real deployments actually live in. We will diagnose the three structural challenges of MARL — non-stationarity, credit assignment, and scalability — and then build the dominant engineering answer to all three: **Centralized Training with Decentralized Execution (CTDE)**. Along the way you will get runnable RLlib `MultiAgentEnv` code, an independent Q-learning baseline you can copy, two fully worked numerical examples, and an honest account of when MARL is the wrong tool and a single centralized controller wins. By the end you should be able to set up a multi-agent training run, reason about why it might oscillate instead of converge, and pick an algorithm family from the structure of your problem.

![Diagram of a Markov game showing a joint state branching into each agent's partial observation, each agent producing an action, and the joint action plus state feeding a shared transition function that produces the next state](/imgs/blogs/multi-agent-rl-fundamentals-1.png)

This is the eleventh-or-so stop in a long series on reinforcement learning, and it sits at a junction. If you have not internalized the single-agent loop — agent observes a state, picks an action from a policy, receives a reward, lands in a new state, and updates the policy to get more reward — start with the unified map of the series at `/blog/machine-learning/reinforcement-learning/reinforcement-learning-a-unified-map`, then come back. Everything here is that same loop, run by many agents at once, and the whole subject is the study of what goes wrong when you do that.

## 1. Why single-agent RL breaks when agents share a world

Let me make the failure precise before we formalize anything, because the formalism only earns its keep once you feel the problem in your hands.

In single-agent RL, the agent faces a **Markov Decision Process (MDP)**: a tuple $(\mathcal{S}, \mathcal{A}, P, R, \gamma)$ where $\mathcal{S}$ is the state space, $\mathcal{A}$ the action space, $P(s' \mid s, a)$ the transition probability of landing in state $s'$ after taking action $a$ in state $s$, $R(s,a,s')$ the reward, and $\gamma \in [0,1)$ the discount factor. The single most important property of an MDP is that $P$ and $R$ are **fixed functions**. The dynamics of the world do not change while you learn. This is exactly the property that makes the Bellman equation a contraction and makes Q-learning converge: you are fitting a value function to a stationary target.

Now drop a second learning agent into the world. From agent 1's seat, the transition it experiences is no longer $P(s' \mid s, a_1)$. It is

$$
P^{\pi_2}(s' \mid s, a_1) = \sum_{a_2} \pi_2(a_2 \mid s)\, P(s' \mid s, a_1, a_2),
$$

a transition function that is **marginalized over agent 2's policy** $\pi_2$. The instant agent 2 updates $\pi_2$ — which it does on every gradient step — the effective transition $P^{\pi_2}$ that agent 1 is trying to learn against changes. Agent 1 is fitting a value function to a target that is sliding out from under it. The Markov property holds for the *joint* system, but from the *individual* agent's local perspective, the environment is non-Markovian and non-stationary. This is not a numerical inconvenience you can tune away with a smaller learning rate. It is a violation of the central assumption every single-agent convergence proof relies on.

The trading anecdote in the intro is exactly this. Each bot treated the other as part of a fixed environment, so each kept best-responding to a snapshot of the other that was already stale. The result was an oscillation — formally, a failure to reach a fixed point — that no amount of single-agent hyperparameter tuning would have fixed, because the problem was structural.

Three concrete consequences follow, and they organize the rest of this post:

1. **Moving targets (non-stationarity).** The thing you are learning against is itself learning. Convergence is not guaranteed and oscillation is common.
2. **Credit assignment.** When a team of agents shares one reward signal, which agent's action actually earned it? A single scalar reward for the whole team gives you almost no gradient information about any individual.
3. **Scalability.** The joint action space is the *product* of the individual action spaces. Two agents with 5 actions each is 25 joint actions; ten agents with 5 actions each is nearly 10 million. Anything that reasons over the joint action space explodes.

We will solve, or at least seriously dent, each of these. But first, the formalism that lets us talk about them precisely.

## 2. The Markov game: the right formalism

The generalization of the MDP to many agents is the **Markov game** (Littman, 1994), also called the **stochastic game** (Shapley introduced stochastic games back in 1953, long before RL existed). It is a tuple

$$
\big(\, n,\ \mathcal{S},\ \{\mathcal{A}_i\}_{i=1}^{n},\ P,\ \{R_i\}_{i=1}^{n},\ \gamma \,\big)
$$

with these pieces:

- $n$: the number of agents.
- $\mathcal{S}$: the set of states of the *world* (shared by all agents).
- $\mathcal{A}_i$: the action space of agent $i$. The **joint action space** is the Cartesian product $\mathcal{A} = \mathcal{A}_1 \times \mathcal{A}_2 \times \cdots \times \mathcal{A}_n$. A joint action is a tuple $\mathbf{a} = (a_1, \ldots, a_n)$.
- $P(s' \mid s, \mathbf{a})$: the transition function. Critically, the next state depends on the **joint** action, not any single agent's action.
- $R_i(s, \mathbf{a}, s')$: a **separate reward function for each agent** $i$. This is the heart of it. Each agent has its own reward, and the relationship between those rewards defines whether the game is cooperative, competitive, or mixed.
- $\gamma$: the discount factor, usually shared.

Each agent $i$ has a policy $\pi_i(a_i \mid s)$ (or $\pi_i(a_i \mid o_i)$ in the partially observed case we get to in Section 4). Agent $i$ wants to maximize its own expected discounted return

$$
J_i(\pi_1, \ldots, \pi_n) = \mathbb{E}\!\left[ \sum_{t=0}^{\infty} \gamma^t\, R_i(s_t, \mathbf{a}_t, s_{t+1}) \right],
$$

and the crucial subtlety is that $J_i$ depends on **all** the policies, not just $\pi_i$. You cannot optimize $J_i$ without making an assumption about what the other agents do. That single fact is why MARL borrows so heavily from game theory: the right notion of "solution" is no longer "the optimal policy" but an **equilibrium** — a joint policy where no agent can improve its own return by unilaterally changing its policy. That is a **Nash equilibrium** of the game.

### How it reduces to an MDP when n = 1

Set $n = 1$. The joint action space collapses to $\mathcal{A} = \mathcal{A}_1$, the transition becomes $P(s' \mid s, a_1)$, and there is a single reward $R_1$. The tuple is exactly $(\mathcal{S}, \mathcal{A}_1, P, R_1, \gamma)$ — an ordinary MDP. The Markov game is a strict generalization: every MDP is a one-player Markov game, and "find the optimal policy" is a degenerate Nash equilibrium because with one player, the best response to nobody is just the optimum. This is reassuring. Nothing we learned in single-agent RL is wrong; it is the $n=1$ special case of something larger.

The figure above shows the Markov-game loop. The shared world state branches out to each agent (in the partially observed case, to each agent's local observation), each agent independently produces an action, those actions are bundled into a joint action, and the joint action plus current state feed the single shared transition function $P$ that produces the next world state. The branch-out-and-merge structure is the whole point: many policies, one world.

#### Worked example: a two-agent grid Markov game

Let me ground the tuple in numbers. Two robots share a $3 \times 3$ grid. State $s$ is the pair of positions, so $|\mathcal{S}| = 9 \times 9 = 81$. Each robot has $|\mathcal{A}_i| = 5$ actions (up, down, left, right, stay), so the joint action space is $|\mathcal{A}| = 5 \times 5 = 25$. There is a goal cell that gives a reward.

If the task is **cooperative** — both robots get $R_1 = R_2 = +1$ when *either* reaches the goal, and $-0.1$ per step otherwise — the agents should coordinate so one heads to the goal while the other stays out of the way. If the task is **competitive** — only the first robot to the goal gets $+1$ and the other gets $-1$ — they should race and block. Same $\mathcal{S}$, same $\mathcal{A}$, same $P$; *different reward structure*, completely different optimal behavior. That is the lever the reward functions $\{R_i\}$ give you, and the next section makes the three regimes precise.

## 3. The three interaction regimes: cooperative, competitive, mixed

The relationship between the reward functions $\{R_i\}$ partitions Markov games into three regimes. Getting this classification right is the single most useful diagnostic step before you pick an algorithm, because each regime has a different solution concept and a different set of tools that work.

![A layered stack figure showing the three multi-agent interaction regimes — fully cooperative with a shared reward, fully competitive zero-sum, and mixed general-sum — stacked from most aligned to least aligned](/imgs/blogs/multi-agent-rl-fundamentals-2.png)

**Fully cooperative.** All agents share a single reward: $R_1 = R_2 = \cdots = R_n = R$. The team maximizes one joint return. This is the friendliest regime because there is no conflict of interest — the only difficulty is coordination and credit assignment, not adversity. Examples: a fleet of warehouse robots, a team of traffic signals minimizing total city-wide delay, cooperative navigation where agents must reach goals without colliding. The cooperative case has a special name when fully observed and shared-reward: a **Multi-agent MDP (MMDP)**, and when partially observed, a **Dec-POMDP** (Section 4). Algorithms: value decomposition (VDN, QMIX), MAPPO, COMA.

**Fully competitive (zero-sum).** Two agents with exactly opposed rewards: $R_1 = -R_2$. One agent's gain is the other's loss. This is the classical game-theory setting — chess, Go, poker, the minimax world. The solution concept is the **minimax equilibrium**: each agent plays to maximize its worst-case return against an adversary playing to minimize it. The beautiful thing about zero-sum two-player games is that they have strong theoretical structure (the minimax theorem guarantees a value), which is why so much of the AlphaGo/AlphaZero lineage lives here. Algorithms: minimax-Q, self-play, MCTS-based methods.

**Mixed (general-sum).** Each agent has its own arbitrary reward $R_i$, with no fixed relationship. There is partial cooperation and partial competition. This is the messiest and, frankly, the most realistic regime — it is where the world actually lives. Examples: autonomous vehicles at an intersection (they want to avoid crashing, which is cooperative, but each wants to get through first, which is competitive); financial markets (traders are not pure adversaries, but they are not teammates either); social dilemmas like the iterated prisoner's dilemma and the tragedy of the commons. General-sum games can have multiple Nash equilibria, some good for everyone and some terrible, and there is no guarantee learning finds a good one. Algorithms: Nash-Q, MADDPG, independent learners with caution.

Here is the comparison as a table you can keep on your desk.

| Property | Single-agent MDP | Cooperative MARL | Competitive MARL (zero-sum) | Mixed (general-sum) |
|---|---|---|---|---|
| Reward structure | one $R$ | shared $R_1 = \cdots = R_n$ | opposed $R_1 = -R_2$ | independent $R_i$ |
| Environment stationarity | stationary | non-stationary | non-stationary | non-stationary |
| Solution concept | optimal policy | optimal joint policy | minimax equilibrium | Nash equilibrium (may be many) |
| Credit assignment | trivial (one agent) | hard (shared reward) | moderate (own reward) | moderate-to-hard |
| Typical algorithms | DQN, PPO, SAC | QMIX, MAPPO, VDN, COMA | minimax-Q, self-play, MCTS | MADDPG, Nash-Q, IPPO |
| Real-world example | one robot arm | warehouse robot fleet | Go, chess, poker | self-driving at intersections |

The "stationarity" column is the punchline of Section 1: every multi-agent column is non-stationary, and that is the tax you pay for sharing a world. The next figure makes that classification operational as a matrix.

![A matrix comparing cooperative, competitive, and mixed multi-agent regimes across reward structure, Nash solution concept, typical algorithm, and a real-world example for each regime](/imgs/blogs/multi-agent-rl-fundamentals-3.png)

## 4. The Dec-POMDP: the formalism real deployments live in

The Markov game in Section 2 assumed each agent sees the full world state $s$. Real systems almost never give you that. A warehouse robot sees what its own sensors see, not the position and intent of every other robot. A trader sees the public order book and its own inventory, not every competitor's book. The correct formalism for cooperative MARL under partial observability is the **Decentralized Partially Observable Markov Decision Process (Dec-POMDP)** (Bernstein et al., 2002).

A Dec-POMDP adds an observation layer to the cooperative Markov game:

$$
\big(\, n,\ \mathcal{S},\ \{\mathcal{A}_i\},\ P,\ R,\ \{\Omega_i\},\ O,\ \gamma \,\big)
$$

where the new pieces are:

- $\Omega_i$: the observation space of agent $i$. Agent $i$ receives a local observation $o_i \in \Omega_i$, **not** the full state $s$.
- $O(\mathbf{o} \mid s', \mathbf{a})$: the observation function, giving the probability that the joint observation is $\mathbf{o} = (o_1, \ldots, o_n)$ after the joint action $\mathbf{a}$ lands the world in state $s'$.
- $R$: a single shared reward (the "Dec" formalism is by definition cooperative — there is one team reward).

Let me lay out the full seven-tuple so nothing is left implicit, because every piece earns its place and the later algorithms are best understood as approximations to one part of it:

$$
\big(\, \mathcal{S},\ \{\mathcal{A}_i\}_{i=1}^{n},\ T,\ \{R_i\}_{i=1}^{n},\ \{\Omega_i\}_{i=1}^{n},\ O,\ \gamma \,\big)
$$

- $\mathcal{S}$ — the set of true world states. No agent observes $s \in \mathcal{S}$ directly; it is the latent variable everyone is implicitly reasoning about.
- $\{\mathcal{A}_i\}$ — the per-agent action sets, with joint action $\mathbf{a} = (a_1, \ldots, a_n) \in \mathcal{A}_1 \times \cdots \times \mathcal{A}_n$.
- $T(s' \mid s, \mathbf{a})$ — the state-transition kernel, driven by the *joint* action exactly as in the Markov game.
- $\{R_i\}$ — in the strict Dec-POMDP these collapse to a single shared team reward $R(s, \mathbf{a})$ (the "Dec" formalism is cooperative by definition); I write the indexed form to keep the parallel to the Markov game visible, with the understanding that $R_1 = \cdots = R_n = R$ here.
- $\{\Omega_i\}$ — the per-agent observation sets. Agent $i$ draws $o_i \in \Omega_i$, never $s$.
- $O(\mathbf{o} \mid s', \mathbf{a})$ — the observation kernel, giving the probability of the joint observation $\mathbf{o} = (o_1, \ldots, o_n)$ after $\mathbf{a}$ lands the world in $s'$. This is the new machinery relative to the Markov game; it is what severs each agent from the true state.
- $\gamma \in [0,1)$ — the shared discount.

The defining difficulty: each agent must choose its action $a_i$ as a function only of its **own observation history** $\tau_i = (o_i^0, a_i^0, o_i^1, a_i^1, \ldots, o_i^t)$, because at execution time it has no access to anyone else's observations or to the true state. A policy is now $\pi_i(a_i \mid \tau_i)$ — conditioned on a *history*, because a single observation is generally not enough (the partial observability means the agent must integrate information over time, exactly as in a single-agent POMDP).

The object the team is actually optimizing is the **joint policy** $\boldsymbol{\pi} = (\pi_1, \ldots, \pi_n)$, the tuple of every agent's individual history-conditioned policy. There is no single controller picking $\mathbf{a}$; the joint action emerges from $n$ separate decisions, $a_i \sim \pi_i(\cdot \mid \tau_i)$, made on $n$ disjoint information streams. The value of a joint policy is the expected discounted team return,

$$
V^{\boldsymbol{\pi}}(s_0) = \mathbb{E}_{\boldsymbol{\pi},\, T,\, O}\!\left[ \sum_{t=0}^{\infty} \gamma^t\, R(s_t, \mathbf{a}_t) \;\Big|\; s_0 \right],
$$

where the expectation runs over the transition kernel $T$, the observation kernel $O$ (which decides what each agent gets to see and therefore which $\tau_i$ it conditions on), and the joint policy $\boldsymbol{\pi}$ itself. The Dec-POMDP solution is the joint policy that maximizes $V^{\boldsymbol{\pi}}$. Write the same backup recursively and the partial-observability tax becomes explicit: an agent cannot condition its value on $s$, so the natural state for a *single* agent is a *belief* $b_i(s) = \Pr(s \mid \tau_i)$ over the true state, maintained from its own history alone. The backup over joint belief is

$$
V^{\boldsymbol{\pi}}(\boldsymbol{\tau}) = \mathbb{E}_{s \sim b(\boldsymbol{\tau})}\!\left[ R(s, \mathbf{a}) + \gamma \sum_{\mathbf{o}'} \Pr(\mathbf{o}' \mid \boldsymbol{\tau}, \mathbf{a})\, V^{\boldsymbol{\pi}}(\boldsymbol{\tau}, \mathbf{a}, \mathbf{o}') \right],
$$

with $\mathbf{a}$ drawn from $\boldsymbol{\pi}(\boldsymbol{\tau})$ and $\boldsymbol{\tau}$ the joint history. The killer is that no single agent has access to the joint history $\boldsymbol{\tau}$ at execution — agent $i$ only has $\tau_i$ — so each agent must implicitly reason about what the *others* have likely seen, a belief over the other agents' beliefs. That nested "belief over beliefs" is the structural reason the value backup does not factorize cleanly per agent, and it is the formal seed of the NEXP-completeness result we cite next.

Why does this matter so much? Because the Dec-POMDP is provably hard. Solving a finite-horizon Dec-POMDP optimally is **NEXP-complete** (Bernstein et al., 2002) — dramatically harder than the P-complete single-agent MDP. The intuition is that each agent must reason about what every other agent has likely observed and will likely do, given that nobody can see the true state, and that recursive belief-about-beliefs reasoning blows up. We do not solve Dec-POMDPs optimally in practice; we approximate them, and the approximation that works is CTDE (Section 8). For now, the takeaway is: **the realistic MARL problem is partially observed, and partial observation is what makes decentralized coordination genuinely hard.**

#### Worked example: why local observation forces coordination protocols

Two delivery drones must cover two depots, A and B, one drone each, and they share a reward of $+10$ if both depots are covered and $0$ otherwise (one depot uncovered, or both drones at the same depot, earns nothing). If both drones saw the full state, the optimal joint policy is trivial: drone 1 always takes A, drone 2 always takes B. Coordination by fiat.

Now make it a Dec-POMDP: each drone sees only its own GPS, not the other's. If both drones run the same deterministic policy "go to the nearest depot," and they start equidistant, they may both pick A and earn 0. They cannot break the symmetry without information about each other. The fixes are exactly the MARL coordination toolkit: (a) **role assignment baked into the policy** (drone 1's policy differs from drone 2's — symmetry broken at training time), (b) **communication** (drone 1 broadcasts "I have A," Section 9), or (c) **a learned convention** that emerges from joint training under CTDE. The expected reward goes from $0.5 \times 10 = 5$ under naive symmetric play to $10$ under any of the three fixes. Partial observability turned a trivial problem into a coordination problem, and that gap is the whole subject.

## 5. Non-stationarity, formalized: why the convergence proofs break

We have circled non-stationarity twice; now let me pin down exactly which proof breaks and why, because understanding the mechanism is what lets you predict and diagnose oscillation in your own runs.

![A before-and-after comparison contrasting single-agent learning with a stationary environment and convergence guarantee against multi-agent learning with a non-stationary environment and no convergence guarantee](/imgs/blogs/multi-agent-rl-fundamentals-4.png)

Recall why single-agent Q-learning converges. Q-learning iterates the **Bellman optimality operator** $\mathcal{T}$:

$$
(\mathcal{T} Q)(s,a) = \mathbb{E}_{s'}\!\left[ R(s,a,s') + \gamma \max_{a'} Q(s',a') \right].
$$

The convergence proof rests on two facts: (1) $\mathcal{T}$ is a **$\gamma$-contraction** in the sup norm, so by the Banach fixed-point theorem it has a unique fixed point $Q^*$ and repeated application converges to it; and (2) the stochastic-approximation conditions (Robbins–Monro: learning rates summing to infinity but their squares summing to finite) let the sampled, noisy version converge to the same fixed point almost surely. Both facts assume $P$ and $R$ — and therefore $\mathcal{T}$ — are **fixed**.

In a Markov game, from agent $i$'s perspective the effective operator is

$$
(\mathcal{T}_i^{\boldsymbol{\pi}_{-i}} Q_i)(s,a_i) = \mathbb{E}_{\mathbf{a}_{-i} \sim \boldsymbol{\pi}_{-i},\, s'}\!\left[ R_i(s, a_i, \mathbf{a}_{-i}, s') + \gamma \max_{a_i'} Q_i(s', a_i') \right],
$$

where $\boldsymbol{\pi}_{-i}$ denotes the policies of all agents except $i$. This operator depends on $\boldsymbol{\pi}_{-i}$. When the other agents learn, $\boldsymbol{\pi}_{-i}$ changes, so the operator $\mathcal{T}_i^{\boldsymbol{\pi}_{-i}}$ changes, so its fixed point moves. You are no longer iterating one contraction toward one fixed point; you are chasing a fixed point that relocates every time anyone updates. The Banach argument requires a *single* fixed operator, and you do not have one. **That is the precise place the proof breaks.** Not the Robbins–Monro conditions, not the function approximation — the contraction-toward-a-fixed-target structure itself dissolves.

### The Non-Stationarity Problem in Detail

It is worth slowing all the way down on a single update, because the failure is sharper than "the operator drifts" — the standard Q-learning *target* is literally not a valid estimate of anything stable. In single-agent Q-learning, after observing the transition $(s, a, r, s')$ you form the bootstrap target

$$
y = r + \gamma \max_{a'} Q(s', a'),
$$

and the justification is airtight: $y$ is an unbiased sample of $(\mathcal{T}Q)(s,a)$, and $\mathcal{T}$ has a fixed point $Q^*$ that $y$ is pulling you toward, because the only randomness in $r$ and $s'$ comes from the *fixed* kernels $R$ and $P$. Now run the identical update for agent $i$ in a Markov game. Agent $i$ observes $(s, a_i, r, s')$ — but it does **not** observe $\mathbf{a}_{-i}$, the other agents' actions, even though both $r$ and $s'$ were generated by the full joint action $(a_i, \mathbf{a}_{-i})$. The reward it saw was really $R_i(s, a_i, \mathbf{a}_{-i}, s')$ and the next state was really drawn from $P(s' \mid s, a_i, \mathbf{a}_{-i})$. So when agent $i$ writes down

$$
y = r + \gamma \max_{a_i'} Q_i(s', a_i'),
$$

it is implicitly treating $\mathbf{a}_{-i}$ as if it were marginalized out under the *current* opponent policies $\boldsymbol{\pi}_{-i}^{(t)}$. The target it is fitting is

$$
y \approx \mathbb{E}_{\mathbf{a}_{-i} \sim \boldsymbol{\pi}_{-i}^{(t)}}\!\big[ R_i(s, a_i, \mathbf{a}_{-i}) \big] + \gamma\, \mathbb{E}_{\mathbf{a}_{-i} \sim \boldsymbol{\pi}_{-i}^{(t)}}\!\big[ \textstyle\max_{a_i'} Q_i(s', a_i') \big],
$$

and the superscript $(t)$ is the whole problem. The expectation is taken under the opponents' policy *at the moment the sample was collected*. One gradient step later the opponents have updated to $\boldsymbol{\pi}_{-i}^{(t+1)}$, and the very same $(s, a_i)$ now has a different correct value, because both the reward distribution and the transition distribution it induces have shifted. There is no fixed $Q_i^*$ for the target to converge to — the target is a moving average of a quantity whose own definition is being rewritten under you. This is the moving-target math made literal: agent $i$ is doing regression onto a label that is a function of a variable ($\boldsymbol{\pi}_{-i}$) it cannot see and that changes every step.

Concretely, suppose at step $t$ the opponent plays action L with probability $0.8$ and R with $0.2$, and against L our action yields $+1$ while against R it yields $-1$. Agent $i$'s sampled targets average toward $0.8(+1) + 0.2(-1) = +0.6$, so $Q_i(s, a_i)$ climbs toward $+0.6$. Now the opponent learns and flips to L with probability $0.2$: the true value of the same $(s, a_i)$ is now $0.2(+1) + 0.8(-1) = -0.6$. Agent $i$'s Q-value, sitting at $+0.6$, is not merely imprecise — it has the *wrong sign*, and the greedy policy it induces is now actively bad. Nothing about a smaller learning rate fixes a target whose ground truth inverted; you would just track the inversion more slowly.

This is also exactly why naive **experience replay breaks** in the multi-agent setting, and it is worth seeing it as a direct corollary rather than a separate phenomenon. A replay buffer's entire premise is that a transition $(s, a_i, r, s')$ stored at step $t_0$ is still a valid sample of the same target distribution when you replay it at step $t_0 + k$. In single-agent RL that premise holds because $R$ and $P$ never changed. In MARL the stored $r$ and $s'$ were generated under $\boldsymbol{\pi}_{-i}^{(t_0)}$ — a *snapshot of opponent behavior that no longer exists*. Replaying it trains agent $i$ to predict the value of actions against opponents it will never face again. The buffer is not a sample of one stationary distribution; it is a blended sample of every opponent policy that ever lived during the buffer's window, and the Bellman regression averages over all of them incoherently. The deeper the buffer, the staler the oldest opponent snapshots, and the worse the blend — which is the precise reason MARL practitioners run short buffers, fingerprint transitions with the collection-time iteration, or importance-weight by an estimate of the opponents' policy then versus now.

There are three practical mitigations people use, none a full cure:

- **Slow the moving target.** Make other agents' policies change slowly relative to your own learning (e.g., target networks, low learning rates, lagged self-play opponents). If the environment is *approximately* stationary on the timescale of your updates, the contraction approximately holds.
- **Centralize the critic (CTDE).** If the value function conditions on the *joint* action and all observations, then from the critic's view the world *is* stationary (the joint policy fully determines dynamics). This is the deep reason CTDE works — it restores stationarity at training time. We build this in Section 8.
- **Experience replay needs care.** Off-policy replay buffers store transitions generated under *old* opponent policies, which are now doubly stale. Vanilla DQN replay is theoretically broken in MARL; fixes include importance weighting, fingerprinting transitions with the training iteration, or short buffers.

The before/after figure above is the mental picture: on the left, one policy ascends a fixed landscape to a guaranteed peak; on the right, $n$ policies each climb a landscape that the others are reshaping, so there is no guaranteed peak — only, at best, an equilibrium where everyone has stopped being able to climb.

There is a deeper way to see why this is more than a quantitative inconvenience, and it is worth a moment because it predicts the *kind* of failure you will see. Think of the joint learning dynamics as a vector field in the space of all the agents' policy parameters: at each point, every agent's gradient pushes its own parameters uphill on its own objective. In single-agent RL, the gradient is the gradient of a scalar function, so the dynamics are a *gradient flow* — they always make monotone progress on a fixed potential and they cannot cycle. In MARL, the combined update is a sum of gradients of *different* objectives, and such a vector field generally is **not** the gradient of any single potential function. Vector fields that are not gradients can rotate. That rotation is precisely the oscillation in my trading example and in the textbook "rock-paper-scissors" learning loop, where each agent's best response chases the others around a cycle forever. So the right intuition is not "convergence is slower"; it is "the dynamics can be rotational rather than convergent, and you will see limit cycles where you expected a fixed point." Recognizing a rotational limit cycle (returns oscillating with a stable period rather than wandering) versus genuine divergence (returns blowing up) versus slow convergence (returns climbing with decreasing amplitude) is one of the most useful diagnostic skills in multi-agent training, and it is invisible if you only ever watch a single smoothed reward curve. Plot per-agent returns separately, and look for anti-correlated oscillation between agents — that is the fingerprint of the non-gradient rotation.

A second, subtler failure that follows from the same root is **catastrophic relearning** under replay. Picture agent 1 with a replay buffer half-full of transitions generated when agent 2 played an old, now-abandoned strategy. Agent 1 trains on a mixture of "what agent 2 does now" and "what agent 2 used to do," and the Bellman target it fits is an incoherent average of two different environments. The symptom is a value function that never sharpens — the loss floors out at a stubbornly high value and the policy never commits. The fix in practice is to keep MARL replay buffers short (so stale opponent behavior ages out quickly), or to *fingerprint* each transition with the training iteration or an estimate of the opponents' policy at collection time so the network can condition on which environment generated the sample. These are not theoretical niceties; they are the difference between a run that converges and one that grinds at a high loss for a million steps. Off-policy methods are the most exposed because the whole point of a replay buffer is to reuse old data, and in MARL "old data" silently means "data from a different environment."

## 6. Independent Q-learning: the naive baseline that sometimes works

The simplest possible thing you can do in MARL is to ignore the problem entirely: give each agent its own single-agent learner and let it treat all other agents as part of the environment. For Q-learning this is **Independent Q-Learning (IQL)** (Tan, 1993); the deep-network version is **Independent DQN (IDQN)**, and the PPO version is **Independent PPO (IPPO)**. Each agent $i$ maintains its own $Q_i(o_i, a_i)$ and updates it with the ordinary single-agent rule, treating the observed reward and next-observation as if they came from a stationary MDP.

The theory says this should not work: the environment is non-stationary, so the per-agent Q-learning has no convergence guarantee. And yet IQL works surprisingly often in practice — well enough that it is the right *first* thing to try and a mandatory baseline before you reach for anything fancier. Why does it sometimes work? Because in many problems the other agents' policies change slowly enough, or the coupling between agents is weak enough, that each agent's effective environment is approximately stationary. When that approximation is good, the contraction in Section 5 approximately holds and IQL approximately converges. When it is not — tightly coupled, adversarial, or fast-changing — IQL oscillates or diverges, exactly as the theory warns.

Here is a complete, runnable IQL implementation for a cooperative grid world. It uses tabular Q-learning with one Q-table per agent. I am deliberately keeping the environment minimal so the *learning* logic is the focus.

```python
import numpy as np

class TwoAgentGrid:
    """Two agents on a 1-D corridor of length L. Shared reward of +1 when
    BOTH agents stand on the two goal cells (0 and L-1) simultaneously,
    -0.05 per step otherwise. Cooperative Markov game with full observation."""
    def __init__(self, L=5):
        self.L = L
        self.goals = (0, L - 1)

    def reset(self):
        self.pos = [self.L // 2, self.L // 2]  # both start in the middle
        return tuple(self.pos)

    def step(self, actions):
        # action 0 = move left, 1 = stay, 2 = move right
        for i, a in enumerate(actions):
            move = {0: -1, 1: 0, 2: +1}[a]
            self.pos[i] = int(np.clip(self.pos[i] + move, 0, self.L - 1))
        covered = {self.pos[0], self.pos[1]} == set(self.goals)
        reward = 1.0 if covered else -0.05
        done = covered
        return tuple(self.pos), reward, done

def independent_q_learning(episodes=20000, L=5, alpha=0.1, gamma=0.95):
    env = TwoAgentGrid(L)
    n_states = L * L              # joint position encoded as a single index
    n_actions = 3
    # ONE Q-table PER AGENT — this is the "independent" in IQL.
    Q = [np.zeros((n_states, n_actions)) for _ in range(2)]
    eps = 1.0
    returns = []
    for ep in range(episodes):
        s = env.reset()
        idx = s[0] * L + s[1]
        total, done, steps = 0.0, False, 0
        while not done and steps < 50:
            actions = []
            for i in range(2):
                if np.random.rand() < eps:
                    actions.append(np.random.randint(n_actions))
                else:
                    actions.append(int(np.argmax(Q[i][idx])))
            s2, r, done = env.step(actions)
            idx2 = s2[0] * L + s2[1]
            for i in range(2):
                a = actions[i]
                # Each agent bootstraps off ITS OWN Q-table only.
                target = r + (0.0 if done else gamma * Q[i][idx2].max())
                Q[i][idx, a] += alpha * (target - Q[i][idx, a])
            idx, total, steps = idx2, total + r, steps + 1
        eps = max(0.05, eps * 0.9995)
        returns.append(total)
    return Q, returns

Q, returns = independent_q_learning()
print(f"mean return, last 500 eps: {np.mean(returns[-500:]):.3f}")
```

On this small, weakly coupled cooperative task, IQL converges fine — the two agents learn to split to the two ends, and the mean episode return climbs from roughly $-2.5$ (random flailing for the full 50 steps) to about $+0.7$ (reaching the goal in a handful of steps). The key line is the per-agent update inside the loop: each agent bootstraps off its *own* Q-table and never looks at the other's, which is precisely what makes it "independent" and precisely what makes it non-stationary. Try cranking the coupling — say, a reward that penalizes the agents for being on the same cell *and* requires them to alternate goals turn by turn — and you will watch IQL start to oscillate. That oscillation is your cue to graduate to a centralized-critic method.

Let me trace that oscillation step by step on a tightly coupled grid, because watching it happen once makes the moving-target math from Section 5 viscerally concrete. Take a corridor with a single narrow doorway in the middle that only one agent can pass at a time, and a shared reward that pays off only when *both* agents have crossed to the far side. The coordination problem is "who goes first," and there are two good joint conventions — *A first, then B* or *B first, then A* — but a collision (both rushing the doorway in the same step) costs the team a penalty. Now run IQL and follow the Q-values:

1. **Round 1 — A learns to go.** Early on, by chance, agent A's exploration finds that rushing the doorway sometimes pays (B happened to wait). A's $Q_A(\text{doorway}, \text{go})$ climbs. From B's seat, the environment in which "B goes" was evaluated assumed an A that *waited*; that environment no longer exists. B's Q-values are now stale — they were fit against a passive A.
2. **Round 2 — B best-responds, A goes stale.** B, still maximizing against its (now-wrong) belief that A waits, also learns to go — and the two start colliding in the doorway, eating the penalty. B's collisions teach B to back off and wait, so $Q_B(\text{doorway}, \text{go})$ falls and $Q_B(\cdot, \text{wait})$ rises. But the moment B settles into waiting, the environment A trained against — a B that *also* rushed — is gone. Now *A's* Q-values are stale.
3. **Round 3 — A re-best-responds, B goes stale.** With B reliably waiting, A's "go" is even better than A first estimated, so A commits harder to going first. Fine — except A's growing aggression now makes B's earlier "wait" overcautious; B starts to think it can sneak through first, nudges $Q_B(\cdot, \text{go})$ back up, and we are back at the collisions of round 2.

Each agent is forever best-responding to a snapshot of the other that its own update just invalidated. A's improvement makes B's Q-values stale; B's correction makes A's Q-values stale; the loop closes and the per-agent returns oscillate with a stable period instead of settling. This is the rotational dynamics of Section 5 in a concrete grid: neither agent is buggy, neither is diverging, but the pair never agrees on a convention because nothing in IQL lets either one *see* that the other has changed. A centralized critic that conditions on both agents' actions collapses the whole loop — it sees the joint "A goes, B waits" as a single stationary configuration to be valued, with no stale snapshot to chase — which is exactly the fix the next sections build.

#### Worked example: IQL convergence on a coordination game versus a matching-pennies game

Let me make the "sometimes works, sometimes oscillates" claim precise with two stage games, because the contrast is the cleanest way to feel where IQL's luck runs out. Strip away the grid and consider a single-state repeated game between two agents, each with two actions, learning their action values with the same tabular update.

First, a **coordination game**: both agents get reward $+1$ if they pick the *same* action (both A or both B) and $0$ otherwise. There are two pure Nash equilibria — (A, A) and (B, B) — and they are both *good*. IQL handles this beautifully: whichever joint action they happen to stumble into early gets reinforced for both agents simultaneously (because the reward is shared and aligned), the Q-values for the matching action climb, exploration decays, and they lock in. Starting from random play (expected reward $0.5$), within a few hundred episodes both agents' Q-value for the locked-in action exceeds the other by a clear margin and the realized reward sits at $1.0$. The environment each agent faces *is* approximately stationary here because the other agent quickly stops changing — they reinforce each other's choice. This is the regime where the contraction in Section 5 approximately holds.

Now flip one sign to get **matching pennies**: agent 1 wins ($R_1 = +1$, $R_2 = -1$) if the actions *match*, and agent 2 wins ($R_1 = -1$, $R_2 = +1$) if they *differ*. This is zero-sum with no pure equilibrium; the only Nash is the mixed strategy where each plays 50/50. Run IQL and watch it fail to settle: agent 1 learns that A is currently winning, so it plays A; agent 2 learns that against A it should play B (to differ), so it switches to B; now agent 1's A is losing, so it switches to B; agent 2 chases. The Q-values rotate, the policies cycle, and the empirical action frequencies orbit the 50/50 point without ever resting on it. Neither agent's environment is stationary — each is the direct cause of the other's regret. This is the rotational limit cycle from Section 5 made concrete, and it is exactly why competitive settings need minimax-Q (which solves for the mixed equilibrium at each state via a small linear program) rather than naive independent maximization. The lesson generalizes: IQL's success is a measure of how *aligned and slow-moving* your agents' incentives are, not a property you can assume.

## 7. The three challenges, and the shape of their solutions

Before we build CTDE, let me lay the three structural challenges side by side, because the rest of the post is essentially a tour of how the field attacks each one. The figure below shows the credit-assignment challenge specifically — a single joint reward fanning out to per-agent contributions and a counterfactual baseline merging them back — but all three challenges deserve a clear statement.

![A graph showing a single joint reward fanning out into per-agent contributions for three agents and a counterfactual difference-reward baseline merging those contributions back into a per-agent credit signal](/imgs/blogs/multi-agent-rl-fundamentals-5.png)

**Challenge 1: Non-stationarity (moving targets).** Covered in Section 5. The other agents are learning, so the environment each agent faces is non-stationary, and the convergence proofs break. *Shape of the solution:* restore stationarity at training time by centralizing the value function over the joint action and all observations (CTDE), and/or slow the other agents' rate of change.

**Challenge 2: Credit assignment.** In a cooperative team with a single shared reward $R$, the team scored a goal — but which agent's action made it happen? If three soccer-playing agents share a reward and the team scores, a naive learner credits all three equally, including the two that stood still. This is the **multi-agent credit assignment problem**, and it is what makes shared-reward learning slow and noisy. *Shape of the solution:* compute a per-agent credit signal. Two clean ideas dominate:

- **Difference rewards / counterfactual baseline.** Credit agent $i$ with how much *worse* the team would have done had agent $i$ taken a default action instead of its actual one. Write $G(s, \mathbf{a})$ for the global team reward (or return) under joint action $\mathbf{a}$, and let $c_i$ be a fixed *default* or *clamped* action for agent $i$. The **difference reward** is

  $$
  D_i = G(s, \mathbf{a}) - G\!\left(s, \mathbf{a}_{-i}, c_i\right),
  $$

  the global outcome with agent $i$'s real action minus the global outcome where everyone else acted identically but agent $i$ was swapped to the default $c_i$. The second term is the **counterfactual baseline** — "what would the team have scored without me?" — and subtracting it leaves exactly agent $i$'s marginal contribution. The property that makes this more than a heuristic: the baseline $G(s, \mathbf{a}_{-i}, c_i)$ does **not** depend on $a_i$, so subtracting it changes neither the sign nor the argmax of agent $i$'s gradient — it is a variance-reducing control variate that leaves the optimum intact, the multi-agent analogue of the baseline in single-agent REINFORCE. The COMA algorithm (Foerster et al., 2018) computes exactly this, replacing the fixed clamp $c_i$ with the *expectation* over agent $i$'s own current policy, so the counterfactual advantage is $A_i = Q(s, \mathbf{a}) - \sum_{a_i'} \pi_i(a_i' \mid \tau_i)\, Q(s, \mathbf{a}_{-i}, a_i')$ — evaluated cheaply because the centralized critic already conditions on $\mathbf{a}_{-i}$ and only $a_i$ has to be swept.

  Why does naive per-agent reward fail, and fail *worse* the denser and more cooperative the task is? Hand every agent the raw shared team reward $R(\mathbf{a})$ and you have given $n$ agents the same scalar — a gradient signal with no information about *who* moved it. In a dense cooperative task where the team reward changes every step, agent $i$'s update is dominated by the variance of the other $n-1$ agents' actions: when the team reward jumps from $+2$ to $+5$, a still agent that did nothing gets credited for the $+3$ exactly as much as the agent that actually caused it. The agent cannot tell its own good actions from a teammate's good actions wrapped in its own noise, so its policy gradient is mostly noise that grows with $n$. The difference reward cancels precisely that noise: the $G(s, \mathbf{a}_{-i}, c_i)$ term contains all the teammate-driven variation, so $D_i$ responds *only* to agent $i$'s own action. This is why a soccer team of three sharing one reward learns glacially under raw reward but cleanly under difference rewards — the standing players' $D_i \approx 0$ (swapping their idleness for the default changes nothing) while the scorer's $D_i$ is large and positive.
- **Value decomposition.** Factor the joint value into per-agent pieces: $Q_{\text{tot}}(s, \mathbf{a}) \approx \sum_i Q_i(o_i, a_i)$ in VDN (Sunehag et al., 2018), or a monotonic mixing $Q_{\text{tot}} = f_{\text{mix}}(Q_1, \ldots, Q_n; s)$ with $\partial Q_{\text{tot}} / \partial Q_i \ge 0$ in QMIX (Rashid et al., 2018). The monotonicity constraint guarantees that the action maximizing each agent's local $Q_i$ also maximizes the joint $Q_{\text{tot}}$, so decentralized greedy execution is consistent with centralized training.

**Challenge 3: Scalability.** The joint action space $|\mathcal{A}| = \prod_i |\mathcal{A}_i|$ grows exponentially in $n$. Any method that explicitly enumerates joint actions (joint Q-learning, Nash-Q with full payoff matrices) is hopeless past a handful of agents. *Shape of the solution:* avoid the joint action space. Decentralized execution (each agent picks its own action from its own policy in $\mathcal{O}(\sum_i |\mathcal{A}_i|)$ rather than $\mathcal{O}(\prod_i |\mathcal{A}_i|)$ time), parameter sharing across homogeneous agents (one network serves all agents of the same type), and mean-field approximations (each agent reacts to the *average* of its neighbors rather than each one individually) all sidestep the explosion.

### Joint versus factored policies

Challenge 3 is sharp enough to deserve its own framing, because the choice it forces — *one policy over the joint action, or one policy per agent* — is the structural fork that everything downstream hangs on. The **joint policy** is the monolithic object $\pi(\mathbf{a} \mid s)$: a single distribution over the entire joint action space $\mathcal{A} = \mathcal{A}_1 \times \cdots \times \mathcal{A}_n$, picking the whole tuple $\mathbf{a} = (a_1, \ldots, a_n)$ at once. It is the "correct" object in the sense that the optimal joint behavior is, in general, *not* representable as $n$ independent choices — coordinated actions (both robots turning the same valve, two cars yielding in a consistent order) are correlations that an independent product cannot express. If you could afford it, a joint policy is what you would learn.

You cannot afford it. The joint action space has size $|\mathcal{A}| = \prod_i |\mathcal{A}_i|$, exponential in the number of agents. The same $10$ agents with $5$ actions each from Section 1 give a joint policy a softmax over $5^{10} \approx 9.8$ million outputs — a network head that is intractable to represent, sample from, or take an argmax over, and that demands data exponential in $n$ to even cover. This is the exponential blowup, and it is fatal past a handful of agents.

The **factored policy** is the escape: approximate the joint policy as a product of per-agent policies conditioned on local information,

$$
\pi(\mathbf{a} \mid s) \;\approx\; \prod_{i=1}^{n} \pi_i(a_i \mid o_i),
$$

so each agent carries a small head over its *own* $|\mathcal{A}_i|$ actions and the total parameter and sampling cost is $\mathcal{O}\!\big(\sum_i |\mathcal{A}_i|\big)$ — *linear* in $n$, not exponential. The factorization is exactly what makes decentralized execution possible in the first place (each agent samples $a_i$ from its own head with no global argmax), and it is why every scalable MARL method is, at execution time, factored. The cost is representational: a pure product of independent factors cannot express correlated joint actions, so the agents can fail to coordinate on the tasks that most need it. This is precisely the tension CTDE resolves — it lets you *train* the factored policies with a centralized signal that understands the joint structure (so the factors learn to be correlated *in expectation* through shared training), while keeping the cheap factored form at deployment. Factored policies are necessary for scale; centralized training is what keeps them from being naive.

Notice that CTDE — centralized training, decentralized execution — attacks Challenge 1 (stationarity at training time) and Challenge 3 (cheap execution) simultaneously, and the value-decomposition and counterfactual ideas that live inside CTDE attack Challenge 2. That is why CTDE is not just one technique among many; it is the architectural backbone that lets all three solutions coexist. Let me make it concrete.

## 8. Centralized training, decentralized execution (CTDE)

Here is the central engineering idea of modern MARL, and it resolves the apparent contradiction between "I need global information to learn well" and "I only have local information at deployment."

**At training time**, you have a luxury you will not have in deployment: you control the simulator, so you can see *everything* — the true state, every agent's observation, every agent's action. Use it. Train a **centralized critic** $Q(s, a_1, \ldots, a_n)$ or $V(s)$ that conditions on the full joint information. Because this critic sees the joint action, the environment is fully determined and therefore **stationary from the critic's perspective** — it directly fixes Challenge 1. The critic provides a low-variance, well-grounded learning signal to each agent's policy.

**At execution time**, you throw the critic away and keep only the **decentralized actors** $\pi_i(a_i \mid o_i)$ — one per agent, each conditioned on its own local observation. Execution is cheap (no joint action space, Challenge 3 solved) and deployable (no need for a global-information channel that does not exist in the real warehouse or order book).

![A layered figure showing centralized training where the full state and all agents' actions feed a centralized critic, the critic then frozen, and decentralized execution where each agent's local observation feeds only its own actor](/imgs/blogs/multi-agent-rl-fundamentals-6.png)

The phrase "centralized training, decentralized execution" was crystallized by the MADDPG paper (Lowe et al., 2017), and it is now the default paradigm. The trick that makes it sound rather than a hack is this: the critic is only used to compute *gradients* for the actors during training. As long as each actor's gradient is a valid (low-variance, correctly-centered) estimate of how to improve that actor's own return, it does not matter that the actor cannot see what the critic saw. The actor learns a decentralized policy that is *good in expectation* over the joint behavior the centralized critic understood. The critic is scaffolding; you remove it once the building stands.

Concretely, the MADDPG actor update for agent $i$ is a deterministic policy gradient that flows through the centralized critic $Q_i$:

$$
\nabla_{\theta_i} J(\mu_i) = \mathbb{E}\!\left[ \nabla_{\theta_i} \mu_i(o_i)\, \nabla_{a_i} Q_i^{\mu}(s, a_1, \ldots, a_n)\big|_{a_i = \mu_i(o_i)} \right],
$$

where $\mu_i$ is agent $i$'s deterministic actor, and $Q_i^{\mu}$ is the centralized critic for agent $i$ that takes *all* agents' actions as input. Each agent has its own centralized critic in MADDPG (so it handles the general-sum case where rewards differ); in fully cooperative MAPPO or QMIX, the critic is shared because the reward is shared.

Here is a compact PyTorch sketch of a centralized critic with decentralized actors — the structural skeleton of CTDE, stripped of the replay-buffer and target-network plumbing so the architecture is legible.

```python
import torch
import torch.nn as nn

class DecentralizedActor(nn.Module):
    """Sees ONLY its own local observation -> picks its own action.
    This is what survives to execution time."""
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.ReLU(),
            nn.Linear(64, act_dim),
        )
    def forward(self, local_obs):
        return torch.tanh(self.net(local_obs))   # continuous action in [-1, 1]

class CentralizedCritic(nn.Module):
    """Sees the FULL state and ALL agents' actions -> one Q value.
    Used only during training; discarded at deployment."""
    def __init__(self, state_dim, total_act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + total_act_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )
    def forward(self, full_state, all_actions):
        x = torch.cat([full_state, all_actions], dim=-1)
        return self.net(x)

def actor_loss(actor_i, critic_i, full_state, local_obs_i,
               others_actions):
    # One CTDE actor update for agent i (the heart of MADDPG).
    a_i = actor_i(local_obs_i)                       # agent i's action
    all_actions = torch.cat([others_actions, a_i], dim=-1)
    q = critic_i(full_state, all_actions)            # centralized eval
    return -q.mean()   # ascend Q by descending -Q (deterministic PG)
```

The shape of the data flow is the whole lesson: the actor's input is `local_obs_i` (small, local, available at deploy time), while the critic's input is `full_state` and `all_actions` (global, available only in the simulator). Gradients from the global critic shape the local actor. At deployment you instantiate only `DecentralizedActor` per agent and never construct a `CentralizedCritic` at all.

#### Worked example: a two-agent cooperative backup, end to end

Let me work one full cooperative task by hand — the joint action count, a Dec-POMDP value backup, and the place where centralized training changes the answer — so the abstract pieces from Sections 4, 7, and 8 land as arithmetic.

Two warehouse robots share an aisle. Each has $|\mathcal{A}_i| = 4$ actions (forward, back, pick, wait). The **joint action space** is therefore $|\mathcal{A}| = |\mathcal{A}_1| \times |\mathcal{A}_2| = 4 \times 4 = 16$ — a tractable number for two agents, but note it is the *product*, so the same robots scaled to a $6$-robot aisle would already be $4^6 = 4096$ joint actions, which is the Challenge-3 blowup in miniature. A joint policy would need a head over all $16$ outputs; the factored policies we actually deploy need two heads of $4$ each, $8$ outputs total.

The shared reward pays $+10$ when exactly one robot executes `pick` on the shelf in front of it while the other holds `wait` (so they do not collide reaching for the same shelf), and $-2$ if both `pick` simultaneously (a collision). Consider a state $s$ where the shelf is reachable and run one **Dec-POMDP value backup** for the joint policy under which each robot, acting on its own observation, picks with probability $p$ and waits otherwise (ignore forward/back here for clarity). The immediate expected team reward is

$$
\mathbb{E}[R] = \underbrace{2p(1-p)\,(+10)}_{\text{exactly one picks}} \;+\; \underbrace{p^2\,(-2)}_{\text{both pick, collide}} \;+\; \underbrace{(1-p)^2\,(0)}_{\text{both wait}} = 20p(1-p) - 2p^2.
$$

Maximizing over $p$: $\tfrac{d}{dp}\big[20p - 20p^2 - 2p^2\big] = 20 - 44p = 0$, so the best *symmetric independent* policy is $p^* = 20/44 \approx 0.455$, yielding $\mathbb{E}[R] \approx 20(0.455)(0.545) - 2(0.455)^2 \approx 4.96$. That is the ceiling when each robot must randomize blindly because it cannot see what the other will do — the partial-observability tax, paid in collisions and missed picks. Fold this into the one-step backup $V(s) = \mathbb{E}[R] + \gamma \sum_{s'} P(s' \mid s, \mathbf{a})\, V(s')$ and the discounted value inherits that same $\approx 4.96$ ceiling on the immediate term.

Now turn on **centralized training**. The centralized critic conditions on *both* robots' observations and the joint action, so during training it can value the deterministic configuration "robot 1 picks, robot 2 waits" directly: $\mathbb{E}[R] = +10$, with *zero* collision probability. The factored actors, trained against that critic, learn to break the symmetry — robot 1's policy collapses toward `pick`, robot 2's toward `wait`, conditioned on cues in their observations — recovering the full $+10$ instead of the $4.96$ that blind independent randomization is stuck at. The lift from $4.96$ to $10$ is precisely what centralized training buys: it does not change the action spaces, the transition kernel, or the deployed factored form, but it gives the actors a learning signal that *sees the joint outcome* and so teaches them a coordinated convention that independent learners, each guessing at the other, provably cannot reach. That is CTDE earning its complexity on a single state.

#### Worked example: the variance reduction CTDE buys you

Let me put a number on why the centralized critic helps beyond just restoring stationarity. Suppose three cooperative agents share a reward, and you estimate each agent's policy gradient. With an *independent* (decentralized) critic, agent 1's advantage estimate is contaminated by the variance of agents 2 and 3's actions — their randomness shows up as noise in the reward agent 1 attributes to itself. Empirically, on the StarCraft Multi-Agent Challenge (SMAC), QMIX and MAPPO (centralized critic) reach roughly 85–95% win rates on the standard maps, while independent learners (IQL) plateau dramatically lower on the hard maps — the gap is largely this variance-and-credit problem. A rough back-of-envelope: if each of the two other agents injects independent noise of variance $\sigma^2$ into the reward signal, an independent critic sees gradient-estimate variance scaling like $3\sigma^2$ (its own plus both others'), while a centralized critic that conditions on the others' actions can subtract that contribution, cutting the variance toward $\sigma^2$ — a roughly $3\times$ reduction in this stylized case. Lower-variance gradients mean you can use a larger effective step size and converge in fewer samples, which is exactly what the SMAC benchmark numbers show.

## 9. Communication: when agents talk to coordinate

Decentralized execution says each agent acts on its own observation — but nothing stops agents from *sending each other messages* as part of their action, and learning what to say. Communication is the bridge between fully decentralized and fully centralized execution, and on many partially-observed coordination tasks it is the difference between failure and success.

There are two broad flavors, and the distinction matters for both gradients and deployment:

- **Differentiable / continuous communication.** Agents emit continuous-valued message vectors, and crucially, gradients flow *through* the communication channel during training — the receiver's loss backpropagates into the sender's message-generating network. This makes learning what to communicate a smooth optimization problem. **CommNet** (Sukhbaatar et al., 2016) is the canonical example: each agent broadcasts a continuous hidden vector, and every agent's next hidden state is a function of its own plus the *mean* of all broadcast vectors (a mean-field-style aggregation that also helps Challenge 3 scalability). **DIAL** (Differentiable Inter-Agent Learning, Foerster et al., 2016) sends continuous messages during training (so gradients flow) but discretizes them at execution time (so the deployed protocol is a discrete bit you could actually put on a wire).
- **Discrete / symbolic communication.** Agents emit discrete tokens. This is closer to real protocols (you can name the messages) but harder to train, because you cannot backpropagate through a discrete symbol — you need reinforcement-style or Gumbel-softmax estimators for the message gradients.

When does communication help? When the task is **partially observed and tightly coupled** — when an agent's optimal action genuinely depends on information only another agent has. In the two-drone Dec-POMDP from Section 4, "I have depot A" is a one-bit message that solves the symmetry-breaking problem outright. When the task is fully observed, or when agents barely interact, communication adds parameters and training instability for little gain — skip it.

QMIX and QPLEX deserve a mention here even though they are value-decomposition methods rather than communication methods, because they achieve coordination *without explicit messaging*: the centralized mixing network during training induces consistent decentralized policies, so the agents coordinate through *shared training* rather than through *runtime messages*. That is often the more deployable choice — no communication channel to engineer, secure, or debug in production. Reach for explicit communication only when shared training alone cannot break the information asymmetry.

| Approach | Channel | Gradients flow? | Best when | Representative method |
|---|---|---|---|---|
| No comms, shared training | none | n/a | coordination learnable offline | QMIX, MAPPO |
| Continuous comms | real vectors | yes (end-to-end) | tight coupling, can train jointly | CommNet |
| Train-continuous, deploy-discrete | bits at exec | yes (train only) | need a real discrete protocol | DIAL |
| Discrete comms | tokens | no (needs estimator) | interpretable protocol required | RIAL, Gumbel-softmax variants |

## 10. A milestone tour: how MARL got here

It helps to see the lineage, because each milestone solved a specific failure of the one before. The timeline figure lays out the arc.

![A timeline of multi-agent reinforcement learning milestones from independent Q-learning in 1993 through minimax-Q, Nash-Q, MADDPG, QMIX, OpenAI Five, and AlphaStar in 2019](/imgs/blogs/multi-agent-rl-fundamentals-7.png)

- **IQL, 1993 (Tan).** The naive baseline: independent learners, no coordination machinery. Established the problem and the simplest attack.
- **Minimax-Q, 1994 (Littman).** Brought game theory into RL for the two-player zero-sum case. Each agent computes a minimax value at each state instead of a max, solving a small linear program per update. Provably converges for zero-sum games — the first MARL convergence guarantee.
- **Nash-Q, 2003 (Hu & Wellman).** Generalized to general-sum games by computing a stage-game Nash equilibrium at each state. Theoretically elegant but requires solving a Nash equilibrium at every update and needs strong assumptions to converge — more important as a conceptual milestone than a practical algorithm.
- **MADDPG, 2017 (Lowe et al.).** The CTDE breakthrough for deep, continuous-action MARL. Centralized critics, decentralized actors, handles cooperative, competitive, and mixed. This is the paper that made CTDE the default.
- **QMIX, 2018 (Rashid et al.).** Value decomposition with a monotonic mixing network — the cooperative-team workhorse, and still a standard SMAC baseline. Solved credit assignment cleanly for shared-reward teams.
- **OpenAI Five, 2019.** Scaled MARL to Dota 2, a five-versus-five game with enormous state and action spaces and long horizons. Largely PPO with shared parameters and a hand-engineered "team spirit" reward-shaping knob to interpolate between selfish and cooperative objectives. Proof that MARL scales to genuinely hard games.
- **AlphaStar, 2019 (DeepMind).** StarCraft II at Grandmaster level. Combined imitation from human games, a league of agents trained against each other (population-based self-play to avoid strategy collapse), and MARL. The league idea — keep a diverse population of opponents so no single exploitable strategy dominates — is one of the most important practical lessons in competitive MARL.

The arc is clear: from "ignore the problem" (IQL) to "solve the game-theory exactly but it does not scale" (minimax-Q, Nash-Q) to "use centralized training to scale to deep nets and real games" (MADDPG, QMIX) to "engineer a population so self-play does not collapse" (OpenAI Five, AlphaStar). Each step traded a little theoretical purity for a lot of practical reach.

## 11. The full RLlib multi-agent setup

Theory and toy code are good for understanding; for actually running multi-agent experiments at scale, Ray RLlib is the standard tool, because it handles the policy-mapping, parameter-sharing, and distributed-rollout plumbing that you do not want to write by hand. Let me walk through a complete `MultiAgentEnv` and a training config.

The first thing to understand about RLlib's multi-agent API is the **policy mapping function**: RLlib maintains a dictionary of *policies* (each a neural network you can train) and a function that maps each *agent id* in your environment to a *policy id*. This is the single most important design decision in a multi-agent run:

- **Shared policy** (all agent ids map to one policy id): every agent uses the same network. Correct for homogeneous agents — a fleet of identical robots — and far more sample-efficient because all agents' experience trains one network. This is parameter sharing, the Challenge-3 scalability fix.
- **Independent policies** (each agent id maps to its own policy id): every agent learns its own network. Necessary for heterogeneous agents or competitive settings where you do not want agents sharing weights with their opponents.

Here is a cooperative two-agent environment in the RLlib API. The contract: `reset` and `step` return **dictionaries keyed by agent id**, not single values — that dictionary structure is how RLlib knows which agents acted and which got rewards on each step.

```python
import gymnasium as gym
import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv

class CooperativeNav(MultiAgentEnv):
    """Two agents on a line must occupy the two endpoints (0 and 1).
    Shared cooperative reward. Each agent observes ONLY its own position
    plus the other's position (so it can coordinate) -- a Dec-POMDP-flavored
    setup with enough info to break symmetry."""
    def __init__(self, config=None):
        super().__init__()
        self.agents = self.possible_agents = ["agent_0", "agent_1"]
        # each agent: observation = [own_pos, other_pos] in [0, 1]
        self.observation_space = gym.spaces.Dict({
            a: gym.spaces.Box(0.0, 1.0, shape=(2,), dtype=np.float32)
            for a in self.agents
        })
        self.action_space = gym.spaces.Dict({
            a: gym.spaces.Discrete(3)   # 0 left, 1 stay, 2 right
            for a in self.agents
        })

    def reset(self, *, seed=None, options=None):
        self.pos = {a: 0.5 for a in self.agents}
        return self._obs(), {}

    def _obs(self):
        p0, p1 = self.pos["agent_0"], self.pos["agent_1"]
        return {
            "agent_0": np.array([p0, p1], dtype=np.float32),
            "agent_1": np.array([p1, p0], dtype=np.float32),
        }

    def step(self, action_dict):
        for a, act in action_dict.items():
            move = {0: -0.1, 1: 0.0, 2: +0.1}[int(act)]
            self.pos[a] = float(np.clip(self.pos[a] + move, 0.0, 1.0))
        p0, p1 = self.pos["agent_0"], self.pos["agent_1"]
        # cooperative shared reward: high when agents occupy the two ends
        covered = abs(p0 - p1)                  # want this near 1.0
        r = covered - 0.05                       # minus a small step cost
        rewards = {a: r for a in self.agents}    # SHARED reward
        done = covered > 0.9
        terminateds = {"__all__": done}
        truncateds = {"__all__": False}
        return self._obs(), rewards, terminateds, truncateds, {}
```

The `"__all__"` key in the `terminateds`/`truncateds` dicts is RLlib's signal that the whole episode is over for every agent at once; you can also terminate individual agents by their id. Now the training config, choosing a **shared** policy because the two agents are homogeneous:

```python
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env

register_env("coop_nav", lambda cfg: CooperativeNav(cfg))

config = (
    PPOConfig()
    .environment("coop_nav")
    .multi_agent(
        # ONE shared policy for both homogeneous agents.
        policies={"shared_policy"},
        policy_mapping_fn=lambda agent_id, *a, **kw: "shared_policy",
    )
    .training(
        train_batch_size=4000,
        lr=3e-4,
        gamma=0.99,
        entropy_coeff=0.01,   # keep exploration up to escape bad equilibria
    )
    .env_runners(num_env_runners=4)
)

algo = config.build()
for i in range(50):
    result = algo.train()
    print(f"iter {i:>2}  mean_reward={result['env_runners']['episode_return_mean']:.3f}")
```

To switch to **independent** policies — for a competitive or heterogeneous setup — you change two lines: list two policy ids and map each agent to its own:

```python
config = config.multi_agent(
    policies={"policy_0", "policy_1"},
    policy_mapping_fn=lambda agent_id, *a, **kw: (
        "policy_0" if agent_id == "agent_0" else "policy_1"
    ),
)
```

That is the entire mechanism: the policy-mapping function is the dial between "everyone shares one brain" and "everyone has their own." For a true CTDE algorithm with a centralized critic, RLlib supports custom centralized-critic models (you override the model to take the other agents' observations and actions as extra inputs to the value head while the policy head stays decentralized), but the policy-mapping decision above is the foundation you build that on.

## 12. Applications: where MARL actually earns its complexity

MARL is more expensive and more fragile than single-agent RL, so it should only be reached for when the problem is *irreducibly* multi-agent. Here are the domains where it genuinely is.

**Multi-robot coordination.** Warehouse fleets (Amazon-style fulfillment), drone swarms, multi-arm manipulation. These are cooperative Dec-POMDPs: each robot has local sensing, the team shares a throughput objective, and the joint action space is enormous. CTDE with parameter sharing is the standard recipe. The payoff is coordination that no hand-coded controller matches at scale.

**Multi-player games.** This is MARL's trophy case. AlphaGo/AlphaZero (zero-sum, self-play, MCTS), OpenAI Five (Dota 2, cooperative-within-team and competitive-across-teams), AlphaStar (StarCraft II, league self-play). Games are where MARL's competitive and mixed regimes get their cleanest demonstrations, with unambiguous win/loss rewards.

**Traffic signal control.** Each intersection is an agent; the shared reward is reduced city-wide travel time. This is a cooperative MARL problem where decentralized execution is *mandatory* — you cannot run all of a city's lights from one centralized controller in real time — so CTDE is a perfect fit. Studies report meaningful reductions in average delay versus fixed-timing and over independent controllers, though magnitudes depend heavily on the traffic model.

**Multi-agent financial markets.** Market-making, optimal execution against other algorithmic traders, and agent-based market simulation. This is the mixed/general-sum regime in its purest form: traders are neither teammates nor pure adversaries. It is also where non-stationarity bites hardest, as my intro anecdote showed — when your counterparties learn, your backtest stops predicting your live performance. For the game-theoretic foundations of strategic interaction in markets, see the game-theory series at `/blog/trading/game-theory/nash-equilibrium-and-best-response` (if you are reading these in order, that post pairs naturally with the Nash-equilibrium material here).

## 13. Case studies with numbers

**OpenAI Five (2019).** Five-versus-five Dota 2. The architecture was essentially independent LSTM policies (one per hero) trained with PPO and *shared parameters* across the team, with a scalar "team spirit" hyperparameter $\tau \in [0,1]$ that blended each agent's individual reward with the team's average reward — annealed from selfish ($\tau \approx 0$) early in training toward cooperative ($\tau \to 1$) later. They trained at a then-staggering scale (on the order of hundreds of years of simulated game-play per day across thousands of GPUs/CPUs) and defeated the world-champion human team OG in 2019. The lesson for practitioners: even at frontier scale, the winning recipe was *parameter-shared independent learners with reward shaping*, not an exotic centralized algorithm. CTDE elegance is not always what wins; sometimes scale plus a good reward-shaping knob does.

**AlphaStar (2019).** StarCraft II, reaching Grandmaster (top 0.2% of human players) on the official ladder. The decisive innovation was the **AlphaStar League**: rather than naive self-play (which collapses into a narrow rock-paper-scissors loop where each new agent only beats the previous one), they maintained a diverse population including "main agents," "main exploiters" (trained to find weaknesses in main agents), and "league exploiters" (trained to find weaknesses across the whole league). This population diversity is the practical antidote to the non-stationarity-driven strategy cycling that plagues competitive self-play. If you take one thing from competitive MARL, take this: **a population beats a single opponent.**

**QMIX on SMAC (2018 onward).** On the StarCraft Multi-Agent Challenge — a suite of cooperative micro-management scenarios — QMIX's monotonic value decomposition reaches high win rates (frequently 85%+ on easy and medium maps) and substantially outperforms independent learners (IQL/IDQN) on the harder maps where credit assignment matters most. SMAC is the standard cooperative MARL benchmark precisely because it isolates the credit-assignment and coordination challenges in a reproducible setting; QMIX and MAPPO are the reference baselines you compare against.

**InstructGPT / RLHF as a degenerate single-agent case (2022).** Worth a mention as the contrast: modern LLM alignment via RLHF is *single-agent* RL (one policy, a fixed reward model), even though it feels social. It is not MARL because the reward model does not learn back during the PPO phase — the environment is stationary. If you ever made the reward model adaptive and adversarial (as in some debate or self-play alignment proposals), it would *become* a Markov game, with all the non-stationarity that entails. For the single-agent RLHF mechanics, the training-techniques series covers it at `/blog/machine-learning/training-techniques/rlhf-from-human-feedback-to-policy` — keep the distinction clear: RLHF as usually practiced is single-agent; making the reward learn back is what turns it multi-agent.

## 14. Choosing an algorithm

Given the structure of your problem, the figure below is the decision tree I actually use. Walk it from the root.

![A decision tree for choosing a multi-agent RL algorithm based on whether the task is cooperative, competitive, or mixed and whether a communication channel is available](/imgs/blogs/multi-agent-rl-fundamentals-8.png)

The logic, in words:

1. **Is it cooperative (shared reward)?** If yes, you want value decomposition or a centralized-critic policy-gradient method. **QMIX** for discrete actions with credit-assignment difficulty; **MAPPO** (multi-agent PPO with a centralized value function) for a strong, simple, stable default — MAPPO has become the go-to cooperative baseline because it is robust and easy to tune. **VDN** if you want the simplest possible decomposition (additive). Add explicit communication (**CommNet**) only if partial observability is severe and shared training alone cannot coordinate the agents.
2. **Is it competitive (zero-sum, two-player)?** Use **minimax-Q** for the tabular/small case (it has convergence guarantees), or self-play with **MCTS** (the AlphaZero recipe) for large games. The key practical move is a *population* of opponents, not a single self-play partner, to avoid strategy collapse.
3. **Is it mixed (general-sum)?** **MADDPG** is the default — centralized critics per agent, decentralized actors, handles continuous actions and arbitrary reward structures. **IPPO** (independent PPO) is a surprisingly strong baseline you should always run first. **Nash-Q** only if the problem is small enough to compute stage-game equilibria.
4. **Always run the independent baseline first.** Before any of the above, run **IQL/IPPO**. If it works, ship it — it is the simplest, cheapest, most maintainable option, and the non-stationarity gods were kind to you. Only graduate to centralized-critic methods when the independent baseline visibly oscillates or plateaus.

## 15. When to use MARL — and when not to

This is the section that will save you the most time, because the most common MARL mistake is *using MARL when you did not have to*.

**Do not use MARL — use a single centralized controller — when:**

- **You actually control all the agents and can execute centrally.** If you genuinely have global observation *at deployment* and a single action channel, then your "multi-agent" problem is just a single-agent MDP with a big factored action space. Solve it as one MDP with a single policy that outputs the joint action (factored if needed). You only need decentralized execution when the deployment physically forbids central control — separate robots, separate intersections, separate trading seats.
- **The agents barely interact.** If the coupling between agents is negligible, run $n$ independent single-agent learners and stop overthinking it. IQL is fine and you will save weeks.
- **You can cheaply simulate a single optimal controller.** If a classical planner or optimization (e.g., a MILP for vehicle routing, a market-clearing LP) solves the centralized problem, use it. RL is for when you cannot write down or cheaply solve the optimum.

**Do use MARL when:**

- **Execution must be decentralized** (physically separate agents, no global-info channel at runtime) — the defining condition. This is why CTDE exists.
- **The strategic interaction is the point** — competitive games, market simulation, mechanism design — where you specifically want to study or exploit equilibrium behavior.
- **The joint problem is too large to solve centrally** but factorizes across agents, so decentralized policies with centralized training are tractable where a monolithic controller is not.

**And expect these costs:** MARL training is less stable (non-stationarity), needs more samples (credit assignment noise), is harder to debug (is the oscillation a bug or an equilibrium cycle?), and has weaker theoretical guarantees. Budget accordingly. When I shipped that trading pair, the right fix was *not* a fancier MARL algorithm — it was recognizing that the two bots should have been one centralized controller with a factored action space, because in production a single firm controlled both seats and had global observation. MARL would have been the wrong tool. Match the formalism to the deployment, not to the org chart.

## Key takeaways

- **The defining problem of MARL is non-stationarity:** when other agents learn, the environment each agent faces changes, and the single-agent convergence proofs (which assume a fixed Bellman operator) break — the fixed point moves.
- **The Markov game** $(n, \mathcal{S}, \{\mathcal{A}_i\}, P, \{R_i\}, \gamma)$ is the right formalism; it reduces exactly to an MDP when $n=1$. The relationship between the $\{R_i\}$ defines the regime.
- **Three regimes:** cooperative (shared reward, coordinate), competitive (zero-sum, minimax), mixed (general-sum, Nash). Classify your problem first — it determines the solution concept and the tools.
- **The realistic formalism is the Dec-POMDP** (local observations, shared reward), and it is NEXP-complete to solve optimally — we approximate, we do not solve.
- **Three challenges, one backbone:** non-stationarity, credit assignment, scalability — and CTDE (centralized training, decentralized execution) attacks all three at once.
- **CTDE works because the centralized critic sees the joint action**, which makes the environment stationary at training time; you discard the critic and keep only the local actors at deployment.
- **Always run the independent baseline (IQL/IPPO) first.** It surprisingly often works; only graduate to centralized critics when it oscillates or plateaus.
- **For competitive self-play, use a population, not a single opponent** — this is the AlphaStar League lesson, the practical antidote to strategy collapse.
- **The most common MARL mistake is using MARL at all.** If you can execute centrally and observe globally at deployment, solve it as one MDP. Reserve MARL for irreducibly decentralized execution.

## Further reading

- Littman, M. (1994). *Markov Games as a Framework for Multi-Agent Reinforcement Learning.* The paper that introduced the Markov game to RL and minimax-Q.
- Bernstein, D. et al. (2002). *The Complexity of Decentralized Control of Markov Decision Processes.* Establishes the Dec-POMDP and its NEXP-completeness.
- Lowe, R. et al. (2017). *Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments* (MADDPG). The CTDE breakthrough paper.
- Rashid, T. et al. (2018). *QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning.* The cooperative value-decomposition workhorse.
- Foerster, J. et al. (2018). *Counterfactual Multi-Agent Policy Gradients* (COMA). Counterfactual credit assignment with a centralized critic.
- Sukhbaatar, S. et al. (2016). *Learning Multiagent Communication with Backpropagation* (CommNet). Differentiable continuous communication.
- Yu, C. et al. (2021). *The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games* (MAPPO). Why a simple centralized-critic PPO is a strong baseline.
- Sutton, R. & Barto, A. *Reinforcement Learning: An Introduction* (2nd ed.). The single-agent foundation everything here generalizes from.
- Within this series: the unified map at `/blog/machine-learning/reinforcement-learning/reinforcement-learning-a-unified-map` and the eventual capstone, `the-reinforcement-learning-playbook`.
