---
title: "MADDPG and CTDE: Centralised Training for Decentralised Agents"
date: "2026-06-22"
publishDate: "2026-06-22"
description: "Understand why multi-agent RL breaks single-agent algorithms, derive the centralised-training/decentralised-execution fix behind MADDPG, QMIX and MAPPO, and build a working multi-agent critic in PyTorch."
tags:
  [
    "reinforcement-learning",
    "deep-learning",
    "multi-agent",
    "actor-critic",
    "policy-gradient",
    "machine-learning",
    "pytorch",
    "maddpg",
    "qmix",
    "mappo",
  ]
category: "machine-learning"
subcategory: "Reinforcement Learning"
author: "Hiep Tran"
featured: true
readTime: 55
image: "/imgs/blogs/maddpg-centralised-training-decentralised-execution-1.png"
---

The first time I tried to train two RL agents in the same environment, I did the obvious thing: I took the DDPG code that had just solved a single-arm reaching task, copied it twice, gave each agent its own replay buffer and its own networks, and pressed go. Two agents, two independent learners, one shared world. It should have worked. It absolutely did not. The return curves climbed, then collapsed, then climbed somewhere else, then collapsed again — a slow oscillation that never settled. I burned three days re-tuning learning rates and target-update speeds before I understood that the problem was not my hyperparameters. The problem was structural, and no amount of tuning was going to fix it.

The structural problem has a name: **non-stationarity**. When you put multiple learning agents in the same environment, each agent's "environment" includes the other agents — and those other agents are *changing* as they learn. From agent 1's point of view, the same action in the same state produces wildly different outcomes from one episode to the next, because agent 2 is now playing differently. The fixed-environment assumption that every single-agent algorithm silently relies on — the Markov property, the stationary transition kernel, the convergence guarantees of Q-learning — all of it quietly breaks. The agent is trying to hit a target that moves every time it moves.

The fix, discovered and popularised by Lowe et al. in 2017, is one of those ideas that feels obvious in hindsight and is genuinely deep once you sit with it. It is called **centralised training with decentralised execution**, or CTDE, and the figure below is the whole idea in one picture: while you are *training*, you let each agent's critic peek at everything — every agent's observation, every agent's action — because at training time you control the whole simulation and you can. But the thing you actually deploy, the actor that picks actions at run time, only ever looks at that one agent's local observation. You train with global information and execute with local information. By the end of this post you will understand exactly why that asymmetry dissolves the non-stationarity problem, you will have derived the MADDPG gradient, you will have a runnable PyTorch implementation of a centralised critic with decentralised actors, and you will know when to reach for MADDPG versus QMIX versus MAPPO instead.

![Diagram showing each agent observation feeding a local actor while all actions and the joint observation feed a centralised critic that produces the actor gradient](/imgs/blogs/maddpg-centralised-training-decentralised-execution-1.png)

This post sits in the multi-agent track of the series. It builds directly on the deterministic policy gradient machinery from [DDPG and TD3](/blog/machine-learning/reinforcement-learning/deterministic-policy-gradient-ddpg-td3), so if the terms "actor", "critic", "target network" and "deterministic policy gradient" are not yet second nature, skim that post first. We will also lean on the [credit assignment problem](/blog/machine-learning/reinforcement-learning/the-credit-assignment-problem), which becomes dramatically harder when many agents share one reward signal.

## 1. Why single-agent RL breaks with more than one learner

Let us be precise about what goes wrong, because the precision is what makes the fix make sense. A single-agent Markov decision process (MDP) is the tuple $(\mathcal{S}, \mathcal{A}, P, R, \gamma)$: states, actions, a transition kernel $P(s' \mid s, a)$, a reward function $R(s, a)$, and a discount $\gamma$. The entire theory of RL — value iteration, Q-learning convergence, the policy gradient theorem — rests on $P$ being *fixed*. The world responds to your action the same way today as it did yesterday. That stationarity is what lets a value function converge: you are estimating a fixed quantity, $Q^*(s, a)$, and your estimate can settle on it.

Now add a second agent. The natural generalisation is a **Markov game** (also called a stochastic game), the tuple $(\mathcal{S}, \{\mathcal{A}_i\}, P, \{R_i\}, \gamma)$ for $n$ agents indexed by $i$. The transition kernel now depends on the *joint* action: $P(s' \mid s, a_1, \ldots, a_n)$. Each agent has its own reward $R_i(s, a_1, \ldots, a_n)$. The joint policy is $\boldsymbol{\pi} = (\pi_1, \ldots, \pi_n)$.

Here is the crux. From agent $i$'s perspective, if it pretends the other agents are just part of the environment, then its effective transition function is

$$
P_i(s' \mid s, a_i) = \sum_{a_{-i}} P(s' \mid s, a_i, a_{-i}) \prod_{j \ne i} \pi_j(a_j \mid s),
$$

where $a_{-i}$ denotes "the actions of everyone except $i$". Look at what this depends on: $\prod_{j \ne i} \pi_j$, the policies of the other agents. As those agents *learn*, their $\pi_j$ change, so $P_i$ changes. Agent $i$'s effective environment is non-stationary. The Markov property — that the future depends only on the current state and action — is violated, because the future now also depends on *which round of training* we are in, since that determines what the other policies look like.

This is not a minor inconvenience. It poisons the core machinery. The Q-learning target $r + \gamma \max_{a'} Q(s', a')$ assumes the transition that produced $s'$ is the same transition you will face when you act in $s'$ again. With non-stationarity, that target chases a moving distribution. Experience replay makes it worse: a transition stored in the buffer was generated under *old* opponent policies, so replaying it teaches the agent about a world that no longer exists. The replay buffer, the one piece of machinery that makes off-policy deep RL sample-efficient, becomes a liability.

Independent Q-learning (IQL), where you just run $n$ separate Q-learners and ignore the interaction, is the baseline that captures all of these pathologies. It sometimes works — in loosely coupled tasks where agents barely affect each other, the non-stationarity is mild. But the more the agents' fates are intertwined, the worse IQL gets. The before-and-after figure below contrasts the IQL failure mode with the CTDE fix that the rest of this post develops.

![Two-column comparison contrasting independent Q-learning seeing only local observations and a non-stationary unstable signal against MADDPG using the full state for stationary stable training](/imgs/blogs/maddpg-centralised-training-decentralised-execution-2.png)

#### Worked example: how non-stationarity wrecks a value estimate

Picture two agents in a tiny grid. Agent 1 gets reward $+1$ if it reaches a door, but the door is *locked* unless agent 2 is standing on a switch. Early in training, agent 2 almost never stands on the switch (its policy is near-random), so from agent 1's view, walking to the door yields reward $\approx 0$ almost every time. Agent 1's critic learns $Q_1(\text{near door}, \text{step toward door}) \approx 0$. Reasonable — given what it has seen.

Now agent 2 learns to camp the switch. Suddenly, stepping toward the door yields $+1$ most of the time. The true value of agent 1's "step toward door" action has jumped from $\approx 0$ to $\approx +1$, but agent 1's critic still holds the stale estimate, and worse, its replay buffer is full of transitions from the era when the door was locked. The critic now has to *unlearn* a confidently-held wrong value while fighting a buffer that keeps replaying the old world. This is exactly the oscillation I watched for three days. The fix is to give agent 1's critic the one piece of information that makes the value well-defined: *whether agent 2 is on the switch* — i.e. agent 2's observation and action.

## 2. The CTDE insight, stated cleanly

The non-stationarity comes from one specific thing: agent $i$ does not know what the other agents are doing, so the others' changing behaviour shows up as a changing environment. The CTDE insight is to *remove the unknown*. If the critic is told the other agents' observations and actions, then the environment, from the critic's point of view, is stationary again.

Make that precise. Define the **joint observation** $x = (o_1, \ldots, o_n)$ — everything all agents see, concatenated. (In a fully observable game $x$ is just the global state $s$; in a partially observable game it is the stack of local observations, possibly with extra global features you only have access to in simulation.) Now consider a critic for agent $i$ that conditions on the joint observation *and* every agent's action: $Q_i^{\boldsymbol{\mu}}(x, a_1, \ldots, a_n)$. The key claim:

> If the critic conditions on all agents' actions, then for *any fixed set of opponent policies*, the value it is estimating is well-defined and stationary — because there is no longer any hidden variable. The transition $P(s' \mid s, a_1, \ldots, a_n)$ is fixed; we have simply supplied all of its inputs.

The non-stationarity was never in the world. It was in agent $i$'s *ignorance* of the other agents' actions. Feed the critic those actions and the ignorance — and with it the non-stationarity — disappears. Formally, even when the opponent policies change between updates, the *function* $Q_i^{\boldsymbol{\mu}}(x, a_1, \ldots, a_n)$ being learned is a fixed map from "full state and all actions" to "expected return for agent $i$", and that map does not move just because the policies that *generate* the actions move. The targets are stable.

That handles training. But a critic that needs everyone's observations and actions is useless at deployment — in a real multi-robot fleet, robot 1 cannot read robot 5's camera in real time, and in a competitive game you obviously cannot see your opponent's private observation. So the second half of CTDE: **the policy you deploy uses only local information.** Agent $i$'s actor is $\mu_i(o_i)$, a function of its own observation alone. The expensive, all-seeing critic exists *only during training* and is thrown away before deployment. The figure below lays out the two phases as a stack.

![Layered diagram of CTDE phases showing training with the full state and joint actions feeding centralised critics, then a freeze step, then decentralised execution from local observations only](/imgs/blogs/maddpg-centralised-training-decentralised-execution-3.png)

This asymmetry — train with God's-eye information, deploy with local information — is the entire trick. It is so general that nearly every modern multi-agent RL algorithm is a variation on it. MADDPG puts the centralised information in a per-agent critic. QMIX puts it in a value-mixing network. MAPPO puts it in a shared value baseline. The CTDE skeleton is identical; the algorithms differ only in *where* the centralised information lives and *how* it is factored.

There is one more thing worth being explicit about, because it is the conceptual hinge of the whole approach: CTDE does not change *what the agents can do at run time*. The set of deployable policies is exactly the set of decentralised policies $\{\boldsymbol{\mu} : a_i = \mu_i(o_i)\}$ — the centralised critic is not a runtime crutch, it is a *learning signal*. You are not making the agents smarter at execution; you are making their *gradients* better-informed during training. The deployed system is no more capable than an independently-trained one in terms of its information access. What changes is that the policy you arrive at is *better*, because the path to it was guided by a value function that did not suffer from non-stationarity. This distinction matters when someone objects "but real agents can't see each other's observations" — correct, and neither can the CTDE-trained ones, once deployed.

## 3. The Markov-game policy gradient, formally

Before the algorithm, it is worth grounding MADDPG in the multi-agent policy gradient theorem, because the derivation is where the per-agent-critic structure becomes inevitable rather than arbitrary. In a single-agent MDP, the policy gradient theorem tells us that for a stochastic policy $\pi_\theta$, the gradient of the expected return $J(\theta)$ is

$$
\nabla_\theta J(\theta) = \mathbb{E}_{s \sim d^\pi, a \sim \pi}\bigl[\nabla_\theta \log \pi_\theta(a \mid s)\, Q^\pi(s, a)\bigr],
$$

where $d^\pi$ is the discounted state-visitation distribution. The clean derivation lives in the [policy gradient theorem post](/blog/machine-learning/reinforcement-learning/the-policy-gradient-theorem); here we only need its shape.

Now lift this to a Markov game. Agent $i$ wants to maximise its own expected return $J_i(\theta_i) = \mathbb{E}[\sum_t \gamma^t r_i^t]$, where the trajectory distribution depends on *all* the agents' policies $\boldsymbol{\pi} = (\pi_1, \ldots, \pi_n)$. Differentiating only $\theta_i$ (agent $i$ controls only its own policy), the multi-agent policy gradient is

$$
\nabla_{\theta_i} J_i = \mathbb{E}_{x \sim d^{\boldsymbol{\pi}},\, a \sim \boldsymbol{\pi}}\bigl[\nabla_{\theta_i} \log \pi_i(a_i \mid o_i)\, Q_i^{\boldsymbol{\pi}}(x, a_1, \ldots, a_n)\bigr].
$$

Stare at the action-value term. It is not $Q_i(o_i, a_i)$ — a function of agent $i$'s observation and action alone — it is $Q_i^{\boldsymbol{\pi}}(x, a_1, \ldots, a_n)$, a function of the *joint* observation and *all* actions. This is not a modelling choice; it falls directly out of the math. Agent $i$'s expected return genuinely depends on what everyone does, so the value function that appears in its gradient genuinely depends on everyone's actions. The single-agent form $Q_i(o_i, a_i)$ is an *approximation* that throws away exactly the information whose absence causes non-stationarity. MADDPG's centralised critic is what you get when you refuse to throw that information away.

The deterministic case (MADDPG proper) replaces the score-function form with the deterministic policy gradient, but the same logic holds: the critic must condition on the joint quantities for the gradient to be a faithful estimate of $\nabla_{\theta_i} J_i$. Everything else — the replay buffer, the target networks, the soft updates — is borrowed wholesale from DDPG. The *only* idea unique to MADDPG is "make the critic see everyone."

A second formal point worth internalising: at any fixed joint policy $\boldsymbol{\pi}$, the game induces a well-defined value $Q_i^{\boldsymbol{\pi}}$, and the Bellman equation for it,

$$
Q_i^{\boldsymbol{\pi}}(x, \boldsymbol{a}) = \mathbb{E}\bigl[r_i + \gamma\, Q_i^{\boldsymbol{\pi}}(x', \boldsymbol{a}') \mid \boldsymbol{a}' = \boldsymbol{\mu}(x')\bigr],
$$

is a contraction in exactly the same way the single-agent Bellman operator is — *provided* the joint policy generating $\boldsymbol{a}'$ is held fixed. This is the precise sense in which the centralised critic faces a stationary problem: for fixed opponent policies it is fitting the fixed point of a contraction, which is the regression a decentralised critic can never set up because it lacks the opponent actions to even write down $\boldsymbol{a}'$.

## 4. The full MADDPG mathematical derivation

The shapes above are correct, but it is worth walking the derivation end to end, because every design choice in the algorithm — why the critic conditions on *all* actions, why the target uses *target actors*, why the actor differentiates through only one slot — is forced by the math, not chosen for convenience. We will derive the centralised critic objective, then the actor gradient, then show precisely how the conditioning on all agents' actions dissolves non-stationarity.

**The centralised critic objective.** Fix the joint policy $\boldsymbol{\mu} = (\mu_1, \ldots, \mu_n)$ for the moment — treat every agent's policy as frozen. Under that fixed joint policy, agent $i$'s action-value is the standard discounted return conditioned on the joint observation $x$ and the full action vector $\boldsymbol{a} = (a_1, \ldots, a_n)$:

$$
Q_i^{\boldsymbol{\mu}}(x, a_1, \ldots, a_n) = \mathbb{E}\Bigl[\sum_{t=0}^{\infty} \gamma^t\, r_i^{\,t} \;\Big|\; x^0 = x,\; \boldsymbol{a}^0 = \boldsymbol{a},\; \boldsymbol{a}^{t \ge 1} = \boldsymbol{\mu}(x^t)\Bigr].
$$

This is a perfectly ordinary action-value — the only twist is that the conditioning event fixes *everyone's* first action, not just agent $i$'s. Because the dynamics $P(x' \mid x, a_1, \ldots, a_n)$ are a fixed kernel once all actions are supplied, this expectation is well-defined and obeys the Bellman recursion

$$
Q_i^{\boldsymbol{\mu}}(x, \boldsymbol{a}) = \mathbb{E}_{x' \sim P(\cdot \mid x, \boldsymbol{a})}\Bigl[r_i(x, \boldsymbol{a}) + \gamma\, Q_i^{\boldsymbol{\mu}}\bigl(x', \mu_1(o_1'), \ldots, \mu_n(o_n')\bigr)\Bigr].
$$

We approximate $Q_i^{\boldsymbol{\mu}}$ with a neural network $Q_i(x, \boldsymbol{a}; \phi_i)$ and fit it by regressing onto a one-step bootstrap of this recursion. Drawing joint transitions $(x, \boldsymbol{a}, r_i, x')$ from the replay buffer $\mathcal{D}$, the critic objective is the mean-squared Bellman error

$$
\mathcal{L}(\phi_i) = \mathbb{E}_{(x, \boldsymbol{a}, r_i, x') \sim \mathcal{D}}\Bigl[\bigl(Q_i(x, a_1, \ldots, a_n; \phi_i) - y_i\bigr)^2\Bigr],
$$

$$
y_i = r_i + \gamma\, Q_i^{\boldsymbol{\mu}'}\bigl(x', a_1', \ldots, a_n'\bigr)\Big|_{a_j' = \mu_j'(o_j')},
$$

where the target value uses the *target critic* $Q_i^{\boldsymbol{\mu}'}$ (parameters $\phi_i'$) evaluated at next-state actions produced by every agent's *target actor* $\mu_j'(o_j')$. This is the equation the team lead asked for, and now its every piece is justified: the target conditions on *all* the $a_j' = \mu_j'(o_j')$ because $Q_i$ is a function of all actions, and using target networks for both the critic and the actors is the same stability device as single-agent DDPG, here lifted to the joint quantities. Treating $y_i$ as a constant (the standard semi-gradient move — we do not differentiate through the bootstrap target), the gradient that the optimiser actually follows is

$$
\nabla_{\phi_i} \mathcal{L}(\phi_i) = \mathbb{E}_{\mathcal{D}}\Bigl[-2\bigl(y_i - Q_i(x, \boldsymbol{a}; \phi_i)\bigr)\, \nabla_{\phi_i} Q_i(x, \boldsymbol{a}; \phi_i)\Bigr].
$$

**The actor gradient, derived from scratch.** Agent $i$ wants to maximise its own expected return under the joint policy. Writing the objective as the value of agent $i$'s critic evaluated at its own actor's output, with the other agents' actions drawn from the buffer's behaviour distribution,

$$
J_i(\theta_i) = \mathbb{E}_{x \sim \mathcal{D}}\Bigl[Q_i\bigl(x, a_1, \ldots, a_{i-1}, \mu_i(o_i; \theta_i), a_{i+1}, \ldots, a_n\bigr)\Bigr].
$$

Only the $i$-th argument depends on $\theta_i$. The other slots $a_{j \ne i}$ are *data* — they are the actions that were actually taken and stored, and $\theta_i$ has no influence over them. So differentiating is a single application of the chain rule through the one slot that moves:

$$
\nabla_{\theta_i} J_i = \mathbb{E}_{x \sim \mathcal{D}}\Bigl[\underbrace{\nabla_{\theta_i}\, \mu_i(o_i; \theta_i)}_{\text{Jacobian of the actor}} \cdot \underbrace{\nabla_{a_i} Q_i(x, a_1, \ldots, a_n)\big|_{a_i = \mu_i(o_i)}}_{\text{gradient of value w.r.t. agent } i\text{'s action}}\Bigr].
$$

This is the deterministic policy gradient, but read what makes it *multi-agent*: the inner term $\nabla_{a_i} Q_i$ is evaluated at the *full* action vector — agent $i$'s own current action in slot $i$, and everyone else's buffered actions in the other slots. The question the gradient answers is not "which way should I move my action to raise value?" but "which way should I move my action to raise value *given exactly what everyone else did*?" That conditioning is the difference between a coherent gradient and one corrupted by the other agents' churn. If we instead used a decentralised critic $Q_i(o_i, a_i)$, the inner gradient would be averaging over an *unknown, shifting* distribution of teammate actions — and that average drifts every time a teammate updates, which is exactly the non-stationary signal that wrecked my original two-agent run.

**Why conditioning on all actions kills non-stationarity — the precise statement.** Here is the heart of it, stated as carefully as it deserves. Define the decentralised value an independent learner would try to estimate by marginalising the others out:

$$
\bar{Q}_i(o_i, a_i) = \mathbb{E}_{a_{-i} \sim \boldsymbol{\pi}_{-i}}\bigl[Q_i^{\boldsymbol{\mu}}(x, a_i, a_{-i})\bigr] = \sum_{a_{-i}} \Bigl(\prod_{j \ne i} \pi_j(a_j \mid o_j)\Bigr) Q_i^{\boldsymbol{\mu}}(x, a_i, a_{-i}).
$$

This $\bar{Q}_i$ is the object IQL is implicitly fitting, and notice that it *carries the opponent policies $\pi_j$ inside it*. When any $\pi_j$ changes — which it does on every update — $\bar{Q}_i$ changes even though nothing about the world's dynamics or agent $i$'s own behaviour changed. The regression target moves because the *function being regressed* is policy-dependent. That is non-stationarity made explicit: the target is a moving average over a moving distribution.

The centralised critic $Q_i^{\boldsymbol{\mu}}(x, a_1, \ldots, a_n)$ has no such dependency baked into its arguments. It is a fixed map from `(full observation, all actions)` to expected return; the opponent policies $\pi_j$ are *not arguments to it* and do not appear in its definition once the actions are supplied. The policies' only role is to *generate the data distribution* over which we sample — and a regression's target function does not move just because the input distribution moves. We are simply fitting more or fewer examples from different regions of the same fixed surface. That is the formal sense in which supplying all the actions converts a moving-target problem into an ordinary supervised-regression problem with a stationary target. The non-stationarity was a consequence of marginalising; refuse to marginalise and it is gone.

## 5. MADDPG: the algorithm

MADDPG — Multi-Agent Deep Deterministic Policy Gradient — is the most direct realisation of CTDE: take DDPG and make each agent's critic centralised. Let me lay out the pieces, assuming the DDPG background from the [DDPG/TD3 post](/blog/machine-learning/reinforcement-learning/deterministic-policy-gradient-ddpg-td3).

Each agent $i$ has:

- A **decentralised deterministic actor** $\mu_i(o_i; \theta_i)$ that maps its local observation to an action. This is what runs at execution time.
- A **centralised critic** $Q_i(x, a_1, \ldots, a_n; \phi_i)$ that takes the joint observation $x$ and *all* agents' actions, and outputs a scalar estimate of agent $i$'s expected return. This exists only during training.
- Target networks $\mu_i'$ and $Q_i'$ with slowly-tracking parameters, exactly as in DDPG.

There is a **shared replay buffer** storing joint transition tuples $(x, a_1, \ldots, a_n, r_1, \ldots, r_n, x')$. Every agent draws from the same buffer because every agent's critic update needs the joint quantities.

The critic update is a multi-agent Bellman regression. For agent $i$, the target is

$$
y_i = r_i + \gamma \, Q_i'\bigl(x', \mu_1'(o_1'), \ldots, \mu_n'(o_n')\bigr),
$$

and the loss is the mean-squared TD error

$$
\mathcal{L}(\phi_i) = \mathbb{E}_{(x, a, r, x') \sim \mathcal{D}} \Bigl[\bigl(Q_i(x, a_1, \ldots, a_n; \phi_i) - y_i\bigr)^2\Bigr].
$$

The next-state actions in the target come from the *target actors* of every agent, $\mu_j'(o_j')$ — this is the deterministic analogue of the DDPG target, lifted to the joint action. Because the critic sees all the actions, this regression is stationary: it is fitting a fixed function.

## 6. Deriving the MADDPG actor gradient

The actor update is where the centralised critic earns its keep. We want to improve $\mu_i$ so that the actions it produces score higher under agent $i$'s critic. This is the deterministic policy gradient, lifted to multi-agent.

Recall the single-agent deterministic policy gradient theorem: for a deterministic policy $\mu(s; \theta)$ and critic $Q(s, a)$, the gradient of the objective $J(\theta) = \mathbb{E}_s[Q(s, \mu(s))]$ is

$$
\nabla_\theta J = \mathbb{E}_s\bigl[\nabla_\theta \mu(s) \, \nabla_a Q(s, a)\big|_{a = \mu(s)}\bigr],
$$

a chain rule: nudge the action via $\nabla_a Q$ (which way to move the action to raise value), then pull that nudge back through the actor via $\nabla_\theta \mu$.

For MADDPG, agent $i$'s objective is its expected return, which we estimate with its centralised critic. The actor $\mu_i$ only controls $a_i$, so it differentiates only through the $a_i$ slot of the critic. The other agents' actions $a_j$ for $j \ne i$ are present in the critic's input — they make the value well-defined — but they are *constants* as far as $\theta_i$ is concerned. Applying the chain rule:

$$
\nabla_{\theta_i} J_i = \mathbb{E}_{x, a \sim \mathcal{D}}\Bigl[\nabla_{\theta_i} \mu_i(o_i) \, \nabla_{a_i} Q_i(x, a_1, \ldots, a_n)\big|_{a_i = \mu_i(o_i)}\Bigr].
$$

Read it slowly. Sample a joint observation $x$ from the buffer. Replace agent $i$'s action with its *current* actor output $a_i = \mu_i(o_i)$ — we differentiate through this. Keep the other agents' actions $a_j$ at whatever was sampled from the buffer. Evaluate the critic, take its gradient with respect to the $a_i$ slot ($\nabla_{a_i} Q_i$: "which way should agent $i$'s action move to raise its value, *given what everyone else did*"), and pull that back through the actor with $\nabla_{\theta_i}\mu_i$. That conditioning on what everyone else did is the whole point — it is why the gradient signal is coherent instead of confused by the other agents' churn.

The figure below shows the per-step update as a pipeline: sample, build targets from all target actors, update every critic, update every actor through its centralised critic, then soft-update the targets.

![Pipeline of one MADDPG update sampling a joint tuple, computing targets from all target actors, updating all critics, updating all actors with centralised critic gradients, and soft updating the targets](/imgs/blogs/maddpg-centralised-training-decentralised-execution-4.png)

#### Worked example: a centralised critic forward pass with joint observations

Concrete numbers make this stick. Three agents, each with a 4-dimensional local observation and a 2-dimensional continuous action (think $x$/$y$ thrust). The joint observation $x$ is $3 \times 4 = 12$ dimensional. The joint action is $3 \times 2 = 6$ dimensional. Agent 1's centralised critic takes a $12 + 6 = 18$-dimensional input and outputs a single scalar.

Say at some training step the sampled joint observation is fixed, agent 2 took action $(0.3, -0.1)$ and agent 3 took $(−0.2, 0.5)$ — these stay fixed. Agent 1's current actor, fed its own observation, outputs $a_1 = (0.4, 0.2)$. We feed $(x, 0.4, 0.2, 0.3, -0.1, -0.2, 0.5)$ into the critic and get, say, $Q_1 = 1.83$. Now we ask: how should $a_1$ move to raise $Q_1$? We compute $\nabla_{a_1} Q_1 = (0.6, -0.9)$ — meaning "push the first action component up, the second down". Backpropagating $(0.6, -0.9)$ through agent 1's actor gives the parameter gradient. The numbers $(0.3, -0.1)$ and $(-0.2, 0.5)$ from the other agents are present in the forward pass — they are why $Q_1 = 1.83$ rather than some other number — but they receive no gradient. That is decentralised actors learning through a centralised critic.

## 7. MADDPG implementation in PyTorch

Here is a self-contained, runnable MADDPG. I have kept it deliberately explicit — one class for the per-agent networks, one orchestrator that holds all agents and runs the joint update. It targets a continuous-action multi-agent environment (the PettingZoo MPE `simple_spread` family is the canonical fit).

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
import random


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=64, final_tanh=False):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )
        self.final_tanh = final_tanh

    def forward(self, x):
        out = self.net(x)
        return torch.tanh(out) if self.final_tanh else out


class Agent:
    """One agent: a local actor and a centralised critic, each with a target copy."""
    def __init__(self, obs_dim, act_dim, joint_obs_dim, joint_act_dim, lr=1e-3):
        # Actor sees only its own observation (decentralised execution).
        self.actor = MLP(obs_dim, act_dim, final_tanh=True)
        self.actor_target = MLP(obs_dim, act_dim, final_tanh=True)
        # Critic sees the joint observation AND every agent's action (centralised).
        self.critic = MLP(joint_obs_dim + joint_act_dim, 1)
        self.critic_target = MLP(joint_obs_dim + joint_act_dim, 1)
        self._hard_update(self.actor_target, self.actor)
        self._hard_update(self.critic_target, self.critic)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

    @staticmethod
    def _hard_update(target, source):
        target.load_state_dict(source.state_dict())

    @staticmethod
    def soft_update(target, source, tau=0.01):
        for t, s in zip(target.parameters(), source.parameters()):
            t.data.copy_(tau * s.data + (1 - tau) * t.data)

    def act(self, obs, noise=0.1):
        with torch.no_grad():
            a = self.actor(torch.as_tensor(obs, dtype=torch.float32))
        a = a + noise * torch.randn_like(a)      # exploration noise
        return torch.clamp(a, -1.0, 1.0).numpy()
```

The agent class is intentionally plain. The only thing that distinguishes it from single-agent DDPG is the critic's input dimension: `joint_obs_dim + joint_act_dim`, not `obs_dim + act_dim`. That one change is what makes it centralised. Now the orchestrator and the joint update:

```python
class MADDPG:
    def __init__(self, n_agents, obs_dims, act_dims, gamma=0.95, tau=0.01):
        self.n = n_agents
        self.gamma = gamma
        self.tau = tau
        self.act_dims = act_dims
        joint_obs_dim = sum(obs_dims)
        joint_act_dim = sum(act_dims)
        self.agents = [
            Agent(obs_dims[i], act_dims[i], joint_obs_dim, joint_act_dim)
            for i in range(n_agents)
        ]
        self.buffer = deque(maxlen=int(1e6))

    def store(self, obs, acts, rews, next_obs, done):
        # Store the JOINT transition: every agent's piece together.
        self.buffer.append((obs, acts, rews, next_obs, float(done)))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        obs, acts, rews, next_obs, done = zip(*batch)
        # Each is a list over agents; stack per-agent into tensors.
        to_t = lambda x: [torch.as_tensor(np.array(a), dtype=torch.float32) for a in zip(*x)]
        return to_t(obs), to_t(acts), to_t(rews), to_t(next_obs), \
            torch.as_tensor(done, dtype=torch.float32)

    def update(self, batch_size=1024):
        if len(self.buffer) < batch_size:
            return
        obs, acts, rews, next_obs, done = self.sample(batch_size)
        joint_obs = torch.cat(obs, dim=1)          # (B, joint_obs_dim)
        joint_acts = torch.cat(acts, dim=1)        # (B, joint_act_dim)

        # Target actions from EVERY agent's target actor (the joint Bellman target).
        with torch.no_grad():
            next_acts = [self.agents[j].actor_target(next_obs[j]) for j in range(self.n)]
            joint_next_obs = torch.cat(next_obs, dim=1)
            joint_next_acts = torch.cat(next_acts, dim=1)

        for i, agent in enumerate(self.agents):
            # --- Critic update: stationary because all actions are supplied ---
            with torch.no_grad():
                q_next = agent.critic_target(
                    torch.cat([joint_next_obs, joint_next_acts], dim=1)
                ).squeeze(1)
                y = rews[i] + self.gamma * (1 - done) * q_next
            q = agent.critic(torch.cat([joint_obs, joint_acts], dim=1)).squeeze(1)
            critic_loss = F.mse_loss(q, y)
            agent.critic_opt.zero_grad()
            critic_loss.backward()
            agent.critic_opt.step()

            # --- Actor update: differentiate ONLY through agent i's action slot ---
            cur_acts = [acts[j] for j in range(self.n)]
            cur_acts[i] = agent.actor(obs[i])      # replace slot i with live actor output
            joint_cur_acts = torch.cat(cur_acts, dim=1)
            actor_loss = -agent.critic(
                torch.cat([joint_obs, joint_cur_acts], dim=1)
            ).mean()                               # maximise Q -> minimise -Q
            agent.actor_opt.zero_grad()
            actor_loss.backward()
            agent.actor_opt.step()

        # Soft-update all targets after every agent has been updated.
        for agent in self.agents:
            agent.soft_update(agent.actor_target, agent.actor, self.tau)
            agent.soft_update(agent.critic_target, agent.critic, self.tau)
```

Two lines deserve a second look. First, in the actor update, `cur_acts[i] = agent.actor(obs[i])` replaces only agent $i$'s action with the live, differentiable actor output; the other agents' actions stay as the buffered constants. That is the derivation from section 4, expressed in code: the gradient flows through slot $i$ only. Second, the critic target uses `next_acts` from *every* agent's target actor — the joint Bellman backup. If you swapped that for a critic that only saw $a_i$, you would be back to a non-stationary single-agent learner. The centralisation is doing real work in exactly these two places.

The training loop that drives it is ordinary:

```python
def train(env, maddpg, episodes=25000, max_steps=25):
    for ep in range(episodes):
        obs, _ = env.reset()
        obs = [obs[a] for a in env.agents]         # list per agent
        for t in range(max_steps):
            acts = [maddpg.agents[i].act(obs[i]) for i in range(maddpg.n)]
            act_dict = {a: acts[i] for i, a in enumerate(env.agents)}
            next_raw, rew_raw, term, trunc, _ = env.step(act_dict)
            next_obs = [next_raw[a] for a in env.agents]
            rews = [rew_raw[a] for a in env.agents]
            done = any(term.values()) or any(trunc.values())
            maddpg.store(obs, acts, rews, next_obs, done)
            obs = next_obs
            maddpg.update(batch_size=1024)
            if done:
                break
```

## 8. Cooperative MADDPG, shared reward, and credit assignment

The setting where MADDPG is most studied is **cooperative**: all agents maximise the *same* team reward, $R_1 = R_2 = \cdots = R_n = R$. Cooperative navigation, where a swarm must cover a set of landmarks, is the textbook example. Here you can simplify — since the reward is shared, all the per-agent critics are estimating the same expected team return, so you can collapse them into a single shared critic $Q(x, a_1, \ldots, a_n)$ used by every agent's actor update. Fewer parameters, faster training.

But a shared reward reintroduces a problem we have met before: [credit assignment](/blog/machine-learning/reinforcement-learning/the-credit-assignment-problem). If the team gets $+10$ because the swarm covered the landmarks, *which agent's action deserves the credit*? A lazy agent that did nothing useful gets the same $+10$ as the agent that made the key move. The shared critic does not, by itself, disentangle individual contributions, so the actor gradient for the lazy agent can be just as positive — it free-rides on its teammates' competence.

The principled fix is a **counterfactual baseline**, the central idea behind COMA (Counterfactual Multi-Agent policy gradients, Foerster et al. 2018). Instead of using the raw critic value in the gradient, subtract a baseline that marginalises out agent $i$'s own action:

$$
A_i(x, a) = Q(x, a_1, \ldots, a_n) - \sum_{a_i'} \pi_i(a_i' \mid o_i)\, Q(x, a_1, \ldots, a_i', \ldots, a_n).
$$

Read the second term as "the expected team value if agent $i$ had acted *according to its current policy distribution* while everyone else held fixed". The difference $A_i$ answers a counterfactual: "how much better than my average did *this specific action* make the team?" An agent that did nothing useful gets $A_i \approx 0$ — its action did not move the team value above what its policy would have done anyway — so its actor gradient is near zero. Credit is assigned to the action that actually changed the outcome. The centralised critic is what makes this baseline computable, because evaluating "what if agent $i$ had done $a_i'$ instead" requires a value function that takes all agents' actions as input. You cannot compute a counterfactual baseline with a decentralised critic; this is a feature CTDE *enables*.

#### Worked example: counterfactual credit in a two-agent push task

Two agents must push a box to a goal; the team gets $+1$ when the box arrives. Agent A pushes hard; agent B happens to be wandering nearby but contributes nothing this episode. Reward $+1$ arrives. Under naive shared-reward MADDPG, both A and B see a positive value and both actors get reinforced — including B's useless wander.

Now apply the counterfactual baseline for B. The critic says: with B's actual action, $Q = 0.92$. Marginalising over B's policy (averaging the critic across the actions B *might* have taken), the baseline comes out to $0.90$. So $A_B = 0.92 - 0.90 = 0.02 \approx 0$ — B's specific action barely mattered, and its gradient is tiny. For A, the critic with A's actual hard push gives $Q = 0.92$, but marginalising over A's policy (which often does *not* push hard) gives a baseline of $0.55$, so $A_A = 0.92 - 0.55 = 0.37$ — a strong positive signal. A learns to push; B is not falsely reinforced. The team converges faster because the gradient finally points at the agent that mattered.

## 9. COMA: counterfactual multi-agent policy gradients in full

The counterfactual baseline deserves a section of its own, because COMA (Foerster et al., 2018) is the cleanest worked-out answer to the credit-assignment problem in cooperative MARL, and understanding it sharpens exactly what MADDPG does and does not solve. COMA, like MADDPG, is a CTDE actor-critic. The difference is in the *critic's job* and the *advantage signal* that drives the actor.

**The setup.** All $n$ agents share a single team reward $r$. Each agent has a stochastic decentralised policy $\pi_i(a_i \mid \tau_i)$ over discrete actions, conditioned on its own action-observation history $\tau_i$. A *single* centralised critic $Q(s, a_1, \ldots, a_n)$ estimates the team's joint action-value from the global state and all agents' actions. So far this looks like a shared-critic MADDPG. The innovation is what COMA feeds into the policy gradient.

**The credit-assignment problem, stated formally.** In vanilla multi-agent policy gradient, each agent's update uses the shared advantage built from the team value:

$$
\nabla_{\theta_i} J = \mathbb{E}\bigl[\nabla_{\theta_i} \log \pi_i(a_i \mid \tau_i)\, Q(s, a_1, \ldots, a_n)\bigr].
$$

Every agent is reinforced in proportion to the *same* $Q$. If the team did well, the lazy free-rider's action gets the same glowing $Q$ as the agent that actually carried the round. The gradient cannot tell whose action caused the success, so it pushes every agent's policy up indiscriminately — and the free-rider's useless behaviour is reinforced right alongside the useful one. This is precisely the lazy-agent pathology from the previous section, now written as an equation.

**The counterfactual baseline.** COMA replaces the raw $Q$ with an advantage that subtracts a baseline marginalising out *agent $i$'s own action while holding everyone else's action fixed*:

$$
A_i(s, \boldsymbol{a}) = Q(s, a_1, \ldots, a_n) - \sum_{a_i'} \pi_i(a_i' \mid \tau_i)\, Q\bigl(s, a_1, \ldots, a_{i-1}, a_i', a_{i+1}, \ldots, a_n\bigr).
$$

Write $a_{-i}$ for "everyone except $i$". The second term is $\mathbb{E}_{a_i' \sim \pi_i}[Q(s, a_{-i}, a_i')]$ — the expected team value if agent $i$ had sampled its action from its current policy while everyone else's action stayed *exactly as observed*. The advantage $A_i$ therefore measures "how much better than my own policy's average did *this specific action* make the team, given what my teammates actually did?" The agent gradient becomes

$$
\nabla_{\theta_i} J = \mathbb{E}\bigl[\nabla_{\theta_i} \log \pi_i(a_i \mid \tau_i)\, A_i(s, \boldsymbol{a})\bigr].
$$

**Why this solves credit assignment.** The baseline is a *true* baseline in the policy-gradient sense: because it does not depend on the actually-sampled $a_i$ (it averages over all of $a_i'$), subtracting it leaves the gradient's expectation unchanged — it does not introduce bias. What it *does* change is the variance and, crucially, the *attribution*. An agent whose action is no better than its own policy's average gets $A_i \approx 0$ and therefore a near-zero gradient — the free-rider stops being reinforced. An agent whose action genuinely lifted the team above its baseline gets a large positive $A_i$ and a strong gradient. Credit flows to the action that moved the needle. The counterfactual "what if agent $i$ had acted differently while everyone else held fixed" is the formal embodiment of the question a human coach asks when assigning credit on a team.

**Why the centralised critic is mandatory here.** Evaluating the baseline requires querying $Q(s, a_{-i}, a_i')$ for *every* alternative action $a_i'$ agent $i$ might have taken, holding the teammates' actions fixed. That is impossible with a decentralised critic $Q_i(o_i, a_i)$, which never sees $a_{-i}$ and so cannot hold them fixed. COMA's efficiency trick is to architect the critic so that a single forward pass outputs the $Q$-values for *all* of agent $i$'s actions at once (the network takes $s$ and $a_{-i}$ as input and emits a vector over agent $i$'s action set), so the entire sum $\sum_{a_i'} \pi_i(a_i') Q(\cdot)$ is one dot product rather than $|\mathcal{A}_i|$ forward passes. This is only tractable because the action space is *discrete* — which is the first axis on which COMA and MADDPG part ways.

**COMA versus MADDPG, head to head.** They are siblings, but their division of labour differs sharply:

- *Action space.* COMA's counterfactual marginalisation $\sum_{a_i'}$ is a finite sum, so it is built for **discrete** actions. MADDPG's deterministic-policy-gradient actor handles **continuous** actions natively (you cannot enumerate a continuous action set to form COMA's baseline; you would have to integrate).
- *Reward structure.* COMA assumes a **shared team reward** — there is one $Q$ and one team advantage. MADDPG keeps a **per-agent critic $Q_i$** and supports cooperative, competitive, *and* mixed rewards. COMA has nothing to say about competition.
- *On-policy vs off-policy.* COMA is **on-policy** (it is an A2C-style actor-critic; the baseline uses the *current* $\pi_i$, so it cannot replay stale data). MADDPG is **off-policy** with a replay buffer. This is the same on-policy/off-policy trade you see between PPO and DDPG, lifted to many agents.
- *What each fixes.* MADDPG's centralised critic primarily fixes **non-stationarity**; it does *not*, by itself, fix credit assignment under a shared reward (hence the lazy-agent equilibrium). COMA's counterfactual baseline is squarely aimed at **credit assignment**. The cleanest mental model: MADDPG stabilises *whose-environment-is-moving*, COMA disentangles *whose-action-mattered*. A cooperative-discrete problem with severe free-riding wants COMA's baseline; a continuous-control or competitive problem wants MADDPG's per-agent critic.

#### Worked example: COMA's counterfactual on a discrete 3-agent task

Three agents, each choosing from $\{$up, down, stay$\}$, share a team reward. On this step agent 2's chosen action was "up". The centralised critic, fed the global state and agents 1 and 3's fixed actions, emits the vector of team values over agent 2's three options: $Q(\cdot, \text{up}) = 4.0$, $Q(\cdot, \text{down}) = 1.0$, $Q(\cdot, \text{stay}) = 1.6$. Agent 2's current policy is $\pi_2 = (0.5, 0.2, 0.3)$ over (up, down, stay). The counterfactual baseline is the policy-weighted average:

$$
b_2 = 0.5(4.0) + 0.2(1.0) + 0.3(1.6) = 2.0 + 0.2 + 0.48 = 2.68.
$$

Agent 2 actually played "up", so its advantage is $A_2 = Q(\cdot, \text{up}) - b_2 = 4.0 - 2.68 = 1.32$ — strongly positive, so the gradient pushes $\pi_2$ further toward "up". Had agent 2 played "down" instead, its advantage would have been $1.0 - 2.68 = -1.68$, a strong push *away* from "down". The baseline $2.68$ is the agent's own expected contribution; the advantage measures the deviation from it, and that deviation — not the raw team value of $4.0$ shared blindly — is what trains the policy. Note the entire calculation needed the critic to output $Q$ across *all three of agent 2's actions* in one pass, which is exactly the discrete-action architecture COMA relies on.

## 10. Competitive and mixed settings

CTDE is not just for cooperation. In **competitive** games — predator-prey, two-player adversarial control — each agent has its own reward, often $R_1 = -R_2$ in the zero-sum case. Each agent keeps its own centralised critic $Q_i$, trained on its own reward, but still conditioned on the joint observation and all actions. The conditioning is what lets agent $i$ *anticipate* its opponent: because $Q_i$ has seen how outcomes depend on the opponent's action, the actor learns a best response to the opponent's current behaviour rather than treating it as random noise.

There is a subtlety in competitive MADDPG that the original paper addresses: at execution time you do not know your opponent's policy, and during training the opponent is improving, so a best response to *today's* opponent may be brittle. The mitigation Lowe et al. propose is **policy ensembles** — train each agent against a small population of opponent policies and sample which one to face each episode. This prevents the actor from overfitting to a single opponent and makes the learned policy robust to a range of adversaries, much like self-play with a pool of past checkpoints in game-playing systems.

The **mixed** setting is the general case: each agent $i$ has its own reward $R_i$ that is neither identical to nor the negation of the others'. Think of an economic simulation where agents have overlapping but distinct objectives. Here you *must* keep per-agent critics $Q_i$ with per-agent rewards. The joint observation still does its job: it stabilises every critic's training by removing the non-stationarity, *even though the rewards differ*. This is the most honest demonstration that CTDE is about stationarity, not about cooperation — the mechanism that fixes the moving-target problem is orthogonal to whether the agents are friends or enemies.

A common confusion worth heading off: "if MADDPG works for competition, doesn't a centralised critic give an unfair advantage by seeing the opponent's private info?" No — because that information is used *only to train the critic*, which is discarded. The deployed actor $\mu_i(o_i)$ never sees the opponent. You are using simulation-time omniscience to shape a policy that is, at run time, perfectly fair and local.

## 11. QMIX: value factorisation for cooperative MARL

MADDPG's per-agent critic $Q_i(x, a_1, \ldots, a_n)$ has a scaling problem: its input grows linearly with the number of agents, and with many agents the joint action space is enormous. For purely cooperative tasks, a different and often better idea is **value factorisation**, and QMIX (Rashid et al. 2018) is its most influential instance.

QMIX learns one *global* action-value $Q_{\text{tot}}$ for the whole team, but factors it as a function of per-agent utilities. Each agent has a network $Q_i(\tau_i, a_i)$ depending only on its own action-observation history $\tau_i$ and action $a_i$ — these are what run decentrally. A **mixing network** $f$ combines them into the team value:

$$
Q_{\text{tot}}(\boldsymbol{\tau}, \boldsymbol{a}) = f_s\bigl(Q_1(\tau_1, a_1), \ldots, Q_n(\tau_n, a_n)\bigr),
$$

where the subscript $s$ indicates that the mixing function's weights depend on the global state $s$ (available at training time). The diagram below shows the structure: per-agent utilities feed a mixing network whose weights come from a hypernetwork conditioned on the state.

![Graph of the QMIX mixing network where per-agent utility values branch into a mixing network with non-negative hypernetwork weights producing a monotone joint action value](/imgs/blogs/maddpg-centralised-training-decentralised-execution-5.png)

The genius of QMIX is a single constraint: **monotonicity**. The mixing network is constrained so that

$$
\frac{\partial Q_{\text{tot}}}{\partial Q_i} \ge 0 \quad \text{for all } i.
$$

Why does this matter so much? Because of what it implies for decentralised execution. At run time, each agent picks its action greedily by maximising its *own* $Q_i$: $a_i = \arg\max_{a_i} Q_i(\tau_i, a_i)$. For that decentralised greedy choice to also maximise the *team* value $Q_{\text{tot}}$, we need

$$
\arg\max_{\boldsymbol{a}} Q_{\text{tot}}(\boldsymbol{\tau}, \boldsymbol{a}) = \bigl(\arg\max_{a_1} Q_1, \ldots, \arg\max_{a_n} Q_n\bigr).
$$

This property is called **IGM** (Individual-Global-Max). Monotonicity is a sufficient condition for IGM: if $Q_{\text{tot}}$ is increasing in every $Q_i$, then pushing each $Q_i$ to its max also pushes $Q_{\text{tot}}$ to its max. So you can train a fully centralised $Q_{\text{tot}}$ on the team reward, yet *execute* by having each agent greedily maximise its own local utility — and the two agree. That is CTDE realised through factorisation rather than through a per-agent critic.

The monotonicity is enforced architecturally: the mixing network's weights are constrained non-negative (via an absolute value or a squaring on the hypernetwork outputs), and a network with non-negative weights and monotone activations is monotone in its inputs. The weights themselves are produced by **hypernetworks** that take the global state $s$ as input — this lets the mixing depend richly on the state (the value of coordination can differ wildly between states) while still being monotone in the per-agent $Q_i$. Here is the mixing network in PyTorch:

```python
class QMixer(nn.Module):
    def __init__(self, n_agents, state_dim, mix_hidden=32):
        super().__init__()
        self.n = n_agents
        self.mix_hidden = mix_hidden
        # Hypernetworks produce the (non-negative) mixing weights from the state.
        self.hyper_w1 = nn.Linear(state_dim, n_agents * mix_hidden)
        self.hyper_w2 = nn.Linear(state_dim, mix_hidden)
        self.hyper_b1 = nn.Linear(state_dim, mix_hidden)
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, mix_hidden), nn.ReLU(),
            nn.Linear(mix_hidden, 1),
        )

    def forward(self, agent_qs, state):
        # agent_qs: (B, n_agents), state: (B, state_dim)
        B = agent_qs.size(0)
        # abs() enforces non-negative weights -> monotonicity -> IGM.
        w1 = torch.abs(self.hyper_w1(state)).view(B, self.n, self.mix_hidden)
        b1 = self.hyper_b1(state).view(B, 1, self.mix_hidden)
        hidden = F.elu(torch.bmm(agent_qs.view(B, 1, self.n), w1) + b1)
        w2 = torch.abs(self.hyper_w2(state)).view(B, self.mix_hidden, 1)
        b2 = self.hyper_b2(state).view(B, 1, 1)
        q_tot = torch.bmm(hidden, w2) + b2          # (B, 1, 1)
        return q_tot.view(B, 1)
```

QMIX is trained end-to-end exactly like DQN: minimise the TD error of $Q_{\text{tot}}$ against a target $r + \gamma \max_{\boldsymbol{a}'} Q_{\text{tot}}'(\boldsymbol{\tau}', \boldsymbol{a}')$, where the max factorises agent-by-agent thanks to IGM. The gradient flows through the mixer into each per-agent network. The per-agent networks are what you deploy.

### The hypernetwork architecture, in detail

The phrase "the mixing weights depend on the state" hides a specific and clever architecture, and it is worth being concrete because the structure is what makes QMIX both expressive and monotone at once. The mixing network is a tiny two-layer feed-forward network whose *inputs* are the per-agent utilities $(Q_1, \ldots, Q_n)$ and whose *weights are not free parameters* — they are produced on the fly by **hypernetworks** that read the global state $s$. Concretely, for a mixing network with a hidden width $H$ (32 in the code above), the pieces are:

- **First-layer weights $W_1$.** A hypernetwork maps $s \mapsto W_1 \in \mathbb{R}^{n \times H}$, then passes the output through an absolute value so $W_1 \ge 0$ elementwise. This is the $n \to H$ projection of the per-agent utilities.
- **First-layer bias $b_1$.** A separate (ordinary, possibly-negative) linear layer maps $s \mapsto b_1 \in \mathbb{R}^{H}$. Biases need not be non-negative — only the *weights* multiplying the $Q_i$ must be, since only they govern monotonicity in the inputs.
- **Hidden activation.** The first layer computes $h = \text{ELU}(Q^\top W_1 + b_1)$, where $Q = (Q_1, \ldots, Q_n)$. ELU is monotone non-decreasing, which preserves monotonicity (a monotone function of a monotone-increasing pre-activation is monotone-increasing).
- **Second-layer weights $W_2$.** Another hypernetwork maps $s \mapsto W_2 \in \mathbb{R}^{H \times 1}$, again passed through an absolute value so $W_2 \ge 0$. This collapses the hidden layer to the scalar $Q_{\text{tot}}$.
- **Second-layer bias $b_2$.** Importantly, $b_2$ is produced by a *deeper* hypernetwork (a two-layer ReLU network on $s$, as in the `hyper_b2` field of the code), because the final state-dependent offset benefits from extra capacity — it can encode "the baseline value of this state regardless of any agent's choice," which is often a rich function of $s$.

The forward pass is therefore $Q_{\text{tot}} = W_2^\top\, \text{ELU}(W_1^\top Q + b_1) + b_2$, with $W_1, W_2 \ge 0$ and everything conditioned on $s$ through the hypernetworks. Why route the state through hypernetworks rather than just concatenating $s$ onto the $Q_i$ inputs? Because if $s$ were a plain input to the mixing network, enforcing $\partial Q_{\text{tot}} / \partial Q_i \ge 0$ would not constrain how $s$ enters, but it *would* be hard to let $s$ modulate the mixing *multiplicatively* (e.g. "in this state, agent 3's utility should count for much more"). Hypernetworks let the state set the *weights themselves*, so the value of coordination can swing wildly between states, while the non-negativity constraint on those weights still guarantees monotonicity in the per-agent utilities. You get state-dependent mixing and a monotonicity guarantee simultaneously — that is the architectural payoff.

### The IGM principle and why monotonicity is sufficient for it

The **Individual-Global-Max (IGM)** principle is the formal contract that any factorised value method must satisfy to make decentralised execution correct. It states that there exists a factorisation such that the joint greedy action equals the vector of per-agent greedy actions:

$$
\arg\max_{\boldsymbol{a}} Q_{\text{tot}}(\boldsymbol{\tau}, \boldsymbol{a}) = \Bigl(\arg\max_{a_1} Q_1(\tau_1, a_1), \ldots, \arg\max_{a_n} Q_n(\tau_n, a_n)\Bigr).
$$

Why is this the *right* contract? Because at execution each agent acts on its own utility alone, $a_i^\star = \arg\max_{a_i} Q_i(\tau_i, a_i)$ — no agent can consult the joint $Q_{\text{tot}}$ or its teammates. IGM is exactly the guarantee that this fully decentralised, greedy, local procedure reproduces the centralised optimum. Without IGM, training a great centralised $Q_{\text{tot}}$ would be useless, because the decentralised agents could not recover its argmax.

**Monotonicity is a sufficient condition for IGM.** Here is the proof, and it is short enough to do in full. Suppose $Q_{\text{tot}} = f(Q_1, \ldots, Q_n)$ with $\partial f / \partial Q_i \ge 0$ for every $i$. Let each agent pick its individual greedy action $a_i^\star = \arg\max_{a_i} Q_i(\tau_i, a_i)$, so that $Q_i(\tau_i, a_i^\star) \ge Q_i(\tau_i, a_i)$ for all $a_i$. Now take *any* other joint action $\boldsymbol{a} = (a_1, \ldots, a_n)$. We can transform $\boldsymbol{a}$ into $\boldsymbol{a}^\star$ one coordinate at a time, and at each step we replace some $a_k$ by $a_k^\star$, which can only *increase* $Q_k$ (by greedy choice) and therefore — since $f$ is non-decreasing in its $k$-th argument and we change nothing else — can only increase $Q_{\text{tot}} = f(\ldots)$. After $n$ such non-decreasing steps we have reached $\boldsymbol{a}^\star$, so

$$
Q_{\text{tot}}(\boldsymbol{\tau}, \boldsymbol{a}^\star) \ge Q_{\text{tot}}(\boldsymbol{\tau}, \boldsymbol{a}) \quad \text{for all } \boldsymbol{a}.
$$

That is the definition of $\boldsymbol{a}^\star$ being the joint argmax. Hence the per-agent greedy actions *are* the joint greedy action — IGM holds. The one-coordinate-at-a-time argument is the whole engine: monotonicity means "raising any single agent's utility never lowers the team value," and that is precisely what lets each agent maximise locally without coordination. Notice also that the $\max$ over the joint action — needed for the DQN-style target $\max_{\boldsymbol{a}'} Q_{\text{tot}}'$ — *factorises* under IGM into $n$ small per-agent maxes, turning an exponential $\prod_i |\mathcal{A}_i|$ search into a linear $\sum_i |\mathcal{A}_i|$ one. The tractability of the Bellman target is itself a gift of monotonicity.

### Why VDN's absolute Q-value sharing is insufficient

QMIX's predecessor, **VDN** (value-decomposition networks), takes the simplest possible monotone factorisation: a plain sum,

$$
Q_{\text{tot}}^{\text{VDN}}(\boldsymbol{\tau}, \boldsymbol{a}) = \sum_{i=1}^{n} Q_i(\tau_i, a_i).
$$

A sum trivially satisfies monotonicity ($\partial Q_{\text{tot}} / \partial Q_i = 1 \ge 0$), so VDN also satisfies IGM and admits decentralised greedy execution. So why is it insufficient and why did QMIX win? Two reasons, and the distinction is instructive.

First, the sum is **state-independent in its mixing**: each agent's utility contributes with a fixed unit weight regardless of the global situation. But the *value of an agent's contribution genuinely depends on state*. In a StarCraft fight, one unit's positioning might be decisive when the enemy is clustered and irrelevant when they are scattered; a fixed unit weight cannot express "agent 3's utility matters ten times more in this state than that one." QMIX's state-conditioned hypernetwork weights $W_1(s), W_2(s)$ recover exactly this flexibility — the mixing is a rich, learned, state-dependent function, of which the VDN sum is the degenerate special case where all weights are pinned to one and the state is ignored.

Second — and this is the subtler point the team lead's phrase "absolute Q-value sharing" gets at — VDN forces each per-agent $Q_i$ to live on a *shared absolute scale* with the others, because they are summed directly. The decomposition implicitly demands that agent $i$'s utility be expressed in the same units as the team value it contributes to, so that the raw magnitudes add up to the true $Q_{\text{tot}}$. That is a strong and often false constraint: an agent whose local observations give it almost no information about the global return is forced to emit absolute values that nonetheless sum correctly, which over-constrains its network and leaks the credit-assignment problem back in. QMIX relaxes this: the per-agent $Q_i$ need only carry the right *ordering* information (which of agent $i$'s actions is better), and the state-dependent mixer is free to rescale and recombine those utilities into the correct absolute team value. The per-agent networks learn *relative* preferences; the mixer learns the *absolute* combination. Empirically this is why QMIX substantially outperforms VDN on the harder SMAC maps — the representational gap between "fixed-weight sum of shared-scale utilities" and "state-dependent monotone mixing of relative utilities" is exactly the gap between coordination QMIX can express and coordination VDN cannot.

#### Worked example: a QMIX factorisation that monotonicity can and cannot represent

QMIX's monotonicity is powerful but not universal. Consider two agents, each choosing action $\in \{A, B\}$, with a team payoff matrix. A *monotone-friendly* payoff: both choosing $A$ gives the team $8$, mismatches give $3$, both $B$ gives $6$. Here a monotone mixer can represent the value — set $Q_1(A) > Q_1(B)$ and $Q_2(A) > Q_2(B)$, and the greedy joint choice $(A, A)$ correctly hits the max of $8$.

Now a payoff that breaks monotonicity, the classic non-monotonic matrix from the QMIX/QTRAN literature: $(A,A) = 8$, $(A,B) = -12$, $(B,A) = -12$, $(B,B) = 0$. The optimum is $(A,A) = 8$, but the action $A$ is catastrophic when the partner chooses $B$. A monotone mixer must assign each agent a *fixed* preference ordering over its own actions independent of the partner, and no such ordering captures "$A$ is great together but terrible alone". QMIX provably cannot represent this; it tends to collapse to the safe $(B,B) = 0$. This is the known limitation that QPLEX and QTRAN were designed to address by relaxing the strict monotonicity while keeping IGM. The lesson: factorised critics trade representational completeness for scalability, and for tightly-coupled coordination that trade can bite.

## 12. MAPPO: the surprisingly strong baseline

For several years the conventional wisdom was that off-policy methods like MADDPG and QMIX were necessary for sample-efficient MARL, and that on-policy policy gradients were too high-variance to compete. Then Yu et al. (2022) ran a careful study and found that **MAPPO** — multi-agent PPO — matches or beats those methods on a wide range of cooperative benchmarks, while being dramatically simpler.

MAPPO is almost embarrassingly direct. Take [PPO](/blog/machine-learning/reinforcement-learning/proximal-policy-optimization-ppo) — the clipped-surrogate, on-policy actor-critic that dominates single-agent deep RL — and make the *value function* centralised. Each agent has a decentralised policy $\pi_i(a_i \mid o_i)$, but they share (or each have) a centralised value baseline $V(x)$ that conditions on the joint observation. The advantage estimate that drives every agent's policy gradient uses this centralised value:

$$
\hat{A}_i = \text{GAE}\bigl(r, V(x)\bigr),
$$

computed with generalised advantage estimation off the centralised critic. The policy update is the ordinary PPO clipped surrogate, per agent:

$$
\mathcal{L}_i = \mathbb{E}\Bigl[\min\bigl(\rho_i \hat{A}_i,\ \text{clip}(\rho_i, 1-\epsilon, 1+\epsilon)\hat{A}_i\bigr)\Bigr], \quad \rho_i = \frac{\pi_i(a_i \mid o_i)}{\pi_i^{\text{old}}(a_i \mid o_i)}.
$$

The centralised *value* (not action-value) is the only structural change from single-agent PPO. Why does this work so well? A few reasons that the paper teases apart. First, the centralised value reduces the variance of the advantage estimate by accounting for the part of the return caused by teammates, which a decentralised value would absorb as noise — this directly attacks the credit-assignment variance. Second, on-policy data sidesteps the stale-buffer pathology entirely: PPO throws away its data after each update, so there is never a buffer full of transitions from extinct opponent policies. The non-stationarity that wrecks off-policy MARL is far milder on-policy, because the data is always fresh. Third, PPO's trust-region clipping keeps each update small, which is exactly what you want when the *ground* (the other agents) is shifting under you.

The practical implication is strong: **if your problem is cooperative and you can afford on-policy sample collection, start with MAPPO.** It is simpler to implement, has fewer moving parts than MADDPG (no per-agent critics, no replay buffer, no target networks for an action-value), and is competitive or better on SMAC, MPE, and Hanabi. Reach for off-policy MADDPG/QMIX when sample efficiency is paramount (real-robot data is expensive) or when the setting is competitive. Setting MAPPO up in RLlib is a few lines:

```python
from ray.rllib.algorithms.ppo import PPOConfig

config = (
    PPOConfig()
    .environment("simple_spread_v3")            # a PettingZoo MPE env wrapper
    .multi_agent(
        policies={"shared_policy"},
        policy_mapping_fn=lambda agent_id, *a, **k: "shared_policy",
    )
    .training(
        train_batch_size=4000,
        sgd_minibatch_size=512,
        num_sgd_iter=10,
        clip_param=0.2,
        gamma=0.99,
        lambda_=0.95,
        entropy_coeff=0.01,                      # keep exploration alive
    )
    .rl_module(
        # The model concatenates the joint observation for the value head:
        # centralised critic, decentralised actor.
        model_config_dict={"vf_share_layers": False},
    )
)
algo = config.build()
for i in range(500):
    result = algo.train()
    print(i, result["env_runners"]["episode_reward_mean"])
```

The matrix below summarises how the four main CTDE families trade off across settings and scale. It is the figure I would pin above my desk when picking an algorithm.

![Matrix comparing IQL, MADDPG, QMIX, MAPPO and QPLEX across centralised training, cooperative, competitive, and scalability dimensions](/imgs/blogs/maddpg-centralised-training-decentralised-execution-6.png)

## 13. The cooperative navigation case study

The canonical demonstration in the MADDPG paper is **cooperative navigation** (also called `simple_spread`): $N$ agents and $N$ landmarks in a 2D plane. The team is rewarded for covering all landmarks — specifically, the reward is the negative sum of distances from each landmark to its nearest agent — and is *penalised* for collisions between agents. Crucially, there is no explicit "agent 1 go to landmark 3" instruction. The agents must implicitly divide the labour.

What MADDPG produces here is genuine emergent coordination. Watch a trained policy and you see the agents *spread out* — each gravitating to a different landmark — without ever being told which one is "theirs". The division of labour falls out of the centralised critic: because each agent's critic sees where the others are heading, the actor learns "if my teammate is already covering the left landmark, I should peel off to the right one." With independent learners (IQL) on the same task, you frequently see two agents pile onto the same landmark while a third sits uncovered — a coordination failure that the non-stationary, blind-to-each-other critics never resolve.

The collision-avoidance behaviour is the subtler emergent result. There is a collision *penalty* in the reward, but no explicit "swerve to avoid" instruction. The agents learn smooth avoidance trajectories anyway, because the centralised critic propagates the penalty back to the action that *caused* the near-collision. The original paper reports that MADDPG agents collide far less often than independently-trained DDPG agents on this task, and reach higher team reward, with the gap widening as the number of agents grows — exactly the regime where non-stationarity bites hardest.

#### A note on measuring this honestly

If you reproduce cooperative navigation, the metric to track is mean episode team reward (less negative is better) and collision count per episode, averaged over many evaluation episodes with exploration noise off. Expect MADDPG to plateau meaningfully above an IQL/independent-DDPG baseline on the 3-agent variant, with the separation growing for 6 agents. Be honest about variance: MARL results are notoriously seed-sensitive, so report across at least 5 seeds with confidence intervals. A single lucky seed showing MADDPG winning proves little — the seed-to-seed spread in MARL can exceed the algorithm-to-algorithm difference, which is one of the field's open embarrassments and a reason to treat any single headline number with suspicion.

## 14. Case studies: where CTDE has actually delivered

**MADDPG on the MPE suite (Lowe et al., 2017).** The originating result. Across cooperative communication, cooperative navigation, predator-prey, and physical deception tasks, MADDPG outperformed independent DDPG, independent DQN, and a centralised-everything baseline. The headline lesson was not a single benchmark number but the demonstration that the centralised-critic/decentralised-actor split *stabilises learning* across cooperative, competitive, and mixed regimes alike. It also introduced the policy-ensemble trick for robustness against non-stationary opponents.

**QMIX on StarCraft II micromanagement (Rashid et al., 2018).** The StarCraft Multi-Agent Challenge (SMAC) became the field's standard cooperative benchmark, and QMIX was the method that put value factorisation on the map there. On hard SMAC maps requiring tight unit coordination — focus-firing, kiting, positioning — QMIX substantially beat independent Q-learning and the earlier VDN (value-decomposition networks, which used a simple *sum* of per-agent values, a special case of QMIX's monotone mixing). The win rates on the harder maps climbed from near-zero for independent learners to strong fractions for QMIX, demonstrating that the monotone mixer's expressiveness over a plain sum mattered in practice.

**MAPPO on SMAC and beyond (Yu et al., 2022).** The result that recalibrated the field. The paper showed MAPPO achieving win rates competitive with or exceeding QMIX and other off-policy methods across the majority of SMAC maps, plus strong results on MPE and the cooperative card game Hanabi — all with a far simpler on-policy recipe and careful but standard implementation tricks (value normalisation, observation concatenation for the centralised critic, action masking, proper GAE). The practical takeaway that propagated through the community: a well-tuned on-policy PPO with a centralised value baseline is a formidable and under-rated MARL baseline, and many papers claiming improvements over weak baselines did not actually beat a properly-tuned MAPPO.

**OpenAI Five (2019), as a CTDE-adjacent point of reference.** OpenAI's Dota 2 system trained five agents that coordinated at superhuman level. While it used a more bespoke architecture (a large shared LSTM policy with per-hero heads and extensive self-play rather than textbook MADDPG/QMIX), it embodies the CTDE spirit at scale: training-time access to rich global game state shaping policies that act on per-hero observations. It is a useful reminder that the CTDE *principle* generalises well beyond the specific algorithms, and that at extreme scale the engineering (self-play curricula, surgery to grow the model, enormous batch sizes) often matters as much as the core algorithm.

## 15. RLlib setup for MADDPG and MAPPO

For anything beyond a toy, you want a battle-tested distributed framework rather than the from-scratch loop in section 7 — the hand-rolled version is for *understanding*, not production. RLlib is the standard choice for multi-agent. The multi-agent interface is `MultiAgentEnv`, where `step` and `reset` return dictionaries keyed by agent id. Here is the skeleton for a MADDPG-style setup; note that RLlib's first-class multi-agent algorithms have shifted over versions, so in current RLlib you often implement MADDPG via a custom centralised-critic model on top of an off-policy learner, while MAPPO maps cleanly onto the multi-agent PPO path shown in section 12.

```python
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from gymnasium import spaces
import numpy as np


class SimpleSpreadEnv(MultiAgentEnv):
    def __init__(self, n_agents=3):
        super().__init__()
        self.n = n_agents
        self.agents = [f"agent_{i}" for i in range(n_agents)]
        self.observation_space = spaces.Dict({
            a: spaces.Box(-np.inf, np.inf, (4,), np.float32) for a in self.agents
        })
        self.action_space = spaces.Dict({
            a: spaces.Box(-1.0, 1.0, (2,), np.float32) for a in self.agents
        })

    def reset(self, *, seed=None, options=None):
        obs = {a: np.zeros(4, np.float32) for a in self.agents}
        return obs, {}

    def step(self, action_dict):
        # ... environment dynamics ...
        obs = {a: np.zeros(4, np.float32) for a in self.agents}
        rew = {a: 0.0 for a in self.agents}            # shared team reward in cooperative case
        terminated = {"__all__": False}
        truncated = {"__all__": False}
        return obs, rew, terminated, truncated, {}
```

The thing to internalise about RLlib multi-agent is the `policy_mapping_fn`: it decides which agents share which policy. For a homogeneous cooperative swarm, map all agents to one shared policy (parameter sharing — fewer parameters, faster learning, and it exploits the symmetry that all agents are interchangeable). For heterogeneous or competitive agents, give each its own policy. Parameter sharing is one of the highest-leverage knobs in cooperative MARL and is orthogonal to the CTDE question — you can share actor parameters across agents while still using a centralised critic.

```python
config = (
    PPOConfig()
    .environment(SimpleSpreadEnv, env_config={"n_agents": 3})
    .multi_agent(
        policies={"shared"},
        policy_mapping_fn=lambda aid, *a, **k: "shared",   # parameter sharing
    )
    .env_runners(num_env_runners=8)                        # parallel rollout workers
    .training(train_batch_size=8000, gamma=0.99, lambda_=0.95)
)
```

## 17. When to use CTDE methods (and when not to)

CTDE is the default for cooperative and mixed multi-agent problems, but it is not free, and there are regimes where simpler choices win. Be decisive about this.

**Use MAPPO when** the task is cooperative, you can collect on-policy data at reasonable cost (simulator or cheap rollouts), and you want the simplest thing that works well. It is the strongest default baseline in 2022-era research and remains an excellent first attempt. Parameter sharing plus a centralised value baseline covers a huge fraction of cooperative problems.

**Use QMIX or QPLEX when** the task is cooperative, sample efficiency matters (off-policy reuse of a replay buffer is worth it), and the action spaces are discrete. QMIX scales well in the number of agents because the per-agent networks are small and the joint value is *factored* rather than represented as a flat function of all actions. Reach for QPLEX or QTRAN if you suspect non-monotonic coordination (the $(A,A)$-great-but-$A$-alone-terrible payoff from section 11).

**Use MADDPG when** the action space is *continuous* (its DDPG lineage handles continuous control natively, which QMIX's discrete-argmax structure does not), or when the setting is *competitive/mixed* with per-agent rewards (the per-agent critic structure is built for this, while QMIX's monotone mixing assumes a shared cooperative value). MADDPG's per-agent critic is the right tool when agents have genuinely different objectives.

**Do not reach for any of these when** a simpler structure suffices. If the agents barely interact — their rewards and dynamics are nearly decoupled — independent learners (IQL or independent PPO) can be perfectly adequate and far simpler; the non-stationarity is mild when coupling is weak. If you have only *one* learning agent and the others are *fixed* (scripted opponents, a frozen policy), there is no non-stationarity and you should use plain single-agent RL — see [on-policy vs off-policy](/blog/machine-learning/reinforcement-learning/on-policy-vs-off-policy-a-practical-guide) for that choice. And if the number of agents is large and homogeneous (hundreds of identical agents), mean-field RL or a properly-shared-parameter independent learner often beats trying to feed a centralised critic a joint observation that grows linearly with the swarm — MADDPG's $O(n)$ critic input becomes the bottleneck. CTDE shines in the middle regime: a handful to a few dozen agents whose actions genuinely entangle.

The decision tree below captures this routing.

![Decision tree for choosing a CTDE algorithm based on whether the task is cooperative, whether a factored value works, whether policy gradients are preferred, and whether the setting is competitive or mixed](/imgs/blogs/maddpg-centralised-training-decentralised-execution-8.png)

The timeline of how these methods arrived helps situate them: MADDPG established the per-agent centralised critic, QMIX and COMA the same year showed factorisation and counterfactual baselines, and the on-policy MAPPO wave plus QPLEX-style relaxations rounded out the toolkit.

![Timeline of CTDE methods from MADDPG in 2017 through QMIX and COMA in 2018 to MAPPO and QPLEX in 2021 and MAT in 2022](/imgs/blogs/maddpg-centralised-training-decentralised-execution-7.png)

## 14. Implementation pitfalls I have hit

A few hard-won notes, because the gap between the clean equations and a run that actually converges is where the time goes.

**The joint observation must actually be joint.** A subtle bug: feeding the critic a concatenation of observations in an *inconsistent order* across the batch (agent indices swapped) silently destroys learning, because the critic cannot tell which slot is whose. Fix the agent ordering once and keep it religiously consistent between the buffer, the actor-action assembly, and the critic input.

**Soft-update both targets, and not too fast.** MADDPG has *two* target networks per agent (actor and critic). A too-large $\tau$ (say $0.1$) reintroduces instability because the targets chase the online networks too quickly while the other agents are also moving — you stack two sources of non-stationarity. I use $\tau = 0.01$ and update after every step; slower-and-steady wins.

**Exploration noise needs care in multi-agent.** Independent Gaussian noise per agent is the default, but in tightly-coupled tasks correlated exploration sometimes helps the team discover coordinated behaviours that independent jitter never stumbles into. If your agents plateau at an uncoordinated local optimum, this is a knob to try.

**Watch for the lazy-agent equilibrium.** In shared-reward cooperative tasks without a counterfactual baseline, MADDPG can settle into an equilibrium where one agent does all the work and the others free-ride (their gradients are weak because the team reward is already decent). If you see one agent dominating, add a COMA-style counterfactual baseline or a small per-agent shaping reward to break the symmetry.

**MARL results are seed-noisy — budget for it.** I said this in the case-study section and I will say it again because it is the single most common way to fool yourself. Run multiple seeds. A method that wins on seed 7 and loses on seeds 1-6 has not won. This is endemic to the field, not a flaw in your setup.

## 15. Key takeaways

- **Multi-agent RL breaks single-agent algorithms through non-stationarity:** each agent's environment includes the other learning agents, so the effective transition function moves as they learn, violating the Markov assumption that all convergence guarantees rest on.
- **CTDE resolves this by an information asymmetry:** train with global information (a critic that sees all observations and actions), execute with local information (an actor that sees only its own observation). The all-seeing critic is discarded before deployment.
- **The critic conditioning on all actions is what restores stationarity** — the non-stationarity was never in the world, it was in each agent's ignorance of the others' actions. Supply those actions and the moving target stops moving.
- **MADDPG = DDPG with a centralised critic.** The actor gradient differentiates only through agent $i$'s own action slot; the other agents' actions are constants that make the value well-defined. It handles continuous, competitive, and mixed settings.
- **Shared-reward cooperation reintroduces credit assignment;** a counterfactual baseline (COMA) marginalises out an agent's own action to ask "did *this* action help beyond my average?", killing the lazy-agent free-rider problem.
- **QMIX factorises the team value monotonically.** The constraint $\partial Q_{\text{tot}} / \partial Q_i \ge 0$ guarantees IGM, so decentralised greedy action selection agrees with the centralised optimum — at the cost of not representing non-monotonic coordination.
- **MAPPO — PPO with a centralised value baseline — is the strong, simple default for cooperative MARL,** often matching or beating off-policy methods while sidestepping the stale-buffer non-stationarity that hurts MADDPG and QMIX.
- **Pick by setting:** MAPPO for cooperative-and-simple, QMIX/QPLEX for cooperative-discrete-sample-efficient, MADDPG for continuous or competitive/mixed; plain single-agent RL when there is only one learner; and always run multiple seeds.

## 16. Further reading

- Lowe, Wu, Tamar, Harb, Abbeel, Mordatch, **"Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments"** (2017) — the MADDPG paper. The source for the centralised-critic/decentralised-actor formulation, the policy-ensemble robustness trick, and the MPE benchmark results.
- Rashid, Samvelyan, de Witt, Farquhar, Foerster, Whiteson, **"QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning"** (2018) — value factorisation, the monotonicity constraint, hypernetwork mixing, and SMAC results.
- Foerster, Farquhar, Afouras, Nardelli, Whiteson, **"Counterfactual Multi-Agent Policy Gradients"** (COMA, 2018) — the counterfactual baseline for multi-agent credit assignment.
- Yu, Velu, Vinitsky, Gao, Wang, Bayen, Wu, **"The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games"** (MAPPO, 2022) — the careful study showing on-policy PPO with a centralised value is a top-tier MARL baseline, plus the implementation details that make it work.
- Wang, Ren, Liu, Yu, Zhang, **"QPLEX: Duplex Dueling Multi-Agent Q-Learning"** (2021) — relaxing strict monotonicity while preserving IGM to represent non-monotonic coordination.
- Samvelyan et al., **"The StarCraft Multi-Agent Challenge"** (SMAC, 2019) — the benchmark that standardised cooperative MARL evaluation.
- Within this series: the [DDPG and TD3](/blog/machine-learning/reinforcement-learning/deterministic-policy-gradient-ddpg-td3) post for the deterministic-policy-gradient machinery MADDPG extends, the [credit assignment problem](/blog/machine-learning/reinforcement-learning/the-credit-assignment-problem) for why shared rewards are hard, and [PPO](/blog/machine-learning/reinforcement-learning/proximal-policy-optimization-ppo) for the single-agent algorithm MAPPO lifts to many agents. The series' unified map and capstone tie the multi-agent track back to the single-agent foundations.
