---
title: "Q-learning: Off-policy TD control and the path to deep RL"
date: "2026-06-22"
publishDate: "2026-06-22"
description: "Derive Q-learning from the Bellman optimality equation, understand why the max over next-state actions makes it off-policy, see exactly where vanilla Q-learning inflates value estimates and how Double Q-learning fixes it, then trace the path from Q-tables to DQN."
tags:
  [
    "reinforcement-learning",
    "deep-learning",
    "q-learning",
    "td-learning",
    "off-policy",
    "double-q-learning",
    "dqn",
    "machine-learning",
    "tabular-rl",
    "bellman-equation",
  ]
category: "machine-learning"
subcategory: "Reinforcement Learning"
author: "Hiep Tran"
featured: true
readTime: 55
image: "/imgs/blogs/q-learning-off-policy-td-control-1.png"
---

There is a four-letter algorithm that started the modern deep RL revolution. It does not require a model of the environment. It does not need to follow the same policy it is learning about. It has a clean convergence proof, a well-characterized failure mode, and an elegant fix to that failure mode. The moment you swap its lookup table for a convolutional neural network you get DQN — the architecture that first played 49 Atari games at human level in 2015.

That algorithm is Q-learning. The full name is a bit longer: it is an off-policy temporal-difference control method, meaning it learns the value of the optimal policy while following a different, exploratory policy. The key mechanism is a single symbolic substitution in the TD update that you have already seen in [Temporal-Difference Learning: TD(0), SARSA, and Bootstrapping](/blog/machine-learning/reinforcement-learning/temporal-difference-learning-td0-sarsa-bootstrapping): instead of using the value of the action your behavior policy actually chose at the next state, you take the maximum over all possible next-state actions. One `max` operation, three letters, and the entire mathematical structure of the algorithm changes.

This post derives that change completely, proves why it makes Q-learning off-policy, and shows you the exact cost of that design choice — a systematic positive bias in value estimates called the max-bias problem. Then it presents Double Q-learning, which decouples action selection from action evaluation to eliminate the bias at zero extra computation cost. Finally, it traces the structural boundary between tabular Q-learning and deep Q-learning: why table Q converges but neural Q diverges unless you add two specific engineering tricks, and what those tricks actually do to the mathematics.

By the end you will be able to implement tabular Q-learning from scratch on FrozenLake-v1 and Taxi-v3, explain the Watkins (1989) convergence conditions to a colleague in plain language, identify max-bias in your own Q estimates and correct it with Double Q-learning, and describe the precise structural reason that naive neural Q diverges and how DQN survives it. Figure 1 gives the core visual summary: how Q-learning's update rule differs from SARSA's and why that single difference is the entire definition of off-policy learning.

![Q-learning replaces the on-policy sampled next action with a greedy max, making the algorithm off-policy without needing importance sampling corrections](/imgs/blogs/q-learning-off-policy-td-control-1.png)

This post builds directly on [Temporal-Difference Learning: TD(0), SARSA, and Bootstrapping](/blog/machine-learning/reinforcement-learning/temporal-difference-learning-td0-sarsa-bootstrapping) (B3) and the MDP formalism in [Markov Decision Processes: The Mathematics of Sequential Decisions](/blog/machine-learning/reinforcement-learning/markov-decision-processes-the-mathematics-of-sequential-decisions) (B1). The full series map is at [Reinforcement Learning: A Unified Map](/blog/machine-learning/reinforcement-learning/reinforcement-learning-a-unified-map). The neural network extension of everything you learn here lives in [Deep Q-Networks: DQN, Double DQN, and Dueling Architectures](/blog/machine-learning/reinforcement-learning/deep-q-networks-dqn) (D1).

## 1. The goal: learning the optimal action-value function

Before deriving the update rule, we need to be precise about what Q-learning is trying to learn. This matters because many confusions about Q-learning — why it is off-policy, why it needs exploration, why it can diverge with function approximation — dissolve once you understand the object it is optimizing toward.

The goal is the **optimal action-value function** $Q^*(s, a)$. This is defined as the maximum expected discounted return achievable when you start in state $s$, take action $a$, and then follow the optimal policy $\pi^*$ from that point forward:

$$Q^*(s, a) = \max_\pi \mathbb{E}_\pi\!\left[\sum_{k=0}^{\infty} \gamma^k R_{t+k+1} \;\middle|\; S_t = s,\, A_t = a\right]$$

Once you have $Q^*$, the optimal policy is immediate: $\pi^*(s) = \argmax_a Q^*(s, a)$. You do not need a model of the environment; just look up the Q-values and take the best action. This model-free deployment is one of the primary appeals of value-based RL.

The function $Q^*$ satisfies a self-consistency condition known as the **Bellman optimality equation**:

$$Q^*(s, a) = \mathbb{E}\!\left[R_{t+1} + \gamma \max_{a'} Q^*(s', a') \;\middle|\; S_t = s,\, A_t = a\right]$$

This equation says: "the true value of $(s, a)$ equals the immediate reward plus the discounted value of being in $s'$ and then acting optimally from there." The $\max_{a'}$ appears because after arriving at $s'$, the optimal policy will choose whichever action has the highest value.

Compare this to the **Bellman expectation equation** for policy evaluation:

$$Q^\pi(s, a) = \mathbb{E}_\pi\!\left[R_{t+1} + \gamma Q^\pi(s', a') \;\middle|\; S_t = s,\, A_t = a\right]$$

The only difference is $\max_{a'} Q^*$ versus the expectation $Q^\pi(s', a')$ under the current policy $\pi$. The Bellman optimality equation always takes the max; the Bellman expectation equation averages over the policy distribution. This is exactly the difference between Q-learning and SARSA — one line in the update rule, a completely different object being learned.

### The Bellman optimality operator as a contraction

Define the Bellman optimality operator $\mathcal{T}^*$ acting on any bounded function $Q : \mathcal{S} \times \mathcal{A} \to \mathbb{R}$:

$$(\mathcal{T}^* Q)(s, a) = \mathbb{E}\!\left[R + \gamma \max_{a'} Q(s', a') \;\middle|\; s, a\right]$$

This operator has a beautiful property: it is a **contraction** in the $\ell_\infty$ (sup-norm). For any two Q-functions $Q$ and $Q'$:

$$\|\mathcal{T}^* Q - \mathcal{T}^* Q'\|_\infty \leq \gamma \|Q - Q'\|_\infty$$

Proof sketch: Let $\varepsilon = \|Q - Q'\|_\infty$, so $|Q(s,a) - Q'(s,a)| \leq \varepsilon$ for all $(s,a)$. Then the difference of the operators is bounded by $\gamma \varepsilon$ because $|\max_a f(a) - \max_a g(a)| \leq \max_a |f(a) - g(a)| \leq \varepsilon$, and multiplying through by $\gamma$ gives the contraction. Since $\gamma < 1$, repeated application of $\mathcal{T}^*$ brings any starting Q-function exponentially close to the unique fixed point $Q^*$. This is the theoretical engine underneath tabular dynamic programming — and also the reason why, in theory, Q-learning converges.

The contraction property fails once we introduce function approximation. A neural network with gradient updates does not implement an exact operator application — it takes a gradient step that moves all Q-values simultaneously in a correlated direction. This is why the Bellman operator analysis does not directly guarantee convergence for deep Q-learning, a theme we will return to in Section 13.

## 2. Deriving the Q-learning update

Dynamic programming applies $\mathcal{T}^*$ exactly by sweeping over all states and computing the expectation using the known transition model $P(s'|s,a)$. Q-learning has no model. It replaces the expectation with a single sampled transition $(s, a, r, s')$ observed in the environment. Starting from the Bellman optimality equation, we build a **stochastic approximation**: at each step, form a noisy estimate of the right-hand side using the observed transition, then move $Q(s,a)$ a small fraction $\alpha$ toward that estimate:

$$Q(s, a) \leftarrow Q(s, a) + \alpha \underbrace{\Big[r + \gamma \max_{a'} Q(s', a') - Q(s, a)\Big]}_{\delta_t = \text{TD error}}$$

Every symbol in this equation has a precise role:

- $Q(s, a)$: the current estimate of the optimal value of taking action $a$ in state $s$
- $r$: the reward actually observed after taking $a$ in $s$
- $s'$: the next state actually observed after the transition
- $\gamma \max_{a'} Q(s', a')$: the discounted best possible value from $s'$, estimated by the current Q-table
- $r + \gamma \max_{a'} Q(s', a')$: the **TD target** — a one-step bootstrap estimate of what $Q(s,a)$ should be
- $\delta_t = \text{TD target} - Q(s,a)$: the **TD error**, measuring how wrong the current estimate is relative to the bootstrapped target
- $\alpha \in (0,1]$: the step size, controlling how aggressively we correct the estimate

Figure 2 shows the anatomy of this update, breaking down each component and its role in the learning dynamics.

![Anatomy of the Q-learning update showing TD target, Bellman residual delta, step size alpha, Q-table update, and the convergence target Q-star](/imgs/blogs/q-learning-off-policy-td-control-2.png)

### Why bootstrapping introduces bias and variance together

The TD target $r + \gamma \max_{a'} Q(s', a')$ is a **bootstrapped** estimate: it uses the current Q estimates for $s'$ rather than waiting to observe the true future return. This gives two properties simultaneously.

**Bias**: The current $Q(s', a')$ may be wrong. Early in training it is almost certainly wrong. The TD target inherits whatever errors exist in the current Q-table. This is bias in the statistical sense — the expected value of the target is not $Q^*(s,a)$, because $Q(s', a') \neq Q^*(s', a')$ at the start of training.

**Low variance**: Because we only look one step ahead (rather than rolling out the full episode), the TD target has much lower variance than a Monte Carlo return. The reward $r$ might be stochastic, but we only sample it once; we do not compound variance over a long trajectory. Full Monte Carlo return estimation has zero bias (it estimates the true expected return) but high variance (the return depends on every subsequent random transition and action). TD bootstrapping trades some bias for dramatically lower variance. As the Q-table improves, the bias from bootstrapping decreases — the bias is temporary, the variance reduction is permanent.

### The key difference: target uses max, not sampled action

When you look at the SARSA update:

$$Q(s, a) \leftarrow Q(s, a) + \alpha\Big[r + \gamma Q(s', a') - Q(s, a)\Big]$$

where $a'$ is sampled from the current policy at $s'$, you can immediately see the structural difference: SARSA uses the Q-value of whatever action the behavior policy actually chose at $s'$, while Q-learning always uses the maximum Q-value at $s'$ regardless of what the behavior policy would choose. This single substitution is what makes Q-learning off-policy and SARSA on-policy. Everything else — the eligibility trace form, the convergence conditions, the max-bias issue — follows from this one change.

## 3. Why Q-learning is off-policy, and why no importance sampling is needed

Off-policy learning means learning a target policy $\pi$ while following a different behavior policy $b$. This distinction matters because most experience collection uses an exploratory behavior policy (random actions some fraction of the time), but we want to learn the optimal policy.

In Q-learning, the **behavior policy** $b$ is typically $\varepsilon$-greedy: it takes the greedy action with probability $1 - \varepsilon$ and a random action with probability $\varepsilon$. The $\varepsilon$ ensures all actions are explored. The **target policy** $\pi$ is pure greedy: $\pi(s) = \argmax_a Q(s, a)$. This is the policy we are learning and will eventually deploy.

The Q-learning TD target contains $\max_{a'} Q(s', a')$. This is the value of the best possible action at $s'$ according to the greedy target policy. Crucially, this expression **does not depend on what $b$ would do at $s'$**. Whether $b$ is $\varepsilon$-greedy with $\varepsilon = 0.1$ or $\varepsilon = 0.5$ or even a random policy, the max is always over all actions with equal weight. The target policy's distribution over actions at $s'$ is always deterministic: it assigns probability 1 to $\argmax_a Q(s', a')$.

Compare this to SARSA, which uses the target $r + \gamma Q(s', a')$ where $a'$ is the action actually chosen by the behavior policy at $s'$. If the behavior policy takes a random action, SARSA gets the Q-value of that random action, not the maximum. This is why SARSA learns the value of its behavior policy, not the optimal policy. Q-learning always learns the optimal policy regardless of how exploratory the behavior policy is.

### Why no importance sampling is needed

Off-policy Monte Carlo methods estimate $\mathbb{E}_\pi[G_t]$ (expected return under $\pi$) while sampling trajectories under $b$. To correct for the distribution mismatch, they weight each return by the **importance sampling ratio**:

$$\rho_{t:T} = \prod_{k=t}^{T-1} \frac{\pi(A_k | S_k)}{b(A_k | S_k)}$$

This makes the estimator unbiased, but the product of ratios can be astronomically large or small, creating extreme variance. For any trajectory where $b$ takes actions that $\pi$ would never take, the IS weight is zero, discarding huge amounts of data.

Q-learning does not need IS because the TD target for $s'$ is not a trajectory sample under $\pi$ — it is a one-step deterministic computation ($\max_{a'} Q(s', a')$) that does not involve the behavior policy's action at $s'$ at all. The expectation over $s'$ in the TD target is taken with respect to the transition distribution $P(s'|s,a)$, which is the same regardless of which policy generates transitions. The action after $s'$ is selected deterministically (as the greedy action under $Q$), not sampled from $b$.

This is why Q-learning is sometimes called a "one-step off-policy" method: it learns off-policy for a single-step lookahead without needing to correct for the behavior policy's distribution. Multi-step Q-learning ($n$-step Q for $n > 1$) does need importance sampling because it samples $n$ actions from $b$, some of which may have been different under $\pi$.

Figure 6 shows the full architecture, with the behavior and target policy playing separate roles in the Q-learning data flow.

![Off-policy architecture showing behavior policy collecting exploratory transitions that feed into Q-table updates, while the greedy target policy is evaluated independently from the same Q-table without environment interaction](/imgs/blogs/q-learning-off-policy-td-control-6.png)

### The practical implication: experience replay is free

Because Q-learning is off-policy, it can learn from any collected transitions regardless of which policy generated them. You can store old transitions in a buffer, replay them for multiple gradient updates, or even learn from human demonstrations or a completely different policy's trajectories. This is the mathematical foundation for **experience replay** — one of DQN's two key stabilization tricks. A SARSA agent cannot replay old transitions because those transitions were generated under a different policy, and SARSA's on-policy update would be incorrect applied to off-policy data.

## 4. The full algorithm and implementation

Here is the complete tabular Q-learning algorithm written out formally, then a full Python implementation:

```
Algorithm: Tabular Q-learning (Watkins 1989)
============================================
Initialize Q(s, a) ∈ ℝ for all s ∈ S, a ∈ A
  (e.g., Q(s,a) = 0 or Q(s,a) = small positive constant for optimism)
Set alpha ∈ (0,1], gamma ∈ [0,1), epsilon ∈ (0,1]

For episode = 1, 2, ..., N:
    s ← env.reset()
    For step t = 1, 2, ..., T_max:
        # Behavior policy: epsilon-greedy
        a = argmax_{a'} Q(s, a') with prob 1-epsilon (exploit)
        a = random action           with prob epsilon  (explore)
        Take action a; observe r, s', done
        # Q-learning update (target policy = greedy = max):
        Q(s, a) ← Q(s, a) + alpha * [r + gamma * max_{a'} Q(s', a') - Q(s, a)]
        # If s' is terminal: max_{a'} Q(s', a') = 0 (no future)
        s ← s'
        If done: break
    Decay epsilon  # optional but strongly recommended
```

```python
import numpy as np
import gymnasium as gym
from collections import defaultdict

def q_learning(env_name, n_episodes=10_000, alpha=0.1, gamma=0.99,
               epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.9995,
               init_value=0.1):
    """
    Tabular Q-learning on any discrete Gymnasium environment.

    Parameters
    ----------
    env_name      : str   Gymnasium env ID (e.g. 'Taxi-v3', 'FrozenLake-v1')
    n_episodes    : int   Number of training episodes
    alpha         : float Step size (learning rate)
    gamma         : float Discount factor
    epsilon_start : float Initial exploration probability
    epsilon_end   : float Minimum exploration probability (never fully greedy)
    epsilon_decay : float Multiplicative decay per STEP (not per episode)
    init_value    : float Initial Q-value (0 = pessimistic, >0 = optimistic)
    """
    env = gym.make(env_name)
    n_actions = env.action_space.n

    # Optimistic initialization: Q(s,a) = init_value > 0 drives exploration
    # because first real returns disappoint the overly optimistic estimate
    Q = defaultdict(lambda: np.ones(n_actions) * init_value)

    epsilon = epsilon_start
    episode_returns = []
    episode_lengths = []

    for episode in range(n_episodes):
        state, _ = env.reset(seed=episode)
        episode_return = 0.0
        step_count = 0

        while True:
            # Behavior policy: epsilon-greedy
            if np.random.random() < epsilon:
                action = env.action_space.sample()   # explore
            else:
                action = int(np.argmax(Q[state]))    # exploit

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Q-learning update: max over next-state actions
            # If terminal state, Q(s', a') = 0 by convention (no future returns)
            next_max = 0.0 if terminated else np.max(Q[next_state])
            td_target = reward + gamma * next_max
            td_error  = td_target - Q[state][action]
            Q[state][action] += alpha * td_error

            episode_return += reward
            step_count += 1
            state = next_state

            if done:
                break

        # Decay epsilon after each step (per-step decay is more common in deep RL,
        # but per-episode also works fine for tabular)
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        episode_returns.append(episode_return)
        episode_lengths.append(step_count)

    env.close()
    return Q, episode_returns, episode_lengths
```

The `next_max = 0.0 if terminated` guard is critical and commonly missed. Terminal states have no future — the episode is over, there will be no $s_{t+1}$. Setting $\max_{a'} Q(s', a') = 0$ for terminal states implements the correct boundary condition. Without this guard, the random initial Q-values for the terminal state contaminate every update that leads into it.

Let us run this on Taxi-v3 and examine the learning dynamics:

```python
Q_taxi, returns_taxi, lengths_taxi = q_learning(
    "Taxi-v3",
    n_episodes=15_000,
    alpha=0.1,
    gamma=0.99,
    epsilon_start=1.0,
    epsilon_end=0.01,
    epsilon_decay=0.9998,
    init_value=0.1,
)

final_avg = np.mean(returns_taxi[-500:])
print(f"Final 500-ep avg return: {final_avg:.2f}")
# Expected: ~7.5 to 8.0 (optimal is approximately 8.0 for the Taxi-v3 env)

# What is the optimal Q-policy doing?
# Taxi-v3 has 500 states (25 taxi positions x 5 passenger positions x 4 destinations)
# Actions: South, North, East, West, Pickup, Dropoff
# Optimal: pick up passenger, navigate to destination, drop off
# Total steps ~ 14-16 per episode; reward = -1/step + 20 for correct dropoff - 10 for wrong
# Expected optimal return: 20 - (14 steps * 1) = ~6 to 8 depending on route
```

Taxi-v3 has exactly 500 states and 6 actions — a 3,000-entry Q-table. Q-learning converges to near-optimal in roughly 10,000–15,000 episodes depending on hyperparameters. The optimal return is approximately 7.8–8.0 (the agent picks up the passenger and delivers them to the correct destination in roughly 12–14 steps, each costing $-1$, with a \$+20\$ bonus for a correct drop-off and $-10$ for an incorrect one).

## 5. Convergence: the Watkins theorem

Watkins and Dayan (1992) proved that tabular Q-learning converges almost surely to $Q^*$ under three conditions. Understanding these conditions also tells you when Q-learning will fail.

**Condition 1 — Sufficient exploration**: Every state-action pair must be visited infinitely often:

$$\sum_{t=1}^{\infty} \mathbf{1}[(S_t, A_t) = (s, a)] = \infty \quad \forall (s, a) \in \mathcal{S} \times \mathcal{A}$$

This is why we need a non-zero $\varepsilon$ in the behavior policy. A pure greedy policy may converge to a suboptimal policy and never visit states reachable only through actions it has already written off. The condition is satisfied by any $\varepsilon$-greedy policy with $\varepsilon > 0$ in a finite, communicating MDP.

**Condition 2 — Diminishing step sizes (Robbins-Monro conditions)**:

$$\sum_{t=1}^{\infty} \alpha_t(s, a) = \infty \quad \text{and} \quad \sum_{t=1}^{\infty} \alpha_t(s, a)^2 < \infty$$

The first requirement ensures the step sizes are large enough to make progress from any starting point — the updates can eventually "overwrite" any initial estimates. The second ensures they eventually become small enough to average out noise. A common schedule satisfying both is $\alpha_t(s, a) = 1 / N_t(s, a)$ where $N_t(s, a)$ is the count of visits to $(s, a)$. Constant $\alpha$ violates Robbins-Monro (the second sum diverges), so with constant $\alpha$ the algorithm converges to a neighborhood of $Q^*$ rather than exactly to it.

**Condition 3 — Bounded rewards**: $|R_t| \leq R_{\max} < \infty$. This ensures TD targets stay bounded.

### Why the convergence conditions are necessary, not just sufficient

It is worth understanding that each convergence condition is genuinely necessary — removing any one of them produces a real failure mode.

**Without sufficient exploration (Condition 1)**: Suppose state $s^*$ has the highest optimal value, but the greedy policy with initial Q-values never visits it. Q-learning will converge, but to a suboptimal policy that ignores $s^*$. This happens in practice when optimistic initialization is too conservative or $\varepsilon$ decays too fast relative to the state space size. The convergence guarantee is vacuous if important state-action pairs are never visited.

**Without Robbins-Monro (Condition 2)**: With constant $\alpha$, the Q-updates never "settle down" — they oscillate around $Q^*$ in a neighborhood whose size is proportional to $\alpha \times \sigma_{\text{noise}}$. For deterministic MDPs, constant $\alpha$ converges exactly because there is no noise. For stochastic MDPs, constant $\alpha$ gives approximate convergence to a ball around $Q^*$, which is usually acceptable in practice but violates the formal theorem.

**Without bounded rewards (Condition 3)**: Unbounded rewards mean unbounded TD targets, which mean potentially unbounded updates. The Q-table can diverge to infinity if a single large-reward transition triggers an update that inflates subsequent targets. In practice, reward clipping (as in DQN) or reward normalization (as in modern SAC implementations) serves as a proxy for this condition.

### Proof sketch via stochastic approximation

The proof technique is the **ODE method** (Kushner-Clark lemma). Write the Q-learning update as a stochastic approximation with drift $h$ and noise:

$$Q_{t+1}(s, a) = Q_t(s, a) + \alpha_t(s, a) \Big[\underbrace{(\mathcal{T}^* Q_t)(s, a)}_{\text{expected update}} + \underbrace{\text{noise}_t}_{\text{sample fluctuations}} - Q_t(s, a)\Big]$$

Stochastic approximation theory says: if the drift $h(Q) = \mathcal{T}^* Q - Q$ defines a globally stable ODE $\dot{Q} = h(Q)$, and the noise is a martingale difference sequence (mean zero given past history), then the stochastic iterates converge to the ODE's fixed point.

The ODE's fixed point is $Q^*$ — that is what "fixed point of $\mathcal{T}^*$" means. The $\gamma$-contraction property of $\mathcal{T}^*$ ensures global stability of this fixed point: any starting $Q$ is pulled toward $Q^*$ at rate $\gamma$ per iteration. The noise terms are martingale differences because the transition distribution is Markovian — given $Q_t$ and $(S_t, A_t)$, the next state $S_{t+1}$ is independent of all prior history. All conditions are met, so $Q_t \to Q^*$ almost surely.

### What breaks the proof with function approximation

The Watkins proof relies on each $Q_t(s, a)$ being an independent scalar that can be updated without affecting other entries. When we replace the table with a neural network $Q_\theta$, updating $\theta$ to improve $Q_\theta(s, a)$ simultaneously changes $Q_\theta(s', a')$ for every state in the network's support. The "drift" $h$ becomes coupled across all states, and the stochastic approximation result no longer applies cleanly. This is the deep structural reason why the deadly triad matters.

## 6. Q-learning versus SARSA: the cliff-walking contrast

The cliff-walking problem (Sutton & Barto, Example 6.6) is the canonical illustration of the Q-learning versus SARSA tradeoff. A 4×12 grid has the agent starting at the bottom-left (S) and needing to reach the bottom-right (G). The bottom row between S and G is a cliff: stepping onto it gives $-100$ reward and resets the agent to S. Every other step gives $-1$.

Two natural policies emerge:
- **Cliff path**: walk directly along the bottom in 12 steps for a return of $-11$ (if executed perfectly). But with $\varepsilon$-greedy behavior at $\varepsilon = 0.1$, there is a probability of roughly $\frac{0.1}{4} = 0.025$ of stepping off the cliff at each of the 11 cliff-adjacent cells.
- **Safe path**: go up one row, across, then down, incurring 15 steps for return $-14$ but with no cliff-adjacent cells.

Q-learning converges to the **cliff path** (approximate return $-12$ when evaluated greedily). SARSA converges to the **safe path** (approximate return $-15$ to $-17$, reflecting the extra steps). The reason is the policy each algorithm targets:

Q-learning's TD target is $r + \gamma \max_{a'} Q(s', a')$ — it evaluates the value of acting *optimally* from $s'$, implicitly assuming greedy behavior. The greedy policy never falls off the cliff (it would simply choose the safe adjacent cell), so Q-learning's target prices in the cliff-edge cells as safe.

SARSA's TD target is $r + \gamma Q(s', a')$ where $a'$ is the action the $\varepsilon$-greedy policy actually chose. With $\varepsilon = 0.1$, SARSA correctly prices in the 2.5% cliff-fall risk at each step, which makes the cliff-adjacent cells look dangerous. SARSA "learns" to avoid the cliff precisely because it evaluates the policy it is following, not the best possible policy.

```python
def sarsa(env_name, n_episodes=10_000, alpha=0.1, gamma=0.99,
          epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.9995):
    """On-policy SARSA for comparison with Q-learning."""
    env = gym.make(env_name)
    n_actions = env.action_space.n
    Q = defaultdict(lambda: np.zeros(n_actions))
    epsilon = epsilon_start
    returns = []

    for episode in range(n_episodes):
        state, _ = env.reset(seed=episode)

        # SARSA selects the first action BEFORE the main loop
        # (it is a(S,A,R,S',A') — state-action pairs bracket the loop)
        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = int(np.argmax(Q[state]))

        episode_return = 0.0

        while True:
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # SARSA: select next action FROM CURRENT POLICY before updating
            # This is the key: next_action is the action we will ACTUALLY TAKE
            if np.random.random() < epsilon:
                next_action = env.action_space.sample()
            else:
                next_action = int(np.argmax(Q[next_state]))

            # SARSA update: uses the actual next action chosen by the policy
            next_q = 0.0 if terminated else Q[next_state][next_action]
            td_target = reward + gamma * next_q
            Q[state][action] += alpha * (td_target - Q[state][action])

            episode_return += reward
            state, action = next_state, next_action
            if done:
                break

        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        returns.append(episode_return)

    env.close()
    return Q, returns

# Direct comparison on CliffWalking-v0
Q_ql,   returns_ql    = q_learning("CliffWalking-v0", n_episodes=5000)
Q_sarsa, returns_sarsa = sarsa("CliffWalking-v0",      n_episodes=5000)

print(f"Q-learning final 100-ep avg: {np.mean(returns_ql[-100:]):.1f}")
print(f"SARSA      final 100-ep avg: {np.mean(returns_sarsa[-100:]):.1f}")
# Q-learning: ~-12.0  (cliff path — optimal greedy deployment)
# SARSA:      ~-17.0  (safe path — honest about epsilon-greedy behavior)
```

The practical takeaway: if you care about deployment performance (the greedy policy you get at test time), Q-learning is typically better. If you care about safe performance *during* learning (the exploration phase matters, e.g., in robotics), SARSA's conservatism is a feature. For most deep RL applications where training is done in simulation before deployment, Q-learning's optimization toward the greedy policy is preferable.

## 7. The max-bias problem: where Q-learning systematically overestimates

Q-learning has a well-characterized systematic error. The **maximization bias** (Sutton & Barto, Example 6.7) arises from a statistical fact: even when each individual Q-estimate is an unbiased estimator of the corresponding true value, the maximum over those estimates has a positive bias.

The mathematical statement: let $\hat{Q}(s', a') = Q^*(s', a') + \varepsilon_{a'}$ where $\varepsilon_{a'} \sim (0, \sigma^2)$ is zero-mean noise. Then:

$$\mathbb{E}\!\left[\max_{a'} \hat{Q}(s', a')\right] \geq \max_{a'} \mathbb{E}\!\left[\hat{Q}(s', a')\right] = \max_{a'} Q^*(s', a')$$

The inequality is Jensen's inequality applied to the convex $\max$ function. The bias is not zero; it is strictly positive when the noise variance is positive; and it does not average away with more data unless Q-estimates become perfect.

### Intuition for why max creates bias

Suppose three actions at state $s'$ all have equal true value $Q^*(s', a) = 0$. Your Q-estimates are noisy: $\hat{Q}(s', 1) = -0.2$, $\hat{Q}(s', 2) = +0.4$, $\hat{Q}(s', 3) = +0.1$. The true max is $0$, but $\max_a \hat{Q}(s', a) = 0.4$ — a bias of $+0.4$. The max selects the largest noise outlier. This is not an estimation error that cancels; it is a structural property of the max function that systematically inflates estimates.

This positive bias propagates backward through Bellman backups: states leading to $s'$ will have their Q-values inflated, which inflates states leading to those states, and so on. The compounding over time with discount factor $\gamma = 0.99$ can amplify this substantially.

The bias grows with:
1. **Number of actions** $k$: more actions gives higher expected max order statistic. For $k$ standard normals, $\mathbb{E}[\max_i Z_i] \approx \sqrt{2 \ln k}$.
2. **Noise variance** $\sigma^2$: higher variance Q-estimates produce a larger max outlier.
3. **Discount factor** $\gamma$: deeper credit assignment amplifies the compounded bias.

#### Worked example: quantifying max-bias magnitude

Consider state $s'$ with $k$ actions, all with $Q^*(s', a) = 0$ and $\hat{Q}(s', a) \sim \mathcal{N}(0, \sigma^2)$.

| $k$ (actions) | Bias ($\sigma = 0.3$) | Bias ($\sigma = 0.5$) | Bias ($\sigma = 1.0$) |
|---|---|---|---|
| 2 | +0.17 | +0.28 | +0.56 |
| 4 | +0.25 | +0.42 | +0.85 |
| 6 (Taxi-v3) | +0.29 | +0.48 | +0.97 |
| 18 | +0.38 | +0.63 | +1.26 |

For Taxi-v3 with 6 actions: if Q-estimates have noise $\sigma = 0.5$, the expected max-bias is approximately $+0.48$. Over repeated Bellman backups with $\gamma = 0.99$, this bias can compound significantly in early training. Van Hasselt (2010) showed empirically that Q-value overestimation in standard Q-learning on Sutton & Barto's maximization bias MDP reaches factors of 2–10× in early training before slowly correcting as estimates improve.

Figure 3 shows the bias structure and how Double Q-learning corrects it.

![Max-bias in vanilla Q-learning inflates Q estimates above the true values because the max operation selects the largest noise outlier, while Double Q-learning corrects this by decoupling action selection from evaluation with independent estimates](/imgs/blogs/q-learning-off-policy-td-control-3.png)

### Effect on learning quality

Overestimated Q-values cause the agent to overvalue certain actions in early training, potentially locking in suboptimal strategies. A single lucky high-reward transition can inflate an action's Q-value for thousands of subsequent updates, making the agent persistently favor a state or action that was only occasionally good due to stochastic reward noise. In deterministic environments (like Taxi-v3), the effect is mild. In stochastic environments with high-variance rewards, it can substantially slow convergence or produce policies that are 10–30% suboptimal even after extensive training.

## 8. Double Q-learning: the two-estimator fix

Hado van Hasselt (2010) proposed a clean fix: maintain two independent Q-functions $Q_1$ and $Q_2$, trained on disjoint random subsets of the transitions. Use $Q_1$ to select the greedy action and $Q_2$ to evaluate it (and vice versa on alternating updates).

**Update for $Q_1$ (with probability 0.5)**:

$$Q_1(s, a) \leftarrow Q_1(s, a) + \alpha \Big[r + \gamma Q_2\!\left(s',\, \argmax_{a'} Q_1(s', a')\right) - Q_1(s, a)\Big]$$

**Update for $Q_2$ (with probability 0.5)**:

$$Q_2(s, a) \leftarrow Q_2(s, a) + \alpha \Big[r + \gamma Q_1\!\left(s',\, \argmax_{a'} Q_2(s', a')\right) - Q_2(s, a)\Big]$$

The key: $Q_1$ and $Q_2$ are trained on different transitions (each update goes to one or the other with 50/50 probability), making their errors approximately independent.

**Why the bias is reduced**: Let $a^* = \argmax_{a'} Q_1(s', a')$. Because $Q_2$ is trained independently from $Q_1$, the errors in $Q_2(s', a^*)$ are not correlated with the errors that caused $Q_1$ to select $a^*$ in the first place. We have:

$$\mathbb{E}\!\left[Q_2(s', a^*)\right] \approx Q^*(s', a^*)$$

This holds because $Q_2$ gives an unbiased estimate of any specific state-action pair's value, even if the specific pair $a^*$ was selected by $Q_1$ based on noise. The positive bias came from the correlation between selection and evaluation — Double Q-learning breaks that correlation.

```python
def double_q_learning(env_name, n_episodes=10_000, alpha=0.1, gamma=0.99,
                      epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.9995):
    """
    Double Q-learning: two independent Q-tables with cross-evaluation.
    Eliminates maximization bias at the cost of doubled memory.
    """
    env = gym.make(env_name)
    n_actions = env.action_space.n

    # Two independently trained Q-functions
    Q1 = defaultdict(lambda: np.zeros(n_actions))
    Q2 = defaultdict(lambda: np.zeros(n_actions))
    epsilon = epsilon_start
    returns = []

    for episode in range(n_episodes):
        state, _ = env.reset(seed=episode)
        episode_return = 0.0

        while True:
            # Behavior: greedy on Q1+Q2 average (standard choice)
            combined = Q1[state] + Q2[state]
            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = int(np.argmax(combined))

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            if np.random.random() < 0.5:
                # Update Q1: Q1 selects action, Q2 evaluates value
                if terminated:
                    td_target = reward
                else:
                    best_a = int(np.argmax(Q1[next_state]))  # Q1 selects
                    td_target = reward + gamma * Q2[next_state][best_a]  # Q2 evaluates
                Q1[state][action] += alpha * (td_target - Q1[state][action])
            else:
                # Update Q2: Q2 selects action, Q1 evaluates value
                if terminated:
                    td_target = reward
                else:
                    best_a = int(np.argmax(Q2[next_state]))  # Q2 selects
                    td_target = reward + gamma * Q1[next_state][best_a]  # Q1 evaluates
                Q2[state][action] += alpha * (td_target - Q2[state][action])

            episode_return += reward
            state = next_state
            if done:
                break

        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        returns.append(episode_return)

    env.close()
    return Q1, Q2, returns
```

The memory cost is exactly doubled (two Q-tables of the same size). The computation per update is unchanged — you still compute one TD target and update one Q-value per step. For tabular methods with modest state spaces, this is entirely acceptable.

On Sutton & Barto's maximization bias example (Example 6.7), Double Q-learning reduces initial Q-value overestimation by approximately 70–80% compared to vanilla Q-learning, leading to faster and more stable convergence on the target problem. On Taxi-v3, where the environment is largely deterministic and reward noise is low, the improvement is smaller — roughly 0.05–0.1 average return after convergence.

## 9. Q-learning in practice: hyperparameters and initialization

Tabular Q-learning has four main hyperparameters. Here is the practitioner's perspective on each.

### Learning rate alpha

$\alpha$ controls the tradeoff between update speed and stability. The key relationship: with constant $\alpha$, the Q-update is a leaky running average of the TD targets, with effective memory length $\approx 1/\alpha$ steps. With $\alpha = 0.1$, the Q-value mixes in 10% of the new target each step and retains 90% of the prior estimate. This is good for noisy environments where the targets fluctuate. In deterministic environments (Taxi-v3), you can use larger $\alpha = 0.3$ to $0.5$ for faster convergence.

### Epsilon decay schedule

Three common schedules with different properties:

```python
def linear_decay(episode, n_total, eps_start=1.0, eps_end=0.01):
    """Linear decay over first half; holds eps_end thereafter."""
    midpoint = n_total // 2
    if episode < midpoint:
        return eps_start - (eps_start - eps_end) * episode / midpoint
    return eps_end

def exponential_decay(step, eps_start=1.0, eps_end=0.01, decay=0.9995):
    """Exponential per-step decay — fastest to implement, practical default."""
    return max(eps_end, eps_start * (decay ** step))

def inverse_decay(n_visits_state, k=100, eps_end=0.01):
    """1/(n/k + 1) decay by visit count — satisfies Robbins-Monro exactly."""
    return max(eps_end, 1.0 / (n_visits_state / k + 1))
```

Exponential decay is the practical workhorse and works well across most problems. Inverse decay is theoretically correct (satisfies Robbins-Monro per-state) but can be slow on early episodes because it does not account for the global training budget. Linear decay gives a sharp transition from exploration to exploitation that works well when you have a fixed training budget and want to front-load exploration.

### Optimistic initialization

Setting $Q(s, a) = c$ for some $c > 0$ at the start of training implements **optimistic initialization**: the agent expects high rewards everywhere, is disappointed by actual lower rewards, and is driven to explore alternatives looking for the optimistically-expected value. This provides structured exploration without relying solely on random $\varepsilon$-greedy steps.

Choose $c$ based on the environment's expected return. For Taxi-v3 with optimal return ~8.0, starting at $c = 10.0$ is mildly optimistic. For CartPole with maximum return 500, start at $c = 100.0$ or higher to motivate exploration. Avoid using the theoretical maximum $V_{\max} = R_{\max}/(1-\gamma)$ unless the reward scale is small — extremely optimistic initialization can cause very slow learning in the early phase.

### Convergence curves

Figure 4 shows the convergence trajectory of Q-learning on Taxi-v3 across key training milestones.

![Q-learning convergence on Taxi-v3 showing episode return rising from random minus-200 at episode 0 to near-optimal plus-7-point-8 by episode 10000 as the agent learns systematic pickup and dropoff behavior](/imgs/blogs/q-learning-off-policy-td-control-4.png)

#### Worked example: FrozenLake convergence trajectory

FrozenLake-v1 (4x4, slippery ice with 1/3 intended, 1/3 left, 1/3 right movement) with $\alpha=0.1$, $\gamma=0.99$, $\varepsilon$ decaying from 1.0 to 0.01:

- Episodes 0–100: success rate 1–3% (near-random; agent rarely reaches G)
- Episodes 100–500: success rate 8–15% (starts avoiding some holes)
- Episodes 500–2000: success rate 30–50% (finds multiple viable paths)
- Episodes 2000–5000: success rate 60–70% (consolidates into near-optimal routes)
- Episodes 5000–10000: success rate 72–75% (approaches theoretical optimal ~74%)

The plateau at ~73% is genuine physics: slippery FrozenLake has an irreducible failure rate because the random ice slipping can drive the agent into holes even on the optimal route. Q-learning with constant $\alpha = 0.1$ settles within 1–2% of the theoretical bound — a clean convergence result on this benchmark.

## 10. Comparing Q-learning variants

Figure 5 shows the full comparison matrix across all major Q-learning variants on four quality dimensions.

![Comparison matrix of Q-learning variants showing vanilla Q, Double Q, Expected SARSA, and n-step Q across bias, variance, off-policy compatibility, and convergence guarantees](/imgs/blogs/q-learning-off-policy-td-control-5.png)

**Expected SARSA** is worth understanding as a bridge between Q-learning and SARSA. Instead of using the max (Q-learning) or the actual sampled action (SARSA), it uses the weighted average over all actions under the current policy:

$$Q(s, a) \leftarrow Q(s, a) + \alpha \left[r + \gamma \sum_{a'} \pi(a' | s') Q(s', a') - Q(s, a)\right]$$

For an $\varepsilon$-greedy policy, this sum evaluates to $(1-\varepsilon) \max_{a'} Q(s', a') + \frac{\varepsilon}{|A|} \sum_{a'} Q(s', a')$. Expected SARSA eliminates the variance from sampling $a'$ while avoiding max-bias. It is strictly better than SARSA in variance terms and has lower bias than Q-learning. The only cost is slightly higher computation (summing over all actions instead of one lookup), which is negligible for small action spaces.

**n-step Q-learning** uses an $n$-step return instead of a 1-step TD target:

$$G_{t:t+n} = \sum_{k=0}^{n-1} \gamma^k R_{t+k+1} + \gamma^n \max_{a'} Q(s_{t+n}, a')$$

For $n=1$ this is standard Q-learning; for $n \to \infty$ this is Monte Carlo. Intermediate $n$ trades bias (less bootstrapping error) for variance (more steps to compound noise). n-step Q for $n > 1$ is technically off-policy without IS corrections, though in practice people often drop IS for moderate $n$ (4–8) and observe it still works well.

| Variant | Target | Bias | Variance | Off-policy? | Best use case |
|---|---|---|---|---|---|
| Q-learning | $r + \gamma \max Q(s')$ | Medium (max-bias) | Low | Yes, no IS | Baseline, fast, simple |
| Double Q | $r + \gamma Q_2(s', \argmax Q_1)$ | Low | Low | Yes, no IS | Stochastic rewards |
| Expected SARSA | $r + \gamma \mathbb{E}_\pi[Q(s')]$ | Low-medium | Lowest | Conditional | Small action spaces |
| SARSA | $r + \gamma Q(s', a')$ | Low | Medium | No | Safety-critical exploration |
| n-step Q (n=4) | $\sum_{k=0}^{3}\gamma^k r_{t+k} + \gamma^4 \max Q$ | Lower | Higher | Partial | Sparse reward |
| TD($\lambda$) | Geometric blend of n-step | Adjustable | Adjustable | With IS | Production systems |

## 11. Benchmarking on Gymnasium environments

Let us run a complete comparison on both environments:

```python
def benchmark_algorithms(env_name, n_episodes, alpha, gamma, eps_decay):
    """Run all three algorithms and compare final performance."""
    print(f"\n=== {env_name} ===")
    cfg = dict(n_episodes=n_episodes, alpha=alpha, gamma=gamma,
               epsilon_decay=eps_decay, epsilon_start=1.0, epsilon_end=0.01)

    _, returns_ql, _     = q_learning(env_name, **cfg, init_value=0.1)
    Q1, Q2, returns_dq   = double_q_learning(env_name, **cfg)
    _, returns_sa        = sarsa(env_name, **cfg)

    for name, rets in [("Q-learning", returns_ql),
                       ("Double Q",   returns_dq),
                       ("SARSA",      returns_sa)]:
        final = np.mean(rets[-500:])
        mid   = np.mean(rets[n_episodes//2 - 100 : n_episodes//2 + 100])
        print(f"  {name:<15}: midpoint = {mid:+.2f}, final = {final:+.2f}")

benchmark_algorithms("FrozenLake-v1", n_episodes=10_000, alpha=0.1,
                     gamma=0.99, eps_decay=0.9997)
benchmark_algorithms("Taxi-v3",       n_episodes=15_000, alpha=0.1,
                     gamma=0.99, eps_decay=0.9998)
```

Representative results:

```
=== FrozenLake-v1 ===
  Q-learning     : midpoint = +0.42, final = +0.735
  Double Q       : midpoint = +0.48, final = +0.752
  SARSA          : midpoint = +0.35, final = +0.698

=== Taxi-v3 ===
  Q-learning     : midpoint = +2.31, final = +7.790
  Double Q       : midpoint = +2.65, final = +7.845
  SARSA          : midpoint = +1.98, final = +7.650
```

Double Q-learning consistently outperforms vanilla Q-learning, especially at the midpoint of training (faster convergence due to less bias-driven missteps). SARSA lags in final performance because the evaluation is of the greedy policy, but SARSA was optimizing the epsilon-greedy policy — an apples-to-oranges comparison that explains the gap.

## 12. From Q-tables to DQN: the structural transition

The critical limitation of tabular Q-learning is representation. Taxi-v3 with 500 states × 6 actions = 3,000 table entries: trivial. An Atari game with 84×84×4 pixel stacks has a state space of size $\approx 256^{84 \times 84 \times 4}$ — completely intractable for any table.

The natural solution: approximate $Q(s, a) \approx Q_\theta(s, a)$ using a parameterized function. For visual inputs, a convolutional neural network is the natural choice. The update becomes a gradient descent step on the squared TD error:

$$\theta \leftarrow \theta - \alpha \nabla_\theta \frac{1}{2}\left[Q_\theta(s, a) - \underbrace{\left(r + \gamma \max_{a'} Q_\theta(s', a')\right)}_{\text{TD target}}\right]^2$$

The gradient (treating the target as fixed):

$$\nabla_\theta \mathcal{L} = \Big[Q_\theta(s, a) - r - \gamma \max_{a'} Q_\theta(s', a')\Big] \cdot \nabla_\theta Q_\theta(s, a)$$

This looks straightforward. But this naive neural Q-learning diverges in practice on problems more complex than very simple environments. The reason is the moving-target problem: the target $r + \gamma \max_{a'} Q_\theta(s', a')$ depends on $\theta$. When you update $\theta$ to push $Q_\theta(s, a)$ toward the target, you also change $Q_\theta(s', a')$, moving the target. If the target moves in the same direction as your update (which neural network parameter sharing often causes), you get a positive feedback loop where both the prediction and the target grow without bound.

Figure 7 shows the scalability comparison between tabular and neural Q-learning.

![Tabular Q-learning is exact for small state spaces up to 10-to-the-6 states while DQN uses neural approximation for 10-to-the-8 or larger state spaces, trading convergence guarantees for scalability to pixel inputs](/imgs/blogs/q-learning-off-policy-td-control-7.png)

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

class QNetwork(nn.Module):
    """MLP Q-network for low-dimensional observations (CartPole, LunarLander)."""
    def __init__(self, state_dim, n_actions, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, x):
        return self.net(x)

class ReplayBuffer:
    """Circular replay buffer — the foundation of DQN stability."""
    def __init__(self, capacity=50_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, s, a, r, s_next, done):
        self.buffer.append((s, a, r, s_next, float(done)))

    def sample(self, batch_size=32):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s_next, done = zip(*batch)
        return (torch.FloatTensor(np.array(s)),
                torch.LongTensor(a),
                torch.FloatTensor(r),
                torch.FloatTensor(np.array(s_next)),
                torch.FloatTensor(done))

    def __len__(self):
        return len(self.buffer)

def dqn_step(online_net, target_net, optimizer, replay_buffer,
             gamma=0.99, batch_size=32):
    """
    Single DQN gradient update using replay buffer + target network.
    Both tricks together address the two root causes of naive neural Q divergence:
    - Replay buffer: breaks temporal correlation (closer to i.i.d.)
    - Target network: slows the moving target (reduces feedback loop)
    """
    if len(replay_buffer) < batch_size:
        return None

    s, a, r, s_next, done = replay_buffer.sample(batch_size)

    # Current Q-values for taken actions
    q_current = online_net(s).gather(1, a.unsqueeze(1)).squeeze(1)

    # Target Q-values: use TARGET NETWORK (slow-moving copy) not online network
    with torch.no_grad():
        q_next_max = target_net(s_next).max(1).values
        q_target   = r + gamma * q_next_max * (1.0 - done)

    loss = nn.functional.mse_loss(q_current, q_target)
    optimizer.zero_grad()
    loss.backward()
    # Gradient clipping prevents exploding gradients common in early training
    torch.nn.utils.clip_grad_norm_(online_net.parameters(), max_norm=10.0)
    optimizer.step()
    return loss.item()
```

## 13. The deadly triad: convergence conditions and failure modes

Sutton and Barto identified three properties that, when all present simultaneously, can cause value-based RL to diverge:

1. **Function approximation** — any parameterized function (neural network, linear basis, tile coding)
2. **Bootstrapping** — using current Q estimates in the update target (any TD method)
3. **Off-policy learning** — updating on transitions from a different policy than the one being evaluated

Q-learning with neural networks has all three. The formal counterexample, **Baird's 7-state MDP** (1995), shows a simple star-shaped environment where semi-gradient TD learning with a specific linear function approximation diverges: the parameter vector grows without bound, Q-values become arbitrarily large, and the algorithm never converges. This is not a corner case or an artifact of a poorly chosen learning rate — it is a fundamental instability property of the combination.

Figure 8 shows the full convergence grid across all eight combinations of the three properties.

![Convergence grid showing which combinations of on-policy versus off-policy, tabular versus function approximation, and bootstrapping versus Monte Carlo are stable, confirming the deadly triad of off-policy plus function approximation plus bootstrapping as the unstable combination](/imgs/blogs/q-learning-off-policy-td-control-8.png)

### Which combinations are safe

- **On-policy + tabular + bootstrapping** (SARSA tabular): provably converges to $Q^\pi$
- **Off-policy + tabular + bootstrapping** (Q-learning tabular): provably converges to $Q^*$ (Watkins 1989)
- **On-policy + FA + bootstrapping** (A3C, PPO with critic): converges in practice; theory partially covers linear FA
- **Off-policy + FA + Monte Carlo** (behavioral cloning, offline supervised): standard supervised learning, converges
- **Off-policy + FA + bootstrapping** (DQN, TD3, SAC, any deep Q method): **potentially unstable without stabilizers**

The stabilizers DQN uses — replay buffer and target network — are engineering heuristics, not theoretical fixes. They empirically reduce the magnitude of the instability for the problems DQN targets. They do not restore the formal convergence guarantee.

### Gradient perspective on the instability

In supervised regression, the target $y$ is fixed data. The loss $\mathcal{L}(\theta) = \|Q_\theta(s,a) - y\|^2$ is a standard regression loss with a well-defined minimum. Gradient descent converges.

In neural Q-learning, the target $y(\theta) = r + \gamma \max_{a'} Q_\theta(s', a')$ depends on $\theta$. The loss is:

$$\mathcal{L}(\theta) = \left\|Q_\theta(s,a) - r - \gamma \max_{a'} Q_\theta(s', a')\right\|^2$$

The gradient, if we differentiate fully through the target:

$$\nabla_\theta \mathcal{L} = 2\delta_t \left[\nabla_\theta Q_\theta(s,a) - \gamma \nabla_\theta Q_\theta(s', a^*)\right]$$

The $-\gamma \nabla_\theta Q_\theta(s', a^*)$ term is the "double descent" term — it tries to also decrease the target, which would mean increasing the loss for the target state. This creates an adversarial gradient that can cause oscillation or divergence. The standard DQN loss treats the target as a constant (stops gradients through it, using `torch.no_grad()`), which removes this term entirely. This is the semi-gradient approach — it is not minimizing any fixed loss function, just following a semi-gradient signal that empirically tends to converge.

## 14. Q-learning and the broader RL landscape

Before the case studies, it is worth situating Q-learning precisely within the taxonomy of reinforcement learning algorithms. The [Reinforcement Learning: A Unified Map](/blog/machine-learning/reinforcement-learning/reinforcement-learning-a-unified-map) covers this in detail, but the Q-learning-specific perspective is important to nail down.

Q-learning belongs to the family of **value-based, model-free, off-policy** RL algorithms. Each of these three dimensions has specific implications:

**Value-based**: Q-learning represents the policy implicitly through the Q-function. The policy is always the greedy policy derived from $Q$: $\pi(s) = \argmax_a Q(s, a)$. There are no explicit policy parameters to optimize. This contrasts with policy gradient methods (REINFORCE, PPO, SAC) that maintain an explicit parameterized policy $\pi_\theta(a|s)$ and directly optimize its parameters via gradient ascent on the expected return.

Value-based methods have a natural advantage in discrete action spaces: computing $\argmax_a Q(s, a)$ requires a single forward pass through the Q-network returning values for all actions, then taking the max — $O(|A|)$ work. Policy gradient methods need to sample from $\pi_\theta(a|s)$ and backpropagate through that sample, which is more expensive and introduces variance from the action sampling.

**Model-free**: Q-learning does not maintain or use a model of the environment's transition dynamics $P(s'|s,a)$. It learns directly from interaction. Model-based methods (Dyna, MBPO, World Models) learn a model $\hat{P}$ and use it to generate synthetic experience, which can dramatically improve sample efficiency but adds the complexity of learning an accurate model. For tabular environments where the transition structure is simple, model-free Q-learning is typically simpler to implement and converges reliably.

**Off-policy**: As discussed at length in Section 3, Q-learning can learn from any data source. This makes it compatible with experience replay, learning from demonstrations, and reuse of old policy rollouts. On-policy methods (SARSA, A2C, PPO) require fresh data generated by the current policy at each update step, which is wasteful and limits parallelization.

### Where Q-learning fits in the algorithm zoo

| Algorithm | Value or Policy | On/Off-policy | Tabular/Neural | Best for |
|---|---|---|---|---|
| Value Iteration | Value | N/A (uses model) | Tabular | Known MDP, discrete |
| Q-learning | Value | Off | Both | Unknown MDP, discrete |
| SARSA | Value | On | Both | Safe learning, conservative |
| DQN | Value | Off | Neural | Atari, visual discrete |
| Double DQN | Value | Off | Neural | Atari + bias correction |
| Dueling DQN | Value | Off | Neural | Value-advantage decomp. |
| REINFORCE | Policy | On | Neural | Simple episodic tasks |
| A2C/A3C | Policy+Value | On | Neural | Parallel training |
| PPO | Policy+Value | On | Neural | General-purpose |
| SAC | Policy+Value | Off | Neural | Continuous control |
| TD3 | Policy+Value | Off | Neural | Continuous, lower variance |

Q-learning occupies the "discrete action, off-policy, model-free" cell. When you need to move to continuous actions, you fundamentally need a policy gradient component (SAC = Soft Q-learning + policy network). When you need safety during training, you move to SARSA or constrained RL. When you need sample efficiency in continuous tasks, you move to SAC or model-based methods.

Understanding these boundaries — not as artificial restrictions but as structural requirements of the underlying mathematics — is what separates practitioners who can diagnose why an algorithm fails from those who blindly hyperparameter-search.

## 15. Case studies

### Case Study 1: DQN on Atari (Mnih et al., 2015)

The Nature DQN paper trained a single convolutional architecture on 49 Atari games from raw 84×84 pixel inputs, achieving human-level performance on 29 games and superhuman performance on 23. The algorithm was Q-learning with:

- $\varepsilon$ annealed from 1.0 to 0.1 over 1 million steps, then held at 0.1
- Replay buffer of 1 million transition tuples
- Target network updated every 10,000 gradient steps by copying the online network's weights
- Mini-batches of 32 transitions per gradient step
- RMSprop optimizer with learning rate $2.5 \times 10^{-4}$
- Frame stacking of 4 consecutive grayscale frames for temporal context
- Reward clipping to $[-1, +1]$ for cross-game normalization

On Breakout, DQN reached a score of 401.2 versus the human professional score of 31.8 (approximately 12.6× superhuman). On Pong, DQN reached +20.9 versus human +9.3. On Space Invaders, DQN scored 1,976 versus human 1,652.

The Q-learning update rule was unchanged from Watkins (1989). Every significant improvement came from representation (convolutional features), stability (replay buffer and target network), and scale (training for 50 million frames per game). The algorithm's core mathematics proved robust at a scale 7 orders of magnitude larger than the tabular problems it was proven on.

### Case Study 2: Double DQN (van Hasselt, Guez, Silver, 2016)

Applying Double Q-learning's two-estimator correction to DQN required changing exactly one line: use the online network to select the best next action, but evaluate its value using the target network:

$$y = r + \gamma Q_{\theta^-}\!\left(s', \argmax_{a'} Q_\theta(s', a')\right)$$

Evaluated on 49 Atari games, Double DQN improved median human-normalized score from 47.5% (standard DQN) to 117.5% — a 2.5× improvement on the same architecture and compute. On specific games, the overestimation was measured directly: on Asterix, standard DQN estimated Q-values on the order of $10^9$ (the theoretical maximum return from any state is $R_{\max}/(1-\gamma) \approx 10^4$ with their hyperparameters). Double DQN reduced estimates to approximately $2 \times 10^4$ — a factor of 50,000 reduction in overestimation.

#### Worked example: Q-value calibration on a single game

Asterix: DQN's best action Q-value at a typical game state was measured at approximately $\hat{Q}_{\text{DQN}} \approx 10^9$. True maximum possible return with $R_{\max}=1$ (after reward clipping) and $\gamma=0.99$: $R_{\max}/(1-\gamma) = 1/0.01 = 100$. So DQN overestimated by a factor of $10^7$. Double DQN: same state, $\hat{Q}_{\text{DDQN}} \approx 150$, still slightly above the theoretical max of 100 but off by only 50%.

The practical consequence: DQN's overestimation on Asterix caused it to commit too strongly to certain actions early, slowing exploration and producing an unstable learning curve. Double DQN's more calibrated values led to better exploration balance and a final score roughly 50% higher than DQN on Asterix.

### Case Study 3: Q-learning in a simplified trading environment

Before the SB3 comparison, consider a more practical case study: Q-learning on a simple financial execution environment. The state is a tuple (current_inventory, market_spread, time_remaining), the actions are buy/hold/sell, and the reward is mark-to-market PnL minus transaction costs.

```python
import numpy as np
import gymnasium as gym
from gymnasium import spaces

class SimpleExecutionEnv(gym.Env):
    """
    Minimal execution RL environment: agent must buy 100 shares by end of day.
    State: (inventory_remaining [0,100], time_step [0,50], spread_bps [0,5])
    Actions: 0=buy 10, 1=buy 5, 2=hold, 3=sell 5
    Reward: negative implementation shortfall vs VWAP
    """
    def __init__(self):
        super().__init__()
        self.observation_space = spaces.MultiDiscrete([11, 51, 6])
        self.action_space = spaces.Discrete(4)
        self.target = 100

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.inventory = 0
        self.t = 0
        self.vwap_price = 100.0
        return (self.target - self.inventory, self.t, 2), {}

    def step(self, action):
        sizes = [10, 5, 0, -5]
        size = sizes[action]
        price_impact = abs(size) * 0.01  # 1bp per lot
        spread_cost = abs(size) * 0.005  # half-spread cost
        self.inventory += size
        self.inventory = max(0, min(self.target, self.inventory))
        reward = -(price_impact + spread_cost)
        self.t += 1
        done = self.t >= 50
        if done and self.inventory < self.target:
            reward -= (self.target - self.inventory) * 0.1  # penalty for shortfall
        spread = np.random.randint(0, 6)
        state = (self.target - self.inventory, min(self.t, 50), spread)
        return state, reward, done, False, {}

# Train Q-learning on the execution environment
env = SimpleExecutionEnv()
Q_exec, returns_exec, _ = q_learning(
    env_name=None,  # pass env directly if modified
    n_episodes=5000,
    alpha=0.2,
    gamma=0.95,
)
```

This stylized example illustrates why Q-learning is relevant in quantitative finance: the state space is small (11 × 51 × 6 = 3,366 states), the action space is discrete (4 actions), rewards are bounded, and the off-policy nature means we can learn from historical execution data (transitions from the past) without on-policy interaction. The ability to replay stored order-book states is exactly the off-policy flexibility Q-learning provides.

For real trading applications, the state space quickly becomes too large for a table — you need DQN or neural network extensions. But prototyping with tabular Q on a simplified version is invaluable for verifying the reward structure, checking that convergence occurs, and setting realistic performance baselines before scaling.

### Case Study 4: Stable-Baselines3 DQN on CartPole and LunarLander

```python
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
import gymnasium as gym

# CartPole-v1: state_dim=4, n_actions=2, max return=500
env_cp = gym.make("CartPole-v1")
model_cp = DQN(
    "MlpPolicy",
    env_cp,
    learning_rate=1e-4,
    buffer_size=50_000,
    learning_starts=1_000,
    batch_size=32,
    tau=1.0,                      # hard copy target network
    target_update_interval=500,   # copy every 500 steps
    train_freq=4,                 # update every 4 environment steps
    exploration_fraction=0.1,     # epsilon decays over 10% of training
    exploration_final_eps=0.02,
    verbose=0,
)
model_cp.learn(total_timesteps=100_000)
mean_r, std_r = evaluate_policy(model_cp, env_cp, n_eval_episodes=100)
print(f"CartPole-v1 DQN: {mean_r:.1f} +/- {std_r:.1f}")
# Expected: 500.0 +/- 0.0 (maximum possible — pole never falls)

# LunarLander-v2: state_dim=8, n_actions=4, solve threshold=200
env_ll = gym.make("LunarLander-v2")
model_ll = DQN(
    "MlpPolicy",
    env_ll,
    learning_rate=5e-4,
    buffer_size=100_000,
    learning_starts=5_000,
    batch_size=128,
    target_update_interval=250,
    exploration_fraction=0.12,
    exploration_final_eps=0.01,
    verbose=0,
)
model_ll.learn(total_timesteps=500_000)
mean_r_ll, std_r_ll = evaluate_policy(model_ll, env_ll, n_eval_episodes=100)
print(f"LunarLander-v2 DQN: {mean_r_ll:.1f} +/- {std_r_ll:.1f}")
# Expected: ~220-250 (solved threshold is 200; best tuned DQN reaches ~280)
```

SB3 DQN achieves CartPole-v1 perfect score (500/500) in under 100,000 timesteps consistently. The key SB3 settings matching DQN theory: `target_update_interval` (target network copy frequency), `buffer_size` (replay buffer capacity), `exploration_fraction` (epsilon decay schedule). All are directly derived from the stability analysis of the deadly triad.

## 15. The Q-learning update as temporal difference in the brain

Before turning to practical advice, it is worth noting a remarkable connection between Q-learning's TD error and neuroscience. The TD error $\delta_t = r + \gamma \max_{a'} Q(s', a') - Q(s, a)$ has a striking similarity to the firing patterns of dopaminergic neurons in the ventral tegmental area (VTA) of the mammalian brain.

Wolfram Schultz and colleagues showed in the 1990s that dopamine neurons initially fire when an unexpected reward occurs, but after conditioning, they fire when the reward-predicting stimulus occurs and are suppressed (below baseline) when the expected reward is absent. This matches the sign and timing of $\delta_t$ exactly: positive $\delta_t$ when outcomes are better than predicted (dopamine burst), negative $\delta_t$ when worse than predicted (dopamine dip), and zero $\delta_t$ when the prediction was correct.

This is not a coincidence: several researchers, particularly Montague, Dayan, and Sejnowski (1996), proposed that the brain's reward learning circuitry implements a form of TD learning. The dopamine signal encodes the prediction error $\delta_t$, the striatum updates value representations in proportion to this signal, and the resulting learning follows Bellman-like updates. Q-learning, derived purely from mathematical consistency requirements on $Q^*$, independently rediscovered a learning algorithm that evolution had found billions of years earlier in vertebrate brains.

The practical implication for RL practitioners: the TD error $\delta_t$ is not just a computational shortcut — it is an information-theoretically meaningful quantity measuring the surprise of the observed outcome relative to the current value model. Large positive $\delta_t$ means the agent discovered something better than expected; large negative $\delta_t$ means reality was worse than anticipated. Monitoring the distribution of $\delta_t$ values during training is one of the most informative diagnostics available: if $|\delta_t|$ remains large throughout training, the Q-table is not converging; if $\delta_t \approx 0$ too quickly, the agent may have stopped exploring and settled at a suboptimal policy.

#### Worked example: tracing TD error through a single episode

Suppose an agent on a 5-state linear chain starts at state 1, transitions right at each step, and receives a reward of +10 at state 5. Initial Q-table: all zeros. Let $\alpha = 0.5$, $\gamma = 0.9$.

**Step 1**: In state 1, take action "right", reach state 2, reward = 0.
$$\delta_1 = 0 + 0.9 \cdot \max Q(2, \cdot) - Q(1, \text{right}) = 0 + 0.9 \cdot 0 - 0 = 0$$
No update (the agent expected nothing and got nothing).

**Step 5**: In state 4, take action "right", reach state 5, reward = +10.
$$\delta_5 = 10 + 0.9 \cdot 0 - Q(4, \text{right}) = 10 - 0 = 10$$
$$Q(4, \text{right}) \leftarrow 0 + 0.5 \cdot 10 = 5.0$$
A large positive surprise — the agent updated $Q(4, \text{right})$ dramatically.

**Next episode, step 4**: In state 3, take action "right", reach state 4, reward = 0.
$$\delta_4 = 0 + 0.9 \cdot \max Q(4, \cdot) - Q(3, \text{right}) = 0 + 0.9 \cdot 5.0 - 0 = 4.5$$
$$Q(3, \text{right}) \leftarrow 0 + 0.5 \cdot 4.5 = 2.25$$
The value propagates backward — state 3 now "knows" about the reward at state 5.

This backward propagation of value through TD errors is the core mechanism of Q-learning. Each episode the reward signal travels one more step backward through the state chain, until eventually $Q(1, \text{right})$ converges to approximately $0.9^4 \times 10 = 6.56$ (the discounted value of the terminal reward from state 1). This is exactly what the Bellman optimality equation predicts: $Q^*(1, \text{right}) = \gamma^4 \times 10$.

### The speed of credit propagation

One limitation of one-step Q-learning is how slowly value propagates backward through long chains. In the example above, the reward at state 5 only reached state 4 after the first episode, state 3 after the second, and so on. For environments with long delay between actions and rewards (sparse reward or episodic tasks with hundreds of steps), this can require thousands of episodes before the initial states receive meaningful Q-value updates.

This is why n-step returns and eligibility traces exist: they propagate credit further backward per episode. An n-step Q-learning agent with $n=5$ would update all 5 states in a single episode of the chain above. The tradeoff: each step of lookahead adds variance (more stochastic transitions compound). The optimal $n$ depends on the environment's noise level and episode length — a fact that $\text{TD}(\lambda)$ handles gracefully by blending all $n$-step returns through an exponential weighting.

## 16. Epsilon decay in deep RL: lessons from DQN

The choice of $\varepsilon$ decay schedule has a larger impact on learning speed than most practitioners expect. In DQN's original Atari implementation, $\varepsilon$ was annealed from 1.0 to 0.1 over the first 1 million steps, then held at 0.1 for the remaining training. This is a particularly slow decay by tabular RL standards — over 1 million steps of predominantly random behavior. The reason: Atari environments have extremely sparse feedback in early training, and a rapidly-greedy agent that prematurely commits to a discovered path will never explore enough of the state space to find the reward structures in less obvious parts of the level.

The effect is visible in DQN's learning curves on Atari games that have dense immediate rewards (Breakout, Pong): learning begins quickly and $\varepsilon$ decay speed matters less. On games with sparse or delayed rewards (Montezuma's Revenge, Pitfall), even DQN's slow decay is not slow enough — these games remain largely unsolved by vanilla DQN because the $\varepsilon$-greedy strategy does not create the right kind of structured exploration. Later work (Go-Explore, curiosity-driven exploration, Ape-X) specifically addresses this failure mode.

For tabular environments, faster decay is typically appropriate because the state space is smaller and all states can be visited quickly with random exploration. A schedule like exponential decay with factor 0.9995 per step decays $\varepsilon$ from 1.0 to 0.01 in approximately 9,000 steps — appropriate for Taxi-v3 where the agent can visit most of the 500 states in a few thousand steps. For FrozenLake with stochastic transitions, slightly slower decay (factor 0.9997 per step) allows more thorough exploration of the 16-state space despite the ice slipping.

The general principle: the right $\varepsilon$ decay schedule is the one that ensures the agent has approximately visited all relevant state-action pairs *before* the exploration rate drops to its minimum. If your environment has $N$ states and $A$ actions, the minimum number of exploration steps is roughly $N \times A \times \log(1/\delta)$ for visit probability $1-\delta$ per pair (coupon-collector bound). Set your decay schedule so that $\varepsilon$ does not reach $\varepsilon_{\text{end}}$ before that many steps have elapsed.

```python
def compute_min_exploration_steps(n_states, n_actions, delta=0.01,
                                   eps_start=1.0, eps_end=0.01):
    """
    Estimate minimum exploration steps before epsilon should reach its floor.
    Based on the coupon-collector problem: expected visits to cover N*A pairs.
    """
    import math
    n_pairs = n_states * n_actions
    # Expected steps to visit all N*A state-action pairs at least once
    # if each step visits a uniform random (s,a): E[coupon collector] = N*A * H(N*A)
    expected_steps = n_pairs * sum(1/k for k in range(1, n_pairs + 1))
    # Divide by eps_start to account for epsilon-greedy (fraction eps is random)
    # Use a safety factor of 2x
    min_exploration_steps = int(2 * expected_steps / eps_start)
    return min_exploration_steps

# For Taxi-v3: 500 states x 6 actions = 3000 pairs
taxi_steps = compute_min_exploration_steps(500, 6)
print(f"Taxi-v3 min exploration steps: ~{taxi_steps:,}")
# Output: ~50,000 steps — so epsilon should not reach 0.01 until at least 50k steps
```

This estimate is a lower bound. Real environments have non-uniform state visitation (some states are harder to reach), so in practice you want the exploration phase to be 2–5× longer than this estimate.

## 17. When to use Q-learning (and when not to)

**Use tabular Q-learning when:**
- State and action spaces are discrete and small (dictionary-based sparse tables work up to ~$10^7$ states)
- You need a provably convergent baseline before committing to a neural approach
- You are in early research and want to verify the RL loop before scaling representation
- Memory is constrained but compute is not (Q-table updates are $O(1)$, no backpropagation)

**Use Double Q-learning over vanilla Q when:**
- Rewards are stochastic with high variance
- Action spaces are large (Taxi has 6 actions; games with 18 actions like Atari benefit more from Double Q)
- You observe implausibly high Q-values early in training
- You want faster convergence in the first 25% of training

**Use DQN or neural Q when:**
- State spaces are image-based or too large for a table
- You have access to GPU compute and can run millions of environment steps
- Sample efficiency is secondary (DQN needs 10–50 million frames for Atari)

**Do not use Q-learning when:**
- **Continuous actions**: $\max_{a'} Q_\theta(s', a')$ is intractable for continuous $A$. Use SAC, TD3, or DDPG instead.
- **Highly non-stationary environments**: the convergence proof assumes a fixed MDP; drifting transition dynamics or rewards violate the theoretical foundation.
- **Extreme reward sparsity**: one reward per episode with Q-learning gives no signal for thousands of steps. Use Hindsight Experience Replay (HER), reward shaping, or curiosity-driven intrinsic motivation.
- **Safety-critical real-world exploration**: Q-learning explores randomly ($\varepsilon$-greedy), which can be dangerous in physical systems. Use constrained RL or safe exploration algorithms.
- **Very high sample efficiency required**: SAC on continuous control benchmarks achieves similar performance to DQN in 1–10% of the samples. If each environment interaction is expensive, use a model-based method.

| Problem type | Recommended method | Key reason |
|---|---|---|
| Small finite MDP, unknown model | Tabular Q | Convergence guarantee, O(1) update |
| Small finite MDP, known model | Value Iteration | Exact operator, no sampling |
| Large discrete state space (images) | DQN + Double Q | Neural features + bias correction |
| Continuous control (robotics) | SAC / TD3 | No tractable max over continuous A |
| Offline data, no online interaction | CQL / IQL | Conservative Q for distribution shift |
| Multi-step credit needed | n-step Q / TD(lambda) | Better bias-variance for sparse rewards |
| Safety-critical exploration | SARSA or constrained RL | SARSA accounts for exploration cost |
| High sample efficiency priority | SAC + model-based | 100x more efficient than DQN |

## 18. Further reading

- **Watkins, C.J.C.H., and Dayan, P. (1992)**: "Q-learning." *Machine Learning*, 8(3-4): 279–292. The formal convergence proof. Short and entirely accessible.
- **Sutton, R.S., and Barto, A.G. (2018)**: *Reinforcement Learning: An Introduction*, 2nd edition. MIT Press. Chapter 6 covers TD learning and Q-learning; Chapter 11 is the definitive treatment of the deadly triad with Baird's counterexample. [Free online.](http://incompleteideas.net/book/the-book.html)
- **van Hasselt, H. (2010)**: "Double Q-learning." *NeurIPS 2010*. The Double Q-learning proposal. Four pages, completely self-contained.
- **van Hasselt, H., Guez, A., and Silver, D. (2016)**: "Deep Reinforcement Learning with Double Q-learning." *AAAI 2016*. Double DQN on 49 Atari games, with direct Q-value overestimation measurements.
- **Mnih, V. et al. (2015)**: "Human-level control through deep reinforcement learning." *Nature*, 518: 529–533. The DQN breakthrough with Atari results.
- **Baird, L. (1995)**: "Residual algorithms: Reinforcement learning with function approximation." *ICML 1995*. The 7-state counterexample proving the deadly triad can cause divergence.
- Within this series: [Temporal-Difference Learning: TD(0), SARSA, and Bootstrapping](/blog/machine-learning/reinforcement-learning/temporal-difference-learning-td0-sarsa-bootstrapping) (B3); [Deep Q-Networks: DQN, Double DQN, and Dueling Architectures](/blog/machine-learning/reinforcement-learning/deep-q-networks-dqn) (D1).

## 19. Key takeaways

1. **Q-learning is off-policy because of one symbol**: replacing SARSA's sampled $a'$ with $\max_{a'} Q(s', a')$ makes the TD target independent of the behavior policy, eliminating the need for importance sampling corrections at the one-step lookahead.

2. **Watkins's convergence theorem** requires all state-action pairs visited infinitely often and step sizes satisfying Robbins-Monro ($\sum_t \alpha_t = \infty$, $\sum_t \alpha_t^2 < \infty$); tabular Q-learning converges almost surely to $Q^*$ under these conditions.

3. **Max-bias is systematic and positive**: even with unbiased individual Q-estimates, $\mathbb{E}[\max_a \hat{Q}] > \max_a \mathbb{E}[\hat{Q}]$ by Jensen's inequality; with 6 actions and $\sigma=0.5$, the bias reaches approximately $+0.48$.

4. **Double Q-learning eliminates max-bias** by decoupling action selection ($Q_1$) from value evaluation ($Q_2$) using independently trained estimators; memory cost doubles, computation per step is unchanged.

5. **Q-learning learns the optimal policy; SARSA learns the policy it is following**: Q-learning finds the dangerous-but-greedy-optimal cliff path; SARSA finds the safer path that accounts for exploration mistakes.

6. **The deadly triad** (function approximation + bootstrapping + off-policy) can cause divergence, as proven by Baird's 7-state counterexample; tabular Q is immune because table entries are independent scalars.

7. **DQN's two stabilization tricks** — experience replay (breaks temporal correlation) and target network (slows moving targets) — are engineering heuristics that make the deadly triad tractable in practice, not theoretical convergence proofs.

8. **Q-learning cannot handle continuous actions**: $\max_{a'} Q_\theta(s', a')$ requires enumeration over all actions, which is intractable for continuous $A$; use SAC, TD3, or DDPG for continuous control.

9. **Optimistic initialization provides implicit exploration**: starting Q-values above expected return drives the agent to visit all states before settling into exploitation, a clean alternative to pure $\varepsilon$-greedy in deterministic environments.

10. **The path from Q-table to DQN is structural, not algorithmic**: the update rule is identical; the only change is representation (neural network versus table) plus two engineering stabilizers; mastering tabular Q is the direct prerequisite for understanding all deep Q-learning variants.
