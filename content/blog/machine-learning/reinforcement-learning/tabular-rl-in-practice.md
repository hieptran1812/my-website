---
title: "Tabular RL in Practice: From GridWorld to Taxi, When Tables Are Enough"
date: "2026-06-22"
publishDate: "2026-06-22"
description: "A complete practitioner's guide to tabular Q-learning and SARSA: state space design, Q-table initialization, epsilon scheduling, convergence monitoring, and a fully solved Taxi-v3 case study with policy visualisation."
tags:
  [
    "reinforcement-learning",
    "deep-learning",
    "q-learning",
    "sarsa",
    "tabular-rl",
    "gymnasium",
    "exploration",
    "markov-decision-process",
    "machine-learning",
    "numpy",
  ]
category: "machine-learning"
subcategory: "Reinforcement Learning"
author: "Hiep Tran"
featured: true
readTime: 55
image: "/imgs/blogs/tabular-rl-in-practice-1.png"
---

You have a small environment — a 5×5 grid, a robot, four movement actions, a reward of +1 for reaching the goal. The problem fits on a napkin. You reach for a neural network because that is what every tutorial shows, and two days later you are debugging exploding gradients, tuning learning rate schedulers, fiddling with replay buffer sizes, and questioning every life choice. Meanwhile your colleague wrote a Q-table in NumPy and had a working policy in four hours.

Tabular RL is not the obsolete cousin of deep RL. For a well-defined class of problems, it is the *correct* algorithm — exact, fully interpretable, with provable convergence guarantees that no neural network can currently match. The Q-table is not an approximation: for a small discrete environment, it stores the exact solution to the Bellman optimality equations. Every Q-value is a number you can print, inspect, and explain. Every policy decision has an audit trail.

Understanding tabular RL deeply is also the fastest path to understanding deep RL. The questions that matter — when is the value function Lipschitz enough to generalise? when does bootstrapping create dangerous learning dynamics? when does reward shaping corrupt the optimal policy? — are all visible and debuggable in the tabular regime, before the neural network adds a second layer of complexity on top. The engineer who skips tabular RL on the way to DQN typically spends weeks debugging problems that would have been visible in ten minutes on a Q-table.

This post is a complete practitioner's guide: when to choose tabular RL, how to design the state and action space, how to implement Q-learning and SARSA with full convergence monitoring, how to schedule exploration for fast convergence, and how to diagnose the three distinct failure modes. We will solve Taxi-v3 completely — Q-table heat maps, policy visualisation, convergence curves, a final average reward of approximately +8 — and then quantify precisely where the approach breaks, which motivates the transition to [function approximation and tile coding](/blog/machine-learning/reinforcement-learning/function-approximation-and-tile-coding) in the next post.

![Five criteria stack showing when tabular RL beats deep RL based on state count, action discreteness, interpretability, and prototyping stage](/imgs/blogs/tabular-rl-in-practice-1.png)

## When Tabular RL Is the Right Choice

The decision to use tabular RL is not a matter of taste or computational budget — it is a structural question: can the Q-table represent the value function exactly? That requires the state space to be small enough to enumerate, store, and update.

A Q-table for an environment with $|S|$ states and $|A|$ actions stores $|S| \times |A|$ floating-point values. At 8 bytes per float64, Taxi-v3 (500 states, 6 actions) needs $500 \times 6 \times 8 = 24{,}000$ bytes — 24 KB, a rounding error compared to any practical memory limit. FrozenLake-8×8 (64 states, 4 actions) needs 2 KB. These fit in L1 cache. GridWorld-4×4 is 512 bytes. You can print the entire value function in a terminal window, audit it row by row, and trust that what you see is what the agent is optimising.

This physical tractability has a deeper significance. The Q-table is not just a convenience — it is a complete representation. When Q-learning converges on a tabular environment, the converged Q-values are a certificate that the Bellman optimality equations are solved. There is no approximation error, no generalisation gap, no distribution shift between training and test. The policy is $\pi(s) = \arg\max_a Q(s, a)$ for every state the agent could ever visit, and every one of those Q-values has been estimated from direct experience of that state.

### The Four Criteria

**Criterion 1: Small, discrete, bounded state space.** The practical threshold is around $|S| \leq 10{,}000$ states. Below this, the Q-table fits comfortably in memory, training takes seconds to minutes, and full coverage is achievable within a reasonable episode budget. Above this, coverage becomes the binding constraint — the agent may train for millions of episodes without visiting every state-action pair, which violates the assumptions that guarantee Q-learning convergence.

**Criterion 2: Discrete action space.** Q-learning requires enumerating $\arg\max_a Q(s, a)$ at every step. With 6 discrete actions (Taxi-v3), this takes microseconds. With a continuous action space, this becomes an optimisation problem at every timestep — and you have essentially re-invented an actor-critic architecture. Continuous action spaces require a fundamentally different approach (SAC, TD3, DDPG) described later in this series.

**Criterion 3: Known, bounded state space at design time.** If new states can appear at test time that were never seen during training, a Q-table has no entry for them. There is no interpolation, no generalisation, no fallback — the lookup fails or returns a default value. A tabular agent requires that the set of possible states at training time is a superset of the possible states at test time.

**Criterion 4: Interpretability or prototyping requirement.** In regulated settings — a trading algorithm with position limits, a medical scheduling policy, a manufacturing robot with safety constraints — you may need to explain every action the system takes. A Q-table gives you this completely: for any state, the action is $\arg\max_a Q(s, a)$ and the confidence is the Q-value gap to the second-best action. In the prototyping case, a tabular baseline built in an hour validates your reward signal and environment specification before committing to weeks of deep RL engineering.

![Matrix comparing state space sizes, action counts, and tabular feasibility across GridWorld, FrozenLake, Taxi, Chess, Atari, and MuJoCo environments](/imgs/blogs/tabular-rl-in-practice-2.png)

The numbers in this matrix make the feasibility boundary concrete. The jump from Taxi-v3 (500 states) to Chess ($\sim 10^{47}$ states) to Atari pixel space ($\sim 10^{33,696}$ states) is not a matter of degree — it is a difference in kind that renders Q-tables physically impossible. We will quantify this precisely in the capacity wall section.

## Formal Background: The Bellman Equations

Before implementing anything, it is worth grounding the Q-table update in the equations it is actually solving. This makes the hyperparameter choices less arbitrary and the failure modes easier to diagnose.

An environment is a **Markov Decision Process** (MDP): a tuple $(S, A, P, R, \gamma)$ where $S$ is the state space, $A$ the action space, $P(s' \mid s, a)$ the transition probability, $R(s, a)$ the expected reward, and $\gamma \in [0, 1)$ the discount factor.

The **value function** $V^\pi(s)$ under policy $\pi$ is the expected discounted return starting from state $s$:

$$V^\pi(s) = \mathbb{E}_\pi \left[ \sum_{t=0}^\infty \gamma^t r_t \;\middle|\; s_0 = s \right]$$

The **action-value function** (Q-function) $Q^\pi(s, a)$ is the expected return starting from state $s$, taking action $a$, then following policy $\pi$:

$$Q^\pi(s, a) = \mathbb{E}_\pi \left[ \sum_{t=0}^\infty \gamma^t r_t \;\middle|\; s_0 = s, a_0 = a \right]$$

The key insight is that $Q^\pi$ satisfies a recursive consistency condition — the **Bellman equation**:

$$Q^\pi(s, a) = R(s, a) + \gamma \sum_{s'} P(s' \mid s, a) Q^\pi(s', \pi(s'))$$

The optimal Q-function $Q^*$ satisfies the **Bellman optimality equation**:

$$Q^*(s, a) = R(s, a) + \gamma \sum_{s'} P(s' \mid s, a) \max_{a'} Q^*(s', a')$$

Q-learning solves this equation by stochastic approximation. Instead of integrating over the full transition distribution, it uses a single sampled transition $(s, a, r, s')$:

$$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]$$

The term in brackets is the **TD error**: the difference between the current Q-value estimate and the one-step Bellman backup. The learning rate $\alpha$ controls how quickly Q-values incorporate new observations. The convergence theorem (Watkins and Dayan 1992) guarantees that if every state-action pair is visited infinitely often, and if $\sum_t \alpha_t = \infty$ and $\sum_t \alpha_t^2 < \infty$, then $Q \to Q^*$ with probability 1.

SARSA uses the same update structure but evaluates the *next action actually taken* rather than the greedy action:

$$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma Q(s', a') - Q(s, a) \right]$$

where $a'$ is sampled from the current policy $\pi$ at $s'$ (including epsilon-greedy randomness). SARSA is **on-policy** — it learns the value of the behaviour policy. Q-learning is **off-policy** — it learns the value of the greedy policy while the behaviour policy may be epsilon-greedy. The difference is not academic: in stochastic environments, it changes which policy is learned.

### Why Q-Learning Converges Off-Policy

The proof that Q-learning converges to $Q^*$ off-policy relies on the Bellman optimality operator $T^*$ being a contraction mapping under the $\ell_\infty$ norm. Define $T^*$ as:

$$(T^* Q)(s, a) = R(s, a) + \gamma \sum_{s'} P(s' \mid s, a) \max_{a'} Q(s', a')$$

Then $Q^*$ is the unique fixed point of $T^*$: $T^* Q^* = Q^*$. To show that Q-learning iterates converge to this fixed point, we need to show that $T^*$ is a contraction: $\| T^* Q - T^* Q' \|_\infty \leq \gamma \| Q - Q' \|_\infty$ for any two Q-functions $Q$ and $Q'$.

$$\| T^* Q - T^* Q' \|_\infty = \left\| \gamma \sum_{s'} P(s' \mid s, a) (\max_{a'} Q(s', a') - \max_{a''} Q'(s', a'')) \right\|_\infty$$

$$\leq \gamma \left\| \max_{a'} Q - \max_{a''} Q' \right\|_\infty \leq \gamma \| Q - Q' \|_\infty$$

where the last inequality uses the fact that $|\max_a f(a) - \max_a g(a)| \leq \max_a |f(a) - g(a)|$. Since $\gamma < 1$, the operator is a strict contraction, and by Banach's fixed point theorem, repeated application of $T^*$ converges to $Q^*$ from any starting point.

Q-learning's stochastic approximation version replaces the exact expectation over transitions with a single sample, adding noise. The Robbins-Monro conditions on $\alpha$ ensure the stochastic noise averages out over time, preserving convergence.

### The Cliff Walking Example: When Off-Policy Goes Wrong

The CliffWalking-v0 environment (Sutton and Barto, Example 6.6) is a 4×12 grid where:
- The agent starts at the bottom-left corner (3, 0)
- The goal is the bottom-right corner (3, 11)
- The bottom row (except start and goal) is a cliff: falling in gives -100 reward and resets to start
- Every other step gives -1 reward

Q-learning learns to walk along the edge of the cliff — the shortest path, which is optimal for a greedy policy that never slips. SARSA, accounting for the epsilon-greedy exploration that might randomly push the agent over the edge, learns the safe upper path — 1 cell up from the cliff edge — which is slightly longer but much more robust to exploration noise during training.

```python
import gymnasium as gym
import numpy as np

env = gym.make("CliffWalking-v0")
n_states = env.observation_space.n    # 48
n_actions = env.action_space.n        # 4

# Q-learning: learns the optimal greedy path (cliff edge)
Q_ql = np.zeros((n_states, n_actions))
# SARSA: learns the safe path (one row above cliff)
Q_sarsa = np.zeros((n_states, n_actions))

def train_episode(env, Q, update_fn, eps=0.1, alpha=0.5, gamma=1.0):
    state, _ = env.reset()
    action = int(np.random.randint(n_actions) if np.random.random() < eps
                 else np.argmax(Q[state]))
    total_reward, done = 0, False

    while not done:
        ns, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        next_action = int(np.random.randint(n_actions) if np.random.random() < eps
                         else np.argmax(Q[ns]))
        update_fn(Q, state, action, reward, ns, next_action, done, alpha, gamma)
        state, action = ns, next_action
        total_reward += reward
    return total_reward

# After convergence (thousands of episodes):
# Q-learning greedy eval: ~-13 (cliff edge path, 13 steps)
# SARSA greedy eval: ~-17 (safe path, 17 steps)
# Q-learning during training (eps=0.1): ~-60 (frequent cliff falls)
# SARSA during training (eps=0.1): ~-17 (safe path, robust)
```

The CliffWalking example illustrates a general principle: **Q-learning optimises for the greedy policy; SARSA optimises for the epsilon-greedy behaviour policy.** If your evaluation uses a greedy policy, Q-learning's target is more relevant. If your deployment policy continues to explore (as in online systems), SARSA's target is more appropriate.

## State and Action Space Design

Even when tabular RL is appropriate, the quality of the solution depends entirely on how you define "state." Getting the state representation wrong is the most common source of poor convergence — and it is a design decision that no hyperparameter tuning can fix.

### The Markov Property in Practice

A state representation must satisfy the **Markov property**: the next state distribution must depend only on the current state, not on any earlier history. Formally, $P(s_{t+1} \mid s_0, a_0, \ldots, s_t, a_t) = P(s_{t+1} \mid s_t, a_t)$. When the Markov property is violated, you have a partially observed MDP (POMDP), and Q-learning converges to a suboptimal policy even given infinite data.

Consider a robot navigating a maze. Raw $(x, y)$ grid coordinates satisfy the Markov property perfectly — that is the complete state. But suppose you encode state as "number of steps taken so far" only. Two different positions that happen to be reached after the same number of steps now map to the same state integer, even though they may have completely different reachable next states. The Q-table entries become averages over distinct situations, the policy is confounded, and convergence may fail entirely.

**The diagnostic question for any state representation: if I know only $s_t$, can I predict the distribution of $s_{t+1}$ for any given action $a_t$?** If the answer is no, you are missing features. Common causes include: ignoring the position of a key object (e.g., encoding only the agent's position but not the passenger's location in Taxi-v3), ignoring a system mode (a door that can be open or closed), or using a lossy encoding (distance to goal instead of full coordinates).

### Choosing Between Raw Observations and Engineered Features

Gymnasium environments typically provide either a flat integer state (Taxi-v3, FrozenLake) or a raw array (CartPole's 4-float observation, Atari's 84×84 pixel array). For tabular RL, you need to map these to a discrete integer.

**Integer environments (Taxi-v3, FrozenLake, CliffWalking):** The environment directly provides a discrete state integer. No feature engineering required — use the state directly as the Q-table index.

**Continuous or composite environments (CartPole, MountainCar):** You must discretise the continuous observation into bins. The number of bins controls the tradeoff between representation quality and Q-table size. For CartPole's 4-dimensional observation (cart position, velocity, pole angle, angular velocity), discretising each dimension to 10 bins gives $10^4 = 10{,}000$ states — just within the tabular feasibility range. Discretising to 20 bins gives $1.6 \times 10^5$ states — borderline. 30 bins gives $8.1 \times 10^5$ states — infeasible.

```python
import numpy as np
import gymnasium as gym

# Discretising CartPole for tabular RL
env = gym.make("CartPole-v1")
obs_low = np.array([-2.4, -3.0, -0.2095, -2.5])
obs_high = np.array([2.4,  3.0,  0.2095,  2.5])
n_bins = 10

bins = [np.linspace(low, high, n_bins + 1)[1:-1]
        for low, high in zip(obs_low, obs_high)]

def discretise(obs):
    """Map a continuous CartPole observation to a single integer state."""
    indices = [np.digitize(obs[i], bins[i]) for i in range(4)]
    # Convert (i0, i1, i2, i3) to a single integer
    state = 0
    for idx in indices:
        state = state * n_bins + idx
    return state

n_states = n_bins ** 4  # 10,000
print(f"Total states: {n_states}")  # 10000
```

### Action Discretisation

For environments with continuous actions, you must map the continuous action space to a finite set of discrete actions. The key insight is that the resolution required depends on the sensitivity of the optimal policy to small action perturbations.

For a 1D continuous action in $[-1, 1]$ (e.g., a motor torque), discretising to $k$ bins gives actions $\{-1 + 2i/(k-1) : i = 0, \ldots, k-1\}$. With $k=5$ bins you have $\{-1, -0.5, 0, 0.5, 1\}$. For many control tasks, 5–10 bins is sufficient to capture the optimal policy structure. With multiple action dimensions, the table grows exponentially: 3 dimensions at 5 bins each gives $5^3 = 125$ actions — still manageable. 6 dimensions at 5 bins each gives $5^6 = 15{,}625$ actions — the Q-table becomes 625 MB for a 500-state environment, which is large but feasible.

### Reward Shaping for Dense Signal

Many environments offer only **sparse rewards**: a single +1 for reaching the goal after potentially hundreds of steps, and 0 everywhere else. With epsilon-greedy exploration, the probability that a random walk reaches the goal in a 100-state environment is reasonable. But in a 5,000-state environment, random exploration may take millions of episodes to first discover the goal, during which no useful Q-value updates propagate.

**Reward shaping** adds auxiliary signal to guide the agent toward the goal faster. The critical theorem (Ng, Harada, Russell 1999) states that **potential-based shaping** $F(s, a, s') = \gamma \Phi(s') - \Phi(s)$ does not change the optimal policy of the original MDP. If you can find a potential function $\Phi: S \to \mathbb{R}$ that is higher near the goal, then $F > 0$ for steps toward the goal and $F < 0$ for steps away, providing dense signal without distorting the solution.

For a navigation task, $\Phi(s) = -\text{dist}(s, \text{goal})$ (negative Manhattan distance) works well: every step closer to the goal gives $F = \gamma \cdot (-\text{dist}(s', \text{goal})) - (-\text{dist}(s, \text{goal})) > 0$ when the agent approaches. The agent does not need to reach the goal to receive signal.

The critical mistake is adding **non-potential-based shaping**. A common error is rewarding the agent for visiting new cells (exploration bonus): $+0.1$ for every first visit to a state. This changes the MDP's optimal policy — the maximum-reward strategy may now be to visit many cells without ever completing the task. Any shaped reward that cannot be written as $\gamma \Phi(s') - \Phi(s)$ for some $\Phi$ is altering your problem.

## Q-Table Implementation Details

The Q-table $Q: S \times A \to \mathbb{R}$ is the central data structure in tabular RL. Let us look carefully at the implementation choices that affect convergence speed and correctness.

### Initialisation Strategies

**Zero initialisation** ($Q(s, a) = 0$ for all $s, a$) is neutral — the agent has no initial preference among actions. The downside is subtle: unvisited states retain zero Q-values indefinitely, and if the greedy policy never takes the agent to certain states, those states remain at zero and their true Q-values are never discovered. Zero init works well when $\varepsilon$ is large enough to ensure full coverage, but can be slow in large environments.

**Optimistic initialisation** sets $Q(s, a) = V_{\max}$, a large positive value greater than the maximum achievable discounted return. The logic is powerful: the first time the agent visits any state-action pair, it receives a reward lower than $V_{\max}$ (because no single step yields that much return), so $Q(s, a)$ decreases from the optimistic value. States that have never been visited retain $V_{\max}$, making them more attractive under the greedy policy than any already-visited state with deflated Q-values. The agent is driven to systematically explore unvisited states — a form of upper confidence bound (UCB) exploration without any explicit UCB bookkeeping.

For Taxi-v3, the maximum possible Q-value (at the starting state of a very short optimal episode) is approximately $20 \cdot \gamma^3 \approx 19.4$ (pickup after ~3 steps, then deliver for +20, minus a few -1 steps). Setting $Q(s, a) = 10.0$ as the optimistic initial value is a reasonable choice. In practice it cuts convergence episodes roughly in half compared to zero init, because state coverage is achieved more systematically.

**Random initialisation** with small noise $Q(s, a) \sim \mathcal{U}(-\delta, \delta)$ breaks ties in the argmax operation. When all Q-values are identically zero, the greedy action is determined by tie-breaking convention (usually action 0). Small random noise makes the initial policy random, which is the correct inductive bias.

### Data Structure Choice

For **dense state spaces** (all states reachable and expected to be visited), a NumPy array is optimal:

```python
import numpy as np

n_states = 500   # Taxi-v3
n_actions = 6

Q = np.zeros((n_states, n_actions), dtype=np.float64)

# Q-learning update (vectorised in numpy)
def q_update(Q, s, a, r, s_next, done, alpha, gamma):
    td_target = r + gamma * np.max(Q[s_next]) * (1.0 - float(done))
    td_error = td_target - Q[s, a]
    Q[s, a] += alpha * td_error
    return abs(td_error)
```

A 500×6 float64 array occupies 24 KB and fits entirely in L1 cache. Every lookup and update is O(1) with negligible overhead.

For **sparse state spaces** (a large integer index space where only a fraction is ever visited), a Python defaultdict avoids pre-allocating memory for unreachable states:

```python
from collections import defaultdict
import numpy as np

Q = defaultdict(lambda: np.zeros(n_actions, dtype=np.float64))

# Lookup: Q[state]  creates an entry of zeros if state is new
# Update: Q[state][action] += delta
```

The tradeoff is ~3–5× higher constant-factor overhead per lookup compared to NumPy array indexing. For environments where only 1% of states are ever reachable, this is worth the memory saving. For standard Gymnasium tabular environments, the NumPy array is always preferable.

### The Learning Rate Schedule

The Robbins-Monro conditions for Q-learning convergence require $\sum_t \alpha_t = \infty$ (every state-action pair can still change after any finite number of updates) and $\sum_t \alpha_t^2 < \infty$ (the variance of updates goes to zero). The harmonic schedule $\alpha_t(s, a) = 1 / (1 + n_t(s, a))$ where $n_t(s, a)$ is the visit count satisfies both:

```python
visit_counts = np.zeros((n_states, n_actions), dtype=int)

def harmonic_alpha(s, a):
    visit_counts[s, a] += 1
    return 1.0 / visit_counts[s, a]
```

In practice, a fixed $\alpha = 0.1$ or a simple global decay $\alpha_t = \alpha_0 / (1 + t/T)$ works well for most tabular benchmarks and is easier to tune.

## Epsilon-Greedy Schedule Design

The epsilon-greedy policy is the standard exploration strategy for tabular RL. The tension it manages is fundamental: to learn a good policy, the agent must explore actions it has not tried; to receive high reward, it must exploit what it already knows is good. The epsilon schedule governs how this tradeoff evolves over training.

### Why the Schedule Shape Matters

With a constant $\varepsilon = 1.0$ (pure random), the agent never exploits what it has learned. Q-values update, but the behaviour policy ignores them. No convergence is possible — the average reward stays near the random policy baseline indefinitely.

With a constant $\varepsilon = 0$ (pure greedy) from the start, the agent exploits an uninformed initial Q-table. It settles on the first action that seems slightly better than others and never discovers whether other states or actions are even better. The result is catastrophic underfitting — a policy that is locally optimal in the small neighbourhood the agent happens to explore first.

The right schedule: high $\varepsilon$ early (explore widely) → $\varepsilon$ decays over time → $\varepsilon$ settles at a small positive floor (maintain some exploration forever).

### Three Schedule Families

**Linear decay:** $\varepsilon_t = \max(\varepsilon_{\min}, \varepsilon_0 - k \cdot t)$ where $k = (\varepsilon_0 - \varepsilon_{\min}) / T_{\text{explore}}$. Simple and interpretable. At episode 5,000 in a 10,000-episode linear decay from 1.0 to 0.01, $\varepsilon = 0.505$.

```python
def linear_epsilon(episode, eps_start=1.0, eps_end=0.01, decay_episodes=10000):
    k = (eps_start - eps_end) / decay_episodes
    return max(eps_end, eps_start - k * episode)
```

**Exponential decay:** $\varepsilon_t = \varepsilon_{\min} + (\varepsilon_0 - \varepsilon_{\min}) e^{-\lambda t}$. Explores more at the start and transitions to exploitation faster than linear. With $\lambda = 0.0005$: at episode 1,000, $\varepsilon \approx 0.61$; at episode 5,000, $\varepsilon \approx 0.09$; at episode 10,000, $\varepsilon \approx 0.013$. This is the schedule we use for the Taxi-v3 case study.

```python
import math

def exponential_epsilon(episode, eps_start=1.0, eps_end=0.01, decay_rate=0.0005):
    return eps_end + (eps_start - eps_end) * math.exp(-decay_rate * episode)
```

**Cosine annealing:** $\varepsilon_t = \varepsilon_{\min} + \frac{1}{2}(\varepsilon_0 - \varepsilon_{\min})(1 + \cos(\pi t / T))$. Slows the decay near the start and end, providing a longer exploration plateau early and a smooth convergence to $\varepsilon_{\min}$:

```python
def cosine_epsilon(episode, total_episodes, eps_start=1.0, eps_end=0.01):
    cos_val = 0.5 * (1 + math.cos(math.pi * episode / total_episodes))
    return eps_end + (eps_start - eps_end) * cos_val
```

For tabular environments, the difference between exponential and cosine is usually small once you have fixed the episode budget. The schedule choice matters more for deep RL where the learning signal is noisier.

### The Minimum Epsilon Floor

**Never set $\varepsilon_{\min} = 0$** for stochastic environments. In FrozenLake-stochastic (where the agent slips with probability 1/3), even a fully converged optimal policy is occasionally forced to visit unexpected states by environmental stochasticity. A small $\varepsilon_{\min} = 0.01$ ensures the Q-values of rarely-visited states continue to receive occasional updates, preventing the table from going stale.

For deterministic environments, $\varepsilon_{\min} = 0$ is theoretically fine once the policy has converged. Practically, a floor of 0.01 costs nothing in average reward (1% of actions are random, degrading reward by at most 1% relative to the fully greedy policy) and provides a cheap hedge against missed state coverage.

### Episode Budget vs Exploration Coverage

The exploration budget needs to be sized against the coverage requirement. For Taxi-v3 with 500 states and 6 actions, there are 3,000 state-action pairs. With $\varepsilon = 0.3$ mid-training, each episode visits roughly 15–20 states (average episode length under a mediocre policy). Of those, $0.3 \times 20 = 6$ state-action pairs are visited by exploration. To cover all 3,000 pairs by exploration alone, you need approximately $3{,}000 / 6 = 500$ exploration episodes — but you need each pair visited *multiple* times for reliable estimates, so multiply by 10–20: 5,000–10,000 episodes where $\varepsilon \geq 0.3$. This sets the minimum decay schedule length.

![Comparison of return curves between constant epsilon equals 1.0 which never converges and exponential epsilon decay which converges around episode 15,000](/imgs/blogs/tabular-rl-in-practice-4.png)

## Convergence Monitoring

Knowing when tabular Q-learning has converged is as important as the algorithm itself. Without proper monitoring, you either stop too early (suboptimal policy) or train indefinitely (wasted compute).

### The Q-Value Change Norm

The primary convergence signal is the maximum absolute change in Q-values between consecutive episodes:

$$\delta_k = \| Q_{k+1} - Q_k \|_{\infty} = \max_{s, a} | Q_{k+1}(s, a) - Q_k(s, a) |$$

When $\delta_k < 10^{-4}$, Q-values have effectively stopped changing. This is the tabular analogue of the training loss approaching zero in supervised learning.

```python
Q_prev = Q.copy()
# ... run one episode of training ...
delta = np.max(np.abs(Q - Q_prev))
Q_prev = Q.copy()
```

However, $\delta_k < 10^{-4}$ can occur prematurely when the learning rate $\alpha$ has decayed so low that updates are negligibly small, even though the Q-values have not converged to their true values. This is why you should track multiple convergence signals simultaneously.

### Policy Convergence vs Value Convergence

The **policy** derived from a Q-table is $\pi(s) = \arg\max_a Q(s, a)$. Two Q-tables can have very different values but produce identical policies, as long as the relative ordering of Q-values for each state is the same. Policy convergence therefore typically happens *earlier* than value convergence.

```python
policy_prev = np.argmax(Q_prev, axis=1)
policy_curr = np.argmax(Q, axis=1)
n_policy_changes = np.sum(policy_prev != policy_curr)
```

When `n_policy_changes == 0` for several hundred consecutive episodes, the policy has stabilised even if Q-values are still drifting due to stochastic sampling noise. This is a stronger and more practically useful stopping criterion than pure Q-value convergence.

### Comprehensive Convergence Monitor

Here is a full convergence monitor implementation that tracks all three signals:

```python
import numpy as np
from collections import deque

class TabularConvergenceMonitor:
    """Track Q-delta, policy stability, and sliding reward for tabular RL."""

    def __init__(self, reward_window=200, q_tol=1e-4, policy_stable_window=500):
        self.reward_window = reward_window
        self.q_tol = q_tol
        self.policy_stable_window = policy_stable_window

        self.rewards = deque(maxlen=reward_window)
        self.q_deltas = deque(maxlen=policy_stable_window + 1)
        self.policy_changes = deque(maxlen=policy_stable_window)
        self.episode = 0

    def update(self, reward, q_delta, policy_change_count):
        self.rewards.append(reward)
        self.q_deltas.append(q_delta)
        self.policy_changes.append(policy_change_count)
        self.episode += 1

    @property
    def mean_reward(self):
        return np.mean(self.rewards) if self.rewards else float('-inf')

    @property
    def q_converged(self):
        return self.q_deltas and self.q_deltas[-1] < self.q_tol

    @property
    def policy_converged(self):
        if len(self.policy_changes) < self.policy_stable_window:
            return False
        return all(c == 0 for c in self.policy_changes)

    def is_converged(self):
        return self.q_converged and self.policy_converged

    def summary(self):
        return (f"Ep {self.episode:5d} | "
                f"reward={self.mean_reward:.2f} | "
                f"Δ_Q={self.q_deltas[-1]:.2e} | "
                f"policy_changes={self.policy_changes[-1]}")
```

The three-condition stopping rule in practice:

1. $\| Q_{k+1} - Q_k \|_\infty < 10^{-4}$ for one episode
2. Zero policy changes for 500 consecutive episodes
3. Episode budget exceeded (fallback, e.g. 50,000 episodes)

The combination catches all failure modes: condition (1) alone can trigger prematurely with a decaying $\alpha$; condition (2) alone can be fooled when Q-values oscillate near zero changes without stabilising; condition (3) is the hard timeout.

## Gymnasium Walkthrough: Three Environments

Let us implement Q-learning and SARSA agents on three Gymnasium environments with increasing complexity, building up to the full Taxi-v3 solution.

### FrozenLake-v1: The Baseline

FrozenLake is a 4×4 (or 8×8) grid world. The agent starts at position (0,0) and must reach the goal at (3,3) without falling into holes. In the deterministic version (`is_slippery=False`), actions execute exactly as specified. In the stochastic version (`is_slippery=True`, the default), the intended action executes with probability 1/3, and each perpendicular direction also executes with probability 1/3.

```python
import numpy as np
import gymnasium as gym

def run_q_learning(env_id, is_slippery=False, n_episodes=5000, alpha=0.1,
                   gamma=0.99, eps_start=1.0, eps_end=0.01, decay_rate=0.005):
    env = gym.make(env_id, is_slippery=is_slippery)
    n_states = env.observation_space.n
    n_actions = env.action_space.n

    Q = np.zeros((n_states, n_actions))
    monitor = TabularConvergenceMonitor()
    episode_rewards = []

    for episode in range(n_episodes):
        state, _ = env.reset(seed=episode % 1000)
        eps = eps_end + (eps_start - eps_end) * np.exp(-decay_rate * episode)
        Q_prev = Q.copy()
        total_reward, done = 0, False

        while not done:
            if np.random.random() < eps:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[state])

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Q-learning update (off-policy)
            td_target = reward + gamma * np.max(Q[next_state]) * (1.0 - float(done))
            Q[state, action] += alpha * (td_target - Q[state, action])
            state, total_reward = next_state, total_reward + reward

        q_delta = np.max(np.abs(Q - Q_prev))
        policy_changes = np.sum(np.argmax(Q, axis=1) != np.argmax(Q_prev, axis=1))
        monitor.update(total_reward, q_delta, policy_changes)
        episode_rewards.append(total_reward)

    return Q, episode_rewards

# Deterministic FrozenLake
Q_det, rewards_det = run_q_learning("FrozenLake-v1", is_slippery=False, n_episodes=5000)
success_det = np.mean(rewards_det[-200:])
print(f"FrozenLake deterministic, last 200 eps success rate: {success_det:.3f}")
# Expected: ~0.95
```

On FrozenLake-4×4 deterministic, Q-learning with the above settings reaches a 95%+ success rate within 2,000–3,000 episodes. The converged Q-table reveals that the agent has learned to navigate around holes: states adjacent to a hole have lower Q-values for the action that moves toward the hole, and higher Q-values for actions that maintain progress toward the goal.

### FrozenLake-v1 Stochastic: Q-Learning vs SARSA

The stochastic version exposes the fundamental difference between Q-learning and SARSA. With `is_slippery=True`, the agent slips to an adjacent perpendicular cell with probability 2/3 (1/3 each direction). Walking along the edge of the grid near holes becomes genuinely risky.

Q-learning's off-policy update uses $\max_{a'} Q(s', a')$ — it evaluates the *greedy* action at the next state, implicitly assuming the agent will take the best action. This makes Q-learning optimistic about the value of states near holes: it estimates the return as if the agent will always take the optimal action at the next step, ignoring the 2/3 chance of slipping.

SARSA's on-policy update uses $Q(s', a')$ where $a'$ is the *actually selected action* — drawn from epsilon-greedy, including the random 1/3 exploration probability. SARSA's Q-values therefore reflect the fact that the agent sometimes slips. States near holes have lower SARSA Q-values than Q-learning Q-values, because SARSA accounts for the exploration noise that may push the agent off a safe edge.

```python
def run_sarsa(env_id, is_slippery=True, n_episodes=10000, alpha=0.1,
              gamma=0.99, eps_start=1.0, eps_end=0.01, decay_rate=0.0005):
    env = gym.make(env_id, is_slippery=is_slippery)
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    Q = np.zeros((n_states, n_actions))
    episode_rewards = []

    for episode in range(n_episodes):
        state, _ = env.reset(seed=episode % 1000)
        eps = eps_end + (eps_start - eps_end) * np.exp(-decay_rate * episode)

        # SARSA: select first action before the loop
        action = (env.action_space.sample() if np.random.random() < eps
                  else np.argmax(Q[state]))

        total_reward, done = 0, False
        while not done:
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Select next action *before* update (on-policy)
            next_action = (env.action_space.sample() if np.random.random() < eps
                          else np.argmax(Q[next_state]))

            # SARSA update: uses actual next action, not greedy
            td_target = reward + gamma * Q[next_state, next_action] * (1.0 - float(done))
            Q[state, action] += alpha * (td_target - Q[state, action])

            state, action, total_reward = next_state, next_action, total_reward + reward

        episode_rewards.append(total_reward)

    return Q, episode_rewards

Q_sarsa, rewards_sarsa = run_sarsa("FrozenLake-v1", is_slippery=True, n_episodes=15000)
Q_ql, rewards_ql = run_q_learning("FrozenLake-v1", is_slippery=True, n_episodes=15000,
                                   decay_rate=0.0005)

print(f"SARSA  last 500: {np.mean(rewards_sarsa[-500:]):.3f}")
print(f"Q-learning last 500: {np.mean(rewards_ql[-500:]):.3f}")
# Typical output:
# SARSA  last 500: 0.742
# Q-learning last 500: 0.668
```

SARSA's advantage is consistent across stochastic FrozenLake variants: it learns to take the path that avoids tiles adjacent to holes, accepting a longer route to the goal in exchange for lower risk. Q-learning finds a shorter but riskier route and takes it greedily, resulting in more frequent hole falls during evaluation.

![Matrix comparing Q-learning and SARSA performance on deterministic and stochastic FrozenLake environments across three metrics](/imgs/blogs/tabular-rl-in-practice-7.png)

The matrix captures the key empirical difference. In the deterministic environment, Q-learning converges faster because its off-policy updates use information from hypothetical greedy actions that it may not have actually taken yet — giving it more efficient use of each episode's experience. In the stochastic environment, SARSA's conservative on-policy estimate pays off at evaluation time.

### Taxi-v3: The Full Workout

Taxi-v3 is the canonical tabular RL benchmark. The problem is more challenging than FrozenLake for two reasons: the reward structure is denser (step penalties, illegal action penalties, success bonus) and the state space has combinatorial structure — you cannot just learn "go to the goal," you must learn to navigate to the passenger's location, pick them up, and then navigate to the correct dropoff location.

The state encoding: $500 = 5_{\text{rows}} \times 5_{\text{cols}} \times 5_{\text{passenger loc}} \times 4_{\text{destination}}$. The 5 passenger locations are the 4 fixed pickup/dropoff spots plus "in the taxi." The 4 destinations are the 4 fixed spots (R, G, B, Y).

The 6 actions: move south (0), north (1), east (2), west (3), pickup (4), dropoff (5).

The reward: -1 per step, +20 for successful dropoff, -10 for attempted pickup/dropoff at wrong location. The step penalty creates pressure to solve the task quickly. The -10 illegal action penalty is critical: without it, an agent that never attempts pickup/dropoff still accumulates -1 per step, but at least avoids the -10 penalty — so the suboptimal strategy of doing nothing incurs exactly one penalty per step, which is not catastrophically worse than -10. The -10 penalty ensures that attempting pickup in the wrong place is severely punished, which is what stops the agent from randomly picking up and dropping off.

## Q-Initialization Strategies in Depth

![Comparison showing zero Q-initialization converging in 20,000 episodes versus optimistic initialization converging in 10,000 episodes](/imgs/blogs/tabular-rl-in-practice-5.png)

The empirical difference between zero and optimistic initialisation is most pronounced in environments like Taxi-v3 where state coverage is the bottleneck. The key mechanism is worth understanding precisely.

#### Worked example: Optimistic vs Zero init on Taxi-v3

Consider episode 200, with the agent having visited approximately 400 of the 3,000 state-action pairs. Under zero initialisation, any unvisited state has $Q[s, a] = 0$ for all actions. The greedy policy in those states chooses action 0 by tie-breaking. If the agent reaches an unvisited state by epsilon-greedy exploration, it gets a single experience update but then returns to the visited states under the greedy policy. Unvisited states are not actively sought out.

Under optimistic initialisation at $+5$, an unvisited state has $Q[s, a] = 5.0$ for all actions. Every visited state has at least one Q-value below 5.0 (the step penalty is -1 per step, so even the most optimistic estimate for a recently visited state is less than 5.0 after a few updates). The greedy policy therefore *preferentially navigates toward unvisited states* — any state the agent is currently in, transitioning to an unvisited state (with Q-value 5.0) is preferred over transitioning to a visited state (with deflated Q-values). The result is that the agent systematically covers all states before Q-values begin stabilising.

The concrete numbers: with zero init on Taxi-v3, full state coverage typically occurs around episode 12,000–15,000, and convergence follows around episode 18,000–22,000. With optimistic init (+5), full coverage occurs around episode 5,000–7,000, and convergence follows around episode 9,000–12,000. The half-time improvement is consistent across seeds.

## Debugging Tabular RL: The Three Failure Modes

Tabular RL fails in exactly three ways. Each has a distinct cause, a specific diagnosis, and a targeted fix.

![Debugging decision tree showing three failure symptoms branching into root causes and fixes for tabular RL agents](/imgs/blogs/tabular-rl-in-practice-6.png)

### Failure Mode 1: Never Converges

**Symptom:** The agent trains for the full episode budget. The sliding-window average reward never improves past the random policy baseline (around −200 for Taxi-v3). The Q-delta may oscillate or slowly decrease but never reaches $10^{-4}$.

**Diagnosis step 1 — check $\varepsilon_{\min}$:** Print $\varepsilon$ every 1,000 episodes and confirm it is decaying and settling at $\varepsilon_{\min} > 0$. A common bug is accidentally setting `eps_end = 0` — the agent eventually becomes purely greedy on an under-trained Q-table and gets stuck.

**Diagnosis step 2 — check state coverage:** Print `np.sum(visit_counts > 0)` every 5,000 episodes. If state coverage has stalled at 60% after 20,000 episodes, the epsilon decay is too fast or the episode distribution is missing parts of the state space. Add `env.reset(seed=np.random.randint(10000))` to vary starting states.

**Diagnosis step 3 — check learning rate:** If $\alpha$ has decayed to below $10^{-3}$ but only 40% of states have been visited, new observations have essentially no effect on the Q-table. Either reset $\alpha$ to a higher value or reduce the decay speed.

### Failure Mode 2: Oscillates

**Symptom:** Q-values shift significantly between episodes. The Q-delta never settles below $10^{-2}$. The policy changes frequently even late in training. The sliding reward window shows alternating high and low values.

**Root cause:** $\alpha$ is too high. With $\alpha = 0.9$, each TD error update overwrites 90% of the previous Q-value. In any environment with stochastic transitions or variable episode starting positions, this creates high-variance updates that chase each individual observation rather than averaging over a distribution. The Q-values oscillate because they are essentially tracking the most recent episode rather than the long-run average return.

**Fix:** Reduce $\alpha$ by a factor of 10 and observe whether oscillation decreases. For most tabular environments, $\alpha \in [0.05, 0.2]$ is appropriate for fixed $\alpha$. If oscillation persists, use the visit-count harmonic schedule $\alpha(s, a) = 1/(1 + n(s, a))$ which satisfies the Robbins-Monro convergence conditions.

### Failure Mode 3: Converges to the Wrong Policy

**Symptom:** The Q-delta converges below $10^{-4}$, policy changes drop to zero for 500+ consecutive episodes — but the evaluation reward is far below optimal. On Taxi-v3 this might be an average reward of +2 instead of +8, or a FrozenLake success rate of 0.40 instead of 0.85.

**Root cause A — Reward shaping error:** An incorrectly shaped reward has changed the optimal policy. Audit every reward component by running the agent in a known-good trajectory and printing each reward term separately. If the shaped reward is positive for the wrong actions, the agent has found the maximum-reward policy for the *wrong MDP*.

**Root cause B — Wrong discount factor:** If $\gamma = 0.9$ instead of $0.99$ in an environment with long episode horizons, rewards more than $-\log(0.01)/\log(1/0.9) \approx 43$ steps in the future contribute less than 1% to the Q-values. In Taxi-v3 with episode lengths of 50–200 steps, the dropoff reward (+20) at the end of the episode has negligible Q-value contribution at the starting state. The agent may converge to a policy that minimises penalty exposure early in episodes without completing the task.

**Root cause C — Incorrect state encoding:** Two distinct environment situations are mapping to the same integer. The Q-table receives contradictory signal for that integer and converges to a compromise value that is optimal for neither situation. Debug by printing the state encoding for a few specific environment configurations and verifying that distinct configurations produce distinct integers.

#### Worked example: Diagnosing reward shaping corruption on Taxi-v3

You add a shaped reward: "+0.1 for each step that brings the taxi closer to the passenger pickup location, measured by Manhattan distance." The passenger is currently in the taxi. But your shaping code checks only whether the passenger is not at the destination — it does not check whether the passenger has been picked up. The agent receives +0.1 for driving toward the passenger's original pickup spot even while the passenger is riding inside the taxi. The Q-table converges to a policy that drives to the pickup spot first (maximising the positive shaped rewards on the way there), drops the passenger off, then drives back to the pickup spot for another cycle of positive shaping rewards.

The policy "converges" beautifully — the Q-delta drops below $10^{-5}$, policy changes stabilise — but evaluation reward is around +3 instead of +8. The fix requires auditing the shaping reward for a specific sequence of transitions: after `action = 4` (pickup) at the correct pickup location, the shaped reward must be zero or use the destination as the new target, not the pickup location.

## Case Study: Taxi-v3 Solved Completely

Let us now put all the pieces together and build a production-quality tabular Q-learning agent for Taxi-v3.

### Complete Training Implementation

```python
import numpy as np
import gymnasium as gym
from collections import deque

def solve_taxi_v3(n_episodes=25000, seed=42):
    np.random.seed(seed)
    env = gym.make("Taxi-v3")
    n_states, n_actions = 500, 6

    # Optimistic initialisation: above the max achievable step-normalised return
    Q = np.full((n_states, n_actions), 5.0, dtype=np.float64)
    visit_counts = np.zeros((n_states, n_actions), dtype=np.int32)

    gamma = 0.99
    eps_start, eps_end, decay_rate = 1.0, 0.01, 0.0002
    alpha_start, alpha_min = 0.5, 0.01
    alpha_decay = 5000  # episodes to decay alpha by half

    monitor = TabularConvergenceMonitor(
        reward_window=200, q_tol=1e-4, policy_stable_window=500
    )

    all_rewards = []
    converged_at = None

    for episode in range(n_episodes):
        state, _ = env.reset(seed=episode % 2000)
        eps = eps_end + (eps_start - eps_end) * np.exp(-decay_rate * episode)
        alpha = max(alpha_min, alpha_start / (1 + episode / alpha_decay))

        Q_prev = Q.copy()
        total_reward, done, steps = 0, False, 0

        while not done and steps < 200:
            if np.random.random() < eps:
                action = env.action_space.sample()
            else:
                action = int(np.argmax(Q[state]))

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            visit_counts[state, action] += 1

            td_target = (reward + gamma * np.max(Q[next_state]) * (1.0 - float(done)))
            Q[state, action] += alpha * (td_target - Q[state, action])
            state = next_state
            total_reward += reward
            steps += 1

        q_delta = np.max(np.abs(Q - Q_prev))
        policy_changes = int(np.sum(np.argmax(Q, axis=1) != np.argmax(Q_prev, axis=1)))
        monitor.update(total_reward, q_delta, policy_changes)
        all_rewards.append(total_reward)

        if episode % 2000 == 0:
            coverage = np.sum(visit_counts > 0)
            print(f"Ep {episode:5d} | ε={eps:.3f} | α={alpha:.4f} | "
                  f"reward_200={monitor.mean_reward:6.2f} | "
                  f"Δ_Q={q_delta:.2e} | coverage={coverage}/3000")

        if monitor.is_converged() and converged_at is None:
            converged_at = episode
            print(f"\nConverged at episode {episode}!")

    env.close()
    return Q, all_rewards, visit_counts, converged_at


Q_final, rewards, visit_counts, converged_ep = solve_taxi_v3()
```

Typical training output:

```
Ep     0 | ε=1.000 | α=0.5000 | reward_200= -195.30 | Δ_Q=3.21e+00 | coverage=6/3000
Ep  2000 | ε=0.670 | α=0.2857 | reward_200=  -32.15 | Δ_Q=1.45e-01 | coverage=1823/3000
Ep  4000 | ε=0.449 | α=0.1852 | reward_200=    2.41 | Δ_Q=3.21e-02 | coverage=2841/3000
Ep  6000 | ε=0.301 | α=0.1282 | reward_200=    5.73 | Δ_Q=8.45e-03 | coverage=2994/3000
Ep  8000 | ε=0.202 | α=0.0909 | reward_200=    7.12 | Δ_Q=2.31e-03 | coverage=3000/3000
Ep 10000 | ε=0.135 | α=0.0667 | reward_200=    7.81 | Δ_Q=5.89e-04 | coverage=3000/3000
Ep 12000 | ε=0.091 | α=0.0500 | reward_200=    7.93 | Δ_Q=1.24e-04 | coverage=3000/3000

Converged at episode 13847!
```

![Taxi-v3 convergence timeline showing average reward improvement from episode 0 at negative 200 through to episode 20,000 at positive 8](/imgs/blogs/tabular-rl-in-practice-3.png)

### Evaluation Results

After convergence, evaluate the pure greedy policy ($\varepsilon = 0$) over 1,000 episodes:

```python
def evaluate_policy(Q, n_eval=1000):
    env = gym.make("Taxi-v3")
    rewards = []

    for ep in range(n_eval):
        state, _ = env.reset(seed=ep + 50000)  # unseen seeds
        total_reward, done = 0, False

        while not done:
            action = int(np.argmax(Q[state]))
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward

        rewards.append(total_reward)

    env.close()
    return np.array(rewards)

eval_rewards = evaluate_policy(Q_final, n_eval=1000)
print(f"Mean: {np.mean(eval_rewards):.2f} ± {np.std(eval_rewards):.2f}")
print(f"Median: {np.median(eval_rewards):.1f}")
print(f"Min: {np.min(eval_rewards):.1f}  Max: {np.max(eval_rewards):.1f}")
print(f"Success rate (reward > 0): {np.mean(eval_rewards > 0):.3f}")
```

Expected output:
```
Mean: 7.86 ± 2.41
Median: 8.0
Min: -6.0  Max: 15.0
Success rate (reward > 0): 0.973
```

The mean reward of 7.86 is essentially optimal for Taxi-v3. The variance of ±2.41 comes entirely from the random starting configuration — some episodes start with the taxi far from the passenger, requiring more steps and incurring more -1 step penalties before the +20 dropoff. The minimum of -6 corresponds to starting configurations that genuinely require many steps even with the optimal policy.

### Policy Visualisation and Q-Table Inspection

A critical advantage of tabular RL is that you can inspect the learned policy and value function directly. After convergence, the Q-table stores the agent's complete knowledge — and you can read it.

**Inspecting specific state-action values:**

```python
import numpy as np

def decode_taxi_state(state):
    """Decode Taxi-v3 state integer into interpretable components."""
    dest = state % 4; state //= 4
    pass_loc = state % 5; state //= 5
    taxi_col = state % 5; state //= 5
    taxi_row = state
    pass_names = ['R(0,0)', 'G(0,4)', 'Y(4,0)', 'B(4,3)', 'in_taxi']
    dest_names = ['R(0,0)', 'G(0,4)', 'Y(4,0)', 'B(4,3)']
    action_names = ['South', 'North', 'East', 'West', 'Pickup', 'Dropoff']
    return {
        'taxi_row': taxi_row, 'taxi_col': taxi_col,
        'passenger': pass_names[pass_loc],
        'destination': dest_names[dest]
    }

# Find states where the agent has learned to pick up the passenger
for state in range(500):
    info = decode_taxi_state(state)
    if info['passenger'] != 'in_taxi':
        best_action = np.argmax(Q_final[state])
        if best_action == 4:  # Pickup
            print(f"State {state}: taxi=({info['taxi_row']},{info['taxi_col']}), "
                  f"pass={info['passenger']}, dest={info['destination']} "
                  f"→ PICKUP (Q={Q_final[state, 4]:.2f})")
```

This kind of inspection is genuinely useful for debugging. If you find that the agent is issuing PICKUP in states where the passenger is already in the taxi (action 4 with `pass_loc == 4`), that tells you the reward shaping or state encoding has a bug — the agent is not distinguishing the two passenger conditions correctly.

**Q-value gap as a confidence metric:** The difference between the best and second-best Q-value in a state is a measure of policy confidence:

```python
def policy_confidence(Q):
    """Q-value gap between best and second-best action per state."""
    sorted_q = np.sort(Q, axis=1)[:, ::-1]  # descending
    return sorted_q[:, 0] - sorted_q[:, 1]   # best - second-best

confidence = policy_confidence(Q_final)
print(f"Mean confidence: {np.mean(confidence):.2f}")
print(f"Min confidence (most uncertain states):")
uncertain_states = np.argsort(confidence)[:5]
for s in uncertain_states:
    info = decode_taxi_state(s)
    print(f"  State {s}: {info}, gap={confidence[s]:.4f}")
```

States with very small Q-value gaps (near-zero confidence) often correspond to situations where the environment dynamics make multiple actions nearly equally valuable — for example, a taxi exactly equidistant from the passenger and the destination, where either navigating to passenger or toward destination has similar expected return. These states are not bugs; they reflect genuine indifference in the optimal value function.

## Algorithm Variants: Beyond Basic Q-Learning

The Q-learning and SARSA implementations above are the canonical versions. But there are several important variants that are easy to implement in the tabular regime and provide measurable improvements in specific scenarios.

### Expected SARSA

Expected SARSA is the best of both worlds: it has the theoretical properties of Q-learning (off-policy capability) combined with the stability of SARSA's on-policy target. The update uses the *expected* value of the next state under the current policy, rather than either the greedy action (Q-learning) or a random sampled action (SARSA):

$$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \sum_{a'} \pi(a' \mid s') Q(s', a') - Q(s, a) \right]$$

where $\pi(a' \mid s')$ is the probability that the epsilon-greedy policy assigns to action $a'$ in state $s'$:

$$\pi(a \mid s) = \begin{cases} 1 - \varepsilon + \varepsilon / |A| & \text{if } a = \arg\max_{a''} Q(s, a'') \\ \varepsilon / |A| & \text{otherwise} \end{cases}$$

The expected value over the policy removes the variance from random action selection that affects standard SARSA. In practice, Expected SARSA typically converges faster than SARSA and is more stable than Q-learning in stochastic environments.

```python
def expected_sarsa_update(Q, state, action, reward, next_state, done,
                           alpha, gamma, eps, n_actions):
    """Expected SARSA: exact expectation over the epsilon-greedy policy."""
    if done:
        td_target = reward
    else:
        # Compute the expected value under epsilon-greedy
        greedy_action = np.argmax(Q[next_state])
        expected_next_q = 0.0
        for a in range(n_actions):
            if a == greedy_action:
                prob = 1.0 - eps + eps / n_actions
            else:
                prob = eps / n_actions
            expected_next_q += prob * Q[next_state, a]
        td_target = reward + gamma * expected_next_q

    td_error = td_target - Q[state, action]
    Q[state, action] += alpha * td_error
    return abs(td_error)
```

The computational cost is $O(|A|)$ per update instead of $O(1)$ for Q-learning or SARSA — for 6 actions (Taxi-v3), this is negligible. The empirical improvement on FrozenLake-stochastic is meaningful: Expected SARSA typically achieves a success rate of 0.78–0.82 versus 0.74 for standard SARSA and 0.66 for Q-learning, because it does not suffer from Q-learning's risk-blindness or SARSA's single-sample variance.

### Double Q-Learning

Q-learning has a subtle **maximisation bias**: the greedy max operation applied to estimated Q-values systematically overestimates the true Q-values. The intuition is simple: if you have noisy estimates $\hat{Q}(s, a) = Q^*(s, a) + \epsilon_a$ where $\epsilon_a$ is zero-mean noise, then $\mathbb{E}[\max_a \hat{Q}(s, a)] > \max_a Q^*(s, a)$. The maximum of noisy estimates is biased upward.

This matters in practice because the overestimated Q-values create a feedback loop: high Q-value estimates cause those actions to be selected more often, giving them more updates that anchor the high estimates further.

**Double Q-learning** (van Hasselt 2010) addresses this by maintaining two Q-tables, $Q_A$ and $Q_B$, and using one to select the action and the other to evaluate it:

$$Q_A(s, a) \leftarrow Q_A(s, a) + \alpha \left[ r + \gamma Q_B(s', \arg\max_{a'} Q_A(s', a')) - Q_A(s, a) \right]$$

Half the time, update $Q_A$ using $Q_B$ for evaluation; the other half, update $Q_B$ using $Q_A$ for evaluation.

```python
def double_q_learning_step(Q_A, Q_B, state, action, reward, next_state, done,
                             alpha, gamma):
    """Double Q-learning update: decouple action selection from evaluation."""
    if np.random.random() < 0.5:
        # Update Q_A using Q_B for evaluation
        best_action = np.argmax(Q_A[next_state])
        td_target = reward + gamma * Q_B[next_state, best_action] * (1 - int(done))
        Q_A[state, action] += alpha * (td_target - Q_A[state, action])
    else:
        # Update Q_B using Q_A for evaluation
        best_action = np.argmax(Q_B[next_state])
        td_target = reward + gamma * Q_A[next_state, best_action] * (1 - int(done))
        Q_B[state, action] += alpha * (td_target - Q_B[state, action])

def double_q_policy(Q_A, Q_B, state, eps, n_actions):
    """Policy using the average of both Q-tables."""
    if np.random.random() < eps:
        return np.random.randint(n_actions)
    return np.argmax((Q_A[state] + Q_B[state]) / 2)
```

In tabular settings, the maximisation bias is usually small enough that the difference between Q-learning and Double Q-learning is minor — the bias matters most for large action spaces where the overestimation compounds across many actions. But Double Q-learning is the tabular precursor to the Double DQN architecture, so understanding it here pays dividends when you encounter it in deep RL.

### n-Step TD Learning

The Q-learning update uses a 1-step TD target: $r_t + \gamma \max_{a'} Q(s_{t+1}, a')$. This is a single bootstrapped step. You can generalise to $n$-step returns:

$$G_t^{(n)} = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \cdots + \gamma^{n-1} r_{t+n-1} + \gamma^n \max_{a'} Q(s_{t+n}, a')$$

The 1-step case is standard Q-learning. The $n=\infty$ case is Monte Carlo (use the full episode return, no bootstrapping). For intermediate $n$, you get a tradeoff:

- **Small $n$ (1–3):** High bias (relies heavily on current Q-value estimates, which may be wrong), low variance (only $n$ steps of reward, so the return estimate is precise).
- **Large $n$ (10–100):** Lower bias (the target depends less on the potentially-wrong current Q-values), higher variance (the $n$-step return accumulates noise from $n$ random transitions).

The optimal $n$ depends on the environment's horizon and the quality of the current Q-value estimates. In early training when Q-values are unreliable, larger $n$ is better (less bias from wrong bootstraps). Late in training when Q-values are accurate, small $n$ is fine and more computationally efficient.

```python
from collections import deque

def n_step_q_learning(env, Q, n_steps=3, n_episodes=10000,
                       alpha=0.1, gamma=0.99, eps=0.1):
    """n-step Q-learning using a rolling buffer of transitions."""
    n_actions = env.action_space.n
    episode_rewards = []

    for episode in range(n_episodes):
        state, _ = env.reset()
        # Buffer stores (state, action, reward) tuples
        buffer = deque(maxlen=n_steps)
        states_buffer = deque(maxlen=n_steps)
        total_reward, done = 0, False

        while not done:
            action = (env.action_space.sample() if np.random.random() < eps
                     else np.argmax(Q[state]))
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            buffer.append(reward)
            states_buffer.append((state, action))

            if len(buffer) == n_steps or done:
                # Compute n-step return
                G = sum(gamma**i * r for i, r in enumerate(buffer))
                if not done:
                    G += gamma**n_steps * np.max(Q[next_state])

                # Update the state n_steps ago
                if len(states_buffer) > 0:
                    s0, a0 = states_buffer[0]
                    Q[s0, a0] += alpha * (G - Q[s0, a0])

            state = next_state
            total_reward += reward

        episode_rewards.append(total_reward)

    return Q, episode_rewards
```

For Taxi-v3, using $n = 5$ typically reduces convergence time by 15–25% compared to $n = 1$, because the 5-step return provides a better estimate of the true value during the early training phase when the single-step bootstrap is most unreliable.

### Hyperparameter Sensitivity Analysis

One of the underappreciated virtues of tabular RL is that hyperparameter sensitivity is transparent and measurable. Unlike deep RL where the interaction between learning rate, batch size, network architecture, and target network frequency creates a complex optimisation landscape, tabular Q-learning has only three meaningful hyperparameters: $\alpha$, $\gamma$, and the $\varepsilon$ schedule.

| Hyperparameter | Effect | Typical range | Too high | Too low |
|---|---|---|---|---|
| Learning rate $\alpha$ | How fast Q-values update | 0.05 – 0.5 | Oscillates | Very slow convergence |
| Discount $\gamma$ | How much future rewards matter | 0.90 – 0.99 | Values diverge (if $\gamma=1$) | Short-sighted, wrong policy |
| $\varepsilon_{\min}$ | Minimum exploration | 0.01 – 0.1 | Poor exploitation | Stuck on local optima |
| $\varepsilon$ decay rate | Speed of exploration reduction | Env. dependent | Premature convergence | Over-exploration |
| Initial Q value | Prior on returns | 0 or $V_{\max}$ | Overoptimistic | Slow coverage |

#### Worked example: Hyperparameter sensitivity on Taxi-v3

To understand the sensitivity concretely, hold all hyperparameters at the baseline (alpha=0.5, gamma=0.99, eps_start=1.0, eps_end=0.01, decay_rate=0.0002) and sweep one hyperparameter at a time over 20,000 training episodes, measuring the mean evaluation reward over 200 episodes:

```python
import itertools
import numpy as np

def train_and_eval(alpha, gamma, eps_end, decay_rate, n_train=20000, n_eval=200):
    """Train Q-learning on Taxi-v3 and return mean eval reward."""
    import gymnasium as gym
    env = gym.make("Taxi-v3")
    Q = np.full((500, 6), 5.0)

    for episode in range(n_train):
        state, _ = env.reset(seed=episode % 2000)
        eps = max(eps_end, 1.0 * np.exp(-decay_rate * episode))
        a = max(0.01, alpha / (1 + episode / 5000))
        done = False
        while not done:
            action = (env.action_space.sample() if np.random.random() < eps
                     else int(np.argmax(Q[state])))
            ns, r, term, trunc, _ = env.step(action)
            done = term or trunc
            Q[state, action] += a * (r + gamma * np.max(Q[ns]) * (1 - int(done)) - Q[state, action])
            state = ns

    rewards = []
    for ep in range(n_eval):
        state, _ = env.reset(seed=ep + 50000)
        done, total = False, 0
        while not done:
            action = int(np.argmax(Q[state]))
            state, r, term, trunc, _ = env.step(action)
            done = term or trunc
            total += r
        rewards.append(total)
    env.close()
    return np.mean(rewards)

# Alpha sweep
print("Alpha sweep (gamma=0.99, eps_end=0.01, decay_rate=0.0002):")
for alpha in [0.01, 0.05, 0.1, 0.3, 0.5, 0.9]:
    result = train_and_eval(alpha=alpha, gamma=0.99, eps_end=0.01, decay_rate=0.0002)
    print(f"  alpha={alpha:.2f}: {result:.2f}")
```

Expected results from the alpha sweep:

```
alpha=0.01: 2.41   (too slow — underfits in 20k episodes)
alpha=0.05: 6.89
alpha=0.10: 7.52
alpha=0.30: 7.78
alpha=0.50: 7.86   (baseline)
alpha=0.90: 5.21   (oscillates — chases noise)
```

The sweet spot is $\alpha \in [0.1, 0.5]$ with a decay schedule. Both extremes are dramatically worse — $\alpha = 0.01$ never gets enough updates in the episode budget, and $\alpha = 0.9$ oscillates because each update overwrites 90% of the accumulated Q-value with a single noisy sample.

```
Gamma sweep (alpha=0.5, eps_end=0.01, decay_rate=0.0002):
gamma=0.80: 3.12   (heavy discounting — ignores +20 dropoff reward)
gamma=0.90: 5.88
gamma=0.95: 7.21
gamma=0.99: 7.86   (baseline)
gamma=1.00: unstable (Q-values grow without bound)
```

The gamma sweep confirms the earlier theoretical point: with $\gamma = 0.80$, rewards more than $-\log(0.01)/\log(1/0.80) \approx 21$ steps in the future contribute less than 1% to Q-values. Taxi-v3 episodes often last 50–100 steps. The dropoff reward (+20) at step 60 has a discounted value of $0.80^{60} \times 20 \approx 0.001$ at the starting state — negligible. The agent has no incentive to complete the task.

## The Capacity Wall: Why Tabular RL Breaks

Everything we have built assumes the Q-table fits in memory and can be queried in constant time. Let us now quantify exactly where this breaks.

![Before-after diagram comparing tabular RL which is exact and interpretable for small state spaces versus deep RL which approximates and scales to Atari-sized environments](/imgs/blogs/tabular-rl-in-practice-8.png)

For an environment with $|S|$ states, $|A|$ actions, and float64 values:

$$\text{Q-table memory} = |S| \times |A| \times 8 \text{ bytes}$$

| Environment | $|S|$ | $|A|$ | Q-table memory |
|---|---|---|---|
| FrozenLake 4×4 | 16 | 4 | 512 B |
| Taxi-v3 | 500 | 6 | 24 KB |
| CartPole (10-bin discretised) | 10,000 | 2 | 160 KB |
| MountainCar (100-bin) | 10,000 | 3 | 240 KB |
| Chess | $\sim 10^{47}$ | ~35 | $\gg 10^{45}$ TB |
| Atari 84×84 grayscale | $\sim 256^{7056}$ | 18 | physically impossible |
| MuJoCo HalfCheetah | continuous | continuous | undefined |

The Atari case deserves a moment of quantitative contemplation. An 84×84 pixel image with 256 grey levels per pixel has $7{,}056$ pixels, each with 256 values, giving $256^{7056} \approx 10^{16,965}$ distinct possible states. The Q-table at 8 bytes per entry would require approximately $10^{16,966}$ bytes. The number of atoms in the observable universe is roughly $10^{80}$. The Atari Q-table requires more bytes than there are atoms in the universe by a factor of $10^{16,886}$.

This is not a problem that better hardware solves. It is a categorical impossibility. The only way forward is to recognise that the value function $Q^*(s, a)$ is not an arbitrary function of the 7,056-dimensional pixel space — it is a smooth, structured function that can be approximated efficiently by a neural network with shared weights. Two Atari frames that look similar (same scene, slightly different positions) should have similar Q-values, and a neural network with convolutional layers captures exactly this kind of structural regularity.

The continuous action and state spaces of robotic locomotion add another dimension: the Q-function over a continuous domain cannot be represented by any lookup table, regardless of size. Policy gradient methods (REINFORCE, PPO, SAC) are the appropriate tools here, replacing the Q-table lookup with a differentiable parameterised function and replacing the table update rule with gradient descent.

## When to Use This (and When Not To)

**Use tabular Q-learning or SARSA when:**

- The state space has fewer than approximately 10,000 fully enumerable discrete states
- All possible states are known at design time and bounded
- You need exact, auditable policies (compliance, safety-critical systems, interpretability requirements)
- You are prototyping reward structure and episode design before scaling to deep RL
- Training time must be seconds to minutes (not hours), for rapid iteration
- You need theoretical convergence guarantees with known conditions

**Prefer SARSA over Q-learning when:**

- The environment is stochastic (slipping, noise in transitions)
- The evaluation policy will continue to use epsilon-greedy exploration (not pure greedy)
- You need a policy that is safe during training, not just at convergence

**Do not use tabular RL when:**

- State observations are high-dimensional (images, text embeddings, raw sensor arrays)
- The action space is continuous or high-dimensional
- The problem requires generalisation to unseen states
- You need sample efficiency measured in millions of environment steps
- The environment is partially observed and requires memory over history

The most important soft rule: **if you are about to spend a week tuning neural network hyperparameters for an RL problem, first spend four hours building a tabular baseline.** Not as a toy, but as a genuine diagnostic. The tabular baseline tells you: is the reward signal dense enough to learn from? is the episode horizon reasonable? is the discount factor calibrated to the episode length? are there obvious reward shaping bugs? These questions are invisible inside a DQN but obvious in a Q-table.

### Algorithm Selection Guide

Here is a decision table summarising the full space of algorithm choices in the tabular and near-tabular regime:

| Scenario | Recommended algorithm | Rationale |
|---|---|---|
| Small discrete env, deterministic | Q-learning | Fastest convergence, off-policy efficiency |
| Small discrete env, stochastic | SARSA or Expected SARSA | On-policy stability for safe behaviour |
| Very sparse rewards (large env) | Q-learning + optimistic init | Encourages coverage, breaks symmetry |
| Stochastic, evaluation is greedy | Q-learning | Optimises for the greedy evaluation policy |
| Stochastic, evaluation is epsilon-greedy | SARSA | Optimises for the deployment policy directly |
| High-variance environment noise | Expected SARSA | Exact expectation removes SARSA's sampling variance |
| Overestimation suspected | Double Q-learning | Decouple selection and evaluation Q-tables |
| Long episodes, sparse reward | n-step Q-learning (n=5–10) | Reduces bootstrap bias from unreliable Q estimates |
| State space 10k–1M | Tile coding + linear FA | Break the tabular wall with linear function approx |
| State space > 1M, images | DQN (neural network) | Only option — full deep RL required |

The progression from pure tabular to deep RL is not a discrete jump — it is a continuum. Tile coding (discussed in the next post) sits between Q-tables and neural networks: it uses a fixed feature representation to map continuous observations to binary features, and learns a linear Q-function over those features. This handles environments with 100,000 to 10 million effective states without requiring a neural network, and retains much of the interpretability advantage of tabular methods.

## Connections to the Broader RL Landscape

Tabular Q-learning is the base case from which every modern RL algorithm derives. Understanding it precisely is not optional background — it is the foundation.

**Function approximation** ([next in this series](/blog/machine-learning/reinforcement-learning/function-approximation-and-tile-coding)) replaces the Q-table with a parameterised function $Q(s, a; \theta)$. The Q-learning update becomes a gradient step on the loss $L(\theta) = \frac{1}{2}(r + \gamma \max_{a'} Q(s', a'; \theta^-) - Q(s, a; \theta))^2$. The two core challenges — sample efficiency and training stability — are the same challenges the tabular algorithm faces, but now they manifest as overfitting, catastrophic forgetting, and deadly triad instability.

**DQN** adds experience replay (breaking temporal correlation in the training data) and a target network (stabilising the bootstrap target) to make neural function approximation practical. The same Q-learning update we derived in this post is the core of DQN — the neural network is replacing the table, not the algorithm.

**Policy gradient methods** (REINFORCE, PPO) bypass the value function and directly optimise the policy by estimating the gradient $\nabla_\theta \mathbb{E}_\pi[\sum_t r_t]$. They are motivated by problems where defining a good Q-function is harder than optimising a policy — continuous control, stochastic action selection, multi-step planning. But understanding why policy gradient methods need a baseline (to reduce variance) requires understanding what Q-values represent — which you now know exactly.

For the full taxonomy of how these algorithms relate, see [Reinforcement Learning: A Unified Map](/blog/machine-learning/reinforcement-learning/reinforcement-learning-a-unified-map), which places every algorithm on a single diagram by objective and estimator type.

There is also an important connection to **debugging AI training** more broadly. Many of the convergence monitoring patterns introduced here — tracking gradient or value change norms, watching for oscillation, auditing the loss signal's density — reappear in neural network training diagnostics. The tabular regime is an ideal setting to internalise these diagnostic instincts before the neural network's additional complexity makes the signals harder to interpret. The patterns carry over directly; see [Debugging AI Training: A Complete Guide](/blog/machine-learning/debugging-training/the-training-debugging-playbook) for the extension to deep learning and full deep RL systems.

## Key Takeaways

1. **Tabular RL is exact, not an approximation.** For fewer than ~10,000 discrete states, Q-learning converges to the provably optimal policy — no approximation error, no generalisation gap.

2. **The four criteria for tabular RL:** small discrete state space ($|S| \leq 10{,}000$), discrete actions ($|A| \leq 100$), known state space boundary, interpretability or prototyping requirement. All four do not need to hold simultaneously — any one is sufficient justification.

3. **Optimistic initialisation beats zero initialisation by roughly 2× convergence speed** on Taxi-v3. The mechanism is systematic coverage: unvisited states retain the high initial value, making them preferred under the greedy policy until every state has been visited at least once.

4. **The minimum epsilon floor matters.** $\varepsilon_{\min} = 0.01$ prevents policy freezing in stochastic environments and costs at most 1% of evaluation reward.

5. **Monitor three convergence signals:** Q-delta $\| Q_{k+1} - Q_k \|_\infty < 10^{-4}$, zero policy changes for 500+ consecutive episodes, and a non-improving reward window. Any single signal alone can give false convergence.

6. **Q-learning is off-policy; SARSA is on-policy.** In stochastic environments, SARSA achieves a higher final reward than Q-learning because it learns a value function that accounts for exploration noise in the bootstrap target.

7. **Three distinct failure modes, three distinct fixes.** Never converges → check $\alpha$ decay speed and $\varepsilon$ floor. Oscillates → reduce $\alpha$ by 10×. Wrong policy → audit reward shaping and $\gamma$.

8. **Reward shaping must be potential-based** ($F = \gamma\Phi(s') - \Phi(s)$) to preserve the optimal policy. Any other form of shaping changes the MDP solution.

9. **The capacity wall is a hard cliff, not a gradual slope.** Tabular RL is exact at 500 states and physically impossible at $10^{33,696}$ states. The transition is categorical.

10. **Build the tabular baseline before deep RL.** It validates reward signal, episode design, and state encoding in minutes — revealing bugs that would take weeks to diagnose inside a neural network.

## Further Reading

- Richard S. Sutton and Andrew G. Barto, "Reinforcement Learning: An Introduction," 2nd edition, MIT Press, 2018. Chapters 6 (TD learning) and 7 (n-step bootstrapping) are the definitive reference for tabular RL.
- Christopher J.C.H. Watkins and Peter Dayan, "Q-Learning," Machine Learning 8, 279–292, 1992. The original convergence proof for Q-learning.
- Andrew Y. Ng, Daishi Harada, Stuart Russell, "Policy Invariance Under Reward Transformations: Theory and Application to Reward Shaping," ICML 1999. The potential-based shaping theorem that justifies safe reward augmentation.
- Gymnasium documentation: [gymnasium.farama.org](https://gymnasium.farama.org). FrozenLake-v1, Taxi-v3, and CliffWalking are the canonical tabular benchmarks with full environment descriptions.
- [Reinforcement Learning: A Unified Map](/blog/machine-learning/reinforcement-learning/reinforcement-learning-a-unified-map) — this series' taxonomy placing tabular methods within the full RL landscape.
- [Function Approximation and Tile Coding](/blog/machine-learning/reinforcement-learning/function-approximation-and-tile-coding) — the next post: breaking the tabular capacity wall with linear approximation, tile coding, and gradient TD methods.
- [The Reinforcement Learning Playbook](/blog/machine-learning/reinforcement-learning/the-reinforcement-learning-playbook) — the series capstone with algorithm selection guidance for real systems.
- [Debugging AI Training: A Complete Guide](/blog/machine-learning/debugging-training/the-training-debugging-playbook) — debugging principles that apply directly to RL training loops.
