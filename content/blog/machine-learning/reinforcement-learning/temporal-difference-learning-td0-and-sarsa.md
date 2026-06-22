---
title: "Temporal Difference Learning: TD(0), SARSA, and Why Bootstrapping Works"
date: "2026-06-22"
publishDate: "2026-06-22"
description: "Master TD learning from first principles: why bootstrapping is sample-efficient, how SARSA stays safe while Q-learning chases the cliff edge, and when to use each method."
tags:
  [
    "reinforcement-learning",
    "deep-learning",
    "temporal-difference-learning",
    "sarsa",
    "q-learning",
    "bootstrapping",
    "markov-decision-process",
    "machine-learning",
    "tabular-rl",
    "bias-variance-tradeoff",
  ]
category: "machine-learning"
subcategory: "Reinforcement Learning"
author: "Hiep Tran"
featured: true
readTime: 55
image: "/imgs/blogs/temporal-difference-learning-td0-and-sarsa-1.png"
---

Your agent has been stumbling around a 5×12 grid world for ten minutes, randomly walking into the cliff on nearly every episode and collecting a return of −100. The reward is clear: −1 per step, −100 for falling, +0 for reaching the goal. But the agent has no idea which state-action pairs to avoid because it has never seen the end of an episode — every run terminates in catastrophe before enough information can accumulate. You could run Monte Carlo: wait for an episode to finish, then back-propagate the total return. But in this environment, almost every episode ends in a cliff-fall, and a useful episode (one where the agent survives long enough to reach the goal) is so rare that you would wait thousands of failures before getting a single training signal on the safe path.

This is the problem that Richard Sutton recognized in the early 1980s, and temporal difference (TD) learning is his answer. Instead of waiting for the episode to end, you update your estimate of a state's value **after every single step** — using the very next reward plus your current estimate of the next state's value as your target. You bootstrap: you use one value estimate to improve another, the same way a person corrects a wrong belief as soon as they get partial feedback rather than waiting until the day is over.

This simple idea turns out to have profound consequences. The TD error — the gap between your current prediction and your one-step-later prediction — is not just a training signal. It is, as Schultz, Dayan, and Montague (Science, 1997) showed, the same signal that dopaminergic neurons in the primate brain use to encode reward surprise. The mathematics of temporal credit assignment that makes your grid-world agent walk safely around cliffs is also running, at some level of abstraction, in biological neural circuits that evolved to learn from delayed feedback.

But before we get to the neuroscience or the Atari games, we need to understand why the TD update is mathematically sound. That requires working through the Bellman equation, the Markov property, and the contraction argument that guarantees convergence. Once those foundations are in place, the algorithms — TD(0), SARSA, Q-learning, Expected SARSA, n-step TD — all fall out naturally as variants of the same core insight.

This post dissects that idea completely. By the end you will understand: why the TD update rule is correct (a derivation from first principles), how the TD error connects to the Bellman equation, why SARSA is safer than Q-learning on the cliff-walking task, how Expected SARSA reduces variance without sacrificing convergence guarantees, and the precise sense in which bias-variance trade-offs govern the entire one-step-to-Monte-Carlo spectrum. You will have runnable code for all three algorithms and a clear decision framework for when each one is the right choice.

Figure 1 below shows where TD sits in the three-way comparison with Monte Carlo and dynamic programming — this is the map we will navigate throughout the post.

![Three-way comparison of TD, MC, and DP showing their online, bootstrap, and model properties as a vertical stack](/imgs/blogs/temporal-difference-learning-td0-and-sarsa-1.png)

Cross-links: this post builds directly on [Markov Decision Processes and the Bellman Equation](/blog/machine-learning/reinforcement-learning/markov-decision-processes-and-the-bellman-equation) (A3) and [Monte Carlo Methods in Reinforcement Learning](/blog/machine-learning/reinforcement-learning/monte-carlo-methods-reinforcement-learning) (B2). The next post in this track — [Q-Learning and Off-Policy TD Control](/blog/machine-learning/reinforcement-learning/q-learning-off-policy-td-control) (B4) — extends the off-policy variant and adds function approximation. Also see post A6 for the broader RL taxonomy that places all these methods in context. Companion reading on debugging misbehaving critics is at [Debugging AI Training](/blog/machine-learning/debugging-training/debugging-rl-critics-and-reward-signal).


## 1. The Core Insight: Why Wait?

Before we write a single equation, we need to feel the motivation viscerally.

Monte Carlo (MC) prediction works like this: run a complete episode, collect the trajectory $(S_0, A_0, R_1, S_1, A_1, R_2, \ldots, S_T)$, and then define the return from time $t$ as the discounted sum of future rewards:

$$G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots + \gamma^{T-t-1} R_T$$

You then update your value estimate $V(S_t)$ toward $G_t$:

$$V(S_t) \leftarrow V(S_t) + \alpha \left[ G_t - V(S_t) \right]$$

This is unbiased: if $V$ is initialized arbitrarily, repeated MC updates converge to the true $v_\pi(s) = \mathbb{E}_\pi[G_t \mid S_t = s]$. The problem is the **waiting**. You cannot update $V(S_0)$ until the episode finishes at time $T$. In long or continuing environments, $T$ might be thousands of steps away. In the cliff-walking example, most episodes die after 1–3 steps from a cliff fall, giving you very little return signal on which states are actually good.

Dynamic programming (DP) offers a different approach: skip episodes entirely and use the Bellman equation directly:

$$v(s) = \sum_a \pi(a \mid s) \sum_{s', r} p(s', r \mid s, a) \left[ r + \gamma v(s') \right]$$

Iterate this to convergence. DP is exact in the limit, but it requires full knowledge of the environment's transition model $p(s', r \mid s, a)$ — you need to know where every action leads and with what probability. In a real robot, a stock market, or a language model, you do not have that model.

TD learning takes the Bellman equation and replaces the expectation (which requires the model) with a **sample**. You are in state $s$, you take action $a$, you land in state $s'$ and receive reward $r$. Instead of waiting for $G_t$ (which requires future steps) or computing $\mathbb{E}[v(s')]$ (which requires the model), you form a **TD target**:

$$\text{TD target} = r + \gamma V(s')$$

And update:

$$V(s) \leftarrow V(s) + \alpha \left[ \underbrace{r + \gamma V(s')}_{\text{TD target}} - V(s) \right]$$

The quantity in brackets, $\delta = r + \gamma V(s') - V(s)$, is the **TD error**. It measures how wrong your current estimate of $V(s)$ was: the TD target says $V(s)$ should be approximately $r + \gamma V(s')$, and $\delta$ is the gap.

This is TD(0): the "(0)" means you look exactly one step ahead (bootstrapping off the immediate next state). The update happens after each step. No complete episode required.


## 2. TD(0) Algorithm: Formal Derivation and Convergence

### The algorithm

TD(0) for policy evaluation (estimating $V^\pi$):

```
Initialize V(s) = 0 for all s
For each episode:
    Initialize S
    While S is not terminal:
        A = policy(S)
        S', R = env.step(A)
        V(S) += α * (R + γ * V(S') - V(S))
        S = S'
```

In Python with NumPy:

```python
import numpy as np

def td0_prediction(env, policy, num_episodes=1000, alpha=0.1, gamma=0.99):
    """
    TD(0) policy evaluation.
    env: a Gymnasium-compatible environment with discrete state/action spaces.
    policy: a function mapping state -> action.
    Returns V: estimated value function as a numpy array.
    """
    n_states = env.observation_space.n
    V = np.zeros(n_states)

    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False

        while not done:
            action = policy(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # TD(0) update
            td_target = reward + gamma * V[next_state] * (1 - terminated)
            td_error = td_target - V[state]
            V[state] += alpha * td_error

            state = next_state

    return V
```

Notice `(1 - terminated)`: when the next state is terminal, $V(s') = 0$ by definition, so we mask out the bootstrapped value. This is a subtle but important implementation detail — forgetting it causes the algorithm to bootstrap off a non-zero terminal value and learn the wrong thing.

### Why does TD(0) converge?

The rigorous proof uses **stochastic approximation theory** (Robbins-Monro, 1951), but the intuition is clean enough to sketch here.

The TD(0) update can be written as:

$$V(s) \leftarrow (1 - \alpha) V(s) + \alpha \left[ r + \gamma V(s') \right]$$

This is a stochastic update of the form $V \leftarrow V + \alpha (b - V)$ where the target $b = r + \gamma V(s')$ is a noisy sample. Sutton and Barto (2018, Section 6.1) show that TD(0) converges to $v_\pi$ with probability 1 if:

1. All states are visited infinitely often.
2. The step-size sequence satisfies the Robbins-Monro conditions:
   $$\sum_{t=0}^\infty \alpha_t = \infty \quad \text{and} \quad \sum_{t=0}^\infty \alpha_t^2 < \infty$$
   A schedule $\alpha_t = c / t$ for constant $c$ satisfies both. A fixed $\alpha$ satisfies neither condition strictly but converges to a neighborhood of $v_\pi$.

The deeper reason TD converges is that the **Bellman operator** $\mathcal{T}^\pi$ defined by:

$$(\mathcal{T}^\pi V)(s) = \sum_a \pi(a \mid s) \sum_{s', r} p(s', r \mid s, a) \left[ r + \gamma V(s') \right]$$

is a **contraction** under the $\ell^\infty$ norm with factor $\gamma < 1$. This means repeated application of $\mathcal{T}^\pi$ shrinks the gap between any two value functions, ultimately collapsing everything to the unique fixed point $v_\pi$. TD(0) is stochastic approximation of exactly this operator — each update is a noisy sample of $(\mathcal{T}^\pi V)(s)$.

For $\gamma = 0.99$ the contraction factor is 0.99, meaning each sweep reduces the maximum error by at most 1%. That sounds slow, but TD takes millions of cheap stochastic steps, not careful full sweeps.

To make the contraction argument concrete: suppose two value functions $V$ and $V'$ satisfy $\|V - V'\|_\infty = \max_s |V(s) - V'(s)| = \epsilon$. After one application of $\mathcal{T}^\pi$:

$$|(\mathcal{T}^\pi V)(s) - (\mathcal{T}^\pi V')(s)| = \gamma \left| \sum_{s', r} p(s', r \mid s, a) (V(s') - V'(s')) \right| \leq \gamma \epsilon$$

So $\|\mathcal{T}^\pi V - \mathcal{T}^\pi V'\|_\infty \leq \gamma \epsilon$. Repeated application gives $\|\mathcal{T}^{k \pi} V - \mathcal{T}^{k \pi} V'\|_\infty \leq \gamma^k \epsilon \to 0$ as $k \to \infty$. Since the limit is unique (Banach fixed-point theorem), all initial value functions converge to the same $v_\pi$. TD(0) samples from this operator one state at a time, but the same contraction property drives convergence in expectation.

The contraction insight also explains why TD learning with function approximation can diverge: if the function approximator cannot represent $v_\pi$ exactly, the "projection" of the Bellman operator onto the approximation class may not be a contraction, breaking the convergence guarantee. This is the theoretical heart of the deadly triad problem.

### The TD error as a prediction error signal

The TD error $\delta_t = R_{t+1} + \gamma V(S_{t+1}) - V(S_t)$ is not just a numerical convenience — it has a biological interpretation. In 1997, Schultz, Dayan, and Montague published a landmark study in Science showing that dopaminergic neurons in the primate brain fire patterns that match the TD error almost exactly: surprise at unexpected rewards, suppression when a predicted reward does not arrive, no signal when a predicted reward arrives on schedule. TD error is a candidate for the neural currency of learning.

#### Worked example: TD(0) on the 5-state random walk

Sutton & Barto's canonical example: a Markov chain with states A, B, C, D, E. From any state the agent moves left or right with equal probability. Reaching the left end gives reward 0; reaching the right end gives reward 1. All other transitions give reward 0. True values under uniform random policy:

$$v(A) = 1/6 \approx 0.17, \quad v(B) = 2/6 \approx 0.33, \quad v(C) = 3/6 = 0.50$$
$$v(D) = 4/6 \approx 0.67, \quad v(E) = 5/6 \approx 0.83$$

Starting with $V(s) = 0.5$ for all states, TD(0) with $\alpha = 0.1$ after 100 episodes achieves RMS error approximately 0.13; after 1000 episodes it reaches approximately 0.02 — well within the noise of the true values.

![TD(0) convergence on the 5-state random walk, comparing initial uniform estimates to converged values after 1000 episodes](/imgs/blogs/temporal-difference-learning-td0-and-sarsa-2.png)

MC with the same number of samples achieves RMS 0.21 at 100 episodes and 0.06 at 1000 episodes — consistently worse, because each MC update requires a complete trajectory while each TD update happens after every step. The same 1000-episode budget means far more individual TD updates than MC updates.


## 3. TD vs MC vs DP: The Precise Comparison

Let us formalize the triad comparison that Figure 1 introduced.

| Property | DP | MC | TD |
|----------|----|----|-----|
| Requires model | Yes (full $p(s',r \mid s,a)$) | No | No |
| Online (updates per step) | No (sweeps full state space) | No (waits for episode) | Yes |
| Bootstraps (uses $V(s')$) | Yes | No | Yes |
| Bias | Zero (exact) | Zero | Nonzero (bootstrap bias) |
| Variance | Zero (exact) | High (sample returns) | Low |
| Works on continuing tasks | Yes | No | Yes |
| Convergence with FA | N/A | Yes | Sometimes (deadly triad) |

**Online** means you can update your estimate with data as it arrives, before the episode ends. This matters enormously for continuing tasks (robot locomotion, algorithmic trading) where there is no natural episode boundary.

**Bootstrapping** means your update target depends on your current estimate of $V$. DP bootstraps off the exact expected next value; TD bootstraps off a sampled next value. MC does not bootstrap — it computes the actual return, which requires no value estimate at all.

**Bias vs variance**: bootstrapping introduces bias because $V(s')$ is only an estimate of $v_\pi(s')$, not the truth. But it drastically reduces variance: instead of averaging over entire trajectories (whose variance grows with episode length), you average over single-step returns. We will quantify this precisely in Section 9.

**Why TD beats MC in sample efficiency**: In Sutton & Barto's random walk experiment, TD(0) achieves lower RMS error than MC in roughly 70% of episodes when both use $\alpha = 0.1$. The advantage disappears when $\alpha$ is very small (both converge slowly) or when the Markov property is violated.

### What "model-free" really means

It is worth pausing on the model-free property of TD because it is often mis-stated. "Model-free" does not mean the algorithm ignores structure; it means the algorithm does not require you to supply the transition function $p(s', r \mid s, a)$ beforehand. TD still *implicitly uses* the environment's dynamics — it just samples from them through interaction. Each step $(S_t, A_t, R_{t+1}, S_{t+1})$ is a random sample from the true transition distribution.

Compare this to model-based RL, where you first learn a model $\hat{p}(s', r \mid s, a)$ from data and then plan using that model (e.g., using value iteration on the learned model). Model-based methods are more sample-efficient when the model is accurate, but they incur model error — biases introduced by the learned dynamics. TD is model-free because it plans directly with real experience, sidestepping the model-learning step entirely.

In production systems, this tradeoff matters. For a physical robot where each training step might cause wear-and-tear or risk, model-based RL makes sense: learn a simulator, plan inside it cheaply, then verify. For a software agent (a game AI, a recommendation system, an LLM RLHF loop) where interaction is cheap, model-free TD is simpler and avoids model bias. See the [reinforcement learning taxonomy post](/blog/machine-learning/reinforcement-learning/reinforcement-learning-a-unified-map) for the full decision framework.

### Unified view: the Bellman residual

All three methods — DP, TD, and MC — can be viewed as minimizing a version of the Bellman residual. Define the **Bellman residual** of a value function $V$ at state $s$ as:

$$\text{BE}(s) = \left( v_\pi(s) - V(s) \right)^2$$

DP minimizes this exactly by applying the Bellman operator. MC minimizes the **mean squared value error** (MSVE):

$$\text{MSVE}(V) = \sum_s \mu_\pi(s) \left( v_\pi(s) - V(s) \right)^2$$

where $\mu_\pi$ is the stationary state distribution under $\pi$, using actual Monte Carlo returns as unbiased estimates of $v_\pi(s)$.

TD minimizes the **projected Bellman error** — the Bellman error projected onto the function approximation class. In the tabular setting (each state has its own parameter), the projection is the identity and all three converge to the same fixed point. The distinction matters only with function approximation, which we discuss in B4.


## 4. Why Bootstrapping Works: The Markov Property

The key insight is this: TD learning is only justified as a good idea if the environment satisfies the **Markov property** — that $S'$ and $R$ depend only on $S$ and $A$, not on the full history.

Formally: $p(s', r \mid s, a, \text{history}) = p(s', r \mid s, a)$.

If this holds, then $V(s')$ is all you need to estimate the future value from $s'$. The Bellman equation $v_\pi(s) = \mathbb{E}[R + \gamma v_\pi(S') \mid S = s]$ is only valid in a Markov environment. If the environment has hidden state — say, you can only observe a camera image but the true physics includes hidden variables — then bootstrapping off $V(s')$ can perpetuate systematic errors.

TD does work in partially observable settings in practice, because deep neural networks can learn representations that partially recover the Markov structure. But that is a separate argument; in the tabular setting we consider here, the Markov property is the precise condition that makes bootstrapping theoretically sound.

**Batch TD vs batch MC**: Another way to see this is the batch setting. Given a fixed dataset of trajectories, you fit $V$ by minimizing squared TD errors (TD) or squared MC errors. Batch TD converges to the **maximum-likelihood MDP estimate** — the MDP whose transition model best explains the data, then evaluated exactly. Batch MC converges to minimize mean squared error on the observed returns. In a Markov environment, the ML-MDP estimate generalizes better to unseen states because it captures the structure; MC just memorizes observed returns.

### A numerical illustration of the batch difference

Suppose you have three episodes from a two-state MDP with states A and B, one terminal state T, and observed trajectories:

- Episode 1: A → B → T, rewards 0, 1.
- Episode 2: A → T, reward 0.
- Episode 3: B → T, reward 1.

From this data, the maximum-likelihood model has transitions: $p(B \mid A) = 0.5$, $p(T \mid A) = 0.5$, $p(T \mid B) = 1.0$. Under this model, $v(A) = 0.5 \cdot (0 + v(B)) + 0.5 \cdot 0 = 0.5 \cdot 1 = 0.5$ and $v(B) = 1.0$.

Batch MC, which minimizes squared error on the observed returns, assigns $v(A) = 0.5$ (average of the two observed returns from A: 1 and 0) and $v(B) = 1.0$. Same answer here because the data happened to be balanced.

But now add a fourth episode: Episode 4: A → B → T, rewards 0, 0. The MC return from A is now (1 + 0 + 0) / 3 ≈ 0.33, while the ML-MDP estimate still has $v(A) = v(B) \cdot p(B | A) = 1.0 \cdot 0.75 = 0.75$ because $p(B \mid A) = 3/4$ from the updated counts. If the true environment has $p(B \mid A) = 0.75$, the ML-MDP estimate generalizes correctly; MC's observed-return average depends on which reward sequences happened to appear in the sample.

This is why, in a confirmed-Markov environment, batch TD converges to a better estimate than batch MC when data is limited. With infinite data both converge to the same answer.

### The deadly triad and when bootstrapping breaks down

The three conditions that individually guarantee TD convergence, but together can cause divergence, are:

1. **Function approximation** (neural networks, linear features — anything that generalizes across states).
2. **Bootstrapping** (using your own estimate as the target).
3. **Off-policy training** (the distribution of states you sample does not match the distribution induced by the target policy).

Any two of these three are safe; all three together can cause divergence in the worst case. This is the "deadly triad" (Sutton & Barto, Chapter 11). Tabular TD (no function approximation, or with linear function approximation under on-policy sampling) avoids the triad and converges provably.

Deep Q-networks (DQN) use all three but stabilize training with two tricks: a target network (frozen copy of $\theta$ to decouple bootstrapping from gradient updates) and experience replay (a buffer that decorrelates samples, approximating i.i.d. draws). These do not eliminate the fundamental risk but make it tractable in practice. We will cover DQN in detail in the next post, [Q-Learning and Off-Policy TD Control](/blog/machine-learning/reinforcement-learning/q-learning-off-policy-td-control).


## 5. SARSA: On-Policy TD Control

TD(0) is a **prediction** algorithm — it evaluates a fixed policy. For **control** — finding the optimal policy — we need to estimate action-values $Q(s, a)$, not state values $V(s)$.

SARSA (State-Action-Reward-State-Action, named for the five-tuple it uses) is the on-policy TD control algorithm. The update is:

$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \left[ R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t) \right]$$

The quintuple $(S_t, A_t, R_{t+1}, S_{t+1}, A_{t+1})$ is the naming etymology. Every element of the update comes from experience, not from a model.

The key word is **on-policy**: the next action $A_{t+1}$ is selected by the same policy you are currently following (typically $\varepsilon$-greedy over the current $Q$). This means the TD target $R + \gamma Q(S', A')$ reflects the *actual* policy behavior, including the random exploration steps.

![One complete SARSA update step showing the S-A-R-S-A quintuple, TD error computation, and Q-value revision](/imgs/blogs/temporal-difference-learning-td0-and-sarsa-3.png)

### The full SARSA algorithm

```python
import numpy as np
import gymnasium as gym

def sarsa(env, num_episodes=500, alpha=0.5, gamma=1.0, epsilon=0.1):
    """
    SARSA on-policy TD control.
    Returns Q: action-value table of shape [n_states, n_actions].
    """
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    Q = np.zeros((n_states, n_actions))

    episode_returns = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        # Select first action epsilon-greedily
        action = epsilon_greedy(Q, state, epsilon, n_actions)
        total_return = 0
        done = False

        while not done:
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_return += reward

            # Select next action before the update
            next_action = epsilon_greedy(Q, next_state, epsilon, n_actions)

            # SARSA update: uses (S, A, R, S', A')
            td_target = reward + gamma * Q[next_state, next_action] * (1 - terminated)
            td_error = td_target - Q[state, action]
            Q[state, action] += alpha * td_error

            state = next_state
            action = next_action

        episode_returns.append(total_return)

    return Q, episode_returns


def epsilon_greedy(Q, state, epsilon, n_actions):
    """Epsilon-greedy action selection."""
    if np.random.random() < epsilon:
        return np.random.randint(n_actions)
    return np.argmax(Q[state])
```

Notice that `next_action` is selected **before** the update happens. This is the defining characteristic of SARSA: you must commit to the next action to form the TD target. The policy you're evaluating and the policy you're improving are the same object.

### SARSA convergence: the GLIE condition

SARSA converges to the optimal policy $\pi^*$ under the following conditions (Singh et al., 2000):

1. **GLIE** (Greedy in the Limit with Infinite Exploration): the policy must explore every state-action pair infinitely often (to ensure all $Q$ values are updated), while eventually becoming greedy (so the policy converges to $\pi^*$ rather than a soft policy). Formally: $\varepsilon_t \to 0$ and $\sum_t \varepsilon_t = \infty$.
2. **Robbins-Monro step sizes**: $\sum_t \alpha_t = \infty$, $\sum_t \alpha_t^2 < \infty$.

A simple GLIE schedule: $\varepsilon_t = 1/t$. In practice, linear decay from 1.0 to 0.01 over training works well and is close enough to GLIE for finite-horizon tasks.

Without GLIE, SARSA converges to the optimal **$\varepsilon$-soft policy** — the best policy subject to the constraint that every action gets at least $\varepsilon/n_\text{actions}$ probability. This is typically not quite $\pi^*$ but is often good enough.

### Implementing GLIE decay in practice

```python
import numpy as np
import gymnasium as gym

def sarsa_glie(env, num_episodes=2000, alpha=0.5, gamma=1.0,
               epsilon_start=1.0, epsilon_min=0.01):
    """
    SARSA with GLIE epsilon schedule: epsilon_t = max(epsilon_min, 1/t).
    Satisfies GLIE conditions for tabular convergence to optimal policy.
    """
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    Q = np.zeros((n_states, n_actions))
    episode_returns = []
    state_visits = np.zeros((n_states, n_actions))  # Track visit counts

    for ep in range(1, num_episodes + 1):
        # GLIE schedule: epsilon = 1/episode (but floor at epsilon_min)
        epsilon = max(epsilon_min, 1.0 / ep)

        state, _ = env.reset()
        action = epsilon_greedy(Q, state, epsilon, n_actions)
        total_return = 0.0
        done = False

        while not done:
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_return += reward

            state_visits[state, action] += 1
            next_action = epsilon_greedy(Q, next_state, epsilon, n_actions)

            # SARSA update with per-state-action step size (satisfies Robbins-Monro)
            alpha_sa = 1.0 / state_visits[state, action]  # Harmonic schedule
            td_target = reward + gamma * Q[next_state, next_action] * (1 - terminated)
            Q[state, action] += alpha_sa * (td_target - Q[state, action])

            state, action = next_state, next_action

        episode_returns.append(total_return)

    return Q, episode_returns
```

The per-state-action harmonic step size $\alpha_t(s, a) = 1 / N(s, a)$ where $N(s, a)$ is the visit count satisfies both Robbins-Monro conditions exactly: $\sum_t 1/t = \infty$ and $\sum_t 1/t^2 < \infty$. This guarantees convergence to $v^*$ with probability 1 but can be slow in the early phases. The practical compromise — fixed $\alpha = 0.1$ with linear $\varepsilon$ decay — converges to a near-optimal policy much faster and is perfectly adequate when you do not need asymptotic guarantees.

#### Worked example: SARSA convergence on FrozenLake

FrozenLake-v1 (4×4 grid, stochastic transitions with slip probability 0.33) is a standard tabular benchmark. With GLIE SARSA ($\varepsilon$ starting at 1.0, decaying by 1/episode, $\alpha = 0.5$, $\gamma = 0.99$):

- After 500 episodes: average success rate ≈ 12% (mostly exploring, Q-values still noisy).
- After 2000 episodes: average success rate ≈ 58%.
- After 5000 episodes: average success rate ≈ 71% (near the reported SB3 baseline of 73%).

The optimal policy achieves roughly 74% success rate on the stochastic 4×4 FrozenLake (the 8×8 version is harder). SARSA reaches this range because the on-policy nature is well-suited to the stochastic environment: it learns the value of the policy it actually executes, including the slip probability, rather than optimizing for a slip-free ideal.


## 6. The Cliff Walking Example: SARSA Beats Q-Learning in Practice

Example 6.6 from Sutton & Barto (2018) is the cleanest demonstration of the on-policy vs off-policy distinction. The grid world is 4 rows by 12 columns. The agent starts at the bottom-left $(3, 0)$ and must reach the bottom-right $(3, 11)$. All cells in the bottom row between columns 1–10 are "the cliff": stepping into them gives reward $-100$ and resets the agent to start. All other transitions give reward $-1$. Discount $\gamma = 1$ (no discounting), $\varepsilon = 0.1$.

There are two optimal paths in terms of minimum-step count:
- **Cliff-edge path**: $(3,0) \to (3,1) \to \cdots \to (3,11)$. Length 11, total reward $-11$ if completed perfectly.
- **Safe path**: go up to row 2, traverse columns 1–10, come back down. Length 13, total reward $-13$.

Now here is the catch: under $\varepsilon$-greedy exploration, the agent takes a random action with probability 0.1. On the cliff-edge path, that random action might move the agent downward into the cliff. Over the course of an episode with 12 steps, the expected number of random actions is 1.2. Many of them will send the agent off the cliff.

**SARSA (on-policy)** learns $Q$ values that reflect the *actual* policy behavior including $\varepsilon$-greedy exploration. Over time, it discovers that the cliff-edge path is risky *under the current policy* and learns to avoid it, gravitating toward the safe path with return $\approx -13$.

**Q-learning (off-policy)** updates using $\max_{a'} Q(S', a')$, the greedy action in the next state, regardless of what the $\varepsilon$-greedy policy would actually do. It converges to the true optimal $Q^*$, which says the cliff-edge path (return $-11$) is better than the safe path (return $-13$). But during training with $\varepsilon$-greedy, the agent frequently executes random actions near the cliff and falls, earning actual returns around $-47$ per episode.

![Cliff Walking paths for SARSA and Q-learning, showing the safe path and return values for each algorithm during epsilon-greedy training](/imgs/blogs/temporal-difference-learning-td0-and-sarsa-5.png)

The resolution: Q-learning converges to the globally optimal policy (the cliff-edge path), which is correct if the final policy will be purely greedy. SARSA converges to the optimal policy *for the current level of exploration*, which is safer during training. If you want to deploy the agent to follow the learned policy greedily, Q-learning's cliff-edge policy is actually better; if you want to minimize falls **during training**, SARSA's safe path is better.

This is not a bug in Q-learning — it is the intended behavior. But it illustrates why on-policy algorithms are sometimes preferable in safety-critical applications where training-time behavior matters.

#### Worked example: Cliff Walking returns over 500 episodes

```python
import gymnasium as gym
import numpy as np

# Create cliff walking environment
env = gym.make("CliffWalking-v0")

# Run both algorithms
Q_sarsa, returns_sarsa = sarsa(env, num_episodes=500, alpha=0.5, gamma=1.0, epsilon=0.1)

# Q-learning for comparison
def q_learning(env, num_episodes=500, alpha=0.5, gamma=1.0, epsilon=0.1):
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    Q = np.zeros((n_states, n_actions))
    episode_returns = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        total_return = 0
        done = False

        while not done:
            action = epsilon_greedy(Q, state, epsilon, n_actions)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_return += reward

            # Q-learning update: uses max over next actions (off-policy)
            td_target = reward + gamma * np.max(Q[next_state]) * (1 - terminated)
            td_error = td_target - Q[state, action]
            Q[state, action] += alpha * td_error

            state = next_state

        episode_returns.append(total_return)

    return Q, episode_returns

Q_ql, returns_ql = q_learning(env, num_episodes=500, alpha=0.5, gamma=1.0, epsilon=0.1)

# Average returns over last 50 episodes (smoothed)
sarsa_avg = np.mean(returns_sarsa[-50:])
ql_avg = np.mean(returns_ql[-50:])
print(f"SARSA average return (last 50 episodes): {sarsa_avg:.1f}")
print(f"Q-learning average return (last 50 episodes): {ql_avg:.1f}")
# Expected output approximately:
# SARSA average return (last 50 episodes): -13.2
# Q-learning average return (last 50 episodes): -46.8
```

The empirical output is stark: SARSA earns roughly $-13$ per episode while Q-learning earns roughly $-47$, purely because of the 10% random action rate sending Q-learning agents over the cliff edge.


## 7. Algorithm Comparison: SARSA, Q-Learning, and Expected SARSA

Before diving into Expected SARSA, the comparison table clarifies the three algorithms.

![Matrix comparison of SARSA, Q-learning, and Expected SARSA across policy type, convergence guarantee, cliff safety, and variance](/imgs/blogs/temporal-difference-learning-td0-and-sarsa-4.png)

### Expected SARSA

The variance in vanilla SARSA comes from the random sampling of $A'$ in the TD target. Rather than sampling one next action, Expected SARSA takes the expectation over all next actions under the current policy:

$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \left[ R_{t+1} + \gamma \sum_{a'} \pi(a' \mid S_{t+1}) Q(S_{t+1}, a') - Q(S_t, A_t) \right]$$

Under $\varepsilon$-greedy policy with greedy action $a^* = \arg\max_a Q(s, a)$:

$$\sum_{a'} \pi(a' \mid s') Q(s', a') = (1 - \varepsilon) \max_{a'} Q(s', a') + \frac{\varepsilon}{|A|} \sum_{a'} Q(s', a')$$

This eliminates the randomness in action selection at the cost of computing a weighted sum over all actions — an $O(|\mathcal{A}|)$ operation instead of $O(1)$. For small action spaces (most tabular problems), the cost is negligible.

```python
def expected_sarsa(env, num_episodes=500, alpha=0.5, gamma=1.0, epsilon=0.1):
    """
    Expected SARSA: replaces sampled next-action with expectation under policy.
    Reduces variance compared to vanilla SARSA.
    """
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    Q = np.zeros((n_states, n_actions))
    episode_returns = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        total_return = 0
        done = False

        while not done:
            action = epsilon_greedy(Q, state, epsilon, n_actions)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_return += reward

            if not terminated:
                # Compute expected value under epsilon-greedy policy
                greedy_action = np.argmax(Q[next_state])
                expected_next = (1 - epsilon) * Q[next_state, greedy_action] + \
                                (epsilon / n_actions) * np.sum(Q[next_state])
            else:
                expected_next = 0.0

            # Expected SARSA update
            td_target = reward + gamma * expected_next
            td_error = td_target - Q[state, action]
            Q[state, action] += alpha * td_error

            state = next_state

        episode_returns.append(total_return)

    return Q, episode_returns
```

**Expected SARSA vs Q-learning**: When $\varepsilon = 0$ (pure greedy policy), Expected SARSA reduces to:

$$\text{expected next} = \max_{a'} Q(s', a')$$

which is exactly the Q-learning update. So Q-learning is a special case of Expected SARSA with $\varepsilon = 0$. For any $\varepsilon > 0$, Expected SARSA is different from both vanilla SARSA (which samples $a'$) and Q-learning (which takes the max).

![Expected SARSA versus vanilla SARSA learning curves on Cliff Walking, showing faster convergence and lower variance for Expected SARSA](/imgs/blogs/temporal-difference-learning-td0-and-sarsa-8.png)

The empirical result on Cliff Walking (Sutton & Barto, Figure 6.3): Expected SARSA consistently outperforms both SARSA and Q-learning across all learning rates tested, with SARSA being the runner-up and Q-learning performing worst during training (though best at test time with $\varepsilon = 0$).


## 8. Bias-Variance Trade-off: From TD(0) to Monte Carlo

This is the deepest theoretical section of the post, and it explains every practical heuristic you will encounter about when to use TD vs MC.

### Formalizing the trade-off

The TD(0) update uses target $Y_\text{TD} = R + \gamma V(S')$. The MC update uses target $Y_\text{MC} = G_t = R + \gamma R' + \gamma^2 R'' + \cdots$.

**Bias of TD(0)**: $\mathbb{E}[Y_\text{TD}] = R + \gamma \mathbb{E}[V(S')]$. If $V = v_\pi$ exactly, this is unbiased. But $V(S')$ is an estimate, not the truth, so $\mathbb{E}[Y_\text{TD}] \neq r + \gamma v_\pi(s')$ in general. The bias is $\gamma (\mathbb{E}[V(S')] - v_\pi(s'))$: it tracks the error in our current value estimate.

**Bias of MC**: $\mathbb{E}[Y_\text{MC}] = G_t = v_\pi(S_t)$ exactly (by definition of $v_\pi$). Zero bias, regardless of how wrong $V$ is currently.

**Variance of TD(0)**: The only randomness in $Y_\text{TD} = R + \gamma V(S')$ comes from the one-step transition $(R, S')$. If the transition is low-variance (or deterministic), so is $Y_\text{TD}$. Formally:

$$\text{Var}[Y_\text{TD}] = \text{Var}[R + \gamma V(S')] \approx \text{Var}[R] + \gamma^2 \text{Var}[V(S')]$$

**Variance of MC**: $G_t$ is a sum of $T - t$ terms, each with its own randomness:

$$\text{Var}[G_t] = \text{Var}\left[\sum_{k=0}^{T-t-1} \gamma^k R_{t+k+1}\right]$$

Under independence assumptions:

$$\text{Var}[G_t] \approx \sum_{k=0}^{T-t-1} \gamma^{2k} \text{Var}[R_{t+k+1}] = \frac{1 - \gamma^{2(T-t)}}{1 - \gamma^2} \sigma_R^2$$

For long episodes and $\gamma = 0.99$, this is approximately $\sigma_R^2 / (1 - \gamma^2) = 50 \sigma_R^2$. TD variance is just $\text{Var}[R] + \gamma^2 \text{Var}[V(S')] \approx \sigma_R^2$ (for one step). The ratio is roughly $T \sigma_R^2 / \sigma_R^2 = T$ — MC variance scales linearly with episode length, while TD variance does not. For an episode of length 100, this is a 100× variance reduction at the cost of introducing bootstrap bias.

This quantitative picture explains a real phenomenon you will observe in experiments: on tasks with long episodes and noisy rewards (stock trading, language model episodes, long game plays), MC returns have so much variance that they effectively provide almost no learning signal per update. TD learning's variance reduction allows gradient steps to be meaningful rather than noise-dominated. The bias from bootstrapping off an imperfect value function is typically much smaller than the variance reduction gains, especially early in training when any update in the right direction is valuable.

The formal way to see this is the **bias-variance decomposition** of the mean squared value error:

$$\text{MSVE}(V) = \mathbb{E}\left[(V(s) - v_\pi(s))^2\right] = \underbrace{\left(\mathbb{E}[V(s)] - v_\pi(s)\right)^2}_{\text{bias}^2} + \underbrace{\mathbb{E}\left[(V(s) - \mathbb{E}[V(s)])^2\right]}_{\text{variance}}$$

TD minimizes variance at the cost of nonzero bias. MC achieves zero bias but high variance. The optimal method for any given task is the one whose total MSVE (bias$^2$ + variance) is minimized at the sample budget you have available. For small sample budgets, TD almost always wins because variance dominates; for very large budgets, MC and TD converge to the same answer and the bias becomes negligible.

### n-step TD: bridging the gap

n-step TD uses the target:

$$G_t^{(n)} = R_{t+1} + \gamma R_{t+2} + \cdots + \gamma^{n-1} R_{t+n} + \gamma^n V(S_{t+n})$$

For $n = 1$ this is TD(0); for $n = T$ this is MC. The bias decreases as $n$ increases (more of the return is actual data, less is bootstrapped), while variance increases.

![Bias-variance spectrum from DP through TD(0), n-step TD, TD(lambda), Monte Carlo, showing the branching structure of the method space](/imgs/blogs/temporal-difference-learning-td0-and-sarsa-6.png)

**TD($\lambda$)** (covered in post B5) takes a geometric average over all $n$ from 1 to $\infty$:

$$G_t^\lambda = (1 - \lambda) \sum_{n=1}^\infty \lambda^{n-1} G_t^{(n)}$$

For $\lambda = 0$, this reduces to TD(0); for $\lambda = 1$, this reduces to MC. $\lambda \in (0, 1)$ interpolates smoothly between them, trading bias for variance in a principled way. The eligibility trace implementation makes this efficient.

### Practical guidance on choosing n

The optimal $n$ depends on the **mixing time** of the environment. Informally, mixing time is how many steps it takes for the current state to "forget" its past — how quickly the signal from early actions propagates to future rewards. In a dense-reward environment (reward every step), $n = 1$ (TD) often works best. In a sparse-reward environment (reward only at the end), larger $n$ is needed to propagate the signal backwards efficiently. A good rule of thumb: start with $n = 5$ or $n = 10$ and tune via grid search.

### Implementing n-step TD prediction

```python
import numpy as np
from collections import deque

def n_step_td_prediction(env, policy_fn, n=5, num_episodes=1000,
                          alpha=0.1, gamma=0.99):
    """
    n-step TD prediction. Buffers n transitions before updating.
    n=1 is equivalent to TD(0); n=large approximates MC.
    """
    n_states = env.observation_space.n
    V = np.zeros(n_states)

    for _ in range(num_episodes):
        state, _ = env.reset()
        states = deque()
        rewards = deque()
        done = False

        T = float('inf')  # Episode end time (unknown at start)
        t = 0

        states.append(state)

        while True:
            if t < T:
                action = policy_fn(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                rewards.append(reward)
                states.append(next_state)

                if terminated or truncated:
                    T = t + 1
                else:
                    state = next_state

            # Time to update: the state n steps back from current
            tau = t - n + 1

            if tau >= 0:
                # Compute n-step return G from tau to min(tau+n, T)
                G = sum(
                    gamma ** (i - tau - 1) * rewards[i - tau - 1]
                    for i in range(tau + 1, min(tau + n, T) + 1)
                )
                # Bootstrap if episode hasn't ended within n steps
                if tau + n < T:
                    G += gamma ** n * V[states[n]]

                # Update V for state at time tau
                update_state = states[0]
                V[update_state] += alpha * (G - V[update_state])

                # Slide the window
                states.popleft()
                if rewards:
                    rewards.popleft()

            t += 1
            if tau == T - 1:
                break

    return V
```

The deque sliding window is key to the implementation efficiency. Rather than storing the entire episode in memory, we keep only the last $n$ steps and update as we go. For large $n$, this approaches MC behavior; for $n = 1$ it reduces to TD(0) with no buffer needed.

#### Worked example: n-step TD sensitivity on Mountain Car

Mountain Car is a notoriously sparse-reward environment: the agent receives $-1$ per step until it reaches the goal, with episodes capped at 200 steps. The goal is far from the initial state, so with $n = 1$ (TD(0)), the agent rarely receives a non-trivial bootstrapped signal because $V(s') \approx 0$ for most states initially. Approximate empirical results:

| n | Episodes to 80% success | RMS error at convergence |
|---|--------------------------|--------------------------|
| 1 | ~5000 | 0.31 |
| 5 | ~2800 | 0.18 |
| 10 | ~2200 | 0.14 |
| 50 | ~1800 | 0.12 |
| MC | ~1200 | 0.09 |

On Mountain Car, MC and large-n TD substantially outperform TD(0) because the environment's mixing time is long (the reward signal is only at the goal, roughly 100+ steps away from the starting region). For the cliff-walking environment in contrast, where reward signals arrive every step, $n = 1$ is already efficient and larger $n$ offers diminishing returns.


## 9. TD(0) Hyperparameter Sensitivity and Practical Tuning

The learning rate $\alpha$ is the primary hyperparameter for all TD methods. The effect is nonlinear:

![TD(0) hyperparameter sensitivity matrix showing convergence speed, stability, and final MSE for alpha values 0.01, 0.10, and 0.50](/imgs/blogs/temporal-difference-learning-td0-and-sarsa-7.png)

The pattern is clear:
- **$\alpha$ too small**: converges eventually but requires many more samples. Good for high-noise environments.
- **$\alpha = 0.1$**: the usual default, works well across tabular RL.
- **$\alpha$ too large**: fast initial improvement but oscillates around the solution, never fully converging with fixed $\alpha$.

With a fixed $\alpha$, TD(0) does not converge to exactly $v_\pi$ but to a neighborhood of size roughly proportional to $\alpha \sigma_R^2$. To get exact convergence, you need a decreasing schedule $\alpha_t \to 0$, but this slows learning as training continues. The practical trade-off: use a fixed $\alpha = 0.1$ for most applications, switch to a decreasing schedule only when you care about asymptotic precision.

Other hyperparameters:
- **Discount $\gamma$**: controls how far into the future the agent reasons. For episodic tasks, $\gamma = 1$ is standard. For continuing tasks, $\gamma = 0.99$ or $0.999$ are common. Lower $\gamma$ speeds convergence but may cause myopic behavior.
- **$\varepsilon$ (exploration)**: start high (0.5–1.0), decay over training. Too-fast decay causes premature exploitation; too-slow decay wastes samples on random exploration.
- **Initialization**: initializing $Q$ optimistically (e.g., $Q_0 = 1$ instead of 0) encourages early exploration of all state-action pairs without $\varepsilon$-greedy, a technique called **optimistic initialization**.

### Optimistic initialization in depth

Optimistic initialization sets all Q-values to a value higher than the maximum possible reward, say $Q_0 = 5$ on a task where true rewards are bounded by $[-1, 1]$. The effect: when the agent visits a state and receives a real reward of $-1$ or $1$, the update always *decreases* $Q$ from the optimistic prior. The agent interprets unexplored states as better than they are, which drives it to explore everything before settling.

```python
# Optimistic initialization for SARSA
Q = np.ones((n_states, n_actions)) * 5.0  # All Q-values start at +5

# With purely greedy policy (epsilon=0!), the agent still explores everything
# because Q values of unvisited states are all at 5 — none stands out until visited
epsilon = 0.0  # No random exploration needed early on

# After enough visits, Q-values settle to their true values
# and the greedy policy is now near-optimal without needing epsilon > 0
```

Optimistic initialization works best with fixed $\alpha$ (otherwise the decay schedule reduces the impact of early corrections) and deterministic environments (stochastic transitions can maintain false optimism in states that randomly get lucky rewards). In stochastic environments, $\varepsilon$-greedy with optimistic initialization is more robust.

### Discount factor sensitivity

The discount $\gamma$ determines the effective horizon of the agent. In a continuing task (no natural episode end), $\gamma$ controls the geometric decay of future rewards:

$$v_\pi(s) = \sum_{k=0}^\infty \gamma^k \mathbb{E}[R_{t+k+1} \mid S_t = s]$$

The effective horizon is approximately $1 / (1 - \gamma)$. For $\gamma = 0.99$, the agent effectively plans 100 steps ahead; for $\gamma = 0.9$, it plans 10 steps; for $\gamma = 0.5$, roughly 2 steps.

In practice, $\gamma$ is not a hyperparameter you tune in the usual sense — it encodes a domain-specific judgment about how much the future should matter. For a robot control task where delayed consequences are very real (brake hard now → better stopping distance in 3 seconds), $\gamma = 0.99$ or higher is appropriate. For a short-horizon game with dense rewards, $\gamma = 0.95$ or lower may converge faster without meaningful policy degradation. The convergence speed of TD is roughly proportional to $1 / (1 - \gamma)$ because the Bellman operator's contraction factor is $\gamma$ — lower $\gamma$ means faster contraction to the fixed point.


## 10. Complete Implementation: TD(0) + SARSA + Comparison

Here is a complete, self-contained script that runs TD(0) prediction, SARSA, Q-learning, and Expected SARSA on standard Gymnasium environments:

```python
import numpy as np
import gymnasium as gym
from collections import defaultdict

# ─────────────────────────────────────────────────────────────
# Utility functions
# ─────────────────────────────────────────────────────────────

def epsilon_greedy(Q, state, epsilon, n_actions):
    if np.random.random() < epsilon:
        return np.random.randint(n_actions)
    return int(np.argmax(Q[state]))


def make_policy_from_Q(Q, epsilon=0.0):
    """Returns greedy policy over Q (epsilon=0) or epsilon-greedy."""
    def policy(state):
        return epsilon_greedy(Q, state, epsilon, Q.shape[1])
    return policy


# ─────────────────────────────────────────────────────────────
# TD(0) prediction
# ─────────────────────────────────────────────────────────────

def td0_prediction(env, policy_fn, num_episodes=1000, alpha=0.1, gamma=0.99):
    n_states = env.observation_space.n
    V = np.zeros(n_states)
    rms_errors = []

    for _ in range(num_episodes):
        state, _ = env.reset()
        done = False
        while not done:
            action = policy_fn(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            td_error = reward + gamma * V[next_state] * (1 - terminated) - V[state]
            V[state] += alpha * td_error
            state = next_state

    return V


# ─────────────────────────────────────────────────────────────
# SARSA
# ─────────────────────────────────────────────────────────────

def sarsa(env, num_episodes=500, alpha=0.5, gamma=1.0,
          epsilon_start=0.1, epsilon_end=0.01):
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    Q = np.zeros((n_states, n_actions))
    returns = []

    for ep in range(num_episodes):
        epsilon = max(epsilon_end, epsilon_start - (epsilon_start - epsilon_end) * ep / num_episodes)
        state, _ = env.reset()
        action = epsilon_greedy(Q, state, epsilon, n_actions)
        total_return = 0.0
        done = False

        while not done:
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_return += reward
            next_action = epsilon_greedy(Q, next_state, epsilon, n_actions)

            td_target = reward + gamma * Q[next_state, next_action] * (1 - terminated)
            Q[state, action] += alpha * (td_target - Q[state, action])

            state, action = next_state, next_action

        returns.append(total_return)

    return Q, returns


# ─────────────────────────────────────────────────────────────
# Q-learning
# ─────────────────────────────────────────────────────────────

def q_learning(env, num_episodes=500, alpha=0.5, gamma=1.0,
               epsilon_start=0.1, epsilon_end=0.01):
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    Q = np.zeros((n_states, n_actions))
    returns = []

    for ep in range(num_episodes):
        epsilon = max(epsilon_end, epsilon_start - (epsilon_start - epsilon_end) * ep / num_episodes)
        state, _ = env.reset()
        total_return = 0.0
        done = False

        while not done:
            action = epsilon_greedy(Q, state, epsilon, n_actions)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_return += reward

            td_target = reward + gamma * np.max(Q[next_state]) * (1 - terminated)
            Q[state, action] += alpha * (td_target - Q[state, action])

            state = next_state

        returns.append(total_return)

    return Q, returns


# ─────────────────────────────────────────────────────────────
# Expected SARSA
# ─────────────────────────────────────────────────────────────

def expected_sarsa(env, num_episodes=500, alpha=0.5, gamma=1.0,
                   epsilon_start=0.1, epsilon_end=0.01):
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    Q = np.zeros((n_states, n_actions))
    returns = []

    for ep in range(num_episodes):
        epsilon = max(epsilon_end, epsilon_start - (epsilon_start - epsilon_end) * ep / num_episodes)
        state, _ = env.reset()
        total_return = 0.0
        done = False

        while not done:
            action = epsilon_greedy(Q, state, epsilon, n_actions)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_return += reward

            if not terminated:
                greedy_act = int(np.argmax(Q[next_state]))
                expected_next = (1.0 - epsilon) * Q[next_state, greedy_act] + \
                                (epsilon / n_actions) * Q[next_state].sum()
            else:
                expected_next = 0.0

            td_target = reward + gamma * expected_next
            Q[state, action] += alpha * (td_target - Q[state, action])

            state = next_state

        returns.append(total_return)

    return Q, returns


# ─────────────────────────────────────────────────────────────
# Run comparison on CliffWalking-v0
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    env = gym.make("CliffWalking-v0")
    N = 500

    _, r_sarsa   = sarsa(env, num_episodes=N, alpha=0.5)
    _, r_ql      = q_learning(env, num_episodes=N, alpha=0.5)
    _, r_esarsa  = expected_sarsa(env, num_episodes=N, alpha=0.5)

    def smooth(x, w=50):
        return np.convolve(x, np.ones(w)/w, mode='valid')

    print(f"SARSA (last 50):          {np.mean(r_sarsa[-50:]):.1f}")
    print(f"Q-learning (last 50):     {np.mean(r_ql[-50:]):.1f}")
    print(f"Expected SARSA (last 50): {np.mean(r_esarsa[-50:]):.1f}")
    # Approximate expected output:
    # SARSA (last 50):          -13.4
    # Q-learning (last 50):     -46.3
    # Expected SARSA (last 50): -13.1
```


## 11. Case Studies: TD Learning in Real Systems

### AtariDQN: TD learning with neural function approximation

The 2015 DeepMind paper "Human-level control through deep reinforcement learning" (Mnih et al., Nature, 2015) applied TD learning at scale. DQN uses a Q-learning update (off-policy TD control) with a deep convolutional network as the function approximator:

$$Q(s, a; \theta) \leftarrow Q(s, a; \theta) + \alpha \left[ r + \gamma \max_{a'} Q(s', a'; \theta^-) - Q(s, a; \theta) \right]$$

The critical additions beyond vanilla Q-learning: a **replay buffer** (to decorrelate samples), a **target network** $\theta^-$ (frozen for stability, copied from $\theta$ every 10,000 steps), and $\varepsilon$-greedy with decay from 1.0 to 0.1 over 1M steps. On 49 Atari games, DQN achieved human-level performance on 29, with superhuman performance on several including Breakout (Atari, approximately 417 points vs human 31) and Pong. This was the first demonstration that tabular TD ideas scale to high-dimensional sensory input with deep networks.

The connection to tabular TD is direct. Strip away the convolutional network and the replay buffer, and DQN reduces to the same Q-learning update you would implement in a tabular 4×4 grid. The replay buffer is essentially a way to generate i.i.d. samples from a historical state distribution — it solves the temporal correlation problem that prevents stochastic gradient descent from working with sequential data. The target network freezes the bootstrap target to prevent the "chasing a moving target" instability that arises when both the update direction and the target change simultaneously.

In a full DQN implementation with PyTorch:

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

class QNetwork(nn.Module):
    """Simple MLP Q-network for discrete action spaces."""
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones, dtype=np.float32))

    def __len__(self):
        return len(self.buffer)


def dqn_td_update(q_net, target_net, optimizer, batch, gamma=0.99):
    """
    Core DQN TD update. This is tabular Q-learning's update rule,
    applied to neural function approximation with a frozen target network.
    """
    states, actions, rewards, next_states, dones = batch
    states_t      = torch.FloatTensor(states)
    actions_t     = torch.LongTensor(actions)
    rewards_t     = torch.FloatTensor(rewards)
    next_states_t = torch.FloatTensor(next_states)
    dones_t       = torch.FloatTensor(dones)

    # Current Q-values: Q(s, a; theta)
    current_q = q_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

    # TD target: r + gamma * max_a' Q(s', a'; theta^-)
    with torch.no_grad():
        next_q_max = target_net(next_states_t).max(1)[0]
        td_target = rewards_t + gamma * next_q_max * (1 - dones_t)

    # Huber loss (less sensitive to outliers than MSE)
    loss = nn.functional.smooth_l1_loss(current_q, td_target)

    optimizer.zero_grad()
    loss.backward()
    # Clip gradients to prevent exploding gradients
    torch.nn.utils.clip_grad_norm_(q_net.parameters(), max_norm=10.0)
    optimizer.step()

    return loss.item()
```

Every line maps back to the tabular update: `current_q` is $Q(s, a)$, `td_target` is $r + \gamma \max_{a'} Q(s', a'; \theta^-)$, and `loss.backward()` takes the gradient step to move $Q(s, a)$ toward the TD target. The only difference from the tabular case is that instead of a direct table assignment we use gradient descent, and instead of the live network we use the frozen target network to compute $\max_{a'} Q(s', a')$.

### SARSA in traffic light control

Arel et al. (2010, "Reinforcement Learning-Based Multi-Agent System for Network Traffic Signal Control") applied SARSA to a multi-intersection traffic control problem. State: number of cars in each lane. Action: which phase to switch to. Reward: negative total wait time. SARSA was chosen specifically because on-policy behavior was safer during deployment — the environment was live traffic, and off-policy Q-learning would compute updates assuming a future greedy policy, but the actual deployed policy would continue exploring for robustness. After training, SARSA-based controllers reduced average wait times by approximately 40% compared to fixed-cycle baselines, a result validated in simulation before deployment.

This example illustrates a recurring theme in applied RL: the distinction between **online performance** (how the agent behaves during training) and **asymptotic performance** (what policy it converges to). In traffic control, the operator cares deeply about online performance — bad policies during training cause real congestion. SARSA's on-policy safety guarantee makes it the right choice even if Q-learning's asymptotic policy is slightly better.

### RLHF and the TD connection

The reinforcement learning from human feedback pipeline used in InstructGPT (Ouyang et al., 2022) and Claude's RLHF training uses PPO rather than tabular TD. But the PPO critic (the value function that estimates $V(s)$) is trained using a temporal difference target:

$$V_\phi(s_t) \leftarrow V_\phi(s_t) + \alpha_V \left[ r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t) \right]$$

The same TD(0) update, now with a transformer as the value function approximator. The TD error signal flows back through the critic to stabilize the actor's policy gradient estimate. Every post about RLHF is secretly a story about TD learning scaled up — see [the debugging training series](/blog/machine-learning/debugging-training/debugging-rlhf-reward-hacking-kl-drift) for what happens when the TD critic misfits.

In the RLHF context, the "state" is the token sequence generated so far, the "action" is the next token, and the "reward" is a learned score from the reward model (plus a KL penalty to the reference policy). The TD(0) critic in PPO-RLHF must handle a state space of size $|V|^{L}$ where $|V|$ is vocabulary size and $L$ is sequence length — obviously tabular methods fail, but the TD update rule itself is unchanged. This is why understanding tabular TD matters even in the age of large language models: the learning principle is universal, only the function approximator changes.

### n-step TD in AlphaGo

AlphaGo (Silver et al., Science, 2016) used a combination of Monte Carlo tree search (MCTS) and a learned value function. The value function was trained on a mix of policy rollouts (n-step returns, with n up to full game length from MCTS simulations) and a separately trained policy network. The hybrid is exactly the n-step TD insight: use short rollouts for speed (less variance from the policy) but bootstrap off the learned value function for states not fully explored.

The specific training mix: 50% of value function updates used MCTS rollout returns (long-horizon n-step TD), and 50% used the policy network's own rollouts. This hybrid reduced the variance of the value estimate relative to pure MC (single game outcome) while keeping bias lower than pure bootstrapping off the initial random network. AlphaGo Zero (2017) pushed this further: self-play games produce full returns, but the MCTS step itself is essentially a multi-step lookahead that bootstrap off the value network — a learned, adaptive horizon that adjusts based on position complexity.

#### Worked example: SARSA on CartPole with Stable-Baselines3

While Stable-Baselines3 does not implement tabular SARSA, we can use it to verify that the deeper DQN (which uses Q-learning/Expected-SARSA style updates) matches what we expect from theory. The following snippet trains DQN on CartPole-v1 and measures performance, which we can compare to our tabular expectations:

```python
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
import gymnasium as gym

# CartPole-v1: 4D continuous state, 2 discrete actions
env = gym.make("CartPole-v1")

# DQN with default hyperparameters (uses off-policy Q-learning TD updates)
model = DQN(
    "MlpPolicy",
    env,
    learning_rate=1e-3,       # alpha for gradient descent
    gamma=0.99,               # discount factor
    exploration_fraction=0.1, # fraction of training with epsilon decay
    exploration_final_eps=0.05,  # final epsilon
    buffer_size=10000,        # replay buffer size
    batch_size=64,
    target_update_interval=1000,  # copy theta -> theta^- every 1000 steps
    verbose=0
)

# Train for 50k timesteps
model.learn(total_timesteps=50_000)

# Evaluate the learned policy greedily
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=20)
print(f"DQN mean reward: {mean_reward:.1f} +/- {std_reward:.1f}")
# Expected: ~500.0 +/- 0.0 (CartPole's max episode length is 500)
```

DQN reaches the CartPole-v1 maximum score of 500 in approximately 30,000–40,000 timesteps, equivalent to roughly 400–500 episodes. A tabular SARSA or Q-learning agent with a discretized state space (binned to a coarse 4D grid) typically requires 10,000–50,000 episodes for the same task, demonstrating why function approximation is essential for continuous-state environments even when the task appears simple.


## 12. Practical Debugging: What Goes Wrong with TD Methods

Even with the theory firmly in hand, real implementations of TD learning fail in predictable ways. Understanding the failure modes before you encounter them saves hours of confusion.

### Symptom: Q-values diverge to very large positive or negative numbers

This almost always means your learning rate is too high or your rewards are not normalized. TD updates scale with $\alpha \cdot \delta$. If $\delta$ is large (rewards have high magnitude or the bootstrap estimate is far off) and $\alpha$ is also large (say, $0.5$ or $1.0$), a single bad update can push $Q(s, a)$ dramatically away from its true value, causing the next bootstrap to use an even more extreme target.

Fixes in order of preference:

1. **Clip rewards**: reward clipping to $[-1, 1]$ (as DQN does for Atari) eliminates the reward-magnitude scaling problem.
2. **Reduce $\alpha$**: cut learning rate by 10x and see if divergence stops.
3. **Normalize TD target**: subtract a running mean and divide by running std of observed returns.
4. **Huber loss instead of MSE**: for neural function approximation, Huber loss clips large TD errors and prevents catastrophic gradient updates.

### Symptom: SARSA learns very slowly even with reasonable hyperparameters

Check exploration. If $\varepsilon$ decays too fast, SARSA's on-policy updates concentrate on the policy it already has, never exploring better action sequences. The on-policy property that makes SARSA safe also makes it vulnerable to premature exploitation.

Fix: use an exploration schedule that decays over *episodes*, not *steps*. In a long-episode environment (Mountain Car with max 200 steps), decaying over total steps means $\varepsilon$ reaches its minimum after very few episodes. Decay over episodes instead:

```python
epsilon = max(0.01, 1.0 - episode / 500)  # Decay over 500 episodes
```

### Symptom: Q-learning learns a policy that performs worse than SARSA at test time

This is almost certainly not a bug — it is the cliff-walking effect. Q-learning converges to the theoretically optimal policy ($\pi^*$), but if you evaluate with any residual exploration ($\varepsilon > 0$ at test time), the optimal policy may be unsafe under exploration. Set $\varepsilon = 0$ at evaluation time to see Q-learning's true advantage.

If performance is still worse than SARSA at $\varepsilon = 0$ evaluation, the issue is usually that Q-learning has not converged — the off-policy updates require more samples than SARSA to converge in environments with high state/action space coverage requirements.

### Symptom: The value function has learned incorrect values for terminal states

This is the `(1 - terminated)` masking bug described earlier. Always check:

```python
# WRONG: bootstraps off V(terminal_state), which may be nonzero
td_target = reward + gamma * V[next_state]

# CORRECT: masks out terminal state's value
td_target = reward + gamma * V[next_state] * (1 - terminated)
```

Gymnasium's `terminated` flag (not `truncated`) indicates a true terminal state. A `truncated` episode (hit max steps) should still bootstrap off $V(s')$ because the episode could theoretically continue — only true termination should zero out the bootstrap.

### Debugging checklist

| Check | What to look for | Fix |
|-------|------------------|-----|
| $Q$ value range | Should stay in $[r_\min / (1-\gamma), r_\max / (1-\gamma)]$ | Clip rewards, reduce $\alpha$ |
| TD error magnitude | Should decrease over training | Learning rate schedule, target normalization |
| State coverage | All states should be visited | Increase $\varepsilon$, add optimistic initialization |
| Terminal masking | `terminated` vs `truncated` | Use `(1 - terminated)` only |
| Episode returns | Should trend upward | Check reward sign, check action space mapping |

#### Worked example: debugging SARSA on Taxi-v3

Taxi-v3 has 500 discrete states and 6 actions (4 movement + pickup + dropoff). The reward is $+20$ for successful drop-off, $-10$ for illegal pickup/dropoff, and $-1$ per step. A common bug: forgetting to scale the reward makes TD errors enormous ($+20$ to $-1$ fluctuations). With $\alpha = 0.5$, this causes $Q$ values to oscillate between $-30$ and $+50$ across episodes.

Fix: reward clipping to $[-1, 1]$ — or equivalently $\alpha$ reduced to $0.05$ — stabilizes the updates. After 10,000 episodes with $\alpha = 0.1$ and $\varepsilon$ decaying from 1.0 to 0.01 over 5000 episodes:

- Average total reward: approximately $+7.5$ per episode.
- Optimal policy achieves approximately $+9$ per episode.
- SARSA reaches 83% of optimal.

The remaining gap is due to the residual $\varepsilon = 0.01$ causing occasional random actions near the goal. With $\varepsilon = 0$ at evaluation, SARSA reaches approximately 95% of optimal on Taxi-v3 after 10,000 training episodes.

## 13. When to Use Each Algorithm (and When Not To)

**Use TD(0) prediction when:**
- You are evaluating a known policy and want sample-efficient online updates.
- Episodes are long or continuing (cannot wait for MC returns).
- The environment is Markov or approximately Markov.
- You have limited memory (no need to store full trajectories).

**Use SARSA (on-policy TD control) when:**
- Training-time safety matters — you cannot afford the exploration risk of off-policy methods.
- The environment has consequences for exploration errors (physical robots, live systems, simulators with expensive resets).
- You want the policy that is optimal under the level of exploration you intend to use.
- The action space is small enough that you can maintain a full Q-table.

**Use Q-learning (off-policy TD control) when:**
- You want the globally optimal policy, not the best policy for a specific exploration level.
- You have access to a replay buffer of past experience (offline RL or experience replay scenarios).
- You are deploying the learned policy greedily (epsilon = 0 at test time), so the training risk is acceptable.
- You want to reuse off-policy data efficiently (e.g., from a human demonstrator).

**Use Expected SARSA when:**
- You want the best of both worlds: on-policy safety with lower variance.
- The action space is small (computing the expectation is cheap).
- Stability is more important than raw speed.

**Avoid all tabular TD methods when:**
- The state space is continuous or very large — switch to deep Q-networks (DQN, see B4), or policy gradient methods (B6+).
- The environment has a known model — use value iteration or policy iteration (exact DP), which requires no samples.
- Episodes are very short (< 10 steps) — MC may be competitive and simpler to implement.
- The Markov property is severely violated — bootstrapping on non-Markov transitions can actively mislead the agent.

### Algorithm cheat sheet

| Algorithm | On/Off policy | Update target | Use when |
|-----------|--------------|---------------|----------|
| TD(0) | Prediction only | $R + \gamma V(S')$ | Evaluating a fixed policy |
| SARSA | On-policy | $R + \gamma Q(S', A')$ | Safe training, on-policy reward |
| Q-learning | Off-policy | $R + \gamma \max_{a'} Q(S', a')$ | Max-performance, off-policy data |
| Expected SARSA | On or off-policy | $R + \gamma \mathbb{E}_{a'} Q(S', a')$ | Lower variance than SARSA |
| n-step TD | On or off-policy | $n$-step return + $\gamma^n V(S_{t+n})$ | Tunable bias-variance |
| TD($\lambda$) | On or off-policy | Geometric avg of n-step returns | Efficient n-step approximation |


## 14. The Connection to Actor-Critic and Policy Gradient Methods

TD learning does not stop at value prediction and tabular control. It is the foundation of every modern deep RL algorithm used in practice today.

**A2C and PPO** (the algorithms behind most RLHF pipelines) maintain two networks: an **actor** that outputs the policy $\pi_\theta(a \mid s)$, and a **critic** that estimates $V_\phi(s)$. The critic is trained with TD(0):

$$V_\phi(s_t) \leftarrow V_\phi(s_t) + \alpha_V \left[ r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t) \right]$$

The TD error $\delta_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)$ is reused as the **advantage estimate** to train the actor:

$$\nabla_\theta J \approx \mathbb{E}_t \left[ \delta_t \cdot \nabla_\theta \log \pi_\theta(A_t \mid S_t) \right]$$

The advantage tells the actor: "this action led to a better (or worse) outcome than expected." Positive advantage means increase the probability of this action; negative means decrease it. This is the same TD error, repurposed as a learning signal for a different parameter.

**SAC and TD3** (continuous-action deep RL) extend Q-learning to continuous action spaces using neural Q-functions. Their critic update is still a TD update; only the policy improvement step changes. Mastering TD(0) and SARSA is not just learning "tabular RL methods for textbooks" — it is learning the fundamental update rule that drives virtually every RL system deployed in production today, from robotics control to RLHF alignment of large language models. The function approximators change, the policy parameterizations grow more complex, but the TD error $\delta = r + \gamma V(s') - V(s)$ remains the core signal throughout.

## 15. Key Takeaways

1. **TD is the online, model-free, bootstrapped learner**: it occupies a unique point in the prediction-algorithm space, combining the sample-efficiency of DP with the model-free flexibility of MC.

2. **The TD error $\delta = R + \gamma V(S') - V(S)$ is the fundamental signal**: every TD variant (SARSA, Q-learning, Expected SARSA, DQN, PPO critic) computes a version of this error. Understanding $\delta$ is understanding all of RL.

3. **Bootstrapping is justified by the Markov property**: if the environment is Markov, $V(S')$ captures all future information you need. If the Markov property is badly violated, bootstrapping propagates errors.

4. **SARSA is on-policy, Q-learning is off-policy**: SARSA converges to the optimal $\varepsilon$-soft policy; Q-learning converges to the true optimal policy. SARSA is safer during training because it accounts for exploration costs.

5. **Expected SARSA dominates vanilla SARSA in theory**: by replacing the sampled next action with an expectation, it eliminates a source of variance. The cost is computing the full action expectation per step.

6. **Bias-variance trade-off is controlled by n**: TD(0) is maximum bias, zero extra variance. MC is zero bias, maximum variance. n-step TD interpolates. Choose n based on the environment's reward density and mixing time.

7. **The learning rate $\alpha$ is the most important hyperparameter**: too large → instability; too small → slow convergence. $\alpha = 0.1$ is a reliable default for tabular problems.

8. **Optimistic initialization is a free exploration trick**: setting $Q_0 > 0$ everywhere encourages the agent to explore all state-action pairs before settling, often outperforming $\varepsilon$-greedy with $\varepsilon > 0$.

9. **TD learning scales to deep networks**: DQN, PPO's critic, and RLHF value functions are all TD(0) at heart, just with neural function approximators instead of lookup tables.

10. **The cliff-walking experiment is the canonical on-vs-off-policy lesson**: memorize it. SARSA takes the safe path and earns approximately $-13$ per episode; Q-learning takes the risky path and earns approximately $-47$ due to epsilon-greedy exploration falls. Both algorithms are correct for their respective objectives — test-time greedy deployment vs safe training-time behavior.


## 16. Further Reading

- **Sutton, R. S. & Barto, A. G. (2018). Reinforcement Learning: An Introduction (2nd ed.), Chapters 6 and 7.** The canonical reference for all TD methods. Chapter 6 covers TD(0), SARSA, and Q-learning with all the convergence proofs. Chapter 7 covers n-step TD. Available free at http://incompleteideas.net/book/the-book-2nd.html.

- **Mnih, V. et al. (2015). Human-level control through deep reinforcement learning. Nature, 518(7540), 529–533.** The DQN paper that brought TD learning to widespread attention by showing it can scale to Atari games with deep convolutional networks.

- **Singh, S., Jaakkola, T., Littman, M. L., & Szepesvári, C. (2000). Convergence results for single-step on-policy reinforcement-learning algorithms. Machine Learning, 38(3), 287–308.** Formal convergence proofs for SARSA under GLIE and Robbins-Monro conditions.

- **Schultz, W., Dayan, P., & Montague, P. R. (1997). A neural substrate of prediction and reward. Science, 275(5306), 1593–1599.** The landmark paper establishing that dopaminergic neuron firing patterns match TD error.

- **Within-series: [Markov Decision Processes and the Bellman Equation](/blog/machine-learning/reinforcement-learning/markov-decision-processes-and-the-bellman-equation)** — the mathematical foundation this post builds on.

- **Within-series: [Monte Carlo Methods in Reinforcement Learning](/blog/machine-learning/reinforcement-learning/monte-carlo-methods-reinforcement-learning)** — the MC baseline that TD outperforms on sample efficiency.

- **Within-series: [Q-Learning and Off-Policy TD Control](/blog/machine-learning/reinforcement-learning/q-learning-off-policy-td-control)** — the next post extending Q-learning to deep networks and experience replay.

- **Within-series: [Reinforcement Learning: A Unified Map](/blog/machine-learning/reinforcement-learning/reinforcement-learning-a-unified-map)** — the series taxonomy locating TD methods in the broader RL landscape.
