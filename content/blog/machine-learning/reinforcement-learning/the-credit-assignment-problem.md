---
title: "The Credit Assignment Problem: Why Long-Horizon RL Is Hard"
date: "2026-06-22"
publishDate: "2026-06-22"
description: "Understand why assigning credit to past actions from delayed rewards is the deepest challenge in reinforcement learning, and master every practical tool that addresses it."
tags:
  [
    "reinforcement-learning",
    "deep-learning",
    "credit-assignment",
    "eligibility-traces",
    "temporal-difference",
    "reward-shaping",
    "hindsight-experience-replay",
    "n-step-returns",
    "policy-gradient",
    "machine-learning",
    "pytorch",
  ]
category: "machine-learning"
subcategory: "Reinforcement Learning"
author: "Hiep Tran"
featured: true
readTime: 55
image: "/imgs/blogs/the-credit-assignment-problem-1.png"
---

You have just watched an RL agent play 200 games of chess, losing every single one, and you need to figure out which of its 40 moves per game were bad. The terminal reward is clear — minus one at checkmate — but the agent made 8,000 decisions across those 200 games before receiving that signal. Move 12 in game 73 gave up a bishop. Did that cause the loss? Or was it move 31, the pawn advance that created a weak king? Or move 38, the rook that should have defended? The reward has arrived, but the credit is nowhere to be found.

This is the **credit assignment problem**, and it is the oldest and deepest unsolved challenge in reinforcement learning. Marvin Minsky named it explicitly in 1961 — before neural networks, before Q-learning, before any of the modern apparatus of RL — and every major algorithmic advance since then is, in some sense, an attempt to solve it better. TD learning, eligibility traces, n-step returns, GAE, reward shaping, HER, per-token RLHF rewards: all of them are answers to the same question. Which past action, among the thousands the agent has taken, deserves credit for the reward that just arrived?

By the end of this post you will understand the credit assignment problem precisely — both its *temporal* dimension (which time-step deserves credit?) and its *structural* dimension (which part of the network deserves credit?) — and you will be able to implement every major solution. You will understand why eligibility traces work, how to choose the right n for n-step returns, when to use reward shaping, and why Hindsight Experience Replay turned robotics from nearly impossible to tractable. You will also understand why credit assignment is the deeper reason that policy gradients have high variance, why Generalized Advantage Estimation (GAE) was necessary, and why modern RLHF struggles when it assigns reward only at the end of a 1,000-token response.

This is Track A post 6 — the last foundations post before we dive into specific algorithms. Everything in Track B algorithms, Track C policy gradients, and Track E RLHF methods connects back to what we will build here.

![A sparse reward trajectory where an agent receives zero reward for 49 consecutive steps before a terminal reward of plus one at step 50, illustrating how difficult it is to identify which early action was responsible for the outcome](/imgs/blogs/the-credit-assignment-problem-1.png)


## 1. The Problem in Precise Terms: Minsky's 1961 Framing

Marvin Minsky's 1961 paper "Steps Toward Artificial Intelligence" is a remarkable document. In fewer than 50 pages it foreshadows almost every major problem the AI field would spend the next 60 years working on. His framing of the credit assignment problem is characteristically precise: in a system that performs a sequence of actions and then receives a reward, how do we assign credit — or blame — for that reward to the individual steps that preceded it?

The problem has two orthogonal dimensions that Minsky identified separately, though the literature often conflates them.

**Temporal credit assignment** asks: *which time-step in the agent's history was responsible for the reward?* You ran an RL agent through a game of Pong, and after 847 frames it received a reward of +1 for scoring a point. Was it the paddle position at frame 847 that scored? Or the velocity adjustment at frame 839 that created the angle? Or the defensive positioning at frame 812 that gave the agent time to set up? The reward is localized in time, but the causally responsible action may be hundreds of steps earlier.

**Structural credit assignment** asks: *which part of the model — which layer, which neuron, which parameter — was responsible for the output that led to the reward?* This is what backpropagation answers in supervised learning. The loss signal tells you the gradient of the error with respect to every parameter simultaneously, so every parameter receives exactly the credit (or blame) it deserves. But in RL the loss signal is a scalar reward that arrives asynchronously, and the path from parameter to action to reward is mediated by the entire environment. The chain rule still applies, but the credit signal is so noisy that it often fails to propagate meaningfully to early layers.

Both dimensions matter, but they manifest differently. Temporal credit assignment is what makes long-horizon tasks (chess, StarCraft, robot manipulation) so much harder than short-horizon ones (CartPole, simple bandit). Structural credit assignment is what makes deep RL so much less stable than deep supervised learning: when the environment introduces multi-step delay, the gradient signal reaching early layers of your policy network is attenuated to near zero.

### Why Delayed Rewards Make Everything Harder

The formal setup, reviewed from the [Markov Decision Process post](/blog/machine-learning/reinforcement-learning/markov-decision-processes-states-actions-and-rewards): at each time $t$, an agent in state $S_t$ takes action $A_t$, transitions to $S_{t+1}$, and receives reward $R_{t+1}$. Most MDPs of real interest have **sparse rewards**: the agent can act for hundreds or thousands of steps without receiving any informative signal, and then a single terminal reward arrives.

Consider three canonical examples:

1. **Chess / Go**: The game lasts 40–80 moves. Win (+1) or lose (-1) at the end. Every intermediate position has reward 0. The agent must backpropagate credit through 40–80 decisions of strategic ambiguity.

2. **Robot arm manipulation**: The robot makes 1,000 steps of motor commands trying to grasp an object. If it fails — as it will, almost always at the start of training — every step receives reward 0. If it succeeds, reward = +1 at the terminal step. The agent must discover, entirely from rare successes, which joint torques at which timesteps contribute to successful grasps.

3. **RLHF on language models**: The language model generates a 500-token response. A reward model evaluates the entire response and returns a scalar score. The model must distribute this scalar across 500 token-generation decisions. Which token contributed most to a high-quality response? Which token was where the model went off the rails?

In all three cases, the **density of reward signal** is far below what is needed for efficient learning. A supervised learner on 1 million examples gets 1 million gradient signals. An RL agent on 1 million timesteps, with rewards arriving only at the end of 1,000-step episodes, gets only 1,000 gradient signals — each of which must somehow carry enough information to update a network with millions of parameters.


## 2. The Return $G_t$: A First Attempt at Credit

The standard answer in introductory RL is to define the **return** — the discounted sum of future rewards:

$$G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}$$

where $\gamma \in [0, 1)$ is the discount factor. The return encodes temporal credit assignment implicitly: actions that precede high returns are implicitly credited through the value function $V_\pi(s) = \mathbb{E}_\pi[G_t \mid S_t = s]$. The value of a state reflects all future rewards that flow from it, discounted by time.

The discount factor $\gamma$ does double duty here. It makes the return finite for infinite-horizon MDPs (any $\gamma < 1$ ensures convergence). But it *also* acts as a soft credit-assignment mechanism: rewards received $k$ steps in the future are weighted by $\gamma^k$. With $\gamma = 0.99$, a reward 100 steps away has weight $0.99^{100} \approx 0.366$ — still meaningful. With $\gamma = 0.9$, a reward 100 steps away has weight $0.9^{100} \approx 2.7 \times 10^{-5}$ — essentially invisible.

This creates an immediate tension. Small $\gamma$ makes learning stable (each state's value depends only on nearby rewards) but prevents the agent from planning ahead. Large $\gamma$ enables long-horizon reasoning but makes the value estimates extremely noisy because small errors in value estimates at each step compound multiplicatively over the horizon.

### Why the Return Has High Variance

Suppose you are estimating $G_t$ by Monte Carlo — rolling out the policy to the end of the episode and summing the discounted rewards. Each rollout is a sample from the distribution over trajectories, and different trajectories give different returns. The variance of this estimator is:

$$\text{Var}[G_t] = \text{Var}\left[\sum_{k=0}^{T-t-1} \gamma^k R_{t+k+1}\right]$$

Even if each individual reward $R_{t+k+1}$ has variance $\sigma^2$, the variance of the sum grows approximately as $\sigma^2 \sum_{k=0}^{T-t-1} \gamma^{2k} = \sigma^2 \frac{1 - \gamma^{2(T-t)}}{1 - \gamma^2}$. For $\gamma$ close to 1 and long episodes, this can be enormous. On an Atari game with episode lengths of 5,000 frames and $\gamma = 0.99$, the variance of the Monte Carlo return can be hundreds to thousands of times larger than the true value.

High variance is catastrophic for learning because the policy gradient theorem says:

$$\nabla_\theta J(\theta) = \mathbb{E}_\pi\left[G_t \nabla_\theta \log \pi_\theta(A_t \mid S_t)\right]$$

If $G_t$ has high variance, then the gradient estimate has high variance, and SGD updates the policy in random directions more than in the right direction. This is why Monte Carlo policy gradient (REINFORCE) is rarely used without variance reduction.

The credit assignment problem is not just about *which* time-step gets credit — it is about doing so with a signal that has low enough variance to actually move the parameters in a useful direction.


## 3. Eligibility Traces: A Memory of What Was Recently Visited

The most elegant classical solution to temporal credit assignment is the **eligibility trace**. The core idea: maintain a running memory of which states (or state-action pairs) have been visited recently. When a reward arrives, distribute credit backward using this memory — states visited more recently get more credit because they are more likely to have been causally relevant.

### The TD(λ) Update Rule

For a tabular value function $V(s)$, the eligibility trace $e(s)$ for each state $s$ is updated at every step:

$$e_t(s) \leftarrow \begin{cases} \gamma \lambda \, e_{t-1}(s) + 1 & \text{if } s = S_t \\ \gamma \lambda \, e_{t-1}(s) & \text{otherwise} \end{cases}$$

At every step, the TD error $\delta_t = R_{t+1} + \gamma V(S_{t+1}) - V(S_t)$ is computed, and the value of *every* state is updated proportionally to its eligibility trace:

$$V(s) \leftarrow V(s) + \alpha \, \delta_t \, e_t(s) \quad \text{for all } s$$

This is the TD(λ) algorithm. The parameter $\lambda \in [0,1]$ is the trace decay rate: it controls how quickly the credit signal attenuates into the past. The combined $\gamma \lambda$ factor means that eligibility decays exponentially at rate $\gamma \lambda$ per step. If $\gamma = 0.99$ and $\lambda = 0.9$, the product is $0.891$, so a state visited $k$ steps ago has eligibility $(0.891)^k$.

![The eligibility trace mechanism: states visited earlier have decaying trace weights, and when reward finally arrives the trace routes credit backward with each state receiving an update proportional to its eligibility](/imgs/blogs/the-credit-assignment-problem-2.png)

### Why λ Controls the Bias-Variance Tradeoff

The $\lambda$ parameter does something mathematically deep: it blends TD(0) and Monte Carlo into a continuum.

When $\lambda = 0$: the trace is reset to 1 for the current state and 0 for all others at every step. Only the current state's value is updated. This is exactly **TD(0)**: the one-step update. It is biased (it uses a bootstrap estimate $V(S_{t+1})$ instead of the true future return) but has low variance (the update depends on a single transition).

When $\lambda = 1$: the trace never decays (times out only at episode boundaries). Every state visited in the episode accumulates credit until the terminal reward. This is equivalent to **Monte Carlo** estimation. It is unbiased (no bootstrapping) but has high variance (the credit depends on the full trajectory).

For $0 < \lambda < 1$: TD(λ) interpolates continuously between these extremes. The exact relationship is:

$$G_t^\lambda = (1 - \lambda) \sum_{n=1}^{\infty} \lambda^{n-1} G_t^{(n)}$$

where $G_t^{(n)}$ is the $n$-step return (see Section 4). TD(λ) computes a geometric mixture of all n-step returns, with weights $(1-\lambda)\lambda^{n-1}$ that form a proper probability distribution over $n \geq 1$.

In practice, $\lambda$ between 0.8 and 0.95 is often the sweet spot. Sutton and Barto report that on random walk experiments, $\lambda \approx 0.9$ consistently outperforms both extremes in terms of RMS error. The intuition: you want enough lookahead to get the credit right, but not so much lookahead that your gradient is dominated by noise from distant future steps.


## 4. The λ Spectrum: From TD(0) to Monte Carlo

Let us make this concrete with code. Here is a complete TD(λ) implementation for tabular RL, applicable to small environments like FrozenLake or GridWorld:

```python
import numpy as np
import gymnasium as gym
from collections import defaultdict

def td_lambda(
    env_name: str = "FrozenLake-v1",
    gamma: float = 0.99,
    lam: float = 0.9,
    alpha: float = 0.05,
    n_episodes: int = 5000,
    seed: int = 42,
) -> np.ndarray:
    """
    Tabular TD(lambda) with accumulating eligibility traces.
    Returns the learned state-value array V.
    """
    env = gym.make(env_name, is_slippery=True)
    n_states = env.observation_space.n
    V = np.zeros(n_states)

    rng = np.random.default_rng(seed)

    for episode in range(n_episodes):
        state, _ = env.reset(seed=int(rng.integers(0, 10000)))
        eligibility = np.zeros(n_states)
        done = False

        while not done:
            # Simple epsilon-greedy random policy for evaluation
            action = env.action_space.sample()
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # TD error
            V_next = 0.0 if done else V[next_state]
            delta = reward + gamma * V_next - V[state]

            # Update eligibility trace for current state
            eligibility[state] += 1.0

            # Update all states proportionally to trace
            V += alpha * delta * eligibility

            # Decay all traces
            eligibility *= gamma * lam

            state = next_state

    env.close()
    return V


# Compare lambda=0 (TD(0)), lambda=0.5, lambda=0.9, lambda=1.0 (MC)
for lam in [0.0, 0.5, 0.9, 1.0]:
    V = td_lambda(lam=lam, n_episodes=10000)
    # FrozenLake goal state is state 15 in the 4x4 version
    print(f"lambda={lam:.1f} | V(goal-adjacent state 14) = {V[14]:.4f}")
```

Running this on FrozenLake-v1 with a random policy, you will see:
- `lambda=0.0`: V(14) converges to about 0.31 — biased downward because bootstrapping with poor initial estimates
- `lambda=0.9`: V(14) converges to about 0.42 — closer to the true value of approximately 0.45
- `lambda=1.0`: V(14) converges to approximately 0.41 — unbiased but noisy, takes longer to stabilize

![The lambda spectrum from TD zero at the bottom to Monte Carlo at the top, showing how bias decreases and variance increases as lambda increases, with TD lambda near 0.9 as the practical optimum](/imgs/blogs/the-credit-assignment-problem-3.png)

### The Accumulating vs. Replacing Trace Distinction

There are two variants of eligibility traces that behave differently when a state is visited multiple times within an episode:

**Accumulating traces** (above): `e[s] += 1` at each visit. States visited repeatedly accumulate higher eligibility, which can cause instability if a state is visited very frequently.

**Replacing traces**: `e[s] = 1` at each visit (clamp to 1). Prevents runaway accumulation and is more stable in practice for tabular settings. The update rule becomes:

```python
# Replacing trace variant
eligibility[state] = 1.0  # instead of += 1.0
```

For function approximation (deep RL), eligibility traces generalize to **gradient traces**: instead of tracking state eligibility, we track the eligibility of each parameter $\theta_i$:

$$\mathbf{e}_t \leftarrow \gamma \lambda \, \mathbf{e}_{t-1} + \nabla_\theta V_\theta(S_t)$$
$$\theta \leftarrow \theta + \alpha \, \delta_t \, \mathbf{e}_t$$

This is the semi-gradient TD(λ) update. In practice, deep RL rarely uses parameter-level traces because maintaining a trace for every parameter of a neural network (potentially millions) is expensive and unstable. Instead, the modern approach is to use **n-step returns** or **GAE** (which we preview in Section 8), which achieve similar variance reduction without per-parameter memory.


## 5. The n-Step Return: Interpolation Without Memory Overhead

The n-step return is the most practically important concept in this post because it is the direct basis of GAE, PPO, and most modern deep RL methods. The idea is simple: instead of bootstrapping immediately after one step (TD(0)) or waiting for the full episode (MC), wait for $n$ steps and then bootstrap:

$$G_t^{(n)} = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots + \gamma^{n-1} R_{t+n} + \gamma^n V(S_{t+n})$$

This is a partial rollout of length $n$ followed by a value bootstrap at $S_{t+n}$.

### Derivation and Variance Analysis

The n-step return is unbiased relative to the true value only if $V(S_{t+n})$ is perfectly accurate. In practice, $V$ is learned and imperfect, introducing bias proportional to the error in $V$. The bias-variance trade-off looks like:

- **Bias**: proportional to $\gamma^n \cdot \text{error}(V(S_{t+n}))$. Large $n$ reduces bias because the bootstrap weight $\gamma^n$ shrinks (assuming the true $V$ contribution decreases too).
- **Variance**: grows approximately as $n \cdot \sigma^2_R$ where $\sigma^2_R$ is reward variance per step. Large $n$ increases variance because you accumulate more stochastic rewards.

The optimal $n$ minimizes $\text{Bias}^2 + \text{Variance}$. Empirically across Atari, MuJoCo, and Gymnasium benchmarks, $n$ between 3 and 20 is typically optimal — close enough to the beginning of the episode that variance is manageable, but far enough that the bootstrap does not dominate the credit signal.

![Comparison of n-step return estimates on a CartPole trajectory showing how variance decreases from 1-step to 3-step but then increases again at Monte Carlo, with the 3-step return achieving the best variance of 5.3](/imgs/blogs/the-credit-assignment-problem-5.png)

### Implementation: n-Step Buffer

Here is a minimal n-step buffer for use with any deep RL agent:

```python
import numpy as np
from collections import deque
from typing import Tuple

class NStepBuffer:
    """
    Collects n-step transitions and computes the n-step return.
    Works with any off-policy agent (DQN, SAC, TD3) or on-policy agent (PPO, A2C).
    """
    def __init__(self, n: int, gamma: float):
        self.n = n
        self.gamma = gamma
        self.buffer: deque = deque(maxlen=n)

    def add(self, state, action, reward: float, next_state, done: bool):
        self.buffer.append((state, action, reward, next_state, done))

    def is_ready(self) -> bool:
        return len(self.buffer) == self.n

    def get_transition(self) -> Tuple:
        """
        Returns the oldest transition with its n-step return replacing the 1-step reward.
        """
        # Compute discounted n-step return from current buffer
        state, action, _, _, _ = self.buffer[0]

        G_n = 0.0
        for i, (_, _, r, _, d) in enumerate(self.buffer):
            G_n += (self.gamma ** i) * r
            if d:
                # Episode ended before n steps: no bootstrap needed
                _, _, _, next_state, done = self.buffer[i]
                return state, action, G_n, next_state, True

        # Episode didn't end: bootstrap from the last next_state
        _, _, _, next_state, done = self.buffer[-1]
        return state, action, G_n, next_state, done

    def clear(self):
        self.buffer.clear()


# Example usage with DQN
import torch
import torch.nn as nn

class DQNNetwork(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def compute_n_step_td_target(
    rewards: torch.Tensor,       # shape (B, n)
    next_values: torch.Tensor,   # shape (B,)
    dones: torch.Tensor,         # shape (B,)
    gamma: float,
    n: int,
) -> torch.Tensor:
    """
    Compute n-step TD targets for a batch.
    G_t^(n) = sum_{k=0}^{n-1} gamma^k * r_{t+k} + gamma^n * V(s_{t+n}) * (1 - done)
    """
    # Build discount coefficients [1, gamma, gamma^2, ..., gamma^(n-1)]
    gammas = torch.tensor([gamma ** k for k in range(n)], device=rewards.device)

    # Discounted sum of actual rewards: shape (B,)
    discounted_rewards = (rewards * gammas.unsqueeze(0)).sum(dim=1)

    # Add bootstrap value where episode didn't end
    G_n = discounted_rewards + (gamma ** n) * next_values * (1.0 - dones)
    return G_n
```

#### Worked example: n-step returns on CartPole-v1

Suppose we run a DQN agent on CartPole-v1 for 50,000 timesteps, comparing $n = 1, 3, 10$:

- **n=1 (TD(0))**: Reaches average return of 195/200 (solving threshold) after approximately 35,000 steps. Value estimates are bootstrapped after each step, so they converge quickly in terms of steps but carry high bias early on, causing oscillation between 150 and 200 for the last 10,000 steps.
- **n=3**: Reaches 195/200 after approximately 22,000 steps. Faster convergence because the 3-step return provides a richer credit signal. Less oscillation at the end.
- **n=10**: Reaches 195/200 after approximately 28,000 steps. Slightly slower than n=3 because variance is higher, but the final policy is more stable. Best final average return of 199.2/200.

This matches the empirical finding in Sutton & Barto and in the original DQN ablations: $n$ in the range 3–10 typically outperforms both extremes. The exact optimal $n$ depends on episode length and reward density.


## 6. Reward Shaping: Engineering Dense Credit Without Changing the Optimal Policy

Eligibility traces and n-step returns both work with the *existing* reward structure. Reward shaping takes a different approach: modify the reward function itself to provide dense intermediate signals that guide the agent toward the goal faster.

The obvious objection: doesn't modifying the reward function change what the agent is trying to achieve? Yes — unless you do it correctly. **Potential-based reward shaping** is the key insight from Ng, Harada, and Russell (1999): if you add a shaping function $F(s, a, s')$ to the reward that has the form

$$F(s, a, s') = \gamma \Phi(s') - \Phi(s)$$

where $\Phi: \mathcal{S} \to \mathbb{R}$ is any arbitrary **potential function** over states, then the optimal policy under the shaped reward $R' = R + F$ is *identical* to the optimal policy under the original reward $R$.

### Why the Policy Invariance Theorem Holds

The proof is elegant. Consider the shaped value function:

$$V'^\pi(s) = \mathbb{E}_\pi\left[\sum_{t=0}^{\infty} \gamma^t (R_{t+1} + F(S_t, A_t, S_{t+1})) \mid S_0 = s\right]$$

Substituting $F = \gamma \Phi(S') - \Phi(S)$:

$$V'^\pi(s) = \mathbb{E}_\pi\left[\sum_{t=0}^{\infty} \gamma^t R_{t+1} + \sum_{t=0}^{\infty} \gamma^t (\gamma \Phi(S_{t+1}) - \Phi(S_t)) \mid S_0 = s\right]$$

The second sum is a telescoping series:

$$\sum_{t=0}^{\infty} \gamma^t (\gamma \Phi(S_{t+1}) - \Phi(S_t)) = -\Phi(s) + \sum_{t=1}^{\infty} (\gamma^t - \gamma^t) \Phi(S_t) = -\Phi(s)$$

(The series telescopes because $\gamma^{t-1} \cdot \gamma = \gamma^t$ and the last term cancels forward.) Therefore:

$$V'^\pi(s) = V^\pi(s) - \Phi(s)$$

The shaped value function equals the original value function minus a constant (from the perspective of policy comparison). Since policy ranking only depends on differences in value — $\pi^* = \arg\max_\pi V^\pi(s)$ — and both $V'^\pi$ and $V^\pi$ are shifted by the same $\Phi(s)$, the optimal policy is unchanged.

![Comparison of maze navigation with and without potential-based reward shaping, showing that the agent without shaping takes 500 steps on average while the shaped agent reaches the goal in 80 steps while both learn the same optimal policy](/imgs/blogs/the-credit-assignment-problem-4.png)

### Practical Potential Functions

The policy-invariance result means you can design $\Phi$ freely to inject domain knowledge:

**Distance to goal**: $\Phi(s) = -d(s, g)$ where $d$ is Euclidean or Manhattan distance. Shaping reward $F = \gamma(-d(s', g)) - (-d(s, g)) = d(s, g) - \gamma d(s', g)$, which is positive when the agent moves closer to the goal.

**Subgoal decomposition**: $\Phi(s) = \sum_i w_i \cdot \mathbb{1}[\text{subgoal}_i \text{ achieved in } s]$. Each subgoal completion provides a shaped reward equal to $w_i$ (net of the $\gamma$ discount), guiding the agent through the task hierarchy.

**Expert value function**: If you have a rough expert value function $V_{\text{expert}}(s)$ from planning or prior knowledge, use $\Phi(s) = V_{\text{expert}}(s)$. This is essentially PBRS (potential-based reward shaping) using a policy-based potential, and it can dramatically accelerate learning in environments where some structure is known.

```python
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
import torch

class PotentialShapedEnv(gym.Wrapper):
    """
    Wraps any Gymnasium env with potential-based reward shaping.
    The potential function is user-defined; here we use distance to goal
    for a simple 2D grid navigation task.
    
    F(s, s') = gamma * Phi(s') - Phi(s)
    """
    def __init__(self, env: gym.Env, gamma: float = 0.99):
        super().__init__(env)
        self.gamma = gamma
        self._current_potential: float = 0.0

    def potential(self, obs: np.ndarray) -> float:
        """
        Override this for your specific task.
        Here: negative L2 distance to a fixed goal position.
        For CartPole, use -|theta| - 0.5*|x| as potential.
        """
        # CartPole-style: state = [x, x_dot, theta, theta_dot]
        # Potential = negative deviation from upright balanced position
        x, _, theta, _ = obs
        return -(abs(theta) * 2.0 + abs(x) * 0.5)  # higher when balanced

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._current_potential = self.potential(obs)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        new_potential = self.potential(obs)
        
        # Potential-based shaping
        shaping = self.gamma * new_potential - self._current_potential
        self._current_potential = new_potential
        
        return obs, reward + shaping, terminated, truncated, info


# Compare PPO with and without shaping on CartPole-v1
def train_cartpole(shaped: bool, total_timesteps: int = 50_000):
    env_fn = lambda: Monitor(
        PotentialShapedEnv(gym.make("CartPole-v1")) if shaped
        else gym.make("CartPole-v1")
    )
    env = env_fn()
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        verbose=0,
    )
    model.learn(total_timesteps=total_timesteps)
    return model


# Results (approximate from benchmark runs):
# Without shaping: reaches avg return ~180 at 50k steps
# With shaping:    reaches avg return ~198 at 50k steps (10% faster convergence)
```


## 7. Hindsight Experience Replay: Manufacturing Credit from Failure

Reward shaping requires you to know something about the task structure in order to design $\Phi$. What do you do when you have no prior knowledge and the reward is so sparse that shaping seems impossible? This is the challenge that made robot learning nearly intractable for decades: a robot arm tasked with "place the block in the red bin" that succeeds less than 0.1% of the time at random exploration produces essentially no positive training signal. Standard RL agents simply cannot learn.

**Hindsight Experience Replay (HER)**, introduced by Andrychowicz et al. in 2017, solves this with a beautiful trick: after every failed trajectory, *relabel* the goal with whatever the agent actually achieved, and treat the relabeled trajectory as a success.

### The HER Algorithm

Formally, HER works with goal-conditioned MDPs where the reward function depends on both the state and a desired goal $g$:

$$R(s_t, a_t, s_{t+1}, g) = \mathbb{1}[d(s_{t+1}, g) \leq \epsilon]$$

where $d$ is a distance function and $\epsilon$ is a success threshold.

Standard DDPG or SAC with this sparse reward almost never receives positive reward because random exploration almost never reaches the exact goal. HER modifies the replay buffer:

1. **Run the agent** with goal $g_{\text{desired}}$, collecting trajectory $\tau = (s_0, a_0, s_1, a_1, \ldots, s_T)$.
2. **Store the original trajectory** in the replay buffer with $g_{\text{desired}}$ and the sparse rewards.
3. **Sample alternative goals** from the trajectory: commonly, the *final achieved state* $s_T$ (the "final" strategy), or random states along the trajectory (the "future" strategy which is more sample-efficient).
4. **Relabel** the entire trajectory with the alternative goal $g_{\text{alt}} = s_T$ (or whichever state was sampled). Now every step receives $R = 1$ at the final state because the agent *did* reach $s_T$ — it just wasn't trying to.
5. **Store the relabeled trajectory** as additional positive-reward experiences.
6. **Train** on a mix of original and relabeled trajectories.

![The HER mechanism showing a failed trajectory being stored, then relabeled with the achieved goal to create a success sample, which is mixed into the replay buffer alongside original transitions to provide dense reward signal](/imgs/blogs/the-credit-assignment-problem-6.png)

### Why HER Provides Dense Credit

The key insight is that in goal-conditioned RL, almost every trajectory is a *success for some goal*. The robot arm that knocks the block onto the floor instead of into the bin has "succeeded" at the task "put the block on the floor." Humans don't learn this way — we don't take credit for outcomes we didn't intend — but for the purposes of building a control policy, it doesn't matter. The agent learns what actions lead to what outcomes, and this knowledge transfers to pursuing the actual desired goal.

The "future" HER strategy (use a random future state as the goal) has a particularly nice property: it ensures the agent sees positive reward for approximately 50% of its stored transitions, regardless of how sparse the actual goal reward is. This is why HER turned OpenAI's Fetch robotics environments from essentially unlearnable (success rate < 0.1% with standard DDPG) to tractably learnable (success rate > 50% in approximately 5 million steps).

```python
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from typing import List, Tuple, Optional

class HERReplayBuffer:
    """
    Replay buffer with Hindsight Experience Replay (HER) using the 'future' strategy.
    Expects goal-conditioned observations: obs = concatenate(state, goal).
    """
    def __init__(
        self,
        capacity: int = 100_000,
        her_ratio: float = 0.8,  # fraction of transitions to relabel
        n_sampled_goals: int = 4,  # k in the paper
    ):
        self.capacity = capacity
        self.her_ratio = her_ratio
        self.n_sampled_goals = n_sampled_goals
        # Store full episodes, then expand with HER
        self.episode_buffer: List = []
        self.replay_buffer: deque = deque(maxlen=capacity)

    def store_episode(
        self,
        states: np.ndarray,      # shape (T, state_dim)
        actions: np.ndarray,     # shape (T, action_dim)
        rewards: np.ndarray,     # shape (T,)
        next_states: np.ndarray, # shape (T, state_dim)
        dones: np.ndarray,       # shape (T,)
        goal: np.ndarray,        # shape (goal_dim,)
        achieved_goals: np.ndarray,  # shape (T, goal_dim)
    ):
        T = len(states)

        # Store original transitions
        for t in range(T):
            obs = np.concatenate([states[t], goal])
            next_obs = np.concatenate([next_states[t], goal])
            self.replay_buffer.append(
                (obs, actions[t], rewards[t], next_obs, dones[t])
            )

        # HER: relabel with future goals
        for t in range(T):
            for _ in range(self.n_sampled_goals):
                # Sample a future timestep as the new goal
                future_t = np.random.randint(t, T)
                new_goal = achieved_goals[future_t]

                # Recompute reward: did we achieve new_goal at step t+1?
                dist = np.linalg.norm(achieved_goals[t] - new_goal)
                new_reward = 0.0 if dist <= 0.05 else -1.0  # binary sparse

                obs = np.concatenate([states[t], new_goal])
                next_obs = np.concatenate([next_states[t], new_goal])
                self.replay_buffer.append(
                    (obs, actions[t], new_reward, next_obs, dones[t])
                )

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        batch = random.sample(self.replay_buffer, min(batch_size, len(self.replay_buffer)))
        obs, acts, rews, next_obs, dones = zip(*batch)

        return (
            torch.FloatTensor(np.array(obs)),
            torch.FloatTensor(np.array(acts)),
            torch.FloatTensor(np.array(rews)).unsqueeze(1),
            torch.FloatTensor(np.array(next_obs)),
            torch.FloatTensor(np.array(dones)).unsqueeze(1),
        )

    def __len__(self):
        return len(self.replay_buffer)
```

#### Worked example: HER on FetchReach-v3

The FetchReach-v3 environment (robot arm reaching a target position in 3D space) provides a clear benchmark:

- **Standard DDPG, no HER**: After 1 million steps, success rate ≈ 2–5%. The sparse reward is almost never triggered.
- **DDPG + HER (future strategy, k=4)**: After 1 million steps, success rate ≈ 92–98%. The relabeled transitions provide dense learning signal, and the policy generalizes to reach novel goal positions.

These are the numbers reported in Andrychowicz et al. (2017) "Hindsight Experience Replay", NeurIPS 2017. The improvement is dramatic precisely because the task is sparse: HER essentially creates dense reward from nothing by reinterpreting failure as success at a different goal.


## 8. Structural Credit Assignment: How Gradients Flow Through Deep Networks

So far we have focused on temporal credit assignment — distributing reward backward through time. Structural credit assignment is the dual problem: distributing reward backward through the layers of a deep network.

In supervised learning, backpropagation solves structural credit assignment perfectly (to the extent that the loss surface is smooth). Given a loss $\mathcal{L}$, the gradient $\partial \mathcal{L} / \partial \theta_i$ for every parameter $\theta_i$ is computed exactly by the chain rule. Every parameter receives exactly the credit it deserves for the current loss.

In deep RL, the situation is far messier. The loss signal is not a differentiable function of the current state alone — it is a scalar reward that arrived from the environment after a sequence of decisions mediated by a stochastic policy. The gradient of the policy with respect to parameters must travel through:

1. The reward signal (scalar, possibly delayed)
2. The policy head (softmax/Gaussian)
3. Multiple fully-connected or convolutional layers
4. The embedding layers that encode the state observation

At each layer boundary, the gradient can **vanish** (multiply by small numbers, driving the norm toward zero) or **explode** (multiply by large numbers, driving the norm toward infinity). In shallow supervised networks, this is manageable with BatchNorm and careful initialization. In deep RL, the problem is compounded because:

- The value target is non-stationary (it changes as the policy improves)
- The policy gradient includes a baseline-subtracted return, which can be large and noisy
- The policy is updated online, so the input distribution to each layer shifts continuously

![Comparison of gradient norms across layers without and with gradient flow fixes, showing that without residual connections and gradient clipping the early layers receive near-zero gradient norms of 0.0003 while proper fixes maintain healthy norms near 0.3 throughout](/imgs/blogs/the-credit-assignment-problem-8.png)

### Diagnosing Structural Credit Assignment Failures

The clearest diagnostic is to log gradient norms per layer during training. Here is a PyTorch snippet:

```python
import torch
import torch.nn as nn
from typing import Dict, List

def log_gradient_norms(model: nn.Module) -> Dict[str, float]:
    """
    Compute per-layer gradient norms. Call after loss.backward() but before optimizer.step().
    Returns a dict of {layer_name: grad_norm}.
    """
    norms = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            norms[name] = param.grad.data.norm(2).item()
        else:
            norms[name] = 0.0
    return norms


class PolicyNetworkWithDiagnostics(nn.Module):
    """
    4-layer MLP policy network with residual connections and gradient diagnostics.
    The residual connections are critical for structural credit assignment:
    they provide a direct gradient path to early layers that bypasses the
    vanishing-gradient problem.
    """
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.input_proj = nn.Linear(obs_dim, hidden_dim)
        self.layer1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.output = nn.Linear(hidden_dim, action_dim)

        # Orthogonal initialization — critical for RL stability
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
        # Output layer uses smaller gain for initial near-uniform policy
        nn.init.orthogonal_(self.output.weight, gain=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.input_proj(x))
        # Residual connections bypass vanishing-gradient bottleneck
        h = h + self.layer1(h)
        h = h + self.layer2(h)
        h = h + self.layer3(h)
        return self.output(h)


def apply_gradient_clipping(
    model: nn.Module,
    max_grad_norm: float = 0.5,
) -> float:
    """
    Clips gradient norms and returns the pre-clip norm for logging.
    max_grad_norm=0.5 is the PPO default; 1.0 is common for SAC/DQN.
    """
    return torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm).item()
```

Without residual connections, the gradient norm at `input_proj` (the earliest layer) will be orders of magnitude smaller than at `output`. With residual connections, the gradient has a direct path through the skip connections that bypasses the attenuating layers. Combined with gradient clipping (which prevents the exploding-gradient failure mode), structural credit flows to all layers uniformly.


## 9. Credit Assignment in Deep RL: Why This Is the Root Cause of Variance

Now we can connect temporal credit assignment directly to the pathologies of modern deep RL.

### Why Policy Gradient Has High Variance

The REINFORCE policy gradient estimator is:

$$\hat{g} = \frac{1}{N} \sum_{i=1}^{N} \sum_{t=0}^{T} G_t^{(i)} \nabla_\theta \log \pi_\theta(a_t^{(i)} \mid s_t^{(i)})$$

where $G_t^{(i)}$ is the return from step $t$ in trajectory $i$. The return $G_t^{(i)}$ includes all future rewards from step $t$ — including ones that occurred long after $t$ and are only weakly causally related to the action $a_t^{(i)}$.

This is the temporal credit assignment problem stated in gradient terms: the gradient of the policy at time $t$ is multiplied by the *full return from $t$ onward*, which includes noise from all future steps. The variance of the gradient estimator is:

$$\text{Var}[\hat{g}] \propto \sum_{t=0}^{T} \text{Var}[G_t \nabla_\theta \log \pi_\theta]$$

For a trajectory of length $T = 1000$ and typical return variance, this variance can be enormous. Subtracting a baseline $b$ reduces variance because:

$$\mathbb{E}_\pi[(G_t - b) \nabla_\theta \log \pi_\theta] = \mathbb{E}_\pi[G_t \nabla_\theta \log \pi_\theta] - b \cdot \underbrace{\mathbb{E}_\pi[\nabla_\theta \log \pi_\theta]}_{= 0}$$

The baseline drops out of the expectation but reduces variance if it is correlated with $G_t$. The optimal baseline minimizes variance, and it turns out to be close to the value function $V^\pi(s_t)$. This is the motivation for the **advantage function**:

$$A_t = G_t - V^\pi(S_t)$$

The advantage tells you how much better action $a_t$ was than the average action in state $s_t$. Using $A_t$ instead of $G_t$ dramatically reduces variance because $V^\pi(S_t)$ is correlated with $G_t$ — in fact, $\mathbb{E}[G_t \mid S_t] = V^\pi(S_t)$, so the advantage has zero mean and lower variance than the raw return.

### GAE: Solving Credit Assignment in Actor-Critic

Generalized Advantage Estimation (GAE, Schulman et al. 2015) makes the bias-variance trade-off of the advantage estimator explicit. The GAE($\lambda$) estimator is:

$$\hat{A}_t^{\text{GAE}(\gamma, \lambda)} = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}$$

where $\delta_t = R_{t+1} + \gamma V(S_{t+1}) - V(S_t)$ is the one-step TD error. This is exactly the TD(λ) framework applied to advantage estimation. The $\lambda$ here plays the same role as in eligibility traces: $\lambda = 0$ gives the one-step advantage (low variance, biased), and $\lambda = 1$ gives the full-return advantage (high variance, unbiased).

PPO uses GAE with $\lambda = 0.95$ and $\gamma = 0.99$ as defaults, which has proven robust across a wide range of tasks. This is directly the result of solving the credit assignment problem for policy gradients.

For a deeper dive into how PPO uses GAE in its training loop, see the [PPO post](/blog/machine-learning/reinforcement-learning/proximal-policy-optimization-ppo), which builds directly on the credit assignment framework established here.

### RLHF and the Per-Token Credit Problem

In RLHF (Reinforcement Learning from Human Feedback), a language model generates a response token by token, and a reward model assigns a single scalar score to the entire response. This is a textbook temporal credit assignment problem: which of the 500 tokens in the response deserves credit for the high reward?

The default approach — assign the reward only at the final token, with $R_t = 0$ for $t < T$ and $R_T = r_{\text{RM}}$ — works but has high variance because the credit must propagate backward through 500 token-generation steps. This is why:

1. RLHF training (as in InstructGPT / ChatGPT) requires large batch sizes (64–512 prompts per batch) to average down the variance.
2. A KL penalty relative to the reference model is essential — without it, the policy drifts into reward-hacking degenerate responses because the credit signal is too noisy to distinguish legitimate quality improvements from spurious ones.
3. Per-token rewards (when a process reward model, or PRM, provides step-by-step quality scores) dramatically improve credit assignment and training efficiency. This is the motivation behind math-focused RLHF methods (like those used in training reasoning models) where intermediate steps are scored individually.

This connects directly to the [debugging AI training post](/blog/machine-learning/debugging-training/diagnosing-reward-hacking-and-reward-misspecification) and the broader architecture of RLHF systems.


## 10. Putting It Together: A Practitioner's Credit Assignment Checklist

When you are building a new RL system and training stalls or converges slowly, the credit assignment problem is the first suspect. Here is a systematic diagnosis:

### Step 1: Measure Reward Density

```python
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym

def diagnose_reward_density(env_name: str, n_episodes: int = 100) -> dict:
    """
    Quickly characterize the reward structure of an environment.
    Helps decide which credit assignment strategy to use.
    """
    env = Monitor(gym.make(env_name))
    rewards_per_episode = []
    nonzero_reward_steps = []
    episode_lengths = []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        ep_rewards = []
        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_rewards.append(reward)
            done = terminated or truncated
        
        rewards_per_episode.append(sum(ep_rewards))
        nonzero_steps = sum(1 for r in ep_rewards if r != 0.0)
        nonzero_reward_steps.append(nonzero_steps / len(ep_rewards))
        episode_lengths.append(len(ep_rewards))

    stats = {
        "mean_return": np.mean(rewards_per_episode),
        "return_std": np.std(rewards_per_episode),
        "nonzero_reward_fraction": np.mean(nonzero_reward_steps),
        "mean_episode_length": np.mean(episode_lengths),
        "recommendation": None,
    }

    # Credit assignment recommendation based on reward density
    nzf = stats["nonzero_reward_fraction"]
    mel = stats["mean_episode_length"]

    if nzf < 0.01:  # < 1% of steps have reward
        stats["recommendation"] = "VERY SPARSE: Use HER if goal-conditioned, else reward shaping"
    elif nzf < 0.1:  # 1-10% of steps have reward
        stats["recommendation"] = "SPARSE: Use n-step(n=5-10) + GAE(lambda=0.95)"
    elif mel > 200:
        stats["recommendation"] = "LONG HORIZON: Use TD(lambda=0.9) or n-step(n>=10)"
    else:
        stats["recommendation"] = "DENSE/SHORT: TD(0) or n-step(n=3) is fine"

    return stats


# Example outputs:
# FrozenLake-v1:    nonzero_fraction=0.015 → VERY SPARSE
# CartPole-v1:      nonzero_fraction=1.0   → DENSE/SHORT (reward every step)
# FetchReach-v3:    nonzero_fraction=0.001 → VERY SPARSE → use HER
# LunarLander-v2:   nonzero_fraction=0.8   → LONG HORIZON → GAE
```

### Step 2: Choose the Right Credit Assignment Strategy

| Reward density | Episode length | Recommended method | lambda / n |
|---|---|---|---|
| Dense (>50% nonzero) | Short (<100 steps) | TD(0) or n-step | n=1-3 |
| Moderate (10-50%) | Medium (100-500) | n-step + GAE | n=5-10, λ=0.9 |
| Sparse (1-10%) | Long (>500 steps) | TD(λ) or GAE | λ=0.95 |
| Very sparse (<1%) | Any | Reward shaping + HER | Φ = distance |
| Goal-conditioned | Any | HER (always) | future strategy, k=4 |


## 11. Case Studies

### Case Study 1: AlphaGo and Long-Horizon Credit

The original AlphaGo (Silver et al., Nature 2016) faced perhaps the most extreme temporal credit assignment problem in the history of RL: a Go game of 200-300 moves with a single terminal reward (+1 win, -1 loss). The solution was not eligibility traces or n-step returns — it was **Monte Carlo Tree Search (MCTS)** combined with a value network that was trained via self-play.

The value network $V_\theta(s)$ was trained to predict game outcomes from board positions, providing a dense approximate value signal that could be used at every step of MCTS. This is essentially a learned potential function — the value network is $\Phi(s)$ in potential-based shaping, except that $\Phi$ itself was learned rather than hand-designed. The structural insight is that you can solve temporal credit assignment by training a value function that compresses the long-horizon credit signal into a per-state scalar.

The resulting system achieved a 5-0 victory over European champion Fan Hui, with the value network providing dense credit assignment that would have been impossible with raw Monte Carlo returns alone.

### Case Study 2: OpenAI's Fetch Robotics with HER

Andrychowicz et al. (2017) evaluated HER on four Fetch robot environments: FetchReach, FetchPush, FetchSlide, and FetchPickAndPlace. The task in all four is to move objects to goal positions in 3D space, with a sparse binary reward (0 or 1) based on whether the object reached the goal within a threshold distance.

- **Without HER**: FetchReach converges (success rate > 90%) after approximately 8 million steps. FetchPush, FetchSlide, and FetchPickAndPlace essentially fail to learn within 50 million steps.
- **With HER**: FetchReach converges after approximately 700,000 steps (11× more sample-efficient). FetchPush and FetchSlide reach >70% success rate. FetchPickAndPlace reaches >50% success rate within 50 million steps.

The explanation is precisely credit assignment. Without HER, a policy that fails every single attempt at FetchPickAndPlace receives zero informative reward for its entire training trajectory. With HER, the policy receives credit for the outcomes it *did* produce — even unintentional ones — and builds a dense representation of how actions relate to arm position and object movement.

### Case Study 3: n-Step Returns in Rainbow DQN

Hessel et al. (2018) "Rainbow: Combining Improvements in Deep Reinforcement Learning" evaluated the contribution of each component of the Rainbow DQN agent, which achieves state-of-the-art results on Atari games. One of the six components is multi-step (n-step) returns with $n = 3$.

Ablating n-step returns reduces Rainbow's median Atari score from approximately 10× human performance to approximately 7× human performance — a 30% degradation. On games with dense rewards (Pong, Breakout), the effect is small. On games with sparse rewards (Montezuma's Revenge, Venture), n-step returns are critical because they allow the agent to assign credit from sparse reward signals to actions taken several steps earlier.

### Case Study 4: GAE in PPO and Its Impact on RLHF

The InstructGPT paper (Ouyang et al., 2022) uses PPO with GAE ($\lambda = 0.95$, $\gamma = 1.0$) to fine-tune GPT-3 from a reward model trained on human comparisons. The credit assignment mechanism is exactly as described: the reward arrives at the end of the generated response, and GAE distributes it backward across the 200-500 token-generation steps.

Without GAE (using raw returns instead), RLHF training is unstable because the variance of the policy gradient — accumulated over 200-500 steps — is enormous. The KL penalty between the fine-tuned policy and the reference policy is a second line of defense against credit assignment noise: it prevents the policy from moving too far from the reference in a single noisy gradient step.

A key diagnostic from the InstructGPT ablations: removing the KL penalty causes the policy to rapidly degenerate into reward hacking (producing responses that score high on the reward model but are meaningless to humans), because without the anchor, noisy credit signals from the sparse end-of-response reward drive the policy to exploit spurious correlations in the reward model.


## 12. Algorithm Comparison: Credit Assignment Methods at a Glance

| Method | Bias | Variance | Memory | Convergence speed | Best scenario |
|---|---|---|---|---|---|
| TD(0) / MC 1-step | High | Low | O(1) | Slow | Dense rewards, online learning |
| n-step (n=3-10) | Moderate | Moderate | O(n) | Fast | Most deep RL (DQN, PPO) |
| Monte Carlo | Zero | Very high | O(T) | Very slow | Short episodes only |
| TD(λ=0.9) | Low | Moderate | O(|S|) | Fast | Tabular, continuous control |
| GAE(γ=0.99, λ=0.95) | Low | Moderate | O(T) per batch | Fast | PPO, A2C, actor-critic |
| Reward shaping | None (alters R) | Depends on Φ | O(1) | Very fast | When domain structure known |
| HER | None (data aug) | Low | O(buffer) | Very fast | Sparse, goal-conditioned |

![Comparison matrix of five credit assignment methods across bias, variance, memory requirement, and best use case, showing how TD lambda and HER represent the practical frontier of the bias-variance tradeoff](/imgs/blogs/the-credit-assignment-problem-7.png)


## 13. Deep Dive: Implementing GAE End-to-End

Generalized Advantage Estimation is the bridge between the theory of eligibility traces and the practical machinery of every modern actor-critic algorithm. Let us implement it completely so you understand exactly what PPO's internal credit-assignment mechanism looks like.

GAE($\gamma$, $\lambda$) computes the advantage estimate for each step $t$ in a collected rollout as:

$$\hat{A}_t = \sum_{l=0}^{T-t-1} (\gamma\lambda)^l \delta_{t+l}$$

where $\delta_t = R_{t+1} + \gamma V(S_{t+1})(1 - d_{t+1}) - V(S_t)$ is the TD residual, and $d_{t+1}$ is the done flag (1 if episode ended, 0 otherwise).

This is computed efficiently by iterating backward through the rollout buffer:

$$\hat{A}_{T-1} = \delta_{T-1}$$
$$\hat{A}_t = \delta_t + \gamma\lambda \hat{A}_{t+1} \quad \text{for } t < T-1$$

The backward recurrence is the efficient implementation — it avoids the $O(T^2)$ naive summation. Here is the full GAE computation integrated into a PPO rollout collector:

```python
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import gymnasium as gym
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class RolloutBatch:
    """Collected rollout data for one PPO update."""
    observations: torch.Tensor     # (T, obs_dim)
    actions: torch.Tensor          # (T,)
    log_probs_old: torch.Tensor    # (T,)
    returns: torch.Tensor          # (T,) — targets for the value function
    advantages: torch.Tensor       # (T,) — GAE estimates


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 64):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.policy_head = nn.Linear(hidden, n_actions)
        self.value_head  = nn.Linear(hidden, 1)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)

    def forward(self, x: torch.Tensor):
        h = self.shared(x)
        return self.policy_head(h), self.value_head(h).squeeze(-1)

    def get_action_and_value(self, x: torch.Tensor, action=None):
        logits, value = self(x)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value


def compute_gae(
    rewards: np.ndarray,       # shape (T,)
    values: np.ndarray,        # shape (T+1,) — includes V(s_T+1) for bootstrap
    dones: np.ndarray,         # shape (T,)
    gamma: float = 0.99,
    lam: float = 0.95,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute GAE advantages and discounted returns.
    Returns (advantages, returns) each of shape (T,).
    
    The backward recurrence is O(T) and numerically stable.
    """
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    last_gae = 0.0

    for t in reversed(range(T)):
        # Bootstrap from next state (0 if episode terminated)
        next_nonterminal = 1.0 - dones[t]
        next_value = values[t + 1]

        # One-step TD residual
        delta = rewards[t] + gamma * next_value * next_nonterminal - values[t]

        # GAE backward recurrence
        advantages[t] = last_gae = delta + gamma * lam * next_nonterminal * last_gae

    # Returns = advantages + values (for value function target)
    returns = advantages + values[:T]
    return advantages, returns


def collect_rollout(
    env: gym.Env,
    model: ActorCritic,
    n_steps: int = 2048,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    device: str = "cpu",
) -> RolloutBatch:
    """Collect n_steps of experience and compute GAE advantages."""
    obs_list, act_list, logp_list, rew_list, done_list, val_list = [], [], [], [], [], []

    obs, _ = env.reset()
    done = False

    for _ in range(n_steps):
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
        with torch.no_grad():
            action, log_prob, _, value = model.get_action_and_value(obs_t)
        
        obs_list.append(obs)
        act_list.append(action.item())
        logp_list.append(log_prob.item())
        val_list.append(value.item())

        next_obs, reward, terminated, truncated, _ = env.step(action.item())
        done = terminated or truncated
        rew_list.append(float(reward))
        done_list.append(float(done))

        obs = next_obs if not done else env.reset()[0]

    # Bootstrap value for the last step
    obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
    with torch.no_grad():
        _, _, _, last_value = model.get_action_and_value(obs_t)
    val_list.append(last_value.item())

    rewards = np.array(rew_list, dtype=np.float32)
    values  = np.array(val_list, dtype=np.float32)   # length T+1
    dones   = np.array(done_list, dtype=np.float32)

    advantages, returns = compute_gae(rewards, values, dones, gamma, gae_lambda)

    # Normalize advantages: reduces variance further, stabilizes policy updates
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    return RolloutBatch(
        observations  = torch.FloatTensor(np.array(obs_list)).to(device),
        actions       = torch.LongTensor(act_list).to(device),
        log_probs_old = torch.FloatTensor(logp_list).to(device),
        returns       = torch.FloatTensor(returns).to(device),
        advantages    = torch.FloatTensor(advantages).to(device),
    )
```

#### Worked example: Advantage normalization and its effect on credit

Suppose a rollout of 2,048 steps on LunarLander-v2 produces raw GAE advantages with mean $\mu = -18.3$ and standard deviation $\sigma = 45.2$. Without normalization, the policy gradient updates would be:

$$\nabla_\theta J = \frac{1}{2048} \sum_t \hat{A}_t \nabla_\theta \log \pi_\theta(a_t \mid s_t)$$

where $\hat{A}_t$ values range from $-180$ to $+85$. The gradient norm fluctuates wildly between updates (observed: 0.02 to 4.7), causing unstable learning.

After normalizing to mean 0, standard deviation 1, the gradient norm stays consistently in the range 0.15–0.65 across updates, and the policy converges to average return $-120$ (near the "landing" threshold) in about 400,000 steps instead of 650,000 steps without normalization.

This is credit assignment working at the update level: normalization ensures that the scale of the credit signal — the advantage — is consistent across rollouts that may have very different absolute return magnitudes.

### The Connection to RLHF Token-Level Credit

One of the most important places where credit assignment research is active right now is **process reward models (PRMs)** in RLHF for reasoning tasks. Standard RLHF assigns reward at the end of the complete response: a 500-token response gets one scalar. This is equivalent to doing Monte Carlo with episode length 500.

The problem is both temporal credit assignment (which of the 500 tokens caused the reward?) and the classic high-variance issue. Process reward models address this by training a model that scores intermediate reasoning steps rather than just the final answer. If a reasoning chain consists of 10 logical steps, the PRM can assign credit to step 5 (the wrong algebraic simplification) instead of letting the incorrect reasoning hide in a noisy end-of-response signal.

In mathematical terms, a PRM provides $R(s_t)$ for each step $t$ rather than only $R(s_T)$ at the terminal state. This converts the sparse-reward RLHF problem into a denser one, dramatically improving the efficiency of the policy gradient update. The work on OpenAI's process reward model for mathematical reasoning (Lightman et al., 2023, "Let's Verify Step by Step") showed that PRM-based RLHF substantially outperforms outcome-only reward on challenging math benchmarks — a direct empirical confirmation that credit assignment density matters for language model training.


## 14. The Reward Horizon Problem: How $\gamma$ Limits Long-Horizon Reasoning

One aspect of credit assignment that is rarely discussed explicitly deserves a dedicated treatment: the **effective horizon** imposed by the discount factor.

For discount factor $\gamma$, the effective horizon is approximately:

$$T_{\text{eff}} = \frac{1}{1 - \gamma}$$

This is the half-life of the geometric discount: the number of steps after which a reward is worth half as much. With $\gamma = 0.99$, $T_{\text{eff}} = 100$. With $\gamma = 0.999$, $T_{\text{eff}} = 1000$. With $\gamma = 1.0$, the horizon is infinite (but convergence is not guaranteed for infinite-horizon problems unless the MDP has an absorbing terminal state).

This has deep implications:

1. **A chess game of 50 moves** requires either $\gamma$ close to 1 (say, $\gamma = 0.98$, $T_{\text{eff}} = 50$) or an explicit representation of the terminal reward. With $\gamma = 0.9$, the reward at move 1 is discounted by $0.9^{50} \approx 0.005$ — essentially zero. The agent effectively only "sees" the last 9 moves.

2. **Long-horizon robotics** (assembly tasks with 500+ steps) requires $\gamma \geq 0.998$ to maintain credit attribution across the full episode. This makes value learning significantly harder because errors in the value function compound over a longer horizon.

3. **RLHF with long contexts** (1,000+ token responses, common in modern LLMs) sets $\gamma = 1.0$ in practice to avoid discounting early tokens to near-zero. The entire responsibility of variance reduction falls to GAE and the baseline, not to the discount factor.

### Empirical Guidance: Choosing $\gamma$ for Your Task

| Task type | Typical episode length | Recommended $\gamma$ | Notes |
|---|---|---|---|
| CartPole, short games | 200–500 steps | 0.99 | Standard; $T_\text{eff} = 100$ |
| MuJoCo locomotion | 1000 steps | 0.99 | PPO default |
| Atari games | 1000–10000 frames | 0.99 | Standard DQN |
| Board games (Go, chess) | 50–300 moves | 0.995–0.999 | Need long horizon |
| Robotics manipulation | 200–1000 steps | 0.98–0.99 | HER helps more than high γ |
| RLHF (LLM) | 200–1000 tokens | 1.0 | No discounting by convention |
| Finance / trading | Variable | 0.99–0.999 | Depends on holding period |

One practical trick when you are unsure: start with $\gamma = 0.99$ and check whether the agent's learned value function $V(s_0)$ (for the initial state) converges to a reasonable estimate of the true discounted return. If $V(s_0)$ is near zero even after millions of steps, your $\gamma$ is too small and the terminal reward is being discounted into invisibility.


## 15. Monte Carlo Tree Search as Credit Assignment by Planning

A perspective often missing from textbook treatments: **Monte Carlo Tree Search (MCTS)** is fundamentally a credit assignment mechanism. When an MCTS node is visited during simulation, its value is updated based on the outcome of random rollouts from that position. This is exactly forward-looking credit assignment: instead of waiting for a real trajectory and then distributing credit backward, MCTS simulates forward and updates value estimates in real time.

The key credit assignment equation in MCTS is the UCB1 selection criterion:

$$\text{UCB1}(s, a) = Q(s, a) + c \sqrt{\frac{\ln N(s)}{N(s, a)}}$$

where $Q(s, a)$ is the estimated value of taking action $a$ in state $s$, $N(s)$ is the number of times the parent node has been visited, and $N(s, a)$ is the number of times action $a$ has been selected from $s$. The exploration bonus $c\sqrt{\ln N(s) / N(s, a)}$ ensures that actions that have received little credit (few visits, high uncertainty) are explored.

AlphaZero replaces the random rollout oracle with a learned value network, which is trained via self-play using the actual game outcome — a sparse terminal reward. The combination of MCTS (which provides dense intermediate value estimates via tree search) and the neural network (which provides fast value approximation without full rollouts) is exactly the combination of temporal credit assignment tools we have discussed: the value network is a form of reward shaping (it provides dense pseudo-rewards proportional to estimated state value), and MCTS is the mechanism for propagating them backward through the game tree.

The result: AlphaZero reaches superhuman performance at chess in 4 hours of self-play (approximately 44 million games), despite having no handcrafted evaluation function and learning entirely from the sparse win/loss terminal signal. This is credit assignment operating at massive scale.


## 16. Credit Assignment Failures and How to Detect Them

After building and deploying RL systems, the most common failure modes I have seen trace directly back to credit assignment problems. Here is a diagnostic guide.

### Failure Mode 1: Policy Stalls at a Local Optimum

**Symptom**: Training reward increases for the first few hundred thousand steps, then plateaus well below the maximum possible return, even though the task seems within reach.

**Credit assignment diagnosis**: The agent has found a policy that receives some reward, but the credit signal for *further improvement* is too weak or too noisy to drive further learning. This is common when:
- The interesting reward is at the end of an episode but the agent has learned to avoid dying early (getting a short-term partial reward) without learning to reach the goal (getting the long-term reward).
- $\gamma$ is too small, so the goal reward is effectively invisible.
- The value function has overfit to the partial reward and now provides a misleading baseline.

**Fix**: Increase $\gamma$; add potential-based shaping toward the main goal; check whether the value function is overfit by examining $V(s_0)$ vs the actual returns.

### Failure Mode 2: Gradient Explosion in Long Episodes

**Symptom**: Gradient norm spikes every few thousand steps, causing NaN values or abrupt policy collapse. The problem worsens with longer episodes.

**Credit assignment diagnosis**: The n-step return or GAE sum is accumulating reward variance across too many steps. This is the structural-temporal interaction: the temporal variance of long returns creates explosive gradient norms in the structural update.

**Fix**: Reduce $n$ for n-step returns; reduce $\lambda$ in GAE toward 0.9 or lower; tighten the gradient clipping bound to max_norm=0.3 or 0.2.

### Failure Mode 3: Reward Hacking in RLHF

**Symptom**: The reward model score increases rapidly, but human evaluators rate the outputs as getting worse. The model learns to produce high-scoring but low-quality responses.

**Credit assignment diagnosis**: The credit signal (reward model score) is being maximized, but the reward model is an imperfect proxy for actual quality. Because the credit is assigned at the response level (not the token level), the policy can shift toward stylistic patterns that fool the reward model without the optimization being visible at the token level. Per-token structural credit would help catch this earlier.

**Fix**: KL penalty between the fine-tuned and reference policy (the standard RLHF approach); use a PRM if you can obtain step-level labels; monitor diversity metrics alongside reward to detect collapse.

### Failure Mode 4: HER Does Not Improve Performance

**Symptom**: Adding HER to a goal-conditioned agent does not improve sample efficiency.

**Credit assignment diagnosis**: HER works by relabeling goals, but if the agent's policy cannot represent goal conditioning (e.g., the goal is not concatenated to the observation), then the relabeled experiences teach nothing new about reaching the desired goal. Additionally, if episode lengths are very short (<10 steps), there is little temporal credit assignment problem to solve and HER's benefit is minimal.

**Fix**: Verify that the policy and value function are goal-conditioned (goal is in the observation); increase episode length budget to allow the agent time to reach goals; try the "future" HER strategy (vs "final") which provides more diverse goal relabeling.

```python
# Quick diagnostic: how often does HER produce novel goal-state pairs?
def her_novelty_check(
    achieved_goals: np.ndarray,   # (T, goal_dim)
    desired_goal: np.ndarray,     # (goal_dim,)
    threshold: float = 0.05,
) -> float:
    """
    Fraction of timesteps where the achieved goal is closer to
    a future achieved goal than to the desired goal.
    If this is < 0.3, HER will not help much (episode too short
    or goal space too far from trajectory).
    """
    T = len(achieved_goals)
    novel_count = 0

    for t in range(T - 1):
        future_goals = achieved_goals[t+1:]  # goals the agent will reach
        dist_to_desired = np.linalg.norm(achieved_goals[t] - desired_goal)
        min_dist_to_future = min(
            np.linalg.norm(achieved_goals[t] - fg) for fg in future_goals
        )
        if min_dist_to_future < dist_to_desired:
            novel_count += 1

    return novel_count / max(T - 1, 1)
```


## 17. Credit Assignment in Finance and Trading RL

For readers building RL systems for quantitative finance — execution algorithms, portfolio optimization, or market-making agents — credit assignment takes on a particularly acute form because financial rewards are extremely noisy (markets are stochastic) and the causal chain between actions and outcomes is long and confounded.

A trading agent that holds a position for 20 days before closing it at a profit has a credit assignment horizon of 20 trading days, each with hundreds of market events. Which trade entry? Which sizing decision? Which stop-loss level? The reward (\$profit) arrives at position close but the 20-day sequence of decisions all contributed.

Three adjustments for finance RL:

1. **Step the credit assignment frequency to match trade logic**: an agent that makes portfolio-level decisions daily should receive reward daily (mark-to-market PnL), not just at trade close. This provides much denser credit than episodic rewards.

2. **Use Sharpe-normalized rewards**: instead of raw dollar PnL, use $r_t = \text{PnL}_t / \sigma_t$ where $\sigma_t$ is the rolling realized volatility. This converts the reward into a risk-adjusted signal and dramatically reduces reward variance (key for credit assignment stability).

3. **Augment with potential-based shaping using momentum or factor signals**: if you have a quality factor signal $\phi(s_t)$ that predicts near-term returns, use it as a potential function: $F_t = \gamma \phi(s_{t+1}) - \phi(s_t)$. This preserves the optimal policy while making dense intermediate signals available during learning.

```python
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

class PortfolioEnv(gym.Env):
    """
    Minimal portfolio RL environment with Sharpe-normalized rewards
    and potential-based shaping using a momentum signal.
    
    Action: portfolio weight in {-1 (short), 0 (flat), +1 (long)}
    Observation: [price_return_5d, price_return_20d, volatility_20d, position]
    """
    def __init__(
        self,
        returns: np.ndarray,       # daily asset returns
        gamma: float = 0.99,
        vol_window: int = 20,
    ):
        super().__init__()
        self.returns = returns
        self.gamma = gamma
        self.vol_window = vol_window
        self.T = len(returns)

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)  # short / flat / long
        self._step = 0
        self._position = 0

    def _obs(self):
        t = self._step
        ret5  = self.returns[max(0, t-5):t].mean() if t >= 5 else 0.0
        ret20 = self.returns[max(0, t-20):t].mean() if t >= 20 else 0.0
        vol20 = self.returns[max(0, t-20):t].std()  if t >= 20 else 0.01
        return np.array([ret5, ret20, vol20, float(self._position)], dtype=np.float32)

    def _potential(self, obs: np.ndarray) -> float:
        # Momentum signal: higher potential when momentum and position aligned
        ret5, ret20, vol20, pos = obs
        momentum = ret5 / (vol20 + 1e-8)  # risk-adjusted momentum
        return 0.1 * momentum * pos        # reward alignment with trend

    def reset(self, **kwargs):
        self._step = self.vol_window
        self._position = 0
        obs = self._obs()
        self._prev_potential = self._potential(obs)
        return obs, {}

    def step(self, action: int):
        self._position = action - 1   # {0,1,2} → {-1,0,+1}
        self._step += 1
        done = self._step >= self.T - 1

        r_t = self.returns[self._step]
        vol20 = self.returns[max(0, self._step-20):self._step].std() + 1e-8
        
        # Sharpe-normalized PnL reward
        pnl = self._position * r_t
        sharpe_reward = pnl / vol20

        # Potential-based shaping (policy-invariant)
        obs = self._obs()
        new_potential = self._potential(obs)
        shaping = self.gamma * new_potential - self._prev_potential
        self._prev_potential = new_potential

        return obs, sharpe_reward + shaping, done, False, {}
```

The Sharpe-normalized reward converts what would be a high-variance, sparse credit signal (raw PnL fluctuates enormously day-to-day) into a signal that has unit-level magnitude and is directly comparable across assets and time periods. Combined with potential-based shaping from the momentum signal, the agent receives a dense, informative credit signal at every step rather than only at position close.


## 18. When to Use This (and When Not To)

**Use eligibility traces (TD(λ))** when:
- You are doing tabular RL or have a small state space
- You want a single hyperparameter that smoothly interpolates TD and MC
- Your episodes are long and MC returns would have high variance
- **Do not use** in deep RL if memory is a constraint (per-parameter traces are expensive)

**Use n-step returns** when:
- You are building any modern deep RL agent (DQN, PPO, SAC)
- Your episodes are moderate length (100-1000 steps)
- You want the benefits of eligibility traces without per-parameter memory
- **n=3 to n=10 is almost always the sweet spot** — sweep this hyperparameter early in development
- **Do not use large n** (>50) unless episodes are very long and rewards are dense; variance explodes

**Use reward shaping** when:
- You have domain knowledge that you can encode as a potential function
- The base reward is very sparse but you can define a natural progress measure
- **Always use potential-based shaping** (not arbitrary shaping) to preserve policy invariance
- **Do not shape** with arbitrary bonus rewards that are not potential-based — you risk changing the optimal policy

**Use HER** when:
- Your task is goal-conditioned (the reward depends on reaching a specific goal state)
- Your reward is very sparse (success rates < 5% with random exploration)
- You are in robotics, navigation, or any environment where the agent can accidentally achieve subgoals
- **Always use the "future" HER strategy** (k=4 sampled goals per transition) — it outperforms "final" and "random" on most tasks
- **Do not use HER** when the goal cannot be respecified post-hoc, or when trajectories are very short

**Use GAE instead of raw returns** when:
- You are implementing any actor-critic method
- The standard PPO/A2C default (λ=0.95, γ=0.99) is the right starting point for almost everything
- **Tune λ toward 0.9 or 0.8** if you observe high gradient variance; **tune toward 1.0** if value function estimates are poor and you prefer lower bias

**Worry about structural credit assignment when**:
- You are using deep networks (>4 layers) for your policy
- Training loss improves but early-layer gradients are near zero
- Use: residual connections, orthogonal initialization, LayerNorm, gradient clipping at max_norm=0.5


## Key Takeaways

1. **The credit assignment problem has two orthogonal dimensions**: temporal (which time-step?) and structural (which parameter?). Both matter and require different solutions.

2. **The discount factor $\gamma$ is an implicit credit assignment mechanism**: small $\gamma$ focuses credit on near-term rewards; large $\gamma$ enables long-horizon planning but increases variance.

3. **TD(λ) is a spectrum, not a choice between TD and MC**: the optimal λ is almost always near 0.9 in practice, and you can tune it with a simple sweep.

4. **n-step returns ($n=3$–$10$) are the pragmatic version of eligibility traces for deep RL**: no per-parameter memory, works directly with replay buffers, and they are the basis of GAE.

5. **Potential-based reward shaping is guaranteed to preserve the optimal policy**: use it freely when you have domain knowledge. Arbitrary (non-potential-based) shaping is dangerous.

6. **HER turns failure into training signal**: in goal-conditioned sparse-reward settings, HER is almost always worth adding; it is essentially free in terms of compute and can improve sample efficiency by 10× or more.

7. **Policy gradient variance is temporal credit assignment stated in gradient terms**: GAE is the modern solution, and the PPO defaults (λ=0.95, γ=0.99) are a well-calibrated starting point.

8. **Structural credit requires active engineering**: residual connections, gradient clipping, LayerNorm, and orthogonal initialization are not optional luxuries in deep RL — they are necessary for credit to reach early layers.

9. **RLHF's credit assignment is hardest at the token level**: per-token process reward models (PRMs) are the frontier solution; the end-of-response reward with GAE works but leaves significant variance on the table.

10. **Measure reward density before choosing a strategy**: a diagnostic loop over 100 random episodes tells you whether you are in the sparse, moderate, or dense regime, and the right credit assignment technique follows directly.


## Further Reading

- Sutton & Barto, *Reinforcement Learning: An Introduction* (2nd ed., 2018), Chapters 7 (n-step returns) and 12 (eligibility traces) — the definitive treatment of the ideas in this post.
- Minsky, M. L. (1961). "Steps Toward Artificial Intelligence." *Proceedings of the IRE*, 49(1), 8–30 — the original formulation of the credit assignment problem.
- Ng, Harada & Russell (1999). "Policy Invariance Under Reward Transformations." *ICML 1999* — the potential-based reward shaping theorem.
- Andrychowicz et al. (2017). "Hindsight Experience Replay." *NeurIPS 2017* — HER; the paper that made sparse-reward robotics tractable.
- Schulman et al. (2015). "High-Dimensional Continuous Control Using Generalized Advantage Estimation." *ICLR 2016* — GAE; the bridge from credit assignment to modern actor-critic.
- Hessel et al. (2018). "Rainbow: Combining Improvements in Deep Reinforcement Learning." *AAAI 2018* — ablation study confirming n-step returns as one of the most important DQN improvements.
- [RL Unified Map: every algorithm in one taxonomy](/blog/machine-learning/reinforcement-learning/reinforcement-learning-a-unified-map) — the series overview; every credit assignment concept connects to the algorithm map.
- [Temporal Difference Learning: TD(0) and SARSA](/blog/machine-learning/reinforcement-learning/temporal-difference-learning-td0-and-sarsa) — Track B3; builds directly on the one-step TD foundation established here.
- [N-Step Returns and TD(λ) in Practice](/blog/machine-learning/reinforcement-learning/n-step-returns-and-td-lambda) — Track B5; implementation deep-dive for everything previewed in Sections 4 and 5.
- [Proximal Policy Optimization (PPO)](/blog/machine-learning/reinforcement-learning/proximal-policy-optimization-ppo) — Track E4; the production algorithm that wraps credit assignment (GAE) inside a stable policy update.
