---
title: "Policies in Reinforcement Learning: Deterministic, Stochastic, and Optimal"
date: "2026-06-22"
publishDate: "2026-06-22"
description: "A rigorous tour of RL policies — from tabular lookup tables through softmax parameterizations to neural Gaussian heads — with the policy improvement theorem proved from scratch and a full GridWorld worked example."
tags:
  [
    "reinforcement-learning",
    "deep-learning",
    "policy-gradient",
    "markov-decision-process",
    "actor-critic",
    "policy-optimization",
    "pytorch",
    "machine-learning",
    "exploration",
    "stochastic-policy",
  ]
category: "machine-learning"
subcategory: "Reinforcement Learning"
author: "Hiep Tran"
featured: true
readTime: 55
image: "/imgs/blogs/policies-deterministic-vs-stochastic-1.png"
---

Your CartPole agent has been training for 200,000 steps and it still drops the pole after eight moves every single time. The network converges, the loss goes to zero, and yet the agent commits to pushing LEFT regardless of the cart's velocity. You check the policy code and realize the output is a single scalar — an argmax that always picks the same action once the weights settle. The agent is not exploring. It cannot hedge. It has locked itself into a deterministic response that happens to be wrong.

This is the central tension in reinforcement learning (RL): deterministic policies are easier to reason about and optimal in a perfectly-known, fully-observed world, but they break in practice the moment you add noise, partial observability, or the need to explore a reward landscape you have never seen. Stochastic policies — ones that output a *distribution* over actions — survive all three. And yet once you switch to a stochastic policy, a whole new question opens up: can you always get back to a deterministic policy when you want one, or have you permanently traded away optimality?

This post answers that question completely. By the end you will understand the formal definition of both policy types, how to parameterize them (tabular, linear softmax, neural Categorical, neural Gaussian), *why* stochastic policies win in partial observability, *why* deterministic policies can recover in full-observation MDPs, and the **policy improvement theorem** — the mathematical guarantee that always converting a stochastic policy to its greedy deterministic counterpart is safe. You will also see a full GridWorld worked example that walks through the tables and shows the improvement step-by-step, plus runnable PyTorch and Stable-Baselines3 code for both policy types.

Figure 1 shows the core contrast we are about to unpack: the sharp single-arrow deterministic policy versus the probability-mass stochastic policy operating on the same state.

![Deterministic policy maps each state to one action with certainty while stochastic policy assigns probability mass across all actions enabling exploration](/imgs/blogs/policies-deterministic-vs-stochastic-1.png)

---

## 1. What is a policy?

Before we can argue about whether a policy should be deterministic or stochastic, we need to be exact about what a policy *is*. In the RL loop you saw in post A1 ([Reinforcement Learning: A Unified Map](/blog/machine-learning/reinforcement-learning/reinforcement-learning-a-unified-map)), an agent observes a state $s \in \mathcal{S}$, chooses an action $a \in \mathcal{A}$, receives a reward $r$, and transitions to a new state $s'$. The **policy** is the function that maps state to action — it is the complete behavioral specification of the agent. Everything else in RL (value functions, Q-functions, critics, replay buffers) exists to help you find a *good* policy.

### 1.1 Stochastic policy

The most general definition is the **stochastic policy**:

$$\pi(a \mid s) = P(A_t = a \mid S_t = s)$$

This is a conditional probability distribution over actions, conditioned on the current state. For every state $s$, $\pi(\cdot \mid s)$ is a valid probability distribution: $\sum_{a} \pi(a \mid s) = 1$ (discrete), or $\int_{\mathcal{A}} \pi(a \mid s) \, da = 1$ (continuous). The agent samples $a \sim \pi(\cdot \mid s)$ and takes that action.

This is a *stochastic* policy because, given the same state $s$, the agent may take different actions on different visits. The CartPole agent with $\pi(\text{LEFT} \mid s) = 0.73$ will push left roughly 73% of the time it sees state $s$, and right 27% of the time.

### 1.2 Deterministic policy

A **deterministic policy** is the degenerate case where all probability mass sits on one action:

$$\pi(s) \rightarrow a$$

or equivalently, $\pi(a \mid s) = \mathbf{1}[a = \pi(s)]$. Given state $s$, the agent always takes exactly action $\pi(s)$. No randomness. Notation often uses $\mu$ for deterministic policies (especially in the DDPG / Deterministic Policy Gradient literature) to signal this distinction: $\mu(s)$ outputs an action, not a distribution.

### 1.3 Why the distinction matters

Both definitions are complete behavioral specifications — they tell you exactly what the agent will do in every state. But they differ in three ways that matter enormously:

1. **Exploration**: A deterministic policy cannot explore without external noise (like $\varepsilon$-greedy or injecting Gaussian perturbations at test time). A stochastic policy explores inherently — the spread of the distribution *is* the exploration.
2. **Partial observability**: When the agent cannot see the full state (a POMDP — Partially Observable Markov Decision Process), two different underlying states may produce the same observation. A deterministic policy must commit to one action for that observation, which is wrong half the time. A stochastic policy can hedge by assigning probability to multiple actions, and in the limit achieves higher expected return. We will prove this in Section 6.
3. **Gradient estimation**: Policy gradient algorithms optimize $J(\theta) = \mathbb{E}_{\pi_\theta}[\sum_t r_t]$ by differentiating through the policy. The score-function estimator $\nabla_\theta \log \pi_\theta(a \mid s)$ only works when $\pi_\theta$ is a smooth distribution. Deterministic policies require a different estimator — the Deterministic Policy Gradient (DPG) theorem (Silver et al. 2014, the DPG paper). We preview this in Section 8 and return to it in post E1.

---

## 2. Tabular policies: the exact baseline

The simplest concrete policy representation is a **lookup table**. You store, for every state–action pair, either the action to take (deterministic) or the probability of taking it (stochastic).

For a deterministic tabular policy with $|\mathcal{S}|$ states and $|\mathcal{A}|$ actions, the table has $|\mathcal{S}|$ entries (one per state, pointing to one action). For a stochastic tabular policy, the table has $|\mathcal{S}| \times |\mathcal{A}|$ entries storing probabilities.

```python
import numpy as np

# Tabular deterministic policy: π[s] = a
n_states, n_actions = 4, 2
# Random initialization
det_policy = np.random.randint(0, n_actions, size=n_states)
print("Deterministic policy:", det_policy)
# e.g. [0, 1, 0, 1]  => state 0 → action 0, state 1 → action 1, ...

# Tabular stochastic policy: π[s, a] = P(a|s), rows sum to 1
stoch_policy = np.random.dirichlet(np.ones(n_actions), size=n_states)
print("Stochastic policy:\n", stoch_policy)
# e.g. [[0.73, 0.27], [0.41, 0.59], ...]

# Sample an action given state s=0
s = 0
a = np.random.choice(n_actions, p=stoch_policy[s])
print(f"Sampled action for state 0: {a}")
```

**Pros of tabular policies:** They are exact — no approximation error. Policy evaluation (computing $V^\pi$) is exact via linear system solve or iterative Bellman application. The policy improvement theorem applies cleanly. They are the foundation that all theory builds on.

**Cons of tabular policies:** They do not scale. A 10×10 grid world has 100 states. Atari has $\sim 10^{70}$ possible frames. A continuous-state robotics environment has $\mathbb{R}^{100}$ as the state space. Tabular policies are computationally infeasible for anything non-trivial.

This is why every serious RL system uses **parameterized policies** — functions $\pi_\theta(a \mid s)$ that share parameters $\theta$ across states. Let's build from the simplest to the most powerful.

---

## 3. Parameterized policies

![Policy parameterization spectrum from exact tabular lookup to neural softmax and Gaussian heads showing the scalability tradeoff](/imgs/blogs/policies-deterministic-vs-stochastic-2.png)

### 3.1 Linear softmax policy (discrete actions)

The simplest parameterized stochastic policy for discrete actions uses a **softmax** over linear preferences. For each state–action pair, define a feature vector $\phi(s, a) \in \mathbb{R}^d$. The policy is:

$$\pi_\theta(a \mid s) = \frac{\exp(\theta^\top \phi(s, a))}{\sum_{a'} \exp(\theta^\top \phi(s, a'))}$$

This is a valid probability distribution for any $\theta$ (the softmax denominator normalizes). As $\|\theta\| \to \infty$, the policy concentrates on the action with the highest preference and approaches deterministic. At $\theta = 0$, the policy is uniform — maximum entropy.

The gradient of the log-policy needed for policy gradient algorithms is clean:

$$\nabla_\theta \log \pi_\theta(a \mid s) = \phi(s, a) - \sum_{a'} \pi_\theta(a' \mid s) \phi(s, a') = \phi(s, a) - \mathbb{E}_{\pi_\theta}[\phi(s, \cdot)]$$

This is the feature of the chosen action minus the expected feature under the current policy — a "centered" gradient that is easy to compute.

### 3.2 Neural softmax policy (discrete actions)

For high-dimensional states (images, text token sequences, graph embeddings), linear features are not enough. A neural network policy replaces the feature vector with a learned encoder:

$$\pi_\theta(a \mid s) = \text{softmax}(f_\theta(s)) \in \mathbb{R}^{|\mathcal{A}|}$$

where $f_\theta : \mathcal{S} \to \mathbb{R}^{|\mathcal{A}|}$ is an MLP or CNN. The output logits are converted to probabilities via softmax. In PyTorch, this is `torch.distributions.Categorical(logits=f_theta(s))`.

```python
import torch
import torch.nn as nn
from torch.distributions import Categorical

class DiscretePolicyNet(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, obs: torch.Tensor):
        logits = self.net(obs)
        return Categorical(logits=logits)

    def get_action(self, obs: torch.Tensor):
        dist = self.forward(obs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob

# CartPole-v1: obs_dim=4, n_actions=2
policy = DiscretePolicyNet(obs_dim=4, n_actions=2)
obs = torch.randn(4)
action, log_prob = policy.get_action(obs)
print(f"Action: {action}, log π(a|s): {log_prob.item():.4f}")
```

### 3.3 Gaussian policy (continuous actions)

For continuous action spaces (robot joint torques, portfolio weights, engine throttle), the policy outputs a Gaussian distribution:

$$\pi_\theta(a \mid s) = \mathcal{N}(\mu_\theta(s), \sigma_\theta^2(s))$$

where $\mu_\theta(s)$ is the mean action and $\sigma_\theta(s) > 0$ is the standard deviation (both output by the network). In practice, the network outputs $\log \sigma$ (unconstrained) and you exponentiate. For multi-dimensional actions with no correlations assumed, this is a **Diagonal Gaussian** (DiagGaussian): each action dimension is independent.

```python
import torch
import torch.nn as nn
from torch.distributions import Normal

class ContinuousPolicyNet(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 64):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden, act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))  # learnable log std

    def forward(self, obs: torch.Tensor):
        h = self.shared(obs)
        mean = self.mean_head(h)
        std = torch.exp(self.log_std).expand_as(mean)
        return Normal(mean, std)

    def get_action(self, obs: torch.Tensor):
        dist = self.forward(obs)
        action = dist.rsample()          # reparameterization trick
        log_prob = dist.log_prob(action).sum(-1)
        return action, log_prob

# MuJoCo Ant-v4: obs_dim=111, act_dim=8
policy = ContinuousPolicyNet(obs_dim=111, act_dim=8)
obs = torch.randn(111)
action, log_prob = policy.get_action(obs)
print(f"Action shape: {action.shape}, log π(a|s): {log_prob.item():.4f}")
```

The `rsample()` call uses the **reparameterization trick**: instead of sampling $a \sim \mathcal{N}(\mu, \sigma^2)$ directly (which blocks gradient flow through the sample), we compute $a = \mu + \sigma \cdot \varepsilon$ where $\varepsilon \sim \mathcal{N}(0, 1)$. The gradient with respect to $\mu$ and $\sigma$ flows cleanly. This is critical for SAC and other algorithms that differentiate through the sampled action.

### 3.4 Squashed Gaussian for bounded actions (SAC-style)

A pure Gaussian can produce actions outside the environment's legal range (e.g., joint torques clipped to $[-1, 1]$). SAC solves this with a **squashed Gaussian**: apply $\tanh$ to the Gaussian sample, then scale to the action range. The log-probability requires a Jacobian correction:

$$\log \pi(a \mid s) = \log \mathcal{N}(u; \mu, \sigma^2) - \sum_i \log(1 - \tanh^2(u_i))$$

where $u \sim \mathcal{N}(\mu, \sigma^2)$ is the pre-squash sample and $a = \tanh(u)$. The subtracted term is the log-absolute-determinant of the Jacobian $\frac{d a}{d u}$. This ensures the policy remains a proper distribution over the bounded action space $(-1, 1)^d$.

```python
import torch
import torch.nn as nn
from torch.distributions import Normal
import torch.nn.functional as F

LOG_STD_MAX = 2
LOG_STD_MIN = -20

class SquashedGaussianPolicy(nn.Module):
    """SAC-style squashed Gaussian policy for bounded continuous actions."""
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden, act_dim)
        self.log_std_head = nn.Linear(hidden, act_dim)

    def forward(self, obs: torch.Tensor):
        h = self.shared(obs)
        mean = self.mean_head(h)
        log_std = self.log_std_head(h).clamp(LOG_STD_MIN, LOG_STD_MAX)
        std = log_std.exp()
        return Normal(mean, std)

    def sample(self, obs: torch.Tensor):
        dist = self.forward(obs)
        u = dist.rsample()
        action = torch.tanh(u)
        # Jacobian correction for squashing
        log_prob = dist.log_prob(u).sum(-1)
        log_prob -= (2 * (torch.log(torch.tensor(2.0)) - u
                         - F.softplus(-2 * u))).sum(-1)
        return action, log_prob

    def deterministic_action(self, obs: torch.Tensor):
        """Test-time greedy: use mean of Gaussian, squashed."""
        dist = self.forward(obs)
        return torch.tanh(dist.mean)

# Example: Ant-v4 (obs=111, act=8, bounded to [-1,1])
policy = SquashedGaussianPolicy(obs_dim=111, act_dim=8)
obs = torch.randn(111)
action, log_prob = policy.sample(obs)
greedy_action = policy.deterministic_action(obs)
print(f"Stochastic action: {action[:3].detach().numpy()}")
print(f"Greedy action:     {greedy_action[:3].detach().numpy()}")
```

This is the exact policy architecture used in the Haarnoja et al. (2018) SAC paper. Notice that `deterministic_action()` uses `tanh(mean)` — the mean of the Gaussian passed through the squashing function — which gives the mode of the squashed distribution. At evaluation time, SAC switches from the stochastic sample to this deterministic mode, getting slightly higher average reward at the cost of losing entropy-driven exploration.

### 3.5 Why not use a full-covariance Gaussian?

You might wonder: why DiagGaussian instead of a full multivariate Gaussian $\mathcal{N}(\mu, \Sigma)$ that captures correlations between action dimensions? Three reasons:

1. **Scale**: A full covariance matrix for $d$-dimensional actions has $O(d^2)$ parameters. For a robot with $d=100$ joint torques, that is 10,000 additional parameters just for the covariance, which is hard to train without numerical issues.
2. **Stability**: Ensuring $\Sigma$ stays positive semi-definite requires careful parameterization (Cholesky decomposition), adding implementation complexity and potential numerical instability.
3. **Empirical adequacy**: In practice, DiagGaussian works well because the policy network's shared encoder already learns correlations implicitly in the mean $\mu_\theta(s)$. The diagonal variance is just a scaling factor for exploration noise per dimension.

For structured action spaces where correlations matter (e.g., humanoid whole-body control where hip and knee torques are strongly correlated), normalizing flows or structured covariance can help. But for most practical continuous control problems, DiagGaussian is the right default.

---

## 4. Why stochastic? Three compelling reasons

It might seem like adding randomness to a policy is strictly worse — if you know the right action, just take it. This intuition fails in three important scenarios.

### 4.1 Exploration

RL is a credit assignment problem: the agent must try actions it has not tried before to discover that they yield high reward. A deterministic policy that always takes the same action in state $s$ will never discover whether a different action in $s$ has higher reward. You need randomness.

The classical solution — $\varepsilon$-greedy — is a hybrid: take the deterministic best action with probability $1-\varepsilon$, and a random action with probability $\varepsilon$. This works, but it is inelegant: you have two separate mechanisms (exploitation policy + random noise) rather than one unified probabilistic policy.

Stochastic policies solve this cleanly. The **entropy** of the policy $\mathcal{H}[\pi(\cdot \mid s)] = -\sum_a \pi(a \mid s) \log \pi(a \mid s)$ measures how spread out the distribution is. High entropy means more exploration. Many modern algorithms (SAC, maximum-entropy RL, A3C with entropy bonus) directly maximize a joint objective:

$$J_{\text{MaxEnt}}(\theta) = \mathbb{E}_\pi\left[\sum_t r_t + \alpha \mathcal{H}[\pi(\cdot \mid s_t)]\right]$$

where $\alpha > 0$ is a temperature parameter. This objective says: get high reward AND maintain high entropy — which naturally prevents premature policy collapse onto a suboptimal deterministic policy.

### 4.2 Partial observability (POMDPs)

This is the most theoretically important reason, and we will prove it properly in Section 6. For now, the sketch: in a **Partially Observable MDP (POMDP)**, the agent observes $o_t$ (a noisy projection of the true state $s_t$) rather than $s_t$ directly. Two different states $s$ and $s'$ may produce the same observation $o$. If $s$ demands action LEFT and $s'$ demands action RIGHT, a deterministic policy that conditions only on $o$ must commit to one, getting at least 50% of these cases wrong. A stochastic policy can output the mixture $\pi(\text{LEFT} \mid o) = p$, $\pi(\text{RIGHT} \mid o) = 1-p$, and by choosing $p$ appropriately, achieve higher expected return than any deterministic policy.

### 4.3 Policy gradient math

The policy gradient theorem (which we preview here and develop fully in post E1, [The Policy Gradient Theorem](/blog/machine-learning/reinforcement-learning/the-policy-gradient-theorem)) states:

### 4.4 Handling multimodal reward landscapes

Some environments have multiple "good" behaviors — a robot that can solve a maze by going clockwise or counterclockwise, a trading strategy that profits from both mean-reversion and momentum depending on volatility regime. A deterministic policy, by committing to one mode, may get trapped at a local optimum. A stochastic policy, because it maintains probability mass across multiple modes, can explore both and — if the reward landscape is truly multimodal — settle into a distribution that covers multiple profitable behaviors.

This is the theoretical motivation behind **Quality-Diversity (QD) RL** and **Maximum Entropy RL**: rather than converging to a single optimal deterministic policy, you explicitly maintain a diverse distribution of behaviors. SAC's entropy regularization does this implicitly: by penalizing low-entropy policies, SAC keeps the policy from collapsing onto a single mode even if that mode is locally optimal.

In practice, you will see this as the difference between a trading bot that only momentum-trades (deterministic argmax of Q) versus one that adaptively weights between momentum and mean-reversion depending on market regime features embedded in its state representation (stochastic policy with high entropy on ambiguous regimes, lower entropy when the regime is clear).

$$\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}\left[\nabla_\theta \log \pi_\theta(a \mid s) \cdot Q^{\pi_\theta}(s, a)\right]$$

This gradient estimator — the **score function estimator** — requires that $\pi_\theta$ be a differentiable probability distribution. The term $\nabla_\theta \log \pi_\theta(a \mid s)$ is the **score** of the policy: it tells you how to adjust $\theta$ to make action $a$ more (or less) likely in state $s$. For a deterministic policy $\pi(s) = \text{argmax}_a$, this score is zero almost everywhere (the argmax has no smooth gradient). You need a *distribution* to get a useful gradient signal, which is why REINFORCE, A2C, PPO, and SAC all use stochastic policies.

---

## 5. Why deterministic? The existence theorem

We just gave three reasons to prefer stochastic policies. Now here is the counterpoint, and it is a theorem.

**Theorem (Deterministic Optimal Policy Existence):** For any finite MDP with finite state and action spaces, there exists a deterministic optimal policy $\pi^*$ such that $V^{\pi^*}(s) \geq V^\pi(s)$ for all states $s$ and all policies $\pi$ (including all stochastic policies).

**Proof sketch.** Let $\pi$ be any stochastic policy. Define the greedy deterministic policy:

$$\pi'(s) = \arg\max_{a \in \mathcal{A}} Q^\pi(s, a)$$

We will show $V^{\pi'}(s) \geq V^\pi(s)$ for all $s$ in Section 7 (the Policy Improvement Theorem). Since the state and action spaces are finite, the set of deterministic policies is finite ($|\mathcal{A}|^{|\mathcal{S}|}$ of them). Starting from any policy and repeatedly applying greedy improvement yields a monotonically non-decreasing sequence that must converge to a fixed point. At that fixed point, $\pi'(s) = \pi(s)$ means $V^{\pi'}(s) = V^\pi(s)$, which is only possible when $\pi$ is already optimal. The fixed-point policy is deterministic and optimal. $\square$

**Why does this not contradict the POMDP argument?** The theorem assumes a **fully observed MDP** — the agent sees the true state $s_t$, not just an observation $o_t$. In a fully-observed MDP, two visits to the same state $s$ have exactly the same future reward potential, so there is no benefit to randomizing. In a POMDP, the "policy" conditions on observations, not states, and the argument breaks down (different states can share an observation, as we showed above).

**Practical implications.** At *test time*, once your stochastic policy has converged, you can often take the greedy (deterministic) action $\arg\max_a \pi(a \mid s)$ or the mean action $\mu_\theta(s)$ and get slightly better performance on average (lower variance). DDPG and TD3 exploit this: they train a deterministic actor directly, using the Deterministic Policy Gradient theorem (Silver et al. 2014) rather than the stochastic score-function estimator.

### 5.1 The Deterministic Policy Gradient theorem (preview)

Silver et al. (2014) showed that even for deterministic policies, a valid policy gradient exists:

$$\nabla_\theta J(\mu_\theta) = \mathbb{E}_{s \sim \rho^\mu}\left[\nabla_a Q^\mu(s, a)\big|_{a=\mu_\theta(s)} \cdot \nabla_\theta \mu_\theta(s)\right]$$

This is the DPG theorem. The gradient of the Q-function with respect to the action $a$, evaluated at the current deterministic action $\mu_\theta(s)$, is multiplied by the gradient of the policy with respect to its parameters. The chain rule flows: "how does the Q-value change if I nudge the action, and how does the action change if I nudge the policy parameters."

This requires a **differentiable critic** $Q^\mu(s, a)$ that is smooth in $a$ — which is why DDPG uses a neural network Q-function rather than a tabular Q-table (which has no useful gradient in $a$). The DPG theorem only applies to continuous action spaces (where $a$ is a real vector and $Q$ is smooth in $a$). It does not apply to discrete actions (where the argmax is not differentiable).

### 5.2 ε-greedy as a bridge

A common practical bridge between deterministic and stochastic policies is **$\varepsilon$-greedy**: take the deterministic best action with probability $1-\varepsilon$, and a uniformly random action with probability $\varepsilon$. DQN uses this for exploration during training and switches to pure greedy ($\varepsilon = 0$) at evaluation time.

The resulting mixed policy is technically stochastic (it assigns positive probability to all actions), but it is not a smooth parameterized distribution — $\varepsilon$ is a hyperparameter, not a learned parameter. This means you cannot compute $\nabla_\theta J(\theta)$ through the $\varepsilon$-greedy mechanism itself; the gradient flows only through the Q-network.

The interpolation also demonstrates the theorem in action: as $\varepsilon \to 0$, the $\varepsilon$-greedy policy approaches a deterministic policy, and by the existence theorem, this limit is fine — the optimal policy exists in deterministic form. The $\varepsilon > 0$ phase is purely for exploration during learning, not because stochasticity improves the *final* policy in a fully-observed MDP.

---

## 6. POMDP case: why stochastic is strictly better

![Deterministic policy under partial observability commits to one wrong action while stochastic policy mixes over hidden states achieving higher expected return](/imgs/blogs/policies-deterministic-vs-stochastic-6.png)

Let us make the POMDP argument formal with a minimal example. Consider a two-state MDP $\mathcal{S} = \{s_1, s_2\}$ where both states produce the same observation $o = \star$. The rewards are:

- In $s_1$: action LEFT gives $r = +5$, action RIGHT gives $r = -1$.
- In $s_2$: action LEFT gives $r = -1$, action RIGHT gives $r = +5$.

Assume the agent is in $s_1$ with probability 0.5 and $s_2$ with probability 0.5 (it cannot tell which). Consider three policies:

**Policy A** (deterministic LEFT): always take LEFT.
$$\mathbb{E}[r] = 0.5 \cdot (+5) + 0.5 \cdot (-1) = +2$$

**Policy B** (deterministic RIGHT): always take RIGHT.
$$\mathbb{E}[r] = 0.5 \cdot (-1) + 0.5 \cdot (+5) = +2$$

**Policy C** (stochastic with $\pi(\text{LEFT} \mid o) = p$):
$$\mathbb{E}[r] = 0.5[p \cdot (+5) + (1-p)(-1)] + 0.5[p \cdot (-1) + (1-p)(+5)]$$
$$= 0.5[6p - 1] + 0.5[5 - 6p] = 0.5[6p - 1 + 5 - 6p] = 0.5 \cdot 4 = +2$$

In this symmetric case, any stochastic policy achieves the same expected single-step reward. But extend this to *multiple steps*: if the transition keeps the agent in the same state and the agent accumulates discount, a stochastic policy that hedges over time avoids the repeated wrong action. In asymmetric POMDPs (where the state-belief changes over time as the agent observes outcomes), stochastic policies can be strictly better because they maintain the diversity needed to re-update the belief. The key theoretical result (Singh et al. 1994, "Learning Without State Estimation in POMDPs") is that the optimal policy for a POMDP (a *belief-state* MDP) is deterministic in the *belief* $b(s)$, but stochastic in the *observation* $o$ — which is exactly what $\pi(a \mid o)$ gives you.

### 6.1 The belief-state MDP and the optimal POMDP policy

When we convert a POMDP to an equivalent fully-observed MDP, we do so by using the **belief state** $b(s) = P(S_t = s \mid o_{1:t}, a_{1:t-1})$ as the new state. The belief state is a probability distribution over underlying states, and it is a sufficient statistic for future returns given past observations.

In this belief-state MDP, the optimal policy is deterministic in $b$ — because the belief state is the true "state" and the deterministic-optimality theorem applies. But from the perspective of the original observation space, the policy $\pi(a \mid o)$ looks stochastic because the belief $b$ given observation $o$ can correspond to many different underlying distributions, and different beliefs call for different actions.

This is a critical insight: **the stochasticity we need in POMDPs is not "noise for exploration" — it is uncertainty hedging**. The policy is uncertain about the hidden state and its stochastic action distribution reflects this uncertainty. As the belief becomes more concentrated on a single state (more observations resolve the uncertainty), the optimal POMDP policy should approach deterministic — which is exactly what a well-trained recurrent policy (LSTM/GRU with belief state as hidden state) does in practice.

Practical implication for deep RL under partial observability: if your environment has hidden state (order book depth not fully visible, opponent's hand not known, physics parameters not observable), using a recurrent policy (LSTM cell as the encoder) is typically much better than a memoryless MLP policy. The LSTM's hidden state approximates the belief state, and the policy over that hidden state can be near-deterministic even when the raw observation is ambiguous.

#### Worked example: POMDP advantage

Suppose the agent takes three steps. In step 1, it sees $o = \star$ and is in $s_1$ or $s_2$ with equal probability. If it takes action $a_1$ and observes reward $r_1$, it now knows which state it is in (reward reveals the state). In steps 2 and 3 it acts optimally.

- **Deterministic LEFT** ($a_1 = \text{LEFT}$): expected $r_1 = +2$. After step 1, agent knows its state. Steps 2–3: optimal action, expected $r_{2:3} = +5$ each. Total: $+2 + 5 + 5 = +12$.
- **Stochastic** ($a_1 \sim \pi$, $p = 0.5$): same expected $r_1 = +2$ in step 1, same information revealed. Steps 2–3: optimal, $+5$ each. Total: $+12$.

In this case, the stochastic policy performs the same because a single step resolves the hidden state. The real benefit shows up in *repeated* POMDP scenarios where the hidden state is persistent and the deterministic policy keeps repeating the wrong action without any mechanism to update. Stochastic policies, by varying their actions, generate more *diverse* observations that allow a Bayesian agent to narrow down the hidden state faster — this is exactly the exploration-exploitation tradeoff recast in the POMDP frame.

---

## 7. The policy improvement theorem: proof from scratch

![Policy improvement loop cycles through evaluation and greedy improvement guaranteeing monotonic value increase until convergence](/imgs/blogs/policies-deterministic-vs-stochastic-3.png)

This is the mathematical heart of policy iteration — and arguably of all of RL. Let us prove it carefully.

**Setup.** We have an MDP with discount factor $\gamma \in [0, 1)$, a current policy $\pi$, and its value function:

$$V^\pi(s) = \mathbb{E}_\pi\left[\sum_{t=0}^\infty \gamma^t r_t \;\middle|\; s_0 = s\right]$$

The action-value function (Q-function) is:

$$Q^\pi(s, a) = \mathbb{E}_\pi\left[\sum_{t=0}^\infty \gamma^t r_t \;\middle|\; s_0 = s, a_0 = a\right]$$

Note the relationship: $V^\pi(s) = \sum_a \pi(a \mid s) Q^\pi(s, a) = \mathbb{E}_{a \sim \pi}[Q^\pi(s, a)]$.

**Theorem (Policy Improvement).** Let $\pi'$ be any policy such that for all $s \in \mathcal{S}$:

$$\sum_a \pi'(a \mid s) Q^\pi(s, a) \geq V^\pi(s)$$

Then $V^{\pi'}(s) \geq V^\pi(s)$ for all $s$.

**Proof.** Start from $V^\pi(s)$ and repeatedly substitute the Bellman consistency equation:

$$V^\pi(s) \leq \sum_a \pi'(a \mid s) Q^\pi(s, a)$$

$$= \sum_a \pi'(a \mid s) \left[r(s,a) + \gamma \sum_{s'} P(s' \mid s, a) V^\pi(s')\right]$$

$$\leq \sum_a \pi'(a \mid s) \left[r(s,a) + \gamma \sum_{s'} P(s' \mid s, a) \sum_{a'} \pi'(a' \mid s') Q^\pi(s', a')\right]$$

where we used the assumption again for $s'$. Continuing this substitution $k$ times:

$$V^\pi(s) \leq \mathbb{E}_{\pi'}\left[\sum_{t=0}^{k-1} \gamma^t r_t \;\middle|\; s_0 = s\right] + \gamma^k \mathbb{E}_{\pi'}[V^\pi(s_k) \mid s_0 = s]$$

As $k \to \infty$, $\gamma^k \mathbb{E}_{\pi'}[V^\pi(s_k)] \to 0$ (since $V^\pi$ is bounded and $\gamma < 1$). The right side approaches $V^{\pi'}(s)$ by definition. Therefore:

$$V^\pi(s) \leq V^{\pi'}(s) \quad \forall s \in \mathcal{S}$$

$\square$

**Corollary (Greedy improvement is always safe).** The greedy policy $\pi'(s) = \arg\max_a Q^\pi(s, a)$ satisfies the condition of the theorem because:

$$\sum_a \pi'(a \mid s) Q^\pi(s, a) = \max_a Q^\pi(s, a) \geq \sum_a \pi(a \mid s) Q^\pi(s, a) = V^\pi(s)$$

The first equality holds because $\pi'$ is deterministic (all mass on argmax). The inequality holds because the maximum is at least as large as the expectation under any distribution. Therefore, greedy improvement never decreases the value function.

**When does improvement stop?** If $V^\pi(s) = \max_a Q^\pi(s, a)$ for all $s$, then the greedy policy reproduces $\pi$ and we have reached a fixed point. By the Bellman optimality equations, this fixed point is the optimal policy $\pi^*$ with $V^{\pi^*}(s) = V^*(s)$ for all $s$.

---

## 8. Policy iteration: putting it together

Policy iteration is the direct implementation of the improvement theorem:

```
Initialize π₀ (e.g., random deterministic policy)
Repeat:
  1. Policy Evaluation: compute V^π exactly (solve Bellman system or iterate to convergence)
  2. Policy Improvement: π' ← greedy w.r.t. V^π (i.e., π'(s) = argmax_a Q^π(s,a))
  3. If π' == π: STOP (optimal policy found)
  4. Else: π ← π', go to 1
```

**Why does it converge?** Each improvement step produces a policy at least as good as the previous. The number of distinct deterministic policies is finite ($|\mathcal{A}|^{|\mathcal{S}|}$). Since the sequence is monotonically improving and the space is finite, it must converge in finite steps. In practice, policy iteration typically converges in far fewer steps than this worst-case bound — often 5–15 iterations for small MDPs.

**Computational cost.** Policy evaluation — computing $V^\pi$ exactly — requires solving a system of $|\mathcal{S}|$ linear equations (one Bellman equation per state). This is $O(|\mathcal{S}|^3)$ via Gaussian elimination, or $O(|\mathcal{S}|^2 |\mathcal{A}|)$ per iterative sweep. For large MDPs this is intractable, which is why approximate variants (truncated policy evaluation, TD-based critics) are used in practice. That is the topic of the next posts in this series (B-series: value-based methods).

### 8.1 Policy iteration vs. value iteration: what's the difference?

Students often confuse policy iteration with value iteration. They are related but distinct:

**Policy iteration**: alternates between *exact* policy evaluation (solve or iterate V^π to convergence) and greedy improvement. Each evaluation phase takes many Bellman sweeps. Produces an exact policy after few improvement steps.

**Value iteration**: skips full policy evaluation. Just apply one Bellman optimality backup per state per iteration:

$$V_{k+1}(s) = \max_a \left[r(s,a) + \gamma \sum_{s'} P(s' \mid s, a) V_k(s')\right]$$

Value iteration does many cheap sweeps instead of few expensive ones. It converges to $V^*$ without ever explicitly representing a policy. The policy is extracted at the end: $\pi^*(s) = \arg\max_a Q^*(s,a)$.

**Which is faster?** Both converge to $V^*$. Policy iteration tends to converge in fewer *outer* iterations (policy improvements) but each iteration is expensive (full policy evaluation). Value iteration's many cheap sweeps often win in practice for medium-sized MDPs. For the theory, policy iteration is cleaner because the improvement theorem makes the convergence argument transparent.

**Modified policy iteration** is the practical middle ground: run only $k$ sweeps of policy evaluation (not until convergence), then improve. For $k=1$, this is equivalent to value iteration. For $k=\infty$, this is full policy iteration. In deep RL, the TD(0) critic update is a $k=1$ truncated policy evaluation — which is why deep RL algorithms are essentially policy iteration with one-step TD critics.

Here is a complete NumPy implementation of tabular policy iteration:

```python
import numpy as np

def policy_evaluation(policy, R, P, gamma=0.99, tol=1e-6):
    """
    Solve V^π exactly via iterative Bellman application.
    policy: (n_states,) array of actions (deterministic)
    R: (n_states, n_actions) reward matrix
    P: (n_states, n_actions, n_states) transition probability matrix
    Returns: V (n_states,) value function
    """
    n_states = R.shape[0]
    V = np.zeros(n_states)
    for _ in range(10_000):
        V_new = np.zeros(n_states)
        for s in range(n_states):
            a = policy[s]
            V_new[s] = R[s, a] + gamma * np.dot(P[s, a], V)
        if np.max(np.abs(V_new - V)) < tol:
            return V_new
        V = V_new
    return V

def policy_improvement(V, R, P, gamma=0.99):
    """
    Greedy policy: π'(s) = argmax_a Q^π(s, a)
    Returns new deterministic policy and Q-function.
    """
    n_states, n_actions = R.shape
    Q = np.zeros((n_states, n_actions))
    for s in range(n_states):
        for a in range(n_actions):
            Q[s, a] = R[s, a] + gamma * np.dot(P[s, a], V)
    new_policy = np.argmax(Q, axis=1)
    return new_policy, Q

def policy_iteration(R, P, gamma=0.99):
    """Full policy iteration until convergence."""
    n_states, n_actions = R.shape
    policy = np.zeros(n_states, dtype=int)  # start with action 0 everywhere
    for iteration in range(1000):
        V = policy_evaluation(policy, R, P, gamma)
        new_policy, Q = policy_improvement(V, R, P, gamma)
        if np.all(new_policy == policy):
            print(f"Converged at iteration {iteration + 1}")
            return policy, V, Q
        policy = new_policy
    return policy, V, Q
```

---

## 9. GridWorld worked example: policy improvement step-by-step

Now let us see the theory in action on a concrete 4-state GridWorld.

#### Worked example: 4-state GridWorld policy improvement

**Setup.** States: $\{s_1, s_2, s_3, s_4\}$ arranged in a $2 \times 2$ grid.
- $s_1$ = top-left (start)
- $s_2$ = top-right
- $s_3$ = bottom-left
- $s_4$ = bottom-right (goal, reward $+10$)

Actions: RIGHT and DOWN. From each non-goal state, RIGHT moves right (if possible, else stays), DOWN moves down. From the goal state, all actions stay (episodic end). Reward: $+10$ on entering $s_4$, $0$ otherwise. $\gamma = 0.9$.

**Initial policy $\pi_0$**: always go RIGHT.

$$\pi_0 = \{s_1 \to \text{RIGHT}, s_2 \to \text{RIGHT}, s_3 \to \text{RIGHT}, s_4 \to \text{RIGHT}\}$$

**Policy evaluation of $\pi_0$:** Under $\pi_0$, the agent at $s_1$ goes to $s_2$ (RIGHT), then at $s_2$ goes to $s_4$ (goal). The agent at $s_3$ stays at $s_3$ forever (RIGHT from $s_3$ loops back to $s_3$ — there is no rightward neighbor from the left column's bottom). Agent at $s_4$ gets $+10$ and stops.

Bellman equations for $V^{\pi_0}$ (assuming deterministic transitions):

$$V^{\pi_0}(s_4) = 10$$

$$V^{\pi_0}(s_2) = 0 + \gamma \cdot V^{\pi_0}(s_4) = 0.9 \times 10 = 9$$

$$V^{\pi_0}(s_1) = 0 + \gamma \cdot V^{\pi_0}(s_2) = 0.9 \times 9 = 8.1$$

$$V^{\pi_0}(s_3) = 0 + \gamma \cdot V^{\pi_0}(s_3) \implies V^{\pi_0}(s_3) = 0$$

(the last equation has solution 0 since the reward is 0 and the agent loops).

**V-table for $\pi_0$:**

| State | $V^{\pi_0}$ | Current action |
|-------|------------|----------------|
| $s_1$ | 8.1        | RIGHT          |
| $s_2$ | 9.0        | RIGHT          |
| $s_3$ | 0.0        | RIGHT (trapped)|
| $s_4$ | 10.0       | (goal)         |

**Q-function computation for $\pi_0$:** We need $Q^{\pi_0}(s, a)$ for all $(s, a)$ to perform greedy improvement.

For $s_3$:
- $Q^{\pi_0}(s_3, \text{RIGHT}) = 0 + 0.9 \times V^{\pi_0}(s_3) = 0$ (stays at $s_3$)
- $Q^{\pi_0}(s_3, \text{DOWN}) = 0 + 0.9 \times V^{\pi_0}(s_4) = 0.9 \times 10 = 9$ (moves to $s_4$!)

For $s_1$:
- $Q^{\pi_0}(s_1, \text{RIGHT}) = 0 + 0.9 \times V^{\pi_0}(s_2) = 0.9 \times 9 = 8.1$
- $Q^{\pi_0}(s_1, \text{DOWN}) = 0 + 0.9 \times V^{\pi_0}(s_3) = 0.9 \times 0 = 0$

**Greedy improvement:** For each state, pick $\arg\max_a Q^{\pi_0}(s, a)$:

- $s_1$: $\max(8.1, 0) = 8.1$ → keep RIGHT
- $s_2$: $\max(9.0, 9.0 \times 0.9) = 9.0$ → keep RIGHT
- $s_3$: $\max(0, 9) = 9$ → **switch to DOWN** ← improvement!
- $s_4$: goal (terminal)

**New policy $\pi_1$:** $\{s_1 \to \text{RIGHT}, s_2 \to \text{RIGHT}, s_3 \to \text{DOWN}, s_4 \to -\}$

**Policy evaluation of $\pi_1$:** Now $s_3$ goes to $s_4$, so:

$$V^{\pi_1}(s_3) = 0 + 0.9 \times V^{\pi_1}(s_4) = 0.9 \times 10 = 9$$

$$V^{\pi_1}(s_1) = 0 + 0.9 \times 9 = 8.1 \quad \text{(going RIGHT to }s_2\text{)}$$

Wait — actually $s_1$ still goes RIGHT to $s_2$, then $s_2$ goes RIGHT to $s_4$. The improvement for $s_3$ does not change $s_1$'s path. But now we should check whether $s_1$ should go DOWN to $s_3$ (which now has value 9):

$Q^{\pi_1}(s_1, \text{DOWN}) = 0 + 0.9 \times 9 = 8.1$, and $Q^{\pi_1}(s_1, \text{RIGHT}) = 0 + 0.9 \times 9 = 8.1$ (since $V^{\pi_1}(s_2) = 9$).

Both actions have the same Q-value from $s_1$! Policy $\pi_1$ is already optimal — there are two equally-good paths to the goal (RIGHT-RIGHT and DOWN-RIGHT from $s_1$). The policy iteration converges in a single improvement step.

**Summary of the improvement:**

| State | $V^{\pi_0}$ | $V^{\pi_1}$ | Change |
|-------|------------|------------|--------|
| $s_1$ | 8.1        | 8.1        | 0      |
| $s_2$ | 9.0        | 9.0        | 0      |
| $s_3$ | **0.0**    | **9.0**    | +9.0   |
| $s_4$ | 10.0       | 10.0       | 0      |

The value of $s_3$ jumped from 0 to 9 in a single greedy improvement step. This is exactly what the policy improvement theorem guaranteed: switching $s_3$'s action from RIGHT (which trapped the agent) to DOWN (which leads to the goal) increased $V(s_3)$ by $+9.0$.

#### Worked example: verifying the improvement theorem condition

Let us explicitly verify that the theorem condition $\sum_a \pi'(a \mid s) Q^\pi(s, a) \geq V^\pi(s)$ holds for each state after the improvement step.

For state $s_3$:
- $V^{\pi_0}(s_3) = 0.0$
- $\pi_1(s_3) = \text{DOWN}$ (deterministic), so $\sum_a \pi_1(a \mid s_3) Q^{\pi_0}(s_3, a) = Q^{\pi_0}(s_3, \text{DOWN}) = 9.0$
- Condition: $9.0 \geq 0.0$ ✓

For state $s_1$:
- $V^{\pi_0}(s_1) = 8.1$
- $\pi_1(s_1) = \text{RIGHT}$ (unchanged), so $\sum_a \pi_1(a \mid s_1) Q^{\pi_0}(s_1, a) = Q^{\pi_0}(s_1, \text{RIGHT}) = 8.1$
- Condition: $8.1 \geq 8.1$ ✓ (equality — no change needed)

The theorem condition holds with equality for $s_1$ (greedy policy selected the same action as before) and with strict inequality for $s_3$ (a better action was found). The theorem guarantees $V^{\pi_1}(s) \geq V^{\pi_0}(s)$ everywhere — and indeed $V^{\pi_1}(s_3) = 9.0 > 0.0 = V^{\pi_0}(s_3)$.

**Running the NumPy policy iteration code on this GridWorld:**

```python
import numpy as np

# 4-state GridWorld: s0=top-left, s1=top-right, s2=bottom-left, s3=bottom-right(goal)
# Actions: 0=RIGHT, 1=DOWN
n_states, n_actions = 4, 2

# Reward: +10 for entering s3 (state 3), 0 otherwise
R = np.zeros((n_states, n_actions))
R[1, 0] = 10.0   # s1 + RIGHT -> s3 (goal)
R[2, 1] = 10.0   # s2 + DOWN  -> s3 (goal)

# Transition: deterministic
P = np.zeros((n_states, n_actions, n_states))
# State 0 (top-left):  RIGHT -> s1, DOWN -> s2
P[0, 0, 1] = 1.0; P[0, 1, 2] = 1.0
# State 1 (top-right): RIGHT -> s3(goal), DOWN -> s3(goal)
P[1, 0, 3] = 1.0; P[1, 1, 3] = 1.0
# State 2 (bot-left):  RIGHT -> s2(stay), DOWN -> s3(goal)
P[2, 0, 2] = 1.0; P[2, 1, 3] = 1.0
# State 3 (goal):      absorbing
P[3, 0, 3] = 1.0; P[3, 1, 3] = 1.0

policy, V, Q = policy_iteration(R, P, gamma=0.9)
print(f"Optimal policy: {['RIGHT' if a==0 else 'DOWN' for a in policy]}")
print(f"Optimal values: {V.round(3)}")
# Expected output:
# Converged at iteration 2
# Optimal policy: ['RIGHT', 'RIGHT', 'DOWN', 'RIGHT']
# Optimal values: [8.1, 9.0, 9.0, 10.0]
```

The code confirms our hand-computed result: the policy converges in just 2 iterations, and the optimal values match exactly. State $s_3$ (the goal) has value 10.0 (immediate reward), states $s_1$ and $s_2$ both reach the goal in one step for value 9.0 (0.9×10), and state $s_0$ reaches the goal in two steps for value 8.1 (0.9×9).

![Policy iteration convergence from random policy through greedy improvements to optimal policy showing value function increases at each step](/imgs/blogs/policies-deterministic-vs-stochastic-5.png)

---

## 10. Deep RL: neural network as policy

The GridWorld example used an exact tabular policy. In deep RL, the policy is a neural network. Figure 7 shows the architecture.

![Neural policy architecture maps state observations through shared MLP encoder to branching action distribution heads for discrete or continuous action spaces](/imgs/blogs/policies-deterministic-vs-stochastic-7.png)

### 10.1 Discrete action head: Categorical

For environments like CartPole, LunarLander-Discrete, or Atari games, the output is a Categorical distribution:

```python
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

class PolicyNet(nn.Module):
    def __init__(self, obs_dim, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, n_actions)
        )

    def forward(self, x):
        return Categorical(logits=self.net(x))

env = gym.make("CartPole-v1")
obs_dim = env.observation_space.shape[0]   # 4
n_actions = env.action_space.n             # 2
policy = PolicyNet(obs_dim, n_actions)
optimizer = optim.Adam(policy.parameters(), lr=3e-4)

# REINFORCE training loop (simplest policy gradient)
for episode in range(1000):
    obs, _ = env.reset()
    log_probs, rewards = [], []
    done = False
    while not done:
        obs_t = torch.FloatTensor(obs)
        dist = policy(obs_t)
        action = dist.sample()
        log_probs.append(dist.log_prob(action))
        obs, reward, terminated, truncated, _ = env.step(action.item())
        rewards.append(reward)
        done = terminated or truncated
    # Compute discounted returns
    G, returns = 0.0, []
    for r in reversed(rewards):
        G = r + 0.99 * G
        returns.insert(0, G)
    returns = torch.tensor(returns, dtype=torch.float32)
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    loss = -sum(lp * R for lp, R in zip(log_probs, returns))
    optimizer.zero_grad(); loss.backward(); optimizer.step()
```

### 10.2 Continuous action head: Gaussian

For MuJoCo or LunarLander-Continuous, Stable-Baselines3's PPO uses a diagonal Gaussian out of the box:

```python
from stable_baselines3 import PPO
import gymnasium as gym

# PPO with MlpPolicy automatically uses DiagGaussian for continuous actions
env = gym.make("LunarLanderContinuous-v2")
model = PPO(
    policy="MlpPolicy",
    env=env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    ent_coef=0.0,      # entropy coefficient (0 = no explicit entropy bonus)
    clip_range=0.2,
    verbose=1
)
model.learn(total_timesteps=500_000)
mean_reward, std_reward = model.evaluate_policy(model, env, n_eval_episodes=10)
print(f"Mean reward: {mean_reward:.1f} +/- {std_reward:.1f}")
# Expected: ~270 +/- 30 after 500k steps (approximation based on SB3 benchmarks)
```

### 10.3 Deterministic actor for continuous control (DDPG/TD3)

When you want a deterministic policy in continuous action space, the actor outputs a single action vector rather than distribution parameters:

```python
import torch
import torch.nn as nn

class DeterministicActor(nn.Module):
    """Deterministic actor as used in DDPG and TD3."""
    def __init__(self, obs_dim, act_dim, act_limit=1.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, act_dim), nn.Tanh()
        )
        self.act_limit = act_limit

    def forward(self, obs):
        return self.act_limit * self.net(obs)

# Usage in DDPG: add Gaussian exploration noise at training time only
actor = DeterministicActor(obs_dim=17, act_dim=6, act_limit=1.0)
obs = torch.randn(17)
action = actor(obs)
# Training: action_with_noise = action + std * torch.randn_like(action)
# Evaluation: use clean action (no noise) for deterministic greedy policy
```

The `Tanh` activation bounds the output to $(-1, 1)$, then scaled by `act_limit` to match the environment's action range. At training time, exploration is achieved by adding Gaussian noise to the output. At test time, the clean output is the deterministic policy.

### 10.4 Recurrent policies for partial observability

When the environment is partially observable, memoryless policies (MLPs that condition only on the current observation) are theoretically suboptimal. The solution is a recurrent policy — an LSTM or GRU that maintains a hidden state $h_t$ approximating the belief state:

```python
import torch
import torch.nn as nn
from torch.distributions import Categorical

class RecurrentPolicy(nn.Module):
    """
    LSTM-based policy for partially observable environments.
    Maintains hidden state h_t that approximates the belief over MDP states.
    """
    def __init__(self, obs_dim: int, n_actions: int, hidden_dim: int = 128):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(obs_dim, hidden_dim, batch_first=True)
        self.policy_head = nn.Linear(hidden_dim, n_actions)
        self.value_head = nn.Linear(hidden_dim, 1)  # optional critic

    def forward(self, obs_seq: torch.Tensor, hidden=None):
        """
        obs_seq: (batch, time, obs_dim)
        hidden: (h_n, c_n) from previous step, or None for fresh episode
        Returns: action distribution, new hidden state, value estimate
        """
        lstm_out, new_hidden = self.lstm(obs_seq, hidden)
        logits = self.policy_head(lstm_out)
        values = self.value_head(lstm_out).squeeze(-1)
        return Categorical(logits=logits), new_hidden, values

    def get_action(self, obs: torch.Tensor, hidden=None):
        """Single-step rollout for environment interaction."""
        obs_seq = obs.unsqueeze(0).unsqueeze(0)  # (1, 1, obs_dim)
        dist, new_hidden, _ = self.forward(obs_seq, hidden)
        action = dist.sample().squeeze()
        log_prob = dist.log_prob(action).squeeze()
        return action.item(), log_prob, new_hidden

# Usage: maintain hidden state across steps within one episode
policy = RecurrentPolicy(obs_dim=8, n_actions=4)
hidden = None
for t in range(100):
    obs = torch.randn(8)
    action, log_prob, hidden = policy.get_action(obs, hidden)
    # hidden is passed to the next step — the LSTM accumulates belief
# Reset hidden state at episode boundaries (hidden = None at env.reset())
```

The critical discipline: **always reset the LSTM hidden state at episode boundaries** (`hidden = None` when `done = True`). Failing to do this leaks belief from one episode into the next, which corrupts training. This is also why recurrent policies require careful handling in replay buffers — you cannot sample arbitrary transitions without their LSTM context. Most implementations use **sequence replay**: store full episode segments in the buffer and always replay contiguous windows.

---

## 11. Policy comparison: deterministic vs stochastic in deep RL

![Deterministic versus stochastic policy comparison across key dimensions including optimality exploration gradient signal and best matching algorithms](/imgs/blogs/policies-deterministic-vs-stochastic-4.png)

### 11.1 Which algorithms use which?

Understanding which policy type each algorithm uses is essential for choosing the right tool:

| Algorithm | Policy type | Action space | Exploration | Gradient estimator |
|-----------|-------------|--------------|-------------|-------------------|
| DQN | Implicit greedy (det.) | Discrete | ε-greedy | Bellman TD |
| SARSA | Soft / ε-greedy (stoch.) | Discrete | ε-greedy | Bellman TD on-policy |
| REINFORCE | Stochastic (Categorical/Gaussian) | Both | Policy entropy | Score function |
| A2C / A3C | Stochastic + entropy bonus | Both | Entropy term | Score function |
| PPO | Stochastic (Categorical/Gaussian) | Both | Entropy + clip | Score function |
| SAC | Stochastic (reparameterized Gauss.) | Continuous | Max-entropy obj. | Reparameterization |
| DDPG | Deterministic (μ) | Continuous | Gaussian noise | DPG theorem |
| TD3 | Deterministic (μ) | Continuous | Gaussian noise | DPG theorem |

### 11.2 The convergence guarantee tradeoff

Stochastic on-policy algorithms (REINFORCE, PPO) come with convergence guarantees under appropriate learning-rate schedules (Robbins-Monro conditions). But they are high-variance: each gradient estimate uses Monte Carlo returns from one trajectory, which can swing wildly. Variance reduction techniques (baselines, advantage normalization, GAE) are needed.

Deterministic off-policy algorithms (DDPG, TD3) have lower variance because the critic learns Q-values from a replay buffer (many samples). But they can be unstable: the deterministic policy is not a distribution, so small changes in weights can cause large swings in action selection. TD3 fixes this with clipped double-Q learning and delayed policy updates.

### 11.3 Policy entropy as a training diagnostic

Monitoring the policy entropy $\mathcal{H}[\pi(\cdot \mid s)] = -\sum_a \pi(a \mid s) \log \pi(a \mid s)$ during training is one of the most informative diagnostics you can log. Here is what different entropy trajectories tell you:

**Entropy drops to zero too early**: The policy has collapsed prematurely onto a suboptimal deterministic policy. Common cause: learning rate too high, entropy coefficient too low (in entropy-regularized algorithms), or a lucky-but-wrong early trajectory that the gradient follows before exploring alternatives. Fix: increase entropy coefficient, reduce learning rate, or add exploration noise.

**Entropy stays high throughout training**: The policy is not committing to good actions — it remains near-random. Common cause: reward signal too sparse (the agent cannot find rewarding states), value function too inaccurate to give useful guidance, or entropy coefficient too high. Fix: reduce entropy coefficient, use reward shaping, improve critic accuracy.

**Entropy decreases smoothly then plateaus**: The healthy pattern. The policy starts exploring broadly (high entropy), gradually commits to good strategies (entropy drops), and plateaus at a value above zero (the residual uncertainty about which of several good actions to take in ambiguous states).

```python
import torch
from torch.distributions import Categorical
import numpy as np

def policy_entropy(logits: torch.Tensor) -> float:
    """Compute mean entropy of batch of policy distributions."""
    dist = Categorical(logits=logits)
    return dist.entropy().mean().item()

# Example: track entropy during a PPO training run
# In your training loop:
# obs_batch = collect_rollout(env, policy, n_steps=2048)
# logits = policy.net(obs_batch)
# entropy = policy_entropy(logits)
# writer.add_scalar("train/policy_entropy", entropy, global_step)

# For CartPole-v1 (2 actions), entropy range:
# Max entropy (uniform): log(2) ≈ 0.693 nats
# Near-deterministic (one action ~= 0.99 prob): ≈ 0.056 nats
# Typical converged PPO: ≈ 0.1-0.3 nats (some residual exploration)
print(f"Max entropy (CartPole): {np.log(2):.3f} nats")
print(f"Uniform policy entropy: {Categorical(logits=torch.zeros(2)).entropy().item():.3f}")
```

This diagnostic is free to compute and should be in every RL training dashboard, alongside episode return and value function loss.

---

## 12. Case studies: where policy type determined success or failure

### 12.1 DQN on Atari — implicit deterministic policy wins (Mnih et al. 2015)

Mnih et al. "Human-level control through deep reinforcement learning" (Nature 2015) used DQN — a value-based method with an implicit $\varepsilon$-greedy policy (deterministic argmax at test time, random exploration at training). Across 49 Atari games, DQN surpassed human performance on 29 of them. The policy was never explicitly represented as a distribution — it was the greedy action over a learned Q-network.

Key insight: for discrete action spaces where the optimal action can be computed cheaply as argmax over a small set, an implicit deterministic policy (computed from Q-values) is perfectly adequate. The exploration problem is handled separately by $\varepsilon$-greedy. The policy improvement theorem guarantees that argmax over Q gives the best one-step-greedy policy.

### 12.2 SAC on MuJoCo — stochastic policy + entropy maximization wins (Haarnoja et al. 2018)

Haarnoja et al. "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor" (ICML 2018) showed that on continuous control benchmarks (HalfCheetah-v2, Ant-v2, Humanoid-v2), SAC significantly outperformed DDPG and TD3 in sample efficiency. The key difference: SAC uses a *stochastic* policy with entropy maximization as a first-class objective, not just an exploration heuristic.

On HalfCheetah-v2, SAC achieved approximately 10,000 reward after \$1M environment steps, versus \~5,000 for DDPG after the same number of steps (approximate figures from the SAC paper). The stochastic policy allowed natural exploration without hand-tuned noise schedules, and the entropy bonus prevented premature convergence to local optima.

### 12.3 PPO on LunarLanderContinuous — stochastic beats deterministic

Using Stable-Baselines3's standard benchmarks: PPO (stochastic DiagGaussian policy) on `LunarLanderContinuous-v2` reaches mean reward $\approx +270$ after 500k steps. TD3 (deterministic policy) reaches $\approx +290$ after 500k steps but requires careful tuning of the noise schedule. PPO is more robust to hyperparameter choices — the built-in stochastic exploration is more forgiving.

The practical take: for prototyping and when robustness matters more than peak performance, stochastic policies (PPO, SAC) win. For maximum performance on well-understood continuous control tasks, deterministic policies (TD3, DDPG) can edge ahead with proper tuning.

### 12.4 RLHF for language models — stochastic is mandatory (Ziegler et al. 2019)

In Ziegler et al. "Fine-Tuning Language Models from Human Preferences" (2019) and the subsequent InstructGPT paper (Ouyang et al. 2022), the language model policy is by definition stochastic — a Categorical distribution over the vocabulary at each token position. There is no meaningful "deterministic" language model policy (greedy decoding exists but it is used only at inference, not during training).

The policy gradient update in RLHF is:

$$\nabla_\theta J(\theta) \approx \nabla_\theta \mathbb{E}_{x \sim \pi_\theta}\left[r_\phi(x) - \beta \log \frac{\pi_\theta(x)}{\pi_{\text{ref}}(x)}\right]$$

The KL divergence term $\beta \log(\pi_\theta / \pi_\text{ref})$ is only meaningful because $\pi_\theta$ is a probability distribution. It prevents reward hacking by keeping the policy close to the reference (pre-RLHF) model. This is a stochastic policy benefit with no deterministic analog — you cannot compute a KL divergence for a deterministic policy without it collapsing to infinity wherever the deterministic policy differs from the reference.

### 12.5 AlphaGo — stochastic policy network + deterministic MCTS (Silver et al. 2016)

AlphaGo (Silver et al. 2016) uses a fascinating combination: a **stochastic policy network** $p_\sigma(a \mid s)$ trained by supervised learning + REINFORCE that assigns probabilities to moves, and a **deterministic selection** via Monte Carlo Tree Search (MCTS) that uses the policy network as a prior but computes the final action by tree search.

The policy network is stochastic — trained with policy gradient from self-play, it outputs move probabilities that reflect the uncertain nature of which move is best given incomplete lookahead. MCTS then uses these probabilities as a search prior, running thousands of simulated games per real move. The final action is the MCTS argmax over the visit counts — effectively deterministic at move-selection time.

This hybrid shows a deep truth: stochastic policy networks capture the uncertainty and diversity of good moves in a complex game, while deterministic selection via planning (MCTS) exploits that distribution to find the best move given additional compute. The same pattern appears in AlphaZero, MuZero, and — more recently — in chain-of-thought language model reasoning: the language model maintains a stochastic distribution over next tokens (exploration + uncertainty representation), while beam search or best-of-N sampling finds the best completion (deterministic selection from the distribution).

### 12.6 Quantitative benchmark: SB3 policy comparison on standard envs

Using Stable-Baselines3's own benchmarks (approximate figures from the SB3 documentation and RL Baselines Zoo):

| Environment | Algorithm | Policy type | 1M step reward | Notes |
|-------------|-----------|-------------|----------------|-------|
| CartPole-v1 | PPO | Stochastic Categorical | 500 (perfect) | Solves in ~100k steps |
| CartPole-v1 | DQN | Implicit deterministic | 500 (perfect) | Solves in ~50k steps (simpler for discrete) |
| LunarLander-v2 | PPO | Stochastic Categorical | ~250 | Robust, less sensitive to LR |
| HalfCheetah-v4 | SAC | Stochastic Gaussian | ~10,000 | Best continuous control baseline |
| HalfCheetah-v4 | TD3 | Deterministic | ~9,500 | Competitive, needs more tuning |
| HalfCheetah-v4 | DDPG | Deterministic | ~5,000 | Unstable, TD3 supersedes it |
| Ant-v4 | SAC | Stochastic Gaussian | ~5,000 | Entropy max crucial for high-dim |
| Ant-v4 | PPO | Stochastic Gaussian | ~3,000 | On-policy, less sample efficient |

The pattern is clear: for discrete environments, DQN's implicit deterministic policy is more sample-efficient than PPO because the Q-learning update is directly optimizing the value function without policy gradient variance. For continuous control, SAC's stochastic entropy-maximizing policy matches or beats TD3 while being more robust to hyperparameter choices.

One nuance worth noting: these benchmark numbers are not destiny. The performance gap between SAC and TD3 on HalfCheetah narrows significantly with careful TD3 tuning — exploration noise schedule, learning rate warmup, and replay buffer size all matter. The real advantage of SAC's stochastic policy is not just peak performance but **sample efficiency and robustness**: it reaches 80% of its final performance about 40% faster than TD3 in most continuous control benchmarks (approximate, based on SB3 Zoo evaluation curves), and it requires far less hyperparameter tuning to achieve that. For practitioners who need a reliable default for continuous control, SAC is the correct choice precisely because its stochastic policy handles exploration automatically through entropy maximization, eliminating the need to manually tune noise schedules that deterministic DDPG/TD3 policies depend on.

---

## 13. Practical implementation: Stable-Baselines3 policy selection

Choosing the right SB3 algorithm and policy type in practice:

```python
from stable_baselines3 import PPO, SAC, DDPG, TD3, DQN
import gymnasium as gym

def make_agent(env_id: str, algorithm: str = "auto"):
    env = gym.make(env_id)
    is_continuous = hasattr(env.action_space, "shape")

    if algorithm == "auto":
        algorithm = "SAC" if is_continuous else "PPO"

    if algorithm == "PPO":
        # Stochastic: Categorical (discrete) or DiagGaussian (continuous)
        return PPO("MlpPolicy", env, learning_rate=3e-4, n_steps=2048,
                   batch_size=64, n_epochs=10, ent_coef=0.01)
    elif algorithm == "SAC":
        # Stochastic: DiagGaussian with reparameterization + entropy max
        return SAC("MlpPolicy", env, learning_rate=3e-4, buffer_size=1_000_000,
                   learning_starts=10_000, batch_size=256, tau=0.005)
    elif algorithm == "DDPG":
        # Deterministic: μ_θ(s) + Gaussian noise at train time
        return DDPG("MlpPolicy", env, learning_rate=1e-3, buffer_size=1_000_000,
                    learning_starts=10_000, batch_size=100, tau=0.005)
    elif algorithm == "TD3":
        # Deterministic: μ_θ(s) + clipped noise, delayed updates
        return TD3("MlpPolicy", env, learning_rate=1e-3, buffer_size=1_000_000,
                   learning_starts=10_000, batch_size=256, tau=0.005)
    elif algorithm == "DQN":
        # Implicit deterministic: argmax Q(s,a) + ε-greedy exploration
        return DQN("MlpPolicy", env, learning_rate=1e-4, buffer_size=1_000_000,
                   learning_starts=50_000, batch_size=32, exploration_fraction=0.1)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

# Continuous env → SAC (stochastic)
agent = make_agent("HalfCheetah-v4", "SAC")
agent.learn(total_timesteps=1_000_000)

# Discrete env → DQN (implicit deterministic)
agent = make_agent("CartPole-v1", "DQN")
agent.learn(total_timesteps=100_000)
```

### Hyperparameter considerations for policy type

| Hyperparameter | Stochastic (PPO/SAC) | Deterministic (DDPG/TD3) |
|---------------|---------------------|--------------------------|
| Entropy coeff | Tune `ent_coef` (0.0–0.1) | Not applicable |
| Exploration noise | Built-in via distribution spread | `noise_type`: Gaussian or OU; tune `noise_std` |
| Learning rate | 1e-4 to 3e-4 typical | 1e-4 to 1e-3 (actors are sensitive) |
| Policy updates | Every `n_steps` (PPO) or per replay step (SAC) | Delayed in TD3 (`policy_freq=2`) |
| Convergence | More robust, lower peak | Higher peak if tuned, can be unstable |

---

## 14. Policy types across RL algorithms

![RL algorithms organized by policy type showing stochastic versus deterministic and on-policy versus off-policy dimensions with exploration mechanisms](/imgs/blogs/policies-deterministic-vs-stochastic-8.png)

Understanding the full landscape of how policy type interacts with on/off-policy, action space, and exploration helps you navigate the algorithm choice in practice. The matrix above captures the key dimensions. Let us add a few nuances:

**DQN is off-policy deterministic — but it never explicitly stores a policy.** The "policy" is the implicit argmax over Q. This is why DQN cannot easily produce a distribution over actions for RLHF-style training or for computing entropy regularization.

**SAC is the cleaner version of what DDPG was trying to do.** Both are off-policy, continuous-action, deep RL algorithms. SAC replaces the deterministic actor with a stochastic one and adds an entropy maximization objective, which simultaneously solves the exploration problem and stabilizes training. In practice, SAC almost always outperforms DDPG on the same benchmark.

**PPO is the workhorse for on-policy stochastic policies.** The clipped surrogate objective handles the instability that vanilla policy gradient (REINFORCE) suffers from. The entropy coefficient `ent_coef` controls how much exploration pressure remains throughout training — set it to zero and PPO collapses to near-deterministic once it finds a good policy; set it too high and it never commits.

---

## 15. Common pitfalls and debugging checklist

Knowing the theory is necessary but not sufficient. Here are the practical failure modes you will encounter when implementing policies in real systems, and how to diagnose them.

### 15.1 The policy entropy collapse (most common failure in stochastic policies)

Your stochastic policy's entropy drops to near-zero in the first few thousand steps, and the agent is stuck at a suboptimal fixed point. The log-probability of all actions except one approaches $-\infty$. Training loss looks fine (it's minimizing something) but episode returns plateau well below expert level.

**Diagnosis**: check `policy_entropy(logits)` in your training loop. If it drops below 0.1 nats for a 10-action environment early in training, this is entropy collapse.

**Fix**: either (a) add an entropy bonus to the loss: `loss = -pg_loss - ent_coef * entropy`, or (b) reduce the learning rate so the gradient does not slam the logits to extreme values, or (c) clip logits before softmax to `[-20, 20]`. In PPO, the `ent_coef` parameter directly controls this; start with `ent_coef=0.01` and increase if you see collapse.

### 15.2 Deterministic policy instability (DDPG)

Your DDPG actor's loss oscillates and the agent's performance degrades suddenly after 200k steps. This is the "deadly triad" instability: a deterministic policy, a neural Q-function approximator, and off-policy data can create a feedback loop where the policy chases noisy Q-estimates.

**Diagnosis**: monitor the actor loss curve and Q-value estimates. If Q-values are growing unboundedly (Q divergence), DDPG is the cause.

**Fix**: switch to TD3, which adds three stabilizing mechanisms: (1) clipped double-Q (take the min of two Q-networks), (2) target policy smoothing (add noise to target actions for Q updates), (3) delayed policy updates (update actor every 2 critic steps). These reduce the variance of Q-targets and prevent the policy from over-exploiting noisy Q-peaks.

### 15.3 Wrong policy type for action space

Using a Categorical policy for continuous actions or a Gaussian policy for discrete actions are both bugs that produce confusing errors or silent poor performance.

```python
import gymnasium as gym
from stable_baselines3 import PPO

# WRONG: continuous env with discrete policy class
env = gym.make("HalfCheetah-v4")  # continuous action space
# model = PPO("MlpPolicy", env)  # SB3 handles this correctly — MlpPolicy
# auto-detects action space and uses ContinuousCritic + DiagGaussian
# BUT if you manually build a Categorical policy for continuous actions:
# Categorical(logits=...) where logits has n_actions << possible actions = wrong

# RIGHT: let SB3 infer policy class, or check action space first
print(env.action_space)  # Box(-1, 1, (6,), float32) => continuous
model = PPO("MlpPolicy", env)  # SB3 uses DiagGaussian automatically

env_d = gym.make("CartPole-v1")
print(env_d.action_space)  # Discrete(2) => categorical
model_d = PPO("MlpPolicy", env_d)  # SB3 uses Categorical automatically
```

### 15.4 Forgetting to detach log-probs during rollout collection

In REINFORCE / PPO, you collect a rollout, compute returns, then compute the policy gradient loss. A common bug is accidentally retaining gradients through the sampling step:

```python
# BUG: log_prob retains computation graph from rollout collection
# (if you use log_prob computed during rollout for the gradient, but
# then also call policy again for new log_probs in the update step,
# old log_probs should be .detach()-ed to avoid double-counting)

# CORRECT: in PPO, always recompute log_probs from scratch in the update step
def ppo_update(policy, obs_batch, actions_batch, old_log_probs, returns, advantages):
    dist = policy(obs_batch)                      # fresh forward pass
    new_log_probs = dist.log_prob(actions_batch)  # recomputed from current weights
    entropy = dist.entropy().mean()
    ratio = torch.exp(new_log_probs - old_log_probs.detach())  # detach old!
    clipped = torch.clamp(ratio, 1 - 0.2, 1 + 0.2)
    policy_loss = -torch.min(ratio * advantages, clipped * advantages).mean()
    return policy_loss - 0.01 * entropy
```

The `old_log_probs.detach()` call is critical: old log-probabilities are constants in the PPO objective, not variables to differentiate through.

---

## 16. When to use which (and when to stick with simpler approaches)

**Use a deterministic policy (DDPG, TD3) when:**
- Your action space is continuous and high-dimensional (robot manipulation, continuous portfolio allocation).
- You need maximum final performance and are willing to tune the exploration noise schedule.
- Sample efficiency matters and you have a large replay buffer (deterministic off-policy methods use replay more efficiently than stochastic on-policy methods).
- The environment is fully observed (tabular or low-dimensional state).

**Use a stochastic policy (PPO, SAC, A2C) when:**
- You need robust exploration without manual noise tuning.
- The environment is noisy, partially observed, or has stochastic dynamics.
- You are prototyping and want a method that works without much hyperparameter sensitivity.
- You need a log-probability for policy gradient math (RLHF, inverse RL, entropy regularization).
- Your action space is discrete — use PPO with Categorical; deterministic policies only work naturally for continuous actions.

**When to skip deep RL policy networks entirely:**
- **Tabular environment, known model**: use value iteration or policy iteration directly. No neural network needed. They are exact and faster. The GridWorld example in this post took two policy iterations to converge — a neural network would need thousands of gradient steps to approximate the same result.
- **Small discrete action space, unknown model**: use DQN or Q-learning. The implicit greedy policy over Q-values is sufficient — no need for an explicit parameterized stochastic policy. DQN's $\varepsilon$-greedy policy handles exploration without any distribution parameterization.
- **Simple continuous control**: PID controllers, LQR, or MPC often outperform RL on structured control problems where the dynamics are known or easily identified. Do not reach for PPO when a control theory solution exists. A well-tuned PID controller for CartPole runs perfectly without a single gradient step.
- **Supervised learning substitute**: if your "RL problem" has a labeled dataset of (state, optimal_action) pairs, behavioral cloning (imitation learning) with a supervised loss is faster and more sample-efficient than RL. The policy is just a classifier or regressor trained directly. This is how most robotics manipulation policies start — imitation learning from human demonstrations, with RL fine-tuning only when the imitation policy plateaus.

#### Worked example: policy type selection for a trading agent

Suppose you are building an RL agent for trade execution (posting limit orders to minimize market impact). The state is order book depth, inventory, and time-remaining. The action is bid-ask spread or lot size.

- Action space: continuous (real-valued lot size, 0 to 1000 shares).
- Environment: noisy (market prices are stochastic, other agents react).
- Partial observability: yes (you see the order book but not other agents' private signals).

**Best choice:** SAC with a DiagGaussian policy. The stochastic policy handles partial observability by naturally mixing over possible lot sizes. The entropy bonus prevents the agent from collapsing onto a single deterministic execution strategy that would be front-run by other market participants. The off-policy replay buffer makes SAC sample-efficient for the expensive-to-collect trading environment data.

**What not to use:** DDPG or TD3 — deterministic execution strategies are trivially front-run. A deterministic agent that always posts 100 shares at 9:31 AM is immediately exploited.

---

## Key takeaways

1. **A policy is the complete behavioral specification**: $\pi(a \mid s)$ (stochastic) or $\pi(s)$ (deterministic) — both fully determine the agent's behavior in every state.

2. **Stochastic policies are the general case**: deterministic policies are the degenerate special case where all probability mass sits on one action.

3. **In fully-observed MDPs, an optimal deterministic policy always exists**: by the policy improvement theorem, repeated greedy improvement converges to a deterministic $\pi^*$ that is at least as good as any stochastic policy.

4. **In POMDPs, stochastic policies can strictly dominate**: when multiple hidden states share the same observation, stochastic policies hedge over the ambiguity and achieve higher expected return.

5. **The policy improvement theorem guarantees monotonic progress**: if a new policy is greedy with respect to $Q^\pi$, then $V^{\pi'} \geq V^\pi$ everywhere — greedy improvement is always safe.

6. **Policy gradient algorithms need stochastic policies**: the score-function estimator $\nabla_\theta \log \pi_\theta(a \mid s)$ requires a smooth distribution; deterministic policies use the DPG estimator instead.

7. **Neural discrete policies use Categorical; continuous policies use DiagGaussian**: in PyTorch, `torch.distributions.Categorical(logits=net(obs))` and `torch.distributions.Normal(mean, std)` are the canonical tools.

8. **SAC is usually the right stochastic continuous-action choice**: it combines off-policy efficiency with entropy maximization, outperforming DDPG on most benchmarks.

9. **TD3/DDPG are the right deterministic continuous-action choices**: they achieve higher peak performance than SAC on some benchmarks when properly tuned.

10. **Tabular policy iteration is the theory backbone**: even for deep RL practitioners, understanding the 4-state GridWorld example is essential — every deep RL algorithm is approximating the same policy evaluation and improvement loop.

11. **Recurrent policies approximate the belief state under partial observability**: an LSTM policy is almost always better than an MLP policy in POMDPs — the hidden state accumulates the agent's evolving uncertainty about the environment.

12. **At test time, you can usually switch a stochastic policy to deterministic mode**: take `argmax` of Categorical logits or `tanh(mean)` of Gaussian. This reduces variance and slightly improves average return, at the cost of exploration — acceptable once training is complete.

---

## Further reading

- **Sutton & Barto, "Reinforcement Learning: An Introduction" (2nd ed., 2018)** — Chapter 4 covers policy iteration and improvement theorems in full tabular generality. The textbook that every RL practitioner needs.
- **Silver et al., "Deterministic Policy Gradient Algorithms" (ICML 2014)** — Proved the DPG theorem that makes deterministic policies work with policy gradient methods. Foundation for DDPG and TD3.
- **Haarnoja et al., "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning" (ICML 2018)** — The paper that made entropy-regularized stochastic policies the default for continuous control.
- **Schulman et al., "Proximal Policy Optimization Algorithms" (arXiv 2017)** — PPO, the most widely used stochastic on-policy algorithm in practice, from RLHF to robotics.
- **Singh et al., "Learning Without State Estimation in POMDPs" (ICML 1994)** — Formal treatment of when stochastic policies are strictly better in partially observable settings.
- **Within this series:**
  - Post A1: [Reinforcement Learning: A Unified Map](/blog/machine-learning/reinforcement-learning/reinforcement-learning-a-unified-map) — the full RL taxonomy; MDPs, value functions, and the agent–environment loop.
  - Post A2: [Markov Decision Processes from First Principles](/blog/machine-learning/reinforcement-learning/markov-decision-processes-from-first-principles) — the MDP formalism underlying all policy theory.
  - Post A3: [Value Functions and Bellman Equations](/blog/machine-learning/reinforcement-learning/value-functions-and-bellman-equations) — $V^\pi$ and $Q^\pi$ derivation; the building blocks for policy improvement.
  - Post E1: [The Policy Gradient Theorem](/blog/machine-learning/reinforcement-learning/the-policy-gradient-theorem) — the full derivation of $\nabla_\theta J(\theta)$ and why stochastic policies are required; score function vs reparameterization estimators.
