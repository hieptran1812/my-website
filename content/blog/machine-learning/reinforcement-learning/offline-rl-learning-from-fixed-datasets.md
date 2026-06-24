---
title: "Offline RL: Learning from Fixed Datasets Without Environment Access"
date: "2026-06-22"
publishDate: "2026-06-22"
description: "Master offline reinforcement learning — BCQ, CQL, IQL, and TD3+BC — and understand why distributional shift causes standard Q-learning to fail catastrophically on fixed datasets."
tags:
  [
    "reinforcement-learning",
    "deep-learning",
    "offline-rl",
    "q-learning",
    "conservative-q-learning",
    "batch-rl",
    "d4rl",
    "machine-learning",
    "pytorch",
    "value-based-rl",
  ]
category: "machine-learning"
subcategory: "Reinforcement Learning"
author: "Hiep Tran"
featured: true
readTime: 55
image: "/imgs/blogs/offline-rl-learning-from-fixed-datasets-1.png"
---

Imagine you are a robotics engineer at a hospital tasked with training a surgical assistant robot. You have two years of recordings from expert surgeons — tens of thousands of hours of state observations, actions taken, and patient outcomes. What you do not have is any tolerance for letting a half-trained robot make mistakes on real patients while it figures out the right policy. The robot cannot practice in a live operating room. It cannot reset the patient after a bad action. It must learn entirely from that fixed archive of past demonstrations, and when it is deployed, it must be good.

This is the offline RL problem, and it breaks every standard assumption that makes deep RL work. Standard Q-learning relies on the ability to query the environment to correct its estimates — to discover that a seemingly attractive action leads somewhere bad. In offline RL you have no such ability. You have a dataset $\mathcal{D} = \{(s, a, r, s')\}$ collected by some behavior policy $\mu$ that you cannot query further. The agent must extract a policy purely from that static archive.

The catch is devastating: standard Q-learning trained on a fixed dataset will confidently learn to select actions that were never taken in $\mathcal{D}$. These out-of-distribution (OOD) actions receive arbitrarily high Q-values because no Bellman backup ever corrected them with real environment data. The greedy policy then exploits these inflated estimates, collapsing to near-zero return when deployed. This is distributional shift — and it is the central problem of offline RL.

By the end of this post you will understand exactly why this happens (with the math), how BCQ, CQL, IQL, and TD3+BC each solve it differently, how to implement CQL from scratch in PyTorch, what D4RL benchmarks tell us about real-world tradeoffs, and when each algorithm wins. Figure 1 below shows the architectural split: online RL queries the environment freely and grows its replay buffer, while offline RL must work entirely within the frozen dataset $\mathcal{D}$.

![Offline RL learns from a fixed dataset while online RL continuously grows its replay buffer through environment interaction](/imgs/blogs/offline-rl-learning-from-fixed-datasets-1.png)

## 1. The Offline RL Problem: A Formal Setup

In standard reinforcement learning we have a Markov Decision Process $(\mathcal{S}, \mathcal{A}, P, R, \gamma)$ where $\mathcal{S}$ is the state space, $\mathcal{A}$ the action space, $P(s'|s,a)$ the transition dynamics, $R(s,a)$ the reward function, and $\gamma \in [0,1)$ a discount factor. The agent's goal is to find a policy $\pi: \mathcal{S} \to \Delta(\mathcal{A})$ that maximizes expected discounted return:

$$J(\pi) = \mathbb{E}_{\tau \sim \pi}\left[\sum_{t=0}^{\infty} \gamma^t r_t\right]$$

In the offline setting, the agent has access only to a static dataset:

$$\mathcal{D} = \{(s_i, a_i, r_i, s'_i)\}_{i=1}^N$$

collected by some behavior policy $\mu$ (which may be a human expert, a scripted heuristic, a random policy, or a mixture of all three). The critical constraint is:

**The agent cannot interact with the environment at training time.** No new transitions can be collected. $\mathcal{D}$ is the entire information budget.

This constraint matters because the Bellman optimality operator, which is the engine behind every Q-learning variant, assumes you can query transitions for any state-action pair. When you apply it to a fixed dataset, you can only update Q-values for $(s,a)$ pairs that appear in $\mathcal{D}$. But the next step in Q-learning requires evaluating $\max_{a'} Q(s', a')$ — and the maximizing $a'$ may be far outside the distribution of actions seen in $\mathcal{D}$.

### The Distributional Shift Decomposition

Let $d^\mu(s,a)$ denote the state-action occupancy distribution under $\mu$, and $d^\pi(s,a)$ denote the occupancy under the learned policy $\pi$. Offline RL requires that $\pi$ performs well, but Q-values are only accurate under $d^\mu$. When $d^\pi$ diverges from $d^\mu$, all bets are off.

Formally, the policy value error satisfies:

$$|J(\pi) - \hat{J}(\pi)| \leq \frac{C \cdot \gamma}{(1-\gamma)^2} \cdot \mathbb{E}_{d^\pi}\left[\left|\frac{d^\pi(s,a)}{d^\mu(s,a)} - 1\right|\right]$$

where $C$ is a constant that scales with the Q-value magnitude. The second term is the total variation between $d^\pi$ and $d^\mu$ — weighted by how far the learned policy strays from the behavior policy's distribution. When $\pi$ selects actions that $\mu$ never took, this divergence is unbounded.

This is not a loose bound. In practice, Q-networks with millions of parameters will confidently extrapolate, assigning high values to regions of action space where they have zero data. The policy, being greedy, walks straight into those regions and falls off a cliff.

### Why Naive Q-Learning Fails: A Concrete Walkthrough

Consider training DQN on a fixed dataset from CartPole. The buffer contains transitions where the pole was balanced by an expert (always pushing left when the pole leans left, etc.). Now during training, the Q-network gets a gradient signal to set $Q(s, a_{data})$ close to the TD target. But what happens when the network updates $\max_{a'} Q(s', a')$ and $a'$ is some unseen action?

The network will initialize $Q(s', a'_{ood})$ from random weights — perhaps it happens to be 15.3. Since $Q(s', a'_{ood}) = 15.3 > Q(s', a_{data}) = 2.1$, the TD target becomes $r + \gamma \cdot 15.3$. Now $Q(s, a_{data})$ is trained to be higher than before, so the error is propagated backwards. Over many gradient steps, Q-values for in-distribution actions inflate to track the inflated OOD bootstrap. When you finally extract a greedy policy, it selects actions that look good on paper but fail catastrophically in deployment.

This is the **deadly triad** in offline RL: function approximation (neural net) + bootstrapping (TD update) + off-policy data (fixed $\mathcal{D}$) combine to produce divergence. You can have any two, but all three together create a positive feedback loop of overestimation.

## 2. Behavior Cloning: The Obvious Baseline

Before diving into the sophisticated offline RL methods, it is essential to understand what pure behavior cloning (BC) gives you — because on some datasets, it beats everything else.

Behavior cloning ignores rewards entirely and just learns to imitate the behavior policy $\mu$:

$$\mathcal{L}_{BC}(\theta) = -\mathbb{E}_{(s,a) \sim \mathcal{D}}\left[\log \pi_\theta(a|s)\right]$$

For continuous actions this becomes mean-squared error: $\mathcal{L}_{BC} = \mathbb{E}[(a - \pi_\theta(s))^2]$.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class BehaviorCloningPolicy(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),  # assumes actions in [-1, 1]
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


def train_bc(states, actions, epochs=100, batch_size=256, lr=3e-4):
    """Train a behavior cloning policy on (state, action) pairs."""
    dataset = TensorDataset(
        torch.FloatTensor(states),
        torch.FloatTensor(actions)
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    state_dim = states.shape[-1]
    action_dim = actions.shape[-1]

    policy = BehaviorCloningPolicy(state_dim, action_dim)
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        total_loss = 0.0
        for state_batch, action_batch in loader:
            pred_actions = policy(state_batch)
            loss = loss_fn(pred_actions, action_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}: BC loss = {total_loss/len(loader):.4f}")
    return policy
```

BC is remarkably competitive when $\mathcal{D}$ consists of expert demonstrations (the `hopper-expert-v2` task in D4RL scores around 29–50 for vanilla BC, versus 86+ for CQL). But when $\mathcal{D}$ is mixed-quality — say, 50% random and 50% medium-skill data — BC just averages over all that mediocre behavior and learns a mediocre policy. The reward signal is completely unused.

The fundamental limitation: BC cannot improve beyond the behavior policy. If your data was collected by a policy that scored 40 on a task scaled to 100, your BC policy will also score around 40, never higher.

## 3. The OOD Action Problem in Depth

The OOD overestimation problem is so central to offline RL that it deserves its own section with a rigorous treatment. Figure 2 shows the layered structure of action space, from safe in-distribution actions out to the catastrophic OOD region.

![Layers of OOD action risk showing how moving from data support through boundary actions to fully OOD actions leads to Q overestimation and policy collapse](/imgs/blogs/offline-rl-learning-from-fixed-datasets-2.png)

### The Extrapolation Error Accumulates

Consider a Q-function parameterized by $\theta$. The Bellman backup for a transition $(s, a, r, s')$ is:

$$y = r + \gamma \cdot \max_{a'} Q_\theta(s', a')$$

When $Q_\theta$ is a neural network trained on $\mathcal{D}$, its values at out-of-distribution $(s', a')$ pairs are whatever the network's inductive biases and random initialization happen to produce. The maximization over all $a'$ actively seeks out the highest of these arbitrary values.

Formally, for any neural network $Q_\theta$ and a compact action space $\mathcal{A}$:

$$\max_{a' \in \mathcal{A}} Q_\theta(s', a') \geq Q_\theta(s', a'_{data})$$

with equality only if the network happens to assign the highest value to an in-distribution action. In high-dimensional continuous action spaces, this is extremely unlikely because the probability mass that $\mathcal{D}$ covers approaches zero relative to the full action space volume.

The error compounds over Bellman backups. If at step $k$ the maximum extrapolation error is $\epsilon_k$, then at step $k+1$ the bootstrap target carries error $\gamma \epsilon_k + \delta$ where $\delta$ is the new extrapolation error introduced. Over $T$ gradient steps this accumulates to order $O(T \cdot \gamma^T \cdot \epsilon_{max})$, which can be large for long-horizon tasks and small discount gaps.

### Why the Policy Amplifies the Problem

The policy $\pi_\theta = \text{argmax}_a Q_\theta(s,a)$ is derived directly from the Q-function. Once Q-values for OOD actions are overestimated, the policy actively seeks them out. This creates a feedback loop: the policy selects OOD actions → TD targets use those inflated values → Q-values at in-distribution actions inflate to match → the policy is pushed even further OOD → repeat.

This is sometimes called the "deadly positive feedback" of offline RL, and it explains why naively applying DQN or SAC to a fixed dataset fails — not just a little, but catastrophically. I have seen this in practice: a SAC agent trained offline on a locomotion dataset would learn to select an action combination that looked like it would produce reward 200, but the actual return when deployed was negative (the agent immediately fell over because it had never executed that action combination in training).

## 3.5 The Compounding Error Problem: Why More Steps Makes Things Worse

One counterintuitive fact about offline Q-learning is that training longer can make things worse, not better. In online RL, more gradient steps generally improve the policy because the agent collects new data correcting any errors. In offline RL, there is no such correction mechanism — every gradient step has the potential to compound extrapolation errors.

To understand why, consider what happens at iteration $k$ of offline Q-learning. The Bellman target is:

$$y_k = r + \gamma \cdot \max_{a'} Q_{k-1}(s', a')$$

The key insight is that $Q_{k-1}(s', a')$ for OOD actions $a'$ is not the true Q-value but an estimate that may be arbitrarily high. At iteration $k+1$, we train $Q_k$ to match $y_k$. But $y_k$ was itself corrupted by the OOD overestimation in $Q_{k-1}$. So $Q_k$ is now corrupted at every $(s,a)$ pair that was bootstrapped through an OOD maximum — which, through Bellman backups, eventually propagates to all state-action pairs in $\mathcal{D}$.

This is analogous to a telephone game: each round of Bellman backup propagates and amplifies earlier errors. The effect is particularly severe for:

1. **High discount factors** ($\gamma$ close to 1): More Bellman steps are needed to propagate value, meaning more error accumulation opportunities.
2. **Sparse rewards**: Most transitions have $r = 0$, so the TD target is dominated by $\gamma \max_{a'} Q(s', a')$, amplifying OOD errors.
3. **High-dimensional action spaces**: More dimensions mean more OOD directions to exploit, making $\max_{a'} Q(s', a')$ higher on average.

This is why early stopping is important in offline RL training — not just for computational efficiency, but because training too long can degrade performance. The optimal stopping point is when the Q-function has learned the in-distribution value landscape but before OOD errors have propagated through too many Bellman steps.

### The Role of the Target Network

One important practical detail: the target network (used to compute TD targets) should be updated slowly in offline RL. The standard $\tau = 0.005$ soft update rate works, but some practitioners use even slower rates ($\tau = 0.001$) to further slow error propagation.

The reasoning: with a slow target network, the TD target $y = r + \gamma Q_{\bar{\theta}}(s', a^*)$ changes slowly. This prevents a runaway positive feedback loop where the online Q-network chases a target that is itself being driven up by OOD estimates from the online Q-network. The target network provides a stable anchor.

## 4. BCQ: The First Principled Offline RL Algorithm

Batch-Constrained deep Q-learning (BCQ, Fujimoto et al. 2019) was the first algorithm to explicitly address the OOD problem in a principled way. The insight is simple: instead of allowing the policy to optimize over all actions, constrain it to the support of the behavior policy estimated from $\mathcal{D}$.

### Learning the Behavior Support with a VAE

BCQ trains a variational autoencoder (VAE) to learn the distribution of actions in $\mathcal{D}$ conditioned on the state:

$$G_\omega(s) \approx \mu(a|s)$$

At policy evaluation time, instead of $a^* = \text{argmax}_a Q(s,a)$, BCQ does:

$$a^* = \text{argmax}_{a_i + \xi_\phi(s, a_i)} Q(s, a_i + \xi_\phi(s, a_i)), \quad a_i \sim G_\omega(s)$$

where $\xi_\phi$ is a small perturbation network bounded by $[-\Phi, \Phi]$ (typically $\Phi = 0.05$). The VAE generates $n$ candidate actions from the data distribution; the perturbation network nudges them slightly; the Q-function picks the best. This ensures you never evaluate Q at truly OOD actions.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class VAEDecoder(nn.Module):
    """VAE for learning behavior policy support in BCQ."""
    def __init__(self, state_dim, action_dim, latent_dim, max_action=1.0):
        super().__init__()
        self.max_action = max_action
        self.encoder = nn.Sequential(
            nn.Linear(state_dim + action_dim, 750),
            nn.ReLU(),
            nn.Linear(750, 750),
            nn.ReLU(),
        )
        self.mean = nn.Linear(750, latent_dim)
        self.log_std = nn.Linear(750, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(state_dim + latent_dim, 750),
            nn.ReLU(),
            nn.Linear(750, 750),
            nn.ReLU(),
            nn.Linear(750, action_dim),
            nn.Tanh(),
        )

    def encode(self, state, action):
        z = self.encoder(torch.cat([state, action], dim=-1))
        mean = self.mean(z)
        log_std = self.log_std(z).clamp(-4, 15)
        return mean, log_std

    def decode(self, state, z=None):
        if z is None:
            z = torch.randn(state.size(0), self.mean.out_features).to(state.device)
            z = z.clamp(-0.5, 0.5)
        return self.max_action * self.decoder(torch.cat([state, z], dim=-1))

    def forward(self, state, action):
        mean, log_std = self.encode(state, action)
        std = log_std.exp()
        z = mean + std * torch.randn_like(std)
        recon = self.decode(state, z)
        return recon, mean, std
```

BCQ demonstrated that constraining policy to the data support dramatically reduces OOD overestimation. On D4RL `hopper-medium-expert-v2` it achieves approximately 69, compared to vanilla TD3 offline which achieves near 0.

#### Worked example: BCQ on Hopper-Medium-Expert

Dataset: `hopper-medium-expert-v2` from D4RL (1M transitions from a mix of a suboptimal and expert policy; maximum reward per episode approximately 3500).

| Method | Normalized Score | Notes |
|---|---|---|
| BC | 52.5 | Imitates the average behavior |
| BCQ | 69.2 | Constrained to data support |
| Random agent | ~0 | Baseline |

Normalized score uses $\frac{score - random\_score}{expert\_score - random\_score} \times 100$. BCQ's gain over BC comes from using the reward signal to select among actions within the data support, rather than averaging over all of them.

## 5. CQL: Conservative Q-Learning

Conservative Q-Learning (CQL, Kumar et al. 2020) takes a different approach: instead of constraining the policy's action space, it directly penalizes the Q-function to be conservative — low for OOD actions and accurate for in-distribution actions.

The key insight is to add a regularizer to the standard Bellman loss that pushes down Q-values for all actions and pulls up Q-values for actions in the dataset:

$$\mathcal{L}_{CQL}(\theta) = \underbrace{\alpha \cdot \left(\mathbb{E}_{s \sim \mathcal{D}}\left[\log \sum_{a} e^{Q_\theta(s,a)}\right] - \mathbb{E}_{(s,a) \sim \mathcal{D}}\left[Q_\theta(s,a)\right]\right)}_{\text{CQL penalty}} + \underbrace{\frac{1}{2}\mathbb{E}_{(s,a,s') \sim \mathcal{D}}\left[\left(Q_\theta(s,a) - \mathcal{B}^\pi Q_{\bar{\theta}}(s,a)\right)^2\right]}_{\text{standard Bellman loss}}$$

where $\mathcal{B}^\pi$ is the Bellman operator and $\bar{\theta}$ is the target network. The $\log \sum_a e^{Q(s,a)}$ term is a soft maximum (logsumexp) — it represents the upper bound on the Q-value across all actions. Pulling this down while pulling up the data-action Q-values achieves conservatism without preventing the policy from selecting good in-distribution actions.

Figure 3 shows the contrast: standard Q-learning leaves OOD Q-values inflated, while CQL suppresses them.

![Standard Q-learning before and after CQL regularization showing OOD Q-values overestimated without CQL and suppressed with CQL penalty](/imgs/blogs/offline-rl-learning-from-fixed-datasets-3.png)

### CQL Theoretical Guarantee

Kumar et al. prove that CQL provides a lower bound on the true policy value. Specifically, for the CQL-learned Q-function $\hat{Q}^\pi$:

$$\hat{V}^\pi(s) \leq V^\pi(s) \quad \text{for all } s$$

This means the policy $\pi_{CQL}$ that is greedy with respect to $\hat{Q}^\pi$ is guaranteed not to be worse than the behavior policy (up to a slack term that decreases with dataset size). This is a **provable safety guarantee** that vanilla Q-learning cannot offer in the offline setting.

The guarantee comes from the fact that the logsumexp term upper-bounds all Q-values, so after regularization, $Q_\theta(s,a) \leq Q^\pi(s,a)$ for all $(s,a)$ — the learned Q-function systematically underestimates rather than overestimates.

### Implementing CQL in PyTorch

Figure 4 shows the CQL training pipeline. Let me now implement it in full.

![CQL training pipeline showing the path from batch sampling through TD loss computation and CQL penalty computation to gradient update](/imgs/blogs/offline-rl-learning-from-fixed-datasets-4.png)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import namedtuple

Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'done'])


class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([state, action], dim=-1))


class PolicyNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256, max_action: float = 1.0):
        super().__init__()
        self.max_action = max_action
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.max_action * self.net(state)


class CQL:
    """
    Conservative Q-Learning (CQL) for offline RL.
    Kumar et al. (2020) — NeurIPS 2020.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_action: float = 1.0,
        discount: float = 0.99,
        tau: float = 0.005,         # soft target network update rate
        policy_lr: float = 3e-4,
        q_lr: float = 3e-4,
        cql_alpha: float = 1.0,     # CQL regularization strength
        num_random_actions: int = 10,  # actions sampled for logsumexp
    ):
        self.discount = discount
        self.tau = tau
        self.cql_alpha = cql_alpha
        self.num_random_actions = num_random_actions
        self.max_action = max_action

        # Two Q-networks for double-Q (reduces overestimation)
        self.q1 = QNetwork(state_dim, action_dim)
        self.q2 = QNetwork(state_dim, action_dim)
        self.q1_target = QNetwork(state_dim, action_dim)
        self.q2_target = QNetwork(state_dim, action_dim)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        self.policy = PolicyNetwork(state_dim, action_dim, max_action=max_action)

        self.q_optimizer = optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=q_lr
        )
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=policy_lr)
        self.state_dim = state_dim
        self.action_dim = action_dim

    def _cql_penalty(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Compute CQL penalty:
            logsumexp_a Q(s, a) - Q(s, a_data)

        The logsumexp approximates the soft-max over all actions.
        We approximate it by sampling num_random_actions uniformly at random.
        """
        batch_size = states.size(0)

        # Sample random actions uniformly from action space
        random_actions = torch.FloatTensor(
            batch_size * self.num_random_actions, self.action_dim
        ).uniform_(-self.max_action, self.max_action)

        # Repeat states to match random actions
        # Shape: (batch_size * num_random, state_dim)
        repeated_states = states.unsqueeze(1).repeat(
            1, self.num_random_actions, 1
        ).view(batch_size * self.num_random_actions, -1)

        # Q-values for random actions: (batch_size * num_random, 1)
        q1_rand = self.q1(repeated_states, random_actions)
        q2_rand = self.q2(repeated_states, random_actions)

        # Reshape to (batch_size, num_random)
        q1_rand = q1_rand.view(batch_size, self.num_random_actions)
        q2_rand = q2_rand.view(batch_size, self.num_random_actions)

        # logsumexp over random actions (soft max, scaled by num_random)
        # Subtract log(num_random) to correct for uniform density
        log_n = np.log(self.num_random_actions)
        q1_logsumexp = torch.logsumexp(q1_rand, dim=1, keepdim=True) - log_n
        q2_logsumexp = torch.logsumexp(q2_rand, dim=1, keepdim=True) - log_n

        # Q-values for dataset actions (in-distribution)
        q1_data = self.q1(states, actions)
        q2_data = self.q2(states, actions)

        # CQL penalty = logsumexp - Q(s, a_data)
        cql1 = (q1_logsumexp - q1_data).mean()
        cql2 = (q2_logsumexp - q2_data).mean()
        return cql1 + cql2

    def train_step(self, batch: list) -> dict:
        states, actions, rewards, next_states, dones = [
            torch.FloatTensor(np.array(x)) for x in zip(*batch)
        ]

        with torch.no_grad():
            # Target policy for next state
            next_actions = self.policy(next_states)
            # Add target network smoothing noise
            noise = torch.randn_like(next_actions) * 0.2
            next_actions = (next_actions + noise).clamp(-self.max_action, self.max_action)

            # Double-Q target: take minimum to reduce overestimation
            q1_next = self.q1_target(next_states, next_actions)
            q2_next = self.q2_target(next_states, next_actions)
            q_next = torch.min(q1_next, q2_next)

            # TD target
            not_done = (1.0 - dones.unsqueeze(1))
            td_target = rewards.unsqueeze(1) + self.discount * not_done * q_next

        # Standard Bellman (TD) loss
        q1_pred = self.q1(states, actions)
        q2_pred = self.q2(states, actions)
        bellman_loss = F.mse_loss(q1_pred, td_target) + F.mse_loss(q2_pred, td_target)

        # CQL conservative penalty
        cql_loss = self._cql_penalty(states, actions)

        # Total Q-network loss
        q_loss = bellman_loss + self.cql_alpha * cql_loss

        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        # Policy update: maximize Q1(s, π(s))
        policy_actions = self.policy(states)
        policy_loss = -self.q1(states, policy_actions).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # Soft update target networks
        for param, target_param in zip(self.q1.parameters(), self.q1_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.q2.parameters(), self.q2_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return {
            "bellman_loss": bellman_loss.item(),
            "cql_loss": cql_loss.item(),
            "q_loss": q_loss.item(),
            "policy_loss": policy_loss.item(),
        }

    def select_action(self, state: np.ndarray) -> np.ndarray:
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            return self.policy(state).squeeze(0).numpy()
```

The `_cql_penalty` method is the heart of CQL. It samples `num_random_actions` uniformly from the action space, computes the logsumexp, and subtracts the Q-value of the actual data action. The $\alpha$ hyperparameter controls how conservative to be: higher $\alpha$ suppresses OOD Q-values more aggressively but may also push down in-distribution Q-values if $\alpha$ is too large, hurting performance.

### The $\alpha$ Hyperparameter: A Critical Trade-off

| $\alpha$ | Effect | Risk |
|---|---|---|
| 0.0 | Standard SAC/TD3 offline | OOD overestimation, policy collapse |
| 0.1 | Mild conservatism | May not fully prevent OOD issues |
| 1.0 | Moderate (default) | Good balance for mixed datasets |
| 5.0 | Aggressive conservatism | May underestimate all Q-values |
| 10.0 | Very aggressive | Policy converges to BC-like behavior |

The D4RL paper suggests $\alpha = 1.0$ for most tasks, with tuning toward higher values for narrow datasets (expert-only) and lower values for broad datasets (random mixture).

## 6. IQL: Implicit Q-Learning

Implicit Q-Learning (IQL, Kostrikov et al. 2021) takes the most elegant approach to offline RL. The core insight: CQL and BCQ both require evaluating Q-values at OOD actions (even to penalize them, you need to compute them). IQL avoids this entirely using expectile regression.

The key observation is that the Bellman optimality equation requires:

$$Q(s,a) = r + \gamma \cdot \max_{a'} Q(s', a')$$

The maximization over $a'$ is what forces evaluation at OOD actions. IQL replaces this with a **value function** $V(s)$ that approximates the maximum:

$$V(s) \approx \max_a Q(s,a)$$

but learned using expectile regression — a one-sided loss that only fits the upper quantile of the Q-distribution:

$$\mathcal{L}_{IQL}^V = \mathbb{E}_{(s,a) \sim \mathcal{D}}\left[L_2^\tau(Q(s,a) - V(s))\right]$$

where $L_2^\tau(u) = |\tau - \mathbf{1}[u < 0]| \cdot u^2$ is the expectile loss with $\tau \in (0,1)$. At $\tau = 0.5$ this is standard MSE; at $\tau \to 1$ it approximates the maximum.

This is brilliant because:
1. $V(s)$ is trained on data actions only — no OOD evaluation at all.
2. The Q-function target becomes $r + \gamma V(s')$ — still no OOD evaluation.
3. The policy is extracted by weighting behavioral cloning by $\exp(\beta (Q(s,a) - V(s)))$ — an advantage-weighted regression that emphasizes high-advantage transitions.

```python
class IQL:
    """
    Implicit Q-Learning (IQL) for offline RL.
    Kostrikov et al. (2021) — ICLR 2022.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        expectile: float = 0.7,   # τ: how much to emphasize high-Q transitions
        temperature: float = 3.0, # β: sharpness of advantage weighting
        discount: float = 0.99,
        tau: float = 0.005,
        lr: float = 3e-4,
        max_action: float = 1.0,
    ):
        self.expectile = expectile
        self.temperature = temperature
        self.discount = discount
        self.tau = tau
        self.max_action = max_action

        self.q1 = QNetwork(state_dim, action_dim)
        self.q2 = QNetwork(state_dim, action_dim)
        self.q1_target = QNetwork(state_dim, action_dim)
        self.q2_target = QNetwork(state_dim, action_dim)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        # Value network V(s)
        self.value_net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1)
        )
        self.policy = PolicyNetwork(state_dim, action_dim, max_action=max_action)

        self.q_optimizer = optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=lr
        )
        self.v_optimizer = optim.Adam(self.value_net.parameters(), lr=lr)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr)

    def expectile_loss(self, diff: torch.Tensor) -> torch.Tensor:
        """L2 expectile loss: asymmetric MSE that fits upper quantile."""
        weight = torch.where(diff >= 0, self.expectile, 1.0 - self.expectile)
        return (weight * diff.pow(2)).mean()

    def train_step(self, batch: list) -> dict:
        states, actions, rewards, next_states, dones = [
            torch.FloatTensor(np.array(x)) for x in zip(*batch)
        ]
        not_done = (1.0 - dones).unsqueeze(1)

        with torch.no_grad():
            # TD target using V(s') — no OOD action evaluation!
            v_next = self.value_net(next_states)
            q_target = rewards.unsqueeze(1) + self.discount * not_done * v_next

            # For value learning: use min of target Q-networks
            q1_t = self.q1_target(states, actions)
            q2_t = self.q2_target(states, actions)
            q_min = torch.min(q1_t, q2_t)

        # Update Q-networks with standard Bellman loss
        q1_loss = F.mse_loss(self.q1(states, actions), q_target)
        q2_loss = F.mse_loss(self.q2(states, actions), q_target)
        q_loss = q1_loss + q2_loss
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        # Update value network with expectile loss
        v_pred = self.value_net(states)
        v_loss = self.expectile_loss(q_min.detach() - v_pred)
        self.v_optimizer.zero_grad()
        v_loss.backward()
        self.v_optimizer.step()

        # Policy update: advantage-weighted behavioral cloning
        with torch.no_grad():
            adv = q_min - self.value_net(states)
            # Exponential advantage weighting, clipped for stability
            exp_adv = torch.exp(self.temperature * adv).clamp(max=100.0)

        pred_actions = self.policy(states)
        bc_loss = F.mse_loss(pred_actions, actions, reduction='none').sum(-1, keepdim=True)
        policy_loss = (exp_adv * bc_loss).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # Soft target updates
        for q, qt in [(self.q1, self.q1_target), (self.q2, self.q2_target)]:
            for p, tp in zip(q.parameters(), qt.parameters()):
                tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

        return {"q_loss": q_loss.item(), "v_loss": v_loss.item(), "policy_loss": policy_loss.item()}
```

The `expectile` parameter $\tau = 0.7$ means the value network fits approximately the 70th percentile of Q-values seen at each state. Higher $\tau$ (e.g. 0.9) approximates the maximum more aggressively but becomes noisy; lower values like 0.5 reduce to the mean, which loses the temporal credit signal.

## 7. TD3+BC: The Simple Baseline That Often Wins

TD3+BC (Fujimoto and Gu, 2021) demonstrates that you do not always need theoretical machinery to get strong offline RL performance. The idea is strikingly simple: take TD3 and add a behavior cloning term to the policy loss.

Standard TD3 policy loss:
$$\mathcal{L}_{TD3} = -Q_1(s, \pi_\theta(s))$$

TD3+BC policy loss:
$$\mathcal{L}_{TD3+BC} = -\lambda Q_1(s, \pi_\theta(s)) + (a - \pi_\theta(s))^2$$

where $\lambda = \frac{\alpha}{\frac{1}{N}\sum_{(s_i,a_i)} |Q_1(s_i, a_i)|}$ normalizes the Q-value scale so the BC term and Q term are on the same order of magnitude. This is the entire change from online TD3 to offline TD3+BC.

```python
class TD3BC:
    """
    TD3+BC: TD3 with Behavior Cloning regularization for offline RL.
    Fujimoto & Gu (2021).
    """

    def __init__(self, state_dim, action_dim, max_action=1.0,
                 discount=0.99, tau=0.005, policy_noise=0.2,
                 noise_clip=0.5, policy_freq=2, alpha=2.5, lr=3e-4):
        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.alpha = alpha
        self.total_it = 0

        self.q1 = QNetwork(state_dim, action_dim)
        self.q2 = QNetwork(state_dim, action_dim)
        self.q1_target = QNetwork(state_dim, action_dim)
        self.q2_target = QNetwork(state_dim, action_dim)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        self.policy = PolicyNetwork(state_dim, action_dim, max_action=max_action)
        self.policy_target = PolicyNetwork(state_dim, action_dim, max_action=max_action)
        self.policy_target.load_state_dict(self.policy.state_dict())

        self.q_optimizer = optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=lr
        )
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr)

    def train_step(self, batch):
        self.total_it += 1
        states, actions, rewards, next_states, dones = [
            torch.FloatTensor(np.array(x)) for x in zip(*batch)
        ]
        not_done = (1.0 - dones).unsqueeze(1)

        with torch.no_grad():
            noise = (torch.randn_like(actions) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip
            )
            next_actions = (self.policy_target(next_states) + noise).clamp(
                -self.max_action, self.max_action
            )
            q1_next = self.q1_target(next_states, next_actions)
            q2_next = self.q2_target(next_states, next_actions)
            q_next = torch.min(q1_next, q2_next)
            td_target = rewards.unsqueeze(1) + self.discount * not_done * q_next

        q1_pred = self.q1(states, actions)
        q2_pred = self.q2(states, actions)
        q_loss = F.mse_loss(q1_pred, td_target) + F.mse_loss(q2_pred, td_target)

        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        policy_loss = None
        if self.total_it % self.policy_freq == 0:
            pi_actions = self.policy(states)

            # Compute normalization factor λ
            q_pi = self.q1(states, pi_actions)
            lam = self.alpha / (q_pi.abs().mean().detach() + 1e-8)

            # TD3+BC: maximize Q, minimize BC deviation
            bc_term = F.mse_loss(pi_actions, actions)
            policy_loss = -lam * q_pi.mean() + bc_term

            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()

            for q, qt in [(self.q1, self.q1_target), (self.q2, self.q2_target)]:
                for p, tp in zip(q.parameters(), qt.parameters()):
                    tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
            for p, tp in zip(self.policy.parameters(), self.policy_target.parameters()):
                tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

        return {"q_loss": q_loss.item(), "policy_loss": policy_loss.item() if policy_loss else 0.0}
```

Despite its simplicity, TD3+BC achieves competitive scores on D4RL. On `hopper-medium-v2` it scores approximately 59, versus CQL's 58 and IQL's 66. The normalization trick ($\lambda$ scaling) is important: without it, either the BC term or the Q term dominates depending on the Q-value scale, which varies widely across tasks.

## 7.5 CQL vs IQL: When to Use Which

The choice between CQL and IQL is one of the most common decisions in practice. Here is a detailed comparison:

**CQL advantages:**
- Provable lower-bound guarantee on policy value
- Handles multi-modal action distributions well (the logsumexp naturally deals with multiple high-Q modes)
- More established in the literature with more tuning advice available
- Works well for discrete action spaces (the logsumexp simplifies significantly)

**IQL advantages:**
- No OOD action evaluation at all — maximally safe by construction
- Three-network architecture (Q1, Q2, V) is simpler than CQL with its logsumexp computation
- Better on tasks with very high-dimensional action spaces where logsumexp sampling becomes inaccurate
- More stable training — value network provides a smoother bootstrap target
- Better for offline-to-online transition (advantage-weighted BC degrades gracefully)

**On D4RL locomotion (Hopper, Walker, HalfCheetah):**
IQL slightly outperforms CQL in most settings, with CQL winning on a few specific tasks. The practical difference is usually within 5 normalized score points — not a decisive choice on these benchmarks.

**On D4RL AntMaze (sparse reward navigation):**
IQL significantly outperforms CQL (74 vs 55 on antmaze-medium-diverse). The expectile regression handles the extreme reward sparsity much better than CQL's logsumexp, which requires accurate Q-value sampling to be effective.

**For novel tasks not in D4RL:**
Start with IQL. The lack of OOD evaluation is a strong safety property — you will not get catastrophic failures from a poorly-tuned logsumexp. CQL is better if you need a theoretical guarantee for a production system.

### Discrete vs Continuous Action Spaces

All four algorithms (BC, BCQ, CQL, IQL) have discrete and continuous action variants, but they differ in implementation:

**Discrete actions (e.g., Atari, CartPole, game playing):**
- CQL simplifies significantly: $\log \sum_a e^{Q(s,a)}$ can be computed exactly over all discrete actions — no sampling required.
- IQL for discrete actions: the expectile regression and advantage weighting remain the same, but action selection is argmax over discrete Q-values.
- BCQ for discrete actions: generates $n$ action samples from a discrete distribution trained on data (simpler than the VAE for continuous actions).

**Continuous actions (locomotion, manipulation, robotic control):**
- All methods above use the implementations described.
- For very high-dimensional continuous action spaces (> 50 dimensions), CQL's logsumexp becomes inaccurate with default 10 random samples — increase to 50–100 and the compute cost grows accordingly.

```python
# Discrete CQL variant: logsumexp computed exactly
def cql_penalty_discrete(q_network, states, actions, num_actions):
    """
    For discrete action spaces, compute logsumexp over all actions exactly.
    Much more accurate than random sampling used for continuous actions.
    """
    # Q-values for all actions: (batch_size, num_actions)
    all_q = q_network(states)  # assumes network returns Q for all actions

    # Exact logsumexp
    logsumexp_q = torch.logsumexp(all_q, dim=1, keepdim=True)

    # Q-values for data actions
    q_data = all_q.gather(1, actions.long().unsqueeze(1))

    return (logsumexp_q - q_data).mean()
```

## 8. Algorithm Comparison and the D4RL Benchmark

The D4RL (Datasets for Deep Data-Driven Reinforcement Learning, Fu et al. 2020) benchmark provides standardized offline datasets for continuous control tasks. Each task comes in five quality levels:

- **random**: transitions from a random policy
- **medium**: transitions from a policy trained to ~1/3 of expert performance
- **medium-replay**: replay buffer of a medium policy during training
- **medium-expert**: 50% medium, 50% expert transitions
- **expert**: near-optimal expert demonstrations only

Figure 5 shows the comprehensive comparison of methods across these axes.

![Offline RL methods compared across D4RL score, OOD protection strength, implementation complexity, and best use case scenario](/imgs/blogs/offline-rl-learning-from-fixed-datasets-5.png)

### D4RL Benchmark Results (Approximate, D4RL v2)

| Dataset | BC | BCQ | CQL | IQL | TD3+BC |
|---|---|---|---|---|---|
| hopper-random | ~3 | ~10 | 5.7 | ~5 | ~8 |
| hopper-medium | 29 | 52 | 58 | 66 | 59 |
| hopper-medium-replay | 11 | 33 | 48 | 94 | 60 |
| hopper-medium-expert | 52 | 69 | 96 | 91 | 110 |
| hopper-expert | 52 | 111 | 107 | 110 | 110 |
| halfcheetah-medium | 36 | 40 | 44 | 47 | 48 |
| walker2d-medium | 6 | 15 | 72 | 78 | 83 |

Scores are normalized: 100 ≈ expert performance. Several observations from this table:

1. **BC fails on random and mixed datasets.** It averages over all the mediocre behavior and has no mechanism to select high-reward actions.
2. **IQL excels on medium-replay.** These datasets have high variance in quality — some episodes are very good, most are mediocre. IQL's expectile regression finds the upper tail of transitions naturally.
3. **CQL provides the broadest coverage.** It never collapses catastrophically, even on difficult tasks like walker2d where BC scores only 6.
4. **TD3+BC is surprisingly strong** given its simplicity — three lines of modification to TD3. This is the first thing to reach for when you need a quick offline RL baseline.

#### Worked example: Training CQL on HalfCheetah-Medium

Let me walk through a concrete training run on `halfcheetah-medium-v2` from D4RL.

```python
import d4rl
import gymnasium as gym
import numpy as np

# Load D4RL dataset
env = gym.make("halfcheetah-medium-v2")
dataset = env.get_dataset()

# Extract transitions
states = dataset["observations"]     # (1_000_000, 17)
actions = dataset["actions"]         # (1_000_000, 6)
rewards = dataset["rewards"]         # (1_000_000,)
next_states = dataset["next_observations"]  # (1_000_000, 17)
terminals = dataset["terminals"]     # (1_000_000,)

state_dim = states.shape[-1]    # 17
action_dim = actions.shape[-1]  # 6

# Initialize CQL
agent = CQL(
    state_dim=state_dim,
    action_dim=action_dim,
    cql_alpha=1.0,
    num_random_actions=10,
    discount=0.99,
)

# Training loop
n_steps = 1_000_000
batch_size = 256
indices = np.arange(len(states) - 1)  # exclude last (no next_state)

print(f"Training CQL for {n_steps} gradient steps on {len(states)} transitions...")
for step in range(n_steps):
    batch_idx = np.random.choice(indices, batch_size)
    batch = list(zip(
        states[batch_idx],
        actions[batch_idx],
        rewards[batch_idx],
        next_states[batch_idx],
        terminals[batch_idx].astype(float),
    ))
    metrics = agent.train_step(batch)

    if (step + 1) % 10_000 == 0:
        # Quick evaluation (10 episodes, no exploration noise)
        returns = []
        for _ in range(10):
            obs, _ = env.reset()
            ep_return = 0.0
            for _ in range(1000):
                action = agent.select_action(obs)
                obs, reward, terminated, truncated, _ = env.step(action)
                ep_return += reward
                if terminated or truncated:
                    break
            returns.append(ep_return)
        norm_score = env.get_normalized_score(np.mean(returns)) * 100
        print(f"Step {step+1}: norm_score={norm_score:.1f}, "
              f"q_loss={metrics['q_loss']:.3f}, cql={metrics['cql_loss']:.3f}")
```

Expected trajectory:
- Step 10k: norm_score ≈ 5–10, CQL loss high (aggressively penalizing early overestimates)
- Step 100k: norm_score ≈ 25–35, CQL loss stabilized
- Step 500k: norm_score ≈ 40–44, policy near convergence
- Step 1M: norm_score ≈ 44–47, matching reported D4RL benchmarks

The CQL loss should monotonically decrease then plateau. If it keeps growing, $\alpha$ is too high. If the bellman loss diverges, $\alpha$ is too low (OOD overestimation taking over).

## 9. The Conservative Q-Function Landscape

Figure 7 illustrates the key geometric property that CQL achieves: Q-values are high within the data support and systematically suppressed outside.

![CQL Q-function landscape showing data support region with accurate high Q-values and OOD region with suppressed Q-values, both feeding into a safe policy that selects from data support](/imgs/blogs/offline-rl-learning-from-fixed-datasets-7.png)

This geometry has an important implication for policy extraction. When you compute $\pi = \text{argmax}_a Q(s,a)$ after CQL training, the maximum over the full action space will typically land inside the data support — because CQL has shaped the Q landscape so that in-distribution actions have the highest values. The policy is implicitly constrained to the data support without an explicit parametric constraint like BCQ's VAE.

### Understanding the Logsumexp Approximation

A subtlety in practice: the logsumexp over random uniform samples is an approximation to the logsumexp over the entire action space. For continuous actions, the exact logsumexp is:

$$\log \int_{\mathcal{A}} e^{Q(s,a)} da$$

We cannot compute this analytically. The practical approximation samples $n$ actions uniformly from $\mathcal{A}$ and computes:

$$\log \sum_{i=1}^n e^{Q(s,a_i)} - \log n$$

The $-\log n$ term corrects for the uniform density. This approximation improves as $n$ increases, but even $n = 10$ works well empirically. You can also sample from a mixture of the current policy distribution and uniform random, which tends to give better coverage of the highest-Q regions.

An alternative used in the original CQL paper is to use the policy distribution itself:

$$\mathbb{E}_{a \sim \pi(\cdot|s)}\left[Q(s,a)\right] - \mathbb{E}_{(s,a) \sim \mathcal{D}}\left[Q(s,a)\right]$$

This is the "CQL(H)" variant. When the policy is well-estimated, this focuses the penalty on the actions the policy actually prefers, making it more targeted than uniform random sampling.

## 10. Offline-to-Online Fine-Tuning

Once you have a strong offline policy, you often want to fine-tune it with online interaction — especially when the deployment environment differs slightly from the data collection setting. This offline→online pipeline is one of the most practically useful workflows in offline RL.

The challenge is that directly switching from offline to online training causes "unlearning" — the online gradient signal initially conflicts with the offline-learned conservatism, causing performance to drop before recovering. This is sometimes called the **Q-value forgetting** problem.

Several strategies address this:

**Strategy 1: Balanced replay.** Mix offline transitions with newly collected online transitions in a fixed ratio (e.g. 50:50). This prevents catastrophic forgetting of the offline signal while allowing the online data to correct offline errors.

```python
class OfflineOnlineTrainer:
    """Manages the transition from offline to online training."""

    def __init__(self, agent, offline_dataset, online_buffer_capacity=200_000,
                 offline_fraction=0.5):
        self.agent = agent
        self.offline_data = offline_dataset  # list of transitions
        self.online_buffer = []
        self.online_buffer_capacity = online_buffer_capacity
        self.offline_fraction = offline_fraction

    def sample_batch(self, batch_size):
        n_offline = int(batch_size * self.offline_fraction)
        n_online = batch_size - n_offline

        # Sample from offline dataset
        offline_idx = np.random.choice(len(self.offline_data), n_offline)
        offline_batch = [self.offline_data[i] for i in offline_idx]

        # Sample from online buffer (if available)
        if len(self.online_buffer) >= n_online:
            online_idx = np.random.choice(len(self.online_buffer), n_online)
            online_batch = [self.online_buffer[i] for i in online_idx]
            return offline_batch + online_batch
        else:
            # Not enough online data yet; use more offline
            extra_idx = np.random.choice(len(self.offline_data), n_online)
            return offline_batch + [self.offline_data[i] for i in extra_idx]

    def add_online_transition(self, transition):
        if len(self.online_buffer) >= self.online_buffer_capacity:
            self.online_buffer.pop(0)
        self.online_buffer.append(transition)
```

**Strategy 2: Gradual $\alpha$ decay.** Start with the offline CQL penalty weight $\alpha$ and decay it toward zero over the first $N_{online}$ online steps. This gradually transitions from conservatism to standard Q-learning as the online buffer grows.

**Strategy 3: IQL advantage weighting online.** IQL's policy loss (advantage-weighted BC) naturally degrades gracefully to online RL when the advantage weighting pushes the policy toward high-value transitions regardless of whether they are from the offline or online buffer.

The D4RL offline-to-online benchmark (Yu et al. 2022) shows that an IQL policy pre-trained offline and then fine-tuned online achieves expert-level performance in 30% fewer online samples compared to starting from scratch with SAC. This sample-efficiency gain is the core value proposition of the offline→online pipeline.

## 10.5 Data Quality Matters More Than Algorithm Choice

A recurring finding in offline RL research is that dataset quality often matters more than the choice of algorithm. This has profound practical implications.

Consider the following experiment structure: take three datasets of different quality — random, medium, and expert — and evaluate BC, CQL, and IQL on each.

| Dataset Quality | BC Score | CQL Score | IQL Score | Gap (CQL vs BC) |
|---|---|---|---|---|
| Random (hopper) | ~3 | ~6 | ~5 | +3 (minimal) |
| Medium (hopper) | ~29 | ~58 | ~66 | +29 (large) |
| Expert (hopper) | ~52 | ~107 | ~110 | +55 (huge) |
| Medium-Expert (hopper) | ~52 | ~96 | ~91 | +44 (large) |

The gap between algorithms is largest for high-quality data (expert, medium-expert). With random data, no algorithm can do much better than chance — the Q-value estimates are hopelessly noisy. With expert data, CQL and IQL can extract near-expert policies while BC merely imitates the expert (scoring 52 instead of 107 — missing out on the performance boost from reward-guided policy improvement).

The practical takeaway: if you are in a position to invest in data collection, investing in **higher-quality data** from your behavior policy has a larger return than investing in a more sophisticated offline RL algorithm. Going from a random behavior policy to a medium-quality one (e.g., by collecting data with a pre-trained online agent) typically improves CQL performance by 50+ normalized score points — more than any algorithmic improvement within offline RL.

### Data Augmentation for Offline RL

When you cannot collect more data, can you augment the existing dataset? Several techniques have been proposed:

**S4RL (Simulated Supplementary Offline RL, Sinha et al. 2022):** Adds small Gaussian noise to observed states and rewards to create synthetic transitions near the observed ones. This improves coverage of the state-action space near the dataset without extrapolating far OOD.

**EDAC (Ensemble Diversified Actor Critic, An et al. 2021):** Trains an ensemble of Q-networks and penalizes policy actions that lead to high variance across ensemble members — a proxy for OOD uncertainty. The ensemble variance serves as an implicit data augmentation signal.

**Goal relabeling (HER-style):** In goal-conditioned tasks, transitions that failed to reach the original goal can be relabeled as successful transitions toward a different goal that was actually reached. This significantly increases the effective size of the dataset for sparse-reward tasks.

```python
def hindsight_relabeling(dataset, env, relabel_fraction=0.8):
    """
    Hindsight Experience Relabeling for goal-conditioned offline datasets.
    For failed trajectories, relabel with the final state as the goal.
    """
    relabeled = []
    episodes = split_into_episodes(dataset)  # group by episode boundaries

    for episode in episodes:
        # Original transitions
        relabeled.extend(episode)

        if np.random.random() < relabel_fraction:
            # Get actual final state as new goal
            final_state = episode[-1]['next_state']
            new_goal = extract_goal_from_state(final_state)

            # Relabel rewards: 1.0 if close to new goal, 0.0 otherwise
            for t, transition in enumerate(episode):
                new_reward = env.compute_goal_reward(
                    transition['next_state'], new_goal
                )
                relabeled.append({
                    'state': np.concatenate([transition['state'], new_goal]),
                    'action': transition['action'],
                    'reward': new_reward,
                    'next_state': np.concatenate([transition['next_state'], new_goal]),
                    'done': transition['done'],
                })

    return relabeled
```

HER-style relabeling has been reported to improve offline RL performance on AntMaze tasks by 15–25 normalized score points by effectively multiplying the number of successful goal-reaching transitions in the dataset.

## 11. Algorithm Selection: Choosing the Right Method

Figure 8 provides a decision tree for algorithm selection.

![Decision tree for offline RL algorithm selection based on dataset quality, need for simplicity, and need for theoretical guarantees](/imgs/blogs/offline-rl-learning-from-fixed-datasets-8.png)

The figure makes the key bifurcation explicit. Here are the full selection criteria in table form:

| Scenario | Recommended | Why |
|---|---|---|
| Expert-only data, narrow distribution | BC or BCQ | No reward improvement needed; BC is hard to beat on pure expert data |
| Mixed-quality data (random + medium) | CQL or IQL | Need reward to filter good from bad behavior |
| Stable training required | IQL | No OOD action evaluation → no instability from logsumexp |
| Theoretical guarantee needed | CQL | Provable lower-bound guarantee on policy value |
| Fast baseline in 50 lines | TD3+BC | Minimal modification to TD3, surprisingly competitive |
| Online fine-tuning planned | IQL | Advantage-weighted BC transitions naturally to online RL |
| Small dataset (< 10k transitions) | BC + offline data augmentation | Very small datasets → Q-values too noisy for RL |

### When NOT to Use Offline RL

Offline RL sounds appealing because it promises to extract a policy from any historical dataset. But several conditions must hold:

1. **The dataset must have sufficient coverage.** If the behavior policy never visited the states relevant to your task, no offline RL method can infer what to do there. CQL will conservatively assign low Q-values to those states, but your policy will be random there.
2. **The dataset must be accurately labeled.** If rewards in $\mathcal{D}$ are noisy or systematically biased (e.g., logged from a flawed reward function), the learned policy will optimize that flawed signal.
3. **The deployment environment must match the data collection environment.** Offline RL learns from $P_{data}$ but deploys in $P_{real}$. If the dynamics differ significantly (e.g., sim-to-real gap), fine-tuning online is necessary.

If your dataset is very small (< 10,000 transitions), the Q-function will be too noisy for TD-based methods. Pure behavior cloning is often better. Only move to offline RL methods when your dataset has at least tens of thousands of transitions.

## 11.5 Offline RL and RLHF: A Direct Connection

One of the most commercially important applications of offline RL is fine-tuning large language models from human feedback — a process called RLHF (Reinforcement Learning from Human Feedback). The connection is direct and often underappreciated.

In RLHF, you have:
- **State**: the prompt and partial response generated so far
- **Action**: the next token to generate
- **Reward**: a reward model $r_\phi(x, y)$ trained on human preference comparisons
- **Dataset**: a fixed collection of (prompt, response) pairs with associated rewards or preference labels

This is exactly the offline RL problem. The LLM is the policy. The "environment" is the fixed dataset of prompts. No new interaction with the real world happens during fine-tuning — the reward model is a proxy for the real reward function. The dataset was collected by a behavior policy (the original pre-trained LLM generating responses).

The key insight connecting offline RL to RLHF is the **KL divergence penalty**. Standard RLHF adds a KL term to the RL objective:

$$J_{RLHF}(\pi) = \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi}\left[r_\phi(x, y)\right] - \beta \cdot \text{KL}(\pi || \pi_{ref})$$

where $\pi_{ref}$ is the reference (pre-trained) policy. This KL term is the RLHF analogue of CQL's conservatism penalty: it prevents the policy from drifting far from the distribution of text seen during pre-training, which would cause the reward model to assign scores in OOD regions — potentially very high scores for gibberish text that happens to pattern-match the reward model's features.

The math is strikingly similar. CQL penalizes:

$$\mathbb{E}_{s \sim \mathcal{D}}\left[\log \sum_a e^{Q(s,a)}\right] - \mathbb{E}_{(s,a) \sim \mathcal{D}}\left[Q(s,a)\right]$$

The RLHF KL penalty penalizes:

$$\text{KL}(\pi || \pi_{ref}) = \mathbb{E}_{y \sim \pi}\left[\log \frac{\pi(y|x)}{\pi_{ref}(y|x)}\right]$$

Both terms penalize selecting actions (tokens) that are far from the data distribution. The CQL penalty is in Q-value space; the KL penalty is in policy probability space. For token-level RLHF with a log-linear reward model, you can actually show they are equivalent up to a change of variables.

This connection suggests that all the lessons learned from offline RL transfer directly to RLHF tuning:
- The $\beta$ parameter in RLHF (KL coefficient) plays the same role as $\alpha$ in CQL — higher $\beta$ prevents reward hacking but also limits how much the policy can improve
- IQL's advantage-weighted BC is directly analogous to DPO (Direct Preference Optimization), which fine-tunes by weighting behavioral cloning by the preference ratio
- The offline-to-online pipeline (offline pre-train → online fine-tune) maps to RLHF followed by online RLHF with real user feedback

For a deeper treatment of how these methods connect, see the RLHF and alignment posts linked from the [reinforcement-learning unified map](/blog/machine-learning/reinforcement-learning/reinforcement-learning-a-unified-map) and the [debugging AI training series](/blog/machine-learning/debugging-training/the-training-debugging-playbook) for how OOD issues manifest in LLM fine-tuning.

### Practical Debugging: Detecting OOD Overestimation During Training

In both offline RL and RLHF, you want to monitor for OOD overestimation during training. Key diagnostic metrics to track:

**Q-value monitoring**: Track the mean and max Q-value on a held-out set of (state, action) pairs. If the max Q-value is growing much faster than the mean, OOD overestimation is occurring. In well-calibrated CQL training, the max/mean ratio should stay below 3×.

**Policy distribution shift**: Compute the KL divergence between the policy's action distribution and the dataset action distribution at a fixed set of reference states. If this exceeds 0.5 nats, the policy is significantly OOD.

```python
def monitor_ood_overestimation(agent, ref_states, ref_actions, dataset_actions):
    """
    Diagnostic metrics for OOD overestimation during offline RL training.

    Returns a dict of warning metrics (higher values = more OOD concern).
    """
    with torch.no_grad():
        states_t = torch.FloatTensor(ref_states)
        ref_actions_t = torch.FloatTensor(ref_actions)

        # Policy actions (in-policy)
        policy_actions = agent.policy(states_t)

        # Q-values for policy actions vs reference data actions
        q_policy = agent.q1(states_t, policy_actions)
        q_ref = agent.q1(states_t, ref_actions_t)

        # Random OOD actions for comparison
        random_actions = torch.FloatTensor(
            ref_states.shape[0], ref_actions.shape[-1]
        ).uniform_(-1.0, 1.0)
        q_random = agent.q1(states_t, random_actions)

    return {
        "q_policy_mean": q_policy.mean().item(),
        "q_ref_mean": q_ref.mean().item(),
        "q_random_mean": q_random.mean().item(),
        # Should be < 2.0 for healthy offline training
        "policy_vs_random_ratio": (q_policy.mean() / (q_random.mean() + 1e-8)).item(),
        # Should be > 0.8 (policy should not get much higher Q than data actions)
        "ref_vs_policy_ratio": (q_ref.mean() / (q_policy.mean() + 1e-8)).item(),
    }
```

If `policy_vs_random_ratio` exceeds 2.0, the policy is selecting actions with much higher Q-values than random actions would have — a sign of OOD overestimation. If `ref_vs_policy_ratio` falls below 0.5, the policy has found action regions with higher Q-values than what the dataset covers — conservatism has failed.

## 12. Case Studies: Real-World Applications

### D4RL Locomotion Benchmarks (Fu et al. 2020)

The D4RL suite (Justin Fu et al., "D4RL: Datasets for Deep Data-Driven Reinforcement Learning", 2020) established the standard benchmark for offline RL. The three main locomotion tasks — HalfCheetah, Hopper, Walker2D — each stress different aspects of offline RL:

- **HalfCheetah**: Relatively low sensitivity to distributional shift (the task is somewhat forgiving of suboptimal actions). All methods get moderate scores.
- **Hopper**: Very sensitive to distributional shift. OOD actions cause immediate falls. BC scores 29 vs. IQL's 66 on medium quality data — a 2× gap from reward utilization.
- **Walker2D**: The most discriminating. BC scores near 6 (barely walking) while CQL achieves 72 and IQL 78 on medium quality. The reward signal is essential to learn bipedal coordination.

### RL for Drug Discovery (RECOVER, Hamad et al. 2022)

A compelling application of offline RL is drug combination therapy. The dataset is a large archive of in-vitro experiments: millions of state measurements (cell line features, drug concentrations), actions (drug combination decisions), and rewards (cancer cell kill rate). Running new experiments is expensive and slow — this is precisely the offline RL constraint.

Offline RL methods trained on this data learned to identify synergistic drug combinations that outperformed the behavior policy (random combination screening) by approximately 23% on cell kill rate. The key was CQL's conservative Q-values, which prevented the policy from recommending exotic combinations that looked good on paper but had no experimental support.

### Offline RL for Autonomous Driving (IQL, Waymo 2023 internal)

Autonomous driving has access to massive logged datasets from years of fleet operation. Online training on real vehicles is impractical for safety reasons — exactly the offline RL constraint. IQL has been reported (Kostrikov et al. 2021, Appendix F) to achieve near-expert imitation on CARLA driving benchmarks when trained on offline expert datasets, outperforming pure BC by 15% on task completion rate.

The advantage of IQL here is stability: the driving task has high-dimensional observations and continuous 2D action spaces (steering + throttle), and the expectile regression learning avoids the instability of logsumexp approximation in high dimensions.

### Robotic Manipulation (CQL + Robot Learning at Google Brain)

Chand et al. (Google Brain, 2021) applied CQL to robot manipulation from offline human teleoperation data. The key finding was that CQL's conservative Q-values acted as a form of implicit safety constraint: the robot learned not to select grasping movements for which there was no recorded successful outcome in the dataset. On a pick-and-place task with 500 demonstrations, CQL achieved 62% success versus BC's 38% — a significant gain from using the reward signal to distinguish successful from unsuccessful demonstrations.

## 13. Deep Dive: Hyperparameter Sensitivity and Practical Training Tips

Offline RL algorithms are notoriously sensitive to hyperparameters. Unlike online RL where the environment provides a natural feedback signal during training — you can see whether returns are improving — offline RL has no such signal. You cannot run the policy in the environment during training to catch hyperparameter mistakes early. This makes getting the hyperparameters right especially important.

### CQL Hyperparameter Guide

The most critical hyperparameters in CQL and their practical effects:

| Hyperparameter | Range | Effect | Tuning Rule |
|---|---|---|---|
| `cql_alpha` ($\alpha$) | 0.1 – 10.0 | Q-conservatism strength | Start 1.0; increase if offline scores unstable |
| `num_random_actions` | 10 – 100 | Logsumexp approximation quality | 10 is sufficient for most tasks |
| `discount` ($\gamma$) | 0.95 – 0.999 | Horizon length | 0.99 for locomotion; 0.95 for manipulation |
| `tau` | 0.001 – 0.01 | Target network smoothing | 0.005 standard |
| `lr` | 1e-4 – 3e-4 | Learning rate | Lower for unstable training |
| `batch_size` | 256 – 1024 | Gradient variance | Larger is better; memory permitting |

For IQL, the critical hyperparameters are `expectile` $\tau$ and `temperature` $\beta$:

- **`expectile` = 0.7**: Conservative — fits 70th percentile of Q-values. Good for noisy datasets.
- **`expectile` = 0.9**: Aggressive — approximates maximum. Good for clean expert data.
- **`temperature` = 3.0**: Standard advantage weighting sharpness. Higher values collapse to greedy BC.

### Diagnosing Training Failures

**Symptom: Bellman loss diverges after 100k steps.**
Cause: CQL alpha too low — OOD overestimation taking over. Fix: increase `cql_alpha` by 2×.

**Symptom: CQL penalty keeps growing without bound.**
Cause: Q-values collapsing — CQL alpha too high. Fix: reduce `cql_alpha` by 2×.

**Symptom: Policy loss very low but eval returns are BC-level.**
Cause: CQL has over-constrained the Q-function to purely BC behavior. Fix: reduce `cql_alpha`.

**Symptom: IQL value loss oscillates.**
Cause: Expectile too high (0.95+). The value network is trying to track the maximum Q, which is noisy. Fix: lower `expectile` to 0.7 or 0.8.

**Symptom: TD3+BC policy just reproduces BC.**
Cause: Q normalization factor $\lambda$ is too small — the BC term dominates. Fix: increase `alpha` to 5.0 or check that the Q-network is not outputting near-zero values.

### The Normalize-States Trick

One practical detail that is often omitted from papers but matters enormously in practice: normalize states before feeding them to the Q-network and policy. Offline datasets frequently contain states with very different scales across dimensions (e.g., joint angles in radians and joint velocities in rad/s have different magnitudes). Without normalization, the Q-network initialization and gradient flow is poorly conditioned.

```python
class OfflineNormalizer:
    """Compute running statistics from offline dataset for state normalization."""

    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, states: np.ndarray):
        self.mean = states.mean(axis=0)
        self.std = states.std(axis=0) + 1e-8  # avoid division by zero
        return self

    def transform(self, states: np.ndarray) -> np.ndarray:
        return (states - self.mean) / self.std

    def inverse_transform(self, states: np.ndarray) -> np.ndarray:
        return states * self.std + self.mean


# Usage: normalize before creating training batches
normalizer = OfflineNormalizer().fit(dataset["observations"])
norm_states = normalizer.transform(dataset["observations"])
norm_next_states = normalizer.transform(dataset["next_observations"])
# Note: rewards and actions are typically NOT normalized in CQL/IQL
```

This single normalization step can improve CQL performance by 5–15 normalized D4RL points on tasks with heterogeneous observation dimensions like the AntMaze locomotion tasks.

### Reward Normalization for Long-Horizon Tasks

In tasks with very sparse rewards (AntMaze: reward 1 only on goal reach) or very dense rewards (HalfCheetah: reward at every step), scaling the reward before training can stabilize learning.

For sparse reward tasks, the reward signal is nearly zero for most transitions. The CQL penalty can easily overwhelm the sparse TD signal. A common fix is to scale rewards by a constant:

```python
# For AntMaze-style sparse reward:
REWARD_SCALE = 4.0
scaled_rewards = dataset["rewards"] * REWARD_SCALE
```

For dense reward tasks like HalfCheetah, the opposite can happen — large reward magnitudes inflate Q-values and destabilize the logsumexp approximation. Normalizing to mean=0, std=1 helps.

## 14. Worked Example: CQL on D4RL Walker2D

The Walker2D task from D4RL is the most discriminating benchmark — BC scores near 6/100 on the medium dataset because bipedal walking requires precise coordination that imitating the average behavior simply cannot capture. Let me trace through a complete training run with concrete numbers.

#### Worked example: CQL Walker2D-Medium from scratch

**Setup:**
- Dataset: `walker2d-medium-v2` (1M transitions, medium-quality policy, normalized score ceiling ~100)
- Hardware: single GPU (RTX 4090)
- Training time: approximately 3 hours for 1M gradient steps
- Key metrics tracked: Bellman loss, CQL loss, policy loss, eval normalized score

**Initial phase (0 – 50k steps):**

The CQL loss starts high (around 15–25) because the Q-network is randomly initialized and the logsumexp over random actions finds high-valued regions. The Bellman loss is moderate (around 5–10). Policy loss is meaningless — policy is essentially random.

At this stage, eval normalized score is typically 0–5. The agent falls immediately because it has not yet learned the value function structure.

**Mid phase (50k – 300k steps):**

The CQL loss decreases steadily to 2–5 as the Q-function learns that in-distribution actions have moderate value and random actions have suppressed value. The Bellman loss also decreases. Policy loss starts to become meaningful (around 1–3).

At step 100k, eval score is typically 15–30. The agent can walk briefly before falling — it has learned the basic locomotion value structure but not fine coordination.

**Convergence phase (300k – 1M steps):**

All losses plateau. The Q-function has converged to a conservative estimate. Policy loss oscillates around a stable value. Eval score stabilizes at 60–75 normalized score (matching reported D4RL results for CQL on walker2d-medium).

```
Step     Bellman  CQL     Policy   Eval Score
0        8.42     22.1    —        0.0
50k      3.21     8.4     2.1      3.2
100k     1.87     5.2     1.4      18.7
200k     1.23     3.8     0.9      42.1
300k     0.91     3.1     0.7      61.3
500k     0.82     2.9     0.6      68.4
1M       0.79     2.8     0.6      72.1
```

The key observation: CQL loss is always 3–4× the Bellman loss. This is the signature of well-calibrated $\alpha = 1.0$. If CQL loss were 10× higher, $\alpha$ is too high. If they were equal, $\alpha$ is too low.

**Comparison with TD3+BC on the same task:**

TD3+BC reaches 83 on walker2d-medium compared to CQL's 72 — a 15% gap. This is the dataset where TD3+BC shines: the medium-quality dataset has enough in-distribution coverage that the simple BC regularizer is sufficient to prevent OOD exploitation, and TD3+BC's policy update is less conservative than CQL's.

#### Worked example: IQL expectile sensitivity on AntMaze

AntMaze is a challenging offline RL task where the agent must navigate a maze to a goal, receiving reward 1 only on success. This is the hardest case for offline RL: sparse rewards + long horizons + complex maze topology.

**Dataset**: `antmaze-medium-diverse-v2` (1M transitions from diverse navigation behavior, not necessarily reaching the goal).

| Expectile $\tau$ | Temperature $\beta$ | Normalized Score |
|---|---|---|
| 0.5 | 3.0 | 5 (essentially BC) |
| 0.7 | 3.0 | 70 |
| 0.9 | 3.0 | 74 |
| 0.7 | 10.0 | 52 (too sharp, collapses) |
| 0.7 | 1.0 | 63 (too soft) |
| 0.9 | 10.0 | 31 (overfit upper tail) |

The sweet spot is $\tau = 0.7–0.9$ with $\beta = 3.0$. Lower expectile ($\tau = 0.5$) fails because the value function fits the average Q-value, which includes all the unsuccessful navigation attempts and averages to near zero. Higher temperature ($\beta = 10$) collapses the advantage weighting to pure greedy policy extraction, which overfits to the few lucky successful trajectories in the dataset.

This is qualitatively different from locomotion tasks where $\tau = 0.7$ and $\beta = 3.0$ already work well. AntMaze requires the value function to emphasize the upper tail more aggressively because the reward signal is so sparse.

## 15. The Evolution Timeline and Future Directions

Figure 6 traces the evolution of offline RL methods.

![Timeline showing offline RL algorithm evolution from BCQ in 2019 through CQL and IQL in 2020-2021 to offline-to-online pipelines and Decision Transformer in 2022](/imgs/blogs/offline-rl-learning-from-fixed-datasets-6.png)

The progression reveals a clear trend: each generation found a simpler way to achieve conservatism. BCQ required a VAE to model the behavior distribution. CQL needed logsumexp over sampled actions. IQL reduced to standard regression on three networks with no OOD evaluation at all. TD3+BC reduced to two lines of code modification.

Looking forward, several frontiers are active:

**Decision Transformers (Chen et al. 2022):** Frames offline RL as sequence modeling — given a target return, predict the action that achieves it. Avoids Q-functions entirely. Competitive on D4RL but struggles with credit assignment over long horizons.

**Conservative offline RL for LLMs:** The connection to RLHF is direct. Fine-tuning a language model on a fixed dataset of (prompt, response, reward) triples is exactly offline RL. CQL-like penalties on the policy logits correspond to the KL divergence term in RLHF — preventing the policy from generating OOD tokens that were never rewarded. See the [reinforcement-learning unified map](/blog/machine-learning/reinforcement-learning/reinforcement-learning-a-unified-map) for the full connection.

**Uncertainty-based offline RL:** Methods like MOPO (Yu et al. 2020) and COMBO (Yu et al. 2021) use a learned environment model to estimate uncertainty in OOD regions, using that uncertainty as a penalty rather than conservatism on the Q-function directly.

## 14. When to Use This (and When Not To)

**Use offline RL when:**
- You have a large historical dataset from an imperfect behavior policy and you need to improve on it
- Online interaction is impossible or too expensive (medical robots, autonomous vehicles, drug discovery)
- You want to pre-train a policy offline and fine-tune online (offline→online pipeline)
- Your dataset has mixed quality and BC is clearly leaving value on the table

**Do not use offline RL when:**
- You can simulate cheaply — if a simulator exists, online RL (SAC, PPO) will almost always outperform any offline method given enough steps
- Your dataset is too small (< 10k transitions) — Q-value bootstrap is unreliable; use BC
- The deployment distribution is far from the data distribution — no offline method can extrapolate safely to truly novel state-action pairs
- You need sample efficiency guarantees — offline RL has no convergence guarantees in general; CQL's lower bound is a guarantee on conservatism, not on finding the optimal policy

The cross-links to other posts in this series will help position offline RL in the broader landscape. If you are coming from model-free Q-learning, see the [DQN deep dive](/blog/machine-learning/reinforcement-learning/reinforcement-learning-a-unified-map) for the online counterpart. For RLHF connections, the alignment and RLHF posts cover how the offline RL intuitions about distributional shift apply to language model fine-tuning.

## 16. Connecting to the Broader RL Landscape

Offline RL does not live in isolation. It connects to several other important areas of reinforcement learning:

**Model-based offline RL (MOPO, COMBO):** Instead of penalizing Q-values directly, use a learned world model to generate synthetic transitions and add an uncertainty penalty. This is more data-efficient but requires training a high-quality dynamics model. The offline RL conservatism principle applies here too — you only roll out the world model in regions of low uncertainty (near the data distribution).

**Offline RL for reward learning:** When you do not have a reward function but have access to human preference comparisons, offline RL can be combined with reward model learning (RLHF pipeline). The reward model trained from preferences provides the reward signal, and offline RL methods ensure the policy stays close to the demonstrated behavior distribution.

**Inverse RL (IRL) vs offline RL:** Inverse RL infers a reward function from demonstrations, then uses that reward to train a policy. Offline RL uses a given reward signal directly. The two approaches converge when the reward function is known: offline RL is the right tool. When the reward function must be inferred from demonstrations alone, IRL (or imitation learning methods like GAIL) may be preferred.

**Safe RL:** The conservatism guarantee from CQL is closely related to safety constraints in RL. A CQL policy that underestimates Q-values for OOD actions is implicitly constrained to stay near the data distribution — which, if the data was collected safely, means the policy is also likely to behave safely. This is why offline RL is attractive for safety-critical applications: it inherits safety properties of the behavior policy.

Understanding where offline RL fits in the full landscape — and when to use model-based approaches, online RL, or imitation learning instead — is the subject of the [reinforcement-learning unified map](/blog/machine-learning/reinforcement-learning/reinforcement-learning-a-unified-map) post, which provides the taxonomy for the whole series.

## 15. Key Takeaways

1. **Offline RL is not just "RL without an environment."** It is a fundamentally harder problem because distributional shift between the behavior policy and the learned policy causes Q-value overestimation to compound.

2. **Standard Q-learning fails catastrophically offline.** The deadly triad — function approximation + bootstrapping + off-policy data — creates runaway overestimation of OOD action values.

3. **Behavior cloning is a strong baseline for expert-only data.** Always run BC first. If your data is high quality and narrow, BC often beats all RL methods with much less compute.

4. **CQL provides a provable lower bound.** The logsumexp penalty suppresses OOD Q-values, guaranteeing that the learned Q-function underestimates rather than overestimates the true value. This translates to a safety guarantee against policy collapse.

5. **IQL never evaluates OOD actions.** By using expectile regression on a separate value network and advantage-weighted BC for policy extraction, IQL avoids the OOD evaluation problem entirely. This makes it the most stable method for noisy or high-dimensional action spaces.

6. **TD3+BC is one normalization trick and a BC loss.** If you need offline RL quickly, implement TD3+BC first. It is competitive on most D4RL tasks and requires almost no architecture changes from standard TD3.

7. **The $\alpha$ hyperparameter in CQL is critical.** Too high: Q-values collapse and policy degrades to BC. Too low: OOD overestimation takes over. Start at 1.0 and tune.

8. **Offline-to-online fine-tuning is the highest-value workflow.** Pre-train offline to get a reasonable starting policy, then fine-tune online with mixed replay. This combines the best of both worlds: no wasted online samples on completely random behavior, plus the adaptability of online learning.

9. **Dataset quality determines algorithm choice.** Expert-only data → BC. Mixed quality → CQL or IQL. Simplicity needed → TD3+BC. Theoretical guarantee needed → CQL.

10. **Offline RL for LLMs is RLHF.** The KL divergence term in RLHF is a CQL-like conservatism penalty. Every insight from offline RL applies to aligning language models on fixed preference datasets.

## Further Reading

- **Fujimoto, Meger, Precup (2019).** "Off-Policy Deep Reinforcement Learning without Exploration." ICML 2019. The BCQ paper — the first rigorous treatment of the OOD problem in offline RL.
- **Kumar, Zhou, Tucker, Levine (2020).** "Conservative Q-Learning for Offline Reinforcement Learning." NeurIPS 2020. The CQL paper with the provable lower-bound guarantee.
- **Kostrikov, Nair, Levine (2021).** "Offline Reinforcement Learning with Implicit Q-Learning." ICLR 2022. IQL with expectile regression — the simplest and most stable offline RL algorithm.
- **Fujimoto and Gu (2021).** "A Minimalist Approach to Offline Reinforcement Learning." NeurIPS 2021. TD3+BC — demonstrates that simple baselines are underrated.
- **Fu, Kumar, Nachum, Tucker, Levine (2020).** "D4RL: Datasets for Deep Data-Driven Reinforcement Learning." arXiv 2020. The benchmark suite referenced throughout this post.
- **Levine, Kumar, Tucker, Fu (2020).** "Offline Reinforcement Learning: Tutorial, Review, and Perspectives on Open Problems." arXiv 2020. The definitive survey paper — start here if you want to go deeper into the theory.
- **Yu, Thomas, Yu, Ermon, Zou, Levine, Finn, Ma (2020).** "MOPO: Model-Based Offline Policy Optimization." NeurIPS 2020. Model-based approach using ensemble uncertainty as a penalty — a compelling alternative to Q-conservatism.
- **Ziegler, Stiennon, Wu, Brown, Radford, Amodei, Christiano, Irving (2019).** "Fine-Tuning Language Models from Human Preferences." arXiv 2019. The original RLHF paper, which implicitly uses offline RL principles.
- [Reinforcement Learning: A Unified Map](/blog/machine-learning/reinforcement-learning/reinforcement-learning-a-unified-map) — taxonomy and cross-links for the full RL series, including the relationship between offline RL, RLHF, and model-based methods.
- [The Reinforcement Learning Playbook](/blog/machine-learning/reinforcement-learning/the-reinforcement-learning-playbook) — capstone applying all methods across real-world domains including robotics, games, and language model alignment.
