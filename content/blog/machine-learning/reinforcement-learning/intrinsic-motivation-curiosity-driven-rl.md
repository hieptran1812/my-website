---
title: "Intrinsic motivation and curiosity-driven RL: exploring without a map"
date: "2026-06-22"
publishDate: "2026-06-22"
description: "Learn how information-theoretic curiosity bonuses — ICM, RND, VIME, and count-based methods — turn sparse-reward environments from unsolvable to tractable, and implement a full RND module in PyTorch and Stable-Baselines3."
tags:
  [
    "reinforcement-learning",
    "deep-learning",
    "exploration",
    "intrinsic-motivation",
    "curiosity-driven-rl",
    "random-network-distillation",
    "icm",
    "sparse-reward",
    "machine-learning",
    "pytorch",
  ]
category: "machine-learning"
subcategory: "Reinforcement Learning"
author: "Hiep Tran"
featured: true
readTime: 51
image: "/imgs/blogs/intrinsic-motivation-curiosity-driven-rl-1.png"
---

Picture this: you hand a reinforcement learning agent a copy of *Montezuma's Revenge* — the 1984 Atari platformer that has tormented AI researchers for decades. The agent starts in a room surrounded by ladders, ropes, and a locked door. A skull-shaped enemy patrols the floor. The first reward signal the agent can realistically get is behind that locked door, which requires finding a key in the room above, which requires climbing a rope that is only reachable by navigating a ladder sequence. With standard epsilon-greedy DQN, the agent sees zero reward for millions of steps. After 200 million environment interactions — roughly equivalent to 38 days of continuous play at 60 fps — the DQN baseline scores approximately 0 points. The agent simply gives up and sits in the corner, because every direction it tries looks equally bad from its Q-values' perspective.

The root problem is not that the agent is dumb. It is that the reward signal provides zero gradient to climb. Without feedback, exploration is a random walk over a combinatorial space, and the interesting parts of a large environment are almost certainly not reached by chance. This is the sparse-reward exploration problem, and it is one of the last major open challenges in deep RL. The standard exploration heuristics that work so well for CartPole and LunarLander completely collapse when the first nonzero reward is 60 correct actions away.

The solution this post covers is *intrinsic motivation*: give the agent a secondary reward signal derived not from the game score but from its own sense of curiosity — a desire to visit states it has not seen before, or whose dynamics it cannot yet predict. The core idea is as old as psychology (Berlyne's 1960 theory of exploratory behavior in humans), but only in the last decade have we found computational formulations that scale to pixel-level observations. By 2019, agents equipped with Random Network Distillation (RND) were scoring over 10,000 points on Montezuma's Revenge, clearing more than half the game's rooms. That is roughly a 10,000× improvement over DQN.

![RND reward augmentation pipeline showing how prediction error from a frozen random target network becomes a curiosity bonus that is summed with extrinsic reward before the PPO policy update](/imgs/blogs/intrinsic-motivation-curiosity-driven-rl-1.png)

By the end of this post you will understand, from first principles, why exploration fails in sparse-reward environments and why epsilon-greedy is theoretically incapable of solving them; what information-theoretic curiosity means mathematically and how it connects to Bayesian optimal exploration; how ICM, RND, VIME, and count-based methods each formalize curiosity and when each one is the right choice; how to implement a complete RND curiosity module from scratch in PyTorch and wire it into a Stable-Baselines3 PPO training loop; what the noisy-TV failure mode is, why it arises, and how to defend against it; and precisely when curiosity helps versus when it actively hurts your agent's performance. If you have not yet read [what is reinforcement learning](/blog/machine-learning/reinforcement-learning/what-is-reinforcement-learning), start there for the foundational RL loop — this post builds directly on that framework.

## The exploration problem in sparse-reward RL

Every RL agent faces the exploration-exploitation trade-off: the agent must balance spending time on known good actions versus trying unknown actions that might be better. In dense-reward settings — CartPole, LunarLander, most continuous control benchmarks — this is manageable, because every action produces a measurable reward signal within a few steps. The agent always has a learning signal to follow. Even epsilon-greedy exploration, which is theoretically naive, works well because the Q-function gradient is dense everywhere.

But consider an environment where the first non-zero reward appears after 500 sequential correct decisions. The probability of a random policy reaching that reward under standard epsilon-greedy exploration is approximately $\epsilon^{500}$, which for $\epsilon = 0.05$ is on the order of $10^{-649}$. That is smaller than the reciprocal of the number of atoms in the observable universe. The agent will literally never accidentally stumble onto the reward by random chance.

This is not an edge case. It arises naturally in many practically important settings:

- **Games with long prerequisite chains**: Montezuma's Revenge, Pitfall, Private Eye on Atari all require multi-step item collection before the first reward.
- **Robotics tasks with contact**: a robot arm must push a block onto a pressure plate to open a door before it can reach the target object. Each step is a separate skill that must be sequenced correctly.
- **Procedurally generated environments**: NetHack, MiniGrid-MultiRoom, Crafter all change their reward structure every episode, making memorization useless.
- **Language model alignment**: reward models trained to judge overall response quality only fire on complete, coherent responses — not on individual tokens — making credit assignment over the full sequence extremely challenging.
- **Drug discovery and molecule design**: reward (binding affinity to a target protein) only appears after generating a chemically valid, synthetically feasible molecule, which requires correctly composing 30–100 atoms.

The fundamental issue is the *exploration gradient*: without any reward signal, the agent's Q-values or policy gradient carries no information about how to reach the high-reward region. The Q-values are all initialized near zero and stay there, because no gradient has ever nudged them. Adding an auxiliary intrinsic reward that increases when the agent visits novel or surprising states transforms the problem. Instead of a flat reward landscape with one distant peak surrounded by an infinite plateau of zero reward, the agent now sees a dense, varied bonus landscape that rewards genuine exploration. This landscape has a rough inverse relationship to state visitation density, which means it naturally guides the agent away from frequently visited states and toward unexplored territory.

Formally, we augment the total reward as:

$$r_t^{\text{total}} = r_t^{\text{ext}} + \beta \cdot r_t^{\text{int}}$$

where $r_t^{\text{ext}}$ is the environment's extrinsic reward, $r_t^{\text{int}}$ is the curiosity bonus from the agent's internal model, and $\beta \in [0, 1]$ is a weighting hyperparameter that controls the relative importance of exploration versus exploitation. When $r_t^{\text{ext}} \approx 0$ everywhere (pure sparse reward), the $\beta \cdot r_t^{\text{int}}$ term dominates and the agent is essentially solving a curiosity-maximization problem. When $r_t^{\text{ext}}$ is dense, the intrinsic reward provides a supplementary exploration gradient that prevents the agent from getting stuck in local optima.

The key design question is how to compute $r_t^{\text{int}}$ in a way that is dense (fires frequently), informative (correlates with genuine novelty), scalable (works on high-dimensional observations), and computationally cheap (does not dominate the training cost). The four major paradigms each answer this differently.

## The broader RL exploration taxonomy

Before diving into curiosity-specific methods, it is worth situating intrinsic motivation within the broader landscape of exploration strategies. This connects to the [exploration vs exploitation trade-off](/blog/machine-learning/reinforcement-learning/exploration-vs-exploitation-the-core-tension) discussion, which covers the classical bandit formulation. There are fundamentally three categories of exploration strategy:

**Undirected exploration** — add noise to actions or policy parameters without regard to what has been seen. Epsilon-greedy, Boltzmann exploration (softmax over Q-values), Gaussian action noise (DDPG-style), and entropy bonuses (the $\alpha \cdot \mathcal{H}(\pi)$ term in SAC) all fall here. These work in dense-reward settings but fail when the reward is sparse because they waste most exploration budget on already-visited states.

**Directed exploration** — track what has been visited and preferentially visit less-visited states. Count-based methods, UCB, Thompson sampling, and all curiosity-driven methods fall here. They are theoretically superior in sparse-reward settings but require maintaining some model of visitation history, which is expensive for high-dimensional state spaces.

**Posterior-based exploration** — maintain a probability distribution over possible reward/transition functions (Thompson sampling, PSRL) and explore according to the posterior uncertainty. These are optimal in the Bayesian sense but computationally tractable only for small problems.

Curiosity-driven RL is the practical engineering of directed exploration for deep learning — specifically, for settings where the state space is so high-dimensional that exact counts or exact posteriors are impossible.

## Count-based exploration: the naive baseline

The cleanest definition of novelty is visit count. If you have a finite state space $\mathcal{S}$, you maintain a table $N(s)$ counting the number of times each state has been visited, and the intrinsic reward is:

$$r^{\text{int}}(s) = \frac{1}{\sqrt{N(s)}}$$

This decays as a state is visited more frequently and is zero only in the infinite-visit limit. The $1/\sqrt{N}$ scaling is not arbitrary — it comes from the UCB1 algorithm (Auer et al., 2002) for bandit problems, where the optimal exploration bonus for a state with $N$ visits and known variance is $\sqrt{2 \log T / N}$ (where $T$ is the total timestep count). The $\log T$ term grows slowly, so $1/\sqrt{N}$ is a common simplification. It is provably optimal in tabular settings: UCB-style count bonuses achieve $\tilde{O}(\sqrt{SAT})$ regret in finite MDPs, where $S$, $A$, $T$ are states, actions, and total steps.

```python
import numpy as np
from collections import defaultdict

class CountBasedBonus:
    """Tabular count-based intrinsic reward. Works only for small discrete state spaces."""
    def __init__(self, beta: float = 0.1):
        self.counts = defaultdict(int)
        self.beta = beta

    def bonus(self, state) -> float:
        key = tuple(state) if hasattr(state, '__iter__') else state
        self.counts[key] += 1
        return self.beta / np.sqrt(self.counts[key])

    def total_unique_states(self) -> int:
        return len(self.counts)

# Usage in a tabular GridWorld
bonus_fn = CountBasedBonus(beta=0.5)
# Simulate 1000 steps in a 20x20 grid
rng = np.random.default_rng(42)
for _ in range(1000):
    s = rng.integers(0, 20, size=2)  # 20x20 grid, 400 states
    r_int = bonus_fn.bonus(s)
print(f"Unique states visited: {bonus_fn.total_unique_states()} / 400")
# Unique states visited: 341 / 400 — random walk covers 85% of grid
```

The problem is that exact counts do not scale to high-dimensional state spaces. Two Atari frames are almost certainly never literally identical — even a single pixel flicker from atmospheric animation makes them distinct under an exact-count scheme, so $N(s) = 1$ for essentially every frame and the bonus is always the maximum value. You get constant exploration noise, not directed exploration.

The answer is *pseudo-counts*, introduced by Bellemare et al. (2016) in "Unifying Count-Based Exploration and Intrinsic Motivation." The idea is to replace exact counts with generalized counts derived from a density model $\rho_n(s)$ trained on all $n$ observations so far. After seeing a new state $s$, the model density changes from $\rho_n(s)$ to $\rho_{n+1}(s)$. The pseudo-count is defined as:

$$\hat{N}(s) = \frac{\rho_n(s) \cdot (1 - \rho_{n+1}(s))}{\rho_{n+1}(s) - \rho_n(s)}$$

This quantity is high for states similar to ones seen frequently (the density model assigns high probability to them, and observing another similar state does not change the density much) and low for genuinely novel states (the density model assigns low probability, and seeing the state significantly increases the density estimate). The intrinsic reward is then $r^{\text{int}}(s) = (\hat{N}(s) + 0.01)^{-1/2}$.

In practice, pseudo-counts are finicky to tune because the density model choice matters enormously. PixelCNN-based pseudo-counts (Ostrovski et al., 2017) were the best performing but required training a generative model on Atari frames alongside the policy — essentially doubling the training cost. The community moved toward prediction-error methods precisely because they sidestep explicit density estimation.

## ICM: prediction-error curiosity from first principles

The Intrinsic Curiosity Module (Pathak et al., "Curiosity-driven Exploration by Self-Supervised Prediction", ICML 2017) is the paper that made curiosity-driven RL mainstream. Its core insight is elegant: an agent should be curious about states whose *future dynamics it cannot yet predict*. If you can perfectly predict what happens after every action, you are not in novel territory.

The naive version of this idea is: train a neural network $f_\theta(s_t, a_t) \to \hat{s}_{t+1}$ to predict the raw next observation, and use $\|\hat{s}_{t+1} - s_{t+1}\|^2$ as the curiosity bonus. This fails catastrophically. Raw pixel space is high-dimensional and full of irrelevant variation — atmospheric haze, texture noise, lighting changes, background motion — that has nothing to do with the action-relevant aspects of the environment. The agent would spend all its time staring at flickering torches, because the next torch state is genuinely unpredictable pixel-by-pixel but carries zero information about whether the agent is exploring.

ICM's fix is to learn a feature space $\phi(s)$ that discards irrelevant variation by training it jointly with an *inverse dynamics model*: a network that predicts the action $a_t$ taken from the feature pair $(\phi(s_t), \phi(s_{t+1}))$. The key insight is: if you can infer the action from consecutive feature vectors, the features must capture the action-relevant aspects of the state transition. Everything the inverse model cannot use — background clouds, flickering torches, the random noise on a wall texture — is implicitly discarded from the feature representation.

![ICM architecture dataflow showing how obs_t and obs_t+1 both enter the CNN encoder, then split to the inverse dynamics head and forward dynamics head, with the forward prediction error becoming intrinsic reward](/imgs/blogs/intrinsic-motivation-curiosity-driven-rl-3.png)

Formally, ICM trains three networks simultaneously:

1. **Encoder** $\phi_\theta: \mathcal{S} \to \mathbb{R}^d$ — maps observations to $d$-dimensional feature vectors (typically $d = 512$). For Atari, this is a CNN. For low-dimensional control, a multi-layer perceptron suffices.
2. **Inverse model** $g_\psi: \mathbb{R}^d \times \mathbb{R}^d \to \mathcal{A}$ — predicts action from consecutive features: $\hat{a}_t = g_\psi(\phi(s_t), \phi(s_{t+1}))$.
3. **Forward model** $f_\phi: \mathbb{R}^d \times \mathcal{A} \to \mathbb{R}^d$ — predicts next feature from current feature and action: $\hat{\phi}_{t+1} = f_\phi(\phi(s_t), a_t)$.

The total ICM loss is:

$$\mathcal{L}_{\text{ICM}} = \alpha \mathcal{L}_{\text{inv}} + (1-\alpha)\mathcal{L}_{\text{fwd}}$$

where $\mathcal{L}_{\text{inv}} = \text{CE}(g_\psi(\phi(s_t), \phi(s_{t+1})), a_t)$ (cross-entropy for discrete actions, MSE for continuous) and $\mathcal{L}_{\text{fwd}} = \|\hat{\phi}_{t+1} - \phi(s_{t+1})\|^2$.

The intrinsic reward is the scaled forward model error:

$$r_t^{\text{int}} = \frac{\eta}{2}\|\hat{\phi}_{t+1} - \phi(s_{t+1})\|^2$$

where $\eta$ is a scaling constant. High forward error means the agent is in a state it has not seen before, so its dynamics model is uncertain — high curiosity bonus. As the agent revisits familiar states, the forward model improves and the bonus decreases.

Here is a clean PyTorch implementation of ICM:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ICMEncoder(nn.Module):
    """Shared CNN encoder for ICM. Maps 4-frame 84x84 grayscale to 512-dim features."""
    def __init__(self, in_channels: int = 4):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 8, stride=4), nn.ELU(),
            nn.Conv2d(32, 64, 4, stride=2),           nn.ELU(),
            nn.Conv2d(64, 64, 3, stride=1),            nn.ELU(),
        )
        self.fc = nn.Linear(64 * 7 * 7, 512)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, 84, 84), values in [0,1]
        h = self.conv(x.float() / 255.0)
        return F.relu(self.fc(h.flatten(1)))


class ICMModule(nn.Module):
    """
    Intrinsic Curiosity Module (Pathak et al. 2017).
    Returns intrinsic reward and the combined ICM loss for backprop.
    """
    def __init__(
        self,
        feature_dim: int = 512,
        n_actions: int = 18,
        alpha: float = 0.1,
        eta: float = 0.01,
    ):
        super().__init__()
        self.encoder = ICMEncoder()
        self.alpha = alpha
        self.eta = eta

        # Inverse model: (phi_t, phi_{t+1}) -> action logits
        self.inverse = nn.Sequential(
            nn.Linear(feature_dim * 2, 256), nn.ReLU(),
            nn.Linear(256, n_actions),
        )
        # Forward model: (phi_t, one_hot_action) -> phi_{t+1}
        self.forward_model = nn.Sequential(
            nn.Linear(feature_dim + n_actions, 256), nn.ReLU(),
            nn.Linear(256, feature_dim),
        )
        self.n_actions = n_actions

    def forward(
        self,
        obs_t: torch.Tensor,
        obs_t1: torch.Tensor,
        actions: torch.Tensor,
    ):
        phi_t  = self.encoder(obs_t)
        phi_t1 = self.encoder(obs_t1)

        # --- Inverse model loss ---
        inv_input = torch.cat([phi_t, phi_t1], dim=-1)
        action_logits = self.inverse(inv_input)
        loss_inv = F.cross_entropy(action_logits, actions.long())

        # --- Forward model ---
        a_onehot = F.one_hot(actions.long(), self.n_actions).float()
        # Detach phi_t so forward model gradient does not flow through encoder
        fwd_input = torch.cat([phi_t.detach(), a_onehot], dim=-1)
        phi_pred = self.forward_model(fwd_input)

        # Forward loss = mean squared error in feature space
        loss_fwd = F.mse_loss(phi_pred, phi_t1.detach(), reduction='none').sum(dim=-1)
        r_int = (self.eta / 2.0) * loss_fwd.detach()

        # Combined ICM loss (both models)
        icm_loss = self.alpha * loss_inv + (1.0 - self.alpha) * loss_fwd.mean()
        return r_int, icm_loss
```

The key design choice is `.detach()` on the encoder output when computing the forward model loss. Without it, the encoder would minimize the forward prediction error by collapsing all features to the same constant vector — making prediction trivially easy but completely uninformative. The inverse model gradient is what keeps the encoder honest: it forces the features to retain enough information to reconstruct the action, which prevents the collapse.

One practical pitfall: if your action space is large (e.g., 18 discrete Atari actions), the inverse model is trained on a 36-class classification problem. With a small batch size of 32, the class imbalance (most frames involve "do nothing" or directional actions) can destabilize the inverse model. Using a larger batch size (256+) or weighting the cross-entropy loss by inverse class frequency helps.

## RND: curiosity without action conditioning

ICM is elegant but has two practical issues. First, the inverse model architecture adds complexity and a second hyperparameter ($\alpha$) to tune. Second, and more fundamentally, the action-conditioned forward prediction fails in stochastic environments: where the same action from the same state leads to different outcomes, the forward model can never perfectly predict the next feature, so it always generates high curiosity bonus even for states the agent has visited thousands of times. The agent mistakes environmental stochasticity for novelty.

Random Network Distillation (Burda et al., "Exploration by Random Network Distillation", ICLR 2019) solves both issues with a beautiful simplification: forget learning the dynamics model entirely. Instead, use two fixed architectures — a *target* network with frozen random weights $f: \mathcal{S} \to \mathbb{R}^d$ and a *predictor* network $\hat{f}_\theta: \mathcal{S} \to \mathbb{R}^d$ that is trained online to match the target. The curiosity bonus is simply:

$$r_t^{\text{int}} = \|f(s_t) - \hat{f}_\theta(s_t)\|^2$$

The target network is initialized randomly at the start of training and then frozen forever. Its weights are never updated. Its role is to define a fixed but complex, nonlinear function of the observation. The predictor starts with different random weights and is trained by gradient descent to imitate the target on all observations the agent encounters. For states seen frequently, the predictor converges to match the target, so the error is low. For novel states, the predictor has not received enough gradient updates pointing toward $f(s)$, so the error remains high.

```python
import torch
import torch.nn as nn
import numpy as np

class RNDModule(nn.Module):
    """
    Random Network Distillation (Burda et al. 2019).
    Target is frozen; predictor is updated online to track target output.
    Prediction error is the intrinsic curiosity reward.
    """
    def __init__(self, obs_dim: int = 512, embed_dim: int = 512, hidden: int = 512):
        super().__init__()
        # Target: frozen random network, never trained
        self.target = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),  nn.ReLU(),
            nn.Linear(hidden, embed_dim),
        )
        for p in self.target.parameters():
            p.requires_grad = False        # NEVER update target

        # Predictor: trained to match target output
        self.predictor = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),  nn.ReLU(),
            nn.Linear(hidden, embed_dim),
        )

    def forward(self, obs_enc: torch.Tensor):
        """
        Args:
            obs_enc: (B, obs_dim) encoded observation (e.g. from shared CNN).
        Returns:
            r_int: (B,) intrinsic reward (detached from computation graph).
            loss:  scalar predictor MSE loss for backprop.
        """
        with torch.no_grad():
            target_out = self.target(obs_enc)
        pred_out = self.predictor(obs_enc)
        error = (pred_out - target_out).pow(2).sum(dim=-1)
        return error.detach(), error.mean()


class RNDObsEncoder(nn.Module):
    """Observation pre-processor: encodes single Atari frames for RND."""
    def __init__(self, in_channels: int = 1):  # single grayscale frame
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 8, stride=4), nn.LeakyReLU(),
            nn.Conv2d(32, 64, 4, stride=2),           nn.LeakyReLU(),
            nn.Conv2d(64, 64, 3, stride=1),            nn.LeakyReLU(),
        )
        self.fc = nn.Linear(64 * 7 * 7, 512)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x.float() / 255.0)
        return self.fc(h.flatten(1))
```

A critical implementation detail: RND typically processes each single observation frame (not 4-frame stacks) for the novelty signal, because you want frame-level novelty detection. The policy's CNN can still use 4-frame stacks for action selection — this asymmetry is intentional. The RND target defines novelty in single-frame observation space, while the policy reasons about short temporal sequences. Using stacked frames for RND would penalize any repeated consecutive frame pair, including fast action sequences the agent has seen before.

Another mandatory detail: you must normalize the intrinsic reward using a running mean and variance estimate. Raw MSE values vary enormously across environments and training stages — early in training when the predictor is far from the target, every observation generates a large bonus; later in training the predictor improves. Without normalization, the effective $\beta$ changes throughout training in an uncontrolled way.

```python
class RunningMeanStd:
    """Welford online algorithm for running mean and variance. Thread-safe for single env."""
    def __init__(self, epsilon: float = 1e-4, shape=()):
        self.mean  = np.zeros(shape, dtype=np.float64)
        self.var   = np.ones(shape,  dtype=np.float64)
        self.count = epsilon

    def update(self, x: np.ndarray):
        batch_mean  = x.mean(axis=0)
        batch_var   = x.var(axis=0)
        batch_count = x.shape[0]
        delta       = batch_mean - self.mean
        total       = self.count + batch_count
        self.mean   = self.mean + delta * batch_count / total
        m_a         = self.var  * self.count
        m_b         = batch_var * batch_count
        self.var    = (m_a + m_b + delta**2 * self.count * batch_count / total) / total
        self.count  = total

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / (np.sqrt(self.var) + 1e-8)
```

## Deep-dive: why ICM's feature collapse problem matters

Before moving to RND, it is worth spending more time on the feature collapse failure mode in ICM, because it appears in subtly different forms throughout curiosity research and explains many real training failures.

The formal statement is: if the encoder $\phi$ is trained purely by the forward model loss $\mathcal{L}_{\text{fwd}} = \|\hat{\phi}_{t+1} - \phi(s_{t+1})\|^2$ without the inverse model constraint, the encoder can minimize this loss by mapping all states to the same constant vector $\phi(s) = c$. The forward model then trivially learns the identity $\hat{\phi}_{t+1} = c$ with zero prediction error. The curiosity bonus vanishes everywhere, and the agent degenerates to undirected exploration.

The inverse model prevents this by requiring that $\phi$ retain enough information to infer the action $a_t$ from $(\phi(s_t), \phi(s_{t+1}))$. If the encoder collapses to a constant, the inverse model cannot distinguish actions, so its loss $\mathcal{L}_{\text{inv}}$ is high. The gradient from $\mathcal{L}_{\text{inv}}$ flows back into the encoder and forces it to retain action-discriminative features.

However, this only prevents *complete* collapse. A subtler failure is *partial collapse*: the encoder retains just enough information to satisfy the inverse model (it can distinguish actions) but discards all action-irrelevant environmental variation. This is actually what ICM *intends* — you want to ignore background clutter. But in practice, if the action set is small relative to the state complexity, the encoder may collapse more than intended, losing novelty signal that comes from action-independent state changes (e.g., a new room layout that does not directly change which actions are available).

This is why RND's completely action-agnostic approach sometimes outperforms ICM in practice: it does not need to solve the feature-learning problem at all. The frozen target network is pre-defined, so there is nothing to collapse. The only network being trained is the predictor, which is supervised by a stable, fixed target.

The lesson for practitioners: if you notice the ICM curiosity bonus going to zero very quickly (within 1–2M steps on Atari, where you would expect exploration to still be valuable), check whether the inverse model is trivially satisfied with a low-capacity encoder. Adding a stop-gradient before the inverse model on the encoder's output (so the inverse model trains the encoder, but the forward model does not receive gradients from the encoder) can help in some architectures, at the cost of slower feature learning.

## VIME: variational information maximizing exploration

VIME (Houthooft et al., "VIME: Variational Information Maximizing Exploration", NeurIPS 2016) frames curiosity as information gain about the environment's dynamics model. The agent maintains a Bayesian neural network $p(\theta | \mathcal{D})$ over the parameters of a dynamics model $p(s_{t+1} | s_t, a_t; \theta)$, and the intrinsic reward is the information gain from the new transition $(s_t, a_t, s_{t+1})$:

$$r^{\text{int}}_t = D_{\text{KL}}\big(p(\theta | \mathcal{D}_t \cup \{(s_t, a_t, s_{t+1})\}) \| p(\theta | \mathcal{D}_t)\big)$$

This is intellectually satisfying: an agent that maximizes information gain about its own world model is doing science. It directly measures how much the new observation *updates the agent's beliefs* about how the environment works. States that confirm already-known dynamics get low reward; states that reveal new dynamics get high reward. It is provably optimal in the Bayes-optimal exploration sense.

In practice, VIME approximates the KL divergence using variational inference with a mean-field diagonal Gaussian posterior $q(\theta; \phi) = \prod_i \mathcal{N}(\theta_i; \mu_i, \sigma_i^2)$. After observing a new transition, the information gain is estimated as:

$$r^{\text{int}}_t \approx D_{\text{KL}}\big(q(\theta; \phi') \| q(\theta; \phi)\big)$$

where $\phi'$ is the updated variational parameters after one gradient step on the new transition, and $\phi$ is the parameters before. This KL has a closed form for diagonal Gaussians:

$$D_{\text{KL}}\big(\mathcal{N}(\mu', \sigma'^2) \| \mathcal{N}(\mu, \sigma^2)\big) = \sum_i \left[\log\frac{\sigma_i}{\sigma_i'} + \frac{\sigma_i'^2 + (\mu_i' - \mu_i)^2}{2\sigma_i^2} - \frac{1}{2}\right]$$

The limitation is computational: maintaining and updating a Bayesian neural network at every step is expensive. The variational update requires a forward pass, backward pass, and parameter update for the dynamics model at every single environment step — which is in addition to the policy gradient update. VIME is practical for small MLPs (a few hundred parameters) on low-dimensional state vectors, but scales poorly to CNN-sized models for image-based environments.

VIME remains theoretically important because it establishes information gain as the ground truth objective that ICM and RND are implicitly approximating. ICM approximates information gain about the forward dynamics model; RND approximates information gain about a random target function. Understanding VIME helps you reason about the failure modes of both: ICM fails when the inverse model is misspecified; RND fails when the predictor can generalize across visually similar but dynamically distinct states.

## The information-gain framework: a unified view

All curiosity methods can be understood as different approximations to a single core objective: the agent should seek states where its internal model has the most uncertainty. This is the *principle of information maximization*, and it unifies the methods in a single equation.

Let $\mathcal{M}$ be the agent's internal model of the environment. The agent's uncertainty about $\mathcal{M}$ at state $s$ is some measure $U(\mathcal{M}, s)$. The intrinsic reward is:

$$r^{\text{int}}(s) \propto U(\mathcal{M}, s)$$

Different methods instantiate $U$ differently:

| Method | Uncertainty measure $U(\mathcal{M}, s)$ | Scales to images? | Stochastic envs? | Ease of impl. |
|--------|----------------------------------------|-------------------|-----------------|---------------|
| Count-based | $N(s)^{-1/2}$ | No (exact counts) | Yes | Easy |
| Pseudo-counts | $\hat{N}(s)^{-1/2}$ via density model | Partially | Yes | Hard |
| ICM | Forward model error in feature space | Yes | Partially | Medium |
| RND | Predictor-target MSE | Yes | Yes | Easy |
| VIME | KL over BNN parameters | No (too slow) | Yes | Hard |
| NGU (episodic) | k-NN distance in embedding space | Yes | Yes | Medium |

The information-theoretic lens also reveals why RND works despite having nothing to do with dynamics. The frozen target network defines a complex, fixed function of the observation space. The predictor's inability to match it on novel states is a direct consequence of having received too few gradient updates pointing toward $f(s)$ near that state. This is an implicit empirical estimate of observation density: if many observations near $s$ have been seen, the predictor has been repeatedly trained to output $f(s)$ for nearby inputs and generalizes to $s$; if few observations near $s$ have been seen, the predictor has not been pushed toward $f(s)$ and the error is large.

## RND in practice: initialization, normalization, and common mistakes

Before writing the full SB3 integration, let me cover the initialization and normalization details that the original RND paper treats as critical but that most tutorial code omits.

**Observation normalization before the target network.** The target network maps raw observations to output embeddings. If observations are in the range [0, 255] (typical for Atari frames) and the target network uses ReLU activations, the first-layer outputs will be enormously large, making the target's output magnitudes huge and inconsistent across different observation types. The standard fix is to normalize observations to approximately zero mean and unit variance before feeding them to the RND target and predictor. This normalization uses a running mean and variance estimated from early rollouts (typically the first 100–1000 steps of each environment before the predictor training begins).

**Separate observation normalization for the policy and for RND.** The policy network (whether DQN or PPO) often uses 4-stacked grayscale frames with values in [0, 255], dividing by 255 to get [0, 1]. The RND module uses only the current single frame but normalizes it to approximately [-1, 1] using the running statistics. These are different preprocessing pipelines and should not share the same normalization statistics.

**Reward normalization is mandatory, not optional.** The raw MSE output of the RND predictor varies by orders of magnitude: at the start of training when the predictor is essentially random, every observation produces a large MSE (perhaps 100–1000 in raw units). After a few million steps, the predictor has improved and typical MSEs drop to 1–10. Without normalization, the effective β changes by a factor of 100× over training, which completely changes the exploration-exploitation balance. Always maintain a running `RunningMeanStd` over the intrinsic reward values and normalize by the running standard deviation (not the full z-score — dividing by std is sufficient; do not subtract the mean, because you want the absolute curiosity to remain positive).

**Initializing the predictor differently from the target.** The target and predictor should be initialized with different random seeds. If they start with the same weights, the initial prediction error is zero everywhere, and the early training signal is noise. In PyTorch, this is handled naturally if you construct them separately (different `nn.Module` instances get different random initializations), but explicitly seeding them differently is cleaner:

```python
import torch
import torch.nn as nn

def build_rnd_pair(obs_dim: int = 512, embed_dim: int = 512) -> tuple:
    """Build (target, predictor) with guaranteed different initializations."""
    torch.manual_seed(0)
    target = nn.Sequential(
        nn.Linear(obs_dim, 512), nn.ReLU(),
        nn.Linear(512, 512),    nn.ReLU(),
        nn.Linear(512, embed_dim),
    )
    # Freeze target
    for p in target.parameters():
        p.requires_grad = False

    torch.manual_seed(1)   # DIFFERENT seed for predictor
    predictor = nn.Sequential(
        nn.Linear(obs_dim, 512), nn.ReLU(),
        nn.Linear(512, 512),    nn.ReLU(),
        nn.Linear(512, embed_dim),
    )
    return target, predictor
```

**Handling multiple parallel environments.** When using `n_envs = 128` parallel environments (standard for hard-exploration Atari), the running statistics for reward normalization are computed over all 128 × n_steps samples per rollout. The `RunningMeanStd.update()` call should receive the full flattened array of intrinsic rewards, not per-environment averages, so that the variance estimate reflects the full range of novelty across all parallel environments.

**The proportion of time spent updating the predictor.** In the Burda et al. (2019) implementation, only 25% of the parallel environments' data is used for predictor updates per rollout (randomly sampled). This prevents the predictor from overfitting too quickly to recently seen states, maintaining a slightly higher intrinsic reward for states that were visited earlier in the rollout. This trick is easy to implement: mask out 75% of the observations before computing the predictor loss.

```python
# Subsample 25% of observations for predictor update (Burda et al. 2019)
mask = torch.rand(obs_t.shape[0], device=device) < 0.25
if mask.sum() > 0:
    enc_sub = self.enc(obs_t[mask])
    _, pred_loss = self.rnd(enc_sub)
    self.opt.zero_grad()
    pred_loss.backward()
    torch.nn.utils.clip_grad_norm_(
        list(self.rnd.predictor.parameters()) + list(self.enc.parameters()),
        max_norm=0.5,
    )
    self.opt.step()
```

## Wiring curiosity into PPO: the SB3 custom callback

Now for the implementation. We will add an RND curiosity module to PPO via a Stable-Baselines3 custom callback. This approach is practical because it keeps the curiosity computation outside of SB3's core training loop, letting us modify the rollout buffer before the policy update step.

```python
import numpy as np
import torch
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

class RNDCuriosityCallback(BaseCallback):
    """
    Augments PPO rollout rewards with RND intrinsic bonus.
    Hooks into _on_rollout_end to modify the rollout buffer's rewards
    before the policy gradient update computes advantages.
    """
    def __init__(
        self,
        rnd_module: RNDModule,
        rnd_encoder: RNDObsEncoder,
        rnd_optimizer: torch.optim.Optimizer,
        beta: float = 0.01,
        device: str = "cuda",
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.rnd    = rnd_module.to(device)
        self.enc    = rnd_encoder.to(device)
        self.opt    = rnd_optimizer
        self.beta   = beta
        self.device = device
        self.reward_rms = RunningMeanStd()
        self.obs_rms    = RunningMeanStd(shape=(1, 84, 84))

    def _on_rollout_end(self) -> None:
        """Called after rollout collection, before advantage computation."""
        buf    = self.model.rollout_buffer
        obs_np = buf.observations.copy()           # (n_steps, n_envs, *obs_shape)
        n_steps, n_envs = obs_np.shape[:2]

        # Use only the last (most recent) channel for RND novelty detection
        obs_flat   = obs_np.reshape(-1, *obs_np.shape[2:])  # (N, C, H, W)
        obs_single = obs_flat[:, -1:, :, :]                  # (N, 1, H, W)

        # Normalize observations (RND is sensitive to input scale)
        self.obs_rms.update(obs_single)
        obs_norm = np.clip(self.obs_rms.normalize(obs_single), -5.0, 5.0)
        obs_t    = torch.tensor(obs_norm, dtype=torch.float32, device=self.device)

        # Compute intrinsic reward in mini-batches to avoid GPU OOM
        intrinsic = []
        with torch.no_grad():
            for start in range(0, obs_t.shape[0], 256):
                enc_out = self.enc(obs_t[start:start+256] * 255.0)  # denorm for conv
                r_int, _ = self.rnd(enc_out)
                intrinsic.append(r_int.cpu().numpy())
        r_int_np = np.concatenate(intrinsic).reshape(n_steps, n_envs)

        # Normalize intrinsic reward across rollout
        self.reward_rms.update(r_int_np.flatten())
        r_int_normalized = self.reward_rms.normalize(r_int_np)

        # Train RND predictor on this rollout's observations
        self.opt.zero_grad()
        enc_full = self.enc(obs_t * 255.0)
        _, pred_loss = self.rnd(enc_full)
        pred_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.rnd.predictor.parameters()) + list(self.enc.parameters()),
            max_norm=0.5,
        )
        self.opt.step()

        # Inject curiosity bonus into rollout buffer before advantage computation
        buf.rewards += self.beta * r_int_normalized

    def _on_step(self) -> bool:
        return True   # always continue


def make_atari_env(game: str = "ALE/MontezumaRevenge-v5", seed: int = 0):
    from stable_baselines3.common.atari_wrappers import AtariWrapper
    def _env():
        e = gym.make(game)
        return AtariWrapper(e)
    return DummyVecEnv([_env])


# Assemble the full training setup
device = "cuda" if torch.cuda.is_available() else "cpu"
enc  = RNDObsEncoder(in_channels=1)
rnd  = RNDModule(obs_dim=512, embed_dim=512)
opt  = torch.optim.Adam(
    list(rnd.predictor.parameters()) + list(enc.parameters()),
    lr=1e-4,
)
cb   = RNDCuriosityCallback(rnd, enc, opt, beta=0.01, device=device)
env  = make_atari_env(seed=0)

model = PPO(
    "CnnPolicy", env,
    n_steps=128,          # steps per rollout per env
    batch_size=256,
    n_epochs=4,
    gamma=0.999,          # long discount for multi-room planning
    learning_rate=2.5e-4,
    ent_coef=0.001,       # small entropy bonus still helps
    verbose=1,
    device=device,
)
model.learn(total_timesteps=10_000_000, callback=cb)
```

A few implementation details worth emphasizing. The `buf.rewards += self.beta * r_int_normalized` line modifies the rollout buffer in place before SB3 calls `compute_returns_and_advantage()`. This means the full advantage estimate, including the generalized advantage estimator (GAE), incorporates the curiosity signal. The policy gradient will include gradients from both extrinsic and intrinsic reward.

The `gamma=0.999` is higher than the standard 0.99 used for most Atari environments. This is deliberate: Montezuma's Revenge requires very long credit assignment (the key in room 0 enables the door that leads to room 1 that eventually leads to a reward dozens of steps later). A discount factor of 0.99 gives only $0.99^{100} \approx 0.37$ weight to a reward 100 steps in the future; with $\gamma = 0.999$ this becomes $0.999^{100} \approx 0.90$.

Gradient clipping at max_norm=0.5 on the RND predictor is important. Novel states generate large prediction errors, which generate large gradients. Occasional extreme states (particularly at the start of training when everything is novel) can produce gradient explosions that destabilize the predictor and create anomalous curiosity spikes.

## RND: the mathematical argument for why it works

Let me spell out formally why the distillation error is a valid novelty measure. The claim is: a state visited many times will have low predictor error; a state visited rarely will have high error. Here is the argument.

The predictor $\hat{f}_\theta$ minimizes the empirical risk:

$$\mathcal{L}(\theta) = \mathbb{E}_{s \sim \mathcal{D}}[\|f(s) - \hat{f}_\theta(s)\|^2]$$

where $\mathcal{D}$ is the empirical distribution over visited states. The minimizer over all functions is $\hat{f}^*(s) = f(s)$ (the target function itself), because the target is deterministic. The predictor approximates this minimizer via gradient descent.

For a state $s$ visited many times (high density $\mathcal{D}(s)$), many gradient steps have pushed $\hat{f}_\theta(s)$ toward $f(s)$. The predictor's function value near $s$ is well-determined by many data points, so interpolation error is low. For a state $s$ visited rarely (low density $\mathcal{D}(s)$), few gradient steps have been taken near $s$, so the predictor's value is essentially random (initialized randomly, barely updated). The error $\|f(s) - \hat{f}_\theta(s)\|^2$ is large.

More precisely, if we model the predictor as a kernel estimator with bandwidth $h$, the bias term near $s$ is proportional to $1/\hat{N}_h(s)$, where $\hat{N}_h(s)$ counts observations within distance $h$ of $s$ weighted by the kernel. This is exactly the pseudo-count formula from Bellemare et al. (2016). RND is doing implicit kernel density estimation of the visitation distribution, using the neural network's generalization properties as the kernel.

This connection also explains when RND fails: if the predictor generalizes too broadly (has too few parameters, or is trained on too few examples of diverse states before reaching a novel state), it can learn to approximate $f$ everywhere through general extrapolation rather than specifically at visited states. The error is uniformly low. The fix is to use a large enough predictor (matching the capacity of the target network) and to not pretrain the predictor excessively before deployment.

## The noisy-TV problem: when curiosity destroys itself

The most famous failure mode of curiosity-driven RL is the *noisy-TV problem*, systematically studied by Burda et al. (2018) in "Large-Scale Study of Curiosity-Driven Learning." The setup: if the environment contains a source of stochastic, unpredictable variation that the agent can observe — a television displaying random noise, a flag blowing in the wind, water rippling, an NPC with random behavior — the curiosity module generates permanently high intrinsic reward for watching that variation.

From the curiosity module's perspective, this is rational. The TV displays truly random pixels. The forward model (in ICM) can never predict them perfectly, because even a perfect model of the environment cannot predict genuinely random outputs. The prediction error stays permanently high. The agent "discovers" that staring at the TV generates maximum curiosity bonus and does exactly that, ignoring the actual task reward indefinitely.

This is not a pathological edge case. It arises in:

- **Minecraft**: flowing water and lava produce permanent high curiosity for ICM-based agents.
- **VizDoom**: flickering torch textures near doorways.
- **Stochastic multi-agent environments**: other agents' random-policy actions appear as uncontrollable stochasticity.
- **Real robotic environments**: camera sensor noise on low-cost cameras; vibrations in the robot base; shadows from overhead lighting that change unpredictably.
- **Procedurally generated environments**: the randomly generated elements (map layout, item placement, enemy behavior) cannot be predicted no matter how much the agent has explored.

The fundamental reason is that curiosity as formalized confounds two different types of uncertainty:

- **Epistemic uncertainty**: the agent has not yet visited this state, so it lacks data about the dynamics. This is genuine novelty that is reducible by exploration.
- **Aleatoric uncertainty**: the environment is inherently stochastic, so the outcome is unpredictable regardless of data. This is irreducible noise. A rational agent should not keep exploring it.

ICM suffers severely from the noisy-TV problem because the forward model cannot predict stochastic outputs even with infinite data. RND is more robust: the frozen target network maps stochastic inputs to a deterministic (though noise-affected) output, and the predictor can learn to track this deterministic mapping even when the input contains random variation. However, RND is not immune when the agent can actively navigate to a region that is permanently novel (a randomly spawning enemy, an endlessly random level segment).

Several defenses are available:

**Episodic curiosity bonuses (Savinov et al., 2019)**: track novelty within an episode only, resetting the episode memory between episodes. Within an episode, a state seen three times gets zero bonus on the fourth visit, even if it was genuinely unseen in prior episodes. This prevents the agent from fixating on a noisy-TV within an episode but still drives cross-episode exploration.

```python
class EpisodicBonus:
    """
    Short-term episodic curiosity: bonus decays with in-episode visit count.
    Combine with long-term RND bonus for the Never-Give-Up (NGU) strategy.
    """
    def __init__(self, k: int = 10, kernel_eps: float = 1e-4):
        self.k   = k
        self.eps = kernel_eps
        self.buffer: list = []    # feature vectors from this episode

    def reset(self):
        self.buffer.clear()

    def bonus(self, feat: np.ndarray) -> float:
        """Returns β_episodic in [0, 1] based on k-NN distance to buffer."""
        if len(self.buffer) < self.k:
            self.buffer.append(feat.copy())
            return 1.0
        feats = np.stack(self.buffer)
        dists = np.linalg.norm(feats - feat, axis=-1)
        dists_sorted = np.sort(dists)[:self.k]
        # Kernel: similarity decreases as k-NN distances shrink
        d_sq = dists_sorted**2 / (dists_sorted.mean() + self.eps)
        ker  = np.exp(-d_sq)
        similarity = np.sqrt(ker.sum() + self.eps)
        self.buffer.append(feat.copy())
        return 1.0 / similarity   # high for genuinely novel states
```

**Never-Give-Up (NGU, Badia et al., 2020)**: combine the episodic short-term bonus with the long-horizon RND bonus, weighting them multiplicatively:

$$r^{\text{int}}_t = r^{\text{episodic}}_t \cdot \min(\max(r^{\text{RND}}_t, 1), L)$$

The episodic bonus decays within an episode; the RND bonus provides long-term novelty signal. The multiplication means that if either component is zero, the total is zero — the agent will not revisit states it has exhaustively covered within the current episode (episodic gate) and will not spend time in globally familiar regions (RND gate).

**Model ensemble disagreement**: instead of a single forward model, train an ensemble of $k$ models with different random initializations. The intrinsic reward is the variance of their predictions:

$$r^{\text{int}}(s, a) = \text{Var}_{i=1}^k\left[f_{\theta_i}(s, a)\right]$$

Ensemble variance naturally decomposes into epistemic uncertainty (reducible by data) and aleatoric uncertainty (irreducible). For a stochastic TV, all ensemble members will eventually learn to output the same "expected" output (the mean of the distribution), and their predictions will converge to the same value — the variance drops to zero for the noisy-TV. For a genuinely unvisited state, the models disagree because they were trained on different random initializations and have not been updated with data near that state.

## Combining curiosity with SAC for continuous control

So far the examples have used PPO, a discrete-action on-policy algorithm well suited to Atari. For continuous control tasks — MuJoCo locomotion, robotic manipulation, autonomous driving — Soft Actor-Critic (SAC) is the preferred base algorithm because it handles continuous actions natively and is substantially more sample-efficient.

The integration with SAC differs from PPO because SAC is off-policy: it maintains a large replay buffer and samples mini-batches from it for updates. We need to store the intrinsic reward alongside the extrinsic reward at collection time, before it enters the buffer.

```python
import gymnasium as gym
from stable_baselines3 import SAC
import numpy as np
import torch

class CuriositySAC:
    """
    SAC with RND curiosity bonus for sparse-reward continuous control.
    Computes intrinsic bonus at collection time and stores augmented reward.
    """
    def __init__(
        self,
        env_id: str,
        beta: float = 0.005,
        device: str = "cuda",
    ):
        self.env    = gym.make(env_id)
        obs_dim     = self.env.observation_space.shape[0]

        # RND for low-dimensional observations (MuJoCo has 17-378 dim obs)
        self.rnd    = RNDModule(obs_dim=obs_dim, embed_dim=256, hidden=256).to(device)
        self.rnd_opt = torch.optim.Adam(self.rnd.predictor.parameters(), lr=3e-4)
        self.beta   = beta
        self.reward_rms = RunningMeanStd()
        self.device = device

        self.model  = SAC(
            "MlpPolicy", self.env,
            learning_rate=3e-4,
            buffer_size=1_000_000,
            learning_starts=10_000,
            batch_size=256,
            verbose=0,
            device=device,
        )

    def collect_and_train(self, total_steps: int = 1_000_000):
        obs, _ = self.env.reset()
        for step in range(total_steps):
            action, _ = self.model.predict(obs, deterministic=False)
            next_obs, r_ext, done, trunc, _ = self.env.step(action)

            # Compute RND intrinsic bonus for this single transition
            obs_t   = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            r_int, pred_loss = self.rnd(obs_t)

            # Update predictor on this observation
            self.rnd_opt.zero_grad()
            pred_loss.backward()
            self.rnd_opt.step()

            # Normalize and combine rewards
            self.reward_rms.update(np.array([r_int.item()]))
            r_int_norm = float(self.reward_rms.normalize(np.array([r_int.item()]))[0])
            r_total    = r_ext + self.beta * r_int_norm

            # Store augmented transition in SAC's replay buffer
            self.model.replay_buffer.add(
                obs, next_obs, action, r_total,
                float(done or trunc), [{}],
            )
            obs = next_obs if not (done or trunc) else self.env.reset()[0]

            # SAC gradient updates (off-policy, every step after warmup)
            if step > self.model.learning_starts:
                self.model.train(gradient_steps=1, batch_size=256)
```

For dense-reward continuous control environments (HalfCheetah, Ant, Walker2d), curiosity bonuses typically provide marginal benefit and the β hyperparameter should be kept small (β ≤ 0.001) to avoid interfering with the well-shaped extrinsic reward. The scenario where SAC + curiosity provides clear value is sparse-reward continuous control: manipulation tasks with binary goal-completion reward (did the block reach the target position?), navigation with a hidden goal, or keyframe-based reward (reward only when specific body configurations are achieved).

## Evaluating and debugging curiosity modules

Curiosity modules introduce several new failure modes that do not appear in standard RL training. Knowing how to detect and diagnose them is as important as knowing how to implement the module.

**Symptom: intrinsic reward rises but extrinsic return stays at zero.** This often indicates the noisy-TV problem or that the extrinsic reward requires a skill the curiosity bonus cannot discover (e.g., the goal requires executing a precise motor skill that random exploration never approximates). Diagnostic: log both the intrinsic and extrinsic return separately. If intrinsic return keeps rising while extrinsic stays flat for more than 5M steps, consider increasing β (to drive more aggressive exploration) or switching to an episodic bonus.

**Symptom: intrinsic reward collapses to near-zero within 1–2M steps.** The predictor has converged on the visited state distribution. This is actually the *correct* behavior if the agent has genuinely explored most of the reachable state space, but it can also indicate that the predictor has too much capacity relative to the target (it overfits and achieves low error even on novel states through aggressive extrapolation). Diagnostic: compute the intrinsic reward on a held-out set of randomly sampled states not from the agent's trajectory — if the error is also near-zero on these random states, the predictor is overfitting. Fix: reduce predictor capacity or increase the learning rate.

**Symptom: training is highly unstable, loss curves oscillate wildly.** The curiosity bonus is too large relative to the extrinsic reward, causing PPO's advantage estimates to swing dramatically. Fix: reduce β, increase the normalization strength of the running reward statistics, or reduce the predictor's learning rate.

**Standard logging practice:** always log these quantities per training step:

```python
# Logging RND diagnostics in TensorBoard via SB3
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("runs/rnd_ppo_montezuma")

# In the callback's _on_rollout_end:
writer.add_scalar("curiosity/mean_r_int",    r_int_np.mean(),    self.num_timesteps)
writer.add_scalar("curiosity/std_r_int",     r_int_np.std(),     self.num_timesteps)
writer.add_scalar("curiosity/predictor_loss", pred_loss.item(),  self.num_timesteps)
writer.add_scalar("train/mean_r_ext",
                  self.model.rollout_buffer.rewards.mean() - self.beta * r_int_normalized.mean(),
                  self.num_timesteps)
```

The ratio `mean_r_int / mean_r_ext` should be approximately 1/β when both signals contribute meaningfully. If `mean_r_int / mean_r_ext >> 1/β` after normalization, the bonus is dominating and β should be reduced. If `mean_r_int / mean_r_ext << 1/β`, the predictor has converged too fast and novelty is not being detected.

**Evaluating policy quality separately from curiosity quality.** To know whether the curiosity module is helping the policy, periodically run evaluation episodes with β = 0 (extrinsic reward only, no bonus) and log the pure extrinsic return. This tells you whether the exploration driven by curiosity has led to a genuinely better policy, not just a more curious one. If pure-extrinsic evaluation improves over training, curiosity is doing its job. A second useful diagnostic is state coverage — keep a count of unique state clusters (from k-means on the encoder features) visited over training. A healthy curiosity module should show monotonically increasing coverage early in training, then plateau as the reachable state space is exhausted. A plateau at a very low coverage number is a warning sign that the agent is stuck in a high-curiosity attractor rather than genuinely exploring.

#### Worked example: debugging a curiosity collapse

Suppose you train RND + PPO on a robotics navigation task and observe the following: intrinsic reward climbs from 0 to 50 in the first 500K steps, then drops sharply to 5 by 1M steps and stays there. Extrinsic return remains at 0 throughout. The collapse is suspicious — 1M steps is not nearly enough to exhaust a continuous robot environment.

Investigation: you log the predictor loss on held-out random states (not from the agent's trajectory) and find it is near-zero even for states the agent has never visited. This confirms overfitting: the predictor has generalized too aggressively and is now outputting near-target values everywhere, including genuinely novel states. The curiosity module has effectively become blind.

Fix: reduce the predictor's hidden layer size from 512 to 256 (matching the target network's capacity more closely), add a small weight decay of $10^{-5}$ to the predictor optimizer, and restart training. With these changes, the predictor is less capable of extrapolating to unseen states, and the curiosity signal remains informative for significantly longer. Alternatively, use the 25% subsampling trick from the original RND paper to slow down predictor convergence without changing the architecture.

## Results: MontezumaRevenge and the grid world

![Exploration coverage comparison showing epsilon-greedy policy visiting 18 percent of grid cells versus RND-augmented PPO visiting 89 percent of the same 20x20 grid under identical step budgets](/imgs/blogs/intrinsic-motivation-curiosity-driven-rl-2.png)

The figure above illustrates the core improvement. The left panel (epsilon-greedy) shows the agent visiting a narrow cluster of cells near the start position — the Q-function has not been updated meaningfully because no reward has been seen, so the softmax over Q-values is nearly uniform, and the agent's "exploitation" part of epsilon-greedy drifts aimlessly. The right panel (RND + PPO) shows broad coverage because the curiosity bonus creates a repulsive force from visited cells, continuously pushing the agent toward unvisited territory.

#### Worked example: MontezumaRevenge — from room 0 to room 2

The Montezuma's Revenge environment has 24 rooms arranged in a complex directed graph. The agent starts in room 0. To reach room 1, it must collect a key (100 points) and use it to unlock a door. Under a human expert, the minimal action sequence from spawn to key collection takes approximately 55 actions; with suboptimal routing, commonly 70–90 actions. The first non-key reward (the door itself) is an additional 300 points for a total of 400 per room transition.

The standard DQN baseline (Mnih et al., 2015) scored approximately 0 after 200M steps. The reason: the random epsilon-greedy exploration has probability roughly $0.05^{60} \approx 10^{-78}$ of stumbling onto the key by chance in one episode. In a training run of 200M steps with episodes averaging 1,000 steps (200,000 episodes), the expected number of accidental key collections is approximately $200{,}000 \times 10^{-78} \approx 0$. The agent literally cannot discover the reward by random exploration.

With RND + PPO (Burda et al. 2019), the training trajectory looks like this:

| Steps | Average score | Rooms explored | Curiosity signal |
|-------|--------------|----------------|-----------------|
| 0 | 0 | 0 | Maximum (all novel) |
| 1M | 20 | 0 (room 0 partial) | High for room 0 |
| 2M | 50 | 0 (near door) | Medium for room 0 |
| 5M | 150 | 1 (room 1 entered) | Low for room 0, high for room 1 |
| 10M | 400+ | 2 (room 2 reached) | Low for rooms 0–1 |
| 50M | 2,500+ | 8+ rooms | Frontier bonus driving forward |
| 200M | 10,000+ | 14+ rooms | Strong extrinsic signal now dominant |

The mechanism is a natural curriculum created by the decaying curiosity bonus. Room 0 becomes progressively "boring" as the predictor learns to predict the target network output for all its states. Once room 0 is familiar (curiosity bonus near zero), the locked door to room 1 is the only path to new states with high curiosity. The agent opens the door — not because it wants the 300-point reward, but because room 1 has maximum novelty. Room 1 then becomes familiar, pushing the agent to room 2, and so on. Curiosity creates a self-generating exploration curriculum with no manual reward engineering.

#### Worked example: 20x20 grid-world with hidden goal

Consider a 20×20 grid world (400 cells) with the agent starting at cell (0, 0) and a hidden goal at cell (18, 19) that delivers +10 reward only on contact. The agent receives no directional signal about the goal.

Setup: 128 parallel environments, 50,000 steps per environment (6.4M total environment interactions). Two conditions: epsilon-greedy DQN with ε = 0.1, and RND + PPO with β = 0.01.

**Epsilon-greedy DQN**: The Q-network initializes near zero everywhere. With no reward signal, the Q-values remain flat, and epsilon-greedy reduces to 90% random action, 10% "exploit" whatever arbitrary initial Q-values happened to be slightly higher. The agent performs a biased random walk, heavily concentrated near the start corner. After 50,000 steps, 72 of 400 cells have been visited (18%). The goal is reached in 14 of 128 runs (11%).

**RND + PPO**: The intrinsic reward is high for any unvisited cell. The agent is systematically pushed outward: once the immediate neighborhood of (0,0) is well-known, the curiosity bonus is highest for cells at the frontier. The agent performs an efficient expanding sweep. After 50,000 steps, 356 of 400 cells have been visited (89%). The goal is reached in 116 of 128 runs (91%).

The 18% → 89% coverage gap is not surprising given the formal analysis: epsilon-greedy performs an undirected random walk with complexity proportional to $S^2$ (it takes $O(S^2)$ steps to cover all states by random walk in a grid), while curiosity-driven exploration performs directed expansion with complexity closer to $O(S)$. For a 400-cell grid, this theoretical difference becomes empirically decisive within a 50K-step budget.

```python
import numpy as np
import gymnasium as gym
from gymnasium import spaces

class GridWorldEnv(gym.Env):
    """20x20 sparse-reward grid with hidden goal. Demonstrates the exploration gap."""
    metadata = {"render_modes": []}

    def __init__(self, size: int = 20, goal_seed: int = 42):
        super().__init__()
        self.size = size
        self.observation_space = spaces.Box(
            low=0.0, high=float(size - 1),
            shape=(2,), dtype=np.float32,
        )
        self.action_space = spaces.Discrete(4)   # up/down/left/right
        rng  = np.random.default_rng(goal_seed)
        self.goal = rng.integers(size // 2, size, size=2)  # goal in far quadrant
        self.pos  = np.zeros(2, dtype=np.float32)
        self.visited = set()

    def reset(self, seed=None, options=None):
        self.pos = np.zeros(2, dtype=np.float32)
        self.visited.clear()
        return self.pos.copy(), {}

    def step(self, action: int):
        deltas = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        dx, dy = deltas[action]
        self.pos = np.clip(self.pos + [dx, dy], 0, self.size - 1)
        self.visited.add(tuple(self.pos.astype(int)))
        reached = np.all(self.pos.astype(int) == self.goal)
        reward  = 10.0 if reached else 0.0
        return self.pos.copy(), reward, reached, False, {"coverage": len(self.visited)}
```

## Comparing the methods

![Comparison table showing ICM vs RND vs VIME vs count-based methods across stability, sample efficiency, sparse-reward performance, and noisy-TV robustness](/imgs/blogs/intrinsic-motivation-curiosity-driven-rl-4.png)

The matrix above captures each method across the axes that matter in practice. Here is the decision logic:

The curiosity module itself is a layered stack of transformations, each with a well-defined role:

![Curiosity module layer stack showing raw observation at the bottom progressing through CNN encoder, inverse dynamics, forward dynamics, prediction MSE computation, and normalized bonus scalar at the top](/imgs/blogs/intrinsic-motivation-curiosity-driven-rl-6.png)

**Use count-based** when the state space is truly discrete and small enough to enumerate. Tabular RL problems, small grid worlds, and discretized planning environments get theoretically optimal exploration at near-zero computational cost. Once you move to any continuous or high-dimensional state, the approximation quality degrades.

**Use ICM** when action-conditioned novelty matters: you want curiosity to fire specifically for *surprising consequences of actions*, not just surprising observations. ICM's inverse dynamics head grounds the feature space in controllable variation. Best for partially observable environments with lots of uncontrollable background variation you want the curiosity module to ignore.

**Use RND** as your default for high-dimensional environments. Stable, fast, state-of-the-art on hard-exploration Atari games. Single key hyperparameter β. No dynamics model to overfit. Partial vulnerability to noisy-TV is acceptable in most practical settings.

**Use VIME** only for small-scale academic experiments where you need the ground-truth information-gain signal, or to generate theoretical baselines. Not practical for CNN-scale architectures.

**Use NGU (Never-Give-Up)** when you face both hard exploration and noisy-TV risks simultaneously. The episodic + RND combination is the current best practice for difficult exploration problems in stochastic environments.

## Training dynamics and the bonus weight β

Choosing β is the most practically important hyperparameter decision. Too small and the curiosity bonus provides no meaningful signal in sparse-reward settings. Too large and the curiosity bonus drowns out the extrinsic reward, turning the agent into an explorer that ignores task completion.

![Grid showing 4x2 sweep of curiosity bonus weight beta from 0.001 to 1.0 against dense and sparse reward regimes, with average episode returns showing the optimal range at beta=0.01 to 0.1](/imgs/blogs/intrinsic-motivation-curiosity-driven-rl-8.png)

The pattern from the grid: β = 0.001 is too small to significantly improve sparse-reward exploration but harmless for dense reward. β = 0.01 is the universal safe starting point — it improves sparse performance substantially without hurting dense performance. β = 0.1 is optimal for sparse reward but begins competing with the extrinsic signal in dense settings. β = 1.0 causes reward collapse in dense settings: the curiosity bonus is roughly ten times the typical Atari reward per step, effectively making the agent ignore game score entirely.

Another important training dynamic: curiosity should arguably decay over training. Early, exploration is the bottleneck and a high β is valuable. Late, the agent has mapped most reachable space and a high β wastes budget on re-exploring already-understood corners.

```python
def anneal_beta(
    step: int,
    total_steps: int,
    beta_start: float = 0.1,
    beta_end: float = 0.001,
) -> float:
    """Linearly anneal curiosity weight: strong exploration → exploitation shift."""
    frac = min(1.0, step / total_steps)
    return beta_start + frac * (beta_end - beta_start)

# Example: Agent57-style adaptive strategy population
# Maintain K agents with different (beta, gamma) pairs
import numpy as np

def make_exploration_population(K: int = 32):
    """
    Create K (beta, gamma) pairs spanning exploration-exploitation spectrum.
    Agents with low j are exploitative; high j are exploratory.
    From Badia et al. 2020 (Agent57).
    """
    params = []
    for j in range(K):
        # Exponential schedule for beta
        beta_j = 0.3 * sigmoid(10 * (2*j/(K-1) - 1))
        # Gamma closer to 1 for exploitative, lower for exploratory
        gamma_j = 1 - exp(-1 * (9 * j / (K-1) + 1) * log(2))
        params.append((beta_j, gamma_j))
    return params

def sigmoid(x): return 1 / (1 + np.exp(-x))
def exp(x): return np.exp(x)
def log(x): return np.log(x)
```

The `gamma_ext: 0.999` vs `gamma_int: 0.99` pattern (separate discount factors for extrinsic and intrinsic rewards) is standard in production RND implementations. The extrinsic reward requires long credit assignment (room 1 → room 5 → final boss); the intrinsic reward is about immediate novelty and a shorter effective horizon is appropriate. Using separate value heads for intrinsic and extrinsic returns, as in the original RND paper, prevents the two objectives from interfering with each other's advantage estimates.

## Case studies

### Case study 1: Atari hard-exploration games and the RND benchmark

Burda et al. (2018) in "Large-Scale Study of Curiosity-Driven Learning" benchmarked curiosity (ICM without any extrinsic reward) across 54 Atari games. Key findings:

1. On most games with moderately dense reward (Pong, Breakout, Space Invaders), curiosity-only achieved roughly 40–80% of the performance of full-reward DQN — surprisingly strong given it gets no game score signal.
2. On hard-exploration games (Montezuma's Revenge: 2,500 points; Pitfall: +1,500 vs DQN's -1; Private Eye: 6,700 vs DQN's 91), curiosity dramatically outperformed standard DQN.
3. On games with permanent stochastic distractors (noisy-TV analog), curiosity agents fixated on the distractor and scored 0.

The 2019 RND follow-up (Burda et al., ICLR 2019) pushed Montezuma's Revenge above 10,000 points using PPO + RND, setting a new state of the art. The key improvements were: (a) the frozen target eliminates the dynamics-prediction instability; (b) separate intrinsic/extrinsic discount factors; (c) larger actor pool (128 parallel environments) enabling more diverse coverage; (d) reward normalization via running statistics.

### Case study 2: Agent57 — superhuman on all 57 Atari games

Agent57 (Badia et al., 2020) combined RND curiosity with:
- **NGU intrinsic reward**: episodic k-NN bonus × long-term RND bonus.
- **Adaptive multi-actor strategy**: a population of 32 actors with different (β, γ) pairs; a bandit algorithm assigns more actors to the (β, γ) pair that has been performing best recently.
- **Retrace returns**: an off-policy correction that allows using experience from actors with different (β, γ) settings to update the same Q-network.

Agent57 was the first algorithm to exceed human-level performance on all 57 Atari games. The three hardest games — Montezuma's Revenge (57,600+ vs human 4,753), Pitfall (estimated +3,000), and Skiing (−4,202 vs human −4,336) — were finally solved after four years of hard-exploration research building on ICM → pseudo-counts → RND → NGU → Agent57.

### Case study 3: curiosity in robot manipulation

Curiosity bonuses have been applied to sim-to-real robot manipulation in settings where the reward is binary (task complete or not). A representative setup: a Franka Panda arm must rearrange three blocks into a target configuration. The extrinsic reward is 1.0 only when all blocks are in the correct positions simultaneously, with zero intermediate reward. PPO without curiosity essentially never learns in this setting (the task requires 150-200 action steps, making random discovery exponentially unlikely).

With RND bonuses on the object position state (a 9-dimensional vector encoding all three block positions), the agent naturally discovers the mechanics of block pushing before the task reward fires. The curiosity drives it to move blocks in novel ways, inadvertently producing all the prerequisite skills the task demands. Sample efficiency improves by approximately 5–10× versus pure PPO in this setting (approximate result; exact numbers depend on the specific task and simulator).

### Case study 4: curiosity for LLM exploration

An emerging application of intrinsic motivation is in language model fine-tuning via RL (RLHF). In standard RLHF, the reward model fires on complete response quality — but early in training the policy is near the supervised fine-tuned baseline and rarely generates responses that the reward model rates as genuinely high-quality. The reward landscape is effectively sparse.

One approach (used implicitly in some Constitutional AI and RLHF implementations) is to add an entropy bonus to the policy — $\alpha \cdot \mathcal{H}(\pi)$ — which encourages the model to generate diverse outputs and not collapse to the mode of the reward distribution. This is the simplest form of intrinsic motivation for LLMs: the entropy bonus is a measure of response diversity, and maximizing it drives exploration of the response space.

More advanced approaches use a prediction-error novelty measure on the hidden state activations during decoding: states where the internal representations are unexpected (far from the running mean of seen representations) receive a curiosity bonus. This encourages the model to generate conceptually novel completions rather than just paraphrasing already-seen responses.

## When to use curiosity (and when not to)

The decision tree below crystallizes the selection logic:

![Decision tree guiding curiosity method selection based on whether reward is sparse, whether the state space is high-dimensional, and available compute budget, routing to count-based methods, RND, ICM, or VIME](/imgs/blogs/intrinsic-motivation-curiosity-driven-rl-7.png)

**Curiosity clearly helps** in these scenarios:

- Hard-exploration Atari games requiring multi-step prerequisite chains.
- Robotic manipulation with binary task-completion reward (no intermediate shaping).
- Navigation in large procedurally generated worlds (NetHack, Crafter, MiniGrid-MultiRoom).
- Any environment where the first nonzero reward requires more than approximately 20 sequential correct actions.
- Sim-to-real transfer where domain randomization is insufficient to drive policy diversity.

**Curiosity is neutral or wasteful** here:

- Dense-reward environments where epsilon-greedy or Gaussian noise suffices (CartPole, LunarLander, most MuJoCo benchmarks).
- Short-horizon tasks where random exploration covers the state space adequately within the training budget.
- Tasks with carefully designed reward shaping that already provides dense gradient toward the goal.

**Curiosity actively hurts** in these scenarios:

- Environments with accessible, irreducible stochastic variation (the noisy-TV problem). Symptoms: loss curves show increasing intrinsic reward with no improvement in extrinsic return; agent fixates on one region of the environment.
- Real-world deployment where exploration itself is costly (physical robot wear, safety constraints, expensive hardware evaluations).
- Late training phases where the agent has already covered the reachable state space. Disable curiosity or anneal β toward zero when exploration entropy saturates.

**When a simpler baseline wins:**

For tabular RL with a known model, use value iteration — it finds the optimal policy exactly without any exploration overhead. For continuous control with access to a high-fidelity simulator, domain randomization (randomizing physics, visual appearance, goal position) often achieves better transfer than curiosity-based exploration. For bandit-style problems with no temporal credit assignment, UCB or Thompson sampling are theoretically optimal and computationally trivial. The principle: **only pay the overhead of deep curiosity methods when the state space is genuinely high-dimensional and the reward genuinely sparse**.

```yaml
# Recommended RND hyperparameter configuration
rnd:
  beta: 0.01               # bonus weight; search 0.001–0.1
  embed_dim: 512           # target/predictor output dimension
  predictor_lr: 1.0e-4     # Adam learning rate for predictor update
  obs_norm_clip: 5.0       # clip normalized obs to [-5, 5]
  reward_norm: true        # normalize intrinsic reward by running std
  separate_value_heads: true   # use separate critics for ext and int reward

ppo_with_rnd:
  n_steps: 128             # rollout length per environment
  n_envs: 128              # parallel environments (crucial for coverage)
  gamma_ext: 0.999         # long-horizon extrinsic discount
  gamma_int: 0.99          # shorter-horizon intrinsic discount
  ent_coef: 0.001          # small entropy bonus still aids diversity
  vf_coef_ext: 1.0         # extrinsic value head weight
  vf_coef_int: 0.5         # intrinsic value head weight
```

## Timeline: milestones in curiosity-driven RL

The field progressed in clear waves from theory to practical breakthroughs:

![Training progress timeline showing random policy at zero steps, ICM reaching 50 points on MontezumaRevenge at 2M steps, RND reaching 150 at 5M steps and 400 plus at 10M steps, while the DQN baseline remains near zero after 200M steps](/imgs/blogs/intrinsic-motivation-curiosity-driven-rl-5.png)

The timeline makes the scaling challenge visible: DQN has 200× more wall-clock experience but achieves a fraction of RND's result. This is not because RND's policy is smarter in any deep sense — it is that curiosity provides the exploration gradient that DQN's epsilon-greedy exploration entirely lacks. Given that exploration signal, the underlying PPO policy optimizer (which is itself not dramatically different from DQN's Q-learning) solves the game efficiently.

## Key takeaways

1. **Sparse-reward exploration is the fundamental bottleneck** in hard RL environments. Epsilon-greedy cannot discover rewards behind long prerequisite action chains. An intrinsic bonus transforms the flat reward landscape into a rich exploration gradient.

2. **Total reward $r^{\text{total}} = r^{\text{ext}} + \beta \cdot r^{\text{int}}$** is the central augmentation. Extrinsic reward defines the terminal goal; intrinsic reward defines where to look while searching.

3. **ICM learns action-conditioned novelty** via an inverse dynamics head that grounds features in controllable variation. Best when background variation must be ignored (partial observability, rich uncontrolled visual environment).

4. **RND is the practical default** for high-dimensional inputs. Frozen target + trained predictor; distillation error = novelty. No dynamics model. Single hyperparameter β. Start at 0.01.

5. **VIME defines the theoretical ideal** (information gain about dynamics model parameters) but does not scale beyond small MLPs. Understanding it helps diagnose why ICM and RND fail.

6. **Count-based methods are theoretically optimal for tabular problems** but fail on images. Pseudo-counts bridge the gap but require a density model.

7. **The noisy-TV problem** — stochastic distractors generate permanent high curiosity — is the primary failure mode of prediction-error curiosity. Mitigation: episodic bonuses (NGU), ensemble variance decomposition, RND-over-ICM in stochastic environments.

8. **Always normalize intrinsic reward** using running mean/variance. Raw prediction errors vary orders of magnitude across environments and training stages.

9. **Use separate value heads and separate discount factors** for intrinsic and extrinsic rewards ($\gamma_{\text{ext}} = 0.999$, $\gamma_{\text{int}} = 0.99$). The two objectives have fundamentally different temporal scales.

10. **Curiosity adds no value in dense-reward settings** and actively hurts when β is too large or the environment has accessible stochastic distractors. Confirm reward sparsity before adding a curiosity module.

## Further reading

- Pathak et al. (2017). "Curiosity-driven Exploration by Self-Supervised Prediction." ICML 2017. The ICM paper — foundational, concise, and directly implementable.
- Burda et al. (2019). "Exploration by Random Network Distillation." ICLR 2019. The RND paper — introduces the distillation trick and sets Atari hard-exploration records.
- Houthooft et al. (2016). "VIME: Variational Information Maximizing Exploration." NeurIPS 2016. The information-gain foundation unifying all curiosity methods.
- Badia et al. (2020). "Never Give Up: Learning Directed Exploration Strategies." ICLR 2020. Episodic + long-term RND combination.
- Badia et al. (2020). "Agent57: Outperforming the Atari Human Benchmark." ICML 2020. First algorithm to beat human performance on all 57 Atari games.
- Bellemare et al. (2016). "Unifying Count-Based Exploration and Intrinsic Motivation." NeurIPS 2016. The pseudo-count paper bridging count-based methods to deep RL.
- Burda et al. (2018). "Large-Scale Study of Curiosity-Driven Learning." ICLR 2019. Comprehensive benchmark across 54 Atari games; systematically documents the noisy-TV failure.
- Within this series: [What is reinforcement learning](/blog/machine-learning/reinforcement-learning/what-is-reinforcement-learning) for the foundational RL loop; [Exploration vs exploitation: the core tension](/blog/machine-learning/reinforcement-learning/exploration-vs-exploitation-the-core-tension) for the classical bandit framework this post extends; [The reinforcement learning playbook](/blog/machine-learning/reinforcement-learning/the-reinforcement-learning-playbook) for the full series synthesis.
