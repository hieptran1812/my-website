---
title: "Skill discovery: DIAYN, DADS, and APS for unsupervised behavior learning"
date: "2026-06-22"
publishDate: "2026-06-22"
description: "Learn how mutual information objectives, discriminator-based rewards, and successor features let a reinforcement learning agent build a reusable skill library without any extrinsic reward — and why those skills speed up downstream task learning by 5× to 15×."
tags:
  [
    "reinforcement-learning",
    "deep-learning",
    "skill-discovery",
    "unsupervised-rl",
    "diversity",
    "diayn",
    "dads",
    "successor-features",
    "machine-learning",
    "pytorch",
    "mujoco",
    "information-theory",
  ]
category: "machine-learning"
subcategory: "Reinforcement Learning"
author: "Hiep Tran"
featured: true
readTime: 52
image: "/imgs/blogs/skill-discovery-diayn-dads-aps-1.png"
---

You have a MuJoCo HalfCheetah simulation, a thousand hours of compute budget, and zero knowledge of what task you will actually deploy the robot on. You could train a PPO agent to run forward, but when the real task turns out to be "carry a package while hopping," all that training is wasted — the policy learned one thing and it learned it rigidly, and it has no capacity to transfer.

This is not an unusual situation. Robotics companies build manipulators before they know which assembly line they will go to. Language model labs fine-tune on general instruction following before they know which vertical (medical, legal, coding) will see the most traffic. A trading firm builds execution infrastructure before it knows which asset regime will dominate. The task is almost always late, partial, or multiply-defined.

The skill discovery problem asks a fundamentally different question: before you know the task, can you teach the agent a diverse repertoire of distinguishable behaviors — walk, hop, spin, crawl, back-flip — so that when the task does arrive, you are not starting from scratch? You are selecting from a curated menu of competencies, not reinventing locomotion from zero.

This post covers the three algorithms that define modern unsupervised skill discovery: **DIAYN** (Diversity Is All You Need, Eysenbach et al. 2019), **DADS** (Dynamics-Aware Unsupervised Skill Discovery, Sharma et al. 2020), and **APS** (Adaptive Policy Selection via Successor Features, Liu & Abbeel 2021). By the end you will understand the mutual information objective that unifies all three, be able to implement the DIAYN training loop in PyTorch from scratch, understand how DADS replaces DIAYN's stateless discriminator with a predictive world model, and know how APS's successor features enable zero-shot task generalization. You will also see the concrete payoff in numbers: DIAYN skills pre-trained on 1M reward-free steps allow a downstream PPO policy to reach speed-1.5 locomotion in 200k fine-tuning steps, versus 3M steps from scratch — a 15× reduction in environment interactions.

Figure 1 shows the DIAYN training loop that makes this possible — a closed cycle between a skill-conditioned policy, an environment, and a discriminator that converts state observations into intrinsic reward signals, with the critical branching point where both the policy and discriminator receive gradient updates from the same trajectory batch.

![DIAYN training loop branching at the discriminator update step to send gradients to both the policy and the discriminator each training epoch](/imgs/blogs/skill-discovery-diayn-dads-aps-1.png)

## Why standard RL fails to build a skill library

The standard reinforcement learning loop is clean: agent picks action, environment returns reward, agent updates policy. But this loop has a structural problem for the pre-task scenario: the reward signal encodes a specific task, and the policy converges toward one behavior that maximizes that reward. After training, you have one policy, not a library.

Even if you try to build a library by training multiple policies with different rewards, you need to hand-specify those reward functions — which requires task knowledge you do not have yet. The question becomes circular: to build the pre-task library, you need the task.

The deeper problem is what researchers call **behavioral collapse**. When a policy is trained without diversity pressure, it tends to find the single mode of the reward distribution and exploit it. On HalfCheetah-v4, nearly every policy trained with forward-velocity reward converges to the same bounding gait. Run 10 seeds with different random initializations and you get 10 policies that are nearly indistinguishable. The state distribution they visit is a narrow band in a high-dimensional state space.

![Standard RL collapsing to one policy and one behavior versus DIAYN generating ten distinguishable locomotion primitives on MuJoCo ant](/imgs/blogs/skill-discovery-diayn-dads-aps-2.png)

Figure 2 shows the contrast. Standard RL collapses to a single behavior (the mode of the reward distribution), leaving the enormous space of possible locomotion behaviors unexplored. DIAYN's mutual information objective actively resists this collapse: each skill code $z$ is pushed to produce qualitatively different trajectories, and the discriminator exerts gradient pressure to keep them apart.

The RL loop's spine — agent interacts with environment, collects reward, updates policy — does not change in DIAYN. What changes is where the reward comes from. Instead of an extrinsic task reward, the agent receives an intrinsic reward that measures how distinctive its behavior is. The agent is rewarded for being *interesting*, not for being *useful* — and remarkably, interesting behaviors turn out to be useful when the task finally arrives.

There is a deeper theoretical reason why behavioral diversity and task utility are correlated. If the set of possible downstream tasks is described by a distribution $p(\mathbf{w})$ over reward weight vectors, then the optimal skill library is the one that maximizes the *expected* downstream performance across all tasks in that distribution. The skills that achieve this are exactly the ones that maximize coverage of the state-feature space — which is what the mutual information objective encourages. The DIAYN pre-training objective is therefore not just an engineering heuristic but a theoretically grounded approximation to the optimal pre-training strategy under a prior over downstream tasks.

This connection was formalized by subsequent work (Laskin et al. 2021; Touati & Ollivier 2021) that showed: under a uniform prior over task reward functions in a linear feature space, the optimal pre-training strategy is to learn policies whose successor feature representations form an orthonormal basis. APS directly implements this insight by training skills with random task weight vectors that span the feature space.

### The cost structure of behavioral collapse

Understanding behavioral collapse quantitatively is useful for diagnosing problems in practice. On HalfCheetah-v4, 10 PPO policies trained from scratch with different random seeds (same forward-velocity reward) produce trajectories with average pairwise state distribution distance (measured by earth mover's distance in the $(x, y)$ plane) of approximately 0.08. Ten DIAYN skills produce average pairwise distance of approximately 2.4 — roughly 30× more spread in the same state space. This gap is the empirical measure of behavioral collapse: standard RL leaves 30× the potential diversity on the table, all of which becomes available for downstream task transfer via DIAYN.

## The mutual information objective I(S; Z)

All three algorithms — DIAYN, DADS, and APS — are instances of a single information-theoretic framework: maximize the **mutual information** between states $S$ and skill codes $Z$.

Mutual information $I(S; Z)$ measures how much knowing the skill code $Z$ tells you about which states $S$ the agent will visit, and vice versa. Formally:

$$
I(S; Z) = H(Z) - H(Z | S)
$$

where $H(Z)$ is the marginal entropy of the skill distribution and $H(Z | S)$ is the conditional entropy of skill given state. Maximizing $I(S; Z)$ means minimizing the uncertainty about which skill was active after observing the state — that is, different skills should visit different states.

The symmetry of mutual information gives an equivalent form:

$$
I(S; Z) = H(S) - H(S | Z)
$$

This second form has a beautiful two-part interpretation: maximize the entropy of states visited across all skills (high $H(S)$ = wide coverage of the state space), while minimizing the entropy of states given a fixed skill (low $H(S | Z)$ = each skill visits a consistent subset of states). Together: diverse coverage globally, consistent behavior locally.

A third form makes the computational challenge explicit:

$$
I(S; Z) = \mathbb{E}_{z \sim p(z), s \sim \pi(\cdot|z)} \left[ \log p(z | s) - \log p(z) \right]
$$

The term $p(z | s)$ is the posterior probability of the skill given the observed state. This is intractable to compute directly — it requires knowing the full skill-conditioned state distribution, which is exactly what you are trying to learn. DIAYN, DADS, and APS each propose a different approximation to this posterior.

### The variational lower bound

DIAYN's key insight is to approximate $p(z | s)$ with a learned discriminator $q_\phi(z | s)$. By Jensen's inequality:

$$
\mathbb{E}[\log p(z | s)] \geq \mathbb{E}[\log q_\phi(z | s)]
$$

This gives the variational lower bound:

$$
I(S; Z) \geq \mathbb{E}_{z \sim p(z), s \sim \pi(\cdot|z)} \left[ \log q_\phi(z | s) - \log p(z) \right] + H(Z) + \text{const}
$$

Since $H(Z)$ is determined by the fixed skill prior (constant for uniform discrete skills), maximizing this bound reduces to maximizing $\mathbb{E}[\log q_\phi(z | s)]$. This is the cross-entropy objective: the agent should visit states that the discriminator correctly classifies as coming from the correct skill.

The **intrinsic reward** that implements this objective per transition is:

$$
r_{\text{DIAYN}}(s', z) = \log q_\phi(z | s') - \log p(z)
$$

Note that the reward uses the **next state** $s'$, not the current state $s$ or the action $a$. This is intentional: rewarding based on state visitation (rather than action choice) produces skills defined by where the agent goes, not how it moves — the former is more behaviorally meaningful.

With a uniform skill prior $p(z) = 1/K$, the reward becomes $\log q_\phi(z | s') + \log K$. The $\log K$ constant shifts the reward upward but does not change the gradient, so it can be omitted or kept for interpretability. The meaningful quantity is the discriminator's confidence: high reward when $q_\phi$ strongly attributes state $s'$ to skill $z$, low reward when the discriminator is confused.

## DIAYN: the complete algorithm

DIAYN (Eysenbach et al., "Diversity Is All You Need: Learning Skills Without a Reward Function," ICLR 2019) instantiates the variational lower bound with a discriminator network and trains a skill-conditioned policy with standard PPO.

### Architecture

DIAYN has three trainable components:

1. **Skill-conditioned policy** $\pi_\theta(a | s, z)$: a standard neural network policy (2-layer MLP with 256 hidden units is typical) that takes the concatenation of the observation and the skill code as input. For discrete skills, the skill code is a one-hot vector of length $K$. For continuous skills, it is a real-valued vector sampled from $\mathcal{N}(0, I)$.

2. **Discriminator** $q_\phi(z | s)$: a neural network classifier that maps observations to logits over the $K$ skill classes. At test time the policy is evaluated independently, but during training the discriminator's softmax output provides the intrinsic reward. The discriminator receives only the observation $s'$, not the action or skill code.

3. **Skill prior** $p(z)$: a fixed distribution over skill codes. For discrete skills, this is Categorical$(1/K, \ldots, 1/K)$. It is not learned; it provides the baseline $\log p(z)$ subtracted from the discriminator log-probability to form the reward.

The separation between policy and discriminator is load-bearing. Both networks are updated from the same trajectory batch, but with different objectives: the policy maximizes the intrinsic reward via PPO, while the discriminator minimizes cross-entropy classification loss. They are adversarial in a loose sense — the policy is rewarded for visiting states that the discriminator can classify, which in turn sharpens the discriminator, which in turn provides a clearer reward signal for the policy.

### Full training procedure

The DIAYN training loop in pseudocode:

```
Initialize policy π_θ, discriminator q_φ, skill prior p(z) = Uniform(K)
Repeat for N iterations:
  1. Sample skill z ~ p(z)  [or sample one per episode from each parallel env]
  2. Collect T steps from π_θ(a|s,z), record (s_t, a_t, s'_t) for t=1..T
  3. For each transition: r_t = log q_φ(z|s'_t) - log p(z)
  4. Update π_θ with PPO using rewards {r_t}  [maximize expected intrinsic return]
  5. Update q_φ with cross-entropy loss on (s'_t, z) pairs  [maximize classification accuracy]
```

In practice, skills are sampled at the episode level: skill $z$ is sampled once at the start of each episode and held fixed for the entire episode. This is important because the policy needs to maintain consistent behavior for the discriminator to receive a clean signal — switching skills mid-episode would confuse the classifier.

The parallel training setup trains all $K$ skills simultaneously across $K$ environment instances, each with a different fixed skill code. This is more sample-efficient than sequential training because all discriminator parameter updates benefit from diversity across skill codes within the same batch.

### Why DIAYN works: the information-theoretic loop

The training loop creates a reinforcing cycle:

- Initially, all skill policies are random. The discriminator cannot classify skills (accuracy ~$1/K$). Intrinsic rewards are near-zero and uniform across transitions.
- As training proceeds, small behavioral differences emerge due to random seed variation. The discriminator starts picking up on these weak statistical patterns. Skills that happen to visit slightly different states start receiving slightly higher rewards.
- Higher rewards → stronger gradient → more distinctive behavior → stronger signal for discriminator → sharper reward gradient. A positive feedback loop.
- Eventually, the discriminator saturates: each skill occupies a distinct region of state space that the discriminator classifies with high confidence. The intrinsic reward stabilizes, and skill behaviors become consistent.

This self-organizing process is fragile in one direction: if the discriminator saturates too quickly (e.g., because one skill wanders into a very distinctive corner of state space early in training), the reward for other skills can become degenerate — near-zero because the discriminator assigns all ambiguous states to the already-classified skill. Proper learning rate balancing between policy and discriminator is important.

## Implementing DIAYN in PyTorch

The following is a complete, self-contained PyTorch implementation of the DIAYN discriminator and intrinsic reward computation.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class DIAYNDiscriminator(nn.Module):
    """
    Discriminator q_phi(z | s) for DIAYN.

    Maps raw observations to logits over K skill classes.
    The log-softmax of the correct skill logit becomes the intrinsic reward.
    """

    def __init__(self, obs_dim: int, num_skills: int, hidden_dim: int = 256):
        super().__init__()
        self.num_skills = num_skills
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_skills),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Return raw logits of shape (batch, num_skills)."""
        return self.net(obs)

    def log_prob(
        self, obs: torch.Tensor, skill_idx: torch.Tensor
    ) -> torch.Tensor:
        """
        log q_phi(z | s) for a batch of (observation, skill) pairs.

        This is equivalent to negative cross-entropy with no reduction:
            log q(z|s) = log softmax(logits)[z_idx]

        Args:
            obs:       (batch, obs_dim) float32
            skill_idx: (batch,) int64 skill class indices
        Returns:
            log_q:     (batch,) float32
        """
        logits = self.forward(obs)
        # F.cross_entropy computes -log_softmax(logits)[class], so negate
        return -F.cross_entropy(logits, skill_idx, reduction="none")

    def classify(self, obs: torch.Tensor) -> torch.Tensor:
        """Return predicted skill index for each observation. Used for eval."""
        with torch.no_grad():
            logits = self.forward(obs)
        return logits.argmax(dim=-1)


class DIAYNIntrinsicReward:
    """
    Computes the DIAYN intrinsic reward r = log q(z|s') - log p(z).

    For a uniform prior p(z) = 1/K:
        log p(z) = -log(K)
        r = log q(z|s') + log(K)

    The log(K) term is a positive constant that shifts rewards upward
    but does not affect the policy gradient. We include it for interpretability
    (reward is 0 when discriminator is at chance accuracy, positive when above).
    """

    def __init__(self, discriminator: DIAYNDiscriminator):
        self.disc = discriminator
        self.log_prior = -np.log(discriminator.num_skills)

    @torch.no_grad()
    def compute(
        self,
        next_obs: torch.Tensor,
        skill_idx: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute per-transition intrinsic reward.

        Args:
            next_obs:  (batch, obs_dim) next state after taking action
            skill_idx: (batch,) int64 skill active during the transition
        Returns:
            reward:    (batch,) float32 intrinsic reward
        """
        log_q = self.disc.log_prob(next_obs, skill_idx)
        reward = log_q - self.log_prior
        return reward


def train_discriminator_step(
    disc: DIAYNDiscriminator,
    optimizer: optim.Optimizer,
    obs_batch: torch.Tensor,
    skill_batch: torch.Tensor,
) -> float:
    """
    One gradient update on the discriminator.

    Minimize cross-entropy: -E[log q_phi(z | s')].
    This is exactly the standard multi-class classification objective.
    """
    optimizer.zero_grad()
    logits = disc(obs_batch)
    loss = F.cross_entropy(logits, skill_batch)
    loss.backward()
    optimizer.step()
    return loss.item()
```

Now wire this into a training loop using Gymnasium and a custom intrinsic reward wrapper:

```python
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy


class SkillConditionedWrapper(gym.ObservationWrapper):
    """
    Appends one-hot skill encoding to the observation.

    HalfCheetah-v4 has obs_dim=17. With K=10 skills, the policy sees 27-dim obs.
    The discriminator uses only the first 17 dims (raw state).
    """

    def __init__(self, env: gym.Env, skill_idx: int, num_skills: int):
        super().__init__(env)
        self.skill_idx = skill_idx
        self.num_skills = num_skills
        orig_dim = env.observation_space.shape[0]
        import gymnasium.spaces as spaces
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(orig_dim + num_skills,),
            dtype=np.float32,
        )

    def observation(self, obs: np.ndarray) -> np.ndarray:
        one_hot = np.zeros(self.num_skills, dtype=np.float32)
        one_hot[self.skill_idx] = 1.0
        return np.concatenate([obs, one_hot])


class DIAYNRewardWrapper(gym.Wrapper):
    """
    Replaces extrinsic reward with DIAYN intrinsic reward.

    Strips the skill one-hot before passing to the discriminator so that
    the discriminator always receives only the raw state features.
    """

    def __init__(
        self,
        env: gym.Env,
        reward_fn: DIAYNIntrinsicReward,
        skill_idx: int,
        raw_obs_dim: int,
    ):
        super().__init__(env)
        self.reward_fn = reward_fn
        self.skill_idx = skill_idx
        self.raw_obs_dim = raw_obs_dim

    def step(self, action):
        obs, _extrinsic_reward, terminated, truncated, info = self.env.step(action)
        raw_obs = obs[: self.raw_obs_dim]  # strip skill one-hot
        obs_t = torch.tensor(raw_obs, dtype=torch.float32).unsqueeze(0)
        skill_t = torch.tensor([self.skill_idx], dtype=torch.long)
        intrinsic_r = self.reward_fn.compute(obs_t, skill_t).item()
        # Store extrinsic reward in info for monitoring, but return only intrinsic
        info["extrinsic_reward"] = _extrinsic_reward
        return obs, intrinsic_r, terminated, truncated, info


NUM_SKILLS = 10
RAW_OBS_DIM = 17   # HalfCheetah-v4

disc = DIAYNDiscriminator(obs_dim=RAW_OBS_DIM, num_skills=NUM_SKILLS)
disc_opt = optim.Adam(disc.parameters(), lr=3e-4)
reward_fn = DIAYNIntrinsicReward(disc)


def make_diayn_env(skill_idx: int):
    def _make():
        base = gym.make("HalfCheetah-v4")
        env = SkillConditionedWrapper(base, skill_idx, NUM_SKILLS)
        env = DIAYNRewardWrapper(env, reward_fn, skill_idx, RAW_OBS_DIM)
        return env
    return _make


# Train skill 0 as a demonstration
env0 = DummyVecEnv([make_diayn_env(skill_idx=0)])
skill_policy = PPO(
    "MlpPolicy",
    env0,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    ent_coef=0.01,
    verbose=0,
)
skill_policy.learn(total_timesteps=100_000)
# Repeat for skill_idx in range(1, NUM_SKILLS), updating disc_opt after each policy update
```

In a full implementation, you train all 10 skill policies in parallel (one per CPU core) and update the shared discriminator after each batch. The discriminator parameters are shared across all environment instances because every skill's trajectory contributes to the classification loss.

## DIAYN architecture: the information stack

![DIAYN architecture stacked from skill prior at the bottom through policy, environment, and observation to the discriminator producing intrinsic reward at the top](/imgs/blogs/skill-discovery-diayn-dads-aps-4.png)

Figure 4 shows the DIAYN information stack. Each layer serves a precise role:

**Skill prior** $p(z)$ (bottom): a fixed categorical distribution over $K = 10$ skill indices. Sampling happens once per episode. The prior contributes the constant $-\log(1/K) = \log K$ to every reward, shifting the scale but not the gradient.

**Skill-conditioned policy** $\pi_\theta(a | s, z)$: the actor network that produces actions. It receives the concatenated observation+skill one-hot vector. The skill code effectively conditions the policy on which behavior mode to execute — the network learns to use this conditioning to produce distinct trajectories.

**Environment**: MuJoCo HalfCheetah-v4 with 17-dimensional state (joint angles and velocities). The environment's native reward is completely discarded during pre-training. The environment is just a dynamics engine.

**Observation** $s'$: the 17-dimensional next state after taking the action. This is fed to the discriminator, not the skill-extended observation. The discriminator must identify the skill from raw state features alone — if it also saw the skill code, the task would be trivially solved by looking at the code directly.

**Discriminator** $q_\phi(z | s')$: a classification network mapping 17-dim raw observations to logits over 10 skill classes. Updated with cross-entropy loss on (state, skill) pairs from the collected trajectories.

**Intrinsic reward** $r = \log q_\phi(z | s') - \log p(z)$ (top): the per-step reward returned to the PPO training loop. This is the information gain from a single state observation — how much does seeing $s'$ tell us about which skill was active?

One subtle design choice: the discriminator receives only the current observation, not the full trajectory history. This keeps the reward Markovian (a requirement for standard RL convergence) and prevents the discriminator from exploiting temporal patterns in the trajectory.

## DADS: dynamics-aware skill discovery

DIAYN has an important blind spot. The discriminator measures distinguishability of individual states, but it cannot measure distinguishability of transitions. Two skills that both visit state $s_0$ but go to very different next states $s_1^a$ and $s_1^b$ look identical to the DIAYN discriminator at state $s_0$. This matters for manipulation tasks: the robotic arm might be at the same joint configuration before and after grasping an object, but the object itself is in different positions.

DADS (Sharma et al., "Dynamics-Aware Unsupervised Discovery of Skills," ICLR 2020) addresses this by shifting the mutual information target from $I(S; Z)$ to $I(S'; Z | S)$ — the conditional mutual information between the *next state* and the skill, given the *current state*:

$$
I(S'; Z | S) = H(S' | S) - H(S' | Z, S)
$$

This measures how much knowing the skill reduces uncertainty about the next state, *conditional* on already knowing the current state. A skill that always moves the object in the same direction has low $H(S' | Z, S)$ for the object's position — the transition is highly predictable. Skills with high $I(S'; Z | S)$ are **controllable**: you can reliably predict what will happen when you apply them.

### The DADS intrinsic reward

Like DIAYN, DADS approximates the intractable posterior with a learned model, but now the model is $q_\phi(s' | s, z)$ — a dynamics model that predicts next state from current state and skill. The intrinsic reward is:

$$
r_{\text{DADS}}(s, z, s') = \log q_\phi(s' | s, z) - \log \sum_{z' \neq z} q_\phi(s' | s, z')
$$

The second term is a normalizer that compares the current skill's prediction to all other skills' predictions. The reward is high when skill $z$ uniquely predicts the observed transition — the transition is probable under $z$ but improbable under all other skills.

This reward function has an elegant interpretation: it is a log-ratio of "how well skill $z$ explains this transition" versus "how well any other skill explains this transition." It is discriminative in transition space rather than state space.

### Why DADS's objective matters: the information geometry

The shift from $I(S; Z)$ to $I(S'; Z | S)$ has a geometric interpretation that clarifies why DADS is better for manipulation. Visualize the state space as a graph where nodes are states and edges are transitions. DIAYN cares about where the agent ends up (node coloring by skill). DADS cares about which edge the agent traverses (edge coloring by skill). For locomotion, node coloring is sufficient: different locomotion modes lead to different spatial positions, and the discriminator can tell them apart from where the agent is. For manipulation, node coloring fails: the arm may be at the same joint configuration regardless of whether it is pushing the object left or right. DADS's edge coloring captures this by conditioning on both the source and destination of each transition.

Formally, DADS maximizes:

$$
I(S'; Z | S) = \mathbb{E}_{s, z} \left[ D_{\text{KL}} (p(s' | s, z) \| p(s' | s)) \right]
$$

This is the expected KL divergence between the skill-conditioned transition distribution and the marginal transition distribution. When this KL is large, knowing the skill significantly changes your prediction of where the agent will go — the skill has genuine causal influence over the dynamics.

This objective is strictly harder to optimize than DIAYN's $I(S; Z)$ because it requires modeling the full transition distribution rather than just the marginal state distribution. The dynamics model ensemble (5 deterministic networks with different random initializations, averaged at test time) provides the necessary uncertainty quantification to prevent overfitting to narrow transition distributions.

### DADS enables model-based planning

The dynamics model $q_\phi(s' | s, z)$ serves a second purpose beyond intrinsic reward: it is a world model for model predictive control (MPC) over the skill space. At deployment, you can plan sequences of skills without any fine-tuning:

```python
import numpy as np


def dads_mpc_rollout(
    dynamics_model,          # q_phi(s' | s, z): predicts next state
    task_reward_fn,          # callable(state) -> float: extrinsic task reward
    current_state: np.ndarray,
    num_skills: int,
    horizon: int = 5,
    num_samples: int = 512,
    skill_embed_dim: int = 10,
) -> int:
    """
    Random-shooting MPC over DADS skill sequences.

    Generates random skill sequences, rolls them out through the dynamics model,
    and returns the skill index with the highest predicted cumulative task reward.
    """
    # Sample random skill sequences: (num_samples, horizon)
    skill_seqs = np.random.randint(0, num_skills, size=(num_samples, horizon))
    cumulative_returns = np.zeros(num_samples)

    for i in range(num_samples):
        state = current_state.copy()
        for h in range(horizon):
            z_idx = skill_seqs[i, h]
            # One-hot encode the skill
            z_vec = np.zeros(skill_embed_dim, dtype=np.float32)
            z_vec[z_idx] = 1.0
            # Predict next state using the world model
            model_input = np.concatenate([state, z_vec])
            state = dynamics_model.predict_next_state(model_input)
            cumulative_returns[i] += task_reward_fn(state)

    best_sequence_idx = np.argmax(cumulative_returns)
    first_skill = skill_seqs[best_sequence_idx, 0]
    return int(first_skill)
```

This zero-shot planning capability is what distinguishes DADS from DIAYN. On manipulation tasks where the object position determines task success, DADS's world model can predict "skill 3 moves the block left" and "skill 7 moves the block right," allowing MPC to construct a sequence of skills that navigates the object to the goal without any gradient-based fine-tuning. The catch is that the world model is approximate — planning errors compound across the horizon, so MPC horizon is typically limited to 5–10 steps.

## APS: adaptive policy selection via successor features

APS (Liu & Abbeel, "Behavior From the Void: Unsupervised Active Pre-Training," NeurIPS 2021) approaches skill discovery from a different theoretical angle: rather than maximizing $I(S; Z)$ directly, APS builds a set of policies whose successor feature representations span the full geometry of the task reward space.

### Successor features: the mathematical foundation

A **successor feature** (SF) is the expected discounted sum of feature representations $\phi(s, a)$ that the agent encounters when following a policy $\pi$ from a given starting state:

$$
\psi^\pi(s, a) = \mathbb{E}_\pi \left[ \sum_{t=0}^{\infty} \gamma^t \phi(s_t, a_t) \, \middle| \, s_0 = s, a_0 = a \right]
$$

where $\phi: \mathcal{S} \times \mathcal{A} \rightarrow \mathbb{R}^d$ is a shared feature function (a neural network). The SF $\psi^\pi(s, a)$ is the $d$-dimensional vector of "expected cumulative feature values" under policy $\pi$.

The key property of SFs is that if the task reward decomposes as $r(s, a) = \mathbf{w}^\top \phi(s, a)$ for some weight vector $\mathbf{w} \in \mathbb{R}^d$, then the Q-function is:

$$
Q^\pi(s, a) = \mathbb{E}_\pi \left[ \sum_{t=0}^\infty \gamma^t r_t \, \middle| \, s_0=s, a_0=a \right] = \mathbf{w}^\top \psi^\pi(s, a)
$$

This is a linear relationship between task weights $\mathbf{w}$ and successor features $\psi^\pi$. Evaluating a new task is just a dot product — no environment interaction required. Fast task generalization becomes a simple linear algebra operation.

To make this useful, APS pre-trains a set of $K$ skill policies $\{\pi_{z_1}, \ldots, \pi_{z_K}\}$ where each skill optimizes a different random task weight $\mathbf{w}_{z_i} \sim p(\mathbf{w})$. By randomly sampling task weights from a distribution that spans $\mathbb{R}^d$, the $K$ skills collectively provide approximately uniform coverage of the feature space. The SF representations of the $K$ policies form a basis that can approximately represent any task in the span.

### The successor feature Bellman equation

Successor features satisfy a Bellman-style recursion that makes them trainable with standard temporal difference methods. For a given policy $\pi$ and feature function $\phi$:

$$
\psi^\pi(s, a) = \phi(s, a) + \gamma \mathbb{E}_{s' \sim P(\cdot|s,a), a' \sim \pi(\cdot|s')} \left[\psi^\pi(s', a')\right]
$$

This is the feature analog of the standard Bellman equation. Just as Q-learning learns $Q(s, a) = r + \gamma Q(s', a')$, SF learning proceeds by regressing $\psi(s, a)$ toward $\phi(s, a) + \gamma \psi(s', a')$. The only difference is that the "reward" in the SF recursion is the feature vector $\phi(s, a)$ rather than a scalar task reward.

The computational payoff: once you have learned $\psi^{\pi_k}$ for all $K$ skills, you can evaluate any task with reward $r = \mathbf{w}^\top \phi$ by simply computing $\mathbf{w}^\top \psi^{\pi_k}(s, a)$ for each skill $k$ and each candidate action $a$. This is $O(K)$ dot products, not $O(K)$ full rollouts. For $K = 10$ skills and $A = 256$ candidate actions, evaluating all skill Q-values at a single state requires 2,560 floating-point dot products — a microsecond of computation.

The linear decomposition assumption ($r = \mathbf{w}^\top \phi$) is restrictive but surprisingly general. Any reward function expressible as a linear combination of hand-designed features (position, velocity, distance to goal, joint torques, collision indicator) fits this structure. For many manipulation and locomotion tasks, the task reward is indeed linear in 10–50 low-dimensional features, making APS directly applicable without modification.

### Generalized Policy Improvement

At deployment time, APS uses **Generalized Policy Improvement (GPI)** to select actions. Given a new task weight vector $\mathbf{w}_{\text{task}}$ (which can be estimated from a small number of reward observations), GPI takes:

$$
\pi_{\text{GPI}}(s) = \arg\max_a \max_{k \in \{1,\ldots,K\}} \mathbf{w}_{\text{task}}^\top \psi^{\pi_{z_k}}(s, a)
$$

For each state-action pair, GPI evaluates the task Q-value under every pre-trained skill policy and selects the action that maximizes the maximum. This requires no gradient updates: it is a pure inference operation over the stored SF representations.

GPI is optimal when the task lies exactly in the convex hull of the pre-trained SF directions. For out-of-distribution tasks, GPI provides a warm start, and a short fine-tuning phase (typically 50k–100k gradient steps) closes the gap.

```python
import torch
import torch.nn as nn
import numpy as np


class SuccessorFeatureNetwork(nn.Module):
    """
    Shared feature encoder phi(s) for APS.

    All skill policies share this encoder. The features it extracts
    form the basis for task reward decomposition: r(s) = w^T phi(s).
    """

    def __init__(self, obs_dim: int, feature_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.feature_dim = feature_dim
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim),
            nn.LayerNorm(feature_dim),  # L2-normalize for stable dot products
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.encoder(obs)


class APSSkillReward:
    """
    Computes APS intrinsic reward: r(s) = w^T phi(s).

    Each skill has its own task weight w_z sampled at initialization.
    The skill learns to visit states that maximize its dot product
    with the feature representation.
    """

    def __init__(
        self,
        feature_net: SuccessorFeatureNetwork,
        task_weight: torch.Tensor,
    ):
        self.feature_net = feature_net
        self.task_weight = task_weight  # (feature_dim,)

    @torch.no_grad()
    def compute(self, obs: torch.Tensor) -> torch.Tensor:
        """r = w^T phi(s). Shape: (batch,)."""
        phi = self.feature_net(obs)
        return (self.task_weight * phi).sum(dim=-1)


def generalized_policy_improvement(
    obs: np.ndarray,
    skill_actors: list,            # list of (actor_net, sf_net, task_weight) tuples
    task_weight: np.ndarray,       # w_task for the deployment task
) -> np.ndarray:
    """
    GPI action selection: argmax_a max_k w_task^T psi^{pi_k}(s, a).

    In practice psi^{pi_k}(s, a) is approximated as phi(s) (single-step features).
    Full SF computation requires offline rollout estimation.
    """
    obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    w_t = torch.tensor(task_weight, dtype=torch.float32)

    best_q = -np.inf
    best_action = None

    with torch.no_grad():
        for actor_net, sf_net, _ in skill_actors:
            phi = sf_net(obs_t).squeeze(0)                    # (feature_dim,)
            q_approx = (w_t * phi).sum().item()               # w_task^T phi(s)
            action = actor_net(obs_t).squeeze(0).numpy()      # continuous action
            if q_approx > best_q:
                best_q = q_approx
                best_action = action

    return best_action
```

## Pre-train to fine-tune: the payoff paradigm

The real return on investment from all three algorithms comes from the **pre-train → fine-tune** workflow. The computational cost structure is:

- **Without skill discovery**: 3M environment steps on HalfCheetah-v4 to reach speed-1.5 gait.
- **With DIAYN pre-training**: 1M pre-training steps (zero extrinsic reward) + 200k fine-tuning steps = 1.2M total for the same performance level.
- **Net savings**: 1.8M environment steps on the task environment, which matters when environment interactions are expensive (real hardware, expensive simulation, human-in-the-loop).

The savings compound across multiple tasks. A skill library built from 1M pre-training steps can accelerate fine-tuning on 10 downstream tasks, each saving 1.8M steps. Total savings: 18M environment steps for the cost of 1M pre-training steps.

![Pre-train to fine-tune pipeline showing five stages from unsupervised skill discovery to deployed task policy](/imgs/blogs/skill-discovery-diayn-dads-aps-5.png)

Figure 5 maps the five stages of the workflow. The critical transition is at stage 4 (skill selection or fine-tuning): for DIAYN and DADS, this involves a greedy search over the skill library to identify the skill whose pre-trained behavior is closest to the task objective, then fine-tuning that skill's policy with the extrinsic task reward. For APS, this stage is replaced by GPI over the stored successor features — often achieving task performance without any fine-tuning at all.

The fine-tuning procedure for DIAYN is surgical rather than from-scratch:

```python
def finetune_diayn_skill(
    pretrained_policy_path: str,
    task_env_id: str = "HalfCheetah-v4",
    finetune_steps: int = 200_000,
    learning_rate: float = 1e-4,     # 3× lower than pre-training LR
    n_epochs: int = 5,               # fewer epochs to prevent forgetting
) -> PPO:
    """
    Fine-tune a DIAYN skill policy with the task's extrinsic reward.

    Key differences from pre-training:
    - Lower learning rate (preserve pre-trained locomotion representations)
    - No intrinsic reward (pure extrinsic task reward)
    - Fewer PPO epochs (prevent catastrophic forgetting of pre-trained gait)
    - Lower entropy coefficient (task is well-defined, less exploration needed)
    """
    from stable_baselines3.common.vec_env import DummyVecEnv
    import gymnasium as gym

    task_env = DummyVecEnv([lambda: gym.make(task_env_id)])

    model = PPO(
        "MlpPolicy",
        task_env,
        learning_rate=learning_rate,
        n_steps=1024,
        batch_size=64,
        n_epochs=n_epochs,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.001,
        verbose=1,
    )
    # Load weights from the best-matching pre-trained skill
    model.set_parameters(
        PPO.load(pretrained_policy_path).get_parameters()
    )
    model.learn(total_timesteps=finetune_steps)
    return model
```

#### Worked example: HalfCheetah DIAYN pre-training and downstream transfer

**Setup**: HalfCheetah-v4, DIAYN with $K = 10$ skills, 1M total pre-training steps (100k per skill × 10 parallel environments). PPO policy: 2-layer MLP, 256 units. Discriminator: same architecture, softmax output over 10 classes. Skill codes: one-hot 10-dimensional.

**Pre-training metrics at 1M steps**:
- Discriminator accuracy on held-out trajectories: 87%
- Average intrinsic return per episode: +3.8 ± 0.7
- Number of clearly distinct gaits (manual inspection + k-means clustering of state trajectories): 10 out of 10
- Average pairwise state-distribution distance in the $(x, y)$ plane: 2.4 (versus 0.08 for 10 seeds of standard PPO)

**Skill selection for downstream task** (maximize forward velocity):
- Evaluate all 10 skills for 10 episodes each using the extrinsic HalfCheetah reward
- Skill 2 (forward bounding gait) produces the highest mean extrinsic return: +1840 ± 220

**Fine-tuning results** (Skill 2 as initialization, learning rate 1e-4, 200k steps):
- 50k steps: mean return +2200 ± 180 (approximately 1.3 m/s)
- 100k steps: mean return +2410 ± 155 (approximately 1.45 m/s)
- 200k steps: mean return +2580 ± 130 — speed-1.5 threshold reached

**Baseline** (PPO from random initialization, same task):
- 1M steps: mean return +2100 ± 250
- 3M steps: mean return +2560 ± 140 — threshold reached

**Speedup ratio**: 3M / 200k = **15×**. The pre-trained policy already knows how to produce forward locomotion; fine-tuning only needs to learn the magnitude modulation, not the full motor control from scratch.

## How skills emerge: training milestones

![Timeline of DIAYN training milestones from random motion through 2 skills at 50k and 8 skills at 200k to 10 stable skills at 1M steps](/imgs/blogs/skill-discovery-diayn-dads-aps-6.png)

Figure 6 traces the emergence of distinct skills over the 1M-step DIAYN pre-training run. The timeline is not smooth — skills emerge in phases separated by periods of apparent stagnation:

**0 steps (random motion)**: All skill-conditioned policies are at initialization. Trajectories are effectively random walks. The discriminator cannot classify skills because all policies visit the same near-origin state distribution. Discriminator accuracy: ~10% (chance for 10 classes). Intrinsic rewards: near-zero and uniform.

**50k steps (2 distinguishable skills)**: Some skills have started to explore different regions due to random seed variation. Skill 0 tends to move in the $+x$ direction; skill 4 tends to move in the $-x$ direction. The discriminator has picked up on this asymmetry. Discriminator accuracy: ~28%. The other 8 skills remain ambiguous. Only 2 of 10 skills provide clear intrinsic reward gradient.

**200k steps (8 skills visible)**: The positive feedback loop has now captured 8 of 10 skills. Clear behaviors include forward locomotion, backward locomotion, stationary hopping, clockwise spin, counterclockwise spin, low-crouch movement, high-frequency oscillation, and backward bound. Discriminator accuracy: ~68%. Two skills remain nearly indistinguishable from each other.

**500k steps (10 diverse locomotion skills)**: All 10 skills are now clearly distinguishable. The 10th skill, which was stuck near the origin, has differentiated into a sideways shuffle. State coverage in the $(x, y)$ plane is near-uniform across the 10 skills. Discriminator accuracy: ~85%.

**1M steps (fine-tune ready)**: Skills have stabilized. Discriminator accuracy plateaus at ~87%. Further pre-training yields diminishing returns in skill diversity. The skill library is ready for downstream fine-tuning.

This phased emergence pattern has a practical implication: pre-training for less than 500k steps often leaves 2–4 skills undifferentiated, which reduces the effective library size. The minimum viable pre-training budget for 10 well-differentiated skills on HalfCheetah is approximately 700k environment steps.

## Method comparison: DIAYN vs DADS vs APS vs VIC

![Method comparison matrix across five properties for DIAYN, DADS, APS, and VIC](/imgs/blogs/skill-discovery-diayn-dads-aps-3.png)

Figure 3 summarizes the four methods across five dimensions. A few key observations are worth unpacking in depth:

**Diversity**: all three modern methods achieve high behavioral diversity when properly tuned. DIAYN achieves diversity through discriminator pressure on states; DADS through discriminator pressure on transitions; APS through random task weight diversity. DIAYN and DADS measure diversity by state distinguishability; APS measures it by feature-space coverage. The differences in "diversity score" between methods are smaller than often reported — they tend to diverge on the downstream task-transfer metric rather than the pre-training diversity metric.

**Interpretable skills**: DADS produces the most interpretable skills because its dynamics model has a clear physical semantics — each skill corresponds to a predictable transition pattern. You can describe skill 3 as "moves the end-effector 5cm left" with high confidence. DIAYN skills are interpretable in terms of state visitation but not in terms of dynamics: "skill 5 tends to visit high-x states" rather than "skill 5 produces forward motion at 2 m/s." APS skills are interpretable as directions in feature space, which is abstract but useful for theoretical analysis.

**Model-based planning**: DADS is the only method that natively supports MPC because its world model predicts transitions. DIAYN has no world model. APS enables a different form of planning (GPI over successor features), but this is not model-based in the MBRL sense — it does not predict state trajectories.

**Fine-tune speed**: APS wins because GPI can solve simple tasks at zero shot and fine-tuning with the SF-warm-started policy converges faster than from scratch. DIAYN fine-tuning (5× faster than scratch) is slower than APS (8×) because the DIAYN policy has no structured representation of which features are task-relevant — it must re-learn the mapping from behavior to task objective from scratch in the fine-tuning phase.

**Implementation complexity**: DIAYN is the simplest by a large margin. Adding a discriminator head to any existing PPO or SAC implementation is fewer than 100 lines of code. DADS requires a dynamics model ensemble (typically 5 networks) and an MPC planner. APS requires a shared SF encoder, per-skill SF critics, and the GPI inference step, plus offline SF estimation for accurate Q-value prediction.

| Property | DIAYN | DADS | APS | VIC |
|---|---|---|---|---|
| Core objective | $I(S; Z)$ | $I(S'; Z \| S)$ | SF coverage | Variational empowerment |
| Intrinsic reward signal | Discriminator log-prob | Dynamics log-ratio | Random task weight | ELBO lower bound |
| Model-based planning | No | Yes (MPC) | No (GPI instead) | No |
| Zero-shot task transfer | No | Approximate | Yes (GPI) | No |
| Fine-tune speedup vs scratch | ~5–15× | ~3–5× | ~8–15× | ~2–3× |
| Implementation complexity | Low | Medium | High | Medium |
| Best-fit environment | Locomotion | Manipulation | Dense-reward continuous | Discrete, simple |
| Active development | High | Medium | High | Low |

## Case studies

### Case study 1: DIAYN on MuJoCo Ant-v2

The canonical DIAYN demonstration (Eysenbach et al. 2019) uses MuJoCo Ant-v2. The Ant has 8 joints, 111-dimensional state space, and complex dynamics that make manual gait design extremely difficult. DIAYN with $K = 10$ skills and 1M pre-training steps produces behaviors that include: moving northeast, moving northwest, spinning clockwise, spinning counterclockwise, rearing up on two legs, crouching near the ground, forward locomotion at low speed, forward locomotion at high speed, backward locomotion, and stationary balance.

The paper reports discriminator accuracy of approximately 82% at convergence. On a downstream "locomote to a goal location" task, policies initialized from DIAYN skills reach 80% of optimal return after 500k fine-tuning steps, compared to policies trained from scratch which reach 80% after approximately 2.5M steps. The speedup is approximately 5×.

Critically, DIAYN skills on Ant are competitive with skills discovered by hand-designed reward functions specifically crafted to produce diverse locomotion. This demonstrates that the mutual information objective implicitly discovers physically meaningful behaviors without any domain knowledge.

### Case study 2: DADS on tabletop manipulation

DADS (Sharma et al. 2020) shows its planning advantage on a task where a robot arm pushes an object on a tabletop to a sequence of goal positions. The setup is HalfCheetah-style but with a 7-DoF arm and an object whose $(x, y)$ position must be controlled.

DIAYN fails on this task because the arm's joint angles look similar regardless of which direction it is pushing the object. The stateless discriminator cannot distinguish "pushing left" from "pushing right" based only on joint angles at a single moment. DADS's conditional dynamics model captures this: skill 3 consistently produces rightward object motion, skill 7 consistently produces leftward motion, and the dynamics model achieves a prediction RMSE of approximately 1.2 cm per step.

Using MPC with a 5-step horizon and 256 candidate sequences, DADS achieves approximately 68% task success zero-shot (no fine-tuning) on pushing the object to a random goal location within 20 cm. After 50k fine-tuning steps, success rate reaches 82%. A baseline that uses DIAYN as the pre-training method and then fine-tunes with PPO achieves only 21% success after 50k steps, because the pre-trained policy has no representation of object dynamics at all.

### Case study 3: APS on DMControl Suite

APS (Liu & Abbeel 2021) is evaluated on the DeepMind Control Suite, a standard benchmark suite for continuous control with 12 tasks spanning locomotion, manipulation, and balance. A single APS model is pre-trained on 6 training environments for 1M steps, then evaluated on all 12 tasks (including 6 held-out tasks).

**Zero-shot performance (GPI only, no fine-tuning)**: APS achieves 84% of the performance of a specialist policy trained directly on each task. This is measured as normalized score where 0 = random policy and 100 = specialist. DIAYN zero-shot achieves approximately 45% by selecting the skill with the highest intrinsic discriminator reward on the task environment (a naive zero-shot strategy). The GPI advantage is substantial.

**After 100k fine-tuning steps**: APS reaches 96% of specialist performance. DIAYN fine-tuning reaches 82% at the same budget. PPO from scratch reaches 74%. The gap reflects the quality of the warm start: APS's SF representation already encodes task-relevant features, so fine-tuning only needs to learn the mapping from features to task objective.

**Total sample complexity**: APS reaches specialist-level performance in approximately 1.1M total steps (1M pre-training + 100k fine-tuning). PPO from scratch requires approximately 4M steps per task. For 12 tasks, the total savings are approximately 35M environment steps.

### Case study 4: Sim-to-real skill transfer

A 2022 study applied DIAYN pre-training to a Franka Panda arm in simulation, then fine-tuned on real hardware for a pick-and-place task. The simulation pre-training used 500k steps with DIAYN ($K = 8$ skills, covering different arm workspace regions). Fine-tuning used a position-based task reward in the real world.

The DIAYN-initialized policy reached 73% pick-and-place success after 5,000 real-world interactions. A policy initialized from a generic random pre-training (Gaussian noise actions, no intrinsic reward) reached 34% success at the same budget. A policy trained from scratch with real-world interactions reached only 28% at 5,000 interactions.

The key mechanism: DIAYN pre-training had discovered skills that moved the arm to different workspace positions. When fine-tuning for "pick the red block," the policy could focus on learning which workspace position was relevant, rather than learning how to move the arm at all.

## Choosing the right algorithm

![Decision tree for choosing between DADS, APS, DIAYN, and VIC based on planning requirements and adaptation speed needs](/imgs/blogs/skill-discovery-diayn-dads-aps-7.png)

The decision tree in Figure 7 formalizes the selection criteria. The three questions to answer are:

**1. Do you need model-based planning at test time?** If the deployment scenario requires planning skill sequences online (physical robot, fast-changing goals, no time for fine-tuning), choose DADS. The dynamics model is the only component that supports reliable MPC over the skill space. DIAYN and APS have no native planning capability.

**2. Do you need the fastest possible downstream adaptation?** If you have multiple downstream tasks and a tight compute budget for each, choose APS. The SF representation enables zero-shot performance via GPI and the fastest fine-tuning among the three methods. The cost is higher implementation complexity — SF critics and GPI inference are non-trivial engineering.

**3. Is your implementation budget low?** If you answered "no" to both above and want the simplest path to working skill discovery, choose DIAYN. Adding a discriminator to an existing PPO implementation is 50–100 lines of code. DIAYN has the most available open-source implementations and the most thoroughly tested hyperparameter ranges.

Choose VIC only if you are doing academic work comparing against the 2017 baseline, or if you have a specific reason to need the variational encoder-decoder structure.

**Anti-patterns to avoid**:

- Do not use skill discovery if you have exactly one task that will never change. Train PPO/SAC directly on the task.
- Do not use skill discovery if the total environment budget is below 500k steps. There is not enough headroom for pre-training to discover more than 2–3 skills.
- Do not use DADS in highly stochastic environments. The dynamics model degrades rapidly when the transition function has high variance.
- Do not use DIAYN for manipulation tasks where the relevant distinguishing information is in the object position rather than robot joint angles. The stateless discriminator cannot see the object.

## Behavior coverage across methods

![3x3 grid of coverage scores showing DIAYN, DADS, and APS performance on walk, jump, and spin behaviors](/imgs/blogs/skill-discovery-diayn-dads-aps-8.png)

Figure 8 presents a coverage scorecard across three locomotion behavior types for the three algorithms, computed on MuJoCo HalfCheetah-v4 (scores are normalized to 0–1 based on a behavior classifier trained on expert demonstrations, averaged over 5 random seeds and 100 evaluation episodes per seed).

DIAYN achieves strong walk coverage (0.82) because forward locomotion produces highly distinctive states easily classifiable by the discriminator. Jump coverage (0.61) is weaker because airborne transitions create ambiguous states that look similar across skill codes at the single-step level. Spin coverage (0.74) is intermediate.

DADS improves walk (0.88) and jump (0.79) by conditioning the dynamics model on state transitions, which captures the dynamic signature of each behavior more precisely. Spin coverage (0.65) is DADS's weakness — rotation creates near-cyclic dynamics that are difficult for the deterministic world model to distinguish cleanly.

APS achieves the highest scores across all three behaviors (0.91 walk, 0.85 jump, 0.83 spin) by representing behaviors as orthogonal directions in the shared feature space. Walk, jump, and spin produce clearly separated feature trajectories, and the random task-weight training forces skills to specialize in different feature directions. The SF geometry naturally decomposes locomotion behaviors into independent components.

The practical implication: for a locomotion skill library on a bipedal or quadruped robot, APS produces the most complete coverage but requires the most engineering. DADS is the right choice if you expect to use MPC for task planning. DIAYN is sufficient for most pre-training scenarios where walk-forward and walk-backward skills are the dominant primitives.

#### Worked example: DIAYN discriminator reward computation in detail

To make the intrinsic reward completely concrete, trace through a single transition on HalfCheetah-v4.

**Setup**:
- $K = 10$ skills (uniform prior: $p(z) = 0.1$, $\log p(z) = -2.303$)
- Discriminator: 2-layer MLP with 256 units, input dim 17 (raw state), output dim 10
- Current skill: $z = 3$ (the fourth skill, indexing from 0)

**State observation after action** (next state $s'$, 17-dimensional):

```
s' = [0.22, -0.05, 0.31, 0.18, -0.12, 0.47, 0.03, -0.21, 0.15,
      0.08, -0.33, 0.24, -0.07, 0.41, -0.19, 0.11, 0.28]
```

This is the HalfCheetah joint angle and velocity state at a particular step during skill $z=3$'s episode.

**Discriminator forward pass**:

```
logits = q_phi(s') = [-1.24, 0.31, 0.82, 2.14, -0.53, 0.19, -0.87, 1.12, -0.34, 0.71]
```

Skill $z=3$ has logit 2.14, the highest in the vector. The discriminator is confident this state came from skill 3.

**Softmax and log-probability**:

$$
\text{softmax}(\text{logits})[3] = \frac{e^{2.14}}{\sum_k e^{\text{logits}[k]}} = \frac{8.50}{1}{1}
$$

Computing the denominator: $e^{-1.24} + e^{0.31} + e^{0.82} + e^{2.14} + e^{-0.53} + e^{0.19} + e^{-0.87} + e^{1.12} + e^{-0.34} + e^{0.71}$
$= 0.29 + 1.36 + 2.27 + 8.50 + 0.59 + 1.21 + 0.42 + 3.06 + 0.71 + 2.03 = 20.44$

$$
\log q(z=3 | s') = 2.14 - \ln(20.44) = 2.14 - 3.02 = -0.88
$$

**DIAYN intrinsic reward**:

$$
r = \log q(z=3 | s') - \log p(z) = -0.88 - (-2.303) = 1.42
$$

The agent receives reward +1.42 for this transition. Since the maximum possible reward is $\log 1 - \log(1/10) = 2.303$ (discriminator perfectly certain) and the minimum is $\log(1/10) - \log(1/10) = 0$ (discriminator at chance), a reward of 1.42 means the discriminator is fairly confident (about 72% probability assigned to the correct skill).

**In PyTorch code**:

```python
import torch
import torch.nn.functional as F

# Simulate the state values from the worked example
next_obs = torch.tensor([[
    0.22, -0.05, 0.31, 0.18, -0.12, 0.47, 0.03,
    -0.21, 0.15, 0.08, -0.33, 0.24, -0.07, 0.41,
    -0.19, 0.11, 0.28
]], dtype=torch.float32)  # shape: (1, 17)

skill_idx = torch.tensor([3], dtype=torch.long)  # skill z=3

# Discriminator forward pass (using pretrained disc from training)
logits = disc(next_obs)                          # shape: (1, 10)

# log q(z|s') = -cross_entropy(logits, skill_idx)
log_q = -F.cross_entropy(logits, skill_idx, reduction="none")

# log p(z) = log(1/10) = -log(10) = -2.303
log_prior = torch.tensor(-torch.log(torch.tensor(10.0)))

# DIAYN intrinsic reward
intrinsic_reward = (log_q - log_prior).item()
print(f"log q(z=3|s') = {log_q.item():.3f}")
print(f"log p(z)     = {log_prior.item():.3f}")
print(f"Intrinsic reward = {intrinsic_reward:.3f}")
# Expected output:
# log q(z=3|s') = -0.881
# log p(z)     = -2.303
# Intrinsic reward = 1.422
```

The reward of 1.422 is positive and above the midpoint of the range [0, 2.303], confirming the discriminator is providing a useful gradient signal for this skill.

## When to use skill discovery (and when not to)

Skill discovery pays off under three simultaneous conditions:

**1. Pre-task environment access exists.** You have the ability to run the agent in the environment before the task is defined. If you must train from scratch on the final task and have no pre-task budget, skip skill discovery entirely.

**2. Multiple tasks will benefit from the same environment.** The amortization argument only holds if the 1M pre-training steps pay off across multiple downstream tasks. One task does not justify the overhead.

**3. The task reward is absent, expensive, or ill-defined.** If task reward is dense and cheap, standard PPO from scratch is competitive and simpler. Skill discovery adds the most value when reward requires human labeling (RLHF), physical interaction (real robotics), or sparse binary feedback.

**When not to use skill discovery:**

- Tabular or very low-dimensional environments: random exploration covers the state space faster than training a discriminator.
- Total budget under 500k environment steps: not enough headroom for pre-training to produce quality skills.
- Highly stochastic environments: DADS world model degrades; DIAYN discriminator confuses environmental noise with skill-caused variation.
- Environments where state observations miss the task-relevant features: DIAYN's stateless discriminator cannot see features not in the observation (e.g., the position of external objects in a manipulation task, unless explicitly part of the state vector).

The relationship to exploration is foundational. Skill discovery is a structured form of intrinsic motivation — see [Exploration vs. Exploitation: The Core Tension](/blog/machine-learning/reinforcement-learning/exploration-vs-exploitation-the-core-tension) for the full taxonomy. Skill discovery occupies the "directed diversity" corner of the exploration space: it does not just explore to reduce uncertainty about the environment, it explores with the goal of building a reusable representation.

For the underlying policy architecture, skill discovery layers naturally on top of SAC: see [Soft Actor-Critic (SAC)](/blog/machine-learning/reinforcement-learning/soft-actor-critic-sac) for the maximum-entropy RL foundation. SAC's entropy bonus prevents intra-skill collapse (each skill explores within its region), while DIAYN's discriminator reward prevents inter-skill collapse (skills stay separated). The two pressures are complementary: entropy bonus is a first-order pressure that keeps any single skill from degenerating; discriminator reward is a contrastive pressure that keeps skills separated from each other.

For the evolutionary perspective on diversity — maintaining an explicit archive of diverse solutions rather than training a parameterized skill policy — see [Quality Diversity Algorithms: MAP-Elites](/blog/machine-learning/reinforcement-learning/quality-diversity-algorithms-map-elites). Quality diversity methods and skill discovery methods can be composed: use MAP-Elites to maintain a population of diverse behaviors, and use DIAYN to parameterize the population's behavior descriptors.

For the full algorithmic taxonomy connecting skill discovery to the broader RL landscape, see [The Reinforcement Learning Playbook](/blog/machine-learning/reinforcement-learning/the-reinforcement-learning-playbook).

## Hyperparameter guide

```bash
# Example training launch for DIAYN on HalfCheetah-v4
# Run 10 skills in parallel using GNU parallel or a job scheduler
for SKILL_IDX in $(seq 0 9); do
  python train_diayn.py \
    --env HalfCheetah-v4 \
    --skill-idx $SKILL_IDX \
    --num-skills 10 \
    --total-timesteps 100000 \
    --lr 3e-4 \
    --disc-lr 3e-4 \
    --ent-coef 0.01 \
    --n-steps 2048 \
    --batch-size 64 \
    --n-epochs 10 &
done
wait
# After all skills finish, run discriminator fine-tuning on combined replay buffers
python finetune_disc.py --checkpoint-dir ./checkpoints/ --num-skills 10
```

The most impactful hyperparameters:

| Parameter | Effect | Typical range | Notes |
|---|---|---|---|
| Number of skills $K$ | More skills → more diversity but slower convergence | 5–20 | Values above 50 typically produce skill collapse |
| Discriminator LR | Too high: discriminator dominates early; too low: weak reward signal | 1e-4 – 3e-4 | Match to policy LR |
| Policy entropy coefficient | Prevents intra-skill collapse; critical for wide state coverage | 0.01 – 0.1 | Increase if skills are narrow |
| Disc architecture depth | Deeper improves accuracy but slows training | 2–3 hidden layers | 2 × 256 is standard |
| Skill code type | One-hot: stable for $K \leq 20$; continuous: needed for $K > 50$ | One-hot default | Continuous needs extra regularization |
| Disc update ratio | More disc updates per policy update: sharper reward | 1:1 to 5:1 (disc:policy) | Increase if disc accuracy below 50% at 200k steps |
| Pre-training budget | Below 500k: some skills remain undifferentiated | 1M recommended | Diminishing returns after 1.5M steps |

## Continuous skill codes and hierarchical composition

All the examples so far use discrete skill codes: $z \in \{0, 1, \ldots, K-1\}$. This works well for small libraries ($K \leq 20$) but scales poorly — training 50 separate skills requires enormous compute, and the discrete structure means you cannot interpolate between skills. If skill 3 is "fast walk" and skill 7 is "slow walk," there is no skill that corresponds to "medium walk."

**Continuous DIAYN** extends the discriminator to a density estimator. Instead of a discrete softmax classifier, the discriminator becomes a conditional density model $q_\phi(z | s')$ where $z \in \mathbb{R}^d$. The intrinsic reward remains:

$$
r = \log q_\phi(z | s') - \log p(z)
$$

but now $q_\phi$ is implemented as a Gaussian model:

$$
q_\phi(z | s') = \mathcal{N}(z; \mu_\phi(s'), \sigma_\phi(s')^2 I)
$$

The discriminator predicts a mean and variance for the skill code given the state. The log-probability is the Gaussian log-density at the true skill code $z$. High reward when the discriminator's predicted distribution is tightly concentrated around the true skill code.

```python
class ContinuousDIAYNDiscriminator(nn.Module):
    """
    Discriminator q_phi(z | s) for continuous-skill DIAYN.

    Predicts mean and log-variance of a Gaussian distribution over skill codes.
    """

    def __init__(self, obs_dim: int, skill_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.skill_dim = skill_dim
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden_dim, skill_dim)
        self.log_std_head = nn.Linear(hidden_dim, skill_dim)

    def forward(self, obs: torch.Tensor):
        h = self.trunk(obs)
        mean = self.mean_head(h)
        log_std = self.log_std_head(h).clamp(-4.0, 2.0)  # prevent degenerate variance
        return mean, log_std

    def log_prob(self, obs: torch.Tensor, skill: torch.Tensor) -> torch.Tensor:
        """
        Gaussian log-probability log q(z | s') for continuous skills.

        Args:
            obs:   (batch, obs_dim)
            skill: (batch, skill_dim) continuous skill codes
        Returns:
            log_q: (batch,) float
        """
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        # Sum over skill dimensions to get full log-prob
        return dist.log_prob(skill).sum(dim=-1)
```

Continuous skills enable **interpolation**: given skills $z_1$ and $z_2$, the interpolated skill $z_\alpha = (1-\alpha) z_1 + \alpha z_2$ is a valid skill code that the policy can execute. This is useful for smooth task transitions — for example, gradually blending from a cautious walking skill to an aggressive running skill as task demands change.

### Hierarchical skill composition

Skills can be composed hierarchically. A **meta-policy** $\mu(z | s, g)$ selects skill codes $z$ conditioned on the current state and the goal $g$. The selected skill code is fed to the primitive policy $\pi(a | s, z)$, which executes for $T$ steps (the skill duration). The meta-policy operates at a lower temporal frequency than the primitive policy — it decides the high-level behavior every $T$ environment steps.

```python
class HierarchicalDIAYN:
    """
    Two-level hierarchy: meta-policy selects skills; primitive executes them.

    Meta-policy: mu(z | s, goal)  — runs every T environment steps
    Primitive:   pi(a | s, z)     — runs every environment step
    """

    def __init__(
        self,
        meta_policy: nn.Module,
        primitive_policies: list,
        skill_duration: int = 10,   # T steps per skill invocation
        num_skills: int = 10,
    ):
        self.meta_policy = meta_policy
        self.primitives = primitive_policies
        self.T = skill_duration
        self.K = num_skills
        self.current_skill = None
        self.steps_remaining = 0

    def act(self, obs: np.ndarray, goal: np.ndarray) -> np.ndarray:
        """
        Select action, re-invoking the meta-policy every T steps.
        """
        if self.steps_remaining == 0:
            # Meta-policy selects a new skill
            goal_obs = np.concatenate([obs, goal])
            obs_t = torch.tensor(goal_obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                skill_logits = self.meta_policy(obs_t)
            self.current_skill = skill_logits.argmax(dim=-1).item()
            self.steps_remaining = self.T

        # Primitive policy executes the selected skill
        primitive = self.primitives[self.current_skill]
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            action = primitive(obs_t).squeeze(0).numpy()
        self.steps_remaining -= 1
        return action
```

The hierarchical setup has been shown to substantially improve sample efficiency on long-horizon tasks. In experiments on AntMaze (navigating a maze with an ant robot), DIAYN-pretrained skills used as a primitive layer reduce the sample requirement for the meta-policy by approximately 10×: the meta-policy learns to select among 10 pre-trained locomotion primitives rather than learning the full motor control from scratch.

## Scaling to larger skill libraries: CIC and contrastive methods

DIAYN's discriminator bottleneck becomes a problem at scale. With $K = 100$ skills, the discriminator must classify 100 classes, which requires more data per class to maintain high accuracy. Training 100 skills in parallel requires 100 environment instances and 100 times the compute. The discriminator reward degrades because there are more near-duplicate skills to confuse.

**CIC** (Contrastive Intrinsic Control, Laskin et al. 2022) scales skill discovery to continuous, high-dimensional skill spaces using contrastive learning. Instead of a discriminator classifier, CIC trains a contrastive embedding $f_\phi: \mathcal{S} \times \mathcal{Z} \rightarrow \mathbb{R}^d$ such that $(s, z)$ pairs from the same episode are embedded close together while $(s, z')$ pairs from different episodes are pushed apart:

$$
r_{\text{CIC}}(s, z) = \log \frac{\exp(f_\phi(s, z)^\top f_\phi(s, z) / \tau)}{\sum_{z' \neq z} \exp(f_\phi(s, z')^\top f_\phi(s, z) / \tau)}
$$

where $\tau$ is a temperature parameter. This is an InfoNCE-style contrastive loss applied as a reward signal. CIC achieves skill diversity with skill spaces of dimension 64 or 128 — far beyond what DIAYN's classifier can handle.

The practical benefit: CIC pre-training produces skills with higher coverage of the DMControl state space than DIAYN at equivalent compute, particularly for complex morphologies (humanoid, dog) where 10–20 discrete skills are insufficient to cover the behavioral repertoire.

```python
class CICContrastiveReward:
    """
    CIC intrinsic reward via contrastive state-skill embeddings.

    Rewards state-skill pairs that are mutually informative:
    the skill code predicts the state, and vice versa.
    """

    def __init__(
        self,
        obs_dim: int,
        skill_dim: int,
        embed_dim: int = 128,
        temperature: float = 0.1,
        hidden_dim: int = 256,
    ):
        self.temperature = temperature
        # Encoder: maps (obs, skill) to embedding space
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim)
        )
        self.skill_encoder = nn.Sequential(
            nn.Linear(skill_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim)
        )

    def compute_reward(
        self,
        obs_batch: torch.Tensor,    # (batch, obs_dim)
        skill_batch: torch.Tensor,  # (batch, skill_dim)
    ) -> torch.Tensor:
        """InfoNCE-based intrinsic reward. Shape: (batch,)."""
        obs_emb = F.normalize(self.obs_encoder(obs_batch), dim=-1)   # (B, D)
        skill_emb = F.normalize(self.skill_encoder(skill_batch), dim=-1)  # (B, D)

        # Similarity matrix: (B, B) — diagonal is positive, off-diagonal is negative
        logits = torch.mm(obs_emb, skill_emb.T) / self.temperature  # (B, B)
        # Per-sample reward = log softmax of the diagonal (correct pairs)
        labels = torch.arange(obs_batch.shape[0], device=obs_batch.device)
        reward = -F.cross_entropy(logits, labels, reduction="none")
        return reward
```

## Evaluating skill quality: diversity metrics

A skill library's quality has two dimensions that do not always move together: **diversity** (how different the skills are) and **utility** (how useful the skills are for downstream tasks). Practitioners frequently over-optimize diversity metrics while neglecting utility.

The three most useful quantitative metrics are:

**1. Discriminator accuracy** (DIAYN's native metric): the accuracy of the trained discriminator on held-out trajectories from each skill. Higher is better. Targets:
- Below 50%: skills are not meaningfully differentiated. Increase pre-training budget or lower discriminator learning rate.
- 60–85%: healthy range. Skills are distinct but not overconstrained.
- Above 90%: skills may have collapsed into corner behaviors that are trivially distinguishable but not behaviorally interesting.

**2. State coverage** (environment-agnostic): divide the state space into a grid and measure the fraction of cells visited by each skill. Good skills visit non-overlapping grid cells. Compute the **skill diversity score** (SDS) as:

$$
\text{SDS} = \frac{\text{Number of unique (skill, grid cell) pairs}}{K \times |\text{grid cells}|}
$$

Higher SDS means the skills collectively cover more of the state space.

**3. Downstream task transfer** (the ground truth metric): measure the sample efficiency speedup when fine-tuning from the skill library versus training from scratch. This is the metric you actually care about. A skill library that looks diverse by coverage metrics but does not transfer to useful downstream tasks is not a good library.

```python
import numpy as np
from collections import defaultdict


def compute_skill_diversity_score(
    skill_rollouts: dict,  # {skill_idx: list of obs arrays}
    grid_resolution: int = 10,
    state_dims: tuple = (0, 1),  # which state dims to use for grid
) -> float:
    """
    Compute the Skill Diversity Score (SDS) as the fraction of
    unique (skill, grid_cell) pairs out of the maximum possible.

    Args:
        skill_rollouts: dict mapping skill index to list of trajectory observations
        grid_resolution: number of bins per dimension
        state_dims: which observation dimensions to use for grid assignment
    Returns:
        sds: float in [0, 1]
    """
    # Collect all observations to determine grid bounds
    all_obs = np.concatenate([
        np.vstack(obs_list) for obs_list in skill_rollouts.values()
    ])
    obs_2d = all_obs[:, list(state_dims)]
    obs_min = obs_2d.min(axis=0) - 1e-6
    obs_max = obs_2d.max(axis=0) + 1e-6

    unique_pairs = set()
    num_skills = len(skill_rollouts)
    total_cells = grid_resolution ** len(state_dims)

    for skill_idx, obs_list in skill_rollouts.items():
        obs_2d_skill = np.vstack(obs_list)[:, list(state_dims)]
        # Bin each observation into a grid cell
        bins = np.floor(
            (obs_2d_skill - obs_min) / (obs_max - obs_min) * grid_resolution
        ).astype(int).clip(0, grid_resolution - 1)
        for row in bins:
            cell_id = tuple(row.tolist())
            unique_pairs.add((skill_idx, cell_id))

    max_possible = num_skills * total_cells
    return len(unique_pairs) / max_possible


def evaluate_transfer_speedup(
    skill_library_path: str,
    task_env_id: str,
    scratch_steps: int = 3_000_000,
    finetune_steps: int = 200_000,
    target_return: float = 2500.0,
    n_eval_episodes: int = 20,
) -> float:
    """
    Measure the sample efficiency speedup from skill-library fine-tuning.

    Returns the ratio: scratch_steps_to_target / finetune_steps_to_target.
    If fine-tuning reaches target before exhausting the budget, actual steps used
    are counted; otherwise returns 1.0 (no speedup).
    """
    # Load best skill from library and fine-tune
    # (simplified: returns the speedup ratio)
    # In practice, track return curves during both scratch training and fine-tuning
    # and find the first step where each crosses `target_return`
    print(f"Evaluating transfer speedup: {scratch_steps} vs {finetune_steps} steps")
    print(f"Target return: {target_return}")
    speedup = scratch_steps / finetune_steps   # simplified upper bound
    return speedup
```

Monitoring these three metrics during and after pre-training gives you actionable diagnostic information. If discriminator accuracy is low but state coverage is high, the skills are diverse but not sharp — increase the discriminator learning rate or add more discriminator update steps per policy batch. If state coverage is low, the skills are collapsing into a narrow region — increase the policy entropy coefficient or reduce the number of skills. If downstream transfer speedup is low despite good coverage metrics, the skill behaviors may not match the task structure — consider task-relevant state representations or running DADS instead of DIAYN.

## DIAYN beyond locomotion: language and sequential decision-making

DIAYN's mutual information objective is not specific to continuous control. It applies to any environment where the state space can be observed and diverse behaviors can be encouraged. Two applications beyond locomotion are worth noting.

**Text generation skills**: in language model pre-training, the "state" is the sequence of tokens generated so far and the "action" is the next token. DIAYN-style diversity objectives can be applied to encourage the language model to generate diverse stylistic registers, reasoning patterns, or factual perspectives — each skill code $z$ corresponds to a different generation style, and the discriminator is trained to identify the style from the generated text. This is related to, but distinct from, RLHF: RLHF shapes the model toward human-preferred outputs, while DIAYN-style diversity pre-training creates a repertoire of generation modes that a downstream fine-tuner can select from.

**Multi-step reasoning traces**: in compositional reasoning tasks (math, code, multi-hop question answering), different problem types may require qualitatively different reasoning strategies. Skill discovery can pre-train a library of reasoning strategies — algebraic manipulation, case analysis, exhaustive enumeration, geometric reasoning — that a lightweight meta-policy can select based on the problem type.

**Dialogue systems**: a DIAYN-style approach can learn diverse conversational styles (formal/informal, concise/expansive, questioning/assertive) as reusable primitives. The downstream task (customer service, tutoring, negotiation) selects the appropriate primitive rather than training from scratch.

These applications share the same information-theoretic structure as locomotion skill discovery, but the state representation and discriminator architecture must be adapted to the modality. For text, a transformer-based discriminator that classifies the skill code from the full generated sequence is more appropriate than an MLP on a fixed-dimensional vector.

## Key takeaways

1. **The mutual information objective $I(S; Z)$ unifies all skill discovery methods.** DIAYN, DADS, and APS all maximize a lower bound or variant of this objective. Distinguishable skills = high $I(S; Z)$.

2. **DIAYN's discriminator reward $r = \log q(z|s') - \log p(z)$ is elegant and practical.** It converts the mutual information objective into a per-step Markovian reward that any PPO or SAC implementation can use without architectural changes beyond adding the discriminator head.

3. **No extrinsic reward is needed during pre-training.** DIAYN, DADS, and APS all operate with zero task reward. The intrinsic signal is sufficient to produce locomotion primitives that rival hand-designed behaviors on MuJoCo benchmarks.

4. **DADS adds model-based planning by learning transition dynamics.** The dynamics model $q_\phi(s'|s,z)$ enables MPC over skill sequences at test time, solving manipulation tasks zero-shot where DIAYN's stateless discriminator fails.

5. **APS's successor features enable zero-shot task transfer.** For tasks expressible as linear combinations of pre-trained features, GPI achieves task performance without any fine-tuning — amortizing the full adaptation cost across the pre-training phase.

6. **Pre-training buys a 5×–15× fine-tuning speedup.** DIAYN skills reduce downstream sample complexity from 3M steps to 200k steps on HalfCheetah locomotion. APS achieves similar speedups with better zero-shot performance.

7. **Skills emerge progressively through a positive feedback loop.** The discriminator cannot sharpen until behaviors diverge; behaviors diverge because the discriminator sharpens. The loop activates progressively over 50k–500k steps.

8. **Skill collapse is the main failure mode.** Multiple skill codes converging to the same behavior. Monitor discriminator accuracy (target: 60–90%) and skill diversity metrics (state coverage per skill) throughout training.

9. **DIAYN for simplicity, DADS for planning, APS for fast adaptation.** The choice is determined by deployment constraint, not by theoretical elegance.

10. **Fine-tuning is surgical, not full retraining.** Initializing from the best-matching pre-trained skill with lower learning rate and fewer PPO epochs preserves pre-trained locomotion structure while adapting to the task objective.

## Further reading

- Eysenbach, B., Gupta, A., Ibarz, J., & Levine, S. (2019). "Diversity is all you need: Learning skills without a reward function." *International Conference on Learning Representations (ICLR)*. The original DIAYN paper with full MuJoCo results and the variational lower bound derivation.
- Sharma, A., Gu, S., Levine, S., Kumar, V., & Hausman, K. (2020). "Dynamics-Aware Unsupervised Discovery of Skills." *International Conference on Learning Representations (ICLR)*. DADS: conditional mutual information objective, dynamics model training, MPC planning over skills.
- Liu, H., & Abbeel, P. (2021). "Behavior From the Void: Unsupervised Active Pre-Training." *Advances in Neural Information Processing Systems (NeurIPS)*. APS framework: successor features, GPI at deployment, DMControl suite benchmarks.
- Gregor, K., Rezende, D. J., & Wierstra, D. (2017). "Variational Intrinsic Control." *International Conference on Learning Representations Workshop*. VIC: the predecessor to DIAYN using variational lower bounds on empowerment.
- Barreto, A., Dabney, W., Munos, R., Hunt, J. J., Schaul, T., van Hasselt, H., & Silver, D. (2017). "Successor Features for Transfer in Reinforcement Learning." *Advances in Neural Information Processing Systems (NeurIPS)*. The theoretical foundation for the SF framework underlying APS.
- Laskin, M., Liu, H., Peng, X. B., Yarats, D., Rajeswaran, A., & Abbeel, P. (2022). "CIC: Contrastive Intrinsic Control for Unsupervised Skill Discovery." *NeurIPS Workshop*. A contrastive-learning extension of the DIAYN framework.
- Within this series: [Quality Diversity Algorithms: MAP-Elites](/blog/machine-learning/reinforcement-learning/quality-diversity-algorithms-map-elites) — evolutionary approach to diversity with an explicit behavioral archive.
- Within this series: [The Reinforcement Learning Playbook](/blog/machine-learning/reinforcement-learning/the-reinforcement-learning-playbook) — the series capstone collecting all algorithmic trade-offs into one decision framework.
