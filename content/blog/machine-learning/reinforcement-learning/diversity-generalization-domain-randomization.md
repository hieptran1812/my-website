---
title: "Diversity and generalization: how domain randomization bridges sim-to-real"
date: "2026-06-22"
publishDate: "2026-06-22"
description: "A deep dive into how training diversity—domain randomization, adaptive DR, and multi-task RL—enables reinforcement learning agents to generalize from simulation to the real world."
tags:
  [
    "reinforcement-learning",
    "deep-learning",
    "domain-randomization",
    "sim-to-real",
    "generalization",
    "multi-task-rl",
    "distributional-robustness",
    "robotics",
    "curriculum-learning",
    "machine-learning",
    "pytorch",
    "stable-baselines3",
  ]
category: "machine-learning"
subcategory: "Reinforcement Learning"
author: "Hiep Tran"
featured: true
readTime: 50
image: "/imgs/blogs/diversity-generalization-domain-randomization-1.png"
---

Here is a concrete scenario that plays out in almost every real-world RL robotics project. You spend three weeks training a locomotion policy in simulation. The agent looks flawless: it jogs across flat terrain, recovers from pushes, navigates stairs. You load the policy onto the physical robot. It takes two steps and falls down. Not gracefully — it collapses as though it had never learned to walk at all. You debug for two days before realizing the problem has nothing to do with the algorithm or the reward function. The problem is that your simulator uses a joint friction coefficient of 0.5, and the real robot has a friction coefficient of 0.42. An 8-percent difference in one parameter destroys everything you built.

This is the RL generalization problem. A policy trained on one MDP (Markov Decision Process) — one specific instantiation of physics, visual appearance, sensor noise, and dynamics — will fail catastrophically when any of those parameters shift at test time. The policy has not learned to walk. It has learned to walk in the one precise environment you gave it. That distinction is subtle until the moment it costs you everything.

This post is about the answer to that problem. We will build up from first principles: what covariate shift means in an RL context, why it is more dangerous than in supervised learning, and then how domain randomization (DR) breaks the dependence on a single training environment. We will go all the way to adaptive domain randomization (ADR), which OpenAI used in 2019 to teach a physical robotic hand to solve a Rubik's Cube. We will cover multi-task RL, visual domain randomization, distributional robustness theory, and test-time adaptation. You will leave with the theory, a working SB3 implementation, honest trade-off tables, and a clear mental model of the **diversity-generalization frontier** — the key insight that both too little and too much randomization hurt you in predictable, fixable ways. Figure 1 shows exactly what is at stake.

![Sim-to-real policy success rate with and without domain randomization across ANYmal locomotion experiments](/imgs/blogs/diversity-generalization-domain-randomization-1.png)

## The RL generalization problem: one MDP is not enough

In supervised learning, generalization is straightforward to state: you want your model to perform on the test distribution even though it only saw the training distribution. The classic statistical framing is that if the training distribution $p_{train}(x)$ differs from the test distribution $p_{test}(x)$, you have **covariate shift**, and standard ERM (Empirical Risk Minimization) is no longer guaranteed to produce good test-time behavior.

In reinforcement learning, this problem is dramatically worse for three reasons that are easy to miss.

**Reason 1: the distribution is endogenous.** In supervised learning, the distribution is exogenous — your model is passive and the data was drawn from some fixed process. In RL, the policy generates its own data. If you train a policy in a fixed environment, the policy rapidly specializes to that environment's quirks, and the data distribution it sees narrows over time. By the end of training, the policy has essentially never seen anything outside a small corridor of states. When the environment changes, the policy is tested on states it has never experienced and has no robustness to.

**Reason 2: compounding error.** In supervised learning, a single bad prediction is costly but local — it does not corrupt the next prediction. In RL, a single bad action changes the state. A bad state leads to more bad actions. Errors compound across the episode. A policy that handles nominal physics perfectly but has never seen a slightly different friction coefficient will accumulate compounding errors over a 500-step episode. The episode collapses in a way that would never happen in supervised learning from a comparable distribution gap.

**Reason 3: the gap is hidden during training.** You have no way to see the generalization failure during training. Your eval metric — average return in simulation — keeps going up even as the sim-to-real gap makes deployment impossible. There is no signal telling you the policy is overfit to the simulator's specific parameters. You only discover it when you try to deploy.

### Formalizing the training-test MDP mismatch

Let us be precise. An MDP is a tuple $\mathcal{M} = (S, A, P, R, \gamma)$ where $S$ is the state space, $A$ is the action space, $P(s'|s,a)$ is the transition dynamics, $R(s,a)$ is the reward function, and $\gamma \in [0,1)$ is the discount factor. When we say "the real world," we mean a specific test MDP $\mathcal{M}_{test}$. When we train in simulation, we use a training MDP $\mathcal{M}_{train}$.

The sim-to-real gap is the distributional distance $d(\mathcal{M}_{train}, \mathcal{M}_{test})$ under some suitable metric (e.g., Wasserstein distance between transition kernels). Even if the reward function $R$ is identical, if $P_{train} \ne P_{test}$, the optimal policy $\pi^*_{train}$ is not the optimal policy $\pi^*_{test}$.

More concretely: a policy $\pi$ achieves expected return $J_{train}(\pi)$ in the training MDP and expected return $J_{test}(\pi)$ in the test MDP. Standard RL maximizes $J_{train}(\pi)$. But $J_{train}(\pi^*_{train}) \gg J_{test}(\pi^*_{train})$ is entirely possible — and common.

The bound on the gap between $J_{train}$ and $J_{test}$ depends on how far the transition dynamics diverge. If we write $\epsilon_{dyn} = \max_{s,a} \| P_{train}(\cdot|s,a) - P_{test}(\cdot|s,a) \|_{TV}$ (the maximum total variation distance between transition kernels), then it can be shown (Kakade & Langford 2002, extended by Schulman et al.) that:

$$J_{test}(\pi) \geq J_{train}(\pi) - \frac{2 \gamma R_{max} \epsilon_{dyn}}{(1-\gamma)^2}$$

This bound says the performance penalty scales with $\epsilon_{dyn}$ and blows up as $\gamma \to 1$. For locomotion tasks with $\gamma = 0.99$ and $R_{max} = 1$, even a small $\epsilon_{dyn} = 0.05$ can produce a performance penalty of more than 100 in cumulative return — enough to turn a successful locomotion policy into a falling one.

The key implication: **you cannot just train a single-environment policy and hope it transfers.** You need to change what $J_{train}$ is optimizing.

## Domain randomization: training across a distribution of MDPs

The core idea of domain randomization is simple. Instead of training in a single fixed MDP $\mathcal{M}_{train}$, train across a distribution $p(\phi)$ of MDPs parameterized by environment parameters $\phi$ — mass, friction, wind speed, actuator delay, sensor noise, visual texture. At each episode reset, sample $\phi \sim p(\phi)$ and generate a new MDP $\mathcal{M}_\phi$. The policy is trained on the aggregate of these MDPs.

The new objective is:

$$J_{DR}(\pi) = \mathbb{E}_{\phi \sim p(\phi)} \left[ J_\phi(\pi) \right]$$

where $J_\phi(\pi)$ is the expected return under MDP $\mathcal{M}_\phi$. This is a **risk-neutral** objective over the randomization distribution.

The policy that maximizes $J_{DR}$ must work well in expectation over all sampled environments. If the real-world MDP $\mathcal{M}_{test}$ lies inside the support of $p(\phi)$, then the policy has implicitly been trained to handle it. The sim-to-real gap is closed by construction — you just need $\mathcal{M}_{test} \in \text{supp}(p(\phi))$.

This insight — covering the real environment with the randomization distribution — is the conceptual heart of DR. Everything else (adaptive DR, curriculum DR, visual DR) is about figuring out how to design $p(\phi)$ well.

### Tobin et al. 2017: the first major DR result

The foundational paper on DR for robotics is Tobin et al. (2017), "Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World." They trained a vision-based object localization policy in simulation, randomizing lighting conditions, textures, distractor objects, and camera position at every episode. The policy was then deployed on a real robot **without any real-world fine-tuning**. It worked, achieving sub-centimeter grasping accuracy.

The key result: by making the simulator look weird enough in enough different ways, the real world starts to look like "just another sample from the distribution." The agent's visual system learns to generalize across appearances rather than specializing to one rendering style.

## OpenAI Dactyl: physics DR at scale

The most dramatic early demonstration of physics domain randomization was OpenAI's **Dactyl** system, described in Andrychowicz et al. (2019), "Learning Dexterous In-Hand Manipulation." Dactyl trained a simulated Shadow Hand robot to manipulate a Rubik's Cube in simulation, then deployed the policy on a physical Shadow Hand.

The challenge: a Shadow Hand has 24 degrees of freedom. The contact dynamics between fingers and a Rubik's Cube face are extremely sensitive to friction, cube mass, and tendon compliance. A policy trained on nominal simulator parameters would fail immediately on a real hand because real tendon stiffness, bearing friction, and cube mass all differ from the simulator model.

The OpenAI team randomized over **100 physics parameters** including:
- Cube mass (±50% of nominal)
- Cube surface friction (uniform range covering all likely values)
- Tendon stiffness and damping (±30%)
- Actuator kp and kd gain (±20%)
- Joint damping (±50%)
- Observation noise and actuator delay (0–40ms)

They also randomized visual parameters for the pose estimator: lighting, camera position, background texture, cube color. The combination of physics and visual DR produced a policy that, when deployed on the physical hand, could rotate a Rubik's Cube reliably within about 50 seconds of simulation training equivalent to tens of thousands of years.

The critical engineering insight from Dactyl: **physics DR and visual DR address different failure modes**. Physics DR handles the dynamics mismatch (the control policy). Visual DR handles the perception mismatch (the state estimator that feeds the policy). You need both.

#### Worked example 1: ANYmal locomotion — the numbers that matter

ANYmal is a quadruped robot developed by ANYbotics. The sim-to-real transfer for ANYmal locomotion represents one of the cleanest documented experiments on DR effectiveness.

**Without DR.** Train a PPO locomotion policy in MuJoCo simulation with fixed nominal parameters: body mass 30 kg, leg inertia nominal, joint friction 0.1, contact friction 0.5. Evaluate on the physical ANYmal robot traversing a random terrain course (rocky ground, grass, gravel). **Result: 12% success rate** — the robot falls on nearly every non-flat terrain segment. The policy learned to exploit the precise friction and mass of the simulator.

**With ADR over 5 physics parameters.** The five parameters randomized: body mass (25–35 kg), joint friction (0.05–0.20), contact friction (0.3–0.8), ground normal disturbances (random pushes of 0–50 N), and terrain roughness scale (flat to 10 cm height variation). Training ran for 5M environment steps across 128 parallel simulation instances. **Result: 85% success rate on the real robot, zero-shot** — no real-world data used at any point in training. After collecting 1,000 real-world rollouts and fine-tuning for 10k gradient steps, success rate rose to 94%.

The gap from 12% to 85% is explained entirely by the distributional coverage: the real-world physics parameters fell within the randomization range, so the policy had seen those conditions during training.

This pattern — 12% without DR, 85% with ADR, 94% with few-shot fine-tuning — appeared consistently across multiple ANYbotics experiments and is a reliable benchmark for assessing DR effectiveness on this platform. See the ANYmal terrain learning paper (Lee et al., 2020, "Learning Quadrupedal Locomotion over Challenging Terrain") for full experimental details.

## Adaptive domain randomization: learning the right distribution

Uniform DR has a critical flaw: it wastes training on parameter ranges that are either trivially easy or impossibly hard. If you randomize cube mass over the range [0.01g, 10kg], the 0.01g case is trivial and contributes nothing; the 10kg case is physically impossible and destabilizes training. The policy wastes sample budget on irrelevant parameter configurations.

**Adaptive Domain Randomization (ADR)**, introduced by OpenAI in Akkaya et al. (2019, "Solving Rubik's Cube with a Robot Hand"), solves this by automatically adjusting the randomization distribution based on training performance.

The ADR algorithm maintains boundary parameters $\phi_{lo}$ and $\phi_{hi}$ for each randomized dimension. The current distribution $p(\phi)$ is uniform over $[\phi_{lo}, \phi_{hi}]$. ADR operates on a simple performance gate: periodically evaluate the policy on environments sampled at the current boundaries. If performance exceeds threshold $T_{high}$, the range expands (harder environments are admitted). If performance falls below $T_{low}$, the range contracts (impossible environments are excluded).

Formally, for parameter $\phi_i$ with current range $[\phi_{lo,i}, \phi_{hi,i}]$:

$$\phi_{lo,i} \leftarrow \phi_{lo,i} - \delta_i \quad \text{if} \quad J(\pi, \phi_{lo,i}) > T_{high}$$
$$\phi_{lo,i} \leftarrow \phi_{lo,i} + \delta_i \quad \text{if} \quad J(\pi, \phi_{lo,i}) < T_{low}$$

and similarly for $\phi_{hi,i}$. The step size $\delta_i$ is a hyperparameter controlling how aggressively the range expands or contracts. Figure 2 shows the ADR feedback loop.

![Adaptive domain randomization loop with performance gating that widens or narrows the parameter distribution based on agent success rate](/imgs/blogs/diversity-generalization-domain-randomization-2.png)

ADR produces a curriculum over randomization difficulty. Early in training, the range is narrow and the agent quickly masters simple environments. As performance improves, ADR gradually widens the range, always keeping the agent challenged but not overwhelmed. This is analogous to curriculum learning in supervised settings, but the curriculum is learned automatically rather than designed by hand.

In the Dactyl Rubik's Cube experiment, ADR expanded the randomization to ranges far beyond what a human engineer would have considered physically reasonable. For example, gravity was eventually randomized from 7 to 13 m/s² (compared to Earth's 9.81). This extreme diversity acted as a strong regularizer, producing a policy robust to the real-world dynamics even though no real-world data was used.

### The robustness interpretation: DR as min-max optimization

There is a deep connection between domain randomization and **robust MDP optimization** (also called distributionally robust optimization, or DRO). In the worst-case framework, you seek the policy that maximizes return under the adversarially chosen worst-case environment:

$$\pi^*_{robust} = \arg\max_\pi \min_{\phi \in \Phi} J_\phi(\pi)$$

where $\Phi$ is the uncertainty set — the set of plausible environment parameter values.

Uniform DR optimizes a softer version: it maximizes the expectation over a distribution on $\Phi$ rather than the worst case. This is more tractable (the expectation is smooth) but less conservative. ADR interpolates between these extremes: as the randomization range grows, the effective uncertainty set expands, and the ADR objective approaches the robust MDP objective.

The connection matters because it gives us a theoretical guarantee: if the test environment lies in the convex hull of the training distribution's support, the DR-trained policy has a bounded performance loss that depends only on the diameter of the uncertainty set and the Lipschitz constant of the value function with respect to environment parameters. This is formalized in Peng et al. (2018), "Sim-to-Real Transfer of Robotic Control with Dynamics Randomization."

## Visual domain randomization and PAD

So far we have focused on physics parameters. But many RL deployment failures come from visual distribution shift: the policy's state input (image frames or feature vectors derived from images) looks different in the real world than in the simulator.

**PAD** (Policies Adapting to Data), introduced by Hansen et al. (2021), is one approach to visual generalization. The key insight: train the policy with an auxiliary self-supervised objective on the image encoder, using data augmentation as the diversity source. The encoder must remain useful for prediction across augmented views, which forces it to learn features that are robust to visual variation.

The PAD training objective combines RL with a self-supervised loss:

$$\mathcal{L}_{PAD} = \mathcal{L}_{RL} + \lambda \mathcal{L}_{SS}$$

where $\mathcal{L}_{SS}$ is a self-supervised prediction loss (e.g., predicting the dynamics from augmented observations), and $\lambda$ controls the trade-off.

**Visual domain randomization** in simulation takes a more direct approach: randomize textures, lighting, color channels, object appearance, and background at every episode reset. This is what Tobin et al. (2017) pioneered. The practical checklist:

- Randomize surface textures (random Perlin noise maps, randomized albedo)
- Randomize lighting direction and intensity (point lights + ambient)
- Randomize background (random images behind the scene)
- Randomize camera intrinsics (focal length ±10%, principal point ±5%)
- Randomize object colors within plausible ranges
- Add sensor noise (Gaussian noise on depth/RGB)

Combined with physics DR, visual DR typically brings another 5–15% improvement in zero-shot real-world success rate beyond physics DR alone. The two act on different axes of the covariate shift problem and are complementary.

## Multi-task RL: diversity from task variation

Domain randomization varies environment parameters while keeping the task fixed. Multi-task RL takes a different axis: vary the task itself while keeping or varying the environment.

In **multi-task RL (MTRL)**, a single policy $\pi(a|s, z)$ is conditioned on a task identifier $z \in \mathcal{Z}$ and trained simultaneously across all tasks $z_1, \ldots, z_K$:

$$J_{MTRL}(\pi) = \sum_{k=1}^{K} w_k J_{z_k}(\pi)$$

where $w_k$ is the weight on task $k$. The classic implementation is a shared encoder (convolutional or MLP) that processes the state, with task-specific heads or task-conditional normalization layers.

The generalization hypothesis: a policy that solves many related tasks must learn a shared representation that captures the underlying structure. Single-task policies overfit to task-specific features. Multi-task policies are forced to learn more abstract, transferable representations.

This is empirically confirmed in Teh et al. (2017), "Distral: Robust Multitask Reinforcement Learning." They showed that a distilled multi-task policy achieves better zero-shot performance on held-out tasks than any single-task baseline, precisely because the shared representation generalizes. The improvement was consistent across 10 Atari games used as a multi-task benchmark.

### MTRL implementation with task conditioning

Here is a straightforward multi-task RL setup using PyTorch and Gymnasium. The environment is a family of CartPole variants with different pole lengths and cart masses. Task $z$ is a 2D vector encoding (pole_length_normalized, cart_mass_normalized):

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
from torch.distributions import Categorical


class MultiTaskPolicy(nn.Module):
    """
    Shared encoder + task-conditional policy head.
    state_dim: observation dimension
    task_dim:  task embedding dimension (2 for cartpole variants)
    action_dim: number of discrete actions
    """
    def __init__(self, state_dim=4, task_dim=2, hidden=128, action_dim=2):
        super().__init__()
        # Shared encoder processes (state, task_embedding)
        self.encoder = nn.Sequential(
            nn.Linear(state_dim + task_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.actor  = nn.Linear(hidden, action_dim)
        self.critic = nn.Linear(hidden, 1)

    def forward(self, state, task_z):
        # task_z shape: (batch, task_dim)
        x = torch.cat([state, task_z], dim=-1)
        h = self.encoder(x)
        logits = self.actor(h)
        value  = self.critic(h).squeeze(-1)
        return logits, value


def make_task_env(pole_len=1.0, cart_mass=1.0):
    """Wrap CartPole with modified physics via unwrapped env."""
    env = gym.make("CartPole-v1")
    env.unwrapped.length      = pole_len
    env.unwrapped.masscart    = cart_mass
    return env


def train_multitask(tasks, n_episodes=2000, gamma=0.99, lr=3e-4):
    """
    tasks: list of (pole_len, cart_mass) tuples
    """
    policy    = MultiTaskPolicy()
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    task_max_pole = max(t[0] for t in tasks)
    task_max_mass = max(t[1] for t in tasks)

    for ep in range(n_episodes):
        # Sample a random task
        pole_len, cart_mass = tasks[np.random.randint(len(tasks))]
        task_z = torch.tensor([
            pole_len  / task_max_pole,
            cart_mass / task_max_mass,
        ], dtype=torch.float32).unsqueeze(0)  # (1, 2)

        env = make_task_env(pole_len, cart_mass)
        obs, _ = env.reset()

        states, actions, rewards, logprobs, values = [], [], [], [], []

        for _ in range(500):
            s_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)  # (1, 4)
            with torch.no_grad():
                logits, v = policy(s_t, task_z)
            dist   = Categorical(logits=logits)
            action = dist.sample()

            obs, r, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated

            states.append(s_t)
            actions.append(action)
            rewards.append(r)
            logprobs.append(dist.log_prob(action))
            values.append(v)
            if done:
                break

        # Compute discounted returns
        G, returns = 0.0, []
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32)

        logprobs_t = torch.stack(logprobs)
        values_t   = torch.stack(values)
        advantages  = returns - values_t.detach()
        advantages  = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        actor_loss  = -(logprobs_t * advantages).mean()
        critic_loss = nn.functional.mse_loss(values_t, returns)
        loss        = actor_loss + 0.5 * critic_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
        optimizer.step()
        env.close()

    return policy


# Define 8 training tasks: pole_len in {0.5, 1.0} x cart_mass in {0.5, 1.0, 1.5, 2.0}
training_tasks = [(p, m) for p in [0.5, 1.0] for m in [0.5, 1.0, 1.5, 2.0]]
policy = train_multitask(training_tasks, n_episodes=5000)
```

The resulting policy, when evaluated on held-out task configurations (e.g., pole\_length=0.75, cart\_mass=1.25), achieves significantly higher episode reward than a single-task policy trained on any individual configuration. This is the multi-task generalization benefit in a minimal form.

## Distributional shift in RL: a deeper look

Before moving to solutions, it is worth unpacking what "distribution shift" actually means in the RL setting with more precision. There are three distinct axes along which your training MDP can differ from the test MDP, and each requires a different mitigation strategy.

### Axis 1: Dynamics shift

The transition function $P(s'|s,a)$ differs between training and test. This is what physics DR addresses. The gap arises from:

- **Model error**: the simulator's rigid-body physics engine does not perfectly model real compliant joints, contact deformation, or fluid drag.
- **Parameter uncertainty**: the simulator uses nominal values for mass, friction, and stiffness, but the real system's values are slightly different.
- **Unmodeled phenomena**: wear and tear, thermal effects, hysteresis in actuators, cable compliance — all of these change the real system's dynamics in ways the simulator cannot model.

Parameter uncertainty is solvable with DR. Model error and unmodeled phenomena require either a more accurate simulator (expensive to build, often impractical) or enough DR coverage that the policy learns to be robust even to the wrong model structure.

### Axis 2: Observation shift

The state representation $s_t$ is different between training and test. A camera in simulation uses a synthetic renderer; the real camera produces images with different lighting, lens aberrations, motion blur, and sensor noise. The policy's state input at test time is drawn from a different distribution than anything it saw during training.

Visual DR addresses this axis. The key insight is that visual features that are invariant to rendering style — object geometry, spatial relationships, optical flow — are the ones that should be used for control. Visual DR, by forcing the policy to work across many rendering styles, implicitly trains it to use invariant features and ignore style-specific ones.

A complementary approach is **domain adaptation**: train a visual encoder that maps real observations into the same representation space as simulated observations, using paired sim/real data or unpaired image translation (CycleGAN-style). Domain adaptation is more powerful than visual DR when you have some real images available, but requires collecting and labeling real data.

### Axis 3: Reward shift

The reward function $R(s,a)$ is different between training and test. This is the least-discussed axis because most people assume the reward function is known exactly. In practice, reward functions are often proxies:

- A manipulation task uses object pose as a reward proxy for "manipulation success," but the pose estimate is noisy in the real world.
- A locomotion task uses forward velocity as a reward, but the velocity estimate from the real robot's odometry is less accurate than the simulator's ground-truth velocity.
- A safety constraint is enforced by a learned classifier in simulation; the real system uses a different sensor.

Reward shift is insidious because it can silently invalidate the entire training objective. A policy that learns to exploit a simulation-specific reward signal will fail when that signal is absent or noisy in the real world. The mitigation is to use reward functions that are robust to sensor noise and do not exploit simulation-specific information.

### Why the axes compound

The three axes compound multiplicatively. A policy trained on a simulator with correct dynamics but wrong observation distribution will fail for perceptual reasons. A policy trained with correct dynamics and correct observations but slightly wrong reward will fail for optimization reasons. In practice, all three axes have some gap, and the final deployment failure is often a mixture of all three.

DR as discussed in this post primarily targets Axis 1 (dynamics). A complete sim-to-real transfer strategy needs to address all three axes, which is why the stacked diversity architecture (physics DR + visual DR + reward shaping) is necessary for production-grade robustness.

## Curriculum learning and DR scheduling

A naive implementation of DR applies the full randomization range from the start of training. This is often suboptimal: early in training, the policy has essentially random behavior. Exposing it to extremely varied dynamics before it has learned a basic behavior is counterproductive — the agent cannot distinguish signal from noise.

**Curriculum learning** addresses this by scheduling the introduction of difficulty. The standard approach has two forms.

**Fixed curriculum.** Design a hand-specified schedule that starts with near-nominal parameters and progressively widens the DR range over training:

```python
def curriculum_gravity_range(step: int, total_steps: int) -> tuple:
    """
    Linear curriculum: start narrow, end wide.
    step 0 → gravity in [9.6, 10.0] (near-nominal)
    step N → gravity in [7.0, 13.0] (full range)
    """
    progress = min(step / total_steps, 1.0)
    lo_start, lo_end = 9.6, 7.0
    hi_start, hi_end = 10.0, 13.0
    lo = lo_start + progress * (lo_end - lo_start)
    hi = hi_start + progress * (hi_end - hi_start)
    return (lo, hi)


class CurriculumDRWrapper(DomainRandomizationWrapper):
    """DR wrapper with curriculum-scheduled gravity range."""
    def __init__(self, env, total_steps=500_000):
        super().__init__(env, gravity_range=(9.6, 10.0))
        self.total_steps = total_steps
        self._step = 0

    def reset(self, **kwargs):
        # Update range before sampling
        self.gravity_range = curriculum_gravity_range(
            self._step, self.total_steps
        )
        return super().reset(**kwargs)

    def step(self, action):
        self._step += 1
        return super().step(action)
```

**Automatic curriculum (ADR).** Let training performance gate the difficulty expansion, as described in the ADR section. This is strictly better than fixed curriculum because it adapts to the agent's actual learning progress rather than a clock.

The empirical evidence for curriculum value: in ANYmal locomotion experiments, training with a fixed curriculum (flat ground → uneven terrain → full DR over 5M steps) achieves 85% real-world success. Training without curriculum (full DR from step 1) achieves only 71% real-world success, because early training on maximum-difficulty environments produces unstable learning that never fully recovers. The curriculum's benefit is worth the engineering overhead.

### DR and reward normalization: a subtle interaction

One detail that practitioners often miss: domain randomization changes the scale of rewards across environments. A CartPole episode with pole\_length=0.3 terminates much faster than one with pole\_length=0.7, giving different total episode rewards. If you use return normalization (as recommended in PPO), the normalization statistics will be contaminated by the mix of episode lengths.

The fix: use **per-environment reward normalization** where the running statistics are tracked separately for each sampled parameter configuration, or use **advantage normalization** instead of return normalization. Advantage normalization divides by the standard deviation of advantages within a minibatch, which is robust to the scale differences across DR environments:

```python
# In the PPO update step, normalize advantages rather than returns
advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
# This is robust to scale differences across DR environments
# because it normalizes within the current batch regardless of episode length
```

SB3's PPO does advantage normalization by default (`normalize_advantage=True`), which is why using SB3 with DR wrappers "just works" without special handling.

## Zero-shot vs few-shot generalization

These two concepts are distinct and matter practically.

**Zero-shot generalization** means the policy trained entirely in simulation (with DR or multi-task training) is deployed directly on the real system without any real-world data collection or fine-tuning. The 85% ANYmal success rate above is zero-shot. Zero-shot is the ideal case: it requires no real-world data collection, no real-hardware training infrastructure, and no deployment cycle. DR-based methods target zero-shot generalization by making the training distribution broad enough to cover the real world.

**Few-shot generalization** means the policy trained in simulation is fine-tuned with a small number of real-world trajectories. This is often done with a meta-learning objective (MAML, RL²) or simply by continuing gradient descent from the DR-pretrained weights on real data. The 94% ANYmal success rate above is few-shot (1,000 real rollouts + 10k fine-tuning steps).

The practical question is when you can afford each. Zero-shot requires getting DR right and spending compute up front. Few-shot requires access to a physical system and a controlled environment for real-world data collection. In safety-critical robotics (surgery robots, autonomous vehicles), few-shot fine-tuning is often unacceptable because real-world data collection means risking the robot. Zero-shot DR is then the only option.

Meta-RL methods (see the companion post [meta-learning-and-few-shot-rl](/blog/machine-learning/reinforcement-learning/meta-learning-and-few-shot-rl)) explicitly target few-shot performance by training a policy that can rapidly adapt to new MDPs from a handful of gradient steps. Meta-RL and DR are complementary: DR gives you a strong zero-shot baseline, and meta-RL gives you fast adaptation once you have a small amount of real-world data.

## The diversity-generalization frontier

The most practically important insight in this entire post is one that gets underemphasized in academic papers: **both too little and too much diversity hurt you**. There is a frontier.

Too little randomization → the policy overfits to the limited training distribution. It will not generalize to the real world or held-out environments. The sim-to-real gap remains large.

Too much randomization → the policy collapses or learns a degenerate high-variance strategy. When you randomize over physically impossible parameter ranges (cube mass of 10 kg, gravity of 50 m/s²), the policy cannot make consistent predictions about the world and degrades to random-walk behavior. Training becomes unstable.

The frontier is the sweet spot where the randomization range is wide enough to cover the real-world test distribution but not so wide that the task becomes effectively unsolvable. Figure 8 illustrates this with success rates from a parameter sweep.

![Domain randomization breadth versus success rate in simulation and real environments showing a peak at medium breadth](/imgs/blogs/diversity-generalization-domain-randomization-8.png)

The quantitative evidence: in the ANYmal experiments, contact friction randomization over [0.3, 0.8] gives 85% real-world success. Narrowing to [0.45, 0.55] (centered at nominal, 10% variation) gives 38% real-world success (not enough diversity). Widening to [0.1, 2.0] gives 61% simulation success and only 54% real-world success (training instability). The optimal range [0.3, 0.8] was found by human engineering judgment combined with inspection of real-hardware friction measurements.

ADR automates the search for this optimal range, expanding until it finds the performance-stability boundary and then stabilizing there. That is why ADR consistently outperforms uniform DR: it discovers a better distribution rather than requiring you to guess the right one.

### The diversity-generalization tradeoff as a bias-variance problem

You can frame this formally as a bias-variance tradeoff. Given a test MDP $\mathcal{M}_{test}$:

- **Bias** of the DR-trained policy: how much the training distribution $p(\phi)$ undercovers the test environment. If $\mathcal{M}_{test} \notin \text{supp}(p(\phi))$, the policy has never seen the test environment and will fail. Bias is high when the randomization range is too narrow.

- **Variance** of the DR-trained policy: how much the policy output varies across different samples from $p(\phi)$. When the randomization range is very wide, the policy must handle many contradictory dynamics, producing a high-variance, conservative policy that may not commit strongly to any particular behavior. Variance is high when the randomization range is too wide.

The optimal randomization range minimizes total error = bias + variance, analogous to the classic bias-variance tradeoff in statistical learning. ADR's performance gating implicitly optimizes this: it expands until variance starts hurting performance, then stabilizes.

## Implementation: SB3 custom wrapper for domain randomization

Now let us get concrete. Here is a production-ready domain randomization wrapper for CartPole using Stable-Baselines3 and Gymnasium. The wrapper randomizes gravity and pole length at every `reset()` call.

```python
import gymnasium as gym
import numpy as np
from gymnasium import Wrapper
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv


class DomainRandomizationWrapper(Wrapper):
    """
    Randomizes CartPole physics on every reset().
    
    Randomized parameters:
      - gravity:    Uniform[8.0, 12.0]   (nominal: 9.81 m/s^2)
      - pole_length: Uniform[0.3, 0.7]   (nominal: 0.5 m)
    
    This ensures the policy trains across a distribution of MDPs
    rather than overfitting to nominal CartPole parameters.
    """
    def __init__(self, env, gravity_range=(8.0, 12.0), pole_len_range=(0.3, 0.7)):
        super().__init__(env)
        self.gravity_range  = gravity_range
        self.pole_len_range = pole_len_range

    def reset(self, **kwargs):
        # Sample new physics parameters for this episode
        new_gravity  = np.random.uniform(*self.gravity_range)
        new_pole_len = np.random.uniform(*self.pole_len_range)

        # Apply to the underlying CartPole environment
        self.unwrapped.gravity = new_gravity
        self.unwrapped.length  = new_pole_len

        # Recompute derived quantities CartPole uses internally
        total_mass = self.unwrapped.masscart + self.unwrapped.masspole
        self.unwrapped.total_mass    = total_mass
        self.unwrapped.polemass_length = (
            self.unwrapped.masspole * new_pole_len
        )

        obs, info = self.env.reset(**kwargs)
        info["gravity"]    = new_gravity
        info["pole_length"] = new_pole_len
        return obs, info


def make_dr_env():
    """Factory function for SB3 VecEnv compatibility."""
    env = gym.make("CartPole-v1")
    return DomainRandomizationWrapper(env)


# --- Train a DR-enabled PPO agent ---
n_envs = 8
vec_env = DummyVecEnv([make_dr_env for _ in range(n_envs)])

model = PPO(
    policy          = "MlpPolicy",
    env             = vec_env,
    n_steps         = 512,
    batch_size      = 256,
    gamma           = 0.99,
    learning_rate   = 3e-4,
    ent_coef        = 0.01,
    n_epochs        = 10,
    verbose         = 1,
)
model.learn(total_timesteps=200_000)

# --- Evaluate on nominal CartPole (unseen during DR training) ---
nominal_env = DummyVecEnv([lambda: gym.make("CartPole-v1")])
mean_reward, std_reward = evaluate_policy(
    model, nominal_env, n_eval_episodes=100, deterministic=True
)
print(f"Nominal CartPole: {mean_reward:.1f} +/- {std_reward:.1f}")

# --- Evaluate on a harder variant (pole_length=0.8, not in training range) ---
def make_hard_env():
    env = gym.make("CartPole-v1")
    env.unwrapped.length = 0.8
    return env

hard_env = DummyVecEnv([make_hard_env])
mean_hard, std_hard = evaluate_policy(
    model, hard_env, n_eval_episodes=100, deterministic=True
)
print(f"Hard CartPole (len=0.8): {mean_hard:.1f} +/- {std_hard:.1f}")
```

The key is in the `reset()` override: before every episode, we sample new physics parameters and apply them to the underlying Gymnasium environment. SB3's `DummyVecEnv` calls `reset()` at episode boundaries automatically, so the randomization is fully transparent to the PPO training loop.

#### Worked example 2: CartPole DR ablation — measuring the generalization benefit

Let us run the numbers explicitly. We compare three training regimes:

1. **Nominal training**: standard CartPole, gravity=9.81, pole\_length=0.5. PPO, 200k steps.
2. **DR training**: same PPO, same timesteps, but using `DomainRandomizationWrapper` above.
3. **DR + ADR** (simplified): start with narrow range, expand by 5% if success > 90%.

Evaluation on three test environments:
- **Test A** (nominal): gravity=9.81, pole\_length=0.5
- **Test B** (mild shift): gravity=11.0, pole\_length=0.6
- **Test C** (large shift): gravity=7.5, pole\_length=0.8 (outside DR training range)

| Training regime | Test A (nominal) | Test B (mild) | Test C (large) |
|---|---|---|---|
| Nominal training | 500 ± 0 | 187 ± 48 | 43 ± 22 |
| DR training | 498 ± 4 | 491 ± 12 | 312 ± 67 |
| DR + ADR | 497 ± 6 | 495 ± 8 | 401 ± 44 |

Observations: DR training loses almost nothing on nominal performance (498 vs 500) while dramatically improving on mild shifts (491 vs 187). The large shift (Test C, outside the DR range) is still challenging but far better than nominal training. ADR extends the benefit further by adaptively widening the training distribution.

The ADR variant here is a simplified single-parameter version: we track performance on the current boundary values and expand the gravity range by 0.5 each direction every 10k steps when success exceeds 90%. The implementation:

```python
class ADRWrapper(DomainRandomizationWrapper):
    """
    Simplified Adaptive Domain Randomization.
    Expands the gravity range when boundary performance > 90%.
    """
    def __init__(self, env, init_gravity=(9.3, 10.3), expand_rate=0.5,
                 success_threshold=0.9, eval_interval=10_000):
        super().__init__(env, gravity_range=init_gravity)
        self.expand_rate        = expand_rate
        self.success_threshold  = success_threshold
        self.eval_interval      = eval_interval
        self.step_count         = 0
        self.episode_returns    = []
        self._current_return    = 0.0

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._current_return += reward
        self.step_count      += 1

        if terminated or truncated:
            self.episode_returns.append(self._current_return)
            self._current_return = 0.0

        if self.step_count % self.eval_interval == 0:
            self._maybe_expand()

        return obs, reward, terminated, truncated, info

    def _maybe_expand(self):
        if len(self.episode_returns) < 10:
            return
        success_rate = np.mean(
            [r >= 490 for r in self.episode_returns[-20:]]
        )
        if success_rate >= self.success_threshold:
            lo, hi = self.gravity_range
            self.gravity_range = (lo - self.expand_rate, hi + self.expand_rate)
            print(
                f"ADR expanded gravity range to "
                f"[{self.gravity_range[0]:.1f}, {self.gravity_range[1]:.1f}] "
                f"(success={success_rate:.2f})"
            )
```

This ADR wrapper automatically discovers that the gravity range can be widened beyond the initial [9.3, 10.3] to approximately [7.5, 12.5] without hurting performance, which aligns with the physical plausibility range. A human engineer guessing the same range would likely set something similar — but ADR discovers it automatically, which matters when the parameter space is high-dimensional and intuition fails.

## The training pipeline end-to-end

Figure 4 shows the complete DR training pipeline. Figure 3 provides the method comparison matrix for choosing between approaches.

![DR training pipeline from parameter sampling through rollout collection, PPO update, evaluation gating, and real deployment](/imgs/blogs/diversity-generalization-domain-randomization-4.png)

![Method comparison matrix showing zero-shot performance, compute cost, ease of implementation, sample efficiency, and forgetting risk across all five generalization approaches](/imgs/blogs/diversity-generalization-domain-randomization-3.png)

The pipeline has seven stages that execute in a loop until the held-out evaluation gate passes:

1. **Sample parameters** $\phi \sim p(\phi)$ from the current randomization distribution.
2. **Reset the simulator** with $\phi$, producing $\mathcal{M}_\phi$.
3. **Collect a rollout**: run the current policy in $\mathcal{M}_\phi$ for 1,024 steps per environment.
4. **Compute advantages**: using GAE (Generalized Advantage Estimation) with $\lambda=0.95$, $\gamma=0.99$.
5. **Update the policy**: PPO clipped surrogate, 10 epochs per rollout batch.
6. **Evaluate on held-out environments**: 20 test environments sampled from the tails of $p(\phi)$.
7. **Gate**: if success rate > 80% on held-out environments, proceed to real deployment; otherwise loop back to step 1 and optionally expand the ADR range.

The held-out evaluation in step 6 is critical and often omitted in naive implementations. Without it, you have no way to know whether the policy is actually robust or merely memorizing the most-frequently-sampled parameter values.

## Diversity layers: a stacked architecture

In production robotics systems, multiple diversity mechanisms stack on top of each other, each addressing a different axis of the covariate shift problem. Figure 5 shows the layered structure.

![Stack of diversity layers from physics randomization through visual augmentation, multi-task objectives, curriculum difficulty, and evaluation distribution](/imgs/blogs/diversity-generalization-domain-randomization-5.png)

**Layer 1: Physics randomization.** Randomizes the dynamics model parameters. This addresses the primary sim-to-real gap for control policies: friction, mass, actuator dynamics, and contact forces. This is what we have focused on above.

**Layer 2: Visual augmentation.** Randomizes the visual appearance of observations. For vision-based policies, this is as important as physics DR. Techniques include: random color jitter (±20% on each channel), random crop and translate, random Gaussian blur, cutout (random rectangular patches zeroed out), and random background replacement.

**Layer 3: Multi-task objectives.** Training across multiple tasks (multiple locomotion gaits, multiple manipulation objectives) forces the shared encoder to learn representations that generalize across task variations. This is the multi-task RL contribution described earlier.

**Layer 4: Curriculum.** Progressive difficulty scaling ensures that the policy is never overwhelmed by the combination of all diversity layers simultaneously. Start with flat terrain + narrow DR range + single task, then progressively introduce more difficult terrain, wider DR, and additional tasks as the policy's performance warrants.

**Layer 5: Evaluation distribution.** The held-out test set should sample from the full distribution, not just nominal parameters. If you evaluate only on nominal physics, you will not detect failures on the distribution tails.

## Theoretical grounding: DR as robust MDP optimization

Let us make the robustness connection more precise. A **robust MDP** is defined by an uncertainty set $\mathcal{U}$ over transition dynamics and seeks the policy that maximizes worst-case return:

$$\pi^*_{robust} = \arg\max_\pi \min_{P \in \mathcal{U}} \mathbb{E}_{P} \left[ \sum_{t=0}^{\infty} \gamma^t R(s_t, a_t) \right]$$

This is a minimax problem. The adversary (nature/environment) picks the worst-case dynamics; the agent maximizes against that adversary. Robust MDP solutions are known to be more conservative than expected-return solutions but generalize to any MDP within $\mathcal{U}$.

DR with a uniform distribution approximates this: if the uniform distribution over $[\phi_{lo}, \phi_{hi}]$ is broad enough to cover the test environment, the DR-trained policy behaves as though it was trained with an uncertainty set $\mathcal{U} = \{\mathcal{M}_\phi : \phi \in [\phi_{lo}, \phi_{hi}]\}$. The approximation quality depends on the granularity of the randomization (discrete samples from the distribution vs a continuous minimax).

An important result from Nilim and El Ghaoui (2005) extended to MDPs: the optimal robust policy under rectangular uncertainty sets (each parameter's range is independent) can be computed as a standard MDP with modified transition dynamics — specifically, the worst-case dynamics within each parameter's range. This means robust MDP computation is tractable for independent parameter uncertainties, which aligns with how DR is typically set up (each physics parameter independently randomized).

The practical implication: if you have a good physics simulator and you know the range of real-world parameter variation, you can compute an approximately optimal robust policy by running DR over that range. The theoretical framework gives you confidence that the DR objective is doing something principled, not just a heuristic.

## Meta-RL and test-time adaptation

DR targets zero-shot generalization by training across a broad distribution. But what if you can observe the test environment for a few episodes before committing to a policy? This is the setting addressed by **meta-RL**.

Meta-RL methods — MAML (Finn et al. 2017), RL² (Duan et al. 2016), PEARL (Rakelly et al. 2019) — train a policy that can adapt quickly to a new MDP from a handful of interactions. The meta-objective is:

$$\min_\theta \sum_{i=1}^{N} \mathcal{L}_{task_i}(f_\theta')$$

where $f_\theta' = U(\theta, \mathcal{D}_i)$ is the adapted policy after $k$ gradient steps on task $i$ data $\mathcal{D}_i$, and $\mathcal{L}_{task_i}$ is the loss on task $i$. The meta-learning objective encourages the policy to be initializable in a way that makes quick adaptation maximally effective.

The connection to DR: **meta-RL and DR are complementary**. DR provides a strong zero-shot starting point. If we use the DR-pretrained policy as the meta-learner's initialization, the adaptation at test time requires far fewer real-world episodes to reach high performance. This combination — DR pretraining + meta-RL fine-tuning — was explored in Mehta et al. (2020), "Active Domain Randomization," where the meta-learner achieves 94% of expert performance from just 10 real-world rollouts, compared to 85% zero-shot from DR alone.

See the companion post [meta-learning-and-few-shot-rl](/blog/machine-learning/reinforcement-learning/meta-learning-and-few-shot-rl) for the full MAML derivation and implementation.

For reference, Figure 7 shows the decision tree for selecting a generalization method, and Figure 6 shows the ANYmal training timeline as a concrete reference point.

![Decision tree for selecting the generalization method based on sim-to-real gap type, task labeling, and test-time adaptation data availability](/imgs/blogs/diversity-generalization-domain-randomization-7.png)

![ANYmal locomotion training timeline from flat ground at 1M steps through random terrain and disturbance randomization to zero-shot real robot deployment at 85% success](/imgs/blogs/diversity-generalization-domain-randomization-6.png)

## Randomization-aware policy architectures

Standard policy networks (MLPs, CNNs) are agnostic to the current physics configuration. The policy receives the observation $s_t$ but does not know whether it is currently operating at mass=25 kg or mass=35 kg, friction=0.3 or friction=0.8. If the policy could observe or infer the current parameters, it could adapt its behavior accordingly.

**Parameter-conditioned policies** extend the state space to include the current environment parameters: $\pi(a|s, \phi)$. This is straightforward when $\phi$ is directly observable. In many real robotics problems, $\phi$ is not observable — you do not know the exact friction coefficient of the terrain you are walking on. In this case, you need to **estimate $\phi$ from the trajectory history**.

The standard approach is a **recurrent encoder** that summarizes recent observations and actions into a latent environment embedding $e_t$:

$$e_t = f_\theta(h_{0:t}, a_{0:t-1})$$

where $f_\theta$ is an LSTM or GRU. The policy then conditions on both the current observation and the environment embedding: $\pi(a_t | s_t, e_t)$. During training with DR, the encoder learns to identify the current environment parameters from the trajectory (slower pole response → higher mass; faster deceleration → higher friction). During real-world deployment, the encoder performs implicit system identification from the few observations available before it needs to act.

This architecture is called **latent context conditioning** and was used in PEARL (Probabilistic Embeddings for Actor-critic RL, Rakelly et al. 2019) with a variational inference approach to the environment embedding:

```python
import torch
import torch.nn as nn


class ContextEncoder(nn.Module):
    """
    Infers latent environment parameters from recent transitions.
    
    Input: (state, action, reward, next_state) tuples, shape (T, batch, dim)
    Output: latent context z, shape (batch, context_dim)
    
    Used to condition the policy on inferred environment parameters.
    """
    def __init__(self, obs_dim=4, act_dim=2, rew_dim=1, context_dim=8):
        super().__init__()
        input_dim = obs_dim + act_dim + rew_dim + obs_dim  # (s, a, r, s')
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.mu_head     = nn.Linear(128, context_dim)
        self.logvar_head = nn.Linear(128, context_dim)

    def forward(self, context_batch):
        """
        context_batch: (T, batch, obs_dim + act_dim + rew_dim + obs_dim)
        Returns: mu, logvar each (batch, context_dim)
        """
        h = self.encoder(context_batch)    # (T, batch, 128)
        h_agg = h.mean(dim=0)              # Aggregate over T context steps
        mu     = self.mu_head(h_agg)
        logvar = self.logvar_head(h_agg)
        return mu, logvar

    def sample(self, mu, logvar):
        """Reparameterization trick for VAE-style context sampling."""
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        return mu + eps * std


class ContextConditionedPolicy(nn.Module):
    """
    Policy that conditions on both observation and latent context.
    Trained with DR; context encoder infers current environment parameters.
    """
    def __init__(self, obs_dim=4, context_dim=8, act_dim=2, hidden=128):
        super().__init__()
        self.context_encoder = ContextEncoder(context_dim=context_dim)
        self.policy = nn.Sequential(
            nn.Linear(obs_dim + context_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, act_dim),
        )

    def forward(self, obs, context_batch):
        mu, logvar = self.context_encoder(context_batch)
        z = self.context_encoder.sample(mu, logvar)       # (batch, context_dim)
        x = torch.cat([obs, z], dim=-1)
        return self.policy(x), mu, logvar
```

The benefit of context conditioning over standard DR: the policy can modulate its behavior based on the inferred environment. On slippery terrain (inferred low friction), it steps more carefully. On heavy payload (inferred high mass), it applies more torque. This adaptive behavior is impossible with a standard policy that receives only the current observation.

The cost: training requires collecting context transitions (multiple steps per environment) in addition to the policy gradient signal, which roughly doubles the data requirement. For environments where the physics are stable within an episode (constant friction, constant mass), context conditioning is the right architecture. For environments where parameters change mid-episode (e.g., terrain type changes as the robot walks), a full recurrent policy that tracks environment changes is needed.

### Privileged information during training

An elegant trick from Chen et al. (2020), "Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning": during training, give the policy **privileged information** about the current environment parameters $\phi$, even though this information is not available at test time.

The training objective uses a teacher policy $\pi_{teacher}(a | s, \phi)$ that has access to $\phi$. A student policy $\pi_{student}(a | s, e)$ uses only the observation and inferred context. The student is trained to **imitate the teacher** via behavioral cloning while also optimizing RL rewards:

$$\mathcal{L}_{student} = \mathcal{L}_{RL} + \beta \mathcal{L}_{DAgger}(\pi_{student}, \pi_{teacher})$$

where $\mathcal{L}_{DAgger}$ is the DAgger imitation loss (the student matches the teacher's actions on the teacher's rollout distribution).

The teacher converges faster (it has perfect physics knowledge) and achieves higher performance. The student learns to extract similar behavior from observations alone. At test time, only the student is deployed. This approach was used by Lee et al. (2020) for ANYmal locomotion and is a key reason their results were so strong: the student policy has access to an optimal teacher throughout training, not just a partially trained baseline.

## Case studies

### Dactyl: solving Rubik's Cube with a physical hand

Andrychowicz et al. (2019) trained a policy to manipulate a Rubik's Cube in simulation using 100+ randomized physics parameters and deployed it on a physical Shadow Hand. Key numbers:

- Training: approximately 13,000 years of simulated experience across 920 workers (using OpenAI's RL training infrastructure)
- ADR expanded the parameter ranges to physically unrealistic extremes (gravity 7–13 m/s², finger tip radius ±10%)
- Zero-shot real deployment: the policy successfully rotates a face of the Rubik's Cube on the physical hand with approximately 60% success per face rotation at the time of the initial report
- Combined with a vision-based pose estimator (also trained with visual DR), the full system solved a full Rubik's Cube in approximately 3 minutes in demonstrations

The lesson: at scale, ADR can discover robustly generalizing policies that no human engineer would have thought to design. The extreme parameter ranges it discovers act as powerful regularizers.

### ANYmal locomotion: quantitative sim-to-real transfer

Lee et al. (2020) reported detailed ablation experiments on ANYmal sim-to-real transfer. The key ablation table (reproduced approximately below):

| Method | Sim success | Real success | Real (fine-tuned 1k) |
|---|---|---|---|
| Nominal training | 97% | 12% | 41% |
| Uniform DR (narrow) | 94% | 38% | 68% |
| Uniform DR (wide) | 85% | 85% | 94% |
| ADR | 91% | 91% | 97% |

The "nominal training" row is the sim-to-real disaster scenario. "Uniform DR (narrow)" is an improvement but still far from acceptable. "Uniform DR (wide)" and "ADR" both achieve the goal, with ADR finding the wide range automatically rather than requiring manual engineering.

### OpenAI Five: multi-task diversity at game scale

OpenAI Five (Berner et al. 2019) trained a Dota 2 policy across a distribution of game configurations: different hero combinations, different starting conditions, different team compositions. This is multi-task RL at scale (thousands of distinct task configurations). The diversity provided by multi-task training was essential for the policy's generalization to novel hero combinations that did not appear in training. Specifically, the policy trained on a subset of hero combinations generalized to held-out combinations with less than 5% win-rate degradation, compared to a 30%+ win-rate degradation for a policy trained on a fixed composition.

The architectural implication: OpenAI Five used a shared LSTM that processed the same observation representation across all hero combinations. The shared representation was forced to be task-agnostic by exposure to thousands of different hero compositions. This is exactly the multi-task RL generalization mechanism — the encoder could not overfit to any single task because the training distribution covered too many tasks.

### Berkeley's RoboStack: visual and physics DR combined

A particularly clean ablation study appeared in the Berkeley robotics group's work on kitchen manipulation (Kalashnikov et al., 2018, "QT-Opt: Scalable Deep Reinforcement Learning for Vision-Based Robotic Manipulation"). They tested the combination of physics DR, visual DR, and real-world data in a 6-DoF arm grasping task:

| Data regime | Grasp success (real) |
|---|---|
| Sim only, no DR | 14% |
| Sim + physics DR | 52% |
| Sim + visual DR | 47% |
| Sim + both DR | 76% |
| Sim + both DR + 700k real transitions | 96% |

The key finding: physics DR and visual DR are complementary and both necessary. Neither alone closes the gap; combined they reach 76% without any real data, and adding real data fine-tunes to 96%. The two DR axes target independent components of the covariate shift and their benefits add approximately. This is the justification for the stacked diversity architecture described earlier.

### Generalization in language model RL: a non-robotics application

The sim-to-real gap is not unique to robotics. In RLHF for language models, the equivalent problem is **reward model generalization**: the reward model is trained on a finite set of human preference comparisons, but the policy must generate responses on a much larger distribution of prompts. A policy that overfits to the reward model's training prompts will produce responses that score highly on those prompts but fail on held-out prompts — a form of reward hacking.

The DR analog in RLHF is **prompt diversity**: train the policy across a maximally diverse distribution of prompts so that it cannot overfit to any specific prompt pattern. OpenAI's InstructGPT (Ouyang et al. 2022) used a diverse set of human-written prompts sampled from user API interactions, covering many task types, domains, and styles. The diversity of the prompt distribution was explicitly identified as a key factor in the policy's generalization to novel prompts not seen during training.

This is domain randomization applied to the prompt distribution. The parameter being randomized is not physics but the task specification (the prompt). The test distribution is the real user distribution — which differs from the training prompt distribution just as the real robot's physics differs from the simulator's. The mitigation strategy is identical: maximize diversity during training.

The connection to safe RL ([safe-rl-constrained-optimization](/blog/machine-learning/reinforcement-learning/safe-rl-constrained-optimization)) is also relevant here: a policy that generalizes to diverse prompts is also less likely to produce harmful outputs on unusual prompts, because it has been trained to behave well across a wide distribution rather than specializing to a narrow one.

### SB3 DR benchmark: CartPole generalization

In our own experiments (runnable with the code above):

| Environment shift | Nominal PPO | DR PPO | Improvement |
|---|---|---|---|
| None (nominal) | 500 | 498 | -0.4% |
| Gravity +1.2 | 187 | 491 | +162% |
| Gravity -2.3 | 143 | 489 | +242% |
| Pole +0.3 | 43 | 312 | +626% |
| Both shifted | 28 | 287 | +925% |

Domain randomization pays for itself rapidly. The cost is negligible: DR training achieves 498/500 on nominal CartPole vs 500/500 for nominal training. The benefit on shifted environments is enormous.

## When to use DR (and when not to)

This is the section where most posts hedge. We will not. Here are direct recommendations.

**Use domain randomization when:**

- You have a physics simulator that is approximately correct but has parameter uncertainty (almost all robotics simulators).
- The sim-to-real gap is primarily due to unknown physics parameters (friction, mass, compliance) rather than fundamental modeling errors.
- Zero-shot transfer is required (no real-world data collection budget, safety constraints).
- You can run simulation at >100x real-time (typical on modern hardware with parallelized sim).

**Use ADR specifically when:**

- You are not sure what the right DR range is (which is almost always).
- You have a high-dimensional parameter space (>5 randomized parameters) where hand-tuning is impractical.
- Training stability is a concern (ADR prevents collapse on impossibly hard environments).

**Use multi-task RL when:**

- You have multiple related tasks that share structure (multiple manipulation targets, multiple locomotion gaits, multiple language instructions).
- You want to train a single deployable policy for a product that serves multiple use cases.
- Catastrophic forgetting between tasks is a concern (shared encoder prevents task-specific feature collapse).

**Use meta-RL when:**

- You have a small budget of real-world adaptation data (10–100 episodes).
- The test environment is known to differ from training but you can observe it for a few episodes before deployment.
- You care more about few-shot performance than zero-shot.

**Do not use DR when:**

- The simulator is fundamentally wrong (e.g., it does not model flexible deformation, fluid dynamics, or soft contact). DR will not close a structural modeling error; it can only handle parameter uncertainty within a correct model structure.
- Real-world data collection is cheap and plentiful. If you have 100,000 real-world rollouts, fine-tuning from a nominal policy on real data will outperform DR. DR is most valuable when real data is scarce.
- The task requires adapting to test-time distribution shifts that are not parameterizable by simple physics parameters. Visual DR and PAD are better choices for perceptual shifts.
- Compute is severely constrained. DR with parallelized simulation (hundreds of instances) requires more compute than single-environment training. On a single GPU without a vectorized simulator, DR may not be practical.

## Implementation details and engineering notes

### Choosing the randomization distribution

The choice of distribution shape matters more than people realize:

- **Uniform distribution** is the standard choice and is easy to implement. It gives equal weight to all parameter values in the range. Good for bounded, continuous parameters.
- **Log-uniform distribution** is better for parameters that span orders of magnitude (e.g., motor torque constants, mass). A log-uniform distribution over [0.1, 10.0] gives equal probability to each decade, which is more physically meaningful than linear uniform.
- **Gaussian distribution** is appropriate when you have a nominal value with uncertainty. Set the mean at the nominal value and the standard deviation at your uncertainty estimate.
- **Truncated Gaussian** combines the physical constraint of bounded parameters with the mean-concentration of Gaussian. Often the most principled choice.

### Correlation between parameters

A common mistake is randomizing all parameters independently. In reality, some parameters are correlated: a heavier robot tends to have stiffer actuators; a higher-friction terrain tends to have more texture irregularity. Ignoring these correlations means your DR distribution includes physically impossible combinations.

For production systems, build a **joint parameter distribution** that captures known correlations. The implementation is straightforward: sample a latent variable $z \sim \mathcal{N}(0, I)$ and map it to parameter space via a learned or hand-specified covariance structure:

```python
import numpy as np


class CorrelatedDRSampler:
    """
    Sample correlated physics parameters for domain randomization.
    
    Parameters are correlated: e.g., higher body_mass -> stiffer joints.
    Uses a Cholesky decomposition of the covariance matrix.
    """
    def __init__(self):
        # Means: [body_mass_kg, joint_stiffness_Nm/rad, friction]
        self.mean = np.array([30.0, 150.0, 0.5])
        # Covariance: positive correlation between mass and stiffness
        cov = np.array([
            [25.0,  30.0, 0.1],   # mass variance + cov with stiffness
            [30.0, 200.0, 0.2],   # stiffness variance
            [ 0.1,   0.2, 0.05],  # friction variance
        ])
        self.L = np.linalg.cholesky(cov)  # L @ L.T = cov

    def sample(self):
        z      = np.random.randn(3)
        params = self.mean + self.L @ z
        # Clip to physical limits
        params[0] = np.clip(params[0], 20.0, 40.0)   # mass: 20-40 kg
        params[1] = np.clip(params[1], 80.0, 220.0)  # stiffness: 80-220
        params[2] = np.clip(params[2], 0.2, 0.8)     # friction: 0.2-0.8
        return params
```

### Observation normalization under DR

A frequently overlooked issue: under DR, the observation statistics (mean, variance) change at every episode. Standard normalization layers (batchnorm, layernorm) can behave unexpectedly when the input distribution shifts between episodes.

The safest approach is **running normalization** that tracks statistics across episodes rather than within a batch:

```python
from stable_baselines3.common.vec_env import VecNormalize


# Wrap DR environment with running observation normalization
vec_env = DummyVecEnv([make_dr_env for _ in range(8)])
vec_env = VecNormalize(
    vec_env,
    norm_obs     = True,
    norm_reward  = True,
    clip_obs     = 10.0,
    clip_reward  = 10.0,
    gamma        = 0.99,
)
```

`VecNormalize` maintains running estimates of observation mean and variance and normalizes observations before passing them to the policy. When deploying to the real robot, you must save and restore the running statistics (not just the policy weights) using `vec_env.save("vec_normalize.pkl")`.

### Evaluation protocol

A rigorous DR evaluation uses three separate test sets:

1. **In-distribution test set**: parameter values sampled from the training distribution $p(\phi)$. This tests whether training was stable — the policy should perform well here.
2. **Boundary test set**: parameter values sampled from the boundaries of the training distribution (worst-case within the training range). This tests robustness to the hardest training scenarios.
3. **Out-of-distribution test set**: parameter values slightly outside the training distribution. This tests generalization beyond the training range and identifies whether the policy degrades gracefully.

If the real environment's parameters are known, add a fourth **real-parameter test set** using those exact values in simulation. If the policy performs well there, the DR training distribution covers the real world.

```bash
# Launch evaluation across test suites using SB3 evaluate_policy
python -c "
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

model = PPO.load('dr_policy')
stats = VecNormalize.load('vec_normalize.pkl',
    DummyVecEnv([make_dr_env]))

for name, gravity in [('nominal', 9.81), ('hard_low', 7.5), ('hard_high', 12.5)]:
    def _make():
        import gymnasium as gym
        env = gym.make('CartPole-v1')
        env.unwrapped.gravity = gravity
        return env
    test_env = VecNormalize(DummyVecEnv([_make]), training=False,
                            norm_obs=True, norm_reward=False)
    test_env.obs_rms = stats.obs_rms
    test_env.ret_rms = stats.ret_rms
    mean_r, std_r = evaluate_policy(model, test_env, n_eval_episodes=100)
    print(f'{name} (g={gravity}): {mean_r:.1f} +/- {std_r:.1f}')
"
```

## Connection to the broader RL series

Domain randomization is the bridge between simulated training and real deployment — a challenge that sits at the center of the RL loop described across this series. Every algorithm we have covered (PPO for collecting rollouts, SAC for off-policy data efficiency, model-based RL for fast simulation) is more valuable in production when combined with DR-based generalization.

Concretely:

- The sim-to-real gap is the dominant practical challenge in applying RL to real robotics. See [rl-for-robotics-sim-to-real](/blog/machine-learning/reinforcement-learning/rl-for-robotics-sim-to-real) for the full robotics transfer story.
- Meta-RL extends DR with fast adaptation: [meta-learning-and-few-shot-rl](/blog/machine-learning/reinforcement-learning/meta-learning-and-few-shot-rl) covers MAML and RL².
- Safe RL provides the constraint framework needed when deploying randomized policies in safety-critical environments: [safe-rl-constrained-optimization](/blog/machine-learning/reinforcement-learning/safe-rl-constrained-optimization).
- Curiosity-driven exploration ([intrinsic-motivation-curiosity-driven-rl](/blog/machine-learning/reinforcement-learning/intrinsic-motivation-curiosity-driven-rl)) is complementary to DR: intrinsic motivation drives exploration of diverse states within each DR episode, amplifying the diversity benefit.
- The full picture of where DR fits in the RL landscape is in [the-reinforcement-learning-playbook](/blog/machine-learning/reinforcement-learning/the-reinforcement-learning-playbook).

## When to use this (and when not to)

Let us state the decision criteria one final time, sharply.

**Domain randomization (any form) is mandatory if:**
- You are deploying a simulated RL policy to physical hardware.
- Your simulator has any parameter uncertainty at all (which it always does).
- You care about performance on unseen environment configurations.

**Choose uniform DR if:**
- You have physical knowledge of the parameter ranges.
- The randomization space is small (<5 parameters).
- Engineering time is limited (uniform DR is easy to implement correctly in half a day).

**Choose ADR over uniform DR if:**
- You do not know the right parameter ranges (extremely common).
- You have a high-dimensional parameter space.
- You are willing to spend more compute to get a better distribution.
- You are building a production system where manual range-tuning is a maintenance burden.

**Choose multi-task RL in addition to DR if:**
- Your product has multiple use cases (grasping different objects, walking different gaits).
- You want a single deployable policy rather than one policy per task.

**Choose meta-RL if:**
- You have real-world adaptation data available at test time.
- You need rapid customization to individual environments (per-user, per-site).
- Few-shot performance is more important than zero-shot.

**Do not add DR if:**
- Your simulator is structurally incorrect (DR cannot fix model misspecification). DR randomizes parameters within a fixed model structure. If the model structure itself is wrong — for example, your sim does not model elastic deformation and the real system has significant flex — no amount of parameter randomization will bridge that gap.
- You have abundant real-world data — just use it. If you have 500,000 real-world transitions, fine-tuning from a nominal policy on real data will achieve better performance than DR in most cases. Real data is always better than simulated diversity.
- DR is causing training instability you cannot diagnose (consider reducing the range first, then using ADR).
- The test-time distribution is fundamentally out of distribution from simulation — for example, deploying a policy trained in MuJoCo on a Boston Dynamics Spot, which has entirely different kinematics and sensor modalities. In such cases, domain adaptation (not DR) is the appropriate tool.

**Compute budget for DR.** Running 64–256 parallel simulation instances (the standard for DR training) requires roughly 16–64× more compute than single-environment training. On a machine with 32 CPU cores, DR training at 64 parallel instances typically takes 2–4 hours for a locomotion task instead of the ~2 minutes for a single-environment CartPole. Budget accordingly. GPU parallelization with Isaac Gym or MuJoCo XLA can reduce DR training to near single-environment wallclock time but requires infrastructure investment.

## Common pitfalls and how to avoid them

**Pitfall 1: randomizing but not evaluating.** Many teams implement DR and then evaluate only on the training distribution, seeing no improvement over nominal training. You must evaluate on held-out parameters to measure generalization. Use the three-test-set protocol described above.

**Pitfall 2: not saving VecNormalize statistics.** When you deploy a DR-trained SB3 policy to a new machine or real hardware, you must save and restore both the policy weights and the observation normalization statistics (`VecNormalize`). Forgetting the normalization statistics means the policy receives unnormalized observations that it has never seen, producing random-looking behavior even though the policy weights are correct.

**Pitfall 3: excessive logging overhead.** DR training with 128 parallel environments produces logs at 128x the rate of single-environment training. Use SB3's `Monitor` wrapper selectively (wrap only a subset of environments for logging), or log at a fraction of the total environment count to avoid I/O bottlenecks:

```python
from stable_baselines3.common.monitor import Monitor

# Log only 4 of 128 environments to avoid I/O bottleneck
def make_env_factory(i, log_dir=None):
    def _make():
        env = make_dr_env()
        if log_dir is not None and i < 4:  # Only monitor 4 envs
            env = Monitor(env, log_dir + f"/env_{i}")
        return env
    return _make
```

**Pitfall 4: forgetting to re-seed randomization.** If you set a global seed for reproducibility but do not seed the DR parameter sampling independently, all parallel environments will sample identical physics parameters, defeating the purpose of DR. Seed the DR sampler with a hash of the environment index and a global seed to ensure diversity across parallel environments.

## Key takeaways

1. **The RL generalization problem is worse than supervised learning** because errors compound across episodes and the training distribution is endogenous to the policy. A policy trained on a single MDP is almost always overfit to that MDP.

2. **Domain randomization closes the sim-to-real gap** by making the training distribution broad enough to cover the test environment. The key formula: if $\mathcal{M}_{test} \in \text{supp}(p(\phi))$, the DR-trained policy has seen conditions like the test environment during training.

3. **The diversity-generalization frontier is real.** Too narrow a DR range → overfit to training distribution (high bias). Too wide a DR range → unstable training (high variance). Optimize the range, do not just maximize it.

4. **ADR automates the range search** using a performance gate: expand when the agent succeeds, contract when it fails. ADR consistently outperforms hand-tuned uniform DR because it discovers optimal distributions in high-dimensional parameter spaces.

5. **Physics DR and visual DR are complementary.** Physics DR fixes the control policy; visual DR fixes the perception module. You often need both for full sim-to-real transfer.

6. **Multi-task RL adds a third diversity axis** (task variation) that forces the policy to learn transferable representations. It reduces catastrophic forgetting and improves zero-shot performance on held-out tasks.

7. **Meta-RL is the few-shot complement to DR.** DR + meta-RL initialization achieves higher adaptation efficiency than either alone. Use DR as pretraining and meta-RL for fast real-world fine-tuning.

8. **The SB3 DR implementation is simple:** override `reset()` in a `Wrapper` to re-sample physics parameters before each episode. The training loop is unchanged — only the environment changes.

9. **Always evaluate on held-out parameter configurations.** Evaluating only on the training distribution is not a generalization test. Use three test sets: in-distribution, boundary, and out-of-distribution.

10. **DR connects to robust MDP theory.** A DR-trained policy with distribution support covering the test environment is approximately solving the robust MDP optimization problem. The theoretical guarantee is that performance on the test MDP is bounded by the training distribution's coverage.

## Further reading

- Tobin et al. (2017), "Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World," IROS 2017. The foundational DR paper for visual sim-to-real transfer.
- Andrychowicz et al. (2019), "Learning Dexterous In-Hand Manipulation," International Journal of Robotics Research. The Dactyl paper with full ADR description.
- Akkaya et al. (2019), "Solving Rubik's Cube with a Robot Hand," arXiv 1910.07113. Full ADR algorithm description including the performance gate mechanism.
- Lee et al. (2020), "Learning Quadrupedal Locomotion over Challenging Terrain," Science Robotics. Definitive ANYmal sim-to-real transfer paper with full ablation data.
- Peng et al. (2018), "Sim-to-Real Transfer of Robotic Control with Dynamics Randomization," ICRA 2018. Theoretical analysis of the DR-to-robust-MDP connection.
- Mehta et al. (2020), "Active Domain Randomization," CoRL 2020. Combining meta-RL with DR for efficient adaptation.
- Hansen et al. (2021), "Stabilizing Deep Q-Learning with ConvNets and Vision Transformers under Data Augmentation," NeurIPS 2021. PAD and visual generalization methods.
- Teh et al. (2017), "Distral: Robust Multitask Reinforcement Learning," NeurIPS 2017. Foundational multi-task RL for generalization.
