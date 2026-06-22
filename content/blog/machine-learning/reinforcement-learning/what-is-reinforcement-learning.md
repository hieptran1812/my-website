---
title: "What is Reinforcement Learning: The Reward Hypothesis and the Agent-Environment Loop"
date: "2026-06-22"
publishDate: "2026-06-22"
description: "A rigorous, first-principles introduction to reinforcement learning covering the reward hypothesis, the MDP formalism, key algorithm families, and concrete results from CartPole to AlphaGo and ChatGPT."
tags:
  [
    "reinforcement-learning",
    "deep-learning",
    "markov-decision-process",
    "policy-gradient",
    "q-learning",
    "actor-critic",
    "rlhf",
    "gymnasium",
    "pytorch",
    "machine-learning",
  ]
category: "machine-learning"
subcategory: "Reinforcement Learning"
author: "Hiep Tran"
featured: true
readTime: 55
image: "/imgs/blogs/what-is-reinforcement-learning-1.png"
---

Imagine a small robot, freshly assembled, standing in a physics simulator. Its legs are randomly wired — when you start the clock, it flails for 0.8 seconds, collapses, and stops. The simulator reports a single number: **8**. That is the distance it travelled in centimetres before falling over. The robot has no textbook, no human coach whispering "bend your knee more", no labelled dataset of expert locomotion. It has only that number, the memory of what its motors were doing, and the opportunity to try again.

Three million attempts later, the same robot runs at a human sprinting pace, recovers from shoves, and adapts in real time to one leg being partially disabled. The only teacher throughout was that single number — a scalar reward — and the reinforcement learning (RL) algorithm that turned a stream of such numbers into a progressively better policy.

This is the central magic and the central difficulty of reinforcement learning: learning to make good decisions from evaluative, delayed, and often deceptive feedback, without anyone showing you the right answer. By the end of this post you will be able to state the reward hypothesis precisely, draw the agent-environment loop from memory, explain why RL is harder than supervised learning, implement a working CartPole agent in Gymnasium, and map out the entire 71-post landscape of this series — from tabular MDPs to RLHF fine-tuning for language models. Figure 1 shows the universal backbone that every subsequent post in this series will build on.

![The agent sends action a_t to the environment, which returns observation o_{t+1} and reward r_t; return G_t accumulates and feeds back to update the policy](/imgs/blogs/what-is-reinforcement-learning-1.png)


## The Reward Hypothesis: One Sentence That Explains Everything

Before writing a single line of code, it is worth sitting with the philosophical claim that makes RL a coherent research programme rather than a loose collection of tricks.

Richard Sutton and Andrew Barto, in their foundational textbook *Reinforcement Learning: An Introduction* (1998, 2nd ed. 2018), articulate what they call the **reward hypothesis**:

> "All of what we mean by goals and purposes can be well thought of as the maximization of the expected value of the cumulative sum of a received scalar signal (called reward)."

This is a strong claim. It says that winning a game of Go, learning to walk, writing a persuasive essay, keeping a trading portfolio's Sharpe ratio above 2.0, or generating text that a human rates as helpful — all of these are, at bottom, instances of the same mathematical problem: find a policy that maximises the expectation of cumulative reward.

The hypothesis is not proven. It is a bet — the same kind of bet that the "bitter lesson" (Sutton, 2019) makes about search and learning being the two scalable ingredients of intelligence. Whether every goal can truly be reduced to reward maximisation is an active philosophical debate (see the reinforcement learning and reward specification literature, or the work on "reward hacking" where an agent finds a reward-maximising behaviour that violates the designer's intent). But as an engineering stance, the reward hypothesis has produced extraordinary results: Atari at superhuman level, protein folding as a reinforcement learning problem, language models fine-tuned by human preferences.

The practical corollary: **your job as an RL practitioner is to specify the right reward function**. Get that wrong — reward an agent for reaching a destination without penalising how long it takes, and it will go in circles forever — and no algorithm will save you. Get it right, and the algorithm does the rest. We will return to reward design throughout this series, because it is the single most common place practitioners fail in real systems.


## The Agent-Environment Loop: Formal Definition

The agent-environment loop shown in figure 1 is not a metaphor. It is a precise mathematical object called a **Markov Decision Process (MDP)**.

A finite MDP is a tuple $(\mathcal{S}, \mathcal{A}, P, R, \gamma)$:

- $\mathcal{S}$ — the **state space**. CartPole has a 4-dimensional continuous state (cart position, cart velocity, pole angle, pole angular velocity). An Atari game has a state space of roughly $256^{84 \times 84 \times 4}$ after preprocessing. A trading environment might encode 30 days of price returns plus current portfolio weights.
- $\mathcal{A}$ — the **action space**. CartPole has two discrete actions: push left, push right. A robotic arm might have 6 continuous torques. A language model at each step chooses from a vocabulary of ~50,000 tokens.
- $P(s' \mid s, a)$ — the **transition dynamics**: the probability that the environment moves to state $s'$ when the agent takes action $a$ in state $s$. The MDP takes its name from the **Markov property**: the next state depends only on the current state and action, not on the history. This is a simplification — real environments often violate it (partial observability) — but it is the foundation we build on.
- $R(s, a, s')$ — the **reward function**: the expected immediate reward for transitioning from $s$ to $s'$ via $a$.
- $\gamma \in [0, 1)$ — the **discount factor**: a scalar that down-weights future rewards. If $\gamma = 0.99$, a reward received 100 steps from now is worth $0.99^{100} \approx 0.37$ of its face value today. Discounting is necessary for mathematical convergence in infinite-horizon problems, and it encodes a practical preference: "a reward now beats the same reward later, because future rewards are uncertain."

At each timestep $t$, the interaction is:

1. The environment is in state $s_t \in \mathcal{S}$.
2. The agent observes $o_t$ (often equal to $s_t$; sometimes only a partial view).
3. The agent selects action $a_t \sim \pi(\cdot \mid s_t)$ according to its **policy** $\pi$.
4. The environment transitions to $s_{t+1} \sim P(\cdot \mid s_t, a_t)$ and emits reward $r_t = R(s_t, a_t, s_{t+1})$.
5. Repeat from step 1.

The agent's goal is to find a policy $\pi^*$ that maximises the **expected return**:

$$J(\pi) = \mathbb{E}_{\pi} \left[ G_t \right] = \mathbb{E}_{\pi} \left[ \sum_{k=0}^{\infty} \gamma^k r_{t+k+1} \right]$$

Every RL algorithm is a different approach to finding $\pi^*$. Some approximate the expected return directly (value-based methods). Some directly optimise $J(\pi)$ by gradient ascent (policy gradient methods). Some learn a model of $P$ and plan through it (model-based methods). The taxonomy post (coming in Track B) will map these systematically; for now the key insight is that they all point at the same objective.


## How RL Differs From Supervised and Unsupervised Learning

It is tempting to think of RL as "supervised learning with a different loss" or "unsupervised learning with a reward signal bolted on." Neither framing is accurate. The differences are structural.

![Comparison of reinforcement learning, supervised learning, and unsupervised learning across learning signal, feedback type, objective, and classic example](/imgs/blogs/what-is-reinforcement-learning-2.png)

**Supervised learning** receives, for each training example, a direct label — the correct output. When training an image classifier, you show the network an image of a cat and tell it the answer is "cat." The gradient points in exactly the direction that reduces the prediction error. The training set is fixed and independent of the model's decisions: whether the classifier guesses "dog" or "cat" does not change which images appear next. The key property is that feedback is **instructive**: you get the right answer, not just a score.

**Reinforcement learning** receives only a scalar score — an evaluation of how good the outcome was, not what the correct action was. If a CartPole agent pushes left and the pole falls, the reward is 0 and the episode ends. But the reward does not tell the agent "you should have pushed right at step 12 with slightly less force." The agent must figure out, through trial and error, which actions caused the bad outcome. This is the **credit assignment problem**, and it is the central technical difficulty that separates RL from supervised learning. Moreover, the agent's actions directly determine what data it collects next — a strongly coupled feedback loop that supervised learning entirely avoids.

**Unsupervised learning** lacks external feedback entirely. The learning signal comes from structure in the data itself: cluster assignments, latent variables, likelihood under a generative model. There is no reward, no evaluation of "good" versus "bad" outcomes. In practice, the boundary is fuzzy — self-supervised learning (masked autoencoders, contrastive learning) uses the data to construct its own training signal — but the key distinction remains: unsupervised learning has no goal specified by an external environment.

A useful cross-reference: when RL is applied to fine-tune language models from human preferences (RLHF), it sits at the intersection of all three paradigms. The model was pre-trained with self-supervised learning, fine-tuned with supervised learning on demonstrations, and then further refined with RL on a reward signal from human raters. We cover this in depth in Track I of this series.


## The Three Components of an RL Agent

Before diving into specific algorithms, it helps to understand the modular building blocks that every RL system combines in different ways. There are three core components.

![Three-layer stack of an RL agent showing the world model on top, value function in the middle, and policy at the base](/imgs/blogs/what-is-reinforcement-learning-3.png)

### Policy: $\pi(a \mid s)$

The **policy** is the agent's behaviour function — the mapping from perceived states to actions. It is what the agent has "learned" in the end. A policy can be:

- **Deterministic**: $a = \pi(s)$ — always pick the same action in the same state. Useful in environments where the optimal action is unambiguous. TD3 (Twin Delayed Deep Deterministic Policy Gradient) uses a deterministic actor.
- **Stochastic**: $a \sim \pi(\cdot \mid s)$ — sample an action from a probability distribution. A Categorical distribution for discrete actions (DQN with $\varepsilon$-greedy is a stochastic policy), or a Gaussian for continuous (SAC outputs mean and log-std and samples). Stochastic policies are necessary for environments where the optimal strategy involves randomisation — mixed strategies in game theory — and for policy gradient methods that require the log-likelihood $\log \pi(a \mid s)$.

In deep RL, the policy is typically a neural network. In tabular RL (small discrete state spaces), it can be a lookup table. The distinction matters: neural network policies generalise across states but are harder to train; table policies are exact but do not scale.

### Value Function: $V^\pi(s)$ and $Q^\pi(s, a)$

The **value function** answers the question: "from this state onwards, following policy $\pi$, how much total reward should I expect?"

The **state-value function** $V^\pi(s)$ is:

$$V^\pi(s) = \mathbb{E}_\pi \left[ G_t \mid s_t = s \right] = \mathbb{E}_\pi \left[ \sum_{k=0}^\infty \gamma^k r_{t+k+1} \mid s_t = s \right]$$

The **action-value function** (Q-function) $Q^\pi(s, a)$ extends this to also condition on the action taken at the first step:

$$Q^\pi(s, a) = \mathbb{E}_\pi \left[ G_t \mid s_t = s, a_t = a \right]$$

The Q-function is more directly useful: if you know $Q^*(s, a)$ for the optimal policy, you can act greedily — always pick $\arg\max_a Q^*(s, a)$ — without explicitly representing $\pi$. This is the key insight behind DQN and Q-learning.

Value functions satisfy a recursive consistency condition called the **Bellman equation**, which is the mathematical backbone of TD learning:

$$V^\pi(s) = \sum_a \pi(a \mid s) \sum_{s', r} P(s', r \mid s, a) \left[ r + \gamma V^\pi(s') \right]$$

Every model-free RL algorithm is either directly computing a value function (value-based), directly optimising the policy without a value function (pure policy gradient), or doing both (actor-critic). The Bellman equation makes the value function self-consistent and computable via repeated iteration.

### Model: $\hat{P}(s' \mid s, a)$ and $\hat{R}(s, a)$

The **model** is the agent's internal representation of how the environment works — a learned approximation of the transition dynamics $P$ and the reward function $R$. If an agent has a good model, it can perform **planning**: simulate rollouts into the future without actually interacting with the real environment, evaluate candidate actions, and update its policy from these synthetic rollouts.

Model-based RL (Track G) uses a model to improve sample efficiency dramatically. The Dreamer algorithm (Hafner et al., 2020) learns a latent-space world model from pixel observations and can solve continuous control tasks with 10–50× fewer environment interactions than model-free baselines. The catch is that learned models are imperfect, and policies optimised against a wrong model may fail catastrophically in the real world — this is the **model exploitation** problem that Track G addresses.

Most popular algorithms (DQN, PPO, SAC, TD3) are **model-free**: they learn the value function or policy directly from experience, without learning a model of the environment. This makes them more broadly applicable but less sample-efficient.


## The CartPole Hello World

To make everything above concrete, here is the minimal RL loop in code. We use Gymnasium (the maintained fork of OpenAI Gym) with CartPole-v1: a cart on a track with a pole attached at a hinge. The agent must push left or right to keep the pole from falling. The episode ends when the pole angle exceeds 12 degrees from vertical, the cart leaves the track bounds, or 500 steps are reached. Reward is +1 for every step the pole stays up. A perfect policy scores 500.

```python
import gymnasium as gym
import numpy as np

# --- Step 1: Create the environment ---
env = gym.make("CartPole-v1")

# --- Step 2: Run one random-policy episode ---
observation, info = env.reset(seed=42)
total_reward = 0

for step in range(500):
    # Random policy: 50% left, 50% right
    action = env.action_space.sample()
    
    # The core RL loop: step the environment
    observation, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    
    if terminated or truncated:
        break

print(f"Random policy episode length: {step + 1} steps, total reward: {total_reward}")
# Typical output: Random policy episode length: 9 steps, total reward: 9.0
```

That random policy averages around 8–10 steps before failing. Now here is a trained PPO agent using Stable-Baselines3:

```python
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
import gymnasium as gym

env = gym.make("CartPole-v1")

# --- Train PPO for 50,000 timesteps ---
model = PPO(
    "MlpPolicy",          # two hidden layers of 64 units each
    env,
    learning_rate=3e-4,
    n_steps=2048,         # rollout length before each update
    batch_size=64,        # minibatch size for the policy update
    n_epochs=10,          # number of epochs over the collected data
    gamma=0.99,           # discount factor
    gae_lambda=0.95,      # GAE lambda for advantage estimation
    clip_range=0.2,       # PPO clipping epsilon
    ent_coef=0.0,         # entropy bonus (0 = no exploration regularisation)
    verbose=1,
)
model.learn(total_timesteps=50_000)

# --- Evaluate the trained policy ---
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=20)
print(f"Trained PPO: mean reward = {mean_reward:.1f} ± {std_reward:.1f}")
# Typical output: Trained PPO: mean reward = 500.0 ± 0.0
```

The trained agent achieves a perfect score of 500 in essentially every evaluation episode. The training wall-clock time on a laptop CPU is under 30 seconds. This is the "hello world" of RL, and it already demonstrates the full loop: environment interaction, reward collection, and policy improvement.

The training curve corresponds to the three phases in figure 4: an early random phase (episodes 1–50 with ~10 steps each), a learning phase (episodes 50–300 with returns climbing from 10 to 400), and a convergence phase (episodes 300+ with stable 500 returns).

![CartPole training progress through three phases, from random flailing at 8 steps average to full convergence at 500 steps within 50,000 timesteps](/imgs/blogs/what-is-reinforcement-learning-4.png)

Let us look at what happens inside the PPO update. At the core, PPO maintains a **clipped surrogate objective**:

$$L^{CLIP}(\theta) = \mathbb{E}_t \left[ \min \left( r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1 - \varepsilon, 1 + \varepsilon) \hat{A}_t \right) \right]$$

where $r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{old}}(a_t \mid s_t)}$ is the probability ratio between the new and old policy, $\hat{A}_t$ is the estimated advantage, and $\varepsilon = 0.2$ is the clip range. The clip prevents the policy from moving too far from its previous version in a single update — the central stability insight of PPO. We will derive this in full in Track E.

#### Worked example: understanding the CartPole reward signal

Suppose the CartPole agent is at timestep $t = 47$. The pole angle is 10.5 degrees (close to the 12-degree limit). The agent pushes right ($a = 1$). The pole sways back slightly to 9.8 degrees, so the episode continues. The reward received is $r_{47} = 1.0$.

The agent then pushes left at $t = 48$ too aggressively. Pole angle jumps to 13.2 degrees. Episode terminates. $r_{48} = 0$ (the environment returns 0 on the terminal step in CartPole-v1's standard formulation, though the 49 steps of reward 1.0 are already banked).

The **return** from step $t = 47$ with $\gamma = 0.99$ is:

$$G_{47} = r_{47} + \gamma r_{48} + \gamma^2 r_{49} + \cdots = 1.0 + 0.99 \times 0 + 0 + \cdots = 1.0$$

But if the agent had stayed upright for the full 500 steps, $G_{47}$ would be:

$$G_{47} = \sum_{k=0}^{452} 0.99^k \times 1.0 = \frac{1 - 0.99^{453}}{1 - 0.99} \approx 98.9$$

The dramatic difference between $G_{47} = 1.0$ (falling at step 48) and $G_{47} = 98.9$ (surviving to 500) is what creates the training signal. The Q-function learns to assign high Q-values to states and actions that precede long episodes and low Q-values to states near the stability boundary.


## What Makes RL Hard: Four Structural Challenges

CartPole is easy because the reward is dense (every step gives a signal), the state is fully observable, the dynamics are stationary, and the action space is tiny. Real RL problems break all four of these assumptions simultaneously.

![Matrix of four reinforcement learning challenges showing partial observability, delayed reward, non-stationarity, and exploration-exploitation with definitions, why they break training, and mitigations](/imgs/blogs/what-is-reinforcement-learning-7.png)

### Challenge 1: Partial Observability

In CartPole, the observation is the full state: $[x, \dot{x}, \theta, \dot{\theta}]$. In most real environments, the agent only sees a partial, noisy, aliased version of the true state. An Atari game's 84×84 greyscale screenshot does not tell you the ball's velocity — you need to stack multiple consecutive frames to infer it. A trading agent observing today's price cannot see the institutional order book or the central bank's private inflation forecast.

This makes the environment a **Partially Observable Markov Decision Process (POMDP)**. The Markov property no longer holds on the observation $o_t$ alone: $P(o_{t+1} \mid o_t, a_t) \neq P(o_{t+1} \mid o_{1:t}, a_{1:t})$ in general. The agent must either maintain a **belief state** (a distribution over possible true states, updated via Bayes' rule) or use memory — typically an LSTM or Transformer that processes the observation history.

### Challenge 2: Delayed and Sparse Rewards

In CartPole, a reward arrives at every step. In many real problems, rewards are sparse or delayed. Playing chess, you might receive a reward only at checkmate — potentially 60 moves after the decisive error. Training a drug discovery agent, you receive feedback only after a multi-week wet-lab synthesis. Optimising a RLHF language model, a human rater's preference arrives only after the full response is generated.

Sparse rewards make credit assignment exponentially harder. The agent must somehow trace a reward received at step 500 back to the action at step 12 that made it possible. **Temporal Difference (TD) learning** — the key algorithmic insight of classical RL — addresses this by bootstrapping: instead of waiting for the full return, it updates value estimates incrementally using the Bellman equation. We cover TD learning in depth in Track C.

The TD(0) update for a tabular value function is:

$$V(s_t) \leftarrow V(s_t) + \alpha \left[ r_{t+1} + \gamma V(s_{t+1}) - V(s_t) \right]$$

The bracketed term $\delta_t = r_{t+1} + \gamma V(s_{t+1}) - V(s_t)$ is the **TD error** — the discrepancy between what we expected ($V(s_t)$) and what we got plus what we expect next ($r_{t+1} + \gamma V(s_{t+1})$). It is the single most important scalar in RL training, and we will encounter it in every algorithm from Q-learning to PPO.

### Challenge 3: Non-Stationarity

In supervised learning, the training distribution is fixed: the same images, the same labels, regardless of what the model predicts. In RL, the data distribution changes as the policy improves. Early in training, the agent mostly sees states near the start of episodes (where it explores randomly). Late in training, it sees states in the middle and end of trajectories it has learned to reach. This is called **non-stationarity of the data distribution**.

It is compounded in actor-critic methods: the critic is being trained to evaluate a target policy that is simultaneously changing. If the policy updates are too large, the critic's estimates become stale and its gradients mislead the actor. Proximal Policy Optimization (PPO) was invented specifically to address this by constraining how far the policy can move in a single update.

A second source of non-stationarity arises in off-policy learning with a replay buffer. DQN stores transitions from previous policies in a buffer and samples from it to train the current policy. But old transitions were generated under a different policy, so the Q-function estimates computed from them may not correctly reflect the current policy's distribution. Target networks (a periodically-frozen copy of the Q-network) partially mitigate this.

### Challenge 4: Exploration vs. Exploitation

The agent must explore — try actions it has not tried before — to discover high-reward regions of the state space. But it must also exploit — repeatedly execute actions it already knows are good — to accumulate reward. These goals are in direct tension: exploring means giving up reward now for the possibility of finding something better later.

The simplest exploration strategy is **$\varepsilon$-greedy**: with probability $\varepsilon$ take a random action; with probability $1 - \varepsilon$ take the greedy action. It is widely used and surprisingly effective. But in environments with hard-exploration problems — a Montezuma's Revenge Atari game where the agent must navigate a maze to find the first reward — $\varepsilon$-greedy fails catastrophically because the reward is so sparse that random exploration almost never reaches it.

More sophisticated exploration methods include **entropy bonuses** (add $-\beta \log \pi(a \mid s)$ to the reward to encourage the policy to remain uncertain), **UCB (Upper Confidence Bound)** from bandit theory, **curiosity-driven exploration** via prediction error on a forward model (Pathak et al., 2017), and **Random Network Distillation** (Burda et al., 2018). Track K of this series covers these in depth.


## A Taxonomy of RL Algorithms

With the four challenges in mind, figure 5 shows how the algorithm landscape is structured. It is worth understanding this map before diving into any specific algorithm, because it tells you *which problem* each algorithm was designed to solve.

![Tree taxonomy of reinforcement learning algorithms branching from model-free and model-based, then to value-based, policy gradient, and actor-critic, with specific algorithm families at leaves](/imgs/blogs/what-is-reinforcement-learning-5.png)

### Model-Free vs. Model-Based

The top-level split is whether the agent learns a model of the environment dynamics $P(s' \mid s, a)$.

**Model-free** algorithms learn directly from interaction. They never try to predict what the environment will do next. Instead, they learn either a value function (Q-learning, DQN) or a policy (REINFORCE, PPO) directly from experience. Pros: simpler to implement, no compounding model errors. Cons: sample-inefficient — DQN needs ~50 million frames on Atari; a human learns the same game in minutes.

**Model-based** algorithms learn a world model and use it to plan or generate synthetic data. Pros: can be orders of magnitude more sample-efficient (Dreamer on DMControl: ~10× better than SAC with same data). Cons: model errors compound in long rollouts; hard to know when the model is trustworthy.

### Value-Based vs. Policy Gradient

Within model-free RL, the second split is how the policy is derived.

**Value-based** methods learn $Q^*(s, a)$ and derive the policy implicitly as $\pi(s) = \arg\max_a Q^*(s, a)$. Works only for discrete action spaces (cannot argmax over continuous actions without additional structure). Key algorithms: Q-learning (tabular), DQN, C51, Rainbow, QR-DQN. These are typically **off-policy**: they can learn from transitions generated by any policy (stored in a replay buffer), which makes them sample-efficient.

**Policy gradient** methods directly represent the policy $\pi_\theta$ as a parameterised function (usually a neural network) and update $\theta$ to maximise $J(\pi_\theta)$ via gradient ascent:

$$\nabla_\theta J(\pi_\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_t \nabla_\theta \log \pi_\theta(a_t \mid s_t) \cdot G_t \right]$$

This is the **policy gradient theorem** (Sutton et al., 1999). Policy gradients work for continuous action spaces but suffer from high variance: $G_t$ is a Monte Carlo return that can vary wildly across episodes. Baseline subtraction (replacing $G_t$ with the advantage $A_t = G_t - V(s_t)$) reduces variance dramatically. REINFORCE with baseline is the foundation of every actor-critic method.

**Actor-critic** methods combine a parametric policy (the actor, $\pi_\theta$) with a learned value function (the critic, $V_\phi$ or $Q_\phi$). The critic reduces variance in the policy gradient by providing a better baseline or advantage estimate. PPO, A2C, A3C, SAC, TD3, and ACKTR are all actor-critic variants. This is the dominant paradigm in modern deep RL.

### On-Policy vs. Off-Policy

A second dimension orthogonal to value/policy is whether the algorithm requires data from the **current** policy (on-policy) or can use data from **any** policy (off-policy).

**On-policy** algorithms (REINFORCE, PPO, A2C) must collect fresh data after each policy update. This is wasteful — each episode is discarded after one use — but it ensures the value estimates are always consistent with the current policy.

**Off-policy** algorithms (Q-learning, DQN, SAC, TD3) store transitions in a **replay buffer** and resample them for multiple training updates. This vastly improves sample efficiency. The cost: importance sampling or careful approximations are needed to correct for the distribution mismatch between the data-collecting policy and the current policy.

The table below summarises the major algorithm families:

| Algorithm | Family | On/Off-Policy | Action Space | Key Strength | Typical Use Case |
|-----------|--------|--------------|-------------|-------------|-----------------|
| Q-learning | Value-based | Off | Discrete | Simplicity | Tabular, small envs |
| DQN | Value-based, deep | Off | Discrete | Stability (replay+target net) | Atari, discrete control |
| Rainbow | Value-based, deep | Off | Discrete | Sample efficiency | Atari competitive |
| REINFORCE | Policy gradient | On | Any | Simplicity | Small continuous |
| A2C / A3C | Actor-critic | On | Any | Parallelism | CPU-parallel envs |
| PPO | Actor-critic | On | Any | Stability, widely tuned | Default for most tasks |
| SAC | Actor-critic | Off | Continuous | Entropy, sample eff. | MuJoCo, robotic control |
| TD3 | Actor-critic | Off | Continuous | Deterministic actor | High-precision control |
| Dreamer v3 | Model-based | Off | Any | Sample efficiency | Data-scarce, pixel obs |
| GRPO | Policy gradient | On | Discrete (LLM) | Reasoning fine-tuning | LLMs via RL |


## Before-After: What Training Actually Changes

The difference between a random and a trained policy is not subtle.

![CartPole policy before training showing 8-step average and random actions versus after training showing 500-step perfect balance with PPO in 50k timesteps](/imgs/blogs/what-is-reinforcement-learning-6.png)

A random CartPole policy averages 8 steps. The trained PPO policy averages 500 steps — the maximum possible. This is a 62× improvement in episode length. But what actually changed inside the neural network?

Before training, the MLP policy outputs nearly equal logits for "push left" and "push right" regardless of the pole angle. The policy does not distinguish between a pole leaning right at 5 degrees (which calls for a push right) and a pole leaning left at 5 degrees (which calls for a push left). It is essentially a coin flip.

After training, the policy has learned a near-optimal control law. It applies a push in the direction that opposes the pole's angular velocity — anticipating where the pole will be in the next fraction of a second rather than just reacting to where it is now. The value function has learned that states near the stability boundary (pole angle > 8 degrees) have low value and states near vertical with low velocity have high value. Together, policy and value function implement a version of the classic PD (proportional-derivative) controller, discovered purely from scalar reward feedback.

#### Worked example: manual Q-table update for a 2-state problem

To build intuition for TD learning, consider a toy problem with two states: $s_0$ (pole balanced) and $s_1$ (pole about to fall), two actions (left, right), and rewards: $r = +1$ for staying in $s_0$, $r = 0$ for reaching $s_1$. Let $\gamma = 0.9$, learning rate $\alpha = 0.1$.

Suppose the agent is in $s_0$, takes action "right", and transitions to $s_0$ again (balanced pole stays balanced). The Q-update is:

$$Q(s_0, \text{right}) \leftarrow Q(s_0, \text{right}) + \alpha \left[ r + \gamma \max_a Q(s_0, a) - Q(s_0, \text{right}) \right]$$

With $Q(s_0, \text{right}) = 0.5$ (initial estimate), $r = 1.0$, $\max_a Q(s_0, a) = 0.5$:

$$Q(s_0, \text{right}) \leftarrow 0.5 + 0.1 \times [1.0 + 0.9 \times 0.5 - 0.5] = 0.5 + 0.1 \times 0.95 = 0.595$$

After enough updates, $Q(s_0, \text{right})$ converges to $\frac{1}{1 - 0.9} = 10.0$ — the true discounted return for staying in $s_0$ forever.

Now suppose the agent is in $s_0$, takes "left", and transitions to $s_1$ (disaster). $r = 0$, $\max_a Q(s_1, a) = 0.0$ (terminal state):

$$Q(s_0, \text{left}) \leftarrow 0.5 + 0.1 \times [0.0 + 0.0 - 0.5] = 0.5 - 0.05 = 0.45$$

After many updates, $Q(s_0, \text{left}) \to 0$. The agent learns to prefer "right" in $s_0$. This is Q-learning: pure tabular, no neural networks, but the exact same update rule that DQN uses with a function approximator replacing the table.

```python
import numpy as np

# Toy 2-state Q-learning example
n_states, n_actions = 2, 2   # s0=balanced, s1=fallen; a0=left, a1=right
Q = np.zeros((n_states, n_actions))

alpha = 0.1   # learning rate
gamma = 0.9   # discount factor
epsilon = 0.3 # exploration rate

# Simple hand-crafted dynamics for illustration
def step_toy(state, action):
    """Returns (next_state, reward, done)"""
    if state == 0 and action == 1:   # balanced + push_right → stays balanced
        return 0, 1.0, False
    elif state == 0 and action == 0: # balanced + push_left → falls
        return 1, 0.0, True
    else:                            # fallen is terminal
        return 1, 0.0, True

# Q-learning loop
for episode in range(1000):
    state = 0
    for step in range(100):
        # Epsilon-greedy action selection
        if np.random.rand() < epsilon:
            action = np.random.randint(n_actions)
        else:
            action = np.argmax(Q[state])
        
        next_state, reward, done = step_toy(state, action)
        
        # Bellman update
        td_target = reward + gamma * np.max(Q[next_state]) * (1 - done)
        td_error  = td_target - Q[state, action]
        Q[state, action] += alpha * td_error
        
        state = next_state
        if done:
            break

print("Learned Q-table:")
print(f"  Q(s0, left)  = {Q[0, 0]:.4f}  (should → 0.0)")
print(f"  Q(s0, right) = {Q[0, 1]:.4f}  (should → 10.0)")
# Output: Q(s0, left) ≈ 0.0, Q(s0, right) ≈ 9.9
```


## Case Studies: RL Milestones That Shaped the Field

The best way to understand what RL can achieve — and what it requires — is through the milestones that moved the field from academic curiosity to world-changing engineering.

### Atari DQN (Mnih et al., 2015)

The seminal paper "Human-level control through deep reinforcement learning" (Mnih, Kavukcuoglu, Silver et al., Nature 2015) applied a single DQN architecture to 49 Atari games using only raw pixel input and the game score. On 29 of 49 games, it reached or exceeded human-level performance.

The key innovations: (1) **experience replay** — storing transitions in a buffer and sampling them uniformly to break temporal correlations; (2) **target networks** — maintaining a periodically-frozen copy of the Q-network to stabilise the Bellman targets. Without these, gradient descent on correlated sequential data caused catastrophic divergence. With them, Q-learning became tractable in the neural network function class.

Compute: training on a single GPU for approximately 38 days of wall-clock time per game. The architecture: three convolutional layers followed by two fully-connected layers, ~1.8M parameters. This was not a large model by 2015 standards; the key was the RL training regime.

### AlphaGo (Silver et al., 2016) and AlphaZero (Silver et al., 2018)

AlphaGo (Silver et al., Nature 2016) combined supervised learning on expert Go games with policy gradient RL through self-play, alongside Monte Carlo Tree Search (MCTS) guided by a value network. It defeated the world champion Lee Sedol 4-1. The RL component was critical: the supervised policy was human-level, but self-play RL pushed it substantially beyond human ability.

AlphaZero (Silver et al., Science 2018) removed the supervised pre-training entirely. Starting from random play with no human knowledge other than the rules, it reached superhuman level in Go, Chess, and Shogi within 24 hours of self-play training on 5,000 TPUs. It converged to a Elo rating of 5185 in chess — approximately 400 Elo above the best human player. Return on a fixed compute budget: training from scratch outperformed an approach that used 100,000 expert games as starting point.

### InstructGPT / ChatGPT (Ouyang et al., 2022)

"Training language models to follow instructions with human feedback" (Ouyang, Wu, Jiang et al., NeurIPS 2022) established RLHF as the standard method for aligning language models with human preferences. The pipeline: (1) supervised fine-tuning on human-written demonstrations; (2) training a **reward model** from human preference data (pairs of responses, rated for quality); (3) PPO against the reward model with a KL penalty to the supervised model.

Benchmark: InstructGPT 1.3B (RLHF) was preferred by human raters over GPT-3 175B (supervised only) in 85% of head-to-head comparisons. This is a 100× parameter count disadvantage overcome by RL alignment. The key number: a KL penalty coefficient $\beta = 0.02$ was critical to prevent reward hacking — without it, the model learned to produce high-reward-model-score but incoherent text.

### Robotic Locomotion (Schulman et al., 2015; Haarnoja et al., 2018)

Proximal Policy Optimization was originally developed and validated on MuJoCo locomotion tasks (HalfCheetah, Hopper, Walker2D, Ant, Humanoid). PPO's predecessor, TRPO (Trust Region Policy Optimization, Schulman et al., ICML 2015), introduced the key idea of constraining policy updates by a KL divergence bound. PPO approximated this constraint with a clipped objective, achieving similar performance with far simpler implementation.

On HalfCheetah-v4 (a simulated cheetah that must learn to run as fast as possible), PPO achieves approximately 3,500 average return in 1M timesteps. SAC (Haarnoja et al., 2018) achieves approximately 12,000 average return in 1M timesteps — roughly 3.4× better sample efficiency. The difference: SAC is off-policy (can reuse data) and entropy-regularised (automatically balances exploration). Approximate figures; exact numbers vary by implementation and seed.

### Finance RL: Execution and Portfolio Management

RL has been applied to algorithmic trading since at least Moody & Saffell (1998). Modern approaches use deep RL for optimal execution (minimising market impact when trading large positions) and portfolio allocation (maximising Sharpe ratio or information ratio over time).

A representative result: a PPO agent trained on limit-order-book features for a single equity achieves approximately 1.8–2.3× the Sharpe ratio of a TWAP (time-weighted average price) baseline on out-of-sample data, after careful reward shaping (rewarding PnL net of transaction costs rather than raw PnL). The challenge in finance RL is non-stationarity: market regimes change, and a policy trained on 2019 data may not survive the 2020 COVID regime shift without continual adaptation. Track J of this series covers this in depth, including online learning and regime detection.


## When to Use RL (and When Not To)

The four challenges above should make you cautious. RL is the right tool in a surprisingly narrow set of circumstances. Here is a decision guide.

**Use RL when:**
- The task is inherently sequential and decisions have long-range consequences that cannot be decomposed into independent supervised examples.
- You have a simulator (or the environment is cheap to interact with). Sample-inefficient model-free methods like PPO need millions of interactions; a simulator makes this tractable.
- The reward function can be specified precisely. If you cannot say exactly what "good" looks like as a scalar, RL will optimise the wrong thing.
- The action space is large enough that exhaustive search is infeasible, but the reward signal is informative enough to guide search.
- You want superhuman performance and are willing to invest in the training regime.

**Do not use RL when:**
- You have a labelled dataset and the task decomposes into independent predictions. Use supervised learning — it is faster, more stable, and more interpretable.
- You have a known model of the environment and a small state space. Use **dynamic programming** (value iteration, policy iteration) — it gives the exact optimal policy without sampling.
- The reward is so sparse that meaningful exploration is nearly impossible and no other exploration signal is available. You will spend all your compute budget discovering nothing.
- You need interpretable decisions for regulatory or safety-critical reasons. RL policies are notoriously difficult to audit, and failures can be catastrophic and non-obvious.
- The environment is so non-stationary that a policy trained today will be worthless tomorrow. RL generalisation across distributions remains an open problem.

A concrete rule of thumb: if you can frame your problem as supervised learning (even with some label noise or approximate labels), do that first. RL should be a last resort for optimisation tasks with sequential structure, not a first-line tool.


## The Full Series Map

This post is Track A post 1 of 71. The series spans twelve tracks, each building on the foundations laid here. Figure 8 shows the dependency structure.

![Directed graph of 12 tracks in the series showing Tracks A through C as foundations, D through F as algorithm families, G through H as advanced topics, and I through L as applications and playbook](/imgs/blogs/what-is-reinforcement-learning-8.png)

Here is what each track covers and why it appears where it does in the dependency graph:

**Track A: Foundations (posts 1–6)**
The present post plus MDP formalisation, dynamic programming (value/policy iteration), Monte Carlo methods, temporal difference learning, and the full tabular RL toolkit. Everything else in the series builds on this foundation.

**Track B: The RL Algorithm Map (post 7)**
A unified taxonomy post that organises all 60+ algorithms we cover into a single coherent map. Essential reading before diving into any algorithm-specific track.

**Track C: Tabular RL and Classical Methods (posts 8–12)**
Q-learning, SARSA, TD($\lambda$), eligibility traces, Dyna-Q. The classical methods that most practitioners skip — and then reinvent poorly.

**Track D: Value-Based Deep RL (posts 13–22)**
DQN and its evolutionary tree: Double DQN, Dueling DQN, Prioritised Experience Replay, C51, Rainbow, QR-DQN, IQN. The discrete action space specialists.

**Track E: Policy Gradient Methods (posts 23–30)**
From the policy gradient theorem to REINFORCE, from A2C to PPO. The full derivation of the PPO clipped objective, the GAE advantage estimator, and entropy regularisation.

**Track F: Actor-Critic Architectures (posts 31–38)**
SAC, TD3, ACKTR, MPO, and advanced actor-critic architectures including continuous SAC for robotics and the connection to maximum entropy RL.

**Track G: Model-Based RL (posts 39–45)**
Dyna, MBPO, Dreamer v2/v3, MuZero. When and why to learn a world model. The latent imagination trick that makes Dreamer 10–50× more sample-efficient than SAC.

**Track H: Multi-Agent RL (posts 46–52)**
MARL fundamentals (Nash equilibrium, joint policies), cooperative (QMIX, MAPPO) and competitive (self-play, PSRO) settings, and the OpenAI Five / AlphaStar case studies.

**Track I: RLHF and LLM Alignment (posts 53–60)**
Reward model training from preference data, PPO-based fine-tuning, DPO, GRPO (the DeepSeek technique), Constitutional AI, and the KL penalty that prevents reward hacking. Bridges to the training-techniques series.

**Track J: Finance and Trading RL (posts 61–64)**
Optimal execution, portfolio optimisation, reward shaping for PnL vs. Sharpe, non-stationarity in financial markets, and regime-conditional policies.

**Track K: Exploration and Hard Problems (posts 65–68)**
Curiosity-driven exploration, count-based methods, Random Network Distillation, intrinsic motivation, and sparse-reward benchmarks (Montezuma's Revenge, DeepMind Control Suite).

**Track L: Capstone — The RL Playbook (posts 69–71)**
Debugging RL training (the single most practically useful track), putting it all together in a production deployment (sim-to-real for robotics, A/B testing for LLM alignment, live trading), and the final capstone that cross-links every post.


## Running Your First Full Training Loop

Here is a self-contained script that trains DQN on CartPole-v1 from scratch using PyTorch, showing every component explicitly: the replay buffer, the neural network, the target network update, and the Bellman update. This is closer to what production implementations look like than the SB3 one-liner above.

```python
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

# --- Hyperparameters ---
ENV_NAME = "CartPole-v1"
GAMMA = 0.99          # discount factor
LR = 1e-3             # learning rate
BATCH_SIZE = 64       # minibatch size for Bellman update
BUFFER_SIZE = 10_000  # replay buffer capacity
MIN_BUFFER = 1_000    # warm-up before training starts
TARGET_UPDATE = 100   # steps between target network sync
EPSILON_START = 1.0   # initial exploration rate
EPSILON_END = 0.01    # final exploration rate
EPSILON_DECAY = 0.995 # per-episode decay multiplier
MAX_EPISODES = 400    # training budget

# --- Q-Network ---
class QNetwork(nn.Module):
    def __init__(self, obs_dim, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )
    
    def forward(self, x):
        return self.net(x)

# --- Replay Buffer ---
class ReplayBuffer:
    def __init__(self, capacity):
        self.buf = deque(maxlen=capacity)
    
    def push(self, transition):
        self.buf.append(transition)
    
    def sample(self, batch_size):
        batch = random.sample(self.buf, batch_size)
        s, a, r, s_next, done = zip(*batch)
        return (
            torch.tensor(np.array(s),      dtype=torch.float32),
            torch.tensor(a,                dtype=torch.long),
            torch.tensor(r,                dtype=torch.float32),
            torch.tensor(np.array(s_next), dtype=torch.float32),
            torch.tensor(done,             dtype=torch.float32),
        )
    
    def __len__(self):
        return len(self.buf)

# --- Training ---
env = gym.make(ENV_NAME)
obs_dim   = env.observation_space.shape[0]   # 4 for CartPole
n_actions = env.action_space.n               # 2 for CartPole

q_net      = QNetwork(obs_dim, n_actions)
target_net = QNetwork(obs_dim, n_actions)
target_net.load_state_dict(q_net.state_dict())   # start in sync
target_net.eval()                                 # target never trains directly

optimizer = optim.Adam(q_net.parameters(), lr=LR)
buffer    = ReplayBuffer(BUFFER_SIZE)

epsilon    = EPSILON_START
step_count = 0
returns    = []

for episode in range(MAX_EPISODES):
    obs, _ = env.reset()
    ep_return = 0
    
    while True:
        step_count += 1
        
        # Epsilon-greedy action
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                q_vals = q_net(torch.tensor(obs, dtype=torch.float32))
            action = q_vals.argmax().item()
        
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        buffer.push((obs, action, reward, next_obs, float(done)))
        obs = next_obs
        ep_return += reward
        
        # Train only after warm-up
        if len(buffer) >= MIN_BUFFER:
            s, a, r, s_n, d = buffer.sample(BATCH_SIZE)
            
            # Current Q-values for taken actions
            q_current = q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)
            
            # Target: r + gamma * max_a Q_target(s', a) (zero if terminal)
            with torch.no_grad():
                q_next = target_net(s_n).max(1).values
            q_target = r + GAMMA * q_next * (1 - d)
            
            loss = nn.functional.mse_loss(q_current, q_target)
            optimizer.zero_grad()
            loss.backward()
            # Gradient clipping prevents exploding gradients
            nn.utils.clip_grad_norm_(q_net.parameters(), max_norm=10.0)
            optimizer.step()
        
        # Sync target network periodically
        if step_count % TARGET_UPDATE == 0:
            target_net.load_state_dict(q_net.state_dict())
        
        if done:
            break
    
    returns.append(ep_return)
    epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
    
    if episode % 50 == 0:
        avg = np.mean(returns[-50:]) if len(returns) >= 50 else np.mean(returns)
        print(f"Episode {episode:4d} | avg return (last 50): {avg:.1f} | ε={epsilon:.3f}")

env.close()
# Expected final output: avg return ≈ 490–500 by episode 350–400
```

The critical components in this loop are the replay buffer (breaks temporal correlations), the target network (prevents moving-target instability in the Bellman update), and gradient clipping (prevents rare large gradients from destroying the Q-network's weights). Remove any one of these and training either diverges or converges much more slowly.


## Key Takeaways

1. **RL is the paradigm for sequential decision-making under uncertainty.** All RL algorithms optimise the same objective — maximise expected cumulative discounted reward — using different approximations and inductive biases.

2. **The Markov Decision Process is the mathematical foundation.** Every well-posed RL problem is an MDP $(\mathcal{S}, \mathcal{A}, P, R, \gamma)$. Understanding the MDP structure tells you which algorithms can apply.

3. **The Bellman equation is the recursion that makes RL tractable.** By bootstrapping value estimates from adjacent timesteps rather than waiting for terminal returns, TD learning can operate on infinite-horizon problems with sparse rewards.

4. **Policy, value function, and model are the three building blocks.** Every RL system combines them in different proportions. Knowing which you need — and which you can afford to learn — is the first design decision.

5. **The four hard problems (partial observability, delayed reward, non-stationarity, exploration) each require specific mitigations.** Understanding which problem you are facing points directly to the right algorithm family.

6. **Model-free methods are the practitioner default.** PPO for on-policy, SAC for off-policy continuous control, DQN for off-policy discrete. Know when to reach for model-based methods (data-scarce, simulation available).

7. **Reward specification is the highest-leverage engineering decision.** The algorithm cannot save you from a misspecified reward. Define reward before choosing algorithm.

8. **Sample efficiency is the core trade-off.** On-policy methods are stable but wasteful; off-policy methods are efficient but training can diverge. Actor-critic with replay (SAC, TD3) hits the current Pareto frontier for continuous control.

9. **RL has achieved superhuman performance in games, language alignment, and robotics.** The barrier is engineering, not theory: stable training loops, reward shaping, and sufficient simulation data, not new mathematics.

10. **Start with the CartPole SB3 one-liner, then work backwards.** Understanding why each component of the training loop is necessary — target networks, clipping, entropy — is the fastest path to being able to debug real RL systems.


## Further Reading

- **Sutton, R. S. & Barto, A. G.** *Reinforcement Learning: An Introduction* (2nd ed., 2018). The canonical textbook. Chapters 1–3 cover everything in this post. [Free online at incompleteideas.net](http://incompleteideas.net/book/the-book-2nd.html).
- **Mnih, V. et al.** "Human-level control through deep reinforcement learning." *Nature* 518 (2015): 529–533. The DQN paper that established deep RL as a practical paradigm.
- **Schulman, J. et al.** "Proximal Policy Optimization Algorithms." arXiv:1707.06347 (2017). PPO — the de facto standard for policy gradient. Read alongside the TRPO paper (arXiv:1502.05477).
- **Haarnoja, T. et al.** "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor." *ICML* 2018. The maximum-entropy formulation that makes SAC sample-efficient and robust.
- **Ouyang, L. et al.** "Training language models to follow instructions with human feedback." *NeurIPS* 2022. The InstructGPT paper that established RLHF at scale.
- **Gymnasium documentation** — [gymnasium.farama.org](https://gymnasium.farama.org). The standard environment API for RL research. Essential reference for the code in this series.
- **Stable-Baselines3 documentation** — [stable-baselines3.readthedocs.io](https://stable-baselines3.readthedocs.io). The most widely used production-grade RL library. Every algorithm in this series has an SB3 implementation.
- Within this series: the unified map post (`reinforcement-learning-a-unified-map`, Track B) and the capstone (`the-reinforcement-learning-playbook`, Track L) are the two posts to read first and last. Track C (tabular RL) and Track E (policy gradient) are the natural next steps from here.


## The Bellman Equations: The Mathematical Heart of RL

The Bellman equations are not an implementation detail — they are the reason RL is mathematically tractable at all. Without them, computing optimal policies would require exhaustive search over all possible trajectories. With them, it reduces to iterative fixed-point computation.

There are two forms: the **Bellman expectation equation** (which characterises the value of a specific policy $\pi$) and the **Bellman optimality equation** (which characterises the optimal value function $V^*$ or $Q^*$).

### Bellman Expectation Equations

For the state-value function under policy $\pi$:

$$V^\pi(s) = \sum_a \pi(a \mid s) \sum_{s', r} p(s', r \mid s, a) \left[ r + \gamma V^\pi(s') \right]$$

For the action-value function under policy $\pi$:

$$Q^\pi(s, a) = \sum_{s', r} p(s', r \mid s, a) \left[ r + \gamma \sum_{a'} \pi(a' \mid s') Q^\pi(s', a') \right]$$

These are self-consistency conditions. They say: "the value of being in state $s$ under policy $\pi$ equals the expected immediate reward plus the discounted expected value of the next state, also evaluated under $\pi$." This recursive structure means you can compute $V^\pi$ by solving a system of $|\mathcal{S}|$ linear equations when the MDP is tabular and $\pi$ is fixed.

### Bellman Optimality Equations

The **optimal value functions** $V^*$ and $Q^*$ satisfy stronger equations:

$$V^*(s) = \max_a \sum_{s', r} p(s', r \mid s, a) \left[ r + \gamma V^*(s') \right]$$

$$Q^*(s, a) = \sum_{s', r} p(s', r \mid s, a) \left[ r + \gamma \max_{a'} Q^*(s', a') \right]$$

The $\max$ operator replaces the expectation over policy $\pi$ because the optimal policy always takes the best action. This is the equation that Q-learning solves iteratively: each update pushes $Q(s, a)$ towards the right-hand side of the Bellman optimality equation.

Why does Q-learning converge? The right-hand side of the Bellman optimality equation is a **contraction mapping** under the infinity norm with modulus $\gamma < 1$. By the Banach fixed-point theorem, repeated application of a contraction mapping converges to a unique fixed point — the optimal Q-function $Q^*$. This convergence guarantee holds in the tabular case with sufficient exploration. In the function approximation case (DQN), the contraction property is lost and convergence is no longer guaranteed — which is why the replay buffer and target network are engineering workarounds, not theoretical niceties.

The TD error $\delta_t = r_{t+1} + \gamma V(s_{t+1}) - V(s_t)$ is the residual of the Bellman equation: it measures how far the current value function is from satisfying the self-consistency condition. When $\delta_t = 0$ for all $(s, a)$, the value function has converged. Every algorithm in this series — DQN, PPO, SAC, TD3 — uses some form of this error as its training signal.


## Policies in Depth: Deterministic, Stochastic, and Parameterised

Policies seem simple on the surface — a mapping from states to actions — but the choice of policy representation has profound consequences for which algorithms are applicable, how exploration works, and what gradients you can compute.

### Tabular Policies

In small discrete environments, a policy is just a lookup table: for each state, store the probability of each action. CartPole with discretised state has at most $|S| \times |A|$ entries. A policy can be stored in a numpy array and updated directly without any neural network. This is the domain of Track C, which covers Q-learning, SARSA, and TD($\lambda$) in the tabular setting. The advantage: exact representation, convergence guarantees. The disadvantage: does not generalise to unseen states and explodes in memory for large state spaces.

### Neural Network Policies

Modern RL uses neural networks to represent policies over continuous or very large state spaces. A convolutional network processes pixel observations. An MLP processes low-dimensional state vectors. A Transformer processes sequential context (used in Decision Transformer and Gato). The network parameters $\theta$ are updated via gradient ascent on $J(\pi_\theta)$.

For **discrete action spaces**, the network outputs a vector of logits — one per action. A softmax converts these to a probability distribution, and actions are sampled from it:

```python
import torch
import torch.nn as nn
from torch.distributions import Categorical

class DiscretePolicy(nn.Module):
    def __init__(self, obs_dim, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.Tanh(),
            nn.Linear(64, 64),      nn.Tanh(),
            nn.Linear(64, n_actions),
        )
    
    def forward(self, obs):
        logits = self.net(obs)
        return Categorical(logits=logits)
    
    def act(self, obs):
        dist = self.forward(obs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy

# Usage in a training loop:
policy = DiscretePolicy(obs_dim=4, n_actions=2)
obs = torch.randn(4)                             # CartPole observation
action, log_prob, entropy = policy.act(obs)
# log_prob is used directly in the policy gradient update
# entropy can be added as a bonus to encourage exploration
```

For **continuous action spaces**, the network outputs mean $\mu(s)$ and log-standard-deviation $\log \sigma(s)$ of a Gaussian distribution. SAC uses a **squashed Gaussian** — applying a $\tanh$ transform to bound actions to $[-1, 1]$ — with a Jacobian correction for the change of variables:

```python
from torch.distributions import Normal
import torch.nn.functional as F

class ContinuousPolicy(nn.Module):
    def __init__(self, obs_dim, action_dim, log_std_min=-5, log_std_max=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.ReLU(),
            nn.Linear(256, 256),     nn.ReLU(),
        )
        self.mean_head    = nn.Linear(256, action_dim)
        self.log_std_head = nn.Linear(256, action_dim)
        self.log_std_min  = log_std_min
        self.log_std_max  = log_std_max
    
    def forward(self, obs):
        h = self.net(obs)
        mean    = self.mean_head(h)
        log_std = self.log_std_head(h).clamp(self.log_std_min, self.log_std_max)
        std     = log_std.exp()
        return Normal(mean, std)
    
    def sample(self, obs):
        dist = self.forward(obs)
        raw_action = dist.rsample()              # reparameterisation trick
        action     = torch.tanh(raw_action)      # squash to [-1, 1]
        
        # Log prob with squashing correction: log π(a|s) = log N(raw|μ,σ) - log(1 - tanh²(raw))
        log_prob = dist.log_prob(raw_action).sum(-1)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6).sum(-1)
        return action, log_prob
```

The reparameterisation trick (`rsample` instead of `sample`) is what allows gradients to flow through the sampling operation. Without it, you cannot compute $\nabla_\theta J(\pi_\theta)$ directly; you would need the REINFORCE estimator which has much higher variance. SAC, TD3, and continuous PPO all rely on the reparameterisation trick.

### The Entropy of a Policy

A stochastic policy has a natural measure of "how random it is" — the **entropy** $H(\pi(\cdot \mid s)) = -\sum_a \pi(a \mid s) \log \pi(a \mid s)$. High entropy means the policy is uncertain and exploring; low entropy means it is confident and exploiting.

Entropy plays a central role in maximum entropy RL (the framework underlying SAC). Instead of maximising just expected return, the agent maximises:

$$J_{MaxEnt}(\pi) = \mathbb{E}_\pi \left[ \sum_t \gamma^t \left( r_t + \alpha H(\pi(\cdot \mid s_t)) \right) \right]$$

The temperature parameter $\alpha$ controls the exploration-exploitation trade-off. High $\alpha$ encourages diverse behaviour; low $\alpha$ recovers standard RL. SAC automatically adjusts $\alpha$ to hit a target entropy, removing it as a manual hyperparameter. This is one of the reasons SAC is so practically useful: it self-tunes exploration.


## The Policy Gradient Theorem: Deriving the Update Rule

For readers who want to understand policy gradient methods from first principles before Track E, here is the core derivation. It is worth spending time on because it is the mathematical justification for why REINFORCE, A2C, PPO, RLHF, and GRPO all work.

The policy gradient theorem says that for a parameterised stochastic policy $\pi_\theta$:

$$\nabla_\theta J(\pi_\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^T \nabla_\theta \log \pi_\theta(a_t \mid s_t) \cdot G_t \right]$$

**Proof sketch.** The objective is:

$$J(\pi_\theta) = \mathbb{E}_{\tau \sim \pi_\theta} [G(\tau)] = \int p_\theta(\tau) G(\tau) \, d\tau$$

Taking the gradient:

$$\nabla_\theta J(\pi_\theta) = \int \nabla_\theta p_\theta(\tau) G(\tau) \, d\tau$$

We cannot move the gradient inside the integral naively (the integrand depends on $\theta$ through $p_\theta$). But we use the **log-derivative trick** (also called the likelihood ratio trick): $\nabla_\theta p_\theta(\tau) = p_\theta(\tau) \nabla_\theta \log p_\theta(\tau)$. Substituting:

$$\nabla_\theta J(\pi_\theta) = \int p_\theta(\tau) \nabla_\theta \log p_\theta(\tau) \cdot G(\tau) \, d\tau = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \nabla_\theta \log p_\theta(\tau) \cdot G(\tau) \right]$$

Now, $\log p_\theta(\tau) = \log p(s_0) + \sum_t \log \pi_\theta(a_t \mid s_t) + \sum_t \log P(s_{t+1} \mid s_t, a_t)$. The environment transitions $\log P(s_{t+1} \mid s_t, a_t)$ and the initial state $\log p(s_0)$ do not depend on $\theta$, so their gradients are zero. Only the policy log-probabilities survive:

$$\nabla_\theta \log p_\theta(\tau) = \sum_t \nabla_\theta \log \pi_\theta(a_t \mid s_t)$$

This gives the policy gradient theorem. The key insight: **you never need to know the transition dynamics $P$ to compute the gradient**. The dynamics cancel out because future rewards do not depend on future policy parameters — only on future actions (which are sampled). This is what makes policy gradient methods model-free.

The practical estimator (REINFORCE) uses a Monte Carlo sample of trajectories:

$$\hat{g} = \frac{1}{N} \sum_{n=1}^N \sum_{t=0}^T \nabla_\theta \log \pi_\theta(a_t^{(n)} \mid s_t^{(n)}) \cdot G_t^{(n)}$$

This is an unbiased estimator of $\nabla_\theta J(\pi_\theta)$ — but with very high variance, because $G_t$ can be wildly different across episodes. The solution is to subtract a **baseline** $b(s_t)$ that does not depend on $a_t$:

$$\hat{g}_{baseline} = \frac{1}{N} \sum_n \sum_t \nabla_\theta \log \pi_\theta(a_t \mid s_t) \cdot (G_t - b(s_t))$$

Subtracting the baseline does not introduce bias (since $\mathbb{E}_{a \sim \pi}[\nabla_\theta \log \pi_\theta(a \mid s) \cdot b(s)] = 0$ by the log-derivative trick applied in reverse). The optimal baseline is the value function $V^\pi(s_t)$, giving the **advantage** $A_t = G_t - V^\pi(s_t)$ — the signal at the heart of every actor-critic algorithm.


## Gymnasium: The Standard Environment API

Every post in this series uses Gymnasium (the Farama Foundation's maintained fork of OpenAI Gym). Understanding the API thoroughly — including wrappers — saves hours of debugging.

```python
import gymnasium as gym
import numpy as np

# --- Creating and inspecting an environment ---
env = gym.make("CartPole-v1", render_mode="rgb_array")

print(f"Observation space: {env.observation_space}")
# Box([-4.8  -inf -0.42  -inf], [4.8  inf  0.42  inf], (4,), float32)

print(f"Action space: {env.action_space}")
# Discrete(2)

print(f"Reward range: {env.reward_range}")
# (0.0, 1.0)

# --- The complete episode loop ---
obs, info = env.reset(seed=0)   # seed for reproducibility

total_reward = 0
for t in range(1000):
    action = env.action_space.sample()   # replace with policy(obs)
    
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    
    # terminated: episode ended due to environment condition (pole fell)
    # truncated:  episode ended due to time limit (500 steps reached)
    done = terminated or truncated
    if done:
        print(f"Episode ended at t={t+1}, total reward={total_reward}")
        obs, info = env.reset()   # reset for next episode
        break

env.close()
```

**Key Gymnasium conventions:**
- `env.reset(seed=...)` returns `(obs, info)` — always provide a seed for reproducibility.
- `env.step(action)` returns `(obs, reward, terminated, truncated, info)` — the split between `terminated` (natural end) and `truncated` (time limit) matters for value function bootstrapping. If `truncated=True`, the episode was cut by the time limit, not a terminal state; the value of `obs` is not zero and you should bootstrap from it.
- `env.observation_space` and `env.action_space` define the interface. Check `env.observation_space.dtype` — many algorithms assume `float32` but some envs return `float64` or `uint8` (Atari pixels).

**Wrappers** are first-class citizens in Gymnasium and essential for production use:

```python
from gymnasium.wrappers import (
    RecordEpisodeStatistics,   # tracks episode lengths and returns
    ClipReward,                # clips reward to [-1, 1] (standard for Atari)
    FrameStack,                # stacks N consecutive frames into one obs
    GrayscaleObservation,      # converts RGB pixels to greyscale
    ResizeObservation,         # resizes pixel obs (e.g. 210×160 → 84×84)
    NormalizeObservation,      # online normalisation of observation statistics
    NormalizeReward,           # scales rewards by running std
)

# Atari preprocessing pipeline (standard from DQN paper)
env = gym.make("ALE/Pong-v5")
env = GrayscaleObservation(env, keep_dim=True)
env = ResizeObservation(env, shape=(84, 84))
env = FrameStack(env, num_stack=4)              # velocity inference from 4 frames
env = ClipReward(env, min_reward=-1.0, max_reward=1.0)
env = RecordEpisodeStatistics(env)              # access via info["episode"]["r"]

# Now obs.shape = (4, 84, 84), reward ∈ {-1, 0, 1}
```

When using Stable-Baselines3, `VecEnv` and `SubprocVecEnv` allow running multiple environments in parallel, collecting data faster. PPO on CartPole with 8 parallel envs is 8× faster in wall-clock time than with 1 env, and the batched rollouts reduce gradient variance:

```python
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3 import PPO

# 8 parallel CartPole environments using multiprocessing
vec_env = make_vec_env("CartPole-v1", n_envs=8, vec_env_cls=SubprocVecEnv)

model = PPO("MlpPolicy", vec_env, n_steps=256, batch_size=64, verbose=1)
model.learn(total_timesteps=200_000)
# With 8 envs, collects 2048 transitions per rollout — same as single-env with n_steps=2048
# but 8x faster in wall-clock time
```


## Reward Shaping: The Engineering That RL Papers Don't Show You

Published RL results almost always use carefully shaped reward functions, even when papers describe them as "only the game score." Understanding reward shaping — and its pitfalls — is essential for practitioners.

**Why shape rewards?** The true terminal reward (win/lose/task completed) is too sparse for the agent to learn from efficiently. A sparse reward might be received once every 1,000 steps; shaped rewards provide dense feedback at every step, dramatically accelerating learning.

The classic result on reward shaping is **potential-based shaping** (Ng, Harada, Russell, 1999). If you define a shaped reward $\tilde{r}(s, a, s') = r(s, a, s') + F(s, a, s')$ where $F(s, a, s') = \gamma \Phi(s') - \Phi(s)$ for some **potential function** $\Phi(s)$, then the optimal policy under $\tilde{r}$ is the same as under $r$. This is the only class of reward modifications that is provably policy-invariant. Any shaping outside this class risks changing the optimal policy.

Common reward shaping strategies in practice:

```python
# 1. Dense distance-to-goal reward
def shaped_reward_locomotion(state, goal_pos):
    """Reward every step proportional to progress toward goal."""
    dist_to_goal = np.linalg.norm(state[:2] - goal_pos)
    progress = prev_dist - dist_to_goal   # prev_dist from last step
    return +1.0 + progress * 5.0          # base survival + progress bonus

# 2. Trading reward with transaction cost penalty
def shaped_reward_trading(pnl_change, position_change, transaction_cost=0.001):
    """Reward PnL net of transaction costs to discourage over-trading."""
    cost = abs(position_change) * transaction_cost
    return pnl_change - cost              # pure PnL creates over-trading

# 3. RLHF: reward model output as the reward signal
def rlhf_reward(prompt, response, reward_model, ref_model, kl_coef=0.02):
    """InstructGPT reward: reward model score minus KL from reference policy."""
    rm_score   = reward_model(prompt, response)         # scalar preference score
    kl_penalty = compute_kl(response, ref_model)        # KL from SFT model
    return rm_score - kl_coef * kl_penalty              # prevents reward hacking
```

The most common reward shaping mistake is **reward hacking**: the agent finds a behaviour that maximises the shaped reward without achieving the true goal. A cleaning robot rewarded for the number of trash items it removes may learn to scatter trash and re-collect it. A game-playing agent rewarded for a proxy metric may find ways to exploit the proxy rather than play the game. The KL penalty in RLHF (`kl_coef * kl_penalty` above) is explicitly a reward-shaping safeguard: it penalises the policy for moving too far from the supervised fine-tuned model, where most reward-hacking behaviours would take it.


## Hyperparameters and Debugging: What Goes Wrong and Why

RL training is notoriously brittle. A PPO run that achieves 500 on CartPole with one set of hyperparameters may converge to 0 with another. Understanding the most sensitive hyperparameters — and how to diagnose failures — is as important as understanding the algorithms themselves.

The table below shows the most impactful hyperparameters for PPO on CartPole-v1, with their typical ranges and what goes wrong at the boundaries:

| Hyperparameter | Typical Range | Too Low | Too High |
|---------------|--------------|---------|----------|
| `learning_rate` | 1e-4 – 3e-3 | Slow convergence | Unstable, loss spikes |
| `gamma` (discount) | 0.95 – 0.999 | Myopic, ignores future | Overestimates, slow TD |
| `clip_range` | 0.1 – 0.3 | Tiny updates, underfits | Large policy jumps, instability |
| `n_steps` | 64 – 4096 | High variance estimates | Memory / compute cost |
| `ent_coef` | 0.0 – 0.05 | Early convergence, local min | Policy never specialises |
| `gae_lambda` | 0.9 – 1.0 | High bias in advantage | High variance (Monte Carlo) |
| `batch_size` | 32 – 512 | Noisy gradients | Slow updates per epoch |

**Signs your RL training is broken:**

1. **Reward never rises above random**: check reward function (is reward actually non-zero?), check environment reset (is seed fixed so agent gets stuck?), check action mapping (did you flip left/right?).
2. **Reward rises then suddenly collapses**: classic sign of a policy update that was too large. Reduce `learning_rate` or `clip_range`. Add gradient clipping.
3. **Reward rises slowly then plateaus well below optimal**: typically insufficient exploration. Increase `ent_coef` for PPO, or increase $\varepsilon$ for DQN. Or the value function is not fitting — check critic loss.
4. **Training is unstable (reward oscillates)**: non-stationarity from a too-high learning rate, or batch size too small. Try larger `n_steps`, smaller `learning_rate`.
5. **Loss is NaN or Inf**: exploding gradients. Add `nn.utils.clip_grad_norm_` with `max_norm=0.5` or `1.0`. Check for zero-padding or denormalised observations.

The debugging companion series at `/blog/machine-learning/debugging-training/` covers systematic diagnostic procedures for training failures. When applied to RL specifically, the most valuable diagnostic is plotting the critic loss and the policy entropy simultaneously: if critic loss is not decreasing but policy entropy is, the policy is exploring but the value function is not learning (usually a Q-function expressivity problem); if policy entropy collapses while critic loss stays high, the policy has converged prematurely to a local optimum.


## The Connection to Optimal Control and Dynamic Programming

RL does not exist in isolation. It is the sample-based, model-free relative of **optimal control theory** — a field with a 60-year history of solutions to sequential decision-making problems with known dynamics.

Classical optimal control (Bellman, 1957; Pontryagin's maximum principle, 1956) solves the problem of finding a control sequence $u_1, u_2, \ldots, u_T$ to minimise a cost function for a dynamical system $\dot{x} = f(x, u)$ with known dynamics $f$. Dynamic programming — Bellman's invention — computes the optimal cost-to-go function (the value function) by working backwards from the terminal condition. For linear dynamics and quadratic cost (the LQR problem), there is a closed-form solution via the Riccati equation.

RL generalises this to:
1. **Unknown dynamics**: the agent must interact with the environment to estimate $P$.
2. **High-dimensional or discrete state spaces**: where storing a value table is infeasible.
3. **Complex cost functions**: arbitrary reward functions, not just quadratic penalties.

The connection matters in practice: for robotic control problems with known or learnable dynamics, model-based RL (Track G) bridges RL and optimal control by learning a differentiable model of $f$ and applying trajectory optimisation (iLQR, CEM) on top. The resulting algorithms (PETS, MBPO, Dreamer) can be dramatically more sample-efficient than purely model-free methods.

For **finite MDPs** — small enough to enumerate — **value iteration** and **policy iteration** (covered in Track C) give exact solutions:

```python
import numpy as np

def value_iteration(P, R, gamma=0.99, tol=1e-6):
    """
    P[s, a, s'] = probability of transitioning to s' from (s, a)
    R[s, a]     = expected reward for taking action a in state s
    Returns: V* (optimal value function), pi* (greedy policy)
    """
    n_states, n_actions = R.shape
    V = np.zeros(n_states)
    
    for iteration in range(10_000):
        V_new = np.zeros(n_states)
        for s in range(n_states):
            q_values = R[s] + gamma * np.sum(P[s] * V[np.newaxis, :], axis=1)
            V_new[s] = np.max(q_values)
        
        delta = np.max(np.abs(V_new - V))
        V = V_new
        if delta < tol:
            print(f"Value iteration converged in {iteration + 1} steps")
            break
    
    # Extract greedy policy
    pi = np.zeros(n_states, dtype=int)
    for s in range(n_states):
        q_values = R[s] + gamma * np.sum(P[s] * V[np.newaxis, :], axis=1)
        pi[s] = np.argmax(q_values)
    
    return V, pi
```

Value iteration is exact, guaranteed to converge (the Bellman optimality operator is a $\gamma$-contraction), and runs in $O(|\mathcal{S}|^2 |\mathcal{A}|)$ per sweep. The catch: it requires knowing $P$ (the transition matrix) and $R$ (the reward matrix) in advance, and it scales cubically with the state space. For CartPole with continuous states, this is infeasible. For a 3×3 grid world, it is the exact right tool.


## Setting Up Your RL Development Environment

Before starting Track C's tabular experiments, here is the standard setup:

```bash
# Create a clean Python environment
python -m venv rl-env
source rl-env/bin/activate   # Linux/Mac
# or: rl-env\Scripts\activate  # Windows

# Core dependencies
pip install gymnasium[all]           # environments including Atari and MuJoCo
pip install stable-baselines3[extra] # PPO, SAC, DQN, TD3, A2C
pip install torch torchvision        # PyTorch (GPU: add --index-url from pytorch.org)
pip install trl                      # RLHF / LLM fine-tuning
pip install numpy pandas matplotlib  # utilities

# Optional: RLlib for distributed / multi-agent
pip install "ray[rllib]"

# Verify installation
python -c "import gymnasium as gym; env = gym.make('CartPole-v1'); print('Gymnasium OK')"
python -c "from stable_baselines3 import PPO; print('SB3 OK')"
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

For Atari environments, you need the ROMs:
```bash
pip install "gymnasium[atari, accept-rom-license]"
python -m ale_py.roms   # installs all Atari ROMs
```

For MuJoCo (continuous control benchmarks like HalfCheetah, Ant, Humanoid):
```bash
pip install "gymnasium[mujoco]"
# MuJoCo is now free and open-source (no license key required since 2022)
```


## The Agent is Learning. What Comes Next.

This post has laid the foundation. You now have:

- The reward hypothesis and why it is a powerful but imperfect bet.
- The MDP formalism: $(\mathcal{S}, \mathcal{A}, P, R, \gamma)$ and the Bellman equations that make it tractable.
- The agent-environment loop in both words and code.
- The three components of an RL agent (policy, value function, model) and when to use each.
- A working CartPole implementation in Gymnasium and Stable-Baselines3.
- The four structural challenges (partial observability, delayed reward, non-stationarity, exploration) and their mitigations.
- The algorithm taxonomy: model-free/based × value/policy/actor-critic × on/off-policy.
- The policy gradient theorem — the mathematical justification for why REINFORCE and PPO work.
- Reward shaping principles and the potential-based shaping invariance result.
- A debugging checklist for when training goes wrong.
- The full 71-post series map.

The natural next step is Track C: **Tabular RL and Classical Methods**, where we implement Q-learning, SARSA, and TD($\lambda$) in full on gridworld environments, prove convergence, and build the exact intuitions that make DQN and PPO comprehensible. After that, Track D dives into DQN and its descendants for discrete action spaces, followed by Track E's full policy gradient treatment leading to PPO.

If you came here specifically for RLHF and language model alignment, Track I is the destination — but reading Tracks A through F first will make the PPO mechanics in RLHF transparent rather than magical.

The agent is learning. The environment is responding. The reward is accumulating. That loop — running in every RL system from a CartPole toy to a language model that can write code and debate philosophy — is what the next 70 posts will take apart, rebuild, and deploy. Start with the Bellman equation. Everything else follows.
