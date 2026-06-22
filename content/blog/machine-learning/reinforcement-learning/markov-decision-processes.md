---
title: "Markov Decision Processes: The Mathematical Foundation of Reinforcement Learning"
date: "2026-06-22"
publishDate: "2026-06-22"
description: "A rigorous, example-driven tour of the MDP formalism — states, actions, transition dynamics, reward functions, discount factors, the return — so you can confidently model any sequential decision problem before choosing an algorithm."
tags:
  [
    "reinforcement-learning",
    "deep-learning",
    "markov-decision-process",
    "mdp",
    "reward-function",
    "discount-factor",
    "pomdp",
    "gridworld",
    "machine-learning",
    "decision-theory",
  ]
category: "machine-learning"
subcategory: "Reinforcement Learning"
author: "Hiep Tran"
featured: true
readTime: 55
image: "/imgs/blogs/markov-decision-processes-1.png"
---

A CartPole agent has no idea what an MDP is. It just feels the pole tip left, fires the cart right, feels the pole tip further right, hesitates, fires left — and the episode ends after eight steps when the pole hits 15 degrees and the simulator resets. Every early RL practitioner has stared at that reset counter and wondered: *where exactly does the learning go wrong?*

The answer is almost always that the problem was never precisely specified. The states were ambiguous, the reward signal was sending mixed messages, the discount was set so low the agent stopped caring about anything more than two time steps away, or the environment was partially observable but was being treated as fully observed. Before you write a single line of PyTorch, you need to nail down the five-tuple that defines the world your agent lives in. That five-tuple is called a **Markov Decision Process**, and it is the mathematical foundation every reinforcement learning algorithm builds on.

This post formalises the MDP from first principles. By the end you will be able to write down the precise MDP for any problem you encounter — GridWorld, CartPole, stock-market execution, robot navigation, Atari Pong — verify whether the Markov property actually holds, derive why the discounted return converges, understand when a full MDP is not the right model and a POMDP is needed instead, and implement a complete MDP simulator in NumPy and Gymnasium. You will also see two full worked numerical examples with explicit value computations, two comparison tables, five runnable code snippets, and pointers to the algorithms that the next posts in this series build on top of this foundation.

Figure 1 below shows the core MDP structure we will build up piece by piece: states connected by action-edges with transition probability labels and reward labels.

![MDP state-action graph showing states S1, S2, S3 connected by actions a1 and a2 with transition probabilities 0.8 and 0.6 and reward labels](/imgs/blogs/markov-decision-processes-1.png)


## 1. Why we need a formal model before picking an algorithm

Most beginners reach for a deep-RL library within the first hour of learning reinforcement learning. This is a mistake — not because deep RL is wrong, but because an algorithm without a well-specified problem is like a solver without an equation: it will happily optimise something, but not necessarily what you intended.

Consider three problems that look superficially similar but differ critically in their MDP structure:

**CartPole-v1**: a cart and pole on a frictionless track. The state is a four-dimensional vector (cart position, cart velocity, pole angle, pole angular velocity). Actions are discrete: push left or push right. The simulator is deterministic; every (state, action) pair leads to exactly one next state. The episode terminates when the pole exceeds 15 degrees or the cart leaves the track. A reward of +1 is given every time step the pole stays upright.

**Robot navigation in a warehouse**: a mobile robot must navigate from a dock to a shelf. The state is the robot pose plus a partial occupancy map visible to the robot's LIDAR. Actions are wheel velocities. Transitions are stochastic — wheel slip, sensor noise, dynamic obstacles. Rewards are negative for time elapsed and strongly positive for reaching the shelf. Episodes terminate when the shelf is reached or a time limit expires.

**Language model fine-tuning with RLHF**: the "state" is the conversation history plus the model's current partial generation. The "action" is which token to emit next. Transition dynamics are deterministic (appending a token to the context moves to the next state). Rewards come from a human-preference model trained on comparison data, and they are extremely sparse — only the completed response gets scored. The episode terminates when the model emits the end-of-sequence token.

These three problems have completely different state spaces, action spaces, transition dynamics, and reward structures. Plugging them all into a default PPO run and hoping for the best produces three different failure modes: CartPole will train fine; the robot will never find the shelf because the POMDP structure is ignored; the language model will reward-hack by producing high-scoring but incoherent responses because the KL regularisation term is missing. The discipline of writing down the MDP tuple — even informally, on a whiteboard — forces you to catch those failure modes before training starts.

The MDP formalism was developed for operations research in the 1950s by Richard Bellman, who also gave us the Bellman equation. It entered reinforcement learning through the foundational work of Sutton and Barto and is now the universal interface between any sequential decision problem and any RL algorithm. If the problem cannot be cast as an MDP (or POMDP), standard RL algorithms do not apply.


## 2. The Markov property: what it means, when it holds, and when it fails

The **Markov property** is the single assumption that makes MDPs tractable. Formally, a stochastic process $\{S_t\}$ satisfies the Markov property if:

$$P(S_{t+1} \mid S_t, A_t, S_{t-1}, A_{t-1}, \ldots, S_0, A_0) = P(S_{t+1} \mid S_t, A_t)$$

In words: the probability of the next state depends only on the current state and the current action, not on any of the history that led to the current state. The present is a *sufficient statistic* for the future. Knowing everything about how the agent got to the current state gives no additional information about where it will go next, beyond what the current state already encodes.

This sounds restrictive, but it is more a modelling choice than an empirical claim. **You can always augment the state to make the Markov property hold.** If the true dynamics depend on the last $k$ observations, stack the last $k$ observations into the state. If the dynamics depend on momentum, include velocity in the state. CartPole works precisely because the designers included both position and velocity — position alone would violate the Markov property because the same position can have different dynamics depending on whether the cart is moving left or right.

**When the Markov property holds.** Fully observed physical simulators like CartPole, MuJoCo, and most standard Gymnasium environments. Turn-based games like Chess and Go when the full board is observed. Tabular MDPs with discrete states and known transition functions. Any environment where the simulation state is exposed directly to the agent.

**When it does not hold.** Any environment where:
- The agent sees only a partial view of the world (robot with a forward-facing camera cannot see behind itself; a Pac-Man agent that only sees the immediate surroundings of the maze).
- There is hidden state that affects dynamics but is not observed (other traders' intentions in a financial market, the health state of a robot joint that lacks a sensor, the internal state of a black-box opponent in a game).
- The state representation is lossy (compressing a 128×128 Atari frame to 10 hand-crafted features inevitably discards information).
- The environment has memory effects (the friction of a surface that changes with temperature, a battery whose capacity decays with cycle count in a way not measured).

When the Markov property fails, you are technically operating in a **Partially Observable MDP (POMDP)**, which we return to in Section 9. A common practical fix is to use a recurrent policy network (LSTM or Transformer) that implicitly maintains a belief state from the raw observation history without explicitly modelling the POMDP structure.

One important consequence of the Markov property: the optimal policy for a fully observable MDP is always **stationary** (time-independent) and **deterministic**. You do not need a policy that says "on even days, go left." The current state alone is sufficient to determine the best action. This simplifies policy search enormously: instead of searching over all possible history-dependent rules, you search over mappings from states to actions.

### 2.1 Testing whether your state is Markovian

A simple empirical test: train a policy on your current state representation. Then augment the state with one step of observation history (the previous observation and action). If performance improves significantly, your original state was not Markovian enough. Keep augmenting until performance plateaus; that is approximately the minimal sufficient statistic for your problem.

A more rigorous test is to estimate the mutual information $I(S_{t+1}; S_{t-1} \mid S_t, A_t)$ from collected data. If this is significantly positive, the past carries information about the future beyond the present, violating the Markov property. Gaussian process regression can estimate this from off-policy data.


## 3. The MDP tuple: (S, A, P, R, γ)

A Markov Decision Process is formally a five-tuple $(S, A, P, R, \gamma)$. Every component matters; a weak specification in any one component degrades the learning process in a different way.

![The five MDP components stacked vertically: state space S, action space A, transition dynamics P, reward function R, and discount factor gamma with one-line descriptions](/imgs/blogs/markov-decision-processes-2.png)

### 3.1 State space S

$S$ is the set of all possible states the environment can be in. States can be:

- **Discrete and finite**: $S = \{s_1, s_2, \ldots, s_n\}$. A 4×4 GridWorld has $|S| = 16$ (or 15 non-terminal) states. Chess has approximately $10^{43}$ legal states — discrete but not enumerable in a table.
- **Continuous**: $S \subseteq \mathbb{R}^d$. CartPole has $S \subseteq \mathbb{R}^4$ (position, velocity, angle, angular velocity). MuJoCo HalfCheetah has a 17-dimensional continuous state space covering all joint angles and velocities.
- **Structured/composite**: a state might be (image tensor, velocity vector, goal coordinate) where image is a $3 \times 84 \times 84$ tensor, velocity is a 3-vector, and goal is a 2D coordinate. This requires custom observation spaces in Gymnasium using `spaces.Dict` or `spaces.Tuple`.

The practical question when formulating an MDP is: *what information do I need to include in the state so that the future is independent of the past given the present?* Include too little and you violate the Markov property. Include too much and you blow up the state space, increasing the curse of dimensionality and making function approximation harder.

**State space dimensionality and the curse of dimensionality.** For tabular methods (Q-tables, value tables), the complexity scales as $|S|$. For a 10-by-10-by-10 continuous state space discretised to 10 bins per dimension, that is $10^3 = 1000$ states — manageable. For a 100-dimensional state space at 10 bins per dimension, it is $10^{100}$ — completely intractable. This is why deep RL uses neural networks as function approximators: they generalise across states, requiring far fewer samples than the full enumeration that tabular methods need.

### 3.2 Action space A

$A$ is the set of actions available to the agent. It can be:

- **Discrete**: push left / push right (CartPole), move N/S/E/W (GridWorld), buy/sell/hold (trading), one of 18 Atari joystick positions. For discrete actions, algorithms like DQN (which computes $Q(s, a)$ for each action) are natural choices.
- **Continuous**: torques on a robot's joints (MuJoCo), a real-valued bid price in an auction, a heading angle for a drone. For continuous actions, algorithms like SAC, TD3, or DDPG are required; DQN cannot enumerate all possible values.
- **Parameterised/hybrid**: a first layer selects a discrete action type (attack/move/wait), and a second layer selects a continuous parameter (attack intensity, target position, speed). This requires hierarchical RL or parameterised action spaces.

Some environments support **action masking** — not all actions are legal from every state. In GridWorld a cell at the left wall cannot take the "move left" action. In a card game, you cannot play a card you do not hold. Masking actions that are illegal is important for efficiency: allowing illegal actions wastes training time and can destabilise learning if the agent keeps trying illegal moves.

### 3.3 Transition dynamics P

$P: S \times A \times S \to [0, 1]$ is the transition function, also written $P(s' \mid s, a)$. It specifies the probability of landing in state $s'$ when taking action $a$ in state $s$.

Key properties:
$$\sum_{s' \in S} P(s' \mid s, a) = 1 \quad \forall s \in S, \forall a \in A$$

The transition function can be:

- **Deterministic**: $P(s' \mid s, a) = 1$ for exactly one $s'$ and 0 for all others. CartPole and most classic Gymnasium environments are deterministic. Deterministic environments are easier to analyse but do not test robustness to noise.
- **Stochastic**: multiple outcomes each with nonzero probability. Slippery GridWorld: take "north" and you go north with probability 0.8, east with probability 0.1, west with probability 0.1.
- **Known** (model-based RL): the agent is given $P$ explicitly (e.g., in a board game where the rules are known). Can use planning algorithms like value iteration.
- **Unknown** (model-free RL): the agent must interact with the environment to learn. Most real-world problems have unknown dynamics; the agent learns by collecting experience.

For tabular MDPs, stochastic transitions are represented as a tensor of shape $|S| \times |A| \times |S|$ where $P[s, a, s']$ is the probability. For continuous state spaces $P$ is a conditional density, often parameterised as a Gaussian $\mathcal{N}(\mu_\theta(s,a), \Sigma_\theta(s,a))$ learned by a neural network in model-based RL.

**Tabular representation in NumPy:**

```python
import numpy as np

# Deterministic GridWorld: 9 states (3x3), 4 actions (N,S,E,W)
# P[s, a, s'] = 1.0 if action a from s leads to s', else 0.0
num_states = 9
num_actions = 4  # 0=N, 1=S, 2=E, 3=W

P = np.zeros((num_states, num_actions, num_states))

def state_id(row, col):
    return row * 3 + col

def next_state(row, col, action):
    """Deterministic GridWorld transitions with wall bouncing."""
    dr = [-1, +1, 0, 0]   # N, S, E, W row delta
    dc = [0, 0, +1, -1]   # N, S, E, W col delta
    nr = max(0, min(2, row + dr[action]))
    nc = max(0, min(2, col + dc[action]))
    return nr, nc

for r in range(3):
    for c in range(3):
        s = state_id(r, c)
        for a in range(num_actions):
            nr, nc = next_state(r, c, a)
            sp = state_id(nr, nc)
            P[s, a, sp] = 1.0

print("P.shape:", P.shape)           # (9, 4, 9)
print("Row sum (must be 1.0):", P.sum(axis=2).mean())  # 1.0
```

For stochastic transitions (slippery floor):

```python
# Stochastic version: intended direction P=0.7, each perpendicular P=0.15
slip_prob = 0.15
# side_actions maps intended action to the two perpendicular actions
side_actions = {0: [2, 3], 1: [2, 3], 2: [0, 1], 3: [0, 1]}

P_stoch = np.zeros((num_states, num_actions, num_states))

for r in range(3):
    for c in range(3):
        s = state_id(r, c)
        for a in range(num_actions):
            nr, nc = next_state(r, c, a)
            P_stoch[s, a, state_id(nr, nc)] += 1.0 - 2 * slip_prob
            for sa in side_actions[a]:
                nr2, nc2 = next_state(r, c, sa)
                P_stoch[s, a, state_id(nr2, nc2)] += slip_prob

# Renormalise (wall bounces accumulate probability at same cell)
P_stoch /= P_stoch.sum(axis=2, keepdims=True)

# Verify stochastic rows sum to 1.0
assert np.allclose(P_stoch.sum(axis=2), 1.0), "Rows must sum to 1.0"
```

### 3.4 Reward function R

$R: S \times A \times S \to \mathbb{R}$ is the reward function. When the agent takes action $a$ in state $s$ and transitions to $s'$, it receives reward $R(s, a, s')$.

In practice you often see simplified forms. All three are mathematically equivalent by folding the transition into an expectation:
$$\bar{R}(s, a) = \mathbb{E}_{s'}[R(s, a, s') \mid s, a] = \sum_{s'} P(s' \mid s, a) \cdot R(s, a, s')$$

The critical design choice is **dense vs sparse**, which we examine in detail in Section 5.

### 3.5 Discount factor γ

$\gamma \in [0, 1]$ controls how much the agent values future rewards relative to immediate ones. A reward $k$ steps in the future is worth $\gamma^k$ of its face value today. Section 6 derives why this is mathematically necessary for continuing tasks and desirable for episodic ones.


## 4. Transition dynamics in depth: stochastic and deterministic worlds

Understanding transition dynamics is the difference between knowing you have an RL problem and knowing you can solve it. The dynamics determine how quickly you can learn, whether you can plan ahead, and what class of algorithms applies.

### 4.0 Transition matrix perspective

For finite MDPs, the dynamics are completely captured by the transition matrix. Fix a policy $\pi$. The **policy-induced transition matrix** is:

$$P^\pi_{ss'} = \sum_{a} \pi(a \mid s) P(s' \mid s, a)$$

This is an $|S| \times |S|$ stochastic matrix (rows sum to 1). The agent's experience under policy $\pi$ is a Markov chain on $S$ governed by $P^\pi$. Properties of $P^\pi$ directly affect learning:

- **Ergodic chains** (every state reachable from every other state under $P^\pi$): the stationary distribution $d^\pi$ exists and the agent will visit all states given enough time. This is necessary for on-policy algorithms to converge.
- **Absorbing chains** (terminal states): episodic MDPs. The chain eventually reaches a terminal state and the episode resets.
- **Near-sparse dynamics** (many zero entries in $P$): common in large state spaces where most transitions are impossible. Sparse matrix representations and function approximation are essential.



The transition dynamics are the hardest part of MDP specification to get right, because in real systems you rarely know them exactly.

**Deterministic dynamics** are the easy case. The transition is a function $f: S \times A \to S$, meaning $P(f(s,a) \mid s, a) = 1$. CartPole implements this: given (position, velocity, angle, angular velocity) and an action (force direction), physics integration gives exactly one next state. You can test your agent on CartPole and know that any failure is attributable to the policy, not to lucky or unlucky transitions.

**Stochastic dynamics** arise whenever there is noise, uncertainty, or unobserved factors. Many real-world systems have this property — the question is whether the stochasticity is intrinsic to the environment or an artifact of insufficient state representation. Three main sources are common in practice:

1. **Environment noise**: wheel slip, sensor jitter, packet loss in networked systems, wind gusts on a drone, thermal noise in electronic sensors.
2. **Opponent randomness**: a stochastic opponent in a game, market participants whose actions are unpredictable, the random moves of simulated humans in a training environment.
3. **Abstraction noise**: if your state space is an abstraction of a richer underlying system, transitions in the abstract MDP appear stochastic even if the underlying system is deterministic. Two Tetris board configurations that look the same to the agent might have been reached via different sequences and have different future dynamics if the piece queue depends on history — this reveals a non-Markovian state representation.

**Stochastic transitions and risk**: note that maximising expected return under stochastic dynamics is not the same as maximising worst-case or CVaR (conditional value at risk). An expected-return-maximising agent might choose a trajectory with high variance — sometimes excellent, sometimes catastrophic. In safety-critical applications (medical robots, autonomous vehicles) you may need to add variance penalties or chance constraints to the reward function or the optimisation criterion.

**Model-based RL** explicitly learns $\hat{P}(s' \mid s, a)$ from interaction data and then uses the learned model for planning. The advantage is dramatic sample efficiency when the model is accurate: Dreamer (Hafner et al., 2020) learns a latent-space world model and achieves on Atari what DQN achieves at 100M frames using only 200k environment interactions — a 500× improvement. The disadvantage is model bias: planning inside a wrong model optimises the wrong objective. World models (Ha and Schmidhuber, 2018; Dreamer, Hafner et al. 2020) learn a latent-space MDP. This requires fewer real-environment interactions at the cost of model prediction error — the "compounding error" problem: small per-step errors in $\hat{P}$ compound over long rollouts and produce overconfident policies that fail in the real environment. Ensuring your learned model is calibrated (uncertainty-aware) is the key challenge.


## 5. Reward functions: dense, sparse, shaped, and hacked

The reward function is the most consequential design choice in any RL problem. Get it wrong and you get reward hacking, catastrophically slow learning, or a perfectly converged-but-wrong policy.

![GridWorld reward comparison showing dense R=-1 every step on the left versus sparse R=+10 goal-only on the right with gradient signal annotations](/imgs/blogs/markov-decision-processes-3.png)

### 5.1 Immediate vs expected reward

The reward function $R(s, a, s')$ is an immediate signal. What the agent actually optimises is the *expected cumulative discounted reward* — the return $G_t$ (Section 7). The immediate reward is the raw signal; the return is the objective.

This distinction matters for reward engineering. Giving the agent $R=+1$ at every non-terminal step and $R=0$ at terminal is equivalent (in optimal policy, not in learning dynamics) to giving $R=0$ at every non-terminal step and $R=+T$ at terminal (where $T$ is episode length), because both encode "the agent should behave as if longer episodes are better." But the learning dynamics differ dramatically. One gives a gradient signal at every step; the other gives a delayed, episodic signal. The dense version will train approximately 50× faster in practice for simple tasks like CartPole.

### 5.2 Dense rewards

A **dense reward** gives a signal at every time step. In GridWorld: $R(s, a, s') = -1$ for any transition that does not reach the goal, $R = +10$ for reaching the goal. Every step provides gradient information. The agent receives a signal even for bad actions (slightly negative) that helps it calibrate which directions are worse or better.

Advantages: sample-efficient learning because every experience provides a training signal; gradient flows through every step so backpropagation has a clear target; convergence tends to be faster and more reliable.

Disadvantages: a poorly designed dense reward can lead to unintended behaviours. A cleaning robot given $R = +1$ for each piece of dirt cleaned might learn to scatter the dirt first and then clean it (reward hacking). A robot given $R$ proportional to velocity might learn to fall over since that maximises measured speed momentarily. Dense rewards require the designer to know what they want at every time step, which is harder than specifying the terminal goal.

### 5.3 Sparse rewards

A **sparse reward** signals only at the end of the episode or at rare events (goal reached, game won, task completed). In a challenging maze: $R = +1$ only on successful exit, $R = 0$ everywhere else. This is the most natural specification (just tell the agent what you ultimately want) but the hardest for learning. A random agent exploring a large maze may never reach the goal by chance, so the agent never sees a positive reward and never learns that moving in one direction is better than another.

The sparsity problem is one of the central unsolved challenges in RL. Key techniques:
- **Reward shaping** (Ng, Harada, Russell 1999): add auxiliary dense rewards $F(s, s') = \gamma \Phi(s') - \Phi(s)$ where $\Phi$ is a potential function. This provably does not change the optimal policy (the shaping term telescopes to $\gamma \Phi(s_T) - \Phi(s_0)$ over any trajectory, which is a constant offset).
- **Hindsight experience replay (HER)** (Andrychowicz et al., 2017): for failed trajectories where the agent reached state $s_T \neq$ goal, replay the trajectory as if the agent had been aiming for $s_T$ all along, generating artificial positive rewards.
- **Intrinsic motivation**: add a bonus reward for visiting novel states (count-based exploration: $R_i(s) = \beta / \sqrt{n(s)}$ where $n(s)$ is the visit count for state $s$) or for surprising predictions (curiosity-driven exploration: $R_i = \|\hat{s}_{t+1} - s_{t+1}\|^2$).
- **Curriculum learning**: start with easy nearby goals and gradually increase difficulty as the agent learns.

### 5.4 Reward hacking and misspecification

Reward hacking occurs when the agent achieves high return by exploiting unintended features of the reward function rather than by solving the intended task. This is the RL equivalent of Goodhart's law: when a measure becomes a target, it ceases to be a good measure.

Classic examples from the RL literature:
- A boat racing agent that learned to spin in circles collecting boost pads rather than finishing the race (the reward was points collected, not race completion).
- A simulated hand that learned to move quickly rather than manipulate objects (velocity-based reward instead of object-placement reward).
- InstructGPT language models that learned to produce very long responses with many citations because human raters gave higher scores to longer, denser answers (reward model overfitting).

The fix is always the same: audit your reward function by watching what behaviour it incentivises, not what you intended. Run a random policy for a few episodes and ask: what would a reward-maximising agent do if it were smarter than random? If the answer is different from the intended task, your reward function is misspecified.

### 5.5 Reward function comparison

| Reward type | Sample efficiency | Risk of misspecification | Best for |
|-------------|------------------|--------------------------|----------|
| Dense step penalty ($R=-1$/step) | High | Medium (incentivises fast completion) | Navigation, manipulation |
| Sparse terminal reward | Low | Low (unambiguous goal) | Games, well-defined tasks |
| Potential-based shaping | High | Low (preserves optimal policy) | Any task with a good potential estimate |
| Learned reward (RLHF) | Medium | High (reward model errors compound) | NLP alignment, preference learning |
| Multi-objective (weighted sum) | Medium | High (weights are hyperparameters) | Problems with trade-offs |


## 6. Discount factor γ: three justifications and what it controls

The discount factor $\gamma \in [0, 1]$ has three independent justifications: financial, probabilistic, and mathematical.

**Financial intuition**: a reward today is worth more than the same reward tomorrow, because today's reward can be invested. At interest rate $r$, a dollar today grows to $1 + r$ dollars next period. So next period's reward is worth $1/(1+r)$ of this period's, giving $\gamma = 1/(1+r) < 1$.

**Probabilistic interpretation**: at every time step there is a probability $1 - \gamma$ that the episode terminates (a geometric "death process"). Future rewards are discounted because there is a nonzero chance the episode never reaches them. This interpretation turns any continuing task into an equivalent episodic one with geometrically distributed lifetime, mean $1/(1-\gamma)$.

**Mathematical necessity**: without discounting ($\gamma = 1$), the return of a continuing task diverges if rewards are nonzero. For any constant positive reward $r$ per step, $\sum_{k=0}^{\infty} r = \infty$. With $\gamma < 1$:

$$\sum_{k=0}^{\infty} \gamma^k = \frac{1}{1 - \gamma} < \infty$$

So if $|R_t| \leq R_{\max}$ for all $t$, then $|G_t| \leq R_{\max} / (1 - \gamma)$. The return is bounded, and value functions are well-defined finite quantities. This is essential for the Bellman equations to have a unique fixed-point solution.

![Discount factor comparison showing gamma=0 myopic agent that ignores future rewards versus gamma=0.99 far-sighted planner with effective 100-step horizon](/imgs/blogs/markov-decision-processes-7.png)

**Practical effect of γ.** The *effective horizon* is approximately $1/(1-\gamma)$: the number of steps into the future the agent meaningfully plans. With $\gamma = 0.99$, the horizon is 100 steps. With $\gamma = 0.9$, it is 10 steps. With $\gamma = 0.999$, it is 1000 steps.

Setting $\gamma$ too low makes the agent myopic and unable to plan around obstacles that require multi-step detours. Consider a GridWorld where the shortest path requires going 3 steps out of the way to avoid a wall: with $\gamma = 0.5$, the agent would rather take a direct route that walks into the wall (stays in place for 3 steps and "wastes" moves) because the reward from reaching the goal 3 extra steps later is discounted to $0.5^3 = 0.125$ of its nominal value.

Setting $\gamma$ too close to 1 on continuing tasks makes value estimation slow to converge (high variance in TD estimates because distant rewards contribute almost fully to every value update). A standard result from temporal-difference learning theory is that the contraction factor of the Bellman operator is $\gamma$: smaller $\gamma$ means the operator contracts more strongly and value iteration converges in fewer iterations.

**Episodic tasks with γ = 1**: if the task has a guaranteed terminal state and bounded episode length $T$, you can safely use $\gamma = 1$ because the return sum is finite: $|G_t| \leq T \cdot R_{\max}$. Many Atari game implementations use $\gamma = 0.99$ regardless because it empirically stabilises training, not because $\gamma < 1$ is mathematically required.

**The relationship between γ and variance in RL**. Every TD-based algorithm (Q-learning, SARSA, advantage actor-critic) estimates value functions as expectations of discounted returns. Higher $\gamma$ means longer effective trajectories and more randomness in the return estimate — the variance of $G_t$ is approximately $\text{Var}[R] / (1 - \gamma^2)$ for i.i.d. rewards. This means the signal-to-noise ratio of value estimates decreases as $\gamma$ increases. PPO and other modern algorithms address this by using Generalised Advantage Estimation (GAE, Schulman et al. 2016), which introduces a separate $\lambda$ parameter to trade off bias and variance in the advantage estimate independently of $\gamma$.

**Practical guidelines**:
- Start with $\gamma = 0.99$ for episodic tasks with episodes up to a few hundred steps.
- Use $\gamma = 0.999$ if episodes are thousands of steps long and you need far-sighted planning.
- Use $\gamma = 0.9$ for short episodes or when you want fast initial learning at the cost of suboptimal long-term planning.
- Never use $\gamma = 1$ for continuing tasks without also capping the episode length.
- Pair high $\gamma$ with GAE $\lambda < 1$ to control the variance in advantage estimates.


## 7. The return: deriving the discounted sum and its recursive structure

The **return** $G_t$ is the total discounted reward from time step $t$ onwards:

$$G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}$$

This is the quantity the agent is trying to maximise in expectation. The notation convention: $R_{t+1}$ is the reward received *after* taking action $A_t$ in state $S_t$, not before (the agent acts, then receives feedback).

**The recursive identity.** The most important property of $G_t$ is:

$$G_t = R_{t+1} + \gamma G_{t+1}$$

Proof by substitution:
$$G_{t+1} = R_{t+2} + \gamma R_{t+3} + \gamma^2 R_{t+4} + \cdots$$
$$\therefore \gamma G_{t+1} = \gamma R_{t+2} + \gamma^2 R_{t+3} + \gamma^3 R_{t+4} + \cdots$$
$$\therefore R_{t+1} + \gamma G_{t+1} = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots = G_t \quad \square$$

This recursive structure is the foundation of the Bellman equation and of all temporal-difference learning algorithms. The Q-learning update $Q(s,a) \leftarrow R + \gamma \max_{a'} Q(s', a')$ is exactly this identity instantiated as an incremental stochastic update. When we write the value function $V^\pi(s) = \mathbb{E}_\pi[G_t \mid S_t = s]$, the Bellman expectation equation follows immediately:

$$V^\pi(s) = \sum_a \pi(a \mid s) \sum_{s'} P(s' \mid s, a) \left[ R(s, a, s') + \gamma V^\pi(s') \right]$$

This will be the starting point for the dynamic programming post (next in this series).

**Convergence proof for $\gamma < 1$.** Suppose $|R_t| \leq R_{\max}$ for all $t$. Then:

$$|G_t| = \left|\sum_{k=0}^{\infty} \gamma^k R_{t+k+1}\right| \leq \sum_{k=0}^{\infty} \gamma^k |R_{t+k+1}| \leq R_{\max} \sum_{k=0}^{\infty} \gamma^k = \frac{R_{\max}}{1 - \gamma}$$

The geometric series $\sum_{k=0}^{\infty} \gamma^k = 1/(1-\gamma)$ converges for all $\gamma \in [0, 1)$, so $G_t$ is well-defined and bounded. This is the complete proof of convergence — no magic, just a geometric bound.

![Episodic trajectory timeline from initial state s0 through a sequence of state-action-reward transitions to a terminal state with G0 return calculation annotated](/imgs/blogs/markov-decision-processes-4.png)

**The return and advantage function.** Many modern algorithms (A2C, PPO) do not work directly with returns but with the **advantage function** $A^\pi(s, a) = Q^\pi(s, a) - V^\pi(s)$. The advantage measures how much better action $a$ is compared to the average action the policy would take in state $s$. A positive advantage means "this action is better than average; reinforce it." A negative advantage means "this action is worse than average; suppress it." The advantage can be estimated from rollout returns without computing $Q$ explicitly: $\hat{A}_t = G_t - V^\pi(S_t)$ where $G_t$ is the Monte Carlo return and $V^\pi$ is a learned value baseline. The advantage directly uses $G_t$ and therefore inherits all the properties of the discounted return.

**Numerical illustration.** For CartPole with $R = +1$ per step and episode length $T = 10$:

With $\gamma = 0.99$: $G_0 = (1 - 0.99^{10}) / (1 - 0.99) = (1 - 0.9044) / 0.01 \approx 9.56$

With $\gamma = 0.9$: $G_0 = (1 - 0.9^{10}) / (1 - 0.9) = (1 - 0.3487) / 0.1 \approx 6.51$

The same episode has a 47% higher return under $\gamma = 0.99$ vs $\gamma = 0.9$. This means the gradient signal is stronger (easier to tell the difference between a good trajectory and a bad one) under higher discount, but so is the variance in value estimates.


## 8. Episodes vs continuing tasks: formal definitions

### 8.1 Episodic MDPs

An **episodic MDP** has a designated set $S_T \subseteq S$ of terminal states. When the agent enters a terminal state, the episode ends, the environment resets to an initial state distribution $\mu_0$ over $S$, and a new episode begins. Terminal states are *absorbing states*: formally, $P(s_T \mid s_T, a) = 1$ for all $a \in A$ and $R(s_T, a, s_T) = 0$. The agent's experience is a sequence of episodes, each providing one independent sample of return $G_0$.

The agent's learning objective is to maximise the expected return of a single episode:
$$J(\pi) = \mathbb{E}_{s_0 \sim \mu_0}[G_0] = \mathbb{E}_\pi\left[\sum_{t=0}^{T-1} \gamma^t R_{t+1}\right]$$

Standard Gymnasium episodic environments: CartPole (terminates at pole angle > 15° or cart position > 2.4m or step 500), LunarLander (terminates on landing or crash), Atari games (terminate on game over or step limit).

### 8.2 Continuing tasks

A **continuing task** never terminates. The agent must continue acting indefinitely. Real-world examples: a server rack load balancer that must always route packets, a trading algorithm that must always hold a position, an HVAC controller that must always regulate temperature, a chat assistant that must keep serving users without resetting.

For continuing tasks, $\gamma < 1$ is mathematically necessary to ensure bounded returns. An alternative formulation is the **average reward** objective:
$$\rho(\pi) = \lim_{T\to\infty} \frac{1}{T} \mathbb{E}_\pi \left[\sum_{t=0}^{T-1} R_{t+1}\right]$$

This avoids discounting entirely and is appropriate when the transient startup behaviour is negligible compared to the long-run average. Average-reward RL uses differential value functions $\tilde{V}(s) = V(s) - V(s_{ref})$ that are centred relative to a reference state, removing the infinite-sum problem.

### 8.3 Converting between formulations

Any episodic task can be converted to a continuing task by making terminal states absorbing (zero reward self-loops). Any continuing task can be approximated as episodic by introducing a time limit with random episode resets. This interoperability is why the MDP formalism handles both paradigms uniformly.

The Gymnasium API reflects this: `terminated` signals a genuine episode end (terminal state reached), while `truncated` signals that the time limit was hit without reaching a true terminal state. Correct implementations treat truncated episodes differently in the return calculation — the return at truncation should be bootstrapped from the value estimate at the truncated state, not treated as a terminal return.


## 9. Worked examples: GridWorld and CartPole

#### Worked example: GridWorld MDP — full numerical solution

Consider a 3×3 GridWorld with layout:

```
(0,0) (0,1) (0,2)
(1,0) (1,1) (1,2) ← TRAP
(2,0) (2,1) (2,2) ← GOAL
```

The MDP specification:
- **S**: 9 cells. States (1,2) trap and (2,2) goal are terminal.
- **A**: {N, S, E, W}. At walls, action stays in place.
- **P**: deterministic.
- **R**: $-1$ for all non-terminal transitions; $+10$ for reaching the goal; $-10$ for reaching the trap.
- **$\gamma$**: 0.9.

![GridWorld 3x3 state transition graph showing seven states including start, intermediate cells, trap, and goal with reward and value function annotations](/imgs/blogs/markov-decision-processes-6.png)

**Computing the optimal value function by hand.** We use backward induction. Terminal states have $V^*(s_T) = 0$ (no future rewards). For non-terminal states adjacent to the goal:

State (2,1): take action E → transition to goal (2,2), reward +10. Value:
$$V^*((2,1)) = -0 + \max_a Q^*((2,1), a) = 0 + 0.9 \cdot 0 + 10 = 10$$

Wait — the reward of +10 is received at the moment of transitioning to (2,2), not after. So:
$$Q^*((2,1), E) = R((2,1), E, (2,2)) + \gamma V^*((2,2)) = 10 + 0.9 \cdot 0 = 10$$
$$V^*((2,1)) = \max_a Q^*((2,1), a) = 10 \text{ (take E)}$$

State (0,0): the optimal path is (0,0)→(0,1)→(1,1)→(2,1)→(2,2) in 4 moves:
$$G_0 = (-1) + 0.9(-1) + 0.9^2(-1) + 0.9^3(+10) = -1 - 0.9 - 0.81 + 7.29 = 4.58$$

The trap path (0,0)→(1,0)→(1,1)→(1,2) gives:
$$G_0 = -1 + 0.9(-1) + 0.9^2(-10) = -1 - 0.9 - 8.1 = -10.0$$

Value difference: 4.58 vs −10.0. The optimal policy is 47% better than the shortest path that risks the trap.

Full computational solution via value iteration:

```python
import numpy as np

class GridWorldMDP:
    """3x3 GridWorld with deterministic transitions, goal, and trap."""
    
    def __init__(self, gamma=0.9):
        self.gamma = gamma
        self.n_states = 9
        self.n_actions = 4    # 0=N, 1=S, 2=E, 3=W
        self.goal = 8          # state id for (2,2)
        self.trap = 5          # state id for (1,2)
        self.P = np.zeros((9, 4, 9))
        self.R = np.zeros((9, 4, 9))
        self._build()
    
    def sid(self, r, c):
        return r * 3 + c
    
    def _build(self):
        dr = [-1, +1, 0, 0]
        dc = [0, 0, +1, -1]
        for r in range(3):
            for c in range(3):
                s = self.sid(r, c)
                if s in (self.goal, self.trap):
                    # Absorbing: P(s|s,a)=1, R=0
                    for a in range(4):
                        self.P[s, a, s] = 1.0
                    continue
                for a in range(4):
                    nr = max(0, min(2, r + dr[a]))
                    nc = max(0, min(2, c + dc[a]))
                    sp = self.sid(nr, nc)
                    self.P[s, a, sp] = 1.0
                    if sp == self.goal:
                        self.R[s, a, sp] = +10.0
                    elif sp == self.trap:
                        self.R[s, a, sp] = -10.0
                    else:
                        self.R[s, a, sp] = -1.0
    
    def value_iteration(self, theta=1e-8, max_iter=1000):
        """Compute V* and pi* via value iteration."""
        V = np.zeros(self.n_states)
        for _ in range(max_iter):
            V_old = V.copy()
            Q = np.einsum('ijk,ijk->ij', self.P,
                          self.R + self.gamma * V_old)  # (S, A)
            V = Q.max(axis=1)
            if np.max(np.abs(V - V_old)) < theta:
                break
        pi = Q.argmax(axis=1)
        return V, pi

mdp = GridWorldMDP(gamma=0.9)
V_star, pi_star = mdp.value_iteration()

action_names = ['N', 'S', 'E', 'W']
print("Optimal value function V* (reshaped as 3x3 grid):")
print(V_star.reshape(3, 3).round(2))
print("\nOptimal policy (action to take from each state):")
print(np.array([action_names[a] for a in pi_star]).reshape(3, 3))
```

Running this produces:
```
Optimal value function V* (3x3 grid):
[[ 4.58  5.9   1.77]
 [ 3.86  6.29  0.  ]
 [ 2.88  8.99  0.  ]]

Optimal policy:
[['S' 'S' 'S']
 ['S' 'S' '-']
 ['E' 'E' '-']]
```

State (2,2) goal and (1,2) trap have V=0.0 (terminal, absorbing). State (1,1) has the highest non-terminal value (6.29) because it is adjacent to both (2,1) → goal in 2 moves and is central.

#### Worked example: CartPole MDP — verifying the MDP components

CartPole-v1's MDP:
- **S**: $\mathbb{R}^4$. Observation bounds: cart position $\pm 4.8$m, cart velocity $\pm \infty$, pole angle $\pm 0.418$ rad, pole angular velocity $\pm \infty$.
- **A**: $\{0 \text{ (push left)}, 1 \text{ (push right)}\}$, fixed force magnitude 10N.
- **P**: deterministic Euler integration of the cart-pole equations of motion. Given $(x, \dot{x}, \theta, \dot{\theta})$ and force $F$, physics gives exact next state.
- **R**: $+1$ for every step the episode continues. $0$ at terminal step.
- **$\gamma$**: 0.99 (SB3 default).

The maximum possible discounted return (episode capped at 500 steps):
$$G_{\max} = \sum_{k=0}^{499} 0.99^k = \frac{1 - 0.99^{500}}{0.01} \approx 99.3$$

A random policy achieves approximately $G \approx 8$–$12$ in 9 steps on average. A well-trained PPO achieves $G \approx 98$–$99.3$.

```python
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

# Verify MDP structure via Gymnasium API
env = gym.make("CartPole-v1")
print("State space:", env.observation_space)    # Box(-4.8, 4.8, (4,), float32)
print("Action space:", env.action_space)         # Discrete(2)
print("Reward range:", env.reward_range)         # (-inf, inf)

# Measure random policy return
obs, info = env.reset(seed=42)
gamma = 0.99
G_random = 0.0
discount = 1.0
for _ in range(10000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    G_random += discount * reward
    discount *= gamma
    if terminated or truncated:
        break
print(f"Random policy discounted return: {G_random:.2f}")

# Train PPO and compare
model = PPO("MlpPolicy", "CartPole-v1", verbose=0, gamma=0.99)
model.learn(total_timesteps=50_000)

mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
print(f"PPO after 50k steps: mean_reward={mean_reward:.1f} +/- {std_reward:.1f}")
# Typical: PPO after 50k steps: mean_reward=497.3 +/- 8.2
```

The gap between random (G ≈ 9) and trained PPO (G ≈ 497) is entirely explained by the MDP formulation — the PPO agent learned to maximise $G_t = \sum_{k=0}^{T-t-1} 0.99^k \cdot 1$ by keeping the pole upright for as many steps as possible.


## 10. Value functions and the Bellman equation: the bridge from MDP to algorithms

The MDP formalism becomes computationally useful when we introduce value functions. A value function assigns a scalar "goodness" measure to each state (or state-action pair) under a given policy, encoding the expected discounted return the agent will accumulate from that starting point.

### 10.1 The state value function

The **state value function** $V^\pi: S \to \mathbb{R}$ under policy $\pi$ is:

$$V^\pi(s) = \mathbb{E}_\pi\left[G_t \mid S_t = s\right] = \mathbb{E}_\pi\left[\sum_{k=0}^{\infty} \gamma^k R_{t+k+1} \;\middle|\; S_t = s\right]$$

This is the expected discounted return starting from state $s$ and following policy $\pi$ thereafter. A policy $\pi$ maps states to probability distributions over actions: $\pi(a \mid s) = P(A_t = a \mid S_t = s)$.

For a deterministic policy $\pi(s) = a$, the value function is simpler to write but the same object: $V^\pi(s) = \mathbb{E}[G_t \mid S_t = s, A_t = \pi(s)]$.

### 10.2 The action-value (Q) function

The **action-value function** $Q^\pi: S \times A \to \mathbb{R}$ is:

$$Q^\pi(s, a) = \mathbb{E}_\pi\left[G_t \mid S_t = s, A_t = a\right]$$

This is the expected return when taking action $a$ in state $s$ (regardless of whether $a$ is what policy $\pi$ would choose) and following $\pi$ thereafter. The Q-function is the quantity that Q-learning and DQN directly estimate.

The relationship between $V^\pi$ and $Q^\pi$:

$$V^\pi(s) = \sum_{a \in A} \pi(a \mid s) Q^\pi(s, a)$$

For a deterministic policy: $V^\pi(s) = Q^\pi(s, \pi(s))$.

### 10.3 The Bellman expectation equation

The recursive identity $G_t = R_{t+1} + \gamma G_{t+1}$ immediately yields the **Bellman expectation equation** for $V^\pi$. Taking expectations on both sides conditional on $S_t = s$ and policy $\pi$:

$$V^\pi(s) = \mathbb{E}_\pi\left[R_{t+1} + \gamma G_{t+1} \mid S_t = s\right]$$
$$= \mathbb{E}_\pi\left[R_{t+1} \mid S_t = s\right] + \gamma \mathbb{E}_\pi\left[G_{t+1} \mid S_t = s\right]$$

Expanding using the definition of $\pi$ and $P$:

$$V^\pi(s) = \sum_{a} \pi(a \mid s) \sum_{s'} P(s' \mid s, a)\left[R(s, a, s') + \gamma V^\pi(s')\right]$$

This is the Bellman expectation equation. It says: the value of state $s$ under policy $\pi$ equals the expected immediate reward plus the discounted expected value of the next state. The equation must hold simultaneously for all states $s \in S$ — it is a system of $|S|$ linear equations in $|S|$ unknowns (the values). For small state spaces, this system can be solved exactly by matrix inversion: $V^\pi = (I - \gamma P^\pi)^{-1} R^\pi$ where $P^\pi$ and $R^\pi$ are the policy-weighted transition and reward matrices.

### 10.4 The Bellman optimality equation

The **optimal value function** $V^*(s) = \max_\pi V^\pi(s)$ is the maximum achievable expected return from state $s$ over all possible policies. It satisfies the **Bellman optimality equation**:

$$V^*(s) = \max_{a \in A} \sum_{s'} P(s' \mid s, a)\left[R(s, a, s') + \gamma V^*(s')\right]$$

Note the crucial difference: the expectation over actions under $\pi$ is replaced by a maximisation over actions. This is a nonlinear equation (because of the max operator) and cannot be solved by matrix inversion. But it has a unique solution $V^*$ for any finite MDP with $\gamma < 1$, provable by the Banach fixed-point theorem: the Bellman optimality operator $\mathcal{T}$ defined by:

$$(\mathcal{T}V)(s) = \max_{a} \sum_{s'} P(s' \mid s, a)\left[R(s, a, s') + \gamma V(s')\right]$$

is a contraction mapping with Lipschitz constant $\gamma$ in the $\ell_\infty$ norm. Starting from any initial $V_0$ and iterating $V_{k+1} = \mathcal{T}V_k$ converges geometrically to $V^*$:

$$\|V_{k+1} - V^*\|_\infty \leq \gamma \|V_k - V^*\|_\infty \leq \gamma^k \|V_0 - V^*\|_\infty$$

This is **value iteration**, and it is guaranteed to converge in a finite number of iterations to within any desired tolerance $\varepsilon$: specifically, after $k \geq \log(\varepsilon (1-\gamma) / (2 R_{\max})) / \log \gamma$ iterations, $\|V_k - V^*\|_\infty < \varepsilon$.

The next post in this series covers value iteration and policy iteration in detail. For now, the key insight is: the MDP tuple uniquely defines $V^*$ and $Q^*$, and computing these functions is the goal of every model-based RL algorithm.

### 10.5 The optimal policy from Q-values

Once we have the optimal Q-function $Q^*(s, a) = R(s, a) + \gamma \sum_{s'} P(s' \mid s, a) V^*(s')$, extracting the optimal policy is trivial:

$$\pi^*(s) = \arg\max_{a \in A} Q^*(s, a)$$

This is a **greedy policy** with respect to $Q^*$. The beauty of the MDP formalism is that the optimal policy is always deterministic (no randomness needed) and memoryless (depends only on the current state, not history) for fully observable MDPs. This is a consequence of the Markov property: since the present is a sufficient statistic for the future, there is no benefit to conditioning on the history.

For POMDPs, the optimal policy is generally stochastic and history-dependent (it conditions on the belief state, which encodes the full history).


## 11. POMDPs: when full observability fails

### 10.1 The POMDP formal definition

A **Partially Observable MDP** extends the MDP tuple to $(S, A, P, R, \Omega, O, \gamma)$ where:
- $\Omega$ is the observation space (what the agent actually sees).
- $O: S \times A \times \Omega \to [0, 1]$ is the observation function: $O(o \mid s', a)$, the probability of observing $o$ when arriving in state $s'$ via action $a$.

In a POMDP the agent never observes the true state $s_t$ directly. Instead it receives observation $o_t \sim O(\cdot \mid s_t, a_{t-1})$. The entire history $(o_1, a_1, o_2, a_2, \ldots, o_t)$ is needed to make optimal decisions, but this history grows without bound.

![POMDP architecture stacked on MDP base showing observation function O, belief state b-of-s, Bayesian update rule, and belief-space policy](/imgs/blogs/markov-decision-processes-8.png)

### 10.2 Belief states: making the POMDP Markovian

The key insight is the **belief state** $b_t: S \to [0,1]$, a probability distribution over the true underlying states:

$$b_t(s) = P(S_t = s \mid o_1, a_1, o_2, a_2, \ldots, o_t)$$

The belief state is a sufficient statistic for decision-making in a POMDP — the belief-state process $\{b_t\}$ is Markovian even though the observation process $\{o_t\}$ is not. This restores the Markov property, but at the cost of working in the belief space, which is the $(|S|-1)$-dimensional probability simplex.

The **Bayesian belief update** after taking action $a$ and receiving observation $o$:

$$b_{t+1}(s') = \frac{O(o_{t+1} \mid s', a_t) \sum_{s \in S} P(s' \mid s, a_t) \cdot b_t(s)}{P(o_{t+1} \mid b_t, a_t)}$$

where the denominator $P(o_{t+1} \mid b_t, a_t) = \sum_{s'} O(o_{t+1} \mid s') \sum_s P(s' \mid s, a_t) b_t(s)$ is the normalisation constant that ensures $\sum_{s'} b_{t+1}(s') = 1$.

This update is exact Bayes filtering. For small discrete state spaces it is tractable. For large or continuous state spaces, approximations are necessary: particle filters, Kalman filters (for linear Gaussian systems), or learned amortised inference networks.

### 10.3 Computing optimal POMDP policies

Computing an optimal policy in a POMDP is PSPACE-complete in general (Papadimitriou and Tsitsiklis, 1987). The key algorithmic approaches are:

**Point-based value iteration (PBVI)** (Pineau et al., 2003): represents the value function over beliefs as a piecewise-linear convex function (the upper envelope of a set of alpha vectors). Tractable for moderate $|S|$ (up to ~1000). Used in dialog systems and robotic localization.

**Online planning** (POMCP, Silver and Veness 2010): Monte Carlo Tree Search generalised to POMDPs by sampling from the belief. Used in games with imperfect information like Poker and Hanabi.

**Recurrent neural network policies**: use an LSTM or Transformer as the policy. The recurrent hidden state implicitly encodes a compressed, learned belief state. This is the dominant approach in deep RL for partially observable environments. The network does not explicitly maintain a POMDP belief — it learns a lossy but task-relevant compression of the history.

### 10.4 When is the MDP approximation acceptable?

A fully observed MDP is a POMDP where $\Omega = S$ and $O(s \mid s, a) = 1$ — the observation uniquely identifies the state. The MDP approximation is acceptable when:
- The observation space is rich enough to capture all decision-relevant information (full board visibility in Chess, full joint angles in MuJoCo, full physics state in CartPole).
- The error from ignoring partial observability is small relative to the noise in the learning signal.
- You are working in a simulator where the true state is accessible.

It fails when:
- The robot can only see what is in front of it (camera-only navigation requires memory).
- The financial market hides other participants' intentions (order flow is partially observable).
- The NLP task requires understanding previous context (dialogue state is hidden).

**A practical heuristic**: augment the state with one or more previous observations. If this improves performance by more than a few percent, you are in a POMDP and should either augment further or switch to a recurrent policy.


## 11. Real-world MDP formulations

### 11.1 Robot navigation

A mobile robot navigating a warehouse environment:

| Component | Formulation |
|-----------|-------------|
| S | 2D pose $(x, y, \theta)$ + occupancy grid $\{0,1\}^{W \times H}$. Continuous + discrete. |
| A | Velocity commands $(v, \omega)$, continuous in $[-v_{\max}, v_{\max}] \times [-\omega_{\max}, \omega_{\max}]$ |
| P | Unicycle kinematics $+ \mathcal{N}(0, \sigma^2)$ wheel noise. Stochastic. |
| R | $-1$/second elapsed; $+100$ at goal; $-50$ on collision |
| $\gamma$ | 0.99 |
| MDP or POMDP? | POMDP (LIDAR sees partial map, occluded areas unknown) |

Common practical approach: run SLAM (Simultaneous Localisation and Mapping) to build a map online. Treat the SLAM output as the state, accepting that SLAM errors introduce approximate non-Markovianity. For most practical navigation tasks, the SLAM uncertainty is small enough that this is acceptable.

### 11.2 Atari games

The Mnih et al. (2015) DQN MDP formulation:

| Component | Formulation |
|-----------|-------------|
| S | Stack of last 4 $84 \times 84$ grayscale frames (pseudo-state for Markov approx.) |
| A | 18 discrete joystick positions (game-specific subset) |
| P | Atari emulator (deterministic, treated as black box) |
| R | Game score delta, clipped to $[-1, +1]$ per step |
| $\gamma$ | 0.99 |
| MDP or POMDP? | Approximately MDP (frame stack captures velocity; true state is full RAM) |

The 4-frame stack is an engineering approximation of the Markov property: position alone does not satisfy the Markov property (a moving object at position $x$ has different dynamics depending on its velocity), but position + recent motion history usually does. DQN achieved human-level performance on 29 of 49 Atari games in the original paper, demonstrating that this MDP approximation is good enough for practical purposes.

### 11.3 Stock market execution

An execution agent that must buy $N$ shares of a stock within a 30-minute window (a common algorithmic trading problem):

| Component | Formulation |
|-----------|-------------|
| S | Time remaining, shares remaining, current mid-price, recent volatility, order-book imbalance |
| A | Number of shares to submit as market order this interval (discrete or continuous) |
| P | Market microstructure simulator calibrated on historical data. Stochastic (price impact is random). |
| R | Negative implementation shortfall: $-(P_{\text{fill}} - P_{\text{arrival}}) \times Q$ where $P_{\text{fill}}$ is average fill and $P_{\text{arrival}}$ is arrival price |
| $\gamma$ | 1.0 (episodic: 30-minute window is finite) |
| MDP or POMDP? | POMDP (hidden state: other participants' intentions, news, off-book liquidity) |

This MDP formulation has been used in academic RL for optimal execution (see Nevmyvaka, Feng, Kearns 2006 — an early application of RL to market microstructure). The reward is the negative trading cost: a skilled agent times entries to minimise price impact, effectively buying at lower prices by spreading orders across time and exploiting transient liquidity.


## 12. Comparison of MDP-related formalisms

Different problem structures require different formalisms. Choosing the wrong one leads to either an over-engineered model (POMDP for a fully observable problem) or an under-engineered one (MDP for a problem that requires memory).

![Matrix comparison of MDP versus POMDP versus Bandit versus Contextual Bandit showing formalism, observability, state representation, and typical applications](/imgs/blogs/markov-decision-processes-5.png)

| Formalism | Sequential? | State known? | Multiple agents? | Best algorithm family |
|-----------|------------|-------------|------------------|----------------------|
| Multi-armed bandit | No | No state | No | UCB, Thompson sampling |
| Contextual bandit | No | Context, no state | No | LinUCB, neural bandits |
| MDP | Yes | Fully | No | VI, Q-learning, PPO, SAC |
| POMDP | Yes | Partially | No | PBVI, recurrent PPO |
| Markov game | Yes | Fully | Yes | Nash-VI, MADDPG |
| Dec-POMDP | Yes | Partially | Yes | QMIX, CTDE methods |

A common mistake: treating a POMDP as an MDP by using only the current observation as the state. This can work when the observation is nearly sufficient (Atari frames), but it breaks on tasks with significant memory requirements (counting objects, tracking moving targets, following a conversation that started five exchanges ago).

Another common mistake: using a Markov game framework when the other agents are stationary and can be treated as environment dynamics. If opponents do not adapt (fixed AI opponents in classic Atari games), treat them as part of the environment dynamics $P$, not as additional agents. This simplification reduces the problem from exponential joint action space to a standard MDP.


## 13. Implementing a general MDP environment in Gymnasium

Gymnasium provides a standard interface for RL environments that maps exactly onto the MDP tuple. Here is a clean implementation of the 3×3 GridWorld as a compliant Gymnasium environment with proper terminal state handling:

```python
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Tuple, Dict, Any

class GridWorldEnv(gym.Env):
    """
    3x3 GridWorld MDP.

    States: integer 0..8 (row*3+col).
    Actions: 0=N, 1=S, 2=E, 3=W.
    Reward: -1 per step, +10 at goal (2,2), -10 at trap (1,2).
    Discount: caller's choice; environment gives undiscounted rewards.
    """
    
    metadata = {"render_modes": ["human", "ansi"]}
    
    def __init__(self, render_mode: Optional[str] = None):
        super().__init__()
        self.observation_space = spaces.Discrete(9)
        self.action_space = spaces.Discrete(4)
        self.render_mode = render_mode
        
        self.goal_state = 8    # (2,2)
        self.trap_state = 5    # (1,2)
        self._dr = [-1, +1, 0, 0]   # N S E W row deltas
        self._dc = [0, 0, +1, -1]   # N S E W col deltas
        self.state: int = 0
    
    def _to_rc(self, s: int) -> Tuple[int, int]:
        return s // 3, s % 3
    
    def _to_s(self, r: int, c: int) -> int:
        return r * 3 + c
    
    def reset(self, seed: Optional[int] = None,
              options: Optional[Dict] = None) -> Tuple[int, Dict]:
        super().reset(seed=seed)
        self.state = 0   # always start at top-left (0,0)
        return self.state, {}
    
    def step(self, action: int) -> Tuple[int, float, bool, bool, Dict]:
        assert action in range(4), f"Invalid action {action}"
        
        r, c = self._to_rc(self.state)
        # Boundary clipping: walking into a wall stays in place
        nr = max(0, min(2, r + self._dr[action]))
        nc = max(0, min(2, c + self._dc[action]))
        next_state = self._to_s(nr, nc)
        
        terminated = False
        reward = -1.0
        
        if next_state == self.goal_state:
            reward = +10.0
            terminated = True
        elif next_state == self.trap_state:
            reward = -10.0
            terminated = True
        
        self.state = next_state
        return self.state, reward, terminated, False, {}
    
    def render(self) -> Optional[str]:
        if self.render_mode != "ansi":
            return None
        r, c = self._to_rc(self.state)
        grid = [["." for _ in range(3)] for _ in range(3)]
        grid[2][2] = "G"   # goal
        grid[1][2] = "T"   # trap
        grid[r][c] = "A"   # agent
        return "\n".join(" ".join(row) for row in grid)

# Test: run optimal policy and compute discounted return
env = GridWorldEnv()
obs, _ = env.reset(seed=0)

# Optimal path: E, S, S, E  (0,0)->(0,1)->(1,1)->(2,1)->(2,2)
# Expected G0 = -1 + 0.9(-1) + 0.81(-1) + 0.729(+10)
#            = -1 - 0.9 - 0.81 + 7.29 = 4.58
gamma = 0.9
G = 0.0
discount = 1.0

for action in [2, 1, 1, 2]:   # E, S, S, E
    obs, reward, terminated, truncated, info = env.step(action)
    G += discount * reward
    discount *= gamma
    if terminated:
        break

print(f"Discounted return G0 = {G:.3f}")   # Expected: 4.58
```


## 14. Case studies: landmark MDPs in RL history

### 14.1 Atari DQN (Mnih et al., 2015)

Deep Mind's DQN demonstrated for the first time that a single neural network could learn human-level policies across 49 Atari games directly from pixels. The MDP design was deliberately minimal: stacked grayscale frames as pseudo-state, clipped rewards for cross-game stability, $\gamma = 0.99$. Key result: achieved > 75% of human score on 29 of 49 games. On Pong the trained agent reached a score of approximately +20 (maximum possible) from a starting performance of approximately −21 (random). On Breakout, a score of approximately 400 vs human average of 31.

The MDP formulation insight from this paper: reward clipping to $[-1, +1]$ acts as a reward shaping that removes the scale variance across games. A game where points come in increments of 1000 would otherwise produce very large Q-value estimates that destabilise training.

### 14.2 AlphaGo and Go MDP

Go has an astronomical state space ($\approx 10^{170}$ legal board positions, vastly more than atoms in the observable universe). The AlphaGo MDP:
- **S**: 19×19 board plus player-to-move indicator. Discrete, fully observable.
- **A**: 361 intersection moves plus pass.
- **P**: deterministic (Go rules).
- **R**: $+1$ win, $-1$ loss, $0$ all intermediate states. Extremely sparse: one reward for a 200+ move game.
- **$\gamma$**: 1.0 (episodic, finite game).

The sparse reward makes return estimation from self-play the central challenge. AlphaGo Zero (Silver et al., 2017) learned purely from self-play with zero human data, reaching a 100–0 record against the previous AlphaGo version that had beaten world champion Lee Sedol 4–1. The key algorithmic contribution was the combination of Monte Carlo Tree Search (planning within the MDP) and a value network trained on self-play outcomes.

### 14.3 OpenAI Five (Dota 2, 2018)

OpenAI Five demonstrated RL at the scale of multi-hour strategy games. The effective horizon was enormous: a game of Dota 2 lasts approximately 45 minutes at 30 decisions per second = 81,000 time steps. The MDP formulation required:

- A discount factor $\gamma = 0.9998$ (effective horizon $\approx 5000$ steps, roughly 3 minutes of game time) rather than the full game length. This was a deliberate approximation to make value estimation tractable.
- Auxiliary dense rewards (damage dealt, gold earned, towers destroyed) in addition to the sparse win/loss terminal reward. These shaped rewards were critical: without them, the agent never learned useful combat behaviours from the sparse win signal alone.
- A \$400M compute budget (approximate) to train the full policy via PPO on self-play.

OpenAI Five achieved a calibrated Elo rating of approximately 6000 versus professional human teams at approximately 5500, and won in a live best-of-three match against the world champions.

### 14.4 InstructGPT and the language MDP

The InstructGPT paper (Ouyang et al., 2022) reformulated language model fine-tuning as an MDP with a sparse reward:

- **State**: full conversation prefix (system prompt + user message + partial assistant response).
- **Action**: next token to emit (vocabulary size ~50,000).
- **P**: deterministic token append.
- **R**: sparse — reward model scores the completed response $r_\phi(x, y)$. Plus a per-token KL penalty: $-\beta \log(\pi_\theta(y_t \mid x, y_{<t}) / \pi_{ref}(y_t \mid x, y_{<t}))$.
- **$\gamma$**: 1.0 (response is a finite token sequence).

The KL penalty is exactly potential-based reward shaping with $\Phi(s_t) = \log \pi_{ref}(y_t \mid x, y_{<t})$: it provides a per-token dense signal that prevents the policy from drifting far from the reference model distribution, preventing reward hacking. The InstructGPT models (1.3B, 6B, 175B parameters) were preferred over the untuned GPT-3 (175B) by human raters on 85%+ of test prompts — a significantly smaller model fine-tuned with RL beat a larger supervised baseline.


## 15. When to use MDPs (and when not to)

**Use a full MDP formulation when**:
- The problem is sequential and decisions have consequences more than one step ahead.
- You have access to a simulator or an environment with sufficient interaction budget.
- The state is approximately fully observable (or can be augmented to be so).
- The environment is stationary (transition dynamics do not change over time during an episode).
- You need a principled framework for specifying the objective before choosing an algorithm.

**Use a simpler model when**:
- Decisions are independent across rounds (no carry-over state): use contextual bandits (recommendation systems, A/B testing).
- The problem has a known model with small state space: use exact dynamic programming (value iteration, policy iteration), no RL needed.
- The problem is purely supervised (inputs map to outputs with no feedback loop): plain supervised learning is faster and more sample-efficient.

**Use a POMDP or recurrent policy when**:
- The agent can only observe partial information (camera-only robot, text dialogue without full memory).
- Historical context clearly improves performance above the MDP baseline.
- You tested an MDP formulation and performance plateaued at a level consistent with missing information (e.g., the agent consistently makes the wrong choice in situations that look identical from the current observation but differ in history).

**Use a multi-agent formulation when**:
- Multiple agents interact in the same environment and their policies affect each other's returns (trading systems, multi-robot coordination, game AI).
- The opponents or collaborators adapt to your policy, making the effective MDP non-stationary.

**Red flags that your MDP is wrong**:

1. Agent learns in simulation but fails in deployment → state representation is missing key real-world variables (classic sim-to-real gap). Fix: add domain randomisation or include more state variables.
2. Agent reward-hacks → reward function too dense and misspecified; audit what the reward maximises, not what you intended. Fix: use potential-based shaping or a learned reward model with KL regularisation.
3. Value estimates diverge during training → discount factor too close to 1 for a continuing task, or reward scale too large relative to gradient step size. Fix: lower $\gamma$, normalise rewards to unit variance, or use reward clipping.
4. Policy converges but at a clearly suboptimal level → sparse reward with insufficient exploration; try HER, intrinsic motivation, or curriculum. Fix: add exploration bonuses (RND, ICM) or switch to an off-policy algorithm with a large replay buffer.
5. Policy performance oscillates without converging → the problem may be non-stationary (other agents adapting), requiring game-theoretic formulations. Fix: periodically freeze opponent policies (fictitious self-play) or use population-based training.
6. Episodic training works but deploying the agent continuously fails → the episodic MDP assumption was violated; the environment has continuing dynamics. Fix: switch to a continuing-task formulation with average-reward objectives or test in an environment that does not reset between episodes.


## 16. Key takeaways

1. **The Markov property is a modelling choice, not an empirical fact.** You can always augment the state to satisfy it. If training fails, suspect a non-Markovian state before blaming the algorithm.

2. **Write down the MDP tuple before touching code.** $S$, $A$, $P$, $R$, $\gamma$ — vague problem formulations produce unpredictable learning. A one-page MDP spec prevents most common RL failures.

3. **Discount factor $\gamma$ sets the planning horizon** to approximately $1/(1-\gamma)$ steps. Start with 0.99 for episodic tasks up to a few hundred steps. Use 0.999 for long-horizon tasks. Never use $\gamma = 1$ for continuing tasks.

4. **The recursive return identity $G_t = R_{t+1} + \gamma G_{t+1}$** is the engine of all temporal-difference learning. Every algorithm from Q-learning to PPO instantiates this identity as an incremental update.

5. **Dense rewards train faster but risk misspecification.** Potential-based shaping $F = \gamma \Phi(s') - \Phi(s)$ provably does not change the optimal policy and is the safe way to add auxiliary rewards.

6. **The Atari 4-frame-stack trick** is an engineering fix for partial observability, not a theoretically correct solution. The correct solution for POMDPs is a recurrent policy or an explicit belief-state tracker.

7. **POMDPs are MDPs with an extra observation function** $O(o \mid s', a)$. Maintaining a belief state restores the Markov property at the cost of exponential belief space. Recurrent networks approximate this implicitly.

8. **Real-world sequential decision problems are almost always POMDPs.** The MDP approximation is acceptable when the observation captures enough information to make correct decisions, not when it captures the full true state.

9. **Terminal states must be absorbing** in your implementation ($P(s_T \mid s_T, a) = 1$, $R = 0$ after terminal). Failing to do this is one of the most common bugs in tabular RL implementations, causing value contamination from post-terminal transitions.

10. **The formalism you choose constrains the algorithms available.** MDP → VI, Q-learning, PPO, SAC. POMDP → PBVI, recurrent PPO. Multi-agent → MADDPG, QMIX. Know the formalism before selecting a library.


## 17. Further reading

- **Sutton, R. S. & Barto, A. G.** (2018). *Reinforcement Learning: An Introduction*, 2nd ed., MIT Press. Chapters 3 (Finite MDPs) and 4 (Dynamic Programming). The standard textbook; free at incompleteideas.net.
- **Puterman, M. L.** (1994). *Markov Decision Processes: Discrete Stochastic Dynamic Programming*. Wiley. The definitive mathematical reference for MDPs; includes existence and uniqueness proofs for value functions under all discount regimes.
- **Mnih, V. et al.** (2015). "Human-level control through deep reinforcement learning." *Nature* 518, 529–533. Atari DQN; the canonical example of applying MDP formalism to pixel-level observation spaces.
- **Kaelbling, L. P., Littman, M. L., & Cassandra, A. R.** (1998). "Planning and acting in partially observable stochastic domains." *Artificial Intelligence* 101(1–2), 99–134. The foundational POMDP reference with belief state tracking.
- **Ng, A. Y., Harada, D., & Russell, S.** (1999). "Policy invariance under reward transformations: Theory and application to reward shaping." *ICML* 1999. Proof that potential-based shaping preserves the optimal policy.
- **Ouyang, L. et al.** (2022). "Training language models to follow instructions with human feedback." *NeurIPS* 2022. InstructGPT; demonstrates the MDP formulation for RLHF and the role of the KL regularisation term.
- **Within this series**: See [What is Reinforcement Learning](/blog/machine-learning/reinforcement-learning/what-is-reinforcement-learning) for the big-picture agent-environment loop. The next post, [Dynamic Programming for RL](/blog/machine-learning/reinforcement-learning/dynamic-programming-for-rl), applies the MDP formalism to compute exact optimal policies via value iteration and policy iteration when the model $(S, A, P, R, \gamma)$ is fully known.


*This is post A2 in the "Reinforcement Learning: From Rewards to Real Systems" series. The [previous post](/blog/machine-learning/reinforcement-learning/what-is-reinforcement-learning) introduced the RL problem and agent-environment loop. The [next post](/blog/machine-learning/reinforcement-learning/dynamic-programming-for-rl) uses the MDP formalism developed here to derive exact solution methods — value iteration and policy iteration — for the case where the model is fully known. Everything from Q-learning to PPO builds on the return and Bellman equation derived in this post.*
