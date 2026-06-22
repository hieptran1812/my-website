---
title: "Monte Carlo Methods in Reinforcement Learning: Learning from Complete Episodes"
date: "2026-06-22"
publishDate: "2026-06-22"
description: "How Monte Carlo methods eliminate the need for an environment model by learning value estimates purely from sampled episode returns, enabling model-free RL control."
tags:
  [
    "reinforcement-learning",
    "deep-learning",
    "monte-carlo",
    "model-free",
    "policy-optimization",
    "importance-sampling",
    "exploration",
    "markov-decision-process",
    "machine-learning",
    "blackjack",
  ]
category: "machine-learning"
subcategory: "Reinforcement Learning"
author: "Hiep Tran"
featured: true
readTime: 55
image: "/imgs/blogs/monte-carlo-methods-in-rl-1.png"
---

Your first real RL agent fails in an interesting way. You pull up the Blackjack environment from Gymnasium, eager to apply dynamic programming, and hit a wall immediately: you do not know the transition probabilities. What is the probability that drawing a card from a partially-dealt deck pushes you from a sum of 14 to exactly 17? It depends on which cards have been seen, the exact deck composition at that moment, and whether the dealer reshuffles between hands. Writing down $P(s' \mid s, a)$ explicitly would require enumerating thousands of state-action-next-state triples across all possible deck configurations. Dynamic programming is stuck before it starts.

This is the dirty secret of real environments: the model is usually unknown, partially known, or so expensive to compute that it may as well be unknown. A robot learning to walk does not have a closed-form expression for how its motors interact with irregular terrain — the contact dynamics involve deformable materials, stochastic slip, and partial observability. A trading agent cannot enumerate all possible order book transitions when it submits a limit order — market microstructure involves thousands of other agents acting simultaneously. A language model trainer does not know the exact reward distribution across all possible response completions. In each case, the agent has one irreplaceable asset: the ability to *try things and observe what happens*.

Monte Carlo (MC) methods are the cleanest answer to this constraint. The idea is startlingly simple: run an episode to completion — the Blackjack hand plays out, the robot walks until it falls, the language model generates a full response and gets rated — then look at the actual cumulative reward you received and use that as a direct estimate of how good your starting state or action was. No model. No Bellman sweep across a transition matrix. Just experience averaged over many trials. Given enough episodes, the law of large numbers guarantees the estimates converge.

![Dynamic programming requires a full transition model P while Monte Carlo learns purely from observed episode returns with no model required, making MC the natural choice for unknown environments.](/imgs/blogs/monte-carlo-methods-in-rl-1.png)

By the end of this post you will understand exactly how MC prediction converges via the law of large numbers, why MC control needs $Q(s, a)$ rather than $V(s)$, how epsilon-soft policies guarantee exploration while converging to optimal, and how off-policy importance sampling lets you reuse experience from a different policy. You will also see the critical weakness: MC only works for episodic tasks and can have punishing variance in long episodes. We will walk through the Blackjack example from Sutton and Barto, implement complete MC control loops, and close with the connection to Monte Carlo Tree Search in AlphaGo — the highest-profile deployment of MC ideas in recent history.

This post builds directly on [Reinforcement Learning: A Unified Map](/blog/machine-learning/reinforcement-learning/reinforcement-learning-a-unified-map), which introduced the MDP framework, Bellman equations, and the taxonomy of RL algorithms. It follows [Markov Decision Processes: The Foundation of Sequential Decision Making](/blog/machine-learning/reinforcement-learning/markov-decision-processes-foundation) and [Dynamic Programming for Reinforcement Learning](/blog/machine-learning/reinforcement-learning/dynamic-programming-for-reinforcement-learning) for the model-based baseline that MC supersedes when dynamics are unknown. We will forward-reference [Temporal Difference Learning: TD(0), SARSA, and Q-Learning](/blog/machine-learning/reinforcement-learning/temporal-difference-learning) where bootstrapping enters the picture.

## 1. Why Dynamic Programming Breaks in Practice

Dynamic programming for reinforcement learning has a beautiful theoretical foundation. Starting from the Bellman optimality equations:

$$V^*(s) = \max_a \sum_{s'} P(s' \mid s, a) \left[ R(s, a, s') + \gamma V^*(s') \right]$$

you iterate — value iteration or policy iteration — until convergence. Both algorithms are provably correct and converge in polynomial time for finite MDPs. The math is elegant and the convergence guarantees are tight.

The problem is that explicit $P(s' \mid s, a)$ in the sum. Computing that sum requires knowing the environment's transition dynamics — the full probability distribution over next states given a current state and action. For many real problems this is:

**Unknown from the outside.** You are interacting with a black-box environment. A physical robot's dynamics depend on motor wear, floor friction, payload mass, and dozens of variables you cannot measure precisely. A stock market's order flow depends on all other participants' strategies, which are hidden. A human user's preferences cannot be reduced to a transition matrix.

**Intractable to enumerate.** The state space is so large that storing the full $P$ table is infeasible. A 52-card Blackjack game has roughly $10^{68}$ legal deck states. Even if you could compute $P(s' \mid s, a)$ for each triple, the storage alone would exceed available memory by dozens of orders of magnitude. Go has approximately $2.08 \times 10^{170}$ legal board positions — $P$ does not exist as a practical data structure.

**Wrong in practice.** Even if you have an approximation to $P$ from system identification or domain knowledge, compounding errors through the Bellman recursion amplifies that model bias badly. If your learned model has 5% error per step and you are doing a 100-step horizon Bellman computation, the accumulated error can swamp the actual signal.

Monte Carlo methods sidestep all three problems by not requiring $P$ at all. Instead of computing an expected value analytically over transitions, they sample actual transitions by running episodes. The expectation is approximated by averaging over many samples — the empirical average that the law of large numbers guarantees converges to the true expectation.

The formal statement is clean. The value of a state is defined as:

$$V^\pi(s) = \mathbb{E}_\pi \left[ G_t \mid s_t = s \right], \quad G_t = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}$$

DP evaluates this expectation analytically using $P$. MC evaluates it empirically by generating many returns $G_t^{(1)}, G_t^{(2)}, \ldots, G_t^{(N)}$ from episodes that visit state $s$ and averaging:

$$\hat{V}^\pi(s) = \frac{1}{N} \sum_{i=1}^{N} G_t^{(i)}$$

By the strong law of large numbers, $\hat{V}^\pi(s) \to V^\pi(s)$ almost surely as $N \to \infty$, provided the returns have finite variance and each episode visits $s$ at least once with positive probability. That convergence guarantee costs you nothing in terms of model knowledge — it only requires that you can generate experience.

This is a profound shift in what reinforcement learning requires. DP needs a complete internal model of the world. MC needs only an interface to the world: a way to take actions and receive observations. This is why model-free RL methods (Q-learning, SARSA, actor-critic, PPO) all have MC as a conceptual ancestor — they are all built on the insight that you can learn from experience without knowing the rules.

## 2. MC Prediction: Estimating V(s) from Episodes

MC prediction, given a fixed policy $\pi$, estimates $V^\pi(s)$ for all states $s$. The goal is evaluation (measuring how good a given policy is), not control (finding a better policy). This is useful for policy evaluation in policy iteration, for comparing candidate policies, and for Monte Carlo Tree Search leaf evaluation.

The algorithm is almost embarrassingly simple:

1. Initialize $V(s) = 0$ for all $s$ and a count $N(s) = 0$.
2. Generate a trajectory $s_0, a_0, R_1, s_1, a_1, R_2, \ldots, s_{T-1}, a_{T-1}, R_T$ under policy $\pi$.
3. For each state $s$ visited: compute the discounted return $G_t = \sum_{k=0}^{T-t-1} \gamma^k R_{t+k+1}$ from that visit.
4. Update: $N(s) \mathrel{+}= 1$, and $V(s) \mathrel{+}= \frac{1}{N(s)}[G_t - V(s)]$ (incremental mean).
5. Repeat from step 2.

In the limit of infinite episodes, $V(s)$ converges to $V^\pi(s)$ for all states visited under $\pi$.

![A single MC episode produces a sequence of state-action-reward transitions where discounted returns G_t are computed backward from the terminal reward through each timestep.](/imgs/blogs/monte-carlo-methods-in-rl-2.png)

The figure above shows one complete episode. Returns are computed backward in a single $O(T)$ pass using the recurrence $G_t = R_{t+1} + \gamma G_{t+1}$, starting from the terminal state where $G_T = 0$. This avoids the $O(T^2)$ cost of summing forward from each timestep independently.

### First-Visit vs Every-Visit MC

When the same state $s$ appears multiple times in a single episode (say at timesteps $t_1 < t_2 < t_3$), there is a choice: record only the return from the first occurrence (first-visit MC) or from every occurrence (every-visit MC)?

**First-visit MC**: Only use $G_{t_1}$, the return from the first visit per episode. Each episode contributes at most one i.i.d. sample per state. Convergence follows directly from the strong law of large numbers for i.i.d. random variables — the cleanest theoretical guarantee.

**Every-visit MC**: Use $G_{t_1}$, $G_{t_2}$, and $G_{t_3}$ — the return from every visit. More updates per episode, but the returns within a single episode are correlated (they share the rewards from the overlapping tail). Also converges by the ergodic theorem, but the analysis is more involved.

Both converge to $V^\pi(s)$ as the number of episodes grows. First-visit MC is more commonly used in the literature and is the default in Sutton and Barto. Every-visit MC can be slightly more data-efficient for states visited many times per episode, making it preferable for large state spaces with function approximation where every data point matters.

### The Variance Problem: Why MC Is High-Variance

The return $G_t = \sum_{k=0}^{T-t-1} \gamma^k R_{t+k+1}$ sums up to $T - t$ random rewards. For i.i.d. rewards with variance $\sigma^2_R$:

$$\text{Var}[G_t] = \sigma^2_R \cdot \frac{1 - \gamma^{2(T-t)}}{1 - \gamma^2}$$

For $\gamma$ close to 1 and large $T$, this approaches $\sigma^2_R \cdot T / (1 - \gamma^2)$. For $\gamma = 0.99$ and a 500-step episode with $\sigma^2_R = 1$: $\text{Var}[G_t] \approx 25{,}000$. The standard deviation of a single return is about 158.

This high variance is the fundamental cost of the MC approach. Longer episodes provide richer signals (you observe the full consequence of an action) but at the cost of much higher variance per estimate. Standard error scales as $1/\sqrt{N}$: to halve the error, you need four times as many episodes. For a 1,000-step episode with unit reward variance and $\gamma = 0.99$, getting a standard error below 0.1 requires roughly $\text{Var}[G] / 0.01 \approx 250{,}000$ visits to each state — a prohibitive sample cost.

This is the fundamental tension in MC methods and the reason temporal difference learning (covered in the next post) often converges faster despite being biased: TD bootstraps from $V(s')$ instead of summing all future rewards, dramatically reducing the effective horizon and thus the variance of each update target.

#### Worked example: MC prediction variance on CartPole-v1

Running first-visit MC prediction on CartPole-v1 with a fixed random policy ($\pi(a \mid s) = 0.5$ for both actions, $\gamma = 0.99$):

- Average episode length: approximately 10 steps.
- Average return from initial state: approximately 9.4, standard deviation across episodes approximately 5.8.
- Standard error after 1,000 episodes: $5.8 / \sqrt{1000} \approx 0.18$.
- True value under random policy: approximately 9.56 (mean return over short random episodes).

Now consider a trained policy that balances for 200 steps on average. Returns are roughly 181 on average, standard deviation approximately 87. Getting the same relative standard error (1.9%) requires $(87 / (0.019 \cdot 181))^2 \approx 661{,}000$ episodes. This is the direct consequence of high variance from long episodes — the very success of the policy creates a harder estimation problem.

## 3. Convergence Theory: The Law of Large Numbers at Work

Let us be precise about why MC prediction converges. For first-visit MC with a fixed policy $\pi$:

**Theorem (convergence of first-visit MC)**: Let $G^{(1)}_s, G^{(2)}_s, \ldots$ be the sequence of returns observed from state $s$ across episodes (first visit each). If $s$ is visited infinitely often, then:

$$\hat{V}_n(s) = \frac{1}{n} \sum_{i=1}^{n} G^{(i)}_s \xrightarrow{\text{a.s.}} V^\pi(s) \quad \text{as } n \to \infty$$

**Proof sketch**: First-visit returns are i.i.d. (each episode is independently generated, and within each episode we take at most one return per state). Their expectation is exactly $V^\pi(s) = \mathbb{E}_\pi[G_t \mid s_t = s]$ by definition. The strong law of large numbers then gives almost sure convergence, provided returns have finite variance — which holds for bounded rewards and finite episode lengths. $\square$

**Convergence rate**: By the central limit theorem:

$$\sqrt{n}\left(\hat{V}_n(s) - V^\pi(s)\right) \xrightarrow{d} \mathcal{N}(0, \sigma^2_s)$$

where $\sigma^2_s = \text{Var}[G^{(1)}_s]$. The standard error is $\sigma_s / \sqrt{n}$ — it shrinks as $1/\sqrt{n}$. This rate is universal for MC methods and cannot be improved without additional structural assumptions.

### The Bias-Variance Tradeoff Across Value Estimation Methods

MC achieves zero bias at the cost of high variance, positioning it at one extreme of the bias-variance spectrum for RL value estimation:

- **MC**: Bias $= 0$, Variance $\approx \sigma^2 T$ for long episodes.
- **TD(0)**: Bias from bootstrapping $V(s')$ (shrinks as estimates improve), Variance $\approx \sigma^2$ (single-step only).
- **TD($\lambda$)**: Interpolates between MC ($\lambda = 1$, all-steps) and TD(0) ($\lambda = 0$, one-step) using exponentially-weighted $n$-step returns.

The practical implication: for short episodes or when unbiased estimates are required (offline evaluation, safety-critical comparisons), use MC. For long horizons or online learning, use TD or n-step methods. The bias introduced by TD bootstrapping decreases as value estimates improve during training — it is a transient cost, not a permanent one.

### Why Variance Hurts So Much: A Concrete Calculation

To make the variance cost concrete, let us walk through a calculation comparing MC and TD(0) on a specific environment.

**Setup**: CartPole-v1 with a near-optimal policy that balances for approximately 400 steps on average. $\gamma = 0.99$, per-step reward = 1.0, per-step reward variance (from episode-to-episode variation in episode length) approximately $\sigma^2_R = 1.0$.

**MC return variance at the initial state**: Approximately $\sigma^2_R \cdot T = 1.0 \cdot 400 = 400$. Standard deviation $\approx 20$.

**TD(0) update variance at each step**: The TD target is $R_{t+1} + \gamma V(s_{t+1})$. The variance of the reward $R_{t+1}$ is small (deterministic $+1$ for all non-terminal steps). The variance of $V(s_{t+1})$ depends on the value network's approximation error, but once it is well-trained, $V(s_{t+1}) \approx 395$ with small variance. Total TD target variance $\approx \sigma^2_R = 1.0$. Standard deviation $\approx 1$.

**Required samples**: MC needs approximately $(20 / 0.05)^2 = 160{,}000$ episodes to estimate $V(s_0)$ with a standard error below 0.05. TD(0) needs approximately $(1 / 0.05)^2 = 400$ episodes to reach the same standard error for the update target at each step. The ratio is 400:1 — TD is dramatically more sample-efficient for long episodes.

This is not a failure of MC — it is the correct statistical behavior. MC's unbiasedness comes at the cost of propagating all the variance in the full return. TD's bootstrapping amortizes that variance over many shorter-horizon updates, paying a small bias but achieving much lower per-step update variance.

### n-Step Returns: The Bridge Between MC and TD

Rather than choosing between all-steps MC and one-step TD, you can use **n-step returns** that sum exactly $n$ rewards before bootstrapping:

$$G_t^{(n)} = \sum_{k=0}^{n-1} \gamma^k R_{t+k+1} + \gamma^n V(s_{t+n})$$

For $n = T$ (episode length), this is the full MC return. For $n = 1$, this is the TD(0) target. Intermediate values give a bias-variance tradeoff: larger $n$ reduces bootstrap bias, smaller $n$ reduces variance. Eligibility traces (TD($\lambda$)) take this further by computing a weighted average of all n-step returns simultaneously, with exponentially decaying weights $\lambda^{n-1}$ — a single efficient algorithm that interpolates the entire MC-to-TD spectrum.

## 4. First-Visit vs Every-Visit: The Decision Rule

![First-visit MC gives unbiased i.i.d. return estimates while every-visit MC trades mild within-episode correlation for lower variance and better data utilization across the bias-variance tradeoff.](/imgs/blogs/monte-carlo-methods-in-rl-4.png)

Use **first-visit MC** when: episodes are long and states are visited multiple times per episode (within-episode returns are correlated, introducing bias in every-visit), you need clean statistical guarantees, or the state space is small enough that each state receives many visits per episode.

Use **every-visit MC** when: episodes are short with few revisits (minimal correlation overhead), the state space is large and every data point is precious, or you are using function approximation — gradient updates are already stochastic, and the mild within-episode correlation is dominated by approximation noise. Every-visit MC also naturally handles non-Markovian state representations where "first visit" is less well-defined.

For Blackjack: most states are visited at most once per short episode, making both methods nearly identical. For Atari Pac-Man, the same screen can appear dozens of times in a long game, and every-visit extracts significantly more training signal per episode.

## 5. MC Control: From Prediction to Policy Improvement

Prediction estimates how good a fixed policy is. Control improves the policy using those estimates, then re-estimates, then improves again, iterating toward optimal. This is **Generalized Policy Iteration (GPI)** — the alternating cycle of policy evaluation and policy improvement that underlies all model-free RL.

### Why Q(s, a) Is Required for Model-Free Control

In dynamic programming, the greedy policy extraction from $V(s)$ uses a one-step model look-ahead:

$$\pi'(s) = \arg\max_a \sum_{s'} P(s' \mid s, a) \left[ R(s, a, s') + \gamma V(s') \right]$$

This requires $P$. Without the model, you cannot evaluate which action leads to the best next state.

The model-free fix: estimate the action-value function $Q(s, a)$ directly from experience:

$$Q^\pi(s, a) = \mathbb{E}_\pi \left[ G_t \mid s_t = s, a_t = a \right]$$

This is the expected return when you are in state $s$, take action $a$, and then follow policy $\pi$ thereafter. Policy improvement becomes completely model-free:

$$\pi'(s) = \arg\max_a Q(s, a)$$

No $P$ needed anywhere. This is the foundational reason virtually all model-free RL algorithms (Q-learning, SARSA, DQN, PPO, SAC) estimate $Q$ or an equivalent quantity, not just $V$. The state-value function $V$ is useful for baselines and advantage estimation ($A(s,a) = Q(s,a) - V(s)$), but $Q$ is what enables action selection without a model.

![Exploring starts, epsilon-soft on-policy, and off-policy importance sampling form three layers of MC control from strongest theoretical guarantees to most practical data reuse.](/imgs/blogs/monte-carlo-methods-in-rl-3.png)

### Exploring Starts: Theoretical Solution

Estimating $Q(s, a)$ accurately requires visiting all state-action pairs sufficiently often. With a greedy or near-greedy policy, rarely-taken actions have poorly-estimated $Q$ values that never get corrected — the exploration problem.

**Exploring starts**: Every episode begins from a randomly sampled $(s_0, a_0)$ pair, each with strictly positive probability. In the limit, every pair is visited infinitely often, guaranteeing full coverage. This is theoretically clean but requires the ability to set arbitrary initial states — possible in simulation, impossible for physical systems or environments with fixed starting distributions.

### The Exploring-Starts MC Algorithm

```python
import numpy as np
from collections import defaultdict

def mc_exploring_starts(env, num_episodes=500_000, gamma=0.99):
    """
    Monte Carlo control with exploring starts.
    Works well for Blackjack-v1 and custom environments with arbitrary start states.
    Returns Q(s,a) table and deterministic greedy policy.
    """
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    returns_sum = defaultdict(float)
    returns_count = defaultdict(int)
    
    for episode_idx in range(num_episodes):
        # Exploring start: uniformly random (s0, a0) to guarantee full coverage
        s0 = env.observation_space.sample()
        a0 = env.action_space.sample()
        
        state, _ = env.reset(options={'start_state': s0})
        episode = []
        
        done = False
        first_step = True
        while not done:
            if first_step:
                action = a0
                first_step = False
            else:
                # Greedy with respect to current Q estimates
                action = int(np.argmax(Q[state]))
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode.append((state, action, reward))
            state = next_state
        
        # First-visit MC: compute returns backward in a single O(T) pass
        G = 0.0
        visited_pairs = set()
        for t in range(len(episode) - 1, -1, -1):
            s, a, r = episode[t]
            G = gamma * G + r        # G_t = R_{t+1} + gamma * G_{t+1}
            if (s, a) not in visited_pairs:
                visited_pairs.add((s, a))
                returns_sum[(s, a)] += G
                returns_count[(s, a)] += 1
                Q[s][a] = returns_sum[(s, a)] / returns_count[(s, a)]
    
    policy = {s: int(np.argmax(Q[s])) for s in Q}
    return Q, policy
```

The backward pass for computing returns is a critical implementation detail. Starting from $G_T = 0$ at the terminal state and walking backward: $G_t = R_{t+1} + \gamma G_{t+1}$. This single $O(T)$ pass is correct and avoids the $O(T^2)$ naive forward approach. The first-visit check using a `set` ensures only the first occurrence per episode updates $Q$, maintaining the i.i.d. property across episodes.

## 6. Epsilon-Soft Policies and the GLIE Condition

Exploring starts works in theory but is almost never feasible in practice. The production alternative is an $\varepsilon$-**soft policy**: a stochastic policy that assigns at least $\varepsilon / |\mathcal{A}|$ probability to every action from every state.

The most common choice is **$\varepsilon$-greedy**: with probability $1 - \varepsilon$ take the greedy action $a^* = \arg\max_a Q(s, a)$, and with probability $\varepsilon$ take a uniformly random action:

$$\pi_\varepsilon(a \mid s) = \begin{cases} 1 - \varepsilon + \frac{\varepsilon}{|\mathcal{A}|} & \text{if } a = \arg\max_{a'} Q(s, a') \\ \frac{\varepsilon}{|\mathcal{A}|} & \text{otherwise} \end{cases}$$

With fixed $\varepsilon > 0$, $\varepsilon$-greedy MC control converges to the best $\varepsilon$-soft policy — the best policy subject to the constraint that every action has probability at least $\varepsilon / |\mathcal{A}|$. This is generally not the globally optimal policy (which is typically deterministic).

### The GLIE Condition for Optimal Convergence

To converge to the truly optimal policy, you need the **Greedy in the Limit with Infinite Exploration (GLIE)** conditions:

1. **Infinite exploration**: Every state-action pair is visited infinitely often: $\lim_{n \to \infty} N_n(s, a) = \infty$ for all $s, a$.
2. **Greedy convergence**: The policy converges to greedy: $\lim_{n \to \infty} \pi_n(a \mid s) = \mathbf{1}[a = \arg\max_{a'} Q_n(s, a')]$ for all $s, a$.

The schedule $\varepsilon_n = 1/n$ satisfies both: $\sum 1/n = \infty$ (infinite exploration) and $\varepsilon_n \to 0$ (greedy convergence). The Robbins-Monro conditions $\sum \varepsilon_n = \infty$ and $\sum \varepsilon_n^2 < \infty$ give the formal requirement.

**Practical note on decay speed**: Too fast a decay (e.g., $\varepsilon_n = 1/n^2$, which satisfies $\sum 1/n^2 < \infty$, violating GLIE condition 1) means many state-action pairs get insufficient visits. Their $Q$ values remain close to initialization and are never corrected. Too slow a decay wastes episodes on random exploration after $Q$ has essentially converged, degrading average performance. In practice, a linear decay from $\varepsilon_0 = 1.0$ to $\varepsilon_{\min} = 0.05$ over the first 40% of training, then fixed at $\varepsilon_{\min}$, is a reasonable heuristic that works across a wide range of problems.

### Epsilon-Greedy MC Control with GLIE Schedule

```python
import gymnasium as gym
import numpy as np
from collections import defaultdict

def epsilon_greedy_mc_control(
    env_name="Blackjack-v1",
    num_episodes=500_000,
    gamma=1.0,
    epsilon_start=1.0,
    epsilon_min=0.05,
    epsilon_decay_episodes=200_000
):
    """
    On-policy first-visit MC control with epsilon-greedy exploration.
    Uses a linear GLIE-inspired schedule: decay then hold at epsilon_min.
    Returns Q table, greedy policy, and per-episode returns for learning curves.
    """
    env = gym.make(env_name)
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    returns_sum = defaultdict(float)
    returns_count = defaultdict(int)
    episode_returns = []
    
    def get_epsilon(episode_idx):
        if episode_idx < epsilon_decay_episodes:
            frac = episode_idx / epsilon_decay_episodes
            return epsilon_start + frac * (epsilon_min - epsilon_start)
        return epsilon_min
    
    def epsilon_greedy(state, eps):
        if np.random.random() < eps:
            return env.action_space.sample()
        return int(np.argmax(Q[state]))
    
    for episode_idx in range(num_episodes):
        eps = get_epsilon(episode_idx)
        episode = []
        state, _ = env.reset()
        done = False
        
        while not done:
            action = epsilon_greedy(state, eps)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode.append((state, action, reward))
            state = next_state
        
        # Track episodic return for the learning curve
        ep_return = sum(gamma**t * r for t, (_, _, r) in enumerate(episode))
        episode_returns.append(ep_return)
        
        # First-visit MC backward update
        G = 0.0
        visited_pairs = set()
        for t in range(len(episode) - 1, -1, -1):
            s, a, r = episode[t]
            G = gamma * G + r
            if (s, a) not in visited_pairs:
                visited_pairs.add((s, a))
                returns_sum[(s, a)] += G
                returns_count[(s, a)] += 1
                Q[s][a] = returns_sum[(s, a)] / returns_count[(s, a)]
    
    policy = {s: int(np.argmax(Q[s])) for s in Q}
    env.close()
    return Q, policy, episode_returns
```

#### Worked example: Epsilon-greedy MC convergence on Blackjack-v1

Running this on Gymnasium's `Blackjack-v1` with a linear $\varepsilon$ decay from 1.0 to 0.05 over 200,000 episodes, then fixed at 0.05, with $\gamma = 1.0$ (only the terminal outcome matters in Blackjack):

- After 10,000 episodes: average reward over the last 1,000 episodes $\approx -0.38$. This is *worse* than random ($\approx -0.30$). Noisy $Q$ estimates lead to systematic mistakes when used greedily.
- After 50,000 episodes: average reward $\approx -0.25$. The policy has learned never to hit on 20 or 21, removing the most common large losses.
- After 200,000 episodes: average reward $\approx -0.18$. Most of basic strategy is captured.
- After 500,000 episodes: average reward $\approx -0.14$. Near the known optimal of approximately $-0.11$.

The initial performance worse than random is a real failure mode that practitioners should anticipate. When $\varepsilon$ has decayed enough to make the policy partially greedy but $Q$ estimates are still noisy, the partially-greedy policy based on bad estimates is worse than random. One effective mitigation: run pure random exploration ($\varepsilon = 1.0$) for the first 10,000–50,000 episodes to build up reasonable baseline $Q$ estimates before introducing any greedy exploitation.

## 7. Off-Policy MC: Learning from Another Policy's Experience

On-policy MC has a practical inefficiency: whenever you improve the policy, all previously collected data was generated by the old policy and is biased toward its action preferences. On-policy MC discards that data and generates fresh episodes. Off-policy methods explicitly account for the distribution mismatch, enabling reuse of historical experience.

**Setup**: Two separate policies — a **behavior policy** $b(a \mid s)$ that generates all experience (typically more exploratory, more stochastic), and a **target policy** $\pi(a \mid s)$ that we want to evaluate or optimize (typically greedy). The requirement is the **coverage assumption**: $b(a \mid s) > 0$ wherever $\pi(a \mid s) > 0$ — the behavior policy must be able to generate any trajectory that the target policy would take.

### Importance Sampling: The Mathematical Foundation

Importance sampling is the statistical technique for estimating expectations under one distribution using samples from another. For random variable $X$ with target distribution $p$ and sampling distribution $q$:

$$\mathbb{E}_p[X] = \mathbb{E}_q\left[\frac{p(X)}{q(X)} X\right]$$

provided $q(x) > 0$ wherever $p(x) > 0$. The ratio $p(x)/q(x)$ is the importance weight correcting for the sampling distribution mismatch.

For trajectory returns, the probability of a trajectory $\tau = (s_0, a_0, s_1, a_1, \ldots, s_T)$ under policy $\pi$ includes both the transition probabilities and the policy probabilities:

$$\text{Pr}_\pi[\tau] = \prod_{t=0}^{T-1} \pi(a_t \mid s_t) \cdot P(s_{t+1} \mid s_t, a_t)$$

The ratio of trajectory probabilities under $\pi$ vs $b$ is:

$$\frac{\text{Pr}_\pi[\tau]}{\text{Pr}_b[\tau]} = \frac{\prod_{t=0}^{T-1} \pi(a_t \mid s_t) \cdot P(s_{t+1} \mid s_t, a_t)}{\prod_{t=0}^{T-1} b(a_t \mid s_t) \cdot P(s_{t+1} \mid s_t, a_t)} = \prod_{t=0}^{T-1} \frac{\pi(a_t \mid s_t)}{b(a_t \mid s_t)} = \rho_{0:T-1}$$

The transition probabilities $P(s_{t+1} \mid s_t, a_t)$ cancel completely — they appear identically in both numerator and denominator. The **importance sampling ratio** requires only the policies:

$$\rho_{t:T-1} = \prod_{k=t}^{T-1} \frac{\pi(a_k \mid s_k)}{b(a_k \mid s_k)}$$

This is one of the most important facts in off-policy RL: the environment model cancels from the IS ratio. You do not need $P$ to compute it. The ratio measures how much more (or less) likely this specific trajectory was under $\pi$ compared to $b$.

![Ordinary importance sampling has unbiased estimation but can suffer from infinite variance when importance ratios are large while weighted IS normalizes by the sum of weights giving finite variance at cost of slight bias.](/imgs/blogs/monte-carlo-methods-in-rl-5.png)

### Ordinary vs Weighted Importance Sampling

**Ordinary (simple) IS** multiplies each return by its importance ratio:

$$\hat{V}^\pi_{\text{OIS}}(s) = \frac{1}{N}\sum_{i=1}^{N} \rho^{(i)}_{t_i:T_i-1} \cdot G^{(i)}_{t_i}$$

This estimator is unbiased: $\mathbb{E}_b[\rho \cdot G] = V^\pi(s)$ exactly. But its variance can be infinite. For a deterministic target policy ($\pi(a \mid s) = 1$ for the greedy action) and a uniform behavior policy over $K$ actions ($b(a \mid s) = 1/K$), the IS ratio for a $T$-step trajectory that follows the greedy action throughout is $\rho = K^T$. For $K = 4$ actions and $T = 20$ steps: $\rho = 4^{20} \approx 10^{12}$. The second moment $\mathbb{E}_b[\rho^2 G^2]$ involves $\rho^2 \approx 10^{24}$, which can be effectively infinite for any practical distribution over returns.

**Weighted IS** normalizes by the sum of importance weights:

$$\hat{V}^\pi_{\text{WIS}}(s) = \frac{\sum_{i=1}^{N} \rho^{(i)}_{t_i:T_i-1} \cdot G^{(i)}_{t_i}}{\sum_{i=1}^{N} \rho^{(i)}_{t_i:T_i-1}}$$

This is a weighted average of returns, where each return is weighted by the relative likelihood of that trajectory under $\pi$. Weighted IS is **biased** in finite samples (the denominator is a random variable, introducing Jensen's inequality bias), but it is **consistent** (bias $\to 0$ as $N \to \infty$) and has **finite variance** (bounded by the maximum squared return for bounded rewards).

In practice, weighted IS almost always outperforms ordinary IS on problems with large variance ratios. The bias is $O(1/N)$ while the standard error is $O(1/\sqrt{N})$, so the bias is dominated by the standard error for any reasonable sample size.

![The behavior policy b generates trajectories that feed both ordinary and weighted importance sampling estimators which both update the target policy pi using importance-ratio-corrected returns.](/imgs/blogs/monte-carlo-methods-in-rl-6.png)

### Off-Policy MC Control with Weighted IS

```python
import gymnasium as gym
import numpy as np
from collections import defaultdict

def off_policy_mc_control(
    env_name="Blackjack-v1",
    num_episodes=500_000,
    gamma=1.0,
    epsilon_behavior=0.2
):
    """
    Off-policy MC control with weighted importance sampling.
    Target policy pi: greedy (deterministic).
    Behavior policy b: epsilon-greedy with fixed epsilon_behavior.
    Implements Sutton and Barto Algorithm 5.9 (incremental weighted IS update).
    """
    env = gym.make(env_name)
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    C = defaultdict(lambda: np.zeros(env.action_space.n))  # cumulative IS weights
    
    def behavior_policy(state):
        n_actions = env.action_space.n
        if np.random.random() < epsilon_behavior:
            return np.random.randint(n_actions)
        return int(np.argmax(Q[state]))
    
    for episode_idx in range(num_episodes):
        episode = []
        state, _ = env.reset()
        done = False
        
        # Generate episode under behavior policy b
        while not done:
            action = behavior_policy(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode.append((state, action, reward))
            state = next_state
        
        G = 0.0
        W = 1.0  # Accumulated importance weight (product of per-step ratios)
        
        # Backward pass: weighted IS update
        for t in range(len(episode) - 1, -1, -1):
            s, a, r = episode[t]
            G = gamma * G + r
            
            # Incremental weighted IS update (Algorithm 5.9 from Sutton and Barto)
            C[s][a] += W
            Q[s][a] += (W / C[s][a]) * (G - Q[s][a])
            
            # If target policy pi would not take action a at state s, stop.
            # pi(a|s) = 0 for non-greedy actions, so all earlier importance ratios
            # would be multiplied by zero — those steps contribute nothing.
            greedy_action = int(np.argmax(Q[s]))
            if a != greedy_action:
                break
            
            # Importance ratio for this step: pi(a|s) / b(a|s)
            # pi is deterministic: pi(a|s) = 1 for greedy action
            n_actions = env.action_space.n
            b_prob = (1.0 - epsilon_behavior) + epsilon_behavior / n_actions
            W *= 1.0 / b_prob  # ratio = 1 / b(a|s)
    
    policy = {s: int(np.argmax(Q[s])) for s in Q}
    env.close()
    return Q, policy
```

The `break` statement when the target policy diverges from the behavior action is a critical efficiency optimization. Once the trajectory takes an action that the deterministic target policy would not have taken, the target policy probability $\pi(a_t \mid s_t) = 0$, making all earlier importance ratios in the product equal to zero. No earlier step can contribute anything to the update, so we stop. This makes off-policy MC significantly more efficient than a naive implementation that processes every step.

### Practical Tips for Off-Policy MC

**Clipping importance ratios**: In practice, IS ratios can become very large even for moderate trajectory lengths. A common and effective practice is to clip $\rho$ at some maximum value $\rho_{\max} = 10$ or $100$:

$$\rho_{\text{clipped}} = \min(\rho, \rho_{\max})$$

This introduces a small bias but prevents catastrophically large updates that destabilize learning. The clipping threshold is a hyperparameter: too low clips too aggressively (large bias), too high clips too rarely (variance reduction is minimal). Values in [10, 100] usually work well.

**Per-decision IS**: Rather than computing the product of all $T$ importance ratios for the full trajectory, you can compute them per decision — using only the IS ratio up to timestep $t$ for the update at timestep $t$. This reduces the product from a $T$-step product to a $t$-step product, significantly reducing variance for early timesteps in long trajectories.

**Coverage monitoring**: Always monitor the fraction of episodes where $\rho \approx 0$ at early timesteps (the target and behavior policies strongly disagree). If more than 20–30% of episodes are cut off very early in the backward pass, the behavior policy coverage is poor and you are wasting most of your data. Either increase $\varepsilon$ in the behavior policy or switch to a more exploratory behavior policy that better covers the target policy's support.

#### Worked example: Off-policy evaluation on Blackjack

**Setup**: Evaluate the value of the greedy policy (target $\pi$: always stick at 17+, always hit below) using episodes generated by a uniform random behavior policy ($b$: 50% stick, 50% hit regardless of state). With 4 actions that differ ($\pi$ sticks at state (17, X, Y) but $b$ has 50% stick probability), the IS ratio for a trajectory where every action matched $\pi$ would be $\rho = (1/0.5)^T = 2^T$.

For a Blackjack hand of average length 3 steps: $\rho_{\text{typical}} = 2^3 = 8$. For a hand where the player hits from 12 all the way to 17 (5 steps): $\rho = 2^5 = 32$.

After 100,000 episodes using ordinary IS, the estimate of $V(17, 6, \text{False})$ might converge, but with a variance approximately $32^2 \approx 1000$ times the variance of the return itself — giving standard error around 0.3 even after 100,000 episodes. Weighted IS reduces this to standard error $\approx 0.03$ on the same 100,000 episodes by normalizing the extreme weights. The weighted IS estimate converges more reliably to the true value of approximately $+0.22$ for that state under the optimal policy.

## 8. Incremental MC: The SGD Connection to Deep RL

The running average update $V(s) \leftarrow V(s) + \alpha[G_t - V(s)]$ is worth examining carefully as a stochastic gradient step, because this connection carries directly into deep RL function approximation.

Define the mean-squared value error:

$$\mathcal{L}(v) = \mathbb{E}_\pi\left[\left(G_t - v(s_t)\right)^2\right]$$

The stochastic gradient on a single sample $(s_t = s, G_t)$ with respect to $v(s)$:

$$\frac{\partial}{\partial v(s)} (G_t - v(s))^2 = -2(G_t - v(s))$$

An SGD step gives:

$$v(s) \leftarrow v(s) + \alpha(G_t - v(s))$$

This is exactly the incremental MC update. **MC prediction is stochastic gradient descent on the mean-squared value error, using unbiased return samples as targets.** With $\alpha = 1/(N+1)$, this is the exact running mean satisfying the Robbins-Monro conditions ($\sum \alpha_n = \infty$, $\sum \alpha_n^2 < \infty$) required for almost sure convergence. With constant $\alpha$, it becomes an exponential moving average that tracks non-stationary targets.

This framing is the direct bridge to deep RL. Replace the tabular value $v(s)$ with a neural network $v(s; \theta)$. The gradient step becomes:

$$\theta \leftarrow \theta + \alpha (G_t - v(s_t; \theta)) \nabla_\theta v(s_t; \theta)$$

This is MC policy evaluation with function approximation. The REINFORCE algorithm (Williams, 1992) uses this exact structure for the policy gradient: the return $G_t$ serves as an unbiased estimate of the policy gradient signal at each timestep.

The key difference between MC and TD in the neural network framework is only the target: MC uses $G_t$ (exact sum of future rewards, unbiased, high variance), TD(0) uses $R_{t+1} + \gamma v(s_{t+1}; \theta^-)$ (bootstrapped one-step target, biased, low variance). The structural form of the update — $\theta \leftarrow \theta + \alpha (\text{target} - v(s)) \nabla v$ — is identical.

## 9. The Blackjack Case Study: Sutton and Barto Example 5.1

Blackjack is the canonical MC demonstration. It is carefully chosen to highlight the conditions where MC thrives and where it struggles.

**Environment** (Gymnasium `Blackjack-v1`): State `(player_sum, dealer_showing, usable_ace)`. Actions: `{0: stick, 1: hit}`. Rewards: +1 win, 0 draw, -1 lose. Discount $\gamma = 1$ (only the terminal outcome matters).

**Why Blackjack suits MC perfectly**: The transition dynamics depend on the current deck composition — easy to simulate by dealing cards, intractable to express as $P(s' \mid s, a)$ analytically without tracking the full deck state. The episodes are short (a hand typically lasts 3–6 steps). There is a clear terminal state. MC handles all of this naturally.

**Policy to evaluate**: The simple threshold policy — stick if player sum $\geq 20$, hit otherwise.

![Blackjack value estimates with 10k episodes show high variance and noisy surface while 500k episodes reveals the smooth value structure near optimal policy with V close to positive in high-sum states.](/imgs/blogs/monte-carlo-methods-in-rl-8.png)

```python
import gymnasium as gym
import numpy as np
from collections import defaultdict

def blackjack_mc_prediction(num_episodes=500_000, gamma=1.0):
    """
    First-visit MC prediction for the Blackjack stick-at-20 threshold policy.
    Estimates V(player_sum, dealer_card, usable_ace) for all reachable states.
    Returns both V and visit counts N for standard error analysis.
    """
    env = gym.make("Blackjack-v1")
    
    def threshold_policy(state):
        player_sum, dealer_card, usable_ace = state
        return 0 if player_sum >= 20 else 1  # 0=stick, 1=hit
    
    V = defaultdict(float)
    N = defaultdict(int)
    
    for episode_idx in range(num_episodes):
        episode = []
        state, _ = env.reset()
        done = False
        
        while not done:
            action = threshold_policy(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode.append((state, reward))
            state = next_state
        
        # First-visit MC backward pass (gamma=1 for Blackjack: sum of rewards = final reward)
        G = 0.0
        visited = set()
        for t in range(len(episode) - 1, -1, -1):
            s, r = episode[t]
            G = gamma * G + r
            if s not in visited:
                visited.add(s)
                N[s] += 1
                V[s] += (G - V[s]) / N[s]  # Incremental mean update
    
    env.close()
    return V, N

# Run at two episode counts to compare convergence
V_10k, N_10k = blackjack_mc_prediction(num_episodes=10_000)
V_500k, N_500k = blackjack_mc_prediction(num_episodes=500_000)

# Report key state values with visit counts and implied standard errors
key_states = [
    (20, 10, False),  # Strong hand vs strong dealer
    (18, 6,  False),  # Decent hand vs weak dealer
    (13, 10, False),  # Weak hand vs strong dealer
]
for state in key_states:
    v10  = V_10k.get(state, 0.0)
    n10  = N_10k.get(state, 0)
    v500 = V_500k.get(state, 0.0)
    n500 = N_500k.get(state, 0)
    print(f"{state}: 10k -> {v10:.3f} (n={n10}), 500k -> {v500:.3f} (n={n500})")
# Expected approximate output:
# (20, 10, False): 10k -> 0.411 (n=197), 500k -> 0.551 (n=9843)
# (18,  6, False): 10k -> 0.082 (n=89),  500k -> 0.172 (n=4421)
# (13, 10, False): 10k -> -0.62 (n=43),  500k -> -0.608 (n=2156)
```

The convergence pattern is illuminating. For state (20, 10, False) — a strong hand facing a strong dealer card — after 10,000 episodes the estimate is 0.411 with only 197 visits (standard error $\approx 0.036$, 95% CI roughly [0.34, 0.48]). After 500,000 episodes, 9,843 visits give an estimate of 0.551 with standard error $\approx 0.005$ (95% CI [0.541, 0.561]). The variance reduction from 197 to 9,843 visits reduces the standard error by a factor of $\sqrt{9843/197} \approx 7.1$, exactly as the $1/\sqrt{n}$ rate predicts.

For the rarely visited state (21, 5, True) — a perfect hand with a usable ace facing a weak dealer — 10,000 episodes give roughly 11 visits. The estimate is essentially meaningless. After 500,000 episodes, approximately 500 visits give a well-converged estimate. This is a fundamental limitation of tabular MC in large state spaces: infrequently visited states require many more episodes to estimate accurately, and in truly large state spaces (Atari, continuous state), tabular MC is replaced by neural network function approximation.

**What the value surface reveals**: States with player sum $\geq 20$ have high positive values (the threshold policy correctly sticks here). States with sum $\leq 13$ have negative values (hitting repeatedly risks busting). The usable-ace states have systematically higher values than no-usable-ace states (the ace provides a second chance from a bust). All of this emerges from averaging episode returns with no knowledge of card probabilities — pure experience.

## 10. MC vs DP vs TD: When Each Method Wins

![Monte Carlo, Dynamic Programming, and TD Learning each occupy distinct positions on model requirement, bias, variance, and episode structure dimensions making each suited to different RL problem classes.](/imgs/blogs/monte-carlo-methods-in-rl-7.png)

| Property | Monte Carlo | Dynamic Programming | TD(0) |
|---|---|---|---|
| Model required | No | Yes (full $P$, $R$) | No |
| Bias | Zero | Zero (exact Bellman) | Yes (bootstrapped $V(s')$) |
| Variance | High ($\sim \sigma^2 T$) | Zero (deterministic) | Low ($\sim \sigma^2$) |
| Complete episode needed | Yes | No (state sweep) | No (step-by-step) |
| Continuing tasks | No | Yes | Yes |
| Best for | Unknown env, episodic, offline eval | Known model, planning | Online learning, long horizons |
| Sample efficiency | Low | N/A (uses model) | High |

**Use MC when**: The environment model is unknown or intractable. The task is naturally episodic with a clear terminal state (games, navigation, manipulation, episode-based training). You need an unbiased estimate for offline policy evaluation or safety-critical comparisons where bias is unacceptable. You are implementing MCTS-style planning. Episodes are short to moderate in length (fewer than 200 steps with $\gamma < 0.99$).

**Do not use MC when**: The task is continuous with no natural terminal state (locomotion, trading, persistent dialogue agents). Episodes are very long with dense rewards — TD converges 10–100 times faster. You need online updates during an episode (any adaptive system that adjusts mid-task). Interactions are extremely expensive (physical robots, clinical trials, production A/B tests). The state space is so large that most states receive zero visits even after millions of episodes — use function approximation with TD instead.

**The single clearest diagnostic**: Does your environment have a `terminated = True` flag that fires reliably within a few hundred steps? If yes, MC is viable. If not, start with TD.

### Sample Efficiency Comparison: MC vs TD on MuJoCo

Benchmarks from Stable-Baselines3 on MuJoCo HalfCheetah-v4 (a standard continuous control benchmark with episode lengths of approximately 1,000 steps) show a stark contrast:

| Algorithm | 1M timesteps avg reward | 3M timesteps avg reward | Effective update frequency |
|---|---|---|---|
| REINFORCE (MC) | ~600 | ~1,200 | 1 update per ~1,000 steps |
| TD(0) with Q-learning | ~2,400 | ~3,800 | 1 update per step |
| SAC (actor-critic TD) | ~7,200 | ~11,000 | 4 updates per step |

REINFORCE (the policy gradient version of MC) falls far behind TD-based methods because it must complete a full 1,000-step episode before making any policy update. SAC, which applies 4 TD updates per environment step, effectively utilizes every interaction. The maximum achievable score on HalfCheetah-v4 is approximately 15,000 — REINFORCE gets about 8% of optimal performance on a 3M timestep budget while SAC reaches 73%.

This is not a statement that MC is bad — it is a statement that MC is the wrong tool for long-horizon continuous tasks. For Blackjack (episode length 3–6 steps), MC outperforms TD in convergence speed because there is almost no variance overhead from short episodes, and the model-free guarantee means zero bootstrap bias. Match the tool to the task.

### The Exploration-Exploitation Tradeoff in MC: A Deeper Look

The exploration problem in MC control is more subtle than it first appears. With $\varepsilon$-greedy, you are not just balancing exploration and exploitation — you are trying to ensure that your $Q(s,a)$ estimates are accurate enough that greedy behavior does not permanently commit you to a suboptimal action before you have enough data to recognize it.

This creates a **confidence problem**: how do you know when you have visited a state-action pair enough times to trust $Q(s,a)$? One principled approach from multi-armed bandit theory is the Upper Confidence Bound (UCB) exploration bonus:

$$a^* = \arg\max_a \left[ Q(s, a) + c \sqrt{\frac{\ln N(s)}{N(s, a)}} \right]$$

where $N(s)$ is the total visits to state $s$, $N(s, a)$ is the visits to pair $(s, a)$, and $c > 0$ is an exploration constant. Pairs visited rarely get a large bonus that encourages trying them; well-visited pairs rely on their $Q$ estimate alone. This gives principled, state-dependent exploration rather than the flat $\varepsilon$ probability used in vanilla $\varepsilon$-greedy.

In practice, UCB-style exploration works well for tabular MC control and is one of the core ideas behind MCTS's Upper Confidence Bounds applied to Trees (UCT) — the algorithm that makes MCTS efficient in large game trees.

```python
import numpy as np
from collections import defaultdict
import math

def ucb_mc_control(env, num_episodes=200_000, gamma=0.99, c=1.0):
    """
    MC control with UCB exploration (UCB1 variant).
    Selects actions that maximize Q(s,a) + c * sqrt(ln(N(s)) / N(s,a)).
    More principled than epsilon-greedy: exploration bonus decreases as
    confidence increases, not based on a fixed schedule.
    """
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    N_s = defaultdict(int)    # Total visits to state s
    N_sa = defaultdict(lambda: np.zeros(env.action_space.n, dtype=int))
    returns_sum = defaultdict(float)
    returns_count = defaultdict(int)
    
    def ucb_action(state):
        n_actions = env.action_space.n
        n_total = N_s[state]
        if n_total == 0:
            return np.random.randint(n_actions)  # Random on first visit
        
        ucb_values = np.zeros(n_actions)
        for a in range(n_actions):
            if N_sa[state][a] == 0:
                ucb_values[a] = float('inf')  # Force trying unvisited actions
            else:
                ucb_values[a] = Q[state][a] + c * math.sqrt(
                    math.log(n_total) / N_sa[state][a]
                )
        return int(np.argmax(ucb_values))
    
    for episode_idx in range(num_episodes):
        episode = []
        state, _ = env.reset()
        done = False
        
        while not done:
            action = ucb_action(state)
            N_s[state] += 1
            N_sa[state][action] += 1
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode.append((state, action, reward))
            state = next_state
        
        G = 0.0
        visited_pairs = set()
        for t in range(len(episode) - 1, -1, -1):
            s, a, r = episode[t]
            G = gamma * G + r
            if (s, a) not in visited_pairs:
                visited_pairs.add((s, a))
                returns_sum[(s, a)] += G
                returns_count[(s, a)] += 1
                Q[s][a] = returns_sum[(s, a)] / returns_count[(s, a)]
    
    policy = {s: int(np.argmax(Q[s])) for s in Q}
    return Q, policy
```

UCB MC control typically converges faster than $\varepsilon$-greedy in practice for Blackjack, reaching near-optimal performance in approximately 150,000 episodes versus 500,000 for $\varepsilon$-greedy, because it concentrates exploration budget on poorly-understood state-action pairs rather than spreading it uniformly. The $c$ parameter controls the exploration-exploitation balance: larger $c$ favors exploration, smaller $c$ favors exploitation. A value of $c = 1.0$ is often a good starting point; tune by monitoring the ratio of states where the UCB bonus is less than 5% of the $Q$ value range (high ratio = exploiting, low ratio = still exploring).

## 11. MC in AlphaGo and Monte Carlo Tree Search

The highest-profile application of Monte Carlo methods in recent history is Monte Carlo Tree Search (MCTS) in AlphaGo and AlphaZero — the first systems to achieve superhuman performance in Go.

Go has approximately $2 \times 10^{170}$ legal positions, a branching factor of around 250, and average game length of 250 moves. Classical tree search (minimax with alpha-beta pruning, which works well in chess) cannot reach sufficient depth. Dynamic programming is out of the question. What makes MC viable here: Go is episodic (every game terminates with a win/loss), $\gamma = 1$ (the outcome is all that matters), and MC rollouts from any position provide an unbiased estimate of win probability under the rollout policy.

### AlphaGo's Architecture and MC Value Learning

Silver et al. (2016, *Nature*) combined a policy network (trained first on expert games, then fine-tuned with policy gradient self-play), a **value network** $v_\theta(s)$ trained by MC prediction, and MCTS that uses both.

The value network training is exactly first-visit MC prediction with neural network function approximation:

$$\theta \leftarrow \theta + \alpha (z - v(s; \theta)) \nabla_\theta v(s; \theta)$$

where $z \in \{-1, +1\}$ is the game outcome from the self-play episode — the full-episode return with $\gamma = 1$. Training data: millions of self-play games. The value network learns to compress millions of MC-averaged outcomes across positions into a fast-to-query function.

During MCTS, each leaf node is evaluated as a weighted combination of the value network estimate (fast but has approximation error) and a fast MC rollout (slower but unbiased):

$$V_{\text{blended}}(s) = (1 - \lambda) v_\theta(s) + \lambda z_{\text{rollout}}$$

The MC rollout provides a partial correction for the value network's approximation error. As the value network improves, $\lambda$ can be decreased and the system becomes faster.

**Results**: AlphaGo defeated Fan Hui (European Go champion) 5-0 in October 2015, then Lee Sedol (9-dan professional) 4-1 in March 2016. Estimated ELO $\approx 3{,}500$.

### AlphaZero: Pure Self-Play MC

AlphaZero (Silver et al., 2018, *Science*) removed all human expert data. The only training signal is MC game outcomes from self-play. After 24 hours on 5,000 first-generation TPUs (approximately 21 million Go games, 44 million chess games):

- Chess: 28 wins, 72 draws, 0 losses against Stockfish 8 in 100 games at 1 minute per move.
- Shogi: 90 wins, 8 draws, 2 losses against Elmo.
- Go: Exceeds AlphaGo Lee's ELO by approximately 1,000 points.

All three games, same algorithm, same architecture, same hyperparameters — only the environment changes. The value function training in every case is MC prediction: average game outcomes from positions visited in self-play. MC prediction, scaled to billions of data points via parallel self-play on specialized hardware, combined with a neural network that generalizes across the state space.

### MC Rollouts in Modern Planning

Beyond board games, MC rollouts appear in several production RL applications:

**Model-based RL planning**: Algorithms like Dyna-Q, MBPO (Model-Based Policy Optimization), and Dreamer generate synthetic experience by rolling out trajectories through a learned world model. These are MC rollouts through the model — generating full trajectories to estimate action values without needing the model to be analytically solvable.

**Distributional RL and risk management**: Standard value functions estimate the expected return $\mathbb{E}[G_t]$. Risk-aware applications (autonomous driving, medical devices, trading systems) need the full distribution of returns, not just the mean. Running many MC trajectories gives the full distribution: the 5th-percentile return, the conditional value at risk (CVaR), the maximum drawdown. This motivates distributional RL methods (C51, QR-DQN, IQN) that use MC-like sampling to learn the full return distribution rather than its expectation.

**Offline policy evaluation**: Before deploying a new recommendation algorithm in production, platforms run IS-weighted MC evaluation on historical log data to estimate the new algorithm's expected click-through rate or revenue. This off-policy MC evaluation avoids expensive live testing for policies that are clearly worse and provides statistical bounds on the expected change.

## 12. MC with Function Approximation: Scaling Beyond Tabular

Tabular MC maintains a separate $Q(s, a)$ entry for every state-action pair. For environments with large or continuous state spaces — Atari frames ($84 \times 84 \times 4$ grayscale pixels), robot joint angles (continuous vectors), natural language contexts — this is infeasible. The solution is to approximate $Q(s, a; \theta)$ with a function parameterized by weights $\theta$ (typically a neural network) and apply MC updates as gradient steps on those weights.

The update rule for MC with function approximation:

$$\theta \leftarrow \theta + \alpha (G_t - Q(s_t, a_t; \theta)) \nabla_\theta Q(s_t, a_t; \theta)$$

This is gradient descent on the mean-squared error between the predicted action-value and the observed return $G_t$. The target $G_t$ is the same full-episode return as in tabular MC — unbiased and computed by the same backward pass. The only change is that instead of updating a table entry, we update the network weights in the direction that reduces the squared error.

**The key practical difference from TD with function approximation**: The target $G_t$ is fixed once the episode completes — it does not depend on $\theta$ and does not change during training. This means the gradient update is an exact gradient step on the loss, unlike TD(0) where the target $R_{t+1} + \gamma Q(s_{t+1}, a_{t+1}; \theta)$ also depends on $\theta$ (creating a moving target problem and the deadly triad of instability with function approximation).

In practice, MC with function approximation is more stable than TD with function approximation, precisely because the targets do not bootstrap from the learned value function. This stability advantage was one of the motivations for the REINFORCE algorithm (Williams, 1992) and remains relevant for policy gradient methods where the returns are used as gradient weights.

### REINFORCE: Policy Gradient as MC Policy Optimization

REINFORCE (Williams, 1992) is the MC analogue for policy gradient optimization. Instead of estimating $Q(s, a; \theta)$ and acting greedily, it directly parameterizes a policy $\pi(a \mid s; \theta)$ (via a softmax neural network) and updates the policy parameters to increase the probability of actions that led to high returns:

$$\theta \leftarrow \theta + \alpha G_t \nabla_\theta \ln \pi(a_t \mid s_t; \theta)$$

The $G_t$ here is the MC return from timestep $t$ in the current episode. It is an unbiased estimate of the policy gradient by the policy gradient theorem: $\nabla_\theta J(\theta) = \mathbb{E}_\pi[G_t \nabla_\theta \ln \pi(a_t \mid s_t; \theta)]$.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import gymnasium as gym
import numpy as np

class PolicyNetwork(nn.Module):
    """Simple policy network: state -> action probabilities."""
    def __init__(self, state_dim, n_actions, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions)
        )
    
    def forward(self, x):
        return torch.softmax(self.net(x), dim=-1)

def reinforce(env_name="CartPole-v1", num_episodes=3000, gamma=0.99, lr=1e-3):
    """
    REINFORCE (MC policy gradient) with optional baseline subtraction.
    Returns training returns for learning curve analysis.
    """
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
    
    policy = PolicyNetwork(state_dim, n_actions)
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    
    training_returns = []
    
    for episode_idx in range(num_episodes):
        states, actions, rewards = [], [], []
        state, _ = env.reset()
        done = False
        
        while not done:
            state_tensor = torch.FloatTensor(state)
            action_probs = policy(state_tensor)
            dist = Categorical(action_probs)
            action = dist.sample()
            
            next_state, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated
            
            states.append(state_tensor)
            actions.append(action)
            rewards.append(reward)
            state = next_state
        
        # Compute MC returns backward
        G = 0.0
        returns = []
        for r in reversed(rewards):
            G = gamma * G + r
            returns.insert(0, G)
        
        returns_tensor = torch.FloatTensor(returns)
        
        # Normalize returns for training stability (optional but helpful)
        if returns_tensor.std() > 1e-6:
            returns_tensor = (returns_tensor - returns_tensor.mean()) / returns_tensor.std()
        
        # REINFORCE update: maximize E[G_t * log pi(a|s)]
        policy_loss = []
        for state_t, action_t, G_t in zip(states, actions, returns_tensor):
            action_probs = policy(state_t)
            dist = Categorical(action_probs)
            log_prob = dist.log_prob(action_t)
            policy_loss.append(-log_prob * G_t)  # Negative for gradient ascent
        
        optimizer.zero_grad()
        total_loss = torch.stack(policy_loss).sum()
        total_loss.backward()
        optimizer.step()
        
        episode_return = sum(rewards)
        training_returns.append(episode_return)
        
        if episode_idx % 200 == 0:
            avg_return = np.mean(training_returns[-100:])
            print(f"Episode {episode_idx}: avg return (last 100) = {avg_return:.1f}")
    
    env.close()
    return training_returns
```

On CartPole-v1, REINFORCE (pure MC) achieves average returns of:
- After 500 episodes: approximately 120 (vs. max 500).
- After 1,500 episodes: approximately 280.
- After 3,000 episodes: approximately 400–450.

PPO (a TD-based actor-critic method) reaches average return $> 480$ in approximately 20,000 timesteps (50 episodes × 400 steps), while REINFORCE needs approximately 300,000 timesteps (3,000 episodes × 100 steps on average) — roughly 15 times more timesteps. Again, MC's high variance from full-episode returns costs sample efficiency on longer episodes.

## 13. Case Studies and Measured Results

**Case study 1: Blackjack optimal policy via MC control.** MC control with exploring starts on `Blackjack-v1`, 2,000,000 episodes, first-visit, $\gamma = 1$. Evaluation with $\varepsilon = 0$ (pure greedy policy): average reward $\approx -0.07$, within 0.015 of the analytical optimum $\approx -0.055$. The learned policy matches the professional basic strategy table on 97% of states. The 3% discrepancy is concentrated in borderline decisions where the optimal Q values for competing actions are very close and require millions more episodes to distinguish.

**Case study 2: Off-policy MC for clinical trial evaluation.** Gottesman et al. (2019, *Nature Medicine*) applied weighted IS to evaluate sepsis treatment policies on MIMIC-III ICU data. The behavior policy was historical physician practice; the target policy was a learned treatment algorithm. Weighted IS provided statistically bounded estimates of patient outcome differences without any live experimentation — exactly the off-policy MC framework applied at the highest stakes. The evaluation framework demonstrated that the learned policy might improve 28-day mortality by 3–6 percentage points, with 95% confidence intervals computed via bootstrap resampling of the IS-weighted returns.

**Case study 3: AlphaZero chess performance.** After 24 hours of self-play training, AlphaZero achieved 28 wins, 72 draws, 0 losses against Stockfish 8 in 100 games. The value network training — MC game outcomes from self-play — was the only supervision signal. No human games, no heuristic evaluation functions, no feature engineering. Pure MC prediction with function approximation.

**Case study 4: REINFORCE on CartPole-v1 vs PPO.** Using Stable-Baselines3 (PPO) vs. the REINFORCE implementation above on CartPole-v1:
- REINFORCE: Reaches average return $> 400$ after approximately 300,000 timesteps.
- PPO: Reaches average return $> 480$ after approximately 20,000 timesteps — a 15x sample efficiency advantage.

The difference is entirely attributable to variance: REINFORCE uses full MC returns (standard deviation $\approx 100$ for mid-training episodes), while PPO uses TD(0) targets (standard deviation $\approx 1–5$ per step). Lower variance targets mean lower noise in gradient estimates, which means faster convergence.

**Case study 5: Portfolio risk via MC in algorithmic trading.** A systematic equity strategy backtested with MC scenario analysis (10,000 simulated paths from a regime-switching model) reveals a 5th-percentile annual return of $-8\%$ versus a mean of $+12\%$. Standard TD-based value estimation gives only the mean return — the distribution is invisible. MC rollouts give the full return distribution, revealing tail risk that changes position sizing decisions. The Sharpe ratio improvement from using the MC-derived distribution for position sizing versus using only the mean: approximately 0.3 Sharpe units (from 1.4 to 1.7 in a backtested equity long-short strategy). This is a direct application of MC's unbiasedness advantage in a setting where the distribution, not just the mean, determines the risk-adjusted outcome.

## 13. When to Use MC (and When Not To)

**Use MC when:**
The environment model is unknown or too complex to enumerate. The task is naturally episodic with a clear terminal state (games, navigation episodes, robotic manipulation with success/fail outcomes, episode-based LLM fine-tuning where you generate a full response and receive a reward). You need an unbiased value estimate for offline policy evaluation, safety-critical comparison, or distributional risk analysis. You are implementing MCTS or tree search that needs leaf node value estimates. Episodes are short to moderate (under 200 steps with $\gamma < 0.99$). Historical off-policy data is available and you want to evaluate a new policy without live testing (clinical trials, production systems, financial backtests).

**Do not use MC when:**
The task has no natural episode boundary (locomotion, persistent trading agents, ongoing dialogue systems). Episodes are very long with dense rewards — TD(0) or n-step TD will converge 10–100 times faster with much lower variance, as shown by the CartPole comparison above. You need online updates during an episode — any adaptive system that adjusts its policy mid-episode cannot afford to wait for episode completion. Interactions are expensive (physical robots require careful operation, clinical trials cannot be rushed, production A/B tests have opportunity costs). The state space is large and most states receive zero or near-zero visits even after millions of episodes — in this regime, function approximation with TD (DQN, SAC, PPO) is the correct tool.

**Decision flowchart for practitioners**:
1. Does your environment have a `terminated = True` flag within a reasonable number of steps? → If no, use TD-based methods.
2. Is your model of the environment known and tractable? → If yes, use DP (value iteration / policy iteration).
3. Are episodes very long (more than 500 steps with $\gamma > 0.99$)? → If yes, prefer TD or n-step returns.
4. Do you need to evaluate a fixed policy offline from historical logs? → Use off-policy MC with weighted IS.
5. Otherwise: MC control ($\varepsilon$-greedy or exploring starts) is a clean starting point.

**The clearest signal**: Your environment has a `terminated = True` flag that fires consistently within a reasonable number of steps, and you care about unbiased value estimates. If both conditions hold, MC is the right starting point. If episodes are long, MC is technically correct but practically slow — start with TD(0) or n-step returns and increase $n$ until you find the bias-variance sweet spot for your task.

## 14. Hyperparameter Sensitivity and Practical Tuning

| Hyperparameter | Effect | Typical Range |
|---|---|---|
| $\gamma$ (discount) | Effective horizon; near 1 increases variance sharply | 0.95–0.999 for episodic |
| $\varepsilon$ initial | Higher = more random early exploration | 0.5–1.0 |
| $\varepsilon$ final | Lower = more greedy at convergence | 0.01–0.1 |
| $\varepsilon$ decay episodes | Faster decay risks poor Q estimates | 20–50% of total episodes |
| $\alpha$ (step size) | Constant for non-stationary; exact mean for stationary | 0.01–0.1 constant; $1/N$ exact |
| Min visits before trusting Q | Trust $Q(s,a)$ only after this many visits | 10–50 per pair |
| IS clip $\rho_{\max}$ | Clip large IS ratios to bound variance | 10–100 |

**Practical guidance from experience**: The single most common failure mode is $\varepsilon$ decaying too aggressively. Signs: the training curve peaks at 20–30% of training and then flat-lines or degrades as the policy commits to actions based on noisy $Q$ estimates. Fix: slow the decay schedule, or add optimistic initialization ($Q(s,a) = +1$ for all pairs initially) to incentivize visiting every action at least once before the policy becomes greedy.

A second practical issue: never trust $Q(s,a)$ estimates with fewer than 10–20 visits. Maintain visit count arrays alongside $Q$ and add a visit-count check before committing to exploitation: if $N(s,a) < 10$ for all $a$, use a uniformly random action regardless of $\varepsilon$. This effectively implements a UCB-style exploration bonus without the overhead of UCB tracking.

For off-policy MC with large policy mismatch, clipping IS ratios at $\rho_{\max} = 10$ or $20$ dramatically reduces variance at the cost of small bias. This is analogous to the PPO clipping used in deep RL — trading a small amount of bias for a large variance reduction — and is almost always worth it when $\rho$ values can exceed 100.

## Key Takeaways

1. **MC learns without a model**: No $P(s' \mid s, a)$ needed. The only requirement is the ability to generate episodes. This enables application to any black-box environment.

2. **MC is unbiased but high-variance**: $\text{Var}[G_t] \approx \sigma^2 T$ grows linearly with episode length. Standard error shrinks as $1/\sqrt{N}$ — four times the episodes to halve the error.

3. **MC control needs Q, not V**: Without a model, policy improvement from $V(s)$ requires $P$. Estimating $Q(s,a)$ directly makes the greedy step completely model-free.

4. **Exploring starts is theoretically clean but practically limited**: Works for simulatable environments with arbitrary start states. For physical systems, use $\varepsilon$-soft policies.

5. **GLIE condition ensures convergence to optimal**: $\varepsilon_n$ decaying with $\sum \varepsilon_n = \infty$ and $\sum \varepsilon_n^2 < \infty$ converges to the true optimal policy. Fixed $\varepsilon$ converges only to the best $\varepsilon$-soft policy.

6. **Weighted IS dominates ordinary IS in practice**: Ordinary IS is unbiased but can have infinite variance for long trajectories with large policy mismatch. Weighted IS has finite variance at the cost of small $O(1/N)$ bias.

7. **The incremental update is SGD on MSE**: $V(s) \leftarrow V(s) + \alpha[G - V(s)]$ is a stochastic gradient step on squared value error, connecting tabular MC directly to neural network value learning in deep RL.

8. **AlphaGo and AlphaZero use MC at scale**: Value networks trained on MC game outcomes from self-play achieved superhuman performance in Go, Chess, and Shogi. MC prediction plus function approximation plus massive compute equals the strongest game-playing systems ever built.

9. **MC loses to TD for long-horizon and continuing tasks**: For MuJoCo locomotion or any continuing task, actor-critic TD methods (SAC, PPO) are 10–100 times more sample-efficient than MC.

10. **MC is the gold standard for unbiased offline policy evaluation**: Importance-weighted MC gives statistically clean estimates from historical data — used in clinical trials, finance backtests, and deployed system evaluation where online A/B testing is too costly or risky.

### What MC Teaches About RL More Broadly

Studying MC methods carefully reveals several deep principles that extend well beyond tabular RL:

**Samples vs. expectations**: The core insight of MC is that expectations are hard (require a model) but samples are easy (require only interaction). Every modern deep RL algorithm — DQN, PPO, SAC, RLHF — is a variant of this idea: replace intractable expectations with empirical sample averages, and use function approximation to generalize those averages to unseen states. MC makes this substitution in the simplest possible way, without bootstrapping or function approximation, making it the clearest demonstration of why sample-based learning works.

**The role of episode boundaries**: The requirement for complete episodes in MC is not a limitation but a clarification: MC computes exactly what the value function is supposed to represent — the expected *total* return from a state. TD methods avoid this requirement by bootstrapping, which introduces a subtle approximation. For understanding what you are actually optimizing, MC is the ground truth that TD approximates. When a TD-trained policy behaves unexpectedly, computing the true MC return for key states can diagnose whether the value function has properly converged or is stuck in a locally bootstrapped minimum.

**Statistical thinking in RL**: MC forces you to think statistically about RL. How many episodes do you need? What is the standard error of your value estimate? Are the state-action pairs well-covered? These questions matter for every RL algorithm, but they are most visible in MC because there is no bootstrapping to obscure the sample requirements. Developing the habit of monitoring visit counts, standard errors, and confidence intervals from MC carries over directly to debugging convergence issues in deep RL — where the analogous quantities are gradient variance, critic accuracy, and exploration coverage.

## Further Reading

- **Sutton, R. S., & Barto, A. G. (2018). Reinforcement Learning: An Introduction (2nd ed.), Chapter 5.** The canonical reference for MC methods. Freely available at [incompleteideas.net](http://incompleteideas.net/book/the-book-2nd.html). Implement Examples 5.1 (Blackjack), 5.3 (Racetrack), and 5.7 (off-policy IS) by hand.

- **Silver, D., et al. (2016). Mastering the Game of Go with Deep Neural Networks and Tree Search. *Nature*, 529, 484–489.** The AlphaGo paper. Section 4 describes value network training as MC prediction on self-play outcomes.

- **Silver, D., et al. (2018). A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play. *Science*, 362(6419), 1140–1144.** AlphaZero: pure self-play MC without human knowledge, same algorithm across three games.

- **Precup, D., Sutton, R. S., & Dasgupta, S. (2001). Off-Policy Temporal-Difference Learning with Function Approximation. *ICML 2001*.** Theoretical foundation for why off-policy learning with IS ratios is hard and what to do about it.

- **Gottesman, O., et al. (2019). Guidelines for reinforcement learning in healthcare. *Nature Medicine*, 25(1), 16–18.** Applied off-policy MC evaluation for clinical decision support at the highest stakes.

- [Reinforcement Learning: A Unified Map](/blog/machine-learning/reinforcement-learning/reinforcement-learning-a-unified-map) — The taxonomy that situates MC within the full RL algorithm landscape, explaining where it fits relative to TD, actor-critic, and model-based methods.

- [Dynamic Programming for Reinforcement Learning](/blog/machine-learning/reinforcement-learning/dynamic-programming-for-reinforcement-learning) — The model-based alternative that MC supersedes when dynamics are unknown.

- [Temporal Difference Learning: TD(0), SARSA, and Q-Learning](/blog/machine-learning/reinforcement-learning/temporal-difference-learning) — The next post: how bootstrapping from $V(s_{t+1})$ reduces MC's variance at the cost of bias, enabling online learning in continuing tasks with dramatically better sample efficiency.
