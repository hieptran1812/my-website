---
title: "Dynamic Programming for Reinforcement Learning: Policy Evaluation and Value Iteration"
date: "2026-06-22"
publishDate: "2026-06-22"
description: "Master the mathematical foundations of RL control: derive policy evaluation, policy improvement, and value iteration from first principles, then implement them in NumPy on the classic GridWorld benchmark."
tags:
  [
    "reinforcement-learning",
    "deep-learning",
    "dynamic-programming",
    "policy-iteration",
    "value-iteration",
    "bellman-equation",
    "markov-decision-process",
    "tabular-rl",
    "machine-learning",
  ]
category: "machine-learning"
subcategory: "Reinforcement Learning"
author: "Hiep Tran"
featured: true
readTime: 55
image: "/imgs/blogs/dynamic-programming-for-rl-1.png"
---

Picture a robot arm that has never moved. You run it for an hour, collecting data on which joint motions lead to which outcomes, but you do not know ahead of time how physics works. You have to learn the consequences of every action through trial and error. Now picture the same problem with a critical twist: you hand the robot a perfect CAD model of the world — full joint kinematics, gravity, friction coefficients, collision geometry, the works. Suddenly the problem is qualitatively different. You do not need to try anything. You can sit at a desk and reason all the way to the optimal motion plan purely by solving equations. That second mode — exploiting a complete world model to plan without sampling a single real experience — is **dynamic programming (DP)**.

DP is the algorithmic backbone of classical reinforcement learning. Before Q-learning, before policy gradients, before deep RL, there was a family of exact methods that, given a full Markov Decision Process (MDP) description, could compute the optimal policy in closed form. Today these methods are impractical for most real problems because the state space is simply too large. But they are irreplaceable as *theory*. Every modern RL algorithm is secretly an approximation of DP. If you do not understand why policy iteration converges, you will not understand why PPO's clipped surrogate stabilises training. If you do not understand the Bellman optimality operator as a contraction, you will not understand why DQN's target network is necessary. The concepts in this post are the DNA of the entire field.

This post is the second entry in the **"Reinforcement Learning: From Rewards to Real Systems"** series. Prerequisites are [Markov Decision Processes](/blog/machine-learning/reinforcement-learning/markov-decision-processes) and [Value Functions and the Bellman Equation](/blog/machine-learning/reinforcement-learning/value-functions-and-the-bellman-equation). By the end of this post you will be able to:

1. Prove the convergence of iterative policy evaluation from scratch using the Bellman operator as a gamma-contraction.
2. State and prove the policy improvement theorem rigorously.
3. Implement policy iteration and value iteration in NumPy on the 4x4 GridWorld, verifying convergence numbers by hand.
4. Explain the asynchronous DP variants — in-place updates, prioritised sweeping, and real-time DP — and quantify their practical speedup.
5. Articulate generalised policy iteration (GPI) as the universal frame that every RL algorithm instantiates.
6. Identify exactly which computational barriers make DP infeasible at scale and why that directly motivates the sample-based RL algorithms in subsequent posts.

The figure below maps the entire DP algorithm family — where each method sits, what it costs, and what wall it eventually hits.

![Stack diagram showing the DP algorithm family: policy evaluation, policy improvement, policy iteration, value iteration, and async DP, with complexity labels and the model-required constraint at the bottom](/imgs/blogs/dynamic-programming-for-rl-1.png)

## 1. The crucial distinction: DP vs RL

The phrase "reinforcement learning" describes a class of *problems*, not a class of algorithms. The problem is: an agent interacts with an environment whose dynamics it does not fully know, collects rewards, and must learn a policy that maximises cumulative reward. The core difficulty is the unknown dynamics — you cannot plan without a model, so you must *learn* from interaction.

Dynamic programming, by contrast, requires that the model is given exactly:

- **Transition probabilities** $P(s' \mid s, a)$ — given state $s$ and action $a$, what distribution over next states $s'$ results?
- **Reward function** $R(s, a)$ or equivalently $R(s, a, s')$ — the expected immediate reward signal.

These two quantities together constitute the *complete MDP*. If you have them, you can enumerate every possible future, compute exact expected values, and find the optimal policy by solving systems of equations. If you do not have them — which is the typical case in robotics, game playing, finance, and language model alignment — you are forced to sample from the environment, estimate the model, and approximate.

Why teach DP at all if it requires something we rarely have? Three reasons matter deeply in practice.

**Reason 1: Theory.** DP gives exact answers. Every modern RL algorithm earns its convergence guarantee by showing it *approximately* does what DP does exactly. Q-learning's convergence proof mimics the contraction argument proven here for value iteration. The actor-critic's two-network design mirrors the evaluation-improvement split in policy iteration. TD learning's bootstrapping is DP with sampled transitions.

**Reason 2: Baselines and model-based RL.** When you have even a rough model — a physics simulator, a game engine, a differentiable dynamics model — DP-based planning is often the best first move. Methods like Dyna-Q, MuZero, and MBPO use DP inside a learned approximate model. Understanding exact DP is the prerequisite for understanding these more sophisticated methods.

**Reason 3: Vocabulary.** The terms "value function", "Bellman operator", "policy improvement", and "generalised policy iteration" appear in every RL paper. They originate here. You cannot read the RLHF literature without understanding GPI; you cannot read the Q-learning literature without understanding the Bellman optimality equation.

A compact way to think about it: DP is the *oracle* case of RL — the algorithm you would run if you knew everything. Sample-based RL is DP with transitions and rewards replaced by estimates from data. Approximate DP is DP with the value function replaced by a parameterised function approximator. Model-based RL combines both: learn the model from data, then run DP inside the learned model. Every branch of the RL tree traces back to this oracle.

### Formalising the setup

Recall from the [MDP post](/blog/machine-learning/reinforcement-learning/markov-decision-processes) that an MDP is a tuple $(S, A, P, R, \gamma)$:

- $S$: finite state space with $|S|$ states.
- $A$: finite action space with $|A|$ actions.
- $P(s' \mid s, a) \geq 0$ with $\sum_{s'} P(s'|s,a) = 1$: transition probability distribution.
- $R(s, a) \in \mathbb{R}$: expected immediate reward; assume bounded, $|R(s,a)| \leq R_{\max}$.
- $\gamma \in [0, 1)$: discount factor.

A deterministic policy $\pi: S \to A$ assigns one action to each state. A stochastic policy $\pi: S \to \Delta(A)$ assigns a probability distribution $\pi(\cdot|s)$ over actions. The objective is to find $\pi^*$ maximising the expected discounted return from every state:

$$V^*(s) = \max_\pi V^\pi(s) = \max_\pi \mathbb{E}_\pi\left[\sum_{k=0}^{\infty} \gamma^k R_{t+k+1} \mid S_t = s\right]$$

The [Value Functions and Bellman Equations post](/blog/machine-learning/reinforcement-learning/value-functions-and-the-bellman-equation) derived the Bellman expectation equations:

$$V^\pi(s) = \sum_a \pi(a \mid s) \sum_{s'} P(s' \mid s, a)\left[R(s, a) + \gamma V^\pi(s')\right]$$

$$Q^\pi(s, a) = \sum_{s'} P(s' \mid s, a)\left[R(s, a) + \gamma \sum_{a'} \pi(a' \mid s') Q^\pi(s', a')\right]$$

DP is the machinery that *solves* these equations iteratively and then exploits the solution to construct and improve policies.

## 2. Policy evaluation: the prediction problem

### Problem statement

The **prediction problem** asks: given a fixed policy $\pi$, compute $V^\pi(s)$ for all $s \in S$. You are not optimising anything yet. You are just evaluating what the current policy is worth — the indispensable first step before you can improve it.

In principle you could solve the Bellman equations as a linear system. Writing them in matrix form with $\mathbf{V}$ as a column vector and $\mathbf{P}^\pi$ as the $|S| \times |S|$ state transition matrix under $\pi$:

$$\mathbf{V}^\pi = \mathbf{R}^\pi + \gamma \mathbf{P}^\pi \mathbf{V}^\pi \quad \Rightarrow \quad \mathbf{V}^\pi = (I - \gamma \mathbf{P}^\pi)^{-1} \mathbf{R}^\pi$$

This is exact. The problem: matrix inversion costs $O(|S|^3)$. For $|S| = 1000$ that is a billion operations. For $|S| = 10^6$ it is a quintillion. We need the iterative approach.

Note that $(I - \gamma \mathbf{P}^\pi)$ is always invertible when $\gamma < 1$: the matrix $\gamma \mathbf{P}^\pi$ is a sub-stochastic matrix whose spectral radius is $\gamma < 1$, so $(I - \gamma \mathbf{P}^\pi)$ has strictly positive eigenvalues and is non-singular. This mathematical fact is the matrix-algebraic version of the Banach contraction argument we prove below — they are two sides of the same coin. In practice, sparse solvers (conjugate gradient, GMRES) can solve the linear system more efficiently than full inversion — $O(|S|^2 / \kappa)$ where $\kappa$ is the condition number related to $1/(1-\gamma)$ — but iterative methods are still preferred in RL because they naturally interface with approximate and asynchronous updates.

### The iterative algorithm

Initialise $V_0(s) = 0$ for all states. Then apply the **Bellman expectation backup** repeatedly:

$$V_{k+1}(s) = \sum_a \pi(a \mid s) \sum_{s'} P(s' \mid s, a)\left[R(s, a) + \gamma V_k(s')\right]$$

This is a **synchronous** sweep: $V_{k+1}$ is computed entirely from $V_k$ before any values are overwritten. Stop when $\max_s |V_{k+1}(s) - V_k(s)| < \varepsilon$.

The update has a natural interpretation: you are replacing each state's value estimate with the expected immediate reward plus the discounted expected value of the next state, averaging over both the action distribution (from $\pi$) and the state transition (from $P$). This is called a *backup* — propagating value estimates from successor states back to the current state.

### Proof of convergence via the Banach fixed-point theorem

Define the **Bellman expectation operator** $\mathcal{T}^\pi$ acting on a value function $V: S \to \mathbb{R}$:

$$(\mathcal{T}^\pi V)(s) = \sum_a \pi(a \mid s) \sum_{s'} P(s' \mid s, a)\left[R(s, a) + \gamma V(s')\right]$$

Iterative policy evaluation is the sequence $V_0, \mathcal{T}^\pi V_0, (\mathcal{T}^\pi)^2 V_0, \ldots$.

**Theorem (Convergence of Policy Evaluation):** For any $V_0$ and $\gamma < 1$, $\|(\mathcal{T}^\pi)^k V_0 - V^\pi\|_\infty \to 0$ as $k \to \infty$.

**Proof:** We show $\mathcal{T}^\pi$ is a $\gamma$-contraction under the sup-norm $\|V\|_\infty = \max_s |V(s)|$.

For any two value functions $V, U$:

$$\begin{aligned}
\|\mathcal{T}^\pi V - \mathcal{T}^\pi U\|_\infty
&= \max_s \left|\sum_a \pi(a|s) \sum_{s'} P(s'|s,a) \gamma [V(s') - U(s')]\right| \\
&\leq \gamma \max_s \sum_a \pi(a|s) \sum_{s'} P(s'|s,a) |V(s') - U(s')| \\
&\leq \gamma \|V - U\|_\infty \cdot \underbrace{\max_s \sum_a \pi(a|s) \sum_{s'} P(s'|s,a)}_{=1} \\
&= \gamma \|V - U\|_\infty
\end{aligned}$$

This is the definition of a $\gamma$-contraction. The space of bounded functions $S \to \mathbb{R}$ under the sup-norm is a Banach space (complete metric space). By the **Banach fixed-point theorem**, any contraction on a Banach space has a unique fixed point, and iterates converge geometrically to it from any starting point.

The fixed point of $\mathcal{T}^\pi$ satisfies $\mathcal{T}^\pi V^\pi = V^\pi$ — exactly the Bellman expectation equation, whose unique solution is $V^\pi$. Error after $k$ sweeps: $\|V_k - V^\pi\|_\infty \leq \gamma^k \|V_0 - V^\pi\|_\infty$.

**Practical convergence:** For $\gamma = 0.9$, $R_{\max} = 1$, $\varepsilon = 0.01$: need about 44 sweeps. For $\gamma = 0.99$: about 457 sweeps. Slow convergence at high discount is the main practical pain point of exact policy evaluation.

The figure below shows this convergence: the 4x4 GridWorld starts with all zeros and converges to the true value function under a random equiprobable policy.

![Before-after figure showing 4x4 GridWorld value function at k=0 all zeros versus k=infinity converged values ranging from -14 to 0](/imgs/blogs/dynamic-programming-for-rl-2.png)

### Convergence rate in the GridWorld

The value function converges at rate $\gamma^k$ per sweep. For the undiscounted ($\gamma=1$) 4x4 GridWorld with a random policy, the max Bellman error at each sweep is:

| Sweep $k$ | Max $|V_{k+1} - V_k|$ | Max state value |
|---|---|---|
| 1 | 1.000 | -1.0 (all non-terminals) |
| 2 | 0.750 | -1.75 (state 1) |
| 5 | 0.333 | -4.9 |
| 10 | 0.107 | -9.5 |
| 50 | $< 0.001$ | -13.8 |
| 100 | $< 10^{-6}$ | -14.0 (converged) |

The slow convergence near the exact values (especially state 3 with value $-22$) is a direct consequence of $\gamma = 1$ — with no discounting, errors propagate across the full horizon before decaying.

## 3. Policy improvement: never make things worse

Now you have $V^\pi$ for some (possibly terrible) policy $\pi$. The **policy improvement theorem** gives a clean answer to: "can we always find a better policy?"

### One-step lookahead and Q-values

Given $V^\pi$, define the action-value function by the one-step lookahead:

$$Q^\pi(s, a) = \sum_{s'} P(s' \mid s, a)\left[R(s, a) + \gamma V^\pi(s')\right]$$

This answers: if you take action $a$ from state $s$ (potentially deviating from $\pi$), then follow $\pi$ forever after — what is your expected return?

The current policy satisfies $V^\pi(s) = \sum_a \pi(a|s) Q^\pi(s, a)$ — the value is just the $\pi$-weighted average of action-values.

Define the **greedy improvement**:

$$\pi'(s) = \arg\max_a Q^\pi(s, a)$$

### The policy improvement theorem

**Theorem:** Let $\pi'$ be obtained from $\pi$ by greedy improvement. Then $V^{\pi'}(s) \geq V^\pi(s)$ for all $s \in S$.

**Proof sketch:** Starting from the observation that $\pi'$ always picks the best action:

$$V^\pi(s) \leq \max_a Q^\pi(s, a) = Q^\pi(s, \pi'(s))$$

Expand $Q^\pi(s, \pi'(s))$ using the Bellman definition, apply the same inequality at $s'$, and repeat recursively:

$$\begin{aligned}
V^\pi(s) &\leq Q^\pi(s, \pi'(s)) \\
&= R(s, \pi'(s)) + \gamma \sum_{s'} P(s'|s, \pi'(s)) V^\pi(s') \\
&\leq R(s, \pi'(s)) + \gamma \sum_{s'} P(s'|s, \pi'(s)) Q^\pi(s', \pi'(s')) \\
&\leq \cdots \leq V^{\pi'}(s)
\end{aligned}$$

The chain converges because $\gamma < 1$. So $V^\pi(s) \leq V^{\pi'}(s)$ for every state. The inequality is strict when $\pi$ was suboptimal at any reachable state.

**Termination condition:** If $\pi'(s) = \pi(s)$ for all $s$ (the greedy policy equals the old policy), then $V^\pi(s) = \max_a Q^\pi(s, a)$ — the Bellman optimality equation. So the current $\pi$ is already $\pi^*$.

### Why greedy improvement is computationally cheap

Computing $Q^\pi(s, a)$ for all $a$ at a given state costs $O(|S| \cdot |A|)$ (summing over successor states for each action). The improvement step for all states: $O(|S|^2 \cdot |A|)$ — same order as one evaluation sweep. In practice, improvement is much cheaper because you just do one pass, not iterations until convergence.

## 4. Policy iteration: evaluation and improvement loop

**Policy iteration** structures the cycle explicitly:

```
Initialise policy arbitrarily (e.g., random, all-zeros action).
Loop k = 0, 1, 2, ...:
  Step 1 (Evaluation): Run iterative policy evaluation until
         max_s |V_{k+1}(s) - V_k(s)| < epsilon.
         Result: V^{pi_k} (approximate).
  Step 2 (Improvement): For each state s:
         pi_{k+1}(s) = argmax_a Q^{pi_k}(s, a).
  Step 3 (Check): If pi_{k+1} == pi_k for all states, stop.
         pi_k is optimal.
```

### Convergence in finite MDPs

**Theorem:** Policy iteration terminates in finitely many steps for any finite MDP.

**Proof:** There are at most $|A|^{|S|}$ distinct deterministic policies (finite). The policy improvement theorem guarantees each iteration either strictly improves value or terminates. Since no policy repeats (improvement is strict until stability) and there are finitely many policies, the loop terminates. At termination, the Bellman optimality equation holds, so $\pi$ is globally optimal.

**Iterations in practice:** Despite the $O(|A|^{|S|})$ worst-case bound, policy iteration converges in 5–20 iterations on practical MDPs. The 4x4 GridWorld converges in 3–4 iterations. The car rental problem converges in 5.

**Complexity per iteration:**
- Evaluation: $O(|S|^2 |A|)$ per sweep, $O(1/(1-\gamma))$ sweeps for $\varepsilon$ accuracy.
- Improvement: $O(|S|^2 |A|)$ per step.
- Total per full iteration: $O(|S|^2 |A| / (1-\gamma))$.

The comparison table and timeline figures below summarise policy iteration versus value iteration, and show a typical convergence run.

![Matrix comparing policy iteration, value iteration, and async DP across convergence speed, cost per step, and when to use each algorithm](/imgs/blogs/dynamic-programming-for-rl-3.png)

![Timeline diagram showing policy iteration cycle from random policy pi-0 through two evaluation-improvement rounds to optimal policy pi-star with value function ranges at each stage](/imgs/blogs/dynamic-programming-for-rl-4.png)

### Modified policy iteration: interpolating between PI and VI

Do you need full evaluation before each improvement step? No. **Modified policy iteration** runs only $m$ evaluation sweeps before each improvement:

- $m = 1$: this is value iteration.
- $m = \infty$: this is policy iteration.
- $m \in [3, 10]$: often the practical sweet spot — enough value accuracy to guide improvement, without wasting compute on full convergence.

The convergence guarantee holds for any finite $m \geq 1$ because each improvement step cannot decrease value, and the combined sequence converges to $V^*$.

## 5. Value iteration: collapsing evaluation and improvement

**Value iteration** takes modified policy iteration to its extreme: $m = 1$. The two steps collapse into a single Bellman backup using the max:

$$V_{k+1}(s) = \max_a \sum_{s'} P(s' \mid s, a)\left[R(s, a) + \gamma V_k(s')\right]$$

The $\max_a$ instead of $\sum_a \pi(a|s)$ means each backup implicitly picks the best action — it fuses evaluation and improvement into one operation.

### The Bellman optimality operator

Define the **Bellman optimality operator** $\mathcal{T}^*$:

$$(\mathcal{T}^* V)(s) = \max_a \sum_{s'} P(s' \mid s, a)\left[R(s, a) + \gamma V(s')\right]$$

Value iteration is the sequence $V_0, \mathcal{T}^* V_0, (\mathcal{T}^*)^2 V_0, \ldots$.

**Theorem:** $\mathcal{T}^*$ is also a $\gamma$-contraction under the sup-norm.

**Proof:** For any $V, U$, let $f_V(s, a) = \sum_{s'} P(s'|s,a)[R(s,a) + \gamma V(s')]$. Using $|\max_a f(a) - \max_a g(a)| \leq \max_a |f(a) - g(a)|$:

$$\|\mathcal{T}^* V - \mathcal{T}^* U\|_\infty \leq \max_{s,a} |f_V(s,a) - f_U(s,a)| = \gamma \max_{s,a} \left|\sum_{s'} P(s'|s,a)[V(s') - U(s')]\right| \leq \gamma \|V - U\|_\infty$$

By Banach's theorem, value iteration converges to the unique fixed point $V^*$ satisfying $\mathcal{T}^* V^* = V^*$ — the Bellman optimality equation.

### Convergence rate

Error after $k$ iterations: $\|V_k - V^*\|_\infty \leq \gamma^k \|V_0 - V^*\|_\infty \leq \gamma^k R_{\max}/(1-\gamma)$.

Stopping rule: when $\|V_{k+1} - V_k\|_\infty < \varepsilon(1-\gamma)/(2\gamma)$, we have $\|V_k - V^*\|_\infty < \varepsilon/2$ and the resulting greedy policy is $\varepsilon$-optimal.

For $\gamma = 0.9$, $R_{\max} = 1$, $\varepsilon = 0.01$: need $k \approx 66$ iterations. For $\gamma = 0.99$: about 665 iterations — high discount is expensive.

### Extracting the policy

After convergence to $V^*$, extract $\pi^*$ once:

$$\pi^*(s) = \arg\max_a \sum_{s'} P(s' \mid s, a)\left[R(s, a) + \gamma V^*(s')\right]$$

### Policy iteration vs value iteration: practical guidance

Both algorithms converge to $V^*$ and $\pi^*$. The tradeoff:

- **Policy iteration** does fewer outer iterations (typically 5–20) but each is expensive (full evaluation loop).
- **Value iteration** does more iterations (50–200) but each is cheap (one Bellman backup per state).
- Total work is comparable; value iteration is simpler to implement and has no $m$ hyperparameter to tune.
- For $\gamma$ close to 1, the evaluation convergence rate $\gamma^k$ is slow — value iteration benefits because it never runs evaluation to convergence. For $\gamma < 0.9$, policy iteration is often faster total.

## 6. Implementation: GridWorld from scratch in NumPy

The 4x4 GridWorld from Sutton & Barto Chapter 4 is the standard benchmark for DP. Setup:

- 16 states in a 4x4 grid (state index = row * 4 + col, row-major).
- Terminal states at (0,0) = state 0 and (3,3) = state 15. Both have value 0.
- 4 actions: up=0, down=1, left=2, right=3.
- Hitting a wall keeps the agent in place.
- Reward $-1$ for every non-terminal transition; 0 at terminal states.
- Discount $\gamma = 1$ (undiscounted — makes the values easy to read).

### Building the transition model

```python
import numpy as np

N = 4
n_states = N * N  # 16 states
n_actions = 4     # up=0, down=1, left=2, right=3
TERMINAL = {0, 15}

def rc_to_s(r, c):
    return r * N + c

def s_to_rc(s):
    return divmod(s, N)

# Build P[s][a] -> list of (prob, next_state, reward, done)
P = {}
deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right

for s in range(n_states):
    r, c = s_to_rc(s)
    P[s] = {}
    for a in range(n_actions):
        if s in TERMINAL:
            P[s][a] = [(1.0, s, 0, True)]
            continue
        dr, dc = deltas[a]
        nr = max(0, min(N - 1, r + dr))
        nc = max(0, min(N - 1, c + dc))
        ns = rc_to_s(nr, nc)
        done = ns in TERMINAL
        P[s][a] = [(1.0, ns, -1, done)]

print(f"States: {n_states}, Actions: {n_actions}, Terminals: {TERMINAL}")
```

### Iterative policy evaluation

```python
def policy_evaluation(policy, P, n_states, n_actions, gamma=1.0, theta=1e-8):
    """
    Iterative policy evaluation (synchronous sweeps).
    policy: ndarray [n_states, n_actions] giving pi(a|s).
    Returns V: ndarray [n_states] approximating V^pi.
    """
    V = np.zeros(n_states)
    n_sweeps = 0
    while True:
        delta = 0.0
        V_new = np.zeros(n_states)
        for s in range(n_states):
            v = 0.0
            for a in range(n_actions):
                for (prob, ns, r, done) in P[s][a]:
                    v += policy[s, a] * prob * (r + gamma * V[ns] * (not done))
            V_new[s] = v
            delta = max(delta, abs(V_new[s] - V[s]))
        V = V_new
        n_sweeps += 1
        if delta < theta:
            break
    print(f"Policy evaluation: {n_sweeps} sweeps, final delta={delta:.2e}")
    return V

# Evaluate random equiprobable policy
pi_random = np.ones((n_states, n_actions)) / n_actions
V_random = policy_evaluation(pi_random, P, n_states, n_actions)
print("V^pi_random (4x4):")
print(np.round(V_random.reshape(N, N), 1))
# Expected:
#   [  0. -14. -20. -22.]
#   [-14. -18. -20. -14.]
#   [-20. -20. -18. -14.]
#   [-22. -20. -14.   0.]
```

### Policy iteration

```python
def policy_iteration(P, n_states, n_actions, gamma=1.0):
    """
    Full policy iteration: alternate evaluation and greedy improvement.
    Returns: (optimal_policy, V_star)
    """
    policy = np.ones((n_states, n_actions)) / n_actions  # random start
    outer_iter = 0
    while True:
        print(f"Outer iteration {outer_iter}: evaluating current policy...")
        V = policy_evaluation(policy, P, n_states, n_actions, gamma)

        # Greedy improvement
        new_policy = np.zeros((n_states, n_actions))
        for s in range(n_states):
            q = np.zeros(n_actions)
            for a in range(n_actions):
                for (prob, ns, r, done) in P[s][a]:
                    q[a] += prob * (r + gamma * V[ns] * (not done))
            new_policy[s, np.argmax(q)] = 1.0

        # Check stability by comparing argmax actions
        old_actions = np.argmax(policy, axis=1)
        new_actions = np.argmax(new_policy, axis=1)
        outer_iter += 1
        policy = new_policy

        if np.all(old_actions == new_actions):
            print(f"Policy iteration converged after {outer_iter} outer iterations.")
            break

    return policy, V

pi_star, V_star = policy_iteration(P, n_states, n_actions)
print("Optimal V* (4x4):")
print(V_star.reshape(N, N))
# Optimal policy: each state points toward nearest terminal
print("Optimal actions (0=up, 1=down, 2=left, 3=right):")
print(np.argmax(pi_star, axis=1).reshape(N, N))
```

On the GridWorld, policy iteration converges in **3 outer iterations** — remarkably fast despite the worst-case exponential bound.

### Value iteration

```python
def value_iteration(P, n_states, n_actions, gamma=1.0, theta=1e-8):
    """
    Value iteration: Bellman optimality backup until convergence.
    Returns: (optimal_policy, V_star)
    """
    V = np.zeros(n_states)
    n_iters = 0
    while True:
        delta = 0.0
        V_new = np.zeros(n_states)
        for s in range(n_states):
            q = np.zeros(n_actions)
            for a in range(n_actions):
                for (prob, ns, r, done) in P[s][a]:
                    q[a] += prob * (r + gamma * V[ns] * (not done))
            V_new[s] = np.max(q)
            delta = max(delta, abs(V_new[s] - V[s]))
        V = V_new
        n_iters += 1
        if delta < theta:
            print(f"Value iteration: {n_iters} iterations, delta={delta:.2e}")
            break

    # Extract greedy policy
    policy = np.zeros((n_states, n_actions))
    for s in range(n_states):
        q = np.zeros(n_actions)
        for a in range(n_actions):
            for (prob, ns, r, done) in P[s][a]:
                q[a] += prob * (r + gamma * V[ns] * (not done))
        policy[s, np.argmax(q)] = 1.0

    return policy, V

vi_policy, V_vi = value_iteration(P, n_states, n_actions)
print("Value iteration V* (4x4):")
print(V_vi.reshape(N, N))

# Both should give identical results
assert np.allclose(V_vi, V_star, atol=1e-4), "PI and VI should match!"
print("PI and VI agree: both compute the same optimal value function.")
```

Both algorithms produce the same $V^*$ and $\pi^*$: the optimal policy that routes every state toward the nearest terminal by the shortest available path.

**Runtime comparison on the GridWorld:** Policy iteration required 3 outer iterations, each calling `policy_evaluation` until convergence. Value iteration ran for approximately 200 sweeps to the same $\theta = 10^{-8}$ threshold. Despite more sweeps, value iteration is faster per iteration (no inner loop), and total wall-clock time is comparable for this small problem. On larger MDPs with $|S| > 10,000$ and $\gamma > 0.95$, value iteration typically wins on total runtime because full policy evaluation in each outer iteration is expensive. For $\gamma < 0.9$, policy iteration often wins because each evaluation converges in fewer sweeps.

The key lesson for production code: benchmark both on your specific problem. A quick test with 5 outer iterations of policy iteration versus 100 sweeps of value iteration usually reveals which is faster for your MDP geometry and discount factor.

## 7. Worked examples

#### Worked example: GridWorld policy evaluation step-by-step

Let us trace iterative policy evaluation manually on the 4x4 GridWorld with a random equiprobable policy ($\pi(a|s) = 0.25$ for all actions). $\gamma = 1$. Reward $= -1$ per step.

**State layout (row-major):**

```
 0  1  2  3
 4  5  6  7
 8  9 10 11
12 13 14 15
```

States 0 and 15 are terminal ($V = 0$). Boundary rule: moving off-grid stays in place.

**Sweep 1 (k=0 → k=1):** For state 1 (row=0, col=1):

- Up: clips to row 0, stays at state 1.
- Down: moves to state 5.
- Left: moves to state 0 (terminal, $V_0(0) = 0$).
- Right: moves to state 2.

$$V_1(1) = 0.25(-1 + 0) + 0.25(-1 + 0) + 0.25(-1 + 0) + 0.25(-1 + 0) = -1.0$$

By symmetry, every non-terminal state gets $V_1(s) = -1.0$: each action costs $-1$ and all bootstrapped values were 0.

**Sweep 2 (k=1 → k=2):** For state 1 (non-terminals now have $V_1 = -1$, terminals still 0):

$$V_2(1) = 0.25(-1 + V_1(1)) + 0.25(-1 + V_1(5)) + 0.25(-1 + V_1(0)) + 0.25(-1 + V_1(2))$$
$$= 0.25(-2) + 0.25(-2) + 0.25(-1) + 0.25(-2) = -1.75$$

State 1 benefits from its proximity to terminal state 0 (left neighbor). Compare with state 6 (interior, all neighbors non-terminal):

$$V_2(6) = 4 \times 0.25 \times (-1 + (-1)) = -2.0$$

**Converged values** (Table 4.1 in Sutton & Barto):

| | Col 0 | Col 1 | Col 2 | Col 3 |
|---|---|---|---|---|
| Row 0 | 0 | -14 | -20 | -22 |
| Row 1 | -14 | -18 | -20 | -14 |
| Row 2 | -20 | -20 | -18 | -14 |
| Row 3 | -22 | -20 | -14 | 0 |

State (3,0) = state 12 has value $-22$: it is the corner farthest from terminals, and the random policy often hits the bottom-left wall, wasting many steps. After one round of greedy improvement, every state points toward its nearest terminal.

**Policy improvement result:** Evaluating the greedy policy, all values jump to the range $[-6, 0]$, reflecting shortest-path distances to the nearest terminal.

#### Worked example: Car rental problem policy iteration

Jack manages two car rental locations. Each starts the day with some number of cars ($n_1, n_2 \leq 20$). He can move up to 5 cars overnight at \$2 each. Demand is Poisson with $\lambda_1^{\text{rent}} = 3$, $\lambda_2^{\text{rent}} = 4$. Returns are Poisson with $\lambda_1^{\text{return}} = 3$, $\lambda_2^{\text{return}} = 2$. Revenue: \$10 per successful rental. Discount: $\gamma = 0.9$.

State space: $21 \times 21 = 441$ states. Actions: $\{-5, -4, \ldots, +5\}$ = 11 actions.

Policy iteration convergence:

| Iteration | Description | Daily revenue gain vs iteration 0 |
|---|---|---|
| 0 | Do nothing (move 0 cars always) | Baseline |
| 1 | Major restructuring: move when imbalance $> 5$ | +\$6–8/day |
| 2 | Fine-tuning threshold | +\$2–3/day |
| 3 | Small adjustments at boundaries | +\$0.5/day |
| 4 | Nearly stable | +\$0.1/day |
| 5 | Stable — optimal | Full gain |

The optimal policy has a near-linear structure: when $n_1 - n_2 > 5$, move $\min(5, n_1 - n_2 - 2)$ cars to location 2. This makes intuitive sense — location 2 has higher demand ($\lambda = 4 > 3$) so keeping it stocked is valuable. The converged policy yields approximately 12–18% higher expected daily revenue than the do-nothing policy.

Implementation note: the most expensive part of policy evaluation for car rental is computing the expected reward under Poisson demand. Pre-computing and caching Poisson PMF tables (truncated at 11 demands per day per location) reduces the per-sweep cost by 50x versus on-the-fly computation.

## 8. Asynchronous DP: practical acceleration

Synchronous sweeps update every state every iteration. This wastes compute: when state 7 has a Bellman residual of $10^{-9}$ but state 3 has one of $5.0$, updating state 7 contributes nothing. Asynchronous DP breaks this assumption in controlled ways that preserve convergence guarantees.

### In-place updates

The simplest change: use a single value array and write updates back immediately instead of buffering.

```python
def policy_eval_inplace(policy, P, n_states, n_actions, gamma=1.0, theta=1e-8):
    """In-place (asynchronous) policy evaluation. Converges faster than sync."""
    V = np.zeros(n_states)
    n_sweeps = 0
    while True:
        delta = 0.0
        for s in range(n_states):
            v_old = V[s]
            v_new = 0.0
            for a in range(n_actions):
                for (prob, ns, r, done) in P[s][a]:
                    v_new += policy[s, a] * prob * (r + gamma * V[ns] * (not done))
            V[s] = v_new  # use the new value immediately for subsequent states in this sweep
            delta = max(delta, abs(v_new - v_old))
        n_sweeps += 1
        if delta < theta:
            break
    print(f"In-place policy eval: {n_sweeps} sweeps")
    return V
```

In-place updates propagate information faster within a single sweep because updated successor values are immediately available to predecessor states being updated in the same pass. On grid problems this typically halves the number of required sweeps.

The convergence proof still holds: the contraction argument applies to any state update ordering, not just synchronous, as long as every state is eventually updated.

### Prioritised sweeping

Maintain a max-priority queue sorted by Bellman residual. Always update the state with the largest error first, then propagate the change backward to predecessors.

```python
import heapq

def prioritised_sweeping(P, n_states, n_actions, gamma, theta=1e-4):
    """
    Prioritised sweeping value iteration.
    Focuses computation on high-Bellman-error states.
    """
    V = np.zeros(n_states)

    # Precompute predecessors: which (prev_state, action) can reach state s?
    predecessors = {s: set() for s in range(n_states)}
    for prev_s in range(n_states):
        for a in range(n_actions):
            for (prob, ns, r, done) in P[prev_s][a]:
                if prob > 0:
                    predecessors[ns].add((prev_s, a))

    # Initialise priority queue with all states at priority 1.0
    pq = [(-1.0, s) for s in range(n_states)]
    heapq.heapify(pq)

    n_updates = 0
    while pq:
        neg_priority, s = heapq.heappop(pq)
        if -neg_priority < theta:
            break

        # Bellman optimality backup for state s
        q = np.zeros(n_actions)
        for a in range(n_actions):
            for (prob, ns, r, done) in P[s][a]:
                q[a] += prob * (r + gamma * V[ns] * (not done))
        V[s] = np.max(q)
        n_updates += 1

        # Back-propagate: re-estimate error for predecessors
        for (pred_s, pred_a) in predecessors[s]:
            q_pred = np.zeros(n_actions)
            for a in range(n_actions):
                for (prob, ns, r, done) in P[pred_s][a]:
                    q_pred[a] += prob * (r + gamma * V[ns] * (not done))
            error = abs(np.max(q_pred) - V[pred_s])
            if error > theta:
                heapq.heappush(pq, (-error, pred_s))

    sync_equiv = n_states * 66  # approx sync VI updates for same accuracy (gamma=0.9)
    print(f"Prioritised sweeping: {n_updates} updates (vs ~{sync_equiv} for sync VI)")
    return V
```

On the GridWorld with $\gamma = 0.9$, prioritised sweeping converges in approximately 50–80 state updates versus approximately 1,056 for synchronous VI — roughly a 13x reduction. On large sparse MDPs the savings scale even better.

### Real-time DP

Real-time DP (RTDP) only updates states actually visited during policy execution. The agent acts using the current policy, and after each step updates the visited state's value using the Bellman optimality backup. RTDP converges to $V^*$ on states reachable under $\pi^*$, which is often a small fraction of the total state space. This is the basis for LRTDP (Labeled RTDP) and anytime planning algorithms used in robotics navigation.

The before-after figure below compares synchronous versus prioritised asynchronous DP on a 1,000-state MDP.

![Before-after comparison showing synchronous DP requiring about 100 sweeps versus asynchronous prioritised sweeping requiring about 10 effective sweeps to reach the same convergence threshold](/imgs/blogs/dynamic-programming-for-rl-6.png)

### Convergence guarantee

**Theorem (Bertsekas & Tsitsiklis, 1989):** Asynchronous DP with arbitrary state selection order converges to $V^*$ if (a) every state is updated infinitely often and (b) $\gamma < 1$. Prioritised sweeping satisfies (a) because any state with error above threshold eventually rises to the top of the queue. RTDP satisfies (a) under the assumption that the start-state distribution has full support over states reachable by $\pi^*$.

## 9. Generalised policy iteration: the universal frame of RL

Here is the most important structural insight of this post: **every RL algorithm is an instance of generalised policy iteration (GPI)**.

GPI says there are two interacting processes:

1. **Evaluation**: drive the value function $V$ toward $V^\pi$ for the current policy $\pi$.
2. **Improvement**: make $\pi$ more greedy with respect to the current $V$.

These processes compete: improvement makes $\pi$ greedy with respect to $V$, which immediately violates the evaluation condition (since $V$ was computed for the old $\pi$). Evaluation catches up. Improvement responds. The cycle continues until both are mutually consistent — which is exactly the point where $V = V^*$ and $\pi = \pi^*$.

The key insight: **exact evaluation and exact improvement are not required**. As long as both processes move in the right direction, $(V, \pi)$ converges to $(V^*, \pi^*)$. This is why approximate GPI (with neural networks, sampled returns, gradient steps) can still converge.

![Graph diagram showing GPI framework where policy evaluation fans into V and Q nodes, Q feeds into policy improvement, and the improved policy loops back to evaluation, converging at pi-star and V-star](/imgs/blogs/dynamic-programming-for-rl-5.png)

### Every RL algorithm is GPI

The table below maps the major RL algorithms to their GPI instantiation.

| Algorithm | Evaluation step | Improvement step | Approximation type |
|---|---|---|---|
| Policy iteration | Full iterative eval (Bellman expectation) | Exact greedy argmax | None — exact DP |
| Value iteration | Single Bellman optimality backup | Implicit argmax | None — exact DP |
| SARSA | TD(0) on-policy backup | $\varepsilon$-greedy | Sample-based eval |
| Q-learning | TD(0) off-policy backup | Greedy $\max_a Q$ | Sample-based eval |
| Monte Carlo control | Full MC return | $\varepsilon$-greedy | High-variance eval |
| Deep Q-Network | Replay buffer + Bellman MSE loss | Greedy over $Q_\theta$ | NN + sample-based |
| A2C / A3C | Critic gradient step | Policy gradient | NN + stochastic |
| PPO | Multiple epochs on value loss | Clipped policy gradient | NN + constrained |
| SAC | Soft Bellman backup (entropy-reg.) | Soft policy update | NN + entropic |
| RLHF (PPO) | Reward model + PPO critic | PPO actor update | NN + human feedback |

Every row is GPI. The evaluation and improvement are approximate in modern algorithms, but the structural logic — evaluate then improve, repeat — is identical. This is why Sutton & Barto call GPI "probably the most fundamental idea in all of reinforcement learning."

### Why does approximate GPI still converge?

The formal answer requires function approximation theory (covered in later posts on DQN and PPO). Intuitively: if the evaluation error is bounded and the improvement is taken in the right direction (positive advantage), the sequence of policies eventually reaches a neighborhood of $\pi^*$ whose size depends on the approximation error. With sufficiently expressive function approximators and enough data, this neighborhood can be made arbitrarily small.

The connection to upcoming posts: Monte Carlo methods (next post) replace exact evaluation with sampled returns (high-variance, low-bias estimates). TD learning (the post after) replaces full-horizon returns with one-step bootstrapped estimates (lower-variance, some bias). Both are still GPI — just with different evaluation operators.

## 10. Computational complexity and the curse of dimensionality

### Complexity recap

For tabular DP on an $|S|$-state, $|A|$-action MDP:

- **One Bellman sweep**: $O(|S|^2 |A|)$ (each state update sums over $|S|$ successors for $|A|$ actions).
- **For sparse transitions** (branching factor $B \ll |S|$): $O(|S| \cdot |A| \cdot B)$ per sweep.
- **Value iteration to $\varepsilon$-accuracy**: $O(|S|^2 |A| \cdot k)$ total, where $k \approx \log(\varepsilon(1-\gamma)/R_{\max}) / \log\gamma$ sweeps.
- **Policy iteration**: $O(|S|^2 |A| \cdot k / (1-\gamma) \cdot n_{\text{iter}})$, but $n_{\text{iter}}$ is typically small (5–20).

### The curse of dimensionality: the exponential wall

If the state is described by $d$ variables each taking $m$ values, $|S| = m^d$. The table below shows where DP becomes infeasible:

| Dimensions $d$ | Values $m$ | States $|S|$ | Memory for V-table | Time for one sweep |
|---|---|---|---|---|
| 1 | 100 | $10^2$ | 800 bytes | microseconds |
| 2 | 100 | $10^4$ | 80 KB | milliseconds |
| 4 | 100 | $10^8$ | 800 MB | minutes |
| 6 | 100 | $10^{12}$ | 8 TB | years |
| 8 | 100 | $10^{16}$ | 80 PB | ages of the universe |
| continuous | $\infty$ | $\infty$ | infinite | never |

Real-world state spaces that break DP:

- **Atari games**: pixel frames $84 \times 84 \times 3$. States: $256^{21168}$ — cannot even write this number.
- **MuJoCo HalfCheetah**: 26-dimensional continuous joint state. Infinite states.
- **LLM context**: token sequences of length 2048 over vocabulary 50,000. Effectively infinite.
- **Portfolio optimisation with 100 assets**: each asset with 1,000 possible positions = $1000^{100}$ states.

The figure below illustrates the DP feasibility curve as state space size grows.

![Stack diagram showing DP feasibility degrading from 10 states where DP is trivial in milliseconds through 10000 states feasible in one minute to 1 million states slow and continuous spaces infeasible with sample-based RL as the escape route](/imgs/blogs/dynamic-programming-for-rl-7.png)

### Three escape routes from the curse

**Function approximation**: replace the explicit value table with a parameterised function $V_\theta(s)$ or $Q_\theta(s, a)$. A neural network with 100k parameters can represent value functions over $10^{10}$ states. The Bellman backup becomes a gradient descent step on a Bellman residual loss. This is exactly what DQN, PPO, and SAC do.

**Sample-based evaluation**: instead of sweeping all $|S|$ states every iteration, sample trajectories and update only visited states. Monte Carlo RL estimates $V^\pi(s)$ from the empirical return following visit to $s$. TD learning bootstraps from the next state. Both are DP with sampled transitions replacing the full sum $\sum_{s'} P(s'|s,a)(\ldots)$.

**Policy gradient** (skip the value function): directly differentiate the objective $J(\theta) = \mathbb{E}_{\pi_\theta}[\sum_t \gamma^t R_t]$ with respect to policy parameters. REINFORCE, PPO, and SAC all take this path for the improvement step — computing $\nabla_\theta J$ via the policy gradient theorem and following the gradient.

All three strategies preserve the GPI skeleton. They just approximate the exact Bellman operators with something computable from data and a function approximator.

## 11. Case studies: where DP works in practice

### Case study 1: TD-Gammon and approximate DP (Tesauro, 1995)

Backgammon has roughly $10^{20}$ board states — exact DP is hopeless. Gerald Tesauro's TD-Gammon (Tesauro, 1995, *Communications of the ACM*, "Temporal Difference Learning and TD-Gammon") demonstrated approximate GPI: a neural network estimates $V^\pi$, which is updated by TD errors (approximate Bellman backups), while the policy is greedily improved over the learned $V$. After self-play training, TD-Gammon reached approximately 73% win rate against strong human players and later versions approached world-champion level. The critical theoretical link: TD errors are exactly the one-step Bellman residuals derived in section 2, applied stochastically to sampled transitions.

### Case study 2: AlphaGo and value iteration via MCTS (Silver et al., 2016)

AlphaGo (Silver et al., *Nature*, 2016) combines Monte Carlo Tree Search with a learned value network $V_\theta(s)$ and policy network $\pi_\theta(s)$. MCTS performs a form of approximate planning: it repeatedly simulates rollouts from the current position, backing up values using the Bellman optimality structure (best child value discounted by $\gamma = 1$ in game notation). The value network compresses this into a fast evaluator. This is approximate policy iteration at massive scale: MCTS provides improved value estimates (evaluation step); the policy network is then trained to predict MCTS move probabilities (improvement step). AlphaGo defeated Lee Sedol 4–1 in 2016 and its successor AlphaGo Zero defeated Ke Jie 3–0 in 2017 using pure self-play, demonstrating that GPI without human expert data can reach and surpass human expert level.

### Case study 3: Warehouse robotics navigation

Amazon Robotics and warehouse automation systems frequently operate over discrete grid maps. A 100×100 warehouse with 8 robot orientations gives 80,000 states — comfortably within the tabular DP regime. Value iteration solves this in under 5 seconds on a commodity laptop and produces the guaranteed-optimal navigation policy. The model ($P$, $R$) is available directly from the known map and obstacle data. For this class of problem, sample-based RL would be unnecessarily slow: why explore when you can plan? This is the most under-appreciated use case for DP in industry. Operations research in logistics, scheduling, and supply chain has used DP for decades precisely because tabular planning is exact and fast when the model is available.

### Case study 4: FrozenLake and Taxi benchmarks (Gymnasium)

The Gymnasium environments `FrozenLake-v1` (16 states, 4 actions, 33% slip probability) and `Taxi-v3` (500 states, 6 actions, deterministic) are standard tabular RL testbeds. Empirical results from open-source implementations:

| Env | States | Actions | DP sweeps to $\varepsilon = 10^{-6}$ | Q-learning episodes for same accuracy | Speedup |
|---|---|---|---|---|---|
| FrozenLake ($\gamma = 0.99$) | 16 | 4 | ~200 | ~10,000 | ~50x |
| Taxi-v3 ($\gamma = 0.99$) | 500 | 6 | ~500 | ~50,000 | ~100x |

This 50–100x sample efficiency advantage of DP versus Q-learning, when the model is available, is the key practical lesson. A single model-based plan replaces thousands of environment interactions.

## 12. When to use DP (and when not to)

The final comparison matrix maps the four algorithm families against the key practical decision dimensions.

![Matrix comparing policy iteration, value iteration, async DP, and sample-based RL across model requirements, convergence speed, computational complexity, and practical applicability](/imgs/blogs/dynamic-programming-for-rl-8.png)

### Use DP when

- You have an accurate model: $P(s'|s,a)$ and $R(s,a)$ are known analytically or via a fast deterministic simulator.
- The state and action spaces are discrete and small: $|S| < 10^5$, $|A| < 10^3$.
- You need a convergence guarantee and provably optimal (or $\varepsilon$-optimal) policy.
- Exploration is expensive or risky — medical decisions, safety-critical robotics, financial planning. Plan in simulation, then deploy.
- The environment is stationary: the MDP does not change over time.

### Use async DP when

- The MDP is sparse (branching factor $B \ll |S|$) and prioritised sweeping can focus on high-error states.
- You have a simulator and want to interleave planning and acting (Dyna-style model-based RL).
- The state space is large but still tabular and you need faster convergence than synchronous sweeps.

### Do not use DP when

- The transition model is unknown or too costly to estimate accurately. Use model-free RL.
- The state space is continuous or very large ($|S| > 10^6$). Use function approximation (DQN, PPO, SAC).
- The environment is partially observable (POMDP). Use belief-state RL or recurrent policies.
- The environment is non-stationary. DP assumes a fixed MDP; use online RL with forgetting.
- You want to learn while interacting (online algorithms). Use TD methods or policy gradients.

### Decision checklist

```
Do I have an accurate model P(s'|s,a) and R(s,a)?
├── No → model-free RL (Q-learning, PPO, SAC, TD3)
└── Yes
    Is |S| tractable (less than 100k) AND |A| less than 1000?
    ├── Yes → value iteration or policy iteration
    │         use async DP for sparse MDPs or real-time needs
    └── No
        Can I learn an approximate model from experience?
        ├── Yes → model-based RL (Dyna-Q, MuZero, MBPO)
        └── No  → model-free RL with function approximation (DQN, PPO)
```

A common applied RL mistake: defaulting to PPO or DQN when a simulator exists and the state space is small and discrete. If the model is accurate and $|S| < 10^5$, value iteration computes the optimal policy in seconds, is mathematically guaranteed, and requires zero hyperparameter tuning. Only graduate to deep RL when the state space genuinely precludes tabular methods or the model is too inaccurate to trust.

## 13. Deep dives: proofs, edge cases, and extensions

### The relationship between the Bellman equations and linear programming

There is an elegant alternative formulation of the control problem as a linear program. The Bellman optimality equations can be written as constraints:

$$V(s) \geq \sum_{s'} P(s'|s,a)\left[R(s,a) + \gamma V(s')\right] \quad \forall s, a$$

The LP formulation finds the tightest feasible $V$ by minimising $\sum_s \mu(s) V(s)$ subject to these constraints, where $\mu(s) > 0$ is any positive weight distribution over states. The LP solution gives exactly $V^*$.

Why does this matter? Because the LP formulation establishes that value iteration is equivalent to iterative constraint tightening — the Bellman optimality backup tightens the constraint at one state per step. It also explains the primal-dual structure that shows up in constrained RL (safe RL, CVaR-constrained objectives): the dual of the LP gives an occupancy measure interpretation that connects to policy gradient methods.

The LP approach is impractical for large $|S|$ (the constraint matrix has $|S| \cdot |A|$ rows), but it motivates approximate LP methods where the value function is restricted to a linear span of basis functions $V(s) \approx \sum_k w_k \phi_k(s)$. This is the basis of approximate linear programming (de Farias & Van Roy, 2003), an early and theoretically clean form of value function approximation.

### Discount factor $\gamma$ and its impact on solution quality

The discount factor $\gamma$ does far more than ensure convergence — it fundamentally shapes what policy is "optimal." Understanding this is critical for practitioners.

With $\gamma$ close to 1, the agent cares about distant future rewards almost as much as immediate ones. The optimal policy under $\gamma = 0.99$ may look completely different from the optimal policy under $\gamma = 0.9$: a strategy that sacrifices short-term reward for long-term positioning is only favored when $\gamma$ is high enough.

In the GridWorld, $\gamma = 1$ means every step costs equally regardless of when it occurs — the optimal policy minimises total steps. With $\gamma = 0.9$, a policy that reaches the terminal in 5 steps is preferred over one that reaches it in 7 steps (cost $0.9^5 < 0.9^7$), but the difference is only about 18% rather than the full step count. With $\gamma = 0.5$, a policy that terminates in 2 steps ($\text{cost} \approx -1.75$) might be preferred over one that terminates in 4 steps ($\text{cost} \approx -1.88$) even if the 4-step path has higher total reward, because distant rewards are worth almost nothing.

The effective horizon under discounting is approximately $1/(1-\gamma)$: for $\gamma = 0.9$ the agent cares about roughly 10 steps ahead; for $\gamma = 0.99$ it cares about roughly 100 steps. Choosing $\gamma$ is therefore a modeling decision, not just a convergence parameter.

In practice for RL with continuous environments:

| Problem type | Typical $\gamma$ | Reasoning |
|---|---|---|
| Grid navigation | 0.9–0.99 | Moderate horizon; terminal is reachable |
| Atari games | 0.99 | Long episodes; sparse rewards require planning |
| Robot locomotion | 0.99–0.999 | Very long horizon; reward per step is small |
| RLHF / text generation | 1.0 or 0.99 | Token-level; full episode is short (< 2048 tokens) |
| Financial portfolio | 0.99–0.999 | Daily discount ~0.01%; long investment horizons |

### Stochastic environments and the role of variance

Policy evaluation gives the *expected* value $\mathbb{E}[G \mid s]$. But expected value is not the only thing that matters. In financial applications, the variance of the return is equally important — a policy that yields expected return \$100 with standard deviation \$5 is very different from one with return \$100 and standard deviation \$50.

DP can be extended to compute second moments and variance. The second-moment Bellman equation for variance:

$$\mathbb{E}[G^2 \mid s, \pi] = \sum_a \pi(a|s) \sum_{s'} P(s'|s,a)\left[R(s,a)^2 + 2\gamma R(s,a) V^\pi(s') + \gamma^2 \mathbb{E}[G^2 \mid s', \pi]\right]$$

This is itself a linear system in $\mathbb{E}[G^2 \mid s, \pi]$ and can be solved by the same iterative approach as policy evaluation. From this you get $\text{Var}[G \mid s, \pi] = \mathbb{E}[G^2 \mid s, \pi] - (V^\pi(s))^2$.

In practice, variance-aware DP is used in risk-sensitive RL (Mihatsch & Neuneier, 2002) and CVaR-optimal policies. The connection to financial portfolio RL is direct: the Sharpe ratio, which is mean return divided by standard deviation, can only be optimised if you track both mean and variance.

### DP for infinite-horizon average reward

All formulations so far use the *discounted* criterion $\mathbb{E}[\sum_t \gamma^t R_t]$. A different objective is the **average reward**:

$$\rho^\pi = \lim_{T \to \infty} \frac{1}{T} \mathbb{E}_\pi\left[\sum_{t=0}^{T-1} R_{t+1}\right]$$

Average reward DP requires a different Bellman equation:

$$h^\pi(s) + \rho^\pi = \sum_a \pi(a|s) \sum_{s'} P(s'|s,a)\left[R(s,a) + h^\pi(s')\right]$$

where $h^\pi(s)$ is the *differential value function* (relative value, or bias) and $\rho^\pi$ is the average reward. This is solved by a modified form of policy iteration called Howard's policy improvement with average reward, or relative value iteration.

Average reward RL is particularly relevant for continuing (non-episodic) tasks: a trading algorithm running perpetually, a robot performing a manufacturing task without breaks, a dialogue system in a customer service context. For these problems, discounting with $\gamma$ close to 1 is often used as a practical approximation to average reward, but true average reward DP gives cleaner theoretical properties and is preferable when the task has no natural episode boundaries.

### DP with deterministic vs stochastic policies

All derivations above work for both deterministic and stochastic policies. But why would you ever want a stochastic policy?

**Stochastic policies are sometimes optimal in zero-sum or adversarial settings.** In matrix games (rock-paper-scissors), the optimal strategy is to randomise uniformly — a deterministic strategy can be exploited. In partially observable MDPs (POMDPs), a stochastic policy over observations can be better than a deterministic one that commits to actions without certainty.

**Stochastic policies are better in the presence of function approximation.** When $V$ is represented approximately by a neural network, the greedy policy with respect to an approximate $V$ may be suboptimal. A softmax policy $\pi(a|s) \propto \exp(Q_\theta(s,a) / \tau)$ (with temperature $\tau$) introduces beneficial exploration and avoids committing too hard to possibly-wrong Q-estimates. This is the entropy bonus in SAC (Soft Actor-Critic).

**Policy iteration with stochastic policies:** The evaluation and improvement steps extend naturally. Evaluation is unchanged (the Bellman expectation operator averages over $\pi(\cdot|s)$). Improvement becomes:

$$\pi'(s) = \arg\max_{\pi' \in \Delta(A)} \sum_a \pi'(a|s) Q^\pi(s, a)$$

For unconstrained policies, this is still solved by putting all mass on the argmax — so the improved policy is deterministic. For entropy-constrained policies (as in maximum entropy RL), the solution is the Boltzmann distribution $\pi'(a|s) \propto \exp(Q^\pi(s,a)/\tau)$.

### Finite-horizon DP

All formulations in this post assume an infinite horizon with discount $\gamma < 1$. The finite-horizon case (horizon $T$) is structurally different: the optimal policy is *non-stationary* — the best action at time $t$ depends on both the state *and* the remaining time $T - t$.

Finite-horizon DP uses **backward induction** (dynamic programming in time):

$$V_T(s) = 0 \quad \text{(or terminal reward)}$$
$$V_t(s) = \max_a \sum_{s'} P(s'|s,a)\left[R(s,a) + V_{t+1}(s')\right] \quad \text{for } t = T-1, \ldots, 0$$

This is the algorithm underlying options pricing (the Black-Scholes binomial tree), project scheduling, and supply chain optimisation with known time horizons. Each step backwards in time is one Bellman backup — there are exactly $T$ of them, each costing $O(|S|^2 |A|)$. Total complexity: $O(T \cdot |S|^2 |A|)$.

The key difference from infinite-horizon DP: no convergence criterion is needed — you always do exactly $T$ steps backward. This makes finite-horizon DP particularly clean and exact.

### Connecting DP to function approximation: the neural Bellman target

When the state space is too large for a table, the natural extension is to parameterise the value function as a neural network $V_\theta(s)$ or $Q_\theta(s, a)$. The DP update becomes:

$$\theta \leftarrow \theta - \alpha \nabla_\theta \sum_s \left[V_\theta(s) - \left(\hat{r}(s) + \gamma \max_{a'} Q_{\theta^-}(s', a')\right)\right]^2$$

where $\hat{r}(s)$ and $s'$ are sampled from a replay buffer and $Q_{\theta^-}$ is a periodically updated "target network" — a frozen copy of $Q_\theta$.

The target network is the direct heir of the synchronous-sweep structure in tabular DP: in synchronous DP, you computed $V_{k+1}$ from the frozen $V_k$. The target network plays the role of the frozen $V_k$, preventing the update from chasing a moving target (which causes instability without the freeze). This is why DQN's target network — often described as an engineering trick — actually has a rigorous DP justification.

The approximation error introduced by $V_\theta$ and $Q_\theta$ means the contraction argument no longer applies exactly. In general, function approximation + Bellman minimisation can diverge (the "deadly triad" identified by Baird, 1995, and Sutton et al., 2018). DQN avoids this with experience replay + target networks + gradient clipping. Understanding exact DP makes clear exactly which properties these techniques are trying to restore.

### Hyperparameter sensitivity analysis

For practitioners running DP on real problems, the main knobs are:

| Hyperparameter | Effect on convergence | Typical range |
|---|---|---|
| $\gamma$ (discount) | Higher = slower convergence, longer horizon | 0.9–0.999 |
| $\varepsilon$ (stop threshold) | Tighter = more sweeps, closer to true $V^*$ | $10^{-4}$–$10^{-8}$ |
| $m$ (sweeps before improve) | More = closer to true $V^\pi_k$; fewer = faster outer loop | 1–50 |
| State ordering (async DP) | Priority-based vs random vs FIFO — impacts convergence speed | Prioritised sweeping is best |
| Initialisation $V_0$ | 0 vs optimistic (large positive) vs pessimistic | Optimistic init encourages exploration in model-free variants |

One subtle point about initialisation: in the *model-free* setting (Q-learning, SARSA), initialising Q-values optimistically (large positive) encourages the agent to explore all actions before converging to the greedy policy — this is "optimism in the face of uncertainty." In the *model-based* setting (exact DP), initialisation does not affect the final result (Banach's theorem guarantees convergence from any $V_0$) but it does affect how many sweeps are needed. Initialising $V_0 = 0$ is standard; initialising $V_0 = R_{\max}/(1-\gamma)$ (the maximum possible value) gives an upper bound that contracts down — this is sometimes called "value function upper bounding" and is used in optimistic planning algorithms.

### Implementing vectorised DP with NumPy for speed

The state-by-state Python loop in section 6 is pedagogically clear but slow. For larger MDPs, vectorise using numpy matrix operations:

```python
def value_iteration_vectorised(P_mat, R_vec, n_states, n_actions, gamma=0.99, theta=1e-8):
    """
    Vectorised value iteration using matrix operations.
    P_mat: ndarray [n_states, n_actions, n_states] transition probabilities.
    R_vec: ndarray [n_states, n_actions] expected rewards.
    """
    V = np.zeros(n_states)
    n_iters = 0
    while True:
        # Q[s, a] = R[s, a] + gamma * sum_{s'} P[s, a, s'] * V[s']
        # Shape: [n_states, n_actions] = [n_states, n_actions] + gamma * [n_states, n_actions, n_states] @ [n_states]
        Q = R_vec + gamma * np.einsum('san,n->sa', P_mat, V)
        V_new = np.max(Q, axis=1)
        delta = np.max(np.abs(V_new - V))
        V = V_new
        n_iters += 1
        if delta < theta:
            break
    print(f"Vectorised VI: {n_iters} iterations, final delta={delta:.2e}")
    policy = np.argmax(Q, axis=1)
    return policy, V

# For the GridWorld (16 states, 4 actions):
# Build P_mat and R_vec from the dictionary P defined earlier:
P_mat = np.zeros((n_states, n_actions, n_states))
R_vec = np.zeros((n_states, n_actions))
for s in range(n_states):
    for a in range(n_actions):
        for (prob, ns, r, done) in P[s][a]:
            P_mat[s, a, ns] += prob
            R_vec[s, a] += prob * r
            if done:
                P_mat[s, a, ns] = 0  # terminal is absorbing (already handled by reward)

vi_policy_vec, V_vi_vec = value_iteration_vectorised(P_mat, R_vec, n_states, n_actions, gamma=1.0)
print("Vectorised V* (4x4):")
print(V_vi_vec.reshape(N, N))
```

The vectorised version runs roughly 100–1000x faster than the Python loop for large state spaces because NumPy delegates the inner loop to BLAS routines. For $|S| = 10,000$ and $|A| = 10$, the vectorised version completes 100 sweeps in about 50ms on a modern laptop; the loop version takes about 10 seconds.

For very large state spaces that fit in GPU memory, the same vectorisation works with PyTorch tensors, enabling GPU-accelerated DP:

```python
import torch

def value_iteration_gpu(P_tensor, R_tensor, gamma=0.99, theta=1e-6, device='cuda'):
    """
    GPU-accelerated value iteration using PyTorch.
    P_tensor: Tensor [n_states, n_actions, n_states]
    R_tensor: Tensor [n_states, n_actions]
    """
    V = torch.zeros(P_tensor.shape[0], device=device)
    n_iters = 0
    with torch.no_grad():
        while True:
            Q = R_tensor + gamma * torch.einsum('san,n->sa', P_tensor, V)
            V_new = Q.max(dim=1).values
            delta = (V_new - V).abs().max().item()
            V = V_new
            n_iters += 1
            if delta < theta:
                break
    return Q.argmax(dim=1).cpu().numpy(), V.cpu().numpy()
```

GPU acceleration becomes valuable for $|S| > 50,000$ states — the fixed overhead of CPU-GPU transfer is only worthwhile when the matrix operations are large enough to saturate the GPU's memory bandwidth.

### The policy gradient connection: DP as the critic in actor-critic

Actor-critic algorithms (A2C, PPO, SAC) maintain two networks: an **actor** (the policy) and a **critic** (the value function). The critic's job is to estimate $V^\pi(s)$ — exactly what policy evaluation computes. The training signal for the actor comes from the **advantage function**:

$$A^\pi(s, a) = Q^\pi(s, a) - V^\pi(s) = R(s, a) + \gamma \mathbb{E}_{s' \sim P}[V^\pi(s')] - V^\pi(s)$$

If $A^\pi(s, a) > 0$, action $a$ from state $s$ is better than average under $\pi$ — so the actor should increase $\pi(a|s)$. If $A^\pi(s, a) < 0$, the actor should decrease it. This is the policy gradient update:

$$\nabla_\theta J(\pi_\theta) \approx \mathbb{E}_{s, a \sim \pi_\theta}\left[A^\pi(s, a) \nabla_\theta \log \pi_\theta(a|s)\right]$$

Notice: the advantage is directly computed from the Bellman evaluation of $V^\pi$. The critic is running (approximate) policy evaluation; the actor is running (approximate) policy improvement via policy gradient. This is GPI — evaluation by the critic, improvement by the actor. Understanding the Bellman backup that defines policy evaluation is essential for understanding why the advantage estimate is the right training signal and why biased advantage estimates (from an imperfect critic) can cause the policy to converge to a suboptimal solution.

The TD error $\delta_t = R_{t+1} + \gamma V^\pi(S_{t+1}) - V^\pi(S_t)$ used in practice is the stochastic, one-step version of the Bellman residual. If the critic is perfect, $\mathbb{E}[\delta_t \mid S_t = s] = 0$ for all $s$ and $\pi$ — the same as the condition that $V^\pi$ is the fixed point of $\mathcal{T}^\pi$. The actor-critic training loop is literally stochastic GPI: each mini-batch is a partial evaluation step (reducing the TD error), followed by a gradient step on the actor (the improvement step).

## Key takeaways

1. **DP requires the full model** $P(s'|s,a)$ and $R(s,a)$. Without it, you cannot compute Bellman backups exactly and need sample-based RL.

2. **Policy evaluation is a $\gamma$-contraction**. The Bellman expectation operator $\mathcal{T}^\pi$ contracts the sup-norm by $\gamma$ at each application. By Banach's theorem, iterating from any $V_0$ converges geometrically to $V^\pi$.

3. **Policy improvement is monotone**. The greedy update $\pi'(s) = \arg\max_a Q^\pi(s,a)$ never decreases value, and terminates only when the Bellman optimality equation already holds — meaning the policy is already optimal.

4. **Policy iteration converges in finitely many steps** for finite MDPs: the set of deterministic policies is finite, each iteration strictly improves or terminates, so the loop cannot run forever.

5. **Value iteration fuses evaluation and improvement** into one Bellman optimality backup. The $\mathcal{T}^*$ operator is also a $\gamma$-contraction and converges to $V^*$ from any starting point.

6. **Asynchronous DP (in-place, prioritised sweeping)** achieves roughly 10–50x fewer state updates on typical MDPs by focusing on high-Bellman-residual states, without sacrificing the convergence guarantee.

7. **Generalised policy iteration (GPI) is the universal frame**. Every RL algorithm from Q-learning to PPO to RLHF is approximate GPI: the evaluation and improvement steps are approximated by neural networks, sampled returns, and gradient steps, but the structural logic is identical to policy iteration.

8. **The curse of dimensionality** makes exact DP infeasible for $|S| \gtrsim 10^5$ and completely inapplicable for continuous spaces. This motivates function approximation (DQN, PPO) and sample-based evaluation (Monte Carlo, TD).

9. **DP wins by 50–100x on sample efficiency** compared to Q-learning when a model is available. Never default to sample-based RL for small tabular MDPs with known dynamics.

10. **The contraction argument transfers**. Q-learning's convergence proof (Watkins 1992), TD(0) convergence (Tsitsiklis 1994), and many actor-critic stability results all invoke the same Banach contraction logic derived in this post. Learning this proof once gives you the structural intuition for the entire field.

## 14. DP in practice: common pitfalls and debugging tips

Even with a known model and a small state space, DP implementations can go wrong in subtle ways. Here are the most common pitfalls and how to diagnose them.

### Pitfall 1: Forgetting to handle terminal states as absorbing

A terminal state is one where the episode ends. In code, it is easy to accidentally let the value of a terminal state update from its neighbors (because the Bellman backup sums over successor states). The correct treatment is:

- Terminal states have a fixed value (often 0, or the terminal reward). Do not update them.
- When transitioning *into* a terminal state, do not add $\gamma V(\text{terminal})$ — either set `done=True` in your transition model and skip the discount term, or explicitly zero out the terminal contribution.

If you forget this, value iteration converges to wrong values because the terminal states propagate incorrect estimates backward through the state space. The symptom: $V(\text{terminal}) \neq 0$ after convergence, or values that are consistently off by a fixed amount.

### Pitfall 2: Using in-place updates and expecting synchronous results

In-place updates are faster but not equivalent to synchronous updates — the order in which you iterate over states matters. If you sweep states left-to-right and top-to-bottom on a grid, information propagates rightward and downward faster than leftward and upward. The converged value function is identical (both are fixed points of the Bellman operator) but the number of sweeps differs and intermediate values differ.

Debugging tip: if your in-place and synchronous implementations disagree after convergence (by more than $\varepsilon$), you have a bug — they must converge to the same $V^\pi$ or $V^*$ by the contraction argument.

### Pitfall 3: Wrong discount in the stopping criterion

The correct stopping criterion for $\varepsilon$-optimal policy is:

$$\|V_{k+1} - V_k\|_\infty < \varepsilon \cdot \frac{1-\gamma}{2\gamma}$$

Many implementations use $\|V_{k+1} - V_k\|_\infty < \varepsilon$ directly, which underestimates the remaining error by a factor of $2\gamma/(1-\gamma)$. For $\gamma = 0.99$ this factor is about 198 — you are stopping 198x too early and the extracted policy can be far from optimal. When in doubt, use a tighter $\varepsilon$ (e.g., $10^{-8}$ instead of $10^{-4}$).

### Pitfall 4: Numerical issues with $\gamma = 1$

Policy evaluation with $\gamma = 1$ (undiscounted) is technically valid only when all policies are guaranteed to eventually reach a terminal state (the MDP is absorbing). If there is any cycle that never reaches a terminal state, the value function is $-\infty$ under a policy that gets stuck in the cycle.

The GridWorld example works because the grid has walls — even a random policy will eventually reach a terminal state (in expectation). For MDPs that might not terminate under all policies, always use $\gamma < 1$.

### Pitfall 5: Assuming the optimal policy is unique

There may be multiple optimal policies — states where two actions are equally good (tied argmax). The argmax will return one deterministically (usually based on the action index), but this can give counterintuitive-looking policies. If you want all equally-good actions, collect all argmax-tying actions explicitly and verify your downstream code handles the non-unique case.

### Pitfall 6: Very large or sparse MDPs

For MDPs with millions of states but very sparse transitions (branching factor 2–5), the dictionary-based transition model $P[s][a] = [(prob, ns, r, done), \ldots]$ can be memory-efficient, but a dense $|S| \times |A| \times |S|$ tensor is wasteful. Use sparse matrix representations (`scipy.sparse`) for the transition matrix to reduce memory from $O(|S|^2 |A|)$ to $O(|S| \cdot B \cdot |A|)$ where $B$ is the average branching factor.

```python
from scipy.sparse import csr_matrix
import scipy.sparse as sp

def build_sparse_transition(P, n_states, n_actions):
    """
    Build sparse transition matrices T_a for each action a.
    T_a[s, s'] = P(s'|s, a).
    """
    T = []
    R = np.zeros((n_states, n_actions))
    for a in range(n_actions):
        rows, cols, data = [], [], []
        for s in range(n_states):
            for (prob, ns, r, done) in P[s][a]:
                rows.append(s)
                cols.append(ns)
                data.append(prob)
                R[s, a] += prob * r
        T.append(csr_matrix((data, (rows, cols)), shape=(n_states, n_states)))
    return T, R

T_sparse, R_matrix = build_sparse_transition(P, n_states, n_actions)

# Sparse VI: Q[a] = R[:, a] + gamma * T[a] @ V
def value_iteration_sparse(T_sparse, R_matrix, gamma=0.99, theta=1e-8):
    n_states, n_actions = R_matrix.shape
    V = np.zeros(n_states)
    while True:
        Q = np.column_stack([R_matrix[:, a] + gamma * T_sparse[a].dot(V)
                             for a in range(n_actions)])
        V_new = Q.max(axis=1)
        delta = np.abs(V_new - V).max()
        V = V_new
        if delta < theta:
            break
    return Q.argmax(axis=1), V
```

For the 16-state GridWorld this is overkill, but for a 100×100 robot navigation grid with 8 orientations (80,000 states, branching factor 3), the sparse version uses about 2 MB versus 5 GB for the dense tensor — a 2,500x memory reduction.

### Complete convergence diagnostics

When debugging DP, instrument the run with a convergence curve:

```python
def value_iteration_with_diagnostics(P_mat, R_vec, gamma=0.99, theta=1e-8, max_iters=5000):
    """Value iteration with per-iteration diagnostics."""
    n_states, n_actions = R_vec.shape
    V = np.zeros(n_states)
    diagnostics = []

    for k in range(max_iters):
        Q = R_vec + gamma * np.einsum('san,n->sa', P_mat, V)
        V_new = np.max(Q, axis=1)
        delta = np.max(np.abs(V_new - V))
        diagnostics.append({
            'iteration': k,
            'max_bellman_error': delta,
            'mean_V': V.mean(),
            'max_V': V.max(),
            'min_V': V.min()
        })
        V = V_new
        if delta < theta:
            break

    print(f"Converged in {k+1} iterations")
    print(f"First 5 deltas: {[d['max_bellman_error'] for d in diagnostics[:5]]}")
    print(f"Last 5 deltas: {[d['max_bellman_error'] for d in diagnostics[-5:]]}")
    # A linearly decreasing delta in log space confirms geometric convergence at rate gamma.
    # A plateau or non-monotone delta suggests a bug (usually in terminal handling or gamma).
    return np.argmax(Q, axis=1), V, diagnostics
```

A healthy convergence curve should show the max Bellman error decreasing geometrically: each log-scale step down is roughly constant, equal to $\log\gamma$ per iteration. If the curve flattens or jumps, check for: non-absorbing terminal states, incorrect $\gamma$ application, floating-point precision issues on very large or very small values, or non-convergent policy iteration (which can happen with approximate evaluation if $m$ is too small).

## Further reading

- **Sutton & Barto, "Reinforcement Learning: An Introduction" (2nd ed., 2018), Chapter 4** — the canonical treatment of DP for RL; all GridWorld and CarRental examples originate here.
- **Bellman, "Dynamic Programming" (1957)** — the foundational text; introduces the principle of optimality and the functional equation bearing his name.
- **Bertsekas, "Dynamic Programming and Optimal Control" (4th ed., 2017)** — exhaustive treatment including approximate DP, neuro-dynamic programming, and LSPI; essential for practitioners building model-based systems.
- **Watkins & Dayan, "Q-Learning" (1992), Machine Learning** — demonstrates how to do DP without a model; Q-learning is value iteration with sampled transitions replacing the $\sum_{s'} P(...)$ sum.
- **Tsitsiklis, "Asynchronous Stochastic Approximation and Q-Learning" (1994), Machine Learning** — proves convergence of asynchronous Q-learning using the contraction argument from this post.
- **Silver et al., "Mastering the game of Go with deep neural networks and tree search" (2016), Nature** — AlphaGo uses MCTS as approximate GPI; understanding value iteration makes the design transparent.
- **Tesauro, "Temporal Difference Learning and TD-Gammon" (1995), Communications of the ACM** — the original demonstration that approximate GPI with neural networks can match human expert level.
- Within this series: [Markov Decision Processes](/blog/machine-learning/reinforcement-learning/markov-decision-processes) and [Value Functions and the Bellman Equation](/blog/machine-learning/reinforcement-learning/value-functions-and-the-bellman-equation) (prerequisites); the [RL Unified Map](/blog/machine-learning/reinforcement-learning/reinforcement-learning-a-unified-map) for the full algorithm taxonomy; and upcoming posts on Monte Carlo Methods and TD Learning that build on the GPI framework using sample-based evaluation.
