---
title: "Value functions and the Bellman equation: the core of every RL algorithm"
date: "2026-06-22"
publishDate: "2026-06-22"
description: "A rigorous yet accessible derivation of V, Q, and A value functions and the Bellman equations, with a hand-solved GridWorld example and NumPy implementation."
tags:
  [
    "reinforcement-learning",
    "deep-learning",
    "bellman-equation",
    "value-functions",
    "dynamic-programming",
    "q-learning",
    "actor-critic",
    "markov-decision-process",
    "machine-learning",
    "policy-gradient",
  ]
category: "machine-learning"
subcategory: "Reinforcement Learning"
author: "Hiep Tran"
featured: true
readTime: 55
image: "/imgs/blogs/value-functions-and-the-bellman-equation-1.png"
---

There is a CartPole agent that has never seen a good run. It flails the pole left, overcorrects right, loses balance in eight steps, and restarts. You watch it fail a hundred times in a row and think: it has no idea which states are dangerous and which are safe, which actions buy it time and which seal its fate. The agent is flying blind.

Now there is a second agent. It has the same network, the same environment, but it has learned something its twin has not: a **value function**. For every position and velocity of the pole, it carries a number that says "from here, I expect to survive for about 487 more timesteps." When it stands perfectly balanced, that number is high. When the pole is already tilting past 10 degrees, the number drops sharply. Every action it takes is informed by how that number will change — reach for the action that leads toward higher numbers, flee from actions that lead toward lower ones.

That number is the heart of reinforcement learning. It is called the **state-value function**, $V^\pi(s)$, and together with the **action-value function** $Q^\pi(s, a)$ and the **advantage function** $A(s, a)$, it powers every practical RL algorithm you will encounter — DQN, PPO, SAC, A3C, and every deep RL system that has beaten humans at Atari, Go, StarCraft, or protein folding. The mathematical machinery that defines these value functions, the **Bellman equations**, is the reason RL can learn from sequential experience at all.

This post derives everything from scratch. We start from the definition of cumulative return, derive the Bellman expectation equation step by step, prove that the Bellman optimality operator is a contraction that guarantees convergence, and then solve a concrete three-state GridWorld entirely by hand. We also implement tabular Bellman evaluation, value iteration, policy iteration, and greedy policy extraction in NumPy — plus a minimal DQN agent in PyTorch — so you can run every piece yourself. By the end, you will be able to look at any RL paper — DQN in 2015, PPO in 2017, SAC in 2018 — and immediately understand what Bellman equation it is solving, how it approximates it, and why that approximation works. You will also know when the Bellman equation is the wrong tool and what to reach for instead.

The post assumes you are familiar with the RL loop (agent, environment, reward, policy) introduced in [What is reinforcement learning](/blog/machine-learning/reinforcement-learning/what-is-reinforcement-learning) and the MDP formalism from [Markov decision processes](/blog/machine-learning/reinforcement-learning/markov-decision-processes). If those are fresh, everything here will connect directly. If you are coming from a supervised-learning background, replace "value function" with "regression target" and "Bellman equation" with "recursive regression with the target depending on the model's own output" — ugly but accurate — and you will follow the math.

Figure 1 shows the fundamental structure we are building toward: a one-step backup from state $s$, where the value function aggregates over all reachable next states through probability-weighted rewards.

![Backup diagram showing state s expanding over two actions and three next states with transition probabilities and rewards labeled at each edge](/imgs/blogs/value-functions-and-the-bellman-equation-1.png)

---

## 1. Return: the raw thing we want to predict

Before defining any value function, we need to agree on what we are predicting. From the [MDP framework introduced in the previous post](/blog/machine-learning/reinforcement-learning/markov-decision-processes), an agent at time $t$ experiences a trajectory:

$$S_t, A_t, R_{t+1}, S_{t+1}, A_{t+1}, R_{t+2}, S_{t+2}, \ldots$$

The **discounted return** $G_t$ is the total reward from timestep $t$ onward, with future rewards discounted by a factor $\gamma \in [0, 1)$:

$$G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}$$

The discount factor $\gamma$ does two things simultaneously. Mathematically, it ensures the sum converges when rewards are bounded (if $|R| \leq R_{\max}$, then $G_t \leq R_{\max} / (1 - \gamma)$). Practically, it encodes a preference for sooner rewards: a reward of 1.0 now is worth more than a reward of 1.0 ten steps later. Setting $\gamma = 0$ gives a purely myopic agent that only cares about the next reward. Setting $\gamma$ close to 1.0 gives a far-sighted agent that treats rewards fifty steps away as nearly as important as rewards right now.

There is a crucial recursive identity that everything else builds on:

$$G_t = R_{t+1} + \gamma G_{t+1}$$

Read this aloud: "the return from now equals the immediate reward plus the discounted return from the next step." This one line, repeated across states and timesteps, is the entire Bellman equation in embryonic form.

The choice of whether to use discounted returns (finite $\gamma < 1$) or undiscounted returns ($\gamma = 1$) deserves a moment's attention. Undiscounted returns are mathematically cleaner — the agent simply maximizes the total sum of all rewards — but only make sense for episodic tasks that must terminate in finite time. If an episode can run forever and rewards are non-negative, the return $G_t$ diverges. Discounting resolves this by making distant rewards exponentially less important. In practice, even for episodic tasks, practitioners often use $\gamma < 1$ (say 0.99) because it provides a form of implicit regularization: the agent focuses on what it can achieve in the near term rather than building elaborate multi-step plans that depend on rare long-horizon coincidences. The cost is a slight bias toward short-term behavior, which is acceptable if the episode length is short relative to the effective horizon $1/(1-\gamma)$.

A second important subtlety is the **average reward** formulation, used in continuing tasks. Instead of discounting, the agent maximizes the long-run average reward per step:

$$\rho^\pi = \lim_{T \to \infty} \frac{1}{T} \mathbb{E}\!\left[\sum_{t=0}^T R_t \;\middle|\; \pi\right]$$

The Bellman equations for the average-reward setting look slightly different — they subtract $\rho^\pi$ from each step — but the same contraction arguments apply. We will not use average-reward in this post, but it is the right formulation for applications like network scheduling, where the task never ends and discounting would introduce artificial preference for near-term queues.

---

## 2. The state-value function $V^\pi(s)$

The return $G_t$ is a random variable. It depends on which actions the policy samples, which next states the environment transitions to, and which rewards those transitions yield. Two trajectories starting from the same state $s$ can produce different returns.

We want a single number per state that summarizes "how good is it to be here, under policy $\pi$?" The answer is to take the expectation:

$$V^\pi(s) \;=\; \mathbb{E}_\pi\!\left[G_t \;\middle|\; S_t = s\right]$$

This is the **state-value function** (or V-function) for policy $\pi$. It averages over the randomness of both the policy (stochastic action selection) and the environment (stochastic transitions), and gives you the expected cumulative return when you start in state $s$ and follow $\pi$ forever after.

Intuition: $V^\pi(s)$ is a score card for each state. In CartPole, the balanced upright state might have $V^\pi = 487$, meaning an agent following $\pi$ from that position will collect about 487 more reward units before termination. A tilted, already-failing state might have $V^\pi = 3$. The policy should steer toward high-V states.

Two boundary conditions are worth noting. For a **terminal state** $s_{\text{term}}$, we define $V^\pi(s_{\text{term}}) = 0$ by convention (there is no future to collect). For the **episodic vs. continuing** distinction: in episodic tasks (CartPole, Atari), episodes end and $\gamma < 1$ keeps the sum finite; in continuing tasks (network routing, power grid control), $\gamma < 1$ is required to prevent infinite returns.

When you think about a concrete CartPole example: the balanced state near the center might have $V^\pi \approx 450$ with a well-trained policy and $\gamma = 0.99$ (an expected ~450 steps of survival). A state where the pole is already tilted at 15 degrees might have $V^\pi \approx 12$ — the agent can correct a little but is mostly doomed. The difference between these two numbers is what the policy exploits: always take the action that moves you from the 12-value state toward the 450-value state.

---

## 3. The action-value function $Q^\pi(s, a)$

The V-function conditions only on where you are. Sometimes we need to evaluate a specific choice: "how good is it to take action $a$ from state $s$, and then follow $\pi$ afterward?"

$$Q^\pi(s, a) \;=\; \mathbb{E}_\pi\!\left[G_t \;\middle|\; S_t = s,\; A_t = a\right]$$

This is the **action-value function** (or Q-function). The agent commits action $a$ at the current step regardless of what $\pi(a|s)$ says, then follows $\pi$ from the next step onward.

The relationship between $V^\pi$ and $Q^\pi$ is immediate. Since $V^\pi(s)$ is the expected return under $\pi$, and $\pi$ selects action $a$ with probability $\pi(a|s)$:

$$V^\pi(s) = \sum_{a \in \mathcal{A}} \pi(a|s)\, Q^\pi(s, a) = \mathbb{E}_{a \sim \pi}\!\left[Q^\pi(s, a)\right]$$

This says: the state value is the policy-weighted average of action values. Equivalently, $Q^\pi(s, a)$ evaluates what happens when you deviate from $\pi$ for one step and then return to it.

Why do we need both V and Q? In model-based settings where we know the transition function $P(s'|s,a)$, we can compute $Q$ from $V$ and therefore only need to store $V$. In model-free settings — which is most of modern RL — we do not have $P$, so we must estimate $Q$ directly from experience. DQN, SARSA, and most practical Q-learning algorithms maintain a Q-table or Q-network.

Concretely, on CartPole-v1 with four continuous observations (cart position, cart velocity, pole angle, pole angular velocity), a discrete action space of size 2 (push left or push right), and a neural Q-network, the table of $Q(s, a)$ values is effectively infinite (continuous state space) and must be approximated by the network. But the Q-value structure remains: $Q(s, \text{left})$ vs. $Q(s, \text{right})$ at each observed state, and the greedy policy selects whichever is higher. The network generalizes across nearby states through shared weights, which is the entire reason deep Q-networks work on high-dimensional inputs.

---

## 4. The advantage function $A(s, a)$

The V-function gives the average performance from state $s$. The Q-function gives the performance of a specific action from state $s$. Their difference is the **advantage function**:

$$A^\pi(s, a) = Q^\pi(s, a) - V^\pi(s)$$

![The three value functions V, Q, and A stacked with their definitions, showing that A equals Q minus V and is the signal used for low-variance policy gradients in PPO and actor-critic methods](/imgs/blogs/value-functions-and-the-bellman-equation-2.png)

The advantage measures how much better (or worse) action $a$ is compared to the average action under $\pi$ from state $s$. Key properties:

- $A^\pi(s, a) > 0$: action $a$ is better than average from $s$.
- $A^\pi(s, a) < 0$: action $a$ is worse than average from $s$.
- $\mathbb{E}_{a \sim \pi}\!\left[A^\pi(s, a)\right] = 0$ always (since $\mathbb{E}[Q] - V = V - V = 0$).
- At an **optimal policy**, $A^*(s, a^*) \geq 0$ for all $a$, with equality for all non-optimal actions.

The advantage function is the signal used in **actor-critic methods** and **PPO**. Instead of updating the policy with the raw return $G_t$ (high variance, because returns can be very large or very small), you subtract the baseline $V^\pi(s)$ and update with the advantage. This does not change the expected gradient direction — subtracting a baseline that does not depend on the action does not introduce bias, by the policy gradient theorem (Williams 1992, REINFORCE) — but it dramatically reduces variance.

To see the variance reduction numerically: on LunarLander-v2, a vanilla REINFORCE implementation with raw returns has a return variance of roughly 50,000 per episode. Adding the V-function baseline as an advantage estimate drops this to around 5,000 — a 10× reduction — while keeping the same expected gradient direction. PPO with GAE ($\lambda = 0.95$) reduces this further to around 200–500, which is why PPO converges to a mean return of ~240 in about 500k steps while REINFORCE may need 5 million or more steps on the same task.

Concretely, in PPO's clipped surrogate objective:

$$L^{\text{CLIP}}(\theta) = \mathbb{E}_t\!\left[\min\!\left(r_t(\theta)\, \hat{A}_t,\; \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\, \hat{A}_t\right)\right]$$

that $\hat{A}_t$ is an estimate of the advantage function. Without it, you would divide by nothing and get the raw return, which is an uncentered, high-variance signal that makes training fragile.

---

## 5. Deriving the Bellman expectation equation for $V^\pi$

This is the most important derivation in reinforcement learning. We are going to expand the definition of $V^\pi(s)$ using the recursive identity for returns.

Start from the definition:

$$V^\pi(s) = \mathbb{E}_\pi\!\left[G_t \;\middle|\; S_t = s\right]$$

Substitute the recursive identity $G_t = R_{t+1} + \gamma G_{t+1}$:

$$V^\pi(s) = \mathbb{E}_\pi\!\left[R_{t+1} + \gamma G_{t+1} \;\middle|\; S_t = s\right]$$

Expand the expectation by summing over actions (from the policy) and next states (from the transition function):

$$V^\pi(s) = \sum_{a} \pi(a|s) \sum_{s'} P(s'|s,a) \left[R(s,a,s') + \gamma\, \mathbb{E}_\pi\!\left[G_{t+1} \;\middle|\; S_{t+1} = s'\right]\right]$$

But $\mathbb{E}_\pi[G_{t+1} | S_{t+1} = s']$ is exactly $V^\pi(s')$ by definition. So:

$$\boxed{V^\pi(s) = \sum_{a} \pi(a|s) \sum_{s'} P(s'|s,a) \left[R(s,a,s') + \gamma V^\pi(s')\right]}$$

This is the **Bellman expectation equation** for $V^\pi$. It says: the value of state $s$ equals the expected immediate reward plus the discounted value of the next state, where expectations are taken over both the policy's action selection and the environment's transition.

The figure below places this derivation in the context of Bellman expectation versus Bellman optimality — two equations that differ by replacing the policy-weighted average with a max.

![Two-column comparison of the Bellman expectation equation averaging over policy actions and the Bellman optimality equation taking a max over actions instead](/imgs/blogs/value-functions-and-the-bellman-equation-3.png)

A compact notation that many papers use writes this as:

$$V^\pi(s) = \mathbb{E}_{a \sim \pi,\; s' \sim P}\!\left[R(s,a,s') + \gamma V^\pi(s')\right]$$

or even more compactly in matrix form for finite MDPs. Define the **policy evaluation system** with $|\mathcal{S}| = n$ states, policy matrix $\Pi$ where $\Pi_{sa} = \pi(a|s)$, and transition matrix $\mathcal{P}^\pi$ where $\mathcal{P}^\pi_{ss'} = \sum_a \pi(a|s) P(s'|s,a)$, and reward vector $\mathbf{r}^\pi$ where $r^\pi_s = \sum_a \pi(a|s) \sum_{s'} P(s'|s,a) R(s,a,s')$. Then:

$$\mathbf{V}^\pi = \mathbf{r}^\pi + \gamma \mathcal{P}^\pi \mathbf{V}^\pi$$

Rearranging: $(I - \gamma \mathcal{P}^\pi)\mathbf{V}^\pi = \mathbf{r}^\pi$, so the exact solution is:

$$\mathbf{V}^\pi = (I - \gamma \mathcal{P}^\pi)^{-1} \mathbf{r}^\pi$$

For small MDPs, we can solve this linear system directly. For large MDPs, we iterate the Bellman equation, as we will see in the worked example.

---

## 6. Deriving the Bellman expectation equation for $Q^\pi$

The derivation for $Q^\pi$ follows analogously. Start from:

$$Q^\pi(s, a) = \mathbb{E}_\pi\!\left[G_t \;\middle|\; S_t = s,\; A_t = a\right]$$

Substitute $G_t = R_{t+1} + \gamma G_{t+1}$ and expand over next states and then next actions:

$$Q^\pi(s, a) = \sum_{s'} P(s'|s,a) \left[R(s,a,s') + \gamma \sum_{a'} \pi(a'|s')\, Q^\pi(s', a')\right]$$

Or, using the V-Q relationship in the inner sum (since $\sum_{a'} \pi(a'|s') Q^\pi(s',a') = V^\pi(s')$):

$$\boxed{Q^\pi(s, a) = \sum_{s'} P(s'|s,a) \left[R(s,a,s') + \gamma V^\pi(s')\right]}$$

This is the Bellman equation for $Q^\pi$. Notice its backup structure: from $(s,a)$ pair, we fan out over environment transitions to $s'$, collect the reward, and add the discounted state value. The figure below illustrates this two-step backup structure.

![Q-value backup diagram starting from the (s,a) pair, transitioning to two next states with probabilities, then expanding over successor actions at each next state](/imgs/blogs/value-functions-and-the-bellman-equation-4.png)

The two Bellman equations are linked. We can substitute to get a pure-V or pure-Q equation:

**Pure Q recursion:**
$$Q^\pi(s, a) = \sum_{s'} P(s'|s,a) \left[R(s,a,s') + \gamma \sum_{a'} \pi(a'|s')\, Q^\pi(s', a')\right]$$

**V from Q:**
$$V^\pi(s) = \sum_a \pi(a|s)\, Q^\pi(s, a)$$

**Q from V (requires model):**
$$Q^\pi(s, a) = \sum_{s'} P(s'|s,a)\!\left[R(s,a,s') + \gamma V^\pi(s')\right]$$

In practice, model-free algorithms maintain Q-tables or Q-networks and update them using the pure-Q recursion, which requires no knowledge of $P$.

---

## 7. The Bellman optimality equations: finding the best policy

The Bellman expectation equations evaluate a *fixed* policy. To find the *best* policy, we need the **Bellman optimality equations**.

The **optimal value functions** are:

$$V^*(s) = \max_\pi V^\pi(s) \quad \text{and} \quad Q^*(s, a) = \max_\pi Q^\pi(s, a)$$

By following the same derivation but replacing the policy-weighted average with a maximum, we get:

$$\boxed{V^*(s) = \max_a \sum_{s'} P(s'|s,a)\!\left[R(s,a,s') + \gamma V^*(s')\right]}$$

$$\boxed{Q^*(s, a) = \sum_{s'} P(s'|s,a)\!\left[R(s,a,s') + \gamma \max_{a'} Q^*(s', a')\right]}$$

The key change: $\sum_a \pi(a|s)$ becomes $\max_a$. Instead of averaging over what the policy would do, we take the best possible action.

Once we have $Q^*$, we are done: the optimal policy is simply:

$$\pi^*(s) = \arg\max_a Q^*(s, a)$$

This is a deterministic greedy policy that picks the highest-Q action at every state. No stochastic sampling, no policy gradient, no actor network needed — if you have $Q^*$ exactly, greedy extraction gives you $\pi^*$ for free.

The relationship between $V^*$ and $Q^*$ is:

$$V^*(s) = \max_a Q^*(s, a)$$

and

$$Q^*(s, a) = \sum_{s'} P(s'|s,a)\!\left[R(s,a,s') + \gamma V^*(s')\right]$$

### Why the Bellman optimality operator is a contraction

This is the theoretical cornerstone that guarantees every Bellman-based algorithm converges. Define the **Bellman optimality operator** $\mathcal{T}^*$ as:

$$(\mathcal{T}^* V)(s) = \max_a \sum_{s'} P(s'|s,a)\!\left[R(s,a,s') + \gamma V(s')\right]$$

We want to show this is a $\gamma$-contraction in the $\ell^\infty$ norm, meaning:

$$\|\mathcal{T}^* V - \mathcal{T}^* U\|_\infty \leq \gamma \|V - U\|_\infty$$

for any two value function estimates $V$ and $U$.

**Proof.** For any state $s$, let $a_V = \arg\max_a [\ldots V\ldots]$ and $a_U = \arg\max_a [\ldots U\ldots]$. Then:

$$(\mathcal{T}^* V)(s) - (\mathcal{T}^* U)(s) \leq \sum_{s'} P(s'|s, a_U)\!\left[\gamma V(s') - \gamma U(s')\right]$$

(we used the fact that the max of a set is no larger than the max if we constrain to any particular action, and the action achieving the max for $U$ is a valid choice). Similarly in the other direction. Taking absolute values and using the $\ell^\infty$ norm:

$$\|(\mathcal{T}^* V)(s) - (\mathcal{T}^* U)(s)\| \leq \gamma \sum_{s'} P(s'|s,a)\, \|V(s') - U(s')\|_\infty \leq \gamma \|V - U\|_\infty$$

Since this holds for all $s$, $\|\mathcal{T}^* V - \mathcal{T}^* U\|_\infty \leq \gamma \|V - U\|_\infty$. $\square$

By the **Banach fixed-point theorem** (also called the contraction mapping theorem), any $\gamma$-contraction on a complete metric space has a unique fixed point and iterating the operator from any starting point converges to it. That fixed point is $V^*$. This theorem is why value iteration converges, why TD learning converges (with appropriate step sizes), and why Q-learning converges: they are all approximating or computing this same contraction.

The figure below illustrates this geometrically.

![Before-after comparison showing an arbitrary value function V with distance 10 to V-star, and after one Bellman operator application the distance shrinks to at most gamma times 10, demonstrating convergence](/imgs/blogs/value-functions-and-the-bellman-equation-7.png)

After $k$ applications, the distance to $V^*$ is at most $\gamma^k \|V_0 - V^*\|_\infty$. With $\gamma = 0.9$ and starting distance 10, after 100 sweeps the distance is at most $0.9^{100} \times 10 \approx 0.000027$. Convergence is guaranteed and exponentially fast.

---

## 8. Iterative policy evaluation

For practical computation with a known model, we do not invert $(I - \gamma \mathcal{P}^\pi)$ directly (that matrix is $|\mathcal{S}| \times |\mathcal{S}|$ and can be enormous). Instead we iterate the Bellman expectation operator:

$$V_{k+1}(s) \leftarrow \sum_{a} \pi(a|s) \sum_{s'} P(s'|s,a)\!\left[R(s,a,s') + \gamma V_k(s')\right]$$

Starting from any $V_0$ (usually all zeros), this sequence converges to $V^\pi$ as $k \to \infty$. Each sweep over all states is called a **Bellman backup** or **Bellman sweep**.

The timeline below shows the convergence on a three-state example over iterations 0, 5, 20, and infinity.

![Timeline showing V(s) estimates at iterations 0, 5, 20, and convergence with the true values, illustrating exponential approach to the fixed point V-pi](/imgs/blogs/value-functions-and-the-bellman-equation-5.png)

Convergence is monotone when starting from $V_0 = 0$ with non-negative rewards: the estimates only increase toward their true values. For general reward functions, the estimates may oscillate slightly but still converge.

---

## 9. Comparing V, Q, and A: a reference table

Before moving to implementation, it helps to consolidate the three value functions side by side.

![Matrix table comparing V-function, Q-function, and advantage A-function across four rows: definition, input, output range, and algorithm usage](/imgs/blogs/value-functions-and-the-bellman-equation-6.png)

| Property | $V^\pi(s)$ | $Q^\pi(s,a)$ | $A^\pi(s,a)$ |
|---|---|---|---|
| **What it measures** | Expected return from state $s$ | Expected return from $(s,a)$ pair | How much better $a$ is vs. average |
| **Inputs** | State only | State + action | State + action |
| **Table size (tabular)** | $|\mathcal{S}|$ | $|\mathcal{S}| \times |\mathcal{A}|$ | $|\mathcal{S}| \times |\mathcal{A}|$ |
| **Mean under $\pi$** | N/A | $V^\pi(s)$ | $0$ (by definition) |
| **Requires model for policy?** | Yes (need $P$ to get $Q$ for greedy) | No (greedy directly from $Q$) | No (greedy from $Q = A + V$) |
| **Key algorithms** | TD(0) critic, A2C baseline | DQN, SARSA, Q-learning | PPO, A3C, GAE |

The advantages column deserves emphasis. The advantage function has mean zero under the current policy, which means it is a **centered** signal. Centered signals have lower variance than uncentered ones. This is why PPO's advantage estimates converge faster than vanilla policy gradient using raw returns, and it is why Generalized Advantage Estimation (GAE) exists as an explicit technique to estimate $A$ with controlled bias-variance tradeoff.

---

## 10. Worked example: solving a 3-state GridWorld by hand

#### Worked example: Bellman policy evaluation on a 3-state chain

Consider a three-state chain MDP: states $\{s_1, s_2, s_3\}$, where $s_3$ is terminal (absorbing). The policy $\pi$ is fixed and deterministic: always move right ($s_1 \to s_2 \to s_3$). The rewards are:

- $R(s_1, \text{right}) = 0$
- $R(s_2, \text{right}) = 1$
- $R(s_3, \text{stay}) = 0$ (terminal, all future reward = 0)

Transitions are deterministic. Discount $\gamma = 0.9$.

**Exact solution via linear system.** The Bellman expectation equations for this deterministic policy are:

$$V^\pi(s_1) = 0 + 0.9 \cdot V^\pi(s_2)$$
$$V^\pi(s_2) = 1 + 0.9 \cdot V^\pi(s_3)$$
$$V^\pi(s_3) = 0$$

Solving bottom-up (since $s_3$ is terminal):
- $V^\pi(s_3) = 0$
- $V^\pi(s_2) = 1 + 0.9 \times 0 = 1.0$
- $V^\pi(s_1) = 0 + 0.9 \times 1.0 = 0.9$

So $\mathbf{V}^\pi = [0.9, 1.0, 0.0]$.

**Iterative solution.** Start with $V_0 = [0, 0, 0]$. Apply Bellman sweeps:

| Iteration | $V(s_1)$ | $V(s_2)$ | $V(s_3)$ |
|---|---|---|---|
| 0 | 0.000 | 0.000 | 0.000 |
| 1 | 0.000 | 1.000 | 0.000 |
| 2 | 0.900 | 1.000 | 0.000 |
| 3 | 0.900 | 1.000 | 0.000 |
| ∞ | **0.900** | **1.000** | **0.000** |

For this simple chain, convergence is exact after 2 iterations (the chain has depth 2). For deeper chains or stochastic transitions, it takes more sweeps.

**Q-values from V.** Since we know the model, we can compute $Q^\pi$:

$$Q^\pi(s_1, \text{right}) = R(s_1, \text{right}) + 0.9 \cdot V^\pi(s_2) = 0 + 0.9 \times 1.0 = 0.9$$
$$Q^\pi(s_2, \text{right}) = R(s_2, \text{right}) + 0.9 \cdot V^\pi(s_3) = 1 + 0.9 \times 0 = 1.0$$

**Advantage values:**

$$A^\pi(s_1, \text{right}) = Q^\pi(s_1, \text{right}) - V^\pi(s_1) = 0.9 - 0.9 = 0$$
$$A^\pi(s_2, \text{right}) = Q^\pi(s_2, \text{right}) - V^\pi(s_2) = 1.0 - 1.0 = 0$$

Both advantages are 0 because the policy is already deterministic — there is only one action, so it is trivially "average." The advantage becomes interesting when the policy is stochastic or when comparing multiple actions.

#### Worked example: Bellman optimality on a branching MDP

Now consider a two-state MDP where the agent has a real choice. State $s_1$ is the only non-terminal state, with two actions:

- Action $a_L$: go to $s_2$ (terminal) with reward $+2.0$
- Action $a_R$: stay at $s_1$ with reward $+1.0$ (discount means this penalizes staying)

$\gamma = 0.9$, $V(s_2) = 0$.

**Bellman optimality for $s_1$:**

$$V^*(s_1) = \max\!\left[R(s_1, a_L) + 0.9 \cdot V^*(s_2),\; R(s_1, a_R) + 0.9 \cdot V^*(s_1)\right]$$

$$V^*(s_1) = \max\!\left[2.0 + 0,\; 1.0 + 0.9 \cdot V^*(s_1)\right]$$

Case 1: choose $a_L$ always. Then $V^*(s_1) = 2.0$. Check: is $1.0 + 0.9 \times 2.0 = 2.8 > 2.0$? Yes, so $a_L$ alone is suboptimal.

Case 2: choose $a_R$ always. Then $V^*(s_1) = 1.0 + 0.9 V^*(s_1)$, so $0.1 V^*(s_1) = 1.0$, giving $V^*(s_1) = 10.0$. Check: is $2.0 < 10.0$? Yes, so staying ($a_R$) is better.

The optimal policy is $\pi^*(s_1) = a_R$, and $V^*(s_1) = 10.0$. This makes sense: collecting $+1.0$ every step forever under $\gamma = 0.9$ gives a geometric series sum of $1.0 / (1 - 0.9) = 10.0$, which beats taking the lump sum of $2.0$ once.

**Key insight:** Bellman optimality equations correctly balance immediate rewards against long-run returns, and the contraction proof tells us that iterating them will find this answer regardless of initialization.

---

## 11. NumPy implementation: tabular Bellman evaluation and policy extraction

Now we move from theory to code. The following implementation covers three things: iterative policy evaluation, value iteration (Bellman optimality), and greedy policy extraction.

```python
import numpy as np
from typing import Optional

# MDP specification
# States: {0, 1, 2, ..., n_states-1}
# Actions: {0, 1, ..., n_actions-1}
# P[s, a, s'] = transition probability
# R[s, a, s'] = reward
# gamma        = discount factor

def bellman_policy_eval(
    policy: np.ndarray,          # shape (n_states, n_actions): policy[s,a] = pi(a|s)
    P: np.ndarray,               # shape (n_states, n_actions, n_states)
    R: np.ndarray,               # shape (n_states, n_actions, n_states)
    gamma: float = 0.9,
    theta: float = 1e-8,         # convergence threshold
    max_iter: int = 10_000,
) -> np.ndarray:
    """Iterative policy evaluation via Bellman expectation backups."""
    n_states = P.shape[0]
    V = np.zeros(n_states)

    for iteration in range(max_iter):
        V_new = np.zeros(n_states)
        for s in range(n_states):
            for a in range(policy.shape[1]):
                pi_sa = policy[s, a]
                if pi_sa == 0:
                    continue
                # Expected reward + discounted next value
                V_new[s] += pi_sa * np.dot(P[s, a], R[s, a] + gamma * V)
        delta = np.max(np.abs(V_new - V))
        V = V_new
        if delta < theta:
            print(f"  Converged at iteration {iteration + 1}, delta={delta:.2e}")
            break
    return V
```

```python
def value_iteration(
    P: np.ndarray,
    R: np.ndarray,
    gamma: float = 0.9,
    theta: float = 1e-8,
    max_iter: int = 10_000,
) -> tuple[np.ndarray, np.ndarray]:
    """Value iteration: solve Bellman optimality equations.
    Returns (V_star, Q_star).
    """
    n_states, n_actions, _ = P.shape
    V = np.zeros(n_states)

    for iteration in range(max_iter):
        # Q[s, a] = sum_{s'} P[s,a,s'] * (R[s,a,s'] + gamma * V[s'])
        Q = np.einsum('sai,sai->sa', P, R[:, :, :] + gamma * V[np.newaxis, np.newaxis, :])
        # V[s] = max_a Q[s, a]
        V_new = Q.max(axis=1)
        delta = np.max(np.abs(V_new - V))
        V = V_new
        if delta < theta:
            print(f"  Value iteration converged at iteration {iteration + 1}")
            break

    # Recompute Q* from V*
    Q_star = np.einsum('sai,sai->sa', P, R + gamma * V[np.newaxis, np.newaxis, :])
    return V, Q_star
```

```python
def extract_greedy_policy(Q_star: np.ndarray) -> np.ndarray:
    """Extract deterministic greedy policy from Q*.
    Returns policy as (n_states, n_actions) one-hot array.
    """
    n_states, n_actions = Q_star.shape
    policy = np.zeros((n_states, n_actions))
    best_actions = np.argmax(Q_star, axis=1)
    policy[np.arange(n_states), best_actions] = 1.0
    return policy


def compute_advantage(
    V: np.ndarray,
    Q: np.ndarray,
    policy: np.ndarray,
) -> np.ndarray:
    """Compute advantage A(s,a) = Q(s,a) - V(s) for all (s,a)."""
    # V has shape (n_states,); expand to (n_states, n_actions) for subtraction
    return Q - V[:, np.newaxis]
```

Now let us demonstrate on the three-state chain:

```python
# Three-state chain: s0 -> s1 -> s2 (terminal)
# Actions: 0 = right (deterministic)
n_states, n_actions = 3, 1

P = np.zeros((n_states, n_actions, n_states))
R = np.zeros((n_states, n_actions, n_states))

# s0 --right--> s1
P[0, 0, 1] = 1.0
R[0, 0, 1] = 0.0

# s1 --right--> s2 (terminal)
P[1, 0, 2] = 1.0
R[1, 0, 2] = 1.0

# s2 is absorbing (self-loop, no reward)
P[2, 0, 2] = 1.0
R[2, 0, 2] = 0.0

# Uniform policy (only one action, so trivially pi(a|s) = 1.0)
policy = np.ones((n_states, n_actions))

gamma = 0.9
V_pi = bellman_policy_eval(policy, P, R, gamma=gamma)
print("V^pi =", V_pi)
# Expected: [0.9, 1.0, 0.0]

V_star, Q_star = value_iteration(P, R, gamma=gamma)
print("V* =", V_star)
print("Q* =", Q_star)

pi_star = extract_greedy_policy(Q_star)
print("Optimal policy (greedy from Q*):", pi_star.argmax(axis=1))

A = compute_advantage(V_pi, Q_star[:, :1], policy)
print("Advantage A(s,a):", A.flatten())
# Expected: [0.0, 0.0, 0.0] for deterministic single-action policy
```

For larger MDPs, a vectorized batch update is more efficient:

```python
def bellman_policy_eval_vectorized(
    policy: np.ndarray,
    P: np.ndarray,
    R: np.ndarray,
    gamma: float = 0.9,
    theta: float = 1e-8,
) -> np.ndarray:
    """Vectorized policy evaluation using einsum — no Python loops over states."""
    n_states = P.shape[0]
    V = np.zeros(n_states)

    # Precompute: expected_reward[s, a] = sum_{s'} P[s,a,s'] * R[s,a,s']
    expected_reward = np.einsum('sai,sai->sa', P, R)
    # transition_matrix[s, a, s'] already stored as P

    for _ in range(100_000):
        # Q[s, a] = expected_reward[s,a] + gamma * sum_{s'} P[s,a,s'] V[s']
        Q = expected_reward + gamma * np.einsum('sai,i->sa', P, V)
        # V[s] = sum_a pi(a|s) * Q(s,a)
        V_new = np.einsum('sa,sa->s', policy, Q)
        if np.max(np.abs(V_new - V)) < theta:
            break
        V = V_new
    return V_new
```

This vectorized version processes all states in a single NumPy call, making it practical for MDPs with thousands of states.

---

## 12. The big picture: how Bellman equations unify all RL algorithms

Every RL algorithm you will encounter is fundamentally a strategy for solving or approximating one of the four Bellman equations ($V^\pi$, $Q^\pi$, $V^*$, $Q^*$). The figure below maps this out.

![Graph showing Bellman equations at the center with arrows to dynamic programming, Monte Carlo, TD learning, and deep RL, all converging to the optimal policy](/imgs/blogs/value-functions-and-the-bellman-equation-8.png)

| Algorithm family | Which Bellman equation | How it approximates | When to use |
|---|---|---|---|
| **Dynamic Programming** (policy eval, value iter, policy iter) | Exact $V^\pi$ or $V^*$ | Exact computation with known model | Small MDPs, known transition function |
| **Monte Carlo** (first-visit MC, every-visit MC) | $Q^\pi$ via full episode returns | Sample full trajectories, average | Episodic tasks, no function approximation needed |
| **Temporal Difference** (TD(0), SARSA, Q-learning) | $V^\pi$ or $Q^\pi$ | Bootstrap off next-step estimate | Large MDPs, online learning, continuous tasks |
| **Deep RL** (DQN, PPO, SAC, A3C, TD3) | $Q^*$ or $V^\pi$ | Neural function approximation + SGD | High-dimensional states/actions, learned features |

The key insight Sutton & Barto emphasize (in *Reinforcement Learning: An Introduction*, second edition): **DP, MC, and TD are not different kinds of RL — they are different ways of solving the same Bellman equations.** DP does it exactly with full sweeps; MC does it by sampling full trajectories; TD does it by bootstrapping off the current estimate of $V$ or $Q$. Deep RL layers a neural network on top of TD. RLHF layers a human reward model on top of deep RL. The Bellman equations are the bedrock that all of them rest on.

This unifying view is not just aesthetically pleasing — it is practically useful. When a deep RL training run diverges or fails to improve, the first question should be: "which part of the Bellman approximation is breaking down?" Is the reward signal too sparse for the TD target to carry useful information? Is the target network's update frequency causing the Bellman residual to chase a moving target? Is the off-policy mismatch between the replay buffer and the current policy too large? Each failure mode has a direct mapping to a property of the Bellman equation, and diagnosing RL failures requires understanding which approximation you are relying on. See the [debugging AI training series](/blog/machine-learning/debugging-training/the-training-debugging-playbook) for a systematic framework for this kind of diagnosis.

From a production standpoint, the Bellman equation is also the lens through which we evaluate whether RL is the right tool at all. If your "environment" is actually a differentiable simulator, you can differentiate through it and use gradient-based planning instead of Bellman backups — no RL needed. If your state space is tiny (a few hundred states), solve the linear system directly and skip the iteration. If the Markov property is badly violated because the agent observes only partial state, the Bellman equation still holds for the underlying MDP, but your observations do not give you the Markov state, so your value function approximation will be systematically wrong. Understanding the Bellman equation's prerequisites — the Markov property, bounded rewards, correct $\gamma$ — tells you when the foundation is sound and when you need a different formulation.

---

## 13. Case studies: Bellman equations in famous systems

### Atari DQN (Mnih et al., 2015)

DeepMind's DQN was the first algorithm to achieve human-level performance on Atari games directly from raw pixels. Its core update is the Bellman optimality equation for $Q^*$, with a neural approximation:

$$\mathcal{L}(\theta) = \mathbb{E}\!\left[\left(R + \gamma \max_{a'} Q(S', a';\, \theta^-) - Q(S, A;\, \theta)\right)^2\right]$$

where $\theta^-$ is a **target network** (a periodically frozen copy of $\theta$). The target network exists because the Bellman operator's contraction proof requires the target to be fixed — if you use the same network to compute both the prediction and the target, you are chasing a moving target, and convergence is no longer guaranteed. The target network approximates "freezing" the contraction operator for a window of updates.

DQN achieved a median score of 121% of human performance on 49 Atari games (Mnih et al. 2015 Nature paper). In Breakout, it learned a tunnel-drilling strategy that human players had independently discovered. The Bellman equation is what made this possible: every update step is a small move toward the optimal Q-function, and after ~50M frames, the contraction has pulled the estimate close enough to play superhuman Breakout.

### AlphaGo / AlphaZero (Silver et al., 2016/2017)

AlphaGo used a combination of a value network (approximating $V^\pi$) and a policy network (approximating $\pi$), trained partly via MCTS-enhanced Q-value estimation. AlphaZero, which learned from self-play alone, maintained explicit value backups via tree search — a form of model-based Bellman backup. The value function here is computed over the game tree, with $V(s) = $ probability of winning from state $s$.

AlphaZero achieved an Elo of ~3200 in chess after just 4 hours of self-play, surpassing Stockfish (Elo ~3400 at the time) at ~3580 — the reported approximate Elo from the 2018 Science paper. The Bellman equation over game trees is the mechanism by which one-step lookahead generalizes to long-horizon winning play.

### PPO and RLHF (InstructGPT, 2022)

InstructGPT (Ouyang et al. 2022) aligned GPT-3 using a PPO loop where the advantage function $A^\pi$ was computed from a separate value head trained via Bellman expectation updates. The value head predicted the expected reward (from a human-trained reward model) given a partial conversation. PPO's clipped objective then pushed the policy to increase probability of high-advantage tokens.

The critical insight: by using $A^\pi$ instead of raw returns, RLHF training is stabilized enough to avoid reward hacking within the relatively short training runs. The advantage function's zero mean means the policy gradient update has no systematic bias toward always increasing or always decreasing all actions — it selectively rewards actions that are genuinely above average.

### Soft Actor-Critic on continuous control (Haarnoja et al., 2018)

SAC introduced the soft Bellman equation that adds an entropy term, changing the value function to simultaneously maximize return and policy entropy:

$$V_{\text{soft}}^\pi(s) = \mathbb{E}_{a \sim \pi}\!\left[Q_{\text{soft}}^\pi(s,a) - \alpha \log \pi(a|s)\right]$$

The entropy coefficient $\alpha$ is automatically tuned via a dual optimization, making SAC nearly hyperparameter-free compared to DDPG or TD3. On MuJoCo HalfCheetah-v4, SAC achieves an average return of approximately 8,000–11,000 in 1 million environment steps (Haarnoja et al. 2018, Table 1), compared to roughly 3,000 for PPO and 5,000–6,000 for TD3 at the same sample budget. The soft value function's entropy bonus is the mechanism by which SAC maintains exploration throughout training — states with low policy entropy (the agent is too confident) get a reduced effective value, nudging the policy to maintain diverse behavior.

### Stable-Baselines3 benchmarks

For reference, SB3's tabular/shallow RL baselines on standard Gymnasium environments:

| Environment | Algorithm | Typical mean return (1M steps) |
|---|---|---|
| CartPole-v1 | PPO | 500.0 (perfect) |
| LunarLander-v2 | PPO | ~240 |
| MountainCar-v0 | DQN | ~-100 |
| Acrobot-v1 | A2C | ~-77 |
| HalfCheetah-v4 | SAC | ~8,000 |

These results are achievable by anyone running SB3 with default hyperparameters. They demonstrate that Bellman-based algorithms converge to good policies within a few million environment steps on the standard benchmark suite.

---

## 14. When to use the Bellman equations (and when not to)

The Bellman equations are not always the right tool. Here is a decisive guide:

**Use Bellman-based methods when:**

- You have a clear numerical reward signal that accumulates over time.
- The task is sequential and decisions interact (future states depend on current actions).
- You need to maximize long-run expected return, not just single-step performance.
- You have computational budget for iterative updates (DP: full sweeps; TD: online updates).

**Use dynamic programming directly (exact Bellman solution) when:**

- You know the transition function $P(s'|s,a)$ and reward function.
- The state space $|\mathcal{S}|$ is small enough for a full table (say, $< 10^6$ states).
- You need exact optimal values for planning, not learning from interaction.
- Examples: elevator scheduling, small inventory control problems, gridworld planning.

**Switch to model-free TD/Q-learning when:**

- You do not know $P(s'|s,a)$ and must learn from experience.
- The environment is real (robot, financial market, game) and cannot be simulated cheaply.
- Off-policy learning is required (learning from demonstrations or replay).

**Switch to deep RL (DQN, PPO, SAC) when:**

- The state space is high-dimensional (images, text, sensor arrays).
- A table is computationally infeasible (Atari has $84 \times 84 \times 4 = 28,224$-dimensional states).
- You need generalization: the agent should handle states it has never seen exactly.

**Avoid Bellman-based RL entirely when:**

- The task is a single-step decision (bandit problems: use UCB or Thompson sampling, not full RL).
- You have a differentiable simulator and can use gradient-based model predictive control.
- The reward function is trivially shaped and a simple supervised learning baseline works (e.g., image classification with a reward for correct labels — just use cross-entropy loss).
- The environment is non-stationary in ways that violate the Markov property badly (use meta-RL or recurrent policies instead).

The Bellman equation assumes the Markov property: the next state depends only on the current state and action, not on the full history. When this fails badly — as it does in partially observed environments, environments with hidden state, or multi-agent settings where other agents' behavior changes the dynamics — the Bellman equation still applies to the underlying (hidden) MDP, but your observations may not capture that state. You then need recurrent networks (LSTM, transformer) to approximate the Markov state, or use POMDP methods.

Specifically for financial and trading environments, the Markov property is often only approximately satisfied — price history, order book depth, and macro regime all affect future prices in ways that a single-step observation cannot capture. This is why RL for trading typically uses a feature vector that includes multiple lags of price and volume, effectively constructing a richer Markov state from the observable history. See the [game theory for markets series](/blog/trading/game-theory) for how strategic considerations further complicate the state representation in order-book environments. When the pseudo-Markov state is well-constructed, Bellman-based Q-learning can learn profitable intraday execution policies, but the quality of that state representation is at least as important as the choice of RL algorithm.

---

## 15. Implementation: Bellman-based Q-learning loop

The Q-learning update rule is the Bellman optimality equation applied sample-by-sample:

$$Q(s, a) \leftarrow Q(s, a) + \alpha\!\left[\underbrace{R + \gamma \max_{a'} Q(s', a')}_{\text{TD target}} - \underbrace{Q(s, a)}_{\text{current estimate}}\right]$$

The term in brackets is the **TD error**: how wrong was your current Q-estimate relative to a one-step Bellman backup. Here is a complete tabular Q-learning implementation on FrozenLake:

```python
import numpy as np
import gymnasium as gym

def run_q_learning(
    env_name: str = "FrozenLake-v1",
    n_episodes: int = 5_000,
    alpha: float = 0.1,         # learning rate
    gamma: float = 0.99,        # discount factor
    epsilon_start: float = 1.0, # exploration rate (decays to epsilon_end)
    epsilon_end: float = 0.01,
    epsilon_decay: float = 0.995,
    seed: int = 42,
) -> tuple[np.ndarray, list[float]]:
    """Tabular Q-learning on a discrete Gymnasium environment."""
    env = gym.make(env_name, is_slippery=True)
    n_states  = env.observation_space.n
    n_actions = env.action_space.n

    # Q-table: initialized to zero
    Q = np.zeros((n_states, n_actions))
    rng = np.random.default_rng(seed)

    episode_returns: list[float] = []
    epsilon = epsilon_start

    for episode in range(n_episodes):
        state, _ = env.reset(seed=seed + episode)
        total_return = 0.0
        done = False

        while not done:
            # Epsilon-greedy action selection
            if rng.random() < epsilon:
                action = env.action_space.sample()           # explore
            else:
                action = int(Q[state].argmax())              # exploit

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Bellman optimality update (Q-learning, off-policy)
            td_target = reward + gamma * Q[next_state].max() * (not terminated)
            td_error  = td_target - Q[state, action]
            Q[state, action] += alpha * td_error

            total_return += reward
            state = next_state

        episode_returns.append(total_return)
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

    env.close()
    return Q, episode_returns


Q_table, returns = run_q_learning()

# Evaluate learned policy
win_rate = sum(r > 0 for r in returns[-500:]) / 500
print(f"Win rate (last 500 episodes): {win_rate:.1%}")
# Typical result: ~73-78% on FrozenLake-v1 with is_slippery=True
```

Contrast this with SARSA, which uses the Bellman **expectation** equation (on-policy):

```python
# SARSA update: uses the action the policy *actually takes* at s'
# (not the max, which would be off-policy Q-learning)
next_action = int(Q[next_state].argmax()) if rng.random() > epsilon else env.action_space.sample()
td_target = reward + gamma * Q[next_state, next_action] * (not terminated)
td_error  = td_target - Q[state, action]
Q[state, action] += alpha * td_error
# next action must be used at next step (on-policy requirement)
```

The single character difference — `Q[next_state].max()` vs `Q[next_state, next_action]` — distinguishes off-policy Q-learning (Bellman optimality) from on-policy SARSA (Bellman expectation). Q-learning converges to $Q^*$; SARSA converges to $Q^\pi$ for the epsilon-greedy policy.

---

## 16. The advantage function in actor-critic: a practical deep RL connection

In modern actor-critic methods like A2C and PPO, the critic learns $V^\pi(s)$ via Bellman expectation updates, and the advantage is estimated as:

$$\hat{A}_t = R_{t+1} + \gamma V(S_{t+1}) - V(S_t) \quad \text{(one-step TD advantage)}$$

This is the **TD error**, which is a biased but low-variance estimate of the advantage. Generalized Advantage Estimation (GAE, Schulman et al. 2015) extends this with a $\lambda$-weighted blend of multi-step TD errors:

$$\hat{A}_t^{\text{GAE}} = \sum_{l=0}^\infty (\gamma\lambda)^l \delta_{t+l}, \quad \delta_t = R_{t+1} + \gamma V(S_{t+1}) - V(S_t)$$

Here $\lambda \in [0,1]$ controls the bias-variance tradeoff: $\lambda = 0$ gives pure one-step TD (low variance, high bias); $\lambda = 1$ gives Monte Carlo returns (zero bias, high variance). The critic learns via minimizing:

$$\mathcal{L}_V(\phi) = \mathbb{E}_t\!\left[\left(V_\phi(S_t) - \hat{R}_t\right)^2\right]$$

where $\hat{R}_t = \sum_{k=0}^{T-t-1} \gamma^k R_{t+k+1}$ is the empirical return. This is a direct application of the Bellman expectation equation: push the critic's prediction toward the target.

```python
import torch
import torch.nn as nn
import torch.optim as optim

class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 64):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.actor  = nn.Linear(hidden, n_actions)   # logits
        self.critic = nn.Linear(hidden, 1)            # V(s)

    def forward(self, obs: torch.Tensor):
        h = self.shared(obs)
        return self.actor(h), self.critic(h).squeeze(-1)


def compute_td_advantage(
    rewards: torch.Tensor,     # shape (T,)
    values: torch.Tensor,      # shape (T,)
    next_value: float,
    gamma: float = 0.99,
    lam: float = 0.95,         # GAE lambda
) -> torch.Tensor:
    """Generalized Advantage Estimation."""
    T = len(rewards)
    advantages = torch.zeros(T)
    last_gae = 0.0

    for t in reversed(range(T)):
        if t == T - 1:
            next_val = next_value
        else:
            next_val = values[t + 1].item()
        delta = rewards[t] + gamma * next_val - values[t].item()
        last_gae = delta + gamma * lam * last_gae
        advantages[t] = last_gae

    return advantages


def update_actor_critic(
    model: ActorCritic,
    optimizer: optim.Optimizer,
    obs: torch.Tensor,          # (T, obs_dim)
    actions: torch.Tensor,      # (T,)
    advantages: torch.Tensor,   # (T,) — precomputed via compute_td_advantage
    returns: torch.Tensor,      # (T,) — empirical discounted returns
    entropy_coef: float = 0.01,
    value_coef: float = 0.5,
) -> dict[str, float]:
    logits, values = model(obs)
    dist = torch.distributions.Categorical(logits=logits)
    log_probs = dist.log_prob(actions)
    entropy   = dist.entropy().mean()

    # Policy loss: -E[log_pi(a|s) * A(s,a)]
    policy_loss = -(log_probs * advantages.detach()).mean()
    # Value loss: MSE between V(s) and empirical returns
    value_loss  = torch.nn.functional.mse_loss(values, returns)
    # Total loss
    loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    optimizer.step()

    return {
        "policy_loss": policy_loss.item(),
        "value_loss": value_loss.item(),
        "entropy": entropy.item(),
    }
```

This actor-critic implementation is the skeleton underlying A2C, and with the addition of the PPO clipped surrogate (replacing `log_probs * advantages` with `min(r_t * A_t, clip(r_t, 1-eps, 1+eps) * A_t)`), it becomes PPO. Both algorithms depend critically on the Bellman expectation equation for the critic, and on the advantage function for the actor update.

---

## 17. Numerical stability and implementation pitfalls

Several practical issues arise when implementing Bellman updates:

**1. Advantage normalization.** Computing advantages via GAE and then normalizing them to zero mean and unit variance per mini-batch is a standard trick that PPO uses. This is not theoretically required but helps stabilize learning:

```python
# Normalize advantages within a mini-batch
advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
```

This rescales the step size implicitly — states with large advantages take larger gradient steps. Without normalization, a few high-reward states can dominate the gradient and cause policy instability.

**2. Bootstrap vs. terminal state handling.** When an episode terminates, the next value should be zero, not the critic's output for the next state (since there is no next state). Always mask out bootstrapped values at terminal steps:

```python
td_target = reward + gamma * next_value * (1.0 - done_flag)
```

Forgetting this masking is one of the most common bugs in RL implementations and causes the value function to underestimate terminal state values.

**3. Discount factor choice.** Common values:

| $\gamma$ | Effective horizon (steps) | Typical use case |
|---|---|---|
| 0.99 | ~100 | MuJoCo continuous control |
| 0.999 | ~1,000 | Long-horizon robotics |
| 0.95 | ~20 | CartPole, quick convergence |
| 1.0 | Infinite | Episodic tasks only (risky) |

The "effective horizon" is approximately $1/(1-\gamma)$ — the number of steps over which rewards are meaningfully discounted. Setting $\gamma$ too high extends the horizon artificially and makes variance explode; too low makes the agent short-sighted.

**4. Double Q-learning for bias correction.** Standard Q-learning overestimates Q-values because $\max_{a'} Q(s', a')$ is a biased estimator of $\max_{a'} Q^*(s', a')$ — the max of noisy estimates is systematically higher than the max of the true values. Double DQN (van Hasselt et al. 2016) corrects this by using one network to select the action and another to evaluate it:

```python
# Standard Q-learning (overestimation bias):
td_target = R + gamma * Q_target(s').max()

# Double DQN (reduced bias):
best_action = Q_online(s').argmax()           # select action with online net
td_target   = R + gamma * Q_target(s')[best_action]  # evaluate with target net
```

Double DQN improves performance on Atari by approximately 10-15% median score across the 49-game benchmark, according to the original Double DQN paper.

---

## 18. Bootstrapping, bias, and the TD-MC spectrum

One of the deepest insights in RL is the tradeoff between **bootstrapping** (using current estimates to update other estimates) and **full returns** (waiting until the episode ends to collect the real total return). The Bellman equation is what makes bootstrapping possible at all.

A Monte Carlo (MC) update for $V^\pi(s)$ uses the actual return $G_t$ observed in an episode:

$$V(S_t) \leftarrow V(S_t) + \alpha\left(G_t - V(S_t)\right)$$

This is unbiased — $\mathbb{E}[G_t | S_t = s] = V^\pi(s)$ exactly — but high variance, because a single return can deviate substantially from the mean depending on the luck of the trajectory.

A one-step TD update replaces $G_t$ with the Bellman target $R_{t+1} + \gamma V(S_{t+1})$:

$$V(S_t) \leftarrow V(S_t) + \alpha\left(R_{t+1} + \gamma V(S_{t+1}) - V(S_t)\right)$$

This is biased (because $V(S_{t+1})$ is not the true $V^\pi(S_{t+1})$ early in training) but lower variance (only one random step contributes to the randomness, not the entire episode). The bias disappears as $V$ converges.

The $n$-step TD update lets you interpolate:

$$G_t^{(n)} = R_{t+1} + \gamma R_{t+2} + \cdots + \gamma^{n-1} R_{t+n} + \gamma^n V(S_{t+n})$$

$$V(S_t) \leftarrow V(S_t) + \alpha\left(G_t^{(n)} - V(S_t)\right)$$

At $n = 1$ you have TD(0). At $n \to \infty$ you have MC. GAE, described earlier, is the $\lambda$-weighted geometric mixture of all $n$-step returns and is the practical sweet spot for most applications.

This spectrum — how much of the return comes from real rollout versus bootstrapped estimate — is one of the central design choices in every RL algorithm:

| Method | Return estimate | Bias | Variance | Typical use |
|---|---|---|---|---|
| MC | Full episode $G_t$ | None | High | Small episodic tasks |
| TD(0) | $R + \gamma V(s')$ | Some | Low | Online, continuing tasks |
| $n$-step TD | $n$-step partial return | Medium | Medium | General purpose |
| GAE ($\lambda$) | Exponential blend | Tunable | Tunable | PPO, A3C standard |
| TD($\lambda$) | Eligibility traces | Tunable | Tunable | Tabular, classic RL |

The contraction proof tells us that TD(0) converges despite its bias, as long as the step size $\alpha$ satisfies the Robbins-Monro conditions: $\sum_t \alpha_t = \infty$ and $\sum_t \alpha_t^2 < \infty$. In practice, using a fixed small $\alpha$ (like 0.001) combined with a large replay buffer achieves similar convergence properties.

---

## 19. The Bellman equation in continuous state and action spaces

Everything we have derived so far uses discrete sums over states and actions. In continuous settings — robot joint angles, financial positions, autonomous driving velocities — the sums become integrals.

**Continuous state Bellman expectation:**

$$V^\pi(s) = \int_a \pi(a|s) \int_{s'} P(s'|s,a)\!\left[R(s,a,s') + \gamma V^\pi(s')\right] ds'\, da$$

The structure is identical; only the sum becomes an integral. Function approximation (neural networks) replaces the table lookup.

**Continuous action Bellman optimality:**

$$V^*(s) = \max_a \int_{s'} P(s'|s,a)\!\left[R(s,a,s') + \gamma V^*(s')\right] ds'$$

The $\max_a$ over a continuous space is the tricky part. In deterministic policy gradient methods (DDPG, TD3), this is handled by learning a deterministic policy $\mu_\phi(s)$ that approximates $\arg\max_a Q(s, a)$, and the Q-function critic is updated via:

$$\mathcal{L}(\theta) = \mathbb{E}\!\left[\left(R + \gamma Q(S', \mu_\phi(S');\, \theta^-) - Q(S, A;\, \theta)\right)^2\right]$$

SAC (Soft Actor-Critic, Haarnoja et al. 2018) handles continuous actions differently, adding an entropy regularization term to the Bellman equation:

$$V^\pi(s) = \mathbb{E}_{a \sim \pi}\!\left[Q^\pi(s, a) - \alpha \log \pi(a|s)\right]$$

where $\alpha$ is a temperature parameter. This **soft Bellman equation** has its own contraction proof and results in policies that are "maximum entropy" — they balance high return against high entropy (exploration and robustness). SAC with this modification achieves roughly 10,000 average return on HalfCheetah-v4 (approximately 3× better than PPO's ~3,000) in 1 million steps, making it the gold standard for continuous control.

---

## 20. Policy iteration: alternating between V and π

With the Bellman equations in hand, the simplest algorithm that finds an optimal policy is **policy iteration**:

1. Start with an arbitrary policy $\pi_0$.
2. **Policy evaluation step:** Solve the Bellman expectation equations for $V^{\pi_k}$ exactly (or to convergence).
3. **Policy improvement step:** Update the policy greedily: $\pi_{k+1}(s) = \arg\max_a \sum_{s'} P(s'|s,a)[R(s,a,s') + \gamma V^{\pi_k}(s')]$.
4. Repeat until the policy stops changing.

#### Worked example: policy iteration on a 4-state chain

Consider a four-state chain: $\{s_0, s_1, s_2, s_3\}$ where $s_3$ is terminal. From each non-terminal state, the agent can go **left** (back toward $s_0$) or **right** (toward $s_3$). Rewards: moving right from $s_2 \to s_3$ gives $+10$; all other moves give $0$. $\gamma = 0.95$.

**Initial policy $\pi_0$:** always go left (suboptimal, will never reach the reward).

**Evaluation step 1:** Under always-left, the agent cycles between $s_0$ and $s_1$ forever, collecting zero reward. So $V^{\pi_0} = [0, 0, 0, 0]$.

**Improvement step 1:** For each state, compute the one-step value of going left vs. right:
- $s_0$: left → $s_0$ (value 0); right → $s_1$ (value 0). Tie — pick right arbitrarily.
- $s_1$: left → $s_0$ (0); right → $s_2$ (0). Tie — pick right.
- $s_2$: left → $s_1$ (0); right → $s_3$ ($+10 + 0$). Right wins: Q = 10.

New policy $\pi_1$: go right everywhere.

**Evaluation step 2:** Under always-right starting from $s_0$:
$$V^{\pi_1}(s_2) = 10 + 0.95 \times 0 = 10.0$$
$$V^{\pi_1}(s_1) = 0 + 0.95 \times 10.0 = 9.5$$
$$V^{\pi_1}(s_0) = 0 + 0.95 \times 9.5 = 9.025$$

**Improvement step 2:** Check if going left is ever better than right. For $s_0$: left → $s_0$ (9.025 going right, or 0 if stuck in left loop). Right is still better. The policy does not change. Converged.

**Result:** $\pi^*(s) = $ right everywhere, $V^* = [9.025, 9.5, 10.0, 0]$.

This example illustrates the **policy improvement theorem**: if the greedy policy under $V^{\pi_k}$ is different from $\pi_k$, then $V^{\pi_{k+1}} \geq V^{\pi_k}$ for all states, with strict improvement somewhere. Policy iteration converges in finitely many steps because there are only finitely many deterministic policies ($|\mathcal{A}|^{|\mathcal{S}|}$), and each iteration strictly improves until the optimum is reached.

```python
def policy_iteration(
    P: np.ndarray,
    R: np.ndarray,
    gamma: float = 0.95,
    eval_theta: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    """Classic policy iteration: alternating exact eval and greedy improvement."""
    n_states, n_actions, _ = P.shape

    # Start with a uniform random policy
    policy = np.ones((n_states, n_actions)) / n_actions

    max_policy_iters = 1000
    for pol_iter in range(max_policy_iters):
        # --- Policy Evaluation ---
        V = bellman_policy_eval(policy, P, R, gamma=gamma, theta=eval_theta)

        # --- Policy Improvement ---
        policy_stable = True
        new_policy = np.zeros_like(policy)
        for s in range(n_states):
            # Q(s, a) = sum_{s'} P[s,a,s'] * (R[s,a,s'] + gamma * V[s'])
            q_values = np.dot(P[s], R[s, :, :].diagonal() + gamma * V)
            # Technically: Q[s,a] = sum_{s'} P[s,a,s'] * (R[s,a,s'] + gamma*V[s'])
            q_sa = np.array([
                np.dot(P[s, a], R[s, a] + gamma * V)
                for a in range(n_actions)
            ])
            old_best = np.argmax(policy[s])
            best_action = np.argmax(q_sa)
            new_policy[s, best_action] = 1.0
            if old_best != best_action:
                policy_stable = False

        policy = new_policy
        if policy_stable:
            print(f"Policy iteration converged at iteration {pol_iter + 1}")
            break

    return V, policy
```

Policy iteration typically converges in 3–10 iterations for moderate MDPs, even when each evaluation step takes many Bellman sweeps. This is because the policy space is much smaller than the value space — you only need to identify the right action at each state, not pin down the exact value.

---

## 21. The linear algebra view: Bellman as a linear system

For a fixed policy $\pi$, the Bellman expectation equation is a **system of linear equations**. This algebraic view reveals the deepest structure and explains why exact solution is so powerful when the MDP is small enough.

The Bellman equation $\mathbf{V}^\pi = \mathbf{r}^\pi + \gamma \mathcal{P}^\pi \mathbf{V}^\pi$ can be written as:

$$(I - \gamma \mathcal{P}^\pi)\mathbf{V}^\pi = \mathbf{r}^\pi$$

The matrix $\mathcal{P}^\pi$ has entries $\mathcal{P}^\pi_{ss'} = \sum_a \pi(a|s) P(s'|s,a)$, which are non-negative and sum to 1 in each row (it is a stochastic matrix). This means $\mathcal{P}^\pi$ has spectral radius 1, and so $\gamma \mathcal{P}^\pi$ has spectral radius $\gamma < 1$. Therefore $(I - \gamma \mathcal{P}^\pi)$ is invertible, and the unique solution is:

$$\mathbf{V}^\pi = (I - \gamma \mathcal{P}^\pi)^{-1} \mathbf{r}^\pi$$

Furthermore, the inverse has a beautiful closed form. Using the Neumann series (valid because $\|\gamma \mathcal{P}^\pi\| < 1$):

$$(I - \gamma \mathcal{P}^\pi)^{-1} = \sum_{k=0}^\infty (\gamma \mathcal{P}^\pi)^k = I + \gamma \mathcal{P}^\pi + \gamma^2 (\mathcal{P}^\pi)^2 + \cdots$$

The $k$-th term $\gamma^k (\mathcal{P}^\pi)^k$ represents the $\gamma$-discounted probability of being in state $s'$ after exactly $k$ steps from state $s$ under policy $\pi$. Summing over all $k$ gives the **discounted state occupancy** — how often, in expectation, each state is visited under $\pi$. This occupancy measure plays a crucial role in importance sampling corrections for off-policy learning, which we will see in later posts on DQN's replay buffer and PPO's reuse ratio.

```python
def exact_bellman_eval(
    policy: np.ndarray,    # (n_states, n_actions)
    P: np.ndarray,         # (n_states, n_actions, n_states)
    R: np.ndarray,         # (n_states, n_actions, n_states)
    gamma: float,
) -> np.ndarray:
    """Exact policy evaluation via matrix inversion. O(|S|^3) but exact."""
    n_states = P.shape[0]
    # Compute P^pi[s, s'] = sum_a pi(a|s) * P[s,a,s']
    P_pi = np.einsum('sa,sas->ss', policy, P)
    # Compute r^pi[s] = sum_a pi(a|s) * sum_s' P[s,a,s'] * R[s,a,s']
    r_pi = np.einsum('sa,sas->s', policy, np.einsum('sas,sas->sa', P, R)[:, :, np.newaxis].squeeze())
    # Actually: r_pi[s] = sum_a pi[s,a] * sum_s' P[s,a,s'] * R[s,a,s']
    r_pi = np.einsum('sa,sa->s', policy, np.einsum('sai,sai->sa', P, R))
    # Solve (I - gamma * P_pi) V = r_pi
    A_mat = np.eye(n_states) - gamma * P_pi
    V = np.linalg.solve(A_mat, r_pi)
    return V
```

For MDPs with up to a few thousand states, matrix inversion is practical (it costs $O(|\mathcal{S}|^3)$). For MDPs with millions of states — or continuous MDPs — we fall back to iterative methods. But the existence of this exact closed form proves the uniqueness of $V^\pi$, which in turn anchors the convergence proofs for all approximate methods.

---

## 22. From tabular to deep: what changes and what stays the same

When we move from tabular RL (a lookup table for $V$ or $Q$) to deep RL (a neural network approximating $V_\theta$ or $Q_\theta$), the Bellman equations remain identical. What changes is *how we solve them*.

**Tabular (exact or iterative):**
- Store $V(s)$ or $Q(s,a)$ as arrays indexed by state/action indices.
- Each update changes exactly one entry.
- Convergence is guaranteed by the contraction theorem.
- Scales to $|\mathcal{S}| \sim 10^5$ to $10^6$ states.

**Neural approximation (DQN, PPO critic, SAC value network):**
- Store $V_\theta(s)$ or $Q_\theta(s,a)$ as a neural network with weights $\theta$.
- Each gradient step changes all entries simultaneously (because the network shares weights across states).
- Convergence is no longer guaranteed in general — the "deadly triad" (function approximation + bootstrapping + off-policy) can cause divergence (Baird 1995, Sutton & Barto Chapter 11).
- Scales to Atari ($84 \times 84 \times 4$ inputs), robot observations, language model hidden states.

The practical fix for the deadly triad in DQN is the **target network**: a periodic copy of the online network that is held fixed for $C$ steps. By not chasing a moving target, the Bellman update more closely resembles the contraction it approximates in theory. In practice, DQN with target networks reliably learns from raw Atari pixels, which suggests the theoretical divergence cases are rare in naturally structured MDPs.

A second crucial fix is the **experience replay buffer**: storing transitions $(s, a, r, s')$ and sampling them randomly to break temporal correlation. This violates the on-policy assumption but dramatically improves sample efficiency and stability, as correlations between consecutive frames (which all look nearly identical) would otherwise cause the gradient to update in nearly the same direction repeatedly.

```python
from collections import deque
import random

class ReplayBuffer:
    """Fixed-size experience replay buffer for off-policy deep RL."""

    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> list:
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        return len(self.buffer)


class DQNAgent:
    """Minimal DQN agent illustrating the Bellman optimality update."""

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        lr: float = 1e-3,
        gamma: float = 0.99,
        buffer_size: int = 100_000,
        batch_size: int = 64,
        target_update_freq: int = 1_000,
    ):
        import torch
        import torch.nn as nn

        self.gamma = gamma
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.target_update_freq = target_update_freq
        self.steps = 0

        self.online_net = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.ReLU(),
            nn.Linear(128, 128),    nn.ReLU(),
            nn.Linear(128, n_actions),
        )
        self.target_net = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.ReLU(),
            nn.Linear(128, 128),    nn.ReLU(),
            nn.Linear(128, n_actions),
        )
        # Hard copy online → target at init
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_size)

    def update(self) -> float | None:
        import torch
        if len(self.buffer) < self.batch_size:
            return None

        batch = self.buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states      = torch.tensor(states,      dtype=torch.float32)
        actions     = torch.tensor(actions,     dtype=torch.long)
        rewards     = torch.tensor(rewards,     dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones       = torch.tensor(dones,       dtype=torch.float32)

        # Current Q-values: Q(s, a) — Bellman left-hand side
        q_values = self.online_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target: r + gamma * max_{a'} Q_target(s', a') — Bellman right-hand side
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1).values
            td_targets = rewards + self.gamma * next_q * (1.0 - dones)

        loss = torch.nn.functional.mse_loss(q_values, td_targets)
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping prevents exploding gradients in early training
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), 10.0)
        self.optimizer.step()

        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        return loss.item()
```

Every line in `update()` maps directly back to the Bellman equation:
- `q_values` computes $Q_\theta(s, a)$ — the current estimate of the left-hand side.
- `next_q.max()` computes $\max_{a'} Q_{\theta^-}(s', a')$ — the right-hand side's bootstrapped target.
- `td_targets` is $R + \gamma \max_{a'} Q^-(s', a')$ — the Bellman optimality target.
- The MSE loss minimizes the squared Bellman residual.
- `target_net.load_state_dict` hard-copies the target, approximating the fixed-point property.

The entire deep RL machinery exists to make this approximation stable at scale. Strip it away and what remains is the Bellman equation.

---

## 23. Function approximation and the deadly triad

The combination of three ingredients that can cause divergence deserves a dedicated treatment, because every practitioner has hit it:

1. **Function approximation**: using a parameterized $V_\theta(s)$ or $Q_\theta(s,a)$ instead of a table.
2. **Bootstrapping**: updating toward a target that uses the same function approximation (TD, Q-learning).
3. **Off-policy learning**: updating on transitions generated by a different policy than the current one.

Baird's counterexample (1995) shows a simple linear function approximation + TD + off-policy combination that diverges to infinity. This is not a rare edge case — it can happen with arbitrary function approximators including neural networks, and the theoretical literature has no complete solution.

Practical mitigations that have been found to work:

| Mitigation | How it helps | Used in |
|---|---|---|
| Target networks | Stabilizes bootstrap target | DQN, DDPG, SAC |
| Replay buffer | Decorrelates samples | DQN, SAC, TD3 |
| On-policy learning | Avoids off-policy correction entirely | PPO, A2C |
| Gradient clipping | Limits update magnitude | All deep RL |
| Value function clipping (PPO) | Prevents large V jumps | PPO |
| Huber loss | Less sensitive to outlier TD errors | DQN variants |

The safest option for beginners is **on-policy methods** (PPO, A2C): they avoid the deadly triad by never using off-policy transitions, at the cost of lower sample efficiency. If sample efficiency matters, **SAC** is the best-understood off-policy method for continuous action spaces, with theoretical convergence guarantees for the soft Bellman equations.

---

## 24. Multi-step returns and eligibility traces

Beyond one-step and full-episode returns, there is an elegant algorithm called **TD($\lambda$)** that uses **eligibility traces** to do multi-step credit assignment online without waiting for the episode to end.

The eligibility trace $e_t(s)$ tracks how recently and frequently each state was visited:

$$e_t(s) = \gamma \lambda e_{t-1}(s) + \mathbf{1}[S_t = s]$$

The trace "fades" by $\gamma \lambda$ each step and "spikes" when the state is visited. When a TD error occurs, it propagates back to all recently visited states proportional to their trace:

$$V(s) \leftarrow V(s) + \alpha \delta_t e_t(s) \quad \text{for all } s$$

where $\delta_t = R_{t+1} + \gamma V(S_{t+1}) - V(S_t)$ is the one-step TD error.

This is the **TD($\lambda$) backward view**. The forward view (equivalent by theory) is the $\lambda$-return:

$$G_t^\lambda = (1-\lambda) \sum_{n=1}^\infty \lambda^{n-1} G_t^{(n)}$$

The equivalence between these two views (proven by Sutton 1988) is a deep result: it means we can compute a geometrically-weighted mixture of multi-step returns purely online, with one-step updates and a running trace — without actually computing any multi-step sum. GAE is a special case of the TD($\lambda$) forward view applied to the advantage function.

```python
def td_lambda_eval(
    episodes: list[list[tuple]],  # list of (state, reward) sequences
    n_states: int,
    gamma: float = 0.99,
    lam: float = 0.7,
    alpha: float = 0.01,
    n_passes: int = 50,
) -> np.ndarray:
    """Tabular TD(lambda) with eligibility traces — backward view."""
    V = np.zeros(n_states)

    for _ in range(n_passes):
        for episode in episodes:
            e = np.zeros(n_states)   # eligibility traces, reset each episode
            for t in range(len(episode) - 1):
                s, r = episode[t]
                s_next = episode[t + 1][0]
                # TD error
                delta = r + gamma * V[s_next] - V[s]
                # Accumulating trace
                e[s] += 1.0
                # Update all states
                V += alpha * delta * e
                # Decay trace
                e *= gamma * lam
    return V
```

For $\lambda = 0$, this reduces to TD(0). For $\lambda = 1$ (with offline episode processing), it converges to MC. Values between 0.7 and 0.95 often perform best empirically.

---

## Key takeaways

1. **$V^\pi(s)$ is the expected return from state $s$ under policy $\pi$** — it gives every state a single-number "score" that summarizes long-run future reward.

2. **$Q^\pi(s, a)$ conditions on the action too** — necessary for model-free learning because it lets you extract a greedy policy without a transition model.

3. **$A^\pi(s, a) = Q^\pi(s, a) - V^\pi(s)$ is zero-mean under $\pi$** — centering the update signal is why actor-critic methods have lower variance than vanilla policy gradient.

4. **The Bellman expectation equation for $V^\pi$ is derived from $G_t = R_{t+1} + \gamma G_{t+1}$** — the recursion in the return definition propagates expectations into the state-value recursion.

5. **The Bellman optimality equation replaces $\mathbb{E}_{a \sim \pi}$ with $\max_a$** — once you optimize, you select the best action, not an average.

6. **The Bellman optimality operator is a $\gamma$-contraction (Banach fixed point)** — this is the proof that value iteration, Q-learning, and TD converge to a unique $V^*$ or $Q^*$.

7. **Every RL algorithm is a different approximation strategy for Bellman equations** — DP solves them exactly, MC approximates via full trajectories, TD bootstraps, deep RL adds a neural function approximator.

8. **Greedy policy extraction from $Q^*$ is trivial**: $\pi^*(s) = \arg\max_a Q^*(s, a)$ — all the work is in computing $Q^*$.

9. **Bootstrap masking at terminal states is one of the most common implementation bugs** — always multiply the next-state value by $(1 - \text{done})$.

10. **The effective horizon is $\approx 1/(1-\gamma)$** — use this to set $\gamma$ deliberately based on how far ahead you need the agent to plan.

---

## Further reading

- **Sutton & Barto, "Reinforcement Learning: An Introduction" (2nd ed., 2018)** — Chapters 3–4 are the canonical reference for value functions and dynamic programming. Freely available at [incompleteideas.net](http://incompleteideas.net/book/the-book-2nd.html).
- **Mnih et al., "Human-level control through deep reinforcement learning" (Nature 2015)** — the DQN paper; shows Bellman optimality scaling to Atari with target networks and replay buffers.
- **Schulman et al., "High-Dimensional Continuous Control Using Generalized Advantage Estimation" (ICLR 2016)** — the GAE paper; derives the $\lambda$-weighted advantage estimator used in every modern actor-critic.
- **van Hasselt, Guez, Silver, "Deep Reinforcement Learning with Double Q-learning" (AAAI 2016)** — addresses the overestimation bias in the Bellman optimality operator and shows it matters empirically on Atari.
- **Bertsekas, "Dynamic Programming and Optimal Control" (4th ed.)** — rigorous treatment of contraction mappings and Bellman operators in the continuous-state setting.
- [What is reinforcement learning](/blog/machine-learning/reinforcement-learning/what-is-reinforcement-learning) — introduces the agent-environment-reward loop that this post's value functions are defined over.
- [Markov decision processes](/blog/machine-learning/reinforcement-learning/markov-decision-processes) — formalizes the MDP, states, actions, transitions, and rewards that the Bellman equations operate on.
- Dynamic programming for RL — the next post in this series; uses the Bellman equations derived here to implement policy evaluation, policy improvement, and value iteration in full.
