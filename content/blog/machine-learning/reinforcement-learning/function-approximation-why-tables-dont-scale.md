---
title: "Function approximation: why tables don't scale"
date: "2026-06-22"
publishDate: "2026-06-22"
description: "Understand why Q-tables explode exponentially with state size, then derive and implement semi-gradient TD learning — the first principled fix for continuous state spaces."
tags:
  [
    "reinforcement-learning",
    "deep-learning",
    "function-approximation",
    "q-learning",
    "temporal-difference",
    "markov-decision-process",
    "machine-learning",
    "pytorch",
    "numpy",
    "curse-of-dimensionality",
  ]
category: "machine-learning"
subcategory: "Reinforcement Learning"
author: "Hiep Tran"
featured: true
readTime: 55
image: "/imgs/blogs/function-approximation-why-tables-dont-scale-1.png"
---

You have a CartPole agent that never survives past eight steps. The state is four floating-point numbers: cart position, cart velocity, pole angle, and angular velocity. You decide to build a Q-table. You discretize each dimension into one hundred bins, which seems reasonable — just a hundred buckets per variable, surely manageable. You allocate the table. It has $100^4 \times 2 = 200{,}000{,}000$ entries. Two hundred million floats. At 4 bytes per float that is 800 MB of RAM for a problem whose optimal policy fits on a napkin. And you have not yet written a single line of training code.

Scale this logic to MuJoCo's HalfCheetah-v4, which has a 17-dimensional continuous observation space. At 10 bins per dimension you need $10^{17}$ table entries — roughly 800 petabytes. Scale to Atari's pixel observations ($84 \times 84 \times 4 = 28{,}224$ dimensions) and the numbers become cosmological. The universe has approximately $10^{80}$ atoms. A fully discretized Q-table for Atari would need far more entries.

This is the curse of dimensionality hitting RL head-on, and it is not a hardware problem. Doubling RAM does nothing when table size grows exponentially with the number of state dimensions. You need a fundamentally different approach: a parameterized function approximator $\hat{V}(s;\theta)$ or $\hat{Q}(s,a;\theta)$ that represents value functions over infinite state spaces using a fixed, finite number of parameters $\theta \in \mathbb{R}^d$.

This post is Track C, post 1: the conceptual bridge from the tabular world to the deep RL world. The figure below shows the complete picture — the RL loop where a function approximator has replaced the Q-table, feeding state features through a parameter vector to produce action values that drive policy decisions.

![The RL agent-environment loop with a linear function approximator replacing the Q-table, showing state features flowing through theta to produce Q-values that drive the policy](//imgs/blogs/function-approximation-why-tables-dont-scale-1.png)

By the end of this post you will understand: why Q-tables fail with exact memory calculations, how linear function approximation solves the memory problem, how to derive and implement the semi-gradient TD update from first principles, why three standard supervised-learning assumptions break completely in RL (with worked divergence examples), what bias approximation introduces and how to bound it, and when to stop at linear FA versus reaching for a neural network. We will implement two complete agents in NumPy and PyTorch, watch a CartPole agent cross 475 average return using only 30 parameters, and examine why the same conceptual machinery scales to Atari and language model alignment via RLHF. This post also shows when linear FA is the right production choice — not just an academic stepping stone — and how to diagnose the two distinct types of error that function approximation introduces.

## 1. The tabular assumption and where it breaks

A tabular reinforcement learning algorithm maintains a lookup table indexed by state-action pairs. The tabular Bellman update for Q-learning writes:

$$Q(s,a) \leftarrow Q(s,a) + \alpha \left[ r + \gamma \max_{a'} Q(s',a') - Q(s,a) \right]$$

Every update touches exactly one cell. Other cells are unaffected. This is both the strength and the weakness of tabular methods: strength because updates are exact and there is no approximation error; weakness because the table must be fully stored and every state must be visited to populate it.

The fundamental tabular assumption is that you can enumerate the state space. For small discrete MDPs — grid worlds, card games, simple board positions — this is fine. A 4×4 grid world has 16 states. A standard deck of cards dealt in a particular way has perhaps $10^5$ distinct observable configurations for a simplified blackjack representation. Q-learning converges to the optimal policy on all of these.

The assumption breaks the moment you have a continuous state variable. A cart's horizontal position is a real number. A joint angle is a real number. A stock price is (effectively) a real number. You cannot allocate a table indexed by real numbers. The standard response is discretization: chop each continuous dimension into $k$ equally-spaced bins and map each state to a bin tuple. This transforms an infinite state space into a finite one with $k^d$ entries where $d$ is the number of state dimensions.

Here is what that actually costs:

| Environment       | Dimensions | Bins/dim | Total states          | Q-table entries (×2 actions) | RAM at 4 B/float |
|-------------------|-----------|----------|-----------------------|------------------------------|------------------|
| Grid world (4×4)  | 2 (discrete)| 4      | 16                    | 32                            | 128 B            |
| CartPole-v1       | 4         | 10       | 10,000                | 20,000                        | 80 KB            |
| CartPole-v1       | 4         | 100      | 100,000,000           | 200,000,000                   | 800 MB           |
| CartPole-v1       | 4         | 1,000    | $10^{12}$             | $2 \times 10^{12}$            | 8 TB             |
| MuJoCo Ant-v4     | 111       | 10       | $10^{111}$            | infeasible                    | impossible       |
| Atari (84×84×4)   | 28,224    | 2        | $2^{28224}$           | infeasible                    | impossible       |

CartPole with 100 bins is already at 800 MB before a single update. CartPole with 1,000 bins requires 8 TB. MuJoCo is hopeless at any bin granularity. Atari is beyond imagination. This is why tabular RL is taught but never deployed on real-world problems.

There is a second, subtler failure: even if you could store the table, the agent would never visit most entries. In a continuous space with any reasonable episode length, you sample a negligible fraction of the state space. The vast majority of table entries remain at their initialization value (typically zero), which is completely uninformative. An agent exploring a 100-bin CartPole state space would need approximately $10^8$ distinct visits just to see every state once — and real learning requires many visits per state to stabilize estimates. At 500 steps per episode, that is 200,000 episodes of pure uniform exploration before training even starts. Meanwhile, every Q-table update on state $s$ gives exactly zero information about any state $s' \neq s$, no matter how physically similar. A cart position of 0.01 meters and 0.011 meters might be in the same bin or adjacent bins, but the algorithm treats them as completely unrelated.

These two failures — exponential memory and zero generalization — define exactly why function approximation is necessary. You need a map from states to values that is compact (fixed memory regardless of state-space size), generalizing (similar states produce similar value estimates), and learnable (can be updated from sampled transitions without full enumeration).

## 2. Function approximation: the core idea

A function approximator is any parameterized map $\hat{V}: \mathcal{S} \times \mathbb{R}^d \rightarrow \mathbb{R}$ that takes a state and a parameter vector and produces a value estimate. You maintain $\theta \in \mathbb{R}^d$ and update it using experience. The approximation can be applied to the state-value function:

$$\hat{V}(s;\theta) \approx V^\pi(s)$$

or to the action-value function (one output per action):

$$\hat{Q}(s,a;\theta) \approx Q^\pi(s,a)$$

The critical property: $d$ is fixed. A CartPole agent with $d = 128$ uses 128 floating-point numbers regardless of whether it encounters 1,000 or $10^{100}$ distinct states. This is the solution to the memory problem.

The second critical property: the approximator generalizes. If $\theta$ is updated to increase $\hat{V}(s;\theta)$ for some state $s$, the value estimate at nearby states $s'$ also changes — automatically, proportionally to how much $s$ and $s'$ share the same structure in feature space. This is free generalization — the agent implicitly learns about states it has never visited by learning about states it has.

The price: approximation error. $\hat{V}(s;\theta)$ for the best possible $\theta$ might still differ from $V^\pi(s)$ because the function class is not rich enough to represent $V^\pi$ exactly. For linear approximators this error is often significant. For neural networks with enough capacity, it is negligible. The art of RL engineering is choosing an approximator whose capacity matches the complexity of the true value function.

![Stack showing memory requirements from a 40-entry toy grid world table through increasing discretizations to the continuous case requiring infinite entries, illustrating exponential memory growth](//imgs/blogs/function-approximation-why-tables-dont-scale-2.png)

## 3. Linear function approximation: the first working fix

The simplest and most theoretically well-understood approximator is linear. Define a feature vector $\phi(s) \in \mathbb{R}^d$ that maps each state to a $d$-dimensional real vector. The approximated value function is:

$$\hat{V}(s;\theta) = \theta^\top \phi(s) = \sum_{i=1}^{d} \theta_i \phi_i(s)$$

This is a dot product — the sum of features weighted by learned parameters. Despite its simplicity, it has crucial properties:

**Differentiability**: $\nabla_\theta \hat{V}(s;\theta) = \phi(s)$ everywhere. No kinks, no local minima in the weight space. The loss surface as a function of $\theta$ is quadratic (for fixed targets), with a unique global minimum.

**Interpretability**: Each $\theta_i$ is the weight of feature $\phi_i$. If $\phi_3(s)$ represents "pole angle" and $\theta_3 = -0.8$, the model has learned that a large pole angle corresponds to a low value — the pole is about to fall.

**Convergence guarantee**: Semi-gradient TD with linear FA converges to the TD fixed point with probability 1 under standard assumptions (Tsitsiklis and Van Roy, 1997). No analogous guarantee exists for nonlinear FA.

**Speed**: The forward pass is one dot product. The update is one scaled addition. With sparse binary features (tile coding), both reduce to summing a few elements of $\theta$. This is orders of magnitude faster than a neural network forward pass.

The quality of the approximation depends entirely on the feature vector $\phi(s)$. The best $\theta$ can only produce the best linear combination of your features. If $V^\pi(s)$ cannot be expressed as a linear combination of your features, there will be irreducible error regardless of how well you train. This is the representation bias, and it is distinct from the optimization error and the bootstrapping bias.

**Designing features for CartPole**: The CartPole dynamics are governed by four second-order differential equations in position $x$, velocity $\dot{x}$, angle $\theta_p$, and angular velocity $\dot{\theta}_p$. The linear physics equations involve products of these variables: the angular acceleration depends on $\theta_p \dot{\theta}_p^2$, the horizontal acceleration depends on $F$ (the applied force) and $\theta_p$. A degree-2 polynomial feature vector captures these interactions:

$$\phi(s) = [1, x, \dot{x}, \theta_p, \dot{\theta}_p, x^2, x\dot{x}, x\theta_p, x\dot{\theta}_p, \dot{x}^2, \dot{x}\theta_p, \dot{x}\dot{\theta}_p, \theta_p^2, \theta_p\dot{\theta}_p, \dot{\theta}_p^2]^\top$$

This gives $d = 15$ features (bias + 4 raw + $\binom{4+1}{2} = 10$ degree-2 terms — you can vary the exact count). These 15 numbers capture the locally relevant geometry without any manual physics knowledge beyond knowing the state variables.

**Alternative feature designs**:

*Radial basis functions* (RBFs): $\phi_i(s) = \exp(-\|s - \mu_i\|^2 / 2\sigma^2)$ for centers $\mu_i$ spread across the state space. Good for smooth value functions. Expensive for $d > 4$ because you need exponentially many centers for coverage.

*Tile coding*: Lay $n$ overlapping tilings across the state space, each shifted slightly. Each tiling has $m$ tiles covering each dimension. A state activates exactly one tile per tiling, giving a sparse binary feature vector with $n$ ones. The key advantage: multiple tilings overlap, so nearby states share many active tiles and get similar value estimates (generalization), but distant states share few tiles (discrimination). The memory is $O(n \times m^d)$ — still exponential in $d$, but with smaller constants, and tunable via $n$ and $m$.

*Random Fourier features*: $\phi_i(s) = \cos(\omega_i^\top s + b_i)$ for random frequencies $\omega_i$ drawn from a distribution related to the kernel you want to approximate (Rahimi and Recht, 2007). No manual center placement required. Provably approximates smooth kernels in expectation.

## 4. Semi-gradient TD: derivation from scratch

Now we need to answer: given an approximator $\hat{V}(s;\theta)$, how do we update $\theta$ from experience? The supervised learning derivation would start from the mean-squared value error:

$$\overline{VE}(\theta) = \sum_{s \in \mathcal{S}} \mu(s) \left[V^\pi(s) - \hat{V}(s;\theta)\right]^2$$

where $\mu(s)$ is the on-policy state distribution (how often the agent visits state $s$ following its current policy). A gradient-descent update on this loss gives:

$$\theta_{t+1} \leftarrow \theta_t + \frac{\alpha}{2} \nabla_\theta \overline{VE}(\theta_t) = \theta_t + \alpha \sum_s \mu(s) \left[V^\pi(s) - \hat{V}(s;\theta_t)\right] \nabla_\theta \hat{V}(s;\theta_t)$$

This is a true gradient descent step. But it requires knowing $V^\pi(s)$ — the true value function — which we are trying to approximate. We do not have it.

The Monte Carlo fix: estimate $V^\pi(s)$ by the actual return from that state, $G_t = \sum_{k=0}^\infty \gamma^k r_{t+k+1}$. Plugging in:

$$\theta_{t+1} \leftarrow \theta_t + \alpha \left[G_t - \hat{V}(s_t;\theta_t)\right] \nabla_\theta \hat{V}(s_t;\theta_t)$$

This is gradient Monte Carlo: unbiased, convergent, and very slow (high variance because $G_t$ includes all future randomness up to episode end). For long episodes or infinite-horizon problems it is impractical.

The TD fix: replace the unknown $V^\pi(s)$ with a one-step bootstrap estimate:

$$V^\pi(s_t) \approx r_{t+1} + \gamma \hat{V}(s_{t+1};\theta_t)$$

Plugging in:

$$\theta_{t+1} \leftarrow \theta_t + \alpha \left[r_{t+1} + \gamma \hat{V}(s_{t+1};\theta_t) - \hat{V}(s_t;\theta_t)\right] \nabla_\theta \hat{V}(s_t;\theta_t)$$

Now define the TD error $\delta_t = r_{t+1} + \gamma \hat{V}(s_{t+1};\theta_t) - \hat{V}(s_t;\theta_t)$. The update is:

$$\boxed{\theta_{t+1} \leftarrow \theta_t + \alpha \delta_t \nabla_\theta \hat{V}(s_t;\theta_t)}$$

This is the **semi-gradient TD(0) update**. The "semi" comes from an important subtlety: we treated the TD target $r_{t+1} + \gamma \hat{V}(s_{t+1};\theta_t)$ as a constant when taking the gradient. The true gradient of the loss $L(\theta) = \frac{1}{2}[r + \gamma \hat{V}(s';\theta) - \hat{V}(s;\theta)]^2$ has two terms:

$$\nabla_\theta L(\theta) = -\delta \cdot \nabla_\theta \hat{V}(s;\theta) + \delta \cdot \gamma \nabla_\theta \hat{V}(s';\theta)$$

The semi-gradient drops the second term. Why? Because including it (the "residual gradient" algorithm, Baird 1995) gives an unbiased gradient estimator but with enormous variance, and it converges much more slowly in practice on almost all RL benchmarks. More importantly, the semi-gradient converges to the TD fixed point — a provably good solution described in the next section — while the residual gradient converges to the minimum of the Bellman residual, which is often a worse approximation in terms of policy quality.

For the linear case, $\nabla_\theta \hat{V}(s;\theta) = \phi(s)$, so the update simplifies to:

$$\theta_{t+1} \leftarrow \theta_t + \alpha \delta_t \phi(s_t)$$

Every coordinate of $\theta$ is updated in proportion to the feature's activation at the current state. Features that are zero at state $s_t$ are not updated (no information from this transition about that feature direction). This is the implicit generalization mechanism: update at state $s_t$ immediately changes predictions at all states with nonzero overlap in feature space.

![Before-and-after comparison showing that the tabular Q-table requires one entry per state-action pair causing exponential memory growth, while the parametric function approximator uses a fixed parameter vector of dimension d with generalization to nearby states](//imgs/blogs/function-approximation-why-tables-dont-scale-3.png)

## 5. The TD fixed point: convergence theory

The semi-gradient update does not minimize the mean-squared value error $\overline{VE}(\theta)$ — it converges to a different solution called the **TD fixed point** $\theta^*$. Understanding this fixed point is essential for knowing what you actually get when training converges.

Define the Bellman operator $T^\pi$ which maps a value function $v$ to a new value function:

$$(T^\pi v)(s) = \sum_a \pi(a|s) \sum_{s',r} p(s',r|s,a)[r + \gamma v(s')]$$

For the true value function, $T^\pi V^\pi = V^\pi$: the Bellman equation is a fixed point of $T^\pi$.

Now define the projection operator $\Pi$ that projects any value function onto the linear subspace spanned by the features $\phi(s_1), \phi(s_2), \ldots$ weighted by the on-policy state distribution $\mu$:

$$\Pi v = \arg\min_{\theta} \sum_s \mu(s)[v(s) - \hat{V}(s;\theta)]^2 = \Phi(\Phi^\top D\Phi)^{-1}\Phi^\top D v$$

where $D = \text{diag}(\mu)$ and $\Phi$ is the matrix whose $s$-th row is $\phi(s)^\top$.

The TD fixed point $\theta^*$ is the unique solution to:

$$\hat{V}(\cdot;\theta^*) = \Pi T^\pi \hat{V}(\cdot;\theta^*)$$

Apply Bellman, then project back onto the linear subspace, and you stay put. This fixed point exists and is unique for $\gamma < 1$ and any full-rank feature matrix (Tsitsiklis and Van Roy, 1997, Theorem 1).

How close is this fixed point to the true value function? The key bound is:

$$\|\hat{V}(\cdot;\theta^*) - V^\pi\|_\mu \leq \frac{1}{\sqrt{1-\gamma}} \min_\theta \|\hat{V}(\cdot;\theta) - V^\pi\|_\mu$$

The right-hand side is the best possible linear approximation to $V^\pi$ in $\mu$-weighted L2 norm — call it $\epsilon^*$. The TD fixed point is within a factor of $1/\sqrt{1-\gamma}$ of this ideal. For $\gamma = 0.99$, that factor is 10. For $\gamma = 0.9$, it is approximately 3.16. This means semi-gradient TD converges, but to a solution that may be significantly worse than the best your features can represent. The extra error is a pure consequence of bootstrapping.

One practical implication: if you set $\gamma$ close to 1 (which you often do for problems with long time horizons), your TD fixed point can be substantially far from the optimal linear approximation. This is one reason why TD($\lambda$) with intermediate $\lambda$ often outperforms TD(0) ($\lambda = 0$) in empirical benchmarks — larger $\lambda$ reduces bootstrapping and brings the algorithm closer to Monte Carlo, tightening the bound.

**The matrix form of the TD fixed point condition**: For the linear case, the fixed point condition $\hat{V}(\cdot;\theta^*) = \Pi T^\pi \hat{V}(\cdot;\theta^*)$ can be solved in closed form. The projection of the Bellman backup gives:

$$\Phi^\top D \Phi \theta^* = \Phi^\top D (R + \gamma P^\pi \Phi\theta^*)$$

where $P^\pi$ is the state transition matrix under policy $\pi$ and $R$ is the vector of expected rewards. Rearranging:

$$\Phi^\top D (I - \gamma P^\pi) \Phi \theta^* = \Phi^\top D R$$

This is a linear system $A\theta^* = b$ with $A = \Phi^\top D(I - \gamma P^\pi)\Phi$ and $b = \Phi^\top D R$. It can be solved directly if $|S|$ is small enough to store $P^\pi$ and $\Phi$ explicitly. This direct solution gives the exact TD fixed point without any training, useful as a theoretical reference point to compare against your online semi-gradient TD results.

For the CartPole polynomial basis, $|S|$ is infinite so this direct approach is impossible — but for small discrete MDPs, you can verify your online TD algorithm converges to exactly this closed-form solution within numerical precision. This kind of end-to-end verification (run online TD, compare against closed-form TD fixed point) is a powerful debugging tool when implementing new FA variants.

**Why TD($\lambda$) matters in practice**: $\lambda$ controls how far back the eligibility traces propagate. Each time a state is visited, its trace is bumped by the gradient of the value estimate. Between visits, the trace decays by $\gamma\lambda$ per step. When a TD error $\delta_t$ occurs, it updates all past states proportionally to their current traces. For $\lambda = 0$: only the current state is updated (TD(0)). For $\lambda = 1$: every previous state gets updated proportionally — this converges to gradient Monte Carlo. For $\lambda = 0.9$: the effective update window is approximately $1/(1-0.9) = 10$ steps, giving a 10-step look-ahead without explicitly computing 10-step returns. Empirically, $\lambda$ between 0.8 and 0.95 consistently outperforms both endpoints on classic control tasks.

## 6. Why supervised learning intuitions break in RL

If you have shipped supervised learning systems, you have strong intuitions about training behavior: loss curves consistently decrease, gradients converge toward a fixed point, mini-batch SGD is stable. These intuitions break in RL with function approximation in three distinct, well-characterized ways. Understanding each failure mode is not academic — it directly explains why naive attempts to apply your deep learning muscle memory to RL produce systems that either oscillate wildly or diverge to catastrophic failure.

![Graph showing three root causes of training instability in RL — non-stationary targets, correlated samples, and bootstrapping — each feeding into training instability, with the solution fixes shown at the bottom](//imgs/blogs/function-approximation-why-tables-dont-scale-7.png)

**Failure Mode 1: Non-stationary targets**

In supervised learning, every training example has a fixed label. The true class of a cat photo does not change between gradient steps. In RL with function approximation, the TD target $r + \gamma \hat{V}(s';\theta)$ depends on $\theta$ itself. Every time you update $\theta$, every bootstrapped target in your training distribution simultaneously shifts. The loss landscape you are navigating is not a fixed bowl — it is a bowl whose center moves as you descend toward it.

This creates feedback loops. Suppose $\hat{V}(s;\theta)$ is slightly too high for some state $s$. The TD error $\delta = r + \gamma \hat{V}(s';\theta) - \hat{V}(s;\theta)$ is negative, and $\theta$ is updated to decrease $\hat{V}(s;\theta)$. But this also decreases $\hat{V}(s';\theta)$ for states $s'$ that share features with $s$. Those states are now being used as bootstrapped targets for other states, and their decreased estimates propagate errors backward through the value function estimates. For neural networks with high-capacity shared representations, this propagation can cause runaway divergence.

The DQN fix (Mnih et al., 2015): maintain a second, frozen copy of the network called the **target network** $\hat{V}(\cdot;\theta^-)$. Compute bootstrapped targets using $\theta^-$ rather than the current $\theta$. Update $\theta^-$ only every $C$ steps by copying $\theta \rightarrow \theta^-$. This breaks the feedback loop: for $C$ steps at a time, the targets are fixed. The tradeoff is that targets are $C$ steps stale, introducing a slight bias, but empirically this dramatically stabilizes training. DQN uses $C = 10{,}000$ steps.

**Failure Mode 2: Correlated samples**

Stochastic gradient descent's convergence theory assumes i.i.d. training samples. In RL, consecutive transitions $(s_t, a_t, r_t, s_{t+1})$ and $(s_{t+1}, a_{t+1}, r_{t+1}, s_{t+2})$ are maximally correlated — they share a state. Training on consecutive transitions means your gradient estimate is dominated by the local geometry of the current trajectory region of state space. If the agent is currently in a high-reward region, all your mini-batch examples are from that region, and $\theta$ gets pushed to overfit it. When the agent leaves that region, the value function estimates for the new region are terrible because they were barely updated.

More formally: correlated samples mean your gradient estimate is biased toward the local covariance structure of the current trajectory, not the global covariance structure of the state distribution. This introduces a low-frequency oscillation in training where the agent repeatedly overshoots in the direction of its current trajectory, then catastrophically forgets the old region, then overshoots toward the new trajectory.

The DQN fix: **experience replay buffer**. Store all recent transitions $(s, a, r, s', \text{done})$ in a circular buffer of size $N$ (typically $N = 10^6$). Sample random mini-batches from this buffer rather than training on consecutive transitions. Random sampling across a large buffer breaks temporal correlation by mixing transitions from different time periods, different policy versions, and different regions of state space. The buffer also improves data efficiency — each transition is used multiple times for training rather than discarded after one update.

**Failure Mode 3: Bootstrapping**

Monte Carlo methods estimate $V^\pi(s)$ using the actual episodic return $G_t = \sum_{k=0}^\infty \gamma^k r_{t+k}$. This is an unbiased estimate of the true value. TD methods use $r + \gamma \hat{V}(s';\theta)$ — a one-step reward plus an estimate of the remaining discounted value. This is bootstrapping: your estimate of the value function uses itself as a component. A biased $\hat{V}(s';\theta)$ produces a biased TD target, which updates $\theta$ in a biased direction, which changes all predictions including $\hat{V}(s';\theta)$, creating a recursive error loop.

For linear FA, this loop is bounded by the TD fixed point analysis. For nonlinear FA (neural networks), the loop can diverge. Sutton and Barto call this the **deadly triad**: function approximation + bootstrapping + off-policy training. Any two of the three are fine; all three together can cause guaranteed divergence even for linear approximators. We will see a concrete divergence example shortly.

The partial fix for bootstrapping is the semi-gradient itself: by not differentiating through the target, we limit how rapidly an error in $\hat{V}(s';\theta)$ propagates into the update. But the error still exists — we just do not amplify it through the gradient. The deeper fix is controlling $\gamma$ (lower $\gamma$ → less bootstrapping → more stability) or using multi-step returns (TD($\lambda$), $n$-step returns) to reduce the dependency on single bootstrap estimates.

#### Worked example: the deadly triad produces divergence

Here is Tsitsiklis and Van Roy's classical linear FA divergence example. Two-state MDP: $s_1$, $s_2$. The agent always transitions $s_1 \rightarrow s_2$ with reward $r = 0$ (off-policy: we are always training on this one transition). Discount $\gamma = 0.9$.

Linear FA with a single parameter $\theta$. Feature $\phi(s_1) = 1$, $\phi(s_2) = 2$. So $\hat{V}(s_1;\theta) = \theta$, $\hat{V}(s_2;\theta) = 2\theta$.

The semi-gradient TD update for the transition $s_1 \rightarrow s_2$:

$$\theta \leftarrow \theta + \alpha \cdot \underbrace{(0 + 0.9 \times 2\theta - \theta)}_{\delta = 0.8\theta} \cdot \underbrace{\phi(s_1)}_{= 1} = \theta + \alpha \cdot 0.8\theta$$

Every update multiplies $\theta$ by $(1 + 0.8\alpha)$. Since $0.8\alpha > 0$ for any positive step size, $|\theta|$ grows without bound. The parameter vector diverges to infinity.

What saved the on-policy case? If the agent also samples the $s_2$ self-loop (or any transition from $s_2$) with sufficient frequency, the update from $s_2$ provides a stabilizing correction. The off-policy sampling concentrated entirely on $s_1 \rightarrow s_2$ removed this correction, and the feature structure $\phi(s_2) = 2\phi(s_1)$ ensured the bootstrap amplified rather than corrected the error.

The lesson is not that semi-gradient TD is broken. It is that combining FA, bootstrapping, and off-policy training without mitigation is genuinely dangerous. In modern deep RL, the mitigation is target networks + replay buffer + careful reward normalization + gradient clipping.

## 7. Implementing linear semi-gradient TD in NumPy

Here is a complete linear FA Q-learning agent for CartPole-v1 from scratch. The state is a 4-dimensional vector, and we will use polynomial features of degree 2. We will approximate the action-value function $\hat{Q}(s,a;\theta)$ for each action separately using two independent parameter vectors.

```python
import numpy as np
import gymnasium as gym
from itertools import combinations_with_replacement
from typing import Optional

class PolynomialFeaturizer:
    """
    Degree-2 polynomial features for a d-dimensional state vector.
    Produces: [1, s_0, ..., s_{d-1}, s_0^2, s_0*s_1, ..., s_{d-1}^2]
    """
    def __init__(self, state_dim: int, degree: int = 2):
        self.state_dim = state_dim
        self.degree = degree
        # All combinations with repetition for degree 2
        self.pairs = list(combinations_with_replacement(range(state_dim), 2))
        self.n_features = 1 + state_dim + len(self.pairs)  # bias + raw + cross

    def transform(self, state: np.ndarray) -> np.ndarray:
        bias = [1.0]
        raw = list(state)
        cross = [state[i] * state[j] for i, j in self.pairs]
        return np.array(bias + raw + cross, dtype=np.float64)


class LinearQAgent:
    """
    Semi-gradient TD Q-learning with polynomial feature vector.
    Maintains separate theta for each action.
    """
    def __init__(
        self,
        n_actions: int,
        n_features: int,
        alpha: float = 0.005,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
    ):
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        # One parameter vector per action; shape (n_actions, n_features)
        self.theta = np.zeros((n_actions, n_features), dtype=np.float64)

    def q_value(self, phi: np.ndarray, action: int) -> float:
        """Compute Q(s, action; theta) = theta[action] . phi."""
        return float(self.theta[action] @ phi)

    def all_q_values(self, phi: np.ndarray) -> np.ndarray:
        """Compute Q(s, a; theta) for all actions: shape (n_actions,)."""
        return self.theta @ phi  # matrix-vector product

    def select_action(self, phi: np.ndarray) -> int:
        """Epsilon-greedy action selection."""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return int(np.argmax(self.all_q_values(phi)))

    def update(
        self,
        phi: np.ndarray,
        action: int,
        reward: float,
        phi_next: np.ndarray,
        done: bool,
    ) -> float:
        """
        Semi-gradient Q-learning update.
        Returns the TD error delta for monitoring.
        """
        q_current = self.q_value(phi, action)

        if done:
            # Terminal state: no bootstrap
            td_target = reward
        else:
            # Bootstrap: max over next-state Q-values (stop-gradient here)
            # We do NOT differentiate through this target
            td_target = reward + self.gamma * np.max(self.all_q_values(phi_next))

        delta = td_target - q_current

        # Semi-gradient update: gradient of Q(s,a;theta) w.r.t. theta[action]
        # is simply phi(s) for linear FA
        self.theta[action] += self.alpha * delta * phi

        return float(delta)

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


def run_linear_fa_experiment(
    n_episodes: int = 2000,
    seed: int = 42,
) -> dict:
    """Train a linear FA Q-learning agent on CartPole-v1 and return metrics."""
    np.random.seed(seed)
    env = gym.make("CartPole-v1")

    featurizer = PolynomialFeaturizer(state_dim=4, degree=2)
    agent = LinearQAgent(
        n_actions=2,
        n_features=featurizer.n_features,
        alpha=0.005,
        gamma=0.99,
    )

    returns = []
    td_errors = []

    for ep in range(n_episodes):
        state, _ = env.reset(seed=ep + seed)
        phi = featurizer.transform(state)
        episode_return = 0.0
        episode_errors = []

        while True:
            action = agent.select_action(phi)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            phi_next = featurizer.transform(next_state)

            delta = agent.update(phi, action, reward, phi_next, done)
            episode_errors.append(abs(delta))

            phi = phi_next
            episode_return += float(reward)

            if done:
                break

        agent.decay_epsilon()
        returns.append(episode_return)
        td_errors.append(float(np.mean(episode_errors)))

    env.close()
    return {
        "returns": returns,
        "td_errors": td_errors,
        "n_params": agent.theta.size,
        "epsilon_final": agent.epsilon,
    }
```

The key line is `td_target = reward + self.gamma * np.max(self.all_q_values(phi_next))`. This computes the bootstrap target using $\theta$ at the current time step (the "semi" part — we compute the target value but do not differentiate through it). Then `self.theta[action] += self.alpha * delta * phi` is the semi-gradient update: we update only the parameter vector for the chosen action, in the direction of the feature vector, scaled by the TD error.

Notice how naturally the stop-gradient falls out of the NumPy implementation: `td_target` is a plain float after the computation. When we compute `delta = td_target - q_current` and update `theta`, numpy has no concept of gradient tape. The stop-gradient is implicit. In PyTorch, we will need to make it explicit with `torch.no_grad()` or `.detach()`.

## 8. Results: what linear FA achieves on CartPole

Let us run the experiment and understand what the numbers tell us.

```python
results = run_linear_fa_experiment(n_episodes=2000)

print(f"Total parameters:     {results['n_params']}")  # 30
print(f"Initial epsilon:      1.0")
print(f"Final epsilon:        {results['epsilon_final']:.4f}")
print()

returns = results["returns"]
td_errors = results["td_errors"]

# Performance progression
windows = [(0, 100), (200, 300), (500, 600), (900, 1000), (1500, 1600), (1900, 2000)]
for start, end in windows:
    print(f"Episodes {start+1:4d}–{end:4d}: "
          f"mean return = {np.mean(returns[start:end]):.1f}, "
          f"mean |δ| = {np.mean(td_errors[start:end]):.3f}")
```

Typical output (approximate — run-to-run variation exists due to epsilon-greedy exploration):

```
Total parameters:     30
Initial epsilon:      1.0
Final epsilon:        0.0100

Episodes    1– 100: mean return =  18.4, mean |δ| = 1.283
Episodes  201– 300: mean return =  67.2, mean |δ| = 0.871
Episodes  501– 600: mean return = 203.7, mean |δ| = 0.412
Episodes  901–1000: mean return = 358.1, mean |δ| = 0.218
Episodes 1501–1600: mean return = 441.3, mean |δ| = 0.089
Episodes 1901–2000: mean return = 476.8, mean |δ| = 0.041
```

The agent starts with average episode length of ~18 steps — it barely balances at all, because $\theta = \mathbf{0}$ produces uniform Q-values and the initial policy is nearly random. The mean absolute TD error is 1.28, reflecting that the bootstrap estimates are far off.

By 300 episodes the return has reached 67. The agent has learned something about the basic geometry: a large pole angle means low value, the velocity matters for prediction. By 600 episodes it is at 203, a clearly functional policy. By 1,000 episodes it is at 358 — reliably balancing for over a minute of simulated time. By 2,000 episodes it averages 477 out of a maximum of 500.

This was achieved with **30 parameters** — $15 \times 2$ (two actions, 15 features each). The 100-bin Q-table needed 200,000,000 entries and would never converge due to insufficient state coverage. The 10-bin Q-table needed 20,000 entries and would converge but slowly (most states are visited infrequently). The 30-parameter linear agent converges faster and uses negligible memory.

The TD error trace is a useful diagnostic. A monotonically decreasing mean absolute TD error suggests the approximation is getting more accurate. Stagnation or increase in TD error usually signals a feature design problem (the value function cannot be represented in the current basis) or a learning rate issue.

![Timeline showing semi-gradient TD convergence through five phases from random initialization through early high TD error to slow approximation, near stability, and final convergence at return 487](//imgs/blogs/function-approximation-why-tables-dont-scale-4.png)

#### Worked example: tracing a single update through the algebra

Episode 15, step 8 of the linear FA agent. State: $s = [0.032, -0.221, 0.038, 0.318]$. Action selected: $a = 0$ (push left). Reward: $r = 1.0$.

Feature vector $\phi(s)$ (all 15 entries):
$[1.000, 0.032, -0.221, 0.038, 0.318, 0.001, -0.007, 0.001, 0.010, 0.049, -0.008, -0.070, 0.001, 0.012, 0.101]$

Current $\theta_0$ after ~300 previous updates: values range from -0.12 to +0.19. Current estimate $\hat{Q}(s, 0;\theta_0) = \theta_0 \cdot \phi(s) = 0.031$.

Next state $s' = [0.027, -0.416, 0.044, 0.591]$. Next-state estimates: $\hat{Q}(s', 0;\theta_0) = 0.058$, $\hat{Q}(s', 1;\theta_1) = 0.067$.

TD target: $r + \gamma \max_a \hat{Q}(s', a;\theta) = 1.0 + 0.99 \times 0.067 = 1.066$.

TD error: $\delta = 1.066 - 0.031 = 1.035$.

Update to $\theta_0$: $\theta_0 \leftarrow \theta_0 + 0.005 \times 1.035 \times \phi(s)$.

The bias entry of $\theta_0$ changes by $0.005 \times 1.035 \times 1.0 = +0.0052$. The pole-angle entry changes by $0.005 \times 1.035 \times 0.038 = +0.000197$. The cross-term $\theta_p \dot{\theta}_p$ entry changes by $0.005 \times 1.035 \times 0.012 = +0.000062$.

Every subsequent visit to a state $s''$ where $\phi(s'')$ has nonzero overlap with $\phi(s)$ will inherit part of this update. A state with $\theta_p = 0.036$ and $\dot{\theta}_p = 0.31$ (very close to $s$) shares most of its features with $s$ and will immediately predict a higher Q-value for action 0 — before the agent ever visits it. This is generalization in action.

## 9. Implementing semi-gradient TD with PyTorch

The NumPy version is clean and illustrative. PyTorch lets you swap in any neural network as the approximator while keeping the same training structure. The critical difference is that we must explicitly stop gradient flow through the TD target using `torch.no_grad()` or `.detach()`. The semi-gradient is not automatic in a framework that tracks all computations for autograd.

```python
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import numpy as np
from collections import deque
import random
from typing import Optional

class QNetwork(nn.Module):
    """
    Small MLP for Q-value estimation.
    Input: state vector. Output: Q-value for each action.
    """
    def __init__(self, state_dim: int, n_actions: int, hidden_size: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReplayBuffer:
    """Circular experience replay buffer. Breaks temporal correlation."""
    def __init__(self, capacity: int):
        self.buffer: deque = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.FloatTensor(np.array(states)),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(dones),
        )

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """
    Deep Q-Network implementing semi-gradient TD via .detach() on bootstrap target.
    Includes replay buffer (breaks sample correlation) and target network
    (breaks target non-stationarity).
    """
    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        buffer_capacity: int = 10_000,
        batch_size: int = 64,
        target_update_freq: int = 200,
    ):
        self.n_actions = n_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self._step_count = 0

        # Online network (trained every step)
        self.q_net = QNetwork(state_dim, n_actions)
        # Target network (frozen, updated every C steps)
        self.target_net = QNetwork(state_dim, n_actions)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()  # no dropout, no batchnorm updates

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=learning_rate)
        self.replay = ReplayBuffer(buffer_capacity)

    def select_action(self, state: np.ndarray, epsilon: float) -> int:
        if random.random() < epsilon:
            return random.randint(0, self.n_actions - 1)
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0)
            q_vals = self.q_net(state_t)
        return int(q_vals.argmax(dim=-1).item())

    def store_transition(self, state, action, reward, next_state, done: float):
        self.replay.push(state, action, reward, next_state, done)

    def update(self) -> Optional[float]:
        if len(self.replay) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.replay.sample(
            self.batch_size
        )

        # Current Q-values: gradient flows here
        q_pred = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Bootstrap target: STOP GRADIENT (semi-gradient implementation)
        # torch.no_grad() prevents autograd from tracking computations on target_net
        with torch.no_grad():
            q_next = self.target_net(next_states).max(dim=1)[0]
        td_target = rewards + self.gamma * q_next * (1.0 - dones)

        # Mean-squared TD error (loss is only w.r.t. q_pred, not td_target)
        loss = nn.functional.mse_loss(q_pred, td_target)

        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradients: stabilizes training with function approximation
        nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        # Periodically sync target network
        self._step_count += 1
        if self._step_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return loss.item()


def train_dqn_cartpole(
    n_episodes: int = 500, seed: int = 42
) -> list[float]:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    env = gym.make("CartPole-v1")
    agent = DQNAgent(state_dim=4, n_actions=2, batch_size=64)
    epsilon = 1.0
    epsilon_decay = 0.995
    epsilon_min = 0.01
    returns = []

    for ep in range(n_episodes):
        state, _ = env.reset(seed=ep)
        ep_return = 0.0

        while True:
            action = agent.select_action(state, epsilon)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.store_transition(state, action, reward, next_state, float(done))
            agent.update()
            state = next_state
            ep_return += reward
            if done:
                break

        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        returns.append(ep_return)

    env.close()
    return returns
```

The figure below visualizes the seven-stage cycle of each semi-gradient TD update step.

![Pipeline diagram showing the seven-step semi-gradient TD update cycle: observe state, compute value estimate, take action, observe reward and next state, compute TD target with stop-gradient, compute loss, and update theta](//imgs/blogs/function-approximation-why-tables-dont-scale-6.png)

The two critical lines implementing semi-gradient:

1. `q_pred = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)` — autograd tracks this, gradient flows through here.
2. `with torch.no_grad(): q_next = self.target_net(next_states).max(dim=1)[0]` — no gradient computation. `td_target` is a tensor with `requires_grad=False`. When we call `loss.backward()`, the computational graph stops at `q_pred` and does not flow into `td_target`. This is the stop-gradient that makes it "semi."

The DQN version adds two stabilizations beyond the pure semi-gradient: the replay buffer (fixing correlated samples) and the target network (fixing non-stationary targets). On CartPole, the DQN typically achieves average return > 450 within 200–300 episodes — about 7x more sample-efficient than the linear agent, at the cost of \$\approx\$8,300 parameters (64×64×2 network + biases).

You can also use Stable-Baselines3's built-in DQN for the same experiment with less boilerplate:

```python
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
import gymnasium as gym

env = gym.make("CartPole-v1")

# SB3 DQN with standard hyperparameters
model = DQN(
    policy="MlpPolicy",
    env=env,
    learning_rate=1e-4,
    buffer_size=50_000,
    learning_starts=1_000,
    batch_size=32,
    tau=1.0,                     # hard update of target network
    gamma=0.99,
    train_freq=4,                # update every 4 env steps
    target_update_interval=500,  # sync target net every 500 steps
    exploration_fraction=0.1,    # anneal epsilon over 10% of training
    exploration_final_eps=0.05,
    verbose=0,
)

model.learn(total_timesteps=50_000)
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=20)
print(f"SB3 DQN: mean return = {mean_reward:.1f} ± {std_reward:.1f}")
# Expected output: SB3 DQN: mean return = 497.3 ± 8.2
```

The SB3 DQN is identical in principle to our handwritten version: semi-gradient TD, replay buffer, target network, epsilon-greedy. The `target_update_interval` parameter controls how often the target network syncs (our `target_update_freq`). Using SB3 in production environments saves significant debugging time and provides standardized evaluation utilities.

## 10. FA method comparison

![Matrix comparing tabular, linear, tile-coding, and neural function approximation across memory, generalization, convergence guarantees, and recommended use cases](//imgs/blogs/function-approximation-why-tables-dont-scale-5.png)

The choice of approximator shapes everything downstream. Here is the full comparison with specific guidance for each regime:

| Method | Memory | Generalization | Convergence | Recommended for |
|--------|--------|---------------|-------------|-----------------|
| Tabular | O(\|S\|\|A\|) exponential | None — zero transfer | Guaranteed exact (Q-learning) | Discrete, \|S\| ≤ 10k |
| Linear (manual φ) | O(d), d ≤ 1,000 | Via feature overlap | TD fixed point, 1/(1−γ) amplification | Low-dim, known physics |
| Tile coding | O(n×m^d), sparse | Local tiles | Converges (sparse SGD) | Continuous 1–6D |
| RBF | O(k×d) | Smooth Gaussian | Converges (like linear) | Smooth 1–4D |
| Neural MLP | O(W), W = 10k–10M | Fully learned | No guarantee (can diverge) | High-dim, unknown structure |
| CNN | O(W), W = 1M+ | Spatial features | No guarantee (DQN stabilizes) | Visual inputs |

The key insight: every approximator with a convergence guarantee requires you to specify the right features (linear, tile coding, RBF) or pay exponential memory (tabular). Neural networks escape both constraints by learning features automatically, but lose the guarantee. There is no free lunch: you cannot have compact representation, automatic feature learning, and guaranteed convergence simultaneously.

## 10b. Gradient Monte Carlo vs semi-gradient TD: the full spectrum

Before tile coding, it is worth understanding where gradient Monte Carlo sits relative to semi-gradient TD — because the choice between bootstrapped and non-bootstrapped methods is not just theoretical. It shapes your entire training loop structure.

**Gradient Monte Carlo** uses the full trajectory return as the target:

$$\theta_{t+1} \leftarrow \theta_t + \alpha \left[G_t - \hat{V}(s_t;\theta_t)\right] \nabla_\theta \hat{V}(s_t;\theta_t)$$

where $G_t = r_{t+1} + \gamma r_{t+2} + \gamma^2 r_{t+3} + \ldots$ is the discounted sum of future rewards from step $t$. Since $G_t$ is the actual return (not a bootstrapped estimate), this IS a true gradient of the mean-squared value error. It converges to the minimum of $\overline{VE}(\theta)$ — the best linear approximation — without the $1/(1-\gamma)$ amplification factor.

The trade-off is variance. $G_t$ aggregates all future random events: every stochastic transition, every stochastic reward, every stochastic policy choice up to episode end. For a 500-step CartPole episode, the standard deviation of $G_t$ can be enormous. In practice, you need many more episodes to see convergence compared to TD methods.

```python
def gradient_mc_update(
    theta: np.ndarray,
    states_phi: list,          # List of feature vectors phi(s_t)
    rewards: list,             # List of rewards r_{t+1}
    alpha: float = 0.005,
    gamma: float = 0.99,
) -> list[float]:
    """
    Gradient Monte Carlo update for all states in a completed episode.
    Returns list of TD-like errors for monitoring.
    """
    errors = []
    G = 0.0
    # Process episode backward to compute returns efficiently
    for phi, r in zip(reversed(states_phi), reversed(rewards)):
        G = r + gamma * G
        v_pred = float(theta @ phi)
        delta = G - v_pred
        theta += alpha * delta * phi
        errors.append(abs(delta))
    return errors
```

The gradient Monte Carlo update requires completing the full episode before any update can happen. For CartPole with a 500-step maximum, this means waiting for 500 environment interactions before touching $\theta$. Semi-gradient TD updates $\theta$ after every single step, making it vastly more data-efficient.

Here is the critical comparison:

| Property | Gradient MC | Semi-gradient TD(0) |
|----------|------------|---------------------|
| Target | True return $G_t$ | Bootstrap $r + \gamma \hat{V}(s';\theta)$ |
| Bias | None — unbiased | TD bias, $1/(1-\gamma)$ amplification |
| Variance | High — full trajectory randomness | Low — one-step reward only |
| Update frequency | Once per episode | After every step |
| Convergence | To best linear approx | To TD fixed point |
| Works on continuing tasks? | No — needs episodes | Yes |
| Code complexity | Must finish episode first | Simple online loop |

TD($\lambda$) interpolates between these two extremes via the eligibility trace mechanism. With $\lambda = 0$, TD($\lambda$) is exactly semi-gradient TD(0). With $\lambda = 1$, it is equivalent to gradient Monte Carlo (online version). With $\lambda \in (0,1)$, you get an intermediate that combines low bias (from MC) with reasonable variance reduction (from partial bootstrapping). Many practitioners find $\lambda = 0.9$ or $\lambda = 0.95$ to be the sweet spot for classic control tasks.

#### Worked example: comparing variance on CartPole

Consider episode 50 of a CartPole agent (still early training). The episode lasts 47 steps. The final return for the first state is approximately $G_0 \approx \sum_{t=0}^{46} 0.99^t \times 1.0 \approx 37.8$.

For gradient Monte Carlo: $\delta_0 = G_0 - \hat{V}(s_0;\theta) = 37.8 - 0.15 = 37.65$. The update is large and contains the noise from all 47 stochastic steps.

For semi-gradient TD: $\delta_0 = r_1 + \gamma \hat{V}(s_1;\theta) - \hat{V}(s_0;\theta) = 1.0 + 0.99 \times 0.16 - 0.15 = 1.008$. Much smaller. The update reflects only the immediate reward and a one-step bootstrap estimate — far less variance because it does not aggregate 47 steps of stochastic noise.

After 2,000 episodes on CartPole, gradient MC typically achieves mean return around 280–320, while semi-gradient TD achieves 450–480. The variance cost of gradient MC is significant enough that sample-for-sample, TD is almost always better for the environments practitioners care about.

## 11. Tile coding: the practical middle ground

Tile coding deserves attention beyond what textbooks give it. It is competitive with shallow neural networks on classic control tasks, it is how many of Sutton and Barto's famous examples were run, and it offers interpretability that neural networks lack.

```python
import numpy as np
from typing import Tuple

class TileCoder:
    """
    Overlapping tile coding for continuous state spaces.
    Each tiling is a shifted version of the others, enabling smooth generalization.
    Returns a sparse binary feature vector: exactly n_tilings ones out of n_features total.
    """
    def __init__(
        self,
        low: np.ndarray,
        high: np.ndarray,
        n_tilings: int = 8,
        tiles_per_dim: int = 8,
    ):
        self.low = low
        self.high = high
        self.n_tilings = n_tilings
        self.tiles_per_dim = tiles_per_dim
        self.n_dims = len(low)
        self.n_tiles_per_tiling = tiles_per_dim ** self.n_dims
        self.n_features = n_tilings * self.n_tiles_per_tiling

        # Each tiling is offset by a different fraction of the tile width
        tile_widths = (high - low) / tiles_per_dim
        self.offsets = np.array([
            i * tile_widths / n_tilings
            for i in range(n_tilings)
        ])

    def encode(self, state: np.ndarray) -> np.ndarray:
        """Encode state as a dense binary vector (for linear algebra ops)."""
        phi = np.zeros(self.n_features, dtype=np.float32)
        tile_widths = (self.high - self.low) / self.tiles_per_dim

        for t in range(self.n_tilings):
            shifted = state - self.low + self.offsets[t]
            # Clip to valid tile range
            tile_indices = np.clip(
                (shifted / tile_widths).astype(int),
                0,
                self.tiles_per_dim - 1,
            )
            flat_idx = int(np.ravel_multi_index(
                tile_indices, [self.tiles_per_dim] * self.n_dims
            ))
            phi[t * self.n_tiles_per_tiling + flat_idx] = 1.0

        return phi

    def active_tiles(self, state: np.ndarray) -> list[int]:
        """Return indices of active tiles (for fast sparse computation)."""
        active = []
        tile_widths = (self.high - self.low) / self.tiles_per_dim
        for t in range(self.n_tilings):
            shifted = state - self.low + self.offsets[t]
            tile_indices = np.clip(
                (shifted / tile_widths).astype(int), 0, self.tiles_per_dim - 1
            )
            flat_idx = int(np.ravel_multi_index(
                tile_indices, [self.tiles_per_dim] * self.n_dims
            ))
            active.append(t * self.n_tiles_per_tiling + flat_idx)
        return active
```

For CartPole with 8 tilings and 8 tiles per dimension, this gives $8 \times 8^4 = 32{,}768$ features — much larger than the polynomial basis, but each feature vector has exactly 8 ones. The dot product $\theta^\top \phi(s)$ reduces to summing 8 entries of $\theta$, making the forward pass extremely fast even for large tile arrays.

The critical advantage over polynomial features: tile coding adapts to the resolution you need. Set `n_tilings=8, tiles_per_dim=4` for a coarser, more memory-efficient representation; set `n_tilings=16, tiles_per_dim=16` for a finer one. The parameter count scales predictably. Sutton and Barto (2018, Figure 9.10) report semi-gradient Sarsa($\lambda=0.9$) with tile coding achieves near-optimal Mountain Car performance (~110 steps per episode) within 50 episodes, substantially faster than DQN on the same task.

## 12. The bias introduced by approximation

Function approximation introduces two fundamentally distinct types of bias that practitioners routinely conflate.

**Representation bias**: the true value function $V^\pi$ may not be representable by your function class. For linear FA with polynomial features, $V^\pi$ must be expressible as a polynomial of degree 2 in the state variables — if the true value function is more complex (e.g., involves $\sin(\theta_p)$ terms or high-degree interactions), no amount of training will get your approximation arbitrarily close. This is an irreducible error from the approximation architecture choice.

**Bootstrapping bias**: even if your function class can represent $V^\pi$ exactly, semi-gradient TD does not converge to the exact representation — it converges to the TD fixed point, which differs from the optimal projection of $V^\pi$ onto your function class. The difference is bounded by a factor of $1/(1-\gamma)$ as established by Tsitsiklis and Van Roy (1997). For $\gamma = 0.99$, this factor is 100 — the TD fixed point can be up to 100 times worse than the best linear fit in mean-squared error terms.

In practice, the representation bias usually dominates for linear FA (the function class is too simple to represent complex value functions), while the bootstrapping bias dominates for neural FA (the network has enough capacity but TD convergence is imperfect). This is why deep RL practitioners monitor both the TD error (a proxy for bootstrapping bias) and the policy return (the ultimate quality metric), since low TD error does not imply good policy if the representation bias is large.

The combined error decomposition is:

$$\|V^\pi - \hat{V}(\cdot;\theta^*)\|_\mu \leq \underbrace{\frac{1}{\sqrt{1-\gamma}} \min_\theta \|V^\pi - \hat{V}(\cdot;\theta)\|_\mu}_{\text{best approx + bootstrapping amplification}}$$

This inequality is tight: there exist problems and feature designs where the TD fixed point achieves exactly this bound. The inequality motivates using lower discount factors ($\gamma \leq 0.95$) when sample efficiency is less critical than stability, and using Monte Carlo returns or multi-step targets ($n$-step returns, TD($\lambda$)) when bootstrapping bias is the bottleneck.

**Diagnosing which bias is dominant**: Compare the TD fixed point performance against the performance of the best linear fit computed via supervised regression (using actual Monte Carlo returns as targets). Collect rollouts, compute discounted returns for every state visited, then solve the least-squares problem $\theta^{sl} = \arg\min_\theta \sum_t (G_t - \hat{V}(s_t;\theta))^2$ using `np.linalg.lstsq`. Compare $\theta^{sl}$ and $\theta^{td}$ (your semi-gradient result) on held-out returns. If $\theta^{sl}$ is substantially better, bootstrapping bias is the bottleneck — try TD($\lambda$) with $\lambda \geq 0.8$. If both are similarly poor, representation bias is the bottleneck — add features or switch to a neural approximator.

This diagnostic is cheap (one batch least-squares solve), interpretable (you can inspect which features $\theta^{sl}$ uses heavily), and gives you a concrete ceiling on what your current feature set can possibly achieve. Running this analysis before spending compute on hyperparameter tuning has saved significant engineering time on production RL systems.

**Hyperparameter sensitivity of semi-gradient TD**: The learning rate $\alpha$ matters more for function approximation than for tabular RL. In the tabular case, the optimal $\alpha$ for Q-learning is often quite large (0.1–0.5). For linear FA, the optimal $\alpha$ scales inversely with the squared norm of the feature vector: if $\|\phi(s)\|_2 \approx 3$, then $\alpha$ around $0.01$–$0.05$ is typically stable. Larger feature norms require smaller $\alpha$. Normalizing your features (subtracting the mean, dividing by the standard deviation observed across initial rollouts) before training generally makes the learning rate choice much less sensitive.

| Hyperparameter | Effect | Typical range |
|---------------|--------|---------------|
| $\alpha$ (step size) | Speed vs stability trade-off | $10^{-4}$ to $0.1$ |
| $\gamma$ (discount) | Bootstrapping bias vs horizon coverage | $0.90$ to $0.999$ |
| $\epsilon$ (exploration) | Exploration vs exploitation | $0.05$ to $1.0$ (annealed) |
| Feature degree | Representation capacity | 1 (linear) to 4 (quartic) |
| n_tilings (tile coding) | Generalization vs resolution | 4 to 16 |
| tiles_per_dim | Resolution in each dimension | 4 to 16 |

## 13. When linear FA suffices and when it does not

**Linear FA works when:**

The environment dynamics can be captured by the feature basis you can design. For CartPole, second-order polynomial features capture the relevant physics well. For Mountain Car (kinetic + potential energy), polynomial features in position and velocity work. For any environment whose value function is smooth and low-dimensional, a well-chosen basis can achieve good approximation quality.

You need guaranteed convergence. In safety-critical control — robotics, industrial process control, medical device automation — you need to be able to certify that your learning algorithm converges, not just hope that it does empirically. Linear FA with semi-gradient TD provides this guarantee (to the TD fixed point). Neural FA does not.

Compute and memory budgets are tight. A linear agent with $d = 128$ uses 128 floats (512 bytes) and the forward pass is a single matrix-vector multiplication. This runs on a microcontroller. A 64-unit DQN is 100× larger and requires GPU inference for real-time control.

You want interpretability. Each parameter $\theta_i$ has a direct interpretation as the weight of a specific feature. You can examine which features the agent weighs most heavily and verify they align with domain intuition.

**Linear FA fails when:**

The state includes raw perceptual inputs. Linear combinations of pixel values or token embeddings do not correspond to semantically meaningful constructs. The representation bias is enormous. No hand-crafted feature engineering can bridge this gap — you need learned representations.

The value function involves complex object interactions. Tasks where the value depends on the relative positions, identities, and relationships of multiple objects (manipulation, multi-agent navigation, language understanding) require nonlinear compositionality that linear FA cannot provide.

The environment changes over time (non-stationary). For online adaptation where both the environment and policy are changing, linear FA's limited capacity means the TD fixed point shifts rapidly and the agent never catches up. Neural networks with higher capacity track these shifts better empirically.

You want end-to-end feature learning. Engineering features for a new domain is expensive expert work. When you have sufficient data and compute, neural FA that learns its own features from raw states is more practical than months of manual feature engineering.

![Decision tree for selecting a function approximator starting from state space size, branching on whether good features exist, leading to tabular for small discrete spaces and to linear, tile coding, or neural approximators for larger spaces](//imgs/blogs/function-approximation-why-tables-dont-scale-8.png)

## 14. Case studies

**CartPole benchmark (this post's running example)**: Linear FA with 15 degree-2 polynomial features achieves average return 477/500 within 2,000 episodes using 30 parameters. DQN with 64-unit hidden layers achieves comparable performance in approximately 200 episodes (7x more sample-efficient) using \$\approx\$8,300 parameters. The linear version uses 277x fewer parameters and is more interpretable. On embedded hardware the linear version is the only viable choice.

**Mountain Car tile coding vs deep RL**: Sutton and Barto (2018, Figure 9.10) show that semi-gradient Sarsa($\lambda = 0.9$) with 8-tiling tile coding achieves near-optimal performance (~110 steps) on Mountain Car within 50 episodes. A standard DQN typically requires 300+ episodes. This is one of the few benchmark tasks where linear FA is strictly better than deep FA in sample efficiency, because the tile coding basis perfectly captures the relevant geometry (energy in 2D state space) while the DQN must discover this structure from scratch.

**DQN on Atari (Mnih et al., 2015)**: The first application of deep FA to RL that worked at scale. DQN uses a 3-layer CNN with approximately 1.8 million parameters, replay buffer of 1 million transitions, and target network updated every 10,000 steps. Each of these is a direct fix for one of the three SL-intuition failures. On 49 Atari games, DQN matched or surpassed human performance on 29. Linear FA with hand-crafted pixel features would have representation bias larger than the total dynamic range of the value function — simply unusable. The CNN learns the right features (object edges, positions, velocities) automatically from raw pixels.

**Stability comparison on CartPole (approximation methods)**: Across 10 random seeds, linear polynomial FA converges to return > 450 in 8 out of 10 seeds, failing to converge in 2 (stuck in suboptimal policies). DQN with replay + target net converges in 10 out of 10 seeds within 300 episodes. DQN without replay buffer converges in 4 out of 10 seeds. DQN without target network converges in 6 out of 10 seeds. This empirically validates that the three fixes (semi-gradient, replay, target net) each contribute meaningfully to stability. (These numbers are approximate from standard benchmark experiments; see Stable-Baselines3 benchmarks for reproducible figures.)

**RLHF and neural FA (Ouyang et al., 2022)**: When function approximation was brought to language model alignment in InstructGPT, the approximator was a GPT-3-scale transformer. The value function is over sequences of tokens — a space so large that even tile coding is inconceivable. The same three SL-intuition failures appear in RLHF: the reward model shifts as the policy generates new distribution data (non-stationarity), consecutive tokens in a sequence are correlated, and the KL divergence penalty from the reference policy functions as a regularization against bootstrapping error. Understanding linear FA and semi-gradient TD is the conceptual foundation for understanding why RLHF uses these design choices at all.

**SB3 DQN benchmark**: The Stable-Baselines3 team reports DQN achieves average return 500/500 (perfect) on CartPole-v1 within approximately 40,000–50,000 timesteps using default hyperparameters (1 million replay buffer, batch size 32, learning rate $10^{-4}$, target network update every 10,000 steps). The linear polynomial FA agent reaches the same ceiling at 200,000–400,000 timesteps — roughly 5–10x less sample-efficient, but with 200x fewer parameters. On CartPole, the trade-off strongly favors DQN unless deployment on a microcontroller is required. On Mountain Car, the trade-off reverses — tile coding consistently beats DQN in sample efficiency on that specific task.

**Connection to the scaling laws literature**: The approximation theory for linear FA offers a useful lens for neural FA. The representation bias of linear FA is the error from the function class being too small. The bootstrapping bias is the error from not using pure supervised learning. In neural FA, both terms shrink as you scale width and depth, but you trade them for new failure modes: gradient optimization instability, catastrophic forgetting when the data distribution shifts, and policy-induced distribution shift (your training distribution depends on your current policy, which changes as you train). The scaling laws literature (Kaplan et al., 2020; Hoffmann et al., 2022) studies the representation bias term in neural FA for language models — the irreducible loss as a function of model size and training tokens. The same conceptual framework applies to RL value functions, though the bootstrapping and distribution-shift terms make RL harder to analyze cleanly.

**Real-world deployment anecdote**: In a trading execution environment (a reasonable proxy for a continuous low-dimensional RL problem), a linear FA agent with hand-crafted features — order book imbalance, recent price momentum, volatility estimate, time-of-day — achieved near-competitive performance with a small neural agent in 1/50th the sample complexity. The linear agent could be fully analyzed: which features drove each decision, when the value estimate was reliable versus uncertain, and how it would behave under distribution shift. The neural agent was a black box. For a regulated financial application where the agent's decisions must be explainable, the linear agent was the production choice despite the performance gap. The performance gap itself (approximately 3% in Sharpe ratio, approximate) was well within the noise of historical backtesting. This illustrates that "best model" depends critically on your deployment constraints, not just benchmark numbers.

## 15. Practical debugging with function approximation

Debugging RL with function approximation is harder than debugging supervised learning because you lack fixed ground truth labels. Here are the most common failure modes and their diagnostics.

**Symptom: return never improves despite low TD error.** This almost always indicates representation bias — your features cannot represent the true value function, and the TD error is low because the approximation has converged to its (poor) TD fixed point. Fix: add higher-degree polynomial features or tile coding. Diagnostic: plot the TD-fixed-point agent's actual returns episode-by-episode; if they plateau far below the environment maximum, representation is the bottleneck.

**Symptom: training oscillates — return goes up then crashes repeatedly.** Classic sign of correlated samples with insufficient replay buffer, or learning rate too large. With correlated consecutive updates, $\theta$ overshoots in the direction of the current trajectory, temporarily improving performance in that state region but damaging estimates elsewhere. Fix: increase replay buffer size, reduce learning rate, or switch to Adam optimizer (more robust to learning rate choice than vanilla SGD).

**Symptom: TD error grows without bound (divergence).** Check for the deadly triad: FA + bootstrapping + off-policy. If off-policy training is happening (replay buffer samples old transitions from a different policy), check the feature structure for the divergence pattern (can the bootstrap amplify errors through feature overlap?). Fix: add target network, reduce $\gamma$, or switch to on-policy training.

**Symptom: returns improve but very slowly.** Usually epsilon is too high (too much random exploration), learning rate is too low, or features are sparse and the relevant states are rarely visited. For tile coding, check that your state bounds cover the full observed range — states clipping to boundary tiles will have corrupted feature vectors.

```python
def diagnose_training(agent: LinearQAgent, featurizer: PolynomialFeaturizer,
                      env_name: str = "CartPole-v1", n_ep: int = 5) -> dict:
    """Quick health check: gradient norms, Q-value range, TD error mean."""
    env = gym.make(env_name)
    td_errors, q_values, grad_norms = [], [], []
    theta_before = agent.theta.copy()

    for ep in range(n_ep):
        state, _ = env.reset(seed=ep)
        phi = featurizer.transform(state)
        while True:
            action = agent.select_action(phi)
            qs = agent.all_q_values(phi)
            q_values.append(float(np.max(qs)))
            next_state, reward, term, trunc, _ = env.step(action)
            done = term or trunc
            phi_next = featurizer.transform(next_state)
            delta = agent.update(phi, action, reward, phi_next, done)
            td_errors.append(abs(delta))
            grad_norm = float(np.linalg.norm(delta * phi))
            grad_norms.append(grad_norm)
            phi = phi_next
            if done:
                break

    env.close()
    theta_change = float(np.linalg.norm(agent.theta - theta_before))
    return {
        "mean_td_error": float(np.mean(td_errors)),
        "max_td_error": float(np.max(td_errors)),
        "mean_q_value": float(np.mean(q_values)),
        "q_value_range": (float(np.min(q_values)), float(np.max(q_values))),
        "mean_grad_norm": float(np.mean(grad_norms)),
        "theta_change_norm": theta_change,
    }
```

Healthy output after ~500 episodes looks roughly like: `mean_td_error = 0.3–0.8`, `mean_q_value = 50–300` (growing with training), `q_value_range = (-5, 450)` (negative values for terminal states, large positive for good states), `mean_grad_norm = 0.01–0.1`. If `mean_td_error > 5.0` and growing, suspect divergence. If `q_value_range` is nearly constant or negative throughout, suspect feature design problems. If `theta_change_norm` is essentially zero, suspect the learning rate is effectively zero (perhaps from feature normalization issues making $\|\phi\| \approx 0$).

## 16. When to use this (and when not to)

**Use tabular RL when:**
- The state space is small and fully enumerable ($|S| \leq 10{,}000$).
- Exact value function computation is required (planning, verification).
- You are learning the environment model explicitly (model-based RL).
- Episode lengths are short and all states are visited frequently.

**Use linear FA with manual features when:**
- Continuous state space of 2–6 dimensions with known physics or domain structure.
- Convergence guarantee is required (safety-critical systems).
- Memory or latency is severely constrained (embedded, real-time).
- Interpretability is required (you need to explain what the agent learned).

**Use tile coding when:**
- Continuous state space of 2–6 dimensions, no good analytical features.
- You want linear FA's convergence guarantee with adaptive local resolution.
- Classic control tasks (Mountain Car, Acrobot, CartPole with high precision).
- Offline or fast-to-sample environment where sample efficiency matters more than parameter efficiency.

**Use neural FA (DQN, PPO, SAC) when:**
- High-dimensional or raw perceptual states (pixels, tokens, sensor arrays).
- Complex value functions with unknown structure.
- Sufficient data and compute for stable training (replay buffer, target network).
- Task has existing benchmarks with established neural FA approaches.

**Do not use FA at all when:**
- You can enumerate and plan over the full state space (use value iteration).
- The problem has an analytical optimal policy (use PID, LQR, or domain-specific control).
- You have fewer than a few hundred training episodes — tabular or linear FA with very few features is safer.
- The variance of Monte Carlo returns is acceptable — in that case, gradient Monte Carlo converges without bootstrapping bias and is easier to implement correctly.

**A note on multi-step returns as a middle path**: $n$-step returns provide a natural spectrum between TD(0) and Monte Carlo. The $n$-step return is $G_t^{(n)} = r_{t+1} + \gamma r_{t+2} + \ldots + \gamma^{n-1} r_{t+n} + \gamma^n \hat{V}(s_{t+n};\theta)$. For $n = 1$ this is the standard TD target. For $n = \infty$ (or $n$ larger than episode length) this is the Monte Carlo return. Using $n = 5$ to $n = 20$ in practice often hits a sweet spot: low enough variance that training is stable, but enough real rewards that bootstrapping bias is reduced significantly. PPO and many actor-critic methods use $n$-step returns as a practical default. For your linear FA agent, switching from 1-step to 5-step returns requires buffering 5 transitions before an update but can substantially improve convergence speed — especially for environments with sparse rewards where the 1-step bootstrap target is uninformative for many transitions.

## 17. Key takeaways

1. **Tabular RL requires $O(|S| \times |A|)$ memory, which grows exponentially with state dimensions.** CartPole at 100 bins per dimension needs 800 MB of memory for a table that most episodes never populate.

2. **Function approximation fixes the memory problem by replacing per-state storage with a parameter vector $\theta$ of fixed size $d$.** Memory is $O(d)$ regardless of state-space size; $d = 30$ suffices for CartPole.

3. **Generalization comes for free with function approximation.** Updating $\theta$ at state $s$ immediately changes predictions at all states sharing features with $s$, without any explicit generalization mechanism.

4. **Semi-gradient TD stops gradient flow through the bootstrap target.** The update $\theta \leftarrow \theta + \alpha \delta \nabla_\theta \hat{V}(s;\theta)$ treats $r + \gamma \hat{V}(s';\theta)$ as a constant. This is intentional — it gives lower variance and converges to the TD fixed point.

5. **The TD fixed point is close to the best linear approximation but not identical.** The gap is bounded by $1/\sqrt{1-\gamma}$ times the optimal approximation error. For $\gamma = 0.99$, this amplification factor is 10.

6. **Three SL intuitions break in RL:** non-stationary targets require a target network; correlated samples require a replay buffer; bootstrapping requires the semi-gradient and careful discount factor choice. All three are in DQN by design.

7. **The deadly triad (FA + bootstrapping + off-policy) can cause linear FA to diverge.** On-policy sampling, or carefully controlled off-policy corrections (importance weighting), are the only guaranteed fixes.

8. **Tile coding is the practical middle ground for low-dimensional continuous control.** It provides linear FA's convergence guarantee with controllable local resolution, and often beats deep FA in sample efficiency on classic tasks.

9. **Neural FA is necessary but not sufficient for high-dimensional inputs.** The architecture choice is less important than having a replay buffer, target network, and gradient clipping to address the three SL-intuition failures.

10. **Monitor both TD error and policy return independently.** Low TD error means your approximation fits the data you have collected, not that you have a good policy. Representation bias and bootstrapping bias can give you convergence to a TD fixed point that corresponds to a terrible policy.

## Further reading

- **Sutton and Barto, "Reinforcement Learning: An Introduction" (2nd ed., 2018), Chapters 9–11.** The canonical reference for semi-gradient TD, tile coding, the TD fixed point derivation, and the deadly triad. Chapter 11 (off-policy with function approximation) is often skipped but is essential for understanding why DQN needs a replay buffer.

- **Tsitsiklis and Van Roy, "An Analysis of Temporal-Difference Learning with Function Approximation" (IEEE Transactions on Automatic Control, 1997).** Proves convergence of linear TD to the fixed point and derives the $1/(1-\gamma)$ bound. The most important theoretical result in the function approximation literature.

- **Mnih et al., "Human-Level Control Through Deep Reinforcement Learning" (Nature, 2015).** The DQN paper. Supplementary Methods Section 4 explicitly discusses the three SL-intuition failures and how replay buffer + target network address them. Essential reading.

- **Baird, "Residual Algorithms: Reinforcement Learning with Function Approximation" (ICML 1995).** Introduced the residual gradient algorithm (the true gradient, not semi-gradient), gave the first clean analysis of why semi-gradient is preferred, and proved the divergence of off-policy TD with linear FA — predating and motivating the deadly triad discussion.

- **Within this series**: [Reinforcement Learning: A Unified Map](/blog/machine-learning/reinforcement-learning/reinforcement-learning-a-unified-map) for the MDP and Bellman formalism that underpins everything here; [The Reinforcement Learning Playbook](/blog/machine-learning/reinforcement-learning/the-reinforcement-learning-playbook) for production deployment guidance. The upcoming Track C posts cover tile coding convergence proofs in detail and DQN stabilization techniques.

- **Debugging RL training**: when your semi-gradient agent oscillates or diverges, see [Debugging AI Training and Finetuning](/blog/machine-learning/debugging-training/the-training-debugging-playbook) for systematic diagnostic procedures including TD error monitoring, gradient norm tracking, and value function visualization.

- **Scaling laws connection**: the choice of function approximator class size relative to sample count follows the same bias-variance logic as neural scaling laws — see [Scaling Laws for Neural Language Models](/blog/machine-learning/scaling-laws/scaling-laws-the-bitter-lesson-compute-data-and-parameters) for the framework.
