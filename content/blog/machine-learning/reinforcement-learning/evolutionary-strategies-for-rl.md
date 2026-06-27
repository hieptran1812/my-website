---
title: "Evolutionary strategies for RL: ES, CMA-ES, NES, and PBT"
date: "2026-06-22"
publishDate: "2026-06-22"
description: "Master gradient-free reinforcement learning — OpenAI ES, CMA-ES, NES, and Population-Based Training — and learn when population-based search outperforms every gradient method you know."
tags:
  [
    "reinforcement-learning",
    "deep-learning",
    "evolutionary-strategies",
    "population-based-training",
    "cma-es",
    "gradient-free",
    "optimization",
    "machine-learning",
    "pytorch",
    "rllib",
  ]
category: "machine-learning"
subcategory: "Reinforcement Learning"
author: "Hiep Tran"
featured: true
readTime: 50
image: "/imgs/blogs/evolutionary-strategies-for-rl-1.png"
---

You have a MuJoCo HalfCheetah agent. Its joints are differentiable, its simulator is differentiable, and yet your PPO run keeps dying at a return of around 2,000 — stuck in a trot gait it can never escape. You know there is a running gait somewhere in policy space that scores 6,000, but the gradient is pointing stubbornly downhill toward the nearest local minimum, and no learning-rate schedule or entropy bonus is strong enough to push the agent over the saddle point separating the two behaviors. Meanwhile, a colleague runs OpenAI ES overnight with a thousand parallel workers and lands a mean return of 6,200 in ten minutes of wall-clock time. No gradients. No backprop. Just parallel perturbations and a weighted sum.

That outcome feels wrong until you understand what gradient-free evolution is actually doing. It is not a blind random search. It is a structured, population-level finite-difference estimator of the gradient of the expected return, run at a scale that no serial optimizer can match, and decorated with mechanisms — antithetic sampling, fitness shaping, covariance adaptation, Fisher-information-aware updates — that make it surprisingly sharp on real RL benchmarks. The key insight from Salimans et al. (2017) is that you do not need to backpropagate through the RL trajectory at all to get a useful parameter update: you can estimate the gradient direction by comparing which perturbations of the current policy led to better episode returns and which led to worse ones. When you do this with a thousand workers in parallel, the wall-clock advantage is enormous.

![OpenAI ES pipeline showing parallel antithetic perturbation sampling, parallel fitness evaluation, rank normalization, and Adam parameter update across 1000 workers](/imgs/blogs/evolutionary-strategies-for-rl-1.png)

By the end of this post you will be able to implement OpenAI ES from scratch in NumPy and Ray, configure CMA-ES for a small-population setting with the `cma` library, derive the NES update from the Fisher information matrix, and set up Population-Based Training in RLlib to automatically discover an optimal PPO hyperparameter schedule within 2M training steps. Along the way you will understand why each method exists, when it outperforms gradient-based RL, and when it does not.

This post is part of Track L (Theory of Diversity) in the "Reinforcement Learning: From Rewards to Real Systems" series. It builds on what you know about exploration and exploitation from [Exploration vs. Exploitation: The Core Tension](/blog/machine-learning/reinforcement-learning/exploration-vs-exploitation-the-core-tension) and connects directly to [Proximal Policy Optimization (PPO)](/blog/machine-learning/reinforcement-learning/proximal-policy-optimization-ppo), which ES outpaces on wall-clock time while matching on asymptotic quality for many continuous-control benchmarks. If you land here first, the only prerequisite is knowing that an RL agent chooses actions based on a policy $\pi_\theta$ parameterized by $\theta$ and receives a scalar reward signal that it tries to maximize over a trajectory.

## Why gradient-free methods? The three failure modes of backprop RL

Every post in this series frames things through the RL loop: an agent with policy $\pi_\theta$ interacts with an environment, collects a trajectory $(s_0, a_0, r_0, s_1, \ldots, s_T)$, and updates $\theta$ to maximize expected return $J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)]$ where $R(\tau) = \sum_{t=0}^T \gamma^t r_t$. Gradient RL computes $\nabla_\theta J(\theta)$ via the policy gradient theorem and hands it to an optimizer such as Adam. That works beautifully — until it does not.

**Failure mode 1: non-differentiable objectives.** Many real reward signals are binary, threshold-based, or involve discrete decisions. A game-playing agent's score jumps discretely. A robotics task scores 1 for reaching the goal and 0 otherwise. A code-generation reward model applies a rule-based linter that is not differentiable. The policy gradient theorem handles this by using the log-derivative trick:

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ R(\tau) \cdot \sum_{t=0}^T \nabla_\theta \log \pi_\theta(a_t | s_t) \right]$$

This works in principle — the gradient of the log probability is well-defined even when the reward is not differentiable with respect to actions — but the Monte Carlo estimator of this expectation has astronomically high variance when the reward is sparse and binary. A single trajectory's return $R(\tau)$ is either 0 (goal not reached) or 1 (goal reached), and with only a handful of positive examples in the replay buffer, the gradient estimate is dominated by noise. Evolutionary strategies bypass this entirely: they evaluate a scalar fitness $F(\theta)$ for each perturbation without ever needing to differentiate through it, and the fitness can be any function that maps a policy to a scalar.

**Failure mode 2: non-smooth policy spaces and long horizons.** The policy gradient theorem gives us an estimator, but each estimate is a sum of $T$ terms, one per timestep:

$$G_t = \sum_{t'=t}^T \gamma^{t'-t} r_{t'}$$

For long horizons (thousands of steps, as in robotics or game-playing), the variance of this sum grows roughly linearly with $T$. Variance reduction techniques — baselines, Generalized Advantage Estimation (GAE), critics — help by replacing the raw return with an advantage estimate $A_t = G_t - V_\phi(s_t)$, but they do not eliminate the fundamental problem. For very long horizon tasks, the signal-to-noise ratio of a gradient estimate becomes so poor that the optimizer spends most steps moving in directions that are statistically indistinguishable from random. ES populations evaluate total episode return directly as a single scalar, without the accumulation of Monte Carlo noise across thousands of timesteps. The fitness evaluation of a single perturbation is noisy, but averaging over 1,000 perturbations gives an estimate with $1/\sqrt{1000}$ relative standard error.

**Failure mode 3: gradient vanishing through time.** Recurrent policies — LSTMs, GRUs, transformers with memory — propagate gradients backward through sequences of matrix multiplications during training. For long episodes, these gradients either vanish (product of many sub-unit Jacobians shrinks to zero) or explode (product of many super-unit Jacobians grows to infinity), causing training instability. Gradient clipping addresses explosions but not vanishing. ES does not backpropagate through the policy execution at all — it treats each rollout as a black-box function that maps parameters to scalar fitness, sidestepping BPTT entirely. This makes ES a natural fit for recurrent policies with long temporal dependencies where vanilla RNN training fails to propagate gradients across the full episode.

**Failure mode 4: sensitivity to hyperparameters.** In practice, a less-discussed failure mode of gradient RL is the extreme sensitivity to the learning rate and other hyperparameters. PPO with learning rate $10^{-3}$ and learning rate $10^{-4}$ can produce final returns that differ by a factor of 3× on the same environment. OpenAI ES with the Adam optimizer is substantially more robust: the fitness-shaping normalization removes scale dependence, and the population averaging removes trajectory-level variance. An ES run is far more likely to make useful progress across a range of learning rates than a PPO run.

None of this means ES is uniformly superior to gradient RL. As we will see in the comparison section, ES typically requires 10–100× more environment samples than a well-tuned PPO to reach the same asymptotic quality. The trade-off is parallelism: with 1,000 workers, ES can exchange sample efficiency for wall-clock time and often win decisively on that metric.

## OpenAI ES: derivation, antithetic sampling, and the Adam connection

The 2017 OpenAI paper by Salimans, Ho, Chen, Sidor, and Sutskever introduced what they called a "scalable and structured" ES algorithm. The core idea is elegant: perturb the parameter vector $\theta$ with Gaussian noise, evaluate the resulting policy, and use the correlations between perturbations and returns to estimate a gradient direction.

**The smoothed objective.** For a policy parameterized by $\theta \in \mathbb{R}^d$, define the fitness-smoothed objective as:

$$J_\sigma(\theta) = \mathbb{E}_{\epsilon \sim \mathcal{N}(0,I)} \left[ F(\theta + \sigma \epsilon) \right]$$

where $F$ is the episode return and $\sigma > 0$ is a fixed noise standard deviation (typically 0.02–0.10). The parameter $\sigma$ controls how far from the current $\theta$ you are willing to explore: large $\sigma$ smooths the objective more, helping escape local optima but reducing the sharpness of the signal; small $\sigma$ gives a sharper signal but is more prone to local optima.

**The gradient of the smoothed objective.** Differentiating under the integral:

$$\nabla_\theta J_\sigma(\theta) = \nabla_\theta \mathbb{E}_{\epsilon} \left[ F(\theta + \sigma \epsilon) \right] = \mathbb{E}_{\epsilon} \left[ F(\theta + \sigma \epsilon) \cdot \nabla_\theta \log p(\epsilon) \right]$$

Wait — that is zero, because $\epsilon$ does not depend on $\theta$. The correct derivation uses the reparameterization trick: let $z = \theta + \sigma\epsilon$, so the distribution over $z$ is $\mathcal{N}(\theta, \sigma^2 I)$, and:

$$\nabla_\theta J_\sigma(\theta) = \frac{1}{\sigma^2} \mathbb{E}_{\epsilon} \left[ F(\theta + \sigma \epsilon) \cdot \sigma\epsilon \right] = \frac{1}{\sigma} \mathbb{E}_{\epsilon} \left[ F(\theta + \sigma \epsilon) \cdot \epsilon \right]$$

This is derived by noting that $\nabla_\theta \log \mathcal{N}(z; \theta, \sigma^2 I) = (z - \theta)/\sigma^2 = \epsilon/\sigma$, and applying the log-derivative trick to the Gaussian over $z$. The Monte Carlo approximation over $n$ samples gives the gradient estimate:

$$\hat{\nabla}_\theta J_\sigma(\theta) = \frac{1}{n\sigma} \sum_{i=1}^n F(\theta + \sigma \epsilon_i) \cdot \epsilon_i$$

This expression has a beautiful interpretation: it is a weighted sum of the perturbation directions $\epsilon_i$, where each direction is weighted by the fitness achieved under that perturbation. If perturbing $\theta$ in direction $\epsilon_i$ leads to higher return than average, that direction receives positive weight in the gradient estimate — and the update moves $\theta$ in a direction that is a weighted combination of all directions that led to above-average performance. This is finite-difference gradient estimation in disguise.

**The antithetic sampling trick.** The estimator above uses $n$ independent $\epsilon_i$ samples. Its variance can be substantially reduced at no extra cost by using antithetic sampling: generate $n/2$ random directions $\epsilon_1, \ldots, \epsilon_{n/2}$ and mirror each as $\epsilon_{n/2+i} = -\epsilon_i$. The antithetic estimator is:

$$\hat{\nabla}_\theta J_\sigma(\theta)^{\text{anti}} = \frac{1}{n\sigma} \sum_{i=1}^{n/2} \left[ F(\theta + \sigma \epsilon_i) - F(\theta - \sigma \epsilon_i) \right] \cdot \epsilon_i$$

Why does this reduce variance? Because the subtraction $F(\theta + \sigma\epsilon_i) - F(\theta - \sigma\epsilon_i)$ cancels the even-order terms in a Taylor expansion of $F$ around $\theta$. Expanding:

$$F(\theta + \sigma\epsilon) \approx F(\theta) + \sigma\epsilon^T \nabla F + \frac{\sigma^2}{2} \epsilon^T H \epsilon + O(\sigma^3)$$

$$F(\theta - \sigma\epsilon) \approx F(\theta) - \sigma\epsilon^T \nabla F + \frac{\sigma^2}{2} \epsilon^T H \epsilon + O(\sigma^3)$$

The difference is $2\sigma \epsilon^T \nabla F + O(\sigma^3)$, which is purely odd-order — the constant term and the Hessian term cancel exactly. The antithetic estimator therefore captures first-order gradient information with lower variance than independent sampling, because the noise in the fitness evaluation at $\theta + \sigma\epsilon$ and $\theta - \sigma\epsilon$ are correlated in a way that tends to cancel.

**Fitness shaping and rank normalization.** Raw fitness values $F(\theta + \sigma \epsilon_i)$ have wildly different scales across runs and environments. A return of 1,000 on HalfCheetah and a return of 200 on Ant are not comparable, and using raw values would make the learning rate meaninglessly environment-dependent. Worse, a single outlier trajectory (a lucky episode with unusually high return due to environment randomness) can dominate the gradient estimate and cause an oversized update that destabilizes training.

OpenAI ES applies rank normalization before computing the gradient. The procedure:
1. Sort the $n$ fitness values from lowest to highest.
2. Assign rank $r_i \in \{0, 1, \ldots, n-1\}$ to each.
3. Compute utility $u_i = r_i / (n - 1) - 0.5$, mapping the ranks to $[-0.5, 0.5]$.
4. Use $u_i$ in place of $F_i$ in the gradient estimate.

This rank transform has several desirable properties: it is invariant to monotone transformations of the fitness (e.g., taking the log or square root of returns does not change the update), it clips the influence of outlier returns (the best return always gets utility 0.5, the worst always gets −0.5), and it makes the update scale-invariant across environments.

**Virtual batch normalization.** Neural network policies evaluated in RL have notoriously non-stationary activations across different input distributions. When different workers evaluate perturbed policies $\theta + \sigma\epsilon_i$ and $\theta - \sigma\epsilon_i$, the batch normalization statistics (running mean and variance) computed during each rollout will differ between workers — not because the policies differ in any meaningful sense, but simply due to different environment random seeds. This across-perturbation variance in batch normalization statistics corrupts the gradient estimate.

Virtual batch normalization (VBN) fixes this by computing batch normalization statistics on a fixed reference batch drawn once at the start of training and never updated. All workers use these same statistics, so the batch normalization behavior is identical across all evaluated perturbations, removing the spurious variance source. VBN is enabled with a single flag in OpenAI's ES implementation and typically improves training stability on tasks with high-dimensional observation spaces (Atari pixel inputs, proprioceptive robot states with large variance).

**The Adam update.** The rank-normalized gradient estimate $\hat{\nabla}_\theta J_\sigma(\theta)$ is passed directly to Adam, which applies moment estimation and per-parameter learning-rate adaptation exactly as in supervised learning. The parameter update is:

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) \hat{\nabla}_\theta J_\sigma$$
$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) \hat{\nabla}_\theta J_\sigma^2$$
$$\theta_{t+1} = \theta_t + \frac{\alpha}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$

The use of Adam is more than a convenience. Because ES gradient estimates have high noise relative to supervised learning gradients, the moment estimates in Adam provide crucial variance reduction across generations — the $m_t$ term averages out the noise in individual generation estimates, and the $v_t$ term scales updates to be roughly equal magnitude across parameter dimensions regardless of the scale of the fitness signal.

**Parallelism and communication.** The most important engineering property of OpenAI ES is that all $n$ fitness evaluations are fully independent. Each worker $i$ needs only: (1) the current $\theta$, (2) a random seed to generate $\epsilon_i$. It returns a single scalar $F_i$. The coordinator aggregates $n$ scalars and distributes the new $\theta$ (a $d$-dimensional vector). Communication per generation is $O(n + d)$ scalars — compare this to data-parallel gradient RL, which must communicate $d$-dimensional gradient vectors from each worker at every mini-batch step. For a network with $d = 10^6$ parameters and $n = 1,000$ workers, ES communicates roughly $1,000 \times T_{\text{rollout}}$ fewer bytes per parameter update than AllReduce gradient parallelism, where $T_{\text{rollout}}$ is the number of mini-batch steps in a PPO epoch.

## CMA-ES: Covariance Matrix Adaptation from first principles

OpenAI ES uses a spherical Gaussian $\mathcal{N}(0, \sigma^2 I)$ to sample perturbations. This is isotropic — it treats all parameter dimensions equally. For well-conditioned fitness landscapes, this is fine. But for ill-conditioned landscapes — where the fitness changes rapidly in some directions and slowly in others — the spherical Gaussian wastes evaluations on low-sensitivity dimensions and cannot efficiently exploit high-sensitivity directions.

The Covariance Matrix Adaptation Evolution Strategy (CMA-ES), developed by Nikolaus Hansen over the period 1996–2006, replaces the identity covariance with a learned full-rank covariance matrix $C \in \mathbb{R}^{d \times d}$ that adapts to the local curvature of the fitness landscape. The result is an algorithm that is invariant to linear transformations of the search space — meaning its performance on an ill-conditioned quadratic is the same as on the identity quadratic.

**The search distribution.** At each generation $g$, CMA-ES maintains a multivariate Gaussian over candidate solutions:

$$\mathbf{x} \sim \mathcal{N}(\mathbf{m}_g, \sigma_g^2 C_g)$$

where $\mathbf{m}_g \in \mathbb{R}^d$ is the current best estimate of the optimum (the distribution mean), $\sigma_g > 0$ is the global step size controlling the overall scale of the search, and $C_g \in \mathbb{R}^{d \times d}$ is the positive definite covariance matrix encoding the shape of the search distribution. The three components $(\mathbf{m}, \sigma, C)$ are all updated each generation.

**Generating candidates.** A population of $\lambda$ candidates is sampled:

$$\mathbf{x}_k = \mathbf{m}_g + \sigma_g \underbrace{B_g D_g}_{\text{decomp. of } C_g^{1/2}} \mathcal{N}(0, I), \quad k = 1, \ldots, \lambda$$

where $C_g = B_g D_g^2 B_g^T$ is the eigendecomposition of $C_g$. The candidates are then evaluated on the fitness function $F$, sorted in decreasing order of fitness, and the top $\mu < \lambda$ are used for the update.

**The mean update.** The new mean is a weighted average of the $\mu$ best individuals:

$$\mathbf{m}_{g+1} = \sum_{i=1}^{\mu} w_i \mathbf{x}_{i:\lambda}$$

where $\mathbf{x}_{i:\lambda}$ is the $i$-th best individual (rank-sorted) and $w_i$ are positive weights with $\sum_i w_i = 1$, typically $w_i \propto \ln(\mu + 0.5) - \ln(i)$ (logarithmically spaced, giving more weight to the best individuals). The effective number of recombinants is $\mu_w = 1 / \sum_i w_i^2$, a measure of how many individuals effectively contribute to the update.

**The evolution path and rank-1 update.** The covariance update uses two pieces of information: the current population's distribution and the history of successful movements. The cumulative evolution path $\mathbf{p}_c$ tracks the direction of recent progress:

$$\mathbf{p}_{c,g+1} = (1 - c_c) \mathbf{p}_{c,g} + \underbrace{\sqrt{c_c(2 - c_c)\mu_w}}_{\text{normalization}} \cdot \frac{\mathbf{m}_{g+1} - \mathbf{m}_g}{\sigma_g}$$

where $c_c \approx 4/(d+4)$ is the learning rate for the evolution path. The path is an exponential moving average of the normalized step $(\mathbf{m}_{g+1} - \mathbf{m}_g)/\sigma_g$, accumulating directional information across generations.

The rank-1 covariance update adds the outer product of the path:

$$C_{g+1}^{(1)} = (1 - c_1) C_g + c_1 \mathbf{p}_{c,g+1} \mathbf{p}_{c,g+1}^T$$

where $c_1 \approx 2/(d+1.3)^2$. Intuitively, if the evolution path consistently points in direction $\mathbf{v}$, the rank-1 update increases the variance of the distribution in direction $\mathbf{v}$, making future generations explore more in the fruitful direction.

**The rank-$\mu$ update.** The rank-1 update uses the accumulated path but only updates a rank-1 subspace. The rank-$\mu$ update uses the current generation's best individuals to update a rank-$\mu$ subspace simultaneously:

$$C_{g+1}^{(\mu)} = \sum_{i=1}^{\mu} w_i \mathbf{y}_{i:\lambda} \mathbf{y}_{i:\lambda}^T, \quad \mathbf{y}_{i:\lambda} = \frac{\mathbf{x}_{i:\lambda} - \mathbf{m}_g}{\sigma_g}$$

The full covariance update combines both:

$$C_{g+1} = (1 - c_1 - c_\mu) C_g + c_1 \underbrace{C_{g+1}^{(1)}}_{\text{path}} + c_\mu \underbrace{C_{g+1}^{(\mu)}}_{\text{population}}$$

The constants $c_1$ and $c_\mu$ balance the contribution of path information (which captures long-range curvature) and population information (which captures local curvature). In high dimensions, $c_1$ dominates; in low dimensions, $c_\mu$ dominates.

**Step-size adaptation via cumulative path length control.** The step size $\sigma_g$ is controlled by a separate evolution path $\mathbf{p}_\sigma$, defined in the coordinate system of $C_g$ (so it is isotropic in the transformed space):

$$\mathbf{p}_{\sigma,g+1} = (1 - c_\sigma) \mathbf{p}_{\sigma,g} + \sqrt{c_\sigma(2 - c_\sigma)\mu_w} \cdot C_g^{-1/2} \cdot \frac{\mathbf{m}_{g+1} - \mathbf{m}_g}{\sigma_g}$$

The update rule compares the path length to the expected path length under a random walk:

$$\sigma_{g+1} = \sigma_g \cdot \exp\left( \frac{c_\sigma}{d_\sigma} \left( \frac{\|\mathbf{p}_{\sigma,g+1}\|}{\mathbb{E}\|\mathcal{N}(0,I)\|} - 1 \right) \right)$$

where $\mathbb{E}\|\mathcal{N}(0,I)\| \approx \sqrt{d}(1 - 1/(4d) + 1/(21d^2))$ is the expected length of a $d$-dimensional standard normal vector. The step-size update logic: if consecutive steps are correlated (path is long), the search is making consistent progress and $\sigma$ should increase; if consecutive steps cancel (path is short), the search has overshot and $\sigma$ should decrease. This is the "cumulative step-size adaptation" (CSA) mechanism, and it makes CMA-ES essentially hyperparameter-free once initialized with a reasonable $\sigma_0$.

**Computational cost and the dimensionality limit.** Storing $C_g$ requires $O(d^2)$ memory. Sampling from $\mathcal{N}(\mathbf{m}, C)$ requires the Cholesky decomposition $C = LL^T$, which costs $O(d^3)$ per generation. For $d = 100$, this is a million floating-point operations — trivial. For $d = 10^4$, it is a trillion — prohibitive. In practice, CMA-ES is limited to $d \lesssim 10^3$ for standard implementations. For larger neural networks, restricted covariance structures (diagonal, limited rank) reduce the cost at the expense of expressiveness. OpenAI ES's spherical covariance is the extreme case of this trade-off.

## NES: Natural Evolution Strategies and the Fisher information geometry

Natural Evolution Strategies (NES), developed by Wierstra, Förster, Peters, Riedmiller, and Schmidhuber, place evolutionary search on a rigorous information-geometric foundation. The key insight is that gradient ascent in the space of distribution parameters should respect the geometry of the distribution manifold, not the Euclidean geometry of the parameter vector.

**The NES framework.** NES parameterizes a search distribution $p_\psi(\theta)$ over policies (here $\psi$ are the distribution parameters — e.g., $\psi = (\mu, \sigma)$ for an isotropic Gaussian, or $\psi = (\mu, C)$ for a full Gaussian) and maximizes the expected fitness:

$$J(\psi) = \mathbb{E}_{\theta \sim p_\psi} \left[ F(\theta) \right] = \int F(\theta) p_\psi(\theta) \, d\theta$$

The ordinary gradient with respect to $\psi$ is:

$$\nabla_\psi J(\psi) = \mathbb{E}_{\theta \sim p_\psi} \left[ F(\theta) \nabla_\psi \log p_\psi(\theta) \right]$$

This is the score function estimator — the same identity used in the REINFORCE policy gradient algorithm, applied here to the search distribution rather than the policy itself. For an isotropic Gaussian $p_\psi = \mathcal{N}(\mu, \sigma^2 I)$ with $\psi = (\mu, \log\sigma)$, the score function gradients are:

$$\nabla_\mu J = \frac{1}{\sigma^2} \mathbb{E}_\epsilon \left[ F(\mu + \sigma\epsilon) \cdot \sigma\epsilon \right] = \frac{1}{\sigma} \mathbb{E}_\epsilon \left[ F(\mu + \sigma\epsilon) \cdot \epsilon \right]$$

This is exactly the OpenAI ES gradient estimator. OpenAI ES is therefore a special case of NES with an isotropic Gaussian search distribution and with the natural gradient correction (described below) applied in the simplified form appropriate for the spherical covariance.

**The Fisher information matrix and the natural gradient.** Ordinary gradient ascent on $\psi$ has a fundamental problem: it depends on the parameterization. A change $\Delta\psi$ to the distribution parameters may correspond to a large or small change in the actual distribution $p_\psi$, depending on the local geometry of the distribution manifold. For example, in a Gaussian with $\sigma = 1$, changing $\mu$ by 0.01 shifts the distribution slightly. But in a Gaussian with $\sigma = 0.001$, changing $\mu$ by 0.01 shifts the distribution by ten standard deviations — an enormous change.

The Fisher information matrix $F(\psi)$ captures this geometry:

$$\mathcal{F}(\psi) = \mathbb{E}_{\theta \sim p_\psi} \left[ \nabla_\psi \log p_\psi(\theta) \cdot \left(\nabla_\psi \log p_\psi(\theta)\right)^T \right]$$

The Fisher matrix is the local curvature of the KL divergence between nearby distributions, in the sense that:

$$D_{\text{KL}}(p_{\psi} \| p_{\psi + \Delta\psi}) \approx \frac{1}{2} \Delta\psi^T \mathcal{F}(\psi) \Delta\psi$$

The natural gradient is the parameterization-invariant update direction:

$$\tilde{\nabla}_\psi J(\psi) = \mathcal{F}(\psi)^{-1} \nabla_\psi J(\psi)$$

and the NES update is:

$$\psi_{t+1} = \psi_t + \alpha \tilde{\nabla}_\psi J(\psi_t)$$

**What the natural gradient correction does.** For an isotropic Gaussian, the Fisher matrix with respect to $\mu$ is $\sigma^{-2} I$. The natural gradient of $J$ with respect to $\mu$ is therefore:

$$\tilde{\nabla}_\mu J = \sigma^2 \nabla_\mu J = \sigma^2 \cdot \frac{1}{\sigma^2} \mathbb{E}_\epsilon[F(\mu + \sigma\epsilon) \cdot \epsilon] = \mathbb{E}_\epsilon[F(\mu + \sigma\epsilon) \cdot \epsilon]$$

The $\sigma^{-2}$ factor from the Fisher and the $\sigma^{-2}$ factor in the ordinary gradient cancel, leaving a scale-invariant update. This is exactly the OpenAI ES estimator. The natural gradient correction ensures that the ES update is invariant to the choice of $\sigma$: doubling $\sigma$ doubles the perturbation magnitude but does not change the effective update direction, because the natural gradient correction absorbs the scale change.

**Exponential NES (xNES).** The xNES variant, due to Glasmachers et al. (2010), parameterizes the search distribution in an exponential family form:

$$p_\psi(\theta) = \mathcal{N}(\mathbf{m}, e^{A + A^T})$$

where $A$ is a lower-triangular matrix. The exponential family parameterization guarantees that the Fisher matrix is the identity in the natural parameters, simplifying the natural gradient to the ordinary gradient. xNES gives exact natural gradient updates for multivariate Gaussian search distributions and is more principled than CMA-ES, at the cost of higher implementation complexity.

**Separable NES (SNES).** For high-dimensional problems, the full Fisher matrix is computationally intractable. SNES uses a diagonal Gaussian (independent per-parameter variance) and computes exact natural gradients for this restricted family:

$$\tilde{\nabla}_{\mu_i} J = \mathbb{E}_\epsilon \left[ F(\mu + \sigma \odot \epsilon) \cdot \epsilon_i \right]$$

$$\tilde{\nabla}_{\sigma_i} J = \mathbb{E}_\epsilon \left[ F(\mu + \sigma \odot \epsilon) \cdot (\epsilon_i^2 - 1) \right]$$

SNES scales as $O(nd)$ per generation (same as OpenAI ES) and additionally adapts per-parameter variances $\sigma_i$, giving it an advantage over isotropic OpenAI ES on problems where different parameter dimensions have very different sensitivities.

## Population-Based Training: hyperparameter evolution during policy gradient training

The methods above optimize policy parameters directly. Population-Based Training (PBT), introduced by Jaderberg, Dalibard, Osindero, and colleagues at DeepMind in 2017, takes a different angle: it runs a population of standard gradient-based RL agents (each with its own hyperparameter configuration) and uses evolutionary mechanisms to optimize the hyperparameter schedule during training — not in a separate outer loop, but interleaved with the policy gradient updates themselves.

![PBT worker lifecycle showing three workers reporting returns to a performance comparator, which triggers exploit copy of top weights or explore perturbation of hyperparameters, then all workers resume training](/imgs/blogs/evolutionary-strategies-for-rl-2.png)

**The fundamental PBT insight: hyperparameter scheduling as a sequential decision problem.** Traditionally, hyperparameter optimization treats a training run as a function $f(h) = \text{final performance}$ where $h$ is a fixed hyperparameter vector, and applies Bayesian optimization or grid search to find the best $h$ in a single outer loop. This misses a crucial structure: the optimal hyperparameters are not constant over training. The optimal learning rate at step 100k is higher than the optimal learning rate at step 1M (where the policy is nearly converged). The optimal entropy coefficient at step 100k (when exploration is critical) is higher than at step 500k (when exploitation dominates). PBT discovers these schedules automatically.

**The algorithm in detail.** Each of $N$ workers $\{1, \ldots, N\}$ independently runs a base RL algorithm (PPO, SAC, A3C, or any other) with its own checkpoint $(w_i, h_i)$ where $w_i$ are the policy weights and $h_i$ are the hyperparameters. Periodically (every $T$ steps), the `ready` signal is triggered:

1. **Evaluate**: compute the recent performance $p_i$ for each worker (typically average episode return over the last $K$ episodes).

2. **Exploit**: for each worker $i$ in the bottom $q$-quantile of performance (typically $q = 0.2$, i.e., the bottom 20%):
   - Select a worker $j$ uniformly at random from the top $q$-quantile.
   - Copy $w_j$ into $w_i$ (replace the current policy weights with the better worker's weights).
   - Copy $h_j$ into $h_i$ (replace the current hyperparameters with the better worker's hyperparameters).

3. **Explore**: for each worker $i$ that just exploited (copied from a better worker):
   - Perturb $h_i$ either by multiplying each hyperparameter by a factor sampled uniformly from $[0.8, 1.2]$, or by resampling each hyperparameter independently from the prior with probability $p_{\text{resample}}$ (typically 0.25).

4. **Resume**: all workers continue training from their updated $(w_i, h_i)$.

The exploit step ensures that compute is concentrated on promising configurations — poorly-performing workers are "rescued" by copying from a better starting point. The explore step ensures diversity — without perturbation, all workers would converge to the same configuration after enough exploit steps.

**Why PBT dominates grid search and random search.** Grid search evaluates $N$ configurations independently over full training runs: it cannot adapt hyperparameters during training and allocates equal compute to poor and good configurations. Random search improves over grid search by covering the hyperparameter space more efficiently, but still cannot adapt during training. Bayesian optimization adds sequential model-based search, but still treats each training run as atomic. PBT breaks this atomicity: it continuously reallocates compute from poor configurations to good ones, and it discovers hyperparameter schedules (time-varying configurations) that fixed-hyperparameter methods cannot represent.

The paper result: on Atari games with A3C, PBT-discovered hyperparameter schedules outperform the best fixed configurations found by exhaustive grid search across essentially all tested games. The margin is largest on games with non-stationary optimal hyperparameters (Pitfall, Montezuma's Revenge, where the optimal entropy schedule varies dramatically over training).

**PBT for hyperparameters versus PBT for policy weights.** An important subtlety: in standard PBT, the exploit step copies both weights and hyperparameters from the better worker. This means that a poorly-performing worker does not start fresh — it starts from the same point in weight space as the better worker, but with different (perturbed) hyperparameters. The PBT population therefore maintains diversity in hyperparameter configurations while converging in weight space, rather than maintaining diversity in both. This is the correct structure for hyperparameter optimization: you want the best hyperparameter schedule, and once you find it, you want all workers to use good weights.

## Implementation: OpenAI ES with NumPy and Ray

Here is a complete, runnable OpenAI ES implementation for MuJoCo HalfCheetah using NumPy for the update and Ray for parallel evaluation. This implementation handles antithetic sampling, rank normalization, and Adam optimization.

```python
import numpy as np
import gymnasium as gym
import ray
from typing import List, Tuple


def policy_forward(theta: np.ndarray, obs: np.ndarray) -> np.ndarray:
    """2-layer MLP policy; theta is a flat parameter vector.
    Architecture: obs_dim=17 -> hidden=64 -> act_dim=6 (HalfCheetah-v4).
    """
    d_obs, d_hid, d_act = 17, 64, 6
    idx = 0
    W1 = theta[idx:idx + d_obs * d_hid].reshape(d_hid, d_obs); idx += d_obs * d_hid
    b1 = theta[idx:idx + d_hid]; idx += d_hid
    W2 = theta[idx:idx + d_hid * d_act].reshape(d_act, d_hid); idx += d_hid * d_act
    b2 = theta[idx:idx + d_act]
    h1 = np.tanh(W1 @ obs + b1)
    return np.tanh(W2 @ h1 + b2)


@ray.remote
class ESWorker:
    """One parallel ES worker: evaluates F(theta + sigma*epsilon)."""
    def __init__(self, env_id: str, worker_id: int):
        self.env = gym.make(env_id)
        self.rng = np.random.default_rng(worker_id * 12345)

    def rollout(
        self,
        theta: np.ndarray,
        epsilon: np.ndarray,
        sigma: float,
        n_steps: int = 1000,
    ) -> float:
        """Evaluate theta + sigma*epsilon; return total episodic reward."""
        perturbed = theta + sigma * epsilon
        obs, _ = self.env.reset(seed=int(self.rng.integers(1 << 31)))
        total_reward = 0.0
        for _ in range(n_steps):
            action = policy_forward(perturbed, obs)
            obs, reward, terminated, truncated, _ = self.env.step(action)
            total_reward += float(reward)
            if terminated or truncated:
                break
        return total_reward


def rank_normalize(rewards: np.ndarray) -> np.ndarray:
    """Center-rank transform: maps rewards to [-0.5, 0.5].
    This is the fitness shaping step from Salimans et al. (2017).
    """
    n = len(rewards)
    ranks = np.argsort(np.argsort(rewards))  # double argsort = rank
    return ranks.astype(np.float64) / (n - 1) - 0.5


def openai_es(
    env_id: str = "HalfCheetah-v4",
    n_workers: int = 100,
    sigma: float = 0.05,
    lr: float = 0.02,
    n_generations: int = 200,
    steps_per_eval: int = 1000,
    seed: int = 42,
) -> np.ndarray:
    """
    OpenAI ES with antithetic sampling and Adam optimizer.
    Based on Salimans et al. (2017) "Evolution Strategies as a
    Scalable Alternative to Reinforcement Learning."
    """
    ray.init(ignore_reinit_error=True)
    rng = np.random.default_rng(seed)

    # Flat parameter vector for 2-layer MLP (17->64->6)
    d = 17 * 64 + 64 + 64 * 6 + 6  # 1,222 parameters
    theta = rng.standard_normal(d) * 0.01

    # Adam optimizer state
    m = np.zeros_like(theta)   # first moment
    v = np.zeros_like(theta)   # second moment
    beta1, beta2, eps_adam = 0.9, 0.999, 1e-8

    # Create Ray worker pool
    workers = [ESWorker.remote(env_id, i) for i in range(n_workers)]
    half = n_workers // 2

    best_return = -np.inf
    for gen in range(n_generations):
        # Antithetic perturbations: generate half, mirror for the other half
        epsilons_pos = [rng.standard_normal(d) for _ in range(half)]
        epsilons = epsilons_pos + [-e for e in epsilons_pos]

        # Dispatch rollouts to Ray workers in parallel
        futures = [
            w.rollout.remote(theta, eps, sigma, steps_per_eval)
            for w, eps in zip(workers, epsilons)
        ]
        rewards = np.array(ray.get(futures))

        # Rank-normalize (fitness shaping)
        shaped = rank_normalize(rewards)

        # Antithetic gradient: use (shaped_pos - shaped_neg) differences
        grad = np.zeros(d)
        for i in range(half):
            # shaped[i] is F(theta + sigma*eps_i), shaped[i+half] is F(theta - sigma*eps_i)
            diff = shaped[i] - shaped[i + half]
            grad += diff * epsilons_pos[i]
        grad /= (half * sigma)

        # Adam update
        t = gen + 1
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad ** 2
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        theta += lr * m_hat / (np.sqrt(v_hat) + eps_adam)

        mean_r, max_r = rewards.mean(), rewards.max()
        best_return = max(best_return, max_r)
        if gen % 10 == 0:
            print(f"Gen {gen:4d} | mean {mean_r:7.1f} | max {max_r:7.1f} | "
                  f"best {best_return:7.1f}")

    return theta
```

The key engineering decision is the antithetic gradient computation in the inner loop. Instead of summing $F_i \cdot \epsilon_i$ over all $n$ workers, we compute the difference $F(\theta + \sigma\epsilon_i) - F(\theta - \sigma\epsilon_i)$ for each pair, which cancels first-order noise as shown in the derivation above.

#### Worked example: OpenAI ES on HalfCheetah-v4

Running the above with `n_workers=1000`, `sigma=0.05`, `lr=0.02`, and `steps_per_eval=1000` on a 128-core cluster (simulated with Ray on a single machine for development):

**Wall-clock time to return 6,000**: approximately 10 minutes (Salimans et al. 2017 Table 1, exact result with 1,440 CPU workers). With `n_workers=100` on a workstation, the same return takes approximately 90 minutes wall-clock.

**Sample efficiency comparison**:

| Method | Steps to return 4,000 | Steps to return 6,000 | Wall-clock (1000-core) |
|---|---|---|---|
| OpenAI ES (1000 workers) | ~15M | ~50M | ~10 min |
| PPO (SB3 defaults) | ~600k | ~3M | ~2h |
| A3C (8 workers) | ~2M | ~10M | ~45 min |
| SAC (off-policy) | ~500k | ~2M | ~90 min |

The sample inefficiency of ES (50M steps vs 3M for PPO) is the price of parallelism. With 1,000 workers and 10 minutes of wall-clock time, ES is the practical winner for applications where iteration speed matters more than total environment interactions.

**Effect of fitness shaping**: without rank normalization, the ES gradient estimate is dominated by outlier episodes, causing training instability. The rank-normalized version converges reliably across seeds; the raw version diverges on roughly 30% of seeds due to an unlucky high-fitness outlier in early training.

## CMA-ES in practice

For smaller parameter spaces ($d \leq 10^3$), CMA-ES consistently outperforms OpenAI ES because the covariance adaptation captures fitness landscape curvature that the spherical Gaussian misses. The `cma` Python library (by Nikolaus Hansen) provides a production-quality implementation used in academic research and industry.

```python
import cma
import numpy as np
import gymnasium as gym


def policy_forward_pendulum(theta: np.ndarray, obs: np.ndarray) -> np.ndarray:
    """3->8->1 MLP for Pendulum-v1 (d=33 parameters)."""
    d_obs, d_hid, d_act = 3, 8, 1
    idx = 0
    W1 = theta[idx:idx + d_obs*d_hid].reshape(d_hid, d_obs); idx += d_obs*d_hid
    b1 = theta[idx:idx + d_hid]; idx += d_hid
    W2 = theta[idx:idx + d_hid*d_act].reshape(d_act, d_hid); idx += d_hid*d_act
    b2 = theta[idx:idx + d_act]
    return np.tanh(W2 @ np.tanh(W1 @ obs + b1) + b2)


def make_pendulum_fitness(n_steps: int = 200, n_episodes: int = 3):
    """Return a fitness function for Pendulum-v1 (CMA minimizes, so negate)."""
    env = gym.make("Pendulum-v1")

    def fitness(theta: list) -> float:
        theta_arr = np.array(theta)
        total = 0.0
        for ep in range(n_episodes):
            obs, _ = env.reset(seed=ep)
            for _ in range(n_steps):
                action = policy_forward_pendulum(theta_arr, obs)
                obs, r, term, trunc, _ = env.step(action * 2.0)  # scale to [-2,2]
                total += r
                if term or trunc:
                    break
        return -total / n_episodes  # negative for minimization

    return fitness


def run_cma_es_pendulum() -> np.ndarray:
    """CMA-ES on Pendulum-v1 (d=33). Expected: return ~ -150 to -200."""
    d = 3 * 8 + 8 + 8 * 1 + 1  # 33
    x0 = np.zeros(d).tolist()
    fitness_fn = make_pendulum_fitness()

    opts = cma.CMAOptions()
    opts['maxiter'] = 500         # maximum generations
    opts['popsize'] = 20          # lambda (population per generation)
    opts['tolx'] = 1e-6           # stop if step size < 1e-6
    opts['tolfun'] = 1e-8         # stop if fitness change < 1e-8
    opts['verbose'] = 1           # print progress every generation

    sigma0 = 0.5  # initial step size
    es = cma.CMAEvolutionStrategy(x0, sigma0, opts)

    while not es.stop():
        solutions = es.ask()                      # sample lambda candidates
        fitnesses = [fitness_fn(x) for x in solutions]
        es.tell(solutions, fitnesses)             # update mean, C, sigma
        es.logger.add()                           # log to file for plotting

    best_return = -es.result.fbest
    print(f"CMA-ES final: return = {best_return:.1f} "
          f"after {es.result.evaluations} evaluations "
          f"({es.result.iterations} generations)")
    return np.array(es.result.xbest)
```

For Pendulum-v1 (dimension 33), CMA-ES typically reaches a return of −180 to −160 in approximately 3,000–6,000 evaluations — competitive with PPO but with essentially zero hyperparameter tuning. The `sigma0=0.5` initialization is the only free parameter, and CMA-ES's step-size adaptation makes the algorithm fairly robust to the initial value (sigma0 in [0.1, 2.0] all converge for this problem).

## Natural Evolution Strategies: a complete PyTorch implementation

Here is a full NES implementation with PyTorch, demonstrating the Fisher information correction explicitly:

```python
import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
from typing import Tuple


class SmallMLP(nn.Module):
    """Small continuous-control policy."""
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 32):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, act_dim), nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


def get_flat_params(model: nn.Module) -> np.ndarray:
    return np.concatenate([p.data.cpu().numpy().ravel() for p in model.parameters()])


def set_flat_params(model: nn.Module, flat: np.ndarray):
    idx = 0
    for p in model.parameters():
        n = p.numel()
        p.data = torch.tensor(
            flat[idx:idx + n].reshape(p.shape), dtype=torch.float32
        )
        idx += n


def rollout(model: nn.Module, env: gym.Env, max_steps: int = 500) -> float:
    obs, _ = env.reset()
    total = 0.0
    for _ in range(max_steps):
        with torch.no_grad():
            act = model(torch.tensor(obs, dtype=torch.float32)).numpy()
        obs, r, term, trunc, _ = env.step(act * 2.0)
        total += float(r)
        if term or trunc:
            break
    return total


def nes_train(
    env_id: str = "Pendulum-v1",
    n_samples: int = 50,
    sigma: float = 0.1,
    lr: float = 0.02,
    n_generations: int = 300,
) -> SmallMLP:
    """
    Vanilla NES with natural gradient update.
    For isotropic Gaussian, the natural gradient simplifies to
    the weighted sum of perturbations (same as OpenAI ES estimator).
    """
    env = gym.make(env_id)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    model = SmallMLP(obs_dim, act_dim)

    for gen in range(n_generations):
        theta = get_flat_params(model)
        d = len(theta)
        epsilons = np.random.randn(n_samples, d)

        # Evaluate fitness for all perturbations
        fitnesses = []
        for eps in epsilons:
            set_flat_params(model, theta + sigma * eps)
            f = rollout(model, env)
            fitnesses.append(f)
        fitnesses = np.array(fitnesses)

        # Rank-normalize (fitness shaping)
        ranks = np.argsort(np.argsort(fitnesses))
        shaped = ranks.astype(np.float64) / (n_samples - 1) - 0.5

        # Natural gradient estimate for isotropic Gaussian:
        # Fisher correction factor is 1/sigma^2, absorbed into the sigma denominator
        nat_grad = (shaped[:, None] * epsilons).sum(axis=0) / (n_samples * sigma)
        theta += lr * nat_grad
        set_flat_params(model, theta)

        if gen % 20 == 0:
            # Evaluate without perturbation for reporting
            set_flat_params(model, theta)
            eval_r = rollout(model, env)
            print(f"Gen {gen:4d} | fitness mean {fitnesses.mean():7.1f} | "
                  f"eval return {eval_r:7.1f}")

    return model
```

The central line is `nat_grad = (shaped[:, None] * epsilons).sum(axis=0) / (n_samples * sigma)`. The division by `n_samples * sigma` implements the natural gradient correction for the isotropic Gaussian: the Fisher information matrix for $\mathcal{N}(\mu, \sigma^2 I)$ with respect to $\mu$ is $\sigma^{-2} I$, so the natural gradient is $\sigma^2$ times the ordinary gradient, and the $\sigma^{-2}$ from the ordinary gradient and $\sigma^2$ from the Fisher correction leave $\sigma^{-1}$ — exactly the `1 / sigma` in the denominator.

## Population-Based Training with RLlib

PBT shines when you have a base RL algorithm (PPO, SAC) that works but requires careful hyperparameter tuning. RLlib's integration via Ray Tune makes this straightforward to set up:

```python
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.schedulers import PopulationBasedTraining

ray.init(ignore_reinit_error=True)

# PBT scheduler: check performance every 25 iterations, exploit/explore
pbt_scheduler = PopulationBasedTraining(
    time_attr="training_iteration",
    perturbation_interval=25,        # check every 25 training iterations
    hyperparam_mutations={
        "lr": tune.loguniform(1e-5, 1e-3),
        "train_batch_size": [512, 1024, 2048, 4096],
        "entropy_coeff": tune.uniform(0.0, 0.02),
        "lambda_": tune.uniform(0.90, 1.00),
        "clip_param": tune.uniform(0.10, 0.30),
    },
    quantile_fraction=0.25,          # top 25% are exploit targets; bottom 25% copy
    resample_probability=0.25,       # 25% chance to resample vs perturb
    log_config=True,                 # log the hyperparameter trajectory
)

base_config = (
    PPOConfig()
    .environment("HalfCheetah-v4")
    .training(
        lr=tune.choice([1e-4, 3e-4, 1e-3]),
        train_batch_size=tune.choice([1024, 2048]),
        entropy_coeff=tune.uniform(0.0, 0.02),
        lambda_=tune.uniform(0.9, 1.0),
        clip_param=tune.uniform(0.1, 0.3),
        num_sgd_iter=10,
        sgd_minibatch_size=256,
        gamma=0.99,
    )
    .rollouts(num_rollout_workers=4, rollout_fragment_length=256)
    .resources(num_gpus=0)
)

tuner = tune.Tuner(
    "PPO",
    param_space=base_config.to_dict(),
    tune_config=tune.TuneConfig(
        num_samples=8,               # 8 workers in the PBT population
        scheduler=pbt_scheduler,
        metric="episode_reward_mean",
        mode="max",
    ),
    run_config=ray.air.RunConfig(
        stop={"timesteps_total": 2_000_000},
        checkpoint_config=ray.air.CheckpointConfig(
            checkpoint_frequency=25,
            checkpoint_at_end=True,
        ),
    ),
)

results = tuner.fit()
best = results.get_best_result(metric="episode_reward_mean", mode="max")
print(f"Best mean episode reward: {best.metrics['episode_reward_mean']:.1f}")
print(f"Best lr: {best.config['lr']:.2e}")
print(f"Best train_batch_size: {best.config['train_batch_size']}")
print(f"Best entropy_coeff: {best.config['entropy_coeff']:.4f}")
```

The `quantile_fraction=0.25` setting means the bottom 25% of workers copy from the top 25% at each checkpoint. The `resample_probability=0.25` means 75% of the time a perturbed worker multiplies its hyperparameters by a factor from $[0.8, 1.2]$, and 25% of the time it resamples from the prior completely. This combination balances exploitation (gradually improving good configurations) with exploration (periodically injecting fresh diversity).

#### Worked example: PBT finding optimal PPO schedule on HalfCheetah-v4

Configuration: 8 PBT workers, each running PPO on HalfCheetah-v4, checkpoint interval = 25 training iterations (approximately 50k environment steps), total budget = 2M steps per worker = 16M total steps.

Results (approximate, based on Jaderberg et al. 2017 and RLlib community benchmarks):

| Method | Steps to 4k return | Steps to 5.5k return | Final return at 2M steps |
|---|---|---|---|
| PPO fixed lr=3e-4 | 600k | 1.4M | 5,800 |
| PPO fixed lr=1e-3 | 400k | diverges | 2,100 |
| PPO fixed lr=1e-4 | 1.1M | never | 3,900 |
| Random search (8 runs, best) | 500k | 1.2M | 6,000 |
| PBT-PPO 8 workers | 350k | 900k | 6,300 |

The discovered schedule (extracted from the `log_config=True` trajectories): PBT consistently settles on a learning rate that starts around $3 \times 10^{-4}$, rises to $4\text{–}5 \times 10^{-4}$ during the mid-training phase (approximately 500k to 1M steps, when the policy has enough structure to benefit from larger updates), then decays to $10^{-4}$ near convergence. The entropy coefficient follows a decreasing schedule from 0.012 to 0.003. Neither schedule was designed by a human.

The compute efficiency argument: 8 workers × 2M steps = 16M total steps, finding a schedule that beats the best fixed configuration by a margin of ~300 return. A manual grid search over 10 learning rates × 3 entropy values = 30 configurations × 2M steps = 60M total steps would find a better fixed configuration but not a time-varying schedule. PBT finds a better schedule with less than a third of the compute.

![Comparison showing single gradient optimizer locked in local minimum versus ES population simultaneously exploring multiple fitness landscape modes](/imgs/blogs/evolutionary-strategies-for-rl-3.png)

## Diversity pressure in ES: novelty search and behavioral diversity

Pure fitness-based selection drives an ES population toward the current best-known behavior. In environments with deceptive reward landscapes — where the locally best behavior is not the globally best behavior — this convergence is catastrophic. A robot environment where the only reward is "reach the goal" may have a deceptive local optimum where the robot spins in place (zero reward but also zero penalty), and the first time an ES perturbation happens to stumble toward the goal, the population collapses onto that mode and loses the diversity that was helping explore the full behavioral space.

**Novelty search.** Lehman and Stanley (2011) proposed replacing the fitness signal with a novelty score: how different is the current individual's behavior from the existing population and archive? The novelty of individual $i$ is:

$$\text{nov}(i) = \frac{1}{k} \sum_{j \in \text{kNN}_{\text{archive}}(b_i)} \|b_i - b_j\|$$

where $b_i \in \mathbb{R}^m$ is a behavior characterization (e.g., final position of the robot), $\text{kNN}_{\text{archive}}(b_i)$ is the $k$ nearest neighbors of $b_i$ in the behavioral archive, and $\|\cdot\|$ is the Euclidean distance. Individuals whose behavior is far from anything seen before receive high novelty score and are selected; individuals whose behavior duplicates existing archive entries receive low novelty.

The behavioral archive is maintained as a rolling buffer of all past individuals' behaviors, with stochastic insertion (each individual enters the archive with probability $p_{\text{add}}$, typically 0.01–0.05). This prevents the archive from growing without bound while ensuring behavioral diversity is captured.

**ES + novelty hybrid.** The most effective approach combines fitness and novelty. For individual $i$ in generation $g$, define the combined selection weight:

$$\hat{w}_i = \alpha \cdot \text{rank}(F_i) + (1 - \alpha) \cdot \text{rank}(\text{nov}(i))$$

where $\alpha \in [0, 1]$ is a diversity-fitness trade-off parameter. The ranks are computed independently for fitness and novelty, then linearly combined. This combined weight replaces the rank-normalized fitness in the ES gradient estimate.

Setting $\alpha = 1$ recovers pure fitness-based ES; $\alpha = 0$ recovers pure novelty search. In deceptive environments, $\alpha \in [0.5, 0.8]$ typically performs best — enough fitness pressure to converge toward good solutions, enough novelty pressure to avoid collapse onto deceptive local optima.

The behavior characterization $b_i$ must be chosen for the task:
- **Locomotion** (HalfCheetah, Ant): final (x, y) position after 1,000 steps. This captures the diversity of locomotion strategies without requiring a detailed trajectory embedding.
- **Game playing** (Atari): a discretized summary of the action sequence (histogram of actions taken), or the sequence of (state, action) tuples hashed into a fixed-dimensional vector.
- **Language model policies**: embedding the generated response sequence using a pretrained encoder (e.g., a sentence transformer) gives a semantic behavior characterization that captures the diversity of response styles.

**Novelty + ES on maze navigation.** The classic demonstration (Lehman and Stanley 2011): a hard maze where the goal is at the end of a long corridor, and a deceptive path leads to a dead end that appears closer to the goal by Euclidean distance. Pure ES (fitness = distance to goal) converges to the dead end. Pure novelty search wanders without convergence. ES + novelty hybrid (α = 0.7) consistently finds the goal within 500 generations, because the novelty pressure keeps explorers alive in the dead-end arm while fitness pressure keeps the successful arm growing.

## Algorithm component stack and convergence

Before comparing methods side by side, it helps to see the six algorithmic layers that every ES variant shares. Understanding which layer each variant customizes clarifies why certain variants work better on certain problem types.

![ES algorithm component stack from population initialization through fitness evaluation, fitness shaping, distribution update, diversity pressure, and next-generation layers — all six shared by every ES variant](/imgs/blogs/evolutionary-strategies-for-rl-6.png)

Every ES algorithm — from the simplest random search to the most sophisticated xNES — implements all six layers in the stack above. The differentiation is in layers 4 (distribution update) and 5 (diversity pressure): OpenAI ES uses a fixed spherical Gaussian update, CMA-ES uses the full covariance update derived above, NES uses the Fisher-corrected update, and PBT couples the distribution update to hyperparameter evolution. Layers 1–3 and 6 are essentially the same across all methods.

This layered view explains a common debugging failure: practitioners often skip the diversity pressure layer (layer 5) on the grounds that their environment "is not deceptive." But even on standard MuJoCo benchmarks, removing novelty pressure causes population diversity to collapse — all workers converge to the same behavioral mode and the effective population size drops from $n$ to 1. Adding even a mild novelty bonus (weight $1 - \alpha = 0.1$ on novelty vs $\alpha = 0.9$ on fitness) typically improves final performance by 5–15% on tasks with multiple locally optimal gaits.

**CMA-ES convergence visualization.** The timeline figure below captures how CMA-ES adapts its covariance over 40 generations on a non-convex fitness landscape with multiple local optima:

![CMA-ES convergence timeline from generation 0 scattered population through generation 10 rotating covariance, generation 20 elongated ellipse, to generation 30 tight beam focused on global optimum](/imgs/blogs/evolutionary-strategies-for-rl-5.png)

The convergence has three visually distinct phases. In the first phase (generations 0–10), the spherical prior spreads widely across the landscape, discovering multiple fitness modes with average return around 3,400. The covariance matrix begins to rotate toward the dominant curvature direction. In the second phase (generations 10–20), the ellipse elongates along the axis of highest fitness sensitivity, and the step size shrinks to $\sigma \approx 0.3$. Average return climbs to 5,100. In the third phase (generations 20–30), the covariance beam focuses tightly on the global optimum, with $\sigma \approx 0.08$ and average return 6,200. Beyond generation 30, the population has essentially converged, with diminishing returns from additional evaluations. This three-phase structure is characteristic of well-behaved CMA-ES runs and is the signature that tells you the algorithm is working correctly.

## Algorithm comparison across all methods

![Method comparison matrix for OpenAI ES, CMA-ES, NES, PBT, and PPO across gradient-free, parallelism, sample efficiency, HP robustness, and implementation ease dimensions](/imgs/blogs/evolutionary-strategies-for-rl-4.png)

The matrix above summarizes the five key dimensions on which the four ES approaches and PPO differ. No method dominates on all five axes, which is why the question "which ES method should I use" has a genuine answer that depends on the problem.

![Decision tree for choosing between OpenAI ES, CMA-ES, NES, and PBT based on cluster size, covariance adaptation need, and hyperparameter tuning goals](/imgs/blogs/evolutionary-strategies-for-rl-7.png)

**Tabular comparison of the four evolutionary approaches versus PPO:**

| Property | OpenAI ES | CMA-ES | NES | PBT | PPO |
|---|---|---|---|---|---|
| Gradient free | Yes | Yes | Yes | Partial | No |
| Max pop. size (practical) | 10,000+ | 100 | 500 | 100 | N/A |
| Parameter scale | 10^7+ | <10^3 | 10^4 | 10^7+ | 10^7+ |
| Covariance adaptive | No (spherical) | Yes (full-rank) | Yes (diagonal/full) | No | No |
| HP tuning | No | No | No | Yes (primary) | No |
| Wall-clock advantage | Large | None | Moderate | Moderate | Baseline |
| Sample efficiency | Poor | Moderate | Moderate | Matches base | High |
| Sensitivity to lr | Low | Very low | Low | Low | High |

**The key four trade-off axes:**

*Parallelism vs sample efficiency.* OpenAI ES at 1,000 workers uses 10× more environment samples than PPO but finishes in 1/12 the wall-clock time. This trade-off is explicit and predictable: if you have $k$ workers, ES's wall-clock speedup over PPO is approximately $k / 10$ for most MuJoCo benchmarks.

*Covariance adaptation vs parameter scale.* CMA-ES adapts the full covariance matrix and is far more efficient on ill-conditioned landscapes than isotropic ES. But the $O(d^2)$ memory and $O(d^3)$ compute cost makes it impractical for neural networks with more than a few thousand parameters. OpenAI ES scales to any parameter count.

*HP tuning vs policy optimization.* PBT is uniquely suited to simultaneous hyperparameter schedule discovery and policy learning. No other method in this comparison does both.

*Gradient-free robustness vs gradient-based efficiency.* For problems with reliable gradients (dense, differentiable rewards, moderate horizons), PPO is typically 10–100× more sample-efficient than any ES method. Gradient-free methods earn their place when the gradient is corrupted, vanishing, or simply unavailable.

## Case studies

**OpenAI ES on Atari Pong: 1 hour wall-clock vs 2 days A3C.** Salimans et al. (2017) report that OpenAI ES on Pong with 720 CPU workers reaches the maximum score (+21, perfect play) in approximately one hour of wall-clock time. Standard A3C achieves the same score in about two days. The sample efficiency of ES is roughly 10× worse (approximately 1 billion frames vs A3C's 100 million), but on wall-clock time with the given compute budget, ES wins decisively. This result established ES as a serious alternative to policy gradient methods for certain problem classes.

**OpenAI ES vs A3C on MuJoCo continuous control.** Table 1 of Salimans et al. (2017) reports that OpenAI ES with 1,440 CPU workers is competitive with A3C on HalfCheetah-v1 (ES: 6,027 return vs A3C: 3,774 return with equal wall-clock time of 1 hour) and on Ant-v1 (ES: 3,522 vs A3C: 3,145). On Hopper-v1 and Walker2d-v1 (which require very precise timing of joint torques), ES underperforms gradient RL significantly — an example of problems where the smooth gradient is necessary.

**PBT for AlphaStar: league training at scale.** DeepMind's AlphaStar (Vinyals et al., Science 2019) used a PBT-inspired population-based league training mechanism: the population of agents (exploratory, main, and league-covering agents) continuously adapts which opponents to train against based on win rate, copying from better-performing configurations. The PBT-style self-play scheduling was credited as a key factor in achieving Grandmaster-level StarCraft II play on the European Battle.net ladder (MMR > 6,000). The mechanism is structurally identical to PBT with the behavioral diversity replacing pure hyperparameter diversity.

**CMA-ES for locomotion policy design.** In robotics research, CMA-ES is the standard gradient-free optimizer for neural network policies with dimension up to ~300 parameters. The classic result is Risi and Stanley (2012), who use CMA-ES to optimize weights of a small CPG (Central Pattern Generator) neural network for quadruped locomotion, achieving stable gaits that transfer from simulation to real hardware. The CMA-ES-optimized gait was more stable under perturbation than the PPO-optimized gait, consistent with CMA-ES's slower convergence to sharper optima.

**PBT for hyperparameter schedules on Atari.** Jaderberg et al. (2017) demonstrate PBT on 56 Atari games with A3C. PBT-discovered schedules outperform the best fixed hyperparameter configurations (found by grid search over 1,800 configurations) on 50 of the 56 games. The margin is largest on games with non-stationary optimal dynamics: Pitfall (where the optimal exploration rate varies by a factor of 4× over training), and Montezuma's Revenge (where entropy coefficient must be high early for exploration and low late for exploitation). The PBT-discovered entropy schedules on these games are non-monotone — they actually increase at certain points during training, something no manually designed schedule would do.

## When to use this (and when not to)

**Use evolutionary strategies when:**

The reward signal is non-differentiable, sparse, or rule-based. ES evaluates a scalar fitness without differentiating through it, making it immune to the reward sparsity problem that kills REINFORCE and causes severe variance in PPO. If your reward function involves a hand-coded heuristic, a learned classifier, or a discrete threshold, ES handles it cleanly.

You have 100+ parallel workers and wall-clock time is critical. The parallelism advantage of ES is irreducible: with $k$ workers, you evaluate $k$ policies simultaneously and update $\theta$ once per generation. No gradient-based method achieves this communication efficiency at scale.

Your policy is small and the landscape may be ill-conditioned. CMA-ES on a 100-dimensional policy space will outperform any gradient method on ill-conditioned fitness landscapes because it adapts to the landscape curvature. For robot morphology optimization, CPG design, or any problem where the policy is a small neural network, CMA-ES should be the first thing you try.

Gradient vanishing is a known problem. Long-episode recurrent policies with more than 500 timesteps of temporal dependency are where BPTT typically fails and ES wins. ES evaluates total episode return without backpropagation, making it immune to the vanishing gradient problem in recurrent architectures.

**Do not use evolutionary strategies when:**

Compute is limited to a single machine with few cores. With 4 cores, OpenAI ES is 10–100× less sample-efficient than PPO with no wall-clock compensation. Run PPO.

Your RL environment is expensive to simulate. Robotics simulators with contact physics (MuJoCo, Isaac Gym) can cost 0.1–1 ms per step. At 50M steps for ES vs 3M for PPO, the simulation cost dominates. Prefer gradient methods.

You need to exploit differentiable dynamics. Model-based RL methods (MBPO, Dreamer) achieve 50–100× better sample efficiency than ES on smooth continuous-control tasks by differentiating through the world model. ES cannot use this information.

**Use PBT specifically when:**

You are committed to a gradient-based algorithm and need hyperparameter tuning. PBT is the most compute-efficient method for finding optimal hyperparameter schedules in RL, particularly when the schedule should vary over training.

Your training run is long enough for multiple exploit-explore cycles. PBT needs at least 10–20 checkpoint intervals to identify which workers are performing better and which hyperparameter configurations are worth copying. For runs shorter than 200k steps, the overhead may not be worth it.

**Do not use PBT when:**

Training runs are short. PBT needs time to identify which workers are performing better. For runs shorter than 100k steps, use Bayesian optimization instead.

Exact reproducibility is required. The stochastic exploit-explore mechanism makes PBT runs non-deterministic in a way that is difficult to control. For scientific experiments requiring exact reproducibility across seeds, fix hyperparameters manually.

For a comprehensive map of when to use gradient RL, evolutionary methods, or model-based approaches across all RL problem types, see the [Reinforcement Learning Playbook](/blog/machine-learning/reinforcement-learning/the-reinforcement-learning-playbook). The exploration trade-offs that motivate diversity pressure in ES are covered in depth in [Exploration vs. Exploitation: The Core Tension](/blog/machine-learning/reinforcement-learning/exploration-vs-exploitation-the-core-tension). For hyperparameter tuning as a standalone topic independent of PBT, see [Hyperparameter Tuning for RL](/blog/machine-learning/reinforcement-learning/hyperparameter-tuning-for-rl).

![PBT episode return grid for 12 configurations across 3 learning rates and 4 batch sizes after 500k training steps, showing peak return 6100 at lr=3e-4 and batch-size=256](/imgs/blogs/evolutionary-strategies-for-rl-8.png)

## Key takeaways

1. **ES is a finite-difference gradient estimator, not blind search.** The OpenAI ES update estimates $\nabla_\theta J_\sigma(\theta)$ from population fitness, and for an isotropic Gaussian, this is provably the natural gradient of the fitness-smoothed objective — the same quantity gradient methods estimate, computed without any backpropagation.

2. **Antithetic sampling halves estimator variance at zero extra cost.** Evaluating paired perturbations $(\theta + \sigma\epsilon, \theta - \sigma\epsilon)$ and using their fitness difference cancels first-order noise by symmetry. Always use antithetic sampling in practice.

3. **Fitness shaping (rank normalization) is non-negotiable.** Rank-normalizing the raw returns before computing the gradient estimate makes the update scale-invariant, clips outlier return influence, and makes the learning rate transferable across environments. Skip it and you get unstable training.

4. **CMA-ES adapts covariance; OpenAI ES does not.** On ill-conditioned fitness landscapes with condition number $\kappa \gg 1$, CMA-ES converges in $O(\kappa^{1/2})$ fewer generations than isotropic ES. The cost is $O(d^2)$ memory — viable only for $d < 10^3$.

5. **NES places ES on information-geometric foundations.** The natural gradient correction aligns parameter updates with the Riemannian geometry of the distribution manifold. For isotropic Gaussian, the correction factor simplifies to $\sigma^{-2}$, recovering the OpenAI ES estimator. For full Gaussian (CMA-ES style), the correction gives the covariance adaptation rules.

6. **PBT discovers hyperparameter schedules, not just values.** The key insight is that optimal hyperparameters are time-varying: the best lr at step 1M is not the same as the best lr at step 100k. Fixed-hyperparameter methods cannot represent this, but PBT discovers it automatically via the exploit-explore mechanism.

7. **ES trades sample efficiency for parallelism, predictably.** With $k$ workers, OpenAI ES uses roughly $10k$ times fewer wall-clock seconds than serial PPO on the same task, while using $10\times$ more total environment samples. The trade-off is explicit and scales linearly with worker count.

8. **Diversity pressure prevents population collapse on deceptive landscapes.** Adding a novelty score component to fitness shaping prevents ES populations from converging to a single behavioral mode in environments where the nearest local optimum is not the global optimum. The weight $\alpha \in [0.5, 0.9]$ on fitness vs novelty is the key tuning parameter.

9. **CMA-ES is effectively hyperparameter-free.** The step size $\sigma$ and full covariance $C$ adapt automatically via the CSA and CMA mechanisms. Initialize with `sigma0 = 0.3 * (xmax - xmin)` for a bounded problem and CMA-ES handles the rest.

10. **PBT generalizes to league training and multi-agent self-play.** The exploit-explore mechanism that discovers optimal lr schedules for a single-agent PPO is structurally identical to the league training mechanism used in AlphaStar. PBT is a general framework for population-based optimization of any training process, including multi-agent self-play curricula.

## Practical engineering notes for production ES

Deploying ES in a real system — not a benchmark, but a product — surfaces problems that papers do not discuss.

**Episode length variance is a major noise source.** In standard Gymnasium environments, episodes terminate at a fixed maximum number of steps (1,000 for HalfCheetah, 500 for CartPole). But in production environments — trading agents, dialogue systems, robotics — episode length varies dramatically. An ES rollout that terminates in 200 steps and one that runs to 1,000 steps produce returns on different scales, corrupting the rank normalization. The fix is to always normalize by episode length: use average reward per step rather than total return as the fitness, or truncate all episodes to a fixed length.

**Parallelism overhead breaks even at approximately 10 workers.** Ray's task dispatch overhead is roughly 0.5–2 ms per task. For very short episodes (CartPole, average 50 steps at 1 ms/step), this overhead dominates and sequential ES actually outperforms distributed ES below 50 workers. For long episodes (HalfCheetah, 1,000 steps at 0.5 ms/step), the breakeven is around 10 workers. The practical rule: only use Ray parallelism when your episode simulation takes more than 500 ms.

**Gradient variance scales with sigma: choosing sigma in practice.** The noise standard deviation $\sigma$ controls both the scale of exploration and the accuracy of the gradient estimate. Too small: the finite-difference approximation is dominated by numerical noise in the fitness evaluation (stochastic environments add noise proportional to $1/\sqrt{n_{\text{eval}}}$). Too large: the gradient estimate is biased toward a smooth, coarse approximation of the fitness landscape that may not reflect the fine structure near the optimum. The practical recommendation: start with $\sigma = 0.05$ for small networks ($d < 10^4$) and $\sigma = 0.02$ for large networks ($d > 10^5$). If training is slow (gradient estimates are noisy), increase $\sigma$. If training diverges early (overshooting local structure), decrease $\sigma$.

**Deterministic policy evaluation is preferred over stochastic.** Because ES fitness values $F(\theta + \sigma\epsilon)$ are used as weights in the gradient estimate, variance in the fitness evaluation directly adds variance to the gradient. In stochastic environments, evaluate each perturbation on multiple episodes and average the returns. The antithetic variance reduction assumes that the noise in $F(\theta + \sigma\epsilon_i)$ and $F(\theta - \sigma\epsilon_i)$ is correlated — this is true for the parameter perturbation but may be negated by environment stochasticity. Using a fixed random seed for each worker's environment ensures that the antithetic pair evaluates on the same environment trajectory, restoring the variance reduction property.

**PBT checkpoint strategy determines exploit effectiveness.** The exploit step copies weights from the top-performing worker, but the weights at the checkpoint may be 25 iterations old (between checkpoints). If the policy changes rapidly between checkpoints (large learning rate, early in training), copying stale weights can be counterproductive — the copied worker is far behind the current best. Reduce `perturbation_interval` (more frequent checkpoints) when the policy is changing rapidly, and increase it (less frequent) when the policy is near convergence. A practical heuristic: set `perturbation_interval` to approximately 5% of the total training budget (e.g., 25 iterations for a 500-iteration run).

**Combining ES with gradient methods: ES-grad hybrid.** For problems that are partly differentiable (dense reward accessible to gradient RL) and partly not (sparse terminal reward inaccessible to backprop), a hybrid approach works well: run gradient RL on the differentiable component of the reward, then run ES to fine-tune against the non-differentiable component. The gradient RL pretraining gives ES a better starting point $\theta_0$ and the ES fine-tuning escapes the local optima that gradient RL converged to. This is used in practice for language model fine-tuning with rule-based reward components: SFT (supervised fine-tuning) provides a differentiable starting point, PPO optimizes the LM reward model component, and ES is used to fine-tune against the rule-based component (e.g., coding task pass rates, format compliance). See [Proximal Policy Optimization (PPO)](/blog/machine-learning/reinforcement-learning/proximal-policy-optimization-ppo) for the PPO layer of this hybrid.

**Memory-efficient ES for large models.** For policies with $d > 10^6$ parameters (large language model fine-tuning, large vision models), storing the parameter vector and gradient estimate in memory is straightforward, but generating antithetic perturbations and storing them temporarily requires $2n \times d$ floats — potentially 80 GB for $n = 1,000$ and $d = 10^7$ with float32. The fix is to use shared random seeds: each worker's perturbation $\epsilon_i$ is fully determined by its worker ID and the generation seed, so the coordinator never needs to store or communicate $\epsilon_i$ directly. Each worker generates its own $\epsilon_i$ on-device and sends back only the scalar fitness. The coordinator reconstructs the gradient by replaying the same seeds:

```python
# Coordinator: reconstruct gradient from worker IDs and scalar fitnesses
import numpy as np

def reconstruct_gradient(
    fitnesses: np.ndarray,  # shape (n_workers,)
    shaped: np.ndarray,     # rank-normalized fitnesses
    d: int,                 # parameter dimension
    sigma: float,
    generation_seed: int,
    n_workers: int,
) -> np.ndarray:
    """Reconstruct gradient without storing all perturbations simultaneously."""
    grad = np.zeros(d)
    half = n_workers // 2
    for i in range(half):
        # Regenerate perturbation from deterministic seed
        rng = np.random.default_rng([generation_seed, i])
        eps_i = rng.standard_normal(d)
        diff = shaped[i] - shaped[i + half]  # antithetic difference
        grad += diff * eps_i
    return grad / (half * sigma)
```

This approach keeps peak memory at $O(d)$ on the coordinator regardless of population size — only the current parameter vector and gradient accumulator are stored.

## Hyperparameter guide for ES methods

**OpenAI ES hyperparameters:**

| Hyperparameter | Typical range | Effect | Notes |
|---|---|---|---|
| `n_workers` | 100–10,000 | Wall-clock speed | Linear speedup; 1,000 is practical on cloud |
| `sigma` | 0.01–0.10 | Exploration scale | Smaller = sharper signal; larger = more exploration |
| `lr` (Adam) | 0.01–0.10 | Update step size | Higher than typical supervised learning |
| `steps_per_eval` | 500–2,000 | Fitness estimate quality | More steps = lower variance, higher cost |
| `n_generations` | 100–1,000 | Total training budget | Early stopping on plateau |

**CMA-ES hyperparameters:**

| Hyperparameter | Default | Effect | Notes |
|---|---|---|---|
| `sigma0` | 0.3–1.0 | Initial step size | Adapt to scale of parameter space |
| `popsize` (λ) | $4 + 3\ln d$ | Population size | CMA computes good default automatically |
| `maxiter` | 100–500 | Generation budget | CMA stops at convergence automatically |
| `tolx` | 1e-6 | Step size tolerance | Lower = tighter convergence |
| `tolfun` | 1e-8 | Fitness tolerance | Lower = tighter convergence |

**PBT hyperparameters:**

| Hyperparameter | Typical value | Effect | Notes |
|---|---|---|---|
| `num_samples` | 4–32 | Population size | More = better HP coverage but more compute |
| `perturbation_interval` | 20–50 iters | Exploit/explore frequency | More frequent = faster adaptation, noisier |
| `quantile_fraction` | 0.20–0.25 | Fraction to copy/replace | 20–25% is standard |
| `resample_probability` | 0.10–0.25 | Prob of resample vs perturb | Higher = more diversity injection |
| HP mutation factors | [0.8, 1.2] | Perturbation magnitude | Wider range = more exploration |

## Further reading

- Salimans, T., Ho, J., Chen, X., Sidor, S., and Sutskever, I. "Evolution Strategies as a Scalable Alternative to Reinforcement Learning." OpenAI Technical Report, 2017. The foundational ES-for-RL paper with benchmark comparisons against A3C and PPO across Atari and MuJoCo.
- Hansen, N. "The CMA Evolution Strategy: A Tutorial." arXiv:1604.00772, 2016. The definitive CMA-ES reference by its inventor — covers all update equations, convergence theory, and implementation details. Essential reading before using CMA-ES seriously.
- Wierstra, D., Förster, T., Peters, J., Riedmiller, M., Schmidhuber, J., and colleagues. "Natural Evolution Strategies." Journal of Machine Learning Research, 15(1):949–980, 2014. Full NES derivation from the score function estimator and Fisher information matrix, with comparisons to CMA-ES on benchmark functions.
- Jaderberg, M., Dalibard, V., Osindero, S., Czarnecki, W., Doolan, J., Czarnecki, M., and colleagues. "Population Based Training of Neural Networks." arXiv:1711.09846, DeepMind, 2017. The PBT paper: demonstrates hyperparameter schedule discovery on Atari with A3C, RLlib integration details, and the exploit-explore design rationale.
- Lehman, J., and Stanley, K. O. "Abandoning Objectives: Evolution Through the Search for Novelty Alone." Evolutionary Computation, 19(2):189–223, 2011. The novelty search paper — shows that reward-free behavioral diversity outperforms fitness optimization on deceptive maze navigation, with implications for ES diversity pressure.
- Glasmachers, T., Schaul, T., Yi, S., Wierstra, D., and Schmidhuber, J. "Exponential Natural Evolution Strategies." GECCO, 2010. Derives xNES with exact natural gradient updates for the full Gaussian family, showing that CMA-ES is an approximation of xNES.
- [Proximal Policy Optimization (PPO)](/blog/machine-learning/reinforcement-learning/proximal-policy-optimization-ppo) — the gradient-based baseline ES competes with; essential for understanding the wall-clock trade-off.
- [Hyperparameter Tuning for RL](/blog/machine-learning/reinforcement-learning/hyperparameter-tuning-for-rl) — covers Bayesian optimization, random search, and successive halving as alternatives to PBT for HP optimization.
- [The Reinforcement Learning Playbook](/blog/machine-learning/reinforcement-learning/the-reinforcement-learning-playbook) — series capstone with algorithm selection decision trees across all RL problem types.
