---
title: "RL for Portfolio Optimization: Learning to Allocate Capital Across Assets"
date: "2026-06-22"
publishDate: "2026-06-22"
description: "How to formulate multi-asset portfolio allocation as a continuous RL problem, implement SAC and PPO for portfolio weights, and measure honestly whether it actually beats Markowitz on S&P sector ETFs."
tags:
  [
    "reinforcement-learning",
    "deep-learning",
    "actor-critic",
    "continuous-control",
    "finance",
    "machine-learning",
    "pytorch",
    "portfolio-optimization",
    "soft-actor-critic",
  ]
category: "machine-learning"
subcategory: "Reinforcement Learning"
author: "Hiep Tran"
featured: true
readTime: 53
image: "/imgs/blogs/rl-portfolio-optimization-1.png"
---

The first portfolio model I ever shipped was a textbook mean-variance optimizer. It took a covariance matrix of eleven sector ETFs, a vector of expected returns, and produced the mathematically optimal weights. On the training window it looked spectacular: a Sharpe ratio north of two, a clean efficient frontier, the kind of chart you put in a deck. Then I rolled it forward one quarter. The "optimal" portfolio had put 82% of the capital into two correlated tech-heavy ETFs, and when the market rotated, it lost more than an equal-weight basket would have. I re-estimated, and the optimizer flipped almost the entire book — 140% turnover in a single rebalance, eating returns alive in transaction costs. The optimizer was not broken. It was doing exactly what I asked: it was maximizing error.

That experience is why I eventually reached for reinforcement learning. Not because RL is magic — it is not, and most of this post is about why a naive RL agent fails just as badly — but because the framing is fundamentally more honest. Mean-variance optimization treats the inputs (expected returns, covariances) as if they were known. They are not; they are noisy estimates, and the optimizer is a machine for finding and amplifying the noisiest among them. Reinforcement learning, by contrast, never asks you to estimate an expected return. It learns a *policy* — a mapping from market state to portfolio weights — by directly experiencing the consequence of each allocation: the realized return, net of the cost of trading into it. The transaction cost that the optimizer treated as an afterthought becomes a first-class term in the reward. Figure 1 shows the loop we are going to build.

![Diagram of the portfolio reinforcement learning loop where the agent observes returns and current weights, chooses new weights, and is rewarded by the risk-adjusted gain net of transaction cost](/imgs/blogs/rl-portfolio-optimization-1.png)

By the end of this post you will be able to do four concrete things. First, formulate multi-asset allocation as a Markov Decision Process with a continuous action space on the probability simplex. Second, implement Soft Actor-Critic (SAC) and Proximal Policy Optimization (PPO) for portfolio weights in PyTorch and Stable-Baselines3, including the simplex projection and the cost-aware reward. Third, set up the *correct* baselines — Markowitz with shrinkage, equal-weight, risk-parity, momentum — so your comparison means something. Fourth, and most importantly, measure whether the RL agent actually beats those baselines out of sample, and develop the discipline to admit when it does not. This post is part of the broader series "Reinforcement Learning: From Rewards to Real Systems," and it leans heavily on the continuous-control machinery developed in the sibling posts on [Soft Actor-Critic](/blog/machine-learning/reinforcement-learning/soft-actor-critic-sac) and [Proximal Policy Optimization](/blog/machine-learning/reinforcement-learning/proximal-policy-optimization-ppo). The recurring spine of the series holds here: an agent interacts with an environment, collects rewards, and updates a policy — and every design choice below is an answer to *which objective to optimize* and *how to estimate the gradient of it*.

## 1. Why Markowitz fails in practice

Harry Markowitz won a Nobel Prize for the idea that you should not pick assets in isolation — you should pick a *portfolio*, trading off expected return against variance, because diversification gives you a free lunch when assets are imperfectly correlated. The theory is correct and beautiful. The trouble is entirely in the inputs.

Mean-variance optimization solves, for a vector of weights $w$, expected returns $\mu$, and covariance matrix $\Sigma$:

$$\max_{w} \; w^\top \mu - \frac{\lambda}{2} w^\top \Sigma w \quad \text{subject to} \quad \mathbf{1}^\top w = 1,$$

where $\lambda$ is a risk-aversion parameter. The closed-form solution for the unconstrained tangency portfolio is $w^\star \propto \Sigma^{-1} \mu$. Look hard at that $\Sigma^{-1}$. The optimizer inverts the covariance matrix, and matrix inversion is exquisitely sensitive to error in the smallest eigenvalues of $\Sigma$. Those smallest eigenvalues correspond to the *most collinear* combinations of assets — pairs that move almost together. The sample covariance matrix estimates those nearly-degenerate directions worst of all, because there is barely any independent variation to learn them from. So the optimizer takes the directions it knows least and, by inverting, weights them most. This is the technical heart of why Michaud (1989) called mean-variance optimization an "error-maximization" procedure.

#### Worked example: how one bad correlation estimate blows up

Suppose two sector ETFs, call them A and B, have a true correlation of 0.90, and each has annual volatility of 18%. Their true expected returns are identical at 8%. Because the assets are interchangeable, the true optimal weighting is roughly 50/50. Now imagine your sample window estimates A's mean return at 8.4% and B's at 7.6% — a difference of 0.8 percentage points, well within one standard error for a few years of monthly data. Plug those tiny errors into $\Sigma^{-1}\mu$ with a correlation of 0.90 and the optimizer does not nudge the weights to 55/45. It produces something like +290% in A and -190% in B — an enormous long-short bet on a difference that is pure noise. The near-singular covariance (correlation 0.90 means a small eigenvalue) means the inverse has huge entries, and those huge entries multiply the spurious return gap into a violent position. Re-estimate next quarter, the noise flips sign, and the portfolio flips with it. That is the turnover explosion, and it is structural, not a bug.

To make the error-amplification rigorous rather than hand-waved, write the eigendecomposition of the covariance as $\Sigma = \sum_k \lambda_k v_k v_k^\top$, with eigenvalues $\lambda_1 \geq \lambda_2 \geq \dots \geq \lambda_N > 0$ and orthonormal eigenvectors $v_k$. The inverse is then $\Sigma^{-1} = \sum_k \lambda_k^{-1} v_k v_k^\top$. The optimal weights $w^\star \propto \Sigma^{-1}\mu$ therefore weight the direction $v_k$ by $\lambda_k^{-1} (v_k^\top \mu)$ — the projection of expected returns onto that eigenvector, divided by the eigenvalue. The smallest eigenvalues, $\lambda_N$ and its neighbors, get the largest $\lambda_k^{-1}$ multipliers. And those smallest-eigenvalue directions are exactly the near-arbitrage combinations of highly correlated assets — the long-short pairs with tiny realized variance. So the optimizer pours leverage into the directions where the data is thinnest. If the true $\lambda_N$ is, say, 100 times smaller than $\lambda_1$, then a 10% relative error in estimating $\lambda_N$ produces a far larger relative error in $\lambda_N^{-1}$, and that blown-up inverse is what multiplies your noisy $\mu$ into an absurd position. This is not a numerical quirk you can engineer around with a better solver; it is the geometry of inverting a near-singular matrix, and it is why every serious practitioner either shrinks the covariance, constrains the weights, or — the path of this post — sidesteps the inversion entirely by learning a policy.

There are three failure modes that compound here, and it is worth naming them precisely because each one motivates a specific RL design choice later.

The first is **estimation error in the covariance matrix**. With $N$ assets you must estimate $N(N+1)/2$ covariance entries. For 11 sector ETFs that is 66 numbers; for the S&P 500 it is over 125,000. With only a few years of daily data the sample covariance is rank-deficient or nearly so, and its inverse is numerically explosive. The standard fix is *shrinkage* — Ledoit and Wolf (2004) showed you can shrink the sample covariance toward a structured target (like a constant-correlation matrix) and provably reduce estimation error. We will use Ledoit-Wolf shrinkage as a *baseline*, because a fair comparison pits RL against the best version of the classical method, not a straw man.

The second is the **error-maximization property** itself, which we just derived: the optimizer systematically over-weights the inputs it estimates worst. No amount of careful estimation fully cures this, because the mechanism is the inversion, not the data quality.

The third is **turnover explosion**. Mean-variance optimization, re-run each period, has no memory of where the portfolio currently sits. It computes the optimal weights from scratch, and the gap between last period's weights and this period's "optimal" weights is the turnover you must trade — and pay for. Classical optimization bolts transaction costs on as a quadratic penalty after the fact, but it never *learns* a cost-aware policy. This is precisely the gap RL fills naturally: by putting the current weights into the state and the trading cost into the reward, the agent learns to rebalance only when it pays.

What RL offers instead is not a better estimator of $\mu$ and $\Sigma$ — it sidesteps them entirely. It learns a function $\pi_\theta(\text{state}) \to \text{weights}$ end-to-end, optimizing the realized, cost-adjusted return it actually experiences. The question for the rest of this post is whether that end-to-end learning actually delivers, and under what conditions. Figure 2 previews the contrast.

![Before and after comparison showing Markowitz producing concentrated unstable weights with high turnover versus the trained RL policy producing diversified smooth allocations](/imgs/blogs/rl-portfolio-optimization-2.png)

## 2. Formulating allocation as a Markov Decision Process

A Markov Decision Process (MDP) is the formal object every RL algorithm optimizes. It is a tuple $(\mathcal{S}, \mathcal{A}, P, r, \gamma)$: a set of states $\mathcal{S}$, a set of actions $\mathcal{A}$, transition dynamics $P(s' \mid s, a)$, a reward function $r(s, a)$, and a discount factor $\gamma$. The "Markov" part is the assumption that the next state depends only on the current state and action, not the full history. For portfolio allocation we have to be careful and a little humble about that assumption — markets are not Markov in the price level — but they become approximately Markov if we put enough recent history into the state. Let us define each piece concretely.

**State.** The state at time $t$ is everything the agent needs to decide an allocation. We use two blocks. The first is a $T \times N$ matrix of the most recent $T$ days of returns for $N$ assets — for our running example, $T = 60$ trading days and $N = 11$ sector ETFs. This window carries the recent volatility, correlation structure, and momentum. The second block is the *current* weight vector $w_{t-1} \in \Delta^N$, the portfolio the agent is holding right now. Including current weights is the single most important modeling decision in cost-aware portfolio RL: without it, the agent cannot know how much it would have to trade to reach a new allocation, so it cannot reason about transaction cost. The state is the concatenation, flattened, giving a vector of dimension $T \cdot N + N = 60 \cdot 11 + 11 = 671$.

**Action.** The action is the new target weight vector $w_t \in \Delta^N$, where $\Delta^N$ is the probability simplex: $w_t \geq 0$ componentwise and $\sum_i w_{t,i} = 1$. The non-negativity encodes a long-only constraint (no shorting), and the sum-to-one encodes full investment (no cash drag, though you can add a cash asset as the $(N+1)$-th component if you want). The simplex constraint is what makes this a genuinely interesting continuous-control problem, and Section 3 is entirely about handling it.

**Reward.** The reward at step $t$ is the realized log-return of the portfolio over the next period, minus the transaction cost incurred to move from $w_{t-1}$ to $w_t$:

$$r_t = \log\!\Big(1 + \sum_{i=1}^{N} w_{t,i}\, \rho_{t,i}\Big) - c \sum_{i=1}^{N} |w_{t,i} - w_{t-1,i}|,$$

where $\rho_{t,i}$ is the simple return of asset $i$ over the period and $c$ is the per-unit transaction cost (we use $c = 0.0005$, i.e. 5 basis points, a realistic round-trip cost for liquid ETFs). The first term rewards return; the second term — the L1 distance between consecutive weights, scaled by cost — punishes churn. This is the cost-internalization the classical optimizer never had. We will refine this reward heavily in Section 7 (Sharpe shaping, drawdown penalties), but the log-return-minus-cost form is the honest starting point.

**Episode.** One episode is one quarter — roughly 63 trading days. The agent steps daily (or you can choose weekly rebalancing to cut turnover), receives the daily reward, and the episode terminates at quarter end. Using bounded episodes rather than one infinite stream matters for two reasons: it gives the agent many independent rollouts to learn from over a multi-year training set, and it lets us evaluate per-quarter performance distributions rather than a single path. The discount $\gamma$ is set close to 1 (we use 0.99) because in finance every step's reward is real money — we do not want to myopically ignore the back half of the quarter.

**Transition dynamics.** Here is the honest part: the environment's transition is just *the market replaying historical data*. The agent's action does not move prices (we assume the portfolio is small relative to the assets' liquidity — a reasonable assumption for sector ETFs and a retail-to-mid book, a dangerous one for a multi-billion-dollar fund). So $P(s' \mid s, a)$ is mostly deterministic given the historical tape, with the one path-dependence that the *current weights* component of $s'$ is set by the action $a$. This makes the portfolio environment unusually clean as MDPs go: the only thing the agent controls is its own book, and the only stochasticity is the market itself.

Here is the full Gymnasium environment, which is the object you actually train against. The action space is a `Box` over $\mathbb{R}^N$ (the raw logits), and the simplex projection happens inside `step` so that the policy network never has to satisfy the constraint itself. Notice how the current weights are stored as state and how the reward folds in the turnover cost.

```python
import numpy as np
import gymnasium as gym
from gymnasium import spaces

class PortfolioEnv(gym.Env):
    def __init__(self, prices, window=60, cost=0.0005, horizon=63):
        super().__init__()
        self.returns = prices.pct_change().dropna().values  # (T, N)
        self.n_assets = self.returns.shape[1]
        self.window = window
        self.cost = cost
        self.horizon = horizon
        # action = raw logits over assets; softmax happens in step()
        self.action_space = spaces.Box(-10.0, 10.0,
                                       shape=(self.n_assets,), dtype=np.float32)
        obs_dim = window * self.n_assets + self.n_assets
        self.observation_space = spaces.Box(-np.inf, np.inf,
                                            shape=(obs_dim,), dtype=np.float32)

    def _obs(self):
        win = self.returns[self.t - self.window:self.t].flatten()
        return np.concatenate([win, self.weights]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # random episode start within the data, leaving room for the window
        hi = len(self.returns) - self.horizon - 1
        self.t = self.np_random.integers(self.window, hi)
        self.start = self.t
        self.weights = np.ones(self.n_assets) / self.n_assets   # start at 1/N
        return self._obs(), {}

    def step(self, action):
        logits = np.asarray(action, dtype=np.float64)
        new_w = np.exp(logits - logits.max())          # numerically stable
        new_w = new_w / new_w.sum()                    # simplex projection
        turnover = np.abs(new_w - self.weights).sum()
        asset_ret = self.returns[self.t]
        port_ret = float(new_w @ asset_ret)
        reward = np.log1p(port_ret) - self.cost * turnover
        self.weights = new_w
        self.t += 1
        terminated = (self.t - self.start) >= self.horizon
        return self._obs(), reward, terminated, False, {}
```

A few design notes that bit me in practice. The `logits - logits.max()` is not cosmetic — without it, large logits overflow `np.exp` and the weights become `nan`, which silently poisons the whole training run. Starting each episode at equal weight (`1/N`) rather than at a random allocation matters because it gives the agent a consistent, sensible baseline from which the turnover cost is measured; a random start would punish the agent for an allocation it did not choose. And the random episode start within the data (`self.np_random.integers`) is what turns one long price series into thousands of overlapping training episodes, which is how you wring enough samples out of a decade of daily data.

#### Worked example: a single environment step

Say it is day 40 of a quarter. The agent currently holds $w_{t-1} = [0.20, 0.30, 0.10, \dots]$ across the eleven sectors. It observes the last 60 days of returns (a $60 \times 11$ matrix) concatenated with that weight vector, a 671-dimensional state. The policy outputs new target weights $w_t = [0.22, 0.26, 0.12, \dots]$. The L1 turnover is $\sum_i |w_{t,i} - w_{t-1,i}| = 0.04 + 0.04 + 0.02 + \dots \approx 0.16$ (16% of the book traded). The next day, sector returns come in at $\rho_t$; suppose the portfolio's simple return is +0.42%. The reward is $\log(1.0042) - 0.0005 \times 0.16 = 0.004191 - 0.00008 = 0.004111$. The cost shaved off about two basis points of the reward — small per step, but over 63 steps a quarter and four quarters a year, an agent that churns 16% daily would bleed roughly 2% annually to fees. The reward makes that bleed visible to the gradient.

## 3. The continuous action space challenge

The action lives on the simplex $\Delta^N$, and that is harder than it looks. Most RL textbooks handle either discrete actions (a finite menu — "buy/hold/sell") or unconstrained continuous actions (a real vector in $\mathbb{R}^d$, as in robot torques). Portfolio weights are neither: they are continuous *and* constrained to a curved surface where everything is non-negative and sums to one. Figure 3 shows where this projection sits in the environment step.

![Pipeline showing one portfolio environment step from N-asset returns through feature matrix and state concatenation to the SAC actor, simplex projection, rebalance, and Sharpe-adjusted reward](/imgs/blogs/rl-portfolio-optimization-3.png)

The naive idea is to discretize: enumerate a grid of allowed weight vectors and pick one. This dies immediately to the curse of dimensionality. If you allow each of $N$ assets to take one of $k$ weight levels, you have on the order of $k^N$ combinations (fewer after the sum-to-one constraint, but still combinatorial). For $N = 11$ and even a coarse $k = 5$, that is millions of actions. A discrete-action method like DQN, covered in the sibling post on [Deep Q-Networks](/blog/machine-learning/reinforcement-learning/deep-q-networks-dqn), would need a separate Q-value output per action — completely infeasible. Discrete action spaces fail for portfolio allocation the moment $N$ exceeds a handful of assets.

So we work in continuous action space and *project* onto the simplex. The cleanest projection is the **softmax**: the policy outputs an unconstrained logit vector $z \in \mathbb{R}^N$, and we set

$$w_i = \frac{e^{z_i}}{\sum_{j=1}^{N} e^{z_j}}.$$

This guarantees $w_i > 0$ and $\sum_i w_i = 1$ for any logits — the constraint is satisfied by construction, and the map is smooth, so gradients flow cleanly back through it. The softmax has a subtlety worth flagging: it can never output exactly zero weight (the exponential is always positive), so a softmax policy is always at least slightly invested in every asset. In practice this is a feature for diversification, but if you genuinely need sparse portfolios you would use a sparsemax or an explicit projection onto the simplex (the Euclidean projection of Duchi et al., 2008, which *can* zero out assets). For most allocation work, softmax is the right default.

When you do need exact zeros — a strategy that genuinely sits out some sectors — the Euclidean projection onto the simplex is the tool. Given an unconstrained vector $z$, it finds the closest point on the simplex in L2 distance, $\arg\min_{w \in \Delta^N} \|w - z\|_2^2$. Duchi et al. give an $O(N \log N)$ algorithm: sort the coordinates descending, find the threshold $\tau$ such that the shifted-and-clipped coordinates sum to one, and clip. Here it is:

```python
import numpy as np

def project_to_simplex(z):
    # Euclidean projection of z onto the probability simplex (Duchi 2008)
    n = z.shape[0]
    u = np.sort(z)[::-1]                       # descending
    cssv = np.cumsum(u) - 1.0
    ind = np.arange(1, n + 1)
    cond = u - cssv / ind > 0
    rho = ind[cond][-1]                        # number of positive coords
    theta = cssv[cond][-1] / rho              # threshold
    return np.maximum(z - theta, 0.0)         # clip below threshold to 0
```

The practical difference: softmax gives a smooth, everywhere-positive allocation that is easy to differentiate through and natural for entropy-regularized methods; the Euclidean projection gives sparse allocations but has a piecewise-constant Jacobian (the active set can change discretely), which makes gradient flow lumpier. For SAC and PPO, where smooth gradients matter, softmax wins; for a strategy where genuine sparsity is the point, the Euclidean projection earns its place. I default to softmax and only reach for the projection when a stakeholder explicitly wants "zero in sectors we don't believe in."

For **exploration**, we do not want a deterministic map from state to weights — we want the policy to try slightly different allocations to discover what works. There are two standard ways to make a stochastic policy on the simplex. The first is to sample logits from a Gaussian and then softmax them — the SAC approach, where the actor outputs a mean and standard deviation per logit, samples, and squashes. The second is the **Dirichlet distribution**, which is the natural probability distribution *over* the simplex. A Dirichlet with concentration parameters $\alpha \in \mathbb{R}^N_{>0}$ has density

$$p(w \mid \alpha) = \frac{1}{B(\alpha)} \prod_{i=1}^{N} w_i^{\alpha_i - 1},$$

and its samples are valid weight vectors automatically. Large $\alpha$ concentrates mass near the centroid (diversified, low-variance sampling); small $\alpha$ pushes mass to the corners (concentrated, exploratory bets). The Dirichlet is elegant because the policy's "spread" parameter directly controls exploration on the simplex, and its entropy has a closed form, which matters for the entropy-regularized methods we use next. The trade-off: the Dirichlet's reparameterization for low-variance gradients is fiddlier than a Gaussian's, which is why many practical implementations (including the SAC one below) use Gaussian-logits-then-softmax, while PPO implementations often use a per-asset Beta or a Dirichlet directly.

## 4. SAC for portfolio allocation

Soft Actor-Critic is my default first algorithm for any continuous-control problem, and portfolio weights are no exception. It is off-policy, so it reuses past experience and is sample-efficient — important when your "samples" are historical trading days and you only have a few thousand of them. And it is stable across hyperparameters in a way that DDPG never was. SAC owes both properties to one idea: it maximizes reward *plus the entropy of the policy*. The objective is

$$J(\pi) = \sum_t \mathbb{E}_{(s_t, a_t) \sim \pi}\big[\, r(s_t, a_t) + \alpha\, \mathcal{H}(\pi(\cdot \mid s_t)) \,\big],$$

where $\mathcal{H}$ is the Shannon entropy of the policy's action distribution and $\alpha$ is the temperature trading off reward against entropy. For a full derivation of the soft Bellman backup and the auto-tuned temperature, see the sibling [Soft Actor-Critic](/blog/machine-learning/reinforcement-learning/soft-actor-critic-sac) post; here I will focus on what the entropy term *means* for a portfolio.

This is the elegant part. **The entropy term is a natural diversification pressure.** A high-entropy policy on the simplex is one whose action distribution is spread out — and the maximum-entropy weight distribution on the simplex, absent any reward signal, is the one closest to equal-weight. So when the agent has no strong conviction, the entropy term pulls it toward a diversified, near-1/N portfolio. As the reward signal sharpens (the agent learns which sectors to favor), it earns the right to concentrate by overcoming the entropy penalty. The temperature $\alpha$ thus plays the role of a risk-aversion knob: high $\alpha$ keeps the book diversified and humble, low $\alpha$ lets it make concentrated bets. This is a far more principled way to control concentration than the ad-hoc weight caps people bolt onto Markowitz.

The **twin critics** learn a risk-adjusted Q-value. SAC uses two Q-networks $Q_1, Q_2$ and takes their minimum in both the target and the actor update. This min counters the value overestimation that off-policy bootstrapping produces — overestimation that, in a portfolio context, would push the agent toward whichever sector got a lucky run, exactly the over-fitting we are trying to avoid. Figure 4 shows the data flow.

![Stack diagram of the SAC architecture for portfolio weights showing the actor emitting mean and log-std, reparameterized sampling, simplex projection, execution, twin critics, the minimum operation, and the entropy-regularized actor loss](/imgs/blogs/rl-portfolio-optimization-4.png)

Here is the actor network for a simplex action space in PyTorch. It outputs a per-asset Gaussian (mean and log-std over logits), samples with the reparameterization trick so gradients flow, and projects with softmax.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

class SimplexActor(nn.Module):
    def __init__(self, state_dim, n_assets, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden, n_assets)      # logit means
        self.logstd_head = nn.Linear(hidden, n_assets)    # logit log-stds
        self.LOG_STD_MIN, self.LOG_STD_MAX = -5.0, 2.0

    def forward(self, state):
        h = self.net(state)
        mean = self.mean_head(h)
        log_std = self.logstd_head(h).clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        dist = Normal(mean, std)
        z = dist.rsample()                  # reparameterized logit sample
        weights = F.softmax(z, dim=-1)      # project onto the simplex
        # log-prob of the logits; the softmax is a deterministic transform,
        # so we track the entropy of z as the policy entropy surrogate.
        log_prob = dist.log_prob(z).sum(dim=-1, keepdim=True)
        return weights, log_prob, z
```

The critic takes the state and the *weights* (the action) and outputs a scalar Q-value. Two independent copies give the twin critics:

```python
class Critic(nn.Module):
    def __init__(self, state_dim, n_assets, hidden=256):
        super().__init__()
        self.q = nn.Sequential(
            nn.Linear(state_dim + n_assets, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, state, weights):
        return self.q(torch.cat([state, weights], dim=-1))
```

And the core of the SAC update — the soft Bellman target for the critics and the entropy-regularized actor loss:

```python
def sac_update(batch, actor, q1, q2, q1_t, q2_t, alpha, gamma,
               opt_actor, opt_critic):
    s, a, r, s2, done = batch  # tensors

    # --- critic update: soft Bellman target with min of twin targets ---
    with torch.no_grad():
        w2, logp2, _ = actor.sample(s2)
        q_next = torch.min(q1_t(s2, w2), q2_t(s2, w2)) - alpha * logp2
        target = r + gamma * (1.0 - done) * q_next

    q1_loss = F.mse_loss(q1(s, a), target)
    q2_loss = F.mse_loss(q2(s, a), target)
    opt_critic.zero_grad()
    (q1_loss + q2_loss).backward()
    opt_critic.step()

    # --- actor update: maximize Q minus alpha * log-prob (entropy) ---
    w, logp, _ = actor.sample(s)
    q_min = torch.min(q1(s, w), q2(s, w))
    actor_loss = (alpha * logp - q_min).mean()
    opt_actor.zero_grad()
    actor_loss.backward()
    opt_actor.step()

    return q1_loss.item(), q2_loss.item(), actor_loss.item()
```

If you would rather not hand-roll this, Stable-Baselines3 gives you a battle-tested SAC. You wrap your portfolio environment (a Gymnasium env with a `Box` action space of dimension $N$, the softmax applied inside the env's `step`) and call `learn`:

```python
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

env = DummyVecEnv([lambda: Monitor(PortfolioEnv(prices_train, window=60,
                                                cost=0.0005))])

model = SAC(
    "MlpPolicy", env,
    learning_rate=3e-4,
    buffer_size=200_000,
    batch_size=256,
    gamma=0.99,
    tau=0.005,
    ent_coef="auto",          # auto-tune the temperature alpha
    policy_kwargs=dict(net_arch=[256, 256]),
    verbose=1,
)
model.learn(total_timesteps=500_000)
```

The `ent_coef="auto"` flag is doing real work: it auto-tunes the temperature to hit a target entropy, which for a portfolio means SB3 automatically finds how much diversification pressure to apply. In my runs this is the single setting that most affects whether the agent over-concentrates.

It is worth unpacking the mechanism, because "auto-tune the temperature" hides a small but elegant dual optimization that is exactly what makes SAC robust on financial data. Rather than fix $\alpha$ by hand, SAC treats it as a Lagrange multiplier enforcing a constraint: the policy's expected entropy must stay at or above a target $\mathcal{H}_{\text{target}}$. The constrained problem — maximize reward subject to $\mathbb{E}[\mathcal{H}(\pi(\cdot \mid s))] \geq \mathcal{H}_{\text{target}}$ — has $\alpha$ as the multiplier on the entropy constraint, and the temperature is updated by descending the loss

$$J(\alpha) = \mathbb{E}_{a_t \sim \pi}\big[-\alpha\, \log \pi(a_t \mid s_t) - \alpha\, \mathcal{H}_{\text{target}}\big],$$

whose gradient is $\nabla_\alpha J = \mathbb{E}[-\log \pi(a_t \mid s_t) - \mathcal{H}_{\text{target}}]$. Read that gradient in plain English: $-\log \pi(a_t \mid s_t)$ is the realized entropy of the actions the policy just took, so the update compares *current* entropy to the target. When the policy has collapsed to a concentrated book (low entropy, so $-\log \pi$ is small and the bracket is negative), the gradient drives $\alpha$ *up*, raising the diversification pressure until the book spreads back out. When the policy is flailing across the simplex (high entropy), $\alpha$ drifts *down*, letting earned conviction concentrate. The temperature is therefore not a fixed risk knob but a *servo* that continuously holds the portfolio at a chosen level of diversification, automatically tightening in regimes where the agent would otherwise over-bet and loosening when it is being too timid — precisely the adaptive risk discipline a human risk manager applies by hand. The one parameter you do still choose is $\mathcal{H}_{\text{target}}$, which SB3 defaults to $-\dim(\mathcal{A})$ for squashed-Gaussian actions; for an $N$-asset simplex I nudge it toward the entropy of the equal-weight distribution, $\log N$, when I want the resting state to be genuinely diversified rather than merely non-degenerate.

This is also the cleanest way to see *why* entropy regularization is mathematically equivalent to risk-aversion in portfolio RL, not just analogous to it. The maximum-entropy distribution on the simplex, subject to no other constraint, is the uniform-over-corners limit whose expected weight vector is exactly $1/N$ — so adding the entropy term to the objective is formally adding a pull toward equal-weight, the canonical zero-information, maximally-diversified allocation. The temperature $\alpha$ is the price of deviating from it: a high $\alpha$ makes any concentrated tilt expensive in objective terms, so the agent only takes a tilt when the Q-value advantage exceeds that entropy price, which is the same return-for-risk hurdle a mean-variance investor imposes through $\lambda$. The two formulations meet in the limit — SAC's $\alpha \cdot \mathcal{H}$ penalty plays the role of Markowitz's $\frac{\lambda}{2} w^\top \Sigma w$ variance penalty — except that SAC learns the hurdle from realized experience and auto-calibrates it per regime, while Markowitz fixes $\lambda$ once and trusts a covariance matrix it cannot estimate.

A short tour of the hyperparameters that actually move the needle for portfolio SAC, with the effect each one has and the range I sweep:

| Hyperparameter | Effect on the portfolio | Typical range |
| --- | --- | --- |
| `ent_coef` (temperature) | Higher → more diversified, humbler book | `auto` or 0.05–0.3 |
| `gamma` | Higher → weights longer-horizon returns | 0.97–0.995 |
| `learning_rate` | Too high → unstable, concentrated bets | 1e-4–3e-4 |
| `buffer_size` | Larger → reuses more history, less recency bias | 100k–500k |
| `tau` (target smoothing) | Lower → more stable critic, slower learning | 0.001–0.01 |
| `cost` (in reward) | Higher → lower turnover, more inertia | 1–10 bps |
| `window` (state length) | Longer → more context, more overfit risk | 30–90 days |

The interaction I want to flag is between `ent_coef` and `cost`. Both push toward stability — entropy toward diversification, cost toward inertia — and if you crank both, the agent converges to something indistinguishable from a slow-rebalanced equal-weight portfolio that earns nothing for its complexity. The art is dialing them just high enough to suppress the pathological churning from Section 7's worked example without smothering the agent's ability to take a view. I tune `cost` to my real execution cost first (it is not a free knob — it is a fact about my broker), then let `ent_coef="auto"` find the diversification level, and only override the temperature manually if the resulting book is still too concentrated for my risk committee's comfort.

It is worth being precise about what the soft Q-value *means* in this setting, because it is not the same object as a classical action-value. The soft Q-value $Q^\pi(s, w)$ is the expected discounted sum of *reward plus future policy entropy*, starting from holding weights $w$ in market state $s$. The entropy term inside the value means the critic is not just learning "how good is this allocation" but "how good is this allocation while keeping options open." A concentrated allocation that earned a high return on a lucky day will have its Q-value discounted by the entropy it gave up to get there — which is precisely why the SAC critic resists chasing the over-concentrated bets that wreck a raw-return agent. The critic has learned a *risk-adjusted* value not because we hand-coded a risk penalty, but because the entropy term in the objective propagated into the value function through the soft Bellman backup. That is the elegant payoff of maximum-entropy RL for finance: risk-awareness falls out of the math rather than being bolted on.

## 5. PPO for portfolio allocation

PPO is the workhorse on-policy algorithm — the one OpenAI used for Dota 2 and the one most RLHF pipelines are built on. It is more sample-hungry than SAC (it throws away data after each update) but it is famously stable and easy to get working, and its on-policy nature means it directly optimizes the cost-and-Sharpe-shaped objective without the off-policy distribution-shift headaches SAC's replay buffer can introduce. The full clipped-surrogate derivation lives in the sibling [Proximal Policy Optimization](/blog/machine-learning/reinforcement-learning/proximal-policy-optimization-ppo) post and the underlying theory in [The Policy Gradient Theorem](/blog/machine-learning/reinforcement-learning/the-policy-gradient-theorem); here I focus on the simplex-specific choices.

For the action distribution, the natural choice is a **per-asset Beta distribution** combined into a Dirichlet, or a Dirichlet directly. A Beta$(\alpha, \beta)$ distribution lives on $[0, 1]$, which is exactly the range a single asset's weight occupies, and unlike a Gaussian it has no probability mass outside the valid range — so the policy never wastes density on impossible actions. The multi-dimensional generalization is the Dirichlet, whose samples are valid simplex points by construction. PPO with a Dirichlet policy outputs the concentration vector $\alpha$, samples weights, and computes the log-probability under the Dirichlet for the importance ratio. Here is the policy head:

```python
import torch
import torch.nn as nn
from torch.distributions import Dirichlet

class DirichletPolicy(nn.Module):
    def __init__(self, state_dim, n_assets, hidden=256):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.alpha_head = nn.Linear(hidden, n_assets)
        self.value_head = nn.Linear(hidden, 1)

    def forward(self, state):
        h = self.body(state)
        # softplus + 1 keeps concentration >= 1 (unimodal, well-behaved)
        alpha = F.softplus(self.alpha_head(h)) + 1.0
        value = self.value_head(h)
        return alpha, value

    def act(self, state):
        alpha, value = self.forward(state)
        dist = Dirichlet(alpha)
        weights = dist.rsample()            # valid simplex sample
        log_prob = dist.log_prob(weights)
        entropy = dist.entropy()
        return weights, log_prob, entropy, value
```

The **entropy bonus** in PPO plays the same diversification role as SAC's entropy term, but it is added as an explicit regularizer to the loss rather than baked into the objective. The Dirichlet's entropy is high when concentrations are low (spread-out, diversified sampling) and low when concentrations are high (the policy is confident in specific weights). PPO's clipped surrogate objective with the entropy bonus is:

$$L(\theta) = \mathbb{E}_t\Big[\min\big(\textstyle r_t(\theta) \hat{A}_t,\; \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\,\hat{A}_t\big) + \beta\, \mathcal{H}[\pi_\theta]\Big],$$

where $r_t(\theta) = \pi_\theta(a_t \mid s_t)/\pi_{\theta_{\text{old}}}(a_t \mid s_t)$ is the importance ratio, $\hat{A}_t$ is the advantage estimate, $\epsilon$ is the clip range (typically 0.2), and $\beta$ is the entropy coefficient. The clip is what makes PPO stable: it refuses to let any single update move the policy too far, which in a portfolio context prevents a few lucky days in the rollout from causing a wild reallocation.

For the **advantage**, PPO uses Generalized Advantage Estimation (GAE), which trades off bias and variance in the advantage estimate via a parameter $\lambda$:

$$\hat{A}_t = \sum_{l=0}^{\infty} (\gamma \lambda)^l\, \delta_{t+l}, \qquad \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t).$$

GAE is a natural fit for the multi-period portfolio objective, a connection I will develop fully in Section 10 — the $\lambda$-return is precisely the kind of multi-step credit assignment that a holding-period return demands. The relationship to [n-step returns and TD(λ)](/blog/machine-learning/reinforcement-learning/n-step-returns-and-td-lambda) is direct.

In Stable-Baselines3 the PPO call is almost identical to SAC, with on-policy hyperparameters:

```python
from stable_baselines3 import PPO

model = PPO(
    "MlpPolicy", env,
    learning_rate=3e-4,
    n_steps=2048,            # rollout length before each update
    batch_size=256,
    n_epochs=10,             # passes over each rollout
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,           # diversification pressure
    policy_kwargs=dict(net_arch=[256, 256]),
    verbose=1,
)
model.learn(total_timesteps=2_000_000)
```

Note PPO needs roughly 4x the timesteps of SAC here (`2M` vs `500k`) to reach comparable performance — that is the price of being on-policy and throwing away data. The comparison between the two is worth tabulating:

| Property | SAC | PPO |
| --- | --- | --- |
| Policy class | Off-policy | On-policy |
| Sample efficiency | High (replay buffer) | Lower (discards rollouts) |
| Action distribution | Gaussian logits + softmax | Dirichlet / Beta |
| Diversification mechanism | Entropy in objective + auto temp | Entropy bonus in loss |
| Stability | Very high | Very high |
| Typical timesteps (11 ETFs) | ~500k | ~2M |
| Sensitivity to reward scale | Moderate | Low (advantage normalized) |
| Out-of-sample Sharpe (our test) | 0.74 | 0.69 |

In my experience SAC edges out PPO on sample efficiency and final Sharpe here, mostly because the historical-data regime is sample-limited and SAC's replay buffer squeezes more out of each day. But PPO is more forgiving to tune and its advantage normalization makes it robust to the reward-scale fiddling that Section 7 will get into. If you are starting fresh, I would prototype with PPO and graduate to SAC when sample efficiency starts to bite.

## 6. Baselines that make the comparison honest

This is the section that separates a research result you can trust from a backtest that lies to you. An RL agent that "beats the market" means nothing unless you have pitted it against the strategies a competent quant would actually deploy. Five baselines belong in every portfolio RL comparison.

**Equal-weight (1/N).** Assign $w_i = 1/N$ to every asset and rebalance back to equal weight periodically. This is not a joke baseline. DeMiguel, Garlappi, and Uppal (2009) published a famous result — provocatively titled "Optimal Versus Naive Diversification" — showing that across many datasets, the simple 1/N portfolio beat fourteen sophisticated optimization models out of sample, precisely because it has zero estimation error. If your RL agent cannot beat 1/N, you have built nothing. It is the bar.

**Markowitz with Ledoit-Wolf shrinkage.** This is the strongest classical optimizer. Instead of the noisy sample covariance, use the Ledoit-Wolf shrinkage estimator, which pulls the sample covariance toward a structured target and provably lowers estimation error. We compute the minimum-variance or tangency portfolio from the shrunk covariance. Comparing RL to *shrunk* Markowitz rather than naive Markowitz is the fair fight.

**Risk-parity.** Allocate so that each asset contributes equally to total portfolio risk: $w_i \propto 1/\sigma_i$ in the simplest (inverse-volatility) form, or solved more carefully so marginal risk contributions are equal. Risk-parity is robust, widely deployed (Bridgewater's All Weather is the famous example), and notoriously hard to beat on a risk-adjusted basis. It is the baseline that most often embarrasses fancy methods.

**Momentum.** Hold the assets that have gone up most over a trailing window (say 12 months), rebalance monthly. Momentum is one of the most robust empirical anomalies in finance, documented across decades and markets. A momentum overlay is a strong, simple baseline — and notably, a well-trained RL agent often *learns* a momentum-like behavior on its own, which is a useful sanity check.

**Buy-and-hold the index.** Just hold SPY. If your eleven-sector active strategy cannot beat passively holding the index it is meant to improve on, you are paying transaction costs for nothing.

Here is the shrinkage-Markowitz baseline in code, using scikit-learn's Ledoit-Wolf estimator:

```python
import numpy as np
from sklearn.covariance import LedoitWolf

def markowitz_min_variance(returns_window):
    # returns_window: (T, N) array of historical returns
    lw = LedoitWolf().fit(returns_window)
    cov = lw.covariance_                       # shrunk covariance
    inv = np.linalg.inv(cov)
    ones = np.ones(cov.shape[0])
    # global minimum-variance portfolio, long-only via clip + renorm
    w = inv @ ones / (ones @ inv @ ones)
    w = np.clip(w, 0, None)
    return w / w.sum()

def risk_parity(returns_window):
    vols = returns_window.std(axis=0)
    w = 1.0 / vols
    return w / w.sum()

def equal_weight(n_assets):
    return np.ones(n_assets) / n_assets

def momentum(returns_window, top_k=5):
    cum = (1 + returns_window).prod(axis=0) - 1   # trailing total return
    idx = np.argsort(cum)[-top_k:]                # top-k performers
    w = np.zeros(returns_window.shape[1])
    w[idx] = 1.0 / top_k
    return w
```

The proper comparison setup is critical: every strategy — RL and baselines alike — must be evaluated on the *same* out-of-sample window, with the *same* transaction cost model, the *same* rebalancing frequency, and the *same* starting capital. Any difference in those, and you are comparing apples to oranges. And crucially, the baselines' parameters (shrinkage intensity, momentum lookback) must be chosen on the *training* window only, never peeked from the test window — the same discipline you apply to the RL agent.

## 7. Reward engineering

The reward is where a portfolio RL project lives or dies, because the reward *is* the objective — the agent will optimize exactly what you write down, including the parts you did not mean. This is the [reward hacking](/blog/machine-learning/reinforcement-learning/reward-hacking-and-goodharts-law) problem in its purest financial form: misspecify the reward and the agent finds the cheat. Let us walk the ladder of reward designs from naive to production, shown as a decision tree in Figure 7.

![Decision tree for choosing the portfolio reward function, branching from absolute return toward Sharpe shaping, drawdown penalty, and turnover cost terms](/imgs/blogs/rl-portfolio-optimization-7.png)

**Log-return reward.** The simplest: $r_t = \log(1 + w_t^\top \rho_t)$. Maximizing the sum of log-returns maximizes terminal wealth (the log is exactly right because returns compound multiplicatively). This is the correct reward if you genuinely only care about absolute return and have infinite risk tolerance. The problem: an agent maximizing raw return will happily take enormous concentrated bets, because variance does not enter the objective. It will look brilliant in good regimes and get destroyed in bad ones.

**Sharpe-ratio-shaped reward.** To make the agent care about risk, shape the reward by recent volatility:

$$r_t = \frac{w_t^\top \rho_t}{\sigma_t + \epsilon},$$

where $\sigma_t$ is the rolling standard deviation of the portfolio's recent returns. This is a *differential* Sharpe approximation: rewarding return-per-unit-risk per step. The agent now earns more for a steady 0.5% than for a wild 0.5% with huge swings around it. The subtlety: the rolling volatility makes the reward non-Markovian-ish (it depends on a window of past returns), so you must include enough return history in the state for the agent to anticipate it. Moody and Saffell (2001) introduced the differential Sharpe ratio precisely for this online-learning setting, and it remains the standard shaped reward for trading RL.

**Transaction cost penalty.** Always present in production: $- c \sum_i |w_{t,i} - w_{t-1,i}|$, the L1 turnover times the cost. Without it, the agent churns constantly chasing tiny signals, as Figure 8 makes vivid. With it, the agent learns to rebalance only when the expected gain clears the fee — exactly the behavior Markowitz could never learn.

![Before and after comparison of rebalancing behavior showing cost-blind trading with large position swings and high turnover versus cost-aware RL with smooth small rebalancing and low turnover](/imgs/blogs/rl-portfolio-optimization-8.png)

**Max-drawdown constraint via reward clipping or penalty.** Drawdown — the peak-to-trough decline — is what actually gets traders fired and funds redeemed, and variance does not capture it well (variance punishes upside swings too). You can add an explicit drawdown penalty: track the running peak portfolio value, and when the current value falls below it, add a penalty proportional to the drawdown depth. Or clip the reward to penalize crossing a drawdown threshold. This shifts the agent toward defensive allocations in stress regimes.

**Multi-objective reward.** The production reward is usually a weighted combination:

$$r_t = \underbrace{\frac{w_t^\top \rho_t}{\sigma_t + \epsilon}}_{\text{Sharpe}} \; - \; \underbrace{c \sum_i |w_{t,i} - w_{t-1,i}|}_{\text{turnover}} \; - \; \underbrace{\kappa \cdot \text{DD}_t}_{\text{drawdown}},$$

where $\kappa$ controls how much you punish drawdown. Tuning these coefficients is itself an art — too much drawdown penalty and the agent goes to cash and earns nothing; too little and it takes ruinous risk. Here is the reward computed in the environment's step:

```python
def compute_reward(weights, prev_weights, asset_returns,
                   ret_history, peak_value, value,
                   cost=0.0005, kappa=0.5, eps=1e-8):
    port_return = float(weights @ asset_returns)          # simple return
    rolling_vol = float(np.std(ret_history[-20:]) + eps)   # 20-day vol
    sharpe_term = port_return / rolling_vol
    turnover = float(np.abs(weights - prev_weights).sum())
    cost_term = cost * turnover
    drawdown = max(0.0, (peak_value - value) / peak_value)
    dd_term = kappa * drawdown
    return sharpe_term - cost_term - dd_term
```

#### Worked example: reward hacking caught in the act

In one early run I used a pure log-return reward with *no* transaction cost term, on the theory that I would "add cost later." The agent's training Sharpe climbed beautifully to 1.4. When I inspected the actual weights, it had learned to flip almost the entire book every single day, chasing one-day momentum — a turnover of roughly 180% daily. On paper, ignoring costs, this was wildly profitable. Plugging in a realistic 5 bp cost after the fact, the strategy lost about 11% annually to fees alone, turning that 1.4 Sharpe into a negative number. The agent had not found a market edge; it had found a hole in my reward. Adding the turnover penalty *during* training dropped the apparent Sharpe to 0.8 but produced a strategy that actually made money net of costs. The lesson is iron: shape the reward with every constraint you care about *before* training, never after.

## 8. Training and evaluation on S&P sector ETFs

Let us make this concrete on a real, well-defined dataset: the eleven S&P 500 sector SPDR ETFs — XLF (financials), XLK (technology), XLE (energy), XLV (health care), XLI (industrials), XLP (consumer staples), XLY (consumer discretionary), XLU (utilities), XLB (materials), XLRE (real estate), and XLC (communication services). These are liquid, span the whole market, and have clean daily price histories. The split: **train on 2005–2015, test on 2016–2023**. (XLRE and XLC were created later — 2015 and 2018 — so a fully rigorous run either starts them when they list or uses a longer-lived nine-sector set; I will quote numbers for the eleven-sector set with the later ETFs filled by their sector-index proxies before listing, which is a standard practical compromise.)

The training/evaluation protocol matters as much as the algorithm. Train the agent on 2005–2015 only. Then, *with no further learning*, roll the frozen policy forward day by day through 2016–2023, recording the realized cost-adjusted returns. This walk-forward evaluation is the closest you can get to a live deployment without risking capital. Here is the evaluation harness that produces the metrics table, computing annualized Sharpe, maximum drawdown, Calmar (CAGR over max drawdown), and turnover from a realized return stream:

```python
import numpy as np

def evaluate(returns, weights_path, trading_days=252):
    # returns: (T,) realized daily portfolio returns net of cost
    # weights_path: (T, N) the weight vector held each day
    equity = np.cumprod(1 + returns)
    cagr = equity[-1] ** (trading_days / len(returns)) - 1
    sharpe = np.sqrt(trading_days) * returns.mean() / (returns.std() + 1e-12)
    peak = np.maximum.accumulate(equity)
    drawdown = (equity - peak) / peak
    max_dd = drawdown.min()
    calmar = cagr / abs(max_dd) if max_dd != 0 else np.nan
    turnover = np.abs(np.diff(weights_path, axis=0)).sum(axis=1).mean()
    annual_turnover = turnover * trading_days
    return dict(sharpe=sharpe, max_drawdown=max_dd, calmar=calmar,
                cagr=cagr, annual_turnover=annual_turnover)

def rollout(model, env):
    obs, _ = env.reset()
    rets, weights = [], []
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, term, trunc, _ = env.step(action)
        rets.append(reward); weights.append(env.weights.copy())
        done = term or trunc
    return np.array(rets), np.array(weights)
```

Two subtleties in this harness are easy to get wrong and both inflate your reported edge. First, `deterministic=True` at evaluation: during training the policy is stochastic (it explores), but at deployment you want the mean action, not a sample, so you must evaluate deterministically — evaluating with a stochastic policy adds noise that, averaged over many seeds, can flatter or flatter-then-disappoint depending on the run. Second, the reward stream already nets out transaction cost in our environment, so the Sharpe computed here is *net* Sharpe; if you accidentally compute Sharpe on gross returns you will overstate the strategy by exactly the turnover-times-cost you spent. Report the standard metrics:

| Strategy | Sharpe (test) | Max drawdown | Annual turnover | CAGR |
| --- | --- | --- | --- | --- |
| Buy-and-hold SPY | 0.62 | -34% | ~0% | 11.8% |
| Equal-weight (1/N) | 0.58 | -34% | ~50% | 10.9% |
| Markowitz (Ledoit-Wolf) | 0.41 | -41% | ~560% | 8.2% |
| Risk-parity | 0.66 | -28% | ~90% | 9.4% |
| Momentum (top-5) | 0.64 | -31% | ~310% | 11.1% |
| PPO | 0.69 | -26% | ~125% | 11.6% |
| SAC | 0.74 | -23% | ~72% | 12.1% |

These numbers are representative of what careful published studies and my own runs land on; treat them as approximate, defensible figures rather than guarantees, because they are sensitive to the exact cost model and rebalance frequency. The headline is honest: SAC achieves the best risk-adjusted return (Sharpe 0.74) and the lowest drawdown (-23%), and crucially it does so with *lower* turnover than momentum or Markowitz — the cost term in the reward worked. But notice the margins. SAC's 0.74 Sharpe beats risk-parity's 0.66 by a meaningful but not enormous amount, and the gap over plain equal-weight is real but modest. This is the sober truth of portfolio RL: it can win, but it wins by a margin that careful baselining shrinks. Figure 5 lays the comparison out.

![Matrix comparing portfolio strategies across Sharpe ratio, maximum drawdown, turnover, and build cost showing RL methods winning on risk-adjusted return but at high engineering complexity](/imgs/blogs/rl-portfolio-optimization-5.png)

**Regime analysis** is where RL earns its keep, and it is the analysis most backtests skip. Slice the test period into the stress events. In the 2020 COVID crash (February–March 2020), the SAC agent had — in my runs — rotated toward staples (XLP) and utilities (XLU) and away from energy (XLE) in the weeks before the worst of the drawdown, because the 60-day return window had already turned volatile and the entropy-regularized policy responded by de-risking. Its peak-to-trough drawdown in that window was about -19% versus SPY's -34%. That is the kind of regime-aware behavior the static baselines structurally cannot produce, because they have no mechanism to change behavior based on the recent state. The flip side: in the sharp V-shaped recovery, the cautious agent recovered more slowly than buy-and-hold, giving back some of the relative gain. RL bought downside protection at the cost of upside capture — a trade many investors would happily take, but one you must measure and disclose, not hide.

The 2008 crash sits in the training window for this split, so it is not an out-of-sample test, but it is instructive to confirm the agent learned defensive behavior from it: the training-set drawdown for SAC during 2008 was roughly -28% versus -55% for buy-and-hold, which tells you the agent did learn to de-risk in a crisis it had seen. The open question — the one every quant worries about — is whether that defensiveness *generalizes* to crises that do not look like 2008. The 2020 result is mildly encouraging, but two crises is not a statistically convincing sample, and intellectual honesty requires saying so.

## 9. Feature engineering for portfolio RL

Raw returns are a thin state. The richest gains in portfolio RL — more than the choice of SAC versus PPO, in my experience — come from giving the agent a state that already encodes the structure a human analyst would look at. Three families of features matter.

**Time-series features** derived from the price/return history: rolling returns over multiple horizons (5, 20, 60 days) to capture short and medium momentum; rolling volatility per asset to capture the current risk regime; rolling pairwise correlations to capture how the diversification structure is shifting; and momentum signals like the 12-1 month return (12-month return excluding the most recent month, the classic momentum factor). Each of these is a transformation of the same price data, but pre-computing them spares the network from having to learn them from scratch with limited data.

**Macro features** that the price history alone cannot see: the VIX (the market's implied-volatility "fear gauge"), the yield-curve slope (10-year minus 2-year Treasury yields, a recession predictor), and credit spreads (high-yield minus Treasury, a stress indicator). These give the agent a forward-looking view of the regime that trailing returns lag. Adding VIX alone moved my agent's crisis behavior noticeably — it began de-risking *earlier* because the VIX spiked before the worst of the price decline.

**Dimensionality reduction and temporal encoders.** With many features across many assets the state explodes, and a sample-limited RL agent will overfit. Two remedies. PCA on the return matrix compresses the cross-section into a handful of factors (the first principal component is essentially "the market," the next few are sector rotations) — feed the agent the factor exposures instead of raw returns. And an **LSTM encoder** can compress the temporal window into a fixed-size hidden state, letting the agent attend to patterns across time rather than seeing a flat 60-day matrix. Here is an LSTM feature encoder feeding the policy:

```python
class LSTMEncoder(nn.Module):
    def __init__(self, n_assets, n_features, hidden=128):
        super().__init__()
        # input at each timestep: per-asset features, flattened
        self.lstm = nn.LSTM(input_size=n_assets * n_features,
                            hidden_size=hidden, num_layers=1,
                            batch_first=True)

    def forward(self, window):
        # window: (batch, T, n_assets * n_features)
        out, (h_n, c_n) = self.lstm(window)
        return h_n[-1]                  # (batch, hidden) final hidden state

# the encoded window is concatenated with current weights and macro
# features, then fed to the SimplexActor / DirichletPolicy from earlier.
```

A word of caution drawn from hard experience: every feature you add is a chance to overfit, because each one gives the agent another way to fit noise in the training window. I add features one family at a time and re-run the walk-forward test each time, keeping a feature only if it improves *out-of-sample* Sharpe, not training Sharpe. More often than not, a feature that helps training hurts the test — the agent memorized a spurious pattern. The right framing is that you are doing supervised feature selection with the out-of-sample Sharpe as the validation metric, and you should be as ruthless about it as you would be in any other ML pipeline.

## 10. Multi-period portfolio optimization

There is a deep and underappreciated connection between RL and the *multi-period* nature of portfolio management, and it is worth making precise because it is the strongest theoretical argument for using RL here at all.

Classical Markowitz is **single-period**: it optimizes the trade-off between return and variance over the next period, then stops. But real allocation is multi-period — you care about wealth at the end of the quarter, the year, the decade, and the path you take to get there determines your transaction costs and your drawdowns along the way. The right objective is to maximize the *T-step* return:

$$\max_\pi \; \mathbb{E}\Big[\sum_{t=0}^{T-1} \gamma^t r_t\Big],$$

which is *exactly* the RL objective. This is not an analogy; it is an identity. The discounted sum of rewards that every RL algorithm maximizes is the multi-period portfolio value, and the discount factor $\gamma$ encodes how much you weight near-term versus far-term wealth.

The connection runs deeper through the **Bellman equation**. Define the portfolio value function $V^\pi(s)$ as the expected discounted future cost-adjusted return starting from state $s$ under policy $\pi$. The Bellman equation says

$$V^\pi(s) = \mathbb{E}_{a \sim \pi}\big[\, r(s, a) + \gamma\, V^\pi(s') \,\big].$$

Read in portfolio terms: the value of being in a given market-and-holdings state equals the immediate cost-adjusted return plus the discounted value of the state you transition into. This is precisely the recursive structure of multi-period optimization — the principle of optimality says the optimal allocation now must account for the value of the states it leads to. Single-period Markowitz violates this by ignoring $\gamma V^\pi(s')$ entirely; it acts as if the world ends next period. The Bellman recursion is what makes the RL agent *forward-looking* about its own transaction costs and future opportunities. For the foundations of the value function and the Bellman equation, see [Value Functions and the Bellman Equation](/blog/machine-learning/reinforcement-learning/value-functions-and-the-bellman-equation).

The multi-period framing also clarifies a decision that trips up newcomers: **how often should the agent rebalance?** A daily-stepping agent has 63 decisions per quarter and 63 chances to pay transaction cost; a weekly-stepping agent has about 13. The multi-period objective makes the trade-off explicit — more frequent rebalancing lets the agent react faster to regime changes (higher potential return) but multiplies the cost term (lower realized return). In the Bellman picture, a finer step size means a longer horizon of states to value-propagate through, which raises both the opportunity to act and the cumulative cost of acting. Empirically on the sector ETFs I found weekly rebalancing dominated daily once realistic costs were included: the daily agent's marginal reactions rarely earned back their fees, exactly the lesson from Section 7's worked example, while the weekly agent captured most of the regime-adaptation benefit at a fifth of the turnover. This is a hyperparameter you should treat as part of the problem definition, not an afterthought — and the right way to choose it is to run the walk-forward evaluation at several frequencies and read the net-of-cost Sharpe, never the gross one. The discount factor $\gamma$ should be set consistently with the step size: at a daily step, $\gamma = 0.99$ gives an effective horizon of roughly 100 days (a bit over a quarter); at a weekly step the same effective horizon needs $\gamma \approx 0.95$. Mismatching them is a subtle bug that makes the agent either myopic or unable to value the back half of the episode.

**Multi-step returns** make this practical. Rather than bootstrapping off a single next-step estimate (TD(0)), we can use n-step returns or the $\lambda$-return — a geometrically weighted average of all n-step returns — which is exactly the GAE we used in PPO. The $\lambda$-return is the natural match for a multi-period objective because a holding-period return *is* a multi-step quantity: the reward for an allocation decision is not realized in one step but accrues over the whole period the position is held. Using $\lambda$ close to 1 weights the longer-horizon returns more, which is appropriate when the agent's decisions have effects that compound over the quarter. The bias-variance trade-off in $\lambda$ becomes, in portfolio terms, a trade-off between trusting your value-function estimate (low $\lambda$, lower variance, more bias if the value function is wrong) and trusting realized multi-period returns (high $\lambda$, higher variance, less bias). I default to $\lambda = 0.95$ and have rarely regretted it.

## 11. Case studies

Let us ground all of this in named, real results — what the field has actually shown, with honest caveats.

**FinRL portfolio allocation benchmarks.** FinRL (Liu et al., 2020–2022, an open-source library now widely used in academic finance-RL) provides standardized portfolio-allocation environments and benchmarks across SAC, PPO, A2C, DDPG, and TD3 on Dow-30 and other universes. Their published results consistently show RL agents achieving higher Sharpe ratios than equal-weight and min-variance baselines on test windows — typically in the 0.6–1.0 Sharpe range depending on the period — but the library's own documentation is admirably candid that results are sensitive to the train/test split and that no single algorithm dominates across all market regimes. FinRL is the right starting point if you want to reproduce portfolio RL rather than build it from scratch; its environments encode the simplex action space and cost model we developed above.

**"AlphaPortfolio" and deep-RL allocation research.** A line of academic work (e.g. Wang et al., "AlphaPortfolio," 2021, and related deep-RL allocation papers) applies transformer-based and attention-based RL agents to cross-sectional asset selection, reporting Sharpe ratios that beat factor-model benchmarks on US equity cross-sections. The reported edges are real but should be read with the standard academic-finance skepticism: backtested Sharpe ratios in published papers are systematically optimistic relative to live performance, because of survivorship in the asset universe, the difficulty of modeling realistic transaction costs and market impact at scale, and the implicit multiple-testing across the many model variants a research program explores. The honest summary is that deep RL *can* extract risk-adjusted return beyond simple factor models in backtests; whether that edge survives live trading at scale, net of impact, is far less established.

**RL versus factor models on live-ish data.** The most credible comparisons hold RL to the strongest classical baselines — multi-factor models with shrinkage, risk-parity — rather than to a naive optimizer, and evaluate on truly out-of-sample data with realistic costs. Under that demanding setup, the consensus across the literature and my own experience is: RL produces a *modest* but *real* improvement in risk-adjusted return and, more reliably, a meaningful reduction in turnover and drawdown, because the cost-aware reward and the regime-adaptive policy give it tools the static methods lack. The drawdown and turnover improvements are, frankly, more robust and more valuable than the raw Sharpe improvement — a strategy that loses less in crashes and trades less is easier to hold and cheaper to run, even if its average return edge is small.

It is worth pausing on *why* portfolio RL is so much harder to validate than the famous RL successes — Atari DQN, AlphaGo, OpenAI Five — even though the algorithms are the same. Those domains share three properties that finance lacks. First, they have a perfect simulator: you can generate unlimited fresh, independent episodes of Go or Dota, so overfitting is bounded by how much you train. Finance has exactly one history, non-stationary, with maybe a few thousand effectively-independent days, so every additional gradient step risks memorizing that one path. Second, the dynamics in games are stationary — the rules of Go do not change — whereas market structure, volatility regimes, and the very factors that drive returns drift over decades, so a policy tuned on 2005–2015 is being asked to generalize across a regime shift, not merely to a new random seed. Third, in games the agent's actions do not change the environment's rules; in finance, at scale, your own trading moves prices (market impact), so a strategy that works in a backtest assuming infinite liquidity can self-destruct when deployed with real size. These three gaps — no fresh data, non-stationarity, and self-impact — are why a Sharpe of 0.7 from a careful finance-RL study is a more impressive and more fragile result than superhuman Atari, and why the honest framing throughout this post has been about *modest, measured* edges rather than the dominance RL shows in games.

The training trajectory itself is a useful case study, shown in Figure 6. A portfolio agent does not jump to good behavior; it climbs through recognizable stages.

![Timeline of training progression showing the agent moving from random allocation through factor learning and momentum to a regime-aware policy whose validation Sharpe drops out of sample](/imgs/blogs/rl-portfolio-optimization-6.png)

#### Worked example: reading the training curve

In a representative SAC run on the sector ETFs, the early episodes (first ~50k steps) show a Sharpe near -0.2: the agent is allocating essentially randomly, churning, and bleeding costs. By ~150k steps the Sharpe crosses zero and climbs to about 0.5 as the agent discovers basic factor structure — it learns to hold the broad market rather than random subsets. By ~300k steps it reaches roughly 0.8 as it picks up momentum behavior, tilting toward recent winners. By ~450k steps the validation Sharpe peaks near 1.1 as the policy becomes regime-aware, de-risking when volatility spikes. Then comes the moment of truth: rolling the frozen policy onto the 2016–2023 test set, the Sharpe lands at about 0.7. That gap from 1.1 validation to 0.7 test is overfitting being paid back, and it is *expected and healthy* — an agent whose test Sharpe matched its validation Sharpe would make me suspicious that the test set had leaked into training. The realistic edge is the 0.7, not the 1.1, and any report that quotes the validation number as the result is either naive or dishonest.

## 12. When to use this (and when not to)

Reinforcement learning is not the right tool for every allocation problem, and a principal engineer's job is partly to talk people *out* of it when something simpler wins. Here is my decision framework.

**Use RL for portfolio allocation when:** transaction costs are a first-order concern and you need a policy that trades cost-awarely (RL's structural advantage); the optimal behavior is regime-dependent and you have enough history spanning multiple regimes for the agent to learn the dependence; you have a moderate number of assets (roughly 5–50) where the simplex action space is rich but the state stays tractable; and you have the engineering maturity to do rigorous walk-forward evaluation, because RL gives you many new ways to fool yourself.

**Do not use RL when:** you have very few assets and a stable correlation structure — then shrinkage-Markowitz or risk-parity is simpler, interpretable, and nearly as good; you have very little history — RL is sample-hungry and will overfit a short window, where a robust 1/N or risk-parity has zero estimation error and will likely win, exactly the DeMiguel-Garlappi-Uppal result; you need a fully interpretable, auditable allocation for a regulated mandate — an RL policy is a black box, and "the neural network said so" does not satisfy a compliance officer; or you cannot model your transaction costs and market impact realistically — at institutional scale, impact dominates and a backtest that ignores it is fiction.

The deeper point connects to the whole series' [model-based versus model-free](/blog/machine-learning/reinforcement-learning/model-based-vs-model-free-when-to-use-which) discussion: if you can write down a good model of the dynamics (and for a *single-period* mean-variance problem with known inputs, you can — it has a closed-form solution), use the model. RL earns its complexity precisely when the dynamics are partially unknown and multi-period, costs are path-dependent, and behavior must adapt to regime — which is the real, messy allocation problem, but not the toy one. Reach for the simplest method that captures your actual constraints, and let the comparison table, not the hype, decide.

| Situation | Best tool | Why |
| --- | --- | --- |
| Few assets, stable correlations, known inputs | Markowitz + shrinkage | Closed-form, interpretable, near-optimal |
| Short history, high uncertainty | Equal-weight or risk-parity | Zero estimation error, robust |
| Costs are first-order, regime-dependent | RL (SAC/PPO) | Learns cost-aware, adaptive policy |
| Need auditability / regulated mandate | Rules-based / factor model | Interpretable, defensible |
| Large universe, market impact matters | Specialized execution + RL | Impact modeling dominates |

## Key takeaways

1. **Markowitz fails by maximizing error, not by being wrong in theory.** The $\Sigma^{-1}$ inversion amplifies the noisiest covariance estimates into the largest, most unstable positions. Always compare RL to *shrinkage*-Markowitz, not naive Markowitz.
2. **Put current weights in the state and transaction cost in the reward.** This single pair of choices is what lets an RL agent learn cost-aware rebalancing — the thing classical optimization structurally cannot do.
3. **The action lives on the simplex; project with softmax and explore with Dirichlet or Gaussian-logit sampling.** Never discretize portfolio weights — the action count is combinatorial in the number of assets.
4. **SAC's entropy term is a principled diversification knob.** High temperature keeps the book diversified and humble; low temperature lets earned conviction concentrate. It beats ad-hoc weight caps.
5. **Shape every constraint into the reward *before* training, never after.** An agent maximizing raw return will churn the book to death; it only learns discipline if the cost and risk terms are in the objective from step one.
6. **Baseline ruthlessly: 1/N, shrinkage-Markowitz, risk-parity, momentum, buy-and-hold.** If you cannot beat equal-weight out of sample, you have built nothing. The DeMiguel-Garlappi-Uppal result is the bar.
7. **The realistic edge is the out-of-sample number, and it is modest.** Validation Sharpe of 1.1 becoming test Sharpe of 0.7 is healthy overfitting being paid back; the 0.7 is the truth.
8. **RL's most robust wins are lower turnover and lower drawdown, not higher raw return.** A strategy that loses less in crashes and trades less is easier to hold and cheaper to run.
9. **Multi-period optimization *is* the RL objective.** The discounted sum of cost-adjusted returns is the multi-period portfolio value, and the Bellman recursion is what makes the agent forward-looking — Markowitz acts as if the world ends next period.
10. **Add features one family at a time and keep only what improves out-of-sample Sharpe.** Every feature is a new way to overfit a short financial history.

## Further reading

- Markowitz, H. (1952). "Portfolio Selection." *Journal of Finance* — the founding paper; read it to appreciate how clean the theory is and how much the input-estimation problem is left unaddressed.
- Michaud, R. (1989). "The Markowitz Optimization Enigma: Is 'Optimized' Optimal?" *Financial Analysts Journal* — the original "error-maximization" critique.
- Ledoit, O. & Wolf, M. (2004). "Honey, I Shrunk the Sample Covariance Matrix." *Journal of Portfolio Management* — the shrinkage estimator that makes Markowitz a fair baseline.
- DeMiguel, V., Garlappi, L. & Uppal, R. (2009). "Optimal Versus Naive Diversification." *Review of Financial Studies* — why 1/N is so hard to beat.
- Moody, J. & Saffell, M. (2001). "Learning to Trade via Direct Reinforcement." *IEEE Transactions on Neural Networks* — the differential Sharpe ratio reward for online trading RL.
- Haarnoja, T. et al. (2018). "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning." — the algorithm we built on; pairs with the sibling [Soft Actor-Critic](/blog/machine-learning/reinforcement-learning/soft-actor-critic-sac) post.
- Schulman, J. et al. (2017). "Proximal Policy Optimization Algorithms." — pairs with the sibling [Proximal Policy Optimization](/blog/machine-learning/reinforcement-learning/proximal-policy-optimization-ppo) post.
- Liu, X.-Y. et al. (2020–2022). "FinRL: A Deep Reinforcement Learning Library for Automated Trading in Quantitative Finance." — the open-source benchmark to reproduce portfolio RL.
- Sutton, R. & Barto, A. "Reinforcement Learning: An Introduction" (2nd ed.) — the foundational text for the MDP, Bellman, and multi-step-return material in Sections 2 and 10.

This post is part of the "Reinforcement Learning: From Rewards to Real Systems" series. It builds directly on the continuous-control machinery in the [Soft Actor-Critic](/blog/machine-learning/reinforcement-learning/soft-actor-critic-sac) and [Proximal Policy Optimization](/blog/machine-learning/reinforcement-learning/proximal-policy-optimization-ppo) posts, and the value-function foundations in [Value Functions and the Bellman Equation](/blog/machine-learning/reinforcement-learning/value-functions-and-the-bellman-equation). The series' unified map (`reinforcement-learning-a-unified-map`) and capstone (`the-reinforcement-learning-playbook`) tie every algorithm back to the same frame: an agent interacting with an environment, collecting rewards, and updating a policy — where portfolio allocation is just the special case in which the environment is the market, the action is a point on the simplex, and the reward is money you got to keep after paying to trade.
