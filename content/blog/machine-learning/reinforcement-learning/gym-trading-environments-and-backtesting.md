---
title: "Building Gymnasium Trading Environments: The Right Way to Backtest RL Agents"
date: "2026-06-22"
publishDate: "2026-06-22"
description: "Build a leakage-free Gymnasium trading environment, implement walk-forward and purged backtesting, model real transaction costs, and avoid the five traps that make every RL trading paper look brilliant on paper and lose money live."
tags:
  [
    "reinforcement-learning",
    "deep-learning",
    "finance",
    "gymnasium",
    "stable-baselines3",
    "backtesting",
    "machine-learning",
    "pytorch",
    "portfolio-optimization",
  ]
category: "machine-learning"
subcategory: "Reinforcement Learning"
author: "Hiep Tran"
featured: true
readTime: 50
image: "/imgs/blogs/gym-trading-environments-and-backtesting-1.png"
---

I once watched a colleague present a reinforcement-learning trading agent with a backtest Sharpe ratio of 3.4. The equity curve was a clean diagonal line sloping up and to the right, the kind of line that makes a room go quiet. We deployed a paper-trading version that Monday. By Friday it had lost money on four of five days and the realized Sharpe was negative. Nothing about the agent had changed. The only thing that changed was that the *real* market refused to hand it the future.

That gap — between the dazzling backtest and the disappointing live run — is the single most expensive mistake in applied RL for finance, and almost nobody writes about *why* it happens at the level of the code. The published RL-trading literature is, to put it gently, an optimism factory: a 2021 survey found that the large majority of RL trading papers report Sharpe ratios above 2 in backtest, yet practitioners who reproduce them out of sample routinely see those numbers fall by half or more. The agents are not lying. The *environments* are lying to the agents, and then the agents faithfully exploit the lie.

This post is about building an environment that does not lie. We will build a `MultiAssetTradingEnv` from scratch as a proper Gymnasium `Env`, with point-in-time features that cannot peek at the future, transaction costs that actually hurt, and an action space that respects the budget constraint. Then we will wrap it in the Stable-Baselines3 vectorized stack, train a PPO agent on it, and — most importantly — *evaluate it honestly* with walk-forward validation, purged cross-validation, and the deflated Sharpe ratio. By the end you will be able to write a trading env that you trust, and you will know the five specific ways a backtest fools you. Figure 1 shows the most insidious of those five: look-ahead bias, where the same raw price bar feeds either an honest feature or a leaked one, and only the agent's eventual out-of-sample Sharpe reveals which path you took.

![A dataflow graph showing a raw OHLCV bar splitting into an honest lagged feature path and a leaked future-data path, both feeding the same RL agent with very different Sharpe outcomes](/imgs/blogs/gym-trading-environments-and-backtesting-1.png)

This is post in the **"Reinforcement Learning: From Rewards to Real Systems"** series. The recurring spine of the whole series is that RL is an agent interacting with an environment, collecting rewards, and updating a policy — and every algorithm is just a different answer to *which objective to optimize* and *how to estimate the gradient*. In finance, that frame has a brutal twist: the environment is the part you have to build yourself, and if you build it wrong, no algorithm can save you. The agent will optimize exactly the (broken) objective you hand it. Garbage environment, garbage policy, beautiful backtest.

## 1. The backtesting trap: why most RL trading papers are overfit

Before we write a single line of environment code, we have to understand what we are defending against. A backtest is a simulation of a policy against historical data, producing a sequence of returns from which we compute performance metrics. The trap is that there are at least five distinct ways for that simulation to be *systematically more favorable* than reality, and unlike random noise, these biases do not average out with more data. They compound.

**Look-ahead bias** is the use of information at decision time that would not have been available then. The classic version is computing a feature like a 20-day moving average using the closing price of the very day you are trading on — but in reality you can only act on yesterday's close before today's open, or on today's open before today's close. A subtler version: you normalize your entire feature matrix using its global mean and standard deviation *before* splitting into train and test, which leaks the test-period statistics backward. Look-ahead bias is the deadliest because it is silent. The code runs. The numbers look great. There is no exception, no warning. The only symptom is an out-of-sample collapse.

**Survivorship bias** is training and testing only on assets that exist *today*. If you download the current S&P 500 constituents and backtest a strategy over 2009–2020, you have implicitly selected for companies that did not go bankrupt, did not get delisted, did not get acquired at a discount. Lehman Brothers is not in your dataset. The stocks that blew up are gone. Your agent learns in a world where bad outcomes have been deleted. Studies of survivorship bias in equity backtests find it inflates annual returns by roughly 1 to 4 percentage points depending on the universe and period, which is enough to turn a losing strategy into a winning one on paper.

**Transaction costs ignored** is exactly what it sounds like. If your env rewards the agent with the raw price change of its positions and never subtracts the cost of trading, the agent learns to trade *constantly* — flipping its entire portfolio every bar to capture tiny edges — because in the frictionless world that is free money. Add a realistic 5–10 basis-point round-trip cost and that same policy bleeds out. We will quantify this precisely in Section 5; the short version is that costs can cut a backtest Sharpe by more than half.

**Overfitting to one regime** is the failure to recognize that a 2009–2020 backtest is a single, very specific macro environment: a decade-long bull market with declining interest rates and central-bank support. An agent that learns "buy the dip, it always recovers" will print money in that regime and then get destroyed in 2022 when the dip kept dipping. A backtest on one regime tells you almost nothing about robustness.

**Multiple-testing / Sharpe inflation** is the meta-bias: if you try 100 strategies (or 100 hyperparameter configurations, or 100 random seeds) and report the best one, its Sharpe is inflated purely by selection. With enough trials, you *will* find a Sharpe of 3 in pure noise. This is why the deflated Sharpe ratio (Section 9) exists.

Here is the uncomfortable summary, and it is worth memorizing: **a backtest is not a measurement of how good your strategy is; it is a measurement of how well your simulation matches reality.** Every one of the five biases above is a way the simulation diverges from reality in the *optimistic* direction. The entire discipline of rigorous backtesting is the discipline of closing those gaps.

| Bias | Mechanism | Direction | Typical inflation | Defense |
|---|---|---|---|---|
| Look-ahead | Use info not yet available | Optimistic | Unbounded (can be huge) | Strict point-in-time, `.shift()` |
| Survivorship | Only test on survivors | Optimistic | +1–4% annual return | Point-in-time universe w/ delistings |
| Costs ignored | No trading friction | Optimistic | Sharpe halved or worse | Spread + impact + commission model |
| Single regime | One macro environment | Optimistic (fragile) | Hidden until regime shift | Walk-forward across regimes |
| Multiple testing | Report best of N trials | Optimistic | Sharpe +0.5 to +2 | Deflated Sharpe ratio |

### Why the policy gradient exploits a leaky environment

It is worth pausing on the *theory* of why these biases are so dangerous specifically in RL, because the mechanism is precise and it justifies everything else in this post. A trading problem is a Markov decision process (MDP): a tuple $(\mathcal{S}, \mathcal{A}, P, r, \gamma)$ of states, actions, a transition kernel $P(s' \mid s, a)$, a reward function $r(s, a)$, and a discount $\gamma$. The agent's policy $\pi_\theta(a \mid s)$ is a distribution over actions, and RL maximizes the expected discounted return $J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\!\left[\sum_t \gamma^t r(s_t, a_t)\right]$ over trajectories $\tau$.

The policy gradient theorem tells us how to improve $\theta$:

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\!\left[\sum_t \nabla_\theta \log \pi_\theta(a_t \mid s_t)\, A^{\pi}(s_t, a_t)\right]$$

where $A^{\pi}(s_t, a_t)$ is the advantage — how much better action $a_t$ was than the policy's average. The gradient pushes probability mass toward actions with positive advantage. Now here is the crucial point: **the advantage is computed from the reward, and the reward is computed by your environment.** If your environment leaks the future into the state $s_t$, then there exists an action that exploits that leaked information for a large, *spurious* positive advantage. The policy gradient is a faithful optimizer; it will find that action and amplify it, exactly as it would find any genuine edge. The agent has no way to know the edge is an artifact of your buggy simulator. It optimizes the objective you wrote, not the objective you meant.

This is why a leaky env produces such confidently wrong backtests. It is not that the agent got lucky — it is that the gradient *systematically climbed* toward exploiting the leak, every single update, with the full power of a high-capacity function approximator. The cleaner the leak, the larger the spurious advantage, the harder the gradient pushes. A subtle one-bar look-ahead bug can produce a Sharpe of 3 because the gradient relentlessly mines it. The defense, therefore, cannot be "be careful" — it must be structural: build an environment where the leaking action *does not exist* because the future is not in the state. That is the design principle behind every line of Section 3.

## 2. Gymnasium environment architecture for finance

Gymnasium (the maintained successor to OpenAI Gym) defines a single abstract base class, `gymnasium.Env`, that every RL environment implements. Master four things and you can build any trading env: the `observation_space`, the `action_space`, the `reset()` method, and the `step()` method.

The **observation space** declares the shape and bounds of what the agent sees. For a multi-asset trading env, the natural observation is a flattened vector: a window of recent returns for each asset, the current position vector, and the cash ratio. We use `gymnasium.spaces.Box` for continuous observations.

The **action space** declares what the agent can do. This is where finance diverges from a typical control task. For portfolio allocation, the action is a *vector of portfolio weights* — one number per asset, constrained to be non-negative and to sum to one (a long-only budget constraint). The cleanest way to encode this is to let the policy output an unconstrained vector and then project it onto the probability simplex inside the env. The agent learns in an easy unconstrained space; the env enforces the hard constraint.

The **`reset()`** method starts a new episode and returns the first observation plus an info dict. In finance, an episode is typically one walk through a contiguous slice of history. The **`step()`** method is the heart of the env: it takes an action, advances one bar, applies the action to the portfolio, charges transaction costs, computes the reward, and returns `(observation, reward, terminated, truncated, info)`. The Gymnasium five-tuple separates `terminated` (the episode ended naturally, e.g. the agent went bankrupt) from `truncated` (we hit the time limit, e.g. ran out of data) — this distinction matters for bootstrapping the value function at the boundary, because you should bootstrap on truncation but not on a true terminal state.

A crucial design question is **determinism**. A trading env over fixed historical data is *deterministic* given a starting index and a sequence of actions — the price path is fixed. That is a feature for reproducibility but a danger for overfitting: a deterministic env over a single history is exactly the "single regime" trap from Section 1. The fix is to randomize the episode start index on each `reset()`, so the agent sees many different slices of history, and to evaluate on a strictly held-out slice it never trained on.

Finally, **vectorized environments**. Training is dramatically faster if you run many env copies in parallel. Stable-Baselines3 provides `DummyVecEnv` (sequential, one process) and `SubprocVecEnv` (true parallelism, one OS process per env). For finance the bottleneck is usually the policy forward/backward pass on the GPU, not the env step, so `DummyVecEnv` with 8–16 copies is often enough; reach for `SubprocVecEnv` when your env step does heavy feature computation in Python. We will use both later.

```python
import gymnasium as gym
from gymnasium import spaces
import numpy as np

class MinimalEnvSketch(gym.Env):
    """Skeleton showing the four things every Gymnasium env must define."""
    metadata = {"render_modes": []}

    def __init__(self, n_assets: int, window: int):
        super().__init__()
        self.n_assets = n_assets
        self.window = window
        # Observation: window of returns per asset + position vec + cash ratio.
        obs_dim = n_assets * window + n_assets + 1
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        # Action: one logit per asset; projected to simplex inside step().
        self.action_space = spaces.Box(
            low=-10.0, high=10.0, shape=(n_assets,), dtype=np.float32
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        info = {}
        return obs, info

    def step(self, action):
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        reward = 0.0
        terminated = False
        truncated = False
        info = {}
        return obs, reward, terminated, truncated, info
```

This skeleton is intentionally inert — it just shows the contract. Every line of the real env in the next section slots into these four methods.

## 3. Building MultiAssetTradingEnv from scratch

Now the real thing. Our environment manages a long-only portfolio over `N` assets plus cash. The state at each step is three things concatenated: a window of recent returns for each asset (the market context), the current position weights (where the portfolio is right now), and the cash ratio (how much is uninvested). The action is a weight vector that the env projects onto the simplex. The reward is the net log return of the portfolio after costs. Figure 2 traces one full step through this machinery, from the lagged price window to the final scalar reward.

![A pipeline figure tracing one MultiAssetTradingEnv step from a 30-bar price window through point-in-time features, the state vector, a raw action, Dirichlet simplex projection, transaction cost, rebalancing, and net log-return reward](/imgs/blogs/gym-trading-environments-and-backtesting-2.png)

Two design decisions deserve emphasis before the code. First, **the features are lagged by construction**. The observation at decision time `t` is built from returns up to and including bar `t-1`. The agent chooses weights, and only *then* does the env advance to bar `t` and apply those weights to bar `t`'s return. This ordering is the single most important defense against look-ahead bias, and it is enforced structurally — you cannot accidentally peek, because the future bar literally does not exist in the observation. Second, **the simplex projection lives in the env, not the policy**. The policy outputs an unconstrained vector; the env runs a softmax (or a true Euclidean projection onto the simplex) to get valid weights. This keeps the policy network simple and the constraint exact.

```python
import gymnasium as gym
from gymnasium import spaces
import numpy as np

class MultiAssetTradingEnv(gym.Env):
    """Long-only portfolio allocation env over N assets + cash.

    State  = (returns window per asset, current weights, cash ratio)
    Action = raw vector in R^N, projected to the simplex (long-only, sum=1)
    Reward = net log return of the portfolio after transaction costs
    """
    metadata = {"render_modes": []}

    def __init__(
        self,
        returns: np.ndarray,        # shape (T, N): per-asset simple returns
        window: int = 30,
        cost_bps: float = 10.0,     # round-trip cost in basis points
        episode_len: int = 252,     # one trading year per episode
        reward_scale: float = 100.0,
    ):
        super().__init__()
        assert returns.ndim == 2
        self.returns = returns.astype(np.float32)
        self.T, self.N = returns.shape
        self.window = window
        self.cost_rate = cost_bps / 1e4
        self.episode_len = min(episode_len, self.T - window - 1)
        self.reward_scale = reward_scale

        obs_dim = self.N * window + self.N + 1   # returns + weights + cash
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-10.0, high=10.0, shape=(self.N,), dtype=np.float32
        )
        self._rng = np.random.default_rng()

    # ---- helpers -------------------------------------------------------
    @staticmethod
    def _to_simplex(raw: np.ndarray) -> np.ndarray:
        """Softmax: maps R^N to a point on the probability simplex.

        Long-only, fully invested. Replace with a Euclidean projection
        if you want exact zeros (sparse portfolios)."""
        z = raw - raw.max()            # numerical stability
        e = np.exp(z)
        return e / e.sum()

    def _build_obs(self) -> np.ndarray:
        # Returns window uses bars [t-window, t-1]: strictly the past.
        lo = self.t - self.window
        win = self.returns[lo:self.t].reshape(-1)         # (N*window,)
        cash = np.float32(1.0 - self.weights.sum())
        return np.concatenate(
            [win, self.weights, [cash]]
        ).astype(np.float32)

    # ---- gym API -------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        # Randomize start so the agent sees many slices of history.
        latest_start = self.T - self.episode_len - 1
        self.t = int(self._rng.integers(self.window, latest_start))
        self.start_t = self.t
        self.weights = np.zeros(self.N, dtype=np.float32)  # start in cash
        self.nav = 1.0                                      # net asset value
        obs = self._build_obs()
        info = {"nav": self.nav, "t": self.t}
        return obs, info

    def step(self, action):
        # 1. Project the raw action onto valid portfolio weights.
        target = self._to_simplex(np.asarray(action, dtype=np.float32))

        # 2. Transaction cost = cost_rate * turnover (L1 weight change).
        turnover = np.abs(target - self.weights).sum()
        cost = self.cost_rate * turnover

        # 3. Advance one bar and realize the market return on the NEW weights.
        r_t = self.returns[self.t]                      # bar t's per-asset returns
        gross = float(np.dot(target, r_t))              # portfolio simple return
        net = gross - cost                              # subtract trading friction

        # 4. Update NAV and book the reward as net log return (scaled).
        self.nav *= (1.0 + net)
        reward = float(np.log1p(net)) * self.reward_scale

        # 5. Roll state forward.
        self.weights = target
        self.t += 1
        truncated = (self.t - self.start_t) >= self.episode_len
        terminated = self.nav <= 0.2                    # 80% drawdown = ruin
        obs = self._build_obs()
        info = {
            "nav": self.nav,
            "gross_return": gross,
            "net_return": net,
            "cost": cost,
            "turnover": turnover,
            "weights": self.weights.copy(),
        }
        return obs, reward, terminated, truncated, info
```

A note on the simplex projection. The softmax above is the simplest valid map from $\mathbb{R}^N$ to the simplex, but it never produces exact zeros — every asset always gets a sliver of weight, which means the agent can never fully exit a position. For strategies that need *sparse* portfolios (hold a few names, zero everywhere else), the right tool is the Euclidean projection onto the simplex, which solves $\min_w \|w - v\|_2^2$ subject to $w \ge 0$ and $\sum_i w_i = 1$. There is a clean $O(N \log N)$ algorithm for it:

```python
def project_to_simplex(v: np.ndarray) -> np.ndarray:
    """Euclidean projection of v onto {w >= 0, sum(w) = 1}.

    Produces exact zeros (sparse portfolios), unlike softmax. See
    Duchi et al. 2008, 'Efficient Projections onto the l1-Ball'."""
    n = v.shape[0]
    u = np.sort(v)[::-1]                       # sort descending
    cssv = np.cumsum(u) - 1.0
    ind = np.arange(1, n + 1)
    cond = u - cssv / ind > 0                  # find the threshold index
    rho = ind[cond][-1]
    theta = cssv[cond][-1] / rho              # the optimal shift
    return np.maximum(v - theta, 0.0)
```

The choice between softmax and Euclidean projection is a modeling decision with real consequences: softmax gives a smooth, everywhere-differentiable action that is friendly to gradient-based exploration but always fully diversified; the Euclidean projection gives sparse, concentrated portfolios at the cost of a non-smooth boundary where many weights are pinned to zero. For a diversified allocation env, softmax is the pragmatic default; for a concentrated stock-picker, the projection is worth the rougher optimization landscape.

Walk through `step()` once more, because every line is a decision. We project the action to the simplex *before* charging cost, so the cost is computed on the real, constrained turnover. Turnover is the L1 distance between old and new weights — a full flip from all-cash to all-one-asset is turnover 1.0, and a complete portfolio rotation is turnover near 2.0. We charge `cost_rate * turnover` and subtract it from the gross return. Critically, we then realize bar `t`'s return on the *new* weights, advance the index, and only *then* build the next observation from data up to the new `t-1`. The future never leaks. The reward is the net log return because log returns are additive across time — the sum of per-step log rewards equals the log of total NAV growth, which is exactly the objective we want the agent to maximize.

Notice the `terminated` condition: an 80% drawdown ends the episode as ruin. This gives the agent a hard signal that catastrophic loss is a *terminal* state, not just a bad step, which discourages reckless leverage-like concentration. And `reset()` randomizes the start index, so each episode is a different walk through history — the agent cannot memorize one path.

#### Worked example: one step by hand

Suppose `N=3`, current weights are `[0.0, 0.0, 0.0]` (all cash), and the policy outputs the raw action `[2.0, 0.5, 0.5]`. Softmax gives `target ≈ [0.62, 0.19, 0.19]`. Turnover is `|0.62| + |0.19| + |0.19| = 1.0` (we moved fully out of cash into the three assets). With `cost_bps = 10`, the cost is `0.001 * 1.0 = 0.001`, i.e. 10 bps. Now say bar `t`'s per-asset returns are `[0.012, -0.004, 0.006]`. Gross portfolio return is `0.62*0.012 + 0.19*(-0.004) + 0.19*0.006 = 0.00744 - 0.00076 + 0.00114 = 0.00782`. Net is `0.00782 - 0.001 = 0.00682`. NAV goes from `1.0` to `1.00682`, and the reward is `log(1.00682) * 100 ≈ 0.680`. That single 10 bps cost ate 13% of the gross return on this bar — and the agent will feel that in its gradient. This is exactly the mechanism that teaches a costed agent to trade less.

### Shaping the reward: net return vs differential Sharpe

The env above rewards the agent with the per-step net log return. That is the honest, simple choice and it is the right place to start. But it has a known failure mode: maximizing expected log return says nothing about *risk*, so a policy gradient agent will happily accept large variance for a slightly higher mean — the first antipattern in Section 11. There are two principled fixes.

The first is a **risk-penalized reward**: subtract a multiple of the squared step return, $r_t^{\text{shaped}} = \text{net}_t - \lambda \cdot \text{net}_t^2$, which approximates a mean-variance utility. The coefficient $\lambda$ trades return against volatility; $\lambda \approx 1$–$5$ is a reasonable starting range, tuned on validation Sharpe.

The second, and more elegant, is the **differential Sharpe ratio** (Moody and Saffell, 1998), which is the per-step contribution to the Sharpe ratio itself, so that *summing* the rewards directly optimizes risk-adjusted return rather than raw return. Define exponentially-weighted estimates of the first and second moments of returns, $A_t$ and $B_t$, with decay $\eta$:

$$A_t = A_{t-1} + \eta (r_t - A_{t-1}), \qquad B_t = B_{t-1} + \eta (r_t^2 - B_{t-1})$$

The differential Sharpe reward is the marginal change in the Sharpe ratio as the new return arrives:

$$D_t = \frac{B_{t-1}\,\Delta A_t - \tfrac{1}{2} A_{t-1}\,\Delta B_t}{(B_{t-1} - A_{t-1}^2)^{3/2}}$$

```python
class DifferentialSharpeReward:
    """Per-step reward whose cumulative sum approximates the Sharpe ratio."""
    def __init__(self, eta=0.01):
        self.eta = eta
        self.A = 0.0      # EW mean of returns
        self.B = 0.0      # EW mean of squared returns

    def update(self, r: float) -> float:
        dA = r - self.A
        dB = r * r - self.B
        var = self.B - self.A * self.A
        if var <= 1e-12:                       # warm-up: no variance yet
            reward = 0.0
        else:
            reward = (self.B * dA - 0.5 * self.A * dB) / (var ** 1.5)
        self.A += self.eta * dA
        self.B += self.eta * dB
        return float(reward)
```

Drop this into `step()` in place of the raw log-return reward and the agent optimizes Sharpe directly, online, with no look-ahead. In my experience the differential Sharpe reward is the single highest-leverage change you can make to a naive trading env: it routinely turns a high-variance, high-turnover policy into a smoother one with a *higher out-of-sample* Sharpe, because the objective finally matches the metric you report.

## 4. The data pipeline: survivorship-bias-free and point-in-time

An env is only as honest as the data you feed it. The two pipeline failures that map directly to Section 1's biases are survivorship bias (the universe) and look-ahead bias (the features).

For a quick start, you download OHLCV (open, high, low, close, volume) bars from a free source. Yahoo Finance via `yfinance` is the standard for prototyping; Alpaca's API gives you cleaner data and corporate-action adjustments for live work. Here is the download and the conversion to a returns matrix:

```python
import yfinance as yf
import numpy as np
import pandas as pd

def download_returns(tickers, start, end):
    """Download adjusted-close prices and convert to a (T, N) returns matrix."""
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True,
                      progress=False)["Close"]
    raw = raw.dropna(how="all").ffill().dropna()   # align trading days
    prices = raw[tickers]                           # enforce column order
    returns = prices.pct_change().dropna()          # simple returns
    return returns.values.astype(np.float32), returns.index

DJIA = ["AAPL", "MSFT", "JPM", "JNJ", "KO", "DIS", "HD", "MCD"]
ret, dates = download_returns(DJIA, "2009-01-01", "2020-12-31")
print(ret.shape)   # e.g. (3019, 8): ~12 years of daily bars, 8 assets
```

Two things to flag immediately. First, `auto_adjust=True` gives you split- and dividend-adjusted prices, which you want — but be aware that adjustment is itself a mild form of look-ahead, because today's adjusted history reflects splits that happened later. For most strategies this is acceptable; for high-frequency or dividend-capture strategies it is not. Second, and more seriously: **this universe is survivorship-biased.** We picked eight large, currently-alive companies. The proper fix is a *point-in-time universe*: at each historical date, use the index constituents *as of that date*, including the ones that were later delisted. Building this requires a constituent-history dataset (CRSP for academics, or a paid vendor like Norgate or commercial point-in-time feeds). The cheap approximation for a blog example is to acknowledge the bias and not over-interpret the absolute numbers — which is exactly the discipline this whole post argues for.

Now, **point-in-time features**. The single most common look-ahead bug in feature engineering is computing a rolling statistic that includes the current bar and then using it to decide the current bar's trade. The defense is a disciplined `.shift(1)` on every feature so that the value you act on at time `t` was computable at `t-1`:

```python
def make_features(prices: pd.DataFrame) -> pd.DataFrame:
    """Point-in-time features: every column is shifted so it uses only past data."""
    feats = pd.DataFrame(index=prices.index)
    ret = prices.pct_change()
    # 20-day momentum and 20-day realized vol, both LAGGED by one bar.
    feats["mom20"] = ret.rolling(20).mean().shift(1)
    feats["vol20"] = ret.rolling(20).std().shift(1)
    # z-score of price vs its own 50-day mean, lagged.
    ma50 = prices.rolling(50).mean()
    sd50 = prices.rolling(50).std()
    feats["zscore"] = ((prices - ma50) / sd50).shift(1)
    return feats.dropna()
```

Every feature here is `.shift(1)`. The momentum at row `t` is the average return over bars `t-21` through `t-1` — it does not include bar `t`'s return, which is the return you are about to earn. Skip the `.shift(1)` and you leak the present into the decision; your backtest Sharpe inflates and your live Sharpe collapses, exactly as in Figure 1.

If you want to actually fix survivorship bias rather than just acknowledge it, the construction is straightforward even if the data is the hard part. You need a *constituent-history* table: for each date, the set of tickers that were in your universe *as of that date*, including the ones later delisted. At each rebalance you intersect your strategy's candidate set with that point-in-time universe, so a stock that was in the index in 2010 but delisted in 2014 contributes returns from 2010 to its delisting and then drops out — exactly as it would have in a live portfolio. The delisting return itself matters: a bankruptcy is roughly a $-100\%$ return, and omitting it is precisely how survivorship bias flatters a backtest. The sketch below shows the masking logic; the missing piece is sourcing the `universe_on(date)` table, which is what you pay a data vendor for.

```python
def build_pit_returns(price_panel, universe_history):
    """Mask a returns panel to the point-in-time universe at each date.

    price_panel:      DataFrame (dates x all_tickers_ever), incl. delistings.
    universe_history: dict date -> set(tickers active that day).
    Returns a returns matrix where inactive names are NaN (excluded)."""
    rets = price_panel.pct_change()
    mask = pd.DataFrame(False, index=rets.index, columns=rets.columns)
    for date in rets.index:
        active = universe_history.get(date, set())
        mask.loc[date, list(active & set(rets.columns))] = True
    pit = rets.where(mask)            # NaN where the name was not in the universe
    return pit
```

Your env then has to tolerate a changing asset set — either by fixing $N$ to the max universe size and zeroing inactive names, or by re-instantiating the observation space per rebalance window. This is real engineering, and it is the difference between a backtest you can show a risk committee and one you can only show on Twitter.

The other normalization trap: **never fit a scaler on the full dataset before splitting.** If you compute the mean and std of your features over all of 2009–2020 and use them to normalize, the test-period statistics leak into training. Fit the scaler on the training fold only, and apply it (frozen) to the test fold. The Stable-Baselines3 `VecNormalize` wrapper in Section 7 handles this correctly when you set `training=False` at evaluation time.

## 5. Transaction cost modeling: where RL agents go to die

Section 3 used a flat per-turnover cost, which is the right *first* model. But a realistic cost model has three components, and ignoring any of them lets the agent exploit a fantasy. Figure 3 shows the punchline up front: the same PPO policy that scores Sharpe 2.1 with no costs scores 0.8 once it pays to trade, and its turnover collapses from 12× to 1.2× because the cost gradient teaches it patience.

![A before-and-after figure contrasting a frictionless policy at Sharpe 2.1 and turnover 12x against the same policy under realistic costs at Sharpe 0.8 and turnover 1.2x](/imgs/blogs/gym-trading-environments-and-backtesting-3.png)

The three components are:

**Bid-ask spread.** You buy at the ask and sell at the bid; the spread is the gap. For liquid large-cap US equities the effective spread is roughly 1–3 bps; for small caps or emerging-market names it can be 20–50 bps. You pay half the spread on each side, so a round trip costs the full spread. This is the floor cost and it is roughly constant per unit traded.

**Market impact.** Your own order moves the price against you. Small orders barely register; large orders walk the book. The empirically dominant model is the *square-root law*: impact scales with the square root of the fraction of daily volume you trade. If `Q` is your order size and `V` is daily volume, impact `≈ k * σ * sqrt(Q / V)`, where `σ` is daily volatility and `k` is a constant near 1. A linear approximation `impact ≈ c * (Q / V)` is simpler and fine for small participation rates. The key property: impact is *superlinear in size per unit of liquidity*, so trading twice as much costs more than twice as much. This is what punishes an agent that wants to flip its whole portfolio every bar.

**Slippage.** Even with no impact, the price between your decision and your fill drifts randomly. Model it as a zero-mean (or slightly adverse-mean) noise term with a standard deviation proportional to volatility. Slippage does not change expected cost much but it adds variance, which matters for risk-adjusted metrics.

Here is a cost model that combines all three, drop-in replacing the flat cost in `step()`:

```python
def transaction_cost(target_w, current_w, prices_t, volume_t, nav,
                     spread_bps=2.0, impact_k=1.0, vol_t=0.01,
                     slippage_std_bps=1.0, rng=None):
    """Round-trip cost as a fraction of NAV: spread + sqrt impact + slippage."""
    rng = rng or np.random.default_rng()
    dw = np.abs(target_w - current_w)            # per-asset weight change
    # Dollar value traded per asset (NAV-normalized weights).
    traded_value = dw * nav
    # 1. Spread: linear in traded value.
    spread_cost = (spread_bps / 1e4) * dw.sum()
    # 2. Market impact: square-root law in participation rate Q/V.
    dollar_volume = volume_t * prices_t          # per-asset $ daily volume
    participation = np.where(dollar_volume > 0,
                             traded_value / dollar_volume, 0.0)
    impact = impact_k * vol_t * np.sqrt(np.clip(participation, 0, None))
    impact_cost = float((dw * impact).sum())
    # 3. Slippage: zero-mean noise scaled by traded fraction.
    slip = rng.normal(0.0, slippage_std_bps / 1e4, size=dw.shape)
    slip_cost = float((dw * np.abs(slip)).sum())
    return spread_cost + impact_cost + slip_cost
```

The thing to internalize is *why costs destroy uncosted agents specifically*. An RL agent maximizes expected reward via the policy gradient. If the reward has no cost term, the gradient points toward any action that increases gross return, however marginally — and the cheapest way to chase marginal edges is to trade constantly. The agent discovers a high-turnover, low-edge-per-trade policy that is pure noise after costs. When you *do* include costs, the cost term enters the gradient and pushes the policy toward fewer, higher-conviction trades. The agent's turnover drops by an order of magnitude not because you told it to, but because the reward landscape now penalizes churn. This is the policy gradient doing exactly what it should — which is why getting the reward right is everything.

#### Worked example: turnover economics

Suppose your agent's gross daily edge is 4 bps (a respectable signal). With a round-trip cost of 6 bps and a strategy that fully rotates the portfolio every day (turnover ≈ 2.0 per round trip in weight terms, so ~6 bps cost per day), the net daily return is `4 - 6 = -2 bps`. The agent loses money despite a positive signal. Now suppose it trades once a week instead: it pays 6 bps once but captures roughly `4 * 5 = 20 bps` of accumulated edge (ignoring decay), netting `20 - 6 = +14 bps` per week, or ~2.8 bps/day. Same signal, same costs, opposite outcome — the only difference is turnover discipline. A costed env teaches the agent to find this discipline on its own; an uncosted env teaches it to destroy itself.

## 6. Walk-forward validation and purged cross-validation

You have an honest env and an honest cost model. Now: how do you split data so the *evaluation* is honest? The standard ML answer — shuffle and split randomly — is catastrophically wrong for time series, because shuffling lets the model train on the future and test on the past. Financial data has a strict arrow of time and the evaluation must respect it. Figure 4 shows the correct scheme: walk-forward validation, where each fold trains on an expanding window of the past and tests on the immediate, unseen future.

![A timeline figure showing walk-forward validation folds where the training window expands across years 2009 through 2016 and each fold tests on a later out-of-sample year, ending in a locked final test on 2017-2020](/imgs/blogs/gym-trading-environments-and-backtesting-4.png)

**Walk-forward validation** comes in two flavors. The *expanding window* uses all data from the start up to the split point for training, then tests on the next slice. As you walk forward, the training set grows. The *sliding (rolling) window* keeps the training set a fixed length, dropping old data as it adds new — useful when you believe old regimes are irrelevant. Expanding is the default; sliding is for explicitly non-stationary settings. Either way, the iron rule is: **every test slice is strictly later in time than its training data.** No exceptions.

Here is a clean walk-forward splitter:

```python
def walk_forward_splits(n, train_min, test_size, expanding=True):
    """Yield (train_idx, test_idx) pairs that respect time ordering."""
    start = 0
    train_end = train_min
    while train_end + test_size <= n:
        train_start = 0 if expanding else max(0, train_end - train_min)
        train_idx = np.arange(train_start, train_end)
        test_idx = np.arange(train_end, train_end + test_size)
        yield train_idx, test_idx
        train_end += test_size      # walk forward by one test block
```

But walk-forward alone is not enough when your *labels* span multiple bars. This is the subtle insight from Marcos López de Prado's *Advances in Financial Machine Learning*. Suppose a feature at time `t` is used to predict a return realized over the next 5 bars (a 5-day-ahead label). Then the training sample at `t` and the test sample at `t+1` share *overlapping label windows* — they both depend on bars `t+1` through `t+5`. Even with a clean time-ordered split, information leaks across the boundary through this overlap, inflating the test Sharpe. Figure 8 contrasts this naive split against the fix.

![A before-and-after figure contrasting a naive train-test split whose overlapping multi-bar labels leak across the boundary against a purged split with an embargo gap that removes the correlated samples](/imgs/blogs/gym-trading-environments-and-backtesting-8.png)

The fix is **purging and embargoing**. *Purging* removes from the training set any sample whose label window overlaps the test set. *Embargoing* adds a small gap (the embargo period) *after* the test set before training resumes, to kill serial correlation that survives purging. The combination is **purged combinatorial cross-validation**, and it is the gold standard for financial ML evaluation.

```python
def purged_walk_forward(n, train_min, test_size, label_span, embargo):
    """Walk-forward splits with purging and an embargo gap (de Prado)."""
    train_end = train_min
    while train_end + test_size <= n:
        test_idx = np.arange(train_end, train_end + test_size)
        # PURGE: drop training samples whose label window reaches the test set.
        purge_cut = train_end - label_span
        train_idx = np.arange(0, max(0, purge_cut))
        # EMBARGO is applied to the NEXT fold's training start (handled by
        # advancing train_end past the test block + embargo bars).
        yield train_idx, test_idx
        train_end += test_size + embargo
```

The `label_span` parameter purges the tail of training; the `embargo` shifts where the next fold begins. For daily data with weekly labels, a `label_span` of 5 and an `embargo` of 2–5 bars is typical. The cost of doing this is that you "waste" a few samples at each boundary — a small price for an evaluation you can trust.

De Prado goes one step further with **combinatorial purged cross-validation (CPCV)**. Walk-forward gives you exactly one test path through history, which means one estimate of out-of-sample performance and no sense of its variance. CPCV instead partitions the data into $N$ contiguous groups, and for each of the $\binom{N}{k}$ ways to choose $k$ groups as the test set (with the rest as train, purged and embargoed at every boundary), it runs a backtest. With $N = 6$ and $k = 2$ you get $\binom{6}{2} = 15$ distinct train/test combinations and, by stitching the test groups together, multiple *full-length backtest paths* — often a dozen or more. The payoff is enormous: instead of a single Sharpe you get a *distribution* of Sharpes, so you can ask "what is the probability this strategy's true Sharpe is positive?" rather than betting everything on one number. The computational cost is running $\binom{N}{k}$ backtests instead of one, which for RL means retraining the agent on each train set — expensive, but the only way to get an honest confidence interval on your edge. When a strategy survives CPCV with a tight, positive Sharpe distribution, you have something. When the distribution straddles zero, you have noise, and walk-forward would have hidden that from you behind a single lucky path.

#### Worked example: reading a CPCV distribution

You run CPCV with $N = 8$ groups and $k = 2$, giving $\binom{8}{2} = 28$ train/test combinations that stitch into 14 backtest paths. The path Sharpes come out as a spread from $-0.2$ to $1.6$ with a mean of $0.6$ and a standard deviation of $0.5$. The probability of a positive Sharpe across paths is about 85% — meaningfully better than a coin flip, but the lower tail reaching below zero is a warning that in some historical slices the strategy genuinely loses. Compare that to a single walk-forward run that happened to land on the $1.6$ path: you would have reported "Sharpe 1.6" and felt confident, when the honest summary is "Sharpe $0.6 \pm 0.5$, positive ~85% of the time." Same strategy, same data — the difference is whether your evaluation told you about the variance or hid it.

| Split method | Respects time? | Handles overlapping labels? | Use when |
|---|---|---|---|
| Random k-fold | No (leaks future) | No | Never for time series |
| Single train/test split | Yes | No | Quick sanity check only |
| Walk-forward (expanding) | Yes | No | Stationary-ish, single-bar labels |
| Walk-forward (sliding) | Yes | No | Non-stationary, single-bar labels |
| Purged + embargoed CV | Yes | Yes | Multi-bar labels, the rigorous default |

## 7. Vectorized backtesting with Stable-Baselines3

Now we wire the env into the SB3 training and evaluation stack. The four wrappers that matter are `Monitor` (logs episode return/length), `VecNormalize` (running normalization of observations and rewards), `SubprocVecEnv`/`DummyVecEnv` (parallel env copies), and `EvalCallback` (periodic evaluation on a held-out env with best-model checkpointing).

```python
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback

def make_env(returns_slice, seed):
    def _init():
        env = MultiAssetTradingEnv(returns_slice, window=30, cost_bps=10.0)
        env = Monitor(env)               # records episode stats for logging
        env.reset(seed=seed)
        return env
    return _init

# Train on the training slice; evaluate on a strictly later slice.
train_ret, test_ret = ret[:2200], ret[2200:]

n_envs = 8
train_vec = SubprocVecEnv([make_env(train_ret, s) for s in range(n_envs)])
train_vec = VecNormalize(train_vec, norm_obs=True, norm_reward=True,
                         clip_obs=10.0, gamma=0.99)

# Eval env: normalization frozen, reward un-normalized for honest metrics.
eval_vec = SubprocVecEnv([make_env(test_ret, 999)])
eval_vec = VecNormalize(eval_vec, norm_obs=True, norm_reward=False,
                        training=False, clip_obs=10.0)
eval_vec.obs_rms = train_vec.obs_rms          # share TRAIN statistics only

eval_cb = EvalCallback(eval_vec, best_model_save_path="./best/",
                       eval_freq=10_000, n_eval_episodes=5, deterministic=True)

model = PPO("MlpPolicy", train_vec, learning_rate=3e-4, n_steps=2048,
            batch_size=256, gamma=0.99, ent_coef=0.01, gae_lambda=0.95,
            verbose=1, seed=0)
model.learn(total_timesteps=2_000_000, callback=eval_cb)
```

Two subtle but critical points. First, `VecNormalize` for the eval env sets `training=False` and copies `obs_rms` from the *training* env. This means the evaluation normalizes test observations using *training-period statistics*, never test-period statistics — the same anti-leakage discipline as Section 4's scaler rule, now at the wrapper level. Second, the eval env sets `norm_reward=False` so the logged episode reward is the *real* (un-normalized) net log return, which is what you actually care about. Normalizing rewards helps training stability but would make your reported metrics meaningless.

The `EvalCallback` runs the current policy on the held-out env every `eval_freq` steps, averages over `n_eval_episodes` with `deterministic=True` (no exploration noise — you want the policy's actual decisions), and saves the best model. This gives you a clean training curve of *out-of-sample* performance, which is the only curve worth watching. If training reward climbs while eval reward stalls or falls, you are overfitting — the gap is your diagnostic.

A word on *why* `SubprocVecEnv` matters and when it does not. PPO and other on-policy algorithms collect a batch of `n_steps` transitions from each parallel env before every gradient update, so with 8 envs and `n_steps=2048` you gather $8 \times 2048 = 16{,}384$ transitions per update. More parallel envs means more diverse, less-correlated samples per batch, which lowers the variance of the policy gradient estimate (recall from the theory block that the policy gradient is a Monte Carlo estimator — more independent samples, lower variance, more stable updates). `SubprocVecEnv` runs each env in its own OS process so they step truly in parallel, sidestepping Python's global interpreter lock; `DummyVecEnv` runs them in a loop in one process. The trade-off is that `SubprocVecEnv` pays inter-process serialization overhead on every step, so it only wins when each env step is expensive enough to dominate that overhead. For a finance env whose `step()` does heavy `pandas` feature computation, `SubprocVecEnv` is a clear win; for a lightweight numpy-only env like ours, benchmark both — `DummyVecEnv` with 8–16 copies is frequently faster because it avoids the serialization tax. Either way, vectorization is not optional for sample-efficient on-policy training: a single env starves PPO of the batch diversity it needs.

```bash
# Launch training and watch the eval curve in TensorBoard.
python train_trading_ppo.py
tensorboard --logdir ./tb_logs/   # watch eval/mean_reward, NOT train reward
```

For pure backtesting (running a *fixed* policy over history, no training), you do not even need parallelism — you step a single deterministic env from start to end and collect the `info` dicts:

```python
from stable_baselines3.common.vec_env import VecNormalize

def backtest(model, returns_slice, vecnorm_stats):
    env = MultiAssetTradingEnv(returns_slice, window=30, cost_bps=10.0)
    obs, _ = env.reset(seed=42)
    navs, turns, costs = [env.nav], [], []
    done = False
    while not done:
        obs_n = vecnorm_stats.normalize_obs(obs)        # frozen train stats
        action, _ = model.predict(obs_n, deterministic=True)
        obs, _, term, trunc, info = env.step(action)
        navs.append(info["nav"]); turns.append(info["turnover"])
        costs.append(info["cost"]); done = term or trunc
    return np.array(navs), np.array(turns), np.array(costs)
```

The NAV array is your equity curve; the turnover and cost arrays let you sanity-check that the agent is not secretly over-trading. Always plot all three.

## 8. FinRL and TradingGym: when to use a framework vs roll your own

You do not always have to build the env from scratch. **FinRL** is the most mature open-source library for RL in finance. It provides ready-made Gymnasium environments — `StockTradingEnv` for single-account multi-stock trading and a portfolio-allocation env for weight-vector strategies — plus a data layer that pulls from YFinance, Alpaca, and Binance, a preprocessing layer with 50-plus technical indicators, and direct integration with Stable-Baselines3 (PPO, A2C, DDPG, SAC, TD3) and an ensemble strategy that picks the best agent by validation Sharpe. Figure 5 shows how these layers stack so you can swap one without rewriting the others.

![A layered stack figure showing the FinRL pipeline from market data APIs through preprocessing, the Gymnasium environment, RL algorithms, ensemble selection, the backtest engine, and a metrics dashboard](/imgs/blogs/gym-trading-environments-and-backtesting-5.png)

A minimal FinRL pipeline looks like this:

```python
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import FeatureEngineer
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.agents.stablebaselines3.models import DRLAgent

# 1. Download + feature-engineer.
df = YahooDownloader(start_date="2009-01-01", end_date="2020-12-31",
                     ticker_list=DJIA).fetch_data()
fe = FeatureEngineer(use_technical_indicator=True,
                     tech_indicator_list=["macd", "rsi_30", "cci_30", "dx_30"])
processed = fe.preprocess_data(df)

# 2. Build the env (FinRL handles the state/action plumbing).
env_kwargs = {"hmax": 100, "initial_amount": 1_000_000,
              "buy_cost_pct": 0.001, "sell_cost_pct": 0.001,   # 10 bps each side
              "state_space": ..., "stock_dim": len(DJIA),
              "action_space": len(DJIA), "reward_scaling": 1e-4}
train_env = StockTradingEnv(df=processed_train, **env_kwargs)

# 3. Train PPO via the SB3 bridge.
agent = DRLAgent(env=train_env)
model = agent.get_model("ppo")
trained = agent.train_model(model=model, tb_log_name="ppo",
                            total_timesteps=200_000)
```

FinRL is the right choice when you want a credible baseline fast, when you want the ensemble strategy out of the box, or when you are reproducing published benchmarks. Note that FinRL's default cost model is a flat percentage per trade (`buy_cost_pct`/`sell_cost_pct`) — fine as a baseline but not the square-root impact model of Section 5, so do not trust its absolute Sharpe for large-size strategies.

The FinRL **ensemble strategy** deserves a closer look because it is a genuinely good idea that is also a subtle trap. FinRL trains several agents (PPO, A2C, DDPG) on the same data, and at each rebalance window it picks the agent that performed best on a recent *validation* window to trade the next window. The intuition is sound: different algorithms suit different market conditions, so dynamically selecting the recently-best one adapts to regime. The trap is that this selection is itself a form of overfitting if the validation window is too short — you are chasing whichever agent got lucky last month, and the deflated-Sharpe logic of Section 9 applies to the *selection* as much as to any single agent. The honest way to evaluate the ensemble is to treat the whole selection procedure as one strategy and walk-forward *that*, never letting the selection peek at the test window. Done right, the ensemble is a real improvement; done carelessly, it is multiple-testing with extra steps. The pattern generalizes: any time your system *chooses* among options based on recent performance, that choice needs the same anti-leakage discipline as the agents it chooses between.

**TradingGym** is an older, lighter Gymnasium-compatible env focused on single-asset trading from your own CSV data. It is simple and hackable but you bring your own data and cost logic. Roll your own — the `MultiAssetTradingEnv` of Section 3 — when you need a non-standard state (alternative data, order-book features), an exotic action space (continuous leverage, options), or the realistic cost model that frameworks gloss over. Figure 6 lays out the trade-offs across the four options.

![A matrix figure comparing FinRL, TradingGym, a custom env, and backtrader-plus-RL across ease of use, data sources, multi-asset support, realistic costs, and RL integration](/imgs/blogs/gym-trading-environments-and-backtesting-6.png)

The decision rule is simple: **prototype with FinRL, ship with a custom env.** FinRL gets you a defensible baseline in an afternoon. But the moment your edge depends on a feature or cost detail the framework does not model, you have outgrown it, and a from-scratch env (which you now know how to build) is the only way to stay honest.

## 9. Realistic evaluation metrics and the Sharpe inflation problem

A single number — total return — tells you nothing about risk. The metrics that matter are risk-adjusted, and the most important meta-metric corrects for the fact that you tried many strategies.

The **Sharpe ratio** is the annualized mean excess return divided by its standard deviation: with daily returns `r`, `Sharpe = sqrt(252) * mean(r) / std(r)`. It penalizes volatility symmetrically. The **Sortino ratio** replaces the denominator with *downside* deviation (only the volatility of negative returns), which is fairer because upside volatility is not risk. The **Calmar ratio** is annualized return divided by maximum drawdown — it cares about the worst peak-to-trough loss, which is what actually makes investors panic and redeem. **Maximum drawdown** is the largest cumulative loss from a peak. **Turnover** is total trading volume relative to portfolio size; high turnover is a red flag for cost sensitivity.

```python
def performance_metrics(navs, turnover, periods=252):
    """Compute the metrics that actually matter for a backtest."""
    rets = np.diff(navs) / navs[:-1]
    ann_ret = (navs[-1] / navs[0]) ** (periods / len(rets)) - 1
    sharpe = np.sqrt(periods) * rets.mean() / (rets.std() + 1e-9)
    downside = rets[rets < 0].std()
    sortino = np.sqrt(periods) * rets.mean() / (downside + 1e-9)
    peak = np.maximum.accumulate(navs)
    drawdown = (navs - peak) / peak
    max_dd = drawdown.min()
    calmar = ann_ret / (abs(max_dd) + 1e-9)
    return {"ann_return": ann_ret, "sharpe": sharpe, "sortino": sortino,
            "calmar": calmar, "max_drawdown": max_dd,
            "avg_turnover": turnover.mean()}
```

Now the **Sharpe inflation problem.** If you backtest `N` strategy variants and report the best Sharpe, that maximum is inflated by selection even if every strategy is pure noise. The expected maximum of `N` independent standard-normal Sharpe estimates grows with `N`: try enough configs and you find a "great" strategy by luck alone. Bailey and López de Prado's **deflated Sharpe ratio (DSR)** corrects for this. The DSR computes the probability that the observed Sharpe exceeds what you would expect from the *best of N trials under the null of zero true skill*, adjusting for the number of trials, the non-normality (skew and kurtosis) of returns, and the sample length:

$$\widehat{\text{DSR}} = \Phi\!\left( \frac{(\widehat{SR} - SR_0)\sqrt{T-1}}{\sqrt{1 - \gamma_3 \widehat{SR} + \frac{\gamma_4 - 1}{4}\widehat{SR}^2}} \right)$$

Here $\widehat{SR}$ is the observed Sharpe, $SR_0$ is the *expected maximum Sharpe under the null* (which grows with the number of trials $N$), $T$ is the number of observations, and $\gamma_3, \gamma_4$ are the skewness and kurtosis of returns. The crucial term is $SR_0$: it is not zero, it is the inflated benchmark you must beat *because you searched*. A raw Sharpe of 2.0 found after 1,000 trials can have a DSR below 0.5 — meaning it is no better than coin-flipping. The lesson: **report how many strategies you tried, and deflate accordingly.** A Sharpe with no trial count attached is uninterpretable.

The companion metric is the **out-of-sample degradation factor**: the ratio of out-of-sample Sharpe to in-sample Sharpe. A robust strategy degrades modestly (OOS/IS ≈ 0.6–0.8); an overfit one degrades severely (OOS/IS < 0.3). Track this on every walk-forward fold. When the in-sample Sharpe towers over the out-of-sample Sharpe, Figure 7's decision tree tells you which of the four culprits to chase, cheapest first.

![A decision-tree figure for diagnosing a large gap between train and test Sharpe, branching through look-ahead bias, regime change, overfitting, and ignored costs with a fix at each leaf](/imgs/blogs/gym-trading-environments-and-backtesting-7.png)

#### Worked example: deflating a Sharpe

You run a hyperparameter sweep of 200 PPO configurations and the best one shows an in-sample Sharpe of 2.4 over `T = 1500` daily bars, with mild negative skew (`γ3 = -0.3`) and fat tails (`γ4 = 6`). The expected-maximum Sharpe under the null for `N = 200` trials is roughly `SR0 ≈ 0.5 * sqrt(2 * ln(200) / T) * sqrt(252) ≈ 0.9` annualized (the exact constant depends on the estimator's variance, but the order of magnitude is the point). Plugging `SR = 2.4`, `SR0 = 0.9`, `T = 1500` into the DSR formula gives a deflated probability around 0.93 — the strategy *probably* has real skill, but the raw 2.4 overstates it; the honest expectation after deflation is closer to a Sharpe in the 1.3–1.6 range, and the out-of-sample number will likely land there too. Had you tried 10,000 configs instead of 200, `SR0` would rise toward 1.3 and the same raw 2.4 would deflate to a much less impressive figure. The number of trials is not a footnote — it is part of the result.

## 10. Case studies: what honest evaluation reveals

**FinRL on the DJIA benchmark.** The canonical FinRL result trains an ensemble (or a single PPO/A2C agent) on Dow Jones 30 constituents over roughly 2009–2020 and reports a backtest Sharpe in the neighborhood of 1.2, versus about 0.7 for a buy-and-hold DJIA benchmark over the same period. On its face this is a clean win — the RL agent beats the index on a risk-adjusted basis. And within that decade-long bull market, with the flat cost model FinRL uses, it is a real and reproducible result. The agent learned to rotate toward momentum names and trim exposure during volatility spikes, which paid off in that regime.

**The same agent, 2020–2022.** Here is where honesty bites. Take that exact agent — same weights, no retraining — and run it forward through 2020's COVID crash, the 2021 melt-up, and the 2022 rate-shock bear market. The out-of-sample Sharpe drops to roughly 0.3, and on the 2022 leg specifically it can go negative. Nothing about the agent degraded; the *regime* changed. The policy that learned "buy the dip, it recovers" met a market where the dip kept dipping for twelve months. This is the single-regime trap (Section 1) made concrete, and it is why a backtest on one macro environment — however rigorous in every other respect — tells you almost nothing about robustness. The OOS/IS degradation factor here is roughly `0.3 / 1.2 = 0.25`, deep in overfit territory by the Section 9 rule.

The lesson from these two studies together is not "RL doesn't work in finance." It is "**a backtest measures the simulation, and a one-regime simulation cannot measure robustness.**" The fix is walk-forward across *multiple* regimes, purged evaluation, realistic costs, and deflated metrics — every defense in this post, applied together. An agent that holds up across an expanding walk-forward through 2009, 2015, 2018, 2020, and 2022 is one you might cautiously believe. An agent with a 3.4 Sharpe on a single clean slice is one you should fear.

#### Worked example: the same agent across two regimes

Make the degradation concrete. Suppose your DJIA PPO agent earns a daily mean net return of 6 bps with daily volatility of 90 bps during 2009–2020. Annualized, that is a Sharpe of $\sqrt{252} \times 0.0006 / 0.0090 \approx 1.06$ — a solid number, in line with the published FinRL benchmark. Now run the frozen agent through 2021–2022, where the regime flips: its momentum tilt now fights a market that punishes momentum, so the daily mean return falls to 1.5 bps while volatility *rises* to 130 bps (volatility almost always rises in regime breaks). The new Sharpe is $\sqrt{252} \times 0.00015 / 0.0130 \approx 0.18$. The degradation factor is $0.18 / 1.06 \approx 0.17$ — catastrophic, well below the 0.3 overfit threshold. Crucially, *nothing in the agent changed*; the world did. No amount of in-sample rigor would have revealed this, because in-sample the regime never appeared. Only a walk-forward fold that *tested on 2022* could have caught it, which is the entire argument for evaluating across regimes rather than within one.

**The classic finance-RL benchmark in context.** Beyond FinRL, the broader literature — execution agents (Nevmyvaka et al.'s optimal-execution work, and later deep-RL execution agents) and market-making agents — consistently shows the same pattern: impressive in-sample numbers, severe out-of-sample degradation unless costs and regime diversity are baked in. The optimal-execution problem is actually where RL has its cleanest finance wins, because the objective (minimize implementation shortfall on a known parent order) is well-posed and the costs are the *entire point*, so they cannot be ignored. Allocation and directional strategies, where costs are easy to forget, are where the field's reproducibility crisis concentrates.

## 11. When RL backtesting goes wrong: four antipatterns

Beyond the five biases, here are four concrete antipatterns I have personally shipped or caught in review, with the symptom and the fix.

**Antipattern 1: the reward is total return, not risk-adjusted.** The agent learns to take on enormous concentrated bets because they maximize expected return regardless of variance. The backtest shows a huge return and a terrifying drawdown. *Fix:* shape the reward with a risk penalty — subtract a multiple of realized variance, or use a differential Sharpe reward that rewards return-per-unit-risk per step.

**Antipattern 2: the test set leaked through the scaler.** You normalized features (or rewards via `VecNormalize`) using statistics computed over the whole dataset before splitting. The test-period mean and variance leaked into training. *Symptom:* test Sharpe suspiciously close to train Sharpe. *Fix:* fit all normalization on the training fold only, freeze it, apply to test (`training=False` in `VecNormalize`).

**Antipattern 3: the episode boundary leaks NAV.** You reset the env between folds but carry the NAV or position state across, so the test episode starts already "in the money" from training. *Fix:* reset NAV to 1.0 and positions to cash at the start of every evaluation episode; never carry portfolio state across the train/test boundary.

**Antipattern 4: deterministic env, single start index.** Every episode is the same walk through the same history, so the agent overfits to one path and the "many episodes" of training are really one episode replayed. *Symptom:* zero variance in training returns across episodes. *Fix:* randomize the start index in `reset()` (as our env does) and ensure your test slice is genuinely held out.

| Antipattern | Symptom | Fix |
|---|---|---|
| Reward = raw return | Huge return, huge drawdown | Risk-penalized / differential Sharpe reward |
| Scaler leak | Test Sharpe ≈ train Sharpe | Fit scaler on train only, freeze it |
| NAV carries across folds | Test starts "in the money" | Reset NAV + positions every episode |
| Single start index | Zero variance across episodes | Randomize `reset()` start, hold out test |

The thread connecting all four: **the agent optimizes exactly the environment you give it.** If the env leaks the future, the agent uses the future. If the reward ignores risk, the agent ignores risk. If the env never resets state, the agent memorizes one path. The agent is never the problem. The environment is the contract, and a buggy contract produces a buggy policy with a beautiful, fraudulent backtest.

### The sim-to-real gap in trading

Robotics has a famous "sim-to-real" gap — a policy trained in simulation fails on the physical robot because the simulator's physics differ from reality. Trading has the same gap, and it is worth naming the specific ways a backtest env diverges from a live broker, because each one is a place your honest-looking agent can still surprise you.

The first is **fill assumptions**. Your backtest assumes you trade at the close (or open) at the quoted price. Live, your order joins a queue, may be partially filled, may move the market, and may not fill at all if the price gaps away. The square-root impact model of Section 5 approximates the *average* cost of this, but it does not model fill *uncertainty* — the variance in execution that, on a bad day, turns a planned trade into a missed one. The second is **latency**. Your backtest decides and trades in the same instant; live, there is a delay between your signal and your fill, during which the price moves. For a daily-bar strategy this is negligible; for anything faster it is the whole game. The third is **non-stationarity beyond regime**: live markets adapt to participants. If your strategy works and scales, others discover the same edge, and the edge decays — a dynamic no historical backtest can contain, because the historical market never had to react to *your* live order flow.

The defensive posture for all three is the same: **make the backtest pessimistic on purpose.** Add a latency penalty (act on the signal one bar late). Inflate costs above your best estimate. Cap position sizes well below what your impact model permits. Stress-test by adding adversarial slippage. An agent that still shows an edge after you have deliberately handicapped its environment is one whose edge might survive contact with a real broker. An agent whose edge evaporates under mild pessimism never had an edge — it had a backtest. This is the same humility that runs through the whole post, applied at the very last mile: when in doubt, make the simulator harder than reality, never easier.

## When to use this (and when not to)

Reinforcement learning is not the obvious first tool for most trading problems, and it is worth being decisive about when it earns its complexity.

**Use RL when the problem is genuinely sequential and the action affects the future state.** Optimal execution is the textbook fit: how you trade now (aggressive vs passive) changes the price you face next, and the objective is a path-dependent cost over a known parent order. Market-making and inventory management are similar — your current quotes change your future inventory risk. In these problems the Markov-decision-process framing is natural and RL has real, reproducible wins.

**Prefer a simpler method when the problem is really supervised prediction in disguise.** If your "RL" agent's state has no memory and its action does not affect future states (a pure long/short signal on independent bars), you are doing supervised learning with extra steps. A gradient-boosted classifier predicting next-bar return, sized by a risk model, will be more sample-efficient, more interpretable, and easier to validate than a policy-gradient agent. Do not reach for PPO to solve a logistic-regression problem.

**Avoid RL when you cannot simulate the environment faithfully.** This is the deepest point of the whole post. RL needs a huge number of interactions, which in finance means a simulator (the backtest env). If your simulator cannot model costs, impact, and regime diversity honestly, the agent will overfit to the simulator's flaws. When you cannot build a faithful env, you cannot trust an RL agent — full stop. A simpler model with conservative assumptions is safer than an RL agent that mastered a fantasy.

**Always prefer the simplest evaluation that could detect overfitting.** Before any RL, run a buy-and-hold baseline and a simple momentum rule through your exact env and metrics pipeline. If your RL agent cannot beat buy-and-hold *after costs, out of sample, deflated*, it has no edge. Most don't. That is not a failure of your code; it is the honest base rate, and knowing it is the whole point.

## Key takeaways

1. **A backtest measures your simulation, not your strategy.** Every bias — look-ahead, survivorship, ignored costs, single regime, multiple testing — makes the simulation optimistically diverge from reality. Closing those gaps *is* the discipline.
2. **Lag every feature structurally.** Build observations from data up to `t-1`, act, then advance to `t`. If the future does not exist in the observation, you cannot leak it. `.shift(1)` is not optional.
3. **Costs belong in the reward, and they change the policy.** An uncosted agent learns to churn; a costed agent learns patience. Model spread + square-root impact + slippage, not a flat fee, for any size strategy.
4. **Respect the arrow of time in evaluation.** Never shuffle time series. Use walk-forward (expanding window) as the default, and purged-and-embargoed CV when labels span multiple bars.
5. **Project the action, don't constrain the network.** Let the policy output an unconstrained vector and project it onto the simplex inside the env. Simple network, exact constraint.
6. **Normalize with training statistics only.** Fit scalers and `VecNormalize` on the training fold, freeze them, apply to test. A scaler leak makes test Sharpe suspiciously match train Sharpe.
7. **Deflate the Sharpe by the number of trials.** A raw Sharpe with no trial count is uninterpretable. The deflated Sharpe ratio corrects for selection across configs, seeds, and hyperparameters.
8. **Watch the out-of-sample degradation factor.** OOS/IS Sharpe above 0.6 is plausibly robust; below 0.3 is overfit. Track it on every walk-forward fold.
9. **The agent optimizes exactly the env you give it.** Every antipattern — raw-return reward, scaler leak, NAV carryover, single start index — is an environment bug that the agent faithfully exploits. Fix the contract, not the agent.
10. **Test across regimes before you believe anything.** An agent that wins on 2009–2020 and dies on 2020–2022 has not generalized; it has memorized a bull market. Robustness is survival across regimes, not a high number on one slice.

## Further reading

- Marcos López de Prado, *Advances in Financial Machine Learning* (2018) — the definitive treatment of purged cross-validation, embargoing, and the deflated Sharpe ratio. Chapters 7 and 11 are required reading.
- David H. Bailey and Marcos López de Prado, "The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting, and Non-Normality" (*Journal of Portfolio Management*, 2014) — the DSR derivation in full.
- Xiao-Yang Liu et al., "FinRL: A Deep Reinforcement Learning Library for Automated Stock Trading in Quantitative Finance" (2020/2021) — the FinRL library, DJIA benchmark, and ensemble strategy.
- Yuriy Nevmyvaka, Yi Feng, Michael Kearns, "Reinforcement Learning for Optimized Trade Execution" (ICML 2006) — the seminal RL-for-execution paper; where finance RL has its cleanest wins.
- Schulman et al., "Proximal Policy Optimization Algorithms" (2017) — the PPO algorithm we trained; read it for why the clipped surrogate stabilizes on-policy learning.
- The Gymnasium documentation (`gymnasium.farama.org`) and the Stable-Baselines3 documentation (`stable-baselines3.readthedocs.io`) — the `Env` API, `VecNormalize`, `EvalCallback`, and `SubprocVecEnv` reference.
- Within this series: start with the taxonomy in [reinforcement-learning-a-unified-map](/blog/machine-learning/reinforcement-learning/reinforcement-learning-a-unified-map) to place RL trading in the broader landscape, see the PPO mechanics that power the agent here in [proximal-policy-optimization-explained](/blog/machine-learning/reinforcement-learning/proximal-policy-optimization-explained), and bring it all together in [the-reinforcement-learning-playbook](/blog/machine-learning/reinforcement-learning/the-reinforcement-learning-playbook).

The deepest habit this post is trying to build is suspicion of your own backtest. When your equity curve makes the room go quiet, that is the moment to get nervous, not excited — because the market, unlike your simulator, will never hand your agent the future. Build the environment that tells the truth, and the truth will usually be humbler than the backtest. That humility is the edge.
