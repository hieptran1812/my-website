---
title: "RL for Algorithmic Trading: Framing Markets as a Markov Decision Process"
date: "2026-06-22"
publishDate: "2026-06-22"
description: "How to turn a live market into a Gymnasium environment, design a non-leaking reward, and train a DQN agent that actually stops over-trading."
tags:
  [
    "reinforcement-learning",
    "deep-learning",
    "finance",
    "markov-decision-process",
    "q-learning",
    "machine-learning",
    "pytorch",
    "actor-critic",
  ]
category: "machine-learning"
subcategory: "Reinforcement Learning"
author: "Hiep Tran"
featured: true
readTime: 51
image: "/imgs/blogs/rl-for-algorithmic-trading-foundations-1.png"
---

The first trading agent I ever shipped lost money in the most embarrassing way possible: it traded itself to death. On paper its policy looked brilliant — it bought dips, sold rips, and on the training window it turned \$100,000 into \$340,000. In live paper trading it churned the account on transaction costs, flipping its position almost every single day, bleeding 5 basis points each time, until the equity curve looked like a slow leak. The model had learned something real about the training data. It had just learned the wrong thing: that frequent trading was rewarded, because in the backtest the costs were too small to matter and the price noise happened to line up with its signals. It had memorized a regime, not a strategy.

That failure is the perfect entry point into reinforcement learning for trading, because it is *not* a bug you fix with a bigger network or more data. It is a consequence of how you framed the problem. RL gives you an extraordinarily expressive language for sequential decision-making under uncertainty — exactly the shape of trading — but it will faithfully optimize whatever objective you actually wrote down, including the objective "trade as often as possible because the reward function forgot about costs." This post is about getting that framing right. We are going to take a live market, turn it into a [Markov Decision Process](/blog/machine-learning/reinforcement-learning/markov-decision-processes), wrap it in a Gymnasium environment, design a reward that does not leak future information or quietly encourage over-trading, train a Deep Q-Network baseline that beats buy-and-hold on a real backtest, and then graduate to a continuous-action Soft Actor-Critic agent that sizes positions instead of just flipping them.

![A closed-loop diagram showing the trading agent observing state, choosing an action, the market transitioning, costs being charged, and a Sharpe-delta reward feeding back into the agent](/imgs/blogs/rl-for-algorithmic-trading-foundations-1.png)

The figure above is the spine of everything that follows, and it is the spine of this whole series: an agent observes a state, picks an action according to its policy, the environment transitions to a new state and pays a reward, and the agent uses that reward to update its policy. Every algorithm we cover — DQN, PPO, SAC — is a different answer to two questions: *which objective do we optimize* and *how do we estimate its gradient*. By the end of this post you will be able to implement that loop for a single-asset daily trading agent, explain from first principles why a Sharpe-shaped reward is more stable than raw profit-and-loss, run a walk-forward evaluation that does not lie to you, and recognize the specific failure modes — reward hacking into cash, NaN losses on extreme moves, replay buffers poisoned by non-stationarity — that send most trading-RL projects into the ground. We will keep one running example throughout: a daily agent trading a single equity, starting from a policy that loses money, ending at a validation Sharpe above 1.0 and an honest out-of-sample Sharpe around 0.6.

## 1. Why reinforcement learning fits trading (and where supervised learning breaks)

Start with the alternative, because RL only earns its complexity if the simpler approaches genuinely fall short. The two classical approaches to systematic trading are rule-based systems and supervised learning.

A rule-based system is a fixed function from market features to actions: "if the 50-day moving average crosses above the 200-day, go long; if it crosses below, go flat." These are transparent, fast, and easy to risk-manage. They are also frozen. The thresholds were tuned on history, and when the market's character changes — volatility regime shifts, correlations invert, a structural break like a central-bank pivot arrives — the rules keep firing the same way into a market that no longer behaves like the one they were tuned on. You can re-tune, but re-tuning is a human in the loop reacting after the regime has already changed.

Supervised learning looks more adaptive but smuggles in a deep mismatch. The standard setup is: predict tomorrow's return (regression) or tomorrow's direction (classification) from today's features, train on labeled history, then trade on the predictions. The mismatch is threefold. First, the label is wrong: maximizing prediction accuracy is not the same as maximizing risk-adjusted profit, because a model can be right 55% of the time and still lose money if its wins are small and its losses are large. Second, and more fundamentally, supervised learning assumes the training samples are independent and identically distributed — i.i.d. — drawn from a fixed distribution. Markets violate both halves of that assumption. They are not independent: today's price is yesterday's price plus a move, and your own trades move the price and change your position, so consecutive samples are correlated through the state. And they are not identically distributed: the return distribution in a calm bull market is nothing like the one during a crash. Third, supervised learning is myopic. It predicts one step ahead and has no notion that holding a position has consequences three days from now, that entering now costs you the option to enter at a better price later, or that a sequence of small correct decisions compounds.

Reinforcement learning is built for exactly the structure supervised learning fights. It models the problem as *sequential* decision-making, where the agent's action at time `t` influences the state at time `t+1` — which is literally true in trading: your buy changes your position, consumes capital, and (at size) moves the price. It optimizes a *cumulative* objective — discounted future reward — instead of a one-step label, so it naturally values holding through a drawdown if the recovery pays off. And it learns a *policy* — a mapping from state to action — rather than a prediction, so the output is directly the decision you wanted.

The key insight, the one to tattoo on the inside of your eyelids before you write a single line: **trading is not i.i.d.** The diagram in the intro shows why. The agent's action feeds into the market transition and into its own next state. That feedback is the whole reason RL is the right tool and also the whole reason it is so easy to get wrong: the moment your samples become correlated and your distribution shifts, naive training procedures that assumed independence — like a replay buffer that mixes a calm 2017 with a violent 2020 as if they were the same world — start producing nonsense. We will hit that wall in section 9. For now, hold the frame: an agent, an environment that reacts to it, a reward, a loop.

It is worth being precise about the two distinct ways the i.i.d. assumption breaks, because they fail differently and require different cures. The first break is *autocorrelation through state*: the trajectory $(s_t, a_t, r_t, s_{t+1})$ has $s_{t+1}$ deterministically downstream of $s_t$ and $a_t$, so any two consecutive transitions share information. A supervised learner that treats them as independent will badly underestimate the variance of its own performance — it will think it has seen thousands of independent examples when it has really seen a handful of correlated trajectories, and its confidence intervals will be a fiction. The second break is *covariate shift across regimes*: the joint distribution of features and returns is not fixed in time. Formally, if $p_{\text{train}}(s, r)$ is the distribution the agent learned on and $p_{\text{test}}(s, r)$ is the one it is evaluated on, these are simply different measures in finance — the volatility, the autocorrelation structure, even the sign of the equity-bond correlation flip across macro regimes. A model trained under one and deployed under another is, in the strict statistical sense, extrapolating. The first break is what experience replay and target networks were invented to manage; the second is the one no algorithmic trick fully solves and which forces the evaluation discipline of section 8. Holding both in mind is the difference between an agent that looks good in a notebook and one that survives contact with a market it has not seen.

If you want the broader market-mechanics grounding for why regimes shift the way they do — why a calm carry regime gives way to a deleveraging panic — the post on [what a capital market is and how money finds its best use](/blog/trading/capital-markets/what-is-a-capital-market-how-money-finds-its-best-use) frames the macro forces; here we just take it as given that the distribution moves and design around it.

## 2. The MDP formulation for a single-asset daily agent

A Markov Decision Process is the formal object underneath the RL loop. It is a tuple $(\mathcal{S}, \mathcal{A}, P, R, \gamma)$: a set of states $\mathcal{S}$, a set of actions $\mathcal{A}$, a transition function $P(s' \mid s, a)$ giving the probability of landing in state $s'$ after taking action $a$ in state $s$, a reward function $R(s, a, s')$, and a discount factor $\gamma \in [0, 1)$. The defining property — the *Markov property* — is that the future depends on the present state alone, not the full history: $P(s_{t+1} \mid s_t, a_t) = P(s_{t+1} \mid s_1, a_1, \dots, s_t, a_t)$. Once you know the current state, the past is irrelevant for predicting the future.

Let us instantiate this for a daily single-asset agent.

**State.** The naive choice is "today's price," but a single number is not Markov for trading: the price alone does not tell you whether the market is trending or mean-reverting, nor what position you currently hold. A workable state vector packs three kinds of information:

- **Recent dynamics**: the last $N$ daily log-returns, $r_{t-N+1}, \dots, r_t$, where $r_t = \log(p_t / p_{t-1})$. Returns rather than prices, for a stationarity reason we will derive in section 3.
- **Technical context**: a small set of derived indicators — the Relative Strength Index (RSI), a z-scored moving-average gap, realized volatility over a window. These compress longer-horizon structure into the state.
- **Agent state**: the current position. This is non-negotiable. If the agent cannot see whether it is already long, it cannot reason about transaction costs (flipping from long to short costs two trades) and the problem stops being Markov, because the consequence of "sell" depends on whether you currently hold.

A typical state vector might be 12 dimensions: ten returns, an RSI value, and a position indicator.

**Action.** For the discrete baseline we use three actions: $a \in \{-1, 0, +1\}$ meaning go short / go flat / go long, or in the simpler long-only variant $\{0, 1\}$ for flat / long. The action is the *target* position, not a delta — this is a design choice that matters, because a target-position action makes "do nothing" trivially expressible (pick the same action as your current position) and that turns out to be exactly the lever the agent needs to stop over-trading. Later, with SAC, the action becomes a continuous number in $[-1, +1]$: the *fraction* of capital to allocate, which lets the agent size its conviction.

**Transition dynamics.** Here trading differs from a video game. In CartPole the environment's transition is governed by physics the agent cannot influence beyond the cart it controls. In trading the transition has two parts: the *exogenous* market move (tomorrow's price, which a small agent cannot affect) and the *endogenous* position change (the agent's new holdings, which it fully controls). For a price-taking single-asset daily agent we assume the market component is exogenous — a reasonable approximation when your size is tiny relative to daily volume. The transition is then: the price advances by the next bar's realized return, and the position becomes whatever the action specified, minus any transaction friction.

**Episode structure.** An episode is one pass through a fixed backtest window — say, trade every day from 2015-01-01 to 2018-12-31. The episode ends (`done = True`) when we reach the last bar of the window, or early if the agent blows up the account (equity below a floor). Training runs many episodes over the same or rolling windows, and crucially we never let a training episode peek at the validation window.

**The discount factor** $\gamma$ controls how far ahead the agent cares. At $\gamma = 0$ it is myopic (only today's reward). At $\gamma \to 1$ it weighs the distant future as heavily as the present. For daily trading over windows of a few hundred bars, $\gamma$ around 0.95–0.99 is typical — high enough to value holding through short drawdowns, low enough to keep the value function's effective horizon finite and the learning stable. As a rule of thumb, the effective horizon is roughly $1/(1-\gamma)$ steps, so $\gamma = 0.99$ means the agent cares about consequences roughly 100 bars out, and $\gamma = 0.95$ about 20. Pick it to match the timescale of the edge you believe exists — a multi-week momentum edge wants a higher $\gamma$ than a one-day mean-reversion edge.

The objective the agent maximizes, to make all of this concrete, is the expected discounted return from the current state under its policy $\pi$:

$$
V^\pi(s) = \mathbb{E}_\pi\left[ \sum_{k=0}^{\infty} \gamma^k R_{t+k+1} \,\Big|\, S_t = s \right]
$$

This is the *value function* — the thing every value-based RL algorithm tries to estimate. For trading it is the expected discounted future risk-adjusted return of being in a given market state with a given position, and the whole project is to learn it well enough to act on.

**The POMDP caveat.** The Markov property is an assumption, and in markets it is only approximately true. The "true state" of a market includes things you cannot observe: order-book depth, other participants' inventories, news not yet priced in. What you actually feed the agent is an *observation* that is a lossy function of the hidden state. That makes the real problem a Partially Observable MDP (POMDP), where the agent sees $o_t$ drawn from $O(o_t \mid s_t)$ rather than $s_t$ itself. The practical fix is to make the observation as Markov-as-possible by stacking a window of recent history (so the agent infers latent state from dynamics) or, in deep RL, using a recurrent policy (an LSTM/GRU that maintains an internal belief). For a daily single-asset baseline, a well-chosen feature window gets you close enough that the MDP framing is productive; just keep in the back of your mind that the Markov assumption is a modeling choice you are making, not a fact about the market.

## 3. State-space engineering: the stationarity requirement

The single most consequential decision in trading RL is what goes into the state, and the single most common mistake is feeding raw prices. Let us derive why, because the reason is not "best practice," it is mathematics.

![A two-panel comparison contrasting a raw-price state that drifts out of the training range at test time against z-scored returns that stay in distribution](/imgs/blogs/rl-for-algorithmic-trading-foundations-8.png)

A neural network — the Q-network or the policy network — is a function approximator trained on the distribution of inputs it saw during training. Its guarantees, such as they are, hold *in distribution*. Feed it inputs from a region it never trained on and it extrapolates, and neural-network extrapolation is wild and unreliable. Now consider the price of an asset. During a 2015–2018 training window it might range from \$80 to \$120. The network learns a Q-function over that range. Come the 2021 test window the price is \$250. Every input the network now sees is outside the convex hull of its training data. It is being asked to extrapolate, and the figure above shows what happens: the policy that looked great in-sample produces a near-random or negative-Sharpe result out-of-sample, not because the strategy was wrong but because the *inputs themselves* left the training domain.

The cure is stationarity. A time series is (weakly) stationary if its mean and variance do not change over time. Raw prices are emphatically non-stationary — they trend, they have a unit root, $E[p_t]$ drifts. Returns are *approximately* stationary: $r_t = \log(p_t / p_{t-1})$ has a roughly constant mean near zero and a variance that, while it clusters (volatility regimes), stays in a bounded band that recurs across years. The return on a \$100 stock and a \$250 stock can both be +1.2% on a given day; the return distribution in 2015 and 2021 overlaps heavily, while the price distributions do not overlap at all. So the network trained on returns sees test inputs from the *same* distribution it trained on, and its predictions remain meaningful.

This is why the state is built from returns and z-scored indicators, never raw price levels. Concretely:

```python
import numpy as np
import pandas as pd

def build_features(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """Turn raw OHLCV into a stationary feature frame. df indexed by date."""
    out = pd.DataFrame(index=df.index)
    # Log returns: approximately stationary, the core signal.
    out["ret"] = np.log(df["close"]).diff()

    # RSI(14): bounded in [0, 100], inherently stationary.
    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    out["rsi"] = 100 - 100 / (1 + rs)

    # Realized volatility over the window: regime feature, scaled.
    out["realized_vol"] = out["ret"].rolling(window).std()

    # Z-score the MA gap so it is unit-scale and stationary.
    ma_fast = df["close"].rolling(10).mean()
    ma_slow = df["close"].rolling(50).mean()
    gap = (ma_fast - ma_slow) / df["close"]
    out["ma_gap_z"] = (gap - gap.rolling(252).mean()) / (gap.rolling(252).std() + 1e-9)

    return out.dropna()
```

Two engineering notes that bite people. First, the z-scoring must use only past data — a rolling mean and standard deviation computed up to and including the current bar, never the full-sample mean. Using the full-sample statistics is a subtle look-ahead leak: at training time you would normalize day 10 using the standard deviation of days 1 through 1000, which includes the future. Always use causal (backward-looking) rolling windows. Second, the position has to be *appended* to this feature vector inside the environment, not precomputed, because the position depends on the agent's own past actions and is not a property of the price series. The state the agent sees is the concatenation: `[ten returns, rsi, realized_vol, ma_gap_z, position]`.

The table below summarizes the state-design decisions and why each goes the way it does.

| State component | Stationary? | Include? | Why |
|-----------------|-------------|----------|-----|
| Raw price level | No | Never | Drifts out of training range at test; net extrapolates |
| Log returns (N-day) | Approximately | Always | Bounded, recurring distribution across years |
| RSI / bounded oscillators | Yes (bounded) | Yes | Inherently in [0,100]; compresses momentum |
| Z-scored MA gap | Yes (causal z) | Yes | Trend signal at unit scale, no level dependence |
| Realized volatility | Approximately | Yes | Regime feature; helps generalize across vol regimes |
| Current position | Yes (discrete) | Always | Required for Markov property and cost reasoning |
| VIX / credit spread | Approximately | Optional | Explicit regime conditioning, cross-asset context |

There is a third, optional, layer of state engineering that often pays off: **regime features** that are external to the asset itself. The VIX (the market's implied-volatility index) tells the agent whether it is in a calm or panicked regime; a credit spread, a term-structure slope, or a realized-correlation measure can do similar work. These are valuable precisely because they help the agent generalize across regimes — they give it an explicit input that says "we are in a different world now," which is exactly the information a position-only state lacks. If you want the macro grounding for why these indicators carry signal, the [macro-correlations work on how indicators move asset prices](/blog/trading/macro-correlations/the-macro-correlation-playbook-capstone) is the right companion read.

## 4. Reward design: the part that decides whether you make money

If state engineering decides whether the agent can *generalize*, reward design decides whether it learns to do anything *useful*. This is where my over-trading bot died, and it is where most trading-RL projects quietly fail. The reward is the only thing the agent actually optimizes; everything else is in service of it. Get it wrong and the agent will exploit your mistake with perfect, soulless efficiency.

![A before-and-after comparison showing a raw PnL reward that spikes and crashes at drawdowns versus a Sharpe-scaled reward that stays smooth and bounds drawdown](/imgs/blogs/rl-for-algorithmic-trading-foundations-2.png)

The obvious reward is **raw profit-and-loss**: the change in account equity from one bar to the next. It is intuitive — make money, get rewarded — and it is a trap, for three reasons. First, it is extremely high variance. Daily PnL is dominated by market noise; the signal-to-noise ratio of "did this specific action help" against the backdrop of the market's daily swing is tiny, and high-variance rewards make the gradient estimates the agent learns from enormously noisy, slowing or preventing convergence. Second, it is non-stationary in scale: a \$1,000 move means something different when the account is \$100,000 versus \$300,000, so the same skillful decision gets rewarded inconsistently across the episode. Third, and most insidiously, raw PnL rewards risk-taking. An agent maximizing expected PnL with no penalty for variance will lever up and concentrate, because higher exposure has higher expected return — right up until a drawdown vaporizes the account. The left panel of the figure shows this: the reward spikes on good days and craters on drawdowns, and an agent chasing those spikes learns a fragile, over-levered policy with a Sharpe near 0.2.

The fix is to reward *risk-adjusted* return, and the canonical risk-adjusted measure is the **Sharpe ratio**: mean return divided by the standard deviation of returns. As a per-step shaped reward we use a running, volatility-scaled return:

$$
R_t = \frac{r^{\text{port}}_t - r_f}{\sigma_t + \epsilon}
$$

where $r^{\text{port}}_t$ is the portfolio's return this bar (position times asset return, minus costs), $r_f$ is a per-bar risk-free rate (often dropped at daily frequency), $\sigma_t$ is a rolling standard deviation of portfolio returns over a trailing window, and $\epsilon$ is a small constant to avoid dividing by zero. The intuition is direct: a gain earned in a calm market (low $\sigma_t$) scores higher than the same gain earned by gambling in a volatile one (high $\sigma_t$). The agent is rewarded for *return per unit of risk*, which is what an investor actually wants. The right panel of the figure shows the consequence: the reward is smooth, drawdowns are bounded (the agent learns to cut risk when volatility rises), and the achieved Sharpe climbs above 1.0.

There is a subtle theoretical point about why the Sharpe-shaped reward stabilizes *learning*, not just the strategy. The variance of the policy gradient estimate scales with the variance of the reward. By dividing returns by their rolling volatility, you are roughly variance-normalizing the reward signal across regimes, which keeps the gradient magnitudes comparable whether the agent is trading a sleepy summer or a chaotic crash. That is a genuine optimization benefit, separate from the economic benefit of caring about risk.

Two refinements worth knowing. The Sharpe ratio penalizes upside volatility as much as downside, which is arguably perverse — you do not mind your returns being volatile when they are volatile *upward*. The **Sortino ratio** divides by downside deviation only,

$$
\text{Sortino} = \frac{\bar{r} - r_f}{\sigma_{\text{down}}}, \quad \sigma_{\text{down}}^2 = \frac{1}{T}\sum_{t} \min(r_t - r_f, 0)^2
$$

rewarding the agent for asymmetry — a strategy that lurches upward and grinds down gently scores better under Sortino than under Sharpe, which is usually what you want. The **Calmar ratio** divides annualized return by maximum drawdown, directly targeting the worst-case loss an investor would actually feel and would actually redeem capital over. Which you pick is a statement about what risk you care about; for a first system, the rolling-Sharpe reward is a robust default, but if your investors care about tail risk — and they always do — shaping toward Sortino or adding an explicit drawdown penalty term to the reward is the principled next step.

There is one more reward-design subtlety that trips people: **dense versus sparse rewards**. A dense reward pays out every bar (our per-step Sharpe contribution); a sparse reward pays only at the end of an episode (final portfolio Sharpe). Sparse rewards are *less biased* — they optimize exactly the metric you care about, with no shaping that could distort behavior — but they are murderously hard to learn from, because the agent gets one scalar of feedback per episode and has to solve the credit-assignment problem of figuring out which of two hundred decisions earned it. Dense rewards are easier to learn from but risk *reward shaping bias*: if the per-step proxy is not perfectly aligned with the true objective, the agent optimizes the proxy. The rolling-Sharpe per-step reward is a carefully chosen dense proxy whose sum approximates the episode Sharpe well enough to learn from while staying aligned. When in doubt, prefer dense and verify alignment by checking that maximizing the per-step reward in a backtest actually maximizes the end-of-episode metric you report.

#### Worked example: how a missing cost term breeds an over-trader

Concretize the over-trading failure with numbers. Suppose the asset has a true daily edge of zero — it is a random walk — and you set the reward to raw PnL with *no* transaction cost. The agent will, correctly, find that under a zero-cost random walk, frequent trading neither helps nor hurts expected PnL, so it is free to trade however the noise happens to correlate with its features on the training set. On the training window it will overfit to that noise and appear to make money: say it flips position 180 times over a 250-day window and posts a training Sharpe of 2.1. Now add a realistic cost of 5 basis points per trade. Each flip costs 0.05% of notional. 180 flips cost roughly 9% of the account, eaten in friction. The "edge" the agent found was smaller than 9%, so on any honest accounting it loses money — and out-of-sample, where the noise it memorized does not repeat, it loses the friction *and* the phantom edge, posting a Sharpe of -0.4. The single change that fixes this is putting the cost into the environment's reward so the agent sees, in training, that each trade is expensive. After adding the cost term, the same agent settles to about 22 trades over the window, a training Sharpe of 1.3, and a far more durable out-of-sample Sharpe near 0.6. The lesson: **costs are not a post-hoc accounting adjustment, they are part of the reward**, and an agent that does not feel them in training will trade as if they do not exist.

#### Worked example: reward hacking into permanent cash

The opposite failure is just as instructive. Make the cost term too punishing — say 50 basis points per trade — or shape the reward so that drawdowns are penalized far more heavily than gains are rewarded (a common over-correction). The agent discovers a degenerate optimum: never trade at all. Sitting in cash earns zero reward but incurs zero cost and zero drawdown, and if your reward function makes any active position look like negative expected value, cash dominates. You train for a million steps, the loss converges beautifully, and the agent has learned to do nothing — position 0 forever, Sharpe exactly 0. This is **reward hacking**: the agent found a literal optimum of your reward function that has nothing to do with your intent. It is not the agent being dumb; it is the agent being too smart for your reward spec. The tell is a flat equity curve and a position that never moves. The fixes are to add an exploration incentive (an entropy bonus, which we cover with SAC), to ensure your cost is realistic rather than punitive, and to check that holding a profitable position actually out-scores cash in your reward arithmetic.

## 5. Implementing the Gymnasium environment

Now we make it concrete. Gymnasium (the maintained successor to OpenAI Gym) defines the standard interface every RL library expects: a `reset` that starts an episode and returns the first observation, and a `step(action)` that advances one tick and returns `(observation, reward, terminated, truncated, info)`. If your trading environment honors that contract, you can drop in Stable-Baselines3's DQN, PPO, or SAC with no glue code. Let us build a `StockTradingEnv` that does.

![A pipeline diagram tracing one environment step from raw OHLCV through feature extraction, state vector, action masking, order execution, transaction cost, to reward and next state](/imgs/blogs/rl-for-algorithmic-trading-foundations-3.png)

The figure traces a single `step`: raw OHLCV at the current bar becomes features, the features plus the current position become the state vector, the action is masked to legal moves, the order executes and incurs cost, and a reward is computed from *realized* prices before the cursor advances to the next bar. The discipline the figure encodes — and the one trap that ruins more trading environments than any other — is in that last clause: **the reward at step `t` must be computable using only information available at time `t`**. If your `step` uses tomorrow's price to decide today's reward, you have built a time machine, your backtest will look spectacular, and it will be worthless. We will guard against it explicitly.

```python
import gymnasium as gym
from gymnasium import spaces
import numpy as np

class StockTradingEnv(gym.Env):
    """Single-asset daily trading env with a Sharpe-shaped, cost-aware, leak-free reward."""

    metadata = {"render_modes": ["human"]}

    def __init__(self, features: np.ndarray, returns: np.ndarray,
                 window: int = 10, cost_bps: float = 5.0,
                 vol_window: int = 20):
        super().__init__()
        # features[t]: stationary feature row at bar t (returns, rsi, vol, ...)
        # returns[t]: the realized log-return EARNED from bar t to bar t+1
        self.features = features.astype(np.float32)
        self.returns = returns.astype(np.float32)
        self.window = window
        self.cost = cost_bps / 1e4          # 5 bps -> 0.0005
        self.vol_window = vol_window
        self.n_steps = len(returns) - 1     # last bar has no next return

        n_feat = features.shape[1]
        # Observation = feature row + scalar current position.
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(n_feat + 1,), dtype=np.float32)
        # Discrete action: 0 = short(-1), 1 = flat(0), 2 = long(+1).
        self.action_space = spaces.Discrete(3)
        self._action_to_pos = {0: -1.0, 1: 0.0, 2: 1.0}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t = self.window          # start after the feature warmup
        self.position = 0.0
        self.port_returns = []        # rolling window of realized port returns
        return self._obs(), {}

    def _obs(self):
        row = self.features[self.t]
        return np.concatenate([row, [self.position]]).astype(np.float32)

    def step(self, action):
        new_pos = self._action_to_pos[int(action)]
        # Transaction cost charged on the CHANGE in position only.
        traded = abs(new_pos - self.position)
        cost = traded * self.cost

        # Reward uses returns[t]: the move from bar t to t+1, realized,
        # decided BEFORE we advance. No look-ahead: the action set the
        # position for the bar whose return we are now collecting.
        bar_return = new_pos * self.returns[self.t] - cost
        self.port_returns.append(bar_return)

        # Sharpe-shaped reward: scale by trailing volatility of port returns.
        recent = self.port_returns[-self.vol_window:]
        vol = np.std(recent) if len(recent) > 1 else 1.0
        reward = bar_return / (vol + 1e-6)

        self.position = new_pos
        self.t += 1
        terminated = self.t >= self.n_steps
        truncated = False
        info = {"bar_return": bar_return, "position": new_pos, "cost": cost}
        return self._obs(), float(reward), terminated, truncated, info

    def render(self):
        print(f"t={self.t} pos={self.position:+.0f} "
              f"last_ret={self.port_returns[-1]:+.4f}")
```

Walk through the leak-prevention logic, because it is the heart of the matter. The array `returns[t]` is defined as the return earned by holding from bar `t` to bar `t+1`. In `step`, the agent's action sets `new_pos` *for* bar `t`, and then we collect `returns[t]` — the move that this position rides into bar `t+1`. The position was chosen using the observation at bar `t`, which contains only features up to and including bar `t`. So the decision precedes the outcome; there is no point at which the agent's action depends on the return it is about to earn. This ordering — decide, then realize — is the entire game. The common bug is to compute the position and the return at the same index and accidentally let the feature for bar `t` include the close of bar `t`'s own return computation; keep the features strictly causal (section 3) and the indexing as above and you are safe.

A few more design points encoded in the code. Transaction cost is charged on the *change* in position, `abs(new_pos - self.position)` — flipping from long to short trades two units of notional and costs twice as much as opening from flat, which is economically correct and is exactly the signal that taught the agent to stop flipping. The `done` condition is simply reaching the end of the window; you can add an equity-floor early termination for realism. And the observation always carries the current position, so the agent can reason about its own state.

**Action masking** deserves a note. In a long-only account, "go short" is illegal. Rather than let the agent waste capacity learning that short is bad, you can mask illegal actions — set their Q-values to negative infinity before the argmax, or use a library like `sb3-contrib`'s `MaskablePPO`. For the three-action long/short/flat env above masking is unnecessary, but the moment you add real account constraints (no shorting, position limits, margin) masking keeps the agent inside the feasible set and speeds learning.

## 6. The DQN baseline: discrete actions, beating buy-and-hold

With a Gymnasium env in hand, the fastest path to a working agent is Deep Q-Networks. DQN learns an action-value function $Q(s, a)$ — the expected discounted future reward of taking action $a$ in state $s$ and acting optimally thereafter — and acts greedily with respect to it. It is the natural fit for our discrete buy/hold/sell action space, and it is off-policy, which means it can reuse old experience from a replay buffer and is therefore relatively sample-efficient.

![A layered stack diagram showing a DQN update flowing from replay buffer to minibatch sample to Q-network forward pass to Bellman target to MSE loss to optimizer step to target-network soft update](/imgs/blogs/rl-for-algorithmic-trading-foundations-4.png)

### The theory: Q-learning and the Bellman optimality equation

The foundation is the Bellman optimality equation. Define the optimal action-value function $Q^*(s, a)$ as the maximum expected discounted return achievable from $(s, a)$. It satisfies a self-consistency condition:

$$
Q^*(s, a) = \mathbb{E}_{s'}\left[ R(s, a, s') + \gamma \max_{a'} Q^*(s', a') \right]
$$

In words: the value of taking action $a$ now equals the immediate reward plus the discounted value of acting optimally from the next state. This is a fixed-point equation, and the operator that maps a $Q$ function to the right-hand side — the Bellman optimality operator $\mathcal{T}$ — is a $\gamma$-contraction in the sup-norm. That contraction property is *why* Q-learning converges, and it is worth seeing the one-line reason. For any two value functions $Q_1$ and $Q_2$,

$$
\|\mathcal{T}Q_1 - \mathcal{T}Q_2\|_\infty \le \gamma \, \|Q_1 - Q_2\|_\infty
$$

because the only difference between $\mathcal{T}Q_1$ and $\mathcal{T}Q_2$ is the discounted $\max$ term, and the $\max$ operator is non-expansive while the $\gamma$ factor strictly shrinks any gap. By the Banach fixed-point theorem a $\gamma$-contraction with $\gamma < 1$ has a *unique* fixed point and repeated application converges to it geometrically — each sweep of value iteration cuts the error by at least a factor of $\gamma$. That is the mathematical bedrock under "Q-learning works." The catch, and it is a big one for trading, is that this guarantee is *tabular*: it assumes you can represent $Q$ exactly for every state. The moment you swap the table for a neural network the contraction can break, because the network's generalization couples the values of states that the Bellman operator treats independently, and the projection of $\mathcal{T}Q$ back onto the network's representable functions need not be a contraction at all. This is the heart of the "deadly triad" — function approximation, bootstrapping, and off-policy data combined — and it is why the guarantee weakens to "usually works with the right tricks." The replay buffer and target network are precisely those tricks, engineered to keep the approximate iteration close enough to the contraction that it converges in practice. For trading, where the data is both off-policy (from a replay buffer) and non-stationary (regime shifts), you are squarely in deadly-triad territory, which is one more reason to lean on the well-tuned Stable-Baselines3 implementation rather than rolling your own and to watch the loss curve like a hawk.

Q-learning turns the fixed-point equation into a sample-based update. After observing a transition $(s, a, r, s')$, it nudges $Q(s, a)$ toward the *TD target* $y = r + \gamma \max_{a'} Q(s', a')$. The difference $\delta = y - Q(s, a)$ is the **temporal-difference (TD) error** — the surprise, how much the observed outcome differed from the current estimate. Deep Q-learning fits a network $Q_\theta$ by minimizing the squared TD error over minibatches, and the stack diagram above shows the full update: sample a minibatch from the replay buffer, forward through the online network for $Q(s,a)$, build the target with a *separate target network* $Q_{\theta^-}$, take the MSE, step the optimizer, and softly update the target network toward the online one.

The two tricks that make it stable are exactly the ones that fight non-stationarity, which is precisely our problem in trading. **Experience replay** stores transitions in a buffer and trains on random minibatches, which breaks the temporal correlation between consecutive samples — without it, the network would train on a contiguous run of correlated bars and chase the local trend. **The target network** holds the bootstrap target fixed for a while, so the network is not chasing a target that moves every gradient step, which would otherwise diverge.

### The implementation

You can write DQN from scratch in PyTorch, and it is worth doing once to understand it, but for a baseline Stable-Baselines3 gives you a battle-tested implementation:

```python
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy

# train_features / train_returns from build_features on the TRAIN window only
train_env = Monitor(StockTradingEnv(train_features, train_returns,
                                    cost_bps=5.0))

model = DQN(
    policy="MlpPolicy",
    env=train_env,
    learning_rate=1e-4,
    buffer_size=100_000,        # replay buffer capacity
    learning_starts=2_000,      # collect random experience first
    batch_size=64,
    gamma=0.99,
    train_freq=4,
    target_update_interval=1_000,
    exploration_fraction=0.3,   # anneal epsilon over 30% of training
    exploration_final_eps=0.05,
    verbose=1,
    seed=42,
)
model.learn(total_timesteps=200_000)
model.save("dqn_trader")
```

If you want to see the mechanism rather than the API, here is the core update written by hand — the replay buffer, the TD target with a target network, and the gradient step:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import random

class QNet(nn.Module):
    def __init__(self, obs_dim, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, n_actions),
        )
    def forward(self, x):
        return self.net(x)

def dqn_update(online, target, optimizer, batch, gamma=0.99):
    s, a, r, s2, done = batch                  # tensors
    q = online(s).gather(1, a.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        # Bellman target: r + gamma * max_a' Q_target(s', a')
        q_next = target(s2).max(dim=1).values
        y = r + gamma * q_next * (1.0 - done)
    loss = F.mse_loss(q, y)                     # squared TD error
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(online.parameters(), 10.0)  # guard NaNs
    optimizer.step()
    return loss.item()

def soft_update(target, online, tau=0.005):
    for tp, op in zip(target.parameters(), online.parameters()):
        tp.data.copy_(tau * op.data + (1 - tau) * tp.data)
```

The `gather` picks out the Q-value of the action actually taken; the `with torch.no_grad()` block builds the target from the *frozen* target network so gradients do not flow through it; the `(1.0 - done)` zeroes the bootstrap at episode end (there is no future after the last bar); and the gradient clip is your first line of defense against the NaN explosions that extreme market moves cause. The soft update with $\tau = 0.005$ slowly tracks the online network — Stable-Baselines3 uses a hard periodic copy by default, but soft updates are smoother for noisy financial rewards.

#### Worked example: one TD update by hand

Make the abstract update concrete with a single numeric step. Suppose the agent is flat (position 0) and observes a state where its Q-network currently estimates the three action-values as $Q(s, \text{short}) = 0.10$, $Q(s, \text{flat}) = 0.05$, $Q(s, \text{long}) = 0.30$. It acts greedily and goes long. The next bar's realized return is +1.2%, and because it changed position from flat to long it pays 5 bps of cost, so the shaped reward (before volatility scaling, for simplicity) is $r = 0.012 - 0.0005 = 0.0115$. At the next state $s'$ the target network's best action-value is $\max_{a'} Q_{\theta^-}(s', a') = 0.40$. With $\gamma = 0.99$ the TD target is $y = 0.0115 + 0.99 \times 0.40 = 0.4075$. The TD error is $\delta = y - Q(s, \text{long}) = 0.4075 - 0.30 = 0.1075$ — a positive surprise, the long was better than the network thought. The MSE loss for this sample is $\delta^2 = 0.0116$, and the gradient step nudges $Q(s, \text{long})$ upward toward 0.4075 by a small fraction governed by the learning rate. Repeat this across millions of transitions and the Q-values converge toward the discounted future Sharpe contribution of each action in each state. Watching a few of these by hand, with real numbers from your own env, is the fastest way to catch a sign error or a reward that is off by a factor of a hundred.

### The transfer from CartPole

A genuinely useful sanity check before you trust any trading result: run the *exact same DQN configuration* on `CartPole-v1` first. CartPole is the "hello world" of RL — balance a pole on a cart, reward +1 per step it stays up, solved at an average return of 475 out of 500. If your DQN code reaches ~475 on CartPole within a few hundred thousand steps, the algorithm and your training loop are correct, and any failure on the trading env is a problem with your *environment or reward*, not your RL code. This separation of concerns has saved me days. The trading env is the variable under test; CartPole is the control.

### The result

On a daily backtest of a liquid large-cap equity with a 2015–2018 train window and a 2019 test window, a DQN agent with the Sharpe-shaped, cost-aware reward typically lands a test Sharpe in the range of 0.5–0.8 against a buy-and-hold Sharpe of roughly 0.4 over the same period, while taking materially less drawdown because it goes flat in stretches buy-and-hold rides down. The honest framing: it is a *modest* edge over buy-and-hold, not a money printer, and as we will see in section 8, even that number needs walk-forward validation before you believe it. But it clears the bar that matters for a baseline — it beats the passive benchmark on risk-adjusted return without the over-trading pathology, because the cost term is in the reward and the state is stationary.

## 7. Continuous actions with SAC: sizing the bet

Three discrete actions throw away information. A great setup and a marginal setup both map to "go long," yet conviction should drive *size*. The natural upgrade is a continuous action $a \in [-1, +1]$ interpreted as the fraction of capital to allocate (negative for short). For continuous control, the strongest off-the-shelf algorithm is **Soft Actor-Critic (SAC)**.

SAC is an off-policy actor-critic method — it learns both a policy (the actor) and a value function (the critic), and reuses replay data like DQN does, so it is sample-efficient. Its defining feature is **maximum-entropy RL**: instead of maximizing expected reward alone, it maximizes expected reward *plus* the entropy of the policy:

$$
J(\pi) = \sum_t \mathbb{E}\left[ R(s_t, a_t) + \alpha \, \mathcal{H}(\pi(\cdot \mid s_t)) \right]
$$

where $\mathcal{H}$ is the policy's entropy and $\alpha$ is a temperature trading off reward against randomness. This entropy term is not a gimmick — it is *automatic, calibrated exploration*. A policy that collapses to a single deterministic action has zero entropy and is penalized, so SAC keeps the policy as random as it can while still earning reward. In trading this matters twice over: it prevents the premature collapse into the always-cash degenerate policy (the entropy bonus makes "do nothing forever" expensive because it is zero-entropy and low-reward), and it keeps the agent probing alternative position sizes rather than locking onto whatever worked first.

The alternative for continuous control is **PPO** (Proximal Policy Optimization), an on-policy method. PPO is robust and simple but on-policy, meaning it throws away each batch of experience after one update and is therefore far less sample-efficient — a real cost when your "samples" are years of market history you cannot manufacture. For trading, where data is finite and precious, SAC's sample efficiency is usually decisive. (If you want the full PPO derivation and its clipped surrogate objective, the [proximal policy optimization post in this series](/blog/machine-learning/reinforcement-learning/proximal-policy-optimization-ppo) covers it; here we use SAC.) TD3 is a third option — deterministic, sometimes more stable, but without SAC's built-in exploration, so it can be more prone to the cash-collapse failure.

Swapping DQN for SAC in our setup requires changing the action space to a `Box` and the reward to operate on continuous position fractions:

```python
from gymnasium import spaces
from stable_baselines3 import SAC

# In the env __init__, replace the Discrete space:
#   self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,),
#                                  dtype=np.float32)
# In step(), the action IS the target position fraction:
#   new_pos = float(np.clip(action[0], -1.0, 1.0))
# the rest (cost on |new_pos - position|, Sharpe reward) is unchanged.

model = SAC(
    policy="MlpPolicy",
    env=train_env,
    learning_rate=3e-4,
    buffer_size=100_000,
    batch_size=256,
    gamma=0.99,
    tau=0.005,
    ent_coef="auto",         # learn the entropy temperature alpha
    train_freq=1,
    gradient_steps=1,
    verbose=1,
    seed=42,
)
model.learn(total_timesteps=300_000)
```

The `ent_coef="auto"` is the modern SAC default that *learns* the temperature $\alpha$ to hit a target entropy automatically, so you do not have to hand-tune the exploration-exploitation knob. Concretely, SAC frames the temperature as a constrained optimization — maximize reward subject to the policy's entropy staying above a target $\bar{\mathcal{H}}$ — and solves it by gradient descent on $\alpha$ alongside the actor and critic. When the policy gets too deterministic (entropy below target) $\alpha$ rises, pushing the actor back toward exploration; when it is exploring plenty, $\alpha$ falls and the agent exploits. For trading this auto-tuning is a genuine convenience because the right amount of exploration changes with the regime: in a choppy market the agent benefits from more position-size randomness to discover what works, while in a clean trend it should commit. Note `learning_rate=3e-4`, the famous "Adam default" that works astonishingly often for actor-critic methods.

The other reason SAC suits trading is its use of **two Q-networks** (the "twin critics" it shares with TD3) and the minimum of the two when forming targets. Q-learning has a well-known *overestimation bias*: the $\max$ over noisy action-value estimates systematically picks the ones that happen to be overestimated, inflating the target. In a market that bias is dangerous because it makes the agent overconfident about the value of taking a position, which leads to over-exposure — exactly the failure we are trying to avoid. Taking the minimum of two independently-trained critics is a cheap, effective antidote: it biases the estimate slightly downward, making the agent appropriately conservative about how good any position really is. The combination of entropy-driven exploration and twin-critic conservatism is why SAC tends to produce trading policies that are both well-explored and risk-aware.

#### Worked example: discrete flips versus continuous sizing

Put numbers on what continuous actions buy you. Take a six-month test stretch with a clear trend interrupted by a sharp two-week pullback. The discrete DQN agent can only be fully long, fully short, or flat. Through the pullback it does the sensible discrete thing — it goes flat — but it is a binary decision: one day fully exposed, the next fully out, and a couple of whipsaw flips around the turn cost it 15 basis points in friction and a missed re-entry. Its Sharpe over the stretch is 0.7. The SAC agent, sizing continuously, *reduces* exposure as realized volatility climbs into the pullback — say from 0.8 of capital down to 0.2 — rides the drawdown with a fraction of the risk, and scales back up as volatility subsides, never paying for a full flip. It trades smaller amounts more often but each adjustment is cheap, and because its position is volatility-responsive its return stream is steadier. Its Sharpe over the same stretch is 1.0. The improvement is not magic; it is the agent expressing conviction as a continuous quantity instead of a three-way switch, which is what a human discretionary trader does naturally and a three-action agent cannot.

The matrix below summarizes when to reach for each algorithm.

![A four-by-four matrix comparing DQN, PPO, SAC, and TD3 across action space, sample efficiency, stability, and best use](/imgs/blogs/rl-for-algorithmic-trading-foundations-5.png)

| Algorithm | Action space | On/off-policy | Sample efficiency | Stability | Best trading use |
|-----------|--------------|---------------|-------------------|-----------|------------------|
| DQN | Discrete | Off-policy | Medium | Medium | Buy/hold/sell signal baseline |
| PPO | Both | On-policy | Low | High | Robust baseline, easy to tune |
| SAC | Continuous | Off-policy | High | High | Position sizing, finite data |
| TD3 | Continuous | Off-policy | High | Medium | Sizing when exploration is less critical |
| A2C | Both | On-policy | Low | Medium | Fast prototyping, multi-env |

## 8. Evaluation methodology: not lying to yourself

This is the section that separates a research toy from something you would risk capital on, and it is the section most tutorials skip. The brutal truth of quantitative trading is that *it is trivially easy to produce a backtest that shows a Sharpe of 3 and means nothing*. Your evaluation methodology is the only thing standing between you and self-deception.

![A timeline of a training run progressing from a random policy at Sharpe minus 0.3 through exploration and convergence to validation Sharpe 1.1 and out-of-sample test Sharpe 0.6](/imgs/blogs/rl-for-algorithmic-trading-foundations-7.png)

The figure shows what an honest training run looks like, and the most important number on it is the *last* one: validation Sharpe 1.1 collapsing to out-of-sample Sharpe 0.6. That gap is not a failure — it is the truth. Every number to the left of the OOS test is contaminated to some degree by the optimization process, and the only one you can quote to an investor is the one earned on data the model and you never touched.

**The cardinal sin is look-ahead bias** — using information in your features, labels, or evaluation that would not have been available at decision time. We guarded against intra-step look-ahead in the environment (section 5) and against normalization look-ahead in the features (section 3), but it sneaks in at the evaluation level too. If you compute your z-scoring statistics on the full dataset including the test period, you have leaked. If you select your *hyperparameters* by looking at test-set performance, you have leaked — the test set is now part of your training, just laundered through your decisions. The defense is a strict temporal split.

**Time-series splits must respect time.** Random train/test splits, the default in supervised learning, are catastrophic here: shuffling lets the model train on 2020 and test on 2019, learning the future to predict the past. The split must be chronological: train on the earliest period, validate on the next, test on the latest, with no overlap and ideally a small gap (an "embargo") between them so that windowed features near the boundary do not bleed across.

**Walk-forward validation** is the gold standard. Instead of a single train/test split, you slide a window: train on years 1–3, test on year 4; then train on years 2–4, test on year 5; then 3–5 testing 6; and so on. You concatenate the out-of-sample test slices into one continuous out-of-sample equity curve. This does two things. It uses your data efficiently (every period except the first gets to be a test period eventually), and it directly measures the thing you care about — *how the strategy performs when periodically retrained and deployed forward into genuinely unseen data*, which is exactly how you would actually run it. A strategy that holds up under walk-forward across multiple regimes is worth attention; one that only shines on a single hand-picked test split is noise.

```python
def walk_forward_sharpe(prices, train_years=3, test_years=1, cost_bps=5.0):
    """Concatenate OOS test slices into one honest equity curve."""
    oos_returns = []
    splits = make_chronological_splits(prices, train_years, test_years)
    for train_slice, test_slice in splits:
        tr_feat, tr_ret = build_env_arrays(train_slice)   # causal features
        te_feat, te_ret = build_env_arrays(test_slice)
        model = SAC("MlpPolicy",
                    Monitor(StockTradingEnv(tr_feat, tr_ret, cost_bps=cost_bps)),
                    ent_coef="auto", seed=42)
        model.learn(total_timesteps=300_000)
        # Roll the trained policy through the untouched test slice.
        oos_returns.extend(rollout(model, te_feat, te_ret, cost_bps))
    oos = np.array(oos_returns)
    return np.mean(oos) / (np.std(oos) + 1e-9) * np.sqrt(252)  # annualized
```

**The multiple-testing trap, a.k.a. Sharpe inflation.** This one is statistical and it is the most insidious. Suppose you try 100 configurations — different feature sets, network sizes, reward shapings, random seeds — and report the best one's test Sharpe. Even if *every* configuration is pure noise with a true Sharpe of zero, the best of 100 random draws will have an impressive-looking Sharpe purely by chance. The expected maximum of 100 standard-normal draws is around 2.5 standard deviations; translate that into Sharpe units over a finite backtest and you can manufacture a "Sharpe 2" strategy out of thin air just by searching. The defenses: minimize the number of things you try, hold out a final test set you look at *exactly once*, use the deflated Sharpe ratio or a Bonferroni-style adjustment that accounts for the number of trials, and treat any result you only got after extensive search with deep suspicion. If you found it by trying a hundred things, it is probably one of the hundred noise draws.

A related discipline borrowed from the [analyst's-edge framework on turning information into an accountable view](/blog/trading/analyst-edge/the-analysts-edge-playbook-the-complete-view-forming-process): write down, *before* you run the test, what result would make you believe the strategy and what would make you abandon it. Pre-committing to the falsification criterion is the single best guard against fooling yourself after the fact.

## 9. Common failure modes and how to debug them

Trading RL fails in a small number of characteristic ways, and the symptom usually points straight at the cause. The decision tree below is the triage I run; let us walk each branch.

![A decision tree mapping a failing agent to three branches: flat Sharpe meaning reward hacking, NaN loss meaning unclipped rewards, and a validation-to-OOS gap meaning overfitting](/imgs/blogs/rl-for-algorithmic-trading-foundations-6.png)

**Symptom: Sharpe near zero, flat equity curve, position never moves.** This is reward hacking into cash (section 4). The agent found that doing nothing is the safe optimum of your reward. Diagnose by logging the action distribution — if it is ~100% "flat," that is it. Fixes, in order: confirm the transaction cost is realistic, not punitive; add or increase the entropy bonus (this is SAC's `ent_coef`, or `ent_coef` in PPO) so the agent is rewarded for keeping its options open; sanity-check that a profitable held position actually scores higher than cash in your reward arithmetic; and verify your action masking is not accidentally forbidding non-flat actions.

**Symptom: loss explodes to NaN, training dies.** Almost always an extreme market move. A single day with a -20% return, fed through a Sharpe reward where the rolling volatility happened to be tiny, produces a gigantic reward, a gigantic TD error, a gigantic gradient, and the network's weights blow up to NaN. The fixes are layered: clip the reward (e.g. to $\pm 3$ rolling standard deviations) so no single bar can dominate; clip the gradient norm (`clip_grad_norm_`, which we already put in the DQN update); add the $\epsilon$ floor to the volatility denominator (also already there); and consider clipping the raw returns fed into features so a data error — a bad tick, a missed split adjustment — cannot inject a 10,000% return. Most NaN deaths trace to either an unclipped reward or a data-quality bug in the price series; check the data first.

**Symptom: great on validation, poor out-of-sample.** Overfitting to the training regime — the classic. The agent learned the idiosyncrasies of the training window (this specific sequence of moves) rather than a generalizable strategy. Diagnose by the size of the val-to-OOS gap (the timeline figure's 1.1 → 0.6) and by testing across *different* regimes — train on a bull market, test on a bear, and watch it fall apart. Fixes: reduce the feature count (fewer dimensions, less to overfit); add regularization (weight decay, dropout in the network, or simply a smaller network); train across multiple regimes via walk-forward so the agent cannot memorize one; add the regime features (section 3) that let it *condition* on the regime instead of memorizing one; and shorten training — RL agents over-train on finite data, and an early-stopped agent often generalizes better than a converged one.

**Symptom: replay buffer poisoned by non-stationarity.** This is subtle and specific to financial RL. The off-policy buffer mixes transitions from across the training history — calm 2017 and violent 2020 in the same minibatch. If your state does not include a regime signal, the agent sees contradictory experience (the same state-action pair led to wildly different outcomes in different regimes) and learns a muddled average policy that fits neither. The buffer's i.i.d.-sampling assumption, which is *exactly* the property that makes replay work for stationary problems like Atari, is violated because the underlying environment distribution itself shifts. Fixes: include explicit regime features so the agent can tell the regimes apart (turning one non-stationary problem into several conditionally-stationary ones); consider a smaller, more recent buffer that emphasizes the current regime; or use prioritized replay carefully. This is the deepest reason trading RL is harder than game RL, and it is the practical face of the "trading is not i.i.d." insight from section 1.

For a much deeper treatment of diagnosing RL training pathologies — vanishing critics, exploding gradients, dead policies — the [debugging-training series in this blog](/blog/machine-learning/debugging-training/the-training-debugging-playbook) is the companion reference. The branches above are the trading-specific subset.

## 10. Case studies: what the literature actually shows

Theory and toy backtests are one thing; named, reproducible results are another. Four worth knowing.

**FinRL on the Dow Jones 30.** FinRL (Liu et al., 2020–2021) is an open-source library that packages Gymnasium-style trading environments and Stable-Baselines3 agents for exactly the kind of multi-asset portfolio allocation we have been building toward. Its widely-cited benchmark trains PPO, A2C, DDPG, and an ensemble on the 30 Dow Jones Industrial Average constituents and reports out-of-sample Sharpe ratios that, in several published runs, exceed the Dow Jones Industrial Average buy-and-hold benchmark over the test periods. The honest caveat the FinRL authors themselves emphasize: results are sensitive to the period, the cost assumptions, and the data, and reproducing the headline numbers requires the same careful walk-forward discipline we covered in section 8. FinRL is the best on-ramp if you want to go from this post's single-asset env to a real multi-asset portfolio — its code is the natural next step.

**DeepLOB for limit order books.** Zhang, Zohren, and Roberts ("DeepLOB: Deep Convolutional Neural Networks for Limit Order Books," 2019) operate at a completely different timescale — high-frequency, modeling the full limit order book rather than daily bars. Their architecture combines convolutional layers (to extract spatial structure from the order book's price-level grid) with an LSTM (to model temporal dynamics), and on the FI-2010 benchmark and on real LSE data it predicts short-horizon price movements with accuracy meaningfully above prior baselines. DeepLOB is not pure RL — it is a supervised price-movement predictor — but it is the canonical reference for *state representation* of microstructure data, and the natural state-engineering input to an RL execution agent operating at that frequency. The lesson it transfers to our setting: at high frequency the order book *is* the state, and getting that representation right matters more than the choice of RL algorithm.

**J.P. Morgan and RL for trade execution.** The most credible *real-system* application of RL in finance is not directional trading but *execution* — the problem of buying or selling a large order over time while minimizing market impact and slippage against a benchmark like VWAP. J.P. Morgan's "LOXM" execution system, reported around 2017–2019, applied reinforcement learning to optimally schedule the slices of a large parent order, learning from historical execution data to beat traditional rule-based execution algorithms on implementation shortfall. This is a telling signal about where RL actually earns its keep in finance: execution is a problem with a clear, dense, low-look-ahead reward (slippage versus benchmark, measured immediately), a genuinely sequential structure (each slice changes the remaining order and the market state), and a forgiving signal-to-noise ratio compared to directional prediction. Directional alpha is hard for everyone; execution optimization is where the sequential-decision framing of RL has a structural advantage.

**Deep Hedging.** Buehler, Gonon, Teichmann, and Wood ("Deep Hedging," 2019) frame the hedging of a derivatives portfolio under transaction costs and market frictions as an RL/deep-learning optimization, learning a hedging *policy* directly rather than relying on the closed-form Black-Scholes delta that breaks down once costs and frictions are real. They demonstrate that a learned policy can hedge a portfolio of options under realistic costs better than the classical delta-hedging benchmark. It is another instance of the same pattern: RL wins where the classical closed-form solution assumes away the very frictions (costs, discreteness, sequential rebalancing) that the learned policy can optimize through.

The thread across all four: RL's edge in finance is strongest in problems that are *genuinely sequential with frictions* (execution, hedging, rebalancing) and weaker in problems that are *essentially one-step prediction* (will the stock go up tomorrow). Directional single-asset trading, our running example, sits in the harder middle — possible, demonstrable, but a modest edge that demands the full evaluation rigor of section 8.

## 11. When RL beats classical quant — and when it doesn't

A decisive section, because the most expensive mistake is reaching for RL when something simpler would win.

**Use RL when** the problem is genuinely sequential and your action changes the future state in a way that compounds: trade execution (each slice moves the market and the remaining order), portfolio rebalancing under transaction costs (today's allocation constrains tomorrow's cheap moves), dynamic hedging under frictions, market making (inventory management is intrinsically sequential). RL also shines when the reward is dense and quick — execution gives you a slippage signal every slice — and when you have, or can simulate, a lot of interaction data. The structural advantage in all these is that the optimal action depends on a *trajectory* of consequences that a one-step predictor cannot see.

**Do not use RL when** a simpler tool solves the problem with less risk of self-deception. If you can write down a model of the market dynamics, classical stochastic control or dynamic programming gives you an optimal policy with guarantees and no sample-efficiency tax — for a known linear-quadratic problem, the closed-form solution beats any neural policy. If the problem is essentially one-step prediction (does the signal predict next-bar return), a well-regularized supervised model with a proper financial loss is simpler, more stable, and easier to risk-manage than an RL agent, and it sidesteps the entire reward-design minefield. If your action space is small and discrete and the environment is roughly stationary, tabular or shallow methods may suffice. And if you cannot generate enough interaction data and cannot simulate the market faithfully, RL's sample hunger will starve it — markets give you one realized history, and that is a tiny dataset by RL standards.

The deciding question is: *does the value of an action depend on a sequence of future consequences that a one-step model cannot capture, and do I have enough interaction data (real or simulated) to learn that?* Two yeses point to RL. Any no points to something simpler. The over-trading bot I opened with was, in retrospect, a problem where RL was defensible (sequential, cost-laden) but where I had skipped the evaluation rigor that would have caught the over-fitting before it cost real money. The framework matters less than the discipline.

## 12. Key takeaways

- **Trading is a non-i.i.d. sequential decision problem.** The agent's action changes its next state, so samples are correlated and the distribution shifts across regimes. This is why RL fits and why naive training (random splits, regime-blind replay buffers) fails.
- **State must be stationary.** Feed returns and z-scored indicators, never raw prices, or the network extrapolates outside its training domain at test time and the policy collapses. Always include the current position.
- **The reward is the only thing the agent optimizes.** Use a volatility-scaled (Sharpe-shaped) reward, not raw PnL, to penalize risk and stabilize the gradient. Put transaction costs *inside* the reward, or the agent will trade itself to death.
- **Two reward-hacking failure modes bracket the design.** Too little cost penalty breeds an over-trader; too much breeds a permanent-cash do-nothing. The entropy bonus in SAC guards the cash side; a realistic cost guards the churn side.
- **Prevent look-ahead at every layer**: causal feature normalization, decide-then-realize ordering in `step`, chronological splits, and hyperparameters never tuned on the test set.
- **DQN is the right discrete baseline; SAC is the right continuous upgrade.** DQN's replay buffer and target network exist precisely to fight the temporal correlation that defines trading data. SAC's entropy term gives calibrated exploration and sample efficiency on finite market history.
- **Walk-forward validation is non-negotiable.** A single train/test split lies; concatenated out-of-sample slices across regimes tell the truth. Expect validation Sharpe to exceed out-of-sample Sharpe — the gap is the honest measure of overfitting.
- **Beware Sharpe inflation from multiple testing.** The best of a hundred random configurations looks great by chance. Minimize trials, hold out a final test set you touch once, and adjust for the number of things you tried.
- **Sanity-check on CartPole first.** If your DQN reaches ~475 on CartPole, the algorithm is correct and any trading failure is in your environment or reward — a separation that saves days.
- **RL's structural edge in finance is in sequential, friction-laden problems** — execution, hedging, rebalancing — more than in one-step directional prediction. Reach for it when the value of an action depends on a trajectory of consequences, and reach for something simpler when it does not.

## Further reading

- Sutton, R. & Barto, A. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). The foundational text — chapters on MDPs, TD learning, and Q-learning underpin everything here.
- Mnih, V. et al. (2015). "Human-level control through deep reinforcement learning." *Nature.* The DQN paper; experience replay and target networks, the two tricks that make our baseline work.
- Haarnoja, T. et al. (2018). "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor." The SAC algorithm we use for continuous position sizing.
- Liu, X.-Y. et al. (2020–2021). "FinRL: A Deep Reinforcement Learning Library for Automated Stock Trading." The natural next step from this post's single-asset env to multi-asset portfolios.
- Zhang, Z., Zohren, S. & Roberts, S. (2019). "DeepLOB: Deep Convolutional Neural Networks for Limit Order Books." The reference for microstructure state representation.
- Buehler, H., Gonon, L., Teichmann, J. & Wood, B. (2019). "Deep Hedging." RL/deep learning for hedging under realistic frictions.
- López de Prado, M. (2018). *Advances in Financial Machine Learning.* The definitive treatment of look-ahead bias, walk-forward, and the deflated Sharpe ratio — required reading for the evaluation discipline in section 8.
- Within this series: the [Markov Decision Processes post](/blog/machine-learning/reinforcement-learning/markov-decision-processes) for the formalism this post builds on, the [Deep Q-Networks deep dive](/blog/machine-learning/reinforcement-learning/deep-q-networks-dqn) and [Soft Actor-Critic deep dive](/blog/machine-learning/reinforcement-learning/soft-actor-critic-sac) for the algorithms used here, the [reward hacking and Goodhart's law post](/blog/machine-learning/reinforcement-learning/reward-hacking-and-goodharts-law) for the failure modes in section 4, and the [Gym trading environments and backtesting post](/blog/machine-learning/reinforcement-learning/gym-trading-environments-and-backtesting) for the env-engineering companion.
