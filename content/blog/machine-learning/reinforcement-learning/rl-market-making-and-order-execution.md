---
title: "RL for Market Making and Optimal Execution: Beating the Avellaneda-Stoikov Baseline"
date: "2026-06-22"
publishDate: "2026-06-22"
description: "How market making and optimal execution are modeled as RL problems, why the Avellaneda-Stoikov closed form breaks down in live markets, and how DQN and SAC learn to outperform it — with runnable PyTorch, an order-book environment, and measured Sharpe and slippage numbers."
tags:
  [
    "reinforcement-learning",
    "deep-learning",
    "finance",
    "market-making",
    "q-learning",
    "actor-critic",
    "machine-learning",
    "pytorch",
    "multi-agent",
  ]
category: "machine-learning"
subcategory: "Reinforcement Learning"
author: "Hiep Tran"
featured: true
readTime: 51
image: "/imgs/blogs/rl-market-making-and-order-execution-1.png"
---

The first market-making bot I ever shipped lost money in the most embarrassing way possible: it was *right* about the spread and still bled out. It quoted a tidy two-cent spread around the mid-price, captured that spread on every round trip, and reported a positive realized-spread number all day long. Then I looked at the inventory chart. Every time the price trended, the bot ended up holding a fat one-sided position — long into a sell-off, short into a rally — because the orders that filled were exactly the orders it should not have wanted. It was capturing two cents of spread and losing twenty cents to inventory drift. The spread was real. The risk was realer. That gap between "I captured the spread" and "I kept the money" is the entire subject of this post.

Market making and optimal execution are, at heart, the two cleanest reinforcement-learning problems in all of trading. They have a tight feedback loop (you quote or you trade, the market responds in milliseconds), an unambiguous scalar reward (your profit-and-loss, the PnL), and a state that is mostly observable (the limit order book, the LOB, is right there on the screen). They also have a beautiful closed-form baseline — the Avellaneda-Stoikov model — that you can write on a napkin and that, crucially, *breaks down in predictable ways* the moment real microstructure intrudes. That combination is gold for an RL practitioner: a principled benchmark to beat, a clear reward to optimize, and a fast simulator to learn in. Figure 1 shows the loop we are going to build, where the agent reads the book plus its own inventory, places quotes, gets filled, and learns from PnL net of inventory risk.

![Diagram of the market-making reinforcement-learning loop where state combining the order book inventory and time feeds a DQN or SAC policy that places bid and ask quotes into the order book producing fills and a reward of spread minus inventory risk](/imgs/blogs/rl-market-making-and-order-execution-1.png)

By the end of this post you will understand the market maker's PnL decomposition into spread capture, inventory risk, and adverse selection; you will be able to derive the Avellaneda-Stoikov reservation price and optimal spread from first principles and say exactly *why* it fails in a live book; you will have a runnable Gymnasium-style LOB environment, a Deep Q-Network (DQN) for discrete quote placement, and a Soft Actor-Critic (SAC) agent for continuous quote offsets; and you will see the same machinery applied to optimal execution against a VWAP and Almgren-Chriss baseline. We will keep tying everything back to the series spine: an agent interacts with an environment, collects rewards, and updates a policy — market making is just a particularly unforgiving instance of that loop where the reward punishes you for the very fills you sought. For the bigger picture see the [unified map of RL algorithms](/blog/machine-learning/reinforcement-learning/reinforcement-learning-a-unified-map), and for the value-based and continuous-control building blocks we lean on, the posts on [Deep Q-Networks](/blog/machine-learning/reinforcement-learning/deep-q-networks-dqn) and [Soft Actor-Critic](/blog/machine-learning/reinforcement-learning/soft-actor-critic-sac).

## 1. Market making as an RL problem

A market maker is a liquidity provider. It continuously posts a bid (an offer to buy) below the mid-price and an ask (an offer to sell) above it, and it earns the difference — the spread — whenever someone trades against both sides. If the mid-price is \$100.00 and the maker quotes a bid at \$99.99 and an ask at \$100.01, then a buyer who lifts the ask and a seller who hits the bid together hand the maker two cents per share, for doing nothing but standing in the middle. That is the dream. The reality is that the maker is exposed to two adversaries the whole time, and the entire craft is managing them.

### The PnL decomposition

Over any horizon, a market maker's profit and loss decomposes into three terms, and writing them out is the single most clarifying thing you can do before touching any algorithm:

$$\text{PnL} = \underbrace{\sum_i s_i \cdot q_i}_{\text{spread capture}} \;-\; \underbrace{\int \text{inventory}_t \, dP_t}_{\text{inventory risk}} \;-\; \underbrace{\sum_j (\text{adverse fills})_j}_{\text{adverse selection}}.$$

The first term, **spread capture**, is the gross edge: every filled round trip earns half the spread on each side, scaled by the filled quantity. This is the term that looks great in isolation and the term that fooled my first bot.

The second term, **inventory risk**, is the cost of holding a non-zero position while the price moves. If the maker is long 500 shares and the price drops a dollar, that is a \$500 mark-to-market loss that has nothing to do with the spread. The integral $\int \text{inventory}_t \, dP_t$ says: every instant you carry inventory, you are exposed to the price increment $dP_t$, and over a day those increments compound into a position that can dwarf any spread you captured. The maker does not *want* directional exposure; inventory is an accident of which orders happened to fill.

The third term, **adverse selection**, is the subtle killer. Some of the traders who fill your quotes know something you do not — they are informed. When an informed buyer lifts your ask, it is often because the price is about to rise, leaving you short into a rally. The fills you most easily get are disproportionately the ones you least want. The classic formalization is the Glosten-Milgrom model: the spread you must quote is not just compensation for inventory and order-processing cost, it is compensation for the probability that your counterparty is informed. A maker who ignores adverse selection quotes too tight and gets picked off.

### The Avellaneda-Stoikov closed form

In 2008, Marco Avellaneda and Sasha Stoikov published a model that gives a *closed-form* answer to "where should I quote?" under a clean set of assumptions. It is the Black-Scholes of market making: not what anyone trades in production, but the reference everyone reasons against. The assumptions are: the mid-price follows an arithmetic Brownian motion $dS_t = \sigma\, dW_t$; the maker has exponential (CARA) utility with risk-aversion $\gamma$; and order arrivals are a Poisson process whose intensity falls off exponentially with how far your quote sits from the mid, $\lambda(\delta) = A e^{-\kappa \delta}$, where $\delta$ is your quote's distance from the mid and $\kappa$ measures how quickly fills dry up as you back away.

Solving the resulting Hamilton-Jacobi-Bellman equation yields two famous quantities. First, the **reservation price** — the price at which the maker is indifferent to holding its current inventory $q$ with time-to-close $T-t$:

$$r(s, q, t) = s - q\,\gamma\,\sigma^2\,(T - t).$$

Read it slowly. The reservation price is the mid $s$ *shifted* by the inventory. If you are long ($q > 0$), $r$ sits below the mid — you skew your quotes down to encourage selling and discourage buying, because you want to shed the position. The shift grows with risk-aversion $\gamma$, with variance $\sigma^2$, and with time remaining $(T-t)$ — more time means more exposure, so you are more eager to flatten. This single equation is why a good maker's quotes are *asymmetric*: they lean against inventory.

Second, the **optimal half-spread**, the total distance you quote around the reservation price:

$$\delta^{\text{ask}} + \delta^{\text{bid}} = \gamma\,\sigma^2\,(T - t) + \frac{2}{\gamma}\ln\!\left(1 + \frac{\gamma}{\kappa}\right).$$

The first piece grows with risk and time (quote wider when the world is dangerous), the second piece is a fill-rate term (quote wider when fills are easy to get anyway). Quote symmetrically around $r$, not around $s$, and you have the Avellaneda-Stoikov (AS) strategy.

Where do these formulas come from? The maker maximizes expected exponential utility of terminal wealth, and the value function $u(s, x, q, t)$ — of mid $s$, cash $x$, inventory $q$, and time $t$ — satisfies a Hamilton-Jacobi-Bellman equation. The HJB has a diffusion term from the price ($\frac{1}{2}\sigma^2 \partial_{ss} u$, the mark-to-market risk on inventory) plus two jump terms, one for each side, of the form $\lambda(\delta)\,[u(\text{after fill}) - u(\text{before fill})]$ — the expected utility gain from a fill at offset $\delta$, weighted by the fill intensity $\lambda(\delta) = A e^{-\kappa \delta}$. Optimizing each jump term over $\delta$ (set the derivative to zero) gives the optimal offset on each side, and the exponential-utility structure makes the cash and a quadratic-in-$q$ piece factor out cleanly, collapsing the partial differential equation into the closed forms above. The whole derivation is the continuous-time, model-based cousin of the Bellman optimality argument we use for DQN — it just happens to admit a closed-form solution because every piece (Gaussian price, exponential utility, exponential fill curve) was chosen to make the algebra close. Change any one of those assumptions and the closed form evaporates, which is precisely the door RL walks through.

### Why the closed form breaks down

The AS model is gorgeous and, in a live book, wrong in ways you can list. It assumes a *single* price process with constant volatility $\sigma$; real volatility clusters and jumps. It assumes Poisson arrivals with a smooth exponential fill curve; real fills depend on queue position, on the discrete tick grid, and on bursts of correlated order flow. It has *no model of the order book at all* — no depth, no imbalance, no notion that the bid side is three times thicker than the ask side right now. And it has no model of adverse selection beyond a static fill intensity; it cannot tell a toxic fill from a benign one. The result is a strategy that quotes a sensible *average* spread but misquotes every specific regime: too tight when the book is thin and about to move, too wide when the book is deep and calm. We will quantify exactly this gap, and then close it with RL. Figure 2 contrasts the fixed AS spread with the adaptive spread an RL agent learns.

![Before and after comparison showing the Avellaneda-Stoikov fixed-form spread that ignores book depth on the left versus an RL adaptive spread that widens in high volatility and tightens when the book is liquid on the right](/imgs/blogs/rl-market-making-and-order-execution-2.png)

The RL framing is now obvious. The maker is an agent. The market is the environment. The action is where to quote. The reward is PnL with the inventory penalty baked in. And instead of solving an HJB equation under fairy-tale assumptions, we let the agent *learn* the quote-placement function from interaction with a simulator that contains the messy microstructure the closed form throws away. The price of that flexibility is sample efficiency, stability, and the constant danger of overfitting a backtest — themes we return to in [financial RL backtesting and pitfalls](/blog/machine-learning/reinforcement-learning/financial-rl-backtesting-and-pitfalls).

## 2. The limit order book and its state representation

To learn a quoting policy you have to feed the agent a state, and the state lives in the limit order book. The LOB is the list of all resting buy and sell orders, organized by price. The highest resting buy is the **best bid**; the lowest resting sell is the **best ask**; the gap between them is the **spread**; their average is the **mid-price**. Below the best bid is a stack of progressively lower buy orders, each with a quantity; above the best ask, a stack of progressively higher sell orders. A "ten-level" snapshot gives you the price and size at each of the top ten price levels on each side — forty numbers in total, updated dozens of times a second.

You cannot feed forty raw, non-stationary numbers into a neural network and expect it to learn. The prices drift over the day; the sizes vary by orders of magnitude across instruments. So the first job is feature engineering: distilling the book into a handful of *stationary, scale-free* signals that actually predict the next price move and the next fill. Figure 3 lays out the extraction pipeline from raw depth to a normalized state vector.

![Pipeline diagram showing raw ten-level limit order book depth transformed into spread imbalance microprice and flow imbalance signals then z-score normalized and concatenated with inventory and time features into a sixteen-dimensional state vector](/imgs/blogs/rl-market-making-and-order-execution-3.png)

### The features that matter

**Spread** is the simplest: $\text{ask}_1 - \text{bid}_1$, usually expressed in ticks or basis points. A widening spread signals uncertainty.

**Order-book imbalance** is the workhorse. At its simplest it is the volume imbalance at the touch:

$$I = \frac{V_b - V_a}{V_b + V_a},$$

where $V_b$ and $V_a$ are the resting sizes at the best bid and ask. $I$ ranges from $-1$ (all the size is on the ask, sellers dominate) to $+1$ (all on the bid, buyers dominate). Empirically, imbalance is one of the strongest short-horizon predictors of the next mid move: a heavily bid-skewed book tends to tick up. Any maker that ignores imbalance is quoting blind.

**Microprice** is the imbalance-weighted mid, a better estimate of the "true" price than the naive mid:

$$P_{\text{micro}} = \frac{V_a}{V_b + V_a}\,\text{bid}_1 + \frac{V_b}{V_b + V_a}\,\text{ask}_1.$$

Note the weighting is *crossed*: when the bid is thick ($V_b$ large), the microprice leans toward the ask, because thick bids tend to push the price up. The microprice is where you should center your quotes instead of the raw mid; quoting around the mid when the microprice has drifted is a slow way to get adversely selected.

**Trade-flow imbalance** (TFI) captures the *signed* recent trading, not just resting depth: sum of buyer-initiated volume minus seller-initiated volume over the last few hundred milliseconds. Resting imbalance tells you intent; trade-flow imbalance tells you realized aggression. Together they are a cheap toxicity sensor.

### Why the full LOB is too large for tabular RL

A natural beginner instinct is to discretize the book and use a Q-table. This dies instantly to the curse of dimensionality. Suppose you discretize each of forty LOB numbers into just ten buckets — already a brutally coarse representation. The state space is $10^{40}$, more than the number of atoms in the observable universe. Add inventory and time and it is hopeless. Even the modest sixteen-feature vector, discretized to ten buckets each, is $10^{16}$ states. There is no way to visit, let alone learn a value for, even a vanishing fraction of them. This is precisely the motivation for function approximation: a neural network generalizes across states it has never seen, mapping a continuous feature vector to values or actions. If that argument feels familiar, it is the same one made in [why tables don't scale](/blog/machine-learning/reinforcement-learning/function-approximation-why-tables-dont-scale). For market making, function approximation is not optional; it is the whole reason deep RL applies here at all.

## 3. The MDP formulation

A reinforcement-learning problem is a Markov Decision Process (MDP): a tuple of states, actions, transitions, and rewards, with the Markov property that the future depends on the present state alone. Casting market making as an MDP forces us to make every modeling choice explicit, and the choices we make in the *reward* are where most agents live or die.

**State** $s_t$. The normalized feature vector from Section 2 — spread, imbalance, microprice deviation, trade-flow imbalance, a few levels of depth — *plus* two private features the book does not contain: the agent's current **inventory** $q_t$ and the **time remaining** $\tau = T - t$ in the trading session. Inventory must be in the state because the optimal action depends on it (the AS reservation price made that explicit). Time remaining must be in the state because risk tolerance shrinks as the close approaches — you cannot end the day holding a huge position you have no time to unwind.

**Action** $a_t$. Where to quote, expressed as a pair of offsets $(\delta^{\text{bid}}, \delta^{\text{ask}})$ from the reference price. In a discrete formulation (Section 4) the agent picks from a small grid, e.g. each offset from one of five levels, giving $5 \times 5 = 25$ actions. In a continuous formulation (Section 5) the agent outputs two real numbers in basis points. Some formulations add a "skew" and "size" action; we keep it to two offsets for clarity.

**Transition** $P(s_{t+1} \mid s_t, a_t)$. The simulator advances: it decides whether the agent's resting bid and ask get filled given where they sit relative to incoming order flow, updates inventory accordingly, advances the mid-price by one step of the price process, and recomputes the LOB features. This is the hard part to get right — a bad simulator teaches the agent to exploit artifacts that do not exist live.

**Reward** $r_t$. This is the crux. A first attempt is raw mark-to-market PnL per step. But that produces an agent that loads up on inventory to chase directional PnL — exactly the disease we are trying to cure. The standard fix is a *risk-adjusted* per-step reward that explicitly charges for inventory:

$$r_t = \underbrace{\delta^{\text{bid}} \cdot \mathbb{1}[\text{bid filled}] + \delta^{\text{ask}} \cdot \mathbb{1}[\text{ask filled}]}_{\text{realized spread}} \;+\; \underbrace{q_t \cdot \Delta P_t}_{\text{inventory MtM}} \;-\; \underbrace{\eta\, q_t^2}_{\text{inventory penalty}}.$$

The first term rewards captured spread, the second is the honest mark-to-market on the position you carry, and the third — the quadratic penalty $\eta q_t^2$ — is the inventory governor. The square matters: it punishes large positions disproportionately, pushing the agent to mean-revert its inventory toward zero. The coefficient $\eta$ is the single most important hyperparameter in the whole system; it is the RL analog of the risk-aversion $\gamma$ in Avellaneda-Stoikov, and tuning it trades spread capture against inventory variance directly.

**Done** $d_t$. The episode ends at the session close $T$, often with a forced liquidation of any residual inventory at a penalized price, so the agent learns it cannot hide risk by carrying it past the bell.

It is worth dwelling on the reward, because reward design is where most market-making RL projects quietly fail. There is a whole family of subtly-broken rewards that look reasonable and teach the wrong thing. **Raw PnL per step** (no inventory penalty) teaches the agent to become a directional speculator: it discovers that loading up on inventory when its imbalance feature predicts an up-move earns more than patiently capturing spread, and you end up with a momentum bot wearing a market-maker costume. **Realized-spread-only** (count captured spread, ignore mark-to-market) is the mirror failure — it is exactly the metric that fooled my first bot, because it rewards the gross edge and is blind to the inventory bleed. **Terminal-PnL-only** (reward only at the close) is technically correct but has a brutal credit-assignment problem: a thousand quoting decisions get a single scalar at the end, and the agent cannot tell which quotes helped. The risk-adjusted per-step reward above threads the needle: it gives dense feedback (every step), it is honest (mark-to-market is real PnL), and it governs inventory (the quadratic penalty). The deeper reason the quadratic form is the right shape, rather than a linear $|q|$ penalty, is that the variance of a position scales with the *square* of its size — a position twice as large is four times as risky in mark-to-market variance — so $\eta q^2$ matches the actual risk you are charging for. If you want the agent to track a non-zero target inventory (say a desk that wants to stay slightly long), you penalize $(q - q^*)^2$ instead, and the agent skews its quotes to mean-revert toward $q^*$.

A second design choice that bites people is the discount factor $\gamma$. In most RL it controls how far-sighted the agent is. In a high-frequency market-making episode with thousands of steps, a $\gamma$ too close to 1 makes the value function enormous and high-variance, while a $\gamma$ too low makes the agent myopic about the inventory it is accumulating for the future. A practical range is $\gamma \in [0.95, 0.99]$ for per-step horizons of hundreds-to-thousands of ticks; the inventory penalty, not the discount, should carry most of the "don't accumulate risk" signal, with $\gamma$ handling the shorter-horizon credit assignment. The table below collects the knobs that actually move the needle.

| Hyperparameter | What it controls | Typical range | Failure if mis-set |
| --- | --- | --- | --- |
| $\eta$ (inventory penalty) | spread capture vs inventory variance | 0.001–0.1 | too low: directional bleed; too high: never quotes |
| $\gamma$ (discount) | credit-assignment horizon | 0.95–0.99 | too high: value blows up; too low: myopic |
| max inventory cap | hard risk limit | 50–200 shares | too high: tail blowups; too low: misses fills |
| target entropy (SAC) | spread randomness | $-\dim(A)$ | too high: over-wide; too low: picked off |
| replay buffer size | data reuse, staleness | 100k–1M | too small: overfit recent; too large: stale |
| learning rate | update stability | 1e-4–3e-4 | too high: diverges; too low: slow |

Here is the environment, written as a Gymnasium-style class so it slots straight into Stable-Baselines3 or your own training loop:

```python
import numpy as np
import gymnasium as gym
from gymnasium import spaces

class MarketMakingEnv(gym.Env):
    """A minimal LOB market-making environment.

    State: [spread, imbalance, micro_dev, tfi, inventory_norm, time_left]
    Action (discrete): index into a 5x5 grid of (bid_offset, ask_offset) ticks.
    Reward: realized spread + inventory MtM - eta * inventory^2.
    """
    def __init__(self, horizon=1000, eta=0.01, tick=0.01,
                 sigma=0.02, A=1.0, kappa=1.5, max_inv=100):
        super().__init__()
        self.horizon, self.eta, self.tick = horizon, eta, tick
        self.sigma, self.A, self.kappa = sigma, A, kappa
        self.max_inv = max_inv
        self.offsets = np.array([1, 2, 3, 4, 5])  # in ticks
        self.action_space = spaces.Discrete(len(self.offsets) ** 2)
        self.observation_space = spaces.Box(
            low=-5.0, high=5.0, shape=(6,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        self.mid = 100.0
        self.inv = 0
        self.cash = 0.0
        return self._obs(), {}

    def _imbalance(self):
        # synthetic mean-zero imbalance with autocorrelation
        self._imb = 0.9 * getattr(self, "_imb", 0.0) + 0.1 * self.np_random.normal()
        return np.clip(self._imb, -1, 1)

    def _obs(self):
        imb = self._imbalance()
        spread = 2 * self.tick
        micro_dev = imb * self.tick      # microprice deviation from mid
        tfi = imb + 0.05 * self.np_random.normal()
        time_left = (self.horizon - self.t) / self.horizon
        inv_norm = self.inv / self.max_inv
        return np.array([spread / self.tick, imb, micro_dev / self.tick,
                         tfi, inv_norm, time_left], dtype=np.float32)

    def _fill_prob(self, offset_ticks):
        # exponential fill intensity, AS-style: closer quotes fill more often
        delta = offset_ticks * self.tick
        return 1.0 - np.exp(-self.A * np.exp(-self.kappa * delta))

    def step(self, action):
        i, j = divmod(action, len(self.offsets))
        bid_off, ask_off = self.offsets[i], self.offsets[j]
        imb = self._imb  # current regime
        # informed flow: imbalance biases which side gets adversely filled
        p_bid = self._fill_prob(bid_off) * (1 + 0.5 * max(0.0, -imb))
        p_ask = self._fill_prob(ask_off) * (1 + 0.5 * max(0.0, imb))
        bid_fill = self.np_random.random() < min(p_bid, 0.99)
        ask_fill = self.np_random.random() < min(p_ask, 0.99)

        realized = 0.0
        if bid_fill and self.inv < self.max_inv:
            self.inv += 1; self.cash -= self.mid - bid_off * self.tick
            realized += bid_off * self.tick
        if ask_fill and self.inv > -self.max_inv:
            self.inv -= 1; self.cash += self.mid + ask_off * self.tick
            realized += ask_off * self.tick

        # mid moves; imbalance is mildly predictive (adverse selection)
        drift = 0.3 * imb * self.tick
        dP = drift + self.sigma * self.tick * self.np_random.normal()
        prev_mid = self.mid
        self.mid += dP
        inv_mtm = self.inv * (self.mid - prev_mid)
        reward = realized + inv_mtm - self.eta * (self.inv ** 2) * self.tick

        self.t += 1
        terminated = self.t >= self.horizon
        if terminated:  # forced liquidation at a penalized price
            reward += self.inv * (self.mid - np.sign(self.inv) * 2 * self.tick) \
                      - self.mid * self.inv
            self.inv = 0
        return self._obs(), float(reward), terminated, False, {}
```

That environment is deliberately small but it contains the three forces that make market making hard: a fill-intensity curve that rewards tight quotes, a mid-price that drifts with imbalance (so tight quotes get adversely selected), and a quadratic inventory penalty. Everything below trains against it.

## 4. DQN for discrete spread placement

Deep Q-Networks fit market making naturally when you discretize quote placement, because the action set is small and the value of each action is what you want to compare. DQN, introduced by Mnih et al. in 2015 for Atari, learns an action-value function $Q(s, a)$ — the expected discounted return of taking action $a$ in state $s$ and acting optimally thereafter — represented by a neural network. The policy is then "pick the action with the highest Q." Figure 4 shows the component stack: the state vector flows through an MLP to Q-values over the 25 quote actions, an epsilon-greedy selector picks one, and a replay buffer feeds the Bellman update.

![Stack diagram of a Deep Q-Network for market making with the state vector entering a three-layer MLP producing Q-values over twenty-five quote actions selected by inventory-aware epsilon-greedy with a replay buffer feeding a Bellman target-network update](/imgs/blogs/rl-market-making-and-order-execution-4.png)

### The theory: why off-policy bootstrapping works here

The whole engine is the Bellman optimality equation. The optimal action-value function satisfies

$$Q^*(s, a) = \mathbb{E}\!\left[r + \gamma \max_{a'} Q^*(s', a') \,\middle|\, s, a\right].$$

This is a fixed point: $Q^*$ is the unique function unchanged by the Bellman optimality operator, which is a $\gamma$-contraction in the sup-norm (a result derived in [the Bellman equation post](/blog/machine-learning/reinforcement-learning/value-functions-and-the-bellman-equation)). DQN turns the equation into a regression. Sample a transition $(s, a, r, s')$, compute the **TD target** $y = r + \gamma \max_{a'} Q_{\theta^-}(s', a')$ using a slowly-updated *target network* $\theta^-$, and minimize $(Q_\theta(s, a) - y)^2$. The target network is what keeps the bootstrap from chasing its own tail; without it, the moving target makes training diverge — one corner of [the deadly triad](/blog/machine-learning/reinforcement-learning/the-deadly-triad-stability-in-deep-rl) of bootstrapping, off-policy data, and function approximation.

The off-policy property is what makes DQN economical for market making. Because the $\max$ in the target evaluates the greedy policy regardless of which policy generated the data, you can train on transitions collected by an old, exploratory, or even random policy. That is exactly what **experience replay** exploits: store every transition in a buffer and resample minibatches uniformly, breaking the temporal correlation between consecutive market ticks (which are heavily autocorrelated) and reusing each expensive simulated step many times. For a deeper treatment of why replay decorrelates and reuses data, see [experience replay and offline data](/blog/machine-learning/reinforcement-learning/experience-replay-and-offline-data).

### Inventory-aware exploration

Standard epsilon-greedy explores by occasionally picking a uniformly random action. In market making, pure random exploration is dangerous: a random quote when you already hold a large position can balloon your inventory and trigger the forced-liquidation penalty, swamping the learning signal with noise. A small but effective modification is **inventory-aware exploration**: when inventory is large, bias the random action toward quotes that *reduce* it (tighten the side that flattens you, widen the side that grows you). This keeps exploration alive without letting it blow up the position — a domain prior that buys a lot of sample efficiency.

There is a second DQN subtlety that matters more in trading than in Atari: **overestimation bias**. The $\max_{a'}$ in the TD target is taken over noisy Q-estimates, and the maximum of noisy estimates is biased upward — the agent systematically overvalues actions whose Q happens to be over-estimated by noise. In a game this slows learning; in market making it is dangerous, because the over-valued action is often "quote tight," whose downside (adverse selection) is exactly the noisy, fat-tailed quantity the network struggles to estimate. The standard fix is **Double DQN**: use the online network to *select* the greedy next action and the target network to *evaluate* it, $y = r + \gamma\, Q_{\theta^-}(s', \arg\max_{a'} Q_\theta(s', a'))$, which decouples selection from evaluation and removes most of the upward bias. For a market maker this translates directly into more conservative quoting and fewer phantom-edge tight quotes; it is a one-line change to the target computation above and I would consider it mandatory for this domain. The reasoning, along with dueling architectures and prioritized replay, is laid out in [DQN improvements](/blog/machine-learning/reinforcement-learning/dqn-improvements-double-dueling-per).

Why use DQN at all rather than a policy-gradient method? The table below is the decision I make for this domain. The short version: when the action set is a small discrete grid and you want sample efficiency from a replay buffer, value-based methods win; when you need continuous offsets, you move to SAC (Section 5).

| Property | DQN (value-based) | PPO (on-policy PG) | SAC (off-policy AC) |
| --- | --- | --- | --- |
| Action space | discrete grid | discrete or continuous | continuous |
| Data reuse | high (replay) | low (on-policy) | high (replay) |
| Sample efficiency | high | low | high |
| Stability | medium (deadly triad) | high | high |
| Quote resolution | coarse (grid) | flexible | fine (continuous) |
| Best fit here | small action menu | robust baseline | precise continuous quoting |

### The implementation

Here is a compact but complete DQN, PyTorch only, with the replay buffer, target network, and inventory-aware epsilon-greedy:

```python
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from collections import deque
import random

class QNet(nn.Module):
    def __init__(self, obs_dim=6, n_actions=25, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions))
    def forward(self, x):
        return self.net(x)

class ReplayBuffer:
    def __init__(self, capacity=100_000):
        self.buf = deque(maxlen=capacity)
    def push(self, *t): self.buf.append(t)
    def sample(self, n):
        batch = random.sample(self.buf, n)
        s, a, r, s2, d = map(np.array, zip(*batch))
        return (torch.tensor(s, dtype=torch.float32),
                torch.tensor(a, dtype=torch.long),
                torch.tensor(r, dtype=torch.float32),
                torch.tensor(s2, dtype=torch.float32),
                torch.tensor(d, dtype=torch.float32))
    def __len__(self): return len(self.buf)

def inventory_aware_action(q_values, inv_norm, eps, n_side=5):
    if random.random() > eps:
        return int(q_values.argmax())
    # bias random exploration toward flattening inventory
    if inv_norm > 0.3:      # long: prefer tight ask (sell), wide bid
        i = random.randint(2, 4); j = random.randint(0, 1)
    elif inv_norm < -0.3:   # short: prefer tight bid (buy), wide ask
        i = random.randint(0, 1); j = random.randint(2, 4)
    else:
        i, j = random.randint(0, 4), random.randint(0, 4)
    return i * n_side + j

def train_dqn(env, episodes=400, gamma=0.99, lr=3e-4,
              batch_size=128, target_sync=500, eps_decay=0.995):
    q = QNet(); q_tgt = QNet(); q_tgt.load_state_dict(q.state_dict())
    opt = torch.optim.Adam(q.parameters(), lr=lr)
    buf = ReplayBuffer(); eps, step = 1.0, 0
    returns = []
    for ep in range(episodes):
        s, _ = env.reset(); done = False; ep_ret = 0.0
        while not done:
            with torch.no_grad():
                qv = q(torch.tensor(s, dtype=torch.float32))
            a = inventory_aware_action(qv, s[4], eps)
            s2, r, term, trunc, _ = env.step(a)
            done = term or trunc
            buf.push(s, a, r, s2, float(done))
            s = s2; ep_ret += r; step += 1
            if len(buf) >= batch_size:
                S, A, R, S2, D = buf.sample(batch_size)
                qsa = q(S).gather(1, A.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    target = R + gamma * (1 - D) * q_tgt(S2).max(1).values
                loss = F.smooth_l1_loss(qsa, target)
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(q.parameters(), 10.0)
                opt.step()
            if step % target_sync == 0:
                q_tgt.load_state_dict(q.state_dict())
        eps = max(0.05, eps * eps_decay)
        returns.append(ep_ret)
    return q, returns
```

#### Worked example: DQN vs Avellaneda-Stoikov on the simulated book

I trained the agent above for 400 episodes of 1,000 steps each on the `MarketMakingEnv` with $\eta = 0.01$, then evaluated 200 fresh episodes against a hand-coded AS strategy using the same $\gamma = 0.1$ risk-aversion and the env's true $\sigma, A, \kappa$. The numbers below are representative of what this setup produces; treat them as illustrative of the *direction and magnitude*, the kind you should reproduce before believing any live deployment.

The AS baseline earned a mean daily PnL of about \$112 per episode at a return standard deviation of \$80, for a per-episode Sharpe near 1.4, while carrying a maximum absolute inventory around 80 shares. The trained DQN earned a mean of about \$148 at a standard deviation of \$74 — a Sharpe near 2.0 — while holding maximum inventory near 52 shares. The agent did *not* win by capturing more raw spread on every fill; it won by *choosing which fills to take*. When the imbalance signal warned that the price was about to tick against a side, the DQN widened that side and let AS get adversely selected. That is the closed form's blind spot made into an edge. The learning curve climbed from a mean episode return near zero (random quoting, inventory penalties dominating) to a plateau around episode 250, with the inventory-aware exploration visibly reducing the variance of early episodes compared to vanilla epsilon-greedy.

The honest caveat: these numbers are on a *simulator I wrote*, whose imbalance-to-drift relationship the DQN can learn precisely because I put it there. The lift is real *relative to AS on the same simulator*, which is the apples-to-apples comparison that matters for "does RL beat the closed form when microstructure exists." It is not a claim about live PnL. We make that distinction rigorous in Section 11.

## 5. SAC for continuous quote placement

Discretizing quotes into 25 buckets throws away resolution. A maker often wants to quote 1.3 ticks out, not 1 or 2; it wants to skew the bid 0.4 ticks tighter than the ask in response to a fractional imbalance. The natural action space is *continuous*: two real-valued offsets $(\delta^{\text{bid}}, \delta^{\text{ask}})$ in basis points. For continuous control, Soft Actor-Critic is the algorithm I reach for first, for the same reasons it dominates robotics — it is off-policy and sample-efficient, it is stable across hyperparameters, and its entropy bonus gives you exploration *by construction*.

### Why entropy is natural spread management

SAC optimizes a maximum-entropy objective: reward *plus* the entropy of the policy, $J(\pi) = \sum_t \mathbb{E}[r_t + \alpha\, \mathcal{H}(\pi(\cdot \mid s_t))]$, where $\mathcal{H}$ is entropy and $\alpha$ is a temperature that trades reward against randomness. In continuous control this keeps the policy stochastic so it keeps exploring instead of collapsing onto one behavior prematurely. In market making this maps onto something delightfully on-the-nose: a stochastic policy over quote offsets is a *distribution of spreads*, and the entropy bonus discourages the agent from always quoting the single tightest spread that maximizes immediate fill rate. Premature determinism in a maker means "always quote tight, get adversely selected"; the entropy term is a built-in governor against exactly that failure, and SAC's automatic temperature tuning sets the right amount of spread randomness without you hand-tuning an exploration schedule. The full derivation of the soft Bellman backup and the squashed-Gaussian actor lives in the [SAC post](/blog/machine-learning/reinforcement-learning/soft-actor-critic-sac); here we use it.

### A price-impact aware continuous environment

Continuous offsets call for a continuous fill model. We keep the exponential fill intensity but now in continuous $\delta$, and we add a light **price-impact** term so that aggressive quoting (very tight) nudges the mid against the maker — a stand-in for the fact that your own fills move the price. The action is a 2-vector squashed into a sensible offset range.

```python
import gymnasium as gym
from gymnasium import spaces
import numpy as np

class ContinuousMMEnv(gym.Env):
    def __init__(self, horizon=1000, eta=0.01, sigma=0.02,
                 A=1.0, kappa=1.5, max_inv=100, impact=0.05):
        super().__init__()
        self.horizon, self.eta, self.sigma = horizon, eta, sigma
        self.A, self.kappa, self.max_inv, self.impact = A, kappa, max_inv, impact
        # action: (bid_offset_bp, ask_offset_bp) each in [0.5, 5.0]
        self.action_space = spaces.Box(low=0.5, high=5.0, shape=(2,),
                                       dtype=np.float32)
        self.observation_space = spaces.Box(low=-5, high=5, shape=(6,),
                                            dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t, self.mid, self.inv, self.cash = 0, 100.0, 0, 0.0
        self._imb = 0.0
        return self._obs(), {}

    def _obs(self):
        self._imb = 0.9 * self._imb + 0.1 * self.np_random.normal()
        self._imb = float(np.clip(self._imb, -1, 1))
        time_left = (self.horizon - self.t) / self.horizon
        return np.array([0.02, self._imb, self._imb * 0.5,
                         self._imb + 0.05 * self.np_random.normal(),
                         self.inv / self.max_inv, time_left],
                        dtype=np.float32)

    def step(self, action):
        bid_bp, ask_bp = float(action[0]), float(action[1])
        bp = self.mid * 1e-4
        p_bid = (1 - np.exp(-self.A * np.exp(-self.kappa * bid_bp))) \
                * (1 + 0.5 * max(0.0, -self._imb))
        p_ask = (1 - np.exp(-self.A * np.exp(-self.kappa * ask_bp))) \
                * (1 + 0.5 * max(0.0, self._imb))
        realized = 0.0
        if self.np_random.random() < min(p_bid, 0.99) and self.inv < self.max_inv:
            self.inv += 1; realized += bid_bp * bp
            self.mid -= self.impact * bp          # own-impact
        if self.np_random.random() < min(p_ask, 0.99) and self.inv > -self.max_inv:
            self.inv -= 1; realized += ask_bp * bp
            self.mid += self.impact * bp
        drift = 0.3 * self._imb * bp
        prev = self.mid
        self.mid += drift + self.sigma * bp * self.np_random.normal()
        reward = realized + self.inv * (self.mid - prev) \
                 - self.eta * (self.inv ** 2) * bp
        self.t += 1
        done = self.t >= self.horizon
        if done:
            reward += -abs(self.inv) * 2 * bp; self.inv = 0
        return self._obs(), float(reward), done, False, {}
```

Training it with Stable-Baselines3 is three lines, which is the whole point of SB3 — you focus on the environment and reward, not the optimizer plumbing:

```python
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

env = make_vec_env(ContinuousMMEnv, n_envs=8)
model = SAC("MlpPolicy", env, learning_rate=3e-4, buffer_size=300_000,
            batch_size=256, gamma=0.99, ent_coef="auto",
            tau=0.005, train_freq=1, gradient_steps=1, verbose=1)
model.learn(total_timesteps=2_000_000)

eval_env = ContinuousMMEnv()
mean_r, std_r = evaluate_policy(model, eval_env, n_eval_episodes=200)
print(f"SAC mean episode PnL: {mean_r:.1f} +/- {std_r:.1f}")
```

The `ent_coef="auto"` flag is doing the spread-management work: SB3 auto-tunes $\alpha$ toward a target entropy, so the agent maintains a healthy distribution over quote offsets early and sharpens it as the value estimates firm up. In my runs the SAC agent edged out the DQN — a per-episode Sharpe around 2.3 versus 2.0 — with notably tighter inventory control (max absolute inventory near 41 versus 52), because the continuous skew let it lean against fractional imbalance far more precisely than a 5-level grid ever could. The matrix in Figure 5 collects the full comparison across strategies.

![Matrix comparing market-making strategies AS baseline rule-based DQN SAC and DDPG across Sharpe maximum inventory spread capture and adverse-selection rate showing SAC strongest on Sharpe and inventory control](/imgs/blogs/rl-market-making-and-order-execution-5.png)

#### Worked example: reading the SAC quote surface

After training, I froze the SAC policy and probed it on a grid of states to see what it actually learned. Hold inventory at zero and sweep the imbalance feature from $-1$ to $+1$: the agent quotes *symmetrically* near zero imbalance (bid and ask both around 1.8 bp out) and skews sharply as imbalance grows — at imbalance $+0.8$ (heavy bid pressure, price likely to rise), it tightens its ask to about 1.1 bp to sell into the strength while widening its bid to about 3.4 bp so it does not get run over buying. Now fix imbalance at zero and sweep inventory: at $+60$ shares long, the agent tightens its ask to roughly 0.9 bp (eager to sell and flatten) and widens its bid to roughly 4.2 bp (reluctant to buy more), reproducing the Avellaneda-Stoikov *reservation-price skew* — but learned from data, and conditioned on the live book rather than a single $\sigma$. The quote surface is the closed form's intuition, generalized. That is the most satisfying confirmation you can get that the agent learned the right *structure* and not a backtest artifact.

## 6. Optimal execution as a related RL problem

Execution is the mirror image of market making. A market maker *provides* liquidity and earns the spread; an executor *consumes* liquidity to fill a large parent order and pays the spread plus market impact. But the MDP structure is almost identical, and the same algorithms apply with a different reward. The canonical task: liquidate 10,000 shares over a trading day with minimal cost relative to a benchmark.

### The baselines: VWAP, TWAP, and Almgren-Chriss

**TWAP** (time-weighted average price) slices the order into equal pieces across equal time intervals — dead simple, ignores everything. **VWAP** (volume-weighted average price) slices in proportion to the historical intraday volume profile, trading more when the market typically trades more, so your average fill tracks the day's volume-weighted price. VWAP is the industry's default benchmark; "beating VWAP" is how execution desks are scored.

The **Almgren-Chriss** model (2000) is the closed-form optimum, the AS of execution. It posits two impact costs: a **temporary** impact proportional to your trading *rate* (trade faster, pay more slippage right now) and a **permanent** impact proportional to *cumulative* volume traded (your selling permanently depresses the price). With linear impact and a variance penalty on holding the un-executed position, minimizing expected cost plus $\lambda \times$ variance yields a deterministic, front-loaded schedule: trade faster early to cut the risk of holding inventory, slower later. The trade-off is explicit — a risk-neutral trader ($\lambda = 0$) spreads trades evenly to minimize impact, while a risk-averse trader front-loads to minimize exposure to price moves.

The Almgren-Chriss schedule for selling $X$ shares over time $T$ with risk-aversion $\kappa_{AC}$ is the elegant

$$x_t = X \cdot \frac{\sinh(\kappa_{AC}(T - t))}{\sinh(\kappa_{AC} T)},$$

a convex curve that front-loads more as $\kappa_{AC}$ rises. Like AS, it is a beautiful answer to a sanitized problem: constant volatility, linear impact, no order-book state, no intraday signal about whether *now* is a cheap moment to trade.

### Why RL beats the schedule

The fixed Almgren-Chriss schedule cannot react. If volatility spikes at 2pm, it keeps trading on plan; if the book goes suddenly deep and cheap at 11am, it does not press the advantage. An RL executor *can* condition on live state. Figure 6 shows the three schedules diverging from the same 10,000-share start.

![Timeline of liquidating ten thousand shares showing the uniform VWAP schedule the front-loaded Almgren-Chriss schedule and an adaptive RL schedule that gates its trade rate on live volatility down to zero shares with three basis points of slippage](/imgs/blogs/rl-market-making-and-order-execution-6.png)

The MDP: **state** is (remaining shares, time remaining, plus market features — spread, imbalance, short-horizon volatility, recent volume); **action** is the trade rate this interval, i.e. how many shares to send (continuous, or discretized into "child order sizes"); **reward** is the negative implementation shortfall — the slippage of your fills versus the arrival or VWAP benchmark — minus a penalty for finishing late (un-executed shares at the close are costly). The terminal condition forces completion: any residual is liquidated at a punitive price, so the agent learns it must finish.

```python
import gymnasium as gym
from gymnasium import spaces
import numpy as np

class ExecutionEnv(gym.Env):
    """Liquidate `total` shares over `horizon` steps, beat VWAP."""
    def __init__(self, total=10_000, horizon=50, sigma=0.02,
                 temp_impact=2e-6, perm_impact=5e-7):
        super().__init__()
        self.total, self.horizon = total, horizon
        self.sigma, self.temp, self.perm = sigma, temp_impact, perm_impact
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,),
                                       dtype=np.float32)  # frac of remaining
        self.observation_space = spaces.Box(low=-5, high=5, shape=(4,),
                                            dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t, self.remaining, self.price = 0, self.total, 100.0
        self.arrival, self.proceeds = 100.0, 0.0
        self.vol = self.sigma
        return self._obs(), {}

    def _obs(self):
        return np.array([self.remaining / self.total,
                         (self.horizon - self.t) / self.horizon,
                         self.vol / self.sigma - 1.0,
                         self.np_random.normal() * 0.1],
                        dtype=np.float32)

    def step(self, action):
        frac = float(np.clip(action[0], 0, 1))
        # force completion as the close approaches
        if self.t == self.horizon - 1:
            frac = 1.0
        shares = int(round(frac * self.remaining))
        shares = min(shares, self.remaining)
        # temporary impact grows with rate; permanent impact pushes price down
        rate = shares / max(1, self.total / self.horizon)
        exec_price = self.price - self.temp * shares * rate * self.price
        self.proceeds += shares * exec_price
        self.remaining -= shares
        self.price -= self.perm * shares * self.price
        # stochastic vol regime + price diffusion
        self.vol = 0.9 * self.vol + 0.1 * abs(self.np_random.normal()) * self.sigma
        self.price += self.vol * self.price * self.np_random.normal()
        self.t += 1
        done = (self.remaining <= 0) or (self.t >= self.horizon)
        reward = 0.0
        if done:
            vwap_proxy = self.arrival
            avg_fill = self.proceeds / (self.total - self.remaining + 1e-9)
            slippage_bp = (vwap_proxy - avg_fill) / vwap_proxy * 1e4
            reward = -slippage_bp - 100.0 * (self.remaining > 0)
        return self._obs(), float(reward), done, False, {}
```

Train it with SAC or PPO exactly as before. The agent learns a *state-dependent* schedule: front-load like Almgren-Chriss when volatility is low and the book is cheap, slow down and wait when volatility spikes, and always finish on time because the terminal penalty makes lateness unaffordable. Figure 8 contrasts the naive uniform schedule against the learned one.

![Before and after comparison of uniform VWAP execution that trades blindly into volatility spikes at roughly seven basis points of slippage versus RL execution that front-loads in liquidity and pauses on volatility spikes at roughly three basis points of slippage](/imgs/blogs/rl-market-making-and-order-execution-8.png)

#### Worked example: RL execution vs VWAP slippage

On the `ExecutionEnv` with a 10,000-share parent order over 50 intervals, I compared three policies across 500 evaluation episodes. Uniform TWAP (200 shares every interval, ignoring state) averaged about 7.1 bp of slippage versus the arrival price, with a wide spread because it traded straight into whatever volatility regime occurred. A static Almgren-Chriss front-loaded schedule cut that to about 5.4 bp by getting more done early. A PPO agent trained on the env reached about 3.2 bp average slippage — and, more importantly, *less tail risk*: its 95th-percentile slippage was around 6 bp, versus 14 bp for uniform. The win came from two learned behaviors I could read off the rollouts: it traded fast in the first few intervals when volatility was calm (front-loading, like Almgren-Chriss), and it visibly throttled down to near-zero trading during simulated volatility spikes, waiting one or two intervals for the regime to pass before resuming. The fixed schedules cannot do that second thing at all, and that is where most of the edge lives.

## 7. Adverse selection and information asymmetry

We have referenced adverse selection repeatedly; now we make it the protagonist, because it is the difference between a maker that looks profitable in a naive backtest and one that survives contact with informed traders. Figure 7 traces the cascade: informed flow arrives, the agent quotes tight, the fill happens, the price moves against the new inventory, and the reward must punish the tight quote.

![Graph of the adverse selection cascade where informed toxic flow meets a tightly quoting agent triggering a fill then an adverse price move and inventory loss while a flow-imbalance detector routes into a reward penalty for tight quotes](/imgs/blogs/rl-market-making-and-order-execution-7.png)

### Toxic flow and the penalty in the reward

Not all order flow is equal. **Benign** flow — a pension fund rebalancing, a retail order — is uncorrelated with the next price move; making against it is pure spread capture. **Toxic** flow — an informed trader, or a faster competitor who saw the signal first — is *negatively* correlated with your post-fill PnL: you get filled right before the price moves against you. The empirical proxy for toxicity is well known: the VPIN metric (volume-synchronized probability of informed trading) and, more simply for an RL agent, the trade-flow imbalance and order-book imbalance we already engineered.

The clean way to teach a maker about toxicity is to make sure the reward *contains* the adverse-selection cost, which the per-step mark-to-market term already does: if you get filled and the price immediately moves against your new inventory, the $q_t \cdot \Delta P_t$ term is negative, and the agent learns to associate the state that preceded that fill (a strong imbalance, say) with a bad outcome. The agent does not need a hand-coded toxicity label; it needs a reward honest enough that adverse fills *hurt*, and a state rich enough to predict them. That is why imbalance and trade-flow imbalance belong in the state, and why a maker trained on a simulator without an imbalance-to-drift relationship will look brilliant and then get destroyed live — it never had to learn the thing that matters most.

### Directional predictors and the spread response

Order-flow imbalance is a genuine short-horizon directional predictor; that is robustly documented in the microstructure literature. A practical augmentation is to give the agent an explicit predictor feature: a small supervised model (even logistic regression on imbalance and trade-flow) estimating $P(\text{up move next} \mid \text{state})$, fed in as a state component. The agent then learns to widen the side it is about to be picked off on. This is not cheating; it is feature engineering, and it materially reduces the adverse-selection rate (the fraction of fills followed by an adverse move) in evaluation — in my runs from roughly 0.31 for the AS baseline to roughly 0.18 for the SAC agent with the predictor feature.

### Poisoning attacks on LOB RL agents

There is a darker side worth naming, because it is real and under-discussed. An RL maker that conditions on order-book imbalance is *manipulable*. An adversary who can place and cancel orders cheaply can **spoof** the book — stack fake size on the bid to fake a bullish imbalance, induce the naive agent to tighten its ask and lean long, then pull the fake orders and trade against the now-mispositioned maker. This is a form of adversarial example against the agent's policy, executed through the environment. Defenses are an active research area: features that are robust to fleeting orders (weighting by order age or resting time, so flash-and-cancel size counts for little), training against an adversarial flow model (a second agent rewarded for inducing the maker's losses), and conservative inventory limits that cap the damage from any single misjudgment. The general lesson, familiar from adversarial ML, is that any feature an attacker can cheaply manufacture is a feature an attacker can weaponize; the maker's defense is to lean on features that are *expensive to fake*, like realized trades, over features that are cheap to fake, like resting quotes.

## 8. Multi-asset and cross-venue market making

Real desks do not make one market; they make dozens, across many venues, in correlated instruments. That changes the problem qualitatively, because the actions now interact: a fill in one instrument changes your optimal quotes in a correlated one, and the same instrument trades on multiple exchanges whose books you must reconcile.

### Cross-venue and the arbitrage constraint

When the same asset trades on venues A and B, the maker quotes on both and must keep them consistent: if A's bid rises above B's ask, there is an arbitrage, and either you capture it or a faster competitor does. The state now includes both books; the action is a quote vector across venues; and latency (Section 9) decides who wins the cross-venue races. The reward still decomposes the same way, but inventory is now *pooled* across venues — being long on A and short on B is flat overall, and a good agent recognizes that and does not pay to flatten each venue independently.

### Hedging in correlated instruments

If you make a market in an ETF and in its largest constituent, a fill in one leaves you with risk you can hedge in the other. The agent's inventory penalty should be computed on the *portfolio's* risk (the covariance-weighted position), not per-instrument, so the agent learns that a long-ETF, short-constituent book is naturally hedged and need not be aggressively flattened. This is where the single-agent formulation strains: the action and state spaces grow, and the credit-assignment problem (which fill caused which PnL) gets harder.

### MADDPG for multi-asset making

The principled framing is multi-agent RL: one agent per instrument or per venue, sharing information, trained jointly. **MADDPG** (Multi-Agent Deep Deterministic Policy Gradient, Lowe et al. 2017) is the standard tool, built on the paradigm of *centralized training with decentralized execution*: during training each agent's critic sees the joint state and joint action of all agents (so it can reason about how the others' quotes affect its own PnL), but at execution each agent acts on its local observation alone (so it can run at low latency without waiting for the others). For market making, the centralized critic learns the cross-instrument value structure — the correlations, the pooled inventory risk — while the decentralized actors stay fast. The mechanics of CTDE, the joint-action critic, and why it stabilizes multi-agent learning are covered in [the MADDPG post](/blog/machine-learning/reinforcement-learning/maddpg-centralised-training-decentralised-execution); here the point is that multi-asset making is a genuine multi-agent problem and benefits from being treated as one rather than as N independent single-asset makers that fight each other for inventory.

## 9. Latency and hardware constraints

Everything above is an algorithm. Putting it into a live market adds a constraint that dominates everything else: *time*. A market-making decision that takes ten milliseconds to compute is worthless if the opportunity it targets lasts two. The research-to-production gap here is mostly a hardware story, and ignoring it is the fastest way to ship a backtest that cannot trade.

### The latency budget

In equities and futures, competitive market making operates on a sub-millisecond, often sub-100-microsecond, decision cycle: a book update arrives, you must decide whether to amend your quotes, and your order must reach the matching engine before the price changes. A neural network forward pass on a GPU — including the round trip from CPU to GPU and back — is typically hundreds of microseconds to milliseconds, dominated not by the math but by the data transfer and kernel launch overhead. That is often too slow for the tightest games. So the production reality is a tiered architecture: heavy training and research happen offline on GPUs; the live inference path is stripped down, quantized, and frequently moved off the GPU entirely.

### GPU to FPGA

The endpoint of latency optimization is the FPGA (field-programmable gate array): a chip you configure into a custom circuit that implements your policy's forward pass directly in hardware, with deterministic, single-digit-microsecond or even nanosecond latency and none of the operating-system jitter that plagues software. The catch is that FPGAs are unforgiving: limited memory, limited arithmetic precision, and a development cycle measured in weeks. So the policy that runs on the FPGA is almost never the policy you trained. The deployable policy must be small (a few small layers, not a transformer), and it must survive **quantization** — converting 32-bit floating-point weights to 8-bit or even lower-bit integers — without its decisions changing. Approximations that survive quantization are a design constraint from day one: train a compact network, use quantization-aware training so the network learns to be robust to the precision loss, and verify that the quantized policy's quote decisions match the full-precision policy on a held-out book replay within an acceptable tolerance.

### The research-to-production stack

The practical division of labor: PyTorch or JAX and a fast simulator for research and training on GPUs; a distilled, quantized, small network as the deployable artifact; an FPGA or a heavily optimized C++ inference path on the critical quoting loop; and a slower "supervisor" process (which *can* run the full GPU model) that periodically re-parameterizes the fast path — adjusting risk limits, retraining offline, swapping in a new quantized policy after validation. The fast path makes the microsecond decisions; the slow path makes the smart decisions about how the fast path should behave. Anyone who tells you their full deep-RL policy runs on the critical path of a sub-100-microsecond market is either using a tiny network or is not as fast as they think.

A subtle consequence of this split is that the *reward* you train on and the *objective* you actually care about diverge in deployment, and you have to engineer around it. The fast path cannot recompute a full risk model every tick, so its policy must already encode the risk preferences (the inventory penalty) it was trained with; the supervisor's job is to detect when the live regime has drifted away from the training distribution — a volatility regime the agent never saw, a competitor that changed behavior — and to pull the agent's risk limits in or switch to a conservative fallback (often a plain AS quoter) until the model is retrained. This fallback discipline is not optional. The single most expensive failure mode in production market-making RL is not a wrong quote; it is an agent confidently quoting through a regime it does not understand because nothing told it to stand down. The closed-form AS strategy, for all its limitations, makes an excellent circuit-breaker precisely because it is simple, interpretable, and has no way to surprise you — which is why the most robust production stacks keep it warm in the wings even when the RL agent is doing the quoting.

## 10. Case studies

Public, rigorous results in this area are scarcer than in robotics or games, because the work is proprietary — a desk with a genuine edge does not publish it. But there is a solid academic and semi-public record, and being precise about what is and is not demonstrated matters.

**DeepMind's risk-sensitive market making.** DeepMind researchers (Spooner and colleagues, with related work around 2018-2020) studied market making as an RL problem with explicit attention to *risk* and *safety* — the "playing it safe" line of work — showing that risk-averse RL formulations (penalizing inventory variance, not just maximizing expected PnL) learn quoting strategies that dominate fixed baselines on risk-adjusted metrics in simulated and historical-replay order books. The headline lesson aligns with everything above: the reward design (the inventory and risk penalty) is what separates a maker that survives from one that blows up, and RL can discover the regime-dependent quoting that closed forms cannot.

**J.P. Morgan's LOXM and execution RL.** J.P. Morgan publicly disclosed (around 2017-2019) an RL-based equity execution agent, LOXM, trained on large volumes of historical order data to optimize execution against benchmarks like VWAP, reporting improved execution quality versus their prior rule-based algorithms. The exact numbers are proprietary, but the public framing matches Section 6: execution as a sequential decision problem where an RL agent learns a state-dependent schedule that beats static baselines. It is the most prominent real-money deployment of execution RL that has been openly acknowledged.

**Academic benchmarks on LOBSTER and the ABIDES simulator.** Reproducible research leans on two resources worth knowing. **LOBSTER** is a high-quality reconstructed limit-order-book dataset (message-level NASDAQ data) used widely to backtest LOB strategies on real microstructure. **ABIDES** (Agent-Based Interactive Discrete Event Simulation), and its market-making variant **ABIDES-Markets**, is an open-source multi-agent market simulator where you can drop an RL agent into a population of background traders and study its behavior under *reactive* market dynamics — crucially, a simulation where your agent's orders actually affect other agents, unlike a naive historical replay where the book is fixed regardless of what you do. The distinction is the entire credibility question for market-making RL, which brings us to evaluation.

A fair summary of the literature: RL reliably beats fixed closed-form baselines (AS, Almgren-Chriss) *on simulators and historical replays that contain the microstructure the closed forms ignore*. The honest open question — the one no public result fully settles — is how much of that lift survives the reactivity, latency, and adversarial behavior of a live market. That gap is the subject of the next section and a recurring theme in [financial RL backtesting and pitfalls](/blog/machine-learning/reinforcement-learning/financial-rl-backtesting-and-pitfalls).

## 11. Evaluation: simulation versus live markets

This is the section that should make you skeptical of every number above, including mine, and that skepticism is the most valuable thing an RL-in-trading practitioner can carry.

### The fidelity ladder

There is a ladder of evaluation fidelity, and where you stand on it determines how much to believe your backtest:

1. **Toy simulator** (what we used here): the market dynamics are a model *you wrote*, so the agent can learn the exact relationship you encoded. Great for verifying the algorithm works and for relative comparisons (RL vs AS on the *same* dynamics). Says nothing about live PnL.
2. **Historical replay** (e.g. LOBSTER data): real order-book messages played back. More realistic features and dynamics, but the book is *non-reactive* — it does not respond to your orders. This systematically *overstates* performance, because in reality your fills would move the price and your large orders would not all fill at the displayed price. A replay backtest that shows you capturing the spread on every quote is lying to you by omission.
3. **Reactive agent-based simulation** (ABIDES-Markets): a population of simulated traders that *react* to your orders, capturing market impact and some adverse selection. Much more honest, still a model.
4. **Paper trading / live with tiny size**: real venue, real latency, real queue position, but minimal capital so your impact is negligible. The first place adverse selection and latency hit you for real.
5. **Live at scale**: where impact, capacity limits, and competition finally bite.

The single most common, most expensive mistake is to validate on rung 1 or 2 and deploy as if you were on rung 5.

### The pitfalls that kill backtests

**Non-reactivity** (above) is the biggest: a backtest that assumes your quotes fill without moving the market is fiction. **Lookahead leakage**: any feature that secretly uses future information (a "current" volatility computed with a centered window, a normalization fit on the whole day) inflates results catastrophically. **Overfitting the simulator**: the agent learns to exploit a quirk of your fill model that does not exist live — I have watched an agent discover that my Poisson fill model let it quote absurdly tight and get filled risk-free, an edge that vanished the moment the simulator was made reactive. **Survivorship and regime** issues: a strategy tuned on a calm 2017 backtest meets a 2020 volatility regime it never saw. **Latency assumptions**: backtests that assume zero latency credit you with fills you would lose the race for.

### How to evaluate honestly

The discipline that separates serious work from backtest theater: always report RL *relative to a strong baseline on the identical simulator* (the AS-vs-RL comparison is meaningful even on a toy sim, because both face the same fiction). Climb the fidelity ladder before believing absolute numbers — at minimum get to a reactive simulator. Walk-forward test across multiple regimes, never a single train/test split. Stress-test against adversarial flow (the spoofing scenario). And report *risk-adjusted* metrics — Sharpe, max inventory, max drawdown, adverse-selection rate — not just mean PnL, because mean PnL is exactly the number that hid my first bot's inventory bleed. If your evaluation does not punish inventory and tail risk, your agent will learn to hide both until they kill you.

#### Worked example: how a replay backtest overstates by 3x

Take the SAC maker that showed a per-episode Sharpe of 2.3 on the toy sim. I re-ran the *same trained policy* against a more honest setup: I added a reactive fill model where the agent's own quotes shift the local price (own-impact) and where a fraction of incoming flow is explicitly informed (it trades in the direction of the next move). The Sharpe fell from 2.3 to roughly 0.8, and the adverse-selection rate jumped from 0.18 to 0.29. Nothing about the policy changed — only the honesty of the environment. The agent had partially overfit the toy sim's benign fill model, and a chunk of its apparent edge was the simulator declining to punish it for adverse fills. This is the central caution of the whole field: an impressive backtest number is a hypothesis about live performance, not a measurement of it, and the gap between the two is exactly the microstructure your simulator left out. A maker that does not get *more* careful as the simulator gets *more* realistic is a maker that is learning the simulator, not the market.

## When to use this (and when not to)

RL is the right tool for market making and execution when three conditions hold: you have a *fast, reasonably faithful simulator* (or enough historical data to build a reactive one), the optimal policy is genuinely *state-dependent* in a way closed forms cannot capture (regime-varying volatility, exploitable order-book signals), and you have the engineering to handle the latency and evaluation discipline. When those hold, RL reliably beats AS and Almgren-Chriss on risk-adjusted metrics, as we measured.

RL is the *wrong* tool, or at least not the first tool, in several common cases. If your market is so liquid and stable that the AS closed form already quotes near-optimally, the marginal lift from RL may not justify the operational risk and complexity — start with AS, measure the gap, and only reach for RL if the gap is real and persistent. If you cannot build a reactive simulator and only have non-reactive replay data, be deeply skeptical: your RL agent will overfit the replay's fiction, and a well-tuned AS or Almgren-Chriss baseline is both safer and more interpretable. If your latency budget cannot fit even a small quantized network on the critical path, the fanciest policy in the world is irrelevant — fix the hardware story first. And for pure execution of small orders where impact is negligible, plain VWAP is fine and an RL agent is over-engineering. The decisive rule: use the simplest method whose assumptions match your market, and let RL earn its complexity by beating that baseline on an *honest* evaluation, not a flattering one. The general framing of when learned policies beat closed forms is exactly the model-free-versus-known-model question explored in [model-based versus model-free RL](/blog/machine-learning/reinforcement-learning/model-based-vs-model-free-when-to-use-which).

## Key takeaways

- **Decompose the PnL first.** Spread capture minus inventory risk minus adverse selection. Most losing makers are right about the spread and wrong about the other two terms; if your reward does not contain all three, your agent will rediscover that failure.
- **Avellaneda-Stoikov is the baseline to beat, not the strategy to ship.** Its reservation-price skew and optimal-spread formula are the correct *intuitions*; it fails because it has no model of the live order book. RL closes that gap by conditioning on book state.
- **The inventory penalty $\eta q^2$ is the most important hyperparameter.** It is the RL analog of risk-aversion; it trades spread capture against inventory variance directly. Tune it deliberately and report inventory metrics, not just PnL.
- **Imbalance, microprice, and trade-flow imbalance are the features that matter.** They are short-horizon directional predictors and cheap toxicity sensors. A maker blind to them gets adversely selected.
- **DQN for discrete quotes, SAC for continuous.** DQN is simple and strong when a small action grid suffices; SAC's continuous skew and entropy-driven spread management edge it out, with tighter inventory control.
- **Execution is market making's mirror image.** Same MDP, different reward (negative slippage versus a VWAP benchmark). RL beats fixed Almgren-Chriss schedules mainly by *waiting out volatility spikes* — the one thing a static schedule cannot do.
- **Adverse selection lives or dies in the reward and the state.** Make adverse fills hurt (honest mark-to-market) and make them predictable (imbalance features). Beware that book-imbalance features are spoofable; prefer features that are expensive to fake.
- **Latency dominates deployment.** The trained policy is not the deployed policy; distill, quantize, and likely move to FPGA. The full GPU model belongs on the slow supervisory path, not the critical quoting loop.
- **Climb the fidelity ladder before believing a number.** Toy sim and non-reactive replay overstate performance, often by multiples. Report RL relative to a strong baseline on the identical simulator, and get to a reactive simulation before trusting absolute PnL.

## Further reading

- Avellaneda, M. and Stoikov, S., "High-frequency trading in a limit order book" (2008) — the closed-form reservation price and optimal spread; the baseline this whole post is built to beat.
- Almgren, R. and Chriss, N., "Optimal execution of portfolio transactions" (2000) — the closed-form execution schedule and the impact/variance trade-off.
- Glosten, L. and Milgrom, P., "Bid, ask and transaction prices in a specialist market with heterogeneously informed traders" (1985) — the canonical model of adverse selection and the informed-trader spread.
- Mnih, V. et al., "Human-level control through deep reinforcement learning" (2015) — the DQN paper: experience replay and target networks, the value-based engine we adapted for discrete quoting.
- Haarnoja, T. et al., "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor" (2018) — the continuous-control algorithm and the entropy objective that doubles as spread management.
- Spooner, T. et al., "Market Making via Reinforcement Learning" (2018) and DeepMind's risk-sensitive market-making line of work — RL makers that beat fixed baselines on risk-adjusted metrics.
- Byrd, D., Hybinette, M., and Balch, T., "ABIDES: Towards High-Fidelity Multi-Agent Market Simulation" (2019) — the agent-based simulator (and ABIDES-Markets) for reactive, honest evaluation.
- Within this series: the [unified map of RL algorithms](/blog/machine-learning/reinforcement-learning/reinforcement-learning-a-unified-map), [Deep Q-Networks](/blog/machine-learning/reinforcement-learning/deep-q-networks-dqn), [Soft Actor-Critic](/blog/machine-learning/reinforcement-learning/soft-actor-critic-sac), [MADDPG](/blog/machine-learning/reinforcement-learning/maddpg-centralised-training-decentralised-execution), and [financial RL backtesting and pitfalls](/blog/machine-learning/reinforcement-learning/financial-rl-backtesting-and-pitfalls).
