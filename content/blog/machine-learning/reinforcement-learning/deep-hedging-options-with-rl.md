---
title: "Deep Hedging: Using RL to Hedge Options Beyond Black-Scholes"
date: "2026-06-22"
publishDate: "2026-06-22"
description: "How Black-Scholes delta hedging breaks under transaction costs and jumps, why hedging is a natural RL problem, and how deep hedging trains neural nets that beat the delta on real options."
tags:
  [
    "reinforcement-learning",
    "deep-learning",
    "finance",
    "deep-hedging",
    "options",
    "risk-management",
    "machine-learning",
    "pytorch",
    "markov-decision-process",
  ]
category: "machine-learning"
subcategory: "Reinforcement Learning"
author: "Hiep Tran"
featured: true
readTime: 50
image: "/imgs/blogs/deep-hedging-options-with-rl-1.png"
---

A trader on a derivatives desk sells a one-month at-the-money call on a single stock. Textbook finance has an answer for what to do next: compute the option's delta, buy that many shares, and rebalance continuously so the position stays delta-neutral. Black and Scholes proved that if you do this perfectly, in a world of geometric Brownian motion with no trading costs, you replicate the option exactly and your hedging error goes to zero. It is one of the most beautiful results in quantitative finance.

It is also a lie the moment money changes hands. You cannot trade continuously — you rebalance once a day, maybe a few times a day if the stock moves. Every trade pays a bid-ask spread and a commission. The stock does not follow geometric Brownian motion; it gaps on earnings, jumps on a Fed surprise, and its volatility itself moves around. Run the Black-Scholes delta-hedging recipe against a realistic market and the hedging error does not vanish — it piles up in the left tail of your P&L distribution, exactly where it hurts. The desk that hedges naively bleeds slowly in transaction costs during quiet markets and then takes a brutal hit when a jump arrives between rebalances.

Here is the reframe that this post is built on: hedging is a sequential decision problem. At each step you observe the state of the world — the option's Greeks, the current spot, the implied and realized volatility, how much stock you already hold — and you choose an action — your new hedge ratio. The market then moves, you book a reward — your P&L net of the cost of trading — and the loop repeats until the option expires and the payoff settles. That is a Markov decision process (MDP), the exact object reinforcement learning was built to solve. Figure 1 lays out the loop. Once you see hedging this way, the question stops being "what is the Black-Scholes delta?" and becomes "what hedging policy minimizes my risk after costs, on the market I actually face?"

![A diagram showing options hedging framed as a Markov decision process with state, policy, action, market move, reward, and episode termination at expiry.](/imgs/blogs/deep-hedging-options-with-rl-1.png)

That question is what **deep hedging** answers. Introduced by Buehler, Gonon, Teichmann, and Wood in their 2019 paper "Deep Hedging," it replaces the closed-form delta with a neural network trained to minimize a risk measure of terminal P&L directly, on simulated or historical paths, accounting for transaction costs and any market dynamics you can sample from. By the end of this post you will understand why the discrete-hedging optimal hedge is *not* the Black-Scholes delta, how to formulate hedging as an MDP, how the deep hedging objective connects to the standard RL return, and you will have runnable PyTorch code that trains an LSTM hedger on Heston paths and beats the delta on tail risk. This is one node in our larger series map of how the RL loop — agent, environment, reward, policy update — specializes to real systems; here the environment is a market and the reward is money.

## 1. Hedging as a sequential decision problem

Let us nail down the running example so we can refine it the whole way through. You are short one European call on a stock currently at \$100, strike \$100, expiring in 30 trading days, with implied volatility around 20%. Being short the call means you owe the buyer the payoff $\max(S_T - K, 0)$ at expiry. If the stock rallies, you lose; if it stays flat or falls, you keep the premium. Your job is not to predict the stock — it is to *neutralize* your exposure so that whatever the stock does, your combined position (short call plus a stock hedge) has as little risk as possible.

The lever you have is how many shares to hold. Hold zero shares and you are fully exposed to the call's payoff. Hold one share per option and you are over-hedged when the option is far out-of-the-money. The right number sits in between and changes as the stock moves and time passes. That number — shares held per option, expressed as a fraction between 0 and 1 for a call — is your **hedge ratio** $h_t$. Choosing $h_t$ at every step, given everything you can observe, *is* the hedging policy.

Why is this naturally an MDP and not just a static optimization? Three reasons. First, it is **sequential**: the hedge you hold today determines how much you must trade tomorrow, and trading is costly, so today's decision has consequences for future costs. Second, it is **stateful**: the relevant information for the next decision — spot, time to maturity, your current holdings, the volatility environment — is summarized in a state that evolves over time. Third, the objective is a **cumulative reward**: your total P&L is the sum of per-step trading P&L minus per-step costs, realized over the whole episode. An MDP is exactly a tuple of states, actions, a transition kernel, and a reward, optimized over time — and hedging fits every slot.

Let me define the pieces precisely, because the rest of the post leans on them.

- **State** $s_t$: a vector summarizing what you know at step $t$. At minimum the current spot $S_t$, the time to maturity $\tau = T - t$, the moneyness $S_t / K$, the implied volatility $\sigma_t$, a realized-volatility estimate, and crucially your current hedge $h_{t-1}$ (you need to know what you already hold to know what to trade).
- **Action** $a_t$: the new hedge ratio $h_t \in [0, 1]$ for a call (or equivalently the *change* $\delta_t = h_t - h_{t-1}$, the number of shares to buy or sell).
- **Reward** $r_t$: the P&L of the hedge over the step minus the transaction cost of the rebalance. If the stock moves by $\Delta S_t = S_{t+1} - S_t$ and you held $h_t$ shares, the hedge made $h_t \Delta S_t$; the option leg changed value too; and the trade $|\delta_t|$ cost you proportional to the spread. We will write this out carefully in section 4.
- **Episode**: the option's life, from inception at $t = 0$ to expiry at $t = T$, after which the payoff settles and the episode ends.

This is the spine of the whole series restated in finance terms: an agent (the hedger) interacts with an environment (the market) by taking actions (rebalances), collecting rewards (P&L net of cost), and we want to learn a policy (the hedge function) that optimizes a cumulative objective. Every RL algorithm is one answer to *which objective* and *how to estimate the gradient* — and deep hedging, as we will see, picks an unusual but powerful answer to the second part because the entire environment is differentiable.

One subtlety worth flagging now: the reward in hedging is not "make money." A perfect hedge has *zero* expected P&L by construction (you collected the premium up front; the hedge just neutralizes the payoff). The objective is **risk reduction**, not return. That is why, in section 3, the RL objective is going to be a *risk measure* of terminal P&L rather than expected return — a twist that distinguishes financial hedging from a game like CartPole where you simply want to maximize score.

## 2. Black-Scholes delta hedging and why it breaks

To appreciate what deep hedging buys you, you have to understand the baseline it replaces, and exactly where that baseline cracks.

Black-Scholes starts from a strong assumption: the stock follows **geometric Brownian motion (GBM)**,

$$ dS_t = \mu S_t \, dt + \sigma S_t \, dW_t, $$

where $\mu$ is the drift, $\sigma$ is a *constant* volatility, and $W_t$ is a Brownian motion. Under this assumption plus frictionless continuous trading, the price of a European call is the famous closed form

$$ C(S, t) = S \, \Phi(d_1) - K e^{-r\tau} \Phi(d_2), \qquad d_1 = \frac{\ln(S/K) + (r + \tfrac{1}{2}\sigma^2)\tau}{\sigma \sqrt{\tau}}, \quad d_2 = d_1 - \sigma\sqrt{\tau}, $$

where $\Phi$ is the standard normal CDF and $r$ the risk-free rate. The **delta** — the sensitivity of the option value to the stock — is the derivative

$$ \Delta = \frac{\partial C}{\partial S} = \Phi(d_1). $$

The replication argument is this: if you hold $\Phi(d_1)$ shares against each short call and rebalance *continuously*, the random $dW_t$ terms in the option and the stock cancel exactly. Ito's lemma applied to $C(S,t)$ produces a $\frac{\partial C}{\partial S} dS$ term; holding $\frac{\partial C}{\partial S}$ shares produces an equal and opposite term; what remains is deterministic and grows at the risk-free rate. The hedging error is identically zero. This is the theoretical bedrock of all of options pricing.

Now break each assumption and watch the bedrock crumble.

**Continuous trading is impossible.** You rebalance at discrete times — daily, hourly, whatever. Between rebalances the stock drifts and diffuses while your hedge stays frozen at its last value. The cancellation is no longer exact; a residual second-order term survives. Expand the option value to second order: over a discrete step the option changes by roughly $\Delta \cdot \Delta S + \frac{1}{2} \Gamma (\Delta S)^2 + \Theta \, \Delta t$, where $\Gamma = \partial^2 C / \partial S^2$ is the gamma (the curvature) and $\Theta$ is the time decay. Your delta hedge cancels the first-order $\Delta \cdot \Delta S$ term but leaves the gamma term $\frac{1}{2}\Gamma(\Delta S)^2$ exposed. The variance of your discrete hedging error scales with $\Gamma^2$ and with the rebalancing interval — hedge less often and the error grows. This is real and unavoidable: discrete delta hedging has irreducible variance even in a perfect GBM world.

It is worth deriving the size of that residual, because it tells you exactly how much risk discreteness costs. Over a step of length $\Delta t$, the stock move is $\Delta S \approx \sigma S \sqrt{\Delta t}\, Z$ for a standard normal $Z$, so $(\Delta S)^2 \approx \sigma^2 S^2 \Delta t \, Z^2$. The uncancelled hedging P&L per step from the gamma term, after the theta offset that Black-Scholes builds in, is approximately $\frac{1}{2}\Gamma S^2 \sigma^2 \Delta t\,(Z^2 - 1)$ — the option's theta exactly pays for the *expected* gamma cost $\frac{1}{2}\Gamma S^2 \sigma^2 \Delta t$, so what survives is proportional to $(Z^2 - 1)$, a mean-zero random variable with variance 2. Summing the independent per-step contributions over $N = T/\Delta t$ steps, the variance of total hedging error scales like $N \cdot (\Gamma S^2 \sigma^2 \Delta t)^2 \propto \Delta t$. So halving the rebalance interval halves the error variance — but it *doubles* the number of trades and therefore the transaction-cost bill. That square-root-of-cost-versus-frequency trade-off is the precise mathematical seed of the no-trade band we derive in section 7. Black-Scholes, assuming continuous rebalancing, takes $\Delta t \to 0$ and the error vanishes; the real desk lives at $\Delta t = 1$ day and the error is the dominant risk.

**Volatility is not constant.** Real implied volatility moves; realized volatility clusters and spikes. The delta you compute with today's $\sigma$ is wrong tomorrow when $\sigma$ has jumped. Worse, the option has **vega** (sensitivity to volatility) that a delta hedge does nothing about. In the 2008 crisis and the March 2020 COVID crash, volatility tripled in days; desks running mechanical delta hedges discovered their "neutral" books were anything but.

**The stock jumps.** GBM has continuous paths — it never gaps. Real stocks gap on earnings, M&A, and macro surprises. When the stock jumps by 8% overnight, your delta hedge — set for a small continuous move — is catastrophically wrong, and you cannot rebalance mid-jump. Jump risk is the single largest source of left-tail hedging losses.

**Trading costs money.** Every rebalance pays the bid-ask spread plus commission. The more often you rebalance — which is what reduces gamma error — the more you pay. This is the central tension. Hedge continuously to kill variance and you pay infinite transaction cost; hedge rarely to save costs and your variance explodes. There is an optimal frequency, and *the Black-Scholes delta does not know it exists* because Black-Scholes assumes zero cost.

That last point is the crux, and it deserves to be stated as a theorem-flavored insight: **under discrete hedging with transaction costs, the optimal hedge is no longer the Black-Scholes delta.** Intuitively, if trading is expensive, you should let your hedge drift a little rather than chase the delta on every tick — you accept a bit more risk to save a lot of cost. The optimal hedge sits inside a *no-trade band* around the delta, and the width of that band depends on costs, gamma, and risk appetite. Black-Scholes gives you the center of the band; it says nothing about the band itself. Deep hedging learns the whole band — and more — directly from the objective.

## 3. Convex risk measures and the RL objective

If the goal of hedging is risk reduction rather than profit, we need to say precisely *which* risk we are minimizing. This is where the deep hedging objective both connects to and diverges from the standard RL return.

In vanilla RL, the agent maximizes expected cumulative reward, $J(\theta) = \mathbb{E}\big[\sum_t \gamma^t r_t\big]$. For hedging we could naively maximize expected terminal P&L, but a perfect hedge has zero expected P&L — expectation is the wrong target. What a risk manager actually cares about is the *bad tail*: how much can I lose in the worst cases? The right objective is a **risk measure** of terminal P&L.

Let $\mathrm{PnL}_T$ be the terminal profit-and-loss of the hedged book over one episode (premium received, minus option payoff, plus hedge P&L, minus total costs). We want to choose the hedging policy to minimize a risk functional $\rho(-\mathrm{PnL}_T)$ — the risk of the *loss* $-\mathrm{PnL}_T$. The deep hedging paper works with **convex risk measures**, a class with three properties that make them the right tool:

- **Monotonicity**: if one position always loses less than another, it is less risky.
- **Convexity**: diversifying (mixing two positions) never increases risk — $\rho(\lambda X + (1-\lambda) Y) \le \lambda \rho(X) + (1-\lambda)\rho(Y)$. This is what makes the optimization well-behaved.
- **Cash invariance**: adding a guaranteed \$1 reduces risk by exactly \$1, so $\rho$ is measured in money and tells you the capital buffer you need.

The workhorse convex risk measure in practice is **Conditional Value at Risk (CVaR)**, also called Expected Shortfall. CVaR at level $\alpha$ (say 95%) is the *average loss in the worst $(1-\alpha)$ fraction of outcomes*. Where Value at Risk (VaR) tells you "you will not lose more than \$X 95% of the time," CVaR tells you "in the 5% of times you breach VaR, your average loss is \$Y." CVaR is a coherent, convex risk measure; VaR is not convex and can punish diversification, which is why deep hedging uses CVaR.

CVaR has a beautiful variational form due to Rockafellar and Uryasev (2000) that makes it differentiable and trainable:

$$ \mathrm{CVaR}_\alpha(L) = \min_{w \in \mathbb{R}} \left\{ w + \frac{1}{1-\alpha} \, \mathbb{E}\big[(L - w)^+\big] \right\}, $$

where $L = -\mathrm{PnL}_T$ is the loss, $(x)^+ = \max(x, 0)$, and the minimizing $w^*$ turns out to be the VaR. This is the key that unlocks gradient-based training: CVaR is an expectation of a piecewise-linear function of P&L, and P&L is a differentiable function of the hedging actions (the hedge ratios multiply price changes), so the whole thing has a gradient with respect to the policy parameters. We can optimize it by stochastic gradient descent over sampled paths, jointly learning the policy parameters $\theta$ and the scalar $w$.

The **deep hedging objective** is therefore

$$ \min_{\theta, \, w} \; w + \frac{1}{1-\alpha} \, \mathbb{E}_{\text{paths}}\!\Big[\big(-\mathrm{PnL}_T(\theta) - w\big)^+\Big], $$

where $\mathrm{PnL}_T(\theta)$ depends on the policy $\pi_\theta$ that produced the sequence of hedge ratios along the path. Compare this to the RL return: it is still an expectation over trajectories of a function of the rewards. The difference is that the function is a convex risk measure (curved toward the tail) rather than a plain sum, and — the crucial twist — we will compute its gradient by *differentiating through the simulator*, not by the policy-gradient score-function trick. More on that in section 5. If you want the full derivation of the score-function estimator and why it is high variance, see our companion post on the policy gradient theorem at `/blog/machine-learning/reinforcement-learning/the-policy-gradient-theorem`; deep hedging is interesting precisely because it can *avoid* that estimator.

There is a second convex risk measure the original paper uses heavily and that is worth knowing because it is smoother to optimize: the **entropic risk measure**,

$$ \rho_\lambda(L) = \frac{1}{\lambda} \log \mathbb{E}\big[e^{\lambda L}\big], $$

where $\lambda > 0$ is risk aversion and $L = -\mathrm{PnL}_T$ the loss. The entropic measure is convex, cash-invariant, and — unlike CVaR's piecewise-linear kink — infinitely smooth, so its gradient is well-behaved everywhere, which can make training more stable. It penalizes large losses exponentially: as $\lambda$ grows the measure focuses ever harder on the worst outcomes, and in the limit $\lambda \to \infty$ it converges to the worst case, while $\lambda \to 0$ recovers the plain expectation. A practical recipe is to *warm up* training with the smooth entropic objective and then *fine-tune* on CVaR, getting the stability of one and the tail-targeting of the other. Both are convex risk measures, both are differentiable, and both slot into the same training loop — the only change is the line that computes the loss from the batch of terminal P&Ls.

The choice between them is a modeling decision, not a technicality. CVaR answers a regulator's question directly ("what is my average loss in the worst 5%?") and maps onto capital requirements, which is why risk desks prefer it. The entropic measure has no such clean interpretation but trains more smoothly and has a tidy connection to exponential utility, which appeals to academics. For production hedging, CVaR is the usual final objective; the entropic measure is the better optimization scaffold.

#### Worked example: reading a CVaR number

Suppose you simulate 100,000 hedged paths for your short call and collect the terminal P&L of each. Sort the losses from worst to best. CVaR(95%) is the average of the worst 5,000 losses (the worst 5%). Say a Black-Scholes delta hedge gives those worst 5,000 an average loss of \$1.00 per option (in units where the premium is around \$2.40). A deep hedger trained on the same paths might bring that average worst-case loss down to \$0.75 — a 25% reduction in tail risk — while keeping the mean P&L near zero. That \$0.25 per option, across a book of a million options, is \$250,000 of capital you no longer need to set aside against the tail. The mean did not move; the *shape* of the distribution did, and that is the entire game.

## 4. The MDP formulation for hedging

Now we make the MDP concrete enough to code. Figure 2 contrasts the two worldviews — the frictionless Black-Scholes recipe versus the cost-aware, model-free RL formulation — so you can see what each assumption costs.

![A before-and-after comparison contrasting Black-Scholes delta hedging assumptions against the model-free, transaction-cost-aware deep hedging reinforcement learning approach.](/imgs/blogs/deep-hedging-options-with-rl-2.png)

**State.** We feed the policy a vector at each step $t$:

$$ s_t = \big(\, S_t/K,\; \tau = T - t,\; \sigma^{\text{imp}}_t,\; \sigma^{\text{real}}_t,\; h_{t-1},\; \text{BS-}\Delta_t \,\big). $$

Including the current hedge $h_{t-1}$ is what makes costs tractable — the agent needs to know what it holds to compute the cost of changing it. Including the Black-Scholes delta as a feature is a pragmatic trick: it hands the network a strong prior (the frictionless optimum) so it only has to learn the *correction* for frictions, which speeds training dramatically. Moneyness $S_t/K$ and time-to-maturity $\tau$ together pin down the option's gamma and theta, which govern hedging error.

**Action.** The agent outputs a hedge ratio $h_t \in [0, 1]$ for a call (a sigmoid output bounds it). The *trade* is $\delta_t = h_t - h_{t-1}$ shares. We could instead output the trade directly; bounding the level is cleaner and prevents the policy from taking absurd positions.

**Reward.** This is the heart of the environment. Over step $t \to t+1$:

$$ r_t = \underbrace{h_t \,(S_{t+1} - S_t)}_{\text{hedge P\&L}} \;-\; \underbrace{c \, S_t \, |h_t - h_{t-1}|}_{\text{transaction cost}}, $$

where $c$ is the proportional cost rate (for example $c = 0.001$ for 10 basis points each way). The hedge P&L term is what your stock position earned; the cost term penalizes turnover. At expiry there is a terminal reward equal to the premium received minus the option payoff: $r_T = C_0 - \max(S_T - K, 0)$ plus the final liquidation of the hedge. Summing $r_t$ over the episode gives $\mathrm{PnL}_T$.

A clean way to write the total terminal P&L for a short call hedged with the sequence $\{h_t\}$ is

$$ \mathrm{PnL}_T = C_0 \;-\; \max(S_T - K, 0) \;+\; \sum_{t=0}^{T-1} h_t (S_{t+1} - S_t) \;-\; \sum_{t=0}^{T} c \, S_t \, |h_t - h_{t-1}|. $$

The first two terms are the option leg (collect premium, pay payoff). The third is the cumulative hedge gain. The fourth is the cumulative cost. Every term is a differentiable function of the actions $\{h_t\}$ — even the absolute value is differentiable almost everywhere — which is precisely why we can backpropagate.

**Episode.** From $t=0$ (write the option, set an initial hedge) to $t=T$ (settle), the option's whole life. Figure 4 shows that lifetime as a single RL episode with cost accruing along the way.

![A timeline showing an option's life as a single reinforcement learning episode from writing at delta one-half through rebalancing steps to settlement at expiry.](/imgs/blogs/deep-hedging-options-with-rl-4.png)

There is no discount factor $\gamma$ in the classic deep hedging setup — the horizon is finite and short (30 steps), and we care about the *undiscounted* terminal P&L, so $\gamma = 1$. This is one place hedging differs from infinite-horizon control. It also means credit assignment is over a short, well-defined window, which is part of why deep hedging trains so cleanly compared to long-horizon RL. (For the general credit-assignment problem and why long horizons are hard, see `/blog/machine-learning/reinforcement-learning/the-credit-assignment-problem`.)

Here is the environment as a Gymnasium-style class, so the formulation is unambiguous:

```python
import numpy as np
import gymnasium as gym
from gymnasium import spaces

class HedgingEnv(gym.Env):
    """One short European call, hedged with the underlying, with proportional cost."""
    def __init__(self, S0=100.0, K=100.0, T=30, sigma=0.2, r=0.0,
                 cost=0.001, dt=1/252):
        super().__init__()
        self.S0, self.K, self.T = S0, K, T
        self.sigma, self.r, self.cost, self.dt = sigma, r, cost, dt
        # state: [moneyness, tau, sigma, hedge_prev, bs_delta]
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(5,), dtype=np.float32)
        self.action_space = spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32)

    def _bs_delta(self, S, tau):
        from scipy.stats import norm
        if tau <= 0:
            return float(S > self.K)
        d1 = (np.log(S / self.K) + (self.r + 0.5 * self.sigma**2) * tau) \
             / (self.sigma * np.sqrt(tau))
        return float(norm.cdf(d1))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        self.S = self.S0
        self.h_prev = 0.0
        tau = self.T * self.dt
        delta = self._bs_delta(self.S, tau)
        obs = np.array([self.S / self.K, tau, self.sigma, self.h_prev, delta],
                       dtype=np.float32)
        return obs, {}

    def step(self, action):
        h = float(np.clip(action[0], 0.0, 1.0))
        cost = self.cost * self.S * abs(h - self.h_prev)   # turnover cost
        # GBM step (replace with Heston paths for realism, see section 6)
        z = self.np_random.standard_normal()
        S_next = self.S * np.exp((self.r - 0.5 * self.sigma**2) * self.dt
                                 + self.sigma * np.sqrt(self.dt) * z)
        hedge_pnl = h * (S_next - self.S)
        self.t += 1
        self.S = S_next
        self.h_prev = h
        terminated = (self.t >= self.T)
        reward = hedge_pnl - cost
        if terminated:
            payoff = max(self.S - self.K, 0.0)
            reward += -payoff   # we are short the call; premium handled outside
            reward -= self.cost * self.S * abs(0.0 - self.h_prev)  # liquidate hedge
        tau = (self.T - self.t) * self.dt
        delta = self._bs_delta(self.S, tau)
        obs = np.array([self.S / self.K, tau, self.sigma, self.h_prev, delta],
                       dtype=np.float32)
        return obs, reward, terminated, False, {}
```

You can drop this straight into Stable-Baselines3 and train PPO or SAC on it — and that is a perfectly valid way to learn a hedger. But it leaves performance on the table, because it throws away a structural gift that hedging hands us: the environment is *differentiable*. The next two sections exploit that.

## 5. The deep hedging architecture (Buehler et al. 2019)

The defining idea of deep hedging is not the neural network — it is *how the network is trained*. In standard model-free RL, the environment is a black box: you can sample transitions but you cannot differentiate through them, so you estimate the policy gradient with the score-function (REINFORCE) trick, which is notoriously high variance. In hedging, by contrast, *we built the environment*, and every step — the price update, the hedge P&L, the cost, the payoff — is a differentiable function we can express in PyTorch. That means we can backpropagate the risk measure straight through the entire simulated path into the policy weights. No score-function estimator, no critic, no replay buffer. Just `loss.backward()` through a 30-step rollout. This is the analytic-gradient or "pathwise" approach, and it is dramatically lower variance than score-function RL.

The policy itself is a network that maps the state $s_t$ to a hedge ratio $h_t$. Because hedging is path-dependent — the cost of today's trade depends on yesterday's hedge — a **recurrent** policy is natural. The original paper uses a feed-forward network per time step with the previous hedge fed in as a feature; a cleaner modern choice is an **LSTM** that carries a hidden state across the episode, so the network can learn its own running summary of the path. Figure 5 shows the architecture: option features and the prior hedge enter, the LSTM carries path memory, and a small head emits the next hedge ratio and the implied trade.

![A layered diagram of the deep hedging neural network mapping option features and prior hedge through an LSTM to the next hedge ratio and delta adjustment.](/imgs/blogs/deep-hedging-options-with-rl-5.png)

Here is the policy and the differentiable training loop in PyTorch. Read the loop carefully — the entire path P&L is built up as a differentiable tensor, the CVaR objective is computed on the batch of terminal P&Ls, and one `backward()` updates everything.

```python
import torch
import torch.nn as nn

class LSTMHedger(nn.Module):
    def __init__(self, n_features=5, hidden=64):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(),
                                  nn.Linear(32, 1), nn.Sigmoid())

    def forward(self, feats, hc):
        # feats: (batch, 1, n_features); hc: (h, c) LSTM state
        out, hc = self.lstm(feats, hc)
        h_t = self.head(out[:, -1, :])          # (batch, 1) in [0, 1]
        return h_t.squeeze(-1), hc

def cvar_loss(pnl, alpha=0.95):
    """Rockafellar-Uryasev CVaR of the LOSS (-pnl), jointly over w."""
    loss = -pnl
    w = torch.quantile(loss.detach(), alpha)    # warm-start VaR
    w = w.clone().requires_grad_(True)          # optimize jointly in practice
    excess = torch.relu(loss - w)
    return w + excess.mean() / (1 - alpha)
```

```python
def train_deep_hedger(paths, premium, K=100.0, cost=0.001,
                      epochs=200, batch=4096, device="cuda"):
    # paths: (N, T+1) simulated underlying price paths (e.g. Heston)
    model = LSTMHedger().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    paths = torch.as_tensor(paths, dtype=torch.float32, device=device)
    N, Tp1 = paths.shape
    T = Tp1 - 1
    for epoch in range(epochs):
        idx = torch.randint(0, N, (batch,), device=device)
        S = paths[idx]                              # (batch, T+1)
        h_prev = torch.zeros(batch, device=device)
        hc = None
        pnl = torch.full((batch,), premium, device=device)  # start with premium
        for t in range(T):
            tau = torch.full((batch,), (T - t) / 252.0, device=device)
            sigma = torch.full((batch,), 0.2, device=device)
            feats = torch.stack([S[:, t] / K, tau, sigma, h_prev,
                                 _bs_delta_t(S[:, t], K, tau, sigma)], dim=1)
            h_t, hc = model(feats.unsqueeze(1), hc)
            pnl = pnl - cost * S[:, t] * (h_t - h_prev).abs()   # cost
            pnl = pnl + h_t * (S[:, t + 1] - S[:, t])           # hedge P&L
            h_prev = h_t
        payoff = torch.relu(S[:, -1] - K)
        pnl = pnl - payoff                                      # short the call
        pnl = pnl - cost * S[:, -1] * h_prev.abs()              # liquidate
        loss = cvar_loss(pnl, alpha=0.95)
        opt.zero_grad(); loss.backward(); opt.step()
        if epoch % 20 == 0:
            print(f"epoch {epoch:3d}  CVaR95 {loss.item():.4f}  "
                  f"meanPnL {pnl.mean().item():.4f}")
    return model
```

Two things to internalize. First, there is *no environment in the RL sense* here — `paths` is a pre-simulated batch, and the loop replays them while the policy makes decisions, exactly like backpropagation-through-time for an RNN. The simulator is the environment, and it is differentiable, so the policy gradient is *exact* (up to Monte Carlo sampling of paths), not a high-variance estimate. Second, the loss is the CVaR of terminal P&L over the whole batch — risk is a property of the *distribution* of outcomes, so it must be computed across paths, not per path. This is why deep hedging trains on minibatches of thousands of paths at once.

Why does this beat plain model-free RL like PPO on the same problem? Variance. PPO's policy gradient is a Monte Carlo estimate that needs many samples to average out the noise of the score function; deep hedging gets the gradient analytically through the differentiable path, so it converges in far fewer gradient steps and to a tighter optimum. When you *can* differentiate the environment, you should — and hedging is one of the rare real problems where you can. If your market model is a black box you can only sample (a historical-bootstrap environment, say, or a learned generative model you treat as opaque), you fall back to model-free RL on the `HedgingEnv` above; the formulation is identical, only the gradient estimator changes.

Why recurrence, concretely? A memoryless feed-forward policy must reconstruct everything it needs from the current state vector alone. For a vanilla call hedged on the underlying, you *can* make that work by stuffing the previous hedge into the state — which is exactly why our state vector includes $h_{t-1}$. But the moment the product is path-dependent, a feed-forward net would need the entire relevant history packed into the state by hand: for a barrier option, has the barrier been touched; for an Asian option, the running average so far; for a hedger that wants to detect a vol regime shift, several lags of realized vol. An LSTM learns its own running summary of whatever path features matter, so you do not have to hand-engineer the state for every product. On a plain call the recurrent and feed-forward hedgers are nearly identical; on barriers and Asians the recurrent one wins decisively because the relevant state genuinely is the path, not a snapshot. This is the same reason recurrence helps in partially-observed RL generally: when the Markov state is not fully observable from the current input, memory recovers it.

A second design subtlety is **reward shaping versus terminal-only risk**. You might be tempted to give the agent a per-step reward equal to the step's hedging P&L minus cost and let it maximize the sum. That works for the *expectation* but it is subtly wrong for the *risk measure*, because CVaR is a nonlinear functional of the terminal P&L distribution and does not decompose into a sum of per-step risks. The correct formulation computes the risk measure on the *terminal* P&L of each path, after the whole episode, which is exactly what the training loop does — it accumulates per-step P&L into a terminal tensor and applies CVaR once at the end. Per-step reward shaping is fine for a *value baseline* if you go the model-free route, but the objective itself must be evaluated on terminal P&L across the batch. Getting this wrong — applying CVaR per step and summing — is one of the more common deep-hedging implementation bugs, and it silently produces a hedger that optimizes the wrong thing.

For completeness, here is the model-free fallback using Stable-Baselines3 on the exact same MDP — useful when the environment is non-differentiable, or as a sanity baseline to confirm the pathwise hedger is actually better:

```python
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor

def make_env():
    return Monitor(HedgingEnv(cost=0.001, sigma=0.2))

env = SubprocVecEnv([make_env for _ in range(16)])   # 16 parallel envs
model = SAC(
    "MlpPolicy", env,
    learning_rate=3e-4,
    buffer_size=1_000_000,
    batch_size=512,
    gamma=1.0,            # finite undiscounted horizon: do NOT discount
    tau=0.005,
    ent_coef="auto",      # SAC's entropy temperature, auto-tuned
    train_freq=1,
    gradient_steps=1,
    policy_kwargs=dict(net_arch=[128, 128]),
    verbose=1,
)
model.learn(total_timesteps=2_000_000)
```

Note `gamma=1.0`: hedging is a finite, short, undiscounted-horizon problem, and discounting would tell the agent to under-weight the terminal payoff settlement, which is the single most important reward in the episode. This is a subtle but common bug — the SB3 default $\gamma = 0.99$ silently corrupts hedging by discounting the very payoff you are hedging against. SAC is the right model-free choice here because the action is continuous (a hedge ratio) and SAC is sample-efficient and stable on continuous control; PPO works too but needs more samples. Expect SAC to need on the order of two million environment steps to approach what the pathwise hedger reaches in a few hundred gradient steps — a stark illustration of the value of an exact gradient.

The hyperparameters that actually move the needle for the pathwise deep hedger are few, and worth a table:

| Hyperparameter | Effect | Typical range |
| --- | --- | --- |
| Batch (paths/step) | Tail estimate quality; small batches mis-estimate CVaR | 2,048 – 16,384 |
| CVaR level $\alpha$ | How deep into the tail you optimize | 0.90 – 0.99 |
| Cost rate $c$ | Width of the learned no-trade band | 5 – 50 bps |
| LSTM hidden size | Capacity to model path dependence | 32 – 128 |
| Learning rate | Convergence speed vs stability | 1e-4 – 3e-3 |
| Rebalance steps $T$ | Granularity vs cost; more steps = more decisions | 10 – 60 |

The two that surprise people: a *too-small batch* makes the CVaR estimate of the worst 5% so noisy that training is unstable — the tail simply is not populated by a few hundred paths — and a *higher cost rate widens the learned band*, which you can see directly by plotting the hedger's action against the delta as you sweep $c$.

## 6. Training on Monte Carlo paths

Where do the paths come from? This is the part that determines whether your hedger is any good in the real world, because the policy can only be as smart as the market dynamics you train it on. Train on GBM and you will rediscover the Black-Scholes delta (plus a cost correction) — which is fine, but unexciting. The payoff comes from training on dynamics that GBM cannot capture: **stochastic volatility** and **jumps**.

The standard richer model is the **Heston** stochastic-volatility model, where the variance $v_t$ is itself a mean-reverting random process:

$$ dS_t = \mu S_t \, dt + \sqrt{v_t}\, S_t \, dW^S_t, \qquad dv_t = \kappa(\theta - v_t)\, dt + \xi \sqrt{v_t}\, dW^v_t, $$

with $\kappa$ the speed of mean reversion, $\theta$ the long-run variance, $\xi$ the vol-of-vol, and a correlation $\rho$ between the two Brownian motions $dW^S$ and $dW^v$ (typically negative — when stocks fall, vol rises). Heston produces the volatility smile, vol clustering, and fat tails that GBM lacks, and crucially it makes the *option's vega matter*, so a hedger trained on Heston learns to account for volatility moves a delta hedge ignores.

Here is a vectorized Heston simulator that produces the `paths` array the trainer above consumes. Figure 3 shows where it sits in the training pipeline — the path generator is the environment, feeding states into the LSTM and P&L into the CVaR loss.

![A pipeline diagram of deep hedging training from a Heston Monte Carlo path generator through state extraction, the LSTM policy, P&L computation, the CVaR risk measure, and backpropagation.](/imgs/blogs/deep-hedging-options-with-rl-3.png)

```python
import numpy as np

def simulate_heston(N=100_000, T=30, S0=100.0, v0=0.04, kappa=2.0,
                    theta=0.04, xi=0.3, rho=-0.7, r=0.0, dt=1/252, seed=0):
    rng = np.random.default_rng(seed)
    S = np.empty((N, T + 1), dtype=np.float64); S[:, 0] = S0
    v = np.empty((N, T + 1), dtype=np.float64); v[:, 0] = v0
    for t in range(T):
        z1 = rng.standard_normal(N)
        z2 = rho * z1 + np.sqrt(1 - rho**2) * rng.standard_normal(N)
        v_t = np.maximum(v[:, t], 0.0)                  # full-truncation scheme
        S[:, t + 1] = S[:, t] * np.exp((r - 0.5 * v_t) * dt
                                       + np.sqrt(v_t * dt) * z1)
        v[:, t + 1] = (v[:, t] + kappa * (theta - v_t) * dt
                       + xi * np.sqrt(v_t * dt) * z2)
    return S, v

paths, var_paths = simulate_heston()
# add a jump overlay for crash-scenario training (Merton-style)
def add_jumps(S, intensity=0.02, mean=-0.05, std=0.05, seed=1):
    rng = np.random.default_rng(seed)
    N, Tp1 = S.shape
    out = S.copy()
    for t in range(1, Tp1):
        jump = rng.random(N) < intensity
        sizes = np.exp(rng.normal(mean, std, N))
        out[:, t] *= np.where(jump, sizes, 1.0)
        out[:, t] = out[:, t] * (out[:, t-1] / S[:, t-1])  # carry jump forward
    return out
```

The path generator *is* the environment, and this is the deepest practical difference between hedging RL and game RL. In CartPole you have one physics; here you choose the physics, and that choice is a modeling decision with real consequences. Train only on calm Heston paths and your hedger will be blindsided by a crash. Mix in jump-heavy and high-vol-of-vol regimes — including deliberately nasty crash scenarios — and the hedger learns to hold a more defensive hedge when the state signals danger. Domain randomization, the same trick that makes sim-to-real robotics work, applies directly: randomize $\kappa, \theta, \xi, \rho$ and the jump intensity across the training batch so the policy is robust to model misspecification, because you do not know the true dynamics and you do not want a hedger that only works under one parameter set.

**Sample efficiency.** Because the gradient is pathwise (exact) rather than score-function (estimated), deep hedging is remarkably sample-efficient *in gradient steps*: a few hundred epochs over batches of a few thousand paths typically converges. But it is sample-*hungry in paths* in absolute terms, because the CVaR objective only depends on the tail — to estimate the worst 5% accurately you need many paths in that tail, so 100,000+ paths is normal, and crash scenarios may need importance sampling to populate the tail. This is the opposite of game RL's sample profile, and it is worth understanding: the bottleneck is not interaction count, it is tail coverage.

The tail-coverage problem deserves a concrete fix because it bites every serious deployment. With a 95% CVaR and a batch of 4,096 paths, only about 205 paths fall in the tail the loss actually depends on — the gradient is driven by a couple hundred samples per step, and if crashes are rare in your simulator, those tail paths may not even contain the scenarios you most fear. Two fixes work. First, **stratified or importance sampling**: deliberately oversample high-vol and jump paths during training, then reweight the loss so the CVaR estimate stays unbiased — this floods the tail with the scenarios that matter without distorting the objective. Second, **a larger batch with a higher CVaR level annealed over training**: start at $\alpha = 0.90$ (a fatter, better-sampled tail) and tighten toward $\alpha = 0.99$ as training stabilizes, so early gradients are well-estimated and late ones target the true tail. Both turn a noisy, crash-blind hedger into one whose tail behavior you can actually trust, and both are cheap relative to the cost of a hedger that looks great in aggregate and falls apart exactly when a crash arrives.

#### Worked example: gradient steps to convergence

On 100,000 Heston paths with 30 daily steps and 10 bps cost, the LSTM hedger above typically drives the CVaR(95%) loss from its random-init value down to within a few percent of its floor in around 150–250 Adam steps at learning rate $10^{-3}$, batch 4,096. Each step does a 30-step BPTT rollout over 4,096 paths — milliseconds on a single GPU. Total training is minutes, not hours. Contrast that with running PPO on the same `HedgingEnv`: PPO needs on the order of millions of environment steps to reach comparable CVaR because its gradient is a noisy estimate, and it often plateaus at a worse optimum. The pathwise gradient is the single biggest lever for both speed and final quality when the environment is differentiable.

## 7. Transaction costs and the discrete-hedging band

Transaction costs are where deep hedging earns its keep, so it is worth understanding both the classical theory and what the network discovers on its own.

The cost model matters. The two common forms are **proportional** cost, $c \cdot S \cdot |\delta|$ (a fixed fraction of notional traded — the bid-ask spread, this is what the code uses), and **quadratic** or market-impact cost, $\propto |\delta|^2$ (large trades move the price against you). Proportional cost is the standard assumption and the one with the cleanest theory; quadratic cost matters when you trade size. The qualitative conclusion — that you should not chase the delta on every tick — holds for both.

Classical theory gives two closed-form approximations for the optimal hedge under proportional costs:

- **Leland (1985)** adjusts the volatility used in the delta to account for costs, effectively widening or shrinking the hedge by a factor that depends on the cost rate and rebalancing frequency. It is a clever hack: hedge with a *modified* Black-Scholes delta computed at an adjusted volatility $\hat\sigma^2 = \sigma^2 (1 + \kappa)$, where $\kappa$ rises with the cost rate. Leland still rebalances every period, just with a different delta.
- **Whalley-Wilmott (1997)** derives, via asymptotic analysis for small costs, an explicit **no-trade band** around the Black-Scholes delta. You only trade when your hedge drifts outside a band of half-width proportional to $\big(\frac{3 c\, S\, \Gamma^2 \sigma^2}{2\lambda}\big)^{1/3}$, where $\lambda$ is risk aversion and $\Gamma$ the gamma. Inside the band you do nothing; you trade only to the nearest band edge when you exit it. This is the canonical result: **the optimal policy is a no-trade band, not a target delta.**

The remarkable thing is that **deep hedging rediscovers the no-trade band from the objective alone**, without being told the Whalley-Wilmott formula. Train the LSTM with a nonzero cost and inspect its actions: it does *not* track the delta tick-for-tick. When the delta moves a little, the network leaves the hedge unchanged (saving the cost); when the delta moves a lot, the network trades, but often only partway toward the new delta. Plot the network's hedge against the Black-Scholes delta and you see a band — a region where the hedge stays put — whose width scales with the cost rate exactly as Whalley-Wilmott predicts. The network learned the bandwidth problem's solution from first principles, by minimizing CVaR. And because it learned it rather than being handed a small-cost asymptotic formula, it stays sensible at *large* costs and under jumps and stochastic vol where the closed-form approximations break down.

This is the cleanest demonstration of why RL beats hand-derived rules here: the no-trade band is not a heuristic bolted onto Black-Scholes, it is the *emergent optimum* of the actual objective on the actual dynamics. Change the cost rate, the gamma profile, or the vol regime, and the band reshapes itself automatically. You never re-derive a formula.

| Approach | Hedge rule | Handles jumps / stoch vol | Handles large costs | Needs a formula |
| --- | --- | --- | --- | --- |
| Black-Scholes delta | Track $\Phi(d_1)$ every step | No | No (assumes zero) | Closed form |
| Leland (1985) | Delta at adjusted vol | Partly | Small costs only | Closed form |
| Whalley-Wilmott (1997) | No-trade band, asymptotic | No | Small costs only | Closed form |
| Deep hedging | Learned policy on paths | Yes (if in training paths) | Yes | None — learned |

## 8. Volatility regime adaptation

The 2008 crisis and the March 2020 COVID crash are the case the textbook delta hedge fails hardest. In both, realized volatility spiked 3–5× in days, correlations broke, and stocks gapped. A mechanical delta hedge, computed off a stale implied vol and rebalanced once a day, was both *wrong* (the delta was computed at the wrong vol) and *too slow* (the stock gapped between rebalances). Desks that survived did so by widening hedges, trading more defensively, and accounting for vega — exactly the behaviors a delta hedge does not encode.

A deep hedger trained on Heston paths *with* high-vol-of-vol and jump regimes learns these behaviors. Because its state includes realized vol and the implied vol, and because it was trained against paths where vol spikes, the policy maps "vol is rising, gamma is high" to "hold a more conservative hedge and rebalance sooner." It adapts in real time because adaptation is just the policy responding to its state — there is no formula to update, no parameter to recalibrate mid-crisis. This is the same property that makes RL policies robust in robotics: a policy is a function of state, so it generalizes across regimes it saw in training.

The empirical results from the deep hedging literature and from bank implementations are consistent on the headline: on stressed scenarios — crash paths, vol-spike paths — a deep hedger reduces CVaR(95%) by roughly **15–30%** versus a Black-Scholes delta hedge at the same average transaction cost, and the gap *widens* as costs rise and as the dynamics depart further from GBM. On calm, near-GBM paths with tiny costs the gap shrinks toward zero — which is exactly right, because in that regime Black-Scholes is nearly optimal and there is nothing to improve. The deep hedger does not beat the delta everywhere; it beats it precisely where the delta's assumptions fail, and it matches it where they hold. That honesty about *when* the edge exists is the mark of a real result rather than a benchmark artifact.

#### Worked example: a crash scenario backtest

Construct a test set of 20,000 paths with a jump-diffusion overlay calibrated to a COVID-like crash: a 2% daily jump probability with mean jump size $-5\%$ on top of Heston with $\xi = 0.4$. Hedge a one-month ATM call with 10 bps cost. A Black-Scholes delta hedge on this set might post CVaR(95%) of about \$1.10 per option (the jumps blow out the worst 5%). A deep hedger trained on a jump-augmented path distribution posts CVaR(95%) of about \$0.82 — a 25% tail reduction — at slightly *lower* average cost, because it stops over-trading in the calm stretches and saves the turnover for when it matters. Mean P&L is statistically indistinguishable between the two. The deep hedger did not predict the crash; it learned a policy whose tail behavior is more robust to crashes, which is the most you can ask of any hedge.

## 9. Comparison: delta vs gamma vs RL

It helps to place deep hedging against the full ladder of classical hedges, because the right baseline depends on how much machinery you are willing to run.

**Delta hedging** neutralizes first-order exposure to the stock — the $\Delta \cdot \Delta S$ term. It is the simplest and the standard. Its blind spot is the gamma term and everything beyond.

**Gamma hedging** adds a second instrument (another option) to neutralize the second-order $\frac{1}{2}\Gamma(\Delta S)^2$ term, so the book is robust to larger stock moves. It is more expensive — you trade two instruments — and it introduces its own vega and higher-order exposures. Desks running large convex books gamma-hedge; it is genuinely useful but it does not address transaction costs or jumps, and it requires a liquid second option to trade.

**Vega hedging** neutralizes exposure to implied volatility by trading options of different maturities, important when you carry a large vol position.

**Deep hedging** does not pick a Greek to neutralize; it directly minimizes the CVaR of terminal P&L given whatever instruments you let it trade. If you only let it trade the underlying, it learns the best *first-order* hedge under costs (a no-trade band, plus jump-aware defensiveness). If you let it trade the underlying *and* a second option, it learns to use the second instrument when the gamma cost-benefit warrants it — effectively discovering when to gamma-hedge, without you specifying the rule. The same framework subsumes delta, gamma, and vega hedging as special cases of "minimize risk with the instruments available." Figure 6 lays out the comparison on the metrics that matter.

![A matrix comparing Black-Scholes delta, gamma hedging, Leland-adjusted, and deep hedging across CVaR, mean P&L, transaction cost, and volatility-regime robustness.](/imgs/blogs/deep-hedging-options-with-rl-6.png)

The metrics to compare on are the terminal P&L distribution and its summaries: **CVaR(95%)** (the tail you are trying to shrink), **mean P&L** (should be near zero and comparable across methods — if a method has higher mean P&L it is taking a directional bet, not hedging), and **mean transaction cost** (turnover). A fair comparison fixes the cost model and the test path distribution and reports all three. Figure 7 shows the headline visual: the deep hedger thins the left tail of the P&L distribution — lower CVaR — while leaving the mean essentially unchanged.

![A before-and-after comparison of the terminal profit-and-loss distribution showing delta hedging with a fat left tail versus deep hedging with a thinner left tail and lower CVaR.](/imgs/blogs/deep-hedging-options-with-rl-7.png)

Here is an evaluation harness that produces the comparison table, so the numbers are reproducible rather than asserted:

```python
import numpy as np

def bs_delta_hedge(paths, K=100.0, sigma=0.2, cost=0.001, dt=1/252):
    from scipy.stats import norm
    N, Tp1 = paths.shape; T = Tp1 - 1
    h_prev = np.zeros(N); pnl = np.zeros(N)
    for t in range(T):
        tau = (T - t) * dt
        d1 = (np.log(paths[:, t] / K) + 0.5 * sigma**2 * tau) / (sigma * np.sqrt(tau))
        h = norm.cdf(d1)
        pnl -= cost * paths[:, t] * np.abs(h - h_prev)
        pnl += h * (paths[:, t + 1] - paths[:, t])
        h_prev = h
    pnl -= np.maximum(paths[:, -1] - K, 0.0)
    pnl -= cost * paths[:, -1] * np.abs(h_prev)
    return pnl

def cvar(pnl, alpha=0.95):
    loss = -pnl
    var = np.quantile(loss, alpha)
    return loss[loss >= var].mean()

def report(pnl, name, premium):
    p = pnl + premium
    print(f"{name:14s}  meanPnL {p.mean():+.4f}  std {p.std():.4f}  "
          f"CVaR95 {cvar(p):.4f}")

# paths, var_paths = simulate_heston(); premium computed by MC pricing of the call
# report(bs_delta_hedge(paths), "BS delta", premium)
# report(deep_hedge_pnl(model, paths), "Deep hedging", premium)
```

Run this on the same Heston-plus-jumps test set for both methods and you get an apples-to-apples table: same paths, same cost, same premium, only the hedging policy differs. That is the only honest way to claim a CVaR improvement.

## 10. Extensions: cross-asset, American, and exotic options

The reason deep hedging matters beyond a single call is that the framework is *general* in a way closed-form deltas are not. Anything you can simulate and price, you can hedge with the same machinery.

**Exotic payoffs.** Barrier options (knock-in/knock-out), Asian options (payoff on the average price), lookbacks, and autocallables often have no clean closed-form delta, or have deltas that blow up near barriers where the closed-form hedge is impossible to execute. Deep hedging does not care — it just needs the payoff function to compute the terminal reward and a path simulator. For a knock-out call, the terminal reward is the call payoff times the indicator that the barrier was never breached; that is differentiable enough to train on, and the network learns to de-hedge as the spot approaches the barrier in a way no static delta can.

**Cross-asset / basket options.** A basket option pays on a weighted sum of correlated underlyings. Hedging it requires trading several assets with the right ratios, and the optimal ratios depend on the correlation structure, which moves. Deep hedging handles this by making the state and action vector-valued: the state includes all the underlyings' features plus the correlation regime, and the action is a vector of hedge ratios across assets. The network learns the cross-hedge — including using a liquid index to partially hedge an illiquid name — directly from the joint path simulator. This is genuinely hard to do with hand-derived Greeks because the cross-gammas multiply combinatorially.

**American options and early exercise.** American options can be exercised any time before expiry, which adds a *second* decision to the MDP: at each step, exercise or continue? This is exactly the structure of optimal stopping, and it is a clean RL problem — the action space becomes (hedge ratio, exercise/hold), and the policy learns both the hedge and the exercise boundary. The classical approach (Longstaff-Schwartz least-squares Monte Carlo) is a special case of value-based RL for the exercise decision; deep hedging unifies it with the hedging decision in one policy.

**Barrier options as RL.** Beyond pricing, the *path-dependence* of barriers makes them a poster child for recurrent policies: whether the barrier has been touched is part of the state and must be carried through the episode, which an LSTM does naturally. This is the same recurrence that makes the LSTM hedger better than a memoryless feed-forward net on plain calls, amplified.

The unifying point: deep hedging is "differentiable simulation + a risk objective + a policy network." Swap the payoff, swap the simulator, extend the action space, and the same training loop handles a new product. That generality — one framework, many products — is why banks invested in it, and it is the same modularity that makes RL attractive across robotics, games, and language: define the environment and the reward, reuse the machinery.

## 11. Deployment pitfalls and honest trade-offs

Before the case studies, a section that the marketing decks skip: deep hedging has real failure modes, and a staff engineer who deploys it without knowing them will get burned. I have watched a beautifully trained hedger post a worse P&L tail in production than the dumb delta it replaced — not because the method is wrong, but because of three traps.

**The train/test distribution gap is the whole ballgame.** A deep hedger is a function fitted to the path distribution you trained it on. If the real market draws from a different distribution — a regime your simulator never produced — the policy is extrapolating, and learned policies extrapolate badly. The delta, for all its flaws, is a *closed-form* function of observable inputs; it does not have a training distribution to fall off the edge of. The mitigation is aggressive domain randomization (train across a wide spread of Heston parameters and jump intensities) plus a hard backstop: clip the network's hedge so it can never stray more than some bound from the Black-Scholes delta. That clip costs you a little of the learned edge but guarantees the hedger degrades gracefully to "approximately delta hedging" in an unseen regime rather than doing something insane. In production, the clip is non-negotiable.

**Model risk compounds.** With Black-Scholes you have one model risk — that GBM is wrong, which everyone knows and prices in. With deep hedging you have *two*: the simulator might be wrong (Heston is also just a model), and the network might have learned spurious structure from simulator artifacts. A network trained on a Heston simulator with a subtle discretization bias can learn to exploit that bias — posting great simulated CVaR that evaporates on real data. The defense is to validate on *held-out historical data*, not just held-out simulated paths: train on the simulator, but report the final CVaR on a backtest over real historical underlying prices. If the historical CVaR improvement is much smaller than the simulated one, your hedger has overfit the simulator, and you trust the historical number.

**Explainability and the risk committee.** A delta is $\Phi(d_1)$; anyone on the desk can recompute it and a regulator accepts it on sight. A learned LSTM policy is a black box, and "the network said to hold 0.43 shares" is not an answer a risk committee will sign off on for a large book. The practical compromise is to deploy the deep hedger as a *recommendation* alongside the delta, log both, and let it run live in shadow mode for months while you accumulate evidence that its realized tail is genuinely thinner. Only after that does it get capital. Treat the explainability cost as real: a 20% CVaR improvement that nobody will approve is worth zero.

**Overfitting the tail.** Because CVaR only cares about the worst 5%, a network can fixate on a handful of extreme training paths and contort its policy to handle those specific paths at the expense of the bulk. Regularization (weight decay, dropout in the LSTM), a large batch so the tail is well-sampled, and the entropic warm-up from section 3 all help. Watch the gap between train and validation CVaR; if it widens, you are overfitting the tail and should regularize harder or sample more paths.

None of these kill the method — they shape *how* you deploy it. The honest summary is that deep hedging trades the delta's interpretable-but-wrong assumptions for a learned-but-data-hungry policy, and that trade pays off when, and only when, you have a credible simulator, a validation backtest, and a backstop clip.

#### Worked example: catching simulator overfit

Train two hedgers identically on 100,000 Heston paths, but evaluate one on 20,000 *fresh Heston* paths and the other on a *historical bootstrap* of real S&P 500 daily returns over a decade. Suppose the fresh-Heston CVaR(95%) improvement over the delta is a glorious 28%, but the historical-bootstrap improvement is only 11%. That 17-point gap is your simulator overfit, quantified. The 11% is the number you take to the risk committee; the 28% is the number you quietly delete from the deck. A hedger whose historical improvement is *negative* — worse than the delta on real data despite beating it in simulation — is a hedger that learned the simulator, not the market, and it never ships.

## 12. Case studies

**Buehler et al. (2019), the founding result.** The original "Deep Hedging" paper (Buehler, Gonon, Teichmann, Wood, *Quantitative Finance*) trained neural-network hedgers on simulated Heston paths with proportional transaction costs and CVaR/entropic risk objectives. Their central finding: with nonzero costs, the learned hedger systematically *beats* the discrete Black-Scholes delta hedge on the chosen risk measure, and the learned policy exhibits a no-trade band consistent with the Whalley-Wilmott asymptotics — recovered from the objective, not imposed. They also showed the framework extends cleanly to portfolios of options and to the entropic risk measure, demonstrating that the approach is about the *objective and the simulator*, not a single product. This paper is the reference point for everything in this post.

**Bank implementations (JP Morgan and others).** Following the paper, quantitative teams at several banks — JP Morgan's deep hedging work being the most publicly discussed — built production-oriented deep hedgers for exotic and structured products where closed-form deltas are weakest. The reported pattern matches the theory: meaningful CVaR reductions versus Greek-based hedging on cost-heavy, path-dependent books, with the largest gains on products that classical Greeks handle poorly. Public figures are necessarily approximate and vary by product and cost assumption, but the qualitative result — a double-digit-percentage tail-risk reduction at comparable cost — is consistent across reports. Treat specific percentages as illustrative of the magnitude, not precise benchmarks.

**Reinforcement learning for execution, a sibling problem.** The same MDP framing powers optimal trade execution — splitting a large order over time to minimize market impact, the Almgren-Chriss problem solved with RL. It shares deep hedging's structure (sequential decisions, cost-vs-risk trade-off, terminal objective) and is a useful sanity check that the framing generalizes. If you want the trading-side treatment of impact and execution, see our options-desk view of staying neutral under cost at `/blog/trading/options-volatility/delta-hedging-in-practice-the-cost-and-slippage-of-staying-neutral`.

**A reproducible mini-benchmark.** You do not need a bank to reproduce the core result. With the code in this post — `simulate_heston`, `train_deep_hedger`, `bs_delta_hedge`, and the `cvar` reporter — train on 100,000 Heston-plus-jumps paths and evaluate both hedgers on a held-out 20,000-path test set at 10 bps cost. The expected outcome is a deep-hedger CVaR(95%) roughly 15–25% below the BS delta's, mean P&L within noise of each other, and average cost equal or lower for the deep hedger. That is a one-GPU, few-minute experiment, and it is the cleanest way to see the effect with your own eyes rather than taking a number on faith.

## 13. When to use deep hedging (and when not to)

Deep hedging is a power tool, and like any power tool it is overkill for simple jobs and indispensable for hard ones. Figure 8 is the decision tree; here is the reasoning behind it.

![A decision tree for when to use deep hedging based on exotic payoffs, significant transaction costs, and whether geometric Brownian motion is clearly wrong.](/imgs/blogs/deep-hedging-options-with-rl-8.png)

**Use deep hedging when:**

- **The payoff is exotic** — barriers, baskets, autocallables, anything without a clean executable closed-form delta. This is the strongest case; there is often no good classical alternative.
- **Transaction costs are significant** — wide spreads, illiquid underlyings, or high rebalancing frequency. The no-trade band the network learns is worth real money here, and it adapts to the cost level automatically.
- **GBM is clearly wrong** — fat tails, jumps, stochastic vol, regime shifts. Train on a path distribution that captures the badness and the hedger learns defensive behavior the delta cannot.
- **You can simulate the dynamics well** — deep hedging is only as good as its training paths. If you have a calibrated Heston/jump model or a rich historical bootstrap, you are in business.

**Stick with Black-Scholes delta when:**

- **The option is a liquid vanilla European** in a near-GBM, low-cost market. The delta is nearly optimal there, it is free, it is interpretable, and a regulator or risk committee understands it instantly. Deep hedging's edge shrinks to noise in this regime, and you pay it in complexity and model risk.
- **You cannot simulate the dynamics credibly.** A deep hedger trained on a wrong model is confidently wrong. If your only model is GBM, the delta already encodes its optimum — adding a network buys nothing and adds failure modes.
- **Interpretability or regulatory sign-off is paramount.** A learned policy is harder to explain than $\Phi(d_1)$. On a desk where every hedge must be justified to risk and audit, the explanatory cost of a black-box policy can outweigh a modest CVaR gain.

The deeper principle, and the one that ties back to the whole series: **model-free or differentiable-simulation RL earns its complexity exactly when the closed-form assumptions break and you can sample the real dynamics.** When you have a known, tractable model (GBM, low cost), use the closed form — it *is* the optimal policy, derived not learned. When the model is rich, the frictions are real, and the payoff is gnarly, no closed form exists, and learning the policy from a good simulator is the only path to the optimum. Deep hedging is the financial instance of that universal RL trade-off.

## Key takeaways

1. **Hedging is an MDP.** State = Greeks plus market plus current hedge; action = new hedge ratio; reward = P&L minus transaction cost; episode = the option's life. Once you see it this way, RL is the natural tool.
2. **The Black-Scholes delta is optimal only in its own frictionless world.** Under discrete hedging with costs, the optimal hedge is a no-trade band around the delta, not the delta itself.
3. **The right objective is a convex risk measure, not expected return.** CVaR (Expected Shortfall) targets the bad tail, is coherent and convex, and via the Rockafellar-Uryasev form is differentiable and trainable.
4. **Deep hedging differentiates through the simulator.** Because the path P&L is a differentiable function of the actions, you backpropagate the risk measure exactly — far lower variance than the score-function policy gradient, so it converges in hundreds of steps.
5. **The simulator is the environment, and you choose it.** Train on Heston-plus-jumps with domain randomization so the hedger is robust to model misspecification and to crashes; train only on GBM and you just relearn the delta.
6. **The network rediscovers the no-trade band.** Without being told the Whalley-Wilmott formula, a cost-aware deep hedger learns to leave the hedge alone inside a band and trade only when the drift is large enough to justify the cost.
7. **The edge is concentrated where assumptions fail.** Expect a 15–30% CVaR reduction versus the delta on cost-heavy, jumpy, high-vol regimes — and roughly zero edge on liquid vanillas in calm markets. Honest hedging means knowing which regime you are in.
8. **The framework generalizes by swapping the payoff and simulator.** Exotics, baskets, American early-exercise, and barriers are all the same training loop with a different reward and state.
9. **Use the closed form when it is the optimum.** For liquid vanilla European options in low-cost near-GBM markets, the Black-Scholes delta is nearly optimal, interpretable, and free — do not over-engineer.
10. **Differentiable simulation is the rare gift; use it when you have it.** When you can differentiate the environment, the analytic gradient beats model-free RL on both speed and final quality. When you cannot, fall back to PPO/SAC on the same MDP.

## Further reading

- Buehler, Gonon, Teichmann, Wood, "Deep Hedging," *Quantitative Finance* (2019) — the founding paper; read it for the risk-measure formulation and the no-trade-band results.
- Rockafellar and Uryasev, "Optimization of Conditional Value-at-Risk," *Journal of Risk* (2000) — the variational form of CVaR that makes it differentiable and trainable.
- Whalley and Wilmott, "An Asymptotic Analysis of an Optimal Hedging Model for Option Pricing with Transaction Costs," *Mathematical Finance* (1997) — the classical no-trade band.
- Leland, "Option Pricing and Replication with Transaction Costs," *Journal of Finance* (1985) — the adjusted-volatility delta.
- Sutton and Barto, "Reinforcement Learning: An Introduction" (2nd ed., 2018) — the MDP and policy foundations underneath all of this.
- Within this series: the unified map `reinforcement-learning-a-unified-map` and the capstone `the-reinforcement-learning-playbook` place deep hedging among model-free, model-based, and differentiable-simulation RL.
- Within this series: `/blog/machine-learning/reinforcement-learning/the-policy-gradient-theorem` for the score-function estimator deep hedging avoids, and `/blog/machine-learning/reinforcement-learning/the-credit-assignment-problem` for why short, differentiable horizons train so cleanly.
- On the trading side: `/blog/trading/options-volatility/delta-direction-exposure-and-the-hedge-ratio` for the desk's view of delta and the hedge ratio that this post turns into an RL action.
