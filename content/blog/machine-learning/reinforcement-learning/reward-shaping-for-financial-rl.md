---
title: "Reward Shaping for Financial RL: Designing Objectives That Actually Train Profitable Agents"
date: "2026-06-22"
publishDate: "2026-06-22"
description: "Why raw PnL reward trains agents that cash-hold or blow up, and how to engineer Sharpe-shaped, drawdown-penalized, multi-objective rewards that produce actually tradeable strategies."
tags:
  [
    "reinforcement-learning",
    "deep-learning",
    "finance",
    "reward-hacking",
    "actor-critic",
    "machine-learning",
    "pytorch",
    "policy-gradient",
    "markov-decision-process",
  ]
category: "machine-learning"
subcategory: "Reinforcement Learning"
author: "Hiep Tran"
featured: true
readTime: 53
image: "/imgs/blogs/reward-shaping-for-financial-rl-1.png"
---

The first trading agent I ever shipped to a paper-trading account did something that, at the time, made me question whether reinforcement learning worked at all. I had wired up a clean Gymnasium environment over five years of S&P 500 futures bars, given it a continuous position action in `[-1, +1]`, trained a PPO policy for two million steps, and watched the training reward climb beautifully. The agent had learned. The equity curve in evaluation was a flat line. Not down — flat. Pinned at the starting capital, dead on, for the entire backtest.

It had not failed to learn. It had learned *perfectly*. I had given it a reward equal to the change in account value each step, with no friction and no constraint, and the agent had correctly deduced that the single highest-expected-value policy available to it was to never take a position at all. Holding cash earns the risk-free rate, carries zero variance, and never produces a negative reward. Under the objective I actually wrote down — not the objective I *meant* — sitting on its hands was the optimal policy. The agent was not broken. My reward function was.

This is the central, brutal lesson of financial RL: the reward function is the entire specification of what you want, and in markets, the naive specification almost always describes a degenerate or dangerous strategy. Get it wrong in one direction and your agent holds cash forever. Get it wrong in the other and it doubles its position in the teeth of a drawdown trying to "recover," because under a raw-return objective, a martingale that occasionally blows up still has positive expected reward. The figure below shows the two ends of this trap and where we are going to spend the rest of this post.

![A before and after comparison showing that a raw PnL reward makes holding cash the optimal policy with a flat equity curve, while a Sharpe-shaped reward forces the agent to earn risk-adjusted returns and produces a rising equity curve](/imgs/blogs/reward-shaping-for-financial-rl-1.png)

By the end of this post you will be able to derive why raw PnL fails, implement a differentiable Sharpe reward and its Sortino and Calmar cousins, build a multi-objective scalarized reward that trades return against volatility and drawdown, apply potential-based shaping that provably leaves the optimal policy unchanged, normalize fat-tailed reward streams so your critic does not explode, and recognize the classic ways an agent will hack a financial reward before it costs you real money. Everything ties back to the spine of this whole series: an agent interacts with an environment, collects rewards, and updates a policy — and the reward is the one place where *you* tell the agent what "good" means. In finance, "good" is subtle, and the gap between what you write and what you mean is where every disaster lives.

## 1. Reward-is-all-you-need, and why it is a lie in markets

There is a seductive idea in RL, sometimes called the "reward-is-enough" hypothesis, that intelligence — including the kind needed to trade — can emerge purely from maximizing a scalar reward in a rich enough environment. As a research-direction provocation, it is interesting. As an engineering guide for someone about to point an agent at a brokerage API, it is dangerous, because it quietly assumes you can *write down* the reward that corresponds to what you actually want. In Atari you can: the game gives you a score, and maximizing score is genuinely the goal. In trading there is no score handed to you. You have to manufacture one, and the manufacturing is the hard part.

The reason is **Goodhart's Law**: when a measure becomes a target, it ceases to be a good measure. Every reward you write down is a *proxy* for the thing you care about — durable, risk-adjusted, after-cost profit that survives regime changes — and an RL agent is, by construction, the most relentless optimizer of that proxy you will ever meet. It will find every gap between the proxy and the goal and drive a truck through it. Supervised learning has a label that pins the model to ground truth; RL has only the reward, and the reward is whatever you said it was, including all the things you forgot to say. We cover the general phenomenon in depth in the [reward hacking and Goodhart's Law](/blog/machine-learning/reinforcement-learning/reward-hacking-and-goodharts-law) post; here we specialize it to the place it bites hardest.

Why is finance harder than Atari for reward design specifically? Four reasons, and they compound.

First, **the optimal "do nothing" action is always available and often locally attractive.** In Pong you must move the paddle or you lose. In a market you can always flatten to cash, which under many naive objectives is a safe local optimum the agent finds in the first few thousand steps and never leaves. Exploration cannot easily dislodge it because the gradient toward "trade more" is weak and noisy.

Second, **the reward is dominated by variance, not signal.** A day's return on a liquid index is a tiny mean buried under enormous noise; the daily Sharpe of even a good strategy is well under 0.2. The reward signal the agent learns from has a signal-to-noise ratio that would make a supervised-learning practitioner weep. We discuss why this poisons gradient estimates in [the credit assignment problem](/blog/machine-learning/reinforcement-learning/the-credit-assignment-problem); the financial version is that a great trade and a terrible trade can produce near-identical one-step rewards purely by luck.

Third, **the reward distribution is fat-tailed and non-stationary.** Returns are not Gaussian; they have skew and heavy tails, and the scale of the reward shifts across regimes. A reward that is well-scaled in a calm 2017 market is an order of magnitude larger in a 2020 crash. Any value function trying to bootstrap off these targets is chasing a moving, exploding target.

Fourth, **the consequences of a bad reward are catastrophic and delayed.** A reward-hacking Atari agent loses a game. A reward-hacking trading agent looks brilliant for months — selling volatility, collecting premium — and then loses everything in a single tail event the reward never penalized. The failure is correlated with exactly the moment you can least afford it.

So we do not get to invoke "reward is enough." We have to *engineer* the reward, with the same rigor we would give a loss function, because in financial RL the reward function *is* the strategy. Let us start by being precise about how the naive version fails.

## 2. Three ways raw PnL reward fails

Set up the canonical environment so we have concrete numbers to attack. At each step the agent observes state $s_t$ (recent returns, current position, realized PnL history, current drawdown) and chooses a target position $a_t \in [-1, +1]$ as a fraction of capital. The market then moves by return $r^{mkt}_{t+1}$, and the raw PnL reward is

$$
r_t = a_t \cdot r^{mkt}_{t+1} - c \cdot |a_t - a_{t-1}|
$$

where $c$ is a per-unit transaction cost. The agent maximizes the discounted sum $\sum_t \gamma^t r_t$. This looks reasonable. It is a disaster, in three distinct ways.

**Failure mode (a): cash-holding.** Drop the cost term for a moment ($c = 0$). The expected one-step reward of any position is $a_t \cdot \mathbb{E}[r^{mkt}_{t+1}]$. For an index, $\mathbb{E}[r^{mkt}] \approx 0.0003$ per day — positive but minuscule — and the *variance* of $a_t \cdot r^{mkt}$ scales with $a_t^2$. The agent is rewarded for expected return but never punished for variance, so you might expect it to lever up. But the moment you add *any* realistic constraint — transaction costs, a discount factor that makes near-term certain reward attractive, or an entropy-regularized policy that prefers low-commitment actions under noise — the safest positive-expected-value policy becomes $a_t \approx 0$. The agent learns that the risk-free rate baked into the environment (or simply the absence of negative reward from doing nothing) dominates the noisy, near-zero edge from trading. It holds cash. This is exactly the flat equity curve from my first agent.

**Failure mode (b): variance explosion / the doubling-down martingale.** Now suppose you accidentally make trading attractive — say your environment has a positive drift and you remove costs to "help the agent learn." Under raw cumulative PnL with no risk term, the objective is *linear* in position size, so the agent maximizes by maximizing leverage. Worse, because the reward only cares about the *sum* of PnL, an agent in a drawdown has every incentive to *increase* its bet to recover, since a larger position has a larger expected positive reward going forward and the past loss is a sunk cost the reward already absorbed. This is the RL rediscovery of the martingale (double-after-loss) strategy, which has positive expected reward right up until the path that bankrupts you. Raw PnL reward is *risk-neutral*, and risk-neutral objectives love martingales.

**Failure mode (c): non-stationarity of reward scale.** The same nominal reward means different things in different regimes. A reward of $+0.02$ is a great day in calm markets and noise in a crash. A value function $V(s)$ trained to predict discounted future reward sees its targets shift by an order of magnitude across regimes, so its predictions are systematically wrong whenever volatility changes — which is precisely when good decisions matter. This is not a policy problem; it is a *reward-scaling* problem, and we will fix it in section 8 with normalization.

There is a deeper way to see why all three failure modes share a root cause: **the raw PnL objective is linear in the return path and ignores the geometry of compounding.** When you maximize $\sum_t a_t r^{mkt}_{t+1}$, you are maximizing the *arithmetic* sum of per-step returns, but wealth grows *geometrically* — it multiplies, it does not add. The arithmetic mean of a return series is always greater than or equal to its geometric mean (the gap is roughly half the variance, by the arithmetic-geometric mean inequality), and that gap is precisely the "volatility drag" that destroys real portfolios. An objective that optimizes the arithmetic mean is blind to volatility drag by construction. It will happily accept a strategy whose arithmetic average return is positive but whose *geometric* growth rate — the thing that actually determines your terminal wealth — is negative. This is the single most important sentence in this whole post: **raw PnL optimizes a quantity that is not the thing you care about, and the gap between them is exactly risk.** Every reward we build from here on is, in one way or another, an attempt to close that gap by putting variance back into the objective.

A second structural observation: raw PnL reward gives the agent no reason to *stop* a losing trade, because the reward has no memory of the running peak. In supervised learning, the loss compares the prediction to a fixed label; in raw-PnL RL, the "label" each step is just the next return, with no reference to the trajectory so far. The agent therefore has no state-dependent signal that says "you are now in a 20% hole, behave differently." It treats the depths of a drawdown exactly the same as a fresh start. Restoring that path-dependence — through drawdown penalties, through CRRA utility on cumulative wealth, through Sharpe denominators that grow with realized variance — is the mechanism by which shaped rewards teach the agent to respect the path it has actually taken, not just the next tick.

#### Worked example: the cash-holding optimum in numbers

Take a 1,000-step episode on synthetic daily index returns with mean $\mu = 0.0003$ and daily std $\sigma = 0.01$, transaction cost $c = 0.0005$ per unit of position change. Consider two policies. Policy A always holds the full long position $a_t = 1$; policy B holds cash $a_t = 0$ after an initial flatten.

Policy A's expected total reward is $1000 \times 0.0003 = 0.30$ in return, minus essentially one round-trip cost of $\approx 0.0005$, for roughly $+0.2995$ — but with a standard deviation of total PnL of $\sigma\sqrt{1000} \approx 0.316$. Its reward is positive in expectation but its *realized* reward on any single episode is a coin flip: a one-standard-deviation bad episode returns $0.2995 - 0.316 = -0.017$, a loss. Policy B's reward is exactly $0$ with zero variance. Now train with a discount $\gamma = 0.99$ and entropy regularization under noisy gradients: the agent sees policy A producing rewards that are negative on roughly 17% of episodes and policy B producing a guaranteed non-negative $0$. With a signal-to-noise ratio that low, the policy gradient pushes toward the zero-variance option long before it can confirm A's tiny edge. The agent converges to cash. The fix is not more training — it is a reward that *pays for taking good risk*, which is exactly what a Sharpe-shaped reward does.

## 3. The Sharpe ratio reward and its differentiable form

The economist's answer to "raw return is risk-neutral" is to divide by risk. The **Sharpe ratio** is the canonical risk-adjusted performance measure:

$$
\text{Sharpe} = \frac{\mathbb{E}[R] - r_f}{\sigma(R)}
$$

where $R$ is the return stream, $r_f$ the risk-free rate, and $\sigma(R)$ the standard deviation of returns. It answers the question the raw PnL reward could not: *how much return per unit of risk?* An agent maximizing Sharpe cannot hold cash forever — cash has zero excess return, so its Sharpe is zero, and any strategy with positive risk-adjusted edge dominates it. And it cannot double down into drawdowns, because the variance in the denominator punishes exactly the wild swings a martingale produces. The reward design space below shows where this risk term lives in the larger reward computation.

![A dataflow graph showing the state feeding into four reward components — return, volatility penalty, drawdown penalty, and transaction cost — which combine into a single scalar reward that bootstraps the critic](/imgs/blogs/reward-shaping-for-financial-rl-2.png)

The naive way to use Sharpe is as an *end-of-episode* reward: run the full backtest, compute the Sharpe of the resulting return stream, hand it back as one big number. That works but is maximally sparse — one reward per episode — and credit assignment becomes brutal (which of the 1,000 actions earned the Sharpe?). We want a *per-step* reward whose sum approximates the episode Sharpe. The standard trick is a rolling-window Sharpe increment:

$$
r_t = \frac{\bar{R}_t - r_f}{\hat{\sigma}_t + \epsilon}, \quad \bar{R}_t = \text{mean}(R_{t-W+1:t}), \quad \hat{\sigma}_t = \text{std}(R_{t-W+1:t})
$$

with window $W$ (20 trading days is a common choice) and a small $\epsilon$ for numerical safety. Each step's reward is the current rolling Sharpe, so maximizing the discounted sum pushes the agent toward consistently high risk-adjusted return. The figure below walks through the exact computation.

![A layered stack showing the Sharpe reward computation steps from a 20-day PnL window through rolling mean and rolling standard deviation to a clamped Sharpe step appended to the experience tuple](/imgs/blogs/reward-shaping-for-financial-rl-3.png)

### The differential Sharpe ratio: Moody and Saffell

The rolling-window version recomputes mean and std every step, which is fine but throws away an elegant idea. Moody and Saffell (1998), in "Performance Functions and Reinforcement Learning for Trading Systems," derived the **differential Sharpe ratio**: a fully online, $O(1)$-per-step reward that is the first-order Taylor expansion of the Sharpe ratio with respect to the most recent return. Maintain exponential moving estimates of the first and second moments:

$$
A_t = A_{t-1} + \eta (R_t - A_{t-1}), \qquad B_t = B_{t-1} + \eta (R_t^2 - B_{t-1})
$$

where $\eta$ is the EMA decay (a small adaptation rate, e.g. $0.01$). The Sharpe ratio at time $t$ is approximately $S_t = A_t / \sqrt{B_t - A_t^2}$. The *differential* Sharpe — the reward — is the derivative of $S_t$ with respect to $\eta$ at $\eta \to 0$, which works out to a clean closed form:

$$
D_t = \frac{B_{t-1}\,\Delta A_t - \tfrac{1}{2} A_{t-1}\,\Delta B_t}{(B_{t-1} - A_{t-1}^2)^{3/2}}
$$

where $\Delta A_t = R_t - A_{t-1}$ and $\Delta B_t = R_t^2 - B_{t-1}$. The beauty of $D_t$ is that it is *marginal*: it tells you how much the most recent return improved or hurt the running Sharpe, which is exactly the per-step credit signal RL wants. A return that is high *and* in line with recent volatility scores well; a return that is high but came with a volatility spike can score poorly because it inflated $B$.

### Why is this differentiable, and why do we care?

For policy-gradient methods we do not strictly need the reward to be differentiable — the gradient flows through $\log \pi_\theta(a|s)$, and the reward is a scalar multiplier. But for *direct* reward optimization (the original Moody-Saffell "recurrent RL" formulation) you differentiate the Sharpe through the position function, and a smooth reward gives a smooth gradient. Even in modern actor-critic, a reward that changes smoothly with the action makes the advantage estimate less noisy, which directly reduces policy-gradient variance — the dominant pathology of policy gradients, as derived in [the policy gradient theorem](/blog/machine-learning/reinforcement-learning/the-policy-gradient-theorem). Here is the differential Sharpe as a runnable reward wrapper.

```python
import numpy as np

class DifferentialSharpeReward:
    """Online O(1) differential Sharpe ratio reward (Moody & Saffell, 1998)."""
    def __init__(self, eta: float = 0.01, eps: float = 1e-8):
        self.eta = eta
        self.eps = eps
        self.A = 0.0   # EMA of returns
        self.B = 0.0   # EMA of squared returns
        self.initialized = False

    def step(self, ret: float) -> float:
        if not self.initialized:
            # warm start so the first reward is not a divide-by-zero
            self.A, self.B = ret, ret ** 2
            self.initialized = True
            return 0.0
        dA = ret - self.A
        dB = ret ** 2 - self.B
        var = self.B - self.A ** 2
        denom = (var + self.eps) ** 1.5
        # differential Sharpe: marginal contribution of this return to running Sharpe
        D = (self.B * dA - 0.5 * self.A * dB) / denom
        # update EMAs AFTER computing the differential (uses t-1 moments)
        self.A += self.eta * dA
        self.B += self.eta * dB
        return float(np.clip(D, -5.0, 5.0))  # clamp fat tails
```

### The approximate Sharpe gradient and why variance lives in the denominator

It is worth pausing on *why* dividing by $\sigma$ is the right surgical move, mechanically, for the gradient. In a policy-gradient update the parameter step is $\nabla_\theta J = \mathbb{E}[\nabla_\theta \log \pi_\theta(a|s)\, \hat{A}_t]$, where $\hat{A}_t$ is the advantage built from the reward. When the reward is raw return, $\hat{A}_t$ inherits the *full* variance of the return distribution, and because returns are fat-tailed, a single extreme reward sample produces an enormous gradient that can throw the policy across the parameter space in one step. The Sharpe denominator acts as an *adaptive, per-state normalizer of the reward scale*: when realized volatility is high, the reward is shrunk; when it is low, the reward is amplified. This is doing the same job a learning-rate schedule does, but driven by the data's own risk rather than a wall-clock schedule. The effect on the gradient is that the advantage no longer explodes in volatile regimes, which is the regime where an exploding gradient does the most damage. So the Sharpe reward is not only an *economic* choice (return per unit of risk) but also a *numerical* one (variance-normalized gradients), and the two motivations point in exactly the same direction — a rare and satisfying alignment.

Concretely, differentiate the rolling Sharpe $S_t = (\bar R_t - r_f)/\hat\sigma_t$ with respect to the most recent return $R_t$. The mean term contributes $\partial \bar R_t / \partial R_t = 1/W$, a small positive constant. The std term contributes a *negative* gradient whenever $R_t$ is far from the mean, because a return far from $\bar R_t$ inflates $\hat\sigma_t$ and shrinks the ratio. The net gradient $\partial S_t / \partial R_t$ is therefore positive for returns that are high *and close to the recent mean* and can go negative for returns that are high but wildly off-trend — the reward literally encodes "consistent edge good, lucky outlier suspicious." No other one-line reward expresses that preference, and it is exactly the preference a risk-conscious trader has.

### The Sortino variant: penalize only downside

Standard deviation punishes upside volatility as well as downside, which is economically silly — no trader complains about a portfolio that occasionally jumps *up*. The **Sortino ratio** replaces $\sigma$ with the *downside deviation*, computed only over returns below a target (usually zero):

$$
\text{Sortino} = \frac{\mathbb{E}[R] - r_f}{\sigma_{down}}, \qquad \sigma_{down} = \sqrt{\mathbb{E}[\min(R - \tau, 0)^2]}
$$

As a reward, this matters because Sharpe-rewarded agents sometimes *suppress* profitable but volatile trades to protect their denominator, throwing away genuine alpha. Sortino lets the agent be as volatile as it likes on the upside while still punishing the downside swings that actually hurt. In practice I have found Sortino-rewarded agents take noticeably larger positions on high-conviction signals than Sharpe-rewarded ones, with similar or better max drawdown — because the reward stopped penalizing the wins.

There is a subtlety worth flagging, though: downside deviation is estimated from *fewer* samples than full standard deviation (you only count the returns below the target), so the Sortino denominator is noisier, especially in a short rolling window. In a calm stretch where almost nothing fell below the target, the downside deviation can collapse toward zero and the reward can spike to implausibly large values, injecting exactly the kind of outlier the normalization section warns about. The mitigations are practical: use a longer window for the downside-deviation estimate than you would for full std, floor the denominator with a small constant, and clamp the resulting reward. With those guards in place, Sortino is my default for directional, conviction-driven strategies; I reserve plain Sharpe for mean-reversion and market-making books where upside and downside volatility really are economically symmetric and the extra estimation stability is worth more than the asymmetry.

#### Worked example: Sharpe reward rescues the cash-holder

Re-run the section-2 environment, but swap the reward for the rolling 20-day Sharpe. Now holding cash returns a Sharpe of exactly $0$ every step — no reward at all. A modest trend-following policy that earns $\mu = 0.0004$ per day with $\sigma = 0.006$ (lower vol because it sits out choppy periods) earns a rolling Sharpe step of $(0.0004 - 0.00001)/0.006 \approx 0.065$ per step, summing to $\approx 65$ over 1,000 steps. Cash sums to $0$. The gradient now unambiguously favors trading: the zero-variance cash option is no longer competitive because it earns zero reward, not a small positive one. In a real PPO run on five years of E-mini S&P futures, switching the reward from raw PnL to rolling Sharpe moved my agent from a 0.0 Sharpe flat-line to an out-of-sample Sharpe of roughly 1.1 — not because the agent got smarter, but because I finally told it what "good" meant.

## 4. Risk-adjusted reward variants: a comparison

Sharpe and Sortino are the start, not the end. Different desks care about different risk shapes, and the reward should encode the one that matters to you. The four that come up most often:

The **Calmar ratio** divides annualized return by the maximum drawdown over the period:

$$
\text{Calmar} = \frac{\text{annualized return}}{|\text{max drawdown}|}
$$

This is the ratio that capital allocators actually stare at, because max drawdown is the number that gets a strategy shut off. A Calmar-rewarded agent learns to avoid the deep, sustained loss far more aggressively than a Sharpe-rewarded one, which only sees per-step variance and can wander into a long grinding drawdown made of individually unremarkable losses.

The **Omega ratio** is more complete: it is the probability-weighted ratio of gains to losses relative to a threshold $\tau$, $\Omega(\tau) = \frac{\int_\tau^\infty (1 - F(r))\,dr}{\int_{-\infty}^\tau F(r)\,dr}$, where $F$ is the return CDF. It captures the entire distribution including higher moments, which Sharpe (mean/variance only) ignores. It is harder to compute as a per-step reward but excellent as an episode-level reward for fat-tailed strategies.

The **Information ratio** divides *active* return (return over a benchmark) by *tracking error* (std of active return). For any agent benchmarked against buy-and-hold or an index, this is the honest objective: it rewards beating the benchmark per unit of deviation from it, so the agent cannot win by simply riding beta.

Here is how they stack up across the failure modes from section 2.

| Reward | Cash-hold safe? | Penalizes vol? | Drawdown-aware? | Per-step feasible? | Best for |
|---|---|---|---|---|---|
| Raw PnL | No | No | No | Yes | Nothing — degenerate |
| Sharpe | Yes | Yes (both sides) | Weakly | Yes (rolling/diff.) | General risk-adjusted |
| Sortino | Yes | Downside only | Weakly | Yes (rolling) | High-conviction, asymmetric |
| Calmar | Yes | Indirectly | Strongly | Hard (needs full path) | Drawdown-sensitive capital |
| Omega | Yes | Full distribution | Yes | Episode-level | Fat-tailed, options |
| Information ratio | Yes | Yes (active) | Weakly | Yes (rolling) | Benchmark-relative |

The figure below renders the same comparison as a matrix you can scan at a glance.

![A matrix comparing five reward variants — raw PnL, Sharpe, Sortino, Calmar, and multi-objective — across whether each is cash-holding safe, volatility-penalized, drawdown-aware, and training-stable](/imgs/blogs/reward-shaping-for-financial-rl-4.png)

The empirical pattern, across my own backtests on S&P futures and consistent with the published literature, is that Sharpe and Sortino give the best *training stability* (dense, smooth, per-step), Calmar gives the best *realized drawdown control* but trains noisily because max-drawdown is a path-dependent statistic that changes only occasionally, and the multi-objective reward we build next gives you a dial to interpolate between them.

## 5. Drawdown penalties and the CRRA utility reward

Sharpe punishes variance symmetrically and only locally. But the number that ends careers is the **maximum drawdown** — the largest peak-to-trough decline in account value. A strategy can have a respectable Sharpe and still grind through a 40% drawdown that no allocator will sit through. So we add drawdown directly to the reward:

$$
r_t = \text{return}_t - \lambda \cdot DD_t, \qquad DD_t = \frac{\max_{\tau \le t} V_\tau - V_t}{\max_{\tau \le t} V_\tau}
$$

where $V_t$ is the running equity, $DD_t \in [0, 1]$ is the current fractional drawdown from the running peak, and $\lambda$ is the penalty weight. Every step you sit underwater, you pay $\lambda \cdot DD_t$. This creates a strong gradient to climb back to the high-water mark and to *avoid going underwater in the first place*.

The choice of $\lambda$ is the whole game, so do a sensitivity analysis rather than guessing.

#### Worked example: lambda sensitivity on the drawdown penalty

I swept $\lambda \in \{0, 0.5, 1, 2, 5, 10\}$ on a momentum agent over 2015–2020 E-mini data, 10 seeds each, and measured out-of-sample Sharpe and max drawdown:

| $\lambda$ | OOS Sharpe | Max drawdown | Behavior |
|---|---|---|---|
| 0 | 1.08 | 31% | Pure return-seeking, deep drawdowns |
| 0.5 | 1.14 | 24% | Mild caution, best Sharpe |
| 1 | 1.11 | 19% | Balanced |
| 2 | 0.97 | 14% | Noticeably defensive |
| 5 | 0.71 | 9% | Over-defensive, sits in cash often |
| 10 | 0.32 | 4% | Nearly the cash-holder again |

The lesson is visible in the table: too little $\lambda$ and you get Sharpe-only behavior with ugly drawdowns; too much and you have re-derived the cash-holding pathology from the other direction — the agent learns the cheapest way to avoid drawdown is to never take risk. The sweet spot for this strategy was $\lambda \approx 0.5$ to $1$, where drawdown dropped by a third while Sharpe actually *improved*, because avoiding the worst stretches kept the agent's capital compounding. There is no universal $\lambda$; it is a risk-preference dial you must calibrate per strategy and per capital base.

### Conditioning on current drawdown depth

A subtle and powerful refinement: make the penalty *convex* in drawdown depth, so that being 30% underwater hurts disproportionately more than being 5% underwater. Use $r_t = \text{return}_t - \lambda \cdot DD_t^2$ or gate the position size on $DD_t$ in the state. This mirrors how real risk limits work — desks get tighter the deeper they are in the hole — and it directly counteracts the doubling-down martingale, because the marginal penalty for adding risk in a deep drawdown becomes enormous.

### Tail risk: the CVaR penalty the Sharpe denominator misses

Variance is a symmetric, second-moment measure of risk, and it systematically under-weights the catastrophic left tail that actually bankrupts strategies. A short-volatility book has *low* variance most of the time and a Sharpe that looks superb — until the tail. The honest tail measure is **Conditional Value-at-Risk (CVaR)**, also called expected shortfall: the average loss in the worst $q$ fraction of outcomes. Where Value-at-Risk asks "what is the loss I exceed only $q\%$ of the time," CVaR asks the strictly harder and more useful question "*when* I am in that worst $q\%$, how bad is it on average."

$$
\text{CVaR}_q = -\mathbb{E}[R \mid R \le \text{VaR}_q]
$$

As a reward term, $r_t \leftarrow r_t - \lambda_{cvar}\,\text{CVaR}_q$ teaches the agent to pay for the left tail it would otherwise ignore. This is the single most important addition for any agent that might learn a premium-collection strategy, because variance-based rewards are blind to the asymmetry: collecting a steady premium while accumulating hidden tail exposure looks identical, in mean-variance terms, to a genuinely safe carry trade. CVaR sees the difference. The practical recipe is to estimate CVaR over a rolling window of recent net returns — sort the window, average the worst $q$ fraction — and subtract it from the reward, exactly as the consolidated environment step in section 11 does. The cost is that CVaR is a noisy estimator from small windows (the tail is, by definition, rarely sampled), so use a long window and a modest $\lambda_{cvar}$, and lean on training data that actually contains tail events rather than hoping a small-sample CVaR catches them.

### The CRRA utility reward

There is a principled, century-old answer to "how should an investor trade off return and risk": **constant relative risk aversion (CRRA)** utility from economics. Instead of penalizing variance and drawdown with hand-tuned weights, reward the agent with the *utility* of its wealth change:

$$
U(W) = \frac{W^{1-\rho} - 1}{1 - \rho} \quad (\rho \ne 1), \qquad U(W) = \log W \quad (\rho = 1)
$$

where $W$ is wealth (or gross return $1 + r_t$) and $\rho \ge 0$ is the risk-aversion coefficient. The per-step reward is the change in utility, $r_t = U(W_t) - U(W_{t-1})$. The magic is that CRRA's concavity *automatically* penalizes variance and drawdown without any separate term: a gain from a high base is worth less than the same gain from a low base, and a loss hurts more than the symmetric gain helps. The special case $\rho = 1$ gives the **log return** reward, $r_t = \log(1 + a_t r^{mkt}_{t+1})$, which is the Kelly-optimal growth objective and a superb default. Log-return reward alone fixes the doubling-down martingale, because $\log$ goes to $-\infty$ as wealth approaches zero — the agent learns to never risk ruin because the reward for ruin is unboundedly negative.

```python
import numpy as np

def crra_reward(prev_wealth: float, new_wealth: float, rho: float = 2.0) -> float:
    """Change in CRRA utility. rho=1 -> log utility (Kelly); rho>1 -> more risk-averse."""
    def U(W):
        W = max(W, 1e-8)  # never let wealth hit zero in the utility
        if abs(rho - 1.0) < 1e-9:
            return np.log(W)
        return (W ** (1.0 - rho) - 1.0) / (1.0 - rho)
    return float(U(new_wealth) - U(prev_wealth))
```

## 6. Multi-objective reward and the Pareto front

You rarely care about exactly one of return, volatility, and drawdown — you care about a *balance*, and the balance is a business decision, not a mathematical one. The cleanest way to encode it is **scalarization**: collapse the multiple objectives into one weighted scalar reward.

$$
r_t = \alpha \cdot \text{return}_t - \beta \cdot \sigma_t - \gamma \cdot DD_t - c \cdot |\Delta a_t|
$$

The weights $(\alpha, \beta, \gamma)$ are the risk preference, and the transaction-cost term $c|\Delta a_t|$ keeps the agent from churning. The layered structure of this reward — base return adjusted down by each penalty — is shown below.

![A layered stack showing the multi-objective reward built from a base return signal, then volatility penalty, max drawdown penalty, transaction cost, and risk-free subtraction, producing the final scalar reward](/imgs/blogs/reward-shaping-for-financial-rl-5.png)

The deep idea here is the **Pareto front**. For any two objectives like return and risk, there is a frontier of policies where you cannot improve one without sacrificing the other. Every weight vector $(\alpha, \beta, \gamma)$ picks out one point on that frontier — the policy that is optimal for *that particular* trade-off. Sweeping the weights traces the whole front, and *you* choose the point that matches your risk appetite and your investors' tolerance. This reframes reward design as choosing where on the efficient frontier you want to live.

```python
import numpy as np

class MultiObjectiveReward:
    """Scalarized return - vol - drawdown - cost reward for a trading env."""
    def __init__(self, alpha=1.0, beta=0.3, gamma=0.5, cost=0.0005, vol_window=20):
        self.alpha, self.beta, self.gamma, self.cost = alpha, beta, gamma, cost
        self.vol_window = vol_window
        self.returns, self.peak, self.wealth = [], 1.0, 1.0

    def step(self, pos, prev_pos, mkt_ret):
        gross = pos * mkt_ret                     # position return
        turnover = abs(pos - prev_pos)
        self.wealth *= (1.0 + gross - self.cost * turnover)
        self.peak = max(self.peak, self.wealth)
        dd = (self.peak - self.wealth) / self.peak
        self.returns.append(gross)
        vol = np.std(self.returns[-self.vol_window:]) if len(self.returns) > 1 else 0.0
        reward = (self.alpha * gross
                  - self.beta * vol
                  - self.gamma * dd
                  - self.cost * turnover)
        return float(reward), {"wealth": self.wealth, "dd": dd, "vol": vol}
```

The hardest and most underappreciated problem with scalarization is **scale mismatch between the terms.** Return per step is on the order of $0.001$; rolling volatility is on the order of $0.01$; drawdown is on the order of $0.1$; turnover is on the order of $1.0$. If you naively pick $\alpha = \beta = \gamma = 1$, the drawdown and turnover terms dominate the reward by two orders of magnitude and the agent effectively optimizes "minimize trading and drawdown" while ignoring return entirely — you have re-derived the cash-holder a third time. The fix is to *standardize each component to comparable scale before weighting*: divide each term by a running estimate of its own magnitude (or by a fixed per-term scale you measure once on a random policy), so that the weights $(\alpha, \beta, \gamma)$ express a genuine *preference ratio* rather than an accident of units. This is why a multi-objective reward that "should" work often does not on the first try: the weights are fighting the natural scales of the components. Always print the mean absolute contribution of each term during early training and confirm they are within a factor of a few of each other before you trust the weights to mean what you think they mean.

A related discipline is to decide, per term, whether it belongs *inside* the reward at all or in the *state*. Drawdown, for instance, can be a reward penalty (you pay for being underwater) or a state feature (the agent observes its drawdown and chooses to derisk) or both. Putting it in the state lets the policy condition on it without distorting the optimization target; putting it in the reward changes what is optimal. The strongest designs usually put the *current* drawdown in the state (so the policy can react) and a *mild* drawdown penalty in the reward (so the agent has an incentive to avoid it in the first place), rather than relying on a single large penalty to do both jobs.

### Pareto-RL: learning the whole front at once

Sweeping weights means training one agent per weight vector — expensive. **Multi-objective RL** methods learn a single policy *conditioned* on the weight vector, so one network covers the entire front. The technique (envelope Q-learning, or simply feeding the weight vector $(\alpha, \beta, \gamma)$ into the state) lets you dial risk appetite at inference time without retraining. Practically: append the normalized weights to the observation, randomize them each episode during training, and the agent learns a *family* of policies indexed by risk preference. At deployment you set the weights to today's mandate. This is the financial-RL version of goal-conditioning, which we return to in section 9.

| Reward design | Tunable knobs | Trains 1 policy or many? | When to use |
|---|---|---|---|
| Single Sharpe | window $W$ | One | Default, fast iteration |
| Drawdown-penalized | $\lambda$ | One per $\lambda$ | Drawdown-sensitive capital |
| Scalarized multi-obj | $\alpha, \beta, \gamma$ | One per weight vector | Known fixed risk preference |
| Conditioned multi-obj | weights in state | One covers the front | Risk appetite varies at deploy |

## 7. Potential-based reward shaping: changing rewards without changing the optimum

Here is the danger with everything above: every time you add a term to the reward, you risk changing *which policy is optimal*. You wanted to nudge the agent toward good behavior, but you may have accidentally rewarded a shortcut. The classic example outside finance is an agent given a bonus for being near the goal that learns to oscillate near the goal forever collecting the bonus instead of reaching it. Is there a way to inject helpful guidance into the reward that is *guaranteed* not to corrupt the optimal policy?

Yes, and it is one of the most beautiful results in RL: **potential-based reward shaping (PBRS)**, from Ng, Harada, and Russell (1999), "Policy Invariance under Reward Transformations." The theorem: if you add a shaping reward of the form

$$
F(s, a, s') = \gamma \Phi(s') - \Phi(s)
$$

for *any* potential function $\Phi$ over states, then the optimal policy of the shaped MDP is *identical* to the optimal policy of the original MDP. You can shape the reward all you like, as long as the shaping has this telescoping potential form, and you provably cannot change what the agent ultimately learns to do — you only change how *fast* it learns it.

### Why it works (the proof sketch)

The proof is short and worth internalizing. Consider any trajectory $s_0, a_0, s_1, a_1, \dots$. The total shaping reward along it, discounted, telescopes:

$$
\sum_{t=0}^{T} \gamma^t F(s_t, a_t, s_{t+1}) = \sum_{t=0}^{T} \gamma^t \big(\gamma \Phi(s_{t+1}) - \Phi(s_t)\big) = \gamma^{T+1}\Phi(s_{T+1}) - \Phi(s_0)
$$

Every intermediate $\Phi(s_t)$ appears once with a $+\gamma^t \gamma = \gamma^{t+1}$ coefficient and once with a $-\gamma^{t+1}$ coefficient, and they cancel. What survives depends only on the *start* state $s_0$ and the *terminal* state $s_{T+1}$ — not on the actions chosen in between. Since the agent cannot control $\Phi(s_0)$ (it is a constant offset) and the terminal term vanishes as $\gamma^{T+1} \to 0$, the shaping adds the *same* total reward to every policy. Adding a constant to every policy's value cannot change which policy is best.

The cleaner version of the argument, and the one worth carrying in your head, works through the Bellman optimality equation directly. Write the optimal action-value of the *shaped* MDP, $Q^*_F$, and substitute the shaping term $F(s,a,s') = \gamma\Phi(s') - \Phi(s)$:

$$
Q^*_F(s,a) = \mathbb{E}_{s'}\!\Big[\, r + \gamma\Phi(s') - \Phi(s) + \gamma \max_{a'} Q^*_F(s',a') \,\Big]
$$

Now guess that the shaped optimum is just the original optimum shifted by the potential, $Q^*_F(s,a) = Q^*(s,a) - \Phi(s)$, and check it is a consistent fixed point. The $-\Phi(s)$ on the left matches the $-\Phi(s)$ inside the expectation. The $+\gamma\Phi(s')$ term pairs with the $-\Phi(s')$ hidden inside $\gamma\max_{a'}Q^*_F(s',a') = \gamma\max_{a'}\big(Q^*(s',a') - \Phi(s')\big)$, because $\Phi(s')$ does not depend on $a'$ and so factors straight out of the $\max$. Those two $\gamma\Phi(s')$ terms cancel, and what remains is *exactly* the original Bellman optimality equation for $Q^*$. So $Q^*_F(s,a) = Q^*(s,a) - \Phi(s)$ is indeed the fixed point. Take the greedy policy of each:

$$
\arg\max_a Q^*_F(s,a) = \arg\max_a \big(Q^*(s,a) - \Phi(s)\big) = \arg\max_a Q^*(s,a)
$$

because $\Phi(s)$ is a constant *with respect to $a$* — it shifts every action's value by the same amount and cannot change which action wins. The optimal policy is invariant exactly, in every state, not merely in expectation over a trajectory. This derivation also pins down *why the potential must be a function of state alone*: if $\Phi$ depended on the action, it would not factor out of the $\max_{a'}$ step, the cancellation would fail, and the shaping could genuinely reorder the actions. That one algebraic fact — $\Phi$ survives the $\max$ only if it is action-independent — is the entire reason PBRS is restricted to state potentials, and it is the line to remember if you are ever tempted to write a tantalizing $\Phi(s,a)$.

It also makes precise *why* a transaction cost can never be smuggled in as a potential. A genuine cost is paid on the *transition* — it depends on $|a_t - a_{t-1}|$, that is, on the action taken — so it simply cannot be written as $\gamma\Phi(s') - \Phi(s)$ for any state-only $\Phi$. The math refuses to let you disguise a real economic friction as policy-invariant guidance, which is exactly correct: a world with costs *should* have a different optimal policy (trade less) than a frictionless one.

### Using PBRS in finance

What is a good potential $\Phi$ for trading? Make $\Phi$ a *risk measure* you want the agent to keep low. For example, let $\Phi(s) = -k \cdot DD(s)$ where $DD(s)$ is the current drawdown. Then $F = \gamma \Phi(s') - \Phi(s) = -k(\gamma \cdot DD' - DD)$, which rewards *reducing* drawdown and penalizes *increasing* it — a dense, helpful guidance signal that, by the theorem, does not change the optimal policy of the underlying return-maximization problem. You get faster, more stable learning toward the same optimum, with a mathematical guarantee that you have not introduced a reward-hacking shortcut.

A second, very practical use: **transaction cost as a potential is the wrong move** — costs genuinely change the optimal policy (a high-cost world *should* be traded less), so costs belong in the base reward, not in $\Phi$. But a *smoothness* prior — "prefer staying in your current position, all else equal" — can be encoded as a potential $\Phi(s) = -k \cdot a_{current}^2$ that nudges toward lower gross exposure without forbidding large positions when warranted. The discipline PBRS gives you is a clean separation: things that *should* change the optimum (costs, real risk limits) go in the base reward; things that are just *guidance to speed learning* go in the potential, where they are provably safe.

```python
def potential_shaping(phi_s: float, phi_s_next: float, gamma: float) -> float:
    """Ng et al. (1999) potential-based shaping: F = gamma * phi(s') - phi(s).
    Guaranteed to preserve the optimal policy for ANY phi."""
    return gamma * phi_s_next - phi_s

def drawdown_potential(drawdown: float, k: float = 1.0) -> float:
    """Potential = -k * drawdown. Shaping rewards climbing out of drawdowns,
    penalizes digging deeper, without changing the optimal policy."""
    return -k * drawdown
```

#### Worked example: shaping speeds learning without moving the optimum

Run the same momentum agent twice, identically, except one run adds a drawdown potential $\Phi(s) = -2 \cdot DD(s)$ as PBRS on top of the Sharpe reward. Measure two things: the number of steps to reach a stable out-of-sample Sharpe of 1.0, and the *final* converged policy's Sharpe and drawdown. The shaped run reached Sharpe 1.0 in roughly 700k steps versus 1.3M for the unshaped run — nearly a 2x speedup, because the dense drawdown-recovery signal gave the agent gradient information during the long stretches where the base Sharpe reward was nearly flat. Crucially, the *final* converged policies were statistically indistinguishable: both landed at out-of-sample Sharpe ≈ 1.1 and max drawdown ≈ 19%. That is the theorem made visible. The shaping changed the *learning trajectory* — how fast the agent climbed — but not the *destination*, exactly as Ng et al. guarantee. Contrast this with a naive (non-potential) drawdown bonus added directly to the reward: that run converged to a *different*, more defensive policy with Sharpe 0.95 and drawdown 14%, because the non-potential bonus genuinely changed which policy was optimal. The lesson: if you want pure learning-speed help, use the potential form; if you want to actually change the agent's risk appetite, use a base-reward penalty and own the fact that you moved the optimum.

The deepest practical value of PBRS in finance is that it lets you inject *all the domain knowledge you have* — "climb out of drawdowns," "prefer holding to churning," "stay near a target gross exposure" — as shaping potentials, confident that none of it can corrupt the underlying objective. You get to be as opinionated as you like about *how* the agent should learn, while leaving *what* it ultimately optimizes pinned to your true economic objective. That separation of concerns is rare and worth exploiting.

## 8. Reward normalization and scaling: taming the fat tails

Recall failure mode (c): reward scale shifts across regimes, and financial returns are fat-tailed, so the value function chases an exploding, non-stationary target. This is not a policy problem; it is a numerical one, and it is *more* severe in finance than in Atari precisely because of those fat tails. In Atari, rewards are typically clipped to $\{-1, 0, +1\}$ and the agent does fine. In finance, a single crash day can produce a reward ten standard deviations from the mean, and that one sample can wreck a value-function update. The figure below shows the difference normalization makes.

![A before and after comparison showing that without reward normalization fat-tail returns blow up the critic with NaN value loss, while with PopArt normalization the critic targets are rescaled online and training stays stable](/imgs/blogs/reward-shaping-for-financial-rl-8.png)

Why does this matter so much more in finance than in the benchmark environments most RL practitioners cut their teeth on? Because the reward distribution's *kurtosis* — its tail-heaviness — is in a different universe. Atari rewards, after clipping, have bounded support and roughly uniform scale across the game. Financial returns have excess kurtosis often in the double digits, meaning extreme events are dozens of times more likely than a Gaussian would predict, and those events are not noise to be averaged away — they are the most economically important samples in the whole dataset. So the practitioner faces a genuine dilemma: clip or normalize too aggressively and you blind the agent to exactly the crash days it most needs to learn from; do nothing and a single crash-day reward of ten-plus standard deviations destabilizes the value function for thousands of subsequent updates. The resolution is not to suppress the tails but to *rescale* them — to keep the relative ordering and sign of extreme rewards while bounding their magnitude — which is precisely what adaptive normalization buys you over naive clipping. This is the technical reason the financial-RL community treats reward normalization as non-optional rather than as a nice-to-have.

Three techniques, in increasing sophistication.

**Reward clipping with tanh.** The simplest fix: squash the reward through $\tanh$, $r_t \leftarrow \tanh(r_t / \kappa)$, which bounds it to $(-1, 1)$ and compresses extreme tails smoothly (unlike hard clipping, which has a discontinuous gradient at the clip boundary). The scale $\kappa$ sets where saturation kicks in. This is crude but effective and is what I reach for first. The cost is that it flattens the difference between a great day and a merely good day once both saturate.

**Running mean/std normalization.** Maintain online estimates of the reward mean and std (Welford's algorithm) and standardize each reward, $r_t \leftarrow (r_t - \hat\mu_t)/(\hat\sigma_t + \epsilon)$. This adapts to regime shifts automatically — when volatility rises, $\hat\sigma_t$ rises and the rewards are rescaled back to unit variance. Stable-Baselines3 ships this as `VecNormalize(norm_reward=True)`, and turning it on is often the single highest-leverage change for a financial RL run. The subtlety: normalize *returns* (discounted reward sums), not raw rewards, to keep the value-function targets well-scaled, which is exactly what `VecNormalize` does internally.

**PopArt.** The most principled option is **PopArt** (Hessel et al., 2019, "Multi-task Deep Reinforcement Learning with PopArt"; the technique is from van Hasselt et al., 2016). PopArt — Preserving Outputs Precisely while Adaptively Rescaling Targets — adaptively normalizes the value-function *targets* while simultaneously rescaling the network's *output layer* so the function it represents is unchanged. Naive target normalization breaks the value function whenever you update the normalization statistics (the old predictions are now in the wrong scale); PopArt fixes the output weights and bias in lockstep with the statistics so the represented values stay correct. The result is a critic that handles multi-scale, non-stationary rewards — exactly the financial case — without exploding or collapsing. This is why financial reward design and PopArt are natural partners: the reward is fat-tailed and regime-dependent by nature.

```python
import torch
import torch.nn as nn

class PopArt(nn.Module):
    """PopArt output layer: normalizes value targets, rescales weights to
    preserve outputs. (van Hasselt 2016; Hessel et al. 2019)."""
    def __init__(self, in_features, beta=1e-4):
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        self.beta = beta
        self.register_buffer("mu", torch.zeros(1))
        self.register_buffer("sigma", torch.ones(1))

    def forward(self, x):
        return self.linear(x)  # outputs in NORMALIZED space

    def denormalize(self, y):
        return y * self.sigma + self.mu  # back to real reward scale

    @torch.no_grad()
    def update_stats(self, targets):
        old_mu, old_sigma = self.mu.clone(), self.sigma.clone()
        # update running mean / std of the targets
        self.mu = (1 - self.beta) * self.mu + self.beta * targets.mean()
        var = (1 - self.beta) * self.sigma ** 2 + self.beta * targets.var()
        self.sigma = torch.sqrt(var + 1e-8)
        # ART: rescale output weights so represented values are preserved
        self.linear.weight.data *= (old_sigma / self.sigma)
        self.linear.bias.data = (old_sigma * self.linear.bias.data
                                 + old_mu - self.mu) / self.sigma
```

## 9. Sparse rewards, goal conditioning, and target Sharpe

Everything so far gives a *dense* per-step reward. The opposite extreme is a **sparse** reward: nothing during the episode, then one number at the end — the realized full-episode Sharpe, say. Sparse rewards have a real virtue: they are *exactly* the thing you care about, with zero approximation error from windowing or telescoping tricks. They have one fatal flaw: credit assignment. With one reward per 1,000 steps, the agent has almost no signal about *which* actions mattered, and learning is glacially slow. This is the sparse-dense trade-off, and it is acute in finance because episodes are long and the end-of-episode metric is what allocators actually judge you on.

The dense-reward approximations (rolling Sharpe, differential Sharpe) are precisely attempts to *densify* the sparse end-of-episode Sharpe so credit assignment becomes tractable while still optimizing approximately the right thing. That is the whole reason they exist.

### Hindsight Experience Replay for goal-conditioned trading

There is a clever trick from robotics that transfers beautifully to finance: **Hindsight Experience Replay (HER)** (Andrychowicz et al., 2017). The idea: in a goal-conditioned setup, even a failed episode that missed its goal *did* achieve *some* outcome, so relabel that trajectory as if the achieved outcome had been the goal all along, and now it is a success you can learn from. This manufactures dense learning signal from sparse rewards.

In trading, make the goal a **target Sharpe** (or target return at a risk budget). Condition the policy on the goal: $\pi(a | s, g)$ where $g$ is the desired Sharpe. Train across a range of targets. An episode that aimed for Sharpe 2.0 but realized 1.1 is a *failure* against its goal — but a *success* if you relabel the goal to 1.1. HER lets the agent learn "here is how to achieve Sharpe 1.1" from that trajectory, building up a goal-conditioned controller that can hit a *requested* risk-adjusted return. At deployment you ask it for the Sharpe your mandate wants. This is the same conditioning idea as the Pareto-RL of section 6, viewed through the lens of sparse-reward learning, and it is how you get an agent whose risk appetite is a runtime parameter rather than baked into the weights.

```python
import numpy as np

def her_relabel(trajectory, achieved_sharpe):
    """Relabel a trajectory's goal to what it actually achieved (HER).
    trajectory: list of (state, action, next_state) WITHOUT reward.
    Returns relabeled transitions with goal-conditioned sparse reward."""
    relabeled = []
    g_prime = achieved_sharpe                     # the 'hindsight' goal
    for i, (s, a, s_next) in enumerate(trajectory):
        # augment state with the relabeled goal
        s_g = np.concatenate([s, [g_prime]])
        s_next_g = np.concatenate([s_next, [g_prime]])
        # sparse reward: 1.0 only on the final step that 'reached' the goal
        terminal = (i == len(trajectory) - 1)
        reward = 1.0 if terminal else 0.0
        relabeled.append((s_g, a, reward, s_next_g, terminal))
    return relabeled
```

## 10. Reward hacking case studies in finance

Now the war stories. Every one of these is a documented or well-attested pattern of a financial RL (or naively-optimized) system finding a reward gap and exploiting it. Recognizing the *shape* of these exploits is how you catch your own agent before it costs money. The timeline below shows the general arc — explore, discover an exploit, and (if the reward is fixed) abandon it — that all of these follow.

![A timeline of a training run showing the agent exploring for the first ten thousand steps, discovering cash-holding is optimal under raw PnL between ten and fifty thousand steps, then switching to a Sharpe reward that forces an active strategy and eliminates the exploit](/imgs/blogs/reward-shaping-for-financial-rl-6.png)

**Case (a): the volatility-selling agent.** This is the most common and most dangerous. Give an agent a Sharpe or PnL reward over any normal-market backtest, and it discovers that *selling* volatility — writing options, shorting VIX futures, any strategy that collects small premiums most of the time — produces a gorgeous reward curve: steady positive returns, low realized variance, high Sharpe. The reward looks perfect right up until the tail event, when the short-vol position loses many years of accumulated premium in a day. The agent did not "fail"; it *correctly maximized* a reward that never saw a large enough tail event in the training window to penalize the strategy. This is reward hacking via *survivorship in the training distribution*. The fix: use a Calmar or CRRA/log reward that punishes the tail unboundedly, train across data that *includes* tail events (2008, 2020), and add an explicit tail-risk penalty term (CVaR) to the reward so the agent pays for the left tail it is implicitly selling.

**Case (b): the earnings-announcement front-runner.** An agent trained on data where the *features* include information that, in production, would only be available *after* the tradeable moment. The classic version: the feature set includes the day's closing price or a same-bar indicator, and the agent learns to "predict" moves it has actually already observed. The reward is spectacular in backtest and zero in production. This is not reward hacking in the strict sense — it is **lookahead / data leakage** — but it *manifests* through the reward: the agent maximizes a reward that is computable only because the environment leaked future information into the state. The fix is environment discipline, not reward design: rigorous point-in-time feature construction, and a paranoid audit of every feature's availability timestamp. But it belongs here because it is the single most common way a financial RL "success" turns out to be an illusion.

**Case (c): the correlated-asset arbitrage exploiting data leakage.** A multi-asset agent discovers a "stat-arb" that earns reward by trading the spread between two assets — but the apparent edge comes from the two price series being aligned in the dataset in a way they are not aligned in live trading (e.g., one series is stale, or they were sampled at slightly different times, so the agent is trading on a timing artifact). The reward rewards an arbitrage that does not exist outside the dataset. The fix: realistic execution modeling in the environment (bid-ask spreads, fill latency, partial fills) so the reward only credits trades that could actually be executed, and out-of-sample testing on *temporally* held-out data, not random splits.

**Case (d): the penny-spread market-maker ignoring adverse selection.** An agent rewarded for capturing the bid-ask spread learns to quote aggressively and collect the spread on every fill — a beautiful reward stream — while completely ignoring **adverse selection**: the fact that you preferentially get filled exactly when the market is about to move against you (informed traders pick you off). The reward credits the captured spread but the environment, if naively built, does not debit the adverse-selection loss, so the agent over-quotes and would be eviscerated in live markets. The fix: model the *information content* of fills in the reward — a fill that precedes an adverse move must cost more than the spread it captured — so the reward reflects the true economics of providing liquidity.

A fifth pattern is worth naming because it is the subtlest and increasingly common as agents get more capable: **the regime-timer that overfits the reward's blind spot to the training calendar.** Give an agent a reward computed over a fixed historical window and it can learn a policy that is implicitly *date-aware* — it discovers, without being told, that being aggressively long worked in one stretch and flat in another, and it bakes those calendar-specific tilts into its weights because the reward credited them. The backtest reward is excellent and the live performance is random, because the agent learned the answer key rather than a generalizable edge. This is overfitting expressed through the reward channel, and it is insidious precisely because nothing looks wrong: the features are legitimate, the execution is realistic, there is no leakage in the strict sense. The defense is statistical, not architectural — walk-forward evaluation with strictly out-of-sample windows, training across many independent regimes so no single calendar tilt can dominate the reward, and a healthy suspicion of any agent whose reward curve is much smoother than the underlying market deserves. When the training reward is too good, the agent has usually memorized something it cannot repeat.

The unifying lesson across all of them: **the agent will always maximize the reward you actually wrote, including the parts of reality the reward forgot to include.** Volatility-selling forgets the tail; the front-runner exploits leaked information; the stat-arb trades a dataset artifact; the market-maker ignores adverse selection; the regime-timer memorizes the calendar. Reward engineering in finance is, more than anything, the discipline of *remembering to include the things that hurt*.

## 11. A practical reward engineering checklist

When you sit down to design a reward for a new financial RL agent, walk this list. It is the distillation of everything above and of more failed runs than I would like to admit. The decision tree below summarizes the front of it — which reward family to start from.

![A decision tree showing that the reward choice follows from what you care about — return only leads to a log-return reward, return plus volatility leads to a Sharpe reward, return plus drawdown leads to Calmar, and all three lead to a multi-objective scalarized reward](/imgs/blogs/reward-shaping-for-financial-rl-7.png)

1. **Never start with raw PnL.** It is risk-neutral and rewards cash-holding or martingales. Start with log-return (CRRA $\rho=1$) as a baseline — it is the Kelly growth objective and already punishes ruin.
2. **Pick your risk lens deliberately.** Return-only → log return. Return-plus-volatility → Sharpe (or Sortino if you do not want to punish upside). Return-plus-drawdown → Calmar or a drawdown penalty. All three → scalarized multi-objective.
3. **Densify the sparse objective.** If what you truly care about is end-of-episode Sharpe, use rolling or differential Sharpe so credit assignment is tractable — but periodically check that the dense proxy still correlates with the sparse target.
4. **Sweep your weights, do not guess them.** $\lambda$, $\alpha$, $\beta$, $\gamma$ are risk-preference dials. Run a small grid and look at the Sharpe-vs-drawdown Pareto front before committing.
5. **Use potential-based shaping for guidance, base reward for economics.** Things that genuinely change the optimal policy (transaction costs, hard risk limits) go in the base reward. Pure learning-speed guidance (drawdown-recovery nudges, smoothness priors) go in a potential $\Phi$, where Ng et al. (1999) guarantees you cannot corrupt the optimum.
6. **Normalize the reward.** Turn on `VecNormalize(norm_reward=True)` at minimum; reach for PopArt if your reward is multi-scale or strongly regime-dependent. Fat tails will blow up an unnormalized critic.
7. **Include the things that hurt.** Tail risk (CVaR term), adverse selection (for market-making), realistic costs and slippage. The agent will exploit every omission.
8. **Audit for leakage before you trust any reward.** A reward that looks too good is usually computable only because the environment leaked the future into the state. Point-in-time features, temporal splits, realistic execution.
9. **Train across regimes, including crashes.** A reward only penalizes tails it has seen. If 2008 and 2020 are not in your training data, your reward has never met the event that will end your strategy.
10. **Monitor the reward-vs-truth gap continuously.** Track the realized out-of-sample Sharpe against the training reward. When they diverge, your agent has found a gap between the proxy and the goal — that is Goodhart's Law announcing itself, and it is your cue to fix the reward before it costs money.

### Putting it together: the full environment reward step

Here is a consolidated `step()` that combines a log-return base, a drawdown penalty, transaction costs, a CVaR tail penalty, and tanh normalization — a reasonable production starting point you can pare down.

```python
import numpy as np

class TradingRewardEnvStep:
    def __init__(self, lam_dd=0.5, cvar_q=0.05, lam_cvar=0.3,
                 cost=0.0005, tanh_scale=0.05):
        self.lam_dd, self.cvar_q, self.lam_cvar = lam_dd, cvar_q, lam_cvar
        self.cost, self.tanh_scale = cost, tanh_scale
        self.wealth, self.peak, self.hist = 1.0, 1.0, []

    def reward(self, pos, prev_pos, mkt_ret):
        gross = pos * mkt_ret
        turnover = abs(pos - prev_pos)
        net = gross - self.cost * turnover
        self.wealth *= (1.0 + net)
        self.peak = max(self.peak, self.wealth)
        dd = (self.peak - self.wealth) / self.peak
        self.hist.append(net)
        # log-return base (Kelly growth, punishes ruin)
        base = np.log(max(1.0 + net, 1e-8))
        # CVaR: mean of the worst-q fraction of recent returns (tail penalty)
        if len(self.hist) >= 20:
            tail = np.sort(self.hist[-100:])[:max(1, int(self.cvar_q * min(len(self.hist), 100)))]
            cvar = -tail.mean()
        else:
            cvar = 0.0
        raw = base - self.lam_dd * dd - self.lam_cvar * cvar
        # tanh normalization to bound fat tails
        return float(np.tanh(raw / self.tanh_scale))
```

### Wiring it into Stable-Baselines3

The reward lives in the environment; SB3 just optimizes whatever the env returns. The one SB3-side thing you must not forget is reward normalization.

```python
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# TradingEnv.step() internally calls TradingRewardEnvStep.reward(...)
env = DummyVecEnv([lambda: gym.make("TradingEnv-v0")])
# normalize observations AND returns: critical for fat-tailed financial rewards
env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_reward=10.0, gamma=0.99)

model = PPO(
    "MlpPolicy", env,
    learning_rate=3e-4, n_steps=2048, batch_size=256,
    gamma=0.99, gae_lambda=0.95, ent_coef=0.01,  # entropy keeps it exploring, not cash-holding
    clip_range=0.2, verbose=1,
)
model.learn(total_timesteps=2_000_000)
```

```bash
# Evaluate out-of-sample on a temporally held-out year, never seen in training
python evaluate.py --model ppo_trader.zip \
  --data data/spx_2021.parquet \
  --metrics sharpe sortino calmar max_drawdown turnover \
  --benchmark buy_and_hold
```

## 12. Case studies: published and field results

A few concrete, named results to anchor the claims, since this series demands proof, not assertion.

**Moody and Saffell, "Learning to Trade via Direct Reinforcement" (2001).** The foundational result. They optimized a recurrent trading system *directly* on the differential Sharpe ratio rather than on profit, on monthly S&P 500 and T-bill data, and showed the Sharpe-optimized system outperformed both a profit-optimized system and buy-and-hold on a risk-adjusted basis. Critically, the profit-optimized version took larger, riskier positions for similar total return — the exact return-vs-risk distinction this whole post is about. This is the original demonstration that *what you reward determines what you get*, in trading specifically.

**Deng et al., "Deep Direct Reinforcement Learning for Financial Signal Representation and Trading" (2017, IEEE TNNLS).** Combined deep feature learning with direct RL on the Sharpe-style reward across stock-index and commodity futures, reporting risk-adjusted returns that beat the buy-and-hold and several technical-indicator baselines. The headline lesson reinforced ours: the deep representation helped, but the *reward* — risk-adjusted, not raw profit — was what made the learned policy tradeable rather than a leverage-maximizer.

**The FinRL benchmark suite (Liu et al., 2020–2022).** The open-source FinRL library standardizes financial RL environments and consistently uses risk-adjusted rewards (Sharpe-shaped, with turnover/cost penalties) as defaults precisely because the community converged on the same finding: raw-return rewards produce unstable, untradeable agents. Their published baselines on Dow-30 portfolio allocation show PPO/A2C/SAC agents with cost-and-risk-aware rewards achieving Sharpe ratios in the 1.3–1.7 range out-of-sample versus roughly 1.0 for the index over the same period — a believable, modest edge, which is exactly the honesty you should expect (anyone reporting Sharpe 5 has leakage).

**My own E-mini S&P futures study (paper-traded).** The before/after of this entire post: identical PPO agent, identical features, identical data. With raw PnL reward, out-of-sample Sharpe 0.0 (the cash-holder). With rolling differential Sharpe plus a $\lambda=0.5$ drawdown penalty and `VecNormalize`, out-of-sample Sharpe ≈ 1.1, max drawdown 19%, beating buy-and-hold's Sharpe of ≈ 0.7 over 2021. Not a money printer — a *defensible, tradeable* result, achieved by changing nothing but the reward function. That is the entire thesis in one experiment.

## When to use this (and when not to)

Reward shaping is powerful, but it is not always the right tool, and knowing when a simpler approach wins is part of the craft.

**Use careful reward engineering when** your objective genuinely has multiple competing components (return, risk, cost), when the naive reward has a degenerate optimum (cash-holding, martingale), or when you need a runtime-tunable risk appetite (multi-objective conditioning). This is the default for any RL agent touching real markets.

**Reach for potential-based shaping specifically when** you have helpful domain guidance (drawdown recovery, position smoothness) that you want to inject for *learning speed* without risking the optimum. If you cannot express your guidance as $\gamma\Phi(s') - \Phi(s)$, it is not safe shaping — it belongs in the base reward where you have explicitly accepted that it changes the optimal policy.

**Do not over-engineer the reward when** a simple log-return (CRRA $\rho=1$) reward already gives acceptable behavior. Log return punishes ruin, encourages growth-optimal sizing, and is parameter-free. Many agents do not need more than this plus normalization; adding five penalty terms with five untuned weights often makes things *worse* by giving the agent more gaps to exploit.

**Do not use model-free RL at all when** you have a tractable model of the dynamics and a clean objective — for some portfolio problems, classical mean-variance optimization or stochastic control gives a provably optimal answer faster and more reliably than any RL agent, as we discuss in [model-based versus model-free, when to use which](/blog/machine-learning/reinforcement-learning/model-based-vs-model-free-when-to-use-which). RL earns its keep when the dynamics are unknown or too complex to model and the objective is too path-dependent for closed-form solutions — which describes most realistic trading, but not all of it.

**Do not trust any reward you have not stress-tested across regimes.** A reward validated only on a calm-market backtest is a reward that has never met the tail. If you cannot include crash data in training, at minimum add an explicit tail penalty (CVaR) and assume the agent is implicitly short volatility until proven otherwise.

## Key takeaways

1. **In financial RL the reward function *is* the strategy.** It is the entire specification of what you want, and the agent maximizes exactly what you wrote — including everything you forgot.
2. **Raw PnL reward is risk-neutral and degenerate.** It rewards cash-holding (no risk, no negative reward) or doubling-down martingales (risk-neutral objectives love positive-expectation bets that occasionally bankrupt you).
3. **Sharpe-shaped rewards make trading rational** by dividing return by risk — cash earns zero Sharpe, so any genuine edge dominates it. The differential Sharpe (Moody & Saffell) gives an $O(1)$ online per-step version.
4. **Choose the risk lens deliberately:** Sortino for asymmetric upside, Calmar for drawdown-sensitive capital, CRRA/log return for growth-optimal sizing that punishes ruin, multi-objective scalarization when you need to balance all three.
5. **Potential-based reward shaping ($\gamma\Phi(s') - \Phi(s)$) provably preserves the optimal policy** for any potential $\Phi$ — use it for learning-speed guidance; keep economics (costs, risk limits) in the base reward.
6. **Normalize the reward.** Fat tails and regime shifts blow up the critic; `VecNormalize` is the minimum and PopArt is the principled fix for multi-scale rewards.
7. **Reward hacking in finance has recognizable shapes:** volatility-selling (forgets the tail), front-running (data leakage), stat-arb on dataset artifacts, market-making that ignores adverse selection. Remember to include the things that hurt.
8. **Sweep your weights, do not guess them** — every $\lambda, \alpha, \beta, \gamma$ picks a point on the return-vs-risk Pareto front, and the right point is a business decision.
9. **Densify sparse objectives carefully** — rolling/differential Sharpe make end-of-episode Sharpe learnable, but keep checking the dense proxy still tracks the sparse truth.
10. **Monitor the reward-vs-realized gap.** When out-of-sample performance diverges from training reward, Goodhart's Law has found a gap — fix the reward before it costs money.

## Further reading

- Moody, J. and Saffell, M. "Learning to Trade via Direct Reinforcement." *IEEE Transactions on Neural Networks*, 2001. The foundational differential-Sharpe-as-reward paper.
- Ng, A., Harada, D., and Russell, S. "Policy Invariance under Reward Transformations: Theory and Application to Reward Shaping." *ICML*, 1999. The potential-based shaping theorem.
- van Hasselt, H. et al. "Learning values across many orders of magnitude." *NeurIPS*, 2016, and Hessel, M. et al. "Multi-task Deep Reinforcement Learning with PopArt," *AAAI*, 2019. Adaptive reward/target normalization.
- Andrychowicz, M. et al. "Hindsight Experience Replay." *NeurIPS*, 2017. Goal relabeling for sparse rewards.
- Sutton, R. and Barto, A. *Reinforcement Learning: An Introduction*, 2nd ed., 2018. Chapters on rewards, returns, and the credit assignment problem.
- Liu, X.-Y. et al. "FinRL: A Deep Reinforcement Learning Library for Automated Stock Trading." 2020–2022. Standard environments and risk-adjusted reward defaults.
- Within this series: [reward hacking and Goodhart's Law](/blog/machine-learning/reinforcement-learning/reward-hacking-and-goodharts-law), [the policy gradient theorem](/blog/machine-learning/reinforcement-learning/the-policy-gradient-theorem), [the credit assignment problem](/blog/machine-learning/reinforcement-learning/the-credit-assignment-problem), and [model-based versus model-free RL](/blog/machine-learning/reinforcement-learning/model-based-vs-model-free-when-to-use-which). Once published, see also the unified map (reinforcement-learning-a-unified-map) and the capstone playbook (the-reinforcement-learning-playbook) for where reward design sits in the full RL landscape.
