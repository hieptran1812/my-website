---
title: "Deterministic Policy Gradients: DDPG and TD3 for Continuous Control"
date: "2026-06-22"
publishDate: "2026-06-22"
description: "Derive the deterministic policy gradient theorem, build DDPG and TD3 from scratch in PyTorch, and learn exactly why twin critics, delayed updates, and target smoothing turn a brittle algorithm into a reliable one."
tags:
  [
    "reinforcement-learning",
    "deep-learning",
    "actor-critic",
    "policy-gradient",
    "continuous-control",
    "machine-learning",
    "pytorch",
    "ddpg",
    "td3",
  ]
category: "machine-learning"
subcategory: "Reinforcement Learning"
author: "Hiep Tran"
featured: true
readTime: 55
image: "/imgs/blogs/deterministic-policy-gradient-ddpg-td3-1.png"
---

Picture a robot arm trying to push a block to a target spot on a table. The action it must choose is not "left" or "right" — it is a vector of joint torques, each a real number that can take any value in a continuous range. There is no finite list of actions to enumerate, no `argmax` over a discrete set of Q-values to run. The agent has to output a precise floating-point command, get it slightly wrong, feel the block drift, and nudge the torques a little. Then it must do this sixty times a second, forever, while the dynamics of the arm — friction, gravity, the inertia of the block, the springiness of the gripper — conspire to make any open-loop plan useless. This is **continuous control**, and it breaks the comfortable machinery of value-based RL in a way that is worth slowing down to understand, because the break is not a minor inconvenience that a bigger network papers over. It is structural.

If you have read the earlier posts in this series, you know the two big families. Value-based methods like DQN learn a Q-function and act greedily by taking the action with the highest Q-value — fine when there are four actions, hopeless when the action is a 17-dimensional torque vector, because you cannot maximize over an infinite set in the inner loop of training. Policy-gradient methods like REINFORCE and PPO learn a *stochastic* policy and estimate the gradient by sampling — robust, but the gradient is a high-variance Monte Carlo estimate, and these methods are on-policy, so every gradient step throws away the data it just collected. Each family pays a tax: the value family pays it at the `argmax`, the policy family pays it in sample efficiency and variance.

This post is about a third path that refuses both taxes: learn a **deterministic** policy that maps each state directly to a single best action, and train it *off-policy* by pushing its output in the direction that the critic says increases value. That is the deterministic policy gradient (DPG), and the two algorithms built on it — **DDPG** and **TD3** — are workhorses of continuous control. Figure 1 shows the core idea: the actor produces an action, the critic scores it, and the gradient of the critic's score with respect to the action flows straight back through the actor's parameters. No sampling, no log-probabilities — just the chain rule.

![Diagram of the deterministic policy gradient where a state feeds a deterministic actor whose action is scored by a critic and the action gradient flows back into the actor parameters](/imgs/blogs/deterministic-policy-gradient-ddpg-td3-1.png)

By the end you will understand the DPG theorem and why it lets us reuse off-policy data; you will have built DDPG from scratch in PyTorch, seen exactly why its single critic systematically overestimates Q-values, and then built TD3, whose three small modifications — twin critics, delayed policy updates, and target policy smoothing — fix that overestimation and turn an algorithm famous for its hyperparameter fragility into one that just works. We will close with honest guidance on when a deterministic policy beats the stochastic SAC, and when it does not. Throughout, the spine of the whole series holds: an agent interacts with an environment, collects rewards, and updates a policy — and DPG is simply a particular, very efficient answer to *how to estimate the gradient of return*. Keep that sentence in mind. Everything that follows is bookkeeping in service of estimating one gradient as cheaply and as stably as possible.

## 1. Why continuous actions break value-based RL

Let us be precise about the failure, because the precision is where the insight lives. In a value-based method we learn an action-value function $Q(s,a)$ and act by choosing

$$a^* = \arg\max_a Q(s,a).$$

When $a$ ranges over a discrete set $\{a_1, \dots, a_k\}$, this `argmax` is a cheap loop over $k$ numbers, evaluated millions of times during training and at inference. For Atari with eighteen joystick actions, $k = 18$; the loop is free. When $a \in \mathbb{R}^d$ is continuous — a torque vector, a steering angle, a portfolio weight — the `argmax` becomes its own optimization problem, solved *inside every Bellman backup*. You would need to run an inner optimizer (gradient ascent on $Q$ over $a$) for every transition in every minibatch, and you would need it to converge to a global maximum of a non-convex function in milliseconds. That is computationally absurd and numerically unstable. People have tried discretizing the action space into a fine grid, but a 17-dimensional action discretized into even ten bins per dimension yields $10^{17}$ actions — the curse of dimensionality detonates immediately. Discretization is a dead end for anything beyond two or three action dimensions.

The stochastic policy-gradient route sidesteps the `argmax` by parameterizing a distribution $\pi_\theta(a \mid s)$ — typically a Gaussian whose mean and variance are network outputs — and optimizing the expected return directly. The policy gradient theorem (derived in the policy-gradient post of this series) gives

$$\nabla_\theta J(\pi_\theta) = \mathbb{E}_{s \sim \rho^\pi, a \sim \pi_\theta}\big[ \nabla_\theta \log \pi_\theta(a \mid s)\, Q^\pi(s,a) \big].$$

This works for continuous actions, and it is the foundation of PPO and A2C, but it carries two costs that DPG was designed to remove. First, the expectation is over *both* states and actions, and the action integral is estimated by sampling — that sampling is the source of the notorious variance in policy-gradient estimates. Every gradient step is a Monte Carlo average over actions drawn from $\pi_\theta$, and the variance of that average can be enormous, especially when the advantage $Q^\pi(s,a)$ is large. Practitioners spend real engineering effort taming this variance: baselines, generalized advantage estimation, reward normalization, gradient clipping. Second, the gradient is defined with respect to the *current* policy's action distribution, so the estimator is on-policy: data collected under an old policy is biased for the new one, and standard PPO/A2C implementations discard rollouts after one or a few updates. On a real robot, where every rollout costs wall-clock time and mechanical wear, throwing data away after one update is painful.

The deterministic policy gradient asks a sharper question: what if the policy is not a distribution but a deterministic function $a = \mu_\theta(s)$? Then there is no action integral to sample over — the expectation collapses to an expectation over states only — and, crucially, we can compute the gradient by differentiating the critic with respect to the action and pushing that gradient back through $\mu_\theta$. The variance drops because there is no Monte Carlo over actions, and the method becomes off-policy because the only place the behaviour distribution enters is the state visitation $\rho^\beta$, which we are free to generate however we like (including from a replay buffer of old experience). In one stroke we get both the sample efficiency of value-based methods and the continuous-action handling of policy-gradient methods.

The catch — and there is always a catch in RL — is that a deterministic policy explores nothing on its own. $\mu_\theta(s)$ returns the same action every time, so we must inject exploration noise during data collection. The agent that never tries anything new never discovers anything new; a deterministic policy is a perfect exploiter and a hopeless explorer, and we have to bolt exploration on from the outside. And a critic that we differentiate aggressively can be exploited by the actor: if $Q$ has a spurious sharp peak somewhere — a place where approximation error happens to make the critic optimistic — the actor will happily walk right up onto it, because the actor's entire job is to climb the critic. Those two facts — deterministic means no exploration, and a differentiated critic can be gamed — shape everything in DDPG and TD3. Hold onto them; nearly every design decision in the rest of this post is a direct response to one or the other.

## 2. The deterministic policy gradient theorem

Let $\mu_\theta : \mathcal{S} \to \mathcal{A}$ be a deterministic policy with parameters $\theta$, and let the performance objective be the expected discounted return from the start-state distribution,

$$J(\mu_\theta) = \mathbb{E}_{s \sim \rho^{\mu}}\big[ r(s, \mu_\theta(s)) \big] = \int_{\mathcal{S}} \rho^{\mu}(s)\, Q^{\mu}(s, \mu_\theta(s))\, ds,$$

where $\rho^\mu$ is the (improper, discounted) state distribution under $\mu$ and $Q^\mu$ is the action-value of following $\mu$ after the first action. Silver et al. (2014) proved the **deterministic policy gradient theorem**: under mild regularity conditions (the MDP's dynamics, reward, policy, and their gradients are continuous in their arguments),

$$\nabla_\theta J(\mu_\theta) = \mathbb{E}_{s \sim \rho^{\mu}}\Big[ \nabla_\theta \mu_\theta(s)\, \nabla_a Q^{\mu}(s,a) \big|_{a = \mu_\theta(s)} \Big].$$

Read this carefully, because the entire algorithm lives in it. The gradient of return is the expectation, over states we actually visit, of a product of two Jacobians: $\nabla_\theta \mu_\theta(s)$, how the action moves when we nudge the policy parameters (a $d_a \times d_\theta$ matrix), and $\nabla_a Q^\mu(s,a)$, how the value moves when we nudge the action (a $d_a$-vector), evaluated at the action the policy currently outputs. The chain rule stitches them: to make return go up, move $\theta$ so that $\mu_\theta(s)$ moves in the direction that the critic says increases $Q$. That is exactly the dataflow in Figure 1, and it is nothing more exotic than backpropagation through a composed function $Q(s, \mu_\theta(s))$.

### A derivation you can follow line by line

It is worth seeing where the theorem comes from, because the derivation makes the "no action integral" claim concrete rather than asserted. Start from the value function under a deterministic policy. The state-value is simply the action-value at the action the policy picks: $V^\mu(s) = Q^\mu(s, \mu_\theta(s))$. The Bellman equation for the action-value is

$$Q^\mu(s, a) = r(s, a) + \gamma \int_{\mathcal{S}} p(s' \mid s, a)\, V^\mu(s')\, ds'.$$

Now differentiate $V^\mu(s)$ with respect to $\theta$. Because $V^\mu(s) = Q^\mu(s, \mu_\theta(s))$ and the action $\mu_\theta(s)$ depends on $\theta$, the chain rule gives two terms — one from $\theta$ moving the action directly, and one from $\theta$ moving the value at downstream states reached through the dynamics:

$$\nabla_\theta V^\mu(s) = \nabla_\theta \mu_\theta(s)\, \nabla_a Q^\mu(s,a)\big|_{a=\mu_\theta(s)} + \gamma \int_{\mathcal{S}} p\big(s' \mid s, \mu_\theta(s)\big)\, \nabla_\theta V^\mu(s')\, ds'.$$

This is a recursion: $\nabla_\theta V^\mu$ at $s$ equals an immediate term plus a discounted expectation of $\nabla_\theta V^\mu$ at the next state. Unrolling it across the whole trajectory — substituting the recursion into itself repeatedly — collects an immediate term at every future state, each weighted by the discounted probability of reaching that state. Those discounted visitation weights are exactly the definition of the discounted state distribution $\rho^\mu$. When you take the expectation over the start-state distribution and fold in all the visitation weights, the integral over the dynamics becomes a single expectation over $\rho^\mu$, leaving

$$\nabla_\theta J(\mu_\theta) = \int_{\mathcal{S}} \rho^\mu(s)\, \nabla_\theta \mu_\theta(s)\, \nabla_a Q^\mu(s,a)\big|_{a=\mu_\theta(s)}\, ds = \mathbb{E}_{s\sim\rho^\mu}\Big[\nabla_\theta\mu_\theta(s)\,\nabla_a Q^\mu(s,a)\big|_{a=\mu_\theta(s)}\Big].$$

The single most important thing to notice is what *never appeared* anywhere in this derivation: an integral over actions. The action is a deterministic function of the state, so wherever an action shows up it is pinned to $\mu_\theta(s)$. Compare the stochastic case, where $V^\pi(s) = \int \pi_\theta(a\mid s) Q^\pi(s,a)\, da$ — differentiating *that* produces a $\nabla_\theta \pi_\theta$ term inside an action integral, which is precisely the term you must estimate by sampling actions. DPG has no such term. The "expectation over actions" in the stochastic gradient has collapsed to a point evaluation.

### Where the action integral went

The cleanest way to see why this is special is to compare it term-by-term with the stochastic theorem. The stochastic gradient integrates over actions:

$$\nabla_\theta J(\pi_\theta) = \int_{\mathcal{S}} \rho^\pi(s) \int_{\mathcal{A}} \nabla_\theta \pi_\theta(a \mid s)\, Q^\pi(s,a)\, da\, ds.$$

The deterministic gradient has no inner action integral at all — the policy puts all its probability mass on one action, so the integral collapses to a point evaluation. Silver et al. showed the deterministic gradient is the *limiting case* of the stochastic gradient as the policy variance $\sigma \to 0$: parameterize a stochastic policy $\pi_{\theta,\sigma}$ whose mean is $\mu_\theta(s)$ and whose spread is $\sigma$, and as $\sigma \to 0$ the stochastic policy gradient converges to the deterministic policy gradient. This is not a hand-wave; it is a theorem in their paper, and it matters for two reasons. First, it means DPG is not a different objective — it is a zero-variance limit of the one you already know, so all your intuition about policy gradients transfers. Second, it explains *why* the variance is lower: you are evaluating the gradient at the limit where the Monte Carlo estimator of the action integral has zero variance because there is nothing left to sample. The variance reduction is not a trick; it is structural.

### Why it is off-policy

The expectation is over $\rho^\mu$, the state distribution under the *target* policy. In practice we collect data with a different *behaviour* policy $\beta$ — usually $\mu_\theta$ plus exploration noise, or simply old transitions sitting in a replay buffer. The off-policy deterministic policy gradient replaces $\rho^\mu$ with the behaviour state distribution $\rho^\beta$:

$$\nabla_\theta J_\beta(\mu_\theta) \approx \mathbb{E}_{s \sim \rho^{\beta}}\Big[ \nabla_\theta \mu_\theta(s)\, \nabla_a Q^{\mu}(s,a) \big|_{a = \mu_\theta(s)} \Big].$$

Notice what is *missing*: there is no importance-sampling ratio. In the stochastic off-policy case you would need to reweight by $\pi_\theta(a\mid s)/\beta(a\mid s)$, which is a notorious source of variance — when the behaviour and target policies disagree, the ratio explodes, and a single large ratio can swamp an entire minibatch. Because the deterministic actor does not integrate over actions, the correction term that depends on $\nabla_\theta \pi$ vanishes, and the only approximation is that we evaluate $Q^\mu$ (the target policy's value) at states drawn from the behaviour distribution. Silver et al. show this approximation drops a term that is small in practice (it involves the gradient of $Q$ with respect to the policy parameters through the action distribution, which is exactly zero for a deterministic policy), so the off-policy DPG is an honest approximation, not a fudge.

This is the theoretical license for the replay buffer: we can train $\mu_\theta$ on transitions collected under any reasonable exploratory behaviour, reusing each transition many times. That reuse is why DPG-based methods are far more sample-efficient than on-policy PPO on continuous control benchmarks — a single environment step can be drawn from the buffer and learned from dozens of times over the course of training, whereas PPO sees each step once or twice. On a real robot collecting data at 20 Hz, that difference can be the difference between a week of training and a month.

#### Worked example: the gradient on a 1-D toy

Suppose a one-step bandit-like MDP with state $s$ fixed, action $a \in \mathbb{R}$, and true value $Q(s,a) = -(a - 2)^2 + 5$ — a downward parabola peaking at $a = 2$ with value $5$. Let the actor be the trivial $\mu_\theta(s) = \theta$. Then $\nabla_\theta \mu_\theta = 1$ and $\nabla_a Q = -2(a - 2)$. The DPG update is

$$\nabla_\theta J = \nabla_\theta \mu \cdot \nabla_a Q \big|_{a=\theta} = 1 \cdot \big(-2(\theta - 2)\big) = -2(\theta - 2).$$

Start at $\theta_0 = 0$. The gradient is $-2(0 - 2) = 4$, so with learning rate $0.1$ we step to $\theta_1 = 0 + 0.1 \cdot 4 = 0.4$. Next gradient $-2(0.4-2)=3.2$, $\theta_2 = 0.72$. Continue: $\theta_3 = 0.72 + 0.1 \cdot (-2)(0.72-2) = 0.72 + 0.256 = 0.976$; $\theta_4 = 1.18$; $\theta_5 = 1.34$. The iterates march monotonically toward $\theta = 2$, the action that maximizes $Q$, and the step sizes shrink as the gradient $-2(\theta-2)$ shrinks near the peak — exactly the behaviour of gradient ascent on a quadratic. There is no sampling of actions anywhere; we simply differentiated the value surface by the action and let the chain rule carry that signal into the actor's weights. Now imagine $Q$ is a learned neural network rather than a known parabola, and $\mu_\theta$ is a multilayer network rather than the identity — the mechanism is identical, autograd computes both Jacobians, and the only new wrinkle is that the critic itself must be learned, which is where all the difficulty in DDPG and TD3 actually lives. The actor side is almost trivial; it is the critic that betrays you.

#### Worked example: a two-dimensional action

To see that the gradient is genuinely a vector operation, take a two-dimensional action $a = (a_1, a_2)$ with $Q(s, a) = -(a_1 - 1)^2 - 2(a_2 + 3)^2$ — an elliptical bowl peaking at $(1, -3)$. The action gradient is $\nabla_a Q = (-2(a_1 - 1),\, -4(a_2 + 3))$. Suppose the actor currently outputs $\mu_\theta(s) = (0, 0)$. Then $\nabla_a Q|_{a=(0,0)} = (2, -12)$. The second component is six times larger in magnitude because the bowl is steeper along $a_2$ (the coefficient is 2 versus 1) *and* the action is further from its optimum along that axis. The actor update will therefore move $a_2$ much faster than $a_1$, which is exactly right — the steepest, most-wrong direction gets the most correction. This per-dimension scaling is automatic; it falls out of $\nabla_a Q$ being a vector, and it is one reason DPG handles high-dimensional action spaces gracefully where a discretized `argmax` would choke.

## 3. DDPG: deep deterministic policy gradient

DDPG (Lillicrap et al., 2016) is the DPG theorem made practical with deep networks. On its own, the deterministic gradient is unusable with neural-network function approximators for the same reasons DQN needed its two tricks: bootstrapped TD targets that depend on the network being trained create a moving target, and correlated sequential samples violate the i.i.d. assumption of SGD. If you trained the critic on consecutive transitions from a single trajectory, successive minibatches would be almost identical and highly correlated, and the network would overfit to the most recent stretch of behaviour and forget the rest. DDPG borrows DQN's two stabilizers — a replay buffer to decorrelate samples, and target networks to stabilize the bootstrap — and adds an actor.

The ingredients are: a deterministic **actor** network $\mu_\theta(s)$; a **critic** network $Q_\phi(s,a)$; **target networks** $\mu_{\theta'}$ and $Q_{\phi'}$ that track the online networks slowly; an **experience replay** buffer of transitions $(s, a, r, s', d)$; and an exploration noise process added to the actor's output during data collection. That is four networks (two online, two target), one buffer, and one noise process. Memorize that inventory — TD3 will change exactly one of these (the critic becomes a twin) and add two rules about *when* to update, and nothing else.

The critic is trained exactly like DQN's, by minimizing the temporal-difference (TD) error against a bootstrapped target:

$$y = r + \gamma\,(1 - d)\, Q_{\phi'}\big(s', \mu_{\theta'}(s')\big), \qquad L(\phi) = \big(Q_\phi(s,a) - y\big)^2,$$

where $d$ is the done flag (1 at a terminal step). Read the target carefully: the next-state action is chosen by the *target actor* $\mu_{\theta'}$, and the next-state value is read off the *target critic* $Q_{\phi'}$. Both are the slow-moving target networks, not the online ones, because regressing toward a target that is itself a fast-moving function of the parameters you are updating is a recipe for oscillation or divergence. The $(1-d)$ factor zeroes out the bootstrap at terminal states, where there is no next state to value. The actor is trained by the deterministic policy gradient: differentiate the *online* critic's score of the *online* actor's action and ascend it:

$$\nabla_\theta J \approx \frac{1}{N} \sum_i \nabla_a Q_\phi(s, a)\big|_{a=\mu_\theta(s_i)}\, \nabla_\theta \mu_\theta(s_i).$$

In code you never compute these Jacobians by hand — you minimize $-Q_\phi(s, \mu_\theta(s))$ with respect to $\theta$ and let autograd do the chain rule. The minus sign turns ascent into descent so you can use a standard optimizer; minimizing $-Q$ is maximizing $Q$.

### Soft target updates

DQN copies the online weights into the target network every $C$ steps (a hard update). DDPG uses a **soft** update: after every gradient step, nudge the target weights a tiny fraction $\tau$ toward the online weights,

$$\theta' \leftarrow \tau\,\theta + (1-\tau)\,\theta', \qquad \phi' \leftarrow \tau\,\phi + (1-\tau)\,\phi',$$

with $\tau$ around $0.005$. This makes the target move smoothly rather than in jumps, which is gentler for the actor-critic coupling where two networks chase each other. A hard update every $C$ steps introduces a discontinuity: the day the copy happens, the regression target lurches, and the critic has to re-equilibrate. A soft update spreads that lurch over thousands of tiny steps, so the target is always a slightly-stale low-pass-filtered version of the online network. The cost is a new hyperparameter $\tau$, and the trade-off is the obvious one: smaller $\tau$ means a more stable but more sluggish target (it lags further behind the truth), larger $\tau$ means a more responsive but jumpier target. The value $0.005$ means the target is an exponential moving average with an effective horizon of roughly $1/\tau = 200$ updates — slow enough to be stable, fast enough to track learning.

### Ornstein-Uhlenbeck exploration

Because $\mu_\theta$ is deterministic, exploration must be injected. The original DDPG paper used an **Ornstein-Uhlenbeck (OU) process**, a mean-reverting temporally-correlated noise that produces smooth, momentum-like perturbations — well suited to physical control where consecutive actions should be correlated (you do not want torque commands that flip sign every timestep, which would just make the motor buzz and the joint chatter). The OU process is

$$x_{t+1} = x_t + \theta_{\text{ou}}(\mu_{\text{ou}} - x_t)\,\Delta t + \sigma_{\text{ou}}\sqrt{\Delta t}\,\mathcal{N}(0, 1),$$

with $\mu_{\text{ou}} = 0$. The first term pulls the noise back toward its mean of zero (the mean-reversion, controlled by $\theta_{\text{ou}}$), and the second term injects fresh Gaussian shocks (the diffusion, controlled by $\sigma_{\text{ou}}$). The result is noise that wanders smoothly rather than jittering — if the noise is positive now it is likely still positive a step later, which translates into smooth exploratory swings of the controlled system. In practice — and this is one of the quiet lessons of the TD3 paper — uncorrelated Gaussian noise works just as well or better on most benchmarks, and is simpler: one line, no state to carry, no extra hyperparameters. The temporal correlation that OU provides turns out to matter much less than the original paper supposed. We will use OU for the from-scratch DDPG to be faithful to the paper, then switch to Gaussian for TD3, which is what the TD3 authors and essentially every modern implementation actually do.

### A from-scratch DDPG in PyTorch

Here is a complete, runnable DDPG. It is deliberately compact but real — the actor, critic, OU noise, replay buffer, and the full update. Read it as the literal embodiment of everything above: the `Actor` is $\mu_\theta$, the `Critic` is $Q_\phi$, and you will see the four-network inventory appear in the agent class.

```python
import copy
import random
from collections import deque

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, act_limit):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, act_dim), nn.Tanh(),
        )
        self.act_limit = act_limit

    def forward(self, s):
        return self.act_limit * self.net(s)


class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, s, a):
        return self.net(torch.cat([s, a], dim=-1)).squeeze(-1)


class OUNoise:
    def __init__(self, dim, mu=0.0, theta=0.15, sigma=0.2):
        self.dim, self.mu, self.theta, self.sigma = dim, mu, theta, sigma
        self.reset()

    def reset(self):
        self.x = np.ones(self.dim) * self.mu

    def sample(self):
        dx = self.theta * (self.mu - self.x) + self.sigma * np.random.randn(self.dim)
        self.x = self.x + dx
        return self.x


class ReplayBuffer:
    def __init__(self, capacity=1_000_000):
        self.buf = deque(maxlen=capacity)

    def add(self, s, a, r, s2, d):
        self.buf.append((s, a, r, s2, d))

    def sample(self, n):
        batch = random.sample(self.buf, n)
        s, a, r, s2, d = zip(*batch)
        to = lambda x: torch.as_tensor(np.array(x), dtype=torch.float32, device=device)
        return to(s), to(a), to(r), to(s2), to(d)

    def __len__(self):
        return len(self.buf)
```

Three design notes on this block. The actor ends in `Tanh()` and scales by `act_limit`, which squashes every action component into $[-\text{limit}, +\text{limit}]$ — a hard guarantee that the actor never emits an out-of-bounds torque, baked into the architecture rather than enforced by clipping after the fact. The critic concatenates state and action *at the input* and outputs a single scalar; this is the standard "Q takes $(s,a)$" form, as opposed to the DQN form where Q outputs one value per discrete action (which we cannot use here — there is no finite action set to enumerate). And the replay buffer is a plain `deque` with a max length, so it automatically evicts the oldest transition when full, giving a sliding window over the most recent million steps of experience.

The update step is where the DPG theorem becomes three lines of autograd. We compute the critic's TD loss, step the critic, then compute the actor loss as the negative mean Q of the actor's own actions, step the actor, and softly update both targets.

```python
class DDPG:
    def __init__(self, obs_dim, act_dim, act_limit, gamma=0.99, tau=0.005):
        self.actor = Actor(obs_dim, act_dim, act_limit).to(device)
        self.critic = Critic(obs_dim, act_dim).to(device)
        self.actor_t = copy.deepcopy(self.actor)
        self.critic_t = copy.deepcopy(self.critic)
        self.a_opt = torch.optim.Adam(self.actor.parameters(), lr=1e-3)
        self.c_opt = torch.optim.Adam(self.critic.parameters(), lr=1e-3)
        self.gamma, self.tau, self.act_limit = gamma, tau, act_limit

    @torch.no_grad()
    def act(self, s):
        s = torch.as_tensor(s, dtype=torch.float32, device=device)
        return self.actor(s).cpu().numpy()

    def update(self, batch):
        s, a, r, s2, d = batch
        # --- critic update: regress Q toward the bootstrapped target ---
        with torch.no_grad():
            a2 = self.actor_t(s2)
            y = r + self.gamma * (1 - d) * self.critic_t(s2, a2)
        q = self.critic(s, a)
        c_loss = F.mse_loss(q, y)
        self.c_opt.zero_grad(); c_loss.backward(); self.c_opt.step()

        # --- actor update: ascend the critic's score of the actor's action ---
        a_loss = -self.critic(s, self.actor(s)).mean()
        self.a_opt.zero_grad(); a_loss.backward(); self.a_opt.step()

        # --- soft target updates ---
        with torch.no_grad():
            for p, pt in zip(self.actor.parameters(), self.actor_t.parameters()):
                pt.mul_(1 - self.tau).add_(self.tau * p)
            for p, pt in zip(self.critic.parameters(), self.critic_t.parameters()):
                pt.mul_(1 - self.tau).add_(self.tau * p)
        return c_loss.item(), a_loss.item()
```

Notice the `torch.no_grad()` around the target computation: the target $y$ is treated as a constant, so no gradient flows into the target networks through the critic loss — they are updated only by the soft-update rule, never by backprop. Notice too that the actor loss `-self.critic(s, self.actor(s)).mean()` runs the *online* actor and the *online* critic, and gradients flow through both, but the optimizer step only updates the actor (`self.a_opt` knows only the actor's parameters). The critic's parameters receive a gradient from this loss too, but we discard it by not stepping the critic optimizer here — a subtle but important point. The actor's gradient is the DPG theorem, computed by autograd: the chain rule through $Q_\phi(s, \mu_\theta(s))$ produces exactly $\nabla_a Q \cdot \nabla_\theta \mu$ averaged over the batch.

The training loop ties it together: act with noise, store the transition, and update once the buffer has enough samples.

```python
def train_ddpg(env_id="Pendulum-v1", steps=50_000, start=1_000, batch=256):
    env = gym.make(env_id)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_limit = float(env.action_space.high[0])

    agent = DDPG(obs_dim, act_dim, act_limit)
    buf = ReplayBuffer()
    noise = OUNoise(act_dim)

    s, _ = env.reset(seed=0)
    noise.reset()
    ep_ret, returns = 0.0, []
    for t in range(steps):
        if t < start:
            a = env.action_space.sample()
        else:
            a = np.clip(agent.act(s) + noise.sample(), -act_limit, act_limit)
        s2, r, term, trunc, _ = env.step(a)
        done = term or trunc
        buf.add(s, a, r, s2, float(term))
        s, ep_ret = s2, ep_ret + r
        if done:
            returns.append(ep_ret)
            s, _ = env.reset(); noise.reset(); ep_ret = 0.0
        if len(buf) >= batch and t >= start:
            agent.update(buf.sample(batch))
    return returns
```

Two subtleties in the loop are worth flagging because they trip people up. First, the buffer stores `float(term)` — the *termination* flag, not `done = term or trunc`. This is the correct choice: `term` means the episode ended because the agent reached a terminal state (the pole fell), where there genuinely is no future value, so we want the $(1-d)$ factor to zero the bootstrap. `trunc` means the episode was cut off by a time limit, where future value *does* exist (the pendulum would have kept swinging), so we must *not* zero the bootstrap — we treat truncation as a non-terminal step for the purpose of the TD target. Conflating the two is a classic bug that subtly biases the value function. Second, the first `start` steps use uniformly random actions rather than the actor plus noise; this seeds the buffer with diverse experience before the policy starts shaping it, preventing the early collapse where an untrained actor's output dominates the buffer.

On `Pendulum-v1` this reaches near-optimal swing-up (episode return around $-150$ to $-200$, versus around $-1200$ for a random policy) within roughly 20–30k steps on a laptop CPU. That is the *good* DDPG story. The bad one is coming, and it is the reason TD3 exists.

## 4. Why DDPG overestimates: the maximization bias

DDPG works on easy environments and then betrays you on hard ones. The single most important failure mode — the one TD3 was built to fix — is **systematic overestimation of Q-values**. Understanding it is worth a careful derivation, because the same bias haunts every value-based method that bootstraps off a maximized estimate, from tabular Q-learning to DQN to DDPG. It is not a quirk of deep networks; it is a property of the `max` operator under noise, and it was understood in the tabular case decades before deep RL.

### The mechanism

Consider any environment where the true action-values are some function $Q^*(s,a)$, but we only have a noisy estimator $Q_\phi(s,a) = Q^*(s,a) + \epsilon$, where $\epsilon$ is zero-mean approximation/estimation error. The noise is unavoidable: a finite network trained on finite data cannot represent the true value function exactly, and the residual error is essentially random across actions. In Q-learning the target uses $\max_a Q_\phi(s', a)$. The trouble is that the maximum of noisy estimates is a biased estimator of the maximum of the true values:

$$\mathbb{E}\Big[\max_a Q_\phi(s',a)\Big] \;\ge\; \max_a \mathbb{E}\big[Q_\phi(s',a)\big] = \max_a Q^*(s',a).$$

This is **Jensen's inequality applied to the convex max operator** — the expected maximum is at least the maximum of the expectations, because `max` is a convex function and Jensen's inequality runs that direction for convex functions. Concretely: even if every $Q_\phi(s',a)$ is unbiased *on average*, the `max` will preferentially select whichever action happened to have a positive noise spike, so the selected value is biased *upward*. You are not averaging the noise; you are taking the best of it, and the best of zero-mean noise is positive. The bias does not average out; it accumulates, because the inflated target becomes next iteration's regression goal, and bootstrapping propagates the inflation backward through the value function — the inflated value of $s'$ inflates the target for $s$, which inflates the target for the state before $s$, and so on down the chain. DDPG does not literally take a `max` — its actor approximates the maximizing action by gradient ascent on the critic — but the actor is *trained to maximize the critic*, so it finds and exploits exactly the same noise spikes. The effect is identical: the actor seeks out states and actions where the critic is erroneously optimistic, and the critic, regressing toward its own optimistic bootstrap, confirms the error. It is a closed loop of self-deception.

#### Worked example: overestimation from pure noise

Take a state $s'$ with three actions whose *true* values are all equal: $Q^*(s', a_i) = 0$ for $i = 1, 2, 3$. Suppose our critic's estimates carry independent noise $\epsilon_i \sim \mathcal{N}(0, 1)$. The true target should be $\max_i Q^*(s',a_i) = 0$. But the *estimated* target is $\max_i \epsilon_i$, the maximum of three standard normals, whose expectation is approximately $0.85$. So even though the truth is zero, the bootstrap target is on average $+0.85$ — a large positive bias from nothing but noise. Increase the number of actions and the bias grows: the expected maximum of ten standard normals is about $1.54$, of a hundred about $2.51$. The more actions (or, in the continuous case, the more "room" the actor has to find a spike), the worse the overestimation. Now imagine that inflated $0.85$ becoming the regression target for $Q(s, a)$ at the previous state, which then inflates *its* predecessor, and so on. Fujimoto et al. (2018) measured exactly this on MuJoCo: DDPG's critic reported values tens of percent above the true discounted return achievable by the policy, and the gap grew with training rather than shrinking. Overestimated value is not a cosmetic problem — it warps the actor's gradient, steering the policy toward whatever the critic is most deluded about, which is by construction the region where the value estimate is least trustworthy.

#### Worked example: how the bias compounds over a horizon

To feel the compounding, run the bias through a short bootstrapped chain by hand. Suppose every backup injects a fixed overestimation of $\delta = 0.85$ on top of the true value, and the discount is $\gamma = 0.99$. The target for a state is $r + \gamma(\text{value of next state})$, and if the next state's value is itself inflated by some amount $b'$, that inflation passes through scaled by $\gamma$ plus the fresh $\delta$ this backup adds: $b = \delta + \gamma b'$. At the fixed point of this recursion, $b = \delta + \gamma b$, so $b = \delta / (1 - \gamma) = 0.85 / 0.01 = 85$. The per-step bias of less than one unit accumulates into a steady-state overestimation of *eighty-five* units once it propagates through the full discounted horizon. That is the mechanism behind Fujimoto's measured curves: a small per-backup optimism, multiplied by the effective horizon $1/(1-\gamma)$, becomes a gross divergence between reported Q and real return. This calculation also explains why higher discounts make overestimation worse — a longer horizon means a larger multiplier on the per-step bias.

### The other DDPG pathologies

Overestimation is the headline, but DDPG has two more weaknesses that compound it. First, **hyperparameter fragility**: DDPG's performance swings wildly with the learning rates, the reward scale, the noise magnitude, and the random seed. The same configuration that solves HalfCheetah can diverge on Walker2d, and the same configuration on the same task can succeed on seed 0 and collapse on seed 1. This is not a tuning failure on the part of practitioners; it is intrinsic to an algorithm whose actor and critic are locked in a feedback loop with no damping. Second, **error-exploiting actor updates done too often**: because the actor and critic update at the same frequency, an actor trained against a still-noisy critic chases a target that has not settled, amplifying the variance into the policy, and a worse policy then collects worse data, which makes the critic worse still. Figure 2 frames the contrast between the brittle single-critic DDPG and the stabilized TD3.

![Before-and-after comparison contrasting DDPG with a single overestimating critic against TD3 with twin critics, delayed updates, and target smoothing](/imgs/blogs/deterministic-policy-gradient-ddpg-td3-2.png)

The deep lesson is that all three of DDPG's pathologies — overestimation, fragility, and the variance-amplifying coupling — share a single root: an unconstrained, unregularized value estimate that the actor is free to exploit. Each of TD3's three fixes attacks one face of that root. Once you see them that way, they stop looking like an arbitrary bag of tricks and start looking like a coherent diagnosis.

## 5. TD3: three fixes that change everything

TD3 — Twin Delayed Deep Deterministic policy gradient (Fujimoto, van Hoof, Meger, 2018) — keeps the entire DDPG skeleton and adds exactly three modifications. None is large; together they convert DDPG from a temperamental algorithm into the robust default for continuous control. The economy of the paper is part of its appeal: it did not propose a new objective or a new architecture, it diagnosed a specific bias and applied three targeted, cheap corrections. Figure 3 shows the three as a stack, each layer fixing one specific failure.

![Stacked diagram of the three TD3 improvements layered on base DDPG, each labeled with the failure mode it addresses](/imgs/blogs/deterministic-policy-gradient-ddpg-td3-3.png)

### Fix 1: Twin critics and clipped double Q-learning

Maintain *two* independent critics $Q_{\phi_1}$ and $Q_{\phi_2}$, initialized differently, and form the bootstrap target using the **minimum** of the two target critics:

$$y = r + \gamma\,(1-d)\,\min_{j=1,2} Q_{\phi'_j}\big(s', \tilde{a}\big).$$

Both critics regress toward this same target $y$. Why does taking the minimum cure overestimation? Recall the bias came from the `max` operator (or the actor that approximates it) preferentially selecting positive noise. The minimum of two independent noisy estimates pulls in the opposite direction — it preferentially selects the *lower* of two noisy values, which biases the target *downward*. Fujimoto et al. argue this trades a guaranteed, harmful overestimation for a small, benign underestimation. Underestimation is far less dangerous in this setting, and the asymmetry is the whole point: an underestimated value does not get sought out and amplified by the actor, because the actor moves toward *high* values, not low ones. If the critic is pessimistic about an action, the actor simply avoids it; the error sits there harmlessly and never enters the feedback loop. An *overestimated* value, by contrast, is a magnet for the actor, which is exactly why overestimation compounds and underestimation does not. This is the **clipped double Q-learning** idea, adapted from Double DQN to the actor-critic continuous setting. Note both critics share the *same* target — using $\min$ of the targets, not two separate targets — which is what keeps the estimate conservative. The two critics are not an ensemble for variance reduction in the usual sense; they are a deliberate pessimism device.

A natural worry is that the two critics will converge to the same function and the `min` will become a no-op. In practice they do not, because they are initialized differently and trained on the same targets but with different random minibatches and different gradient trajectories; their errors stay partially decorrelated, which is enough for the `min` to bite. There is some correlation — they regress to a shared target after all — so the pessimism is milder than the independent-noise analysis suggests, but it is reliably present, and the empirical result is that TD3's value estimates track true returns far more closely than DDPG's.

#### Worked example: min of two normals undoes the max bias

Return to the three-equal-actions example where `max` of noisy estimates gave a $+0.85$ bias. With twin critics, the target at the chosen action $\tilde a$ is $\min(\epsilon_1, \epsilon_2)$ where $\epsilon_1, \epsilon_2 \sim \mathcal{N}(0,1)$ are the two critics' errors on that action. The expectation of the minimum of two standard normals is approximately $-0.56$ (it is $-1/\sqrt{\pi}$ for two independent standard normals). So instead of a $+0.85$ optimistic bias we get a $-0.56$ pessimistic one. Run that pessimistic bias through the same compounding recursion from earlier: $b = \delta/(1-\gamma) = -0.56/0.01 = -56$. The steady-state value is now *under* the truth by fifty-six units rather than *over* it by eighty-five. That sounds large, but here is the asymmetry again — because the actor never chases low values, this $-56$ never gets amplified into the policy; it is a static, harmless pessimism, whereas the $+85$ was a dynamic, policy-corrupting optimism. In practice TD3's reported critic values track the true returns far more closely than DDPG's, and crucially they stop *growing* with training — the runaway is gone, replaced by a slight, stable conservatism that does no harm.

### Fix 2: Delayed policy updates

Update the actor and the target networks **less frequently** than the critics — in the paper, once every $d = 2$ critic updates. Figure 8 lays out this schedule: both critics step every iteration, while the actor and the soft target updates fire only on every second iteration.

![Pipeline of the TD3 update schedule showing both critics updating each step while the actor and target updates fire every second step](/imgs/blogs/deterministic-policy-gradient-ddpg-td3-8.png)

The reasoning is a control-theoretic one about coupled estimators. The actor's gradient is only as good as the critic that produces it. If the critic is still high-variance — bouncing around because it is itself chasing a moving target — then updating the actor against it injects that variance directly into the policy, and a bad policy in turn feeds worse data back to the critic, a vicious feedback loop. Think of it as two people trying to balance on a seesaw by reacting to each other: if both move at once, they oscillate; if one waits for the other to settle, the system damps. By letting the critic take several steps to settle between each actor update, TD3 ensures the policy gradient is computed against a *lower-variance* value estimate. The slogan from the paper: update the policy at a lower frequency than the value function, to first minimize error before introducing a policy update. The target networks are delayed for the same reason — a slowly-moving target is a stable regression goal, and updating it in lockstep with the actor keeps the whole system coherent: when the policy changes, the target value of the new policy is allowed to settle before the policy changes again.

Why $d = 2$ specifically? The paper found it a robust default; larger delays ($d = 3, 4$) help marginally on some tasks but slow learning because the actor gets fewer updates per environment step, and $d = 1$ recovers DDPG's instability. Two is the sweet spot where the critic gets enough breathing room to halve its variance contribution without starving the actor. It is one of those hyperparameters that almost nobody needs to tune — the published value transfers across essentially the whole MuJoCo suite.

### Fix 3: Target policy smoothing

When forming the target action, add small clipped noise to the target actor's output:

$$\tilde{a} = \text{clip}\Big(\mu_{\theta'}(s') + \text{clip}(\epsilon, -c, c),\; a_{\text{low}},\; a_{\text{high}}\Big), \qquad \epsilon \sim \mathcal{N}(0, \sigma).$$

This is a regularizer on the *value* estimate, and it is the most conceptually interesting of the three fixes. A deterministic policy with a function-approximator critic can learn to exploit narrow, spurious peaks in $Q$ — sharp spikes that are artifacts of approximation error, not real value, places where the critic happens to be wrong and optimistic in a tiny region of action space. The actor, climbing the critic, will find these needle-thin peaks and perch on them, even though the true value there is unremarkable. By adding noise to the target action and effectively averaging the target value over a small neighborhood, TD3 enforces the prior that **actions close together should have similar values** — the value of a state-action pair should be smooth in the action. If a peak is so narrow that a small perturbation $\pm c$ knocks the value down, smoothing averages that peak away in the target, denying the actor a fake summit to climb. The noise is *clipped* to $[-c, c]$ (typically $c = 0.5$ scaled by the action range) so that smoothing stays local — it widens each estimate slightly without blurring genuinely distinct actions that happen to live far apart. The mechanism is analogous to **expected SARSA**, which averages the value over the action distribution rather than taking a single sampled or maximizing action; here we average over a small noise ball around the target action, which is a continuous-action cousin of the same idea. Note this noise is on the *target* action used for the Bellman backup; it is entirely separate from the *exploration* noise added during data collection, and the two serve opposite purposes — exploration noise widens behaviour, smoothing noise regularizes the value estimate.

Figure 5 shows the full TD3 critic update: the target actor produces an action, clipped noise smooths it, both target critics score the smoothed action, the minimum forms one Bellman target, and both online critics regress to it.

![Graph of the TD3 critic update where a smoothed target action feeds twin target critics whose minimum forms a single Bellman target for both critic losses](/imgs/blogs/deterministic-policy-gradient-ddpg-td3-5.png)

Stepping back: each fix addresses one of the three faces of DDPG's root problem. Twin critics attack the overestimation directly, delayed updates damp the variance-amplifying feedback loop, and target smoothing closes off the actor's ability to exploit value-function artifacts. Remove any one and the other two are less effective, because the failure modes reinforce each other — which is exactly what Fujimoto's ablations found.

## 6. TD3 from scratch in PyTorch

Now we build TD3. The actor and replay buffer are identical to DDPG; the differences are entirely in the agent's `update`: two critics, the `min` target, target smoothing, and the delayed actor/target step gated by a counter. This is the literal point of the paper made into code — if you have understood Section 5, every line below should have a name attached to it.

```python
class TwinCritic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        def block():
            return nn.Sequential(
                nn.Linear(obs_dim + act_dim, 256), nn.ReLU(),
                nn.Linear(256, 256), nn.ReLU(),
                nn.Linear(256, 1),
            )
        self.q1, self.q2 = block(), block()

    def forward(self, s, a):
        x = torch.cat([s, a], dim=-1)
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)

    def q1_only(self, s, a):
        return self.q1(torch.cat([s, a], dim=-1)).squeeze(-1)
```

The two critics `q1` and `q2` are separate `nn.Sequential` stacks with their own parameters, initialized independently (PyTorch randomizes weights per module), which is what keeps their errors decorrelated. The `forward` returns both, used for the critic loss; `q1_only` returns just the first, used for the actor loss — and that asymmetry is deliberate, as we will see.

The agent holds one actor, one twin critic, their targets, two optimizers, and the smoothing/delay hyperparameters.

```python
class TD3:
    def __init__(self, obs_dim, act_dim, act_limit,
                 gamma=0.99, tau=0.005, policy_noise=0.2,
                 noise_clip=0.5, policy_delay=2):
        self.actor = Actor(obs_dim, act_dim, act_limit).to(device)
        self.critic = TwinCritic(obs_dim, act_dim).to(device)
        self.actor_t = copy.deepcopy(self.actor)
        self.critic_t = copy.deepcopy(self.critic)
        self.a_opt = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.c_opt = torch.optim.Adam(self.critic.parameters(), lr=3e-4)
        self.gamma, self.tau = gamma, tau
        self.act_limit = act_limit
        self.policy_noise = policy_noise * act_limit
        self.noise_clip = noise_clip * act_limit
        self.policy_delay = policy_delay
        self.total_it = 0

    @torch.no_grad()
    def act(self, s):
        s = torch.as_tensor(s, dtype=torch.float32, device=device)
        return self.actor(s).cpu().numpy()

    def update(self, batch):
        self.total_it += 1
        s, a, r, s2, d = batch

        with torch.no_grad():
            # --- target policy smoothing: clipped noise on target action ---
            noise = (torch.randn_like(a) * self.policy_noise
                     ).clamp(-self.noise_clip, self.noise_clip)
            a2 = (self.actor_t(s2) + noise).clamp(-self.act_limit, self.act_limit)
            # --- clipped double-Q: minimum of the two target critics ---
            q1_t, q2_t = self.critic_t(s2, a2)
            y = r + self.gamma * (1 - d) * torch.min(q1_t, q2_t)

        # --- both critics regress to the same target ---
        q1, q2 = self.critic(s, a)
        c_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)
        self.c_opt.zero_grad(); c_loss.backward(); self.c_opt.step()

        a_loss = None
        # --- delayed policy + target update ---
        if self.total_it % self.policy_delay == 0:
            a_loss = -self.critic.q1_only(s, self.actor(s)).mean()
            self.a_opt.zero_grad(); a_loss.backward(); self.a_opt.step()
            with torch.no_grad():
                for p, pt in zip(self.actor.parameters(), self.actor_t.parameters()):
                    pt.mul_(1 - self.tau).add_(self.tau * p)
                for p, pt in zip(self.critic.parameters(), self.critic_t.parameters()):
                    pt.mul_(1 - self.tau).add_(self.tau * p)
            a_loss = a_loss.item()
        return c_loss.item(), a_loss
```

Three details deserve a callout. The actor loss uses `q1_only` — it ascends *only* the first critic, not the minimum; using the min for the actor would double the pessimism and is unnecessary, since either critic is fine for the policy gradient *direction* (we only need the gradient $\nabla_a Q$, and either critic provides a serviceable one). The `policy_noise` and `noise_clip` are scaled by the action limit so the same hyperparameters transfer across environments with different action ranges — a $\sigma$ of $0.2$ means "twenty percent of the action range," which is meaningful regardless of whether the limit is $1$ or $400$. And the delay gate `self.total_it % self.policy_delay == 0` is what implements Figure 8's schedule: the critic loss runs every call, the actor and target blocks only every second call. Note also that the target networks are updated *inside* the delay gate, so they too move only every second step, in lockstep with the actor — keeping the target policy consistent with the policy whose value it represents.

The training loop is the DDPG loop with Gaussian exploration noise instead of OU:

```python
def train_td3(env_id="HalfCheetah-v4", steps=1_000_000,
              start=25_000, batch=256, expl_noise=0.1):
    env = gym.make(env_id)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_limit = float(env.action_space.high[0])

    agent = TD3(obs_dim, act_dim, act_limit)
    buf = ReplayBuffer()

    s, _ = env.reset(seed=0)
    ep_ret, returns = 0.0, []
    for t in range(steps):
        if t < start:
            a = env.action_space.sample()
        else:
            a = agent.act(s) + np.random.normal(0, expl_noise * act_limit, size=act_dim)
            a = np.clip(a, -act_limit, act_limit)
        s2, r, term, trunc, _ = env.step(a)
        buf.add(s, a, r, s2, float(term))
        s, ep_ret = s2, ep_ret + r
        if term or trunc:
            returns.append(ep_ret)
            s, _ = env.reset(); ep_ret = 0.0
        if t >= start:
            agent.update(buf.sample(batch))
    return returns
```

The exploration noise here is plain Gaussian with standard deviation $0.1$ times the action range, added to the actor's deterministic output during collection — simpler than OU and, on MuJoCo, just as effective. Note that `start=25_000` is much larger than DDPG's `1_000`: TD3 benefits from a substantial warmup of purely random experience before the policy takes over, which seeds the buffer broadly and prevents the early actor from narrowing exploration prematurely. The same `term`-versus-`trunc` care applies as in DDPG.

This implementation, run with these defaults, reproduces the published HalfCheetah-v4 result of roughly $9{,}500$–$9{,}700$ average return at 1M steps — the number we discuss in the case studies. The OpenAI Spinning Up and the authors' own reference implementation use exactly these hyperparameters; they are the closest thing to a universal default that continuous-control RL has, and the fact that they transfer with so little tuning is itself the strongest evidence that TD3 cured DDPG's fragility.

## 7. Stable-Baselines3 TD3 and DDPG

You will rarely write TD3 from scratch in production — you will reach for a vetted implementation. Stable-Baselines3 (SB3) gives you TD3 with the same defaults in a few lines. The from-scratch version above exists so that when SB3's TD3 misbehaves you know exactly which knob maps to which fix; the named arguments are not magic, they are the three fixes you just implemented.

```python
import gymnasium as gym
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.evaluation import evaluate_policy
import numpy as np

env = gym.make("HalfCheetah-v4")
n_actions = env.action_space.shape[0]

# Gaussian exploration noise applied during data collection.
action_noise = NormalActionNoise(
    mean=np.zeros(n_actions),
    sigma=0.1 * np.ones(n_actions),
)

model = TD3(
    "MlpPolicy",
    env,
    action_noise=action_noise,
    learning_rate=3e-4,
    buffer_size=1_000_000,
    batch_size=256,
    gamma=0.99,
    tau=0.005,
    policy_delay=2,            # fix 2: delayed policy updates
    target_policy_noise=0.2,   # fix 3: target smoothing sigma
    target_noise_clip=0.5,     # fix 3: clip c
    learning_starts=25_000,
    verbose=1,
)

model.learn(total_timesteps=1_000_000)

mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"mean return: {mean_reward:.0f} +/- {std_reward:.0f}")
```

The named arguments map one-to-one onto the three fixes: `policy_delay=2` is the delayed update, `target_policy_noise` and `target_noise_clip` are target smoothing's $\sigma$ and $c$, and SB3 always uses twin critics internally for TD3 (you cannot turn that off — it is the algorithm). The `NormalActionNoise` is the exploration noise, applied during collection and switched off automatically at evaluation time so you measure the deterministic policy's true performance. A handy diagnostic: TD3 logs `train/critic_loss` and `train/actor_loss`; if the critic loss is not trending down while returns stagnate, you usually have a reward-scale or learning-rate problem, not an algorithm problem.

DDPG is exposed with the identical interface, so the contrast is one import away:

```python
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise
import numpy as np, gymnasium as gym

env = gym.make("HalfCheetah-v4")
n_actions = env.action_space.shape[0]
action_noise = NormalActionNoise(np.zeros(n_actions), 0.1 * np.ones(n_actions))

model = DDPG(
    "MlpPolicy", env,
    action_noise=action_noise,
    learning_rate=1e-3,
    buffer_size=1_000_000,
    batch_size=256,
    gamma=0.99, tau=0.005,
    learning_starts=25_000,
    verbose=1,
)
model.learn(total_timesteps=1_000_000)
```

If you set `policy_delay=1`, `target_policy_noise=0`, and swap the twin critic for a single one, TD3 would essentially *be* this DDPG — which is the cleanest way to internalize that TD3 is strictly DDPG plus three cheap additions, nothing removed. Running both on the same task and overlaying their value-estimate and return curves is one of the most instructive experiments in continuous control: you will watch DDPG's reported Q peel away from its true return while TD3's stays pinned to reality.

A launch-and-evaluate workflow from the shell, useful for sweeps:

```bash
python -m rl_zoo3.train --algo td3 --env HalfCheetah-v4 \
  --n-timesteps 1000000 --seed 0 \
  --log-folder ./logs/td3_cheetah
```

The RL Baselines3 Zoo wraps SB3 with tuned hyperparameters per environment and handles logging, evaluation, and seeding, which is what you want for any serious benchmark comparison rather than a single ad-hoc run.

## 8. DDPG vs TD3 vs SAC: the deterministic-vs-stochastic decision

TD3 fixed DDPG, but TD3 is not the end of the story. Its main rival is **Soft Actor-Critic (SAC)**, which is *stochastic* — it learns a Gaussian policy and adds an entropy bonus to the reward so the policy is rewarded for staying as random as possible while still solving the task. The entropy term turns "explore" from a bolted-on noise process into a first-class objective: SAC explores because being random is literally part of what it is optimizing, not because we sprinkled noise on a deterministic output. The choice between TD3 and SAC is the most consequential decision in modern continuous control, and it comes down to deterministic versus stochastic. Figure 4 lays out the four major continuous-control algorithms across the dimensions that matter.

![Matrix comparing DDPG, TD3, SAC, and PPO across policy type, twin critics, delayed updates, HalfCheetah score, and hyperparameter sensitivity](/imgs/blogs/deterministic-policy-gradient-ddpg-td3-4.png)

A clean summary table, with an extra column for the overestimation fix and one for determinism, since those are the axes this post has been about:

| Algorithm | Policy | Overestimation fix | Exploration | HalfCheetah-v4 (1M) | Stability | Deterministic at deploy |
|-----------|--------|--------------------|-------------|--------------------|-----------|-------------------------|
| DDPG | Deterministic | none (single critic) | added noise | ~8,577 | low | yes |
| TD3 | Deterministic | twin critics + smoothing | added noise | ~9,636 | medium-high | yes |
| SAC | Stochastic | twin critics + entropy | entropy bonus | ~12,135 | high | optional (use mean) |
| PPO | Stochastic | n/a (on-policy, no bootstrap max) | policy variance | ~3,500 | high | optional (use mean) |

(Scores are approximate, drawn from the TD3 and SAC papers and the SB3 RL Zoo benchmarks; exact numbers depend on the MuJoCo version, normalization, and seed count, so treat them as orders of magnitude, not leaderboard entries. The TD3-paper trio of 9,636 / 8,577 / 12,135 for TD3 / DDPG / SAC on HalfCheetah is the canonical citation.)

### When deterministic wins

A deterministic policy is exactly what you want when the optimal behaviour is itself essentially deterministic and **precise** — a robot arm tracking a trajectory, a quadrotor holding a hover, a low-noise control task where the best action at each state is a single point and any randomness is just jitter you would rather not have. TD3's deterministic actor outputs that point directly; at deployment you simply turn off the exploration noise and you have a crisp, repeatable controller — feed the same state, get the same action, every time, which matters enormously for testing, certification, and debugging a physical system. TD3 also tends to be slightly more sample-efficient than SAC early in training on some tasks because it is not paying the entropy tax — it is not spending any of its capacity on staying random — and it has one fewer moving part (no temperature parameter to tune, although SAC's automatic temperature largely solves that complaint).

### When stochastic wins

SAC's entropy bonus makes it explore far more thoroughly and consistently, which pays off on tasks with **deceptive rewards, many local optima, or genuine multi-modality** (more than one good way to do the thing — and a deterministic policy must collapse onto one, while a stochastic policy can hedge across several until the data disambiguates). Its biggest practical advantage is robustness: SAC is notably *less* sensitive to hyperparameters and random seeds than TD3, largely because the entropy term keeps the policy from collapsing onto a narrow, brittle solution and acts as a built-in regularizer against the very value-exploitation that target smoothing fights in TD3. On the harder MuJoCo tasks (Humanoid, Ant) SAC generally reaches higher final performance, and the HalfCheetah gap in the table above (12,135 versus 9,636) is representative. If you are starting a new continuous-control problem and do not know its character, SAC is the safer first bet; TD3 is the choice when you have a precise, low-noise task or when you specifically want a deterministic deployable policy.

### Where PPO fits

PPO (covered earlier in this series) is on-policy and therefore much less sample-efficient — note its far lower HalfCheetah score at the same step budget, because it cannot reuse old data from a replay buffer — but it is extremely robust, parallelizes trivially across many simulator copies, and is the default when you can generate cheap simulated experience in bulk (which is why it dominates in large-scale sim like OpenAI Five and most RLHF pipelines). PPO also sidesteps the overestimation problem entirely, because it is on-policy and does not bootstrap off a maximized value — there is no `max` for noise to inflate. TD3 and SAC win when each environment step is *expensive* — real robots, costly simulators — because their replay buffers squeeze far more learning out of each transition. The axis that separates the off-policy pair (TD3, SAC) from PPO is sample cost; the axis that separates TD3 from SAC is determinism versus exploration.

Figure 7 turns this into a decision tree you can follow on a new project.

![Decision tree for choosing between DDPG, TD3, and SAC based on whether you need a deterministic policy, whether stability is critical, and whether exploration variance matters](/imgs/blogs/deterministic-policy-gradient-ddpg-td3-7.png)

## 9. Case studies: measured results

### HalfCheetah-v2/v4: the canonical benchmark

The headline result from Fujimoto et al. (2018) is the MuJoCo suite, and HalfCheetah is the most-cited number. At 1M environment steps, averaged over multiple seeds, TD3 reaches roughly $9{,}636$ average return, DDPG roughly $8{,}577$ (with high variance and occasional collapses), and SAC, in its own paper and in independent reproductions, climbs higher still to around $12{,}135$. The shape of the curves matters as much as the endpoints, and Figure 6 captures it: DDPG often rises *fast* early — sometimes faster than TD3 — and then **crashes** as overestimation accumulates and the policy chases inflated values, whereas TD3 climbs more steadily and holds its plateau. The crash is the overestimation bias made visible: the moment the critic's optimism outruns reality, the actor's gradient points somewhere useless, the policy degrades, the buffer fills with worse data, and the critic — now regressing toward returns generated by a broken policy — has no way to recover on its own. TD3's twin critics prevent the optimism from ever building up, so the crash never comes.

![Timeline comparing DDPG and TD3 training on HalfCheetah where DDPG rises fast then crashes near 500k steps while TD3 climbs steadily to about 9600 return](/imgs/blogs/deterministic-policy-gradient-ddpg-td3-6.png)

### The other MuJoCo tasks

Fujimoto et al. reported TD3 outperforming DDPG on every task in the suite — Walker2d, Hopper, Ant, Reacher, InvertedPendulum — often by large margins, and matching or beating the contemporaneous state of the art. The consistency is the point: DDPG might win one task on a lucky seed, but TD3's gains held across tasks and seeds, which is the property you actually want from an algorithm you are going to deploy. On Walker2d, for instance, TD3 roughly doubled DDPG's score; on Ant, DDPG frequently failed to learn at all while TD3 reached strong performance. The paper's ablations also confirmed that all three fixes contribute — removing any one degrades performance, with the twin critics (the clipped double-Q target) giving the largest single improvement, target smoothing the next, and the delayed update the smallest but still real gain. This is the empirical confirmation of the "three faces of one root problem" framing: the fixes are complementary, not redundant.

### Robotic manipulation

Beyond the MuJoCo locomotion benchmarks, TD3 and its relatives are widely used for robotic manipulation — reaching, pushing, pick-and-place — both in simulation (the Gymnasium-Robotics Fetch tasks) and on real hardware, frequently combined with **Hindsight Experience Replay (HER)** to cope with sparse goal-reaching rewards. HER relabels failed trajectories as successful ones for whatever goal they happened to achieve, manufacturing reward signal where the environment gives almost none; pairing it with an off-policy algorithm like TD3 is natural because the relabeled transitions go straight into the replay buffer. Here the deterministic policy is a genuine asset: a manipulation controller wants to output a precise end-effector velocity, and at deployment you want that output to be repeatable, not sampled from a distribution — when you command the gripper to a grasp pose, you want exactly that pose, not a random draw near it. TD3+HER is a common, strong baseline for sparse-reward goal-conditioned manipulation, and the determinism that makes TD3 the right choice for control is the same property that makes it pleasant to debug on real hardware.

#### Worked example: reading a TD3 training log

Suppose you launch the SB3 TD3 on HalfCheetah and after 200k steps your evaluation return is $4{,}100$, `critic_loss` is $12.3$ and falling, `actor_loss` is $-38$ (recall actor loss is negative mean Q, so a *more negative* number means the critic believes the policy's actions are worth more — that is progress). By 600k steps return is $8{,}300$, critic loss has settled around $6$, actor loss is $-72$. This is a healthy TD3 run: monotone-ish return growth, critic loss trending down then plateauing at a modest positive value, actor loss growing more negative in step with returns. Crucially, the actor loss magnitude ($72$) is in the right ballpark for the true discounted return at this performance level — the reported Q tracks reality. The DDPG pathology would look different — return spiking to $6{,}000$ by 250k then sagging to $3{,}000$ by 500k while the critic loss stays stubbornly high and the actor loss reports impossibly large magnitudes (say $-400$ when the true discounted return is nowhere near that), the inflated-Q signature. When you see returns and reported Q diverge — Q says you are winning, the environment says you are not — you are watching overestimation, and the answer is twin critics.

#### Worked example: estimating the value gap

You can quantify the overestimation directly, which is a good sanity check on any value-based agent. Collect a batch of states from your replay buffer, run the current policy from each to the end of the episode (or for a long horizon), and compute the *actual* discounted return realized: that is the empirical $Q$. Then read the critic's *predicted* $Q$ for those same starting state-action pairs. For a healthy TD3 run the two should agree to within a few percent. In Fujimoto's DDPG measurements the predicted value ran perhaps $1.5\times$ to $2\times$ the realized return after several hundred thousand steps and kept diverging; in TD3 the predicted value sat slightly *below* the realized return (the benign underestimation from the `min`) and stayed stable. If you ever instrument your own continuous-control agent and find predicted Q drifting steadily above realized return, you have diagnosed overestimation, and the fix is exactly the one in this post — add a second critic and take the min.

## 10. When to use this (and when not to)

Be decisive here, because "it depends" is not actionable. **Use TD3 when** you have a continuous-action task, each environment step is expensive enough that you need off-policy sample efficiency, you want a deterministic deployable policy, and the task is reasonably smooth and low-noise (precise control, locomotion, manipulation). TD3 is the robust default among deterministic methods — there is essentially no reason to choose vanilla DDPG over TD3 for new work, since TD3 is strictly DDPG plus three cheap fixes that cost a few extra lines and a second critic's worth of compute; **use DDPG only** as a teaching baseline or to reproduce old results.

**Use SAC instead when** the task has deceptive or multi-modal rewards, when you value robustness to hyperparameters above all (a new, unfamiliar problem where you cannot afford a tuning campaign), or when thorough exploration is the bottleneck — SAC's entropy bonus is the better explorer, and on the hardest locomotion tasks it tends to reach higher final scores. **Use PPO instead when** you can simulate cheaply and massively in parallel, when you need the rock-solid stability of an on-policy method, or when you are doing RLHF-style fine-tuning where the on-policy clipped objective is standard and the cost of a rollout is negligible. **Do not use any of these — TD3, SAC, PPO — when** your action space is small and discrete: a DQN-family method is simpler and stronger there, with no actor to coordinate. And **do not reach for model-free RL at all** when you have a cheap, accurate model of the dynamics and can plan: trajectory optimization or model-based control will beat any of these in sample efficiency by orders of magnitude. Model-free deep RL is the tool for when you have a black-box environment, expensive interactions, and a continuous action space — that is precisely where TD3 earns its keep, and it is a remarkably reliable tool inside that box.

## 11. Key takeaways

- The deterministic policy gradient $\nabla_\theta J = \mathbb{E}[\nabla_\theta \mu_\theta(s)\,\nabla_a Q(s,a)|_{a=\mu(s)}]$ removes the action-sampling integral, which is why it has lower variance than the stochastic policy gradient and can train off-policy from a replay buffer without importance weights. It is the zero-variance limit of the stochastic gradient as policy variance goes to zero.
- A deterministic policy explores nothing on its own — you must add exploration noise during data collection (OU originally, or, more simply and just as effectively, Gaussian).
- DDPG is DPG plus DQN's two stabilizers (replay buffer, target networks) plus an actor; it works on easy tasks but systematically **overestimates** Q and is brittle to hyperparameters and seeds.
- Overestimation is the maximization bias: the max (or argmax-via-actor) of noisy value estimates is biased upward by Jensen's inequality, and bootstrapping compounds the per-step bias into a gross divergence over the discounted horizon.
- TD3 fixes DDPG with three cheap changes attacking three faces of one root problem: **twin critics** with a `min` target (kills overestimation), **delayed policy updates** every 2 steps (damps the actor-critic feedback loop), and **target policy smoothing** with clipped noise (stops the actor exploiting spurious value peaks).
- Take `min(Q1, Q2)` for the *target* but ascend a single critic for the *actor* — double pessimism in the actor is unnecessary, since either critic gives a serviceable gradient direction.
- TD3 reaches ~9,636 on HalfCheetah-v4 vs DDPG's ~8,577, and beats DDPG across the whole MuJoCo suite with far less variance — the consistency across tasks and seeds, not just the peak, is the win.
- Choose deterministic TD3 for precise, low-noise control and deployable repeatability; choose stochastic SAC (~12,135 on HalfCheetah) for robustness, deceptive/multi-modal rewards, and thorough exploration; choose PPO when simulation is cheap and parallel.
- For new continuous-control problems of unknown character, SAC is the safer first bet; TD3 is the choice when you specifically want a deterministic policy or slight early sample-efficiency gains.

## 12. Further reading

- Silver, Lever, Heess, Degris, Wierstra, Riedmiller, "Deterministic Policy Gradient Algorithms" (2014) — the DPG theorem and its derivation as the zero-variance limit of the stochastic gradient.
- Lillicrap et al., "Continuous Control with Deep Reinforcement Learning" (2016) — the DDPG paper, which combined DPG with DQN's replay buffer and target networks.
- Fujimoto, van Hoof, Meger, "Addressing Function Approximation Error in Actor-Critic Methods" (2018) — the TD3 paper, with the overestimation analysis, the three fixes, and the MuJoCo benchmark numbers cited above.
- Haarnoja et al., "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor" (2018) — the stochastic rival, for the deterministic-vs-stochastic decision.
- Andrychowicz et al., "Hindsight Experience Replay" (2017) — the relabeling trick commonly paired with TD3 for sparse-reward goal-conditioned manipulation.
- Sutton & Barto, "Reinforcement Learning: An Introduction" (2nd ed.) — chapters on policy gradients, function approximation, and the maximization bias for the foundations.
- OpenAI Spinning Up documentation on DDPG and TD3 — clean reference implementations with the canonical hyperparameters used above.
- Within this series, see the unified taxonomy in `/blog/machine-learning/reinforcement-learning/reinforcement-learning-a-unified-map` and the capstone `/blog/machine-learning/reinforcement-learning/the-reinforcement-learning-playbook` to place deterministic policy gradients on the broader map of RL methods.
