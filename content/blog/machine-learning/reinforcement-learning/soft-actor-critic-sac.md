---
title: "Soft Actor-Critic: Maximum Entropy RL for Continuous Control"
date: "2026-06-22"
publishDate: "2026-06-22"
description: "A from-scratch, theory-to-production walkthrough of Soft Actor-Critic — why the entropy bonus changes the objective, how the twin critics and auto-tuned temperature work, and full PyTorch and Stable-Baselines3 code that hits 12,000 return on HalfCheetah."
tags:
  [
    "reinforcement-learning",
    "deep-learning",
    "actor-critic",
    "continuous-control",
    "exploration",
    "machine-learning",
    "pytorch",
    "maximum-entropy",
  ]
category: "machine-learning"
subcategory: "Reinforcement Learning"
author: "Hiep Tran"
featured: true
readTime: 55
image: "/imgs/blogs/soft-actor-critic-sac-1.png"
---

The first time I trained a robot arm in simulation, I used a tuned DDPG agent. It worked — once. I changed the random seed and it collapsed into a corner of the action space and stayed there, jittering, for two million steps. The policy had become deterministic far too early: it found one mediocre behavior and stopped looking. When I dropped that same task into Soft Actor-Critic with almost no tuning, it converged on the first try, on the second seed, on a Tuesday afternoon, and it kept a little bit of randomness in its hands the whole time — which, it turned out, is exactly why it survived when I later perturbed the joint friction. That contrast is the whole story of this post.

Soft Actor-Critic (SAC), introduced by Haarnoja et al. in 2018, is the algorithm I reach for first on any continuous-control problem: robot joints, vehicle steering, portfolio weights, anything where the action is a real-valued vector rather than a discrete menu. It is off-policy (so it reuses old data and is sample-efficient), it is stable across hyperparameters in a way that DDPG never was, and it owes both of those properties to one deceptively simple idea: instead of maximizing reward alone, it maximizes reward *plus the entropy of the policy*. Figure 1 shows the machinery we will build piece by piece.

![Diagram of the SAC architecture showing one stochastic actor feeding twin critics whose minimum drives the actor loss, critic loss, and temperature update](/imgs/blogs/soft-actor-critic-sac-1.png)

By the end of this post you will understand the maximum-entropy objective and *why* it is a different objective rather than a bolted-on exploration trick; you will be able to derive the soft Bellman backup, the squashed-Gaussian actor update, and the dual problem that auto-tunes the temperature; and you will have a complete PyTorch implementation plus the three-line Stable-Baselines3 equivalent, both of which reach roughly 12,000 average return on HalfCheetah-v4. We will keep tying everything back to the series spine: an agent interacts with an environment, collects rewards, and updates a policy — SAC is simply a particular, unusually robust answer to *which objective to optimize* and *how to estimate its gradient*. For the bigger picture see the [unified map of RL algorithms](/blog/machine-learning/reinforcement-learning/reinforcement-learning-a-unified-map).

## 1. The problem SAC was built to solve

Continuous control is where the classic deep-RL workhorses get uncomfortable. You cannot run Q-learning's `argmax` over actions when there are infinitely many of them, so DQN is out. Policy-gradient methods like PPO work, but they are on-policy: every gradient step throws away the data it just collected, which makes them sample-hungry — fine in a fast simulator, painful on a real robot where every episode costs wall-clock time and wear on the hardware. DDPG and its successor TD3 are off-policy and do handle continuous actions, but DDPG in particular is famous for being brittle: a deterministic policy with a separate exploration noise process that you have to tune, and a single critic that systematically overestimates values.

Three failure modes recur in this regime. First, **premature determinism**: the policy commits to one behavior before it has seen enough of the space, the exact trap my robot arm fell into. Second, **value overestimation**: a single learned Q-function, maximized over actions, accumulates an optimistic bias because the max of noisy estimates is itself biased upward, and the policy then chases that phantom value. Third, **hyperparameter fragility**: the reward scale, the exploration noise, and the learning rates all interact, so a setting that works on Hopper fails on Humanoid.

SAC attacks all three at once. The entropy bonus keeps the policy stochastic and exploratory by *design* rather than by an external noise schedule, killing premature determinism. Twin critics with a minimum operator suppress overestimation. And automatic temperature tuning removes the single most finicky knob — how much to value exploration — by solving for it. The result is the most reliable continuous-control algorithm in common use, and the rest of this post explains exactly how each mechanism earns its keep.

It helps to be precise about what "continuous control" means as a formal object before we start optimizing it. We are in a Markov decision process (MDP) defined by a tuple $(\mathcal{S}, \mathcal{A}, p, r, \gamma)$: a state space $\mathcal{S}$ (here a real vector — joint angles, velocities, sensor readings), an action space $\mathcal{A} \subseteq \mathbb{R}^d$ that is now a bounded box rather than a finite menu, a transition density $p(s' \mid s, a)$, a reward function $r(s, a)$, and a discount $\gamma \in [0, 1)$. A policy $\pi(a \mid s)$ is a *distribution* over actions given a state. The agent's job is to find the policy that maximizes expected discounted return. The defining difficulty of the continuous case is that we can no longer enumerate actions: the operations "take the best action" ($\arg\max_a Q(s,a)$) and "average over all actions" ($\sum_a$) both become intractable integrals over $\mathbb{R}^d$. Every continuous-control algorithm is, at heart, a different strategy for approximating those two operations, and SAC's strategy — sample from a learned stochastic policy and differentiate through the sample — is what gives it its character.

A second framing worth internalizing early: the distinction between *on-policy* and *off-policy* learning, because it is the axis along which SAC's sample efficiency lives. An on-policy algorithm can only learn from data generated by the policy it is currently improving; the moment you update the policy, the old data is stale and must be discarded. PPO is on-policy. An off-policy algorithm can learn from data generated by *any* policy, including old versions of itself stored in a replay buffer, because it learns a value function (the critic) that is defined independently of the data-collection policy. SAC is off-policy, which is precisely why it can reuse a million stored transitions and why it needs an order of magnitude fewer environment interactions than PPO to reach the same competence.

Let me make the sample-efficiency claim concrete, because it is the whole economic argument for using an off-policy method. Suppose your environment runs at 1,000 steps per second in a simulator — that is a fast MuJoCo task. PPO collects a rollout of, say, 2,048 steps, does a handful of epochs of minibatch updates over exactly those 2,048 transitions, and then throws every one of them away. Each transition contributes to the gradient a fixed small number of times and is then gone forever. SAC, by contrast, stores every transition in a buffer that holds a million of them, and at every environment step it draws a fresh minibatch of 256 transitions for a gradient update. Over a million environment steps, a single transition collected early can participate in *hundreds* of gradient updates as it lingers in the buffer. That reuse is the lever: the same hard-won environment interaction does far more learning work. On a real robot, where 1,000 steps might be 30 seconds of physical motion rather than a millisecond of simulation, that lever is the difference between a two-hour training run and a two-week one.

The flip side, which we will return to in the "when not to use" section, is that off-policy reuse assumes the stored transitions remain *informative* — that the dynamics have not shifted out from under the buffer. If the environment is highly non-stationary, the buffer fills with stale dynamics and the off-policy assumption quietly breaks. For the vast majority of robotics and control problems the dynamics are stationary enough that this is a non-issue, which is why off-policy methods dominate the field.

| Algorithm | Action space | On/off-policy | Exploration | Headline weakness |
| --- | --- | --- | --- | --- |
| DQN | Discrete only | Off-policy | epsilon-greedy | No continuous actions |
| PPO | Both | On-policy | Stochastic policy | Sample-hungry |
| DDPG | Continuous | Off-policy | External noise | Brittle, overestimates |
| TD3 | Continuous | Off-policy | External noise | Still needs noise tuning |
| SAC | Continuous | Off-policy | Built-in entropy | More compute per step |

A useful way to read that table is as a lineage. DQN proved deep networks could carry a Q-function for discrete actions. DDPG ported the idea to continuous actions by replacing the `argmax` with a learned deterministic actor — but inherited DQN's overestimation and added a brittle external-noise exploration scheme. TD3 patched DDPG's overestimation with twin critics and delayed updates but kept the deterministic-plus-noise exploration. SAC took TD3's twin critics, threw out the deterministic actor and external noise in favor of a stochastic actor whose randomness *is* the exploration, and grounded the whole thing in the maximum-entropy objective. Each step in the lineage fixed a specific failure of its predecessor, and SAC sits at the end of it as the synthesis.

## 2. Maximum-entropy RL: a different objective, not a trick

Standard RL maximizes the expected discounted return:

$$J_{\text{std}}(\pi) = \mathbb{E}_{\tau \sim \pi}\left[\sum_{t=0}^{\infty} \gamma^t\, r(s_t, a_t)\right].$$

Maximum-entropy RL changes the objective itself by adding the entropy of the policy at every timestep:

$$J(\pi) = \mathbb{E}_{\tau \sim \pi}\left[\sum_{t=0}^{\infty} \gamma^t \left( r(s_t, a_t) + \alpha\, \mathcal{H}\big(\pi(\cdot \mid s_t)\big)\right)\right],$$

where the entropy of a continuous policy is $\mathcal{H}(\pi(\cdot \mid s)) = -\mathbb{E}_{a \sim \pi}[\log \pi(a \mid s)]$ and $\alpha \ge 0$ is the *temperature* that trades reward against entropy. This is the equation worth tattooing on your forearm; everything else in SAC is a consequence of it.

Let me unpack the objective slowly, because every symbol earns its place. The outer expectation $\mathbb{E}_{\tau \sim \pi}$ averages over whole trajectories $\tau = (s_0, a_0, s_1, a_1, \dots)$ generated by running the policy $\pi$ through the environment. Inside, at each timestep $t$, we sum two things, both discounted by $\gamma^t$. The first is the ordinary reward $r(s_t, a_t)$. The second is $\alpha \mathcal{H}(\pi(\cdot \mid s_t))$ — the temperature times the *entropy of the policy's action distribution at that state*. Note carefully: the entropy is a property of the distribution $\pi(\cdot \mid s_t)$, not of the particular action $a_t$ that was sampled. It measures how spread out the policy's choices are at $s_t$. A policy that always picks the same action there has zero (or, for continuous distributions, very negative) entropy; a policy that picks uniformly across the action box has maximal entropy.

So the agent is rewarded, at every step, both for the reward it collects *and* for keeping its options open. Rewriting the per-step term as $r(s_t, a_t) - \alpha \log \pi(a_t \mid s_t)$ (since the entropy is the expected value of $-\log \pi$) gives the form we will actually use in the Bellman equation: the agent collects reward minus the temperature times the log-probability of the action it took. Improbable-under-its-own-policy actions are rewarded, which is just another way of saying the agent is paid to stay uncertain.

Why is this not merely an exploration heuristic, the way an entropy *regularizer* is in vanilla policy gradients? Because the entropy term lives *inside* the value function. The agent does not just get a one-time nudge to be random at the current step; it is rewarded for reaching *future* states from which it will still have many good options. The optimal MaxEnt policy is the one that, at every state, is as random as possible *while still* collecting high reward. Formally, the optimal policy has a closed form: it is a Boltzmann distribution over soft Q-values,

$$\pi^*(a \mid s) \propto \exp\!\left(\tfrac{1}{\alpha} Q_{\text{soft}}^*(s, a)\right),$$

which makes the connection explicit — actions with higher soft value are exponentially more likely, but no action is ever assigned probability zero unless its value is $-\infty$. A standard greedy policy is the $\alpha \to 0$ limit of this, where the exponential collapses onto the single best action. So the entropy term is a knob that continuously interpolates between "always pick the argmax" and "pick uniformly at random," and the optimum is genuinely different for every $\alpha > 0$, not just a noisier version of the same answer.

It is worth deriving that Boltzmann form, because it is short and it cements *why* the entropy term reshapes the optimum rather than just perturbing it. Fix a state $s$ and ask: among all distributions $\pi(\cdot \mid s)$, which one maximizes the one-step soft objective $\mathbb{E}_{a \sim \pi}[Q(s,a)] + \alpha \mathcal{H}(\pi(\cdot \mid s))$? Write the entropy out and add a Lagrange multiplier $\lambda$ to enforce that $\pi$ integrates to one:

$$\mathcal{L} = \int \pi(a)\,Q(s,a)\,da - \alpha \int \pi(a)\log\pi(a)\,da + \lambda\!\left(\int \pi(a)\,da - 1\right).$$

Take the functional derivative with respect to $\pi(a)$ and set it to zero: $Q(s,a) - \alpha(\log\pi(a) + 1) + \lambda = 0$. Solving for $\pi(a)$ gives $\pi(a) = \exp\!\big(\tfrac{1}{\alpha}Q(s,a) + \tfrac{\lambda}{\alpha} - 1\big)$, and absorbing the constants into a normalizer $Z(s)$ yields exactly $\pi^*(a \mid s) = \frac{1}{Z(s)}\exp(\tfrac{1}{\alpha}Q(s,a))$. The temperature $\alpha$ sits in the exponent's denominator: large $\alpha$ flattens the distribution toward uniform, small $\alpha$ sharpens it toward the argmax. This is not a noisy version of the greedy answer — it is a categorically different object, a *distribution* whose shape is dictated by the value surface and the temperature together.

![Before-and-after comparison contrasting reward-only standard RL with a brittle greedy policy against maximum-entropy RL with a robust stochastic policy](/imgs/blogs/soft-actor-critic-sac-2.png)

The practical payoff, shown in Figure 2, is robustness. A MaxEnt policy that keeps probability mass spread across several near-optimal actions degrades gracefully when the dynamics shift slightly — exactly the situation in sim-to-real transfer. The policy has, in effect, learned a *family* of ways to solve the task rather than one fragile recipe. Haarnoja et al. demonstrated this directly: MaxEnt policies transferred to perturbed environments far better than their reward-only counterparts.

There is a deeper reason the entropy term improves *optimization*, separate from its effect on the final policy, and it is worth dwelling on because it explains why SAC trains so smoothly. Consider the loss landscape of a reward-only critic. When the policy is nearly deterministic, the gradient that the actor receives comes from a single point in action space — wherever the policy currently puts its mass. If that point sits on a flat or misleading part of the Q-surface, the actor has no signal telling it where the better actions are; it is gradient-descending with a one-pixel view of the landscape. A high-entropy policy, by contrast, spreads its samples across a *region* of action space, so the expected gradient effectively averages the Q-surface's slope over that region. This smooths the optimization landscape and provides gradient information about a neighborhood rather than a point. It is the same reason temperature helps in simulated annealing: a little noise lets you feel the shape of the basin you are in, rather than committing to the first downhill direction you find. As training proceeds and $\alpha$ shrinks, the effective temperature cools, the policy sharpens, and the agent commits — but only after it has explored enough to commit wisely.

There is also a probabilistic-inference reading of MaxEnt RL that some practitioners find clarifying. If you introduce a binary "optimality" variable $\mathcal{O}_t$ at each timestep whose likelihood is $p(\mathcal{O}_t = 1 \mid s_t, a_t) \propto \exp(r(s_t, a_t))$, then inferring the posterior distribution over actions given that all timesteps are optimal turns out to be *exactly* the maximum-entropy RL problem. Reward becomes log-likelihood, the entropy term falls out of the variational lower bound, and the soft value functions are the inference messages. This "RL as inference" framing, developed by Levine and others, is why SAC's machinery looks so much like variational inference — the reparameterization trick, the KL-style policy improvement — and it is no coincidence: SAC is, under this lens, approximate inference in a graphical model where being optimal is the evidence we condition on.

Let me also be explicit about a unit subtlety that trips people up. Entropy for a continuous distribution is *differential* entropy, measured in nats, and unlike discrete entropy it can be negative. A very peaked Gaussian (small $\sigma$) has negative differential entropy; a broad one has large positive entropy. Concretely, a one-dimensional Gaussian with standard deviation $\sigma$ has differential entropy $\tfrac{1}{2}\log(2\pi e \sigma^2)$, which goes negative as soon as $\sigma < 1/\sqrt{2\pi e} \approx 0.242$. This is why the target entropy in SAC's auto-tuning, $\bar{\mathcal{H}} = -\dim(\mathcal{A})$, is a *negative* number — it corresponds to a moderately peaked policy, not a maximally spread one. Reading the entropy as "how many nats of randomness per action dimension" keeps the bookkeeping straight.

#### Worked example: entropy as insurance on a 2-action bandit

Suppose at some state two actions both yield expected reward near 1.0: action A gives exactly 1.00 and action B gives 0.98. A greedy policy puts all mass on A. Now the environment changes and A's reward silently drops to 0.50 while B stays at 0.98. The greedy agent collects 0.50 until it relearns; it is blind to B. A MaxEnt policy with a modest temperature might have split mass 0.6 / 0.4 across A and B, so its expected reward after the shift is $0.6 \times 0.50 + 0.4 \times 0.98 = 0.692$ — and crucially it is *still sampling* B, so it discovers the new ranking almost immediately. The small entropy "tax" it paid before the shift (collecting $0.6 \times 1.00 + 0.4 \times 0.98 = 0.992$ instead of 1.00) bought it insurance worth far more than 0.008 when the world moved.

We can even tie the 0.6 / 0.4 split back to the Boltzmann form. For two actions with values 1.00 and 0.98, the softmax probability of A is $\frac{e^{1.00/\alpha}}{e^{1.00/\alpha} + e^{0.98/\alpha}} = \frac{1}{1 + e^{-0.02/\alpha}}$. To get the 0.6 / 0.4 split we need $e^{-0.02/\alpha} = 0.4/0.6 = 0.667$, so $-0.02/\alpha = \log 0.667 = -0.405$, giving $\alpha \approx 0.049$. A temperature of about 0.05 produces exactly that hedge. Crank $\alpha$ down to 0.005 and the split becomes roughly 0.98 / 0.02 — almost greedy, almost no insurance. Crank it up to 0.5 and the split is nearly 0.51 / 0.49 — maximal hedging but you are now leaving real reward on the table on the much larger gaps you will encounter elsewhere. The temperature *is* the price you are willing to pay for the insurance, and the auto-tuning in Section 7 sets that price for you.

## 3. The temperature parameter alpha

The temperature $\alpha$ is the exchange rate between a unit of reward and a nat of entropy. Three regimes are worth internalizing. When $\alpha \to 0$, the entropy term vanishes and we recover standard RL with a near-deterministic optimal policy. When $\alpha \to \infty$, reward becomes irrelevant and the optimal policy is uniform — maximum randomness, zero competence. For any finite $\alpha$ in between, the policy is a tempered Boltzmann distribution that balances the two.

The trouble is that the *right* $\alpha$ is not a universal constant. It depends on the reward scale (rewards in the thousands need a larger $\alpha$ to matter than rewards near 1.0), on how far into training you are (early on you want more entropy, late you want less), and on the dimensionality of the action space. In the original 2018 SAC paper, Haarnoja et al. tuned $\alpha$ by hand per environment, and it was the single most sensitive hyperparameter. Their follow-up — and the version everyone uses today — replaces hand-tuning with a *constrained* optimization that solves for $\alpha$ automatically. We will derive that dual problem in Section 7; for now, hold the intuition that $\alpha$ is a learnable Lagrange multiplier enforcing a target amount of entropy.

The reward-scale dependence deserves emphasis because it bites people who try to port hyperparameters between tasks. The objective adds $r$ and $\alpha \mathcal{H}$ in the *same units*. If your environment hands out rewards on the order of thousands per episode (HalfCheetah) and your entropy is a handful of nats, then a temperature of 0.2 makes the entropy term a rounding error — the agent will behave nearly greedily. If instead your rewards are sparse and tiny, on the order of 0.01 per step, then that same $\alpha = 0.2$ makes entropy *dominate* and the agent will wander randomly forever. This coupling between reward scale and the right temperature is exactly why a fixed $\alpha$ that works on one task fails on another, and exactly why the automatic tuning that decouples the two was such a practical breakthrough. The auto-tuner targets a fixed amount of *entropy* regardless of reward scale, so it adapts $\alpha$ to whatever reward magnitude the environment happens to use.

| alpha regime | Policy behavior | Effect on learning |
| --- | --- | --- |
| alpha to 0 | Near-deterministic, greedy | Recovers standard RL, risks collapse |
| small alpha (~0.05) | Mostly exploit, slight randomness | Good late-training setting |
| moderate alpha (~0.2) | Balanced explore/exploit | Common starting point |
| large alpha (~1.0) | Highly stochastic | Strong exploration, slow exploitation |
| alpha to infinity | Uniform random | No useful learning |

## 4. The soft value functions and the soft Bellman backup

To turn the MaxEnt objective into something trainable we need *soft* versions of the value functions. The soft Q-function is defined so that its Bellman backup folds the entropy bonus into the bootstrap target. Starting from the MaxEnt objective, the soft state value is

$$V_{\text{soft}}(s) = \mathbb{E}_{a \sim \pi}\!\left[Q_{\text{soft}}(s, a) - \alpha \log \pi(a \mid s)\right],$$

which is just the expected Q-value plus the entropy contribution $-\alpha \log \pi$ at that state. The soft Q-function then satisfies the soft Bellman equation

$$Q_{\text{soft}}(s, a) = r(s, a) + \gamma\, \mathbb{E}_{s'}\!\left[V_{\text{soft}}(s')\right] = r(s,a) + \gamma\, \mathbb{E}_{s', a'}\!\left[Q_{\text{soft}}(s', a') - \alpha \log \pi(a' \mid s')\right].$$

Read the last expression carefully, because the $-\alpha \log \pi(a' \mid s')$ term is the crux of SAC and the part newcomers most often drop. The target for $Q(s,a)$ is the immediate reward, plus the discounted *next* Q-value, *minus* the temperature times the log-probability of the next action. Subtracting $\alpha \log \pi$ is the same as *adding* the next-state entropy bonus (since $-\log \pi$ is high when the policy is uncertain). If you forget that term, you collapse SAC back to ordinary off-policy actor-critic and lose the entropy objective entirely — the agent will train, but it will not be doing maximum-entropy RL, and it will lose the robustness that justified using SAC in the first place.

It is worth deriving the first equality — that $V_{\text{soft}}(s) = \mathbb{E}_a[Q_{\text{soft}}(s,a) - \alpha \log \pi(a \mid s)]$ — from the objective directly, so the entropy term's placement does not feel arbitrary. The soft value of a state is the expected discounted MaxEnt return starting from it. The first thing that happens at $s$ is: the policy samples an action $a$, which incurs an entropy contribution $\alpha \mathcal{H}(\pi(\cdot \mid s)) = \mathbb{E}_a[-\alpha \log \pi(a \mid s)]$, and the action's soft Q-value $Q_{\text{soft}}(s,a)$ already accounts for the reward and all *future* entropy. So the state value is the expected action value plus the entropy generated *at this very state*: $V_{\text{soft}}(s) = \mathbb{E}_a[Q_{\text{soft}}(s,a)] + \alpha \mathcal{H}(\pi(\cdot \mid s)) = \mathbb{E}_a[Q_{\text{soft}}(s,a) - \alpha \log \pi(a \mid s)]$. The convention here — which is the one SAC uses — is that $Q_{\text{soft}}(s,a)$ includes the entropy of all states *after* $s$ but *not* the entropy of the action choice at $s$ itself; $V_{\text{soft}}$ adds that last piece back. Getting this bookkeeping convention right is what makes the soft Bellman equation consistent, and it is why the bootstrap target subtracts $\alpha \log \pi(a' \mid s')$ at the *next* state but the current state's action entropy is handled by the actor loss rather than the critic target.

#### Worked example: a two-step soft return by hand

To make the soft value tangible, trace a deterministic two-step episode by hand. Start at $s_0$, take action $a_0$ with reward $r_0 = 2.0$ and $\log \pi(a_0 \mid s_0) = -1.0$, transition to $s_1$, take $a_1$ with reward $r_1 = 3.0$ and $\log \pi(a_1 \mid s_1) = -0.5$, then terminate. Use $\alpha = 0.2$ and $\gamma = 0.9$. The standard return is $r_0 + \gamma r_1 = 2.0 + 0.9 \times 3.0 = 4.7$. The soft return adds the discounted per-step entropy bonuses $-\alpha \log \pi$: at step 0 that is $-0.2 \times (-1.0) = +0.2$, at step 1 it is $-0.2 \times (-0.5) = +0.1$ discounted by $\gamma$. So the soft return is $(2.0 + 0.2) + 0.9 \times (3.0 + 0.1) = 2.2 + 0.9 \times 3.1 = 2.2 + 2.79 = 4.99$. The entropy bonus added 0.29 to the return purely for the policy having been uncertain along the way — and that 0.29 is exactly what the critic must learn to predict if it is estimating the *soft* Q-function rather than the ordinary one. An agent maximizing this soft return will, all else equal, prefer the trajectory that keeps more options open, which is the behavioral signature of MaxEnt RL.

The soft Bellman operator is a contraction in the sup norm for $\gamma < 1$, exactly like the ordinary Bellman operator, which is why repeated application converges to a unique fixed point $Q_{\text{soft}}^*$. Let me sketch *why* it is a contraction, because the argument is short and it is the load-bearing theorem under everything else. Define the soft Bellman operator $\mathcal{T}^\pi$ acting on a Q-function $Q$ by

$$(\mathcal{T}^\pi Q)(s, a) = r(s, a) + \gamma\, \mathbb{E}_{s', a'}\big[Q(s', a') - \alpha \log \pi(a' \mid s')\big].$$

Take two Q-functions $Q$ and $Q'$. Their images differ only in the bootstrap term, because the reward and the entropy term are identical:

$$\big|(\mathcal{T}^\pi Q)(s,a) - (\mathcal{T}^\pi Q')(s,a)\big| = \gamma\,\big|\mathbb{E}_{s', a'}[Q(s',a') - Q'(s',a')]\big| \le \gamma\, \|Q - Q'\|_\infty.$$

Taking the sup over $(s, a)$ on the left gives $\|\mathcal{T}^\pi Q - \mathcal{T}^\pi Q'\|_\infty \le \gamma\, \|Q - Q'\|_\infty$, which is exactly the definition of a $\gamma$-contraction. The entropy term, because it does not depend on $Q$ at all, simply cancels — it shifts the fixed point but never threatens the contraction. By the Banach fixed-point theorem, iterating $\mathcal{T}^\pi$ from any starting Q-function converges geometrically to the unique soft Q-function for $\pi$.

### The soft policy improvement theorem

That contraction is the theoretical guarantee underwriting *soft policy iteration*: alternately evaluate the soft Q-function for the current policy (soft policy evaluation, the contraction above) and improve the policy toward the Boltzmann form (soft policy improvement). The improvement half deserves its own statement, because it is what guarantees the alternation actually climbs.

**Claim.** Let $\pi_{\text{old}}$ be the current policy with soft Q-function $Q^{\pi_{\text{old}}}$, and define the improved policy by projecting the Boltzmann target onto the policy class:

$$\pi_{\text{new}}(\cdot \mid s) = \arg\min_{\pi'} \mathrm{KL}\!\left(\pi'(\cdot \mid s)\,\Big\|\, \frac{\exp(\tfrac{1}{\alpha} Q^{\pi_{\text{old}}}(s, \cdot))}{Z^{\pi_{\text{old}}}(s)}\right).$$

Then $Q^{\pi_{\text{new}}}(s,a) \ge Q^{\pi_{\text{old}}}(s,a)$ for all $(s,a)$, and consequently $J(\pi_{\text{new}}) \ge J(\pi_{\text{old}})$.

**Why it holds.** Because $\pi_{\text{new}}$ is the KL-minimizer against the Boltzmann distribution built from $Q^{\pi_{\text{old}}}$, it achieves at least as small a value of the objective $\mathbb{E}_{a \sim \pi'}[\alpha \log \pi'(a \mid s) - Q^{\pi_{\text{old}}}(s,a)]$ as $\pi_{\text{old}}$ does (since $\pi_{\text{old}}$ is one feasible choice of $\pi'$, the minimizer can only do better or equal). Rearranging that inequality gives, at every state,

$$\mathbb{E}_{a \sim \pi_{\text{new}}}\big[Q^{\pi_{\text{old}}}(s,a) - \alpha \log \pi_{\text{new}}(a \mid s)\big] \;\ge\; \mathbb{E}_{a \sim \pi_{\text{old}}}\big[Q^{\pi_{\text{old}}}(s,a) - \alpha \log \pi_{\text{old}}(a \mid s)\big] = V^{\pi_{\text{old}}}(s).$$

That is, under the new policy the one-step soft value at $s$ is at least the old soft value. Now expand the soft Bellman equation for $Q^{\pi_{\text{old}}}$ and repeatedly substitute this inequality into the bootstrap term, unrolling the recursion: at each step the next-state value can only rise, so the bound propagates forward through the whole trajectory. In the limit the telescoped inequality gives $Q^{\pi_{\text{old}}}(s,a) \le Q^{\pi_{\text{new}}}(s,a)$ for every $(s,a)$. Taking expectations over the start-state distribution turns the pointwise Q inequality into $J(\pi_{\text{new}}) \ge J(\pi_{\text{old}})$. This is the soft analogue of the classical policy improvement theorem, and it is the reason the SAC actor update — which is exactly a gradient step on that KL objective — is a *principled* improvement step rather than a heuristic.

Haarnoja et al. prove that iterating soft policy evaluation and soft policy improvement converges to the optimal MaxEnt policy in the tabular case. SAC is the deep-function-approximation version of that procedure, with neural networks standing in for the tabular Q and policy, and stochastic gradient descent standing in for the exact evaluation and improvement steps. The theory does not survive function approximation intact — no deep-RL convergence proof does — but it tells us the *target* the gradients are chasing is well-defined and unique, which is more than many algorithms can claim.

## 5. The SAC architecture: stochastic actor, twin critics, temperature

Figure 3 stacks the components. From the bottom: the environment supplies transitions; a replay buffer stores them (off-policy, so old data is reusable); twin critics estimate the soft Q-function; a stochastic actor proposes actions; and the auto-temperature module adjusts $\alpha$.

![Stacked diagram of SAC components from environment up through replay buffer, twin critics, stochastic actor, and automatic temperature](/imgs/blogs/soft-actor-critic-sac-3.png)

The **actor** is a squashed Gaussian. The network outputs a mean $\mu(s)$ and a log-standard-deviation $\log \sigma(s)$; we sample a pre-activation $u \sim \mathcal{N}(\mu, \sigma^2)$ and squash it through $\tanh$ so the action lands in $[-1, 1]$. The squashing is essential — robot joints and throttle commands are bounded — but it complicates the log-probability, as we will see in Section 6. Note the architectural choice that the actor outputs a *state-dependent* standard deviation, not a global one: the policy can be confident (small $\sigma$) in states it understands and uncertain (large $\sigma$) in states it does not. This is what "smart" exploration means in SAC — the randomness is allocated where it is useful rather than sprayed uniformly.

The **twin critics** $Q_{\theta_1}$ and $Q_{\theta_2}$ are two independent networks with identical architecture but different random initialization. Each takes the state-action pair and outputs a scalar soft Q-value. We also keep slowly-updated *target* copies $Q_{\bar\theta_1}, Q_{\bar\theta_2}$ for computing stable bootstrap targets. The two critics are *not* a single network with two heads — the independence of their initializations and their gradient noise is precisely what makes their errors decorrelated enough for the minimum operator to cancel bias, so sharing a trunk between them defeats the purpose.

The **temperature** $\alpha$ is a single learnable scalar (we actually learn $\log \alpha$ for positivity). It is updated by its own optimizer against the dual objective of Section 7. That is five neural networks plus one scalar in total: actor, two live critics, two target critics, and $\log\alpha$. The target *actor* that TD3 maintains is absent — SAC does not need it, because its stochastic actor already provides the action smoothing that TD3 gets from target-policy noise.

Figure 4 places SAC against its peers across the dimensions that matter for picking an algorithm.

![Comparison matrix of SAC, TD3, PPO, DQN, and DDPG across continuous actions, off-policy, auto-temperature, sample efficiency, stability, and HalfCheetah score](/imgs/blogs/soft-actor-critic-sac-4.png)

## 6. The actor update and the reparameterization trick

The actor wants to maximize expected soft value, which means minimizing

$$J_\pi(\phi) = \mathbb{E}_{s \sim \mathcal{D}}\,\mathbb{E}_{a \sim \pi_\phi}\!\left[\alpha \log \pi_\phi(a \mid s) - \min_{i=1,2} Q_{\theta_i}(s, a)\right].$$

The term $\alpha \log \pi$ pushes toward higher entropy (more negative $\log \pi$ lowers the loss), and $-Q$ pushes toward high-value actions. The subtlety is the inner expectation over $a \sim \pi_\phi$: the thing we are differentiating with respect to ($\phi$) also parameterizes the distribution we are sampling from. A naive score-function (REINFORCE) gradient here would be high-variance. SAC instead uses the **reparameterization trick**: write the action as a deterministic, differentiable function of the state and an *external* noise sample whose distribution does not depend on $\phi$.

$$a = \tanh\!\big(\mu_\phi(s) + \sigma_\phi(s) \odot \epsilon\big), \qquad \epsilon \sim \mathcal{N}(0, I).$$

Now the randomness is in $\epsilon$, which is fixed during the backward pass, so gradients flow cleanly through $\mu_\phi$ and $\sigma_\phi$. This is the same trick that powers variational autoencoders, and it is shown in Figure 5.

![Diagram of the reparameterization trick where external noise epsilon is combined with mean and standard deviation, passed through tanh, and used to compute the log-probability via change of variables](/imgs/blogs/soft-actor-critic-sac-5.png)

It is worth being precise about *why* the reparameterization estimator has lower variance than the score-function one, because it is the single design choice that most distinguishes SAC's actor update. The score-function (REINFORCE) estimator writes the gradient of $\mathbb{E}_{a \sim \pi_\phi}[f(a)]$ as $\mathbb{E}[f(a)\,\nabla_\phi \log \pi_\phi(a)]$. The factor $f(a)$ here is the whole soft value $\alpha \log \pi - Q$, which on HalfCheetah is a number in the hundreds; multiplying that large, high-variance scalar by the score $\nabla_\phi \log \pi_\phi$ produces a gradient estimate that swings wildly from sample to sample. The reparameterization estimator instead pushes the gradient *through* $f$: $\nabla_\phi \mathbb{E}_\epsilon[f(\tanh(\mu_\phi + \sigma_\phi \epsilon))] = \mathbb{E}_\epsilon[\nabla_\phi f(\cdots)]$. Now the gradient sees the *local slope* of the value surface around the sampled action, not the action's absolute value scaled by a score. Because the value surface is smooth, that slope is far more stable across noise samples. Empirically the variance reduction is large enough that SAC trains fine with a single noise sample per state per update, whereas a REINFORCE actor would need many samples or a carefully tuned baseline to be usable at all.

The $\tanh$ squashing means we cannot use the plain Gaussian log-density. The change-of-variables formula for an invertible transform $a = \tanh(u)$ gives

$$\log \pi(a \mid s) = \log \mu_{\mathcal{N}}(u \mid s) - \sum_{i} \log\!\big(1 - \tanh^2(u_i)\big),$$

where $\mu_{\mathcal{N}}$ is the Gaussian density of the pre-activation $u$ and the second term is the log-determinant of the $\tanh$ Jacobian, summed over action dimensions. The general change-of-variables rule for a random variable $a = g(u)$ is $p_a(a) = p_u(u)\,\big|\det \tfrac{dg}{du}\big|^{-1}$, and taking logs turns the inverse-determinant into the *subtracted* log-determinant above. For the elementwise $\tanh$, the Jacobian is diagonal with entries $\tfrac{d}{du_i}\tanh(u_i) = 1 - \tanh^2(u_i)$, so its log-determinant is the sum $\sum_i \log(1 - \tanh^2 u_i)$. This is the term that accounts for how $\tanh$ compresses the infinite real line of pre-activations into the finite box $[-1,1]$ of actions: near the boundaries $\tanh$ squashes hard, $1 - \tanh^2$ is near zero, its log is large and negative, and the correction *inflates* the action's log-density to reflect the compression. In code we add a small $\epsilon$ inside the log for numerical stability. Here is the actor:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

LOG_STD_MIN, LOG_STD_MAX = -20, 2

class SquashedGaussianActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.mu_head = nn.Linear(hidden, act_dim)
        self.log_std_head = nn.Linear(hidden, act_dim)

    def forward(self, obs):
        h = self.net(obs)
        mu = self.mu_head(h)
        log_std = self.log_std_head(h).clamp(LOG_STD_MIN, LOG_STD_MAX)
        std = log_std.exp()
        # reparameterized sample
        normal = torch.distributions.Normal(mu, std)
        u = normal.rsample()                  # mu + std * eps, differentiable
        a = torch.tanh(u)
        # change-of-variables log-prob for the tanh squashing
        log_prob = normal.log_prob(u).sum(dim=-1)
        log_prob -= (2 * (torch.log(torch.tensor(2.0)) - u - F.softplus(-2 * u))).sum(dim=-1)
        return a, log_prob
```

The compact `2*(log2 - u - softplus(-2u))` expression is a numerically stable rewrite of $\sum \log(1 - \tanh^2 u_i)$ and is worth copying verbatim — implementing the naive version overflows for large $|u|$. To see the identity, note $1 - \tanh^2 u = \operatorname{sech}^2 u = 4 / (e^u + e^{-u})^2$, so $\log(1 - \tanh^2 u) = \log 4 - 2\log(e^u + e^{-u}) = 2\log 2 - 2u - 2\log(1 + e^{-2u}) = 2(\log 2 - u - \operatorname{softplus}(-2u))$. The softplus form never exponentiates a large positive number, so it stays finite where the direct $\log(1 - \tanh^2 u)$ underflows to $\log 0 = -\infty$ once $|u|$ exceeds about 18 in float32. This is the single most common silent bug in hand-rolled SAC implementations — the loss looks fine for a while, then produces NaNs the moment the actor pushes an action toward the boundary.

A related detail: we clamp `log_std` to $[-20, 2]$ before exponentiating. The lower bound prevents the standard deviation from collapsing to zero (which would make $\log \pi \to +\infty$ and the entropy term explode), and the upper bound prevents an absurdly broad Gaussian early in training. These clamps are not cosmetic; without the lower clamp the policy can collapse to a deterministic spike and the whole maximum-entropy machinery silently switches off. One more subtlety for *evaluation*: at test time you usually want the deterministic action $a = \tanh(\mu_\phi(s))$ rather than a fresh sample — the entropy was a means to good exploration during learning, and once the policy is fixed you want its best guess, which is the mean. Sampling at evaluation is not wrong, but it adds avoidable variance to your reported returns.

#### Worked example: the gradient the actor actually receives

Walk through one actor gradient on a 2-dimensional action. Suppose at state $s$ the network outputs $\mu = (0.3, -0.1)$ and $\sigma = (0.5, 0.4)$, and we draw noise $\epsilon = (1.0, -0.5)$. Then the pre-activation is $u = \mu + \sigma \epsilon = (0.8, -0.3)$ and the action is $a = \tanh(u) = (0.664, -0.291)$. The Gaussian log-density of $u$ is $\sum_i [-\tfrac{1}{2}\log(2\pi\sigma_i^2) - \tfrac{(u_i - \mu_i)^2}{2\sigma_i^2}]$; the squashing correction subtracts $\sum_i \log(1 - \tanh^2 u_i) = \log(1 - 0.664^2) + \log(1 - 0.291^2) = \log 0.559 + \log 0.915 = -0.582 - 0.089 = -0.671$, so it *adds* 0.671 to the log-prob magnitude. Suppose $\log \pi(a \mid s) = -1.4$ after all terms, the critics give $\min(Q_1, Q_2) = 47.2$, and $\alpha = 0.2$. The per-sample actor loss is $\alpha \log \pi - Q = 0.2 \times (-1.4) - 47.2 = -47.48$. Because $u$ is a differentiable function of $\mu$ and $\sigma$ (the noise $\epsilon$ is frozen), backprop sends a gradient through $Q(s, \tanh(u))$ into both heads, nudging $\mu$ toward higher-Q actions and adjusting $\sigma$ to trade off the $\alpha \log \pi$ entropy term against value — all from this one reparameterized sample, with none of the variance a REINFORCE estimator would have injected.

## 7. The critic update and automatic temperature tuning

The critics regress toward the soft Bellman target. For a sampled transition $(s, a, r, s', d)$ with done flag $d$, the target uses a freshly sampled next action $a'$ from the *current* actor and the *target* critics:

$$y = r + \gamma (1 - d)\Big(\min_{i=1,2} Q_{\bar\theta_i}(s', a') - \alpha \log \pi_\phi(a' \mid s')\Big),$$

and each critic minimizes $\big(Q_{\theta_i}(s,a) - y\big)^2$. The two pieces that make this SAC rather than DDPG are the $\min$ over the twin target critics and the $-\alpha \log \pi$ entropy term. Notice three details that matter in practice. First, the next action $a'$ is sampled from the *current* actor, not from whatever policy generated the stored transition — this is what makes the update off-policy correct, because the soft Q-function is defined for the current policy regardless of how the data was collected. Second, the target uses the *target* critics $Q_{\bar\theta_i}$, not the live ones, for stability. Third, the $(1-d)$ factor zeroes the bootstrap on terminal transitions, where there is no next state to value — but be careful to only set $d=1$ for *true* environment terminations, not for time-limit truncations, because cutting the bootstrap on a truncation tells the critic the episode genuinely ended when in fact it would have continued.

### Why a single critic overestimates in MaxEnt RL

The **twin critics and the minimum trick** address overestimation, and the mechanism is worth spelling out because it is subtle. Suppose the true soft Q-value at some next action is $q$, but our learned critic produces a noisy estimate $\hat Q = q + \varepsilon$ with zero-mean error $\varepsilon$. The bootstrap target wants $\mathbb{E}[\hat Q] = q$, which a single unbiased critic delivers in expectation. The problem appears once the *policy* optimizes against that critic: the actor is trained to find actions where $\hat Q$ is large, and "where the noisy estimate is large" systematically coincides with "where the noise $\varepsilon$ happened to be positive." So the actions the policy actually visits are selected for upward noise, and the value the critic reports on them is biased high — a selection effect, not a modeling error. This optimistic bias then feeds back into the next bootstrap target, the critic learns the inflated value, the policy chases it harder, and the estimates can spiral. In MaxEnt RL the effect is if anything *worse* than in deterministic settings during early training, because the broad stochastic policy probes a wide swath of action space and is more likely to stumble onto the critic's noisy peaks.

The fix, borrowed from TD3's clipped double-Q learning, is to train two critics with independent initializations and use $\min(Q_1, Q_2)$ in the target. Because the two critics' errors are largely decorrelated, the action that fooled critic 1 with a positive noise spike usually does *not* fool critic 2, so taking the minimum cancels most of the upward selection bias. The minimum is a deliberately *pessimistic* estimator — it slightly underestimates on average — but a small consistent underestimate is enormously safer than an unbounded overestimate that the policy can exploit into divergence. It costs a second network and the compute to train it, but it buys dramatically more stable values, and the empirical evidence across TD3 and SAC is overwhelming that this trade is worth it.

### Deriving automatic temperature tuning

**Automatic temperature tuning** comes from posing MaxEnt RL as a constrained problem rather than an unconstrained one. Instead of picking $\alpha$ and hoping, we *demand* that the policy's average entropy stay at least at some target $\bar{\mathcal{H}}$, and we let the optimization figure out the temperature that enforces it:

$$\max_{\pi}\ \mathbb{E}_{\tau \sim \pi}\!\left[\sum_t r(s_t, a_t)\right] \quad \text{subject to} \quad \mathbb{E}_{(s,a) \sim \pi}\big[-\log \pi(a \mid s)\big] \ge \bar{\mathcal{H}}.$$

Form the Lagrangian by attaching a multiplier $\alpha \ge 0$ to the entropy constraint. By Lagrangian duality, at the optimum the multiplier $\alpha$ is exactly the temperature: it measures how much reward we would gain by relaxing the entropy floor by one nat — the shadow price of entropy. Holding the policy fixed and minimizing the dual over $\alpha$ gives the temperature objective

$$J(\alpha) = \mathbb{E}_{a \sim \pi}\big[-\alpha\,(\log \pi(a \mid s) + \bar{\mathcal{H}})\big].$$

Differentiating, the gradient is $\nabla_\alpha J = \mathbb{E}[-(\log \pi + \bar{\mathcal{H}})] = \mathbb{E}[-\log \pi] - \bar{\mathcal{H}}$, i.e. (current entropy − target entropy). So $\alpha$ rises when the policy's entropy is below target and falls when it is above — a clean feedback loop, drawn in Figure 8. In practice we optimize $\log \alpha$ rather than $\alpha$ directly, both to keep $\alpha$ positive without a projection and because gradient steps in log-space give multiplicative, well-scaled updates: $\alpha = \exp(\log\alpha)$, and the optimizer moves $\log\alpha$ freely on the whole real line. The target entropy is conventionally set to $\bar{\mathcal{H}} = -\dim(\mathcal{A})$, one negative nat per action dimension, which works remarkably well across tasks and is why SAC needs almost no per-environment tuning. The intuition for that heuristic: $-\dim(\mathcal{A})$ corresponds to a moderately concentrated Gaussian in each action dimension — enough randomness to keep exploring, not so much that the policy is flailing — and because it is specified in *entropy* units it automatically adapts $\alpha$ to whatever the reward scale turns out to be.

![Pipeline diagram of automatic temperature tuning that measures current entropy, compares to the target entropy, takes a dual gradient step, and updates alpha](/imgs/blogs/soft-actor-critic-sac-8.png)

Here are the twin critics and the full update step:

```python
class QCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs, act):
        return self.net(torch.cat([obs, act], dim=-1)).squeeze(-1)


def sac_update(batch, actor, q1, q2, q1_t, q2_t,
               log_alpha, target_entropy, optims, gamma=0.99, tau=0.005):
    s, a, r, s2, d = batch
    alpha = log_alpha.exp()

    # --- critic update ---
    with torch.no_grad():
        a2, logp2 = actor(s2)
        q_next = torch.min(q1_t(s2, a2), q2_t(s2, a2)) - alpha * logp2
        target = r + gamma * (1 - d) * q_next
    q1_loss = F.mse_loss(q1(s, a), target)
    q2_loss = F.mse_loss(q2(s, a), target)
    optims["q"].zero_grad(); (q1_loss + q2_loss).backward(); optims["q"].step()

    # --- actor update ---
    a_pi, logp = actor(s)
    q_pi = torch.min(q1(s, a_pi), q2(s, a_pi))
    actor_loss = (alpha.detach() * logp - q_pi).mean()
    optims["pi"].zero_grad(); actor_loss.backward(); optims["pi"].step()

    # --- temperature update ---
    alpha_loss = -(log_alpha * (logp.detach() + target_entropy)).mean()
    optims["alpha"].zero_grad(); alpha_loss.backward(); optims["alpha"].step()

    # --- soft target update (Polyak) ---
    with torch.no_grad():
        for net, net_t in [(q1, q1_t), (q2, q2_t)]:
            for p, p_t in zip(net.parameters(), net_t.parameters()):
                p_t.mul_(1 - tau).add_(tau * p)
    return q1_loss.item(), actor_loss.item(), alpha.item()
```

Three implementation details in that function are easy to get wrong. The critic target is computed under `torch.no_grad()` so gradients do not flow back through the bootstrap — we are regressing toward a fixed target, not differentiating through it. In the actor loss, `alpha.detach()` stops the actor's gradient from leaking into the temperature; the temperature has its own separate update. And the temperature loss uses `logp.detach()` for the same reason in reverse — the temperature update treats the policy's entropy as a fixed measurement, not something to backprop through. Mixing these gradients up is a classic bug that produces an agent that trains but never quite stabilizes.

## 8. Soft target updates and why SAC is stable off-policy

The target critics are not trained directly; they are an exponentially-moving average of the live critics, updated every step by Polyak averaging:

$$\bar\theta \leftarrow \tau\,\theta + (1 - \tau)\,\bar\theta, \qquad \tau = 0.005.$$

With $\tau = 0.005$, the targets move at roughly one two-hundredth of the speed of the live networks — slow enough that the bootstrap target $y$ is nearly stationary within any short window, which prevents the feedback loop where a network chases its own moving estimate and diverges. This is the same stabilizer DQN introduced as periodic hard target copies; the soft (continuous) version is smoother and is now standard. A useful mental model: the effective averaging horizon of an EMA with coefficient $\tau$ is about $1/\tau$ updates, so $\tau = 0.005$ means each target network is roughly a running average of the last 200 versions of its live counterpart. That lag is what gives the regression target enough inertia not to wobble.

Why is SAC stable *off-policy* when so many actor-critic methods are not? Three reasons compound. The replay buffer decorrelates samples and lets each transition contribute to many updates, so the critics see diverse data. The twin-critic minimum keeps value estimates from running away, which is the usual divergence mode. And the entropy term keeps the policy stochastic, so the buffer keeps containing varied actions rather than collapsing onto a narrow slice of the action space that the critic has never had to evaluate elsewhere. Off-policy divergence is usually a story of a critic extrapolating confidently into action regions it never saw; SAC's permanent exploration keeps refilling those regions with data.

This is also where SAC's contrast with DDPG and TD3 becomes a *robustness* story rather than just a performance one. DDPG must inject external exploration noise — typically an Ornstein-Uhlenbeck or Gaussian process whose scale you tune by hand. Too little and the policy collapses to a deterministic rut (premature determinism again); too much and the data is so noisy the critic cannot learn. The right amount differs per task and per training stage, so you babysit it. SAC eliminates the knob entirely: the policy's own state-dependent standard deviation *is* the exploration, and the auto-tuned temperature scales it to maintain a target entropy. TD3 sits in between — it keeps the deterministic actor and external noise but adds twin critics and a few stabilizers (delayed actor updates, target-policy smoothing). Empirically this is why you can take SAC's default hyperparameters to a brand-new continuous-control task and have a strong chance of it working on the first run, whereas DDPG demands per-task noise tuning and TD3 demands somewhat less. That hyperparameter robustness, more than peak score, is why SAC became the default.

#### Worked example: tracing one update's numbers

Take a single HalfCheetah transition midway through training. Suppose the live critics give $Q_1(s,a) = 305.2$ and $Q_2(s,a) = 298.7$. The actor samples a next action $a'$ with $\log \pi(a' \mid s') = -3.1$ (so its entropy contribution is $+3.1$ nats), and the target critics give $Q_{\bar 1}(s',a') = 312.0$, $Q_{\bar 2}(s',a') = 309.5$. With $\alpha = 0.18$, $\gamma = 0.99$, $r = 4.3$, and not-done:

$$y = 4.3 + 0.99\big(\min(312.0, 309.5) - 0.18 \times (-3.1)\big) = 4.3 + 0.99(309.5 + 0.558) = 311.26.$$

Critic 1's TD error is $311.26 - 305.2 = 6.06$, critic 2's is $311.26 - 298.7 = 12.56$, and each is squared into the MSE loss. Notice the entropy term *added* 0.558 to the bootstrap — small here, but if the policy were uncertain ($\log \pi = -8$) it would add $0.18 \times 8 \times 0.99 \approx 1.43$, actively rewarding the agent for reaching high-optionality states. For the temperature, if the batch-average $\log \pi$ is $-3.1$ and the target entropy for a 6-dimensional action space is $\bar{\mathcal{H}} = -6$, then current entropy $3.1 < 6$, so $\alpha$ is pushed *up* this step to encourage more randomness. Watch the target-critic minimum too: it picked 309.5 over 312.0, shaving 2.5 off the bootstrap — that small pessimism, applied every single update, is what keeps the value estimates from inflating over a million steps.

## 9. Putting it together: the full training loop

The replay buffer and the loop tie the pieces into a runnable agent. This is deliberately minimal but complete — it trains.

```python
import numpy as np
import gymnasium as gym

class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, size=1_000_000):
        self.s = np.zeros((size, obs_dim), np.float32)
        self.a = np.zeros((size, act_dim), np.float32)
        self.r = np.zeros(size, np.float32)
        self.s2 = np.zeros((size, obs_dim), np.float32)
        self.d = np.zeros(size, np.float32)
        self.ptr, self.full, self.size = 0, False, size

    def add(self, s, a, r, s2, d):
        i = self.ptr
        self.s[i], self.a[i], self.r[i], self.s2[i], self.d[i] = s, a, r, s2, d
        self.ptr = (i + 1) % self.size
        self.full = self.full or self.ptr == 0

    def sample(self, n=256):
        hi = self.size if self.full else self.ptr
        idx = np.random.randint(0, hi, size=n)
        to_t = lambda x: torch.as_tensor(x)
        return (to_t(self.s[idx]), to_t(self.a[idx]), to_t(self.r[idx]),
                to_t(self.s2[idx]), to_t(self.d[idx]))


def train_sac(env_id="HalfCheetah-v4", total_steps=1_000_000, start_steps=10_000):
    env = gym.make(env_id)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    actor = SquashedGaussianActor(obs_dim, act_dim)
    q1, q2 = QCritic(obs_dim, act_dim), QCritic(obs_dim, act_dim)
    q1_t, q2_t = QCritic(obs_dim, act_dim), QCritic(obs_dim, act_dim)
    q1_t.load_state_dict(q1.state_dict()); q2_t.load_state_dict(q2.state_dict())

    log_alpha = torch.zeros(1, requires_grad=True)
    target_entropy = -float(act_dim)
    optims = {
        "pi": torch.optim.Adam(actor.parameters(), lr=3e-4),
        "q": torch.optim.Adam(list(q1.parameters()) + list(q2.parameters()), lr=3e-4),
        "alpha": torch.optim.Adam([log_alpha], lr=3e-4),
    }
    buf = ReplayBuffer(obs_dim, act_dim)

    s, _ = env.reset(seed=0)
    for step in range(total_steps):
        if step < start_steps:
            a = env.action_space.sample()
        else:
            with torch.no_grad():
                a, _ = actor(torch.as_tensor(s, dtype=torch.float32).unsqueeze(0))
                a = a.squeeze(0).numpy()
        s2, r, term, trunc, _ = env.step(a)
        buf.add(s, a, r, s2, float(term))
        s = s2 if not (term or trunc) else env.reset()[0]
        if step >= start_steps:
            sac_update(buf.sample(256), actor, q1, q2, q1_t, q2_t,
                       log_alpha, target_entropy, optims)
    return actor
```

The `start_steps` warmup of pure random action collection fills the buffer with diverse data before any gradient step, a small detail that markedly improves early stability. Note there is no separate exploration noise schedule anywhere — the stochastic actor *is* the exploration. A few things to flag for anyone extending this skeleton toward a serious run. The done flag stored is `term` (true termination), not `term or trunc`, so time-limit truncations do not incorrectly zero the bootstrap — the subtle correctness point from Section 7. The update-to-data ratio here is one gradient step per environment step; cranking it higher (more gradient steps per collected transition) buys sample efficiency at a compute cost and is the lever that algorithms like REDQ push to the extreme. And for genuine reproducibility you would seed NumPy, PyTorch, and the action space, evaluate on a separate deterministic-action env periodically, and log the running $\alpha$, the entropy, and both critic losses — when SAC misbehaves, those three traces almost always reveal why.

#### What healthy training traces look like

Knowing what a *good* run looks like saves hours of debugging. Early in training, $\alpha$ typically *rises* from its initialization because the random-ish initial policy already has high entropy but the auto-tuner is finding the right scale; then as the policy starts exploiting, entropy naturally falls below target, $\alpha$ climbs to push back, and the two settle into a slow decay together as the task is mastered. The critic losses spike during the rapid-learning phase (the targets are moving fast) and then subside. The policy entropy should hover near the target $-\dim(\mathcal{A})$ once tuning equilibrates — if it pins to the `LOG_STD_MIN` clamp, your policy has collapsed and something (often a reward-scale mismatch or a learning rate that is too high) has broken the entropy machinery. If $\alpha$ runs away to a huge value, your target entropy is unreachable given the action bounds, usually a sign the actions are saturating against the $\tanh$ boundary.

## 10. The Stable-Baselines3 shortcut

For production you rarely hand-roll SAC; Stable-Baselines3 ships a battle-tested implementation. The from-scratch version above is for *understanding*; the production version is for *shipping*. Here is the full configuration with every parameter annotated:

```python
from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy
import gymnasium as gym

env = gym.make("HalfCheetah-v4")
model = SAC(
    "MlpPolicy",          # 2x256 ReLU nets for actor and critics, SAC's default
    env,
    learning_rate=3e-4,   # Adam LR shared by actor, critics, and temperature
    buffer_size=1_000_000,# replay capacity; bigger = more off-policy reuse, more RAM
    batch_size=256,       # transitions per gradient update
    tau=0.005,            # Polyak coefficient for soft target updates (~200-step EMA)
    gamma=0.99,           # discount; 0.99 = ~100-step effective horizon
    ent_coef="auto",      # automatic temperature tuning (the dual problem of Section 7)
    target_entropy="auto",# defaults to -dim(action_space)
    learning_starts=10_000,    # random warmup steps before gradient updates begin
    train_freq=1,         # one gradient step per environment step (the UTD ratio)
    gradient_steps=1,     # how many updates per train_freq trigger
    use_sde=False,        # state-dependent exploration; leave off for MuJoCo
    verbose=1,
)
model.learn(total_timesteps=1_000_000)

eval_env = gym.make("HalfCheetah-v4")
mean_r, std_r = evaluate_policy(model, eval_env, n_eval_episodes=10)
print(f"mean return {mean_r:.0f} +/- {std_r:.0f}")
```

The crucial flag is `ent_coef="auto"`, which switches on the dual temperature optimization from Section 7. Set it to a float like `0.2` to fix $\alpha$ if you ever need to ablate, or to `"auto_0.5"` to start the auto-tuner from an initial value of 0.5. The `target_entropy="auto"` resolves to $-\dim(\mathcal{A})$, matching our from-scratch heuristic. The `train_freq` and `gradient_steps` pair together set the update-to-data ratio; raising `gradient_steps` to 4 or 8 (with a matching `train_freq`) trades compute for sample efficiency and is the first knob to reach for on expensive real-robot runs. The defaults shown match the published SAC hyperparameters and reproduce the original results on the MuJoCo suite. You can launch a tuned run and log to TensorBoard with one shell command:

```bash
python -m rl_zoo3.train --algo sac --env HalfCheetah-v4 \
    --tensorboard-log ./tb --eval-freq 10000 --n-eval-episodes 10
```

For a real-robot or evaluation-heavy workflow, you typically wrap the training in callbacks that checkpoint the model and run periodic deterministic evaluations:

```python
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

eval_cb = EvalCallback(
    gym.make("HalfCheetah-v4"),
    best_model_save_path="./best",
    eval_freq=10_000,
    n_eval_episodes=10,
    deterministic=True,   # use tanh(mu), not a fresh sample, at eval time
)
ckpt_cb = CheckpointCallback(save_freq=50_000, save_path="./ckpts")
model.learn(total_timesteps=1_000_000, callback=[eval_cb, ckpt_cb])
```

The `deterministic=True` in the evaluation callback is the production echo of the evaluation note from Section 6: report the policy's mean action, not a sample, so your benchmark numbers are not inflated with avoidable variance.

## 11. SAC variants worth knowing

SAC has spawned a family of refinements, each targeting a specific limitation. Knowing the map helps you pick the right one when vanilla SAC is not quite enough.

**SAC-Discrete (Christodoulou, 2019)** adapts SAC to discrete action spaces. The squashed Gaussian is replaced by a categorical policy that outputs a probability for each of the $n$ discrete actions, and crucially the entropy and the soft value can now be computed *exactly* by summing over all actions rather than estimating them from a single sample. The actor loss becomes $\mathbb{E}_s[\sum_a \pi(a \mid s)(\alpha \log \pi(a \mid s) - \min_i Q_i(s,a))]$ — a full expectation, no reparameterization needed because there is nothing to differentiate through. The target entropy heuristic changes too, typically to a fraction (around 0.98) of the maximum entropy $\log n$. This is the version to use for Atari-style discrete control when you want SAC's entropy-driven stability instead of DQN's epsilon-greedy.

```python
# SAC-Discrete actor loss: full sum over the n actions, no sampling
def discrete_actor_loss(s, actor, q1, q2, alpha):
    logits = actor(s)                       # (batch, n_actions)
    probs = torch.softmax(logits, dim=-1)
    log_probs = torch.log_softmax(logits, dim=-1)
    with torch.no_grad():
        q = torch.min(q1(s), q2(s))         # (batch, n_actions), Q per action
    # expectation under the categorical policy, summed over actions
    loss = (probs * (alpha * log_probs - q)).sum(dim=-1).mean()
    entropy = -(probs * log_probs).sum(dim=-1).mean()
    return loss, entropy
```

**REDQ (Randomized Ensembled Double Q-learning, Chen et al., 2021)** pushes the twin-critic idea to an ensemble of $N$ critics (often $N=10$) and, at each target computation, takes the minimum over a *random subset* of $M$ of them (often $M=2$). Combined with a very high update-to-data ratio — 20 gradient steps per environment step — REDQ matches the sample efficiency of model-based methods while staying purely model-free. The large ensemble plus random subsampling controls the overestimation bias more finely than a fixed pair of critics, which is what lets the aggressive UTD ratio remain stable. The cost is obvious: ten critics and twenty updates per step is a lot of compute, so REDQ shines exactly when *environment* steps are the bottleneck and compute is cheap.

**SAC-N and ensemble variants** generalize the same lever for offline RL: a larger critic ensemble produces a more reliable pessimistic value estimate, which is precisely what offline RL needs to avoid over-valuing out-of-distribution actions. EDAC and similar methods add a diversity penalty so the ensemble members disagree usefully rather than collapsing to the same function.

| Variant | Key change | Best for |
| --- | --- | --- |
| SAC (vanilla) | Twin critics, squashed Gaussian, auto-temp | Continuous control, default choice |
| SAC-Discrete | Categorical policy, exact entropy sum | Discrete actions with entropy stability |
| REDQ | N-critic ensemble, random min-of-M, high UTD | Maximizing sample efficiency, cheap compute |
| SAC-N / EDAC | Large critic ensemble, diversity penalty | Offline RL, OOD-action pessimism |

## 12. Case studies

**HalfCheetah-v4 (Haarnoja et al. 2018).** This is the canonical SAC benchmark. SAC reaches roughly 12,000 average return within about one million environment steps and continues climbing modestly thereafter. Figure 6 sketches the four phases I see on essentially every run: the random warmup sits near zero return; once learning begins the agent finds a forward-leaning gait and climbs to around 2,000; a rapid-gain phase carries it to roughly 8,000; and it converges near 12,000. The curve is notably smooth compared to DDPG, whose HalfCheetah runs are infamous for sudden collapses.

![Timeline of SAC learning on HalfCheetah moving through random, exploring, rapid-gain, and converged phases reaching about 12000 return](/imgs/blogs/soft-actor-critic-sac-6.png)

To put numbers on those phases: with the warmup at 10,000 steps and a buffer that fills steadily, you typically see the return cross 2,000 somewhere around 100,000 steps, blow through 8,000 by roughly 400,000 steps, and asymptote near 12,000 by 800,000 to a million steps. Meanwhile the temperature $\alpha$, starting from 1.0 (since `log_alpha` initializes at zero), drifts down into the 0.1–0.2 range as the policy sharpens, and the policy entropy settles near the target of $-6$ (HalfCheetah's action space is 6-dimensional). If you watch the two critic losses, they peak during the 100k–400k rapid-gain window and then decline as the value estimates stabilize. These are the traces I check first when a run looks off.

**SAC vs TD3 vs PPO across MuJoCo.** On the standard four-task comparison (HalfCheetah, Hopper, Walker2d, Ant), the pattern from the literature is consistent. SAC and TD3 are both far more sample-efficient than PPO — they reach competitive returns in roughly a quarter to a third of the environment steps, because they reuse data off-policy. Between SAC and TD3, SAC is typically more robust to hyperparameters and seeds (TD3 needs its exploration noise tuned), while TD3 occasionally edges out a slightly higher final score on a well-tuned task. PPO's strength is wall-clock throughput in massively parallel simulators and its forgiving stability, not sample efficiency.

| Task | SAC (1M steps) | TD3 (1M steps) | PPO (1M steps) |
| --- | --- | --- | --- |
| HalfCheetah-v4 | ~12,000 | ~10,000 | ~3,000 |
| Hopper-v4 | ~3,300 | ~3,400 | ~2,400 |
| Walker2d-v4 | ~4,900 | ~4,700 | ~3,500 |
| Ant-v4 | ~5,800 | ~5,400 | ~2,200 |

These are approximate figures consistent with the SB3 benchmark tables and the TD3 and SAC papers; exact numbers vary with seeds and MuJoCo version, so treat them as orders of magnitude, not gospel. The headline reading is that on the harder, higher-dimensional tasks (HalfCheetah, Ant) SAC's entropy-driven exploration opens up a clear lead, while on the lower-dimensional, contact-sensitive tasks (Hopper) SAC and TD3 are nearly tied. PPO trails on sample efficiency throughout — but remember the figures are *at one million steps*; given a cheap simulator and ten million steps, PPO closes much of the gap on wall-clock terms.

The full four-dimensional comparison — objective, auto-temperature, sample efficiency, and stability — is worth tabulating because it is the decision matrix I actually use:

| | SAC | TD3 | PPO | DDPG |
| --- | --- | --- | --- | --- |
| Objective | reward + entropy | reward only | clipped surrogate | reward only |
| Auto-temperature | yes | n/a | n/a | n/a |
| Sample efficiency | high | high | low | high |
| HalfCheetah (~1M) | ~12,000 | ~10,000 | ~3,000 | ~7,000 (unstable) |
| Stability | very high | high | high | low |

**SAC vs TD3 on six MuJoCo tasks (Haarnoja 2018 Table 1).** The "Soft Actor-Critic Algorithms and Applications" follow-up reported a six-task sweep where SAC with automatic temperature tuning matched or beat the strongest baseline on every task, with the largest margins on the high-dimensional Humanoid (where a 21-dimensional action space makes good exploration decisive — SAC reached competitive Humanoid performance where DDPG essentially failed to learn). The headline takeaway from that table is not that SAC always wins by the most on every task, but that it wins *reliably* across the suite with a single set of hyperparameters and no per-task temperature tuning — which is the whole practical point.

**Robotic manipulation and locomotion on hardware.** Haarnoja et al. trained a real Minitaur quadruped to walk directly in the physical world with SAC in about two hours of real-time interaction — a striking result precisely because off-policy sample efficiency makes real-robot training feasible at all. The entropy-maximizing policy's robustness was the reported reason it transferred across terrain it had not seen during training: the same policy, trained on flat ground, walked over wooden blocks and up a slope without retraining. They also trained a real robotic hand to manipulate objects and a real-world valve-turning task, in each case leaning on the same two properties — sample efficiency to make physical training tractable, and entropy-driven robustness to survive the gap between the controlled training conditions and test-time perturbations.

## 13. Why SAC dominates continuous control

Pulling the threads together, SAC's dominance rests on three reinforcing properties. **Sample efficiency** comes from being off-policy: the million-transition replay buffer means every interaction is reused dozens of times, so SAC needs far fewer environment steps than on-policy PPO — decisive when steps are expensive. **Robustness to hyperparameters** comes from automatic temperature tuning and the twin-critic minimum: the two knobs that historically broke continuous-control agents (exploration amount and value overestimation) are now handled by the algorithm rather than by you. **Exploration without epsilon-greedy** comes from the entropy objective: the policy explores by being genuinely stochastic in a state-dependent, learned way, which is far smarter than injecting uniform noise — it is uncertain where it should be and confident where it has learned.

That said, SAC is not free. It maintains five networks (actor, two critics, two targets) plus a temperature, so it does more compute per environment step than PPO. In a cheap, massively-parallelizable simulator where you can throw billions of steps at the problem, PPO's throughput can win on wall-clock time even though it is less sample-efficient. SAC's edge is sharpest exactly where steps are precious. There are also tasks where SAC's continuous Gaussian assumption is a poor fit — highly multimodal optimal policies, where the right action distribution has several distinct peaks, are something a unimodal squashed Gaussian cannot represent, and you would reach for a normalizing-flow or mixture policy instead. But for the broad center of continuous-control problems, SAC is the default for the same reason TCP is the default transport protocol: not because it is optimal for every case, but because it is robustly good for almost all of them with almost no tuning.

## 14. SAC for real robots and sim-to-real

The entropy objective has a special significance for transferring policies from simulation to physical hardware. A reward-only policy tends to overfit to the exact dynamics of the simulator — it finds a razor-thin gait that exploits the friction coefficient the simulator happened to use, and that gait shatters when the real robot's friction differs by ten percent. A MaxEnt policy, by keeping probability mass spread across a *family* of near-optimal actions, behaves more like a controller that has learned several redundant ways to accomplish the task. When the real dynamics perturb it off its preferred trajectory, it has a fallback already in its support, so it recovers instead of collapsing.

This connects SAC to the broader sim-to-real toolkit. It composes naturally with domain randomization — randomize the simulator's physics parameters during training and the entropy bonus encourages a policy robust across that whole distribution. The two techniques attack the sim-to-real gap from complementary angles: domain randomization widens the *distribution of dynamics* the policy is trained against, while the entropy objective widens the *distribution of actions* the policy keeps ready at each state. Together they produce policies that are robust both to which world they wake up in and to where that world pushes them. In soft-robotics and contact-rich manipulation, where dynamics are hard to model and contacts are discontinuous, the stochasticity also helps the agent avoid getting stuck against a poorly-modeled contact surface — a deterministic policy that presses uselessly against a wall the simulator did not predict will press forever, whereas a stochastic one samples its way out. Figure 7 gives the decision tree I actually use to decide whether SAC is the right tool.

![Decision tree for choosing SAC when actions are continuous, off-policy learning is acceptable, and sample efficiency matters, routing other cases to PPO or DQN](/imgs/blogs/soft-actor-critic-sac-7.png)

A practical sim-to-real recipe that leans on SAC: train with domain randomization over the uncertain physics parameters (mass, friction, motor delays), keep the auto-tuned temperature on so the policy maintains genuine action entropy throughout, evaluate the *deterministic* mean policy on the real hardware first (it is the safest single behavior), and if it underperforms, allow a small amount of stochasticity at deployment to let the policy's learned redundancy kick in. The Minitaur and valve-turning results from Haarnoja et al. followed essentially this shape, and the entropy term was, by their account, the load-bearing ingredient in the transfer.

## When to use this (and when not to)

Use SAC when your action space is **continuous** (real-valued vectors: joint torques, steering, portfolio weights), when **off-policy reuse is acceptable** (the environment is roughly stationary so old transitions remain informative), and when **sample efficiency matters** (real hardware, expensive simulation, or limited interaction budget). In that intersection — which covers most robotics and a lot of control — SAC is the default and is hard to beat.

Do *not* reach for SAC when your actions are **discrete** — use DQN or its variants, or SAC-Discrete if you specifically want the entropy-driven stability, but the vanilla squashed Gaussian does not apply where the `argmax` over a finite action set is trivial. Do not use it when you need **on-policy guarantees** or are training at massive parallel scale in a cheap simulator — PPO's throughput and stability often win there; see the [PPO deep-dive](/blog/machine-learning/reinforcement-learning/proximal-policy-optimization-ppo) for that regime. And if your environment is **highly non-stationary** (the dynamics shift faster than the buffer turns over), the off-policy buffer becomes a liability because it stores stale dynamics, and an on-policy method may be safer. If the optimal policy is genuinely **multimodal** — several distinct action peaks are equally good — the unimodal Gaussian will struggle and a flow or mixture policy is the better fit. For the full taxonomy of these choices, the [unified map](/blog/machine-learning/reinforcement-learning/reinforcement-learning-a-unified-map) lays them out, and the [capstone playbook](/blog/machine-learning/reinforcement-learning/the-reinforcement-learning-playbook) collects the decision rules.

## Key takeaways

- Maximum-entropy RL maximizes reward **plus** policy entropy, $J(\pi) = \mathbb{E}[\sum \gamma^t (r + \alpha \mathcal{H})]$; this changes the objective, not just exploration, and yields robust stochastic policies whose optimum is a Boltzmann distribution over soft Q-values.
- The temperature $\alpha$ is the reward-versus-entropy exchange rate measured in the *same units as reward*; this coupling to reward scale is exactly why a fixed $\alpha$ does not transfer between tasks and why auto-tuning matters.
- The soft Bellman target subtracts $\alpha \log \pi(a' \mid s')$ — equivalently adds next-state entropy. Dropping this term silently turns SAC into ordinary actor-critic. The soft Bellman operator is a $\gamma$-contraction, and soft policy improvement provably does not decrease the soft value.
- The squashed-Gaussian actor uses the reparameterization trick $a = \tanh(\mu + \sigma \epsilon)$ for low-variance gradients, with a numerically stable $\tanh$ change-of-variables correction (`2*(log2 - u - softplus(-2u))`) in the log-probability.
- Twin critics with $\min(Q_1, Q_2)$ suppress the overestimation bias — a *selection effect* where the policy exploits whichever critic noise spiked high — that destabilizes single-critic methods like DDPG.
- Automatic temperature tuning solves a dual problem: $\log\alpha$ is a Lagrange multiplier that rises when entropy is below the target $\bar{\mathcal{H}} = -\dim(\mathcal{A})$ and falls when above, removing the hardest hyperparameter.
- Soft target updates with $\tau = 0.005$ (a ~200-step EMA) keep bootstrap targets nearly stationary, which together with the replay buffer and permanent stochastic exploration is the backbone of SAC's off-policy stability.
- SAC reaches ~12,000 return on HalfCheetah and trains real robots in hours; variants extend it to discrete actions (SAC-Discrete), extreme sample efficiency (REDQ), and offline RL (SAC-N). It dominates continuous control when steps are expensive, but PPO can win on wall-clock time in cheap parallel simulators.

## Further reading

- Haarnoja, Zhou, Abbeel, Levine, "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor" (2018) — the original algorithm.
- Haarnoja et al., "Soft Actor-Critic Algorithms and Applications" (2018) — adds automatic temperature tuning and the Minitaur hardware results.
- Fujimoto, van Hoof, Meger, "Addressing Function Approximation Error in Actor-Critic Methods" (2018) — the TD3 paper introducing clipped double-Q, which SAC borrows.
- Christodoulou, "Soft Actor-Critic for Discrete Action Settings" (2019) — the SAC-Discrete adaptation.
- Chen, Wang, Zhou, Ross, "Randomized Ensembled Double Q-Learning: Learning Fast Without a Model" (2021) — REDQ and high update-to-data ratios.
- Levine, "Reinforcement Learning and Control as Probabilistic Inference" (2018) — the RL-as-inference framing that grounds the MaxEnt objective.
- Ziebart, "Modeling Purposeful Adaptive Behavior with the Principle of Maximum Causal Entropy" (2010) — the maximum-entropy RL foundations.
- Sutton & Barto, "Reinforcement Learning: An Introduction" (2nd ed.) — for the underlying value-function and policy-iteration machinery.
- Stable-Baselines3 SAC documentation and the RL Baselines3 Zoo for tuned hyperparameters and reproducible benchmarks.
- Within this series: the [unified map of RL algorithms](/blog/machine-learning/reinforcement-learning/reinforcement-learning-a-unified-map), the [PPO deep-dive](/blog/machine-learning/reinforcement-learning/proximal-policy-optimization-ppo), and the [capstone playbook](/blog/machine-learning/reinforcement-learning/the-reinforcement-learning-playbook).
