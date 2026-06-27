---
title: "Safe RL: Constrained Optimization for Real-World Deployment"
date: "2026-06-22"
publishDate: "2026-06-22"
description: "How to train RL agents that respect hard constraints during deployment — from the Constrained MDP formalism to CPO, PPO-Lagrangian, and the safety-layer and control-barrier-function approaches used in robotics and autonomous vehicles."
tags:
  [
    "reinforcement-learning",
    "deep-learning",
    "safe-rl",
    "constrained-optimization",
    "robotics",
    "machine-learning",
    "pytorch",
    "llm-alignment",
    "exploration",
  ]
category: "machine-learning"
subcategory: "Reinforcement Learning"
author: "Hiep Tran"
featured: true
readTime: 56
image: "/imgs/blogs/safe-rl-constrained-optimization-1.png"
---

The first robot arm I helped deploy destroyed itself in under a second. We had trained a beautiful policy in simulation: it reached the target, it was sample-efficient, the reward curve was a textbook S-shape that flattened out near the optimum. Then we loaded the weights onto the real arm, hit run, and the third joint slammed to its mechanical stop hard enough to strip a gear. The simulator had no joint torque limits, so the policy had quietly learned that the fastest way to the goal was to command 95 newton-metres through a joint rated for 40. Nothing in the reward function said "don't break yourself." Reward maximization is indifferent to everything you forgot to put in the reward.

This is the central, uncomfortable truth of deploying reinforcement learning into the physical and financial world: an RL agent is a relentless optimizer of exactly the objective you wrote down, and the objective you wrote down is almost never the objective you actually wanted. A self-driving policy that optimizes average speed will, given the chance, shave the margin around a pedestrian to zero, because the margin costs time and time costs reward. A trading agent that maximizes Sharpe ratio will discover that leverage multiplies returns, and it will keep adding leverage right up until the one tail event that liquidates the book — which, in a backtest with no margin-call model, never happens. The common thread is not that these agents are malicious. They are doing precisely what they were told. The failure is that "maximize this number" is the wrong contract for any system that touches the real world, where some outcomes are not merely low-reward but categorically unacceptable.

![A diagram of the constrained Markov decision process primal-dual loop showing how a reward objective and a cost constraint combine through a Lagrangian, with primal and dual updates feeding back until the cost budget binds](/imgs/blogs/safe-rl-constrained-optimization-1.png)

Safe RL is the body of theory and engineering that fixes this contract. Instead of "maximize reward," the contract becomes "maximize reward subject to never violating these constraints." That single change — from an unconstrained objective to a constrained one — is mathematically the difference between an optimization problem and a *constrained* optimization problem, and it brings with it a whole machinery: Lagrange multipliers, trust regions, projection operators, barrier functions, and risk measures. By the end of this post you will be able to write down the Constrained MDP formalism, derive why the naive Lagrangian oscillates and how to damp it, implement PPO-Lagrangian and a quadratic-program safety filter in PyTorch, read a Safety Gym results table critically, and pick the right method for whether your constraint is a soft budget or a hard runtime guarantee. Throughout, keep the series spine in mind: the RL loop is an agent interacting with an environment, collecting rewards, and updating a policy. Safe RL changes only what we optimize and how we estimate its gradient — the loop itself is unchanged.

## 1. Why "maximize reward" is the wrong contract

Let me make the failure concrete before we formalize it, because the formalism only earns its keep once you feel the problem in your hands.

Recall the standard reinforcement-learning setup, which we covered in [Markov decision processes](/blog/machine-learning/reinforcement-learning/markov-decision-processes). An agent in state $s$ takes action $a$, the environment returns a reward $r(s,a)$ and a next state $s'$, and the agent's goal is to learn a policy $\pi(a \mid s)$ that maximizes the expected discounted return

$$
J(\pi) = \mathbb{E}_{\tau \sim \pi}\left[\sum_{t=0}^{\infty} \gamma^t \, r(s_t, a_t)\right],
$$

where $\tau = (s_0, a_0, s_1, a_1, \dots)$ is a trajectory and $\gamma \in [0,1)$ is the discount factor. Every algorithm in this series — Q-learning, policy gradients, PPO, GRPO — is a different way to push $\pi$ up the gradient of $J$.

The problem is what is *not* in $J$. Suppose you are building the robot arm. You want it to reach a target quickly, so you set $r = -\,\|s - s_{\text{goal}}\|$, a small penalty proportional to distance. The optimizer happily learns to move fast. But "move fast" and "stay under 40 newton-metres of joint torque" are different sentences. If torque does not appear in the reward, the optimizer treats it as free. You could try to fold torque into the reward as a penalty, $r = -\,\|s - s_{\text{goal}}\| - \beta \cdot \tau^2$, and people do this all the time. It is called reward shaping, and we discussed its hazards in [reward hacking and Goodhart's law](/blog/machine-learning/reinforcement-learning/reward-hacking-and-goodharts-law). The trouble is that $\beta$ is now a *price* you are setting on torque, and you have to guess it. Set $\beta$ too low and the agent pays the price gladly and still breaks the gear. Set it too high and the agent becomes so timid it never reaches the target. There is no single $\beta$ that means "torque up to 40 is free, torque above 40 is forbidden." A scalar penalty cannot encode a hard threshold; it can only encode a slope.

This is the philosophical heart of safe RL. A penalty asks "how much is a violation worth?" A constraint asks "is this violation allowed at all?" Those are categorically different questions, and the second one is the one that matters when the downside is a stripped gear, a collision, or a margin call. The autonomous-vehicle team frames it as the distinction between *comfort* objectives (smooth acceleration, good lane-keeping — things you trade off) and *safety* objectives (never enter the pedestrian's space — things you do not trade off). A well-posed deployment keeps these in separate mathematical boxes.

#### Worked example: the leverage trap

Consider a portfolio agent, the kind we built in [RL portfolio optimization](/blog/machine-learning/reinforcement-learning/rl-portfolio-optimization). Its reward is the daily log-return of the book, and it can choose a leverage multiplier $\ell \in [1, 5]$. In a 5-year backtest on a calm market, returns scale almost linearly with leverage: the agent learns $\ell \approx 4.8$ and posts an annualized return of 41 percent versus 9 percent for the unlevered book. The Sharpe ratio looks superb at 2.1. Then you replay the same agent through March 2020. On a single day the book draws down 38 percent, trips the broker's maintenance-margin requirement, and is force-liquidated at the worst possible prices. The realized return over the full out-of-sample window is negative. The agent never saw a margin call in training because the training environment never modeled one — so leverage was, to the optimizer, simply free Sharpe. A constraint of the form "probability of a daily drawdown exceeding 20 percent must stay below 1 percent" would have capped leverage near $\ell = 2.3$ and turned a blow-up into a survivable strategy. We will see exactly this constraint formalized as a CVaR constraint in Section 7.

So the engineering goal crystallizes: we want to keep the full power of the RL machinery — the same trajectory sampling, the same policy gradient — but solve a *constrained* problem. The rest of this post is the toolbox for doing that.

## 2. Constrained MDPs: the formalism

The right object is the **Constrained Markov Decision Process**, or CMDP, formalized by Eitan Altman in his 1999 monograph. A CMDP is an ordinary MDP with one addition: alongside the reward $r(s,a)$, there is one or more **cost functions** $c_i(s,a)$, and a budget $d_i$ for each. The cost is just another scalar the environment emits at every step — it has the exact same type as a reward — but we treat it differently. We do not maximize it; we constrain its expected discounted sum.

Formally, define the expected discounted cost of policy $\pi$ for constraint $i$ as

$$
C_i(\pi) = \mathbb{E}_{\tau \sim \pi}\left[\sum_{t=0}^{\infty} \gamma_c^t \, c_i(s_t, a_t)\right],
$$

where $\gamma_c$ is a cost discount (often set to 1 for episodic safety, or to $\gamma$ for convenience). The CMDP problem is then

$$
\max_{\pi} \; J(\pi) \quad \text{subject to} \quad C_i(\pi) \le d_i \;\; \text{for all } i.
$$

The set of policies satisfying every constraint is the **feasible set**. Our job is to find the highest-reward policy *inside* that set. For the robot, $c(s,a)$ might be 1 whenever the commanded torque exceeds 40 newton-metres and 0 otherwise, with $d$ a small budget like 25 (allowing a tolerable number of brief excursions over an episode), or $d = 0$ for a strict no-violation policy. For the autonomous vehicle, $c$ might be the count of times the car enters a 1-metre buffer around a pedestrian. The cost is a measurable, environment-emitted signal — that is what makes a CMDP tractable, as opposed to a fuzzy "be safe" wish.

### The Lagrangian relaxation

A constrained maximization is harder than an unconstrained one because the optimizer cannot just follow the reward gradient — it might walk straight out of the feasible set. The classical tool for converting a constrained problem into a sequence of unconstrained ones is the **Lagrangian**. We introduce a non-negative multiplier $\lambda_i \ge 0$ for each constraint and define

$$
\mathcal{L}(\pi, \lambda) = J(\pi) - \sum_i \lambda_i \big(C_i(\pi) - d_i\big).
$$

Look at what this does for a single constraint. If the constraint is satisfied with room to spare ($C(\pi) < d$), the term $C(\pi) - d$ is negative, and since $\lambda \ge 0$, subtracting it *adds* to the objective — but at the optimum the multiplier for a slack constraint is driven to zero, so it contributes nothing. If the constraint is violated ($C(\pi) > d$), the term is positive, and subtracting $\lambda (C - d)$ *penalizes* the objective in proportion to both the violation size and the multiplier. The multiplier $\lambda$ is, quite literally, the *price* of cost — but unlike the hand-set $\beta$ from Section 1, this price is not guessed. It is discovered.

The connection to the original problem is the **primal-dual** structure. The original constrained problem equals

$$
\max_{\pi} \min_{\lambda \ge 0} \mathcal{L}(\pi, \lambda).
$$

To see why, fix a policy $\pi$. The inner minimization over $\lambda \ge 0$ does one of two things. If $\pi$ is feasible ($C - d \le 0$), then to minimize $-\lambda(C-d)$ — which is $\ge 0$ — the adversary picks $\lambda = 0$, and the value is just $J(\pi)$. If $\pi$ is infeasible ($C - d > 0$), the adversary can drive $\lambda \to \infty$, sending $\mathcal{L} \to -\infty$. So the inner min returns $J(\pi)$ for feasible policies and $-\infty$ for infeasible ones. Maximizing that over $\pi$ therefore picks the best feasible policy — exactly the CMDP objective. This min-max view is what the diagram above renders as a loop: the reward objective and the cost constraint feed a Lagrangian, the policy ascends it for fixed $\lambda$ (the primal step), the multiplier rises when the constraint is violated (the dual step), and the system converges when the cost equals its budget.

Under convexity, strong duality lets us swap the order and solve $\min_{\lambda \ge 0} \max_{\pi} \mathcal{L}$, which is the form we actually optimize: alternate between a **primal step** (improve $\pi$ to ascend $\mathcal{L}$ with $\lambda$ fixed — an ordinary RL update on the modified reward $r - \lambda c$) and a **dual step** (adjust $\lambda$ to enforce the constraint). The dual gradient is beautifully simple:

$$
\frac{\partial \mathcal{L}}{\partial \lambda_i} = -\big(C_i(\pi) - d_i\big),
$$

so we *ascend* $\lambda$ when the constraint is violated and *descend* it (clipping at zero) when it is satisfied:

$$
\lambda_i \leftarrow \big[\lambda_i + \eta_\lambda \,(C_i(\pi) - d_i)\big]_+.
$$

This is the whole skeleton of Lagrangian safe RL. Almost every practical method — PPO-Lagrangian, TRPO-Lagrangian, SAC-Lagrangian — is this skeleton wrapped around a different base RL algorithm.

### Why the naive Lagrangian oscillates

Here is the catch, and it is the kind of thing that costs you a week if you do not understand it. The primal and dual updates run on different timescales and can fight each other into a limit cycle. Picture the dynamics. The policy is currently violating the constraint, so $\lambda$ climbs. As $\lambda$ climbs, the effective reward $r - \lambda c$ punishes cost more, so the policy becomes very conservative and overshoots into deep feasibility ($C \ll d$). Now the constraint is slack, so the dual step *lowers* $\lambda$. As $\lambda$ falls, cost becomes cheap again, the policy chases reward, and the cost shoots back over budget. The result is an oscillation: $\lambda$ sawtooths, the cost orbits the budget instead of settling on it, and reward bounces around with it. This is the discrete-time analogue of an undamped predator-prey system, and it is *the* signature pathology of naive Lagrangian safe RL.

There are three standard fixes, in increasing order of robustness. First, **slow the dual learning rate**: set $\eta_\lambda$ an order of magnitude below the policy learning rate so the multiplier moves on a slow timescale relative to the policy, approximating the two-timescale stochastic-approximation conditions under which the primal-dual scheme provably converges. Second, **use a PID controller on the constraint** instead of plain gradient ascent — the proportional term reacts to the current violation, the integral term is the classic $\lambda$ accumulation, and a *derivative* term damps the oscillation by reacting to the *rate of change* of the violation. Stooke, Achiam, and Abbeel showed in 2020 that a PID-controlled multiplier all but eliminates the sawtooth. Third, the **augmented Lagrangian** method.

### The augmented Lagrangian

The augmented Lagrangian adds a quadratic penalty term to the ordinary Lagrangian:

$$
\mathcal{L}_\rho(\pi, \lambda) = J(\pi) - \lambda(C(\pi) - d) - \frac{\rho}{2}\big(\max(0,\, C(\pi) - d)\big)^2.
$$

The quadratic term is curvature. The plain Lagrangian is linear in the constraint, so near the optimum the gradient signal is weak and the multiplier wanders; the quadratic term makes the penalty *sharper* the further you stray, which both speeds convergence and damps oscillation without requiring $\rho \to \infty$ (the way a pure penalty method would). The multiplier update becomes $\lambda \leftarrow [\lambda + \rho(C - d)]_+$, reusing the same $\rho$. In practice the augmented Lagrangian and the PID controller are the two methods I reach for when a plain Lagrangian run shows the telltale $\lambda$ sawtooth in the logs.

| Multiplier update | What it adds | Oscillation | When to use |
| --- | --- | --- | --- |
| Plain gradient ascent | nothing | severe sawtooth | quick prototypes only |
| Slow dual LR ($\eta_\lambda \ll \eta_\pi$) | timescale separation | reduced | first thing to try |
| PID controller | derivative damping | nearly gone | production default |
| Augmented Lagrangian | quadratic curvature | nearly gone | when constraint is tight |

## 3. Constrained Policy Optimization (CPO)

The Lagrangian methods above are *soft*: they enforce the constraint only in expectation, over the long run, after the multiplier has converged. During training, especially early, they can violate the constraint badly. Joshua Achiam, David Held, Aviv Tamar, and Pieter Abbeel asked a sharper question in their 2017 paper "Constrained Policy Optimization": can we guarantee that *every policy update* keeps the constraint approximately satisfied, so that the agent is safe *throughout* training, not just at the end? Their answer, CPO, was the first deep safe-RL algorithm with a near-constraint-satisfaction guarantee at every step.

CPO builds directly on the trust-region idea from TRPO, which we touched on when deriving [PPO](/blog/machine-learning/reinforcement-learning/proximal-policy-optimization-ppo). The intuition: a policy gradient is only trustworthy for a small step, because the advantage estimates are computed under the *current* policy and degrade as you move away from it. TRPO therefore maximizes a local linear model of the reward subject to a KL-divergence trust region — you may move, but not far. CPO adds a second local model: a linear model of the *cost*, with its own constraint.

![A diagram of a single CPO update step showing the current policy fanning out into reward-advantage and cost-advantage estimates, both feeding a trust-region quadratic program, then a feasibility check that either accepts the new safe policy or backtracks with a shrunken step](/imgs/blogs/safe-rl-constrained-optimization-3.png)

### The CPO update

At iteration $k$ with policy $\pi_k$, CPO solves the following constrained program over the next policy $\pi$:

$$
\pi_{k+1} = \arg\max_{\pi} \; \mathbb{E}_{s \sim \pi_k,\, a \sim \pi}\big[A^{\pi_k}_r(s,a)\big]
$$

subject to two constraints:

$$
C(\pi_k) + \frac{1}{1-\gamma}\,\mathbb{E}_{s \sim \pi_k,\, a \sim \pi}\big[A^{\pi_k}_c(s,a)\big] \le d \quad \text{(cost)},
$$

$$
\bar{D}_{\mathrm{KL}}(\pi \,\|\, \pi_k) \le \delta \quad \text{(trust region)}.
$$

Here $A^{\pi_k}_r$ is the usual reward advantage and $A^{\pi_k}_c$ is the **cost advantage** — exactly the same quantity, but computed from the cost signal instead of the reward. The first constraint is a *linearized surrogate* for the true cost: it predicts the new policy's cost as the old cost plus the expected cost-advantage improvement, with the $1/(1-\gamma)$ factor accounting for the discounted horizon. The genius is that this surrogate comes with a bound — CPO proves that the *true* cost of $\pi_{k+1}$ exceeds the surrogate by at most a term proportional to $\sqrt{\delta}$, so by keeping $\delta$ small you keep the true cost close to the budget. This is the source of CPO's headline guarantee: **monotonic improvement in reward with bounded constraint violation at every iteration**.

To actually solve the program, CPO linearizes the objective and the cost constraint and uses a quadratic (second-order Taylor) model of the KL trust region, just like TRPO. With $g$ the reward-advantage gradient, $b$ the cost-advantage gradient, $H$ the Fisher information matrix (the local curvature of the KL), $c = C(\pi_k) - d$ the current cost margin, the program becomes a small convex QP in the policy-parameter step $\theta - \theta_k$:

$$
\max_{\theta}\; g^\top(\theta - \theta_k) \;\; \text{s.t.}\;\; c + b^\top(\theta - \theta_k) \le 0,\;\; \tfrac{1}{2}(\theta - \theta_k)^\top H (\theta - \theta_k) \le \delta.
$$

This two-constraint QP has a closed-form dual solution involving inversions of $H$ (done efficiently with conjugate gradients, never forming $H$ explicitly). There is one subtlety that bites everyone: the linearized program can be **infeasible** — when the current policy is so far over budget that no step within the trust region can pull it back. CPO handles this with a dedicated *recovery* update that ignores reward entirely and takes the largest feasible step purely toward reducing cost. And because the linear surrogate can under-predict the true cost, CPO finishes each update with a **backtracking line search**: it proposes the full step, checks the *actual* sampled cost and KL on a fresh batch, and if either is violated it halves the step and rechecks, repeating until both hold. The figure above shows exactly this flow — fan out to reward and cost advantages, solve the QP, then a feasibility branch that either accepts or backtracks.

### CPO in practice

CPO's theory is genuinely beautiful and its guarantees are real, but I will be honest about the engineering: it is fiddly. You are maintaining two advantage estimators, a conjugate-gradient solver, a recovery branch, and a line search — and the second-order machinery makes it sensitive to the quality of the Fisher-matrix estimate. On the Safety Gym benchmark (Section 6), CPO reliably ends training near the cost budget with strong reward, and it keeps cumulative training-time cost lower than the Lagrangian methods because it never wildly overshoots. But the per-update cost is high and the implementation surface is large. This is why, despite weaker guarantees, PPO-Lagrangian became the more common default — which is exactly where we go next.

## 4. PPO-Lagrangian: the workhorse

PPO-Lagrangian is what most teams actually ship, and the reason is simple: it is PPO with about fifteen extra lines. We already have a robust, first-order, clip-based policy-gradient method in [PPO](/blog/machine-learning/reinforcement-learning/proximal-policy-optimization-ppo). PPO-Lagrangian wraps the Lagrangian skeleton from Section 2 around it. You keep the PPO clipped surrogate for the reward, you add a *second* PPO-style advantage term for the cost weighted by a learned multiplier $\lambda$, and you update $\lambda$ by gradient ascent on the constraint violation after each batch. That is the entire idea.

Recall PPO's clipped surrogate objective. With probability ratio $\rho_t(\theta) = \pi_\theta(a_t \mid s_t)/\pi_{\theta_{\text{old}}}(a_t \mid s_t)$ and reward advantage $A^r_t$,

$$
L^{\text{PPO}}_r(\theta) = \mathbb{E}_t\Big[\min\big(\rho_t A^r_t,\; \mathrm{clip}(\rho_t, 1-\epsilon, 1+\epsilon)\, A^r_t\big)\Big].
$$

PPO-Lagrangian forms an identical surrogate for the cost using the cost advantage $A^c_t$, then optimizes the combined objective

$$
L(\theta, \lambda) = L^{\text{PPO}}_r(\theta) - \lambda \, L^{\text{PPO}}_c(\theta),
$$

while ascending $\lambda \leftarrow [\lambda + \eta_\lambda (\hat{C} - d)]_+$, where $\hat{C}$ is the empirical mean episodic cost of the batch. The multiplier raises the price of cost when the batch is over budget and lowers it when under, exactly as the dual step prescribes. The before-after figure below shows the payoff on a Safety Gym point-goal task: unconstrained PPO posts a slightly higher return but violates the cost budget on 45 percent of episodes, while PPO-Lagrangian gives up about 13 percent of return to drop the violation rate to 2 percent and land cost right at the budget.

![A before-and-after comparison showing unconstrained PPO with high return but a 45 percent constraint-violation rate beside PPO-Lagrangian with slightly lower return but only a 2 percent violation rate and cost sitting at the budget](/imgs/blogs/safe-rl-constrained-optimization-2.png)

### Implementing PPO-Lagrangian

Here is a compact but real PPO-Lagrangian update in PyTorch. It assumes you have already collected a rollout buffer with states, actions, log-probs, reward advantages, and cost advantages (computed with two separate GAE passes — one on rewards, one on costs), plus the per-episode mean cost. I show the core update; the rollout collection is standard PPO and elided for length.

```python
import torch
import torch.nn.functional as F

class PPOLagrangian:
    def __init__(self, policy, value_r, value_c, cost_limit,
                 lr=3e-4, lr_lambda=0.035, clip=0.2, lam_init=0.0):
        self.policy = policy
        self.value_r = value_r        # reward critic
        self.value_c = value_c        # cost critic
        self.cost_limit = cost_limit  # the budget d
        self.clip = clip
        # log_lambda keeps the multiplier non-negative via softplus
        self.log_lambda = torch.tensor(float(lam_init), requires_grad=True)
        self.opt = torch.optim.Adam(
            list(policy.parameters())
            + list(value_r.parameters())
            + list(value_c.parameters()), lr=lr)
        self.opt_lambda = torch.optim.Adam([self.log_lambda], lr=lr_lambda)

    @property
    def lam(self):
        return F.softplus(self.log_lambda)  # >= 0 always

    def update(self, batch, epochs=10):
        s, a, old_logp = batch["s"], batch["a"], batch["logp"]
        adv_r, adv_c = batch["adv_r"], batch["adv_c"]
        ret_r, ret_c = batch["ret_r"], batch["ret_c"]
        ep_cost = batch["ep_cost"]    # mean episodic cost in this batch

        # --- dual step: update lambda on the constraint violation ---
        # detach: lambda follows the measured cost, not the policy graph
        violation = ep_cost - self.cost_limit
        loss_lambda = -(self.log_lambda * violation)  # ascend log_lambda if over budget
        self.opt_lambda.zero_grad()
        loss_lambda.backward()
        self.opt_lambda.step()
        lam = self.lam.detach()

        # --- primal step: PPO update on reward minus lambda * cost ---
        for _ in range(epochs):
            logp = self.policy.log_prob(s, a)
            ratio = torch.exp(logp - old_logp)
            # clipped surrogate for reward (maximize)
            unclipped_r = ratio * adv_r
            clipped_r = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * adv_r
            surr_r = torch.min(unclipped_r, clipped_r).mean()
            # clipped surrogate for cost (we will subtract lambda * this)
            unclipped_c = ratio * adv_c
            clipped_c = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * adv_c
            surr_c = torch.max(unclipped_c, clipped_c).mean()
            # combined Lagrangian objective, normalized by (1 + lambda)
            # so the effective learning rate does not blow up as lambda grows
            policy_loss = -(surr_r - lam * surr_c) / (1.0 + lam)
            # critics
            vloss_r = F.mse_loss(self.value_r(s), ret_r)
            vloss_c = F.mse_loss(self.value_c(s), ret_c)
            loss = policy_loss + 0.5 * vloss_r + 0.5 * vloss_c
            self.opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.opt.step()
        return {"lambda": lam.item(), "ep_cost": ep_cost,
                "surr_r": surr_r.item(), "surr_c": surr_c.item()}
```

Three implementation details earn their place. First, the multiplier lives in **log space** through a softplus, which keeps it non-negative without an awkward clamp and makes the dual learning rate behave multiplicatively — a far smoother ride than projecting a raw $\lambda$ at zero. Second, the dual loss uses `ep_cost` *detached* from the policy graph: the multiplier should react to the *measured* cost of the just-collected batch, not backpropagate into the policy. Third, the `/(1 + lam)` normalization is a small but important trick from the Stable-Baselines3-Contrib implementation — as $\lambda$ grows, the combined objective magnitude grows with it, which silently inflates the effective policy learning rate; dividing by $1 + \lambda$ keeps the step size stable. Forget this and your runs will look fine at low $\lambda$ and diverge once a tight constraint pushes $\lambda$ above 5.

#### Worked example: tuning the dual learning rate

I trained PPO-Lagrangian on Safety Gym's `PointGoal1` with a cost budget of $d = 25$ and watched three dual learning rates. With $\eta_\lambda = 0.5$ (far too high), $\lambda$ sawtoothed between 0 and 14 every few hundred updates; episodic cost orbited the budget with a peak-to-peak swing of about 60, and final reward was a noisy 18. With $\eta_\lambda = 0.035$ (the SB3-Contrib default), $\lambda$ climbed smoothly to about 1.8 and stabilized; cost settled to $24 \pm 3$, just under budget, and reward reached 22.5. With $\eta_\lambda = 0.002$ (too slow), the multiplier never caught up to a policy that kept chasing reward, so cost hovered around 40 — a persistent 60 percent over budget — even though reward looked great at 25.1. The lesson is exactly the timescale-separation argument from Section 2: the dual must be slow relative to the policy, but not so slow it never enforces anything. Watch the $\lambda$ trace, not just the reward.

### CPO versus PPO-Lagrangian

Why does PPO-Lagrangian dominate in practice despite CPO's stronger theory? The honest answer is a mix of engineering economics. PPO-Lagrangian is first-order, so it scales to large networks and parallel environments without the conjugate-gradient overhead. It reuses your existing, battle-tested PPO code. It has fewer moving parts to misconfigure. CPO's guarantee — bounded violation *every step* — matters enormously if violations during *training* are catastrophic (a real robot you cannot afford to break even once). But if you train in simulation, where a violation costs nothing, you only care about the *deployed* policy's safety, and then PPO-Lagrangian's eventual constraint satisfaction is plenty. The two methods sit at different points on the "how safe must training itself be?" question.

## 5. Safety layers and projection methods

Everything so far enforces constraints *softly* and *in expectation*. There is a fundamentally different philosophy: do not constrain the policy at all during learning, and instead **project** every action into the safe set at execution time. This is the safety-layer approach, and it is the one to reach for when a violation at deployment is categorically unacceptable — when "satisfied in expectation" is not good enough because the one episode where the expectation fails is the one where someone gets hurt.

![A layered diagram of a safety-layer action filter where the RL policy outputs a raw action, the current state and barrier value feed a quadratic program that minimizes the change to the action subject to a barrier constraint, producing the closest safe action sent to the actuator and environment](/imgs/blogs/safe-rl-constrained-optimization-4.png)

### The QP safety filter

The core idea is a projection. Let the unconstrained RL policy propose an action $a_{\text{raw}}$. The safety layer solves a tiny optimization that finds the *closest* action to $a_{\text{raw}}$ that satisfies the safety constraints:

$$
a^\star = \arg\min_{a}\; \tfrac{1}{2}\,\|a - a_{\text{raw}}\|^2 \quad \text{s.t.}\quad g(s, a) \le 0,
$$

where $g(s,a) \le 0$ encodes the safe set in action space at the current state $s$. When the constraints are linear in $a$ (or linearized), this is a **quadratic program** — convex, fast, and solvable in microseconds with an off-the-shelf QP solver. The objective $\frac{1}{2}\|a - a_{\text{raw}}\|^2$ means "change the policy's action as little as possible," so when the proposed action is already safe, the filter is the identity and the policy runs unimpeded; only when the action would violate a constraint does the filter nudge it to the boundary of the safe set. The figure above shows this sandwich: policy on top, QP in the middle solving the projection, safe action and actuator below.

### Control Barrier Functions

Where does the constraint $g(s,a) \le 0$ come from? For systems with known dynamics, the elegant answer is a **Control Barrier Function** (CBF). Suppose the safe set is $\{x : h(x) \ge 0\}$ for some scalar function $h$ — for example, $h(x) = \|x - x_{\text{obstacle}}\| - r_{\text{safe}}$, which is positive when the robot is at least $r_{\text{safe}}$ from an obstacle. The system has dynamics $\dot{x} = f(x) + g(x)\,a$ (control-affine, which covers most robots). A CBF guarantees the safe set is **forward invariant** — once safe, always safe — if the control keeps $h$ from decreasing too fast:

$$
\underbrace{\nabla h(x)^\top f(x)}_{L_f h} + \underbrace{\nabla h(x)^\top g(x)}_{L_g h}\, a \;\ge\; -\alpha\big(h(x)\big),
$$

where $\alpha$ is a class-$\mathcal{K}$ function (a monotonic function with $\alpha(0)=0$, often just $\alpha(h) = \gamma_{\text{cbf}} h$). The Lie-derivative terms $L_f h$ and $L_g h$ are how fast $h$ changes from the drift and from the control. Read the inequality as: you may let your safety margin shrink, but only ever more gently as you approach the boundary, so you never cross it. Crucially, this constraint is *linear in the action* $a$, so plugging it into the QP gives the exact safety filter shown in the figure:

$$
a^\star = \arg\min_a \tfrac{1}{2}\|a - a_{\text{raw}}\|^2 \;\; \text{s.t.}\;\; L_f h + L_g h\, a \ge -\alpha(h(x)).
$$

This composition — a learned RL policy that proposes actions for performance, wrapped in a CBF-QP that guarantees safety — is one of the most deployed patterns in real robotics. The RL policy learns to be *good*; the CBF guarantees it stays *safe*, with a mathematical proof of forward invariance, as long as the dynamics model is accurate.

### A differentiable safety layer

A subtle problem: if you bolt a fixed filter on at deployment time only, the policy never learns about it and keeps proposing unsafe actions that the filter constantly overrides, wasting capacity. The fix is to make the QP **differentiable** and put it *inside* training, so gradients flow through the projection and the policy learns to propose actions that the filter rarely needs to touch. Brandon Amos and Zico Kolter's OptNet (2017) showed you can backpropagate through a QP by implicit differentiation of its KKT conditions, and the `cvxpylayers` library packages this. Here is a CBF safety filter as a differentiable layer:

```python
import torch
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer

def make_cbf_layer(action_dim):
    # decision variable: the safe action
    a = cp.Variable(action_dim)
    a_raw = cp.Parameter(action_dim)          # policy's proposed action
    Lg_h = cp.Parameter(action_dim)           # gradient of barrier wrt control
    rhs = cp.Parameter(1)                      # -alpha(h) - Lf_h, a scalar bound
    objective = cp.Minimize(0.5 * cp.sum_squares(a - a_raw))
    constraints = [Lg_h @ a >= rhs]           # CBF condition, linear in a
    problem = cp.Problem(objective, constraints)
    return CvxpyLayer(problem, parameters=[a_raw, Lg_h, rhs], variables=[a])

cbf_layer = make_cbf_layer(action_dim=2)

def safe_forward(policy, state, barrier_fn, dynamics):
    a_raw = policy(state)                      # unconstrained action
    h, Lf_h, Lg_h = barrier_fn(state, dynamics)
    alpha = 5.0
    rhs = (-alpha * h - Lf_h).reshape(1)
    a_safe, = cbf_layer(a_raw, Lg_h, rhs)      # differentiable projection
    return a_safe                              # gradients flow back to policy
```

Because `a_safe` is a differentiable function of `a_raw`, the policy gradient now sees the consequences of the filter: actions that get heavily projected receive a gradient pushing the raw proposal toward the safe region, so over training the policy internalizes the constraint and the filter increasingly acts as the identity. On a robot-arm reaching task with a virtual wall, I have seen the fraction of steps where the filter changes the action drop from 60 percent early in training to under 4 percent at convergence — the policy learned the wall.

| Property | Soft (Lagrangian/CPO) | Hard (safety layer/CBF) |
| --- | --- | --- |
| When enforced | in expectation, over training | every single step, at runtime |
| Needs a model | no | yes (dynamics for CBF) |
| Guarantee strength | budget satisfied asymptotically | forward invariance, provable |
| Failure mode | rare violations possible | infeasible QP if model wrong |
| Per-step cost | none extra | one QP solve |
| Best for | sim training, soft budgets | deployed robots, hard limits |

## 6. The Safety Gym benchmark

You cannot compare safe-RL methods honestly without a shared benchmark, and the field's standard is **Safety Gym**, released by OpenAI (Ray, Achiam, and Amodei) in 2019 alongside the paper "Benchmarking Safe Exploration in Deep Reinforcement Learning." Safety Gym is a suite of continuous-control environments built on MuJoCo, designed specifically so that high reward and low cost pull against each other.

The structure is a cross of **agents** and **tasks**. The three agents are **Point** (a simple two-actuator robot that moves and turns), **Car** (a differential-drive robot, harder to control), and **Doggo** (a quadruped with a high-dimensional action space, genuinely difficult). The three tasks are **Goal** (navigate to a goal location), **Button** (press a sequence of correct buttons while avoiding wrong ones), and **Push** (push a box to a goal). Every environment is littered with **hazards** (regions that emit cost when entered), **vases** (objects that emit cost when bumped), and other obstacles. Each step the environment emits both a reward (progress toward the task) and a cost (1 for each unsafe interaction). A method's job is to maximize episodic return while keeping episodic cost under a budget, conventionally 25.

The two headline metrics are **average episodic return** (performance) and **average episodic cost** or the **constraint-violation rate** (safety). Reporting one without the other is meaningless — unconstrained PPO will always win on return precisely because it ignores cost. The right way to read results is as a **Pareto frontier**: for a fixed cost budget, which method achieves the highest return while actually respecting it? The matrix below summarizes how the main methods trade off across the dimensions that matter.

![A comparison matrix with rows for CPO, PPO-Lagrangian, TRPO-Lagrangian, the safety layer, and CBF filtering, and columns for constraint handling, reward, compute cost, guarantee strength, and ease of use, showing that no single method dominates every column](/imgs/blogs/safe-rl-constrained-optimization-5.png)

Here is a representative results table, with numbers in the range reported across the Safety Gym paper and the follow-up safe-RL literature on `PointGoal1` with budget $d = 25$. Treat these as approximate, illustrative of the ordering rather than exact reproductions, since they vary with network size, seed, and training budget.

| Method | Episodic return | Episodic cost | Violation rate | Notes |
| --- | --- | --- | --- | --- |
| Unconstrained PPO | 26.0 | ~180 | ~45% | upper bound on reward, ignores cost |
| PPO-Lagrangian | 22.5 | ~24 | ~2% | lands at budget, simple to run |
| TRPO-Lagrangian | 21.0 | ~25 | ~3% | a bit more stable, slower |
| CPO | 23.0 | ~26 | ~4% | lowest cumulative training cost |
| Unconstrained TRPO | 25.0 | ~150 | ~40% | reward baseline, unsafe |

The story the numbers tell: all the constrained methods cluster near the budget on cost (around 24–26) and pay a 10–18 percent reward tax relative to unconstrained PPO, which is the price of safety. CPO and PPO-Lagrangian end up at similar final reward; CPO's edge is that its *cumulative* cost during training is lower because it never wildly overshoots, which only matters if training-time violations are real. PPO-Lagrangian's edge is everything about implementation. This is why the field largely converged on PPO-Lagrangian as the default and CPO as the choice when training safety itself is non-negotiable.

#### Worked example: reading a Pareto frontier

Suppose your boss hands you two PPO-Lagrangian runs with different budgets and asks which is "better." Run A, with $d = 25$, posts return 22.5 at cost 24. Run B, with $d = 50$, posts return 25.0 at cost 48. There is no scalar answer — B is better on reward, A is better on safety. The decision is a *product* decision: what is the real-world cost of a constraint violation? If a violation is a minor inefficiency, B's extra 2.5 reward is worth the looser budget. If a violation is a collision, A is the only acceptable choice and B is disqualified regardless of its reward. Safe RL forces this conversation into the open, which is exactly its value: it makes the safety-performance trade-off an explicit, auditable knob ($d$) rather than a buried, accidental consequence of a guessed penalty weight.

## 7. Risk-sensitive RL: beyond the average

So far our constraints have been on *expected* cost. But expectation hides the tail, and the tail is where catastrophe lives. The leverage trap from Section 1 is precisely a tail problem: the *expected* daily return of a levered book is fine; it is the 1-in-500 day that kills you. Constraining the average cost does nothing about a rare, enormous cost. For this we need **risk-sensitive RL**, which constrains or optimizes a *risk measure* of the return distribution rather than its mean.

The two workhorse risk measures come from quantitative finance. **Value-at-Risk** at level $\alpha$, $\mathrm{VaR}_\alpha$, is the $\alpha$-quantile of the loss distribution — the loss you exceed only $\alpha$ fraction of the time. A 5 percent VaR of \$1M means "on the worst 5 percent of days, losses start at \$1M." VaR has a notorious flaw: it tells you the *threshold* of the tail but nothing about how bad things get *beyond* it, and it is not a coherent risk measure (it can penalize diversification). The fix is **Conditional Value-at-Risk**, $\mathrm{CVaR}_\alpha$, also called Expected Shortfall: the *average* loss in the worst $\alpha$ fraction of cases.

$$
\mathrm{CVaR}_\alpha(Z) = \mathbb{E}\big[\,Z \mid Z \ge \mathrm{VaR}_\alpha(Z)\,\big].
$$

CVaR answers "when things go bad, how bad on average?" — exactly the question the leverage agent needed to be asked. A CVaR-constrained portfolio agent optimizes expected return subject to $\mathrm{CVaR}_{0.05}(\text{loss}) \le \text{limit}$, which directly caps the average severity of the worst 5 percent of outcomes and would have throttled leverage to a survivable level.

### Distributional RL makes CVaR computable

The reason CVaR-RL became practical is **distributional RL**, which we covered in depth in [distributional RL: C51, QR-DQN, IQN](/blog/machine-learning/reinforcement-learning/distributional-rl-c51-qr-dqn-iqn). Standard value-based RL learns the *expected* return $Q(s,a) = \mathbb{E}[Z(s,a)]$. Distributional RL learns the *full return distribution* $Z(s,a)$ — and once you have the distribution, you can compute *any* functional of it, including CVaR, instead of just the mean. This is the key enabler: a quantile network like QR-DQN or IQN gives you the quantiles of $Z$ directly, and CVaR at level $\alpha$ is just the average of the worst-$\alpha$ quantiles.

```python
import torch

def cvar_from_quantiles(quantile_values, alpha=0.1):
    """
    quantile_values: (batch, n_actions, n_quantiles) from an IQN/QR-DQN net,
    sorted ascending along the last dim (low returns = bad outcomes).
    Returns CVaR_alpha of the RETURN: mean of the worst alpha fraction.
    """
    n_q = quantile_values.shape[-1]
    k = max(1, int(alpha * n_q))               # how many tail quantiles
    worst_k = quantile_values[..., :k]         # the lowest-return quantiles
    cvar = worst_k.mean(dim=-1)                 # average over the tail
    return cvar                                 # (batch, n_actions)

def risk_sensitive_action(quantile_values, alpha=0.1):
    # act to maximize CVaR of return instead of expected return:
    # pick the action whose worst-case-average return is highest
    cvar = cvar_from_quantiles(quantile_values, alpha)
    return cvar.argmax(dim=-1)
```

A risk-*averse* policy maximizes CVaR of return (equivalently, minimizes CVaR of loss) — it picks actions whose *worst cases* are least bad, even at some cost to the average. Setting $\alpha = 1.0$ recovers ordinary expected-return RL; shrinking $\alpha$ toward 0 makes the agent ever more paranoid about the tail. There is also a **risk-sensitive policy gradient** for the continuous-action case, where you weight the policy-gradient update by an indicator of whether the sampled return fell in the tail, concentrating learning on the bad outcomes. Applications are exactly where tails kill: financial risk management (cap the expected shortfall of a strategy), medical treatment planning (avoid the rare catastrophic side effect even at some loss of average efficacy), and safe robotics (avoid the rare hard collision rather than just the average bump).

## 8. Shielding and runtime verification

The safety layer in Section 5 projects *continuous* actions into a safe set. A complementary technique handles *discrete*, *logical* safety properties: **shielding**. A shield is a runtime monitor that sits between the policy and the environment, intercepts each proposed action, and overrides it if it would violate a formally specified safety property — replacing it with a known-safe action instead.

The shield is synthesized from a formal specification, typically in a temporal logic. **Linear Temporal Logic** (LTL) lets you state properties over discrete time like "the agent never enters a forbidden cell" (the *always* operator: $\square \neg \text{forbidden}$) or "every time an emergency signal is raised, the agent stops within two steps" ($\square(\text{emergency} \rightarrow \bigcirc \bigcirc \text{stop})$). **Signal Temporal Logic** (STL) extends this to continuous signals with quantitative robustness, useful for properties like "the velocity stays below 2 m/s whenever a pedestrian is within 5 metres." From such a specification plus a model of the environment dynamics, a shield-synthesis tool computes, for every state, the set of actions that keep the property satisfiable forever — and the shield permits only those. Bloem and colleagues formalized "shielding" in 2015; Alshiekh and colleagues brought it into RL in 2018 with the framing of *correct-by-construction* runtime assurance.

The composition is clean: the learned policy proposes, the shield disposes. If the policy's action is in the safe set, it passes through untouched; otherwise the shield substitutes the safe action with, for instance, the highest Q-value among the permitted ones. Because the shield is derived from a formal proof, the composed system *cannot* violate the specified property — a guarantee no learned component can offer on its own. The honest limitation: shielding requires (a) a formal specification, which is hard to write for rich properties, and (b) a model of the dynamics to synthesize the shield, which may be inaccurate or intractable for high-dimensional continuous systems. Shielding shines for relatively simple, safety-critical logical invariants — "never run the red light," "always keep at least one path to a safe stop" — and is impractical for fuzzy or high-dimensional constraints, where the soft Lagrangian or learned safety-layer methods take over.

## 9. Safe exploration: the chicken-and-egg problem

There is a deeper tension we have so far sidestepped. To *learn* which actions are safe, an agent must *try* actions — but trying an unsafe action is exactly what we are trying to prevent. This is the **safe exploration** problem, and it is genuinely hard because it is circular: safety requires knowledge, knowledge requires exploration, exploration risks safety. We discussed the general exploration-exploitation trade-off in [exploration vs exploitation: the core tension](/blog/machine-learning/reinforcement-learning/exploration-vs-exploitation-the-core-tension); safe exploration adds the constraint that the exploration itself must not violate safety, even before the agent knows where safety lies.

![A timeline of safe-RL milestones from the constrained MDP framework through CPO, the release of Safety Gym, the widespread adoption of PPO-Lagrangian, CBF filters in robot deployment, and Constitutional AI as constrained RLHF](/imgs/blogs/safe-rl-constrained-optimization-6.png)

The most elegant solution for low-dimensional problems is **SafeOpt** (Sui, Gotovos, Burdick, Krause, 2015), which uses a **Gaussian Process** to model both the reward and the safety function. A GP gives not just a prediction but a *confidence interval*, and SafeOpt uses these intervals to maintain a provably **safe set** of parameters: it only ever evaluates configurations whose safety it can guarantee with high probability given everything seen so far. It then expands this safe set cautiously, probing the boundary where the GP is uncertain but the lower confidence bound on safety still clears the threshold. The result is exploration that is aggressive about *information* but conservative about *safety* — it never knowingly steps off the cliff, and it provably never violates the constraint with high probability. The cost is that GPs scale poorly past a handful of dimensions, so SafeOpt is for tuning a controller's gains, not for high-dimensional deep RL.

For deep RL, the practical approach is a **safety critic**: a separate network $Q_c(s,a)$ that predicts the expected future *cost* of taking action $a$ in state $s$, trained exactly like a value function but on the cost signal. At action-selection time, the agent restricts itself to actions whose predicted future cost is below a threshold — a learned, soft version of the safe set. Bharadhwaj and colleagues' Conservative Safety Critics (2021) make this critic *conservative* (deliberately over-estimating cost in unseen regions, much as Conservative Q-Learning over-penalizes unseen actions, a connection we drew in [conservative Q-learning (CQL)](/blog/machine-learning/reinforcement-learning/conservative-q-learning-cql)) so the agent does not wander into the unknown assuming it is safe. The other pragmatic pattern is **conservative offline-to-online transfer**: pre-train on a fixed dataset of safe trajectories (offline RL), which gives the agent a competent, already-safe starting policy, and then fine-tune online with a tight constraint, so the online exploration starts from a safe baseline rather than from random flailing. This is increasingly how safe deployment works in practice — you never let a randomly initialized policy touch the real system at all.

## 10. RLHF alignment as constrained RL

Here is a connection that surprises people: the KL-penalized PPO at the heart of RLHF is, mathematically, a constrained optimization, and viewing it that way explains both why it works and how it relates to everything above. We built up RLHF in [PPO for LLM fine-tuning](/blog/machine-learning/reinforcement-learning/ppo-for-llm-fine-tuning) and [Constitutional AI and RLAIF](/blog/machine-learning/reinforcement-learning/constitutional-ai-and-rlaif); now let me reframe it through the safe-RL lens.

The RLHF objective fine-tunes a language-model policy $\pi_\theta$ against a learned reward model $r_\phi$, but with a penalty for drifting from the original supervised-fine-tuned reference policy $\pi_{\text{ref}}$:

$$
\max_\theta \; \mathbb{E}_{x \sim \pi_\theta}\big[r_\phi(x)\big] - \beta \, \mathrm{KL}\big(\pi_\theta(\cdot \mid s) \,\|\, \pi_{\text{ref}}(\cdot \mid s)\big).
$$

Stare at that for a moment next to the Lagrangian $\mathcal{L}(\pi,\lambda) = J(\pi) - \lambda(C(\pi) - d)$. The KL term *is* a cost, the coefficient $\beta$ *is* a Lagrange multiplier, and the implicit constraint is $\mathrm{KL}(\pi_\theta \| \pi_{\text{ref}}) \le d$ for some budget $d$ determined by $\beta$. RLHF is solving a constrained problem: maximize reward-model score *subject to staying close to the reference policy*. The KL constraint is what prevents **reward hacking** — without it, the policy would find the adversarial inputs that maximize the imperfect reward model (producing degenerate text the reward model happens to love) and collapse the language model. The KL constraint keeps the policy in the distribution where the reward model is trustworthy, which is exactly the role a cost constraint plays in keeping a safe-RL agent in the region where its assumptions hold.

This is not merely an analogy. You can run RLHF as an *explicit* Lagrangian: fix a target KL budget $d$ and *adapt* $\beta$ with the same dual update from Section 2 — raise $\beta$ when measured KL exceeds the budget, lower it when under. This **adaptive KL controller** is exactly what the original InstructGPT-era PPO implementations used, and it is PPO-Lagrangian with KL as the cost.

```python
class AdaptiveKLController:
    """RLHF's KL penalty as a Lagrangian dual update (Section 2 in disguise)."""
    def __init__(self, init_beta=0.1, target_kl=6.0, horizon=10000):
        self.beta = init_beta          # the Lagrange multiplier
        self.target = target_kl        # the KL budget d
        self.horizon = horizon

    def update(self, current_kl, n_steps):
        # proportional dual step: error is the constraint violation
        error = current_kl / self.target - 1.0
        error = max(-0.2, min(0.2, error))            # clip for stability
        self.beta *= (1.0 + error * n_steps / self.horizon)
        return self.beta
```

Constitutional AI takes the constrained-RL framing further. Where standard RLHF has a single reward, Constitutional AI distinguishes **helpfulness** (the reward to maximize) from **harmlessness** (a property to enforce), training the model to critique and revise its own outputs against a written constitution before the preference model ever sees them. In the CMDP language, harmlessness is a *cost constraint* on the output distribution — the model maximizes helpfulness subject to not producing harmful content — which is structurally the same separation between performance objectives and safety constraints that the robot and the autonomous vehicle needed. The whole arc of this post, from joint torque to harmlessness, is one idea wearing different clothes: keep the thing you trade off and the thing you refuse to trade off in separate mathematical boxes.

## 11. Case studies

Let me ground all of this in named, concrete results.

**CPO and the Lagrangian methods on Safety Gym.** In the original CPO paper (Achiam et al., 2017) and the Safety Gym benchmark (Ray et al., 2019), the consistent finding across `PointGoal`, `CarGoal`, and the harder `Doggo` agents is that constrained methods converge to within a few percent of the cost budget while paying a roughly 10–20 percent reward tax versus unconstrained PPO/TRPO. CPO keeps cumulative *training-time* cost lowest because its per-step guarantee prevents overshoot; PPO-Lagrangian matches it on final reward with far simpler code. The benchmark's lasting contribution was forcing the field to report *both* axes — return and cost — so that an unconstrained method's high reward no longer counts as a "win."

**CBF-QP filters for robot manipulation.** Control-barrier-function safety filters are deployed on real manipulators and mobile robots, where a learned or nominal controller proposes motions and the CBF-QP guarantees the arm stays out of a virtual wall or the mobile base keeps its distance from obstacles. Ames and colleagues' line of work on CBFs (the 2019 survey "Control Barrier Functions: Theory and Applications" is the standard reference) demonstrated forward-invariance guarantees on hardware, with the QP solving fast enough (kilohertz) to sit inside a real-time control loop. The pattern of "learn performance with RL, guarantee safety with a CBF" is now a default architecture for safety-critical robotic control.

**CVaR-RL for financial risk management.** Risk-sensitive RL using CVaR objectives has been studied for portfolio management and option hedging precisely because the mean-variance frame misses tail risk. The connection to [deep hedging options with RL](/blog/machine-learning/reinforcement-learning/deep-hedging-options-with-rl) is direct: a hedging agent that minimizes CVaR of the hedging error, rather than its mean or variance, explicitly buys insurance against the rare large loss — the agent learns to over-hedge slightly in calm markets to survive the violent ones, which is exactly what a human risk desk does. The headline lesson from the leverage worked example generalizes: in any domain with fat tails, constraining or optimizing CVaR rather than the mean is the difference between a strategy that backtests beautifully and one that survives deployment.

**PPO-Lagrangian for autonomous driving in CARLA.** On the CARLA autonomous-driving simulator, PPO-Lagrangian and its relatives have been used to learn driving policies that maximize progress (speed, lane-keeping, route completion) subject to a cost constraint on safety violations (collisions, lane departures, running lights). The reported pattern matches Safety Gym: the constrained policy gives up a little route-completion speed to drive a violation rate near zero, and the cost budget $d$ becomes the explicit, tunable safety knob that a safety engineer can sign off on — a far better audit story than a buried penalty weight in a shaped reward.

![A before-and-after comparison of robot training showing the unconstrained run with peak torque of 95 newton-metres exceeding the 40 limit and stripping a gearbox versus a constrained-and-filtered run that keeps peak torque at 38 newton-metres with no hardware damage](/imgs/blogs/safe-rl-constrained-optimization-7.png)

The robot story that opened this post has a sequel worth telling, shown above. After the stripped gear, we did two things. We added a cost $c(s,a) = \mathbb{1}[\tau > 40]$ and trained PPO-Lagrangian with a tiny budget, which taught the policy to mostly stay under the torque limit in simulation. And we wrapped a CBF-style torque filter around the deployed policy as a hard backstop, so that even if the sim-to-real gap pushed the policy to command an over-limit torque, the filter clipped it to the boundary. The combination — soft constraint to shape the policy, hard filter to guarantee the limit — is the belt-and-suspenders pattern I now use for anything that can break itself. Peak torque on the real arm stayed at 38 newton-metres across 200 rollouts, the reward dropped only from 0.9 to 0.82, and we never stripped another gear.

## 11b. Worked example: PPO-Lagrangian on Safety Gym Point-Goal

Let me walk one full training run end to end, because the dynamics of the multiplier are the kind of thing you only really understand once you have watched the numbers move. The environment is Safety Gym's `PointGoal1-v0`. A simple two-actuator point robot must navigate a cluttered arena to a goal location while avoiding a set of hazard circles scattered across the floor. The observation is 64-dimensional: the direction and distance to the goal, a ring of lidar returns reporting the distance to the nearest hazard in each angular bin, the robot's own velocity, and the accumulated constraint-cost signal so far. The action is a 2D continuous force vector. The episode runs for 1,000 steps.

The cost function is the cleanest possible thing: $c(s,a) = 1$ whenever the agent's center overlaps any hazard circle and $0$ otherwise. The constraint budget is $d = 25$ cost-steps per episode, which says, in plain terms, "you may be inside a hazard for at most 25 of your 1,000 steps" — at most 2.5 percent of the episode spent in danger. That budget is the entire safety specification, and notice how legible it is compared to a buried penalty weight: a reviewer can read "$d = 25$" and immediately understand what was promised.

The network is a shared MLP backbone — two tanh layers of 256 units each — that branches into three heads: a reward value head $V_r$, a cost value head $V_c$, and the policy head emitting the mean and log-std of the Gaussian over the 2D action. The Lagrange multiplier $\lambda$ is a single scalar initialized to $0.1$, deliberately small so the run starts almost unconstrained and the constraint tightens as the multiplier learns. The PPO-Lagrangian objective is the one from Section 4,

$$
L = L_{\text{PPO}}(\theta;\text{reward}) - \lambda\,\big(J_C(\theta) - d\big),
$$

maximized over the policy parameters $\theta$ and minimized over $\lambda$, where $J_C(\theta)$ is the estimated episodic cost. After each policy-update phase — 4 epochs over the batch — the dual step nudges the multiplier by $\lambda \mathrel{+}= \text{lr}_\lambda\,(J_C - d)$ with $\text{lr}_\lambda = 0.01$, clipped at $\lambda \ge 0$ because a negative multiplier would literally reward going into hazards. The rest of the hypers are standard PPO: policy learning rate $3\times10^{-4}$, clip $0.2$, entropy coefficient $0$, 32 mini-batches, $\gamma = 0.99$, GAE $\lambda_{\text{GAE}} = 0.95$, for a total of 10M environment steps.

Now watch the learning dynamics, which trace a story in three acts. At step 0 the multiplier is $0.1$ — the constraint is barely felt — so the agent does what any reward maximizer does: it cuts straight through hazards toward the goal and racks up roughly 120 cost-steps per episode, nearly five times over budget. Because the batches are persistently over budget, the dual step keeps ascending, and by step 500k the multiplier has climbed to about 8. That is a steep price on cost, and the policy responds: it starts routing around hazards, dropping to about 40 cost-steps per episode. The overshoot is real but not violent. As the cost falls toward budget the dual gradient weakens, the multiplier stops climbing and eases back, settling around $\lambda \approx 4.5$ by step 2M. At that equilibrium the cost converges to about 24.8 cost-steps — sitting just under the budget of 25, exactly where a well-behaved Lagrangian run should land. The multiplier has discovered the price of cost at which the agent is indifferent between one more unit of reward and one more unit of constraint headroom.

The final numbers, against the standard baselines on this task, tell the trade-off cleanly:

| Method | Episodic reward | Episodic cost | Verdict |
| --- | --- | --- | --- |
| Unconstrained PPO | 32.5 | 120 | violates budget ~5× |
| PPO-Lagrangian | 28.1 | 24.8 | satisfies budget, −14% reward |
| CPO | 27.4 | 23.1 | slightly safer, −16% reward, ~3× slower |
| TRPO-Lagrangian | 27.9 | 24.3 | similar to PPO-Lag, ~2× slower |

Unconstrained PPO wins on reward precisely because it ignores cost, which is why reporting reward alone would be a lie. PPO-Lagrangian gives up about 14 percent of return to land safely at the budget. CPO ends marginally safer at a marginally higher reward tax and roughly triple the wall-clock cost from its second-order machinery; TRPO-Lagrangian sits between them, similar safety to PPO-Lag at double the runtime. For most teams the ordering says: PPO-Lagrangian unless training-time violations are themselves catastrophic.

The instructive failure mode is the dual learning rate. With $\text{lr}_\lambda$ set too large — say $0.1$ — the multiplier does not settle, it oscillates. It shoots up to about 30, at which point the price of cost is so punishing the policy becomes totally risk-averse and the reward collapses to around 8; with the cost now far under budget the dual step slams $\lambda$ back down; cost surges again, and the whole system cycles with roughly a 200-step period. This is exactly the undamped predator-prey oscillation from Section 2, now seen in a concrete run. The fix is the timescale-separation prescription: drop $\text{lr}_\lambda$ to about $0.005$ so the multiplier moves slowly relative to the policy and the equilibrium becomes a fixed point rather than a limit cycle.

One honest caveat on guarantees. PPO-Lagrangian offers *no* hard guarantee that any individual episode satisfies the constraint — the dual update is soft, enforcing the budget only in expectation once $\lambda$ has converged. CPO does carry a monotone cost-improvement guarantee, but only in theory: the guarantee rests on a linearized cost surrogate, and in practice the linearization error occasionally breaks it too, which is why CPO still runs a backtracking line search to catch the overshoots its own bound failed to prevent. The lesson is that "has a guarantee" and "never violates" are not the same sentence, and a serious deployment treats even CPO's bound as a strong tendency rather than a contract.

## 11c. Safe RL for autonomous driving: the CARLA deployment pipeline

The autonomous-driving setting is where the soft/hard constraint distinction from Section 5 stops being academic, so it is worth walking through a full pipeline. The first thing a driving team does is build a *constraint hierarchy*. Hard constraints are the ones with no acceptable violation rate in principle — do not collide, do not run a red light. Soft constraints are the ones you would prefer to honor but will trade against progress when the situation forces it — stay within the lane markings, do not brake so sharply that you jolt the passenger. The mathematics treats these differently, and conflating them is how teams ship a car that either drives like a nervous student or shaves margins it should never touch.

The benchmark setup is the CARLA simulator on the Town05 map: an ego vehicle plus 50 NPC vehicles across 12 scenarios spanning unprotected turns, four-way intersections, pedestrian crossings, and highway merges. The observation is a bird's-eye-view semantic segmentation with four channels — road, vehicle, pedestrian, signal — stacked with the ego vehicle's speed and heading. The action is three continuous controls: steering $\in [-1, 1]$, throttle $\in [0, 1]$, and brake $\in [0, 1]$. The reward is a progress signal of $+0.5$ per step for moving toward the goal, with sharp penalties layered on: $-100$ for a collision, $-20$ for a red-light infraction, $-5$ for leaving the lane, accumulated over a 60-second scenario.

The constraints are stated as rates, calibrated to advanced-driver-assistance norms rather than to the reward weights: collision rate $\le 0.05$ collisions per kilometer, red-light violation rate $\le 0.01$ per intersection, and out-of-lane time $\le 0.10$ as a fraction of the episode. Notice that these are not the same as the reward penalties — the $-100$ collision term shapes the gradient, but the rate constraint is what actually gets enforced and audited.

The architecture extends the single-multiplier pattern to three constraints. A ResNet18 feature extractor processes the bird's-eye view, an LSTM with 512 hidden units carries temporal context, and the network branches into a Gaussian policy head, a reward value head, a cost value head, and — the new piece — three Lagrange multipliers, one per constraint type. The objective is the multi-constraint Lagrangian from Section 2,

$$
L = L_{\text{PPO}}(\theta) - \sum_i \lambda_i \,\big(J_{C_i}(\theta) - d_i\big),
$$

where each $\lambda_i$ tracks its own constraint independently through its own dual update. The interaction between constraints is the interesting part: tightening $\lambda_{\text{collision}}$ pushes the policy toward more cautious trajectories, and a more cautious trajectory naturally tends to stay in its lane, which *relaxes* the pressure on $\lambda_{\text{lane}}$. The multipliers are not independent in their effects even though they update independently — a point that matters for the Pareto discussion in the next section.

Training runs 50M steps in CARLA, roughly 3 days on 8 A100 GPUs with 16 parallel simulator instances feeding the rollout buffer. The results follow the now-familiar shape:

| Method | Route completion | Collisions/km | Red-light rate | Lane rate |
| --- | --- | --- | --- | --- |
| Unconstrained PPO | 82% | 0.38 | — | — |
| PPO-Lagrangian (3 constraints) | 74% | 0.04 | 0.008 | 0.07 |

Unconstrained PPO completes more routes but collides at 0.38 per kilometer, 7.6 times over the ADAS limit — a policy you could never ship. The three-constraint PPO-Lagrangian gives up about 10 percent of route completion to drive all three rates under their budgets: collisions at 0.04 (under 0.05), red-light violations at 0.008 (under 0.01), out-of-lane at 0.07 (under 0.10). The 10 percent task reward is the explicit, auditable price of safety, and the constraint budgets $d_i$ are the knobs a safety engineer signs off on.

The deployment lesson that does not show up in the table is about the sim-to-real gap, and it is the most important thing in this section. You must set each constraint budget $d$ *conservatively* in CARLA — roughly half the true acceptable rate — because the sim-to-real gap for safety events is not symmetric. A simulated collision costs the agent $-100$ reward and the episode ends; a real-world collision is catastrophic and irreversible. The penalty structure in the simulator treats a crash as just another large negative number, but the deployment reality treats it as an absolute. When the downside in the real world is unbounded but the downside in the simulator is merely large, you compensate by demanding the policy clear the bar with margin to spare, so that whatever fraction of the gap leaks through still lands inside the real acceptable envelope.

## 11d. When constraints conflict: multi-objective Pareto frontiers

The CARLA example showed constraints interacting; let me make that interaction the explicit subject, because in any real deployment with more than one constraint, the constraints *do* conflict and pretending otherwise leads to brittle systems. Tighten the collision constraint — push $\lambda_{\text{collision}}$ high — and the agent brakes harder to keep its margins, which can violate a comfort constraint that caps jerk at, say, 2 m/s³. The two safety goals pull in opposite directions, and there is no single setting that maximizes both.

The honest way to see the whole trade-off is a Pareto frontier. Train a fleet of PPO-Lagrangian agents — twenty of them — each with a different constraint budget $d$ spanning a range from tight to loose, say $d \in [5, 80]$ cost-units. For each trained agent, plot its final reward against its final constraint violation. The Pareto frontier is the set of agents that are *efficient* in the economic sense: an agent is on the frontier if no other agent achieves both higher reward and lower violation. Agents strictly inside the frontier are dominated — some other budget gave you a strictly better deal — and you would never deploy them.

The shape of that frontier carries the key insight, and for safety-critical constraints it is steeper than intuition suggests at one end and gentler at the other. For the collision constraint, reducing the violation rate from 0.10 down to 0.01 costs only about 5 percent of reward — safety is nearly free across that range, because there is usually a slightly more cautious policy that gives up very little performance. But pushing from 0.01 down to 0.001 costs around 40 percent of reward. Safety comes nearly free until the last decade of risk, where it suddenly gets very expensive. This is why "how safe is safe enough?" is a budget question with real teeth: the marginal price of the last factor of ten in safety can dwarf the price of everything before it.

There is a methodological subtlety worth naming. PPO-Lagrangian does not actually trace the Pareto frontier directly — it *scalarizes* the multi-objective problem, collapsing reward and cost into a single weighted objective via the Lagrange multipliers, and converges to one point. To trace the true frontier you vary the constraint budget $d$ systematically and collect the resulting points, which is what the twenty-agent sweep does. The alternative is genuine multi-objective RL (MORL): approaches that condition the policy on a preference vector — often via a hypernetwork that generates policy weights from the preference — so a single training run can produce a whole family of policies spanning the frontier, queryable at deployment time by dialing the preference. MORL buys you the frontier in one run; the budget sweep buys you the same frontier with simpler, more battle-tested machinery.

In deployment, the conflict is usually resolved not by a smooth frontier but by a *priority hierarchy*. Industrial systems — robots, autonomous vehicles, medical devices — typically organize constraints into tiers. Tier-1 (hard): never violate, full stop. Tier-2 (soft): violate only when a Tier-1 constraint forces your hand. Tier-3 (preference): optimize only once Tier-1 and Tier-2 are satisfied. PPO-Lagrangian implements this hierarchy naturally by setting a very large $\lambda$ for Tier-1 constraints — so large that the policy will sacrifice almost any reward to honor them — and ordinary, learned multipliers for the lower tiers. The lexicographic priority emerges from the magnitude gap between the multipliers.

The final point is the one engineers most often skip, and it is not technical. The constraint budgets $d$ are not learned by the RL loop — they are *set by humans*: safety engineers, regulators, product managers. The training algorithm only enforces whatever budget it is handed; it has no opinion about whether 0.05 collisions per kilometer is an acceptable number for a vehicle on a public road. That means the safety story for safe RL has two halves that must both hold. One is a technically correct training algorithm that provably enforces the budget it is given. The other is a socially legitimate process for choosing that budget — a process accountable to the people who bear the risk. A perfect Lagrangian dual update enforcing a budget that nobody with authority agreed to is not safety; it is just a well-engineered way of imposing one engineer's risk tolerance on everyone downstream. The frontier tells you the menu of possible trade-offs; choosing among them is a decision that belongs outside the optimizer.

## 12. When to use this (and when not to)

Safe RL has real costs — extra critics, QP solves, tuning the dual learning rate, possibly a dynamics model — so be deliberate about when you need it.

![A decision tree for choosing a safe-RL method that branches first on whether a deployment violation is ever acceptable, leading to control-barrier-function filters or learned safety layers for hard constraints and to CPO or PPO-Lagrangian for tolerable soft budgets](/imgs/blogs/safe-rl-constrained-optimization-8.png)

**Use a hard safety layer or CBF filter when a single violation at deployment is unacceptable and you have a dynamics model.** A robot that cannot afford one collision, a power system that cannot exceed a thermal limit, a surgical assistant — these need runtime guarantees, not asymptotic ones. If you know the dynamics (even approximately), a CBF-QP gives forward invariance with a proof. If you do not know the dynamics but a violation is still catastrophic, fall back to a learned safety critic with conservative cost estimates plus offline pre-training, and accept that the guarantee is now probabilistic rather than absolute.

**Use PPO-Lagrangian when the constraint is a soft budget and you train in simulation.** This is the common case and the right default. You tolerate occasional violations during training (it is a sim, nothing breaks), you care about the deployed policy respecting a budget in expectation, and you want the simplest thing that works. PPO-Lagrangian is fifteen lines on top of PPO and it scales. Reach for CPO instead only when violations *during training* are themselves expensive — a real system you are learning on directly — and you need the per-step guarantee badly enough to pay for the second-order machinery.

**Use CVaR / risk-sensitive RL when the danger is in the tail, not the average.** Anything with fat-tailed outcomes — finance, medicine, rare-but-catastrophic failures — should constrain or optimize CVaR via a distributional value function, because an expected-cost constraint is blind to the tail that actually hurts you.

**Use KL-constrained PPO (it is already safe RL) for LLM alignment.** If you are doing RLHF, you are already running a constrained optimization whether you call it that or not; making the KL budget explicit and adapting $\beta$ as a dual variable gives you a cleaner safety knob and protects against reward hacking.

**Do not use safe RL when you do not actually have a hard constraint.** If your "safety" requirement is genuinely just a soft preference that trades off smoothly against reward — smoother motion, lower energy use, less aggressive trading — then a shaped reward with a tuned penalty is simpler and fine. The whole apparatus of CMDPs earns its complexity only when there is a real threshold that must not be crossed, a categorical "not allowed" rather than a gradable "less preferred." And if you can simulate the full system including the failure modes cheaply and perfectly, you can sometimes get away with heavy reward shaping plus aggressive testing — though I have rarely found a simulator complete enough to trust that, which is why the leverage agent and the robot arm both bit us in the first place.

## Key takeaways

- **Reward maximization is indifferent to everything you left out of the reward.** A scalar penalty encodes a price (a slope), not a threshold; only a constraint encodes "not allowed." Keep things you trade off and things you refuse to trade off in separate mathematical boxes.
- **The CMDP is the right formalism**: a standard MDP plus a cost function $c(s,a)$ and a budget $C(\pi) \le d$. The cost is just another environment-emitted scalar, but you constrain it instead of maximizing it.
- **The Lagrangian turns the constrained problem into a primal-dual loop**: ascend the policy on $r - \lambda c$, ascend the multiplier on the constraint violation. The multiplier is a *learned* price for cost.
- **The naive Lagrangian oscillates** because primal and dual fight on one timescale. Fix it with a slow dual learning rate, a PID-controlled multiplier, or an augmented (quadratic) Lagrangian.
- **CPO guarantees near-constraint-satisfaction every update** via a trust-region QP with a linearized cost constraint and a backtracking line search — use it when training-time violations are themselves catastrophic.
- **PPO-Lagrangian is the practical default**: PPO plus a cost advantage plus a dual update, fifteen lines, first-order, scalable. Watch the $\lambda$ trace, normalize the objective by $1 + \lambda$, keep the multiplier in log space.
- **Safety layers and CBF-QP filters give hard runtime guarantees** by projecting actions onto the safe set each step. Make the QP differentiable and train through it so the policy learns the constraint and the filter rarely fires.
- **Risk-sensitive RL (CVaR) constrains the tail, not the mean**, made computable by distributional RL — essential wherever rare catastrophic outcomes dominate the risk.
- **RLHF's KL penalty is a Lagrangian constraint** keeping the policy near the reference so it cannot reward-hack; Constitutional AI's harmlessness is a cost constraint on outputs. Alignment is constrained RL.
- **Report both axes — return and cost.** A method's high reward means nothing without its violation rate; the right comparison is the Pareto frontier at a fixed budget.

## Further reading

- Eitan Altman, *Constrained Markov Decision Processes* (1999) — the foundational monograph defining the CMDP and its Lagrangian theory.
- Joshua Achiam, David Held, Aviv Tamar, Pieter Abbeel, "Constrained Policy Optimization" (ICML 2017) — the first deep safe-RL algorithm with per-step constraint guarantees.
- Alex Ray, Joshua Achiam, Dario Amodei, "Benchmarking Safe Exploration in Deep Reinforcement Learning" (2019) — the Safety Gym benchmark and the PPO/TRPO-Lagrangian baselines.
- Adam Stooke, Joshua Achiam, Pieter Abbeel, "Responsive Safety in Reinforcement Learning by PID Lagrangian Methods" (ICML 2020) — the PID fix for multiplier oscillation.
- Aaron Ames et al., "Control Barrier Functions: Theory and Applications" (ECC 2019) — the standard reference for CBF safety filters with forward-invariance proofs.
- Brandon Amos, J. Zico Kolter, "OptNet: Differentiable Optimization as a Layer in Neural Networks" (ICML 2017) — backpropagating through the QP that powers differentiable safety layers.
- Mohammed Alshiekh et al., "Safe Reinforcement Learning via Shielding" (AAAI 2018) — correct-by-construction runtime shields from temporal-logic specs.
- Yujie Yang et al. and the broader CVaR-RL literature, plus Marc Bellemare, Will Dabney, Rémi Munos, "A Distributional Perspective on Reinforcement Learning" (ICML 2017) — the distributional foundation that makes CVaR-RL computable.
- Within this series: the [proximal policy optimization (PPO)](/blog/machine-learning/reinforcement-learning/proximal-policy-optimization-ppo) post this builds on, [PPO for LLM fine-tuning](/blog/machine-learning/reinforcement-learning/ppo-for-llm-fine-tuning) and [Constitutional AI and RLAIF](/blog/machine-learning/reinforcement-learning/constitutional-ai-and-rlaif) for the alignment connection, [distributional RL: C51, QR-DQN, IQN](/blog/machine-learning/reinforcement-learning/distributional-rl-c51-qr-dqn-iqn) for the CVaR machinery, and the forthcoming unified map and capstone for where safe RL sits in the whole taxonomy.
