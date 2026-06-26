---
title: "Trust Region Policy Optimization: The Monotonic Improvement Guarantee"
date: "2026-06-22"
publishDate: "2026-06-22"
description: "Why a single policy-gradient step can erase weeks of training, and how TRPO turns gradient ascent into a provably non-degrading update using a KL trust region, the natural gradient, and conjugate gradient."
tags:
  [
    "reinforcement-learning",
    "deep-learning",
    "policy-gradient",
    "trpo",
    "natural-gradient",
    "actor-critic",
    "machine-learning",
    "pytorch",
  ]
category: "machine-learning"
subcategory: "Reinforcement Learning"
author: "Hiep Tran"
featured: true
readTime: 55
image: "/imgs/blogs/trust-region-policy-optimization-trpo-1.png"
---

The first time I watched a policy-gradient agent destroy itself, I did not believe the plot. A continuous-control agent on HalfCheetah had been climbing steadily for three hundred updates, the average return crossing nine hundred, the gait starting to look like an actual run instead of a seizure. Then in one update the return fell off a cliff — from nine hundred to eighty, a number it had passed on the way up in the first ten minutes of training. The optimizer had not crashed. The learning rate had not changed. A single gradient step had walked the policy into a region of parameter space where it produced garbage, and because the next batch of data was now collected by that garbage policy, there was no gradient pointing back to where it had been. Weeks of compute, gone in one step.

That failure is not a bug in your code. It is the defining pathology of naive policy-gradient methods, and it is the reason Trust Region Policy Optimization (TRPO) exists. The diagnosis is simple to state: when you optimize a policy by taking a gradient step in parameter space, the size of that step in *parameter* space tells you almost nothing about the size of the change in *behavior*. A tiny nudge to the weights of a neural network can move the action distribution enormously, and once the policy moves too far, the data you collected under the old policy stops being a valid estimate of how the new policy behaves. The surrogate objective you were ascending becomes a fantasy. You step confidently off a cliff.

![Unconstrained policy gradient can collapse return after one bad step while TRPO keeps each update inside a KL-bounded trust region and improves monotonically](/imgs/blogs/trust-region-policy-optimization-trpo-1.png)

TRPO, introduced by Schulman, Levine, Abbeel, Jordan, and Moritz in 2015, fixes this by refusing to take a step that changes the policy too much. It formalizes "too much" with the Kullback-Leibler (KL) divergence between the old and new action distributions, and it solves a constrained optimization problem at every update: maximize the expected improvement subject to a hard cap on how far the policy is allowed to move. The payoff is a theorem — a guarantee that, under the idealized version of the algorithm, every update improves (or at least does not worsen) the true expected return. No more cliffs. By the end of this post you will understand the monotonic improvement theorem that makes that guarantee, derive the trust-region surrogate from it, see why the natural gradient is the right step direction, and implement the conjugate-gradient and line-search machinery that makes TRPO actually run. You will also understand exactly why, despite all that elegance, most practitioners reach for PPO instead — and when they should not.

This post sits in Track E of the series, building directly on the policy-gradient and actor-critic foundations. If you have not internalized the policy gradient theorem and the advantage function, skim the [unified map of RL methods](/blog/machine-learning/reinforcement-learning/reinforcement-learning-a-unified-map) first; TRPO is the principled answer to a problem those methods leave open. The thread running through everything below is a single question — *how far can I trust data collected under the old policy to predict the behavior of a new one?* — and you will see that question answered three times, at increasing levels of rigor: first as an intuition, then as a theorem with an explicit constant, and finally as a system of linear algebra you can run on a GPU.

## 1. Why vanilla policy gradient is brittle

Let us be precise about the failure. A policy $\pi_\theta(a \mid s)$ is a distribution over actions parameterized by $\theta$ (the weights of a neural network). The objective in reinforcement learning is the expected discounted return:

$$
J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^{\infty} \gamma^t r(s_t, a_t)\right],
$$

where $\tau$ is a trajectory, $\gamma \in [0,1)$ is the discount factor, and $r$ is the reward. The policy gradient theorem tells us how to differentiate this:

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}\left[\nabla_\theta \log \pi_\theta(a \mid s)\, A^{\pi_\theta}(s, a)\right],
$$

where $A^{\pi_\theta}(s, a) = Q^{\pi_\theta}(s,a) - V^{\pi_\theta}(s)$ is the advantage — how much better action $a$ is than the policy's average behavior in state $s$. The vanilla update is then plain gradient ascent: $\theta \leftarrow \theta + \alpha \nabla_\theta J(\theta)$.

There are two compounding problems hiding in that one line.

The first is **variance**. The gradient is a Monte Carlo estimate over sampled trajectories, and the variance of that estimate is enormous because the advantage signal is sparse and the score function $\nabla_\theta \log \pi_\theta$ can be large. A high-variance gradient estimate means individual updates point in wildly inconsistent directions. We attack variance with baselines, GAE, and large batches — that is a separate fight, covered in the actor-critic posts. It matters here only because the variance and the step-size problem interact: a noisy gradient with an occasional large component is exactly the thing most likely to launch a too-large step into the abyss.

The second problem, the one TRPO is built to solve, is **step size in the wrong geometry**. Gradient ascent picks the step that maximizes improvement per unit of *Euclidean* distance in parameter space:

$$
\Delta\theta^\star = \arg\max_{\Delta\theta} \nabla_\theta J^\top \Delta\theta \quad \text{s.t.} \quad \|\Delta\theta\|_2 \le \epsilon.
$$

But we do not care about Euclidean distance in parameter space. We care about how much the *policy* changed, because the policy is what generates data and earns reward. The map from $\theta$ to the distribution $\pi_\theta$ is highly nonlinear and anisotropic: in some directions a small $\|\Delta\theta\|$ barely perturbs the action distribution, and in others it flips it entirely. A fixed learning rate $\alpha$ that is safe in one region is catastrophic in another. This is why tuning the learning rate for vanilla policy gradient feels like defusing a bomb: too small and it never learns, slightly too large and it detonates.

To make the anisotropy concrete, consider what "parameter space" actually contains for a neural network. Two different parameter vectors can encode the *same* function — permute the hidden units of a layer and you get an identical policy at a completely different point in $\theta$-space. Euclidean distance between those two points is large; the behavioral distance is zero. Conversely, scaling the final-layer weights of a softmax policy by a factor of ten leaves $\|\Delta\theta\|$ modest but sharpens the action distribution dramatically, moving the behavior an enormous amount. The Euclidean metric is simply blind to the structure that matters. Any method that measures step size in raw parameter units is measuring the wrong thing, and the only reason it ever works is that with a small enough learning rate every metric agrees locally. TRPO's contribution is to stop relying on "small enough" and measure the right quantity directly.

#### Worked example: how one step erases progress

Concretely, picture a Gaussian policy for a 1-D action: $\pi_\theta(a \mid s) = \mathcal{N}(\mu_\theta(s), \sigma^2)$ with $\sigma = 0.5$. Suppose at some state the current mean is $\mu = 0.0$ and the advantage estimate strongly favors larger actions, so the gradient pushes $\mu$ upward. With a learning rate that *was* fine yesterday, this update moves $\mu$ to $3.0$. The KL divergence between the old and new action distributions at that state is

$$
D_{\text{KL}}\big(\mathcal{N}(0, 0.25)\,\|\,\mathcal{N}(3, 0.25)\big) = \frac{(3-0)^2}{2 \cdot 0.25} = 18.0.
$$

A KL of 18 nats means the new policy assigns essentially zero probability to the actions the old policy was taking. Every transition in your replay batch was collected under the old policy. The importance weights $\pi_{\text{new}}/\pi_{\text{old}}$ for those transitions are now astronomically large or tiny — the surrogate objective evaluated on this data is meaningless, and the policy you just produced has never been tested. It collects a fresh batch, the returns are garbage, and there is no gradient back. That single uncontrolled step — a KL of 18 where a safe step would have been a KL of 0.01 — is the cliff. TRPO's entire job is to make that KL of 18 impossible.

Put a number on "essentially zero probability." The new policy $\mathcal{N}(3, 0.25)$ evaluated at $a = 0$ — a typical action under the old policy — has density proportional to $\exp(-(0-3)^2 / (2 \cdot 0.25)) = \exp(-18) \approx 1.5 \times 10^{-8}$ relative to its peak. The old policy's typical actions are now eight orders of magnitude down the tail of the new one. The importance ratio for such a sample is on the order of $10^{8}$ or $10^{-8}$ depending on which way it falls, and a handful of $10^{8}$-weighted samples will completely dominate a batch average that is supposed to estimate a quantity of order one. The estimator has not just become noisy; it has become a lottery decided by whichever one or two samples happened to land in the overlap of the two distributions. *That* is why the surrogate becomes a fantasy — not because the math is wrong, but because the finite-sample estimate of a ratio between nearly disjoint distributions is meaningless.

It is worth pausing on *why* there is no gradient back, because this is the asymmetry that makes the failure permanent rather than recoverable. In supervised learning a bad gradient step is self-correcting: the loss on the *fixed* dataset goes up, the next gradient points back down, and you recover. The dataset does not change when your model changes. In reinforcement learning the "dataset" is the distribution of states and actions the policy visits, and that distribution *is generated by the policy itself*. When the policy collapses into a degenerate mode — always outputting the same saturated action, say — every subsequent rollout consists of that one behavior repeated. The advantage estimates over a single repeated behavior carry almost no signal about which alternative actions would be better, because no alternative actions are ever sampled. The agent has stopped exploring and started staring at a wall. The gradient is not wrong; it is empty. This is the deep reason RL training is so much less forgiving than supervised training, and it is the reason a method that *prevents* the bad step is worth far more than a method that tries to recover from it.

A second subtlety: the brittleness is not uniform across training. Early on, when the policy is nearly uniform and high-entropy, large parameter steps produce only modest changes in the action distribution — the policy is in a flat region of the KL landscape. As training progresses and the policy sharpens (entropy drops, the action distribution concentrates), the *same* parameter step produces a much larger KL, because a sharp distribution is far more sensitive to a shift in its mean or logits. So a learning rate that was perfectly safe at update 10 becomes lethal at update 300 — which is exactly the pattern in the collapse I described in the opening, and exactly why a *fixed* step size in parameter space cannot be safe throughout training. The trust region adapts automatically, because it constrains the KL directly rather than the parameter norm.

You can see the entropy effect quantitatively in the Gaussian case. The KL for a mean shift $\Delta\mu$ at variance $\sigma^2$ is $(\Delta\mu)^2/(2\sigma^2)$. Early in training the policy is wide, $\sigma = 1.0$, and a mean shift of $0.3$ costs $0.045$ nats — a gentle change. Late in training the policy has sharpened to $\sigma = 0.1$ because it has learned a confident gait, and the *same* mean shift of $0.3$ now costs $0.3^2 / (2 \cdot 0.01) = 4.5$ nats — a hundred times larger, deep into cliff territory. The optimizer producing that $0.3$ shift has no idea the variance has changed underneath it. A fixed learning rate is, in effect, a bet that the policy's sensitivity stays constant across millions of updates, and that bet loses. The trust region simply refuses to take the second step, automatically scaling the parameter move down by the same factor of a hundred so the *behavioral* move stays fixed at $\delta$.

## 2. The monotonic improvement theorem

The theoretical heart of TRPO is a result that bounds the true return of a new policy in terms of a quantity we *can* estimate from old-policy data. It is worth deriving the intuition because everything else follows from it.

Start with the identity (Kakade and Langford, 2002) that relates the return of a new policy $\tilde\pi$ to the return of an old policy $\pi$:

$$
J(\tilde\pi) = J(\pi) + \mathbb{E}_{\tau \sim \tilde\pi}\left[\sum_{t=0}^{\infty} \gamma^t A^{\pi}(s_t, a_t)\right].
$$

Read this carefully: the improvement of $\tilde\pi$ over $\pi$ is exactly the expected advantage of $\pi$, accumulated along trajectories *drawn from* $\tilde\pi$. This is exact, but it is useless as written, because to evaluate the expectation we would need to roll out the new policy $\tilde\pi$ — the very thing we have not committed to yet.

The identity itself deserves a moment, because it is not obvious why the *old* policy's advantage should measure the *new* policy's improvement. The intuition is a telescoping argument. Walk along a trajectory generated by $\tilde\pi$. At each state, the new policy takes some action; the quantity $A^\pi(s_t, a_t)$ asks "how much better or worse was that action than what the old policy's value function expected from this state?" Summing those per-step surprises along the whole trajectory, discounted, telescopes: all the intermediate value terms cancel, and what survives is exactly the difference in total return between following $\tilde\pi$ and following $\pi$ from the start. The new policy improves precisely to the extent that, at the states it actually visits, it consistently picks actions the old value function underrated. That is a clean and complete characterization of improvement — its only flaw is the "states it actually visits" clause, which forces the expectation under $\tilde\pi$.

The trick is to approximate the expectation over $\tilde\pi$'s trajectories with an expectation over $\pi$'s state distribution. Define the **surrogate objective**:

$$
L_\pi(\tilde\pi) = J(\pi) + \mathbb{E}_{s \sim \rho_\pi,\, a \sim \tilde\pi}\left[A^{\pi}(s, a)\right],
$$

where $\rho_\pi$ is the discounted state-visitation distribution of the *old* policy. The only difference from the exact identity is that we use $\pi$'s state distribution instead of $\tilde\pi$'s. This is something we can estimate from old-policy rollouts. The surrogate matches the true objective to first order: $L_\pi(\pi) = J(\pi)$ and $\nabla_\theta L_\pi(\tilde\pi)\big|_{\tilde\pi = \pi} = \nabla_\theta J(\tilde\pi)\big|_{\tilde\pi=\pi}$. So improving $L$ a little improves $J$ a little — *as long as the state distributions have not drifted too far apart.*

That first-order matching is worth dwelling on, because it is the bridge between TRPO and the ordinary policy gradient. The gradient of the surrogate at $\theta_{\text{old}}$ is *exactly* the policy gradient $\nabla_\theta J$. So an infinitesimal step that ascends the surrogate is identical to an infinitesimal step of vanilla policy gradient. The two methods agree on direction in the limit of zero step size; they differ entirely on how far to go. Vanilla PG trusts the first-order model out to some Euclidean radius set by a learning rate; TRPO trusts it out to a KL radius set by the theorem below. Everything that distinguishes the two algorithms lives in the answer to "how far," not "which way."

The genius of the 2015 paper is to make "not too far apart" rigorous with a bound. The result, in the form TRPO uses, is:

$$
J(\tilde\pi) \ge L_\pi(\tilde\pi) - C \cdot D_{\text{KL}}^{\max}(\pi, \tilde\pi), \qquad C = \frac{4\epsilon\gamma}{(1-\gamma)^2},
$$

where $\epsilon = \max_{s,a}|A^\pi(s,a)|$ and $D_{\text{KL}}^{\max}$ is the largest KL divergence between the policies over all states. (The original statement uses the total-variation distance and Pinsker's inequality to convert to KL; the practical version above is the one that drives the algorithm.)

This inequality is the whole game. It says: the *true* return of the new policy is at least the surrogate value minus a penalty proportional to how far the policy moved in KL. So if we maximize the right-hand side, we are maximizing a guaranteed lower bound on the true return. And here is the monotonic guarantee: let $M_i(\tilde\pi) = L_{\pi_i}(\tilde\pi) - C \cdot D_{\text{KL}}^{\max}(\pi_i, \tilde\pi)$. Then

$$
J(\pi_{i+1}) \ge M_i(\pi_{i+1}) \quad \text{and} \quad J(\pi_i) = M_i(\pi_i).
$$

Subtracting, $J(\pi_{i+1}) - J(\pi_i) \ge M_i(\pi_{i+1}) - M_i(\pi_i)$. If at each step we choose $\pi_{i+1}$ to maximize $M_i$, then $M_i(\pi_{i+1}) \ge M_i(\pi_i)$, so $J(\pi_{i+1}) \ge J(\pi_i)$. **The true return never decreases.** That is the monotonic improvement guarantee, and it is the reason the whole apparatus is worth building.

Notice the structure of that argument, because it is a pattern you will meet again in optimization: it is a **minorize-maximize** (MM) scheme, the same family as the EM algorithm. $M_i$ is a function that touches the true objective $J$ at the current point ($M_i(\pi_i) = J(\pi_i)$) and lies *below* it everywhere else ($J \ge M_i$). Maximize the lower bound, and because you started on the true curve, wherever the lower bound goes up the true curve must be at least as high. You never have to evaluate $J$ directly; you only ever optimize the surrogate $M_i$, rebuild a new surrogate at the new point, and repeat. The picture is a sequence of bowls, each tangent to the true objective from below, each maximized in turn, the iterates marching monotonically uphill on the true curve they never directly see. TRPO is conservative policy iteration recast as an MM algorithm — that is the cleanest one-sentence summary of the theory.

![The TRPO surrogate is an importance-ratio-weighted advantage gated by a hard KL trust-region constraint between the old and new policies](/imgs/blogs/trust-region-policy-optimization-trpo-2.png)

Intuitively: the penalty $C \cdot D_{\text{KL}}$ is the price you pay for trusting old data to predict new behavior. Move a little, pay a little, and the surrogate is a faithful guide. Move a lot, pay a lot, and the surrogate's promise evaporates. The theorem turns "don't move too far" from a hand-wave into an inequality with a constant.

It helps to see where the penalty term comes from, at least in sketch, because it demystifies the otherwise-magic constant. The gap between the surrogate $L_\pi(\tilde\pi)$ and the true objective $J(\tilde\pi)$ is entirely due to the mismatch between the two policies' state-visitation distributions, $\rho_\pi$ versus $\rho_{\tilde\pi}$. If the two policies are close, those state distributions are close, and the surrogate is accurate. The original derivation bounds the state-distribution mismatch by the total-variation distance between the policies — the probability mass that the two action distributions disagree on — and then uses the fact that a single step of disagreement compounds geometrically over a trajectory, which is where the $(1-\gamma)^{-1}$ factors come from (a $\gamma$-discounted infinite horizon has effective length $1/(1-\gamma)$, and the mismatch can accumulate at each step). The total-variation distance is then converted to KL via Pinsker's inequality, $D_{\text{TV}}(p, q) \le \sqrt{\tfrac{1}{2} D_{\text{KL}}(p \,\|\, q)}$, which is why the bound ends up in terms of KL. The $\epsilon = \max|A^\pi|$ factor is the worst-case magnitude of the advantage — the most a single mismatched action can cost. Put together: penalty = (worst advantage) × (how badly the policies disagree) × (how long that disagreement compounds over the horizon). Every factor in $C = 4\epsilon\gamma/(1-\gamma)^2$ has a plain meaning.

#### Worked example: the penalty constant is brutally large

It pays to compute $C$ for a realistic setting, because the number explains every engineering decision that follows. Take $\gamma = 0.99$, a standard continuous-control discount, and suppose the advantages have been normalized so that $\epsilon = \max|A^\pi| \approx 1$. Then

$$
C = \frac{4 \cdot 1 \cdot 0.99}{(1 - 0.99)^2} = \frac{3.96}{0.0001} = 39{,}600.
$$

The penalty on a KL of even $0.01$ nats is therefore $39{,}600 \times 0.01 = 396$ — utterly dwarfing any plausible surrogate improvement, which for a single update is of order $0.01$ to $0.1$. If you maximized $L - C \cdot D_{\text{KL}}$ literally, the penalty term would forbid essentially any move at all; the optimizer would conclude the safest policy is the one it already has and take a step of size near zero. The theorem is *true* but the constant is so conservative that the algorithm it prescribes does not learn in any reasonable time. This single calculation is why Section 3 exists: TRPO keeps the *shape* of the bound (KL controls trustworthiness) but throws away the literal constant, replacing the soft penalty with a hard constraint whose radius is chosen empirically rather than from the worst-case theory.

## 3. From penalty to trust region

The lower-bound version suggests a penalized objective: maximize $L_\pi(\tilde\pi) - C \cdot D_{\text{KL}}^{\max}$. In principle you could just do that — pick the penalty coefficient $C$ from the formula and run gradient ascent on the penalized surrogate. In practice $C = 4\epsilon\gamma/(1-\gamma)^2$ is brutally conservative, as the worked example just showed. With $\gamma = 0.99$, the factor $(1-\gamma)^{-2} = 10{,}000$, so the penalty dwarfs the surrogate and the steps become microscopic. Training would take forever.

TRPO makes the engineering choice that defines it: instead of a *penalty* on KL, impose a *constraint* on KL. Replace the soft penalty with a hard trust-region radius $\delta$:

$$
\max_\theta \; \mathbb{E}_{s \sim \rho_{\theta_{\text{old}}},\, a \sim \pi_{\theta_{\text{old}}}}\left[\frac{\pi_\theta(a \mid s)}{\pi_{\theta_{\text{old}}}(a \mid s)}\, A^{\pi_{\theta_{\text{old}}}}(s, a)\right] \quad \text{s.t.} \quad \bar{D}_{\text{KL}}(\theta_{\text{old}}, \theta) \le \delta.
$$

Two changes from the theory are worth flagging. First, the surrogate is written with an **importance-sampling ratio** $r(\theta) = \pi_\theta / \pi_{\theta_{\text{old}}}$ instead of "sample $a$ from $\tilde\pi$" — these are equivalent in expectation, but the ratio form lets us evaluate the objective and its gradient entirely on samples drawn from the *old* policy. This is what makes the update computable from a single batch of rollouts. The equivalence is the standard importance-sampling identity: $\mathbb{E}_{a \sim \tilde\pi}[A^\pi(s,a)] = \mathbb{E}_{a \sim \pi}[\tfrac{\tilde\pi(a|s)}{\pi(a|s)} A^\pi(s,a)]$. We collected the batch under $\pi_{\text{old}}$, so the right-hand form is the one we can actually compute — every action in the buffer was sampled from $\pi_{\text{old}}$, and we reweight it by how much more or less likely the candidate policy $\pi_\theta$ would have made it. Second, TRPO uses the **average** KL $\bar{D}_{\text{KL}}$ over states rather than the max KL $D_{\text{KL}}^{\max}$, because the max over a continuous state space is intractable and the average is a fine heuristic that works in practice. A typical value is $\delta = 0.01$ nats.

This constrained form has a clean interpretation, which the figure above captures: we are searching for the best policy *within a ball of radius $\delta$* around the current policy, where distance is measured in KL. Inside that ball, the surrogate is trustworthy; outside it, we refuse to go. The constraint is the trust region.

Why is a constraint better than a penalty when they are formally interchangeable (every constrained problem has an equivalent Lagrangian penalty)? Because the *right* penalty coefficient changes as training progresses, and a constraint sidesteps that. Early in training the advantages are large and noisy, so the theoretically-correct $C$ is huge; late in training the advantages shrink toward zero as the policy nears optimal, so $C$ should shrink too. A fixed penalty coefficient is wrong at almost every point in training — too timid early, too aggressive late. A fixed *KL radius* $\delta$, by contrast, is a stable, interpretable quantity: it says "change the policy by at most this much per step," and that instruction is sensible throughout training regardless of the advantage scale. This is the same reason practitioners prefer to specify "clip the gradient norm to 1.0" rather than "multiply the gradient by 0.001" — constraining the *outcome* is more robust than constraining the *coefficient*. TRPO's $\delta$ is a trust-region radius in exactly that spirit.

There is also a worked intuition for the magnitude of $\delta$. A KL of $0.01$ nats between two Gaussians of equal variance $\sigma^2$ corresponds, from the formula $D_{\text{KL}} = (\Delta\mu)^2/(2\sigma^2)$, to a mean shift of $\Delta\mu = \sqrt{2 \cdot 0.01}\,\sigma \approx 0.14\,\sigma$ — about a seventh of a standard deviation. That is the scale of behavioral change TRPO permits per update: small enough that the old data still describes the new policy well, large enough that thousands of such steps compound into a dramatically different (and better) policy. Contrast the disastrous KL of 18 from the worked example in Section 1, which corresponds to a mean shift of six standard deviations — utterly outside the trust region. The radius $\delta$ is the dial between these two regimes.

It is also instructive to relate $\delta$ to the more familiar notion of an effective learning rate. The natural-gradient step length, derived in the next section, is $\sqrt{2\delta / (g^\top F^{-1} g)}$. The KL budget $\delta$ appears under a square root, so halving the trust region only shrinks the step by a factor of $\sqrt{2} \approx 1.4$, not by half — the relationship between trust-region size and step size is sublinear. This is reassuring in practice: TRPO is not hypersensitive to the exact value of $\delta$, and the common range $0.005$ to $0.02$ spans only a $2\times$ variation in effective step length. Compare that to a raw learning rate, where a $4\times$ change routinely separates "too slow to converge" from "diverges," and you see another reason the trust-region parameterization is more forgiving than a learning rate.

The question now is purely computational: how do we solve a constrained optimization problem where the objective is the expected importance-weighted advantage and the constraint is an average KL divergence, both estimated from samples, with $\theta$ being millions of neural-network weights? The answer is a beautiful piece of second-order optimization, and it starts with the natural gradient.

## 4. The natural gradient

Here is the key reframing. We want to maximize the surrogate subject to $\bar{D}_{\text{KL}} \le \delta$. Both the objective and the constraint can be approximated near $\theta_{\text{old}}$ by Taylor expansion. The surrogate $L(\theta)$ is approximately linear: $L(\theta) \approx g^\top (\theta - \theta_{\text{old}})$, where $g = \nabla_\theta L$ is the policy gradient (the constant term and zeroth-order surrogate value drop out of the argmax). The KL constraint is approximately *quadratic*, because the KL divergence between a distribution and itself is zero with zero gradient, so its leading term is second order:

$$
\bar{D}_{\text{KL}}(\theta_{\text{old}}, \theta) \approx \frac{1}{2}(\theta - \theta_{\text{old}})^\top F (\theta - \theta_{\text{old}}),
$$

where $F$ is the **Fisher information matrix**, the Hessian of the KL at $\theta_{\text{old}}$:

$$
F = \mathbb{E}_{s \sim \rho,\, a \sim \pi}\left[\nabla_\theta \log \pi_\theta(a\mid s)\, \nabla_\theta \log \pi_\theta(a \mid s)^\top\right].
$$

The reason the KL has no linear term is worth stating plainly, because it is the structural fact that makes the whole second-order story work. $\bar{D}_{\text{KL}}(\theta_{\text{old}}, \theta)$ as a function of $\theta$ achieves its global minimum of exactly zero at $\theta = \theta_{\text{old}}$ — a distribution has zero KL from itself, and KL is non-negative everywhere. A smooth function at its minimum has zero gradient, so the first-order term vanishes identically and the leading behavior is the quadratic form set by the Hessian. This is why the constraint is a clean ellipsoid (the Fisher quadratic form) to leading order, with no linear tilt: the trust region is a centered ellipsoid around $\theta_{\text{old}}$, and the only question is its shape, which $F$ supplies.

So the local problem is: maximize $g^\top \Delta\theta$ subject to $\frac{1}{2}\Delta\theta^\top F \Delta\theta \le \delta$. This is a quadratically-constrained linear program, and it has a closed-form solution via Lagrange multipliers. Form the Lagrangian $g^\top \Delta\theta - \lambda(\tfrac12 \Delta\theta^\top F \Delta\theta - \delta)$, set its gradient with respect to $\Delta\theta$ to zero, and you get $g - \lambda F \Delta\theta = 0$, i.e. $\Delta\theta = \tfrac{1}{\lambda} F^{-1} g$. The optimal direction is therefore

$$
\Delta\theta \propto F^{-1} g.
$$

That direction, $F^{-1}g$, is the **natural gradient**. Compare it to the Euclidean gradient $g$. The Euclidean gradient is steepest ascent under the assumption that distance in parameter space is what matters. The natural gradient is steepest ascent under the assumption that distance in *policy distribution space* — measured by the Fisher metric, which is the local quadratic form of the KL — is what matters. Since reward depends on the policy's behavior, not its raw weights, the Fisher metric is the geometrically correct one. The Fisher matrix reweights the gradient: directions in which the policy is very sensitive to $\theta$ (large Fisher eigenvalues) get shrunk, and insensitive directions get amplified, so that every step changes the policy by the same KL amount.

A crucial property of the natural gradient, and one of the original motivations for it (Amari, Kakade), is **invariance to reparameterization**. If you change how the policy is parameterized — rescale a layer, switch from logits to log-probabilities, reorder units — the Euclidean gradient points in a different direction, but the natural gradient points to the same *distribution*. It chases the behavior, not the coordinates. This is exactly the property the brittleness discussion in Section 1 demanded: a method whose step is blind to behaviorally-irrelevant parameter structure. The Fisher metric is, in a precise sense, the unique Riemannian metric on the space of probability distributions that is invariant under reparameterization (Čencov's theorem), which is why "the natural gradient" is *the* natural one and not merely *a* sensible preconditioner.

![The Euclidean gradient steps in parameter space while the natural gradient preconditions by the Fisher matrix to step in policy distribution space at constant KL](/imgs/blogs/trust-region-policy-optimization-trpo-6.png)

A short but important aside on the Fisher matrix itself, because the definition above hides a small miracle. The Fisher information matrix can be defined two ways: as the expected outer product of score functions (the formula above), or as the expected *negative Hessian* of the log-likelihood. For probability distributions these two definitions coincide — that is a standard identity — and they also equal the Hessian of the KL divergence evaluated at the point of zero divergence. This triple coincidence is what lets us compute the Fisher matrix as the Hessian of the average KL (which is what the code in Section 5 does) rather than as an outer product of scores. Practically it means we never have to think about scores at all: we write down the average KL between the frozen old policy and the live policy, and autograd's second derivative *is* the Fisher matrix. The KL formulation is also numerically better behaved than the outer-product one, because it is naturally curvature-aware and the damping term has a clean interpretation as a trust-region prior.

It is worth being explicit that the Fisher matrix is *not* the Hessian of the surrogate objective $L$, and confusing the two is a common conceptual error. The Hessian of $L$ would give you Newton's method on the surrogate — a second-order method that tries to find where the surrogate's *gradient* is zero. That is not what we want; the surrogate is only a local linear model and its curvature is noise. The Fisher matrix is the Hessian of the *constraint*, the KL, and it encodes the geometry of the policy manifold, not the curvature of the reward landscape. TRPO is therefore a first-order method in the objective (it uses only $g$) and a second-order method in the *metric* (it uses $F$ to define distance). That hybrid is exactly the natural gradient: steepest ascent of a linear objective under a curved notion of distance.

With the direction $F^{-1}g$ fixed, the trust-region radius sets the step length. Solving the Lagrangian gives the maximal step that saturates the KL constraint:

$$
\Delta\theta = \sqrt{\frac{2\delta}{g^\top F^{-1} g}}\; F^{-1} g.
$$

The scalar in front is exactly the step size that makes the quadratic KL approximation equal $\delta$. To see it, substitute $\Delta\theta = \beta F^{-1} g$ into the constraint $\tfrac12 \Delta\theta^\top F \Delta\theta = \delta$: you get $\tfrac12 \beta^2 g^\top F^{-1} F F^{-1} g = \tfrac12 \beta^2 g^\top F^{-1} g = \delta$, so $\beta = \sqrt{2\delta / (g^\top F^{-1} g)}$. This is the natural policy gradient (NPG) update — and TRPO is NPG plus two engineering refinements (efficient computation of $F^{-1}g$, and a line search to handle the approximation error), which we build next.

#### Worked example: the natural step rescales by sensitivity

Suppose, in a two-parameter toy policy, the raw gradient is $g = [1, 1]^\top$ and the Fisher matrix is diagonal, $F = \text{diag}(100, 1)$ — meaning parameter 1 moves the policy distribution 100× more per unit than parameter 2. Euclidean gradient ascent would step equally in both, $\Delta\theta \propto [1,1]$, and the large move in the sensitive parameter 1 blows the KL budget. The natural gradient is $F^{-1}g = [0.01, 1]^\top$: it steps a hundred times *less* in the sensitive direction and full speed in the safe one. Plugging into the step formula with $\delta = 0.01$: $g^\top F^{-1} g = 1 \cdot 0.01 + 1 \cdot 1 = 1.01$, so the scale is $\sqrt{2 \cdot 0.01 / 1.01} \approx 0.141$, and $\Delta\theta \approx [0.0014, 0.141]$. Both components produce the same contribution to KL. That automatic rescaling — large in safe directions, tiny in dangerous ones — is what a single global learning rate can never achieve, and it is the whole reason the natural gradient does not fall off cliffs.

Push the example one step further to feel the contrast with a learning rate. Suppose you tried to reproduce the natural step with plain gradient ascent and a single scalar $\alpha$, choosing $\alpha$ so the *safe* direction (parameter 2) moves the right amount, $0.141$. Then $\alpha = 0.141$, and the *dangerous* direction would also move $0.141$ — a hundred times farther than the natural gradient permits, contributing $\tfrac12 \cdot 100 \cdot 0.141^2 \approx 0.99$ nats of KL from that one coordinate alone, ninety-nine times over budget. Conversely, choose $\alpha$ small enough to keep the dangerous direction safe, $\alpha = 0.0014$, and the safe direction crawls at $0.0014$ — a hundred times slower than it could. *No single $\alpha$ is right for both coordinates.* This is the precise sense in which a scalar learning rate is the wrong object: the policy's sensitivity is a matrix, and only a matrix preconditioner ($F^{-1}$) can equalize the behavioral step across directions. The natural gradient is that preconditioner, derived rather than tuned.

## 5. Computing the natural gradient without forming F

The natural gradient needs $F^{-1}g$, and there is the rub: $F$ is a $d \times d$ matrix where $d$ is the number of policy parameters — easily millions. Forming $F$ costs $O(d^2)$ memory and inverting it costs $O(d^3)$. For a network with a million parameters that is a $10^{12}$-entry matrix and a $10^{18}$-operation inverse. Completely infeasible. TRPO never forms $F$.

The escape is **conjugate gradient (CG)**. CG solves the linear system $F x = g$ for $x = F^{-1}g$ iteratively, and crucially it only ever needs to compute *matrix-vector products* $F v$, never $F$ itself. Each CG iteration drives the residual down, and for a well-conditioned system it converges to a good approximation in ten to twenty iterations regardless of $d$. So the question reduces to: how do we compute $F v$ for an arbitrary vector $v$ without materializing $F$?

A word on *why* CG, specifically, and not some other iterative solver. Conjugate gradient is the method of choice for solving $Fx = g$ when $F$ is symmetric positive-definite, which the (damped) Fisher matrix is. Its key property is that it does not just descend the residual greedily; it builds a sequence of mutually $F$-conjugate search directions, so that progress made in one direction is never undone by a later one. In exact arithmetic CG converges to the true solution in at most $d$ steps, but far more usefully, it converges to a *good* approximation in a number of steps governed by the *number of distinct eigenvalue clusters* of $F$, not by $d$. Fisher matrices in deep RL tend to have a few large eigenvalues (the sensitive directions) and a long tail of small ones; CG knocks out the large-eigenvalue components first, which are exactly the dangerous directions we most need to handle correctly. Ten iterations typically captures the action, which is why `cg_max_steps` of 10 to 15 is standard.

The trick to the matrix-vector product is the **Hessian-vector product**. Recall $F$ is the Hessian of the average KL with respect to $\theta$. For any function $\phi(\theta)$ with Hessian $H$, the product $Hv$ equals the gradient of the directional derivative: $Hv = \nabla_\theta\big((\nabla_\theta \phi)^\top v\big)$. So we compute $Fv$ in two backward passes: first take the gradient of the KL, dot it with $v$ (a scalar), then take the gradient of *that* scalar. Two autograd calls, $O(d)$ memory, no matrix.

Why is this exact and not an approximation? The inner expression $(\nabla_\theta \phi)^\top v$ is a scalar function of $\theta$ (with $v$ held constant). Its gradient is $\nabla_\theta[(\nabla_\theta\phi)^\top v] = (\nabla_\theta^2 \phi)\, v = Hv$ by the chain rule — the Jacobian of the gradient is the Hessian. Autograd computes this to machine precision; there is no finite-difference approximation and no truncation. The "two backward passes" are literally a backward pass to get $\nabla\phi$ (kept in the graph via `create_graph=True`), a cheap dot product with $v$, and a second backward pass through that scalar. The cost is roughly twice a normal gradient, independent of $d$, which is what makes CG affordable: ten iterations cost about twenty gradient-equivalents.

![Conjugate gradient solves for the natural direction using only Hessian-vector products, then a line search bounds the final step by the KL trust region](/imgs/blogs/trust-region-policy-optimization-trpo-8.png)

Here is the Fisher-vector product and conjugate-gradient solver in PyTorch. This is the computational core of TRPO.

```python
import torch

def flat_grad(y, params, retain_graph=False, create_graph=False):
    grads = torch.autograd.grad(
        y, params, retain_graph=retain_graph, create_graph=create_graph,
        allow_unused=True,
    )
    grads = [g if g is not None else torch.zeros_like(p)
             for g, p in zip(grads, params)]
    return torch.cat([g.reshape(-1) for g in grads])

def fisher_vector_product(policy, states, vector, damping=0.1):
    # Average KL between the fixed old policy (detached) and current policy.
    kl = policy.mean_kl(states)            # scalar tensor, KL(old || new), = 0 at theta_old
    grad_kl = flat_grad(kl, policy.parameters(), create_graph=True)
    # Directional derivative grad_kl . vector, then differentiate -> Hessian-vector product.
    grad_vector_dot = (grad_kl * vector).sum()
    fvp = flat_grad(grad_vector_dot, policy.parameters(), retain_graph=True)
    # Damping (Tikhonov) keeps F positive-definite and CG well-conditioned.
    return fvp + damping * vector

def conjugate_gradient(Avp_fn, b, n_iters=10, tol=1e-10):
    x = torch.zeros_like(b)
    r = b.clone()          # residual = b - A x, with x = 0
    p = b.clone()          # search direction
    r_dot = torch.dot(r, r)
    for _ in range(n_iters):
        Ap = Avp_fn(p)
        alpha = r_dot / (torch.dot(p, Ap) + 1e-12)
        x += alpha * p
        r -= alpha * Ap
        new_r_dot = torch.dot(r, r)
        if new_r_dot < tol:
            break
        beta = new_r_dot / r_dot
        p = r + beta * p
        r_dot = new_r_dot
    return x
```

Read the CG loop once more with the conjugacy in mind. The vector `r` is the residual $b - Fx$, how far the current $x$ is from solving $Fx = b$. The vector `p` is the current search direction. The step length `alpha` is chosen to minimize the $F$-norm of the error along `p`; `beta` then builds the next direction to be $F$-conjugate to all previous ones, $p_{k+1}^\top F p_j = 0$ for $j \le k$. That conjugacy is the magic: it guarantees the components of the error along already-searched directions stay zero, so each iteration makes monotone progress and the whole thing terminates in at most $d$ steps exactly (and far fewer approximately). The only place $F$ enters is `Avp_fn(p)` — a single Fisher-vector product per iteration. We solve a million-dimensional linear system having never written down a single entry of the million-by-million matrix.

Two implementation notes that cost people days. First, the **damping** term `damping * vector` adds $\lambda I$ to $F$. The Fisher estimate from a finite batch is often near-singular, and a singular $F$ makes CG diverge; the damping (a Tikhonov regularizer) keeps it positive-definite. Typical $\lambda = 0.1$. There is a tradeoff in $\lambda$: too small and CG diverges or returns a noisy direction dominated by the near-null space of the sampled Fisher; too large and the damping term dominates, $F + \lambda I \approx \lambda I$, and the "natural" gradient degenerates back toward the plain gradient $g/\lambda$ — you lose the preconditioning that was the whole point. The sweet spot $\lambda \in [0.01, 0.1]$ is wide because of the square-root step formula's forgivingness, but it is not optional. Second, `create_graph=True` on the first gradient is mandatory — without it the second autograd call has nothing to differentiate through and you get zeros, silently. The `mean_kl` method must compute KL between a *detached* snapshot of the old policy and the live one, so that at $\theta_{\text{old}}$ the KL is zero and its Hessian is exactly the Fisher matrix.

A third, subtler note: the states fed to `fisher_vector_product` should be the *same* batch used to compute the gradient $g$, because $F$ and $g$ are both expectations over the old policy's state-visitation distribution, and they must be consistent for the natural direction $F^{-1}g$ to mean what the theory says. Some implementations subsample the batch for the Fisher-vector products to save compute (the Fisher is a smoother object than the gradient and tolerates a smaller sample), which is a legitimate optimization, but the subsample must still be drawn from the same rollout. Mixing batches here is a silent correctness bug that merely slows learning rather than crashing — the worst kind, as the limitations section will lament.

## 6. The backtracking line search

Conjugate gradient gives us a step *direction* $x \approx F^{-1}g$, and the formula from Section 4 gives a *length* that should saturate the KL budget under the quadratic approximation. But the quadratic approximation of the KL is only local. Take the full step and you may find the actual KL exceeds $\delta$, or the surrogate actually went *down* — the approximations lied. TRPO does not trust them blindly. It does a **backtracking line search**.

Compute the proposed full step $\Delta\theta = \sqrt{2\delta / (x^\top F x)}\; x$. Then try step fractions $\beta^j \Delta\theta$ for $\beta \in (0,1)$, $j = 0, 1, 2, \dots$ Accept the first $j$ for which two conditions hold on the *actual* (not approximated) quantities: the average KL is within budget, $\bar{D}_{\text{KL}} \le \delta$, and the surrogate improved. If no fraction satisfies both within, say, ten halvings, take no step at all this iteration. This last fallback is what guarantees TRPO never makes things worse — in the limit it reproduces the monotonic improvement theorem's promise, because a rejected step is a step of size zero.

The reason the quadratic approximation lies is concrete and worth naming. The true KL as a function of step length along the direction $x$ is a smooth curve that equals zero at the origin and curves upward; the quadratic approximation is the parabola tangent to it there. For small steps the two agree, but the true KL almost always grows *faster* than the parabola once you leave the immediate neighborhood, because the Fisher matrix is the curvature *at the start point* and the curvature typically increases as the policy sharpens along the step. So the analytically-computed full step, calibrated to hit $\delta$ under the parabola, routinely overshoots the true KL. The line search is the correction: it walks the step length down by halving until the *measured* KL is back under budget. Empirically the full step is accepted maybe half the time on easy problems and rarely on hard ones; one or two backtracks is the common case, and that is the algorithm working as designed, not a sign of trouble.

```python
def line_search(policy, states, advantages, old_log_probs, full_step,
                expected_improve, max_backtracks=10, accept_ratio=0.1):
    old_params = policy.get_flat_params()
    old_surrogate = surrogate_loss(policy, states, advantages, old_log_probs)
    for j in range(max_backtracks):
        step_frac = 0.5 ** j
        new_params = old_params + step_frac * full_step
        policy.set_flat_params(new_params)
        new_surrogate = surrogate_loss(policy, states, advantages, old_log_probs)
        kl = policy.mean_kl(states).item()
        actual_improve = new_surrogate - old_surrogate
        # Require: KL within trust region AND a real (sufficient) improvement.
        if kl <= TRPO_DELTA and actual_improve > 0 and \
           actual_improve / (step_frac * expected_improve + 1e-12) > accept_ratio:
            return True          # accept this step
    policy.set_flat_params(old_params)   # reject: revert, take no step
    return False

def surrogate_loss(policy, states, advantages, old_log_probs):
    log_probs = policy.log_prob(states)            # log pi_theta(a|s) for stored actions
    ratio = torch.exp(log_probs - old_log_probs)   # importance ratio r(theta)
    return (ratio * advantages).mean()
```

The third condition — `actual_improve / (step_frac * expected_improve) > accept_ratio` — deserves explanation, because it is more than a positivity check. `expected_improve` is the surrogate gain the linear model *predicted* for the full step, $g^\top \Delta\theta$. Scaled by `step_frac`, it is the gain the linear model predicts for the current fractional step. The ratio of *actual* to *predicted* improvement is a classic trust-region quality measure (the same $\rho$ that appears in textbook trust-region optimization): if it is close to 1, the linear model was accurate; if it is small or negative, the model badly overpredicted and we should not trust this step even if the KL happens to be within budget. Requiring the ratio to exceed `accept_ratio = 0.1` rejects steps where the surrogate barely improved relative to what was promised — a sign the step wandered into a region where the model is unreliable. It is a sufficient-decrease (Armijo-style) condition adapted to the trust-region setting.

The line search is the difference between natural policy gradient (NPG), which takes the analytic step on faith, and TRPO, which verifies it. In high dimensions with neural networks the quadratic approximation is routinely off enough that the unverified NPG step violates the KL constraint or hurts the surrogate; the line search is what makes TRPO robust in practice rather than merely elegant in theory. NPG without a line search will, on hard problems, occasionally take a step whose true KL is several times $\delta$ — and a single such step can begin the drift toward collapse that the whole method exists to prevent. The line search closes that gap at the cost of a handful of cheap forward passes (no backward passes — the surrogate and KL evaluations are forward-only), which is the best bargain in the algorithm.

#### Worked example: a line search that rejects three times

Walk through a single update on a CartPole agent partway through training. Conjugate gradient returns a direction, and the scaling formula proposes a full step with expected surrogate improvement $0.04$. The line search tries $\beta^0 = 1.0$: the actual average KL comes out to $0.031$ — over the $\delta = 0.01$ budget, because the quadratic approximation underestimated the curvature in this region. Reject. Try $\beta^1 = 0.5$: KL is $0.018$, still over. Reject. Try $\beta^2 = 0.25$: KL is $0.009$, within budget, and the actual surrogate improvement is $0.012$ — positive, and the ratio of actual-to-expected improvement is $0.012 / (0.25 \cdot 0.04) = 1.2 > 0.1$, so the sufficient-improvement test passes. Accept the quarter step. The policy moved a real but bounded amount, the KL budget held, and the return ticked up. Without the line search, the full step's KL of $0.031$ would have been taken on faith — three times the budget, the start of exactly the kind of drift that compounds into a collapse. This is not a contrived scenario; on hard continuous-control tasks the first one or two backtracks are rejected on a large fraction of updates, and that rejection is the algorithm doing its job.

Notice also the near-quadratic way KL fell as the step shrank: $0.031 \to 0.018 \to 0.009$ as the fraction went $1.0 \to 0.5 \to 0.25$. Halving the step roughly *quarters* the excess over the curve's linear part — consistent with KL being quadratic in step length to leading order, $D_{\text{KL}} \approx \tfrac12 \beta^2 \cdot (\text{const})$. From $\beta = 1.0$ at KL $0.031$, the quadratic model predicts $\beta = 0.5$ gives $0.031/4 \approx 0.008$; the measured $0.018$ is higher because the true curve is super-quadratic in this region, which is *why* the analytic full step overshot in the first place. The line search does not need to know any of this — it just halves until the measurement says it is safe — but seeing the numbers confirms the mechanism.

## 7b. The running example: TRPO on the CartPole that keeps falling

Tie the machinery back to the series' running example — a CartPole agent that cannot keep the pole up. At initialization the policy is near-uniform over the two actions (push left, push right), and an episode ends after the pole tips past 12 degrees, typically within 8 to 20 steps, for a return well under 30 out of a maximum of 500. The question is whether TRPO turns this flailing into balance, and what the trust region buys over a naive baseline.

Here is a complete, runnable training loop on `CartPole-v1` using the components from the previous sections, with a small discrete-action policy. It is deliberately self-contained so you can read the full data flow in one place.

```python
import gymnasium as gym
import torch
import torch.nn as nn

class CategoricalPolicy(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, n_actions),
        )
        self._old_logits = None      # frozen snapshot for KL / Fisher

    def logits(self, obs):
        return self.net(obs)

    def dist(self, obs):
        return torch.distributions.Categorical(logits=self.logits(obs))

    def log_prob_of(self, obs, actions):
        return self.dist(obs).log_prob(actions)

    def snapshot_old(self):
        with torch.no_grad():
            self._old_logits = None  # forces recompute against detached params below

    def mean_kl(self, obs):
        # KL(old || new) averaged over states; old logits are detached.
        new_logits = self.logits(obs)
        old_logits = new_logits.detach()       # at theta_old, old == new -> KL 0, Hessian = Fisher
        old_logp = torch.log_softmax(old_logits, dim=-1)
        new_logp = torch.log_softmax(new_logits, dim=-1)
        old_p = old_logp.exp()
        return (old_p * (old_logp - new_logp)).sum(-1).mean()

    def get_flat_params(self):
        return torch.cat([p.data.reshape(-1) for p in self.parameters()])

    def set_flat_params(self, flat):
        i = 0
        for p in self.parameters():
            n = p.numel()
            p.data.copy_(flat[i:i + n].reshape(p.shape))
            i += n

def collect_rollout(env, policy, n_steps=2048):
    obs_buf, act_buf, rew_buf, done_buf, nobs_buf = [], [], [], [], []
    obs, _ = env.reset(seed=0)
    for _ in range(n_steps):
        obs_t = torch.as_tensor(obs, dtype=torch.float32)
        with torch.no_grad():
            action = policy.dist(obs_t).sample().item()
        nobs, rew, term, trunc, _ = env.step(action)
        done = term or trunc
        obs_buf.append(obs); act_buf.append(action)
        rew_buf.append(rew); done_buf.append(float(done)); nobs_buf.append(nobs)
        obs = nobs if not done else env.reset()[0]
    to_t = lambda x, d=torch.float32: torch.as_tensor(x, dtype=d)
    return (to_t(obs_buf), to_t(act_buf, torch.long), to_t(rew_buf),
            to_t(done_buf), to_t(nobs_buf))
```

A note on the `mean_kl` design that trips up nearly everyone implementing this from scratch. The detach trick — computing `old_logits = new_logits.detach()` from the *same* forward pass — looks like it must give identically zero KL, and at the exact point $\theta_{\text{old}}$ it does. But that is precisely correct and required: we want the KL and its *gradient* to be zero at $\theta_{\text{old}}$ (so the linear term vanishes, Section 4) while its *Hessian* is the Fisher matrix. Because `old_logits` is detached, autograd treats it as a constant; differentiating the KL with respect to the live parameters twice yields exactly $F$. If instead you stored a genuine snapshot of the old parameters and computed KL between two different parameter vectors, the KL would be nonzero, its gradient would be nonzero, and the Hessian-vector product would no longer equal the Fisher matrix — a subtle bug that yields a wrong search direction and an agent that learns slowly for no visible reason. The "same forward pass, detach one side" pattern is the canonical correct implementation.

Run the `trpo_step` from Section 7 in a loop over rollouts, and the learning curve has a characteristic shape worth describing because it is the visible signature of the trust region. The return climbs steadily — not explosively, because each step is KL-bounded — and, critically, it does not crash. Around 30 to 50 updates of 2048 steps each, a well-tuned TRPO agent on CartPole-v1 reaches the maximum return of 500 and holds it. The pole that fell in 8 steps now stays up for the full 500-step episode indefinitely. A vanilla policy-gradient baseline on the same task can reach 500 faster on a lucky seed but will, on a meaningful fraction of seeds, spike to 500 and then collapse back to 30 when an unlucky large step destabilizes the policy — the heartbeat-monitor curve again. The measured difference is not primarily final return (both can hit 500); it is *variance across seeds and across updates*. TRPO trades peak speed for reliability, and on CartPole you can watch that trade happen in real time.

If you do not want to implement TRPO yourself — and for production you should not — Stable-Baselines3's contrib package ships a tested implementation. The whole training run reduces to:

```python
from sb3_contrib import TRPO
import gymnasium as gym

env = gym.make("CartPole-v1")
model = TRPO(
    "MlpPolicy", env,
    target_kl=0.01,        # the trust-region radius delta
    cg_max_steps=15,       # conjugate-gradient iterations
    cg_damping=0.1,        # Fisher damping (Tikhonov)
    line_search_max_iter=10,
    gamma=0.99, gae_lambda=0.95,
    n_steps=2048,
    verbose=1,
)
model.learn(total_timesteps=200_000)
```

```bash
# Evaluate the trained policy and report mean return over 20 episodes.
python -c "
from stable_baselines3.common.evaluation import evaluate_policy
import gymnasium as gym
mean, std = evaluate_policy(model, gym.make('CartPole-v1'), n_eval_episodes=20)
print(f'mean return {mean:.1f} +/- {std:.1f}')
"
```

The `target_kl`, `cg_max_steps`, `cg_damping`, and `line_search_max_iter` arguments map one-to-one onto the machinery we built by hand. Seeing the same parameters in a maintained library is a good sanity check that you have understood what TRPO actually does: a KL target, a conjugate-gradient solve, Fisher damping, and a line search. Nothing else. When something goes wrong with a library run, this mapping is your debugging map: an exploding KL points at `target_kl` or `cg_damping`; a direction that never improves the surrogate points at `cg_max_steps` (too few iterations, a poor $F^{-1}g$); steps that always get rejected by the line search point at a Fisher estimate that is too noisy, usually a `n_steps` that is too small for the network size.

## 7. The full TRPO algorithm

Now assemble the pieces. One TRPO iteration: collect a batch of trajectories under the current policy; estimate advantages (GAE is standard); compute the policy gradient $g$; run conjugate gradient to get the natural direction $F^{-1}g$ using Fisher-vector products; scale it to the trust-region radius; backtracking-line-search to find the largest accepted fraction; update. Then separately fit the value function (the critic) by regression on the empirical returns.

![One TRPO iteration stacks rollout collection, advantage estimation, a conjugate-gradient natural step, and a KL-bounded backtracking line search before updating the policy](/imgs/blogs/trust-region-policy-optimization-trpo-3.png)

Here is the iteration tying the previous snippets together. The critic update is ordinary supervised regression and is shown briefly; the policy update is the interesting part.

```python
import torch
import torch.nn.functional as F_loss

TRPO_DELTA = 0.01      # KL trust-region radius (nats)
CG_ITERS   = 10
GAMMA, LAM = 0.99, 0.95

def trpo_step(policy, value_net, value_opt, batch):
    states, actions, rewards, dones, next_states = batch

    # --- advantages via GAE, using the critic ---
    with torch.no_grad():
        values      = value_net(states).squeeze(-1)
        next_values = value_net(next_states).squeeze(-1)
    advantages, returns = compute_gae(rewards, values, next_values, dones,
                                      GAMMA, LAM)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # --- store old log-probs and detach the old-policy snapshot for KL ---
    with torch.no_grad():
        old_log_probs = policy.log_prob_of(states, actions)
    policy.snapshot_old()      # freeze pi_old for the KL / Fisher computations

    # --- policy gradient g of the surrogate at theta_old ---
    log_probs = policy.log_prob_of(states, actions)
    ratio     = torch.exp(log_probs - old_log_probs)
    surrogate = (ratio * advantages).mean()
    g = flat_grad(surrogate, list(policy.parameters()), retain_graph=True)

    # --- natural gradient direction: solve F x = g with CG + Fisher-vector products ---
    def Avp(v):
        return fisher_vector_product(policy, states, v, damping=0.1)
    step_dir = conjugate_gradient(Avp, g, n_iters=CG_ITERS)

    # --- scale to the trust region: sqrt(2 delta / (x^T F x)) ---
    shs       = 0.5 * torch.dot(step_dir, Avp(step_dir))      # 0.5 x^T F x
    step_size = torch.sqrt(TRPO_DELTA / (shs + 1e-12))
    full_step = step_size * step_dir
    expected_improve = torch.dot(g, full_step).item()

    # --- backtracking line search on the ACTUAL KL and surrogate ---
    line_search(policy, states, advantages, old_log_probs, full_step,
                expected_improve)

    # --- critic regression (separate optimizer, plain MSE) ---
    for _ in range(5):
        value_opt.zero_grad()
        v_loss = F_loss.mse_loss(value_net(states).squeeze(-1), returns)
        v_loss.backward()
        value_opt.step()
```

Notice what is *absent*: there is no `policy_optimizer.step()`, no learning rate for the policy. The policy update is the line-searched natural-gradient step, parameter-set directly. The only learning rate in TRPO is the critic's. That structural fact has consequences — it is precisely why TRPO does not play nicely with shared actor-critic backbones, as we will see.

Trace the data flow once end to end, because the *order* of operations encodes correctness constraints. Advantages are computed first and normalized — normalization is not cosmetic; it keeps the gradient $g$ at a stable scale across training so the step formula's behavior does not drift as raw returns grow. Old log-probs are stored *before* `snapshot_old`, because the surrogate's importance ratio is defined against the policy that *collected* the data, and that must be captured before any parameters move. The gradient $g$ is then the surrogate gradient at $\theta_{\text{old}}$, where the ratio is identically 1 — so $g$ here equals the plain policy gradient, confirming the first-order-matching claim from Section 2. CG turns $g$ into the natural direction; the `shs` line recomputes $x^\top F x$ with one more Fisher-vector product (it is cheaper to recompute than to thread it out of CG); and the line search commits the move. The critic trains last and entirely separately — its gradients never touch the policy parameters, which is what makes the value head's ordinary Adam coexist with the policy's exotic update.

The GAE helper, for completeness:

```python
def compute_gae(rewards, values, next_values, dones, gamma, lam):
    advantages = torch.zeros_like(rewards)
    gae = 0.0
    for t in reversed(range(len(rewards))):
        mask  = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_values[t] * mask - values[t]
        gae   = delta + gamma * lam * mask * gae
        advantages[t] = gae
    returns = advantages + values
    return advantages, returns
```

GAE's $\lambda$ is the bias-variance dial on the advantage estimate, and it interacts with the trust region in a way worth flagging. A low $\lambda$ (near 0) gives a low-variance but biased advantage (essentially the one-step TD residual); a high $\lambda$ (near 1) gives the high-variance, low-bias Monte Carlo return. TRPO's gradient $g$ inherits that variance, and a noisier $g$ means a noisier natural direction and more line-search rejections. The standard $\lambda = 0.95$ is a good default precisely because it keeps $g$ clean enough that the trust-region step is reliable. If you see TRPO's line search rejecting on nearly every update, lowering $\lambda$ to trade a little advantage bias for a much cleaner gradient is often the fix — another instance of the recurring theme that the quality of the step depends on the quality of the data feeding it.

## 8. Why TRPO is stable

Step back and see why this works. The KL constraint guarantees that the policy never moves more than $\delta$ in distribution space per update. Because the surrogate is a faithful first-order model of the true return *within* that KL ball (Section 2), every step that improves the surrogate inside the ball also improves — or the line search rejects it. The constraint is not a heuristic safety margin; it is the radius within which the monotonic improvement theorem's lower bound is tight enough to be useful.

There is an honest gap between theory and practice worth naming. The *theoretical* TRPO that provably never decreases return uses the max-KL penalty with the conservative constant $C$. The *practical* TRPO uses average KL, a fixed $\delta$, sampled estimates of the gradient and Fisher, and a quadratic approximation to KL. So the strict monotonic guarantee does not literally hold for the algorithm you run — Monte Carlo noise in the advantage estimates alone can produce a step that is "improving" on the sampled surrogate but slightly worsening on the true objective. What holds in practice is far better than vanilla PG: empirically TRPO almost never suffers the catastrophic collapses that plague unconstrained methods, and its return curves are visibly smoother and more monotone. The line search's reject-and-revert fallback is the practical safety net that the theory provides in principle.

It is worth being precise about *which* of the four practical relaxations costs you the guarantee, because they are not equal. The max-to-average KL swap is the largest theoretical compromise: a step with small *average* KL can still have a large KL at a few rarely-visited states, and the theorem's bound is stated over the max. In practice this matters little because the rarely-visited states contribute little to the return, but it is the loosest link in the chain. The quadratic approximation to KL is fully repaired by the line search, which checks the *actual* KL — so that relaxation costs nothing in the end. The sampled Fisher and gradient introduce noise that the damping and the sufficient-decrease test partly absorb. And the fixed $\delta$ in place of the theory's $C$-derived radius simply trades the worthless worst-case bound for an empirically-tuned one. The net: TRPO is not provably monotone as run, but each relaxation was chosen so that the *empirical* behavior stays close to the theoretical promise, and the line search is the component doing the most work to keep it there.

![On HalfCheetah vanilla policy gradient suffers a return collapse near step 300 while TRPO improves monotonically to a far higher final return](/imgs/blogs/trust-region-policy-optimization-trpo-5.png)

#### Worked example: reading a stability curve

Take a HalfCheetah-v4 run with two agents. The vanilla policy-gradient agent climbs to an average return near 900 by update 150, then at update 300 a too-large step pushes the policy into a degenerate gait and the return collapses to roughly 80 — it never recovers within the budget because the collapsed policy generates uninformative data. The TRPO agent on the same env, with $\delta = 0.01$, shows no such cliff: return at update 300 is around 1600 and still climbing, reaching roughly 2900 by update 600. The two curves tell the whole story — the variance of the *update-to-update return change* is small and almost always positive for TRPO, large and occasionally catastrophically negative for vanilla PG. When you tune RL and your return curve looks like a heartbeat monitor flatlining at random, the trust region is the fix.

Quantify the "smoothness" claim, because it is the measurable signature you can check on your own runs. Compute the per-update return change $\Delta R_i = R_{i+1} - R_i$ over a run. For vanilla PG the distribution of $\Delta R_i$ is wide and heavy-tailed on the negative side — most updates help a little, but a handful drop the return by hundreds, and those rare catastrophic updates dominate the variance and define the failure. For TRPO the distribution of $\Delta R_i$ is tight and centered slightly positive, with the negative tail clipped: a bad update costs you a little because the step was bounded, never a lot, because the KL constraint forbids the large move that a large drop requires. You do not need to eyeball the curve; you can log $\Delta R_i$ and watch its negative-tail percentile. A 5th-percentile $\Delta R$ that is mildly negative is healthy TRPO; a 5th-percentile that is catastrophically negative means your trust region is leaking — usually too large a $\delta$, or a Fisher estimate too noisy to constrain the step honestly.

## 9. The limitations that sent everyone to PPO

For all its elegance, TRPO has real costs, and they are the reason it is rarely the default in 2026.

**Computational cost per update.** Every iteration runs conjugate gradient, and every CG iteration runs two backward passes for the Fisher-vector product. That is roughly $2 \times \text{CG\_iters}$ extra backward passes per update — twenty-ish on top of the gradient itself. For large networks this is a serious tax, two to four times the wall-clock of a comparable first-order method per update. The cost scales with the parameter count $d$ only linearly (each Fisher-vector product is $O(d)$), which is what keeps it feasible, but the constant — twenty-plus gradient-equivalents per update — is unavoidable and grows with the CG iteration count you need for a well-conditioned solve. On a large transformer-scale policy this overhead becomes prohibitive, which is one reason TRPO never made the jump to the regimes where modern policy optimization lives.

**No minibatches.** The Fisher-vector product and the KL constraint are defined over the *whole* batch of collected transitions; you cannot cleanly split the trust-region step into minibatch SGD because the constraint couples all the data. TRPO processes the full rollout per update. This wastes the memory-bandwidth efficiency that minibatched first-order methods exploit, and it caps the batch size at what fits in memory for the Fisher-vector products. PPO, by contrast, runs many epochs of minibatch Adam over the same collected batch, extracting far more gradient signal per environment step — a sample-efficiency advantage that compounds over a long run and that TRPO structurally cannot match.

**Hostile to parameter sharing.** Modern actor-critic implementations share a network backbone between the policy and value heads — it saves compute and improves representation learning. TRPO breaks this. The policy update is a direct parameter-set from a natural-gradient line search with no learning rate, while the critic is trained by ordinary SGD with its own optimizer. Splicing a constrained natural-gradient step into half a shared network while the other half wants gradient descent is a mess; in practice TRPO uses *separate* actor and critic networks. That is a real architectural restriction. The shared-backbone pattern is not a minor convenience either — for pixel-based or other high-dimensional observations, the shared encoder is where most of the representation learning happens, and forcing two separate encoders doubles that cost and can hurt sample efficiency.

**Implementation complexity.** Conjugate gradient, Fisher-vector products with `create_graph`, damping, the line search with its accept-ratio and backtracks — there are many places to introduce a silent bug that merely makes the agent learn slowly instead of crashing, which is the worst kind of bug to debug. A missing `create_graph=True` gives a zero Fisher-vector product and a natural direction that is just $g/\lambda$; a Fisher computed on a different batch than the gradient gives a subtly wrong direction; a detach in the wrong place makes the KL nonzero at $\theta_{\text{old}}$ and corrupts the curvature. None of these crash. All of them leave you staring at a learning curve that is merely worse than it should be, with no error to grep for. PPO's clipped surrogate, by contrast, is a dozen lines that either obviously work or obviously do not.

PPO, which we cover in its own post, keeps the trust-region *idea* — limit how far the policy moves per step — but enforces it with a cheap, first-order trick instead of a constrained second-order solve.

![TRPO offers the strongest monotonic-improvement guarantee but loses the minibatch and parameter-sharing flexibility that PPO recovers at far lower per-update compute cost](/imgs/blogs/trust-region-policy-optimization-trpo-4.png)

## 10. Comparison with PPO

PPO (Schulman et al., 2017) is best understood as TRPO's practical heir. Both want bounded policy change. TRPO enforces it with a hard KL constraint solved exactly per step. PPO replaces the constraint with a *clipped surrogate objective* that removes the incentive to move the importance ratio $r(\theta) = \pi_\theta/\pi_{\theta_{\text{old}}}$ outside $[1-\epsilon, 1+\epsilon]$:

$$
L^{\text{CLIP}}(\theta) = \mathbb{E}\left[\min\big(r(\theta) A,\; \text{clip}(r(\theta), 1-\epsilon, 1+\epsilon)\, A\big)\right].
$$

When the policy tries to move the ratio too far in the advantageous direction, the clip flattens the objective so the gradient vanishes — a soft, first-order trust region. PPO then runs ordinary minibatch Adam for several epochs over the batch. It is dramatically simpler: no CG, no Fisher, no line search, no separate networks. It loses TRPO's clean theoretical guarantee — the clip is a heuristic, not a constraint, and a determined optimizer can still drift the *average* policy outside the trust region over many epochs — but in practice it matches or beats TRPO's sample efficiency on most benchmarks at a fraction of the compute and complexity.

The deeper way to see the relationship: both methods are trying to optimize the same surrogate subject to the same "don't move too far" intuition, but they enforce the constraint in dual ways. TRPO enforces it *exactly and globally* — it solves a constrained optimization whose feasible set is precisely the KL ball, paying the full cost of a second-order solve to do so. PPO enforces it *approximately and per-sample* — the clip removes the gradient signal that would push any individual sample's ratio outside the band, so there is no incentive to leave, but also no hard wall stopping a policy that drifts through accumulated small moves over many epochs. TRPO measures distance in the correct Fisher geometry; PPO's ratio-clip is a crude Euclidean-flavored proxy for that distance that happens to work well enough. The remarkable empirical fact of the last decade of deep RL is that the crude proxy, run cheaply for many epochs, beats the exact constraint run expensively once — because the compute saved buys more environment interaction and more gradient steps, and in RL data and steps usually win.

The table below summarizes the trade-offs that matter when you actually choose.

| Property | Vanilla PG | Natural PG | TRPO | PPO |
| --- | --- | --- | --- | --- |
| Trust-region mechanism | none | implicit (fixed natural step) | hard KL constraint + line search | clipped surrogate |
| Step-direction geometry | Euclidean | Fisher (natural) | Fisher (natural) | Euclidean (Adam) |
| Improvement guarantee | none | weak | strongest (theoretical monotone) | heuristic |
| Per-update compute | low | high (Fisher solve) | high (CG + line search) | low |
| Minibatch / multi-epoch | yes | limited | no | yes |
| Shared actor-critic backbone | yes | awkward | no | yes |
| Implementation difficulty | low | high | high | low-moderate |
| Typical MuJoCo score (relative) | low, unstable | moderate | high, stable | high, stable |
| Typical use today | teaching | rare | baseline / theory | default |

And the hyperparameters specific to TRPO:

| Hyperparameter | Effect | Typical range |
| --- | --- | --- |
| KL radius $\delta$ | trust-region size; larger = faster but riskier steps | 0.005 – 0.02 |
| CG iterations | accuracy of $F^{-1}g$; more = better direction, slower | 10 – 20 |
| CG damping $\lambda$ | conditions $F$; too small diverges, too large blunts step | 0.01 – 0.1 |
| Line-search backtracks | how hard to try shrinking before giving up | 10 |
| GAE $\lambda$ | advantage bias-variance trade-off | 0.95 – 0.98 |

Here is a compact head-to-head on the same CartPole policy, useful for seeing in code exactly how little PPO needs relative to the machinery above. The PPO update that replaces the entire CG-plus-line-search block of `trpo_step` is just:

```python
def ppo_update(policy, policy_opt, states, actions, advantages,
               old_log_probs, clip_eps=0.2, epochs=10, minibatch=64):
    n = states.shape[0]
    for _ in range(epochs):
        idx = torch.randperm(n)
        for start in range(0, n, minibatch):
            mb = idx[start:start + minibatch]
            log_probs = policy.log_prob_of(states[mb], actions[mb])
            ratio = torch.exp(log_probs - old_log_probs[mb])
            unclipped = ratio * advantages[mb]
            clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages[mb]
            loss = -torch.min(unclipped, clipped).mean()   # maximize -> minimize negative
            policy_opt.zero_grad()
            loss.backward()
            policy_opt.step()
```

No Fisher matrix, no conjugate gradient, no line search, no parameter-setting — a standard optimizer, minibatches, multiple epochs, and a one-line clipped loss. The contrast with the dozens of lines and twenty backward passes of the TRPO update is the entire argument for why the field moved. PPO recovers most of TRPO's stability from the `clip` and the `min`, which together remove any reward for pushing a sample's ratio past the band, and it does so while keeping every flexibility — minibatches, shared backbones, a plain Adam learning rate — that TRPO sacrifices.

## 11. Case studies

**MuJoCo continuous control (Schulman et al., 2015).** The original TRPO paper benchmarked on simulated locomotion: Swimmer, Hopper, Walker. TRPO learned robust gaits — a hopping Hopper, a walking Walker — directly from raw state vectors with a single set of hyperparameters across tasks, a robustness that vanilla policy gradient could not match without per-task learning-rate tuning. The headline contribution was not a record score but *reliability*: the same algorithm, the same $\delta$, worked across qualitatively different control problems. That generality is what made TRPO a landmark. Before TRPO, getting a policy-gradient method to learn a MuJoCo gait at all often meant a painful per-task learning-rate search; TRPO's $\delta = 0.01$ transferred across tasks because it constrains the behavioral step, which is task-agnostic, rather than the parameter step, which is not.

**Atari from pixels (same paper).** TRPO was also applied to Atari games from raw images with a convolutional policy, learning to play several games at a level competitive with the contemporary DQN results — notable because it showed a policy-gradient method, not just value-based DQN, could handle high-dimensional pixel input. It was not state-of-the-art on every game, but it demonstrated the trust-region approach scaled beyond low-dimensional control. The pixel result also foreshadowed TRPO's eventual limitation: a convolutional policy is exactly the setting where you most want a shared encoder between actor and critic, and TRPO's hostility to parameter sharing made its Atari results more expensive to obtain than a value-based method's — a hint of why the pixel-heavy frontier would later prefer PPO.

**HalfCheetah, TRPO vs PPO vs vanilla PG.** On HalfCheetah-v4, the qualitative picture that practitioners reproduce constantly: vanilla policy gradient is fast early but unstable, prone to the collapse described in Section 8; TRPO is stable and monotone but slow in wall-clock because of the CG overhead; PPO matches TRPO's stability and final return while running several times faster per update thanks to minibatched Adam. Approximate final average returns after a few million steps land in the low thousands for both TRPO and PPO, with PPO winning decisively on returns-per-wall-clock-hour. This three-way comparison is exactly why the field standardized on PPO while keeping TRPO as the theoretical reference point and a stability baseline.

**Four-environment comparison.** Spelling out the pattern across a small benchmark suite makes the tradeoff concrete. On **Swimmer**, a deceptively hard low-dimensional task where naive methods get stuck in a local optimum, TRPO's stable steps reliably escape where vanilla PG stalls. On **Hopper**, a task prone to catastrophic falls, TRPO's monotone curve avoids the late-training collapses that plague unconstrained gradient ascent. On **Walker2d**, both TRPO and PPO reach competent walking gaits, but PPO gets there in roughly a third of the wall-clock because of minibatch reuse. On **HalfCheetah**, the running example, both reach the low thousands in return; TRPO's curve is marginally smoother, PPO's is dramatically faster. Across all four the consistent story is: TRPO and PPO reach similar *final* performance and similar *stability*, and PPO wins *wall-clock* by a wide margin. There is essentially no task in this suite where TRPO's extra cost buys a higher final score — only a (marginally) smoother path to the same place. That empirical near-tie at much higher cost is the whole case for PPO as the default.

**Robotics and the legacy.** TRPO's natural-gradient-plus-trust-region template directly seeded a line of work — PPO, and later constrained-RL methods like CPO (Constrained Policy Optimization) that reuse TRPO's machinery to enforce *safety* constraints, not just KL. In safety-critical RL where you must bound a cost (not exceed a torque limit, never enter an unsafe state), the hard-constraint formulation that TRPO pioneered remains the natural starting point in a way PPO's soft clip is not. CPO literally adds a second constrained-optimization layer on top of TRPO's trust-region solve — the conjugate-gradient and line-search machinery from Sections 5 and 6 is reused almost verbatim, now serving a cost constraint alongside the KL constraint. So even where TRPO itself is not run, its computational core lives on wherever a *certifiable* constraint, not just a soft penalty, is required.

## 12. When to use TRPO (and when not to)

Be decisive here, because the honest answer is that you usually should not reach for TRPO first.

**Use PPO by default.** For nearly all continuous-control and discrete on-policy problems, PPO gives you TRPO's stability at a fraction of the engineering and compute cost, supports minibatches and shared backbones, and is the better-supported, better-documented choice. If someone hands you a new RL problem and says "make it work," start with PPO.

**Reach for TRPO when you need the formal guarantee.** If your setting genuinely requires a provable, near-monotonic improvement property — a research result that depends on it, a safety argument, or a regime where you cannot afford a single catastrophic policy regression and have the compute to pay for it — TRPO's hard constraint earns its cost. The decision tree below is the shortest version of this advice.

![TRPO is the right choice only when you need a formal monotonic guarantee, have a large compute budget, and use no shared actor-critic backbone, otherwise prefer PPO](/imgs/blogs/trust-region-policy-optimization-trpo-7.png)

**Use TRPO's machinery for constrained RL.** If your problem has hard *safety* constraints (cost budgets, no-go states), the TRPO/CPO line of methods that solve a constrained optimization per step is the right family — PPO's soft clip does not give you a constraint you can certify. The distinction is exactly the penalty-versus-constraint argument from Section 3, now applied to safety rather than to KL: a soft penalty lets the agent buy its way out of a safety violation if the reward is high enough, while a hard constraint forbids it categorically. When "never exceed this torque" must mean *never*, you need the constraint formulation, and TRPO's solver is its canonical implementation.

**Do not use TRPO if** you want to share a network between policy and value, you need minibatch efficiency on a large model, you are compute-constrained, or you simply want the thing to work with minimal tuning. In every one of those cases PPO or a modern off-policy method (SAC for continuous control) dominates.

A useful framing: TRPO is the *principled* answer and PPO is the *practical* one. Understanding TRPO is non-negotiable if you want to reason about *why* PPO's clip works — the clip is a cheap approximation to TRPO's trust region — but in production the approximation almost always wins. The pedagogical payoff is real even if the production payoff is not: every time you set PPO's `clip_eps`, you are choosing a trust-region radius, and the only reason that knob is sane is the theory TRPO worked out. Learn TRPO to understand the field; run PPO to ship.

## Key takeaways

1. Vanilla policy gradient is brittle because step size in *parameter* space is uncorrelated with change in *policy behavior*; one oversized step can collapse the policy with no gradient back, and the brittleness worsens as the policy sharpens during training.
2. The monotonic improvement theorem bounds the true return of a new policy by the surrogate minus a KL penalty: maximize that lower bound and the true return provably never decreases. It is a minorize-maximize scheme — a sequence of lower bounds tangent to the true objective.
3. TRPO converts the KL penalty into a hard constraint — maximize the importance-weighted advantage subject to average $\bar{D}_{\text{KL}} \le \delta$ — because the penalty's theoretical constant ($C \approx 40{,}000$ at $\gamma = 0.99$) is far too conservative to train with.
4. The natural gradient $F^{-1}g$ is steepest ascent in policy-distribution space under the Fisher metric; it is invariant to reparameterization and automatically takes small steps in policy-sensitive directions and large steps in safe ones.
5. Conjugate gradient solves $Fx = g$ using only Fisher-vector products (two backward passes each), so $F$ is never formed or inverted — the only way to make natural gradient feasible at neural-network scale.
6. The backtracking line search verifies the *actual* KL and surrogate before committing, reverting to a zero step if no fraction qualifies; this is what gives the practical algorithm its stability and repairs the quadratic-approximation error.
7. TRPO's costs — CG overhead (~20 backward passes per update), no minibatches, no shared backbone, silent-bug-prone implementation — are exactly the costs PPO eliminates with a first-order clipped surrogate.
8. Use PPO by default; reach for TRPO when you need a formal monotonic or safety guarantee and have the compute to pay for it. TRPO's solver also lives on inside constrained-RL methods like CPO.

## Further reading

- Schulman, Levine, Abbeel, Jordan, Moritz, "Trust Region Policy Optimization," ICML 2015 — the original paper; read it for the full monotonic improvement proof and the MuJoCo/Atari results.
- Kakade, Langford, "Approximately Optimal Approximate Reinforcement Learning," ICML 2002 — the conservative policy iteration result that TRPO's theorem extends.
- Kakade, "A Natural Policy Gradient," NeurIPS 2001 — the natural-gradient foundation.
- Amari, "Natural Gradient Works Efficiently in Learning," Neural Computation 1998 — the information-geometry origin of the Fisher-metric gradient.
- Schulman, Wolski, Dhariwal, Radford, Klimov, "Proximal Policy Optimization Algorithms," 2017 — TRPO's practical successor; read alongside this post.
- Schulman, Moritz, Levine, Jordan, Abbeel, "High-Dimensional Continuous Control Using Generalized Advantage Estimation" (GAE), ICLR 2016 — the advantage estimator TRPO pairs with.
- Achiam, Held, Tamar, Abbeel, "Constrained Policy Optimization," ICML 2017 — extends TRPO's machinery to hard safety constraints.
- Sutton & Barto, "Reinforcement Learning: An Introduction," 2nd ed. — for the policy-gradient and advantage foundations.
- Within this series: the [unified map of RL methods](/blog/machine-learning/reinforcement-learning/reinforcement-learning-a-unified-map) for where TRPO sits among policy-gradient methods, and the [RL playbook capstone](/blog/machine-learning/reinforcement-learning/the-reinforcement-learning-playbook) for choosing an algorithm in practice.
