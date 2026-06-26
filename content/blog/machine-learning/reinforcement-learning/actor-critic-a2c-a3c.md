---
title: "Actor-Critic, A2C, and A3C: Combining Policy and Value Learning"
date: "2026-06-22"
publishDate: "2026-06-22"
description: "Why a learned value function tames policy-gradient variance, how GAE tunes the bias-variance dial, and how to build A2C and A3C in PyTorch from first principles to measured results."
tags:
  [
    "reinforcement-learning",
    "deep-learning",
    "actor-critic",
    "policy-gradient",
    "a2c",
    "a3c",
    "gae",
    "machine-learning",
    "pytorch",
    "stable-baselines3",
  ]
category: "machine-learning"
subcategory: "Reinforcement Learning"
author: "Hiep Tran"
featured: true
readTime: 55
image: "/imgs/blogs/actor-critic-a2c-a3c-1.png"
---

Here is a scene I have watched play out more times than I would like to admit. You implement REINFORCE — the clean, beautiful, textbook policy gradient — and you point it at CartPole-v1, an environment so easy it is practically a "hello world." You expect the pole to be balanced within a couple of minutes. Instead, the return graph looks like a seismograph during an earthquake. It climbs to 200, crashes to 30, climbs to 450, crashes to 60. After 2,000 episodes it is *sometimes* good and *sometimes* terrible, and you cannot tell whether your code has a bug or whether this is just how it is.

This is just how it is. REINFORCE works, but it estimates the policy gradient using the **full Monte Carlo return** of each episode, and that return is an extremely noisy random variable. A single lucky or unlucky trajectory can swing the gradient wildly. The fix is not a better optimizer or a smaller learning rate — it is a structural change: introduce a second network, a **critic**, whose only job is to predict how good a state is, and use that prediction to *subtract out the noise* from the actor's learning signal. This is the actor-critic architecture, and it is the foundation under almost every modern on-policy algorithm, PPO included.

![Dataflow showing the environment feeding a state into both an actor policy and a critic value head, with the TD error driving both updates](/imgs/blogs/actor-critic-a2c-a3c-1.png)

By the end of this post you will understand *why* the critic reduces variance (the bias-variance math, derived, not asserted), you will be able to derive **Generalized Advantage Estimation (GAE)** as an exponentially-weighted average of n-step returns, and you will have a complete, runnable PyTorch implementation of **A2C** with a shared backbone and GAE — plus a clear picture of how **A3C** uses asynchronous workers to get the same effect without a replay buffer. We will measure everything on CartPole, LunarLander, and Atari Pong, and finish with a decision rule for when to reach for A2C/A3C versus PPO, DQN, or SAC. Throughout, keep the series spine in mind: every RL algorithm is just a different answer to *which objective to optimize* and *how to estimate its gradient*. Actor-critic's answer is "the policy-gradient objective, with the gradient estimated using a learned baseline."

## 1. The problem actor-critic solves: policy-gradient variance

Let me set up the notation we will use throughout. We have a policy $\pi_\theta(a \mid s)$ — a neural network with parameters $\theta$ that outputs a distribution over actions given a state. The RL objective is to maximize the expected return:

$$
J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[ \sum_{t=0}^{T} \gamma^t r_t \right]
$$

where $\tau = (s_0, a_0, r_0, s_1, \dots)$ is a trajectory and $\gamma \in [0,1)$ is the discount factor. Before we use the policy gradient theorem, it is worth a sentence on *why* it has the form it does, because the structure explains everything that follows. We cannot differentiate $J(\theta)$ by differentiating through the environment — the dynamics $p(s_{t+1} \mid s_t, a_t)$ are unknown and not differentiable. The trick is the **score function** (or REINFORCE) estimator: for any distribution $p_\theta(x)$ and function $f(x)$, $\nabla_\theta \mathbb{E}[f(x)] = \mathbb{E}[f(x) \nabla_\theta \log p_\theta(x)]$. The gradient moves *inside* the expectation and lands on the log-probability, which we *can* differentiate (it is our network). The environment dynamics drop out because they do not depend on $\theta$. This single identity is the entire reason policy gradients work, and it is also the entire reason they are high-variance: a score-function estimator multiplies a gradient direction by a scalar outcome, and scalar outcomes in RL are very noisy.

The **policy gradient theorem** is that identity applied to the RL objective, and it gives us the gradient without needing to differentiate through the environment dynamics:

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[ \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t \mid s_t) \, G_t \right]
$$

where $G_t = \sum_{k=t}^{T} \gamma^{k-t} r_k$ is the **return from time $t$ onward**. The intuition is clean: push up the log-probability of actions that were followed by high return, push down those followed by low return. The score function $\nabla_\theta \log \pi_\theta(a_t \mid s_t)$ tells you the direction in parameter space that makes action $a_t$ more likely; you weight that direction by how good the outcome $G_t$ was.

It is worth sketching where this theorem comes from, because the derivation reveals the structure we exploit. Start from the trajectory form. A trajectory's probability under the policy is $p_\theta(\tau) = p(s_0) \prod_t \pi_\theta(a_t|s_t) p(s_{t+1}|s_t,a_t)$. Take the log and you get a sum: $\log p_\theta(\tau) = \log p(s_0) + \sum_t \log \pi_\theta(a_t|s_t) + \sum_t \log p(s_{t+1}|s_t,a_t)$. Now differentiate with respect to $\theta$. The initial-state term and the dynamics terms do not depend on $\theta$, so they vanish, leaving $\nabla_\theta \log p_\theta(\tau) = \sum_t \nabla_\theta \log \pi_\theta(a_t|s_t)$. Plug this into the score-function identity $\nabla_\theta J = \mathbb{E}[R(\tau) \nabla_\theta \log p_\theta(\tau)]$ and you get the trajectory-level gradient. The final step — replacing the full trajectory return $R(\tau)$ with the *reward-to-go* $G_t$ for each step — uses **causality**: an action at time $t$ cannot affect rewards earned before time $t$, so those earlier rewards have zero expected contribution to the gradient of $a_t$ and can be dropped. That causality argument is the first variance reduction, applied before we even reach the baseline. The baseline is the second.

REINFORCE estimates this expectation with a single (or batched) Monte Carlo sample: run an episode, compute the actual $G_t$ for every step, multiply by the score, and average. It is unbiased — the sample-average converges to the true gradient as you collect more episodes. But unbiased is not the same as *useful*. The problem is **variance**.

It helps to make the actor update completely explicit before we attack the variance, because the rest of the post is a sequence of substitutions into a single template. The thing we actually feed the optimizer is

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\pi}\Big[ \nabla_\theta \log \pi_\theta(a \mid s) \, \Psi_t \Big]
$$

where $\Psi_t$ is some scalar weight on the score function. Every algorithm in the policy-gradient family is just a different choice of $\Psi_t$. REINFORCE uses $\Psi_t = G_t$, the full return. The baselined version uses $\Psi_t = G_t - b(s_t)$. Actor-critic uses $\Psi_t = \hat{A}_t$, an estimate of the advantage — first the one-step TD error, then the n-step return, then GAE. Schulman's GAE paper actually opens by cataloguing these choices of $\Psi_t$ (total reward, reward-to-go, baselined reward-to-go, Q-function, advantage, TD residual) and showing they all yield the same gradient *in expectation* while differing wildly in variance. Hold that template in your head: when we "derive the actor update," all we are ever doing is justifying a new $\Psi_t$ that keeps the expectation correct while shrinking the variance.

### Why the variance is so bad

Consider the magnitude of $G_t$. On CartPole-v1, where every step alive yields reward 1 and an episode can run up to 500 steps, $G_t$ ranges from near 0 (pole fell immediately) to near 500 (perfect episode). That is a return whose standard deviation across episodes can easily be over 100. Now multiply a score-function vector by a scalar that bounces between 30 and 500, episode to episode, *for the same state-action pair*. The gradient estimate inherits that bounce. The optimizer is being shoved around by trajectory-level luck, not by whether action $a_t$ was genuinely better than the alternatives.

There is a deeper structural issue. The return $G_t$ credits action $a_t$ with everything that happened *after* it — including the consequences of all the later actions $a_{t+1}, a_{t+2}, \dots$, which $a_t$ had nothing to do with. If you took a great action at step 5 but then took terrible actions at steps 6 through 50, the return $G_5$ will be low, and REINFORCE will *punish* the great action at step 5. This is noise injected by the policy's own later randomness.

### The baseline trick: subtract a state-dependent constant

The first and most important variance reduction is the **baseline**. Here is the key fact, which we will prove because it is the entire justification for the critic: you can subtract any function $b(s_t)$ that depends only on the state (not the action) from $G_t$ without changing the expected gradient. That is,

$$
\nabla_\theta J(\theta) = \mathbb{E}\left[ \sum_t \nabla_\theta \log \pi_\theta(a_t \mid s_t) \big(G_t - b(s_t)\big) \right]
$$

is still equal to the true gradient, for *any* baseline $b(s_t)$. Why? Because the expected value of the score-times-baseline term is zero:

$$
\mathbb{E}_{a \sim \pi_\theta}\left[ \nabla_\theta \log \pi_\theta(a \mid s) \, b(s) \right]
= b(s) \sum_a \pi_\theta(a \mid s) \nabla_\theta \log \pi_\theta(a \mid s)
= b(s) \sum_a \nabla_\theta \pi_\theta(a \mid s)
$$

and since the probabilities sum to one, $\sum_a \nabla_\theta \pi_\theta(a \mid s) = \nabla_\theta \sum_a \pi_\theta(a \mid s) = \nabla_\theta 1 = 0$. The baseline contributes nothing in expectation, so subtracting it leaves the gradient unbiased. But it can dramatically *reduce variance*, because now we are weighting the score by $(G_t - b(s_t))$ — "how much better than expected was this outcome" — instead of by the raw return.

It is worth being precise about *which* baseline minimizes variance, because "use the value function" is folklore that is almost — but not exactly — true. The variance of the gradient estimate for a single state-action pair is proportional to $\mathbb{E}[\|\nabla_\theta \log \pi_\theta\|^2 (G_t - b)^2]$. Minimizing this over the scalar $b$ by setting the derivative to zero gives the true optimum:

$$
b^\star(s) = \frac{\mathbb{E}\big[\|\nabla_\theta \log \pi_\theta(a|s)\|^2 \, G_t\big]}{\mathbb{E}\big[\|\nabla_\theta \log \pi_\theta(a|s)\|^2\big]}
$$

— a gradient-magnitude-weighted average of the return. The state-value function $V^\pi(s)$ is the *unweighted* expected return, which is the optimum only if the squared score is uncorrelated with the return. In practice $V^\pi(s)$ is close enough to $b^\star(s)$, far easier to estimate (we already need a value function for bootstrapping), and it has a clean interpretation, so everyone uses it. This is a good example of the kind of trade-off that runs through all of RL: the theoretically optimal object is intractable, so we use a tractable approximation that captures most of the benefit. The variance reduction from the value baseline is typically the difference between an algorithm that learns and one that does not, even though it is not the last word in variance optimality.

The state-value function is $V^\pi(s) = \mathbb{E}_{a \sim \pi}[Q^\pi(s,a)]$, the expected return from state $s$ under the current policy. When $b(s_t) = V^\pi(s_t)$, the weighting term $G_t - V^\pi(s_t)$ becomes an estimate of the **advantage**:

$$
A^\pi(s_t, a_t) = Q^\pi(s_t, a_t) - V^\pi(s_t)
$$

which measures exactly the right thing: *how much better is taking action $a_t$ in state $s_t$ than the policy's average behavior in that state?* That is the signal we want. The critic's job is to learn $V^\pi(s)$ so we can compute advantages. This is the whole idea of actor-critic in one sentence.

#### Worked example: how much variance does the baseline actually remove?

Numbers make the baseline argument concrete in a way that the algebra does not. Stay on CartPole and pick a single state $s_t$ whose true value is $V^\pi(s_t) = 60$. Suppose the return-from-here $G_t$ is roughly Gaussian across episodes with mean 60 and standard deviation 50 (a realistic spread for mid-episode CartPole). The REINFORCE weight is $G_t$ itself, with $\mathbb{E}[G_t^2] = \mu^2 + \sigma^2 = 60^2 + 50^2 = 6100$. The baselined weight is $G_t - 60$, with $\mathbb{E}[(G_t - 60)^2] = \sigma^2 = 2500$. The variance term that multiplies the squared score in the gradient variance fell from 6100 to 2500 — about a $2.4\times$ reduction — purely from subtracting a constant that contributes nothing to the mean. And that is with a *perfect* baseline equal to the true value; a learned critic that is even roughly right captures most of this. The lesson is that the baseline's leverage is largest exactly when the return is large in absolute terms (long-horizon, dense-reward tasks), which is why value baselines matter more on Atari than on a 10-step toy. The advantage formulation pushes this further still by also using the bootstrap to shrink the *spread* $\sigma$, not just recenter the mean — that is the next section.

## 2. The actor-critic architecture

So we have two functions. The **actor** is the policy $\pi_\theta(a \mid s)$. The **critic** is a value function $V_\phi(s)$ with its own parameters $\phi$, trained to predict the expected return from a state. The actor decides what to do; the critic tells the actor how surprised it should be by the outcome, and that surprise — the advantage — is the learning signal. Figure 1 above shows the dataflow: the environment emits a state, the actor maps it to an action distribution and the critic maps it to a scalar value, the action goes back to the environment, and the resulting reward combined with the critic's predictions produces a temporal-difference error that drives both updates.

The actor update follows directly from the baselined policy gradient theorem:

$$
\nabla_\theta J(\theta) = \mathbb{E}\left[ \nabla_\theta \log \pi_\theta(a_t \mid s_t) \, \hat{A}_t \right]
$$

where $\hat{A}_t$ is an *estimate* of the advantage. The simplest such estimate uses the critic to bootstrap a one-step target, which brings us to the temporal-difference error.

### The TD error is a single-sample advantage estimate

The **temporal-difference (TD) error** is defined as

$$
\delta_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)
$$

Read it carefully. $r_t + \gamma V_\phi(s_{t+1})$ is a one-step estimate of the return from $s_t$ if we took action $a_t$ — the immediate reward plus the discounted value of where we landed. Subtracting $V_\phi(s_t)$, the predicted value of where we started, gives "how much better than expected did this one step turn out." That is precisely a single-sample, bootstrapped estimate of the advantage $A^\pi(s_t, a_t)$. In fact, under the true value function, $\mathbb{E}[\delta_t \mid s_t, a_t] = A^\pi(s_t, a_t)$, because $\mathbb{E}[r_t + \gamma V^\pi(s_{t+1}) \mid s_t, a_t] = Q^\pi(s_t, a_t)$.

Let me prove the unbiasedness claim properly, because the post promised it and it is short. Condition on the state-action pair $(s_t, a_t)$ and take the expectation of $\delta_t$ over the next-state randomness, using the *true* value function $V^\pi$:

$$
\mathbb{E}[\delta_t \mid s_t, a_t] = \mathbb{E}\big[r_t + \gamma V^\pi(s_{t+1}) \mid s_t, a_t\big] - V^\pi(s_t).
$$

The first expectation is, by the very definition of the action-value function, $Q^\pi(s_t, a_t) = \mathbb{E}[r_t + \gamma V^\pi(s_{t+1}) \mid s_t, a_t]$ — that is just the Bellman equation for $Q^\pi$ written out. Substituting,

$$
\mathbb{E}[\delta_t \mid s_t, a_t] = Q^\pi(s_t, a_t) - V^\pi(s_t) = A^\pi(s_t, a_t).
$$

So the one-step TD error is an *unbiased single-sample estimate of the true advantage*, provided the critic equals the true value function. The catch lives in that proviso: in practice we use the learned $V_\phi$, not $V^\pi$, so the realized $\delta_t$ carries a bias equal to the critic's error, $\gamma(V_\phi - V^\pi)(s_{t+1}) - (V_\phi - V^\pi)(s_t)$. This is the precise origin of the bias we keep referring to — it is exactly the critic's approximation error, propagated one bootstrap step. When the critic is good the bias is small; when it is randomly initialized the bias is large, which is why the first few thousand actor updates are partly noise and why a too-fast actor (which races ahead of a still-wrong critic) is the classic divergence mode.

So the one-step actor-critic uses $\hat{A}_t = \delta_t$. The actor pushes up actions with positive TD error and down those with negative TD error. The critic, meanwhile, is trained to make $\delta_t$ small — to predict values accurately — by minimizing the squared TD error:

$$
L_{\text{critic}}(\phi) = \mathbb{E}\left[ \big( r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t) \big)^2 \right]
$$

This is just regression toward a bootstrapped target $y_t = r_t + \gamma V_\phi(s_{t+1})$. Crucially, the target $y_t$ is treated as a *constant* during the critic gradient — we do not backpropagate through $V_\phi(s_{t+1})$ in the target — which is the semi-gradient TD update from Sutton & Barto. In code this means calling `.detach()` on the target.

Notice what just happened to the variance. The actor's learning signal $\delta_t$ depends only on one reward $r_t$ and two value predictions, not on the entire downstream return. The wild episode-level noise is gone. We have traded it for a small amount of **bias**: $\delta_t$ uses the critic's imperfect estimate $V_\phi$ rather than the true value, so early in training, when the critic is wrong, the actor is learning from a biased signal. That bias-variance trade-off is the central tension of this whole family of methods, and GAE is the dial that controls it.

### The chicken-and-egg coupling, and why it converges anyway

There is something philosophically uncomfortable about actor-critic that is worth confronting head-on, because it is the source of most "why won't this train" mysteries. The actor is being trained using advantages computed by the critic, but the critic is being trained to predict the returns generated by the actor. Each depends on the other. If the critic is garbage, the actor learns from garbage advantages; if the actor is changing fast, the critic is chasing a moving target ($V^\pi$ shifts as $\pi$ shifts). This is a coupled, two-timescale optimization, and naively you might expect it to spiral.

It works in practice because of a **two-timescale** principle, made rigorous by Borkar and others: if the critic learns *faster* than the actor (or equivalently the actor moves slowly enough that the critic can track it), then from the actor's perspective the critic is approximately converged at each step, so the actor sees approximately-correct advantages, and the whole system converges to a local optimum of the policy-gradient objective. In code, the two-timescale idea shows up as the critic typically using a larger or equal learning rate, the value loss coefficient being non-trivial, and — importantly — keeping the actor's steps small (low learning rate, gradient clipping, and in PPO's case the clip itself). When people see actor-critic diverge, the usual culprit is the actor outrunning the critic: the policy changes so fast that the critic's value estimates are stale, the advantages become wrong-signed, and the policy chases its own noise. Slow the actor down or speed the critic up and it usually recovers. Keep this coupling in mind — it explains why so many "tricks" in this post are really just ways of keeping the two timescales separated.

![Before-and-after comparison of REINFORCE using full Monte Carlo returns versus actor-critic using a bootstrapped TD signal](/imgs/blogs/actor-critic-a2c-a3c-3.png)

#### Worked example: variance of the learning signal on CartPole

Suppose we are in a state $s_t$ midway through a CartPole episode, and the true value is $V^\pi(s_t) = 60$ (we expect about 60 more steps of survival on average). Consider the same state-action pair sampled across three episodes that happened to end at very different times.

In episode A the agent survived 140 more steps, so the Monte Carlo return is $G_t = 140$. In episode B it survived 20 more steps, $G_t = 20$. In episode C, 75 steps, $G_t = 75$. The REINFORCE weights are 140, 20, 75 — a spread of 120, with the same action being credited as wildly good, then bad, then mediocre. Now the actor-critic TD signal. Say the critic is decent and the per-step values are roughly accurate. The TD errors $\delta_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)$ with $\gamma = 0.99$, $r_t = 1$, $V_\phi(s_t) = 60$, $V_\phi(s_{t+1}) = 59$ come out to $1 + 0.99 \times 59 - 60 = -0.59$ — a small, stable number near zero, the same in every episode regardless of how the episode eventually ended. The variance of the actor's weight dropped from "spread of 120" to "spread of fractions." That is the entire reason actor-critic learns CartPole in roughly 400 episodes where REINFORCE needs thousands and oscillates the whole way.

## 3. Advantage estimation: from TD(0) to n-step to GAE

The one-step TD error $\delta_t$ is the lowest-variance, highest-bias end of a spectrum. At the other end is the Monte Carlo advantage $G_t - V_\phi(s_t)$, which is unbiased but high-variance. In between live the **n-step returns**, and GAE is the principled way to blend all of them.

![Stack of advantage estimators from low-variance TD(0) at the bottom to high-variance Monte Carlo at the top with GAE in the balanced middle](/imgs/blogs/actor-critic-a2c-a3c-2.png)

### n-step returns

Instead of bootstrapping after one step, bootstrap after $n$ steps. The n-step return target is

$$
G_t^{(n)} = r_t + \gamma r_{t+1} + \dots + \gamma^{n-1} r_{t+n-1} + \gamma^n V_\phi(s_{t+n})
$$

and the corresponding n-step advantage is $\hat{A}_t^{(n)} = G_t^{(n)} - V_\phi(s_t)$. For $n=1$ this is exactly the TD error; for $n = \infty$ (run to the end of the episode) it is the Monte Carlo advantage. As $n$ grows, you use more real rewards and less of the critic's prediction, so bias drops but variance rises (you are accumulating more random rewards). The choice of $n$ is a knob, but it is a *discrete* knob and it weights the n-step return entirely on one horizon. GAE does something smoother.

### Deriving GAE as an exponential mixture

**Generalized Advantage Estimation** (Schulman et al., 2016) defines the advantage as an exponentially-weighted average of all the n-step advantage estimators. Here is the derivation, because it is genuinely elegant and you should see where the formula comes from.

First, express each n-step advantage in terms of the per-step TD errors. Define $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$ as before. A short telescoping computation shows that the n-step advantage is just a discounted sum of TD errors:

$$
\hat{A}_t^{(n)} = \sum_{l=0}^{n-1} \gamma^l \, \delta_{t+l}
$$

You can verify the $n=2$ case by hand: $\delta_t + \gamma \delta_{t+1} = (r_t + \gamma V(s_{t+1}) - V(s_t)) + \gamma(r_{t+1} + \gamma V(s_{t+2}) - V(s_{t+1})) = r_t + \gamma r_{t+1} + \gamma^2 V(s_{t+2}) - V(s_t)$, which is exactly $G_t^{(2)} - V(s_t)$. The $V(s_{t+1})$ terms cancel — that is the telescoping.

Now GAE takes an exponentially-weighted average of these n-step advantages with weight $(1-\lambda)\lambda^{n-1}$ for the n-step estimator (the geometric weights sum to one). Substituting the TD-error form and collecting terms, the whole thing collapses into a beautifully simple sum:

$$
\hat{A}_t^{\text{GAE}(\gamma, \lambda)} = \sum_{l=0}^{\infty} (\gamma \lambda)^l \, \delta_{t+l}
$$

That is the GAE estimator. It is just a discounted sum of future TD errors, where the discount is $\gamma\lambda$ instead of $\gamma$. Two special cases fall out immediately:

- $\lambda = 0$: only the first term survives, $\hat{A}_t = \delta_t$. This is the one-step TD advantage — low variance, high bias.
- $\lambda = 1$: $\hat{A}_t = \sum_l \gamma^l \delta_{t+l} = G_t - V(s_t)$, the full Monte Carlo advantage — low bias, high variance.

So $\lambda$ slides continuously between these extremes. In practice $\lambda \in [0.9, 0.99]$ is the sweet spot — close enough to 1 to keep bias low, far enough below to tame variance. The standard default is $\lambda = 0.95$, $\gamma = 0.99$.

### The bias-variance trade-off as a function of $\lambda$, made quantitative

It is worth pinning down *why* $\lambda$ trades bias for variance, rather than just asserting it, because the shape of the trade-off tells you which way to turn the knob when something is wrong. Two competing effects are in play, and both are visible directly in the formula $\hat{A}_t = \sum_l (\gamma\lambda)^l \delta_{t+l}$.

**Bias.** Each $\delta_{t+l}$ carries the critic's approximation error (derived above). When $\lambda = 0$ the estimator is a *single* TD error, so it inherits the critic's error in full — maximal bias. As $\lambda \to 1$ the estimator accumulates more and more *real* rewards and leans less on any single bootstrap, so the systematic dependence on the critic's error shrinks toward zero; at $\lambda = 1$ it is the pure Monte Carlo advantage, which is unbiased regardless of how wrong the critic is. So **bias is monotonically decreasing in $\lambda$.**

**Variance.** Each additional $\delta_{t+l}$ that gets meaningful weight injects the randomness of one more reward and one more sampled transition. A small $\lambda$ exponentially down-weights distant terms ($(\gamma\lambda)^l$ decays fast), so the estimator effectively averages over a short, low-noise window. A large $\lambda$ keeps many noisy future terms alive, so the accumulated stochasticity of the whole rollout flows into the estimate. So **variance is monotonically increasing in $\lambda$.** The effective horizon of the sum is roughly $1/(1 - \gamma\lambda)$ terms: at $\gamma=0.99,\ \lambda=0.95$ that is about 17 steps of meaningful credit, versus a single step at $\lambda=0$ and the whole episode at $\lambda=1$. That number — "how many future TD errors actually matter" — is the most intuitive handle on the dial.

The practical reading: if training is **noisy and unstable** (return oscillates, gradient norms spike), your variance is too high — *lower* $\lambda$ toward 0.9. If training is **stable but stuck below the achievable score** because the critic is dragging the advantages toward a wrong systematic value, your bias is too high — *raise* $\lambda$ toward 0.99. The default 0.95 sits where, empirically across the MuJoCo and Atari suites in Schulman's paper, the sum of the two error sources is near its minimum for most tasks. Note also that $\gamma$ and $\lambda$ play subtly different roles even though they multiply together: $\gamma$ encodes the *true* discounting of the problem (the planning horizon you care about), while $\lambda$ is a pure *estimator* knob that does not change what you are optimizing, only how you estimate its gradient. Conflating them — e.g. lowering $\gamma$ to reduce variance — silently changes the objective and is a common beginner mistake; lower $\lambda$ instead.

### The backward recursion

The reason GAE is cheap to compute is that the infinite sum has a simple backward recursion. Within a rollout, compute it from the last step backward:

$$
\hat{A}_t = \delta_t + \gamma \lambda \, \hat{A}_{t+1}
$$

with $\hat{A}_{T} = \delta_{T}$ at the rollout boundary (and a bootstrap value for the final next-state if the episode did not terminate). One pass, $O(T)$, no matrix anything. Here it is in NumPy, the exact routine that sits inside every A2C/PPO implementation:

```python
import numpy as np

def compute_gae(rewards, values, dones, next_value, gamma=0.99, lam=0.95):
    """
    rewards:    array [T]   reward at each step
    values:     array [T]   critic V(s_t) for each step
    dones:      array [T]   1.0 if episode terminated at step t else 0.0
    next_value: scalar      V(s_{T}) bootstrap for the step after the rollout
    returns:    advantages [T], value_targets [T]
    """
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    last_adv = 0.0
    for t in reversed(range(T)):
        # if step t ended the episode, there is no bootstrap from t+1
        next_v = next_value if t == T - 1 else values[t + 1]
        mask = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_v * mask - values[t]
        last_adv = delta + gamma * lam * mask * last_adv
        advantages[t] = last_adv
    value_targets = advantages + values  # critic regresses toward these
    return advantages, value_targets
```

Note the `mask = 1.0 - dones[t]` factor: at a terminal step there is no next state, so we zero out both the bootstrap and the recursion carry. Forgetting this `mask` is the single most common GAE bug — it leaks value across episode boundaries and quietly wrecks training. The `value_targets = advantages + values` line is worth pausing on: since $\hat{A}_t = G_t^{\lambda} - V(s_t)$, adding back $V(s_t)$ recovers the $\lambda$-return $G_t^{\lambda}$, which is exactly the target the critic should regress toward. We reuse the same advantage computation for both heads.

#### Worked example: GAE on a three-step rollout

Let us trace the recursion by hand. Take $\gamma = 0.99$, $\lambda = 0.95$, so $\gamma\lambda = 0.9405$. Suppose a short rollout produced TD errors $\delta_0 = 0.5$, $\delta_1 = -0.2$, $\delta_2 = 1.0$, with no terminations inside the rollout. Working backward: $\hat{A}_2 = \delta_2 = 1.0$. Then $\hat{A}_1 = \delta_1 + 0.9405 \times \hat{A}_2 = -0.2 + 0.9405 \times 1.0 = 0.7405$. Then $\hat{A}_0 = \delta_0 + 0.9405 \times \hat{A}_1 = 0.5 + 0.9405 \times 0.7405 = 1.1965$.

Now compare to the pure one-step estimates, which would have been $\{0.5, -0.2, 1.0\}$. GAE has propagated the strong positive surprise at step 2 backward, so step 1 — which looked mildly bad on its own ($\delta_1 = -0.2$) — is now credited positively ($0.7405$) because it led to a state from which good things happened. That backward credit assignment, smoothly discounted by $\gamma\lambda$, is exactly what raw TD(0) cannot do and what makes GAE so much more sample-efficient. Crank $\lambda$ to 0 and you would lose all of that propagation; crank it to 1 and the propagation would be undiscounted and noisier.

![Tree of advantage-estimator choices branching from lambda zero through lambda one with GAE as the balanced recommended path](/imgs/blogs/actor-critic-a2c-a3c-8.png)

## 4. Shared backbone: one network or two?

A practical architecture question: should the actor and critic be entirely separate networks, or share their lower layers? Both appear in the literature and in production.

The case for **sharing** a feature backbone is representational and computational. The features that help you decide *what to do* (the actor) and the features that help you decide *how good a state is* (the critic) overlap heavily — both need to understand "the pole is tilting fast" or "the lander is drifting left." Sharing means you learn those features once, with twice the gradient signal flowing into them, and you halve the forward-pass cost. For high-dimensional inputs like Atari frames, where the convolutional stack is the bulk of the compute, a shared CNN backbone is standard.

The case **against** sharing is the **gradient interference** problem. The actor loss and the critic loss have very different scales and dynamics. The critic's MSE loss can produce large gradients early in training (value predictions start near zero and targets can be hundreds), and those gradients flow into the shared backbone and can swamp the actor's much smaller policy-gradient signal — or vice versa. When the two heads pull the shared features in conflicting directions, neither learns well. The standard mitigation is to scale the critic loss by a coefficient $c_1$ (commonly 0.5) and add an entropy bonus $c_2$ to keep the actor exploring:

$$
L_{\text{total}} = \underbrace{-\mathbb{E}[\log \pi_\theta(a_t|s_t) \hat{A}_t]}_{\text{actor}} + c_1 \underbrace{\mathbb{E}[(V_\phi(s_t) - G_t^\lambda)^2]}_{\text{critic}} - c_2 \underbrace{\mathbb{E}[H(\pi_\theta(\cdot|s_t))]}_{\text{entropy bonus}}
$$

The entropy term $H(\pi) = -\sum_a \pi(a) \log \pi(a)$ is maximized by a uniform policy; subtracting it (with a positive $c_2$, so it is *rewarded*) discourages the policy from collapsing to a deterministic choice too early, which is the classic on-policy failure mode where the agent stops exploring and gets stuck. Typical values: $c_1 = 0.5$, $c_2 = 0.01$.

My rule of thumb after shipping a few of these: for low-dimensional vector observations (CartPole, LunarLander), use **separate networks** — the compute saving is negligible and you sidestep interference entirely, which makes debugging far easier. For pixel inputs (Atari), use a **shared CNN backbone** with separate small linear heads, and tune $c_1$ carefully. If your critic loss is dominating, lower $c_1$; if your actor stops improving while the critic is fine, that is the tell.

To put numbers on the trade-off: a typical Atari setup uses the Nature-DQN CNN — three conv layers totalling roughly 1.6M parameters in the trunk — followed by a 512-unit dense layer and then the two tiny heads (a few thousand parameters each). With a **shared** backbone you run that 1.6M-parameter conv stack *once* per state and get both outputs; with **separate** networks you run it *twice*, nearly doubling both the forward-pass FLOPs and the memory for activations during backprop. On a batch of $16 \times 5 = 80$ stacked-frame states that is the difference between, say, ~6 ms and ~11 ms per forward pass on a mid-range GPU — small per step but multiplied over tens of millions of steps it is hours of wall-clock. That is why pixel inputs almost always share. For a CartPole MLP, by contrast, the "backbone" is two 128-unit layers — about 35k parameters — and running it twice costs microseconds, so the compute argument evaporates and the interference-avoidance argument wins. The crossover is essentially "is the shared trunk the dominant cost?" — for convolutional or transformer encoders, yes; for small MLPs, no.

There is one more shared-backbone subtlety worth internalizing. When the two heads share a trunk, the *value targets* and the *advantage signal* are computed from the **same** network's outputs, which couples the bias of the critic into the advantage in a slightly tighter loop than the separate-network case. Some implementations (including parts of SB3 and the original OpenAI baselines) therefore keep the value-function and policy *optimizer statistics* somewhat decoupled or clip the value loss. You do not need to do this for CartPole, but it is the reason production code that looks needlessly elaborate often is not — those guards exist precisely to keep the coupled timescales from interfering.

#### Worked example: diagnosing gradient interference on LunarLander

Here is a real failure pattern. You train shared-backbone A2C on LunarLander-v2 and the return climbs to about 120, then plateaus and starts oscillating, never reaching the solved threshold of 200. You log the two loss components and the per-head gradient norms. You see the critic loss starting at around 4,000 (because early value targets near $\pm 200$ produce huge MSE) and the gradient norm into the shared backbone from the critic head is roughly 50, while the actor head contributes a norm of about 0.8. The actor's signal is being drowned by a factor of ~60. The fix is not mysterious: lower `vf_coef` from 0.5 to 0.25, which halves the critic's contribution, and confirm the two gradient norms come into the same order of magnitude. After the change the return climbs past 200. The general principle — measure the *relative* gradient magnitudes of the two heads into the shared trunk, and keep them within an order of magnitude — is the single most useful debugging habit for shared-backbone actor-critic. When you cannot get them balanced, fall back to fully separate networks.

### Continuous actions: the Gaussian policy head

Everything above used a `Categorical` (discrete) actor, but actor-critic handles continuous actions with a trivial change to the actor head: output a mean $\mu_\theta(s)$ and a (state-independent or state-dependent) log-standard-deviation, and sample from a Gaussian. The score function $\nabla_\theta \log \pi_\theta(a|s)$ is now the log-density of a normal distribution, which PyTorch computes for you:

```python
from torch.distributions import Normal

class GaussianActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=128):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.mu_head = nn.Linear(hidden, act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))  # state-independent
        self.critic_head = nn.Linear(hidden, 1)

    def act(self, x):
        z = self.backbone(x)
        mu = self.mu_head(z)
        std = self.log_std.exp()
        dist = Normal(mu, std)
        action = dist.sample()
        logp = dist.log_prob(action).sum(-1)   # sum over action dims
        ent = dist.entropy().sum(-1)
        value = self.critic_head(z).squeeze(-1)
        return action, logp, ent, value
```

The only subtlety is the `.sum(-1)` over action dimensions: the multivariate Gaussian with a diagonal covariance factorizes, so the joint log-probability is the sum of per-dimension log-probabilities, and the joint entropy is the sum of per-dimension entropies. Everything else — GAE, the combined loss, advantage normalization — is byte-for-byte identical to the discrete case. This is the quiet power of the actor-critic decomposition: the advantage machinery is completely agnostic to the action space, so switching from CartPole to a MuJoCo continuous-control task is a five-line change to the actor head, not a new algorithm.

## 5. A2C: synchronous advantage actor-critic

Here is a subtle historical irony worth knowing. A3C (asynchronous) came first, in Mnih et al. 2016. Then researchers at OpenAI noticed that the *asynchrony* was not actually the secret sauce — a *synchronous* version that waited for all workers and averaged their gradients worked just as well or better, while being far simpler to implement and reason about. They called it **A2C** (Advantage Actor-Critic), dropping the "asynchronous." So A2C is really "A3C without the headache," and it is what you should usually reach for.

The A2C loop is a clean cycle, shown in figure 4: maintain $N$ parallel environment copies, step them all in lockstep for $n$ steps to collect a rollout of $N \times n$ transitions, compute GAE advantages, take one combined gradient step on the total loss, and repeat. The parallelism is what decorrelates the samples — at any moment your batch contains $N$ states from $N$ different points in $N$ different episodes, so the gradient is not dominated by one trajectory's idiosyncrasies. This is on-policy: after each update the old data is thrown away, because it was generated by the now-stale policy.

### How synchronous parallel workers actually work

The phrase "step them all in lockstep" hides a real engineering decision: the $N$ environments can live in the *same* process (a `SyncVectorEnv`, stepped in a Python loop) or in $N$ *separate* processes (a `SubprocVecEnv`, stepped truly in parallel across CPU cores and communicating over pipes). The trade-off is the usual one. A `SyncVectorEnv` has no inter-process overhead and is faster when each environment step is cheap (CartPole's physics is a few floating-point operations — process startup and pipe serialization would cost more than they save). A `SubprocVecEnv` pays a fixed serialization cost per step but genuinely runs the environments on different cores, which wins decisively when each step is *expensive* — Atari frame rendering and preprocessing, a physics simulator, or anything calling into a heavy C++ engine. The rule: cheap envs → in-process vectorization; expensive envs → subprocess vectorization. Here is the subprocess version wired up explicitly with SB3's vector-env machinery:

```python
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
import gymnasium as gym

def make_env(env_id, rank, seed=0):
    def _init():
        env = gym.make(env_id)
        env.reset(seed=seed + rank)   # distinct seed per worker → diverse trajectories
        return env
    return _init

if __name__ == "__main__":           # SubprocVecEnv requires the spawn guard on macOS/Windows
    num_envs = 16
    env_fns = [make_env("LunarLander-v2", rank=i) for i in range(num_envs)]
    vec_env = SubprocVecEnv(env_fns)     # 16 OS processes, one env each
    vec_env = VecMonitor(vec_env)        # logs per-episode return/length across workers

    # vec_env.step(actions) now blocks until ALL 16 workers have stepped once,
    # then returns stacked arrays of shape (16, ...). That synchronous barrier is
    # exactly the "lockstep" — the gradient step waits for every worker.
```

The distinct per-worker seed (`seed + rank`) is not cosmetic: it is what guarantees the 16 workers explore *different* trajectories rather than 16 identical copies, and identical copies would defeat the entire decorrelation purpose. The `if __name__ == "__main__"` guard is mandatory because `SubprocVecEnv` uses process spawning, which re-imports the module in each child — without the guard you get an infinite fork bomb on macOS and Windows. These two lines are the most common first-time `SubprocVecEnv` mistakes.

### Collecting the rollout and averaging gradients

Once the vector env is stepping in lockstep, the rollout collection is mechanical: for $n$ steps, call the policy on the batched $N$-vector of observations (one batched forward pass, not $N$ separate ones — this is why vectorization is GPU-friendly), step all envs, and stash the log-probs, values, rewards, and dones into tensors of shape $(n, N)$. After $n$ steps you have $N \times n$ transitions. Crucially, the gradient is computed on the *whole* $(n, N)$ batch at once: when we write `actor_loss = -(log_probs * adv.detach()).mean()`, the `.mean()` is over both the time axis and the worker axis, which is *exactly* the "average gradients across workers" that A2C is named for. A2C does not literally compute $N$ separate gradients and average them in a parameter server (that mental model comes from A3C); it concatenates all workers' transitions into one big batch and takes one gradient of the mean loss, which is *mathematically identical* to averaging per-worker gradients but far more efficient on a GPU. That equivalence — averaging gradients equals one gradient of the averaged loss, by linearity of differentiation — is why the synchronous formulation is both simpler and faster than A3C's literal gradient-averaging.

![Pipeline of the A2C training loop from resetting parallel environments through rollout collection, GAE, combined loss, and a synchronous gradient step](/imgs/blogs/actor-critic-a2c-a3c-4.png)

### Full PyTorch A2C with shared backbone and GAE

Here is a complete, runnable A2C. It uses Gymnasium's synchronous vector env for the parallel workers, a shared MLP backbone with separate actor and critic heads, and the GAE routine from section 3. This is not pseudocode — it trains CartPole to ~500 return.

```python
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class ActorCritic(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden=128):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.actor_head = nn.Linear(hidden, n_actions)
        self.critic_head = nn.Linear(hidden, 1)

    def forward(self, x):
        z = self.backbone(x)
        logits = self.actor_head(z)
        value = self.critic_head(z).squeeze(-1)
        return logits, value

    def act(self, x):
        logits, value = self.forward(x)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value
```

The training loop drives $N$ envs, collects an $n$-step rollout, bootstraps the value of the final state, runs GAE, and applies the combined loss:

```python
def compute_gae_torch(rewards, values, dones, next_value, gamma=0.99, lam=0.95):
    T = rewards.shape[0]
    adv = torch.zeros_like(rewards)
    last = torch.zeros(rewards.shape[1])
    for t in reversed(range(T)):
        next_v = next_value if t == T - 1 else values[t + 1]
        mask = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_v * mask - values[t]
        last = delta + gamma * lam * mask * last
        adv[t] = last
    return adv, adv + values

def train_a2c(env_id="CartPole-v1", num_envs=16, n_steps=5,
              total_steps=200_000, lr=7e-4, gamma=0.99, lam=0.95,
              vf_coef=0.5, ent_coef=0.01, max_grad_norm=0.5):
    envs = gym.make_vec(env_id, num_envs=num_envs)
    obs_dim = envs.single_observation_space.shape[0]
    n_actions = envs.single_action_space.n
    model = ActorCritic(obs_dim, n_actions)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    obs, _ = envs.reset(seed=0)
    obs = torch.as_tensor(obs, dtype=torch.float32)
    updates = total_steps // (num_envs * n_steps)

    for update in range(updates):
        log_probs, values, rewards, dones, entropies = [], [], [], [], []
        for _ in range(n_steps):
            action, logp, ent, value = model.act(obs)
            next_obs, r, term, trunc, _ = envs.step(action.numpy())
            done = np.logical_or(term, trunc)
            log_probs.append(logp)
            values.append(value)
            entropies.append(ent)
            rewards.append(torch.as_tensor(r, dtype=torch.float32))
            dones.append(torch.as_tensor(done, dtype=torch.float32))
            obs = torch.as_tensor(next_obs, dtype=torch.float32)

        with torch.no_grad():
            _, next_value = model.forward(obs)

        rewards = torch.stack(rewards)
        values_t = torch.stack(values)
        dones_t = torch.stack(dones)
        log_probs = torch.stack(log_probs)
        entropies = torch.stack(entropies)

        adv, returns = compute_gae_torch(rewards, values_t, dones_t,
                                         next_value, gamma, lam)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)  # normalize

        actor_loss = -(log_probs * adv.detach()).mean()
        critic_loss = F.mse_loss(values_t, returns.detach())
        entropy_loss = -entropies.mean()
        loss = actor_loss + vf_coef * critic_loss + ent_coef * entropy_loss

        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        opt.step()

    return model
```

A few details that matter and that beginners routinely get wrong. **Advantage normalization** (`adv = (adv - adv.mean()) / (adv.std() + 1e-8)`) standardizes the actor's learning signal each batch, which stabilizes the gradient scale enormously — leave it out and CartPole becomes flaky. **Detaching the advantage** in the actor loss (`adv.detach()`) is essential: the advantage is a target for the actor, not something we backprop the policy gradient through. **Detaching the returns** in the critic loss is the semi-gradient TD update. **Gradient clipping** (`clip_grad_norm_`) caps the occasional exploding gradient when the critic is badly wrong early on. And `n_steps=5` with `num_envs=16` is the canonical A3C/A2C setting — a *short* rollout, because the critic's bootstrap lets you learn without waiting for full episodes.

### A2C via Stable-Baselines3

If you do not want to write the loop, SB3 gives you a battle-tested A2C in a few lines, with the same defaults:

```python
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

vec_env = make_vec_env("CartPole-v1", n_envs=16)
model = A2C(
    "MlpPolicy", vec_env,
    n_steps=5, gamma=0.99, gae_lambda=0.95,
    ent_coef=0.01, vf_coef=0.5, learning_rate=7e-4,
    max_grad_norm=0.5, verbose=1,
)
model.learn(total_timesteps=200_000)

mean_reward, std_reward = evaluate_policy(model, vec_env, n_eval_episodes=20)
print(f"CartPole-v1 A2C: {mean_reward:.1f} +/- {std_reward:.1f}")
# typical output: CartPole-v1 A2C: 498.4 +/- 4.2
```

The SB3 hyperparameter names map one-to-one onto our derivation: `gae_lambda` is the $\lambda$ from section 3, `vf_coef` is $c_1$, `ent_coef` is $c_2$, `n_steps` is the rollout length. That is the payoff of understanding the math — the config file stops being a pile of magic numbers and becomes a set of knobs you know how to turn.

## 6. A3C: asynchronous advantage actor-critic

A3C, the original (Mnih et al. 2016, "Asynchronous Methods for Deep Reinforcement Learning"), replaces synchronous batching with **asynchronous parallelism**. You run $K$ worker threads, each with its own copy of the environment and a *local* copy of the network. Each worker independently collects a short rollout, computes its own gradients, and then *asynchronously applies those gradients to a shared global network* — without waiting for the other workers. After pushing its gradients, the worker pulls the latest global parameters and continues. Figure 5 shows the structure: a central parameter server holding the global $\theta$, surrounded by workers that each loop "pull params, roll out, push gradients."

![Graph of three asynchronous A3C workers each holding local parameters and pushing gradients into a shared global parameter server](/imgs/blogs/actor-critic-a2c-a3c-5.png)

### Why asynchrony works without a replay buffer

DQN needed a replay buffer to decorrelate consecutive, highly-correlated samples (off-policy experience replay). A3C achieves the same decorrelation *for free* through parallelism: at any instant, the different workers are exploring different parts of the state space in different episodes, so the stream of gradients hitting the global network is naturally diverse. This is why A3C is on-policy and buffer-free yet still stable — the decorrelation comes from spatial diversity across workers rather than temporal diversity from a buffer.

### Why staleness is surprisingly OK

The obvious worry with asynchronous updates is **staleness**: a worker computes gradients against parameters $\theta_{\text{old}}$, but by the time it applies them, other workers have already moved the global parameters to $\theta_{\text{new}}$. So you are applying a gradient computed for the wrong point. In principle this is a bias. In practice, with a small number of workers (the paper used 16) and short rollouts, the parameters drift only slightly between pull and push, so the staleness is tiny and the noise it adds is dominated by the variance reduction from parallelism. The empirical result in the paper is striking: A3C trained on a *single 16-core CPU* matched or beat GPU-based DQN on Atari, in roughly *half the wall-clock time*, with no GPU at all. The asynchrony let them exploit cheap CPU parallelism that DQN's replay buffer could not.

### The Hogwild! result: why lock-free updates do not corrupt training

The deeper theoretical license for A3C's lock-free asynchronous writes comes from **Hogwild!** (Niu, Recht, Ré, Wright, 2011), a result about stochastic gradient descent that predates A3C and that the A3C authors lean on explicitly. The Hogwild! setting is multiple threads all running SGD on the *same* parameter vector in shared memory, with **no locks at all** — threads read and write parameters concurrently, so an individual update may be based on a partially-stale read and may be partially overwritten by another thread mid-write. Naively this looks like it should corrupt the optimization. Hogwild! proves that when the updates are **sparse** (each gradient touches only a small subset of the coordinates, so two threads rarely collide on the same parameter) and the staleness is **bounded** (a thread's parameters are never more than $\tau$ updates behind the global state, where $\tau$ is on the order of the number of threads), lock-free SGD converges at essentially the same rate as the serial version — the lost work from collisions is asymptotically negligible. The key quantities are the *collision probability* (low when gradients are sparse) and the *delay bound* $\tau$ (small when there are few workers and short rollouts).

A3C is Hogwild! applied to RL. The bounded-staleness condition holds because the paper uses only ~16 workers doing short 5-step rollouts, so between a worker's parameter pull and its gradient push, the global parameters have moved by at most a handful of other workers' small steps — $\tau$ is tiny. The sparsity condition is weaker for dense neural-network gradients than for the sparse linear models Hogwild! originally targeted, which is part of why A3C in practice tolerates the staleness *empirically* more than it is guaranteed to *theoretically*; but the bounded-staleness intuition is exactly right and is the reason the noise from asynchrony is dominated by the variance reduction from parallelism. In one sentence: **staleness is bounded by the worker count, the bound is small for few workers, and a small bounded bias is a good trade for free CPU parallelism and decorrelated gradients.**

There is a wall-clock-versus-sample-efficiency nuance here. A3C is fast in *wall-clock* time because the workers never idle waiting for each other. But it is not more *sample-efficient* than A2C — it uses about the same number of environment steps to reach a given score. A2C, by waiting and averaging, gets a cleaner gradient per update and runs beautifully on a single GPU with vectorized envs. Once GPUs became the standard substrate, A2C's "wait and batch" pattern fit the hardware better, which is the practical reason A2C largely displaced A3C. A3C's legacy is the *idea* — advantage actor-critic with parallel decorrelated workers — more than the specific asynchronous mechanism.

#### Worked example: bounding A3C staleness on a 16-worker run

Make the staleness concrete. Run A3C with $K=16$ workers, each doing $n=5$-step rollouts, on an environment where one rollout (5 env steps plus a forward and backward pass) takes about 10 ms of wall-clock per worker. In that 10 ms window, the other 15 workers each push roughly one gradient update, so by the time our worker finishes its rollout and goes to push, the global parameters have absorbed on the order of 15 other small SGD steps. With a learning rate of $7\times10^{-4}$ and gradient norms clipped to 0.5, each of those steps moves any given parameter by at most $\sim 3.5\times10^{-4}$ in magnitude, so the total drift between our worker's pull and push is bounded by roughly $15 \times 3.5\times10^{-4} \approx 5\times10^{-3}$ per parameter — a fraction of a percent for typical weight scales. The gradient our worker computed was "for" parameters that have since moved by half a percent. That is the staleness, quantified: it is real, it is a bias, and it is small enough that the decorrelation benefit from 16 independent explorers swamps it. Crank $K$ to 256 workers, though, and the drift grows roughly linearly — which is precisely why A3C does not scale to hundreds of workers without the staleness starting to bite, and why the synchronous A2C design (zero staleness by construction) ultimately scaled better on big machines.

### A3C worker pseudocode (real PyTorch with shared memory)

PyTorch makes the global-shared-parameters pattern concrete via `share_memory()` and `torch.multiprocessing`. The core worker loop:

```python
import torch.multiprocessing as mp

def worker(global_model, optimizer, env_id, gamma=0.99, lam=0.95, n_steps=5):
    local_model = ActorCritic(obs_dim, n_actions)
    env = gym.make(env_id)
    obs, _ = env.reset()
    while not stop_flag.value:
        # 1. sync local params from the global model
        local_model.load_state_dict(global_model.state_dict())

        log_probs, values, rewards, dones, ents = [], [], [], [], []
        for _ in range(n_steps):
            x = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            action, logp, ent, value = local_model.act(x)
            obs, r, term, trunc, _ = env.step(action.item())
            log_probs.append(logp); values.append(value); ents.append(ent)
            rewards.append(r); dones.append(float(term or trunc))
            if term or trunc:
                obs, _ = env.reset()

        # 2. compute advantages locally (GAE), then the combined loss
        loss = compute_a3c_loss(log_probs, values, rewards, dones, ents,
                                local_model, obs, gamma, lam)

        # 3. compute LOCAL gradients, then push them onto the GLOBAL model
        optimizer.zero_grad()
        loss.backward()
        for lp, gp in zip(local_model.parameters(), global_model.parameters()):
            gp._grad = lp.grad           # transplant local grads onto global params
        optimizer.step()                 # optimizer holds GLOBAL params

# launch: global_model.share_memory() then spawn K worker processes
```

The key line is the gradient transplant: each worker computes gradients on its *local* network, then copies those gradients onto the *global* network's parameters and steps the shared optimizer. Because the optimizer wraps the global (shared-memory) parameters, every worker's `optimizer.step()` mutates the same tensors — that is the asynchronous update. The `Hogwild!`-style lock-free writes are exactly the "staleness is OK" bet in action.

## 7. Comparing the family: A2C, A3C, PPO, SAC

These four algorithms form the modern on-policy-and-friends landscape, and knowing where each sits saves you from picking the wrong tool.

![Matrix comparing A2C, A3C, PPO and SAC across parallelism, variance, stability, continuous-action support, and wall-clock speed](/imgs/blogs/actor-critic-a2c-a3c-6.png)

| Algorithm | On/Off-policy | Parallelism | Variance | Stability | Continuous actions | Sample efficiency | Wall-clock |
|---|---|---|---|---|---|---|---|
| A2C | On-policy | Synchronous vector envs | Medium | Good | Yes (Gaussian head) | Low | Fast on GPU |
| A3C | On-policy | Asynchronous CPU workers | Medium | OK (staleness) | Yes | Low | Very fast on multi-core CPU |
| PPO | On-policy | Synchronous vector envs | Low (clipping + GAE) | Excellent | Yes | Low-medium | Medium |
| SAC | Off-policy | Replay buffer | Low | Excellent | Best (max-entropy) | High | Slow per-step, few steps needed |

It is just as useful to compare the three *on-policy* steps of the lineage directly against their common ancestor REINFORCE, since that axis isolates what each refinement actually bought:

| Algorithm | Parallelism | Variance reduction | On-policy | Stability | Sample efficiency |
|---|---|---|---|---|---|
| REINFORCE | None (single trajectory) | Baseline only (optional) | Yes | Poor (high variance, oscillates) | Lowest |
| A2C | Synchronous vector envs | Critic baseline + GAE | Yes | Good | Low |
| A3C | Asynchronous CPU workers | Critic baseline + GAE + worker decorrelation | Yes | OK (bounded staleness) | Low |
| PPO | Synchronous vector envs | Critic + GAE + clipped surrogate | Yes | Excellent | Low–medium |

Read left to right and the story is a single accumulating idea. REINFORCE has the right objective but no variance control beyond an optional baseline, so it oscillates. A2C adds the *learned critic* (advantages instead of returns), GAE (a tunable bias-variance dial), and *parallel decorrelation* (many envs instead of one trajectory) — three independent variance reductions stacked on the same gradient. A3C swaps synchronous batching for asynchronous workers, trading a little staleness bias for wall-clock speed and CPU scalability. PPO keeps everything A2C has and adds the clipped surrogate plus multi-epoch reuse, buying the stability and the modest sample-efficiency gain that make it the default. None of them changes *what* is optimized — all four optimize the same policy-gradient objective — they only change *how well the gradient is estimated and how safely the step is taken.* That is the whole on-policy family in one paragraph.

The throughline: **A2C and A3C are the simplest members and the conceptual base.** PPO is "A2C plus a clipped surrogate objective and multiple epochs per rollout," which lets it safely reuse each batch several times and bounds the policy update so it cannot take a catastrophic step — that is why PPO is the stability champion of the on-policy family and the default for most new projects. SAC abandons on-policy entirely: it is an off-policy, max-entropy actor-critic with a replay buffer and twin Q-critics, which makes it dramatically more *sample-efficient* (it reuses old data) and the best choice for expensive continuous-control like real robots, at the cost of more moving parts and slower per-step compute.

Here is the lineage stated plainly: actor-critic is the architecture; A2C/A3C are the first deep instantiations; PPO is the hardened on-policy production version; SAC is the off-policy cousin. If you understand the GAE and advantage math in this post, PPO is a small delta away — it swaps the plain policy-gradient loss for the clipped surrogate and reuses the rollout for a few epochs, but the critic, GAE, and entropy machinery are identical.

It is worth naming the *one* thing that breaks when you cross from on-policy to off-policy, because it explains the whole right-hand column of the table. Everything in this post — the policy gradient theorem, the baseline proof, GAE — assumes the data was generated by the *current* policy $\pi_\theta$. The expectations are all $\mathbb{E}_{\pi_\theta}[\cdot]$. The moment you reuse old data from a stale policy (a replay buffer), those expectations are over the *wrong* distribution and the gradient is biased. On-policy methods (REINFORCE, A2C, A3C, PPO) pay for correctness by throwing data away after one rollout; that is why they are sample-inefficient. PPO's clipped surrogate is best understood as a *controlled* violation — it lets you reuse a rollout for a few epochs by bounding how far the policy can drift from the one that generated the data, so the importance-weighting error stays small. SAC goes all the way to a fully off-policy correction (a learned Q-function plus a replay buffer plus the max-entropy objective), which recovers sample efficiency at the cost of the extra machinery. So the single conceptual axis underneath the whole family is: *how much do you let the data-generating policy diverge from the policy you are updating, and what do you pay to stay correct when it does?* A2C says "not at all"; PPO says "a little, with a clip"; SAC says "as much as you like, with a Q-function." Once you see that axis, every algorithm in the table has an obvious place on it.

## 8. Case studies and measured results

### A3C on Atari (Mnih et al. 2016)

The headline result from the A3C paper: across 57 Atari games, A3C running on a single 16-core CPU surpassed the then-state-of-the-art (DQN, Gorila, and others that used GPUs and replay) on a majority of games, while training in roughly half the wall-clock time. On Pong specifically, A3C goes from the random-policy score of around $-21$ (losing every point) to near the maximum of $+21$ within hours. Figure 7 sketches that learning curve: a long stretch near $-21$, then a fairly sharp climb through zero, then convergence near the ceiling. The breakthrough was not a higher final score on most games but the *efficiency* — getting there on commodity CPU hardware without the memory cost of a replay buffer.

![Timeline of an A3C learning curve on Atari Pong climbing from a random losing score to a near-maximal stabilized score](/imgs/blogs/actor-critic-a2c-a3c-7.png)

#### Worked example: reading the Pong curve honestly

When you train A3C on Pong yourself, the return curve has a characteristic and initially alarming shape. For the first several million frames it sits flat near $-21$ and looks completely dead — the agent has not yet discovered that moving the paddle toward the ball is good. Novices kill the run here thinking it is broken. Then, somewhere around 5-10 million frames, it breaks out: the score rises steeply through $-5$, crosses zero around 15 million frames, and reaches the high teens by 20-40 million frames, finally stabilizing near $+20$. The lesson, which I have re-learned the expensive way, is that **sparse-reward Atari games have a long flat exploration phase before the breakout**, and a flat curve is not the same as a failed run. Check that policy entropy is still healthy (the agent is still exploring) and that the critic loss is decreasing before you pull the plug. Premature stopping is the number-one waste of compute on these environments.

### A2C versus DQN sample efficiency

On Atari, the comparison is nuanced and worth stating carefully. DQN, being off-policy with a large replay buffer, is generally **more sample-efficient** — it squeezes more learning out of each environment frame by replaying it many times. A2C/A3C, being on-policy, throw each batch away after one (A2C) update, so they need *more frames* to reach the same score on many games. But A2C/A3C are far more **wall-clock efficient** when you have parallel envs, because they have no replay buffer to manage and they vectorize cleanly. The practical takeaway: if environment steps are cheap (fast simulator) and you have many parallel envs, A2C's wall-clock speed wins; if steps are expensive, the sample efficiency of an off-policy method like DQN or SAC matters more.

### SB3 A2C on CartPole and LunarLander

For reproducible numbers on the standard control suite, the SB3 A2C baselines are a good reference. On **CartPole-v1**, A2C with 16 envs reliably reaches the maximum return of 500 within roughly 100-200k timesteps — a few minutes on a laptop. On **LunarLander-v2**, A2C reaches the "solved" threshold of mean return 200 within a few hundred thousand to a couple million timesteps, though it is noticeably noisier than PPO on this environment and benefits from a slightly larger `n_steps` and careful `ent_coef`. The honest comparison: PPO on LunarLander is both more stable and reaches a higher final mean return than A2C with default hyperparameters, which is exactly why PPO is the more common default — A2C's appeal is its simplicity and its role as the conceptual stepping stone, not its raw performance.

```bash
# Reproduce the CartPole A2C baseline with the SB3 RL Zoo
python -m rl_zoo3.train --algo a2c --env CartPole-v1 \
    --n-timesteps 200000 --eval-freq 10000 --eval-episodes 20
# expected eval: mean_reward ~ 500 by the end of training
```

## 9. Hyperparameters and a debugging checklist

A2C has few hyperparameters, but each one has a clear role tied to the math above. Here is the table I keep next to me, with the effect each knob has and a sane range.

| Hyperparameter | Symbol | Effect | Typical range |
|---|---|---|---|
| Number of parallel envs | $N$ | Decorrelation and throughput; more envs = cleaner gradient | 8–64 |
| Rollout length | `n_steps` | Bias-variance of the return horizon; longer = less bootstrap | 5–128 |
| Discount | $\gamma$ | Effective planning horizon $\approx 1/(1-\gamma)$ | 0.99 (0.999 for long tasks) |
| GAE lambda | $\lambda$ | Advantage bias-variance dial | 0.9–0.99 (default 0.95) |
| Value loss coef | $c_1$ | Critic-vs-actor gradient balance | 0.25–1.0 |
| Entropy coef | $c_2$ | Exploration pressure; too low = premature collapse | 0.0–0.02 |
| Learning rate | $\alpha$ | Actor/critic step size; too high = the actor outruns the critic | 1e-4–1e-3 |
| Max grad norm | — | Caps exploding gradients early in training | 0.5–5.0 |

Notice how each knob maps onto a concept we derived: `gae_lambda` is the section-3 dial, $c_1$ is the section-4 interference balance, $c_2$ is the section-4 entropy bonus, and the learning rate is the section-2 two-timescale constraint. There are no magic numbers here once you know what each one is doing.

When training misbehaves, work this checklist in order — it catches the overwhelming majority of A2C/A3C failures:

- **Return is flat and the policy entropy is collapsing fast.** The actor went deterministic before it explored. Raise `ent_coef`, lower the learning rate.
- **Return is flat but entropy is healthy.** You may be in the legitimate long exploration phase (especially Atari) — check that the critic loss is decreasing before concluding it is broken.
- **Return oscillates wildly and never stabilizes.** The actor is outrunning the critic. Lower the learning rate, raise `vf_coef`, add or tighten gradient clipping.
- **Critic loss explodes to huge numbers.** You probably forgot to detach the value target, or your `done` mask is wrong and value is leaking across episode boundaries. Re-check the GAE `mask` line.
- **Everything looks right but learning is slow.** You forgot advantage normalization, or `n_steps` is too short for the reward sparsity of your task.

I have personally lost days to each of these. The reason they are worth memorizing is that the symptoms are distinctive — flat-with-collapsing-entropy looks nothing like oscillating-return — so once you have seen them, diagnosis takes seconds rather than a re-read of your whole training loop.

## 10. When to use A2C/A3C (and when not to)

Be decisive here, because the wrong default wastes weeks.

**Reach for A2C when** you want a simple, transparent on-policy baseline to understand a new environment, you have many cheap parallel environments (fast simulator), and you value implementation simplicity and debuggability over squeezing out the last few percent of performance. A2C is also an excellent teaching and debugging tool: because there is no clipping, no replay buffer, and no multi-epoch reuse, when something breaks you can actually find it.

**Reach for A3C when** you are CPU-bound with many cores and no GPU, or you specifically want lock-free asynchronous parallelism. In 2026 this is rare — GPUs are ubiquitous and A2C on a GPU is usually the better engineering choice — so A3C is mostly of historical and conceptual importance now.

**Prefer PPO when** you want a production-grade on-policy algorithm with strong stability — which is *most of the time* for on-policy work. PPO's clipped objective lets it reuse each rollout for several epochs (better sample efficiency than A2C) while bounding the update size (better stability). If you were going to use A2C and you do not have a specific reason not to, use PPO instead; it is A2C's strictly-better successor for most tasks.

**Prefer DQN or its variants when** you have a discrete action space and expensive environment steps, so off-policy sample efficiency from a replay buffer matters, and you do not need a stochastic policy.

**Prefer SAC (or TD3) when** you have continuous action spaces and sample efficiency is paramount — robotics, expensive simulators. SAC's off-policy replay and max-entropy objective make it the sample-efficiency leader for continuous control.

**Do not use any model-free RL** — A2C included — if you have a cheap, accurate model of the environment dynamics. For tabular problems with a known model, value iteration or policy iteration is exact and far faster. For problems where you can plan, model-based methods or even classical control will crush model-free RL on sample efficiency. RL earns its keep when the dynamics are unknown and you must learn from interaction.

## 11. Key takeaways

- **The critic exists to reduce variance, not to add a second objective.** It learns $V(s)$ so the actor can be trained on advantages $A(s,a)$ — "how much better than average" — instead of raw, noisy returns.
- **The baseline subtraction is provably unbiased.** Subtracting any state-only function $b(s)$ leaves the policy gradient's expectation unchanged, because the expected score function is zero; it only changes variance.
- **The TD error $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$ is a one-step, bootstrapped advantage estimate** — low variance, some bias from the imperfect critic.
- **GAE is a single elegant formula**, $\hat{A}_t = \sum_l (\gamma\lambda)^l \delta_{t+l}$, that interpolates between TD(0) ($\lambda{=}0$, low variance/high bias) and Monte Carlo ($\lambda{=}1$, high variance/low bias). Default $\lambda = 0.95$.
- **A2C is the synchronous, simpler, GPU-friendly version of A3C**, and it generally works as well or better — the asynchrony was never the secret; the parallelism-for-decorrelation was.
- **Asynchronous staleness is tolerable** because with few workers and short rollouts the parameters barely move between gradient pull and push; A3C's value was matching GPU-DQN on cheap CPUs.
- **Always normalize advantages, detach targets, clip gradients, and add an entropy bonus** — these four practical details are the difference between flaky and reliable training.
- **PPO is A2C's strictly-better successor for most on-policy work**; A2C/A3C are the conceptual foundation you build PPO and SAC understanding on top of.

## 12. Further reading

- Mnih, Badia, Mirza, Graves, Lillicrap, Harley, Silver, Kavukcuoglu, "Asynchronous Methods for Deep Reinforcement Learning" (2016) — the A3C paper, and the source of the A2C-as-synchronous-variant observation.
- Schulman, Moritz, Levine, Jordan, Abbeel, "High-Dimensional Continuous Control Using Generalized Advantage Estimation" (2016) — the GAE derivation in full.
- Sutton & Barto, "Reinforcement Learning: An Introduction" (2nd ed., 2018) — chapters 13 (policy gradient) and 6 (TD learning) for the foundations.
- Schulman, Wolski, Dhariwal, Radford, Klimov, "Proximal Policy Optimization Algorithms" (2017) — the natural next step after A2C.
- Stable-Baselines3 documentation, the `A2C` and `PPO` API references and the RL Zoo tuned hyperparameters — for reproducible baselines.
- Within this series: start from the taxonomy in [a unified map of reinforcement learning](/blog/machine-learning/reinforcement-learning/reinforcement-learning-a-unified-map), and bring it all together in [the reinforcement learning playbook](/blog/machine-learning/reinforcement-learning/the-reinforcement-learning-playbook).
