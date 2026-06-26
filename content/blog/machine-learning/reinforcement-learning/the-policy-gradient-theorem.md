---
title: "The Policy Gradient Theorem: From REINFORCE to Variance Reduction"
date: "2026-06-22"
publishDate: "2026-06-22"
description: "Derive the policy gradient theorem from first principles, build REINFORCE in PyTorch, and learn exactly why baselines and the advantage function tame its punishing variance."
tags:
  [
    "reinforcement-learning",
    "deep-learning",
    "policy-gradient",
    "actor-critic",
    "machine-learning",
    "pytorch",
    "markov-decision-process",
    "variance-reduction",
  ]
category: "machine-learning"
subcategory: "Reinforcement Learning"
author: "Hiep Tran"
featured: true
readTime: 55
image: "/imgs/blogs/the-policy-gradient-theorem-1.png"
---

A CartPole agent with random weights falls over in about eight steps. The pole tips, the cart lurches the wrong way, and the episode ends with a return of 8 out of a possible 500. You could spend a week hand-tuning a controller, or you could let the agent improve itself: collect a few episodes, notice which actions tended to keep the pole up, and shift the policy a little toward those actions. Do that a few hundred times and the same network that fell in eight steps now balances for the full 500. The procedure that makes this work — turning sampled experience directly into a gradient on the policy's parameters — is the **policy gradient**, and the result that justifies it is the **policy gradient theorem**.

This post is about that theorem and everything that hangs off it. We will derive the gradient from first principles using the log-derivative trick, build the REINFORCE algorithm in PyTorch and run it on CartPole, and then confront the single biggest practical problem with policy gradients: their variance is brutal. A naive REINFORCE update can have a gradient estimate whose noise swamps its signal, so training crawls or diverges. The fix — subtracting a **baseline** and arriving at the **advantage function** — is one of the most important ideas in all of RL, and it follows from the same theorem with almost no extra machinery. Figure 1 shows the loop we are going to build and then sharpen.

![The policy gradient training loop collecting a trajectory computing returns scoring actions and taking a gradient step](/imgs/blogs/the-policy-gradient-theorem-1.png)

By the end you will be able to: derive $\nabla_\theta J(\theta) = \mathbb{E}\big[\nabla_\theta \log \pi_\theta(a\mid s)\, Q^\pi(s,a)\big]$ yourself; explain why any action-independent baseline leaves that gradient unbiased; implement REINFORCE-with-baseline that takes CartPole from an average return near 8 to near 490; and decide when a policy gradient is the right tool versus when a value-based method like Q-learning would crush it. This is the foundation of the entire policy-optimization family — actor-critic, A2C, TRPO, PPO, GRPO — so it is worth getting exactly right. Everything in this series ties back to the same spine: an agent interacts with an environment, collects rewards, and updates a policy. Policy gradients are simply the answer to "what if we differentiate the expected reward with respect to the policy directly?"

A note on how to read this post. The mathematics is not decoration — each step in the derivation removes a specific obstacle, and the payoff at the end is a single equation you can implement in twenty lines of PyTorch. If you have ever trained a classifier with cross-entropy loss, you already know nine-tenths of the machinery; the policy gradient is cross-entropy loss with a per-example weight that says "how good was this outcome." Hold that analogy as you read. By the time we reach the code in Section 4 it will feel inevitable rather than magical, and by Section 8 you will understand why the obvious version is too noisy to use and what one extra subtraction does to fix it.

## 1. Why optimize the policy directly?

Most introductory RL starts with **value-based methods**. You learn a value function — $Q(s,a)$, the expected return from taking action $a$ in state $s$ and acting well thereafter — and then act greedily with respect to it: $\pi(s) = \arg\max_a Q(s,a)$. Q-learning and DQN are the canonical examples, and they are superb when they fit. DQN learned to play dozens of Atari games from pixels in 2015 using exactly this recipe.

But that $\arg\max$ hides an assumption: you can enumerate the actions and pick the best one. For a discrete action space of four buttons, fine. For a robot arm with seven continuous joint torques, the action space is $\mathbb{R}^7$, and $\arg\max_a Q(s,a)$ is itself a nontrivial optimization problem you would have to solve *inside every single environment step*. Methods like DDPG and SAC bolt a separate actor network on top of the critic precisely to approximate that inner maximization — which is, not coincidentally, a policy gradient method wearing a value-based hat.

There are three places value-based methods struggle, and each is a reason to optimize the policy directly:

- **Continuous or high-dimensional actions.** When actions live in $\mathbb{R}^d$ you cannot take an $\arg\max$ over a table. A parameterized policy $\pi_\theta(a\mid s)$ that outputs the parameters of a distribution (say the mean and variance of a Gaussian over torques) sidesteps this entirely.
- **Stochastic optimal policies.** In many problems the best policy is genuinely random. In a game with imperfect information — rock-paper-scissors, poker bluffing, a pursuit game where you must be unpredictable — any deterministic policy is exploitable. Value-based methods produce deterministic greedy policies; a policy gradient can learn a stochastic one directly because the policy *is* a distribution.
- **Smooth policy improvement.** A tiny change in $Q$ values can flip the $\arg\max$ and cause a discontinuous, destabilizing jump in the policy. Policy gradients move the parameters $\theta$ smoothly, so small steps produce small policy changes — which turns out to matter enormously for stability and is the whole reason trust-region methods like TRPO and PPO exist.

The trade-off is not free. Value-based methods can reuse old data (they are off-policy) and are often dramatically more sample-efficient. Vanilla policy gradients are on-policy — they need fresh samples from the current policy for every update — and, as we will spend half this post establishing, they are high-variance. The unified-map post in this series (`/blog/machine-learning/reinforcement-learning/reinforcement-learning-a-unified-map`) places both families on the same taxonomy; here we focus on making the policy-gradient half work.

It helps to be concrete about what a *parameterized policy* even is, because it is the object we differentiate and it is easy to wave at without pinning down. A policy $\pi_\theta(a\mid s)$ is a conditional probability distribution over actions, with parameters $\theta$ — in deep RL, the weights of a neural network. For discrete actions, the network maps a state to a vector of logits, and we pass those through a softmax to get a categorical distribution; the parameters $\theta$ are the network weights, and changing them reshapes the distribution. For continuous actions, the network typically outputs the mean (and sometimes the log-standard-deviation) of a Gaussian, and we sample the action from that Gaussian. In both cases the policy is *differentiable in $\theta$* — nudging the weights smoothly changes the probabilities — which is the property the whole theorem will exploit. A value-based method has no such directly-differentiable policy object; its "policy" is the discontinuous $\arg\max$ of a learned value function, and you cannot take a clean gradient of an $\arg\max$.

This distinction also explains why policy gradients handle exploration differently. A value-based agent explores by adding noise on top of a greedy choice — epsilon-greedy picks a random action with probability $\epsilon$ and the greedy one otherwise. The exploration is bolted on, external to the policy. A stochastic policy explores *intrinsically*: because $\pi_\theta(a\mid s)$ assigns nonzero probability to multiple actions, the agent naturally tries them in proportion to their current probability, and the policy gradient itself adjusts those probabilities. Exploration and exploitation live in the same object. The downside, which we will meet in the convergence section, is that nothing stops the policy from collapsing to near-deterministic too early and killing its own exploration — which is why entropy bonuses exist.

The object we want to maximize is the **expected return** of the policy. For an episodic task with trajectories $\tau = (s_0, a_0, s_1, a_1, \dots, s_T)$, define the return of a trajectory as the discounted sum of rewards $R(\tau) = \sum_{t=0}^{T} \gamma^t r_t$, and the objective as

$$
J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\big[ R(\tau) \big] = \sum_\tau P(\tau\mid\theta)\, R(\tau).
$$

Here $P(\tau\mid\theta)$ is the probability of the whole trajectory under policy $\pi_\theta$ and the environment dynamics. We want $\nabla_\theta J(\theta)$ so we can do gradient *ascent*: $\theta \leftarrow \theta + \alpha \nabla_\theta J(\theta)$. The trouble is that $J(\theta)$ is an expectation over trajectories, the trajectory distribution depends on $\theta$, and the environment dynamics inside $P(\tau\mid\theta)$ are unknown. Differentiating through that looks hopeless. The policy gradient theorem is the result that makes it not just possible but easy.

One framing that helps before we dive into the calculus: there are two distinct difficulties tangled together in $\nabla_\theta J$. The first is that we are differentiating an expectation whose *sampling distribution itself depends on $\theta$* — the very thing we vary changes what we average over. The second is that the environment dynamics are a black box we cannot differentiate through. The log-derivative trick in the next section dissolves the first difficulty, and the trajectory-factorization in Section 3 dissolves the second. Watching both fall away is the whole intellectual content of the theorem, and once you see it the algorithm writes itself.

## 2. The log-derivative trick

The entire derivation rests on one elementary calculus identity. For any positive, differentiable function $p(x; \theta)$:

$$
\nabla_\theta p(x;\theta) = p(x;\theta)\,\frac{\nabla_\theta p(x;\theta)}{p(x;\theta)} = p(x;\theta)\,\nabla_\theta \log p(x;\theta).
$$

The last step is just the chain rule applied to $\log$: $\nabla_\theta \log p = \frac{\nabla_\theta p}{p}$. This is the **log-derivative trick** (also called the *score function* trick — $\nabla_\theta \log p$ is known in statistics as the score). It looks trivial, but it is the hinge of the whole field, because it converts a gradient of a probability into the probability times a gradient of a log-probability. Figure 2 traces the path from the objective to a sampleable estimator.

![Graph showing the log derivative trick path from objective to trajectory probability to score function to a sampleable expectation](/imgs/blogs/the-policy-gradient-theorem-2.png)

Why does this matter? Because whenever you differentiate an expectation, the trick lets you push the gradient *inside* and recover another expectation. Start from the objective:

$$
\nabla_\theta J(\theta) = \nabla_\theta \sum_\tau P(\tau\mid\theta)\, R(\tau) = \sum_\tau \nabla_\theta P(\tau\mid\theta)\, R(\tau).
$$

The reward $R(\tau)$ does not depend on $\theta$, so it comes out of the gradient. Now apply the log-derivative trick to $\nabla_\theta P(\tau\mid\theta)$:

$$
\nabla_\theta J(\theta) = \sum_\tau P(\tau\mid\theta)\, \nabla_\theta \log P(\tau\mid\theta)\, R(\tau) = \mathbb{E}_{\tau \sim \pi_\theta}\big[ \nabla_\theta \log P(\tau\mid\theta)\, R(\tau) \big].
$$

That is already a profound improvement: the gradient of the expected return *is itself an expectation*, which means we can estimate it by sampling trajectories and averaging. We have turned an analytic gradient we could not compute into a Monte Carlo estimate we can. The price — and it is the price we pay for the rest of the post — is that Monte Carlo estimates have variance.

Step back and notice *what just happened to the troublesome $\theta$-dependence of the sampling distribution*. Before the trick, the gradient $\nabla_\theta \sum_\tau P(\tau\mid\theta) R(\tau)$ had $\theta$ buried inside the distribution we sum over, so we could not write it as "an average of something over trajectories." After the trick, we have $\mathbb{E}_{\tau\sim\pi_\theta}[\,\cdot\,]$ — an honest expectation over the current policy's trajectories, of a quantity ($\nabla_\theta\log P \cdot R$) that we can evaluate sample by sample. The $\theta$-dependence of the *distribution* has been converted into a $\theta$-dependence of the *integrand*. That conversion is the entire reason we can estimate the gradient by simply running the policy and averaging. It is the single most important conceptual move in policy-based RL, and it is worth being able to reproduce it on a whiteboard from memory.

It is worth dwelling on *why* the trick is the only move available, because beginners often wonder whether there is a slicker route that avoids the variance. There is a competing technique — the **reparameterization trick**, the engine behind variational autoencoders and the SAC algorithm — which rewrites the random action as a deterministic function of $\theta$ and an external noise variable, $a = f_\theta(s, \epsilon)$, and then differentiates straight through $f_\theta$. Reparameterization is lower-variance when it applies, but it demands two things the general RL setting often lacks: a continuous, reparameterizable action distribution, and a *differentiable* reward or value with respect to the action. In the general case the reward comes from a black-box environment with no gradient — you call `env.step(a)` and get back a number, not a derivative. The score-function estimator needs *none* of that. It only needs to be able to *sample* actions and *evaluate* $\log \pi_\theta(a\mid s)$, which any parameterized policy gives you for free. That generality is exactly why the high variance is the accepted cost: it is the price of working with a black-box, non-differentiable environment. Keep this contrast in mind — when we reach continuous-control methods later in the series, SAC's lower variance comes precisely from being able to reparameterize where REINFORCE cannot.

There is also a deep statistical fact hiding in the score function that we will use twice more in this post, so it is worth isolating now. The expected score of any properly normalized distribution is zero: $\mathbb{E}_{a\sim\pi_\theta}[\nabla_\theta \log \pi_\theta(a\mid s)] = 0$. The proof is one line — $\mathbb{E}[\nabla\log\pi] = \sum_a \pi \frac{\nabla\pi}{\pi} = \sum_a \nabla\pi = \nabla\sum_a \pi = \nabla 1 = 0$ — and it says something concrete: on average, over actions drawn from the policy itself, the score points nowhere. Only when you *weight* the score by something correlated with the outcome (the return) does it acquire a direction. This zero-mean property is exactly what makes baseline subtraction free in Section 6, and it is exactly why the policy gradient does not blow up the policy in a fixed direction when all returns are equal. File it away.

One more subtlety: the expectation we just derived is taken over the *current* policy's trajectory distribution. That is what makes vanilla policy gradients on-policy. The instant we update $\theta$, the distribution $P(\tau\mid\theta)$ shifts, and our old samples were drawn from the old distribution — they no longer give an unbiased estimate of the new gradient. This is why REINFORCE throws away its data after a single update and collects fresh episodes. Importance sampling can partially reuse old data by reweighting, $\frac{\pi_{\text{new}}(a\mid s)}{\pi_{\text{old}}(a\mid s)}$, and that reweighting ratio is precisely the object PPO clips to keep the policy from moving too far — but that is a later post. For now, on-policy and one-update-per-batch is the contract.

## 3. Deriving the policy gradient theorem

We still have $\nabla_\theta \log P(\tau\mid\theta)$ in the expression, and $P(\tau\mid\theta)$ contains the unknown environment dynamics. Watch what happens when we expand it. The probability of a trajectory factorizes into the initial state distribution, the policy at each step, and the transition dynamics:

$$
P(\tau\mid\theta) = \rho_0(s_0) \prod_{t=0}^{T-1} \pi_\theta(a_t\mid s_t)\, P(s_{t+1}\mid s_t, a_t).
$$

Take the log, which turns the product into a sum:

$$
\log P(\tau\mid\theta) = \log \rho_0(s_0) + \sum_{t=0}^{T-1} \Big[ \log \pi_\theta(a_t\mid s_t) + \log P(s_{t+1}\mid s_t, a_t) \Big].
$$

Now take the gradient with respect to $\theta$. Here is the magic: $\rho_0(s_0)$ and the transition probabilities $P(s_{t+1}\mid s_t, a_t)$ **do not depend on $\theta$** — they are properties of the environment, not the policy. Their gradients are zero. Only the policy terms survive:

$$
\nabla_\theta \log P(\tau\mid\theta) = \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t\mid s_t).
$$

This is why the policy gradient is **model-free**. We never needed to know the dynamics; they differentiated away. Substituting back:

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[ \left( \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t\mid s_t) \right) R(\tau) \right].
$$

This is one form of the policy gradient theorem. We can sharpen it. The current form multiplies the *sum of all* score functions by the *total* trajectory return $R(\tau)$. But an action at time $t$ cannot influence rewards collected *before* $t$ — causality forbids it. So weighting $\nabla_\theta \log \pi_\theta(a_t\mid s_t)$ by the full-trajectory return needlessly includes past rewards, which are pure noise with respect to $a_t$. Dropping them (the "reward-to-go" or "causality" trick, which is unbiased) gives

$$
\nabla_\theta J(\theta) = \mathbb{E}\left[ \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t\mid s_t)\, G_t \right], \quad G_t = \sum_{k=t}^{T} \gamma^{k-t} r_k.
$$

where $G_t$ is the return *from time $t$ onward*. Let us be careful about *why* dropping the past rewards leaves the estimator unbiased, because it is the same zero-mean-score argument from Section 2 applied locally. The full-trajectory return splits into rewards before $t$ and rewards from $t$ onward: $R(\tau) = R_{<t} + G_t$. When we multiply the score at time $t$ by the past reward $R_{<t}$ and take the expectation, the past rewards are determined by states and actions that are already fixed by the time we choose $a_t$, so conditioning on the history up to $t$ they are constants with respect to $a_t$, and $\mathbb{E}_{a_t}[\nabla\log\pi_\theta(a_t\mid s_t)\, R_{<t}] = R_{<t}\,\mathbb{E}_{a_t}[\nabla\log\pi_\theta] = R_{<t}\cdot 0 = 0$. The past-reward term vanishes in expectation, which means removing it changes the variance (favorably) but not the mean. This is the first concrete payoff of the zero-mean score property, and the baseline in Section 6 is the second.

### From the return-to-go to the action value

And in its most general statement, the expected reward-to-go given the state and action is exactly the action-value $Q^\pi(s_t, a_t)$, which yields the textbook form:

$$
\boxed{\;\nabla_\theta J(\theta) = \mathbb{E}_{s,a \sim \pi_\theta}\big[ \nabla_\theta \log \pi_\theta(a\mid s)\, Q^\pi(s,a) \big]\;}
$$

The substitution of $Q^\pi(s,a)$ for $G_t$ deserves its own justification, because it is exactly the link between the Monte Carlo form we can sample and the abstract theorem in textbooks. By definition $Q^\pi(s_t,a_t) = \mathbb{E}_\pi[G_t \mid s_t, a_t]$ — the action value *is* the expectation of the return-to-go, conditioned on the state and action, with the expectation taken over everything that happens afterward. So the Monte Carlo estimate $G_t$ is an **unbiased sample of $Q^\pi(s_t,a_t)$**: $\mathbb{E}[G_t \mid s_t,a_t] = Q^\pi(s_t,a_t)$ by construction. This is why REINFORCE, which uses the raw sampled $G_t$, estimates the *same* gradient as the textbook $Q^\pi$ form — it just uses a single noisy sample of $Q^\pi$ instead of its true value. The price of that single noisy sample is, once again, variance: a single rollout's $G_t$ can land far from its mean $Q^\pi(s_t,a_t)$, especially over long horizons. The bias, however, is zero. This unbiasedness is what lets us prove that REINFORCE, run for long enough with small enough steps, ascends the true objective — it is not optimizing a proxy.

Read the boxed equation intuitively. The score $\nabla_\theta \log \pi_\theta(a\mid s)$ points in the direction in parameter space that makes action $a$ *more likely* in state $s$. We weight it by $Q^\pi(s,a)$: how good that action was. Good actions (high $Q$) get pushed up; bad actions (low or negative weighted return) get pushed down. Sum over the states and actions you actually visited, and you have a gradient that increases the probability of high-return behavior. That is the whole idea, and it is exact — no approximation in the theorem itself. There is a subtle weighting hidden in the "sum over states you visited": states are visited in proportion to the *discounted state-visitation distribution* $d^\pi(s)$ induced by the policy, so the fully-written-out theorem is $\nabla_\theta J(\theta) = \sum_s d^\pi(s) \sum_a \nabla_\theta \pi_\theta(a\mid s)\, Q^\pi(s,a)$, or in integral form for continuous spaces $\nabla_\theta J(\theta) = \int_S d^\pi(s) \int_A \nabla_\theta \pi_\theta(a\mid s)\, Q^\pi(s,a)\,da\,ds$. The practical point is that you do not have to know $d^\pi(s)$ — you sample from it for free just by running the policy, because the states your rollouts actually visit *are* draws from $d^\pi$.

A common and useful framing is to compare this to **supervised learning with soft labels**. In supervised classification you minimize $-\sum \log \pi_\theta(y\mid x)$ for the true label $y$ — you push up the log-probability of the correct class with unit weight. The policy gradient is *weighted maximum likelihood*: it pushes up the log-probability of the action you took, but weighted by how good the outcome was. An action followed by a return of 50 gets fifty times the "label weight" of an action followed by a return of 1. So you can read REINFORCE as: treat the actions you sampled as labels, and do supervised learning on them, but scale each label's gradient by its return. This is why the implementation is so short — it is a cross-entropy loss with a per-sample weight. It is also why the variance is so bad: in supervised learning your labels are correct and fixed; here your "labels" are sampled from the very policy you are training, and their "weights" are noisy Monte Carlo returns. You are doing weighted maximum likelihood on a constantly shifting, noisily-labeled dataset that you generate yourself.

There is one more reading of the theorem that pays off later. Notice the gradient does not care about the *absolute* probability of an action, only its *log*-probability gradient. The score $\nabla_\theta \log \pi_\theta$ has a self-normalizing property: it automatically accounts for how concentrated the policy already is. If an action already has probability 0.99, its score is small — there is little room to push it higher, and the gradient reflects that. If an action has probability 0.01, a good outcome produces a large score, aggressively raising it. The estimator naturally allocates more gradient to actions the policy currently considers unlikely but that turned out well. This is the seed of why entropy and the policy's current shape matter so much for learning dynamics.

#### Worked example: weighting two actions in one state

Consider a state $s$ visited twice in a batch. The first time the agent took action $a_1$ and the return-to-go was $G = 10$; the second time it took $a_2$ and the return-to-go was $G = 2$. The policy currently assigns $\pi(a_1\mid s) = 0.3$ and $\pi(a_2\mid s) = 0.7$. For a softmax policy the score of taking $a_1$ raises its logit by roughly $(1 - 0.3) = 0.7$ per unit gradient, weighted by 10, giving a push of $\approx 7$ toward $a_1$. Taking $a_2$ pushes its logit by $(1 - 0.7) = 0.3$ weighted by 2, a push of $\approx 0.6$ toward $a_2$. Net effect: the policy shifts decisively toward $a_1$, the action that produced the far higher return, and shifts toward it *more* because it was the less-likely action to begin with. Over many such updates the policy concentrates on the high-return actions in each state — which is exactly policy improvement, executed by gradient steps rather than by a Bellman backup.

#### Worked example: a single bandit step

Strip away the sequential structure to see the mechanics. Consider a one-step problem with two actions, left and right, and a softmax policy with a single logit difference parameter $\theta$, so $\pi_\theta(\text{right}) = \sigma(\theta)$ and $\pi_\theta(\text{left}) = 1 - \sigma(\theta)$, where $\sigma$ is the logistic sigmoid. Suppose $\theta = 0$, so each action has probability 0.5. The reward for right is 1, for left is 0.

The score for right is $\nabla_\theta \log \sigma(\theta) = 1 - \sigma(\theta) = 0.5$. The score for left is $\nabla_\theta \log(1-\sigma(\theta)) = -\sigma(\theta) = -0.5$. Suppose on this sample we drew the action *right* and got reward 1. The single-sample gradient estimate is $\text{score} \times \text{reward} = 0.5 \times 1 = 0.5$, a positive number, so the ascent step $\theta \leftarrow \theta + \alpha \cdot 0.5$ increases $\theta$, which raises $\sigma(\theta)$, which makes *right* more likely. Exactly what we want. Had we drawn *left* (reward 0), the estimate would be $-0.5 \times 0 = 0$ — no update, because the reward was zero. Notice already a hint of the variance problem: the update depends entirely on which action happened to be sampled. Over many samples it averages to the true gradient, but any single estimate is noisy.

Let us push this bandit example one step further to make the unbiasedness concrete, because seeing it average out builds intuition for the whole post. The *true* expected gradient is $\mathbb{E}[\text{score}\cdot r] = \pi(\text{right})\cdot(1-\sigma)\cdot 1 + \pi(\text{left})\cdot(-\sigma)\cdot 0 = 0.5 \cdot 0.5 \cdot 1 = 0.25$. Now check that the single-sample estimates average to it: with probability 0.5 we draw right and get an estimate of $0.5$; with probability 0.5 we draw left and get $0$. The average is $0.5 \cdot 0.5 + 0.5 \cdot 0 = 0.25$. The single samples ($0.5$ or $0$) straddle the truth ($0.25$) — neither is correct on its own, but their mean is exactly right. That gap between any single sample and the mean is the variance, and every technique in the second half of this post is a way to shrink it without moving the mean off $0.25$.

## 4. REINFORCE: the Monte Carlo policy gradient

REINFORCE, introduced by Ronald Williams in 1992, is the most direct possible implementation of the theorem: sample a full episode, compute the return-to-go $G_t$ at each step, and take a gradient ascent step using the empirical version of the boxed equation. With $N$ sampled episodes, the estimator is

$$
\nabla_\theta J(\theta) \approx \frac{1}{N} \sum_{i=1}^{N} \sum_{t=0}^{T_i-1} \nabla_\theta \log \pi_\theta(a_t^{(i)}\mid s_t^{(i)})\, G_t^{(i)}.
$$

In code, you do not compute that gradient by hand — you build a **surrogate loss** whose gradient equals the policy gradient, and let autograd do the rest. The surrogate is $L(\theta) = -\frac{1}{N}\sum_i \sum_t \log \pi_\theta(a_t\mid s_t)\, G_t$ (negative because optimizers minimize). Here is the core in PyTorch with Gymnasium on CartPole:

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import gymnasium as gym

class PolicyNet(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, obs):
        return self.net(obs)  # logits

def select_action(policy, obs):
    logits = policy(torch.as_tensor(obs, dtype=torch.float32))
    dist = Categorical(logits=logits)
    action = dist.sample()
    return action.item(), dist.log_prob(action)

def discounted_returns(rewards, gamma=0.99):
    G, out = 0.0, []
    for r in reversed(rewards):
        G = r + gamma * G
        out.append(G)
    out.reverse()
    return torch.tensor(out, dtype=torch.float32)
```

The training loop runs one episode, stores log-probabilities and rewards, computes returns-to-go, and forms the surrogate loss:

```python
def train_reinforce(episodes=1500, gamma=0.99, lr=1e-2, seed=0):
    env = gym.make("CartPole-v1")
    torch.manual_seed(seed)
    policy = PolicyNet(env.observation_space.shape[0], env.action_space.n)
    opt = optim.Adam(policy.parameters(), lr=lr)
    history = []

    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        log_probs, rewards, done = [], [], False
        while not done:
            action, logp = select_action(policy, obs)
            obs, r, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            log_probs.append(logp)
            rewards.append(r)

        G = discounted_returns(rewards, gamma)
        log_probs = torch.stack(log_probs)
        loss = -(log_probs * G).sum()   # the surrogate

        opt.zero_grad()
        loss.backward()
        opt.step()

        history.append(sum(rewards))
        if ep % 100 == 0:
            avg = sum(history[-100:]) / len(history[-100:])
            print(f"ep {ep:4d}  avg100 {avg:6.1f}")
    return history
```

Run this and you will watch the average return climb from around 20–30 in the first hundred episodes toward 400+ after a thousand or so — but with violent swings. Some hundred-episode windows look great; others collapse back toward 50. That instability is not a bug in the code. It is the variance of the estimator, and understanding it precisely is the rest of this post.

A few things in that loop deserve a second look because they are exactly where beginners get tripped up. First, notice we accumulate `dist.log_prob(action)` as a tensor *with its computation graph attached* — we do not detach it, because we need to backpropagate through it later. The `torch.stack(log_probs)` at the end preserves the graph for all timesteps so a single `loss.backward()` computes the gradient for the whole episode. Second, the returns-to-go are computed *backwards* from the end of the episode, which is the natural and efficient way: $G_t = r_t + \gamma G_{t+1}$ is a simple reverse scan. Third — and this is a real footgun — we use the *sum* over timesteps, not the mean. Using the mean would silently rescale the gradient by $1/T$, coupling your effective learning rate to the episode length, which on CartPole (where episodes get *longer* as the agent improves) would make your learning rate effectively shrink as you succeed. Some implementations do use the mean and compensate; just be deliberate about which you choose.

Here is a minimal evaluation harness so you can measure the policy rather than eyeball the training prints. It runs deterministic-ish rollouts (taking the argmax action) and reports the average return, which is what we will quote in the results section:

```python
def evaluate(policy, n_episodes=20, seed=10_000):
    env = gym.make("CartPole-v1")
    returns = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        total, done = 0.0, False
        while not done:
            with torch.no_grad():
                logits = policy(torch.as_tensor(obs, dtype=torch.float32))
                action = torch.argmax(logits).item()  # greedy eval
            obs, r, term, trunc, _ = env.step(action)
            total += r
            done = term or trunc
        returns.append(total)
    return sum(returns) / len(returns)
```

Evaluating with the greedy action rather than a sample is a small but important habit: during training you sample for exploration, but to report a policy's quality you usually want its mode. On CartPole the gap is small; on harder environments it can be substantial.

It is also worth pausing on the relationship between this twenty-line function and the boxed theorem, because the gap is smaller than it looks. The surrogate loss `-(log_probs * G).sum()` is not the policy gradient — it is a *scalar whose gradient equals the policy gradient*. When autograd differentiates `log_probs` with respect to $\theta$, it computes exactly $\nabla_\theta \log\pi_\theta(a_t\mid s_t)$ for each timestep; multiplying by the (detached) returns $G_t$ and summing reproduces $\sum_t \nabla_\theta\log\pi_\theta(a_t\mid s_t)\,G_t$. The minus sign converts gradient ascent on $J$ into gradient descent on $L$ so a standard optimizer can run it. A common bug is to forget that $G$ must be a *constant* with respect to $\theta$ — if you accidentally let gradients flow into the returns (for example by computing them from a value network without detaching), you are no longer optimizing the policy gradient at all; you are optimizing some other objective that usually trains beautifully on the loss curve and not at all on the return curve.

#### Worked example: tracing one REINFORCE update

Suppose an episode lasts 4 steps with rewards $[1, 1, 1, 1]$ and $\gamma = 0.99$. The returns-to-go are $G_0 \approx 3.94$, $G_1 \approx 2.97$, $G_2 \approx 1.99$, $G_3 = 1.0$. If the sampled log-probabilities were each about $\log 0.5 = -0.69$ (a near-uniform policy), the surrogate loss is $-(\sum_t \log\pi_t \cdot G_t) = -(-0.69)(3.94+2.97+1.99+1.0) \approx 6.84$. Every action in this successful short episode gets pushed up, weighted by how much future return followed it: the first action (weight 3.94) gets shoved up nearly four times harder than the last (weight 1.0). Now imagine the very next episode lasts 200 steps with returns-to-go around 180 at the start. The gradient contribution from that episode is roughly *fifty times larger in magnitude* than from the 4-step episode — even though both episodes were "good." The optimizer sees wildly different gradient scales depending on episode length. That scale mismatch is the heart of the variance problem.

## 5. Why the variance is so punishing

The REINFORCE gradient is a Monte Carlo estimator, and Monte Carlo estimators have variance that you control only by averaging more samples. But policy gradients have several structural reasons their variance is unusually bad. Figure 3 stacks the four main culprits.

![Stack of four variance sources in REINFORCE long horizons sparse rewards credit assignment and no baseline](/imgs/blogs/the-policy-gradient-theorem-3.png)

**Variance grows with the horizon.** The per-trajectory gradient is a sum of $T$ score-function terms, each multiplied by a return that can be as large as the total horizon. As we saw in the worked example, a 200-step episode produces gradient magnitudes roughly two orders larger than a 4-step episode. Worse, the *return itself* $G_t$ is a sum of many random rewards, so its variance grows with the number of remaining steps. Long episodes therefore contribute both larger and noisier gradient terms. Empirically, the variance of the REINFORCE estimator grows roughly with $T^2$ — one factor of $T$ because the gradient is a *sum* of $T$ score terms, and another factor of $T$ because each return-to-go is a sum of up to $T$ rewards whose magnitude grows with the horizon. We can see the $O(T^2)$ scaling without heavy machinery. The full-trajectory estimator is $\hat g = \big(\sum_{t=0}^{T-1}\nabla\log\pi_t\big)R(\tau)$. The bracketed sum of scores is itself a sum of $T$ roughly-independent zero-mean terms, so its magnitude grows like $\sqrt{T}$ and its squared magnitude like $T$. The return $R(\tau)$, being a sum of $T$ rewards, grows like $T$ in a dense-reward environment such as CartPole (one point per step). The product's variance therefore picks up the score-sum variance ($\propto T$) multiplied against the squared return scale ($\propto T^2$ if the return grows linearly), and even after the reward-to-go trick trims this, the residual horizon dependence is super-linear. The intuition to carry away: **long episodes do not just give you more data, they give you noisier gradient estimates per episode**, and on environments like CartPole where success *lengthens* the episodes, the variance gets worse exactly as the agent gets better — a vicious feedback that baselines exist to break.

**Sparse and delayed rewards make the signal tiny.** If reward is zero everywhere except at a goal reached on step 180, then for 179 steps the score functions are multiplied by a return that is dominated by a single distant event. The estimator is mostly multiplying meaningful gradients by noisy, weakly-correlated returns.

**Credit assignment is crude.** REINFORCE attributes the entire return-to-go $G_t$ to action $a_t$, even though much of that return was determined by *later* actions and by environment randomness, not by $a_t$ itself. The estimator is unbiased, so over infinitely many samples this averages out — but in any finite batch, the same action can be credited with wildly different returns depending on what happened afterward, which is variance.

**The full-return scale.** This is the subtle one and the one we can fix cheaply. Consider a state where every action yields a return between 100 and 102. The policy gradient weights each action's score by its return, so *every* action gets pushed up hard (by ~100), and the only thing distinguishing the good action from the slightly-less-good one is the difference of 2 against a baseline scale of 100. The signal-to-noise ratio is about 2:100. If we could subtract off the common 100 — the "expected return from this state regardless of action" — then the good action would be weighted by +1 and the bad by −1, a 1:1 signal. That subtraction is the **baseline**, and remarkably, it costs us nothing in bias.

It is worth making the variance statement quantitative, because "high variance" is too vague to act on. For a Monte Carlo estimator $\hat{g} = \frac{1}{N}\sum_i X_i$ of a quantity with per-sample variance $\sigma^2$, the estimator's variance is $\sigma^2 / N$ — it falls only as $1/N$, so halving the noise requires *quadrupling* the samples. That is a slow, expensive lever. For the policy gradient, $X_i$ is a sum of $T$ score terms each scaled by a return, so its variance has contributions that scale with both the horizon and the squared magnitude of the returns. Concretely, if the returns have a common offset $c$ and a spread $\delta$ around it, the variance of the weighted score is dominated by the $c^2 \cdot \mathbb{E}[(\nabla\log\pi)^2]$ term — the offset contributes a huge amount of variance while carrying *zero* information about which action is better (it is common to all actions). Subtracting a baseline near $c$ kills that dominant term outright. This is why a baseline can cut variance by one to two orders of magnitude in high-return environments while leaving the expected gradient untouched: it removes the largest variance contributor, which happened to be pure noise with respect to the decision.

#### Worked example: variance with and without a baseline

Take a state where two actions have true returns $Q(\text{a}) = 102$ and $Q(\text{b}) = 100$, each sampled with probability 0.5. Without a baseline, the gradient weight is the raw return: the estimator's per-sample weight is either 102 or 100, with mean 101. The variance of those weights around their mean is $\frac{1}{2}(102-101)^2 + \frac{1}{2}(100-101)^2 = 1$. That sounds small until you remember the *mean* is 101 — the estimator is constantly pushing both actions up by ~101 and only the tiny variance-1 wiggle carries the learning signal. Now subtract the baseline $V(s) = 101$ (the average). The weights become $+1$ and $-1$. Same variance of 1, but now the mean per-action weight is the *advantage* — exactly the signal — and the large common offset is gone. The gradient stops wasting its magnitude on the shared component and concentrates entirely on the differences. In real high-return environments the common offset can be hundreds of times larger than the advantage, so removing it is the difference between training and not training.

To put a number on "the difference between training and not training": suppose the score's second moment is $\mathbb{E}[(\nabla\log\pi)^2] = m$. Without a baseline, the variance contribution from the offset alone is $c^2 m = 101^2 m \approx 10{,}201\,m$. With the baseline subtracted, the analogous term is the advantage scale squared, $1^2 m = m$. The baseline cut the dominant variance term by a factor of about ten thousand, while the bias stayed exactly zero. That ten-thousand-fold reduction is not a typo and it is not unusual — it is the routine experience of anyone who has trained a policy gradient on a dense-reward environment and watched a single `advantages = returns - values` line turn a divergent run into a converging one.

## 6. Baseline subtraction: free variance reduction

Here is the claim that makes baselines work, and it is worth proving rather than asserting. We can subtract *any* function $b(s)$ that depends on the state but **not** on the action from the return, and the gradient stays exactly unbiased:

$$
\nabla_\theta J(\theta) = \mathbb{E}\big[ \nabla_\theta \log \pi_\theta(a\mid s)\, (Q^\pi(s,a) - b(s)) \big].
$$

To prove the subtracted term vanishes in expectation, fix a state $s$ and take the expectation of $\nabla_\theta \log \pi_\theta(a\mid s)\, b(s)$ over actions drawn from the policy:

$$
\mathbb{E}_{a \sim \pi_\theta}\big[ \nabla_\theta \log \pi_\theta(a\mid s)\, b(s) \big] = b(s) \sum_a \pi_\theta(a\mid s)\, \nabla_\theta \log \pi_\theta(a\mid s).
$$

We pulled $b(s)$ out because it does not depend on $a$. Now apply the log-derivative trick *in reverse*: $\pi_\theta \nabla_\theta \log \pi_\theta = \nabla_\theta \pi_\theta$. So the sum becomes

$$
b(s) \sum_a \nabla_\theta \pi_\theta(a\mid s) = b(s)\, \nabla_\theta \sum_a \pi_\theta(a\mid s) = b(s)\, \nabla_\theta 1 = 0.
$$

The probabilities sum to 1 for every $\theta$, so the gradient of their sum is the gradient of a constant, which is zero. This is exactly the zero-mean-score property from Section 2, now doing real work: because $\mathbb{E}_a[\nabla\log\pi] = 0$, multiplying the score by anything that does *not* depend on the action leaves the expectation at zero. The crucial hypothesis is the *action-independence* of $b$ — if $b$ depended on $a$ we could not pull it out of the sum, and the term would not vanish. That is precisely why the baseline must be a function of the state alone, never of the action: a baseline $b(s,a)$ that depended on the action would introduce bias, because it would be correlated with the score it multiplies. The advantage's denominator $V(s)$ respects this; an action-value baseline $Q(s,a)$ would not and is therefore forbidden as a plain subtraction (it reappears, correctly handled, only in the more elaborate machinery of control variates). The baseline contributes nothing to the expected gradient — it is unbiased — yet it can change the *variance* dramatically. Figure 4 contrasts the gradient scale before and after.

![Before and after comparison showing gradient scale collapsing from plus minus fifty to plus minus five once a baseline is subtracted](/imgs/blogs/the-policy-gradient-theorem-4.png)

**Which baseline minimizes variance?** The truly optimal baseline is a per-dimension, score-weighted average of the returns:

$$
b^* = \frac{\mathbb{E}\big[\lVert\nabla_\theta \log \pi_\theta\rVert^2\, G \big]}{\mathbb{E}\big[\lVert\nabla_\theta \log \pi_\theta\rVert^2\big]},
$$

derived by writing the variance of the baselined estimator as a function of $b$, differentiating, and setting it to zero. Let us actually do that derivation, because it is short and it explains the otherwise mysterious form of $b^*$. The variance of the baselined estimator (for a single state, single dimension) is $\text{Var}[\nabla\log\pi\,(G-b)] = \mathbb{E}[(\nabla\log\pi)^2(G-b)^2] - \big(\mathbb{E}[\nabla\log\pi\,(G-b)]\big)^2$. The second term does not depend on $b$ (we just proved baselines do not change the mean), so to minimize variance we minimize the first term over $b$. Expanding $(G-b)^2 = G^2 - 2bG + b^2$ and differentiating $\mathbb{E}[(\nabla\log\pi)^2(G-b)^2]$ with respect to $b$ gives $-2\mathbb{E}[(\nabla\log\pi)^2 G] + 2b\,\mathbb{E}[(\nabla\log\pi)^2] = 0$, which solves to exactly the $b^*$ above. So the optimal baseline is not simply the mean return $\mathbb{E}[G]$ — it is the return weighted by the *squared score magnitude*, giving more weight to states and actions where the gradient is large and therefore where reducing variance matters most. In practice nobody uses $b^*$ directly — it requires estimating those second moments. The standard, near-optimal, and far simpler choice is the **state-value function** $b(s) = V^\pi(s)$, the expected return from state $s$ under the current policy. It is a natural "what did I expect from this state" reference, and subtracting it gives the most important quantity in policy-based RL: the **advantage function**.

## 7. The advantage function

Define the advantage as

$$
A^\pi(s,a) = Q^\pi(s,a) - V^\pi(s).
$$

It answers a sharper question than $Q$ alone: not "how good is this action in absolute terms" but "how much *better than average* is this action in this state." If $A > 0$, the action beat the policy's own expectation and should be made more likely; if $A < 0$, it underperformed and should be made less likely. Figure 8 shows the decomposition — $Q$ splits into a state baseline $V$ that carries no action-specific information and an advantage $A$ that carries all of it.

![Graph decomposing the action value Q into a state value baseline V and an advantage A that carries the learning signal](/imgs/blogs/the-policy-gradient-theorem-8.png)

Substituting into the theorem gives the advantage form of the policy gradient, which is the basis of every actor-critic method:

$$
\nabla_\theta J(\theta) = \mathbb{E}\big[ \nabla_\theta \log \pi_\theta(a\mid s)\, A^\pi(s,a) \big].
$$

Why does this reduce variance without adding bias? *No bias* because $V^\pi(s)$ is a valid action-independent baseline — we just proved any such baseline is unbiased. *Lower variance* because $A^\pi(s,a)$ is centered near zero: by construction $\mathbb{E}_{a\sim\pi}[A^\pi(s,a)] = 0$, since the average advantage over the policy's own action distribution is exactly $\mathbb{E}[Q] - V = V - V = 0$. The estimator weights are therefore small numbers scattered around zero rather than large numbers scattered around $V(s)$, and the variance of a Monte Carlo estimator scales with the squared magnitude of those weights. We removed the large common component that contributed only noise.

There is a beautiful intuition here worth holding onto: the advantage is the *learning signal* and the value is the *expectation we measure it against*. A reward of +100 in a state where you expected +101 is bad news (advantage −1) and the policy should move *away* from that action, even though the raw return is large and positive. REINFORCE without a baseline would have stupidly increased the probability of that action because its return was positive. The baseline is what lets the agent be appropriately disappointed.

In practice we estimate $V^\pi(s)$ with a second neural network — the **critic** — trained by regression to the observed returns. When the critic supplies the baseline and the actor (the policy) is updated with the resulting advantage, you have an **actor-critic** method, the subject of the next post in this track. For a pure Monte Carlo baseline you can even use a learned $V_\phi(s)$ and compute $A_t \approx G_t - V_\phi(s_t)$, which is what we will implement now.

There is a spectrum of advantage estimators worth knowing about, because the choice trades bias against variance and every modern method picks a point on it. At one extreme is the pure Monte Carlo advantage $A_t = G_t - V_\phi(s_t)$: the return $G_t$ is an unbiased estimate of $Q^\pi(s_t, a_t)$, so this advantage is (nearly) unbiased but carries the full Monte Carlo variance of the return. At the other extreme is the one-step temporal-difference advantage $A_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)$, which bootstraps off the critic's own estimate of the next state: it is much lower variance (it sums only one real reward plus a smooth value estimate) but *biased* whenever the critic is wrong, which it always is during training. Generalized Advantage Estimation (GAE, Schulman et al., 2015) interpolates between these two extremes with a parameter $\lambda \in [0,1]$: $\lambda = 1$ recovers Monte Carlo (low bias, high variance), $\lambda = 0$ recovers one-step TD (high bias, low variance), and intermediate values blend them. PPO almost always runs GAE with $\lambda \approx 0.95$. For this post we use the simplest member of the family — Monte Carlo $G_t - V_\phi(s_t)$ — because it follows directly from the theorem and keeps the bias story clean, but understand that the very next rung up the ladder is choosing where on the bias-variance spectrum to sit.

A subtle but important point about the critic: it is trained to predict $V^\pi(s)$ *for the current policy* $\pi$, and the policy keeps changing. So the critic is always chasing a moving target — the value function it should predict shifts every time the actor updates. This non-stationarity is one reason critics are trained with a separate, often more aggressive optimizer, and why critic-fitting bugs are a leading cause of mysterious policy-gradient failures. If your baseline is systematically wrong — say it underestimates $V(s)$ everywhere — then your advantages are systematically inflated, every action looks good, and the policy never learns to discriminate. A good debugging habit is to log the critic's *explained variance* against the actual returns; if it is near zero or negative, your baseline is contributing noise rather than removing it.

## 8. REINFORCE with a baseline, end to end

This is the full runnable upgrade: an actor network for the policy, a critic network for $V_\phi(s)$, and an update that uses $G_t - V_\phi(s_t)$ as the advantage. The critic is trained to minimize the squared error between its prediction and the actual return-to-go.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import gymnasium as gym

class Actor(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, n_actions),
        )
    def forward(self, obs):
        return self.net(obs)

class Critic(nn.Module):
    def __init__(self, obs_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )
    def forward(self, obs):
        return self.net(obs).squeeze(-1)
```

The training loop collects an episode, computes returns-to-go, then takes one update for the critic (regression to returns) and one for the actor (advantage-weighted score):

```python
def train_reinforce_baseline(episodes=1500, gamma=0.99,
                             lr_a=1e-2, lr_c=1e-2, seed=0):
    env = gym.make("CartPole-v1")
    torch.manual_seed(seed)
    obs_dim = env.observation_space.shape[0]
    actor = Actor(obs_dim, env.action_space.n)
    critic = Critic(obs_dim)
    opt_a = optim.Adam(actor.parameters(), lr=lr_a)
    opt_c = optim.Adam(critic.parameters(), lr=lr_c)
    history = []

    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        states, log_probs, rewards, done = [], [], [], False
        while not done:
            s = torch.as_tensor(obs, dtype=torch.float32)
            dist = Categorical(logits=actor(s))
            a = dist.sample()
            obs, r, term, trunc, _ = env.step(a.item())
            done = term or trunc
            states.append(s)
            log_probs.append(dist.log_prob(a))
            rewards.append(r)

        # returns-to-go
        G, returns = 0.0, []
        for r in reversed(rewards):
            G = r + gamma * G
            returns.append(G)
        returns.reverse()
        returns = torch.tensor(returns, dtype=torch.float32)
        states = torch.stack(states)
        log_probs = torch.stack(log_probs)

        # critic: regress V(s) toward returns-to-go
        values = critic(states)
        critic_loss = nn.functional.mse_loss(values, returns)
        opt_c.zero_grad(); critic_loss.backward(); opt_c.step()

        # actor: advantage = return - baseline (detach the critic)
        with torch.no_grad():
            advantages = returns - critic(states)
        # normalize advantages: a cheap, big variance-reduction trick
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        actor_loss = -(log_probs * advantages).sum()
        opt_a.zero_grad(); actor_loss.backward(); opt_a.step()

        history.append(sum(rewards))
        if ep % 100 == 0:
            avg = sum(history[-100:]) / len(history[-100:])
            print(f"ep {ep:4d}  avg100 {avg:6.1f}")
    return history
```

Three implementation notes that matter more than they look:

- **Detach the critic when forming the advantage.** The advantage is used to weight the actor's gradient; we do not want gradients flowing from the actor loss into the critic. The `torch.no_grad()` (or a `.detach()`) enforces that.
- **Advantage normalization.** Subtracting the batch mean and dividing by the batch standard deviation is a standard, slightly-biased-but-enormously-helpful trick. It keeps the gradient scale roughly constant across episodes of different lengths, directly attacking the scale-mismatch variance from Section 5. PPO uses the same trick.
- **Two optimizers, two losses.** The critic learns to predict returns; the actor learns to exploit the resulting advantages. They are trained together but their losses are separate.

If you want to plot the training curves to *see* the variance difference rather than read about it, here is a small driver that runs both algorithms across several seeds and plots the mean and the inter-seed band. The band — not the mean line — is the thing this whole post is about: the baselined version's band is dramatically tighter.

```python
import numpy as np
import matplotlib.pyplot as plt

def smooth(x, k=20):
    if len(x) < k:
        return np.array(x, dtype=float)
    c = np.cumsum(np.insert(np.array(x, dtype=float), 0, 0))
    return (c[k:] - c[:-k]) / k

def run_many(train_fn, seeds=(0, 1, 2, 3, 4), **kw):
    curves = [smooth(train_fn(seed=s, **kw)) for s in seeds]
    L = min(len(c) for c in curves)
    return np.stack([c[:L] for c in curves])  # (n_seeds, L)

plain = run_many(train_reinforce, episodes=1200)
base = run_many(train_reinforce_baseline, episodes=1200)

fig, ax = plt.subplots(figsize=(8, 5))
for label, curves, color in [("REINFORCE", plain, "tab:red"),
                             ("REINFORCE + baseline", base, "tab:blue")]:
    m, sd = curves.mean(0), curves.std(0)
    x = np.arange(len(m))
    ax.plot(x, m, color=color, label=label)
    ax.fill_between(x, m - sd, m + sd, color=color, alpha=0.2)
ax.set_xlabel("episode"); ax.set_ylabel("smoothed return")
ax.set_title("CartPole-v1: variance of the learning curve")
ax.legend(); fig.tight_layout()
fig.savefig("pg_curves.png", dpi=120)
```

For completeness, here is the same task using a maintained library — Stable-Baselines3 — so you can see how little of this you write in production and confirm the numbers independently. SB3 does not ship vanilla REINFORCE, but A2C is the smallest step up (REINFORCE with a learned critic and bootstrapped advantages) and is the honest production-grade comparison point:

```python
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

env = make_vec_env("CartPole-v1", n_envs=8)
model = A2C(
    "MlpPolicy", env,
    learning_rate=7e-4, gamma=0.99,
    ent_coef=0.01,          # the entropy bonus from Section 9
    normalize_advantage=True,
    verbose=0, seed=0,
)
model.learn(total_timesteps=200_000)

eval_env = make_vec_env("CartPole-v1", n_envs=1)
mean_r, std_r = evaluate_policy(model, eval_env, n_eval_episodes=20)
print(f"A2C CartPole-v1: {mean_r:.1f} +/- {std_r:.1f}")
```

Notice that every from-scratch idea in this post appears as a one-word argument in the library: `ent_coef` is the entropy bonus, `normalize_advantage` is the normalization trick, and the learned critic is implicit in `MlpPolicy`. The from-scratch code is for understanding; the library call is for shipping. Both train the same object — the advantage-weighted score function.

#### Worked example: the gradient scale before and after, in numbers

Run both versions on CartPole with the same seed and log the gradient norm of the actor (`torch.nn.utils.clip_grad_norm_` returns it without clipping if you set a huge threshold). In a representative run, plain REINFORCE shows actor gradient norms swinging between roughly 5 and 80 across episodes, with the largest norms on the longest episodes — exactly the scale mismatch we predicted. With the value baseline *and* advantage normalization, the gradient norms tighten to roughly 2–8 and stop correlating with episode length. The average return curve reflects it: plain REINFORCE reaches an average-100 return around 350 but oscillates by ±150 between windows; the baseline version reaches ~480 and holds it with oscillation under ±40. Same network, same learning rate — the only change is subtracting a learned $V(s)$ and normalizing. That is the entire payoff of the theory in this post.

## 9. Convergence: what policy gradients can and cannot promise

It is tempting to assume that because we are doing gradient ascent on $J(\theta)$, we will reach the optimal policy. The honest answer is more limited, and worth stating plainly because it explains a lot of real-world frustration.

For a tabular softmax policy (one logit per state-action pair) with exact gradients and a suitably decaying learning rate, policy gradient ascent **converges to a local optimum** of $J(\theta)$. Recent theory (Agarwal et al., 2021, on the global convergence of policy gradient methods) shows that under certain parameterizations and with enough exploration, tabular softmax policy gradient actually reaches the *global* optimum — but the convergence rate can be very slow, and the result leans on the policy retaining enough probability mass on all actions to keep exploring. The moment you introduce function approximation (a neural network instead of a table), or estimate the gradient from finite samples, those guarantees weaken to "converges to a stationary point of the objective" at best.

Contrast this with dynamic programming. Value iteration on a known MDP converges to the *global* optimal value function, geometrically, because the Bellman optimality operator is a contraction (covered in this series' DP post). Policy gradients have no such contraction structure — $J(\theta)$ is generally non-concave in $\theta$, so there can be local optima, plateaus, and saddle points. In practice the failure modes you actually hit are:

- **Premature determinism.** The policy entropy collapses, one action gets probability ~1 in a state, the score function for the other actions vanishes, and the agent stops exploring. This is why an **entropy bonus** (adding $\beta\, H(\pi_\theta(\cdot\mid s))$ to the objective) is standard practice — it keeps the policy stochastic enough to keep learning.
- **Plateaus from vanishing gradients.** If the policy is nearly uniform and all returns are similar, the advantage is near zero everywhere and the gradient barely moves.
- **High-variance-induced divergence.** A few unlucky large-return episodes can shove $\theta$ into a bad region from which it does not recover. Baselines, advantage normalization, and (later) trust regions exist to prevent exactly this.

The takeaway: policy gradients are a *local* optimizer of a *non-concave* objective estimated from *noisy* samples. They work astonishingly well in practice when you control the variance, but they come with none of the clean global guarantees of dynamic programming on a known model.

#### Worked example: entropy collapse on CartPole

Run plain REINFORCE without an entropy bonus and log the policy entropy $H(\pi_\theta(\cdot\mid s)) = -\sum_a \pi_\theta(a\mid s)\log\pi_\theta(a\mid s)$ averaged over a batch. Early in training the entropy sits near its maximum, $\log 2 \approx 0.69$ for two actions, because the policy is nearly uniform. As learning proceeds the entropy falls — good, the policy is committing to choices. But on some seeds it crashes toward 0.05 within a few hundred episodes, the policy becomes almost deterministic, and *then training stalls* even though the average return is only ~150. The cause is the self-normalizing score property from Section 3 running in reverse: once an action has probability 0.99, the score for the *other* action is tiny, so even when that other action would have been better, the gradient cannot push it back up fast enough. The agent has prematurely committed and can no longer explore its way out. Adding an entropy bonus of $\beta = 0.01$ — appending $+\beta H$ to the objective, which in code is `actor_loss = actor_loss - beta * dist.entropy().sum()` — keeps the entropy from collapsing below ~0.2 and lets the same run climb past 480. The fix is one line; finding that you needed it is the hard part, which is why logging entropy is non-negotiable.

## 10. Practical diagnostics: reading a policy-gradient training run

Because the gradient is noisy and the guarantees are weak, you debug policy gradients by *instrumentation*, not by staring at the return curve alone. A handful of logged quantities tell you which failure mode you are in. This is hard-won; I have lost days to runs that were silently broken in ways the reward curve hid.

- **Average return and its rolling standard deviation.** The headline metric, but the standard deviation across recent episodes is what tells you whether you have a variance problem. A return curve that goes up *on average* but with a band wider than the trend is a variance signal — reach for a baseline and advantage normalization before touching anything else.
- **Policy entropy.** As the entropy-collapse example showed, a too-fast entropy crash means premature determinism; an entropy that never drops means the policy is not committing and your learning rate may be too low or your advantages too small.
- **Critic explained variance.** $1 - \frac{\text{Var}(G - V_\phi)}{\text{Var}(G)}$. Near 1 means the baseline is doing its job; near 0 or negative means the critic is useless or actively harmful, and your "advantages" are mostly the raw returns again.
- **Gradient norm.** Log the actor's gradient norm every update. If it spikes by 10× on long episodes, you have the scale-mismatch problem; advantage normalization or gradient clipping (`clip_grad_norm_` at, say, 0.5) is the fix.
- **Advantage statistics.** Mean and standard deviation of the advantages per batch. After normalization the mean should be ~0 and the std ~1; if the raw advantages have a huge mean, your baseline is biased.

Here is a compact logging block you can drop into either training loop to surface all of these at once:

```python
def log_diagnostics(ep, rewards, advantages, log_probs, dist_entropy,
                    grad_norm, returns, values):
    avg_return = sum(rewards)
    expl_var = 1.0 - (returns - values).var() / (returns.var() + 1e-8)
    print(
        f"ep {ep:4d} | return {avg_return:6.1f} "
        f"| entropy {dist_entropy.mean().item():.3f} "
        f"| adv_mean {advantages.mean().item():+.3f} "
        f"| adv_std {advantages.std().item():.3f} "
        f"| expl_var {expl_var.item():+.3f} "
        f"| grad_norm {grad_norm:.3f}"
    )
```

If you only remember one diagnostic, remember **explained variance of the critic**: it is the single best early indicator of whether your variance-reduction machinery is actually working or just decorating a broken run. A debugging walkthrough for training instabilities like these lives in the training-debugging series (`/blog/machine-learning/debugging-training/the-training-debugging-playbook`).

## 11. Case studies and measured results

**REINFORCE on CartPole-v1.** This is the running example and the cleanest demonstration. A two-layer MLP policy, Adam at `lr=1e-2`, $\gamma=0.99$. A random policy averages about 8 return — the pole falls almost immediately. Plain REINFORCE typically reaches an average-100 return of 350–450 within ~1000 episodes but with large variance between runs and windows; some seeds stall near 100 for hundreds of episodes before catching. Adding the learned $V(s)$ baseline plus advantage normalization reliably pushes the average toward 480–500 (CartPole-v1 caps at 500) and roughly halves the episodes-to-solve while cutting between-window oscillation by 3–4×. The arc is the one Figure 5 sketches: random return ≈ 8, climbing through ≈ 150 where plain REINFORCE often stalls, and converging near 490 once the baseline tames the variance. These numbers are reproducible on a laptop CPU in a few minutes with the code above; treat them as representative rather than exact, since they depend on seed and hyperparameters. Concretely, a typical baselined run logs something like: episode 0 → avg 22; episode 200 → avg 95; episode 400 → avg 240; episode 600 → avg 410; episode 800 → avg 475; episode 1000 → avg 492. The same seed without a baseline might read: episode 0 → avg 21; episode 200 → avg 78; episode 400 → avg 130; episode 600 → avg 290 then back to 160; episode 800 → avg 350; episode 1000 → avg 340 — higher on average than the start but jagged, never locking onto 500.

**REINFORCE's place in history.** Williams' 1992 paper introduced REINFORCE and the baseline idea simultaneously — the variance problem was understood from day one. The algorithm itself is rarely used in production today precisely because of variance, but every modern policy-optimization method is REINFORCE plus variance reduction: A2C/A3C add a learned critic and bootstrap the advantage; TRPO (Schulman et al., 2015) adds a trust region to bound the policy change; PPO (Schulman et al., 2017) replaces the trust region with a clipped surrogate objective and became the default RL algorithm for everything from MuJoCo locomotion to RLHF. GRPO, used in recent LLM training, is REINFORCE with a *group-relative* baseline — it estimates the baseline from the mean reward of a group of sampled responses, dropping the learned critic entirely. The lineage is direct.

**Policy gradients at scale.** OpenAI Five (Dota 2, 2019) was trained with a PPO-based policy gradient at massive scale — thousands of GPUs, the equivalent of hundreds of years of self-play per day — and beat the world champions. RLHF for InstructGPT (Ouyang et al., 2022) used PPO to fine-tune a language model against a learned reward model, moving human-preference win-rates substantially above the supervised baseline. In both cases the core update is the advantage-weighted score function from this post; the engineering is in controlling variance and the policy step size at scale. The RLHF connection is covered in this series and in the training-techniques posts (`/blog/machine-learning/training-techniques/rlhf-from-human-feedback`).

Here is the family laid out so you can see exactly which variance- and bias-knob each method turns relative to the REINFORCE baseline:

| Method | Variance | Bias | Sample efficiency | Implementation complexity |
| --- | --- | --- | --- | --- |
| REINFORCE (no baseline) | Very high | None | Very low | Trivial |
| REINFORCE + baseline | High | None (MC advantage) | Low | Low |
| Actor-Critic (A2C, TD advantage) | Medium | Some (bootstrapping) | Medium | Medium |
| PPO (GAE + clipped surrogate) | Low–medium | Tunable via $\lambda$ | Medium–high | High |

Read the table as a single trajectory: each row buys lower variance with a little more bias and a little more code. REINFORCE has zero bias and pays for it with crippling variance; PPO accepts a controllable sliver of bias (through GAE's $\lambda$ and the value bootstrap) and trades a lot of engineering for a lot of stability. There is no free lunch — only a well-mapped menu, and this post is the leftmost entry on it.

## 12. When to use this

Figure 6 lays out the algorithm trade-offs, and Figure 7 is the decision tree. The short version: policy gradients earn their variance cost when value-based methods cannot easily apply.

![Matrix comparing REINFORCE actor-critic PPO and SAC across variance bias sample efficiency action space and stability](/imgs/blogs/the-policy-gradient-theorem-6.png)

**Use a policy gradient when:**

- **Actions are continuous or high-dimensional.** Robot control, continuous portfolio weights, anything where $\arg\max_a Q(s,a)$ is intractable. SAC and PPO dominate here.
- **You need a genuinely stochastic policy.** Partially observed or adversarial settings where being predictable is being exploitable.
- **You want smooth, controllable policy updates.** When stability matters more than raw sample efficiency, the smooth parameter updates of policy gradients (especially with a trust region) beat the brittle $\arg\max$ flips of value methods.
- **You are doing preference-based fine-tuning.** RLHF and its descendants optimize a language-model policy against a reward model; the action space (next token, or whole response) and the black-box reward make the score-function estimator the natural fit.
- **The reward is non-differentiable and the environment is a black box.** Whenever you can only *sample* outcomes and *score* them, not differentiate through them, the score-function estimator is the tool that still works.

**Reach for something else when:**

- **The action space is small and discrete and the environment is cheap.** Q-learning or DQN are usually more sample-efficient because they reuse data off-policy. For tabular problems with a known model, plain value iteration crushes everything — do not bring a Monte Carlo gradient to a dynamic-programming fight.
- **Samples are extremely expensive and you cannot simulate.** On-policy policy gradients throw away every batch after one update. If each sample costs real money or real robot wear, prefer an off-policy or model-based method.
- **You can reparameterize the policy and differentiate the reward.** If both hold (as in some continuous-control settings), the reparameterization-based estimator behind SAC is lower-variance than the score-function one.

![Decision tree for choosing policy gradients based on whether actions are discrete continuous or require a stochastic policy](/imgs/blogs/the-policy-gradient-theorem-7.png)

Figure 5 reminds us of the practical arc: with the right variance control, the same algorithm walks from random flailing to near-optimal balance.

![Timeline of CartPole training progress from random return 8 through converging to near optimal return 490 with a baseline](/imgs/blogs/the-policy-gradient-theorem-5.png)

Here is a compact comparison of the on-policy vs off-policy trade-off that sits underneath the choice:

| Property | Policy gradient (on-policy) | Q-learning / DQN (off-policy) |
| --- | --- | --- |
| Data reuse | None — fresh samples per update | Replay buffer, reuse old data |
| Sample efficiency | Lower | Higher |
| Action space | Continuous + discrete | Discrete (or needs an actor) |
| Policy type | Stochastic, directly parameterized | Deterministic greedy |
| Update smoothness | Smooth in $\theta$ | Can jump on $\arg\max$ flips |
| Variance | High (the central problem) | Lower, but bias from bootstrapping |
| Convergence | Local optimum, non-concave | Global for tabular, fragile with NN |

And a hyperparameter cheat-sheet for REINFORCE-with-baseline specifically:

| Hyperparameter | Effect | Typical range |
| --- | --- | --- |
| Learning rate (actor) | Step size on policy; too high diverges | 1e-3 to 1e-2 |
| Learning rate (critic) | Baseline fit speed | 1e-3 to 1e-2 |
| Discount $\gamma$ | Effective horizon / credit reach | 0.95 to 0.99 |
| Entropy bonus $\beta$ | Keeps exploration alive | 0.0 to 0.01 |
| Advantage normalization | Stabilizes gradient scale | on (almost always) |
| Hidden width | Policy/critic capacity | 64 to 256 |

## 13. Key takeaways

- **The policy gradient theorem is exact.** $\nabla_\theta J(\theta) = \mathbb{E}[\nabla_\theta \log \pi_\theta(a\mid s)\, Q^\pi(s,a)]$ follows from the log-derivative trick with no approximation; the environment dynamics differentiate away, which is why it is model-free.
- **REINFORCE is the theorem made literal:** sample an episode, weight each action's score by its return-to-go, ascend. It is correct but high-variance, because $G_t$ is a single noisy sample of $Q^\pi(s_t,a_t)$.
- **Variance is the enemy and it scales badly.** It grows roughly as $O(T^2)$ with the horizon, worsens with sparse rewards, and is inflated by the full-return scale of the gradient weights.
- **Any action-independent baseline is unbiased** because $\sum_a \nabla_\theta \pi_\theta(a\mid s) = \nabla_\theta 1 = 0$ — the zero-mean-score property. This is the single most leveraged fact in policy-based RL.
- **Subtract $V(s)$ to get the advantage** $A(s,a) = Q(s,a) - V(s)$. It is centered at zero, so it strips the large common offset and slashes variance without bias — the foundation of actor-critic.
- **Advantage normalization is nearly free and nearly always worth it.** It fixes the gradient-scale mismatch across episode lengths.
- **Convergence is local, not global.** $J(\theta)$ is non-concave; expect local optima, premature determinism, and the need for entropy bonuses and trust regions.
- **Choose policy gradients for continuous actions, stochastic policies, and black-box rewards;** choose value methods for small discrete actions and expensive samples.
- **Everything modern descends from REINFORCE plus variance reduction** — A2C, TRPO, PPO, GRPO. Master this post and the rest of the track is variations on a theme.

## Further reading

- Williams, R. J. (1992). "Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning." *Machine Learning* — the original REINFORCE paper, including the baseline.
- Sutton, R. S., McAllester, D., Singh, S., & Mansour, Y. (2000). "Policy Gradient Methods for Reinforcement Learning with Function Approximation." *NeurIPS* — the formal policy gradient theorem.
- Sutton, R. S. & Barto, A. G. (2018). *Reinforcement Learning: An Introduction*, 2nd ed., Chapter 13 — the definitive textbook treatment.
- Schulman, J., et al. (2015). "High-Dimensional Continuous Control Using Generalized Advantage Estimation" — GAE, the modern advantage estimator that bridges this post and PPO.
- Schulman, J., et al. (2017). "Proximal Policy Optimization Algorithms" — where the advantage-weighted score function becomes a production default.
- Mnih, V., et al. (2016). "Asynchronous Methods for Deep Reinforcement Learning" — A3C/A2C, the first widely-used learned-critic actor-critic.
- Agarwal, A., Kakade, S. M., Lee, J. D., & Mahajan, G. (2021). "On the Theory of Policy Gradient Methods: Optimality, Approximation, and Distribution Shift" — convergence theory.
- Ouyang, L., et al. (2022). "Training Language Models to Follow Instructions with Human Feedback" — InstructGPT, PPO applied to RLHF.
- Within this series: the unified map (`/blog/machine-learning/reinforcement-learning/reinforcement-learning-a-unified-map`), the actor-critic post that builds directly on this advantage formulation, and the capstone playbook (`/blog/machine-learning/reinforcement-learning/the-reinforcement-learning-playbook`).
