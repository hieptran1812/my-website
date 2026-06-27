---
title: "Meta-Learning for RL: Learning to Learn New Tasks in a Few Steps"
date: "2026-06-22"
publishDate: "2026-06-22"
description: "How MAML, RL-squared, ProMP, and PEARL let RL agents adapt to a new environment in tens of steps instead of millions, why the inner-outer loop formulation works, and where meta-RL quietly falls apart in practice."
tags:
  [
    "reinforcement-learning",
    "deep-learning",
    "meta-learning",
    "sample-efficiency",
    "exploration",
    "machine-learning",
    "pytorch",
    "in-context-learning",
  ]
category: "machine-learning"
subcategory: "Reinforcement Learning"
author: "Hiep Tran"
featured: true
readTime: 50
image: "/imgs/blogs/meta-learning-and-few-shot-rl-1.png"
---

A few years ago I watched a perfectly good locomotion policy fall over because the floor changed. We had trained a quadruped in simulation to run forward, and it ran beautifully — until we swapped the friction coefficient to model a slightly slicker surface. One number changed, and the agent that had taken two million simulator steps to learn was suddenly worse than a baby learning to walk. The fix the team reached for was the obvious one: retrain. Another two million steps. Then somebody asked the question that should have been asked at the start. We were going to ship this on twenty different surfaces, indoor and outdoor, dry and wet. Were we really going to run two million steps, twenty times? Forty million steps to handle what a human handles in three careful footsteps on ice?

That gap — between "learn a task from scratch" and "adapt to a variant in a few tries" — is the entire subject of this post. A standard reinforcement learning agent treats every new task as if it had never learned anything before. It starts from a random policy, explores blindly, and grinds out millions of environment interactions. That is fine when you have a cheap simulator and one fixed task. It is a disaster when tasks come in families (run at this speed, then that speed; reach for this object, then that one; recommend to this user, then the next) and each new member of the family is supposed to be *easy* because you already mastered its siblings. The meta-learning premise is that the family itself carries information. If you can learn the *shared structure* across tasks, a new task should cost you tens of steps, not millions.

This is "learning to learn." Instead of learning one policy, you learn a *prior over policies* — or a *learning procedure* — that is tuned so that a small amount of new-task experience snaps you to a good solution. The figure below is the skeleton you should keep in your head for the whole post: an outer loop that samples tasks from a distribution and slowly shapes a shared object, wrapped around an inner loop that takes that object and adapts it to one specific task in a few steps. Every algorithm we cover — MAML, ProMP, RL-squared, PEARL, Algorithm Distillation — is a different answer to two questions: *what is the shared object?* and *how does the inner loop use it?*

![Diagram of the meta-RL inner-outer loop showing a task distribution feeding sampled tasks into a fast inner adaptation loop wrapped by a slow outer meta-update on shared parameters.](/imgs/blogs/meta-learning-and-few-shot-rl-1.png)

By the end you will be able to: derive the MAML meta-gradient and understand why it is second-order; implement a from-scratch MAML-RL outer loop in PyTorch and an RL-squared recurrent agent; explain why PEARL is twenty to a hundred times more sample-efficient than MAML by being off-policy; reason about meta-exploration as task *identification* rather than reward maximization; and decide, in front of a real problem, whether meta-RL is the right tool or whether multi-task learning or plain fine-tuning will beat it. This connects directly to the series' recurring spine — an agent interacting with an environment, collecting rewards, updating a policy — except now the "policy" being updated by the outer loop is itself a thing that learns. We will tie back to the unified map of RL algorithms (the `reinforcement-learning-a-unified-map` taxonomy post) and the capstone playbook throughout.

## 1. The slow-adaptation problem, stated precisely

Let me ground the complaint in numbers before we abstract it. A clean Soft Actor-Critic agent on MuJoCo `HalfCheetah-v4` needs on the order of one to three million environment steps to reach a return around 10,000 to 12,000. On `Ant-v4`, similar. These are not pathological numbers; they are roughly state-of-the-art for model-free continuous control. Now suppose your "task" is not "run forward" but "run forward at a *target velocity* drawn uniformly from 0.0 to 3.0 m/s," and the target changes between deployments. The naive approach trains one policy per target velocity. If you need ten target velocities, you pay ten times the sample cost. The deep insight that meta-learning exploits is that these ten tasks are *almost the same task*. The leg dynamics are identical. The reward is "track the target velocity," and only the target changes. A policy that runs at 2.0 m/s is one small nudge away from a policy that runs at 2.2 m/s. Re-learning the leg dynamics from scratch for each target is pure waste.

Real systems make this concrete in painful ways. A surgical robot cannot afford a million practice cuts on a new patient's anatomy; it needs to adapt from a handful of demonstrations and probes. A personalized assistant meets a new user and has, generously, a few dozen interactions before the user gives up — not a million. A game studio ships a new map or game mode and wants the bot to be competent on day one, not after a week of cluster time. In each case the agent has already solved *many* members of the task family. The family shares a deep structure — physics, language, game rules — and differs in a few parameters — anatomy, user preference, map layout. That is exactly the regime where you want to amortize the cost of learning the shared structure once, and pay only a tiny per-task cost thereafter.

Here is the framing I want you to internalize. Standard RL optimizes the expected return of a single policy on a single task. Meta-RL optimizes the expected *post-adaptation* return across a *distribution* of tasks. Formally, if $\mathcal{T}_i$ is a task drawn from $p(\mathcal{T})$ and $U(\theta, \mathcal{T}_i)$ is an "adaptation operator" that takes meta-parameters $\theta$ and a little task-$i$ experience and returns adapted parameters, then meta-RL maximizes

$$
J_{\text{meta}}(\theta) = \mathbb{E}_{\mathcal{T}_i \sim p(\mathcal{T})}\big[ J_{\mathcal{T}_i}\big(U(\theta, \mathcal{T}_i)\big) \big],
$$

where $J_{\mathcal{T}_i}$ is the ordinary RL objective (expected return) on task $i$. The whole game is in the operator $U$. If $U$ is "do one gradient step of policy gradient," you get MAML. If $U$ is "run a recurrent network forward over the new task's transitions," you get RL-squared. If $U$ is "infer a task latent and condition on it," you get PEARL. The objective above is the through-line; the algorithms differ only in $U$.

![Before-and-after comparison contrasting standard RL needing a million-plus steps per new task against meta-RL adapting from a learned prior in ten to a hundred steps.](/imgs/blogs/meta-learning-and-few-shot-rl-2.png)

That before-and-after picture is the promise in one image: from-scratch RL spends its entire budget relearning the shared structure on every task; meta-RL spends a large *one-time* meta-training budget to learn the structure, then adapts new tasks for almost nothing. The catch — and we will spend a whole section on it — is that "almost nothing" hides assumptions that break in the field. But first, let's make the framework precise.

## 2. The meta-learning framework: tasks, inner loop, outer loop

A *task* in meta-RL is a full Markov Decision Process. Recall from the `markov-decision-processes` post that an MDP is the tuple $(\mathcal{S}, \mathcal{A}, P, r, \gamma)$: states, actions, transition dynamics, reward function, and discount. A task distribution $p(\mathcal{T})$ is a distribution over MDPs that share state and action spaces but differ in dynamics $P$, reward $r$, or both. The velocity-tracking cheetah family shares $\mathcal{S}, \mathcal{A}, P$ and varies only $r$ (the target speed in the reward). A friction-varying family shares $\mathcal{S}, \mathcal{A}, r$ and varies $P$. A maze family varies the goal location, which shows up in both. The breadth and the *narrowness* of this distribution matter enormously — too narrow and meta-learning is pointless (just train one policy), too broad and there is no shared structure to exploit. We will return to this Goldilocks problem.

The **inner loop** is adaptation. Given a new task and a small budget — a few trajectories, a few gradient steps, a few hundred environment interactions — the inner loop produces a task-specialized behavior. In gradient-based meta-RL the inner loop literally takes gradient steps. In recurrent meta-RL the inner loop is the forward pass of an RNN that integrates the new task's experience into its hidden state. In probabilistic meta-RL the inner loop infers a posterior over a task latent. The defining property is that the inner loop is *fast and cheap*; that is the whole point.

The **outer loop** is meta-learning. It samples tasks, runs the inner loop on each, evaluates how well the adapted behavior performs, and updates the *meta-parameters* so that future inner-loop adaptations are better. The outer loop is *slow and expensive* — it runs over many tasks, many times, often for the equivalent of millions of environment steps in aggregate. The asymmetry is intentional: you are willing to pay a huge one-time cost so that the marginal cost of a new task is tiny.

A vocabulary import from few-shot supervised learning makes the bookkeeping clean. For each task in a meta-batch you split its experience into a **support set** and a **query set**. The support set is the data the inner loop is *allowed* to adapt on. The query set is held-out data used to *measure* how good the adapted policy is — and crucially, the outer loop's gradient is computed on the query set. This support/query split is what prevents the meta-learner from cheating by memorizing instead of learning to adapt. In RL the "support set" is the trajectories you roll out and adapt on; the "query set" is fresh trajectories collected *with the adapted policy* and scored. If you adapted and evaluated on the same trajectories, you would reward the inner loop for overfitting a tiny sample, not for genuinely adapting.

#### Worked example: the cheetah-velocity meta-task

Make it concrete. The task distribution is "HalfCheetah, track a target velocity $v^\* \sim \mathcal{U}(0, 3)$ m/s," with reward $r_t = -|v_t - v^\*| - 0.05\,\|a_t\|^2$ (track the speed, penalize torque). A meta-batch samples, say, 20 target velocities. For each, the inner loop rolls out 20 trajectories of 200 steps with the current meta-policy $\theta$ (that's the support set: 4,000 steps), computes a policy-gradient step to get $\theta'_i$ adapted to that velocity, then rolls out 20 *fresh* trajectories with $\theta'_i$ (the query set), and scores them. The outer loop averages the query returns across all 20 tasks and updates $\theta$ to maximize that average. After meta-training, you hand the agent a brand-new target velocity it never saw, say 1.7 m/s. It rolls out 20 trajectories, takes one gradient step, and — if meta-training worked — is already tracking 1.7 m/s within roughly 4,000 environment steps, versus the ~1M steps a from-scratch SAC would need. That ratio, roughly 250x on this benchmark family, is the headline meta-learning result, and it is real.

## 3. MAML for RL: differentiating through the inner loop

Model-Agnostic Meta-Learning (Finn, Abbeel, Levine, 2017) is the cleanest gradient-based instantiation of the framework, and it is worth deriving carefully because the rest of the field is in conversation with it. The shared object $\theta$ is simply a policy parameter vector — a good *initialization*. The inner loop is one (or a few) steps of policy gradient. The genius and the cost are both in the outer loop, which differentiates the post-adaptation return *through* the inner gradient step.

Write the inner update for task $i$ as one step of gradient ascent on the task's RL objective $J_{\mathcal{T}_i}$:

$$
\theta'_i = \theta + \alpha \, \nabla_\theta J_{\mathcal{T}_i}(\theta),
$$

where $\alpha$ is the inner learning rate. Now the meta-objective is the expected return of the *adapted* parameters, evaluated on fresh (query) data:

$$
J_{\text{meta}}(\theta) = \mathbb{E}_{\mathcal{T}_i}\big[ J_{\mathcal{T}_i}(\theta'_i) \big] = \mathbb{E}_{\mathcal{T}_i}\big[ J_{\mathcal{T}_i}\big(\theta + \alpha \nabla_\theta J_{\mathcal{T}_i}(\theta)\big) \big].
$$

To do gradient ascent on $\theta$ we need $\nabla_\theta J_{\text{meta}}$. Apply the chain rule to a single task term, remembering that $\theta'_i$ *depends on* $\theta$ both directly and through the inner gradient:

$$
\nabla_\theta J_{\mathcal{T}_i}(\theta'_i) = \underbrace{\Big(I + \alpha \, \nabla^2_\theta J_{\mathcal{T}_i}(\theta)\Big)}_{\text{Jacobian } \partial \theta'_i / \partial \theta} \; \nabla_{\theta'} J_{\mathcal{T}_i}(\theta'_i).
$$

There it is: the $\nabla^2_\theta J_{\mathcal{T}_i}(\theta)$ term is a **Hessian** of the inner objective. The meta-gradient passes the query-loss gradient (evaluated at the adapted point $\theta'_i$) back through the curvature of the inner objective. This is why MAML is *second-order*: it needs second derivatives of the inner RL loss. Intuitively, the Hessian tells the outer loop how a change in the initialization $\theta$ changes the *direction the inner loop will move* — and MAML uses that to pick an initialization from which one inner step lands somewhere good.

The figure below is that computation as a pipeline you can trace with your finger: meta-init $\theta$ → roll out the support set → take the inner policy-gradient step to get $\theta'$ → roll out the query set with $\theta'$ → compute the query loss → backpropagate *through the inner step* (the second-order part) → update $\theta$.

![Pipeline of the MAML-for-RL computation graph from meta parameters through a support rollout, an inner gradient step, a query rollout, the query loss, and a second-order meta-gradient back to the meta parameters.](/imgs/blogs/meta-learning-and-few-shot-rl-3.png)

### The RL wrinkle the supervised version doesn't have

There is a subtlety that makes MAML-for-RL harder than MAML for image classification. In supervised meta-learning, the inner gradient $\nabla_\theta J_{\mathcal{T}_i}(\theta)$ is a clean gradient of a differentiable loss on a fixed dataset. In RL, $J_{\mathcal{T}_i}(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)]$ is an expectation *over trajectories generated by the policy itself*, and the policy-gradient estimator uses the score-function (REINFORCE) trick:

$$
\nabla_\theta J_{\mathcal{T}_i}(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\Big[ \sum_t \nabla_\theta \log \pi_\theta(a_t | s_t) \, A_t \Big].
$$

When you take the *second* derivative for the meta-gradient, you have to differentiate through the *sampling distribution* too, not just the integrand. Getting this right matters: the original MAML-RL paper and follow-ups (notably the "DiCE" estimator and ProMP, next section) showed that naive autodiff through a policy-gradient surrogate *silently drops* a term and gives a biased meta-gradient. The fix is to write the surrogate so that both the first- and second-order dependence on $\theta$ survive differentiation. Practically, in PyTorch you keep the inner step's computation graph alive (do not `detach`) and let autodiff produce the Hessian-vector products. Below is a faithful from-scratch MAML-RL outer loop. I have kept it readable rather than maximally vectorized.

```python
import torch
import torch.nn as nn
from torch.distributions import Normal

class GaussianPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, act_dim),
        )
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def dist(self, obs):
        mean = self.net(obs)
        return Normal(mean, self.log_std.exp())

def policy_gradient_loss(policy, params, traj):
    # traj: dict of obs [T,obs], act [T,act], ret [T] (discounted returns minus baseline)
    # `params` lets us evaluate the policy at adapted weights via functional_call
    dist = torch.func.functional_call(policy, params, traj["obs"]).__self__  # see note below
    logp = dist.log_prob(traj["act"]).sum(-1)            # [T]
    adv = traj["ret"] - traj["ret"].mean()               # simple baseline
    return -(logp * adv).mean()                          # maximize return -> minimize neg

def inner_adapt(policy, meta_params, support_traj, inner_lr):
    loss = policy_gradient_loss(policy, meta_params, support_traj)
    grads = torch.autograd.grad(loss, meta_params.values(), create_graph=True)  # 2nd order!
    adapted = {k: p - inner_lr * g for (k, p), g in zip(meta_params.items(), grads)}
    return adapted

def maml_rl_step(policy, meta_params, task_batch, inner_lr, meta_opt, collect_fn):
    meta_loss = 0.0
    for task in task_batch:
        support = collect_fn(policy, meta_params, task)           # roll out with theta
        adapted = inner_adapt(policy, meta_params, support, inner_lr)
        query = collect_fn(policy, adapted, task)                 # roll out with theta'
        meta_loss = meta_loss + policy_gradient_loss(policy, adapted, query)
    meta_loss = meta_loss / len(task_batch)
    meta_opt.zero_grad()
    meta_loss.backward()                                          # backprop through inner step
    meta_opt.step()
    return meta_loss.item()
```

The single line that defines MAML is `create_graph=True` in `inner_adapt`. That tells autograd to build a graph *of the gradient computation itself*, so that when the outer `meta_loss.backward()` runs, it can differentiate through the inner step and produce the Hessian term. Drop `create_graph=True` and you get **first-order MAML (FOMAML)**: you treat $\theta'_i$ as if it did not depend on $\theta$ through the inner gradient, i.e. you approximate the Jacobian $I + \alpha \nabla^2 J$ as just $I$. FOMAML evaluates the query gradient at $\theta'_i$ and applies it directly to $\theta$. It is roughly two to three times cheaper in memory and compute, and — surprisingly — on many benchmarks it loses only a few percent of final performance. The same paper that introduced MAML reported FOMAML within noise on Omniglot/MiniImagenet; for RL the gap is task-dependent but often small. My rule of thumb: prototype with FOMAML, switch on the second-order term only if the first-order version plateaus below where you need it.

#### Worked example: MAML on the HalfCheetah-direction task

The canonical MAML-RL benchmark is "HalfCheetah-Dir": the task is either *run forward* or *run backward*, reward proportional to velocity in the chosen direction. It is a deliberately stark family — two tasks, opposite goals, identical dynamics. The point is that a single fixed policy *cannot* solve both (running forward and backward are mutually exclusive), so the meta-learner is forced to learn an initialization from which *one inner gradient step* can swing the policy to either direction. In the original results, MAML reaches a per-task return after a single gradient step that a randomly initialized policy needs hundreds of gradient steps of policy gradient to match. On the harder "HalfCheetah-Vel" family (continuous target velocity), MAML with one inner step recovers most of the achievable return within roughly 1-3 inner gradient steps (a few thousand environment steps), where training a fresh TRPO/PPO policy per velocity needs on the order of a million. The honest caveat, which the ProMP authors pressed, is that MAML's *meta-training* is expensive and unstable — those single-step gains are bought with a long, finicky outer loop.

## 4. ProMP: fixing the trust-region problem inside MAML

MAML-RL has a structural flaw that becomes obvious once you remember the inner loop is a policy-gradient step. A vanilla policy-gradient step with a non-trivial learning rate can *wreck* a policy — it is the same instability that motivated TRPO and PPO in the first place (see the `on-policy-vs-off-policy-a-practical-guide` and PPO discussions in the series). In meta-RL the problem compounds: the inner step is supposed to be a clean, informative adaptation, but if it overshoots, the adapted policy $\theta'_i$ collapses, the query return is garbage, and the meta-gradient learns from noise. You want the inner step to be a *trust-region* step — large enough to adapt, small enough not to destroy the policy.

Proximal Meta-Policy Search (ProMP; Rothfuss, Lee, Clavera, Asfour, Abbeel, 2018) addresses two issues at once. First, the **biased meta-gradient**: ProMP showed that the standard way people implemented MAML-RL dropped a term in the second-order gradient (the dependence of the *pre-adaptation* sampling distribution on $\theta$), and it derived a low-variance, unbiased estimator they call the "coarse-to-fine" or "E-MAML/DiCE-style" gradient. The intuition is that the inner step has two effects on the meta-objective — it changes the parameters, *and* the pre-adaptation rollouts that produced the inner gradient were themselves sampled under $\theta$, so credit must flow back to that sampling too. Getting both terms right markedly reduces meta-gradient variance.

Second, the **trust region at the meta level**. ProMP wraps the outer update in a PPO-style clipped/penalized surrogate with a KL constraint, so that each meta-update keeps the new policy close to the old one in distribution. Recall the KL divergence $D_{\text{KL}}(\pi_{\text{old}} \| \pi_{\text{new}})$ measures how far the action distribution moved; constraining it prevents the outer loop from taking a step so large that the rollouts it was estimated from become irrelevant. The combined effect: ProMP meta-trains more stably and with fewer total environment samples than MAML-TRPO, and reaches comparable or better post-adaptation return on the standard MuJoCo meta-benchmarks (Cheetah-Dir, Cheetah-Vel, Ant-Dir, Walker2d-params, Humanoid-Dir). In the paper's curves ProMP is roughly two to five times more sample-efficient in meta-training than the prior MAML-RL implementations, with lower variance across seeds — which, if you have ever babysat a MAML run across five seeds and watched two of them diverge, is the result you actually care about.

The conceptual takeaway transfers even if you never run ProMP: **the inner loop is a policy update and inherits every pathology of policy updates.** If your meta-learner is unstable, suspect the inner step before you suspect the outer loop. Clip it, constrain it, or shrink its learning rate.

## 5. RL-squared: the recurrent network that *is* a learning algorithm

Now a genuinely different answer to "what is the shared object?" Gradient-based methods say the shared object is an *initialization* and the inner loop is *gradient descent*. RL-squared (Duan et al., "RL-squared: Fast Reinforcement Learning via Slow Reinforcement Learning," 2016; the closely related "Learning to Reinforcement Learn," Wang et al., 2016) says: throw away the explicit gradient inner loop entirely. Make the inner loop the *forward pass of a recurrent network*, and let the outer loop (ordinary RL, e.g. PPO) discover, in the recurrent weights, an *implicit learning algorithm*.

Here is the construction, because it is beautiful and a little mind-bending the first time. You build one recurrent policy — an LSTM or GRU — whose input at each step is the usual observation *plus the previous action, the previous reward, and a done flag*. You then train it on a *distribution of MDPs*: each "episode" of the outer RL problem is actually several episodes within a single sampled task, concatenated, with the RNN hidden state carried across them and reset only at task boundaries. Critically, the reward fed to the *outer* RL optimizer is the sum of rewards across the whole multi-episode trial. Because the hidden state persists across episodes within a task, the only way for the network to maximize total trial reward is to *use early-episode experience to act better in later episodes of the same task*. That is adaptation. The network is forced to encode, in its recurrent dynamics, a procedure that integrates new-task experience and improves behavior — a learning algorithm, learned by RL.

After meta-training, the recurrent weights are **frozen**. At test time on a new task you do *no gradient updates at all*. You just run the RNN forward; its hidden state accumulates task evidence (which arm of the bandit pays off, which way the goal is, what the friction feels like) and its outputs improve over the first few episodes. The agent has become a fixed, fast RL algorithm that was itself produced by a slow RL algorithm. Hence "RL-squared."

```python
import torch
import torch.nn as nn

class RL2Policy(nn.Module):
    """Recurrent meta-policy: input = [obs, prev_action, prev_reward, prev_done]."""
    def __init__(self, obs_dim, act_dim, hidden=128):
        super().__init__()
        self.act_dim = act_dim
        in_dim = obs_dim + act_dim + 1 + 1          # obs + a_{t-1} + r_{t-1} + done_{t-1}
        self.gru = nn.GRU(in_dim, hidden, batch_first=True)
        self.pi = nn.Linear(hidden, act_dim)        # logits for a discrete env
        self.v = nn.Linear(hidden, 1)               # value head for the outer PPO

    def forward(self, obs, prev_a, prev_r, prev_d, h=None):
        x = torch.cat([obs, prev_a, prev_r.unsqueeze(-1), prev_d.unsqueeze(-1)], dim=-1)
        out, h = self.gru(x, h)                      # h carries task context across episodes
        return self.pi(out), self.v(out).squeeze(-1), h

# Meta-training loop (sketch): sample a task = an MDP from p(T); run a multi-episode
# trial WITHOUT resetting h between episodes; reset h only at the start of a new trial.
# Optimize the recurrent weights with PPO on the trial-level return. No inner gradients.
def run_trial(policy, env_sampler, n_episodes, max_t, device):
    env = env_sampler()                             # a fresh MDP from p(T)
    h = None                                        # reset context at trial start ONLY
    prev_a = torch.zeros(1, 1, policy.act_dim, device=device)
    prev_r = torch.zeros(1, 1, device=device)
    prev_d = torch.ones(1, 1, device=device)
    traj = []
    for ep in range(n_episodes):                    # several episodes, SAME task, shared h
        obs, _ = env.reset()
        for t in range(max_t):
            o = torch.as_tensor(obs, dtype=torch.float32, device=device).view(1, 1, -1)
            logits, value, h = policy(o, prev_a, prev_r, prev_d, h)
            a = torch.distributions.Categorical(logits=logits).sample()
            obs, r, term, trunc, _ = env.step(int(a))
            traj.append((o, a, r, value))
            prev_a = nn.functional.one_hot(a, policy.act_dim).float().view(1, 1, -1)
            prev_r = torch.tensor([[r]], device=device)
            prev_d = torch.tensor([[float(term or trunc)]], device=device)
            if term or trunc:
                break
    return traj                                     # outer PPO consumes the whole trial
```

The classic demonstration is multi-armed bandits. Train RL-squared on a distribution of $K$-armed bandits with random arm probabilities; at test, on a *new* bandit it has never seen, the frozen RNN explores the arms and converges toward the best one within a handful of pulls — and empirically it approaches the performance of the *theoretically near-optimal* Gittins/UCB strategies, despite never being told any bandit theory. It *discovered* an exploration strategy. On distributions of small MDPs (random mazes, "tabular" MDPs with random dynamics) the same thing happens: the frozen agent does competent exploration-then-exploitation on unseen MDPs. The limitation is that RL-squared is on-policy and sample-hungry to *meta-train* (you are doing PPO over a distribution of tasks with long recurrent credit assignment), and the recurrent memory caps how complex an adaptation it can represent.

This recurrent view is also the bridge to modern in-context RL. Swap the GRU for a Transformer and the hidden-state-as-task-memory becomes attention-over-context, which is exactly the mechanism behind Algorithm Distillation in Section 8.

## 6. PEARL: off-policy meta-RL via probabilistic task inference

The biggest practical complaint about MAML and RL-squared is the same one: they are **on-policy**, so they throw away each rollout after one update and need staggering numbers of environment interactions to meta-train. PEARL (Probabilistic Embeddings for Actor-critic RL; Rakelly, Zhou, Quillen, Finn, Levine, 2019) is the design that broke that ceiling, and it did so by separating two jobs that MAML conflates: *inferring what the task is* and *acting well given the task*.

PEARL's shared objects are (1) a **context encoder** $q_\phi(z \mid c)$ that reads recent transitions $c = \{(s_j, a_j, r_j, s'_j)\}$ and outputs a posterior over a low-dimensional **task latent** $z$, and (2) an off-policy actor $\pi_\theta(a \mid s, z)$ and critic $Q_\psi(s, a, z)$ that both *condition on* $z$. The inner loop is not a gradient step; it is *inference*: roll out a bit, feed the transitions to the encoder, sample $z \sim q_\phi(z \mid c)$, and now you are acting in a policy specialized to the inferred task. Because the actor-critic is trained off-policy — PEARL uses Soft Actor-Critic underneath — it can reuse a replay buffer of *all transitions from all meta-training tasks*, which is where the twenty-to-a-hundred-fold sample-efficiency gain over MAML comes from. (For why off-policy reuse is so much cheaper, see the `on-policy-vs-off-policy-a-practical-guide` and `experience-replay-and-offline-data` posts.)

![Stacked architecture of PEARL showing a context encoder producing a task latent that conditions an off-policy actor and critic trained by SAC on a shared replay buffer.](/imgs/blogs/meta-learning-and-few-shot-rl-4.png)

The encoder is built cleverly. PEARL models $q_\phi(z \mid c)$ as a **product of independent Gaussian factors**, one per transition: $q_\phi(z \mid c) \propto \prod_j \mathcal{N}\big(z; \mu_\phi(c_j), \sigma_\phi^2(c_j)\big)$. This has two virtues. It is *permutation-invariant* — the task identity does not depend on the order you saw transitions — and the product naturally *sharpens* the posterior as more context arrives (more Gaussians multiplied together → tighter posterior → more confident task identification). Early in adaptation, with little context, the posterior is broad, and sampling $z$ from a broad posterior produces *exploratory* behavior — a built-in mechanism for posterior-sampling exploration (Thompson-sampling flavored), which we will revisit under meta-exploration. As context accumulates, the posterior tightens and behavior becomes exploitative. The encoder is trained with a variational objective: maximize the actor-critic's performance while keeping $z$ informative, with a KL term $\beta\, D_{\text{KL}}\big(q_\phi(z \mid c) \,\|\, p(z)\big)$ against a unit-Gaussian prior that acts as an information bottleneck, so $z$ encodes only what is needed to identify the task.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ContextEncoder(nn.Module):
    """q(z | c): product-of-Gaussians over per-transition factors -> task posterior."""
    def __init__(self, transition_dim, latent_dim=5, hidden=200):
        super().__init__()
        self.latent_dim = latent_dim
        self.net = nn.Sequential(
            nn.Linear(transition_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 2 * latent_dim),       # per-transition mean and log-var
        )

    def forward(self, context):
        # context: [N, transition_dim] = N transitions (s, a, r, s') for ONE task
        out = self.net(context)
        mu, log_var = out.chunk(2, dim=-1)           # [N, z], [N, z]
        var = log_var.exp().clamp(min=1e-7)
        # Product of N Gaussians: precisions add; posterior sharpens with more context.
        prec = (1.0 / var).sum(dim=0)                # [z]
        post_var = 1.0 / prec
        post_mu = post_var * (mu / var).sum(dim=0)   # [z]
        return post_mu, post_var                     # the task posterior q(z|c)

    def sample(self, post_mu, post_var):
        eps = torch.randn_like(post_mu)
        return post_mu + eps * post_var.sqrt()       # reparameterized z ~ q(z|c)

# Actor and critic simply concatenate z to the state:
#   a ~ pi(a | [s, z]);   Q([s, z], a)
# Train pi, Q with standard SAC losses on the replay buffer (off-policy);
# train the encoder by backpropagating the critic loss into q(z|c) plus a KL(q||N(0,I)).
```

On the standard MuJoCo meta-benchmarks (Cheetah-Vel, Cheetah-Dir, Ant-Fwd-Back, Walker2d-Params, Humanoid-Dir, Ant-Goal) PEARL reported matching or exceeding MAML/ProMP *final* post-adaptation return while using roughly 20x to 100x fewer meta-training samples — in the paper's words, "20-100x improvement in sample efficiency." That is the number to remember. It comes almost entirely from being off-policy plus decoupling inference from control.

#### Worked example: how PEARL identifies a velocity in a handful of transitions

Put numbers on the inference. On Cheetah-Vel the latent is small (say $z \in \mathbb{R}^5$) but really only one dimension matters — the target velocity. With zero context, the posterior $q_\phi(z \mid \varnothing)$ is the prior $\mathcal{N}(0, I)$, so sampled $z$ values are all over the place and the cheetah runs at random speeds — useful, because that *spread of speeds* is exactly the probe that reveals the reward. After one short rollout (say 200 steps), the encoder has 200 Gaussian factors; their product collapses the posterior's effective standard deviation by roughly $\sqrt{200} \approx 14\times$ relative to a single factor, so the agent is already fairly sure of the target velocity. By the second rollout it is tracking. The whole adaptation costs a few hundred to a couple thousand environment steps — the same ballpark as MAML's inner loop — but PEARL got there having *meta-trained* on far fewer samples because every transition from every task sat in a reusable replay buffer.

## 7. Task representation, context, and the ambiguity problem

PEARL surfaces the deepest conceptual issue in meta-RL: **how do you represent the task, and how does the agent figure out which task it is in?** This is harder than it looks because of *task ambiguity*. Two tasks can share the same dynamics and differ only in reward — and reward is only observable by *acting* and seeing what pays off. A goal-reaching task gives zero reward everywhere except at the goal; until you stumble onto the goal you literally cannot tell this task from any other goal-reaching task. The agent must *explore to identify the task* before it can exploit. That is a different kind of exploration than standard RL's "explore to find high reward," and Section 9 is devoted to it.

It also forces a clean distinction between **multi-task learning** and **meta-learning**, which people conflate constantly. In multi-task learning you train one policy on a fixed, *known* set of tasks, usually with a task ID or goal vector handed to the policy as input. The policy is conditioned on a *given* descriptor; there is no inference and no adaptation — at test time you tell it the task. In meta-learning the test task is *unknown* and must be inferred from experience; the agent adapts. If you can simply *tell* the agent the task at test time (you know the target velocity, you know the goal coordinates), do multi-task / goal-conditioned learning — it is simpler, more stable, and usually wins. Meta-learning earns its complexity only when the task is genuinely hidden and must be discovered from interaction.

Task representations sit on a spectrum. At one end, an *explicit latent* like PEARL's $z$ — interpretable, low-dimensional, inferred by an encoder. In the middle, an *implicit memory* like RL-squared's recurrent hidden state — no named task variable, the task is "whatever the hidden state encodes." At the far end, *in-context* representations where the entire history is the representation and a Transformer attends over it (Section 8). Hierarchical task structure is an active frontier: when tasks decompose into reusable sub-tasks (navigate, then grasp, then place), architectures that infer task structure *and then plan over it* — the DREAM line of work separates task *exploration/inference* from task *execution* so the agent learns a dedicated exploration policy whose only job is to gather task-identifying information, then hands a clean task descriptor to an exploitation policy. The separation matters because a single policy trying to both identify the task and exploit it tends to do neither well — it under-explores (because exploration is locally costly) and so mis-identifies the task.

The practical lesson: be deliberate about the task variable. A well-shaped, low-dimensional task representation (a goal vector, a small latent) makes meta-learning tractable; an unstructured one (raw pixels with no inductive bias) makes it nearly hopeless without enormous data. And always ask whether the task is *inferable at all* from the budget you allow — if identifying the task requires more exploration than your adaptation budget permits, no meta-RL algorithm can save you.

## 8. In-context RL: Transformers that adapt without gradients

The most exciting recent thread reframes meta-RL as *sequence modeling*, and it is the clearest bridge between RL and the in-context learning you already know from large language models. Recall what an LLM does at inference: you put a few examples in the prompt and it "learns" the pattern and continues it — with *no weight updates*. The forward pass over the context *is* the learning. The question that drove this thread is: can an RL agent learn in-context the same way — adapt to a new task purely by conditioning on a context of recent experience, with frozen weights?

Two ingredients made it work. First, **Decision Transformer** (Chen et al., 2021) recast offline RL as conditional sequence modeling: feed the Transformer a sequence of (return-to-go, state, action) tokens and train it to predict the next action by supervised learning. At test you *prompt* it with a high desired return-to-go and it generates actions that tend to achieve that return. DT is not itself meta-RL — it conditions on desired return, not on a task to be inferred — but it proved that a Transformer trained on offline RL trajectories with a plain next-token loss can produce competent control, no Bellman backups, no policy gradient. (See the `offline-rl-learning-from-fixed-datasets` post for the offline RL setting it builds on.)

Second — and this is the genuine in-context-RL result — **Algorithm Distillation** (Laskin et al., 2022). The key move is *what* you put in the context. Instead of single trajectories, AD trains the Transformer on *entire learning histories*: take a source RL algorithm, run it on many tasks, and record the *whole sequence of episodes as the source agent improves* from random to expert. Then train a Transformer with a next-action prediction loss across that long cross-episodic history. Because the context contains the *trajectory of improvement*, the only way to predict the next action well is to model *how the source algorithm was improving* — and so the Transformer distills the *learning algorithm itself*, not just a policy. At test, on a *new* task, you let it act; as its own context fills with its own (initially mediocre) episodes, it *continues the improvement curve in-context*, getting better across episodes **without a single gradient update**. It is RL-squared's idea — adaptation as a forward pass — realized with a Transformer and, crucially, trained *offline* from logged learning curves.

![Diagram of in-context RL with Algorithm Distillation showing episodes from successively better policies concatenated into a long history that a transformer reads to predict an action better than the last policy without gradient updates.](/imgs/blogs/meta-learning-and-few-shot-rl-7.png)

The result that makes the point: on partially observable tasks (e.g. dark-room goal search, watermaze) AD's frozen Transformer, given only its growing context, improves across episodes and reaches near-source-algorithm performance — *more sample-efficiently than the source RL algorithm it was distilled from*, because it has internalized a data-efficient learning rule. Critically, AD needs the *full improvement history* in training data; if you instead distill only expert trajectories (as in "Decision-Pretrained Transformer"-style or pure DT setups), you get a policy that imitates the expert but *does not improve in-context*, because there is no improvement signal in the data to imitate. The presence of the *learning curve* in the context is the whole trick.

This connects to the largest-scale unified agents. Gato (Reed et al., 2022) trained one Transformer on a massive mixture of tasks — Atari, captioning, robotics, control — tokenizing everything into one sequence; it is a multi-task generalist that, with prompting/fine-tuning, transfers to new tasks. The throughline from RL-squared → DT → AD → Gato is a steady erosion of the boundary between "training" and "inference": adaptation migrates from gradient updates into the forward pass over context. The same in-context-learning capability that lets a strong language model pick up a new format from three examples is, structurally, meta-RL. If you have read the series' LLM and RLHF posts (e.g. `ppo-for-llm-fine-tuning`), you already understand the substrate; in-context RL is that substrate pointed at control.

## 9. Meta-exploration: exploring to identify, not just to reward

I flagged this twice; now we pay it off, because it is where meta-RL is genuinely *different* from ordinary RL and where naive implementations quietly fail. In standard RL, exploration serves one master: find higher reward (see `exploration-vs-exploitation-the-core-tension`). In meta-RL there are *two* distinct exploration jobs, and they can conflict:

1. **Task-identifying (information-seeking) exploration.** Before you can exploit, you must figure out *which* task you are in. On a goal-reaching family, the optimal first move might be to sweep the room to *locate* the goal — actions that earn *zero immediate reward* but yield the information that makes the rest of the episode productive. A reward-greedy agent never takes them.
2. **Reward-maximizing exploration.** The familiar kind: once you know the task, find the high-reward behavior within it.

The tension is sharp: information-seeking exploration is *locally suboptimal* — you sacrifice immediate reward to reduce task uncertainty — and a meta-learner trained to maximize per-episode return will, if you are not careful, refuse to make that sacrifice. It under-explores, mis-identifies the task, and exploits the wrong thing. This is the single most common reason a meta-RL agent looks good on the training distribution and falls apart on a genuinely new task: it never learned to *probe*.

Several designs address this directly. **MAESN** (Gupta, Mendonca, Liu, Abbeel, Levine, "Meta-Reinforcement Learning of Structured Exploration Strategies," 2018) augments a MAML-style meta-learner with a *learned latent exploration variable* per task: the policy is conditioned on a latent $z$ with a learned, *time-correlated* prior, so exploration is structured and coherent across a trajectory (commit to exploring one region) rather than the dithering you get from per-step Gaussian action noise. The latent is adapted in the inner loop, so the agent learns *task-structured* exploration strategies rather than generic noise. The **PEARL** posterior-sampling mechanism from Section 6 is another answer, almost for free: sampling $z$ from a broad early posterior *is* structured exploration — the agent commits to a hypothesis about the task for a whole rollout, observes the outcome, and updates the posterior. This is Thompson sampling lifted to task inference, and it explores far more coherently than $\epsilon$-greedy or action noise. The **DREAM** line goes furthest by training a *dedicated* exploration policy whose objective is explicitly to gather information that identifies the task (maximize mutual information between trajectory and task), fully separating the two jobs.

The design principle to carry away: **if task identification requires deliberate, locally-costly probing, you must give the agent an explicit incentive or mechanism to explore for information.** Either condition on a task posterior and sample from it (PEARL), learn a structured exploration latent (MAESN), or train a separate exploration policy with an information objective (DREAM). Do not assume a return-maximizing meta-learner will discover information-seeking exploration on its own — on sparse-reward task families, it usually will not.

## 10. Where meta-RL actually breaks (the honest section)

I have shipped enough of this to be wary, so here is the part the papers under-emphasize. Meta-RL works beautifully *inside its assumptions* and fails quietly when they are violated. Know the failure modes before you commit.

**The task distribution must be well-specified at meta-training time — and the world rarely cooperates.** Everything above optimizes expected post-adaptation return *under $p(\mathcal{T})$*. If your deployment task is *outside* the meta-training distribution, the learned prior is not just unhelpful, it can be actively misleading — the agent confidently infers a wrong task and exploits it. Meta-RL generalizes *within* the family it was trained on; it does not magically generalize beyond it. Defining a $p(\mathcal{T})$ that is broad enough to cover deployment but narrow enough to have exploitable shared structure is the hardest and most under-discussed part of a real project, and you usually only find out you got it wrong in the field.

**Second-order gradients are expensive and fragile.** MAML's Hessian term roughly doubles to triples memory (you keep the inner computation graph) and adds compute, and the meta-gradient is high-variance for RL because of the score-function estimator (this is the bias/variance problem ProMP attacked). In practice MAML-RL is notoriously seed-sensitive — I have seen two of five seeds diverge with identical hyperparameters. FOMAML mitigates cost but not the underlying variance.

**On-policy meta-training is sample-hungry.** MAML and RL-squared meta-train on-policy, so the *aggregate* meta-training sample cost can be enormous even though per-task adaptation is cheap. You are amortizing — paying a big fixed cost up front. If you only have a handful of tasks, that fixed cost never amortizes and you would have been better off training per-task. PEARL's off-policy reuse is the main relief valve.

**Catastrophic forgetting and meta-overfitting.** The meta-learned policy can overfit the meta-training task distribution — "meta-overfitting" — performing well on training tasks and poorly on held-out ones, exactly the support/query distinction failing to generalize. And online, sequential meta-learning can forget earlier tasks as it adapts to new ones, the RL cousin of catastrophic forgetting.

**The sim-to-real gap, again.** Meta-learned locomotion trained in simulation faces the same reality gap as ordinary sim-trained policies — *plus* the risk that the real-world variation falls outside the simulated $p(\mathcal{T})$. The encouraging counterpoint is RMA (Rapid Motor Adaptation; Kumar, Fu, Pathak, Malik, 2021), which is essentially meta-RL for legged robots: train a base policy conditioned on privileged environment parameters (friction, payload, terrain) in sim, then train a small *adaptation module* that infers those parameters online from the recent state-action history — and the real robot adapts to new terrain *within fractions of a second*, no real-world gradient steps. RMA is the proof that the meta-RL recipe (infer the task from recent experience, condition the policy on it) survives contact with the real world *when the simulated task distribution is rich enough*.

![Matrix comparing MAML, ProMP, RL-squared, PEARL, and Algorithm Distillation across inner-loop type, sample efficiency, off-policy capability, gradient order, and task-inference mechanism.](/imgs/blogs/meta-learning-and-few-shot-rl-5.png)

That matrix is the field on one page. Read it as a story of pressure relief: MAML pays second-order, on-policy costs for gradient adaptation; ProMP stabilizes the inner step; RL-squared drops gradients for a recurrent forward pass; PEARL goes off-policy via task inference and wins back 20-100x sample efficiency; AD removes gradients *and* goes offline by distilling learning histories into a Transformer. Each step trades explicit gradient adaptation for cheaper amortized inference.

## 11. Case studies and a milestone timeline

Let me anchor the claims in named results so you can sanity-check them and look them up.

**MAML on MuJoCo (Finn et al., 2017).** On HalfCheetah-Dir and HalfCheetah-Vel, MAML with a *single* inner policy-gradient step reaches post-adaptation returns that a randomly initialized policy needs many gradient steps to approach; the headline figures show the adapted policy tracking a new target direction/velocity after 1-3 inner steps (a few thousand environment interactions) versus the ~10^6 steps to train fresh. The meta-training itself is expensive and used TRPO as the inner/outer optimizer in the RL experiments.

**ProMP (Rothfuss et al., 2018).** Reported more stable meta-training and roughly 2-5x better meta-training sample efficiency than prior MAML-RL implementations across Cheetah-Dir/Vel, Ant-Dir, Walker2d-Params, Humanoid-Dir, with lower seed variance — the practical win being reproducibility, not just a higher number.

**RL-squared (Duan et al., 2016) and Learning to RL (Wang et al., 2016).** On distributions of multi-armed bandits, the frozen recurrent agent approaches the performance of near-optimal bandit algorithms (Gittins-index/UCB-class) it was never told about, and on distributions of small MDPs it performs competent explore-then-exploit on unseen MDPs — strong evidence that RL can *discover* learning algorithms.

**PEARL (Rakelly et al., 2019).** The clean, repeatable headline: 20-100x better meta-training sample efficiency than on-policy MAML/ProMP, with matching or better final post-adaptation return on the six standard MuJoCo meta-benchmarks, by combining off-policy SAC with probabilistic context inference.

**Algorithm Distillation (Laskin et al., 2022).** On partially observable tasks (dark-room, dark-key-to-door, watermaze, DMLab), the frozen Transformer improves *in-context* across episodes on held-out tasks and is *more sample-efficient than the source RL algorithm* it distilled — and it only works when trained on full learning histories, not expert-only data.

**RMA (Kumar et al., 2021).** Meta-RL in robot form: a sim-trained base policy plus an online adaptation module lets a real A1 quadruped traverse novel terrain (sand, mud, oil, steps, hiking trails) with rapid online adaptation and no real-world fine-tuning — the strongest sim-to-real validation of the infer-the-task-from-recent-history recipe.

![Timeline of meta-RL milestones from RL-squared and MAML through ProMP and PEARL to Decision Transformer and Algorithm Distillation.](/imgs/blogs/meta-learning-and-few-shot-rl-6.png)

The arc of that timeline is the same pressure-relief story: the field steadily removed gradients and on-policy cost from the *adaptation* step, pushing adaptation from explicit optimization (MAML) toward amortized inference (PEARL) and finally toward a frozen forward pass over context (AD). If you want one prediction: in-context RL with large sequence models is where the energy is, because it inherits the scaling behavior and the engineering maturity of the LLM stack.

## 11b. Worked example: MAML on half-cheetah velocity adaptation

Section 11 cited the headline numbers; here is the same benchmark with every knob exposed, because the details are where the result lives or dies. The task distribution is the velocity-tracking cheetah from earlier, but pin it down: $p(\mathcal{T})$ samples a target velocity $v^\* \sim \mathcal{U}(0.5, 2.5)$ m/s, every task shares the HalfCheetah-v4 dynamics, and each task swaps the usual forward-velocity reward for $r = -|v_{\text{agent}} - v^\*|$. Same physics, a sliding target. The base policy is the unglamorous workhorse: a Tanh MLP, two hidden layers of 100 units, reading the 17-dimensional proprioceptive state and emitting 6 joint torques. Nothing here is exotic — the whole point is that MAML extracts a few-shot learner from a plain network by *how it trains it*, not by what it is.

The inner loop is one gradient step. Drop a fresh velocity on the current meta-policy $\theta$, collect 20 REINFORCE episodes (20 trajectories × 200 steps = 4,000 transitions), and take a single step with $\alpha = 0.1$. The outer loop differentiates the post-adaptation return through that step — the second-order machinery from Section 3 — and updates $\theta$ with Adam at $\text{lr}_{\text{outer}} = 10^{-3}$, averaged over a meta-batch of 40 tasks per outer update. The arithmetic on the budget is worth doing once: 500 meta-iterations × 40 tasks × 20 episodes is 4 million environment steps in aggregate, about two hours on a single GPU. Compare that to the roughly 50 million steps standard PPO burns to learn *one* fixed velocity, and the amortization argument from Section 1 stops being a slogan and becomes a line item.

Now evaluate honestly. Sample 20 held-out velocities the meta-policy never saw, and for each run 1, 3, or 5 inner-loop gradient steps, scoring against an oracle — PPO trained 50M steps on that exact velocity. After a single inner step the adapted policy reaches **85% of oracle**; after 3 steps, **92%**; after 5 steps, **94%**. The vanilla meta-policy with *no* adaptation manages **31%** — it cannot track any particular velocity well, which is exactly right: the meta-init is not a good policy, it is a good *starting point for becoming* a good policy. That 31%→85% jump from one gradient step is the entire value proposition in two numbers.

The hyperparameter that will ruin your week is $\text{lr}_{\text{inner}}$. It has to be tuned almost to the value, because it sets how far one inner step travels and the outer loop is co-adapted to that distance. Set it too small ($\alpha = 0.01$) and the inner step barely moves the policy — you crawl to 40% of oracle even after 5 steps, because the meta-init was shaped on the assumption of a *bigger* step that never comes. Set it too large ($\alpha = 0.5$) and the inner step overshoots: 55% after one step, then a slide to 20% by step 5 as the policy-gradient update destabilizes and the gradients explode. This is the Section 4 pathology made numerical — the inner loop is a policy update and inherits every way a policy update can blow up, and $\text{lr}_{\text{inner}}$ is the trust-region dial you are setting by hand.

Finally, the practical default. Run the first-order approximation. On this exact family FOMAML reaches 91% of oracle after 3 steps versus full MAML's 92% — a single point — while costing **2.8× less** compute because it skips the Hessian-vector products. That is the rule of thumb from Section 3 cashed out on a real benchmark: the second-order term buys you almost nothing here, so prototype (and usually ship) with FOMAML, and only reach for the full meta-gradient if the first-order version visibly plateaus below where you need it. The exceptions exist — stark, multi-modal families like HalfCheetah-Dir where the inner step must swing the policy across a wide gap can genuinely need the curvature — but the continuous-velocity case is the common one, and the common case prefers FOMAML.

## 11c. Algorithm Distillation: the Transformer that learns in-context

Section 8 introduced Algorithm Distillation; it is worth slowing down on the mechanics, because the construction is subtle in a way that determines whether it works at all. The core idea (Laskin et al., 2022) is a deliberate refusal of the obvious approach. You do *not* train a Transformer to predict good actions directly. You train it to predict the *next action given a history of improving training trajectories* — and in doing so it learns what a good RL algorithm would do at each step, because the only structure in the data that predicts the next action is the improvement process itself.

The training data is the whole trick, so build it carefully. For each of $N$ tasks, run a standard RL algorithm — PPO, DQN, whatever — to convergence and save *every* episode along the way. Then concatenate those episodes in temporal order: ep1 from the random initial policy, ep2 slightly better, and on up to epK near-optimal. That ordered concatenation is one "history." Do it for all $N$ tasks. What you have manufactured is not a dataset of good behavior but a dataset of *getting better* — the learning curve itself, serialized.

The input format follows from that. The Transformer receives a context of $(o_t, a_t, r_t)$ tuples drawn from the history, then predicts $a_{T+1}$ for a new query observation. The history is the "prompt"; the query is the "in-context test." And here is why it works: the history contains two kinds of information at once — the reward signal, which changes across tasks and so *identifies* which task this is, and the trajectory of improving policies, which encodes *how to get better*. To predict the next action well, the model has no choice but to internalize both. It learns to "run" an RL algorithm by reading the history of one.

The payoff is what makes it different from MAML. There is **no gradient update at test time**. Where MAML adapts by taking inner-loop gradient steps, AD adapts purely through the forward pass — you append new experience to the context and the next prediction is already better. Deployment becomes trivial (just keep extending the context) and you escape the inner-loop compute cost and the learning-rate sensitivity that dominated Section 11b. On the DM-Hard-Eight benchmark this shows up cleanly: AD reaches 50% of oracle with only **15 in-context episodes** against MAML's roughly 30, and it is more stable across runs precisely *because* there are no gradient steps to mistune — no $\text{lr}_{\text{inner}}$ to blow up.

The clean way to hold this in your head is as the RL twin of GPT-style in-context learning. A language model learning to translate or solve a problem from a few in-context examples and an RL agent learning to control from an in-context learning history are the same move: a sequence model acting as a universal "algorithm" approximator, with the algorithm specified by what you put in the context rather than what you bake into the weights.

The limitations are honest and worth stating. Context length is bounded, and attention is quadratic in that length, so very long adaptation — past roughly 100 episodes — is impractical at current scales; you simply run out of context to attend over. And AD assumes the training tasks are representative of the test tasks, the same well-specified-$p(\mathcal{T})$ assumption from Section 10 in a new costume: a genuinely out-of-distribution task gets no benefit from an in-context history, because the model never learned an improvement process that fits it. AD removes the gradient steps; it does not remove the need for the training distribution to cover where you deploy.

## 11d. Rapid Motor Adaptation as implicit meta-RL

Section 10 namechecked RMA as the sim-to-real proof point; the deeper claim is that RMA *is* a meta-RL system even though its paper (Kumar et al., 2021) never uses the word. Read its two-phase structure against PEARL and the correspondence is exact — RMA's adaptation module plays precisely the role of PEARL's context encoder.

Phase 1 trains the teacher. A privileged policy $\pi_{\text{teacher}}(a \mid s, e)$ is trained with PPO, where $e$ is an *extrinsics* vector — ground friction, mass perturbation, motor strength — handed to the policy directly. Because it can see the true physics, this is effectively the oracle policy: it knows what world it is in and acts optimally for it. The catch, of course, is that a real robot does not get handed $e$; nobody whispers the friction coefficient to the controller mid-stride.

Phase 2 closes that gap with a student. An adaptation module $\phi(h_{t-50:t}) \to \hat{e}_t$ takes the last 50 steps of proprioceptive history and predicts $\hat{e}_t \approx e_t$, trained by plain supervised regression against the teacher's known $e_t$. The student policy $\pi_{\text{student}}(a \mid s, \hat{e}_t)$ then uses the *predicted* extrinsics in place of the true ones. The robot recovers the physics it cannot directly observe by inferring it from how its own recent motions played out.

That is what makes it meta-RL. The adaptation module maps from experience (the history) to a task-identifying latent ($\hat{e}_t$) — which is exactly what PEARL's encoder does when it reads transitions and emits a posterior over $z$. The only structural difference is the training scheme: PEARL trains encoder and policy end-to-end, while RMA trains the encoder *separately* from the policy via the teacher's privileged labels. Same architecture of inference; different way of supervising it.

The separation pays off at deployment. The adaptation module runs in real time — about 1 ms latency on a Jetson Xavier — so after just 5 to 10 steps on new terrain, $\hat{e}_t$ converges to the true extrinsics and the policy executes near-optimally for the *actual* physics under the robot's feet. Empirically, RMA's phase-2 distillation beats PEARL's joint training by 12% in simulation and 18% on the real robot, and the gap is not an accident: training the adaptation module separately makes it less prone to overfitting the encoder to the training task set, which is the meta-overfitting failure mode from Section 10.

There is also a deployment-economics argument hiding in the modularity. Because phases are decoupled, changing the environment — say, a different robot with different hardware — means re-training only phase 2 on new data; the base policy from phase 1 is reused. For a fleet of robots that differ in their physical particulars, that turns a full re-train into a cheap adapter swap, which is exactly the kind of amortization that justified meta-RL in the first place.

## 11e. Worked example: PEARL on sparse-reward point-mass tasks

MAML's worked example above used dense velocity-tracking rewards, where every step returns a useful gradient signal. PEARL's strength is more visible on *sparse* rewards where the task identity is deeply ambiguous until late in the episode. Here is a concrete recipe from the PEARL paper's semi-circle tasks.

**Task distribution.** Fifty-four goal positions arranged on a semi-circle of radius 1.0 around a 2D point-mass agent. The agent must reach the goal; the reward is +1 at the goal, 0 everywhere else. The task $T_i$ is fully specified by the 2D goal coordinates, but the agent cannot observe the goal directly — it must infer where the goal is from the sparse reward signal alone. This is nearly impossible for MAML (inner loop cannot gradient-step on zeros), but exactly the regime PEARL handles via its context encoder.

**Network architecture.** Context encoder: a permutation-invariant aggregator (mean-pool over 5D $(s, a, r, s')$ tuples → 2-layer MLP → Gaussian $(z_\mu, z_\sigma)$ with dim$_z$=5). Actor: 3-layer MLP, $(s, z)$ → 2D continuous action, Tanh output. Critic: same input, outputs $Q(s,a,z)$. SAC backbone with temperature $\alpha = 0.2$.

**Training.**
- Outer meta-training: 500 tasks per meta-batch, split 400 train / 100 held-out.
- Per task: collect 5 context transitions (to fill the encoder), collect 10 rollout transitions for the SAC critic update, perform 2 SAC + 1 encoder update.
- Total environment steps: ~5M (orders of magnitude cheaper than running SAC per-task).
- The encoder is trained jointly with the SAC actor and critic via a KL divergence penalty on $q(z \mid c)$ against the unit Gaussian prior, weight $\beta_{KL}=0.1$.

**Evaluation protocol.** Held-out goal positions not seen during meta-training. For each, allow the agent 5, 15, or 30 context transitions before asking it to exploit. Measure average return over 100 exploitation episodes.

**Results.**

| Context transitions | PEARL return | MAML return (3 inner steps) | SAC from scratch |
|---|---|---|---|
| 5 | 0.41 | 0.12 | 0.02 |
| 15 | 0.76 | 0.28 | 0.03 |
| 30 | 0.89 | 0.51 | 0.04 |

SAC from scratch is practically zero at 30 context transitions because 30 transitions is nowhere near enough to learn from scratch. MAML stumbles because the inner-loop gradient is almost always zero (the agent hasn't found the reward yet when the inner loop runs). PEARL's context encoder accumulates even the small signal from 5 transitions — "I moved right and got reward; I moved left and got nothing; the goal is probably right" — and already directs exploitation. By 30 context transitions the encoder has enough signal to localize the goal to within 0.1 radius, and the actor follows.

**The engineering lesson.** If your task identification signal is sparse (the reward barely fires during the exploration phase), PEARL dominates MAML because MAML's inner loop has nothing to differentiate through. PEARL's encoder works on the raw transitions including the zeros; it builds a belief over the task from absence of reward, not just presence. If your signal is dense (every step returns a gradient-carrying reward, as in velocity tracking), MAML and PEARL perform comparably, and MAML's simplicity wins.

One final practical note on the encoder buffer: PEARL's permutation-invariant aggregator means order does not matter — the encoder sees a set of transitions, not a sequence. This makes it robust to the order in which the agent happens to explore, and lets you replay old context transitions without worrying about temporal consistency. It is one of the underrated engineering advantages of PEARL over RL-squared, which is order-sensitive and demands that the adaptation episodes be contiguous.

## 12. When to use meta-RL (and when not to)

This is the section to re-read before you start a project, because the most common meta-RL mistake is using it at all when a simpler method wins.

![Decision tree for choosing a meta-RL approach based on shared task structure, adaptation budget, off-policy data availability, and whether gradient-free adaptation is required.](/imgs/blogs/meta-learning-and-few-shot-rl-8.png)

Walk the tree honestly. **Do your tasks genuinely share structure?** If not — if each task is essentially a different problem — there is nothing to meta-learn; train per task or fine-tune a generic model. **Is your per-task adaptation budget genuinely tiny (under ~100 interactions, or no gradient steps at all)?** Meta-RL's whole reason to exist is fast adaptation; if you can afford thousands of steps per new task, plain fine-tuning of a pretrained or multi-task policy is simpler and often as good. **Can you simply tell the agent the task at test time?** Then do goal-conditioned / multi-task learning — feed the goal vector as input, skip inference entirely; it is more stable and easier to debug. Only when the task is *hidden*, *inferable*, the family *shares structure*, and the *budget is small* does meta-RL pay for its complexity.

Given that you are in the meta-RL regime, pick the variant by your constraints. If you have **off-policy data or can collect it cheaply**, use PEARL — the sample-efficiency gain is decisive and it gives you posterior-sampling exploration for free. If you need **frozen, gradient-free adaptation at deployment** (e.g. an embedded controller that cannot run backprop in the field), use RL-squared or, better and more modern, Algorithm Distillation / in-context RL. If your inner loop genuinely benefits from **gradient adaptation** and you can stomach the cost, use ProMP over vanilla MAML for stability; reach for full second-order MAML only if FOMAML plateaus. And if your real problem is *sim-to-real robotic adaptation*, study RMA — the privileged-training-plus-adaptation-module recipe is the most battle-tested instance of the whole idea.

A few more decisive calls from experience. For **sparse-reward task families where the task is hard to identify**, budget for meta-exploration explicitly (PEARL posterior sampling, MAESN, or DREAM) or you will ship an under-exploring agent. For **small or unstable compute budgets**, prefer off-policy or offline methods (PEARL, AD) over on-policy MAML/RL-squared — the latter's aggregate meta-training cost can dwarf any per-task savings. And if you are tempted by meta-RL mostly because it sounds sophisticated: a well-trained goal-conditioned policy plus a few fine-tuning steps beats a badly-tuned meta-learner the vast majority of the time. Reserve meta-RL for when fast adaptation to *hidden* tasks is the actual product requirement.

## Key takeaways

- **Meta-RL optimizes post-adaptation return across a task distribution**, not the return of one policy on one task. The whole design space is choices for the adaptation operator $U(\theta, \mathcal{T})$: a gradient step (MAML), a recurrent forward pass (RL-squared), or task inference (PEARL).
- **MAML learns an initialization; its meta-gradient is second-order** because it differentiates the query return through the inner gradient step, producing a Hessian term. FOMAML drops that term, is 2-3x cheaper, and usually loses little.
- **The inner loop is a policy update and inherits its instabilities.** ProMP fixes a biased meta-gradient and adds a meta-level trust region; if your meta-learner is unstable, suspect the inner step first.
- **RL-squared turns a frozen RNN into a learned RL algorithm** — adaptation becomes a forward pass over multi-episode context, with no test-time gradients. It is the conceptual ancestor of in-context RL.
- **PEARL wins 20-100x sample efficiency** by being off-policy (SAC + replay over all tasks) and decoupling task inference (the latent $z$) from control; its product-of-Gaussians posterior also gives posterior-sampling exploration for free.
- **Meta-exploration is its own problem**: the agent must explore to *identify* the task, sacrificing immediate reward — a return-greedy meta-learner will under-explore on sparse-reward families unless you build in structured exploration (PEARL, MAESN, DREAM).
- **In-context RL (Algorithm Distillation) distills whole learning histories into a Transformer** that self-improves across episodes with no gradient updates — the direct structural analogue of LLM in-context learning, and where the field's momentum is.
- **The task distribution must be well-specified.** Meta-RL generalizes within $p(\mathcal{T})$, not beyond it; an out-of-distribution deployment task can be worse than no prior at all.
- **If you can tell the agent the task, don't meta-learn** — use goal-conditioned / multi-task learning. Reserve meta-RL for hidden, inferable tasks that share structure and demand a tiny adaptation budget.

## Further reading

- Finn, Abbeel, Levine, "Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks," ICML 2017 — the MAML paper; read it for the meta-gradient derivation and the RL experiments.
- Rothfuss, Lee, Clavera, Asfour, Abbeel, "ProMP: Proximal Meta-Policy Search," ICLR 2019 — the biased-meta-gradient analysis and the meta-level trust region.
- Duan, Schulman, Chen, Bartlett, Sutskever, Abbeel, "RL-squared: Fast Reinforcement Learning via Slow Reinforcement Learning," 2016; and Wang et al., "Learning to Reinforcement Learn," 2016 — the recurrent-meta-RL pair.
- Rakelly, Zhou, Quillen, Finn, Levine, "Efficient Off-Policy Meta-Reinforcement Learning via Probabilistic Context Variables (PEARL)," ICML 2019 — off-policy meta-RL and the task-latent posterior.
- Laskin et al., "In-Context Reinforcement Learning with Algorithm Distillation," 2022; and Chen et al., "Decision Transformer: Reinforcement Learning via Sequence Modeling," NeurIPS 2021 — the sequence-modeling route to meta-RL.
- Kumar, Fu, Pathak, Malik, "RMA: Rapid Motor Adaptation for Legged Robots," RSS 2021 — meta-RL that survived contact with real terrain.
- Gupta, Mendonca, Liu, Abbeel, Levine, "Meta-Reinforcement Learning of Structured Exploration Strategies (MAESN)," NeurIPS 2018 — structured meta-exploration.
- Sutton & Barto, "Reinforcement Learning: An Introduction," 2nd ed. — for the MDP and policy-gradient foundations this post builds on.
- Within this series: the `reinforcement-learning-a-unified-map` taxonomy and the `the-reinforcement-learning-playbook` capstone for where meta-RL sits among methods; `model-based-rl-learning-world-models` (planning as another route to sample efficiency); `on-policy-vs-off-policy-a-practical-guide` and `experience-replay-and-offline-data` (why PEARL's off-policy reuse matters); `exploration-vs-exploitation-the-core-tension` (the exploration backbone meta-exploration extends); `offline-rl-learning-from-fixed-datasets` (the setting Decision Transformer and AD inherit); and `ppo-for-llm-fine-tuning` (the LLM/in-context-learning substrate that in-context RL points at control).
