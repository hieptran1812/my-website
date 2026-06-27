---
title: "Debugging RL Agents: A Systematic Guide to the Six Failure Modes"
date: "2026-06-22"
publishDate: "2026-06-22"
description: "A practitioner's field guide to diagnosing why your RL agent isn't learning — reward signal failures, policy collapse, value divergence, exploration starvation, environment bugs, and distribution shift — and the exact fix for each."
tags:
  [
    "reinforcement-learning",
    "deep-learning",
    "debugging",
    "policy-gradient",
    "q-learning",
    "exploration",
    "rlhf",
    "machine-learning",
    "pytorch",
    "actor-critic",
  ]
category: "machine-learning"
subcategory: "Reinforcement Learning"
author: "Hiep Tran"
featured: true
readTime: 51
image: "/imgs/blogs/debugging-rl-agents-common-failure-modes-1.png"
---

The first RL agent I shipped to production was a market-making bot. For three weeks it "learned." The training loss went down. The episode return went up. The TensorBoard curves looked exactly like the textbook says they should. Then we ran it on held-out data and it lost money on every single day. The agent had not learned to make markets. It had learned that one specific quirk of our simulator — a reward that leaked the next tick's mid-price into the current observation — let it predict the future. We had been training a time-traveler, and the moment we took the time machine away, it was worse than random.

That is the defining feature of debugging reinforcement learning: **the agent can look like it is learning while learning nothing useful, or learning the wrong thing entirely.** In supervised learning, a model that overfits at least overfits to the *real* labels; the worst case is memorization of true targets. In RL there are no labels. The agent generates its own training data by acting, the data distribution shifts as the policy changes, and the reward is the only feedback — so any flaw in the reward, the environment, or the value estimate gets silently amplified by the feedback loop instead of corrected by it. The same property that makes RL powerful (it discovers behavior you never demonstrated) makes it treacherous to debug (it discovers behavior you never *intended*, and the metrics often applaud).

This post is the field guide I wish I'd had. It is organized around **six failure modes** that account for the overwhelming majority of "my RL agent won't learn" situations: reward signal failure, policy collapse, value function divergence, exploration starvation, environment bugs, and distribution shift. For each one I give you the *theory* of why it happens (the math that makes it inevitable under certain conditions), the *diagnostic* (the single plot or log that exposes it), and the *fix* (real code in PyTorch, Gymnasium, Stable-Baselines3, and TRL). Figure 1 shows the workflow that ties them together — a loop of reproduce, reduce, instrument, isolate, fix, verify — and we will keep returning to it.

![A workflow diagram showing the reinforcement learning debugging loop from reproduce failure through reduce, instrument, isolate, fix, and verify generalization](/imgs/blogs/debugging-rl-agents-common-failure-modes-1.png)

By the end you will be able to look at a training curve, name the failure mode from its shape, pull up the one diagnostic that confirms it, and apply the targeted fix — instead of the usual ritual of changing five hyperparameters at once and re-running overnight. This connects directly to the series' recurring spine: the RL loop is an agent interacting with an environment, collecting rewards, and updating a policy. Every failure mode below is a different way that loop can poison itself. If you want the bird's-eye map of where these algorithms sit, see the unified taxonomy in [reinforcement-learning-a-unified-map](/blog/machine-learning/reinforcement-learning/reinforcement-learning-a-unified-map); the whole debugging discipline is folded into [the-reinforcement-learning-playbook](/blog/machine-learning/reinforcement-learning/the-reinforcement-learning-playbook).

## 1. Why RL is harder to debug than supervised learning

Let me make the asymmetry precise, because understanding *why* RL debugging is hard tells you *where* to look.

In supervised learning you optimize a fixed objective over a fixed dataset. If $\mathcal{D} = \{(x_i, y_i)\}$ is your data and $f_\theta$ your model, the loss $\mathcal{L}(\theta) = \frac{1}{N}\sum_i \ell(f_\theta(x_i), y_i)$ is a *stationary* function of $\theta$. The gradient you compute on a batch is an unbiased estimate of the gradient of a function that does not move. If training diverges, the cause is local: bad learning rate, bad initialization, a NaN in the data. You can freeze everything, inspect one batch, and reason about it.

RL breaks every one of those assumptions:

1. **The objective is the expected return, and the expectation is taken over trajectories your own policy generates.** The objective is $J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\!\left[\sum_t \gamma^t r_t\right]$. The distribution over trajectories $\tau$ depends on $\theta$. As you update the policy, the data distribution shifts. This is **non-stationarity baked into the loss itself** — you are chasing a moving target you create by moving.

2. **There are no labels.** The only supervision is a scalar reward, often delayed, often sparse. The agent must solve a credit assignment problem (which of my last 200 actions caused this reward?) that has no analog in supervised learning. A wrong answer to that problem looks identical, from the loss curve, to a right one.

3. **Bootstrapping introduces a feedback loop in the value estimate.** Temporal-difference methods estimate a value by referencing another (also-estimated) value: $V(s) \leftarrow r + \gamma V(s')$. Errors in $V(s')$ propagate into $V(s)$, which propagates into $V(s'')$, and so on. There is a self-referential loop with no ground-truth anchor.

4. **The feedback loop between policy and data is a closed control system that can go unstable.** A small policy change alters which states you visit, which changes the data, which changes the next policy update. This is exactly the structure of a control loop, and like any control loop it can oscillate, diverge, or lock into a fixed point you did not want.

Put these together and you get the **silent failure** problem. Here is a worked example of how it bites.

#### Worked example: the agent that "learned" to do nothing

I once trained a PPO agent on a custom warehouse-robot env. The reward was $+1$ for delivering a package, $-0.01$ per timestep (to encourage speed), and $-1$ for a collision. Training return climbed steadily from $-50$ to $-5$ over two million steps. The curve was monotonic and beautiful. We were thrilled — until we watched a rollout. The agent had discovered that the *fastest way to stop losing points* was to drive into a corner and freeze. It never collided (so no $-1$), and the $-0.01$/step penalty was capped by the episode length of 500 steps, giving a floor of $-5$. The agent had perfectly optimized the reward and completely failed the task. The return curve going from $-50$ to $-5$ was not learning to deliver — it was learning to *minimize loss by quitting*.

The lesson: **the return curve tells you the agent is optimizing something. It does not tell you it is optimizing what you meant.** This is why the very first habit you must build is to *watch rollouts*, not just curves. Render the policy. Log the actual $(s, a, r, s')$ tuples. The curve is necessary but never sufficient.

The systematic approach — the spine of this whole post — has three moves, borrowed from how good engineers debug any complex system, and detailed in our companion guide [debugging-ai-training-the-six-places-a-bug-hides](/blog/machine-learning/debugging-training/debugging-ai-training-the-six-places-a-bug-hides):

- **Reduce.** Shrink the problem until it is trivial. Can your agent solve a one-state bandit? CartPole-v1? If it cannot solve CartPole in 100k steps with default hyperparameters, your implementation has a bug — full stop. CartPole is the RL equivalent of "does it compile."
- **Isolate.** Change one thing at a time. RL has a dozen interacting hyperparameters and any of them can mask a bug. If you change the learning rate, the network architecture, and the reward scale together, you learn nothing from the result.
- **Instrument.** Log the internal signals (entropy, value loss, gradient norm, Q-value magnitude, KL divergence) that distinguish the six failure modes. The whole point of the diagnostics below is that each failure mode has a *different* signature in these logs.

The rest of this post is the six failure modes, in the order I check them.

## 2. Failure mode 1: Reward signal failure

The reward is the entire specification of what you want. If it is wrong, everything downstream is wrong, and no amount of algorithm tuning will save you. Reward failures come in four flavors.

**Reward too sparse.** If the agent only gets a non-zero reward on success, and success is rare under the initial random policy, the agent never sees a learning signal. The policy gradient $\nabla_\theta J(\theta) = \mathbb{E}\left[\sum_t \nabla_\theta \log \pi_\theta(a_t|s_t)\, R_t\right]$ is identically zero in expectation when $R_t = 0$ for every trajectory the agent has ever experienced. There is literally nothing to learn from. Montezuma's Revenge is the canonical example: thousands of steps between rewards, and naive DQN scores zero.

**Reward too dense (and gameable).** The opposite failure. To fix sparsity, people add shaping rewards — small bonuses for sub-goals. But a poorly designed dense reward gives the agent a way to accumulate points *without* doing the task, exactly like my warehouse robot. This is **Goodhart's law**: when a measure becomes a target, it ceases to be a good measure. The agent optimizes the proxy, not the goal.

**Reward scale wrong.** If rewards are on the order of thousands, the value targets $r + \gamma V(s')$ are huge, the TD errors are huge, the gradients are huge, and the network's weights explode or oscillate. If rewards are on the order of $10^{-4}$, the gradient signal is swamped by noise and the network never moves. Neural networks like inputs and targets roughly in the $[-1, 1]$ to $[-10, 10]$ range; rewards outside that band cause trouble.

**Reward hacking.** The most dangerous, because it is invisible in the metrics by construction. The agent finds an unintended high-reward behavior. CoastRunners (an OpenAI example) is famous: a boat-racing agent learned to drive in a circle hitting the same regeneration targets forever, scoring 20% higher than any human while never finishing the race.

### The diagnostic: plot the reward distribution per episode

The first thing to check is whether the reward signal *has variance*. If every episode returns the same value — or every step returns zero — the agent has no gradient to climb.

```python
import numpy as np
import gymnasium as gym

def diagnose_reward_signal(env_id="MountainCar-v0", n_episodes=200):
    """Run a random policy and inspect the reward distribution."""
    env = gym.make(env_id)
    episode_returns = []
    per_step_rewards = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        ep_return = 0.0
        while not done:
            action = env.action_space.sample()  # random policy
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_return += reward
            per_step_rewards.append(reward)
        episode_returns.append(ep_return)
    env.close()

    returns = np.array(episode_returns)
    steps = np.array(per_step_rewards)
    print(f"Episode return: mean={returns.mean():.3f} "
          f"std={returns.std():.3f} min={returns.min():.1f} max={returns.max():.1f}")
    print(f"Per-step reward: mean={steps.mean():.4f} "
          f"std={steps.std():.4f} nonzero_frac={(steps != 0).mean():.4f}")
    # The smoking gun: a random policy that never sees variance cannot learn.
    if returns.std() < 1e-6:
        print("WARNING: zero return variance under random policy -> reward too sparse")
    return returns
```

Run this *before* you train anything. On MountainCar-v0 you will see that a random policy basically never reaches the flag, so `nonzero_frac` for the "success" signal is essentially zero and `returns.std()` reflects only the step penalty. That tells you up front: this is a sparse-reward problem and you need exploration help or shaping, not a bigger network.

### The fixes

**Reward normalization.** Keep a running estimate of reward mean and standard deviation and normalize. Stable-Baselines3 ships `VecNormalize` for exactly this:

```python
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

env = DummyVecEnv([lambda: gym.make("BipedalWalker-v3")])
# norm_reward divides rewards by a running std estimate (clipped).
env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_reward=10.0)

model = PPO("MlpPolicy", env, learning_rate=3e-4, n_steps=2048,
            gamma=0.99, verbose=1)
model.learn(total_timesteps=1_000_000)
# IMPORTANT: save the running statistics or eval will be wrong.
env.save("vecnormalize.pkl")
```

A subtle trap here, and a great example of why you isolate changes: `VecNormalize` keeps running statistics that must be saved and restored at evaluation time. If you forget, your evaluation observations are normalized with the wrong mean and the agent that scored 300 in training scores 30 in eval — which looks exactly like a distribution-shift failure (mode 6). Always save the normalizer.

**Potential-based reward shaping.** This is the one shaping technique with a theoretical guarantee. Ng, Harada, and Russell (1999) proved that if your shaping reward has the form

$$F(s, a, s') = \gamma\,\Phi(s') - \Phi(s)$$

for any potential function $\Phi$, then the optimal policy is *unchanged*. The shaping only changes how fast you learn, never *what* you learn. Here is why, in two lines. The return with shaping telescopes: $\sum_t \gamma^t F(s_t, a_t, s_{t+1}) = \sum_t (\gamma^{t+1}\Phi(s_{t+1}) - \gamma^t\Phi(s_t)) = -\Phi(s_0)$ (for an episode that ends in an absorbing state with $\Phi = 0$). The total shaping reward depends only on the start state, a constant with respect to the policy, so it cannot change the $\arg\max$ over policies. That is the entire proof, and it is the reason potential-based shaping is the *only* shaping I trust in production.

```python
def potential(state):
    # Domain knowledge: closer to the goal is better. Bounded, smooth.
    goal = np.array([0.5, 0.0])
    return -np.linalg.norm(state[:2] - goal)

def shaped_reward(raw_reward, state, next_state, gamma=0.99):
    # F = gamma * phi(s') - phi(s) : optimal policy provably preserved.
    return raw_reward + gamma * potential(next_state) - potential(state)
```

Compare this to the naive version where you just add `+0.1 * (1 / distance_to_goal)` every step. That naive bonus is *not* potential-based, and an agent will happily learn to hover near the goal collecting the bonus instead of touching it — the reward-hacking pattern in Figure 6.

![A before-and-after comparison contrasting naive dense reward shaping that causes the agent to spin in circles against potential-based shaping that reaches the goal](/imgs/blogs/debugging-rl-agents-common-failure-modes-6.png)

**Curriculum.** For genuinely sparse tasks, start the agent near the goal (where reward is easy to find) and progressively move the start state back. Each stage's policy bootstraps the next. We will see in mode 6 that curricula introduce their own debugging hazard (stale buffer data), so this fix is not free.

| Reward problem | Symptom | First fix | Why it works |
| --- | --- | --- | --- |
| Too sparse | Return variance ≈ 0 under random policy | Curriculum or exploration bonus | Manufactures a learning signal |
| Too dense / gameable | Return climbs but task fails | Potential-based shaping | Provably preserves optimal policy |
| Scale too large | Gradient norm explodes, loss NaN | Reward normalization / clip | Keeps TD targets in a sane range |
| Scale too small | No learning, flat curve | Scale up or normalize | Lifts signal above noise floor |
| Reward hacking | High reward, wrong behavior | Re-specify reward, watch rollouts | Removes the exploit |

## 3. Failure mode 2: Policy collapse and mode collapse

Suppose the reward is fine. The next thing that goes wrong is the policy itself collapsing — converging prematurely to a single action or a tiny set of actions, and getting stuck there. The curve typically improves for a while, then plateaus far below optimal, or rises and then *falls back*.

### The theory: entropy as the exploration budget

A stochastic policy $\pi_\theta(a|s)$ has an entropy $H(\pi_\theta(\cdot|s)) = -\sum_a \pi_\theta(a|s) \log \pi_\theta(a|s)$. Entropy measures how spread-out the action distribution is: a uniform policy over $n$ actions has entropy $\log n$ (maximal), and a deterministic policy has entropy $0$ (minimal). Entropy *is* the policy's exploration budget. When entropy hits zero, the policy is deterministic, it stops exploring, and it can never escape whatever action it has locked onto — even if that action is sub-optimal.

Policy collapse happens because the policy gradient has a positive feedback loop. Suppose at some state the agent slightly prefers action A. It takes A more often, so it gathers more data about A, so its advantage estimate for A becomes more confident, so it takes A even more often. If A is genuinely best, this is convergence. If A merely *looked* best early due to noise, this is collapse to a sub-optimal action — and once entropy is gone, the agent cannot gather the data that would reveal action B is actually better. This is the RL form of the "rich get richer" dynamic.

There is a particularly insidious variant: **the safe sub-optimal policy.** Consider a state where action B has a high mean reward but high variance (sometimes great, sometimes catastrophic), and action A has a low but certain reward. Early in training, before the value estimates are accurate, the variance of B's returns inflates the variance of its gradient. The agent, optimizing average return, can rationally learn to avoid B and lock onto A — and then never gather the data that would teach it B is better on average. The agent learned to be timid. I have seen this destroy trading agents that learned to never take a position because the variance of returns frightened the optimizer into the do-nothing corner. The relationship between variance and learning here echoes the bias-variance dynamics covered in [policy-gradient-methods-from-reinforce-to-ppo](/blog/machine-learning/reinforcement-learning/policy-gradient-methods-from-reinforce-to-ppo).

### The diagnostic: plot entropy during training

Entropy is the single most informative scalar to log for policy-gradient methods. Its trajectory tells you the story:

- Entropy decaying *gradually* toward a non-zero floor as return rises: healthy convergence.
- Entropy *crashing* to near zero in the first few thousand updates while return is still low: collapse. The agent committed before it explored.
- Entropy *oscillating* wildly: your updates are too large (see mode 9 on PPO KL).

![A before-and-after diagram showing policy entropy collapsing to near zero without an entropy bonus versus holding steady at 1.5 with an entropy bonus and improving return](/imgs/blogs/debugging-rl-agents-common-failure-modes-3.png)

```python
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

def policy_loss_with_entropy(logits, actions, advantages, ent_coef=0.01):
    """Standard policy-gradient loss with an entropy bonus and logging."""
    dist = Categorical(logits=logits)
    log_probs = dist.log_prob(actions)
    pg_loss = -(log_probs * advantages.detach()).mean()
    entropy = dist.entropy().mean()
    # Subtract entropy * coef: maximizing entropy = staying exploratory.
    loss = pg_loss - ent_coef * entropy
    # LOG THIS EVERY UPDATE. It is your collapse early-warning system.
    return loss, entropy.item()
```

### The fixes

**Entropy bonus.** Add $+\beta\, H(\pi_\theta)$ to the objective. The gradient of this term pushes the policy *toward* uniform, counteracting premature commitment. In Stable-Baselines3 this is the `ent_coef` hyperparameter:

```python
from stable_baselines3 import PPO
# ent_coef raises the entropy floor. 0.0 (the default) collapses on hard envs.
model = PPO("MlpPolicy", "LunarLander-v3",
            ent_coef=0.01,      # the collapse fix
            learning_rate=3e-4,
            n_steps=1024, batch_size=64, gamma=0.999,
            verbose=1)
model.learn(total_timesteps=1_000_000)
```

On LunarLander-v3, the default `ent_coef=0.0` works for PPO because the env is forgiving, but I routinely bump it to `0.01` on environments with deceptive local optima. The cost is slightly slower final convergence; the benefit is not getting stuck. That is the trade-off in one sentence.

**KL penalty / trust region.** Instead of (or in addition to) the entropy bonus, constrain how far each update can move the policy. PPO does this with a clipped surrogate objective; TRPO does it with an explicit KL constraint. We will dig into the PPO machinery in mode 9. The intuition: if every update is small, a single noisy batch cannot collapse the policy in one step.

**Temperature / SAC.** Soft Actor-Critic makes entropy a first-class part of the objective — it maximizes return *plus* entropy, with an automatically tuned temperature $\alpha$. We will see in mode 10 how that auto-tuning can itself break.

#### Worked example: rescuing a collapsed LunarLander agent

I had a PPO LunarLander agent plateau at a mean return of about 40 (a passing score is 200). The entropy log showed the answer immediately: entropy had crashed from 1.38 (near the maximum of $\log 4 \approx 1.386$ for 4 actions) down to 0.05 within 60k steps. The action histogram confirmed it — the agent fired the main engine 94% of the time and basically never used the side thrusters. It had collapsed onto "thrust up and hope." The fix was one line: `ent_coef=0.0` → `ent_coef=0.01`. With the bonus, entropy stabilized around 0.6, the agent kept experimenting with side thrusters, and mean return climbed to 250 over the next 800k steps. One hyperparameter, diagnosed in thirty seconds by looking at the right scalar. That is the entire value proposition of instrumentation.

## 4. Failure mode 3: Value function divergence

Now we leave policy-gradient land for value-based methods (DQN and friends), where the characteristic failure is the value estimate diverging to infinity. You will see max Q-values of 500, then 5,000, then NaN, on an environment where the true value can be at most, say, 20.

### The theory: the deadly triad

Sutton and Barto named the three ingredients that, when combined, can make value learning diverge. They call it the **deadly triad**:

1. **Function approximation** — using a neural network (or any parametric function) for $Q$, so updating $Q(s, a)$ for one state-action also changes $Q$ for others.
2. **Bootstrapping** — updating an estimate toward a target that is itself an estimate: $Q(s,a) \leftarrow r + \gamma \max_{a'} Q(s', a')$.
3. **Off-policy training** — learning about the greedy policy while behaving according to a different (exploratory) policy, so the distribution of updates does not match the policy being evaluated.

Each alone is fine. Tabular Q-learning (no function approximation) converges. Monte Carlo with function approximation (no bootstrapping) converges. On-policy TD (no off-policy mismatch) converges. But put all three together — which is *exactly* what DQN does — and the update is no longer a contraction mapping, so there is no fixed-point guarantee, and the iterates can run off to infinity.

Here is the mechanism in concrete terms. The Q-learning target is $y = r + \gamma \max_{a'} Q_\theta(s', a')$. If you compute gradients treating $y$ as a function of the *same* $\theta$ you are updating, then raising $Q_\theta(s, a)$ to match $y$ also raises $y$ (because $y$ depends on $Q_\theta$). You are chasing a target that runs away from you as you approach it. With the $\max$ on top, the overestimation compounds. This is why Q-values explode.

There is a second, subtler bias even before divergence: **overestimation from the max operator.** Because $\mathbb{E}[\max_a Q] \geq \max_a \mathbb{E}[Q]$ by Jensen's inequality, taking the max over noisy Q-estimates produces an upward-biased target. Every update injects a little optimism, and the optimism accumulates.

### The diagnostic: plot max and mean Q-value over time

This is the cleanest diagnostic in all of RL. Log the maximum and mean predicted Q-value over your batch every update step.

- Mean Q tracking a plausible value (bounded by $r_{\max}/(1-\gamma)$): healthy.
- Mean Q climbing without bound, or max Q an order of magnitude above the theoretical ceiling: the triad is biting.

For a reward bounded by $r_{\max} = 1$ and $\gamma = 0.99$, the theoretical max value is $\frac{1}{1 - 0.99} = 100$. If you see Q-values of 800 on such an env, you have a divergence problem, period.

![A graph comparing Q-value behavior with no target network diverging to infinity, a target network stabilizing values, and Double DQN tracking true values](/imgs/blogs/debugging-rl-agents-common-failure-modes-4.png)

```python
import torch
import torch.nn as nn

def dqn_diagnostics(q_net, batch_states, gamma=0.99, r_max=1.0):
    """Log the signals that expose value divergence."""
    with torch.no_grad():
        q_values = q_net(batch_states)          # (B, n_actions)
        max_q = q_values.max().item()
        mean_q = q_values.mean().item()
    theoretical_ceiling = r_max / (1.0 - gamma)
    # If max_q >> ceiling, the deadly triad is diverging your value fn.
    danger = max_q > 2.0 * theoretical_ceiling
    print(f"mean_Q={mean_q:.2f} max_Q={max_q:.2f} "
          f"ceiling={theoretical_ceiling:.1f} danger={danger}")
    return danger
```

### The fixes

**Target network.** This is the single most important stability trick in deep value learning, introduced by Mnih et al. in the 2015 Atari DQN paper. Keep a *frozen* copy $Q_{\theta^-}$ of the network for computing targets, and only update it every $C$ steps. Now the target $y = r + \gamma \max_{a'} Q_{\theta^-}(s', a')$ does not move when you update $\theta$, so you are no longer chasing a runaway target.

```python
import copy
import torch
import torch.nn.functional as F

class DQNTrainer:
    def __init__(self, q_net, lr=1e-4, gamma=0.99, target_update=1000):
        self.q_net = q_net
        self.target_net = copy.deepcopy(q_net)   # frozen target
        self.target_net.eval()
        self.opt = torch.optim.Adam(q_net.parameters(), lr=lr)
        self.gamma = gamma
        self.target_update = target_update
        self.step_count = 0

    def update(self, states, actions, rewards, next_states, dones):
        q = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            # Target uses the FROZEN network -> no runaway target.
            next_q = self.target_net(next_states).max(dim=1).values
            target = rewards + self.gamma * next_q * (1.0 - dones)
        loss = F.smooth_l1_loss(q, target)        # Huber: robust to outliers
        self.opt.zero_grad()
        loss.backward()
        # Gradient clipping: a second line of defense against blowups.
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=10.0)
        self.opt.step()
        self.step_count += 1
        if self.step_count % self.target_update == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
        return loss.item()
```

**Double DQN.** Decouple action *selection* from action *evaluation* to kill the max-operator overestimation bias. Select the next action with the online network but evaluate it with the target network: $y = r + \gamma\, Q_{\theta^-}(s', \arg\max_{a'} Q_\theta(s', a'))$. This costs one extra forward pass and removes most of the optimism bias (van Hasselt et al., 2016).

```python
with torch.no_grad():
    # Online net picks the action; target net scores it. That decoupling
    # removes the upward bias of taking max over a single noisy estimate.
    next_actions = self.q_net(next_states).argmax(dim=1, keepdim=True)
    next_q = self.target_net(next_states).gather(1, next_actions).squeeze(1)
    target = rewards + self.gamma * next_q * (1.0 - dones)
```

**Gradient clipping and a lower critic learning rate.** When you suspect divergence is starting, clip the gradient norm (shown above, `max_norm=10.0`) and drop the critic learning rate by 10× (e.g. $10^{-4} \to 10^{-5}$). Divergence is a runaway positive-feedback process; a smaller step size gives the target network time to stabilize between updates. The recovery recipe I use is: target network first, then Double DQN, then if it still creeps, cut the learning rate. In that order, because that is the order of effect size.

| Trick | Removes which triad ingredient | Cost | Typical effect |
| --- | --- | --- | --- |
| Target network | Tames bootstrapping (frozen target) | Memory for a 2nd net | Essential; without it DQN rarely works |
| Double DQN | Off-policy max overestimation | One extra forward pass | Q bias from +30% to <5% |
| Gradient clip | Caps any single bad update | Negligible | Prevents NaN spikes |
| Lower critic LR | Slows feedback loop | Slower convergence | Last-resort stabilizer |

## 5. Failure mode 4: Exploration starvation

The agent stops improving and parks at a plateau well below optimal. Unlike policy collapse (mode 2), the policy here may still be stochastic — it just never tries the actions that would lead somewhere new. The agent has explored the easy part of the space, exploited what it found, and run out of curiosity.

### The theory: exploration-exploitation under a vanishing budget

Every RL algorithm must balance exploring (gathering information about unknown actions) against exploiting (taking the best-known action). The classic schedule is $\epsilon$-greedy: with probability $\epsilon$ take a random action, otherwise take the greedy one. The standard practice is to *decay* $\epsilon$ from 1.0 to a small floor (say 0.05) over training. The bug is decaying it *too fast* or to *too low a floor* before the agent has found the good behavior. Once $\epsilon$ is near zero, the agent only ever takes greedy actions, and if the greedy action is a local optimum, it is stuck forever.

For methods that explore through posterior uncertainty (Thompson sampling, Bayesian bandits), the analog failure is **posterior collapse**: the posterior variance shrinks toward zero as data accumulates, so the agent stops sampling exploratory actions. If the posterior collapses onto a wrong belief — which happens when early data was unlucky — the agent commits to a sub-optimal arm and never gathers the data that would correct it.

The deep reason exploration is hard is that in large state spaces, undirected exploration ($\epsilon$-greedy random actions) takes exponentially long to reach distant rewarding states. Random walks diffuse as $\sqrt{t}$; reaching a goal $d$ steps away by random walk takes on the order of $d^2$ steps *if you are lucky enough that the geometry cooperates*, and far longer with obstacles. This is why Montezuma's Revenge defeated $\epsilon$-greedy DQN entirely.

### The diagnostic: action entropy and action-frequency histogram

Two complementary views:

- **Action entropy over training.** Distinct from policy entropy in mode 2 — here we mean the empirical entropy of the actions actually *taken* in recent rollouts. If it sits at a low constant while return plateaus, the agent has stopped exploring.
- **Action-frequency histogram.** Bin the actions taken over the last N episodes. If 3 out of 18 actions account for 95% of behavior and the agent is plateaued, you have starvation. The remaining actions are unexplored.

```python
import numpy as np
from collections import Counter

def action_frequency_report(action_log, n_actions):
    """action_log: list of ints from recent rollouts."""
    counts = Counter(action_log)
    total = len(action_log)
    probs = np.array([counts.get(a, 0) / total for a in range(n_actions)])
    # Empirical entropy of the behavior policy.
    nz = probs[probs > 0]
    entropy = -(nz * np.log(nz)).sum()
    max_entropy = np.log(n_actions)
    print(f"behavior entropy={entropy:.3f} / max {max_entropy:.3f} "
          f"({entropy / max_entropy:.0%} of uniform)")
    unused = [a for a in range(n_actions) if counts.get(a, 0) == 0]
    if unused:
        print(f"NEVER-TAKEN actions: {unused}  <- exploration starvation")
    return probs
```

If `action_frequency_report` prints never-taken actions while your return is flat, that is your confirmation.

### The fixes

**Raise epsilon or slow its decay.** The cheapest fix. If $\epsilon$ decayed to 0.01 over the first 100k steps and you train for 5M, you spent 98% of training with almost no exploration. Stretch the decay schedule or raise the floor.

```python
def epsilon_schedule(step, eps_start=1.0, eps_end=0.05, decay_steps=1_000_000):
    # Linear decay over a LONG horizon, with a non-trivial floor.
    frac = min(1.0, step / decay_steps)
    return eps_start + frac * (eps_end - eps_start)
```

**Curiosity / intrinsic motivation.** For genuinely hard exploration, add an intrinsic reward that rewards visiting novel states. Two workhorses:

- **ICM (Intrinsic Curiosity Module)**: train a forward model that predicts the next state's features; reward the agent proportionally to the prediction error. High error = unfamiliar state = worth exploring.
- **RND (Random Network Distillation)**: keep a fixed random network and train a predictor to match it; the predictor's error is low on states seen often and high on novel states. Use that error as a bonus. RND is simpler and was the first method to beat Montezuma's Revenge.

```python
import torch
import torch.nn as nn

class RND(nn.Module):
    """Random Network Distillation intrinsic reward."""
    def __init__(self, obs_dim, hidden=128):
        super().__init__()
        # Target is FROZEN random; predictor learns to imitate it.
        self.target = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.predictor = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        for p in self.target.parameters():
            p.requires_grad = False

    def intrinsic_reward(self, obs):
        with torch.no_grad():
            t = self.target(obs)
        p = self.predictor(obs)
        # Novel states -> high predictor error -> high curiosity bonus.
        return ((p - t) ** 2).mean(dim=1)

    def update_loss(self, obs):
        t = self.target(obs).detach()
        p = self.predictor(obs)
        return ((p - t) ** 2).mean()
```

**UCB exploration.** For smaller action spaces, Upper Confidence Bound chooses the action maximizing $Q(s,a) + c\sqrt{\frac{\ln N}{n(s,a)}}$, where $n(s,a)$ is how often you have tried that action. The bonus term shrinks as you gather data, so exploration is *directed* toward under-tried actions rather than random. This is provably efficient for bandits and a strong baseline for small discrete problems.

#### Worked example: breaking a MountainCar plateau

MountainCar-v0 is a textbook starvation trap. The reward is $-1$ per step until you reach the flag (a true sparse reward), and the car cannot reach the flag by driving straight — it must rock back and forth to build momentum, a behavior $\epsilon$-greedy almost never stumbles into. A vanilla DQN with $\epsilon$ decaying to 0.05 over 50k steps plateaus at $-200$ (it times out every episode, never reaching the flag). The action histogram shows the agent oscillating between "full left" and "full right" almost randomly, never holding a direction long enough to build the swing. Adding an RND bonus scaled at 0.1 changed the picture: the intrinsic reward pushed the car into never-before-seen high-velocity, high-position states, it accidentally crested the hill around step 80k, and from there the extrinsic signal took over. Final performance went from $-200$ (failure) to about $-110$ (solved). The fix was not a bigger network — it was a better exploration signal, identified by the never-taken-state pattern in the diagnostics.

## 6. Failure mode 5: Environment bugs

This is the failure mode that has cost me, personally, the most wall-clock time, and it is the one beginners suspect last and should suspect first. **Your environment almost certainly has a bug.** RL environments are stateful simulators with tricky bookkeeping around rewards, terminations, and observations, and the bugs are subtle — the kind that do not crash, do not throw, and produce training curves that look *almost* right. My market-maker time-traveler from the intro was an environment bug.

Here are the four I see most often:

**Off-by-one in reward timing.** The reward for action $a_t$ gets attached to step $t+1$ (or $t-1$). The agent then learns to associate the wrong action with the reward — credit assignment is corrupted at the source. This is devastating because the agent *does* learn something; it just learns a shifted, wrong policy.

**Observation leak.** Information about the future (or about the reward directly) sneaks into the observation. The classic is including a feature that is computed from the next state. The agent exploits it, scores beautifully in training, and falls apart when the leak is absent at deployment. This is the single most common cause of "great in sim, useless in real."

**Action not normalized / wrong scale.** For continuous control, if your environment expects actions in $[-1, 1]$ but your policy outputs in $[-10, 10]$ (or vice versa), every action saturates and the agent's fine control is destroyed. SB3's `MlpPolicy` assumes a symmetric action space; mismatches here silently cripple learning.

**Wrong done signal.** Confusing *termination* (the episode ended because the agent reached a terminal state) with *truncation* (the episode was cut off by a time limit). This matters for bootstrapping: you should bootstrap the value $\gamma V(s')$ across a *truncation* (the episode would have continued) but NOT across a *termination* (there is no future). Gymnasium split these into two return values precisely because conflating them is so common and so damaging. Getting this wrong inflates or deflates value targets in ways that look like value divergence (mode 3).

### The diagnostic: log five steps and read them by hand

There is no clever automated substitute for this. Run a fixed (e.g. all-zeros or scripted) policy, log the full $(s, a, r, s', \text{terminated}, \text{truncated})$ tuple for the first five steps, and *read them with your eyes*. Does the reward make sense for that action? Does $s'$ actually follow from $s$ and $a$? Does `terminated` fire when it should?

```python
import gymnasium as gym

def manual_env_trace(env_id, n_steps=5, seed=0):
    env = gym.make(env_id)
    obs, info = env.reset(seed=seed)
    print(f"reset -> obs={obs}")
    for t in range(n_steps):
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, info = env.step(action)
        print(f"\nstep {t}")
        print(f"  action     = {action}")
        print(f"  reward     = {reward}")
        print(f"  terminated = {terminated}  truncated = {truncated}")
        print(f"  obs   (s)  = {obs}")
        print(f"  next  (s') = {next_obs}")
        # Read these by hand. Does r match (s, a)? Does s' follow from s, a?
        obs = next_obs
        if terminated or truncated:
            obs, info = env.reset(seed=seed + 1)
            print("  --- episode end, reset ---")
    env.close()
```

### The automated backstop: Gymnasium's env checker

Before you trust a custom environment, run it through Gymnasium's checker, which validates the API contract — observation/action space conformance, reset/step return signatures, dtype consistency:

```python
from gymnasium.utils.env_checker import check_env
from my_envs import WarehouseRobotEnv

env = WarehouseRobotEnv()
# Raises on space/dtype/return-signature violations. Run this FIRST.
check_env(env)
```

`check_env` catches the structural bugs (wrong dtype, out-of-bounds observations, malformed return tuples). It does *not* catch semantic bugs (off-by-one reward, observation leak) — those you find only by reading traces and watching rollouts.

### The fixes: assertions and invariants

Bake assertions into the environment so a bug *crashes* instead of silently corrupting training:

```python
import numpy as np

class SafeEnvWrapper(gym.Wrapper):
    """Catch the silent environment bugs by asserting invariants."""
    def step(self, action):
        # Action-scale bug: assert the action is in-bounds.
        assert self.action_space.contains(action), \
            f"action {action} outside {self.action_space}"
        obs, reward, terminated, truncated, info = self.env.step(action)
        # Observation leak / NaN guard.
        assert self.observation_space.contains(obs), \
            f"obs outside declared space -> possible leak or scale bug"
        assert np.isfinite(reward), f"non-finite reward {reward}"
        # Done-signal sanity: cannot be both at once in most envs.
        assert not (terminated and truncated), \
            "terminated AND truncated in same step -> done-signal bug"
        return obs, reward, terminated, truncated, info
```

And the correct bootstrapping logic, which is where the done-signal bug actually does its damage:

```python
# Bootstrap across truncation (episode would continue) but NOT termination.
# Getting this backwards corrupts every value target near episode boundaries.
if truncated and not terminated:
    target = reward + gamma * value(next_obs)   # episode cut off: bootstrap
elif terminated:
    target = reward                              # true end: no future value
else:
    target = reward + gamma * value(next_obs)   # mid-episode: bootstrap
```

The meta-lesson: when an RL agent fails, the prior probability that the bug is in your environment is higher than the probability it is in your algorithm. Check the environment first. Run `check_env`, trace five steps by hand, and only then start tuning the learner.

## 7. Failure mode 6: Distribution shift and non-stationarity

The final failure mode is the most production-relevant and the hardest to catch in a single training run. The data the agent learns from no longer matches the situation it acts in. Performance is good in sim and bad in real, or good before a curriculum transition and bad after, or good against one opponent and bad against the next.

### The theory: why a stale buffer poisons off-policy learning

Off-policy methods (DQN, SAC) train from a replay buffer that holds transitions collected by *older* policies. This is normally fine — that is the whole point of off-policy learning. But it relies on an assumption: the environment's dynamics and reward are stationary. When the environment *changes during training*, the buffer fills with transitions from a world that no longer exists, and the agent optimizes for a stale reality.

Three common sources of non-stationarity:

- **Curriculum progression.** You change the task (move the goal, raise the difficulty). Old transitions describe the old task.
- **Multi-agent learning.** Other agents' policies change, so from any one agent's view the environment is non-stationary by construction. This is the central difficulty of multi-agent RL, covered in depth in [multi-agent-rl-when-agents-learn-together](/blog/machine-learning/reinforcement-learning/multi-agent-rl-when-agents-learn-together).
- **Sim-to-real.** The real world differs from the simulator. The policy's training distribution (sim) and deployment distribution (real) diverge.

The mathematical handle on this is **importance sampling**. When you evaluate a target policy $\pi$ using data from a behavior policy $\mu$, the correct correction is the importance weight $\rho = \frac{\pi(a|s)}{\mu(a|s)}$. When $\pi$ and $\mu$ are close, $\rho \approx 1$ and the off-policy estimate is fine. When they diverge — which is exactly what stale data means — $\rho$ explodes or collapses, the variance of the estimate blows up, and the updates become garbage. Monitoring importance-weight statistics is your early-warning system for off-policy divergence, the same machinery PPO uses (mode 9).

### The diagnostic: importance-weight distribution and buffer age

```python
import numpy as np
import torch

def offpolicy_divergence_report(new_log_probs, old_log_probs):
    """Importance weights rho = pi_new / pi_old. Near 1.0 is healthy."""
    log_ratio = (new_log_probs - old_log_probs).detach()
    ratio = torch.exp(log_ratio)
    # An approximate KL estimate (Schulman's k3 estimator).
    approx_kl = ((ratio - 1) - log_ratio).mean().item()
    print(f"IS weight: mean={ratio.mean():.3f} max={ratio.max():.3f} "
          f"approx_KL={approx_kl:.4f}")
    # Heavy right tail or KL >> target -> stale / off-policy data.
    if ratio.max() > 10.0 or approx_kl > 0.1:
        print("WARNING: large policy-data mismatch -> distribution shift")
    return approx_kl

def buffer_age_report(buffer_step_tags, current_step):
    """How old (in env steps) is the data we are training on?"""
    ages = current_step - np.array(buffer_step_tags)
    print(f"buffer age: median={np.median(ages):.0f} "
          f"p95={np.percentile(ages, 95):.0f} steps")
    return ages
```

If, right after a curriculum transition, your buffer's p95 age is older than the transition point, you are training on a dead world.

### The fixes

**Clear (or down-weight) the buffer on a curriculum transition.** The bluntest, most reliable fix. When the task changes, dump stale transitions:

```python
class CurriculumBuffer:
    def __init__(self, capacity):
        from collections import deque
        self.buffer = deque(maxlen=capacity)

    def on_curriculum_transition(self):
        # Old task's transitions are now misleading. Drop them.
        self.buffer.clear()
        print("curriculum transition -> buffer cleared")
```

**Use a smaller buffer.** A smaller replay buffer holds fewer, more-recent transitions, so it naturally tracks a slowly-changing environment. The trade-off is sample efficiency (you reuse data less) against recency (your data is fresher). On non-stationary problems, recency usually wins.

**Prioritize recent data.** A middle path: keep a large buffer but bias sampling toward recent transitions, so the agent emphasizes the current world without throwing away all history.

**For sim-to-real: domain randomization.** Train across a *distribution* of simulators (randomized friction, mass, latency, sensor noise) so the real world is just one more sample from a distribution the agent already handles. This is how OpenAI's Dactyl learned in-hand manipulation that transferred to a physical Shadow Hand.

#### Worked example: the curriculum cliff

A robotics team I advised had a manipulation agent that learned to grasp a 5cm cube beautifully — 92% success. They advanced the curriculum to a 3cm cube and watched success *crash* to 11% and stay there for 500k steps, far worse than training on 3cm from scratch. The buffer-age report told the story: 80% of the replay buffer still held 5cm-cube transitions, so the agent kept reinforcing a grasp width tuned for the bigger cube. Clearing the buffer at the transition recovered learning — success climbed back past 70% within 200k steps. The non-obvious lesson: a curriculum, the standard *fix* for sparse rewards (mode 1), introduces a distribution-shift *bug* (mode 6) if you do not manage the buffer. Failure modes interact, which is exactly why the diagnosis tree in Figure 2 routes you to one mode at a time.

![A decision tree routing training symptoms to one of six root-cause failure modes from reward signal to distribution shift](/imgs/blogs/debugging-rl-agents-common-failure-modes-2.png)

## 8. The diagnostic toolkit

Every fix above started from a *log*. The discipline that makes RL debugging tractable is logging the right signals at the right layers, so the failure mode announces itself. Figure 7 shows the four-layer instrumentation stack I set up on day one of any new RL project.

![A layered stack diagram showing environment, algorithm, training, and evaluation instrumentation layers for reinforcement learning](/imgs/blogs/debugging-rl-agents-common-failure-modes-7.png)

The logging recipe, layer by layer:

- **Environment layer:** raw reward (per step and per episode), episode length, the `terminated`/`truncated` flags, step counter. This is where environment bugs (mode 5) surface.
- **Algorithm layer:** policy entropy (mode 2), value loss, actor loss, KL divergence between consecutive policies (modes 2 and 9), max/mean Q-value (mode 3).
- **Training layer:** gradient norm (modes 1 and 3), learning rate, replay buffer fill fraction and age (mode 6).
- **Evaluation layer:** mean episode return on a *deterministic* eval policy, success rate, and — critically — performance on *held-out* environments to catch overfitting and distribution shift.

Here is the TensorBoard/WandB logging block I drop into every training loop:

```python
import torch
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

def log_step(writer, step, *, ep_return, ep_len, entropy, value_loss,
             actor_loss, approx_kl, grad_norm, lr, buffer_fill, max_q):
    # Environment layer
    writer.add_scalar("env/episode_return", ep_return, step)
    writer.add_scalar("env/episode_length", ep_len, step)
    # Algorithm layer  (entropy & KL catch collapse; max_q catches divergence)
    writer.add_scalar("algo/policy_entropy", entropy, step)
    writer.add_scalar("algo/value_loss", value_loss, step)
    writer.add_scalar("algo/actor_loss", actor_loss, step)
    writer.add_scalar("algo/approx_kl", approx_kl, step)
    writer.add_scalar("algo/max_q", max_q, step)
    # Training layer
    writer.add_scalar("train/grad_norm", grad_norm, step)
    writer.add_scalar("train/learning_rate", lr, step)
    writer.add_scalar("train/buffer_fill", buffer_fill, step)

def grad_norm(model):
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.data.norm(2).item() ** 2
    return total ** 0.5
```

### The smoke tests: bandit, then CartPole

Before you trust *any* RL code, run two smoke tests. They take minutes and they catch the majority of implementation bugs.

**The one-state bandit.** The simplest possible RL problem: one state, $k$ actions, fixed reward per action, no transitions. If your agent cannot learn to pick the highest-reward arm of a 2-armed bandit, the bug is in your update rule, not your environment or your exploration. There is nowhere else for it to hide.

```python
import torch
import torch.nn as nn
from torch.distributions import Categorical

def bandit_smoke_test(n_actions=2, best_action=1, steps=2000):
    """If your policy-gradient code can't solve this, it is broken."""
    logits = nn.Parameter(torch.zeros(n_actions))
    opt = torch.optim.Adam([logits], lr=0.1)
    for t in range(steps):
        dist = Categorical(logits=logits)
        action = dist.sample()
        reward = 1.0 if action.item() == best_action else 0.0
        loss = -dist.log_prob(action) * reward     # REINFORCE
        opt.zero_grad(); loss.backward(); opt.step()
    final = Categorical(logits=logits).probs[best_action].item()
    print(f"P(best action) = {final:.3f}  (should be > 0.95)")
    assert final > 0.95, "policy-gradient update is BROKEN"
```

**CartPole-v1 as a test fixture.** CartPole is the "hello world" of deep RL. Default PPO solves it (return 500/500) in well under 100k steps. If yours does not, do not even *look* at your hard environment — the bug is in your agent. Keep a CartPole run as a regression test you can fire any time you touch the learner.

```python
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
import gymnasium as gym

def cartpole_regression_test():
    model = PPO("MlpPolicy", "CartPole-v1", verbose=0)
    model.learn(total_timesteps=50_000)
    mean_r, std_r = evaluate_policy(model, gym.make("CartPole-v1"),
                                    n_eval_episodes=20)
    print(f"CartPole eval: {mean_r:.1f} +/- {std_r:.1f}")
    assert mean_r > 450, "agent broken: cannot solve CartPole"
```

The principle behind both: **reduce until trivial.** A bug that is invisible on your 200-dimensional trading environment is glaring on a 2-armed bandit.

## 9. Debugging policy gradient methods (PPO/A3C)

Policy-gradient methods have their own characteristic failure signatures beyond the generic six. The root cause is almost always **updates that are too large.**

### The theory: why a single big update destroys a policy gradient

The policy gradient theorem gives $\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}\!\left[\nabla_\theta \log \pi_\theta(a|s)\, A^\pi(s,a)\right]$, where $A$ is the advantage. This gradient is only valid *locally* — it is the gradient of the return *under the current policy's state distribution*. If you take a large step, you move to a policy whose state distribution is different, so the gradient you computed no longer points uphill. Take too large a step and you can fall off a cliff: the new policy visits states you have no data for, the advantage estimates are garbage there, and performance collapses. This is the precise reason PPO and TRPO exist — to bound the step size so the local gradient stays valid.

PPO's clipped surrogate objective is the practical solution. Let $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}$ be the probability ratio. PPO maximizes

$$L^{\text{CLIP}}(\theta) = \mathbb{E}_t\!\left[\min\!\big(r_t(\theta)\,A_t,\; \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\,A_t\big)\right]$$

The clip removes the incentive to move the ratio beyond $[1-\epsilon, 1+\epsilon]$ (typically $\epsilon = 0.2$). When the advantage is positive and the ratio already exceeds $1+\epsilon$, the objective flattens — no more gradient — so the update cannot push the policy arbitrarily far in one step.

```python
import torch

def ppo_clip_loss(new_log_probs, old_log_probs, advantages, clip_eps=0.2):
    ratio = torch.exp(new_log_probs - old_log_probs)
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
    # The min() makes the objective pessimistic: it ignores ratio moves
    # that would over-improve, so a single batch can't blow up the policy.
    loss = -torch.min(unclipped, clipped).mean()
    # Diagnostics that tell you if PPO itself is misbehaving:
    with torch.no_grad():
        approx_kl = ((ratio - 1) - (new_log_probs - old_log_probs)).mean()
        clip_frac = ((ratio - 1).abs() > clip_eps).float().mean()
    return loss, approx_kl.item(), clip_frac.item()
```

### The diagnostics specific to PPO

**KL divergence check.** This is the PPO-specific tripwire. After each update, measure the approximate KL between the new and old policy. If `approx_kl` exceeds roughly $1.5\times$ your target KL (default target around 0.01–0.02), the update was too aggressive — something is wrong with your learning rate, your advantage normalization, or your reward scale. SB3 exposes `target_kl` to early-stop the update epoch when this happens:

```python
from stable_baselines3 import PPO
model = PPO("MlpPolicy", "BipedalWalker-v3",
            n_steps=2048, batch_size=64, n_epochs=10,
            clip_range=0.2,
            target_kl=0.02,        # early-stop the epoch if KL blows past this
            learning_rate=3e-4, verbose=1)
```

A rule I live by: if `approx_kl` per update is consistently above 0.05 while your `target_kl` is 0.02, stop and find out why before you waste a training run. Common causes: learning rate too high, advantages not normalized, or a reward-scale problem (mode 1) feeding huge advantages into the ratio.

**Clip fraction.** If `clip_frac` is near zero, your updates are tiny and clipping does nothing (you could train faster). If it is above ~0.3, a large fraction of your updates are being clipped, meaning the policy is trying to move too far — lower the learning rate.

**Value function lag.** In actor-critic methods the advantage $A_t = R_t - V(s_t)$ depends on an accurate critic. If the critic is slow to converge — and on some environments it takes 10× the actor's updates — your advantages are biased and the actor learns from bad signal. The diagnostic is the *explained variance* of the value function: $1 - \frac{\text{Var}(R - V)}{\text{Var}(R)}$. If explained variance is near zero or negative, your critic is useless and the actor is flying blind. The fix is usually a higher critic learning rate or more critic update epochs per actor update.

**Actor-critic gradient interference.** When actor and critic share a network trunk, their gradients can fight — the value loss (often large) can dominate the policy loss and corrupt the shared features. The fix is to scale the value loss down (`vf_coef`, default 0.5) or use separate networks for actor and critic, which trades parameter efficiency for stability.

## 10. Debugging Q-learning methods (DQN/SAC)

We covered value divergence in mode 3; here are the deeper, method-specific pathologies of value-based deep RL.

### The deadly triad in practice: recognizing and recovering

The signs the triad is biting, in increasing severity: max Q-values creeping above the theoretical ceiling, then climbing without bound, then NaN. The recovery recipe, in order:

1. Confirm a target network exists and updates infrequently (every 1k–10k steps, or a soft update with $\tau = 0.005$).
2. Add Double DQN.
3. If Q still creeps, cut the critic learning rate by 10× (e.g. $2.5\times10^{-4} \to 2.5\times10^{-5}$).
4. Clip the gradient norm at 10.
5. As a last resort, clip the reward to $[-1, 1]$ (the original Atari DQN did this), which bounds the value range outright.

```python
# Soft target update (Polyak averaging): a smoother alternative to hard copies.
def soft_update(target_net, online_net, tau=0.005):
    for t_param, o_param in zip(target_net.parameters(), online_net.parameters()):
        # tau small -> target moves slowly -> stable bootstrap target.
        t_param.data.copy_(tau * o_param.data + (1 - tau) * t_param.data)
```

### SAC temperature auto-tuning failures

Soft Actor-Critic maximizes return plus entropy, $J = \mathbb{E}[\sum_t r_t + \alpha H(\pi(\cdot|s_t))]$, where the temperature $\alpha$ trades off reward against exploration. Modern SAC tunes $\alpha$ automatically by gradient descent toward a target entropy (usually $-\dim(\mathcal{A})$). This auto-tuning is wonderful when it works and a debugging nightmare when it does not:

- **$\alpha \to 0$:** the entropy term vanishes, SAC degenerates into a deterministic policy, and you get policy collapse (mode 2). Usually caused by a target entropy set too high (the optimizer drives $\alpha$ down to compensate) or a reward scale so large that the entropy term is negligible by comparison.
- **$\alpha \to \infty$:** the entropy term dominates, the policy stays near-uniform and never exploits, and return flatlines. Usually a target entropy set too low or a reward scale too small.

The diagnostic is simply to **log $\alpha$**. A healthy SAC run shows $\alpha$ starting around 0.1–1.0 and settling to a stable positive value. If $\alpha$ shoots to zero or explodes, fix the reward scale first (mode 1, again — reward scale is the root of an astonishing number of failures) and the target entropy second.

```python
import torch
import torch.nn.functional as F

class SACTemperature:
    """Auto-tuned entropy temperature with the diagnostic logged."""
    def __init__(self, action_dim, lr=3e-4):
        # Optimize log_alpha for numerical stability (alpha = exp(log_alpha)).
        self.log_alpha = torch.zeros(1, requires_grad=True)
        self.opt = torch.optim.Adam([self.log_alpha], lr=lr)
        self.target_entropy = -float(action_dim)  # standard heuristic

    def update(self, log_probs):
        alpha = self.log_alpha.exp()
        # Push alpha so that current entropy matches the target.
        loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
        self.opt.zero_grad(); loss.backward(); self.opt.step()
        a = alpha.item()
        # LOG THIS. alpha -> 0 means collapse; alpha exploding means no exploit.
        if a < 1e-4 or a > 100:
            print(f"WARNING: SAC alpha={a:.5f} unstable -> check reward scale")
        return a
```

### The churn problem

A subtler deep-Q pathology: **churn.** Because a neural network generalizes, updating $Q$ for one state-action perturbs $Q$ for *many others* — including the bootstrap targets of unrelated transitions. The targets churn faster than the policy can track them, and learning stalls or wobbles even with a target network. The signature is a value loss that refuses to decrease and a policy that keeps changing its action preferences without improving return. Mitigations include larger target-network update intervals, smaller learning rates, and architectural choices (layer normalization, smaller networks) that reduce generalization-induced interference. Churn is an active research area; the practical takeaway is that a value loss that won't settle, combined with a wandering policy, points at churn rather than at any of the six primary modes.

| Method | Characteristic failure | Diagnostic | First fix |
| --- | --- | --- | --- |
| DQN | Q-values explode | max Q vs ceiling | Target network |
| DQN | Value overestimation | Q bias vs Monte-Carlo return | Double DQN |
| SAC | $\alpha \to 0$ (collapse) | log $\alpha$ trajectory | Fix reward scale, lower target entropy |
| SAC | $\alpha \to \infty$ (no exploit) | log $\alpha$ trajectory | Raise reward scale or target entropy |
| Deep Q (general) | Loss won't settle, policy wanders | value loss + policy churn | Slower target updates, smaller LR |

## 11. Debugging RLHF systems

Reinforcement Learning from Human Feedback deserves its own section because it stacks a *learned* reward model on top of an RL loop, which means every failure mode above can be triggered by a *second* model that itself can be wrong. RLHF is where I have seen the most creative reward hacking. If you want the full RLHF pipeline, see [rlhf-aligning-language-models-with-human-preferences](/blog/machine-learning/training-techniques/rlhf-aligning-language-models-with-human-preferences) and the DPO alternative in [direct-preference-optimization-rl-free-alignment](/blog/machine-learning/training-techniques/direct-preference-optimization-rl-free-alignment).

**Reward model inconsistency.** A reward model trained on pairwise preferences can be miscalibrated. The classic bug is **position bias**: the same response scored differently depending on whether it was presented as "response A" or "response B" during preference annotation, baking the annotation-order artifact into the reward model. The diagnostic is to score the same prompt-response pair multiple times in different contexts and check for variance that should not exist. The fix is de-biased preference collection (randomize order) and reward-model ensembling.

**Policy-reward mismatch (length hacking).** The most famous RLHF failure: the policy discovers that the reward model gives higher scores to *longer* responses (because annotators mildly preferred thorough answers), so it learns to pad every response to maximum length regardless of quality. This is Goodhart's law (mode 1) wearing a language-model costume. The diagnostic is to plot mean response length against the reward-model score over training; if length rises while genuine quality (measured by a held-out human eval or a different judge) does not, you are length-hacking. Fixes include length-penalizing the reward, length-normalizing the reward model, or using a reward model explicitly trained to be length-invariant.

**KL penalty too small.** RLHF adds a KL penalty against a reference (the pre-RLHF SFT) policy: the objective is $\mathbb{E}[r_\phi(x, y)] - \beta\, \text{KL}(\pi_\theta \,\|\, \pi_{\text{ref}})$. This term is the whole ballgame for preventing reward hacking — it keeps the policy near the distribution the reward model was trained on, where the reward model's scores are trustworthy. If $\beta$ is too small, the policy drifts far from the reference, exploits the reward model's blind spots in out-of-distribution regions, and **mode-collapses** onto whatever gibberish-prefix or repetitive pattern the reward model happens to over-score. The diagnostic is to log the KL between policy and reference; if it climbs past a few nats while reward keeps rising, the policy is exploiting, not improving. Raise $\beta$. This is the same collapse mechanism as mode 2, with the reference KL playing the stabilizing role the entropy bonus plays in vanilla PG.

**Reward model extrapolation.** A reward model trained on a preference distribution gives *unreliable* scores outside that distribution — and a strong RL optimizer will deliberately push the policy toward the highest-scoring regions, which are often exactly the out-of-distribution regions where the reward model is wrong and over-optimistic. This is reward over-optimization: reward-model score keeps rising while true quality peaks and then *declines*. The canonical diagnostic is the gold-vs-proxy reward curve (Gao et al., 2022): plot the proxy reward-model score and a held-out "gold" evaluation against KL from the reference; the gap between them *is* the over-optimization. The fix is the KL penalty (keep the policy in-distribution), reward-model ensembling, and early stopping when gold reward peaks.

```python
import torch
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead
from trl.core import respond_to_batch

# The KL coefficient is the single most important RLHF stability knob.
config = PPOConfig(
    learning_rate=1.4e-5,
    init_kl_coef=0.2,        # too small -> policy hacks the reward model
    adap_kl_ctrl=True,       # adaptively raise/lower beta toward a target KL
    target=6.0,              # target KL (nats) from the reference policy
    batch_size=128,
)

def rlhf_diagnostics(rewards, kl_to_ref, response_lengths):
    """Log the three RLHF tripwires every batch."""
    print(f"reward mean={rewards.mean():.3f}  "
          f"KL_to_ref={kl_to_ref.mean():.3f}  "
          f"resp_len mean={response_lengths.float().mean():.1f}")
    # Length climbing with reward but quality flat -> length hacking.
    # KL climbing past target with reward -> reward-model exploitation.
    if kl_to_ref.mean() > 12.0:
        print("WARNING: policy drifting far from reference -> raise init_kl_coef")
```

#### Worked example: catching length-hacking in a summarization RLHF run

On a TL;DR summarization fine-tune, our reward-model score climbed steadily and we almost declared victory. The length plot stopped us: mean summary length had grown from 31 tokens to 94 tokens over training, and the KL to the reference had climbed to 14 nats. A held-out human eval told the real story — actual summary quality (faithfulness, conciseness) had *peaked* around the midpoint and then declined as the model padded summaries with redundant restatement that the reward model mistook for thoroughness. Two changes fixed it: we raised `init_kl_coef` from 0.2 to 0.5 (pulling the policy back toward the trustworthy region) and added a mild length normalization to the reward. Win-rate against the SFT baseline on a fresh human eval went from an over-optimized 51% (the padded model was actually *worse* than where it had been mid-training) back up to 68%. The number that lied was the proxy reward; the numbers that told the truth were KL and length.

## 12. Case studies

**Atari DQN (Mnih et al., 2015).** The paper that started deep RL was, at its heart, a debugging paper. Vanilla Q-learning with a neural network diverges (the deadly triad). The two fixes that made it work — experience replay (decorrelating the data to reduce the off-policy mismatch) and the target network (freezing the bootstrap target) — are precisely the mode-3 fixes above. DQN reached human-level play on 29 of 49 Atari games, but *only* with both stabilizers. Remove either and Q-values diverge. The result stands as the canonical demonstration that deep value learning is a debugging problem before it is a capability problem.

**OpenAI Five (Dota 2, 2018–2019).** A study in distribution shift and non-stationarity at scale. The agents trained against *past versions of themselves* (self-play), a deliberately non-stationary curriculum. The team had to manage exactly the mode-6 hazard — stale opponents in the training mix — and used a large but carefully managed pool of past policies. They also discovered that surprisingly large batch sizes and aggressive reward shaping (with hand-designed potentials for last-hits, gold, and experience) were necessary. The system consumed the equivalent of decades of game experience per day, which only paid off because the stability machinery (PPO clipping, careful KL control) kept those updates from diverging.

**InstructGPT (Ouyang et al., 2022).** The production RLHF system behind the ChatGPT lineage. Its public reports are candid about reward over-optimization (mode 11): without the KL penalty to the SFT reference, the policy exploits the reward model. The KL coefficient and the reward-model quality were the load-bearing hyperparameters. InstructGPT showed that a 1.3B-parameter RLHF model could be preferred by human raters over a 175B-parameter base model — a result entirely contingent on *not* reward-hacking, which is to say, on getting the mode-11 debugging right.

**Sim-to-real robotics (OpenAI Dactyl, 2018).** In-hand cube manipulation trained purely in simulation and transferred to a physical Shadow Hand. The whole project was a distribution-shift (mode 6) defense: domain randomization over physics parameters (friction, mass, latency, sensor noise) so the real world was within the training distribution. Without randomization the sim policy failed instantly on hardware; with it, the policy achieved tens of consecutive successful rotations on the physical hand. The lesson generalizes far beyond robotics: if deployment differs from training, train across the distribution of possible deployments.

## 13. When to use this debugging process (and when a simpler check wins)

The full six-mode protocol is for when you have a deep RL agent that is genuinely not learning and you do not know why. It is overkill in several common situations, and knowing when to short-circuit it saves hours:

- **If you have not run the smoke tests, run them first.** Do not debug your 100-dimensional environment before confirming the agent solves CartPole. The bandit and CartPole tests catch most implementation bugs in minutes and tell you whether the problem is the *learner* or the *task*.
- **If the model is known-good (e.g. stock SB3 PPO) and only the environment is custom, suspect the environment (mode 5) first.** A library implementation that has solved a hundred benchmarks did not suddenly break on yours. Trace five steps by hand and run `check_env` before touching the algorithm.
- **If you can simulate cheaply and the model is known, you may not need model-free RL at all.** For a small, known MDP, value iteration or policy iteration converges with guarantees and nothing to debug. RL earns its complexity only when you *cannot* enumerate the dynamics. Reach for the planning methods in [model-based-rl-learning-world-models](/blog/machine-learning/reinforcement-learning/model-based-rl-learning-world-models) before a model-free agent when a model is available.
- **For tabular problems, the deadly triad does not apply** — drop the target network and Double DQN machinery; plain tabular Q-learning converges. Adding deep-RL stabilizers to a tabular problem just adds knobs to misconfigure.
- **For small discrete action spaces, value-based methods (DQN) are often easier to debug than policy gradients,** because Q-values are directly inspectable against a theoretical ceiling. The mode-3 diagnostic (max Q vs ceiling) is a clearer signal than reading entropy curves.

The protocol is most valuable in the messy middle: a continuous-control or partially-observed task, a custom environment, a deep network, where any of the six modes could be the culprit and the symptoms overlap. That is where having a *checklist* — symptom to diagnostic to fix, as in Figure 5 — converts a multi-day fishing expedition into a directed search.

![A matrix mapping six reinforcement learning failure modes to their symptom, diagnostic, primary fix, and secondary fix](/imgs/blogs/debugging-rl-agents-common-failure-modes-5.png)

## 14. The historical arc of RL stability

It is worth seeing the six failure modes as the negative image of RL's own progress. Almost every landmark advance in deep RL was, fundamentally, the fix for one specific failure mode that had previously made a class of problems intractable. Figure 8 lays out that arc.

![A timeline of deep reinforcement learning stability advances from target networks in 2015 to process reward models in 2023](/imgs/blogs/debugging-rl-agents-common-failure-modes-8.png)

The 2015 Atari DQN added the target network and replay buffer to fix value divergence (mode 3). The 2016 Dueling and Double DQN papers fixed the overestimation bias (mode 3, deeper). The 2017 PPO paper fixed the destructive-large-update problem (modes 2 and 9) with clipping and a KL check. The 2018 SAC paper made entropy a first-class objective with auto-tuned temperature to fight collapse (mode 2). The 2020-era RLHF systems added reward-model calibration and the reference KL penalty to fight reward hacking and over-optimization (mode 11). And the 2023 wave of process reward models (rewarding correct *reasoning steps*, not just final answers) is the latest answer to sparse, gameable reward (mode 1) in the language-model setting. The whole field's progress is a sequence of debugging victories, which is exactly why understanding the failure modes teaches you the algorithms from the inside.

## 15. Key takeaways

- **Watch rollouts, not just curves.** A rising return curve proves the agent is optimizing *something*, never that it is optimizing what you meant. Render the policy and read $(s, a, r, s')$ tuples by hand.
- **Reduce before you debug.** If the agent can't solve a 2-armed bandit and CartPole-v1, the bug is in your learner, not your hard environment. The smoke tests are minutes; the alternative is days.
- **Each failure mode has one signature diagnostic.** Reward variance for sparsity, policy entropy for collapse, max Q-value for divergence, action histogram for starvation, a five-step trace for environment bugs, importance weights and buffer age for distribution shift. Log all of them from day one.
- **Suspect the environment first.** Off-by-one rewards, observation leaks, and `terminated`-vs-`truncated` confusion are more common than algorithm bugs and far harder to spot. Run `check_env` and trace by hand.
- **Reward scale is the root of a startling number of failures** — it shows up in mode 1 (gradient explosion), mode 3 (value blowup), and mode 10 (SAC temperature collapse). Normalize rewards as a default.
- **Potential-based shaping is the only shaping with a guarantee** — $\gamma\Phi(s') - \Phi(s)$ provably preserves the optimal policy. Any other dense bonus is a reward-hacking invitation.
- **Entropy and KL are the stabilizers.** An entropy bonus prevents premature collapse; a KL constraint (PPO clip, RLHF reference KL) prevents destructive updates and reward-model exploitation. They are the same idea — keep the policy from moving too far, too fast.
- **In RLHF, the KL-to-reference penalty is load-bearing.** Too small and the policy hacks the reward model. Watch KL and response length, not just reward-model score — the proxy reward is the number most likely to lie to you.
- **Change one thing at a time.** RL's interacting hyperparameters mean a multi-variable change teaches you nothing from its result. Isolate.

## 16. Further reading

- Sutton & Barto, *Reinforcement Learning: An Introduction* (2nd ed., 2018) — the deadly triad (Chapter 11), exploration, and the theoretical foundations behind every diagnostic here. The single most important book to own.
- Mnih et al., "Human-level control through deep reinforcement learning" (Nature, 2015) — the Atari DQN paper; target network and replay buffer as the mode-3 fix.
- van Hasselt, Guez & Silver, "Deep Reinforcement Learning with Double Q-learning" (2016) — the overestimation-bias diagnosis and fix.
- Schulman et al., "Proximal Policy Optimization Algorithms" (2017) — the clipped surrogate and KL check; the foundation of modes 9 and 2.
- Haarnoja et al., "Soft Actor-Critic" (2018) and the temperature auto-tuning follow-up — entropy as a first-class objective and the mode-10 failure of $\alpha$ tuning.
- Ng, Harada & Russell, "Policy Invariance Under Reward Transformations" (1999) — the proof that potential-based shaping preserves the optimal policy (mode 1).
- Gao, Schulman & Hilton, "Scaling Laws for Reward Model Overoptimization" (2022) — the gold-vs-proxy curve and the quantitative story of mode-11 reward hacking.
- Ouyang et al., "Training language models to follow instructions with human feedback" (InstructGPT, 2022) — production RLHF, the reference KL penalty, and reward over-optimization in practice.
- Within this series: the bird's-eye view in [reinforcement-learning-a-unified-map](/blog/machine-learning/reinforcement-learning/reinforcement-learning-a-unified-map), the PPO mechanics in [policy-gradient-methods-from-reinforce-to-ppo](/blog/machine-learning/reinforcement-learning/policy-gradient-methods-from-reinforce-to-ppo), and the complete decision framework in [the-reinforcement-learning-playbook](/blog/machine-learning/reinforcement-learning/the-reinforcement-learning-playbook).
