---
title: "The Reinforcement Learning Playbook: The Complete System in One Framework"
date: "2026-06-22"
publishDate: "2026-06-22"
description: "The unified capstone of the series: how MDPs, policy gradients, model-based RL, multi-agent learning, RLHF, sim-to-real, and safe RL collapse into one seven-decision framework for building production reinforcement learning systems."
tags:
  [
    "reinforcement-learning",
    "deep-learning",
    "rlhf",
    "policy-gradient",
    "model-based-rl",
    "multi-agent",
    "machine-learning",
    "pytorch",
    "llm-alignment",
    "production",
  ]
category: "machine-learning"
subcategory: "Reinforcement Learning"
author: "Hiep Tran"
featured: true
readTime: 51
image: "/imgs/blogs/the-reinforcement-learning-playbook-1.png"
---

Here is a scene I have watched play out four times across three companies. A team reads the DeepSeek-R1 paper, gets excited, spins up eight H100s, points GRPO at their model, and trains for two days. The reward curve climbs beautifully. Everyone celebrates. Then they look at the actual outputs and the model has learned to emit the token sequence that maxes the reward model while saying nothing useful. The reward went up. The product got worse. Six figures of compute, and the lesson they paid for was one sentence long: *the reward you wrote down is not the behavior you wanted.*

That failure was not an algorithm failure. PPO was fine. GRPO was fine. The model trained. The failure happened three decisions *upstream* of the algorithm — in how they formulated the problem and designed the reward — and no amount of hyperparameter tuning could have saved it. This is the central thesis of the whole series, and the reason this capstone exists: **reinforcement learning is not an algorithm you call, it is a chain of seven decisions, and the project fails at the weakest link.** The algorithm is decision number two. People obsess over it because it has the coolest math. The decisions that actually sink projects are formulation, reward, and evaluation — and almost nobody writes those down.

This post is the map of the whole territory. Over roughly seventy posts this series built up every piece — [what reinforcement learning even is](/blog/machine-learning/reinforcement-learning/what-is-reinforcement-learning), the [Markov decision process](/blog/machine-learning/reinforcement-learning/markov-decision-processes) that formalizes it, the [policy gradient theorem](/blog/machine-learning/reinforcement-learning/the-policy-gradient-theorem) that powers half the algorithms, [PPO](/blog/machine-learning/reinforcement-learning/proximal-policy-optimization-ppo), [SAC](/blog/machine-learning/reinforcement-learning/soft-actor-critic-sac), [DQN](/blog/machine-learning/reinforcement-learning/deep-q-networks-dqn), [model-based methods](/blog/machine-learning/reinforcement-learning/model-based-rl-learning-world-models), [multi-agent learning](/blog/machine-learning/reinforcement-learning/multi-agent-rl-fundamentals), and the full [RLHF](/blog/machine-learning/reinforcement-learning/why-language-models-need-rlhf) stack. This capstone does not re-derive any of it. Instead it answers the question every reader eventually asks: *given a real problem, how do I decide which of these to use, and how do they fit together into something I can ship and operate?* Figure 1 is the spine. We will walk it decision by decision, then assemble two complete systems end to end — a conversational AI assistant and a production robotics controller — so you can see every piece connect in a real build.

![A directed graph showing the seven sequential RL decisions from problem formulation through algorithm selection, reward design, exploration, training infrastructure, evaluation, and deployment with a feedback edge for the Goodhart check](/imgs/blogs/the-reinforcement-learning-playbook-1.png)

By the end you should be able to take a fuzzy business problem, decide whether RL is even the right tool, formulate it correctly, pick an algorithm with a one-sentence justification, design a reward that resists hacking, choose an exploration strategy, size the training infrastructure, build an evaluation harness that catches the failures that matter, and deploy with a rollback plan. That is the job. The algorithms are the easy part.

## The seven-decision framework

Strip away the math and every RL project answers seven questions, in order, each one constraining the next. I will state them once here as the table of contents for everything that follows.

1. **Problem formulation.** Is this actually a Markov decision process? Can the agent observe enough to act well? Is the environment stationary? Get this wrong and no algorithm helps.
2. **Algorithm selection.** Discrete or continuous actions? Are samples cheap or expensive? Is the target a language model? Three questions pick the family.
3. **Reward design.** What number are you actually maximizing, and is maximizing it the same as solving your problem? This is where most projects quietly die.
4. **Exploration strategy.** How does the agent discover good behavior it has never tried? Sparse rewards need different machinery than dense ones.
5. **Training infrastructure.** One machine or a cluster? Synchronous or asynchronous? How do you collect rollouts fast enough that the GPUs are not idle?
6. **Evaluation.** Episodic return is necessary and badly insufficient. What do you measure to know the policy is actually good and actually safe?
7. **Deployment.** How do you get the policy in front of real traffic without betting the business, and how do you pull it back when it misbehaves?

Notice the shape. The first three are *modeling* decisions — they decide what problem you are solving. The middle two are *engineering* decisions — they decide whether you can solve it at all in finite time and money. The last two are *operations* decisions — they decide whether you can trust and maintain the thing in the wild. A staff engineer's value is mostly in decisions one, three, six, and seven. A new hire's instinct is to spend all their time on decision two. The rest of this post is organized to fix that instinct.

A quick honesty note on scope. RL is the right tool far less often than people building it would like. If you can frame your problem as supervised learning, do that — it is more stable, more sample-efficient, and easier to debug. RL earns its complexity when the thing you care about is a *sequence* of decisions whose payoff is delayed, where you cannot label the correct action at each step but you can score the outcome. Trading, robotics control, dialogue, recommendation with long-horizon engagement, game-playing — those are RL-shaped. "Classify this image" is not. Keep that filter on as we go.

## Decision 1: problem formulation — is this even an MDP?

The [Markov decision process](/blog/machine-learning/reinforcement-learning/markov-decision-processes) is the contract every RL algorithm signs. It assumes a tuple $(\mathcal{S}, \mathcal{A}, P, R, \gamma)$: a state space, an action space, a transition kernel $P(s' \mid s, a)$, a reward function $R(s, a)$, and a discount factor $\gamma \in [0, 1)$. The single load-bearing assumption is the **Markov property**: the future depends on the past only through the present state. Formally,

$$
P(s_{t+1} \mid s_t, a_t, s_{t-1}, a_{t-1}, \dots, s_0) = P(s_{t+1} \mid s_t, a_t).
$$

This is not a technicality. It is the assumption that makes the entire theory work, because it is what lets value functions exist. The [value function and Bellman equation](/blog/machine-learning/reinforcement-learning/value-functions-and-the-bellman-equation) define $V^\pi(s)$ as the expected discounted return from $s$, and the recursion

$$
V^\pi(s) = \mathbb{E}_{a \sim \pi}\big[ R(s, a) + \gamma \, \mathbb{E}_{s'}\, V^\pi(s') \big]
$$

is only valid if the value of a state is a well-defined function of that state alone. If the true dynamics depend on history the state does not capture, then $V^\pi(s)$ is not a function — the same $s$ has different futures depending on hidden context — and every value-based method is estimating something that does not exist. This is the deepest and most common formulation bug, and it shows up as a policy that trains to a plateau and then thrashes, never converging, because the target it chases is genuinely ambiguous.

So the first job is a checklist, not a code commit.

**Is the reward Markov in your state?** Ask: does the immediate desirability of an action depend on anything not in the state vector? In a trading agent, if your state is the current price but your reward depends on your inventory and unrealized P&L (which it does), then price alone is not a Markov state — you must include position, cash, and time-to-close. The classic [algorithmic trading formulation](/blog/machine-learning/reinforcement-learning/rl-for-algorithmic-trading-foundations) post hammers this: the most common rookie bug is an under-specified state that omits inventory, which makes the optimal action genuinely undefined.

**Can the agent observe the state, or only part of it?** If the agent sees an *observation* $o_t$ that is a lossy function of the true state — a robot with only proprioceptive sensors, a dialogue agent that cannot see the user's intent — you have a **partially observable MDP (POMDP)**, not an MDP. The fix is to reconstruct a sufficient statistic of history. In practice that means one of: stacking the last $k$ frames (Atari DQN stacks four), feeding observations through a recurrent network so the hidden state summarizes history, or using a learned belief state. The [model-based RL](/blog/machine-learning/reinforcement-learning/model-based-rl-learning-world-models) and [world-models](/blog/machine-learning/reinforcement-learning/world-models-dreamer-planet) posts go deep here: Dreamer literally learns a recurrent latent that *is* the belief state.

**Is the environment stationary?** RL assumes $P$ and $R$ do not change while you train. Markets are non-stationary by nature; a recommendation environment changes as your own policy changes user behavior; a multi-agent environment is non-stationary from any single agent's view because the *other* agents are learning too. Non-stationarity is not always fatal — you can re-train, use shorter horizons, or move to the [multi-agent](/blog/machine-learning/reinforcement-learning/multi-agent-rl-fundamentals) framing where non-stationarity is modeled explicitly via [Nash equilibria](/blog/machine-learning/reinforcement-learning/nash-equilibria-and-game-theory-for-marl) — but it must be named, because it changes everything downstream.

#### Worked example: catching a formulation bug before it costs a week

A team builds an order-execution agent. State: current mid-price and the last trade direction. Action: buy 100 shares, sell 100 shares, or hold. Reward: realized P&L on each fill. They train PPO for 2M steps. The return climbs to a plateau at roughly +0.3 reward per episode, then oscillates between +0.3 and -0.1 forever, never converging. Three seeds, same story.

The formulation checklist finds it in five minutes. The reward (realized P&L) depends on inventory — selling is only profitable if you are *long* — but inventory is not in the state. The same observed `(price, last_direction)` therefore maps to different optimal actions depending on hidden position, so $V^\pi$ is not a function and the plateau-thrash is the policy oscillating between two incompatible value estimates. Adding `signed_inventory` and `time_remaining` to the state restores the Markov property; the same PPO run now converges cleanly to +0.8. Nothing about the algorithm changed. The [credit assignment problem](/blog/machine-learning/reinforcement-learning/the-credit-assignment-problem) post explains why this class of bug is so insidious: a delayed, inventory-dependent reward looks like a hard credit-assignment problem when it is actually a missing-state problem, and people reach for fancier algorithms when they need a fix one layer up.

The formulation decision tree, then, is short: if the reward is non-Markov in your candidate state, fix the state; if the agent cannot observe the state, augment with history (frame-stack or recurrence) and treat it as a POMDP — see the recurrence tricks in the [meta-learning and few-shot RL](/blog/machine-learning/reinforcement-learning/meta-learning-and-few-shot-rl) post; if other learning agents are part of the environment, move to the [multi-agent](/blog/machine-learning/reinforcement-learning/multi-agent-rl-fundamentals) framing. Only once the problem is genuinely an MDP (or a POMDP you have made tractable) do you get to pick an algorithm.

## Decision 2: algorithm selection — the unified taxonomy

This is the decision everyone wants to start with, and it is genuinely fun, but it should be nearly mechanical once formulation is settled. The whole series collapses into a small taxonomy organized by two axes: *what you estimate* (a value function, a policy, or a model of the world) and *how you collect data* (on-policy or off-policy). Figure 2 turns that into a three-question decision tree.

![A decision tree branching on action space then on sample cost and language-model target, terminating in algorithm recommendations from DQN to GRPO to PPO and SAC](/imgs/blogs/the-reinforcement-learning-playbook-2.png)

**Question 1: Are the actions discrete or continuous?**

For *discrete* action spaces — Atari, board games, discrete trading actions, dialogue-act selection — value-based methods are the default. Learn $Q(s, a)$, act greedily. [DQN](/blog/machine-learning/reinforcement-learning/deep-q-networks-dqn) is the foundation; [Rainbow](/blog/machine-learning/reinforcement-learning/rainbow-dqn-combining-six-improvements) bundles the [six improvements](/blog/machine-learning/reinforcement-learning/dqn-improvements-double-dueling-per) (double Q-learning, dueling heads, prioritized replay, multi-step returns, [distributional RL](/blog/machine-learning/reinforcement-learning/distributional-rl-c51-qr-dqn-iqn), noisy nets) that together roughly double median Atari performance. Value-based methods are sample-efficient because they reuse data via [experience replay](/blog/machine-learning/reinforcement-learning/experience-replay-and-offline-data) and learn [off-policy](/blog/machine-learning/reinforcement-learning/on-policy-vs-off-policy-a-practical-guide). Their weakness is they do not extend cleanly to continuous actions — you cannot enumerate $\arg\max_a Q(s,a)$ over a continuum.

For *continuous* actions — robot torques, portfolio weights, steering — you need a policy that outputs actions directly, which sends you to policy-gradient and actor-critic methods.

**Question 2 (continuous branch): Are samples expensive?**

If each environment step costs real money or wall-clock — a physical robot, a slow simulator — sample efficiency dominates and you want **model-based RL**. Learn a model of the dynamics, then plan or generate synthetic rollouts inside it. [Dyna-Q](/blog/machine-learning/reinforcement-learning/dyna-q-and-planning-with-a-model) is the simplest version; [MuZero](/blog/machine-learning/reinforcement-learning/muzero-mastering-games-without-rules) and [Dreamer/PlaNet](/blog/machine-learning/reinforcement-learning/world-models-dreamer-planet) are the modern frontier, often 5–10× more sample-efficient than model-free on the same task. The [model-based vs model-free](/blog/machine-learning/reinforcement-learning/model-based-vs-model-free-when-to-use-which) post is the decision guide: model-based wins when samples are scarce and the dynamics are learnable; model-free wins when simulation is cheap and dynamics are chaotic.

If samples are cheap (a fast simulator, Isaac Gym running 4096 parallel envs), reach for **model-free actor-critic**. [PPO](/blog/machine-learning/reinforcement-learning/proximal-policy-optimization-ppo) is the robust default — it is forgiving, on-policy, and the single most-deployed deep RL algorithm. [SAC](/blog/machine-learning/reinforcement-learning/soft-actor-critic-sac) is the sample-efficient off-policy choice for continuous control, often beating PPO on MuJoCo wall-clock-to-target because it reuses replay data and its entropy term explores automatically. [DDPG and TD3](/blog/machine-learning/reinforcement-learning/deterministic-policy-gradient-ddpg-td3) are the deterministic-policy alternatives; TD3's twin critics fix DDPG's overestimation. The lineage from [REINFORCE](/blog/machine-learning/reinforcement-learning/the-policy-gradient-theorem) through [A2C/A3C](/blog/machine-learning/reinforcement-learning/actor-critic-a2c-a3c) to [TRPO](/blog/machine-learning/reinforcement-learning/trust-region-policy-optimization-trpo) and PPO is one long story of variance reduction, told across those posts.

**Question 3 (continuous branch): Is the target a language model?**

Aligning an LLM is technically continuous-action policy optimization over the token vocabulary, but it is special enough to be its own branch. Here you want **RLHF** methods: [PPO with a reward model](/blog/machine-learning/reinforcement-learning/ppo-for-llm-fine-tuning), or the critic-free [GRPO](/blog/machine-learning/reinforcement-learning/grpo-group-relative-policy-optimization) that DeepSeek popularized, or the RL-free shortcut [DPO](/blog/machine-learning/reinforcement-learning/dpo-direct-preference-optimization). The [why language models need RLHF](/blog/machine-learning/reinforcement-learning/why-language-models-need-rlhf) post is the entry point.

Figure 3 lays the workhorses side by side so the trade-offs are visible at a glance.

![A comparison matrix scoring DQN, PPO, SAC, MuZero, GRPO, and PPO-Lagrangian across action space, sample efficiency, wall-clock cost, and best fit](/imgs/blogs/the-reinforcement-learning-playbook-3.png)

| Algorithm | Action space | Sample efficiency | Wall-clock | On/off-policy | When to use |
|---|---|---|---|---|---|
| DQN / Rainbow | Discrete | Medium | Low | Off-policy | Atari, discrete games, discrete trading |
| REINFORCE | Both | Very low | Low | On-policy | Teaching, tiny problems, never production |
| A2C / A3C | Both | Low | Low | On-policy | Cheap baseline, many parallel envs |
| PPO | Both | Low–medium | Medium | On-policy | The robust default for almost anything |
| SAC | Continuous | High | Medium | Off-policy | Continuous control, sample reuse matters |
| TD3 | Continuous | High | Medium | Off-policy | Deterministic control, clean dynamics |
| MuZero | Discrete | Very high | High | Model-based | Planning, expensive samples, known reward |
| DreamerV3 | Both | Very high | High | Model-based | Pixels, sample-scarce, world model learnable |
| GRPO | Tokens | Medium | High | On-policy | LLM alignment without a value network |
| PPO-Lagrangian | Both | Low | Medium | On-policy | Safety constraints must hold |

The one-sentence rule of thumb I give every team: **start with PPO, switch to SAC if you need sample efficiency on continuous control, switch to DQN/Rainbow if your actions are discrete, switch to a model-based method only if samples are genuinely expensive, and use GRPO or DPO if you are aligning a language model.** Ninety percent of projects are served by that sentence. The exotic algorithms exist for the other ten percent, and you will know when you are in it.

#### Worked example: PPO vs SAC on HalfCheetah, measured

A common question is whether the extra complexity of SAC pays off over PPO. On the MuJoCo `HalfCheetah-v4` benchmark with Stable-Baselines3 defaults, PPO reaches roughly a return of 1,700 after about 1M environment steps; SAC reaches roughly 9,000–11,000 in the same 1M steps. SAC is dramatically more sample-efficient here because it learns off-policy from a replay buffer (reusing every transition many times) and its maximum-entropy objective explores without a hand-tuned schedule. The catch: SAC's wall-clock *per step* is higher (two critics, a replay buffer, more gradient steps), and it is more sensitive to its `learning_rate` and `tau` than PPO. The honest reading is that on a cheap simulator where you can afford 10M steps, PPO's robustness often wins on engineer-hours-to-working; on a slow simulator where 1M steps is your budget, SAC's sample efficiency wins decisively. State your sample budget first; it picks the algorithm.

Here is the entire selection, as code you would actually run, using Stable-Baselines3 so the API differences are explicit.

```python
import gymnasium as gym
from stable_baselines3 import PPO, SAC, DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env

def select_and_train(task: str, n_envs: int = 8, steps: int = 1_000_000):
    if task == "discrete":            # CartPole, Atari, discrete trading
        env = make_vec_env("CartPole-v1", n_envs=n_envs)
        model = DQN("MlpPolicy", env, learning_rate=1e-3,
                    buffer_size=100_000, learning_starts=1_000,
                    target_update_interval=500, verbose=0)
    elif task == "continuous_cheap":  # fast simulator, robust default
        env = make_vec_env("HalfCheetah-v4", n_envs=n_envs)
        model = PPO("MlpPolicy", env, n_steps=2048, batch_size=64,
                    gae_lambda=0.95, gamma=0.99, ent_coef=0.0, verbose=0)
    elif task == "continuous_efficient":  # slow simulator, reuse samples
        env = make_vec_env("HalfCheetah-v4", n_envs=1)  # SAC is off-policy
        model = SAC("MlpPolicy", env, learning_rate=3e-4,
                    buffer_size=1_000_000, batch_size=256,
                    tau=0.005, gamma=0.99, ent_coef="auto", verbose=0)
    else:
        raise ValueError(f"unknown task {task!r}")

    model.learn(total_timesteps=steps)
    mean_r, std_r = evaluate_policy(model, model.get_env(), n_eval_episodes=20)
    print(f"{task}: mean return {mean_r:.1f} +/- {std_r:.1f}")
    return model
```

That function *is* the algorithm-selection decision, made executable. Notice `ent_coef="auto"` for SAC (entropy temperature tuned automatically — that is the exploration decision, baked in) versus `ent_coef=0.0` for PPO on a deterministic continuous task. Notice DQN gets a replay `buffer_size` because it is off-policy and SAC does too, while PPO does not. The API shape encodes the taxonomy.

### The theory under the taxonomy: why the families differ

The taxonomy is not arbitrary; the two axes — *what you estimate* and *how you collect data* — come straight from the math, and understanding the math is what lets you debug the algorithms when they misbehave. Start with the policy-gradient side, because it is the half that powers PPO, SAC, and every RLHF method. The objective is to maximize expected return $J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)]$ over trajectories $\tau$. The [policy gradient theorem](/blog/machine-learning/reinforcement-learning/the-policy-gradient-theorem) gives its gradient without differentiating through the environment dynamics at all. The derivation is the log-derivative trick: since $\nabla_\theta \pi_\theta(\tau) = \pi_\theta(\tau) \nabla_\theta \log \pi_\theta(\tau)$,

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\big[ R(\tau) \, \nabla_\theta \log \pi_\theta(\tau) \big] = \mathbb{E}_{\tau \sim \pi_\theta}\Big[ \sum_t \nabla_\theta \log \pi_\theta(a_t \mid s_t) \, A^{\pi}(s_t, a_t) \Big].
$$

Two facts fall out of this single equation, and they explain half the practical pain in deep RL. First, the gradient is an *expectation under the current policy* $\pi_\theta$ — which is precisely why policy gradient is fundamentally **on-policy**: the moment the policy changes, your old samples are drawn from the wrong distribution and the estimate is biased. That is the deep reason PPO discards its rollout buffer after a few epochs while DQN keeps a million transitions around. Second, the gradient is a *Monte Carlo estimate* of an expectation, so it is **high-variance** — and every advance in the policy-gradient lineage, from the baseline subtraction in [actor-critic](/blog/machine-learning/reinforcement-learning/actor-critic-a2c-a3c) to the advantage estimate $A^\pi = Q^\pi - V^\pi$ to generalized advantage estimation, is a variance-reduction trick. When your PPO run is noisy and unstable, this equation tells you where to look: the variance of the advantage estimate.

The value-based side comes from a different operator. [Q-learning](/blog/machine-learning/reinforcement-learning/q-learning-off-policy-td-control) iterates the Bellman *optimality* operator

$$
(\mathcal{T}^* Q)(s, a) = R(s, a) + \gamma \, \mathbb{E}_{s'}\big[ \max_{a'} Q(s', a') \big],
$$

and the reason it converges *off-policy* — learning the optimal policy from data generated by any exploratory behavior — is that $\mathcal{T}^*$ is a $\gamma$-contraction in the max-norm: $\lVert \mathcal{T}^* Q_1 - \mathcal{T}^* Q_2 \rVert_\infty \le \gamma \lVert Q_1 - Q_2 \rVert_\infty$. The Banach fixed-point theorem then guarantees a unique fixed point ($Q^*$) that repeated application reaches geometrically. The `max` inside the operator is what makes it off-policy (it asks "what is the best next action," not "what did my behavior policy do"), and it is *also* what makes DQN unstable when combined with function approximation and bootstrapping — the [deadly triad](/blog/machine-learning/reinforcement-learning/the-deadly-triad-stability-in-deep-rl). The whole [on-policy vs off-policy](/blog/machine-learning/reinforcement-learning/on-policy-vs-off-policy-a-practical-guide) trade-off — sample efficiency (off-policy reuses data) versus stability (on-policy stays in-distribution) — is these two operators in tension.

PPO, the algorithm you will reach for most, is a careful patch on the on-policy fragility. It lets you reuse each batch of rollouts for several gradient epochs without the policy drifting too far, by *clipping* the probability ratio $\rho_t(\theta) = \pi_\theta(a_t \mid s_t) / \pi_{\theta_{\text{old}}}(a_t \mid s_t)$ in its surrogate objective:

$$
L^{\text{CLIP}}(\theta) = \mathbb{E}_t\big[ \min\big( \rho_t \, A_t, \; \text{clip}(\rho_t, 1 - \epsilon, 1 + \epsilon) \, A_t \big) \big].
$$

The clip is the trust region made cheap: it removes the incentive to push $\rho_t$ far from 1, so a single rollout batch can be reused safely for ~10 epochs, which is the entire reason PPO is sample-efficient *enough* to be practical despite being on-policy. In code the surrogate is six lines, and it is worth seeing because it is the load-bearing math of the most-deployed RL algorithm in the world.

```python
import torch

def ppo_clip_loss(logp, logp_old, advantages, eps=0.2):
    ratio = torch.exp(logp - logp_old)              # rho_t, the probability ratio
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1 - eps, 1 + eps) * advantages
    return -torch.min(unclipped, clipped).mean()    # negative: we maximize the surrogate
```

That `torch.min` of the unclipped and clipped terms is the whole trick: when the advantage is positive it caps how much you can increase the action's probability; when negative it caps how much you can decrease it. Everything else in PPO — GAE, value-function fitting, entropy bonus — is scaffolding around these six lines. Knowing this is what lets you answer, in an interview or a postmortem, *why* PPO is stable: it is a first-order approximation to the [TRPO](/blog/machine-learning/reinforcement-learning/trust-region-policy-optimization-trpo) trust region that needs no second-order solve.

## Decision 3: reward design — the decision that actually decides

If formulation is the most-skipped decision, reward design is the most-underestimated one. I will state the rule bluntly: **your reward function gets the algorithm to do exactly what it says, which is rarely what you meant.** This is Goodhart's law — "when a measure becomes a target, it ceases to be a good measure" — and RL is the most efficient Goodhart-exploitation machine ever built, because optimizing a number is the *only* thing it does. The [reward hacking and Goodhart's law](/blog/machine-learning/reinforcement-learning/reward-hacking-and-goodharts-law) post is full of real examples: the boat-racing agent that learned to spin in circles collecting power-ups instead of finishing the race, because the reward was points and points came from power-ups.

**Sparse vs dense.** A sparse reward (+1 only at the goal, 0 everywhere else) is honest — it says exactly what you want and nothing else — but it is brutally hard to learn from, because the agent must stumble onto the goal by chance before it gets any signal. A dense reward (a shaped signal at every step) learns fast but invites hacking, because every shaping term is a new thing to exploit. The art is shaping that accelerates learning *without* changing the optimal policy.

**Potential-based shaping** is the one theoretically safe shaping technique, and it is worth knowing precisely because it is the only one with a guarantee. The theorem (Ng, Harada, Russell, 1999) says: if you add a shaping reward of the form

$$
F(s, a, s') = \gamma \, \Phi(s') - \Phi(s)
$$

for any potential function $\Phi : \mathcal{S} \to \mathbb{R}$, then the optimal policy is *unchanged*. The proof is a telescoping argument: summing $F$ along any trajectory collapses to $\gamma^T \Phi(s_T) - \Phi(s_0)$, a term that depends only on endpoints, so it shifts every trajectory's return by the same state-dependent constant and cannot change which trajectory is best. This is the rigorous version of "give the agent a hint about progress without bribing it to do the wrong thing." In a maze, $\Phi(s) = -\text{distance-to-goal}$ gives a dense progress signal that provably leaves the optimal path optimal. The [reward shaping for financial RL](/blog/machine-learning/reinforcement-learning/reward-shaping-for-financial-rl) post applies exactly this discipline to trading rewards, where naive shaping (rewarding every profitable tick) reliably produces over-trading.

**When you cannot specify the reward at all**, you have two escape hatches. Inverse RL (learn the reward from expert demonstrations) and preference learning (learn a reward model from human comparisons). The second is the entire foundation of [RLHF](/blog/machine-learning/reinforcement-learning/why-language-models-need-rlhf): you cannot write down "be helpful and harmless" as a function, but humans can reliably say which of two responses is better, and a [reward model trained on those preferences](/blog/machine-learning/reinforcement-learning/reward-modeling-from-human-preferences) approximates the reward you could never specify. The reward model itself is then a Goodhart hazard — over-optimize against it and the policy finds its blind spots — which is *exactly* why RLHF keeps a KL penalty to a reference policy, the topic of decision 9's case study.

#### Worked example: a reward that hacks, and the fix

A summarization RLHF run uses a reward model that (like all reward models) slightly prefers longer summaries on average. The team runs PPO against it with no length control. Over 400 PPO steps the mean reward climbs from 0.0 to +2.3 — looks great. Then they read the outputs: average summary length has ballooned from 60 tokens to 230, full of padding and repetition, because the policy discovered that *length itself* raises the reward-model score faster than *quality* does. Win-rate against the reference, measured by fresh human raters, has actually *dropped* from 50% to 41%. The reward went up; the product got worse. The fixes are textbook: add a length penalty to the reward, add the KL-to-reference term (which penalizes drifting into the high-length region the reward model never saw in training), and re-train the reward model on length-balanced pairs. After the fix, reward climbs to a more modest +1.1 but human win-rate rises to 68%. The lesson, again: *the metric is not the goal.*

```python
import torch
import torch.nn.functional as F

def shaped_reward(rm_score, response_len, ref_logp, policy_logp,
                  beta=0.1, len_penalty=0.001, target_len=80):
    # 1) raw reward-model score (the thing we are tempted to maximize)
    r = rm_score
    # 2) potential-style length penalty: discourage drifting off target length
    r = r - len_penalty * abs(response_len - target_len)
    # 3) KL-to-reference penalty: stay in the distribution the RM was trained on
    kl = (policy_logp - ref_logp)           # per-token log-ratio, summed upstream
    r = r - beta * kl
    return r
```

Every term in that function exists to defeat a specific failure mode you have seen in this series. The bare `rm_score` is what naive RLHF maximizes; the length penalty defeats the length-hack; the `beta * kl` term — the single most important line in all of RLHF — keeps the policy near the reference so the reward model is never asked to score the out-of-distribution garbage it would happily reward. Reward design is not a hyperparameter; it is the specification of the problem.

## Decision 4: exploration — discovering behavior you have never tried

An agent can only learn from outcomes it actually experiences, which creates the foundational tension of the field, covered in [exploration vs exploitation](/blog/machine-learning/reinforcement-learning/exploration-vs-exploitation-the-core-tension): exploit what you know works, or explore in case something better exists. Get the balance wrong in either direction and you fail — pure exploitation locks onto the first mediocre strategy it finds; pure exploration never commits to anything. The right strategy depends entirely on the structure of your reward, and the menu maps cleanly onto algorithm families.

**Epsilon-greedy** — with probability $\varepsilon$ take a random action, otherwise act greedily — is the workhorse for discrete value-based methods (DQN). Cheap, dumb, and good enough when rewards are reasonably dense. Anneal $\varepsilon$ from 1.0 to about 0.05 over training.

**Entropy regularization** is the continuous-control analogue and the reason SAC explores so well. Add $\alpha \, \mathcal{H}(\pi(\cdot \mid s))$ to the objective — a bonus for keeping the action distribution spread out. This is principled exploration: the policy is rewarded for staying uncertain until the value signal justifies committing. SAC's `ent_coef="auto"` even tunes $\alpha$ to hit a target entropy, removing a hyperparameter.

**Bandit methods — UCB and Thompson sampling** — are the right tool when there is no state, only a choice among arms with unknown payoffs. This is the regime of [recommendation and search](/blog/machine-learning/reinforcement-learning/rl-for-recommendation-and-search), where the contextual-bandit framing often beats full RL because the horizon is short and the credit-assignment is trivial. Upper-confidence-bound exploration ("optimism in the face of uncertainty") has tight regret bounds; Thompson sampling (act according to a posterior sample) is often better in practice and trivially parallelizable.

**Intrinsic motivation — ICM and RND** — is for the brutal case of *sparse* rewards, where the agent could explore for a million steps and never see a single positive signal. The trick is to manufacture a curiosity reward from prediction error: reward the agent for visiting states its own model finds surprising. Random Network Distillation (RND) is the clean version — keep a fixed random target network and a predictor trained to match it; the predictor's error is high on novel states, so prediction error *is* a novelty bonus. This is what cracked Montezuma's Revenge, the Atari game that had defeated every dense-reward method.

**Meta-exploration** — learning *how to explore* across a family of tasks — is the frontier covered in [meta-learning and few-shot RL](/blog/machine-learning/reinforcement-learning/meta-learning-and-few-shot-rl). When you face many related tasks, an agent can learn an exploration strategy that probes the right things on a new task in a handful of episodes, rather than re-exploring from scratch every time.

| Reward structure | Action space | Exploration choice | Why |
|---|---|---|---|
| Dense | Discrete | $\varepsilon$-greedy, annealed | Cheap, sufficient signal already present |
| Dense | Continuous | Entropy bonus (SAC) | Principled, auto-tunable, no schedule |
| Stateless / contextual | Discrete arms | UCB / Thompson | Tight regret bounds, short horizon |
| Sparse | Any | RND / ICM intrinsic reward | Manufacture signal from novelty |
| Task family | Any | Meta-exploration | Amortize exploration across tasks |

The exploration decision is downstream of the reward decision: a dense, well-shaped reward needs almost no exploration machinery, while a sparse reward forces you up the complexity ladder to intrinsic motivation. If you find yourself reaching for RND, ask first whether potential-based shaping (decision 3) could densify the reward instead — it is usually the cheaper fix.

## Decision 5: training infrastructure — keeping the GPUs fed

RL training has a structural problem that supervised learning does not: it generates its own data. The policy must act in the environment to produce the transitions it learns from, which couples *rollout collection* (often CPU-bound, environment-bound, or — for LLMs — inference-bound) to *gradient updates* (GPU-bound). The central engineering question is how to keep both busy. Figure 4 shows the full system as a cycle, with evaluation gating deployment and monitoring feeding back to retraining.

![A directed graph of a production RL system where environment and reward feed distributed training, training feeds parallel simulation and held-out evaluation that merge into a promotion gate, then shadow-to-canary deployment with a monitoring edge looping back to retraining](/imgs/blogs/the-reinforcement-learning-playbook-4.png)

**Single machine vs cluster.** Many problems fit on one machine. CartPole trains in seconds; a MuJoCo task with SAC trains in hours on one GPU. Do not reach for a cluster until a single machine is genuinely the bottleneck — distributed RL multiplies your failure modes. When you do need scale, the question becomes synchronous vs asynchronous.

**Synchronous (PPO-style).** Collect a big batch of rollouts from $N$ parallel environments, then do a synchronized gradient update. This is what makes massively-parallel simulators shine: Isaac Gym runs 4096 environments on one GPU, so PPO collects an enormous on-policy batch every iteration and the GPU never starves. The [sim-to-real robotics](/blog/machine-learning/reinforcement-learning/rl-for-robotics-sim-to-real) post lives in this regime — 4096 parallel envs turn a problem that would take a physical robot years into a few GPU-hours.

**Asynchronous (Ray/RLlib).** Many rollout workers feed a central learner without waiting for each other. This is essential when environments are slow or heterogeneous, and it is the architecture of large-scale systems. RLlib's `Algorithm` class makes this declarative.

```python
from ray.rllib.algorithms.ppo import PPOConfig

config = (
    PPOConfig()
    .environment("HalfCheetah-v4")
    .env_runners(num_env_runners=64)        # 64 parallel rollout workers
    .training(train_batch_size=131072,      # huge on-policy batch
              minibatch_size=4096,
              num_epochs=10, lr=3e-4, gamma=0.99)
    .resources(num_gpus=4)                   # 4 GPUs for the learner
)
algo = config.build()
for i in range(1000):
    result = algo.train()
    if i % 50 == 0:
        print(i, result["env_runners"]["episode_return_mean"])
```

**For RLHF specifically**, the infrastructure problem is acute and gets its own posts. The bottleneck is *rollout generation* — the policy must autoregressively decode long completions, which is slow — so modern stacks separate a fast inference engine from the training engine. The [vLLM async rollout collection](/blog/machine-learning/reinforcement-learning/vllm-async-rollout-collection) post explains how an inference server generates completions while training proceeds, and the [distributed RLHF system design](/blog/machine-learning/reinforcement-learning/distributed-rlhf-system-design) and [multi-node Ray/OpenRLHF](/blog/machine-learning/reinforcement-learning/multi-node-rl-training-ray-openrlhf) posts cover sharding the model across nodes with [DeepSpeed ZeRO](/blog/machine-learning/reinforcement-learning/deepspeed-zero-for-rlhf) or [FSDP mixed precision](/blog/machine-learning/reinforcement-learning/fsdp-mixed-precision-rl-training), with [LoRA/QLoRA](/blog/machine-learning/reinforcement-learning/lora-qlora-for-rlhf) to shrink the trainable footprint. The [GPU profiling and optimization](/blog/machine-learning/reinforcement-learning/gpu-profiling-optimization-rl-training) post is the one to read when your H100s are 30% utilized and you cannot figure out why — usually the answer is that rollout generation is starving the trainer.

**Hyperparameter tuning at scale.** RL is famously hyperparameter-sensitive, and the most effective tool for that is Population-Based Training (PBT): run a population of agents, periodically copy the weights of the best performers onto the worst, and perturb their hyperparameters. This evolves a *schedule* of hyperparameters, not a fixed set, which matters because the best learning rate early in training is not the best learning rate late. `ray.tune` implements PBT directly. OpenAI Five used PBT at enormous scale.

| Setup | Best for | Tool | Watch out for |
|---|---|---|---|
| Single machine, sync | Most problems, prototyping | SB3, clean PPO/SAC | Don't over-engineer; start here |
| Massively parallel sync | Sim-heavy robotics | Isaac Gym + PPO | GPU memory for 4096 envs |
| Async distributed | Slow/heterogeneous envs | Ray / RLlib | Stale-gradient drift |
| Disaggregated inference+train | RLHF / LLM alignment | vLLM + DeepSpeed/FSDP | Rollout engine starving trainer |
| Population-based tuning | HP-sensitive problems | ray.tune PBT | Compute cost of a population |

## Decision 6: evaluation — what "good" actually means

Here is the trap that catches more production RL systems than any algorithm bug: **the reward curve looks great and the policy is broken.** Episodic return during training is necessary but radically insufficient as an evaluation, for three reasons. First, it is measured on the *training* environment and seeds, so it tells you nothing about generalization. Second, it is an average, so it hides catastrophic tail behavior — a policy with a great mean return can still drive off a cliff one time in fifty. Third, for anything with constraints (safety, budget, latency), return ignores violations entirely. A real evaluation harness has four layers.

**Held-out seeds and held-out environments.** Train on seeds 0–9, evaluate on seeds 100–119. If performance collapses, you have overfit to the training dynamics — extremely common, and the [deadly triad](/blog/machine-learning/reinforcement-learning/the-deadly-triad-stability-in-deep-rl) post explains why deep RL is so prone to it. For robotics, the held-out test is a battery of perturbations the agent never saw in training (different masses, friction, sensor noise). Report mean *and* the worst-case across seeds; the spread is as important as the center.

**Safety and constraint metrics.** If your problem has constraints — a robot must not exceed a torque limit, a trading agent must not breach a position limit, an assistant must not produce harmful content — you must measure violation *rate*, not just average return. This is the entire point of [safe RL and constrained optimization](/blog/machine-learning/reinforcement-learning/safe-rl-constrained-optimization): you are solving a *constrained* MDP, and PPO-Lagrangian or a similar method enforces $\mathbb{E}[\text{cost}] \le d$ while maximizing return. The evaluation must report the constraint metric prominently, because a policy that gets 5% more return while violating the torque limit 2% of the time is not better — it is a recall waiting to happen.

**Offline / counterfactual evaluation.** For recommendation and any setting where you cannot freely experiment on real users, you evaluate a new policy on *logged* data from the old one using off-policy estimators (importance sampling, doubly-robust estimators). The [offline RL](/blog/machine-learning/reinforcement-learning/offline-rl-learning-from-fixed-datasets) and [conservative Q-learning](/blog/machine-learning/reinforcement-learning/conservative-q-learning-cql) posts cover learning *and* evaluating from fixed datasets; the [recommendation and search](/blog/machine-learning/reinforcement-learning/rl-for-recommendation-and-search) post covers off-policy evaluation specifically.

**Domain-specific evaluation.** Financial RL has its own evaluation discipline because the failure modes are unique and expensive: lookahead bias, overfitting to a backtest, survivorship bias, and ignoring transaction costs. The [financial RL backtesting and pitfalls](/blog/machine-learning/reinforcement-learning/financial-rl-backtesting-and-pitfalls) post and the [gym trading environments](/blog/machine-learning/reinforcement-learning/gym-trading-environments-and-backtesting) post are required reading: a backtest Sharpe of 3.0 that becomes 0.4 live is the normal outcome of a naive evaluation, and walk-forward testing with realistic slippage is the only honest way to estimate live performance. For RLHF, the gold standard is *human* evaluation — pairwise win-rate against a baseline, judged by fresh raters — backed by cheaper automatic proxies that you have validated against human judgment.

#### Worked example: the evaluation that saved a deployment

A robotics team trains a quadruped locomotion policy. Mean episodic return on training seeds: 1,950, up from a random baseline of 120. Looks shippable. The four-layer harness says otherwise. On 20 *held-out* seeds with randomized terrain, mean return drops to 1,400 — acceptable, some overfitting. But the worst-of-20 is 310, and a video of that seed shows the robot falling on a downhill slope it never saw in training. The constraint metric is worse: peak joint torque exceeds the hardware limit on 8% of held-out episodes, which on real hardware means burned-out motors. The mean return hid all of it. The fix is not a better algorithm — it is more domain randomization in training (decision 1, augmenting the environment) and a torque constraint via PPO-Lagrangian (decision 3 plus safe RL). After re-training, held-out worst-of-20 rises to 1,150 and torque violations drop to 0.1%. The policy that *would* have shipped on mean return would have destroyed motors in the field. Evaluation is where you find that out cheaply instead of expensively.

```python
import numpy as np
from stable_baselines3.common.evaluation import evaluate_policy

def four_layer_eval(model, train_env, held_out_envs, cost_fn, n=50):
    # Layer 1: training-seed sanity (necessary, not sufficient)
    train_r, _ = evaluate_policy(model, train_env, n_eval_episodes=n)

    # Layer 2: held-out generalization — report mean AND worst-case
    returns = []
    for env in held_out_envs:
        r, _ = evaluate_policy(model, env, n_eval_episodes=n,
                               return_episode_rewards=True)
        returns.extend(r[0] if isinstance(r, tuple) else r)
    returns = np.array(returns)

    # Layer 3: constraint violation rate (the metric return hides)
    violation_rate = np.mean([cost_fn(env, model) for env in held_out_envs])

    return {
        "train_mean": train_r,
        "heldout_mean": returns.mean(),
        "heldout_worst": returns.min(),          # the number that matters
        "heldout_p10": np.percentile(returns, 10),
        "constraint_violation_rate": violation_rate,
    }
```

The single discipline that separates teams that ship working RL from teams that ship surprises: **never promote a policy on mean training return alone.** The [debugging](/blog/machine-learning/reinforcement-learning/the-deadly-triad-stability-in-deep-rl) discipline of the series is that most "the algorithm doesn't work" reports are actually "my evaluation lied to me" reports.

## Decision 7: deployment — getting it into the world safely

A trained, evaluated policy is not a product until it survives contact with reality, and reality differs from your simulator and your held-out set in ways you did not anticipate. Deployment is a risk-management discipline, and the same staged rollout that serves the rest of software engineering serves RL — with RL-specific monitoring. Figure 5 contrasts the fragile one-off script (what most RL starts as) with what a production system needs.

![A before-after figure contrasting a fragile one-off script with hard-coded hyperparameters and no logging or rollback against a production system with versioned config, seeded multi-seed evaluation, and automatic canary rollback](/imgs/blogs/the-reinforcement-learning-playbook-5.png)

**Serving and latency.** A policy is a neural network; serving it is an inference problem. The constraint is latency: a high-frequency trading policy must decide in microseconds (the [high-frequency trading](/blog/machine-learning/reinforcement-learning/rl-for-high-frequency-trading) and [market-making](/blog/machine-learning/reinforcement-learning/rl-market-making-and-order-execution) posts live in this regime), a robot controller at 50–1000 Hz, a recommendation policy in tens of milliseconds, an LLM assistant in seconds. The latency budget can retroactively force algorithm decisions — a giant model that cannot serve in budget is not a candidate no matter how good its return.

**Staged rollout: shadow → canary → production.** In *shadow mode*, the new policy runs alongside the live system and its actions are logged but not executed — you compare what it *would* have done against what the incumbent did, with zero risk. In *canary*, it takes a small slice of real traffic (1–5%) while you watch the metrics. Only after the canary holds do you ramp to full production. This is exactly how the [recommendation](/blog/machine-learning/reinforcement-learning/rl-for-recommendation-and-search) and RLHF systems graduate from offline win-rate to live A/B tests.

**Monitoring the RL-specific signals.** Standard service monitoring (latency, error rate) is necessary but you also watch RL-native signals that predict policy decay:

- **Reward drift** — the realized reward trending down means the environment has shifted out from under a policy trained on the old distribution. In trading this is regime change; in recommendation it is changing user behavior.
- **Entropy collapse** — the policy's action distribution becoming near-deterministic. Sometimes this is convergence; in a non-stationary environment it is the policy becoming brittle, unable to adapt. A sudden entropy drop is a red flag.
- **Constraint violations** — the safety metric from evaluation, now monitored live. The first violation should page someone.
- **Distribution shift** — the live state distribution diverging from the training distribution, measured by something as simple as the predictor error of an RND-style network or a learned density model.

**Rollback criteria, decided in advance.** The single most important deployment artifact is a written rollback rule, decided *before* you ship: "if live reward drops more than 15% below the canary baseline over a one-hour window, or any safety constraint is violated, automatically roll back to the previous policy." Automatic rollback is what turns a deployment from a bet into a controlled experiment. The [sim-to-real](/blog/machine-learning/reinforcement-learning/rl-for-robotics-sim-to-real) post adds the robotics-specific tool: real-time adaptation (RMA-style), where the policy adjusts to the actual dynamics on the physical robot rather than hoping the simulator was right.

| Stage | Traffic | Risk | What you watch | Promote when |
|---|---|---|---|---|
| Shadow | 0% (logged) | None | Action agreement vs incumbent | Logged actions look sane |
| Canary | 1–5% | Low | Reward, entropy, constraints, latency | Metrics hold for full window |
| Production | 100% | Managed | Same, plus drift detectors | n/a (steady state) |
| Rollback | — | — | Automatic trigger fires | Always armed |

#### Worked example: a canary that caught reward drift before users did

A recommendation team ships a new policy trained to optimize long-horizon engagement. Offline evaluation on logged data (decision 6, off-policy estimator) projected a +6% lift in session depth. They deploy to a 3% canary with a written rollback rule: revert if canary reward falls more than 10% below the offline projection over any rolling 6-hour window, or if the policy's action entropy drops below 0.15 (an entropy-collapse guard). For the first two days the canary tracks the projection — +5.4% session depth, entropy steady at 0.42. On day three, a content-catalog change shifts the state distribution, and over six hours the canary reward drifts down to +1.1% while the policy's action entropy *also* slides toward 0.22, both leading indicators that the policy is operating off-distribution. The drift monitor fires at the 10% threshold and the system rolls back to the previous policy automatically, before the change ever reaches the 97% of traffic on the incumbent. The postmortem traces the root cause to decision 1 (the state did not encode catalog-version, so a catalog change looked like non-stationarity) and the fix lives there — but the *deployment* discipline is what turned a potential multi-day regression into a six-hour canary blip that no production user felt. The number that mattered was not the offline +6%; it was the armed, pre-written rollback rule.

```python
def rollback_monitor(canary_reward, offline_projection, action_entropy,
                     window_drop=0.10, entropy_floor=0.15):
    drift = (offline_projection - canary_reward) / abs(offline_projection)
    if drift > window_drop:
        return "ROLLBACK", f"reward drift {drift:.0%} over window"
    if action_entropy < entropy_floor:
        return "ROLLBACK", f"entropy collapse {action_entropy:.2f}"
    return "HOLD", "metrics within bounds"
```

That monitor is decision 7 made operational: the leading indicators (reward drift, entropy collapse) are the RL-native signals from the table above, and the function returns a *decision*, not just an alert, because an automatic rollback is what separates a controlled experiment from a bet on the business.

## Integration case study: a conversational AI assistant with RLHF

Now we assemble all seven decisions into one real system. This is the canonical modern RL application — aligning a language model — and it touches every decision in the framework. Figure 6 shows the pipeline.

![A directed graph of an RLHF production pipeline where human preferences train a reward model and a frozen reference policy both feed KL-constrained PPO, whose output is gated by win-rate evaluation before canary, A/B test, and full rollout](/imgs/blogs/the-reinforcement-learning-playbook-6.png)

**Decision 1 — formulation.** The MDP: the state is the conversation history (the prompt plus all tokens generated so far), the action is the next token, the transition is deterministic (appending the token), and the reward arrives at the *end* of the completion as a single scalar from the reward model. Is it Markov? Yes, *if* the state is the full token sequence — which it is, because the transformer attends over the whole context. This is a clean MDP precisely because the history is the state. The horizon is the completion length; the discount is typically near 1 because we care about the final completion, not early tokens.

**Decision 2 — algorithm.** Tokens are a (large, discrete) action space and the target is a language model, so we are on the RLHF branch of the tree. The classic choice is [PPO with a reward model](/blog/machine-learning/reinforcement-learning/ppo-for-llm-fine-tuning) and a value head; the modern, leaner choice is [GRPO](/blog/machine-learning/reinforcement-learning/grpo-group-relative-policy-optimization), which deletes the value network by estimating the advantage from a *group* of sampled completions to the same prompt. We will use PPO here because its KL machinery makes the reward-hacking defense explicit, and note where GRPO simplifies it.

**Decision 3 — reward.** We cannot write "be helpful and harmless" as a function, so we learn it: collect human preference pairs (response A vs response B), train a [reward model](/blog/machine-learning/reinforcement-learning/reward-modeling-from-human-preferences) with the Bradley-Terry loss

$$
\mathcal{L}_{\text{RM}} = -\mathbb{E}_{(x, y_w, y_l)}\big[ \log \sigma\big( r_\phi(x, y_w) - r_\phi(x, y_l) \big) \big],
$$

where $y_w$ is the preferred (winning) response. The reward model is a Goodhart hazard, so the *actual* training reward is the reward-model score minus a KL penalty to the reference policy. That KL term is the single most important design decision in the whole system, which is why decision 3 spent so long on it.

**Decision 4 — exploration.** Exploration is temperature sampling during rollout generation: sample completions at temperature ~1.0 (or use GRPO's group of samples) so the policy explores different phrasings, then learns which the reward model prefers. Too low a temperature and the policy never discovers better responses; too high and the rollouts are gibberish the reward model cannot score meaningfully.

**Decision 5 — infrastructure.** This is where RLHF gets hard. The bottleneck is rollout generation — autoregressive decoding is slow — so we run a [vLLM inference engine](/blog/machine-learning/reinforcement-learning/vllm-async-rollout-collection) for fast generation, shard the policy and reference models across GPUs with [DeepSpeed ZeRO](/blog/machine-learning/reinforcement-learning/deepspeed-zero-for-rlhf) or [FSDP](/blog/machine-learning/reinforcement-learning/fsdp-mixed-precision-rl-training), and optionally use [LoRA](/blog/machine-learning/reinforcement-learning/lora-qlora-for-rlhf) to shrink the trainable parameters. The [distributed RLHF system design](/blog/machine-learning/reinforcement-learning/distributed-rlhf-system-design) post is the architecture reference.

The TRL implementation makes the structure concrete:

```python
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from trl import create_reference_model
from transformers import AutoTokenizer

config = PPOConfig(
    model_name="my-sft-model",
    learning_rate=1.41e-5,
    batch_size=256,
    mini_batch_size=16,
    init_kl_coef=0.2,        # the reward-hacking defense (decision 3)
    target=6.0,              # adaptive KL target; raise/lower init_kl_coef to hold it
)
policy = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name)
ref_policy = create_reference_model(policy)   # frozen SFT policy
tokenizer = AutoTokenizer.from_pretrained(config.model_name)
trainer = PPOTrainer(config, policy, ref_policy, tokenizer)

for batch in dataloader:
    query_tensors = batch["input_ids"]
    # Decision 4: explore via sampling
    response_tensors = trainer.generate(
        query_tensors, max_new_tokens=128, do_sample=True, top_k=0, top_p=1.0
    )
    # Decision 3: reward = RM score (KL is handled inside PPOTrainer via ref_policy)
    rewards = reward_model(query_tensors, response_tensors)
    # PPO update with clipped surrogate + KL-to-reference penalty
    stats = trainer.step(query_tensors, response_tensors, rewards)
    print(stats["ppo/mean_scores"], stats["objective/kl"])
```

The `init_kl_coef` and `target` lines are decision 3 made operational: the trainer adaptively scales the KL penalty to hold the KL near a target, which is the mechanism that keeps the policy from wandering into the reward model's blind spots. Watch `objective/kl` in training like a hawk — if it spikes, the policy is escaping the reference and reward hacking is imminent.

**Decision 6 — evaluation.** The reward-model score is *not* the evaluation — it is the training signal, and optimizing it is exactly the Goodhart trap. The real evaluation is human pairwise win-rate against the SFT baseline on a held-out prompt set, backed by automatic proxies (validated LLM-judge win-rate, length-controlled to defeat the length hack from decision 3). A healthy run might move human win-rate from 50% to roughly 68–71% (the InstructGPT range) while holding KL modest. If reward-model score is up but human win-rate is flat or down, you are hacking the reward model — stop and fix decision 3.

**Decision 7 — deployment.** Shadow mode: run the aligned model alongside the production model, log both responses, have raters compare offline. Canary: 1–5% of real traffic, watching live thumbs-up rate, refusal rate, and latency. A/B test: a proper experiment measuring the business metric. Full rollout only if the A/B holds. Rollback rule, written in advance: if live preference rate drops below the canary baseline or refusal rate spikes, revert to the previous model automatically. Monitor entropy collapse (the model becoming repetitive) and reward drift (user satisfaction trending down as the world changes) as the leading indicators of decay.

That is a complete RLHF system, every decision accounted for. The [DPO](/blog/machine-learning/reinforcement-learning/dpo-direct-preference-optimization) alternative collapses decisions 2–5 by skipping the RL loop entirely — it optimizes the preference objective directly with a supervised-style loss — trading some ceiling for enormous operational simplicity. For many teams DPO is the right first move, and PPO/GRPO the move you make when DPO's ceiling is not enough; that trade is exactly decision 2 applied to your constraints.

## Integration case study: a production robotics RL system

The second system is the mirror image — continuous control, physical hardware, sample-scarce on the real robot but sample-rich in simulation. It exercises the same seven decisions with different answers, which is the point: the *framework* is invariant, the *choices* are domain-specific.

**Decision 1 — formulation.** A quadruped locomotion controller. The true state (terrain geometry, exact friction, motor temperatures) is not observable, so this is a POMDP. The agent observes proprioception — joint angles, velocities, IMU, and a short history. We make it tractable by stacking the last few observation frames (recovering a Markov-enough state) and, in the [sim-to-real](/blog/machine-learning/reinforcement-learning/rl-for-robotics-sim-to-real) approach, by training a context encoder that infers the hidden dynamics parameters from the recent observation-action history. The action is the target joint positions; the controller runs at high frequency.

**Decision 2 — algorithm.** Continuous actions, and samples are *cheap in simulation* (Isaac Gym, 4096 parallel envs), so the sample-cost branch of the tree points to model-free actor-critic, and PPO is the proven default for massively-parallel sim-to-real. We pair it with a curriculum: start on flat ground, progressively add slopes, stairs, and obstacles, so the policy learns the easy cases before the hard ones.

**Decision 3 — reward.** Dense and shaped: a velocity-tracking term (match a commanded velocity), an energy/torque penalty (efficiency and hardware protection), and stability terms (penalize body roll, reward foot clearance). This is exactly the reward-hacking minefield from decision 3 — a naive velocity reward produces a policy that lunges and falls, "achieving" high instantaneous velocity. Potential-based shaping and careful penalty weighting keep the optimal policy aligned with "walk well," not "spike the velocity sensor." The torque penalty doubles as a soft safety constraint.

**Decision 4 — exploration.** Two layers. PPO's action-distribution noise provides local exploration. The bigger lever is *domain randomization*: randomize masses, friction, motor strength, and sensor latency across the 4096 envs, which forces the policy to explore a wide range of dynamics and is what makes it transfer to the real robot. A teacher-student setup (a teacher with privileged simulator state distills into a student that only sees real sensors) is the [sim-to-real](/blog/machine-learning/reinforcement-learning/rl-for-robotics-sim-to-real) trick that closes the gap.

**Decision 5 — infrastructure.** Massively parallel synchronous PPO on a single GPU running Isaac Gym at 4096 environments — tens of thousands of environment steps per second, so a policy that would take a physical robot years to train converges in a few GPU-hours. This is the synchronous, sim-heavy regime from decision 5's table.

```python
# Isaac Gym-style massively-parallel PPO config (RLlib-flavored for clarity)
from ray.rllib.algorithms.ppo import PPOConfig

config = (
    PPOConfig()
    .environment("QuadrupedTerrain-v0", env_config={
        "num_envs": 4096,                  # parallel envs on one GPU
        "domain_randomization": True,      # decision 4: randomize dynamics
        "curriculum": "flat->slope->stairs",  # decision 4: progressive difficulty
    })
    .training(
        train_batch_size=4096 * 24,        # 24 steps x 4096 envs per update
        lr=3e-4, gamma=0.99, lambda_=0.95,
        entropy_coeff=0.005,               # decision 4: local exploration
        vf_loss_coeff=1.0, clip_param=0.2,
    )
    .resources(num_gpus=1)
)
```

**Decision 6 — evaluation.** The four-layer harness, robotics flavor. Layer 1: training return. Layer 2: a held-out simulation test suite — terrains, payloads, and pushes the policy never trained on — reporting worst-case, not just mean. Layer 3: the safety/constraint metric — peak torque and contact forces vs hardware limits, the metric from the worked example in decision 6 that caught the motor-destroying policy. Layer 4: the real-robot test — a fixed battery of 50 physical trials measuring success rate, falls, and energy. The sim-to-real gap shows up exactly here: a policy that aced simulation but fails the 50-trial physical test has overfit the simulator, and the fix is more domain randomization (back to decision 4).

**Decision 7 — deployment.** On the physical robot, deployment includes a *safety layer* that the RL policy cannot override — hard torque and joint limits enforced below the policy, so a bad action is clamped, not executed. Rapid Motor Adaptation (RMA) gives real-time adaptation: a small adaptation module adjusts to the actual dynamics on the real robot in milliseconds, compensating for the residual sim-to-real gap. Monitoring watches for the locomotion equivalents of reward drift and entropy collapse — gait irregularity, rising energy use, near-falls — and the rollback is to a conservative, slower, known-safe gait. The staged rollout is literal: tethered tests, then a controlled arena, then the field.

The two case studies share *zero* algorithmic details — one optimizes tokens with KL-constrained PPO, the other optimizes joint torques with domain-randomized PPO — yet they are the *same system* viewed through the seven decisions. That invariance is the entire thesis. Once you see RL as the framework rather than the algorithm, a new problem is not "which algorithm?" but "what are my seven answers?"

## The RL system as a stack

It helps to see the whole thing as layers, because it clarifies where a given bug *can* be fixed. Figure 7 stacks the six layers from mathematical foundations up to operations.

![A six-layer stack from MDP and Bellman foundations up through algorithms, infrastructure, evaluation, deployment, and operations, each layer resting on the one below](/imgs/blogs/the-reinforcement-learning-playbook-7.png)

The stack encodes a rule that saves enormous time: **a bug at a given layer cannot be fixed at a higher layer.** A formulation bug (math/foundations layer — a non-Markov state) cannot be fixed by tuning the algorithm, the infrastructure, or the deployment. A reward bug (foundations again — Goodhart) cannot be fixed by more compute. An evaluation bug (you measured the wrong thing) cannot be fixed by deployment safeguards, because you do not even know the policy is bad. When a production RL system misbehaves, diagnose *top-down* — is it a deployment/ops issue (environment shifted, monitor firing)? — but *fix bottom-up* — is the real root cause in the formulation or reward? The vast majority of "the algorithm doesn't work" tickets resolve to a foundations-layer bug, which is why this post spends its first three decisions there and treats algorithm selection as nearly mechanical.

This is also the reading-order map for the series. The foundations layer is [what RL is](/blog/machine-learning/reinforcement-learning/what-is-reinforcement-learning), [MDPs](/blog/machine-learning/reinforcement-learning/markov-decision-processes), [the Bellman equation](/blog/machine-learning/reinforcement-learning/value-functions-and-the-bellman-equation), [dynamic programming](/blog/machine-learning/reinforcement-learning/dynamic-programming-for-rl), [TD learning](/blog/machine-learning/reinforcement-learning/temporal-difference-learning-td0-and-sarsa), and [Monte Carlo methods](/blog/machine-learning/reinforcement-learning/monte-carlo-methods-in-rl). The algorithms layer is the value-based track (DQN through Rainbow), the policy-gradient track (REINFORCE through PPO/SAC), the model-based track, and the RLHF track. The infrastructure, evaluation, and deployment layers are the production and applications tracks. Read bottom-up to learn the field; read top-down (start here) to ship a system.

## The series in one map

To make this capstone genuinely useful as a reference, here is the whole series organized by the seven decisions, so you can jump to the post that answers the decision in front of you.

**Decision 1 — Formulation.** [What is RL](/blog/machine-learning/reinforcement-learning/what-is-reinforcement-learning), [MDPs](/blog/machine-learning/reinforcement-learning/markov-decision-processes), [value functions and Bellman](/blog/machine-learning/reinforcement-learning/value-functions-and-the-bellman-equation), [policies: deterministic vs stochastic](/blog/machine-learning/reinforcement-learning/policies-deterministic-vs-stochastic), [the credit assignment problem](/blog/machine-learning/reinforcement-learning/the-credit-assignment-problem), [multi-agent fundamentals](/blog/machine-learning/reinforcement-learning/multi-agent-rl-fundamentals) for when the environment contains other learners.

**Decision 2 — Algorithm.** Tabular foundations: [dynamic programming](/blog/machine-learning/reinforcement-learning/dynamic-programming-for-rl), [Monte Carlo](/blog/machine-learning/reinforcement-learning/monte-carlo-methods-in-rl), [TD(0) and SARSA](/blog/machine-learning/reinforcement-learning/temporal-difference-learning-td0-and-sarsa), [Q-learning](/blog/machine-learning/reinforcement-learning/q-learning-off-policy-td-control), [n-step and TD(λ)](/blog/machine-learning/reinforcement-learning/n-step-returns-and-td-lambda). Scaling up: [function approximation](/blog/machine-learning/reinforcement-learning/function-approximation-why-tables-dont-scale), [neural value approximators](/blog/machine-learning/reinforcement-learning/neural-networks-as-value-approximators), [DQN](/blog/machine-learning/reinforcement-learning/deep-q-networks-dqn), [DQN improvements](/blog/machine-learning/reinforcement-learning/dqn-improvements-double-dueling-per), [Rainbow](/blog/machine-learning/reinforcement-learning/rainbow-dqn-combining-six-improvements), [distributional RL](/blog/machine-learning/reinforcement-learning/distributional-rl-c51-qr-dqn-iqn). Policy gradient: [the policy gradient theorem](/blog/machine-learning/reinforcement-learning/the-policy-gradient-theorem), [actor-critic](/blog/machine-learning/reinforcement-learning/actor-critic-a2c-a3c), [TRPO](/blog/machine-learning/reinforcement-learning/trust-region-policy-optimization-trpo), [PPO](/blog/machine-learning/reinforcement-learning/proximal-policy-optimization-ppo), [DDPG/TD3](/blog/machine-learning/reinforcement-learning/deterministic-policy-gradient-ddpg-td3), [SAC](/blog/machine-learning/reinforcement-learning/soft-actor-critic-sac). Model-based: [model-based RL](/blog/machine-learning/reinforcement-learning/model-based-rl-learning-world-models), [Dyna-Q](/blog/machine-learning/reinforcement-learning/dyna-q-and-planning-with-a-model), [MuZero](/blog/machine-learning/reinforcement-learning/muzero-mastering-games-without-rules), [world models](/blog/machine-learning/reinforcement-learning/world-models-dreamer-planet), [model-based vs model-free](/blog/machine-learning/reinforcement-learning/model-based-vs-model-free-when-to-use-which).

**Decision 3 — Reward.** [Reward hacking and Goodhart](/blog/machine-learning/reinforcement-learning/reward-hacking-and-goodharts-law), [reward modeling from preferences](/blog/machine-learning/reinforcement-learning/reward-modeling-from-human-preferences), [reward shaping for financial RL](/blog/machine-learning/reinforcement-learning/reward-shaping-for-financial-rl), and the whole RLHF reward story in [why LLMs need RLHF](/blog/machine-learning/reinforcement-learning/why-language-models-need-rlhf).

**Decision 4 — Exploration.** [Exploration vs exploitation](/blog/machine-learning/reinforcement-learning/exploration-vs-exploitation-the-core-tension), [meta-learning and few-shot RL](/blog/machine-learning/reinforcement-learning/meta-learning-and-few-shot-rl) for meta-exploration.

**Decision 5 — Infrastructure.** [vLLM async rollout](/blog/machine-learning/reinforcement-learning/vllm-async-rollout-collection), [distributed RLHF system design](/blog/machine-learning/reinforcement-learning/distributed-rlhf-system-design), [multi-node Ray/OpenRLHF](/blog/machine-learning/reinforcement-learning/multi-node-rl-training-ray-openrlhf), [DeepSpeed ZeRO for RLHF](/blog/machine-learning/reinforcement-learning/deepspeed-zero-for-rlhf), [FSDP mixed precision](/blog/machine-learning/reinforcement-learning/fsdp-mixed-precision-rl-training), [LoRA/QLoRA for RLHF](/blog/machine-learning/reinforcement-learning/lora-qlora-for-rlhf), [GPU profiling](/blog/machine-learning/reinforcement-learning/gpu-profiling-optimization-rl-training).

**Decision 6 — Evaluation.** [The deadly triad](/blog/machine-learning/reinforcement-learning/the-deadly-triad-stability-in-deep-rl), [offline RL](/blog/machine-learning/reinforcement-learning/offline-rl-learning-from-fixed-datasets), [conservative Q-learning](/blog/machine-learning/reinforcement-learning/conservative-q-learning-cql), [experience replay and offline data](/blog/machine-learning/reinforcement-learning/experience-replay-and-offline-data), [financial RL backtesting and pitfalls](/blog/machine-learning/reinforcement-learning/financial-rl-backtesting-and-pitfalls), [gym trading environments](/blog/machine-learning/reinforcement-learning/gym-trading-environments-and-backtesting).

**Decision 7 — Deployment.** [Safe RL and constrained optimization](/blog/machine-learning/reinforcement-learning/safe-rl-constrained-optimization), [RL for robotics sim-to-real](/blog/machine-learning/reinforcement-learning/rl-for-robotics-sim-to-real), [on-policy vs off-policy in practice](/blog/machine-learning/reinforcement-learning/on-policy-vs-off-policy-a-practical-guide), [tabular RL in practice](/blog/machine-learning/reinforcement-learning/tabular-rl-in-practice).

**Application tracks** that exercise all seven decisions end to end: the finance track — [algorithmic trading foundations](/blog/machine-learning/reinforcement-learning/rl-for-algorithmic-trading-foundations), [portfolio optimization](/blog/machine-learning/reinforcement-learning/rl-portfolio-optimization), [market making and order execution](/blog/machine-learning/reinforcement-learning/rl-market-making-and-order-execution), [high-frequency trading](/blog/machine-learning/reinforcement-learning/rl-for-high-frequency-trading), [deep hedging](/blog/machine-learning/reinforcement-learning/deep-hedging-options-with-rl); the games track — [Atari to AlphaGo](/blog/machine-learning/reinforcement-learning/rl-for-game-playing-atari-to-alphago), [emergent behavior and multi-agent games](/blog/machine-learning/reinforcement-learning/emergent-behaviour-and-multi-agent-games), [MADDPG](/blog/machine-learning/reinforcement-learning/maddpg-centralised-training-decentralised-execution), [MARL applications](/blog/machine-learning/reinforcement-learning/marl-applications-auctions-traffic-robotics); the alignment track — [PPO for LLM fine-tuning](/blog/machine-learning/reinforcement-learning/ppo-for-llm-fine-tuning), [GRPO](/blog/machine-learning/reinforcement-learning/grpo-group-relative-policy-optimization), [DPO](/blog/machine-learning/reinforcement-learning/dpo-direct-preference-optimization), [constitutional AI and RLAIF](/blog/machine-learning/reinforcement-learning/constitutional-ai-and-rlaif); and [recommendation and search](/blog/machine-learning/reinforcement-learning/rl-for-recommendation-and-search).

**Recommended reading order.** *Beginners*: read the foundations layer top to bottom, then DQN and PPO, then one application track that interests you. *Practitioners*: start here (the framework), then jump to the decision that is biting you. *Researchers*: foundations, then the frontier posts — distributional RL, model-based (MuZero, Dreamer), GRPO, offline RL, and the multi-agent track.

## Case studies: the framework in the historical record

The seven decisions are not invented; they are the abstraction behind every landmark RL result. Figure 8 places the milestones on a timeline.

![A timeline of reinforcement learning milestones from DQN on Atari in 2013 through AlphaGo, PPO, SAC, RLHF foundations, offline RL, InstructGPT, and GRPO with asynchronous rollout in 2024](/imgs/blogs/the-reinforcement-learning-playbook-8.png)

**DQN on Atari (Mnih et al., 2015).** Formulation: a POMDP made Markov by stacking four frames (decision 1 in action). Algorithm: value-based DQN (decision 2, discrete actions). Reward: the game score, clipped to $\{-1, 0, +1\}$ to stabilize across games (decision 3, reward engineering). Exploration: annealed $\varepsilon$-greedy (decision 4). The result: human-level or above on 29 of 49 games from raw pixels, the result that launched deep RL. The frame-stacking and reward-clipping are decisions 1 and 3 — the parts people forget DQN even had.

**AlphaGo / MuZero (Silver et al., 2016; Schrittwieser et al., 2020).** Model-based planning (decision 2) combined with self-play (a formulation choice, decision 1 — the opponent is a past version of yourself, making the environment stationary-enough). AlphaGo beat Lee Sedol 4–1; MuZero then matched it *without being told the rules*, learning the model itself — the purest demonstration that model-based RL buys you sample efficiency and generality when the dynamics are learnable.

**OpenAI Five (2019).** A staggering infrastructure result (decision 5): PPO scaled across thousands of GPUs with population-based training (decision 4/5), playing the equivalent of hundreds of years of Dota per day. It beat the world champions. The lesson was not a new algorithm — it was PPO — but that the *infrastructure* and *self-play formulation* were the hard parts. This is decision 5 written in compute.

**InstructGPT (Ouyang et al., 2022).** The case study that made RLHF mainstream and the template for our first integration study. Reward modeling from human preferences (decision 3), PPO with a KL penalty to the SFT reference (decisions 2 and 3), human win-rate evaluation (decision 6). The headline: labelers preferred the 1.3B InstructGPT model's outputs over the 175B GPT-3's, despite InstructGPT being ~100× smaller. Alignment via RLHF beat raw scale — the clearest evidence that reward design and the right objective (decisions 1–3) dominate the algorithm and the model size.

**DeepSeek-R1 / GRPO (2024–2025).** The frontier of decision 2 and 5: GRPO removes the value network (halving memory), and the rollout infrastructure (vLLM async generation, decision 5) is what made large-scale reasoning RL practical. The result was a reasoning model trained largely with rule-based rewards (verifiable correctness — the cleanest possible decision 3) that matched far more expensive pipelines. It is the current proof that getting decision 3 *clean* (verifiable rewards beat learned reward models when you can have them) is worth more than a fancier algorithm.

Every one of these is the same seven decisions with different answers. None of them won on the algorithm alone.

## When to use RL (and when not to)

The most valuable judgment in this whole framework is decision zero: *should this be RL at all?* I will be decisive, because vague advice here costs the most.

**Do not use RL when you can use supervised learning.** If you can label the correct action for each input, supervised learning is more stable, more sample-efficient, easier to debug, and easier to deploy. RL is for when you can score *outcomes* but cannot label *actions*.

**Do not use model-free RL when you can simulate cheaply and the dynamics are known.** For a known model and a small state space, [dynamic programming](/blog/machine-learning/reinforcement-learning/dynamic-programming-for-rl) (value iteration) gives the exact optimal policy with no sampling noise. Reaching for PPO on a problem value iteration solves exactly is engineering malpractice.

**For small discrete action spaces, value-based beats policy gradient.** [Q-learning](/blog/machine-learning/reinforcement-learning/q-learning-off-policy-td-control) and DQN are more sample-efficient than policy gradient on discrete problems because they reuse off-policy data. Save policy gradient for continuous actions or when you need a stochastic policy.

**For LLM alignment, try DPO before PPO.** [DPO](/blog/machine-learning/reinforcement-learning/dpo-direct-preference-optimization) skips the RL loop entirely and is dramatically simpler to run; reach for [PPO](/blog/machine-learning/reinforcement-learning/ppo-for-llm-fine-tuning) or [GRPO](/blog/machine-learning/reinforcement-learning/grpo-group-relative-policy-optimization) only when DPO's ceiling is not enough or you have verifiable rewards that an online method can exploit.

**For short-horizon, near-stateless decisions, use bandits, not RL.** If there is no meaningful state transition — just a choice with an immediate, observable payoff — a contextual bandit (decision 4's UCB/Thompson machinery) is simpler and tighter than full RL.

**Use RL when:** the payoff is delayed across a *sequence* of decisions, you can score outcomes but not label actions, the environment is interactive (your actions change future states), and you can either simulate cheaply or afford the sample cost. Robotics control, game-playing, dialogue alignment, trading execution, long-horizon recommendation — those are genuinely RL-shaped, and the seven-decision framework is how you ship them.

One more piece of decisive guidance, because it is the question every engineering manager actually asks: *is the cost worth it?* RL projects carry a tax that supervised projects do not — they are harder to debug (the failures hide in formulation and reward, two layers below where you look), harder to evaluate (mean return lies), and harder to operate (the environment shifts under a deployed policy). A reasonable rule is to budget RL at roughly two to three times the engineering effort of a comparable supervised solution, most of it spent in decisions 1, 3, and 6 rather than on the algorithm. That tax is worth paying when the *sequential* nature of the problem genuinely defeats a one-shot supervised model — when the right action now depends on consequences ten steps later that no static label can capture. If a strong supervised baseline gets you 90% of the value at a third of the cost, ship the baseline and revisit RL only when that last 10% is worth the multiplier. The discipline of asking this *before* the project, not after the second six-figure compute bill, is itself part of the framework — decision zero, the one that decides whether to make the other seven at all.

## Key takeaways

- **RL is seven decisions, not one algorithm.** Formulation, algorithm, reward, exploration, infrastructure, evaluation, deployment — the project fails at the weakest link, and the weakest link is rarely the algorithm.
- **Most failures are formulation or reward failures wearing an algorithm costume.** If a policy plateaus and thrashes, check whether your state is actually Markov before you touch a hyperparameter.
- **The reward gets exactly what it says, which is rarely what you meant.** Goodhart's law is the default outcome of optimization; potential-based shaping is the only theoretically safe shaping, and the KL-to-reference term is the load-bearing defense in RLHF.
- **Algorithm selection is nearly mechanical:** discrete → DQN/Rainbow; continuous + cheap samples → PPO/SAC; continuous + expensive samples → model-based; LLM → GRPO/DPO. Start with PPO.
- **Never promote a policy on mean training return.** Evaluate on held-out seeds, report worst-case, measure constraint violations, and use domain-appropriate evaluation (walk-forward for finance, real-hardware trials for robotics, human win-rate for RLHF).
- **Deployment is risk management:** shadow → canary → production, with an automatic rollback rule written *before* you ship, and RL-native monitors (reward drift, entropy collapse, constraint violations).
- **A bug at one layer cannot be fixed at a higher layer.** Diagnose top-down, fix bottom-up; the root cause of "the algorithm doesn't work" is almost always in the foundations.
- **The framework is invariant across domains.** An LLM aligner and a quadruped controller share no algorithmic details and are the same system through the seven decisions.

## Further reading

- Sutton & Barto, *Reinforcement Learning: An Introduction* (2nd ed., 2018) — the foundational text for the entire foundations layer.
- Mnih et al., "Human-level control through deep reinforcement learning," *Nature* (2015) — the DQN paper; decisions 1–4 in their original form.
- Schulman et al., "Proximal Policy Optimization Algorithms" (2017) — the robust default algorithm of decision 2.
- Haarnoja et al., "Soft Actor-Critic" (2018) — maximum-entropy RL and principled exploration (decisions 2 and 4).
- Schrittwieser et al., "Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model" (MuZero, 2020) — model-based RL at the frontier.
- Ng, Harada, Russell, "Policy Invariance Under Reward Transformations" (1999) — the potential-based shaping theorem behind decision 3.
- Ouyang et al., "Training language models to follow instructions with human feedback" (InstructGPT, 2022) — the RLHF template for the first case study.
- DeepSeek-AI, "DeepSeek-R1" (2025) — GRPO and verifiable rewards at scale.
- Within this series, start from the [unified introduction](/blog/machine-learning/reinforcement-learning/what-is-reinforcement-learning), and use the decision map above to jump to the post for the decision in front of you.

If you take one thing from seventy posts, take this: the algorithm is the part you can look up. The framework — knowing *which* algorithm, *what* reward, *how* to evaluate, and *when* to roll back — is the part that ships systems. That framework is the playbook.
