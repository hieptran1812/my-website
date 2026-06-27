---
title: "RL for Game Playing: From Atari to AlphaGo to OpenAI Five"
date: "2026-06-22"
publishDate: "2026-06-22"
description: "The story and mechanics of RL's greatest game-playing achievements: how DQN conquered Atari, AlphaGo and AlphaZero defeated the world champion, MuZero learned the rules, and OpenAI Five solved Dota 2 with PPO at massive scale."
tags:
  [
    "reinforcement-learning",
    "deep-learning",
    "q-learning",
    "model-based-rl",
    "multi-agent",
    "exploration",
    "machine-learning",
    "pytorch",
    "monte-carlo-tree-search",
  ]
category: "machine-learning"
subcategory: "Reinforcement Learning"
author: "Hiep Tran"
featured: true
readTime: 52
image: "/imgs/blogs/rl-for-game-playing-atari-to-alphago-1.png"
---

In 2013, a small team at a London startup called DeepMind hooked a neural network up to an Atari 2600 emulator and gave it a deal that sounds almost cruel: here are the raw pixels, here is the joystick, here is the score. Figure out the rest. No rules, no hand-coded features, no description of what a "paddle" or a "brick" or an "alien" even is. The same network architecture, the same hyperparameters, dropped into game after game. On Breakout, it spent the first hour flailing — losing ball after ball, scoring almost nothing. Then it learned to track the ball. Then it learned to keep the paddle under it. And then, somewhere past the millionth frame, it discovered something no one had told it to look for: dig a tunnel up the side of the brick wall, send the ball *behind* the wall, and let it ricochet along the top racking up points while you do nothing. A human strategy, found by a machine that had never been told the game had a strategy at all.

That single result — published as a 2015 *Nature* paper — kicked off a decade in which reinforcement learning systems climbed a ladder of progressively harder games, and at each rung a long-standing assumption about what machines could not do quietly fell over. Atari taught us that pixels-to-actions was learnable. Go, long held up as the game that needed human *intuition*, fell to AlphaGo in 2016. AlphaGo Zero then threw away the human games entirely and got *stronger*. MuZero threw away the rulebook itself and learned to plan inside a model it built from scratch. And then OpenAI Five and AlphaStar walked into the messiest games we have — Dota 2 and StarCraft II, with hidden information, hundreds of decisions per second, and time horizons measured in tens of thousands of steps — and beat the best humans on Earth.

This post is the engineering story behind that ladder, and more importantly the *mechanics*. We will work through why games became the benchmark of choice, exactly which two ideas made DQN stable when naive Q-learning with a neural network diverges, how Monte Carlo Tree Search turns a so-so policy network into a champion, what it actually means for MuZero to "plan in a learned model," and why Dota 2 is qualitatively — not just quantitatively — harder than Go. You will see runnable PyTorch and Stable-Baselines3 code, the loss functions written out, and honest numbers for compute, sample counts, and final performance. The milestone ladder we will climb is in the figure below, and we will visit every rung.

![Timeline of reinforcement learning game-playing milestones from the 2013 DQN Atari demonstration through Agent57 solving all 57 Atari games in 2022](/imgs/blogs/rl-for-game-playing-atari-to-alphago-1.png)

Throughout, keep the spine of this whole series in mind: an RL agent interacts with an environment, collects rewards, and updates a policy. Every system below is a different answer to two questions — *which objective do we optimize* and *how do we estimate the gradient or value* — under a different kind of pressure. The games just turn the pressure up.

## 1. Why games are the benchmark of choice

Before the algorithms, the meta-question: why did the field obsess over games for a decade? It is not because researchers wanted to win at Pong. Games are the *Drosophila* of RL — a controlled organism that exposes exactly the variables you want to study, while holding everything else fixed.

Concretely, games give you four properties that real-world RL desperately lacks. First, **well-defined rules and a simulator you can run faster than real time.** A robot that fails costs you a robot; a Dota 2 bot that fails costs you a few milliseconds and you reset. You can collect billions of frames. Second, **an unambiguous, verifiable reward.** The score *is* the objective, or close to it. There is no debate about whether the agent "did well" the way there is when you train a chatbot. Third, **a clean difficulty gradient.** You can pick games that isolate a single capability — reflexes, search, long-horizon planning, multi-agent coordination, dealing with hidden information — and study that capability in isolation. Fourth, and most underrated, **a calibrated human baseline.** We know how good the world champion is. "Superhuman" is a measurable, falsifiable claim in a game, in a way it almost never is in production.

The genius of the decade-long arc is that the games were chosen to climb a ladder of *qualitatively distinct* hardness, not just bigger numbers:

- **Atari** (2013–2015) tests **perception and reflex**. The state is raw pixels, the action space is tiny (up to 18 joystick actions), the horizon is short, and most games reward you densely. The challenge is mapping high-dimensional pixels to good actions — a representation-learning problem wearing an RL costume.
- **Chess** (solved by classical search decades earlier, then re-solved by AlphaZero) tests **search plus evaluation**. Perfect information, modest branching factor (~35), and a position you can evaluate. Deep Blue beat Kasparov in 1997 with brute-force search and a hand-tuned evaluation function.
- **Go** tests **intuition under a search explosion**. The branching factor is ~250 and the board is 19×19, so brute-force search is hopeless — the number of legal positions exceeds the atoms in the observable universe. Strong play seemed to require something humans called "intuition," and for twenty years Go was the standing rebuke to AI hype.
- **Dota 2 and StarCraft II** test **imperfect information, real-time control, long horizons, and multi-agent strategy all at once**. Fog of war hides the enemy. You make hundreds of decisions per second. A single game runs 20,000–80,000 steps. And you are coordinating (or competing) with other agents whose policies are themselves moving targets.

Each rung needed a genuinely new idea, which is why this is a *story* and not a single algorithm scaled up. Let us start at the bottom.

## 2. DQN and Atari: the two ideas that made deep Q-learning work

Recall ordinary Q-learning, the off-policy temporal-difference control algorithm covered in detail in [Q-learning, off-policy TD control](/blog/machine-learning/reinforcement-learning/q-learning-off-policy-td-control). We learn an action-value function $Q(s, a)$ estimating the expected discounted return of taking action $a$ in state $s$ and acting optimally thereafter. The update nudges $Q$ toward the **Bellman optimality target**:

$$
Q(s, a) \leftarrow Q(s, a) + \alpha \Big[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \Big]
$$

In a small grid world you store $Q$ as a table and this converges. But Atari has roughly $256^{84 \times 84 \times 4}$ possible states after preprocessing — you cannot tabulate that. So you replace the table with a neural network $Q_\theta(s, a)$, a function approximator, and turn the update into a regression problem: minimize the squared TD error

$$
\mathcal{L}(\theta) = \mathbb{E}_{(s,a,r,s')} \Big[ \big( r + \gamma \max_{a'} Q_\theta(s', a') - Q_\theta(s, a) \big)^2 \Big].
$$

Here is the problem, and it is the central drama of deep RL. **This naive combination diverges.** This is the [deadly triad](/blog/machine-learning/reinforcement-learning/the-deadly-triad-stability-in-deep-rl): function approximation, bootstrapping (the target depends on your own estimate), and off-policy learning, combined, can make the value estimates blow up to infinity. Two specific failure modes bite hard:

1. **Correlated samples.** Consecutive frames in a game are almost identical. If you train on them in order, your gradient steps are highly correlated, you overfit to the current stretch of gameplay, and the network catastrophically forgets what it learned five seconds ago. Stochastic gradient descent assumes roughly i.i.d. samples; sequential play violates that hard.
2. **A moving target.** The regression target $r + \gamma \max_{a'} Q_\theta(s', a')$ uses the *same* network $\theta$ you are updating. So every gradient step shifts the target you are chasing. It is like trying to hit a dartboard that lurches a foot to the left every time you throw. The feedback loop is unstable and the values oscillate or explode.

DQN's two famous innovations are precise antidotes to these two failures.

**Experience replay** fixes the correlation. Instead of learning from the live stream, you push every transition $(s, a, r, s', \text{done})$ into a large circular buffer — one million transitions in the original paper — and train on *random minibatches* sampled from it. Random sampling breaks the temporal correlation, recycles each transition many times (better sample efficiency), and smooths the data distribution. This buffer is covered deeply in [experience replay and offline data](/blog/machine-learning/reinforcement-learning/experience-replay-and-offline-data); for now the key insight is that it is what makes the gradients behave like supervised learning's i.i.d. assumption.

**A target network** fixes the moving target. You keep a *second*, frozen copy of the network, $Q_{\theta^-}$, and compute the TD target with it: $r + \gamma \max_{a'} Q_{\theta^-}(s', a')$. You update the online network $\theta$ every step but only copy $\theta \to \theta^-$ every 10,000 steps. Now the target stays still for thousands of updates at a time, the regression problem becomes locally stationary, and the divergence largely goes away. It is a beautifully cheap fix: one extra copy of the weights and a periodic sync.

Two more engineering details mattered enormously. **Convolutional preprocessing**: the raw 210×160 RGB Atari frame is downsampled to 84×84 grayscale, and **four consecutive frames are stacked** as the input. That stacking is doing something subtle and essential — a single frame tells you *where* the ball is but not *which way it is moving*. Four frames encode velocity and acceleration, restoring the Markov property to an otherwise partially observed environment. **Reward clipping** to $\{-1, 0, +1\}$ let one set of hyperparameters work across games with wildly different score scales. The full architecture is below.

![The DQN architecture showing four stacked frames passing through three convolutional layers and a fully connected layer to produce Q-values, stabilized by a replay buffer and a target network](/imgs/blogs/rl-for-game-playing-atari-to-alphago-2.png)

The 2015 *Nature* result: a single architecture, trained on 200 million frames per game (about 38 days of game experience, run faster than real time), reached **human-level or above on 29 of 49 games**, and superhuman on classics like Video Pinball, Boxing, and Breakout. The same network. No game-specific tuning. That generality was the shock.

Here is the core training loop in PyTorch, written so you can read every piece. This is the algorithm, not pseudocode.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
import random

class QNetwork(nn.Module):
    def __init__(self, n_actions):
        super().__init__()
        # Input: (batch, 4, 84, 84) -- four stacked grayscale frames
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512), nn.ReLU(),
            nn.Linear(512, n_actions),
        )

    def forward(self, x):
        x = x / 255.0                     # normalize pixel ints to [0, 1]
        x = self.conv(x)
        x = x.flatten(start_dim=1)
        return self.fc(x)                 # one Q-value per action

class ReplayBuffer:
    def __init__(self, capacity=1_000_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, s, a, r, s2, done):
        self.buffer.append((s, a, r, s2, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s2, done = zip(*batch)
        return (torch.tensor(np.array(s), dtype=torch.float32),
                torch.tensor(a, dtype=torch.int64),
                torch.tensor(r, dtype=torch.float32),
                torch.tensor(np.array(s2), dtype=torch.float32),
                torch.tensor(done, dtype=torch.float32))

online = QNetwork(n_actions=4)
target = QNetwork(n_actions=4)
target.load_state_dict(online.state_dict())   # start identical
opt = torch.optim.Adam(online.parameters(), lr=1e-4)
buf = ReplayBuffer()
gamma = 0.99

def train_step(batch_size=32):
    s, a, r, s2, done = buf.sample(batch_size)
    # Q(s, a) for the actions actually taken
    q = online(s).gather(1, a.unsqueeze(1)).squeeze(1)
    with torch.no_grad():                       # target net is frozen here
        next_q = target(s2).max(dim=1).values   # max_a' Q_target(s', a')
        td_target = r + gamma * next_q * (1 - done)
    loss = F.smooth_l1_loss(q, td_target)       # Huber loss, robust to outliers
    opt.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(online.parameters(), 10.0)
    opt.step()
    return loss.item()

# Every C=10_000 environment steps:
#   target.load_state_dict(online.state_dict())
```

Two lines carry the whole stability story: `with torch.no_grad()` wrapping the *target* network's evaluation (the frozen target), and the periodic `target.load_state_dict(...)` sync (the slow update). Remove either and watch the loss diverge — this is the single most instructive experiment a newcomer to deep RL can run.

#### Worked example: Breakout from random to superhuman

Start a fresh DQN on `ALE/Breakout-v5`. At step 0 the policy is $\epsilon$-greedy with $\epsilon = 1.0$ (pure random) and the average score per episode is about **1.2 points** — the agent occasionally hits the ball by luck. By 2 million frames $\epsilon$ has decayed toward 0.1, the network reliably tracks the ball, and the score climbs to roughly **15–20**. By 10 million frames it has discovered the tunneling strategy and scores **300+**. The original paper, trained to 200M frames, reports a mean of **401.2** versus a human expert's 31.8 — about a **13× superhuman** margin. The learning curve is not smooth: there are long plateaus where the score sits flat for a million frames, then a discontinuous jump when the network discovers a new strategy. Those jumps correspond to the agent *restructuring its value estimates*, not gradual polishing — a fingerprint of RL that you rarely see in supervised learning.

Where DQN failed is just as instructive. On **Montezuma's Revenge**, DQN scored **0**. The game requires climbing ladders, collecting a key, and crossing rooms before *any* reward arrives — a sparse-reward, long-horizon exploration problem. With $\epsilon$-greedy exploration, the probability of randomly executing the exact 100-step sequence that earns the first point is astronomically small, so the agent never sees a reward, never gets a learning signal, and stays at zero forever. We return to this hard exploration problem in Section 9; it is the thread that runs all the way to Agent57.

## 3. Rainbow and its successors: stacking the improvements on Atari

DQN was the spark, but the next three years produced a flurry of independent improvements, each fixing a specific weakness. **Rainbow** (Hessel et al., 2018) is the famous result that combined six of them into one agent and showed they are largely complementary. It is worth knowing what each lever does, because the same ideas reappear across modern RL.

| Improvement | Fixes | Mechanism |
|---|---|---|
| Double DQN (DDQN) | Overestimation bias | Select the action with the online net, evaluate it with the target net |
| Dueling network | Wasted capacity | Split into a state-value $V(s)$ and an advantage $A(s,a)$ stream |
| Prioritized replay (PER) | Uniform sampling wastes effort | Sample transitions in proportion to TD error magnitude |
| Multi-step returns | Slow reward propagation | Use an $n$-step target instead of one-step bootstrap |
| Distributional (C51) | A point estimate loses information | Predict the full return *distribution*, not just its mean |
| Noisy nets | Crude $\epsilon$-greedy exploration | Learnable parametric noise in the weights drives exploration |

The overestimation fix deserves a sentence of theory because it is so clean. Standard Q-learning's $\max_{a'} Q(s', a')$ uses the *same* values both to choose and to evaluate the best next action. When values are noisy, the max systematically picks the actions whose noise happened to be positive, biasing the target upward — a kind of winner's curse. **Double DQN** decouples the two: the online network picks the argmax action, the target network supplies its value. In one line:

$$
\text{target} = r + \gamma \, Q_{\theta^-}\!\Big(s', \arg\max_{a'} Q_\theta(s', a')\Big).
$$

These six improvements and how they interact are dissected in [DQN improvements: double, dueling, PER](/blog/machine-learning/reinforcement-learning/dqn-improvements-double-dueling-per), the distributional view in [distributional RL: C51, QR-DQN, IQN](/blog/machine-learning/reinforcement-learning/distributional-rl-c51-qr-dqn-iqn), and the full combination in [Rainbow DQN: combining six improvements](/blog/machine-learning/reinforcement-learning/rainbow-dqn-combining-six-improvements). The headline number: Rainbow reached a **median human-normalized score of ~223%** across the 57-game Atari suite, versus DQN's ~79%, and reached DQN's final performance in roughly **7× fewer frames**. The ablations matter as much as the total — removing prioritized replay or multi-step returns hurt the most, while removing double-Q or dueling hurt least, telling you where the leverage actually lived.

The story did not end with Rainbow. **Agent57** (Badia et al., 2020) was the first single agent to beat the human baseline on *all 57* Atari games — including Montezuma's Revenge and Pitfall, the sparse-reward holdouts. It did so by training a *family* of policies ranging from purely exploratory to purely exploitative, and learning a meta-controller (a bandit) to choose which policy to run, plus a separate intrinsic-reward stream for novelty. **MuZero**, which we cover in depth in Section 6, then matched or exceeded the state of the art on Atari *and* board games with one algorithm and no hand-coded rules — the unification that the whole field had been building toward.

In practice you rarely implement Rainbow from scratch. Stable-Baselines3 gives you a tuned DQN and you compose the improvements via config and wrappers:

```python
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

# 4-frame stacking, grayscale, and Atari preprocessing handled by the wrappers
env = make_atari_env("BreakoutNoFrameskip-v4", n_envs=8, seed=0)
env = VecFrameStack(env, n_stack=4)

model = DQN(
    "CnnPolicy", env,
    learning_rate=1e-4,
    buffer_size=100_000,        # replay buffer capacity
    learning_starts=100_000,    # collect random data before learning
    batch_size=32,
    gamma=0.99,
    target_update_interval=1000,
    train_freq=4,               # one gradient step every 4 env steps
    exploration_fraction=0.1,   # epsilon decay schedule
    exploration_final_eps=0.01,
    verbose=1,
)
model.learn(total_timesteps=10_000_000)
```

## 4. AlphaGo: when search met intuition

Atari is reflexes. Go is the opposite end of the spectrum — a game so combinatorially vast that for decades it was the canonical example of a task requiring human intuition. The branching factor is around 250 and games last ~150 moves, so a full game tree has on the order of $250^{150} \approx 10^{360}$ leaves. Brute-force search, the technique that cracked chess, is hopeless here.

AlphaGo's insight was to make search *tractable* by replacing two expensive things — exhaustive expansion and random rollouts — with learned neural networks that supply good priors and good evaluations. The training pipeline had several stages, shown below.

![The AlphaGo training pipeline showing expert games bootstrapping a supervised policy, self-play producing an RL policy and value network, and Monte Carlo Tree Search combining policy priors with value estimates](/imgs/blogs/rl-for-game-playing-atari-to-alphago-3.png)

**Stage 1, supervised policy.** Train a 13-layer convolutional network to predict the move a human expert played, from a database of 30 million positions from strong amateur games. This *supervised learning* policy $p_\sigma$ reached **57% top-1 move-prediction accuracy** — meaning more than half the time it guessed the exact move a strong human would play. That is the "intuition" prior: a fast suggestion of plausible moves.

**Stage 2, self-play RL policy.** Take $p_\sigma$ and improve it with policy gradient self-play — it plays against past versions of itself and is rewarded for winning. This RL policy $p_\rho$ won roughly **80% of games against the supervised policy.** This is the [policy gradient theorem](/blog/machine-learning/reinforcement-learning/the-policy-gradient-theorem) at work: the gradient of expected reward with respect to policy parameters, estimated from self-play game outcomes.

**Stage 3, value network.** Use the self-play games to train a *value network* $v_\theta(s)$ that predicts the probability the current player wins from position $s$. This is the crucial piece. In classical MCTS you estimate a leaf's value by playing the game out to the end with a fast random "rollout" policy — slow and noisy. The value network replaces those rollouts with a single forward pass that says "this position is worth 0.62 to Black." That is the key innovation: **a learned value function makes deep search affordable.**

Now the search itself. **Monte Carlo Tree Search (MCTS)** builds a search tree incrementally, balancing exploitation of good moves against exploration of uncertain ones. At each node it selects the child maximizing a score:

$$
a^* = \arg\max_a \Big( Q(s, a) + c \, P(s, a) \frac{\sqrt{\sum_b N(s, b)}}{1 + N(s, a)} \Big)
$$

where $Q(s,a)$ is the mean value of simulations through that edge, $N(s,a)$ is the visit count, $P(s,a)$ is the policy network's prior probability, and $c$ controls exploration. The first term exploits, the second term — the **PUCT** bonus — pulls search toward moves the policy likes but that have not been tried much, and decays as a move gets visited. AlphaGo evaluated leaves with a blend of the value network and a fast rollout. After thousands of simulations, it plays the move with the **highest visit count** (the most robustly-search-validated move), not the highest single value — a subtle but important choice that makes the move selection robust to value-estimate noise.

#### Worked example: Move 37, game 2 vs Lee Sedol

In March 2016 AlphaGo played Lee Sedol, an 18-time world champion, and won the five-game match **4–1**. The defining moment was Move 37 in game 2: a "shoulder hit" on the fifth line that violated centuries of Go orthodoxy (such moves are conventionally played on the third or fourth line). Commentators initially thought it was a mistake. The probability that a *human* would play that move, according to AlphaGo's own policy network, was about **1 in 10,000**. But the *search* — guided by the value network — found that the move's long-term value was high, and overrode the low prior. Lee Sedol left the room to compose himself; the move proved decisive forty moves later. This is the mechanical signature of AlphaGo's design: the policy network proposes, but MCTS plus the value network *disposes*, and the search can discover moves that no human prior would suggest. (Lee Sedol's lone win, game 4's Move 78 "wedge," exploited a blind spot in AlphaGo's value estimates — a reminder that the system was strong, not infallible.)

The compute behind the match was substantial: the distributed version of AlphaGo used **1,920 CPUs and 280 GPUs**. That is the price of running thousands of network evaluations inside the search for every single move under tournament time controls.

## 5. AlphaGo Zero and AlphaZero: throwing away the human

AlphaGo had a philosophical wart: it was bootstrapped on 30 million human moves. Was the human data *necessary*, or just a convenient warm start? AlphaGo Zero (Silver et al., 2017) answered the question by deleting it.

AlphaGo Zero starts from **random weights** and learns purely by self-play, with **no human games at all**. It also collapses the separate policy and value networks into **one residual network with two heads** that outputs both a move-probability vector $\mathbf{p}$ and a scalar value $v$ from a single shared trunk. The before/after of this redesign is striking.

![Comparison of AlphaGo and AlphaGo Zero showing the move from three weeks of training on human games and separate networks to three days of training from random play with a single unified network winning one hundred games to zero](/imgs/blogs/rl-for-game-playing-atari-to-alphago-4.png)

The conceptual leap — and this is one of the most beautiful ideas in all of RL — is to view **MCTS as a policy improvement operator.** Here is the loop. The network $f_\theta$ produces a "raw" policy $\mathbf{p}$. You then run MCTS using $\mathbf{p}$ as the prior; the search, by looking ahead, produces an *improved* policy $\boldsymbol{\pi}$ (the normalized visit counts at the root). Because search looks ahead, $\boldsymbol{\pi}$ is provably at least as good as $\mathbf{p}$ in the searched positions. You then train the network to *imitate its own improved search*: push $\mathbf{p}$ toward $\boldsymbol{\pi}$ and push $v$ toward the eventual game outcome $z$. The single loss is elegantly simple:

$$
\mathcal{L} = (z - v)^2 - \boldsymbol{\pi}^\top \log \mathbf{p} + c \lVert \theta \rVert^2
$$

The first term is value regression (mean-squared error toward the game result), the second is a cross-entropy that distills the search-improved policy back into the network, and the third is weight decay. That is the entire objective. Iterate — better network gives better search gives a better training target gives a better network — and you get a self-reinforcing spiral that climbs from random play to superhuman in **three days on 4 TPUs**, after about **4.9 million self-play games**. AlphaGo Zero beat the Lee Sedol version of AlphaGo **100 games to 0.**

Here is the policy-improvement loop in PyTorch-flavored code, simplified to show the mechanism rather than a production MCTS:

```python
import torch
import torch.nn.functional as F

def alphazero_loss(net, states, search_policies, outcomes, c_l2=1e-4):
    """One AlphaZero training step on a batch of self-play positions.
    states:          board tensors                (B, C, H, W)
    search_policies: MCTS visit-count targets pi  (B, n_moves)  -- improved policy
    outcomes:        game results z in {-1, 0, 1} (B,)          -- from this player's view
    """
    logits, value = net(states)              # net has policy head + value head
    log_p = F.log_softmax(logits, dim=1)
    # Distill the search-improved policy into the network's raw policy:
    policy_loss = -(search_policies * log_p).sum(dim=1).mean()
    # Regress the value head toward the actual game outcome:
    value_loss = F.mse_loss(value.squeeze(-1), outcomes)
    l2 = sum((p ** 2).sum() for p in net.parameters())
    return value_loss + policy_loss + c_l2 * l2

def mcts_policy_improvement(net, root_state, n_sim=800, c_puct=1.5):
    """Sketch: run n_sim simulations, return normalized root visit counts as pi.
    Each simulation: select via PUCT down to a leaf, expand using the net's
    policy prior, evaluate the leaf with the net's value head, then back up.
    The visit-count distribution at the root is the improved policy pi.
    """
    root = expand(root_state, net)
    for _ in range(n_sim):
        leaf, path = select_to_leaf(root, c_puct)   # PUCT selection
        value = expand_and_evaluate(leaf, net)      # one network forward pass
        backup(path, value)                         # update Q and N up the path
    visits = torch.tensor([child.N for child in root.children])
    return visits / visits.sum()                    # pi: the search-improved policy
```

**AlphaZero** (Silver et al., 2018) then made the obvious-in-hindsight generalization: the algorithm has nothing Go-specific in it. Point it at chess and shogi and it works unchanged. Starting from random, AlphaZero surpassed Stockfish (the strongest classical chess engine of the time) in **about 4 hours** of training, and reached superhuman shogi in **2 hours**. It did this while searching only ~80,000 positions per second versus Stockfish's ~70 million — a thousand-fold *less* search, compensated by a far better learned evaluation and a far better search-guidance prior. That trade — replace brute-force breadth with learned judgment — is the whole thesis of the AlphaGo lineage.

## 6. MuZero: planning without being told the rules

AlphaZero still had one crutch: it was *given the rules of the game.* To run MCTS, you must be able to ask "if I play this move, what is the resulting board?" — that requires a perfect simulator. Chess and Go hand you one. Atari does not give you a clean next-state function, and the real world certainly does not. **MuZero** (Schrijver et al. / Schrittwieser et al., 2020) removed this crutch: it *learns* a model good enough to plan with, without ever modeling the actual game dynamics.

The trick is that MuZero does not try to predict the next *observation* (predicting raw pixels is wasteful and hard). It only learns to predict the things that matter for *planning*: reward, value, and policy. It has three learned functions, and the planning loop wires them together as shown below.

![The MuZero planning loop showing an observation encoded by the representation function into a latent state, expanded by the dynamics function inside Monte Carlo Tree Search, and evaluated by the prediction function for policy and value](/imgs/blogs/rl-for-game-playing-atari-to-alphago-5.png)

- **Representation** $h$: encodes the observation (or stack of frames) into an abstract latent state $s^0 = h(o)$. This is the root of the search tree.
- **Dynamics** $g$: a learned transition model. Given a latent state $s^k$ and an action $a$, it predicts the next latent state and an immediate reward: $(s^{k+1}, r^{k+1}) = g(s^k, a)$. This is what replaces the simulator — MuZero "imagines" the consequences of actions entirely in latent space.
- **Prediction** $f$: from a latent state, predict a policy and a value: $(\mathbf{p}^k, v^k) = f(s^k)$. Same role as the AlphaZero head.

MCTS runs *inside* this learned model: the search expands nodes by rolling the learned dynamics function forward, never touching the real environment. The whole system is trained end-to-end so that, when unrolled, the predicted rewards match observed rewards, the predicted values match the search returns, and the predicted policies match the search visit counts. Crucially, **the latent states are not required to mean anything** — there is no reconstruction loss forcing $s^k$ to decode back to a board. They only have to be *useful for predicting reward, value, and policy.* This is what lets one algorithm cover both board games and pixel-based Atari with no domain-specific code.

The numbers: MuZero used an **800-simulation MCTS budget** per move on board games (50 on Atari), matched AlphaZero on Go, chess, and shogi, and set a new state of the art on the 57-game Atari suite — all with a learned model. The compute was formidable: training used **thousands of TPUs** running self-play and learner jobs in parallel. The lesson that carries forward into [model-based RL: learning world models](/blog/machine-learning/reinforcement-learning/model-based-rl-learning-world-models) and [MuZero: mastering games without rules](/blog/machine-learning/reinforcement-learning/muzero-mastering-games-without-rules) is profound: you do not need to model the world accurately, only the *consequences relevant to your decisions.*

| System | Needs a simulator? | Needs human data? | Domains covered |
|---|---|---|---|
| AlphaGo | Yes (game rules) | Yes (30M moves) | Go |
| AlphaGo Zero | Yes (game rules) | No | Go |
| AlphaZero | Yes (game rules) | No | Go, chess, shogi |
| MuZero | No (learned model) | No | Go, chess, shogi, Atari |

## 7. OpenAI Five: PPO at industrial scale on Dota 2

Board games are *perfect information* — you can see the whole board. Real games are not. Dota 2 is a 5-vs-5 real-time strategy game with **fog of war** (you cannot see enemies you have no vision of), a complex in-game economy (gold, items, experience), around **80,000 distinct hero-ability-item combinations**, and matches that run roughly 45 minutes. At the bot's decision rate of about 7.5 actions per second, a single game is on the order of **20,000 decisions per agent** — and the reward (winning) arrives only at the very end. This is the long-horizon, sparse, high-dimensional, multi-agent regime, and it is qualitatively harder than Go.

The remarkable thing about OpenAI Five (2018–2019) is that it did *not* invent a new algorithm. It used **Proximal Policy Optimization** — the same workhorse PPO covered in [Proximal Policy Optimization (PPO)](/blog/machine-learning/reinforcement-learning/proximal-policy-optimization-ppo) — and scaled it to an absurd degree. The bet was that brute-force scale plus a stable on-policy algorithm could substitute for clever search. It worked.

Let us recall the PPO objective, because the *stability* it provides is exactly why it survives at this scale. PPO maximizes a clipped surrogate that prevents the policy from changing too much in one update:

$$
\mathcal{L}^{\text{CLIP}}(\theta) = \mathbb{E}_t \Big[ \min\big( r_t(\theta) \hat{A}_t, \ \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\, \hat{A}_t \big) \Big]
$$

where $r_t(\theta) = \frac{\pi_\theta(a_t | s_t)}{\pi_{\theta_{\text{old}}}(a_t | s_t)}$ is the probability ratio between the new and old policies, $\hat{A}_t$ is the advantage estimate, and the clip keeps the ratio inside $[1-\epsilon, 1+\epsilon]$. The clip is the safety belt: it removes the incentive to make a huge policy update just because one batch of trajectories looked good, which is precisely the failure mode that destroys naive policy gradient at scale. When you are running thousands of machines, a single destabilizing update can waste hours of GPU time, so this conservatism pays for itself many times over.

The architecture is, by modern standards, almost shockingly simple: a large **LSTM core (4096 units)** processes a per-step observation of roughly 20,000 numbers (engineered features, not pixels — OpenAI Five read the game state through the bot API, not the screen), and each of the five heroes is controlled by a *separate* copy of the network with the same weights. The structure is below.

![The OpenAI Five architecture showing per-agent observations feeding an LSTM core, five parallel hero policy heads, a shared team reward, and a PPO update running across roughly 180,000 CPU cores](/imgs/blogs/rl-for-game-playing-atari-to-alphago-7.png)

Two design choices made the multi-agent coordination work. First, **team reward via "team spirit."** Each hero's reward was a mix of its own performance and the team's, controlled by a coefficient $\tau$ that was **annealed from 0 to 1** over training. Early on, agents optimized selfishly (easier to learn basic mechanics); later, $\tau \to 1$ forced fully cooperative play (sacrifice yourself for the team's win). Second, **sheer scale of experience.** OpenAI Five trained on the equivalent of roughly **180 years of self-play per day**, running on about **180,000 CPU cores** for rollouts plus hundreds of GPUs for the optimizer, across roughly **10 months** of continuous training (the project total spanned ~18 months including iterations). The system consumed millions of years of game experience in total — orders of magnitude more than any human could play in a thousand lifetimes.

The payoff: in April 2019, OpenAI Five beat **OG**, the reigning Dota 2 world champions (winners of The International 2018), **2–0** in a best-of-three. It then played 7,000+ games against the public and won over **99%** of them. The systems-design challenges of collecting rollouts at this scale — async rollout workers, experience queues, stale-policy handling — connect directly to the production patterns in [distributed RLHF system design](/blog/machine-learning/reinforcement-learning/distributed-rlhf-system-design) and [multi-node RL training with Ray and OpenRLHF](/blog/machine-learning/reinforcement-learning/multi-node-rl-training-ray-openrlhf); OpenAI Five was, in a real sense, a dress rehearsal for large-scale RLHF.

A note on robustness that the marketing skipped: a public experiment let humans play *against* Five with arbitrary strategies, and creative players found exploits — for example, surrounding and "training" the bots into bad positions. The agents were superhuman *within the distribution they had trained on* and brittle outside it. That generalization gap is a recurring theme in game-playing RL and the honest counterweight to the highlight reel.

Here is a Stable-Baselines3 PPO configuration that mirrors the *shape* of the OpenAI Five setup — a recurrent policy, many parallel environments, and the key PPO hyperparameters — at a scale you can actually run:

```python
from sb3_contrib import RecurrentPPO        # LSTM policy lives in sb3-contrib
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env

# Massively parallel rollout collection -- OpenAI Five used ~180k cores;
# here we use as many subprocesses as the machine allows.
env = make_vec_env("CartPole-v1", n_envs=32, vec_env_cls=SubprocVecEnv)

model = RecurrentPPO(
    "MlpLstmPolicy", env,
    n_steps=256,            # rollout length per env before each update
    batch_size=256,
    n_epochs=4,             # passes over each batch of rollouts
    gamma=0.999,            # long horizon -> high discount
    gae_lambda=0.95,        # GAE advantage smoothing
    clip_range=0.2,         # the PPO epsilon -- the safety belt
    ent_coef=0.01,          # entropy bonus to keep exploring
    learning_rate=2.5e-4,
    verbose=1,
)
model.learn(total_timesteps=2_000_000)
```

The `clip_range=0.2` is the exact knob that keeps a 180,000-core training run from blowing itself up on a single bad batch. Scale changes everything *except* the algorithm's core; the core is why it survives the scale.

## 8. AlphaStar: StarCraft II and the league

StarCraft II adds a different flavor of difficulty on top of Dota's. You manage an *economy* (mine minerals, build structures, research upgrades), you command *many units simultaneously* with fine-grained micro-management, you operate under fog of war, and the action space is enormous — roughly $10^{26}$ possible actions at each step when you account for selecting any subset of units and targeting anywhere on the map. Above all, StarCraft has a brutal **non-transitivity** problem: strategy A beats B, B beats C, but C beats A. There is no single "best" policy, so naive self-play just chases its own tail, cycling between counters forever.

AlphaStar (Vinyals et al., 2019) attacked this with two ideas. The first is the **architecture**: a deep network that combines a **transformer** to process the set of units, a **deep LSTM core** to carry memory across the long game, and an **autoregressive, pointer-network action head** that emits a structured action (what to do, which units, where, when) one component at a time conditioned on the previous components. It also began with **supervised pre-training on ~970,000 human replays**, which gave it a sane prior over the vast action space — pure RL from random in $10^{26}$ actions would never get off the ground.

The second and more important idea is **league training**, the answer to non-transitivity. Instead of one agent playing copies of itself, AlphaStar maintained a whole *population* of agents with different roles. Some were **main agents** trying to be generally strong; some were **main exploiters** that specifically hunted the current main agents' weaknesses; some were **league exploiters** that found weaknesses in the entire league. New agents were periodically frozen and added to the pool, so the main agents always faced a *diverse, non-cycling* set of opponents and could not collapse into a single exploitable strategy. This is a population-based, multi-agent solution to a game-theoretic problem — it pushes the league toward something like a Nash equilibrium over strategies, a connection developed in [Nash equilibria and game theory for MARL](/blog/machine-learning/reinforcement-learning/nash-equilibria-and-game-theory-for-marl) and the broader [multi-agent RL fundamentals](/blog/machine-learning/reinforcement-learning/multi-agent-rl-fundamentals).

The result: AlphaStar reached **Grandmaster** rank on the official StarCraft II ladder — above 99.8% of ranked human players — playing all three races. The compute was again enormous: each agent in the league trained for the equivalent of up to **200 years of real-time StarCraft play**, across **44 days** of wall-clock training on TPUs. AlphaStar's later versions were also constrained to roughly human-like action rates and camera limits, to answer the fair criticism that earlier bots won partly through inhuman click speed rather than strategy alone.

## 9. Exploration in hard games: the Montezuma problem

We left a loose end in Section 2: DQN scored **zero** on Montezuma's Revenge because $\epsilon$-greedy exploration never stumbles onto the first reward. This sparse-reward, long-horizon exploration problem is arguably the deepest unsolved-then-solved thread in the whole game-playing story, and it deserves its own treatment. The general principle — that you must sometimes act to *gain information* rather than to *exploit known reward* — is the subject of [exploration vs exploitation: the core tension](/blog/machine-learning/reinforcement-learning/exploration-vs-exploitation-the-core-tension).

The theoretical anchor is **count-based exploration.** If you knew how many times you had visited each state $s$, denoted $N(s)$, you could add an exploration bonus that rewards novelty:

$$
r^+(s) = r(s) + \frac{\beta}{\sqrt{N(s)}}
$$

States you have rarely visited get a big bonus, drawing the agent toward the unexplored. This is provably near-optimal in tabular settings. The problem in Atari is that you never visit the exact same pixel-state twice, so $N(s)$ is always 1 — the count is useless. The clever modern methods are all ways to *approximate* a meaningful novelty signal in high-dimensional state spaces.

**ICM (Intrinsic Curiosity Module)** rewards the agent for states where a learned forward model is *surprised* — where its prediction of the next state's features is wrong. The intuition is that prediction error is high in novel situations and low in familiar ones, so curiosity becomes a dense, self-generated reward. It crucially predicts in a *feature* space (learned by an inverse dynamics model) rather than raw pixels, so it ignores unpredictable-but-irrelevant noise like flickering backgrounds — the infamous "noisy TV problem" where an agent gets addicted to a source of pure randomness.

**RND (Random Network Distillation)** is even simpler and remarkably effective. Initialize a *fixed, random* target network $f$. Train a second "predictor" network $\hat f$ to match $f$'s output on every state the agent visits. The prediction error $\lVert \hat f(s) - f(s) \rVert^2$ is the novelty bonus: it is high on states you have seen rarely (the predictor has not learned them yet) and low on familiar ones. Because the target is fixed and deterministic, RND sidesteps the noisy-TV trap that plagues forward-model methods. RND was the first method to *beat the human baseline on Montezuma's Revenge* and was a key ingredient in later sparse-reward agents.

```python
import torch
import torch.nn as nn

class RND(nn.Module):
    """Random Network Distillation intrinsic reward.
    A fixed random target; a trained predictor that chases it.
    Prediction error = novelty bonus."""
    def __init__(self, obs_dim, feat_dim=128):
        super().__init__()
        self.target = nn.Sequential(           # FIXED, never trained
            nn.Linear(obs_dim, 256), nn.ReLU(), nn.Linear(256, feat_dim))
        self.predictor = nn.Sequential(        # trained to match target
            nn.Linear(obs_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, feat_dim))
        for p in self.target.parameters():
            p.requires_grad = False            # freeze the target

    def intrinsic_reward(self, obs):
        with torch.no_grad():
            t = self.target(obs)
        pred = self.predictor(obs)
        # Per-state squared error = the novelty signal:
        return (pred - t).pow(2).sum(dim=1)

    def distillation_loss(self, obs):
        with torch.no_grad():
            t = self.target(obs)
        return (self.predictor(obs) - t).pow(2).sum(dim=1).mean()
```

**Go-Explore** (Ecoffet et al., 2021) took a radically different and pragmatic route. Its insight: the reason agents fail at hard exploration is *detachment* (forgetting how to get back to a promising frontier) and *derailment* (random exploration knocking you off course before you reach the frontier). Go-Explore fixes both by **remembering promising states, resetting directly back to them, and exploring onward from there** — first reach a good state reliably, *then* explore from it. On Montezuma's Revenge it scored over **43,000 points** (versus a human expert's ~4,700), and it was the first to solve Pitfall, the other sparse-reward holdout. The "reset to a saved state" trick assumes a resettable simulator, which games provide and the real world does not — a caveat worth keeping in mind.

And finally, **Agent57** unified the threads: it ran a population of policies spanning exploratory to exploitative, used an episodic novelty module plus RND-style lifelong novelty for its intrinsic reward, and learned (via a bandit meta-controller) how much to explore on each game. It was the first agent to beat the human baseline on **all 57** Atari games — closing the book that DQN opened seven years earlier.

#### Worked example: the exploration bonus that breaks the zero

Picture a DQN on Montezuma's Revenge with and without an RND bonus. **Without** it, over 200 million frames the extrinsic reward seen is exactly 0 in the overwhelming majority of episodes — the agent dies before reaching the key — and the final score is **0**. The value function correctly learns that every state is worth zero, because from the agent's experience, it is. **With** an RND intrinsic reward of, say, $\beta = 0.5$ scaling the normalized prediction error, the *total* reward in early training is dominated by novelty: the agent is paid to reach new rooms, descend new ladders, and pick up the key simply because those states are unfamiliar. Once it accidentally collects the key, the *extrinsic* +100 reward finally appears, and now there is a real signal to bootstrap from. RND-equipped agents reach scores in the **8,000–11,000** range on Montezuma — from literally zero. The bonus does not solve the game; it solves the *cold-start* problem of getting any reward signal at all, and that is the entire ballgame in sparse-reward domains.

## 10. The hardware and compute dimension

There is no honest telling of this story without the compute, because the second half of the arc is as much a systems-engineering achievement as an algorithmic one. The compute did not just make training faster — at this scale it became a first-class design constraint that shaped the algorithms.

| System | Compute | Wall-clock | Experience consumed |
|---|---|---|---|
| DQN (2015) | 1 GPU | ~7–10 days/game | 200M frames/game |
| AlphaGo (2016) | 1,920 CPU + 280 GPU (match) | months (training) | tens of millions of games |
| AlphaGo Zero (2017) | 4 TPUs | 3 days | 4.9M self-play games |
| AlphaZero (2018) | thousands of TPUs (gen) | hours to superhuman | tens of millions of games |
| MuZero (2019) | thousands of TPUs | ~12 hours (board) | millions of games |
| OpenAI Five (2019) | ~180,000 CPU + ~1,500 GPU | ~10 months | ~years/day of play |
| AlphaStar (2019) | TPU pods, full league | 44 days | up to 200 years/agent |

Read this table as a story about a **trade between human knowledge and compute.** AlphaGo Zero used *less* wall-clock than AlphaGo (3 days vs 3 weeks) and *no* human data, but it ran on TPUs that were doing far more computation per second. AlphaZero needed only hours to beat the best chess engine ever written — but those hours ran across thousands of TPUs generating self-play games in parallel. The general law that emerges, consistent with the broader [scaling laws](/blog/machine-learning/scaling-laws/) literature, is that **for self-play game RL, performance scales remarkably smoothly with the total amount of experience generated, and experience scales with parallel compute.** OpenAI Five is the cleanest example: its capability curve climbed predictably as they threw more cores at rollout generation, with surprisingly few algorithmic changes over the 10-month run.

There is a sample-efficiency dimension hiding behind the raw compute, too. DQN needed 200M frames to learn one Atari game — about 38 days of real-time play for a game a human grasps in minutes. That gap between human and machine sample efficiency is *enormous*, and closing it (rather than just throwing more compute at it) is what modern model-based methods like MuZero, and the world-model approaches in [world models: Dreamer and PlaNet](/blog/machine-learning/reinforcement-learning/world-models-dreamer-planet), are really chasing. The lesson for any practitioner: these results are landmarks of what is *possible*, not templates you can run on a single GPU. The compute is the silent co-author of every headline below.

## 11. Case studies

We have woven results throughout, but it is worth pinning down four clean, citable case studies side by side — each one isolates a different capability the ladder demanded.

**DQN on Atari (2015).** One architecture, 49 games, no per-game tuning. Superhuman on 29 of 49, with a mean of 401 on Breakout (13× the human expert) and famously *zero* on Montezuma's Revenge. The capability proven: **end-to-end pixels-to-actions control** with two stabilizers (replay, target net). The honest caveat: catastrophic failure on sparse rewards, and 200M-frame sample inefficiency.

**AlphaGo vs Lee Sedol (2016).** A 4–1 match win over an 18-time world champion in a game that was the standing symbol of un-automatable intuition. The capability proven: **deep search guided by learned value and policy networks** can beat human intuition, and can even *discover* moves (Move 37, ~1-in-10,000 human prior) that humans would never play. The caveat: it still needed 30M human games to bootstrap, and game 4's loss showed exploitable value blind spots.

**OpenAI Five vs OG (2019).** A 2–0 best-of-three win over the reigning Dota 2 world champions, then a 99%+ win rate over thousands of public games. The capability proven: **plain PPO, scaled to ~180,000 cores, solves long-horizon, partially observed, multi-agent strategy** without search or a new algorithm. The caveat: brittleness to out-of-distribution human strategies — superhuman in-distribution, exploitable outside it.

**AlphaStar at Grandmaster (2019).** Above 99.8% of ranked human StarCraft II players, all three races, under increasingly human-like constraints. The capability proven: **league training resolves strategic non-transitivity** (rock-paper-scissors strategy cycles) where naive self-play collapses, via a population of main agents and exploiters. The caveat: the league and 200-years-per-agent compute were heroic, and early-version action-rate advantages drew fair criticism.

Put together, these four are not four versions of one result. They are four distinct *kinds* of hardness — perception, search-vs-intuition, scale-and-coordination, and strategic non-transitivity — each cracked by a genuinely different idea. That is why the decade mattered.

## 12. When to use which method (and when not to)

Strip away the celebrity and the practical question is: given a new sequential-decision problem with a game-like structure, which family do you reach for? The decision tree below is the honest answer, and the rest of this section walks it.

![A decision tree for selecting a game-playing reinforcement learning method based on whether the game has perfect or imperfect information and whether the rules are known](/imgs/blogs/rl-for-game-playing-atari-to-alphago-8.png)

**Perfect information, known rules, modest-to-large branching → AlphaZero-style MCTS + self-play.** If you can perfectly simulate the next state given an action (board games, many scheduling and combinatorial-optimization problems dressed as games), the MCTS-plus-learned-evaluation recipe is the gold standard. The search gives you a policy-improvement operator for free, and self-play gives you unlimited curriculum. Do *not* use this if you have no simulator.

**Perfect information but no clean simulator (pixels, unknown dynamics) → MuZero.** When you cannot write down the transition function — Atari, or a real system where you only get observations — MuZero's learned latent model lets you keep the planning machinery. It is more complex and compute-hungry to train, so only reach for it when planning genuinely helps (long horizons, sparse rewards) rather than reflex tasks.

**Atari-style pixel input, short horizon, reflex-dominated → DQN / Rainbow.** For dense-reward, short-horizon, discrete-action problems with high-dimensional observations, value-based methods are simple, sample-efficient relative to the alternatives, and battle-tested. Rainbow is the strong default; plain DQN is the teaching baseline. Do not over-engineer a search method for a game that is fundamentally about reflexes.

**Imperfect information, real-time, multi-agent, long horizon → large-scale on-policy PPO (plus league training if non-transitive).** When fog of war, continuous time, and other learning agents are in play, search becomes intractable and you fall back on a robust policy-gradient method scaled with parallelism. Add population-based / league training when strategies cycle non-transitively. The catch is the compute: this is the regime where you genuinely need hundreds of thousands of cores, and where most teams should ask whether the problem really requires it.

And the meta-rule that the whole series keeps returning to: **if you have a cheap, accurate simulator and a known model, plan — do not learn model-free from scratch.** Model-free deep RL is a tool of last resort for when you cannot search. Much of the practical guidance lives in [model-based vs model-free: when to use which](/blog/machine-learning/reinforcement-learning/model-based-vs-model-free-when-to-use-which). For a tabular problem with a known transition matrix, value iteration is faster, exact, and needs no GPUs at all — the celebrity methods above are answers to problems where that simple option is off the table.

The systems-level comparison of all seven landmark systems — game, human data, known model, key innovation — is consolidated in the matrix below.

![A comparison matrix of seven landmark game-playing systems across game type, human data requirements, model knowledge, and key innovation](/imgs/blogs/rl-for-game-playing-atari-to-alphago-6.png)

## When to use this (and when not to)

To make the trade-offs blunt and decisive, here is the short version a practitioner should internalize:

- **Use DQN/Rainbow** when your problem has discrete actions, a short horizon, and you mostly need good perception and reflexes. It is the simplest thing that works for pixel control.
- **Use AlphaZero** when you have a perfect simulator, perfect information, and search pays off (turn-based, combinatorial). Nothing beats MCTS + self-play here.
- **Use MuZero** when planning helps but you lack a simulator — and you have the compute budget to learn a model. Otherwise its complexity is not worth it.
- **Use large-scale PPO** when the game is real-time, partially observed, and multi-agent, and you genuinely have the parallel compute. Add league training only when you observe strategy cycling.
- **Do NOT use any of these** when value iteration or classical search on a known, small model would do. Do not reach for model-free RL when you can plan cheaply. And do not assume superhuman-in-distribution means robust — every one of these systems had exploitable blind spots outside its training distribution.

## 12b. Worked example: DQN on Breakout — the complete training recipe

Abstract guidance is useful, but the value of these systems lives in the exact numbers. So here is the full recipe a practitioner would actually run to reproduce the canonical DQN-on-Breakout result, with every hyperparameter that matters and the failure modes that ruined a thousand reimplementations.

Start with the environment. We use `Breakout-v5` from the Arcade Learning Environment. The raw frame is 210×160 RGB; we convert to grayscale and downsample to 84×84, then stack the four most recent frames so the network sees motion, not a still life. A single 84×84 image cannot tell you which way the ball is travelling — the four-frame stack restores the Markov property, the same trick we leaned on throughout this post. We apply a frame-skip of 4, meaning each chosen action repeats for four emulator frames and we only observe and act on every fourth. This is not laziness; it cuts the effective decision rate to something a value function can actually learn over, and it is why a "200M frame" run only involves ~50M agent decisions. Finally, we clip every reward to the set {−1, 0, +1}. This single line is the most important one in the whole recipe.

The network is small by modern standards: three convolutional layers — 32 filters of 8×8 at stride 4, then 64 filters of 4×4 at stride 2, then 64 filters of 3×3 at stride 1 — feeding a fully connected layer of 512 units, which fans out to four Q-values, one per action (FIRE, LEFT, RIGHT, NOOP). No batch norm, no residual connections, no attention; just the 2015 architecture, because it works and because reproducing the original result means matching the original network.

The replay buffer holds one million transitions. We do not start learning until it contains at least 50,000 transitions, so the first 50k steps are pure random exploration that fills the buffer with a diverse-enough base of experience. Train any earlier and you are fitting to a handful of near-identical early states, which sends the value estimates off a cliff before the buffer can rescue them. The discount factor is γ = 0.99. The optimizer is RMSprop with a learning rate of 0.00025 and a numerical-stability epsilon of 0.01 — note this is the optimizer epsilon, not the exploration epsilon, a confusion that has bitten many a reader. The target network is a frozen copy of the online network, refreshed every 10,000 steps; between refreshes it provides the stable bootstrap target that keeps the regression from chasing its own tail.

Exploration follows an ε-greedy schedule that decays linearly from 1.0 to 0.1 over the first one million steps, then holds at 0.1 for the bulk of training, with some implementations annealing further to 0.01 near the end to squeeze out the last few points. Each gradient step samples a minibatch of 32 transitions uniformly from the buffer. Because of frame-skip, the relationship between "training frames" and "real emulator frames" is roughly 1-to-4, so when papers quote 200M frames they mean ~50M optimizer-relevant steps — keep the units straight or your learning curves will look mysteriously slow.

What does this cost and what do you get? On the 2013-era single K80 the original work used, fifty million steps took on the order of seven days of wall-clock time. On a modern V100 the same run finishes in roughly four hours, a clean illustration of how much of the "RL is slow" reputation was really a hardware artifact. The learning curve climbs steadily and the average episode reward plateaus around 400 after about 25 million steps. The human expert baseline on Breakout is roughly 31, so trained DQN is on the order of 13× human — one of the cleaner superhuman margins in the whole Atari suite. One evaluation subtlety matters: the standard protocol resets each episode with a random number of no-op actions (up to 30) so the agent cannot memorize a single deterministic opening, which would inflate scores without reflecting real skill.

Now the bug to avoid, because it is the one nearly everyone hits. If you forget to clip rewards, the magnitude of the reward signal varies wildly across and within games — in Breakout, clearing a brick high in the wall is worth more than one low down, and a full clear can spike the raw return. Those large rewards dominate the gradient, the value targets swing across orders of magnitude, and RMSprop — tuned for the clipped scale — destabilizes. The fix is the one line above: clip to {−1, 0, +1}. You lose the ability to distinguish a big reward from a small one, but you gain a single consistent scale across all 49 games, which is exactly what let one set of hyperparameters work everywhere.

Two upgrades are worth folding in once the baseline runs. **Double DQN** decouples action selection from action evaluation: pick the maximizing action with the online network, but read its value from the target network. This removes the systematic overestimation that plain DQN's single max operator introduces — overestimation shrinks by roughly 15%, and on Breakout that translates to about +8 average reward, free of any other change. **C51**, the distributional upgrade, goes further: instead of predicting a single scalar Q-value, it models a categorical distribution over 51 discrete return atoms spanning −10 to +10. Learning the *shape* of the return distribution, not just its mean, gives the network a far richer signal, and on Breakout it buys roughly a 30% improvement in average reward with no other modification. Stack Double DQN, C51, and the rest of the Rainbow ingredients and you have the strong default this post keeps recommending for arcade-style games.

## 12c. Self-play beyond board games: the AlphaStar league in depth

Self-play won Go because Go is transitive: if A reliably beats B and B reliably beats C, then A beats C, and a single agent driving itself ever upward converges toward genuine mastery. Real-time strategy games break that assumption, and understanding why is the key to understanding why DeepMind built a *league* rather than just running self-play harder.

The problem is strategic non-transitivity. In StarCraft II — as in Dota 2, and indeed rock-paper-scissors — strategies cycle. An aggressive early rush beats a greedy economic build; the economic build beats a slow defensive turtle; the turtle beats the rush. There is no single dominant strategy, only a web of counters. Pure self-play against a single evolving opponent collapses into a Nash equilibrium that is locally stable but globally brittle: the agent becomes excellent at beating its current self and quietly *forgets* how to beat the strategies it left behind ten thousand games ago. Send it back against an early-training rush and it folds, because nothing in its recent experience reminded it that rushes exist. This is the "forgotten strategy" problem, and it is fatal in a cyclic game.

The League is the answer, and its structure is the interesting part. Rather than one agent, DeepMind maintained a *population* with distinct roles. **Main Agents** are the ones you ultimately ship — they train to beat everyone, the whole population, robustly. **Main Exploiters** train specifically to find and punish the weaknesses of the current Main Agents; their entire job is to be the adversary that surfaces a blind spot. **League Exploiters** are broader still: they hunt for weaknesses anywhere in the entire league, including in past versions and in other exploiters, ensuring no globally weak strategy survives unchallenged. The exploiters are not meant to be good all-rounders; they are sharp specialists whose discovered counters get folded back into what the Main Agents must learn to handle.

Matchmaking ties the population together with deliberate probabilities. A Main Agent spends roughly 35% of its games in straight self-play, 50% against Main Exploiters, and 15% against League Exploiters — a diet heavy on the agents purpose-built to beat it. Crucially the matchmaking is payoff-based: you oversample the opponents you currently struggle against, so compute flows toward your weaknesses rather than toward easy wins. The frozen archive of past agents is what cures the forgetting — because old strategies persist in the league as live opponents, the Main Agent never gets to abandon the skill of beating them. The archive *is* the solution to non-transitivity.

The scale was extraordinary. At its peak the AlphaStar league held around 900 agents, each continuously training against the matchmade population. A single StarCraft II game runs about 20 seconds of real time at training speed, and the infrastructure ran at roughly 200× real-time, which works out to something on the order of 40 years of game experience accumulated per day. That is the brute fact behind "200 years of play per agent" that critics rightly flagged as heroic.

Measuring progress required its own innovation, because ELO — designed for transitive games like chess — is misleading when strategies cycle. A rock that beats every scissors in the pool can post a gaudy ELO while losing to the one paper agent that matters. DeepMind used **Nash averaging**: rather than a flat win-rate, you compute the Nash equilibrium of the meta-game (the matrix of who-beats-whom across the population) and weight each agent by its frequency in that equilibrium. The result is a strength measure that cannot be gamed by farming wins against a single exploitable strategy — it rewards being hard to counter, which is the property you actually want.

The payoff was concrete. AlphaStar reached Grandmaster rank in all three races — Terran, Zerg, and Protoss — placing it above roughly 99.8% of ranked human players, in the top 0.15%. Earlier, in the December 2018 / January 2019 showcase, an earlier build defeated the professional player TLO and then MaNa, with the headline 5–0 result against a pro on a live broadcast that put StarCraft II alongside Go and Dota 2 as a game where machines had reached the human elite. The honest caveat, which we noted in the case studies, is that the earliest exhibition versions enjoyed superhuman action-execution advantages; the Grandmaster result came under progressively more human-like interface and action-rate constraints, which is why it is the one worth quoting.

## 12d. Compute efficiency and the scaling frontier

The decade's headlines were about capability, but the quieter and arguably more important story is about *cost*. Lay the systems side by side and a striking pattern emerges — followed by an equally striking exception.

Consider the raw compute footprints. DQN trained on roughly 200M frames on a single GPU over about a week. AlphaGo Zero played 4.9 million self-play games on 4 TPUs over 3 days. AlphaStar consumed something equivalent to 200 CPU-years of experience per agent across a months-long league run. OpenAI Five ran on the order of 180,000 CPU cores for about 10 months. These are not differences of degree; they span many orders of magnitude, and the spread maps directly onto the kind of game being solved.

The more revealing lens is efficiency: capability per unit of compute, or ELO per FLOP. By that measure DQN is the bargain of the bunch — it reached superhuman play on 29 Atari games for something on the order of 10^18 FLOPs. AlphaZero sits a few orders up, mastering Go in the neighborhood of 10^21 FLOPs, which is enormous in absolute terms but extraordinarily cheap for the difficulty of the problem. OpenAI Five and AlphaStar are dramatically more expensive *per unit of skill*: real-time, partially observed, multi-agent games appear to demand brute force at the current state of the art, with no clever search to amortize the cost.

For a while it looked like the field was on a clean efficiency-improving trajectory. AlphaGo (bootstrapped on human games) gave way to AlphaGo Zero (no human data, stronger), which gave way to AlphaZero (one algorithm, multiple games, less total compute for higher performance), which gave way to MuZero (the same strength without even being told the rules, and more efficient again). Four steps, each doing more with less — a genuine Moore's-law-of-algorithms feel. And then OpenAI Five and AlphaStar broke the trend hard: they needed staggering brute force and did *not* slot into the tidy efficiency curve, because Dota 2 and StarCraft II simply do not yield to search the way board games do.

Sample efficiency — how much *game experience* you need, separate from raw FLOPs — tells a parallel story. DQN's 200M frames became the standard yardstick, and for years matching human-level Atari performance meant roughly that many frames at roughly that compute. World-model methods complicated the picture: MuZero hits human-level Atari in a comparable frame budget but at meaningfully different compute, because it spends its budget planning inside a *learned* model rather than reacting to raw observations — more sample-efficient in some regimes, but with the added complexity and compute of training the model itself.

The sharpest evidence that the frontier has not plateaued is the DreamerV3 result from 2023. By learning a world model and training the policy almost entirely in compact latent space — imagining rollouts rather than executing them in the environment — DreamerV3 reaches near-superhuman Atari with on the order of 40M frames, roughly 5× more sample-efficient than DQN's 200M. That is a large jump well after the field had supposedly matured, and it argues that the compute-performance frontier for game-playing RL is still moving.

Which raises the question of scaling laws. LLMs enjoy clean power-law relationships between compute, data, parameters, and loss; game-playing RL conspicuously does not. There is no tidy exponent that tells you how much more compute buys how much more skill, because the *game itself* dominates the relationship. Perfect-information board games scale cleanly and predictably; real-time strategy games are messy, with returns that depend on whether your method can exploit structure or is reduced to brute force. The lesson is that "scale it up" is good advice in language modeling and unreliable advice in game-playing RL — what scales depends entirely on the game.

So the practical guidance, distilled for someone starting a new game-playing project, is unglamorous and money-saving. Reach for **Rainbow DQN** for arcade-style games with dense rewards and short horizons; it is sample-efficient relative to the alternatives and battle-tested. Reach for **AlphaZero** when you have a perfect simulator and perfect information; nothing matches MCTS plus self-play there. Reach for **PPO** for continuous-control and real-time games where search is intractable. And only reach for **AlphaStar-scale league infrastructure** when you genuinely have an environment that is Atari-simple to simulate but must be played at industrial scale against a non-transitive strategy space — which, for almost every team, is a bar you will never need to clear. The frontier systems are demonstrations of what is possible at the limit, not the default tool for the problem on your desk.

## Key takeaways

1. **Two ideas made deep Q-learning stable**: experience replay (breaks correlation, restores i.i.d.) and a frozen target network (stops the moving target). Remove either and the value estimates diverge — this is the deadly triad biting.
2. **Frame stacking restores the Markov property.** One frame shows position; four frames show velocity. Many "RL won't converge" bugs are really partial-observability bugs in disguise.
3. **MCTS is a policy improvement operator.** Search a step ahead with the network's prior, take the improved visit-count policy as a training target, distill it back into the network, and iterate. That loop is the engine of the entire AlphaGo lineage.
4. **A learned value network replaces expensive rollouts**, making deep search affordable — the single innovation that turned Go from impossible to solved.
5. **You can throw away the human data and get stronger** (AlphaGo Zero 100–0), and you can throw away the *rules* too (MuZero) by learning a model that predicts only reward, value, and policy — not the full next observation.
6. **Scale can substitute for search.** OpenAI Five solved Dota 2 with plain PPO and ~180,000 cores, no new algorithm — a reminder that a stable algorithm plus massive parallelism is its own kind of breakthrough.
7. **League training beats strategic non-transitivity.** When strategies cycle rock-paper-scissors style, naive self-play collapses; a population of main agents and exploiters pushes toward an unexploitable equilibrium.
8. **Sparse rewards need intrinsic motivation.** Count-based bonuses, ICM, RND, and Go-Explore solve the cold-start problem of getting *any* reward signal — the difference between scoring 0 and 11,000 on Montezuma's Revenge.
9. **Superhuman in-distribution is not robust.** Every landmark system had exploitable blind spots outside its training distribution. Treat benchmark dominance as a ceiling on capability, not a guarantee of reliability.
10. **The compute is a silent co-author.** These are demonstrations of what is possible at the frontier, not recipes for a single GPU. Match the method to your actual compute and simulator budget, not to the headline.

## Further reading

- Mnih et al., "Human-level control through deep reinforcement learning," *Nature*, 2015 — the DQN paper; replay + target network.
- Hessel et al., "Rainbow: Combining Improvements in Deep Reinforcement Learning," AAAI, 2018 — the six-improvement ablation.
- Silver et al., "Mastering the game of Go with deep neural networks and tree search," *Nature*, 2016 — AlphaGo.
- Silver et al., "Mastering the game of Go without human knowledge," *Nature*, 2017 — AlphaGo Zero; MCTS as policy improvement.
- Silver et al., "A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play," *Science*, 2018 — AlphaZero.
- Schrittwieser et al., "Mastering Atari, Go, chess and shogi by planning with a learned model," *Nature*, 2020 — MuZero.
- OpenAI et al., "Dota 2 with Large Scale Deep Reinforcement Learning," 2019 — OpenAI Five.
- Vinyals et al., "Grandmaster level in StarCraft II using multi-agent reinforcement learning," *Nature*, 2019 — AlphaStar and league training.
- Burda et al., "Exploration by Random Network Distillation," ICLR, 2019 — RND on Montezuma's Revenge.
- Sutton & Barto, *Reinforcement Learning: An Introduction* (2nd ed.), 2018 — the foundational text for everything above.
- Within this series: [deep Q-networks (DQN)](/blog/machine-learning/reinforcement-learning/deep-q-networks-dqn), [Rainbow DQN](/blog/machine-learning/reinforcement-learning/rainbow-dqn-combining-six-improvements), [MuZero](/blog/machine-learning/reinforcement-learning/muzero-mastering-games-without-rules), [PPO](/blog/machine-learning/reinforcement-learning/proximal-policy-optimization-ppo), and [exploration vs exploitation](/blog/machine-learning/reinforcement-learning/exploration-vs-exploitation-the-core-tension). The forthcoming unified map and the capstone playbook tie this game-playing track back to the rest of the series.
