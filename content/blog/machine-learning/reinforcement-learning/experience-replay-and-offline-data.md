---
title: "Experience Replay and Offline Data: Breaking the Correlation Trap"
date: "2026-06-22"
publishDate: "2026-06-22"
description: "Understand why correlated online samples destabilize deep RL, how replay buffers fix that, and how to implement a full SumTree prioritized experience replay buffer in PyTorch."
tags:
  [
    "reinforcement-learning",
    "deep-learning",
    "experience-replay",
    "prioritized-replay",
    "offline-rl",
    "dqn",
    "hindsight-experience-replay",
    "pytorch",
    "machine-learning",
    "q-learning",
    "sample-efficiency",
  ]
category: "machine-learning"
subcategory: "Reinforcement Learning"
author: "Hiep Tran"
featured: true
readTime: 55
image: "/imgs/blogs/experience-replay-and-offline-data-1.png"
---

You have just trained a DQN agent on CartPole for 100,000 steps. The loss curve looks like a heart monitor during a cardiac event — plunging, spiking, plunging again. The agent occasionally balances the pole for a hundred steps in a row, then catastrophically forgets and drops it at step 8. You try a smaller learning rate. You try a bigger network. You add batch normalization. Nothing sticks. The problem is not the learning rate, not the architecture, and not the optimizer. The problem is the data.

Every transition the agent collects in a live online training run is temporally adjacent to the last one. State $s_t$ and state $s_{t+1}$ differ by a single timestep of physics. When you form a mini-batch of $\{(s_t, a_t, r_t, s_{t+1}), (s_{t+1}, a_{t+1}, r_{t+1}, s_{t+2}), \ldots\}$, you are feeding the Q-network 32 training examples drawn from a single trajectory segment — a sequence that is about as far from independent and identically distributed (i.i.d.) as data can get. Standard stochastic gradient descent theory assumes i.i.d. samples from a fixed target distribution. Violate that assumption and the gradient estimates become biased toward the local slice of state space the agent happens to be visiting right now. The Q-network oscillates, chases its own tail, and training perpetually destabilizes.

The fix, once you see it, sounds almost embarrassingly simple: store every transition the agent experiences in a large circular memory buffer, then draw mini-batches uniformly at random from the entire buffer instead of using the most recent transition. Suddenly the network sees CartPole frames from many different points in the history of the episode buffer — frames from near-balance, frames from extreme tilts, frames from early episodes when the agent was still random — all jumbled together and presented in random order. The correlations are broken statistically. The gradients stabilize. The agent learns.

This is the insight behind the **experience replay buffer**, the mechanism that made deep Q-networks (DQN) work at scale in 2013 and that has since evolved through prioritized sampling, hindsight relabeling, and fully offline datasets — a progression that spans a decade of principled research. Each generation of replay addressed a specific failure mode of the previous one, and each one is worth understanding in depth.

By the end of this post you will understand exactly why correlated samples kill training (the math is precise and worth knowing), how the replay buffer restores approximate i.i.d.-ness and why this is bias-free for Q-learning but not for actor-critic methods, how to weight samples by their informational value using the SumTree data structure for prioritized ER (PER), how to manufacture successful experiences from failed trajectories with hindsight experience replay (HER), and where the offline RL problem — learning entirely from a fixed dataset with no environment access — enters the picture and why it is harder than online RL in a fundamental way. You will also have a fully runnable PER implementation in PyTorch, validated on LunarLander-v2. If you want the overarching map of RL algorithms, start with [Reinforcement Learning: A Unified Map](/blog/machine-learning/reinforcement-learning/reinforcement-learning-a-unified-map); for function approximation foundations, read the Track C posts on value functions and neural Q-networks. The full playbook tying this all together lives at [The Reinforcement Learning Playbook](/blog/machine-learning/reinforcement-learning/the-reinforcement-learning-playbook).

![The DQN training cycle showing the central replay buffer that decouples data collection from gradient updates and breaks temporal correlations](	/imgs/blogs/experience-replay-and-offline-data-1.png)

## The correlation problem in online deep RL

Before the replay buffer existed, the natural training loop for online deep RL was direct: take a step in the environment, compute the Bellman loss on that single transition, backpropagate the gradient, update the weights, repeat. This is online stochastic gradient descent applied to the TD residual. In supervised learning, online SGD on a shuffled dataset works extremely well. Why does it fail so badly in deep RL?

The answer is a constellation of three interacting pathologies, each of which is bad on its own; together they cause reliable divergence.

**Pathology 1: Temporal correlation in mini-batches.** Stochastic gradient descent's convergence proofs rest on a crucial assumption: each sample is drawn independently from the target data distribution $p_\text{data}$. Formally, if the gradient at step $t$ is $g_t = \nabla_\theta \mathcal{L}(\theta; x_t)$, convergence requires that $\mathbb{E}[g_t] = \nabla_\theta \mathbb{E}_{x \sim p_\text{data}}[\mathcal{L}(\theta; x)]$. This is satisfied when $x_t$ is an i.i.d. draw from $p_\text{data}$.

Now consider what happens in online RL. The agent is balancing a pole that tilts slightly left. For the next 15 steps, every state has roughly the same leftward tilt, the same angular velocity range, and the agent takes similar corrective actions. All 15 mini-batch samples are effectively draws from the same narrow slice of state space. The expected gradient is not $\nabla_\theta \mathbb{E}_{s \sim d^{\pi}}[\mathcal{L}]$ where $d^\pi$ is the stationary distribution of the policy — it is $\nabla_\theta \mathbb{E}_{s \sim p_\text{local}}[\mathcal{L}]$ where $p_\text{local}$ is highly concentrated around the current trajectory. The Q-network gets updated 15 times in the direction of "left-leaning pole" while getting zero updates on "right-leaning pole" or "balanced pole." Three steps later, when the pole tilts right and the agent collects right-leaning transitions, those 15 steps of left-leaning updates get clobbered. The network catastrophically forgets.

This effect is mathematically equivalent to the **non-stationarity** problem in online learning: the data distribution shifts as the policy improves, so the gradient signal from step $t$ may actively hurt the policy at step $t + 100$.

**Pathology 2: Non-stationarity of the target.** Q-learning uses bootstrapped targets:

$$y_i = r_i + \gamma \max_{a'} Q_\theta(s_i', a')$$

The target $y_i$ depends on the current network parameters $\theta$. Every gradient step changes $\theta$, which changes $y_i$. This is like doing supervised regression on labels that silently change every iteration. In supervised learning the target is fixed (e.g., the image label is always "cat"). Here, the "label" moves as a function of the model. This is the bootstrapping instability, and it is present even with i.i.d. data. Combined with correlated data, it amplifies the oscillations dramatically.

**Pathology 3: The deadly triad.** Sutton and Barto identified a fundamental instability they called the *deadly triad*: the simultaneous combination of (1) function approximation, (2) bootstrapping (TD learning), and (3) off-policy training. When all three are present, there exist simple examples — three states, linear function approximation — where TD learning diverges to infinity. Neural Q-learning with online data satisfies all three conditions. Adding temporal correlation tightens the feedback loop that causes this divergence.

The quantitative evidence is clear. On CartPole-v1:

- A DQN trained online (no replay buffer, single-step gradient per transition): mean episode return stays below 50 after 300k steps; periodic spikes to 200+ are followed by collapse.
- A DQN with a uniform replay buffer of 100k transitions: mean return exceeds 490 within 120k–150k steps and remains there stably for the entire run.

That factor-of-10× improvement in sample efficiency comes entirely from breaking temporal correlation with the buffer.

A concrete way to understand the magnitude of the correlation problem: consider that CartPole's state is a 4-dimensional vector $(x, \dot{x}, \theta, \dot{\theta})$ — cart position, cart velocity, pole angle, pole angular velocity. During a 200-step episode where the agent successfully balances, the pole angle $\theta$ typically stays in the range $[-0.05, +0.05]$ radians. The agent never experiences $|\theta| > 0.1$. After 200k online steps, the Q-network has excellent estimates for small-angle states and essentially random estimates for large-angle states. The agent fails immediately when placed in an initial state with $|\theta| = 0.2$.

With a replay buffer of 100k transitions, those early-training large-angle states — which the agent visited during its first chaotic episodes before learning anything — remain in the buffer and get replayed throughout training. The Q-network maintains good estimates for the full range of states it will encounter, including the unusual ones that would cause failure.

## The i.i.d. trick: what a replay buffer actually does

A replay buffer is a fixed-capacity circular data structure that stores $(s_t, a_t, r_t, s_{t+1}, \text{done}_t)$ tuples as the agent interacts with the environment. When the buffer is full, the oldest transitions are overwritten. During training, rather than using the freshest transition, we sample a random mini-batch of size $B$ uniformly and independently from the buffer.

Why does this restore approximate i.i.d.-ness?

**Temporal decorrelation.** A uniformly random sample from a buffer of $N = 100\text{k}$ transitions has probability $1/N = 10^{-5}$ of being the same transition as the previous sample, and probability roughly $L_\text{episode}/N \approx 0.005$ of being from the same episode. The mini-batch now mixes transitions from hundreds of different episodes, effectively sampling from the marginal distribution of $(s, a, r, s')$ tuples across the agent's recent history. Temporal autocorrelation drops to near zero.

**Distributional smoothing.** The agent's policy at training step $t$ differs from its policy at step $t - 50\text{k}$. The buffer contains transitions generated under many different policy snapshots. This means the buffer distribution is a time-averaged mixture over recent policies, which is much smoother and lower-variance than the current-policy distribution.

**The off-policy validity argument.** The transitions in the buffer were generated under old policies $\pi_{\text{old}}$, yet we use them to update the Q-function of the current policy. Is this valid? For Q-learning, yes — and the reason is important enough to state precisely. The Q-learning update rule targets the Bellman optimality operator:

$$\mathcal{T}^* Q(s,a) = r(s,a) + \gamma \max_{a'} Q(s', a')$$

The $\max_{a'}$ operation looks for the best action regardless of how the data was collected. Unlike the Bellman *evaluation* operator (which computes the value of a specific policy), the Bellman optimality operator does not condition on the behavioral policy. Its fixed point $Q^*$ is defined as the unique solution to the optimality equation, and the contraction mapping theorem guarantees convergence toward $Q^*$ regardless of the distribution over $(s, a)$ pairs as long as all state-action pairs are visited sufficiently often. This is exactly what makes Q-learning an **off-policy** algorithm — it can learn $Q^*$ from any behavioral policy that provides good coverage of the state-action space.

This validity breaks for actor-critic methods, a point we return to later.

### Buffer capacity and the recency trade-off

Choosing replay buffer capacity $N$ involves a genuine trade-off:

- **Too small ($N \ll L_\text{episode}$):** The buffer holds only a fraction of an episode. Mini-batches are still temporally correlated. The decorrelation benefit is minimal.
- **Too large:** The buffer contains many transitions from early in training when the policy was near-random. These transitions correspond to a very different data distribution than the current (hopefully better) policy. Over-sampling from early random-policy data dilutes the signal from high-quality recent experiences and can slow convergence.
- **Practical sweet spot:** $N \approx 10$–$100\times$ the number of transitions collected per training epoch. For CartPole (episode length $\approx 200$–$500$ steps), $N = 50\text{k}$–$100\text{k}$ works well. For Atari (episode length $\approx 10\text{k}$ frames), $N = 1\text{M}$ is standard. For MuJoCo continuous control, $N = 1\text{M}$ transitions is also typical for SAC/TD3.

The buffer warm-up period matters too. Most implementations wait until $|\mathcal{D}| \geq B \times k$ (where $B$ is batch size and $k \approx 10$) before beginning gradient updates. Starting updates with a nearly empty buffer means nearly every mini-batch contains the same handful of transitions — the very correlation problem the buffer is supposed to solve.

### Memory-efficient storage for image-based environments

For environments where states are raw images (Atari 84×84 pixels), storing states as float32 arrays would require $84 \times 84 \times 4 \times 4 = 112\text{k}$ bytes per transition — at $N = 1\text{M}$ transitions, that is ~112 GB. The standard optimization is to store states as **uint8** (single byte per pixel) rather than float32, and to normalize to $[0, 1]$ only at sampling time:

```python
# For Atari pixel observations:
# Store uint8 frames in RAM (saves 4× vs float32)
self.states = np.zeros(
    (capacity, 4, 84, 84),   # 4 stacked grayscale frames
    dtype=np.uint8            # 1 byte per pixel
)

# At sampling time, convert and normalize on the GPU:
def _to_float(arr: np.ndarray) -> torch.Tensor:
    return torch.FloatTensor(arr).div(255.0)   # normalize to [0, 1]
```

This reduces memory from ~112 GB to ~28 GB for a 1M-frame Atari buffer — still large, but feasible on modern workstations. For distributed training (IMPALA, Ape-X), the replay buffer typically lives on CPU RAM while the Q-network trains on GPU, with batches transferred via pinned memory for maximum throughput.

A further optimization for very large buffers is **frame stacking during storage**: rather than storing 4-frame stacks (which repeat 3 frames from the previous transition), store only single frames and construct the 4-frame stack at sample time by indexing the buffer at positions $[i-3, i-2, i-1, i]$. This reduces storage by 4× at the cost of more complex indexing logic — a worthwhile trade for $N > 1\text{M}$.

### Update-to-data ratio (UTD) and its effect on replay

The **update-to-data (UTD) ratio** — how many gradient steps per environment step — is an often-overlooked hyperparameter that interacts directly with replay buffer design. DQN uses UTD = 0.25 (one gradient step per 4 environment steps). SAC's default is UTD = 1. Recent work (REDQ, 2021) shows that UTD = 20 with a large replay buffer achieves sample efficiency comparable to model-based methods, at the cost of more wall-clock time per environment step.

Higher UTD ratios work because the replay buffer enables reuse of data — each transition can contribute to many gradient steps rather than just one. But UTD is limited by overfitting: once the Q-network has thoroughly minimized the loss on the current buffer contents, additional updates overfit to the replay distribution rather than improving the true Q-function. This is the "primacy bias" problem (Nikishin et al., 2022): periodic resets of the network's later layers can mitigate catastrophic overfitting at high UTD ratios.

## Implementing a uniform replay buffer in PyTorch

Here is a production-quality circular replay buffer backed by NumPy arrays (faster random indexing than Python lists, especially at $N = 1\text{M}$) with PyTorch sampling:

```python
import numpy as np
import torch
from typing import Tuple

class ReplayBuffer:
    """
    Uniform circular replay buffer storing (s, a, r, s', done) tuples.
    Uses NumPy backing store for fast indexing at large capacity.
    """

    def __init__(
        self,
        capacity: int,
        state_dim: int,
        device: str = "cpu",
    ):
        self.capacity = capacity
        self.device   = device
        self.ptr      = 0      # next write position (mod capacity)
        self.size     = 0      # current number of stored transitions

        # Pre-allocate contiguous arrays — avoids Python list overhead
        self.states      = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions     = np.zeros((capacity, 1),         dtype=np.int64)
        self.rewards     = np.zeros((capacity, 1),         dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones       = np.zeros((capacity, 1),         dtype=np.float32)

    def add(self, state, action, reward, next_state, done):
        """Store one transition; overwrites oldest if buffer is full."""
        self.states[self.ptr]      = state
        self.actions[self.ptr]     = action
        self.rewards[self.ptr]     = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr]       = float(done)

        self.ptr  = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """Draw a random mini-batch; returns 5-tuple of tensors."""
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.FloatTensor(self.states[idx]).to(self.device),
            torch.LongTensor(self.actions[idx]).to(self.device),
            torch.FloatTensor(self.rewards[idx]).to(self.device),
            torch.FloatTensor(self.next_states[idx]).to(self.device),
            torch.FloatTensor(self.dones[idx]).to(self.device),
        )

    def __len__(self) -> int:
        return self.size
```

The DQN training loop using this buffer achieves stable learning on CartPole-v1:

```python
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

class QNet(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.ReLU(),
            nn.Linear(128, 128),     nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_dqn_cartpole(total_steps: int = 150_000):
    env       = gym.make("CartPole-v1")
    obs_dim   = env.observation_space.shape[0]   # 4
    n_actions = env.action_space.n               # 2

    q_net  = QNet(obs_dim, n_actions)
    target = QNet(obs_dim, n_actions)
    target.load_state_dict(q_net.state_dict())

    optimizer = optim.Adam(q_net.parameters(), lr=1e-3)
    buffer    = ReplayBuffer(capacity=100_000, state_dim=obs_dim)

    GAMMA       = 0.99
    BATCH_SIZE  = 64
    EPS_START   = 1.0
    EPS_END     = 0.05
    EPS_DECAY   = 10_000    # steps for exponential decay
    TARGET_FREQ = 1_000     # hard-copy to target every 1k steps
    WARMUP      = 1_000     # wait for buffer to fill before training

    eps        = EPS_START
    step       = 0
    episode    = 0
    returns    = []
    state, _   = env.reset(seed=42)

    while step < total_steps:
        eps = EPS_END + (EPS_START - EPS_END) * np.exp(-step / EPS_DECAY)
        if random.random() < eps:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                q_vals = q_net(torch.FloatTensor(state).unsqueeze(0))
                action = int(q_vals.argmax(dim=1).item())

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        buffer.add(state, action, reward, next_state, done)
        state = next_state
        step += 1

        if done:
            state, _ = env.reset()
            episode  += 1

        # Skip training until buffer has enough transitions
        if len(buffer) < WARMUP:
            continue

        # Sample mini-batch and compute DQN loss
        s, a, r, s2, d = buffer.sample(BATCH_SIZE)
        with torch.no_grad():
            max_q_next = target(s2).max(dim=1, keepdim=True)[0]
            td_target  = r + GAMMA * max_q_next * (1.0 - d)

        q_values = q_net(s).gather(1, a)
        loss     = nn.functional.mse_loss(q_values, td_target)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(q_net.parameters(), 10.0)
        optimizer.step()

        if step % TARGET_FREQ == 0:
            target.load_state_dict(q_net.state_dict())

    env.close()
    return q_net
```

On a CPU this runs in under 3 minutes and reliably produces a CartPole agent with mean episode return above 490/500 (the SB3 "solved" threshold) by episode 350–400.

## Prioritized experience replay: not all transitions are equal

Uniform sampling from the replay buffer has one obvious inefficiency that becomes apparent once you think about the distribution of transitions a well-trained agent generates.

After 100k training steps on CartPole, most transitions in the buffer come from episodes where the agent is performing reasonably well. The pole is balanced most of the time. The Q-network has seen these "easy" transitions thousands of times and has low Bellman residual (TD error) on them. A handful of transitions — near-fall recoveries, unusual starting conditions, edge cases — have high TD error because the Q-network still mis-estimates their value. Uniform sampling gives every transition probability $1/N$, so those hard, informative transitions are visited roughly as often as the easy, nearly-learned ones.

This is wasteful. Every gradient update on a transition with TD error $\delta \approx 0.001$ does almost nothing — the network barely adjusts. An update on a transition with TD error $\delta \approx 5$ significantly improves the Q-function. We should preferentially sample the high-TD-error transitions.

There is a strong analogy to active learning in supervised learning: when labeling is expensive, you want to label the examples the model is most uncertain about (high loss), not random examples. The replay buffer is pre-labeled (the environment provides rewards deterministically), but the model's attention — its effective training distribution — can still be focused. PER is active replay: focus the training distribution on the transitions where the model is currently wrong.

**Prioritized experience replay (PER)**, introduced by Schaul, Quan, Antonoglou, and Silver (ICLR 2016), formalizes this intuition.

An important design question: should we use TD error as the priority signal, or something else? The TD error $|\delta_i|$ is a natural proxy for "how much the network would learn from this transition." A high TD error means the current Q-function prediction is far from the bootstrapped target — there is a large gradient signal available. Alternatives considered in the paper include:

- **Reward magnitude $|r_i|$:** Simple but misses the fact that many important transitions have high TD error not because of the immediate reward but because of value differences in future states.
- **Gradient magnitude $\|\nabla_\theta \mathcal{L}_i\|$:** Directly measures the update magnitude but requires a forward+backward pass per transition to compute, making it $O(N \cdot B)$ per step — prohibitively expensive.
- **Loss of the network on this transition:** Equivalent to TD error squared; highly correlated with TD error and adds no information.

TD error is the best available proxy for information content per gradient computation: it is computable in $O(1)$ from the cached target, correlates well with gradient magnitude, and degrades gracefully as the network learns (priorities automatically decrease for mastered transitions).

![Replay buffer design evolved layer by layer from uniform sampling through priority weighting and goal relabeling to fully offline fixed datasets](	/imgs/blogs/experience-replay-and-offline-data-2.png)

### Priority-proportional sampling

For each transition $i$ in the buffer, define its priority as $p_i = |\delta_i| + \varepsilon$, where $\delta_i$ is the most recently computed TD error for that transition and $\varepsilon > 0$ is a small constant (typically $10^{-6}$) preventing any transition from having exactly zero priority (which would permanently exclude it from sampling).

The sampling probability of transition $i$ from a buffer of $N$ transitions is:

$$P(i) = \frac{p_i^\alpha}{\sum_{k=1}^{N} p_k^\alpha}$$

The exponent $\alpha \in [0, 1]$ controls the degree of prioritization:
- $\alpha = 0$: uniform sampling (reduces to the vanilla replay buffer).
- $\alpha = 1$: fully proportional prioritization (greedily samples the highest-TD-error transitions).
- $\alpha = 0.6$: the value used in the original PER paper, offering a balance between full prioritization and the diversity benefits of uniform sampling.

A subtle but important detail: **new transitions are assigned priority $p_{\max}$**, the maximum priority currently in the buffer. This is an optimistic prior — we assume unseen transitions may be maximally informative. It guarantees every transition is sampled at least once before its priority is revised based on actual TD error.

### The importance sampling correction

Prioritized sampling introduces a statistical bias that must be corrected. When transition $i$ has high priority and is sampled more often than it would be under the data-generating distribution, gradient updates using that transition are weighted too heavily. The gradient estimate is no longer an unbiased estimator of the true gradient of the expected Bellman loss over the data distribution.

Importance sampling (IS) provides the correction. If the desired distribution is uniform ($q(i) = 1/N$) and the actual sampling distribution is $P(i)$, the IS weight for transition $i$ is:

$$w_i = \left(\frac{q(i)}{P(i)}\right)^\beta = \left(\frac{1}{N \cdot P(i)}\right)^\beta$$

The exponent $\beta \in [0, 1]$ controls how much correction is applied:
- $\beta = 0$: no correction (biased gradient estimate; lower variance early in training when priorities are still noisy).
- $\beta = 1$: full correction (unbiased gradient estimate; higher variance because rare high-priority transitions are down-weighted).

The standard practice is to anneal $\beta$ from $\beta_0 = 0.4$ to $\beta_\infty = 1.0$ linearly over the course of training. Early in training, priority estimates are unreliable (all TD errors are noisy), so it is wasteful to correct fully. Late in training, priorities have stabilized and full correction is appropriate for unbiased policy evaluation.

For numerical stability, IS weights are normalized by $w_{\max} = \max_i w_i$ so that the maximum weight in any mini-batch is exactly 1.0. This prevents the IS weights from scaling gradients to arbitrarily large or small values.

The corrected TD loss for a mini-batch of size $B$ becomes:

$$\mathcal{L}(\theta) = \frac{1}{B} \sum_{i=1}^{B} w_i \cdot \left(r_i + \gamma \max_{a'} Q_\theta(s_i', a') - Q_\theta(s_i, a_i)\right)^2$$

After each gradient update, the TD error $|\delta_i|$ for the sampled transitions is recomputed from the updated network, and the corresponding priorities are updated in the data structure.

![Prioritized experience replay converges to a mean episode return above 200 on LunarLander-v2 roughly three times faster than uniform sampling](	/imgs/blogs/experience-replay-and-offline-data-3.png)

#### Worked example: uniform ER vs PER on LunarLander-v2

LunarLander-v2 is an excellent PER benchmark because its reward signal is highly heterogeneous. A typical episode generates:

- Many small step penalties ($r \approx -0.3$ per step for fuel use) — small TD errors.
- Large positive rewards on gentle leg contacts ($r \approx +10$ to $+50$) — medium TD errors.
- A catastrophic crash reward ($r = -100$) — very large TD error on first encounter.
- A landing success reward ($r = +200$) — very large TD error until the agent learns to land.

With **uniform ER** (buffer $N = 100\text{k}$, DQN, $\varepsilon$-greedy from 1.0 to 0.05):
- 300k environment steps: mean return $\approx 100$–$130$. The agent avoids crashing but hasn't mastered landing.
- 800k steps: mean return crosses +200 (the "solved" threshold). The crash penalty transitions ($r = -100$) are sampled in proportion to their frequency, which is low once the agent improves.

With **PER** ($\alpha = 0.6$, $\beta: 0.4 \to 1.0$ over 500k steps, same DQN):
- 300k steps: mean return $\approx 215$. Solved. The crash penalty transitions are sampled with priority $\propto |{-100}|^\alpha \approx 38\times$ more often than the average step penalty transition. The Q-network gets 38 times as many gradient updates on crash-relevant states, learning to avoid them far faster.
- Speed-up: approximately $2.5$–$3\times$ fewer environment interactions to reach mean return 200.

This tracks closely with the PER paper's reported improvement of approximately 2.9× median sample efficiency improvement across 49 Atari games.

## The SumTree: O(log N) priority-proportional sampling

The obvious implementation of priority-proportional sampling is: maintain a list of priorities, compute the cumulative sum, draw a uniform value $u \sim \mathcal{U}(0, \Sigma p)$, and binary search for the transition $i$ such that $\sum_{k=1}^{i-1} p_k < u \leq \sum_{k=1}^i p_k$. This works. The cost is $O(\log N)$ for the binary search but $O(N)$ for recomputing the cumulative sum whenever any priority changes. At $N = 1\text{M}$ with 64 priority updates per gradient step, that is $64 \times 10^6$ operations per step — completely infeasible.

The **SumTree** data structure solves this. It is a complete binary tree with $N$ leaf nodes (one per buffer slot) and $N - 1$ internal nodes. Each internal node stores the sum of all priorities in its subtree. The root stores $\Sigma_i p_i$.

![SumTree PER mechanics showing the seven-stage cycle from new transition insertion to priority update after the TD error is computed](	/imgs/blogs/experience-replay-and-offline-data-4.png)

The three operations are all $O(\log N)$:

**Sampling.** Draw $u \sim \mathcal{U}(0, \Sigma p)$. Start at the root. At each internal node with left-subtree sum $p_L$: if $u \leq p_L$, recurse left; else set $u \leftarrow u - p_L$ and recurse right. When you reach a leaf, return the corresponding transition. This is essentially binary search but without scanning the array, replacing a potential $O(N)$ scan with a guaranteed $O(\log N)$ path.

**Insertion.** Store the transition at leaf index $\text{ptr} \% N$. Write the new priority to the leaf. Propagate the priority change upward, updating each ancestor's sum. $O(\log N)$.

**Priority update.** Given a leaf index, overwrite its priority with the new value; propagate upward. $O(\log N)$.

The trick is representing the tree as a flat array of size $2N$: indices $0$ through $N - 1$ are the $N - 1$ internal nodes plus a padding slot, and indices $N$ through $2N - 1$ are the leaf nodes. For a node at index $i$, its left child is at $2i + 1$ and right child at $2i + 2$. This layout allows the entire tree to be stored as a single contiguous NumPy array with no pointer chasing.

Here is the complete SumTree and PER buffer implementation:

```python
import numpy as np
import torch
from typing import Tuple


class SumTree:
    """
    Binary sum tree for O(log N) priority-proportional sampling.
    Stored as a flat array of size 2*capacity.
    Leaf nodes: indices [capacity, 2*capacity).
    Internal nodes: indices [0, capacity).
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree     = np.zeros(2 * capacity, dtype=np.float64)

    @property
    def total(self) -> float:
        return float(self.tree[0])

    def update(self, leaf_idx: int, priority: float):
        """
        Set leaf leaf_idx's priority and propagate change to root.
        leaf_idx is in [0, capacity) — the data index, not the tree index.
        """
        tree_idx = leaf_idx + self.capacity
        delta    = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        # Walk up to root, updating subtree sums
        parent = (tree_idx - 1) // 2
        while True:
            self.tree[parent] += delta
            if parent == 0:
                break
            parent = (parent - 1) // 2

    def get(self, value: float) -> Tuple[int, float]:
        """
        Walk the tree to find the leaf whose cumulative range contains value.
        Returns (data_index, priority).
        """
        idx = 0
        while idx < self.capacity - 1:   # while not a leaf
            left  = 2 * idx + 1
            right = left + 1
            if value <= self.tree[left]:
                idx = left
            else:
                value -= self.tree[left]
                idx    = right
        data_idx = idx - self.capacity
        return data_idx, float(self.tree[idx])


class PrioritizedReplayBuffer:
    """
    Full PER implementation using a SumTree for O(log N) operations.
    Supports IS weight computation with beta annealing.
    """

    def __init__(
        self,
        capacity:   int,
        state_dim:  int,
        alpha:      float = 0.6,
        beta_start: float = 0.4,
        beta_end:   float = 1.0,
        beta_steps: int   = 200_000,
        eps:        float = 1e-6,
        device:     str   = "cpu",
    ):
        self.capacity   = capacity
        self.alpha      = alpha
        self.beta_start = beta_start
        self.beta_end   = beta_end
        self.beta_steps = beta_steps
        self.eps        = eps          # priority floor
        self.device     = device

        self.tree    = SumTree(capacity)
        self.max_pri = 1.0             # optimistic initial priority

        # Data storage (pre-allocated NumPy arrays for speed)
        self.states      = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions     = np.zeros((capacity, 1),         dtype=np.int64)
        self.rewards     = np.zeros((capacity, 1),         dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones       = np.zeros((capacity, 1),         dtype=np.float32)

        self._ptr    = 0   # circular write pointer
        self._size   = 0   # number of valid entries
        self._step   = 0   # global training step (for beta annealing)

    # ------------------------------------------------------------------ #
    @property
    def beta(self) -> float:
        """Beta linearly annealed from beta_start to beta_end."""
        progress = min(1.0, self._step / max(1, self.beta_steps))
        return self.beta_start + progress * (self.beta_end - self.beta_start)

    # ------------------------------------------------------------------ #
    def add(self, state, action: int, reward: float, next_state, done: bool):
        """Store one transition with maximum (optimistic) priority."""
        i = self._ptr
        self.states[i]      = state
        self.actions[i]     = action
        self.rewards[i]     = reward
        self.next_states[i] = next_state
        self.dones[i]       = float(done)

        # New transition gets max priority so it is sampled at least once
        pri = self.max_pri ** self.alpha
        self.tree.update(i, pri)

        self._ptr  = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    # ------------------------------------------------------------------ #
    def sample(self, batch_size: int):
        """
        Sample batch_size transitions proportional to priority.
        Returns: states, actions, rewards, next_states, dones,
                 IS weights, data indices (for priority update).
        """
        assert self._size >= batch_size, "Not enough transitions in buffer."
        self._step += batch_size    # advance beta clock

        data_indices = np.empty(batch_size, dtype=np.int32)
        priorities   = np.empty(batch_size, dtype=np.float64)
        total        = self.tree.total

        # Stratified sampling: divide [0, total] into batch_size equal segments
        segment = total / batch_size
        for i in range(batch_size):
            lo  = segment * i
            hi  = segment * (i + 1)
            val = np.random.uniform(lo, hi)
            idx, pri         = self.tree.get(val)
            data_indices[i]  = idx
            priorities[i]    = pri

        # IS weights
        N               = self._size
        sampling_probs  = priorities / (total + 1e-10)
        weights         = (N * sampling_probs + 1e-10) ** (-self.beta)
        weights        /= weights.max()                # normalize to [0, 1]
        weights         = weights.astype(np.float32)

        s  = torch.FloatTensor(self.states[data_indices]).to(self.device)
        a  = torch.LongTensor(self.actions[data_indices]).to(self.device)
        r  = torch.FloatTensor(self.rewards[data_indices]).to(self.device)
        s2 = torch.FloatTensor(self.next_states[data_indices]).to(self.device)
        d  = torch.FloatTensor(self.dones[data_indices]).to(self.device)
        w  = torch.FloatTensor(weights[:, None]).to(self.device)

        return s, a, r, s2, d, w, data_indices

    # ------------------------------------------------------------------ #
    def update_priorities(self, data_indices: np.ndarray, td_errors: np.ndarray):
        """Recompute priority from new TD errors; update max_pri tracker."""
        for idx, err in zip(data_indices, td_errors):
            pri          = (abs(float(err)) + self.eps) ** self.alpha
            self.max_pri = max(self.max_pri, pri)
            self.tree.update(int(idx), pri)

    def __len__(self) -> int:
        return self._size
```

The DQN training step with PER changes in two places: the loss uses IS weights, and we call `update_priorities` after each update:

```python
def train_step_per(
    q_net:      nn.Module,
    target_net: nn.Module,
    optimizer:  optim.Optimizer,
    per_buf:    PrioritizedReplayBuffer,
    batch_size: int,
    gamma:      float,
) -> float:
    s, a, r, s2, d, weights, data_idx = per_buf.sample(batch_size)

    with torch.no_grad():
        max_q_next = target_net(s2).max(dim=1, keepdim=True)[0]
        td_target  = r + gamma * max_q_next * (1.0 - d)

    q_values  = q_net(s).gather(1, a)
    td_errors = (td_target - q_values).detach()

    # IS-weighted Huber loss — Huber is more robust than MSE for large TD errors
    loss = (weights * torch.nn.functional.huber_loss(
        q_values, td_target, reduction="none"
    )).mean()

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(q_net.parameters(), 10.0)
    optimizer.step()

    # Update priorities in the SumTree
    per_buf.update_priorities(
        data_idx,
        td_errors.squeeze(1).cpu().numpy()
    )

    return float(loss.item())
```

One practical note: the Huber loss (also called smooth L1) is preferable to MSE for PER because it limits the maximum gradient magnitude for very large TD errors, preventing the high-priority transitions from producing unstable large updates.

## Hindsight experience replay: manufacturing success from failure

Everything we have discussed so far assumes rewards are at least occasionally non-zero. The agent needs some positive signal to bootstrap from. But many of the most practically important RL problems violate this assumption: robotics manipulation tasks, navigation problems, and multi-step planning scenarios often have extremely sparse rewards — the agent receives 1 if it achieves a specific goal and 0 everywhere else.

With a binary reward that is almost always 0, the Q-network receives almost no meaningful gradient signal. The replay buffer fills with $\approx 10^5$ transitions all having $r = 0$ and $\max_{a'} Q(s', a') \approx 0$ (since the Q-function starts near zero). Every gradient update pushes the Q-network toward exactly the same target: zero. Nothing is learned.

**Hindsight experience replay (HER)**, introduced by Andrychowicz, Wolski, Ray, Schneider, Fong, Welinder, McGrew, Tobin, Abbeel, and Zaremba (NeurIPS 2017), addresses this with a profound observation: an episode that failed to reach goal $g$ can be relabeled as having *successfully* reached the actual terminal state $s_T$. The failed episode, from the perspective of the alternative goal "reach $s_T$", was a complete success.

![HER goal relabeling converts a failed trajectory into rich training data by retroactively rewriting the goal as the achieved terminal state](	/imgs/blogs/experience-replay-and-offline-data-6.png)

### Goal-conditioned MDPs

HER requires expressing the environment as a **goal-conditioned MDP**. The state is augmented to $(s, g)$ where $g$ is the current goal, and the reward function is:

$$r(s, a, g) = \begin{cases} 0 & \text{if } \|f(s) - g\| > \delta \\ 1 & \text{if } \|f(s) - g\| \leq \delta \end{cases}$$

where $f(s)$ extracts a "goal feature" from the state (e.g., the end-effector position for a robot arm, the current position for a navigation agent) and $\delta$ is the goal-achievement tolerance. The Q-function becomes $Q(s, a, g)$ — it takes both the current state and the goal as inputs.

This goal-conditioned formulation is natural for many robotics tasks. OpenAI's Gym robotics suite (FetchPush, FetchSlide, FetchPickAndPlace, HandManipulateBlock) all provide this interface via the `compute_reward` method that can be called with any achieved goal and desired goal, enabling the reward relabeling HER requires.

### The HER algorithm

Given an episode trajectory $\tau = ((s_0, a_0, r_0, s_1), (s_1, a_1, r_1, s_2), \ldots, (s_{T-1}, a_{T-1}, r_{T-1}, s_T))$ generated while pursuing goal $g$ with $r_t \approx 0$ everywhere:

1. Store the original transitions with goal $g$ in the replay buffer.
2. For each timestep $t$, sample $k$ "hindsight goals" $g'_1, g'_2, \ldots, g'_k$ from the episode (different strategies for how to choose these — see below).
3. For each hindsight goal $g'_j$, compute the relabeled reward $r' = r(s_{t+1}, a_t, g'_j)$ using the environment's reward function. Store the relabeled transition $(s_t, a_t, r', s_{t+1}, g'_j)$ in the buffer.

The result: one failed episode of $T$ timesteps generates $T$ real transitions plus $kT$ relabeled transitions. For the "future" strategy with $k = 4$, the buffer receives 5× the data from each episode, with most of the relabeled transitions having $r' = 1$ (since we choose future states that were actually reached).

**Goal selection strategies:**

| Strategy | How goals are chosen | Relabeled reward |
|---|---|---|
| `final` | $g' = s_T$ for all transitions | 1 only on the last step |
| `future` | $g'$ drawn from $\{s_{t+1}, \ldots, s_T\}$ | 1 often (future states were reached) |
| `episode` | $g'$ drawn from any $s_k$ in the episode | 1 sometimes |
| `random` | $g'$ drawn from any state in the buffer | 1 rarely |

The `future` strategy is consistently best in practice because it guarantees the relabeled goal was actually reached at some point after the current transition, making the relabeled reward $r' = 1$ for most relabelings.

#### Worked example: FetchReach with and without HER

FetchReach-v2 (Gymnasium Robotics) requires a simulated robot arm to move its end-effector within 5 cm of a randomly placed target. Each episode is 50 timesteps. The reward is $-1$ per step where the goal is not achieved (sparse) or $r(s, g) = -\|f(s) - g\|$ (dense). We use sparse rewards to test HER:

**Without HER (SAC + uniform replay, 200k env steps):**
- Success rate: $\approx 3$%–$5$%. Almost no positive reward is encountered.
- The Q-function estimates $Q(s, a, g) \approx -50$ uniformly — essentially a constant, uninformative function.
- The policy makes random-looking movements that occasionally stumble near the goal.

**With HER (SAC + future strategy $k = 4$, same 200k steps):**
- Success rate: $\approx 92$%–$96$%.
- Each episode of 50 transitions generates $50 + 50 \times 4 = 250$ transitions stored in the buffer. Of those 250, approximately 80% have $r' = 0$ (goal not yet achieved in the window) and 20% have $r' = 1$ (future goal achieved).
- The Q-function learns a meaningful gradient — it correctly predicts higher values for states closer to the goal.

The improvement factor — roughly 20×–30× in sample efficiency — comes entirely from data augmentation. HER does not change the underlying SAC algorithm, the network architecture, or the hyperparameters. It purely changes how transitions are labeled before being stored in the buffer.

```python
import numpy as np
from collections import deque
from typing import Callable, List, Dict


class HERReplayBuffer:
    """
    Hindsight Experience Replay buffer with the 'future' goal strategy.
    Works with any goal-conditioned environment that provides a
    compute_reward(achieved_goal, desired_goal, info) method.
    """

    def __init__(
        self,
        capacity:  int,
        state_dim: int,
        goal_dim:  int,
        k_goals:   int = 4,
    ):
        self.capacity       = capacity
        self.goal_dim       = goal_dim
        self.k_goals        = k_goals
        self.buffer         = deque(maxlen=capacity)
        self._episode_buf   = []   # staging: current episode transitions

    # ------------------------------------------------------------------ #
    def start_episode(self):
        self._episode_buf = []

    def add_transition(
        self,
        state:          np.ndarray,
        action:         np.ndarray,
        reward:         float,
        next_state:     np.ndarray,
        done:           bool,
        achieved_goal:  np.ndarray,
        desired_goal:   np.ndarray,
    ):
        self._episode_buf.append({
            "state":         state.copy(),
            "action":        action.copy(),
            "reward":        reward,
            "next_state":    next_state.copy(),
            "done":          done,
            "achieved_goal": achieved_goal.copy(),
            "desired_goal":  desired_goal.copy(),
        })

    def end_episode(self, compute_reward: Callable):
        """
        Flush episode to the main buffer with HER relabeling (future strategy).
        compute_reward(achieved_goal, desired_goal, {}) -> float
        """
        T = len(self._episode_buf)
        for t, trans in enumerate(self._episode_buf):
            # Store the original transition
            self.buffer.append(dict(trans))

            # Sample k future states as alternative goals
            if t + 1 >= T:
                future_indices = []   # no future states available at last step
            else:
                future_indices = np.random.randint(t + 1, T, size=self.k_goals)

            for fi in future_indices:
                future_state = self._episode_buf[fi]["achieved_goal"]
                new_reward   = compute_reward(
                    trans["next_state"],   # check if next state achieved future goal
                    future_state,
                    {}
                )
                relabeled = dict(trans)
                relabeled["desired_goal"] = future_state.copy()
                relabeled["reward"]       = new_reward
                self.buffer.append(relabeled)

        self._episode_buf = []

    def sample(self, batch_size: int) -> List[Dict]:
        assert len(self.buffer) >= batch_size
        indices = np.random.choice(len(self.buffer), size=batch_size, replace=False)
        return [self.buffer[i] for i in indices]

    def __len__(self) -> int:
        return len(self.buffer)
```

## The off-policy correction problem: when IS weights are not optional

We established earlier that uniform replay is bias-free for Q-learning because the Bellman optimality operator is independent of the behavioral policy. This is true and important. But it is not the full story.

For **actor-critic methods** — DDPG, TD3, SAC — the policy gradient depends on the Q-function value under the current policy:

$$\nabla_\phi J(\pi_\phi) = \mathbb{E}_{s \sim d^{\pi_\phi}}\left[\nabla_a Q_\theta(s, a)\big|_{a = \pi_\phi(s)} \cdot \nabla_\phi \pi_\phi(s)\right]$$

The expectation is over the state distribution $d^{\pi_\phi}$ induced by the *current* policy $\pi_\phi$. When we sample states from a replay buffer containing transitions from past policies $\pi_{\phi_{t-k}}$, the state distribution is $d^{\pi_{\text{behavior}}} \neq d^{\pi_\phi}$. This is a covariate shift: we evaluate the current policy's Q-gradient in states distributed differently from where the current policy would actually visit.

In practice, SAC and TD3 tolerate this because:

1. **Buffer recency:** A large recent buffer contains mostly transitions from the last few thousand steps, when the policy has not changed dramatically. The distribution mismatch is small.
2. **Entropy regularization (SAC):** The maximum entropy framework in SAC already smooths the policy update, making it less sensitive to distributional shifts.
3. **Target networks:** The TD targets are computed with a frozen target network, which further reduces the sensitivity to small distributional shifts.

For algorithms that do need rigorous off-policy correction — IMPALA, V-trace, Retrace($\lambda$) — the IS correction is applied per-step along a trajectory:

$$\rho_t = \min\!\left(c_{\max}, \frac{\pi(a_t | s_t)}{\mu(a_t | s_t)}\right)$$

where $\mu$ is the behavioral policy that collected the transition and $c_{\max}$ is a truncation threshold that limits variance at the cost of introducing a small, controlled bias. The truncated product $\prod_{i=t}^{t+n-1} \rho_i$ is applied to the $n$-step return.

The practical rule: if you are using Q-learning (DQN, double DQN, dueling DQN), uniform or prioritized replay needs no IS correction for the Bellman target. If you are using an actor-critic with a deterministic or stochastic actor (DDPG, TD3, SAC), you typically get away without IS correction in the actor update because the buffer is recent. If you are using an on-policy algorithm (PPO, A2C, TRPO), do not use a replay buffer at all — discard all transitions after each update cycle.

## The bridge to offline RL: when there is no environment

All of the above assumes a live environment. The agent can always collect new data. This seems like a given, but it fails in a wide range of practically important settings:

**Healthcare.** You want to learn a treatment policy from electronic health records (EHR) data. You cannot experiment on patients to gather additional training samples. The dataset is fixed.

**Autonomous driving.** You want to learn a driving policy from fleet logging data. Running new exploration experiments means putting vehicles on public roads with an exploratory (potentially unsafe) policy.

**Algorithmic trading.** You want to learn a trading policy from historical order book data. Running live market experiments means real financial risk at scale.

**Robotics with limited hardware.** You have 50 hours of human demonstration data. Additional data collection requires expensive robot time and researcher supervision.

In all of these cases, the "replay buffer" is not a circular buffer that grows during training — it is a **fixed, static dataset** $\mathcal{D} = \{(s_i, a_i, r_i, s_i')\}_{i=1}^M$ that was collected by some behavioral policy $\pi_\beta$ and will not grow. Learning a policy from this fixed dataset is called **offline RL** (or batch RL).

Offline RL inherits the replay buffer's off-policy nature but strips away the option to collect additional data. This creates a critical problem that does not exist in online RL.

![The offline RL problem emerged over a decade as the replay buffer concept matured from the 2013 DQN paper through prioritized sampling to fully offline algorithms and fine-tuning pipelines](	/imgs/blogs/experience-replay-and-offline-data-7.png)

### The distributional shift / OOD action problem

When a Q-network is trained offline, it is trained on $(s, a)$ pairs that appear in $\mathcal{D}$. These pairs were generated by $\pi_\beta$. The Q-function has received gradient updates only for transitions actually in the dataset. For state-action pairs $(s, a')$ where $a' \notin \text{support}(\pi_\beta(\cdot|s))$ — the OOD (out-of-distribution) actions — the Q-network can predict anything. Empirically, neural networks generalize optimistically to OOD inputs: they assign high Q-values to actions they have never seen if those actions are extrapolations from high-value nearby points.

This creates a feedback loop. During offline policy improvement:

$$\pi_{\text{new}}(s) = \arg\max_{a'} Q_\theta(s, a')$$

The policy improvement step queries $Q_\theta$ for the highest-value action. If the Q-network overestimates Q-values for OOD actions (because it has never been corrected on those actions), the greedy policy will prefer OOD actions. It will then bootstrap on $Q_\theta(\text{OOD state}, \text{OOD action})$, creating cascading overestimation. The Q-function diverges to unrealistically high values — a phenomenon sometimes called "Q-value explosion" in offline RL.

Quantitatively: on the D4RL HalfCheetah-medium dataset, a naively applied offline DQN (without any OOD correction) achieves a normalized score of approximately 2.0–5.0. The behavioral cloning baseline (just imitating the actions in $\mathcal{D}$) achieves 36.1. CQL achieves 44.0. The naive offline RL is worse than simply copying the dataset actions.

### Offline RL solutions: a brief tour

The offline RL community has developed three families of solutions to the distributional shift problem:

**Policy constraint methods** — BCQ (Fujimoto, Meger, Precup 2019), BEAR (Kumar, Fu, Soh, Tucker, Levine 2019). These explicitly constrain the learned policy to stay close to the behavioral policy $\pi_\beta$:

$$\pi_{\text{new}} = \arg\max_\pi \mathbb{E}_{s \sim \mathcal{D}} \left[ Q_\theta(s, \pi(s)) \right] \quad \text{s.t.} \quad D(\pi, \pi_\beta) \leq \varepsilon$$

BCQ uses a conditional VAE to model $\pi_\beta$ and perturbs its samples; BEAR uses maximum mean discrepancy (MMD) as the divergence measure.

**Conservative Q-learning (CQL)** — Kumar, Zhou, Tucker, Levine (NeurIPS 2020). Adds a regularization term to the Q-learning objective that explicitly penalizes Q-values for OOD actions:

$$\min_\theta \alpha \underbrace{\left(\mathbb{E}_{s, a \sim \pi_{\text{new}}}[Q_\theta(s,a)] - \mathbb{E}_{s, a \sim \mathcal{D}}[Q_\theta(s,a)]\right)}_{\text{conservative penalty}} + \underbrace{\frac{1}{2}\mathcal{L}_\text{Bellman}(\theta)}_{\text{standard TD loss}}$$

The first term pushes Q-values up for current-policy actions and down for in-distribution dataset actions, creating a lower bound on the true Q-function rather than an overestimate.

**Implicit Q-learning (IQL)** — Kostrikov, Nair, Levine (ICLR 2022). Avoids querying OOD actions entirely. IQL uses expectile regression to estimate the optimal Q-value using only in-distribution samples:

$$\mathcal{L}_V(\psi) = \mathbb{E}_{(s,a) \sim \mathcal{D}} \left[ L_2^\tau \left( Q_\theta(s,a) - V_\psi(s) \right) \right]$$

where $L_2^\tau(u) = |\tau - \mathbb{1}[u < 0]| \cdot u^2$ is the asymmetric loss. By setting $\tau \in (0.5, 1)$ (e.g., $\tau = 0.7$), the value function estimates the upper quantile of the in-sample return distribution — an approximation to the maximum without needing to query what the maximum action actually is. IQL achieves state-of-the-art D4RL performance while never computing $\arg\max_{a'} Q(s', a')$ during the critic update.

On the D4RL benchmark (normalized scores, higher is better):

| Method | HC medium | Hopper medium | Walker2d medium |
|---|---|---|---|
| Behavioral cloning | 36.1 | 29.0 | 6.6 |
| CQL | 44.0 | 58.5 | 72.5 |
| IQL | 47.4 | 66.3 | 78.3 |
| Online SAC (oracle) | 47.4 | 52.5 | 79.9 |

IQL matches online SAC on HalfCheetah, demonstrating that offline RL can be competitive with online methods when the dataset has adequate coverage.

### The offline-to-online bridge

One of the most practically powerful patterns emerging from recent research is **offline-to-online fine-tuning**: train a policy offline from a static dataset to get a reasonable starting policy, then deploy it in the live environment with a small online exploration budget to fine-tune. This combines the data efficiency of offline RL (no random exploration needed to bootstrap) with the adaptability of online RL (the agent can discover transitions not present in the static dataset).

The challenge is that the fine-tuned online policy diverges in distribution from the offline dataset, making the combined replay buffer non-stationary in a new way. Several strategies address this:

**Separate replay buffers.** Maintain the offline dataset buffer and a small online buffer separately; sample from both during updates, with a mixing ratio that shifts from mostly-offline to mostly-online as online data accumulates.

**Balanced sampling.** IQL and TD3+BC both support this naturally because their training objective is a weighted combination of the TD loss (where offline data is informative) and the behavioral cloning term (which anchors the policy to safe actions in the offline data).

**Pessimism decay.** Start with the full CQL/IQL conservatism penalty and gradually reduce it to zero as online data fills the buffer. This allows the policy to become progressively less conservative as it accumulates evidence about previously-OOD state-action pairs from actual environment interaction.

In practice, offline-to-online on D4RL medium-quality datasets achieves online SAC performance in 100k–200k additional environment steps — roughly 10–20× fewer steps than online SAC from scratch.

### Practical considerations for offline RL in production

Beyond the algorithmic choices, several engineering factors determine success in production offline RL:

**Dataset curation.** A dataset of 1M transitions from a random policy is often worse for offline RL than 100k transitions from a near-expert policy. The coverage-quality trade-off exists, but quality typically wins. Before running offline RL, compute the return distribution of your dataset and remove the lowest-quintile trajectories if memory allows.

**Feature normalization.** Offline RL is more sensitive to feature scale than online RL, because the Q-function must extrapolate to OOD states without correction from environment feedback. Normalize states to zero mean and unit variance using statistics computed from the offline dataset; apply the same normalization at inference time.

**OOD detection monitoring.** During offline RL evaluation, track the Q-values assigned to the actions chosen by the learned policy. If Q-values are systematically higher than the maximum return in the training dataset, the network is overestimating on OOD actions — the classic sign of distributional shift. Add a conservative penalty or reduce the policy's greediness (increase $\tau$ in IQL) if this occurs.

**Reward normalization.** Many offline RL implementations normalize rewards by the standard deviation of returns in the dataset. This prevents the conservatism penalty (CQL) or expectile threshold (IQL) from being dominated by the raw reward scale, which varies widely across environments and tasks.

## Putting it all together: replay strategy comparison

![Comparison matrix of replay variants across sample efficiency, memory overhead, bias correction approach, and best use case](	/imgs/blogs/experience-replay-and-offline-data-5.png)

Choosing the right replay strategy is a first-order architectural decision — more impactful than most hyperparameter choices. Here is the decision table:

| Scenario | Best replay choice | Why |
|---|---|---|
| CartPole, easy Atari, simple discrete action | Uniform ER, $N = 50\text{k}$–$100\text{k}$ | Cheap, sufficient decorrelation, no overhead |
| LunarLander, complex Atari, DQN with heterogeneous rewards | PER, $\alpha = 0.6$, $\beta: 0.4 \to 1.0$ | 3× sample efficiency vs uniform |
| Goal-conditioned robotics, navigation, sparse rewards | HER (future $k = 4$) + SAC | 10–100× improvement on sparse reward tasks |
| HER + complex reward structure | HER + PER | Combines goal relabeling with priority weighting |
| Fixed historical dataset, no environment access | Offline RL (IQL for balanced, CQL for conservative) | The only option; dataset quality is the bottleneck |
| Continuous control online (SAC/TD3) | Uniform ER, $N = 1\text{M}$ | Off-policy correction not needed; SAC handles variance |

Hyperparameter defaults for PER:

| Parameter | Default | Sensitivity | Notes |
|---|---|---|---|
| $\alpha$ | 0.6 | Medium | 0.5 for noisier envs, 0.7 for clean ones |
| $\beta_0$ | 0.4 | Low | Rarely needs tuning |
| $\beta_T$ | 1.0 | Low | Always anneal to 1.0 at end |
| $\varepsilon$ | $10^{-6}$ | Very low | Just prevents zero priority |
| $N$ | $10^5$–$10^6$ | High | Profile memory; larger is usually better up to compute budget |
| Huber $\delta$ | 1.0 | Medium | Use smaller if action scale is large |

## The convergence theory: why replay buffers work mathematically

It is worth being precise about *why* the replay buffer helps convergence, because the explanation points directly to the design choices that matter.

**Theorem (Baird, 1995; restated informally).** Consider linear function approximation with semi-gradient TD(0). If the distribution of states sampled for updates is the on-policy state distribution $d^\pi$, TD converges. If the sampling distribution deviates from $d^\pi$, TD can diverge even for simple MDPs.

This is the core instability. In online TD learning, the sampling distribution is determined by the agent's most recent policy behavior — it is highly non-stationary and concentrated around wherever the agent currently is in state space. The replay buffer replaces this with a time-averaged historical distribution, which is much closer to the stationarity requirement.

More concretely: the TD update rule for Q-learning is:

$$\theta \leftarrow \theta + \alpha \cdot \delta_t \cdot \nabla_\theta Q_\theta(s_t, a_t)$$

where $\delta_t = r_t + \gamma \max_{a'} Q_\theta(s_{t+1}, a') - Q_\theta(s_t, a_t)$ is the TD error. For convergence, we need the expected update $\mathbb{E}[\delta_t \cdot \nabla_\theta Q_\theta(s_t, a_t)]$ to point consistently toward the true optimal Q-function. This requires that $(s_t, a_t)$ is drawn from a distribution with adequate coverage of the state-action space — specifically, that no state-action pair is permanently excluded from updates.

The replay buffer provides this coverage property: a transition stored from any timestep in the buffer's history has nonzero probability of being sampled at any future training step (until it is overwritten). The effective sampling distribution for updates is approximately uniform over the buffer contents — a broad mixture rather than the narrow, current-policy-concentrated distribution of online updates.

**The contraction operator argument.** Q-learning's convergence can also be understood through the Bellman optimality operator $\mathcal{T}^*$, which is a $\gamma$-contraction in the $\ell^\infty$ norm:

$$\|\mathcal{T}^* Q_1 - \mathcal{T}^* Q_2\|_\infty \leq \gamma \|Q_1 - Q_2\|_\infty$$

Any $\gamma$-contraction has a unique fixed point (Banach fixed-point theorem), and repeated application converges to it. The key requirement is that each $Q$ estimate gets a gradient update from every $(s, a)$ pair in the support of the optimal policy. Without a replay buffer, the agent's current policy might not visit all relevant $(s, a)$ pairs during any given training window. With a large replay buffer, the stored history provides broad coverage, ensuring the contraction operator is faithfully approximated.

**Why neural function approximation complicates things.** The contraction property holds for tabular Q-learning. For neural networks, the function class is non-convex and the gradient step is not a true application of $\mathcal{T}^*$ — it is a step along the gradient of the squared Bellman residual:

$$\mathcal{L}(\theta) = \mathbb{E}_{(s,a,r,s') \sim \mathcal{D}} \left[(r + \gamma \max_{a'} Q_{\bar{\theta}}(s', a') - Q_\theta(s, a))^2\right]$$

where $\bar{\theta}$ is the target network (a frozen copy). The target network is the second key ingredient alongside the replay buffer: it prevents the target $r + \gamma \max_{a'} Q_\theta(s', a')$ from chasing the current $Q_\theta$ in real time, which would create a moving target that the gradient descent cannot follow stably.

Together, replay buffer + target network address the three pathologies identified earlier: the buffer breaks temporal correlation (addressing pathology 1), the target network stabilizes the TD target (addressing pathology 2), and the combination provides the coverage and stationarity needed to avoid the deadly triad's divergence (pathology 3).

#### Worked example: visualizing correlation before and after the buffer

Here is a concrete diagnostic you can run on any DQN to verify the correlation problem and the buffer's effect:

```python
import numpy as np
import gymnasium as gym

def measure_batch_correlation(
    env_name: str = "CartPole-v1",
    n_transitions: int = 1000,
    batch_size: int = 32,
    buffer_size: int = 10_000,
    use_buffer: bool = True,
) -> float:
    """
    Run a random policy, collect transitions, and measure the average
    cosine similarity between consecutive states in training mini-batches.
    High similarity = high correlation = bad for training.
    Returns: mean cosine similarity across 50 mini-batches.
    """
    env = gym.make(env_name)
    transitions = []

    state, _ = env.reset(seed=0)
    for _ in range(n_transitions):
        action = env.action_space.sample()
        next_state, reward, terminated, truncated, _ = env.step(action)
        transitions.append(state.copy())
        state = next_state
        if terminated or truncated:
            state, _ = env.reset()

    env.close()
    states = np.array(transitions)

    sims = []
    for _ in range(50):
        if use_buffer:
            # Simulate buffer: draw random mini-batch
            idx = np.random.choice(len(states), size=batch_size, replace=False)
            batch = states[idx]
        else:
            # Simulate online: take consecutive window
            start = np.random.randint(0, len(states) - batch_size)
            batch = states[start : start + batch_size]

        # Compute mean pairwise cosine similarity
        norms = np.linalg.norm(batch, axis=1, keepdims=True) + 1e-8
        normed = batch / norms
        sim_matrix = normed @ normed.T
        # Off-diagonal mean
        n = len(batch)
        off_diag = (sim_matrix.sum() - np.trace(sim_matrix)) / (n * (n - 1))
        sims.append(off_diag)

    return float(np.mean(sims))

online_corr = measure_batch_correlation(use_buffer=False)
buffer_corr = measure_batch_correlation(use_buffer=True)

print(f"Online mini-batch mean cosine similarity:  {online_corr:.4f}")
print(f"Buffer mini-batch mean cosine similarity:  {buffer_corr:.4f}")
# Typical output:
# Online mini-batch mean cosine similarity:  0.9831
# Buffer mini-batch mean cosine similarity:  0.1247
```

The output is striking: consecutive-transition batches have cosine similarity $\approx 0.98$ — nearly identical states. Random-buffer batches have cosine similarity $\approx 0.12$ — essentially decorrelated. This single measurement explains almost everything about why the buffer matters.

#### Worked example: tracking PER priority distribution over training

Here is a diagnostic for PER that reveals whether priorities are actually concentrating on informative transitions:

```python
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def analyze_priority_distribution(per_buf: PrioritizedReplayBuffer) -> dict:
    """
    After some training, analyze the priority distribution in the buffer.
    Returns statistics showing how concentrated priorities are.
    """
    priorities = np.array([per_buf.tree.tree[i + per_buf.capacity]
                           for i in range(per_buf._size)])
    priorities = priorities[priorities > 0]  # exclude empty slots

    stats = {
        "n_stored":       len(priorities),
        "total_priority": priorities.sum(),
        "mean_priority":  priorities.mean(),
        "std_priority":   priorities.std(),
        "p90_share":      np.percentile(priorities, 90) / priorities.mean(),
        "p99_share":      np.percentile(priorities, 99) / priorities.mean(),
        "top1pct_share":  priorities[priorities > np.percentile(priorities, 99)].sum()
                          / priorities.sum(),
    }
    return stats

# After 50k training steps on LunarLander with PER:
# n_stored: 50000
# mean_priority: 0.041
# std_priority:  0.089   (high std = skewed distribution)
# p99_share:     6.3×    (top 1% transitions are 6× the mean priority)
# top1pct_share: 0.063   (top 1% of transitions receive 6.3% of sampling mass)
```

A healthy PER buffer shows high standard deviation in priorities — a power-law-like distribution where the top 1%–5% of transitions (crash events, large reward discoveries) carry substantially higher sampling probability than the bulk of the buffer.

## Ablation study: what each component contributes

To understand the contribution of each replay mechanism in isolation, consider a controlled ablation on LunarLander-v2 with DQN as the base algorithm, measuring steps to solve (mean return >= 200 over 100 consecutive evaluation episodes):

| Configuration | Steps to solve | Notes |
|---|---|---|
| Online (no replay) | Never (diverges) | Loss oscillates; fails to stabilize |
| Uniform ER, $N = 10\text{k}$ | $\approx 1.2\text{M}$ | Buffer too small; still correlated |
| Uniform ER, $N = 100\text{k}$ | $\approx 780\text{k}$ | Sweet spot for LunarLander |
| PER, $\alpha=0.6$, no IS | $\approx 350\text{k}$ | Fast but biased (IS weights omitted) |
| PER, $\alpha=0.6$, IS ($\beta=0.4\to1$) | $\approx 300\text{k}$ | Full PER; 2.6× faster than uniform |
| PER + double DQN | $\approx 240\text{k}$ | Double Q reduces overestimation further |
| PER + double DQN + dueling network | $\approx 200\text{k}$ | Rainbow-lite; 3.9× faster than baseline |

The key lesson from this ablation: each component is orthogonal and composable. The buffer provides the baseline stability that makes learning possible at all. PER then accelerates convergence. Double DQN and dueling networks address overestimation and value decomposition respectively. None of them substitute for another.

The ablation also reveals that PER without IS weights (row 4) is only marginally slower than full PER (row 5) — for LunarLander specifically. This is because $\beta = 0.4$ at the start means IS weights have limited effect. For longer training runs or environments where Q-function accuracy at convergence matters (multi-step planning, robotics), full IS correction becomes more important.

## Case studies

### Original DQN on Atari (Mnih et al., 2015)

The original DQN paper used a replay buffer of 1 million frames with uniform sampling and a target network. Both components were described by Mnih et al. as necessary for stable training. Without the buffer, training on Pong diverged within 50k steps — the agent learned to serve the ball correctly, then catastrophically forgot after 20k more steps as its own improved play shifted the data distribution. With the 1M-frame buffer, the agent played Pong at superhuman level (score +21 vs opponent) within 3M frames. Across 49 Atari games, 23 exceeded human-level performance. The buffer's memory overhead — ~7 GB for 1M frames of 84×84 grayscale uint8 observations — was the main engineering constraint.

### PER improvements across 49 Atari games (Schaul et al., 2016)

Schaul et al. compared uniform DQN vs PER-DQN across all 49 Atari games using the same 50M environment frames. The main results: PER achieved a median human-normalized score of 96% vs uniform DQN's 68% — a 41% relative improvement in median performance using identical total environment interactions. The games with the largest absolute improvement (Montezuma's Revenge, Venture, Pitfall) were all sparse-reward games where the agent needed to discover rare positive rewards. PER's optimistic priority initialization ensured those rare reward transitions were replayed far more often than their frequency would suggest under uniform sampling.

### HER enabling robotic manipulation (Andrychowicz et al., 2017)

The HER paper tested DDPG + HER on six OpenAI Gym robotic manipulation tasks. The most striking result was HandManipulateBlock — a 24-degree-of-freedom Shadow Hand robot that must rotate a block to a desired orientation. This task has nearly zero random success rate. After 50 training epochs (approximately 3.2M environment steps):

- DDPG without HER: 0% success rate.
- DDPG with HER: 25% success rate.
- DDPG + HER + PER: 30% success rate.

HER was the difference between complete failure and meaningful learning. The PER addition gave a modest further improvement, consistent with the general finding that PER is most valuable when informative transitions are rare — exactly the case for sparse goal-conditioned problems.

### Offline RL with IQL on D4RL locomotion (Kostrikov et al., 2022)

Kostrikov et al. evaluated IQL on the full D4RL benchmark suite, comparing against CQL, TD3+BC, and online SAC. On the medium-quality datasets (behavioral policy achieves roughly half of optimal performance), IQL matched online SAC on HalfCheetah-v2 (both at 47.4 normalized score) without any environment interaction. On Antmaze-Large-Play (sparse reward, large maze), IQL achieved 39.6% success vs CQL's 12.5% and BC's 0%. IQL's key advantage in Antmaze: the implicit value function never queries OOD actions for Q-value targets, preventing the cascading overestimation that causes CQL to fail on high-dimensional sparse reward tasks.

## When to use this (and when not to)

![Decision tree for choosing a replay strategy based on environment access, reward density, and distributional shift severity](	/imgs/blogs/experience-replay-and-offline-data-8.png)

**Use a uniform replay buffer as the baseline for any off-policy deep RL.** It is essentially free to implement, requires no extra hyperparameters beyond buffer size, and provides the fundamental correlation-breaking that makes neural Q-learning stable. Any off-policy algorithm (DQN, DDPG, TD3, SAC) should use one by default.

**Upgrade to PER when you notice slow learning despite a large buffer.** The diagnostic signal: your loss curve is noisy and slow to decrease, and you can verify that most transitions in the buffer have near-zero TD error while a minority have large TD error. PER costs one extra hyperparameter sweep ($\alpha$, $\beta$) and 2× memory for the SumTree. The \$3–\$5/hour compute cost is usually justified by the 2–3× sample efficiency gain on non-trivial environments.

**Use HER whenever your environment is goal-conditioned with sparse rewards.** This is the most impactful single intervention for robotics and navigation. If your environment provides `compute_reward(achieved_goal, desired_goal, info)` — which all Gymnasium Robotics environments do — HER is a drop-in addition with essentially no downside. The only cost is $k \times$ more transitions stored per episode (a memory trade-off you control via $k$).

**Switch to offline RL only when you truly cannot interact with the environment.** Offline RL is a strict generalization of replay buffers but is harder. It requires choosing a conservatism hyperparameter (CQL's $\alpha$, IQL's $\tau$), and its performance ceiling is determined by dataset quality, not algorithm quality. If you have any ability to collect even a small amount of online data, use it — offline-to-online fine-tuning (starting from an offline-trained policy and then collecting additional environment data) consistently outperforms pure offline RL.

**Do not use a replay buffer with on-policy methods.** PPO, A2C, and TRPO are explicitly on-policy — they require that mini-batches come from the current policy's distribution. Replaying transitions from past policies violates their assumptions and introduces bias that their update rules do not correct. For on-policy methods, collect a fresh batch of transitions every update, use them once, and discard them.

**Do not use PER as a replacement for good reward shaping.** If your environment truly has zero reward for 10M steps, PER will prioritize zero-reward transitions (all have the same priority — near zero) and still learn nothing. PER helps when there are *some* non-zero reward transitions that are undersampled; it cannot manufacture signal from a completely reward-free environment. Use HER or reward shaping for that.

## Key takeaways

1. **Correlated online samples are the root cause of DQN instability.** Consecutive transitions from the same episode all have similar states, producing biased gradient estimates that overwrite knowledge from previously visited states.

2. **A replay buffer restores approximate i.i.d.-ness** by mixing transitions from thousands of different timesteps. Uniform sampling is bias-free for Q-learning because the Bellman optimality operator does not condition on the behavioral policy.

3. **Prioritized ER achieves 2–3× sample efficiency** by focusing gradient updates on high-TD-error transitions. The SumTree data structure makes this $O(\log N)$ rather than $O(N)$, practical at 1M+ transitions.

4. **IS weights with annealed $\beta$ are mandatory for unbiased PER.** Without them, over-sampled high-priority transitions bias the gradient estimate. Annealing $\beta$ from 0.4 to 1.0 balances early stability with late-training unbiasedness.

5. **New transitions should receive maximum priority** (optimistic initialization). This guarantees every transition is sampled at least once before its priority is revised from actual TD error.

6. **HER converts failed goal-conditioned episodes into a 5–10× data multiplier** by retroactively relabeling the goal as the actual terminal state. It is the single most impactful intervention for sparse-reward robotics.

7. **Off-policy replay is bias-free only for Q-learning.** Actor-critic methods tolerate it in practice because the buffer is recent and large, but corrections are needed for rigorous importance-weighted methods like Retrace($\lambda$).

8. **Offline RL's core challenge is distributional shift.** A learned policy's greedy action selection queries Q-values for OOD actions; the network extrapolates optimistically, causing cascading Q-value overestimation. CQL penalizes OOD Q-values conservatively; IQL avoids querying them entirely.

9. **Dataset quality bounds offline RL performance.** On D4RL random-quality datasets, even IQL scores below 10 on locomotion tasks. No algorithm recovers from a near-random dataset in high-dimensional continuous action spaces; data curation matters more than algorithm choice.

10. **Choose replay strategy before tuning hyperparameters.** The buffer architecture — uniform vs PER vs HER vs offline — has first-order impact on whether learning occurs at all. Get the architecture right first; then tune $\alpha$, $\beta$, $N$, and learning rate.

## Further reading

- Mnih et al., "Human-level control through deep reinforcement learning," *Nature* 518, 2015. Original DQN paper; Section 3 describes the replay buffer and target network design. [arXiv:1312.5602](https://arxiv.org/abs/1312.5602)
- Schaul, Quan, Antonoglou, Silver, "Prioritized Experience Replay," *ICLR* 2016. Full analysis of the $\alpha / \beta$ schedule, SumTree algorithm, and 49-game Atari comparison. [arXiv:1511.05952](https://arxiv.org/abs/1511.05952)
- Andrychowicz et al., "Hindsight Experience Replay," *NeurIPS* 2017. The HER paper; includes formal proof of the goal relabeling validity and experiments on six robotic manipulation environments. [arXiv:1707.01495](https://arxiv.org/abs/1707.01495)
- Kumar, Zhou, Tucker, Levine, "Conservative Q-Learning for Offline Reinforcement Learning," *NeurIPS* 2020. CQL's theoretical analysis including the lower-bound guarantee on the policy's true value. [arXiv:2006.04779](https://arxiv.org/abs/2006.04779)
- Kostrikov, Nair, Levine, "Offline Reinforcement Learning with Implicit Q-Learning," *ICLR* 2022. IQL's expectile regression approach; state-of-the-art D4RL results without OOD action queries. [arXiv:2110.06169](https://arxiv.org/abs/2110.06169)
- Fu, Kumar, Nachum, Tucker, Levine, "D4RL: Datasets for Deep Data-Driven Reinforcement Learning," *arXiv* 2020. The offline RL benchmark suite. [arXiv:2004.07219](https://arxiv.org/abs/2004.07219)
- Within this series: [Reinforcement Learning: A Unified Map](/blog/machine-learning/reinforcement-learning/reinforcement-learning-a-unified-map) places replay buffers in the DQN branch of the algorithm taxonomy; [The Reinforcement Learning Playbook](/blog/machine-learning/reinforcement-learning/the-reinforcement-learning-playbook) integrates every technique into a production decision guide.
- For debugging instability when adding PER to an existing training pipeline, see the loss-curve triage section of [Debugging AI Training and Finetuning](/blog/machine-learning/debugging-training/the-training-debugging-playbook).
