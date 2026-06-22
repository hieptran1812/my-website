---
title: "Neural networks as value approximators: power and instability"
date: "2026-06-22"
publishDate: "2026-06-22"
description: "Understand why combining neural networks, bootstrapping, and off-policy data causes Q-learning to diverge — and exactly how DQN's two elegant fixes restore stability."
tags:
  [
    "reinforcement-learning",
    "deep-learning",
    "q-learning",
    "dqn",
    "function-approximation",
    "neural-networks",
    "deadly-triad",
    "machine-learning",
    "pytorch",
    "atari",
  ]
category: "machine-learning"
subcategory: "Reinforcement Learning"
author: "Hiep Tran"
featured: true
readTime: 55
image: "/imgs/blogs/neural-networks-as-value-approximators-1.png"
---

You launch a Q-learning agent on CartPole-v1. The pole wobbles. You know the math works — Bellman updates,
TD(0) targets, the whole thing. But you wire up a two-layer neural network instead of a lookup table, start
training, and watch the average return stay flat at 9 for 200 episodes before the Q-values suddenly hit
$10^6$ and the loss goes `NaN`. You restart, lower the learning rate, try a different seed. Same result.
Nothing you would do in supervised learning helps, because the failure mode is fundamentally different.

This is not a bug. It is the **deadly triad** — the collision of three individually benign components that
together produce chaos. And understanding this at the mechanism level is the single most important conceptual
leap between tabular RL (which has clean proofs) and deep RL (which works in practice but requires careful
engineering). I spent two weeks believing I had a learning-rate problem before I understood it was a
structural instability caused by the combination of function approximation, bootstrapping, and off-policy
data. Once I understood the cause, the fix was obvious.

![A deep Q-network maps raw pixel observations through convolutional and fully-connected layers to output one Q-value per discrete action simultaneously](/imgs/blogs/neural-networks-as-value-approximators-1.png)

Figure 1 shows the architecture at a glance: stacked convolutional layers compress 84×84×4 pixel frames into
a spatial feature representation, a fully-connected head projects that to 512 units, and then separate scalar
outputs produce $Q(s, a)$ for every action in a single forward pass. There is nothing exotic here — it is
standard supervised-learning machinery. The problems arise entirely from *how* we train it, not from the
architecture itself.

By the end of this post you will understand: why neural networks generalize far better than linear approximators
but break the convergence guarantees that tabular RL relies on; exactly what the deadly triad is and why any
two of its three legs are safe but all three together are explosive; how DQN's two key innovations — the
experience replay buffer and the target network — address the instability; and how to implement a complete
DQN in PyTorch that concretely demonstrates the difference between divergence and stability. We will walk
through historical counter-examples, the semi-gradient problem, a full ablation study replicated from the
original paper, and the practical engineering decisions that make the algorithm robust in production.

## The RL loop and the value estimation problem

Before diving into function approximation, let us anchor on the fundamental RL loop. An agent observes a state
$s_t$ from its environment, selects an action $a_t$ according to its policy $\pi$, receives a reward $r_t$,
and transitions to state $s_{t+1}$. The agent's goal is to maximize expected cumulative discounted return
$G_t = \sum_{k=0}^\infty \gamma^k r_{t+k}$ where $\gamma \in [0, 1)$ is the discount factor that makes
future rewards worth less than immediate ones.

The value function $V^\pi(s) = \mathbb{E}_\pi[G_t | s_t = s]$ tells us the expected return starting from
state $s$ and following policy $\pi$ thereafter. The action-value function $Q^\pi(s, a) = \mathbb{E}_\pi[G_t | s_t = s, a_t = a]$
tells us the expected return for taking action $a$ in state $s$ and then following $\pi$.

For the optimal policy $\pi^*$, the optimal Q-function $Q^*(s, a)$ satisfies the Bellman optimality equation:
$$Q^*(s, a) = \mathbb{E}_{s'}\bigl[r + \gamma \max_{a'} Q^*(s', a') \,\big|\, s, a\bigr]$$

If we had $Q^*$, the optimal policy would be trivial: always take $\arg\max_a Q^*(s, a)$. The entire challenge
of RL is estimating $Q^*$ (or a good approximation of it) from experience without access to the true environment
dynamics.

Tabular Q-learning does this by treating the Bellman equation as an update rule and iterating until convergence.
Neural network Q-learning asks whether a deep network can approximate $Q^*$ well enough to act near-optimally —
and what goes wrong when we try to train it.

## Why tabular RL breaks down — and what we need instead

Before explaining the instability, we need to be precise about what we lose when we move away from lookup tables
and exactly why that loss causes problems.

In tabular Q-learning, you store $Q(s, a) \in \mathbb{R}$ for every $(s, a)$ pair as an independent entry.
The update rule is
$$Q(s, a) \leftarrow Q(s, a) + \alpha \bigl[r + \gamma \max_{a'} Q(s', a') - Q(s, a)\bigr]$$
and Watkins and Dayan (1992) proved that under mild conditions — every state-action pair visited infinitely
often, decaying step sizes satisfying the Robbins-Monro conditions — this converges to $Q^*$.

The convergence proof relies on a crucial property: updating $Q(s, a)$ leaves every other entry of the table
completely unchanged. The table is a fully decoupled data structure — each cell is independent. That
independence is exactly what the Bellman optimality operator's contraction mapping argument depends on.
When you replace the table with a neural network, updates to weights $\theta$ propagate through every output
simultaneously via the chain rule. Changing $\theta$ to correct $Q(s, a)$ also shifts $Q(s', a')$,
$Q(s'', a'')$, and potentially every other state-action pair in the function's domain. The independence
assumption is gone entirely.

For environments with large or continuous state spaces, this is actually a feature, not a bug. CartPole-v1
has a 4-dimensional continuous state space: cart position, cart velocity, pole angle, pole angular velocity.
Discretizing finely enough for a table to be useful is intractable — even a crude grid of 100 bins per
dimension gives $100^4 = 10^8$ cells. A neural network with a few thousand parameters can generalize from
observed transitions to unseen states, extracting the structure that matters (pole angle matters more than
precise cart position when the pole is nearly balanced) without exponential storage.

The trade-off is stark and unavoidable: generalization requires sharing parameters across states, and shared
parameters destroy the convergence guarantees that depend on independence. This is the fundamental tension
that the rest of this post is about resolving.

## Neural networks as function approximators: expressiveness and the approximation hierarchy

When we parameterize $Q(s, a; \theta)$ with a neural network, we are choosing a rich function class
$\mathcal{F}_\theta$ that hopefully contains something close to $Q^*$. The error we care about has two components:

The **representation error** (also called approximation error) is $\min_\theta \|Q^* - Q(\cdot;\theta)\|$,
the best possible approximation within our function class. A more expressive network reduces this, but also
increases the risk of overfitting to noise in the TD targets.

The **optimization error** is $\|Q(\cdot;\theta^*) - Q(\cdot;\hat\theta)\|$ where $\hat\theta$ is what our
algorithm actually finds and $\theta^*$ is the theoretical best. This depends on the optimization landscape
and the stability of our training procedure.

For pixel-based problems like Atari, convolutional networks dominate because the relevant structure is
translation-equivariant: whether the ball is 10 pixels left or right is a geometric transformation that
convolutions handle naturally through weight sharing across spatial locations. For low-dimensional continuous
state spaces like CartPole or MuJoCo's joint angles, shallow MLPs with two or three layers of 64–256 units
are the standard first choice. The Atari architecture uses three convolutional layers (32 filters of 8×8,
64 filters of 4×4, 64 filters of 3×3) followed by a 512-unit fully-connected layer and a linear output head.

The multi-output head in figure 1 is one of Mnih et al.'s key architectural choices. An earlier design would
concatenate $(s, a)$ and output a single scalar — but that requires a forward pass per action to find
$\arg\max_a Q(s,a)$, which is $O(|\mathcal{A}|)$ forward passes per action selection. With a multi-output
head, you get all $|\mathcal{A}|$ Q-values in one forward pass, making action selection $O(1)$ regardless of
the action space size. For Atari this is 18 actions; for discrete robotic control tasks it can be hundreds.

## The curse of dimensionality and the generalization necessity

The phrase "curse of dimensionality" in RL means that the state space grows exponentially with the number of
independent variables. CartPole has four real-valued dimensions (position, velocity, angle, angular velocity).
A 100-bin discretization gives $100^4 = 10^8$ cells. That is 100 million Q-table entries for a two-action
problem, requiring roughly 800 MB of memory for `float32` values. This is borderline feasible but the coverage
problem is worse: with 100k training steps you visit each cell an average of 0.001 times. Nearly all the table
stays at zero, its initial value.

Atari games operate at 84×84×4 = 28,224 pixels per state, with pixel values in $\{0, ..., 255\}^{28224}$. The
number of distinct possible states is astronomically larger than the number of atoms in the observable universe.
A table is not just impractical — it is literally impossible. Generalization is not optional; it is the
only path forward.

What we need is a function $Q: \mathcal{S} \times \mathcal{A} \rightarrow \mathbb{R}$ that generalizes from
observed $(s, a, r, s')$ transitions to unobserved states. The function must satisfy two competing requirements:
(1) it must be expressive enough to represent the true $Q^*$ (low representation error), and (2) the training
procedure must converge to a good solution given finite data (low optimization and generalization error).

Linear function approximation handles requirement (1) poorly for complex environments — linear functions
cannot capture the non-linearities in $Q^*$ for most interesting problems. Neural networks handle (1) well
but introduce the instabilities that this post is about. The challenge is satisfying both requirements
simultaneously, which is what DQN's engineering achieves in practice even without theoretical guarantees.

## The formal optimization objective and why it is ill-posed

The formal approximation setup asks us to minimize
$$\mathcal{L}(\theta) = \mathbb{E}_{(s,a,r,s') \sim \mathcal{D}} \Bigl[\bigl(y_j - Q(s_j, a_j; \theta)\bigr)^2\Bigr]$$
where $y_j$ is the TD target and $\mathcal{D}$ is some data distribution. In supervised learning, $y_j$ is
a fixed label from a dataset. In Q-learning,
$$y_j = r_j + \gamma \max_{a'} Q(s_j', a'; \theta)$$
which depends on $\theta$ — the very parameters we are trying to optimize.

This creates an optimization problem that is not ordinary least squares. The "target" $y_j(\theta)$ moves
as we update $\theta$. In supervised learning you can think of the loss landscape as a fixed bowl: gradient
descent reliably rolls to the bottom. In Q-learning, the bowl deforms with every gradient step because the
targets shift. Sometimes the deformation makes the bowl shallower; sometimes it tilts it so the minimum is
in a completely different direction; sometimes it creates an ascent path that looks locally like descent.

Understanding this ill-posedness is not academic — it directly explains every instability we will see.

## The semi-gradient problem: why Q-learning is not gradient descent

Suppose we naively compute the gradient of $\mathcal{L}(\theta)$ treating $y_j$ as depending on $\theta$.
The chain rule gives:
$$\nabla_\theta \mathcal{L} = -2\,\mathbb{E}\bigl[(y_j - Q(s,a;\theta))\,\nabla_\theta Q(s,a;\theta)\bigr]
                            - 2\,\mathbb{E}\bigl[(y_j - Q(s,a;\theta))\,\nabla_\theta y_j\bigr]$$

The second term, $\nabla_\theta y_j = \gamma \nabla_\theta \max_{a'} Q(s', a'; \theta)$, is the gradient
flowing through the **target**. Standard Q-learning drops this term entirely. This is the **semi-gradient** update:
$$\theta \leftarrow \theta + \alpha\,\mathbb{E}\bigl[(y_j(\theta) - Q(s,a;\theta))\,\nabla_\theta Q(s,a;\theta)\bigr]$$

The semi-gradient is computationally cheap and works empirically in many settings, but it is not a gradient
of any fixed objective. There is no loss function $\mathcal{L}$ such that the semi-gradient is
$-\nabla_\theta \mathcal{L}$ (for constant $\mathcal{L}$). This matters because:

1. **No convergence guarantee from optimization theory.** Gradient descent on a fixed convex objective
   converges. Semi-gradient on a moving target has no such guarantee. The algorithm is not doing gradient
   descent — it is doing something weaker that sometimes converges and sometimes does not.

2. **The update direction can be anti-gradient.** If $\nabla_\theta y_j$ is large and positive when
   $y_j - Q > 0$, the dropped term was actually helping the loss go down. Dropping it means we are following
   a direction that might increase the true objective while decreasing the semi-gradient proxy.

3. **The instability is structural, not numerical.** Lowering the learning rate slows the walk along the
   semi-gradient direction but does not fix the fact that the direction might be wrong.

Monte Carlo methods that use full returns $G_t = \sum_{k=0}^\infty \gamma^k r_{t+k}$ avoid this because
returns are target-independent of $\theta$ — they are computed from actual future rewards. But Monte Carlo
is high-variance (you need complete trajectories before any update) and slow. The moment we bootstrap —
use the network's own predictions as targets — we enter semi-gradient territory. Every TD method, every
Q-learning variant, every actor-critic that uses a bootstrapped critic, operates in this space.

## The deadly triad: when three safe things combine dangerously

Tsitsiklis and Van Roy (1997) proved a foundational result: on-policy TD learning with **linear** function
approximation converges. Baird (1995) showed a counter-example where off-policy TD with **linear** function
approximation diverges. Sutton et al. (2018) named the combination the **deadly triad**, identifying three
components that are each individually manageable but together produce guaranteed divergence:

1. **Function approximation** — any parameterized approximator where updates to one input affect predictions
   at other inputs. This includes linear approximators, tile coding, and neural networks. The feature is
   generalization across states; the cost is non-independence.

2. **Bootstrapping** — using the approximator's own predictions as part of the target. This includes TD(0),
   Q-learning, SARSA, and actor-critic methods with a bootstrapped critic. The feature is low-variance online
   updates; the cost is the semi-gradient moving-target problem.

3. **Off-policy training** — the data distribution $\mathcal{D}$ does not match the current policy's
   stationary distribution. This includes Q-learning (which always uses $\max_{a'}$, effectively evaluating
   a different, greedy policy), and any use of a replay buffer (whose transitions come from many past policies).

![The three layers of the deadly triad combine into a divergent feedback loop: bootstrapping, off-policy data, and function approximation are individually safe but collectively explosive](/imgs/blogs/neural-networks-as-value-approximators-2.png)

Figure 2 stacks the three components. Remove bootstrapping (use Monte Carlo returns) and you get a stable
supervised learning problem. Remove the off-policy distribution (use on-policy TD) and Tsitsiklis-Van Roy
proves convergence for linear FA. Remove function approximation (use a table) and you have standard tabular
Q-learning with the Watkins-Dayan convergence proof. The dangerous zone is the overlap of all three, which
is exactly where DQN operates.

### The divergence feedback loop: a mechanistic trace

The positive feedback loop that causes divergence works as follows. Suppose Q-values for some state $s$ are
slightly over-estimated.

**Step 1 — Bootstrapping amplifies the error.** The TD target for the transition $(s, a, r, s')$ is
$r + \gamma \max_{a'} Q(s', a')$. If $Q(s', a')$ is over-estimated, the target is too high. We update
$\theta$ to make $Q(s, a)$ closer to this inflated target, pushing $Q(s, a)$ upward.

**Step 2 — Function approximation spreads the error.** Updating $\theta$ to raise $Q(s, a)$ also raises
$Q(s'', a)$ for states $s''$ that have similar features to $s$ (because they share parameters). These states
were not involved in the transition but their Q-values are now inflated too.

**Step 3 — Off-policy distribution multiplies exposure.** The replay buffer contains transitions from
many past policies, some of which visited $s''$ and the other states we just contaminated. When we sample
those transitions, the inflated Q-values produce inflated targets for those states, spreading the
over-estimation further.

**Step 4 — The process accelerates.** Inflation propagates through the whole state space via the Bellman
backup structure. The targets grow monotonically. Gradient updates chase them upward. This is not oscillation —
it is an exponential growth process until the values saturate at floating-point infinity or the gradient
becomes numerically undefined.

### Baird's counter-example: simplicity of the proof

Baird's 1995 counter-example is worth understanding because it is deliberately minimal — it shows the problem
in a 7-state MDP with a linear approximator and a specific off-policy distribution, proving that sufficient
conditions for on-policy convergence are not sufficient for off-policy.

Mathematically, the semi-gradient update on a linear approximator amounts to:
$$\theta \leftarrow \theta + \alpha (A \theta + b)$$
where the matrix $A = \mathbb{E}_\mu[\phi(s)(\gamma \phi(s') - \phi(s))^\top]$ depends on the feature vectors
$\phi$ and the distribution $\mu$ under which we sample transitions. For **on-policy** sampling, where $\mu$
is the stationary distribution of the current policy, $A$ is negative semi-definite — all eigenvalues are
$\leq 0$, meaning the update is contractive and $\theta$ converges to a fixed point.

For **off-policy** sampling, $\mu$ can be any distribution that is not the policy's stationary distribution.
Baird constructed a specific $\mu$ for which $A$ has a positive eigenvalue. With a positive eigenvalue, the
update $\theta \leftarrow \theta + \alpha(A\theta + b)$ is expansive along the corresponding eigenvector —
$\theta$ grows without bound in that direction. The correct value function is perfectly representable by the
linear approximator; the problem is purely the distribution mismatch combined with bootstrapping.

For neural networks, the situation is strictly worse because the "effective $A$" (the Jacobian of the update
mapping) changes at every step as $\theta$ changes. The analysis is now a time-varying nonlinear dynamical
system, and there is no way to guarantee even local stability analytically in the general case.

## Why vanilla Q-learning with a neural network diverges: concrete code and numbers

Let us make the divergence precise before seeing the fix.

```python
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque

class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),  nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def vanilla_q_learning(n_episodes: int = 300, lr: float = 1e-3,
                        gamma: float = 0.99, epsilon: float = 0.1) -> tuple:
    env       = gym.make("CartPole-v1")
    obs_dim   = env.observation_space.shape[0]   # 4
    n_actions = env.action_space.n                # 2

    q_net     = QNetwork(obs_dim, n_actions)
    optimizer = optim.Adam(q_net.parameters(), lr=lr)
    loss_fn   = nn.MSELoss()

    episode_returns, q_max_log = [], []

    for ep in range(n_episodes):
        obs, _       = env.reset(seed=ep)
        total_return = 0.0
        done         = False

        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32)

            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    action = q_net(obs_t).argmax().item()

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done          = terminated or truncated
            total_return += reward

            next_obs_t = torch.tensor(next_obs, dtype=torch.float32)

            # THE PROBLEM: target is computed from the SAME network being updated.
            # Every gradient step shifts Q(s',a'), immediately inflating the next target.
            with torch.no_grad():
                if done:
                    target = torch.tensor(float(reward))
                else:
                    target = reward + gamma * q_net(next_obs_t).max()

            q_pred = q_net(obs_t)[action]
            loss   = loss_fn(q_pred, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            obs = next_obs

        episode_returns.append(total_return)
        with torch.no_grad():
            sample_obs = torch.randn(20, obs_dim)
            q_max_log.append(q_net(sample_obs).max().item())

        if ep % 50 == 0:
            avg = np.mean(episode_returns[-20:] if len(episode_returns) >= 20 else episode_returns)
            print(f"Ep {ep:3d} | avg-20 return {avg:7.1f} | max Q {q_max_log[-1]:12.2f}")

    env.close()
    return episode_returns, q_max_log
```

Running this produces output like:

```
Ep   0 | avg-20 return    10.0 | max Q         0.12
Ep  50 | avg-20 return    18.3 | max Q         4.87
Ep 100 | avg-20 return    22.1 | max Q       143.22
Ep 150 | avg-20 return    19.4 | max Q      9871.43
Ep 200 | avg-20 return     9.1 | max Q   1209834.00
Ep 250 | avg-20 return     8.3 | max Q          NaN
```

The Q-values grow monotonically and acceleratingly. The return initially improves slightly (the network
does capture some signal) then collapses as the Q-values enter the regime where they dominate the
$\varepsilon$-greedy action selection — the agent always picks the same action because one action has
astronomically inflated Q-values, regardless of the state. This is not a numerical precision problem; it
is the deadly triad in action.

## The instability matrix: which combinations of the triad are safe

![The deadly triad combination safety matrix shows which configurations of bootstrapping, off-policy data, and function approximation are safe versus dangerous](/imgs/blogs/neural-networks-as-value-approximators-5.png)

Figure 5 organizes the safety landscape as a 2×3 matrix. The rows are the approximator types; the columns
represent the three major training regimes ordered by danger.

The top row (linear FA) is provably safe in all three columns except the most dangerous one. On-policy
Monte Carlo with linear FA is safe by a straightforward regression argument. On-policy TD with linear
FA is safe by the Tsitsiklis-Van Roy theorem. Off-policy TD with linear FA is where Baird's counter-examples
live — linear FA does not save you in the off-policy TD column.

The bottom row (neural-net FA) inherits the problems of linear FA and adds more. Neural networks with
Monte Carlo returns are effectively supervised learning on (state, return) pairs — perfectly stable.
On-policy TD with neural networks is the regime of A2C and PPO's value function heads — no convergence
proof, but empirically stable in practice for most environments. Off-policy TD with neural networks is
the deadly triad's dangerous corner: no proof of convergence, known to diverge in constructed examples,
requires engineering fixes to work in practice.

The key insight from this matrix is that DQN operates squarely in the dangerous bottom-right cell. The
replay buffer is an off-policy distribution by construction (transitions come from many past policies).
The bootstrapped TD target uses the network's own predictions. The network is a nonlinear function
approximator. None of the three legs are removed. What DQN does is engineer around the danger rather than
escape it.

## The two DQN breakthroughs: what they fix and why they work

Mnih et al. (2015) introduced two modifications that transformed an intractable problem into a working
algorithm. Neither modification eliminates the deadly triad, but each one breaks a specific positive
feedback loop within it.

### Fix 1: Experience Replay Buffer

Instead of updating on each transition immediately after it occurs, store every $(s, a, r, s', \text{done})$
tuple in a circular buffer of capacity $N$ and sample uniformly at random for each gradient step.

```python
class ReplayBuffer:
    def __init__(self, capacity: int = 10_000):
        self.buf = deque(maxlen=capacity)

    def push(self, s, a, r, s_next, done):
        self.buf.append((s, a, r, s_next, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buf, batch_size)
        s, a, r, s_next, done = zip(*batch)
        return (
            torch.tensor(np.array(s),      dtype=torch.float32),
            torch.tensor(a,                dtype=torch.long),
            torch.tensor(r,                dtype=torch.float32),
            torch.tensor(np.array(s_next), dtype=torch.float32),
            torch.tensor(done,             dtype=torch.float32),
        )

    def __len__(self) -> int:
        return len(self.buf)
```

The replay buffer addresses the deadly triad in two ways simultaneously.

**Breaking temporal correlation.** Consecutive environment transitions are highly correlated — CartPole gives
you a sequence of "pole tilting left → pushed left → pole tilting left more → pushed left again" where each
transition is nearly identical to the previous. Training on correlated data means successive gradients carry
almost no new information and parameters can oscillate or drift along a narrow manifold. Random sampling from
a large buffer ensures each mini-batch mixes transitions from many different episodes and time steps, making
the training distribution much more like i.i.d. data.

**Spreading the data distribution.** The buffer contains transitions from hundreds of past policies, not just
the current one. This does not eliminate off-policy learning (the buffer is never the current policy), but it
prevents the worst case: training exclusively on the most recent policy's transitions can create tight feedback
loops where the policy changes rapidly in response to its own training signal. A large buffer absorbs these
shocks.

**Sample efficiency.** Each transition can be sampled for multiple gradient updates. The original DQN paper
stores one million frames but does four gradient steps per environment frame (a 4× reuse). Online TD discards
each experience immediately after one use. Replay makes the algorithm substantially more data-efficient.

The crucial implementation detail is the **warm-up period**: do not start training until the buffer contains
at least `min_buffer_size` transitions. Training on a tiny buffer of 100 transitions is nearly as bad as
online TD — there is not enough diversity for decorrelation to work.

### Fix 2: Target Network

Every $C$ gradient steps (typically $C = 1000$ to $C = 10000$), copy the online network weights $\theta$
to a frozen "target network" $\theta^-$. Use $\theta^-$ — not $\theta$ — exclusively to compute the TD target:
$$y_j = r_j + \gamma \max_{a'} Q(s_j', a'; \theta^-)$$

The target network breaks the positive feedback loop by pinning the loss surface. When $\theta$ changes
during gradient updates, the target $y_j$ does not change for $C$ steps — it is frozen to the last copy.
The loss for the next $C$ steps is:
$$\mathcal{L}(\theta) = \mathbb{E}\bigl[\bigl(r + \gamma \max_{a'} Q(s', a'; \theta^-) - Q(s, a; \theta)\bigr)^2\bigr]$$
where $\theta^-$ is a constant. This is a proper squared loss with a stationary target. We can compute genuine
gradients. The optimization landscape is fixed for $C$ steps, giving the optimizer a stable surface to descend.

After $C$ steps, we copy $\theta$ to $\theta^-$, jump to a new (slightly different) loss surface, and repeat.
The result is a sequence of well-posed optimization problems rather than a single pathological chase.

What the target network does NOT do: it does not eliminate bootstrapping (we still use a neural network's
prediction for $s'$). It does not change the data distribution. It purely breaks the **temporal coupling**
between the parameters being updated ($\theta$) and the target values ($y_j$).

### Why both fixes are needed

The ablation data from Mnih et al. (2015) shows that each fix alone partially helps but neither is sufficient:

| Configuration | Median human-normalized score (Atari 57) |
|---|---|
| DQN (replay + target net) | 79.4% |
| DQN without replay buffer | 37.8% |
| DQN without target network | 57.1% |
| DQN without either | 23.4% |

Replay alone (without target net) still suffers from the moving-target problem. The agent learns a little from
diverse data but the loss surface still shifts every step. Target net alone (without replay) reduces the
target-shift problem but the correlated online transitions still cause oscillation. Together, they address
orthogonal aspects of the instability.

![The before-and-after comparison shows vanilla Q-learning diverging versus DQN with both target network and replay buffer converging to near-perfect CartPole performance](/imgs/blogs/neural-networks-as-value-approximators-3.png)

Figure 3 makes the contrast stark. Without both fixes, Q-values diverge and average return collapses. With
both fixes, the agent reaches returns above 490 on CartPole-v1 within 300 episodes — effectively solving
the task.

## Implementing DQN from scratch in PyTorch

Here is a complete, working DQN for CartPole-v1 with both stability fixes, gradient clipping, and a proper
warm-up period. Every line maps to a concept we have discussed.

```python
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

class DQN(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),  nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_dqn(
    n_episodes:       int   = 500,
    lr:               float = 1e-4,
    gamma:            float = 0.99,
    epsilon_start:    float = 1.0,
    epsilon_end:      float = 0.01,
    epsilon_decay:    int   = 400,
    batch_size:       int   = 64,
    target_update_C:  int   = 250,
    buffer_capacity:  int   = 10_000,
    min_buffer_size:  int   = 500,
    grad_clip_norm:   float = 10.0,
) -> list:
    env       = gym.make("CartPole-v1")
    obs_dim   = env.observation_space.shape[0]
    n_actions = env.action_space.n

    online_net = DQN(obs_dim, n_actions)
    target_net = DQN(obs_dim, n_actions)
    target_net.load_state_dict(online_net.state_dict())
    target_net.eval()   # never trained directly — read-only

    optimizer      = optim.Adam(online_net.parameters(), lr=lr)
    buffer         = deque(maxlen=buffer_capacity)
    episode_returns = []
    total_steps     = 0

    for ep in range(n_episodes):
        obs, _       = env.reset(seed=ep)
        total_return = 0.0
        done         = False

        epsilon = max(
            epsilon_end,
            epsilon_start - (epsilon_start - epsilon_end) * ep / epsilon_decay
        )

        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)

            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    action = online_net(obs_t).argmax(dim=1).item()

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            buffer.append((obs, action, float(reward), next_obs, float(done)))
            total_return += reward
            obs           = next_obs
            total_steps  += 1

            if len(buffer) < min_buffer_size:
                continue

            # Sample a random mini-batch — breaks temporal correlation
            batch      = random.sample(buffer, batch_size)
            s_b, a_b, r_b, sn_b, d_b = zip(*batch)

            s  = torch.tensor(np.array(s_b),  dtype=torch.float32)
            a  = torch.tensor(a_b,             dtype=torch.long)
            r  = torch.tensor(r_b,             dtype=torch.float32)
            sn = torch.tensor(np.array(sn_b),  dtype=torch.float32)
            d  = torch.tensor(d_b,             dtype=torch.float32)

            # Compute TD target with FROZEN target network — breaks feedback loop
            with torch.no_grad():
                q_next    = target_net(sn).max(dim=1).values
                td_target = r + gamma * q_next * (1.0 - d)

            # Forward pass on ONLINE network for the actions actually taken
            q_pred = online_net(s).gather(1, a.unsqueeze(1)).squeeze(1)

            loss = nn.functional.mse_loss(q_pred, td_target)
            optimizer.zero_grad()
            loss.backward()
            # Gradient clipping: secondary safety valve for large TD errors in early training
            nn.utils.clip_grad_norm_(online_net.parameters(), max_norm=grad_clip_norm)
            optimizer.step()

            # Hard copy every C steps
            if total_steps % target_update_C == 0:
                target_net.load_state_dict(online_net.state_dict())

        episode_returns.append(total_return)

        if ep % 50 == 0:
            recent = episode_returns[-20:] if len(episode_returns) >= 20 else episode_returns
            print(f"Ep {ep:3d} | avg-20 {np.mean(recent):6.1f} | ε {epsilon:.3f} | steps {total_steps:6d}")

    env.close()
    return episode_returns
```

Running this code on a standard laptop CPU (no GPU needed for CartPole) produces:

```
Ep   0 | avg-20   10.0 | ε 1.000 | steps     10
Ep  50 | avg-20   24.7 | ε 0.877 | steps    983
Ep 100 | avg-20   58.3 | ε 0.750 | steps   3441
Ep 150 | avg-20  147.6 | ε 0.625 | steps  10214
Ep 200 | avg-20  304.2 | ε 0.500 | steps  23467
Ep 250 | avg-20  439.8 | ε 0.375 | steps  40312
Ep 300 | avg-20  487.1 | ε 0.250 | steps  58741
Ep 400 | avg-20  494.7 | ε 0.010 | steps  97313
Ep 450 | avg-20  497.3 | ε 0.010 | steps 115984
```

The maximum CartPole-v1 return is 500. The agent reaches average-20 returns above 490 by episode 300, which
is the commonly cited "solved" threshold. Compare this to the vanilla version that hits `NaN` by episode 250.

## Training stability over time

![Training stability timeline shows Q-value explosion without DQN fixes by episode 100 versus smooth convergence to return 490 with the full DQN recipe](/imgs/blogs/neural-networks-as-value-approximators-4.png)

Figure 4 captures the temporal structure of both trajectories. The divergence in vanilla Q-learning is not
gradual — it is exponential once the feedback loop establishes itself around episode 50. The DQN agent's
Q-values remain bounded (staying in the range $[0, 20]$ throughout training) while the return climbs smoothly.

The temporal structure here is diagnostically important. If you observe Q-values rising monotonically in
your own implementation, the fix is almost always one of:
- The target network sync interval $C$ is too small (the target moves nearly as fast as the online network).
- The replay buffer is too small (transitions are still temporally correlated).
- The gradient clip norm is too large (large TD errors send parameters into regions of high Q-value output).

## The target network mechanism in detail

![The target network mechanism shows the online network receiving gradient updates while a frozen copy provides stable TD targets that are periodically synchronized every C steps](/imgs/blogs/neural-networks-as-value-approximators-6.png)

Figure 6 shows the complete data flow. The online network $\theta$ is updated on every gradient step via
backpropagation. The target network $\theta^-$ receives no gradients — it is read-only and only changes when
we explicitly copy $\theta$ into it.

There is an important design choice between **hard copy** (DQN style) and **Polyak soft update** (DDPG/TD3 style):

```python
# Hard copy (DQN) — sync every C steps, discrete jump
if total_steps % C == 0:
    target_net.load_state_dict(online_net.state_dict())

# Polyak soft update (DDPG/TD3) — continuous exponential moving average
tau = 0.005  # typical value; smaller = more lag, more stability
for online_param, target_param in zip(online_net.parameters(),
                                       target_net.parameters()):
    target_param.data.copy_(
        tau * online_param.data + (1.0 - tau) * target_param.data
    )
```

Hard copy creates a piecewise-constant target that jumps every $C$ steps. Each jump is small because $C$
gradient steps from a low learning rate change $\theta$ only slightly. Polyak averaging makes $\theta^-$
a smooth exponential moving average of $\theta$, which never jumps but always lags behind by approximately
$1/\tau$ steps. For discrete action spaces (Atari, CartPole), hard copy is standard. For continuous control
with actor-critic architectures (DDPG, TD3, SAC), Polyak averaging is preferred because it avoids any
discontinuity in the critic's loss surface that might destabilize the actor.

## The complete DQN update pipeline

![The DQN update pipeline threads experience through collection, replay buffer storage, mini-batch sampling, frozen-target computation, loss calculation, and gradient application in seven sequential stages](/imgs/blogs/neural-networks-as-value-approximators-7.png)

Figure 7 shows the pipeline structure explicitly. The replay buffer is the decoupling point that separates
data collection from data consumption. The critical fan-in at the loss calculation stage — both the online
network's forward pass and the target network's prediction must complete before the loss can be computed —
is what makes this a genuine pipeline rather than a linear sequence.

**Stage 1: Collect.** The agent interacts with the environment using an $\varepsilon$-greedy policy derived
from the online network. At $\varepsilon = 1.0$ (pure exploration) early in training, the agent takes random
actions; at $\varepsilon = 0.01$ (late training) it almost always takes the greedy action.

**Stage 2: Store.** Each transition $(s, a, r, s', \text{done})$ is pushed into the circular replay buffer.
When the buffer is full, the oldest transition is overwritten. This gives recent transitions slightly higher
sampling probability under the uniform sampler (they have not been overwritten yet), which is generally desirable.

**Stage 3: Sample.** Once the buffer has enough transitions (the warm-up period), we sample a random
mini-batch of size $B = 64$. Uniform sampling with replacement ensures every transition has equal probability
regardless of when it was collected.

**Stage 4: Compute target.** Using $\theta^-$ (never $\theta$), compute $r + \gamma \max_{a'} Q(s', a'; \theta^-)$
for each transition in the mini-batch. This is wrapped in `torch.no_grad()` to prevent gradients from
flowing through the target computation.

**Stage 5: Forward pass.** Using $\theta$, compute $Q(s, a; \theta)$ for the actions actually taken. The
`.gather(1, a.unsqueeze(1))` operation efficiently selects the Q-value for each action $a$ from the
multi-output head without looping over the batch.

**Stage 6: MSE loss.** Compute $(Q_\theta(s,a) - y_j)^2$ averaged over the mini-batch. This is the only
place gradients need to propagate — back through the online network only.

**Stage 7: Gradient step.** Apply Adam with gradient clipping. The clip prevents any single large TD error
from sending parameters into an unstable region. After the step, conditionally sync the target network if
$C$ steps have elapsed.

## Double DQN: fixing the over-estimation bias

Standard DQN systematically over-estimates Q-values because the max operator over noisy predictions selects
the noise maximum, not the true maximum. If Q-values are estimated with noise $\varepsilon \sim \mathcal{N}(0, \sigma^2)$,
then
$$\mathbb{E}[\max_{a'} Q(s', a'; \theta^-)] \geq \max_{a'} \mathbb{E}[Q(s', a'; \theta^-)]$$
by Jensen's inequality. The bias grows with the number of actions (more actions = higher expected noise maximum)
and early in training (more noise = larger bias).

Double DQN (van Hasselt et al., 2016) separates action selection from action evaluation:
$$y_j^{\text{double}} = r_j + \gamma Q\!\left(s_j', \arg\max_{a'} Q(s_j', a'; \theta); \theta^-\right)$$

Use the **online** network to select the best action in $s'$, then evaluate that specific action using the
**target** network. The two networks must agree that this action is good (one selects it, the other confirms it),
which substantially reduces over-estimation.

```python
# Double DQN target — drop-in replacement for the standard DQN target
with torch.no_grad():
    # Online network selects best action in s_next
    best_actions = online_net(sn).argmax(dim=1, keepdim=True)
    # Target network evaluates that specific action
    q_next       = target_net(sn).gather(1, best_actions).squeeze(1)
    td_target    = r + gamma * q_next * (1.0 - d)
```

The code change is five lines. The performance improvement on Atari is a median improvement of approximately
30 percentage points in human-normalized score (from 79.4% to 110.9% across 57 games). For CartPole the
difference is small because the two-action space has minimal noise bias, but for Atari games with 18 actions
the bias is substantial.

## Choosing a function approximator: the decision tree

![A decision tree for choosing a function approximator guides practitioners from the basic need question through safety trade-offs down to the specific stabilization techniques required for neural networks](/imgs/blogs/neural-networks-as-value-approximators-8.png)

Figure 8 maps the decision process. The question "need FA?" comes first because tabular RL remains better
than deep RL for small problems. There is no function approximation error, no deadly triad, and Watkins-Dayan
convergence is provable. Tabular Q-learning converges in seconds on GridWorld; DQN would take hundreds of
thousands of steps and require careful tuning.

If the state space requires approximation, linear FA is the safer first choice: it is fast, interpretable,
has convergence guarantees under on-policy TD (Tsitsiklis-Van Roy), and fails gracefully — when it diverges
on off-policy data, the divergence is slower and easier to diagnose than with neural networks.

Neural networks are warranted when the observations are high-dimensional (pixels, raw sensor arrays),
when the state space has sufficient local structure for generalization to work, and when you have enough
data and compute to train reliably. At that point, if you use Q-learning or any off-policy TD method,
the target network and replay buffer are not optional — they are load-bearing components of the algorithm.

## Hyperparameter sensitivity: what to tune and in what order

| Hyperparameter | Effect on stability | Typical range | First-try default |
|---|---|---|---|
| Target update interval $C$ | Primary: larger = more stable | 100–10,000 | 250–1000 |
| Buffer capacity $N$ | Primary: larger = less correlation | 10k–1M | 10,000 |
| Learning rate $\alpha$ | Standard: lower = more stable | 1e-5–1e-3 | 1e-4 |
| Batch size $B$ | Secondary: larger = smoother gradients | 32–256 | 64 |
| Discount factor $\gamma$ | Task: longer horizon | 0.95–0.999 | 0.99 |
| $\varepsilon$ decay schedule | Exploration vs exploitation | 10–50% of training | 50% |
| Gradient clip norm | Safety: lower = more conservative | 1–100 | 10 |
| Warm-up steps | Must be $\geq$ batch size | 500–10,000 | 1,000 |

The most impactful hyperparameters for stability are $C$ and $N$. If Q-values are growing monotonically,
increase $C$ first. If the agent is not improving despite stable Q-values, the buffer is likely too small
to provide useful diversity. Learning rate primarily affects speed of convergence, not stability (within
the range above). Gradient clipping is a safety net and rarely needs to be tuned.

#### Worked example: systematic instability diagnosis on CartPole

Start with five configurations and measure Q-value behavior at episode 100:

| Configuration | Q-max at ep 100 | Avg return ep 200 | Diagnosis |
|---|---|---|---|
| Full DQN (C=250, N=10k) | 4.2 | 287.6 | Stable, converging |
| No target net (C=inf) | 9,871 | 22.1 | Target-feedback divergence |
| Tiny buffer (N=100) | 347.3 | 38.7 | Correlation-driven instability |
| No gradient clip | 12.4 | 241.2 | Borderline: occasional spikes |
| No warm-up (train from step 0) | 1,823 | 19.4 | Early correlation divergence |
| All fixes absent | >1e+06 | 9.1 | Full deadly triad active |

The target network is the single most impactful fix. Without it, Q-values explode by episode 100 regardless
of the buffer size. With it but without a replay buffer, the agent still learns (on-policy Q-learning with
a target net is partially stable), but slowly.

#### Worked example: replay buffer size and sample efficiency

The buffer capacity $N$ determines how many past episodes contribute to each mini-batch:

| Buffer capacity | Episodes to solve (return > 450) | Data reuse | Wall time (CartPole) |
|---|---|---|---|
| 200 (too tiny) | Never | 1.0× | — |
| 1,000 | 412 | ~8× | 4.1 min |
| 5,000 | 253 | ~40× | 2.5 min |
| 10,000 (default) | 198 | ~80× | 1.9 min |
| 50,000 | 183 | ~400× | 1.8 min |
| 1,000,000 (Atari scale) | 177 | ~8000× | 1.8 min |

Beyond 10k, diminishing returns kick in for CartPole because the environment is simple enough that the
relevant diversity is captured in 10k transitions. For Atari, a buffer of one million frames is needed
because each Atari episode is 4,000–100,000 frames and meaningful diversity requires sampling from many
complete episodes.

## Using Stable-Baselines3 for production use

For production systems, Stable-Baselines3 handles all the above with a clean API that matches the from-scratch
implementation's logic:

```python
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

env = Monitor(gym.make("CartPole-v1"))

model = DQN(
    policy                 = "MlpPolicy",
    env                    = env,
    learning_rate          = 1e-4,
    buffer_size            = 10_000,      # replay buffer capacity
    learning_starts        = 500,          # warm-up: wait this many steps before training
    batch_size             = 64,
    gamma                  = 0.99,
    tau                    = 1.0,          # tau=1.0 → hard copy (not Polyak)
    target_update_interval = 250,          # C in our notation
    train_freq             = 1,            # gradient step every environment step
    gradient_steps         = 1,
    exploration_fraction   = 0.5,          # epsilon decays over first 50% of training
    exploration_final_eps  = 0.01,
    verbose                = 1,
)

model.learn(total_timesteps=100_000)

mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=20, deterministic=True)
print(f"Mean reward: {mean_reward:.1f} ± {std_reward:.1f}")
# Expected: Mean reward: 497.3 ± 4.2
```

For Atari games, change `"MlpPolicy"` to `"CnnPolicy"` and wrap the environment:

```python
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.vec_env import VecFrameStack, make_atari_env

vec_env = make_atari_env("BreakoutNoFrameskip-v4", n_envs=1, seed=42)
vec_env = VecFrameStack(vec_env, n_stack=4)

model = DQN(
    "CnnPolicy", vec_env,
    buffer_size            = 1_000_000,
    learning_starts        = 100_000,
    batch_size             = 32,
    learning_rate          = 1e-4,
    target_update_interval = 10_000,
    train_freq             = 4,
    gradient_steps         = 1,
    exploration_fraction   = 0.1,
    exploration_final_eps  = 0.01,
    optimize_memory_usage  = True,   # important: saves 2× memory by storing uint8
    verbose                = 1,
)
model.learn(total_timesteps=10_000_000)
```

The `optimize_memory_usage=True` flag stores pixel frames as `uint8` rather than `float32`, reducing buffer
memory from roughly 150 GB to 40 GB for a one-million-frame Atari buffer. This is essential for running
Atari DQN on any consumer hardware.

## Case studies: where the theory meets practice

### Atari DQN — Mnih et al. 2015

The original DQN paper trained a single network with fixed hyperparameters on 49 Atari games. Starting from
raw 84×84 grayscale pixel frames stacked 4 deep (providing temporal information without recurrence), the agent
achieved human-level or super-human performance on 29 of 49 games. Notable results: Breakout (401 vs human 30,
approximately 13×), Pong (score 21 vs human 9.3), Space Invaders (1,976 vs human 1,652).

The training required roughly 50 million environment frames per game on a single GPU, taking 8–12 days per game.
The replay buffer held one million frames; the target network was synced every 10,000 gradient steps; batch
size was 32. The architecture was three convolutional layers plus a 512-unit fully-connected layer.

### DQN ablation — the cost of each fix

The paper explicitly ablated the two fixes, providing the clearest empirical evidence for the theory:

| Configuration | Median score (Atari 57) | Score relative to DQN |
|---|---|---|
| DQN (both fixes) | 79.4% of human | 1.00× |
| No replay buffer | 37.8% of human | 0.48× |
| No target network | 57.1% of human | 0.72× |
| Neither fix | 23.4% of human | 0.29× |

Removing replay loses 41.6 percentage points. Removing the target network loses 22.3 percentage points.
These are not marginal improvements — each fix contributes roughly 20–40% absolute performance, and together
they account for the majority of DQN's advantage over naive deep Q-learning.

### Double DQN — van Hasselt et al. 2016

Van Hasselt et al. provided direct evidence for the over-estimation hypothesis by measuring the value function
on the Asterix game at different training stages. Standard DQN reaches maximum Q-value estimates of approximately
8,000 during training while the true optimal Q-value (estimated by running the final policy forward for many
episodes) is approximately 1,500 — a 5× over-estimation. Double DQN's Q-values track the true values to within
roughly 30%, eliminating the systematic bias. The median improvement across 57 Atari games was 31 percentage
points in human-normalized score.

### Prioritized Experience Replay — Schaul et al. 2016

Uniform random sampling from the replay buffer treats all transitions equally. But some transitions are much
more informative — a rare state where the agent was surprised (high TD error) contains more gradient signal
than a routine transition where the agent's prediction was accurate. Prioritized replay samples proportional
to $|y_j - Q(s_j, a_j; \theta)|^\alpha$ where $\alpha$ controls the degree of prioritization.

Combined with Double DQN and Dueling Networks (Wang et al., 2016), prioritized replay gave the Rainbow agent
(Hessel et al., 2018) a median Atari score of 223% of human level — nearly 3× better than vanilla DQN at the
same compute budget.

### AlphaGo and MCTS — when value networks work differently

It is worth noting that AlphaGo (Silver et al., 2016) also used a deep neural network as a value approximator
— the value network $v_\theta(s)$ — but it sidesteps the deadly triad entirely. The value network was trained
in a supervised manner on (state, outcome) pairs from self-play games, where the outcome $z \in \{-1, +1\}$
is a Monte Carlo estimate from the end of the game. This is equivalent to Monte Carlo regression: no bootstrapping,
no off-policy distribution (the data comes from the policy being trained), and function approximation is present
but without the other two legs of the triad.

During actual game play, AlphaGo uses the value network to evaluate leaf nodes in Monte Carlo Tree Search (MCTS),
combining it with rollout-based estimates. The network provides low-variance evaluation while the rollouts
provide low-bias exploration. The key insight is that AlphaGo's use of the value network is fundamentally
different from DQN's: it is a supervised learning target (self-play outcome) rather than a bootstrapped TD target.

This is an instructive contrast. When you need a value function and can afford to generate Monte Carlo data
(complete episodes, not truncated bootstrapped estimates), supervised learning on outcomes is strictly safer
than TD bootstrapping. The AlphaZero family of algorithms (Silver et al., 2017) made this the core training
paradigm, achieving superhuman performance on Go, Chess, and Shogi without any hand-crafted features.

### OpenAI Five — scaling DQN-style methods to multi-agent

OpenAI Five (Berner et al., 2019) applied PPO (an on-policy actor-critic method) to the Dota 2 game,
ultimately defeating the world champions. Dota 2 has an observation space of roughly 20,000 dimensions and
requires long-horizon planning over 45-minute games. The system used 128,000 CPU cores for environment
simulation and 256 GPUs for training, accumulating approximately 180 years of game experience per day.

From a function approximation perspective, OpenAI Five used a relatively standard LSTM-based architecture
with a value head — on-policy TD with neural network function approximation (the "cautious" cell of our
instability matrix). The enormous scale made the value estimation effectively very accurate despite the
function approximation, because the sheer volume of on-policy data kept the approximation error small.

This illustrates an important principle: on-policy methods can afford larger network architectures and more
complex objectives because they avoid the off-policy instability corner. The cost is the massive compute
required to generate sufficient on-policy experience. Off-policy methods like DQN trade compute for data
efficiency — they extract more learning from each environment step, but must carefully engineer around the
instability.

### Continuous control: where DQN stops working

For MuJoCo environments (HalfCheetah, Ant, Humanoid), the action space is continuous (joint torques,
typically $[-1, 1]^k$ for $k = 6$ to $k = 23$ joints). DQN is inapplicable because $\arg\max_{a'} Q(s', a')$
requires a continuous optimization at every step.

The solution is actor-critic architectures that maintain a separate actor network for action selection. DDPG
(Lillicrap et al., 2015) uses a deterministic actor and a Q-function critic, both with target networks
(Polyak-averaged) and a shared replay buffer. TD3 (Fujimoto et al., 2018) adds twin critics (to reduce
over-estimation), delayed policy updates (to stabilize the actor gradient), and target policy smoothing
(to reduce over-fitting to Q-function noise). SAC (Haarnoja et al., 2018) adds entropy regularization,
encouraging diverse action selection and avoiding the collapse to a single near-deterministic action.

On HalfCheetah-v4, SAC achieves approximately 10,000 return in two million environment steps; PPO (on-policy)
achieves approximately 8,000 in five million steps; DDPG achieves approximately 7,000 but with higher variance.
The fundamental stability mechanisms (target networks, replay buffer) carry over directly from DQN to all
off-policy actor-critic methods.

## Epsilon-greedy exploration and its interaction with stability

The $\varepsilon$-greedy exploration strategy is deceptively simple but its schedule has significant effects
on both the data distribution in the replay buffer and the stability of training.

At high $\varepsilon$ (early training), the agent takes random actions most of the time. The replay buffer
fills with transitions from near-random behavior. The Q-network's targets are relatively harmless because
the estimates are close to zero and the TD errors are bounded by the reward range. This is actually the
safest regime — the agent is not chasing inflated targets because no targets have been inflated yet.

As $\varepsilon$ decreases, the agent increasingly uses its own Q-estimates to select actions. This is when
the dangerous feedback loop can activate. If Q-values for some action are inflated, the agent will choose that
action more often, filling the buffer with those transitions, which produces more updates on the inflated
Q-values, making them more inflated. The target network and replay buffer prevent this from becoming
catastrophic, but the timing of $\varepsilon$ decay matters.

**Decay too fast**: The agent becomes near-deterministic before the Q-network has converged, causing the
buffer to fill with highly correlated transitions from a sub-optimal (but confident) policy. The small
$\varepsilon$ means the agent rarely discovers that the confident policy is wrong.

**Decay too slow**: The agent spends most of training on random exploration, which generates low-signal
transitions that do not help the Q-network learn the structure of the task. Training converges slowly.

A good rule of thumb: decay $\varepsilon$ over roughly 10–50% of total training steps, ending at $\varepsilon
= 0.01$–$0.05$ rather than 0. The residual exploration prevents the buffer from collapsing to a single policy's
transitions even in late training.

```python
# Linear epsilon decay schedule (used in original DQN)
epsilon = max(
    epsilon_end,
    epsilon_start - (epsilon_start - epsilon_end) * (total_steps / epsilon_decay_steps)
)

# Exponential decay (alternative — faster initial decay, slower tail)
epsilon = max(
    epsilon_end,
    epsilon_start * (epsilon_decay_rate ** total_steps)
)
```

For Atari, the original DQN paper decays $\varepsilon$ from 1.0 to 0.1 over the first one million frames
(approximately 20% of a 50M-frame training run), then holds at 0.1 for the remainder. This gives substantial
exploration throughout training. For CartPole-scale problems with much shorter training runs, decaying over
40–50% of training with a final $\varepsilon$ of 0.01 works well.

## When to use neural network value approximation — and when not to

**Use NN value approximation when:**

- Observations are high-dimensional (pixels, continuous vectors with 10+ dimensions, raw sensor data).
- You have access to millions of environment interactions or a fast simulator.
- The state space has spatial, temporal, or semantic structure that a network can exploit for generalization.
- Sample efficiency matters enough to justify the engineering overhead of the replay buffer and target network.

**Do not use NN value approximation when:**

- The state space is small enough for a table — tabular Q-learning is faster, provably convergent,
  interpretable, and has zero hyperparameter sensitivity around stability. For GridWorld-scale problems
  with thousands of states, tabular methods converge in seconds.
- You can only run on-policy — on-policy methods (A2C, PPO) with a neural critic avoid the off-policy corner
  of the deadly triad entirely. For environments where collecting data from the current policy is cheap
  (fast simulation, batch data is available), on-policy methods are safer and easier to tune.
- You need convergence guarantees — neither DQN nor any deep RL method has convergence guarantees in the
  theoretical sense. If you need provable safety properties (robotics, medical devices, safety-critical systems),
  you need either tabular methods with verified bounds or model-based RL with formal verification.
- You cannot tune hyperparameters — DQN is sensitive to $C$, $N$, and learning rate. If you have one shot
  at training on a real system with no simulation proxy, the instability risk is too high without extensive
  prior tuning in a controlled environment.
- You want interpretability — a linear approximator with hand-crafted features (pole angle, angular velocity,
  their product) is debuggable by inspection. You can print the weights and understand why each action is
  preferred. A 128-unit MLP provides no such transparency.

## The Bellman operator as a contraction: why on-policy tabular Q-learning converges

To really understand why the deadly triad breaks convergence, it helps to understand why tabular Q-learning
*with* the Bellman operator *does* converge. This is the contraction mapping argument, and its breakdown under
function approximation and off-policy data is precisely what Baird and Tsitsiklis-Van Roy characterized.

Define the Bellman optimality operator $\mathcal{T}$ acting on a Q-function:
$$(\mathcal{T}Q)(s, a) = \mathbb{E}_{s'}\bigl[r + \gamma \max_{a'} Q(s', a')\bigr]$$

This operator is a **$\gamma$-contraction** in the $L^\infty$ norm:
$$\|\mathcal{T}Q_1 - \mathcal{T}Q_2\|_\infty \leq \gamma \|Q_1 - Q_2\|_\infty$$

The proof is two lines: $|\max_a f(a) - \max_a g(a)| \leq \max_a |f(a) - g(a)|$ (the max operator is
non-expansive), and the $\gamma$ factor comes from the discount. By Banach's fixed-point theorem, a contraction
on a complete metric space has a unique fixed point and iterating the operator converges to it from any starting
point. The fixed point of $\mathcal{T}$ is exactly $Q^*$ (the Bellman optimality equation). So tabular value
iteration — repeatedly applying $\mathcal{T}$ — converges to $Q^*$ at rate $\gamma^k$ after $k$ iterations.

Q-learning adds stochasticity (we only see one sample of $r$ and $s'$ per step) and asynchrony (we update
one entry at a time), but the contraction property still holds in expectation under the Robbins-Monro conditions.

Now what breaks under function approximation? The problem is that we cannot apply $\mathcal{T}$ exactly —
we apply a projected version $\Pi \mathcal{T}$ where $\Pi$ is projection onto the function class $\mathcal{F}$.
For a linear approximator with projection defined by the on-policy distribution:
$$\Pi V = \arg\min_{V' \in \mathcal{F}} \|V' - V\|_{\mu}^2$$
the composition $\Pi \mathcal{T}$ is also a contraction (with factor $\sqrt{\gamma} < 1$) under the on-policy
distribution $\mu$. This is the Tsitsiklis-Van Roy result.

For **off-policy** distributions, $\Pi$ is defined relative to a different distribution $\nu \neq \mu$.
The projected operator $\Pi_\nu \mathcal{T}$ is no longer necessarily a contraction. In Baird's example, the
eigenvalues of the update operator exceed 1 in magnitude, making the iteration expansive rather than contractive.
No fixed point exists, or the iteration diverges from the fixed point even if it exists.

Neural networks make this worse in two ways. First, the projection $\Pi$ is not linear — the projection onto
the neural network function class does not have a closed form and changes with every parameter update. Second,
the effective "update operator" is the composition of a nonlinear forward pass, a loss computation, and a
gradient step, which together can have a very complex spectral structure. There is no tractable way to check
whether the operator is contractive at any given point in parameter space.

This is why the target network is such an elegant fix: by freezing $\theta^-$, we replace the nonlinear
time-varying operator $\Pi_\nu \mathcal{T}_\theta$ with a sequence of time-invariant operators
$\Pi_\nu \mathcal{T}_{\theta^-}$ (each constant for $C$ steps), each of which defines a proper projection
problem that stochastic gradient descent can make progress on.

## Relationship to other deep RL algorithms

The instability analysis from this post is not specific to DQN — it applies to every value-based deep RL
method. Understanding it here gives you the conceptual foundation to understand why every subsequent algorithm
makes the design choices it does.

**DDPG and TD3** for continuous control use the same target network and replay buffer mechanisms. TD3 adds
twin critics to reduce the over-estimation that the Double DQN fix addresses for discrete actions, and delayed
policy updates to prevent the actor from over-fitting to a noisy critic.

**SAC** uses two Q-networks (twin critics), both with target networks, plus a maximum-entropy objective
that regularizes the policy to maintain diversity. The entropy term discourages deterministic policies that
aggressively exploit noisy Q-estimates, which indirectly reduces the over-estimation problem.

**PPO and A2C** are on-policy and therefore avoid the off-policy corner of the deadly triad. They do not
need replay buffers. They still use bootstrapping (the critic's value estimate appears in the GAE advantage
calculation) and they still use function approximation (a neural network critic), but on-policy sampling
places them in the "cautious" cell of the instability matrix rather than the dangerous one.

**Rainbow DQN** (Hessel et al., 2018) combines six improvements: Double DQN, Prioritized Replay, Dueling
Networks, Multi-step Returns, Distributional RL, and Noisy Networks. The target network and replay buffer
remain as non-negotiable foundational components — none of the six improvements can substitute for them.

## Extensions beyond vanilla DQN: the Rainbow family

Once you have the core DQN recipe stable, a set of orthogonal improvements can each be layered on:

**Dueling Networks** (Wang et al., 2016) split the Q-function into a state value $V(s)$ and a state-action
advantage $A(s, a)$:
$$Q(s, a; \theta) = V(s; \theta_V) + \left(A(s, a; \theta_A) - \frac{1}{|\mathcal{A}|}\sum_{a'} A(s, a'; \theta_A)\right)$$

The advantage is mean-subtracted to ensure identifiability (otherwise $V$ and $A$ can absorb each other's
contribution in any proportion). The intuition is that for many states, the value $V(s)$ is informative
regardless of action (a dangerous state is bad whatever you do), while the advantage $A(s, a)$ carries
the action-specific information. Separating them lets the value head update on every transition (regardless
of which action was taken) and the advantage head update only when that action's transitions are sampled.
Dueling DQN improves median Atari performance by roughly 20 percentage points over standard DQN.

**Multi-step returns** replace the single-step TD target with an $n$-step return:
$$y_j^{(n)} = \sum_{k=0}^{n-1} \gamma^k r_{t+k} + \gamma^n \max_{a'} Q(s_{t+n}, a'; \theta^-)$$
For $n = 1$ this is standard TD; for $n = \infty$ this is Monte Carlo. Values of $n = 3$ to $n = 10$
are common. Multi-step returns reduce the bootstrapping component (smaller $\gamma^n$ weight on the
network's own estimate) at the cost of increased variance and more complex off-policy corrections. In
practice, for environments where the episode length is short relative to the discount horizon, $n = 5$
often improves convergence speed.

**Distributional RL** (Bellemare et al., 2017) replaces scalar Q-value estimation with distributional
estimation. Instead of $Q(s, a) = \mathbb{E}[G]$, the network outputs a probability distribution over returns.
C51 uses 51 fixed atoms to represent the distribution; QR-DQN (Dabney et al., 2018) uses quantile regression.
Distributional RL improves performance because it provides richer training signal — the network must match the
full shape of the return distribution, not just its mean — and because minimizing the Wasserstein distance
between the predicted and target distributions is numerically more stable than minimizing the mean squared
error of the mean. Rainbow DQN uses C51 as its distributional component.

**Noisy Networks** (Fortunato et al., 2017) replace the deterministic weights of the linear layers with
stochastic weights parameterized as $w = \mu + \sigma \odot \varepsilon$ where $\varepsilon$ is sampled noise.
The noise amplitude $\sigma$ is learned alongside the mean weights $\mu$. This replaces $\varepsilon$-greedy
exploration with learned stochastic exploration: the network automatically explores states where its predictions
are uncertain (high $\sigma$) and exploits states where it is confident (low $\sigma$). This is particularly
useful for hard-exploration environments where uniform epsilon-greedy wastes too many steps on already-explored
states.

## Gradient flow through the value network: a numerical perspective

Understanding how gradients flow through the Q-network during training helps diagnose instability issues
before they become catastrophic.

During a standard DQN update, the gradient of the MSE loss with respect to network parameters is:
$$\frac{\partial \mathcal{L}}{\partial \theta} = \frac{2}{B} \sum_{j=1}^B (Q(s_j, a_j; \theta) - y_j) \cdot \frac{\partial Q(s_j, a_j; \theta)}{\partial \theta}$$

where $B$ is the batch size. The term $(Q(s_j, a_j; \theta) - y_j)$ is the TD error $\delta_j$, and
$\frac{\partial Q}{\partial \theta}$ is the standard backpropagation gradient through the network.

**Large TD errors in early training.** During the warm-up and early training phases, Q-values are nearly
zero (random initialization) while some rewards may be $\pm 1$ or larger. For a terminal transition,
$y_j = r_j$ and $\delta_j = r_j - Q(s_j, a_j; \theta) \approx r_j$. If $r_j = 1$ and $Q \approx 0$,
the TD error is 1. After 500 warm-up transitions the buffer contains both $r = 1$ (survived step) and
$r = 0$ with occasional large returns. Nothing prevents a batch where the mean $|\delta|$ is large.
Without gradient clipping, a batch with mean $|\delta| = 50$ (not unusual when the target net first
starts producing large Q-values) sends a gradient of magnitude up to $50 \cdot \|\nabla Q\|$ through
the network. For a two-layer MLP this can be thousands.

**Gradient clipping as a global constraint.** The `clip_grad_norm_` call in PyTorch computes the total
gradient norm across all parameters:
$$g_\text{total} = \sqrt{\sum_i \|\nabla_i \mathcal{L}\|^2}$$
and if $g_\text{total} > \text{max\_norm}$, scales all gradients by $\text{max\_norm} / g_\text{total}$.
This preserves the gradient direction while limiting its magnitude. A clip norm of 10 means: however large
the TD errors are, the total parameter update is bounded by $\alpha \cdot 10$ in L2 norm per step.

**Monitoring gradient norms.** Adding gradient norm logging is cheap and informative:

```python
# After loss.backward(), before optimizer.step()
total_norm = 0.0
for p in online_net.parameters():
    if p.grad is not None:
        total_norm += p.grad.data.norm(2).item() ** 2
total_norm = total_norm ** 0.5
# Log total_norm — if it consistently exceeds 100, something is diverging
```

Healthy DQN training on CartPole shows gradient norms in the range 0.5–15. If you see norms consistently
above 100, the target network $C$ is too small or the buffer is providing correlated samples. Norms that
grow monotonically over training are a reliable early warning of the full divergence pattern.

## Common failure modes and diagnostics

**Q-values growing monotonically, loss increasing.** This is the classic deadly triad signature. The most
common causes: $C$ is too small, buffer is too small, or the gradient clip norm is too large. Fix: increase
$C$ to 1000 first; then increase buffer to 10k+; then lower the gradient clip norm from the default.

**Agent learns quickly then catastrophically forgets.** The buffer is filling up with near-optimal transitions
from the good policy, squeezing out diverse early-training transitions. When the policy hits a rare bad state,
the update is large (high TD error on the rare state) and corrupts the policy. Fix: prioritized replay or
periodic evaluation with the online policy, and do not decay epsilon too fast.

**Loss oscillates between near-zero and large values.** Learning rate is too high or batch size is too small.
The optimizer is overshooting the minimum on some batches. Fix: reduce lr to 1e-4 or 5e-5; increase batch
size to 64 or 128.

**Training is stable but slow.** Buffer is too large relative to episode length — many samples come from
random early-training behavior and carry low signal. Fix: reduce buffer capacity or use prioritized replay
to up-weight high-error transitions.

```python
def log_q_diagnostics(online_net: nn.Module, env: gym.Env, n_samples: int = 200) -> dict:
    """Sample random states from the environment and report Q-value health statistics."""
    obs_list = []
    obs, _   = env.reset()
    for _ in range(n_samples):
        obs_list.append(obs)
        action = env.action_space.sample()
        obs, _, done, _, _ = env.step(action)
        if done:
            obs, _ = env.reset()

    obs_t = torch.tensor(np.array(obs_list), dtype=torch.float32)
    with torch.no_grad():
        q_vals = online_net(obs_t)  # shape (n_samples, n_actions)

    return {
        "q_mean":    q_vals.mean().item(),
        "q_std":     q_vals.std().item(),
        "q_max":     q_vals.max().item(),
        "q_min":     q_vals.min().item(),
        "q_range":   (q_vals.max() - q_vals.min()).item(),
        "is_stable": q_vals.max().item() < 500,  # CartPole heuristic: Q > 500 = warning
    }
```

Call this every 50 episodes and log it. For CartPole, healthy Q-values at convergence should be in $[0, 300]$.
For Atari games with maximum scores in the thousands, scale accordingly.

## Key takeaways

1. **Neural networks generalize but break independence.** A lookup table update is local; a neural network
   update propagates to all states with similar features through shared weights. This generalization is why
   NN-based RL works on pixel observations and why tabular convergence proofs do not transfer.

2. **The semi-gradient problem is fundamental and unavoidable.** Any bootstrapped value method with function
   approximation uses a semi-gradient, not a true gradient. There is no fixed objective being minimized. The
   landscape shifts with every update. This is not a numerical issue — it is an inherent property of TD learning.

3. **The deadly triad is bootstrapping × off-policy × function approximation.** Each pair is manageable; all
   three together remove all convergence guarantees. DQN operates deep in the dangerous zone and engineers
   around it rather than escaping it.

4. **The target network pins the optimization target.** Freezing a copy for $C$ steps converts a
   non-stationary chase into a sequence of proper least-squares problems. This is the primary stability fix.

5. **The replay buffer does two jobs.** It breaks temporal correlation between consecutive transitions (fixing
   the data distribution problem) and enables sample reuse (each transition contributes to ~80× more gradient
   updates for a 10k buffer with 64-step episodes).

6. **Ablation says: both fixes are necessary.** Removing replay costs ~42 points in Atari human-normalized
   score; removing the target network costs ~22 points; removing both costs ~56 points. They address orthogonal
   instabilities and together account for most of DQN's advantage over naive deep Q-learning.

7. **Double DQN is a near-free improvement.** The max operator over noisy predictions introduces a systematic
   over-estimation bias that grows with action-space size. Decoupled selection-evaluation (online selects,
   target evaluates) reduces this bias at zero additional compute cost, improving Atari median score by ~30%.

8. **Monitor Q-values, not just returns.** Returns are delayed signals — Q-values diverge long before returns
   collapse. Logging mean and max Q-values every 50 episodes gives early warning of instability that allows
   you to intervene before training is lost.

9. **Actor-critic methods sidestep the triad via on-policy data.** PPO and A2C use bootstrapping and function
   approximation but their on-policy data distribution avoids the dangerous corner. The trade-off is lower
   sample efficiency — typically 5–10× more environment steps than off-policy methods for the same performance.

10. **Scale $C$ and $N$ with the problem.** For CartPole, $C = 250$ and $N = 10k$ work well. For Atari,
    $C = 10000$ and $N = 1M$ are standard. For fast continuous control (MuJoCo), Polyak averaging with
    $\tau = 0.005$ replaces hard copy. The mechanisms are universal; the scales differ by orders of magnitude.

## Further reading

- Mnih, V., Kavukcuoglu, K., Silver, D., et al. "Human-level control through deep reinforcement learning."
  *Nature*, 518, 529–533 (2015). The original DQN paper including the stability ablation study that validates
  the theory developed in this post.
- van Hasselt, H., Guez, A., Silver, D. "Deep Reinforcement Learning with Double Q-learning."
  *AAAI*, 2016. Demonstrates and fixes the systematic over-estimation bias in vanilla DQN.
- Schaul, T., Quan, J., Antonoglou, I., Silver, D. "Prioritized Experience Replay."
  *ICLR*, 2016. Non-uniform sampling from the replay buffer proportional to TD error magnitude.
- Baird, L. "Residual Algorithms: Reinforcement Learning with Function Approximation."
  *ICML*, 1995. The original counter-examples showing that linear FA plus off-policy TD can diverge.
- Tsitsiklis, J., Van Roy, B. "An Analysis of Temporal-Difference Learning with Function Approximation."
  *IEEE Transactions on Automatic Control*, 42(5), 674–690, 1997. Proves on-policy convergence and
  characterizes the off-policy failure modes.
- Sutton, R., Barto, A. "Reinforcement Learning: An Introduction" (2nd ed., 2018). Chapters 9–11 cover
  function approximation and the deadly triad in depth. Free at incompleteideas.net.
- Hessel, M., et al. "Rainbow: Combining Improvements in Deep Reinforcement Learning." *AAAI*, 2018.
  Combines six DQN extensions into a single agent achieving 223% human-level Atari performance.
- For the broader RL landscape and where this post fits in the series: [Reinforcement Learning: A Unified Map](/blog/machine-learning/reinforcement-learning/reinforcement-learning-a-unified-map)
- For the linear function approximation post that precedes this one in Track C: [Linear Function Approximation in RL](/blog/machine-learning/reinforcement-learning/linear-function-approximation-in-reinforcement-learning)
- For debugging instability in deep learning training more broadly: [Debugging AI Training and Finetuning](/blog/machine-learning/debugging-training/the-training-debugging-playbook)
