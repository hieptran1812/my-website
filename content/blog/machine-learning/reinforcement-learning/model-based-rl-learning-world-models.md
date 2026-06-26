---
title: "Model-Based RL: Learning World Models to Plan"
date: "2026-06-22"
publishDate: "2026-06-22"
description: "Why learning a dynamics model first can cut real-world samples by 10-100x, how Dyna-Q plans inside its own head, why model error compounds, and how PILCO, PETS, and Dreamer turn world models into policies you can actually deploy."
tags:
  [
    "reinforcement-learning",
    "deep-learning",
    "model-based-rl",
    "world-models",
    "sample-efficiency",
    "machine-learning",
    "pytorch",
    "markov-decision-process",
  ]
category: "machine-learning"
subcategory: "Reinforcement Learning"
author: "Hiep Tran"
featured: true
readTime: 55
image: "/imgs/blogs/model-based-rl-learning-world-models-1.png"
---

A robot arm I once helped train spent eleven days learning to slot a peg into a hole. Eleven days of physical motion: motors heating, bearings wearing, a human nearby to reset the workspace every time the arm flung the peg across the table. The agent — a clean, well-tuned implementation of Soft Actor-Critic — needed roughly two million environment steps to get reliable. Two million steps on real hardware is not a tuning detail. It is the entire project budget.

The frustrating part was that the *physics* of the task was almost trivial. The peg moves where you push it. The hole does not move. A first-year mechanical engineering student could write down the dynamics in an afternoon. Our model-free agent threw all of that structure away. It treated every interaction as an opaque sample from a black box and learned, slowly, statistically, by brute repetition. It never once tried to *predict* what would happen if it pushed left. It only ever reacted to what had already happened.

Model-based reinforcement learning (MBRL) is the response to exactly this waste. Instead of learning a policy directly from millions of real interactions, the agent first learns a **world model** — a function that predicts the next state and reward given the current state and action — and then uses that learned model to *plan*, to *imagine*, to generate enormous quantities of synthetic experience for almost free. One real transition can fund twenty, fifty, or a thousand simulated updates. The same peg-insertion task, attacked with a model-based method, has been solved in the published literature in well under an hour of real interaction. That is the gap this post is about.

This is the broad shape we will trace, summarized in the figure below: the agent collects a real transition, uses it both to update its policy *and* to refine a learned dynamics model, and then runs that model forward to generate cheap simulated experience that drives many more policy updates. By the end you will understand the theory of why this multiplies sample efficiency, the precise reason model error caps how far you can plan, and you will have a complete, runnable Dyna-Q agent, a probabilistic neural dynamics model with a CEM planner, and a clear map of how PILCO, PETS, and Dreamer scale the idea to real robots and pixels.

![Diagram of the model-based reinforcement learning loop where one real transition both updates the policy and trains a dynamics model that generates many simulated updates.](/imgs/blogs/model-based-rl-learning-world-models-1.png)

If you have not yet read the series taxonomy post, `reinforcement-learning-a-unified-map` is the frame this all sits inside: every RL algorithm is a different answer to *which objective to optimize* and *how to estimate the gradient that improves it*. Model-based methods add a second axis — *do you learn the dynamics, and if so, how do you use them?* — and that axis turns out to dominate the sample-efficiency story.

## 1. Model-free versus model-based: the fundamental trade-off

Let us be precise about what these two families actually optimize, because the difference is structural, not cosmetic.

In the standard reinforcement learning setup, an agent interacts with a **Markov Decision Process** (MDP), defined by a tuple $(\mathcal{S}, \mathcal{A}, P, r, \gamma)$. Here $\mathcal{S}$ is the set of states, $\mathcal{A}$ the set of actions, $P(s' \mid s, a)$ the transition dynamics (the probability of landing in state $s'$ after taking action $a$ in state $s$), $r(s, a)$ the reward function, and $\gamma \in [0, 1)$ a discount factor that makes future rewards worth slightly less than immediate ones. The agent wants a policy $\pi(a \mid s)$ that maximizes the expected discounted return $J(\pi) = \mathbb{E}\left[\sum_{t=0}^{\infty} \gamma^t r(s_t, a_t)\right]$.

A **model-free** method — PPO, SAC, DQN, A2C — never tries to learn $P$ or $r$. It learns a policy $\pi$ or a value function $Q(s, a)$ *directly* from sampled transitions, using the rewards as the only teaching signal. The dynamics live inside the environment and the agent only ever sees the consequences of acting, never a usable description of the rules. Model-free is the dominant paradigm precisely because it makes no assumptions: it works on Atari pixels, on language tokens, on robot torques, on order books, all with the same code. You pay for that generality in samples.

A **model-based** method learns an explicit, queryable approximation of the dynamics, $\hat{P}_\theta(s' \mid s, a)$ and $\hat{r}_\theta(s, a)$, and then uses that model in one of two ways: to *plan* (search forward through imagined futures to pick the best action) or to *generate synthetic data* (roll the model out to produce extra transitions that train a policy or value function). The model is a reusable artifact. Once you have it, you can query it millions of times without touching the real environment.

Here is the trade-off in one sentence: **model-free methods spend real samples and save modeling effort; model-based methods spend modeling effort to save real samples.** Which side you want depends entirely on what is scarce. If you have a fast simulator and can collect a billion steps overnight, model-free is simpler and often stronger at convergence. If every real step costs money, wall-clock time, or hardware wear — robotics, real-world control, expensive scientific experiments — model-based can be the difference between a feasible project and an impossible one.

It helps to make the cost asymmetry concrete with a number. A modern Atari or MuJoCo simulator on a single machine runs somewhere between several thousand and tens of thousands of steps per second; a fleet of parallel environments pushes that into the hundreds of thousands. At that rate, a billion environment steps — enough for PPO to master most benchmark tasks — is a matter of a day or two of wall-clock time and a few dollars of compute. Now contrast a physical robot. A single manipulation episode might take fifteen seconds of motion, plus a human or an automated rig to reset the workspace, plus the slow accumulation of mechanical wear. At fifteen seconds per episode and, say, fifty steps per episode, a million steps is roughly two thousand hours of continuous robot operation — months of real time, assuming nothing breaks. The same million steps the simulator hands you in a minute costs a robotics lab a quarter. *That* is the asymmetry MBRL is built to exploit: when the simulator is free, model-free is the rational choice; when each step is a quarter of a robot-hour, learning a model so you can stop touching the robot is not a luxury, it is the only way the project ships.

There is a second, subtler axis beyond raw sample cost: **structure**. Model-free methods are deliberately structure-blind. That blindness is a feature — it is why the identical SAC code that controls a robot arm also trades a portfolio and also plays an Atari game. But when the environment *does* have exploitable structure (smooth physics, locally linear dynamics, conserved quantities), a model-free agent rediscovers that structure only implicitly, buried inside the weights of a value network, and only after seeing it demonstrated thousands of times. A model-based agent can capture the same structure explicitly in a dynamics network that generalizes from far fewer examples, because predicting "the cart moves where I push it" is a much easier supervised-learning problem than inferring the optimal action under that physics. The model gets to learn from a dense, per-step regression signal (the observed next state) instead of the sparse, delayed, high-variance signal that is reward.

That last point about *signal density* is worth dwelling on, because it is the deepest reason model-based methods are sample-efficient and it is easy to miss. Consider the information content of a single environment step. A model-free value learner extracts from it exactly one scalar: the reward $r$ (and, eventually, a bootstrapped target). If the reward is sparse — zero almost everywhere, one at the goal — then the overwhelming majority of steps carry *no learning signal at all* for a value-only method, which is why sparse-reward tasks are the graveyard of model-free RL. A dynamics learner, by contrast, extracts a full vector from every single step: the observed next state $s'$ is a rich, dense regression target with as many supervised dimensions as the state has components. A 17-dimensional MuJoCo state gives the model 17 numbers of supervision per step, every step, reward or no reward. The model is doing self-supervised learning on the structure of the world; the policy is doing reinforcement learning on the (often sparse) reward. Decoupling those two problems, and letting the easy one (predict the next state) carry the data-efficient load, is the architectural insight underneath everything in this post.

| Property | Model-free (PPO, SAC, DQN) | Model-based (Dyna, PETS, Dreamer) |
| --- | --- | --- |
| What is learned | Policy and/or value function | Dynamics model, then policy/value |
| Real sample efficiency | Low (often $10^6$–$10^8$ steps) | High (often $10^4$–$10^5$ steps) |
| Compute per real step | Low | Higher (model training + planning) |
| Asymptotic performance | Often higher | Capped by model accuracy |
| Compounding error risk | None (no model) | High (errors grow with horizon) |
| Best when | Cheap simulation, complex obs | Expensive real steps, low-dim dynamics |
| Failure mode | Slow, sample-hungry | Policy exploits model errors |

That last row deserves a flag now, because it is the recurring tension of this entire post: a model-based agent is only ever as good as its model, and a policy optimizer is *spectacularly* good at finding and exploiting the places where the model is wrong. Half of practical MBRL is engineering against that exploitation.

## 2. What a world model actually is

Strip away the branding and a world model is one or two learned functions:

$$
\hat{s}_{t+1} = f_\theta(s_t, a_t), \qquad \hat{r}_{t+1} = g_\phi(s_t, a_t).
$$

The dynamics function $f_\theta$ predicts the next state; the reward function $g_\phi$ predicts the immediate reward. In many environments you already know $g$ (you wrote the reward function yourself), so only $f$ needs learning. In others — pixel-based games, real robots with hand-shaped rewards — you learn both.

There is a fork right at the start that shapes everything downstream: is the model **deterministic** or **stochastic**?

A **deterministic** model outputs a single predicted next state: $\hat{s}_{t+1} = f_\theta(s_t, a_t)$, trained with mean-squared error against observed transitions. It is simple, fast, and fine when the environment itself is nearly deterministic (most robotics tasks, most control problems). Its weakness is that it cannot represent genuine randomness, and worse, it gives you no signal about its own confidence. A deterministic network handed a state-action pair from a region it has never seen will return a crisp, plausible-looking prediction with exactly the same outward confidence as a prediction in a well-trodden region — and that false confidence is precisely what a policy optimizer exploits.

A **stochastic** model outputs a *distribution* over next states. The most common form predicts the parameters of a Gaussian, $\hat{s}_{t+1} \sim \mathcal{N}(\mu_\theta(s_t, a_t), \Sigma_\theta(s_t, a_t))$, training by maximizing the log-likelihood of observed transitions. This lets the model express **aleatoric uncertainty** — irreducible randomness in the environment, like dice rolls or sensor noise. We will see in Section 6 that a *different* kind of uncertainty, **epistemic** uncertainty (uncertainty from limited data), is even more important, and that you need a different tool — ensembles — to capture it.

Let me enumerate the model types in slightly more depth, because each is a real design point you will choose between, and the right choice is driven by your observation dimensionality and how much data you have.

**(a) Deterministic neural network.** A multilayer perceptron $f_\theta(s, a) = \hat{s}'$ trained with mean-squared error. This is the baseline, the thing you reach for first when the state is a low-dimensional vector of physical quantities and the dynamics are near-deterministic. It is fast to train, fast to query, and trivial to implement. Its two failure modes are that it cannot represent stochasticity and it provides no uncertainty estimate, so on its own it is unsafe to plan deep rollouts through.

**(b) Gaussian-output head.** The same network, but the final layer emits both a mean $\mu_\theta(s,a)$ and a (log-)variance $\log \sigma^2_\theta(s,a)$ per state dimension, trained by minimizing the negative log-likelihood of a diagonal Gaussian. Now the model can say "the next position is $3.2 \pm 0.4$" rather than just "$3.2$". The predicted variance captures *aleatoric* noise — genuine randomness the model has learned exists. This single change makes planning more honest, because a CEM planner can be told to downweight high-variance predictions. But a single Gaussian network still does not capture epistemic uncertainty: ask it about an unvisited region and it will often report a *small* variance with a *wrong* mean, because nothing in its training told it to be unsure where it has no data.

**(c) Ensemble of N networks.** Train $N$ (typically 5-7) independent Gaussian-output networks, each from a different random initialization and seeing data in a different order (and optionally on bootstrapped subsets). Where the data is plentiful, the networks converge to nearly the same function and their *means agree*. Where data is sparse, each network extrapolates differently and their means *fan out*. The spread of the ensemble means is a direct, cheap estimate of *epistemic* uncertainty — the dangerous kind. This is the PETS recipe, and it is the single most important practical advance that made neural-network MBRL reliable, because it finally gave the planner a signal for "do not trust me here."

**(d) Recurrent stochastic state-space model (RSSM).** When the observation is high-dimensional (pixels), you do not model dynamics in observation space at all. Instead you learn an encoder that compresses each observation into a compact latent code, a recurrent transition model that predicts how that latent evolves given an action, and small decoder heads that predict reward and (during training) reconstruct the observation. The recurrence carries a deterministic memory across time, and a stochastic latent variable at each step captures uncertainty and multimodality. This is the engine inside PlaNet and Dreamer, and it is what lets world models work on Atari and Minecraft. The point of the RSSM is that it predicts only the parts of the future that matter for choosing actions, in a representation where dynamics are smooth and learnable.

A useful way to hold all of this: a world model is a *learned simulator*. The real environment is a simulator written by nature or by a game studio; a world model is a smaller, differentiable, queryable simulator the agent builds for itself from experience. The whole field is the study of how to build that learned simulator well enough that planning inside it transfers back to the real thing.

There is a design decision hiding in *what space* the model predicts in, and it turns out to matter enormously. The most direct choice is to model dynamics in the **observation space**: given the raw state (or pixels), predict the raw next state (or pixels). This is what tabular Dyna and the low-dimensional PETS-style models do, and it is fine when the observation is a compact vector of physically meaningful quantities. But for high-dimensional observations like camera images, predicting raw future pixels is a disaster: most of the pixels are irrelevant background, the model wastes its capacity rendering texture instead of capturing dynamics, and the per-pixel error gives a misleading training signal. The modern answer, which we return to in the case studies, is to model dynamics in a learned **latent space** — first compress the observation into a low-dimensional latent code, then predict how that *code* evolves. PlaNet, Dreamer, and MuZero all do this. The lesson generalizes: a world model only needs to predict the parts of the future that matter for choosing actions, and choosing the representation in which to predict is half the battle.

A second design decision is whether to predict the *absolute* next state or the *change* (the delta) from the current state. In practice almost everyone predicts the delta $s_{t+1} - s_t$. The reason is numerical and statistical: in most physical systems the change per timestep is small and well-scaled relative to the absolute state, so a network predicting deltas starts close to the right answer (predict zero and you are already roughly correct) and only has to learn the small correction. Predicting the absolute next state forces the network to relearn the identity function on top of the dynamics, which wastes capacity and hurts accuracy. This single trick — predict deltas — recurs in nearly every neural dynamics model in the literature, and it appears in the code below.

There is also a quiet but crucial preprocessing step that lives alongside delta prediction: **input and target normalization**. Physical states mix quantities on wildly different scales — an angle in radians (order 1), a position in meters (order 10), an angular velocity (order 100). If you feed these raw into a network, the large-magnitude dimensions dominate the loss and the network barely learns the small ones. Standard practice is to standardize each input dimension to zero mean and unit variance using running statistics from the replay buffer, and to standardize the delta targets the same way. This is not a cosmetic detail; on benchmarks it is frequently the difference between a model that plans well and one that diverges. I mention it because it is the kind of thing that is omitted from papers and then silently sinks reimplementations.

#### Worked example: a one-line linear world model

Take a CartPole-like setup reduced to a single dimension: a cart whose position $x$ evolves as $x_{t+1} = x_t + 0.02 \cdot v_t$ and whose velocity changes with the applied force, $v_{t+1} = v_t + 0.1 \cdot a_t$. Suppose the true dynamics are exactly this and we observe ten transitions. A linear regression model $\hat{s}_{t+1} = W [s_t; a_t]$ will recover $W$ almost exactly, because the true dynamics are linear. With $s = [x, v]$ and a single scalar action $a$, after fitting on transitions where the velocity moved by $0.1 a$, the learned coefficient on the action for the velocity component lands at $0.1000 \pm 0.0003$. The model now predicts the next state to four decimal places. Plan inside it all you like — for *this* environment, imagined rollouts are nearly free and nearly perfect. The trouble, of course, starts the moment the dynamics are nonlinear and your data is sparse.

Here is that worked example as runnable code, so you can see the recovery happen and confirm that imagined rollouts inside a correct model are nearly perfect:

```python
import numpy as np

# true 1-D cart dynamics: x' = x + 0.02 v,  v' = v + 0.1 a
def true_step(s, a):
    x, v = s
    return np.array([x + 0.02 * v, v + 0.1 * a])

# collect 10 random transitions
rng = np.random.default_rng(0)
S, A, Snext = [], [], []
for _ in range(10):
    s = rng.normal(size=2)
    a = rng.normal()
    S.append(s); A.append([a]); Snext.append(true_step(s, a))
S, A, Snext = np.array(S), np.array(A), np.array(Snext)

# fit s' = W [s; a]  by least squares
X = np.hstack([S, A])               # shape (10, 3)
W, *_ = np.linalg.lstsq(X, Snext, rcond=None)   # shape (3, 2)
print("learned W:\n", np.round(W, 4))

# 5-step imagined rollout vs. truth from a fresh start
s_true = s_model = np.array([0.0, 1.0])
for t in range(5):
    a = 0.5
    s_true = true_step(s_true, a)
    s_model = np.hstack([s_model, [a]]) @ W
    print(f"t={t+1}  true={np.round(s_true,4)}  model={np.round(s_model,4)}")
```

The learned `W` recovers the dynamics to four decimals, and the 5-step imagined rollout tracks the truth essentially exactly. This is the best case — linear dynamics, noise-free data — and it is the standard against which every harder case degrades. Everything difficult about MBRL is what happens when the world is not this kind.

## 3. Dyna: the original architecture

The first clean statement of model-based RL is Richard Sutton's **Dyna** architecture, introduced in his 1991 paper "Integrated Architectures for Learning, Planning, and Reacting Based on Approximating Dynamic Programming." Its idea is so simple it is almost embarrassing that it works as well as it does: *interleave real experience with simulated experience, and use the same learning update for both.*

Sutton's framing in that paper is worth stating precisely, because it is the conceptual seed of nearly everything that follows. He observed that *learning* and *planning* are not two different processes — they are the same process applied to two different sources of experience. Learning uses real experience drawn from the environment; planning uses simulated experience drawn from a model. If the learning update does not care where its transitions came from, then a single Q-learning rule can serve both. This unification — "planning is just learning from imagined data" — is the entire intellectual move, and it is why Dyna code is so short: there is no separate planner, only the same update fed by two hoses.

Dyna decomposes the agent into **three interleaved processes**, and it pays to name them explicitly because every modern method is a variation on this same three-way split:

1. **Direct RL** — learning from real experience. The agent acts in the environment, observes a real transition $(s, a, r, s')$, and applies its value-learning update directly. This is exactly what a model-free agent does and it is always running.
2. **Model learning** — fitting the model from real experience. The same real transition is also used to improve the model: in tabular Dyna-Q, "improving the model" is a dictionary insert; in neural MBRL it is a gradient step on a regression loss.
3. **Planning** — learning from simulated experience. The agent queries its model for transitions it has seen before (or, in the neural case, can interpolate), and applies the *same* value-learning update to those imagined transitions. This is the planning process, and it runs $k$ times per real step.

The beauty is that processes 1 and 3 share an update rule, so the model-based contribution is "do process 1, then do process 3 a bunch more times." Concretely, Dyna-Q maintains a Q-table $Q(s, a)$, a learned model that simply *memorizes* observed transitions ($\text{Model}(s,a) \to (r, s')$ for the deterministic tabular case), and a loop that, after every real step, performs $k$ extra "planning" steps. Each planning step samples a previously-seen state-action pair, queries the model for what happened, and applies a standard Q-learning update using that remembered transition.

The Q-learning update is the familiar one. For a transition $(s, a, r, s')$:

$$
Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right].
$$

The bracketed term is the **temporal-difference (TD) error**: the gap between the value we currently estimate, $Q(s,a)$, and a better estimate built from the observed reward plus the discounted best next value. Dyna applies this exact update twice over: once on the real transition, and then $k$ more times on transitions replayed from the model.

The figure below shows the loop. Notice that the model-training step in tabular Dyna-Q is trivial — it is a dictionary insert — but the *consequence* is profound: a single real reward can propagate backward through the value function in one sweep instead of waiting many episodes for the credit to trickle back.

![Pipeline diagram of the Dyna-Q loop showing a real environment step, a Q update on real data, model training, and k planning updates from the model before repeating.](/imgs/blogs/model-based-rl-learning-world-models-4.png)

Here is the complete Dyna-Q algorithm, written out to mirror the three processes above. The planning ratio $k$ is the single knob that controls how aggressively the agent trades extra computation for fewer real steps.

```python
import numpy as np

class DynaQ:
    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.95,
                 epsilon=0.1, k_planning=10):
        self.Q = np.zeros((n_states, n_actions))
        self.model = {}  # (s, a) -> (r, s_next), deterministic memory
        self.alpha, self.gamma = alpha, gamma
        self.epsilon, self.k = epsilon, k_planning
        self.n_actions = n_actions

    def act(self, s):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        return int(np.argmax(self.Q[s]))

    def q_update(self, s, a, r, s_next):
        td_target = r + self.gamma * np.max(self.Q[s_next])
        self.Q[s, a] += self.alpha * (td_target - self.Q[s, a])

    def step(self, s, a, r, s_next):
        # process 1 - direct RL: learn from the real transition
        self.q_update(s, a, r, s_next)
        # process 2 - model learning: here, just remember it
        self.model[(s, a)] = (r, s_next)
        # process 3 - planning: k simulated updates from remembered transitions
        seen = list(self.model.keys())
        for _ in range(self.k):
            ps, pa = seen[np.random.randint(len(seen))]
            pr, pns = self.model[(ps, pa)]
            self.q_update(ps, pa, pr, pns)
```

The `step` method *is* Dyna. The first call is direct RL (process 1), the dictionary insert is model learning (process 2), and the planning loop is process 3 — and that planning loop, the model-based contribution, is what makes Dyna learn dramatically faster per real step than plain Q-learning.

It is worth pausing on *why* the planning loop is so cheap and so effective in the tabular case. The model here is not a neural network; it is a Python dictionary mapping each visited $(s, a)$ to the reward and next state observed. Querying it is a hash lookup. So a "planning step" costs essentially nothing — no forward pass, no gradient through a network, just a dictionary read and one arithmetic update of the Q-table. This is why $k$ can be set absurdly high (50, 100, even 1000) in tabular Dyna without the wall-clock cost mattering much, and it is the reason Dyna-Q on a small grid converges in a handful of real episodes. The planning is, quite literally, the agent sitting still and re-deriving the consequences of everything it has already seen.

A natural worry: if the model only ever replays transitions the agent has *already observed*, how can planning teach the agent anything new? The answer is that learning the value function is not the same as observing transitions. A single real transition tells you one reward and one next state; it does not tell you the *long-run value* of the state, because value depends on the whole downstream trajectory. The TD update propagates value information one hop at a time, and planning lets that propagation happen many times over, in arbitrary order, across all remembered transitions — so the agent extracts the full value-function consequence of its data instead of the sliver a single forward pass through the environment would reveal. Planning does not invent new data; it finishes the inference the data already supports.

There is a refinement of Dyna that is worth knowing because it captures the spirit of "spend planning where it matters": **prioritized sweeping**. Replaying remembered transitions uniformly at random, as the code above does, wastes planning steps on parts of the state space whose values have not changed. Prioritized sweeping instead keeps a priority queue of state-action pairs ranked by the magnitude of their last TD error, and replays the highest-priority pairs first; when updating a pair changes its value, the *predecessors* that lead into it get their priorities bumped, so value information sweeps backward from the reward along the most informative path. On large grids this can cut the planning steps needed for convergence by an order of magnitude. It is the tabular ancestor of prioritized experience replay in deep RL, and it is the first hint that *which* imagined transitions you replay matters as much as how many.

One important limitation of *tabular* Dyna, which the neural versions later fix, is that the dictionary-model cannot generalize. It only knows transitions for exact $(s, a)$ pairs it has seen; query an unseen pair and it has nothing to say. In a small discrete grid this is fine — you will visit every relevant pair. In a large or continuous state space it is fatal, and that is precisely where you must replace the dictionary with a function approximator that *interpolates*: a neural network that, having seen nearby transitions, can predict the dynamics of a state-action pair it has never literally encountered. That move — from memorization to generalization — is the leap from Dyna-Q to PETS and Dreamer, and it is what brings the compounding-error and uncertainty problems of the next sections into play.

Let us run it on a GridWorld and measure the effect of $k$. The environment is a simple maze: an agent starts in a corner, must reach a goal, and gets reward $1$ at the goal and $0$ otherwise.

```python
import numpy as np

class GridWorld:
    def __init__(self, size=7):
        self.size, self.start, self.goal = size, 0, size * size - 1
        self.s = self.start
    def reset(self):
        self.s = self.start
        return self.s
    def step(self, a):  # 0 up, 1 down, 2 left, 3 right
        r, c = divmod(self.s, self.size)
        if a == 0: r = max(0, r - 1)
        elif a == 1: r = min(self.size - 1, r + 1)
        elif a == 2: c = max(0, c - 1)
        elif a == 3: c = min(self.size - 1, c + 1)
        self.s = r * self.size + c
        done = self.s == self.goal
        return self.s, (1.0 if done else 0.0), done

def run(k_planning, episodes=50, seed=0):
    np.random.seed(seed)
    env = GridWorld(size=7)
    agent = DynaQ(env.size * env.size, 4, k_planning=k_planning)
    steps_per_episode = []
    for _ in range(episodes):
        s, done, steps = env.reset(), False, 0
        while not done and steps < 2000:
            a = agent.act(s)
            s_next, r, done = env.step(a)
            agent.step(s, a, r, s_next)
            s, steps = s_next, steps + 1
        steps_per_episode.append(steps)
    return steps_per_episode

for k in [0, 5, 50]:
    curve = run(k)
    print(f"k={k:>2}  episode 2 steps={curve[1]:>4}  "
          f"episode 10 steps={curve[9]:>4}")
```

If you want the side-by-side learning curves rather than two sampled episodes, the snippet below collects all three runs and prints a compact table of steps-to-goal per episode — the model-based equivalent of a learning curve, where lower is better and faster descent means more efficient learning:

```python
import numpy as np

curves = {k: run(k, episodes=12) for k in [0, 5, 50]}
print(f"{'episode':>8} | {'k=0':>6} {'k=5':>6} {'k=50':>6}")
for ep in range(12):
    row = "  ".join(f"{curves[k][ep]:>5}" for k in [0, 5, 50])
    print(f"{ep+1:>8} | {row}")
# optimal path length on a 7x7 grid from corner to corner is 12 steps
```

#### Worked example: counting how much planning helps

Running the code above on a 7x7 grid, the second episode tells the story. With $k=0$ (plain Q-learning, no planning) the agent typically still wanders for roughly 200-400 steps in its second episode, because a single pass of value information has barely propagated. With $k=5$, the second episode drops to around 60-90 steps. With $k=50$, the agent often reaches near-optimal behavior — roughly 12 steps, the Manhattan-optimal path length for this grid — by the second or third episode. The *real* environment steps the agent took to get there are nearly identical across the three runs in episode one; what differs is how much the agent squeezed out of those steps by replaying them through its model. That is sample efficiency made concrete: same real data, vastly different learning, controlled entirely by $k$.

Let me put numbers on the *real-step budget* to make the efficiency claim quantitative rather than impressionistic. Suppose all three agents need to have driven value information across roughly the 49 cells of the grid before they act near-optimally. With $k=0$, value propagates one TD hop per cell visit, so the agent must physically re-traverse paths dozens of times — on the order of several thousand real steps before the corner reward reliably steers behavior from the start. With $k=50$, the single first episode that stumbles into the goal already carries enough remembered transitions that the planning loop, replaying them 50 times per real step, sweeps coherent values across nearly the whole grid before the second episode begins — so the agent reaches near-optimal in roughly an order of magnitude fewer *real* steps. Same environment, same reward, same exploration; the only difference is that the $k=50$ agent paid in cheap arithmetic (CPU planning sweeps) to avoid paying in expensive environment interaction. On a grid the savings are a curiosity. On a robot, where each "real step" is a second of motor wear, that order of magnitude is the whole ballgame.

## 4. Why model-based wins on sample efficiency

The intuition is direct: each real transition $(s, a, r, s')$ gets stored in the model and then *reused* $k$ times as a training target. So the number of value updates per real step jumps from $1$ to $1 + k$. If your bottleneck is real environment interaction — and in robotics or real-world control it almost always is — you have just multiplied the learning you extract from each precious real sample by roughly $(1+k)$.

![Before and after comparison contrasting model-free learning at one update per real step against model-based learning that produces k updates per real step.](/imgs/blogs/model-based-rl-learning-world-models-2.png)

The figure makes the accounting visible. On the left, model-free: one real step buys one gradient update, and you grind through a million of them. On the right, model-based: one real step buys $k$ simulated updates, and you converge in a small fraction of the real steps.

Let me make the gradient-update accounting fully explicit, because it is the cleanest way to see *why* the sample-efficiency multiplier is real and not a hand-wave. Suppose you run for $N_\text{real}$ real environment steps. A model-free agent that does one gradient update per real step performs exactly $N_\text{real}$ updates to its policy/value parameters over the run. A model-based Dyna-style agent does one update on the real transition *plus* $k$ updates on model-generated transitions per real step, for a total of

$$
N_\text{updates} = N_\text{real} \cdot (1 + k).
$$

So with $k = 5$ you have performed six times as many gradient updates on the same real-data budget; with $k = 50$, fifty-one times as many. The real environment was touched $N_\text{real}$ times in both cases — that is the scarce, expensive resource and it is held fixed — but the *learning* extracted from it scales with $(1+k)$. This is the entire arithmetic of model-based sample efficiency in one line: **the model is a multiplier on updates-per-real-step, and updates-per-real-step is what drives convergence when the bottleneck is interaction rather than compute.**

The obvious follow-up question is: why not set $k$ to a million and converge in one real step? Two ceilings stop you. The first is that planning can only *replay information you have actually collected* (tabular) or *interpolate within the region your model has learned* (neural) — it cannot conjure information that was never observed, so beyond some $k$ the extra updates are re-deriving value consequences that have already fully propagated and add nothing. The second, which only appears in the neural case, is that each imagined update is trained against a *wrong* target proportional to the model's error; cranking $k$ too high in a neural MBRL method overfits the policy to the model's mistakes, which is the model-exploitation failure of Section 8. So $(1+k)$ is the *upper bound* on the multiplier, realized only when the model is accurate and the value information has not yet saturated.

There is a cleaner theoretical way to see why this should help even at modest $k$. Tabular Q-learning converges to the optimal $Q^*$ in the limit, but its *rate* is governed by how quickly value information propagates across the state space. Each real transition only directly updates one $(s,a)$ entry; the information about a distant reward has to diffuse back one TD step per visit. Planning lets that diffusion happen in software, against the model, without burning real steps. In the GridWorld above, the reward sits in one corner; with enough planning sweeps, a single episode that reaches the goal is enough to back-propagate sensible values to most of the grid, because the model lets you replay every remembered transition in the right order.

Formally, the sample-complexity improvement is not unbounded — you cannot conjure information that was never collected. Planning can only replay transitions you have actually seen (in tabular Dyna) or interpolate within the region your model has learned (in neural Dyna). The win is in *extracting more learning from the same data*, not in collecting better data. This is precisely why model-based methods dominate the low-data regime and then often *lose* their edge asymptotically: once you have enough real data that value information has fully propagated anyway, the planning replay stops adding much, and the model's approximation error starts to cap your performance below what a model-free method would eventually reach. The relationship between data volume and the right algorithm choice is itself a kind of scaling question — see the broader treatment in `/blog/machine-learning/scaling-laws/data-constrained-scaling-laws` for how data scarcity reshapes which methods win.

## 5. Compounding model error: the central limitation

Now the bad news, and it is the single most important thing to internalize about MBRL. A learned model is never perfect. Call its one-step prediction error $\epsilon_1$ — the typical distance between $f_\theta(s, a)$ and the true next state on a transition drawn from the data distribution. The problem is that planning requires rolling the model forward: you feed the model's own prediction back into itself to predict two steps ahead, three steps, and so on. Errors do not stay put. They *compound*.

![Stacked diagram showing prediction error growing roughly linearly from epsilon at step one to T times epsilon at the planning horizon.](/imgs/blogs/model-based-rl-learning-world-models-3.png)

The figure shows the basic shape. After one step you are off by about $\epsilon_1$. But at step two you are predicting *from an already-wrong state*, so two sources of error stack: the fresh one-step error you incur at step two, plus whatever error you carried in from step one (possibly amplified by the dynamics). At step $t$, under mild Lipschitz-continuity assumptions on the dynamics, the accumulated error grows at least linearly:

$$
\text{error}(T) \;\le\; \sum_{t=0}^{T-1} L^{t} \, \epsilon_1,
$$

where $L$ is the Lipschitz constant of the true dynamics (how much a small state error gets amplified per step). Let me derive the two regimes that this sum collapses into, because they are the whole practical story.

**The benign regime, $L \approx 1$.** When the dynamics neither amplify nor damp small errors — which is the typical case for well-behaved, stable physical systems near an operating point — every term in the sum is approximately $\epsilon_1$, and the sum of $T$ of them is

$$
\text{error}(T) \;\approx\; \epsilon_1 \cdot T.
$$

The error grows **linearly** in the horizon. Double the horizon, double the accumulated error. This is the case that justifies "plan to a short horizon": if a one-step error of $\epsilon_1 = 0.01$ is tolerable but $0.1$ is not, then you can trust roughly the first $0.1 / 0.01 = 10$ steps and no more. In practice this is exactly why you see planning horizons of $H \le 5$ to $H \le 15$ throughout the literature — beyond that, the linear accumulation has eaten the signal.

**The malignant regime, $L > 1$.** When the dynamics are unstable or chaotic — a pole near vertical, a system at a tipping point, anything where small differences blow up — the geometric series no longer behaves. With $L > 1$,

$$
\text{error}(T) \;\approx\; \epsilon_1 \cdot \frac{L^{T} - 1}{L - 1},
$$

which grows **exponentially** in $T$. Now even a tiny one-step error is fatal within a handful of steps, because each step multiplies the carried error by $L$. This is why planning more than a few steps ahead in a chaotic system is hopeless no matter how good your model is: you are fighting an exponential, and the exponential always wins. The honest response in this regime is to plan extremely short and lean heavily on a learned value function to summarize everything past the horizon.

This single fact explains an enormous amount of MBRL design. It is *why* you rarely see model-based methods plan to a 1000-step horizon. It is why **short-horizon** planning — looking only a few dozen steps ahead and then bootstrapping with a learned value function — is the dominant modern pattern. It is why a tiny improvement in one-step accuracy $\epsilon_1$ pays off so dramatically: halving $\epsilon_1$ roughly halves the error at *every* horizon, which can roughly double how far you can usefully plan. And it is why the practical horizon $H$ is almost always kept to $H \le 5$ in the hardest settings: the product of "model accuracy you can realistically achieve" and "amplification the dynamics impose" leaves a very short window of trustworthy lookahead.

#### Worked example: where the rollout stops being trustworthy

Suppose your neural dynamics model has a one-step error of $\epsilon_1 = 0.01$ (in normalized state units) and the true dynamics have $L \approx 1.05$ — mildly unstable, like a system that tends to drift. The accumulated error at horizon $T$ is approximately $\epsilon_1 \cdot (L^T - 1) / (L - 1)$. At $T=5$ that is $0.01 \cdot (1.276 - 1)/0.05 \approx 0.055$ — still small, planning is reliable. At $T=20$ it is $0.01 \cdot (2.653 - 1)/0.05 \approx 0.33$ — a third of the state range is now error, and any value estimate built on this rollout is suspect. At $T=50$ it explodes to $\approx 2.1$, larger than the state range itself: the rollout is pure fiction. The practical lesson is to plan to roughly $T=10$ here, bootstrap with a value function for everything beyond, and never let the optimizer chase rewards that only exist at the far end of a hallucinated trajectory.

Contrast that with the benign $L = 1.0$ case and the same $\epsilon_1 = 0.01$: now the error at horizon $T$ is just $0.01 \cdot T$, so $T = 5$ gives $0.05$, $T = 20$ gives $0.20$, and $T = 50$ gives $0.50$. Even at fifty steps you are off by half a state-unit rather than two — bad, but bounded and roughly survivable for a short while. The single parameter $L$ — whether the world forgives or amplifies your mistakes — is what separates "I can plan twenty steps" from "I can plan five." Knowing which regime your environment is in is the first thing to establish before you tune a planner.

## 6. Model uncertainty: epistemic versus aleatoric

The fix for compounding error is not just "make the model better" — it is "make the model *know when it does not know*." Two kinds of uncertainty matter, and conflating them is a classic mistake.

**Aleatoric uncertainty** is irreducible randomness in the environment. A slot machine, a noisy sensor, a dice roll: even a perfect model cannot predict the exact outcome, only its distribution. You capture aleatoric uncertainty with a *probabilistic* model that outputs a distribution, like the Gaussian-output network from Section 2. More data does not reduce it — the dice are still random no matter how many rolls you have watched.

**Epistemic uncertainty** is uncertainty from *limited data*. In regions of state-action space the agent has visited often, a well-trained model is confident and correct. In regions it has never seen, the model is extrapolating, and its predictions can be wildly, confidently wrong. This is the dangerous kind, because a policy optimizer will happily steer toward exactly those unexplored regions if the model hallucinates high reward there. Crucially, epistemic uncertainty *shrinks with more data* — and that is what makes it actionable: high epistemic uncertainty is a signal to either collect more real data there or to avoid relying on the model there.

The distinction is not academic; it changes what you *do*. Aleatoric uncertainty you accept and plan around — you hedge, you choose actions robust to the noise. Epistemic uncertainty you *resolve or avoid* — you go collect data there (if you are exploring) or you refuse to trust the model there (if you are planning a deployment-quality action). A method that lumps the two together cannot tell "this is genuinely random" apart from "I have never been here," and so it cannot make either decision correctly.

The standard tool for estimating epistemic uncertainty is an **ensemble**: train several models with different random initializations and different data orderings, and look at how much they *disagree*. Where they agree, the data was plentiful and the prediction is trustworthy. Where they fan out, you are extrapolating. The mechanism is intuitive — many functions can fit the same sparse data, and they only have to agree where the data pins them down; away from the data they are free to diverge, and that divergence *is* the epistemic uncertainty made visible.

![Graph diagram showing one input fed to three dynamics models whose disagreement produces a high-variance uncertainty signal that drives a conservative action choice.](/imgs/blogs/model-based-rl-learning-world-models-8.png)

The figure shows the mechanism: feed the same $(s, a)$ to several models, and the spread of their predictions *is* your epistemic uncertainty estimate. This is exactly what the PETS algorithm (next section) does, and it is the single most important practical advance that made neural-network MBRL reliable.

Here is a minimal probabilistic ensemble in PyTorch — a network that outputs both a mean and a (log) variance, so each member captures aleatoric uncertainty, and the ensemble captures epistemic uncertainty.

```python
import torch
import torch.nn as nn

class GaussianDynamics(nn.Module):
    """Predicts mean and log-variance of the next-state delta."""
    def __init__(self, state_dim, action_dim, hidden=200):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
        )
        self.mean = nn.Linear(hidden, state_dim)
        self.logvar = nn.Linear(hidden, state_dim)

    def forward(self, s, a):
        h = self.net(torch.cat([s, a], dim=-1))
        mean = self.mean(h)
        logvar = torch.clamp(self.logvar(h), -10.0, 2.0)
        return mean, logvar

    def nll_loss(self, s, a, s_next):
        mean, logvar = self.forward(s, a)
        delta = s_next - s            # predict the change, not absolute state
        inv_var = torch.exp(-logvar)
        # negative log-likelihood of a diagonal Gaussian
        return (0.5 * (inv_var * (delta - mean) ** 2 + logvar)).mean()

class Ensemble(nn.Module):
    def __init__(self, n_models, state_dim, action_dim):
        super().__init__()
        self.models = nn.ModuleList(
            [GaussianDynamics(state_dim, action_dim) for _ in range(n_models)])

    def predict(self, s, a):
        means = torch.stack([m.forward(s, a)[0] for m in self.models])
        # epistemic uncertainty: disagreement of the ensemble means
        epistemic = means.var(dim=0).sum(dim=-1)
        return means.mean(dim=0), epistemic
```

Two details earn their keep here. First, the model predicts the *delta* $s_{t+1} - s_t$ rather than the absolute next state; this is a near-universal trick in MBRL because the change is usually small and well-scaled even when the absolute state is large. Second, the `nll_loss` trains each member to be calibrated (the predicted variance should match the actual squared error), so the ensemble's disagreement is a meaningful uncertainty signal rather than noise.

It is worth being precise about *which* variance you read off for *which* purpose. The `logvar` head of a single member estimates the aleatoric variance — the noise that member believes is intrinsic at that input. The *variance across members' means* (the `epistemic` term in `predict`) estimates the epistemic uncertainty — how much the models disagree about the mean prediction. A fully principled total predictive variance combines both: the average of the members' aleatoric variances (the noise everyone agrees exists) plus the variance of their means (the disagreement). When you penalize a planner for uncertainty, it is almost always the *epistemic* term you want, because that is the part that flags "you are off the data and should not trust me" — exactly the signal that keeps a policy optimizer from chasing hallucinations.

## 7. Model capacity: from Gaussian processes to neural ensembles

The history of MBRL is partly a history of *what function class* you use for the model, because that choice determines both how data-efficient the model is and how well it scales.

**PILCO** (Deisenroth & Rasmussen, 2011) used a **Gaussian process** (GP) as its dynamics model, and it deserves a full description because it remains the most sample-efficient neural-free result and it teaches the central lesson cleanly. A GP is a Bayesian non-parametric model that gives you a full predictive distribution — mean *and* calibrated uncertainty — out of the box, with no architecture to tune and no risk of confident extrapolation, because a GP's predictive variance *grows automatically* as you move away from the training data. That last property is the whole point: a GP cannot be confidently wrong in a region it has never seen, because its uncertainty estimate inflates there by construction.

PILCO's genuinely clever move was *how it used* that uncertainty. Rather than sampling rollouts and averaging (the Monte Carlo approach PETS would later take), PILCO propagated the *full Gaussian state distribution* analytically forward through the model, approximating the distribution over states at each future step with a Gaussian via moment matching. Because the policy's expected long-term cost was then a *differentiable, closed-form* function of the policy parameters — uncertainty included — PILCO could compute an **analytic policy gradient** through the model and do gradient-based policy improvement directly, never once sampling a noisy rollout. The effect of carrying uncertainty through this computation is that the policy optimizer is automatically penalized for relying on uncertain predictions: a high-variance region contributes high expected cost, so the policy avoids betting on it. This is uncertainty-aware planning in its purest form, and it is *why* PILCO was so data-efficient.

The headline numbers: PILCO learned to swing up and balance a cart-pole from scratch using on the order of **17 trials totaling roughly 20 seconds of interaction**, learning the full nonlinear control from essentially nothing — a result that was almost an order of magnitude beyond anything model-free at the time. The catch is purely computational: GPs scale cubically in the number of data points, $O(n^3)$, and poorly in input dimension, so PILCO is wonderful on a 4-dimensional cart-pole and unusable on a 100-dimensional humanoid or on pixels. The lesson PILCO leaves us is not "use GPs" — almost nobody does at scale — but "carry calibrated uncertainty and let it penalize the policy," which is exactly the principle the neural methods had to reinvent.

**Deterministic neural networks** (the simplest possible deep model) scale to high dimensions and large datasets but give you no uncertainty and are prone to the confident-extrapolation failure described above. They were the obvious thing to try when the field wanted to scale past the GP, and they were a disappointment for years precisely *because* they threw away the uncertainty that made PILCO work.

**PETS** (Chua et al., 2018) — "Probabilistic Ensembles with Trajectory Sampling," from the paper titled "Deep Reinforcement Learning in a Handful of Trials using Probabilistic Dynamics Models" — is the synthesis that made deep MBRL competitive: an *ensemble* of *probabilistic* neural networks (exactly the `Ensemble` class above), giving you both aleatoric uncertainty (each member is Gaussian) and epistemic uncertainty (the members disagree), at neural-network scale. The "trajectory sampling" half of the name refers to how PETS propagates uncertainty during planning: rather than averaging the ensemble at each step (which would wash out the disagreement), it samples *particles* and propagates each particle through a consistently-chosen ensemble member across the rollout, so the spread of particle outcomes faithfully reflects both kinds of uncertainty. PETS matched model-free SAC on MuJoCo locomotion tasks while using roughly an order of magnitude fewer samples — on HalfCheetah it reached a return of around 6,000 in roughly **800 environment steps** (a handful of episodes), where SAC and PPO needed hundreds of thousands of steps to get to comparable performance.

| Model class | Data efficiency | Scales to high-dim? | Uncertainty? | Used by |
| --- | --- | --- | --- | --- |
| Gaussian process | Excellent | No ($O(n^3)$) | Calibrated, built-in | PILCO |
| Deterministic NN | Moderate | Yes | None | early neural Dyna |
| Probabilistic NN | Good | Yes | Aleatoric only | single-model baselines |
| Probabilistic NN ensemble | Good | Yes | Aleatoric + epistemic | PETS |
| Latent recurrent model | Good | Yes (pixels) | Aleatoric + epistemic | PlaNet, Dreamer |

PETS pairs its ensemble model with **decision-time planning** via the Cross-Entropy Method (CEM): at each real step, it samples thousands of candidate action sequences, scores each one by rolling it forward through the ensemble (propagating particles to be robust to model error), keeps the top-scoring fraction, refits a sampling distribution to them, and repeats. The first action of the best sequence is executed; then the whole search runs again at the next step. Here is the core of CEM planning:

```python
import torch

def cem_plan(ensemble, reward_fn, state, action_dim, horizon=15,
             pop=400, elites=40, iters=5):
    """Decision-time planning: search action sequences in the model."""
    mean = torch.zeros(horizon, action_dim)
    std = torch.ones(horizon, action_dim)
    for _ in range(iters):
        # sample a population of candidate action sequences
        seqs = mean + std * torch.randn(pop, horizon, action_dim)
        seqs = seqs.clamp(-1.0, 1.0)
        returns = torch.zeros(pop)
        s = state.repeat(pop, 1)
        for t in range(horizon):
            a = seqs[:, t, :]
            delta, _ = ensemble.predict(s, a)
            s_next = s + delta
            returns += reward_fn(s, a, s_next)
            s = s_next
        # keep the elite sequences, refit the sampling distribution
        elite_idx = returns.topk(elites).indices
        elite_seqs = seqs[elite_idx]
        mean = elite_seqs.mean(dim=0)
        std = elite_seqs.std(dim=0) + 1e-6
    return mean[0]  # execute only the first action, then replan
```

Notice the horizon is only 15 — a direct consequence of the compounding-error analysis. CEM does not plan to the end of the episode; it plans a short window and re-plans every step (this is Model Predictive Control, covered in Section 9).

To make PETS fully concrete, here is the training side — fitting the ensemble on a replay buffer of real transitions — so you can see how the pieces from Section 6 and this section fit together into a working loop. This is the model-learning half of a PETS agent; the `cem_plan` above is the planning half.

```python
import torch

def train_ensemble(ensemble, buffer, epochs=5, batch=256, lr=1e-3):
    """Fit each ensemble member on the replay buffer of real transitions.

    buffer yields tensors (s, a, s_next). Each member sees an independently
    shuffled stream so their disagreement reflects epistemic uncertainty.
    """
    opt = torch.optim.Adam(ensemble.parameters(), lr=lr)
    S, A, Snext = buffer.tensors()          # (N, ds), (N, da), (N, ds)
    n = S.shape[0]
    for _ in range(epochs):
        for m in ensemble.models:
            perm = torch.randperm(n)        # different ordering per member
            for i in range(0, n, batch):
                idx = perm[i:i + batch]
                loss = m.nll_loss(S[idx], A[idx], Snext[idx])
                opt.zero_grad(); loss.backward(); opt.step()
    return ensemble

# the full PETS outer loop, in pseudocode-as-code:
def pets_iteration(env, ensemble, buffer, reward_fn, action_dim,
                   episode_len=200):
    s = torch.tensor(env.reset(), dtype=torch.float32)
    for _ in range(episode_len):
        a = cem_plan(ensemble, reward_fn, s, action_dim)   # decision-time plan
        s_next, _, done = env.step(a.numpy())
        buffer.add(s, a, torch.tensor(s_next, dtype=torch.float32))
        s = torch.tensor(s_next, dtype=torch.float32)
        if done:
            break
    train_ensemble(ensemble, buffer)        # refit model on all real data so far
    return ensemble, buffer
```

The shape of the outer loop — collect a short burst of real data with the current model, then refit the model on *all* data collected so far, then repeat — is the data flywheel of the next section made literal. Each iteration the model gets a little better, so the CEM plans get a little better, so the data gets a little more informative.

## 8. The data flywheel and how errors propagate

Step back and look at the full cycle, because the *order* of operations matters and a subtle failure lurks in it. The loop is: collect real data → train model → generate simulated data (or plan) → improve policy → deploy policy → collect more real data. It is a flywheel, and like all flywheels it can spin up beautifully or wobble apart.

The healthy version: a decent policy visits informative states, the model gets better data exactly where the policy cares, the model's improved accuracy lets the policy improve, and the improved policy collects even better data. PILCO, PETS, and Dreamer all spin this flywheel and it converges fast. Notice the virtuous coupling — the policy and the model improve *each other*, because the policy decides where data is collected and the model decides how well that data is exploited.

The failure version is **distribution shift** colliding with **model exploitation**, and it is worth tracing the failure step by step because every term in the chain is a place you can intervene. Step one: the policy improves and starts visiting a new region of state space the model has never seen — this is distribution shift, and it is *inevitable* in a method whose whole point is that the policy keeps changing. Step two: the model's predictions in that region are pure extrapolation, carrying high epistemic uncertainty, but a naive policy optimizer has no access to that uncertainty and treats the extrapolated prediction as fact. Step three: because a high-capacity model *must* guess somewhere, somewhere in that unexplored region it will hallucinate a state that looks high-reward. Step four: the optimizer, doing its job perfectly, drives the policy *straight toward the hallucination* — the better your optimizer, the worse this is. Step five: the policy collects real data there, discovers the reward was fictional, and you have burned real samples chasing a ghost; worse, the policy may have unlearned good behavior on the way. The flywheel has thrown itself apart.

This is the deepest reason ensembles and uncertainty penalties matter. The robust fix, used in modern methods like MBPO and in offline MBRL, is to *penalize the policy's reward by the model's epistemic uncertainty*:

$$
\tilde{r}(s, a) = \hat{r}(s, a) - \lambda \cdot u(s, a),
$$

where $u$ is the ensemble disagreement. This makes the optimizer treat "the model is unsure here" as "this is probably not as good as it looks," which keeps the policy honest and inside the region where the model is trustworthy. It is the same instinct that the KL-penalty serves in RLHF — keep the optimizer from wandering into regions where your learned signal is unreliable — and if you have read `/blog/machine-learning/training-techniques/rlhf-reward-modeling` the parallel will feel familiar.

Another stabilizer, and the one that defines MBPO ("When to Trust Your Model," Janner et al., 2019), is to keep model rollouts **short and branched from real states**. The MBPO insight is a direct application of the compounding-error analysis: rather than imagining a single long trajectory from the initial state (where error compounds catastrophically, as Section 5 showed), MBPO branches *short* rollouts — often just 1 to 5 steps — starting from real states sampled out of the replay buffer. Every imagined trajectory begins from a state the agent actually reached, so it never strays far from the data before being re-anchored. The theoretical contribution of the MBPO paper is a bound showing exactly this trade-off: longer rollouts give the policy more synthetic data but accumulate more model error, and there is an optimal (short) rollout length that maximizes the net benefit. This is the practical reconciliation of "generate lots of synthetic data" with "model error compounds": generate lots of synthetic data, but keep each piece of it short and anchored. In MBPO that short synthetic data is then fed to an off-policy model-free learner (SAC), so MBPO is best understood as Dyna with a neural ensemble model, short branched rollouts, and a modern actor-critic doing the value learning — a genuinely hybrid design.

```python
def branched_rollout(ensemble, policy, real_states, reward_fn, length=3):
    """MBPO-style short rollouts branched from real buffer states."""
    synthetic = []
    s = real_states.clone()              # anchored to real experience
    for _ in range(length):
        a = policy.sample_action(s)
        delta, epistemic = ensemble.predict(s, a)
        s_next = s + delta
        r = reward_fn(s, a, s_next) - 0.1 * epistemic  # uncertainty penalty
        synthetic.append((s, a, r, s_next))
        s = s_next
    return synthetic
```

The `0.1 * epistemic` term is the uncertainty penalty $\lambda \cdot u(s,a)$ in code, and `length=3` is the short, anchored rollout. Both lines exist for the same reason: to stop the optimizer from believing the model in places the model has no right to be believed.

## 9. Background planning versus decision-time planning

There are two fundamentally different ways to *use* a model, and almost every MBRL method is one or the other (or a blend). This distinction — Sutton and Barto call it the difference between *background* and *decision-time* planning — is the single most useful lens for organizing the whole field, so it is worth a full treatment.

**Background planning** uses the model offline to improve a stored policy or value function. Dyna is the canonical example: between real steps (or in a background thread), it replays model transitions to update the Q-table. The expensive thinking happens ahead of time; at action selection, the agent just reads off its already-improved policy, which is fast. MBPO, Dreamer, and Dyna all do background planning — they use the model to *manufacture training data* for a policy. The defining characteristic is that planning and acting are *decoupled in time*: the model's work is amortized into the policy parameters during a training phase, and at deployment the policy is just a fast function call. This is exactly what you want for high-frequency control, where you cannot afford to run a search inside the control loop, and it is what lets Dreamer train on millions of imagined trajectories and then act in real time.

**Decision-time planning** uses the model online, *at the moment of acting*, to search forward and pick the best action right now. PETS with CEM is the canonical example, and so is **Model Predictive Control** (MPC): at every step, plan an optimal action sequence over a short horizon, execute only the first action, then throw the rest away and re-plan from the new state. MuZero's Monte Carlo Tree Search is also decision-time planning. The cost is that you pay for a search *every single step*, which can be too slow for high-frequency control, but the benefit is decisive — you always plan from the *true current state*, so error can only compound over the short planning horizon before being reset by the next real observation. Decision-time planning is in a sense self-correcting: even if last step's plan was based on a flawed rollout, this step's plan starts fresh from where you actually are.

The trade-offs come down to *where you spend compute and how much you trust your model over distance*. Background planning front-loads compute (train once, act cheaply forever) but lets the policy drift toward whatever the model rewards, so it leans hard on uncertainty penalties to stay honest. Decision-time planning spends compute continuously (a search every step) but re-anchors to reality every step, so it tolerates a somewhat less accurate model at the cost of per-step latency. A useful way to choose: if your control frequency is high and your compute budget per step is tiny (a drone's flight controller), you almost have to use background planning and accept the model-exploitation risk. If you can afford tens of milliseconds of search per action and you want maximal robustness to model error (a slow manipulation task, a board game), decision-time planning is the safer bet.

![Decision tree for choosing a method, branching on whether real steps are expensive and whether observations are visual or low-dimensional.](/imgs/blogs/model-based-rl-learning-world-models-7.png)

| Aspect | Background planning (Dyna, Dreamer) | Decision-time planning (PETS, MPC, MuZero) |
| --- | --- | --- |
| When the model is used | Offline / between steps | Online, at every action |
| What it produces | An improved policy/value function | A single action, right now |
| Cost at action time | Cheap (read the policy) | Expensive (run a search) |
| Re-anchoring to real state | Indirect (via training data) | Direct (replans from current state) |
| Robustness to model error | Lower (policy can drift) | Higher (resets every step) |
| Best for | High-frequency control, reuse | Tasks where per-step compute is affordable |

The two are not mutually exclusive, and the strongest modern systems blend them. MuZero learns a value function (background) *and* runs MCTS at decision time, using the value function to truncate the search so it never plans too deep — exactly the short-horizon-plus-bootstrap pattern the compounding-error analysis demands. Dreamer learns its policy entirely in background imagination but uses a value function to summarize returns past its imagination horizon. The recurring pattern across all the strong systems is: *plan short with the model, bootstrap the rest with a learned value function, and choose background or decision-time based on your per-step compute budget.*

## 10. When MBRL helps and when it hurts

Let us be decisive, because this is where engineers waste the most time. The decision tree in the figure above encodes the logic; here is the reasoning behind each branch.

The first and most important question is: **how expensive is a real environment step?** If you have a fast, cheap simulator and can collect hundreds of millions of steps overnight — most game environments, most academic benchmarks — then model-free PPO or SAC is usually the right call. It is simpler, it has fewer failure modes, and it will often reach higher asymptotic performance because it is not capped by model error. Do not reach for MBRL to win a benchmark on a cheap simulator; you will spend weeks fighting model exploitation for a result a model-free baseline gets for free.

MBRL earns its complexity when **real steps are expensive**: physical robots (hardware wear, human supervision, wall-clock time), real-world control systems, expensive scientific or industrial experiments, anything where you genuinely cannot collect a million samples. There, a 10-100x reduction in real samples is the difference between feasible and impossible.

Within the "expensive steps" branch, the observation modality decides the algorithm. **Low-dimensional state** (joint angles, positions, velocities) points to PETS or PILCO — ensemble or GP models over a compact state vector. **High-dimensional visual observation** (camera pixels) points to **latent world models** like PlaNet and Dreamer, which learn a compressed latent state and model dynamics *in latent space* rather than in pixel space, because predicting raw future pixels is both wasteful and error-prone.

Where MBRL hurts: complex visual environments with sparse rewards and long horizons (the model errors compound faster than the planning helps), highly stochastic environments where the model can only predict distributions (planning becomes hedging), and any setting with severe distribution shift between training and deployment (the model's extrapolation is unreliable exactly where you need it). And the universal caveat: a model-based agent will exploit every flaw in its model, so plan for adversarial debugging — when your agent achieves an impossibly high *imagined* return but fails in reality, the model is lying and the optimizer believed it.

## 11. Case studies with numbers

**PILCO (Deisenroth & Rasmussen, 2011).** The headline result that put data-efficient MBRL on the map: PILCO learned to swing up and balance a cart-pole from scratch using roughly **17 trials**, totaling on the order of **20 seconds of total interaction time** — roughly an order of magnitude less interaction than the best model-free methods of the era, which needed thousands of episodes. Its secret was the Gaussian-process model's calibrated uncertainty, which it propagated analytically through the policy's expected cost so the policy optimizer never over-trusted the model. PILCO remains the cleanest demonstration that *uncertainty-aware* modeling, not just modeling, is what buys data efficiency. The cost — $O(n^3)$ GP inference — is why nobody runs PILCO on pixels, but the principle survived into every method below.

**PETS (Chua et al., 2018).** PETS showed that the GP's data efficiency could be recovered with neural networks — and thus scaled to higher dimensions — by using a *probabilistic ensemble*. On the MuJoCo HalfCheetah benchmark, PETS reached a return of roughly **6,000 in about 800 environment steps** (a handful of episodes of a few hundred steps each), matching the asymptotic performance of model-free SAC and PPO while using on the order of **10-25x fewer environment steps** — those model-free methods typically need hundreds of thousands of steps to reach comparable returns. PETS established the now-standard recipe: an ensemble of Gaussian-output networks, trajectory sampling to propagate uncertainty, and CEM-based MPC at decision time. It is the reference point for "neural MBRL that actually works on continuous control."

**Dreamer (Hafner et al., 2020) and its successors.** Dreamer learns a **latent world model** — a recurrent state-space model (RSSM) that compresses pixels into a compact latent and predicts dynamics, rewards, and values entirely in that latent space — and then trains an actor-critic *purely on imagined latent rollouts*, never touching pixels during policy learning. This is background planning at scale: the policy is trained on millions of cheap imagined trajectories. On the DeepMind Control Suite of visual continuous-control tasks, Dreamer solved a range of tasks from pixels in around **100k environment steps**, where model-free pixel-based methods needed many times more. DreamerV2 was the first agent to reach human-level performance on the Atari 200M benchmark using a world model, and DreamerV3 went further, solving a wide range of tasks — including the famously hard exploration challenge of collecting diamonds in Minecraft from scratch — with a *single* set of hyperparameters across domains. Dreamer is the clearest evidence that learned world models scale to visual, long-horizon problems where everyone once assumed only model-free methods could survive.

**MuZero (Schrittwieser et al., 2020).** MuZero unified planning and learning by learning a model *in a latent space optimized for planning* rather than for reconstruction — it never predicts the actual next observation, only the quantities needed to plan: reward, value, and policy. It then runs MCTS at decision time over this learned latent model. MuZero matched AlphaZero on Go, chess, and shogi *without being given the rules*, and set state-of-the-art on Atari. It is the strongest argument that you do not need to predict the world accurately in every detail — you only need to predict the parts that matter for choosing good actions.

#### Worked example: PETS sample efficiency versus model-free, in dollars

Make the PETS HalfCheetah result concrete in the currency that actually matters for a robotics team: time and money. Suppose you are training the equivalent task on a real legged robot rather than in simulation, at the robot rates from Section 1 — roughly 50 steps per 15-second episode, so about 3.3 real steps per second of robot operation, and say a fully-loaded cost of a few dollars per robot-hour once you count the rig, the supervision, and the wear. PETS reaching strong performance in roughly 800 steps means about 240 seconds — **four minutes** of robot motion, a fraction of a robot-hour, a few cents of wear. A model-free agent needing, conservatively, 300,000 steps to reach the same return means about 90,000 seconds, or **25 hours** of continuous robot operation — and that is assuming nothing breaks and no human ever has to reset a fallen robot, which on real hardware is a fantasy. The model-based agent finishes over lunch; the model-free agent ties up the hardware for a week and runs up a meaningful hardware-wear bill. *That* 300-to-1 ratio in real-step count is exactly why robotics labs invest in MBRL despite its extra complexity: the complexity is paid once, in engineering; the sample cost is paid every single run, on hardware you cannot easily replace.

To put the family on one comparison so the design space is concrete, here is the algorithm landscape across the axes that actually drive a choice:

| Algorithm | Real sample efficiency | Asymptotic performance | Model type | Compute overhead per real step |
| --- | --- | --- | --- | --- |
| DQN (model-free) | Very low ($10^7$–$10^8$ Atari) | High on its domain | None | Very low |
| PPO (model-free) | Low ($10^6$–$10^8$) | High | None | Low |
| SAC (model-free) | Low–moderate ($10^5$–$10^6$) | High, strong on continuous control | None | Low |
| Dyna-Q | High on tabular tasks | Optimal on small MDPs | Tabular dictionary | Very low (hash lookups) |
| PILCO | Extremely high ($\sim$17 trials) | High on low-dim control | Gaussian process | High ($O(n^3)$ GP inference) |
| PETS | Very high ($\sim$10-25x over SAC) | Matches SAC on MuJoCo | Probabilistic NN ensemble | High (CEM search every step) |
| MBPO | Very high | Matches/exceeds SAC | NN ensemble + short rollouts | Moderate (background rollouts) |
| Dreamer (V2/V3) | High on pixels ($\sim$100k DMC) | Human-level Atari, solves Minecraft | Latent RSSM | Moderate (imagination in latent space) |

The pattern in the table is the thesis of the whole post: as you move down from the model-free methods to the model-based ones, real sample efficiency climbs by one to three orders of magnitude, asymptotic performance stays roughly comparable (slightly capped by model error in the worst cases), the model gets more sophisticated, and the compute-per-real-step bill grows. You are, in every row, buying sample efficiency with engineering complexity and per-step compute. Whether that trade is worth it is decided entirely by how expensive your real steps are.

## When to use MBRL

Use model-based RL when real environment steps are genuinely expensive and you cannot simply collect more — robotics, real-world control, costly experiments — and when the dynamics are learnable enough that a model trained on modest data generalizes. Reach for PETS or PILCO on low-dimensional state, and for Dreamer or PlaNet on pixels. Always use an ensemble or otherwise quantify epistemic uncertainty, and always penalize the policy for relying on uncertain model regions; skipping this is the number-one cause of MBRL projects that look great in imagination and fail in reality. Keep your planning horizon short — $H \le 5$ in unstable systems, a few dozen steps at most in benign ones — and bootstrap everything past the horizon with a learned value function.

Do not use model-based RL when you have a cheap, fast simulator — model-free PPO or SAC is simpler and will likely match or beat MBRL's final performance without the model-exploitation headaches. For small, discrete, tabular problems with a *known* model, skip learning entirely and run value iteration or policy iteration directly; there is nothing to learn. And if your environment is dominated by irreducible stochasticity or severe distribution shift, be skeptical — the model can only predict what it has seen, and planning through a fog of aleatoric noise rarely beats a robust model-free policy. For the broader algorithm-selection picture across the whole family, the capstone `the-reinforcement-learning-playbook` lays out the full decision flow.

## Key takeaways

- A world model is a learned function $f(s, a) \to (s', r)$; model-based RL learns it first, then *plans* or *generates synthetic data* with it, trading modeling effort and per-step compute for real samples. The model learns from a dense per-step regression signal, which is why it is so much more data-efficient than learning from sparse reward alone.
- Dyna's three interleaved processes — direct RL on real data, model learning, and planning on simulated data — share one update rule, so the model-based contribution is "do the real update, then do $k$ more on imagined transitions." That single knob $k$ multiplies value updates per real step from $1$ to $(1+k)$ without collecting more data.
- Model error compounds along a rollout: linearly ($\approx \epsilon_1 T$) when the dynamics are benign ($L \approx 1$) and exponentially ($\approx \epsilon_1 (L^T-1)/(L-1)$) when they are unstable ($L > 1$). This is *why* modern methods plan short horizons (often $H \le 5$) and bootstrap the rest with a value function.
- Epistemic uncertainty (from limited data) is the dangerous kind, because it is where a policy optimizer chases hallucinations; estimate it with an ensemble's disagreement and penalize the policy's reward by it, $\tilde{r} = \hat{r} - \lambda u(s,a)$. Aleatoric uncertainty (irreducible noise) is captured by a Gaussian output head and you plan around it rather than resolving it.
- PILCO (GP, analytic uncertainty propagation, $\sim$17 trials on cart-pole) proved uncertainty-aware modeling buys data efficiency; PETS (probabilistic NN ensemble + CEM-MPC, $\sim$800 steps on HalfCheetah) scaled that principle to neural networks; Dreamer (latent RSSM, $\sim$100k steps on visual DMC) scaled it to pixels; MuZero learned a planning-only latent model and beat AlphaZero without the rules.
- Predict state *deltas*, normalize inputs and targets, and branch *short* rollouts from *real* buffer states to keep synthetic data anchored to reality — the MBPO pattern, justified by a bound on the optimal rollout length.
- Background planning (Dyna, Dreamer) front-loads compute to improve a stored policy and acts cheaply, ideal for high-frequency control; decision-time planning (PETS, MPC, MuZero) searches at every step and re-anchors to the true state, more robust to model error but slower per action. The best systems blend both.
- MBRL wins decisively in the low-data, expensive-step regime (a 10-100x real-sample reduction, which on real hardware is the difference between four minutes and twenty-five hours) and often *loses* asymptotically, because model approximation error caps performance once data is plentiful.
- A model-based agent will exploit every flaw in its model; when imagined return vastly exceeds real return, the model is lying and the optimizer believed it. Half of practical MBRL is engineering against that exploitation.

## Further reading

- Sutton, R. "Integrated Architectures for Learning, Planning, and Reacting Based on Approximating Dynamic Programming" (1991) — the original Dyna paper, including the three-process decomposition and prioritized sweeping.
- Sutton, R. & Barto, A. "Reinforcement Learning: An Introduction" (2nd ed., 2018) — Chapter 8 covers planning, Dyna, prioritized sweeping, and the background-versus-decision-time distinction rigorously.
- Deisenroth, M. & Rasmussen, C. "PILCO: A Model-Based and Data-Efficient Approach to Policy Search" (2011) — Gaussian-process dynamics and analytic policy-gradient propagation of uncertainty.
- Chua, K. et al. "Deep Reinforcement Learning in a Handful of Trials using Probabilistic Dynamics Models" (PETS, 2018) — probabilistic ensembles and trajectory sampling.
- Janner, M. et al. "When to Trust Your Model: Model-Based Policy Optimization" (MBPO, 2019) — the short-branched-rollout analysis and the optimal-rollout-length bound.
- Hafner, D. et al. "Dream to Control: Learning Behaviors by Latent Imagination" (Dreamer, 2020), "Mastering Atari with Discrete World Models" (DreamerV2, 2021), and "Mastering Diverse Domains through World Models" (DreamerV3, 2023).
- Schrittwieser, J. et al. "Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model" (MuZero, 2020) — planning-oriented latent models and decision-time MCTS.
- Within this series: `reinforcement-learning-a-unified-map` for where MBRL sits in the taxonomy, and `the-reinforcement-learning-playbook` for the full algorithm-selection decision flow.

The history of these ideas, from tabular Dyna to deep latent imagination, is worth seeing on one line.

![Timeline of model-based RL milestones from Dyna in 1991 through PILCO, PETS, PlaNet, Dreamer, and MuZero in 2020.](/imgs/blogs/model-based-rl-learning-world-models-6.png)

And the algorithm family, laid out by model type and planning style, makes the design space concrete.

![Matrix comparing Dyna-Q, PILCO, PETS, Dreamer, and MuZero across model type, planning style, visual-observation support, and continuous-action support.](/imgs/blogs/model-based-rl-learning-world-models-5.png)

The thread running through all of it is the spine of this whole series: an agent interacts with an environment, collects rewards, and updates a policy. Model-based RL adds one move — *learn the rules of the environment, then practice against your own copy* — and that single move, done carefully, can turn eleven days of robot time into less than an hour.
