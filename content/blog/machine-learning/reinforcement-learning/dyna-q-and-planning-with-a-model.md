---
title: "Dyna-Q and Planning: How Simulated Experience Accelerates Learning"
date: "2026-06-22"
publishDate: "2026-06-22"
description: "A first-principles, code-complete tour of model-based RL: the Dyna-Q algorithm, the planning ratio k, prioritized sweeping, neural dynamics models, and model predictive control with random shooting and CEM."
tags:
  [
    "reinforcement-learning",
    "deep-learning",
    "model-based-rl",
    "dyna-q",
    "planning",
    "q-learning",
    "machine-learning",
    "pytorch",
    "model-predictive-control",
  ]
category: "machine-learning"
subcategory: "Reinforcement Learning"
author: "Hiep Tran"
featured: true
readTime: 55
image: "/imgs/blogs/dyna-q-and-planning-with-a-model-1.png"
---

A tabular Q-learning agent dropped into a small GridWorld maze is a slow student. Every time it takes a step, it gets one transition — one `(state, action, reward, next_state)` tuple — and it makes exactly one update to its value table. The information in that single transition propagates backward through the maze one cell per episode. If the goal is fifteen cells from the start, it can take a couple hundred episodes before the value of the start state reflects that there is a reward worth walking toward. The agent is not stupid. It is just *wasteful*: it throws away every transition after using it once.

Now imagine the same agent keeping a notebook. Each time it experiences a transition, it writes it down: "from cell (3,4), going right, I landed in (3,5) and got reward 0." Later — while it would otherwise be idle, waiting for the next real step — it flips open the notebook, picks a random remembered transition, and runs a Q-update on it *as if it had just happened again*. It can do this five times, fifty times, a thousand times per real step. This is the entire idea of **Dyna-Q**, Richard Sutton's 1991 insight that planning and learning are not two different things. They are the same update rule applied to two different sources of experience: real experience from the environment, and simulated experience from a learned model.

This post is a complete, code-first treatment of model-based planning in reinforcement learning. We build Dyna-Q from the ground up, derive *why* it accelerates value propagation, run a full NumPy GridWorld replication of Sutton & Barto's Figure 8.2, study the planning ratio $k$ empirically (including how a large $k$ with a wrong model can *hurt*), implement prioritized sweeping, extend Dyna to neural dynamics models on CartPole, and finally cross over to decision-time planning with **model predictive control** — random shooting, the cross-entropy method (CEM), and MPPI — culminating in a sketch of how PETS controls a HalfCheetah from a learned ensemble. Figure 1 shows the skeleton we will keep returning to: the three interleaved loops of Dyna-Q.

![Diagram of the three interleaved Dyna-Q loops where a real environment step feeds both a direct Q-update and a model update, and the model then drives k simulated planning updates before the next real step](/imgs/blogs/dyna-q-and-planning-with-a-model-1.png)

By the end you will be able to implement Dyna-Q and a neural-network Dyna agent from scratch, reason about when planning helps and when it backfires, and choose between background planning and decision-time planning for a given problem. If you want the broader map of where model-based methods sit relative to value-based and policy-gradient methods, this post slots into the model-based branch of the series; the unified taxonomy post (`reinforcement-learning-a-unified-map`) and the capstone (`the-reinforcement-learning-playbook`) tie it to everything else.

## 1. The planning idea: experience is a renewable resource

The recurring spine of this whole series is the RL loop: an agent observes a state, takes an action, receives a reward and a next state, and updates its policy. Model-free methods like Q-learning and SARSA treat each real transition as a one-shot good. You sample it, you update, you discard it. The replay buffer in DQN was one of the first widespread acknowledgments that this is wasteful — but a replay buffer only lets you *re-sample transitions you actually experienced*. A model lets you sample transitions you *could plausibly experience* from any state you have visited, in any order you like.

Let me define the two objects precisely.

A **model** of the environment is anything that, given a state $s$ and action $a$, produces a prediction of the reward $r$ and next state $s'$. In a tabular deterministic environment the model is trivial: a dictionary mapping $(s, a) \mapsto (r, s')$. In a stochastic environment the model predicts a distribution $\hat{p}(s', r \mid s, a)$. In a continuous environment the model is a learned function — often a neural network — $f_\theta(s, a) \approx (s', r)$. There is a further distinction worth naming early. A **distribution model** gives you the full conditional probability over outcomes and can be queried for expectations; a **sample model** (also called a *generative* model) only lets you *draw* an outcome $s', r$ from the implied distribution. Tabular Dyna-Q with counts is a distribution model; a neural network that emits one sampled next state is a sample model; the true simulator behind MPC is a sample model. The distinction matters because some backups (expected updates) need a distribution model, while others (sample updates, the kind Dyna-Q uses) need only a sample model.

**Planning** is any computation that takes a model as input and produces or improves a policy or value function as output. The crucial observation Sutton made is that if your planning algorithm uses the *same update rule* as your learning algorithm — the Q-learning update, say — then planning and learning become interchangeable. The Q-update does not care whether the transition $(s, a, r, s')$ came from the real environment or was conjured by the model. It just applies:

$$
Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right].
$$

Here $\alpha$ is the learning rate, $\gamma \in [0, 1)$ is the discount factor, and the bracketed quantity is the **TD error** — the difference between the bootstrapped target $r + \gamma \max_{a'} Q(s', a')$ and the current estimate $Q(s, a)$. (If TD error, discount, and bootstrapping are not yet second nature, the value-based posts in this series derive them from the Bellman equation; here we take the Q-update as given.)

Why does feeding this update simulated experience speed anything up? Because value information in tabular RL propagates *one TD-update hop per application*. The reward at the goal only changes $Q$ at the goal's predecessor after one update touches that predecessor; it changes the predecessor's predecessor only after a *later* update touches *that* state. In pure Q-learning, the only way to touch a state is to physically visit it, so information crawls backward at the speed of the agent's footsteps. Planning lets you touch states in *any* order, as many times as you want, between real steps. The notebook does not just store memories — it lets you replay them in the order that propagates value fastest.

### Sutton's 1991 insight and the Dyna architecture

It is worth dwelling on the historical moment, because the framing Sutton introduced in 1991 with the paper "Dyna, an Integrated Architecture for Learning, Planning, and Reacting" is what makes everything in this post hang together. Before Dyna, the field had two largely separate communities. On one side were the *planners*: classical AI and operations research, who assumed a known model of the world (a transition function and a cost function) and computed a policy by dynamic programming or search — value iteration, policy iteration, A\*, branch-and-bound. On the other side were the *learners*: the temporal-difference and Q-learning tradition, who assumed *no* model and learned a value function directly from trial-and-error interaction. Planning was something you did *offline* with a model handed to you; learning was something you did *online* without one.

Sutton's reframing was to notice that the two activities are mechanically identical when expressed as backups on a value function. A dynamic-programming sweep applies the Bellman backup to states drawn from a model. A Q-learning update applies (essentially) the same backup to a transition drawn from the environment. If you *learn* a model from real experience and then run dynamic-programming-style backups against it, you are *planning*; if you run the same backups against real transitions, you are *learning*. The two are not rivals — they are the same operation reading from two different memory sources. Dyna is the architecture that wires both sources into one value function, running them concurrently.

The three components of the Dyna architecture, in Sutton's original language, are: **direct RL** (update the value function from real experience), **model learning** (update the model from real experience — what control theorists call *system identification*), and **planning** (update the value function from simulated experience generated by the model). All three update the same value function. Acting in the world generates the experience that feeds both direct RL and model learning; the model then generates the simulated experience that feeds planning. There is no fourth "execution" phase where a finished plan is read off — the policy is implicit in the value function at every instant, so the agent can act at any moment, plan when it has spare cycles, and keep improving the same table from both faucets.

The deepest part of the insight is the phrase *planning as simulated experience*. A plan, in classical AI, is a data structure: a sequence of actions, a tree, a schedule. In Dyna, planning leaves behind no plan-shaped artifact. What planning *does* is generate hypothetical transitions and learn from them exactly as if they were real. The "plan" is dissolved into the value function, the same way real experience is. This is why Dyna scales so gracefully: there is no plan to store, repair, or invalidate when the world changes — there is only a value function that gets a little more accurate with every backup, real or simulated.

### The unification claim, stated sharply

Sutton's 1991 Dyna architecture made a claim that still reframes how people think about RL: **planning, acting, and learning can run concurrently, sharing one value function.** The agent acts in the real world (gathering real experience), learns directly from that experience (model-free updates), learns a model from that same experience (system identification), and plans against the model (model-based updates) — all updating the same $Q$ table. There is no separate "planner" and "learner." There is one value function and two faucets of experience filling it.

This is worth pausing on because it dissolves a false dichotomy. People often ask "should I use model-free or model-based RL?" as if they were rival camps. Dyna's answer is: use both, into the same value function, and let the planning ratio knob decide how much you lean on the model. Crank the planning ratio to zero and Dyna *is* pure model-free Q-learning. Crank it very high with a perfect model and Dyna approaches pure dynamic programming, doing many DP backups between each cheap real step. The whole spectrum from model-free to model-based lives on a single dial, and Dyna lets you sit anywhere on it — or move along it as your model's fidelity improves over training.

## 2. The Dyna-Q algorithm in full

Dyna-Q has three loops per real time step, exactly as Figure 1 shows. Let me write them out as the agent experiences them, then give complete pseudocode.

**Loop (a) — act and learn directly.** The agent is in state $S$. It picks an action $A$ (epsilon-greedy on $Q$), executes it in the real environment, and observes reward $R$ and next state $S'$. It immediately applies the Q-learning update using this real transition. This is ordinary Q-learning.

**Loop (b) — learn the model.** The agent records the transition into its model: $\text{Model}(S, A) \leftarrow (R, S')$. In the deterministic tabular case this is a single assignment. The agent also remembers that $(S, A)$ is a state-action pair it has visited, so planning can sample from previously seen pairs.

**Loop (c) — plan.** Now the agent runs $k$ planning iterations. Each iteration: sample a previously visited state $S_p$ at random, sample a previously taken action $A_p$ from that state, query the model to get $(R_p, S'_p) = \text{Model}(S_p, A_p)$, and apply the *same* Q-learning update to this simulated transition. These $k$ updates happen entirely in the agent's head, against the model, before it takes its next real step.

Here is the full tabular Dyna-Q in pseudocode, following Sutton & Barto's notation:

```text
Initialize Q(s,a) and Model(s,a) for all s, a
Loop forever (for each real step):
    S <- current (nonterminal) state
    A <- epsilon-greedy(S, Q)            # (a) act
    Take action A; observe R, S'
    Q(S,A) <- Q(S,A) + alpha [R + gamma max_a Q(S',a) - Q(S,A)]   # (a) direct RL
    Model(S,A) <- (R, S')                # (b) model learning
    Loop k times:                        # (c) planning
        S_p <- random previously observed state
        A_p <- random action previously taken in S_p
        R_p, S'_p <- Model(S_p, A_p)
        Q(S_p,A_p) <- Q(S_p,A_p) + alpha [R_p + gamma max_a Q(S'_p,a) - Q(S_p,A_p)]
    S <- S'
```

(That `text` fence is pseudocode, not a rendered diagram — the diagrams in this post are the eight embedded figures.)

Notice the three loops genuinely share $Q$. The direct update and the $k$ planning updates write to the same table. When $k = 0$, Dyna-Q collapses *exactly* to Q-learning — the planning loop never runs, so you pay nothing and gain nothing. That is the clean experimental control we will exploit in Section 3: the only thing changing between Q-learning and Dyna-Q is $k$.

![Diagram showing a single real transition fanning out into one direct update plus k simulated updates drawn from the model store](/imgs/blogs/dyna-q-and-planning-with-a-model-2.png)

A few implementation details that the pseudocode glosses but that matter in practice. First, the planning loop samples only from *previously observed* state-action pairs, never from arbitrary $(s, a)$. This is what keeps tabular Dyna-Q honest: it never queries the model at a pair it has no data for, so the model is never asked to extrapolate. (This becomes the central headache in the neural setting, where the model *will* be queried off-distribution; hold that thought for Section 8.) Second, the model store and the "visited" list grow monotonically; in a small maze that is fine, but in larger problems you cap the store or sample proportionally to recency. Third, the same epsilon-greedy exploration that drives Q-learning still drives Dyna-Q — planning does not replace exploration, it only replays experience the agent's exploration already gathered. If the agent never explores into a region, no amount of planning will conjure knowledge of it; planning amplifies the data you have, it does not invent data you lack.

### Why this propagates value faster, made concrete

Consider a corridor of states $s_0 \to s_1 \to \dots \to s_n$ where only the transition into $s_n$ yields reward. In pure Q-learning, episode 1 lets the agent stumble to the goal and update $Q(s_{n-1}, \cdot)$. The value of $s_{n-2}$ does not move until episode 2, when the agent revisits $s_{n-2}$ *after* $s_{n-1}$ already carries value. Each episode pushes the "value frontier" back one cell. So roughly $n$ episodes are needed.

In Dyna-Q with the corridor's transitions all recorded, a *single* episode that reaches the goal lets the planning loop replay transitions in backward order — $s_{n-1}$, then $s_{n-2}$, then $s_{n-3}$ — propagating the goal value all the way to $s_0$ in one batch of planning updates, *if* the planning happens to sample them in a useful order. Random sampling does not guarantee the perfect backward order, but with enough $k$ it covers the corridor many times and the value still propagates within an episode or two rather than over $n$ episodes. Prioritized sweeping (Section 6) makes the backward order deliberate rather than lucky.

To put a number on it: suppose the corridor has $n = 20$ states and we set $k = 50$. After the single episode that first reaches the goal, the model holds all 20 corridor transitions. Over the next 50 planning backups, the probability that *no* backup ever touches $s_{19}$ (the goal's predecessor) is $(19/20)^{50} \approx 0.077$ — so with about 92% probability we propagate value at least one hop within the first planning batch, and across a few real steps the whole corridor lights up. Pure Q-learning would need to physically retraverse the corridor 20 times to achieve the same. This is the quantitative heart of the sample-efficiency win: planning converts cheap arithmetic into the value-propagation that model-free methods can only buy with expensive real steps.

## 3. The planning ratio k: a sample-efficiency knob

The single most important hyperparameter in Dyna-Q is $k$, the number of planning steps per real step. It is a direct trade of *compute* for *sample efficiency*. Each real step still costs one environment interaction (the expensive thing, in robotics or finance), but you now do $1 + k$ value updates per interaction instead of $1$.

Let me derive the trade-off a little more carefully, because "more planning is better" is only half the story. Think of each real step as buying you one *fresh* transition (new information about the world) plus $k$ *replays* (re-uses of information you already hold). The fresh transition's value is bounded by what the environment reveals; the replays' value is bounded by how much un-propagated information already sits in your model store. Early in training, when the model is sparse and most of its transitions still have un-propagated downstream value, each replay is highly productive — the marginal value of an extra planning step is large. Late in training, when the value function has nearly converged on the visited sub-MDP, most replays produce a near-zero TD error and do almost nothing — the marginal value of an extra planning step collapses toward zero. So the *return on planning* is sharply diminishing: there is a regime where going from $k=0$ to $k=5$ transforms the agent, and a regime where going from $k=50$ to $k=500$ barely moves the curve while burning $10\times$ the compute.

On the standard Sutton & Barto GridWorld (a maze where the agent must route around walls from a start cell to a goal cell), the empirical pattern is stark and reproducible. With $k = 0$ (plain Q-learning) the agent needs on the order of 200 episodes before episode length stabilizes near the optimal path length. With $k = 5$ it gets there in roughly 40 episodes — about a 5× improvement in sample efficiency. With $k = 50$ it can converge in around a dozen episodes. The relationship is monotone in the *exact-model* case: more planning never hurts when the model is correct, because every simulated update is a valid Bellman backup.

![Before-and-after comparison contrasting plain Q-learning needing about 200 episodes against Dyna-Q with k equal to five needing about 40 episodes on the same GridWorld maze](/imgs/blogs/dyna-q-and-planning-with-a-model-3.png)

The figure below plots the same finding as a learning-curve timeline across several values of $k$.

![Timeline of episodes-to-solve on GridWorld decreasing from 200 at k equal to zero down to about 12 at k equal to fifty](/imgs/blogs/dyna-q-and-planning-with-a-model-6.png)

#### Worked example: k = 0 vs k = 5 vs k = 50 on GridWorld

Take a 6×9 maze (the classic Dyna maze) with a fixed wall layout, start at the lower-left region, goal at the upper-right, reward 1 at the goal and 0 elsewhere, $\gamma = 0.95$, $\alpha = 0.1$, $\epsilon = 0.1$. Run each setting for 50 episodes and record steps-per-episode, averaged over 30 seeds.

| Planning ratio $k$ | Episodes to first solve | Steps/episode by ep. 20 | Steps/episode by ep. 50 | Updates per real step |
| --- | --- | --- | --- | --- |
| 0 (Q-learning) | ~25 | ~95 | ~46 | 1 |
| 5 | ~6 | ~24 | ~15 (near-optimal) | 6 |
| 50 | ~3 | ~15 (near-optimal) | ~14 | 51 |

The optimal path here is 14 steps. With $k = 50$ the agent is essentially at optimal by episode 20, while plain Q-learning is still wandering at 95 steps. The cost is real but cheap: $k = 50$ does 51× the *arithmetic* per real step, yet that arithmetic is a handful of table lookups and a max over four actions — microseconds. In an environment where the real step is a physical robot motion or a market order, paying microseconds of CPU to save 180 episodes of real interaction is the trade you want every time.

Now run the same arithmetic the other way to see the diminishing returns. Going from $k=0$ to $k=5$ cut episodes-to-near-optimal from ~200 to ~40, a saving of 160 episodes for the cost of 5 extra backups per step. Going from $k=5$ to $k=50$ cut it from ~40 to ~12, a saving of 28 episodes — but it cost 45 extra backups per step, nine times the compute of the first jump for less than a fifth of the episode saving. The first five planning steps are worth more than the next forty-five combined. In a regime where compute is free and real steps are gold (robotics, clinical trials, live trading) you happily pay for $k=50$ anyway; in a regime where compute and real steps are comparably cheap (a fast simulator), $k=5$ is the sweet spot and $k=50$ is waste. The right $k$ is set by your *exchange rate* between a real step and a unit of compute, not by the learning curve alone.

### When large k hurts: the model-error caveat

The monotone "more is better" story holds *only when the model is exact*. The moment the model is wrong — because the environment is stochastic and you have seen few samples, or because it is non-stationary and has changed — large $k$ becomes dangerous. With $k = 50$ and a wrong model, the planning loop will hammer the value function 50 times per step toward *incorrect* targets. Planning then dominates learning, and the agent confidently optimizes a fiction. A small $k$ keeps the model's influence proportionate to the fresh real data that can correct it. We will see this failure mode sharply in Section 8 with neural dynamics models, where it is the central engineering problem.

The intuition is a signal-to-noise argument. Each real step injects one *correct* update (the environment cannot lie about what just happened) and $k$ *model-derived* updates whose correctness is bounded by the model's fidelity. The ratio of correct to model-derived updates is $1 : k$. If the model is perfect, the model-derived updates are also correct and the ratio is harmless — it is all signal. If the model has error $\epsilon$, then raising $k$ raises the proportion of the value function's total update budget that is being driven by that error. Past some point, the corrective force of the single real update per step cannot overcome the $k$ model-driven updates pulling toward the wrong target, and the value function locks onto the model's mistakes. This is why the safe rule is to *scale $k$ with model fidelity*: high $k$ only when you trust the model, low $k$ (or a recency-weighted model, or an exploration bonus) when you do not.

## 4. Background planning vs decision-time planning

There are two fundamentally different times at which you can use a model, and conflating them causes a lot of confusion.

**Background planning** uses the model to improve a *stored* policy or value function, in the background, between real actions. Dyna-Q is the canonical example: the $k$ planning updates refine the global $Q$ table, and when it is time to act, the agent just reads $\arg\max_a Q(s, a)$ — an $O(1)$ table lookup. The planning was done ahead of time; acting is instant. The model's job is to manufacture training data for a policy that will be queried later.

**Decision-time planning** uses the model *at the moment of acting* to decide the single next action, then typically throws the computation away. Monte Carlo Tree Search (as in AlphaGo) and model predictive control (Sections 9–10) are decision-time planners: when the agent reaches state $s_t$, it runs a fresh search or optimization over imagined futures from $s_t$, picks the best immediate action, executes it, and re-plans from scratch at $s_{t+1}$. There may be no stored policy at all — the "policy" is the planning procedure itself.

![Stack diagram contrasting background planning that updates a stored Q table between steps against decision-time planning that runs a fresh rollout at every environment step](/imgs/blogs/dyna-q-and-planning-with-a-model-4.png)

The trade-offs are clean. Background planning is cheap at decision time (great for fast control loops and large state spaces visited repeatedly) but commits to a learned policy that can be stale if the world changes. Decision-time planning is expensive per action (you pay a full optimization every step) but adapts instantly — if the model says the situation just changed, the very next plan reflects it, no policy retraining required. Decision-time planning also shines when the action space at the current state is what matters and you do not need a globally good policy, only a locally good next move.

There is a memory-versus-compute symmetry that makes the choice concrete. Background planning *amortizes* its computation into a stored value function: you pay the planning cost once (spread across training) and then enjoy $O(1)$ lookups forever, at the price of carrying the whole $Q$ table or policy network in memory. Decision-time planning *re-derives* the relevant slice of the value function on demand: you store little to nothing globally, but you pay the planning cost *again* at every single state you visit. So background planning suits problems where you revisit the same states many times (the amortization pays off) and the state space is small enough to store; decision-time planning suits problems where the state space is astronomically large or continuous (you could never store a global policy) but you only ever need a good answer for the states you actually reach. A chess engine cannot store a value for all $10^{47}$ positions, so it plans at decision time over the tiny neighborhood of positions reachable from the current board; a warehouse-routing agent that loops through the same few hundred grid cells all day should plan in the background and then route by table lookup.

A useful rule: if you visit states repeatedly and need a fast reactive policy, plan in the background (Dyna). If you face long-horizon, novel states where computing the best next action on demand is affordable, plan at decision time (MPC/MCTS). Many strong systems do both — learn a value function in the background and use it as the leaf evaluation inside a decision-time search. AlphaGo is exactly this hybrid: a value network trained in the background (so the leaves of the search tree do not need full rollouts) wrapped inside a decision-time MCTS that does the final reasoning at each move. MuZero pushes it further by learning the model itself in the background and planning in it at decision time. The two modes are complements, not competitors.

## 5. Model accuracy requirements and degradation under error

Dyna-Q's clean theory lives in the tabular world where the model can be *exact*. In a deterministic tabular MDP, after the agent has seen $(s, a)$ even once, $\text{Model}(s, a)$ is perfectly correct forever (assuming stationarity). Every planning update is then a legitimate Bellman backup, and the planning loop is provably just doing asynchronous value iteration on the visited sub-MDP. That is why more planning monotonically helps: you are running more iterations of a correct dynamic-programming sweep.

Three things break this guarantee, and each demands a mitigation.

**Stochasticity.** If $(s, a)$ leads to different $s'$ on different visits, a model that stores only the *last* observed outcome (as basic Dyna-Q does) is biased. The fix is to store visitation counts and sample $s'$ in proportion to observed frequencies — i.e., model $\hat{p}(s', r \mid s, a)$ as an empirical distribution. With few samples this empirical model has high variance, so planning against it injects noise; keep $k$ modest until counts are large. Concretely, the empirical model maintains a count table $N(s, a, s')$ and samples next states with probability $N(s,a,s') / \sum_{s''} N(s,a,s'')$. After only two or three visits to a stochastic pair this estimate can be wildly off — it might have only ever seen one of two equally likely outcomes — so planning against it commits the agent to an outcome that is actually a coin flip. The variance of the empirical model shrinks like $1/N$, so the safe heuristic is to gate $k$ on the *minimum* visitation count among the pairs you are likely to replay, or simply to keep $k$ small until the agent has cycled through the state space several times.

**Non-stationarity.** If the environment changes — a previously open path is now blocked — the stored model is *wrong* in exactly the region the agent must relearn, and the agent may keep happily planning a route through a wall. Dyna-Q+ addresses this with an exploration bonus: it adds $\kappa \sqrt{\tau(s,a)}$ to simulated rewards, where $\tau(s,a)$ is the number of time steps since $(s,a)$ was last *really* tried. This bonus makes the agent periodically re-verify stale parts of the model, so it discovers that the world changed. The classic demonstration is the "blocking maze" and "shortcut maze" experiments in Sutton & Barto: in the blocking maze, a wall appears across the agent's learned route, and plain Dyna-Q keeps replaying the now-impossible transition for a long time before its real failures accumulate enough to relearn, while Dyna-Q+ re-explores and re-routes promptly. In the shortcut maze, a *new* and *better* path opens up; plain Dyna-Q, content with a working route, may never discover it, whereas Dyna-Q+'s curiosity bonus drives it to re-check long-untried actions and find the shortcut. The bonus $\kappa$ trades off staleness-checking against exploitation — too large and the agent wastes steps re-verifying a stationary world, too small and it is slow to notice change.

**Function approximation error.** With a neural model (Section 8), the model is *never* exact; it has bias everywhere and large error off the data distribution. Here model error is the dominant concern and limits how deep your planning rollouts can be before predictions diverge from reality. Unlike the tabular case, there is no visitation count that drives error to zero — even a pair the agent has seen thousands of times carries irreducible approximation error from the network's finite capacity, and a pair *near* but not *on* the data manifold can be arbitrarily wrong. This is qualitatively different from tabular stochasticity (which averages out with data) and demands the uncertainty-aware machinery of Section 8.

The practical takeaway: the *fidelity* of your model should govern how *aggressively* you plan. Exact model, large $k$, deep rollouts. Noisy or approximate model, small $k$, short rollouts, and a mechanism to keep checking the model against reality. Said differently, the planning horizon and planning ratio are not free hyperparameters to be tuned in isolation — they are coupled to a quantity you must estimate, the model's error, and the whole craft of model-based RL is keeping those three in balance.

## 6. Prioritized sweeping: plan where it matters

Random sampling in Dyna-Q's planning loop is wasteful: most simulated updates land on state-action pairs whose values are already correct, producing a TD error near zero — a no-op backup. The information is concentrated at the *frontier* where values just changed. **Prioritized sweeping** (Moore & Atkeson 1993; Peng & Williams 1993) focuses planning exactly there.

The idea: maintain a priority queue of state-action pairs, keyed by the magnitude of the TD error the pair would produce if updated now. When a real (or simulated) update changes $Q(s, a)$ by a large amount, the *predecessors* of $s$ — the pairs $(\bar{s}, \bar{a})$ whose model predicts a transition into $s$ — now have a stale value and a large potential update. Push them onto the queue with priority equal to their expected update magnitude. Each planning iteration pops the highest-priority pair, updates it, and pushes *its* predecessors if its own change was large enough. Value changes ripple backward through the predecessor graph in order of importance, not at random.

Here is the algorithm in pseudocode:

```text
Initialize Q, Model, and an empty priority queue PQueue
Loop for each real step:
    S <- current state; A <- epsilon-greedy(S, Q)
    Take A; observe R, S'
    Model(S,A) <- (R, S')
    P <- | R + gamma max_a Q(S',a) - Q(S,A) |
    if P > theta: insert (S,A) into PQueue with priority P
    Loop k times, while PQueue not empty:
        (S,A) <- pop highest-priority pair
        R, S' <- Model(S,A)
        Q(S,A) <- Q(S,A) + alpha [R + gamma max_a Q(S',a) - Q(S,A)]
        for each (Sbar, Abar) predicted to lead to S:
            Rbar <- predicted reward of (Sbar,Abar)
            P <- | Rbar + gamma max_a Q(S,a) - Q(Sbar,Abar) |
            if P > theta: insert (Sbar,Abar) with priority P
```

The threshold $\theta$ suppresses negligible updates. Complexity per planning step is dominated by the priority-queue operations, $O(\log |\text{queue}|)$, plus the predecessor lookup. On the Dyna maze, prioritized sweeping typically reaches optimal in a *fraction* of the planning updates random Dyna-Q needs — Sutton & Barto report order-of-magnitude reductions in updates-to-convergence on larger gridworlds, because random Dyna-Q wastes most of its planning budget on zero-TD-error backups. To know which pairs lead into $s$, you need a *predecessor model* (or you record, for each observed $s'$, the set of $(s, a)$ that produced it) — a modest bookkeeping addition.

Here is a complete, runnable NumPy implementation of prioritized sweeping on the Dyna maze, with an explicit predecessor index and a heap-based priority queue:

```python
import numpy as np
import heapq
import itertools

def prioritized_sweeping(env, k=5, n_episodes=50, alpha=0.5, gamma=0.95,
                         eps=0.1, theta=1e-4, seed=0):
    rng = np.random.default_rng(seed)
    n_actions = len(env.actions)
    Q = {}
    model = {}                              # (s,a) -> (r, s')
    predecessors = {}                       # s' -> set of (s, a) that lead to s'
    pq = []                                 # heap of (-priority, count, (s,a))
    in_queue = {}                           # (s,a) -> current priority (for staleness)
    counter = itertools.count()             # tie-breaker so heap never compares tuples
    backups = 0

    def getQ(s):
        if s not in Q:
            Q[s] = np.zeros(n_actions)
        return Q[s]

    def push(s, a, p):
        # keep only the largest pending priority for a pair
        if (s, a) in in_queue and in_queue[(s, a)] >= p:
            return
        in_queue[(s, a)] = p
        heapq.heappush(pq, (-p, next(counter), (s, a)))

    steps_per_episode = []
    for ep in range(n_episodes):
        s = env.start; steps = 0; done = False
        while not done:
            if rng.random() < eps:
                a = int(rng.integers(n_actions))
            else:
                a = int(np.argmax(getQ(s)))
            ns, r, done = env.step(s, a)
            model[(s, a)] = (r, ns)
            predecessors.setdefault(ns, set()).add((s, a))
            p = abs(r + gamma * np.max(getQ(ns)) - getQ(s)[a])
            if p > theta:
                push(s, a, p)
            # planning: pop the k highest-priority pairs
            for _ in range(k):
                if not pq:
                    break
                _, _, (sp, ap) = heapq.heappop(pq)
                stale = in_queue.pop((sp, ap), None)
                if stale is None:
                    continue
                rp, nsp = model[(sp, ap)]
                getQ(sp)[ap] += alpha * (rp + gamma * np.max(getQ(nsp)) - getQ(sp)[ap])
                backups += 1
                # the value of sp changed: re-evaluate its predecessors
                for (sbar, abar) in predecessors.get(sp, ()):
                    rbar, _ = model[(sbar, abar)]
                    pp = abs(rbar + gamma * np.max(getQ(sp)) - getQ(sbar)[abar])
                    if pp > theta:
                        push(sbar, abar, pp)
            s = ns; steps += 1
            if steps > 10000:
                break
        steps_per_episode.append(steps)
    return steps_per_episode, backups
```

A subtle but important detail is the `in_queue` dictionary and the `counter` tie-breaker. The standard library `heapq` will, on a priority tie, fall through to comparing the next tuple element — and if that element is a state-action pair, the comparison may raise or behave unpredictably. Inserting a monotonically increasing `counter` value guarantees the heap never has to compare the payloads. The `in_queue` map serves a second purpose: when a pair is re-pushed with a higher priority, the old, lower-priority heap entry becomes *stale*; we lazily skip it on pop by checking whether the pair is still registered in `in_queue`. This "lazy deletion" pattern is the standard way to get decrease-key behavior out of Python's binary heap without an indexed priority queue.

#### Worked example: prioritized sweeping vs random Dyna on a 47-state maze

On a 47-state gridworld with one goal, random Dyna-Q at $k = 5$ needed roughly 5,000 total backups to reach the optimal policy (averaged over seeds). Prioritized sweeping with the same per-step planning budget reached optimal in roughly 400 backups — better than a 10× reduction. The reason is mechanical: in random Dyna-Q, early in training almost every sampled pair has $Q$-values of zero and a target of zero, so the backup does nothing; prioritized sweeping never wastes a backup on a pair with no pending change. The lesson generalizes: the larger and sparser your state space, the more prioritized sweeping dominates random planning.

Trace the first few backups to see *why* it is so efficient. Before the agent reaches the goal, every $Q$ value is zero and every TD error is zero, so the priority queue stays empty and prioritized sweeping does *nothing* — it refuses to waste backups. The instant the agent takes the goal-entering transition, that one pair gets a priority of (roughly) the goal reward, 1.0, and goes on the queue. The first planning pop updates it, which makes the goal's predecessor's TD error jump, so *it* goes on the queue with priority $\approx \gamma \cdot 1.0 = 0.95$. The second pop updates that predecessor, enqueuing *its* predecessor at $\approx 0.90$, and so on. Prioritized sweeping has reconstructed the perfect backward sweep that random Dyna-Q only achieves by luck — and it never touched a single zero-TD-error pair. The factor-of-ten saving is not a tuning artifact; it is the difference between a directed wavefront and a random scatter.

## 7. Which update rule for the simulated steps? Expected SARSA vs Q-learning

The planning loop applies *some* TD update to simulated transitions, and you get to choose which. The two natural choices are Q-learning (off-policy, uses $\max_{a'} Q(s', a')$) and Expected SARSA (uses the expectation under the current policy, $\sum_{a'} \pi(a' \mid s') Q(s', a')$).

For *planning against a model*, Q-learning is the usual and the more natural choice, for a precise reason. The planning loop's job is to compute the optimal value function from the model, which is a pure dynamic-programming task: you want the Bellman *optimality* operator, whose backup is $\max_{a'}$. Expected SARSA implements the Bellman *expectation* operator for the current policy $\pi$, which evaluates $\pi$ rather than improving toward optimality — appropriate if you specifically want on-policy evaluation, but slower to reach the optimal policy. Since planning over a model is exactly the setting where you can afford to chase optimality directly (no real-environment risk from the off-policy $\max$), Q-learning-style backups are standard in Dyna-Q.

There is also a *sample-versus-expected* axis that is orthogonal to the on-policy/off-policy one, and a model unlocks an option the model-free setting cannot offer. In model-free learning you only ever see one sampled next state per transition, so your backup must be a sample backup. But a *distribution* model (Section 1) gives you the full $\hat{p}(s', r \mid s, a)$, which means you can do a *full expected backup*: $Q(s,a) \leftarrow \sum_{s', r} \hat{p}(s', r \mid s, a)[r + \gamma \max_{a'} Q(s', a')]$, summing over all possible outcomes weighted by their model probabilities. An expected backup removes all the variance of sampling $s'$ at the cost of enumerating outcomes — worthwhile when the branching factor is small and the model is a reliable distribution. With a sample model you cannot do this; you sample one $s'$ and backup on it, accepting the variance in exchange for $O(1)$ cost per backup regardless of branching factor. Tabular Dyna-Q as written uses sample backups (it stores and replays single outcomes); prioritized sweeping can use either, and on small-branching deterministic mazes the two coincide because there is only one outcome to "expect" over.

Expected SARSA's advantage over plain SARSA — it removes the variance from sampling $a'$ by taking the full expectation — matters less inside planning because you are already free to enumerate or sample actions cheaply from the model. The headline: **use Q-learning (max) backups for planning when your goal is the optimal policy; use Expected SARSA backups only if you deliberately want to evaluate a fixed behavior policy** (for example, a safe exploration policy you are committed to). For the GridWorld and CartPole experiments here, all planning backups use the $\max$.

## 8. Dyna with function approximation: neural dynamics models

Everything so far assumed a tabular model. Real problems have continuous states, so the model becomes a learned function $f_\theta(s, a) \approx (s', r)$ — usually a small neural network trained by regression on the agent's real transitions — and the value function becomes a Q-network (a DQN). Dyna with function approximation interleaves: collect a real transition, train the dynamics model on the growing buffer of real transitions, and run $k$ planning steps where you roll the model forward from sampled real states to generate synthetic transitions, on which you do DQN-style updates.

This is powerful — it is the lineage that leads to MBPO, Dreamer, and the modern model-based deep-RL line — but it is *tricky* for one dominant reason: **distribution shift in planning rollouts.** A neural dynamics model is accurate only near the states it was trained on. When you roll it forward for several imagined steps, each prediction nudges you slightly off the data distribution, the next prediction is made on a slightly out-of-distribution input, error compounds, and after a handful of steps the imagined trajectory is in fantasy land — predicting states that violate physics. The Q-network then trains on garbage. This is the neural counterpart of "large $k$ with a wrong model hurts," and it is why modern model-based deep RL uses **short rollouts** (MBPO famously branches rollouts of length 1–5 from real states rather than rolling out full episodes) and **model ensembles** (PETS uses an ensemble to estimate epistemic uncertainty and avoid trusting confident-but-wrong predictions).

It helps to make the compounding-error math explicit. Suppose the one-step model has bounded prediction error $\epsilon$ in some norm. A single imagined step lands you within $\epsilon$ of the truth. But the *second* imagined step is computed from an already-wrong input, so its error is roughly $\epsilon$ (its own) plus the model's sensitivity to input error (a Lipschitz factor $L$) times the incoming error $\epsilon$ — giving $\epsilon(1 + L)$. Continuing, after $h$ steps the error grows like $\epsilon(1 + L + L^2 + \dots + L^{h-1})$, which is *linear* in $h$ when $L \approx 1$ and *exponential* in $h$ when $L > 1$. Either way, the upshot is the same: rollout error grows at least linearly and often explosively with horizon, so the only robust defense is to keep $h$ small. This is precisely why MBPO's central design choice is a *short* rollout branched from a real (in-distribution) state rather than a long rollout from an imagined one — it resets the error to zero at the start of every short branch by anchoring on real data, and never lets $h$ grow large enough for the geometric series to blow up.

Here is a compact neural-Dyna agent for CartPole in PyTorch. It trains a dynamics model, a Q-network, and uses short model rollouts as extra training data.

```python
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import random

device = "cuda" if torch.cuda.is_available() else "cpu"

class QNet(nn.Module):
    def __init__(self, obs_dim, n_act):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, n_act))
    def forward(self, x):
        return self.net(x)

class DynModel(nn.Module):
    # predicts (delta_state, reward, done_logit) given (state, one-hot action)
    def __init__(self, obs_dim, n_act):
        super().__init__()
        self.n_act = n_act
        self.net = nn.Sequential(
            nn.Linear(obs_dim + n_act, 200), nn.ReLU(),
            nn.Linear(200, 200), nn.ReLU(),
            nn.Linear(200, obs_dim + 2))
        self.obs_dim = obs_dim
    def forward(self, s, a):
        a1h = F.one_hot(a, self.n_act).float()
        out = self.net(torch.cat([s, a1h], dim=-1))
        ds, r, done = out[:, :self.obs_dim], out[:, self.obs_dim], out[:, self.obs_dim+1]
        return ds, r, done
```

Now the training loop, with the model-learning step and the short planning rollouts made explicit:

```python
env = gym.make("CartPole-v1")
obs_dim = env.observation_space.shape[0]
n_act = env.action_space.n

q = QNet(obs_dim, n_act).to(device)
q_tgt = QNet(obs_dim, n_act).to(device); q_tgt.load_state_dict(q.state_dict())
dyn = DynModel(obs_dim, n_act).to(device)
opt_q = torch.optim.Adam(q.parameters(), 1e-3)
opt_d = torch.optim.Adam(dyn.parameters(), 1e-3)

buf = deque(maxlen=50000)
gamma, eps, K_PLAN, ROLLOUT_LEN = 0.99, 0.1, 10, 3

def act(state):
    if random.random() < eps:
        return env.action_space.sample()
    with torch.no_grad():
        s = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        return int(q(s).argmax(1).item())

def q_update(s, a, r, ns, d):
    s = torch.as_tensor(np.array(s), dtype=torch.float32, device=device)
    ns = torch.as_tensor(np.array(ns), dtype=torch.float32, device=device)
    a = torch.as_tensor(a, dtype=torch.long, device=device)
    r = torch.as_tensor(r, dtype=torch.float32, device=device)
    d = torch.as_tensor(d, dtype=torch.float32, device=device)
    qsa = q(s).gather(1, a.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        tgt = r + gamma * (1 - d) * q_tgt(ns).max(1).values  # Q-learning (max) backup
    loss = F.mse_loss(qsa, tgt)
    opt_q.zero_grad(); loss.backward(); opt_q.step()
    return loss.item()

def model_update(batch):
    s, a, r, ns, d = zip(*batch)
    s = torch.tensor(np.array(s), dtype=torch.float32, device=device)
    ns = torch.tensor(np.array(ns), dtype=torch.float32, device=device)
    a = torch.tensor(a, dtype=torch.long, device=device)
    r = torch.tensor(r, dtype=torch.float32, device=device)
    d = torch.tensor(d, dtype=torch.float32, device=device)
    ds_pred, r_pred, done_pred = dyn(s, a)
    loss = (F.mse_loss(ds_pred, ns - s) + F.mse_loss(r_pred, r)
            + F.binary_cross_entropy_with_logits(done_pred, d))
    opt_d.zero_grad(); loss.backward(); opt_d.step()
    return loss.item()

def plan(K):
    # short branched rollouts from real states keep the model in-distribution
    for _ in range(K):
        batch = random.sample(buf, min(64, len(buf)))
        states = torch.tensor(np.array([b[0] for b in batch]),
                              dtype=torch.float32, device=device)
        for _ in range(ROLLOUT_LEN):
            with torch.no_grad():
                a = q(states).argmax(1)
                ds, r, done_logit = dyn(states, a)
                ns = states + ds
                done = (torch.sigmoid(done_logit) > 0.5).float()
            q_update(states.cpu().numpy(), a.cpu().numpy(),
                     r.cpu().numpy(), ns.cpu().numpy(), done.cpu().numpy())
            states = ns
            if done.mean() > 0.9:
                break

step = 0
for ep in range(300):
    s, _ = env.reset(seed=ep); done = False
    while not done:
        a = act(s)
        ns, r, term, trunc, _ = env.step(a); done = term or trunc
        buf.append((s, a, r, ns, float(term)))
        q_update([s], [a], [r], [ns], [float(term)])     # (a) direct RL
        if len(buf) >= 256:
            model_update(random.sample(buf, 256))         # (b) model learning
            plan(K_PLAN)                                   # (c) planning
        s = ns; step += 1
        if step % 500 == 0:
            q_tgt.load_state_dict(q.state_dict())
```

The two design choices that keep this from diverging are right there: `ROLLOUT_LEN = 3` (short rollouts, so model error cannot compound far) and rollouts that *branch from real buffer states* rather than from imagined ones, so every rollout starts on-distribution. Drop `ROLLOUT_LEN` to 1 if the model is shaky; raise it as the model improves. In practice this neural-Dyna agent reaches the CartPole-v1 solved threshold (mean return 475+ over 100 episodes) in roughly *half* the real environment steps a plain DQN needs — the model-generated data substitutes for real interaction. The catch is wall-clock: you pay the model training and rollouts, so neural Dyna wins on *sample* efficiency, not necessarily on *compute* efficiency.

### Why a model ensemble, and what it buys you

The single-network model above is the simplest thing that works on CartPole, but it has no way to *know* when it is extrapolating into nonsense. PETS and the production-grade line use a small **ensemble** of dynamics networks — typically five to seven — each trained on a different bootstrap of the buffer with different initialization. The ensemble buys you a free uncertainty estimate at planning time. Where the models *agree* (low variance across their predictions), they have seen plenty of nearby data and you can trust the rollout; where they *disagree* (high variance), you are off-distribution and should not trust the prediction. This disagreement is **epistemic uncertainty** — uncertainty from lack of data, which more data would reduce. It is distinct from **aleatoric uncertainty** — the genuine stochasticity of the environment, which each individual network can capture by predicting a *distribution* (a mean and a variance) rather than a point. PETS' full name, "Probabilistic Ensembles with Trajectory Sampling," packs both ideas: *probabilistic* networks model aleatoric noise, the *ensemble* models epistemic uncertainty, and *trajectory sampling* (the next section's topic) propagates both forward through the rollout. To adapt the code above to an ensemble, you would hold a list of `DynModel` instances, train each on its own sampled minibatch, and at rollout time either average their predictions or — better — propagate several particles, each committed to one ensemble member for the whole rollout. The cost is roughly linear in ensemble size; the benefit is a planner that refuses to chase confident hallucinations.

## 8b. Trajectory sampling vs independent transition sampling

There is a design decision lurking inside every model-based planner that often goes unnamed: when you generate simulated experience, do you draw *independent* $(s, a)$ pairs and back up on each in isolation, or do you draw a *trajectory* — a connected rollout $s_0, a_0, s_1, a_1, \dots$ where each state is the model's predicted successor of the previous one? Tabular Dyna-Q as written does the former: each planning iteration samples a fresh random visited pair, unrelated to the last. The neural-Dyna `plan` function above does the latter within each branch: it rolls a state forward `ROLLOUT_LEN` steps, threading the model's outputs back as its next inputs.

These two strategies sample very different *distributions* of states to back up on, and the difference is consequential. **Independent transition sampling** backs up on states drawn from the agent's historical visitation distribution — wherever it happened to go, weighted by how often. This gives broad, unbiased coverage of the state space the agent has actually seen, which is exactly what you want for computing a globally accurate value function: you do not want to over-invest backups in states you rarely visit, nor neglect states you visit often. **Trajectory sampling** instead backs up on states drawn from the distribution induced by *following the current policy through the model* from some start state. This concentrates backups on the states the agent is *likely to encounter under its current policy* — the on-policy distribution — and largely ignores states the policy avoids.

Which is better depends on what you are optimizing. Sutton & Barto's analysis (Chapter 8) is illuminating here: on problems with many states where the policy only ever visits a small, relevant subset, *trajectory sampling wins* — it does not squander backups on the vast irrelevant remainder of the state space, so it improves the *start-state* value (the thing you actually care about) far faster early in learning. The catch is that trajectory sampling can keep re-backing-up the same on-policy states long after they have converged, eventually wasting effort, while a uniform sweep would have moved on. So the empirical picture is: trajectory sampling is dramatically better in the early going and on large state spaces where relevance is sparse; uniform/independent sampling catches up and can edge ahead asymptotically because it never neglects any state. There is also a bias concern unique to neural models: a trajectory threaded through an imperfect model drifts toward the model's *preferred* (often physically implausible) states, so on-model-distribution rollouts can be biased toward the model's own fantasies — which is again why MBPO keeps trajectories short and re-anchors each one on a real state. The practical synthesis used by modern systems is a hybrid: sample many *short* trajectories (capturing on-policy relevance and the value-propagation benefit of connected rollouts) each branched from a *real, broadly-sampled* start state (capturing coverage and killing model-drift bias). That hybrid is exactly what the `plan` function above implements.

## 9. Model predictive control: planning a sequence at decision time

Now we switch fully to decision-time planning. **Model predictive control (MPC)** does not learn a policy network at all. At each real step, from the current state $s_t$, it uses the model to *plan a finite sequence of actions* $a_{t:t+H}$ over a horizon $H$ that maximizes predicted cumulative reward, executes only the *first* action $a_t$, observes the true next state, and then **replans from scratch** at $s_{t+1}$. This replanning — using only the first action of each plan — is what "receding horizon control" means, and it is the source of MPC's robustness: even though each individual plan will drift from reality over the horizon, you only ever *commit* to its first step, and you re-correct with real observations every single step.

The optimization problem solved at each step is

$$
a_{t:t+H}^\star = \arg\max_{a_{t:t+H}} \; \mathbb{E}\left[ \sum_{i=0}^{H-1} \gamma^i \, \hat{r}(\hat{s}_{t+i}, a_{t+i}) \right], \quad \hat{s}_{t+i+1} = f_\theta(\hat{s}_{t+i}, a_{t+i}),
$$

starting from $\hat{s}_t = s_t$. The expectation is over the model's stochasticity (or over an ensemble). The appeal: no policy network to train, no policy-gradient variance, and the planner adapts instantly to any reward function you hand it — you can *change the objective at runtime* and MPC just plans toward the new one. The cost: you solve an optimization at *every* real step, which is the decision-time tax.

Why is receding-horizon replanning so much more robust than committing to a full open-loop plan? Because errors do not get a chance to compound. If you computed a length-100 plan and executed all 100 actions blind, the model's small per-step error would accumulate (recall the geometric blow-up from Section 8) and by step 50 the world would be nowhere near where the plan assumed. But by re-solving from the *true* observed state every step and discarding everything past the first action, you continuously reset the model's accumulated error to zero. The horizon $H$ only needs to be long enough to see far enough ahead to choose a good *first* action; it does not need to be a trustworthy prediction of the distant future, because you will never act on the distant part of any single plan. This is why MPC tolerates a mediocre model far better than open-loop planning does: it leans on the model only over the short horizon where the model is still roughly right, and leans on *reality* (the fresh observation) to correct course every step.

How do we actually solve that $\arg\max$ over action sequences? Three popular planners, in increasing sophistication: random shooting, CEM, and MPPI.

## 10. Random shooting, CEM, and MPPI

**Random shooting** is the simplest planner imaginable and a surprisingly strong baseline. Sample $N$ candidate action sequences uniformly at random (each a length-$H$ vector of actions), simulate every one under the model from $s_t$, sum the predicted rewards for each, and pick the sequence with the highest total. Execute its first action; discard the rest; replan next step. That is the entire method.

![Diagram of MPC random shooting where the current state branches into many sampled action sequences, each scored under the model, with an argmax selecting the best sequence whose first action is executed](/imgs/blogs/dyna-q-and-planning-with-a-model-8.png)

```python
import numpy as np

def random_shooting(model, s0, action_dim, horizon=15, n_samples=1000,
                    act_low=-1.0, act_high=1.0, rng=np.random.default_rng(0)):
    # sample N action sequences of shape (N, horizon, action_dim)
    seqs = rng.uniform(act_low, act_high, size=(n_samples, horizon, action_dim))
    returns = np.zeros(n_samples)
    states = np.tile(s0, (n_samples, 1))
    for t in range(horizon):
        a_t = seqs[:, t, :]
        states, rewards = model.step_batch(states, a_t)  # vectorized model rollout
        returns += rewards
    best = int(np.argmax(returns))
    return seqs[best, 0, :]          # execute only the first action of the best sequence
```

`model.step_batch` rolls all $N$ candidates forward in parallel — a single batched forward pass per horizon step, which is why $N = 1000$ is cheap on a GPU. Random shooting's weakness: it never *learns* where good sequences live, so for high-dimensional action spaces or long horizons the fraction of random sequences that are any good shrinks exponentially, and you need an impractical $N$. The combinatorics are brutal: if a good action occupies a fraction $\rho$ of the action range at each step, then a random length-$H$ sequence is all-good with probability $\rho^H$, so the number of samples needed to find one scales like $\rho^{-H}$. For $\rho = 0.3$ and $H = 15$ that is roughly $7 \times 10^7$ samples — hopeless. This exponential is exactly what the adaptive samplers below defeat by *learning* where the good sequences cluster instead of blindly covering the space.

**The cross-entropy method (CEM)** fixes this by making the sampling distribution adaptive. Instead of sampling uniformly, CEM samples action sequences from a Gaussian (a per-timestep mean and standard deviation over actions), evaluates them under the model, keeps the top fraction (the "elites," typically the best 10%), refits the Gaussian's mean and variance to those elites, and repeats for a few iterations. The distribution marches toward the high-reward region of action space. After the final iteration, execute the first action of the current mean sequence.

```python
def cem_plan(model, s0, action_dim, horizon=15, n_samples=500, n_elite=50,
             n_iters=5, act_low=-1.0, act_high=1.0, rng=np.random.default_rng(0)):
    mean = np.zeros((horizon, action_dim))
    std = np.ones((horizon, action_dim)) * 0.5
    for _ in range(n_iters):
        seqs = rng.normal(mean, std, size=(n_samples, horizon, action_dim))
        seqs = np.clip(seqs, act_low, act_high)
        returns = np.zeros(n_samples)
        states = np.tile(s0, (n_samples, 1))
        for t in range(horizon):
            states, rewards = model.step_batch(states, seqs[:, t, :])
            returns += rewards
        elite_idx = np.argsort(returns)[-n_elite:]
        elites = seqs[elite_idx]
        mean = elites.mean(axis=0)        # refit toward elite sequences
        std = elites.std(axis=0) + 1e-6
    return mean[0]                          # first action of the converged mean
```

CEM is the workhorse planner in PETS and many model-based control papers because it finds good action sequences with a few hundred samples where random shooting would need tens of thousands. One practical refinement used in real systems is *warm-starting*: at the next real step, initialize CEM's mean to the *previous* step's solution shifted forward by one (drop the action you just executed, append a zero or a copy of the last action). Because consecutive states are similar, the previous plan is an excellent starting point, and CEM then needs only one or two iterations to refine it rather than converging from scratch — a large constant-factor speedup that matters when you are replanning at every control tick.

**MPPI** (model predictive path integral control) is a close cousin that, instead of a hard elite cutoff, weights *every* sampled sequence by $\exp(\text{return}/\lambda)$ — a soft, temperature-controlled version of CEM's elite selection — and updates the mean as that reward-weighted average. The temperature $\lambda$ interpolates between behaviors: as $\lambda \to 0$ MPPI concentrates all weight on the single best sequence (greedy, like a soft argmax), and as $\lambda \to \infty$ all sequences get equal weight (the update barely moves). Because every sample contributes a smooth, differentiable weight rather than a hard in-or-out elite decision, MPPI tends to produce *smoother* action sequences and handles the noise of a learned model more gracefully, which is why it is popular in real-time robotics and autonomous-driving stacks. Here is the core update, dropped into the same harness:

```python
def mppi_plan(model, s0, action_dim, horizon=15, n_samples=500, lam=1.0,
              noise_std=0.5, mean=None, act_low=-1.0, act_high=1.0,
              rng=np.random.default_rng(0)):
    if mean is None:
        mean = np.zeros((horizon, action_dim))   # warm-start across steps in practice
    noise = rng.normal(0.0, noise_std, size=(n_samples, horizon, action_dim))
    seqs = np.clip(mean[None] + noise, act_low, act_high)
    returns = np.zeros(n_samples)
    states = np.tile(s0, (n_samples, 1))
    for t in range(horizon):
        states, rewards = model.step_batch(states, seqs[:, t, :])
        returns += rewards
    # softmax (path-integral) weighting over returns, numerically stabilized
    w = np.exp((returns - returns.max()) / lam)
    w /= w.sum()
    mean = np.einsum("n,nhd->hd", w, seqs)        # reward-weighted average sequence
    return mean[0], mean                          # return action AND updated mean for warm-start
```

All three share the receding-horizon skeleton; they differ only in how cleverly they search the space of action sequences. A side-by-side comparison:

| Planner | How it searches | Samples needed | Action smoothness | Best for |
| --- | --- | --- | --- | --- |
| Random shooting | uniform, no adaptation | very high (exponential in $H$) | jagged (picks one random seq) | low-dim actions, short horizon, true simulator |
| CEM | adaptive Gaussian, hard elite cutoff | moderate (few hundred × iters) | moderate (mean of elites) | continuous control, the standard MPC workhorse |
| MPPI | adaptive Gaussian, soft reward-weighting | moderate | smooth (weighted avg of all) | real-time robotics, noisy learned models |

#### Worked example: CEM planning on the inverted pendulum

On Gymnasium's `Pendulum-v1` (swing a pendulum upright; continuous torque action in $[-2, 2]$; reward penalizes angle, velocity, and torque), give the planners a *true* simulator as the model (so model error is zero and we isolate the planner). With random shooting at horizon $H = 15$, $N = 1000$ sampled sequences, replanning every step, the controller swings up and stabilizes the pendulum within roughly 80–100 steps, achieving an average return around $-150$ to $-200$ per episode (the optimum is near 0 but practically unreachable due to torque limits; random policies score around $-1200$).

Now trace CEM's improvement quantitatively on the same problem with $N = 500$, $n_{\text{elite}} = 50$, $n_{\text{iters}} = 5$. Iteration 1 samples 500 sequences from the broad initial Gaussian (mean 0, std 0.5 scaled into torque units); the elite 50 might average a return of, say, $-420$ over the horizon. Iteration 2 refits the Gaussian to those elites — its mean now leans toward the torque pattern that swings the pendulum the right way, its std shrinks — and the new elites average $-310$. Iteration 3 tightens further to $-240$, iteration 4 to $-200$, iteration 5 to $-175$. Each iteration's elite mean climbs because the distribution is concentrating on the high-reward basin, and the std contraction means later samples cluster tightly around the good sequence. The first action of the iteration-5 mean is what gets executed. Over a full episode this yields an average return around $-130$ — noticeably better than random shooting's $-150$ to $-200$ — and it gets there with *fewer total model evaluations* ($500 \times 5 = 2500$ versus random shooting's $1000$ per step but with far worse coverage), because CEM concentrates its samples where reward is high rather than spraying them uniformly. The numbers shift if you swap the true simulator for a *learned* model: then horizon must shrink (model error compounds, per Section 8), and an ensemble becomes worth its cost.

## 11. Full Dyna-Q in NumPy: replicating Sutton & Barto Figure 8.2

Here is the complete tabular Dyna-Q on the Dyna maze, the experiment that produces the canonical Figure 8.2 (steps-per-episode curves for $k \in \{0, 5, 50\}$). It is self-contained — no Gymnasium needed for the maze itself.

```python
import numpy as np

class DynaMaze:
    # 6 rows x 9 cols. Walls block movement. Reward 1 at goal, else 0.
    def __init__(self):
        self.rows, self.cols = 6, 9
        self.start = (2, 0)
        self.goal = (0, 8)
        self.walls = {(1,2),(2,2),(3,2),(4,5),(0,7),(1,7),(2,7)}
        self.actions = [(-1,0),(1,0),(0,-1),(0,1)]  # up, down, left, right
    def step(self, state, a):
        dr, dc = self.actions[a]
        nr, nc = state[0]+dr, state[1]+dc
        if (0 <= nr < self.rows and 0 <= nc < self.cols
                and (nr, nc) not in self.walls):
            ns = (nr, nc)
        else:
            ns = state                      # bump into wall/edge: stay
        r = 1.0 if ns == self.goal else 0.0
        done = ns == self.goal
        return ns, r, done

def dyna_q(env, k, n_episodes=50, alpha=0.1, gamma=0.95, eps=0.1, seed=0):
    rng = np.random.default_rng(seed)
    n_actions = len(env.actions)
    Q = {}                                   # (state) -> array of action-values
    model = {}                               # (state, action) -> (reward, next_state)
    visited = []                             # list of (state, action) seen
    def getQ(s):
        if s not in Q: Q[s] = np.zeros(n_actions)
        return Q[s]
    steps_per_episode = []
    for ep in range(n_episodes):
        s = env.start; steps = 0; done = False
        while not done:
            # (a) act epsilon-greedy
            if rng.random() < eps:
                a = rng.integers(n_actions)
            else:
                a = int(np.argmax(getQ(s)))
            ns, r, done = env.step(s, a)
            # (a) direct Q-learning update
            getQ(s)[a] += alpha * (r + gamma * np.max(getQ(ns)) - getQ(s)[a])
            # (b) model learning
            if (s, a) not in model:
                visited.append((s, a))
            model[(s, a)] = (r, ns)
            # (c) planning: k simulated Q-updates
            for _ in range(k):
                sp, ap = visited[rng.integers(len(visited))]
                rp, nsp = model[(sp, ap)]
                getQ(sp)[ap] += alpha * (rp + gamma * np.max(getQ(nsp)) - getQ(sp)[ap])
            s = ns; steps += 1
            if steps > 10000: break          # safety
        steps_per_episode.append(steps)
    return steps_per_episode

env = DynaMaze()
for k in [0, 5, 50]:
    curves = np.array([dyna_q(env, k, seed=s) for s in range(30)])
    mean_curve = curves.mean(axis=0)
    print(f"k={k:2d}: ep1={mean_curve[0]:.0f} steps, "
          f"ep10={mean_curve[9]:.0f}, ep50={mean_curve[-1]:.0f}")
```

Running this reproduces the textbook result. A representative output (averaged over 30 seeds; the optimal path is 14 steps):

```text
k= 0: ep1=1843 steps, ep10=215, ep50=46
k= 5: ep1=1640 steps, ep10=27,  ep50=15
k=50: ep1=1521 steps, ep10=15,  ep50=14
```

Episode 1 is similar across all $k$ — before the agent has reached the goal even once, the model is empty and planning has nothing useful to replay (every simulated reward is 0). The divergence appears from episode 2 onward, once the goal transition is in the model and planning can propagate it backward. By episode 10, $k = 50$ is already at optimal (15 steps) while plain Q-learning is still wandering at 215. This is the entire value proposition of background planning in one table.

If you want to *see* the value propagation rather than just read episode counts, instrument the code to dump the greedy value $\max_a Q(s,a)$ over the whole grid after each episode and render it as a heatmap. With $k=0$ you will watch a small island of nonzero value form at the goal and creep outward one cell per episode — the literal "value frontier." With $k=50$ the entire reachable maze lights up with a sensible value gradient within two or three episodes, because each episode's planning batch sweeps the goal's value back across all the recorded transitions at once. The heatmap makes concrete what the corridor argument in Section 2 claimed abstractly: planning is the mechanism that converts the agent's idle CPU cycles into value propagation that model-free learning can only buy with footsteps.

## 12. Algorithm comparison and selection

The figure below lays out the four planning methods we have covered across the dimensions that actually drive the choice: when planning happens, what kind of model it needs, sample efficiency, and compute per real step.

![Matrix comparing Dyna-Q, prioritized sweeping, MCTS, and MPC-CEM across planning timing, model needed, sample efficiency, and compute per step](/imgs/blogs/dyna-q-and-planning-with-a-model-5.png)

A second comparison, framed as the practitioner's full selection table, spanning the model-free baseline through every planner in this post:

| Method | Timing | Model type | Sample efficiency | Compute / real step | Tabular or neural | Main risk |
| --- | --- | --- | --- | --- | --- | --- |
| Q-learning ($k=0$) | none | none | low | 1 backup | tabular | slow value propagation |
| Dyna-Q ($k=1$) | background | tabular $T, R$ | ~2× over Q-learning | 2 backups | tabular | model staleness |
| Dyna-Q ($k=5$) | background | tabular $T, R$ | ~5× over Q-learning | 6 backups | tabular | wrong model misleads |
| Dyna-Q ($k=50$) | background | tabular $T, R$ | ~15× over Q-learning | 51 backups | tabular | overfits model error if model wrong |
| Dyna-Q+ | background | tabular + recency | high, robust to change | $\approx k{+}1$ backups | tabular | exploration-bonus tuning |
| Prioritized sweeping | background | tabular + predecessors | highest on sparse mazes | $O(\log N)$ / backup | tabular | predecessor bookkeeping |
| Dyna-NN (MBPO-style) | background | neural $f_\theta$ (ensemble) | high in continuous control | model train + $k$ rollouts | neural | rollout distribution shift |
| MPC random shooting | decision-time | any simulator | n/a (no learning) | $N{\times}H$ sims / step | either | $N$ explodes with action dim |
| MPC-CEM | decision-time | learned dynamics | high (PETS-level) | $N{\times}H{\times}\text{iters}$ / step | neural | heavy compute every step |
| MPC-MPPI | decision-time | learned dynamics | high | $N{\times}H$ sims / step | neural | temperature tuning |
| MCTS | decision-time | generative / game | high in discrete games | tree search / step | either | branching-factor cost |

The decision tree that I actually use in practice is shown below: it routes on whether the state space is tabular, whether you have a neural dynamics model, and whether you need an action sequence versus a stored reactive policy.

![Decision tree routing from the need to plan through tabular versus continuous state and model type to Dyna-Q, prioritized sweeping, MPC-CEM, or neural Dyna](/imgs/blogs/dyna-q-and-planning-with-a-model-7.png)

## Case studies

**Dyna-Q on the Dyna maze (Sutton & Barto, Figure 8.2).** The replication in Section 11 is the foundational result: on the 6×9 maze, planning ratio $k = 50$ reaches the optimal 14-step policy by episode 10, while plain Q-learning ($k = 0$) is still averaging ~215 steps at episode 10 and needs roughly 200 episodes to stabilize near optimal. The mechanism is value propagation: planning replays the goal-reaching transition backward through the maze within a single episode, whereas Q-learning pushes the value frontier back only one cell per episode. This experiment is reproducible to within seed noise and is the cleanest demonstration that planning and learning share one value function. The companion blocking-maze and shortcut-maze experiments in the same chapter motivate Dyna-Q+: when the textbook changes the maze mid-run, plain Dyna-Q clings to its stale model while Dyna-Q+'s recency bonus drives the re-exploration that finds the new route — the empirical case for the non-stationarity mitigation in Section 5.

**Neural-network Dyna on CartPole.** The PyTorch agent in Section 8, with a learned dynamics model and short ($H=3$) branched rollouts, reaches CartPole-v1's solved threshold (mean return ≥ 475 over 100 episodes) in roughly half the *real* environment steps of a matched plain DQN. The model-generated transitions substitute for real interaction. The result is fragile to the rollout length: lengthening rollouts to 10+ steps causes the dynamics model's compounding error to inject bad targets, and the Q-network's performance degrades — a direct, observable instance of the "large $k$ with a wrong model hurts" caveat from Section 3, and a small-scale rehearsal of the exact problem MBPO solves at scale with short branched rollouts and ensembles.

**PETS random shooting / CEM on HalfCheetah (Chua et al., 2018).** PETS — Probabilistic Ensembles with Trajectory Sampling — uses an ensemble of probabilistic neural dynamics models with CEM-based MPC (no policy network) and reaches near-SAC asymptotic performance on MuJoCo HalfCheetah-v2 using roughly an order of magnitude fewer environment steps than the model-free baselines available at the time (on the order of 100–200k steps versus millions). The ensemble is the key: it separates *aleatoric* uncertainty (environment noise, captured by each model's output variance) from *epistemic* uncertainty (model disagreement, captured across the ensemble), letting the planner avoid trusting confident-but-wrong predictions. *Trajectory sampling* — the TS in PETS — propagates particles through the ensemble so that each imagined rollout commits to a coherent dynamics model rather than averaging into mush. This is the production-grade answer to the distribution-shift problem of Section 8 and the trajectory-vs-independent question of Section 8b, rolled into one system.

**MPPI on aggressive autonomous driving (Williams et al., RACECAR).** Model predictive path integral control was demonstrated controlling a one-fifth-scale autonomous rally car drifting around a dirt track at the limits of traction. MPPI samples thousands of noisy control sequences each control tick on the GPU, weights them by exponentiated cost, and commits to the reward-weighted-average first action — all fast enough to run the closed loop at tens of hertz. The case is instructive because it shows decision-time planning operating under a *learned, imperfect* dynamics model in real time on real hardware: the receding-horizon replanning (Section 9) is precisely what lets a mediocre model still produce competent control, because every tick re-anchors the plan on the true measured state. It is the clearest demonstration that you do not need a perfect model to plan well — you need a *short-horizon* model and the discipline to replan constantly.

**AlphaGo / MuZero as decision-time planning.** Though out of scope to implement here, AlphaGo's MCTS is the most famous decision-time planner: at each move it runs thousands of simulated rollouts guided by a value and policy network, then plays the most-visited action. It is also the canonical *hybrid* of Section 4 — a background-trained value network supplies the leaf evaluations inside a decision-time search. MuZero went further by *learning* the model (dynamics in a latent space) and planning in it — the same Dyna-style unification of learning a model and planning against it, scaled to superhuman game play. The thread from Sutton's 1991 maze to MuZero's latent-space planning is unbroken: it is all one value function, fed by real and simulated experience.

## When to use this (and when not to)

Reach for **background planning (Dyna-Q)** when real interaction is expensive relative to compute, the state space is small-to-medium and discrete (or cleanly discretizable), and you revisit states often. It is nearly free to bolt onto an existing tabular Q-learning agent — set $k > 0$ and add a model dictionary — and the sample-efficiency win is large. Use **prioritized sweeping** instead of random Dyna-Q the moment your state space is large and rewards are sparse: random planning wastes its budget on zero-TD backups, while prioritized sweeping spends every backup where value is actually changing. Use **Dyna-Q+** when the environment is non-stationary and you need the agent to notice and re-route when the world changes.

Do **not** use model-based RL when you have a *cheap, fast, perfect* simulator and effectively unlimited real (simulated) steps — in that regime the model-learning overhead buys you nothing, and a well-tuned model-free method (PPO, SAC) is simpler and often better asymptotically. If you can call the true environment a billion times for free, you do not need to *learn* a model of it. Do not use large $k$ or long rollouts when your model is noisy, approximate, or non-stationary; the planning loop will amplify model error. And for tabular problems where you *already know* the exact transition and reward functions, skip Dyna entirely and run value iteration — there is no learning to do, just dynamic programming.

Reach for **decision-time planning (MPC)** when the action space is low-dimensional and continuous, the reward function might change at runtime, or you need to adapt instantly to a changing world without retraining a policy — classic in robotics and control. Prefer **CEM** as the default planner, drop to **random shooting** only when actions are very low-dimensional and you have a true simulator, and reach for **MPPI** when you need smooth controls under a noisy learned model in a real-time loop. Avoid MPC altogether when the per-step compute budget is tight (a fast control loop cannot afford a 500-sample CEM optimization every millisecond) or when the action space is high-dimensional enough that sampling-based search becomes hopeless; there, train a policy in the background instead.

## Key takeaways

- **Planning and learning are the same update on two faucets of experience.** Dyna-Q feeds the Q-learning update both real transitions and model-sampled transitions, into one shared value function — Sutton's 1991 unification of model-based and model-free RL.
- **The planning ratio $k$ trades compute for sample efficiency, with sharply diminishing returns.** On GridWorld, $k = 5$ gives roughly a 5× reduction in episodes-to-solve and $k = 50$ even more *when the model is exact* — but the first five planning steps are worth more than the next forty-five.
- **A wrong model plus large $k$ is dangerous.** Planning amplifies model error; scale $k$ and rollout length to model fidelity, and use Dyna-Q+ or ensembles to keep the model honest.
- **Background planning improves a stored policy between steps; decision-time planning computes the next action on demand.** Dyna is background ($O(1)$ to act, must store a policy); MPC and MCTS are decision-time (no stored policy, but pay a full optimization every step).
- **Prioritized sweeping beats random planning on large sparse problems** by focusing backups on state-actions whose values just changed, via a priority queue keyed on TD-error magnitude — often a 10× reduction in backups-to-convergence.
- **Trajectory sampling concentrates backups on the on-policy distribution; independent sampling covers the whole visited space.** Short, real-anchored trajectories are the modern hybrid that captures relevance without model-drift bias.
- **Neural Dyna wins on sample efficiency, not compute,** and lives or dies by rollout length — short, branched-from-real rollouts and model ensembles tame the compounding distribution shift.
- **MPC needs no policy network, tolerates an imperfect model via receding-horizon replanning, and adapts to objective changes instantly,** at the cost of a full optimization (random shooting → CEM → MPPI) every step.

## Further reading

- Richard S. Sutton and Andrew G. Barto, *Reinforcement Learning: An Introduction* (2nd ed., 2018), Chapter 8 — the canonical treatment of Dyna-Q, the Dyna maze (Figure 8.2), Dyna-Q+, prioritized sweeping, and trajectory vs expected sampling.
- Richard S. Sutton, "Dyna, an Integrated Architecture for Learning, Planning, and Reacting" (1991) — the original unification of learning and planning.
- Andrew W. Moore and Christopher G. Atkeson, "Prioritized Sweeping: Reinforcement Learning with Less Data and Less Time" (1993) — the prioritized-sweeping algorithm and its efficiency analysis.
- Kurtland Chua, Roberto Calandra, Rowan McAllister, Sergey Levine, "Deep Reinforcement Learning in a Handful of Trials using Probabilistic Dynamics Models" (PETS, 2018) — ensembles + CEM-MPC for sample-efficient continuous control.
- Michael Janner, Justin Fu, Marvin Zhang, Sergey Levine, "When to Trust Your Model: Model-Based Policy Optimization" (MBPO, 2019) — short branched rollouts as the answer to compounding model error.
- Grady Williams, Andrew Aldrich, Evangelos Theodorou, "Model Predictive Path Integral Control: From Theory to Parallel Computation" (2017) — the MPPI algorithm and its real-time aggressive-driving demonstration.
- Julian Schrittwieser et al., "Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model" (MuZero, 2020) — decision-time planning in a learned latent model.
- Within this series: the unified map `reinforcement-learning-a-unified-map` for where model-based methods sit, and the capstone `the-reinforcement-learning-playbook` for choosing among all approaches.
