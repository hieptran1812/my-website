---
title: "MuZero: Mastering Games Without Knowing the Rules"
date: "2026-06-22"
publishDate: "2026-06-22"
description: "How MuZero plans with a learned model it builds from scratch — three networks, a latent-space tree search, and superhuman play on Go, chess, shogi, and 57 Atari games without ever being told the rules."
tags:
  [
    "reinforcement-learning",
    "deep-learning",
    "model-based-rl",
    "monte-carlo-tree-search",
    "planning",
    "machine-learning",
    "pytorch",
    "alphazero",
  ]
category: "machine-learning"
subcategory: "Reinforcement Learning"
author: "Hiep Tran"
featured: true
readTime: 55
image: "/imgs/blogs/muzero-mastering-games-without-rules-1.png"
---

Hand AlphaZero a chessboard and it is unstoppable. Hand it an Atari screen and it is helpless — not because it cannot see the pixels, but because its entire method depends on something it does not have: a perfect simulator of what happens next. AlphaZero searches by *playing out* moves. To search "what if I push this pawn," it asks a hard-coded rules engine to compute the resulting board. On a chessboard that engine exists and is exact. On an Atari frame, "what happens if I press LEFT" has no clean answer you can write down — the dynamics live inside the game ROM, and the agent only ever sees pixels coming back.

MuZero (Schrittwieser et al., 2020, "Mastering Atari, Go, chess and shogi by planning with a learned model") removes that crutch. It plans exactly the way AlphaZero does — Monte Carlo Tree Search guided by a neural network — but it *learns its own model of the environment's dynamics* and runs the entire search inside that learned model's latent space. It is never told the rules. It infers, from reward and value signals alone, just enough of a model to plan well. The result: it matched or beat AlphaZero on Go, chess, and shogi while *also* reaching a mean human-normalized score of 731% across 57 Atari games — a domain AlphaZero cannot touch.

The figure below is the whole idea in one frame: three learned functions — a representation network, a dynamics network, and a prediction network — standing in for the simulator AlphaZero took for granted.

![Diagram showing MuZero observation passing through the representation network into a latent state, the dynamics network mapping a latent state and action to a next latent state and reward, and the prediction network mapping a latent state to a policy and value](/imgs/blogs/muzero-mastering-games-without-rules-1.png)

By the end of this post you will understand each of those three functions and why none of them needs the true rules; how MCTS runs inside a latent space the agent invented; the exact training objective that ties policy, value, and reward together; the Reanalyze trick that makes MuZero sample-efficient; and why MuZero's learned latent is fundamentally different from a world model that reconstructs pixels. You will see runnable PyTorch sketches of all three networks, a full self-play-plus-training loop in pseudocode, several worked examples with concrete numbers, and the case studies — Atari, board games, EfficientZero, video compression — that pinned down what the method can do. This sits in the model-based corner of the [unified RL map](/blog/machine-learning/reinforcement-learning/reinforcement-learning-a-unified-map): every RL algorithm is a different answer to *which objective to optimize* and *how to estimate the gradient*, and MuZero's answer is "optimize a planning policy, estimate gradients through a model you learned yourself."

## 1. The AlphaGo line, and the wall it hit

To see what MuZero changes, walk the lineage it grew out of. The thread running through all of it is the same: combine a slow, deliberate *search* with a fast, learned *intuition*, and let each improve the other. Each step in the lineage removes one dependency the previous step took for granted, and the whole arc is best read as a steady stripping-away of human input, then of the rulebook itself.

**AlphaGo (2016)** beat Lee Sedol with a stack of carefully engineered pieces. It used a *policy network* trained first by supervised learning on roughly thirty million human expert moves from online Go servers, then refined by self-play reinforcement learning. It used a separate *value network* estimating the probability of winning from a given board. And it stitched these together with Monte Carlo Tree Search that, at the leaves, performed fast rollouts using yet another small, hand-built rollout policy to estimate outcomes. Three things stand out about this design. First, it *needed human data* — the supervised bootstrap on expert games was load-bearing, and without it the policy network started from noise. Second, it used several specialized networks rather than one. Third, and most importantly for our story, AlphaGo searched over *real board states*: the rules of Go were hard-coded into the tree, so expanding a node meant applying the actual game logic to place a stone and check for captures, ko, and territory.

**AlphaZero (2017)** stripped that down to its essence. One network with two heads — a policy head $p$ and a value head $v$ — trained purely by self-play from random initialization, with no human games at all and no separate rollout policy. The training loop is a beautiful feedback cycle. MCTS uses the network to guide search; the search produces a stronger move distribution than the raw policy head alone (because lookahead corrects the network's mistakes); that stronger distribution becomes the training target for the policy head; the improved network then makes the *next* search even stronger. This is *policy improvement by search*, and it is the engine under everything that follows. AlphaZero removed the human-data dependency entirely — it learned chess, shogi, and Go from scratch, surpassing all prior programs, including AlphaGo, within hours of self-play on each game.

But AlphaZero kept one hard dependency baked into its search. At every node of the tree, to expand a child it must answer the question "if I take action $a$ from this state, what state do I land in, and is the game over, and did anyone win?" It answers by calling a **perfect simulator** — the actual game rules. For Go, chess, and shogi that simulator is a few hundred lines of exact, deterministic code: given the board and a move, it returns the next board with certainty. The MCTS in AlphaZero is therefore search over *real* future states, each one computed exactly by the rules engine.

That is the wall. The world is full of decision problems — playing Atari from pixels, controlling a datacenter's cooling, picking a video bitrate, scheduling chips on a wafer — where you do not have a perfect simulator that you can call hundreds of times per decision. You have observations coming back and a reward signal, and the true transition dynamics are either unknown, stochastic, or far too expensive to query inside a search loop running hundreds of iterations per move. AlphaZero simply cannot run there: there is no rules engine to expand its tree.

The naive fix is "learn a simulator that predicts the next observation, then plan in it." People tried; it works poorly for planning, for a deep reason. Predicting the next *full observation* — every pixel of the next Atari frame — is a brutally hard generative problem, and crucially it spends the model's capacity on details that have nothing to do with making good decisions. A learned pixel-predictor that gets the score counter wrong but the background clouds right is useless for planning, even though its reconstruction loss looks excellent. The objective is misaligned: low reconstruction error does not imply good decisions, and the easiest pixels to predict (large, slow-moving backgrounds) are often the least decision-relevant, while the hardest (small, fast game objects) are exactly what matters.

MuZero's insight is to ask for far less. It does not learn to predict observations at all. It learns to predict only the three quantities that MCTS actually consumes: the **reward**, the **policy**, and the **value**. Everything else about the environment — including what the next frame literally looks like — it is free to ignore. That single design decision is why MuZero generalizes past board games, and it is the thread we will pull on for the rest of this post.

## 2. The three functions MuZero learns

MuZero is defined by three parameterized functions, all neural networks trained jointly end-to-end. Using the paper's notation, superscripts index *steps inside an imagined rollout* (the hypothetical steps MCTS takes), while subscripts index *real time* in the actual episode. Keeping those two clocks separate is the single most important thing to get straight: a quantity like $s^3$ is "the latent three imagined steps deep from the current root," whereas $o_t$ is "the real observation at real timestep $t$."

**The representation function** $h_\theta$. It maps the agent's recent real observations to an initial latent (hidden) state:

$$ s^0 = h_\theta(o_1, o_2, \ldots, o_t) $$

For Atari this is a convolutional stack over the last 32 frames plus the actions that produced them (the action history matters because a single frame does not reveal velocity); for board games it is a residual conv tower over the board planes. The output $s^0$ is an abstract vector (or feature grid). It is the *root* of the search tree. There is no constraint forcing $s^0$ to mean anything human-interpretable — it only has to be a sufficient statistic for predicting future rewards, values, and policies. Crucially, $h_\theta$ is *not* trained for reconstruction. No loss term ever asks $s^0$ to be decodable back into the observation it came from. Its only job is to be a good launching point for the imagined rollouts that follow.

**The dynamics function** $g_\theta$. Given a latent state and an action, it predicts the next latent state and the immediate reward:

$$ s^{k}, \; r^{k} = g_\theta(s^{k-1}, a^{k}) $$

This is MuZero's learned simulator, and it is the heart of what makes the method model-based. It never touches an observation. It maps latent-to-latent — a hallucinated transition. Starting from $s^0$ and a sequence of hypothetical actions $a^1, a^2, \ldots$, you can roll forward entirely in latent space, generating a chain $s^0 \to s^1 \to s^2 \to \cdots$ with a predicted reward at every step — exactly what MCTS needs to expand nodes without ever calling the real environment. Think of $g_\theta$ as answering the counterfactual "if I were in this latent state and took this action, where would I end up and what would I earn," using nothing but learned weights. The transition is "hallucinated" in the precise sense that it never corresponds to any rendered observation; the model dreams forward in its own compressed coordinates.

**The prediction function** $f_\theta$. Given a latent state it outputs a policy (action prior) and a value:

$$ \mathbf{p}^{k}, \; v^{k} = f_\theta(s^{k}) $$

This is structurally the AlphaZero head, just applied to a *learned* state instead of a real one. The policy $\mathbf{p}^k$ is a prior over actions that guides which child the search should explore first; the value $v^k$ estimates the expected discounted return from $s^k$ and gives MCTS something to back up without rolling all the way to the end of the game. Without a value estimate, search would have to play to a terminal state to evaluate any leaf — fine for short games, catastrophic for long ones. The value head is what makes truncated, bootstrapped search possible.

Put together: $h$ turns the present into a starting latent, $g$ rolls that latent forward under hypothetical actions while predicting rewards, and $f$ reads off a policy and value at any latent node. Notice what is *absent* — nothing predicts the next observation, nothing reconstructs pixels, and nothing was told what a legal move is. The networks discover whatever internal representation makes their reward, value, and policy predictions accurate, and that representation is theirs to shape however the loss prefers.

One more structural point worth internalizing now, because it recurs throughout: $g_\theta$ is the *only* function applied repeatedly during a rollout. $h_\theta$ runs exactly once per search (to build the root from real observations), and $f_\theta$ runs once per node (to score it). But $g_\theta$ runs at every edge of the imagined tree, so its errors *compound* — a small prediction error at step 1 feeds into step 2, which feeds into step 3, and so on. Much of the engineering in MuZero and its successors is about keeping this compounding under control, and it is why EfficientZero's extra signal on the dynamics net (Section 9) pays off so handsomely.

#### Worked example: tracing one root expansion

Suppose we are playing Atari Breakout. The agent has just seen 32 frames; the paddle is mid-screen, two rows of bricks remain. We run $s^0 = h_\theta(\text{frames})$ to get a 6×6×256 latent feature grid — call it the root. We call $f_\theta(s^0)$ and get a policy $\mathbf{p}^0 = [\text{NOOP}: 0.10, \text{LEFT}: 0.55, \text{RIGHT}: 0.30, \text{FIRE}: 0.05]$ and a value $v^0 = 3.8$ (the net expects roughly 3.8 more points of discounted reward from here). To imagine pressing LEFT, we call $g_\theta(s^0, \text{LEFT})$ and receive $s^1$ (another 6×6×256 grid) and a predicted reward $r^1 = 0.0$ (no brick broken yet). We never rendered a frame. We can now call $f_\theta(s^1)$ to get the policy and value at this imagined state — say $\mathbf{p}^1 = [\text{NOOP}: 0.08, \text{LEFT}: 0.40, \text{RIGHT}: 0.47, \text{FIRE}: 0.05]$ and $v^1 = 3.9$ — and keep going, applying $g_\theta(s^1, \text{RIGHT})$ to reach $s^2$, and so on. That chain $s^0 \xrightarrow{\text{LEFT}} s^1 \xrightarrow{\text{RIGHT}} s^2 \to \cdots$, with a reward and a value at each node, is the entire mechanical vocabulary MuZero needs. The search is going to call $h$ once, then weave together hundreds of $g$ and $f$ calls into a tree.

## 3. MCTS inside a learned model

The search is Monte Carlo Tree Search, the same family AlphaZero uses, but every "what happens next" is answered by $g_\theta$ instead of the rules. A simulation has four phases — selection, expansion, evaluation, and backup — and we run many simulations (50 per move for Atari, 800 for board games) before committing to a single real action. The tree is built incrementally: it starts as just the root, and each simulation adds one new node.

![Stack diagram of one MuZero simulation: select a child by PUCT, expand it with the dynamics network, evaluate with the prediction network, back up the value, repeat many times, then read the improved policy from visit counts](/imgs/blogs/muzero-mastering-games-without-rules-2.png)

**Selection.** Starting at the root, descend the existing tree by repeatedly choosing the action that maximizes the PUCT score (Predictor + Upper Confidence bounds applied to Trees) — a sum of an exploitation term (the action's current mean value $Q$) and an exploration bonus that prefers actions with high prior probability and low visit count:

$$ a^k = \arg\max_a \left[ Q(s, a) + P(s, a) \cdot \frac{\sqrt{\sum_b N(s, b)}}{1 + N(s, a)} \cdot \left( c_1 + \log\frac{\sum_b N(s,b) + c_2 + 1}{c_2} \right) \right] $$

Here $N(s,a)$ is the visit count of the edge, $P(s,a)$ is the prior from $f_\theta$, $Q(s,a)$ is the running mean of backed-up values through that edge, and $c_1, c_2$ are constants (the paper uses $c_1 = 1.25$, $c_2 = 19652$; the logarithmic $c_2$ term lets the exploration constant grow very slowly as the tree gets large). The structure of this formula encodes a precise trade-off. Early on, when $N(s,a) = 0$ for an action, its $Q$ is undefined and the bonus is large, so the *prior* $P(s,a)$ dominates — search trusts the network's intuition. As an action accumulates visits, the $\sqrt{\sum_b N}/(1+N(s,a))$ factor shrinks for that action and its $Q$ becomes well-estimated, so the *value* takes over. This is the knob that balances exploration against exploitation inside the tree, and it is why a good prior $P$ makes search dramatically more efficient: it focuses the budget on plausible moves instead of spreading it uniformly.

**Expansion.** When selection reaches a leaf — a state-action edge that has not yet been expanded into a node — we apply the dynamics network to its parent latent and the chosen action: $s^{k}, r^{k} = g_\theta(s^{k-1}, a^{k})$. This is where the learned model does its work, fabricating the next latent and its reward.

**Evaluation.** We then evaluate the new latent with the prediction network: $\mathbf{p}^{k}, v^{k} = f_\theta(s^{k})$. The new node stores its predicted reward $r^k$ (on the edge into it), its value $v^k$, and the policy priors $\mathbf{p}^k$ over its own children (so they are ready to guide future selection). Note there is no rollout-to-termination here, unlike classical MCTS — the value head *replaces* the random rollout that vanilla MCTS would use to estimate a leaf, exactly as in AlphaZero.

**Backup.** We propagate the value back up the path we descended. Because MuZero predicts a reward at *every* edge (not just at terminal states the way board-game AlphaZero does), the backup must accumulate those intermediate rewards. For a path from the root to the new leaf, the bootstrapped return used in the backup, working from the leaf's value $v$ and folding in each edge reward as we climb, accumulates as

$$ G_{\text{at node }i} = r_{i+1} + \gamma \, G_{\text{at node }i+1}, \qquad G_{\text{at leaf}} = v^{\text{leaf}}. $$

Equivalently, for a path of length $\ell$ the value backed up to the root edge is $\sum_{\tau=0}^{\ell-1} \gamma^{\tau} r^{1+\tau} + \gamma^{\ell} v^{\ell}$. Each node $i$ on the path then updates its statistics: $Q(s,a) \leftarrow \frac{N(s,a) \cdot Q(s,a) + G}{N(s,a) + 1}$ and $N(s,a) \leftarrow N(s,a) + 1$. Because rewards and values can be on wildly different scales across Atari games (Pong tops out near 21, while Q*bert scores in the tens of thousands), MuZero normalizes $Q$ values into $[0,1]$ using the running min and max seen so far in *this* tree before plugging them into PUCT — a small but essential detail for a single hyperparameter set to work across all 57 games. Without it, the exploration bonus and exploitation term would be on incomparable scales in high-scoring games, and search would degenerate.

**Repeat, then act.** After running all the simulations, the root's children have visit counts $N(s^0, a)$. The **improved policy** is simply those counts, normalized and optionally temperature-scaled:

$$ \pi(a \mid s^0) = \frac{N(s^0, a)^{1/T}}{\sum_b N(s^0, b)^{1/T}}. $$

This $\pi$ is sharper and better than the raw network prior $\mathbf{p}^0$ — search has *improved* the policy by allocating more visits to actions that the value backups confirmed are good. The agent samples its real action from $\pi$ (high temperature $T$ early in training for exploration, $T \to 0$ later for maximum strength), and $\pi$ also becomes the training target for the policy head. That is the policy-improvement loop AlphaZero pioneered, now running entirely in MuZero's invented latent space. The deep reason this works is a classic result: visit counts from a well-tuned tree search form a *policy improvement operator* — the search policy is provably at least as good as the prior under mild conditions, so imitating it pushes the network forward.

A subtle and important point: in board games AlphaZero's tree only knows about legal moves because the simulator tells it. MuZero has no such filter inside the tree — it can imagine *illegal* actions, and the dynamics network will happily produce a latent for them. This is fine because MuZero only ever masks to legal moves at the *real* root (where it does know the legal action set), and the internal tree is allowed to be a fuzzy approximation. The model is judged solely by whether the plans it produces lead to good real actions, not by whether its internal states are physically real. This tolerance for "wrong but useful" internal states is a recurring theme, and it is the same freedom that lets MuZero discard irrelevant pixels.

## 4. The training objective

Now the part that makes it learn. MuZero is trained on trajectories collected by self-play. For a sampled position at real time $t$, we *unroll* the model $K$ steps into the future (typically $K=5$) using the actions that were actually taken in that trajectory, and we ask every prediction the unrolled model makes to match a target drawn from the real trajectory. The structure is exactly like training a recurrent network through time: $h_\theta$ produces the initial state, then $g_\theta$ is applied $K$ times in sequence, and at each step $f_\theta$ produces predictions whose errors are backpropagated all the way back through every dynamics step.

![Pipeline diagram of MuZero training: self-play stores trajectories into a replay buffer, a window is sampled, the model is unrolled K steps through the dynamics network, reward value and policy losses are computed, and all three networks are updated by shared backprop](/imgs/blogs/muzero-mastering-games-without-rules-4.png)

Concretely, from observation $o_t$ we compute $s^0 = h_\theta(o_t)$, then for $k = 1 \ldots K$ apply $s^k, r^k = g_\theta(s^{k-1}, a_{t+k})$ where $a_{t+k}$ is the *real action that was actually taken* $k$ steps later in this trajectory. We do not use imagined actions for training the unroll — we condition on the real action sequence so that the targets are well-defined real quantities. At each step $k$ we read $\mathbf{p}^k, v^k = f_\theta(s^k)$. There are three losses, summed over the unroll.

**Policy loss.** The policy head at unrolled step $k$ should match the MCTS-improved policy $\pi_{t+k}$ that was actually computed during self-play at that real timestep — this is a KL divergence between the search policy and the network prior, implemented as a cross-entropy:

$$ \ell^p_k = \mathrm{KL}(\pi_{t+k} \,\|\, p^k) = -\sum_a \pi_{t+k}(a) \log p^k(a) + \text{const}. $$

Minimizing this trains the prior to mimic what search discovered — the policy improvement gets distilled into the network. Note this loss is applied at *every* unroll step $k$, not just at $k=0$: the model is asked to predict, from a latent it imagined $k$ steps deep, the search policy that was computed at the real timestep $t+k$. This is what forces the dynamics network's latents to remain "policy-predictive" several steps into the future.

**Value loss.** The value head should match a bootstrapped **n-step return** target:

$$ z_{t+k} = \sum_{i=0}^{n-1} \gamma^{i} u_{t+k+i} + \gamma^{n} \, \nu_{t+k+n}, $$

where $u$ are the real observed rewards and $\nu_{t+k+n}$ is the MCTS root value computed at that future step during self-play (the bootstrap — it summarizes everything beyond the n-step horizon). The value loss is squared error (board games, where returns are $\pm 1$) or a cross-entropy over a categorical value representation (Atari, where MuZero predicts a *distribution* over a transformed reward scale — this categorical trick, borrowed from distributional RL and combined with an invertible $\text{sign}(x)(\sqrt{|x|+1}-1)+\varepsilon x$ scaling, stabilizes learning across the huge dynamic range of Atari scores):

$$ \ell^v_k = (z_{t+k} - v^k)^2 \quad \text{or categorical cross-entropy}. $$

**Reward loss.** The predicted reward at each dynamics step should match the real reward observed at that step:

$$ \ell^r_k = (u_{t+k} - r^k)^2 \quad \text{or categorical}. $$

The total loss for the sampled trajectory, with L2 regularization, is

$$ L_t = \sum_{k=0}^{K} \left( \ell^p_k + \ell^v_k + \ell^r_k \right) + c \, \lVert \theta \rVert^2. $$

(There is no reward loss at $k=0$, since the root edge has no incoming action and hence no predicted reward; the sum for $\ell^r$ runs $k=1 \ldots K$.) Two things deserve emphasis. First, gradients flow *through the dynamics network* — the unroll is a recurrent computation, so training $g_\theta$ to predict good rewards and to produce latents from which $f_\theta$ predicts good values and policies is what shapes the latent space. The dynamics net is not trained against any ground-truth next-state; there is no "$s^k$ should equal $h_\theta(o_{t+k})$" term in vanilla MuZero. It is trained only so that its outputs support accurate reward, value, and policy predictions. This is the principle of **value equivalence**: the model is correct if it produces the same values and rewards the real environment would, *regardless of whether its internal states resemble real states at all*.

Second, **there is no observation-reconstruction term**. Nothing asks the latent to be decodable back into pixels. The latent is whatever internal code makes the three predictions accurate — a purely *predictive* representation. To keep gradients well-scaled through the recurrent unroll, the paper makes two adjustments: it scales the gradient at the start of the dynamics function by $1/2$ at each step (so each unroll step contributes comparably to the dynamics-net gradient rather than the earliest steps dominating), and it scales the total loss by $1/K$ so that the effective learning rate does not depend on how deep the unroll is.

#### Worked example: computing a value target

Take a slice of an Atari trajectory. The agent is at real step $t=100$. Over the next steps it observes rewards $u_{100}=0, u_{101}=1, u_{102}=0, u_{103}=4, u_{104}=0$, and the MCTS root value computed at step $105$ during self-play was $\nu_{105}=6.5$. With discount $\gamma=0.997$ and an n-step horizon $n=5$, the value target for step 100 is

$$ z_{100} = 0 + 0.997 \cdot 1 + 0.997^2 \cdot 0 + 0.997^3 \cdot 4 + 0.997^4 \cdot 0 + 0.997^5 \cdot 6.5 \approx 0.997 + 3.964 + 6.402 = 11.36. $$

So the value head, when reading the latent the model produced for step 100, is trained to output ≈ 11.36. The bootstrap term $\gamma^5 \nu_{105}$ contributes more than half of it — which is exactly why the quality of the search values $\nu$ matters so much, and why Reanalyze (next section) pays off: it lets us recompute better $\nu$ targets after the network has improved. Notice also how the n-step horizon trades bias for variance: a larger $n$ leans more on observed rewards (low bias, but those rewards are noisy in stochastic games), while a smaller $n$ leans more on the bootstrap (low variance, but biased by the current value net's errors). MuZero's choice of $n=5$ for Atari (and $n = $ episode length for board games, since returns are exact there) is a deliberate point on that trade-off.

#### Worked example: a full K=3 unroll target computation

Make the unroll concrete with a tiny $K=3$ example so you can see every target line up. Suppose at real step $t=40$ in a grid-world the trajectory recorded: actions $a_{41}=\text{RIGHT}, a_{42}=\text{RIGHT}, a_{43}=\text{UP}$; observed rewards $u_{40}=0, u_{41}=0, u_{42}=1, u_{43}=0$; MCTS-improved policies $\pi_{40}, \pi_{41}, \pi_{42}, \pi_{43}$ (each a distribution over the four moves); and MCTS root values $\nu_{40}=0.7, \nu_{41}=0.8, \nu_{42}=0.9, \nu_{43}=0.85$. With $\gamma = 0.99$ and $n=3$:

- Compute $s^0 = h_\theta(o_{40})$, read $p^0, v^0 = f_\theta(s^0)$. Policy target: $\pi_{40}$. Value target: $z_{40} = u_{40} + \gamma u_{41} + \gamma^2 u_{42} + \gamma^3 \nu_{43} = 0 + 0 + 0.99^2 \cdot 1 + 0.99^3 \cdot 0.85 \approx 0.980 + 0.832 = 1.812$.
- Apply $s^1, r^1 = g_\theta(s^0, \text{RIGHT})$, read $p^1, v^1 = f_\theta(s^1)$. Reward target: $u_{41}=0$. Policy target: $\pi_{41}$. Value target: $z_{41} = u_{41} + \gamma u_{42} + \gamma^2 u_{43} + \gamma^3 \nu_{44}$ — and if the episode ran on, $z_{41} \approx 0 + 0.99 \cdot 1 + 0 + \gamma^3 \nu_{44}$.
- Apply $s^2, r^2 = g_\theta(s^1, \text{RIGHT})$, read $p^2, v^2$. Reward target: $u_{42}=1$ (this is the step where the model must predict the reward spike). Policy target: $\pi_{42}$. Value target $z_{42}$ built the same way.
- Apply $s^3, r^3 = g_\theta(s^2, \text{UP})$, read $p^3, v^3$. Reward target: $u_{43}=0$. Policy target $\pi_{43}$, value target $z_{43}$.

The single backward pass through this $K=3$ unroll updates $h_\theta$ (once, at the root), $g_\theta$ (three times, with gradients summed and the half-scaling applied at each entry), and $f_\theta$ (four times, once per node). The reward target $u_{42}=1$ at step $k=2$ is the most informative signal in this slice — it is the only place the model learns that this particular RIGHT-RIGHT sequence cashes in a point, and getting $r^2$ right is what lets future searches plan toward it.

## 5. Reanalyze: squeezing more out of old games

The expensive part of MuZero is self-play — running MCTS hundreds of times per move to generate trajectories. The cheap part is gradient steps. Vanilla MuZero throws away a lot of value: a trajectory played early in training carries policy and value *targets* ($\pi$ and $\nu$) computed by a weak network, and once the network improves, those targets are stale — you are training the new, stronger network to imitate the search of a weaker past self.

**MuZero Reanalyze** (introduced in the same line of work, detailed in Schrittwieser et al. 2021, "Online and Offline Reinforcement Learning by Planning with a Learned Model") fixes this by re-running MCTS on *stored* trajectories using the *latest* network, to recompute fresh policy and value targets, without playing a single new game. The observations and actions in the replay are fixed — they were physically generated by the environment and cannot change — but the *targets* derived from them get refreshed by re-searching from each stored observation with the current weights.

![Graph diagram of how a MuZero value target is built from n-step real rewards plus a bootstrapped MCTS search value, with Reanalyze re-running the latest network to refresh the search value before both feed into the final value target](/imgs/blogs/muzero-mastering-games-without-rules-8.png)

The algorithm is mechanically simple. For a stored trajectory, pick the timesteps you want to train on. For each, take the *real observation* that was recorded, build a fresh root with the current $h_\theta$, run MCTS with the current $g_\theta$ and $f_\theta$, and read off a new improved policy $\pi'$ and a new root value $\nu'$. Replace the stale $\pi, \nu$ stored with the trajectory by these fresh ones. Then compute the n-step value targets using the *refreshed* $\nu'$ as the bootstrap. The reward targets $u$ never change (they are observed facts), and neither do the observations or actions — only the search-derived quantities are recomputed.

Why does this help so much? In supervised terms, your training labels improve over time for free. The policy target $\pi'$ recomputed by a stronger network is a better policy than the one the weak network found at collection time; training toward it is genuine policy improvement on data you already paid to collect. The value bootstrap $\nu'$ recomputed by the stronger network is a lower-bias estimate of the true return. Reanalyze lets MuZero do many more gradient updates per environment step — the replay ratio (gradient updates per environment frame) can be pushed dramatically higher than vanilla off-policy methods tolerate, because the targets are continuously re-derived from a fresh search rather than from a frozen, increasingly-stale snapshot. That is precisely what you want when *environment interaction* is the budget you care about: real robots where each step has wall-clock and wear cost, expensive high-fidelity simulators, or sample-limited Atari benchmarks where the frame count is the score.

There is a clean way to see why this is more than a cheap trick. A standard off-policy value-learning method like DQN reuses old transitions, but its bootstrap target $\max_a Q(s', a)$ is computed by a one-step lookahead under the current network — a relatively weak estimate. Reanalyze's bootstrap $\nu'$ is computed by a *full tree search* under the current network — a much stronger estimate that has already incorporated lookahead. So Reanalyze is reusing old data the way DQN does, but with a far higher-quality target attached to each reused sample. In the Atari sample-efficiency setting, Reanalyze is the difference between "needs a billion frames" and "competitive in a few hundred million," and it is the foundation that EfficientZero (Section 9) builds on to reach the 100k-frame regime. The mechanism is simple and the payoff is large: never let a collected trajectory's targets go stale when recomputing them is comparatively cheap.

## 6. The representation network: pixels to abstract latent, no reconstruction

It is worth dwelling on the representation network because it is where the most common misconception lives. People assume MuZero must be learning a *model of the world* in the sense of "predict the next frame." It is not, and the difference is the whole point — it is the difference between MuZero and a reconstruction-based world model, which we will sharpen in Section 8.

The representation network $h_\theta$ for Atari is a convolutional residual tower. The paper takes a stack of recent frames (96×96 resolution, with the action history and several frames concatenated into the channel dimension) and downsamples it through several stride-2 convolution blocks interleaved with residual blocks, ending at a 6×6 spatial feature grid with 256 channels. For board games it is a residual tower over the board-state planes (for chess, planes encoding piece types, castling rights, repetition counts, and so on). In both cases the output is a tensor with *no prescribed meaning*. The only pressure on it comes from downstream: $f_\theta$ must read accurate policies and values off it, and after applying $g_\theta$ one or more times the resulting latents must *also* yield accurate predictions. There is no decoder, no generator, no reconstruction loss anywhere in the graph.

Because there is no decoder and no reconstruction loss, the network is under zero obligation to preserve information that does not help predict reward, value, or policy. If the cloud textures in the Atari background are irrelevant to the score, the latent is free to discard them entirely. If a flickering animation cycle of a sprite that never affects gameplay costs bits to represent, those bits get dropped. This is a feature, not a bug: it concentrates representational capacity on the decision-relevant structure of the environment. The latent is a *task-shaped* compression of the observation, not a faithful one. Contrast this with an autoencoder, whose entire objective is faithful reconstruction and which therefore *must* spend capacity on every visually salient detail regardless of whether it matters for control.

This also means MuZero's latent states are not guaranteed to correspond one-to-one with real environment states. Two genuinely different game positions could in principle map to similar latents if they imply the same future rewards and optimal play — the model is allowed to *abstract over* distinctions that do not change the value or the best move. A latent reached by imagining an illegal sequence of moves need not correspond to any real position at all; it is a point in a learned space that the dynamics net invented, useful only insofar as the predictions read off it lead to good real decisions. The model is consistent enough to plan with, not faithful enough to render. That looseness is exactly what lets it skip the impossible job of pixel prediction — and, as a side benefit, it is part of why the representation tends to generalize: by throwing away decision-irrelevant variation, the latent space is smaller and smoother in the directions that matter for planning.

## 7. MuZero vs AlphaZero: same engine, no rulebook

The clean way to see what MuZero buys you is side by side with its parent. The search engine is the same — PUCT selection, neural-guided expansion, value backup, visit-count policy. What changes is *what fills the role of the environment* inside that engine.

![Before-after diagram contrasting AlphaZero, which needs a real simulator with known rules and works on board games only, against MuZero, which uses a learned latent simulator, needs no rules, and extends to Atari and other domains](/imgs/blogs/muzero-mastering-games-without-rules-3.png)

On the board games AlphaZero was built for, MuZero matched or slightly exceeded it. Trained from scratch by self-play, MuZero reached AlphaZero's level in Go and surpassed its published results in chess and shogi, despite never being given the rules of any of them — it had to infer legal play, captures, check, promotion, and terminal conditions purely from the reward signal and self-play outcomes. That a learned model could match a *perfect* one on the exact games the perfect model was designed for is the surprising result. The intuition for why this is even possible: in board games the reward is sparse and terminal ($\pm 1$ at the end), so the model only needs its latents to support an accurate value (who is winning) and policy (what to play) — it does not need to reconstruct the exact board, only to preserve whatever about the board determines the outcome. Apparently that is learnable from self-play alone.

Then comes the part AlphaZero structurally cannot do: Atari. Across the 57-game Atari benchmark (the Arcade Learning Environment), MuZero achieved a mean human-normalized score of **731%** and a median around 1047% (with the large-scale training budget), placing it at or above the previous model-free state of the art (the R2D2 and Agent57-era methods) while being the first method to bring tree-search planning to pixel-based Atari at that level. Same search engine, same self-play improvement loop, no rulebook — and it crossed from board games into video games. The only structural change required was the one captured in the table below.

| Property | AlphaZero | MuZero |
|---|---|---|
| Needs the rules / a perfect simulator | Yes | No |
| Model of dynamics | Given (hard-coded) | Learned ($g_\theta$) |
| Search space | Real game states | Learned latent states |
| Predicts intermediate rewards | No (terminal only) | Yes (every edge) |
| Domains | Go, chess, shogi | + Atari (57 games) |
| Reconstructs observations | n/a | No |
| Headline result | Superhuman board play | 731% mean human-normalized Atari |

The reason "predicts intermediate rewards" flips to Yes is itself instructive. Board games are sparse: reward only at the end ($\pm 1$). Atari is dense: points arrive throughout an episode, often hundreds of them. To plan in a dense-reward domain you must predict reward at every step of the rollout, because the value of a plan is the sum of rewards along it — you cannot wait for a terminal signal that may be thousands of steps away or may never come. That is exactly why MuZero's dynamics net outputs a reward and why the backup accumulates discounted rewards along the path. That single generalization — reward at every edge instead of only at terminals — is what lets one architecture span both sparse board games and dense arcade games. AlphaZero's terminal-only backup is just the special case of MuZero's reward-at-every-edge backup where all intermediate rewards happen to be zero.

## 8. Why MuZero is not a "world model"

MuZero is frequently lumped in with world-model methods like Dreamer, and the distinction is worth getting right because it governs *when each is the correct tool*. Both learn a latent dynamics model; both plan or learn a policy by rolling that model forward in latent space. The split is in *what objective shapes the latent*.

A world model in the Dreamer (Hafner et al.) sense learns a generative latent dynamics model trained substantially by **reconstructing observations**. Concretely, Dreamer's recurrent state-space model is trained to maximize a variational lower bound (the ELBO), one term of which is a *reconstruction loss*: from the latent, a decoder must regenerate the observation (the image), and the latent dynamics must predict future latents whose decoded images match future frames. The latent must therefore be rich enough to *predict and decode* the next image. The agent then learns an actor and critic by imagining long rollouts inside this generative model. The representation is shaped, in large part, by an observation-prediction objective: it has to capture whatever is needed to reproduce what the world looks like.

MuZero's latent is shaped by an entirely different objective: predict *reward, value, and policy*, and nothing else. No reconstruction, no decoder, no observation-prediction term in the loss, no ELBO. This is the value-equivalence principle in its purest form: a model is "good" if and only if it agrees with the real environment on the quantities that drive decisions (rewards and values under the relevant policies), and it is explicitly *not* required to agree on observations. The consequence is that MuZero's latent can be far more compressed and task-specialized — it keeps only what affects returns — whereas a reconstruction-trained world model must keep everything visually salient, including decision-irrelevant detail like background textures, particle effects, or cosmetic animation. MuZero trades the ability to *visualize* its imagined future for sharper focus on the quantities that drive decisions.

Why does this matter for generalization, beyond mere efficiency? Two reasons. First, a value-equivalent model only has to be accurate in the directions that change the value function, which is a far lower-dimensional and smoother target than "every pixel," so a given amount of model capacity goes much further on the part of the problem that planning actually uses. Second, reconstruction objectives are vulnerable to a specific failure: they will faithfully model a large, easy-to-predict but irrelevant part of the observation (say, a scrolling background) while under-modeling a small, hard-to-predict but decision-critical part (a fast projectile), because the reconstruction loss is dominated by pixel count, not decision-relevance. Value equivalence is immune to this by construction — it never optimizes for pixels, so it never wastes capacity defending the wrong ones. The flip side is the obvious cost: a value-equivalent model cannot render its imagination, so you cannot inspect "what does it think will happen" as an image, and the model is less reusable across tasks with different rewards (the latent is specialized to *this* reward structure).

This is why the two methods shine in different regimes. When observations are cheap to simulate and you want a general, reusable model of the environment (and perhaps to inspect what the model "thinks" will happen), a reconstruction-based world model is attractive. When you specifically need *strong planning* and do not care whether the model can render — and especially when reconstruction would waste capacity on irrelevant pixels — MuZero's predictive-only latent is the better bet. The decision tree below summarizes the choice across the model-based family.

![Tree diagram for choosing a method: if no planning is needed use PPO or SAC, if planning is needed and rules are known use AlphaZero, if planning is needed and rules are unknown use MuZero, and if there is a cheap visual simulator a Dreamer-style world model fits](/imgs/blogs/muzero-mastering-games-without-rules-7.png)

It is worth noting that the line between the two has blurred since: EfficientZero (next section) adds a *self-supervised consistency* term to MuZero that pushes the predicted latent to agree with the encoded next observation — which is a step *toward* a reconstruction-like signal, but in latent space rather than pixel space, capturing some of the benefit (a denser learning signal for the dynamics net) without the cost (decoding to pixels). So the design space is a spectrum from pure value-equivalence (vanilla MuZero) to pure reconstruction (Dreamer), and the most sample-efficient methods sit in between.

## 9. Compute, the simulation budget, and EfficientZero

Honesty about cost: original MuZero is expensive. The large-scale board-game and Atari results were trained on substantial TPU fleets — the Go-scale training used on the order of hundreds of TPUs (around 800 third-generation TPUs were used for training the board-game agents, with many more actors generating self-play), with self-play actors running MCTS continuously to feed a central learner in a distributed actor-learner setup. The dominant cost is the simulation budget: every real move requires $K$ network evaluations to build the tree ($K=50$ simulations per move for Atari, $800$ for board games), and self-play needs millions of moves to generate enough trajectories. Doubling the simulation count roughly doubles per-move search compute; halving it weakens play because the improved policy $\pi$ is built from fewer, noisier visit counts. The 731% Atari figure came from a regime measured in hundreds of millions to billions of frames, not a laptop run, and the board-game results consumed a comparable or larger budget.

There is a useful way to decompose where the cost goes. Per real move, MuZero pays for: one $h_\theta$ call (the root encoder, the heaviest single network), then per simulation one $g_\theta$ call and one $f_\theta$ call. With 50 simulations that is one big encoder pass plus fifty smaller dynamics+prediction passes. Multiply by the millions of moves in self-play, and the actors — not the learner — dominate the wall-clock and accelerator bill. This is exactly the asymmetry Reanalyze exploits: gradient steps on the learner are cheap relative to self-play on the actors, so squeezing more learning out of each collected trajectory is the highest-leverage optimization.

That cost motivated **EfficientZero** (Ye et al., 2021, "Mastering Atari Games with Limited Data"). It targets the punishing Atari 100k benchmark — only 100,000 environment interactions, roughly two hours of human play, versus the ~200 hours (tens of millions of frames) a classic DQN needs to reach comparable scores — and reaches a mean human-normalized score around **194%**, the first method to exceed human-level mean performance in that data-limited regime, a large jump over prior model-free methods, and it does so on the order of a handful of GPUs (roughly 4 GPUs for about 7 hours per game in the authors' setup) rather than a TPU fleet. It does so with three additions to MuZero, all attacking the same root problem — too few samples to learn a good model when your only signals are sparse reward, value, and policy losses:

1. **A self-supervised consistency loss.** The latent that the dynamics net predicts for the next step, $s^{k}$, is pushed (via a SimSiam-style self-supervised objective, with a stop-gradient and a predictor head to avoid collapse) to agree with the latent the *representation* net computes from the *actual* next observation, $h_\theta(o_{t+k})$. This injects a dense learning signal into the dynamics model at every step, beyond the sparse reward/value/policy losses, dramatically improving the learned model's one-step accuracy when data is scarce. It is the partial step toward a reconstruction signal mentioned in Section 8 — but in latent space, so it never has to decode a pixel.
2. **Value prefix prediction.** Instead of predicting per-step reward and summing it through the backup (where each step's error compounds), it predicts the *cumulative reward over a horizon* (the "value prefix") with an LSTM. Predicting a sum directly is easier to learn correctly than predicting each addend and accumulating, and it sidesteps the "state aliasing" problem where the model cannot tell exactly *when* a reward will arrive but can tell it arrives within a window. This reduces compounding model error in the backup.
3. **Off-policy correction in the value target.** Stored trajectories age; their bootstrap targets were computed under an older policy and become biased as the policy moves on. EfficientZero adjusts how far the n-step return reaches based on the data's age — using shorter horizons (more bootstrap, less reliance on stale observed rewards from an old policy) for older data — controlling the bias from stale targets.

EfficientZero is the proof that MuZero's framework is not inherently sample-hungry — the hunger came from learning a model with only sparse signals. Add a richer (self-supervised) signal to the dynamics net, make reward prediction easier, and correct for off-policy staleness, and the same planning machinery becomes sample-efficient enough for the 100k regime, on commodity GPUs. The lineage as a whole is one steady removal of dependencies: AlphaGo removed nothing (it needed human data, rules, and a fleet); AlphaZero removed human data; MuZero removed the rules; EfficientZero removed the enormous data and compute appetite.

![Timeline from AlphaGo 2016 through AlphaZero 2017, MuZero 2020 reaching games and Atari, MuZero Reanalyze 2021, EfficientZero 2021, to MuZero applied to video compression in 2022](/imgs/blogs/muzero-mastering-games-without-rules-6.png)

## 10. MuZero beyond games: the video-compression result

The strongest evidence that MuZero is a general planner, not a game-specific trick, is that DeepMind deployed it on a real production problem with no notion of "winning." In Mandhane et al. (2022, "MuZero with Self-competition for Rate Control in VP9 Video Compression"), MuZero was applied to **adaptive bitrate / rate control** in the VP9 video codec used at YouTube scale.

The framing fits RL naturally. The agent observes statistics about the video and the encoder state — features describing the content's complexity, the bits already spent, the buffer state, and the quality achieved so far. Its action is a quantization parameter (QP) for the next chunk, which effectively controls how aggressively to compress it: a high QP spends fewer bits but lowers quality, a low QP spends more bits for higher quality. The environment is the actual VP9 encoder, whose true dynamics — how a QP choice on *this particular content* maps to bits used and quality achieved, given everything encoded before it — are unknown, content-dependent, and certainly not expressible as a clean rules engine you could call inside a search loop. There is no rulebook. So MuZero learns a model of the encoder's behavior from logged encodes and plans bitrate decisions to hit a target bitrate while maximizing quality, exactly as it learned a model of chess and planned moves.

The reported result was a meaningful reduction in average bitrate — on the order of a few percent (low-single-digit percent, with the paper reporting roughly 4% average bitrate savings on its evaluation set of YouTube videos) at matched quality versus the production heuristic rate controller — and at YouTube's volume, a few percent of bytes served is an enormous absolute saving in storage and bandwidth. A clever piece of the work, *self-competition*, addresses a problem specific to this setting: the natural objective (minimize bits at a quality target) is a continuous, absolute quantity, but MuZero's machinery is built around the kind of crisp win/loss signal that board games provide. Self-competition turns the absolute objective into a *relative* signal by comparing the agent's current episode against a baseline drawn from its own past performance on the same video — "did I beat my previous best on this content?" — recovering the binary, normalized reward structure MuZero's search and value normalization handle best.

The takeaway is not the exact percentage. It is that the *identical* method that learned chess from nothing learned a video encoder's behavior well enough to beat a hand-tuned production controller — because at no point did MuZero need the rules of either. Any sequential decision problem with a reward, unknown or expensive dynamics, and a genuine need to look ahead (the QP you pick now affects the bits and quality available for everything after it, so greedy per-chunk decisions are suboptimal) is in scope. That is the generality claim, demonstrated on a system serving billions of video views.

## 11. Implementation: the three networks and the loops

Here is a compact but real PyTorch sketch of the three networks, kept architecturally honest while small enough to read. Latents are vectors here for clarity; the real Atari version uses conv feature grids of shape 6×6×256, and the board-game version uses residual towers, but the interfaces are identical.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class Representation(nn.Module):
    """h: stacked observations -> initial latent state s^0."""
    def __init__(self, obs_dim, latent_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, latent_dim),
        )

    def forward(self, obs):
        s = self.net(obs)
        return normalize_latent(s)   # scale to [0,1] per-sample, see below

class Dynamics(nn.Module):
    """g: (latent s, one-hot action a) -> (next latent s', reward r)."""
    def __init__(self, latent_dim, n_actions, hidden=256):
        super().__init__()
        self.n_actions = n_actions
        self.trunk = nn.Sequential(
            nn.Linear(latent_dim + n_actions, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.to_state = nn.Linear(hidden, latent_dim)
        self.to_reward = nn.Linear(hidden, 1)

    def forward(self, s, a_onehot):
        x = self.trunk(torch.cat([s, a_onehot], dim=-1))
        s_next = normalize_latent(self.to_state(x))
        r = self.to_reward(x)
        return s_next, r

class Prediction(nn.Module):
    """f: latent s -> (policy logits p, value v)."""
    def __init__(self, latent_dim, n_actions, hidden=256):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden, n_actions)
        self.value_head = nn.Linear(hidden, 1)

    def forward(self, s):
        x = self.trunk(s)
        return self.policy_head(x), self.value_head(x)

def normalize_latent(s):
    # MuZero scales hidden states into [0,1] to keep them on the same
    # range as the action encoding fed into the dynamics network.
    s_min = s.min(dim=-1, keepdim=True).values
    s_max = s.max(dim=-1, keepdim=True).values
    return (s - s_min) / (s_max - s_min + 1e-5)
```

The latent normalization at the end of both $h$ and $g$ is not cosmetic: keeping latents in $[0,1]$ matches the range of the one-hot action encoding the dynamics net concatenates onto its input, which keeps the two information streams (state and action) on comparable scales and stabilizes the recurrent unroll. This is one of those small details that the paper found mattered for the single hyperparameter set to transfer across very different domains.

The training step unrolls the model $K$ steps and sums the three losses. Note the half-scaling of the gradient into the dynamics trunk, which keeps the recurrent unroll well-conditioned so that the earliest unroll steps do not receive disproportionately large gradients.

```python
def muzero_loss(repr_net, dyn_net, pred_net, batch, K=5, gamma=0.997):
    # batch: obs[t], actions[t..t+K], target_pi[t..t+K],
    #        target_value[t..t+K], target_reward[t..t+K]
    obs, actions, target_pi, target_v, target_r = batch
    s = repr_net(obs)                       # s^0 from real observation
    policy_logits, value = pred_net(s)

    loss = 0.0
    # step 0: only policy + value (no reward predicted at the root edge)
    loss = loss + F.cross_entropy(policy_logits, target_pi[:, 0])
    loss = loss + F.mse_loss(value.squeeze(-1), target_v[:, 0])

    for k in range(1, K + 1):
        a_onehot = F.one_hot(actions[:, k - 1], dyn_net.n_actions).float()
        s, r = dyn_net(s, a_onehot)
        # scale the gradient flowing back into the dynamics path by 1/2
        s = 0.5 * s + 0.5 * s.detach()
        policy_logits, value = pred_net(s)
        loss = loss + F.cross_entropy(policy_logits, target_pi[:, k])
        loss = loss + F.mse_loss(value.squeeze(-1), target_v[:, k])
        loss = loss + F.mse_loss(r.squeeze(-1), target_r[:, k])
    return loss / K   # scale by unroll length so LR is K-independent
```

Next, the construction of the training targets from a stored trajectory — the n-step value bootstrap that the worked examples computed by hand, now as code. This is what turns a recorded game into the `target_v` and `target_r` arrays the loss consumes.

```python
def make_targets(traj, t, K=5, n=5, gamma=0.997):
    """Build (policy, value, reward) targets for unroll steps 0..K from a
    stored trajectory, starting at real timestep t."""
    target_pi, target_v, target_r = [], [], []
    for k in range(K + 1):
        step = t + k
        if step < len(traj.rewards):
            # n-step bootstrapped value target
            z = 0.0
            for i in range(n):
                if step + i < len(traj.rewards):
                    z += (gamma ** i) * traj.rewards[step + i]
            boot = step + n
            if boot < len(traj.root_values):
                z += (gamma ** n) * traj.root_values[boot]  # nu, the bootstrap
            target_v.append(z)
            target_pi.append(traj.search_policies[step])     # pi from MCTS
            target_r.append(traj.rewards[step - 1] if k > 0 else 0.0)
        else:
            # past episode end: absorbing targets (zero reward, zero value)
            target_v.append(0.0)
            target_pi.append(uniform_policy(traj.n_actions))
            target_r.append(0.0)
    return target_pi, target_v, target_r
```

The MCTS itself, condensed to its load-bearing logic — selection by PUCT, expansion through the dynamics net, evaluation by the prediction net, and reward-aware backup:

```python
import math

class Node:
    def __init__(self, prior):
        self.prior = prior          # P(s,a) from f
        self.visit_count = 0        # N(s,a)
        self.value_sum = 0.0
        self.reward = 0.0           # predicted r on the edge into this node
        self.latent = None          # s for this node
        self.children = {}          # action -> Node

    def value(self):
        return self.value_sum / self.visit_count if self.visit_count else 0.0

def ucb_score(parent, child, c1=1.25, c2=19652, min_q=0.0, max_q=1.0):
    pb = (math.log((parent.visit_count + c2 + 1) / c2) + c1)
    pb *= math.sqrt(parent.visit_count) / (child.visit_count + 1)
    prior_term = child.prior * pb
    if child.visit_count > 0:
        q = child.reward + 0.997 * child.value()
        q = (q - min_q) / (max_q - min_q + 1e-8)   # normalize Q into [0,1]
    else:
        q = 0.0
    return q + prior_term

def run_mcts(root, dyn_net, pred_net, n_simulations=50, gamma=0.997):
    for _ in range(n_simulations):
        node, path = root, [root]
        # SELECT: descend until we hit an unexpanded node
        while node.children:
            action, node = max(node.children.items(),
                               key=lambda kv: ucb_score(path[-1], kv[1]))
            path.append(node)
        # EXPAND + EVALUATE: roll the dynamics net forward one step in latent space
        parent = path[-2]
        a_onehot = one_hot(action)
        s_next, r = dyn_net(parent.latent, a_onehot)
        logits, v = pred_net(s_next)
        node.latent, node.reward = s_next, float(r)
        priors = softmax(logits)
        for a, p in enumerate(priors):
            node.children[a] = Node(prior=float(p))
        # BACKUP: accumulate discounted reward + bootstrapped value up the path
        g = float(v)
        for nd in reversed(path):
            nd.value_sum += g
            nd.visit_count += 1
            g = nd.reward + gamma * g
    visits = {a: c.visit_count for a, c in root.children.items()}
    total = sum(visits.values())
    return {a: n / total for a, n in visits.items()}   # improved policy pi
```

And the outer self-play-plus-training loop, in faithful pseudocode. This is the cycle that the whole post has been building toward: collect a game by searching, store it, then learn from it (optionally re-searching old games via Reanalyze).

```python
def muzero_train(env, repr_net, dyn_net, pred_net, optimizer,
                 total_steps, sims=50, K=5, reanalyze=True):
    buffer = ReplayBuffer(capacity=1_000_000)
    while step_count() < total_steps:
        # ---- SELF-PLAY: generate a trajectory using MCTS ----
        obs, traj = env.reset(), Trajectory()
        done = False
        while not done:
            root = Node(prior=0.0)
            root.latent = repr_net(encode(obs))
            logits, _ = pred_net(root.latent)
            expand_root(root, softmax(logits))          # legal-move mask here
            add_exploration_noise(root)                 # Dirichlet at root
            pi = run_mcts(root, dyn_net, pred_net, sims) # improved policy
            action = sample(pi, temperature=current_temperature())
            next_obs, reward, done, _ = env.step(action)
            traj.store(obs, action, reward, pi, root.value())
            obs = next_obs
        buffer.add(traj)

        # ---- TRAINING: many gradient steps per game ----
        for _ in range(num_updates_per_game()):
            batch = buffer.sample_window(K=K)
            if reanalyze:
                batch = reanalyze_targets(batch, repr_net, dyn_net,
                                          pred_net, sims)  # fresh pi, v
            loss = muzero_loss(repr_net, dyn_net, pred_net, batch, K=K)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
```

Two implementation details in that loop are easy to miss but are load-bearing. The `add_exploration_noise(root)` line mixes Dirichlet noise into the root priors *only at the real root* (not in the interior of the tree), which guarantees every legal action gets some search even when the network's prior is very confident — this is the exploration source that keeps self-play from collapsing onto a narrow set of openings. And `expand_root` is the *one place* the legal-move mask is applied: inside the tree, illegal actions are tolerated (Section 3), but the action MuZero actually commits to must be legal, so the root is masked.

The `reanalyze_targets` call is the entire Section 5 idea in one line: re-run MCTS on the stored observations/actions with the *current* networks to refresh $\pi$ and the value bootstrap before computing the loss. A minimal sketch makes the data flow explicit:

```python
def reanalyze_targets(batch, repr_net, dyn_net, pred_net, sims):
    """Refresh policy & value targets on stored data using current weights.
    Observations and actions are FIXED; only the search-derived targets change."""
    fresh = []
    for sample in batch:
        root = Node(prior=0.0)
        root.latent = repr_net(encode(sample.obs))   # current h_theta
        logits, v = pred_net(root.latent)
        expand_root(root, softmax(logits))
        pi_new = run_mcts(root, dyn_net, pred_net, sims)  # current g, f
        sample.target_pi = pi_new                     # better policy target
        sample.root_value = root.value()              # better value bootstrap
        fresh.append(sample)
    return rebuild_value_targets(fresh)               # n-step with fresh nu
```

A practical launch, if you would rather use a maintained open-source implementation than write the learner from scratch:

```bash
# Use a community MuZero implementation (e.g. muzero-general) to train CartPole.
git clone https://github.com/werner-duvaud/muzero-general
cd muzero-general
python muzero.py cartpole        # config: 50 sims/move, K=5 unroll, gamma=0.997
# Watch mean episode return climb toward 500 (CartPole-v1 cap) over training.
```

#### Worked example: a tiny MCTS tree by hand

Take a root with three legal actions and priors $P = [0.5, 0.3, 0.2]$, all $Q=0$, all $N=0$. With $c_1=1.25, c_2=19652$ and $N_{\text{parent}}=0$, the exploration term's $\sqrt{N_{\text{parent}}}$ factor is 0 on the very first selection, so we fall back to the implementation's convention of seeding the root by expanding all children, then on the first real simulation $N_{\text{parent}}=1$ and the PUCT score reduces to roughly $P(s,a)\cdot(c_1 + \log(\cdots))/(1+0)$. Action 0 (prior 0.5) wins, gets expanded, and suppose the dynamics net returns reward $r=0$ and the prediction net value $v=0.8$; the backup sets $Q(a_0)=0 + 0.997\cdot 0.8 \approx 0.80$ and $N(a_0)=1$. On simulation 2, action 0's exploration bonus has shrunk (its $N$ is now 1, so its $1/(1+N)$ factor halved) while actions 1 and 2 still have $N=0$, so action 1 — with prior 0.3 and a full exploration bonus — is now competitive and gets selected. Suppose it returns $v=0.4$, a worse latent: $Q(a_1) \approx 0.40$. The search has started to *spread*, sampling alternatives, but it has also learned action 0 looks better.

#### Worked example: a 4-step simulation tree with backup

Now follow a deeper single simulation through a small tree to see the reward-aware backup do real work. Imagine the tree already has a few nodes, and selection descends: root $\to$ child A (edge reward $r_A = 0$, an already-expanded node) $\to$ grandchild B (edge reward $r_B = 1$, the agent broke a brick on this imagined step) $\to$ unexpanded leaf via action $a$. We expand the leaf: $g_\theta$ returns the leaf latent and edge reward $r_{\text{leaf}} = 0$, and $f_\theta$ returns leaf value $v_{\text{leaf}} = 2.0$. With $\gamma = 0.997$, the backup climbs from the leaf:

- At the leaf: $G = v_{\text{leaf}} = 2.0$. Update the leaf's stats with 2.0.
- Climb to B: $G \leftarrow r_{\text{leaf}} + \gamma G = 0 + 0.997 \cdot 2.0 = 1.994$. Update B with 1.994.
- Climb to A: $G \leftarrow r_B + \gamma G = 1 + 0.997 \cdot 1.994 = 1 + 1.988 = 2.988$. Update A with 2.988.
- Climb to root edge: $G \leftarrow r_A + \gamma G = 0 + 0.997 \cdot 2.988 = 2.979$. Update the root's child-A statistic with 2.979.

Notice how the intermediate reward $r_B = 1$ enters the accumulated return *at B's level and above* but not below it — the brick broken on the B edge counts toward the value of choosing A from the root, but it is already "in the past" from the leaf's perspective, so the leaf's own value does not include it. This is the discounted-reward backup that AlphaZero never needed (its only nonzero reward was at terminals) and that MuZero requires to plan in dense-reward Atari. After many such simulations the visit counts at the root might land at $[31, 13, 6]$ over the three root actions, giving an improved policy $\pi=[0.62, 0.26, 0.12]$ — sharper toward the best action than the raw prior $[0.5, 0.3, 0.2]$, because the value backups confirmed it. That sharpening is the policy improvement the network is then trained to imitate.

## 12. Case studies

**Atari (Schrittwieser et al., 2020).** Across all 57 Arcade Learning Environment games with a single architecture and a single hyperparameter set, large-scale MuZero reached a **mean human-normalized score of 731%** (and a median around 1047%), the first time tree-search planning was brought to pixel Atari at state-of-the-art level. The same agent that learned board games from scratch learned 57 distinct video games, never told the controls or scoring of any of them — it inferred each game's reward structure, the effect of each button, and the relevant on-screen objects purely from play. This is the headline proof that the learned model generalizes far past the board-game domain it was conceived for, and that a single value-equivalent model can span games as different as Pong, Montezuma's Revenge, and Ms. Pac-Man.

**Go, chess, and shogi (same paper).** MuZero matched AlphaZero's superhuman Go strength and exceeded its published chess and shogi results — *without the rules*. It had to learn legal moves, captures, check, promotion, en passant, castling, and terminal/draw conditions purely from self-play outcomes and the terminal $\pm 1$ reward. Matching a perfect-simulator method on the exact games that method was designed for, using only a learned model, is the result that surprised the field most — there was a reasonable prior expectation that a learned model would be strictly worse than a perfect one, and it turned out not to be, because the learned latent only had to be value-equivalent, not faithful.

**EfficientZero (Ye et al., 2021).** On the Atari 100k benchmark (only 100k interactions, ~2 hours of play), EfficientZero reached a **mean human-normalized score of ~194%** and median ~109%, the first method to exceed human-level *mean* performance in that data-limited regime, far ahead of the prior model-free state of the art and achieved on a handful of GPUs rather than a TPU fleet. It added a self-supervised consistency loss, value-prefix prediction, and off-policy correction to MuZero — demonstrating that the framework can be made dramatically sample-efficient by giving the dynamics net a denser signal than sparse reward/value/policy alone. The DQN baseline it is measured against needs roughly two *hundred* hours of gameplay to reach comparable levels, so the headline is a ~100× improvement in sample efficiency for human-level mean play.

**VP9 video compression (Mandhane et al., 2022).** Deployed on YouTube-scale video encoding across a large evaluation set of real videos, MuZero learned the unknown dynamics of the VP9 encoder and planned bitrate (QP) decisions to cut average bitrate by roughly 4% at matched quality versus the production heuristic rate controller — a large saving at that scale, and a demonstration of MuZero on a real, non-game, production problem with no rulebook and a continuous, content-dependent objective tamed via self-competition.

| Result | Domain | Metric | Source |
|---|---|---|---|
| MuZero | 57 Atari games | 731% mean human-normalized | Schrittwieser 2020 |
| MuZero | Go / chess / shogi | matched/beat AlphaZero, no rules | Schrittwieser 2020 |
| EfficientZero | Atari 100k | ~194% mean human-normalized | Ye 2021 |
| MuZero | VP9 rate control | ~4% bitrate cut at matched quality | Mandhane 2022 |

The full lineage, side by side on the axes that decide when to reach for each method, makes the progression concrete:

| Method | Rules required | Sample efficiency | Atari median (human-norm.) | Compute | Domain |
|---|---|---|---|---|---|
| AlphaGo (2016) | Yes + human data | Low | n/a (board only) | TPU fleet | Go |
| AlphaZero (2017) | Yes (perfect sim) | Low | n/a (board only) | TPU fleet | Go, chess, shogi |
| MuZero (2020) | **No** | Low–medium | ~1047% (large-scale) | ~800 TPUs (train) | + 57 Atari |
| MuZero Reanalyze (2021) | No | Medium–high | competitive at far fewer frames | TPU/GPU | board + Atari + offline RL |
| EfficientZero (2021) | No | **High** (Atari 100k) | ~109% in 2 hours of play | ~4 GPUs | Atari (data-limited) |

## 13. When to use MuZero (and when not to)

MuZero is the right tool in a specific corner, and a heavy hammer everywhere else. Reach for it when **all** of these hold: you need *lookahead planning* (the problem rewards thinking several moves ahead, not just reacting — a decision now meaningfully constrains or enables decisions later), the *true dynamics are unknown or too expensive to query* inside a search loop, and you can afford the *self-play compute* to learn a model (or you can afford EfficientZero's smaller-but-still-real budget). Sequential games without an available simulator, and real-world control problems with unknown dynamics and a genuine need to plan (the video-compression case), are the sweet spot.

Do **not** reach for MuZero when a simpler method clearly wins:

- **The rules/simulator are known and exact.** Use AlphaZero — there is no reason to learn a model you already have, and the learned one can only be worse (or at best tie) the perfect one while costing more to train. The only exception is if the perfect simulator is too slow to call inside search, in which case a learned fast surrogate can help.
- **No lookahead is needed.** If a reactive policy suffices (most continuous-control and many real-time tasks where good behavior is "see state, act well"), a model-free method like [PPO](/blog/machine-learning/reinforcement-learning/reinforcement-learning-a-unified-map) or SAC is far simpler, cheaper, and well understood. Planning earns its cost only when it actually changes decisions; if the greedy action under a good value function is already optimal, search is wasted compute.
- **You want an inspectable, reusable world model or have a cheap visual simulator.** A reconstruction-based world model (Dreamer-style) may serve better, since it can render its imagined futures, capture general environment structure, and be reused across tasks with different rewards — MuZero's latent is specialized to one reward structure and cannot be rendered.
- **You are sample-rich but compute-poor.** MuZero's self-play search is compute-intensive (every move is dozens of network passes); if environment steps are essentially free but accelerators are not, a model-free method amortizes better.
- **Tiny tabular problems with a known model.** Just run value iteration or policy iteration; the entire neural-planning machinery is overkill for a small known MDP.

The matrix below places MuZero in its lineage on the axes that actually decide the choice — does it need rules, what domains does it reach, how sample-efficient is it, and where does it land on Atari.

![Matrix comparing AlphaGo, AlphaZero, MuZero, and EfficientZero across rules required, domains supported, sample efficiency, and Atari result, showing MuZero and EfficientZero needing no rules and extending to Atari](/imgs/blogs/muzero-mastering-games-without-rules-5.png)

## 14. Key takeaways

- **MuZero plans with a model it learned itself.** It runs AlphaZero-style MCTS, but every "what happens next" comes from a learned dynamics network operating in latent space, not from a known simulator. It is never told the rules and infers everything it needs from reward, value, and policy signals.
- **Three functions, one objective.** Representation $h$ (observation → latent), dynamics $g$ (latent + action → next latent + reward), prediction $f$ (latent → policy + value), trained jointly through a $K$-step unroll to predict reward, value, and the MCTS-improved policy — gradients flow through $g$, shaping the latent space with no ground-truth-state supervision.
- **It predicts only what the search consumes.** No observation reconstruction, no decoder — the latent is a task-shaped (value-equivalent) compression that keeps only what affects returns. That is why it scales past the impossible job of pixel prediction.
- **Reward at every edge** generalizes the board-game terminal-only backup to dense-reward domains like Atari, which is what lets one architecture span both sparse and dense reward structures.
- **Reanalyze refreshes stale targets for free**, re-running search on old trajectories with the latest network to recompute better policy and value targets — the key to high replay ratios and sample efficiency.
- **MuZero is not a world model in the Dreamer sense.** Different objective (value-equivalent prediction vs reconstruction via an ELBO) implies different strengths; pick by whether you need strong planning or a renderable, reusable model.
- **EfficientZero proves the framework can be sample-efficient** by adding a self-supervised latent-consistency signal, value-prefix prediction, and off-policy correction — reaching human-level mean Atari in ~2 hours of play on a handful of GPUs.
- **It is general.** The same method learned chess from nothing and cut YouTube's video bitrate by ~4% — anywhere with a reward, unknown or expensive dynamics, and a need to look ahead is in scope.
- **Use it only when planning + unknown dynamics + compute budget all hold.** Known rules → AlphaZero; no planning needed → PPO/SAC; cheap simulator + want renders → Dreamer; tiny known MDP → value iteration.

The capstone, [the reinforcement learning playbook](/blog/machine-learning/reinforcement-learning/the-reinforcement-learning-playbook), places MuZero in the full decision tree of methods; come back to it once you have read the model-free track.

## 15. Further reading

- Schrittwieser, Antonoglou, Hubert, et al. "Mastering Atari, Go, chess and shogi by planning with a learned model." *Nature*, 2020. (The MuZero paper — read the pseudocode appendix, which is the cleanest specification of the algorithm anywhere.)
- Schrittwieser, Hubert, Mandhane, et al. "Online and Offline Reinforcement Learning by Planning with a Learned Model." *NeurIPS*, 2021. (MuZero Reanalyze and the extension to offline RL.)
- Ye, Liu, Kurutach, Abbeel, Gao. "Mastering Atari Games with Limited Data." *NeurIPS*, 2021. (EfficientZero — the sample-efficient variant, with the consistency loss and value-prefix details.)
- Mandhane, Zhernov, Rauh, et al. "MuZero with Self-competition for Rate Control in VP9 Video Compression." 2022. (MuZero in production at YouTube scale.)
- Silver, Hubert, Schrittwieser, et al. "A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play." *Science*, 2018. (AlphaZero — the predecessor whose search engine MuZero inherits.)
- Silver, Huang, Maddison, et al. "Mastering the game of Go with deep neural networks and tree search." *Nature*, 2016. (AlphaGo — the start of the lineage.)
- Grimm, Barreto, Singh, Silver. "The Value Equivalence Principle for Model-Based Reinforcement Learning." *NeurIPS*, 2020. (The theory behind why value-equivalent models suffice for planning.)
- Hafner, Lillicrap, Norouzi, Ba. "Mastering Atari with Discrete World Models" (DreamerV2), 2021. (The reconstruction-based world-model contrast.)
- Sutton & Barto. *Reinforcement Learning: An Introduction*, 2nd ed. (Chapter 8 on planning, learning, and learned models.)
- Within series: [the unified RL map](/blog/machine-learning/reinforcement-learning/reinforcement-learning-a-unified-map) and [the RL playbook](/blog/machine-learning/reinforcement-learning/the-reinforcement-learning-playbook).
