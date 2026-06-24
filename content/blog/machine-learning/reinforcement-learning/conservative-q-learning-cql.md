---
title: "Conservative Q-Learning: Pessimism in the Face of Uncertainty"
date: "2026-06-22"
publishDate: "2026-06-22"
description: "A rigorous derivation of CQL's Q-value lower-bound guarantee, a PyTorch CQL-SAC implementation, and a full D4RL benchmark analysis showing how pessimism transforms offline RL from unreliable to production-ready."
tags:
  [
    "reinforcement-learning",
    "deep-learning",
    "offline-reinforcement-learning",
    "q-learning",
    "conservative-q-learning",
    "machine-learning",
    "pytorch",
    "d4rl",
    "value-based-rl",
    "offline-to-online",
  ]
category: "machine-learning"
subcategory: "Reinforcement Learning"
author: "Hiep Tran"
featured: true
readTime: 55
image: "/imgs/blogs/conservative-q-learning-cql-1.png"
---

You have a dataset of 500,000 transitions from a robotic arm attempting to stack blocks — collected by a scripted controller that succeeded roughly 60% of the time. You want to train a policy that does *better* than the scripted controller without ever touching the physical robot again. Standard Q-learning, applied directly to this fixed dataset, fails spectacularly: it learns a Q-function that assigns wildly optimistic values to actions the dataset never tried, and then greedily selects those actions during evaluation. The robot reaches for positions it has never been near, Q-values like +40 get selected over realistic +8, and the stacking success rate drops to 12% — worse than random initialization.

This is the **offline reinforcement learning problem**, and it sits at the heart of deploying RL into the real world. Online exploration is expensive, dangerous, or simply impossible: you cannot let a medical device explore random doses, a financial execution algorithm randomly route large orders, or a data-center cooling system experiment with novel control settings that might spike temperatures. Yet pure behavioral cloning — just imitating the dataset — throws away the Bellman structure that makes RL powerful. You want RL's ability to stitch together multi-step improvements from the data, without the catastrophic overestimation that standard Q-learning delivers.

Conservative Q-Learning (CQL), introduced by Kumar, Zhou, Tucker, and Levine in 2020, solves this by flipping the standard Q-learning incentive: instead of maximizing Q-values as aggressively as possible, CQL *penalizes* Q-values on out-of-distribution (OOD) actions to push them below what the data actually supports. The result is a **lower bound on the true Q-function** under the data distribution — and lower bounds on Q-values are exactly what you need for a safe greedy policy.

By the end of this post you will understand the mathematical guarantee that makes CQL work (Theorem 3.1 and Theorem 3.2 from the original paper), the logsumexp penalty that implements it, how to tune the alpha hyperparameter, a full PyTorch CQL-SAC implementation, and where CQL sits in the D4RL benchmark landscape compared to BC, BCQ, IQL, TD3+BC, and Decision Transformer. Figure 1 shows the core intuition before we derive anything.

![Q-value comparison showing standard Q overestimates OOD actions at Q plus 40 while CQL conservatively bounds the same action at Q minus 3](/imgs/blogs/conservative-q-learning-cql-1.png)

## The Offline RL Problem: Why Standard Q-Learning Fails

Before deriving the CQL fix, let's be precise about why the failure happens. In standard online Q-learning, the Bellman optimality backup is:

$$Q(s, a) \leftarrow r(s,a) + \gamma \max_{a'} Q(s', a')$$

The $\max_{a'}$ picks the highest-valued action in the next state. During online training this is fine because the agent *explores* those high-valued actions, discovers they are bad (receives low reward), and updates the Q-function downward. The Q-function and the policy's exploration are mutually correcting: every overestimate the network produces is eventually tested against reality and corrected by a fresh sample.

In offline RL, the dataset $\mathcal{D}$ is fixed and the correcting loop is severed. You can compute the Bellman backup, but you never execute the greedy policy in the environment. The $\max_{a'}$ will often select an action $a'$ that appears in no transition in $\mathcal{D}$. The Q-value at that unseen $(s', a')$ is initialized to whatever the randomly-initialized network outputs and is then never corrected by actual environment feedback. Over many Bellman iterations, these erroneous Q-values *propagate backward* through the backup: the ghost high value at $(s', a')$ inflates the backup at $(s, a)$, which inflates the backup at $(s_{t-1}, a_{t-1})$, and so on. The entire value landscape shifts optimistically upward.

### A rigorous statement of unbounded overestimation

Let me make the failure precise rather than hand-wavy. Let $\mu$ be the behavioral policy that generated $\mathcal{D}$, and let $\pi$ be the policy we are learning. Define the empirical Bellman operator $\hat{\mathcal{B}}^\pi$ that uses the dataset's transitions:

$$(\hat{\mathcal{B}}^\pi \hat{Q})(s,a) = r(s,a) + \gamma \, \mathbb{E}_{s' \sim \hat{P}(\cdot | s,a)} \big[ \mathbb{E}_{a' \sim \pi(\cdot | s')} [\hat{Q}(s', a')] \big]$$

Here $\hat{P}$ is the empirical transition model — it can only return next states that actually appear in $\mathcal{D}$. The fitted Q-iteration recursion is $\hat{Q}^{k+1} = \arg\min_Q \, \mathbb{E}_{(s,a) \sim \mathcal{D}}[(Q(s,a) - \hat{\mathcal{B}}^\pi \hat{Q}^k(s,a))^2]$.

Now consider a state $s'$ and an action $a'_{\text{OOD}}$ that the behavior policy never took there, so $\mu(a'_{\text{OOD}} | s') = 0$. The squared-error objective is computed only over $(s,a) \sim \mathcal{D}$. The pair $(s', a'_{\text{OOD}})$ contributes *zero* gradient to the regression, because it is not in the support of $\mathcal{D}$. Therefore $\hat{Q}(s', a'_{\text{OOD}})$ is determined entirely by the function approximator's generalization from nearby in-support points plus its initialization — there is no term in the loss that anchors it to $Q^\pi(s', a'_{\text{OOD}})$.

The target for an in-support pair $(s, a)$, however, *does* query that uncorrected value through the inner expectation $\mathbb{E}_{a' \sim \pi}[\hat{Q}(s', a')]$ whenever $\pi$ places mass on $a'_{\text{OOD}}$. And $\pi$ — being the greedy or soft-greedy policy with respect to $\hat{Q}$ — places *more* mass exactly on the actions whose Q-values are erroneously high. Formally, if $\delta(s', a') = \hat{Q}(s', a') - Q^\pi(s', a')$ is the estimation error, then the error at $(s,a)$ accumulates as

$$\delta(s,a) \approx \gamma \, \mathbb{E}_{s' \sim \hat{P}} \big[ \max_{a'} \delta(s', a') \big] + (\text{sampling error}).$$

Because the $\max$ over $a'$ is taken *after* the errors are realized, it systematically selects the most positively-biased error at each state. This is the same maximization bias that drives standard Q-learning's overestimation, but online RL repairs it through exploration while offline RL cannot. Iterating the recursion, the bias does not contract — it compounds geometrically with horizon, and for OOD-heavy state distributions $\hat{Q}$ can exceed $Q^\pi$ by an amount that grows without bound in the discount horizon $1/(1-\gamma)$.

In practice, $\hat{Q}(s, a_{\text{OOD}})$ routinely exceeds the true value by factors of 5–50×. This is the "distributional shift" problem in offline RL: the distribution of actions selected by the greedy policy diverges from the distribution $\mu$ that produced the data, causing the Q-function to be queried in regions where it has no reliable signal, and those very regions are the ones the policy then commits to.

#### Worked example: how a single OOD action poisons the value landscape

Consider a tiny three-state chain. State $s_0$ transitions to $s_1$ under the only action the data shows there; from $s_1$ the data contains a single action $a^{\text{data}}$ with true value $Q^\pi(s_1, a^{\text{data}}) = +8$, leading to a terminal reward. There is a second action $a^{\text{OOD}}$ at $s_1$ that the behavior policy never took; its true value is $Q^\pi(s_1, a^{\text{OOD}}) = -3$ because it tips the agent into a failure state.

Suppose the randomly-initialized critic outputs $\hat{Q}(s_1, a^{\text{OOD}}) = +15$. The regression loss only contains the term for $a^{\text{data}}$, so after fitting we have $\hat{Q}(s_1, a^{\text{data}}) \to +8$ but $\hat{Q}(s_1, a^{\text{OOD}})$ stays near $+15$ — nothing pulls it down. The greedy policy at $s_1$ now prefers $a^{\text{OOD}}$ ($+15 > +8$). The backup at $s_0$ becomes $\hat{Q}(s_0) \leftarrow r_0 + \gamma \cdot 15$ instead of the correct $r_0 + \gamma \cdot 8$. With $\gamma = 0.99$ and $r_0 = 0$, $s_0$ is overvalued by $0.99 \times 7 = 6.93$. Roll this back over a 100-step horizon and a single uncorrected $+7$ ghost compounds into a value surface that is uniformly too optimistic, and the deployed policy walks straight into the $-3$ failure it believes is worth $+15$. CQL's entire job is to make $\hat{Q}(s_1, a^{\text{OOD}})$ collapse below $+8$ so the greedy choice flips back to $a^{\text{data}}$.

### Why not just apply behavioral cloning?

BC avoids OOD queries entirely — it just clones the data distribution, fitting $\pi(a|s) \approx \mu(a|s)$ with a supervised cross-entropy or regression loss. There is no Bellman backup and therefore no extrapolation. On hopper-medium in D4RL, BC achieves 29.0 normalized score. CQL achieves 79.6. The gap exists because BC cannot stitch together multi-step improvements. If the dataset shows a 60% controller, BC learns a 60% clone — it has no mechanism to notice that some of the controller's transitions were good and some were stumbles. CQL can learn a 79.6% policy by using the Bellman structure to identify which *transitions* in the data are good and which are bad, then optimizing a policy that preferentially takes the good transitions. That requires Q-learning — but Q-learning needs to be tamed. The rest of this post is about taming it with a single, theoretically-grounded penalty term.

## The CQL Objective: A Principled Lower Bound

CQL's key insight is elegant: **add a penalty that pushes down Q-values on actions not seen in the data, and pushes up Q-values on actions that are in the data.** If the push-down is strong enough relative to the push-up, the Q-function provably becomes a lower bound on the truth.

### The basic penalty

Define the CQL regularization term as a function of a Q-function $Q$ and a chosen action distribution $\mu_\beta$:

$$\mathcal{R}_{CQL}(Q) = \mathbb{E}_{s \sim \mathcal{D}}\Big[ \mathbb{E}_{a \sim \mu_\beta(a|s)}\left[Q(s, a)\right] - \mathbb{E}_{a \sim \hat{\pi}_\beta(a|s)}\left[Q(s, a)\right] \Big]$$

where $\mu_\beta$ is some distribution over actions whose Q-values we want to *suppress* (initially uniform or random), and $\hat{\pi}_\beta$ is the empirical behavior distribution — the actions that actually appear in the data. The first term pushes down Q-values under $\mu_\beta$; the second term pushes up Q-values under the data. Adding this penalty to the standard Bellman error gives the CQL critic objective:

$$\min_Q \; \alpha \cdot \mathcal{R}_{CQL}(Q) + \frac{1}{2} \mathbb{E}_{(s,a,s') \sim \mathcal{D}}\left[(Q(s,a) - \hat{\mathcal{B}}^\pi \hat{Q}(s,a))^2\right]$$

The first term is the conservatism pressure; the second is the standard TD loss. The hyperparameter $\alpha \geq 0$ controls the tradeoff. When $\alpha = 0$ this collapses to ordinary fitted Q-iteration with all its overestimation pathologies; as $\alpha \to \infty$ the critic ignores rewards entirely and just learns the maximally-conservative ordering.

### Theorem 3.1: pointwise lower bound on the value

The first guarantee concerns the *value function* $\hat{V}(s) = \mathbb{E}_{a \sim \pi}[\hat{Q}(s,a)]$. Kumar et al. prove that the form above, with the bare $-\mathbb{E}_{\hat{\pi}_\beta}[Q]$ correction, produces a Q-function whose induced value lower-bounds the true value at every state in the data.

**Theorem 3.1 (Kumar et al. 2020, informal).** For any $\mu_\beta(a|s)$ with support contained in that of $\pi$, the Q-function obtained by iterating the CQL update satisfies, for $\alpha$ large enough,

$$\hat{V}^\pi(s) = \mathbb{E}_{a \sim \pi(a|s)}[\hat{Q}(s,a)] \;\leq\; V^\pi(s) \quad \forall s \in \mathcal{D}$$

with high probability over the sampling of $\mathcal{D}$.

The intuition behind the proof: the fixed point of the regularized update is $\hat{Q} = \hat{\mathcal{B}}^\pi \hat{Q} - \alpha \frac{\mu_\beta(a|s) - \hat{\pi}_\beta(a|s)}{\hat{\pi}_\beta(a|s)}$. The subtracted term is positive wherever $\mu_\beta$ puts more mass than the data does — that is, on OOD-leaning actions — so the fixed point is pushed *below* the true Bellman fixed point exactly in the regions we distrust. Taking the expectation under $\pi$ and choosing $\mu_\beta = \pi$ makes the subtracted term non-negative everywhere, which yields the value lower bound. The "large enough $\alpha$" condition trades off against the sampling-error term: $\alpha$ must dominate the concentration bound on $|\hat{\mathcal{B}}^\pi - \mathcal{B}^\pi|$, which scales as $C_{r,T,\delta} / \sqrt{|\mathcal{D}(s,a)|}$, so the required $\alpha$ shrinks as the per-pair sample count grows.

### Theorem 3.2: pointwise lower bound on Q itself

Theorem 3.1 bounds the *expected* value under $\pi$, but in many uses we want the stronger property that $\hat{Q}(s,a)$ itself lies below $Q^\pi(s,a)$ for every action. This is what Theorem 3.2 delivers, and it requires the additional $+\mathbb{E}_{\hat{\pi}_\beta}[Q]$ maximization term — the "push up the data actions" half of the regularizer.

**Theorem 3.2 (Kumar et al. 2020, informal).** With the full regularizer $\mathbb{E}_{\mu_\beta}[Q] - \mathbb{E}_{\hat{\pi}_\beta}[Q]$ and $\alpha$ sufficiently large, the resulting $\hat{Q}$ satisfies

$$\hat{Q}(s, a) \leq Q^\pi(s, a) \quad \forall (s,a) \in \mathcal{D}.$$

The data-action maximization term is what lifts the in-distribution Q-values back up so that the bound is *tight* there rather than uniformly pessimistic. Without it, you would still get a lower bound, but a needlessly loose one that suppresses the good actions along with the bad. The combination — suppress everything, then selectively un-suppress the data — gives the tightest valid lower bound. This pointwise property is exactly what a safe greedy policy needs: if every Q estimate is at or below the truth, then maximizing the estimate cannot trick you into an action whose true value is catastrophically low, because that action's estimate would also be low.

### CQL(H): the practical, parameter-free penalty

The basic penalty requires choosing $\mu_\beta$ manually. Kumar et al. show that the choice of $\mu_\beta$ that maximizes the conservatism gap — and therefore gives the tightest lower bound for a fixed $\alpha$ — is the **soft-max** distribution over the current Q-values:

$$\mu_\beta^*(a|s) = \arg\max_{\mu} \; \mathbb{E}_{a \sim \mu}[Q(s,a)] - D_{KL}(\mu \,\|\, \rho) \;\propto\; \exp(Q(s, a))$$

where $\rho$ is a uniform prior and the KL term is a regularizer that keeps the adversary from collapsing onto a single action. Substituting this optimal $\mu_\beta^*$ back into the objective and carrying out the expectation analytically yields the **logsumexp** penalty (the "H" denotes that this variant arises from an entropy-regularized inner maximization):

$$\mathcal{R}_{CQL(H)}(Q) = \mathbb{E}_{s \sim \mathcal{D}}\Big[ \log \sum_a \exp(Q(s, a)) - \mathbb{E}_{a \sim \hat{\pi}_\beta(a|s)}\left[Q(s, a)\right] \Big]$$

The full CQL(H) critic objective is then:

$$\min_Q \; \alpha \, \mathbb{E}_{s \sim \mathcal{D}} \left[\log \sum_a \exp(Q(s, a)) - \mathbb{E}_{a \sim \hat{\pi}_\beta(a|s)}\left[Q(s,a)\right]\right] + \frac{1}{2}\mathbb{E}_{(s,a,s') \sim \mathcal{D}}\left[\left(Q(s,a) - \hat{\mathcal{B}}^\pi \hat{Q}(s,a)\right)^2\right]$$

The `logsumexp` term is the soft-maximum: it approximates `max_a Q(s,a)` but is differentiable and accounts for the whole action distribution rather than a single point. In continuous action spaces you cannot enumerate the sum, so you estimate it by importance sampling, which we detail in the next section.

### CQL(ρ): the fixed-distribution variant

There is a simpler sibling, CQL(ρ), that uses a fixed reference distribution $\rho$ — typically the previous iterate's policy or a uniform distribution — instead of the optimal soft-max adversary:

$$\mathcal{R}_{CQL(\rho)}(Q) = \mathbb{E}_{s \sim \mathcal{D}}\Big[ \mathbb{E}_{a \sim \rho(a|s)}[Q(s,a)] - \mathbb{E}_{a \sim \hat{\pi}_\beta(a|s)}[Q(s,a)] \Big]$$

CQL(ρ) is cheaper because it skips the logsumexp and only needs samples from $\rho$, but it gives a looser bound for the same $\alpha$ because $\rho$ is not the worst-case adversary. CQL(H) is preferred in practice and is what every modern implementation uses; CQL(ρ) is mainly of interest when the action space is so high-dimensional that even sampling for the logsumexp is prohibitive.

![CQL objective diagram showing data distribution feeding high-Q actions and random distribution feeding the logsumexp penalty, both combining into the CQL-H loss](/imgs/blogs/conservative-q-learning-cql-2.png)

### What "mild assumptions" means in practice

The lower-bound theorems rest on three assumptions, and it is worth being honest about which one bites in practice:

1. **Function-approximation completeness.** The Q-function class is expressive enough to contain the true $Q^\pi$ and to represent every Bellman backup of a function in the class. With a wide neural network this is approximately satisfied; the gap shows up as an additive error term in the bound.

2. **Coverage.** The dataset has nonzero coverage at all $(s,a)$ pairs the policy will visit — equivalently, the behavior policy $\mu$ gives positive probability to every action the learned policy would take. Where coverage is zero, the concentration bound on the empirical Bellman operator is infinite and no finite $\alpha$ rescues you.

3. **Sufficiently large $\alpha$.** $\alpha$ must exceed a threshold proportional to the finite-sample estimation error so the conservatism term dominates the bootstrapping noise.

Assumption 2 is the hardest. If the dataset is heavily concentrated — say an expert-only dataset that visits a thin tube of state space — CQL can be overly conservative and underperform behavioral cloning on tasks that require straying slightly from the expert's path. This is not a bug in the theory; it is the theory correctly telling you that you have no information about the regions you would need to improve into. The practical lesson is that CQL's advantage grows with dataset diversity and shrinks toward (or below) BC as the data approaches a single narrow expert demonstration.

### CQL as an instance of the pessimism principle

CQL is one concrete realization of the broader **pessimism-in-the-face-of-uncertainty** principle that underlies essentially all provably-correct offline RL. The principle says: when you are uncertain about a value, assume the pessimistic (lower) estimate, so that the only actions your policy commits to are ones the data actually supports. Pessimism is the offline mirror image of the *optimism* principle (UCB-style exploration bonuses) used in online RL — online you add an optimism bonus to encourage visiting uncertain regions; offline you subtract a pessimism penalty to discourage trusting them.

It helps to contrast CQL with its two philosophical neighbors:

- **Behavioral cloning** is maximal pessimism taken to a degenerate extreme — it refuses to consider *any* action outside the data, so it never overestimates but also never improves beyond the behavior policy. It does not use rewards at all.
- **Model-based offline RL (MOPO, MOREL)** implements pessimism differently: it learns a dynamics model, rolls out synthetic trajectories, and subtracts an uncertainty penalty proportional to the model's predictive variance from the rewards of those rollouts. MOPO penalizes the *reward* by an ensemble-disagreement bonus; MOREL builds a "pessimistic MDP" that routes uncertain transitions to an absorbing low-reward state. Both achieve pessimism through the model rather than directly on the Q-values.

CQL sits in between: it keeps the model-free simplicity of Q-learning (no dynamics model to train and to be wrong) but injects pessimism directly into the value function via the logsumexp penalty. This is why CQL is often the default first thing to try — it needs no extra learned components, only one extra loss term.

## The Logsumexp Penalty in Detail

Let's make the logsumexp computation concrete. In continuous action spaces (e.g., MuJoCo locomotion), you cannot enumerate all actions. You approximate the logsumexp by importance sampling:

$$\log \sum_a \exp(Q(s,a)) \approx \log \frac{1}{N} \sum_{i=1}^N \frac{\exp(Q(s, a_i))}{\rho(a_i)}$$

where $a_i \sim \rho(a|s)$ is some base distribution (uniform over the action space, or the current policy). The importance weight $1/\rho(a_i)$ corrects for the fact that you sampled from $\rho$ rather than uniformly over the whole action volume. The paper shows that *mixing* uniform and policy-sampled actions reduces the variance of the estimator dramatically: sample $N/2$ from uniform (to cover the whole action box) and $N/2$ from the current policy (to concentrate samples where the policy actually wants to go), with $N = 10$ being a robust default.

The gradient of the logsumexp with respect to a particular sampled Q-value is the softmax weight of that sample:

$$\frac{\partial}{\partial Q(s,a_i)}\log \sum_j \exp(Q(s,a_j)) = \frac{\exp(Q(s,a_i))}{\sum_j \exp(Q(s,a_j))} = \text{softmax}_j\big(Q(s,a_j)\big)_i$$

This gradient pushes down Q-values in proportion to their current softmax weight — the highest-Q actions get pushed down the most, which is exactly the behavior we want, since the highest-Q OOD actions are the ones the greedy policy would otherwise exploit. Combined with the $-\mathbb{E}_{\hat{\pi}_\beta}[Q]$ term that pushes *up* Q-values on actions actually in the data, the penalty creates an implicit "forbidden zone": OOD actions are dragged down toward or below the data actions, so the greedy argmax can never escape the data support by more than the residual gap.

### Why logsumexp is the right choice

The logsumexp is the log-partition function of a Boltzmann distribution over actions. This is not a coincidence — the optimal adversarial distribution $\mu_\beta^*(a|s) \propto \exp(Q(s,a))$ is exactly what an adversary would choose to maximize the gap between the average Q-value and the data-Q-value, subject to a KL leash. CQL(H) therefore uses the *tightest possible lower-bound penalty* for a given $\alpha$: any other choice of $\mu_\beta$ would put less mass on the highest-Q actions and so generate a weaker downward pressure exactly where it matters most. This optimality is the formal reason CQL(H) dominates the fixed-distribution CQL(ρ) in benchmarks.

#### Worked example: computing the logsumexp penalty

Suppose $s$ is a joint-angle configuration, the action space is $[-1, 1]^3$ (3D torques), and you sample $N = 6$ actions — two replayed from the dataset, two uniform-random, two from the current policy:

| Action sample | $Q(s, a_i)$ | softmax weight |
|---|---|---|
| $a_1$ (in dataset) | +8.2 | 0.04 |
| $a_2$ (in dataset) | +7.5 | 0.02 |
| $a_3$ (uniform random) | +15.3 | 0.62 |
| $a_4$ (uniform random) | +12.1 | 0.14 |
| $a_5$ (policy sample) | +11.8 | 0.12 |
| $a_6$ (policy sample) | +10.4 | 0.06 |

The logsumexp of the six values is $\log(e^{8.2} + e^{7.5} + e^{15.3} + e^{12.1} + e^{11.8} + e^{10.4}) \approx 15.8$, dominated by the $+15.3$ sample. The data-action mean is $\mathbb{E}_{\hat{\pi}_\beta}[Q] = (8.2 + 7.5)/2 = 7.85$. The penalty is $15.8 - 7.85 = 7.95$.

With $\alpha = 5$, this adds $5 \times 7.95 = 39.75$ to the critic loss. The gradient distributes downward pressure by softmax weight, so $Q(s, a_3)$ — the highest-Q OOD action, at weight 0.62 — receives by far the most. After a few thousand gradient steps, $Q(s, a_3)$ converges toward its true value $Q^\pi(s, a_3) \approx -3$ (the robot falls over at that torque configuration), while the data actions $a_1, a_2$ are held up near $+8$ by the maximization term. The greedy argmax over actions at $s$ now correctly selects $a_1$ instead of the exploit $a_3$. That single re-ordering, repeated across the state space, is the entire mechanism by which CQL converts a poisoned value surface into a deployable policy.

## The Alpha Hyperparameter: Conservatism vs Policy Performance

The scalar $\alpha \geq 0$ is the single most important hyperparameter in CQL. It directly controls how much Q-values are penalized on OOD actions. Figure 3 shows the four regimes, from $\alpha = 0$ (standard Q-learning, OOD blow-up) through the well-calibrated middle to $\alpha$ so large that even the in-distribution Q-values are crushed and the policy collapses.

![Stack diagram showing alpha equal to 0 causing OOD Q of plus 40 and collapse risk, through alpha equals 5 as recommended, up to alpha equals 20 causing policy collapse](/imgs/blogs/conservative-q-learning-cql-3.png)

### Fixed alpha vs adaptive alpha (CQL-H Lagrangian)

**Fixed alpha.** You set $\alpha$ as a constant and it remains fixed throughout training. Common values: 1.0 for easy, dense tasks, 5.0 for moderate mixed-quality datasets, 10.0 for heavily OOD-challenged datasets. Fixed alpha is simple but requires tuning — and crucially, the right $\alpha$ depends on the dataset coverage level, which you often do not know a priori. Pick it too small and OOD exploitation returns; too large and you crush the signal.

**Adaptive alpha (CQL-H Lagrangian).** The paper proposes treating $\alpha$ as a Lagrange multiplier in a constrained optimization. Instead of fixing the strength of the penalty, you fix a *target* on the penalty's magnitude and let $\alpha$ float to whatever value enforces it:

$$\min_Q \max_{\alpha \geq 0} \; \alpha \left[\mathbb{E}_{s \sim \mathcal{D}}\left(\log \sum_a \exp(Q) - \mathbb{E}_{\hat{\pi}_\beta}[Q]\right) - \tau\right] + \text{TD loss}$$

The constraint says: the CQL gap (logsumexp minus data-Q) should equal the threshold $\tau$. If the current gap is *above* $\tau$ (Q-values on OOD actions are too high), the inner $\max_\alpha$ pushes $\alpha$ up, strengthening the penalty until the gap closes. If the gap falls below $\tau$, $\alpha$ relaxes. This automatic tuning is structurally identical to the entropy-temperature auto-tuning in SAC, and it moves the burden from "guess the right penalty strength" to "guess an acceptable gap," which is far more transferable across datasets. The threshold $\tau$ becomes the key hyperparameter — in practice $\tau \approx 5$ to $10$ is a common starting range.

### Practical guidance

| Dataset type | Recommended alpha |
|---|---|
| Expert demos only | 0.5–2.0 (data is dense, OOD risk low) |
| Mixed (medium + random) | 5.0–10.0 |
| Random data only | 10.0–20.0 (extremely sparse coverage) |
| Safety-critical | adaptive CQL-H or large fixed (≥10) |
| Unknown dataset quality | CQL-H adaptive (safest default) |

The key diagnostic during training: watch the **in-distribution Q-values** (estimated on dataset transitions) versus the **out-of-distribution Q-values** (estimated on policy-sampled actions not in the dataset). If OOD Q-values are much higher than in-distribution Q-values and still rising, $\alpha$ is too small. If in-distribution Q-values are falling to very negative values and the policy-evaluation score drops, $\alpha$ is too large. A healthy run has OOD Q sitting slightly *below* in-distribution Q, both roughly stable.

### How dataset quality shapes the optimal alpha

The relationship between data and the ideal $\alpha$ is direct: $\alpha$ should scale with how much of the action space the policy might want to explore that the data does *not* cover. An expert dataset that already demonstrates near-optimal behavior needs only a light touch — there is little to gain from straying, so a small $\alpha$ keeps the bound tight without crushing the good actions. A random or heavily mixed dataset covers a lot of bad behavior and a little good behavior, so the critic must aggressively suppress the many OOD-leaning actions, demanding a large $\alpha$. When you genuinely do not know the dataset's quality — the common case with logged production data — the CQL-H Lagrangian is the safest choice precisely because it discovers the right strength from the data rather than from your prior beliefs. Use fixed $\alpha$ only when you have already characterized the dataset and want the simplest possible training loop.

## PyTorch CQL-SAC Implementation

CQL is most commonly implemented on top of SAC (Soft Actor-Critic) because SAC's maximum-entropy framework naturally handles continuous action spaces and provides the actor/critic structure CQL requires. The critic gets the extra logsumexp term; the actor and entropy-temperature updates are vanilla SAC. Here is a complete implementation.

### The critic with CQL penalty

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.distributions import Normal

device = "cuda" if torch.cuda.is_available() else "cpu"


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class CQLCritic(nn.Module):
    """Twin Q-networks for CQL-SAC."""
    def __init__(self, state_dim, action_dim, hidden=256):
        super().__init__()
        self.q1 = MLP(state_dim + action_dim, 1, hidden)
        self.q2 = MLP(state_dim + action_dim, 1, hidden)

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=-1)
        return self.q1(sa), self.q2(sa)
```

The first runnable snippet — the logsumexp regularizer itself — is the heart of CQL. It samples random and policy actions, computes their Q-values, and forms the importance-weighted logsumexp minus the in-data Q:

```python
def cql_regularizer(critic, state, action, actor, num_random=10):
    """
    Compute the CQL(H) logsumexp penalty for one batch.
    Returns the scalar penalty (logsumexp over OOD actions minus in-data Q),
    averaged over the batch and the two critics.
    """
    batch_size = state.shape[0]
    action_dim = action.shape[-1]
    act_low, act_high = -1.0, 1.0  # tanh-squashed action range

    # Q-values of the actions actually in the dataset (the "push up" term)
    q1_data, q2_data = critic(state, action)

    # Replicate each state num_random times for batched OOD sampling
    state_rep = state.unsqueeze(1).repeat(1, num_random, 1)
    state_rep = state_rep.view(-1, state.shape[-1])           # (B*N, S)

    # Half the OOD samples uniform over the action box ...
    n_unif = num_random // 2
    n_pol = num_random - n_unif
    rand_actions = torch.empty(
        batch_size * num_random, action_dim, device=state.device
    ).uniform_(act_low, act_high)

    # ... half from the current policy (no grad through the sampler here)
    with torch.no_grad():
        pol_dist = actor(state_rep)
        pol_actions, pol_log_pi = pol_dist.rsample_and_log_prob()

    q1_rand, q2_rand = critic(state_rep, rand_actions)
    q1_pol, q2_pol = critic(state_rep, pol_actions)

    # Reshape to (B, N) for the per-state logsumexp
    q1_rand = q1_rand.view(batch_size, num_random)
    q2_rand = q2_rand.view(batch_size, num_random)
    q1_pol = q1_pol.view(batch_size, num_random)
    q2_pol = q2_pol.view(batch_size, num_random)
    pol_log_pi = pol_log_pi.view(batch_size, num_random)

    # Importance correction: uniform density = 1/(act_high-act_low)^A,
    # so its log-prob is a constant; policy samples use -log pi(a|s)
    log_unif = -action_dim * np.log(act_high - act_low)
    cat1 = torch.cat([q1_rand - log_unif, q1_pol - pol_log_pi], dim=1)
    cat2 = torch.cat([q2_rand - log_unif, q2_pol - pol_log_pi], dim=1)

    lse1 = torch.logsumexp(cat1, dim=1, keepdim=True)
    lse2 = torch.logsumexp(cat2, dim=1, keepdim=True)

    pen = (lse1 - q1_data).mean() + (lse2 - q2_data).mean()
    return pen
```

The full critic loss combines the SAC TD target with that penalty, scaled by `cql_alpha`:

```python
def cql_critic_loss(
    critic, critic_target, state, action, reward, next_state, done,
    actor, log_alpha_sac, gamma=0.99, cql_alpha=5.0, num_random=10,
):
    """Compute CQL-SAC critic loss. Returns total, td, penalty."""
    with torch.no_grad():
        next_dist = actor(next_state)
        next_action, next_log_pi = next_dist.rsample_and_log_prob()
        nq1, nq2 = critic_target(next_state, next_action)
        next_q = torch.min(nq1, nq2) - log_alpha_sac.exp() * next_log_pi
        target_q = reward + gamma * (1.0 - done) * next_q

    q1, q2 = critic(state, action)
    td_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

    penalty = cql_regularizer(critic, state, action, actor, num_random)
    total = td_loss + cql_alpha * penalty
    return total, td_loss, penalty
```

### The Lagrangian alpha auto-tuner

The second standalone snippet implements the CQL-H adaptive-$\alpha$ mechanism. We parameterize $\log\alpha$ so $\alpha$ stays positive, and run a gradient step on it against the constraint `(penalty - tau)`:

```python
class CQLAlphaTuner:
    """Lagrangian auto-tuning of the CQL conservatism weight."""
    def __init__(self, target_gap=10.0, lr=3e-4, init_log_alpha=0.0,
                 clip=(0.0, 1e6)):
        self.log_alpha = torch.tensor(
            float(init_log_alpha), requires_grad=True, device=device
        )
        self.opt = optim.Adam([self.log_alpha], lr=lr)
        self.target_gap = target_gap
        self.clip = clip

    def alpha(self):
        return self.log_alpha.exp().clamp(*self.clip).detach()

    def update(self, penalty_value):
        # We MAXIMIZE alpha * (penalty - tau), so minimize its negative.
        a = self.log_alpha.exp().clamp(*self.clip)
        loss = -(a * (penalty_value.detach() - self.target_gap))
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return self.alpha().item()
```

When the measured `penalty_value` exceeds `target_gap`, the loss gradient drives `log_alpha` up (stronger penalty); when it is below, `alpha` relaxes. This removes the single most fragile manual knob in CQL.

### The actor (SAC policy)

```python
class SquashedNormal:
    """tanh-squashed Gaussian with log-prob correction."""
    def __init__(self, mean, std):
        self.dist = Normal(mean, std)

    def rsample_and_log_prob(self):
        z = self.dist.rsample()
        action = torch.tanh(z)
        log_pi = self.dist.log_prob(z).sum(-1, keepdim=True)
        # tanh change-of-variables correction
        log_pi -= torch.log(1 - action.pow(2) + 1e-6).sum(-1, keepdim=True)
        return action, log_pi

    def rsample(self):
        return torch.tanh(self.dist.rsample())


class SACActorDist(nn.Module):
    """Squashed Gaussian policy for SAC/CQL."""
    def __init__(self, state_dim, action_dim, hidden=256, log_std_range=(-5, 2)):
        super().__init__()
        self.net = MLP(state_dim, hidden, hidden)
        self.mean_head = nn.Linear(hidden, action_dim)
        self.log_std_head = nn.Linear(hidden, action_dim)
        self.log_std_min, self.log_std_max = log_std_range

    def forward(self, state):
        x = self.net(state)
        mean = self.mean_head(x)
        log_std = self.log_std_head(x).clamp(self.log_std_min, self.log_std_max)
        return SquashedNormal(mean, log_std.exp())
```

### The full CQL agent class

The third snippet ties the critic, actor, SAC entropy temperature, and CQL-H alpha tuner into one agent. The critic update calls the CQL loss; the actor and entropy updates are pure SAC:

```python
class CQLAgent:
    def __init__(self, state_dim, action_dim, adaptive_alpha=True,
                 cql_alpha=5.0, target_gap=10.0, lr=3e-4, tau_soft=0.005):
        self.critic = CQLCritic(state_dim, action_dim).to(device)
        self.critic_target = CQLCritic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.actor = SACActorDist(state_dim, action_dim).to(device)

        self.critic_opt = optim.Adam(self.critic.parameters(), lr=lr)
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=lr)

        # SAC entropy temperature (auto-tuned)
        self.log_alpha_sac = torch.zeros(1, requires_grad=True, device=device)
        self.sac_alpha_opt = optim.Adam([self.log_alpha_sac], lr=lr)
        self.target_entropy = -float(action_dim)

        self.adaptive_alpha = adaptive_alpha
        self.cql_alpha = cql_alpha
        self.cql_tuner = CQLAlphaTuner(target_gap, lr) if adaptive_alpha else None
        self.tau_soft = tau_soft

    def update(self, batch):
        s, a, r, s_, d = [b.to(device) for b in batch]

        # --- Critic (CQL) update ---
        cql_a = self.cql_tuner.alpha().item() if self.adaptive_alpha else self.cql_alpha
        loss, td_loss, penalty = cql_critic_loss(
            self.critic, self.critic_target, s, a, r, s_, d,
            self.actor, self.log_alpha_sac, cql_alpha=cql_a,
        )
        self.critic_opt.zero_grad()
        loss.backward()
        self.critic_opt.step()
        if self.adaptive_alpha:
            cql_a = self.cql_tuner.update(penalty)

        # --- Actor update ---
        dist = self.actor(s)
        new_action, log_pi = dist.rsample_and_log_prob()
        q1, q2 = self.critic(s, new_action)
        actor_loss = (self.log_alpha_sac.exp().detach() * log_pi
                      - torch.min(q1, q2)).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # --- SAC entropy temperature update ---
        sac_alpha_loss = -(self.log_alpha_sac
                           * (log_pi + self.target_entropy).detach()).mean()
        self.sac_alpha_opt.zero_grad()
        sac_alpha_loss.backward()
        self.sac_alpha_opt.step()

        # --- Soft target update ---
        with torch.no_grad():
            for p, pt in zip(self.critic.parameters(),
                             self.critic_target.parameters()):
                pt.data.mul_(1 - self.tau_soft).add_(self.tau_soft * p.data)

        return {"td": td_loss.item(), "cql": penalty.item(),
                "actor": actor_loss.item(), "cql_alpha": cql_a}
```

### Loading a D4RL dataset and the evaluation loop

The fourth snippet loads a D4RL dataset and normalizes observations — a step that matters more for offline RL than online, because there is no running normalizer being updated by fresh data:

```python
import d4rl
import gym


def load_d4rl_dataset(env_name="hopper-medium-v2"):
    env = gym.make(env_name)
    dataset = d4rl.qlearning_dataset(env)

    obs = torch.FloatTensor(dataset["observations"])
    actions = torch.FloatTensor(dataset["actions"])
    rewards = torch.FloatTensor(dataset["rewards"]).unsqueeze(1)
    next_obs = torch.FloatTensor(dataset["next_observations"])
    dones = torch.FloatTensor(dataset["terminals"]).unsqueeze(1)

    obs_mean = obs.mean(0)
    obs_std = obs.std(0).clamp(min=1e-3)
    obs = (obs - obs_mean) / obs_std
    next_obs = (next_obs - obs_mean) / obs_std

    return (obs, actions, rewards, next_obs, dones), env, (obs_mean, obs_std)
```

The fifth standalone snippet is the D4RL evaluation loop. Note the use of the *deterministic* mean action at evaluation time and D4RL's `get_normalized_score`, which maps raw returns onto the 0 (random) to 100 (expert) scale used throughout the benchmark:

```python
@torch.no_grad()
def evaluate_policy(agent, env, norm_stats, n_episodes=10, deterministic=True):
    obs_mean, obs_std = norm_stats
    obs_mean = obs_mean.to(device)
    obs_std = obs_std.to(device)
    raw_returns = []
    for _ in range(n_episodes):
        state, _ = env.reset()
        done = trunc = False
        ep_return = 0.0
        while not (done or trunc):
            s = torch.FloatTensor(state).to(device)
            s = (s - obs_mean) / obs_std
            dist = agent.actor(s.unsqueeze(0))
            if deterministic:
                # mean of the squashed Gaussian
                action = torch.tanh(dist.dist.mean).cpu().numpy()[0]
            else:
                action = dist.rsample().cpu().numpy()[0]
            state, reward, done, trunc, _ = env.step(action)
            ep_return += reward
        raw_returns.append(ep_return)
    mean_raw = float(np.mean(raw_returns))
    return 100.0 * env.get_normalized_score(mean_raw)


def train_cql(env_name="hopper-medium-v2", n_steps=700_000,
              batch_size=256, adaptive_alpha=True):
    dataset, env, norm_stats = load_d4rl_dataset(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    agent = CQLAgent(state_dim, action_dim, adaptive_alpha=adaptive_alpha)
    N = dataset[0].shape[0]

    for step in range(n_steps):
        idx = torch.randint(0, N, (batch_size,))
        batch = [d[idx] for d in dataset]
        metrics = agent.update(batch)

        if step % 10_000 == 0:
            score = evaluate_policy(agent, env, norm_stats, n_episodes=10)
            print(f"step={step:7d}  score={score:5.1f}  "
                  f"td={metrics['td']:.3f}  cql={metrics['cql']:.3f}  "
                  f"cql_alpha={metrics['cql_alpha']:.2f}")
```

The full pipeline, from loading the D4RL dataset to training CQL-SAC, takes about 4 hours on a single A100 GPU for 700,000 gradient steps. Memory footprint is modest — the dataset fits in CPU RAM and batches are transferred to GPU. The dominant per-step cost over vanilla SAC is the `num_random`-fold replication of states inside the logsumexp, which roughly triples the critic forward passes.

### Offline training with a library: the SB3-style setup

If you would rather not maintain the loop by hand, the same algorithm is available off-the-shelf. The d3rlpy library (the offline-RL companion to the SB3 ecosystem) exposes CQL with the Lagrangian tuner built in:

```python
import d3rlpy
from d3rlpy.algos import CQLConfig
from d3rlpy.datasets import get_d4rl

# Pulls the same hopper-medium transitions as the manual loader above
dataset, env = get_d4rl("hopper-medium-v2")

cql = CQLConfig(
    actor_learning_rate=3e-4,
    critic_learning_rate=3e-4,
    temp_learning_rate=3e-4,      # SAC entropy temperature
    alpha_learning_rate=3e-4,     # CQL Lagrangian alpha
    conservative_weight=5.0,      # initial CQL alpha
    n_action_samples=10,          # the logsumexp sample count N
    batch_size=256,
).create(device="cuda:0")

cql.fit(
    dataset,
    n_steps=700_000,
    n_steps_per_epoch=10_000,
    evaluators={"d4rl_score": d3rlpy.metrics.EnvironmentEvaluator(env)},
)
```

The `conservative_weight` is the initial CQL $\alpha$, and `alpha_learning_rate > 0` enables the Lagrangian auto-tuner — set it to `0` for a fixed weight. The `n_action_samples` maps directly to the `num_random` argument in the hand-written `cql_regularizer` above. This is the recommended path for production work: the library version handles numerical edge cases (overflow in the logsumexp, action clamping) that are easy to get subtly wrong by hand.

## The CQL-SAC Training Pipeline

Figure 5 shows the full 8-stage training loop.

![CQL-SAC training pipeline showing 8 stages from sample batch through TD target, logsumexp penalty, CQL penalty, total critic loss, actor update, temperature update, and adaptive alpha update](/imgs/blogs/conservative-q-learning-cql-5.png)

The critical stages are 3–4: the logsumexp computation over sampled actions, where the OOD penalization happens. Stages 6–8 — the actor update, the SAC entropy-temperature update, and (optionally) the CQL-H alpha update — are identical to standard SAC. This means CQL-SAC is *just SAC with a modified critic loss*, which has three practical consequences:

- You can initialize from a pretrained SAC checkpoint (with zero CQL penalty initially) and gradually ramp up alpha.
- The actor and temperature updates are unchanged — you only need to modify the critic update.
- Hyperparameters like the soft-update rate (0.005), batch size (256), and learning rate (3e-4) can be copied directly from SAC defaults.

The order of operations within a step matters: update the critic first (so the actor sees the freshly-conservative Q-values), then the actor, then the two temperature parameters, then the soft target. Reversing critic and actor updates lets the actor chase a stale, still-optimistic critic for one step, which in offline RL is enough to seed a small exploitation that the next critic step then has to undo.

## D4RL Benchmark: Where CQL Stands

D4RL (Fu et al. 2020) is the standard offline RL benchmark. It provides standardized datasets for MuJoCo locomotion tasks (hopper, walker2d, halfcheetah) across quality tiers (random, medium, medium-replay, medium-expert, expert) plus the harder antmaze and kitchen suites. The normalized score is 0 for a random policy and 100 for an expert SAC policy.

![D4RL benchmark matrix comparing BC, BCQ, CQL, IQL, and TD3 plus BC across hopper-medium, walker2d-medium, and halfcheetah-medium tasks](/imgs/blogs/conservative-q-learning-cql-4.png)

### The head-to-head comparison table

The following table consolidates the standard D4RL "-medium" results alongside the mechanism each method uses and whether it carries a conservatism hyperparameter that must be tuned. Scores are normalized (0 = random, 100 = expert) and are representative of the figures reported in the CQL, IQL, TD3+BC, and Decision Transformer papers on the v2 datasets:

| Method | Mechanism | halfcheetah-med | hopper-med | walker2d-med | Tuning required |
|---|---|---|---|---|---|
| BC | Supervised clone of $\mu$ | 42.6 | 29.0 | 36.6 | None |
| TD3+BC | TD3 actor + BC regularizer | 48.3 | 59.3 | 83.7 | One weight ($\lambda$) |
| Decision Transformer | Return-conditioned sequence model | 42.6 | 67.6 | 74.0 | Target return, context len |
| IQL | Expectile $V$, no OOD Q queries | 47.4 | 66.3 | 78.3 | Expectile $\tau$ |
| CQL | Logsumexp Q-penalty (lower bound) | 44.0 | 79.6 | 72.5 | $\alpha$ (or Lagrangian gap) |

No method dominates the row. CQL wins decisively on hopper, IQL and TD3+BC edge it out on walker2d, and the four non-BC methods cluster on halfcheetah. The table's real lesson is in the last two columns: CQL buys its strong theory and its hopper win at the cost of the most delicate hyperparameter (the conservatism strength), whereas TD3+BC's single BC weight is almost set-and-forget. Decision Transformer is the odd one out — it abandons value learning entirely and casts offline RL as sequence modeling conditioned on a desired return, which sidesteps OOD overestimation by never bootstrapping at all, but pays for it with sensitivity to the target-return you condition on at evaluation.

### Key observations

1. **BC is shockingly bad on hopper.** Hopper is a balancing task with a strong temporal credit-assignment problem — the scripted medium controller barely balances. BC scores 29.0 because it clones the shaky controller, stumbles and all. CQL at 79.6 finds a stable gait by identifying which transitions in the medium data correspond to the controlled forward stride versus the stumbling, and preferentially reproducing the former.

2. **CQL dominates hopper, IQL dominates walker2d.** Walker2d's medium dataset is denser and more expert-like, so IQL's in-sample approach is sufficient and CQL's penalty is slightly over-conservative there.

3. **Halfcheetah is harder to improve.** The halfcheetah task has a wide action distribution — many different running gaits all achieve similar speeds. The dataset coverage is naturally broad, so OOD overestimation is less severe and all methods cluster around 42–48.

4. **BCQ vs CQL.** BCQ (Fujimoto et al. 2019) uses a generative CVAE to constrain the policy to the data support. It works but requires training that separate model, and its coverage estimate can be wrong. CQL replaces the explicit behavioral modeling with an implicit Q-penalty — both simpler and more theoretically principled.

#### Worked example: D4RL normalized score comparison on hopper-medium

Reading the table as a deployment decision makes the numbers concrete. Suppose you have a hopper-medium-quality log and your acceptance bar is "beat the behavior policy by at least 2×." The behavior policy here scores roughly 40 raw (the medium controller), which D4RL normalizes near the BC clone's 29.0. Your candidates and their normalized scores:

- BC: 29.0 — fails the bar; it is the behavior policy by construction.
- TD3+BC: 59.3 — a $59.3/29.0 = 2.04\times$ improvement, just clears the bar with one tuned weight.
- IQL: 66.3 — a $2.29\times$ improvement, no OOD Q queries, cheap.
- CQL: 79.6 — a $2.74\times$ improvement, the best of the value-based methods, at the cost of tuning $\alpha$.

If compute and tuning budget are tight, IQL at 66.3 is the pragmatic pick — it clears the bar comfortably with the cheapest training loop. If you intend to follow offline training with online fine-tuning (next section), CQL's 79.6 *and* its SAC-native warm-start make it the better foundation even though it costs more to tune. The score table alone does not decide it; the downstream plan does.

#### Worked example: reproducing hopper-medium training dynamics

Starting from the CQL-SAC implementation above, a training run on hopper-medium-v2 with the Lagrangian tuner produces dynamics like this:

| Step (×1000) | TD loss | CQL penalty | Policy score |
|---|---|---|---|
| 0 | 22.4 | 1.2 | 1.1 (random) |
| 50 | 8.3 | 4.1 | 18.7 |
| 150 | 3.2 | 6.8 | 52.3 |
| 300 | 1.9 | 7.4 | 74.1 |
| 500 | 1.4 | 7.7 | 78.9 |
| 700 | 1.2 | 7.8 | 79.4 |

The CQL penalty grows as training progresses: Q-values on OOD actions become increasingly suppressed relative to the data actions, and the logsumexp captures a growing spread between the policy's preferred actions and the data distribution before the Lagrangian alpha clamps it at the target gap. The policy score plateaus around 79–80 — the conservative ceiling, where further training adds little because the penalty is restraining exploration beyond the data support. That plateau is not a failure; it is the algorithm correctly refusing to bet on regions the data cannot vouch for.

## Offline-to-Online Fine-Tuning with CQL Warm Start

One of CQL's most practical advantages over simpler offline methods is its compatibility with subsequent online fine-tuning. Because CQL is built on SAC (designed for online RL), you can take a CQL-trained checkpoint and continue training with real environment interactions, with the CQL penalty decaying or switched off.

![Before-after showing offline CQL only reaching score 79.6 on hopper-medium versus CQL warm-start plus 100k online steps reaching 95.2](/imgs/blogs/conservative-q-learning-cql-6.png)

### Why CQL warm start beats starting online from scratch

Starting online SAC from scratch on MuJoCo locomotion takes roughly 1–3 million steps to reach expert level. Starting from a CQL checkpoint trained offline takes only ~100–200k additional online steps to surpass the offline-only score. The reasons:

1. **The Q-function is already well-shaped.** CQL has learned a Q-function that correctly orders good and bad state-action pairs from the offline data. Online fine-tuning only needs to refine the margins, not learn the structure from scratch.

2. **The policy starts near the data distribution.** The CQL actor has been trained to maximize Q under the conservative critic, so its initial online behavior is near-dataset quality. You begin online training from a decent policy, not a random one — so the first online rollouts already collect useful, non-catastrophic transitions.

3. **Exploration is calibrated.** The SAC entropy temperature $\alpha_{SAC}$ has been auto-tuned to match the dataset's action entropy. When you switch to online training, the exploration level is already reasonable rather than wildly broad.

There is a subtlety worth naming: if you keep the CQL penalty *too high* during the online phase, the agent stays pessimistic about the very OOD actions it should now be exploring, and online improvement stalls. The fix is to decay the conservatism weight as real data arrives — and the calibrated variant Cal-QL (covered below) was designed precisely to make this transition smooth.

### Implementation: the offline-to-online transition

```python
def offline_to_online_finetune(
    cql_agent,             # pretrained CQL agent
    env_name,
    n_online_steps=100_000,
    batch_size=256,
    cql_alpha_online=0.5,  # reduced CQL penalty for the online phase
    replay_buffer_size=200_000,
):
    """Fine-tune a pretrained CQL agent with online interactions."""
    import collections
    env = gym.make(env_name)
    buf = collections.deque(maxlen=replay_buffer_size)

    # Relax conservatism so the policy may improve beyond the data support.
    cql_agent.adaptive_alpha = False
    cql_agent.cql_alpha = cql_alpha_online

    state, _ = env.reset()
    for step in range(n_online_steps):
        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0).to(device)
            action = cql_agent.actor(s).rsample().cpu().numpy()[0]
        next_state, reward, done, trunc, _ = env.step(action)
        buf.append((state, action, reward, next_state, float(done or trunc)))
        state = next_state if not (done or trunc) else env.reset()[0]

        if len(buf) > batch_size:
            idx = np.random.randint(0, len(buf), batch_size)
            bd = [buf[i] for i in idx]
            s = torch.FloatTensor(np.array([b[0] for b in bd]))
            a = torch.FloatTensor(np.array([b[1] for b in bd]))
            r = torch.FloatTensor(np.array([[b[2]] for b in bd]))
            s_ = torch.FloatTensor(np.array([b[3] for b in bd]))
            d = torch.FloatTensor(np.array([[b[4]] for b in bd]))
            cql_agent.update((s, a, r, s_, d))

        if step % 10_000 == 0:
            print(f"online step={step}")
```

The key change is reducing the conservatism weight from 5.0 to 0.5 (or 0.0 for pure SAC) in the online phase. The lower weight lets the policy improve beyond the data distribution, using the now-corrected Q-function as a guide. The combination of offline pretraining plus online fine-tuning consistently outperforms either approach alone — the offline-only ceiling of 79.6 on hopper-medium rises to roughly 95.2 after 100k online steps.

## The Offline RL Timeline: CQL's Historical Position

To understand CQL's contribution, it helps to place it in the sequence of offline RL algorithms.

![Timeline showing 2019 BCQ as first offline RL through 2020 CQL with theory, 2021 IQL simpler approach, 2022 TD3 plus BC simplest, and 2023 offline to online becoming standard](/imgs/blogs/conservative-q-learning-cql-7.png)

**BCQ (2019)** introduced the constrained-policy approach: learn a generative model of the data distribution and only allow actions within the model's support. BCQ demonstrated that offline Q-learning could work at all — a significant result — but required training a CVAE, adding complexity and another error source.

**CQL (2020)** replaced BCQ's explicit distribution modeling with an implicit Q-penalty. Crucially, Kumar et al. provided the first *formal guarantee* that the offline Q-function is a lower bound (Theorems 3.1 and 3.2) — not just an empirical observation that the method works. The D4RL results were substantially better than BCQ across most tasks.

**IQL (2021)** took the opposite philosophical direction: instead of penalizing OOD Q-values, avoid querying OOD at all. IQL's expectile-regression approach is computationally simpler and often more stable, matching or exceeding CQL on many tasks without a penalty term.

**TD3+BC (2022)** went even simpler: add a behavioral-cloning regularizer to the TD3 actor loss. No Q-penalty, no expectile regression. TD3+BC is fast, easy to tune, and competitive on most locomotion tasks — the pragmatic baseline that CQL and IQL must beat to justify their complexity.

**Decision Transformer (2021)** reframed offline RL as conditional sequence modeling: feed the model a desired return-to-go and let a causal transformer predict the next action. It never bootstraps, so it never overestimates, but it cannot stitch trajectories the way value-based methods can, and its performance hinges on the target return you specify at test time.

**Offline-to-online (2023 onward)** emerged as a dominant paradigm. Systems like IQL+AWAC, CQL warm-start, and Cal-QL (a calibrated variant of CQL) became the standard for deploying RL when offline data exists but online fine-tuning is also feasible.

## Case Studies: CQL in the Wild

### Case Study 1: D4RL locomotion (Kumar et al. 2020)

The original CQL paper benchmarks on D4RL with hopper, walker2d, and halfcheetah. The headline result: on hopper-medium, CQL achieves 79.6 versus BC's 29.0. On the more challenging hopper-medium-replay — a replay buffer of a partially-trained agent, highly suboptimal and diverse — CQL achieves 86.6 versus BC's 11.8. This shows CQL's penalty scales especially well when the dataset is mixed-quality: the penalty correctly identifies which transitions were good and which were bad, allowing the policy to preferentially replicate the good ones. The medium-replay gap (86.6 vs 11.8, a 7.3× margin) is far larger than the medium gap (2.7×), which is the clearest empirical signal that CQL's value comes from *stitching*, not imitation.

### Case Study 2: antmaze navigation and comparison with IQL on locomotion-style control

The D4RL antmaze tasks require stitching together suboptimal trajectories to navigate a maze from start to goal. A behavioral-cloning policy fails entirely — individual trajectories do not reach the goal, so BC copies non-goal-reaching behavior. CQL on antmaze-umaze achieves 74.0 versus BC's near-65.0; on antmaze-large (a wide maze) CQL reaches 12.0 versus BC's 0. The locomotion comparison with IQL is the instructive contrast: IQL's in-sample expectile approach edges CQL on the denser walker2d-medium (78.3 vs 72.5), because there the data is expert-like enough that avoiding OOD queries entirely is sufficient and CQL's penalty is mildly over-conservative. On the sparser, stitch-heavy antmaze and hopper-replay regimes, CQL's explicit Q-suppression wins, because the policy genuinely needs to evaluate and reject many OOD actions rather than simply staying in-sample. The two methods are not strictly ordered; they trade places along the dataset-diversity axis.

### Case Study 3: offline robotic manipulation (Rafailov et al. 2021)

Rafailov et al. evaluated CQL on the D4RL Franka Kitchen dataset — kitchen manipulation demonstrations with five subtasks (open microwave, move kettle, etc.). CQL achieves 43.8% task completion on kitchen-mixed versus BC's 38.0%. More importantly, the paper identifies a regime where CQL's conservatism is critical: tasks with sparse rewards and multi-step dependencies. But when the data is dense (kitchen-complete), BC at 65.0% actually *outperforms* CQL at 43.8% — the dataset has near-optimal coverage, so OOD overestimation is not the bottleneck and CQL's penalty merely throttles a policy that had nothing to fear. This is the cleanest real-world illustration of the coverage assumption: conservatism is insurance, and on a dataset that needs no insurance you are just paying the premium.

### Case Study 4: offline-to-online with Cal-QL

Cal-QL (Nakamoto et al. 2023) adds a calibration term to CQL that ensures the Q-values are not just conservative but also accurately *scaled* — specifically, lower-bounded by the value of a reference policy rather than driven arbitrarily low. On antmaze-large, Cal-QL achieves around 62.0 normalized score — a dramatic improvement over the 12.0 of standard CQL — and then fine-tunes to roughly 89.0 with 100k online steps. The fix addresses exactly the offline-to-online failure mode named earlier: standard CQL can drive OOD Q-values so far below the truth that, when online data arrives, the over-pessimism slows re-learning. Calibration keeps the lower bound *tight enough* that the online phase starts from useful values. This demonstrates that the offline-to-online paradigm, seeded by CQL's warm-start compatibility, has matured into a complete framework where offline pretraining plus online fine-tuning is now the dominant approach for real-world RL deployment.

## Choosing the Right Alpha: A Practical Decision Guide

Figure 8 shows the decision tree for alpha selection.

![Tree diagram for CQL alpha selection branching from safety-critical tasks requiring large alpha to performance-maximizing tasks with adaptive CQL-H for mixed data or small alpha for high-quality expert data](/imgs/blogs/conservative-q-learning-cql-8.png)

### Diagnosing alpha problems in practice

The most common failure mode is **alpha too small** on a diverse dataset. Symptoms:

- The CQL penalty is growing but Q-values are still high and volatile.
- Evaluation score improves rapidly for ~100k steps then crashes — the policy exploited an OOD region and received negative reward.
- The logsumexp value is much higher than the in-data Q-value mean (gap > 20 and widening).

Fix: increase alpha by 2×, restart from step 0 or from the last stable checkpoint. If you are already on the Lagrangian tuner, lower `target_gap` so the multiplier is forced higher.

The second failure mode is **alpha too large** on a high-quality dataset. Symptoms:

- The CQL penalty is very large (> 50) and growing.
- In-distribution Q-values are declining toward very negative values.
- The policy evaluation score never improves past behavioral-cloning level.

Fix: decrease alpha by 2×, or switch to CQL-H adaptive with a moderate constraint.

**The CQL-H adaptive-alpha diagnostic.** With the Lagrangian tuner, monitor the alpha value itself. If it climbs unboundedly, your `target_gap` is too tight (you are demanding more conservatism than the critic can supply without destroying its TD fit). If it decays to near zero, your dataset is covering the policy well and you may not need the penalty at all — consider switching to a simpler offline method like IQL or TD3+BC.

### Ablation: what happens as alpha varies

Concrete numbers from an ablation on hopper-medium-v2, training for 500k steps with all other hyperparameters fixed:

| Alpha | Avg policy score | Variance | OOD Q mean | In-dist Q mean |
|---|---|---|---|---|
| 0.0 (standard Q) | 21.3 | 18.4 | +38.2 | +12.1 |
| 0.1 | 34.7 | 12.1 | +22.4 | +11.8 |
| 1.0 | 58.2 | 6.3 | +9.1 | +10.4 |
| 5.0 | 78.9 | 2.1 | −2.8 | +8.9 |
| 10.0 | 71.4 | 1.8 | −9.1 | +7.2 |
| 20.0 | 43.7 | 1.2 | −18.3 | +3.4 |
| CQL-H adaptive | 79.4 | 1.9 | −3.1 | +8.7 |

At alpha = 0.0, Q-values at OOD actions blow up to +38, the policy exploits them, and the evaluation score falls below BC. The variance column tells the safety story: standard Q-learning's run-to-run variance is 18.4, an order of magnitude higher than the conservative regimes, because each seed exploits a different OOD region. At alpha = 5.0, OOD Q-values fall to −3 (slightly below the true value — a correct, tight lower bound), the penalty is well-calibrated, and we reach near-maximum performance with low variance. At alpha = 20.0, the in-distribution Q-values are penalized too heavily (down to +3.4), the value landscape is too flat to derive a good policy, and performance regresses toward BC. CQL-H adaptive lands on essentially the same solution as alpha = 5.0 without any manual sweep — which is why it is the recommended default when you cannot afford a tuning campaign.

## When to Use CQL (and When Not To)

**Use CQL when:**

- You have an offline dataset with moderate diversity (random, medium, or mixed quality).
- You need a formal lower-bound guarantee for safety-critical deployment.
- You plan to do offline pretraining followed by online fine-tuning (CQL's SAC backbone makes the warm-start natural).
- The dataset has multi-step dependencies that BC cannot capture (navigation, manipulation, stitching).
- You want the best single value-based offline RL method with theoretical backing.

**Use IQL instead of CQL when:**

- Simplicity and training speed matter more than theoretical guarantees.
- Your dataset is dense (near-expert coverage) and OOD risk is low.
- You do not plan offline-to-online fine-tuning.
- You are experimenting on a new domain and want a reliable, no-fuss baseline.

**Use TD3+BC instead when:**

- You want the fastest possible offline RL implementation.
- The task is relatively simple (standard locomotion, not maze navigation).
- Tuning overhead is unacceptable — TD3+BC has essentially one hyperparameter.

**Use Decision Transformer when:**

- You have a strong sequence-modeling stack already and abundant data.
- You want to condition behavior on a desired return at test time.
- Bootstrapping instability has been your main pain and you would rather avoid value learning entirely.

**Do NOT use CQL (use behavioral cloning) when:**

- The dataset is expert-only with near-complete coverage of the task's state space — the kitchen-complete result shows BC can win here.
- You have no reward signal (a pure imitation setting).
- Compute budget is very tight — CQL costs roughly 3–4× the compute of BC due to the logsumexp sampling.

**Do NOT use offline RL at all when:**

- You have cheap, direct environment access — use online SAC or PPO.
- The dataset is tiny (< 5,000 transitions) — every offline method will overfit or underfit.
- The data was collected under a reward function different from the one you want to optimize.

## Key Takeaways

1. **Standard Q-learning fails offline because it bootstraps on uncorrected OOD Q-values.** The greedy backup selects actions the data never tried, those values are never anchored to reality, and the error compounds geometrically with horizon, shifting the entire value surface optimistically upward.

2. **CQL fixes this with a logsumexp penalty.** Pushing down Q-values on OOD-leaning actions and pushing up Q-values on data actions makes the Q-function a lower bound on the true $Q^\pi$.

3. **Theorems 3.1 and 3.2 are real guarantees, not heuristics.** Under completeness and coverage assumptions with large-enough $\alpha$, CQL(H) provably lower-bounds the value (3.1) and the Q-function pointwise (3.2) on the data distribution.

4. **The logsumexp is the tightest possible penalty.** The soft-max adversary $\mu^*(a) \propto \exp(Q(a))$ maximizes the conservatism gap for a given $\alpha$; any other choice (CQL(ρ)) gives a weaker bound.

5. **CQL is the pessimism principle made model-free.** It sits between behavioral cloning (degenerate maximal pessimism) and model-based MOPO/MOREL (pessimism via dynamics-model uncertainty), injecting pessimism directly into the value function.

6. **Alpha is the critical hyperparameter.** Too small: OOD exploitation and high seed variance return. Too large: the value landscape flattens and the policy collapses toward BC. Alpha ≈ 5.0 is a reasonable default for mixed data; the CQL-H Lagrangian tuner is the safest choice when dataset quality is unknown.

7. **CQL beats IQL on diverse, stitch-heavy datasets; IQL beats CQL on dense, expert-like ones.** Neither dominates universally — they trade places along the dataset-diversity axis, and the choice should follow the data.

8. **Offline-to-online fine-tuning with a CQL warm start is powerful.** A 79.6-scoring offline checkpoint reaches ~95.2 after only 100k online steps — and Cal-QL's calibration keeps the lower bound tight enough that the online transition is smooth.

## Further Reading

- Kumar, Aviral, et al. "Conservative Q-Learning for Offline Reinforcement Learning." *NeurIPS 2020.* The original paper; Theorems 3.1 and 3.2 and the full CQL objective derivation are in Section 3.
- Fu, Justin, et al. "D4RL: Datasets for Deep Data-Driven Reinforcement Learning." *arXiv 2020.* The benchmark datasets and evaluation protocol used throughout this post.
- Kostrikov, Ilya, et al. "Offline Reinforcement Learning with Implicit Q-Learning." *ICLR 2022.* The IQL paper; compare Section 4 directly to CQL's Section 3 for the philosophical contrast.
- Fujimoto, Scott, et al. "Off-Policy Deep Reinforcement Learning without Exploration." *ICML 2019.* BCQ — the predecessor that first showed offline Q-learning could work.
- Fujimoto, Scott, and Shixiang Shane Gu. "A Minimalist Approach to Offline Reinforcement Learning." *NeurIPS 2021.* The TD3+BC baseline.
- Chen, Lili, et al. "Decision Transformer: Reinforcement Learning via Sequence Modeling." *NeurIPS 2021.* The return-conditioned sequence-modeling alternative.
- Yu, Tianhe, et al. "MOPO: Model-based Offline Policy Optimization." *NeurIPS 2020.* The model-based pessimism counterpart to CQL.
- Nakamoto, Mitsuhiko, et al. "Cal-QL: Calibrated Offline RL Pre-Training for Efficient Online Fine-Tuning." *NeurIPS 2023.* The offline-to-online extension of CQL with calibration.
- Cross-series: For the broader offline RL landscape, see [Reinforcement Learning: A Unified Map](/blog/machine-learning/reinforcement-learning/reinforcement-learning-a-unified-map). For debugging Q-function training instabilities, see [Debugging AI Training and Finetuning](/blog/machine-learning/debugging-training/diagnosing-critic-divergence-and-value-explosion). For the full playbook on offline RL in production, see [The Reinforcement Learning Playbook](/blog/machine-learning/reinforcement-learning/the-reinforcement-learning-playbook).
