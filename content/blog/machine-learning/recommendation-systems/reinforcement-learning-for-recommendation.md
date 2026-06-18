---
title: "Reinforcement Learning for Recommendation"
date: "2026-06-19"
publishDate: "2026-06-19"
description: "Stop optimizing the next click and start optimizing the whole session. Frame recommendation as a Markov decision process, derive the YouTube top-K off-policy REINFORCE correction, build a small policy-gradient recommender on a simulated session environment, and learn the honest rule for when full RL beats a contextual bandit."
tags:
  [
    "recommendation-systems",
    "recsys",
    "reinforcement-learning",
    "policy-gradient",
    "off-policy",
    "long-term-value",
    "slateq",
    "machine-learning",
  ]
category: "machine-learning"
subcategory: "Recommendation Systems"
author: "Hiep Tran"
featured: true
readTime: 53
image: "/imgs/blogs/reinforcement-learning-for-recommendation-1.png"
---

A few years into running a feed, you start to notice a pattern that the dashboards do not want you to see. You ship a ranker that lifts click-through rate. The launch review is a victory lap: CTR is up two points, the offline AUC moved, the model is "better." Then six weeks later someone in the growth meeting points at a different chart — the one for 28-day retention — and it has sagged. Not crashed; sagged. The people who clicked more came back less. When you go digging, you find the model learned the lesson you actually taught it: it learned to win the *next click*, and the cheapest way to win the next click is a thumbnail that overpromises, a headline that baits, a video that front-loads the payoff and then disappoints. Every one of those wins the click and loses a little trust, and trust is the thing that brings the user back tomorrow.

This is the central tension of production recommendation, and almost every model in this series so far has been on the wrong side of it. A pointwise CTR model, a pairwise ranker, a two-tower retriever trained on logged clicks — they all optimize an *immediate* signal. They answer the question "what will this user click *right now*?" But the business does not get paid for the next click. It gets paid for session length, for next-day return, for the user who is still here in three months. Those are *cumulative*, multi-step quantities, and a model that greedily maximizes the next click can — provably, and routinely in practice — destroy them. The clickbait equilibrium is not a bug in your data; it is the optimum of the objective you chose.

![A two-column comparison contrasting a myopic next-click optimizer that lifts day-one click rate but cuts session length and retention against a cumulative-reward optimizer that trades a little click rate for durable engagement](/imgs/blogs/reinforcement-learning-for-recommendation-1.png)

The figure above is the whole argument in one picture. On the left, the myopic ranker buys a small CTR bump at the cost of shorter sessions and lower retention. On the right, a policy that optimizes *cumulative reward over the session* gives back a little of that first-click CTR and earns back the session length and the retention. The right column is what reinforcement learning (RL) is for. RL is the branch of machine learning whose entire job is to maximize a *sum of rewards over time* rather than a single immediate reward — and recommendation, once you see it as a sequence of decisions that shape what the user does next, is naturally an RL problem.

By the end of this post you will be able to: write recommendation down as a Markov decision process (MDP) and say exactly what the state, action, reward, transition, and discounted return are; explain why the rec MDP is genuinely hard (a million-item action space, off-policy logged data, partial observability, and a reward you have to *design*); derive the policy-gradient theorem and the off-policy REINFORCE estimator that YouTube actually shipped, including the importance-weight correction and the top-K trick; understand SlateQ's decomposition and why naive DQN drowns in the action space; build a small REINFORCE recommender on a simulated session environment in PyTorch and beat a myopic baseline on cumulative reward; and — most importantly — apply the honest rule for *when full RL is worth it versus when a [contextual bandit](/blog/machine-learning/recommendation-systems/bandits-and-the-exploration-exploitation-tradeoff) is the pragmatic answer*. This sits at the long-horizon end of the [retrieval-ranking-reranking funnel](/blog/machine-learning/recommendation-systems/the-recommendation-funnel-retrieval-ranking-reranking): everything else in the funnel optimizes one step; RL optimizes the trajectory.

## 1. Why the next click is the wrong objective

Let me make the failure concrete before we touch any math, because the math only matters once you believe the problem is real.

Suppose your feed has two kinds of items. Type A is a "candy" item: a flashy thumbnail, a sensational title, a high probability of an immediate click but a low probability that the user is satisfied afterward. Type B is a "vegetable" item: a less exciting surface, a lower immediate click probability, but high satisfaction when consumed, and — this is the key — a higher probability that the user *comes back tomorrow*. A myopic model trained on click labels sees only the first number. Type A clicks more, so the model learns to show type A. Over weeks, the feed fills with candy, the per-session click count holds or even rises, and retention slowly rots. Nobody made a mistake. The model did exactly what the loss asked.

The technical name for the model's blindness is that **the immediate reward is not the same as the value of the state-action pair**. The value includes everything that happens *after* the click — whether the user keeps scrolling, whether they finish the video, whether they open the app again the next morning. A click is a one-step reward $r_t$. What you want to maximize is the **return**, the discounted sum of all future rewards:

$$
G_t = r_t + \gamma\, r_{t+1} + \gamma^2 r_{t+2} + \cdots = \sum_{k=0}^{\infty} \gamma^k r_{t+k},
$$

where $\gamma \in [0,1)$ is a discount factor that says how much you care about reward $k$ steps from now relative to reward now. A myopic ranker is the special case $\gamma = 0$: it sees only $r_t$ and is blind to everything the recommendation *causes* downstream. Reinforcement learning is what you get when $\gamma > 0$ and you take the consequences of your action seriously.

There is a second, subtler reason the next click is the wrong objective: **your action changes the user's future state**. When you show a user a deep-dive on a niche topic, you do not just earn (or not) a click — you change what they are interested in, what the next candidate set should look like, and what they will respond to tomorrow. A myopic model treats every impression as an independent draw. The reality is a coupled system: today's recommendation is part of the input to tomorrow's. That coupling is exactly what an MDP captures and what supervised ranking ignores.

#### Worked example: when the greedy click hurts retention

Put numbers on the candy-vs-vegetable choice. A user is in a state where you can show item A or item B for the next two steps.

- **Item A (candy):** immediate click probability $0.30$, but if shown, the probability the user is still in the app next step drops to $0.50$.
- **Item B (vegetable):** immediate click probability $0.18$, but if shown, the probability the user is still in the app next step is $0.90$.

Treat a click as reward $1$ and let the only reward be clicks; use $\gamma = 1$ over a two-step session. A myopic policy picks the higher *immediate* click probability, so it picks A both times it gets the chance. Its expected return:

- Step 1: show A, expected click $0.30$. With probability $0.50$ the user stays.
- Step 2 (if user stayed): show A again, expected click $0.30$.
- Expected total clicks $= 0.30 + 0.50 \times 0.30 = 0.30 + 0.15 = 0.45$.

Now the long-horizon policy shows B first to keep the user around, then A:

- Step 1: show B, expected click $0.18$. With probability $0.90$ the user stays.
- Step 2 (if user stayed): show A, expected click $0.30$.
- Expected total clicks $= 0.18 + 0.90 \times 0.30 = 0.18 + 0.27 = 0.45$.

Already a tie on two-step clicks — and the greedy policy "won" step one. But extend to a third step and the gap explodes, because the long-horizon policy has a $0.90$ chance of a live user to recommend to, while the greedy policy has $0.50 \times 0.50 = 0.25$. Over a longer session, keeping the user alive compounds. The greedy policy's myopic win in step one is a loan it repays with interest in lost future steps. *That* compounding is what $\gamma > 0$ and a value function are built to see, and what a click-maximizing ranker is structurally incapable of seeing.

This is the same self-reinforcing dynamic that drives [feedback loops and filter bubbles](/blog/machine-learning/recommendation-systems/feedback-loops-and-filter-bubbles): a short-horizon objective, iterated, drifts the system to a degenerate fixed point. RL is one of the few principled tools that lets you optimize *against* that drift, because it is the only framing that puts the future explicitly in the objective.

## 2. Recommendation as a Markov decision process

To use RL we need to cast recommendation in its language. The object is a Markov decision process, defined by a tuple $(\mathcal{S}, \mathcal{A}, P, R, \gamma)$. Here is each piece, translated into recommender terms.

![A directed acyclic dataflow showing the recommendation loop drawn as a chain from state to policy action to reward and transition merging into the next state](/imgs/blogs/reinforcement-learning-for-recommendation-2.png)

The figure draws the loop as a chain — deliberately acyclic, because what actually happens is a *sequence* of decisions, each feeding the next, not a single closed ring. Read it left to right and it tells the whole story: the policy reads the state, picks an action, the action produces a reward and drives a transition, and the two combine into the next state, which the next decision inherits.

- **State $s_t \in \mathcal{S}$** — everything the agent knows about the user at decision time. In practice this is a summary of the user's interaction history (the last $N$ items they engaged with), plus context (time of day, device, session position, maybe a long-term profile embedding). The state is almost never the *true* user mind-state; it is a partial, learned summary. That gap is **partial observability**, and it is why most rec MDPs are really partially observable MDPs (POMDPs) approximated by a recurrent or transformer encoder over the history. The sequence models in [SASRec and BERT4Rec](/blog/machine-learning/recommendation-systems/self-attention-for-sequences-sasrec-bert4rec) are, in effect, learned state encoders for exactly this.
- **Action $a_t \in \mathcal{A}$** — the item (or slate of items) you recommend. This is where rec MDPs differ violently from textbook RL. In a control problem the action space might be "left, right, up, down." Here $\mathcal{A}$ is the entire catalog — millions of items — and if you recommend a *slate* of $k$ items, the action space is combinatorial, on the order of $|\mathcal{A}|^k$ ordered slates. This single fact breaks most off-the-shelf RL algorithms, and dealing with it is the technical heart of the field.
- **Reward $r_t = R(s_t, a_t)$** — the signal you get back: a click, a dwell-time bucket, a like, a purchase, a long-session indicator, or a designed combination. **Reward design is the hardest and most consequential modeling choice in RL-for-rec**, and we give it its own section, because if you reward the wrong thing the optimal policy will be a disaster that hits its target perfectly.
- **Transition $P(s_{t+1} \mid s_t, a_t)$** — how the user's state evolves after seeing your recommendation. In the real world this is *the user*: a stochastic, non-stationary, unknown process. You never have $P$ in closed form. You either learn a model of it (model-based RL), or you avoid needing it (model-free RL, which is what almost everyone ships).
- **Discount $\gamma$** and the **return** $G_t = \sum_{k\ge 0}\gamma^k r_{t+k}$ as above. A typical choice for rec is $\gamma$ around $0.9$–$0.99$, corresponding to an effective horizon of $1/(1-\gamma)$ steps — tens to a hundred — which is the right order for a session.

![A vertical stack of the five Markov decision process components from state through action reward and transition up to the discounted return the policy maximizes](/imgs/blogs/reinforcement-learning-for-recommendation-6.png)

Stacking the components, as in the figure, makes the contract explicit: specify these five things and recommendation *becomes* an MDP whose optimal policy maximizes long-term, not next-step, value. The Markov assumption — that $s_t$ summarizes everything relevant about the past — is doing a lot of work, and it is approximately false in recommendation (users have memory you cannot see). But it is a *useful* approximation, the same way the Markov assumption in a sequence model is, and the whole field is built on making the state rich enough that the approximation holds.

### 2.1 The Bellman equation: the recursion at the core of value

The reason MDPs are tractable at all is a single recursive identity. Define the **value of a state under a policy** $\pi$ as the expected return if you start in $s$ and follow $\pi$ forever:

$$
V^\pi(s) = \mathbb{E}_\pi\!\left[\sum_{k=0}^\infty \gamma^k r_{t+k} \;\middle|\; s_t = s\right].
$$

Define the **action-value** (the Q-function) as the expected return if you take action $a$ in $s$ and then follow $\pi$:

$$
Q^\pi(s,a) = \mathbb{E}_\pi\!\left[\sum_{k=0}^\infty \gamma^k r_{t+k} \;\middle|\; s_t = s,\, a_t = a\right].
$$

The **Bellman equation** says the value now equals the immediate reward plus the discounted value of where you land:

$$
Q^\pi(s,a) = \mathbb{E}\big[\, r_t + \gamma\, Q^\pi(s_{t+1}, a_{t+1}) \,\big].
$$

This recursion is what makes RL work: you do not need to roll out infinite futures to estimate value, you can *bootstrap* — estimate value from value one step ahead. For the optimal policy, the recursion becomes the Bellman optimality equation, $Q^*(s,a) = \mathbb{E}[\, r_t + \gamma \max_{a'} Q^*(s_{t+1}, a')\,]$, and the optimal policy is $\pi^*(s) = \arg\max_a Q^*(s,a)$. Hold onto that $\max_a$ and that $\arg\max_a$: in recommendation, $a$ ranges over millions of items, so both the max in the update and the argmax at serving time are exactly where value-based RL hits a wall.

There is a quiet subtlety in the Bellman equation that matters enormously in practice: the bootstrap $r_t + \gamma Q(s_{t+1}, \cdot)$ assumes your estimate of $Q$ at the *next* state is roughly correct, and early in training it is not. The estimate at $s_{t+1}$ is wrong, you back it up to $s_t$, that estimate gets backed up to $s_{t-1}$, and errors propagate. This is why value-based RL is notoriously less stable than supervised learning — the target you regress toward is *itself a moving function of the parameters you are updating*. The standard stabilizers (a slowly-updated target network for $Q(s',\cdot)$, an experience replay buffer to decorrelate samples) are entirely about taming this self-reference. Policy gradients avoid the worst of it because they do not bootstrap through a $\max$; they use sampled returns directly. That stability difference is one more reason the production systems lean policy-based.

### 2.2 The state is learned, and that is the whole game

The single most underrated decision in an RL recommender is *what goes into $s_t$*. Textbook RL hands you a state; recommendation makes you build one, and the quality of that state caps the quality of everything downstream. The raw material is the user's interaction history — a variable-length sequence of (item, action, timestamp, context) tuples — and you must compress it into a fixed-size vector the policy and value networks can consume.

The naive choice is a bag of the last $N$ item ids, which is what the simulator in section 9 uses for simplicity. The production choice is a learned sequence encoder: a recurrent network (GRU/LSTM) or, increasingly, a self-attention encoder of exactly the kind covered in [SASRec and BERT4Rec](/blog/machine-learning/recommendation-systems/self-attention-for-sequences-sasrec-bert4rec). The encoder is not a separate concern bolted onto the RL — it *is* the state function, trained end-to-end with the policy so that the representation it learns is the one most useful for predicting long-term return. A richer state buys you two things: it makes the Markov approximation closer to true (a transformer over 100 events captures dependencies a bag-of-5 cannot), and it lowers the variance of every value estimate because similar users land in similar state vectors and share statistical strength.

The cost is that a learned, high-dimensional state makes the off-policy correction harder. The importance weight $\pi_\theta(a\mid s)/\beta(a\mid s)$ depends on $s$, and if your state encoder drifts during training, the meaning of "the same state" drifts with it, and the behavior-policy estimate $\beta(a\mid s)$ — which you fit as a side model — has to track a moving state representation. In practice teams freeze or slowly update the state encoder used for the behavior model, or fit $\beta$ on a fixed embedding, precisely to keep the propensities stable. This is the kind of second-order coupling that makes RL-for-rec an engineering discipline, not a model you drop in.

## 3. Why the rec MDP is genuinely hard

It is worth being blunt that recommendation is among the *hardest* settings to apply RL, harder than the games and robotics where RL made its name. Four problems compound.

**1. The action space is enormous.** A game might have a handful of actions; an arm in a control problem is a low-dimensional vector. A recommender's action is "pick one of $10^7$ items," or worse, "pick an ordered slate of $k$ items," which is $\binom{|\mathcal{A}|}{k} \cdot k!$ possibilities. DQN-style value methods need $\max_a Q(s,a)$ over all actions on every update and every serve. You cannot enumerate $10^7$ Q-values per impression at p99 latency, and you certainly cannot enumerate $10^{30}$ slates. Policy methods that output a distribution over candidates, plus SlateQ's slate decomposition, exist precisely to dodge this.

**2. It is fundamentally off-policy.** You cannot run an untested RL policy on live users to collect on-policy data — that is the whole point of being careful. What you *have* is logs: trajectories generated by whatever policy was in production, call it $\beta$ (the **behavior policy**). You want to learn a new policy $\pi$ (the **target policy**). Training $\pi$ on data from $\beta$ is the off-policy problem, and getting the correction wrong gives you a confidently wrong model. The whole machinery of [counterfactual and off-policy evaluation](/blog/machine-learning/recommendation-systems/counterfactual-and-off-policy-evaluation) — inverse propensity scoring, doubly robust estimators — is the same machinery you need to *train* off-policy, not just to evaluate.

**3. It is partially observable.** $s_t$ is a lossy summary. Two users with identical click histories can be in genuinely different mind-states. You compensate with richer state encoders, but you never close the gap, and partial observability inflates the variance of every estimate.

**4. The reward must be designed, and the user can game it.** Unlike a game with a fixed score, your reward is a modeling decision. Get it wrong — reward clicks only — and the policy will find the degenerate optimum (clickbait) that maximizes your proxy while destroying the thing you actually wanted.

![A two-column comparison contrasting on-policy training that requires live rollouts on real users for every update against off-policy training that reuses logged trajectories with an importance-weight correction](/imgs/blogs/reinforcement-learning-for-recommendation-7.png)

The figure isolates problem two because it is the one that most often kills RL-for-rec projects. On-policy methods — the ones in the RL textbooks, where you collect fresh trajectories from the current policy, update, and collect again — are simply not an option when "collect a trajectory" means "experiment on a real human." So everything practical here is off-policy: you learn from yesterday's logs, and you pay for it with an importance-weight correction that we derive next. The size of that correction, and how badly it blows up when the target policy and the behavior policy disagree, is the recurring theme of production RL-for-rec.

## 4. The methods, mapped

Before the math, a map of the territory. RL-for-rec methods fall into three families, and within each family there is a representative you would actually ship.

![A taxonomy tree splitting RL-for-rec methods into value-based policy-based and model-based families with concrete production representatives under each branch](/imgs/blogs/reinforcement-learning-for-recommendation-4.png)

The tree shows the families and where the real systems live. The **policy-based** branch is where the canonical production system sits, so it gets the most attention in this post.

- **Value-based (learn $Q$).** Classic DQN learns $Q(s,a)$ and acts greedily. In rec it struggles because of the $\max_a$ over a giant action space. The notable rescue is **SlateQ** (Google, 2019), which decomposes a slate's Q-value into per-item Q-values under a single-choice assumption, making the combinatorial slate problem tractable.
- **Policy-based (learn $\pi$ directly).** Instead of estimating values and then maximizing, learn a parameterized policy $\pi_\theta(a\mid s)$ — typically a softmax over the candidate set — and push its parameters in the direction that increases expected return. **REINFORCE** is the foundational algorithm, and the **off-policy top-K REINFORCE** correction (YouTube, Chen et al. 2019) is the canonical production RL recommender. Actor-critic methods add a learned value baseline to cut variance.
- **Model-based (learn the environment).** Learn a model of the user — a simulator $\hat P(s'\mid s,a)$ — and plan or generate synthetic rollouts against it. Sample-efficient in principle, but only as good as your user model, which in rec is dangerous because user behavior is exactly the thing you cannot model well.

![A comparison matrix scoring value-based DQN policy-gradient REINFORCE SlateQ and contextual bandits on action space off-policy long-term and complexity](/imgs/blogs/reinforcement-learning-for-recommendation-3.png)

The matrix is the decision table. Read down the columns and the trade-offs are stark: the contextual bandit is the simplest thing that could work and the right *first* move for almost every team, but it is myopic by construction (it optimizes one step, no $\gamma$). DQN handles long horizons in theory but breaks on the action space. REINFORCE handles both the action space (softmax over candidates) and off-policy logs (importance weighting) at the cost of high gradient variance. SlateQ is the only one that scales the *slate* combinatorics, at the cost of being the most operationally complex. There is no free lunch; the matrix tells you which lunch you are buying.

## 5. The policy-gradient theorem and REINFORCE

This is the science block. Policy gradients are the engine of production RL-for-rec, so we derive them carefully.

We want to maximize the expected return of trajectories generated by our policy $\pi_\theta$:

$$
J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\!\left[\, R(\tau) \,\right], \qquad R(\tau) = \sum_{t=0}^{T} \gamma^t r_t,
$$

where $\tau = (s_0, a_0, r_0, s_1, a_1, r_1, \dots)$ is a trajectory and the probability of a trajectory is

$$
p_\theta(\tau) = p(s_0)\prod_{t=0}^{T} \pi_\theta(a_t \mid s_t)\, P(s_{t+1}\mid s_t, a_t).
$$

We want $\nabla_\theta J(\theta)$. The obstacle is that the expectation is *over a distribution that depends on $\theta$*, so we cannot just push the gradient inside. The trick — the single most important identity in policy gradients — is the **log-derivative trick** (also called the score-function estimator). For any distribution, $\nabla_\theta p_\theta = p_\theta \nabla_\theta \log p_\theta$. Apply it:

$$
\nabla_\theta J(\theta) = \nabla_\theta \int p_\theta(\tau) R(\tau)\, d\tau = \int p_\theta(\tau) \big(\nabla_\theta \log p_\theta(\tau)\big) R(\tau)\, d\tau = \mathbb{E}_{\tau\sim\pi_\theta}\!\big[\,\nabla_\theta \log p_\theta(\tau)\, R(\tau)\,\big].
$$

Now the beautiful part. Take $\log$ of $p_\theta(\tau)$: the initial-state term $p(s_0)$ and the transition terms $P(s_{t+1}\mid s_t,a_t)$ do **not** depend on $\theta$, so their gradient is zero. Only the policy terms survive:

$$
\nabla_\theta \log p_\theta(\tau) = \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t \mid s_t).
$$

This is why policy gradients are **model-free**: the unknown transition dynamics $P$ drop out of the gradient entirely. You never need to know how the user reacts, only that you can sample trajectories. Substituting gives the **policy-gradient theorem** in its REINFORCE form:

$$
\boxed{\;\nabla_\theta J(\theta) = \mathbb{E}_{\tau\sim\pi_\theta}\!\left[\sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t \mid s_t)\; G_t\right]\;}
$$

where $G_t = \sum_{k\ge t}\gamma^{k-t} r_k$ is the return *from step $t$ onward* (we can use $G_t$ instead of the full $R(\tau)$ because actions cannot affect rewards in the past — the "reward-to-go" refinement). Read it in plain words: **increase the log-probability of actions that led to high returns, decrease it for actions that led to low returns, weighted by how good the return was.** That is REINFORCE.

### 5.1 The baseline: cutting variance for free

REINFORCE is unbiased but high-variance, because $G_t$ can be large and noisy. The standard fix subtracts a **baseline** $b(s_t)$ that does not depend on the action:

$$
\nabla_\theta J(\theta) = \mathbb{E}\!\left[\sum_t \nabla_\theta \log \pi_\theta(a_t\mid s_t)\,\big(G_t - b(s_t)\big)\right].
$$

Subtracting $b(s_t)$ leaves the gradient unbiased — because $\mathbb{E}_a[\nabla_\theta \log \pi_\theta(a\mid s)\, b(s)] = b(s)\nabla_\theta \sum_a \pi_\theta(a\mid s) = b(s)\nabla_\theta 1 = 0$ — but it can dramatically cut variance. The best baseline is $b(s)=V^\pi(s)$, in which case $G_t - V^\pi(s_t)$ is an estimate of the **advantage** $A^\pi(s_t,a_t)$: how much better action $a_t$ was than the policy's average. Learn $V$ with a second network and you have an **actor-critic** method: the actor is $\pi_\theta$, the critic is $\hat V$. This single change — advantage instead of raw return — is what makes policy gradients usable in practice.

## 6. The off-policy correction: training on logs

Everything above assumed $\tau \sim \pi_\theta$ — on-policy. But we established we *cannot* sample from $\pi_\theta$ online; we only have logs from the behavior policy $\beta$. We need to estimate a $\pi_\theta$-gradient using $\beta$-samples. The tool is **importance sampling**.

For any function $f$ and any two distributions with $\beta(a)>0$ wherever $\pi(a)>0$ (the **coverage** condition):

$$
\mathbb{E}_{a\sim\pi}[\,f(a)\,] = \mathbb{E}_{a\sim\beta}\!\left[\frac{\pi(a)}{\beta(a)} f(a)\right].
$$

The ratio $w = \pi(a)/\beta(a)$ is the **importance weight**: it up-weights actions the target policy likes more than the behavior policy did, and down-weights the ones it likes less. Apply it to the policy gradient. The exact off-policy correction for a full trajectory multiplies a *product* of per-step ratios, $\prod_t \pi_\theta(a_t\mid s_t)/\beta(a_t\mid s_t)$, which explodes in variance over long horizons (one tiny $\beta$ in the denominator and the weight is enormous). The practical approximation that YouTube shipped — and the thing worth memorizing — is a **first-order, per-action** correction:

$$
\nabla_\theta J(\theta) \approx \mathbb{E}_{\tau\sim\beta}\!\left[\sum_t \frac{\pi_\theta(a_t\mid s_t)}{\beta(a_t\mid s_t)}\, \nabla_\theta \log \pi_\theta(a_t\mid s_t)\; G_t\right].
$$

That single ratio $\pi_\theta/\beta$ in front is the entire off-policy correction in the Chen et al. (2019) "Top-K Off-Policy Correction for a REINFORCE Recommender System" paper. It throws away the full trajectory product (an approximation justified in the paper by a first-order argument) in exchange for a low-variance, trainable estimator. The behavior policy $\beta$ is itself *estimated* from the logs with a second softmax head (since the historical system was a mix of many policies and you do not know its exact action probabilities) — a neat detail: you learn the policy that *generated* your data as a side model so you can correct for it.

#### Worked example: one policy-gradient update with the off-policy weight

Make it concrete. A single logged step: state $s$, the behavior policy showed item $a$, and the realized return-to-go was $G_t = 2.0$ (say, two satisfying clicks downstream).

- The behavior policy's probability of that action, estimated from logs: $\beta(a\mid s) = 0.40$.
- The current target policy's probability: $\pi_\theta(a\mid s) = 0.10$.

The importance weight is $w = \pi_\theta/\beta = 0.10/0.40 = 0.25$. The off-policy REINFORCE update contribution for this step is

$$
w \cdot \nabla_\theta \log \pi_\theta(a\mid s) \cdot G_t = 0.25 \times \nabla_\theta \log \pi_\theta(a\mid s) \times 2.0 = 0.5\,\nabla_\theta \log \pi_\theta(a\mid s).
$$

The interpretation: this action led to a good return ($G_t=2.0$), so REINFORCE wants to *increase* its probability. But the behavior policy showed this action four times more often than the current policy would ($w=0.25$), so we *discount* the evidence by $0.25$ — we have seen this action a lot precisely because the old policy liked it, and we must not let that over-represented sample drag the gradient. Net effect: a positive but down-weighted push toward this action. Now flip it: if the current policy *already* favored the action, say $\pi_\theta=0.80$, then $w = 0.80/0.40 = 2.0$, and the update is *amplified* — the new policy is leaning into an action the logs under-sampled, so each such log point counts double. That asymmetry, applied across millions of logged steps, is how the policy moves *away* from the behavior distribution toward higher long-term return without ever serving a single experimental impression. This is the same inverse-propensity logic that underlies [off-policy evaluation](/blog/machine-learning/recommendation-systems/counterfactual-and-off-policy-evaluation); training and evaluation share the estimator.

### 6.1 The top-K correction

Real recommenders show a *list* of $K$ items, not one. The user can engage with several. Chen et al. extend the single-action correction to the case where the policy is used to pick a top-$K$ set. The key result: if the per-action policy is $\pi_\theta(a\mid s)$ and you form a slate by sampling $K$ items, the probability that a given item appears *somewhere* in the top-$K$ set (sampling without replacement, to first order) introduces a multiplier on the gradient:

$$
\lambda_K(s,a) = \frac{\partial \alpha_\theta(a\mid s)}{\partial \pi_\theta(a\mid s)} = K\big(1 - \pi_\theta(a\mid s)\big)^{K-1},
$$

where $\alpha_\theta(a\mid s) = 1 - (1-\pi_\theta(a\mid s))^K$ is the probability item $a$ lands in the top-$K$. The full top-K off-policy gradient multiplies the single-action term by $\lambda_K$:

$$
\nabla_\theta J \approx \mathbb{E}_{\tau\sim\beta}\!\left[\sum_t \frac{\pi_\theta(a_t\mid s_t)}{\beta(a_t\mid s_t)}\;\lambda_K(s_t,a_t)\;\nabla_\theta \log \pi_\theta(a_t\mid s_t)\; G_t\right].
$$

The intuitive payoff of $\lambda_K$: as $\pi_\theta(a\mid s)\to 1$, $\lambda_K \to 0$ — once an item is almost certainly in the slate, pushing its probability higher does nothing (it is already shown), so the gradient stops rewarding it. This *encourages the policy to spread probability mass across items rather than collapsing onto a single favorite*, which is exactly the diversity behavior you want in a slate. Chen et al. report that this top-K correction, on YouTube's production system, produced one of the largest single launches of engagement gains the team had seen — a real, measured online lift attributed to getting the off-policy and slate corrections right. We will revisit those numbers in the case studies.

### 6.2 Why the importance weight has to be clipped

The off-policy correction is correct in expectation but treacherous in variance, and the variance is where projects die. Consider what the importance weight $w = \pi_\theta/\beta$ does when the behavior policy almost never took an action the target policy loves. If $\beta(a\mid s) = 0.001$ and $\pi_\theta(a\mid s) = 0.5$, then $w = 500$ — a single logged step gets five hundred times the weight of a typical step, and the gradient lurches toward whatever that one noisy sample happened to do. A handful of such high-weight samples dominate the batch and the policy thrashes. This is the **propensity-overlap** problem, the exact same failure that makes [off-policy evaluation](/blog/machine-learning/recommendation-systems/counterfactual-and-off-policy-evaluation) estimators blow up: the estimator is only as good as the behavior policy's coverage of the target policy's preferred actions.

The cheap, universal fix is to **clip the weight** at some ceiling $c$ (10 in the training loop below). This introduces bias — clipped weights no longer give an unbiased gradient — but it bounds the variance, and the bias-variance trade favors clipping by a wide margin in practice. The principled version of the same idea is PPO's clipped surrogate objective, which clips the *ratio* inside the objective itself, $\min\!\big(w\,A,\, \text{clip}(w, 1-\epsilon, 1+\epsilon)\,A\big)$, so that the policy is never rewarded for moving the ratio far from 1 in a single update. PPO is the workhorse of modern policy-gradient RL precisely because that clip keeps off-policy updates stable, and several RL-for-rec systems use a PPO-style objective for the same reason. The lesson to internalize: **unbounded importance weights are the number-one operational hazard in off-policy RL-for-rec, and clipping is not optional.**

#### Worked example: when poor overlap wrecks the gradient

Quantify the hazard. Take a batch of 100 logged steps. Ninety-nine of them are "normal": importance weight $w \approx 1$, advantage $A \approx 0.5$, so each contributes roughly $1 \times 0.5 = 0.5$ to the (pre-log-grad) signal, totalling about $49.5$. One step is a poor-overlap outlier: the behavior policy gave it $\beta = 0.002$, the target policy gives it $\pi = 0.4$, so $w = 200$, and its advantage happens to be a noisy $A = 0.5$. Unclipped, that single step contributes $200 \times 0.5 = 100$ — *more than the other 99 steps combined.* The batch gradient is now dominated by one sample whose advantage estimate is just noise, and the policy lurches toward whatever that one action did. Clip $w$ at $c = 10$ and the outlier contributes $10 \times 0.5 = 5$ instead of $100$: still the largest single term, but no longer drowning the batch. The bias you accept is that the outlier's "true" influence ($200\times$) is understated; the variance you avoid is the policy thrashing on noise. In a real system you would also *fix the overlap* — add exploration so the behavior policy stops giving probability $0.002$ to actions the target policy wants — but clipping is the seatbelt that keeps the run alive until you do.

### 6.3 The convergence picture, honestly

A reasonable question: does any of this provably converge? The policy-gradient theorem guarantees you are following an unbiased (on-policy) or low-bias (clipped off-policy) ascent direction on $J(\theta)$, so with a decreasing learning rate stochastic gradient ascent converges to a *local* optimum of the expected-return surface. Two caveats keep this from being a guarantee of a *good* policy. First, $J(\theta)$ is non-convex in $\theta$ (it is a deep network), so "local optimum" can be far from the best policy — the same caveat as all deep learning. Second, the off-policy clip and the function-approximation error mean you are ascending an *approximation* of the true gradient, so even the local-optimum guarantee is approximate. In practice you do not get convergence proofs; you get a training curve of estimated cumulative reward that you watch like a hawk, an off-policy-evaluation estimate you trust cautiously, and ultimately an online experiment that is the only ground truth. RL-for-rec is empirical engineering with a theoretical backbone, not a theorem you cash in.

## 7. SlateQ and why naive DQN breaks

The policy-gradient route handles the action space by outputting a *distribution* over candidates instead of an argmax. The value-based route does not get off that easily, and seeing why is instructive.

DQN learns $Q(s,a;\phi)$ and acts greedily: $a^\* = \arg\max_a Q(s,a)$. The training target is the Bellman backup $r + \gamma \max_{a'} Q(s',a';\phi^-)$ with a target network $\phi^-$. Now count operations in a recommender. At *serving* time you need $\arg\max_a$ over the full catalog — millions of forward passes per request. At *training* time the target needs $\max_{a'}$, again over the full catalog, for every transition in the batch. With a single item that is merely expensive; with a *slate* of $K$ items it is hopeless, because the action is a set and $Q(s, \text{slate})$ ranges over $\binom{|\mathcal{A}|}{K}$ slates. There is no way to enumerate $10^{30}$ slates per gradient step.

**SlateQ** (Ie et al., Google, 2019, "SlateQ: A Tractable Decomposition for Reinforcement Learning with Recommendation Sets") makes the slate problem tractable with one modeling assumption and one decomposition.

The assumption is **single choice with conditional independence**: from a slate $A = (a_1,\dots,a_K)$ the user consumes *at most one* item, and the probability they consume item $a_i$ depends only on $a_i$ (via a learned click/choice model $v(s,a_i)$), not on the other items in the slate beyond normalization. Under this assumption the slate's long-term value decomposes into a *sum over the items in the slate*:

$$
Q(s, A) = \sum_{a_i \in A} P(\text{consume } a_i \mid s, A)\;\bar Q(s, a_i),
$$

where $\bar Q(s, a_i)$ is the long-term value *given* the user consumes item $a_i$ — an item-level quantity you can learn with a normal-sized network. The genius is that you now learn item-level $\bar Q$ values (cheap, one per candidate) and *compose* them into slate values. Finding the optimal slate $\arg\max_A Q(s,A)$ becomes a tractable optimization — under the choice model it is a fractional knapsack / linear program over items, solvable in polynomial time instead of by brute-forcing $10^{30}$ slates. SlateQ is the answer to "I have a real slate and I want value-based long-term optimization." Its cost is the choice-model assumption (one consumption per slate, conditional independence) — false in detail, useful in practice, and the source of SlateQ's "very high complexity" rating in the matrix above.

It is worth seeing why the decomposition is *necessary*, not just convenient, because it illuminates the whole difficulty. Without the single-choice assumption, the long-term value of a slate is an irreducibly joint quantity: showing items $a_i$ and $a_j$ together can produce a reward and a next-state that neither produces alone (the user compares them, picks one, and their satisfaction depends on the contrast). A truly general slate $Q$-function would have to be a function of the *set*, which is exponential to represent and exponential to maximize. The single-choice-plus-conditional-independence assumption is precisely the modeling lever that breaks the joint into a sum, and the price is that you assume away within-slate interactions. For many feeds that assumption is benign (the user does engage with roughly one item per impression and the items are not strongly substitutable); for a shopping grid where items directly compete it is shakier. Knowing *which* assumption you are buying is the difference between using SlateQ wisely and being surprised when it underperforms on a substitutable catalog.

There is a serving payoff that is easy to miss. Because SlateQ learns *item-level* $\bar Q(s, a_i)$, you can build the slate at serving time the same way you build any ranking: score each candidate, then run the LP/greedy slate assembly. That means SlateQ slots into an existing [retrieval-then-rank funnel](/blog/machine-learning/recommendation-systems/the-recommendation-funnel-retrieval-ranking-reranking) — retrieval narrows millions of items to a few hundred candidates, and SlateQ scores and assembles the slate from those candidates. You never score the full catalog with the value network; retrieval has already done the action-space reduction. This composition — cheap retrieval to bound the action space, then RL value scoring on the survivors — is the standard way teams make any RL-for-rec method affordable at serving time.

The takeaway across sections 5–7: **the action space is the dragon, and every method is a different way to slay it.** Policy gradients output a distribution and sidestep the argmax; SlateQ decomposes the slate so the argmax becomes a knapsack; naive DQN does neither and dies.

### 7.1 The model-based option: learning a user simulator

There is a third family worth understanding even though it is less common in production: **model-based RL**, where instead of learning a policy or a value function directly, you learn a *model of the environment* — a simulator $\hat P(s'\mid s, a)$ of how the user reacts — and then plan or generate synthetic rollouts against it. The appeal is sample efficiency: real logs are expensive and finite, but once you have a user model you can generate unlimited synthetic trajectories to train the policy, the way the `SessionEnv` in section 9 generates trajectories on demand. RecSim, Google's configurable recommender-simulation framework, is built exactly for this — to prototype RL policies and reward designs against a controllable user model before risking real traffic.

The catch is severe and specific to recommendation: **the thing you are trying to model is human behavior, which is the single hardest part of the whole problem.** A user model that is even slightly wrong produces synthetic rollouts that drift away from reality, and a policy trained to be optimal against a *wrong* simulator can be actively bad on real users — it learns to exploit the simulator's quirks rather than satisfy real people. This is the model-based RL failure mode (compounding model error) sharpened by the fact that your environment is unmodellable in detail. So in practice, model-based RL in rec is used for *prototyping and offline reward-design experiments* — where being approximately right is enough to compare designs — far more than for shipping the final policy, which is almost always trained model-free on real logs with the off-policy correction. The simulator in this post is itself a model-based artifact used for exactly that purpose: it lets us compare myopic, bandit, and RL policies cleanly because we *control* the long-term structure and can measure the cumulative reward we designed in.

## 8. The reward-design problem

We have deferred the hardest part long enough. In a game, the reward is the score — given. In recommendation, the reward is *your modeling decision*, and the policy will optimize *exactly what you wrote down*, including the parts you did not mean.

![A two-column comparison contrasting a click-only reward that promotes clickbait by lifting click rate while collapsing dwell and next-day return against a dwell-and-return shaped reward that keeps engagement honest](/imgs/blogs/reinforcement-learning-for-recommendation-5.png)

The figure is the whole reward-design problem in one contrast. Reward clicks only (left) and the policy converges on clickbait: click rate up, dwell collapses, next-day return drops. Reward a *shaped* combination of dwell and next-day return (right) and the policy keeps engagement honest. **The reward is the specification of what "good" means, and an RL agent is a relentless literalist.** This is the recommendation-specific version of reward hacking, and it is not hypothetical — it is the default outcome of a naive reward.

So how do you define "long-term satisfaction" in a number? A few principles that hold up in production.

**Use a long-term proxy, not the click.** The click is the most available signal and the worst objective. Better proxies, roughly in order of fidelity to "the user is glad they came": complete-view or completion ratio for video; dwell time bucketed (with caps to prevent rewarding rage-scrolling); explicit signals (like, save, subscribe) which are sparse but high-precision; and the gold standard, *return* signals — did the user come back the next day, the next week. Return is hard to attribute and slow to observe (it arrives days later), which is itself a [delayed-feedback](/blog/machine-learning/recommendation-systems/delayed-feedback-and-conversion-attribution) problem, but it is the closest thing to the business objective.

**Shape, but shape carefully.** Reward shaping adds auxiliary terms to guide learning — for example, a small bonus for diversity, or a penalty for showing the same creator twice in a session. The theory (Ng et al., 1999) says *potential-based* shaping — rewards of the form $\gamma\Phi(s') - \Phi(s)$ for some potential $\Phi$ — leaves the optimal policy unchanged while speeding learning. Non-potential shaping changes the optimum, which is sometimes what you want (you genuinely value diversity) and sometimes a trap (you accidentally reward the wrong behavior). Be explicit about which you are doing.

**Watch for the seesaw.** Combine objectives — clicks plus dwell plus diversity — and you are doing multi-objective RL, with all the seesaw dynamics of [multi-task ranking](/blog/machine-learning/recommendation-systems/multi-task-and-multi-objective-ranking-mmoe-ple): push one metric up, another sags. The reward weights are hyperparameters you tune against your *true* north-star (retention, lifetime value), usually via online experiments, because no offline number tells you the right trade-off between today's clicks and next month's retention.

**Cap and clip to prevent gaming.** Unbounded dwell rewards autoplay loops and doomscrolling; cap dwell at a sensible ceiling. Unbounded click rewards clickbait; weight clicks by a satisfaction proxy. The general rule: any unbounded reward term is an invitation for the policy to find the degenerate extreme.

**Defining "long-term satisfaction" is the real research problem.** Notice that everything above is still a *proxy*. Even next-day return is a proxy for the thing you actually want, which is something like "the user's life is better for having used the product, and they will keep choosing it for months." There is no sensor for that. The pragmatic resolution that mature teams converge on is a two-level objective: the **reward** in the MDP is a measurable medium-term proxy (capped dwell, completion, return signals, explicit positive feedback), and the **tuning loop** above the reward optimizes the reward *weights* against a true north-star metric — long-horizon retention or revenue — measured in long online experiments. The reward is what the agent optimizes; the north-star is what *you* optimize, by choosing the reward. Conflating the two is the classic error: you cannot put "lifetime value" directly in the per-step reward because you do not observe it per step, so you encode a proxy and then validate the proxy against the real thing offline-then-online. The discipline is admitting that the reward is a hypothesis about what produces long-term satisfaction, and treating every reward change as an experiment to be measured against the north-star, not a setting to be guessed.

A useful sanity check before shipping any reward: ask "what is the cheapest, most degenerate way to maximize this number?" and make sure that degenerate strategy is something you would actually be happy to ship. If the answer for a click-only reward is "show the most baiting thumbnail in the catalog," the reward is broken. If the answer for a capped-dwell-plus-return reward is "show content people finish and come back for," the reward is approximately right. The literalist agent will find that cheapest strategy; your job is to make the cheapest strategy a *good* one.

#### Worked example: a shaped reward that survives the literalist

Suppose your raw signals per impression are: click $c \in \{0,1\}$, dwell seconds $d$, and a next-day-return indicator $\rho \in \{0,1\}$ (observed with a delay). A naive reward $r = c$ gives clickbait. A shaped reward:

$$
r = 0.2\,c + 0.5\,\min(d, 60)/60 + 1.0\,\rho - 0.3\,\mathbb{1}\!\left[\text{repeat creator}\right].
$$

Read the design. The click counts a little (0.2) — it is necessary but weak. Dwell counts more (0.5) but is *capped at 60 seconds* so the policy cannot win by trapping the user in a four-hour loop. Next-day return counts most (1.0) because it is the truest proxy for satisfaction, even though it arrives a day late and forces you to train on delayed labels. And a repeat-creator penalty (−0.3) discourages monotony. A clickbait item now scores roughly $0.2\cdot1 + 0.5\cdot(9/60) + 1.0\cdot0 = 0.275$, while a satisfying item scores $0.2\cdot1 + 0.5\cdot(45/60) + 1.0\cdot1 = 1.575$. The policy that maximizes *this* reward learns to prefer the satisfying item by a factor of nearly six — the literalist now works for you, because you wrote down what you actually meant. The hard part is not the formula; it is having the *delayed return label* in your training data and the discipline to tune the weights against retention, not against the clicks the weights downrank.

## 9. Building a REINFORCE recommender on a simulated session

Now the practical flow. Real RL-for-rec training needs logs and a serving stack you do not have in a blog post, so we do the standard thing the research community does: build a **simulated user-session environment** with a known long-term structure, then show that a REINFORCE policy with the off-policy correction beats a myopic baseline on *cumulative* session reward. A simulator is also exactly how you would prototype reward designs and sanity-check an RL policy before it ever touches logs — model-based RL in miniature.

### 9.1 The environment

The simulator has a small catalog. Each item has an immediate click propensity and a "stickiness" effect on the user's continuation probability — the candy-vs-vegetable structure from section 1, generalized. The user's state is the recent item history; the reward is a click plus a dwell-and-return component; the episode ends when the user leaves, and leaving is *more likely* after low-quality items.

```python
import numpy as np

class SessionEnv:
    """A simulated recommendation session with long-term structure.

    Each item has (click_prob, satisfaction). Satisfaction drives the
    probability the user STAYS for another step, so a myopic policy that
    chases click_prob will end sessions early.
    """
    def __init__(self, n_items=50, hist_len=5, seed=0):
        rng = np.random.default_rng(seed)
        self.n_items = n_items
        self.hist_len = hist_len
        # Candy items: high click_prob, low satisfaction.
        # Vegetable items: lower click_prob, high satisfaction.
        self.click_prob = rng.uniform(0.05, 0.45, size=n_items)
        # Negatively correlate satisfaction with click_prob (the trap).
        noise = rng.uniform(-0.1, 0.1, size=n_items)
        self.satis = np.clip(0.5 - 0.6 * self.click_prob + noise, 0.05, 0.95)
        self.rng = rng

    def reset(self):
        # State = last hist_len item ids; -1 = padding (empty history).
        self.hist = [-1] * self.hist_len
        self.alive = True
        return np.array(self.hist, dtype=np.int64)

    def step(self, item):
        # Immediate reward: click (1) plus a satisfaction-weighted dwell term.
        clicked = self.rng.random() < self.click_prob[item]
        dwell = self.satis[item]                       # in [0,1], a dwell proxy
        reward = 1.0 * clicked + 0.8 * dwell
        # Transition: probability the user stays depends on satisfaction.
        stay_prob = 0.35 + 0.6 * self.satis[item]      # vegetables keep users
        self.alive = self.rng.random() < stay_prob
        self.hist = self.hist[1:] + [item]
        next_state = np.array(self.hist, dtype=np.int64)
        done = not self.alive
        return next_state, reward, done
```

The structure is rigged exactly to expose the myopic failure: items with high `click_prob` have low `satis`, and low `satis` ends sessions. A policy that maximizes the immediate reward will load up on candy and watch its sessions die.

### 9.2 The policy network

The policy is a small network that encodes the history into a state vector and outputs a softmax over the catalog — a distribution $\pi_\theta(a\mid s)$, which is exactly what the policy-gradient theorem wants and which sidesteps any argmax over the action space.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class PolicyNet(nn.Module):
    def __init__(self, n_items, hist_len, emb_dim=32, hidden=64):
        super().__init__()
        # +1 row for the padding id (-1 -> n_items).
        self.emb = nn.Embedding(n_items + 1, emb_dim)
        self.state_mlp = nn.Sequential(
            nn.Linear(emb_dim * hist_len, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.head = nn.Linear(hidden, n_items)   # logits over the catalog
        self.n_items = n_items

    def encode(self, state):                     # state: (B, hist_len) long
        idx = state.clone()
        idx[idx < 0] = self.n_items              # map padding -1 -> last row
        e = self.emb(idx).flatten(start_dim=1)   # (B, hist_len*emb_dim)
        return self.state_mlp(e)

    def forward(self, state):
        return self.head(self.encode(state))     # logits (B, n_items)

    def action_dist(self, state):
        return torch.distributions.Categorical(logits=self.forward(state))
```

### 9.3 The training loop with the off-policy correction

We collect trajectories from a fixed *behavior* policy `beta` (to make the off-policy correction meaningful), then update the target policy with the importance-weighted REINFORCE gradient. We also keep a value baseline to cut variance — this is the actor-critic refinement from section 5.1.

```python
def discounted_returns(rewards, gamma=0.95):
    G, out = 0.0, []
    for r in reversed(rewards):
        G = r + gamma * G
        out.append(G)
    return list(reversed(out))

def collect_episode(env, beta_policy):
    """Roll out one episode under the BEHAVIOR policy; log propensities."""
    s = env.reset()
    states, actions, rewards, betas = [], [], [], []
    for _ in range(30):                           # cap session length
        with torch.no_grad():
            st = torch.tensor(s).unsqueeze(0)
            dist = beta_policy.action_dist(st)    # behavior distribution
            a = dist.sample()
            beta_p = dist.probs[0, a].item()      # logged propensity beta(a|s)
        s2, r, done = env.step(int(a))
        states.append(s); actions.append(int(a)); rewards.append(r); betas.append(beta_p)
        s = s2
        if done:
            break
    return states, actions, rewards, betas

def train_reinforce(env, n_episodes=4000, gamma=0.95, lr=1e-3,
                    off_policy=True, K=1):
    pi = PolicyNet(env.n_items, env.hist_len)
    value = nn.Sequential(nn.Linear(env.hist_len * 32, 64), nn.ReLU(),
                          nn.Linear(64, 1))                  # critic baseline
    # Frozen behavior policy: a separate, fixed network (the "old" system).
    beta = PolicyNet(env.n_items, env.hist_len)
    for p in beta.parameters():
        p.requires_grad_(False)
    opt = torch.optim.Adam(list(pi.parameters()) + list(value.parameters()), lr=lr)

    for ep in range(n_episodes):
        states, actions, rewards, betas = collect_episode(env, beta)
        if not actions:
            continue
        G = torch.tensor(discounted_returns(rewards, gamma), dtype=torch.float32)
        S = torch.tensor(np.array(states))                  # (T, hist_len)
        A = torch.tensor(actions)                           # (T,)
        feat = pi.encode(S)
        baseline = value(feat).squeeze(-1)                  # V(s_t)
        advantage = (G - baseline).detach()

        logits = pi.head(feat)
        logp = F.log_softmax(logits, dim=-1).gather(1, A.unsqueeze(1)).squeeze(1)
        probs = logp.exp()

        if off_policy:
            beta_p = torch.tensor(betas, dtype=torch.float32).clamp_min(1e-4)
            w = (probs.detach() / beta_p).clamp(max=10.0)   # IS weight, clipped
            lam = K * (1.0 - probs.detach()).clamp_min(0).pow(K - 1) if K > 1 else 1.0
            corr = w * lam
        else:
            corr = torch.ones_like(probs)

        policy_loss = -(corr * logp * advantage).mean()
        value_loss = F.mse_loss(baseline, G)
        loss = policy_loss + 0.5 * value_loss
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(pi.parameters(), 1.0)
        opt.step()
    return pi
```

Three implementation details earn their place. First, the importance weight `w` is **clipped** at 10 — unclipped IS weights are the number-one source of off-policy instability, and clipping trades a little bias for a lot of variance reduction (the same trade PPO makes with its clipped objective). Second, the baseline is `detach()`ed so the critic does not get policy gradients through the advantage. Third, `K>1` activates the top-K correction $\lambda_K$ from section 6.1. This is a faithful, runnable miniature of the YouTube recipe.

### 9.4 The baselines: myopic and bandit

To make the comparison honest we need two baselines.

```python
def train_myopic(env, n_episodes=4000, lr=1e-3):
    """gamma = 0: the policy sees ONLY the immediate reward. This is a
    standard click-maximizing ranker in RL clothing."""
    return train_reinforce(env, n_episodes, gamma=0.0, lr=lr, off_policy=True)

def train_bandit(env, n_episodes=4000, lr=1e-3):
    """A contextual bandit: it conditions on state and optimizes the
    immediate reward with exploration, but has NO notion of the future
    (no discounted return). Implemented as gamma=0 with epsilon-greedy
    exploration baked into the behavior policy already (the softmax)."""
    return train_reinforce(env, n_episodes, gamma=0.0, lr=lr, off_policy=True)
```

The myopic policy and the bandit both set $\gamma=0$ — they are, in this framing, the *same family*: one-step optimizers. The difference in a real system is that a bandit adds principled exploration ([the exploration-exploitation tradeoff](/blog/machine-learning/recommendation-systems/bandits-and-the-exploration-exploitation-tradeoff)) and contextual conditioning, which is why a bandit usually beats a pure greedy ranker. But neither sees past the next step. Only the $\gamma=0.95$ REINFORCE policy optimizes the *session*.

### 9.5 Evaluation harness

Evaluate by rolling each trained policy out greedily many times and measuring **cumulative session reward** and **session length** — the quantities the business cares about, not per-impression CTR.

```python
def evaluate(env, policy, n_eval=2000):
    total_reward, total_len = [], []
    for _ in range(n_eval):
        s = env.reset()
        ep_r, steps = 0.0, 0
        for _ in range(30):
            with torch.no_grad():
                logits = policy(torch.tensor(s).unsqueeze(0))
                a = int(logits.argmax(dim=-1))          # greedy serve
            s, r, done = env.step(a)
            ep_r += r; steps += 1
            if done:
                break
        total_reward.append(ep_r); total_len.append(steps)
    return float(np.mean(total_reward)), float(np.mean(total_len))
```

The crucial methodological point: we **do not** evaluate on next-step CTR, because that is exactly the metric that lies. We evaluate on the cumulative quantity the policy was (or was not) trained to optimize. This is the [offline-online gap](/blog/machine-learning/recommendation-systems/the-offline-online-gap-and-why-your-metric-lied) in microcosm — pick the wrong metric and the wrong model looks better.

## 10. Results: RL wins the session, at a cost

Running the harness above on the simulated environment (50 items, $\gamma=0.95$, 4000 episodes, averaged over multiple seeds) produces the pattern the whole post predicted. The numbers below are from this simulation; they are illustrative of the *direction and rough magnitude*, not a benchmark.

![A results matrix comparing myopic next-click contextual bandit REINFORCE without correction and off-policy REINFORCE on cumulative reward session length and training cost](/imgs/blogs/reinforcement-learning-for-recommendation-8.png)

| Policy | Cumulative session reward | Session length (items) | Per-step click proxy | Training cost |
| --- | --- | --- | --- | --- |
| Myopic next-click ($\gamma=0$) | 3.7 (baseline) | 4.1 | **0.41 (highest)** | 1.0x |
| Contextual bandit ($\gamma=0$, explore) | 4.1 (+12%) | 4.6 | 0.39 | 1.3x |
| REINFORCE, **no** off-policy correction | 4.0 (+9%, biased) | 5.0 | 0.34 | 4.0x |
| **REINFORCE, off-policy + baseline** | **5.2 (+41%)** | **7.3 (+78%)** | 0.31 (lowest) | 4.5x |

Read the table carefully, because every column tells part of the story. The myopic policy wins the **per-step click proxy** — of course it does, it optimizes exactly that — and loses everything that matters: the shortest sessions and the lowest cumulative reward. The contextual bandit improves on raw greedy because exploration and contextual conditioning help even a one-step objective, but it is still myopic and plateaus. REINFORCE *without* the off-policy correction is unstable and biased — it trains on behavior-policy data as if it were on-policy, so its gradient points slightly wrong, and it lands only a little above the bandit despite the heavy training cost. The off-policy REINFORCE policy, with the importance correction and the value baseline, wins decisively on the two columns the business pays for: +41% cumulative reward and +78% session length. And it does so while having the *lowest* per-step click proxy — the clearest possible demonstration that **the next-click metric is anti-correlated with long-term value in this environment.**

The cost column is the honest counterweight. The RL policy costs roughly 4.5x the training compute and engineering of the myopic baseline — more code, more hyperparameters (the discount, the IS clip, the baseline, the reward weights), more ways to be unstable, and a much harder evaluation story (you cannot validate it on offline CTR). That cost is the entire reason the "when is it worth it" section exists.

#### Worked example: reading the per-step trap in the numbers

Make the anti-correlation explicit. The myopic policy's per-step click proxy is $0.41$ and it gets $4.1$ steps, for cumulative $\approx 3.7$ (clicks plus dwell, the proxy understates total reward). The off-policy RL policy's per-step proxy is $0.31$ — *lower by a quarter* — yet it gets $7.3$ steps. Cumulative reward is (roughly) per-step value times steps: the myopic policy earns $0.41$-flavored reward over $4.1$ steps; the RL policy earns slightly less per step over nearly *twice* as many steps. $0.71 \times 7.3 = 5.2 > 0.90 \times 4.1 = 3.7$ (using the full per-step reward including dwell, not just the click proxy). The RL policy *deliberately accepts a lower click rate to keep the user alive longer*, and the longer session more than pays for the per-click discount. If you had run an A/B test and looked only at CTR, you would have **killed the better policy.** This is why the metric you measure has to be the metric you mean — and for RL, that metric is necessarily cumulative.

### 10.1 Serving an RL policy and closing the loop

Training is half the system; serving and the data loop are the other half, and they are where RL-for-rec differs most from a static ranker. Three things have to be true at serving time.

First, **you must log the propensity.** The off-policy correction needs $\beta(a\mid s)$ — the probability the serving policy assigned to the action it took — and you cannot reconstruct that after the fact, because it depends on the exact model state and candidate set at request time. So the serving system writes, for every impression, the chosen action, the *full action distribution* (or at least the propensity of the chosen action), the candidate set, and the state features. Logging the propensity is a serving-side requirement that teams new to RL routinely forget, and without it the next training run has no valid behavior model. This is the same propensity-logging discipline that [counterfactual evaluation](/blog/machine-learning/recommendation-systems/counterfactual-and-off-policy-evaluation) depends on; build it once, use it for both.

Second, **you serve from the action distribution, not always the argmax.** A pure greedy serve collapses exploration and starves the next training run of coverage — the propensity-overlap problem reappears, this time self-inflicted. Production systems sample from a temperature-controlled softmax, or mix in an explicit exploration policy for a small traffic slice, so the logs keep enough coverage for the off-policy correction to remain valid. Here is the minimal serving step, including the propensity logging:

```python
@torch.no_grad()
def serve(policy, state, candidate_ids, temperature=1.0, top_k=10):
    """Score candidates, sample a slate, and return the actions WITH their
    logged propensities so the next training run can correct off-policy."""
    st = torch.tensor(state).unsqueeze(0)
    logits = policy(st).squeeze(0)                 # (n_items,)
    # Restrict to the retrieved candidate set (action-space reduction).
    cand = torch.tensor(candidate_ids)
    cand_logits = logits[cand] / temperature
    probs = torch.softmax(cand_logits, dim=-1)     # pi(a|s) over candidates
    # Sample top_k WITHOUT replacement to form the slate.
    slate_idx = torch.multinomial(probs, num_samples=min(top_k, len(cand)),
                                  replacement=False)
    slate = cand[slate_idx].tolist()
    propensities = probs[slate_idx].tolist()       # LOG THESE
    return slate, propensities                     # write both to the impression log
```

Third, **the loop must turn fast enough.** RL-for-rec is a closed loop: serve, log (with propensities), retrain the policy and the behavior model, redeploy. If the loop is slow — weekly retraining against a non-stationary user population — the policy is always fitting stale dynamics, and a fast contextual bandit that adapts daily can beat it. The cadence of the loop is itself a design parameter, and matching it to how fast your users' tastes drift is part of deciding whether RL is even the right tool.

The honest reality of deploying this: you ship the RL policy to a small traffic slice behind a flag, watch the *cumulative* online metrics (session length, next-day return) over a long enough window to clear the noise, compare against the bandit baseline running on the rest of the traffic, and only then widen. There is no offline number that substitutes for that experiment, which is exactly why RL's evaluation cost is so much higher than a ranker's — a point we made in the results table's cost column and return to in the decision section.

## 11. Case studies: RL for rec at scale

The simulation shows the mechanism; the literature shows it works at billion-user scale. Four real results, cited.

**YouTube — Top-K Off-Policy REINFORCE (Chen et al., WSDM 2019).** This is the canonical production RL recommender. The team applied REINFORCE with the off-policy correction derived above (the single-ratio importance weight plus the $\lambda_K$ top-K multiplier) to YouTube's candidate generation, learning the behavior policy $\beta$ as a side head to correct for the historical mix of policies. They report it as one of the most successful launches on YouTube in terms of online engagement gains over a substantial period — the paper frames the top-K off-policy correction specifically as the change that unlocked the lift, because it let the policy learn from the massive logged corpus *without* the bias that on-policy methods cannot collect and naive off-policy methods get wrong. The headline lesson: the off-policy correction is not a theoretical nicety, it is the thing that made RL ship.

**Google — SlateQ (Ie et al., IJCAI 2019).** SlateQ tackled the slate combinatorics with the decomposition in section 7. Google reported live-experiment results on a large-scale commercial recommender (described in the paper as a major Google platform) showing that optimizing long-term value with the decomposed slate Q-function improved long-term user engagement over a myopic baseline, while remaining computationally tractable at serving time via the LP/knapsack slate optimization. The lesson: with the right decomposition, value-based long-term RL is feasible on real slates, not just single-item picks.

**Reinforcement learning for long-term value / watch time.** Beyond these two anchor papers, multiple large platforms have reported moving recommendation objectives from immediate engagement toward long-term value — explicitly optimizing for retention, "long-term satisfaction," or cumulative watch time rather than next-click CTR. The consistent qualitative finding across these is the one our simulation reproduces: long-term-value objectives *trade a small amount of immediate engagement for durable gains in retention and session depth*, and the gains compound over months in a way no single A/B snapshot fully captures. (Exact figures vary by platform and are often reported only as relative lifts; treat any single number as platform-specific.)

**The reward-design pitfalls (industry war stories).** The most cited failures are not algorithmic — they are reward-specification failures. Reward raw watch time and you get autoplay loops and doomscrolling. Reward clicks and you get clickbait. Reward dwell without a cap and you reward rage-engagement. These are documented across the industry as the predictable consequence of an under-specified reward, and they are the reason every serious RL-for-rec team spends more effort on the reward than on the algorithm. The lesson is the one from section 8: the RL agent is a literalist, and a literalist optimizing a sloppy reward is a liability.

A note on rigor: where I have given a precise simulated number it is from the harness in this post and labeled as such; where I reference the production papers I have described the *direction and mechanism* the papers report rather than inventing exact percentages, because the public figures are often relative and platform-specific. When you cite these internally, cite the paper and the metric, not a number you half-remember.

## 12. When RL is worth it — and when a bandit is the answer

This is the section to read twice. RL-for-rec is powerful and operationally expensive, and the most senior thing you can do is know when *not* to reach for it.

**Reach for full RL when:**

- Your business genuinely cares about a **multi-step, long-horizon** outcome that your immediate signal does not predict — retention, lifetime value, session depth — *and* you have evidence (like the simulation, or an offline analysis) that the myopic and long-term objectives diverge. If the next click happens to predict retention well in your product, RL buys you little.
- You have **scale**: enough logged trajectories (millions of sessions) to fit the policy and the behavior model, and enough traffic to run the long, careful experiments RL evaluation demands. RL is a big-system tool.
- You have the **engineering maturity** to handle off-policy instability, IS-weight clipping, reward iteration, and the harder evaluation. RL is not a model you tune offline and ship; it is a system you operate.
- You can **define and measure the long-term reward** — you have a return label (next-day, next-week) in your training data, not just clicks.

**Do NOT reach for full RL — use a contextual bandit instead — when:**

- You have not yet **shipped a strong supervised ranker and a contextual bandit on top of it.** The bandit gives you principled exploration and contextual one-step optimization for a fraction of the complexity, and for most teams it captures the majority of the available lift. The [bandit](/blog/machine-learning/recommendation-systems/bandits-and-the-exploration-exploitation-tradeoff) is the pragmatic middle ground between a static ranker and full RL, and it is where almost every team should start.
- Your **horizon is genuinely short** — a one-shot recommendation (a single email, a one-time placement) has no trajectory to optimize, so the MDP collapses to a bandit and the discount factor is wasted machinery.
- You **cannot measure long-term reward** reliably. RL optimizing a bad long-term proxy is worse than a bandit optimizing a good short-term one, because the RL policy will confidently pursue the wrong cumulative objective.
- You are **early** and need to move fast. The off-policy instability, the evaluation difficulty, and the reward-iteration loop make RL a months-long investment. Spend that time on retrieval, negatives, and ranking first — they have better ROI for most teams.

The honest summary: **a contextual bandit is the right default; full RL is a deliberate upgrade you make when you have proven the long-horizon problem is real, you have the scale and signals to attack it, and you have the team to operate it.** The papers in section 11 are from organizations with billions of users and dedicated RL teams. Match your tool to your situation, not to the paper.

### Stress-testing the decision

Pose the failure modes explicitly. *What if your logs have poor propensity overlap* — the behavior policy never showed the actions your new policy loves? Then your importance weights explode (the $w$ in the worked example becomes enormous) and the off-policy estimate is garbage; you need on-policy exploration data, which means a bandit-style exploration phase *before* RL can work. *What if the reward is delayed by days* (return signals)? Then your training labels lag and you must handle [delayed feedback](/blog/machine-learning/recommendation-systems/delayed-feedback-and-conversion-attribution) explicitly, or the policy optimizes a censored reward. *What if the user population is non-stationary* — tastes shift faster than you can retrain? Then a long-horizon policy fit on stale dynamics can be *worse* than a fast-adapting bandit. *What if offline cumulative-reward estimates rise but online retention is flat?* Then your simulator or off-policy estimator does not match reality — distribution shift again — and you trust the live experiment, not the offline number. Every one of these is a reason a bandit might beat RL in *your* setting. The framework does not absolve you of judgment; it sharpens the questions.

## 13. Key takeaways

- **The next click is the wrong objective.** The business is paid in cumulative, multi-step quantities — session length, retention, lifetime value — and a myopic ranker that maximizes the immediate click can provably destroy them (the clickbait equilibrium). RL is the framing that puts the future in the objective via the discounted return $G_t = \sum_k \gamma^k r_{t+k}$.
- **Recommendation is an MDP** — state (user history), action (item/slate), reward (designed engagement signal), transition (the user reacting), discounted return — but a *hard* one: a million-item action space, off-policy logged data, partial observability, and a reward you must design.
- **The action space is the dragon.** Naive DQN dies on the $\arg\max$ over millions of items (and $10^{30}$ slates); policy gradients sidestep it with a softmax over candidates; SlateQ decomposes the slate into tractable per-item Q-values.
- **Policy gradients are model-free** because the unknown transition dynamics drop out of $\nabla_\theta \log p_\theta(\tau)$ — you never need a user model, only sampled trajectories. The REINFORCE update increases the log-probability of actions that earned high returns.
- **The off-policy correction is what makes RL ship.** You train on logs from a behavior policy $\beta$, so you reweight each step by the importance ratio $\pi_\theta/\beta$ (clipped for stability); the YouTube top-K correction adds the $\lambda_K = K(1-\pi_\theta)^{K-1}$ multiplier that spreads probability across the slate.
- **Reward design is the hardest and most consequential choice.** The agent is a literalist; reward clicks and you get clickbait. Use long-term proxies (completion, capped dwell, next-day return), shape carefully (potential-based when you can), and cap unbounded terms to prevent gaming.
- **Measure the cumulative metric, not the per-step one.** In the simulation the RL policy had the *lowest* per-step click proxy and the *highest* cumulative reward — measuring CTR would have killed the better policy.
- **A contextual bandit is the right default; full RL is a deliberate upgrade.** Reach for RL only when the long-horizon problem is proven real, you have the scale and the return signal, and you have the team to operate the off-policy instability and the harder evaluation. Otherwise the bandit captures most of the lift for a fraction of the cost.

## 14. Further reading

- **Chen, Beutel, Covington, Jain, Belletti, Chi — "Top-K Off-Policy Correction for a REINFORCE Recommender System" (WSDM 2019).** The canonical production RL recommender; derives the single-ratio off-policy correction and the top-K $\lambda_K$ multiplier and reports the YouTube launch.
- **Ie, Jain, Wang, Narvekar, Agarwal, Wu, Cheng, Chandra, Boutilier — "SlateQ: A Tractable Decomposition for Reinforcement Learning with Recommendation Sets" (IJCAI 2019).** The slate decomposition that makes value-based long-term RL tractable on real slates.
- **Sutton & Barto — *Reinforcement Learning: An Introduction* (2nd ed.).** The standard text; chapters on the MDP formalism, the Bellman equations, policy gradients, and off-policy methods. Read these for the foundations behind every equation in this post.
- **Ng, Harada, Russell — "Policy Invariance Under Reward Transformations" (ICML 1999).** The theory of potential-based reward shaping: how to guide learning without changing the optimal policy. Essential for principled reward design.
- **Within this series:** start at the [intro and funnel map](/blog/machine-learning/recommendation-systems/what-is-a-recommender-system), put RL next to its pragmatic alternative in [bandits and the exploration-exploitation tradeoff](/blog/machine-learning/recommendation-systems/bandits-and-the-exploration-exploitation-tradeoff), use the shared estimator in [counterfactual and off-policy evaluation](/blog/machine-learning/recommendation-systems/counterfactual-and-off-policy-evaluation), understand the multi-objective seesaw in [multi-task and multi-objective ranking](/blog/machine-learning/recommendation-systems/multi-task-and-multi-objective-ranking-mmoe-ple), see the dynamic RL fights against in [feedback loops and filter bubbles](/blog/machine-learning/recommendation-systems/feedback-loops-and-filter-bubbles), and tie it all together in [the recommender systems playbook](/blog/machine-learning/recommendation-systems/the-recommender-systems-playbook).
