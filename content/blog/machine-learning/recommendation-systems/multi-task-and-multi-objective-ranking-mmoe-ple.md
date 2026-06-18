---
title: "Multi-Task and Multi-Objective Ranking: MMoE and PLE"
date: "2026-06-19"
publishDate: "2026-06-19"
description: "Why a real feed ranker predicts click, like, watch-time, share, and not-regret all at once, why a single shared trunk forces a seesaw between conflicting tasks, and how MMoE, PLE, and ESMM fix it — with the gating math, a PyTorch implementation, a shared-bottom vs MMoE vs PLE ablation, and the serving fusion that turns several calibrated heads into one score."
tags:
  [
    "recommendation-systems",
    "recsys",
    "multi-task-learning",
    "mmoe",
    "ple",
    "esmm",
    "ranking",
    "mixture-of-experts",
    "machine-learning",
  ]
category: "machine-learning"
subcategory: "Recommendation Systems"
author: "Hiep Tran"
featured: true
readTime: 51
image: "/imgs/blogs/multi-task-and-multi-objective-ranking-mmoe-ple-1.png"
---

The launch that taught me multi-task ranking did not start as a multi-task problem. We had a video feed with a single, healthy ranker that predicted click-through rate, sorted by it, and shipped. Engagement was fine. Then a product reviewer pulled up the feed on a Friday and said the thing every recommender engineer dreads to hear from someone who outranks them: "Why is it all bait?" The feed had quietly learned that the surest way to maximize predicted clicks was to surface thumbnails that promised more than the video delivered. Clicks were up. Watch time per session was down. The seven-day return rate was sliding. The ranker was doing exactly what we asked — predict the click — and that was the whole problem. We had optimized one objective and silently sacrificed three others nobody had told the model to care about.

The fix was not a better CTR model. The fix was to stop pretending a feed has one objective. A real ranker has to predict, for the same candidate, the probability of a *click* and a *like* and the expected *watch time* and the probability of a *share* and — the one that matters most and is hardest to measure — the probability the user will *not regret* the recommendation, the absence of a quick swipe-away or a "show me less of this." These objectives are correlated but not aligned; pushing one up can drag another down. You cannot train five separate models cheaply, you cannot ignore four of them, and you cannot just average them in a single network and hope. This post is about the architecture and the math that let one model predict many things at once without the things fighting each other into a stalemate.

The reason this is hard, and the reason it has its own named failure mode, is a phenomenon the literature calls the **seesaw**: when tasks conflict, improving task A by tuning the shared model reliably hurts task B, and you spend your quarter sliding up one side and down the other with no net win. The whole subject of multi-task ranking is the science of getting off the seesaw — sharing what the tasks have in common to gain data efficiency and consistency, while protecting what they do not, so that both metrics can move up together. The two architectures that won this fight in production are **MMoE** (Multi-gate Mixture-of-Experts, Ma et al. 2018, used inside YouTube's ranker) and **PLE** (Progressive Layered Extraction, Tang et al. 2020 at Tencent, RecSys best paper), and the picture of how MMoE wires a shared input through a pool of experts into per-task gates and towers is the map for the whole post.

![A branching graph diagram showing input features flowing into three shared experts then into a per-task click gate and watch gate and finally into a click tower and a watch tower](/imgs/blogs/multi-task-and-multi-objective-ranking-mmoe-ple-1.png)

By the end of this post you will be able to explain precisely why a single shared trunk produces the seesaw and frame it as a gradient-conflict problem, derive the MMoE output equation with its softmax gates and see exactly how it lets low-correlation tasks route to different experts, explain how PLE's split into shared and task-specific experts across stacked layers fixes MMoE's expert collapse, implement MMoE and a PLE-style network in PyTorch for two tasks and run the shared-bottom vs MMoE vs PLE ablation that reveals the seesaw and its relief, and write the serving-time multi-objective fusion that turns several calibrated heads into the one score that decides the order. This post sits in the ranking layer of the funnel this series follows; if you want the single-objective foundation first it is in [the ranking model: CTR prediction foundations](/blog/machine-learning/recommendation-systems/the-ranking-model-ctr-prediction-foundations), the series intro and funnel map is [what is a recommender system](/blog/machine-learning/recommendation-systems/what-is-a-recommender-system), and the synthesis of everything is the [recommender systems playbook](/blog/machine-learning/recommendation-systems/the-recommender-systems-playbook).

## 1. Why one objective is never enough

Start from the surface, because the surface decides the objectives. An ads system has a relatively clean single objective — expected revenue, which factors into click probability times bid — and even there a careful team adds a "landing-page quality" or "post-click satisfaction" signal so the auction does not reward clickbait. A feed is the opposite: there is no single number that captures "good session." The platform wants the user to engage *now* (click, watch, like) and to come back *tomorrow* (not feel manipulated, not be fed a monotone of one creator, not regret the time spent). Those are different objectives on different time horizons, and the short-term ones are easy to measure while the long-term ones are the ones that actually pay the bills.

Concretely, a modern feed ranker typically predicts some subset of these per candidate:

- **p(click)** — will the user tap the item. Abundant label, fast feedback, easy to game.
- **p(like)** or **p(positive reaction)** — an explicit endorsement. Sparser than clicks, stronger signal of genuine value.
- **E[watch time]** or **p(complete)** — for video, how long they watch, or whether they finish. A regression target, and the single best near-term proxy for "was this actually good."
- **p(share)** or **p(comment)** — costly engagement that signals the item was worth passing on. Very sparse, very valuable.
- **p(not-regret)** — the absence of a fast skip, a hide, a "show me less," or a same-session unfollow. Often modeled as the complement of a "negative feedback" head. This is the guardrail that keeps the optimizer honest.

Each of these is a head — a small output layer predicting one quantity. The question is what feeds those heads. You have three structural choices, and the entire rest of this post is about choosing well among them.

**Option one: separate models.** Train an independent network per objective. This is the most flexible — each model can specialize completely — and it is what teams reach for first because it is the simplest to reason about. It is also the most expensive and, surprisingly, often the *least* accurate on the sparse tasks. You pay N times the training cost, N times the serving cost (every candidate goes through N forward passes), and N separate pipelines to monitor. Worse, the sparse tasks (share, comment) have so little label that a dedicated model overfits or underfits; they would have benefited from the representation the abundant click task learned, but a separate model cannot borrow it.

**Option two: one model, one objective, fuse heuristically.** Predict only p(click), then hand-craft a re-ranking rule that penalizes clickbait. This is where my feed started, and it does not scale: every guardrail is a hand-tuned heuristic that fights the model's own gradient, and the model keeps finding new ways to be the thing you penalized.

**Option three: one model, many heads, trained jointly.** This is **multi-task learning** (MTL): a single network with a shared body and one small tower per task, trained to minimize the *sum* of the per-task losses. This is the right default, and the reason is data efficiency and consistency. Data efficiency: the abundant click task teaches the shared body a representation of users and items that the sparse share task gets for free, so the sparse tasks improve. Consistency: a single representation cannot tell two contradictory stories about the same user, so the heads are coherent in a way independent models are not. The catch — the entire catch, the reason MMoE and PLE exist — is that "shared body" is where conflicting tasks go to fight.

It is worth being precise about *why* sharing helps, because "data efficiency and consistency" is the kind of phrase that sounds right and explains nothing. The data-efficiency argument is a statement about inductive bias and sample complexity. A neural network's body is a feature extractor; learning a good feature extractor requires a lot of labeled examples to constrain its many parameters. When five tasks share one body, every task's labels contribute to learning that body, so the body sees the *union* of all five tasks' supervision. The abundant click task — millions of labels — does most of the work pinning down a good representation of users and items; the sparse share task — thousands of labels — could never have learned that representation alone, but it does not have to, because it inherits it. In statistical terms, the shared body's effective sample size is the sum across tasks, while a separate model's is just its own. This is why MTL reliably *beats* separate models on the data-starved tasks even though it constrains them to share a body: the constraint is a feature, a regularizer that injects the abundant task's knowledge into the sparse one. The literature calls this **positive transfer**, and it is the upside that the seesaw's negative transfer is the dark mirror of.

The consistency argument is subtler and more about production sanity than raw accuracy. When you run five independent models, nothing forces their views of a user to agree. One model can believe a user is a sports fan while another, trained on a different objective with different sampling, believes the opposite, and the two predictions can be jointly incoherent in ways that show up as bizarre rankings (a candidate scored high on click and absurdly low on watch because the two models disentangle the user differently). A single shared representation cannot tell two contradictory stories about the same user, so the heads, whatever else they get wrong, get it wrong *coherently*. In a system where the heads are later fused into one score, this coherence is not a nicety — incoherent heads produce a fused score that is the product of two inconsistent worldviews, and that score is noise. There is one cost to consistency worth naming: a bug or a bad feature in the shared body now poisons *every* task at once, where separate models would have contained the blast radius. That is the trade — shared fate for shared learning — and it is why the shared body deserves the most careful monitoring in the system.

#### Worked example: the cost of separate models

Suppose your feed predicts four objectives and serves 500 candidates per request at p99 of 30 ms with a single multi-task model whose body is 2 dense layers of 512 units. The shared body is roughly 0.5 M parameters and the four towers add maybe 0.1 M total. Now do it with four separate models, each with its own 0.5 M body: that is 2.0 M parameters of body where MTL had 0.5 M, four times the training FLOPs, and four times the candidate forward passes at serving — so either your p99 blows past 30 ms or you provision four times the ranking fleet. On the sparse "share" task with, say, 0.3% positive rate, the separate model sees 1.5 positives per 500-impression request on average; the MTL model's body has already learned a rich user/item representation from the 5% click rate, so the share tower starts from a far better feature space and typically lands a meaningfully higher AUC for a fraction of the cost. That asymmetry — cheaper *and* better on the sparse tasks — is why MTL is the default and the only question worth arguing is which sharing architecture.

## 2. Shared-bottom MTL and the seesaw

The classic MTL architecture is the **shared-bottom** (Caruana, 1997): one shared trunk of dense layers transforms the input into a representation $h = f(x)$, and then each task $k$ has its own small tower $g_k$ producing its output $y_k = g_k(h)$. Train with the summed loss

$$
\mathcal{L} = \sum_{k=1}^{K} w_k\, \mathcal{L}_k\big(y_k, \hat{y}_k\big),
$$

where $\mathcal{L}_k$ is the per-task loss (binary cross-entropy for the classification heads, MSE or a Poisson/Tweedie loss for watch-time regression) and $w_k$ is a task weight. Gradients from every task flow back through the *same* shared bottom. When the tasks agree about what the representation should encode, this is wonderful — every task's gradient reinforces the others, and you get the data-efficiency win in full. When the tasks disagree, the shared bottom is forced to find a single representation that is a compromise between contradictory demands, and that compromise is the seesaw.

![A before and after comparison contrasting a single shared trunk that forces a compromise and a seesaw against an MMoE with an expert pool and per-task gates where both tasks improve](/imgs/blogs/multi-task-and-multi-objective-ranking-mmoe-ple-2.png)

It helps to make the conflict precise, because "the tasks disagree" is vague hand-waving until you see it in the gradients. Consider the shared bottom's parameters $\theta$. Task A wants to step $\theta$ in the direction of $-\nabla_\theta \mathcal{L}_A$; task B wants $-\nabla_\theta \mathcal{L}_B$. The actual step the optimizer takes is along the *sum* $-(\nabla_\theta \mathcal{L}_A + \nabla_\theta \mathcal{L}_B)$. Decompose this with the cosine of the angle between the two task gradients:

$$
\cos\phi = \frac{\nabla_\theta \mathcal{L}_A \cdot \nabla_\theta \mathcal{L}_B}{\|\nabla_\theta \mathcal{L}_A\|\,\|\nabla_\theta \mathcal{L}_B\|}.
$$

If $\cos\phi > 0$ the gradients point in broadly the same direction and both tasks make progress on every step — the happy case. If $\cos\phi < 0$ the gradients have a destructive component: the part of task A's gradient that opposes task B gets canceled, and vice versa, so the shared step makes less progress on *both* than either would alone. This is **gradient conflict**, and it is the mechanical root of negative transfer — the situation where adding a task makes the others worse instead of better. The seesaw you observe in metrics (task A up, task B down as you reweight) is gradient conflict expressed in the loss landscape: the Pareto frontier of (AUC_A, AUC_B) achievable by a shared bottom is bowed inward, so moving along it trades one for the other.

This is not a rare pathology; it is the default for tasks with low correlation. Click and watch-time on a feed are moderately correlated, so a shared bottom does okay. Click and "share" are weakly correlated — a thing many people click is not the same as a thing people share — so a shared bottom forces a representation that is a mushy average of "what gets tapped" and "what gets passed on," and both heads suffer. The empirical finding that launched MMoE is exactly this: in controlled experiments where you can *dial* the correlation between two synthetic tasks, shared-bottom performance degrades smoothly as task correlation drops, and the degradation is the seesaw.

There is a geometric way to see why the seesaw is unavoidable for a shared bottom, and it is worth holding onto because it tells you exactly what an architecture has to do to escape it. Picture the set of all achievable pairs (AUC_A, AUC_B) as you vary the shared-bottom model — its weights, its task loss weights, its size. This is a region in the (AUC_A, AUC_B) plane, and the upper-right boundary of that region is the **Pareto frontier**: the set of points where you cannot improve one task without giving up the other. For a shared bottom on conflicting tasks, that frontier is *concave* — it bows toward the origin — which means as you slide along it gaining on A you lose on B at a steepening exchange rate. Every weight setting you can choose lands you somewhere on (or inside) this bowed frontier, and reweighting just walks you along it. The single-task points sit *outside* the shared bottom's joint frontier on their respective axes, which is the formal statement of negative transfer: the joint model cannot reach the per-task accuracy that a dedicated model can. What MMoE and PLE do is not "find a better point on the frontier" — they *change the frontier itself*, pushing it outward so that points dominating the single-task baselines on both axes become reachable. That is the only kind of win that matters, and it is why reweighting a shared bottom is rearranging deck chairs.

The reason the frontier is concave traces straight back to the gradient-conflict picture. A point on the frontier is a stationary point of $\sum_k w_k \mathcal{L}_k$ for some weights $w_k$; changing the weights moves the stationary point. When the task gradients are aligned ($\cos\phi \approx 1$), the stationary points all cluster near the joint optimum and the frontier is a tight, near-convex blob in the good corner — you barely have to trade. When they conflict ($\cos\phi$ negative), the stationary points spread out along a bowed arc because the shared parameters can only satisfy one task's gradient direction at the expense of the other's, and the more you favor one the more aligned the shared step becomes with it and the more misaligned with the other. The curvature of the frontier *is* the conflict, quantified. So the engineering question "how do I get off the seesaw" has a precise answer: give the conflicting gradients somewhere separate to go, so the shared parameters no longer have to choose. Both architectures in this post are different ways of doing exactly that.

#### Worked example: quantifying the seesaw on conflicting tasks

Take two tasks and a shared-bottom model. Train it three ways by sweeping the task weight $w_A$ from 0.3 to 0.7 with $w_B = 1 - w_A$. At $w_A = 0.3$ you measure AUC_A = 0.788, AUC_B = 0.842. At $w_A = 0.5$: AUC_A = 0.795, AUC_B = 0.836. At $w_A = 0.7$: AUC_A = 0.801, AUC_B = 0.829. Notice the pattern — every 0.006 you gain on A costs you about 0.007 on B. The sum AUC_A + AUC_B is essentially flat (1.630, 1.631, 1.630). That is the seesaw in three numbers: the shared bottom has a *fixed budget of representational capacity* it can spend on one task or the other, and reweighting just slides the budget around. No weight setting moves both up. The only way to lift the frontier — to make AUC_A + AUC_B exceed 1.631 — is to change the *architecture* so the tasks stop competing for the same parameters. That is precisely what MMoE and PLE do.

## 3. MMoE: per-task gates over shared experts

The MMoE idea is elegant and, in hindsight, obvious: instead of one shared bottom, use a *pool of experts* — several independent subnetworks — and give each task its own **gate** that decides how much weight to put on each expert. Tasks that want different representations learn gates that emphasize different experts; tasks that agree learn similar gates and share experts. The sharing is no longer all-or-nothing; it is *soft and learned per task*.

Formally, let there be $n$ experts $f_1, \dots, f_n$, each a small subnetwork mapping the input $x$ to a vector $f_i(x) \in \mathbb{R}^d$. For each task $k$ there is a gating network $g^k$ that produces an $n$-dimensional weight vector over the experts, normalized by a softmax:

$$
g^k(x) = \mathrm{softmax}\big(W_{g^k}\, x\big), \qquad g^k_i(x) = \frac{\exp\big((W_{g^k} x)_i\big)}{\sum_{j=1}^{n} \exp\big((W_{g^k} x)_j\big)}.
$$

The task-$k$ representation is the gate-weighted sum of the expert outputs, and the task's tower $h_k$ turns that into the prediction:

$$
y_k = h_k\!\left( \sum_{i=1}^{n} g^k_i(x)\, f_i(x) \right).
$$

This is the equation to internalize. Read it from the inside out: every expert $f_i$ sees the full input and produces a representation; the gate $g^k$ produces a probability distribution over experts that depends on the input $x$ (so the routing can be *example-dependent* — a sports video can route differently than a cooking video); the task representation is the convex combination $\sum_i g^k_i(x) f_i(x)$; and the tower $h_k$ reads it out. There is one gate per task, hence *multi*-gate; the single-gate variant (MoE, one gate shared by all tasks) was the predecessor and is strictly weaker because it cannot route tasks differently.

![A branching graph diagram showing input features into three shared experts and then into a per-task click gate and watch gate feeding a click tower and a watch tower with task heads](/imgs/blogs/multi-task-and-multi-objective-ranking-mmoe-ple-1.png)

Why does this defeat the seesaw? Go back to the gradient-conflict view. In a shared bottom, every task's gradient hits the same $\theta$. In MMoE, task A's gradient flows through its gate $g^A$ and the experts $g^A$ emphasizes; task B's gradient flows through $g^B$ and the experts *it* emphasizes. If the tasks conflict, the gradients *select different experts*, so they no longer destructively interfere — the conflicting components land on different parameters. The gates learn to partition the expert pool along the lines of task disagreement. When the tasks agree, the gates converge onto overlapping experts and you still get the data-efficiency win. The softmax is what makes this differentiable and trainable end-to-end: the gate weights are continuous, so the model can smoothly discover the right routing by gradient descent rather than needing a discrete, non-differentiable expert assignment.

A subtle and important detail: the gate is a function of $x$, not a single learned constant. This means MMoE routes *per example*, not just per task. Two different requests can use different expert mixes for the same task, which lets a single gate represent "for power users emphasize expert 2, for new users emphasize expert 4." In practice the gate is usually a single linear layer plus softmax — deliberately cheap, because the experts carry the capacity and the gate only needs to *route*, not to compute.

A short lineage clears up a common confusion. The "mixture of experts" idea is old (Jacobs, Jordan, Nowlan, Hinton, 1991): a gating network softly combines specialist sub-models. The modern *sparse* MoE used in large language models takes this further — a top-k gate activates only a few experts per token so you can scale parameters without scaling compute. MMoE is a *different* application of the same primitive: its experts are not about scaling a single task's capacity, they are about *separating tasks*, and the multi-gate part — one gate per task rather than one shared gate — is the whole innovation. The single-gate predecessor, plain MoE applied to MTL, shares one gate across all tasks, so every task gets the *same* expert mixture and the experts cannot specialize per task; it is barely better than a shared bottom. The jump from one gate to one-gate-per-task is what lets the architecture model *task relationships* — tasks that agree learn similar gates and share experts, tasks that conflict learn divergent gates and partition them. The experts in MMoE are dense (all are computed for every example); the sparsity that matters for LLM-MoE compute is not the point here, because a feed ranker has only a handful of experts and the cost is dominated by the candidate count, not the expert count.

Why the softmax specifically, and not, say, a sigmoid per expert or a learned constant mixture? Three reasons. First, the softmax produces a *convex combination* — the weights are non-negative and sum to one — so the gated representation $\sum_i g^k_i(x) f_i(x)$ stays on the convex hull of the expert outputs and cannot blow up in scale, which keeps training stable. Second, it is differentiable everywhere, so the routing is learned end-to-end by the same backprop that trains the experts and towers; there is no separate, brittle assignment step. Third, the softmax's competition — raising one expert's weight necessarily lowers the others' — is exactly the inductive bias you want for *specialization*: it pressures each task's gate to commit to the experts that serve it rather than spreading mass uniformly. That same competitive pressure is, ironically, the seed of the expert-collapse failure in the next section, because nothing stops a gate from committing *all* its mass to one expert. The softmax gives you specialization and the risk of over-specialization in the same operator, which is the tension PLE resolves with structure.

#### Worked example: a 2-expert MMoE gated combination by hand

Take $n = 2$ experts and 2 tasks. For a given example $x$, suppose the experts output the 3-dim vectors $f_1(x) = (1.0,\ 0.0,\ -0.5)$ and $f_2(x) = (0.2,\ 0.8,\ 0.4)$. The click gate produces logits $(2.0,\ 0.0)$ over the two experts. Softmax: $\exp(2.0) = 7.389$, $\exp(0.0) = 1.0$, sum $= 8.389$, so $g^{\text{click}} = (0.881,\ 0.119)$ — the click task leans hard on expert 1. The click representation is $0.881 \cdot (1.0, 0.0, -0.5) + 0.119 \cdot (0.2, 0.8, 0.4) = (0.881 + 0.024,\ 0 + 0.095,\ -0.440 + 0.048) = (0.905,\ 0.095,\ -0.392)$. Now the watch gate produces logits $(0.0,\ 1.5)$ — it leans the other way. Softmax: $\exp(0) = 1$, $\exp(1.5) = 4.482$, sum $= 5.482$, $g^{\text{watch}} = (0.182,\ 0.818)$. The watch representation is $0.182 \cdot (1.0, 0.0, -0.5) + 0.818 \cdot (0.2, 0.8, 0.4) = (0.182 + 0.164,\ 0 + 0.654,\ -0.091 + 0.327) = (0.346,\ 0.654,\ 0.236)$. The two tasks now consume *different* representations of the same input — click gets a representation dominated by expert 1, watch gets one dominated by expert 2 — even though both experts saw the identical input. That divergence, learned automatically by the gates, is how MMoE sidesteps the forced compromise of a shared bottom.

## 4. The MMoE comparison and what it does not fix

Lay the three architectures side by side on the properties that decide which to ship.

![A matrix comparing shared bottom and MMoE and PLE across negative transfer and expert specialization and layers and a benchmark AUC outcome](/imgs/blogs/multi-task-and-multi-objective-ranking-mmoe-ple-3.png)

| Architecture | Sharing mechanism | Negative transfer | Params (relative) | When it wins |
| --- | --- | --- | --- | --- |
| Shared-bottom | One trunk, all tasks | High when tasks conflict | 1x (cheapest) | Highly correlated tasks |
| MMoE | Soft per-task gates over shared experts | Reduced | ~1.2–1.5x | Mixed correlation, 2–4 tasks |
| PLE / CGC | Shared + task-specific experts, stacked | Lowest | ~1.5–2x | Conflicting tasks, ≥3 tasks |

MMoE is a large, real improvement over the shared bottom, and for two or three moderately conflicting tasks it is often all you need. But it has two known weaknesses that PLE was designed to fix, and you will hit both if you push it.

**Weakness one: expert collapse.** Nothing in MMoE forces the gates to *use* all the experts. A gate is free to collapse onto a single expert — to put nearly all its softmax mass on one $f_i$ — especially early in training when one expert happens to be slightly better. When two tasks collapse onto the *same* expert, you are back to a shared bottom with extra steps; when a task collapses onto one expert, the others become dead capacity. The polarization of gates toward one or two experts is a documented MMoE failure mode, and it wastes the very capacity you added the experts for.

**Weakness two: every expert is shared, so there is no protected per-task capacity.** In MMoE all experts are in one pool and every task's gate can pull on every expert. There is no expert that *only* task A trains and only task A uses. That means even when the gates do specialize, the experts themselves are still pushed by every task's gradient (weighted by the gate), so a heavily-shared expert is still a site of gradient conflict. MMoE softens the conflict by routing; it does not give any task a private place to put the representation it cannot share.

**Weakness three: it is flat.** MMoE has a single layer of experts. Some task relationships are hierarchical — a low-level representation that all tasks share, then a mid-level split, then task-specific top layers — and a single flat gating layer cannot express that progression.

These are not reasons to skip MMoE; they are the reasons PLE exists. In fact the right way to read the next section is: PLE is MMoE plus protected per-task experts plus depth.

#### Worked example: detecting expert collapse from gate entropy

You can measure collapse directly. For a task's gate over $n$ experts, compute the average gate distribution over a validation batch, $\bar{g}^k_i = \frac{1}{N}\sum_x g^k_i(x)$, and its entropy $H = -\sum_i \bar{g}^k_i \log \bar{g}^k_i$. With $n = 4$ experts, the maximum entropy (uniform use) is $\log 4 = 1.386$ nats. Suppose you train MMoE and find the click gate has $\bar{g} = (0.92, 0.04, 0.03, 0.01)$, giving $H = -(0.92\ln 0.92 + 0.04\ln 0.04 + 0.03\ln 0.03 + 0.01\ln 0.01) = -(-0.0767 - 0.1288 - 0.1052 - 0.0461) = 0.357$ nats — only 26% of the maximum. That is collapse: the click task is effectively using one expert and ignoring three. The watch gate at $\bar{g} = (0.05, 0.10, 0.83, 0.02)$ has $H \approx 0.62$ nats, also collapsed onto a different expert. Two collapsed gates pointing at different experts is the *better* case (at least they are not fighting), but you have paid for four experts and are using two. Monitoring per-task gate entropy is the cheapest early-warning system for MMoE capacity waste, and a stubbornly low entropy is the signal to move to PLE's explicit shared-plus-specific structure.

## 5. PLE: progressive layered extraction

PLE (Tang et al., 2020) and its single-layer core CGC (Customized Gate Control) make the sharing structure *explicit*. Instead of one undifferentiated pool of experts, PLE has, for each task, a set of **task-specific experts** that only that task's gate can read, plus a set of **shared experts** that every task's gate can read. The task-specific experts give each task a protected place to encode what it cannot share; the shared experts carry the cross-task signal. And PLE stacks these blocks into *layers*, so the model can progressively separate shared from task-specific information with depth — hence "progressive layered extraction."

![A before and after comparison contrasting MMoE with one flat shared expert pool against PLE with separate shared experts and task-specific experts in stacked layers](/imgs/blogs/multi-task-and-multi-objective-ranking-mmoe-ple-4.png)

Concretely, in a single CGC layer with two tasks A and B, you have three expert groups: experts $E_A$ private to A, experts $E_B$ private to B, and experts $E_S$ shared. Task A's gate selects over $E_A \cup E_S$ (its own experts plus the shared ones) and produces a representation; task B's gate selects over $E_B \cup E_S$. Critically, task A's gate *cannot* see $E_B$ and task B's gate cannot see $E_A$. The gating equation is the same softmax-weighted sum as MMoE, just over the task's *allowed* set of experts:

$$
y_A = h_A\!\left( \sum_{f_i \in E_A \cup E_S} g^A_i(x)\, f_i(x) \right),
$$

and symmetrically for B. The shared experts also have their own gate that mixes all expert groups, because in a multi-layer PLE the shared experts at layer $\ell$ feed both the task experts and the shared experts at layer $\ell + 1$.

This structure fixes MMoE's two main weaknesses by construction. **Expert collapse is bounded:** even if task A's gate ignores the shared experts entirely, it still has its private $E_A$, so the task always has dedicated capacity; the shared experts cannot all be stolen by one task because each task only reaches them through its own gate alongside its private experts. **There is protected per-task capacity:** the private experts $E_A$ are trained *only* by task A's gradient, so they are a conflict-free home for the representation A needs and B would corrupt. The shared experts still see both tasks' gradients, but now the *part* of each task that conflicts can flow into the private experts instead, so the shared experts converge on the genuinely-shared signal.

The "progressive" part — stacking CGC layers into PLE — matters for three or more tasks with hierarchical relationships. Lower layers extract broadly-shared features; the gates at each layer decide how much of the shared representation to route into each task's path; by the top layer the task-specific experts have a representation tailored to their task. In the original paper, on Tencent's video platform with the VCR (view-completion-ratio) and VTR (view-through-rate) tasks — two tasks that exhibit a strong seesaw under MMoE — PLE moved *both* tasks above their single-task baselines simultaneously, which is the definitional win: it relieves the seesaw rather than just sliding along it. That result is why PLE won the RecSys 2020 best paper and became the default shared-architecture for conflicting-task ranking.

It is worth walking the multi-layer mechanism slowly because the word "progressive" is doing real work and is easy to skip past. In a two-layer PLE, the *first* CGC layer takes the raw input and produces, for each task, a representation from its private experts plus the shared experts — and separately produces a *shared* representation from a gate that mixes all expert groups. The second layer then takes, as input to task A's experts, A's representation from layer one; as input to the shared experts, the shared representation from layer one; and so on. The effect is a gradual purification: at the bottom, the shared experts carry a broad representation that everything draws from; as you climb, each task's path increasingly draws on its own private experts and the representation flowing into each task's tower becomes progressively more task-specific. This matches a real structure in the data — there is a layer of "what is this user and item about" that all engagement tasks share, then a layer of "what does *clicking* depend on versus what does *finishing* depend on" that starts to diverge, then task-specific top-level features. A single flat gating layer (MMoE) collapses all three of those levels into one routing decision; PLE lets the separation happen in stages, which is why it handles hierarchically-related tasks that MMoE cannot cleanly express.

A useful way to see the relationship: shared-bottom, MMoE, and PLE form a spectrum of *how much structure you impose on the sharing*. Shared-bottom imposes maximal sharing (one trunk, no choice). MMoE imposes soft, learned, flat sharing (one pool, per-task gates). PLE imposes structured sharing (explicit shared vs private groups, stacked). More structure means more parameters and a bigger hyperparameter search, but also stronger guarantees against the failure modes — PLE's private experts are a *hard* guarantee of per-task capacity, where MMoE's specialization is only a *soft tendency* the gates may or may not learn. You are trading flexibility for safety as you move along the spectrum, and the right point depends on how badly your tasks conflict.

There is a cost, and you should name it: PLE has more experts and more gates than MMoE, so it has more parameters and more FLOPs, and the structure (how many private experts, how many shared, how many layers) is a real hyperparameter search. A rough budget: if MMoE uses $n$ experts shared by $K$ tasks, a comparable single-layer CGC uses $K \cdot m_{\text{spec}} + m_{\text{shared}}$ experts (private per task plus shared), which for $K = 2$, one private expert each, and two shared experts is four experts versus MMoE's typical four — similar at one layer, but stacking two layers roughly doubles it again. The gates multiply too: MMoE has $K$ gates, CGC has $K$ task gates plus a shared gate per layer. For two well-correlated tasks the extra structure may not pay for itself over MMoE; for three-plus conflicting tasks it routinely does, and the parameter increase is small relative to the embedding tables that dominate a real ranker's memory anyway, so the FLOP cost (a few extra small MLP forward passes per candidate) is usually the binding constraint, not memory. The decision rule lives in section 12.

## 6. Implementing MMoE in PyTorch

Here is a clean MMoE for two tasks — say click and conversion — that you can copy and adapt. The shape is exactly the equation from section 3: a pool of expert MLPs, one linear-plus-softmax gate per task, and one tower per task. We will use it in the ablation in the next section.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class Expert(nn.Module):
    """One expert subnetwork: a small MLP shared across tasks via the gates."""
    def __init__(self, in_dim, hidden, out_dim, p_drop=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(p_drop),
            nn.Linear(hidden, out_dim), nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class Tower(nn.Module):
    """Per-task head: maps the gated representation to one logit."""
    def __init__(self, in_dim, hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),  # raw logit; BCEWithLogits applies the sigmoid
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class MMoE(nn.Module):
    def __init__(self, in_dim, n_experts=4, expert_hidden=128, expert_out=64,
                 tower_hidden=64, n_tasks=2):
        super().__init__()
        self.n_experts, self.n_tasks = n_experts, n_tasks
        self.experts = nn.ModuleList(
            [Expert(in_dim, expert_hidden, expert_out) for _ in range(n_experts)]
        )
        # one gate per task: linear over the input, softmax over experts
        self.gates = nn.ModuleList(
            [nn.Linear(in_dim, n_experts) for _ in range(n_tasks)]
        )
        self.towers = nn.ModuleList(
            [Tower(expert_out, tower_hidden) for _ in range(n_tasks)]
        )

    def forward(self, x):
        # stack expert outputs: (batch, n_experts, expert_out)
        expert_out = torch.stack([e(x) for e in self.experts], dim=1)
        logits = []
        gate_weights = []  # keep for entropy monitoring
        for t in range(self.n_tasks):
            g = F.softmax(self.gates[t](x), dim=-1)          # (batch, n_experts)
            gate_weights.append(g)
            # gate-weighted sum over experts -> (batch, expert_out)
            rep = torch.einsum("be,beo->bo", g, expert_out)
            logits.append(self.towers[t](rep))
        return logits, gate_weights
```

The `einsum("be,beo->bo", g, expert_out)` is the literal $\sum_i g^k_i(x) f_i(x)$ from the gating equation — it contracts the expert axis $e$ against the gate weights. Returning `gate_weights` lets you compute the gate entropy from the worked example in section 4 to watch for collapse.

The training loop sums the per-task losses. Two details matter: the per-task loss should be computed only over rows where that task's label is *valid* (conversion is only defined for clicked impressions unless you use ESMM — see section 8), and you weight the losses so neither task dominates by virtue of scale.

```python
def train_step(model, batch, optimizer, task_weights=(1.0, 1.0)):
    x, y_click, y_conv, conv_mask = batch  # conv_mask: 1 where conversion label exists
    logits, _ = model(x)
    loss_click = F.binary_cross_entropy_with_logits(logits[0], y_click)
    # conversion loss only over rows with a valid conversion label
    conv_logits = logits[1][conv_mask.bool()]
    conv_labels = y_conv[conv_mask.bool()]
    loss_conv = F.binary_cross_entropy_with_logits(conv_logits, conv_labels)
    loss = task_weights[0] * loss_click + task_weights[1] * loss_conv
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item(), loss_click.item(), loss_conv.item()
```

A note on the task weighting `task_weights`: tuning these by hand is a chore, and the literature offers principled alternatives. **Uncertainty weighting** (Kendall et al., 2018) learns a per-task weight $\frac{1}{2\sigma_k^2}$ tied to each task's homoscedastic uncertainty, adding a $\log \sigma_k$ regularizer so the model cannot just shrink all losses to zero. **GradNorm** (Chen et al., 2018) rebalances the weights so every task trains at a similar rate. **PCGrad** (Yu et al., 2020) attacks the gradient conflict directly: when two task gradients have negative cosine, it *projects* each onto the normal plane of the other, removing the destructive component before the step. These compose with MMoE/PLE — the architecture reduces where conflict can happen, and the gradient-surgery methods clean up what remains.

## 7. The ablation: shared-bottom vs MMoE vs PLE

To see the seesaw and its relief in numbers you need a two-task dataset where the tasks genuinely diverge. The community's standard MTL proxy is the **UCI Census-Income (KDD)** dataset, used in both the MMoE and PLE papers: predict two binary attributes from the same demographic features — for example *income > 50K* and *marital status (never-married vs not)* — which are correlated but not aligned, so a shared bottom seesaws between them. (For a recommender-native version, **Ali-CCP** / Taobao with the click and conversion tasks is the production-grade equivalent; Census is the reproducible teaching set.)

The PLE-style network reuses the MMoE pieces with the shared/specific split:

```python
class CGC(nn.Module):
    """Single Customized Gate Control layer for two tasks."""
    def __init__(self, in_dim, n_shared=2, n_specific=1,
                 expert_hidden=128, expert_out=64, n_tasks=2):
        super().__init__()
        self.n_tasks = n_tasks
        self.shared = nn.ModuleList(
            [Expert(in_dim, expert_hidden, expert_out) for _ in range(n_shared)]
        )
        # task-specific experts: a list per task
        self.specific = nn.ModuleList([
            nn.ModuleList([Expert(in_dim, expert_hidden, expert_out)
                           for _ in range(n_specific)])
            for _ in range(n_tasks)
        ])
        # each task gate selects over (its specific experts + shared experts)
        self.gates = nn.ModuleList(
            [nn.Linear(in_dim, n_specific + n_shared) for _ in range(n_tasks)]
        )

    def forward(self, x):
        shared_out = [e(x) for e in self.shared]
        task_reps = []
        for t in range(self.n_tasks):
            spec_out = [e(x) for e in self.specific[t]]
            experts = torch.stack(spec_out + shared_out, dim=1)  # (b, k, out)
            g = F.softmax(self.gates[t](x), dim=-1)
            rep = torch.einsum("be,beo->bo", g, experts)
            task_reps.append(rep)
        return task_reps


class PLE(nn.Module):
    def __init__(self, in_dim, expert_out=64, tower_hidden=64, n_tasks=2):
        super().__init__()
        # one CGC layer here; stack two for the "progressive" version
        self.cgc = CGC(in_dim, expert_out=expert_out, n_tasks=n_tasks)
        self.towers = nn.ModuleList(
            [Tower(expert_out, tower_hidden) for _ in range(n_tasks)]
        )

    def forward(self, x):
        reps = self.cgc(x)
        return [self.towers[t](reps[t]) for t in range(len(self.towers))], None
```

Train all four models — single-task baselines (one network per task), shared-bottom, MMoE, PLE — on the identical temporal/random split with the same optimizer, the same total parameter budget where feasible, and the same epochs, then report per-task AUC. Computing AUC honestly: use a held-out test split the model never trained on, compute the full AUC (not a sampled approximation), and average over a few seeds because MTL training has real run-to-run variance and a 0.001 AUC difference inside the noise band is not a result.

![A matrix of per-task AUC results for single-task and shared bottom and MMoE and PLE showing the shared bottom seesaw and PLE lifting both tasks](/imgs/blogs/multi-task-and-multi-objective-ranking-mmoe-ple-8.png)

Representative numbers in the shape the papers report (Census-Income, income and marital tasks, two-attribute setup, AUC, mean over seeds):

| Model | Income AUC | Marital AUC | Sum | Verdict |
| --- | --- | --- | --- | --- |
| Single-task (2 nets) | 0.9410 | 0.9840 | 1.9250 | Baseline, 2x cost |
| Shared-bottom | 0.9380 | 0.9860 | 1.9240 | Seesaw: income down, marital up |
| MMoE | 0.9430 | 0.9870 | 2.0300 | Both above single-task |
| PLE | 0.9450 | 0.9890 | 2.0340 | Best on both, seesaw relieved |

Read the table the way you would read it on a launch review. The shared-bottom row is the seesaw caught in the act: marital ticks up but income drops *below* its single-task baseline — adding the second task made the first worse, the textbook negative-transfer signature. MMoE pulls both tasks above their single-task baselines, the data-efficiency win finally realized because the gates kept the conflicting gradients off each other. PLE edges MMoE on both tasks because the task-specific experts give each task a protected representation and the shared experts carry the genuine cross-task signal. The exact decimals depend on the split and the architecture sizing, but the *ordering* — shared-bottom seesaws, MMoE fixes it, PLE fixes it slightly better — is the robust, reproducible result across the literature and your own runs.

![A before and after comparison contrasting a seesaw where one task rises while the other falls against a balanced multi-task result where both tasks improve](/imgs/blogs/multi-task-and-multi-objective-ranking-mmoe-ple-6.png)

The honest caveat: on Census the tasks are only mildly conflicting, so the gaps are small (thousandths of AUC). On a real feed with click vs share, or VTR vs VCR as in the PLE paper, the seesaw is far more violent and the MMoE-and-PLE gains over shared-bottom are correspondingly larger — and, crucially, they show up *online* as simultaneous lifts on conflicting engagement metrics that a shared-bottom could only trade against each other.

A word on measuring this honestly, because a multi-task ablation is unusually easy to fool yourself with. First, hold *everything* constant except the architecture: same features, same split, same optimizer, same epochs, same total parameter budget within reason, same per-task loss weights. If you let the MMoE run use different loss weights than the shared bottom, you have confounded the architecture comparison with a reweighting that — per the seesaw — moves the metrics on its own, and you can no longer attribute the gain. Second, average over seeds and report the variance; MTL training has real run-to-run noise from the random gate initialization, and a 0.001 AUC "win" inside a 0.002 standard deviation is nothing. Third, watch *both* tasks, always together — the whole point is that a single-task metric can rise while the joint result is a seesaw, so a per-task AUC in isolation is meaningless without its partner. Fourth, if the eventual system fuses the heads, the offline metric that actually predicts online success is not any single AUC but a *fused* offline metric (the fusion score's ranking quality against logged engagement), because that is what you serve; per-task AUC is diagnostic, not the launch criterion. Fifth, prefer a temporal split (train on past, test on future) over a random split for any production claim, because a random split leaks future information through shared users and inflates every model equally, hiding the architecture differences that a temporal split exposes. None of this is exotic; it is just the discipline that separates a result you can ship on from a number that looks good in a slide.

## 8. ESMM and the conversion sample-selection-bias chain

There is a second, distinct multi-task problem that is not about conflicting tasks but about *biased data*, and it has its own canonical solution: **ESMM** (Entire Space Multi-task Model, Ma et al., 2018, Alibaba). The setting is the click-to-conversion (CTR → CVR) chain. You want to predict conversion rate (CVR) — the probability of a purchase *given* a click. The trap is in the words "given a click": conversion is only *observed* on impressions that were clicked, so the natural training data for a CVR model is the click-only subset. But at serving you must score CVR for *every* impression in the candidate set, the vast majority of which were never clicked. The model trained on clicked impressions and applied to all impressions is suffering **sample selection bias (SSB)**: the training distribution (clicked) does not match the inference distribution (all impressions), so the CVR predictions are systematically off on the un-clicked-looking candidates that dominate at serving. A second pain compounds it: clicks are already sparse, so the click-only CVR training set is *tiny*, and the model is data-starved.

ESMM's fix is a clean identity. Define three quantities over the *entire impression space*: $\text{pCTR} = p(\text{click} \mid \text{impression})$, $\text{pCVR} = p(\text{buy} \mid \text{click})$, and $\text{pCTCVR} = p(\text{click and buy} \mid \text{impression})$. By the chain rule of probability:

$$
\underbrace{p(\text{buy}, \text{click}\mid x)}_{\text{pCTCVR}} = \underbrace{p(\text{click}\mid x)}_{\text{pCTR}} \times \underbrace{p(\text{buy}\mid \text{click}, x)}_{\text{pCVR}}.
$$

ESMM builds two towers over a *shared embedding* — a CTR tower predicting pCTR and a CVR tower predicting pCVR — and multiplies them to get pCTCVR. Then it trains on two losses defined over *all impressions*: the CTR loss (label = clicked, available for every impression) and the CTCVR loss (label = clicked-and-bought, also available for every impression). The CVR tower is never trained on a click-only loss directly; it is trained *implicitly* through the CTCVR product over the entire space. This is the trick — the CVR network gets a gradient on every impression, not just clicked ones, so it learns over the inference distribution and the sample-selection bias is gone. As a bonus, the shared CTR/CVR embedding is trained on the abundant click data, so the data-sparsity problem on CVR is relieved too.

![A branching graph diagram showing all impressions into a shared embedding then into a CTR tower and a CVR tower whose product forms CTCVR feeding a joint CTR plus CTCVR loss](/imgs/blogs/multi-task-and-multi-objective-ranking-mmoe-ple-7.png)

In code the two-loss structure is the whole point:

```python
class ESMM(nn.Module):
    def __init__(self, in_dim, hidden=128):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU())
        self.ctr_tower = nn.Sequential(nn.Linear(hidden, 64), nn.ReLU(),
                                       nn.Linear(64, 1))
        self.cvr_tower = nn.Sequential(nn.Linear(hidden, 64), nn.ReLU(),
                                       nn.Linear(64, 1))

    def forward(self, x):
        h = self.shared(x)
        p_ctr = torch.sigmoid(self.ctr_tower(h)).squeeze(-1)   # over all space
        p_cvr = torch.sigmoid(self.cvr_tower(h)).squeeze(-1)   # over all space
        p_ctcvr = p_ctr * p_cvr                                # the identity
        return p_ctr, p_cvr, p_ctcvr


def esmm_loss(p_ctr, p_ctcvr, y_click, y_click_and_buy, eps=1e-7):
    # both labels are defined over EVERY impression (no click-only subset)
    p_ctr = p_ctr.clamp(eps, 1 - eps)
    p_ctcvr = p_ctcvr.clamp(eps, 1 - eps)
    loss_ctr = F.binary_cross_entropy(p_ctr, y_click)
    loss_ctcvr = F.binary_cross_entropy(p_ctcvr, y_click_and_buy)
    return loss_ctr + loss_ctcvr   # CVR is supervised only through the product
```

On Alibaba's production data, the original ESMM paper reported a clear AUC lift on CVR over the click-only-trained baseline (on the order of a couple of AUC points on their dataset), precisely because the click-only baseline was both biased and data-starved and ESMM was neither. The detail that matters for your own build: the CVR tower's *raw* output `p_cvr` is the calibrated conversion probability you serve, but you never trained a loss on it directly — you trained on `p_ctcvr` — so you must validate that `p_cvr` is itself well-behaved (clamp it, monitor its calibration), because nothing in the loss forced it to be anything except a factor that makes the product correct. ESMM stacks with MMoE/PLE in modern systems: the shared embedding can itself be an MMoE/PLE body, and the CTCVR identity sits on top. The delayed-feedback wrinkle — conversions that arrive hours or days after the click, so your label is censored at training time — is a related but separate problem covered in the delayed-feedback post linked at the end.

#### Worked example: why click-only CVR is biased, in numbers

Suppose 1,000 impressions, 50 of them clicked (5% CTR), and among clicked ones 10 converted (20% CVR on clicks). A click-only CVR model trains on those 50 rows. The 950 un-clicked impressions never appear in CVR training, but at serving you score CVR for *all 1,000*. If the un-clicked impressions are systematically different — lower-intent users, off-target items — the click-only model has never seen their feature distribution and extrapolates badly; its CVR on those is unreliable, often biased high because the training set was the high-intent click survivors. ESMM instead supervises CVR through pCTCVR over all 1,000 rows. The CTCVR positive rate is $10/1000 = 1\%$, the product $0.05 \times 0.20 = 0.01$ matches, and the gradient touches every impression, so the CVR tower learns a function valid over the full 1,000-row inference distribution rather than the 50-row clicked one. Same data, no extra labels, bias removed by construction.

## 9. Combining the predictions: multi-objective serving fusion

Training a multi-task model gives you several calibrated heads per candidate. Serving requires *one number* to sort by. The step that turns many heads into one score is the **multi-objective fusion**, and it is where the business actually expresses what it wants. The standard form is a *weighted product* of the calibrated heads:

$$
\text{score} = p_{\text{click}}^{\,w_1} \cdot E[\text{watch}]^{\,w_2} \cdot p_{\text{share}}^{\,w_3} \cdot (1 - p_{\text{skip}})^{\,w_4} \cdots
$$

A weighted product (rather than a weighted sum) is the common choice for a reason worth understanding. Taking logs, the product becomes a weighted *sum of log-scores*, $\log \text{score} = \sum_k w_k \log(\text{head}_k)$, which means each weight $w_k$ scales a *relative* (multiplicative) change in that head, not an absolute one. That is usually what you want: a candidate that is twice as likely to be shared should get the same proportional boost whether its click probability is high or low. It also makes the fusion robust to the different *scales* of the heads — a probability in [0,1] and an expected watch time in seconds do not need to be normalized to the same range because the exponent operates on each in its own units (in practice you still bound or log-transform watch time so a single huge value cannot dominate). A sum, by contrast, requires you to normalize every head to a comparable scale first, and it lets a single dominant head swamp the others.

![A layered stack diagram showing calibrated heads flowing into a weighted product that is steered by business tuning and emits a final score that sorts candidates](/imgs/blogs/multi-task-and-multi-objective-ranking-mmoe-ple-5.png)

The exponents $w_k$ are the **business control surface**. They are not learned by the model; they are *set by the product goal* and tuned online. If retention is the quarter's priority, you raise the weight on watch-time and the not-regret head and lower the weight on raw click. If you are launching a creator economy and want sharing, you raise $w_3$. The tuning loop is: pick a weight vector, run an A/B test, read the trade-off on the real online metrics (this is where the seesaw becomes a *business* decision rather than a modeling artifact — you are explicitly choosing where on the engagement-vs-retention frontier to sit), and iterate. Some teams automate this with a bandit or Bayesian-optimization loop over the weights against a composite north-star metric; most start with hand-tuned weights and a careful A/B discipline.

The reason the heads must be **calibrated** for this to work is fundamental, and it is why the calibration post is a prerequisite. The fusion multiplies the heads, so if $p_{\text{click}}$ is systematically inflated by 20% relative to $E[\text{watch}]$, the click factor silently dominates the product regardless of the weights you set — your $w_1$ no longer means what you think it means. Only if each head is a *trustworthy probability or expectation on its own scale* does the weighted product compose them correctly and do the exponents have a stable interpretation. This is the bridge between this post and [calibration and the prediction you can trust](/blog/machine-learning/recommendation-systems/calibration-and-the-prediction-you-can-trust): multi-task ranking *produces* the multiple heads, calibration is what makes them *fuseable*.

#### Worked example: computing a serving fusion score for two candidates

Take a video feed with fusion $\text{score} = p_{\text{click}}^{1.0} \cdot E[\text{watch}]^{0.6} \cdot (1 - p_{\text{skip}})^{0.8}$, where $E[\text{watch}]$ is in seconds. Candidate X (a clickbait thumbnail) has $p_{\text{click}} = 0.30$, $E[\text{watch}] = 12$ s, $p_{\text{skip}} = 0.55$. Candidate Y (a quieter but satisfying video) has $p_{\text{click}} = 0.18$, $E[\text{watch}] = 95$ s, $p_{\text{skip}} = 0.10$. Compute X: $0.30^{1.0} = 0.30$; $12^{0.6} = e^{0.6 \ln 12} = e^{0.6 \cdot 2.485} = e^{1.491} = 4.44$; $(1-0.55)^{0.8} = 0.45^{0.8} = e^{0.8 \ln 0.45} = e^{0.8 \cdot (-0.799)} = e^{-0.639} = 0.528$. Score X $= 0.30 \cdot 4.44 \cdot 0.528 = 0.703$. Compute Y: $0.18^{1.0} = 0.18$; $95^{0.6} = e^{0.6 \ln 95} = e^{0.6 \cdot 4.554} = e^{2.732} = 15.36$; $(1-0.10)^{0.8} = 0.90^{0.8} = e^{0.8 \cdot (-0.105)} = e^{-0.084} = 0.919$. Score Y $= 0.18 \cdot 15.36 \cdot 0.919 = 2.54$. Candidate Y wins by a wide margin — 2.54 vs 0.70 — even though X has nearly double the click probability, because the watch-time and not-skip factors reward Y for actually delivering. A pure CTR ranker would have put X first; the fused multi-objective ranker puts the genuinely-better video first. Now notice the control: if the business cuts the watch-time exponent to $w_2 = 0.2$, recompute X's watch factor $12^{0.2} = e^{0.2 \cdot 2.485} = 1.65$ and Y's $95^{0.2} = e^{0.2 \cdot 4.554} = 2.49$, giving Score X $= 0.30 \cdot 1.65 \cdot 0.528 = 0.261$ and Score Y $= 0.18 \cdot 2.49 \cdot 0.919 = 0.412$ — Y still wins but the margin narrows sharply. That single exponent is the dial the business turns to choose how hard the feed leans toward retention over raw clicks.

## 10. Case studies: MMoE at YouTube, PLE at Tencent, ESMM at Alibaba

These three architectures are not academic; each shipped at a platform you have used, and the published results anchor the claims in this post.

**MMoE, Ma et al. 2018 (Google / YouTube).** The originating paper "Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts" (KDD 2018) introduced MMoE and demonstrated it on synthetic data with controllable task correlation (showing shared-bottom degrades as correlation drops while MMoE holds up), on the UCI Census-Income MTL benchmark, and on a large-scale Google content recommendation problem. MMoE then became a core component of YouTube's production ranking system, described in "Recommending What Video to Watch Next: A Multitask Ranking System" (Covington-lineage work, Zhao et al., RecSys 2019), where the ranker predicts multiple engagement and satisfaction objectives (clicks, watch time, likes, dismissals) with an MMoE body and explicitly addresses position bias with a shallow side-tower. The takeaway the YouTube paper emphasizes — and the reason this whole post matters — is that separating *engagement* objectives from *satisfaction* objectives and modeling them as distinct tasks is what let them optimize for long-term user value rather than short-term clicks, which is exactly the "all bait" problem from the intro, solved at scale.

**PLE, Tang et al. 2020 (Tencent).** "Progressive Layered Extraction (PLE): A Novel Multi-Task Learning Model for Personalized Recommendations" (RecSys 2020, best paper) named and quantified the seesaw phenomenon directly on Tencent's video platform, with the VTR (view-through-rate) and VCR (view-completion-ratio) tasks that exhibit a strong negative correlation. They showed that prior MTL models including MMoE could not improve both tasks simultaneously — they slid along the seesaw — whereas PLE's explicit shared-plus-task-specific experts with progressive separation lifted *both* tasks above their single-task baselines at once. The paper reports both offline AUC gains and meaningful *online* lifts on view count and watch time in Tencent's live system, which is the proof that relieving the seesaw is not just an offline curiosity but a real engagement win.

**ESMM, Ma et al. 2018 (Alibaba).** "Entire Space Multi-Task Model: An Effective Approach for Estimating Post-Click Conversion Rate" (SIGIR 2018) introduced the CTR → CVR entire-space formulation and the CTCVR identity. On Alibaba's production e-commerce data and the public Ali-CCP (Alibaba Click and Conversion Prediction) dataset it released, ESMM beat click-only-trained CVR baselines on CVR AUC by addressing both sample-selection bias and data sparsity simultaneously. Ali-CCP became a standard public benchmark for the CTR/CVR multi-task setting, and ESMM (and its successors like ESM2 and the entire-space variants) remains the reference approach for the conversion-attribution chain in commerce ranking.

The pattern across all three: the architecture is the easy part to copy; the hard part is *deciding what your objectives are and how to fuse them*, which is a product question as much as a modeling one. Every one of these systems pairs the multi-task body with an explicit, tuned fusion and a guardrail objective (satisfaction, completion, not-regret) that keeps the engagement objectives honest.

## 11. Stress-testing the design

A design is only trustworthy after you have poked it where it might break. Here are the failure modes I check before shipping a multi-task ranker.

**What if the tasks are highly correlated?** Then you do not need MMoE or PLE — a shared bottom is cheaper and just as good, because there is no seesaw to relieve. Measure the correlation first (train single-task models, look at whether adding the second task to a shared bottom helps or hurts each). Reaching for PLE when a shared bottom suffices is over-engineering: you pay extra parameters and a hyperparameter search for no win. The architectures earn their keep specifically when tasks *conflict*.

**What if one task has 100x more data than another?** This is the click-vs-share situation, and the summed-loss MTL handles it gracefully *if* you weight the losses so the sparse task is not drowned, and *especially* if the architecture lets the sparse task borrow the abundant task's representation (which MMoE/PLE shared experts do). Watch for the sparse task's tower silently learning the prior (predicting the base rate for everyone) — that shows up as a near-0.5 AUC on the sparse task and means its gradient is too weak; raise its loss weight or use uncertainty weighting.

**What if a new task is added mid-life?** Adding a fifth head to a four-task model is a real operational event. With a shared bottom you risk disturbing the four existing tasks (the new task's gradient now flows through the shared trunk they all depend on). With PLE you add task-specific experts for the new task and let its gate pull on the existing shared experts, which insulates the incumbents — this composability is an underrated practical advantage of the explicit shared/specific split.

**What if offline AUC goes up but online engagement is flat?** The classic recsys reality gap, and multi-task makes it subtler because you have several offline metrics and one online north star. A common cause: you improved a task whose offline AUC the fusion barely uses (low weight), or you improved a task that is a poor proxy for the online metric. Another: the fusion weights are stale relative to the new heads' calibration. The discipline is to A/B the *fused* ranker, not to ship on offline per-task AUC alone, and to re-tune the fusion weights whenever the heads' calibration shifts. This is the [offline vs online](/blog/machine-learning/recommendation-systems/offline-vs-online-the-two-worlds-of-recsys) gap wearing a multi-task costume.

**What if the gates collapse?** Monitor per-task gate entropy (the section 4 worked example). If MMoE gates collapse onto one expert and stay there, you have effectively a shared bottom; try more experts, add a small entropy regularizer on the gates, or move to PLE where the task-specific experts give each task guaranteed private capacity regardless of gate behavior.

**What if a guardrail task is gameable?** The not-regret head is your honesty guardrail, but if it is a weak signal (skip rate is noisy) the optimizer can learn to satisfy the *measured* guardrail while still degrading the *true* user experience. The defense is a richer satisfaction signal — survey-based satisfaction labels, longer-horizon return-rate labels — fed as additional tasks, exactly as YouTube does with its satisfaction objectives. No fusion weight saves you if every objective you can measure is gameable; the model will find the gap.

**What if the multi-task model interacts badly with the feedback loop?** This is the most insidious one and the reason this series keeps returning to the serve-log-train cycle. Your multi-task ranker's output, fused into a score, decides what gets shown; what gets shown decides what gets clicked, watched, and shared; those logged engagements become tomorrow's training labels for every one of your tasks. If the fusion currently over-weights click, the system shows clicky items, the logs fill with clicks on clicky items, and the next model learns an even stronger "clicky items get clicked" signal — a self-reinforcing loop that can quietly drift the whole feed toward whatever the current fusion rewards, *regardless of what the architecture can express*. The multi-task body is not the culprit here; the fusion and the loop are. The practical defenses are the ones from the bias and feedback-loop posts — log the propensity (the score that caused the impression) so you can debias the training data, inject exploration so the logs are not purely exploitative, and watch long-horizon metrics that the short-term tasks cannot directly inflate. A multi-task ranker with a good architecture and a thoughtless fusion plus an uncorrected loop will still walk itself into a clickbait corner; the architecture buys you the *ability* to optimize for satisfaction, but only the fusion and the loop discipline make it actually happen.

## 12. When to go multi-task, and which architecture

Here is the decision the section titles have been building toward, stated plainly.

**Go multi-task** when you genuinely care about more than one objective for the same item — which is almost every feed, video, and commerce surface — and you want data efficiency (sparse tasks borrowing from abundant ones) and consistency (coherent heads from one representation). Do *not* go multi-task if you truly have one objective and can express your guardrails in the candidate generation or a hard re-ranking rule; the extra heads are then cost without benefit.

**Use a shared-bottom** when your tasks are strongly positively correlated. It is the cheapest, it gets the full data-efficiency win, and there is no seesaw to fix. Verify the correlation empirically before assuming it.

**Use MMoE** when you have two to four tasks of mixed correlation and a shared bottom shows a seesaw. It is a modest parameter increase over shared-bottom, it is simple to implement and tune, and it captures most of the available gain. This is the right default for the majority of multi-task rankers and the reason it is in YouTube's stack.

**Use PLE** when tasks clearly conflict (you have measured a strong seesaw), when you have three or more tasks, or when you anticipate adding tasks over time and want the insulation of task-specific experts. It costs more parameters and a structure search (how many shared vs specific experts, how many layers) but it most reliably lifts conflicting tasks together, which is why it won RecSys and is the production default for the hard cases at Tencent.

**Use ESMM** (orthogonally — it composes with the above) whenever you have a *chained* objective with sample-selection bias, the canonical case being CTR → CVR where conversion is only observed on clicks. ESMM is not an alternative to MMoE/PLE; it is the entire-space training formulation you put *on top of* a shared or gated body to estimate the post-click objective without bias.

And on the fusion: always serve a *weighted product of calibrated heads*, tune the exponents online against your north-star metric through disciplined A/B tests, and keep a guardrail objective (satisfaction, completion, not-regret) in the mix so the engagement objectives cannot run away into clickbait. The architecture decides whether the heads are *good*; the fusion decides whether the *system* serves the user.

## Key takeaways

- **Real rankers are multi-objective.** A feed must predict click, like, watch-time, share, and not-regret at once; optimizing one in isolation silently sacrifices the others (the "all bait" failure).
- **Multi-task learning is the right default** over separate models — it is cheaper and, on sparse tasks, more accurate, because the abundant task teaches the shared representation.
- **The shared-bottom seesaw is gradient conflict.** When task gradients point in opposing directions ($\cos\phi < 0$) the shared step makes both tasks worse; the achievable (AUC_A, AUC_B) frontier bows inward and reweighting just trades one for the other.
- **MMoE defeats the seesaw with soft per-task routing:** $y_k = h_k(\sum_i g^k_i(x) f_i(x))$ with softmax gates lets conflicting tasks select different experts so their gradients stop interfering.
- **PLE fixes MMoE's expert collapse and lack of protected capacity** by splitting experts into shared plus task-specific groups across stacked layers — the explicit structure that lifts conflicting tasks above their single-task baselines together.
- **ESMM removes CVR sample-selection bias** by training over the entire impression space via the identity pCTCVR = pCTR x pCVR, so the CVR tower gets gradient on every impression, not just clicked ones.
- **Calibrate before you fuse.** The serving score is a weighted product of heads; if the heads are miscalibrated the fusion weights stop meaning what you set them to.
- **The fusion exponents are the business control surface** — set by the product goal, tuned online, with a guardrail objective that keeps engagement honest.
- **Pick by conflict and task count:** shared-bottom for correlated tasks, MMoE for mixed 2–4 tasks, PLE for conflicting or many tasks, ESMM on top for chained biased objectives.
- **Ship on the fused, online metric, not offline per-task AUC** — and monitor gate entropy to catch expert collapse early.

## Further reading

- Ma, Zhao, Yi, Chen, Hong, Chi, "Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts" (KDD 2018) — the MMoE paper, with the synthetic task-correlation experiments and the Census-Income benchmark.
- Tang, Liu, Zhao, Gao, "Progressive Layered Extraction (PLE): A Novel Multi-Task Learning Model for Personalized Recommendations" (RecSys 2020, best paper) — names the seesaw, introduces CGC and PLE, reports offline and online lifts at Tencent.
- Zhao et al., "Recommending What Video to Watch Next: A Multitask Ranking System" (RecSys 2019) — YouTube's production MMoE ranker, engagement vs satisfaction objectives, position-bias side tower.
- Ma, Zhu, Yang, Wang, Zhu, Gai, "Entire Space Multi-Task Model: An Effective Approach for Estimating Post-Click Conversion Rate" (SIGIR 2018) — ESMM, the CTCVR identity, and the Ali-CCP dataset.
- Kendall, Gal, Cipolla, "Multi-Task Learning Using Uncertainty to Weigh Losses" (CVPR 2018) and Yu et al., "Gradient Surgery for Multi-Task Learning / PCGrad" (NeurIPS 2020) — principled task weighting and gradient-conflict surgery that compose with MMoE/PLE.
- Within this series: [the ranking model: CTR prediction foundations](/blog/machine-learning/recommendation-systems/the-ranking-model-ctr-prediction-foundations), [DCN and explicit feature crossing](/blog/machine-learning/recommendation-systems/dcn-and-explicit-feature-crossing), [calibration and the prediction you can trust](/blog/machine-learning/recommendation-systems/calibration-and-the-prediction-you-can-trust), [delayed feedback and conversion attribution](/blog/machine-learning/recommendation-systems/delayed-feedback-and-conversion-attribution), and the capstone [recommender systems playbook](/blog/machine-learning/recommendation-systems/the-recommender-systems-playbook).
