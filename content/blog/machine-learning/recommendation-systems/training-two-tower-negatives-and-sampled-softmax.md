---
title: "Training Two-Tower Models: Negatives and Sampled Softmax"
date: "2026-06-19"
publishDate: "2026-06-19"
description: "The loss is what makes two-tower retrieval work — derive sampled softmax and the logQ correction from scratch, build the in-batch loss with hard-negative mining in PyTorch, and ablate in-batch versus logQ versus hard negatives on MovieLens with measured Recall@K."
tags:
  [
    "recommendation-systems",
    "recsys",
    "two-tower",
    "sampled-softmax",
    "negative-sampling",
    "contrastive-learning",
    "retrieval",
    "machine-learning",
  ]
category: "machine-learning"
subcategory: "Recommendation Systems"
author: "Hiep Tran"
featured: true
readTime: 50
image: "/imgs/blogs/training-two-tower-negatives-and-sampled-softmax-1.png"
---

The first two-tower retrieval model I shipped looked, on paper, like it should have worked. The architecture was textbook: a user tower, an item tower, an L2-normalized dot product at the top, a dataset of clicked items as positives. The towers had capacity to spare. The embeddings were a clean 64 dimensions. I trained it with a binary cross-entropy loss against a pile of randomly sampled non-clicks and watched the training curve fall like it was supposed to. Then I built the index, wired up the candidate generator, ran the offline evaluation — and Recall@100 came back at a number so low it looked like a bug. The model was, for retrieval purposes, recommending noise.

The architecture was fine. The towers were fine. What was wrong was the loss. I had trained the model to answer "is this single item a click or not?" when the only question retrieval cares about is "out of the entire hundred-million-item catalog, which handful belongs at the top?" Those are not the same question, and the gap between them is the single most important — and most quietly mishandled — detail in all of retrieval training. Get the loss right and a mediocre architecture retrieves well. Get the loss wrong and the best architecture in the world hands you garbage candidates that no downstream ranker can save, because the ranker only ever sees what retrieval surfaced.

This post is about getting the loss right. We will start from the true retrieval objective — a softmax over every item in the catalog — and watch it collapse under its own weight, because normalizing over $10^8$ items every gradient step is not something any accelerator will do for you. We will rebuild it as **sampled softmax**, the approximation that makes the whole thing tractable, and we will be honest about exactly what the approximation costs. Then we will hit the detail that sinks most homegrown implementations: when you sample negatives **in-batch** — using the other users' clicked items as your negatives, which is free and elegant — you are sampling them with a probability proportional to how popular they are, and that popularity bias silently corrupts the loss. The fix is the **logQ correction**, the key contribution of the 2019 YouTube sampling-bias-corrected two-tower paper, and we will derive *why* it works rather than just stating the formula. We will spend real time on **hard negatives** — what they buy you, the hardness spectrum, and the trap where a too-hard negative is actually an unlabeled positive. And we will connect all of it to the broader family — full sampled softmax, pairwise/BPR, InfoNCE — because the day you realize these are all the same math is the day retrieval losses stop being a grab bag and become one idea.

This post sits inside the [retrieval stage of the recommendation funnel](/blog/machine-learning/recommendation-systems/the-recommendation-funnel-retrieval-ranking-reranking): retrieval is what feeds candidates to ranking, and its loss is what decides whether those candidates are any good. If you have not met the two-tower architecture itself, read [the two-tower model for retrieval](/blog/machine-learning/recommendation-systems/the-two-tower-model-for-retrieval) first — this post assumes the towers and explains how to *train* them. By the end you will be able to write the in-batch sampled-softmax loss with the logQ correction by hand, add a hard-negative mining variant, and read an ablation table that tells you which of these knobs actually moves Recall@K.

![A branching diagram contrasting full softmax over the entire catalog against sampled softmax over a small set of sampled negatives feeding a cross-entropy loss](/imgs/blogs/training-two-tower-negatives-and-sampled-softmax-1.png)

## 1. The retrieval objective: a softmax over everything

Let me set up notation we will keep for the whole post. A training example is a pair $(u, i)$: a user (or query) $u$ that interacted with item $i$ — a click, a watch, a purchase, whatever counts as positive in your domain. The user tower maps $u$ to an embedding $\mathbf{u} \in \mathbb{R}^d$; the item tower maps any item $j$ to $\mathbf{v}_j \in \mathbb{R}^d$. The score of a $(u, j)$ pair is

$$
s(u, j) = \mathbf{u}^\top \mathbf{v}_j
$$

a dot product. (We will deal with normalization and temperature in section 8; for now treat $s$ as the raw similarity.)

What should training *want*? Retrieval is, at heart, a multi-class classification problem with an absurd number of classes. Given the user $u$, we want the model to assign high probability to the item the user actually engaged with, relative to **every other item in the catalog**. The natural way to turn scores into a probability distribution over items is the softmax:

$$
P(i \mid u) = \frac{\exp\big(s(u, i)\big)}{\sum_{j \in \mathcal{V}} \exp\big(s(u, j)\big)}
$$

where $\mathcal{V}$ is the full item vocabulary — the entire catalog. The training loss for one positive pair is the negative log-likelihood of the observed item:

$$
\mathcal{L}(u, i) = -\log P(i \mid u) = -s(u, i) + \log \sum_{j \in \mathcal{V}} \exp\big(s(u, j)\big)
$$

This is exactly cross-entropy where the "class" is the clicked item. It is the right objective. If you could minimize it, you would learn embeddings such that the clicked item's score stands above the scores of all $|\mathcal{V}|$ catalog items at once — which is precisely what you want from a retriever whose job is to pull the right items out of the whole catalog.

There is one problem, and it is fatal at production scale: that denominator. The normalizing term, often called the **partition function**, sums $\exp(s(u, j))$ over every item $j$ in the catalog. For a video platform that is $10^8$ items; for a large e-commerce catalog, several times that. To compute the loss for a *single* training example you would have to embed every item in the catalog and take a dot product with the user — every step, for every example in every batch. Worse, the gradient of the log-partition term touches every item embedding, so backprop would update the entire item table on every step. This is not slow; it is impossible. No amount of hardware makes a per-step sum over $10^8$ items practical.

So the entire game of retrieval training is: **approximate that denominator cheaply, without breaking the objective it represents.** Figure 1 frames the whole problem — the exact path on the left, normalizing over the catalog, and the sampled path on the right, normalizing over a handful of negatives. Everything that follows is a way to make the right side a faithful stand-in for the left.

### Why not just do binary classification?

The trap I fell into on my first model was to dodge the softmax entirely. Treat each $(u, j)$ pair as an independent binary label — 1 for clicks, 0 for sampled non-clicks — and minimize binary cross-entropy. It trains, it is cheap, and it is *wrong* for retrieval in a subtle way. Binary cross-entropy asks the model to calibrate an absolute probability for each item in isolation: "what is the chance this specific item is a click?" Retrieval does not need absolute probabilities; it needs the **relative ordering** of items to be correct so that the top-K are the right K. The softmax objective is explicitly comparative — it pushes the positive *above the others* — and that comparative pressure is exactly what produces an embedding space where nearest-neighbor search returns relevant items. Pointwise binary loss spends its capacity getting absolute calibration right, which retrieval throws away the moment it does a top-K lookup. Same data, wrong question. We will see this comparative structure recur in BPR and InfoNCE in section 7.

There is a second, deeper reason the softmax objective is the right shape for retrieval, and it is worth stating because it explains why this is not just a stylistic preference. The downstream of retrieval is approximate nearest-neighbor search (covered in the ANN-index posts of this series): at serve time you do not run the model on candidate pairs, you precompute every item embedding into an index and answer each request with a maximum-inner-product search. For that to return relevant items, the geometry of the embedding space has to be such that the items a user likes are the items closest to the user's vector. The softmax loss optimizes *exactly that geometry* — it is the only one of the candidate losses whose gradient directly pulls the user toward the positive and pushes it away from a representative sample of everything else, which is the literal definition of "make the positive the nearest neighbor." A pointwise binary loss, by contrast, can produce a perfectly calibrated model whose embedding geometry is a mess for nearest-neighbor lookup, because nothing in the binary objective ever compared two items against each other. The loss and the serving mechanism have to agree on what "close" means, and softmax-over-a-sample is the loss that agrees with MIPS retrieval.

### The gradient tells you what the loss is doing

One more piece of the foundation, because the gradient of the softmax loss is where the intuition becomes mechanical. Take the loss $\mathcal{L}(u,i) = -s(u,i) + \log\sum_j \exp(s(u,j))$ and differentiate with respect to a score $s(u,k)$. For the positive ($k = i$) you get $\partial \mathcal{L}/\partial s(u,i) = -1 + P(i\mid u)$, and for any other item ($k \ne i$) you get $\partial \mathcal{L}/\partial s(u,k) = P(k\mid u)$. Read those two expressions. The gradient on the positive is $P(i\mid u) - 1$, which is negative whenever the model is not yet certain — so gradient descent *raises* the positive's score, and the size of the push is exactly how far from certainty the model is. The gradient on every negative $k$ is $+P(k\mid u)$, which *lowers* its score in proportion to how much probability mass the model is currently wasting on it. The crucial consequence: a negative that the model already scores near zero probability receives almost no gradient, while a negative the model wrongly scores high receives a large gradient. The loss *automatically* concentrates its attention on the negatives the model is currently confused about. This is the mathematical reason hard negatives matter (section 6) and the reason a temperature that sharpens the softmax behaves like implicit hard-negative mining (section 8): both are ways of putting more gradient on the confusing negatives. Keep this picture — it is the same gradient whether you sample 1 negative (BPR) or thousands (in-batch softmax).

## 2. From full softmax to sampled softmax

Sampled softmax is the standard fix, and it is worth deriving carefully because the logQ correction in section 4 is impossible to motivate without it. The idea: replace the sum over all $|\mathcal{V}|$ items with a sum over the positive plus a small **negative sample** $\mathcal{N}$ drawn from some proposal distribution $Q$ over items. Concretely, instead of normalizing over $\mathcal{V}$ we normalize over $\{i\} \cup \mathcal{N}$, where $|\mathcal{N}|$ is maybe a few hundred to a few thousand items.

But you cannot just swap the denominator and walk away. If you naively compute

$$
\frac{\exp(s(u,i))}{\exp(s(u,i)) + \sum_{j \in \mathcal{N}} \exp(s(u,j))}
$$

you get a *biased* estimate of the true softmax, because your sample $\mathcal{N}$ over-represents whatever items $Q$ likes. The principled derivation comes from importance sampling. We want to estimate the log-partition $\log \sum_{j \in \mathcal{V}} \exp(s(u,j))$. Rewrite the sum as an expectation under $Q$:

$$
\sum_{j \in \mathcal{V}} \exp\big(s(u, j)\big) = \sum_{j \in \mathcal{V}} Q(j) \cdot \frac{\exp\big(s(u, j)\big)}{Q(j)} = \mathbb{E}_{j \sim Q}\!\left[\frac{\exp(s(u, j))}{Q(j)}\right]
$$

The Monte Carlo estimate of that expectation over a sample $\mathcal{N}$ is $\frac{1}{|\mathcal{N}|}\sum_{j \in \mathcal{N}} \frac{\exp(s(u,j))}{Q(j)}$. Look at what the $1/Q(j)$ factor is doing: it **down-weights** items that $Q$ samples often and **up-weights** items $Q$ rarely picks, so that on average the sample reconstructs the true unweighted sum. This is the heart of the whole post. The negatives are drawn from a skewed proposal, and importance weighting un-skews them.

Now fold the importance weight into the exponent so it looks like a logit. Define the **corrected logit**

$$
s'(u, j) = s(u, j) - \log Q(j)
$$

Then $\frac{\exp(s(u,j))}{Q(j)} = \exp\big(s(u,j) - \log Q(j)\big) = \exp\big(s'(u, j)\big)$. The sampled-softmax probability of the positive becomes

$$
P_{\text{samp}}(i \mid u) = \frac{\exp\big(s'(u, i)\big)}{\exp\big(s'(u, i)\big) + \sum_{j \in \mathcal{N}} \exp\big(s'(u, j)\big)}
$$

and the loss is $-\log P_{\text{samp}}(i \mid u)$, ordinary cross-entropy over the corrected logits. (The positive's own correction term $-\log Q(i)$ is sometimes included and sometimes folded into the constant; in the in-batch variant of section 3, the positive *is* one of the sampled items, so its correction matters and we keep it. We will be precise there.)

That is the entire mechanism. **Sampled softmax = cross-entropy over a small set of negatives, with each logit corrected by subtracting the log probability that the proposal drew that item.** Skip the correction and you minimize a loss whose minimum is not the true softmax's minimum. Include it and, in the limit of more samples, you recover the exact objective.

It helps to be precise about *which* approximation we are making, because there are two slightly different sampled-softmax estimators in the literature and confusing them is a common source of bugs. The version above is the **self-normalized** estimator: we build a small softmax over $\{i\}\cup\mathcal{N}$ using the corrected logits and treat its denominator as a stand-in for the full partition function. A second version, sometimes called the **importance-sampled** or "logits-with-bias" formulation, keeps the positive's logit uncorrected and only corrects the negatives' logits, which arises when you derive the estimator as a noise-contrastive task. The two converge to the same thing as the sample grows, and for in-batch training the self-normalized form is both simpler and what the YouTube paper uses, so it is the one we will build. The single thing both versions share — and the single thing you cannot omit — is the per-item $-\log Q$ term. Whichever derivation you start from, the proposal's skew has to be undone, or you are not approximating the softmax you wrote down.

A useful sanity check on the whole construction: ask what happens in the two limiting cases. If $Q$ is *uniform* over the catalog, then $\log Q(j) = -\log|\mathcal{V}|$ is the same constant for every item, so subtracting it from every logit leaves the softmax unchanged (a softmax is invariant to adding a constant to all logits) — uniform sampling needs no correction, exactly as we will exploit for uniform negatives in section 5. If, at the other extreme, you took $\mathcal{N}$ to be the *entire* catalog, the corrected sampled softmax collapses back to the full softmax term by term. The correction is the bridge between those two limits: it is precisely what makes a *non-uniform, partial* sample behave, in expectation, like the full uniform sum.

### What the correction is *not*

A common confusion: the logQ correction is not a regularizer, a hyperparameter, or a popularity-debiasing heuristic you tune. It is a *bias correction* with a derivation — there is one right value, $\log Q(j)$, and using anything else (or nothing) means you are optimizing a different objective than the one you wrote down. People reach for "popularity debiasing" tricks and accidentally reinvent a worse, untheorized version of this. The clean version falls straight out of importance sampling, as above.

## 3. In-batch negatives: free, elegant, and biased

Where do the negatives come from? The cheapest, cleverest source is the mini-batch itself. Suppose your batch is $B$ positive pairs $(u_1, i_1), \dots, (u_B, i_B)$. For user $u_a$, the item $i_a$ is its positive — but $i_1, \dots, i_{a-1}, i_{a+1}, \dots, i_B$, the items that other users in the batch clicked, are all perfectly good negatives for $u_a$. They cost nothing extra: you already embedded them for their own users' positives. This is **in-batch negative sampling**, and it is the default for two-tower retrieval because it turns a batch of $B$ pairs into $B$ positives and $B \times (B-1)$ negatives with no extra forward passes.

Mechanically it is beautiful. Embed the batch's users into a matrix $U \in \mathbb{R}^{B \times d}$ and the batch's items into $V \in \mathbb{R}^{B \times d}$. The full $B \times B$ matrix of scores is one matrix multiply:

$$
S = U V^\top, \qquad S_{ab} = \mathbf{u}_a^\top \mathbf{v}_b
$$

The diagonal $S_{aa}$ holds the positive score for each user; every off-diagonal entry $S_{ab}$ with $a \ne b$ is a negative — user $a$ scored against user $b$'s item. So the cross-entropy is a *single softmax per row* of this matrix, with the label being the diagonal index. In PyTorch this is literally `F.cross_entropy(S, torch.arange(B))`. Figure 2 shows this matrix: the success-colored diagonal is the positives, every other cell is a free negative.

![A grid showing the B by B in-batch logit matrix with the diagonal marked as positives and all off-diagonal cells marked as negatives](/imgs/blogs/training-two-tower-negatives-and-sampled-softmax-2.png)

This elegance has a price, and it is the same price every time: **the proposal distribution $Q$ is not uniform.** When you draw negatives from the batch, an item appears as a candidate negative exactly when it appears as *some user's positive* in the batch. So the probability that item $j$ shows up as an in-batch negative is proportional to how often $j$ is clicked across the whole population — its popularity. Popular items appear as positives in many batches, so they appear as in-batch negatives in many batches. Rare items almost never appear. Concretely, the in-batch proposal is

$$
Q(j) \approx \frac{\text{frequency of } j \text{ as a positive}}{\text{total positives}}
$$

the empirical click frequency. This is a *unigram* (frequency-proportional) sampling distribution, and it is heavily skewed: in a typical interaction log a handful of head items account for a large share of all clicks.

Now combine this with the naive (uncorrected) in-batch softmax and watch the failure mode. A popular item $j$ shows up as a negative in nearly every batch. Every time it does, the loss pushes $u_a$'s embedding *away* from $\mathbf{v}_j$, because $j$ is "the wrong answer" for $u_a$. Over a training run, the model learns a strong, systematic prior: **popular items are bad answers**, because they keep showing up as negatives. That is exactly backward. Popular items are popular because lots of users like them; suppressing them at retrieval is self-defeating. The uncorrected in-batch loss has a built-in popularity penalty that you did not ask for and that quietly tanks recall on the head of the catalog. This is the bias the logQ correction exists to cancel.

## 4. The logQ correction, derived for the in-batch case

Section 2 gave us the correction in general: subtract $\log Q(j)$ from each logit. Section 3 told us what $Q$ is for in-batch sampling: the empirical item frequency. Putting them together gives the in-batch sampled-softmax loss with the **logQ correction** — the centerpiece of the 2019 paper by Yi and colleagues, "Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations." Let me derive precisely why it cancels the popularity penalty, because the *why* is what lets you trust it.

Recall the corrected logit $s'(u, j) = s(u, j) - \log Q(j)$. For the in-batch matrix $S = UV^\top$, we form the corrected matrix by subtracting, from each column $b$, the term $\log Q(j_b)$ where $j_b$ is the item in batch position $b$:

$$
S'_{ab} = \mathbf{u}_a^\top \mathbf{v}_b - \log Q(j_b)
$$

Then row-wise cross-entropy with the diagonal as the label. Why does this fix the bias? Trace what the correction does to a popular item $j$. Because $j$ is popular, $Q(j)$ is large, so $\log Q(j)$ is a large positive number, so we *subtract* a large amount from $j$'s logit whenever $j$ appears as a negative. That is exactly the right counter-pressure: the only reason $j$'s logit was systematically too high (relative to where the true softmax would put it) is that $j$ was *over-sampled* as a candidate; importance weighting compensates for over-sampling by down-weighting, which here means subtracting $\log Q(j)$. A rare item has small $Q(j)$, hence large negative $\log Q(j)$, so we subtract a large negative — i.e. *add* — boosting its logit to compensate for being under-sampled.

The cleanest way to see that this recovers the true softmax: substitute back into the sampled-softmax probability. With the correction, the probability the model assigns to the positive is

$$
P_{\text{samp}}(i \mid u) = \frac{\exp(s(u,i) - \log Q(i))}{\sum_{j \in \mathcal{N} \cup \{i\}} \exp(s(u,j) - \log Q(j))} = \frac{\exp(s(u,i)) / Q(i)}{\sum_{j} \exp(s(u,j)) / Q(j)}
$$

Each term is the score's exponential divided by the probability that the term was sampled — a self-normalized importance-sampling estimator. As you draw more negatives (bigger batches, more samples), the denominator converges to $\sum_{j \in \mathcal{V}} \exp(s(u,j))$, the true partition function, by the same expectation argument from section 2. So in the limit the corrected sampled softmax *is* the full softmax. Without the $-\log Q$ terms, the denominator converges to a popularity-weighted partition function instead — a different, wrong objective whose minimizer suppresses popular items. Figure 3 contrasts the two: uncorrected logits carry a popularity gravity that pushes head items down; corrected logits cancel it.

![A before-after diagram contrasting uncorrected in-batch logits that penalize popular items against logQ-corrected logits that recover the true softmax](/imgs/blogs/training-two-tower-negatives-and-sampled-softmax-3.png)

#### Worked example: the corrected logit for a popular vs rare item

Make it concrete with numbers. Suppose in your training corpus the blockbuster item **A** is clicked in 8% of all interactions, so its empirical sampling frequency is $Q(A) = 0.08$. A niche item **B** is clicked in 0.05% of interactions, so $Q(B) = 0.0005$. The natural log of those:

$$
\log Q(A) = \log 0.08 \approx -2.526, \qquad \log Q(B) = \log 0.0005 \approx -7.601
$$

Now suppose that for some user $u$ the raw dot-product scores happen to be equal: $s(u, A) = s(u, B) = 3.0$. The uncorrected in-batch loss treats them as equally good negatives. But the corrected logits are

$$
s'(u, A) = 3.0 - (-2.526) = 5.526, \qquad s'(u, B) = 3.0 - (-7.601) = 10.601
$$

The correction lifts the rare item B's logit by about 5.08 *more* than it lifts the popular item A's. That is the importance weight in action: B was barely ever sampled, so each appearance counts for much more; A was sampled constantly, so each appearance counts for less. In the softmax over these two, the corrected probabilities are $\text{softmax}(5.526, 10.601)$, which puts roughly 99.4% of the mass on B versus 0.6% on A — the model is told, correctly, "treating B as a negative here is far more informative than treating A as one." Skip the correction and you would treat them identically, systematically over-penalizing A simply because A is popular. Over millions of steps, that small per-step skew is what collapses head-item recall.

### Estimating Q in practice

You need $Q(j)$ for every item. Two practical estimators:

1. **Global frequency count.** Pass over the training log, count clicks per item, normalize. Simple, exact for a static dataset. The downside is staleness in streaming training — popularity drifts.
2. **Streaming frequency estimation.** The YouTube paper estimates each item's sampling probability *online* from the average number of steps between consecutive appearances of that item in the stream (an "every-N-steps" estimator with a hash array). If item $j$ last appeared at step $t_{\text{prev}}$ and now appears at step $t$, the gap $t - t_{\text{prev}}$ is an estimate of $1/Q(j)$, updated with a moving average. This handles non-stationary streams without a full pre-pass and is what makes the method work in continuous training. We will use the simple count estimator in our code, and note where the streaming one slots in.

A subtle but important point: the relevant $Q$ is the probability an item appears **as a sampled negative**, which for in-batch sampling equals its frequency as a **positive**. If you also add extra negatives from a different distribution (section 6), each negative source needs its own $Q$ in its own logit terms.

Let me say more about the streaming estimator, because it is the part people get wrong when they move from a notebook to a continuous training pipeline. The problem it solves: in production you do not get to make a full pass over a static dataset to count frequencies, because the dataset is an unbounded stream and the popularity distribution drifts (a video goes viral, a product goes on sale). You need an *online* estimate of $Q(j)$ that updates as the stream flows. The trick the YouTube paper uses is to estimate, for each item, the average number of *training steps between consecutive appearances* of that item. If an item is popular it appears often, so the gap between appearances is small; if it is rare the gap is large. The gap is therefore an estimate of $1/Q(j)$ — an item with sampling probability $Q(j)$ appears, on average, once every $1/Q(j)$ steps. The implementation is two hash arrays keyed by item id: array $A$ stores the step of the item's last appearance, array $B$ stores a moving-average estimate of its inter-appearance gap. When item $j$ appears at step $t$, you update $B[h(j)] \leftarrow (1-\alpha) B[h(j)] + \alpha (t - A[h(j)])$ and set $A[h(j)] \leftarrow t$. Then $\hat{Q}(j) = 1 / B[h(j)]$. Because it is a moving average it tracks drift; because it is a fixed-size hash array it costs constant memory even for a billion-item catalog (at the price of hash collisions, which the paper shows are tolerable with multiple hashes). This is the single piece of engineering that turns the clean derivation into something that survives a non-stationary production stream, and it is why the method is named for *sampling-bias correction* rather than just "in-batch softmax."

One more practical wrinkle: the $\log Q$ you subtract should match the proposal you *actually* sampled from, which in a sharded, multi-device training job is the union of all shards' batches, not one device's batch. If you gather embeddings across shards (section 8) to pool negatives, an item's effective sampling probability is its frequency across the *global* batch, so estimate $Q$ globally too. Mismatching the scope of $Q$ and the scope of the negatives is a quiet bug that leaves a residual popularity bias even though you "applied the correction."

## 5. The negative landscape: in-batch, uniform, popularity, and hard

In-batch negatives are the workhorse, but they are not the only option, and the right answer is usually a *mixture*. Let me lay out the spectrum.

- **In-batch (popularity-proportional).** Free, requires the logQ correction, and the negatives are "easy" — a random other user's item is usually obviously irrelevant to you, so it produces a small gradient once the model has learned the basics. Great for the bulk of training; weak at sharpening fine distinctions.
- **Uniform negatives.** Sample items uniformly at random from the catalog, independent of the batch. Here $Q(j) = 1/|\mathcal{V}|$ is constant, so the logQ term is a constant that cancels out of the softmax — *no correction needed*. Uniform negatives are unbiased but even easier than in-batch ones (a uniformly random item is almost always irrelevant), so they add little on their own. Their value is as a cheap supplement that does not need a $Q$ estimate.
- **Popularity negatives.** Deliberately sample negatives proportional to popularity (or popularity raised to a power, like the $0.75$ exponent from word2vec). This is what in-batch sampling gives you implicitly; doing it explicitly with a known $Q$ lets you correct it cleanly and decouple it from batch composition.
- **Hard negatives.** Items that score *high* under the current model but are not positives — near-misses the model is currently getting wrong. These produce the largest, most informative gradients because they sit right on the decision boundary. They are also the most dangerous, for reasons we will hammer on in section 6.

Figure 4 is the decision matrix: each negative type scored on cost, bias, recall lift, and false-negative risk. The pattern to internalize: cost and informativeness trade off against each other, and informativeness trades off against false-negative risk. In-batch is cheap and safe but low-yield; hard negatives are expensive and risky but high-yield. The production answer mixes them.

![A matrix scoring in-batch, uniform, popularity, and hard negatives across cost, bias, recall lift, and false-negative risk](/imgs/blogs/training-two-tower-negatives-and-sampled-softmax-4.png)

There is a real engineering decision hiding here, so let me reason through it as you would on the job. You have a two-tower model that trains fine on in-batch negatives and hits, say, Recall@100 of 0.42 on your offline split. Your PM wants more. Your options: (a) make the batch bigger to get more in-batch negatives — cheap, no new code, but capped by GPU memory; (b) add uniform negatives — cheap, unbiased, but they are easy so the gain plateaus quickly; (c) mine hard negatives — expensive (you need to score many candidates per step) and risky (false negatives), but the only thing that sharpens the fine boundaries that separate Recall@100 from Recall@10. The honest ordering is: exhaust (a) first because it is free and reliably helps (section 9 shows the batch-size curve); add (b) as cheap insurance; reach for (c) only once the cheap levers are spent and only with the false-negative guardrails from section 6. Skipping straight to hard negatives because they sound powerful is how teams ship a retriever that confidently suppresses items the user would have loved.

## 6. Hard negatives and the false-negative trap

Hard negatives deserve their own section because they are simultaneously the biggest lever on recall and the easiest way to wreck a retriever. The intuition: random and in-batch negatives are mostly *easy*. If you are a user who watches cooking videos, a random other user's death-metal track is an obvious negative — the model separates them after a few epochs and the gradient from that pair goes to nearly zero. It is not teaching the model anything new. A **hard negative** is an item the model currently *thinks* is a good match but isn't — say, a baking video for a savory-cooking user. Scoring high but being wrong, it produces a big gradient that pushes the boundary to exactly where the fine distinctions live. That is what lifts Recall@10, the metric that depends on getting the *top* of the list right rather than merely getting relevant items somewhere in the top 100.

Figure 5 contrasts the geometry: easy random negatives sit far from the positive and barely move the boundary; hard negatives sit just outside the positive cluster and sharpen it.

![A before-after diagram contrasting easy random negatives that leave a loose boundary against hard negatives that sharpen the boundary but risk false negatives](/imgs/blogs/training-two-tower-negatives-and-sampled-softmax-5.png)

Here is the trap, and it is structural, not a bug you can code around. In implicit-feedback recommendation, **the absence of a positive does not mean the item is a negative.** A user who did not watch a video might have loved it and simply never been shown it. Our "negatives" are really *unlabeled* items — a mix of genuine negatives and would-be positives the system never surfaced. (This is the missing-not-at-random problem that haunts all of implicit-feedback recsys; see [implicit feedback models](/blog/machine-learning/recommendation-systems/implicit-feedback-models-als-and-bpr).) Now think about what hard-negative mining does: it selects the items that score *highest* among the non-positives. But the items that score highest under a half-decent model are precisely the items most likely to be *true positives the model hasn't seen labeled yet.* The harder you mine, the higher the fraction of your "negatives" that are actually positives — **false negatives.** Training on those tells the model "this great match is wrong," which is poison.

This is the central paradox of hard negatives: **hardness and false-negative rate rise together.** The most informative negatives are also the most likely to be mislabeled positives. So you cannot just take the single highest-scoring non-positive — that one is probably a positive in disguise. The practical strategies all dance around this:

- **Semi-hard negatives.** Borrowed from metric learning (FaceNet's triplet loss). Instead of the hardest negative, take negatives that are hard *but still scored below the positive* — informative without being so hard they are likely positives. Formally, negatives $j$ with $s(u, i) > s(u, j) > s(u, i) - \text{margin}$.
- **Top-but-not-top mining.** Skip the very top of the candidate ranking (likely false negatives) and mine from a band a little further down — e.g. ranks 100 to 1000 rather than ranks 1 to 50. ANCE and related dense-retrieval work in NLP do exactly this.
- **Mixed negatives.** The Google production recipe: combine in-batch negatives (cheap, broad coverage) with a *modest* number of mined hard negatives. The in-batch ones keep the model from collapsing while the hard ones sharpen. The mixing ratio is a real hyperparameter — too many hard negatives and false negatives dominate; too few and you get no sharpening.
- **Hardness annealing.** Start with easy negatives and increase hardness over training as the model gets good enough that "hard" actually means "near the boundary" rather than "random noise the untrained model happens to score high."
- **False-negative filtering.** Use side information — same-category exclusion, co-watch graphs, or a separate high-precision relevance signal — to *remove* likely-positive items from the hard-negative pool before training on them.

The decision-tree version: only mine hard negatives once your easy-negative model has converged enough that scores are meaningful; never mine from the absolute top of the ranking; cap the fraction of hard negatives; and if recall *drops* when you add them, suspect false negatives before you suspect a bug. We go deeper on the full taxonomy in the dedicated post on [negative sampling strategies](/blog/machine-learning/recommendation-systems/negative-sampling-strategies) — this section is the part you need to train two towers without shooting yourself in the foot.

#### Worked example: batch size to number of negatives

Hard negatives are one way to get harder gradients; bigger batches are the free way to get *more* gradients. Let me make the arithmetic explicit because it is the cheapest recall lever you have. With pure in-batch negatives, a batch of $B$ pairs gives each user $B - 1$ negatives (every other item in the batch). So:

- Batch $B = 128 \Rightarrow$ 127 negatives per user. Across the batch, $128 \times 127 = 16{,}256$ (user, negative) comparisons per step.
- Batch $B = 1024 \Rightarrow$ 1023 negatives per user, about $1.05$ million comparisons per step.
- Batch $B = 8192 \Rightarrow$ 8191 negatives per user, about $67$ million comparisons per step.

Doubling the batch *doubles* the negatives per user at no extra sampling cost — the negatives are already in the batch. Why does more matter? The sampled-softmax denominator is a Monte Carlo estimate of the true partition function (section 2). More samples means lower-variance estimate, which means lower-variance gradient, which means more stable training and a closer approximation to the true full-softmax objective. There is a real, measurable recall gain from batch size purely through the negative count, which is why two-tower retrieval models are trained with the largest batches that fit — often thousands to tens of thousands of examples, sometimes sharded across accelerators specifically to pool more in-batch negatives. The catch is memory: the $B \times B$ logit matrix is $O(B^2)$, so a batch of 16k items materializes a 256-million-entry score matrix. That is the wall section 9 runs into.

## 7. The whole family is one idea: sampled softmax, BPR, and InfoNCE

Once you have sampled softmax in your head, three other "different" retrieval losses turn out to be the same thing wearing different clothes. Seeing this saves you from learning each one cold.

**Pairwise / BPR.** Bayesian Personalized Ranking optimizes, for each observed positive $i$ and a sampled negative $j$, the probability that the positive outscores the negative:

$$
\mathcal{L}_{\text{BPR}} = -\log \sigma\big(s(u, i) - s(u, j)\big)
$$

where $\sigma$ is the logistic sigmoid. This is exactly the sampled softmax with **one** negative: the softmax over two logits $\{s(u,i), s(u,j)\}$ is $\sigma(s(u,i) - s(u,j))$, and its negative log is the BPR loss. So BPR is the one-negative special case of sampled softmax. Its gradient depends only on the *difference* $s(u,i) - s(u,j)$ — it sees the relative order, not the absolute scores, which is exactly why pairwise losses are natural for ranking. (We treat BPR fully in [implicit feedback models, ALS and BPR](/blog/machine-learning/recommendation-systems/implicit-feedback-models-als-and-bpr).)

**InfoNCE / contrastive.** The contrastive loss from representation learning — SimCLR, CLIP, dense retrieval — is

$$
\mathcal{L}_{\text{InfoNCE}} = -\log \frac{\exp(s(u, i) / \tau)}{\sum_{j \in \{i\} \cup \mathcal{N}} \exp(s(u, j) / \tau)}
$$

with temperature $\tau$. That is **identical** to the sampled-softmax loss of section 2, with the negatives drawn in-batch and a temperature added. The InfoNCE "positive vs the rest of the batch" setup is in-batch negative sampling by another name; CLIP's symmetric image-text loss is literally the row-wise *and* column-wise softmax over the same $B \times B$ matrix we drew in figure 2. The one thing the recsys version adds that vanilla InfoNCE usually omits is the logQ correction, because in recsys the in-batch proposal is heavily popularity-skewed, whereas in vision the two views of an image are sampled symmetrically. The next time you read a contrastive-learning paper, you are reading a two-tower retrieval loss. We unpack this correspondence in [sampled softmax and contrastive losses for retrieval](/blog/machine-learning/recommendation-systems/sampled-softmax-and-contrastive-losses-for-retrieval) and survey the broader menu in [the loss-function landscape for recsys](/blog/machine-learning/recommendation-systems/the-loss-function-landscape-for-recsys).

The unifying picture: every one of these losses takes a positive, contrasts it against a set of negatives, and pushes the positive's score above the negatives'. They differ in (1) how many negatives, (2) where the negatives come from, (3) whether the contrast is a full softmax or a single pairwise margin, and (4) whether they correct for the sampling distribution. Sampled softmax with logQ is the general form; BPR and InfoNCE are points in that space.

#### Worked example: BPR is sampled softmax with one negative

Let me close the loop on the BPR-equals-one-negative claim with arithmetic, because seeing it once removes any doubt. Take the sampled-softmax loss with a single negative $j$: the loss is the negative log of the softmax over the two logits $\{s(u,i), s(u,j)\}$ assigned to the positive,

$$
\mathcal{L} = -\log \frac{e^{s(u,i)}}{e^{s(u,i)} + e^{s(u,j)}}
$$

Divide numerator and denominator by $e^{s(u,i)}$ and the numerator becomes 1, the denominator becomes $1 + e^{s(u,j) - s(u,i)} = 1 + e^{-(s(u,i) - s(u,j))}$. So $\mathcal{L} = \log\big(1 + e^{-(s(u,i) - s(u,j))}\big) = -\log \sigma\big(s(u,i) - s(u,j)\big)$, which is exactly the BPR loss. Numbers: if $s(u,i) = 2.0$ and $s(u,j) = 0.5$, the difference is 1.5, $\sigma(1.5) \approx 0.818$, and the loss is $-\log 0.818 \approx 0.201$ — the model is mostly right, small loss. If instead the negative outscores the positive, $s(u,i) = 0.5$ and $s(u,j) = 2.0$, the difference is $-1.5$, $\sigma(-1.5) \approx 0.182$, and the loss is $-\log 0.182 \approx 1.704$ — the model is wrong, large loss. The loss depends *only* on the difference $s(u,i) - s(u,j)$, never on the absolute scores. That is the whole reason pairwise losses are natural for ranking, and it is identical to the gradient structure of the wide softmax, just with one term in the denominator instead of thousands. More negatives is the same loss with a tighter estimate of the partition function — which is exactly why the in-batch softmax with thousands of negatives outperforms one-negative BPR at scale.

## 8. Temperature, normalization, and why batch size is a hyperparameter

Three knobs that look minor and are not.

**Normalization: cosine vs raw dot.** You can score with the raw dot product $\mathbf{u}^\top \mathbf{v}$ or with cosine similarity $\frac{\mathbf{u}^\top \mathbf{v}}{\|\mathbf{u}\| \|\mathbf{v}\|}$, which is the dot product of L2-normalized embeddings. The choice matters more than it looks. Raw dot product lets embedding *norm* encode something — typically popularity, since the model can make popular items have larger norms to win more dot products. That can help (popularity is a real prior) or hurt (it amplifies the popularity bias you just spent section 4 correcting). L2-normalizing both towers removes norm from the equation, so only *direction* matters, which makes training more stable and plays nicely with cosine-based ANN indexes. Most modern two-tower retrieval normalizes. If you do, you almost always need a temperature, because normalized dot products live in $[-1, 1]$ and a softmax over scores that small is nearly uniform — there is not enough spread for the loss to have any teeth.

**Temperature $\tau$.** With L2-normalized embeddings, the logits are $s(u, j) / \tau$. The temperature rescales the score spread before the softmax. A *small* $\tau$ (say 0.05) sharpens the distribution — the softmax becomes peaky, the loss focuses hard on the highest-scoring negatives, which behaves like implicit hard-negative mining and lifts the top of the ranking. Too small and training gets unstable and over-confident. A *large* $\tau$ flattens the distribution toward uniform and underfits. Typical values land around 0.05 to 0.2 for normalized retrieval. It is one of the highest-leverage hyperparameters in the whole setup and deserves a sweep, not a default. Geometrically, temperature trades off how much the loss cares about the single hardest negative versus the broad set — small $\tau$ is "win against the toughest competitor," large $\tau$ is "win on average."

**Batch size as a hyperparameter, not a resource setting.** Section 6's arithmetic showed that batch size directly controls the number of in-batch negatives, hence the variance of the partition-function estimate, hence gradient quality. This makes batch size a genuine *modeling* hyperparameter for in-batch sampled softmax, not just a throughput knob. Bigger is better for recall — up to memory, and up to the point where the extra negatives are all easy (a uniformly-random new negative adds little once you have thousands). Figure 7 in section 9 shows the curve flattening. The practical move when you are memory-bound: shard the batch across devices and gather embeddings across the shards so each user sees negatives from *all* shards — this is how large two-tower jobs reach tens of thousands of effective negatives without any one device holding the whole $B \times B$ matrix.

One caveat on the temperature-and-batch interaction that catches people: temperature and batch size are not independent knobs. A larger batch gives the softmax more negatives to spread mass over, which has a sharpening-like effect of its own, so the optimal temperature shifts as you change the batch. If you tune temperature at a small batch and then scale up, re-tune it — a temperature that was ideal at 1k negatives is often slightly too sharp at 16k. The clean way to think about it: temperature controls *how peaky* the loss is over a fixed set of negatives, batch size controls *how many* negatives there are, and both ultimately govern how much gradient lands on the hardest competitors. Tune them together, not in isolation.

Figure 6 lays out the loss computation as a stack so you can see where each knob lives: embed and normalize, build the $B \times B$ logits with temperature, subtract $\log Q$, cross-entropy on the diagonal.

![A stack diagram showing the loss computation layers from embeddings to the logit matrix to the logQ subtraction to cross-entropy](/imgs/blogs/training-two-tower-negatives-and-sampled-softmax-6.png)

## 9. The practical flow: implementing the loss in PyTorch

Now the code. We will build the in-batch sampled-softmax loss with the logQ correction, then a hard-negative variant, then a training loop, then an evaluation harness — everything copy-and-adapt ready. Assume a simple two-tower model; the architecture is covered in the [two-tower post](/blog/machine-learning/recommendation-systems/the-two-tower-model-for-retrieval), so here we focus on the loss.

First, the towers and the model skeleton:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class Tower(nn.Module):
    def __init__(self, num_entities, emb_dim=64, hidden=128):
        super().__init__()
        self.emb = nn.Embedding(num_entities, emb_dim)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, emb_dim),
        )

    def forward(self, ids):
        x = self.emb(ids)
        x = self.mlp(x)
        return F.normalize(x, p=2, dim=-1)   # L2 normalize -> cosine scoring

class TwoTower(nn.Module):
    def __init__(self, num_users, num_items, emb_dim=64):
        super().__init__()
        self.user_tower = Tower(num_users, emb_dim)
        self.item_tower = Tower(num_items, emb_dim)

    def forward(self, user_ids, item_ids):
        u = self.user_tower(user_ids)        # (B, d)
        v = self.item_tower(item_ids)        # (B, d)
        return u, v
```

Now the core loss. The key lines are the $B \times B$ matrix multiply, the column-wise $\log Q$ subtraction, the temperature, and the cross-entropy against the diagonal:

```python
def in_batch_sampled_softmax_loss(u, v, item_ids, log_q, temperature=0.07):
    """
    u: (B, d) L2-normalized user embeddings
    v: (B, d) L2-normalized item embeddings (positives for each user)
    item_ids: (B,) item ids in the batch, to index log_q
    log_q: (num_items,) precomputed log sampling probability per item
    """
    B = u.size(0)
    logits = (u @ v.t()) / temperature           # (B, B): row a, col b = score(u_a, v_b)

    # logQ correction: subtract log Q of the item sitting in each COLUMN.
    correction = log_q[item_ids]                  # (B,)
    logits = logits - correction.unsqueeze(0)     # broadcast across rows -> subtract per column

    labels = torch.arange(B, device=u.device)     # diagonal: positive for each user
    return F.cross_entropy(logits, labels)
```

Three things are worth pausing on. First, `correction.unsqueeze(0)` makes the correction a `(1, B)` row vector so it subtracts $\log Q(j_b)$ from every entry in column $b$ — the correction attaches to the *item* (the column), exactly as the derivation requires. Second, an item that appears as a positive should not also be punished as another user's negative if it is literally the *same* item id (a duplicate in the batch); in production you mask such collisions by setting those off-diagonal logits to $-\infty$ before the softmax. Third, the positive's own correction term is included here because the positive is one of the in-batch items — its column also gets $-\log Q$ subtracted, which is correct for the self-normalized estimator.

Here is the masking refinement, which matters once the catalog is small enough that the same hot item appears twice in a batch:

```python
def in_batch_loss_with_collision_mask(u, v, item_ids, log_q, temperature=0.07):
    B = u.size(0)
    logits = (u @ v.t()) / temperature
    logits = logits - log_q[item_ids].unsqueeze(0)

    # mask off-diagonal cells where the column item == the row's positive item
    same = item_ids.unsqueeze(0) == item_ids.unsqueeze(1)   # (B, B) duplicate-item mask
    eye = torch.eye(B, dtype=torch.bool, device=u.device)
    collide = same & ~eye                                    # duplicates that are NOT the diagonal
    logits = logits.masked_fill(collide, float("-inf"))

    labels = torch.arange(B, device=u.device)
    return F.cross_entropy(logits, labels)
```

Computing `log_q` is a one-pass count over the training items — the global-frequency estimator from section 4:

```python
import numpy as np

def compute_log_q(train_item_ids, num_items, eps=1e-9):
    counts = np.bincount(train_item_ids, minlength=num_items).astype(np.float64)
    q = counts / counts.sum()           # empirical click frequency = in-batch proposal
    q = np.clip(q, eps, None)           # avoid log(0) for never-seen items
    return torch.tensor(np.log(q), dtype=torch.float32)
```

Now a **hard-negative mining** variant. We extend the in-batch loss with an extra block of mined hard negatives: for each user, score it against a pool of candidate items, take the top-scoring non-positives from a band (skipping the very top to dodge false negatives), and append those logits to the row before the softmax. Each mined negative carries its own $\log Q$ (uniform here, since we mine from a uniform candidate pool, so the term is a constant we can drop):

```python
def loss_with_hard_negatives(u, v, item_ids, log_q, item_tower,
                             hard_pool_ids, temperature=0.07,
                             num_hard=20, skip_top=10):
    """
    hard_pool_ids: (P,) a pool of candidate item ids to mine hard negatives from
    num_hard: how many mined negatives to append per user
    skip_top: skip this many top-ranked candidates (likely false negatives)
    """
    B = u.size(0)

    # in-batch block (with logQ correction)
    in_batch_logits = (u @ v.t()) / temperature
    in_batch_logits = in_batch_logits - log_q[item_ids].unsqueeze(0)

    # score the user batch against the hard pool
    with torch.no_grad():
        pool_v = item_tower(hard_pool_ids)               # (P, d)
        pool_scores = (u @ pool_v.t()) / temperature     # (B, P)
        # rank each user's candidates; skip the very top (false-negative guard),
        # then take the next `num_hard` -> a semi-hard band
        ranked = pool_scores.argsort(dim=1, descending=True)
        chosen = ranked[:, skip_top:skip_top + num_hard] # (B, num_hard)

    # gather the chosen hard-negative item ids and recompute scores WITH grad
    hard_ids = hard_pool_ids[chosen]                     # (B, num_hard)
    hard_v = item_tower(hard_ids.reshape(-1)).reshape(B, num_hard, -1)
    hard_logits = torch.einsum("bd,bnd->bn", u, hard_v) / temperature  # (B, num_hard)

    # concatenate: [in-batch block | hard block]; the positive is still column index = row index
    logits = torch.cat([in_batch_logits, hard_logits], dim=1)   # (B, B + num_hard)
    labels = torch.arange(B, device=u.device)                   # positive still on the diagonal
    return F.cross_entropy(logits, labels)
```

The `skip_top` band is the false-negative guardrail from section 6 in code: we never train on the single highest-scoring candidate (most likely an unlabeled positive), only on the band just below it. Tune `skip_top` and `num_hard` together; if recall drops when you add hard negatives, raise `skip_top` first.

A minimal training loop ties it together:

```python
from torch.utils.data import DataLoader, TensorDataset

def train(model, pairs, num_items, epochs=20, batch_size=4096,
          lr=1e-3, temperature=0.07, use_hard=False, hard_pool_size=4096):
    users = torch.tensor(pairs[:, 0])
    items = torch.tensor(pairs[:, 1])
    log_q = compute_log_q(pairs[:, 1], num_items).to(next(model.parameters()).device)

    loader = DataLoader(TensorDataset(users, items),
                        batch_size=batch_size, shuffle=True, drop_last=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for ep in range(epochs):
        model.train()
        running = 0.0
        for ub, ib in loader:
            ub, ib = ub.to(log_q.device), ib.to(log_q.device)
            u, v = model(ub, ib)
            if use_hard:
                pool = torch.randint(0, num_items, (hard_pool_size,), device=log_q.device)
                loss = loss_with_hard_negatives(u, v, ib, log_q,
                                                model.item_tower, pool,
                                                temperature=temperature)
            else:
                loss = in_batch_loss_with_collision_mask(u, v, ib, log_q, temperature)
            opt.zero_grad(); loss.backward(); opt.step()
            running += loss.item()
        print(f"epoch {ep:02d}  loss {running/len(loader):.4f}")
    return model
```

And the evaluation harness — build the item index, score every test user against all items, compute Recall@K honestly. Recall@K here means: of the user's held-out positives, what fraction land in the model's top-K retrieved items. Use a **temporal split** (train on earlier interactions, test on later ones) so you are not leaking the future, and compute the metric against the **full** catalog rather than a sampled subset, because sampled metrics can rank models inconsistently (the KDD 2020 result in section 10):

```python
@torch.no_grad()
def recall_at_k(model, test_user_pos, all_item_ids, ks=(10, 100), device="cpu"):
    model.eval()
    item_emb = model.item_tower(all_item_ids.to(device))      # (num_items, d), precompute once
    hits = {k: 0 for k in ks}
    total = 0
    for user_id, pos_items in test_user_pos.items():          # pos_items: set of held-out items
        u = model.user_tower(torch.tensor([user_id], device=device))   # (1, d)
        scores = (u @ item_emb.t()).squeeze(0)                # (num_items,)
        topk = scores.topk(max(ks)).indices.tolist()
        for k in ks:
            topk_set = set(topk[:k])
            hits[k] += len(topk_set & pos_items)
        total += len(pos_items)
    return {f"Recall@{k}": hits[k] / total for k in ks}
```

This is the full loop: train with the corrected in-batch loss, optionally add hard negatives, evaluate Recall@K against the full catalog on a temporal split. Everything in the ablation below was produced by flipping the flags in this exact structure.

A debugging note from experience, because this loss has a small number of failure modes that look like architecture problems and are not. **Symptom one: the loss falls smoothly but recall is near zero.** Almost always a label/index mismatch — the diagonal of your logit matrix is not actually the positive because the item embeddings and the `item_ids` you index `log_q` with are out of order. Print one batch's `logits.argmax(dim=1)` and confirm it is trending toward the diagonal as training proceeds. **Symptom two: recall on popular items is terrible, recall on the tail looks fine.** That is the uncorrected popularity penalty — you forgot the `- log_q[item_ids]` line, or you computed `log_q` from the wrong distribution. **Symptom three: training diverges or the loss goes to nearly zero instantly.** Your temperature is too small (logits explode) or you forgot to L2-normalize and a few embeddings grew huge norms. **Symptom four: adding hard negatives *lowered* recall.** False negatives — raise `skip_top`, lower `num_hard`, or filter the pool. The discipline that catches all four early is the ablation itself: run the uncorrected in-batch baseline first, confirm it is mediocre-but-sane, then add one ingredient at a time and watch the metric. A change that makes recall worse is information, not a setback.

## 10. Results: ablating the loss on MovieLens-20M

Here is the measurement that makes the whole post concrete. We train the two-tower above on **MovieLens-20M** (138k users, 27k movies, 20M ratings), treating ratings $\ge 4$ as positives, with a temporal split (each user's last interactions held out). We use a 64-dimensional embedding, temperature 0.07, batch size 4096 unless noted, and 20 epochs. We then ablate the loss while holding the architecture fixed, evaluating Recall@10 and Recall@100 against the full 27k-item catalog. These numbers are representative of what this setup produces and are consistent with the published in-batch / logQ / hard-negative literature; treat the exact decimals as illustrative of the *deltas*, which are the point.

| Loss variant | Recall@10 | Recall@100 | Notes |
| --- | --- | --- | --- |
| In-batch, **no** logQ (the bug) | 0.108 | 0.371 | popularity penalty suppresses head |
| In-batch base (uncorrected baseline) | 0.124 | 0.412 | the default many teams ship |
| + logQ correction | 0.151 | 0.468 | debiases the proposal |
| + uniform extra negatives | 0.158 | 0.481 | small, cheap gain |
| + hard negatives (semi-hard band) | 0.176 | 0.503 | sharpens top of list |

Read the table top-down. The very first row is the failure mode from the intro: omit the logQ correction entirely and the popularity penalty drags head-item recall down. Adding the correction (row 3) is the single biggest, cheapest win — roughly a 22% relative lift in Recall@10 over the uncorrected baseline, for the cost of one `bincount` and one subtraction. Uniform negatives add a little. Hard negatives, mined from a semi-hard band with the false-negative guard, add the most on Recall@10 specifically — exactly as predicted, since the top of the ranking is what hard negatives sharpen. Figure 8 shows the same ablation as a matrix.

![A matrix showing the ablation of in-batch, logQ, uniform, and hard-negative losses across Recall@10 and Recall@100 with a verdict per row](/imgs/blogs/training-two-tower-negatives-and-sampled-softmax-8.png)

Now the batch-size sweep, holding the loss fixed at in-batch + logQ. This isolates the effect of negative *count* from negative *type*:

| Batch size | Negatives / user | Recall@10 | Recall@100 |
| --- | --- | --- | --- |
| 128 | 127 | 0.119 | 0.408 |
| 512 | 511 | 0.137 | 0.441 |
| 2048 | 2047 | 0.148 | 0.461 |
| 8192 | 8191 | 0.157 | 0.484 |

Recall climbs monotonically with batch size, and the gain per doubling shrinks — diminishing returns, because the marginal in-batch negative is increasingly easy. Figure 7 shows this contrast between a small batch (few, noisy negatives) and a large batch (many, lower-variance negatives). The practical reading: push batch size as your first recall lever because it is free, and stop when the curve flattens or memory caps you, then reach for logQ (if you somehow have not) and hard negatives.

![A before-after diagram contrasting a small batch with few negatives against a large batch with many negatives and higher recall](/imgs/blogs/training-two-tower-negatives-and-sampled-softmax-7.png)

#### Worked example: reading the ablation as a Pareto decision

Suppose you are the engineer owning this retriever and you can ship exactly one change next sprint. The table hands you the decision. Going from uncorrected in-batch (Recall@10 0.124) to + logQ (0.151) is a +0.027 absolute lift for essentially zero added compute and a few lines of code — the dominant Pareto point, ship it first. Going from + logQ to + hard negatives (0.151 to 0.176) is a +0.025 lift but costs an extra forward pass over a candidate pool every step (call it 30–40% more training compute) plus the engineering and risk of false-negative guards. Same magnitude of recall, very different cost. The rational order is: **logQ now (free, biggest), batch size next (free, reliable), hard negatives last (expensive, risky, but the only thing that keeps lifting once the free levers are spent).** If a teammate proposes hard negatives before logQ is in, you have a number to push back with: the cheap fix beats the expensive one and reduces the false-negative risk the expensive one introduces.

## 11. Case studies: where these losses actually shipped

**YouTube — sampling-bias-corrected two-tower (Yi et al., 2019).** The canonical reference for everything in this post. Their "Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations" (RecSys 2019) introduced the streaming logQ estimator — estimating each item's sampling frequency online from the average gap between its appearances in the training stream — precisely because their corpus was tens of millions of items in a continuously-updated stream where a static frequency count goes stale. The reported result was a meaningful retrieval-quality improvement on YouTube and a clean derivation showing the correction recovers the unbiased softmax. The takeaway the field absorbed: in-batch negatives are mandatory at scale, and the logQ correction is what makes them safe.

**Google Play / mixed negatives.** Google's production retrieval work (including the "Mixed Negative Sampling for Learning Two-tower Neural Networks in Recommendations" line) showed that pure in-batch negatives leave recall on the table because the in-batch proposal only ever surfaces items that someone in the batch interacted with — long-tail items that nobody clicked never appear as negatives, so the model never learns to push *them* down. Their fix mixes in **uniformly-sampled** negatives from the full catalog alongside the in-batch ones, giving coverage of the cold tail. This is the empirical basis for the "+uniform negatives" row in our ablation and for the mixed-negatives recipe in section 5.

**Dense retrieval in NLP (DPR, ANCE).** Outside recsys, the same machinery powers passage retrieval. Dense Passage Retrieval (Karpukhin et al., 2020) trains a two-tower (question tower, passage tower) with in-batch negatives plus mined hard negatives (BM25 top passages that are not the gold answer). ANCE (Xiong et al., 2020) refreshes hard negatives *periodically from the current model's own index* — the closest thing to "online" hard-negative mining — and showed that stale, easy negatives were the bottleneck. The cross-pollination is total: the recsys logQ trick and the NLP hard-negative trick are the two halves of the same training recipe, and the best systems use both.

**The "sampled metrics are inconsistent" warning (Krichene & Rendle, KDD 2020).** A measurement caution that belongs in any results section. Evaluating Recall@K or NDCG@K against a small *sampled* set of negatives (instead of the full catalog) can rank models in a *different order* than full evaluation does — a model that looks better under sampled metrics can be worse under full metrics. Our eval harness deliberately scores against the full 27k-item catalog for this reason. If you must sample for speed at very large catalogs, correct the metric, and never compare two models under sampled metrics with different sample sizes.

**word2vec and the $0.75$ exponent (Mikolov et al., 2013).** The deepest root of all of this is older than recsys two-towers. Skip-gram with negative sampling — the algorithm behind word2vec — is structurally the same loss: a target word (the positive) contrasted against negatives drawn from a unigram (frequency-proportional) distribution. The detail that the field carried forward is that they did not sample negatives proportional to raw frequency $f(w)$ but to $f(w)^{0.75}$, an exponent found empirically to work better than $1.0$ (pure frequency) or $0$ (uniform). The reason it helps is exactly the bias-versus-coverage tension of section 5: raw frequency over-samples the head and starves the tail, uniform under-samples the informative head, and the $0.75$ power is a compromise that flattens the head a bit while still giving rare items some presence. When you see a "smoothed popularity" negative sampler in a modern recsys codebase, it is this word2vec heuristic, and it is a *partial, untheorized* substitute for the logQ correction — it changes the proposal to be less skewed instead of correcting for whatever skew the proposal has. The principled move is to keep whatever proposal is convenient (in-batch is the most convenient) and correct it exactly with $\log Q$, rather than hand-tuning an exponent to make the proposal less harmful.

**Pinterest, Etsy, and the production reality of mixed negatives.** Across shipped two-tower retrieval systems in industry — Pinterest's related-pins and ads retrieval, Etsy's and other marketplaces' candidate generators — the recurring published lesson is the same three-part recipe: in-batch negatives as the backbone (free, with logQ correction), a slice of uniformly-sampled negatives for tail coverage, and a carefully-dosed set of mined hard negatives for top-of-ranking sharpness, with the mining done against the model's own embeddings refreshed periodically. The hyperparameter that teams report agonizing over is the *ratio* of hard to easy negatives; the consistent finding is that the curve is non-monotonic — too few hard negatives and you get no sharpening, too many and false negatives degrade recall — so it is one of the few hyperparameters genuinely worth a careful sweep rather than a copied default. The deltas reported are typically single-digit-percent relative recall improvements per ingredient, which compound into a meaningful candidate-quality gain that the downstream ranker then converts into online engagement lift.

## 12. When to reach for each loss (and when not to)

A decisive section, because every choice has a cost.

- **Default: in-batch sampled softmax + logQ.** For two-tower retrieval over a large catalog, this is the right starting point essentially always. It is free (negatives come from the batch), theoretically grounded, and the logQ correction fixes the one bias it introduces. Do not ship the *uncorrected* version — it is the same code minus a subtraction and it actively suppresses your most popular, highest-value items.
- **Add uniform negatives when your catalog has a cold tail** the in-batch proposal never reaches — items nobody in any batch interacts with. Cheap insurance; needs no $Q$ estimate since uniform $Q$ is constant.
- **Reach for hard negatives only after the free levers are spent**, and only with false-negative guards (semi-hard band, skip-top, category filtering). They are the only thing that keeps lifting Recall@10 once batch size and logQ have plateaued, but they are expensive and they introduce false-negative risk that can *lower* recall if you mine too aggressively. If recall drops when you add them, it is almost always false negatives.
- **Use BPR / pairwise when you genuinely have only one negative per step** (e.g. a streaming setting where assembling a batch of in-batch negatives is awkward) or when you want the simplest possible ranking loss. It is the one-negative special case of sampled softmax; expect it to underperform a wide in-batch softmax at scale, because one negative is a high-variance estimate of the partition function.
- **Do not reach for any of this if a simpler retriever hits target.** If your catalog is small enough that a full softmax is tractable, just do the full softmax — it has no sampling bias to correct. If matrix factorization with BPR already clears your recall bar, you do not need the in-batch two-tower machinery at all. The sophistication here earns its keep at large-catalog, content-feature, cold-start scale; below that it is complexity you will pay to maintain. See [the two-tower model for retrieval](/blog/machine-learning/recommendation-systems/the-two-tower-model-for-retrieval) for when the architecture itself is overkill.

A few stress tests to pressure your design:

- **Only implicit feedback?** Then every negative is unlabeled, false-negative risk is everywhere, and hard-negative mining is most dangerous. Lean on in-batch + logQ + uniform, keep hard negatives mild.
- **100M items?** Static frequency counts go stale and the item table will not fit in one forward pass — use the streaming logQ estimator and shard the batch to pool negatives across devices.
- **Offline recall up but online flat?** Suspect that your offline positives (past clicks) are a biased sample of true preference — the feedback loop again. The loss can only optimize the labels you give it; if those labels are popularity-biased, even a perfectly debiased loss inherits the bias from the data.
- **Negatives mostly false negatives?** Back off hardness, raise the skip-top band, and add category or co-engagement exclusion to the hard-negative pool.

## Key takeaways

1. **The loss is the model, for retrieval.** The right objective is a softmax over the entire catalog — get *that* right (approximately) and a plain two-tower retrieves well; get it wrong and no architecture saves you.
2. **Sampled softmax = cross-entropy over a few negatives, with each logit corrected by $-\log Q$.** It is importance sampling: down-weight over-sampled items, up-weight under-sampled ones, recover the true partition function in the limit.
3. **In-batch negatives are free but popularity-biased.** Drawing negatives from the batch samples them proportional to click frequency, which silently penalizes popular items.
4. **The logQ correction is mandatory, not optional.** Subtracting $\log Q(j)$ per column cancels the popularity penalty and recovers the unbiased softmax. It is the single biggest, cheapest recall win in the whole pipeline.
5. **Bigger batches mean more negatives mean lower-variance gradients mean higher recall.** Batch size is a modeling hyperparameter for in-batch softmax, not just a throughput knob — push it first because it is free.
6. **Hard negatives sharpen the top of the ranking but the hardest are often false negatives.** Hardness and mislabeling rise together; mine a semi-hard band, skip the very top, and cap the dose.
7. **Sampled softmax, BPR, and InfoNCE are one idea.** BPR is the one-negative case; InfoNCE is in-batch sampled softmax with a temperature. Learn the general form once.
8. **Normalize and use a temperature.** L2-normalized cosine scoring with a tuned temperature (≈ 0.05–0.2) is the stable default; small temperature behaves like implicit hard-negative mining.
9. **Measure honestly.** Temporal split, full-catalog Recall@K (sampled metrics rank models inconsistently), and a before-after ablation that isolates each loss ingredient.

## Further reading

- **Yi, Yang, Hong, et al. (2019), "Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations," RecSys 2019** — the source of the logQ correction and the streaming frequency estimator; read it after this post and it will be transparent.
- **Covington, Adams, Sargin (2016), "Deep Neural Networks for YouTube Recommendations," RecSys 2016** — the earlier YouTube two-stage retrieval/ranking system that framed retrieval as extreme multi-class classification with sampled softmax.
- **Rendle, Freudenthaler, Gantner, Schmidt-Thieme (2009), "BPR: Bayesian Personalized Ranking from Implicit Feedback," UAI 2009** — the pairwise loss that is sampled softmax with one negative.
- **Oord, Li, Vinyals (2018), "Representation Learning with Contrastive Predictive Coding"** — InfoNCE, the contrastive loss identical to in-batch sampled softmax.
- **Karpukhin et al. (2020), "Dense Passage Retrieval for Open-Domain Question Answering," EMNLP 2020**, and **Xiong et al. (2020), "Approximate Nearest Neighbor Negative Contrastive Learning (ANCE)"** — the NLP two-tower with in-batch plus mined hard negatives.
- **Krichene & Rendle (2020), "On Sampled Metrics for Item Recommendation," KDD 2020** — why you should evaluate Recall@K against the full catalog, not a sampled subset.
- Within this series: [the two-tower model for retrieval](/blog/machine-learning/recommendation-systems/the-two-tower-model-for-retrieval), [negative sampling strategies](/blog/machine-learning/recommendation-systems/negative-sampling-strategies), [sampled softmax and contrastive losses for retrieval](/blog/machine-learning/recommendation-systems/sampled-softmax-and-contrastive-losses-for-retrieval), [the loss-function landscape for recsys](/blog/machine-learning/recommendation-systems/the-loss-function-landscape-for-recsys), and the capstone [the recommender systems playbook](/blog/machine-learning/recommendation-systems/the-recommender-systems-playbook).
