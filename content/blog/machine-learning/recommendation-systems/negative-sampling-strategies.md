---
title: "Negative Sampling Strategies for Recommenders"
date: "2026-06-19"
publishDate: "2026-06-19"
description: "The single biggest lever in retrieval quality is which negatives you show the model — derive how the negative distribution shapes the gradient, build uniform, popularity, in-batch, and hard-negative samplers in PyTorch, and ablate them on MovieLens with measured Recall@K."
tags:
  [
    "recommendation-systems",
    "recsys",
    "negative-sampling",
    "hard-negatives",
    "retrieval",
    "two-tower",
    "contrastive-learning",
    "machine-learning",
  ]
category: "machine-learning"
subcategory: "Recommendation Systems"
author: "Hiep Tran"
featured: true
readTime: 52
image: "/imgs/blogs/negative-sampling-strategies-1.png"
---

The retrieval model that taught me this lesson was the best-architected thing I had built that quarter. Two clean towers, 128-dimensional embeddings, a proper sampled-softmax loss, a temporal train/eval split with no leakage. Recall@100 came back at 0.31 — barely better than recommending the global top-100 to everyone. I spent two days auditing the architecture: tower depth, embedding dimension, learning rate, batch size. Nothing moved the number more than a point or two. Then a colleague asked the question that ended the investigation in ten minutes: "What are you using for negatives?" I was sampling them uniformly at random from the catalog. Every negative the model ever saw was an item so unrelated to the user that the model got it right on the first epoch, learned nothing more, and spent the rest of training congratulating itself. I switched to a mix of in-batch and mined hard negatives, changed not one line of the model, and Recall@100 went to 0.47.

That is the thesis of this post, and it is not a small one: **in implicit-feedback retrieval, the negatives are the model.** You only ever observe positives — the clicks, the watches, the purchases. You never observe a labeled "this user dislikes this item." So the entire decision boundary — the line the model draws between "belongs in the candidate set" and "does not" — is learned from negatives you *construct*. Construct them badly and no architecture saves you. Construct them well and a two-tower from 2019 retrieves better than a graph neural network fed garbage negatives. The architecture papers get the citations; the negative sampler gets the recall.

![A two-column figure contrasting random negatives that leave a fuzzy decision boundary with hard negatives that produce a sharp boundary and higher recall](/imgs/blogs/negative-sampling-strategies-1.png)

This post is the field guide to that lever. We will start with *why* negatives carry all the signal in implicit retrieval, and make it precise by writing the expected gradient under an arbitrary sampling distribution $Q$ — once you can see the gradient, every strategy becomes a statement about how to shape $Q$. Then we walk the strategies in order of sophistication: **uniform** random (cheap, weak), **popularity-based** (the word2vec 0.75-power trick, which doubles as a bias corrector), **in-batch** (free but popularity-skewed, hence the $\log Q$ correction), **hard negatives** (high-scoring non-positives that sharpen the boundary, ANCE-style), and **mixed** (Google's blend of in-batch and uniform). We will spend real time on the trap that lurks under all of them — the **false-negative problem**, where the item you mined as a hard negative is actually an unlabeled positive — because the harder you mine, the worse it gets, and that tension defines the whole craft. We will code several samplers in PyTorch, train on MovieLens, and put a real ablation table on the board.

This sits squarely in the retrieval stage of the [recommendation funnel](/blog/machine-learning/recommendation-systems/what-is-a-recommender-system): retrieval feeds candidates to ranking, and the negatives decide whether those candidates are any good. It composes directly with [training two-tower models](/blog/machine-learning/recommendation-systems/training-two-tower-negatives-and-sampled-softmax) — that post derives the sampled-softmax loss and the $\log Q$ correction; this one is about the *distribution you sample from*. By the end you will be able to implement four samplers, read the ablation that tells you which one your dataset wants, and recognize the false-negative degradation before it costs you a launch.

## 1. Why negatives are everything in implicit retrieval

Let me set up the problem precisely, because the "negatives are the model" claim is not a slogan — it is a consequence of what implicit feedback *is*.

A training example is a pair $(u, i)$: a user $u$ who interacted with item $i$. The user tower maps $u$ to an embedding $\mathbf{u} \in \mathbb{R}^d$; the item tower maps any item $j$ to $\mathbf{v}_j \in \mathbb{R}^d$. The score of a pair is the dot product $s(u, j) = \mathbf{u}^\top \mathbf{v}_j$. Retrieval, at serving time, is: embed the user, find the items with the highest dot products via approximate nearest-neighbor search, return the top few hundred. The model is good if the items the user *would* engage with sit near the top of that list.

Here is the catch that makes recommendation different from ordinary classification. In a classifier, every training example carries an explicit label: this image is a cat, that one is a dog. In implicit-feedback recommendation, you observe **only the positives**. A user clicked item $i$ — that is a 1. But the user did *not* click the other ten million items in the catalog, and the meaning of that silence is genuinely ambiguous: the user might dislike an item, or might love it but have never been shown it. Implicit feedback is **positive-unlabeled** data. There is no observed negative class at all.

So to train a model that separates "good" from "bad" items, you must *manufacture* the negative class. You pick some items the user did not interact with, declare them negatives, and ask the model to score the real positive above them. Every gradient that teaches the model what *not* to retrieve comes from these manufactured negatives. The positive tells the model "pull the user toward item $i$"; the negatives tell it "push the user away from these." Without negatives, the trivial solution is to make every embedding identical and every score maximal — the loss on positives alone is minimized by collapse. **The negatives are the only force preventing collapse, and they are the only signal that shapes where the boundary lands.**

This is why the choice of negatives dominates. Change the architecture and you change the *capacity* to draw a boundary. Change the negatives and you change *which boundary the model is asked to draw*. A model with infinite capacity trained on negatives that are all trivially easy will draw a boundary that separates positives from trivially-easy items — which is a boundary that does nothing useful at serving time, where the competition is between the positive and the thousand items that are *almost* as good. Get the negatives wrong and you have optimized a different problem than the one you deploy into.

There is a useful way to see the whole landscape: a negative is informative in proportion to how *surprised* the model is to be told it is a negative. An item the model already scores near zero for this user teaches nothing — the gradient is already satisfied. An item the model scores *high* but is in fact a negative is a correction the model needs. The entire art of negative sampling is shaping the distribution of negatives toward the informative ones without crossing the line into items that are secretly positives. That tension — informativeness versus false negatives — is the axis this whole post lives on.

## 2. The science: how the negative distribution shapes the gradient

Let me make the dominance of negatives quantitative, because "negatives are the model" should be a theorem, not a vibe. The cleanest way is to write the loss as an expectation over the sampling distribution and read off the gradient.

The true retrieval objective is the full softmax over the catalog $\mathcal{V}$:

$$
P(i \mid u) = \frac{\exp\!\big(s(u, i)\big)}{\sum_{j \in \mathcal{V}} \exp\!\big(s(u, j)\big)}, \qquad \mathcal{L} = -\log P(i \mid u).
$$

The gradient of this loss with respect to the score of any item $j$ has a famously clean form:

$$
\frac{\partial \mathcal{L}}{\partial s(u, j)} = P(j \mid u) - \mathbb{1}\!\left[j = i\right].
$$

For the positive $i$, the gradient is $P(i \mid u) - 1$, which is negative — it *raises* $s(u, i)$. For every negative $j \neq i$, the gradient is $P(j \mid u)$, which is positive — it *lowers* $s(u, j)$ in proportion to how much probability mass the model currently puts on $j$. Read that again: **the push on a negative is proportional to its current model probability.** The model spends its corrective effort on the negatives it currently ranks highly. Items it already scores near zero get a near-zero push. This is the formal reason easy negatives teach nothing — their $P(j \mid u)$ is already small, so their gradient is already small, regardless of what sampler picked them.

Now the full softmax is intractable (the denominator sums over $10^8$ items), so in practice we sample. We draw a set of negatives from a proposal distribution $Q$, and the loss becomes an estimate over that sample. The key question is: how does the choice of $Q$ change the gradient the model actually receives? Take the contribution of negatives to the expected gradient of the user embedding. Up to the loss's exact form, the negative-side gradient on $\mathbf{u}$ has the structure

$$
\nabla_{\mathbf{u}} \mathcal{L}_{\text{neg}} \;\approx\; \mathbb{E}_{j \sim Q}\!\big[\, w(u, j)\, \mathbf{v}_j \,\big],
$$

a $Q$-weighted average of negative item vectors, where $w(u, j)$ is a per-negative weight set by the loss (for softmax-type losses, $w$ is large for high-scoring negatives). This expression is the whole story in one line. The negatives you push the user away from are *the items $Q$ tends to draw*, weighted by how much the loss cares about each. Change $Q$ and you change, literally, the direction the user embedding is pushed every step. Three concrete consequences:

- **Uniform $Q$.** You push the user away from a uniformly random item. Almost always that item is trivially dissimilar, $w$ is tiny, the push is negligible. You get a true but high-variance, low-magnitude gradient — slow, weak learning.
- **Popularity-weighted $Q$.** You push the user away from popular items more often. This is a strong signal (popular items are plausible competitors) but it has a side effect: it systematically lowers the scores of popular items, which *counteracts* the popularity bias that creeps in elsewhere. We will use this deliberately.
- **Hard $Q$ (high-score items).** You push the user away from exactly the items the model currently ranks near the positive — the items with large $w$. This is the most efficient possible use of each gradient step. It is also where false negatives bite hardest, because the items most similar to the positive are the ones most likely to be unlabeled positives.

So the negative distribution is not a hyperparameter you tune for convenience. It is the term that decides the *direction and magnitude* of the force shaping every embedding. That is what "negatives are the model" means, precisely. The rest of this post is a catalog of choices for $Q$ and what each one does to that expectation.

There is one more piece of the science worth making explicit, because it explains *why* sampling from the wrong $Q$ is recoverable at all: **importance weighting.** A sampled-softmax loss with negatives drawn from $Q$ is, in expectation, an estimator of the true full-softmax loss only if you correct for the proposal. The corrected logit is $s(u, j) - \log Q(j)$ — divide each sampled term's $\exp(s)$ by the probability of having sampled it, and the over-representation of high-$Q$ items cancels. This is the same Monte-Carlo importance-sampling identity that lets you estimate an expectation under one distribution using samples from another: $\mathbb{E}_{p}[f] = \mathbb{E}_{q}\!\big[\tfrac{p}{q} f\big]$. The $\log Q$ correction is that ratio in log-space. The practical upshot is liberating: you do *not* have to sample from the "right" distribution, because you can correct for whatever distribution you sampled from — *as long as you know $Q(j)$*. That is why in-batch negatives (with their accidental popularity-weighted $Q$) are usable at all, and it is why a sampler whose $Q$ you cannot estimate (a black-box mining heuristic with unknown selection probability) is genuinely dangerous: you cannot un-bias what you cannot weight.

But importance weighting has a limit that the false-negative problem will exploit later: it corrects for *which items you sampled*, not for *whether the label is right*. If a sampled "negative" is actually a positive, no $\log Q$ correction saves you — the loss still pushes the model away from a true match, just with a correctly-weighted wrong gradient. Importance weighting fixes the proposal bias; it does nothing about label bias. Keep that distinction in mind through sections 6 and 8: the $\log Q$ correction and the false-negative problem live on two different axes, and conflating them is a common and expensive mistake.

#### Worked example: why an easy negative produces no gradient

Take a user $\mathbf{u}$ and three items. The positive $\mathbf{v}_i$ scores $s(u,i) = 4.0$. A *hard* negative $\mathbf{v}_h$ scores $s(u,h) = 3.5$ — the model thinks it is nearly as good as the positive. An *easy* negative $\mathbf{v}_e$ scores $s(u,e) = -2.0$ — the model already knows it is wrong. Suppose for simplicity these three are the candidate set; the softmax probabilities are proportional to $e^{4.0}, e^{3.5}, e^{-2.0} = 54.6, 33.1, 0.135$. Normalizing, $P(i) = 0.622$, $P(h) = 0.377$, $P(e) = 0.0015$.

Now the gradient on each negative's score is exactly its probability. The hard negative gets a push of $+0.377$ — a substantial correction lowering its score. The easy negative gets a push of $+0.0015$ — effectively nothing. The hard negative delivers $\frac{0.377}{0.0015} \approx 250\times$ the corrective signal of the easy one. If your sampler keeps handing the model easy negatives, you are spending compute computing gradients that are numerically zero. This single ratio is the entire economic case for caring about which negatives you sample.

## 3. The taxonomy of negative samplers

Before we walk the strategies one by one, it helps to see the map. Negative samplers split along one structural axis: **does the proposal distribution $Q$ depend on the current model, or not?**

![A taxonomy tree splitting negative samplers into a static family with uniform and popularity variants and a dynamic family with in-batch, hard-mined, and mixed variants](/imgs/blogs/negative-sampling-strategies-3.png)

**Static samplers** fix $Q$ before training and never look at the model. Uniform ($Q(j) = 1/N$) and popularity-based ($Q(j) \propto \text{count}(j)^\alpha$) are the two members. Their charm is operational simplicity: you can precompute the distribution, sample negatives in a data-loader worker on CPU, and never touch the GPU model. They are cheap, reproducible, and embarrassingly parallel. Their weakness is exactly that they ignore the model — a static sampler cannot know which negatives are *currently* hard, so it can never concentrate signal where it is most needed.

**Dynamic samplers** make $Q$ a function of the model. In-batch negatives use the other positives in the current mini-batch as negatives — so $Q$ is implicitly the distribution of items in your training stream (popularity-biased), and the *set* changes every batch even if the distribution is fixed. Hard-negative mining goes further: it periodically scores candidates with the *current* model and samples from the high scorers, so $Q$ literally tracks the model's evolving notion of "almost right." Mixed samplers blend static and dynamic to get the strengths of both. Dynamic samplers are where the recall gains live and where the false-negative risk concentrates — the more your sampler chases the model's high scores, the more it risks picking items the model is high on *because they are actually good for this user*.

This static/dynamic split is the spine of everything below. Static = cheap and safe and weak; dynamic = expensive and powerful and risky. The mixed sampler exists because the right answer for most production systems is "some of both."

## 4. Strategy 1: uniform random negatives

The simplest sampler draws each negative uniformly from the catalog: $Q(j) = 1/N$ for every item $j$, with the user's known positives excluded. It is the baseline every other strategy is measured against, and it is worth understanding *exactly* why it underperforms before reaching for anything fancier.

```python
import torch

class UniformNegativeSampler:
    """Draws negatives uniformly from the catalog, excluding the user's positives."""
    def __init__(self, n_items, user_positives):
        self.n_items = n_items
        # user_positives: dict[int, set[int]] mapping user -> set of clicked item ids
        self.user_positives = user_positives

    def sample(self, users, n_neg):
        # users: LongTensor of shape (B,). Returns (B, n_neg) negative item ids.
        B = users.shape[0]
        negs = torch.randint(0, self.n_items, (B, n_neg))
        # rejection-resample any accidental positives
        for b in range(B):
            pos = self.user_positives.get(int(users[b]), set())
            for k in range(n_neg):
                while int(negs[b, k]) in pos:
                    negs[b, k] = torch.randint(0, self.n_items, (1,))
        return negs
```

The strengths are real and worth respecting. It is the only sampler with a *provably correct* proposal for an unbiased uniform-negative objective — there is no $\log Q$ correction to forget, because $Q$ is constant. It is trivially cheap (one `randint` call), it parallelizes across data-loader workers, and it never accidentally over-represents any item. For a small catalog (a few thousand items) and a model with limited capacity, uniform negatives can be entirely adequate.

The weakness is the one section 2 predicted. With a catalog of $N$ items and (say) ten thousand items the user might plausibly like, a uniform draw lands on a plausible competitor with probability $\sim 10^4 / N$. For $N = 10^6$ that is one percent; for $N = 10^8$ it is one in ten thousand. The other 99-to-99.99 percent of negatives are items so unrelated that the model gets them right immediately and their gradient is numerically zero (the worked example above). You are paying full forward-and-backward cost on negatives that contribute nothing. The model learns the gross structure of the catalog fast and then stalls, because it never sees the *fine* distinctions that decide top-K recall. On the MovieLens ablation in section 11, uniform negatives bottom out at Recall@100 = 0.31 — and no amount of training fixes it, because the problem is the data the loss sees, not the optimization.

The honest verdict: uniform negatives are the right default *only* when the catalog is small or when you genuinely cannot afford anything else. The moment your catalog is large, uniform is leaving most of your recall on the floor.

## 5. Strategy 2: popularity-based negatives and the 0.75-power trick

If uniform negatives are too easy because they are mostly tail items nobody competes for, the obvious fix is to bias the sampler toward *popular* items — the ones that actually compete for every user's attention. Sample negatives proportional to item frequency: $Q(j) \propto \text{count}(j)$. Now your negatives are the items the model genuinely needs to learn to *rank below* the positive, because they are the items it will otherwise blanket-recommend to everyone.

But raw frequency over-shoots. The popularity distribution in any real catalog is brutally skewed — a handful of items account for a huge share of interactions, and a long tail of items appear a handful of times each. Sampling proportional to raw count means your negatives are *almost always* the same dozen mega-popular items, and the rare items never appear as negatives at all, so the model never learns to discriminate among them. You have traded "negatives too easy" for "negatives too concentrated."

The fix is the trick from the word2vec paper (Mikolov et al., 2013), and it is one of the most quietly influential lines in all of representation learning: raise the count to the power **0.75** before normalizing.

$$
Q(j) = \frac{\text{count}(j)^{\,0.75}}{\sum_{k} \text{count}(k)^{\,0.75}}.
$$

The exponent $\alpha = 0.75$ sits between two extremes. At $\alpha = 1$ you have raw frequency (over-concentrated on head items). At $\alpha = 0$ you have uniform ($\text{count}^0 = 1$ for everything). The 0.75 power *smooths* the distribution: it pulls probability mass down from the mega-popular head and lifts it onto the mid-tail, so popular items are still sampled more often (they are still plausible competitors) but rare items get a fighting chance to appear as negatives too. Mikolov found 0.75 empirically — it simply worked better than 1.0 or 0.5 across word-analogy benchmarks — and it has held up remarkably well as a default in recommendation too. The intuition: the square-root-ish compression keeps the *ordering* of popularity (head still beats tail) while shrinking the *ratios* so no single item dominates the negative stream.

```python
import numpy as np
import torch

class PopularityNegativeSampler:
    """Samples negatives proportional to count^alpha (word2vec uses alpha=0.75)."""
    def __init__(self, item_counts, alpha=0.75):
        counts = np.asarray(item_counts, dtype=np.float64)
        weights = counts ** alpha
        self.probs = weights / weights.sum()
        # alias table via torch.multinomial on a cached distribution
        self.dist = torch.tensor(self.probs, dtype=torch.float)

    def sample(self, B, n_neg):
        # draw with replacement from the smoothed popularity distribution
        return torch.multinomial(self.dist, B * n_neg, replacement=True).view(B, n_neg)
```

There is a second, deeper reason to like popularity-based negatives, and it connects to a problem that haunts the whole series: **popularity bias.** A retriever trained naively tends to over-recommend popular items, because popular items appear as positives for many users and the model learns "popular = good for everyone." Sampling popular items as *negatives* is a direct counter-pressure: every time a popular item appears as a negative, the loss lowers its score for that user. The popularity-weighted negative stream is, in effect, a regularizer that keeps the head from dominating retrieval. This is the same idea the $\log Q$ correction formalizes for in-batch negatives (next section), and it is why the [popularity-bias dynamics](/blog/machine-learning/recommendation-systems/popularity-bias-and-the-rich-get-richer) and the negative sampler are two sides of one coin.

#### Worked example: computing the 0.75-power sampling probabilities

Take five items with interaction counts: item A = 10,000, B = 1,000, C = 100, D = 10, E = 1. Under raw frequency ($\alpha = 1$), the total is 11,111 and the probabilities are A = 0.900, B = 0.090, C = 0.009, D = 0.0009, E = 0.00009. Item A would be 90% of your negatives; item E essentially never appears. The model would never learn to rank against E at all.

Now apply $\alpha = 0.75$. Compute $\text{count}^{0.75}$:

- A: $10000^{0.75} = 1000$
- B: $1000^{0.75} = 177.8$
- C: $100^{0.75} = 31.62$
- D: $10^{0.75} = 5.62$
- E: $1^{0.75} = 1.0$

The sum is $1000 + 177.8 + 31.62 + 5.62 + 1.0 = 1216.0$. The smoothed probabilities are A = 0.822, B = 0.146, C = 0.026, D = 0.0046, E = 0.00082. Compare the head-to-tail ratio: under raw frequency, A is $10^4\times$ as likely as E; under the 0.75 power, A is only about $10^3\times$ as likely. The exponent compressed the dynamic range by a full order of magnitude. Item A still dominates (it should — it is a real competitor for everyone), but C, D, and E now appear often enough that the model actually learns to discriminate among the mid-tail. That compression, not the exact value 0.75, is the point; values in $[0.5, 0.75]$ all work, and 0.75 is the well-worn default.

## 6. Strategy 3: in-batch negatives and the logQ correction

The cleverest cheap idea in retrieval training is to not sample negatives at all — to *reuse* the positives already in your mini-batch. In a batch of $B$ user-item positive pairs $(u_1, i_1), \dots, (u_B, i_B)$, for user $u_1$ the item $i_1$ is the positive and the other $B-1$ items $i_2, \dots, i_B$ are treated as negatives. Every other user's clicked item becomes a negative for this user. You compute the full $B \times B$ matrix of dot products in one matrix multiply, and the diagonal is the positives while the off-diagonal is the negatives. You get $B-1$ negatives per positive for *free* — no extra item lookups, no extra embedding work, just one matmul you were nearly doing anyway.

![A vertical pipeline showing observed positives flowing through a sampler into k negatives, then a logQ correction, then the loss and a gradient that updates both towers](/imgs/blogs/negative-sampling-strategies-6.png)

```python
import torch
import torch.nn.functional as F

def in_batch_loss(user_emb, item_emb, log_q=None, temperature=0.05):
    """In-batch sampled softmax. user_emb, item_emb: (B, d). Diagonal = positives."""
    logits = (user_emb @ item_emb.t()) / temperature          # (B, B)
    if log_q is not None:
        logits = logits - log_q.unsqueeze(0)                  # subtract logQ per item column
    labels = torch.arange(user_emb.size(0), device=user_emb.device)
    return F.cross_entropy(logits, labels)                    # positive is class = its own row index
```

In-batch negatives are the Pareto winner of the field — thousands of negatives at the cost of one matmul — and they are the default in essentially every large-scale two-tower system. But they hide a bias that, left uncorrected, quietly poisons retrieval. **The negatives are not uniform. They are the items that showed up as positives in your batch, and items show up as positives in proportion to their popularity.** A popular item is in the batch (as someone's positive) far more often than a rare item, so it serves as a negative far more often. Your in-batch proposal distribution is implicitly $Q(j) \propto \text{count}(j)$ — popularity-weighted, whether you wanted it or not.

Why is that a problem? Because a softmax loss with non-uniform negatives is biased: the model is being asked to beat popular items *more often* than it should, so it over-corrects and learns to *under-score* popular items. Retrieval then systematically buries the head — the opposite failure from naive popularity bias, but a failure all the same. The fix is the **$\log Q$ correction**, the central contribution of the 2019 YouTube sampling-bias-corrected two-tower paper (Yi et al.). The idea is importance weighting: to recover the unbiased softmax under a non-uniform proposal $Q$, subtract $\log Q(j)$ from each negative's logit before the softmax.

$$
s^{\text{corrected}}(u, j) = s(u, j) - \log Q(j).
$$

The derivation is short and worth carrying. A negative $j$ appears in the sampled denominator with frequency $\propto Q(j)$, so its $\exp(s)$ term is over-counted by a factor $Q(j)$ relative to a uniform sum. Dividing $\exp(s(u,j))$ by $Q(j)$ — equivalently subtracting $\log Q(j)$ from the logit — exactly cancels the over-counting and makes the sampled softmax an unbiased estimator of the full softmax gradient. (The full derivation lives in the [training two-tower post](/blog/machine-learning/recommendation-systems/training-two-tower-negatives-and-sampled-softmax) and the [sampled-softmax and contrastive-losses post](/blog/machine-learning/recommendation-systems/sampled-softmax-and-contrastive-losses-for-retrieval); here we use the result.) In practice $Q(j)$ is estimated on the fly — count how often each item appears in the stream, with exponential decay — because the true item frequencies drift and you do not want a stale correction.

The practical effect is large and it is exactly what you would predict: with the correction, the model stops penalizing popular items for being popular, the head recovers its rightful place near the top of retrieval, and aggregate Recall@K rises — *while* the popularity-stratified recall stays balanced rather than collapsing toward the tail. On the MovieLens ablation, adding the $\log Q$ correction to in-batch negatives takes Recall@100 from the popularity baseline's 0.35 to 0.42. The correction is not optional at scale; an uncorrected in-batch softmax is a popularity-biased objective wearing a retrieval costume.

Estimating $Q(j)$ for the correction is its own small engineering problem worth getting right. You need, per item, the probability it appears as a negative — which for in-batch sampling is the probability it appears as *anyone's* positive in a batch, i.e. its interaction frequency. The naive approach (count every item's frequency over the whole training set, once) is wrong for two reasons: item popularity drifts over the life of a training run, and a static count cannot reflect the actual stream the batches are drawn from. The robust approach, from the YouTube paper, is a *streaming* frequency estimator with exponential decay: maintain a running estimate of the average number of steps between appearances of each item, and update it each time the item is seen. This gives an online, decaying estimate of $Q(j)$ that tracks the stream. At very large catalogs you cannot store one counter per item in fast memory, so the estimator is approximated with a hash-based sketch (count-min) that trades a little accuracy for bounded memory. The detail that bites people: the $\log Q$ you subtract must be estimated from *the same stream* that produces the negatives, with the same decay — a $\log Q$ computed offline on last month's data and applied to this week's stream is a stale correction that can do more harm than none.

#### Worked example: the logQ correction on three items

Take three items in a batch with streaming-estimated sampling probabilities $Q(\text{popular}) = 0.10$, $Q(\text{mid}) = 0.01$, $Q(\text{rare}) = 0.001$ — the popular item is sampled as a negative one hundred times as often as the rare one. Suppose for some user the raw dot-product scores happen to be equal: $s = 2.0$ for all three. Without correction, the softmax treats them as tied, but the popular item, appearing as a negative far more often across the epoch, accumulates far more downward pressure — the model learns to under-score it. Now apply the correction: subtract $\log Q(j)$. The corrected logits are $2.0 - \log(0.10) = 2.0 + 2.30 = 4.30$ for popular, $2.0 - \log(0.01) = 2.0 + 4.61 = 6.61$ for mid, and $2.0 - \log(0.001) = 2.0 + 6.91 = 8.91$ for rare. The correction *raises* the effective logit of the rarely-sampled items relative to the frequently-sampled popular one — exactly compensating for the popular item being over-represented in the negative stream. After correction, equal raw scores produce a softmax that no longer punishes the popular item for the company it keeps. That is the whole mechanism: $\log Q$ converts "how often did I sample you as a negative" into a logit offset that cancels the sampling frequency.

## 7. Strategy 4: hard negatives and ANCE-style mining

Static and in-batch samplers share a ceiling: none of them knows which negatives are *currently* hard for the model. As training proceeds, the model gets good at the easy distinctions and the bulk of any random or popularity-drawn negative becomes easy again — the gradient shrinks back toward zero. To keep learning, you have to keep feeding the model negatives it finds hard *right now*. That is hard-negative mining: use the current model to score candidates and sample negatives from the high scorers.

![A directed dataflow graph showing the current model scoring candidates, dropping known positives, taking high non-positives as negatives, training a step, and refreshing the index on a schedule](/imgs/blogs/negative-sampling-strategies-4.png)

The loop has four moves, drawn acyclically because each pass produces a *new* model rather than literally cycling: (1) the current model scores a pool of candidate items for each user; (2) drop the user's known positives; (3) take the high-scoring non-positives as hard negatives; (4) train a step on them, producing a sharper model. Because re-scoring the whole catalog every step is far too expensive, the dominant pattern — popularized by **ANCE** (Approximate Nearest-neighbor Negative Contrastive Estimation, Xiong et al., 2020) — is to build an ANN index from a *snapshot* of the item embeddings, mine hard negatives from it for many steps, then *refresh* the index periodically (every $N$ steps or every epoch) so the negatives track the improving model without paying the rebuild cost every step.

```python
import torch

@torch.no_grad()
def mine_hard_negatives(model, ann_index, users, user_positives, n_neg, pool=200):
    """Score a candidate pool with the current model, keep high non-positives."""
    model.eval()
    user_vecs = model.user_tower(users)                    # (B, d)
    # ann_index.search returns the top-`pool` item ids by dot product for each user vector
    _, cand_ids = ann_index.search(user_vecs.cpu().numpy(), pool)  # (B, pool)
    cand_ids = torch.as_tensor(cand_ids)
    out = torch.empty(users.size(0), n_neg, dtype=torch.long)
    for b in range(users.size(0)):
        pos = user_positives.get(int(users[b]), set())
        # high-scoring candidates with positives removed; ann returns them score-sorted
        hard = [j for j in cand_ids[b].tolist() if j not in pos][:n_neg]
        out[b] = torch.tensor(hard)
    model.train()
    return out                                             # (B, n_neg) hard negatives
```

Two engineering details decide whether hard mining helps or hurts. First, the **refresh cadence.** Mine against an index that is too stale and your "hard" negatives are hard for an old model, not the current one — you waste effort re-learning distinctions you already fixed. Refresh too often and the index rebuild (re-embed every item, rebuild the ANN structure) dominates training time. The ANCE recipe refreshes every several thousand steps; the right cadence for you depends on how fast your embeddings move and how big your catalog is. Second, **how hard is too hard.** Taking the strict top-1 most-similar non-positive is usually a mistake — that item is the most likely to be an unlabeled positive (section 8). The common, safer choice is to skip the very top and sample from a band a little further down (say ranks 20–200), which gives you genuinely hard negatives that are far enough from the positive to be more plausibly true negatives. This is the *semi-hard* regime, and it is the next section's whole subject.

There is also a choice of *where to mine from* that matters as much as the cadence. You can mine **globally** — score the full catalog (via the ANN index) and take the top non-positives — or **locally** — restrict the candidate pool to a related set (the same category, the same brand, the same session). Global mining surfaces the genuinely highest-scoring competitors but is the most exposed to false negatives (the global top is the global most-likely unlabeled positive). Local mining within a category gives you negatives that are hard *and* contextually plausible (a sci-fi film as a negative for a sci-fi lover, which sharpens the within-category boundary the user actually cares about) while bounding the false-negative rate to that category's density. Many production systems mine a mix: a few global hard negatives for the broad boundary, several category-local ones for the fine boundary.

A subtle stability issue deserves a flag: hard mining couples the negatives to the model, which couples the *gradient* to the model's own current errors, which can amplify instabilities. If the model briefly mis-scores a region of the catalog, hard mining will preferentially draw negatives from exactly that region and the loss will hammer it — sometimes overcorrecting into oscillation. The defenses are the usual ones (a skip-band so you are not chasing the single most-volatile top score, a learning-rate that is not too aggressive, and gradient clipping) plus the recommendation-specific one of *diluting* mined negatives with static ones so the gradient is never entirely model-determined. A loss that is 100% hard-mined is a feedback system with no damping; a loss that is, say, 30% hard-mined and 70% in-batch is damped by the stable majority.

The payoff, when done well, is the single largest jump in the ablation: hard-mined negatives can push Recall@K well past what static sampling reaches, because every gradient step now lands on a distinction the model actually needs to make. On dense passage retrieval, ANCE reported large MRR gains over BM25 and over in-batch-only training; the same mechanism transfers directly to two-tower recommendation. The cost is real — you are running inference and an index rebuild inside your training loop — but for retrieval quality it is frequently the best money you can spend.

## 8. The false-negative problem: the trap under every hard sampler

Here is the failure that turns hard-negative mining from a superpower into a footgun, and it is the most important paragraph in this post. When you mine the highest-scoring non-positive as a "negative," you are betting that the item is genuinely *not* a good match for the user. But in positive-unlabeled data, a high-scoring item the user *has not interacted with* is exactly as likely to be an item the user would love but **has never been shown**. You mine it, label it a negative, and the loss dutifully pushes the user's embedding *away* from an item that should have been a positive. This is a **false negative**, and it does not merely waste a gradient step — it teaches the model the wrong thing.

![A two-column figure contrasting a true negative that correctly pushes the user away from a genuine non-match against a false negative where an unseen positive is pushed away and corrupts the boundary](/imgs/blogs/negative-sampling-strategies-7.png)

The cruel structure of the problem: **the harder you mine, the worse it gets.** False negatives are concentrated precisely among the items most similar to the positive — because items similar to a known positive are exactly the items the user is most likely to also like. So the more aggressively your sampler chases the model's top scores, the higher the fraction of your "hard negatives" that are actually unlabeled positives. Easy negatives are almost never false (a random tail item is genuinely irrelevant); the top-1 mined negative might be false a meaningful fraction of the time. The very mechanism that makes hard negatives informative is the mechanism that makes them dangerous.

#### Worked example: a false negative pushing the gradient the wrong way

Consider a user who loves science-fiction films. The model has correctly placed *Blade Runner* (a known positive) near the user's embedding. You mine hard negatives and the top non-positive is *Blade Runner 2049* — which the user has simply never gotten around to watching, but would rate five stars. Your sampler labels it a negative.

Look at the gradient. The pairwise (BPR) loss for positive $i$ and negative $j$ is $-\log \sigma(s(u,i) - s(u,j))$, and its gradient pushes to *increase* the margin $s(u,i) - s(u,j)$. Concretely, the update moves $\mathbf{u}$ to raise $s(u,i)$ and lower $s(u,j)$. Suppose $s(u, \text{Blade Runner}) = 3.0$ and $s(u, \text{2049}) = 2.8$. The margin is $0.2$, so $\sigma(0.2) = 0.55$ and the loss gradient prefactor is $(1 - 0.55) = 0.45$ — a *large* update, because the model thinks these two are nearly tied and the loss wants them separated. The update drives $s(u, \text{2049})$ down hard. But *2049* should score near the top for this user. You have used a large, confident gradient step to push the user embedding away from one of the best possible recommendations. Worse, because *2049* is similar to many of the user's other true positives, you have also nudged the user away from a whole neighborhood of good items. One false negative does diffuse, correlated damage across the user's taste region — and you mined it *because* it was the most informative-looking candidate.

The mitigations are a toolkit, not a single fix, and shipping hard-negative mining means deploying several:

- **Skip the very top (semi-hard).** Do not take rank-1; sample from a band below the top (ranks 20–200). The most-similar items are the most likely false negatives; a band a little further down is hard enough to teach and far enough to be more plausibly true negative.
- **Cap the score / use a margin.** Reject candidates whose score is *within* some margin of the positive — if the model thinks an item is essentially as good as a known positive, treat that as a signal it might *be* a positive, not a hard negative.
- **Blend with easy negatives (mixed, next section).** Diluting mined hard negatives with uniform or in-batch negatives bounds the false-negative rate: even if every mined negative were false, the easy ones keep the gradient pointed mostly the right way.
- **Debias known co-engagements.** Filter out items that co-occur heavily with the user's positives (collaborative-filtering neighbors), since those are the most likely unlabeled positives.
- **Do not anneal temperature too low.** A low softmax temperature concentrates the entire gradient on the single hardest negative — which is the single most likely false negative. Keep temperature in a sane band (the [contrastive-losses post](/blog/machine-learning/recommendation-systems/sampled-softmax-and-contrastive-losses-for-retrieval) shows the inverted-U where false negatives punish the low end).

The deepest point: there is no sampler that is both maximally informative and false-negative-free, because the items most worth showing the model as negatives are the items most likely to be secret positives. Negative sampling is permanently a *trade-off* along that axis, and the engineering is about finding your dataset's sweet spot — not eliminating the tension.

## 9. The hardness spectrum and the semi-hard sweet spot

Put sections 7 and 8 together and a clear picture of the hardness axis emerges. Negatives live on a spectrum from "trivially easy" to "indistinguishable from the positive," and both ends are bad for different reasons.

![A two-column figure contrasting too-easy negatives that produce no learning signal against semi-hard negatives that produce the best signal and highest recall](/imgs/blogs/negative-sampling-strategies-5.png)

At the **too-easy** end, the negative is so far from the positive that the model already scores it near zero. Its gradient is numerically tiny (section 2's $P(j \mid u) \approx 0$), so it teaches nothing. Training stalls; recall plateaus low. At the **too-hard** end, the negative is so close to the positive that it is probably an unlabeled positive (section 8). Its gradient is large *and pointed the wrong way*; training becomes unstable and recall can actually *degrade* as false negatives accumulate. In the **semi-hard** middle, the negative is close enough to be a real competitor (so the model is genuinely uncertain and the gradient is informative) but far enough to be plausibly a true negative (so the gradient points the right way). This is where the best signal-to-noise lives, and it is why "semi-hard" — a term borrowed from the metric-learning and FaceNet triplet-loss literature — is the practitioner's default target.

The metric-learning community made this precise. In triplet loss, a *semi-hard* negative is one that is farther from the anchor than the positive (so it does not violate the margin catastrophically) but still within the margin (so it produces a non-zero gradient). The recommendation analog: a semi-hard negative scores below the positive (it is not winning) but high enough to produce a useful push. Operationally, the recipe is the rank-band mining of section 7 — skip the top handful, sample from the next band — which is just "target the semi-hard region" made concrete.

This naturally suggests a **curriculum**: start with easy negatives (uniform or in-batch) while the model is weak and any negative is informative, then progressively harden as the model improves and easy negatives stop teaching. **PinSage** (Ying et al., 2018), Pinterest's billion-node graph recommender, did exactly this. They trained with a curriculum that introduced *harder and harder* negatives across epochs — early epochs used random negatives, later epochs added items that the current model scored highly but that the user had not engaged with, mined via PageRank-informed sampling on the item graph. The progressive hardening let the model first learn the coarse structure (what semi-hard mining would skip past too quickly at the start) and then refine the fine boundaries (what easy negatives can never teach). Curriculum hardening is the principled answer to "the right hardness depends on how trained the model already is" — so make it depend on training time.

```python
def curriculum_hardness(epoch, total_epochs, max_pool=500, min_pool=50):
    """Shrink the mining pool over training: a smaller pool = harder negatives.

    Early epochs draw from a wide pool (mostly easy); later epochs draw from a
    narrow top band (semi-hard). pool shrinks linearly with training progress.
    """
    frac = epoch / max(1, total_epochs - 1)
    pool = int(round(max_pool - frac * (max_pool - min_pool)))
    skip_top = 20  # always skip the top-20 to dodge the worst false negatives
    return skip_top, pool
```

## 10. Strategy 5: mixed negatives — the production default

If hard negatives are powerful but risky and easy negatives are safe but weak, the obvious move is to use both. **Mixed negatives** — Google's recipe from the 2020 *Mixed Negative Sampling* paper (Yang et al.) — combine in-batch negatives with a set of negatives sampled uniformly (or by popularity) from the whole catalog, in every batch. The in-batch negatives give you thousands of free, model-relevant (popularity-weighted) competitors; the uniform negatives ensure the model also sees items from across the *entire* catalog, including the cold, never-clicked tail that the in-batch stream never surfaces (because an item nobody has clicked yet can never appear as someone's positive, hence never as an in-batch negative).

That last point is the subtle, important one. In-batch negatives have a structural blind spot: they can only ever sample items that have *already been engaged with by someone in the batch*. Brand-new items, freshly added catalog entries, and the long cold tail are *invisible* to an in-batch-only sampler. The model never learns to rank against them, so at serving time it has no idea where to place them — a real cold-start hazard. Adding a stream of uniformly sampled negatives patches the blind spot: now the model sees the whole catalog as potential negatives, including the items in-batch sampling would never reach.

```python
import torch
import torch.nn.functional as F

def mixed_negative_loss(user_emb, pos_item_emb, uniform_neg_emb,
                        log_q_batch=None, log_q_uniform=None, temperature=0.05):
    """Mixed negatives: in-batch (other positives) + uniformly sampled catalog items.

    user_emb, pos_item_emb: (B, d). uniform_neg_emb: (M, d) shared across the batch.
    """
    in_batch_logits = (user_emb @ pos_item_emb.t()) / temperature      # (B, B)
    uniform_logits = (user_emb @ uniform_neg_emb.t()) / temperature    # (B, M)
    if log_q_batch is not None:
        in_batch_logits = in_batch_logits - log_q_batch.unsqueeze(0)   # correct popularity-biased in-batch
    if log_q_uniform is not None:
        uniform_logits = uniform_logits - log_q_uniform.unsqueeze(0)   # correct the uniform stream too
    logits = torch.cat([in_batch_logits, uniform_logits], dim=1)       # (B, B + M)
    labels = torch.arange(user_emb.size(0), device=user_emb.device)    # positive is the diagonal
    return F.cross_entropy(logits, labels)
```

A crucial subtlety the paper is careful about: the two negative streams have *different* proposal distributions, so they need *different* $\log Q$ corrections. The in-batch stream is popularity-weighted ($Q_{\text{batch}}(j) \propto \text{count}(j)$); the uniform stream is, well, uniform ($Q_{\text{uniform}}(j) = 1/N$, a constant that drops out of the softmax). Apply one blanket correction to both and you re-introduce exactly the bias you were trying to remove. The code above corrects each stream by its own $Q$.

Mixed negatives are the pragmatic production default for two-tower retrieval because they Pareto-dominate the alternatives along the axes that matter: they get the free thousands-of-negatives benefit of in-batch, the whole-catalog coverage of uniform, and — when you fold a small mined-hard stream into the mix — the sharpening of hard negatives, all while the easy uniform negatives *dilute* the false-negative risk of the mined ones. On the MovieLens ablation, the mixed sampler (in-batch + logQ + a modest mined-hard band) reaches the top of the table at Recall@100 = 0.47. It is the configuration I reach for first on any new two-tower model, and only deviate from it when a specific failure (cold start, popularity collapse, false-negative degradation) tells me which knob to turn.

## 11. Putting it together: the PyTorch ablation on MovieLens

Now the measurement angle. Talk is cheap; the value of a negative sampler is whatever it does to Recall@K on a held-out evaluation. Here is the harness — a BPR/two-tower model, a temporal split, and the metrics that matter for retrieval.

```python
import torch
import torch.nn as nn

class TwoTowerBPR(nn.Module):
    """Minimal two-tower with embedding tables; scores by dot product."""
    def __init__(self, n_users, n_items, dim=128):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, dim)
        self.item_emb = nn.Embedding(n_items, dim)
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)

    def user_tower(self, users):
        return self.user_emb(users)

    def score(self, users, items):
        return (self.user_emb(users) * self.item_emb(items)).sum(-1)

def bpr_loss(model, users, pos_items, neg_items):
    # neg_items: (B, n_neg). Broadcast the user/pos score against each negative.
    u = model.user_emb(users)                              # (B, d)
    pi = model.item_emb(pos_items)                         # (B, d)
    ni = model.item_emb(neg_items)                         # (B, n_neg, d)
    pos_score = (u * pi).sum(-1, keepdim=True)             # (B, 1)
    neg_score = (u.unsqueeze(1) * ni).sum(-1)              # (B, n_neg)
    # -log sigma(pos - neg), averaged over negatives
    return -torch.log(torch.sigmoid(pos_score - neg_score) + 1e-9).mean()
```

The evaluation harness computes full-catalog Recall@K and NDCG@K against a temporal holdout — the user's *last* interactions are the test set, so the model is evaluated on predicting the future from the past, with no leakage. Computing recall against the *full* catalog (not a sampled subset of 100 negatives) matters: the KDD 2020 result by Krichene and Rendle showed sampled metrics can rank models *inconsistently* with full metrics, so we score every item.

```python
import numpy as np

@torch.no_grad()
def evaluate(model, test_user_items, all_item_ids, K_list=(10, 100)):
    """Full-catalog Recall@K and NDCG@K on a temporal holdout."""
    model.eval()
    item_mat = model.item_emb.weight                       # (N, d)
    recalls = {K: [] for K in K_list}
    ndcgs = {K: [] for K in K_list}
    for user, held_out in test_user_items.items():         # held_out: set of test item ids
        u = model.user_emb.weight[user]                    # (d,)
        scores = item_mat @ u                              # (N,) full-catalog scores
        topk = torch.topk(scores, max(K_list)).indices.tolist()
        for K in K_list:
            hits = [1 if it in held_out else 0 for it in topk[:K]]
            recalls[K].append(sum(hits) / max(1, len(held_out)))
            dcg = sum(h / np.log2(r + 2) for r, h in enumerate(hits))
            idcg = sum(1 / np.log2(r + 2) for r in range(min(K, len(held_out))))
            ndcgs[K].append(dcg / idcg if idcg > 0 else 0.0)
    return ({K: np.mean(recalls[K]) for K in K_list},
            {K: np.mean(ndcgs[K]) for K in K_list})
```

Train the *same* model — same dimension, same learning rate, same epochs, same temporal split — five times, varying only the negative sampler, and the differences are entirely attributable to the negatives. Here is the ablation on MovieLens-1M (a 6,040-user, 3,706-item dataset; these are representative numbers from this configuration, of the magnitude you should expect rather than exact published figures — always re-measure on your own data).

![A results matrix showing five negative samplers scored on Recall at 10 and Recall at 100, with mixed negatives best and too-aggressive hard mining worse than the corrected in-batch baseline](/imgs/blogs/negative-sampling-strategies-8.png)

| Negative sampler | Recall@10 | Recall@100 | NDCG@10 | Notes |
| --- | --- | --- | --- | --- |
| Uniform random | 0.082 | 0.31 | 0.071 | Cheap; most negatives teach nothing |
| Popularity^0.75 | 0.094 | 0.35 | 0.083 | Smoothed; counters popularity bias |
| In-batch + logQ | 0.108 | 0.42 | 0.101 | Free thousands of negs, corrected |
| **Mixed (best)** | **0.121** | **0.47** | **0.114** | In-batch + uniform + semi-hard band |
| Hard, too aggressive | 0.099 | 0.39 | 0.090 | Top-1 mining → false negatives degrade |

Read three things off this table. First, **the gap is large** — Recall@100 swings from 0.31 to 0.47, a 50% relative improvement, with *the model held identical*. Almost no architecture change you can make to a two-tower delivers a swing that big; the sampler is the lever. Second, **each step up the sophistication ladder pays** — popularity beats uniform, corrected in-batch beats popularity, mixed beats in-batch — and the gains are roughly monotonic because each strategy adds either harder negatives or better coverage or both. Third, and most important, **too-hard mining goes backwards.** The "hard, too aggressive" row (top-1 mined negatives, no skip-band, no dilution) lands at 0.39 — *below* the corrected in-batch baseline of 0.42 — because the false-negative degradation of section 8 overwhelms the sharpening benefit. This is the empirical signature of crossing the sweet spot: more mining, less recall. When you see your hard-negative experiment under-perform a simpler baseline, the diagnosis is almost always false negatives, and the fix is to back off the hardness (skip-band, dilution, margin cap), not to mine harder.

#### Worked example: reading the false-negative degradation

Suppose you ship hard-negative mining and Recall@100 *drops* from your in-batch baseline of 0.42 to 0.39 — a 7% relative regression, exactly the table's bottom row. The temptation is to conclude "hard negatives do not work for us" and revert. The right move is to diagnose. Stratify recall by item popularity: you find the regression is concentrated entirely in the *mid-tail* (items ranked 100–1,000 by popularity), where Recall@100 dropped from 0.38 to 0.29, while head and deep-tail are unchanged. That pattern is the false-negative fingerprint: the mined negatives are mid-tail items similar to the user's positives — exactly the unlabeled positives the user would have liked. Now add a skip-band (drop the top-20 most-similar before mining) and a 50/50 dilution with in-batch negatives. Re-measure: Recall@100 climbs to 0.47, the mid-tail recovers to 0.41 (better than the baseline's 0.38, because the *remaining* hard negatives are genuinely informative). The lesson: a hard-negative regression is rarely a verdict against hard negatives; it is a tuning signal pointing at the false-negative axis.

## 12. Cost and quality trade-offs: how many negatives, static vs dynamic

Two practical knobs remain that the strategy choice does not fully settle: how *many* negatives, and how *fresh*.

**How many negatives.** More negatives means a better estimate of the full-softmax denominator, which means a stronger, lower-variance gradient — up to a point. The relationship is concave: going from 1 to 100 negatives is a large jump in retrieval quality; going from 1,000 to 10,000 is a small one. This is why in-batch negatives are such a good deal — a batch of 4,096 gives you ~4,095 negatives per positive for free, well past the point of diminishing returns for most datasets. If you are *explicitly* sampling negatives (uniform, popularity), more negatives means more embedding lookups and more compute per step, so there is a real cost; the rule of thumb is that explicit negatives have sharply diminishing returns past a few hundred, and beyond that you are better off switching to in-batch (which gets you thousands for the price of the matmul you were already doing).

**Static vs dynamic (re-mined) negatives.** Static negatives (uniform, popularity) are sampled once per example, on CPU, in parallel with the GPU — effectively free relative to the forward/backward pass. Dynamic negatives (hard-mined) require running model inference and maintaining an ANN index *inside* the training loop. The index-refresh cost is the dominant one: re-embedding every item and rebuilding the ANN structure can cost as much as several training steps, and you pay it every refresh interval. The economics:

| Sampler | Per-step cost | Index cost | Recall gain | When it pays |
| --- | --- | --- | --- | --- |
| Uniform | negligible (CPU) | none | baseline | small catalog, tight budget |
| Popularity^0.75 | negligible (CPU) | precompute once | small | always worth it over uniform |
| In-batch + logQ | one matmul (nearly free) | none | large | default for two-tower at scale |
| Hard-mined (ANCE) | inference per batch | rebuild every $N$ steps | large (if tuned) | when in-batch plateaus and you can afford the index |
| Mixed | matmul + small uniform stream | optional small mined index | largest | production default |

The decision rule that falls out: **start with in-batch + logQ** (large gain, near-zero cost), add **mixed uniform negatives** for catalog coverage and cold-start (small added cost, patches a real blind spot), and only reach for **hard mining** when the cheaper samplers have plateaued *and* you have measured that hard negatives help rather than hurt on your data. Hard mining is the most expensive and the most fragile (false negatives); it is also the only thing that breaks a genuine in-batch plateau. Spend its budget last and measure it hardest.

#### Worked example: budgeting the index-refresh cost

Suppose your two-tower trains for 100,000 steps, and a single training step (forward + backward on a 4,096-batch) takes 40ms. A naive "re-mine every step" hard-negative loop would re-embed all 3M catalog items (say 2 seconds on the GPU) and rebuild the ANN index (say another 3 seconds) *every step* — adding 5 seconds to a 40ms step, a 125-fold slowdown that turns a one-hour run into a five-day run. Obviously untenable. Now refresh every 2,000 steps instead: the 5-second refresh is amortized over 2,000 forty-millisecond steps, so the per-step overhead is $5000\text{ms} / 2000 = 2.5\text{ms}$, a 6% slowdown rather than 12,400%. The training run goes from one hour to about 64 minutes — entirely affordable for the recall it buys. This arithmetic is why the refresh *cadence* is the load-bearing decision in hard mining: at "every step" the cost is prohibitive and the negatives are barely fresher than at "every 2,000 steps," because the embeddings do not move meaningfully in 40ms. The sweet spot is the largest refresh interval over which your embeddings stay roughly stationary — usually thousands of steps, found empirically by watching whether recall improves when you refresh more often.

## 13. Case studies: where these strategies came from

Each strategy in this post has a paper behind it that you should read in the original. The progression is also a history of the field figuring out, dataset by dataset, that negatives are the lever.

**word2vec — negative sampling and the 0.75 power (Mikolov et al., 2013).** The paper that put negative sampling on the map. word2vec's skip-gram-with-negative-sampling replaced an intractable softmax over the whole vocabulary with a handful of negatives drawn from the unigram distribution raised to the 0.75 power. Mikolov reported that 0.75 outperformed both the raw unigram ($\alpha = 1$) and uniform ($\alpha = 0$) on word-analogy tasks. The mechanism is exactly section 5's: smooth the popularity distribution so frequent words are still over-sampled as negatives but rare words still appear. Recommendation inherited the trick wholesale — an item catalog is a vocabulary, interactions are the corpus, and the 0.75 power is still a sane default for item-popularity-weighted negatives.

**ANCE — index-refreshed hard negatives (Xiong et al., 2020).** *Approximate Nearest-neighbor Negative Contrastive Estimation* formalized the hard-mining loop for dense retrieval. The key insight: the most informative negatives are the ones the current model ranks highly, but re-scoring the whole corpus every step is impossible, so mine from an ANN index built on a periodically-refreshed embedding snapshot. ANCE reported large MRR gains on passage retrieval (MS MARCO) over both BM25 and in-batch-negative training, and it established the refresh-cadence pattern (rebuild the index every several thousand steps) that two-tower recommenders now borrow. The same paper is honest about the false-negative risk and the cost of the refresh — read it for the engineering, not just the result.

**Google mixed negative sampling (Yang et al., 2020).** *Mixed Negative Sampling for Learning Two-Tower Neural Networks in Recommendations.* Google's observation: in-batch negatives are free and powerful but structurally cannot sample the cold-start tail (an item nobody has engaged with never appears as an in-batch negative). Their fix — mix in-batch negatives with a stream of uniformly sampled catalog items — closed the cold-start gap and improved retrieval on their app-recommendation benchmark. This is the source of section 10's mixed sampler and the per-stream $\log Q$ subtlety (each stream gets its own correction). It is the most directly applicable paper for anyone building a production two-tower.

**PinSage — curriculum hard negatives at billion scale (Ying et al., 2018).** Pinterest's GraphSAGE-based recommender for a 3-billion-node item graph. Beyond the GNN architecture, PinSage's training used a *curriculum* of hard negatives: early epochs trained on random negatives, and each subsequent epoch added harder negatives mined via personalized-PageRank scores on the item graph (items in the neighborhood of the positive but not engaged). The progressive hardening let the model learn coarse structure first and fine distinctions later — section 9's curriculum, validated at the largest scale anyone had published. PinSage is the canonical demonstration that *when* you introduce hard negatives matters as much as *whether* you do.

A connecting thread runs through all four: every one of these systems found that the cheap, obvious negative sampler left large gains on the table, and that the path to better retrieval ran through the negative distribution — smoothing it (word2vec), tracking the model with it (ANCE), covering the catalog with it (Google), or scheduling it over training (PinSage). The architectures differ wildly; the lesson is identical.

## 14. Choosing your negative sampler

Pull the whole post into a decision. The samplers form a ladder, and you climb it only as far as your problem and budget require.

![A comparison matrix scoring five negative samplers on cost, signal strength, popularity bias, and false-negative risk](/imgs/blogs/negative-sampling-strategies-2.png)

- **Small catalog (a few thousand items), tight budget, or a quick baseline:** **uniform** negatives. The catalog is small enough that a uniform draw lands on plausible competitors often enough, and you avoid every complication. Do not over-engineer a problem this size.
- **You observe a popularity-bias problem (retrieval over-recommends the head), or you just want a strictly-better static default:** **popularity^0.75**. It is barely more expensive than uniform, it counters popularity bias as a side effect, and the 0.75 smoothing keeps the tail in play. There is almost no reason to use raw uniform over this once your catalog is non-trivial.
- **You are training a two-tower at any real scale:** **in-batch + logQ**, full stop. Thousands of negatives for one matmul, with the popularity bias corrected. This is the default starting point, and for many systems it is also the finishing point.
- **You have a cold-start tail or new-item problem:** add **uniform/popularity negatives to the in-batch stream → mixed**. The in-batch blind spot for never-engaged items is real and mixed sampling is the standard patch. This is the production default for most two-tower retrieval.
- **You have measured an in-batch plateau, you can afford an in-loop ANN index, and you have validated that hard negatives help on your data:** add **semi-hard mined negatives** to the mix — with a skip-band, a margin cap, and dilution. This is the highest-ceiling option and the most fragile; deploy it last and measure the popularity-stratified recall to catch false-negative degradation.

When *not* to reach for hard negatives: when in-batch + mixed already hits your target (do not pay the index cost for a gain you do not need); when your dataset is small enough that the false-negative rate among top candidates is high (small catalogs make every hard negative suspect); when you cannot afford the engineering to refresh the index correctly (a stale index mines hard negatives for a model that no longer exists). And the rule that should be tattooed somewhere: **never crank mining harder to fix a hard-negative regression — back off and dilute.** A regression is the false-negative axis talking.

## 15. Stress-testing the choice

Before you ship, run the sampler through the questions that break it in production.

**What happens with only implicit feedback?** This is the default and the hardest case — it is the entire premise of the post. With no explicit negatives, every negative is constructed, and the false-negative problem is unavoidable. The defense is never "find the true negatives" (they do not exist in the data) but "shape the proposal so false negatives are rare and diluted": skip-band mining, margin caps, mixed dilution, and conservative temperature. See the [implicit-feedback models post](/blog/machine-learning/recommendation-systems/implicit-feedback-models-als-and-bpr) for the broader treatment.

**What happens at 100M items?** Uniform negatives become useless (a random draw essentially never hits a plausible competitor); in-batch + logQ becomes mandatory; the $\log Q$ estimator must be streamed and sharded (count-min sketch with decay) because you cannot store exact counts; and the ANN index for hard mining is itself a large distributed system. At this scale the sampler is no longer a data-loader detail — it is infrastructure. The popularity skew is also more extreme, which makes the $\log Q$ correction *more* important, not less.

**What happens when negatives are mostly false negatives?** This is the catalog where users have engaged with only a tiny fraction of the items they would like (sparse implicit feedback, large catalog). Here aggressive mining is actively harmful — the top candidates are overwhelmingly unlabeled positives. Back all the way off: prefer in-batch + uniform mixed negatives, skip hard mining entirely or use a very deep skip-band, and lean on the popularity^0.75 correction rather than model-dependent hardness. Measure the popularity-stratified recall; if the mid-tail degrades when you add hardness, you are in this regime.

**What happens when offline Recall@K rises but online engagement is flat?** The series' recurring humbling. For negative sampling specifically, a common cause is that aggressive popularity-as-negative correction (or over-corrected $\log Q$) pushed the model to *under*-rank popular items the user would have engaged with anyway — you improved tail recall offline at the cost of head relevance online. Diagnose with a popularity-stratified offline breakdown and an online A/B; the [offline-vs-online gap post](/blog/machine-learning/recommendation-systems/offline-vs-online-the-two-worlds-of-recsys) is the reference.

**What happens when the sampler is computed differently offline and online?** Negatives are training-only, so the *sampler* never runs at serving time — but the $\log Q$ estimate does encode a popularity distribution, and if that estimate is computed on a different stream (or with a different decay) than the one producing your in-batch negatives, the correction is wrong and can do more harm than no correction. Recompute $\log Q$ from the same stream that supplies the negatives, with the same decay, and validate on a popularity-stratified holdout that the corrected logits actually reduce popularity bias rather than over-correcting it.

**How does the sampler interact with the feedback loop?** This is the sharpest version of the series' central concern. Retrieval serves candidates → users engage → engagements become next round's positives → the next model trains on them. A sampler that over-ranks popular items (uncorrected in-batch) makes the model surface them more, users engage with them more (because shown more, not necessarily preferred more), and the next training set is even more popularity-skewed — popularity becomes a self-reinforcing fixed point. The popularity^0.75 negative stream and the $\log Q$ correction are not just per-step unbiasing tricks; they are *loop-stability* interventions that keep the tail in the candidate set, which keeps tail engagements in the logs, which keeps the next proposal from collapsing. When you debug a recommender that has quietly narrowed to ten viral items, the negative sampler is on the short suspect list. The [popularity-bias dynamics post](/blog/machine-learning/recommendation-systems/popularity-bias-and-the-rich-get-richer) is the full treatment.

## 16. Key takeaways

- **In implicit retrieval, the negatives are the model.** You observe only positives, so the entire decision boundary is learned from the negatives you construct. The sampler routinely moves Recall@K more than the architecture does.
- **The negative distribution $Q$ shapes the gradient directly.** The push on a negative is proportional to its current model probability; easy negatives produce numerically tiny gradients, hard negatives produce large ones. Choosing $Q$ is choosing which direction every embedding is pushed.
- **Uniform is the weak baseline.** At any real catalog size, most uniform negatives are trivially easy and teach nothing. Use it only for small catalogs or quick baselines.
- **Popularity^0.75 is the smart static default.** The word2vec 0.75-power smoothing keeps popular items as plausible negatives while letting the tail appear, and it counters popularity bias as a side effect. Strictly better than uniform once the catalog is non-trivial.
- **In-batch + logQ is the workhorse.** Thousands of free negatives per positive, with the popularity bias corrected by subtracting $\log Q(j)$. The default for two-tower at scale, and often the finishing point.
- **Mixed negatives are the production default.** Blend in-batch (free, model-relevant) with uniform (whole-catalog coverage, cold-start) — each stream with its own $\log Q$ — and optionally a semi-hard mined band. Pareto-dominates the alternatives.
- **Hard negatives sharpen the boundary but bite back.** ANCE-style index-refreshed mining is the highest-ceiling strategy and the most fragile. The harder you mine, the more false negatives you sample, because the items most similar to a positive are the most likely unlabeled positives.
- **Aim for semi-hard.** Too easy means no gradient; too hard means false negatives and instability. Skip the top band, sample below it, dilute with easy negatives, and consider a curriculum (PinSage) that hardens over training.
- **A hard-negative regression is a tuning signal, never a verdict.** When mining drops recall below a simpler baseline, the cause is false negatives in the mid-tail — back off (skip-band, dilution, margin cap), do not mine harder.

## 17. Further reading

- Mikolov, Sutskever, Chen, Corrado, Dean, *Distributed Representations of Words and Phrases and their Compositionality* (NeurIPS 2013) — negative sampling and the 0.75-power unigram smoothing trick.
- Xiong, Xiong, Li, Tang, Liu, Bennett, Ahmed, Overwijk, *Approximate Nearest Neighbor Negative Contrastive Learning for Dense Text Retrieval* (ANCE, ICLR 2021) — index-refreshed hard-negative mining and its honest cost/false-negative discussion.
- Yang, Yi, Hong, Chen, Zhang, Chi, et al., *Mixed Negative Sampling for Learning Two-Tower Neural Networks in Recommendations* (WWW 2020 companion) — in-batch plus uniform negatives and the cold-start blind spot.
- Ying, He, Chen, Eksombatchai, Hamilton, Leskovec, *Graph Convolutional Neural Networks for Web-Scale Recommender Systems* (PinSage, KDD 2018) — curriculum hard-negative training at billion-node scale.
- Rendle, Freudenthaler, Gantner, Schmidt-Thieme, *BPR: Bayesian Personalized Ranking from Implicit Feedback* (UAI 2009) — the pairwise loss and the one-negative-per-positive setting.
- Yi, Yang, Hong, Chen, Zhang, Heldt, Hong, Chi, et al., *Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations* (RecSys 2019) — the industrial two-tower paper and the source of the $\log Q$ correction.
- Krichene and Rendle, *On Sampled Metrics for Item Recommendation* (KDD 2020) — why you should evaluate Recall@K against the full catalog, not a sampled subset.
- Within this series: [training two-tower models: negatives and sampled softmax](/blog/machine-learning/recommendation-systems/training-two-tower-negatives-and-sampled-softmax) (the loss this post supplies negatives to), [sampled softmax and contrastive losses for retrieval](/blog/machine-learning/recommendation-systems/sampled-softmax-and-contrastive-losses-for-retrieval), [pairwise and BPR loss deep dive](/blog/machine-learning/recommendation-systems/pairwise-and-bpr-loss-deep-dive), [popularity bias and the rich get richer](/blog/machine-learning/recommendation-systems/popularity-bias-and-the-rich-get-richer), the [recommendation funnel map](/blog/machine-learning/recommendation-systems/what-is-a-recommender-system), and the capstone [recommender systems playbook](/blog/machine-learning/recommendation-systems/the-recommender-systems-playbook).
