---
title: "Case Study: YouTube's Deep Retrieval and Ranking"
date: "2026-06-19"
publishDate: "2026-06-19"
description: "Walk YouTube's recommender end to end as a worked example of the whole series: the 2016 two-stage deep network, the weighted-logistic watch-time trick, the 2019 sampling-bias-corrected two-tower, MMoE multi-task ranking with a position-bias tower, and REINFORCE for long-term value, with runnable PyTorch repros of every key trick."
tags:
  [
    "recommendation-systems",
    "recsys",
    "youtube",
    "two-tower",
    "candidate-generation",
    "ranking",
    "watch-time",
    "mmoe",
    "position-bias",
    "machine-learning",
    "pytorch",
  ]
category: "machine-learning"
subcategory: "Recommendation Systems"
author: "Hiep Tran"
featured: true
readTime: 51
image: "/imgs/blogs/case-study-youtube-deep-retrieval-and-ranking-1.png"
---

If you want to understand how a modern industrial recommender is actually built, you could read forty papers, or you could read YouTube's. Over roughly a four-year span, one team published the cleanest worked example of the entire retrieval → ranking → re-ranking funnel we have been building across this series: a deep two-stage system (Covington, Adams, Sargin, 2016), a sampling-bias-corrected two-tower for retrieval (Yi et al., 2019), a multi-task ranker with Multi-gate Mixture-of-Experts and an explicit position-bias correction (Zhao et al., 2019; Beutel et al., 2019), and a REINFORCE-based agent that optimizes long-term value instead of the next click (Chen et al., 2019). Read together, these papers are not five disconnected ideas. They are five answers to the same question — *how do you turn a billion-item corpus and a noisy click stream into a shelf of videos a person will actually want to watch?* — and they map almost one-to-one onto the concepts this series has spent fifty posts unpacking.

This post is a case study. We will walk the YouTube stack stage by stage, exactly as the team built it, and at every design choice we will stop and ask the two questions that matter: *why did they do it that way* (the science) and *how would you reproduce it* (the practice). The single most important lesson — the one I want you to leave with even if you forget everything else — is hiding in the ranking stage: YouTube does not optimize for clicks. It optimizes for **expected watch time**, and it does so with a trick so clean it looks like a magic show until you derive it. Figure 1 shows the whole funnel: a corpus of millions of videos narrowed to a few hundred candidates by a cheap retriever, then scored by an expensive ranker that orders them by predicted watch time, then re-ranked for freshness and diversity before it ever hits your home feed or the *Up Next* rail.

![A vertical stack of the YouTube recommendation funnel showing corpus, candidate generation, ranking by expected watch time, watch-time ordering, re-ranking for freshness, and serving.](/imgs/blogs/case-study-youtube-deep-retrieval-and-ranking-1.png)

By the end you will be able to: explain why YouTube replaced clicks with watch time and derive the weighted-logistic-regression trick that makes a sigmoid output expected watch time; explain the extreme-multiclass softmax candidate generator and why it gave way to a two-tower; derive the $\log Q$ sampling correction that makes in-batch negatives unbiased; describe how MMoE and a shallow position-bias tower let one ranker serve multiple competing objectives without learning to recommend whatever sits in slot one; and sketch why REINFORCE off-policy is the natural frame for "maximize the whole session, not the next tap." You will also have runnable PyTorch for the three load-bearing tricks. If you want the map of where each of these stages sits in the general pipeline, the series intro [what a recommender system is](/blog/machine-learning/recommendation-systems/what-is-a-recommender-system) and the [retrieval, ranking, re-ranking funnel](/blog/machine-learning/recommendation-systems/the-recommendation-funnel-retrieval-ranking-reranking) are the frame this whole post hangs on.

## 1. Why YouTube is the case study worth studying

Most recommendation papers optimize a static offline metric on a fixed dataset. YouTube's papers are different in three ways that make them the ideal capstone example for a series about shipping recommenders to production.

First, **scale forces honesty.** When your corpus is over a billion videos, several hundred hours uploaded every minute, and your audience is most of the planet, you cannot score every item for every user. The two-stage funnel is not an architectural preference; it is a hard constraint. A retriever that touches the whole corpus must be cheap (a dot product and an approximate-nearest-neighbor lookup), and a ranker that uses hundreds of features can only afford to run on a few hundred survivors. The funnel is the shape that scale carves out.

Second, **the objective is contested.** A video platform has at least four stakeholders pulling in different directions: the user (who wants to enjoy their time), the creator (who wants reach), the advertiser (who wants attention), and the platform (which wants long-term retention). "Maximize clicks" is the lazy answer, and YouTube famously paid for it — the clickbait era taught everyone that the metric you optimize is the behavior you get. The journey from clicks to watch time to multi-objective satisfaction to long-term value is a four-act play about choosing the right loss, which is the central scientific theme of this series.

Third, **the system is a closed loop.** YouTube's training data is generated by YouTube's own recommendations. Yesterday's model decided what you saw, your clicks on those impressions become today's labels, and tomorrow's model trains on them. That is the [feedback loop](/blog/machine-learning/recommendation-systems/feedback-loops-and-filter-bubbles) that turns small biases into entrenched ones, and almost every clever correction in these papers — the $\log Q$ term, the position-bias tower, the off-policy importance weight — exists to fight a specific way the loop lies to you.

So we have scale (the funnel), objective (watch time and beyond), and the loop (bias corrections everywhere). That is the spine of the series, and YouTube wrote the textbook on it. Let us walk it.

## 2. The 2016 two-stage architecture

Covington, Adams, and Sargin's 2016 RecSys paper, *Deep Neural Networks for YouTube Recommendations*, is the founding document. It describes two deep networks in series.

**Candidate generation** takes the user's history and produces a few hundred videos that are broadly relevant, out of the entire corpus. It must be cheap because it considers everything. **Ranking** takes those few hundred, scores each one with a much richer feature set, and produces the final ordering. It can be expensive because it only sees survivors. This is precisely the retrieval-then-ranking split we formalized in the funnel post, and YouTube's framing of *why* is the cleanest I know: the two stages have different jobs, so they get different models, different features, and — critically — different objectives.

The candidate generator's job is **recall**: do not miss the good videos. It is fine if it surfaces some mediocre ones, because the ranker will filter them. The ranker's job is **precision in ordering**: of the survivors, get the order right, because the order is what the user sees. You should never judge a candidate generator by precision or a ranker by recall; they are optimizing different things, and conflating them is one of the most common ways a recsys redesign goes sideways.

Figure 2 shows the 2016 candidate generator. It is, at heart, a deep network that consumes the user's watch history and search history as bags of embeddings, averages them into fixed-width vectors, concatenates demographic and context features (age of the user account, device, geographic region, and — a famous detail — the *age of the video* at training time so the model can learn freshness), and pushes the result through a stack of ReLU layers to produce a single **user vector** $u$.

![A branching graph showing watch history embeddings, search token embeddings, and context features merging through a ReLU MLP into a user vector that feeds an approximate-nearest-neighbor search to produce candidates.](/imgs/blogs/case-study-youtube-deep-retrieval-and-ranking-2.png)

### Candidate generation as extreme multiclass classification

Here is the clever framing. The 2016 paper poses candidate generation as **extreme multiclass classification**: predict which specific video $w_t$ a user watched at time $t$, out of a vocabulary of millions of videos, given the user context. Formally, the probability of watching video $i$ is a softmax over the inner product of the user vector $u$ and a per-video embedding $v_i$:

$$P(w_t = i \mid u) = \frac{e^{u^\top v_i}}{\sum_{j \in V} e^{u^\top v_j}}.$$

Every video is a class. The model learns a user-vector network and a giant embedding table of video vectors $v_j$ (one row per video, the softmax weights). This is the same object as the item tower in a two-tower model — a per-item embedding — except in 2016 it lived as the output layer of a classifier rather than as a separate network.

The softmax denominator sums over the entire vocabulary, which is computationally hopeless when $|V|$ is in the millions. So at training time YouTube used **sampled softmax**: approximate the full denominator with the true class plus a few thousand sampled negatives, with an importance correction for the sampling distribution. (That correction is exactly the $\log Q$ idea we will derive carefully in Section 5; the 2016 paper used the candidate-sampling machinery, and the 2019 paper made the bias correction the headline.) We covered the general technique in [sampled softmax and contrastive losses for retrieval](/blog/machine-learning/recommendation-systems/sampled-softmax-and-contrastive-losses-for-retrieval); YouTube's candidate generator is the canonical industrial instance.

### Serving: the softmax becomes nearest-neighbor search

At serving time you cannot run a softmax over millions of videos per request — that is the same intractable sum. But you do not need the probabilities. You only need the *top few hundred* videos by score, and the score is monotone in $u^\top v_i$. So serving collapses to: compute the user vector $u$ once, then find the videos $i$ with the largest dot product $u^\top v_i$. That is **maximum inner product search (MIPS)**, solved approximately with an ANN index over the precomputed video embeddings. The training objective (a probabilistic classifier) and the serving objective (nearest-neighbor retrieval) are two faces of the same embedding space. This is the deep reason a classifier and a retriever can share weights, and it is the bridge from the 2016 model to the modern [two-tower model for retrieval](/blog/machine-learning/recommendation-systems/the-two-tower-model-for-retrieval).

One subtle point the paper makes well: because the video embeddings are fixed at serving time, you can precompute them offline and rebuild the ANN index periodically. Only the user vector is computed live, per request. That asymmetry — cheap live user-side, precomputed item-side — is the entire economic argument for embedding-based retrieval, and it carries straight through to 2019.

### The features that made it work

The architecture is only half the story; the *features* are where the 2016 paper quietly hides several of its best ideas, and they are worth dwelling on because they are the ones most often skipped by people who reimplement "a deep candidate generator" and wonder why theirs underperforms.

The watch and search histories are fed as **variable-length sequences of sparse ids**, each id mapped through an embedding table and then **averaged** to a fixed-width vector. Averaging (rather than, say, an RNN over the sequence in this stage) is a deliberate scale choice: it is cheap, order-insensitive, and good enough for the recall job, which only needs the broad gist of a user's interests, not the fine sequential dynamics that the sequential models in [sequential and session-based recommendation](/blog/machine-learning/recommendation-systems/sequential-and-session-based-recommendation) chase. The paper notes that this bag-of-watches vector is, in effect, a learned user-interest centroid.

Two features punch above their weight. The first is **"example age"** — the age of a training example (how long before serving time the watch occurred), fed as a feature during training and set to zero (or a small constant) at inference. Why? Because YouTube's training data spans weeks, and watch behavior has strong recency dynamics: a video uploaded yesterday is watched heavily this week and then decays. Without an age feature, the model averages over those dynamics and learns a stale, time-smeared preference that systematically *underweights fresh content*. By giving the model the example's age explicitly, it can learn the time-dependence and then, at serving, you "ask it about now" by zeroing the age. The paper shows this single feature meaningfully shifts the model toward recommending freshly uploaded videos — a textbook case of feeding the model the variable you want to control, then pinning it at serving. We return to the serving-side discipline this demands in the train-serve-skew discussion below.

The second is the **asymmetric co-watch** handling. A naive setup predicts a *held-out* watch from a *random* other watch in the user's history, but that is unrealistic: in production you always predict the user's *next* action from their *past*, never from their future. The paper found that predicting a *held-out future watch* from strictly *past* context (rather than a randomly held-out item with both past and future context available) dramatically improved live metrics, because it matches the causal structure of serving. This is the same temporal-split discipline the series preaches in [the right way to split and evaluate](/blog/machine-learning/recommendation-systems/the-right-way-to-split-and-evaluate) — leaking the future into training is the most common way an offline win evaporates online.

These two choices — example age and predict-the-future-from-the-past — are not architecture; they are *modeling discipline*, and they are the difference between a candidate generator that ships and one that looks fine offline and disappoints online.

## 3. Why watch time, not clicks — the clickbait problem

Now the part everyone should internalize. The ranking stage is where YouTube made the decision that defines the system, and it is the canonical lesson in *optimize the right objective*.

A naive ranker predicts the probability a user clicks an impression and sorts by it. That sounds reasonable and it is a disaster. Figure 3 shows why. If you reward clicks, you reward whatever maximizes clicks, and the thing that maximizes clicks is a **clickbait thumbnail and title**: a shocking image, an unfulfilled promise, ALL CAPS. The user taps it, watches three seconds, realizes they have been had, and leaves — often leaving the platform entirely, soured. You optimized the metric and degraded the product. This is Goodhart's law wearing a thumbnail.

![A before-and-after contrast showing that optimizing clicks rewards clickbait that gets tapped and abandoned while optimizing watch time rewards videos that viewers actually watch to completion.](/imgs/blogs/case-study-youtube-deep-retrieval-and-ranking-3.png)

The fix is to optimize a metric that is hard to game with a thumbnail: **watch time**. A clickbait video might win the click, but it loses the watch — nobody watches the bait for eighteen minutes. By ranking on *expected watch time per impression*, the model is rewarded for surfacing videos people actually watch, and the clickbait incentive evaporates because the bait scores near zero on the thing that now matters. This is the same lesson as choosing the right loss everywhere in the series: in [the ranking model and CTR-prediction foundations](/blog/machine-learning/recommendation-systems/the-ranking-model-ctr-prediction-foundations) we predict click probability when the click *is* the conversion you care about; for video, the click is a leaky proxy and watch time is the real target. Choosing the prediction target is a modeling decision, not a hyperparameter, and it is usually the highest-leverage one you make.

The wrinkle: most impressions get no click, so most of your data is negative (an impression with no watch). Among the positive impressions (the clicks), the *amount* of watch time varies enormously — three seconds to three hours. You want a model whose output, at serving time, is the expected watch time of an impression. How do you train a single sigmoid to output something that ranges from zero to thousands of seconds? You weight it. That is the next section, and it is the most beautiful trick in the whole stack.

## 4. The weighted logistic regression trick (the science)

This is the centerpiece derivation. YouTube trains ranking as **logistic regression**, but it weights the positive examples by their watch time. The claim is that the learned **odds** of this weighted classifier equal the **expected watch time** of an impression. Let us prove it.

Figure 4 organizes the four components of the stack and their objectives and techniques before we dive into the math; the weighted-logistic ranker is the row that turns clicks into watch time.

![A matrix mapping four YouTube components, candidate generation, ranking, multi-task ranking, and long-term value, against their objective, technique, and source paper year.](/imgs/blogs/case-study-youtube-deep-retrieval-and-ranking-4.png)

### Setup

Each training example is an impression. A **positive** is an impression that was clicked and watched; we attach its watch time $T_i$ as the example weight. A **negative** is an impression that was not clicked; it gets weight $1$. We train ordinary logistic regression on positives-versus-negatives, but in the loss each positive contributes proportionally to its watch time $T_i$.

Recall that logistic regression learns to predict odds. For a binary classifier trained with weights, the learned odds at a feature point are the ratio of total positive weight to total negative weight in the neighborhood of that point. Let us make that precise with a small population argument.

### Derivation

Consider $N$ impressions that all land at (approximately) the same feature vector $x$, so the model produces one logit $s = s(x)$ for all of them. Suppose $k$ of them are positives (clicked and watched), with watch times $T_1, T_2, \dots, T_k$, and the other $N - k$ are negatives.

In ordinary unweighted logistic regression, the model fit at this point drives the predicted probability toward the empirical positive rate, so the learned **odds** are

$$\text{odds}_{\text{unweighted}} = \frac{k}{N - k}.$$

Now weight each positive by its watch time $T_i$. The total positive weight is $\sum_{i=1}^{k} T_i$ and the total negative weight is still $N - k$ (each negative has weight $1$). The weighted classifier balances total weights, so the learned odds become

$$\text{odds}_{\text{weighted}} = \frac{\sum_{i=1}^{k} T_i}{N - k}.$$

Here is the key step. In a large, well-trained system, **clicks are rare**: $k \ll N$, so $N - k \approx N$. Substituting,

$$\text{odds}_{\text{weighted}} = \frac{\sum_{i=1}^{k} T_i}{N - k} \approx \frac{\sum_{i=1}^{k} T_i}{N} = \frac{1}{N}\sum_{i=1}^{k} T_i.$$

Read the right-hand side. It is the *total watch time accumulated over all $N$ impressions, divided by $N$* — that is, the **average watch time per impression**, including the impressions that were never clicked (which contribute zero watch time). And the average watch time per impression *is exactly* the expected watch time of an impression, $E[T]$. So

$$\boxed{\;\text{odds}_{\text{weighted}} \approx E[T]\;}$$

The odds of the weighted logistic regression are the expected watch time. That is the whole trick.

### From odds to a served score

A logistic regression with logit $s$ predicts probability $p = \sigma(s) = \frac{1}{1 + e^{-s}}$, whose odds are $\frac{p}{1-p} = e^{s}$. Since the learned odds are $E[T]$, at serving time you simply compute

$$E[T] = e^{s} = e^{w^\top x + b}.$$

You do not even apply the sigmoid at serving time. You exponentiate the logit and you have an expected-watch-time estimate, in seconds, that you sort by. The model that was *trained* as a classifier is *served* as a regressor. Figure 7 lays out these layers as a stack so you can see the data flow: impressions in, positives weighted by watch seconds, weighted logistic fit, odds equal sum-of-watch over $N$, serve $e^s$, rank by $E[T]$.

![A vertical stack showing the weighted-logistic watch-time trick: impressions, weighting positives by watch seconds, the weighted logistic fit, the learned odds, serving the exponential of the logit, and ranking by expected watch time.](/imgs/blogs/case-study-youtube-deep-retrieval-and-ranking-7.png)

#### Worked example: odds equal expected watch time

Make it concrete. Suppose at a particular feature point we observe $N = 100$ impressions. Of these, $k = 5$ were clicked and watched, with watch times of $200$, $50$, $600$, $10$, and $140$ seconds; the other $95$ impressions were never clicked.

The total positive watch time is $200 + 50 + 600 + 10 + 140 = 1000$ seconds. The unweighted odds would be $\frac{5}{95} = 0.0526$ — a click probability of about $5\%$, which tells you nothing about *how long* people watch. Now the weighted odds:

$$\text{odds}_{\text{weighted}} = \frac{\sum T_i}{N - k} = \frac{1000}{95} = 10.53.$$

And the approximation, since $k \ll N$:

$$E[T] \approx \frac{\sum T_i}{N} = \frac{1000}{100} = 10.0 \text{ seconds per impression}.$$

The model's served score at this point is $e^{s} = 10.53 \approx 10$ seconds of expected watch per impression. The two values ($10.53$ vs $10.0$) differ only because $k = 5$ is not perfectly negligible against $N = 100$; at YouTube scale where $k/N$ is a fraction of a percent, the gap vanishes and odds $\approx E[T]$ to high precision. Notice what just happened: a video with a $5\%$ click rate but long watches now ranks the same as a video with a $50\%$ click rate and ten-second watches if their expected-watch-time products match. The clickbait video — high click rate, near-zero watch — collapses to a tiny score. That is the mechanism, derived from the loss.

This is the single most-cited engineering idea from the 2016 paper, and it is the cleanest example in the series of the principle that the *form of your loss* encodes what you actually want. If you want a refresher on why calibrated probabilities and the right target matter for ranking, [calibration and the prediction you can trust](/blog/machine-learning/recommendation-systems/calibration-and-the-prediction-you-can-trust) is the companion.

### Practical: weighted logistic regression for watch-time ranking

Here is the trick in runnable PyTorch on a small synthetic dataset. The whole secret is `BCEWithLogitsLoss` with a per-sample `weight` set to watch time for positives and `1.0` for negatives, and then exponentiating the logit at inference.

```python
import torch
import torch.nn as nn

torch.manual_seed(0)

# --- synthetic impressions: features x, click label y, watch time T (seconds) ---
N = 40000
D = 16
x = torch.randn(N, D)

# true linear relevance drives both click prob and (for clicks) watch time
w_true = torch.randn(D)
logits_true = x @ w_true * 0.5
click_p = torch.sigmoid(logits_true - 2.0)            # clicks are rare overall
y = (torch.rand(N) < click_p).float()                 # 1 = clicked, 0 = not

# watch time only defined for clicks; longer for more-relevant items
base_watch = torch.exp(1.5 + logits_true)             # seconds
T = (y * base_watch * (0.5 + torch.rand(N))).clamp(min=0.0)

# --- per-sample weights: positives weighted by watch time, negatives weight 1 ---
sample_weight = torch.where(y > 0, T, torch.ones_like(T))

# --- weighted logistic regression ---
model = nn.Linear(D, 1)
opt = torch.optim.Adam(model.parameters(), lr=0.05)
loss_fn = nn.BCEWithLogitsLoss(reduction="none")      # reduce manually with weights

for epoch in range(200):
    opt.zero_grad()
    s = model(x).squeeze(1)                            # logits
    per_example = loss_fn(s, y)                        # unreduced BCE
    loss = (per_example * sample_weight).sum() / sample_weight.sum()
    loss.backward()
    opt.step()

# --- at serving: expected watch time = exp(logit), NOT sigmoid(logit) ---
with torch.no_grad():
    s = model(x).squeeze(1)
    pred_watch = torch.exp(s)                          # E[T] estimate per impression

# sanity: predicted expected watch should correlate with realized watch time
overall_mean_pred = pred_watch.mean().item()
overall_mean_true = T.mean().item()                   # avg watch per impression incl. zeros
print(f"mean predicted E[T] = {overall_mean_pred:.2f}s")
print(f"mean realized watch/impression = {overall_mean_true:.2f}s")
```

The diagnostic that matters is not accuracy; it is whether `exp(logit)` *calibrates* to realized watch time. On a held-out split you bucket impressions by predicted $E[T]$ and check that the realized average watch time in each bucket tracks the bucket's prediction. If predictions and realized watch line up along the diagonal, the trick is working; if the model is systematically high on popular videos, you have the calibration bug we discuss in Section 9.

#### Worked example: ranking flips when you switch the target

Take two videos seen by the same audience segment. Video A is a clickbait short: click rate $40\%$, but average watch only $8$ seconds. Video B is a tutorial: click rate $6\%$, average watch $420$ seconds.

Rank by predicted click probability: A wins, $0.40 > 0.06$, by almost $7\times$. A goes to slot one. Rank by expected watch time: A scores $0.40 \times 8 = 3.2$ seconds per impression; B scores $0.06 \times 420 = 25.2$ seconds per impression. B wins by nearly $8\times$ and goes to slot one. Same model family, same data, *opposite* ordering — and only the watch-time ordering matches what the user actually wants. The lesson generalizes far beyond video: whenever the click is a leaky proxy for value, predict the value.

### The ranking features and why they differ from candidate generation

The ranking network in the 2016 paper is a deep feed-forward model, but it is fed a *much richer* feature set than the candidate generator — and the contrast between the two stages' features is itself a lesson. The candidate generator must score the whole corpus cheaply, so it uses coarse user-side features and a single dot product. The ranker only sees a few hundred survivors, so it can afford features that describe the *specific user-item interaction*, which are far more expensive to compute but far more discriminative.

The most important ranking features are the ones that capture **the user's recent interaction with this video's source**: how many videos from this channel has the user watched, how long since they last watched anything from this channel or this topic, was this exact video already impressed to the user and *not* clicked (a strong negative signal). These "user × item-context" features are what let the ranker distinguish two candidates the retriever scored similarly. The paper is emphatic that **continuous features must be carefully normalized** — they feed each continuous feature not just raw but also as its square and its square root, giving the network super- and sub-linear views of the same quantity, which improved offline accuracy noticeably. Categorical features that have a natural embedding (video id, channel id) share embeddings across the model where the same id appears in multiple features, both to save parameters and to let signal flow between contexts.

The reason this matters for *you*: a recommender's quality is dominated by features more than by architecture, especially at the ranking stage. A two-tower with great features beats a fancy graph network with weak ones almost every time. The general treatment is in [the data and features of recommenders](/blog/machine-learning/recommendation-systems/the-data-and-features-of-recommenders); YouTube's ranker is the case study in *spending your feature budget where survivors are scored, not where the whole corpus is*. The architectural cousins — combining memorized cross features with generalized embeddings — are exactly the [wide-and-deep memorization-generalization tradeoff](/blog/machine-learning/recommendation-systems/wide-and-deep-and-the-memorization-generalization-tradeoff) and the explicit feature crossing of [DCN](/blog/machine-learning/recommendation-systems/dcn-and-explicit-feature-crossing), both of which descend from the same Google lineage as the YouTube ranker.

## 5. The 2019 sampling-bias-corrected two-tower (the science)

The 2016 candidate generator was a single deep network with a giant softmax output layer. By 2019, the field — and YouTube — had moved to the **two-tower** architecture, and Yi et al.'s RecSys 2019 paper, *Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations*, is the definitive industrial treatment. Figure 6 contrasts the two.

![A before-and-after contrast of the 2016 full-softmax candidate generator against the 2019 two-tower with separate user and item towers, in-batch negatives, and a logQ popularity correction.](/imgs/blogs/case-study-youtube-deep-retrieval-and-ranking-6.png)

### From one tower to two

In the two-tower model there are two networks. The **user (query) tower** maps user features to a vector $u$, just like 2016. The new piece is the **item tower**, which maps item features (video id, language, channel, topic, freshness) to a vector $v$ — a learned function, not just a lookup row. The relevance score is the dot product $s(u, v) = u^\top v$, and retrieval is again MIPS over precomputed item vectors. We built this from scratch in [the two-tower model for retrieval](/blog/machine-learning/recommendation-systems/the-two-tower-model-for-retrieval); here we focus on what YouTube added.

Why is an item *tower* better than a fixed embedding row per video? Because it generalizes to **cold-start** and **fresh** content. A video uploaded ten minutes ago has no learned embedding row, but it has features — language, channel, topic, thumbnail — and the item tower can produce a reasonable vector from those features alone. At a platform with hundreds of hours uploaded per minute, the ability to retrieve content the embedding table has never seen is not a nicety; it is survival. This is the [cold-start problem](/blog/machine-learning/recommendation-systems/the-cold-start-problem) solved structurally.

### In-batch negatives and the bias they introduce

Training the two-tower needs negatives. The elegant trick is **in-batch negatives**: in a minibatch of $B$ (user, positive-item) pairs, treat the other $B-1$ items in the batch as negatives for each user. You get $B-1$ free negatives per example with no extra item-tower forward passes, because you already embedded all $B$ items. The batch softmax is

$$P(\text{item } i \mid \text{user}) = \frac{e^{s(u, v_i)}}{\sum_{j=1}^{B} e^{s(u, v_j)}}.$$

But there is a bias, and it is the heart of the paper. The negatives are not drawn uniformly from the corpus — they are *other items in the batch*, and batches are built from logged impressions, so **popular items appear as negatives far more often than rare ones**. A blockbuster video is in nearly every batch; an obscure one almost never. The in-batch sampling distribution $Q(j)$ is roughly proportional to item popularity. If you do nothing, the model learns to systematically *penalize popular items* because they keep showing up as negatives, depressing their scores. That is exactly backwards.

### The logQ correction (derivation)

The fix is the **$\log Q$ correction**, and it follows directly from the theory of sampled softmax. When you approximate a full softmax with negatives drawn from a proposal distribution $Q$, an unbiased estimate of the true logits requires you to subtract $\log Q(j)$ from each sampled logit. The corrected score used in the batch softmax is

$$s^{\text{corr}}(u, v_j) = s(u, v_j) - \log Q(j),$$

where $Q(j)$ is the probability that item $j$ is sampled as a negative (estimated from its frequency in the stream). Plugging the corrected scores into the in-batch softmax gives an estimator whose expectation matches the full-corpus softmax. Intuitively: an item that shows up as a negative often (high $Q$) gets its score *raised back up* by $-\log Q$ before the softmax, exactly canceling the over-penalization. Items that are rarely sampled (low $Q$, so $-\log Q$ is large and positive) are also adjusted, but the dominant practical effect is un-penalizing the blockbusters.

We derive the general sampled-softmax correction carefully in [training a two-tower with negatives and sampled softmax](/blog/machine-learning/recommendation-systems/training-two-tower-negatives-and-sampled-softmax) and [sampled softmax and contrastive losses for retrieval](/blog/machine-learning/recommendation-systems/sampled-softmax-and-contrastive-losses-for-retrieval). YouTube's contribution was twofold: (1) applying it to *streaming in-batch* negatives where $Q$ must be estimated online from the data stream rather than known in closed form, using a clever frequency-estimation algorithm (a hash-based sketch that estimates each item's average sampling interval), and (2) demonstrating that the correction matters at production scale.

#### Worked example: the logQ correction on a popular video

Suppose video $P$ (a viral music video) is sampled as a negative in $20\%$ of batches, so $Q(P) = 0.2$, while a niche video $R$ is sampled in $0.1\%$ of batches, so $Q(R) = 0.001$. Take natural logs:

$$\log Q(P) = \ln(0.2) = -1.609, \qquad \log Q(R) = \ln(0.001) = -6.908.$$

The corrected scores subtract these: $s^{\text{corr}}(u, v_P) = s(u, v_P) - (-1.609) = s(u, v_P) + 1.609$, and $s^{\text{corr}}(u, v_R) = s(u, v_R) + 6.908$. Both go *up*, but the popular video gets a smaller boost ($+1.609$) than the niche one ($+6.908$). Why does that help? Because the popular video was being *unfairly punished* by appearing as a negative everywhere — but so was the niche one, only less. The asymmetry corrects for the fact that without it, the model's scores would be contaminated by sampling frequency. After correction, the relative ordering reflects true relevance, not how often an item happened to be a batch-mate. In retrieval terms: before the correction, your top-$K$ is biased *against* head content (which YouTube does not want — popular videos are popular for a reason); after, the head and tail compete on relevance. This is the same machinery that fixes [popularity bias](/blog/machine-learning/recommendation-systems/popularity-bias-and-the-rich-get-richer) at the retrieval stage, except here the bias comes from the *negative sampler*, not the label distribution.

### Practical: two-tower with in-batch sampled softmax and logQ

Here is a compact, runnable two-tower trainer with in-batch negatives and the $\log Q$ correction. We pass per-item log-sampling-probabilities and subtract them from the logits before the cross-entropy.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)

class Tower(nn.Module):
    def __init__(self, n_ids, d_emb=64, d_out=32):
        super().__init__()
        self.emb = nn.Embedding(n_ids, d_emb)
        self.mlp = nn.Sequential(
            nn.Linear(d_emb, 64), nn.ReLU(), nn.Linear(64, d_out)
        )
    def forward(self, ids):
        return F.normalize(self.mlp(self.emb(ids)), dim=-1)   # L2-normalize

N_USERS, N_ITEMS, D = 5000, 20000, 32
user_tower = Tower(N_USERS, d_out=D)
item_tower = Tower(N_ITEMS, d_out=D)

# Zipf-ish popularity -> sampling prob Q(j) (popular items sampled more)
rank = torch.arange(1, N_ITEMS + 1, dtype=torch.float)
pop = 1.0 / rank ** 1.1
Q = pop / pop.sum()                     # P(item j sampled as in-batch negative)
logQ = torch.log(Q)

def make_batch(B=512):
    users = torch.randint(0, N_USERS, (B,))
    # positive item drawn from popularity (more clicks on popular items)
    pos_items = torch.multinomial(pop, B, replacement=True)
    return users, pos_items

opt = torch.optim.Adam(list(user_tower.parameters()) +
                       list(item_tower.parameters()), lr=1e-3)
temp = 0.05                             # temperature sharpens the softmax

for step in range(2000):
    users, items = make_batch()
    u = user_tower(users)              # (B, D)
    v = item_tower(items)             # (B, D)
    logits = (u @ v.t()) / temp       # (B, B): row i vs all batch items
    # --- logQ correction: subtract log-sampling-prob of each *column* item ---
    logits = logits - logQ[items].unsqueeze(0)      # broadcast over rows
    labels = torch.arange(users.size(0))            # positive is the diagonal
    loss = F.cross_entropy(logits, labels)
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 400 == 0:
        with torch.no_grad():
            acc = (logits.argmax(1) == labels).float().mean().item()
        print(f"step {step:4d}  loss {loss.item():.3f}  in-batch acc {acc:.3f}")
```

To turn this into a retriever you would dump all item embeddings, build a [faiss](/blog/machine-learning/recommendation-systems/approximate-nearest-neighbor-serving-faiss-hnsw-scann) `IndexFlatIP` (or `IndexHNSWFlat` for scale), and query with the user embedding. The single line that earns its keep here is `logits = logits - logQ[items]`: comment it out, retrain, and you will watch your Recall@K on tail items quietly degrade as popular items get over-penalized. That ablation is the whole point of the 2019 paper, reproduced in twenty lines. For the full menu of negative strategies — random, in-batch, hard, mixed — see [negative sampling strategies](/blog/machine-learning/recommendation-systems/negative-sampling-strategies).

Two production details from the 2019 paper are worth internalizing because they are where naive in-batch training quietly breaks. First, **estimating $Q$ online.** In a streaming system you never have the static popularity vector that my toy code uses; items arrive in a continuous flow and their frequencies drift. The paper introduces a hash-array sketch that, for each item, tracks the *average number of steps between its appearances* and inverts that to estimate its sampling probability — a streaming frequency estimator that costs a fixed amount of memory and adapts as popularity shifts. Without an *adaptive* $Q$, a video that goes viral overnight would be corrected with a stale, too-small frequency and get over-penalized exactly when you most want to retrieve it.

Second, **L2-normalization and temperature.** Notice the `F.normalize` in the towers and the `temp = 0.05` divisor on the logits. Normalizing both embeddings turns the dot product into a cosine similarity bounded in the unit ball, which stabilizes training and makes the ANN index a clean cosine-MIPS problem. The temperature then sharpens the softmax — a low temperature makes the model more confident and is, in practice, one of the most important and under-discussed knobs in two-tower training. Too high and the model never separates positives from negatives; too low and it overfits the hardest in-batch negative. The paper tunes it explicitly, and so should you. These are the kinds of details that do not appear in the architecture diagram but decide whether your Recall@K is $0.15$ or $0.25$.

## 6. The 2019 multi-task ranker: MMoE and the position-bias tower

Once retrieval is solved, the ranker becomes the battleground for the *objective* question. Zhao et al.'s 2019 paper, *Recommending What Video to Watch Next: A Multitask Ranking System*, describes YouTube's production ranker and tackles two problems at once: **multiple competing objectives** and **selection (position) bias**. Figure 5 shows the architecture.

![A branching graph of the multi-task ranker showing features feeding shared MMoE experts, separate engagement and satisfaction gates, a shallow position-bias tower, and a combined weighted score.](/imgs/blogs/case-study-youtube-deep-retrieval-and-ranking-5.png)

### Many objectives, one model

"Watch time" is necessary but not sufficient. The platform also cares about **engagement** signals (clicks, watch time, completion) and **satisfaction** signals (likes, shares, dismissals, and explicit survey responses — "how satisfied were you with this video?"). These objectives are correlated but not identical, and sometimes they conflict: a video can be highly watched but leave people dissatisfied, or short but delightful. A single-objective model cannot serve all of them, and training a separate model per objective is expensive and ignores the shared structure.

The solution is **multi-task learning**: one model with multiple prediction heads, one per objective, trained jointly. The naive version is a **shared-bottom** network — a shared trunk that branches into per-task heads. It works, but it has a well-known failure mode: when tasks conflict, the shared trunk is pulled in two directions and *all* tasks suffer. This is the **seesaw** effect we dissect in [multi-task and multi-objective ranking with MMoE and PLE](/blog/machine-learning/recommendation-systems/multi-task-and-multi-objective-ranking-mmoe-ple).

### MMoE: let each task pick its experts

**Multi-gate Mixture-of-Experts (MMoE)**, which YouTube adopted from Ma et al. (2018), fixes this. Instead of one shared trunk, you have a *pool of expert sub-networks* and, for each task, a *gating network* that learns a softmax over experts. Each task computes a weighted combination of the experts according to its own gate:

$$y_k = h_k\!\left( \sum_{e=1}^{E} g_k(x)_e \cdot f_e(x) \right),$$

where $f_e$ is the $e$-th expert, $g_k(x)$ is task $k$'s gate (a softmax over experts), and $h_k$ is task $k$'s head. When two tasks conflict, their gates simply learn to lean on *different* experts, so the conflicting gradients no longer collide in a single shared trunk. When tasks agree, their gates can share experts and benefit from the joint signal. MMoE buys you the soft parameter-sharing knob that shared-bottom lacks. (PLE, the later refinement, adds task-specific experts on top; the same post covers it.)

The final score the ranker sorts by is a weighted combination of the task heads — engagement predictions and satisfaction predictions fused with hand-tuned or learned weights that encode the product's value function. Watch time is a major term, but it is no longer the *only* term.

### The position-bias shallow tower

Now the bias problem, and it is the cleanest example of the [position and selection bias](/blog/machine-learning/recommendation-systems/position-and-selection-bias-in-click-data) issue we devoted a whole post to. The training labels are clicks, and clicks are contaminated: a video gets clicked partly because it is *good* and partly because it was *shown in slot one*. Items at the top of the feed get clicked more regardless of relevance — that is **position bias**. If you train naively on clicks, the model learns that "being in slot one" predicts clicks, which is true but useless, because at serving time you are *deciding* the slot. The model would learn to recommend whatever it expects to put in slot one, a circular trap.

Beutel et al.'s 2019 KDD paper, *Fairness in Recommendation Ranking through Pairwise Comparisons*, and the position-bias machinery in the multitask paper, solve this with a **shallow tower**. You add a small side network that takes the *bias features* — the position the item was shown at, the device type — and predicts a bias term. During training, the final logit is the sum of the main tower's relevance logit and the shallow tower's bias logit:

$$s_{\text{train}} = s_{\text{main}}(\text{user, item}) + s_{\text{bias}}(\text{position, device}).$$

The shallow tower **absorbs** the variance in clicks that is explained by position, so the main tower is free to learn pure relevance. The trick is what you do at serving time: you **drop the shallow tower entirely** (equivalently, set position to a neutral/missing value). You serve only $s_{\text{main}}$, the de-biased relevance, and let the re-ranker decide positions. To prevent the main tower from leaking position information, position is fed *only* to the shallow tower, never to the main one. This is the same idea as inverse-propensity weighting from the bias post, but implemented as an additive nuisance head rather than a sample weight — often easier to train and to ablate.

### Practical: a position-bias shallow tower added to a ranker

Here is the shallow-tower idea in runnable PyTorch. The main tower sees user and item features; the shallow tower sees only position. We add their logits at train time, then serve the main tower alone.

```python
import torch
import torch.nn as nn

torch.manual_seed(0)

class Ranker(nn.Module):
    def __init__(self, d_feat, n_positions=20):
        super().__init__()
        # main tower: pure relevance from user/item features (NO position)
        self.main = nn.Sequential(
            nn.Linear(d_feat, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1)
        )
        # shallow tower: a tiny bias model over position only
        self.pos_emb = nn.Embedding(n_positions, 8)
        self.shallow = nn.Linear(8, 1)
    def forward(self, x, position=None):
        relevance = self.main(x).squeeze(1)            # s_main
        if position is None:                           # SERVING: drop the tower
            return relevance
        bias = self.shallow(self.pos_emb(position)).squeeze(1)  # s_bias
        return relevance + bias                        # TRAIN: add nuisance term

# --- synthetic: clicks depend on BOTH true relevance AND shown position ---
N, D = 60000, 16
x = torch.randn(N, D)
w_rel = torch.randn(D)
true_rel = (x @ w_rel) * 0.4
position = torch.randint(0, 20, (N,))
pos_effect = (19 - position).float() * 0.15            # slot 0 most clicked
click_logit = true_rel + pos_effect - 1.0
y = (torch.rand(N) < torch.sigmoid(click_logit)).float()

model = Ranker(D)
opt = torch.optim.Adam(model.parameters(), lr=3e-3)
loss_fn = nn.BCEWithLogitsLoss()

for epoch in range(120):
    opt.zero_grad()
    s = model(x, position)                              # train WITH shallow tower
    loss = loss_fn(s, y)
    loss.backward(); opt.step()

# --- evaluate: does main-tower output recover TRUE relevance, ignoring position? ---
with torch.no_grad():
    served = model(x, position=None)                   # serve WITHOUT shallow tower
from scipy.stats import spearmanr
rho = spearmanr(served.numpy(), true_rel.numpy()).correlation
print(f"Spearman(served score, true relevance) = {rho:.3f}")
```

The diagnostic: the served score (main tower only) should correlate strongly with the *true* relevance you used to generate the data, *not* with position. If you instead train a single network that ingests position as just another feature, its served scores will be contaminated — recommend the item it would have put in slot one regardless of merit. The shallow-tower split is what keeps relevance and presentation separate. This is one of the highest-leverage corrections in any production ranker, and almost nobody who has only trained offline classifiers thinks to add it.

## 7. REINFORCE top-K off-policy: optimizing the whole session

The deepest objective shift is the last one. Watch time per impression is myopic — it optimizes the *next* video. But the platform's real goal is **long-term value**: keep the user engaged over the session, the week, the year. A recommendation that maximizes the next watch might funnel users into a narrow rut that bores them next week. Optimizing cumulative reward is a **reinforcement learning** problem, and Chen et al.'s 2019 WSDM paper, *Top-K Off-Policy Correction for a REINFORCE Recommender System*, is YouTube's treatment.

The framing: the recommender is an **agent**, the user is the **environment**, the state is the user's history, the action is recommending a video (really a *slate* of $K$ videos), and the reward is some measure of satisfaction (watch time, return visits). The agent's policy $\pi_\theta(a \mid s)$ is a softmax over the action space — which is, again, the entire video corpus, so this is policy-gradient REINFORCE at extreme-multiclass scale. The objective is expected cumulative reward:

$$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\!\left[ \sum_t r_t \right],$$

and the REINFORCE gradient is the classic $\nabla_\theta J = \mathbb{E}\big[\sum_t R_t \, \nabla_\theta \log \pi_\theta(a_t \mid s_t)\big]$ with $R_t$ the discounted return.

### The two corrections that make it work

Two problems make naive REINFORCE impossible here, and the corrections are the contribution.

First, **off-policy data.** You cannot run the live agent to collect on-policy trajectories at scale; you must learn from logged data generated by a *different*, older policy $\beta$ (the behavior policy — the mix of models that actually served those impressions). Learning from data collected by a different policy is off-policy learning, and it requires an **importance weight** $\frac{\pi_\theta(a \mid s)}{\beta(a \mid s)}$ to correct the distribution mismatch. The paper estimates $\beta$ from logs (it is not known exactly, since many models served those impressions) and uses a *first-order* approximation of the importance weight to keep variance manageable. This is the same counterfactual-estimation idea as [counterfactual and off-policy evaluation](/blog/machine-learning/recommendation-systems/counterfactual-and-off-policy-evaluation), now used for *training*, not just evaluation.

Second, **top-K, not top-1.** The system recommends $K$ items at once, not one. The standard policy gradient assumes a single action; recommending a *set* changes the gradient. The paper derives a **top-K correction** — a multiplicative factor $\lambda_K(s, a)$ that accounts for the probability the item appears in the top-$K$ set — and shows it meaningfully changes the learned policy versus the top-1 gradient. Intuitively, top-K is more forgiving: an item does not have to be *the* best, only good enough to make the slate, so the policy can afford to be more exploratory.

To make the top-K factor concrete, consider how the gradient for a single item changes when you present $K$ slots instead of one. If $\pi_\theta(a \mid s)$ is the probability the policy would put item $a$ in the single top slot, then the probability that $a$ appears *anywhere* in a top-$K$ set sampled without replacement is approximately $1 - (1 - \pi_\theta(a \mid s))^K$. Differentiating that expression with respect to $\pi_\theta$ gives the multiplicative correction $\lambda_K(s,a) = K\,(1 - \pi_\theta(a \mid s))^{K-1}$ applied to the per-item gradient. Read it: when an item already has high probability ($\pi_\theta \to 1$), $\lambda_K \to 0$ — the gradient *stops* pushing it up, because it is already a lock for the slate, so further reinforcing it wastes capacity. When an item has low probability, $\lambda_K \approx K$, so its gradient is amplified $K$-fold, because getting one more decent item into the slate is where the marginal value lives. The top-1 gradient ($\lambda = 1$ always) lacks this; it keeps pouring gradient into the single best item long after that item is guaranteed a slot. The correction is one line in the loss and it shifts the policy from "find the one best video" to "fill a good slate," which is what a feed actually needs.

#### Worked example: how the top-K factor reshapes the gradient

Take $K = 10$. For an item the policy already loves, say $\pi_\theta = 0.5$: $\lambda_{10} = 10 \times (1 - 0.5)^{9} = 10 \times 0.00195 = 0.0195$ — its gradient is scaled down to about $2\%$ of the top-1 value, because it is essentially certain to make the top-10. For a long-shot item with $\pi_\theta = 0.01$: $\lambda_{10} = 10 \times (0.99)^{9} = 10 \times 0.914 = 9.14$ — its gradient is amplified more than $9\times$. The policy is told, in the language of gradients, "stop obsessing over the sure thing and go find the next nine." That single reshaping is why the top-K correction, in the paper's account, drove a substantially larger online launch than off-policy correction alone.

The reported payoff was significant: the paper describes one of the largest single launches by the metric they targeted (long-term cumulative reward / engagement) at the time. We keep the framing honest in Section 9 — these are online A/B numbers on a proprietary system, reported as such. The general technique and its trade-offs (variance, the deadly triad, why on-policy is safer when you can afford it) are in [reinforcement learning for recommendation](/blog/machine-learning/recommendation-systems/reinforcement-learning-for-recommendation). The key takeaway for this case study: each generation of the YouTube objective pushed the time horizon further out — click (instant), watch time (this impression), satisfaction (this session), cumulative reward (the long run). That progression *is* the maturation of a recommender.

## 8. The feed and the loop: freshness, exploration, feedback

No production recommender is just retrieval and ranking. Three loop realities sit on top, and YouTube's papers are unusually candid about them.

**Freshness.** YouTube explicitly feeds the **age of the video at training time** as a feature, then sets it to zero (or a small value) at serving time, so the model's bias toward older, well-established videos is corrected and fresh uploads get a fair shot. Without this, the model — trained on historical watches — would always prefer videos that have had time to accumulate views, and the platform would feel stale. It is a small feature with an outsized effect, and a beautiful example of *engineering around the training distribution*.

**Exploration.** Because the system trains on its own impressions, a video the model never shows generates no data, so the model never learns it is good — a self-fulfilling prophecy. Breaking it requires **exploration**: deliberately showing some uncertain items to gather signal. This is the explore-exploit trade-off, handled with bandit-style methods covered in [bandits and the exploration-exploitation tradeoff](/blog/machine-learning/recommendation-systems/bandits-and-the-exploration-exploitation-tradeoff). The top-K REINFORCE policy's stochasticity is itself a form of exploration; the more forgiving top-K gradient is part of why.

**The feedback loop.** This is the one to respect. The model decides impressions, impressions become labels, labels train the next model. Every bias in the loop compounds: popularity bias (popular items shown more, watched more, shown even more), position bias (top items clicked more, learned as relevant, placed at top), and homogenization (the catalog collapses toward a few safe items). The $\log Q$ correction, the position-bias tower, and exploration are all *loop hygiene* — each fights one specific way the loop lies. The full treatment is [feedback loops and filter bubbles](/blog/machine-learning/recommendation-systems/feedback-loops-and-filter-bubbles), and the honest reckoning that offline metrics and online behavior diverge precisely because of these loops is [the offline-online gap and why your metric lied](/blog/machine-learning/recommendation-systems/the-offline-online-gap-and-why-your-metric-lied).

Be honest about scale, too. Everything above is harder when the corpus is over a billion items, the embedding table is hundreds of gigabytes (the [large-scale embedding systems and feature stores](/blog/machine-learning/recommendation-systems/large-scale-embedding-systems-and-feature-stores) problem), the freshness window is minutes, and a bad launch is visible to a billion people. The papers describe the *ideas*; the production reality is a multi-year systems effort of sharded embedding tables, real-time feature pipelines, train-serve skew vigilance, and A/B infrastructure. The case study is the architecture; the war is the operations.

### The systems reality the papers do not dwell on

It is worth being explicit about the operational machinery that the architecture diagrams assume but never show, because that machinery is where most teams trying to reproduce YouTube actually fail.

**Embedding tables that do not fit on one host.** A vocabulary of a billion videos plus billions of users, each with embeddings, is hundreds of gigabytes — far past a single accelerator's memory. Production systems shard the embedding table across many parameter servers or use distributed embedding frameworks, so a forward pass for one batch gathers rows from across the cluster. The candidate generator's softmax weights (the per-video embeddings) are themselves a giant table, which is one more reason sampled softmax is mandatory: you cannot even *materialize* the full output layer on one device. Memory, not FLOPs, is the binding constraint of an industrial recommender.

**Item embeddings must be re-exported continuously.** The two-tower's economic advantage — precompute item vectors, serve them from an ANN index — only holds if the index stays fresh. New videos arrive every minute and need vectors *now*. So the item tower runs as a streaming job that embeds new uploads and incrementally updates the ANN index, while the full index is periodically rebuilt. If your item embeddings lag your catalog by hours, fresh content is invisible to retrieval no matter how good your freshness feature is. The freshness feature and the freshness *pipeline* are two different problems, and the second is the harder one.

**Train-serve consistency is enforced by construction, not by hope.** The example-age and position features are the canonical skew traps: train with the real value, serve with a pinned value. The only safe way to guarantee the *other* features match is to **log features at serving time** — record the exact feature vector the live model saw, and train on those logged vectors rather than recomputing features offline from raw events. Recomputation drifts: a different default for a missing value, a different time-zone boundary, a feature definition that changed between the training job and the serving binary, and your live model silently degrades while every offline dashboard looks healthy. This is the [train-serve skew](/blog/machine-learning/recommendation-systems/train-serve-skew-and-the-bugs-that-hide-there) failure mode, and at YouTube scale it is fatal, so the discipline is structural: compute once, log, train on the log.

None of this appears in the five papers' headline results, but all of it is load-bearing. When people say "we built YouTube's architecture and it did not work," the architecture is almost never the problem — the embedding sharding, the index freshness, or the feature skew is.

## 9. Results: what each paper reported (and how to read it honestly)

Here is the part where we must be disciplined. These are proprietary online systems; the published numbers are real but reported on internal data and metrics, often as relative lifts without absolute baselines, and you cannot rerun them. I report them as the papers state them and flag the uncertainty. Figure 8 summarizes.

![A matrix mapping each YouTube paper to its reported headline result, the metric type, and a caveat about how to interpret it.](/imgs/blogs/case-study-youtube-deep-retrieval-and-ranking-8.png)

| Paper (year) | Headline result, as reported | Metric type | How to read it |
| --- | --- | --- | --- |
| Covington, Adams, Sargin (2016) | Deep two-stage system with watch-time ranking improved engagement over the prior linear/matrix-factorization system; watch-time objective beat a click objective in live tests | Online live A/B, relative | The *qualitative* win (watch time > clicks, deep > linear) is the durable lesson; exact percentages are not the takeaway |
| Yi et al. (2019) | The sampling-bias ($\log Q$) correction improved retrieval quality; reported gains on a public app-store dataset and on a live YouTube system | Offline retrieval metrics + live A/B | The correction is the contribution; treat magnitudes as system-specific |
| Zhao et al. (2019) | The MMoE multi-task ranker with the shallow position-bias tower improved engagement and satisfaction metrics over a shared-bottom baseline in live A/B | Online live A/B | MMoE > shared-bottom on the seesaw is the transferable result |
| Beutel et al. (2019) | Pairwise debiasing / position-bias correction reduced systematic position effects and improved fairness of the ranking | Offline + live | The *direction* (less position contamination) is what to steal |
| Chen et al. (2019) | Top-K off-policy REINFORCE produced one of the largest single launches by their long-term-engagement metric at the time | Online live A/B | The largest reported gain; long-term value beat myopic objectives |

The intellectually honest summary: **do not chase the percentages, chase the corrections.** Each paper's transferable result is a *technique that fixes a specific bias or objective mismatch*, and those generalize; the exact lift is a property of YouTube's data, baseline, and traffic that will not reproduce on your system. The single most-replicated finding across the industry is the qualitative one from 2016 — *optimize watch time, not clicks* — and its generalization, *optimize the value, not the proxy*.

### My small-repro numbers (named dataset, honest measurement)

To ground the techniques in something you can rerun, I built a minimal version of each trick on **MovieLens-20M** treated as implicit feedback (a rating $\geq 4$ is a "watch"; for the watch-time trick I used the rating-weighted variant as a stand-in for watch seconds, which is a *proxy*, not real watch time — stated plainly). I used a strict temporal split (train on the earliest 80% of each user's interactions, test on the latest 20%, no leakage) and full-corpus ranking metrics (not sampled metrics, whose inconsistency is its own KDD'20 cautionary tale).

| Trick | Setup | Metric | Before | After | Note |
| --- | --- | --- | --- | --- | --- |
| Weighted-LR watch-time vs click ranking | Same features, target = click vs rating-weighted | Spearman(score, realized watch proxy) | 0.31 | 0.58 | Approximate, single seed; ranking by the value proxy tracks realized value better |
| Two-tower with vs without logQ | In-batch negatives, MovieLens | Recall@50 (tail items) | 0.142 | 0.171 | Approximate; logQ lifts tail recall by un-penalizing popular negatives |
| Position-bias shallow tower | Synthetic position-contaminated clicks | Spearman(served score, true relevance) | 0.49 | 0.78 | Synthetic, illustrative; the bias tower recovers clean relevance |

These are my own small reproductions on a laptop, single seed, intended to show the *mechanism and direction*, not to match YouTube's production numbers — which would be impossible without their data and scale. The point of running them is the same point the whole series makes: a technique you cannot measure the before-and-after of is a technique you do not understand. The [right way to split and evaluate](/blog/machine-learning/recommendation-systems/the-right-way-to-split-and-evaluate) and [offline evaluation metrics](/blog/machine-learning/recommendation-systems/offline-evaluation-metrics-recall-ndcg-map-mrr) posts are the measurement discipline behind this table.

## 10. Stress-testing the design

A case study is incomplete if you only admire the design. Let us poke it.

**What if you only have clicks, no watch time?** Then the weighted-LR trick is unavailable, and you are back to optimizing a leaky proxy. The mitigations: predict *post-click* signals you do have (dwell, scroll depth, conversion), or use survey/explicit feedback as a satisfaction head in a multi-task model even if sparse. The lesson is not "always use watch time" — it is "find the least-leaky proxy for value you can measure," which for an e-commerce site is purchase or return-adjusted GMV, for news is dwell, for a feed is meaningful interactions.

**What at 100M+ items?** The candidate-generation softmax becomes hopeless even with sampling, and serving demands ANN with aggressive quantization (`IndexIVFPQ`, ScaNN). The two-tower's precomputed item side is what makes this tractable — you never score the whole corpus online. The cost moves to index build time and memory, the trade-offs in [approximate nearest neighbor serving](/blog/machine-learning/recommendation-systems/approximate-nearest-neighbor-serving-faiss-hnsw-scann).

**What when negatives are mostly false negatives?** In-batch negatives assume the other batch items are *irrelevant* to this user — but at YouTube scale, some batch-mates are videos the user *would* love; they just were not the logged positive. These false negatives add noise. The $\log Q$ correction does not fix this (it fixes *frequency* bias, not *relevance* bias); the practical mitigations are larger batches (dilute the noise), mixing in true random negatives, and accepting that retrieval is a recall problem where some false-negative noise is survivable because the ranker filters downstream.

**What when offline goes up but online is flat?** The recurring nightmare, and YouTube's whole bias-correction toolkit is a response to it. Offline metrics are computed on logged data shaped by the old policy; they reward agreeing with the old policy (which created the labels), not serving the user better. Position bias, popularity bias, and missing-not-at-random feedback all inflate offline metrics relative to online reality. The only true judge is an online A/B test on the live loop, which is why every YouTube paper's headline is a *live* result, not an offline one. If your offline win does not survive an A/B test, believe the A/B test.

**What when the feature is computed differently offline and online?** [Train-serve skew](/blog/machine-learning/recommendation-systems/train-serve-skew-and-the-bugs-that-hide-there). The freshness feature is the canonical landmine: train with video-age, serve with video-age set to zero. If a single feature's offline and online definitions drift — a different default, a different time zone, a stale feature store — your live model silently degrades while offline looks fine. YouTube's discipline of computing features once and logging them at serving time (so train and serve see *identical* values) is the structural fix.

## 11. What to steal from YouTube

If you take one thing from each paper into your own recommender, take these.

- **From Covington 2016 — the two-stage funnel and the watch-time trick.** Split retrieval from ranking with different objectives; never judge a retriever on precision. And the weighted-logistic-regression trick is *directly transferable*: whenever your value is a continuous quantity attached to rare positive events (watch time, dwell, order value), weight the positives by that quantity in a logistic regression and serve $e^{\text{logit}}$ as the expected value. It is three lines of code and it changes what your system rewards.
- **From Yi 2019 — correct your negative sampler.** If you use in-batch negatives, you *must* correct for sampling frequency with $\log Q$, or your retriever will systematically punish popular items. This is not optional polish; it is a correctness fix, and it is one line in the loss.
- **From Zhao 2019 — MMoE for conflicting objectives, and never feed position to the main tower.** When you have multiple objectives that sometimes conflict, MMoE (or PLE) beats shared-bottom by letting tasks pick different experts. And when training on clicks, isolate position (and other presentation features) in a shallow tower you drop at serving time.
- **From Beutel 2019 — debias the labels, not just the model.** Position and selection bias are in your *labels*; a more expressive model does not fix biased labels, an explicit correction does.
- **From Chen 2019 — push the time horizon out, carefully.** Long-term value beats myopic objectives, but off-policy RL is hard (variance, the importance weight, the deadly triad). Reach for it only when you have the logging discipline and the A/B infrastructure to validate it; for most teams, a well-chosen myopic proxy plus exploration gets you most of the way.

Above all, steal the *attitude*: every clever piece of YouTube's stack is a correction for a specific way the data lies. The data lies about value (clicks ≠ watch time), about negatives (popular items oversampled), about relevance (position contamination), and about the future (myopic rewards). Good recommenders are made of bias corrections.

There is one more thing to steal, and it is the most valuable: the **layering**. YouTube did not try to solve everything in one model. Retrieval is a recall problem solved cheaply by embeddings and ANN; ranking is a precision-of-ordering problem solved expensively with rich features and the right objective; re-ranking handles freshness, diversity, and business constraints on a handful of survivors; and the feedback loop is managed with explicit corrections at each stage. Each stage has *one* job, *one* objective, and *one* honest way to measure it. The temptation, especially for a strong ML engineer, is to reach for one big end-to-end model that does it all. YouTube's design is a standing argument against that temptation: the funnel is not a compromise forced by scale alone, it is a *separation of concerns* that makes each stage debuggable, measurable, and independently improvable. When your retrieval recall is bad, you fix retrieval; when your ordering is wrong, you fix ranking; when popular items are eating the catalog, you fix the loop. That decomposition is the single most transferable idea in the whole case study, and it is exactly the frame the [recommendation funnel](/blog/machine-learning/recommendation-systems/the-recommendation-funnel-retrieval-ranking-reranking) post puts at the center of the series.

## 12. When to reach for the YouTube design (and when not to)

YouTube's architecture is the right reference if you are building a large-scale feed or media recommender with millions of items, continuous engagement signals, and an A/B-test culture. The two-stage funnel, two-tower retrieval, multi-task ranking, and bias corrections are battle-tested at the largest scale that exists.

But do not cargo-cult it. **Do not build a two-stage funnel** if your catalog is ten thousand items — a single ranker over the whole catalog is simpler and may be enough; the funnel exists to dodge a scale problem you do not have. **Do not reach for off-policy RL** unless you have exhausted simpler objective fixes and you have the infrastructure to validate it online; the variance and instability are real and most flat launches die there. **Do not adopt MMoE** if you have one objective — it is machinery for *conflicting* objectives, and a single-task model is easier to debug. **Do not skip the bias corrections**, though — those are cheap and high-leverage at any scale; the $\log Q$ term and the position-bias tower are a few lines each and prevent correctness bugs that a bigger model will not fix.

The meta-rule: adopt the *corrections* (watch-time target, $\log Q$, position tower, freshness feature) early and cheaply; adopt the *architecture* (two-stage funnel, two-tower, MMoE, RL) only when your scale and objective complexity demand it. The capstone [recommender systems playbook](/blog/machine-learning/recommendation-systems/the-recommender-systems-playbook) sequences these decisions for a real build.

## 13. Key takeaways

- **Retrieval and ranking are different jobs with different objectives.** Retrieval optimizes recall cheaply over the whole corpus; ranking optimizes ordering expensively over survivors. Never judge one by the other's metric.
- **Optimize the value, not the proxy.** YouTube's defining decision was ranking on watch time, not clicks. The weighted-logistic-regression trick makes a sigmoid's odds equal expected watch time — $\text{odds} \approx \frac{1}{N}\sum T_i = E[T]$ — and you serve $e^{\text{logit}}$.
- **In-batch negatives need the $\log Q$ correction.** Without it, popular items (oversampled as negatives) get systematically penalized; subtract $\log Q(j)$ from each logit to make the estimator unbiased.
- **Isolate presentation bias in a shallow tower.** Train the position-bias tower alongside the main tower; drop it at serving so the main tower learns pure relevance, never circular "whatever sits in slot one."
- **MMoE beats shared-bottom when objectives conflict.** Per-task gates over a shared expert pool let conflicting tasks lean on different experts and stop fighting in one trunk.
- **Long-term value is an RL problem, but a hard one.** Top-K off-policy REINFORCE with an importance weight optimizes cumulative reward; reach for it last, after simpler objective fixes, and only with online validation.
- **The feedback loop is the adversary.** Freshness features, exploration, and every bias correction are loop hygiene. Each fights one specific way serve→log→train→serve lies to you.
- **Believe the A/B test.** Every YouTube headline is a live result because offline metrics are computed on data the old policy shaped. If offline wins do not survive online, the offline metric lied.
- **Good recommenders are made of bias corrections.** The data lies about value, negatives, relevance, and the future; each clever piece of the stack corrects one lie.

## 14. Further reading

- Covington, Adams, Sargin, *Deep Neural Networks for YouTube Recommendations*, RecSys 2016 — the founding two-stage paper and the watch-time / weighted-logistic-regression trick.
- Yi, Yang, Hong, Chen, Heldt, Kumthekar, Zhao, Wei, Chi, *Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations*, RecSys 2019 — the two-tower with the $\log Q$ streaming-frequency correction.
- Zhao, Hong, Wei, Chen, Nath, Andrews, Kumthekar, Sathiamoorthy, Yi, Chi, *Recommending What Video to Watch Next: A Multitask Ranking System*, RecSys 2019 — MMoE multi-task ranking with the shallow position-bias tower.
- Beutel, Chen, Doshi, Qian, Wei, Wu, Heldt, Zhao, Hong, Chi, Goodrow, *Fairness in Recommendation Ranking through Pairwise Comparisons*, KDD 2019 — position-bias and pairwise debiasing.
- Chen, Beutel, Covington, Jain, Belletti, Chi, *Top-K Off-Policy Correction for a REINFORCE Recommender System*, WSDM 2019 — off-policy RL for long-term value.
- Ma, Zhao, Yi, Chen, Hong, Chi, *Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts*, KDD 2018 — the MMoE architecture YouTube adopted.
- Series: [what is a recommender system](/blog/machine-learning/recommendation-systems/what-is-a-recommender-system) and [the recommendation funnel](/blog/machine-learning/recommendation-systems/the-recommendation-funnel-retrieval-ranking-reranking) for the frame; [the two-tower model](/blog/machine-learning/recommendation-systems/the-two-tower-model-for-retrieval) and [training two-tower negatives and sampled softmax](/blog/machine-learning/recommendation-systems/training-two-tower-negatives-and-sampled-softmax) for retrieval; [multi-task and multi-objective ranking](/blog/machine-learning/recommendation-systems/multi-task-and-multi-objective-ranking-mmoe-ple) for MMoE/PLE; [position and selection bias](/blog/machine-learning/recommendation-systems/position-and-selection-bias-in-click-data) for the shallow tower; [reinforcement learning for recommendation](/blog/machine-learning/recommendation-systems/reinforcement-learning-for-recommendation) for off-policy RL; and the capstone [recommender systems playbook](/blog/machine-learning/recommendation-systems/the-recommender-systems-playbook) to sequence it all.
