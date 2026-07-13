---
title: "Sequential and Session-Based Recommendation"
date: "2026-06-19"
publishDate: "2026-06-19"
description: "Stop scoring all-time averages and start predicting the next click: Markov chains, FPMC, and GRU4Rec, with runnable PyTorch and measured Recall@20/MRR@20 lift from modeling order."
tags:
  [
    "recommendation-systems",
    "recsys",
    "sequential-recommendation",
    "session-based",
    "gru4rec",
    "rnn",
    "next-item-prediction",
    "machine-learning",
    "pytorch",
  ]
category: "machine-learning"
subcategory: "Recommendation Systems"
author: "Hiep Tran"
featured: true
readTime: 50
image: "/imgs/blogs/sequential-and-session-based-recommendation-1.png"
---

A user lands on your e-commerce site at 9:14 PM. They are not logged in. You have never seen this browser before. They click a running shoe, then a second running shoe in a different color, then a pair of compression socks. It is now 9:16 PM. What do you put in the "you might also like" rail?

Your matrix-factorization model has nothing to say. It learned a single static vector per user from months of history, and this person has no user id and no history. Your popularity baseline shrugs and shows the ten best-selling items on the whole site, which right now are a phone case, a phone charger, and a kitchen gadget that went viral last week. None of them are shoes. The session is screaming "I am shopping for running gear, right now," and your funnel is deaf to it.

This is the gap that sequential and session-based recommendation closes. Instead of asking "what does this user like on average," it asks a sharper, more useful question: **given the ordered sequence of things this user just interacted with, what is the very next thing they will interact with?** Order matters. Recency matters. What you watched five minutes ago beats your all-time average almost every time. The figure below contrasts the two framings — the static user-by-item view that misses the moment, and the sequential next-item view that rides it.

![A two column comparison showing a static user by item model that scores an all-time average preference on the left versus a sequential model that predicts the next click from recent ordered events on the right](/imgs/blogs/sequential-and-session-based-recommendation-1.png)

In the funnel frame this whole series keeps returning to — retrieval, then ranking, then re-ranking, all fed by the serve-log-train feedback loop — sequential models live mostly in retrieval and early ranking. They generate next-item candidates from the live session. By the end of this post you will be able to: explain exactly why an order-aware model beats a bag-of-items model; derive the next-item softmax objective and the ranking losses that GRU4Rec introduced; implement GRU4Rec in PyTorch with session-parallel mini-batches and a sampled ranking loss; build the leave-last-out evaluation harness that computes Recall@20 and MRR@20 honestly; and read a results table that shows how much lift you actually get from modeling order. We will build from the pre-deep baselines (Markov chains and FPMC), through the first strong neural session model (GRU4Rec), to the convolutional alternative (Caser), and point forward to the attention-based models that took the crown.

## 1. The framing shift: from preference to next-item prediction

For most of this series the object of study has been a **user-item preference matrix**. Rows are users, columns are items, cells are ratings or interactions, and the job is to fill in the blanks. Matrix factorization, collaborative filtering, two-tower retrieval — they all learn a representation of "how much does user $u$ like item $i$" averaged over all the times $u$ and $i$ could have met. It is a static snapshot. The model has one vector per user, and that vector is a smear of every taste the user has ever expressed.

That works beautifully for some products. If you are recommending which movies a long-time Netflix subscriber might enjoy this weekend, their stable long-run taste is genuinely the strongest signal. But it breaks in three common situations, and those three situations are most of the modern web:

1. **Anonymous, short sessions.** A first-time visitor to a news site or an online store has no id and no history. All you have is the handful of clicks in the current session. There is no "user vector" to look up.
2. **Intent that shifts fast.** A logged-in user's all-time average says they love cooking content. But tonight they are deep in a travel-planning session. The average is a liability; it pulls recommendations back toward cooking when the user is clearly somewhere else right now.
3. **Order carries the signal.** Buying a phone, then a case, then a screen protector is a meaningful trajectory. The next click is highly predictable *from the order*. A bag-of-items model that pools those three clicks into one unordered set throws away exactly the structure that makes the next click guessable.

Sequential recommendation reframes the task. The data is no longer a matrix; it is a set of **sequences**. Each user (or each anonymous session) is an ordered list of interactions $s = (i_1, i_2, \dots, i_t)$, and the objective is to predict $i_{t+1}$. This is the same shape as language modeling — predict the next token given the prefix — and it is no accident that the field borrowed RNNs, CNNs, and eventually transformers from NLP.

### Session-based vs sequential: a distinction worth keeping

The terms get used loosely, but the distinction is real and it changes your modeling choices:

- **Session-based recommendation** works with **short, anonymous sessions**. No persistent user id, no long history — just the current session, often a handful of clicks over a few minutes. The classic setting is e-commerce clickstream (a user browsing without logging in) and news (readers who never sign in). The model must work entirely from the session itself. GRU4Rec was designed for exactly this.
- **Sequential recommendation** works with **long per-user histories**. You have a user id and you can see their last 50, 200, or 1000 interactions stretching over months. Here the model can blend long-run taste with short-run intent. SASRec and BERT4Rec, which the next post covers, target this setting.

In practice the boundary blurs — a logged-in shopper has both a long history and a hot current session, and good systems use both. But the framing matters because it tells you what data you can rely on. If a large fraction of your traffic is logged-out, you cannot lean on user embeddings; you must squeeze the signal out of the session. The science and the code in this post work in both settings; I will flag where session-only constraints bite.

The rest of the post is organized as a ladder of model power. We start with the order-blind baselines you must beat, climb to first-order Markov and FPMC (which see only the last click), then to GRU4Rec and Caser (which see the whole session), and finally point at attention. At each rung we keep the same scoreboard: Recall@20 and MRR@20 on a named session dataset, so the lift from each idea is concrete.

## 2. The baselines you must beat (and why they are not silly)

Before any neural model, you owe yourself two baselines. Skipping them is how teams convince themselves a fancy model is winning when it is barely keeping up with a one-liner.

**Popularity.** Recommend the globally most-clicked items, ignoring the session entirely. This is the floor. On a typical e-commerce session dataset, popularity gets a Recall@20 around 0.20 — meaning the true next item is in the top 20 most popular items about one time in five. That is higher than newcomers expect, because real catalogs are heavily skewed and people do click popular things. Any model that cannot clear popularity by a wide margin is broken.

**Item-kNN (co-visitation).** For each item, precompute the items most frequently clicked in the same session, normalized by popularity (cosine over the item-session matrix). At serve time, look at the last clicked item and return its neighbors. This is a one-step, memory-based, order-light model: it uses only the last item and a co-occurrence count, no learning beyond counting. It is shockingly strong. On the same data it often hits Recall@20 around 0.52, beating early neural models that were tuned badly. Item-kNN is the baseline that humbles people.

Why is item-kNN so good? Because the **single strongest predictor of the next click is usually the last click**. Most of the predictable signal in a session lives in the most recent step. A model that nails just the last-step transition already captures the bulk of it. The whole game of sequential modeling is squeezing out the *remaining* signal — the part that needs more than the last item — without losing the last-step signal in the process.

```python
import numpy as np
from collections import defaultdict

def fit_item_knn(sessions, top_k=100):
    """sessions: list of item-id lists (one per session).
    Returns, for each item, its top_k co-visited neighbors by cosine."""
    item_counts = defaultdict(int)
    cooc = defaultdict(lambda: defaultdict(int))
    for sess in sessions:
        uniq = set(sess)
        for it in uniq:
            item_counts[it] += 1
        items = list(uniq)
        for a in range(len(items)):
            for b in range(a + 1, len(items)):
                i, j = items[a], items[b]
                cooc[i][j] += 1
                cooc[j][i] += 1
    neighbors = {}
    for i, row in cooc.items():
        scored = []
        for j, c in row.items():
            # cosine over binary session-occurrence vectors
            sim = c / np.sqrt(item_counts[i] * item_counts[j])
            scored.append((j, sim))
        scored.sort(key=lambda x: -x[1])
        neighbors[i] = scored[:top_k]
    return neighbors

def knn_predict(neighbors, last_item, top_n=20):
    return [j for j, _ in neighbors.get(last_item, [])[:top_n]]
```

Notice what item-kNN cannot do: it conditions only on the **last** item, so the prediction after clicking shoe, shoe, socks is identical to the prediction after socks, shoe, shoe, even though the first sequence is clearly "shopping for running gear" and the second is "drifting." It also has no notion of personalization — every user who just clicked item 42 gets the same neighbors. Those two limitations are exactly what the learned models fix.

One more thing about item-kNN that newcomers get wrong: the normalization is load-bearing. If you rank co-visited items by raw count $n_{ij}$, the most popular items dominate every neighbor list, and you have reinvented popularity with extra steps. Dividing by $\sqrt{n_i\, n_j}$ (the cosine denominator) downweights items that co-occur with *everything* and surfaces items that co-occur *specifically* with the query item. There are a dozen variants of this normalization in the literature — pure cosine, conditional probability $n_{ij}/n_i$, a popularity-discounted score with a tunable exponent $n_{ij}/(n_i^\alpha n_j^{1-\alpha})$ — and the choice of $\alpha$ alone can move Recall@20 by several points. The reason I keep belaboring item-kNN is that a *carefully tuned* item-kNN, with the right normalization and a small recency weighting over the last few clicks rather than only the very last, is a genuinely competitive baseline that has embarrassed more than one neural-model paper. Tune it before you decide your RNN is winning.

Here is the trade-off table for the two baselines and the two learned models we will build, so you can see at a glance what each buys and costs:

| Model | What it conditions on | Learns? | Personalized? | Cold-start transitions? | Serving cost |
|---|---|---|---|---|---|
| Popularity | nothing (global counts) | no | no | n/a | trivial (precomputed list) |
| Item-kNN | last item, co-visitation | counting only | no | no (zero-count = no opinion) | cheap (neighbor lookup) |
| FPMC | last item + user | yes (factorized) | yes | yes (embedding transfer) | cheap (two dot products) |
| GRU4Rec | whole session | yes (recurrent) | via session state | yes | moderate (run GRU live) |

The progression is exactly a ladder of "how much context, learned how richly, at what serving cost." Each rung up adds context (last item, then last item plus user, then the whole session) and each rung adds serving work. That cost column is not decoration — it is the reason a shop with a tiny rail should think hard before climbing past FPMC.

The figure below previews the scoreboard for the whole post: four models scored on whether they use order, whether they reach beyond the last step, whether they work session-only, and their typical Recall@20.

![A comparison matrix with rows for popularity item-kNN FPMC and GRU4Rec and columns for whether each uses order reaches long range works session only and its Recall at 20 score](/imgs/blogs/sequential-and-session-based-recommendation-3.png)

## 3. Markov chains and FPMC: modeling the last step, learned

The first step up from counting co-visitations is to *learn* the transition between consecutive items rather than just count it. A **first-order Markov chain** models the probability of the next item given the current one:

$$
P(i_{t+1} = j \mid i_t = i) = \frac{n_{ij}}{\sum_{k} n_{ik}}
$$

where $n_{ij}$ is the number of times item $j$ followed item $i$ across all sessions. This is item-kNN's cousin: instead of symmetric co-occurrence within a session, it uses the *directed* count of "$j$ came right after $i$." For catalogs with strong directionality (you buy a console *then* the games, rarely the reverse), the direction helps.

The fatal flaw of a raw Markov transition table is **sparsity**. With $|I|$ items there are $|I|^2$ possible transitions, and almost all of them are never observed. A million items means a trillion-cell transition matrix that is essentially all zeros. You cannot estimate $n_{ij}$ for pairs you never saw, so the model has no opinion about most next-items. The fix that made this practical was factorization.

### FPMC: factorized personalized Markov chains

Rendle, Freudenthaler, and Schmidt-Thieme introduced **Factorized Personalized Markov Chains (FPMC)** in 2010 (the same Rendle behind BPR and factorization machines). The idea is to factorize the transition tensor instead of estimating it cell by cell, and to make it *personalized* by adding the user.

Think of the data as a three-way tensor: user $u$, last item $l$, next item $i$. The entry is the probability that user $u$ buys $i$ given they just bought $l$. FPMC factorizes this tensor into low-rank embeddings. The score for "user $u$, who just had item $l$, picks item $i$ next" is:

$$
\hat{x}_{u,l,i} = \underbrace{\langle \mathbf{v}^{UI}_u, \mathbf{v}^{IU}_i \rangle}_{\text{user-item: long-run taste}} + \underbrace{\langle \mathbf{v}^{IL}_i, \mathbf{v}^{LI}_l \rangle}_{\text{item-item: last-step transition}}
$$

Read that carefully because it is the whole model. There are two dot products. The **first** is ordinary matrix factorization: how much does user $u$ like item $i$ in general, the static taste term. The **second** is a factorized first-order Markov transition: how likely is $i$ to follow $l$, learned through embeddings so it generalizes to unseen pairs. FPMC adds them. It is the cleanest possible statement of "blend long-run preference with the last-step transition."

The factorization is what beats sparsity. Even if user $u$ never clicked $i$ after $l$, FPMC can score the triple because the embeddings $\mathbf{v}^{IL}_i$ and $\mathbf{v}^{LI}_l$ were trained on *other* transitions involving $i$ and $l$ separately. Two items that behave similarly get similar embeddings, so the model transfers. This is the same generalization trick matrix factorization uses for the user-item matrix, applied to the transition matrix.

FPMC is trained with the same pairwise ranking objective as BPR, which this series covered in [implicit feedback models](/blog/machine-learning/recommendation-systems/implicit-feedback-models-als-and-bpr): for each observed transition, sample a negative next-item and push the observed item's score above the negative's. We will derive the pairwise ranking loss in detail in the GRU4Rec section because GRU4Rec's TOP1 and BPR-max losses are descendants of it.

Concretely, the FPMC training step looks like this. For an observed triple "user $u$, last item $l$, next item $i^+$," draw a random item $i^-$ the user did *not* pick next, and maximize the margin between their scores:

$$
\max_{\Theta} \; \log \sigma\big(\hat{x}_{u,l,i^+} - \hat{x}_{u,l,i^-}\big) - \lambda \lVert \Theta \rVert^2
$$

where $\Theta$ collects all four embedding tables and $\lambda$ is L2 regularization. The gradient flows into both dot products: it nudges $i^+$'s next-item embedding toward both the user's taste vector *and* the last item's transition vector, while pushing $i^-$ away from both. That dual update is the mechanism behind the cold-start transfer in the worked example — a new item's embedding gets pulled toward the contexts it appears in, even contexts it never directly followed. The cost per step is $O(d)$ for the dot products, so FPMC trains and serves cheaply; the whole model is four embedding tables and an addition.

A subtle but important property of the FPMC form: the two terms are **not** equally weighted in practice — you can scale them, and how you scale them is a knob for the static-vs-recent balance. Crank up the transition term and the model becomes nearly a pure Markov chain (great for fast-moving sessions, bad for stable taste); crank up the preference term and it drifts back toward plain matrix factorization (great for stable taste, deaf to the moment). The "right" balance is dataset-specific and worth a hyperparameter sweep. This same tension — long-run preference versus short-run intent — reappears in every sequential model; FPMC just makes it two literal terms you can see and tune.

#### Worked example: a Markov transition the model can and cannot see

Suppose the true catalog dynamics are: after buying a **phone** (item P), 60% of users buy a **case** (C), 30% buy a **charger** (G), 10% buy something random. A raw count-based Markov chain that observed 1,000 phone purchases sees roughly 600 transitions P to C, 300 P to G, 100 scattered. It estimates $P(C \mid P) = 0.60$, $P(G \mid P) = 0.30$ cleanly. Good.

Now a brand-new case **C2** launches. It has zero observed transitions because no one has bought a phone-then-C2 yet. The count-based Markov chain estimates $P(C2 \mid P) = 0$ — it will never recommend C2 after a phone, forever, until enough P-to-C2 transitions accumulate. FPMC does better: C2's embedding $\mathbf{v}^{IL}_{C2}$ is trained from C2's behavior in *other* contexts (it co-occurs with phone accessories, sits in the "case" cluster), so its embedding lands near C1's. The dot product $\langle \mathbf{v}^{IL}_{C2}, \mathbf{v}^{LI}_{P} \rangle$ inherits C1's affinity for following a phone. FPMC recommends C2 after a phone on day one. That cold-start transfer is the entire payoff of factorizing the transition.

The hard limit of FPMC, shared with all first-order Markov models, is the word *first-order*: it conditions only on the **last** item $l$. The session shoe, shoe, socks and the session socks, shoe, shoe both end on the same item and produce the same FPMC prediction. Everything before the last click is invisible. To use the whole session you need a model with memory, and that is the RNN.

## 4. Why a bag-of-items model cannot do what a sequence model can

Before we build the RNN, it is worth being precise about *what* order buys you, because "order matters" is easy to say and easy to overclaim. The cleanest way to see it is to contrast a **bag-of-items** model against a **sequence** model on the same session.

A bag-of-items model pools the session into one unordered representation — say, the mean of the item embeddings — and scores next-items from that pooled vector. It is permutation-invariant by construction: shuffle the clicks and the prediction is identical. The figure below makes the contrast concrete.

![A two column comparison showing a bag of items model that pools clicks with no order and is shuffle invariant on the left versus an ordered sequence model that weights recent clicks and reads intent shifts on the right](/imgs/blogs/sequential-and-session-based-recommendation-6.png)

When does permutation-invariance hurt? Three ways:

1. **Recency.** The last click is usually the strongest predictor of the next, but a mean pools it equally with the first click of the session. If a user browsed laptops for ten clicks and then pivoted to laptop *bags*, the mean is still dominated by laptops; a sequence model weights the recent pivot and follows the user to bags.
2. **Directionality.** Phone-then-case is a different signal from case-then-phone. The bag sees the same set either way. A sequence model encodes the arrow.
3. **Repetition patterns.** A session that clicks the same item three times in a row signals strong intent or indecision; a session that clicks it once early and then moves on signals it was rejected. The bag collapses both to "clicked once."

There is a fourth, subtler point — long-range structure. Consider a session: red dress, red shoes, red bag, then a click on a **blue dress**. The intent just shifted from "red outfit" to maybe "blue outfit." Predicting the next item well requires noticing both the early color theme *and* the recent break from it. A bag-of-items mean blends all four into a muddy average. A sequence model with memory can represent "was building a red set, just switched to blue" as a trajectory in hidden-state space. The figure below contrasts a Markov model (which sees only the last item, so it sees only "blue dress" and loses the red context) against an RNN (which carries the whole session in its hidden state).

![A two column comparison showing a first order Markov chain that conditions only on the last item on the left versus an RNN that carries a hidden state remembering the whole session and longer patterns on the right](/imgs/blogs/sequential-and-session-based-recommendation-5.png)

Here is the precise claim, stated as something you could prove: a permutation-invariant function $f(\{i_1, \dots, i_t\})$ assigns the same score to every ordering of the same multiset of clicks. Therefore it **cannot** distinguish any two sessions that differ only in order. If the next-item distribution genuinely depends on order — and for clickstreams it provably does, because $P(i_{t+1} \mid i_1, \dots, i_t) \neq P(i_{t+1} \mid \text{shuffle})$ in measured data — then any permutation-invariant model has a non-zero irreducible error that an order-aware model can beat. That irreducible gap is the headroom sequential models capture. In practice on session data it is worth several points of Recall@20, as the results table will show.

## 5. GRU4Rec: the first strong neural session model

Hidasi, Karatzoglou, Baltrunas, and Tikk introduced **GRU4Rec** in 2016 ("Session-Based Recommendations with Recurrent Neural Networks," ICLR workshop, expanded in later work). It was the first RNN to clearly beat the strong item-kNN baseline on session data, and it brought three ideas that are still worth knowing even though attention models have since overtaken it: a **GRU over the session**, **session-parallel mini-batches**, and **ranking losses tailored to session rec** (TOP1 and later BPR-max).

### The architecture: embed, recur, score

The model is a clean next-item predictor. At each step it takes the current item, embeds it, updates a recurrent hidden state, and emits a score for every item in the catalog. The figure below shows it unrolled in time as a layered dataflow — not as a recurrent ring, but as the forward pass actually computes it, step by step.

![A layered dataflow diagram of GRU4Rec showing item ids being embedded then passed through GRU steps unrolled in time with the final hidden state scoring every catalog item for the next click](/imgs/blogs/sequential-and-session-based-recommendation-2.png)

The recurrence is a GRU (Gated Recurrent Unit), a lighter cousin of the LSTM. At step $t$, given the embedded current item $\mathbf{x}_t$ and the previous hidden state $\mathbf{h}_{t-1}$, the GRU computes:

$$
\begin{aligned}
\mathbf{z}_t &= \sigma(W_z \mathbf{x}_t + U_z \mathbf{h}_{t-1}) \quad &\text{(update gate)}\\
\mathbf{r}_t &= \sigma(W_r \mathbf{x}_t + U_r \mathbf{h}_{t-1}) \quad &\text{(reset gate)}\\
\tilde{\mathbf{h}}_t &= \tanh(W \mathbf{x}_t + U(\mathbf{r}_t \odot \mathbf{h}_{t-1})) \quad &\text{(candidate)}\\
\mathbf{h}_t &= (1 - \mathbf{z}_t) \odot \mathbf{h}_{t-1} + \mathbf{z}_t \odot \tilde{\mathbf{h}}_t \quad &\text{(blend)}
\end{aligned}
$$

The point of the gates: the **update gate** $\mathbf{z}_t$ decides how much of the old state to keep versus how much new information to write, and the **reset gate** $\mathbf{r}_t$ decides how much past context to forget when forming the candidate. Together they let the hidden state carry information across many steps without the vanishing-gradient problems of a vanilla RNN. That is what gives the model *memory*: $\mathbf{h}_t$ is a learned summary of the whole session so far, and unlike a Markov model it does not collapse to the last item.

The output layer scores all items by projecting the hidden state: $\mathbf{s}_t = \mathbf{h}_t W_o^\top + \mathbf{b}$, a vector of one score per catalog item. The next-item prediction is the top-K of $\mathbf{s}_t$.

### The next-item softmax objective and why we approximate it

The "correct" objective is a softmax over the entire catalog. The probability the model assigns to the true next item $i^+$ at step $t$ is:

$$
P(i^+ \mid \mathbf{h}_t) = \frac{\exp(s_{t, i^+})}{\sum_{j \in I} \exp(s_{t, j})}
$$

and we minimize the negative log-likelihood $-\log P(i^+ \mid \mathbf{h}_t)$ summed over all steps. This is exactly language modeling with the catalog as the vocabulary.

The problem: that denominator sums over *every item in the catalog*. With a million items, every training step computes a million-way dot product and softmax. That is far too expensive. The same scaling wall that forces sampled softmax in two-tower retrieval ([training two-tower negatives and sampled softmax](/blog/machine-learning/recommendation-systems/training-two-tower-negatives-and-sampled-softmax)) forces an approximation here. GRU4Rec's contribution was a *ranking-based* approximation that turned out to fit session rec especially well.

### TOP1 and BPR-max: ranking losses for session rec

Instead of a full softmax, GRU4Rec scores the positive item against a small set of sampled negatives and uses a **pairwise ranking loss**. Start from the BPR loss this series derived in [implicit feedback models](/blog/machine-learning/recommendation-systems/implicit-feedback-models-als-and-bpr). For a positive score $s^+$ and a negative score $s^-_k$ (one of $N$ sampled negatives), BPR maximizes the probability that the positive ranks above the negative:

$$
L_{\text{BPR}} = -\frac{1}{N}\sum_{k=1}^{N} \log \sigma(s^+ - s^-_k)
$$

The gradient sees the *difference* $s^+ - s^-_k$, not the absolute scores, which is exactly why pairwise losses beat pointwise for top-K ranking: the model is optimized to get the order right, not to nail a calibrated probability it never needs.

GRU4Rec's **TOP1** loss is a tailored variant. It combines a ranking term (push the positive above each negative) with a regularizer that pushes negative scores toward zero, which stabilizes training and prevents score inflation:

$$
L_{\text{TOP1}} = \frac{1}{N}\sum_{k=1}^{N} \Big[ \sigma(s^-_k - s^+) + \sigma\big((s^-_k)^2\big) \Big]
$$

The first term is the ranking pressure; the second is the score regularizer. TOP1 worked well but had a weakness: when many negatives are easy (their score is already far below the positive), their gradients vanish and learning stalls — the same "gradient washout" that plagues naive negative sampling everywhere in recsys.

The fix, introduced in the 2018 follow-up ("Recurrent Neural Networks with Top-k Gains for Session-based Recommendations"), is **BPR-max**: weight each negative by how relevant it is, focusing the gradient on the negatives that actually threaten the positive's rank. With softmax weights $w_k = \text{softmax}(s^-)_k$ over the negative scores:

$$
L_{\text{BPR-max}} = -\log \sum_{k=1}^{N} w_k\, \sigma(s^+ - s^-_k) + \lambda \sum_{k=1}^{N} w_k\, (s^-_k)^2
$$

The intuition: the hardest negative (highest $s^-_k$) gets the most weight, so the model spends its gradient budget on the negatives that are closest to beating the true item. This is hard-negative mining baked into the loss, and it gave the "top-k gains" that the paper's title advertises — a meaningful Recall and MRR bump over plain TOP1 and BPR.

### The session-parallel mini-batch trick

Here is the engineering idea that made GRU4Rec trainable at all, and it is genuinely clever. Sessions have wildly different lengths — some are 2 clicks, some are 200. If you batch sessions naively (pad them all to the longest), short sessions are mostly padding and you waste compute, and the GPU sits idle. If you process one session at a time, you get no parallelism.

GRU4Rec's solution: process sessions **in parallel along the time axis**, not the session axis. Put $B$ sessions side by side as the rows of a batch. At each time step, advance *all* of them by one click simultaneously — column by column. When a session ends (runs out of clicks), drop it and slot the next session into its row, resetting that row's hidden state. The batch size stays constant; sessions of different lengths coexist; no padding is wasted. The figure below shows the layout: sessions as rows, time steps as columns, with a finished session swapped in place.

![A grid showing session parallel mini batches with sessions in rows and time steps in columns where each column is one synchronized GRU step and a finished session is swapped in place](/imgs/blogs/sequential-and-session-based-recommendation-7.png)

A second payoff falls out of this layout for free: **session-parallel negative sampling**. At each step, the positive next-items of the *other* sessions in the batch make excellent negatives for the current session. They are real items, they are popularity-distributed (popular items appear more often as positives, so they appear more often as negatives — desirable, because you want to learn to rank against the items that actually compete), and they cost nothing extra to gather because they are already in the batch. This is the session-rec version of in-batch negatives, and it is why GRU4Rec's loss only needs $B-1$ negatives per step without a separate sampler. For very large catalogs the original paper adds extra sampled negatives drawn proportional to popularity, but the in-batch ones do most of the work.

### Why an RNN captures order a bag-of-items model cannot, made rigorous

It is worth being precise about *why* the recurrence works, because "it has memory" is hand-wavy. The claim to prove is: the GRU computes a function of the session that is **not** permutation-invariant, and moreover it can in principle represent any function the next-item distribution requires (it is a universal sequence approximator), so it has strictly more representational power than any pooling model.

Start with the hidden-state recursion $\mathbf{h}_t = g(\mathbf{h}_{t-1}, \mathbf{x}_t)$. Unroll it: $\mathbf{h}_t = g(g(\dots g(\mathbf{h}_0, \mathbf{x}_1)\dots, \mathbf{x}_{t-1}), \mathbf{x}_t)$. The function $g$ is applied in a specific *nested* order, so $\mathbf{h}_t$ depends on the inputs through that nesting. Because $g$ is in general non-commutative — $g(g(h, a), b) \neq g(g(h, b), a)$ for almost all gated cells — swapping two inputs changes $\mathbf{h}_t$. Contrast the pooling model $\mathbf{h} = \frac{1}{t}\sum_k \mathbf{x}_k$, where addition *is* commutative, so any permutation gives the same $\mathbf{h}$. That single algebraic difference — non-commutative composition versus commutative summation — is the formal reason the RNN can tell shoe-then-socks from socks-then-shoe and the bag-of-items model cannot.

The deeper point is about *what the hidden state stores*. The GRU's gates let $\mathbf{h}_t$ act as a learned, lossy compression of the entire prefix, with the model free to choose what to keep. For the red-dress-then-blue-dress session, the model can learn to write "color theme = red, recent shift = blue" into different coordinates of $\mathbf{h}_t$, so the next-item scorer sees both the long-range theme and the recent break. A bag-of-items mean cannot represent "recent shift away from the early theme" in any coordinate, because the mean has thrown away which clicks were early and which were late. The cost of this expressiveness is the **sequential bottleneck**: $\mathbf{h}_t$ is a fixed-size vector, so a very long session must be squeezed through the same $d$ numbers, and early information can be overwritten. That bottleneck is precisely what self-attention removes by letting the scorer read any past item directly — which is the bridge to SASRec.

### The cost of training: backprop through time

The recurrence has a price tag. Forward, the GRU does $O(d^2)$ work per step (the gate matrix multiplies), so a session of length $T$ costs $O(T d^2)$ — linear in length, which is fine. The catch is the *backward* pass. To compute exact gradients you backpropagate through the whole unrolled chain (backpropagation through time, BPTT), which requires storing every intermediate hidden state, costing $O(T d)$ memory and $O(T d^2)$ time, and the gradient can vanish or explode over long chains. **Truncated BPTT** — the `hidden.detach()` in the training loop — cuts the chain at each step (or every $k$ steps), bounding memory to $O(k d)$ and sidestepping the vanishing-gradient blow-up, at the cost of not learning dependencies longer than $k$ steps. For session rec this is usually fine because sessions are short and most signal is recent, but it is a real limitation: a truncated-at-1 GRU literally cannot learn a dependency that spans more than one step through gradients, so in the limit it degrades toward a learned Markov model. Pick your truncation length to cover the dependency range your domain actually has.

## 6. Implementing GRU4Rec in PyTorch

Enough theory. Here is a compact, runnable GRU4Rec. I will show the model, the session-parallel data loader, the BPR-max loss, the training loop, and the leave-last-out evaluation harness. This is real PyTorch you can copy and adapt; the only thing abstracted away is loading your specific dataset into the `sessions` list of item-id lists.

### The model

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class GRU4Rec(nn.Module):
    def __init__(self, n_items, emb_dim=64, hidden=128, n_layers=1, dropout=0.1):
        super().__init__()
        # index 0 is reserved as the padding / no-item id
        self.item_emb = nn.Embedding(n_items + 1, emb_dim, padding_idx=0)
        self.gru = nn.GRU(emb_dim, hidden, n_layers,
                          batch_first=True, dropout=dropout if n_layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        # output projection tied to the item space
        self.out = nn.Linear(hidden, emb_dim)
        self.n_items = n_items

    def forward(self, item_seq, hidden=None):
        # item_seq: (B,) one step, or (B, T) a whole session
        if item_seq.dim() == 1:
            item_seq = item_seq.unsqueeze(1)          # (B, 1)
        x = self.item_emb(item_seq)                    # (B, T, emb)
        out, hidden = self.gru(x, hidden)              # (B, T, hidden)
        out = self.drop(out)
        proj = self.out(out)                           # (B, T, emb)
        return proj, hidden

    def score(self, proj, item_ids):
        # proj: (B, emb) hidden projection ; item_ids: (N,) candidate items
        cand = self.item_emb(item_ids)                 # (N, emb)
        return proj @ cand.t()                         # (B, N) scores
```

Two design choices worth calling out. First, `padding_idx=0` makes the embedding for the pad token a fixed zero vector with no gradient, so padding contributes nothing — this is the mask. Second, scoring against `self.item_emb(item_ids)` ties the output to the input embedding table, which both halves the parameters and is known to help next-item models (the same tying trick used in language models). You can instead use a separate output embedding; tying is the cheaper default.

### Session-parallel batching

```python
import numpy as np

class SessionParallelLoader:
    """Yields (input_items, target_items, mask_reset) per time step.
    Implements the GRU4Rec session-parallel mini-batch trick."""
    def __init__(self, sessions, batch_size=64):
        # keep sessions with >= 2 clicks (need at least one transition)
        self.sessions = [s for s in sessions if len(s) >= 2]
        self.batch_size = batch_size

    def __iter__(self):
        B = self.batch_size
        order = np.random.permutation(len(self.sessions))
        next_sess = B                          # pointer to the next session to load
        # each row holds (session_index, position_within_session)
        active = [(order[i], 0) for i in range(min(B, len(order)))]
        finished = False
        while not finished:
            inp = np.zeros(B, dtype=np.int64)
            tgt = np.zeros(B, dtype=np.int64)
            reset = np.zeros(B, dtype=bool)    # rows whose hidden state must reset
            for row, (sidx, pos) in enumerate(active):
                if sidx < 0:                   # exhausted row
                    continue
                sess = self.sessions[sidx]
                inp[row] = sess[pos]
                tgt[row] = sess[pos + 1]
            yield (torch.as_tensor(inp), torch.as_tensor(tgt),
                   torch.as_tensor(reset))
            # advance positions; swap in new sessions where needed
            new_active = []
            for row, (sidx, pos) in enumerate(active):
                if sidx < 0:
                    new_active.append((-1, 0)); continue
                sess = self.sessions[sidx]
                if pos + 2 < len(sess):        # still has a next transition
                    new_active.append((sidx, pos + 1))
                else:                          # session done, load a new one
                    if next_sess < len(order):
                        new_active.append((order[next_sess], 0))
                        reset[row] = True      # tell trainer to reset this row
                        next_sess += 1
                    else:
                        new_active.append((-1, 0))
            active = new_active
            if all(s < 0 for s, _ in active):
                finished = True
```

The `reset` mask is the bookkeeping for the swap-in: when a row finishes its session and a new one slots in, that row's GRU hidden state must be zeroed so the new session does not inherit the old one's memory. The trainer applies it.

### The BPR-max loss with in-batch negatives

```python
def bpr_max_loss(scores_pos, scores_neg, reg=1.0):
    """scores_pos: (B,) score of each row's true next item.
    scores_neg: (B, N) scores of N negatives for each row.
    Implements BPR-max: softmax-weight the negatives, then rank."""
    weights = torch.softmax(scores_neg, dim=1)              # (B, N)
    diff = scores_pos.unsqueeze(1) - scores_neg             # (B, N)
    ranking = -torch.log((weights * torch.sigmoid(diff)).sum(dim=1) + 1e-24)
    score_reg = reg * (weights * scores_neg.pow(2)).sum(dim=1)
    return (ranking + score_reg).mean()
```

### The training loop

```python
def train_gru4rec(model, loader, n_items, epochs=5, lr=1e-3, device="cuda"):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        total, steps = 0.0, 0
        hidden = None
        for inp, tgt, reset in loader:
            inp, tgt = inp.to(device), tgt.to(device)
            # reset hidden state of rows whose session just swapped in
            if hidden is not None and reset.any():
                mask = (~reset).float().to(device).view(1, -1, 1)
                hidden = hidden * mask
            proj, hidden = model(inp, hidden)              # (B, 1, emb)
            hidden = hidden.detach()                       # truncated BPTT
            proj = proj[:, -1, :]                          # (B, emb)
            # in-batch negatives: every row's target is a negative for the others
            pos_emb = model.item_emb(tgt)                  # (B, emb)
            scores_pos = (proj * pos_emb).sum(dim=1)       # (B,)
            scores_neg = proj @ pos_emb.t()                # (B, B) full matrix
            # mask out the diagonal (a row's own positive is not a negative)
            eye = torch.eye(scores_neg.size(0), device=device).bool()
            scores_neg = scores_neg.masked_fill(eye, -1e9)
            loss = bpr_max_loss(scores_pos, scores_neg)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item(); steps += 1
        print(f"epoch {epoch}: loss {total/max(steps,1):.4f}")
```

The `hidden = hidden.detach()` is **truncated backpropagation through time**: we carry the hidden state forward across steps so the model has memory, but we cut the gradient at each step so we are not backpropagating through the entire session length. This keeps memory bounded and is standard for streaming RNN training. The in-batch negative construction — score every row's projection against every row's positive, then mask the diagonal — is the session-parallel negative trick in three lines.

### Evaluation: leave-last-out, Recall@20 and MRR@20

The evaluation protocol for next-item recommendation is **leave-last-out** (also called the autoregressive or next-item protocol). For each session in the test set, you feed the model all but the last click and ask it to rank the catalog; the held-out last click is the ground truth. You score the rank of that true item.

```python
def evaluate(model, test_sessions, n_items, k=20, device="cuda"):
    model.eval()
    all_items = torch.arange(1, n_items + 1, device=device)
    recall_hits, mrr_sum, n = 0, 0.0, 0
    with torch.no_grad():
        for sess in test_sessions:
            if len(sess) < 2:
                continue
            prefix = torch.as_tensor(sess[:-1], device=device).unsqueeze(0)  # (1,T)
            target = sess[-1]
            proj, _ = model(prefix)                       # (1, T, emb)
            proj = proj[:, -1, :]                         # (1, emb): last step
            scores = model.score(proj, all_items).squeeze(0)  # (n_items,)
            # rank of the true item (1-based)
            true_score = scores[target - 1]
            rank = (scores > true_score).sum().item() + 1
            if rank <= k:
                recall_hits += 1
                mrr_sum += 1.0 / rank
            n += 1
    return recall_hits / n, mrr_sum / n                   # Recall@k, MRR@k
```

Two correctness notes that bite people. First, **full ranking, not sampled**: this harness ranks against the *entire* catalog. The KDD 2020 result "On Sampled Metrics for Item Recommendation" (Krichene and Rendle) showed that ranking the true item against a small random sample of negatives — a once-common shortcut — produces metrics that are *inconsistent*: they can reverse the ranking of models compared to full evaluation. For a fair comparison, evaluate against the full catalog (or use the exact correction from that paper). Second, **temporal split, no leakage**: split sessions by time so the test sessions come strictly after the training sessions. Splitting randomly lets the model peek at the future and inflates every number.

### Or skip the boilerplate: GRU4Rec in RecBole

If you would rather not hand-roll the training loop, **RecBole** ships GRU4Rec, FPMC, Caser, SASRec, and BERT4Rec behind a single config-driven interface with a unified, leakage-safe evaluation harness. The session-parallel batching, the loss, the leave-one-out split, and full-catalog metrics are all handled. A minimal config for a fair GRU4Rec run looks like this:

```yaml
# gru4rec.yaml — RecBole config for a session dataset
model: GRU4Rec
dataset: yoochoose-clicks

# fields the loader expects in the .inter file
USER_ID_FIELD: session_id
ITEM_ID_FIELD: item_id
TIME_FIELD: timestamp
load_col:
  inter: [session_id, item_id, timestamp]

# treat each session as a sequence; cap length
MAX_ITEM_LIST_LENGTH: 50
ITEM_LIST_LENGTH_FIELD: item_length

# model hyperparameters
embedding_size: 64
hidden_size: 128
num_layers: 1
dropout_prob: 0.1
loss_type: BPR              # CE for sampled-softmax, BPR for pairwise

# leakage-safe evaluation: temporal leave-one-out, FULL ranking
eval_args:
  split: { LS: valid_and_test }   # leave-one-out
  order: TO                       # time-ordered, no future leakage
  mode: full                      # rank against the WHOLE catalog, not a sample
  group_by: user
metrics: [Recall, MRR, NDCG, Hit]
topk: [10, 20]
valid_metric: MRR@20
```

```bash
# run it
python run_recbole.py --config_files gru4rec.yaml
```

Two flags in that config are the ones that decide whether your numbers are trustworthy. `order: TO` forces a **time-ordered** split so the test interaction is strictly the user's last in time — no future leakage. `mode: full` forces **full-catalog ranking** instead of sampled negatives — the KDD 2020 correctness requirement. Get those two right and RecBole's leaderboard is comparable across models; get them wrong and you are back to inconsistent numbers. Swap `model: GRU4Rec` for `FPMC`, `Caser`, `SASRec`, or `BERT4Rec` and the same harness reruns, which makes RecBole the fastest honest way to produce the comparison table in this post on your own data.

### Stress-testing the implementation

A model that trains is not a model that ships. Reason through the failure modes before you trust the numbers:

- **What if 80% of sessions have only one click?** Then there is no transition to learn from those sessions (you need at least two clicks to form an input-target pair), and the loader drops them. Your effective training set may be a fraction of your raw traffic, and the model is learning only from the engaged minority. The metric on the long sessions can look great while most users get effectively a popularity fallback. Always report what fraction of sessions are model-eligible.
- **What if the catalog has 50 million items?** The output layer and the in-batch-only negatives stop being enough — the chance that a batch of 64 contains a good hard negative for any given positive is tiny in a 50M-item space. You add sampled negatives drawn from a popularity distribution (the same $\log Q$-corrected sampled softmax as two-tower retrieval), and at serve time you stop scoring all items and switch to ANN retrieval over the item embeddings using the hidden state as the query. The model code barely changes; the loss and the serving path do.
- **What if most negatives are false negatives?** In session data, an item the user did not click next is not necessarily an item they dislike — they just clicked something else first; they might click your "negative" three steps later. This is the missing-not-at-random problem that haunts implicit feedback. In-batch negatives are mostly safe (other sessions' positives are genuinely unrelated to this session), but aggressive hard-negative mining can start penalizing items the user actually likes. If hard negatives hurt, suspect false negatives.
- **What if offline Recall@20 rises but online clicks are flat?** The usual suspects: a temporal-split violation inflating offline numbers, a train-serve skew (the sequence is built differently offline than online — for example offline you include add-to-cart events but online you only have clicks), or position bias (the rail's top slot gets clicks regardless of model quality). Before blaming the model, diff the exact sequence the model sees offline against the one it sees online for the same session. Feature skew in sequence construction is the silent killer of session-rec launches.

## 7. Caser: convolutions as an alternative to recurrence

GRU4Rec processes the session step by step. **Caser** (Tang and Wang, "Personalized Top-N Sequential Recommendation via Convolutional Sequence Embedding," WSDM 2018) took a different route: treat the recent sequence as an **image** and run convolutions over it.

Stack the embeddings of the last $L$ items into an $L \times d$ matrix — $L$ rows (time), $d$ columns (embedding dimensions). Caser slides two kinds of filters over this matrix:

- **Horizontal filters** span several time steps across all embedding dimensions (filter shape $h \times d$ for window $h$). These capture **union-level** patterns: "items A and B and C together, in any nearby arrangement, predict D." They model *which combination* of recent items matters, picking up multi-step skip patterns that a strict first-order Markov model misses.
- **Vertical filters** span all time steps within a single embedding dimension (filter shape $L \times 1$). These act like a learned **weighted sum over time** — a point-level pattern that aggregates each latent dimension across the recent window, which is essentially a learned recency weighting.

The two filter banks are concatenated, optionally combined with a user embedding for personalization, and projected to item scores. Caser's pitch was that convolutions capture both *point-level* (one item predicts the next) and *union-level* (a set of recent items jointly predicts the next) and *skip behaviors* (non-consecutive influence), all without a sequential bottleneck — and convolutions parallelize across the window where an RNN must step.

In benchmarks Caser is broadly competitive with GRU4Rec — sometimes a bit better on datasets with strong short skip patterns, sometimes a bit worse on long sessions where recurrence shines. Neither is a clear universal winner. What matters historically is that both proved order-aware neural models beat the FPMC and item-kNN baselines, and both were soon overtaken by **self-attention**.

The move to attention is the natural next rung: an RNN compresses the whole session into a single fixed hidden state (a bottleneck), and convolutions only see a fixed window. Self-attention lets the model look directly at *any* previous item with a learned weight, so it can capture long-range dependencies without a recurrence bottleneck or a fixed window. That is **SASRec** (self-attentive sequential recommendation) and **BERT4Rec** (bidirectional, masked-item training), which the post on [self-attention for sequences](/blog/machine-learning/recommendation-systems/self-attention-for-sequences-sasrec-bert4rec) covers in full. For now, the mental model to carry forward: FPMC sees one step back, GRU4Rec and Caser see a window or a compressed history, and attention sees everything with learned weights.

There is also a parallelism story that pushed the field toward attention and CNNs and away from RNNs. The GRU's recurrence is inherently **sequential**: to compute $\mathbf{h}_t$ you must first compute $\mathbf{h}_{t-1}$, so the forward pass cannot be parallelized across time steps within one session. Caser's convolutions and the transformer's attention both compute all positions at once, which on modern accelerators is a large throughput win for training (you saturate the GPU instead of stepping a recurrence). At serving time the gap is smaller — you usually encode one session at a time and the session is short — but for training on hundreds of millions of sessions, the ability to process a whole session in one parallel pass is a real reason the post-2018 literature is dominated by attention. The accuracy gains of SASRec over GRU4Rec are typically modest (a point or two of Recall@20 on standard benchmarks); the *engineering* gains in training throughput are part of why it won.

### Serving a sequential model

The architecture decision does not end at the loss; it continues into the serving path, and sequential models have a serving wrinkle that static models do not: the **query is computed from the live session**, so you cannot precompute the user vector offline the way two-tower retrieval does. Every request runs the encoder over the current session to produce a query, then retrieves. There are two regimes:

- **Small catalog (up to ~100K items):** score every item with a single matrix multiply of the hidden state against the item-embedding table. A 100K-by-64 matrix-vector product is well under a millisecond on CPU. No ANN index needed.
- **Large catalog (millions of items):** treat the hidden state as a query vector and run approximate nearest neighbor retrieval over the item embeddings, exactly the MIPS setup from [approximate nearest neighbor serving](/blog/machine-learning/recommendation-systems/approximate-nearest-neighbor-serving-faiss-hnsw-scann). The item embeddings are static, so you build the ANN index once and refresh it as the catalog changes; only the query (the session encoding) is computed per request.

The practical cost is the encoder forward pass. For a GRU over a 50-item session at hidden size 128, that is 50 sequential gate computations — a few hundred microseconds to low single-digit milliseconds depending on hardware and batching. Caching helps: if the session has not changed since the last request (the user is idle), reuse the cached hidden state. And because the GRU hidden state is *incrementally updatable* — when a new click arrives you advance one step from the cached $\mathbf{h}_{t}$ rather than re-encoding the whole session — sequential models are actually well-suited to streaming serving, which is a quiet advantage over re-encoding everything on each request.

## 8. The session-rec pipeline end to end

Step back from individual models and look at the full path from a raw click to a served recommendation, because the model is only the middle of it. The figure below lays out the pipeline as layers.

![A vertical stack diagram of the session recommendation pipeline showing clickstream events becoming ordered per session sequences then a sequential model then scoring all items then a served next item top K list](/imgs/blogs/sequential-and-session-based-recommendation-4.png)

Walking the layers:

1. **Clickstream events.** Raw `(user_or_anon_id, item_id, event_type, timestamp)` rows streaming off your front end. You typically filter to one event type (clicks, or add-to-carts) or weight types differently.
2. **Ordered sequences.** Group events into sessions (a common rule: a session ends after 30 minutes of inactivity), sort each by timestamp, drop sessions shorter than 2 events, and cap very long sessions to the last $N$ (say 50) items so the model and serving stay bounded. This is where you pad and mask: pad short sequences to a fixed length with the reserved id 0, and mask the padding in the loss and attention.
3. **Sequential model.** GRU4Rec, Caser, SASRec — whatever you chose. Trained with leave-last-out and a ranking or sampled-softmax loss.
4. **Score all items.** At serve time, run the live session through the model to get a hidden state, then score candidates. For a small catalog you score everything; for a large catalog you turn the hidden state into a query vector and do **approximate nearest neighbor** retrieval against item embeddings, exactly the MIPS setup from [approximate nearest neighbor serving](/blog/machine-learning/recommendation-systems/approximate-nearest-neighbor-serving-faiss-hnsw-scann). Sequential retrieval is just two-tower retrieval where the user tower is the sequence encoder.
5. **Next-item top-K.** The ranked list, possibly re-ranked for diversity or business rules before display.

### Practical issues that decide whether this ships

**Variable-length sequences, padding, and masking.** Real sessions range from 2 to hundreds of clicks. You pad to a fixed length for batching and mask the pad positions so they contribute zero to the loss and (for attention) are not attended to. The `padding_idx=0` trick handles the embedding side; the loss must also skip pad targets. Getting masking wrong is the single most common silent bug in sequence models — the model trains, the loss goes down, and the metrics are quietly capped because the model is learning to predict padding.

**Recency bias.** Sequence models naturally weight recent items, which is usually right, but it can over-fixate on the very last click. If a user mis-clicks an item they did not want, a recency-greedy model chases the mistake. Mitigations: train on slightly older targets too (predict $i_{t}$ from $i_{\lt t}$ for several $t$, not only the last), or blend the sequence score with a longer-run preference term (which is exactly what FPMC's first dot-product does, and what hybrid sequential models do).

**Repeat consumption.** In many domains the next item is one the user *already* interacted with — replaying a song, reordering groceries, revisiting a product. A naive next-item model often under-recommends repeats because the training signal for "new item" dominates. Some systems add an explicit **repeat-or-explore** gate (RepeatNet is the canonical example) that decides whether to recommend something from the session or something new. If your domain has high repeat rates (music, grocery), measure repeat-recall separately; a model with great overall Recall@20 can be terrible at the repeats that drive retention.

**Evaluation protocol.** Use leave-last-out, a temporal split, full-catalog ranking (not sampled), and report both Recall@K (did the true item make the top K) and MRR@K (how high did it rank). Recall is forgiving; MRR rewards putting the true item near the top, which is what users feel. Report both because a model can win Recall and lose MRR.

#### Worked example: computing MRR@20 for a next-item prediction by hand

MRR (Mean Reciprocal Rank) is the average of $1/\text{rank}$ of the true item, where rank is its position in the model's ranked list, and it is zero if the true item falls outside the top K. Let me work three test sessions with K = 20.

- **Session 1.** The held-out true next item is ranked **3rd** by the model. Reciprocal rank = $1/3 = 0.333$. It is within the top 20, so it also counts as a Recall@20 hit.
- **Session 2.** The true item is ranked **1st**. Reciprocal rank = $1/1 = 1.000$. A Recall@20 hit.
- **Session 3.** The true item is ranked **47th** — outside the top 20. Reciprocal rank for MRR@20 = $0$ (it fell outside K). Not a Recall@20 hit.

MRR@20 = $\frac{0.333 + 1.000 + 0.000}{3} = \frac{1.333}{3} = 0.444$.

Recall@20 = $\frac{2 \text{ hits}}{3 \text{ sessions}} = 0.667$.

Notice how the two metrics tell different stories. Recall@20 says "two of three times the true item was somewhere in the top 20." MRR@20 says "but on average it sat around rank 2 to 3 for the hits and missed entirely once." If a competing model moved session 1's true item from rank 3 to rank 1, Recall@20 would not change (still a hit either way), but MRR@20 would rise from 0.444 to $\frac{1.000 + 1.000 + 0}{3} = 0.667$. That is exactly why you report both: MRR is sensitive to improvements *within* the top K that Recall is blind to.

## 9. Results: how much does modeling order actually buy you

Here is the scoreboard the whole post has been building toward. The numbers below are **literature-consistent** approximate figures for a single e-commerce session dataset of the YooChoose/RetailRocket style (the exact values depend on preprocessing, session-length caps, and the train/test split; treat them as representative, not as a benchmark you should reproduce to three decimals). The point is the *shape* of the lift, which is robust across papers and datasets.

| Model | Uses order? | Long-range? | Recall@20 | MRR@20 | Notes |
|---|---|---|---|---|---|
| Popularity | no | no | ~0.205 | ~0.069 | the floor; ignores the session |
| Item-kNN | last item only | no | ~0.523 | ~0.196 | co-visitation; very strong for one trick |
| FPMC | last item only | no | ~0.621 | ~0.249 | factorized + personalized transition |
| GRU4Rec (TOP1) | full session | yes | ~0.665 | ~0.270 | RNN over the session |
| GRU4Rec (BPR-max) | full session | yes | ~0.682 | ~0.286 | top-k gains from weighted negatives |

The figure below renders the headline result — the climb in both Recall@20 and MRR@20 as you move from order-blind to order-aware models.

![A results matrix with rows for popularity item-kNN FPMC and GRU4Rec and columns for Recall at 20 MRR at 20 and whether order is modeled showing both metrics rising as order modeling improves](/imgs/blogs/sequential-and-session-based-recommendation-8.png)

Read the lifts honestly:

- **Popularity to item-kNN** is the biggest single jump (0.205 to 0.523 Recall@20). This is the "use the last click at all" lift, and it is huge. Most of the predictable signal is in the last step, and item-kNN captures it with pure counting. Lesson: a strong, cheap baseline closes most of the gap. Do not skip it.
- **Item-kNN to FPMC** (0.523 to 0.621) is the "learn and personalize the last-step transition" lift. Factorization generalizes to unseen and cold-start transitions; personalization adds the user's long-run taste. Real, but smaller than the first jump.
- **FPMC to GRU4Rec** (0.621 to ~0.68) is the "use the *whole session*, not just the last item" lift. This is the headroom that order-aware memory captures beyond the last step — the part item-kNN and FPMC structurally cannot reach. It is worth several points of Recall and a meaningful MRR bump, but it is the *smallest* of the three jumps. That is the honest, slightly humbling truth of sequential modeling: the last step dominates, and the marginal value of long-range memory, while real, is modest on many datasets.
- **TOP1 to BPR-max** (~0.665 to ~0.682) is the "focus gradients on hard negatives" lift — a loss-function improvement, not a model-architecture one. It is free in the sense that it costs no extra parameters or serving latency, which makes it an unusually good return.

#### Worked example: is the GRU4Rec lift worth the operational cost?

Suppose you run a mid-size shop with 8 million sessions a day and a session-rec rail that drives 12% of clicks. Your current model is FPMC at Recall@20 ~0.62. You are considering GRU4Rec (BPR-max) at ~0.68.

The relative Recall@20 lift is $\frac{0.682 - 0.621}{0.621} \approx 9.8\%$. Online lift is always smaller than offline lift (the offline-online gap this series keeps flagging), so discount it — say a third of the offline relative lift translates, giving roughly **3% more clicks on the rail**. If the rail drives 12% of total clicks and clicks convert at your normal rate, that is roughly $0.12 \times 0.03 \approx 0.36\%$ more total clicks site-wide from this one change. On a business doing, say, \$50M GMV a year through the affected surface, a 0.36% click lift that converts proportionally is on the order of \$180K of GMV — and that is before BPR-max's free extra lift. Against that you weigh: GRU inference is heavier than an FPMC dot product (you must run the GRU over the live session, adding maybe a few milliseconds of p99), you now maintain an RNN training pipeline, and you need the session-parallel batching infrastructure. For a shop at this scale the lift clears the cost. For a shop with 50K sessions a day and a rail driving 2% of clicks, the same percentages produce a few thousand dollars of lift against the same engineering burden — and the honest answer is **stay on FPMC or even item-kNN**. The lift from order is real but modest; whether it pays depends entirely on your scale and how much traffic the rail moves.

## 10. Case studies and named results

**FPMC (Rendle, Freudenthaler, Schmidt-Thieme, WWW 2010).** "Factorizing Personalized Markov Chains for Next-Basket Recommendation." Evaluated on Rossmann drugstore basket data, FPMC beat both standalone matrix factorization (which ignores the sequence) and a standalone factorized Markov chain (which ignores long-run taste). The paper's central empirical point — that *combining* the two terms beats either alone — is the reason the additive two-dot-product form is still the textbook way to think about blending preference and transition. It also established BPR-style pairwise training for sequential models.

**GRU4Rec (Hidasi et al., ICLR 2016 workshop; extended 2018).** "Session-Based Recommendations with Recurrent Neural Networks." On the RecSys 2015 Challenge data (the YooChoose e-commerce clickstream), the original GRU4Rec reported roughly a 20–30% relative improvement in Recall@20 and MRR@20 over the item-kNN baseline once the session-parallel training and TOP1 loss were in place — and a key honest detail from the literature is that *early* GRU4Rec configurations did **not** clearly beat a well-tuned item-kNN; it took the right loss and training tricks to pull ahead. The 2018 follow-up's BPR-max loss ("Recurrent Neural Networks with Top-k Gains") added a further notable Recall and MRR gain by focusing the gradient on hard negatives. This is the canonical "the loss mattered as much as the architecture" story in session rec.

**Caser (Tang and Wang, WSDM 2018).** "Personalized Top-N Sequential Recommendation via Convolutional Sequence Embedding." On MovieLens and Gowalla-style sequential data, Caser was competitive with or slightly ahead of GRU4Rec, and its horizontal/vertical filter design gave an interpretable story (union-level vs point-level patterns). It is the standard reference for "you can do sequential rec with CNNs, not just RNNs."

**Session rec at e-commerce and news scale.** Beyond academic benchmarks, session-based recommendation is the workhorse for logged-out traffic. E-commerce platforms run session models to power "customers who viewed this also viewed" and the live "continue shopping" rail; news aggregators (where the vast majority of readers are anonymous) rely almost entirely on session signals because there is no long user history to fall back on. The 2015 RecSys Challenge, built on YooChoose data, was the catalyst that turned session-based recommendation from a niche into a standard track — and the practical lesson the winning solutions taught was the same one this post has hammered: a strong co-visitation baseline plus a well-tuned sequence model, with honest temporal evaluation, beats a fancy model evaluated sloppily.

**The sampled-metrics warning (Krichene and Rendle, KDD 2020).** "On Sampled Metrics for Item Recommendation" is not a model but a methodology result that every sequential-rec practitioner should internalize. Many SASRec/BERT4Rec/GRU4Rec papers reported metrics by ranking the true item against 100 random negatives instead of the full catalog. The paper proved this is *inconsistent*: the model that looks best under sampled metrics can be worse under full evaluation, because sampling distorts which models the metric favors. If you compare sequential models, rank against the full catalog (the eval harness above does) or apply the paper's correction. This single methodological mistake has muddied a lot of the published sequential-rec leaderboard.

## 11. When sequence modeling pays off (and when it does not)

Sequential modeling is not a free upgrade. Here is the decisive version.

**Reach for sequential/session models when:**

- A large fraction of your traffic is **anonymous or cold** (logged-out e-commerce, news, fresh installs). You have no user vector, so the session is all you have, and a session model is the right tool, full stop.
- **Intent shifts fast within a visit** and the last few interactions genuinely predict the next better than the all-time average — fashion browsing, multi-step purchases (phone then accessories), content binges. Measure it: if Recall@20 from "last item only" is far above popularity, order has signal.
- You already have the **clickstream infrastructure** — event logging, sessionization, a model-serving path that can run a sequence encoder live. The data plumbing is most of the cost.

**Do not reach for it (or do not reach past the baseline) when:**

- **Item-kNN or FPMC already hits your target.** The biggest jump is "use the last click at all," and a co-visitation baseline captures most of it for almost no cost. If item-kNN gets you to the metric you need, shipping a GRU is gold-plating. Prove the marginal lift on a temporal split before you commit to the RNN pipeline.
- **Sessions are too short to have order.** If most sessions are 1–2 clicks, there is almost no sequence to model; the "next item" is essentially "the item co-visited with the one click you saw," which item-kNN nails. Sequence depth needs session length.
- **Your value is long-run taste, not the moment.** A subscription video service recommending a weekend movie to a five-year subscriber gets more from stable preference than from the last click. Use the sequence as a *feature* in a ranker, not as the whole model.
- **You cannot evaluate honestly.** If you only have random-split, sampled-negative evaluation, you cannot trust your offline numbers (per the KDD 2020 result), and you risk shipping a model that wins offline and flops online — the recurring failure mode of this whole field. Fix evaluation before chasing architecture.

The general rule that holds across this series: **the cheapest model that hits your target wins.** Sequential modeling is the right answer when the session genuinely carries signal your static model is throwing away — which, for the logged-out, fast-moving modern web, is often. But prove it with a temporal split against a strong baseline before you pay the operational tax.

## 12. Key takeaways

- **Reframe from preference to next-item.** Static user-by-item models score an all-time average; sequential models predict the next interaction from the ordered, recent session. Order and recency carry signal a bag-of-items model provably cannot capture.
- **Session-based (anonymous, short) and sequential (long per-user history) are different settings.** Session-based must work without a user vector; that constraint, not the architecture, often decides the design.
- **Beat the baselines first.** Popularity is the floor; item-kNN (co-visitation on the last click) is shockingly strong because most predictable signal lives in the last step. Many neural models that "won" had simply not been compared to a well-tuned item-kNN.
- **FPMC = matrix factorization + a factorized, personalized last-step Markov transition.** Factorizing the transition tensor beats the sparsity that kills count-based Markov chains and gives cold-start transfer. Its limit is *first-order*: only the last item.
- **GRU4Rec brought three durable ideas:** a GRU over the whole session (memory beyond the last item), session-parallel mini-batches (constant batch size across variable-length sessions, with free in-batch negatives), and ranking losses tailored to session rec (TOP1, then BPR-max, which focuses gradients on hard negatives for "top-k gains").
- **The full next-item softmax is too expensive; approximate it.** Pairwise/ranking losses see the order, not absolute scores, and that is exactly what top-K recommendation needs.
- **Caser is the CNN alternative** — horizontal filters for union-level patterns, vertical filters for point-level recency. Competitive with GRU4Rec; both were overtaken by self-attention (SASRec, BERT4Rec), which sees any past item with a learned weight, no recurrence bottleneck or fixed window.
- **Evaluate with leave-last-out, a temporal split, and full-catalog ranking.** Report Recall@K and MRR@K together. Sampled-negative metrics are inconsistent (KDD 2020) and can reverse model rankings — do not trust them for comparisons.
- **The lift from order is real but front-loaded.** "Use the last click at all" is the biggest jump; "use the whole session" is the smallest of the three. Whether the GRU's marginal lift pays depends on your scale and how much traffic the rail moves.
- **Masking, recency bias, and repeat consumption are the practical traps.** Wrong masking silently caps your metrics; recency greed chases mis-clicks; high-repeat domains need a repeat-or-explore gate measured separately.

## Further reading

- Rendle, Freudenthaler, Schmidt-Thieme, "Factorizing Personalized Markov Chains for Next-Basket Recommendation," WWW 2010 — the FPMC paper; the additive preference-plus-transition form.
- Hidasi, Karatzoglou, Baltrunas, Tikk, "Session-Based Recommendations with Recurrent Neural Networks," ICLR 2016 workshop — GRU4Rec, session-parallel mini-batches, the TOP1 loss.
- Hidasi and Karatzoglou, "Recurrent Neural Networks with Top-k Gains for Session-based Recommendations," CIKM 2018 — BPR-max and the top-k gains over TOP1.
- Tang and Wang, "Personalized Top-N Sequential Recommendation via Convolutional Sequence Embedding," WSDM 2018 — Caser, horizontal and vertical convolutions.
- Krichene and Rendle, "On Sampled Metrics for Item Recommendation," KDD 2020 — why sampled-negative evaluation is inconsistent; evaluate on the full catalog.
- PyTorch `nn.GRU` and `nn.Embedding` documentation — the building blocks used in the implementation above.
- Within this series: [self-attention for sequences: SASRec and BERT4Rec](/blog/machine-learning/recommendation-systems/self-attention-for-sequences-sasrec-bert4rec) for the attention models that took the crown; [implicit feedback models: ALS and BPR](/blog/machine-learning/recommendation-systems/implicit-feedback-models-als-and-bpr) for the pairwise ranking loss FPMC and GRU4Rec build on; [the ranking model: CTR prediction foundations](/blog/machine-learning/recommendation-systems/the-ranking-model-ctr-prediction-foundations) for where the next-item candidates get scored downstream; [what is a recommender system](/blog/machine-learning/recommendation-systems/what-is-a-recommender-system) for the retrieval-ranking-reranking map; and the capstone [the recommender systems playbook](/blog/machine-learning/recommendation-systems/the-recommender-systems-playbook).
