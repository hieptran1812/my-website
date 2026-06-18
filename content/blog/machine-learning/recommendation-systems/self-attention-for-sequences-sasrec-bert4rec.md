---
title: "Self-Attention for Sequences: SASRec and BERT4Rec"
date: "2026-06-19"
publishDate: "2026-06-19"
description: "Build the transformer backbone of sequential recommendation from first principles: derive why self-attention beats RNNs for item sequences, implement a compact causal SASRec in PyTorch, contrast it with BERT4Rec's bidirectional cloze objective, and measure Recall@10 and NDCG@10 against GRU4Rec and Caser on MovieLens."
tags:
  [
    "recommendation-systems",
    "recsys",
    "sasrec",
    "bert4rec",
    "self-attention",
    "transformers",
    "sequential-recommendation",
    "machine-learning",
    "pytorch",
    "movielens",
  ]
category: "machine-learning"
subcategory: "Recommendation Systems"
author: "Hiep Tran"
featured: true
readTime: 55
image: "/imgs/blogs/self-attention-for-sequences-sasrec-bert4rec-1.png"
---

I was once handed a sequential recommender that worked, technically, and was somehow always wrong. It was a GRU trained on session click streams, and on a quiet afternoon it would happily predict your next click. But the moment a user did something interesting at the start of a long session — searched for a stand mixer, browsed three of them, then wandered off into spatulas and mixing bowls for twenty more clicks — the model forgot the mixer. By the end of the session the only signal it could see was "this person likes kitchen stuff," which is the recommendation equivalent of a shrug. The mixer, the actual intent, the thing that should have anchored every subsequent suggestion, had been squeezed out of a single hidden vector forty clicks ago. The recurrence had a memory, and the memory had a budget, and the budget was one fixed-size state that the most recent items always won.

The fix was not a bigger GRU. It was a different shape of computation entirely. Instead of forcing the whole history through one hidden state that gets overwritten step by step, you let the model look back at *every* past item directly, and learn, per prediction, how much weight to put on each one. When the user is staring at a checkout page, the model can reach all the way back to the mixer and say "that, forty clicks ago, is what this session is about." That mechanism is self-attention, and it is the reason the two models in this post — SASRec and BERT4Rec — quietly took over sequential recommendation between 2018 and 2020 and have stayed there. They are the transformer backbone of the recommender funnel's most behaviorally rich stage: predicting what a user does *next*, given the ordered trail of what they did before.

![Diagram of a SASRec block where item embeddings and learned position embeddings are summed, passed through a causally masked self-attention layer and a feed-forward network, and turned into next-item logits by a dot product with the shared item embedding table](/imgs/blogs/self-attention-for-sequences-sasrec-bert4rec-1.png)

This is a post in the series **Recommendation Systems: From Click to Production**, and it sits squarely in the retrieval-and-ranking funnel laid out in [what is a recommender system](/blog/machine-learning/recommendation-systems/what-is-a-recommender-system). The funnel's spine is retrieval to ranking to re-ranking, fed by the serve-log-train feedback loop and read off the offline-versus-online gap. Sequential models are a special, powerful kind of retrieval-and-scoring model: instead of a static "what does this user like on average," they answer "what will this user do *next*," which is the question that drives session-based feeds, autoplay, and "because you just watched" rails. We build on the motivation from [sequential and session-based recommendation](/blog/machine-learning/recommendation-systems/sequential-and-session-based-recommendation) and go deep on the architecture that won. By the end you will be able to explain *why* attention beats an RNN for a sequence, derive the scaled dot-product attention computation and the causal mask, implement a compact SASRec in PyTorch with a sampled binary loss and a full cross-entropy variant, sketch BERT4Rec's cloze training, evaluate Recall@10 and NDCG@10 with a leave-last-out split, and reason honestly about when the bidirectional model's extra expressiveness is worth its train-serve mismatch.

## 1. The problem: predict the next item from an ordered history

Let me pin down the task before any architecture, because the architecture only makes sense once the task is precise.

A user $u$ has produced an ordered sequence of interactions $S_u = (v_1, v_2, \ldots, v_t)$, where each $v_k$ is an item id (a movie, a product, a song). The interactions are in time order: $v_1$ happened first, $v_t$ happened most recently. The job of a sequential recommender is, given the prefix $(v_1, \ldots, v_t)$, to predict the next item $v_{t+1}$ — or, more usefully for serving, to produce a score for *every* item in the catalog so we can rank them and return the top-K.

This is different from the static collaborative-filtering view in [matrix factorization](/blog/machine-learning/recommendation-systems/matrix-factorization-the-workhorse), where a user is one fixed vector summarizing all-time taste. A static model would represent the kitchen-shopper above as "likes kitchen items" and stop there. A sequential model represents the user as *the trajectory itself*, and lets the representation change as the trajectory grows. Order matters: a user who watched the first three episodes of a show in order is in a very different state than one who watched episode three, then one, then seven. Recency matters: what you did five minutes ago is usually a stronger signal than what you did five months ago, but not always — the long-ago mixer can dominate if the recent clicks are noise. A good sequential model learns *which* past items matter for *this* prediction, rather than hard-coding "recent is best."

Three families of architecture have been used to consume that ordered history:

- **Markov chains and Caser-style CNNs** model short, local patterns: the next item depends mostly on the last one or two (a first-order Markov chain) or on a small window of recent items convolved by learned filters (Caser, "Convolutional Sequence Embedding Recommendation," Tang and Wang 2018). Cheap and parallel, but their reach is the window size.
- **Recurrent networks (GRU4Rec)** model the whole history through a recurrent hidden state updated one step at a time (Hidasi et al., "Session-based Recommendations with Recurrent Neural Networks," 2016). In principle unbounded reach; in practice a serial bottleneck.
- **Self-attention (SASRec, BERT4Rec)** lets every position look directly at every other allowed position, weighting them adaptively. Parallel to train, direct long-range reach, learned per-step weighting.

The rest of this post is the case for the third family, the math that makes it work, and the two canonical instances.

One more framing point before the architecture, because it determines where this model lives in your system. There are two distinct ways to use a next-item predictor. The first is **session-based recommendation**, where you have no persistent user id (an anonymous visitor, a logged-out session) and the *only* signal is the current session's clicks; the sequence model must work entirely from the trail in front of it. The second is **sequential recommendation for known users**, where you have a long history and the sequence model refines a persistent profile with recent intent. SASRec and BERT4Rec serve both — the architecture does not care whether the sequence is one session or a year of history — but the data preparation and the evaluation differ. Session-based setups truncate hard to the current session and value recency heavily; known-user setups keep a longer window and let attention balance recent intent against long-term taste. Knowing which regime you are in tells you how to set `max_len`, how to split for evaluation, and whether cold start is a session-level or user-level problem.

## 2. Why attention beats the RNN and the CNN

There are three concrete reasons self-attention took over, and each one maps to a real engineering pain you feel with the alternatives.

**Reason one: parallel training.** A GRU computes hidden state $h_k$ from $h_{k-1}$ and input $v_k$. You cannot compute $h_5$ until you have $h_4$, which needs $h_3$, and so on. Training a recurrent model over a sequence of length $n$ is inherently $O(n)$ *sequential* steps — the GPU sits mostly idle waiting for the recurrence to unroll. Self-attention computes all positions' representations in one matrix multiply: there is no step that must wait for a previous step, so the whole sequence is processed in parallel. On modern accelerators this is the difference between a model that trains in an afternoon and one that trains overnight, for the same data.

It is worth being precise about *why* this matters on a GPU specifically. A GPU is a throughput machine: it has tens of thousands of arithmetic units and is happiest when you hand it one enormous matrix multiply that keeps all of them busy. A recurrence does the opposite — it hands the GPU a tiny matrix-vector product, waits for it to finish, then hands it the next one. The arithmetic per step is trivial, so the GPU spends almost all its time on kernel-launch overhead and memory latency rather than on useful math. Measured utilization on an RNN training loop is often single-digit percent. Self-attention turns the whole sequence into one $n \times n$ score matrix and one $n \times d$ value aggregation, both dense matrix multiplies that saturate the hardware. The original SASRec paper measured an order-of-magnitude wall-clock training speedup over GRU4Rec on the same GPU for this reason, not because the attention does less arithmetic — it does *more* (the $n^2$ term) — but because the arithmetic it does is the kind a GPU can actually run fast.

**Reason two: direct long-range access.** In an RNN, information from $v_1$ reaches the prediction at step $n$ only by surviving $n-1$ overwrites of the hidden state. Each step mixes new input into a fixed-size vector, and old signal decays — the vanishing-gradient and information-bottleneck problems that gated units like the GRU mitigate but do not eliminate. Self-attention gives position $n$ a direct edge to position $1$: a single learned weight, no decay, no intermediate squeezing. The mixer forty clicks ago is one attention weight away, not forty overwrites away.

There is a clean way to quantify this. Define the *path length* between two positions as the number of sequential operations a signal must traverse to get from one to the other. In an RNN that path length is $O(n)$ — the signal walks the recurrence one step at a time. In a CNN with kernel width $w$ it is $O(n/w)$ for a single layer, or $O(\log_w n)$ if you stack dilated convolutions, still growing with distance. In self-attention the path length between *any* two positions is $O(1)$ — one hop, through one attention edge. Shorter paths mean gradients flow more directly during training and signal does not decay during the forward pass. This is the same maximum-path-length argument the transformer paper made for language, and it transfers cleanly to item sequences: the further back the relevant item, the more decisively attention wins over recurrence.

The information-bottleneck framing makes the same point from the forward direction. An RNN compresses the entire prefix $(v_1, \ldots, v_k)$ into a single $d$-dimensional vector $h_k$ before the next prediction can use it. That vector is a lossy summary, and its capacity is fixed regardless of how long the prefix is — a length-5 history and a length-500 history both get squeezed into the same $d$ numbers. Self-attention never compresses the prefix into one vector; it keeps a separate state per position and lets the query decide, at prediction time, which states to read. There is no fixed-capacity summary to overflow. The cost is that you must store and attend over all $n$ states, the $O(n^2)$ price, which is exactly the trade: the RNN saves memory by compressing and pays in lost signal, while attention spends memory to keep everything and pays in quadratic compute.

![Before and after comparison contrasting a GRU funneling the whole history through one serial hidden state with self-attention reaching any past item directly in one parallel pass](/imgs/blogs/self-attention-for-sequences-sasrec-bert4rec-6.png)

**Reason three: adaptive per-step weighting.** A CNN convolves a fixed filter over a fixed window — the same pattern detector everywhere, blind to anything outside the window. An RNN has a fixed update rule. Self-attention computes, for each query position, a *fresh* probability distribution over the past, conditioned on the content of the items. When the query is a checkout page, attention can concentrate on the mixer; when the query is a casual browse, it can spread weight evenly. The weighting is data-dependent and recomputed for every prediction, which is exactly the flexibility the kitchen-shopper example demanded.

The figure above contrasts the two regimes. On the left, the GRU's single hidden state is a literal bottleneck: every bit of history must fit through it, and the most recent writes dominate. On the right, attention keeps all the per-position states and lets the query pull from any of them with a learned weight. The price, as we will see in Section 7, is that attention costs $O(n^2)$ in the sequence length — every position attends to every other — whereas the RNN is $O(n)$. For the sequence lengths typical in recommendation (tens to a couple hundred items), the quadratic cost is cheap and the modeling win is large. That trade flips only at very long sequences, which is where lifelong-sequence models and attention approximations come in, a topic for another post.

#### Worked example: when the RNN forgets and attention does not

Take the kitchen session: positions 1 through 4 are mixer-related (search "stand mixer," view mixer A, view mixer B, view mixer reviews), then positions 5 through 24 are low-intent browsing (spatulas, bowls, a recipe book, more spatulas). We want to predict position 25, where the user is on a "complete your kitchen" page.

A GRU's hidden state $h_{24}$ is a fixed-size vector that has been overwritten 24 times. The last 20 updates were all low-intent kitchen items, so $h_{24}$ encodes "generic kitchen browsing." The mixer signal, written into $h_4$, has been diluted across 20 subsequent updates. Empirically, recurrent recommenders show a recency bias precisely because of this: the effective context window is short even though the architecture is nominally unbounded.

A self-attention model computes, for the query at position 25 (the last position, which is what we predict from), an attention weight over all 24 past positions. If the mixer items carry a distinctive learned signature and the browsing items are noise, the softmax can place, say, 0.45 of its weight on positions 1 through 4 and spread the remaining 0.55 thinly over positions 5 through 24. The output representation is then dominated by the mixer, and the next-item prediction is "a stand mixer accessory" rather than "another spatula." The model did not need a longer memory; it needed the ability to *reach back and choose*. That is the whole pitch.

## 3. The science: scaled dot-product self-attention on a sequence

Now the math, because the "look back and weight" intuition has to become a precise computation. I will assume you have met multi-head attention before; if you want the full derivation of the mechanism itself, the transformer attention machinery is covered in the large-language-model track of this blog and in the original "Attention Is All You Need" (Vaswani et al., 2017). Here I focus on what changes when the tokens are *items in a user's history* rather than words in a sentence.

We have a sequence of item embeddings stacked into a matrix $X \in \mathbb{R}^{n \times d}$, one row per position, $d$ the embedding dimension. Self-attention turns $X$ into a new matrix of the same shape, where each output row is a learned weighted average of the input rows. The mechanism projects $X$ three ways:

$$Q = X W^Q, \qquad K = X W^K, \qquad V = X W^V,$$

where $W^Q, W^K, W^V \in \mathbb{R}^{d \times d}$ are learned. $Q$ holds the *queries* (what each position is looking for), $K$ the *keys* (what each position offers), $V$ the *values* (the content each position contributes). The attention output is

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^\top}{\sqrt{d}}\right) V.$$

Read it inside out. $Q K^\top \in \mathbb{R}^{n \times n}$ is the matrix of all pairwise dot products: entry $(i, j)$ is how much query $i$ matches key $j$. Divide by $\sqrt{d}$ so the dot products do not blow up as the dimension grows (a $d$-dimensional dot product of unit-variance vectors has variance $d$, so dividing by $\sqrt{d}$ restores unit scale and keeps the softmax out of its saturated, near-zero-gradient regime). The row-wise softmax turns each row into a probability distribution: row $i$ is "how much position $i$ attends to each position $j$." Multiplying by $V$ replaces each position's representation with the attention-weighted sum of all value vectors. Output row $i$ is

$$z_i = \sum_{j} \alpha_{ij}\, v_j, \qquad \alpha_{ij} = \frac{\exp(q_i^\top k_j / \sqrt{d})}{\sum_{j'} \exp(q_i^\top k_{j'} / \sqrt{d})}.$$

That $\alpha_{ij}$ is the "look back and weight" coefficient from the intuition, made exact. It is a normalized similarity between what position $i$ wants and what position $j$ offers. For our kitchen example, $i$ is the last position and the $\alpha_{ij}$ for the mixer positions are large.

**Multi-head attention** runs $h$ of these in parallel on lower-dimensional projections (each head gets $d/h$ dimensions), concatenates the outputs, and projects back to $d$. Concretely, head $m$ has its own projections $W^Q_m, W^K_m, W^V_m \in \mathbb{R}^{d \times d/h}$, computes $\text{head}_m = \text{Attention}(X W^Q_m, X W^K_m, X W^V_m)$, and the layer output is $\text{Concat}(\text{head}_1, \ldots, \text{head}_h) W^O$ with $W^O \in \mathbb{R}^{d \times d}$. Why split into heads at all instead of one big attention over the full $d$ dimensions? Because a single softmax can only put its mass in one place — a single head is forced to commit to one weighting of the past per query. Multiple heads let the model attend to several things at once: head 1 can lock onto the most recent item while head 2 simultaneously reaches back to the long-ago anchor intent, and the concatenation fuses both views. Each head is a separate, lower-rank "lens" on the same sequence.

Different heads can specialize — one head might track "same category as the most recent item," another "the long-ago anchor intent," a third "the typical next step after this kind of item." For recommendation, SASRec uses a small number of heads (the original paper used one or two; ML-1M did fine with one), because item sequences are shorter and lower-rank than language and do not need eight heads of redundancy. This is a recurring difference from NLP worth internalizing: a sentence has rich syntactic structure that benefits from many specialized heads, but an item sequence is closer to a noisy set of preferences with a temporal trend, so a couple of heads usually capture all the structure there is. Adding more heads on a small recommendation dataset mostly adds parameters to overfit with.

There is also a position-wise **feed-forward network** after the attention sublayer, and it is easy to dismiss as plumbing — but it does real work. Attention only ever produces *weighted averages* of value vectors; it is a linear mixing operation (the nonlinearity is in the softmax over weights, not in the combination of values). Without a per-position nonlinearity, stacking attention layers would collapse toward a single linear map. The feed-forward network, $\text{FFN}(z) = W_2\,\text{ReLU}(W_1 z + b_1) + b_2$ applied independently at each position, injects the per-position nonlinear transformation that lets the model compute features of each position's mixed representation rather than just re-mixing them. Attention decides *what to look at*; the feed-forward layer decides *what to make of it*.

### 3.1 Position embeddings: attention is order-blind without them

There is a subtlety that bites everyone the first time. The attention computation above is *permutation-equivariant*: if you shuffle the rows of $X$, the rows of the output shuffle the same way, but the *content* of each output row is unchanged, because $\alpha_{ij}$ depends only on the dot product $q_i^\top k_j$, not on the indices $i$ and $j$. Put plainly, raw self-attention is a set operation — it has no idea that $v_1$ came before $v_2$. For a sequence model, that is fatal: order is the whole point.

The fix is a **position embedding**. SASRec adds a learned vector $p_k \in \mathbb{R}^d$ to the item embedding at position $k$, so the input to the first attention layer is $\hat{x}_k = e_{v_k} + p_k$, where $e_{v_k}$ is the embedding of the item at position $k$ and $p_k$ is the embedding of the position itself. Now position 1 and position 24 have different inputs even for the same item, and the dot products $q_i^\top k_j$ can encode "how far apart are these two positions." SASRec uses a *learned* position table of size (max sequence length $\times d$), which is simple and works well for the fixed, modest lengths in recommendation. The number of position slots equals the truncation length (Section 6); positions beyond it are dropped.

A design question worth pausing on: should position be encoded as *absolute* (this is slot 7 of the sequence) or *relative* (this item is 3 steps before the query)? SASRec uses absolute learned positions, which is the simplest and is what I would start with. Relative position encodings — where the attention score between positions $i$ and $j$ depends on their offset $i - j$ rather than their absolute indices — can be more natural for recommendation, because "the item immediately before the query" should mean the same thing whether the session is 10 or 100 items long. Some follow-up sequence-rec work adopts relative or time-aware positions (encoding the actual *time gap* between interactions, not just their order, so a click five seconds ago is treated differently from one five days ago). The honest guidance: absolute learned positions are a strong, simple baseline; reach for relative or time-aware encodings only after you have confirmed your sequences have meaningful variable lengths or irregular time gaps that the absolute scheme is mishandling. Adding encoding complexity before you have a measured reason to is a common way to overfit a small dataset.

The figure at the top of the post shows this assembly: item embeddings and position embeddings are summed (with dropout), then fed into the causal self-attention layer, then a feed-forward network, and finally a dot product with the shared item table produces next-item logits. That summed input is what makes the rest of the stack order-aware.

#### Worked example: reading an attention weight distribution

Let me make the "look back and choose" mechanism fully concrete with numbers, because the attention distribution is the single most interpretable object in the model. Take a five-item history with the last item as the query position (position 5, from which we predict the next item). Suppose the scaled dot products $q_5^\top k_j / \sqrt{d}$ for $j = 1, \ldots, 5$ come out as the logit vector $(2.3, 0.1, 0.0, 0.4, 1.1)$, where position 1 is the anchor item (the mixer), positions 2 through 4 are low-signal browsing, and position 5 is the recent item. To get the attention weights, exponentiate and normalize:

$$e^{2.3} = 9.97,\quad e^{0.1} = 1.11,\quad e^{0.0} = 1.00,\quad e^{0.4} = 1.49,\quad e^{1.1} = 3.00.$$

The sum is $9.97 + 1.11 + 1.00 + 1.49 + 3.00 = 16.57$, so the attention weights are

$$\alpha_5 = (0.60,\ 0.07,\ 0.06,\ 0.09,\ 0.18).$$

Read that distribution. The model puts 60% of its weight on the anchor item at position 1 and only 18% on the most recent item — it has *learned* that the mixer, four steps back, is what this session is about, and it is reaching past the recent noise to attend to it. The output state $z_5 = 0.60\, v_1 + 0.07\, v_2 + 0.06\, v_3 + 0.09\, v_4 + 0.18\, v_5$ is dominated by the mixer's value vector, so the next-item logits will favor mixer accessories. A GRU could not produce this: its hidden state is a fixed recency-biased blend it cannot re-weight at query time.

Now flip one number to see the adaptivity. If the recent item were a strong intent signal — say position 5's logit were $3.5$ instead of $1.1$ — then $e^{3.5} = 33.1$, the sum becomes $46.7$, and the weights shift to roughly $(0.21,\ 0.02,\ 0.02,\ 0.03,\ 0.71)$: now 71% on the recent item. *Same model, same parameters*, a completely different weighting because the query content changed. That is the per-step adaptivity the architecture buys you, and it is why SASRec's authors observed attention concentrating on recent items for sparse data and spreading to long-range items for dense data — the model discovers the right reach from the data rather than having it hard-coded.

## 4. SASRec: causal self-attention, trained autoregressively

SASRec — "Self-Attentive Sequential Recommendation," Kang and McAuley, 2018 — is the cleaner of the two models, and the better default. Its core idea: use self-attention but make it **causal**, so position $i$ can attend only to positions at or before $i$. Then train it to predict the next item at every position, autoregressively, exactly the way a left-to-right language model is trained.

### 4.1 The causal mask

If we let every position attend to every other, position $i$ would attend to position $i+1$ — it would see the answer it is supposed to predict. That is label leakage: the model learns the trivial "the next item is the next item" rather than anything useful. The causal mask forbids it. Before the softmax, we set the masked entries of $Q K^\top$ to $-\infty$ (in practice a large negative number), so that after softmax their attention weights are exactly zero:

$$\alpha_{ij} = 0 \quad \text{for all } j > i.$$

Concretely, the score matrix $S = Q K^\top / \sqrt{d}$ is modified by adding a mask $M$ where $M_{ij} = 0$ for $j \le i$ and $M_{ij} = -\infty$ for $j > i$, then softmax is applied row-wise to $S + M$. The result is a *lower-triangular* attention pattern: row $i$ has nonzero weight only in columns $1$ through $i$.

![Grid showing a three by three causal attention mask where query position i may attend only to key positions at or before it, so the lower triangle is allowed and the upper triangle is masked to minus infinity](/imgs/blogs/self-attention-for-sequences-sasrec-bert4rec-4.png)

The grid above is the mask for a length-3 sequence: row $i$ is the query, column $j$ is the key, green cells are allowed, red cells are masked. Position 1 can attend only to itself. Position 2 can attend to positions 1 and 2. Position 3 can attend to all three. The output at position $i$ is therefore a representation of the prefix $(v_1, \ldots, v_i)$ — a summary of everything up to and including $v_i$, and nothing after — which is exactly the context you want for predicting $v_{i+1}$.

#### Worked example: tracing the causal mask for a length-4 sequence

Take the session $(v_1, v_2, v_3, v_4) = (\text{mixer}, \text{bowl}, \text{spatula}, \text{whisk})$ and walk the mask. The raw score matrix $S = QK^\top/\sqrt{d}$ is $4 \times 4$; the mask zeroes the upper triangle. After masking and softmax, the attention rows are:

- Row 1 (query = mixer): can see only key 1. Softmax over one element is 1.0, so $z_1$ is just the mixer's value vector. The prefix is "mixer," and that is all position 1 knows.
- Row 2 (query = bowl): can see keys 1 and 2. Suppose the masked scores are $(0.8, 1.2)$ for (mixer, bowl). Softmax gives roughly $(0.40, 0.60)$, so $z_2 = 0.40\, v_{\text{mixer}} + 0.60\, v_{\text{bowl}}$ — a blend that still remembers the mixer.
- Row 3 (query = spatula): can see keys 1, 2, 3. Scores $(0.5, 0.7, 1.5)$ give softmax about $(0.22, 0.27, 0.51)$, so $z_3$ leans on the spatula but keeps the mixer alive at 0.22.
- Row 4 (query = whisk): can see all four keys. This is the position we predict $v_5$ from.

The training target at each position is the *next* item: position 1 should predict $v_2$ (bowl), position 2 should predict $v_3$ (spatula), position 3 should predict $v_4$ (whisk), and position 4 should predict $v_5$ (the held-out next item). One forward pass over the length-4 sequence yields four supervised predictions, all in parallel, none leaking the future. That is the efficiency of causal training: $n$ training signals per sequence, computed simultaneously, with the mask guaranteeing each prediction sees only its legitimate past. Contrast the RNN, which would also give $n$ predictions but must compute them in $n$ serial steps.

### 4.2 The architecture stack

SASRec stacks $b$ identical self-attention blocks. Each block is a causal self-attention layer followed by a position-wise feed-forward network (two linear layers with a ReLU between them, applied independently at each position), each wrapped with a residual connection, layer normalization, and dropout — the standard transformer block, applied to items.

![Stack diagram of the SASRec layers from the embedding layer through two self-attention plus feed-forward blocks to a prediction head that scores items by dot product with the tied embedding table](/imgs/blogs/self-attention-for-sequences-sasrec-bert4rec-5.png)

The stack above reads bottom to top: the embedding layer produces item-plus-position vectors; each block refines them with self-attention and a feed-forward transform while preserving the residual path; layer normalization stabilizes the final states; and the prediction head turns the top state at each position into next-item logits. SASRec keeps the stack shallow — two blocks is the canonical setting, sometimes one — because deeper stacks overfit on the small interaction datasets without careful regularization. The depth-versus-data trade-off is real: language models go 12 to 96 layers deep because they have billions of tokens; a sequential recommender on MovieLens-1M has about a million interactions and saturates at two blocks.

### 4.3 The prediction head and the loss

At the top of the stack, the state $z_t^{(b)}$ at position $t$ (the output of the final block) is the model's summary of the prefix $(v_1, \ldots, v_t)$. To score a candidate next item $i$, SASRec takes the dot product of this state with the item's embedding from the *shared* embedding table — the same table used at the input. Tying the input and output embeddings cuts parameters and tends to help. The relevance score of item $i$ at position $t$ is

$$r_{t, i} = z_t^{(b)\top} e_i.$$

The loss is binary cross-entropy over a positive and sampled negatives. At each position $t$ the positive is the true next item $o_t = v_{t+1}$; a negative $o_t^-$ is a random item the user did not interact with at that step. A subtle but important detail: the negative must be sampled fresh per training step and excluded from the user's known items, otherwise you penalize an item the user *did* like and inject label noise. SASRec minimizes

$$\mathcal{L} = -\sum_{u}\sum_{t} \Big[ \log \sigma(r_{t, o_t}) + \log\big(1 - \sigma(r_{t, o_t^-})\big) \Big],$$

where $\sigma$ is the sigmoid. The first term pushes the score of the true next item up; the second pushes a sampled negative down. This is a sampled, pointwise-on-pairs objective: cheap, because you score one positive and one (or a few) negatives per position rather than the whole catalog. It is closely related to the pairwise BPR loss from [implicit feedback models](/blog/machine-learning/recommendation-systems/implicit-feedback-models-als-and-bpr) in spirit — both ask the model to rank the observed item above an unobserved one — and to the sampled-softmax objective from [training the two-tower retrieval model](/blog/machine-learning/recommendation-systems/training-two-tower-negatives-and-sampled-softmax). A common modern variant replaces the binary loss with a full softmax cross-entropy over the entire item vocabulary, which removes the sampling noise at the cost of an $O(|\mathcal{V}|)$ output layer; on catalogs up to a few hundred thousand items this is affordable and often improves NDCG. We will implement both.

## 5. BERT4Rec: bidirectional attention with a cloze objective

BERT4Rec — "BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer," Sun et al., 2019 — asks a sharp question: why restrict each position to its left context? When you are *modeling* a user's taste (not strictly generating the next click), an item's meaning depends on what comes after it too. The third episode of a show is understood differently if episodes four and five follow. So BERT4Rec drops the causal mask and lets every position attend to every other — **bidirectional** self-attention — borrowing the encoder design from BERT in NLP.

![Before and after comparison contrasting SASRec reading left to right and predicting the next item with BERT4Rec masking items inside the sequence and predicting them from both sides](/imgs/blogs/self-attention-for-sequences-sasrec-bert4rec-2.png)

But you cannot train a bidirectional model with the next-item objective, because if position $i$ can see position $i+1$, the next-item target leaks immediately — every position trivially "predicts" the item to its right by just reading it. BERT4Rec solves this with the **cloze task** (also called masked language modeling in NLP): randomly replace a fraction of items in the sequence with a special `[mask]` token, and train the model to predict the original items at the masked positions, using context from *both* sides. The figure above contrasts the two objectives: SASRec's left-to-right next-item prediction versus BERT4Rec's mask-and-fill-the-blanks.

### 5.1 The cloze objective

Formally, for a sequence $S_u$, BERT4Rec samples a mask set, replaces those positions with `[mask]`, and minimizes the negative log-likelihood of the true items at the masked positions:

$$\mathcal{L} = \frac{1}{|S_u^m|} \sum_{v_m \in S_u^m} -\log P\big(v_m = v_m^* \,\big|\, S_u'\big),$$

where $S_u^m$ is the set of masked positions, $S_u'$ is the sequence with masks inserted, $v_m^*$ is the true item that was masked, and $P$ is a softmax over the item vocabulary computed from the bidirectional encoder's output at the masked position. The masking proportion is a hyperparameter (the paper used values around 0.2 to 0.6 depending on dataset; denser sequences tolerate more masking).

The payoff is **richer context**: when predicting a masked item, the model sees both the items before it and the items after it, which is strictly more information than the left context alone. The cloze task also generates *many* training signals per sequence — every masked position is a supervised target — and it forces the model to build representations that are robust to missing items, a useful inductive bias.

### 5.2 The train-serve mismatch

Here is the catch that you must understand before reaching for BERT4Rec. The serving task is *next-item* prediction: given the full history, what comes next? But the model was never trained on "predict the item *after* the last one" — it was trained on "fill in a blank *somewhere inside* the sequence." There is no blank at the end during training. At inference, BERT4Rec manufactures one: it **appends a `[mask]` token to the end** of the user's history and reads the prediction at that appended position.

![Before and after comparison contrasting BERT4Rec training by masking items inside the sequence with serving by appending a mask token at the end, a train-serve mismatch](/imgs/blogs/self-attention-for-sequences-sasrec-bert4rec-7.png)

The figure above shows the gap. During training (left), masks land at random interior positions with two-sided context. During serving (right), the mask is at the very end, where the model has only left context — a configuration it rarely saw in training, since interior masks usually have items on both sides. The original paper mitigates this by *also* masking the last item with some probability during training, so the model gets practice at the serving configuration. Even so, the train-serve distribution mismatch is the model's structural weakness, and it is the seed of the replication debate we cover in Section 9. The takeaway: BERT4Rec buys bidirectional context, but it pays in a serving configuration that does not match its training distribution, and that mismatch must be managed, not ignored.

### 5.3 Sketching the cloze training

The architectural change from SASRec to BERT4Rec is small in code — drop the causal mask, add a mask token, and change how targets are constructed. Here is the cloze masking and loss, reusing the same encoder shape but with bidirectional attention (no `attn_mask`). The mask token gets its own embedding id, which is why we reserved room for it.

```python
import torch
import torch.nn.functional as F

MASK_ID = None  # set to n_items + 1; give the mask its own embedding row

def cloze_batch(seq, n_items, mask_id, mask_prob=0.2):
    # seq: (B, L) padded item ids; returns masked input and per-position targets.
    masked = seq.clone()
    labels = torch.full_like(seq, fill_value=-100)  # -100 = ignored by cross_entropy
    is_item = seq != 0
    draw = torch.rand_like(seq, dtype=torch.float)
    chosen = is_item & (draw < mask_prob)           # positions to mask
    labels[chosen] = seq[chosen]                    # predict the original item there
    masked[chosen] = mask_id                        # replace input with the mask token
    return masked, labels

def cloze_loss(model, seq, n_items, mask_id, mask_prob=0.2):
    masked, labels = cloze_batch(seq, n_items, mask_id, mask_prob)
    # BERT4Rec encoder: SAME stack as SASRec but NO causal mask (bidirectional).
    states = model.seq_encode_bidirectional(masked)        # (B, L, d)
    logits = states @ model.item_emb.weight.t()            # (B, L, n_items + 2)
    return F.cross_entropy(logits.view(-1, logits.size(-1)),
                           labels.view(-1), ignore_index=-100)
```

Two differences from SASRec to notice. First, the targets are *only at masked positions* (everything else is `-100`, ignored), whereas SASRec supervises *every* position with the next item — so per sequence BERT4Rec produces fewer training signals at a given step, which is part of why it needs more epochs to see the same number of supervised predictions. Second, `seq_encode_bidirectional` is the same block stack as `SASRec.seq_encode` but with the `attn_mask` removed so every position attends both ways; you would still keep the `key_padding_mask` so positions ignore padding. At inference you call it on the history with a mask appended at the end and read the prediction at that final position — the serving trick from Section 5.2 made literal. The smallness of this diff is exactly why the two models are so directly comparable, and why the replication study could isolate that the *objective and training budget*, not the architecture, drive most of the reported difference.

## 6. The practical flow: a compact SASRec in PyTorch

Enough theory. Here is a SASRec you can train. I will build it in pieces: the data and the leave-last-out split, the model, the loss, the training loop, and the evaluation harness. The code is real and idiomatic; adapt the hyperparameters to your dataset.

### 6.1 Data: sequences and the leave-last-out split

We use MovieLens-1M (about 1 million ratings, 6,040 users, 3,706 movies), the standard SASRec benchmark. We treat any rating as an implicit positive (the user interacted with the movie), sort each user's interactions by timestamp, and build one sequence per user. The evaluation protocol is **leave-last-out**: for each user, the last item is the test target, the second-to-last is the validation target, and everything before is training context. This is a temporal split per user — no future leaks into the past — which is the honest way to evaluate a next-item model, as discussed in [offline versus online evaluation](/blog/machine-learning/recommendation-systems/offline-vs-online-the-two-worlds-of-recsys).

```python
import numpy as np
import pandas as pd
from collections import defaultdict

def build_sequences(ratings_path, min_len=5):
    # MovieLens-1M: userId::movieId::rating::timestamp
    cols = ["user", "item", "rating", "ts"]
    df = pd.read_csv(ratings_path, sep="::", names=cols, engine="python")
    df = df.sort_values(["user", "ts"])
    # Reindex items to a dense 1..n range; reserve 0 for padding (and a mask id later).
    item_ids = {iid: idx + 1 for idx, iid in enumerate(df["item"].unique())}
    df["item"] = df["item"].map(item_ids)
    seqs = defaultdict(list)
    for u, it in zip(df["user"].values, df["item"].values):
        seqs[u].append(it)
    # Drop users with too few interactions to form train/val/test.
    seqs = {u: s for u, s in seqs.items() if len(s) >= min_len}
    n_items = len(item_ids)
    return seqs, n_items

def leave_last_out(seqs):
    train, val, test = {}, {}, {}
    for u, s in seqs.items():
        train[u] = s[:-2]          # everything but last two
        val[u]   = (s[:-2], s[-2]) # context, target
        test[u]  = (s[:-1], s[-1]) # context, target
    return train, val, test
```

### 6.2 The model

The model is the stack from Section 4: an item embedding (with id 0 reserved for padding), a learned position embedding, $b$ causal self-attention blocks, and a tied output projection. I use PyTorch's `nn.MultiheadAttention` for the attention sublayer and build the causal mask explicitly.

```python
import torch
import torch.nn as nn

class SASRec(nn.Module):
    def __init__(self, n_items, max_len=200, d=64, n_blocks=2, n_heads=1, dropout=0.2):
        super().__init__()
        self.max_len = max_len
        # +1 for the padding id 0; item ids run 1..n_items.
        self.item_emb = nn.Embedding(n_items + 1, d, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, d)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList()
        for _ in range(n_blocks):
            self.blocks.append(nn.ModuleDict({
                "ln1": nn.LayerNorm(d),
                "attn": nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True),
                "ln2": nn.LayerNorm(d),
                "ffn": nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d)),
            }))
        self.last_ln = nn.LayerNorm(d)

    def _causal_mask(self, n, device):
        # True where attention is FORBIDDEN (upper triangle, strictly future).
        return torch.triu(torch.ones(n, n, dtype=torch.bool, device=device), diagonal=1)

    def seq_encode(self, seq):
        # seq: (B, L) padded item ids; 0 = pad.
        B, L = seq.shape
        positions = torch.arange(L, device=seq.device).unsqueeze(0).expand(B, L)
        x = self.item_emb(seq) + self.pos_emb(positions)
        x = self.dropout(x)
        pad_mask = (seq == 0)              # (B, L) True where padded
        attn_mask = self._causal_mask(L, seq.device)
        for blk in self.blocks:
            h = blk["ln1"](x)
            a, _ = blk["attn"](h, h, h, attn_mask=attn_mask,
                               key_padding_mask=pad_mask, need_weights=False)
            x = x + a                       # residual
            h = blk["ln2"](x)
            x = x + blk["ffn"](h)           # residual
        return self.last_ln(x)              # (B, L, d) state at every position

    def forward(self, seq):
        return self.seq_encode(seq)
```

Two things to flag. The `attn_mask` (`True` = forbidden) implements the causal triangle so position $i$ cannot see the future. The `key_padding_mask` (`True` = pad) stops positions from attending to padding slots, which matters because we left-pad short sequences to `max_len`. The encoder returns the state at *every* position, which is what lets us compute a loss at every position in training and read the last position at inference.

### 6.3 The loss: sampled binary, and full cross-entropy

The training loss scores the true next item against sampled negatives at every position. I show the sampled binary loss (close to the original paper) and a full-softmax variant.

```python
import torch.nn.functional as F

def sampled_bce_loss(model, seq, pos, neg):
    # seq, pos, neg: (B, L). pos[t] is the true next item at position t; neg[t] a random negative.
    states = model.seq_encode(seq)             # (B, L, d)
    pos_emb = model.item_emb(pos)              # (B, L, d)
    neg_emb = model.item_emb(neg)             # (B, L, d)
    pos_logit = (states * pos_emb).sum(-1)    # (B, L)
    neg_logit = (states * neg_emb).sum(-1)    # (B, L)
    valid = (pos != 0).float()                # ignore padded target positions
    loss = -(F.logsigmoid(pos_logit) * valid
             + F.logsigmoid(-neg_logit) * valid).sum() / valid.sum().clamp(min=1)
    return loss

def full_ce_loss(model, seq, pos):
    # Full cross-entropy over the whole item vocab (affordable up to ~1e5 items).
    states = model.seq_encode(seq)             # (B, L, d)
    # Tie weights: logits = states @ item_emb.weight^T
    logits = states @ model.item_emb.weight.t()  # (B, L, n_items+1)
    valid = (pos != 0)
    loss = F.cross_entropy(logits[valid], pos[valid], ignore_index=0)
    return loss
```

The sampled binary loss is $O(1)$ negatives per position and trains fast; the full cross-entropy is $O(|\mathcal{V}|)$ per position but removes the sampling noise. On ML-1M with about 3,700 items, full cross-entropy is the better choice and is what I would ship. On a catalog of millions, switch to sampled softmax with the $\log Q$ correction (covered in the two-tower training post) or in-batch negatives.

The choice of *which* negatives you sample matters as much as the loss form. The default — a uniformly random item — is fine for getting started, but it produces *easy* negatives: a random movie from a 3,700-item catalog is almost always obviously irrelevant, so the model learns to push down items it already scores low, which teaches it little. **Hard negatives** — items that are plausible but wrong, such as popular items the user did not click or items from the same category as the positive — produce a sharper gradient because they sit near the decision boundary. The catch is that with implicit feedback you cannot tell a true negative (user saw it, did not want it) from a *false* negative (user simply never saw it), so aggressive hard-negative mining risks penalizing items the user would have liked. The honest middle ground that I use in practice: sample most negatives uniformly and a minority from a popularity-weighted distribution, which gives some hard signal without over-committing to items that might be false negatives. This is the same tension explored in depth in [implicit feedback models](/blog/machine-learning/recommendation-systems/implicit-feedback-models-als-and-bpr) and in the two-tower training post, and it applies identically to sequence models: the loss form is half the story, the negative distribution is the other half.

### 6.4 The training loop with on-the-fly next-item targets

The targets are the sequence shifted left by one: position $t$ predicts the item at position $t+1$. We build the input/target pair per batch and sample one negative per position.

```python
from torch.utils.data import Dataset, DataLoader

class SeqDataset(Dataset):
    def __init__(self, train_seqs, n_items, max_len=200):
        self.users = list(train_seqs.keys())
        self.seqs = train_seqs
        self.n_items = n_items
        self.max_len = max_len

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        s = self.seqs[self.users[idx]][-(self.max_len + 1):]  # truncate to last max_len+1
        seq = s[:-1]                       # input
        pos = s[1:]                        # next-item targets (shifted left)
        # left-pad to max_len
        pad = self.max_len - len(seq)
        seq = [0] * pad + seq
        pos = [0] * pad + pos
        seen = set(s)
        neg = [0 if p == 0 else self._sample_neg(seen) for p in pos]
        return (torch.tensor(seq), torch.tensor(pos), torch.tensor(neg))

    def _sample_neg(self, seen):
        while True:
            j = np.random.randint(1, self.n_items + 1)
            if j not in seen:
                return j

def train(model, train_seqs, n_items, epochs=200, lr=1e-3, bs=128, device="cuda"):
    ds = SeqDataset(train_seqs, n_items, max_len=model.max_len)
    dl = DataLoader(ds, batch_size=bs, shuffle=True, num_workers=4)
    opt = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98))
    model.to(device).train()
    for ep in range(epochs):
        total = 0.0
        for seq, pos, neg in dl:
            seq, pos, neg = seq.to(device), pos.to(device), neg.to(device)
            opt.zero_grad()
            loss = sampled_bce_loss(model, seq, pos, neg)  # or full_ce_loss(model, seq, pos)
            loss.backward()
            opt.step()
            total += loss.item()
        if ep % 20 == 0:
            print(f"epoch {ep}  loss {total/len(dl):.4f}")
```

Note the truncation: we keep only the last `max_len` items, because attention is $O(n^2)$ and old history adds diminishing value. The `max_len` is a key knob — ML-1M sequences are long and benefit from 200; sparse Amazon categories use 50. Left-padding plus the padding mask handles variable lengths cleanly.

### 6.5 The evaluation harness: Recall@10 and NDCG@10

At test time, for each user we feed the context (everything but the last item), read the state at the last *real* position, score all items, and rank. We report Recall@10 (is the held-out item in the top 10?) and NDCG@10 (rewards a higher rank of the held-out item). To match the SASRec paper's reported protocol we can rank the true item against a sampled set of 100 negatives, but I strongly recommend *full* ranking over the whole catalog — sampled-metric ranking is known to be inconsistent (Krichene and Rendle, KDD 2020), a point we return to in the case studies.

```python
@torch.no_grad()
def evaluate(model, eval_set, train_seqs, n_items, k=10, device="cuda", full_rank=True):
    model.eval().to(device)
    hits, ndcgs = [], []
    all_items = torch.arange(1, n_items + 1, device=device)
    item_table = model.item_emb.weight[1:]            # (n_items, d), drop pad row
    for u, (context, target) in eval_set.items():
        s = context[-model.max_len:]
        pad = model.max_len - len(s)
        seq = torch.tensor([[0] * pad + s], device=device)
        state = model.seq_encode(seq)[0, -1]          # (d,) last position
        scores = item_table @ state                    # (n_items,)
        # mask items already in the user's history so we rank novel items
        seen = set(train_seqs.get(u, [])) | set(context)
        for it in seen:
            if 1 <= it <= n_items:
                scores[it - 1] = -1e9
        topk = torch.topk(scores, k).indices + 1       # back to 1-based item ids
        topk = topk.tolist()
        if target in topk:
            rank = topk.index(target)                  # 0-based
            hits.append(1.0)
            ndcgs.append(1.0 / np.log2(rank + 2))
        else:
            hits.append(0.0)
            ndcgs.append(0.0)
    return float(np.mean(hits)), float(np.mean(ndcgs))
```

Two correctness details that bite people: mask out items already in the user's history before ranking (you should not recommend a movie they already watched, and counting it as a hit inflates the metric), and use full-catalog ranking unless you have a specific reason for sampled negatives. NDCG@10 here is $1/\log_2(\text{rank}+2)$ when the target lands in the top 10 (rank 0 gives the maximum $1.0$), and 0 otherwise — the single-relevant-item special case of the general NDCG defined in [learning to rank for recommenders](/blog/machine-learning/recommendation-systems/learning-to-rank-for-recommenders).

## 7. Complexity, memory, and the practical knobs

A sequential model lives in production, so let me be concrete about cost. Self-attention over a length-$n$ sequence with embedding dimension $d$ costs $O(n^2 d)$ time and $O(n^2)$ memory for the attention matrix (per head, per layer). The feed-forward sublayer is $O(n d^2)$. For the lengths in recommendation, the $n^2$ term is small: at $n = 200$ and $d = 64$, the attention matrix is $200 \times 200 = 40{,}000$ entries per head, trivial. The dominant memory cost is almost always the **item embedding table**, not the attention. With $|\mathcal{V}| = 3{,}706$ items and $d = 64$, the table is about $3{,}706 \times 64 \times 4$ bytes, under a megabyte. Scale that to a 50-million-item catalog at $d = 128$ and the table is about 25 GB — now the embedding table is your hardest serving constraint, exactly the regime where you shard it or push retrieval to a two-tower model and use the sequence model only for ranking.

The practical knobs, with sensible defaults and what they trade:

| Knob | Default (ML-1M) | What it controls | When to change |
| --- | --- | --- | --- |
| `max_len` | 200 | History window; attention cost $O(n^2)$ | Lower to 50 for sparse data; raise for dense, long sessions |
| `d` (embedding dim) | 64 | Capacity and table memory | Raise for huge catalogs; watch memory |
| `n_blocks` | 2 | Depth | 1 for tiny data; rarely above 2–3 |
| `n_heads` | 1–2 | Specialization | Keep small; items are lower-rank than text |
| `dropout` | 0.2–0.5 | Regularization | Raise for sparse data, it overfits fast |
| loss | full CE | Sampling noise vs output cost | Sampled/in-batch for million-item catalogs |

The single most important honest warning: sequential recommenders **overfit fast** on the small interaction datasets that dominate the literature. ML-1M has about a million interactions, which is tiny by deep-learning standards. Heavy dropout, shallow stacks, and early stopping on a validation leave-one-out are not optional. A two-block SASRec with `dropout=0.5` on a sparse Amazon category often beats a deeper, lightly regularized one. This is the recurring lesson of the whole series: the model is rarely the bottleneck; the data and the evaluation are.

### 7.1 Stress-testing the design

A principal engineer's job is not to ship the happy path; it is to know where it breaks. Let me stress-test the SASRec design against the failure modes that actually page you.

**What happens with only implicit feedback?** Almost always, that is the situation — you have clicks and watches, not ratings, and a non-click is ambiguous (did not see it, or saw it and rejected it). SASRec handles this gracefully because its objective only needs positives and sampled negatives; it never assumes a non-interaction is a true dislike. But the evaluation must respect the ambiguity: do not count a recommended item the user never saw as a "miss" with confidence, and prefer ranking metrics over classification metrics, since the labels are positive-only. The model is fine; the metric interpretation is where people go wrong.

**What happens at 100 million items?** The attention is still cheap (it is $O(n^2)$ in *sequence* length, not catalog size), but two things break. First, the embedding table no longer fits on one host, so you shard it or move to a hashed/compositional embedding. Second, the full-softmax loss and full-ranking evaluation become $O(|\mathcal{V}|)$ and intractable — you must switch to sampled softmax with the $\log Q$ correction for training and approximate-nearest-neighbor retrieval for serving. At that scale the sequence model stops being a retriever and becomes a *ranker* over a few hundred candidates produced by a two-tower retriever, which is the architecture I would actually deploy. Trying to make a transformer score 100 million items per request is a category error.

**What happens when the offline metric rises but online is flat?** This is the classic recsys heartbreak, and sequential models have a specific version of it. A higher offline Recall@10 on a leave-last-out split can come from the model getting better at predicting *popular* recent items — which the logged data over-represents because of position and exposure bias — without getting better at the long-tail, exploratory recommendations that actually move engagement. The diagnosis is to slice the offline metric by item popularity: if the lift is concentrated in the head, expect a muted online result. The fix lives upstream in how you sample negatives and in debiasing the training data, topics from the bias and offline-online-gap posts in this series, not in the architecture.

**What happens when the sequence is mostly noise?** Real session logs are full of accidental clicks, bot traffic, and rage-clicking. Attention is somewhat robust here because it can learn to down-weight noisy positions, but it is not magic: if half the sequence is garbage, the softmax still has to allocate weight, and a long run of noise can drown the signal. The practical defenses are pre-filtering (drop sessions below a dwell-time threshold), capping `max_len` so ancient noise is truncated, and dropout, which forces the model not to rely on any single position. If your sequences are extremely noisy, a simpler recency model can outperform attention, because attention's flexibility becomes a liability when there is nothing reliable to attend to.

## 8. Results: attention beats the RNN and the CNN

Here is the headline, and the nuance. I report leave-last-out Recall@10 and NDCG@10 on MovieLens-1M, with values consistent with the SASRec and BERT4Rec papers and the well-known RecBole reproductions. Treat these as representative order-of-magnitude figures under a full-ranking (or large-negative-sample) protocol; exact numbers shift with preprocessing, the negative-sampling choice, and the metric protocol, which is precisely the problem the replication literature flags.

![Matrix comparing GRU4Rec, Caser, SASRec, and BERT4Rec across context type, parallel training, long-range reach, and Recall@10 on MovieLens-1M](/imgs/blogs/self-attention-for-sequences-sasrec-bert4rec-3.png)

The comparison matrix above lines up the four models on the qualities that matter: context direction, whether training parallelizes, long-range reach, and a representative Recall@10. The recurrent GRU4Rec trains serially and loses long-range signal; Caser's convolution parallelizes but is window-limited; the two attention models parallelize *and* reach any position directly, and that combination is what lifts the metrics.

![Matrix of model versus Recall@10 and NDCG@10 on MovieLens-1M showing the two self-attention models beating the GRU and CNN baselines under leave-last-out](/imgs/blogs/self-attention-for-sequences-sasrec-bert4rec-8.png)

The results matrix puts numbers on it. Both attention models clear the baselines on both metrics; BERT4Rec edges SASRec under a tuned cloze protocol. In a flat table:

| Model | Context | Recall@10 (ML-1M) | NDCG@10 (ML-1M) | Trains in parallel |
| --- | --- | --- | --- | --- |
| GRU4Rec | left-to-right (RNN) | ~0.55 | ~0.33 | no (serial) |
| Caser | local window (CNN) | ~0.57 | ~0.35 | yes |
| SASRec | left-to-right (attn) | ~0.62 | ~0.39 | yes |
| BERT4Rec | bidirectional (attn) | ~0.63 | ~0.40 | yes |

### 8.1 What the attention actually learns

One underrated benefit of attention over an RNN is *interpretability you can act on*. Because the attention weights $\alpha_{ij}$ are an explicit probability distribution over past positions, you can read them out for any prediction and see what the model thinks matters. In practice three patterns recur, and each one tells you something operational. On dense, intent-driven sequences the last-position attention concentrates on a few far-back anchor items (the mixer pattern) — a sign the model is using long-range structure and that your `max_len` should be generous. On sparse sequences the weight collapses onto the single most recent item — a sign there is little sequence structure and a simpler recency or Markov model might match it for a fraction of the cost. And a pathological third pattern, where attention spreads almost uniformly across all positions, usually means the model has not learned anything discriminative and is averaging — a red flag for undertraining or a sequence that is mostly noise. Logging the entropy of the last-position attention distribution across a validation set is a cheap, continuous health metric: rising entropy over training epochs often signals the model giving up and reverting to an average, which correlates with a stalling validation NDCG. None of this is available from a GRU's hidden state, which is an opaque blend with no per-position weights to inspect.

#### Worked example: turning the metric delta into a decision

Suppose your baseline ranker uses GRU4Rec and you are deciding whether to switch to SASRec. The offline jump is Recall@10 from ~0.55 to ~0.62, a relative lift of about 13%, and NDCG@10 from ~0.33 to ~0.39, about 18% relative. Is that worth the migration? Two checks. First, sanity-check the lift on a *temporal* split (not a random one) to be sure you are not measuring leakage; sequential models are especially prone to looking good on shuffled splits that let future interactions leak into training. Second, translate offline to online expectation: a robust offline NDCG lift of this size typically corresponds to a *smaller* online engagement lift — the offline-online gap from [the two worlds of recsys](/blog/machine-learning/recommendation-systems/offline-vs-online-the-two-worlds-of-recsys) means a 18% offline NDCG gain might be a 1–4% online CTR gain after position bias and the missing-not-at-random nature of logged data eat into it. For most teams a clean 13% Recall@10 lift with the *same* serving cost (SASRec serves as cheaply as a GRU and trains faster) is an easy yes. The decision flips only if your data is so sparse that the attention model overfits below the GRU, which does happen on the tiniest catalogs.

## 9. The replication debate: do BERT4Rec's gains hold up?

This is where a staff engineer earns their salary: not believing a leaderboard. The original BERT4Rec paper reported gains over SASRec, and for a few years the bidirectional model was treated as the stronger choice by default. Then Petrov and Macdonald ("A Systematic Review and Replicability Study of BERT4Rec for Sequential Recommendation," RecSys 2022) tried to reproduce those numbers and found something uncomfortable: across many public BERT4Rec implementations, the reported results often *did not replicate*, and crucially, BERT4Rec only matched or beat SASRec when it was trained far longer than the papers' default schedules — sometimes an order of magnitude more training steps. Under matched, modest training budgets, a well-tuned SASRec frequently equaled or beat BERT4Rec.

The mechanism is the train-serve mismatch from Section 5. BERT4Rec's cloze objective is a harder, denser task than next-item prediction, so it needs more optimization to converge, and the appended-mask serving configuration is off-distribution. SASRec's autoregressive objective matches the serving task exactly, so it converges faster and serves what it trained on. The lesson is not "BERT4Rec is bad" — with enough training it is a strong model and bidirectional context is genuinely useful for representation. The lesson is **methodological**: headline gains in sequential recommendation are extremely sensitive to the training budget, the negative-sampling protocol, and the evaluation metric, and many published deltas evaporate under matched conditions. When you read "model X beats model Y by 3% NDCG," ask: same training budget? full ranking or sampled negatives? temporal or random split? If you cannot answer, you cannot trust the delta.

This connects to a second, deeper measurement problem. Krichene and Rendle ("On Sampled Metrics for Item Recommendation," KDD 2020) showed that ranking the true item against a *small sample* of negatives — the protocol many sequential-rec papers use, often 100 negatives — produces metrics that are *inconsistent* with full-catalog ranking: a model that wins on sampled metrics can lose on full metrics, and vice versa. Many of the SASRec-versus-BERT4Rec comparisons in the wild used the sampled protocol, which is one more reason the deltas wobble. If you take one practical rule from this section: **evaluate on the full catalog, with a temporal split, under a matched training budget**, and re-run the comparison yourself before you believe any leaderboard.

### 9.1 Deploying a sequence model in the funnel

Suppose the offline comparison checks out and you decide to ship SASRec as a ranker. Here is what production actually demands, beyond the training loop, because this is where most teams discover the parts the paper did not mention.

**Feature freshness and train-serve skew.** The sequence model's input is the user's recent history, and that history must be assembled *identically* offline (during training) and online (at serve time). This is the train-serve skew trap that silently halves precision: if your offline pipeline includes an interaction that, online, has not yet propagated to the serving feature store, the model trains on a future it cannot see at inference. The defense is to construct training sequences from a *point-in-time snapshot* of the feature store — only events that would have been visible at the moment of the prediction — and to share the exact sequence-assembly code between offline and online paths. A sequence model is especially sensitive to this because order and recency are its whole signal; an off-by-one in how recency is computed offline versus online is enough to wreck it.

**Serving latency.** A two-block SASRec over a length-200 sequence is a handful of small matrix multiplies — single-digit milliseconds on a GPU, low tens of milliseconds on a CPU for a batch of candidates. The cost scales with the candidate count (you score the history once and dot it against each candidate's embedding), so as a *ranker* over a few hundred retrieved candidates it is cheap. The latency trap is using it as a retriever: scoring the full catalog turns the cheap dot product into millions of dot products per request. Keep it at the ranking stage, fed by a [two-tower retriever](/blog/machine-learning/recommendation-systems/the-two-tower-model-for-retrieval), and latency is a non-issue.

**The cold-start and the empty sequence.** A brand-new user has no history, so the sequence is empty and the model has nothing to attend to. SASRec degrades to predicting from the position embeddings alone, which is essentially a popularity prior — acceptable as a fallback but not great. The standard production pattern is to route cold users to a separate content-based or popularity model from [content-based and hybrid recommenders](/blog/machine-learning/recommendation-systems/content-based-and-hybrid-recommenders) until they accumulate enough interactions for the sequence model to have signal, then hand them over. Do not expect the sequence model to solve cold start; it is a warm-user model by construction.

**Retraining cadence and the feedback loop.** Sequence models drift as the catalog and user behavior change, and because they are trained on logged interactions, they participate in the serve-log-train feedback loop that can quietly concentrate the catalog onto a few popular items. Retrain on a regular cadence (daily or weekly depending on catalog velocity), monitor the *diversity* of recommendations alongside the accuracy metric, and inject exploration so the logged data does not collapse into a self-fulfilling popularity prophecy. This is the feedback-loop frame from the series spine: a sequence model is not a static artifact, it is one turn of a loop, and the loop is what you actually operate.

## 10. SASRec vs BERT4Rec vs GRU: the decision

Put it together into a recommendation you can act on.

| Dimension | GRU4Rec | SASRec | BERT4Rec |
| --- | --- | --- | --- |
| Context | left-to-right | left-to-right | bidirectional |
| Training | serial, slow | parallel, fast | parallel, but needs long schedule |
| Objective | next-item | next-item (matches serving) | cloze (train-serve mismatch) |
| Long-range reach | decays | direct | direct |
| Serving | append nothing | append nothing | must append a mask |
| Default verdict | legacy baseline | **the default** | use with care, train long |

SASRec is the default. It is the model you reach for first when you need sequential recommendation: it trains fast, it serves cheaply, its objective matches the serving task, it has no train-serve mismatch, and it is robust to reproduce. It beats the GRU and CNN baselines reliably, and it equals or beats BERT4Rec under matched training budgets. Reach for BERT4Rec when you specifically need a *representation* of the sequence (for downstream tasks beyond next-item), when bidirectional context is genuinely informative for your domain, and when you can afford the longer training schedule and have validated the appended-mask serving on your own data. Keep GRU4Rec only as a baseline or in a legacy system; its serial training and decaying memory are strictly dominated for new work.

The broader frame: a sequential model is usually a *ranking* (or late-stage retrieval) component, not the whole system. In the funnel, you might use a [two-tower model](/blog/machine-learning/recommendation-systems/the-two-tower-model-for-retrieval) for cheap candidate generation over a huge catalog, then a SASRec-style sequence model to *rank* the few hundred candidates using the user's recent trajectory, and finally re-rank for diversity and business rules. The sequence model's strength — fine-grained, order-aware scoring — is wasted if you make it score the entire catalog, and its $O(n^2)$ cost is irrelevant at ranking-stage candidate counts. Multi-objective ranking, where next-item relevance is one of several heads, is covered in [multi-task and multi-objective ranking](/blog/machine-learning/recommendation-systems/multi-task-and-multi-objective-ranking-mmoe-ple).

## 11. Case studies and real numbers

Four anchors from the literature and production, with honest framing.

**SASRec (Kang and McAuley, ICDM 2018).** The original self-attentive sequential recommender. On Amazon, Steam, and MovieLens datasets it outperformed GRU4Rec, Caser, and FPMC (a factorized Markov chain), with the largest gains on *dense* datasets where long-range dependencies exist (ML-1M) and smaller gains on extremely sparse ones where there is little sequence structure to model. A key reported finding: SASRec's attention learned interpretable patterns — on dense data it attended to long-range items, on sparse data it concentrated on the most recent item, which is the model *discovering* the right reach for the data. The paper also showed SASRec trained an order of magnitude faster than the RNN baselines on GPU, the parallelism payoff made concrete.

**BERT4Rec (Sun et al., CIKM 2019).** Introduced the cloze objective and bidirectional encoder to sequential recommendation, reporting gains over SASRec on Beauty, Steam, and ML-1M/ML-20M. The architecture is essentially BERT applied to item sequences, with the appended-mask serving trick. Influential as a proof that bidirectional context helps representation; controversial in hindsight because of reproducibility (next case).

**The replication study (Petrov and Macdonald, RecSys 2022).** Systematically re-evaluated BERT4Rec and found that many public implementations underperformed their papers' claims, and that BERT4Rec needed substantially more training than SASRec to realize its advantage. Their tuned BERT4Rec (with a long schedule and a careful implementation) did reach the reported gains, but a matched-budget SASRec was competitive. The practical conclusion the field took: report training budgets, evaluate on full rankings, and do not trust a single-number leaderboard. This is the most useful paper in the post for a practitioner, because it is about *method*, not architecture.

**Sampled metrics (Krichene and Rendle, KDD 2020).** Not specific to sequences, but it invalidates a swath of sequential-rec comparisons. Showed that ranking against a small negative sample yields metrics inconsistent with full ranking, and proposed corrections. The takeaway for this post: when you compare SASRec and BERT4Rec, rank against the *full* catalog, or your conclusion may flip.

**Transformers in production sequence rec.** Beyond the academic benchmarks, self-attention sequence models underpin many industrial recommenders. Alibaba's behavior-sequence transformer (Chen et al., 2019) applied self-attention to user behavior sequences in a click-through-rate ranker; the broader DIN/DIEN line at Alibaba (Zhou et al., 2018/2019) used attention over user behavior to weight history by relevance to the candidate. More recently, generative-retrieval approaches like TIGER (Rajput et al., 2023) and the "Actions Speak Louder than Words" generative recommender from Meta (Zhai et al., 2024) push the transformer-over-actions idea to retrieval scale. The thread through all of them is the same one this post argues: attention over a user's ordered behavior, weighting the past by learned relevance, beats fixed-window and recurrent alternatives — and increasingly, the same architecture that powers language models is becoming the backbone for recommendation, which is why finetuning LLMs for recommendation, covered in [finetuning LLMs for recommendation in practice](/blog/machine-learning/recommendation-systems/finetuning-llms-for-recommendation-in-practice), is a natural next step.

## 12. When to reach for this (and when not to)

A decisive section, because every choice is a cost.

**Reach for a self-attention sequence model when** the order and recency of interactions carry real signal (sessions, watch sequences, browse trails), when you have enough interactions per user for the model to learn order patterns (dense sequences), and when you are scoring a manageable candidate set (ranking stage, or retrieval over a moderate catalog). SASRec specifically when you want the simplest, fastest, most reproducible option that matches the serving task.

**Do not reach for it when** your data is so sparse that each user has a handful of interactions with no temporal structure — a static collaborative or content model from [content-based and hybrid recommenders](/blog/machine-learning/recommendation-systems/content-based-and-hybrid-recommenders) will do as well or better, without the overfitting risk. Do not use a sequence model as your *retrieval* engine over a hundred-million-item catalog by scoring every item with the full transformer — that is $O(|\mathcal{V}|)$ forward passes and will not serve; use a two-tower retriever and let the sequence model rank. Do not ship BERT4Rec because a paper said it wins — validate the appended-mask serving and the training budget on your data first, or default to SASRec. Do not trust a sampled-metric leaderboard delta; re-run with full ranking. And do not skip heavy regularization on small datasets — a lightly regularized deep stack will overfit and lose to a well-regularized two-block model.

A useful way to make the decision concretely is to ask three questions in order. *Does order carry signal in my domain?* If shuffling a user's history would barely change what they want next (some commodity e-commerce, some news), a sequence model is overkill — a static model captures the taste and you save the complexity. *Do I have enough interactions per user for attention to learn the order patterns?* Below roughly five interactions per user on average, the sequence is too short to learn from and a Markov or recency heuristic is competitive; the deep model's capacity goes to memorizing noise. *Am I scoring a candidate set or the whole catalog?* If the whole catalog, you are at the retrieval stage and should not be using a transformer scorer at all. Only when all three answers point the right way — order matters, sequences are long enough, and you are ranking a candidate set — is a self-attention sequence model the right tool, and then SASRec is your first call. This is the same disciplined "every choice is a cost" reasoning the capstone [recommender systems playbook](/blog/machine-learning/recommendation-systems/the-recommender-systems-playbook) applies across the whole funnel: reach for the heaviest hammer only when the lighter ones have demonstrably failed.

## 13. Key takeaways

- **Self-attention beats RNNs and CNNs for sequences on three axes**: parallel training, direct long-range access, and adaptive per-step weighting of the past. The kitchen-shopper's mixer is one attention weight away, not forty hidden-state overwrites away.
- **The mechanism is $\text{softmax}(QK^\top/\sqrt{d})V$** applied to item embeddings, with **learned position embeddings** added to the input because raw attention is order-blind.
- **SASRec uses a causal mask** so position $i$ sees only $\le i$, and trains autoregressively to predict the next item. Its objective matches the serving task exactly — no train-serve mismatch — which makes it the robust default.
- **BERT4Rec uses bidirectional attention with a cloze objective**: mask interior items and predict them from both sides. Richer context, but it must *append a mask* at serving, a configuration it rarely saw in training.
- **The BERT4Rec replication debate is a lesson in method**: its gains need a long training schedule and often do not reproduce; under matched budgets a tuned SASRec is competitive. Report your training budget.
- **Evaluate honestly**: full-catalog ranking (not a 100-negative sample, which is inconsistent), a temporal leave-last-out split, and mask out already-seen items before ranking.
- **The embedding table, not the attention, is your memory bottleneck** at scale; the $O(n^2 d)$ attention cost is cheap for the short sequences in recommendation.
- **Overfitting is the real enemy** on the small public datasets: shallow stacks, heavy dropout, early stopping. The model is rarely the bottleneck; the data and the evaluation are.
- **Use the sequence model as a ranker**, fed by cheap retrieval; do not make a transformer score the whole catalog.

## 14. Further reading

- Kang and McAuley, "Self-Attentive Sequential Recommendation" (SASRec), ICDM 2018 — the causal self-attention sequence model.
- Sun et al., "BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer," CIKM 2019 — the bidirectional cloze model.
- Petrov and Macdonald, "A Systematic Review and Replicability Study of BERT4Rec for Sequential Recommendation," RecSys 2022 — read this before trusting any sequential-rec leaderboard.
- Krichene and Rendle, "On Sampled Metrics for Item Recommendation," KDD 2020 — why sampled-negative metrics mislead.
- Vaswani et al., "Attention Is All You Need," NeurIPS 2017 — the transformer and scaled dot-product attention.
- Hidasi et al., "Session-based Recommendations with Recurrent Neural Networks" (GRU4Rec), ICLR 2016; Tang and Wang, "Personalized Top-N Sequential Recommendation via Convolutional Sequence Embedding" (Caser), WSDM 2018 — the baselines attention replaced.
- Within this series: [what is a recommender system](/blog/machine-learning/recommendation-systems/what-is-a-recommender-system) (the funnel map), [sequential and session-based recommendation](/blog/machine-learning/recommendation-systems/sequential-and-session-based-recommendation), [the two-tower model for retrieval](/blog/machine-learning/recommendation-systems/the-two-tower-model-for-retrieval), [multi-task and multi-objective ranking](/blog/machine-learning/recommendation-systems/multi-task-and-multi-objective-ranking-mmoe-ple), [finetuning LLMs for recommendation in practice](/blog/machine-learning/recommendation-systems/finetuning-llms-for-recommendation-in-practice), and the capstone [the recommender systems playbook](/blog/machine-learning/recommendation-systems/the-recommender-systems-playbook).
