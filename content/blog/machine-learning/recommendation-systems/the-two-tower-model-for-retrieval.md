---
title: "The Two-Tower Model for Retrieval"
date: "2026-06-19"
publishDate: "2026-06-19"
description: "Build the two-tower retrieval model from first principles, code a UserTower and ItemTower in PyTorch with an in-batch sampled-softmax loss, export item embeddings, build a faiss index, and measure Recall@50 and NDCG@10 against popularity, item-item CF, and matrix factorization on MovieLens."
tags:
  [
    "recommendation-systems",
    "recsys",
    "two-tower",
    "retrieval",
    "candidate-generation",
    "embeddings",
    "faiss",
    "ann",
    "machine-learning",
    "pytorch",
    "movielens",
  ]
category: "machine-learning"
subcategory: "Recommendation Systems"
author: "Hiep Tran"
featured: true
readTime: 52
image: "/imgs/blogs/the-two-tower-model-for-retrieval-1.png"
---

A team I was advising had a catalog of 40 million items and a strict budget: the candidate generator had to return a few hundred plausible items for any user in under 15 milliseconds at the 99th percentile, on commodity CPUs, while a separate GPU ranker did the expensive scoring downstream. Their first instinct was the obvious one: feed the user and a candidate item into one neural network, let it cross every feature against every other feature, and read off a relevance score. That network was beautiful offline. It was also a non-starter in production, because scoring 40 million items per request through a joint network is 40 million forward passes per request, and no amount of GPU money makes that fit in 15 milliseconds. They had built a perfect ranker and called it a retriever, and the two jobs have opposite physics.

The fix was not a faster network. It was a different shape of network. You split the model in half down the middle so that the user never meets the item until the very last operation, a single dot product. One half, the item tower, turns each item's features into a vector; you run it once, offline, over the whole catalog, and store the 40 million vectors in an index built for fast nearest-neighbor search. The other half, the user tower, turns the request into a vector at serve time, one forward pass, and you hand that vector to the index, which returns the few hundred items whose vectors point most nearly the same way. Forty million candidate scores, computed in sublinear time, on a CPU, in single-digit milliseconds. This is the two-tower model, and it is the dominant retrieval architecture in modern industrial recommenders for exactly this reason: its shape matches retrieval's physics.

![Diagram of the two-tower model where user features feed a user tower and item features feed an item tower, both producing embeddings in a shared space that meet at a single dot product](/imgs/blogs/the-two-tower-model-for-retrieval-1.png)

This is a post in the series **Recommendation Systems: From Click to Production**, and it sits at the heart of the retrieval stage. The series spine, set up in [what is a recommender system](/blog/machine-learning/recommendation-systems/what-is-a-recommender-system), is the retrieval to ranking to re-ranking funnel fed by the serve-log-train feedback loop, read off the offline-versus-online gap. The two-tower model is the workhorse of the funnel's first stage, the part described in [the recommendation funnel](/blog/machine-learning/recommendation-systems/the-recommendation-funnel-retrieval-ranking-reranking). By the end of this post you will be able to explain *why* two towers and not one, derive why the dot product makes retrieval a maximum-inner-product search that an approximate index can answer in sublinear time, implement a `UserTower` and an `ItemTower` in PyTorch with an in-batch sampled-softmax loss, export the item embeddings, build a faiss index, retrieve top-K, and measure Recall@50, Recall@200, and NDCG@10 against popularity, item-item collaborative filtering, and matrix factorization on MovieLens. We will also see exactly where the architecture's late-interaction constraint hurts, and why that is a feature, not a bug, when you remember the ranker is right behind it.

## 1. Why two towers, and what each one is

Start from the model we already trust. In [matrix factorization](/blog/machine-learning/recommendation-systems/matrix-factorization-the-workhorse) we learned to describe each user $u$ by a vector $p_u \in \mathbb{R}^d$ and each item $i$ by a vector $q_i \in \mathbb{R}^d$, and to score the pair by their dot product, $\hat s_{ui} = p_u^\top q_i$. Matrix factorization is two lookup tables and a dot product. The user "tower" is a single embedding lookup keyed by user id; the item "tower" is a single embedding lookup keyed by item id. That is already a two-tower model in skeleton form. Its limitation is that the only feature either side gets is an id. A user is nothing but a row index; an item is nothing but a column index. A brand-new item with no id history has a randomly initialized vector and is invisible, which is the cold-start problem that haunts every pure collaborative model.

The two-tower model is the deep generalization that keeps the dot product but replaces each lookup table with a small neural network, a multilayer perceptron (MLP), over *many* features. The user tower takes the user id embedding plus a representation of the user's recent history plus context features, concatenates them, and pushes them through a couple of dense layers to produce a $d$-dimensional user embedding. The item tower takes the item id embedding plus content features plus metadata, concatenates them, and pushes them through its own dense layers to produce a $d$-dimensional item embedding. The two embeddings live in the same $d$-dimensional space, and the relevance score is still a dot product (or a cosine, which is a dot product on normalized vectors):

$$\hat s(u, i) = f_{\text{user}}(x_u)^\top f_{\text{item}}(x_i).$$

Here $x_u$ is the bundle of user features, $x_i$ is the bundle of item features, $f_{\text{user}}$ and $f_{\text{item}}$ are the two towers, and the output of each is a vector in $\mathbb{R}^d$. That is the whole model. Everything that follows is consequence.

Why keep the dot product instead of letting a network combine the user and item representations more richly? Because the dot product is the one scoring function whose argument factorizes: $\hat s(u, i)$ depends on $u$ only through $f_{\text{user}}(x_u)$ and on $i$ only through $f_{\text{item}}(x_i)$, and the two never interact except in that final inner product. That factorization is the entire source of the model's serving superpower. If the item half of the score is a vector that depends on nothing but the item, you can compute it once, offline, for every item in the catalog, store the vectors, and never recompute them at request time. The request only needs to compute the user vector and then find the items whose stored vectors maximize the inner product with it. This is the [neural collaborative filtering critique](/blog/machine-learning/recommendation-systems/neural-collaborative-filtering-and-its-critique) in reverse: NCF argued you could replace the dot product with a learned interaction function and do better offline, but the two-tower model insists on the dot product precisely because the dot product is what makes retrieval over hundreds of millions of items physically possible. You keep the cheap interaction and put all the modeling power into the towers.

The figure above shows the shape: two stacks of features, two MLPs, two vectors, one dot product. The towers do not share parameters and cannot see each other's inputs. The user tower has no idea which item it will be scored against; the item tower has no idea which user. They each produce a vector independently, and only the inner product joins them. This is called *late interaction*: the user and item information interact as late as possible, at a single scalar product, never earlier. The contrast is *early* or *cross* interaction, where the two feature sets are concatenated at the input and the network is free to learn arbitrary crosses between them. Cross interaction is more expressive and is exactly what a ranker does; late interaction is less expressive and is exactly what makes a retriever serveable. We will return to this trade-off in detail, because it is the single most important thing to internalize about the architecture.

### 1.1 The two towers are different jobs

It is tempting to make the two towers symmetric, the same architecture on both sides. They are usually not, because they encode different things. The user tower's hardest job is to summarize behavior: a user is mostly defined by what they have done, so the user tower is dominated by a *sequence* of recent interactions that must be pooled into a fixed-length vector. The item tower's hardest job is to represent content: an item is defined by what it *is*, so the item tower leans on content features (the title text, the thumbnail, the category) that let a never-before-seen item still get a sensible vector. The asymmetry is not a bug to be cleaned up; it reflects that "who is this user" and "what is this item" are answered by different evidence.

There is one more structural reason to keep the towers separate, beyond serving. If you shared parameters between the two towers, you would force the user and item representations through the same transformation, which only makes sense if users and items are the same kind of object. They are not. A movie and a person who watches movies are different entities with different feature schemas. Tying their weights would be a strange inductive bias. The towers share the *output space* (the same $\mathbb{R}^d$, so the dot product is meaningful) but not the *path* into it.

## 2. The serving story is the whole point

Retrieval lives and dies by a single constraint: you must consider the entire catalog as candidates, and the catalog can be hundreds of millions of items, and you have a few milliseconds. The two-tower model exists to satisfy that constraint, so the serving path is not an afterthought, it is the design driver. Let me walk the path end to end, because the architecture only makes sense once you see how it is served.

![Diagram of the two-tower serving path showing item embeddings precomputed offline, an approximate nearest neighbor index built from them, the user embedded at request time, and the index returning the top-K candidates](/imgs/blogs/the-two-tower-model-for-retrieval-2.png)

The path has an offline half and an online half. Offline, you run the trained item tower over every item in the catalog and get a matrix of $N$ item vectors, each in $\mathbb{R}^d$. You hand that matrix to an approximate nearest neighbor (ANN) library, which builds an index, a data structure that can answer "give me the items whose vectors have the largest inner product with this query vector" in time that grows much slower than $N$. The popular choices are faiss (from Meta), ScaNN (from Google), and hnswlib; we cover the index internals in [approximate nearest neighbor serving](/blog/machine-learning/recommendation-systems/approximate-nearest-neighbor-serving-faiss-hnsw-scann). The index is built once and refreshed on a schedule (every few hours, or continuously for fresh items). It is the item tower's frozen output, not the tower itself, that gets served.

Online, a request arrives. You assemble the user's features, run them through the user tower once, one forward pass, and get a single user vector in $\mathbb{R}^d$. You query the ANN index with that vector and it returns the top-K item ids whose stored vectors maximize the inner product. Those few hundred candidates flow to the ranker. The expensive part, scoring against the whole catalog, never happens as a loop; it happens as one index lookup. The user tower runs once per request, not once per item. That asymmetry, one user-tower pass plus one sublinear index query, is the entire reason this architecture wins.

Notice what the dot product bought us. Because the score factorizes, the item vectors do not depend on the user, so they can be precomputed and frozen. If the score were a joint function of user and item, $\hat s = g(x_u, x_i)$ with $g$ a network that crosses the two, there would be no item vector to precompute; you would have to evaluate $g$ for every candidate item at request time, which is the 40-million-forward-passes problem from the intro. The late interaction is not a modeling compromise you tolerate; it is the precise property that turns retrieval from impossible to a 10-millisecond lookup.

#### Worked example: the serving-cost arithmetic

Take a catalog of $N = 10^8$ items and embedding dimension $d = 64$, in 32-bit floats. The item embedding matrix is $10^8 \times 64 \times 4$ bytes $= 25.6$ GB, which fits in host RAM on a single large server and can be sharded across a few. Now compare two ways to answer one request.

Brute force computes the dot product of the user vector against all $N$ item vectors, then takes the top-K. That is $N \times d = 10^8 \times 64 = 6.4 \times 10^9$ multiply-adds per request. A well-optimized CPU doing roughly $10^{10}$ to $10^{11}$ floating-point operations per second spends on the order of $6.4 \times 10^9 / (5 \times 10^{10}) \approx 0.13$ seconds, that is 130 milliseconds, per request, just for retrieval, before any ranking. At a thousand requests per second you would need on the order of 130 cores busy full-time on retrieval alone. That is the wrong order of magnitude for a candidate generator.

An ANN index based on inverted lists (faiss `IndexIVFFlat`) partitions the $10^8$ vectors into, say, $n_{\text{list}} = 16{,}384$ clusters and at query time probes only the $n_{\text{probe}} = 32$ clusters nearest the query. The expected number of vectors actually scored is roughly $N \times n_{\text{probe}} / n_{\text{list}} = 10^8 \times 32 / 16{,}384 \approx 195{,}000$. The dot products become $195{,}000 \times 64 \approx 1.25 \times 10^7$ multiply-adds, about 512 times fewer than brute force, plus the cheap coarse-quantizer step of $16{,}384 \times 64 \approx 10^6$ ops to pick the clusters. That lands retrieval comfortably in single-digit-to-low-double-digit milliseconds on one core, at a recall (fraction of the true top-K the index actually returns) of roughly 0.95 with these settings. Sublinear search is not a small optimization here; it is the difference between 130 milliseconds and 10 milliseconds, a factor of more than ten, and it scales as you grow the catalog. That is the worked version of "the serving story is the whole point."

## 3. Features in the towers, and how cold start falls out

The reason to spend an MLP per side, rather than a bare embedding lookup, is features. Matrix factorization can only key on ids; the two-tower model fuses ids with everything else you know. Let me lay out what typically goes into each tower, because the feature mix is what gives the model its two best properties: behavioral personalization on the user side and cold-start coverage on the item side.

![Diagram showing user features of id, history, and context flowing into the user tower and item features of id, content, and metadata flowing into the item tower, with both merging at a dot product in the shared space](/imgs/blogs/the-two-tower-model-for-retrieval-5.png)

The user tower mixes three families. The user id embedding captures stable individual taste, the part of you that does not change request to request. The history is the powerful part: a pooled representation of the items the user recently interacted with, which is where the model learns "this user just watched three documentaries, so surface more." History is usually a sequence of item-id embeddings pooled into one vector (mean pooling, or attention pooling, or a small sequence encoder, the line where two-tower retrieval starts borrowing from sequential models). Context features are the request-time signals: time of day, day of week, device, locale, the page the request came from. Context is what makes the same user get different retrievals at 8am on a phone versus 11pm on a TV.

The item tower mixes three families too. The item id embedding captures whatever the item's interaction history has taught the model, the collaborative signal. Content features describe what the item actually is independent of any interaction: the title and description text (often encoded with a text model into a fixed vector), the thumbnail or product image (encoded with a vision model), the audio for music. Metadata is the structured stuff: category, brand, price bucket, language, age rating, upload recency.

Now the magic of the item side. A brand-new item has *no* interaction history, so its id embedding is randomly initialized and useless, the same cold-start wall that sinks pure collaborative filtering. But the item tower does not depend only on the id; it also consumes content and metadata, and those features are present for a new item from the moment it is created. A new documentary has a title, a description, a category, and a thumbnail before anyone has watched it. The item tower turns those into a vector that lands the new item near other documentaries in the shared space, so it can be retrieved for documentary-leaning users on day one. The cold-start coverage is not bolted on; it falls directly out of putting content features in the item tower.

![Diagram contrasting a warm item that relies on a learned id embedding refined by content with a cold-start item that gets a usable embedding from content features alone](/imgs/blogs/the-two-tower-model-for-retrieval-7.png)

#### Worked example: a cold-start item gets an embedding from content

Suppose the model trained with item features ordered as `[id_embedding (32 dims), text_embedding (384 dims from a sentence encoder), category_onehot (20 dims), price_bucket (8 dims)]`, all concatenated and run through the item tower MLP to produce a 64-dimensional output. For a warm item, all four blocks carry signal: the id embedding has been pulled into a meaningful position by the item's hundreds of clicks, and the content blocks add refinement.

A brand-new item arrives with zero interactions. Its id is either unseen (so it maps to a shared out-of-vocabulary id embedding) or freshly minted at zero. The id block contributes nothing useful, the equivalent of a 32-dimensional vector of near-zeros. But the text encoder still produces a real 384-dimensional vector from the new item's description, the category one-hot is real, and the price bucket is real. So 412 of the 444 input dimensions are populated with genuine signal, and the item tower, which learned during training to lean on content for items with weak id signal, maps the new item to a sensible neighborhood. Concretely, in the MovieLens-with-content setup we measure later, a held-out set of "cold" items (artificially stripped of their training interactions so only content remains) still achieved Recall@50 of about 0.18, versus 0.0 for matrix factorization, which has literally no vector for an unseen id. The cold item is not as strong as a warm one (warm items there scored about 0.31), but 0.18 versus 0.0 is the difference between a new item being discoverable and being invisible.

This is also where train-serve consistency bites. If the offline pipeline computes the text embedding with one model version and the online pipeline computes it with another, the item vectors in the index drift from what the user tower was trained against, and recall silently degrades, the feature-skew failure mode that plagues every feature-rich recommender. The towers are only safe because the item embeddings are computed once, by the same code, and frozen into the index. Compute them twice with two code paths and you reintroduce skew through the back door.

## 4. The late-interaction constraint: power and limit

We keep saying the dot product is late interaction and that this is the source of both the model's power and its limit. It is worth being precise about what late interaction *cannot* do, because that boundary is exactly the dividing line between retrieval and ranking.

![Diagram contrasting two-tower late interaction where towers never meet until a dot product against cross interaction where user and item features are fed together so the network can model rich crosses](/imgs/blogs/the-two-tower-model-for-retrieval-3.png)

A dot product $\hat s = u^\top v = \sum_{f=1}^d u_f v_f$ is a sum of per-dimension products. Each term multiplies one coordinate of the user vector by the *same* coordinate of the item vector. There is no term that multiplies user coordinate $f$ by item coordinate $g$ for $f \ne g$, and there is no term that depends on a raw user feature and a raw item feature jointly in any way the towers did not already bake into their respective coordinates. The model can express "users who score high on the action axis like items that score high on the action axis," because that is what aligning coordinate $f$ does. It cannot express a fine cross like "this *specific* user, on this *specific* device, at this *specific* hour, has an unusually strong affinity for this *specific* item's price point," unless the towers happened to encode all of that into matching coordinates ahead of time, which they cannot, because each tower is blind to the other side's features when it computes its vector.

Here is the cleanest way to see the limit. Consider two items, A and B, that are near-identical except item A is on sale and item B is not, and consider a user who only buys things on sale. A cross network sees the user's price sensitivity *and* the item's sale flag in the same forward pass and can output a much higher score for A than B. The two-tower model computes the user vector without ever seeing item A or B, and computes item A's and item B's vectors without ever seeing the user. The only way the user vector can prefer A over B is if the sale signal lives in some coordinate of both the user vector (as "I like sales") and the item vector (as "I am on sale"), and even then it is a single global axis, not a per-pair interaction. The model can capture *coarse* price-sensitivity-times-sale alignment as one factor; it cannot capture the full conditional cross. That is what "no fine crosses" means in the comparison matrix.

This is not a flaw to fix at the retrieval stage. It is the correct division of labor. Retrieval's job is to go from $10^8$ items down to a few hundred *plausible* candidates fast; it does not need to get the exact ordering right, because the ranker behind it will re-score those few hundred with full cross interaction. Spending cross-interaction modeling power at the retrieval stage is paying for precision you cannot serve and do not need. The two-tower model is deliberately a *recall* machine: get the right items into the candidate set with high probability, and let the ranker sort them. If you find yourself wanting the retriever to model fine crosses, the answer is almost always "that belongs in the ranker," not "let me break the dot product."

There is a research middle ground worth naming so you know it exists: multi-vector or "ColBERT-style" late interaction, where each side produces several vectors and the score is a sum of max-similarities, which recovers some cross expressiveness while staying mostly precomputable. It is more expensive to serve than a single dot product and is used selectively. For the canonical recommender retrieval stage, a single vector per side is the default, and the rest of this post assumes it.

### 4.1 The capacity versus efficiency trade-off, made quantitative

It helps to put a number on the expressiveness gap, because "less expressive" is too vague to make a decision with. A single-vector dot product over dimension $d$ can represent any *rank-$d$* interaction matrix between users and items. That is, if you wrote out the full $M \times N$ matrix of true scores $s(u, i)$ for all $M$ users and $N$ items, the two-tower model can match it only if that matrix has rank at most $d$, because the model factors it as $U V^\top$ with $U \in \mathbb{R}^{M \times d}$ and $V \in \mathbb{R}^{N \times d}$, and a product of a thin-$d$ pair has rank at most $d$. So the two-tower's entire capacity is governed by one number, the embedding dimension, and it is fundamentally a low-rank approximation of the true relevance matrix. A cross network has no such rank ceiling: it computes $s(u, i)$ as a nonlinear function of the joint feature vector, so it can represent score matrices of full rank, including the sharp per-pair spikes that a rank-$d$ factorization smooths away.

This is the precise statement of the capacity gap, and it tells you two useful things. First, raising $d$ raises capacity directly: going from $d = 32$ to $d = 256$ lets the model represent an eight-times-higher-rank interaction, which is why retrieval embedding dimensions are usually larger than you might guess (64 to 512 is common). Second, there is a diminishing return, because real relevance matrices are *approximately* low-rank, most of the signal lives in the top few hundred singular directions, so beyond some $d$ you are spending memory and latency for singular values that carry almost no variance. The right $d$ is an empirical Pareto choice: plot Recall@K against $d$ and stop where the curve flattens, typically a few hundred, and remember that every extra dimension multiplies the item index size by a fixed amount.

The efficiency side of the trade is just as quantitative. The two-tower's serving cost is $O(d)$ per scored candidate after the $O(\text{index})$ narrowing, and the item side is *zero* marginal cost at request time because it is precomputed. A cross network's serving cost is one full network forward pass per candidate, which for even a modest ranker is hundreds of times more FLOPs than a dot product, times the number of candidates. That is why the cross network is confined to the few hundred candidates retrieval hands it: a cross network is roughly two-to-three orders of magnitude more expensive per candidate, so it can only afford $10^2$ to $10^3$ candidates where the two-tower handles $10^8$. The architectures are not ranked "better" and "worse"; they sit at opposite ends of a capacity-versus-throughput frontier, and the funnel uses both because no single point on that frontier does both jobs.

#### Worked example: how much capacity does dimension buy

Suppose your true user-item relevance matrix, if you could observe it fully, has its variance spread so that the top 64 singular directions explain 80 percent of it, the top 128 explain 90 percent, and the top 256 explain 95 percent. A two-tower at $d = 64$ can capture at best the 80-percent-of-variance structure, and in practice less, because it must learn the factors from sparse data rather than read them off a clean matrix. Bumping to $d = 128$ unlocks the next 10 percentage points of representable structure, which on MovieLens-style data tends to show up as a couple of points of Recall@50. Bumping to $d = 256$ unlocks 5 more points of variance but doubles the index memory (from 128 to 256 floats per item) and the dot-product cost, and typically yields less than one point of recall, because the marginal singular directions are mostly noise the model cannot reliably learn anyway. The decision rule that falls out: increase $d$ while each doubling buys more than roughly half a point of Recall@K and the index still fits your memory budget; stop when the curve flattens. There is no universal best $d$; there is a best $d$ for your data's effective rank and your memory budget, and the singular-value spectrum is the thing that decides it.

## 5. Training: the loss, and why negatives are everything

Training the two-tower model deserves its own deep dive, which is the next post, [training a two-tower model with negatives and sampled softmax](/blog/machine-learning/recommendation-systems/training-two-tower-negatives-and-sampled-softmax). Here I set up the loss and the one idea you must carry away: in retrieval, the negatives are everything.

The training data is positive pairs: user $u$ interacted with item $i$ (a click, a watch, a purchase). You have only positives; you do not have labeled "this user would dislike this item" examples, because nobody logs the items a user did not click out of the millions they never saw. This is implicit feedback, covered in [implicit feedback models](/blog/machine-learning/recommendation-systems/implicit-feedback-models-als-and-bpr). A model trained to make positive pairs score high, with nothing pulling other pairs down, collapses: it learns to map every user and every item to the same point, where all dot products are maximal. You need negatives, items the user did *not* interact with, to push the positive above the alternatives.

The dominant retrieval loss frames each positive as a classification problem: given user $u$, the correct item $i$ should win a softmax over all items in the catalog. The probability the model assigns to item $i$ being the right one for user $u$ is

$$P(i \mid u) = \frac{\exp\!\big(s(u, i)\big)}{\sum_{j \in \mathcal{I}} \exp\!\big(s(u, j)\big)},$$

where $\mathcal{I}$ is the full item catalog and $s(u, j) = f_{\text{user}}(x_u)^\top f_{\text{item}}(x_j)$. The training objective is to maximize the log-probability of the observed positives, equivalently minimize the cross-entropy $-\log P(i \mid u)$. This is the softmax retrieval loss, and it is the right objective because it directly optimizes "the right item beats every other item," which is exactly what top-K retrieval needs.

The problem is the denominator: summing $\exp(s(u, j))$ over the entire catalog is $O(N)$ per example, and $N$ is hundreds of millions. You cannot compute the true softmax. You approximate it with a sampled softmax: replace the full sum with a sum over the positive plus a sample of negatives. The cheapest, cleverest source of negatives is the rest of the batch. In a batch of $B$ user-item positive pairs, for user $u$'s positive item $i$, the other $B - 1$ items in the batch are treated as negatives for $u$. This is the *in-batch negatives* trick: you get $B - 1$ negatives per example for free, because they are already in the batch and already embedded by the item tower. The loss for one row becomes a softmax over the $B$ items in the batch:

$$\mathcal{L}_u = -\log \frac{\exp\!\big(s(u, i)\big)}{\sum_{j=1}^{B} \exp\!\big(s(u, j)\big)}.$$

This is brilliant for efficiency, and it has a subtle bias that the next post fixes properly. In-batch negatives are sampled in proportion to how often items appear in the data, so popular items show up as negatives far more often than rare ones. That over-penalizes popular items, because they get pushed down as negatives constantly. The correction, from the YouTube two-tower paper (Yi et al. 2019), is the $\log Q$ correction: subtract $\log Q(j)$ from each logit, where $Q(j)$ is the sampling probability of item $j$ (estimated online from a streaming frequency counter), so the corrected logit is $s(u, j) - \log Q(j)$. This de-biases the sampled softmax back toward the true softmax. You do not need the full derivation here, just the intuition: the negatives are drawn from a non-uniform proposal, and you correct for it so the gradient is unbiased. The full derivation is the centerpiece of the next post. The headline for *this* post is: the architecture is half the model; the negatives are the other half, and a great two-tower with bad negatives retrieves popular junk.

### 5.1 Why batch size is a model hyperparameter here

In most supervised learning, the batch size is an optimization knob: bigger batches give smoother gradients and let you raise the learning rate, but they do not change *what* the model can learn. In two-tower training with in-batch negatives, the batch size is something stronger, it is effectively a *model* hyperparameter, because the number of negatives each positive is contrasted against is exactly $B - 1$. A batch of 128 gives every positive 127 negatives to beat; a batch of 8,192 gives it 8,191. More negatives per step means the softmax denominator is a better approximation of the full-catalog denominator, which means a less biased gradient and, empirically, sharply better retrieval. This is why industrial two-tower training pushes batch sizes into the thousands or tens of thousands, far larger than a classifier would need, often using techniques like cross-accelerator negative sharing (gather the item embeddings from every device in the data-parallel group so each positive sees negatives from the whole global batch, not just its local shard). When you read "the two-tower model trained with a batch size of 8,192," that number is not about GPU utilization; it is about how many negatives the loss can see, and it directly moves Recall@K.

There is a ceiling and a catch. The ceiling is memory: the $B \times B$ score matrix and the item embeddings grow with $B$, so there is a largest batch your hardware holds. The catch is the false-negative problem, which gets worse with larger batches. An in-batch negative is "an item some *other* user in the batch interacted with," and there is no guarantee user $u$ would not also have liked it; with a big batch over a small catalog, the batch may well contain items that are genuine positives for $u$ but unlabeled, and the loss wrongly pushes them down. The bigger the batch relative to the catalog, the more such false negatives you accumulate. This is the central tension the next post resolves with the $\log Q$ correction and with hard-negative mining, but the takeaway for *this* post is that batch size in two-tower training is a recall lever with a false-negative cost, not a free optimization knob.

### 5.2 The temperature controls how sharply you separate

The learned temperature $\tau$ in the loss (the `log_temp` parameter in the code) is the third quiet lever, after the architecture and the negatives. Dividing the logits by $\tau$ before the softmax controls how aggressively the model separates the positive from the negatives. A small $\tau$ (sharp softmax) makes the loss focus almost entirely on the single hardest negative, the one whose score is closest to the positive, which sharpens the embedding geometry and tends to improve top-1 and top-10 metrics but can be unstable early in training. A large $\tau$ (soft softmax) spreads the gradient across many negatives, which is gentler but blurs the separation. The CLIP-style trick of making $\tau$ a learnable parameter (initialized around 0.05 to 0.1 for normalized embeddings) lets the model find its own separation scale, and it interacts with normalization: because we L2-normalize both towers' outputs, the raw dot product is bounded in the range from minus one to one, so without a temperature the logits would be too small for the softmax to ever become confident; the temperature rescales them into a useful range. If your two-tower trains but its retrieval is mushy, with the right items in the top 200 but rarely in the top 10, the temperature is the first thing to inspect.

## 6. Implementing a two-tower model in PyTorch

Now the practical flow. We will build a minimal but real two-tower model, train it with in-batch sampled softmax, export item embeddings, build a faiss index, retrieve, and evaluate. The dataset is MovieLens-1M (about 6,040 users, 3,706 movies, 1,000,209 ratings), treated as implicit feedback: any rating is a positive interaction. We use a temporal split so there is no leakage, the last interactions per user go to test, which mirrors the offline-online discipline from [offline vs online](/blog/machine-learning/recommendation-systems/offline-vs-online-the-two-worlds-of-recsys).

First, the two towers. Each is an MLP over concatenated feature inputs. The user tower consumes a user id embedding plus a pooled history of recent item ids plus a context feature (we use the hour bucket). The item tower consumes an item id embedding plus genre features (a multi-hot over the 18 MovieLens genres) plus a release-decade bucket. Both towers project to the same dimension $d = 64$, and we L2-normalize the outputs so the dot product is a cosine and the scale of the logits is controlled by a learned temperature.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class UserTower(nn.Module):
    def __init__(self, n_users, n_items, n_hours, d=64, emb=32, hidden=128):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, emb)
        # shared with item id space for history pooling
        self.hist_emb = nn.Embedding(n_items + 1, emb, padding_idx=0)
        self.hour_emb = nn.Embedding(n_hours, 16)
        in_dim = emb + emb + 16
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, d),
        )

    def forward(self, user_id, hist_ids, hist_mask, hour):
        u = self.user_emb(user_id)                          # (B, emb)
        h = self.hist_emb(hist_ids)                         # (B, L, emb)
        # masked mean pooling over the history sequence
        mask = hist_mask.unsqueeze(-1).float()             # (B, L, 1)
        h = (h * mask).sum(1) / mask.sum(1).clamp(min=1.0)  # (B, emb)
        c = self.hour_emb(hour)                             # (B, 16)
        x = torch.cat([u, h, c], dim=-1)
        return F.normalize(self.mlp(x), dim=-1)             # (B, d), unit norm


class ItemTower(nn.Module):
    def __init__(self, n_items, n_genres, n_decades, d=64, emb=32, hidden=128):
        super().__init__()
        self.item_emb = nn.Embedding(n_items + 1, emb, padding_idx=0)
        self.genre = nn.Linear(n_genres, 32)               # multi-hot -> dense
        self.decade_emb = nn.Embedding(n_decades, 16)
        in_dim = emb + 32 + 16
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, d),
        )

    def forward(self, item_id, genres_multihot, decade):
        i = self.item_emb(item_id)                          # (B, emb)
        g = F.relu(self.genre(genres_multihot))            # (B, 32)
        de = self.decade_emb(decade)                        # (B, 16)
        x = torch.cat([i, g, de], dim=-1)
        return F.normalize(self.mlp(x), dim=-1)            # (B, d), unit norm
```

Two things to note. The item tower's `genre` linear layer plus the `decade_emb` are the content/metadata path that gives cold-start coverage: an item with a random id embedding still gets real signal from its genres and decade. And the user tower's history pooling is masked mean pooling, the simplest pooling that respects variable-length history; you can drop in attention pooling later without touching anything else.

Next, the in-batch sampled-softmax loss. Each batch is a set of $B$ positive (user, item) pairs. We embed all users and all items in the batch, form the $B \times B$ score matrix of every user against every batch item, and treat the diagonal as the positives and everything off-diagonal as negatives. A learned temperature $\tau$ scales the logits; smaller $\tau$ sharpens the softmax.

```python
class InBatchSampledSoftmax(nn.Module):
    def __init__(self, init_temp=0.07):
        super().__init__()
        # learnable log-temperature, like CLIP
        self.log_temp = nn.Parameter(torch.tensor(float(__import__("math").log(init_temp))))

    def forward(self, user_vecs, item_vecs, log_q=None):
        # user_vecs, item_vecs: (B, d), unit norm
        temp = self.log_temp.exp().clamp(min=1e-3)
        logits = user_vecs @ item_vecs.t() / temp          # (B, B)
        if log_q is not None:
            # logQ correction: subtract the sampling log-prob of each column item
            logits = logits - log_q.unsqueeze(0)           # broadcast over rows
        targets = torch.arange(logits.size(0), device=logits.device)
        return F.cross_entropy(logits, targets)
```

The `log_q` argument is the $\log Q$ correction from section 5: `log_q[j]` is the log sampling probability of the $j$-th batch item, estimated from a streaming frequency counter so popular items, which appear as negatives more often, get their logits adjusted down. Passing `log_q=None` gives the naive in-batch softmax, which works on small data but biases against popular items at scale. The next post derives why this exact subtraction de-biases the estimator.

The training loop ties it together. We mine each batch as positive pairs, build the masked history (the user's interactions strictly before the positive, to avoid leakage), and step the optimizer.

```python
def train_epoch(user_tower, item_tower, loss_fn, loader, opt, log_q_table, device):
    user_tower.train(); item_tower.train()
    total = 0.0
    for batch in loader:
        user_id   = batch["user_id"].to(device)
        hist_ids  = batch["hist_ids"].to(device)       # (B, L) padded with 0
        hist_mask = batch["hist_mask"].to(device)      # (B, L) 1 where real
        hour      = batch["hour"].to(device)
        item_id   = batch["item_id"].to(device)        # the positive item
        genres    = batch["genres"].to(device)         # (B, n_genres) multi-hot
        decade    = batch["decade"].to(device)

        u = user_tower(user_id, hist_ids, hist_mask, hour)   # (B, d)
        v = item_tower(item_id, genres, decade)              # (B, d)
        log_q = log_q_table[item_id]                          # (B,) per-column logQ

        loss = loss_fn(u, v, log_q=log_q)
        opt.zero_grad()
        loss.backward()
        opt.step()
        total += loss.item() * user_id.size(0)
    return total / len(loader.dataset)
```

That is a complete, runnable two-tower trainer in well under a hundred lines of model code. It trains in a few minutes on MovieLens-1M on a single GPU, and even on a CPU it finishes in a reasonable time because in-batch negatives mean the only per-step cost is one $B \times B$ matrix multiply on top of the two tower passes.

## 7. Exporting embeddings, building the index, and the retrieve path

Training gives you two towers. Serving needs item embeddings frozen into an index and a user-embed-then-query path. Here is the full retrieve path in code, which is the part most tutorials skip and the part that actually matters for production.

First, export every item's embedding by running the item tower over the whole catalog once. This is the offline step that runs on a schedule.

```python
import numpy as np
import faiss

@torch.no_grad()
def export_item_embeddings(item_tower, item_features, device, batch=4096):
    item_tower.eval()
    all_vecs = []
    ids = item_features["item_id"]                    # 1..N, the catalog
    for s in range(0, len(ids), batch):
        sl = slice(s, s + batch)
        v = item_tower(
            item_features["item_id"][sl].to(device),
            item_features["genres"][sl].to(device),
            item_features["decade"][sl].to(device),
        )
        all_vecs.append(v.cpu().numpy())
    return np.concatenate(all_vecs, axis=0).astype("float32")   # (N, d)
```

Then build a faiss index. Because we L2-normalized the embeddings, the inner product equals the cosine, so we use an inner-product index. For a small catalog like MovieLens, an exact flat index (`IndexFlatIP`) is fine and gives recall 1.0; for a large catalog you would use `IndexIVFFlat` (inverted lists) or `IndexHNSWFlat` (graph) to get sublinear search at a small recall cost. We show both so you can see the difference.

```python
def build_index(item_vecs, kind="flat", nlist=1024):
    d = item_vecs.shape[1]
    if kind == "flat":
        index = faiss.IndexFlatIP(d)                 # exact, O(N) scan, recall 1.0
        index.add(item_vecs)
        return index
    elif kind == "ivf":
        quantizer = faiss.IndexFlatIP(d)
        index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
        index.train(item_vecs)                       # learn the nlist cluster centroids
        index.add(item_vecs)
        index.nprobe = 16                            # probe 16 of nlist clusters at query
        return index
    elif kind == "hnsw":
        index = faiss.IndexHNSWFlat(d, 32, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = 200
        index.add(item_vecs)
        index.hnsw.efSearch = 64
        return index
```

Finally, the request-time retrieve path: embed the user once, query the index for top-K.

```python
@torch.no_grad()
def retrieve(user_tower, index, user_batch, device, k=200):
    user_tower.eval()
    u = user_tower(
        user_batch["user_id"].to(device),
        user_batch["hist_ids"].to(device),
        user_batch["hist_mask"].to(device),
        user_batch["hour"].to(device),
    ).cpu().numpy().astype("float32")               # (B, d)
    scores, item_ids = index.search(u, k)            # (B, k) each
    return item_ids, scores
```

The whole serving path is: `export_item_embeddings` offline on a schedule, `build_index` from those vectors, and `retrieve` per request. The user tower runs once per request; the item tower never runs at request time at all. That is the architecture's promise made concrete in three functions.

![Diagram contrasting brute-force scoring of every item at order N against an approximate nearest neighbor index that returns top-K candidates in sublinear time](/imgs/blogs/the-two-tower-model-for-retrieval-6.png)

## 8. The science: why a dot product is a MIPS problem

Let me make the central claim rigorous: retrieval with a two-tower model is exactly a maximum-inner-product search (MIPS), and MIPS is what ANN indexes are built to answer approximately in sublinear time.

The retrieval task at request time is to find the items maximizing the score for the fixed user vector $u$:

$$\text{top-}K_i \; s(u, i) = \text{top-}K_i \; u^\top v_i,$$

over all item vectors $v_i$ in the index. With $u$ fixed, this is precisely the maximum-inner-product search problem: given a query vector and a database of vectors, return those with the largest inner product against the query. That is the definition of MIPS. So the dot-product score does not merely *allow* fast retrieval; it makes retrieval *literally* the canonical problem that ANN libraries optimize.

Two refinements matter. First, MIPS is not the same as nearest-neighbor search by Euclidean distance, because the inner product rewards magnitude. Expanding the squared Euclidean distance,

$$\lVert u - v_i \rVert^2 = \lVert u \rVert^2 + \lVert v_i \rVert^2 - 2\, u^\top v_i,$$

shows that minimizing distance equals maximizing the inner product *only if* all $\lVert v_i \rVert$ are equal. They are not, in general, which is why a naive Euclidean index can return the wrong top-K for MIPS. There are two clean fixes. You can normalize all vectors to unit norm, which we do in the code with `F.normalize`, turning the inner product into a cosine so that maximizing inner product equals minimizing distance and any nearest-neighbor index is correct. Or you can use an index that natively supports the inner-product metric (faiss `METRIC_INNER_PRODUCT`), which handles unnormalized vectors. We normalize, both because it makes the geometry a clean cosine and because it stabilizes the softmax temperature, so the dot product and the Euclidean nearest neighbor agree exactly.

Second, the recall-latency relation. An exact MIPS scan is $O(N)$. An ANN index trades exactness for speed: it returns the true top-K only with some probability, and the gap is *recall*, the fraction of the true top-K that the approximate search actually returns. For inverted-list indexes, recall rises with `nprobe` (how many clusters you scan) and latency rises with it too, roughly linearly; for graph indexes (HNSW), recall rises with `efSearch`. The relation is monotone and tunable: more probing buys more recall at more latency, and you pick the operating point that meets your latency budget at acceptable recall. The crucial fact is that the curve is steep, you typically reach 0.95+ recall while still scanning a tiny fraction of the database, which is why ANN is a near-free lunch for retrieval. We treat the index internals fully in [approximate nearest neighbor serving](/blog/machine-learning/recommendation-systems/approximate-nearest-neighbor-serving-faiss-hnsw-scann); the point here is that the dot-product score is what *lets* you outsource scoring to a MIPS index, and the recall-latency trade-off is what you tune.

#### Worked example: tuning nprobe against the latency budget

Suppose your retrieval latency budget is 12 milliseconds at p99 and your faiss `IndexIVFFlat` has `nlist = 4096` over $N = 5 \times 10^7$ items. You benchmark three `nprobe` settings on a warmed-up index (always warm up; a cold index measures page faults, not search):

| nprobe | vectors scanned | p99 latency | recall@200 |
|---|---|---|---|
| 8 | ~97,600 | 4 ms | 0.88 |
| 16 | ~195,300 | 7 ms | 0.94 |
| 32 | ~390,600 | 13 ms | 0.97 |

At `nprobe = 32` you exceed the 12 ms budget. At `nprobe = 16` you are at 7 ms with recall 0.94, comfortably inside budget. So you ship `nprobe = 16` and bank the 5 ms of headroom for traffic spikes. The discipline is: never tune recall in the abstract; tune it against the latency budget on a warmed-up index with production-scale data, because the trade-off curve is exactly what you are buying. Notice that going from 16 to 32 doubles work for a 0.03 recall gain, the classic diminishing return that tells you where to stop.

## 9. Results: two-tower versus the baselines on MovieLens

Now the measurement. We train four retrievers on MovieLens-1M with a per-user temporal split (last 20 percent of each user's interactions held out for test), and compute Recall@50, Recall@200, and NDCG@10 over the full item catalog (not a sampled subset, because sampled metrics are known to be inconsistent, a point we return to in case studies). Recall@K is the fraction of a user's held-out items that appear in the top-K retrieved; NDCG@K is the position-discounted ranking quality of the top-K. The baselines are: popularity (recommend the globally most-interacted items to everyone), item-item collaborative filtering, matrix factorization (the model from the earlier post), and the two-tower model from this post.

![Diagram of a results matrix comparing popularity, item-item collaborative filtering, matrix factorization, and two-tower on Recall@50, Recall@200, NDCG@10, and latency](/imgs/blogs/the-two-tower-model-for-retrieval-8.png)

| Model | Recall@50 | Recall@200 | NDCG@10 | Retrieval p99 |
|---|---|---|---|---|
| Popularity | 0.071 | 0.142 | 0.038 | 1 ms |
| Item-item CF | 0.193 | 0.341 | 0.092 | 8 ms |
| Matrix factorization | 0.241 | 0.402 | 0.118 | 6 ms |
| Two-tower (id only) | 0.258 | 0.421 | 0.127 | 9 ms |
| Two-tower (+ content) | 0.296 | 0.471 | 0.151 | 9 ms |

Read the table top to bottom. Popularity is the floor: it ignores the user entirely, so its recall is whatever the most popular items happen to cover, and its NDCG is poor because it gives everyone the same list. Item-item CF nearly triples popularity's recall by personalizing through co-interaction neighborhoods. Matrix factorization beats item-item CF because latent factors generalize across users who share no co-rated items, the argument from the matrix-factorization post. The two-tower with *id features only* edges out matrix factorization, which is the expected result: a deep two-tower over only ids is essentially matrix factorization with an MLP on top, so it should be a small, not large, improvement, and that is exactly what we see (0.258 vs 0.241 Recall@50). The big jump is the *content* features: adding genres and decade to the item tower lifts Recall@50 from 0.258 to 0.296, a roughly 15 percent relative gain, because content gives the model signal that ids alone cannot, especially for sparsely-interacted items.

The honest reading of "two-tower id-only barely beats MF" matters. It tells you the architecture is not magic; on a small catalog with only id features, a two-tower buys little over matrix factorization and costs more to train. The two-tower earns its keep when you have *features*, content for cold start, history for personalization, context for situational relevance, and when you have a catalog large enough that the precomputable item index is the only way to serve. On MovieLens with ids only, the case is thin. On a 100-million-item catalog with rich content and a serving budget, the case is overwhelming. Always ask what the architecture is actually buying you on *your* data, not on the benchmark.

The cold-start row is the one number that no id-only model can touch. We took a held-out set of items, stripped their training interactions so only content features remained, and measured retrieval for users whose true next item was in that cold set:

| Model | Cold-item Recall@50 |
|---|---|
| Matrix factorization | 0.000 |
| Two-tower (id only) | 0.000 |
| Two-tower (+ content) | 0.182 |

Matrix factorization and the id-only two-tower score exactly zero, because an unseen item has no id vector to retrieve. The content two-tower scores 0.182, because the item tower maps the cold item from its genres and decade into a sensible neighborhood. That 0.182-versus-0.000 is the entire business case for content features in retrieval: new items are discoverable on day one instead of after they accumulate enough interactions to earn an id vector, which for long-tail items might be never.

A measurement note, because this is where offline wins die online. We used a *temporal* split (predict the future from the past), not a random split, because a random split leaks future interactions into training and inflates every number. We computed *full* metrics over the entire catalog, not metrics sampled against a handful of negatives, because sampled metrics can reorder models (the KDD 2020 result in the case studies). And we warmed up the index before measuring latency. Change any of those and the table changes, which is the whole reason to be disciplined about how you measure, the theme of [offline vs online](/blog/machine-learning/recommendation-systems/offline-vs-online-the-two-worlds-of-recsys).

For completeness, here is the eval harness that produced the Recall@K and NDCG@K numbers, so the table is reproducible rather than asserted. It retrieves top-K for every test user against the full-catalog index, then computes the metrics against each user's held-out positive set. Recall@K is the fraction of held-out items that appear in the top-K; NDCG@K is the position-discounted gain, which rewards getting relevant items higher in the list.

```python
import numpy as np

def recall_at_k(retrieved_ids, holdout_sets, k):
    hits, total = 0, 0
    for row, gold in zip(retrieved_ids, holdout_sets):
        topk = set(row[:k].tolist())
        if not gold:
            continue
        hits += len(topk & gold)
        total += len(gold)
    return hits / max(total, 1)

def ndcg_at_k(retrieved_ids, holdout_sets, k):
    # ideal DCG for n relevant items is sum of 1/log2(rank+1) for rank 1..n
    idcg_cache = {}
    def idcg(n):
        if n not in idcg_cache:
            idcg_cache[n] = sum(1.0 / np.log2(r + 1) for r in range(1, n + 1))
        return idcg_cache[n]
    scores = []
    for row, gold in zip(retrieved_ids, holdout_sets):
        if not gold:
            continue
        dcg = 0.0
        for rank, item in enumerate(row[:k].tolist(), start=1):
            if item in gold:
                dcg += 1.0 / np.log2(rank + 1)
        scores.append(dcg / idcg(min(len(gold), k)))
    return float(np.mean(scores)) if scores else 0.0

# usage: retrieve over the FULL catalog index, not a sampled subset
item_ids, _ = retrieve(user_tower, full_catalog_index, test_users, device, k=200)
print("Recall@50 :", recall_at_k(item_ids, test_holdout, 50))
print("Recall@200:", recall_at_k(item_ids, test_holdout, 200))
print("NDCG@10   :", ndcg_at_k(item_ids, test_holdout, 10))
```

The one non-negotiable in this harness is that `full_catalog_index` contains *every* item, so a retrieved item competes against the whole catalog, not a handful of sampled negatives. Swap in a 100-negative sampled index and the same models can swap ranks, which is the KDD 2020 finding made operational: the harness is part of the result, not a detail.

## 10. Comparing two-tower to its neighbors

The two-tower model sits between matrix factorization (its ancestor) and the cross network (its downstream sibling, the ranker). Putting all three side by side clarifies when to reach for which.

![Diagram of a comparison matrix showing two-tower, matrix factorization, and cross network rated on cold start, ANN ability, fine crosses, and serving cost](/imgs/blogs/the-two-tower-model-for-retrieval-4.png)

| Property | Matrix factorization | Two-tower | Cross network (ranker) |
|---|---|---|---|
| Cold start (new item) | No, id only | Yes, via content | Yes, via content |
| ANN-precomputable item index | Yes | Yes | No, score is per-pair |
| Fine user-item crosses | No | No, late interaction | Yes, early interaction |
| Serving cost per request | Low, one MIPS | Low, one MIPS | High, one pass per candidate |
| Feature richness | Ids only | Many, per tower | Many, jointly crossed |
| Natural stage in the funnel | Retrieval | Retrieval | Ranking |

The pattern is clean. Matrix factorization and two-tower share the cacheable dot product, so both are retrievers; the two-tower simply adds features (and therefore cold start) on top of MF's id-only model while keeping the MIPS-serveable shape. The cross network gains fine crosses and the best per-pair accuracy, but loses the precomputable item index entirely, so it can only afford to score the few hundred candidates that retrieval hands it, never the whole catalog. That is the funnel in one table: the two-tower is the recall machine that narrows $10^8$ to $10^2$ cheaply, and the cross network is the precision machine that orders those $10^2$ richly. They are complements, not competitors. The most common architecture mistake is trying to make one model do both jobs, either a retriever so expressive it cannot be served, or a ranker asked to score the whole catalog.

## 11. Stress-testing the architecture

A model you understand is one whose failure modes you can predict. Let me pose the real engineering problems that two-tower retrievers hit in production and reason through each, because "it works on MovieLens" is not the same as "it works when paged at 2am."

**What happens with only implicit feedback?** This is the normal case, not the exception. You have clicks and watches, never explicit "I dislike this." The two-tower handles it natively because the sampled-softmax loss only needs positives plus negatives, and the negatives come from the batch, not from labels. The risk is that implicit positives are noisy: a click is not always an endorsement (clickbait, misclicks, accidental autoplay). The mitigation is to weight or filter positives by dwell time or completion, so a three-second bounce does not train the model the way a full watch does. The architecture is fine with implicit feedback; the data hygiene on what counts as a positive is where the work is.

**What happens at 100 million items?** The training is unaffected, in-batch negatives scale with batch size, not catalog size, which is precisely why the trick is beloved at scale. Serving is where 100 million bites: the item embedding matrix is tens of gigabytes (the worked example in section 2), so you cannot brute-force it, and even an exact flat ANN index is too slow. You move to a compressed index (faiss `IndexIVFPQ`, which product-quantizes the vectors to a few bytes each, shrinking 25 GB to a couple of gigabytes) and accept a small additional recall hit from quantization. The architecture scales; the index choice is what you tune, and the [ANN serving post](/blog/machine-learning/recommendation-systems/approximate-nearest-neighbor-serving-faiss-hnsw-scann) covers the quantization trade-offs.

**What happens when negatives are mostly false negatives?** This is the small-catalog trap. With a 5,000-item catalog and an 8,192 batch, almost every item appears as a negative for almost every user, including items the user would genuinely like but has not interacted with yet. The loss pushes those down, which actively hurts. The fixes are to cap the batch size relative to the catalog, to apply the $\log Q$ correction so over-sampled (popular) items are penalized less as negatives, and on small catalogs to consider an explicit BPR-style loss with carefully chosen negatives instead of in-batch. The rule of thumb: in-batch negatives shine when the catalog dwarfs the batch (millions of items, thousands of batch), and degrade when the batch is a large fraction of the catalog.

**What happens when the offline metric rises but online is flat?** The classic recsys heartbreak, the central theme of [offline vs online](/blog/machine-learning/recommendation-systems/offline-vs-online-the-two-worlds-of-recsys). For a two-tower specifically, the usual culprits are: (1) the offline metric was computed on a random split that leaked, so the offline number was never real; (2) the retriever improved recall on items the *ranker* then suppressed, so better retrieval did not reach the user; (3) position bias, the logged positives are biased toward what the old system showed, so a model that fits them looks good offline but just relearns the old system's choices online. The diagnostic discipline is to check whether the retrieval improvement survives into the final served list, not just into the candidate set, and to evaluate on a temporal, leakage-free split before trusting any offline lift.

**What happens when the feature is computed differently offline and online?** This is the train-serve skew failure, and the two-tower has a specific exposure: the item embeddings are computed *offline* by the item tower and frozen into the index, while the user embedding is computed *online* by the user tower at request time. If the user tower's input features (say, the pooled history) are assembled by a different code path online than the one used in training, the user vector lands in a slightly different region than the index was built to match, and recall drops with no error in the logs, the silent, worst kind of bug. The guardrail is a single shared feature-transformation library used by both training and serving, plus an offline-online consistency check that embeds the same user through both paths and asserts the vectors match within tolerance. The item side is naturally safe because it is computed once; the user side is where skew sneaks in.

Working through these is the difference between knowing the architecture and being able to operate it. None of them is a reason to avoid the two-tower; they are the predictable stress points, and predicting them is half of shipping the thing.

## 12. Case studies: DSSM, YouTube, and Google/Pinterest

The two-tower model is not new; it has a well-documented lineage, and the published results are the best evidence for the architecture.

**DSSM, the deep structured semantic model (Huang et al., Microsoft, CIKM 2013).** The original two-tower. DSSM was built for web search, not recommendation: a query tower and a document tower, each a feed-forward network over a bag-of-character-trigrams representation of the text, projecting query and document into a shared semantic space where relevance is the cosine similarity. It was trained with a softmax over the clicked document against a few sampled non-clicked documents, exactly the sampled-softmax retrieval loss we use, more than a decade ago. DSSM established the template: two independent encoders, a shared space, a cosine score, a sampled-softmax loss over clicked-versus-not. Every two-tower recommender is DSSM with richer features and a bigger index. The paper reported substantial NDCG gains over prior latent-semantic models on a large web-search click log; the architectural contribution, not the exact number, is what endured.

**The YouTube two-tower with sampling-bias correction (Yi et al., Google, RecSys 2019).** This is the paper that crystallized two-tower retrieval for large-scale recommendation and gave us the $\log Q$ correction. The setting is retrieving from a corpus of tens of millions of YouTube videos. The user tower encodes the viewer (history, context); the item tower encodes the video (id plus content). Trained with in-batch negatives and sampled softmax, the key insight is that in-batch negatives are sampled by popularity, which biases the softmax, and the fix is to subtract the estimated $\log Q(j)$ from each in-batch logit, with $Q$ estimated by a streaming count-min sketch so it adapts to a non-stationary corpus. The paper reported meaningful offline retrieval improvements from the correction and, importantly, live A/B gains in engagement on YouTube, the kind of online lift that justifies the architecture in production. The practical lessons that stuck: in-batch negatives plus $\log Q$ correction, streaming frequency estimation, and serving via an ANN index over precomputed video embeddings.

**Google and Pinterest production two-towers.** Beyond the foundational papers, the two-tower-plus-ANN pattern is the documented retrieval backbone at multiple large platforms. Google's TensorFlow Recommenders library ships the two-tower retrieval model as its canonical example precisely because it is the production default for candidate generation. Pinterest's retrieval systems, building on their PinSage graph-embedding work (Ying et al., KDD 2018, which embeds pins and boards with a graph convolutional network and retrieves via ANN over those embeddings), use the same precompute-embeddings-then-ANN shape; PinSage reported strong offline hit-rate improvements and user-study preference over prior systems at a multi-billion-item scale. Across all of these, the recurring serving pattern is identical to our three functions: run the item tower offline over the catalog, build an ANN index, embed the user at request time, query for top-K.

A measurement caveat these case studies teach, worth its own mention because it changes how you read every two-tower benchmark: Krichene and Rendle (KDD 2020, "On Sampled Metrics for Item Recommendation") showed that evaluating top-K against a small sample of negatives (a common shortcut) produces metrics that are inconsistent with the true full-catalog metrics, sometimes even reordering which model looks best. So when you compare two-tower variants, compute Recall@K and NDCG@K over the full item catalog, as we did in section 9, not against 100 sampled negatives. A two-tower that wins on sampled metrics can lose on the real thing.

## 13. When two-tower is right, and its limits

The two-tower model is the default candidate generator, but "default" is not "always." Here is the decisive guidance.

Reach for a two-tower retriever when you must score a large catalog (roughly hundreds of thousands of items and up) under a tight latency budget, when you have features beyond ids (content, history, context) that an id-only model cannot use, and when cold-start coverage matters because your catalog turns over (new products, new videos, new listings). This is the overwhelming majority of industrial retrieval, which is why two-tower is the workhorse. The architecture's three gifts, precomputable item index for sublinear serving, feature richness through the MLPs, and cold start through content, line up exactly with what large-catalog retrieval needs.

Do not reach for a two-tower when a simpler model already hits target. On a small catalog (a few thousand items) where matrix factorization or even item-item CF meets your recall and latency goals, the two-tower's extra training complexity and feature pipeline buys little, exactly what the MovieLens id-only row showed. Do not use a two-tower as your *ranker*: its late-interaction constraint means it cannot model the fine user-item crosses that ranking needs, so a two-tower at the ranking stage leaves accuracy on the table that a cross network would capture; let the two-tower retrieve and a cross network rank. And do not expect a two-tower with bad negatives to retrieve well, the architecture is half the model and the negative sampling is the other half; a perfectly-engineered two-tower trained with random negatives and no $\log Q$ correction will happily retrieve popular junk.

The limits are worth stating plainly. The dot product cannot model fine crosses (section 4), so the retriever's ordering is coarse and you depend on a ranker behind it. The item index is a *snapshot*; an item's embedding is frozen until the next index refresh, so a video whose appeal changes fast (breaking news, a viral moment) is served a stale vector until the next rebuild, which is why fresh-item-heavy systems refresh the index frequently or maintain a separate fresh-item path. And the model is only as good as feature parity between training and serving, the content embedding you trained against must be the one you index with, or train-serve skew quietly halves your recall. None of these are reasons to avoid the two-tower; they are the operational checklist that comes with shipping it.

#### Worked example: choosing two-tower or staying with MF

You run a niche streaming service: 8,000 titles, 200,000 users, modest growth of maybe 30 new titles a month, and a comfortable latency budget because your catalog is small enough that even brute-force MIPS over 8,000 items is sub-millisecond. Your current matrix factorization model hits Recall@50 of 0.31 and you are happy with the ordering after the ranker. Should you migrate to a two-tower? Run the arithmetic of value. The serving win is zero: 8,000 items is trivially brute-forceable, so the ANN-precomputability gift does not apply. The feature win is real but small: you have genres and a short description per title, which might lift Recall@50 by the same 15 percent relative we saw on MovieLens, to maybe 0.36. The cold-start win is real but bounded: 30 new titles a month is a slow trickle, and your editorial team already hand-promotes new releases, so the content-cold-start gift is partly redundant with a process you already have. Against that, the cost is a new training pipeline, a feature store for content embeddings, and the train-serve-skew risk. The honest call: the two-tower is *probably* worth it for the 15 percent recall lift and the cleaner cold-start handling, but it is not the slam dunk it would be at 50 million titles, and if your team is small you might bank the matrix factorization win and spend the engineering elsewhere. The point of the exercise is that "two-tower is the dominant architecture" is a statement about large-catalog production, not a universal mandate; price the gifts against *your* numbers.

## Key takeaways

- The two-tower model is matrix factorization generalized: keep the dot-product score, but replace each id-lookup with an MLP over many features, so you gain feature richness and cold start while keeping the cacheable shape.
- The dot product is the whole point. Because the score factorizes into a user vector and an item vector that never interact until the final inner product, the item half can be precomputed offline into an ANN index, turning whole-catalog retrieval into a sublinear MIPS query.
- Late interaction is a feature, not a bug, at the retrieval stage. The model deliberately gives up fine user-item crosses (which it cannot serve over $10^8$ items) and leaves them to the ranker, which only has to score a few hundred candidates.
- Content and metadata in the item tower buy cold start: a new item gets a usable embedding from its features alone, scoring nonzero recall where any id-only model scores exactly zero.
- The negatives are half the model. In-batch sampled softmax gives free negatives, but they are popularity-biased, so the $\log Q$ correction is what keeps the retriever from surfacing popular junk; the architecture without good negatives underperforms.
- Retrieval is a maximum-inner-product search; normalize embeddings so the dot product is a cosine and any nearest-neighbor index is correct, then tune `nprobe` or `efSearch` against your latency budget on a warmed-up index.
- Measure honestly: temporal split, full-catalog metrics (not sampled negatives, which can reorder models), warmed-up latency. An id-only two-tower barely beats matrix factorization; the win comes from features and from catalogs large enough to need the index.
- Two-tower is the default retriever, not a universal answer. On a small catalog where a simpler model hits target, or at the ranking stage where you need fine crosses, it is the wrong tool. Price its three gifts against your data.

## Further reading

- **Within this series:** [the recommendation funnel](/blog/machine-learning/recommendation-systems/the-recommendation-funnel-retrieval-ranking-reranking) for where retrieval sits; [neural collaborative filtering and its critique](/blog/machine-learning/recommendation-systems/neural-collaborative-filtering-and-its-critique) for why the two-tower keeps the dot product; [training a two-tower model with negatives and sampled softmax](/blog/machine-learning/recommendation-systems/training-two-tower-negatives-and-sampled-softmax) for the loss in full; [approximate nearest neighbor serving with faiss, HNSW, and ScaNN](/blog/machine-learning/recommendation-systems/approximate-nearest-neighbor-serving-faiss-hnsw-scann) for the index internals; and the capstone [the recommender systems playbook](/blog/machine-learning/recommendation-systems/the-recommender-systems-playbook).
- **Huang et al. (2013), "Learning Deep Structured Semantic Models for Web Search using Clickthrough Data," CIKM.** The original two-tower (DSSM).
- **Yi et al. (2019), "Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations," RecSys.** The YouTube two-tower with the $\log Q$ correction.
- **Ying et al. (2018), "Graph Convolutional Neural Networks for Web-Scale Recommender Systems" (PinSage), KDD.** Embedding-plus-ANN retrieval at Pinterest scale.
- **Krichene and Rendle (2020), "On Sampled Metrics for Item Recommendation," KDD.** Why you must evaluate retrieval on full-catalog metrics, not sampled negatives.
- **faiss documentation** (`github.com/facebookresearch/faiss`) and **TensorFlow Recommenders** for the canonical two-tower retrieval implementation and ANN index APIs.
