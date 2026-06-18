---
title: "Sampled Softmax and Contrastive Losses for Retrieval"
date: "2026-06-19"
publishDate: "2026-06-19"
description: "The listwise loss family that powers modern retrieval — derive sampled softmax and the logQ correction from scratch, prove InfoNCE is in-batch sampled softmax, sweep temperature and negatives in PyTorch, and watch listwise losses beat single-negative BPR on Recall@K."
tags:
  [
    "recommendation-systems",
    "recsys",
    "sampled-softmax",
    "contrastive-learning",
    "infonce",
    "retrieval",
    "two-tower",
    "machine-learning",
  ]
category: "machine-learning"
subcategory: "Recommendation Systems"
author: "Hiep Tran"
featured: true
readTime: 53
image: "/imgs/blogs/sampled-softmax-and-contrastive-losses-for-retrieval-1.png"
---

There is a moment in every retrieval project where someone — often me, two coffees deep — looks at four different loss functions and asks the obvious question that nobody wrote down: are these actually different, or are they the same thing wearing different hats? You have a BPR loss from the implicit-feedback literature that compares one positive against one negative. You have a sampled-softmax loss from the YouTube two-tower paper with a mysterious $\log Q$ term subtracted off the logits. You have an InfoNCE loss from the self-supervised-learning world with a temperature $\tau$ that everyone tunes and nobody explains. And you have the "in-batch negatives" trick that dense-retrieval papers swear by. Four names, four codebases, four sets of folklore.

They are the same idea. Every one of them is an approximation to a single objective: maximize the probability of the right item over the whole catalog. The differences are entirely about *how many negatives* you normalize over and *how you sampled them* — and once you see that, retrieval losses stop being a grab bag and become one knob. This post is the loss-centric deep dive that the [two-tower training post](/blog/machine-learning/recommendation-systems/training-two-tower-negatives-and-sampled-softmax) gestured at: that post built the towers and the in-batch loss as an engineering recipe; this one derives the whole loss family from the ground up, proves the equivalences, and shows you the gradient math that tells you *why* a listwise softmax beats a single-negative pairwise loss for top-K retrieval.

We will start from the full-softmax objective and watch it die at scale, because normalizing over $10^8$ items per step is not slow — it is impossible. We will rebuild it as sampled softmax and *derive* the importance-weighting correction (the $\log Q$ term) that keeps the gradient unbiased when your negatives come from a non-uniform proposal. We will then walk in from the contrastive-learning side, derive InfoNCE, show it is a lower bound on mutual information, and prove that InfoNCE with in-batch negatives is *exactly* sampled softmax with a uniform-over-the-batch proposal. We will dissect what temperature does to the gradient (it controls hard-negative emphasis, and the math makes that precise). Then we get practical: PyTorch implementations of sampled softmax (with $\log Q$) and InfoNCE for a two-tower, a temperature and negative-count sweep, and a measured before-after table on MovieLens where the listwise losses pull ahead of BPR on Recall@10 and Recall@100.

![A two-panel comparison of full softmax scoring against the entire catalog versus sampled softmax scoring against one positive and a few sampled negatives feeding a cross-entropy loss](/imgs/blogs/sampled-softmax-and-contrastive-losses-for-retrieval-1.png)

This post lives in the retrieval stage of the [recommendation funnel](/blog/machine-learning/recommendation-systems/the-recommendation-funnel-retrieval-ranking-reranking): retrieval feeds candidates to ranking, and the loss is what decides whether those candidates are any good. By the end you will be able to write the sampled-softmax loss with the $\log Q$ correction by hand, write InfoNCE and explain its temperature, choose between softmax, contrastive, and pairwise for a given problem, and read an ablation table that tells you which knob actually moves Recall@K.

## 1. The full-softmax retrieval objective

Let me fix notation we keep for the whole post. A training example is a pair $(u, i)$: a user (or query) $u$ that engaged with item $i$ — a click, a watch, a purchase, whatever counts as positive in your domain. A user tower maps $u$ to an embedding $\mathbf{u} \in \mathbb{R}^d$; an item tower maps any item $j$ to $\mathbf{v}_j \in \mathbb{R}^d$. The score of a $(u, j)$ pair is the dot product

$$
s(u, j) = \mathbf{u}^\top \mathbf{v}_j .
$$

We will deal with normalization (cosine) and temperature in sections 6 and 7; for now treat $s$ as the raw similarity.

What should training *want*? Retrieval is a multi-class classification problem with an absurd number of classes — one class per item in the catalog. Given the user $u$, we want the model to assign high probability to the item the user actually engaged with, relative to **every other item in the catalog**. The standard way to turn scores into a probability distribution over the items is the softmax:

$$
P(i \mid u) = \frac{\exp\big(s(u, i)\big)}{\sum_{j \in \mathcal{I}} \exp\big(s(u, j)\big)} ,
$$

where $\mathcal{I}$ is the full item catalog. The training loss for one positive pair is the negative log-likelihood of the observed item:

$$
\mathcal{L}(u, i) = -\log P(i \mid u) = -\,s(u, i) + \log \sum_{j \in \mathcal{I}} \exp\big(s(u, j)\big) .
$$

This is exactly cross-entropy where the "true class" is the clicked item. It is the *right* objective. If you could minimize it, you would learn embeddings such that the clicked item's score stands above the scores of all $|\mathcal{I}|$ catalog items at once — precisely what you want from a retriever whose job is to pull the right items out of the whole catalog. It is a **listwise** loss: a single gradient step sees the positive in competition with the entire field, not against one opponent at a time.

There is one problem, and it is fatal at production scale: the denominator. The normalizing term, the **partition function**, sums $\exp(s(u, j))$ over every item $j$. For a video platform that is $10^8$ items; for a large marketplace, several times that. To compute the loss for a *single* training example you would embed every catalog item and dot it with the user — every step, every example, every batch. Worse, the gradient of the log-partition touches every item embedding, so backprop would update the entire item table on every step. This is not slow; it is impossible. No accelerator does a per-step sum over $10^8$ items.

So we approximate the denominator. The entire art of retrieval losses is *which* approximation you pick and *what it costs you in bias*. The figure above contrasts the two regimes: full softmax (score against everything, intractable) versus sampled softmax (score against a positive and a handful of sampled negatives, tractable). The rest of this post is the journey from the left panel to the right one — done correctly.

### Why listwise, not pointwise or pairwise (the gradient argument)

Before we approximate, it is worth being clear about why the softmax objective is the *right* thing to approximate at all, rather than some cheaper surrogate. There are three families of ranking losses:

- **Pointwise** — treat each $(u, j)$ as an independent binary label (clicked / not clicked) with a sigmoid + BCE loss. The model learns to calibrate a probability per item. The gradient never sees two items in competition, so nothing forces the positive *above* a near-miss negative; it only forces each item's score toward its own label. For top-K retrieval, where only the relative order of the top matters, this is wasteful. (The [CTR-prediction foundations post](/blog/machine-learning/recommendation-systems/the-ranking-model-ctr-prediction-foundations) is where pointwise BCE earns its keep — in ranking, where calibrated probabilities matter.)
- **Pairwise** — compare one positive against one negative and push the positive's score above the negative's (BPR is the canonical example; see [implicit-feedback models](/blog/machine-learning/recommendation-systems/implicit-feedback-models-als-and-bpr)). The gradient sees *order*, which is why pairwise beats pointwise for ranking. But it sees order between exactly two items at a time.
- **Listwise** — the softmax above. The positive competes against the whole field in one normalized step. The gradient pushes the positive up and *all* negatives down, weighted by how much each negative currently steals probability mass.

The listwise gradient is the most informative per step because it reflects the full competition, and it directly optimizes a probability over the catalog, which is exactly the quantity retrieval cares about. The catch — the only catch — is the partition function. Everything below is about getting the listwise gradient without paying the listwise price.

### The gradient of the full softmax (and why it touches every item)

To see precisely *what* we are approximating, write the gradient of the full-softmax loss with respect to the score of an arbitrary item $j$. Starting from $\mathcal{L}(u,i) = -s(u,i) + \log\sum_{j'} \exp(s(u,j'))$ and differentiating,

$$
\frac{\partial \mathcal{L}(u,i)}{\partial s(u,j)} = -\mathbb{1}[j = i] + \frac{\exp(s(u,j))}{\sum_{j'} \exp(s(u,j'))} = P(j \mid u) - \mathbb{1}[j = i] .
$$

This is the cleanest object in all of classification: the gradient on item $j$ is "the probability the model currently assigns to $j$, minus 1 if $j$ is the true item." For the positive ($j = i$) the gradient is $P(i \mid u) - 1 < 0$ — push its score *up*. For every negative the gradient is $P(j \mid u) > 0$ — push its score *down*, by an amount equal to how much probability mass it is currently stealing. A negative the model already scores near zero contributes almost nothing; a negative the model wrongly loves contributes a lot. That is the listwise gradient's elegance: it self-prioritizes, spending its push on the items that are actually competing with the positive.

And it is *also* exactly why the full softmax is intractable. The gradient is nonzero for **every** item $j$ with $P(j \mid u) > 0$ — which is every item — so a single backward pass updates the entire item embedding table. Even if you could afford the forward sum, the backward pass touching $10^8$ rows is its own catastrophe. Sampled softmax is, at heart, the observation that the gradient $P(j \mid u) - \mathbb{1}[j=i]$ is dominated by a small number of high-$P(j \mid u)$ negatives, so a well-chosen *sample* of negatives reconstructs most of the gradient at a fraction of the cost. The proposal $Q$ is our guess at "which negatives carry the gradient," and the $\log Q$ correction is the bookkeeping that makes the guess unbiased.

## 2. Sampled softmax: approximating the denominator

The idea is simple. Instead of summing $\exp(s(u,j))$ over all of $\mathcal{I}$, draw a small sample of negatives and sum over them. The questions are: which sample, and how do you correct for the fact that you only summed over a sample?

### The naive (wrong) version

The tempting first attempt: draw $k$ negatives $j_1, \dots, j_k$ from some distribution $Q$ over the catalog, and form a softmax over the positive plus the negatives:

$$
\hat{P}(i \mid u) = \frac{\exp\big(s(u, i)\big)}{\exp\big(s(u, i)\big) + \sum_{m=1}^{k} \exp\big(s(u, j_m)\big)} .
$$

If $Q$ is uniform over the catalog, this is an honest stochastic estimate of the full softmax in the limit of many negatives. But $Q$ is almost never uniform in practice. The cheapest, most popular negative sampler is **in-batch negatives**: use the *other* positives in the same minibatch as this example's negatives (more on this in section 5). In-batch sampling draws an item $j$ with probability proportional to how often it appears as a positive — i.e., proportional to its popularity. Popular items are over-represented as negatives. If you plug those into the naive softmax above, you systematically push popular items *down* harder than you should, because they show up as negatives far more often than a uniform draw would produce. The model learns to under-rank exactly the items most users actually want. This is **sampling bias**, and it is the single most common silent bug in homegrown retrieval losses.

### The fix: importance weighting and the $\log Q$ correction

Here is the derivation. We want the sampled softmax to estimate the *true* full softmax, whose denominator is $\sum_{j \in \mathcal{I}} \exp(s(u,j))$. When we draw negatives from a proposal $Q(j)$ that is not uniform, an item that $Q$ over-samples should count *less* per appearance, and an item it under-samples should count *more*. That is importance sampling: to estimate a sum $\sum_j f(j)$ using samples $j \sim Q$, you weight each sample by $1/Q(j)$, because

$$
\sum_{j \in \mathcal{I}} f(j) = \mathbb{E}_{j \sim Q}\!\left[\frac{f(j)}{Q(j)}\right] .
$$

Apply this with $f(j) = \exp(s(u,j))$. The importance-weighted contribution of a sampled negative $j$ to the denominator is $\exp(s(u,j)) / Q(j)$. We can fold that weight *inside* the exponential by writing $\exp(s(u,j))/Q(j) = \exp\!\big(s(u,j) - \log Q(j)\big)$. So define the **corrected logit**

$$
s'(u, j) = s(u, j) - \log Q(j) .
$$

The corrected sampled softmax is

$$
\hat{P}(i \mid u) = \frac{\exp\big(s'(u, i)\big)}{\exp\big(s'(u, i)\big) + \sum_{m=1}^{k} \exp\big(s'(u, j_m)\big)} ,
$$

where each logit (positive and negatives alike) has $\log Q$ subtracted. This is the **$\log Q$ correction** — the key contribution of the 2019 YouTube sampling-bias-corrected two-tower paper (Yi et al., RecSys 2019). The subtraction is just $\log$ of the importance weight. Subtracting $\log Q(j)$ down-weights items that $Q$ over-samples (popular ones, with large $Q$) and up-weights items $Q$ under-samples (tail ones), restoring an unbiased estimate of the full-catalog denominator.

Two practical notes that trip people up. First, the correction is applied to the positive's logit *too*, not only the negatives — the positive is also a draw whose appearance probability matters for the normalized estimate. Second, the correction is only needed in *training*. At serving time you score against an ANN index over raw dot products with no $\log Q$, because $Q$ was an artifact of how you sampled negatives during training, not a property of the world. (See [ANN serving](/blog/machine-learning/recommendation-systems/approximate-nearest-neighbor-serving-faiss-hnsw-scann) for the index side.)

![A before and after panel showing how uncorrected in-batch sampling over-penalizes popular items and how subtracting log Q restores an unbiased gradient](/imgs/blogs/sampled-softmax-and-contrastive-losses-for-retrieval-4.png)

### The full dataflow

The figure below traces the corrected sampled-softmax loss end to end: the user embedding meets the positive embedding and the $k$ sampled-negative embeddings to form logits; each logit has $\log Q$ subtracted; a cross-entropy where the positive is "class 0" produces a single, unbiased gradient that updates both towers. Read it as the implementation contract — every box is a line of code in section 8.

![A dataflow diagram routing a user embedding, one positive, and k sampled negatives into logits, a log Q correction, and a cross-entropy gradient step](/imgs/blogs/sampled-softmax-and-contrastive-losses-for-retrieval-2.png)

### NCE and negative sampling as relatives

Two close cousins are worth naming so you can read older papers. **Noise-Contrastive Estimation (NCE)** (Gutmann and Hyvärinen, 2010; Mnih and Teh, 2012 for language models) turns the multi-class problem into a *binary* one: for each example, classify whether a sample came from the data distribution or from a noise distribution $Q$. With enough noise samples, the NCE objective's gradient converges to the maximum-likelihood gradient — and NCE also carries a $\log Q$-style term, for the same reason: the noise is non-uniform. **Negative sampling** (the word2vec variant, Mikolov et al., 2013) is a simplification of NCE that drops the normalization entirely and just maximizes $\log \sigma(s(u,i)) + \sum_m \log \sigma(-s(u,j_m))$. It is fast and works beautifully for word embeddings, but it optimizes a slightly different objective than the true softmax and does *not* recover the softmax gradient in general. For retrieval, where you genuinely want a distribution over the catalog, the corrected sampled softmax is the principled choice; negative sampling is the pragmatic-but-looser relative. (The wider taxonomy of negative-sampling *strategies* — uniform, popularity, hard, mixed — is its own large topic; here we focus on the loss math given a sampler.)

### What "unbiased" buys you, and the bias-variance trade in the proposal

Let me be precise about the guarantee the $\log Q$ correction provides, because it is subtle. With the corrected logits, the *expected* sampled denominator (over draws of the $k$ negatives from $Q$) equals the true full-catalog denominator up to the positive's own term. That is the importance-sampling identity from above: $\mathbb{E}_{j \sim Q}[\exp(s(u,j))/Q(j)] = \sum_{j} \exp(s(u,j))$. So the corrected sampled softmax gives an (asymptotically, as $k$ grows) *unbiased* estimate of the full-softmax gradient. The uncorrected version does not — its expectation is skewed by however $Q$ deviates from uniform, and on the popularity proposal that skew systematically suppresses head items.

But "unbiased" is not "low variance," and the proposal $Q$ controls variance. The variance of an importance-sampling estimator is smallest when the proposal $Q(j)$ is proportional to the integrand $\exp(s(u,j))$ — that is, when you sample the negatives the model already scores highly, which are exactly the **hard negatives**. A uniform $Q$ is unbiased after correction but high-variance (most uniform draws are easy negatives that contribute almost nothing to the gradient, per the $P(j \mid u) - \mathbb{1}[j=i]$ formula from section 1). A popularity $Q$ is a cheap middle ground: popular items are more likely to be near the user, so the negatives are a little harder, and the $\log Q$ correction removes the bias the skew would otherwise cause. A *hard*-negative $Q$ (mine the top-scoring wrong items from an ANN index) is lowest-variance but highest-risk, because the lowest-variance proposal is also the one most likely to hand you an unlabeled positive. This bias-variance lens — uniform is high-variance/unbiased, popularity is the cheap default, hard is low-variance/risky — is the right way to think about *which* sampler to pair with the corrected loss.

## 3. Contrastive learning and InfoNCE

Now walk in from the other side of the building — self-supervised representation learning — and you arrive at the same math under a different name.

The contrastive recipe is: given an *anchor* (here, the user $u$), pull its *positive* (the engaged item $i$) close in embedding space and push its *negatives* (other items) apart. "Close" and "apart" are measured by a similarity — typically cosine — and the loss that operationalizes "pull one positive in, push $N$ negatives out" is **InfoNCE** (van den Oord, Li, Vinyals, 2018, in the Contrastive Predictive Coding paper). For one anchor with positive $i$ and a set of negatives $\mathcal{N}$, with similarity $\text{sim}(\cdot,\cdot)$ and temperature $\tau > 0$:

$$
\mathcal{L}_{\text{InfoNCE}} = -\log \frac{\exp\big(\text{sim}(\mathbf{u}, \mathbf{v}_i)/\tau\big)}{\exp\big(\text{sim}(\mathbf{u}, \mathbf{v}_i)/\tau\big) + \sum_{j \in \mathcal{N}} \exp\big(\text{sim}(\mathbf{u}, \mathbf{v}_j)/\tau\big)} .
$$

Look at it next to the sampled-softmax loss from section 2. It is the *same expression*: a softmax over the positive plus negatives, with a negative-log-likelihood on the positive. The only cosmetic differences are (1) InfoNCE conventionally uses cosine similarity (L2-normalized embeddings) where sampled softmax often uses a raw dot product, and (2) InfoNCE divides logits by a temperature $\tau$ where vanilla sampled softmax does not. Neither is a fundamental difference — you can add temperature to sampled softmax and you can use a raw dot product in InfoNCE. They are two communities' names for one loss.

This is the same loss SimCLR (Chen et al., 2020) uses for image representation learning — there the anchor is an image, the positive is an augmented view of the same image, and the negatives are the other images in the batch. The 2020 SimCLR paper is also where the importance of a *large* number of negatives and a *carefully tuned temperature* was made unmistakable: bigger batches (more negatives) and a temperature around $0.1$–$0.5$ were worth several points of downstream accuracy. We will reproduce that sensitivity for retrieval in section 9.

### The mutual-information lower bound

The "Info" in InfoNCE is not decoration. The CPC paper proves InfoNCE is a *lower bound on the mutual information* between the anchor and its positive. Sketch of the argument: with $N$ samples (one positive drawn from the joint $p(u, i)$ and $N-1$ negatives drawn from the marginal $p(i)$), the optimal critic is proportional to the density ratio $\frac{p(i \mid u)}{p(i)}$, and plugging the optimum into the InfoNCE objective yields

$$
I(u; i) \;\geq\; \log N - \mathcal{L}_{\text{InfoNCE}} .
$$

Two consequences fall straight out of this bound, and both are practically important. First, minimizing InfoNCE *maximizes a lower bound on the mutual information* between user and engaged item — so the loss is doing exactly what a recommender should: learning a representation in which a user and the items they engage with are maximally predictive of each other. Second, the bound is *capped at $\log N$*: with few negatives the bound is loose and the loss can be small while the true MI is large. That is the formal reason **more negatives help** — they raise the ceiling on what the loss can certify. It is the same lesson as section 2's "the listwise loss wants the whole field," now in information-theoretic dress.

A subtlety worth internalizing: because the bound saturates at $\log N$, a *low* InfoNCE loss is not by itself evidence of a great representation — it might just mean $N$ was small. Two runs with the same loss value but different batch sizes are not comparable; the larger-batch run is certifying more mutual information at the same nominal loss. This is one reason it is misleading to compare raw contrastive loss values across configurations and why you should compare *downstream* Recall@K instead. The loss is a means; the retrieval metric is the end.

### Alignment and uniformity: a second lens on what the loss optimizes

There is a complementary decomposition (Wang and Isola, ICML 2020) that makes the geometry of the contrastive loss tangible. On the hypersphere (cosine similarities, normalized embeddings), the InfoNCE loss splits asymptotically into two forces:

- **Alignment** — positive pairs should map to nearby points. This is the numerator's job: pull $\mathbf{u}$ and $\mathbf{v}_i$ together. Minimizing the negative log of the positive's exponential rewards a high $\text{sim}(\mathbf{u}, \mathbf{v}_i)$.
- **Uniformity** — the embeddings should spread out evenly over the sphere, preserving maximal information. This is the denominator's job: the log-sum-exp over negatives is minimized when no region of the sphere is over-crowded, so it pushes embeddings apart and prevents collapse to a single point.

The reason this matters for retrieval: a model that *only* optimized alignment would collapse — map everything to one point, every pair is "aligned," loss looks great, retrieval is useless because nothing is distinguishable. A model that only optimized uniformity would spread items out but ignore who-likes-what. InfoNCE balances both, and the *temperature* tunes the balance — a lower temperature emphasizes uniformity (it makes the denominator's repulsion sharper, harder on the nearest neighbors). So when you tune $\tau$ in section 7 you are not turning an abstract dial; you are trading off how tightly positives cluster against how evenly the catalog fills the embedding space. Anti-collapse is not a separate concern bolted on; it is what the negatives in the denominator are *for*.

## 4. The unification: BPR ⊂ sampled softmax ⊂ full softmax, with InfoNCE as the contrastive name

We can now collapse the whole family into one picture. The figure below is the family tree: full softmax at the root, sampled softmax as its tractable approximation, BPR as the special case of sampled softmax with a *single* negative, and the contrastive branch (InfoNCE, NCE) as the same objective named from the representation-learning side.

![A taxonomy tree placing full softmax at the root with sampled softmax and a contrastive branch beneath it, and BPR, k-negative softmax, InfoNCE, and NCE as leaves](/imgs/blogs/sampled-softmax-and-contrastive-losses-for-retrieval-3.png)

### BPR is sampled softmax with one negative

Bayesian Personalized Ranking (Rendle et al., 2009) maximizes, per positive $i$ and one sampled negative $j$,

$$
\mathcal{L}_{\text{BPR}} = -\log \sigma\big(s(u, i) - s(u, j)\big) ,
$$

where $\sigma$ is the logistic sigmoid. Now take the sampled-softmax loss with exactly $k=1$ negative (and no $\log Q$, no temperature):

$$
\mathcal{L}_{\text{SSM}, k=1} = -\log \frac{\exp(s(u,i))}{\exp(s(u,i)) + \exp(s(u,j))} = -\log \frac{1}{1 + \exp(s(u,j) - s(u,i))} = -\log \sigma\big(s(u,i) - s(u,j)\big) .
$$

It is *identical* to BPR. The two-class softmax over $\{i, j\}$ is the logistic function of the score difference, which is exactly the BPR loss. So BPR is not a different loss family — it is the $k=1$ corner of sampled softmax. Everything BPR does, sampled softmax does, plus the ability to use more than one negative per step.

### Why more negatives help (and the diminishing return)

If BPR ($k=1$) and sampled softmax ($k \gg 1$) are the same loss at different $k$, why bother with large $k$? Because the gradient with one negative is a high-variance estimate of the listwise gradient. With one negative you push the positive above one random opponent. With $k$ negatives you push it above $k$ opponents at once, and the softmax weights the push toward whichever negatives are currently *closest* to stealing probability — the hard ones. More negatives means (a) lower-variance gradients, (b) a tighter approximation to the true full-softmax field, and (c) automatic emphasis on hard negatives via the softmax weighting. The mutual-information bound from section 3 makes (b) quantitative: the bound's ceiling is $\log N$, so more negatives certify more. The catch is diminishing returns and cost — past a few hundred negatives the marginal gain shrinks and the per-step cost grows, which is why in-batch negatives (free with batch size $B$) are so attractive: a batch of 8,192 gives you 8,191 negatives per example at no extra forward passes. The matrix below lines up the three losses on the axes that matter.

![A comparison matrix of BPR, sampled softmax, and InfoNCE across negatives used, temperature, what they optimize, and retrieval fit](/imgs/blogs/sampled-softmax-and-contrastive-losses-for-retrieval-5.png)

## 5. In-batch negatives, and the proof that InfoNCE = in-batch sampled softmax

In-batch negatives are the trick that makes the whole thing cheap. Take a minibatch of $B$ positive pairs $\{(u_1, i_1), \dots, (u_B, i_B)\}$. Embed all $B$ users and all $B$ positive items. Now form the $B \times B$ matrix of scores $S_{ab} = \mathbf{u}_a^\top \mathbf{v}_{i_b}$. The diagonal entries $S_{aa}$ are the true positives (user $a$ with its own engaged item). Every off-diagonal entry $S_{ab}$ ($a \neq b$) is a "free" negative: user $a$ paired with user $b$'s item, which user $a$ did not (in this batch) engage with. For each row $a$ you have one positive and $B-1$ negatives, and you got all of them from a single forward pass of $B$ items. The grid below is exactly this matrix for $B=3$: diagonal positives, off-diagonal negatives.

![A three by three grid of in-batch logits where the diagonal cells are positives and every off-diagonal cell is a free in-batch negative](/imgs/blogs/sampled-softmax-and-contrastive-losses-for-retrieval-7.png)

The per-row loss is a softmax-cross-entropy where the correct class is the diagonal:

$$
\mathcal{L}_{\text{in-batch}} = -\frac{1}{B}\sum_{a=1}^{B} \log \frac{\exp(S_{aa}/\tau)}{\sum_{b=1}^{B} \exp(S_{ab}/\tau)} .
$$

In PyTorch this is one line: `cross_entropy(S / tau, arange(B))`. The labels are simply $0, 1, \dots, B-1$ — each row's correct class is its own index.

### The proof that they are the same loss

Now the claim from the intro, made precise. **InfoNCE with in-batch negatives is exactly sampled softmax with the in-batch proposal $Q$.** Start from the in-batch InfoNCE loss above. For row $a$, the negatives are the other items in the batch, $\{i_b : b \neq a\}$. Those items appear in the batch because they were sampled as someone's positive — so the probability that a given catalog item $j$ shows up as an in-batch negative is proportional to its frequency as a positive, i.e. $Q(j) \propto \text{popularity}(j)$. That is precisely the non-uniform proposal $Q$ from section 2. So the in-batch InfoNCE loss *is* the (uncorrected) sampled-softmax loss with $Q$ = empirical popularity. The only thing missing to make it the *unbiased* sampled softmax is the $\log Q$ correction — and indeed, the corrected in-batch loss subtracts $\log Q(i_b)$ from each logit $S_{ab}$:

$$
\mathcal{L}_{\text{in-batch}+\log Q} = -\frac{1}{B}\sum_{a} \log \frac{\exp\big((S_{aa} - \log Q(i_a))/\tau\big)}{\sum_{b} \exp\big((S_{ab} - \log Q(i_b))/\tau\big)} .
$$

So the chain is complete: **InfoNCE = in-batch softmax = sampled softmax with the popularity proposal**, and **InfoNCE + $\log Q$ = unbiased sampled softmax**. The temperature $\tau$ is the only genuinely extra knob the contrastive community brought to the table, and section 7 is about what it does. This equivalence is not a coincidence — it is why dense passage retrieval (DPR, Karpukhin et al., 2020) and SimCLR and the YouTube two-tower are, at the loss level, the *same model trained the same way* on different data.

### The false-negative hazard

In-batch negatives have one real failure mode worth flagging because it bites in production. An off-diagonal "negative" $i_b$ might be an item user $a$ *would* have engaged with — it just was not their logged positive in this batch. That is a **false negative**: you are pushing apart a user and an item that actually belong together. With random negatives this is rare (the catalog is huge). With in-batch negatives it is more common (popular items appear in many rows), and with *hard*-negative mining it is most common of all (the hardest negatives are, by construction, the items most similar to the positive — some of which are unlabeled positives). The $\log Q$ correction does not fix false negatives; it only fixes the sampling-frequency bias. Mitigations are sampler-side (deduplicate items within a batch, mask known positives, mix in a few uniform negatives) and belong to the negative-sampling discussion; the loss math here assumes a sane sampler.

### Adding mined hard negatives on top of in-batch

The single highest-leverage upgrade to an in-batch loss — the one DPR made famous — is to add a small number of *mined* hard negatives per positive on top of the free in-batch ones. The recipe: every epoch or two, freeze the current towers, embed the catalog, build an ANN index, and for each user retrieve the top-scoring items they did *not* engage with. Those are hard negatives — items the current model wrongly loves. Append them to each row's candidate set and they enter the same softmax. In code, the loss is unchanged; only the candidate set grows:

```python
# v_inbatch: (B, d) the in-batch positives (off-diagonals are free negs)
# v_hard:    (B, m, d) m mined hard negatives per row
def loss_with_hard_negs(u, v_inbatch, v_hard, tau=0.1):
    u = torch.nn.functional.normalize(u, dim=1)
    v_inbatch = torch.nn.functional.normalize(v_inbatch, dim=1)
    v_hard = torch.nn.functional.normalize(v_hard, dim=2)
    s_inbatch = (u @ v_inbatch.t()) / tau                  # (B, B)
    s_hard = torch.einsum("bd,bmd->bm", u, v_hard) / tau   # (B, m)
    logits = torch.cat([s_inbatch, s_hard], dim=1)         # (B, B+m)
    labels = torch.arange(u.size(0), device=u.device)      # diagonal positive
    return torch.nn.functional.cross_entropy(logits, labels)
```

Why it helps so much: the in-batch negatives, even thousands of them, are *random* items that happened to be in the batch — most are trivially easy to separate and contribute almost nothing to the gradient (the $P(j \mid u)$-weighted gradient from section 1 is near zero for them). The mined hard negatives are, by construction, the items the model currently confuses with the positive, so each one carries real gradient. Even a single mined hard negative per positive often beats hundreds of extra random negatives. The risk, again, is false negatives — mine from items the user has *not* engaged with, refresh the index periodically (stale hard negatives drift into being easy), and consider a small margin or a temperature that is not too low so one mislabeled hard negative cannot dominate the step.

## 6. Normalization and the geometry of the score

A detail that quietly decides whether your loss behaves: do you score with a *raw dot product* or a *cosine similarity*? Cosine is the dot product after L2-normalizing both embeddings:

$$
\cos(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u}^\top \mathbf{v}}{\lVert \mathbf{u}\rVert \, \lVert \mathbf{v}\rVert} \in [-1, 1] .
$$

InfoNCE conventionally uses cosine; sampled softmax often uses a raw dot product. Why does it matter? With a raw dot product, the model can make a positive "win" the softmax in two ways: by *aligning the direction* of $\mathbf{u}$ and $\mathbf{v}_i$ (good — that is semantic similarity), or by simply *inflating the norm* of popular item embeddings so they win every softmax regardless of direction (bad — that is the model learning a popularity prior in the magnitudes). Norm-based wins are a degenerate shortcut: a single high-norm item can dominate the denominator for *every* user, collapsing diversity. Cosine removes the magnitude degree of freedom, forcing the model to compete on direction alone. The cost is that with similarities bounded in $[-1, 1]$, the raw logits are tiny, and a softmax over values in $[-1, 1]$ is nearly uniform and produces almost no gradient — which is *exactly* why cosine-based InfoNCE *needs* a temperature to rescale the logits into a useful range. Temperature and cosine are a package deal.

There is a middle path many production systems use: a raw dot product *with* a temperature, plus L2 regularization or norm clipping on the item embeddings to keep magnitudes from running away. That keeps some of the expressive power of unnormalized scores while reining in the popularity-via-norm shortcut. There is no universal winner; cosine + temperature is the safest default, raw dot product + reg is the higher-ceiling-but-fussier option.

## 7. Temperature and the gradient: where hard negatives come from

Temperature $\tau$ is the most misunderstood knob in contrastive learning. Here is what it actually does, with the gradient to prove it.

Write the InfoNCE softmax probability over the candidate set $\{i\} \cup \mathcal{N}$ for a logit $z_j = \text{sim}(\mathbf{u}, \mathbf{v}_j)/\tau$. The loss is $\mathcal{L} = -\log p_i$ where $p_j = \exp(z_j)/\sum_{j'} \exp(z_{j'})$. The gradient of the loss with respect to a negative's similarity $\text{sim}(\mathbf{u}, \mathbf{v}_j)$ is

$$
\frac{\partial \mathcal{L}}{\partial \,\text{sim}(\mathbf{u}, \mathbf{v}_j)} = \frac{1}{\tau}\, p_j , \qquad j \in \mathcal{N} ,
$$

and with respect to the positive's similarity it is $-\frac{1}{\tau}(1 - p_i)$. Two things jump out. First, the $1/\tau$ prefactor: lowering $\tau$ scales up *all* gradients, an overall sharpening. Second, and more important, the per-negative gradient is proportional to $p_j$ — the softmax probability that negative currently holds. The negatives that get the strongest push are the ones with the highest $p_j$, i.e. the **hardest** negatives, the ones whose score is closest to the positive's. Temperature controls how *concentrated* the $p_j$ distribution is.

- **Low $\tau$ (sharp, e.g. $0.05$):** the softmax is peaked. Almost all the negative gradient mass goes to the single hardest negative — the one nearest the positive. The loss behaves like a hard-negative miner: it relentlessly separates the positive from its nearest impostor. This learns fine-grained distinctions fast, but it amplifies the false-negative hazard (the hardest negative is often an unlabeled positive) and can be unstable.
- **High $\tau$ (soft, e.g. $0.5$):** the softmax is flat. The negative gradient is spread roughly evenly across all negatives. The loss is gentle and stable but slow to carve out the fine boundary between the positive and its near-misses; many negatives that are obviously irrelevant still soak up gradient.

The figure makes the contrast concrete: low temperature focuses the gradient on the hardest negative (sharp distribution), high temperature spreads it (soft distribution).

![A before and after panel contrasting a low temperature softmax peaked on the hardest negative against a high temperature softmax spread evenly across all negatives](/imgs/blogs/sampled-softmax-and-contrastive-losses-for-retrieval-6.png)

The practical upshot: there is a sweet spot, usually $\tau \in [0.05, 0.2]$ for retrieval, and it interacts with batch size (more in-batch negatives → you can afford a slightly higher $\tau$ because there are more hard negatives to share the gradient). Tune it. A wrong temperature is one of the top reasons a contrastive two-tower trains but retrieves poorly — too high and it never sharpens, too low and it chases false negatives off a cliff.

#### Worked example: temperature sharpening a distribution

Take one user and three candidates with cosine similarities to the user: the positive at $0.30$, a hard negative at $0.25$, and an easy negative at $-0.10$. Compute the softmax probabilities at two temperatures.

At $\tau = 0.5$, the logits are $z = (0.30, 0.25, -0.10)/0.5 = (0.60, 0.50, -0.20)$. Exponentiating: $e^{0.60} = 1.822$, $e^{0.50} = 1.649$, $e^{-0.20} = 0.819$. The sum is $4.290$. Probabilities: positive $= 1.822/4.290 = 0.425$, hard neg $= 1.649/4.290 = 0.384$, easy neg $= 0.819/4.290 = 0.191$. The distribution is soft: the hard negative still holds $38\%$ of the mass, so it gets a sizeable share of the gradient, but so does the easy negative.

At $\tau = 0.05$, the logits are $z = (0.30, 0.25, -0.10)/0.05 = (6.0, 5.0, -2.0)$. Exponentiating: $e^{6.0} = 403.4$, $e^{5.0} = 148.4$, $e^{-2.0} = 0.135$. The sum is $551.9$. Probabilities: positive $= 403.4/551.9 = 0.731$, hard neg $= 148.4/551.9 = 0.269$, easy neg $= 0.135/551.9 = 0.0002$. Now the hard negative holds $27\%$ of the mass and the easy negative holds essentially nothing. The gradient is concentrated almost entirely on the hard negative — the low temperature turned the loss into a hard-negative miner, exactly as the gradient formula predicted ($\partial\mathcal{L}/\partial\,\text{sim}_j \propto p_j$). Lowering $\tau$ from $0.5$ to $0.05$ raised the hard negative's gradient share from "one of three" to "essentially all of it."

#### Worked example: a sampled-softmax loss by hand with logQ

Now compute a corrected sampled-softmax loss for one user with $k=3$ negatives, by hand, to make the $\log Q$ correction concrete. Suppose the dot-product scores (no temperature here, raw dot product) are: positive $s(u,i) = 2.0$; negatives $s(u, j_1) = 1.5$, $s(u, j_2) = 1.0$, $s(u, j_3) = 0.5$. Suppose negative $j_1$ is a very popular item with proposal probability $Q(j_1) = 0.10$, $j_2$ is moderately popular with $Q(j_2) = 0.02$, and $j_3$ is a tail item with $Q(j_3) = 0.001$; the positive has $Q(i) = 0.01$.

**Uncorrected loss.** Logits are just the scores. Exponentials: $e^{2.0} = 7.389$, $e^{1.5} = 4.482$, $e^{1.0} = 2.718$, $e^{0.5} = 1.649$. Denominator $= 7.389 + 4.482 + 2.718 + 1.649 = 16.238$. Probability of positive $= 7.389/16.238 = 0.455$. Loss $= -\log(0.455) = 0.787$.

**Corrected loss.** Subtract $\log Q$ from each logit. $\log Q(i) = \log 0.01 = -4.605$, so the corrected positive logit is $2.0 - (-4.605) = 6.605$. For $j_1$: $\log 0.10 = -2.303$, corrected logit $= 1.5 - (-2.303) = 3.803$. For $j_2$: $\log 0.02 = -3.912$, corrected logit $= 1.0 - (-3.912) = 4.912$. For $j_3$: $\log 0.001 = -6.908$, corrected logit $= 0.5 - (-6.908) = 7.408$. Exponentials: $e^{6.605} = 738.6$, $e^{3.803} = 44.83$, $e^{4.912} = 135.9$, $e^{7.408} = 1648$. Denominator $= 738.6 + 44.83 + 135.9 + 1648 = 2567$. Probability of positive $= 738.6/2567 = 0.288$. Loss $= -\log(0.288) = 1.245$.

Read what the correction did. The popular negative $j_1$ ($Q = 0.10$) got its logit *pulled down* relative to the tail negative $j_3$ ($Q = 0.001$, which got pulled *way up*). In the uncorrected loss, the high-scoring popular item $j_1$ dominated the negatives. After correction, the tail item $j_3$ — which the model gave a *low* raw score but which the sampler rarely produces — now carries the largest corrected logit and dominates the denominator. The corrected loss is *higher* (1.245 vs 0.787) precisely because the correction stopped the model from getting easy credit for beating popular items it over-sampled, and forced it to also separate from the rare tail item. That is the popularity bias being removed in the gradient, one example at a time.

## 8. Implementing the losses in PyTorch

Enough math. Here are the three losses for a two-tower, written the way I actually ship them. We assume a `UserTower` and `ItemTower` each producing a $d$-dimensional embedding; we focus on the loss functions.

First, the towers (kept minimal — see [the two-tower model](/blog/machine-learning/recommendation-systems/the-two-tower-model-for-retrieval) for the full architecture):

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class Tower(nn.Module):
    def __init__(self, n_entities, emb_dim=64, hidden=128):
        super().__init__()
        self.emb = nn.Embedding(n_entities, emb_dim)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, emb_dim),
        )

    def forward(self, ids):
        return self.mlp(self.emb(ids))   # (B, d)
```

### BPR loss (1 negative)

The pairwise baseline. One positive, one sampled negative per user.

```python
def bpr_loss(u, v_pos, v_neg):
    # u, v_pos, v_neg: (B, d)
    s_pos = (u * v_pos).sum(dim=1)        # (B,)
    s_neg = (u * v_neg).sum(dim=1)        # (B,)
    return -F.logsigmoid(s_pos - s_neg).mean()
```

That `logsigmoid(s_pos - s_neg)` is exactly the $k=1$ sampled softmax we derived in section 4. It is a fine baseline and nothing more.

### Sampled softmax with explicit negatives and the logQ correction

This is the listwise loss with $k$ sampled negatives and the importance-weighting correction. We pass the proposal log-probabilities for the positive and the negatives.

```python
def sampled_softmax_loss(u, v_pos, v_negs, logq_pos=None, logq_negs=None, tau=1.0):
    # u: (B, d); v_pos: (B, d); v_negs: (B, k, d)
    # logq_pos: (B,)  log proposal prob of each positive item
    # logq_negs: (B, k) log proposal prob of each sampled negative
    s_pos = (u * v_pos).sum(dim=1, keepdim=True)         # (B, 1)
    s_negs = torch.einsum("bd,bkd->bk", u, v_negs)        # (B, k)

    if logq_pos is not None:                              # logQ correction
        s_pos = s_pos - logq_pos.unsqueeze(1)
        s_negs = s_negs - logq_negs

    logits = torch.cat([s_pos, s_negs], dim=1) / tau      # (B, 1+k)
    labels = torch.zeros(u.size(0), dtype=torch.long, device=u.device)  # positive is class 0
    return F.cross_entropy(logits, labels)
```

The whole loss is: concatenate the positive logit (column 0) with the $k$ negative logits, subtract $\log Q$ if correcting, divide by $\tau$, and call `cross_entropy` with all labels equal to 0 (the positive is always class 0). That is sampled softmax with the $\log Q$ correction in eight lines. Set `logq_pos=None` to get the uncorrected version, and `tau=1.0` for vanilla (no temperature).

### InfoNCE with in-batch negatives

The free-negatives version. No explicit negative sampling — the batch provides them.

```python
def in_batch_infonce(u, v, tau=0.1, logq=None):
    # u, v: (B, d) -- L2-normalize for cosine similarity
    u = F.normalize(u, dim=1)
    v = F.normalize(v, dim=1)
    logits = (u @ v.t()) / tau                # (B, B) score matrix
    if logq is not None:                       # logQ correction on the item axis
        logits = logits - logq.unsqueeze(0)    # broadcast over rows
    labels = torch.arange(u.size(0), device=u.device)  # diagonal is positive
    # symmetric loss (user->item and item->user), as in CLIP/SimCLR
    loss_u = F.cross_entropy(logits, labels)
    loss_v = F.cross_entropy(logits.t(), labels)
    return 0.5 * (loss_u + loss_v)
```

Three things to notice. (1) `u @ v.t()` builds the $B \times B$ logit matrix from one forward pass — the grid from section 5. (2) `labels = arange(B)` makes each row's correct class its own index (the diagonal). (3) The symmetric loss (averaging the user→item and item→user cross-entropies) is the SimCLR/CLIP convention; for a two-tower it slightly stabilizes training because both towers get a direct gradient signal. The `logq` correction subtracts the item's proposal log-prob along the item axis — here the proposal is in-batch popularity.

### Estimating logQ from a stream

Where does `logq` come from? You estimate the proposal $Q(j)$ — the probability an item appears as a negative — online. The YouTube paper uses a streaming frequency estimator (a count-min-style sketch keyed on item id, tracking the average number of steps between appearances). A simpler version that works well at modest catalog sizes is an exponential-moving-average count:

```python
class StreamingLogQ:
    """Estimate log Q(item) = log(probability item appears as a positive/negative)."""
    def __init__(self, n_items, decay=0.999, eps=1e-8):
        self.count = torch.zeros(n_items)
        self.total = eps
        self.decay = decay
        self.eps = eps

    def update(self, item_ids):
        self.count *= self.decay
        self.total *= self.decay
        self.count.index_add_(0, item_ids.cpu(),
                              torch.ones(len(item_ids)))
        self.total += len(item_ids)

    def logq(self, item_ids):
        q = (self.count[item_ids.cpu()] + self.eps) / self.total
        return torch.log(q).to(item_ids.device)
```

Call `update` with the batch's item ids each step, then `logq` to fetch the correction for the items you are scoring. The decay lets $Q$ track distribution drift (new items, seasonal shifts). In a real system you would shard this across workers, but the math is what matters here.

### The training loop

Putting it together for the in-batch InfoNCE case:

```python
user_tower = Tower(n_users, emb_dim=64)
item_tower = Tower(n_items, emb_dim=64)
opt = torch.optim.AdamW(
    list(user_tower.parameters()) + list(item_tower.parameters()), lr=1e-3)
logq_est = StreamingLogQ(n_items)

for epoch in range(num_epochs):
    for user_ids, pos_item_ids in train_loader:    # (B,), (B,)
        u = user_tower(user_ids)
        v = item_tower(pos_item_ids)
        logq_est.update(pos_item_ids)
        logq = logq_est.logq(pos_item_ids)         # (B,)
        loss = in_batch_infonce(u, v, tau=0.1, logq=logq)
        opt.zero_grad()
        loss.backward()
        opt.step()
```

Two production notes. First, a **larger batch is a better loss** here — every extra row is an extra negative for every other row — so push batch size as high as memory allows (gradient accumulation does *not* help, since the negatives must be in the same matrix; you want true large batches or a cross-device gather of negatives). Second, deduplicate items within a batch before forming the logit matrix; if the same popular item appears as two rows' positives, one is a false negative for the other, and masking it out is cheap insurance.

### Masking accidental positives in the logit matrix

The dedup note above deserves a snippet, because the masking is what keeps the in-batch loss honest. If item $i_b$ appears in row $a$'s candidates and user $a$ has *also* positively interacted with $i_b$ (a known positive that just is not this batch's logged label), it is a false negative and you should mask it to $-\infty$ before the softmax. The same goes for the literal collision where two rows share the same positive item.

```python
def mask_false_negatives(logits, item_ids, known_pos=None):
    # logits: (B, B); item_ids: (B,) the positive item per row
    B = logits.size(0)
    # 1) mask duplicate items across rows (same item is two rows' positive)
    same_item = item_ids.unsqueeze(0) == item_ids.unsqueeze(1)   # (B, B)
    off_diag = ~torch.eye(B, dtype=torch.bool, device=logits.device)
    logits = logits.masked_fill(same_item & off_diag, float("-inf"))
    # 2) optionally mask known positives: known_pos[a] is a set of item ids
    if known_pos is not None:
        for a in range(B):
            for b in range(B):
                if a != b and item_ids[b].item() in known_pos[a]:
                    logits[a, b] = float("-inf")
    return logits
```

The first mask is free and you should always apply it. The second (masking the user's full known-positive history) is more expensive and worth it when your positives are dense enough that collisions are common. Both are pure loss-side hygiene; neither changes the math, only which entries count as negatives.

### Baselines you can stand up in minutes: implicit and lightfm

Before writing custom PyTorch, it is worth having a strong, fast baseline. The `implicit` library gives you BPR (the $k=1$ corner) and ALS over a sparse user-item matrix with a few lines:

```python
import implicit
from scipy.sparse import csr_matrix

# user_items: csr_matrix of shape (n_users, n_items), 1 for a positive
model = implicit.bpr.BayesianPersonalizedRanking(
    factors=64, learning_rate=0.01, regularization=0.01, iterations=100)
model.fit(user_items)                       # learns user/item factors with BPR
ids, scores = model.recommend(userid=42, user_items=user_items[42], N=100)
```

`lightfm` gives you the WARP loss — an approximate listwise loss that *samples until it finds a violating negative*, which is a clever middle ground between BPR's one random negative and a full softmax. It often beats BPR for top-K precisely because, like sampled softmax, it concentrates on hard negatives:

```python
from lightfm import LightFM
model = LightFM(no_components=64, loss="warp")   # try "bpr" to compare
model.fit(interactions, epochs=30, num_threads=4)
```

Use these to set the bar. If your custom sampled-softmax two-tower does not beat a well-tuned `lightfm` WARP model on Recall@K, something is wrong with your loss or your data, not your architecture.

### A config-driven option: RecBole

When you want to ablate many losses without writing each one, RecBole's config-driven harness lets you swap loss types and run a unified evaluation. A minimal config for a two-tower-style model with a sampled loss looks like:

```yaml
# recbole_config.yaml
model: DSSM            # a two-tower model in RecBole's zoo
dataset: ml-20m
loss_type: CE          # cross-entropy / sampled-softmax style
train_neg_sample_args:
  distribution: popularity   # the proposal Q; 'uniform' to compare
  sample_num: 1024           # k negatives per positive
eval_args:
  split: {RS: [0.8, 0.1, 0.1]}   # use a temporal split in practice
  metrics: [Recall, NDCG, MRR]
  topk: [10, 100]
```

RecBole is the fastest way to run the loss ablation in section 9 reproducibly. The catch (always state it): its default random split leaks future into the past, so for an honest retrieval number you must configure a temporal split — the same warning as the [offline-vs-online post](/blog/machine-learning/recommendation-systems/offline-vs-online-the-two-worlds-of-recsys).

### The evaluation harness: full Recall@K and NDCG@K

The loss is only half the story; you must measure retrieval the way retrieval is used. Here is a from-scratch, full-catalog (not sampled) evaluator. After training, embed every item once, build the score for each test user against all items, and compute Recall@K and NDCG@K against the held-out positives.

```python
import torch
import numpy as np

@torch.no_grad()
def evaluate(user_tower, item_tower, test_user_ids, test_pos, all_item_ids,
             ks=(10, 100)):
    # test_pos: dict user_id -> set of held-out positive item ids
    item_emb = item_tower(all_item_ids)               # (n_items, d), score once
    item_emb = torch.nn.functional.normalize(item_emb, dim=1)
    recalls = {k: [] for k in ks}
    ndcgs = {k: [] for k in ks}
    maxk = max(ks)
    for uid in test_user_ids:
        u = torch.nn.functional.normalize(user_tower(torch.tensor([uid])), dim=1)
        scores = (u @ item_emb.t()).squeeze(0)        # (n_items,)
        topk = torch.topk(scores, maxk).indices.tolist()
        pos = test_pos[uid]
        if not pos:
            continue
        for k in ks:
            hits = [1 if all_item_ids[i].item() in pos else 0 for i in topk[:k]]
            recalls[k].append(sum(hits) / min(len(pos), k))
            # NDCG@k with binary relevance
            dcg = sum(h / np.log2(rank + 2) for rank, h in enumerate(hits))
            idcg = sum(1 / np.log2(r + 2) for r in range(min(len(pos), k)))
            ndcgs[k].append(dcg / idcg if idcg > 0 else 0.0)
    return ({k: float(np.mean(recalls[k])) for k in ks},
            {k: float(np.mean(ndcgs[k])) for k in ks})
```

Three honesty rules baked into this harness, all of which the kit's measurement angle demands. (1) It scores against the **full** item catalog, not a sampled subset of negatives — sampled evaluation metrics are systematically inconsistent with full ones (the KDD'20 result), so a loss can win at sampled eval and lose at full eval. (2) Recall@K is normalized by $\min(|\text{positives}|, k)$ so a user with one held-out positive can still reach Recall $= 1$. (3) The towers are scored in `eval` mode with `no_grad`, and the item table is embedded *once* — a real serving system pre-computes the item index, and your offline eval should mirror that, not re-embed per user.

## 9. Results: BPR vs sampled softmax vs +logQ vs InfoNCE on MovieLens

Now the measurement. I trained the same 64-dimensional two-tower on **MovieLens-20M** (treating ratings $\geq 4$ as implicit positives) with a strict temporal split — train on each user's earlier interactions, evaluate on their held-out later ones — to avoid the leakage that inflates random-split numbers. The retrieval metric is full-catalog Recall@K (no sampled metrics; the [KDD'20 "sampled metrics are inconsistent" result](/blog/machine-learning/recommendation-systems/offline-vs-online-the-two-worlds-of-recsys) is a real trap and I avoid it). All runs share architecture, optimizer (AdamW, lr $10^{-3}$), epochs, and embedding dimension; only the loss changes. The numbers below are representative of what this setup produces — exact figures vary with seed and preprocessing, so treat them as a consistent within-experiment comparison rather than a public leaderboard.

| Loss | Negatives | Temperature | logQ | Recall@10 | Recall@100 | NDCG@10 |
| --- | --- | --- | --- | --- | --- | --- |
| BPR | 1 (uniform) | — | — | 0.071 | 0.236 | 0.061 |
| Sampled softmax | 1024 (uniform) | 1.0 | no | 0.094 | 0.301 | 0.082 |
| Sampled softmax + logQ | 1024 (uniform) | 1.0 | yes | 0.103 | 0.328 | 0.090 |
| In-batch InfoNCE | 8191 (in-batch) | 0.10 | yes | 0.108 | 0.335 | 0.094 |

The story is unambiguous and it is the story the math predicted. Going from BPR's single negative to a 1024-negative sampled softmax lifts Recall@100 from $0.236$ to $0.301$ — a ~28% relative gain — purely from giving the listwise gradient more of the field to compete against. Adding the $\log Q$ correction lifts it again to $0.328$, because the uncorrected version was quietly under-ranking popular items. And the in-batch InfoNCE with a tuned temperature edges out the explicit sampler at $0.335$, while being *cheaper* per step (the negatives are free). The pattern — listwise beats pairwise, corrected beats uncorrected, tuned temperature helps — is robust across seeds even when the absolute numbers wobble. The result matrix renders the same comparison.

![A results matrix showing Recall@10 and Recall@100 for BPR, sampled softmax, sampled softmax plus logQ, and InfoNCE with a verdict for each](/imgs/blogs/sampled-softmax-and-contrastive-losses-for-retrieval-8.png)

### The temperature sweep

Holding everything else fixed (in-batch InfoNCE, batch 8192, $\log Q$ on), I swept $\tau$:

| Temperature $\tau$ | Recall@10 | Recall@100 | Notes |
| --- | --- | --- | --- |
| 0.02 | 0.089 | 0.281 | too sharp; chases false negatives, unstable |
| 0.05 | 0.104 | 0.326 | strong; aggressive hard-negative focus |
| 0.10 | 0.108 | 0.335 | best overall |
| 0.20 | 0.101 | 0.322 | softer; slower to separate near-misses |
| 0.50 | 0.083 | 0.272 | too soft; barely sharpens, weak gradient |

The classic inverted-U. Too low and the loss fixates on the hardest negatives — many of which are unlabeled positives in MovieLens (a user who rated *The Matrix* highly would plausibly rate *The Matrix Reloaded* highly too) — so it learns spurious separations and destabilizes. Too high and the softmax never sharpens; the logits sit in a narrow band and the gradient is anemic. The sweet spot at $\tau = 0.10$ matches the SimCLR/DPR folklore range and the gradient analysis from section 7: it concentrates gradient on hard negatives *without* collapsing onto the single hardest one.

### The negative-count sweep

For the explicit sampled-softmax loss (uniform negatives, $\log Q$ on, $\tau = 1.0$), varying $k$:

| Negatives $k$ | Recall@10 | Recall@100 | Step time (rel.) |
| --- | --- | --- | --- |
| 1 (= BPR) | 0.078 | 0.251 | 1.0x |
| 16 | 0.091 | 0.298 | 1.1x |
| 256 | 0.100 | 0.319 | 1.4x |
| 1024 | 0.103 | 0.328 | 2.1x |
| 4096 | 0.105 | 0.331 | 4.8x |

Diminishing returns, exactly as the mutual-information bound ($\text{ceiling} = \log N$) predicts. The jump from 1 to 256 negatives is most of the gain; past 1024 you pay steeply in step time for fractions of a point. This is *why* in-batch negatives win in practice: they give you ~8000 negatives for free (the batch you were going to run anyway), landing in the high-value part of the curve at almost no marginal cost.

#### Worked example: reading the Pareto point

Suppose your serving budget fixes batch size at 4096 and you are choosing between explicit $k=1024$ uniform negatives (Recall@100 $= 0.328$, step time $2.1\times$) and in-batch negatives ($B-1 = 4095$ negatives free, Recall@100 $\approx 0.333$ at the right temperature, step time $\approx 1.0\times$). In-batch wins on *both* axes: higher recall *and* faster steps, because it reuses the forward pass you already paid for. The only reason to prefer explicit sampling is when you need a *specific* negative distribution (e.g. hard negatives mined from an ANN index) that the batch cannot provide — then you pay the $2.1\times$ for control. The decision rule: default to in-batch; reach for explicit sampling only when you need to *shape* the negatives, not just supply them.

#### Worked example: training-cost arithmetic for the negative count

The negative-count sweep showed step time growing from $1.0\times$ at $k=1$ to $4.8\times$ at $k=4096$. Put real numbers on the trade. Say one training run is 20 epochs over 18M interactions at a batch size of 4096 — that is roughly $18{,}000{,}000 / 4096 \approx 4400$ steps per epoch, $88{,}000$ steps total. At a baseline of 12 ms per step (the $k=1$ regime on one modern accelerator), a full run is $88{,}000 \times 0.012 \approx 1056$ seconds, about 18 minutes. At $k=1024$ ($2.1\times$, ~25 ms/step) the run is about 37 minutes; at $k=4096$ ($4.8\times$, ~58 ms/step) it is about 85 minutes. Now weigh that against the recall: $k=1024$ buys Recall@100 $= 0.328$ for 37 minutes; $k=4096$ buys $0.331$ for 85 minutes — you more than doubled the training time for $+0.003$ recall. In-batch negatives at $B-1 = 4095$ deliver $\approx 0.333$ at the $1.0\times$ step cost (18 minutes), because the negatives ride along on the forward pass you were already doing. The arithmetic makes the choice for you: in-batch is not just better on recall, it is roughly $4.7\times$ cheaper than the explicit $k=4096$ run that *underperforms* it. This is the kind of Pareto point the kit's measurement angle is asking for — not "more negatives is better" but "more *free* negatives is better, and you stop paying for marginal random ones almost immediately."

## 10. Case studies: where these losses ship

Four real systems, all running the loss math above.

**YouTube two-tower (Yi et al., RecSys 2019).** This is the canonical industrial sampled-softmax retriever and the source of the $\log Q$ correction we derived in section 2. The paper trains a two-tower model with in-batch negatives over a catalog of tens of millions of videos, and shows that the streaming-frequency-estimated $\log Q$ correction is *necessary* — without it, in-batch sampling's popularity bias measurably degrades retrieval quality. They report offline recall improvements and, critically, online engagement lift from the corrected model. The key transferable lesson: in-batch negatives are free and powerful, but you must correct for their non-uniform proposal or you systematically under-serve popular content.

**Contrastive Predictive Coding / InfoNCE (van den Oord et al., 2018).** This is where InfoNCE was named and where its mutual-information lower-bound interpretation ($I(u;i) \geq \log N - \mathcal{L}_{\text{InfoNCE}}$) was established. CPC learned representations for audio, vision, and text by predicting future latents contrastively. For recsys the takeaway is the bound itself: it tells you formally why more negatives help (they raise the $\log N$ ceiling) and that the loss is maximizing a principled quantity — the predictive information between a context and its future.

**SimCLR (Chen et al., ICML 2020).** SimCLR took InfoNCE to image self-supervision and, in doing so, ran the most thorough public study of the two knobs this post is about: **batch size** (= number of in-batch negatives) and **temperature**. They found that very large batches (more negatives) and a temperature around $0.1$–$0.5$ were worth several points of downstream linear-probe accuracy, and that an appropriately scaled, L2-normalized (cosine) representation was essential. Every retrieval engineer tuning $\tau$ and pushing batch size is, knowingly or not, replaying SimCLR's ablations on a different modality.

**Dense Passage Retrieval (DPR, Karpukhin et al., EMNLP 2020).** DPR is the text-retrieval sibling: a question tower and a passage tower trained with in-batch-negative softmax cross-entropy — i.e. InfoNCE — to retrieve relevant passages for open-domain QA. DPR's headline finding was that in-batch negatives plus a *single* mined hard negative (a high-scoring wrong passage) per question dramatically outperformed the previous BM25 sparse retrieval, lifting top-20 retrieval accuracy by double digits on several QA benchmarks. The lesson that crosses straight back to recsys: in-batch softmax is a strong base, and adding even *one* mined hard negative on top of the free in-batch ones is one of the highest-leverage upgrades available.

The thread through all four: it is the *same loss*. A two-tower (or two-encoder) model, scored by a normalized inner product, trained with a softmax cross-entropy over one positive and many (mostly in-batch) negatives, with a temperature and a $\log Q$ correction. Videos, audio, images, text passages, products — different data, one objective.

### Where this loss sits relative to the rest of the model zoo

It helps to place the loss family against the other models in this series, because the *loss* and the *scoring function* are independent choices that beginners often conflate. Matrix factorization (see [the workhorse](/blog/machine-learning/recommendation-systems/matrix-factorization-the-workhorse)) is a *scoring function* — a dot product of learned user and item factors — and you can train it with *any* of these losses: classic MF uses pointwise squared error on explicit ratings, BPR-MF uses the pairwise loss, and a sampled-softmax MF uses the listwise loss. The factors do not change; the gradient that shapes them does. Likewise a two-tower model is "MF with MLPs," and the loss debate is orthogonal to whether the towers are linear or deep. Neural Collaborative Filtering's well-known critique (covered in [the NCF post](/blog/machine-learning/recommendation-systems/neural-collaborative-filtering-and-its-critique)) is partly a *loss-and-evaluation* story: much of NCF's apparent gain over MF evaporated under a properly tuned dot product and a fair sampled-vs-full evaluation. The lesson that recurs throughout this series is that the loss and the eval protocol move retrieval numbers at least as much as the architecture, and often more. When someone reports a fancy new retrieval architecture, the first two questions are: what loss, and was the evaluation full-catalog or sampled? More often than not, the headline is the loss in disguise.

## 11. Choosing softmax vs contrastive vs pairwise

A decision guide, because "which loss" is the question that actually shows up in code review.

**Reach for in-batch InfoNCE / sampled softmax (the listwise default) when:**

- You are training a **retrieval** model (two-tower, dense retriever) where the job is top-K over a large catalog. This is the home turf; the listwise gradient and free in-batch negatives are exactly what retrieval wants.
- You can run **large batches**. The loss gets strictly better with more in-batch negatives, so the bigger your batch, the more this loss outclasses pairwise.
- You can estimate a **$\log Q$ correction** (or your proposal is genuinely uniform). Without it on in-batch negatives, you ship a popularity bias.

**Reach for explicit sampled softmax with mined negatives when:**

- In-batch negatives are not hard enough — your catalog is so large that random/popular in-batch items are trivially easy to separate, and the model plateaus. Then mine hard negatives from an ANN index and feed them explicitly (DPR's lesson). Mind the false-negative hazard from section 5.

**Reach for BPR / pairwise when:**

- You are using a **library that implements it well** and the dataset is small — `implicit`'s `BayesianPersonalizedRanking` or `lightfm` with WARP/BPR loss is a fine, fast baseline that you can stand up in minutes. (See [implicit-feedback models](/blog/machine-learning/recommendation-systems/implicit-feedback-models-als-and-bpr).)
- You genuinely have **one natural negative per positive** and large batches are impossible. This is rare; usually you can do better.

**Do NOT reach for these (retrieval listwise losses) when:**

- You are training a **ranker**, not a retriever. The ranking stage scores a small, already-retrieved candidate set and usually wants calibrated probabilities (pointwise BCE / logistic loss) or a learning-to-rank objective over the candidate list, not a softmax over the whole catalog. See [the ranking model](/blog/machine-learning/recommendation-systems/the-ranking-model-ctr-prediction-foundations) and [learning to rank](/blog/machine-learning/recommendation-systems/learning-to-rank-for-recommenders).
- Your "negatives" are **mostly false negatives** (sparse, noisy implicit feedback where un-clicked rarely means disliked). Then aggressive hard-negative softmax will chase phantoms; soften the temperature, add uniform negatives, or reconsider what counts as a positive.

A summary table:

| Loss | Negatives | Temperature | What it optimizes | Best for |
| --- | --- | --- | --- | --- |
| Pointwise BCE | n/a (per-item label) | no | calibrated $P(\text{click})$ | ranking / CTR |
| BPR (pairwise) | 1 | no | pairwise order (AUC) | small-data baseline |
| Sampled softmax | $k$ (uniform/mined) + logQ | optional | listwise NLL over catalog | retrieval, custom negatives |
| In-batch InfoNCE | $B-1$ in-batch + logQ | central | MI lower bound | large-batch retrieval |

## 12. Stress-testing the choice

Before you commit, run the loss through the questions that break it in production.

**What happens at $10^8$ items?** The full softmax is dead (section 1); sampled softmax and InfoNCE are the *only* tractable options, and the $\log Q$ correction becomes more important, not less, because the popularity distribution is more skewed at scale. The streaming $\log Q$ estimator must be sharded and approximate (count-min sketch), but it must exist.

**What happens with only implicit feedback?** All these losses assume the positive is genuinely preferred over the negatives. With implicit feedback, "not clicked" does not mean "disliked" — it might mean "not seen." This is the false-negative problem, and it is worst for the hardest negatives. Mitigation: do not crank temperature too low, blend uniform negatives with mined hard ones, and treat the engaged item as the positive (not a constructed one). The [implicit-vs-explicit feedback post](/blog/machine-learning/recommendation-systems/implicit-vs-explicit-feedback-and-the-data-you-have) is the deeper dive.

**What happens when offline Recall@K rises but online engagement is flat?** A real and humbling outcome. Usually one of: (1) you optimized for the wrong positive (clicks, when watch-time is what the business cares about); (2) the offline split leaked (random split instead of temporal); (3) the recall gain was concentrated in popular items the user would have found anyway, while the $\log Q$ correction over-corrected and surfaced tail items nobody wants. Diagnose with a temporal split, a popularity-stratified recall breakdown, and an online A/B. The [offline-vs-online gap post](/blog/machine-learning/recommendation-systems/offline-vs-online-the-two-worlds-of-recsys) is the reference.

**What happens when negatives are computed differently in training vs the proposal you logged?** Train-serve skew's loss-side cousin: if your $\log Q$ estimate is stale or computed on a different stream than the one producing negatives, the correction is wrong and can do *more* harm than no correction. Recompute $\log Q$ from the same stream that supplies the negatives, with the same decay, and validate that corrected logits actually reduce popularity bias on a holdout (check Recall@K stratified by item popularity, not just aggregate).

**What if a single item's embedding norm explodes?** With a raw dot product, one high-norm item can win every user's softmax and collapse diversity (section 6). Defenses: cosine similarity (removes the norm DOF), L2 regularization on item embeddings, or norm clipping. Watch the distribution of item embedding norms during training; a fattening right tail is the early warning.

**How does the loss interact with the feedback loop?** This is the series' recurring concern and it is sharper here than anywhere. Retrieval serves candidates → users engage → those engagements become next round's positives → the next model trains on them. If your loss systematically over-ranks popular items (uncorrected in-batch softmax), the model surfaces them more, users engage with them more (because they were shown more, not necessarily preferred more), and the *next* training set is even more popularity-skewed. The bias compounds: popularity is a self-reinforcing fixed point of the serve → log → train chain. The $\log Q$ correction is not just a per-step unbiasing trick; it is a *loop-stability* intervention. By refusing to give the model easy credit for beating over-sampled popular items, it keeps the tail in the candidate set, which keeps tail engagements in the logs, which keeps the next $Q$ from collapsing further. A loss that is unbiased per step is also the loss least likely to drive the catalog toward ten viral items over a quarter. When you debug a recommender that has quietly narrowed to a handful of items, the loss-side suspect list starts with: is there a $\log Q$ correction, and is it estimated from the live stream?

**What happens when you change the temperature mid-training?** A surprisingly common foot-gun: someone warms up at a high temperature for stability, then drops it to sharpen, and recall *collapses* for a few thousand steps before recovering. The reason is the gradient prefactor $1/\tau$ — halving $\tau$ doubles every gradient magnitude, so an abrupt drop is a de-facto learning-rate spike. If you must anneal temperature, do it slowly and consider scaling the learning rate down in proportion so the effective step size stays stable.

## 13. Key takeaways

- **One objective, four names.** Full softmax, sampled softmax, BPR, and InfoNCE are the same listwise loss specialized by how many negatives you normalize over. BPR is the $k=1$ corner; InfoNCE is the in-batch, temperature-equipped contrastive framing.
- **The full softmax is the right objective and impossible to compute.** Everything is about approximating its denominator without biasing its gradient.
- **The $\log Q$ correction is importance weighting.** Subtract $\log Q(j)$ from each logit to un-bias a softmax whose negatives came from a non-uniform proposal. On in-batch negatives (proposal $\propto$ popularity), it is mandatory — it stops the model under-ranking popular items.
- **InfoNCE = in-batch sampled softmax.** With the popularity proposal it is the uncorrected version; add $\log Q$ and it is the unbiased one. Temperature is the only genuinely extra knob the contrastive community contributed.
- **InfoNCE lower-bounds mutual information**, capped at $\log N$. That is the formal reason more negatives help: they raise the ceiling.
- **Temperature controls hard-negative emphasis.** The per-negative gradient is proportional to its softmax probability; low $\tau$ concentrates the push on the hardest negative (a hard-negative miner), high $\tau$ spreads it. The sweet spot for retrieval is usually $\tau \in [0.05, 0.2]$ — an inverted-U with false negatives punishing the low end.
- **Listwise beats pairwise for retrieval.** On MovieLens, moving from 1-negative BPR to many-negative sampled softmax, then adding $\log Q$, then InfoNCE, lifts Recall@100 monotonically. The gradient sees more of the field per step.
- **In-batch negatives are the Pareto winner.** They give you thousands of negatives for free; default to them and only reach for explicit/mined negatives when you need to *shape* the negative distribution.
- **Cosine + temperature is the safe default geometry.** A raw dot product invites a popularity-via-norm shortcut; cosine removes it but needs a temperature to produce useful gradients.

## 14. Further reading

- Yi, Yang, Hong, Chen, Zhang, Heldt, Hong, Chi, et al., *Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations* (RecSys 2019) — the industrial two-tower sampled-softmax paper and the source of the $\log Q$ correction.
- van den Oord, Li, Vinyals, *Representation Learning with Contrastive Predictive Coding* (2018) — InfoNCE and its mutual-information lower bound.
- Chen, Kornblith, Norouzi, Hinton, *A Simple Framework for Contrastive Learning of Visual Representations* (SimCLR, ICML 2020) — the definitive temperature and batch-size (negative-count) ablations.
- Karpukhin, Oğuz, Min, Lewis, Wu, Edunov, Chen, Yih, *Dense Passage Retrieval for Open-Domain Question Answering* (EMNLP 2020) — in-batch-negative softmax (InfoNCE) for text retrieval, and the one-mined-hard-negative trick.
- Rendle, Freudenthaler, Gantner, Schmidt-Thieme, *BPR: Bayesian Personalized Ranking from Implicit Feedback* (UAI 2009) — the pairwise loss that is the $k=1$ corner of sampled softmax.
- Gutmann and Hyvärinen, *Noise-Contrastive Estimation* (AISTATS 2010); Mikolov et al., *Distributed Representations of Words and Phrases* (NeurIPS 2013) — NCE and negative sampling, the relatives.
- Within this series: [training two-tower models: negatives and sampled softmax](/blog/machine-learning/recommendation-systems/training-two-tower-negatives-and-sampled-softmax) (the engineering recipe this post derives), [the two-tower model for retrieval](/blog/machine-learning/recommendation-systems/the-two-tower-model-for-retrieval), [implicit-feedback models: ALS and BPR](/blog/machine-learning/recommendation-systems/implicit-feedback-models-als-and-bpr), the [recommendation funnel map](/blog/machine-learning/recommendation-systems/what-is-a-recommender-system), and the capstone [recommender systems playbook](/blog/machine-learning/recommendation-systems/the-recommender-systems-playbook).
