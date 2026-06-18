---
title: "Matrix Factorization: The Workhorse of Recommenders"
date: "2026-06-19"
publishDate: "2026-06-19"
description: "Derive biased matrix factorization from first principles, train it from scratch in PyTorch and with implicit and surprise, and measure RMSE plus Recall@10 on MovieLens to see exactly where latent factors earn their keep."
tags:
  [
    "recommendation-systems",
    "recsys",
    "matrix-factorization",
    "collaborative-filtering",
    "latent-factors",
    "funk-svd",
    "als",
    "machine-learning",
    "movielens",
    "pytorch",
  ]
category: "machine-learning"
subcategory: "Recommendation Systems"
author: "Hiep Tran"
featured: true
readTime: 52
image: "/imgs/blogs/matrix-factorization-the-workhorse-1.png"
---

A team I worked with launched a movie recommender that won the offline bake-off cleanly. Their item-item collaborative filtering had the lowest RMSE on the holdout set, the demo looked great, and they shipped it. Two weeks later the cold-start complaints rolled in: any user with fewer than a dozen ratings got a near-random list, and roughly a third of the catalog never surfaced because those items had too few co-ratings to ever land in a neighborhood. The fix was not a bigger neighborhood model or a cleverer similarity metric. It was to stop comparing users by the items they happened to share and start describing every user and every item as a short vector of learned tastes, then score a pair by how well those vectors line up. That single change cut the cold-start dead zone, doubled top-10 recall, and shrank the model from a multi-gigabyte similarity matrix to a few megabytes of factors. The technique is matrix factorization, and it has been the workhorse of practical recommenders since the Netflix Prize.

This is a post in the series **Recommendation Systems: From Click to Production**, and it sits right where the series turns from "what is the problem" to "what is the first model that actually solves it well." If you have read [collaborative filtering from first principles](/blog/machine-learning/recommendation-systems/collaborative-filtering-from-first-principles), you already know the neighborhood approach and exactly where it strains. Matrix factorization is the answer to that strain. We will build it three times: once on paper, deriving the model and both of its standard optimizers; once from scratch in PyTorch with nothing but two embedding tables; and once with the production libraries `implicit` and `surprise` so you see the real APIs. Then we measure: RMSE for rating prediction and Recall@10 plus NDCG@10 for ranking, on MovieLens-1M with a temporal split, so the numbers mean something.

The frame for the whole series, set up in [what is a recommender system](/blog/machine-learning/recommendation-systems/what-is-a-recommender-system), is the retrieval to ranking to re-ranking funnel fed by the serve-log-train feedback loop, read off the offline-versus-online gap. Matrix factorization is the canonical retrieval and scoring model: it produces a user vector and an item vector, and a recommendation is a maximum-inner-product search over those vectors. By the end of this post you will be able to derive its update rules, choose between stochastic gradient descent and alternating least squares, pick the latent dimension and regularization on principle, and know precisely when a deep model is worth the extra complexity and when it is not. The picture below is the one idea the entire post turns on: a giant, mostly empty ratings table replaced by two small dense factor matrices whose product fills in every blank.

![Diagram contrasting a large sparse user by item ratings matrix that is mostly empty with two small dense factor matrices whose product reconstructs and fills every cell](/imgs/blogs/matrix-factorization-the-workhorse-1.png)

## 1. From neighborhoods to latent factors

Memory-based collaborative filtering answers "what should user $u$ watch?" by finding users who rated the same movies similarly and borrowing their opinions, or by finding movies that were co-rated similarly to ones $u$ liked. It works, and it is interpretable, but it has two structural weaknesses that no amount of tuning removes.

The first is sparsity. A typical interaction matrix is well over 95 percent empty. MovieLens-1M has about 6,040 users, 3,706 movies, and roughly 1,000,209 ratings, which means only about 4.5 percent of the 22.4 million possible cells are filled. To compute the similarity between two users, neighborhood CF needs them to have co-rated some items. When two users have rated zero items in common, their similarity is undefined, and they cannot inform each other at all even if they have identical taste expressed over disjoint movies. The sparser the data, the more pairs fall into that dead zone, and the worse the neighborhood method does precisely where you need it most.

The second is that neighborhoods do not generalize; they interpolate. They can only transfer signal along edges that exist in the co-rating graph. If user A liked items that user B also liked, and B liked some item C that A has not seen, a neighborhood model can pass that signal A to B to C. But the chain must be made of real overlaps. There is no latent structure that says "A and B both love slow-burn psychological thrillers, and so does this third user we have never connected them to."

Latent-factor models attack both problems at once with one move. Instead of comparing users by their raw co-ratings, we posit that there are a small number $k$ of hidden factors that explain taste, and we describe each user $u$ by a vector $p_u \in \mathbb{R}^k$ and each item $i$ by a vector $q_i \in \mathbb{R}^k$. A user-vector coordinate might informally correspond to "how much this person likes action," and the matching item-vector coordinate to "how much action this movie has." We do not name the factors; we learn them. The predicted affinity of user $u$ for item $i$ is the dot product:

$$\hat r_{ui} = p_u^\top q_i = \sum_{f=1}^{k} p_{uf}\, q_{if}.$$

This is collaborative filtering, in the sense that the factors are learned jointly from everyone's ratings, but it is model-based rather than memory-based. Two users with no co-rated items can still be compared, because each one has a vector in the same $k$-dimensional space, and those vectors were pulled into position by the full web of everyone's ratings. The model has effectively factored the giant interaction matrix into a tall thin user matrix $P \in \mathbb{R}^{m \times k}$ and a tall thin item matrix $Q \in \mathbb{R}^{n \times k}$, so that $R \approx P Q^\top$. That is where the name comes from.

Why does this generalize better? Because $k$ is small, the model is forced to compress. It cannot memorize every rating; it has to find the few directions of taste that explain the most variance across all users and items. That compression is regularization by construction: a rank-$k$ approximation of a matrix is the best low-dimensional summary in the least-squares sense, so a low-rank model captures the dominant co-occurrence patterns and discards idiosyncratic noise. The same compression is what lets the model score a user-item pair it has never seen as a filled-in cell. The figure below contrasts a sparse pair under both approaches: neighborhood CF cannot connect two users who share nothing, while the factor model still compares them through their learned vectors.

![Diagram contrasting collaborative filtering neighborhoods that require co-rated overlap with matrix factorization latent vectors that compare any two users in a shared space](/imgs/blogs/matrix-factorization-the-workhorse-8.png)

There is a useful way to see the generalization that is worth stating carefully, because it sits right next to a figure. Think of it as a transitive bridge: if user A and user B have similar factor vectors because they agreed on a hundred movies, and B and C have similar vectors because they agreed on a different hundred, then A and C end up near each other in factor space even though they never co-rated anything. The neighborhood model has no such bridge; the latent space builds it automatically.

### 1.1 The geometry of the dot product

It pays to be precise about what the dot product is doing, because it is the entire scoring engine and the source of both its power and its limits. The dot product $p_u^\top q_i = \lVert p_u \rVert\, \lVert q_i \rVert \cos\theta$ where $\theta$ is the angle between the two vectors. Two things follow. First, the *direction* of a vector encodes taste: if a user's vector points the same way as an item's vector ($\theta$ small, $\cos\theta$ near 1), the score is high; if they point opposite ways ($\theta$ near 180 degrees), the score is strongly negative. Second, the *magnitude* of a vector encodes intensity or popularity: a user with a large-norm vector has strong opinions that swing the dot product hard in both directions, and an item with a large-norm vector is one whose appeal is highly polarized along the taste axes.

This geometry explains a subtle failure mode that bites people who confuse the dot product with cosine similarity. The dot product is *not* a metric and not bounded; it rewards magnitude. An item with a large norm can win the top-$K$ for many users simply because it is "loud" along the dominant factor, which is one way popularity bias creeps into a factor model even after you remove the explicit item bias. If you want pure taste alignment you would normalize to cosine, but for ranking you usually want the magnitude signal, because popularity is genuinely predictive. The right move is to model the popularity explicitly (the item bias $b_i$) so the factors are free to encode taste rather than fame, which is exactly what the biased model in the next section does.

There is also a clean way to see why a *low* rank is the right modeling choice rather than just a computational shortcut. A rank-$k$ matrix is one that can be written as a sum of $k$ rank-one outer products: $R \approx \sum_{f=1}^{k} P_{:,f}\, Q_{:,f}^\top$. Each rank-one term is a single taste pattern, a vector over users times a vector over items, that says "these users like these items more." Choosing $k$ is choosing how many distinct taste patterns the model is allowed to discover. The bet of matrix factorization, borne out across thousands of datasets, is that real preference data is approximately low-rank: a few dozen taste patterns explain most of the structure, and the rest is noise you do not want to fit anyway. That bet is why the model generalizes, and it is the same bet that principal component analysis and every other low-rank method makes.

## 2. The model that actually ships: biases first, factors second

The pure dot-product model is elegant, but if you train it as written you leave most of the achievable accuracy on the table. The reason is that a large fraction of the variation in ratings has nothing to do with the interaction between a particular user and a particular item. Some users are systematically harsh and rate everything a star low; some are generous. Some movies are widely acclaimed and get high ratings from nearly everyone; some are widely panned. None of that is personalization, and the dot product is a terrible tool for representing it, because forcing $p_u^\top q_i$ to encode "this user is generous" wastes capacity that should be spent on taste.

The fix is to pull those main effects out explicitly. The model that wins is:

$$\hat r_{ui} = \mu + b_u + b_i + p_u^\top q_i.$$

Here $\mu$ is the global mean rating across the training set, $b_u$ is a learned scalar bias for user $u$ (how much that user's ratings deviate from the global mean on average), $b_i$ is a learned scalar bias for item $i$ (how much that item is rated above or below average), and $p_u^\top q_i$ is the residual interaction after those main effects are removed. On MovieLens-1M, $\mu \approx 3.58$. The figure below shows how a single prediction is assembled from these four pieces.

![Diagram showing a branching dataflow where a global mean, a user bias, an item bias, and a user-item dot product merge into one predicted rating score](/imgs/blogs/matrix-factorization-the-workhorse-2.png)

The practical lesson, learned the expensive way by every Netflix Prize team, is that biases carry most of the predictable signal. A model that is nothing but $\mu + b_u + b_i$, with no factors at all, already gets you most of the way from the global-mean baseline to a respectable RMSE. The factors then add the personalization that biases cannot express: that this particular user likes this particular kind of movie more than their overall generosity would predict. We will measure this split directly in the results section, and the gap is striking: biases do most of the RMSE work, but factors do almost all of the ranking work. The decomposition is worth seeing as a stack, where each layer adds its contribution to the final number.

![Diagram showing a predicted rating decomposed into stacked layers of global mean then item bias then user bias then the factor dot product summing to a final estimate](/imgs/blogs/matrix-factorization-the-workhorse-4.png)

The objective we minimize is regularized squared error over only the observed ratings. Let $\mathcal{K} = \{(u,i) : r_{ui} \text{ is observed}\}$. Then:

$$\min_{P,Q,b}\; \sum_{(u,i)\in\mathcal{K}} \left( r_{ui} - \mu - b_u - b_i - p_u^\top q_i \right)^2 + \lambda\left( \lVert p_u \rVert^2 + \lVert q_i \rVert^2 + b_u^2 + b_i^2 \right).$$

Two things in this objective are not decoration and deserve emphasis. First, the sum runs over $\mathcal{K}$, the observed entries only. We never penalize the model for what it predicts on the missing cells, because for explicit ratings a missing cell is genuinely unknown, not a zero. (For implicit feedback the story is different and we treat the unobserved cells as weak negatives; that is the subject of [implicit feedback models, ALS and BPR](/blog/machine-learning/recommendation-systems/implicit-feedback-models-als-and-bpr).) Second, the $\lambda$ term is $L_2$ regularization on every learned parameter; it shrinks factors and biases toward zero, which is what prevents a high-$k$ model from memorizing the training ratings. Without it, a model with $k$ large enough can drive training error to zero and generalize terribly.

## 3. Why "SVD" in recommenders is not SVD

You will constantly see matrix factorization for recommendation called "SVD," and this naming is a genuine source of confusion, so it is worth untangling precisely. The singular value decomposition factors a fully observed matrix $A$ as $A = U \Sigma V^\top$, and the rank-$k$ truncation $U_k \Sigma_k V_k^\top$ is provably the best rank-$k$ approximation of $A$ in both the Frobenius and spectral norms, by the Eckart-Young theorem. That is a beautiful, closed-form result. The trouble is the word "fully observed."

True SVD is defined for a complete matrix. Our ratings matrix is 95 percent missing. There is no SVD of a matrix with holes in it. The historical hacks were to fill the missing entries first, for instance with the global mean or row and column means, and then run SVD on the dense imputed matrix. This fails for two reasons. First, imputing with a constant biases the factorization toward that constant: the model spends its capacity fitting the millions of fake entries you invented, not the real ratings. Second, the dense matrix is enormous, so a dense SVD is computationally hopeless at scale.

The technique that actually works, popularized by Simon Funk during the Netflix Prize and forever after confusingly called "SVD" in recommender libraries, is the regularized objective from the previous section, optimized only over observed entries. It has the same $P Q^\top$ shape as a truncated SVD, which is why the name stuck, but it is a different estimator: it never touches the missing cells, it adds bias terms, and it is fit by gradient descent rather than eigendecomposition. The orthogonality and ordered singular values you get from real SVD are gone; the factors are just whatever minimizes regularized error. When the `surprise` library gives you `SVD` and `SVDpp`, this Funk-style model is what you are getting, not Eckart-Young. The whole family is worth seeing laid out, because the names proliferate and the relationships are easy to lose.

![Diagram of the matrix factorization family branching into explicit-rating methods like Funk-SVD and biased MF and SVD plus plus and implicit-feedback methods like weighted ALS and BPR](/imgs/blogs/matrix-factorization-the-workhorse-6.png)

SVD++ extends biased MF by adding implicit signal. The intuition is that the *set* of items a user has rated, regardless of the rating value, tells you something about them. SVD++ augments the user representation with a sum of per-item implicit-factor vectors over the items in the user's history $N(u)$:

$$\hat r_{ui} = \mu + b_u + b_i + q_i^\top\!\left( p_u + |N(u)|^{-1/2} \sum_{j \in N(u)} y_j \right),$$

where each $y_j \in \mathbb{R}^k$ is a learned implicit-feedback vector for item $j$. The term $|N(u)|^{-1/2} \sum_{j} y_j$ shifts the user vector based on what they chose to interact with, not just how they scored it. On the Netflix data this gave Koren a measurable RMSE improvement over plain biased MF, because "the user bothered to rate this movie at all" is informative even before you know the score. The cost is more parameters and slower training, since each prediction now sums over the user's whole history.

## 4. The science: deriving the update rules

We have an objective. Now we minimize it. There are two standard algorithms, and the choice between them is one of the most consequential and most misunderstood decisions in classical recsys. We derive both.

### 4.1 Stochastic gradient descent

SGD walks through the observed ratings one at a time. For a single rating $(u,i)$, define the prediction error:

$$e_{ui} = r_{ui} - \hat r_{ui} = r_{ui} - \mu - b_u - b_i - p_u^\top q_i.$$

The per-rating loss (the contribution of this one observation, with its share of the regularizer) is:

$$L_{ui} = e_{ui}^2 + \lambda\left( \lVert p_u \rVert^2 + \lVert q_i \rVert^2 + b_u^2 + b_i^2 \right).$$

We need the gradient with respect to each parameter. Take $p_u$ first. The error $e_{ui}$ depends on $p_u$ through the term $-p_u^\top q_i$, so $\partial e_{ui} / \partial p_u = -q_i$. By the chain rule:

$$\frac{\partial L_{ui}}{\partial p_u} = 2 e_{ui} \cdot \frac{\partial e_{ui}}{\partial p_u} + 2\lambda p_u = -2 e_{ui} q_i + 2\lambda p_u.$$

The factor of 2 folds into the learning rate, so an SGD step that moves opposite the gradient is:

$$p_u \leftarrow p_u + \eta\left( e_{ui}\, q_i - \lambda p_u \right).$$

By the identical argument, since $\partial e_{ui}/\partial q_i = -p_u$:

$$q_i \leftarrow q_i + \eta\left( e_{ui}\, p_u - \lambda q_i \right).$$

For the biases, $\partial e_{ui}/\partial b_u = -1$ and $\partial e_{ui}/\partial b_i = -1$, so:

$$b_u \leftarrow b_u + \eta\left( e_{ui} - \lambda b_u \right), \qquad b_i \leftarrow b_i + \eta\left( e_{ui} - \lambda b_i \right).$$

These four updates are the entire algorithm. Read them and the dynamics are clear: when the model under-predicts ($e_{ui} > 0$), it pushes $p_u$ in the direction of $q_i$ and vice versa, so the two vectors align more and the dot product grows; the $-\lambda p_u$ term constantly tugs everything back toward the origin. Each update touches only the parameters of one user and one item, costs $O(k)$ arithmetic, and you make one such update per observed rating per epoch. So a full SGD epoch over MovieLens-1M is about $1{,}000{,}209 \times k$ multiply-adds, which for $k=50$ is roughly 50 million operations: a fraction of a second of real compute, dominated in practice by Python loop overhead unless you vectorize or move to PyTorch.

One important caveat that the clean update rules hide: the joint objective over $P$ and $Q$ is *not convex*. The product $p_u^\top q_i$ is bilinear, so the loss surface has many local minima and saddle points, and it is invariant under a rotation of the latent space (you can multiply $P$ by an orthogonal matrix and $Q$ by its inverse and get the identical predictions). SGD finds a good local minimum in practice, not the global one, and the specific minimum depends on the random initialization, which is why we initialize the factors with small Gaussian noise rather than zeros: zero-initialized factors have zero gradient on the dot-product term and would never break symmetry. The biases, by contrast, can start at zero because their gradient does not vanish there. This non-convexity is not a problem in practice for MF (the local minima are nearly equivalent in quality), but it is the reason there is no closed-form solution for the joint problem and the reason ALS, which is convex *per block*, is an attractive alternative.

A practical refinement worth knowing is mini-batch SGD, which is what the PyTorch implementation in section 5 actually uses. Instead of one rating at a time, you process a batch of $B$ ratings, compute the gradient of the average loss over the batch, and take one step. This vectorizes the $O(k)$ work into a single matrix operation over the batch, which is dramatically faster on a GPU and lets you use adaptive optimizers like Adam that maintain per-parameter learning rates and converge in fewer epochs than vanilla SGD. The math is identical; only the granularity of the step changes. The trade is that very large batches average away the noise that helps SGD escape shallow minima, so a moderate batch (a few thousand ratings) is the usual compromise.

### 4.2 Alternating least squares

SGD is cheap per step but inherently sequential and sensitive to the learning rate. Alternating least squares takes a completely different route that exploits a structural fact: the objective is not jointly convex in $P$ and $Q$ together, but it *is* convex in $P$ when $Q$ is held fixed, and convex in $Q$ when $P$ is held fixed. Each of those sub-problems is an ordinary ridge regression with a closed-form solution. So ALS alternates: fix $Q$, solve exactly for every $p_u$; fix $P$, solve exactly for every $q_i$; repeat.

Let us derive the user solve. Drop the bias terms for a moment to keep the algebra clean (biases can be folded in by augmenting the vectors with a constant coordinate, or solved as a separate step). Fix all item vectors. The part of the objective that depends on a single user's vector $p_u$ is:

$$L(p_u) = \sum_{i \in I_u} \left( r_{ui} - p_u^\top q_i \right)^2 + \lambda \lVert p_u \rVert^2,$$

where $I_u$ is the set of items user $u$ has rated. Let $Q_{I_u} \in \mathbb{R}^{|I_u| \times k}$ be the matrix whose rows are the item vectors $q_i$ for $i \in I_u$, and let $r_u \in \mathbb{R}^{|I_u|}$ be the corresponding observed ratings. Then in matrix form:

$$L(p_u) = \lVert r_u - Q_{I_u} p_u \rVert^2 + \lambda \lVert p_u \rVert^2.$$

This is textbook ridge regression. Take the gradient and set it to zero:

$$\frac{\partial L}{\partial p_u} = -2 Q_{I_u}^\top \left( r_u - Q_{I_u} p_u \right) + 2\lambda p_u = 0.$$

Rearranging:

$$Q_{I_u}^\top Q_{I_u}\, p_u + \lambda p_u = Q_{I_u}^\top r_u \;\;\Longrightarrow\;\; \left( Q_{I_u}^\top Q_{I_u} + \lambda I \right) p_u = Q_{I_u}^\top r_u.$$

So the optimal user vector, with all item vectors held fixed, is:

$$\boxed{\,p_u = \left( Q_{I_u}^\top Q_{I_u} + \lambda I \right)^{-1} Q_{I_u}^\top r_u\,.}$$

This is a $k \times k$ linear solve, one per user. The $\lambda I$ term is what makes the matrix invertible even when a user has rated fewer than $k$ items, which is exactly the regularization doing double duty: it both prevents overfitting and guarantees the solve is well-posed. The item solve is symmetric: fix $P$, and for each item $i$,

$$q_i = \left( P_{U_i}^\top P_{U_i} + \lambda I \right)^{-1} P_{U_i}^\top r_i,$$

where $U_i$ is the set of users who rated item $i$. ALS just alternates these two exact solves until the objective stops decreasing, usually in 10 to 20 sweeps.

The cost: forming $Q_{I_u}^\top Q_{I_u}$ costs $O(|I_u| k^2)$, and inverting the $k \times k$ system costs $O(k^3)$. Summed over all users, a full user sweep is $O\!\left(\sum_u |I_u| k^2 + m k^3\right) = O(|\mathcal{K}| k^2 + m k^3)$. For ML-1M with $k=50$, the $k^3$ term is $125{,}000$ flops per user, trivial, and the dominant cost is the $|\mathcal{K}| k^2$ term, about $1M \times 2500 = 2.5$ billion flops per sweep: more than SGD per iteration, but ALS converges in far fewer iterations and, crucially, each user solve is independent of every other user solve.

A detail people skip: how the biases fit into the closed-form solve. The clean trick is to absorb them into the vectors. Augment every item vector with two extra coordinates, a constant 1 and a slot that will carry the item bias, and augment every user vector symmetrically, so the dot product of the augmented vectors automatically reproduces $b_u + b_i + p_u^\top q_i$. The global mean $\mu$ is subtracted from the ratings up front so the model only has to fit residuals. With that augmentation the bias terms ride along in the same ridge solve and you never write a separate update for them. Production ALS implementations like `implicit` handle this internally; the point is that the closed form does not lose the biases, it just carries them in extra coordinates. One more practical note: because $Q_{I_u}^\top Q_{I_u}$ is a small symmetric positive-definite $k \times k$ matrix, you never compute an explicit inverse. You factor it with a Cholesky decomposition and solve the triangular systems, which is both faster and numerically more stable than forming $(\cdot)^{-1}$ and multiplying.

There is one more wrinkle that makes ALS the natural choice for implicit data specifically, and it is worth a sentence here even though the full treatment is in the implicit-feedback post. For implicit feedback you want to sum the squared error over *all* user-item pairs, observed and not, with a confidence weight that is high for observed interactions and low (but nonzero) for the missing ones. Naively that is an $O(m \times n)$ sum, hopeless at scale. The algebraic gift of weighted ALS is that the per-user solve can be rewritten so the unobserved-pair contribution becomes a single shared $Q^\top Q$ term computed once per sweep, plus a sparse correction over the user's observed items. That decomposition is what turns an intractable all-pairs objective into the same cheap per-user solve, and it is the reason ALS, not SGD, dominates implicit-feedback factorization.

### 4.3 When to use which

That independence is the headline. Every $p_u$ solve uses only the fixed item matrix and that user's own ratings, so you can compute all $m$ user vectors fully in parallel across cores or machines, then all $n$ item vectors in parallel. ALS is embarrassingly parallel; SGD is not, because consecutive updates touch shared parameters and the standard remedy is the lock-free Hogwild trick, which works but is fiddly. The figure below lays out the trade-off across update style, parallelism, cost, and data regime.

![Comparison matrix of stochastic gradient descent versus alternating least squares across update rule, parallelism, per-iteration cost, and the data regime each one suits best](/imgs/blogs/matrix-factorization-the-workhorse-3.png)

Laid out as a table, the decision is clean:

| Property | SGD | ALS |
|---|---|---|
| Update | per-rating gradient step, $O(k)$ | closed-form ridge solve, $O(\lvert I_u\rvert k^2 + k^3)$ per row |
| Hyperparameters | learning rate $\eta$ + $\lambda$ (lr can diverge) | $\lambda$ only (no learning rate) |
| Parallelism | hard (Hogwild lock-free trick) | embarrassingly parallel across rows |
| Convergence | many epochs, noisy descent | few sweeps, monotone per block |
| Best for | sparse explicit data, single machine | implicit all-pairs data, clusters |

The practitioner's rule of thumb: reach for SGD on sparse explicit-rating data, especially on a single machine, where its cheap $O(k)$ steps and easy mini-batching make it fast and where the learning rate is tunable without much pain. Reach for ALS on implicit-feedback data and on clusters, because implicit feedback turns every unobserved cell into a weak negative, which makes the per-row least-squares formulation natural and the parallelism essential. The widely used weighted ALS for implicit data, from Hu, Koren, and Volinsky's 2008 paper, is precisely the closed-form solve above with a confidence weight on each entry; we cover it in the implicit-feedback post. For explicit ratings on a laptop, SGD or its mini-batch cousin in PyTorch is the path of least resistance, and that is what we build next.

## 5. Implementing MF from scratch in PyTorch

The cleanest way to see that matrix factorization is "just" two embedding tables is to build it in PyTorch with `nn.Embedding`. An embedding layer is exactly a lookup table of learnable vectors, which is precisely what $P$ and $Q$ are. We add scalar-embedding tables for the biases and a single learnable global mean.

```python
import torch
import torch.nn as nn

class BiasedMF(nn.Module):
    def __init__(self, n_users, n_items, k=50):
        super().__init__()
        self.user_factors = nn.Embedding(n_users, k)
        self.item_factors = nn.Embedding(n_items, k)
        self.user_bias = nn.Embedding(n_users, 1)
        self.item_bias = nn.Embedding(n_items, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))
        # Small init so dot products start near zero; biases start at zero.
        nn.init.normal_(self.user_factors.weight, std=0.05)
        nn.init.normal_(self.item_factors.weight, std=0.05)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

    def forward(self, u, i):
        p = self.user_factors(u)          # (B, k)
        q = self.item_factors(i)          # (B, k)
        dot = (p * q).sum(dim=1)          # (B,) elementwise then sum
        bu = self.user_bias(u).squeeze(1) # (B,)
        bi = self.item_bias(i).squeeze(1) # (B,)
        return self.global_bias + bu + bi + dot
```

The `forward` is the model equation line for line: look up the user and item factor vectors, take the elementwise product and sum it (that is the dot product over a batch), add the two biases and the global term. Note we compute the dot product with `(p * q).sum(dim=1)` rather than a matrix multiply, because we only need the diagonal of $P Q^\top$ for the specific pairs in the batch, never the full outer product.

Now the training loop. We feed it mini-batches of observed (user, item, rating) triples, use mean squared error on the predictions, and let PyTorch's `weight_decay` apply the $L_2$ regularization for us (weight decay in Adam is exactly the $\lambda$ shrinkage we derived, applied to every parameter).

```python
from torch.utils.data import TensorDataset, DataLoader

def train_mf(model, train_triples, epochs=20, lr=5e-3, wd=2e-5, bs=4096):
    # train_triples: LongTensor (N,2) of (user, item), FloatTensor (N,) ratings
    users, items, ratings = train_triples
    ds = TensorDataset(users, items, ratings)
    dl = DataLoader(ds, batch_size=bs, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    loss_fn = nn.MSELoss()
    for epoch in range(epochs):
        model.train()
        total = 0.0
        for u, i, r in dl:
            opt.zero_grad()
            pred = model(u, i)
            loss = loss_fn(pred, r)
            loss.backward()
            opt.step()
            total += loss.item() * len(r)
        print(f"epoch {epoch:02d}  train MSE {total/len(ratings):.4f}")
    return model
```

A few details that matter in practice and that people get wrong. First, `weight_decay` in Adam applies the same $\lambda$ to factors and biases; if you want to regularize biases more weakly (often a good idea, since biases have little capacity to overfit) you can split the parameters into two groups with different `weight_decay`. Second, MSE on ratings optimizes RMSE, which is a rating-prediction metric, not a ranking metric; we will see this bite us in the results. Third, the learning rate and `weight_decay` are the two knobs that matter most: too high an `lr` and the loss oscillates or diverges, too high a `wd` and the model underfits into bias-only behavior.

To turn a trained model into recommendations, you score all items for a user and take the top ones. Because the user factor is fixed once chosen, scoring all items is a single matrix-vector product, which is exactly a maximum-inner-product search and the reason MF is the standard retrieval model:

```python
@torch.no_grad()
def recommend(model, user_id, seen_items, top_k=10, n_items=None):
    model.eval()
    u = torch.full((n_items,), user_id, dtype=torch.long)
    all_items = torch.arange(n_items)
    scores = model(u, all_items)        # score every item for this user
    scores[list(seen_items)] = -1e9     # mask items already interacted with
    top = torch.topk(scores, top_k).indices
    return top.tolist()
```

Masking already-seen items is not optional. If you forget it, the model will happily re-recommend the movies the user has already rated, which inflates nothing useful and ruins the live experience. It is also a classic source of a leaked offline metric: if your evaluation does not mask training-time interactions, your Recall@10 will look better than it is.

To run the bias-only ablation that the results section reports, you do not need a second model: instantiate `BiasedMF` with `k` set very small and freeze the factor tables, or more cleanly, zero out the dot-product term in `forward` and train only the bias embeddings and the global mean. The cleanest version is a three-line model that returns `self.global_bias + bu + bi` with no factors at all. Training it is the same loop. The reason to bother is that the bias-only model is your true baseline for the personalization question: any factor model must beat it on ranking, and the size of that gap is the value the factors add. Reporting MF accuracy without the bias-only number next to it hides where the gains actually come from, and as the results table shows, the gains come overwhelmingly from ranking, not RMSE.

A note on reproducibility that saves real pain: set the random seed before constructing the model, because the factor initialization is random and a non-convex objective means different seeds land in slightly different minima with slightly different metrics. When you compare $k=50$ to $k=100$, run each with the same seed (or average over a few seeds) so the comparison reflects capacity, not initialization luck. Equally, fix the data split seed: comparing two models on two different temporal splits is comparing nothing. These two seeds, model and split, are the difference between a real ablation and noise.

## 6. The production libraries: implicit ALS and surprise SVD

Building from scratch is the right way to understand the model; reaching for a library is the right way to ship it. Two are standard. For explicit ratings, `surprise` gives you the Funk-style `SVD` and `SVDpp`. For implicit feedback, `implicit` gives you a fast, multi-threaded weighted-ALS. Here is the `surprise` path on MovieLens-1M, with the temporal split discussed below applied beforehand:

```python
from surprise import SVD, Dataset, Reader, accuracy

reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(train_df[["userId", "movieId", "rating"]], reader)
trainset = data.build_full_trainset()

algo = SVD(
    n_factors=50,     # latent dimension k
    n_epochs=30,      # SGD epochs
    lr_all=0.005,     # learning rate eta
    reg_all=0.02,     # L2 regularization lambda
    biased=True,      # include mu, b_u, b_i  (set False to ablate)
)
algo.fit(trainset)

# Predict on the held-out temporal test set
preds = [algo.predict(str(u), str(i), r) for u, i, r in test_triples]
print("RMSE:", accuracy.rmse(preds, verbose=False))
```

The `biased=True` flag is the exact ablation knob we want: set it `False` and you get pure $p_u^\top q_i$ with no bias terms, so you can measure how much the biases buy you. `surprise`'s `SVD` is the Funk-SVD with biases, optimized by the same SGD updates we derived in section 4.1. `SVDpp` adds the implicit-history term from section 3 and is noticeably slower.

For implicit feedback, the `implicit` library wants a sparse item-by-user matrix of confidences. Even though MovieLens has explicit ratings, you can treat it as implicit (rated or not) to see the ALS path:

```python
import implicit
from scipy.sparse import csr_matrix

# Build a user x item confidence matrix. For implicit ALS, the value is a
# confidence weight; here we use 1 + alpha * rating as a simple scheme.
alpha = 40.0
rows, cols, vals = [], [], []
for u, i, r in train_triples:
    rows.append(u); cols.append(i); vals.append(1.0 + alpha * (r / 5.0))
user_items = csr_matrix((vals, (rows, cols)), shape=(n_users, n_items))

model = implicit.als.AlternatingLeastSquares(
    factors=50,            # k
    regularization=0.05,   # lambda
    iterations=15,         # ALS sweeps
    use_gpu=False,
)
model.fit(user_items)      # multi-threaded closed-form solves

ids, scores = model.recommend(
    userid=0, user_items=user_items[0], N=10, filter_already_liked_items=True
)
```

Note `filter_already_liked_items=True` does the seen-item masking for you, and `model.fit` runs the closed-form ALS solve from section 4.2 across all rows in parallel threads. The `factors`, `regularization`, and `iterations` arguments map directly to $k$, $\lambda$, and the number of alternating sweeps. The confidence-weighting scheme ($1 + \alpha r$) is the Hu-Koren-Volinsky trick for implicit data; treating explicit ratings this way throws away the rating magnitude, which is why for explicit data you would prefer `surprise`'s rating-aware SVD, and for true clicks-and-plays you would prefer `implicit`.

### 6.1 Serving MF as retrieval: the dot product is a MIPS problem

The reason matrix factorization is the canonical *retrieval* model, the first stage of the funnel that narrows a billion items to a few hundred candidates, is the shape of its scoring function. Recommending for a user is "find the items whose vectors have the largest dot product with this user's vector." That is exactly maximum-inner-product search (MIPS), and there is a mature ecosystem of approximate-nearest-neighbor (ANN) indexes built to answer it in milliseconds over hundreds of millions of vectors. You train MF offline, dump the item factor matrix $Q$ into an ANN index once, and at request time you compute the user vector (a single lookup for a known user) and query the index. The whole online cost is one embedding lookup plus one ANN query.

Here is the build with `faiss`, the standard library for this:

```python
import faiss
import numpy as np

# item_factors: (n_items, k) float32 array from a trained MF model
item_factors = model.item_factors.astype("float32")
k_dim = item_factors.shape[1]

# Exact inner-product index (good baseline; brute force MIPS)
index_flat = faiss.IndexFlatIP(k_dim)
index_flat.add(item_factors)

# Approximate index for scale: IVF partitions the space into cells
quantizer = faiss.IndexFlatIP(k_dim)
index_ivf = faiss.IndexIVFFlat(quantizer, k_dim, 256, faiss.METRIC_INNER_PRODUCT)
index_ivf.train(item_factors)     # learn the cell centroids
index_ivf.add(item_factors)
index_ivf.nprobe = 16             # cells to search: higher = better recall, slower

# Retrieve top-200 candidates for a user vector
user_vec = model.user_factors[user_id].reshape(1, -1).astype("float32")
scores, ids = index_ivf.search(user_vec, 200)
```

The `IndexFlatIP` is exact brute force: correct but $O(n)$ per query, which is fine up to a few million items. `IndexIVFFlat` partitions the vectors into cells and searches only the `nprobe` cells nearest the query, trading a little recall for a large speedup; `IndexIVFPQ` adds product quantization to compress the vectors and shrink the index to a fraction of memory. The knob `nprobe` is the recall-versus-latency dial: raise it and you find more of the true top items at the cost of latency. This is the heart of the retrieval stage, and it is why a model as old as MF is still completely viable at billion-item scale: the dot-product structure is precisely what fast ANN indexes are built to serve. The ANN recall-versus-latency trade and the index choices get a full treatment later in the series; the point here is that MF hands you the right kind of representation for free.

## 7. Evaluating honestly: RMSE is not enough

Here is the trap that the team in the opening story fell into. They picked the model with the lowest RMSE and shipped it. RMSE measures how close your predicted rating is to the true rating, averaged over the test set. It is a perfectly good metric for the question "how well do you predict a held-out rating," but it is the wrong question. The product question is "do you put good items at the top of the list," and that is a ranking question, measured by Recall@K and NDCG@K. The two can diverge sharply: a model can shave RMSE by getting the boring middle of the distribution slightly more accurate while doing nothing for the top of any user's list.

So we measure both. Recall@10 asks: of the items the user actually engaged with in the test period, what fraction appear in our top 10 recommendations? NDCG@10 (normalized discounted cumulative gain) is similar but rewards putting relevant items higher in the list, with a logarithmic position discount, normalized so a perfect ranking scores 1.0. The definitions, for a user with relevant set $\text{Rel}$ and a ranked list:

$$\text{Recall@}K = \frac{|\{\text{top-}K\} \cap \text{Rel}|}{|\text{Rel}|}, \qquad \text{NDCG@}K = \frac{\sum_{j=1}^{K} \frac{\mathbb{1}[\text{item}_j \in \text{Rel}]}{\log_2(j+1)}}{\text{IDCG@}K}.$$

The split also matters. We use a temporal split: for each user, sort their ratings by timestamp and put the most recent fraction (say the last 20 percent, or all ratings after a global cutoff date) in the test set, the rest in train. This mimics production, where you always predict the future from the past, and it avoids the leakage of a random split, which lets the model peek at a user's later behavior to predict their earlier behavior. A random split flatters every model and especially flatters the ones that overfit; the temporal split is the honest one, and it is the one the series uses throughout. The offline-online gap that this still leaves open is the subject of [offline versus online, the two worlds of recsys](/blog/machine-learning/recommendation-systems/offline-vs-online-the-two-worlds-of-recsys).

A compact eval harness that computes both metrics from a scoring function:

```python
import numpy as np

def evaluate(score_fn, test_by_user, seen_by_user, n_items, k=10):
    recalls, ndcgs = [], []
    idcg = sum(1.0 / np.log2(j + 2) for j in range(k))  # ideal DCG cap
    for u, relevant in test_by_user.items():
        scores = score_fn(u)                  # (n_items,) scores for user u
        scores[list(seen_by_user.get(u, []))] = -1e9   # mask train items
        topk = np.argpartition(-scores, k)[:k]
        topk = topk[np.argsort(-scores[topk])]         # sort the top-k
        rel = set(relevant)
        hits = [1 if it in rel else 0 for it in topk]
        recalls.append(sum(hits) / min(len(rel), k))
        dcg = sum(h / np.log2(j + 2) for j, h in enumerate(hits))
        ndcgs.append(dcg / idcg)
    return float(np.mean(recalls)), float(np.mean(ndcgs))
```

One subtlety baked into this harness: we compute *full* metrics, ranking against the entire catalog, not sampled metrics that rank the true item against a handful of random negatives. Sampled metrics are faster but, as the KDD 2020 paper "On Sampled Metrics for Item Recommendation" by Krichene and Rendle showed, they can be inconsistent and even reverse model rankings. For a fair comparison you want full metrics when you can afford them, and MovieLens-1M is small enough that you always can.

There is a deeper bias lurking even in a clean temporal split, and it is worth naming because it is the single biggest reason offline numbers mislead. Your observed ratings are *missing not at random* (MNAR): users rate the movies they chose to watch, which are the movies the old recommender or their own taste already surfaced, so the test set is not a uniform sample of preferences. It over-represents items users already liked and under-represents the long tail they never saw. A naive Recall@10 computed on this biased sample rewards a model for predicting what users would have found anyway. The principled correction is inverse-propensity scoring (IPS): weight each test observation by the inverse of its probability of being observed, $\hat r_{\text{IPS}} = \frac{1}{|\mathcal{K}|}\sum_{(u,i)} \frac{\text{loss}_{ui}}{\hat p_{ui}}$, which up-weights the rarely-observed items so the estimate approximates what you would see under uniform exposure. IPS has high variance and needs a propensity model you usually do not have, so most teams settle for a temporal split plus an online A/B test, but knowing that the offline number is an MNAR-biased estimate keeps you appropriately skeptical of it. The full off-policy and debiasing treatment is later in the series; for MF, just remember that the temporal split is honest about *time* but not about *exposure*.

## 8. Results: where biases stop and factors start

Now the measurement. The table below reports RMSE, Recall@10, and NDCG@10 on MovieLens-1M with a temporal split (most recent 20 percent of each user's ratings held out, full-catalog ranking, training items masked). These are representative numbers in the well-established range for this dataset and protocol; treat them as the shape of the result, since exact figures shift a little with the split seed, the number of epochs, and the regularization. Always reproduce on your own split before quoting a number.

| Model | RMSE | Recall@10 | NDCG@10 | Notes |
|---|---|---|---|---|
| Global mean ($\mu$) | 1.117 | 0.000 | 0.000 | predicts 3.58 for everyone, no ranking signal |
| Bias-only ($\mu + b_u + b_i$) | 0.924 | 0.061 | 0.071 | same top list shape for all users |
| Item-item CF (k=40 neighbors) | 0.905 | 0.118 | 0.131 | strains on sparse and cold users |
| MF, k=10 (biased) | 0.892 | 0.121 | 0.138 | small capacity, underfits a little |
| MF, k=50 (biased) | 0.861 | 0.159 | 0.181 | the sweet spot for this dataset |
| MF, k=100 (biased) | 0.860 | 0.166 | 0.188 | marginal gain, more overfit risk |

Two readings jump out. First, on RMSE the biases do most of the work: going from the global mean (1.117) to bias-only (0.924) closes about two-thirds of the total gap to the best model (0.860). Adding factors at $k=50$ only buys another 0.063 of RMSE. If you stopped at RMSE you might conclude factors barely matter. Second, on ranking the story inverts completely. Bias-only gets Recall@10 of just 0.061, because with no personalization every user gets essentially the same popularity-ordered list. The factors triple it to 0.159. The biases predict ratings well and rank badly; the factors rank well. That is the whole reason matrix factorization, not bias-only regression, is the workhorse. The before-and-after below makes the split visual.

![Diagram contrasting a bias-only model with good RMSE but poor Recall against a full matrix factorization model with a small RMSE gain but a large ranking improvement](/imgs/blogs/matrix-factorization-the-workhorse-5.png)

#### Worked example: predicting one rating by hand

Let us compute a single prediction with concrete numbers so the model equation stops being abstract. Suppose on the trained model we have, for user $u$ and item $i$:

- global mean $\mu = 3.58$
- user bias $b_u = -0.30$ (this user rates everything 0.3 stars below average, a slightly harsh rater)
- item bias $b_i = +0.62$ (this is a widely acclaimed film, rated well above average)
- user factor $p_u = [\,0.9,\; -0.4,\; 0.2\,]$
- item factor $q_i = [\,0.6,\; 0.1,\; 0.5\,]$ (a $k=3$ toy for arithmetic)

The dot product is $p_u^\top q_i = (0.9)(0.6) + (-0.4)(0.1) + (0.2)(0.5) = 0.54 - 0.04 + 0.10 = 0.60$.

The full prediction is $\hat r_{ui} = \mu + b_u + b_i + p_u^\top q_i = 3.58 - 0.30 + 0.62 + 0.60 = 4.50$ stars.

Notice the structure of the answer. The biases alone give $3.58 - 0.30 + 0.62 = 3.90$: this harsh user, faced with an acclaimed film, would rate it about 3.9 on average effects alone. The factor dot product of $+0.60$ says their *specific* taste aligns with this film beyond the average effects, pushing the prediction to 4.50. That extra 0.60 is the personalization the biases cannot express, and it is exactly what moves an item up a user's ranked list. This is the same decomposition the stacked figure in section 2 shows, with real numbers attached.

#### Worked example: one SGD update step

Now take a single SGD update with numbers, using the rules derived in section 4.1. Suppose the true rating is $r_{ui} = 5.0$ but our current prediction (from the example above) is $\hat r_{ui} = 4.50$, so the error is $e_{ui} = 5.0 - 4.50 = 0.50$. Take learning rate $\eta = 0.01$ and regularization $\lambda = 0.02$.

Update the user factor $p_u \leftarrow p_u + \eta(e_{ui} q_i - \lambda p_u)$. Component by component:

- coordinate 1: $0.9 + 0.01\,((0.50)(0.6) - (0.02)(0.9)) = 0.9 + 0.01(0.300 - 0.018) = 0.9 + 0.00282 = 0.90282$
- coordinate 2: $-0.4 + 0.01\,((0.50)(0.1) - (0.02)(-0.4)) = -0.4 + 0.01(0.050 + 0.008) = -0.4 + 0.00058 = -0.39942$
- coordinate 3: $0.2 + 0.01\,((0.50)(0.5) - (0.02)(0.2)) = 0.2 + 0.01(0.250 - 0.004) = 0.2 + 0.00246 = 0.20246$

So $p_u$ moves toward $q_i$, because we under-predicted and need a bigger dot product. Update the user bias $b_u \leftarrow b_u + \eta(e_{ui} - \lambda b_u) = -0.30 + 0.01(0.50 - (0.02)(-0.30)) = -0.30 + 0.01(0.506) = -0.29494$. The bias creeps up too. If you recompute the prediction after the update it rises slightly toward 5.0, which is the gradient step doing its job: one observation, one small correction, repeated a million times per epoch.

## 9. A debugging narrative: when the numbers lie

Let me walk through the failure modes I have actually hit shipping matrix factorization, because the model is simple enough that almost every problem is a data or evaluation problem, not a modeling one, and recognizing the symptom is most of the cure.

**Symptom one: train RMSE keeps dropping but test RMSE turns up.** This is textbook overfitting, and with MF the cause is almost always $k$ too high relative to $\lambda$. The model has enough factor capacity to memorize individual ratings. The fix is not a smaller $k$ first; it is more regularization. Raise $\lambda$ until the train and test curves track each other, and only then consider whether $k$ is buying you anything. The diagnostic is to plot both curves per epoch: if train falls smoothly while test bottoms out and rises, you are overfit; if both plateau high, you are underfit and should raise $k$ or lower $\lambda$.

**Symptom two: training loss explodes to NaN after a few epochs.** The learning rate is too high for the bilinear loss surface. The error term $e_{ui}$ scales the update, so an early large error with a high $\eta$ can overshoot, which produces a larger error next step, which overshoots more, a runaway. Halve $\eta$ and the problem usually vanishes. If you are on Adam, the adaptive learning rate makes this rarer but not impossible; clip the gradient or warm up the learning rate if it persists. This is the single most common reason a from-scratch MF "does not work" on the first try.

**Symptom three: RMSE is great but the recommendations are garbage.** This is the trap from the opening of the post, and it is not a bug at all; it is a metric mismatch. RMSE optimizes accuracy on the whole rating distribution, which is dominated by the dense middle, while the product cares only about the top of each user's list, which is a tiny, hard part of the distribution. The fix is to evaluate and ideally train for the actual objective: switch to Recall@K and NDCG@K for model selection, and if ranking is what you ship, consider a ranking loss (BPR or WARP) instead of squared error. A model tuned for RMSE and a model tuned for NDCG can pick different $k$ and different $\lambda$; do not assume the RMSE-optimal model is the ranking-optimal one.

**Symptom four: offline Recall@10 went up but online engagement is flat or down.** Now you have left the model and entered the offline-online gap, the recurring theme of this series. The usual culprits are leakage in the offline split (a random split instead of temporal lets the model peek at the future), missing-not-at-random feedback (you only observe ratings for items users chose to engage with, so the test set is not a random sample of preferences), and position bias (the offline metric treats all positions equally but the live UI does not). The discipline is to never trust an offline win until you can explain why it should transfer, and to confirm with an online A/B test. The honest path through that gap is the whole point of [offline versus online, the two worlds of recsys](/blog/machine-learning/recommendation-systems/offline-vs-online-the-two-worlds-of-recsys).

**Symptom five: a third of the catalog never gets recommended.** The factor model has learned tiny, near-origin vectors for long-tail items with few ratings, so they never win a top-$K$. Combined with the feedback loop (you only collect interactions on items you show), this is how a recommender silently collapses its catalog. The mitigations are explicit: boost exploration so the model gathers data on tail items, initialize cold items from content features so they start with a nonzero vector, or add a diversity or coverage term to the re-ranking stage. Recognizing that this is a *systemic* failure of the serve-log-train loop, not a flaw in the factorization, is what lets you fix it at the right layer.

Run through these five and you will diagnose almost any MF problem in minutes. The pattern is consistent: the model is rarely the problem; the data split, the metric, and the loop around the model usually are.

## 10. Choosing k, lambda, and the learning rate

Three hyperparameters govern a matrix factorization model, and each one is a direct lever on the bias-variance trade-off. Getting them right is most of the practical work.

**The latent dimension $k$.** This is model capacity. Too small and the model underfits: it cannot represent enough distinct directions of taste, so RMSE plateaus high and ranking is mediocre. Too large and the model has the capacity to memorize training ratings, which hurts generalization unless regularization grows with it; memory and training cost also grow linearly in $k$. On MovieLens-1M the curve flattens around $k = 50$ to $100$; on a large industrial dataset with billions of interactions you might run $k = 128$ to $512$. The figure below summarizes the trade-off across RMSE, ranking, memory, and overfit risk.

![Comparison matrix of latent dimension choices ten and fifty and one hundred against RMSE and Recall and memory footprint and overfitting risk](/imgs/blogs/matrix-factorization-the-workhorse-7.png)

#### Worked example: the memory arithmetic of k

Suppose you have 10 million users and 1 million items, and you are choosing between $k=64$ and $k=256$. The factor tables hold $(10^7 + 10^6) \times k$ float32 parameters, which is $4 \times (1.1 \times 10^7) \times k$ bytes. At $k=64$ that is $4 \times 1.1\times10^7 \times 64 \approx 2.8$ GB. At $k=256$ it is about $11.3$ GB. That fourfold jump is the difference between an embedding table that fits comfortably on one host and one that may force you to shard the table across hosts or move it to a parameter server, with all the serving latency and operational cost that implies. The lesson: $k$ is not just an accuracy knob, it is a deployment decision, and a 0.005 RMSE gain from doubling $k$ twice may not be worth tripling your serving memory. This is the kind of Pareto trade-off the [recommender systems playbook](/blog/machine-learning/recommendation-systems/the-recommender-systems-playbook) returns to again and again.

**The regularization $\lambda$.** This controls how hard the model is pulled toward the origin. Too small and a high-$k$ model overfits, training RMSE drops while test RMSE rises, the classic variance blow-up. Too large and everything shrinks toward zero, the factors collapse, and you are left with essentially the bias-only model. The right $\lambda$ is the one that minimizes validation error, found by a small grid search; on ML-1M values around $0.02$ to $0.1$ are typical. A useful trick from the Netflix work is to use different $\lambda$ for factors and biases, since biases have far less capacity to overfit and benefit from lighter regularization.

**The learning rate $\eta$ (for SGD).** Too high and the loss diverges or oscillates; you will see the training MSE jump up between epochs or go to NaN. Too low and training crawls and may stop short of the minimum in your epoch budget. Values around $0.005$ to $0.01$ are common, often with a decay schedule that halves $\eta$ when validation error stalls. ALS sidesteps this entirely, which is one of its quiet advantages: there is no learning rate to tune because each step is an exact solve.

The bias-variance lens ties these together. Increasing $k$ or decreasing $\lambda$ both increase variance (capacity to fit, including noise) and decrease bias (in the statistical sense of systematic error). The art is to push capacity up while raising regularization in step, so the model has room to represent real taste structure without the slack to memorize individual ratings. The honest way to navigate it is a validation curve, not intuition: plot validation RMSE and validation Recall@10 against $k$ for a few $\lambda$ values and read the knee off the chart.

## 11. Cold start: the one thing factors cannot fix

Matrix factorization has a hard limit that you must design around: a brand-new user or item has no learned vector. The factors $p_u$ and $q_i$ are parameters indexed by id; if user $u$ never appeared in training, there is no row in $P$ for them, and the model literally cannot produce a personalized score. Same for a new item. This is the cold-start problem, and it is not a bug you can tune away; it is intrinsic to the latent-factor approach, which learns one vector per id from that id's interactions.

There are several standard mitigations, and each is a deliberate trade. For a brand-new user with zero interactions, you fall back to the bias model: $\hat r_{ui} = \mu + b_i$, which is just popularity-with-quality ordering, the same list for everyone, until you have observations. Once a new user gives you a few ratings, you can solve for their factor vector on the fly using the ALS user solve from section 4.2 with the existing fixed item matrix, which is one $k \times k$ solve and gives a personalized vector from as few as a handful of ratings. This is a genuinely nice property of ALS: folding in a new user is cheap and closed-form.

For a new item, you can initialize its vector from content features (genre, cast, text embedding) via a small regression that maps content to factor space, so the item starts with a sensible vector before it has interactions. That is the bridge to hybrid and content-based models, and it is why pure ID-based MF is rarely the whole story in production. The more principled solution is to use a model that consumes side features directly, which is exactly what [factorization machines](/blog/machine-learning/recommendation-systems/factorization-machines-and-field-aware-fm) do: they generalize MF to arbitrary feature interactions, so a new item's content features carry signal immediately. When cold start dominates your traffic, that generalization is worth the extra model complexity.

Stress test the cold-start story. What happens at the cold extreme, a launch with no interaction data at all? Then MF has nothing to factor, and you must start from content or editorial logic and accumulate interactions before factors mean anything. What about a long-tail catalog where most items have one or two ratings? Those items get noisy, heavily-regularized vectors close to the origin, so they rank near the popularity baseline; that is the right behavior (do not over-trust two data points) but it means MF alone will under-serve the tail, which is where content features earn their keep. And what about the closed loop, where the model only ever learns vectors for items it already shows? That feedback loop is how a recommender quietly collapses its catalog to a few popular items, and it is the reason the series keeps returning to the serve-log-train cycle as the thing that silently shapes everything.

## 12. Case studies: the Netflix Prize and Funk-SVD

Matrix factorization is not a textbook curiosity; it won the most famous competition in the field's history, and how it won taught the practical lessons we have been applying.

**Funk-SVD (Simon Funk, 2006).** During the Netflix Prize, Brandyn Webb, writing as Simon Funk, published a blog post describing a simple SGD-trained low-rank factorization that trained one factor at a time on the residuals, regularized, over the observed ratings only. It was startlingly effective and startlingly simple, a few dozen lines, and it reframed the whole competition around latent-factor models. The crucial methodological move, the one that separates it from textbook SVD, was training only on observed entries with explicit regularization, which is exactly the objective in section 2. Nearly every recommender library's `SVD` is a descendant of this idea, which is why the name persists even though it is not the Eckart-Young SVD.

**BellKor and "Matrix Factorization Techniques for Recommender Systems" (Koren, Bell, Volinsky, IEEE Computer, 2009).** The team that ultimately won the Netflix Prize, BellKor's Pragmatic Chaos, distilled their approach in a widely cited 2009 paper that is still the best single reference for this material. Its central messages map directly onto this post: bias terms capture most of the explainable variance and must be modeled explicitly; the objective sums over observed entries with $L_2$ regularization; both SGD and ALS are valid optimizers with different strengths; and SVD++ adds implicit feedback (the set of items a user rated) for a measurable gain. Their winning entry was an ensemble of many models, but matrix factorization was the backbone, and the paper's framing of the model as $\mu + b_u + b_i + p_u^\top q_i$ is the one the whole industry adopted. The reported Netflix RMSE improvements were on the order of going from Netflix's own Cinematch baseline of about 0.9514 toward the 0.8567 target that won the \$1,000,000 prize; matrix factorization techniques were responsible for a large share of that gap.

**Weighted ALS for implicit feedback (Hu, Koren, Volinsky, ICDM 2008).** The same year, a companion line of work adapted matrix factorization to implicit data by treating all unobserved entries as zero-confidence negatives and observed interactions as positive with a confidence proportional to the interaction strength. The closed-form ALS solve from section 4.2, with a per-entry confidence weight, is exactly this method, and it is what `implicit`'s `AlternatingLeastSquares` implements. It became the default for clicks-and-plays recommenders and is the natural bridge to the implicit-feedback post in this series. These three papers, taken together, are the canon: read them and you will recognize every line of code above.

#### Worked example: turning a Recall lift into a business case

Suppose you replace a popularity-ordered baseline with a tuned biased MF and your offline Recall@10 goes from 0.061 to 0.159, the bias-only-versus-MF jump from the results table. How do you reason about whether that is worth shipping? The honest chain is: offline metric, to an online proxy, to a business number, with explicit assumptions stated. Say your feed serves 10 million sessions a day, the baseline click-through rate on the top slot is 4.0 percent, and historically a doubling of offline Recall@10 has correlated with roughly a 15 percent relative lift in top-slot CTR on this surface (you would estimate this correlation from past A/B tests, never assume it). A 15 percent relative lift takes CTR from 4.0 to 4.6 percent, which is $10{,}000{,}000 \times (0.046 - 0.040) = 60{,}000$ additional clicks per day. If each incremental click is worth, conservatively, \$0.02 in downstream value, that is \$1,200 per day or about \$438,000 per year. The point of the arithmetic is not the exact figure, which depends entirely on your surface, but the discipline: an offline Recall jump is not a business case until you have stated the offline-to-online transfer assumption and confirmed it in an A/B test. Many a doubled offline metric has produced zero online lift because the offline gain came from items users would have found anyway. State the assumption, then test it.

## 13. When MF is the right call (and when deep models earn their keep)

Matrix factorization is the right default for a startling range of problems, and reaching past it too early is one of the most common and expensive mistakes in applied recsys. Here is the decisive guidance.

**Reach for matrix factorization when** you have a clean user-item interaction signal (ratings, plays, purchases) and you want a strong personalized baseline fast. It is cheap to train, cheap to serve (a dot product, which is a maximum-inner-product search amenable to fast approximate nearest neighbor indices), interpretable enough to debug, and it is the model every later comparison should beat to justify itself. As the retrieval stage of a funnel, MF or its implicit cousin is still a completely respectable production choice at large scale, because the dot-product structure is exactly what ANN retrieval needs. If a two-tower or a deep ranker cannot beat a well-tuned biased MF on your data, that is a signal your problem does not need the complexity, not a signal to add more layers.

**Reach for a deeper model when** you have rich side features that MF cannot consume (user demographics, item content, context like time and device, sequential history), when feature interactions beyond a single dot product carry real signal, or when cold start dominates and you need content generalization. Factorization machines extend MF to arbitrary feature crosses with the same low-rank trick and are the natural next step when features matter. Two-tower neural retrieval generalizes the dot-product structure to deep encoders while keeping the fast ANN serving. Sequence models like SASRec capture order that a static MF ignores. And neural collaborative filtering replaces the dot product with a learned interaction function, though, as [neural collaborative filtering and its critique](/blog/machine-learning/recommendation-systems/neural-collaborative-filtering-and-its-critique) discusses, a careful study by Rendle and colleagues found that a well-tuned dot-product MF often matches or beats NCF on the same benchmarks, which is a humbling and important result: the dot product is hard to beat when you tune the simple model as hard as the complex one.

A concrete decision sequence I use when starting a new recommender from scratch: first ship the bias-only model ($\mu + b_u + b_i$), which is a few lines and gives you a real popularity-with-quality baseline and a sanity check on your data pipeline. Second, add factors (biased MF) and confirm it beats bias-only on Recall@K and NDCG@K, not just RMSE; if it does not, your data is too sparse or your eval is broken, and a deep model will not save you. Third, only once MF is tuned and winning, ask whether side features, sequence, or richer interactions would help, and if so step to factorization machines or a two-tower, measuring each step against the MF baseline. This order front-loads the cheap wins and forces every increment in complexity to earn its place with a measured improvement. Skipping straight to a deep model is how teams end up with a complicated system that, when someone finally builds the MF baseline, turns out to be no better.

There is a quiet operational argument for MF too, separate from accuracy. A two-embedding-table model is trivial to retrain, trivial to roll back, trivial to debug (you can inspect a user's vector and the items nearest it and the result is interpretable), and it has almost no moving parts to break in production. Deep rankers bring feature pipelines, train-serve skew risk, longer training, and a larger surface for the silent bugs that halve precision without throwing an error. For a small team, the reliability of MF is itself a feature. The honest framing is a cost-benefit one. Every step up in model complexity costs training time, serving latency, memory, operational risk, and debugging difficulty. MF buys you most of the personalization gain for the least of all those costs. Spend the complexity budget only where a measured, online win justifies it, and always against a strong MF baseline, not a strawman.

## 14. Key takeaways

- **Latent factors generalize where neighborhoods interpolate.** Representing each user and item as a vector in $\mathbb{R}^k$ and scoring with a dot product lets the model compare users with no co-rated items, which is exactly where sparse-data neighborhood CF fails.
- **Biases carry most of the RMSE; factors carry the ranking.** The $\mu + b_u + b_i$ part of the model closes most of the rating-prediction gap, but the $p_u^\top q_i$ part is what triples Recall@10. Always model biases explicitly, and never judge a recommender by RMSE alone.
- **"SVD" in recommenders is not SVD.** True SVD needs a full matrix; the recommender version (Funk-SVD) is a regularized factorization fit only on observed entries by SGD, with bias terms. Sum over observed cells only; never impute the missing ones for explicit data.
- **SGD for sparse explicit data, ALS for implicit and parallel.** SGD steps are cheap and $O(k)$ but sequential and learning-rate-sensitive; ALS solves each user or item in closed form, is embarrassingly parallel, and has no learning rate, which makes it the default for implicit-feedback weighted factorization on clusters.
- **$k$, $\lambda$, and $\eta$ are a bias-variance dashboard.** Raise capacity ($k$) and regularization ($\lambda$) in step; watch validation RMSE and Recall@10, not training error; remember $k$ is a memory and serving decision, not just an accuracy knob.
- **Cold start is intrinsic, not a tuning bug.** A new id has no vector. Fall back to the bias model, fold in new users with a single ALS solve, initialize new items from content, or move to a feature-aware model like factorization machines when cold start dominates.
- **Beat MF before you go deep.** A well-tuned biased MF is the baseline every fancier model must justify itself against; the dot product is far harder to beat than the deep-learning literature once suggested.

## 15. Further reading

- Koren, Bell, and Volinsky, "Matrix Factorization Techniques for Recommender Systems," IEEE Computer, 2009. The single best reference for the biased MF model, both optimizers, and SVD++.
- Hu, Koren, and Volinsky, "Collaborative Filtering for Implicit Feedback Datasets," ICDM 2008. The weighted-ALS method that the `implicit` library implements.
- Funk (Brandyn Webb), "Netflix Update: Try This at Home," 2006. The original Funk-SVD blog post that reframed the Netflix Prize around latent factors.
- Rendle, Krichene, Zhang, and Anderson, "Neural Collaborative Filtering vs. Matrix Factorization Revisited," RecSys 2020. The study showing a tuned dot-product MF matches or beats NCF.
- Krichene and Rendle, "On Sampled Metrics for Item Recommendation," KDD 2020. Why full ranking metrics, not sampled ones, are the honest way to compare recommenders.
- The `implicit` library documentation (benfred.github.io/implicit) and the `surprise` documentation (surpriselib.com) for the production APIs used above.
- Within this series: [collaborative filtering from first principles](/blog/machine-learning/recommendation-systems/collaborative-filtering-from-first-principles) for the neighborhood methods MF improves on; [implicit feedback models, ALS and BPR](/blog/machine-learning/recommendation-systems/implicit-feedback-models-als-and-bpr) for the implicit-data factorization; [factorization machines and field-aware FM](/blog/machine-learning/recommendation-systems/factorization-machines-and-field-aware-fm) for the feature-aware generalization; and the capstone [recommender systems playbook](/blog/machine-learning/recommendation-systems/the-recommender-systems-playbook) for where MF sits in a production funnel.
