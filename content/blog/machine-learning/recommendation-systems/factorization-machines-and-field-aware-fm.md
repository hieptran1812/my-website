---
title: "Factorization Machines and Field-Aware FM"
date: "2026-06-19"
publishDate: "2026-06-19"
description: "Derive factorization machines and the linear-time interaction trick from scratch, see how FM generalizes matrix factorization and SVD++, then build FM and field-aware FM in PyTorch and measure AUC and logloss on Criteo-style CTR data."
tags:
  [
    "recommendation-systems",
    "recsys",
    "factorization-machines",
    "field-aware-fm",
    "ctr-prediction",
    "feature-interactions",
    "machine-learning",
    "criteo",
    "pytorch",
  ]
category: "machine-learning"
subcategory: "Recommendation Systems"
author: "Hiep Tran"
featured: true
readTime: 53
image: "/imgs/blogs/factorization-machines-and-field-aware-fm-1.png"
---

A few years ago I inherited a click-through-rate model for an ad-ranking system that was a plain logistic regression over a few thousand features: user country, device, hour of day, advertiser, creative size, the page category, and a long tail of categorical IDs. It was fast and stable, and it had a wall it could not climb. Everyone on the team knew the magic was in the crosses: device-times-advertiser, country-times-category, hour-times-creative. The trouble was that the moment you one-hot those crosses, you get a feature space with hundreds of millions of columns, and almost every column is a pair that appeared a handful of times in the training log or never at all. Logistic regression learns one independent weight per column, so a cross that co-occurred three times gets a weight estimated from three examples, and a cross that never co-occurred gets a weight of exactly zero forever. The model could not learn the very interactions everyone agreed were the signal. We hand-picked a few dozen crosses, watched offline AUC creep up by a thousandth, and gave up on the rest.

The fix was not a bigger logistic regression or a feature-selection sweep. It was to stop giving each cross its own weight and instead give each *feature* a short latent vector, then define the strength of a cross as the dot product of the two vectors. Suddenly a cross that never co-occurred in training still has a predicted strength, because both of its features were pulled into position by all the *other* pairs they did appear in. That single change is the factorization machine, and it is the model that turned a flat logistic-regression CTR pipeline into something that could actually learn interactions on sparse data. It is also the second-order backbone that every deep CTR model since, DeepFM and DCN among them, is built on top of.

This is a post in the series **Recommendation Systems: From Click to Production**, and it sits exactly at the hinge between the classic collaborative-filtering models and the deep ranking stack. If you have read [matrix factorization, the workhorse of recommenders](/blog/machine-learning/recommendation-systems/matrix-factorization-the-workhorse), you already know the idea of describing a user and an item by latent vectors and scoring the pair by a dot product. A factorization machine is that same idea taken to its logical conclusion: instead of just two fields (user and item), let there be *any* number of sparse features, and factorize *every* pairwise interaction the same way. The picture below is the one contrast the whole post turns on. Logistic regression with one-hot crosses gives each cross its own weight and starves on sparse co-occurrence; FM factorizes interactions into shared latent vectors that borrow strength across pairs.

![Diagram contrasting logistic regression with one-hot crosses that learns one independent weight per cross with factorization machines that factor interactions into shared latent vectors](/imgs/blogs/factorization-machines-and-field-aware-fm-1.png)

The frame for the whole series, set up in [what is a recommender system](/blog/machine-learning/recommendation-systems/what-is-a-recommender-system), is the retrieval to ranking to re-ranking funnel fed by the serve-log-train feedback loop, read off the offline-versus-online gap. Factorization machines live squarely in the **ranking** stage: they take a candidate item plus a rich bag of user, item, and context features and produce a calibrated click probability. By the end of this post you will be able to derive the FM equation and the linearization trick that makes it run in linear time, explain on principle why factorized interactions beat independent cross weights under sparsity, build both FM and field-aware FM from scratch in PyTorch, train them with logloss on a CTR-style dataset, and read off AUC and logloss to decide between LR, FM, and FFM with eyes open about the parameter cost.

## 1. The problem: linear models cannot learn interactions, and crosses explode

Start with the model FM is built to beat. Logistic regression for CTR takes a feature vector $x \in \mathbb{R}^n$ and predicts a click probability with

$$\hat y(x) = \sigma\!\left(w_0 + \sum_{i=1}^{n} w_i x_i\right), \qquad \sigma(z) = \frac{1}{1 + e^{-z}}.$$

Here $x$ is almost always a giant sparse one-hot encoding. If you have a categorical field like "advertiser" with 50,000 distinct values, it becomes 50,000 binary columns, exactly one of which is 1 for any given example. Stack a dozen such fields and $n$ is in the millions, but only a dozen entries of $x$ are nonzero per row. That extreme sparsity is the defining property of CTR data, and it shapes everything that follows.

Logistic regression is linear in $x$, which has a precise and damaging consequence: it can only express *main effects*. The weight $w_i$ says "when feature $i$ is present, shift the logit by $w_i$, regardless of what else is present." It cannot say "the device matters more for this advertiser than that one," because that statement is about a *pair* of features, and a linear model has no term for pairs. In CTR, the pairs are where most of the money is. Whether someone clicks a travel ad depends enormously on the interaction of their device, the hour, and the destination, not on any one of those in isolation.

### 1.1 The manual-cross trap

The classic workaround is to manufacture the interactions by hand. You define a new categorical feature that is the *cross* of two existing ones: a single feature whose value is the pair `(device=iOS, advertiser=Expedia)`. One-hot that crossed feature and feed it to the same logistic regression, and now the model has a weight for that specific pair. This is exactly what Google's Wide and Deep did in its "wide" branch, and it works, to a point.

The point where it breaks is sparsity, and it breaks hard. Suppose two fields have $a$ and $b$ distinct values. Their full cross has $a \times b$ columns. Cross device (say 5 values) with advertiser (50,000) and you have 250,000 columns; cross advertiser with page category (10,000) and you have 500 million. The number of columns grows as the *product* of cardinalities, so a handful of crosses among high-cardinality fields blows the feature space past anything you can store, let alone train. And here is the part that actually kills the approach: even if you could store all those columns, almost none of them have enough data to learn a weight. The cross `(advertiser=Expedia, page=obscure-blog)` may have appeared zero times in your training log. Logistic regression learns each cross weight from only the rows where that exact cross is present, in complete isolation from every other cross. A cross seen zero times gets weight zero. A cross seen three times gets a noisy weight estimated from three labels. There is no mechanism for the model to say "I have never seen Expedia on this page, but I have seen Expedia on similar pages and this page with similar advertisers, so let me interpolate."

#### Worked example: how fast crosses explode

Take a modest CTR setup with five categorical fields and these cardinalities: user-ID 2,000,000, item-ID 500,000, advertiser 50,000, page-category 10,000, and device 5. The one-hot feature space for the *main effects* is the sum, about 2.56 million columns. Now add all ten pairwise crosses. The user-times-item cross alone is $2{,}000{,}000 \times 500{,}000 = 10^{12}$ columns. The whole set of pairwise crosses is on the order of $10^{12}$ columns. If your training log has 100 million rows, then the user-times-item cross has at most 100 million nonzero columns ever observed, out of a trillion possible, so at least 99.99 percent of those columns are guaranteed to be all-zero in training and useless at serving time. You cannot brute-force interactions on sparse data. You need a model that *shares* statistical strength across the pairs, and that is precisely the gap FM fills.

The deeper diagnosis here connects to a theme we hit repeatedly in this series: sparse categorical data is not a nuisance to be one-hot-encoded away, it is the central modeling challenge. The same sparsity that makes neighborhood collaborative filtering fail (two users who share no co-rated items cannot be compared) makes independent cross weights fail. The cure in both cases is identical in spirit: replace explicit per-pair parameters with *shared latent vectors* so that statistical strength flows between pairs that look alike. We covered exactly which sparse features a recommender has to deal with in [the data and features of recommenders](/blog/machine-learning/recommendation-systems/the-data-and-features-of-recommenders); FM is the model that finally puts those features to work as interactions.

### 1.2 Why not just regularize the polynomial model?

A reasonable objection: if the problem with the explicit second-order model $\sum_{i\lt j} w_{ij} x_i x_j$ is that the cross weights are estimated from too little data, why not just regularize them harder? Add an $L_2$ penalty on $w_{ij}$ and shrink the under-observed weights toward zero. The answer reveals exactly what factorization buys you and regularization cannot. Regularization shrinks each $w_{ij}$ toward zero *independently*. It does not let information flow *between* weights. A pair seen zero times is regularized to zero, which is the same as not having it; a pair seen three times is shrunk toward zero, which throws away the little signal it had. Regularization can only make a noisy estimate smaller, never borrow strength from a related pair.

Factorization does something regularization cannot: it ties the estimates together through shared vectors. When $v_i$ is updated by the pair $(i, l)$, that update changes the model's prediction for *every* pair involving $i$, including the unobserved $(i, j)$. The information from the $(i, l)$ pair flows to the $(i, j)$ pair through the shared vector $v_i$. This is structural, not statistical: it is a property of the model's parameterization, not of any penalty you add. You can regularize the FM latent vectors too (and you should), but the strength-sharing comes from the factorized structure, and regularizing a polynomial model does not give you that structure. This is the same reason a low-rank matrix factorization generalizes where an $L_2$-regularized full-rank reconstruction does not: the rank constraint *shares* parameters across cells, while the penalty merely shrinks them.

There is a second, subtler reason factorization wins, having to do with the geometry of the solution. The explicit polynomial model has a free parameter for every pair, so its interaction matrix $W$ can be anything, including a high-rank matrix that fits noise. FM constrains $W$ to be low-rank (rank at most $k$, since $W = V V^\top$ with $V \in \mathbb{R}^{n \times k}$). That constraint is a strong inductive bias that says "the true interaction structure is low-dimensional," which is empirically true for preference and click data, the same low-rank bet that powers matrix factorization. The bet pays off precisely because real feature interactions are governed by a small number of latent factors (taste dimensions, intent dimensions, context dimensions), not by an independent quirk for every pair of categorical values.

## 2. The FM model: factorize every interaction

A factorization machine of order two predicts

$$\hat y(x) = w_0 + \sum_{i=1}^{n} w_i x_i + \sum_{i=1}^{n}\sum_{j=i+1}^{n} \langle v_i, v_j \rangle\, x_i x_j,$$

where $w_0 \in \mathbb{R}$ is the global bias, each $w_i \in \mathbb{R}$ is a per-feature linear weight (exactly the logistic-regression weights), and each $v_i \in \mathbb{R}^k$ is a $k$-dimensional latent vector attached to feature $i$. The notation $\langle v_i, v_j \rangle = \sum_{f=1}^{k} v_{i,f}\, v_{j,f}$ is the dot product. For CTR you pass the whole expression through a sigmoid; for regression you use it raw. The first two terms are just logistic regression. The third term is the entire idea: it adds, for every pair of features $i$ and $j$, an interaction whose *strength* is not a free parameter but the dot product of the two features' latent vectors, weighted by the product of the feature values $x_i x_j$.

The figure below traces the forward pass. The sparse input splits into a linear path and a factorized pairwise path, the pairwise path runs through the per-feature embeddings, and the bias, linear term, and pairwise term merge into a single logit.

![Diagram of the factorization machine forward pass where a sparse input feeds a linear term and a factorized pairwise term that merge with the global bias into one logit](/imgs/blogs/factorization-machines-and-field-aware-fm-2.png)

### 2.1 Why factorizing the interaction is the whole trick

Compare the FM interaction term to the explicit alternative. A full second-order polynomial model would write the interaction as $\sum_{i\lt j} w_{ij}\, x_i x_j$ with a free scalar weight $w_{ij}$ for every pair. That is the manual-cross model in disguise, and it has the same fatal flaw: $w_{ij}$ is learned only from rows where both $i$ and $j$ are active, in isolation. FM replaces the matrix of pair weights $W \in \mathbb{R}^{n \times n}$ with a low-rank factorization. It posits that $w_{ij} \approx \langle v_i, v_j \rangle$, that is, the entire interaction matrix is the Gram matrix $V V^\top$ of a tall thin matrix $V \in \mathbb{R}^{n \times k}$. Instead of $\binom{n}{2}$ independent weights, you have $nk$ parameters, and crucially, every weight in row $i$ of $W$ shares the same vector $v_i$.

That sharing is the source of all the magic, and it is worth being precise about why it works under sparsity. Suppose the pair $(i, j)$ never co-occurs in training, so a polynomial model can never estimate $w_{ij}$. In FM, $v_i$ is still estimated, from every *other* pair $(i, l)$ that $i$ does appear in; and $v_j$ is estimated from every pair $(j, m)$ that $j$ appears in. The interaction $\langle v_i, v_j \rangle$ is then defined and meaningful even though the specific pair was never seen, because it is built from the two features' positions in a shared latent space. This is exactly the transitive-bridge argument from matrix factorization, generalized to arbitrary features. If feature $i$ behaves like feature $i'$ in their interactions, they get similar vectors, and then $i$ automatically inherits a sensible interaction with every feature $i'$ interacts with. The polynomial model has no such bridge; the factorized model builds it for free.

### 2.1.1 The geometry of the shared latent space

It pays to be precise about what the latent vectors *mean*, because that geometry is the source of both FM's power and its limits. Each feature, whether it is a user ID, an item ID, a device, or an hour bucket, lives as a point in the same $k$-dimensional space. The interaction strength between two features is their dot product, which is large and positive when their vectors point the same way, near zero when they are orthogonal, and negative when they point in opposite directions. So the model learns to arrange *all* features, across all fields, in one shared space such that features that interact positively (this device tends to click this kind of advertiser) end up aligned, and features that interact negatively end up anti-aligned.

This single-space arrangement is exactly what lets strength flow between pairs. If two advertisers end up near each other in the space because they both interact similarly with many devices and contexts, then a device's interaction with one automatically resembles its interaction with the other, even for device-advertiser pairs never seen together. The space is doing collaborative filtering over feature interactions: it places similar features near each other and lets proximity stand in for co-occurrence. This is the precise generalization of the matrix-factorization geometry, where users and items shared a space; FM extends the same shared space to every feature of every field. The limit of this geometry is the same limit MF has: a single dot product is a symmetric, second-order similarity, so it cannot natively express "feature $i$ likes $j$ but $j$ does not like $i$" or higher-order conjunctions, which is exactly the gap FFM (asymmetric per-field vectors) and the deep models (higher orders) later address.

### 2.2 FM generalizes matrix factorization, SVD++, and more

Here is the result that makes FM feel inevitable rather than arbitrary: with the right feature encoding, a factorization machine *is* matrix factorization, and SVD++, and several other named models. They are all special cases of one equation.

Take plain biased matrix factorization. Encode each training example as a sparse vector with exactly two active features: a one-hot for the user and a one-hot for the item. So $x$ has a 1 in the user-$u$ slot and a 1 in the item-$i$ slot, zeros everywhere else. Now write out the FM prediction. The linear term $\sum_l w_l x_l$ picks out exactly $w_u + w_i$, which are the user bias and item bias. The pairwise term $\sum_{l < m} \langle v_l, v_m \rangle x_l x_m$ has only one nonzero summand, the pair (user-$u$, item-$i$), giving $\langle v_u, v_i \rangle$. So the whole prediction collapses to

$$\hat y = w_0 + w_u + w_i + \langle v_u, v_i \rangle,$$

which is exactly the biased matrix-factorization model $\mu + b_u + b_i + p_u^\top q_i$. FM with two one-hot fields *is* biased MF. There is nothing left over.

Now add a third field. SVD++ improves MF by also conditioning on the *set* of items the user has interacted with. In FM you reproduce it by adding, for each example, a third group of active features: a normalized multi-hot over the user's interacted-item history. Concretely, if user $u$ has interacted with the item set $N(u)$, you add features that are $1/\sqrt{|N(u)|}$ in each history-item slot. The pairwise term then automatically produces the interaction between each history item and the target item, which is exactly the implicit-feedback term $\sum_{l \in N(u)} \langle q_i, y_l \rangle / \sqrt{|N(u)|}$ that SVD++ adds by hand. FM recovers it for free, with no special-case code. The same construction recovers timeSVD++ (add a time-bucket field), factorized personalized Markov chains for sequential recommendation (add the previous-item field), and attribute-aware models (add content fields). This is the practical punchline of the original Rendle paper: you do not need to invent and re-implement a new factorization model for every new signal. You add the signal as a field in the feature vector, and FM factorizes its interactions with everything else for free. We will revisit this idea when we get to deep models, because it is the same lesson DeepFM and DCN learn: make the feature interaction a first-class, learnable thing.

### 2.2.1 Numeric features and the value term

One detail in the FM equation that beginners skip over is the $x_i x_j$ factor on the interaction. For one-hot categorical features every active $x_i = 1$, so the factor disappears and the interaction is just $\langle v_i, v_j \rangle$. But the value term is what lets FM handle *real-valued* features cleanly, and that matters because Criteo has 13 numeric fields (counts and ratios). A numeric feature like "number of times this user clicked an ad today" enters the model as a single slot with $x_i$ equal to the (usually log-transformed and normalized) value, not a one-hot. Its latent vector $v_i$ is shared across all examples, and the value $x_i$ scales both its linear contribution $w_i x_i$ and every interaction it participates in, $\langle v_i, v_j \rangle x_i x_j$. So FM treats a numeric feature as "one feature whose interaction strength scales with its magnitude," which is a sensible and learnable behavior. In practice you log-transform skewed counts ($\log(1 + x)$) and standardize, because the squared term in the linearization is sensitive to scale, and an unnormalized count of 100,000 will dominate the pooled sum and destabilize training. This scale sensitivity is a recurring FM gotcha: always normalize numeric features before they enter an FM, more aggressively than you would for a tree model.

## 3. The linearization trick: O(kn), not O(kn^2)

Read the FM equation again and you will worry about cost. The pairwise term is a double sum over all pairs of features, $\sum_{i=1}^{n}\sum_{j=i+1}^{n}$, and each term is a $k$-dimensional dot product. Naively that is $O(k n^2)$ work per prediction. With $n$ in the millions, $n^2$ is a non-starter. If FM really cost $O(kn^2)$ it would be a curiosity, not the workhorse of CTR. The reason it ships is a beautiful algebraic identity that collapses the double sum into a single sum, taking the pairwise term from $O(kn^2)$ to $O(kn)$. Let me derive it, because the derivation is the single most important page in the FM story and it explains exactly why FM is fast on sparse data.

### 3.1 The derivation

We want to evaluate, for a fixed latent dimension $f$, the quantity

$$\sum_{i=1}^{n}\sum_{j=i+1}^{n} v_{i,f}\, v_{j,f}\, x_i x_j.$$

The key observation is that a sum over *unordered* pairs $i < j$ is half of the sum over *all ordered* pairs minus the diagonal. Formally, for any quantities $a_i = v_{i,f} x_i$,

$$\sum_{i}\sum_{j} a_i a_j = \left(\sum_i a_i\right)^2,$$

because the left side is the full outer product summed, which factors into the product of the two marginal sums. That full double sum counts every unordered pair *twice* (once as $(i,j)$ and once as $(j,i)$) and also counts every diagonal term $(i,i)$ once. So

$$\left(\sum_i a_i\right)^2 = \sum_{i}\sum_{j} a_i a_j = 2\sum_{i < j} a_i a_j + \sum_i a_i^2.$$

Solve for the quantity we want, the sum over $i < j$:

$$\sum_{i < j} a_i a_j = \frac{1}{2}\left[\left(\sum_i a_i\right)^2 - \sum_i a_i^2\right].$$

Substitute back $a_i = v_{i,f} x_i$ and sum over all latent dimensions $f$:

$$\sum_{i < j} \langle v_i, v_j \rangle x_i x_j = \frac{1}{2}\sum_{f=1}^{k}\left[\left(\sum_{i=1}^{n} v_{i,f} x_i\right)^2 - \sum_{i=1}^{n} v_{i,f}^2 x_i^2\right].$$

That is the whole trick. Look at the cost of the right-hand side. For each of the $k$ latent dimensions you compute two sums over the $n$ features: one of $v_{i,f} x_i$ and one of $v_{i,f}^2 x_i^2$. Each sum is $O(n)$, so the total is $O(kn)$. The double sum over pairs is gone. And because $x$ is sparse, the sums only run over the nonzero entries of $x$, so the *real* cost is $O(k \bar n)$ where $\bar n$ is the number of active features (a dozen, not a million). Computing the entire pairwise interaction over a million-dimensional feature space costs a dozen multiply-adds per latent dimension. This is why FM is fast on sparse data: it never touches the zeros, and it never enumerates pairs.

### 3.2 Reading the trick intuitively

There is a clean intuition for the identity. Think of $s_f = \sum_i v_{i,f} x_i$ as the $f$-th coordinate of a single pooled embedding obtained by summing all the active features' latent vectors (weighted by their values). Then $s_f^2$ is the squared pooled embedding, which contains *all* pairwise products including the self-products $v_{i,f}^2 x_i^2$. Subtracting off the self-products and halving leaves exactly the cross-products you wanted. In other words, the FM interaction is "the squared sum minus the sum of squares, over two." You pool, you square, you subtract the diagonal. Practitioners who implement FM in a deep-learning framework often write the pairwise term in literally those words, as we will see in the PyTorch section, because the squared-sum-minus-sum-of-squares form is two `sum` calls and an elementwise square.

#### Worked example: the linearization by hand

Let us compute an FM interaction term both ways for a tiny example and confirm they match. Take three active features with values $x = (1, 1, 1)$ (one-hot, all present) and latent dimension $k = 2$. Let the latent vectors be $v_1 = (1, 0)$, $v_2 = (0, 1)$, and $v_3 = (1, 1)$.

The naive way, summing over the three pairs:

$\langle v_1, v_2 \rangle = 1\cdot0 + 0\cdot1 = 0$.
$\langle v_1, v_3 \rangle = 1\cdot1 + 0\cdot1 = 1$.
$\langle v_2, v_3 \rangle = 0\cdot1 + 1\cdot1 = 1$.

Sum over pairs $= 0 + 1 + 1 = 2$.

Now the linearized way, dimension by dimension. For $f = 1$ the values $v_{i,1} x_i$ are $(1, 0, 1)$, so the sum is $2$, its square is $4$, and the sum of squares is $1 + 0 + 1 = 2$; the contribution is $\frac{1}{2}(4 - 2) = 1$. For $f = 2$ the values $v_{i,2} x_i$ are $(0, 1, 1)$, so the sum is $2$, its square is $4$, and the sum of squares is $0 + 1 + 1 = 2$; the contribution is $\frac{1}{2}(4 - 2) = 1$. Total $= 1 + 1 = 2$. The two agree, and the linearized version only ever summed over the active features, never enumerating pairs. On a million-feature row with twelve actives, the naive form would do 66 dot products and the linearized form does 24 scalar sums. That is the difference between a model you can serve and one you cannot.

#### Worked example: a full FM prediction end to end

Let us run one complete prediction so the three terms become concrete numbers. Take a tiny CTR model with three active features on this example: user=Ann (index 3), item=Lamp (index 41), context=Evening (index 88). All three are one-hot, so $x = 1$ for each. Suppose the trained parameters are: global bias $w_0 = -1.20$ (a base logit corresponding to about a 23 percent click rate, high for illustration); linear weights $w_3 = 0.10$, $w_{41} = 0.20$, $w_{88} = 0.10$; and latent dimension $k = 2$ with vectors $v_3 = (0.3, 0.1)$, $v_{41} = (0.2, 0.4)$, $v_{88} = (0.1, 0.5)$.

The linear term is $w_3 + w_{41} + w_{88} = 0.10 + 0.20 + 0.10 = 0.40$.

The pairwise term, via the linearization. For $f = 1$, the pooled sum is $0.3 + 0.2 + 0.1 = 0.6$, its square is $0.36$, and the sum of squares is $0.09 + 0.04 + 0.01 = 0.14$; contribution $\frac{1}{2}(0.36 - 0.14) = 0.11$. For $f = 2$, the pooled sum is $0.1 + 0.4 + 0.5 = 1.0$, its square is $1.00$, and the sum of squares is $0.01 + 0.16 + 0.25 = 0.42$; contribution $\frac{1}{2}(1.00 - 0.42) = 0.29$. Pairwise total $= 0.11 + 0.29 = 0.40$.

The logit is $w_0 + \text{linear} + \text{pairwise} = -1.20 + 0.40 + 0.40 = -0.40$, and the predicted click probability is $\sigma(-0.40) = 0.40$. Two things are worth noticing. First, the interaction term ($0.40$) is exactly as large as the linear term here, which is typical of CTR data and is the whole reason FM beats LR: the linear-only model would have predicted $\sigma(-1.20 + 0.40) = \sigma(-0.80) = 0.31$, and the interactions pushed it up to $0.40$. Second, every number traces to the pooled-sum-squared form, never to an enumeration of pairs, which is what the model actually computes at serving time. This is the decomposition the stack figure visualizes.

## 4. The gradient: how FM trains

To train FM by gradient descent you need the partial derivative of the prediction with respect to each parameter. The bias and linear terms are trivial: $\partial \hat y / \partial w_0 = 1$ and $\partial \hat y / \partial w_i = x_i$. The interesting one is the latent vector. Differentiate the linearized pairwise term with respect to $v_{i,f}$. Only the dimension-$f$ summand depends on $v_{i,f}$, and inside it,

$$\frac{\partial}{\partial v_{i,f}}\, \frac{1}{2}\left[\left(\sum_l v_{l,f} x_l\right)^2 - \sum_l v_{l,f}^2 x_l^2\right] = x_i \sum_{l=1}^{n} v_{l,f} x_l - v_{i,f} x_i^2.$$

Define the pooled sum $s_f = \sum_l v_{l,f} x_l$ once per example (you already computed it for the forward pass). Then the gradient is

$$\frac{\partial \hat y}{\partial v_{i,f}} = x_i\, s_f - v_{i,f}\, x_i^2 = x_i\left(s_f - v_{i,f} x_i\right).$$

This is the punchline that makes FM efficient to train as well as to serve: $s_f$ does not depend on $i$, so you compute it once and reuse it for every active feature's gradient. Training a single example is $O(kn)$, the same as the forward pass, and again only over active features. For the CTR setting, the final gradient on the parameters is the chain rule through the logloss. If $p = \sigma(\hat y)$ is the predicted probability and $y \in \{0, 1\}$ is the label, the logloss is $\ell = -\,[y \log p + (1-y)\log(1-p)]$, and its derivative with respect to the raw score is the clean $\partial \ell / \partial \hat y = p - y$. So the per-parameter gradient is just $(p - y)$ times the partials above. Multiply by the learning rate, add an $L_2$ penalty, and you have stochastic gradient descent for FM.

This whole structure (pool, square, subtract diagonal, backprop the pooled sum) is exactly what makes FM trivial to implement in a deep-learning framework, where the autograd engine computes those derivatives for you. You only have to write the forward pass in the linearized form, and the gradients above are what the framework will reproduce automatically. We turn to that next, but first the decomposition figure, because it makes the three additive pieces of a single FM logit concrete.

![Diagram decomposing a factorization machine logit into stacked layers of global bias then linear term then factorized pairwise term then the sigmoid probability](/imgs/blogs/factorization-machines-and-field-aware-fm-4.png)

The stack makes the engineering reality vivid. The global bias sets the base click rate (CTR is usually a few percent, so $w_0$ is a negative logit). The linear term adds the main effects, the things logistic regression already captured. The pairwise term is the *only* layer that encodes interactions, and on real CTR data it is frequently the layer that moves the model from "barely better than predicting the base rate" to "worth shipping." When you debug an FM that is not learning, this decomposition is your friend: log the magnitude of each term and you can see immediately whether the interaction term is contributing anything at all.

### 4.1 Regularization and the three trainers

The objective FM actually minimizes is the data loss plus an $L_2$ penalty on every parameter group:

$$\mathcal{L} = \sum_{(x, y)} \ell\big(\hat y(x), y\big) + \lambda_0 w_0^2 + \lambda_w \sum_i w_i^2 + \lambda_v \sum_i \lVert v_i \rVert^2,$$

where $\ell$ is logloss for CTR or squared error for rating prediction. The penalty on the latent vectors $\lambda_v$ is the single most important hyperparameter in an FM, because the latent vectors are where overfitting lives. A pair of features that co-occurs once can drive their interaction $\langle v_i, v_j \rangle$ to perfectly fit that one label unless the penalty holds the vectors back. Too little $\lambda_v$ and the model memorizes rare crosses; too much and it cannot learn real interactions. The gradient of the penalty is simply $2\lambda_v v_i$ added to each latent gradient, which is what weight decay in Adam does for you.

There are three standard ways to fit an FM, and Rendle's `libFM` implements all three. **SGD** is what we derived: cheap per step, needs you to tune the learning rate and the three $\lambda$ values, and is the default for very large data. **ALS (alternating least squares)** fixes all parameters but one coordinate and solves it in closed form, cycling through coordinates; it is the same idea as ALS for matrix factorization, has no learning rate, and converges in few epochs, but each step is more expensive. **MCMC (Markov chain Monte Carlo)** treats the parameters and the regularization strengths as random variables with priors and samples from the posterior with Gibbs sampling. Its killer feature is that it *integrates out* the regularization: you do not tune $\lambda_v$ at all, the sampler infers it. This is why `libFM` with MCMC famously won portions of several KDD Cups with almost no hyperparameter search, and it is the trainer to reach for when you cannot afford a regularization sweep. The trade-off is that MCMC does not produce a single point estimate you can serve directly; you either average the samples or run the chain to a maximum-a-posteriori point.

### 4.2 A stress test: what breaks an FM

Pose the engineering problem honestly. You have built FM, the offline AUC is good, and now you stress-test it before shipping. What breaks?

First stress: *false negatives in implicit data.* If you train FM on clicks where "not clicked" is the negative, many of those negatives are items the user simply never saw, not items they disliked. The interaction term will happily learn to suppress popular-but-unseen items, which is wrong. The fix is the same as everywhere in implicit recommendation, careful negative sampling, which we cover in the implicit-feedback posts; the FM-specific wrinkle is that a corrupted negative poisons not just one weight but every interaction the negative's features participate in.

Second stress: *a high-cardinality feature with a long cold tail.* A brand-new ad ID appears at serving time with a randomly initialized (or hash-collided) latent vector. Its main effect $w_i$ is near zero and its interactions are noise. FM degrades gracefully here, because the global bias and the other active features still drive a reasonable prediction, but you should expect cold items to be mispriced until they accumulate a few hundred impressions. This is a genuine cold-start failure mode, and it is why production CTR stacks pair FM with content features (which generalize to cold items) rather than relying on ID embeddings alone.

Third stress: *scale at 100M features.* The embedding table dominates memory. At $k = 16$ and 100M features that is 6.4 GB of latent parameters, which may not fit on one serving host. The mitigations are feature hashing (cap the table size and accept collisions), a smaller $k$, and quantizing the embeddings to int8 at serve time. FFM makes this far worse, which is the central reason FFM is rare in production despite its Kaggle dominance: the table simply does not fit.

Fourth stress: *the offline-online gap.* The classic trap is that offline AUC rises but online CTR is flat. The usual culprit for an FM is feature skew (a feature computed differently offline and online) corrupting the interactions, or a positional-bias artifact the offline metric rewards but the live system does not benefit from. The discipline here is the same as for any ranker: validate the feature parity between training and serving, and trust the A/B test over the offline metric. The interaction term raises the stakes, because a single skewed feature corrupts a whole row of the interaction matrix, not just its own main effect.

## 5. Building FM from scratch in PyTorch

The cleanest way to internalize the linearization is to write it. Here is a complete FM for CTR in PyTorch. The model holds two embedding tables: a width-1 table for the linear weights $w_i$ and a width-$k$ table for the latent vectors $v_i$. We use `nn.Embedding` with the *feature index* as input, which is how you handle sparse one-hot data without ever materializing the one-hot vector. Each example is a list of active feature indices (and, for non-binary features, their values).

```python
import torch
import torch.nn as nn

class FactorizationMachine(nn.Module):
    def __init__(self, num_features: int, k: int = 16):
        super().__init__()
        self.w0 = nn.Parameter(torch.zeros(1))
        self.linear = nn.Embedding(num_features, 1)        # w_i, one per feature
        self.v = nn.Embedding(num_features, k)             # v_i, latent vectors
        nn.init.zeros_(self.linear.weight)
        nn.init.normal_(self.v.weight, std=0.01)           # small init is important

    def forward(self, feat_idx, feat_val):
        # feat_idx, feat_val: (batch, num_active) padded with index 0, value 0
        # linear term: sum_i w_i x_i
        lin = (self.linear(feat_idx).squeeze(-1) * feat_val).sum(dim=1)   # (batch,)

        # pairwise term via the linearization:
        #   0.5 * sum_f [ (sum_i v_if x_i)^2 - sum_i (v_if x_i)^2 ]
        vx = self.v(feat_idx) * feat_val.unsqueeze(-1)      # (batch, num_active, k)
        sum_sq = vx.sum(dim=1).pow(2)                       # (sum_i v x)^2  -> (batch, k)
        sq_sum = vx.pow(2).sum(dim=1)                       # sum_i (v x)^2  -> (batch, k)
        pair = 0.5 * (sum_sq - sq_sum).sum(dim=1)           # (batch,)

        return self.w0 + lin + pair                        # raw logit
```

That is the entire model. Notice the pairwise term is the squared-sum-minus-sum-of-squares identity, written in two lines: `vx.sum(dim=1).pow(2)` is the squared pooled embedding, `vx.pow(2).sum(dim=1)` is the sum of squared embeddings, and their half-difference summed over $k$ is the interaction. Because we index with `feat_idx`, the model only ever touches the active features; a million-column one-hot space costs nothing extra. The padding convention (pad with feature index 0 and value 0) makes batches rectangular; the zero values zero out the padded contributions in both the linear and the pairwise sums, so padding is harmless.

### 5.1 Preparing sparse CTR data

The model consumes a list of active feature indices and values per example, so the data layer's job is to turn raw categorical rows into that sparse representation. The standard approach maps each (field, value) string to an integer index via a feature-hashing trick (so unbounded cardinality maps into a fixed table) and emits padded index and value tensors. Here is a compact `Dataset` that reads rows of categorical strings and numeric values.

```python
import torch
from torch.utils.data import Dataset

def hash_feature(field: str, value: str, n_buckets: int) -> int:
    # deterministic hashing keeps the table size fixed for unbounded cardinality
    return (hash(f"{field}={value}") % n_buckets)

class CTRDataset(Dataset):
    def __init__(self, rows, cat_fields, num_fields, n_buckets, max_active):
        self.rows = rows                 # list of dicts: {field: value, ..., "label": 0/1}
        self.cat_fields = cat_fields     # categorical field names
        self.num_fields = num_fields     # numeric field names
        self.n_buckets = n_buckets
        self.max_active = max_active      # pad/truncate to this many active features

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        idx, val = [], []
        for f in self.cat_fields:                       # one-hot categoricals: value = 1.0
            idx.append(hash_feature(f, str(row[f]), self.n_buckets))
            val.append(1.0)
        for f in self.num_fields:                       # numeric: value = log1p-normalized
            idx.append(hash_feature(f, "_numeric_", self.n_buckets))
            val.append(float(row[f]))                    # assume already log1p + standardized
        # pad to max_active with index 0, value 0 (harmless: zeroed in both sums)
        while len(idx) < self.max_active:
            idx.append(0); val.append(0.0)
        return (torch.tensor(idx[:self.max_active]),
                torch.tensor(val[:self.max_active], dtype=torch.float32),
                torch.tensor(row["label"], dtype=torch.float32))
```

Two production notes hide in this small class. First, the numeric fields share *one* index per field (not one per value), so a numeric feature is a single slot whose value carries the magnitude, exactly the value-term behavior from earlier; you must log-transform and standardize those values before they reach here, or the squared term in the linearization blows up. Second, hashing into a fixed `n_buckets` is what makes the model robust to new categorical values appearing at serving time; the cost is occasional collisions where two distinct features share an embedding, which is empirically harmless at a few-percent collision rate and is the standard CTR engineering trade-off.

### 5.2 The training loop with logloss

The training loop is ordinary. Use binary cross-entropy on the logit (`BCEWithLogitsLoss` is the numerically stable choice, fusing the sigmoid and the loss), add weight decay for the $L_2$ regularization that keeps the latent vectors from overfitting the sparse pairs, and use Adam.

```python
from torch.utils.data import DataLoader

model = FactorizationMachine(num_features=N_FEATURES, k=16)
opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
loss_fn = nn.BCEWithLogitsLoss()

for epoch in range(10):
    model.train()
    for feat_idx, feat_val, y in train_loader:        # y is 0/1 click label
        opt.zero_grad()
        logit = model(feat_idx, feat_val)
        loss = loss_fn(logit, y.float())
        loss.backward()
        opt.step()
```

The only FM-specific subtleties are in the data, not the loop. First, the embedding initialization: start the latent vectors small (standard deviation around 0.01), because a large initialization makes the squared term dominate early and the model diverges. Second, the regularization: the latent vectors are where overfitting hides, because a pair seen once can drive $\langle v_i, v_j \rangle$ to fit that single label; weight decay on `self.v` is what prevents it. Third, the feature hashing: real CTR data has unbounded cardinality (new ad IDs every day), so production FM almost always hashes feature strings into a fixed table of $N$ buckets rather than maintaining an exact index, accepting a small collision rate in exchange for a fixed memory footprint. The sparse one-hot input the model consumes is exactly the kind of feature vector shown below, one active slot per field with everything else zero.

![Diagram of a sparse one-hot input vector with one active slot per field for user and item and context feeding the model](/imgs/blogs/factorization-machines-and-field-aware-fm-6.png)

### 5.3 Computing AUC and logloss honestly

The two metrics that matter for CTR are area under the ROC curve (AUC), which measures ranking quality (the probability a random positive is scored above a random negative), and logloss, which measures calibration (whether the predicted probabilities are right in an absolute sense, not just ordered correctly). You want both, because a model can rank well but be miscalibrated, and a calibrated CTR is what an ad auction actually bids on. Compute them on a *temporal* holdout, training on earlier days and evaluating on a later day, never on a random split, because a random split leaks future information and inflates the numbers.

```python
import numpy as np
from sklearn.metrics import roc_auc_score, log_loss

model.eval()
probs, labels = [], []
with torch.no_grad():
    for feat_idx, feat_val, y in test_loader:
        p = torch.sigmoid(model(feat_idx, feat_val))
        probs.append(p.numpy())
        labels.append(y.numpy())
probs = np.concatenate(probs)
labels = np.concatenate(labels)

print(f"AUC     = {roc_auc_score(labels, probs):.4f}")
print(f"Logloss = {log_loss(labels, probs):.4f}")
```

A subtle but important note: for CTR, an AUC improvement of 0.001 is genuinely meaningful, because the base rates are low and the volume is enormous. A 0.001 AUC lift on a billion daily impressions can be a seven-figure revenue change. This is why the CTR literature reports four decimal places and treats the third decimal as significant, which would be absurd in most ML settings but is correct here.

## 6. Using a real FM library: xlearn and libFM

You will not always hand-roll FM in PyTorch. Two battle-tested implementations exist: `libFM` (Rendle's original C++ tool, which supports SGD, ALS, and MCMC training) and `xlearn` (a fast C++ implementation with a Python API that does both FM and FFM and was popular in Kaggle CTR competitions). They consume data in the LIBSVM or libffm sparse format, which is exactly the active-index-plus-value representation we used above.

```python
import xlearn as xl

# FM model
fm_model = xl.create_fm()
fm_model.setTrain("./train.libsvm")     # label idx:val idx:val ...
fm_model.setValidate("./valid.libsvm")

param = {
    "task": "binary",       # classification, logloss
    "lr": 0.2,
    "lambda": 0.0002,       # L2 regularization
    "k": 16,                # latent dimension
    "metric": "auc",
    "epoch": 30,
    "opt": "adagrad",
}
fm_model.fit(param, "./fm_model.out")

fm_model.setSigmoid()       # output probabilities, not raw scores
fm_model.predict("./fm_model.out", "./fm_pred.txt")
```

The libffm input format is worth understanding because it is where the field idea becomes literal. A plain LIBSVM line for FM is `label index:value index:value ...`. The libffm format adds the field in front of each feature: `label field:index:value field:index:value ...`. That extra `field:` is the only data difference between FM and FFM, and it is the bridge to the next section. Switching the model from FM to FFM in xlearn is one call: `xl.create_ffm()` instead of `xl.create_fm()`, plus the field-annotated data. The headline reason these libraries existed is speed: `libFM` with MCMC famously won several KDD Cup tasks because the Bayesian treatment auto-tunes the regularization, and `xlearn`'s FFM was the workhorse behind many Criteo and Avazu Kaggle leaderboards before deep models took over.

## 7. Field-aware FM: a vector per field

Now the refinement that won Kaggle CTR competitions. Field-aware factorization machines (FFM), from Juan et al. in 2016, observed that plain FM shares a single vector $v_i$ across *all* of feature $i$'s interactions, and argued that this is too coarse. The way feature "device=iOS" interacts with an *advertiser* feature is qualitatively different from the way it interacts with a *time-of-day* feature, yet FM forces both interactions to be read off the same vector $v_{\text{iOS}}$. FFM gives each feature a *separate* latent vector for each *field* it might interact with.

Formally, group the $n$ features into $F$ fields (user, item, advertiser, device, hour, and so on). FFM attaches to feature $i$ not one vector but $F$ vectors, one per field. Write $v_{i, f_j}$ for the vector feature $i$ uses when it interacts with a feature $j$ that belongs to field $f_j$. The FFM prediction is

$$\hat y(x) = w_0 + \sum_{i=1}^{n} w_i x_i + \sum_{i=1}^{n}\sum_{j=i+1}^{n} \langle v_{i, f_j},\, v_{j, f_i} \rangle\, x_i x_j.$$

Read the subscripts carefully, because they are the entire idea. When feature $i$ (say, device=iOS, in the device field) interacts with feature $j$ (say, advertiser=Expedia, in the advertiser field), FFM uses $v_{i, f_j}$, the vector iOS reserves for talking to the *advertiser* field, dotted with $v_{j, f_i}$, the vector Expedia reserves for talking to the *device* field. iOS has a different vector for talking to the time field, and yet another for the page field. In plain FM there was one iOS vector for everything. The figure below contrasts the two parameterizations directly.

![Diagram contrasting plain FM with one latent vector per feature against field-aware FM with one latent vector per feature per field](/imgs/blogs/factorization-machines-and-field-aware-fm-5.png)

### 7.1 The cost of field awareness

FFM's extra expressiveness is not free, and the cost is exactly $F$, the number of fields. FM has $nk$ latent parameters. FFM has $n F k$ latent parameters, one length-$k$ vector per feature *per field*. With $F$ fields that is an $F$-fold blowup in the embedding table. Worse, the linearization trick *does not survive* the field-aware change. The reason is structural: in plain FM, the inner vector $v_i$ is the same regardless of $j$, which is exactly what let us pull the pooled sum $\sum_i v_{i,f} x_i$ out and reuse it. In FFM the vector used depends on the *other* feature's field, so $v_{i, f_j}$ changes as $j$ ranges over fields, and you can no longer factor the double sum. FFM evaluation is back to $O(k n^2)$ over active features, or more precisely $O(k \bar n^2)$ where $\bar n$ is the active-feature count. Since $\bar n$ is small (a dozen or two), the quadratic is tolerable, but FFM is meaningfully slower than FM at both train and serve time, on top of being $F$ times larger in memory.

### 7.2 FFM from scratch in PyTorch

The implementation difference from FM is exactly the loss of the linearization. Because each feature has $F$ vectors and the vector used depends on the other feature's field, you must enumerate the active-feature pairs explicitly. With only a dozen or two active features per row this is fine; the quadratic is over the active count, not the full feature space. The model needs to know each active feature's field, so the data layer now emits field indices alongside feature indices.

```python
import torch
import torch.nn as nn

class FieldAwareFM(nn.Module):
    def __init__(self, num_features: int, num_fields: int, k: int = 4):
        super().__init__()
        self.w0 = nn.Parameter(torch.zeros(1))
        self.linear = nn.Embedding(num_features, 1)
        # one length-k vector per feature PER field: shape (num_features, num_fields, k)
        self.v = nn.Embedding(num_features, num_fields * k)
        self.num_fields, self.k = num_fields, k
        nn.init.zeros_(self.linear.weight)
        nn.init.normal_(self.v.weight, std=0.01)

    def forward(self, feat_idx, feat_field, feat_val):
        # feat_idx, feat_field, feat_val: (batch, m) for m active features
        B, m = feat_idx.shape
        lin = (self.linear(feat_idx).squeeze(-1) * feat_val).sum(dim=1)
        V = self.v(feat_idx).view(B, m, self.num_fields, self.k)   # per-field vectors
        pair = torch.zeros(B, device=feat_idx.device)
        for a in range(m):                       # explicit pair loop over active feats
            for b in range(a + 1, m):
                # feature a uses its vector for field of b, and vice versa
                va = V[torch.arange(B), a, feat_field[:, b]]        # (B, k)
                vb = V[torch.arange(B), b, feat_field[:, a]]        # (B, k)
                dot = (va * vb).sum(dim=1)
                pair = pair + dot * feat_val[:, a] * feat_val[:, b]
        return self.w0 + lin + pair
```

The double loop is the structural cost of field awareness: there is no squared-sum-minus-sum-of-squares shortcut, because `feat_field[:, b]` changes which slice of $V$ you read for feature $a$. In a real high-throughput serving system you would vectorize this with gather operations rather than a Python loop, but the loop makes the algorithm transparent: for every pair, look up feature $a$'s vector reserved for $b$'s field and feature $b$'s vector reserved for $a$'s field, dot them, and scale by the values. Note the latent dimension is $k = 4$, much smaller than FM's $k = 16$, which is the standard FFM practice of trading a large $k$ for the per-field structure to keep the table from exploding.

#### Worked example: FM versus FFM parameter count

Take a CTR dataset with $F = 39$ fields (the Criteo dataset has exactly 39: 13 numeric and 26 categorical) and, after one-hot encoding and hashing, $n = 1{,}000{,}000$ features, with latent dimension $k = 16$. Plain FM has $n k = 1{,}000{,}000 \times 16 = 16$ million latent parameters, plus a million linear weights, so about 17 million total. FFM has $n F k = 1{,}000{,}000 \times 39 \times 16 = 624$ million latent parameters, plus a million linear weights, so about 625 million total. FFM is roughly 37 times larger. At four bytes per float that is 68 MB for FM versus 2.5 GB for FFM. The Kaggle winners typically used a *smaller* $k$ for FFM (often $k = 4$) precisely to keep the table manageable, which is itself a clue: FFM gets its power from the per-field structure, not from a large latent dimension, so you can shrink $k$ and let the extra field vectors do the work. That is a genuine and often-overlooked engineering knob.

The trade-off comparison across the three models is worth pinning down before we look at numbers, because it drives the "which one do I reach for" decision.

![Diagram comparing logistic regression and FM and FFM across learned interactions and parameter count and behavior on sparse data and compute cost](/imgs/blogs/factorization-machines-and-field-aware-fm-3.png)

## 8. Results: LR vs FM vs FFM on CTR data

Now the measurements. The two canonical CTR benchmarks are **Criteo** (a week of display-ad logs, 45 million examples, 13 numeric and 26 categorical fields, the dataset from the 2014 Criteo Kaggle competition) and **Avazu** (40 million mobile-ad click examples, 23 fields, the 2015 Avazu Kaggle competition). These are the datasets every CTR paper reports on, so they make a fair common yardstick. The numbers below are representative figures from the FFM paper and subsequent benchmarks; treat them as approximate and directionally reliable, not as a guarantee of what you will get on your own data, because preprocessing, hashing, and hyperparameters all move the third decimal.

| Model | Criteo AUC | Criteo logloss | Avazu AUC | Avazu logloss | Latent params |
| --- | --- | --- | --- | --- | --- |
| Logistic regression | 0.7916 | 0.4615 | 0.7560 | 0.3895 | 0 (linear only) |
| FM ($k=16$) | 0.7990 | 0.4565 | 0.7625 | 0.3835 | $\sim$16M |
| FFM ($k=4$) | 0.8035 | 0.4540 | 0.7665 | 0.3805 | $\sim$160M |

Read the table the way a practitioner does. Going from LR to FM buys roughly a 0.007 AUC lift on Criteo and a corresponding logloss drop, which on real ad traffic is a large, ship-it improvement, and it costs you one embedding table. Going from FM to FFM buys another 0.004 or so, which is real and was decisive in a Kaggle competition where the margin between first and tenth was a few thousandths, but it costs you roughly an order of magnitude more parameters and a quadratic interaction cost. That is the central trade-off: FM is the best *return on complexity* in this family, and FFM is what you reach for when the last thousandth of AUC is worth a much heavier model. The results matrix below summarizes the same picture.

![Diagram comparing logistic regression and FM and FFM on AUC and logloss and parameter count as a results matrix](/imgs/blogs/factorization-machines-and-field-aware-fm-7.png)

### 8.1 How to measure this honestly

A few guardrails, because CTR benchmarks are easy to fool yourself with. First, always use a temporal split: train on the earlier days, test on the last day. A random split lets the model see the same ad campaign on both sides of the split and inflates AUC by a thousandth or more, which is exactly the magnitude you are trying to measure. Second, report logloss alongside AUC, because a model can win AUC while being miscalibrated, and CTR is consumed as an absolute probability in the auction. Third, watch for the feature-skew trap that has burned every CTR team: if a feature is computed one way in your offline training pipeline (say, hour-of-day in UTC) and a different way at serving time (hour-of-day in the user's local zone), your offline AUC is measuring a model that does not exist in production. We treat this offline-online divergence as a first-class problem in [offline versus online, the two worlds of recsys](/blog/machine-learning/recommendation-systems/offline-vs-online-the-two-worlds-of-recsys); for FM specifically, the interaction term makes skew *worse*, because a skewed feature corrupts not only its own main effect but every interaction it participates in.

#### Worked example: is the FFM upgrade worth it?

Suppose you serve 500 million ad impressions a day at an average CTR of 4 percent and an average revenue of \$0.50 per click. That is 20 million clicks and \$10M of daily click revenue. Your FM model has AUC 0.799. You measure that FFM lifts AUC to 0.8035 offline, and a careful A/B test (the only number that actually counts) shows a 1.2 percent relative CTR lift online, from 4.00 to 4.05 percent. That is 240,000 extra clicks a day, about \$120,000 of incremental daily revenue, or roughly \$44M a year. Against that, FFM costs you a 2.5 GB embedding table instead of 68 MB, a quadratic interaction cost that roughly triples serving latency for the ranker (say from p99 12 ms to p99 35 ms), and a heavier training job. For a large ad platform that math is a clear yes. For a small product where the table will not fit on the serving host and 35 ms blows your latency budget, the same math is a clear no, and FM is the right call. The decision is never "which has higher AUC"; it is "which Pareto point fits my constraints."

### 8.2 Serving cost: the number that decides FFM

The AUC table tells you what FFM can win; the serving cost table tells you whether you can afford it. Three numbers govern an FM-family ranker's serving budget: the embedding-table memory, the per-candidate scoring latency, and the throughput (candidates scored per second per host), since the ranker must score every candidate the retrieval stage hands it, often a few hundred to a few thousand per request.

For FM, scoring one candidate is the linearized $O(k \bar n)$ with $\bar n \approx 15$ active features and $k = 16$, which is a few hundred multiply-adds, single-digit microseconds on a CPU. Scoring 500 candidates is a few milliseconds, comfortably inside a 50 ms ranking budget, and the 68 MB table sits in cache. For FFM, scoring one candidate is $O(k \bar n^2)$ with $k = 4$ and $\bar n^2 \approx 225$, plus the gather over a 2.5 GB table that does not fit in CPU cache, so each score is dominated by memory-bandwidth-bound random reads. In practice FFM's per-candidate latency is several times FM's, and the large table forces either a beefier host or sharding the embedding across machines, which adds a network hop per lookup. This is the concrete reason FFM, despite winning Kaggle, is uncommon in high-QPS production rankers: the offline AUC win does not survive the latency and memory budget, and a DeepFM that shares one compact embedding usually delivers more AUC per millisecond.

#### Worked example: latency budget for a ranker

Suppose your ranking stage has a 40 ms p99 budget and must score 800 candidates per request. With FM at roughly 5 microseconds per candidate, scoring is 800 times 5 = 4 ms, leaving 36 ms for feature fetch and the rest of the pipeline; comfortable. With FFM at roughly 20 microseconds per candidate (the quadratic plus the out-of-cache table reads), scoring is 800 times 20 = 16 ms, four times more, eating a big chunk of the budget and pushing p99 toward the edge under load. If the FFM also needs its 2.5 GB table sharded across two hosts, you add a network round trip and the latency story gets worse. The decision falls out of the arithmetic: at 800 candidates and a 40 ms budget, FM fits easily and FFM is tight; if the candidate count rises to 3,000, FM is still fine at 15 ms while FFM at 60 ms blows the budget outright. The right model is the one whose Pareto point fits the budget, and that is almost always FM unless the AUC delta is worth a serving redesign.

## 9. The bridge to deep: FM is the second-order backbone

Factorization machines are a second-order model: they capture pairwise interactions and nothing higher. Real CTR signal often involves third-order and higher interactions (device-times-advertiser-times-hour), and FM cannot express them directly, because extending the FM equation to order three reintroduces the combinatorial blowup the linearization saved you from. This is the gap that deep CTR models fill, and almost every one of them keeps FM as a component.

The two most important descendants make the lineage explicit. **DeepFM** (Guo et al., 2017) runs an FM and a deep multi-layer perceptron *in parallel over the same embedding table* and sums their outputs. The FM branch handles the low-order (first and second) interactions exactly as we derived; the MLP branch learns higher-order interactions implicitly through its hidden layers; and crucially they *share* the feature embeddings, so the model gets both kinds of interaction with one set of latent vectors and no manual feature engineering. **DCN** (Wang et al., 2017, Deep and Cross Network) replaces the FM branch with a "cross network" that explicitly computes higher-order feature crosses layer by layer, each layer adding one more order of interaction, again over a shared embedding. Both are direct generalizations of the move FM made: factorize the interaction into shared latent vectors, then add capacity for higher orders. We trace the full architecture of these models in [DeepFM and automatic feature interactions](/blog/machine-learning/recommendation-systems/deepfm-and-automatic-feature-interactions); here the point is just the lineage, which the figure below lays out.

![Diagram of the factorization lineage from matrix factorization to factorization machines to field-aware FM and the deep DeepFM and DCN models](/imgs/blogs/factorization-machines-and-field-aware-fm-8.png)

The single most important thing to take from the lineage is conceptual, not architectural. The history of CTR modeling is the history of how feature interactions get represented. Logistic regression represents them as manual crosses (explicit, sparse, doomed). FM represents them as factorized dot products (shared, dense, learnable). FFM refines the factorization to be field-aware. The deep models add higher orders on top of the same factorized embedding. Every step is the same idea pushed further: make the interaction a first-class, learnable, *shared* object rather than a per-pair free parameter. Once you see FM as the moment that idea crystallized, the rest of the deep CTR zoo reads as variations on a theme, and the [ranking model CTR prediction foundations](/blog/machine-learning/recommendation-systems/the-ranking-model-ctr-prediction-foundations) post becomes a much easier read.

### 9.1 FM versus LR versus deep, side by side

It helps to lay the four model families against the dimensions that actually drive a production decision: which interaction orders they capture, how they get those interactions, their parameter and serving cost, and the regime where each is the right call.

| Model | Interaction order | How interactions are learned | Params | Serving cost | Reach for it when |
| --- | --- | --- | --- | --- | --- |
| Logistic regression | First only (or manual crosses) | Hand-engineered one-hot crosses | $n$ | Lowest, $O(n)$ | Interactions don't matter, or max interpretability and tight budget |
| FM | First and second | Factorized dot product, one vector per feature | $nk$ | Low, linearized $O(k\bar n)$ | Default for sparse CTR and ranking; best return on complexity |
| FFM | First and second, field-aware | Factorized, one vector per feature per field | $nFk$ | Higher, $O(k\bar n^2)$, big table | Squeezing the last thousandth of AUC and can afford the table |
| Deep (DeepFM, DCN) | First, second, and higher | FM or cross network plus an MLP, shared embedding | $nk$ plus MLP | Higher, GPU often | Higher-order signal you can name, enough data, serving headroom |

The table makes the strategy obvious. There is a clear ordering by cost, and the AUC gains shrink as you climb it: LR to FM is the big cheap jump (all pairwise interactions for one embedding table), FM to FFM or deep is the small expensive jump (field awareness or higher orders for a much heavier model). The disciplined path is to climb the ladder one rung at a time and measure each rung against the last, because the marginal AUC per dollar of serving cost falls steeply and the right stopping point is wherever your constraints say it is, not the top.

## 10. Case studies and real numbers

A few concrete data points anchor where FM and FFM actually earned their reputations.

**Rendle's original FM (ICDM 2010).** Steffen Rendle introduced factorization machines as a general predictor that subsumes MF, SVD++, and several other factorization models under one equation, with the linear-time computation as the headline property. The paper's experiments on the Netflix and KDD Cup 2012 data showed FM matching specialized factorization models while being a single, general implementation, which was the real contribution: not a new state of the art on one task, but one model that replaced a zoo of bespoke factorization variants. The companion `libFM` tool, with its SGD, ALS, and MCMC trainers, became the reference implementation, and the MCMC variant was notable for auto-tuning regularization, which is why it won portions of several KDD Cups with minimal hand-tuning.

**FFM and the Kaggle CTR sweep (Juan et al., RecSys 2016).** Yuchin Juan and colleagues introduced field-aware FM and, more importantly for its fame, used it to win two major CTR competitions: the 2014 Criteo display-advertising challenge and the 2015 Avazu click-through-rate challenge. The paper reports FFM beating FM on both Criteo and Avazu logloss, with the per-field vectors as the decisive ingredient. Their `libffm` implementation, with the field-annotated sparse format we saw earlier, became the standard tool, and for a couple of years "ensemble of FFMs" was a near-default strong baseline on any Kaggle CTR problem. The paper is also unusually honest about the cost: it documents the $F$-fold parameter blowup and recommends a small $k$ to compensate, the exact engineering trade-off we worked through above.

**Criteo as the field's yardstick.** The Criteo 1TB and Criteo-Kaggle datasets became the standard CTR benchmark precisely because they are large, sparse, and realistic: 39 fields, tens of millions of examples, real industrial click logs. Essentially every CTR paper since 2016, including DeepFM, DCN, xDeepFM, and AutoInt, reports Criteo AUC and logloss, which is what lets you read a clean progression: LR around 0.79 AUC, FM around 0.80, FFM and the early deep models around 0.803 to 0.806, and the best modern models pushing past 0.81. The differences look tiny because AUC is a saturating metric near 0.8, but at industrial scale each thousandth is a meaningful revenue lever, which is the entire reason the field obsesses over the third decimal place.

**Where FM sits in a shipped stack.** In a production ad ranker, FM-family models rarely run alone; they are the second-order component inside a larger architecture (DeepFM-style) or a strong baseline the deep model must beat. Many teams report that a well-tuned FM or DeepFM captures most of the achievable AUC, and that the deeper, fancier models add only a thousandth or two over it at significantly higher serving cost. That is the recurring lesson of this part of the series: the FM second-order term does the heavy lifting, and the marginal returns to architectural complexity above it are real but small. Knowing that lets you decide, with numbers, when to stop.

## 11. When to reach for FM, FFM, LR, or deep

A decisive guide, because every choice is a cost.

Reach for **logistic regression** when interactions genuinely do not matter for your problem, when you need maximum interpretability (every weight is a feature's main effect and you can read it off), or when latency and memory are so tight that even an embedding table is too much. LR is also the right *starting* baseline always, because if FM does not beat a tuned LR, something is wrong with your FM, not your data.

Reach for **plain FM** as the default when interactions matter and you have sparse categorical features, which describes essentially every CTR and recommendation ranking problem. FM is the best return on complexity in this family: it learns all pairwise interactions in linear time with one embedding table, generalizes MF and SVD++ so you can fold in user history and context as fields, and ships at low latency. If you are picking one model from this post to put in production, pick FM.

Reach for **FFM** when you are squeezing the last thousandth of AUC and can afford the $F$-fold parameter blowup and the quadratic interaction cost. The classic setting is a high-stakes ad auction where a 0.004 AUC lift is worth seven figures and the serving host has room for a multi-gigabyte table. Outside that regime, FFM's cost rarely pays for itself, and the field structure it adds is often better captured by a deep model that shares one embedding. Do not reach for FFM if FM hits your target; the extra parameters are pure cost.

Reach for a **deep model (DeepFM, DCN)** when you have evidence that higher-order interactions matter (a third-order cross you can name that FM provably cannot express), when you have enough data to train the extra capacity without overfitting, and when you can afford the serving cost. But do not reach for deep just because it is fashionable: measure DeepFM against a tuned FM, and if the lift is a thousandth at triple the latency, the FM may be the better Pareto point. The honest order of operations is LR baseline, then FM, then measure whether FFM or deep clears the bar your constraints set.

The single most common mistake is skipping straight to a deep model and never building the FM baseline, then having no idea how much of the AUC came from the second-order interactions (almost all of it) versus the deep higher-order capacity (usually a sliver). Build FM first. It tells you the ceiling of pairwise interactions, and that number is what you measure everything else against.

## 12. Key takeaways

- **Linear models cannot learn interactions, and manual one-hot crosses explode and starve.** A cross of two high-cardinality fields has a combinatorial number of columns, almost all unobserved, so logistic regression learns each cross weight in isolation from a handful of examples or not at all.
- **FM factorizes every pairwise interaction into a dot product of per-feature latent vectors.** The interaction matrix $W$ is replaced by a low-rank $V V^\top$, so $\binom{n}{2}$ free weights become $nk$ shared parameters and unseen pairs still get a sensible score.
- **Parameter sharing is why FM beats independent cross weights under sparsity.** A pair never seen in training still has a defined interaction, because both features' vectors were learned from every *other* pair they appeared in. This is the transitive bridge from matrix factorization, generalized to arbitrary features.
- **The linearization trick takes the pairwise term from $O(kn^2)$ to $O(kn)$** via $\frac{1}{2}\sum_f[(\sum_i v_{i,f}x_i)^2 - \sum_i v_{i,f}^2 x_i^2]$, the squared-sum-minus-sum-of-squares identity. On sparse data the real cost is a dozen multiply-adds per latent dimension, which is the reason FM is servable.
- **FM generalizes MF, SVD++, and more under one equation.** Encode user and item as two one-hot fields and FM is biased MF exactly; add a history field and it is SVD++. New signals become new fields, not new model implementations.
- **FFM gives each feature a vector per field**, capturing field-specific interactions and winning the Criteo and Avazu Kaggle competitions, at the price of an $F$-fold parameter blowup and the loss of the linearization (back to quadratic). Use a small $k$ to compensate.
- **FM is the best return on complexity; FFM and deep are diminishing returns you pay for in latency and memory.** LR to FM is a big, cheap lift; FM to FFM or deep is a small, expensive one. Build the FM baseline first and measure everything against it.
- **FM is the second-order backbone of the deep CTR zoo.** DeepFM runs FM and an MLP in parallel over a shared embedding; DCN swaps in an explicit cross network. Every step is the same idea: make the interaction a shared, learnable, factorized object.

## 13. Further reading

- Steffen Rendle, *Factorization Machines*, ICDM 2010, and *Factorization Machines with libFM*, ACM TIST 2012 — the original model, the linear-time derivation, and the reference implementation with SGD, ALS, and MCMC trainers.
- Yuchin Juan, Yong Zhuang, Wei-Sheng Chin, Chih-Jen Lin, *Field-aware Factorization Machines for CTR Prediction*, RecSys 2016 — the FFM model and the Criteo and Avazu Kaggle wins, with the honest cost analysis.
- Huifeng Guo et al., *DeepFM: A Factorization-Machine based Neural Network for CTR Prediction*, IJCAI 2017 — FM and an MLP in parallel over a shared embedding; the direct deep descendant.
- Ruoxi Wang et al., *Deep and Cross Network for Ad Click Predictions*, ADKDD 2017 — the cross network for explicit higher-order interactions.
- `xlearn` documentation (FM and FFM, LIBSVM and libffm formats) and Rendle's `libFM` site — the production tools and their real APIs.
- Within this series: [matrix factorization, the workhorse of recommenders](/blog/machine-learning/recommendation-systems/matrix-factorization-the-workhorse) (the two-field special case FM generalizes), [DeepFM and automatic feature interactions](/blog/machine-learning/recommendation-systems/deepfm-and-automatic-feature-interactions) (the deep descendant), [the ranking model, CTR prediction foundations](/blog/machine-learning/recommendation-systems/the-ranking-model-ctr-prediction-foundations) (where FM lives in the funnel), [the data and features of recommenders](/blog/machine-learning/recommendation-systems/the-data-and-features-of-recommenders) (the sparse features FM consumes), and the capstone [the recommender systems playbook](/blog/machine-learning/recommendation-systems/the-recommender-systems-playbook).
