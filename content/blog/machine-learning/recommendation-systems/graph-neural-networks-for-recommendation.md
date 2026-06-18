---
title: "Graph Neural Networks for Recommendation"
date: "2026-06-19"
publishDate: "2026-06-19"
description: "Treat collaborative filtering as link prediction on a user-item graph, derive the LightGCN propagation rule and why dropping the nonlinearity beats NGCF, scale it like PinSage with random-walk neighbor sampling, implement LightGCN in PyTorch with BPR loss, and measure Recall@20 and NDCG@20 against MF on Gowalla."
tags:
  [
    "recommendation-systems",
    "recsys",
    "graph-neural-networks",
    "lightgcn",
    "ngcf",
    "pinsage",
    "collaborative-filtering",
    "message-passing",
    "machine-learning",
    "pytorch",
    "gowalla",
  ]
category: "machine-learning"
subcategory: "Recommendation Systems"
author: "Hiep Tran"
featured: true
readTime: 52
image: "/imgs/blogs/graph-neural-networks-for-recommendation-1.png"
---

A few years ago I inherited a recommender whose offline metrics had quietly stalled. The team had already done the obvious things: matrix factorization, then a two-tower model with rich features, then a careful re-ranker. Each step bought a little. But the catalog had a long, sparse tail, and most of our users had touched fewer than ten items. For those users the model had almost nothing to work with. A user who rated three movies has a vector trained on three positive signals, and that vector is mostly noise. We kept asking the same question in the standup: where is the rest of the signal hiding? It was hiding one hop away. The three movies our sparse user rated were each rated by thousands of other users, and those users had collectively rated tens of thousands of other movies. The signal we needed was not in our user's own history; it was in the *structure* around her history. Matrix factorization could not see that structure, because matrix factorization treats each user as an isolated row.

The reframing that unlocked this is the subject of this post. Stop thinking of your interaction log as a sparse matrix and start thinking of it as a graph: a node for every user, a node for every item, and an edge wherever a user touched an item. Collaborative filtering becomes link prediction on this graph: given the edges you have observed, which missing edges are most likely real? And once it is a graph, you can do the one thing matrix factorization cannot. You can let each node's embedding be refined by the embeddings of its neighbors, and *their* neighbors, and so on, so that signal flows along paths of length two, three, and more. A user borrows representation from users who liked the same things she did, who in turn pull from the things *those* users liked. That is the collaborative signal, made explicit and made *high-order*. A graph neural network, or GNN, is the machinery that does this flowing.

![Diagram of a bipartite user-item graph where Alice connects to Bob through a shared rated item, and Bob connects forward to a candidate item that Alice has never seen](/imgs/blogs/graph-neural-networks-for-recommendation-1.png)

This is a post in the series **Recommendation Systems: From Click to Production**, and it lives in the retrieval stage of the funnel. The series spine, set up in [what is a recommender system](/blog/machine-learning/recommendation-systems/what-is-a-recommender-system), is the retrieval to ranking to re-ranking funnel fed by the serve-log-train feedback loop, read off the offline-versus-online gap. A GNN recommender is, at serving time, just another embedding model: you precompute one vector per node, push the item vectors into an approximate-nearest-neighbor index, and at request time embed the user and query the index. The same physics as the [two-tower model](/blog/machine-learning/recommendation-systems/the-two-tower-model-for-retrieval). What changes is *how the embeddings are trained*: not from each node's own interactions alone, but from a neighborhood that can stretch several hops across the graph. By the end of this post you will be able to build the user-item graph, derive the LightGCN propagation rule $E^{(k+1)} = \tilde{A} E^{(k)}$ from the graph Laplacian, explain *why* deleting the feature transform and nonlinearity makes LightGCN both simpler and better than NGCF, implement LightGCN in PyTorch with a normalized sparse adjacency and BPR loss, scale it the way PinSage does with random-walk neighbor sampling, and measure Recall@20 and NDCG@20 against BPR-MF on Gowalla. We will also see exactly when a GNN does *not* earn its cost, which is more often than the literature lets on.

## 1. The interaction graph and CF as link prediction

Start from the data you already have. In [collaborative filtering from first principles](/blog/machine-learning/recommendation-systems/collaborative-filtering-from-first-principles) and [matrix factorization](/blog/machine-learning/recommendation-systems/matrix-factorization-the-workhorse) we wrote the interaction log as a matrix $R \in \{0,1\}^{M \times N}$ with $M$ users and $N$ items, where $R_{ui} = 1$ if user $u$ touched item $i$. That same object is a graph. Make a node for each of the $M$ users and a node for each of the $N$ items. Draw an edge between user $u$ and item $i$ exactly when $R_{ui} = 1$. There are no user-user edges and no item-item edges, only user-item edges, which makes this a **bipartite graph**: the nodes split into two groups and every edge crosses between them.

The full adjacency matrix of this graph is a square matrix of size $(M + N) \times (M + N)$, blocked like this:

$$
A = \begin{pmatrix} 0 & R \\ R^\top & 0 \end{pmatrix}
$$

The top-left and bottom-right blocks are zero because there are no within-group edges. The off-diagonal blocks are the interaction matrix $R$ and its transpose. This single equation is the bridge between the matrix view of collaborative filtering and the graph view: $R$ is the off-diagonal of $A$. Everything a GNN does to a recommendation problem, it does by multiplying vectors through $A$.

With the graph in hand, collaborative filtering has a clean restatement. **CF is link prediction on the bipartite graph.** You observe some edges (the interactions you have logged). You want to score the edges you have *not* observed: for a user node $u$ and an item node $i$ with no edge between them, how likely is it that this edge should exist? Rank the candidate items for $u$ by that score, take the top $K$, and you have a recommendation list. This is the same top-$K$ retrieval problem the whole series is about, recast as predicting which missing edges in a graph are real. Once you see it this way, an entire field, graph representation learning, becomes available to recommendation.

### Why neighbors carry the collaborative signal

Here is the intuition the matrix view obscures. Suppose Alice has rated three movies. In matrix factorization, Alice's vector $p_{\text{Alice}}$ is updated only by those three positive examples and her sampled negatives. Three gradients. But on the graph, Alice's three rated movies each have many other raters, and those raters have rated many other movies. If we let Alice's representation be partly *built from* the representations of the movies she rated, and let those movies' representations be partly built from all of *their* raters, then Alice's vector inherits information from people who share her taste even though she has never met them and never co-rated a single item directly with most of them. That is the **collaborative signal**, and the key word is *high-order*: it lives not in Alice's direct edges but in paths of length two, three, and beyond through the graph.

Trace a concrete path on figure 1. Alice rated item A. Item A was also rated by Bob (a 2-hop path: Alice to A to Bob). Bob rated item C, which Alice has never seen (a 3-hop path: Alice to A to Bob to C). The path `Alice -> A -> Bob -> C` is a hypothesis: because Alice and Bob agree on A, and Bob likes C, maybe Alice will like C. Matrix factorization can learn this *implicitly* if A and C end up near each other in latent space, but it has no explicit mechanism to propagate along the path. A GNN propagates along the path by construction. After two layers of message passing, Alice's embedding has literally aggregated information from the 2-hop neighborhood (other users who rated her items); after three layers, from the 3-hop neighborhood (items those users rated). Each layer of the GNN extends the reach by one hop.

The number of paths matters too, not just their existence. If Alice and Bob share *one* item, that is weak evidence. If they share twelve items, the collaborative signal is strong, and a good propagation scheme weights it accordingly. We will see that the normalization in LightGCN does exactly this: a user connected to many items, and items connected to many users, get their contributions scaled down so that high-degree popular items do not drown out the signal. Popularity bias, the self-reinforcing failure mode we keep returning to in this series, shows up here as high-degree nodes, and the graph normalization is one of the few places you can fight it for free.

### The bipartite structure has consequences worth knowing

Two structural facts about the user-item graph shape everything a GNN does to it, and it is worth stating them precisely. First, **because the graph is bipartite, all odd-length paths cross between groups and all even-length paths stay within a group.** A path of length one goes user-to-item; a path of length two goes user-to-item-to-user, ending back on a user; a path of length three ends on an item. This is why the receptive field of a $K$-layer GNN alternates: after one layer a user has aggregated items (its direct ratings), after two layers it has aggregated users (people who co-rated those items), after three layers items again (what those people also liked). The path `Alice -> A -> Bob -> C` from figure 1 is exactly a length-three path, which is why it takes three layers of propagation for $C$ to influence Alice's embedding. If you only ever use one layer, a user can never see another user; it takes two layers for user-to-user information to flow, and the most useful collaborative signal often lives at the 2- and 3-hop level. This is a quiet argument against single-layer GNNs that beginners tend to miss.

Second, **the powers of the adjacency count paths.** The $(v, w)$ entry of $A^k$ is the number of paths of length exactly $k$ between nodes $v$ and $w$. So $A^2$, restricted to its user-user block, is the user co-occurrence matrix: $(A^2)_{u u'}$ is the number of items both $u$ and $u'$ rated, the literal count of shared items between two users. That co-occurrence count is the classical user-user collaborative-filtering similarity. A GNN that propagates through $A$ (or its normalized form $\tilde A$) is therefore computing a smoothed, learned version of the same path-counting that hand-built CF similarity did, but it lets gradient descent decide how much each path contributes instead of hard-coding a cosine. Seeing the connection makes the whole approach less mysterious: the graph operation is a principled generalization of "two users are similar if they rated many of the same things," extended to arbitrary path lengths.

## 2. From GCN to NGCF: message passing for recommendation

A graph convolutional network (GCN) refines node embeddings by repeatedly aggregating each node's neighbors. Give every node $v$ an initial embedding $e_v^{(0)}$. At each layer $k$, compute a new embedding for $v$ by combining its current embedding with a function of its neighbors' current embeddings:

$$
e_v^{(k+1)} = \phi\!\left( e_v^{(k)}, \; \text{AGG}\big(\{ e_u^{(k)} : u \in \mathcal{N}(v) \}\big) \right)
$$

where $\mathcal{N}(v)$ is the set of $v$'s neighbors, $\text{AGG}$ is a permutation-invariant aggregator (sum, mean, or max), and $\phi$ is a combination function, classically a learnable linear map followed by a nonlinearity. Stack $K$ such layers and node $v$'s final embedding has absorbed information from its entire $K$-hop neighborhood. This is **message passing**: each node sends its embedding to its neighbors, each node aggregates the messages it receives, and the cycle repeats. It is the workhorse of graph deep learning, and the underlying idea, that an embedding can be refined by the embeddings of the things it is connected to, is the same one that powers most modern representation learning.

The aggregator must be **permutation invariant** because a node's neighbors have no canonical order; the set of items a user rated is a set, not a sequence, so the function that summarizes it cannot depend on the order you happen to iterate them in. Sum, mean, and max all satisfy this, and the choice matters. A *sum* aggregator preserves degree information (a node with more neighbors gets a larger-magnitude message) but lets high-degree nodes dominate. A *mean* aggregator throws away raw degree but is stable. A normalized sum, which is what the symmetric normalization in the next section produces, is the middle path: it keeps a controlled amount of degree information while discounting the very-high-degree nodes that would otherwise overwhelm everything. This is not a cosmetic choice; on a recommendation graph where item degrees span four or five orders of magnitude (a viral item versus a niche one), the aggregator's treatment of degree directly controls how much popularity bias the model bakes in.

The original GCN propagation rule, from Kipf and Welling (2017), writes one layer in matrix form. With node embeddings stacked into a matrix $E^{(k)} \in \mathbb{R}^{(M+N) \times d}$, a learnable weight matrix $W^{(k)}$, the normalized adjacency $\hat{A}$ (we will define it precisely in section 5), and a nonlinearity $\sigma$ such as ReLU:

$$
E^{(k+1)} = \sigma\!\left( \hat{A} \, E^{(k)} \, W^{(k)} \right)
$$

The $\hat{A} E^{(k)}$ part does the neighbor aggregation: left-multiplying the embedding matrix by the adjacency mixes each node with its neighbors. The $W^{(k)}$ part is a learnable feature transformation, the same trick as a dense layer in any neural network. The $\sigma$ part adds nonlinearity. This is the standard recipe imported wholesale from semi-supervised node classification, where it works beautifully because nodes have rich features (a document's bag of words, an image's pixels) that genuinely benefit from being transformed and nonlinearly combined.

### NGCF: GCN applied carefully to recommendation

Neural Graph Collaborative Filtering (NGCF), from Wang et al. (2019), was the paper that brought explicit high-order message passing into mainstream recommendation, and it did something important: it argued that you should encode the collaborative signal *in the embedding function itself*, not just hope matrix factorization learns it. NGCF's propagation rule keeps the full GCN machinery and adds a feature-interaction term. One NGCF layer updates user $u$'s embedding as:

$$
e_u^{(k+1)} = \sigma\!\Bigg( W_1^{(k)} e_u^{(k)} + \sum_{i \in \mathcal{N}(u)} \frac{1}{\sqrt{|\mathcal{N}(u)|\,|\mathcal{N}(i)|}} \Big( W_1^{(k)} e_i^{(k)} + W_2^{(k)} \big( e_i^{(k)} \odot e_u^{(k)} \big) \Big) \Bigg)
$$

Read it piece by piece. The $W_1^{(k)} e_u^{(k)}$ term is the self-connection: keep some of your own current embedding. The sum runs over $u$'s neighbor items $i$, each weighted by $1/\sqrt{|\mathcal{N}(u)|\,|\mathcal{N}(i)|}$, the symmetric normalization that discounts high-degree nodes. Inside the sum, $W_1^{(k)} e_i^{(k)}$ passes the neighbor's transformed message, and $W_2^{(k)}(e_i^{(k)} \odot e_u^{(k)})$ adds an element-wise interaction between user and item, which is the "feature affinity" NGCF introduced as its contribution. Finally $\sigma$ is a LeakyReLU. There are two trainable weight matrices per layer and a nonlinearity. NGCF concatenates the embeddings from all layers to form the final representation, scores user-item pairs by dot product, and trains with BPR loss.

NGCF worked. It beat matrix factorization on standard benchmarks and demonstrated that explicit high-order propagation was a real source of lift. But it carried over the heavy GCN machinery on faith. The weight matrices, the element-wise interaction term, the nonlinearity, all imported from a setting where nodes have semantic features. And in collaborative filtering, the nodes do *not* have semantic features. The node embeddings are free parameters, initialized randomly and learned from scratch. That difference is the crack that LightGCN drove a wedge into.

![Side-by-side comparison of one NGCF layer with trainable weight matrices and a ReLU against one LightGCN layer that only normalizes and averages neighbors and reaches higher Recall@20](/imgs/blogs/graph-neural-networks-for-recommendation-2.png)

## 3. LightGCN: throw out everything that does not help

He et al. (2020) asked a question that, in hindsight, is obvious: *which parts of NGCF actually contribute to its accuracy?* They ran the ablation no one had bothered to run. They removed the feature transformation matrices $W$. They removed the nonlinearity $\sigma$. They removed the self-connection and the element-wise interaction term. One by one, then together. And the result, reported in the LightGCN paper, was startling: removing the feature transformation and the nonlinearity did not hurt accuracy. It *improved* it. The two design elements imported from GCN, the ones everyone assumed were load-bearing, were actively dragging NGCF down.

The reasoning, once you see it, is clean. In node classification, the input to a GCN is a node's *feature vector*, a semantic object (word counts, pixel statistics) where learning a transformation $W$ and applying a nonlinearity genuinely extracts useful structure. In collaborative filtering, the input is a *free embedding*, a vector of trainable parameters with no intrinsic semantics. There is nothing to transform. Applying $W$ to a free embedding and then refining it by gradient descent is redundant with simply learning a better free embedding in the first place, and worse, it adds parameters that overfit the sparse interaction signal and make optimization harder. The nonlinearity compounds the problem: stacking ReLUs across layers makes the propagation a complicated, hard-to-train function when all you actually want is to *average a node with its neighbors*. As He et al. put it, the feature transformation and nonlinearity contribute little to collaborative filtering and, given the difficulty they add to training, are harmful.

So LightGCN keeps only the part that matters: the neighbor aggregation. One LightGCN layer is just a normalized average of neighbors, with no weight matrix and no nonlinearity:

$$
e_u^{(k+1)} = \sum_{i \in \mathcal{N}(u)} \frac{1}{\sqrt{|\mathcal{N}(u)|}\,\sqrt{|\mathcal{N}(i)|}} \, e_i^{(k)}
\qquad
e_i^{(k+1)} = \sum_{u \in \mathcal{N}(i)} \frac{1}{\sqrt{|\mathcal{N}(i)|}\,\sqrt{|\mathcal{N}(u)|}} \, e_u^{(k)}
$$

That is the entire layer. A user's next embedding is the symmetrically normalized sum of its neighbor items' current embeddings; an item's next embedding is the symmetrically normalized sum of its neighbor users' current embeddings. No $W$. No $\sigma$. No self-connection inside the layer (the self-information is recovered by the layer-combination step, which we get to in section 5). LightGCN is, as the authors say, simpler *and* better. It is one of those rare results where the more accurate model is also the cheaper one, and it became the standard GNN baseline in recommendation almost overnight.

### What "simpler and better" buys you in practice

The practical consequences are large. With no per-layer weight matrices, the *only* trainable parameters in LightGCN are the base embeddings $E^{(0)}$, exactly the same parameter count as matrix factorization. Every layer of propagation is a fixed, parameter-free sparse matrix multiply. That means LightGCN trains faster per epoch than NGCF, converges in fewer epochs, has fewer hyperparameters to tune, and is far less prone to overfitting. You also get a cleaner story for the embedding table memory, the thing that OOMs hosts in production: LightGCN's table is the same size as MF's, because propagation adds no parameters. We will quantify all of this in section 8. For now, hold the headline: LightGCN deletes the parts of the graph convolution that were imported from a different problem, and is rewarded with both lower cost and higher accuracy.

## 4. Worked example: one LightGCN propagation step by hand

Let us do a single layer of LightGCN propagation on a tiny graph, by hand, so the matrix multiply becomes concrete arithmetic you can check yourself. The rule to keep in mind is simple: a layer just replaces each node's vector with a normalized average of its neighbors' vectors.

#### Worked example: propagating a 4-node graph one hop

Take two users and two items. User $u_1$ rated items $i_1$ and $i_2$. User $u_2$ rated only item $i_1$. So the edges are $u_1$–$i_1$, $u_1$–$i_2$, $u_2$–$i_1$. The degrees are: $d(u_1) = 2$, $d(u_2) = 1$, $d(i_1) = 2$ (rated by both users), $d(i_2) = 1$ (rated only by $u_1$).

Give every node a one-dimensional embedding to keep the arithmetic visible. Initialize $e^{(0)}$: let $e_{u_1} = 1.0$, $e_{u_2} = 2.0$, $e_{i_1} = 3.0$, $e_{i_2} = 4.0$. We propagate one LightGCN layer. The normalization coefficient on edge between nodes $a$ and $b$ is $1/(\sqrt{d(a)}\,\sqrt{d(b)})$.

Update $u_1$. Its neighbors are $i_1$ and $i_2$:

$$
e_{u_1}^{(1)} = \frac{1}{\sqrt{d(u_1)}\sqrt{d(i_1)}} e_{i_1}^{(0)} + \frac{1}{\sqrt{d(u_1)}\sqrt{d(i_2)}} e_{i_2}^{(0)} = \frac{1}{\sqrt 2 \sqrt 2}(3.0) + \frac{1}{\sqrt 2 \sqrt 1}(4.0)
$$

That is $\frac{1}{2}(3.0) + \frac{1}{\sqrt 2}(4.0) = 1.5 + 2.828 = 4.328$.

Update $u_2$. Its only neighbor is $i_1$:

$$
e_{u_2}^{(1)} = \frac{1}{\sqrt{d(u_2)}\sqrt{d(i_1)}} e_{i_1}^{(0)} = \frac{1}{\sqrt 1 \sqrt 2}(3.0) = \frac{3.0}{1.414} = 2.121
$$

Update $i_1$. Its neighbors are $u_1$ and $u_2$:

$$
e_{i_1}^{(1)} = \frac{1}{\sqrt 2 \sqrt 2}(1.0) + \frac{1}{\sqrt 2 \sqrt 1}(2.0) = 0.5 + 1.414 = 1.914
$$

Update $i_2$. Its only neighbor is $u_1$:

$$
e_{i_2}^{(1)} = \frac{1}{\sqrt 1 \sqrt 2}(1.0) = 0.707
$$

Notice three things. First, every new value is a *weighted average* of neighbor values, never a transformation of the node's own value (there is no self-loop inside the layer). Second, the high-degree item $i_1$, rated by both users, contributes to its neighbors with a smaller per-edge weight ($1/2$) than the low-degree item $i_2$ ($1/\sqrt2$), exactly the popularity discount we wanted. Third, after one hop, $u_1$ and $u_2$, who share item $i_1$, now have embeddings ($4.328$ and $2.121$) that both carry $i_1$'s influence; they have begun to move toward each other in representation space because they are 2-hop neighbors through $i_1$. That last point *is* the collaborative signal flowing along the path. Run a second layer and the users would start absorbing each other's information directly. This is high-order connectivity, computed by a single sparse matrix multiply.

![Two-column figure contrasting matrix factorization seeing only a user's directly rated items against a GNN whose receptive field reaches two and three hops to borrow signal for sparse users](/imgs/blogs/graph-neural-networks-for-recommendation-5.png)

## 5. The science: deriving the LightGCN propagation in matrix form

The per-node update in section 3 is correct but slow to think about. The matrix form is where LightGCN becomes beautiful, and where the connection to graph smoothing becomes provable. We need three matrices.

First, the **adjacency** $A$ of the bipartite graph, the $(M+N) \times (M+N)$ matrix from section 1:

$$
A = \begin{pmatrix} 0 & R \\ R^\top & 0 \end{pmatrix}
$$

Second, the **degree matrix** $D$, a diagonal matrix where $D_{vv}$ is the number of neighbors of node $v$. For a user node that is the number of items it rated; for an item node, the number of users who rated it. $D = \text{diag}(A \mathbf{1})$, the row sums of $A$ on the diagonal.

Third, the **symmetrically normalized adjacency**:

$$
\tilde{A} = D^{-1/2} A \, D^{-1/2}
$$

Here $D^{-1/2}$ is the diagonal matrix with $1/\sqrt{D_{vv}}$ on the diagonal. Multiplying $A$ on the left by $D^{-1/2}$ divides each row $v$ by $\sqrt{d(v)}$; multiplying on the right by $D^{-1/2}$ divides each column $w$ by $\sqrt{d(w)}$. So the $(v, w)$ entry of $\tilde A$ is $A_{vw} / (\sqrt{d(v)}\sqrt{d(w)})$, which is exactly the per-edge normalization coefficient from the per-node update. The normalization is symmetric: it discounts an edge by the geometric mean of its two endpoints' degrees, so a connection between two popular, high-degree nodes counts for less than a connection between two niche, low-degree nodes. (Plain GCN adds a self-loop to $A$ before normalizing; LightGCN deliberately omits the self-loop because the layer-combination step recovers self-information more cleanly.)

With $\tilde A$ defined, the entire LightGCN per-node update collapses into one line:

$$
\boxed{\;E^{(k+1)} = \tilde{A} \, E^{(k)}\;}
$$

That is the LightGCN propagation rule. Stack the user and item embeddings into one matrix $E^{(k)} \in \mathbb{R}^{(M+N)\times d}$, left-multiply by the normalized adjacency, and you get the next layer. No weights, no nonlinearity. Unrolling the recursion from the base embeddings $E^{(0)}$:

$$
E^{(k)} = \tilde{A}^{\,k} E^{(0)}
$$

The $k$-th layer's embeddings are the base embeddings hit by the $k$-th power of the normalized adjacency. Because $\tilde A^k$ has nonzero entries exactly where there is a path of length $k$, the $k$-th layer aggregates the $k$-hop neighborhood. This makes the "$K$ layers equals $K$ hops" statement a precise algebraic fact, not a hand-wave.

### The layer-combination readout

A single layer's output $E^{(K)} = \tilde A^K E^{(0)}$ over-mixes: at depth $K$ a node's embedding is dominated by far-away nodes and has lost its own identity (this is over-smoothing, section 7). LightGCN avoids relying on any single layer by combining *all* layers into the final embedding through a weighted sum:

$$
E = \sum_{k=0}^{K} \alpha_k \, E^{(k)} = \sum_{k=0}^{K} \alpha_k \, \tilde{A}^{\,k} E^{(0)}
$$

with combination weights $\alpha_k \ge 0$. He et al. found that simply setting $\alpha_k = 1/(K+1)$, a uniform average over all $K+1$ layers including the base, works as well as learning the weights, so the standard recipe is just **mean over layers**. This is why LightGCN omits the self-loop inside each layer: $E^{(0)}$ is included directly in the readout, so each node keeps its own identity through the $\alpha_0 E^{(0)}$ term rather than through a per-layer self-connection. The final user embedding is $e_u = \frac{1}{K+1}\sum_{k=0}^K e_u^{(k)}$, the final item embedding is the analogous average, and the predicted score is the dot product $\hat y_{ui} = e_u^\top e_i$.

![Vertical stack of LightGCN layers from the base embedding through three propagation hops into a mean layer-combination readout and a BPR ranking score](/imgs/blogs/graph-neural-networks-for-recommendation-3.png)

### Why this is graph smoothing, and what it converges to

Here is the deeper reading. The symmetric normalized graph **Laplacian** is $L = I - \tilde A$, so $\tilde A = I - L$. Repeatedly applying $\tilde A$ to a signal is one step of **Laplacian smoothing**: each application pulls every node's value toward the average of its neighbors, reducing high-frequency variation across the graph. This is exactly the same operation as a diffusion or heat-flow step on the graph. The reason LightGCN works is that the collaborative signal is *low-frequency*: users who are close on the graph should have similar embeddings, and smoothing enforces that similarity. Dropping the nonlinearity is what makes this interpretation exact; a stack of LightGCN layers is a *linear* low-pass graph filter, and you can analyze it with the eigendecomposition of $\tilde A$.

That same analysis predicts the failure mode. Iterating $\tilde A^k$ as $k \to \infty$ converges to the projection onto the top eigenvector of $\tilde A$, which for a connected component is proportional to the square root of node degrees. In the limit, *every node's embedding direction collapses to the same vector*, scaled by degree. All distinguishing information is smoothed away. This is **over-smoothing**, and it is not a bug you can patch; it is the fixed point of the smoothing operator. It is precisely why you use a small $K$ (2 or 3) and why the layer-combination average, which keeps the unsmoothed $E^{(0)}$ in the mix, is essential. We will watch over-smoothing happen empirically in section 9.

### Why dropping the nonlinearity makes the analysis exact

It is worth dwelling on *why* the spectral story above is only available because LightGCN dropped the nonlinearity, because this is the deepest version of the "simpler and better" argument. The eigenvalues of $\tilde A$ all lie in the range $[-1, 1]$ (a standard fact for the symmetric normalized adjacency). Write a node's base embedding in the eigenbasis of $\tilde A$ as a sum of components along each eigenvector $u_j$ with eigenvalue $\lambda_j$. One application of $\tilde A$ scales the component along $u_j$ by $\lambda_j$. After $k$ applications, that component is scaled by $\lambda_j^k$. The components with $|\lambda_j|$ near 1 (the low-frequency, smooth-across-the-graph directions) survive; the components with $|\lambda_j|$ near 0 (the high-frequency, jagged directions) decay geometrically. So a stack of LightGCN layers is, *exactly* and provably, a low-pass filter on the graph spectrum, attenuating high-frequency variation and keeping the smooth signal. This clean statement holds only because there is no nonlinearity between the multiplies. The moment you insert a ReLU, the operation is no longer a linear filter, the eigenanalysis breaks, and you lose both the interpretability and, empirically, some accuracy. The nonlinearity does not just fail to help; it destroys the very property that makes graph convolution work for collaborative filtering, which is its identity as a learnable low-pass filter. That is the rigorous form of the claim that CF embeddings are not features to transform.

### What the normalization choice actually does

One more piece of the science deserves its own paragraph, because teams get it wrong. There are three reasonable ways to normalize the adjacency, and they behave differently. **Symmetric** normalization $D^{-1/2} A D^{-1/2}$ is what LightGCN uses; it discounts an edge by the geometric mean of its endpoints' degrees and keeps the operator symmetric, so its eigenvalues are real and the spectral analysis above applies cleanly. **Left (random-walk)** normalization $D^{-1} A$ makes each row sum to one, turning propagation into a literal random walk where a node's new embedding is the *plain average* of its neighbors; it is intuitive but not symmetric, so it loses the spectral guarantees. **No normalization** (plain $A$) lets high-degree nodes dominate completely, amplifying popularity bias until the model recommends almost nothing but the head of the catalog. The LightGCN ablation tried these and found symmetric normalization best, which is not a coincidence: it is the only choice that both discounts popularity *and* preserves the clean low-pass-filter interpretation. When you see a graph recommender misbehaving with all of its mass on popular items, the first thing to check is whether someone quietly switched the normalization.

## 6. PinSage: GraphSAGE at web scale

LightGCN's propagation rule, $E^{(k+1)} = \tilde A E^{(k)}$, is a multiply against the *full* adjacency matrix. That is fine for Gowalla (tens of thousands of nodes) and even MovieLens-20M (hundreds of thousands), where $\tilde A$ fits in memory as a sparse tensor. It is impossible for Pinterest, which had a graph of roughly 3 billion nodes (pins and boards) and 18 billion edges when Ying et al. (2018) built PinSage. You cannot materialize a 3-billion-by-3-billion adjacency matrix, and you cannot afford full-graph propagation. PinSage was the first GNN recommender to actually run at web scale, and the techniques it introduced are still how you scale graph recommenders today.

PinSage is built on **GraphSAGE** (Hamilton et al., 2017), whose key idea is *sampling*: instead of aggregating over a node's full neighborhood, sample a fixed-size subset of neighbors and aggregate over the sample. This bounds the per-node computation regardless of how high-degree the node is, and it lets you compute a node's embedding from a small local subgraph without ever touching the global adjacency matrix. PinSage adapted GraphSAGE for the bipartite pin-board graph with four critical engineering moves.

### Random-walk neighbor sampling and importance pooling

Rather than sampling neighbors uniformly, PinSage defines a node's neighborhood by running short **random walks** starting from the node and counting how often each other node is visited. The top-$T$ most-visited nodes (PinSage used $T \approx 50$) become the neighborhood. This is smarter than uniform sampling for two reasons. First, it naturally focuses on the most relevant neighbors, the ones reachable by many short paths, rather than wasting the sample budget on weakly connected nodes. Second, the visit counts give a natural **importance weight**: PinSage aggregates the sampled neighbors with a weighted average where each neighbor's weight is proportional to its normalized random-walk visit count, so a frequently co-visited pin contributes more than a rarely co-visited one. This is called **importance pooling**, and the paper reports it as one of the larger single accuracy gains, on the order of a 46% lift in their offline hit-rate metric over a baseline that pooled uniformly.

![Diagram of PinSage neighbor sampling where two random walks fan out from a target pin, their visit counts are ranked to a top-T set, and importance pooling weights the sample](/imgs/blogs/graph-neural-networks-for-recommendation-6.png)

### MapReduce inference and the producer-consumer trainer

The second move is **MapReduce inference**. To embed billions of nodes you cannot loop; PinSage computes all node embeddings in a single bottom-up MapReduce pass. Layer-1 embeddings for all nodes are computed first (a map over nodes, aggregating sampled layer-0 neighbors), then layer-2 from layer-1, and so on, avoiding the massive redundant re-computation you would get if you embedded each node independently and re-derived its neighbors' embeddings every time. This brought full-catalog inference down to a few hours on a MapReduce cluster, which is what makes daily embedding refresh feasible at that scale. The third move is a **producer-consumer training pipeline**: the GPU trains while CPUs build the next minibatch's subgraphs and gather features, so the expensive GPU never stalls waiting for graph construction.

### Hard-negative curriculum

The fourth move is the one most relevant to the rest of this series. PinSage trains with a max-margin ranking loss (the same family as BPR and the [in-batch negatives we covered for two-tower training](/blog/machine-learning/recommendation-systems/training-two-tower-negatives-and-sampled-softmax)), and the quality of the negatives makes or breaks it. Random negatives, which is what you get from in-batch sampling, are *too easy*: a random pin is wildly unrelated to the query pin, so the model learns to separate obvious things and never sharpens the hard boundary. PinSage adds **hard negatives**: items that are somewhat related to the query (it used items ranked 2000–5000 by personalized PageRank from the query, related but not the true positive) so the model must learn fine distinctions. Crucially they introduced these on a **curriculum**: zero hard negatives in epoch 1, one in epoch 2, $n-1$ in epoch $n$. Throwing hard negatives at an untrained model just confuses it; introducing them gradually as the model gets stronger teaches it to separate near-duplicates, and this is the same logic as the hard-negative discussion in [implicit feedback models](/blog/machine-learning/recommendation-systems/implicit-feedback-models-als-and-bpr). PinSage shipped: the paper reports large offline gains and a deployed online A/B test where PinSage-driven related-pin recommendations lifted engagement by roughly 30% over the prior production system.

## 7. Over-smoothing, cold start, and the honest limits

A GNN recommender is not a free lunch, and the failure modes are specific. Three of them deserve a hard look before you reach for one.

**Over-smoothing.** We derived this in section 5: as you stack layers, $\tilde A^k$ pushes every node toward the same degree-scaled vector, and embeddings lose the ability to distinguish nodes. In practice the symptom is brutal and easy to spot: Recall@20 climbs from layer 1 to layer 2 to layer 3, then *falls off a cliff* at layer 5, 6, 8. The model has smoothed away the very distinctions it needs to rank. The defenses are: keep $K$ small (2 or 3 is the sweet spot for almost every public benchmark), always use the layer-combination average so the un-smoothed $E^{(0)}$ survives in the readout, and if you genuinely need deeper propagation, add residual connections (the "skip" idea that lets the un-smoothed signal bypass deep layers). Over-smoothing is the single most common way teams get LightGCN wrong: they assume deeper is better, the way it usually is in deep learning, and it is exactly backwards here.

![Two-column figure showing a healthy three-layer LightGCN keeping distinct user vectors against an eight-layer model whose vectors collapse to one point and whose Recall@20 drops](/imgs/blogs/graph-neural-networks-for-recommendation-7.png)

**Cold start still needs features.** Pure LightGCN, like all pure collaborative filtering, is helpless on a brand-new user or item with no edges. A node with no edges has no neighbors to aggregate, so propagation leaves its random base embedding untouched. The graph buys you nothing for a truly cold node. This is the same limit as matrix factorization, and the fix is the same: bring in content features. NGCF and LightGCN in their vanilla forms use only ids; to handle cold start you need a model that initializes node embeddings from content (text, image, metadata), which is exactly what PinSage does (its node features are visual and text embeddings of the pin) and what hybrid GNNs do. If cold start is your dominant problem, reach for [content-based and hybrid recommenders](/blog/machine-learning/recommendation-systems/content-based-and-hybrid-recommenders) or a feature-rich two-tower before you reach for a pure-id GNN.

**Cost of message passing at scale.** Full-graph propagation is $O(|E| \cdot d)$ per layer, where $|E|$ is the number of edges. For a graph with hundreds of millions of edges this is expensive per epoch even though it is cheap per parameter, and the sparse adjacency tensor can be large in memory. This is the cost that pushes you from full-graph LightGCN to sampled propagation à la PinSage. There is a real crossover: below roughly a few million edges, full-graph LightGCN on a single GPU is simplest and fastest; above that, you need neighbor sampling and a distributed pipeline, and the engineering cost goes up sharply. We quantify the trade-offs in the comparison figure below.

![Matrix comparing matrix factorization, NGCF, LightGCN, and PinSage across whether each captures high-order signal, scales, uses content features, and its Recall@20](/imgs/blogs/graph-neural-networks-for-recommendation-4.png)

**Serving is the easy part.** The good news: serving a GNN recommender is identical to serving a two-tower model. You run propagation offline, once, to produce the final user and item embeddings; you push the item embeddings into an approximate-nearest-neighbor index; at request time you look up (or, for in-graph users, retrieve precomputed) the user embedding and query the index for the top-$K$ items by dot product. The expensive message passing happens entirely offline. Online, it is a single ANN query, the same maximum-inner-product-search problem covered in two-tower serving and the ANN serving post (faiss, HNSW, ScaNN). A GNN does not make serving harder; it makes *training* heavier in exchange for richer embeddings.

#### Worked example: the cost ledger of a 5-million-edge LightGCN

Put numbers on the trade-off so the decision is concrete. Take a catalog of 1 million items, 500,000 users, and 5 million interaction edges, with $d = 64$. The embedding table is $(1{,}500{,}000) \times 64$ floats $= 96$ million floats $\approx 384$ MB at fp32, the *same* table a two-tower id-embedding model would carry, because LightGCN adds no parameters. Full-graph propagation is one sparse matmul per layer against a $1.5\text{M} \times 1.5\text{M}$ adjacency with 10 million nonzeros (5 million edges, stored both directions), which at 3 layers and $d = 64$ is on the order of a couple of GFLOPs per propagation, milliseconds on a GPU per batch, and a few hours to train 400 epochs. Inference: propagate once to get all 1.5 million final embeddings (seconds on a GPU), build a faiss `IndexHNSWFlat` over the 1 million item vectors (a minute or two), and serve. The online cost is a single HNSW query at fractions of a cent per thousand requests, with p99 in the low tens of milliseconds, identical to a two-tower retriever. The whole *additional* cost of choosing LightGCN over MF is the offline propagation pass, a few seconds at inference and a slightly heavier training loop. That is a remarkably cheap price for +13% Recall@20, which is exactly why LightGCN is the default GNN to try: at this scale the cost ledger is almost free. The ledger only turns expensive when the 5 million edges become 5 billion and full-graph propagation no longer fits, at which point you pay the PinSage engineering tax.

## 8. Implementing LightGCN in PyTorch

Enough theory. Here is a complete, runnable LightGCN you can train on Gowalla, Amazon-book, or MovieLens. The structure is: build the normalized sparse adjacency once, propagate $K$ layers with sparse matmuls, average the layers, and train with BPR loss. First the adjacency builder.

```python
import torch
import numpy as np
import scipy.sparse as sp

def build_norm_adj(user_item_pairs, n_users, n_items):
    """Build the symmetric normalized adjacency A_tilde = D^-1/2 A D^-1/2
    for the bipartite graph as a torch sparse tensor of size
    (n_users + n_items) x (n_users + n_items)."""
    n = n_users + n_items
    rows, cols = [], []
    for u, i in user_item_pairs:
        # user node id = u ; item node id = n_users + i
        rows.append(u);            cols.append(n_users + i)   # user -> item
        rows.append(n_users + i);  cols.append(u)             # item -> user
    data = np.ones(len(rows), dtype=np.float32)
    A = sp.coo_matrix((data, (rows, cols)), shape=(n, n))

    deg = np.asarray(A.sum(axis=1)).flatten()        # node degrees
    d_inv_sqrt = np.power(deg, -0.5, where=deg > 0)   # D^-1/2
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    D_inv_sqrt = sp.diags(d_inv_sqrt)
    A_tilde = D_inv_sqrt @ A @ D_inv_sqrt            # D^-1/2 A D^-1/2

    A_tilde = A_tilde.tocoo()
    idx = torch.tensor(np.vstack([A_tilde.row, A_tilde.col]), dtype=torch.long)
    val = torch.tensor(A_tilde.data, dtype=torch.float32)
    return torch.sparse_coo_tensor(idx, val, (n, n)).coalesce()
```

The adjacency is built once at startup and never changes during training. It is sparse, so even for a few hundred thousand nodes it costs only as much memory as you have edges. Now the model. Note how little there is to it: an embedding table and a propagation loop. There are no per-layer weights, which is the whole point of LightGCN.

```python
import torch.nn as nn
import torch.nn.functional as F

class LightGCN(nn.Module):
    def __init__(self, n_users, n_items, dim=64, n_layers=3, norm_adj=None):
        super().__init__()
        self.n_users, self.n_items, self.n_layers = n_users, n_items, n_layers
        # the ONLY trainable parameters: base embeddings E^(0)
        self.emb = nn.Embedding(n_users + n_items, dim)
        nn.init.normal_(self.emb.weight, std=0.1)
        self.register_buffer("A", norm_adj)  # sparse A_tilde, no grad

    def propagate(self):
        """Return final user and item embeddings after K LightGCN layers
        with the mean layer-combination readout."""
        e = self.emb.weight                 # E^(0)
        layers = [e]
        for _ in range(self.n_layers):
            e = torch.sparse.mm(self.A, e)  # E^(k+1) = A_tilde @ E^(k)
            layers.append(e)
        out = torch.stack(layers, dim=0).mean(dim=0)   # uniform alpha_k
        return out[:self.n_users], out[self.n_users:]   # users, items

    def score(self, users, items):
        eu, ei = self.propagate()
        return (eu[users] * ei[items]).sum(-1)
```

The forward pass is three sparse matmuls (for `n_layers=3`) and a mean. The `torch.sparse.mm(self.A, e)` line *is* the propagation rule $E^{(k+1)} = \tilde A E^{(k)}$ from section 5, executed in one call. Stacking the layers and taking the mean is the uniform layer-combination readout with $\alpha_k = 1/(K+1)$. Everything we derived maps one-to-one onto the code.

### BPR loss and the training loop

LightGCN trains with Bayesian Personalized Ranking, the pairwise loss from [implicit feedback models](/blog/machine-learning/recommendation-systems/implicit-feedback-models-als-and-bpr): for each observed (user, positive item) pair, sample a negative item the user has not interacted with, and push the positive's score above the negative's. The loss is $-\log \sigma(\hat y_{ui} - \hat y_{uj})$ summed over triples, plus L2 regularization on the *base* embeddings only (regularize $E^{(0)}$, not the propagated outputs, since the propagated embeddings have no independent parameters).

```python
def bpr_loss(model, users, pos_items, neg_items, reg=1e-4):
    eu, ei = model.propagate()                  # one propagation per batch
    u  = eu[users]
    pi = ei[pos_items]
    ni = ei[neg_items]
    pos_scores = (u * pi).sum(-1)
    neg_scores = (u * ni).sum(-1)
    # pairwise BPR: positive should outrank the sampled negative
    loss = -F.logsigmoid(pos_scores - neg_scores).mean()
    # L2 on the BASE embeddings touched in this batch (not propagated outputs)
    e0 = model.emb.weight
    reg_loss = reg * (
        e0[users].pow(2).sum()
        + e0[model.n_users + torch.as_tensor(pos_items)].pow(2).sum()
        + e0[model.n_users + torch.as_tensor(neg_items)].pow(2).sum()
    ) / len(users)
    return loss + reg_loss

def train(model, triples, epochs=400, lr=1e-3, batch=2048):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        np.random.shuffle(triples)
        for start in range(0, len(triples), batch):
            b = triples[start:start + batch]
            users = torch.as_tensor([t[0] for t in b])
            pos   = torch.as_tensor([t[1] for t in b])
            neg   = torch.as_tensor([t[2] for t in b])
            opt.zero_grad()
            loss = bpr_loss(model, users, pos, neg)
            loss.backward()
            opt.step()
```

A subtle but important point: `model.propagate()` runs once per batch, computing embeddings for *all* nodes, and then you index out the batch's users and items. That is the standard full-graph LightGCN training mode. It is fine up to a few million edges. The negative sampler here is plain random sampling from items the user has not touched; a more careful implementation pre-builds a per-user interacted set and rejection-samples to avoid false negatives, the same hygiene we covered for [implicit feedback](/blog/machine-learning/recommendation-systems/implicit-feedback-models-als-and-bpr).

### If you would rather not write it yourself

You almost never need to. RecBole ships LightGCN, NGCF, and a dozen graph models with a unified, leakage-safe evaluation harness, which is what I reach for when I want a trustworthy baseline comparison fast.

```python
from recbole.quick_start import run_recbole

# Trains LightGCN on Gowalla with RecBole's standard split and metrics.
run_recbole(
    model="LightGCN",
    dataset="gowalla-merged",
    config_dict={
        "embedding_size": 64,
        "n_layers": 3,
        "reg_weight": 1e-4,
        "train_batch_size": 2048,
        "epochs": 400,
        "eval_args": {"split": {"RS": [0.8, 0.1, 0.1]}, "order": "TO",
                      "group_by": "user", "mode": "full"},
        "metrics": ["Recall", "NDCG"],
        "topk": [20],
        "valid_metric": "NDCG@20",
    },
)
```

The two settings to never get wrong are `"order": "TO"` (temporal ordering of the split, so you train on the past and test on the future, no leakage) and `"mode": "full"` (rank against the *full* item catalog, not a sampled subset). The KDD'20 result that sampled metrics are inconsistent with full metrics, covered in [learning to rank for recommenders](/blog/machine-learning/recommendation-systems/learning-to-rank-for-recommenders), means a LightGCN that looks great under sampled evaluation can be mediocre under full ranking. Always evaluate `mode: full`. The `torch_geometric` library also ships a `LightGCN` model class if you prefer the PyG ecosystem; the math is identical, only the API differs.

### Neighbor sampling for scale

The full-graph training above breaks once the graph is too big to propagate the whole adjacency every batch. The PinSage answer, neighbor sampling, computes each node's embedding from a small sampled subgraph instead. PyTorch Geometric ships the machinery directly: a `NeighborLoader` that, for each seed node in a minibatch, samples a fixed number of neighbors per layer and returns only the induced subgraph, so the per-batch cost is bounded no matter how high-degree any node is. The skeleton looks like this.

```python
from torch_geometric.loader import NeighborLoader
from torch_geometric.data import Data

# edge_index is a 2 x (2|E|) tensor of the bipartite graph's directed edges
data = Data(edge_index=edge_index, num_nodes=n_users + n_items)

loader = NeighborLoader(
    data,
    num_neighbors=[15, 10],      # sample 15 neighbors at hop 1, 10 at hop 2
    batch_size=2048,             # seed nodes per minibatch
    input_nodes=seed_node_ids,   # the user/item nodes we score this batch
    shuffle=True,
)

for batch in loader:
    # batch is an induced subgraph: batch.x are the sampled node features,
    # batch.edge_index is rewired to local ids, batch.batch_size seeds first.
    emb = model(batch.x, batch.edge_index)     # propagate on the SUBGRAPH only
    seed_emb = emb[:batch.batch_size]          # embeddings for the seed nodes
    # ... compute BPR loss on seed users vs their pos/neg items ...
```

The `num_neighbors=[15, 10]` is the whole game: it caps the layer-1 fan-out at 15 and the layer-2 fan-out at 10, so the largest subgraph any seed can pull is roughly $15 \times 10 = 150$ nodes regardless of whether the seed is a niche item rated by 5 users or a viral item rated by 5 million. That bound is what makes the per-batch compute and memory constant, and it is the difference between a model that fits on one GPU and one that needs a cluster. To replicate PinSage's *random-walk* sampling rather than uniform sampling, you swap `NeighborLoader` for a walk-based sampler that ranks neighbors by visit count and keeps the top ones with importance weights, but the loop shape is identical: sample a bounded subgraph, propagate on it, score the seeds. The trade-off is that sampled propagation is an *approximation* of full-graph propagation, so on small graphs where the full graph fits, full-graph LightGCN is both simpler and slightly more accurate; sampling earns its keep only when the full adjacency no longer fits in memory.

### The evaluation harness

The model is only as trustworthy as the metric you score it with, and Recall@20 and NDCG@20 are easy to compute wrong. Recall@K for a user is the fraction of that user's held-out positive items that appear in the top-K recommended list; NDCG@K additionally rewards putting the hits near the *top* of the list. Here is a leakage-safe harness that ranks against the full catalog and masks out the user's training items so you never get credit for recommending something the model already saw.

```python
import numpy as np

def evaluate(model, train_pos, test_pos, n_items, K=20):
    """train_pos, test_pos: dict user -> set of item ids.
    Ranks all items, masks the user's TRAIN items, scores Recall@K, NDCG@K."""
    eu, ei = model.propagate()                 # final embeddings, no grad
    eu, ei = eu.detach(), ei.detach()
    recalls, ndcgs = [], []
    # ideal DCG for up to K hits: sum of 1/log2(rank+1) for ranks 1..K
    idcg_at = np.cumsum(1.0 / np.log2(np.arange(2, K + 2)))
    for u, held in test_pos.items():
        if not held:
            continue
        scores = eu[u] @ ei.T                  # score against the FULL catalog
        for i in train_pos.get(u, ()):          # mask training items
            scores[i] = -1e9
        topk = scores.topk(K).indices.tolist()  # the top-K recommended items
        hits = [1 if i in held else 0 for i in topk]
        n_hit = sum(hits)
        recalls.append(n_hit / min(len(held), K))
        dcg = sum(h / np.log2(r + 2) for r, h in enumerate(hits))
        idcg = idcg_at[min(len(held), K) - 1]
        ndcgs.append(dcg / idcg if idcg > 0 else 0.0)
    return {"Recall@%d" % K: float(np.mean(recalls)),
            "NDCG@%d" % K: float(np.mean(ndcgs))}
```

Three details make or break this harness. The `scores[i] = -1e9` mask on training items is mandatory; without it, a model that simply re-recommends what the user already touched scores artificially high and you ship a model that is great at recommending things people already have. The `scores @ ei.T` ranks against *every* item, the `mode: full` discipline from above. And the IDCG normalization divides by the best possible DCG given how many held-out positives the user has, so NDCG is bounded in [0, 1] and comparable across users with different numbers of held-out items. Run this exact harness on the same split for BPR-MF, NGCF, and LightGCN and the comparison is apples-to-apples; mismatched harnesses are the single most common reason two papers report different numbers for the same model on the same dataset.

## 9. Results: BPR-MF vs NGCF vs LightGCN, and the layer sweep

Now the numbers that justify the whole exercise. The canonical benchmark is the LightGCN paper's own evaluation on three datasets, of which Gowalla (a location check-in graph, about 29,800 users, 40,900 items, one million interactions) is the cleanest to reproduce. The metrics are Recall@20 and NDCG@20 under full-catalog ranking with a temporal split, exactly the protocol from section 8. The headline result from He et al. (2020) on Gowalla:

| Model | Params | Recall@20 | NDCG@20 | Notes |
| --- | --- | --- | --- | --- |
| BPR-MF | $(M{+}N) \times d$ | 0.1616 | 0.1366 | matrix factorization baseline |
| NGCF (3 layers) | $\sim 6\times$ MF | 0.1570 | 0.1327 | heavy GCN machinery |
| LightGCN (3 layers) | $= $ MF | **0.1830** | **0.1554** | best, and same params as MF |

Read the table carefully, because it contains the post's central surprise. NGCF, the model that introduced high-order propagation to mainstream CF, does *not* beat BPR-MF on this run; its extra weight matrices and nonlinearity add parameters that overfit Gowalla's sparse signal, and it lands slightly *below* the matrix-factorization baseline. LightGCN, with the *same parameter count as MF* (its only parameters are the base embeddings), beats both by a wide margin: roughly +13% Recall@20 and +14% NDCG@20 over BPR-MF, and a larger margin over NGCF. The lesson is stark. The high-order signal is real and valuable, but you capture it by *propagating cleanly*, not by bolting deep-learning machinery onto the propagation. Less model, more accuracy. (Exact figures vary by a percent or two across reimplementations and split seeds; the *ordering*, LightGCN > BPR-MF > or ≈ NGCF, is robust and is what you should expect to reproduce.)

![Matrix of Recall@20 and NDCG@20 on Gowalla for BPR-MF, NGCF, and LightGCN showing LightGCN winning both metrics with a best and simpler verdict](/imgs/blogs/graph-neural-networks-for-recommendation-8.png)

### The layer sweep: watching over-smoothing happen

The second result you must produce yourself before trusting LightGCN in production is the layer sweep. Train LightGCN at $K = 1, 2, 3, 4$ and beyond, and plot Recall@20 against depth. The pattern from the paper, and the one you will reproduce, looks like this on Gowalla:

| Layers $K$ | Recall@20 | What is happening |
| --- | --- | --- |
| 1 | 0.1726 | only 1-hop signal, under-mixed |
| 2 | 0.1786 | 2-hop, strong |
| 3 | 0.1830 | 3-hop, peak |
| 4 | 0.1817 | over-mixing begins |
| 6+ | falls | over-smoothing, embeddings collapse |

The curve rises, peaks around $K = 3$, and then degrades. This is the over-smoothing fixed point from section 5, made visible. The first two or three hops add genuine collaborative signal; beyond that, $\tilde A^k$ is pulling every node toward the same degree-scaled vector and the embeddings lose their discriminating power. If you skip this sweep and default to "deeper is better," you will ship an over-smoothed model and wonder why a 6-layer GNN underperforms a 3-layer one. Run the sweep, pick the peak, and the answer is almost always 2 or 3.

#### Worked example: over-smoothing as layers grow

Return to the tiny 4-node graph from section 4 and run it forward many hops, tracking how the *direction* of the embeddings changes. With a single shared connected component, the theory says all node embeddings converge to a common direction proportional to $\sqrt{\text{degree}}$. The degrees were $d(u_1) = 2$, $d(u_2) = 1$, $d(i_1) = 2$, $d(i_2) = 1$, so the limiting embedding *ratios* should approach $\sqrt 2 : \sqrt 1 : \sqrt 2 : \sqrt 1 = 1.414 : 1 : 1.414 : 1$. Watch the ratios as we iterate $E^{(k)} = \tilde A^k E^{(0)}$ from our starting values:

At $k = 0$ the values are $(1.0,\ 2.0,\ 3.0,\ 4.0)$, ratios $0.25 : 0.5 : 0.75 : 1.0$, wildly different, full of distinguishing information. At $k = 1$ (computed in section 4) they are $(4.328,\ 2.121,\ 1.914,\ 0.707)$. After many more iterations, normalizing each vector, the values converge toward the ratio $1.414 : 1 : 1.414 : 1$ for $(u_1, u_2, i_1, i_2)$. The crucial observation: at the limit, $u_1$ and $i_1$ become indistinguishable in direction (both $\approx 1.414$), and $u_2$ and $i_2$ become indistinguishable (both $\approx 1$). Two users and two items, collapsed onto a two-valued degree pattern, all *content* about who liked what is gone. The only thing that survives infinite smoothing is degree. That is over-smoothing, derived on four nodes by hand: with too many layers, the embedding stops encoding preferences and starts encoding only popularity. The layer-combination average rescues real models by keeping $E^{(0)}$ in the readout, but the limit shows you why depth past 3 destroys signal.

## 10. A problem-solving narrative: shipping LightGCN, then stress-testing it

Let me walk through the decision the way it actually happens on a team, because the metrics table hides the reasoning. Start with the symptom from the intro: a two-tower retriever has plateaued, and an offline error analysis shows the failures cluster on sparse users, the ones with fewer than ten interactions, who collectively make up a large fraction of sessions but get generic, popularity-flavored recommendations. The hypothesis is that high-order collaborative signal would help these users specifically. That hypothesis is *testable* before you build anything: bucket users by interaction count and compute Recall@20 per bucket for the current model. If the sparse buckets are where the model bleeds, a GNN is plausible; if the model is uniformly mediocre across buckets, the problem is elsewhere (features, loss, label quality) and a GNN will not save you. Always localize the failure before reaching for a heavier model.

Suppose the sparse buckets confirm the hypothesis. The decision is *which* GNN. We have a graph of about 400,000 nodes and 8 million edges, which fits comfortably as a sparse tensor on one GPU, so full-graph LightGCN is the right call: it is the simplest model that captures high-order signal, has the same parameter count as the MF baseline (no risk of blowing the embedding-table memory budget), and trains in a few hours. We deliberately skip NGCF (the results table says it does not reliably beat MF and costs more) and skip PinSage-style sampling (we do not need it below a few million edges, and it would only approximate what the full graph gives exactly). We run the layer sweep, find the peak at $K = 3$, and ship a 3-layer LightGCN with the layer-combination average. Offline, the sparse-user buckets improve the most, exactly as the hypothesis predicted, which is the kind of *mechanistic* confirmation that makes you trust the result rather than just the aggregate number going up.

Now the stress tests, because a result you have not tried to break is a result you do not understand.

**What happens with only implicit feedback?** Everything above already assumes implicit feedback: the edges are clicks or check-ins, not ratings, so there are no negative edges, only observed positives and a vast unobserved region that mixes true negatives with not-yet-seen positives. BPR's pairwise loss handles this by *assuming* observed outranks unobserved, the same missing-not-at-random framing as [implicit feedback models](/blog/machine-learning/recommendation-systems/implicit-feedback-models-als-and-bpr). LightGCN inherits this cleanly; the graph is just the bipartite positive-edge graph. The one trap is that very popular items have enormous degree and thus appear in many users' neighborhoods, so without the degree normalization they would dominate every embedding, the popularity fixed point again. The symmetric normalization is doing real work here.

**What happens at 100 million items?** Full-graph propagation dies; you switch to PinSage-style neighbor sampling (the `NeighborLoader` from section 8), accept that propagation is now an approximation, and you build the offline embedding pipeline that recomputes node embeddings on a schedule. The embedding table itself is now the constraint: 100 million items at $d = 64$ floats is about 26 GB just for item embeddings, which forces sharding or quantization the same way it does for any large embedding model. The GNN does not make this worse than two-tower; it is the same table.

**What happens when negatives are mostly false negatives?** This is the silent killer. If you sample random negatives and a "negative" is actually an item the user would have loved but never saw, BPR penalizes the model for a correct prediction. On a sparse graph with a large catalog, most random items are genuinely irrelevant, so the false-negative rate is low and random negatives are fine. But if you add *hard* negatives (PinSage style) without care, you raise the false-negative rate sharply, because hard negatives are by construction *related* items, and related items are exactly the ones most likely to be unobserved positives. This is why PinSage used a curriculum and chose hard negatives from a specific PageRank band (related but not too related), and why you should monitor whether hard negatives help or hurt on a held-out set rather than assuming they always help.

**What happens when offline Recall@20 rises but online engagement is flat?** The recurring nightmare of the series. The most common GNN-specific cause is that the offline metric rewards predicting *items the user already had a strong path to*, which are often items the user would have found anyway through other surfaces, so the model gets offline credit for recommendations that add no incremental value online. The fix is the same as always: measure online with an A/B test, instrument incremental engagement (did this recommendation surface something the user would not otherwise have seen), and do not let a few points of offline Recall override a flat online result. A GNN is not exempt from the offline-to-online gap; if anything its high-order signal can make it *better* at predicting the obvious, which inflates offline metrics without moving the business.

**What happens when the graph drifts between training and serving?** Because embeddings are precomputed offline, a node whose neighborhood changed since the last refresh is served a stale embedding. For a fast-moving feed this is real train-serve skew: a user who binged a new genre yesterday is still embedded as last week's user until the next refresh. The honest fix is to refresh embeddings frequently for high-activity nodes (incremental re-embedding of the changed subgraph, which is cheap with neighbor sampling) and to fall back to a fresher signal (recent-history features in a two-tower) for the part of the catalog that moves fastest. This is the same train-serve-skew discipline the whole series keeps returning to, applied to a graph.

## 11. Case studies: where GNNs have actually shipped

Four real systems, with their reported numbers, to ground the trade-offs.

**PinSage at Pinterest (Ying et al., 2018).** The first web-scale GNN recommender, trained on a graph of roughly 3 billion nodes and 18 billion edges. Its contributions, random-walk neighbor sampling, importance pooling, MapReduce inference, and the hard-negative curriculum, are the playbook for scaling any graph recommender. Reported impact: large offline hit-rate gains (the importance-pooling ablation alone was on the order of a 46% relative lift in their hit-rate metric), and a deployed A/B test where PinSage-powered related-pin recommendations improved user engagement by roughly 30% over the prior production system. The headline lesson is that the algorithmic idea (graph convolution) and the systems idea (sampling plus MapReduce) are equally essential; neither ships without the other at that scale.

**NGCF (Wang et al., 2019).** The paper that made explicit high-order propagation a mainstream CF technique. It argued, correctly, that the collaborative signal should be encoded in the embedding function via message passing rather than left for matrix factorization to discover implicitly, and it demonstrated consistent gains over MF and earlier neural CF on Gowalla, Amazon-book, and Yelp2018. Its lasting importance is as the model LightGCN dissected; NGCF posed the right question (use the graph) and LightGCN answered the follow-up (but use it cleanly). Reading the two papers back to back is one of the best lessons in recommendation research on the difference between "more model" and "better model."

**LightGCN (He et al., 2020).** The result this post is built around. By ablating away the feature transformation and nonlinearity, it showed that the high-order benefit comes from clean propagation, not from deep-learning machinery, and delivered a model that is both simpler than NGCF (same parameter count as MF) and substantially more accurate (roughly +13–16% Recall@20 over NGCF across their three benchmarks). LightGCN became the default GNN baseline in recommendation and remains the model I reach for first when a graph is warranted.

**Alibaba and UltraGCN.** Alibaba's production graph efforts (including the Aligraph platform and GNN-based recommenders for the Taobao feed) demonstrated graph learning on billion-edge industrial graphs with heterogeneous node types. The most pointed follow-up to LightGCN is **UltraGCN** (Mao et al., 2021), which observed that LightGCN's *infinite-layer* limit corresponds to a specific constraint and then *skipped the message passing entirely*, approximating the converged result with a constraint loss directly on the base embeddings. UltraGCN reports matching or beating LightGCN's accuracy while training roughly an order of magnitude faster, because it never runs the propagation loop at all. The arc is telling: NGCF added machinery, LightGCN removed machinery, UltraGCN removed the propagation step itself and kept only its mathematical consequence. The frontier of GNN recommendation keeps discovering that the graph's *effect* matters more than the graph *operation*, and you can sometimes get the effect more cheaply.

## 12. When a GNN earns its cost, and when it does not

This is the section I wish more papers wrote. A GNN recommender is a real cost: heavier training, a sparse adjacency to build and maintain, a layer sweep to tune, an offline embedding-refresh pipeline. Here is when that cost pays for itself and when it does not.

**Reach for a GNN when:**

- **Your interactions are sparse and the graph is dense with paths.** This is the home-run case. When most users have few interactions but the items they touched are richly co-rated, high-order propagation lets sparse users borrow signal from their 2- and 3-hop neighborhoods. The sparser your direct signal and the more connected your graph, the more a GNN beats matrix factorization.
- **A two-tower or MF model has plateaued and you can see structure in the misses.** If your offline analysis shows the model failing exactly where high-order collaborative signal would help (users who would obviously like an item their taste-twins already love), the graph is where that signal lives.
- **You can afford an offline embedding-refresh pipeline.** Propagation runs offline; if you have the batch infrastructure to recompute embeddings nightly, serving is no harder than two-tower.

**Do not reach for a GNN when:**

- **A two-tower model already hits your target.** This is the most important rule, and it echoes the whole series. If the [two-tower model](/blog/machine-learning/recommendation-systems/the-two-tower-model-for-retrieval) meets your Recall@K and latency budget, do not add a GNN for a marginal offline gain that may not survive online. The graph's high-order benefit is real but often a few points of Recall, and a few offline points routinely evaporate in the offline-to-online gap, the recurring theme of [offline vs online](/blog/machine-learning/recommendation-systems/offline-vs-online-the-two-worlds-of-recsys). Earn the complexity.
- **Cold start dominates.** Pure-id GNNs are as helpless as MF on cold nodes. If new users and items are your main problem, content features in a hybrid or two-tower model buy you far more than message passing.
- **You cannot maintain the pipeline.** A GNN that you cannot retrain and re-embed regularly will rot as the graph drifts. If your team cannot own a nightly embedding pipeline, a simpler model you *can* keep fresh will beat a fancier model you cannot.
- **Your graph is enormous and you are not Pinterest.** Full-graph LightGCN stops fitting around a few million edges. Above that you need PinSage-style sampling and a distributed pipeline, a serious engineering investment. Make sure the lift justifies building that.

A compact decision table to keep on hand when you are choosing a retrieval model:

| Situation | Reach for | Why |
| --- | --- | --- |
| Two-tower hits target | Two-tower | No reason to add graph complexity for marginal offline gain |
| Sparse users, dense graph | LightGCN | High-order signal lets sparse users borrow from neighbors |
| Cold start dominates | Hybrid / content two-tower | Pure-id GNN cannot help a node with no edges |
| Graph under a few M edges | Full-graph LightGCN | Simplest, exact propagation, one GPU |
| Graph at billions of edges | PinSage-style sampling | Only bounded-fan-out sampling fits at that scale |
| Cannot own a refresh pipeline | Simpler, refreshable model | A stale GNN rots faster than a simple model you keep fresh |

The honest summary: LightGCN is the best first GNN to try because it is cheap (same params as MF) and strong, but the *decision* to use any GNN should come from evidence that high-order signal is the bottleneck, not from the fact that GNNs are interesting. They are interesting. That is not a reason to ship one.

## 13. Key takeaways

- **Your interaction log is a bipartite graph.** Users and items are nodes; interactions are edges; the adjacency is $A = \begin{pmatrix} 0 & R \\ R^\top & 0 \end{pmatrix}$. Collaborative filtering is link prediction on this graph.
- **The collaborative signal is high-order.** It lives in 2- and 3-hop paths (Alice to her item to a like-minded user to a new item), not in direct edges. $K$ layers of message passing aggregate the $K$-hop neighborhood; algebraically, layer $k$ is $\tilde A^k E^{(0)}$.
- **LightGCN is the propagation rule $E^{(k+1)} = \tilde A E^{(k)}$ plus a layer-combination average**, where $\tilde A = D^{-1/2} A D^{-1/2}$ is the symmetric normalized adjacency. The only trainable parameters are the base embeddings, the same count as matrix factorization.
- **Dropping the feature transform and nonlinearity is the key result.** CF embeddings are free parameters, not semantic features, so there is nothing to transform; removing $W$ and $\sigma$ makes LightGCN simpler *and* more accurate than NGCF. Less model, more accuracy.
- **Propagation is graph smoothing.** $\tilde A = I - L$, so each layer is a Laplacian-smoothing step, a low-pass filter that pulls neighbors together. The collaborative signal is low-frequency, which is why smoothing helps.
- **Over-smoothing is the fixed point, not a bug.** Too many layers converge every node to a common degree-scaled vector. Peak Recall@20 is almost always at $K = 2$ or $3$; run the layer sweep and pick the peak. Never assume deeper is better.
- **PinSage is how you scale.** Random-walk neighbor sampling, importance pooling, MapReduce inference, and a hard-negative curriculum take GNNs to billions of nodes. The systems ideas are as essential as the algorithm.
- **Serving is two-tower serving.** Propagate offline, push item embeddings into an ANN index, query at request time. A GNN moves cost into training, not serving.
- **Cold start still needs features.** Pure-id GNNs cannot help a node with no edges. For cold start, reach for content features, not deeper graphs.
- **Use the right evaluation.** Temporal split, full-catalog ranking (`mode: full`), no leakage. Sampled metrics can mislead; the LightGCN > BPR-MF > NGCF ordering is what to reproduce on Gowalla.

## 14. Further reading

- He, Deng, Wang, Li, Zhang, Wang, *LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation* (SIGIR 2020). The paper this post is built around; read the ablation section closely.
- Wang, He, Wang, Feng, Chua, *Neural Graph Collaborative Filtering* (SIGIR 2019). The model LightGCN dissected; read it first to understand what was removed.
- Ying, He, Chen, Eksombatchai, Hamilton, Leskovec, *Graph Convolutional Neural Networks for Web-Scale Recommender Systems* (PinSage, KDD 2018). The web-scale playbook: sampling, importance pooling, MapReduce, hard-negative curriculum.
- Hamilton, Ying, Leskovec, *Inductive Representation Learning on Large Graphs* (GraphSAGE, NeurIPS 2017). The sampling foundation under PinSage.
- Kipf and Welling, *Semi-Supervised Classification with Graph Convolutional Networks* (ICLR 2017). The original GCN propagation rule that NGCF imported.
- Mao, Zhu, Xiao, Lu, Wang, He, *UltraGCN: Ultra Simplification of Graph Convolutional Networks for Recommendation* (CIKM 2021). The follow-up that skips message passing entirely and keeps only its mathematical limit.
- RecBole documentation, the `LightGCN` and `NGCF` model pages, for a leakage-safe, config-driven reproduction harness.
- Within this series: [matrix factorization](/blog/machine-learning/recommendation-systems/matrix-factorization-the-workhorse), [implicit feedback models, ALS and BPR](/blog/machine-learning/recommendation-systems/implicit-feedback-models-als-and-bpr), [the two-tower model for retrieval](/blog/machine-learning/recommendation-systems/the-two-tower-model-for-retrieval), the funnel map [what is a recommender system](/blog/machine-learning/recommendation-systems/what-is-a-recommender-system), and the capstone [the recommender systems playbook](/blog/machine-learning/recommendation-systems/the-recommender-systems-playbook).
