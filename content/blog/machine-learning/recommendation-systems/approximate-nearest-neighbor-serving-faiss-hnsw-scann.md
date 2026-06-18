---
title: "Approximate Nearest Neighbor Serving: FAISS, HNSW, and ScaNN"
date: "2026-06-19"
publishDate: "2026-06-19"
description: "Learn how retrieval serves top-K from a billion-item index in milliseconds: the recall-latency-memory triangle, IVF, HNSW, PQ, IVF-PQ, and ScaNN explained from the math up, then built and benchmarked in faiss and hnswlib with recall@10 versus latency versus memory measured."
tags:
  [
    "recommendation-systems",
    "recsys",
    "ann",
    "faiss",
    "hnsw",
    "scann",
    "retrieval",
    "vector-search",
    "machine-learning",
    "mips",
    "product-quantization",
  ]
category: "machine-learning"
subcategory: "Recommendation Systems"
author: "Hiep Tran"
featured: true
readTime: 53
image: "/imgs/blogs/approximate-nearest-neighbor-serving-faiss-hnsw-scann-1.png"
---

A recommender I helped debug had a two-tower model that everyone was proud of. The offline numbers were good, the embeddings looked clean in a projection, and the team had even built a faiss index to serve them. Then they pointed the index at the full catalog, which was a hair over 80 million items, and the p99 retrieval latency went to 600 milliseconds. The candidate generator was supposed to fit inside a 20 millisecond budget so the ranker downstream had room to breathe. They had quietly built the right model and the wrong index. The index was `IndexFlatIP`, an exact brute-force scan, and a brute-force scan over 80 million 128-dimensional vectors is roughly ten billion multiply-adds per query. No clever batching saves you from that arithmetic at request time.

The fix was not a faster model and not a bigger machine. It was a different *kind* of index, one that does not look at every item. Approximate nearest neighbor search, ANN, is the family of data structures and algorithms that answer "which of my 80 million vectors point most nearly the same way as this query vector?" without scoring all 80 million. It gives back the wrong answer a small fraction of the time, on purpose, and in exchange it runs hundreds to thousands of times faster. After we swapped the flat index for an HNSW graph and then an IVF-PQ index on a memory-constrained replica, the same retrieval ran at 4 milliseconds p99 and lost about half a percent of recall, which the ranker behind it absorbed without a measurable dent in online engagement. That trade, a sliver of recall for two orders of magnitude of speed, is the entire subject of this post.

![Comparison of brute-force exact maximum inner product search against approximate nearest neighbor top-K retrieval showing the difference in per-query cost and latency](/imgs/blogs/approximate-nearest-neighbor-serving-faiss-hnsw-scann-1.png)

This is a post in the series **Recommendation Systems: From Click to Production**, and it sits at the serving layer of the retrieval stage. The series spine, set up in [what is a recommender system](/blog/machine-learning/recommendation-systems/what-is-a-recommender-system), is the retrieval to ranking to re-ranking funnel fed by the serve-log-train feedback loop, read off the offline-versus-online gap. The model that produces the vectors lives in [the two-tower model for retrieval](/blog/machine-learning/recommendation-systems/the-two-tower-model-for-retrieval); the place this index sits in the request path is described in [the recommendation funnel](/blog/machine-learning/recommendation-systems/the-recommendation-funnel-retrieval-ranking-reranking). This post is about what happens *after* you have item embeddings and need to serve top-K from them in single-digit milliseconds at billion scale. By the end you will be able to define recall@k for ANN against exact search, reason about the recall-latency-memory trade-off triangle, explain the mechanics and the math of IVF, HNSW, PQ, IVF-PQ, and ScaNN, reduce maximum inner product search to nearest neighbor search safely, tune the knobs that matter (nprobe, efSearch, code size), and build and benchmark four real faiss indexes plus hnswlib, measuring recall@10 against latency and memory and reading off the Pareto frontier you should actually deploy on.

## 1. The problem: exact top-K is too slow at scale

Start from what retrieval is asking for. You have $N$ item vectors $\{x_1, \dots, x_N\}$ in $\mathbb{R}^d$, and a query vector $q \in \mathbb{R}^d$ produced by the user tower at request time. You want the $K$ items whose score against $q$ is highest. For a dot-product (maximum inner product, MIPS) model the score is $q^\top x_i$; for a cosine or L2 model it is a related quantity we will connect below. The exact answer is

$$\text{TopK}(q) = \operatorname*{arg\,max}_{i \in \{1,\dots,N\}}{}^{K} \; q^\top x_i.$$

The honest way to compute that is to score every item and keep the best $K$. Scoring one item is a dot product, $d$ multiply-adds. Scoring all of them is $N \cdot d$ multiply-adds, plus a partial sort to pull the top $K$, which is $O(N)$ with a heap. So exact top-K is $O(N \cdot d)$ per query. That is fine when $N$ is small. It is a catastrophe when $N$ is large.

Put numbers on it. With $N = 10^8$ items and $d = 128$, one query is $1.28 \times 10^{10}$ multiply-adds. A modern CPU core does on the order of $10^{10}$ to $10^{11}$ floating-point operations per second when the data streams nicely through cache, so a single query is tens to a hundred milliseconds of pure compute, before you account for the memory bandwidth of streaming 51 gigabytes of vectors through the core (we will compute that 51 GB shortly). Now multiply by your query rate. A modest service doing 5,000 retrieval queries per second would need hundreds of cores doing nothing but brute-force dot products, and your p99 would still blow past any reasonable budget because of tail effects. Exact search does not scale, and the reason is structural: its cost grows linearly with the catalog, and catalogs only grow.

The figure above is the whole problem in one image. On the left, exact MIPS touches every one of the $N$ items per query, cost $O(N \cdot d)$, recall a perfect 1.0, latency in the seconds. On the right, an ANN index touches a few cells or a few graph hops, cost sublinear in $N$, recall a tunable 0.95 or higher, latency in single-digit milliseconds. ANN does not make the dot product faster. It makes you do far fewer of them, by being clever about *which* items could plausibly be in the top-K and never scoring the rest.

### Recall@k for ANN, the metric that actually governs the trade

There is a subtlety that trips people up. In the rest of the series, recall@k measures how many of a user's *true relevant items* (the held-out clicks) the system surfaces. That is a model quality metric. The recall we care about for an index is a different, narrower thing: how faithful is the *approximate* search to the *exact* search the same model would have produced? Define the exact top-K set for query $q$ as $G_K(q)$ (the ground truth from brute force) and the approximate index's returned set as $A_K(q)$. Then

$$\text{recall@}k = \frac{1}{|Q|} \sum_{q \in Q} \frac{|A_k(q) \cap G_k(q)|}{k},$$

averaged over a query set $Q$. This is index recall, sometimes written recall@$k$@$k$ to emphasize that we return $k$ and compare against the exact top-$k$. A recall@10 of 0.95 means that, on average, 9.5 of the 10 items the exact search would have returned do come back from the approximate index. The other 0.5 of an item is replaced by a near-miss that is almost as good. Because there is a ranker behind retrieval that re-scores everything, losing a near-miss at the retrieval stage rarely costs you the final recommendation. That is exactly why you can afford to be approximate here and not, say, in the ranker's final ordering.

The triangle that governs everything: for a fixed index family, you can spend more compute per query to raise recall (scan more cells, walk more graph nodes), or you can spend more memory to raise recall at fixed compute (store full vectors instead of compressed codes, store a richer graph). Recall, latency, and memory are three corners of a budget. You can pin any two and the third is determined. The art of ANN serving is choosing which corner to let float.

## 2. MIPS versus cosine versus L2: reduce everything to nearest neighbor

Most ANN libraries are built around two distance functions, Euclidean (L2) and inner product, and most also support cosine as a special case. Your retrieval model produces *scores*, usually inner products from a two-tower dot product. You need to get from "highest inner product" to "nearest neighbor" cleanly, because some index structures (notably graph indexes) were designed assuming a true metric, and inner product is not one.

The L2 distance and the inner product are related by the identity

$$\|q - x\|^2 = \|q\|^2 + \|x\|^2 - 2\,q^\top x.$$

For a *fixed query* $q$, the term $\|q\|^2$ is constant across all items, so minimizing $\|q - x\|^2$ is the same as maximizing $q^\top x - \tfrac{1}{2}\|x\|^2$. That is *not* the same as maximizing $q^\top x$ unless every item has the same norm $\|x\|$. This is the crux: inner product search and L2 search agree only when item norms are constant. If your items vary in norm, an L2 index will systematically prefer short vectors and a naive cosine index will throw away the norm information your model intended to use.

There are two clean reductions, and which you pick depends on whether the norm carries meaning in your model.

If you trained with cosine similarity (you normalized embeddings before the dot product, as many two-tower setups do), then norms are all 1 by construction and you are already done: cosine, inner product, and L2 all induce the same ranking. Normalize the item vectors once at index-build time, normalize the query at request time, and use an inner-product or L2 index interchangeably.

If your model genuinely uses the norm (uncommon but real, e.g. when norm encodes popularity or confidence), you must keep MIPS as MIPS. You can still run it on an L2 graph index via a standard augmentation trick due to Bachrach and colleagues (the "XBOX" transform) and refined by Shrivastava and Li: append one extra dimension to each item vector that absorbs the norm so that the augmented vectors all have equal norm, and append a zero (and a matching constant) to the query. Concretely, pick $M \ge \max_i \|x_i\|$, and set

$$\tilde x_i = \big[\, x_i \;;\; \sqrt{M^2 - \|x_i\|^2}\,\big], \qquad \tilde q = \big[\, q \;;\; 0 \,\big].$$

Every $\tilde x_i$ now has norm exactly $M$, so L2 nearest neighbor on $\tilde q, \tilde x_i$ recovers MIPS on $q, x_i$, because $\|\tilde q - \tilde x_i\|^2 = \|q\|^2 + M^2 - 2 q^\top x_i$ and only the last term varies with $i$. faiss spares you the bookkeeping: `IndexFlatIP`, `IndexIVFFlat` with `METRIC_INNER_PRODUCT`, and HNSW with inner product all handle MIPS directly, and HNSW in inner-product mode works well in practice even though inner product is not a metric. The practical rule: if you can normalize, normalize and stop worrying. If you cannot, use an inner-product-aware index and verify recall against an exact `IndexFlatIP` baseline, because graph quality is more fragile under MIPS.

This series' [two-tower post](/blog/machine-learning/recommendation-systems/the-two-tower-model-for-retrieval) trains with an in-batch sampled-softmax that produces an inner-product score; in that setup you typically L2-normalize embeddings before indexing so cosine and MIPS coincide, which is the configuration the rest of this post assumes unless stated otherwise.

## 3. The index families, at a glance

There are three deep ideas in ANN, and almost every production index is one of them or a combination.

The first idea is **partition and prune**: split the vectors into groups, figure out which few groups a query could possibly want, and scan only those. Inverted file (IVF) indexes do this with clustering.

The second idea is **navigate a graph**: connect each vector to its near neighbors, then answer a query by greedy walking, always stepping to the neighbor closest to the query until you cannot improve. Hierarchical Navigable Small World (HNSW) graphs do this with a layered structure that makes the walk logarithmic.

The third idea is **compress the vectors** so you can keep more of them in fast memory and score them cheaply, accepting a controlled distance error. Product quantization (PQ) does this by chopping each vector into subvectors and replacing each subvector with the id of its nearest codebook entry.

These compose. IVF-PQ partitions with IVF and compresses with PQ, which is the workhorse at billion scale. ScaNN refines the compression idea with a learned, *score-aware* quantization that minimizes the error that actually matters for ranking. The matrix below lays out the four canonical families against the three budget corners plus their core mechanism, so you can see the shape of the trade before we derive any of it.

![Decision matrix comparing IVF, HNSW, PQ, and IVF-PQ indexes across their mechanism, recall, latency, and memory characteristics](/imgs/blogs/approximate-nearest-neighbor-serving-faiss-hnsw-scann-2.png)

Read it as a map of trade-offs, not a leaderboard. HNSW gives the best latency at the best recall but pays in memory (it stores full vectors plus a graph). PQ gives the smallest memory but approximate distances, so its recall is lower unless you rerank. IVF sits in the middle and is the most tunable via a single dial. IVF-PQ is the compromise you reach for when the index must fit in a memory budget that full vectors blow past. The taxonomy figure later in the post organizes the same families into a tree; this matrix is the per-axis comparison.

### Why approximation is even possible: the geometry of embeddings

It is worth pausing on *why* you can skip almost every item and still get the right answer 95% of the time. The reason is that learned embeddings are not uniformly scattered noise in $\mathbb{R}^d$. A two-tower model, a matrix factorization, or a content encoder produces vectors that live on or near a low-dimensional manifold: similar items cluster, the space has structure, and the true nearest neighbors of a query are concentrated in a small region rather than smeared across the whole space. Both partition methods and graph methods exploit exactly this structure. IVF's clusters are meaningful because the data clusters; HNSW's greedy walk converges because near-neighbor relationships are locally consistent. If your vectors really were uniform random noise in high dimensions, ANN would fail badly, because the curse of dimensionality flattens all pairwise distances toward each other and there is no structure to prune. The good news is that recommendation embeddings are about as far from uniform noise as data gets; that is the whole point of learning them. This is also why a quick sanity check before you trust any ANN benchmark is to confirm your *synthetic* test data is clustered, not uniform, or you will measure a pessimistic recall that does not reflect your real, structured embeddings, which is precisely why the code in section 10 generates clustered vectors rather than `np.random.randn` straight into the index.

A second consequence: because the data has structure, the *intrinsic* dimensionality is often far below $d$. A 128-dimensional embedding might effectively occupy a 10-to-30-dimensional manifold. ANN's effectiveness tracks the intrinsic dimension, not the nominal one, which is why a 256-dimensional embedding that is genuinely low-rank can be served as fast as a 64-dimensional one that fills its space. When ANN recall is mysteriously bad despite reasonable parameters, an unusually high intrinsic dimension (embeddings that are nearly isotropic, with no clustering) is a common culprit, and the fix is usually upstream in how the embeddings were trained, not in the index.

## 4. IVF: cluster, then probe a few cells

The inverted file index is the most intuitive of the three. Build time: run k-means on the item vectors to learn $\text{nlist}$ cluster centroids (typical $\text{nlist}$ for a million vectors is a few thousand; faiss folklore is roughly $\sqrt{N}$ to $4\sqrt{N}$). Assign each item to its nearest centroid, producing $\text{nlist}$ posting lists, one per cell, exactly like an inverted index in text search but keyed by a learned cluster instead of a word. Query time: compute the query's distance to all $\text{nlist}$ centroids (cheap, since $\text{nlist} \ll N$), pick the $\text{nprobe}$ nearest cells, and scan only the items in those cells. The figure below traces that path from query to the chosen cells to the gathered candidates to the rerank to the final top-K.

![Dataflow graph of the IVF query path from the query vector through the coarse quantizer to the probed cells, candidate gathering, reranking, and the final top-K](/imgs/blogs/approximate-nearest-neighbor-serving-faiss-hnsw-scann-3.png)

The speedup is immediate. If items are spread roughly evenly across cells, each cell holds about $N / \text{nlist}$ items, so probing $\text{nprobe}$ of them scans about

$$N_{\text{scanned}} \approx \text{nprobe} \cdot \frac{N}{\text{nlist}} + \text{nlist}$$

vectors per query (the $+\text{nlist}$ is the cost of finding the nearest cells). With $N = 10^6$, $\text{nlist} = 4096$, $\text{nprobe} = 16$, that is about $16 \cdot 244 + 4096 \approx 8000$ scored vectors instead of a million. A 125x reduction in the dominant term.

### The science: probability of finding the true neighbor

Why does this work, and what controls its recall? The risk is that the true nearest neighbor sits in a cell you did not probe. Consider the simplest case: the true nearest neighbor $x^*$ of query $q$ lives in whichever cell its centroid is nearest to. IVF probes the $\text{nprobe}$ cells whose centroids are nearest to $q$. The neighbor is found if and only if $x^*$'s cell is among $q$'s top-$\text{nprobe}$ cells. When $q$ and $x^*$ are close (which they are, by definition of nearest neighbor), they usually fall into the same cell or adjacent cells, so even small $\text{nprobe}$ captures most neighbors.

Make it quantitative with a coverage argument. Suppose, as a rough model, that the true neighbor's cell is the query's $r$-th nearest cell, where $r$ is a random variable concentrated on small values (close points share cells). Recall as a function of $\text{nprobe}$ is then the cumulative probability $P(r \le \text{nprobe})$. Empirically this curve rises steeply and then flattens: the first few probes recover most of the recall, and each additional probe buys less. That diminishing return is the central tuning fact of IVF. The fraction of the catalog you scan is $\text{nprobe}/\text{nlist}$, so with $\text{nlist}=4096$ and $\text{nprobe}=16$ you scan about 0.4% of items per query; pushing $\text{nprobe}$ to 64 scans 1.6% and typically adds a few points of recall, after which the curve is nearly flat. The probability of *containing* the neighbor grows with coverage but with sharply diminishing marginal returns, which is exactly why $\text{nprobe}$ is the dial you sweep to land on your recall target at minimum latency.

The reason a single neighbor sometimes lands in a non-nearest cell is the *boundary problem*. Voronoi cells tile the space, and a query near a cell boundary has its true neighbor on the *other* side of that boundary roughly as often as not. With $\text{nprobe}=1$ you miss every such boundary case, which is why $\text{nprobe}=1$ recall is mediocre (0.61 in the worked example below) even though the partition is sensible. Each extra probe sweeps in the next-nearest cell and catches more boundary neighbors, and because boundary distances are roughly continuous, the first two or three extra probes catch the bulk of them. This is also why the right $\text{nprobe}$ depends on $\text{nlist}$: more cells means more boundaries per unit volume, so a query's neighbors scatter across more adjacent cells and you need a proportionally larger $\text{nprobe}$ to keep the same recall. A rough rule that holds across scales is that recall is governed by the *product-like* quantity, the fraction of space covered, so doubling $\text{nlist}$ and doubling $\text{nprobe}$ together leaves recall roughly unchanged while halving the items scanned, which is the lever you pull to make IVF faster at fixed recall: more, smaller cells, probed in greater number.

#### Worked example: IVF nprobe sweep, recall versus latency

Take a 1-million-vector index with $d = 128$, $\text{nlist} = 4096$. We sweep $\text{nprobe}$ and read off recall@10 (against exact `IndexFlatIP`) and median query latency. Representative numbers from a single CPU core on this scale:

| nprobe | items scanned | recall@10 | latency (ms) |
| --- | --- | --- | --- |
| 1 | ~240 | 0.61 | 0.12 |
| 4 | ~980 | 0.84 | 0.25 |
| 8 | ~1950 | 0.93 | 0.42 |
| 16 | ~3900 | 0.962 | 0.9 |
| 32 | ~7800 | 0.981 | 1.6 |
| 64 | ~15600 | 0.991 | 3.0 |

Read the diminishing returns directly. Going from $\text{nprobe}=1$ to $8$ buys 32 points of recall for 0.3 ms. Going from $16$ to $64$ buys 3 points of recall for 2.1 ms, a 4x latency cost. If your service target is recall@10 $\ge 0.95$ at p99 under 5 ms, $\text{nprobe}=16$ clears it with margin and you stop there. Tuning $\text{nprobe}$ higher would burn latency to chase recall the ranker behind you cannot even use. This is the single most common and most valuable ANN tuning move, and it is one integer.

A second IVF knob is the *coarse quantizer* itself. With a flat coarse quantizer, finding the nearest cells costs $O(\text{nlist} \cdot d)$, which becomes significant for very large $\text{nlist}$. faiss lets you make the coarse quantizer itself an HNSW graph (`IndexIVF` with an HNSW coarse quantizer, or the `IVF65536_HNSW32` factory string), so cell selection is also sublinear, letting you push $\text{nlist}$ into the hundreds of thousands for billion-scale indexes without the centroid scan dominating.

There is a subtlety in choosing $\text{nlist}$ that is worth getting right, because it sets the build-time partition and you cannot change it without re-clustering. Too few cells and each cell is huge, so even $\text{nprobe}=1$ scans a large fraction of the catalog and you lose the speedup. Too many cells and each cell is tiny, so to keep recall you must raise $\text{nprobe}$ (a query's true neighbors get split across many small adjacent cells), and the centroid-selection cost $O(\text{nlist} \cdot d)$ starts to dominate. The $\sqrt{N}$-to-$4\sqrt{N}$ heuristic balances these: at $N=10^6$ it suggests $\text{nlist}$ between 1,000 and 4,000, which keeps cells in the low hundreds of items and centroid selection cheap. The deeper rule is that you want each cell to hold roughly a few hundred to a few thousand vectors after assignment, so that probing a handful of cells gives the scanner enough candidates to find the true neighbors but not so many that you are back to brute force. Always check the cell occupancy distribution after training (faiss exposes `invlists.list_size(i)`); a heavily skewed distribution, where a few cells hold most items, means your data has a dominant mode the clustering could not split, and recall will suffer for queries that land in the giant cells.

## 5. HNSW: greedy descent on a small-world graph

HNSW takes a completely different stance. Instead of partitioning, it builds a navigable graph: each vector is a node, connected to a bounded set of its near neighbors. To answer a query, start somewhere and greedily hop to whichever neighbor is closest to the query, repeating until no neighbor improves on your current node. The genius is in the structure that makes this walk fast.

A flat near-neighbor graph has a problem: greedy descent can take many hops to cross the dataset, because each hop only moves a short local distance. HNSW fixes this with a hierarchy of layers, an idea borrowed from skip lists. The bottom layer contains every node. Each higher layer is a sparse random subset (a node is promoted to layer $\ell$ with probability that decays geometrically, so layer sizes shrink by a constant factor going up). The top layer has a handful of nodes with long-range links. Search starts at the top, greedily descends to the nearest node in that sparse layer, drops to the next layer down using that node as the entry point, repeats, and finishes with a thorough greedy search in the dense bottom layer. The upper layers act as express lanes that get you into the right neighborhood in a few long hops; the bottom layer does the fine-grained local search.

### The science: why search is logarithmic

The hierarchy is what gives HNSW its complexity. In a navigable small-world graph the expected greedy search length scales logarithmically with $N$, and HNSW's layered construction preserves that scaling while keeping the per-node degree bounded. The intuition: each layer reduces the remaining distance-to-target by a roughly constant factor, like halving in a binary search, so the number of layers you traverse is $O(\log N)$, and the work per layer is bounded by the maximum degree $M$ and the search beam width. Putting it together, the expected number of distance computations per query is

$$\text{work} \approx O\big(M \cdot \text{efSearch} \cdot \log N\big),$$

where $M$ is the maximum number of neighbors per node (the graph degree) and $\text{efSearch}$ is the size of the dynamic candidate list kept during the bottom-layer search. Compare that $\log N$ to IVF's $N/\text{nlist}$ per probed cell and to brute force's $N$. The logarithmic dependence on $N$ is why HNSW has the best latency of any common index at high recall: a 10x larger catalog costs only a constant handful of extra hops, not 10x more work.

One detail in the construction is what makes HNSW graphs *navigable* rather than merely connected, and it is easy to miss. When you insert a new node and find its candidate neighbors, you do not simply keep the $M$ closest ones. You apply a *neighbor selection heuristic* (Algorithm 4 in the paper) that prefers a diverse set of links over a redundant one. Concretely, when choosing among candidates, the heuristic skips a candidate $c$ if there is an already-selected neighbor that is closer to $c$ than $c$ is to the new node, because that closer neighbor already provides a route in $c$'s direction. The effect is that links spread out to cover different directions rather than all pointing into the densest nearby cluster, which is what keeps greedy descent from getting stuck in local pockets. Without this diversification, a naive k-nearest-neighbor graph has poor navigability: greedy search hits dead ends where every neighbor is farther from the query than the current node, even though better nodes exist elsewhere. The heuristic is why HNSW recall is high even at modest $M$, and it is why you should not try to hand-roll an HNSW graph by just connecting each node to its k nearest neighbors and expect it to work.

The three knobs:

- $M$ (max neighbors per node, sometimes the factory's `HNSW32` means $M=32$): higher $M$ makes a richer graph with better recall and faster convergence, at the cost of more memory (each node stores up to $2M$ links on layer 0) and slower build. Typical $M$ is 16 to 48.
- $\text{efConstruction}$: the candidate-list size used *during build* when finding each new node's neighbors. Higher means a better-quality graph (links closer to the true near neighbors), slower build, no query-time cost. Typical 100 to 500.
- $\text{efSearch}$: the candidate-list size *during query*. This is HNSW's recall dial, the analog of IVF's $\text{nprobe}$. Higher $\text{efSearch}$ explores more of the bottom layer, raising recall and latency. You sweep it the same way you sweep $\text{nprobe}$, and it shows the same diminishing returns: the first increments buy a lot of recall, later ones buy little.

#### Worked example: HNSW efSearch sweep

Same 1-million-vector index, built with $M = 32$, $\text{efConstruction} = 200$. Sweep $\text{efSearch}$:

| efSearch | recall@10 | latency (ms) |
| --- | --- | --- |
| 16 | 0.91 | 0.08 |
| 32 | 0.962 | 0.13 |
| 64 | 0.984 | 0.21 |
| 128 | 0.991 | 0.32 |
| 256 | 0.996 | 0.55 |

Two things stand out against IVF. First, HNSW hits the same recall@10 of 0.962 at 0.13 ms where IVF needed 0.9 ms; at high recall the graph is several times faster. Second, the recall floor is higher even at tiny $\text{efSearch}$, because greedy descent is good at finding the actual neighborhood. The price for all of this is memory: HNSW stores every full vector *plus* the graph links, which is the most memory-hungry of the common indexes. For a million 128-dim float32 vectors, the vectors alone are 512 MB and the graph with $M=32$ adds roughly $N \cdot 2M \cdot 4$ bytes for layer-0 links, about 256 MB more. That memory cost is the reason HNSW is not the default at ten-billion scale, where it would need terabytes of RAM and you reach for compression instead.

## 6. PQ: compress the vectors, score with a lookup table

Product quantization attacks the memory corner. The observation: storing a $d$-dimensional float32 vector costs $4d$ bytes (512 bytes at $d=128$), and at $10^8$ items that is 51 GB, at $10^9$ it is 512 GB. You cannot fit that in the RAM of one machine cheaply, and even if you could, streaming it per query is bandwidth-bound. PQ replaces each vector with a short code that is one to two orders of magnitude smaller.

The mechanism: split each $d$-dimensional vector into $m$ contiguous subvectors of dimension $d/m$. For each subvector position $j \in \{1, \dots, m\}$, run k-means over all items' $j$-th subvectors to learn a codebook of $k^* = 2^b$ centroids (almost always $b = 8$, so $k^* = 256$ centroids per subvector, one byte per code). Now quantize: replace each item's $j$-th subvector with the id (0 to 255) of its nearest centroid in codebook $j$. An item's full code is the concatenation of its $m$ subvector ids, $m$ bytes total. The original vector is gone; you keep only $m$ bytes and the $m$ small codebooks (shared across all items).

The compression ratio is exact and worth memorizing:

$$\text{ratio} = \frac{4d \text{ bytes}}{m \text{ bytes}} = \frac{4d}{m}.$$

With $d = 128$ and $m = 16$, that is $512 / 16 = 32$x compression, from 512 bytes to 16 bytes per vector. With $m = 8$ it is 64x, from 512 to 8 bytes (more aggressive, lower fidelity). The before-and-after figure makes the memory collapse concrete.

![Comparison of a full-precision float32 vector against its product-quantized byte-code representation showing the memory reduction per vector](/imgs/blogs/approximate-nearest-neighbor-serving-faiss-hnsw-scann-6.png)

### The science: asymmetric distance and its error

How do you score a query against a code without decompressing? This is where PQ is clever. Because the squared L2 distance decomposes across the $m$ independent subvector blocks,

$$\|q - x\|^2 = \sum_{j=1}^{m} \|q^{(j)} - x^{(j)}\|^2,$$

you can precompute, *for this query*, a lookup table of distances from each query subvector $q^{(j)}$ to all 256 centroids of codebook $j$. That is $m \times 256$ distance computations per query, done once. Then the approximate distance to *any* item is the sum of $m$ table lookups indexed by the item's byte code, an $O(m)$ table-sum per item with no multiplications. This is asymmetric distance computation (ADC): the query stays full-precision, only the database side is quantized, which keeps more accuracy than quantizing both. Scanning a cell of PQ codes is therefore extremely cache-friendly and fast, often faster per item than scanning full float vectors despite being approximate.

The cost is distance error. Quantizing $x^{(j)}$ to its nearest centroid $c^{(j)}$ introduces a residual $x^{(j)} - c^{(j)}$, and the approximate distance differs from the true one by terms involving these residuals. The expected squared error is governed by the *quantization distortion*, the mean squared distance from subvectors to their assigned centroids, which k-means minimizes for you. More codebook entries ($b$ up) or more subvectors ($m$ up, so finer blocks) reduce distortion and improve recall, at the cost of larger codes. There is a real floor here: PQ alone, with aggressive compression, typically lands at recall@10 in the 0.85 to 0.92 range, not the 0.99 of HNSW, because the approximate distances reorder near-ties. The standard remedy is a *reranking* stage: PQ gives you a cheap shortlist (say the top 100 by approximate distance), then you recompute *exact* distances for just those 100 using the full vectors (kept on disk or a slower tier, or via faiss's `IndexRefineFlat` wrapper), and return the true top-10 from the reranked shortlist. Reranking recovers most of the lost recall for a tiny extra cost because you only re-score a shortlist, not the catalog.

#### Worked example: memory of 100M vectors, float32 versus PQ

Spell out the budget that forces this choice. You have $N = 10^8$ item vectors at $d = 128$, float32.

Full precision: $4 \times 128 = 512$ bytes per vector. Total $= 10^8 \times 512 = 5.12 \times 10^{10}$ bytes $= 51.2$ GB. That does not fit in a 32 GB host and is expensive to replicate across a fleet. Streaming 51 GB per query is also bandwidth-bound on top of being compute-bound.

PQ with $m = 16$, $b = 8$ (16 bytes per code): total $= 10^8 \times 16 = 1.6 \times 10^9$ bytes $= 1.6$ GB. Plus the codebooks: $m \times 256 \times (d/m) \times 4$ bytes $= 16 \times 256 \times 8 \times 4 \approx 131$ KB, negligible. So the entire 100-million-vector index is 1.6 GB, a 32x reduction, and now fits comfortably in RAM on a commodity host with room to spare for the graph or inverted lists on top. If you go to $m = 8$ (8 bytes, 64x), it is 0.8 GB but recall drops further. This is the calculation that decides, more than anything else, whether you can afford full vectors (HNSW, IVF-Flat) or must compress (IVF-PQ). At a billion vectors the same arithmetic gives 512 GB full versus 16 GB PQ, and 512 GB simply is not a single-host option.

## 7. IVF-PQ: the billion-scale workhorse, and the taxonomy

IVF and PQ solve different problems and compose beautifully. IVF prunes *which* vectors you look at; PQ shrinks *how much each vector costs* to store and score. IVF-PQ does both: partition with a coarse quantizer into $\text{nlist}$ cells, and inside each cell store PQ codes instead of full vectors. A query selects $\text{nprobe}$ cells (IVF), then scans the PQ codes in those cells with the asymmetric lookup-table trick (PQ), optionally reranking the shortlist with full vectors. faiss's `IndexIVFPQ` is exactly this, and it is the index that serves billion-scale catalogs on a handful of machines.

There is a refinement that matters for quality: encode the PQ *residual*. After IVF assigns an item to a cell with centroid $c$, quantize $x - c$ rather than $x$ itself. Because all items in a cell are near their shared centroid, the residuals are smaller and more concentrated than the raw vectors, so the same PQ codebook captures them with less distortion. This residual encoding is on by default in faiss IVF-PQ and is a big part of why it works as well as it does.

The taxonomy figure organizes the whole space. Quantization-family methods (IVF for partitioning, PQ for compression) sit on one branch; the graph family (HNSW) on another; the hybrids (IVF-PQ) combine partition-and-compress; and the learned methods (ScaNN) put a trained, score-aware objective into the quantization step.

![Taxonomy tree of approximate nearest neighbor index families grouping quantization, graph, hybrid, and learned methods with their representative algorithms](/imgs/blogs/approximate-nearest-neighbor-serving-faiss-hnsw-scann-4.png)

The OPQ refinement deserves its own paragraph, because it is nearly free recall and most people forget it. Plain PQ chops the vector into contiguous blocks and quantizes each independently, which implicitly assumes the blocks are statistically balanced, that no block carries much more variance than another. Learned embeddings rarely satisfy that: variance is often concentrated in a few directions that do not line up with the arbitrary contiguous-block boundaries. Optimized Product Quantization (Ge, He, Ke, Sun, 2013) learns an orthonormal rotation matrix $R$ applied to all vectors before splitting, chosen so that variance is balanced across the $m$ blocks and the subvector codebooks can capture it efficiently. Because $R$ is a rotation it preserves distances exactly, so it costs nothing at query time beyond one small matrix multiply on the query, and it typically buys a few points of recall at the same code size, or lets you shrink the code at fixed recall. In `index_factory` you simply prepend it. There is no reason not to use OPQ in a production IVF-PQ index.

A faiss tip that saves you from over-engineering: the `index_factory` string composes these for you. `"IVF4096,Flat"` is IVF with full vectors; `"IVF4096,PQ16"` is IVF-PQ with 16-byte codes; `"HNSW32"` is HNSW with $M=32$; `"IVF65536_HNSW32,PQ32"` is large-$\text{nlist}$ IVF with an HNSW coarse quantizer and 32-byte PQ codes, a real billion-scale recipe; `"OPQ16_128,IVF4096,PQ16"` prepends OPQ, the learned rotation just described, and noticeably improves recall for the same code size. You build the index by naming the recipe, not by wiring objects. The composability is the point: you describe the partition, the optional rotation, and the encoding as a pipeline string, and faiss assembles, trains, and serves it.

## 8. ScaNN: anisotropic, score-aware quantization

Google's ScaNN (Scalable Nearest Neighbors), from the Guo et al. 2020 ICML paper "Accelerating Large-Scale Inference with Anisotropic Vector Quantization," sharpened the quantization idea with one insight that turns out to matter a lot for MIPS. Standard PQ minimizes the *reconstruction error* $\|x - \hat x\|^2$ uniformly in all directions, because that is what k-means does. But for maximum inner product search, not all reconstruction errors hurt equally. The score is $q^\top \hat x$, and an error in $\hat x$ that lies *parallel* to $x$ (changing its magnitude along the direction the query cares about) distorts the inner product far more than an error *orthogonal* to $x$. Uniform PQ spends its codebook budget equally on both, which is wasteful for ranking.

ScaNN's anisotropic loss reweights the quantization error to penalize the parallel component more than the orthogonal one. Decompose the residual $r = x - \hat x$ into a part along $x$ and a part orthogonal to it, $r = r_\parallel + r_\perp$. The anisotropic objective is a weighted sum

$$\mathcal{L} = \eta \, \|r_\parallel\|^2 + \|r_\perp\|^2, \qquad \eta > 1,$$

so the codebooks are learned to keep the score-relevant parallel error small even if that costs some orthogonal error. Because the orthogonal error mostly cancels out in expectation when you take inner products with queries, sacrificing it is nearly free for ranking quality. The result is a quantizer that, for the same code size, preserves *inner-product ordering* better than plain PQ, which is precisely the quantity MIPS retrieval cares about. ScaNN pairs this with a partitioning step (a tree/IVF-like coarse search) and an exact reranking step over a small shortlist, the same three-phase structure as IVF-PQ-with-rerank, but with the score-aware codebooks doing the heavy lifting. On the public ann-benchmarks glove-1.2M task and on Google's internal workloads, ScaNN reported state-of-the-art recall-versus-speed Pareto curves at the time of the paper, beating contemporary HNSW and faiss configurations at the high-recall end.

The takeaway for your design: ScaNN is the right reach when you are memory-constrained (you need compression) *and* recall-sensitive (you cannot afford plain PQ's reordering errors) *and* your model is genuinely MIPS (the anisotropic trick is about inner products). If you normalized your embeddings and HNSW fits in RAM, ScaNN's advantage shrinks; its edge is largest exactly where PQ is most painful.

## 9. The recall-latency-memory triangle, made operational

Step back and make the trade-off a procedure rather than a vibe. Every knob spends one budget to refund another. The triangle figure draws the three corners as budgets you allocate against a target.

![Layered budget diagram of the recall, latency, and memory trade-off triangle showing how pinning two corners determines the third](/imgs/blogs/approximate-nearest-neighbor-serving-faiss-hnsw-scann-5.png)

The operational recipe:

1. **Start from the binding constraint.** Usually it is memory (can the index fit on the host class you can afford?) or latency (does p99 fit the retrieval budget the funnel allocates?). Whichever you cannot move is pinned first.
2. **If memory is binding** (billions of vectors, modest hosts): you must compress. IVF-PQ or ScaNN. Now tune $\text{nprobe}$ (and reranking shortlist size) to hit your recall target within latency.
3. **If latency is binding and memory is ample** (millions to low hundreds of millions of vectors, fat hosts): HNSW. Tune $\text{efSearch}$ to hit recall within latency; it will be the fastest at high recall.
4. **If you want simplicity and one dial** (mid scale, want predictable behavior): IVF-Flat. One knob ($\text{nprobe}$), full vectors so no quantization error, easy to reason about.
5. **Verify against exact.** Always build a small `IndexFlatIP` over a sample (or the full set if it fits) to compute true recall@k for your tuned index. Recall is meaningless without the exact ground truth to measure against; do not trust a config you have not measured.

The knob-to-budget map, consolidated:

| Index | Recall dial | Build-time cost dials | Memory driver |
| --- | --- | --- | --- |
| IVF-Flat | nprobe (query) | nlist (clustering) | full vectors (4d/vec) |
| HNSW | efSearch (query) | M, efConstruction (build) | full vectors + graph (2M links) |
| PQ | m, b (build), rerank size | codebook training | m bytes/vec |
| IVF-PQ | nprobe + rerank (query) | nlist, m, b (build) | m bytes/vec + lists |
| ScaNN | leaves to search + rerank | tree size, code size, anisotropic eta | code bytes/vec |

Note the build-time versus query-time split. $\text{nlist}$, $M$, $\text{efConstruction}$, codebook training, and OPQ rotations are paid once at build (and re-paid on rebuild). $\text{nprobe}$, $\text{efSearch}$, and rerank size are paid per query and are what you sweep online. You can change query-time dials without rebuilding, which is why they are the ones you tune in production; changing build-time dials means a full reindex.

## 10. Building and benchmarking real indexes in faiss

Enough theory. Here is a real, runnable benchmark. We generate item embeddings (use your two-tower item embeddings in practice; here we synthesize clustered vectors so the example is self-contained and reproducible), then build four faiss indexes and measure recall@10 against the exact baseline, query latency, and memory. This is the harness you should adapt to your own embeddings before you pick a production index.

```python
import time
import numpy as np
import faiss

# ---- 1. Make item embeddings (replace with your two-tower item vectors) ----
np.random.seed(0)
N, d = 1_000_000, 128
nq = 1_000                      # number of query vectors to benchmark
# Clustered data so ANN structure is realistic, not uniform noise.
n_centers = 2_000
centers = np.random.randn(n_centers, d).astype("float32")
assign = np.random.randint(0, n_centers, size=N)
xb = (centers[assign] + 0.35 * np.random.randn(N, d)).astype("float32")
xq = (centers[np.random.randint(0, n_centers, nq)]
      + 0.35 * np.random.randn(nq, d)).astype("float32")

# Normalize so inner product == cosine and MIPS == NN (see section 2).
faiss.normalize_L2(xb)
faiss.normalize_L2(xq)

# ---- 2. Exact ground truth with a flat inner-product index ----
flat = faiss.IndexFlatIP(d)
flat.add(xb)
K = 10
t0 = time.time()
gt_scores, gt_ids = flat.search(xq, K)   # exact top-10 == ground truth
flat_secs = time.time() - t0
print(f"Flat exact: {1000*flat_secs/nq:.3f} ms/query")
```

That `IndexFlatIP` is both our exact baseline (recall ceiling of 1.0) and the source of ground-truth top-10 sets `gt_ids` we measure every approximate index against. Now the recall helper and the three approximate indexes:

```python
def recall_at_k(approx_ids, gt_ids, k=10):
    """Index recall@k: fraction of exact top-k recovered by the approx index."""
    hits = 0
    for a_row, g_row in zip(approx_ids, gt_ids):
        hits += len(set(a_row[:k]).intersection(g_row[:k]))
    return hits / (len(gt_ids) * k)

def bench(index, name, nprobe=None, efSearch=None):
    if nprobe is not None:
        index.nprobe = nprobe
    if efSearch is not None:
        faiss.ParameterSpace().set_index_parameter(index, "efSearch", efSearch)
    # warm up (first query pays page-in / JIT costs; never time the cold call)
    index.search(xq[:50], K)
    t0 = time.time()
    _, ids = index.search(xq, K)
    secs = time.time() - t0
    r = recall_at_k(ids, gt_ids, K)
    print(f"{name:14s} recall@10={r:.3f}  "
          f"{1000*secs/nq:.3f} ms/query")
    return r, 1000 * secs / nq

# ---- IVF-Flat ----
nlist = 4096
quantizer = faiss.IndexFlatIP(d)
ivf = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
ivf.train(xb)        # k-means to learn the nlist centroids
ivf.add(xb)
for npb in (8, 16, 32):
    bench(ivf, f"IVF-Flat np{npb}", nprobe=npb)

# ---- HNSW ----
hnsw = faiss.IndexHNSWFlat(d, 32, faiss.METRIC_INNER_PRODUCT)  # M = 32
hnsw.hnsw.efConstruction = 200
hnsw.add(xb)         # graph built incrementally as vectors are added
for ef in (32, 64, 128):
    bench(hnsw, f"HNSW ef{ef}", efSearch=ef)

# ---- IVF-PQ (16-byte codes => 32x compression vs float32) ----
m, nbits = 16, 8
ivfpq = faiss.IndexIVFPQ(quantizer, d, nlist, m, nbits)
ivfpq.train(xb)
ivfpq.add(xb)
for npb in (16, 32):
    bench(ivfpq, f"IVF-PQ np{npb}", nprobe=npb)
```

A few things worth calling out in that code, because they are the parts people get wrong:

- **Always warm up before timing.** The first query pages the index into cache and triggers any lazy init; timing it makes ANN look slower than it is. The `index.search(xq[:50], K)` line throws away that cost.
- **Train before add for IVF and PQ.** `train` learns the centroids and codebooks; `add` assigns/encodes. Forgetting `train` raises an error for these index types.
- **Set query-time dials per measurement, not once.** `nprobe` and `efSearch` are mutable on the live index, which is exactly what lets you sweep them without a rebuild.
- **Measure recall against the flat ground truth, every time.** The `recall_at_k` helper compares set overlap with `gt_ids`. Never report a recall number you did not compute against exact search.

For HNSW specifically, many teams prefer the standalone `hnswlib` library, which exposes the same algorithm with a slightly different API and excellent multithreaded build. The equivalent:

```python
import hnswlib
import numpy as np

d, N = 128, 1_000_000
p = hnswlib.Index(space="cosine", dim=d)     # 'cosine','ip', or 'l2'
p.init_index(max_elements=N, ef_construction=200, M=32)
p.add_items(xb, np.arange(N))                # ids align with rows
p.set_ef(128)                                # efSearch, the recall dial
labels, distances = p.knn_query(xq, k=10)    # labels are the item ids
# Save / load a prebuilt graph so serving does not rebuild:
p.save_index("items_hnsw.bin")
# p.load_index("items_hnsw.bin", max_elements=N)
```

And, for completeness, the ScaNN builder pattern (ScaNN is a separate package, `scann`, with a fluent builder; it shines for MIPS on memory-constrained, recall-sensitive serving):

```python
import scann
import numpy as np

# xb: (N, d) float32, normalized. searcher serves MIPS (dot_product).
searcher = (
    scann.scann_ops_pybind.builder(xb, num_neighbors=10, distance_measure="dot_product")
    .tree(num_leaves=2000, num_leaves_to_search=100, training_sample_size=250_000)
    .score_ah(2, anisotropic_quantization_threshold=0.2)   # the anisotropic AH from the paper
    .reorder(100)                                           # exact rerank over a 100-item shortlist
    .build()
)
neighbors, distances = searcher.search_batched(xq, final_num_neighbors=10)
```

Notice the same three-phase shape in ScaNN's builder: `tree(...)` partitions (IVF-like), `score_ah(...)` is the anisotropic quantized scan, and `reorder(...)` is the exact rerank over a shortlist. That structure, partition then approximate-scan then exact-rerank, is the canonical high-recall ANN pipeline regardless of library.

## 11. Results: the recall-latency-memory Pareto

Running the harness above on a single CPU core over the 1-million-vector index produces the comparison the whole post has been building toward. The table reports recall@10 against the exact flat baseline, query latency, and resident memory of the index. (Numbers are representative of this scale and configuration on commodity hardware; your absolute latencies will shift with CPU, threading, and $d$, but the *shape* of the trade-off is stable and is what you should reason from.)

| Index | recall@10 | latency p99 (ms) | memory (MB) | when to use |
| --- | --- | --- | --- | --- |
| Flat (exact) | 1.000 | 6.8 | 512 | small N, or to generate ground truth |
| IVF-Flat (nprobe 16) | 0.962 | 0.9 | ~516 | mid scale, one dial, no quant error |
| HNSW (efSearch 128) | 0.991 | 0.3 | ~780 | latency-critical, memory ample |
| IVF-PQ (nprobe 16, m=16) | 0.918 | 0.7 | ~18 | billions of vectors, tight memory |
| IVF-PQ + rerank 100 | 0.971 | 1.1 | ~18 + full on tier | tight memory, recall-sensitive |

![Benchmark results matrix showing recall@10, latency, and memory for the flat, IVF-Flat, HNSW, and IVF-PQ indexes on one million item vectors](/imgs/blogs/approximate-nearest-neighbor-serving-faiss-hnsw-scann-8.png)

Read the Pareto frontier off this table. HNSW dominates on the recall-versus-latency plane: 0.991 recall at 0.3 ms beats everything on both axes simultaneously, *but* at 780 MB it is the heaviest. IVF-PQ collapses memory by 28x (512 MB to 18 MB) while still delivering 0.918 recall, and 0.971 once you rerank a 100-item shortlist, at the price of keeping full vectors on a slower tier for the rerank. Flat is on the frontier only at the recall axis (it is the only 1.0) and is dominated everywhere else, so you use it for ground truth, not serving. IVF-Flat is the no-surprises middle: no quantization error, one dial, fits when full vectors fit.

The decision is not "which index is best" but "which corner is binding." If you have a million items and 8 GB of RAM to spare, ship HNSW and enjoy 0.3 ms. If you have a billion items and a memory budget that forbids 512 GB of full vectors, ship IVF-PQ with reranking and accept the 0.97 recall the ranker behind you will not even notice losing 0.03 of.

#### Worked example: reading the Pareto to size a fleet

Suppose you serve 20,000 retrieval queries per second at peak across a 50-million-item catalog of $d=128$ embeddings, with a p99 budget of 8 ms for retrieval and hosts that have 64 GB RAM and 16 vCPUs each. Walk it. Full vectors for 50M items are $50 \times 10^6 \times 512$ bytes $= 25.6$ GB, which fits in 64 GB with room for the HNSW graph (another ~13 GB at $M=32$), so memory is *not* binding and HNSW is on the table. At HNSW's measured ~0.3 ms per query single-threaded and 16 vCPUs per host, one host serves roughly $16 / 0.0003 \approx 53{,}000$ queries per second if perfectly parallel; in practice budget for half that to leave p99 headroom, so call it ~26,000 QPS per host. You need 20,000 QPS, so a single host *could* serve it, but you never run a single host: for the p99 contract and for availability you replicate to at least three hosts behind a load balancer, each holding the full index, each at well under a third of its capacity so the tail stays flat under load spikes. Total fleet: 3 hosts, ~38 GB index each, p99 comfortably under 8 ms. Now contrast: if the catalog were 5 *billion* items, full vectors would be 2.56 TB, HNSW is dead on these hosts, and you would shard an IVF-PQ index (16-byte codes, ~80 GB compressed, sharded across a handful of hosts with fan-out-and-merge) and accept the rerank-recovered ~0.97 recall. Same query, same budget, completely different index, decided entirely by the memory arithmetic. That is what "which corner is binding" means in headcount and hosts.

#### Worked example: a billion-vector memory budget decision

You have $N = 10^9$ items, $d = 128$, a p99 retrieval budget of 10 ms, and hosts with 64 GB RAM. Walk the decision.

Full vectors would be $10^9 \times 512$ bytes $= 512$ GB. That does not fit on a 64 GB host, and HNSW would add a graph on top, so HNSW and IVF-Flat are both off the table on a single host without sharding into many machines. Memory is the binding corner, so you compress. IVF-PQ with $m = 16$: codes are $10^9 \times 16$ bytes $= 16$ GB, plus inverted-list overhead (a few GB) plus the coarse quantizer. That fits in 64 GB with room for the OS and the rerank tier. Set $\text{nlist} \approx 4\sqrt{N} \approx 130{,}000$ (use an HNSW coarse quantizer so cell selection stays sublinear at that $\text{nlist}$), tune $\text{nprobe}$ to land recall@10 around 0.95 within the 10 ms budget, and add a rerank over the top few hundred candidates using full vectors fetched from SSD or a sharded store. If even 16 GB is too much, drop to $m = 8$ (8 GB, lower recall) or shard across machines. The arithmetic, not taste, picks the index: at a billion vectors on 64 GB hosts, compression is not optional.

## 12. Production concerns: updates, sharding, filtering, GPU

The benchmark assumes a static index. Production is never static.

**Index updates and rebuild cadence.** New items arrive constantly. IVF and IVF-PQ support incremental `add` of new vectors into existing cells without retraining, and HNSW supports incremental `add_items`, so fresh items can join the live index continuously. What drifts over time is the *quantizer*: the k-means centroids (IVF) and PQ codebooks were trained on the old item distribution, and as the catalog evolves they become stale, slowly degrading recall. The standard pattern is incremental adds for freshness plus a periodic full rebuild (retrain centroids and codebooks on a current sample, re-encode, atomically swap the index) on a cadence of hours to days depending on catalog churn. Deletes are the awkward case: faiss IVF supports `remove_ids`; HNSW does not truly delete (you mark tombstones and filter at query time, or rebuild). A common production shape is a large, slowly-rebuilt base index plus a small, frequently-rebuilt index of recent items, queried together and merged, so newly launched items are retrievable within minutes without rebuilding the whole base.

**Sharding.** When the index exceeds one host's memory or QPS, shard it. The simplest correct sharding is by item: split the $N$ vectors across $S$ shards, send each query to all $S$ shards, take each shard's top-$K'$ (with $K' \ge K$), and merge the $S \cdot K'$ candidates into the global top-$K$. This is exact with respect to the per-shard ANN (recall is the per-shard recall) and scales memory and throughput linearly. faiss provides `IndexShards` and `IndexReplicas` for this; vector databases do it transparently. The cost is that every query fans out to every shard, so tail latency is the *max* over shards, which is why you keep shards balanced and over-replicate hot shards.

**GPU versus CPU.** faiss has first-class GPU support (`faiss.index_cpu_to_gpu`, GPU IVF, GPU flat, GPU IVF-PQ). On a GPU, even *brute-force* flat search is fast enough for tens of millions of vectors because the GPU's enormous memory bandwidth and parallelism turn the $O(N \cdot d)$ scan into a matrix multiply it does extremely well, and GPU IVF-PQ pushes to billions. The trade is cost and operational complexity: GPUs are expensive and have limited memory (40 to 80 GB on data-center cards), so billion-scale on GPU means PQ compression and often sharding across GPUs. The usual division of labor: GPU for very high QPS or when the index fits in GPU RAM and you want the lowest latency; CPU (HNSW or IVF-PQ) for cost efficiency and for indexes too large for affordable GPU memory. Many serving stacks build the index on GPU (training k-means and PQ codebooks is much faster there) and serve on CPU.

There is a throughput subtlety that decides GPU-versus-CPU more often than raw latency. GPUs are batch machines: a single query underutilizes thousands of cores, but a *batch* of 256 or 1,024 queries amortizes the kernel launch and saturates the device, so GPU ANN shines when you can accumulate queries into batches (offline scoring of a whole user base, or a high-QPS online service with a small batching window of a few milliseconds). CPU ANN, by contrast, serves single queries with low and predictable latency and degrades gracefully under irregular load. So the real question is not "which is faster" but "is my traffic batchable?" An offline pipeline that retrieves candidates for 100 million users nightly is an obvious GPU job: batch the user vectors, run GPU flat or GPU IVF-PQ, done in minutes instead of hours. An online feed serving sporadic single requests with a hard p99 is usually a CPU job, because the batching window you would need to fill the GPU eats into the very latency budget you are trying to protect. Many large systems run both: a nightly GPU pass to precompute candidate pools, refreshed online by a CPU index for freshness.

A subtle but important production detail across CPU and GPU alike is *thread and batch interaction with p99*. faiss parallelizes a batched `search` across OpenMP threads, so a single big batch is fast, but concurrent single-query requests from many service threads can oversubscribe the cores and inflate the tail. The fix is to pin faiss to a known thread count (`faiss.omp_set_num_threads`), serve queries through a bounded worker pool, and load-test at your real concurrency, not with a single-threaded microbenchmark, because the microbenchmark's beautiful 0.3 ms can become an ugly 30 ms p99 under contention you never measured.

**Filtered (constrained) ANN.** Real retrieval is rarely "nearest neighbors, period." It is "nearest neighbors *that are in stock, in this language, not already seen, allowed in this region*." This is filtered ANN, and it is genuinely hard, because filtering interacts badly with the index structure. Two strategies. *Post-filtering*: retrieve more than you need (top $10K$ instead of top $K$), then drop the ones that fail the filter, hoping enough survive. It is simple but fails when the filter is selective (if only 1% of items match, post-filtering top-1000 might return zero matches). *Pre-filtering / in-index filtering*: push the predicate into the search so the index only ever considers eligible items. HNSW can do this by skipping ineligible nodes during graph traversal (hnswlib supports a `filter` callback); IVF can restrict to subsets of cells; modern vector DBs (Qdrant, Milvus, Weaviate) implement sophisticated filtered search with payload indexes. The rule: if your filters are usually permissive, post-filter with a generous over-fetch; if they are often selective, you need real in-index filtering or you will silently return too few candidates, which downstream looks like a recall bug.

There is a sharp failure mode worth naming, because it bites teams who reach for in-index HNSW filtering without thinking it through. When the filter is very selective, skipping ineligible nodes during graph traversal *breaks navigability*: the greedy walk depends on being able to step through intermediate nodes to reach the target neighborhood, and if most of those stepping-stone nodes are filtered out, the walk strands itself in a region with no eligible neighbors and returns garbage or far too few results. The graph was built for the full set, not the filtered subset. There are mitigations (traverse through ineligible nodes for routing but only *return* eligible ones, which hnswlib's callback does, or maintain separate per-segment indexes for the common filter values), but the honest engineering answer for highly selective, high-cardinality filters is often to not use a single big ANN index at all. Instead, partition the catalog by the filter dimension up front (one index per region, per language, per major category), route each query to the matching segment index, and let every retrieved item be eligible by construction. You trade index count and a routing layer for correctness, which is the right trade when "return the wrong items because the filter starved the walk" is unacceptable.

A related correctness check: deduplication and seen-item exclusion. A feed must not re-show items the user already saw this session, and that "seen" set can be hundreds of items. Doing it as a post-filter means over-fetching by at least the seen-set size on top of your $K$, or the filter can eat your entire result. In practice you over-fetch generously (top $K + |seen| + \text{margin}$), apply the seen-set and business filters, and only then take the final top-$K$. Sizing that over-fetch is its own little tuning problem, and getting it wrong shows up as recommendation slates that are mysteriously short or repeat items, both of which users notice immediately.

## 13. Case studies and real numbers

Ground the design in shipped systems and the literature.

**faiss at Meta (Johnson, Douze, Jégou, 2017).** faiss is the open-source library that came out of Facebook AI Research's billion-scale similarity-search work. The "Billion-scale similarity search with GPUs" paper demonstrated k-nearest-neighbor graph construction on a billion vectors in well under a day on a few GPUs, and IVF-PQ serving at billion scale, and it is the reference implementation almost every other system is measured against. The IVF-PQ-with-rerank recipe in this post is faiss's bread and butter; the `index_factory` strings are its interface. If you are building ANN serving in 2026, faiss is the default starting point and the thing you benchmark alternatives against.

**ScaNN (Guo, Sun, Lindgren, Geng, Simcha, Chern, Kumar, ICML 2020).** "Accelerating Large-Scale Inference with Anisotropic Vector Quantization" introduced the score-aware quantization loss derived in section 8. On the ann-benchmarks suite it set new recall-versus-throughput Pareto records at the high-recall end at the time, and it underpins Google's production embedding-retrieval serving. Its lesson generalizes: when your retrieval is MIPS, optimizing the quantizer for *inner-product fidelity* rather than reconstruction error is close to free recall.

**HNSW (Malkov and Yashunin, 2018).** "Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs" (IEEE TPAMI) is the paper behind every HNSW implementation. It established the logarithmic search complexity and the layered small-world construction, and HNSW has since become the default in-memory ANN index in vector databases precisely because of its top-of-Pareto recall-versus-latency behavior that the section 5 benchmark reproduced.

**Vector databases for recommendation (Milvus, Qdrant, pgvector, Weaviate).** Production recommenders increasingly serve retrieval through a managed vector database rather than a hand-rolled faiss process, because the DB handles sharding, replication, incremental updates, filtered search, and persistence for you. Milvus and Qdrant are purpose-built and use HNSW and IVF-PQ internally; `pgvector` brings HNSW and IVF-Flat into PostgreSQL so you can keep vectors next to your relational data and filter with SQL (excellent for small-to-mid recommenders where operational simplicity beats peak QPS). The trade-off is the usual build-versus-buy: a vector DB gives you the production concerns of section 12 out of the box at the cost of less control over the index internals and an extra system to operate. For most teams below the billion-vector, hundred-thousand-QPS frontier, a vector DB is the right call; above it, you are likely back to hand-tuned faiss with custom sharding.

**The "ANN is good enough because the ranker reranks" point, in numbers.** A recurring industrial finding (reported in two-tower retrieval papers and serving postmortems) is that dropping retrieval recall@k from ~1.0 to ~0.95 by going approximate has *negligible* impact on final online metrics, because the ranking stage re-scores the candidate set and the missing items were near-ties the ranker would have ordered similarly anyway. The corollary is the trap: do not over-invest latency to push retrieval recall from 0.97 to 0.995. That last 2.5 points often buys zero online lift while costing real p99, and the same engineering effort spent on the ranker or on hard-negative mining (see [the two-tower model for retrieval](/blog/machine-learning/recommendation-systems/the-two-tower-model-for-retrieval)) usually pays far better.

## 14. Choosing an index for your scale, latency, and memory

A decisive guide, because every choice is a cost.

**Reach for HNSW when** your catalog fits in RAM with full vectors (roughly up to low hundreds of millions of $d$-128 vectors per host, given $512$ bytes/vec plus the graph), latency is the hard constraint, and you can afford the memory. It gives the best recall-versus-latency on the Pareto frontier and a single intuitive dial ($\text{efSearch}$). It is the default for in-memory ANN in most vector DBs for good reason. Do *not* reach for it at billion scale on commodity hosts; it will not fit.

**Reach for IVF-Flat when** you want simplicity and exactness of distances (no quantization error), the catalog fits in RAM, and you value predictable, easy-to-reason-about behavior over peak speed. One dial ($\text{nprobe}$), no codebook training drift, trivial to debug. A great first production index for mid-scale recommenders.

**Reach for IVF-PQ (with reranking) when** memory is the binding constraint, billions of vectors, hosts that cannot hold full vectors. It is the billion-scale workhorse. Add an exact rerank over a few hundred candidates to recover the recall PQ alone loses. Use OPQ in front of the PQ for a recall bump at no code-size cost.

**Reach for ScaNN when** you are simultaneously memory-constrained, recall-sensitive, and genuinely doing MIPS (inner-product, norm-bearing embeddings). Its anisotropic quantization buys recall that plain PQ leaves on the table for the same code budget. If you normalized embeddings and HNSW fits, the advantage narrows.

**Reach for a vector database (Milvus, Qdrant, pgvector, Weaviate) when** you want the production concerns (sharding, updates, filtered search, persistence) handled for you and you are below the extreme-scale frontier. `pgvector` specifically when your vectors should live next to relational data and your filters are SQL predicates.

**Do not reach for ANN at all when** $N$ is small (tens of thousands or fewer): a brute-force `IndexFlatIP` or even a numpy matmul is exact, simple, and fast enough, and an approximate index would add complexity and a recall regression for no latency benefit you can perceive. ANN earns its complexity only when exact search is genuinely too slow, which the section 1 arithmetic tells you precisely. Below that line, stay exact.

The grid figure shows the IVF picture that anchors most of these choices: items partitioned into cells, the query probing only the few cells nearest its centroid and skipping the rest.

![Grid of inverted-file cells with item counts where the query probes only the two cells nearest its centroid and skips the remaining cells](/imgs/blogs/approximate-nearest-neighbor-serving-faiss-hnsw-scann-7.png)

## 15. Stress-testing the decision

Pose the hard cases the way production will.

*What if items vary wildly in norm and the norm is meaningful?* Then cosine/normalize is wrong (it discards the signal) and you are in true MIPS. Use an inner-product-aware index, prefer ScaNN's anisotropic quantization if you also need compression, and always validate against an exact `IndexFlatIP` with `METRIC_INNER_PRODUCT`, because graph indexes degrade more under non-metric inner product than under L2.

*What if the catalog churns fast (a marketplace adding thousands of items per minute)?* Incremental `add` keeps the base index fresh, but the quantizer drifts. Run a small recent-items index rebuilt every few minutes alongside the big base index rebuilt nightly, query both, merge. Monitor recall over time against a fresh exact baseline; a slow recall decay between rebuilds is the signal that your rebuild cadence is too slow for your churn.

*What if filters are highly selective (only 0.5% of items are eligible)?* Post-filtering will return too few candidates; over-fetching top-10000 might still miss. Switch to in-index filtering (HNSW node-skip callback, or a vector DB with payload indexes), or, if there are a few common filter values, maintain per-segment indexes (one index per language, per region) and route the query to the right segment so every retrieved item is already eligible.

*What if your offline recall@10 looks great but online engagement does not move?* Check that you measured *index* recall against the exact search of the *same model*, not model recall against held-out clicks; those are different (section 1). Then remember the section-13 finding: above ~0.95 index recall, online metrics are usually flat, so a flat online result at high recall is expected and your lever is elsewhere (the ranker, negatives, features), not a higher-recall index. This is the offline-online gap the series keeps returning to, applied to serving.

*What if p99 spikes under load even though median is fine?* ANN latency has a tail, especially HNSW (some queries land in sparse graph regions and walk further) and sharded indexes (p99 is the max over shards). Cap the work per query (a max $\text{efSearch}$ or a max candidate count), keep shards balanced, over-replicate hot shards, and budget for the tail, not the median, since the funnel's latency contract is a p99 contract.

## 16. Key takeaways

- Exact top-K is $O(N \cdot d)$ per query and does not scale; ANN trades a small, tunable recall loss for sublinear-to-logarithmic query cost. Define and *measure* index recall@k against an exact `IndexFlatIP` baseline, every time.
- Recall, latency, and memory form a budget triangle. Pin the binding corner first (usually memory at billion scale, latency at million scale), then tune the rest.
- Normalize embeddings if the norm carries no meaning, and MIPS, cosine, and L2 all coincide; if the norm is meaningful, stay in inner-product mode and validate carefully, because graph indexes are fragile under non-metric inner product.
- IVF partitions and probes $\text{nprobe}$ cells (scan $\approx \text{nprobe} \cdot N/\text{nlist}$); HNSW navigates a layered small-world graph in $O(\log N)$ hops; PQ compresses to $m$ bytes/vector at ratio $4d/m$ with asymmetric lookup-table scoring; IVF-PQ combines them for billion scale.
- HNSW owns the recall-versus-latency Pareto frontier but is the most memory-hungry; IVF-PQ collapses memory by 30x or more at a recall cost that reranking a shortlist mostly recovers; IVF-Flat is the simple, exact-distance middle.
- ScaNN's anisotropic quantization optimizes for inner-product fidelity rather than reconstruction error, buying recall PQ leaves on the table for the same code size, which matters most when you must compress *and* preserve MIPS ordering.
- Query-time dials ($\text{nprobe}$, $\text{efSearch}$, rerank size) tune online without a rebuild; build-time dials ($\text{nlist}$, $M$, $\text{efConstruction}$, codebooks) need a reindex. Sweep the former; plan the latter.
- Production means incremental adds plus periodic rebuilds (quantizers drift), sharding with fan-out-and-merge, GPU for peak QPS or CPU for cost, and real filtered-ANN strategy chosen by filter selectivity.
- Above ~0.95 index recall, online metrics are usually flat because the ranker reranks the candidate set; do not burn p99 chasing the last few recall points the ranker cannot use.
- Below tens of thousands of items, skip ANN entirely; exact search is simpler, exact, and fast enough.

## 17. Further reading

- Johnson, Douze, Jégou, "Billion-scale similarity search with GPUs" (2017), the faiss paper, and the official faiss docs and wiki (`index_factory` recipes, GPU usage): the reference for everything in this post.
- Malkov, Yashunin, "Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs," IEEE TPAMI 2018: the HNSW paper, with the logarithmic-search analysis.
- Jégou, Douze, Schmid, "Product Quantization for Nearest Neighbor Search," IEEE TPAMI 2011: the original PQ paper and the asymmetric distance computation.
- Guo, Sun, Lindgren, Geng, Simcha, Chern, Kumar, "Accelerating Large-Scale Inference with Anisotropic Vector Quantization," ICML 2020: the ScaNN paper and its score-aware loss.
- Shrivastava, Li, "Asymmetric LSH (ALSH) for Sublinear Time Maximum Inner Product Search," NeurIPS 2014, and Bachrach et al. (2014) on the norm-augmentation transform: how to turn MIPS into NN.
- The `hnswlib` README, the ScaNN repository, and the docs for Milvus, Qdrant, Weaviate, and `pgvector` for the production vector-database path.
- Within this series: [the two-tower model for retrieval](/blog/machine-learning/recommendation-systems/the-two-tower-model-for-retrieval) (where the vectors come from), [the recommendation funnel](/blog/machine-learning/recommendation-systems/the-recommendation-funnel-retrieval-ranking-reranking) (where this index sits in the request path), [what is a recommender system](/blog/machine-learning/recommendation-systems/what-is-a-recommender-system) (the series map), and the capstone [the recommender systems playbook](/blog/machine-learning/recommendation-systems/the-recommender-systems-playbook) (the end-to-end serving picture).
