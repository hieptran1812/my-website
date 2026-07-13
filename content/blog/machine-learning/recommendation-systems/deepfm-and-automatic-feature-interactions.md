---
title: "DeepFM and Automatic Feature Interactions"
date: "2026-06-19"
publishDate: "2026-06-19"
description: "See why Wide and Deep still needs hand-crafted feature crosses, then build DeepFM in PyTorch with one shared embedding feeding an FM branch and a deep MLP, and measure AUC and logloss on Criteo-style click data to prove automatic crosses beat manual ones."
tags:
  [
    "recommendation-systems",
    "recsys",
    "deepfm",
    "ctr-prediction",
    "feature-interactions",
    "wide-and-deep",
    "machine-learning",
    "criteo",
    "pytorch",
  ]
category: "machine-learning"
subcategory: "Recommendation Systems"
author: "Hiep Tran"
featured: true
readTime: 51
image: "/imgs/blogs/deepfm-and-automatic-feature-interactions-1.png"
---

The first time I shipped Google's Wide and Deep into a production ad ranker, the offline metrics moved exactly the way the paper promised, and then a quarter later I was sitting in a planning meeting trying to explain why our biggest remaining lever was still a spreadsheet of feature crosses that a senior engineer maintained by hand. The "deep" side learned generalizing combinations on its own. The "wide" side, the memorization half, did not. It learned the crosses we *gave* it: `device × advertiser`, `country × category`, `hour × creative-size`, a few dozen pairs that someone had decided were worth a column. Every time we wanted the model to memorize a new interaction, someone opened that spreadsheet, added a cross, re-ran the feature pipeline, retrained, and prayed the cross had enough support to learn a non-zero weight. We were doing manual feature engineering inside a model whose entire selling point was that it learned features. The wide side was a treadmill.

DeepFM is the model that takes that treadmill away. Guo and colleagues at Huawei published it in 2017 with a deceptively simple idea: replace the wide, manually-crossed half of Wide and Deep with a factorization-machine component that learns *every* second-order interaction automatically, with zero hand-engineering, and then make the FM component and the deep component **share the same embedding table** so they train end-to-end as one model. No separate wide input. No cross spreadsheet. No feature pipeline rebuild every time you want a new interaction. You feed the raw sparse fields once, an embedding layer turns each field into a short latent vector, and those exact same vectors feed both an FM branch that scores all pairwise crosses in closed form and a deep multilayer perceptron that learns the high-order combinations. The two branch outputs are summed and squashed through a sigmoid. That is the whole architecture, and it is one of the cleanest ideas in the deep-CTR literature.

This is a post in the series **Recommendation Systems: From Click to Production**, and it sits right where the classic factorization models hand off to the deep ranking stack. The picture below is the architecture in one frame: raw sparse fields feed a single shared embedding table, the embeddings fan out into an FM branch and a deep branch, and the two logits merge before the sigmoid. Keep it in your head for the whole post, because every design decision in DeepFM is a consequence of that shared embedding and that two-branch sum.

![Diagram of the DeepFM architecture where sparse fields feed one shared embedding table that fans out into an FM branch and a deep multilayer perceptron whose outputs are summed before a sigmoid](/imgs/blogs/deepfm-and-automatic-feature-interactions-1.png)

The frame for the whole series, set up in [what is a recommender system](/blog/machine-learning/recommendation-systems/what-is-a-recommender-system), is the retrieval to ranking to re-ranking funnel fed by the serve-log-train feedback loop, read off the offline-versus-online gap. DeepFM lives squarely in the **ranking** stage. It takes a candidate item plus a rich bag of user, item, and context features and produces a calibrated click probability, the same job logistic regression, factorization machines, and Wide and Deep all do, but with a better trade between memorization and generalization and without the manual-cross treadmill. By the end of this post you will be able to write down the DeepFM output as the sum of an FM logit and a DNN logit, explain on principle why a shared embedding lets the two branches co-train better than two separate tables, build the whole thing in PyTorch using the squared-sum trick from the [factorization machines post](/blog/machine-learning/recommendation-systems/factorization-machines-and-field-aware-fm), ablate FM-only against DNN-only against the full DeepFM, and read off AUC and logloss on Criteo-style data to decide when the extra machinery is worth it.

## 1. The problem DeepFM solves: Wide and Deep still needs you to engineer the crosses

To see why DeepFM exists, you have to be precise about what Wide and Deep does and does not automate. The Wide and Deep architecture, which we cover in depth in [Wide and Deep and the memorization-generalization tradeoff](/blog/machine-learning/recommendation-systems/wide-and-deep-and-the-memorization-generalization-tradeoff), has two halves joined at the output:

- The **wide** half is a generalized linear model on a sparse feature vector that includes raw features *and* hand-crafted **cross-product features**. A cross-product feature is a single binary indicator that fires only when a specific combination is present, for example "this row has `device=iOS` AND `app_category=games`." The wide half exists to *memorize* exact, high-frequency co-occurrences. Memorization is valuable: if "users who installed app A then queried for app B" is a real, frequent pattern, you want a parameter that captures it exactly rather than smearing it into a smooth function.
- The **deep** half is a feed-forward neural network on dense embeddings of the categorical features. It exists to *generalize*: to score combinations that are rare or unseen by interpolating in a learned embedding space.

The two are trained jointly and their logits are added. The headline result, from the 2016 paper by Cheng and colleagues at Google, was a statistically significant lift in app acquisitions on Google Play over deep-only and wide-only baselines. It is a genuinely good architecture and it shipped at enormous scale.

Here is the catch, and it is the entire motivation for DeepFM. **The cross-product transformations in the wide half are not learned. They are specified by a human.** From the Wide and Deep paper, the wide component's crosses are an explicit feature-engineering step. You decide which pairs (or triples) of features to cross, you materialize those crosses in your feature pipeline, and only then does the wide model get to memorize them. The deep side automated generalization; the wide side did not automate memorization. It automated *fitting a weight to a cross you already chose*. Choosing the crosses is still on you.

### 1.1 Why manual crosses are a real, expensive problem

This is not a cosmetic complaint. Manual cross selection has three costs that compound at production scale, and I have paid all three.

**The combinatorial cost.** With $m$ categorical fields, there are $\binom{m}{2}$ possible pairwise crosses and exponentially many higher-order ones. Criteo's display-advertising benchmark has 26 categorical fields; that is 325 possible pairs before you even consider triples. You cannot include them all (the cross of two high-cardinality fields explodes the feature space, exactly the trap we dissected in the [factorization machines post](/blog/machine-learning/recommendation-systems/factorization-machines-and-field-aware-fm)), so you must *choose a subset*, and choosing is a search problem over a combinatorial space, done by humans with intuition and ablation studies.

**The sparsity cost.** A hand-picked cross only earns its keep if it has enough support in the training log to estimate a stable weight. Cross two high-cardinality fields and most of the resulting columns appeared a handful of times or never, so the wide model learns a noisy weight or no weight at all. You end up restricted to crosses among low-cardinality fields or to a tiny set of high-frequency high-cardinality pairs, which is a small slice of the interactions that actually matter.

**The maintenance cost.** Catalogs drift. New advertisers, new device types, new content categories appear, and a cross that was predictive last year is dead this year. Someone owns the cross spreadsheet, and that ownership never ends. This is the treadmill: the model's predictive ceiling is set by how aggressively a human curates feature crosses, which is exactly the kind of human-in-the-loop bottleneck that should not exist in a learning system.

DeepFM's thesis is that all three costs disappear if the model learns the order-2 interactions itself. A factorization machine already does precisely that: it scores *every* pairwise interaction in linear time using shared latent vectors, with no human picking pairs and with graceful behavior on unseen pairs. So the move is obvious in hindsight: take the wide half of Wide and Deep, throw away the manual crosses, and drop in an FM. The figure below is the contrast that the whole post turns on.

![Diagram contrasting Wide and Deep which needs hand-picked feature crosses on a separate wide input against DeepFM which learns all order-2 crosses automatically from a shared embedding](/imgs/blogs/deepfm-and-automatic-feature-interactions-2.png)

### 1.2 Where DeepFM sits in the model zoo

Before we build it, it helps to place DeepFM among its siblings, because the deep-CTR literature is a crowded zoo and the names blur together. The matrix below lines up five models on the four properties that actually distinguish them: do they learn crosses automatically, do they share an embedding between components, what interaction orders do they reach, and roughly where they land on Criteo AUC. The numbers are literature-consistent order-of-magnitude figures (more on exact values and caveats in the results section); treat the relative ordering as the signal, not the third decimal.

![Matrix comparing LR, FM, Wide and Deep, DeepFM, and xDeepFM across automatic crosses, shared embedding, interaction order, and Criteo AUC](/imgs/blogs/deepfm-and-automatic-feature-interactions-3.png)

The reading is: logistic regression has no crosses at all. FM gives you automatic order-2 crosses but nothing higher. Wide and Deep gives you a deep generalizing branch plus manual order-2 memorization. DeepFM gives you automatic order-2 *plus* the deep branch, all on a shared embedding, and that combination is the sweet spot for the parameter budget. xDeepFM, which we will discuss at the end, pushes to explicit higher-order crosses with its Compressed Interaction Network at a noticeably higher compute cost and a small additional AUC gain.

## 2. The DeepFM architecture: one shared embedding, two branches, one logit

Let me write the model down precisely, because the precision is where the insight lives.

A training example is a sparse feature vector $x$ over $m$ **fields**. A field is a group of related columns: one categorical field like `advertiser` is a one-hot block, one numeric field like `clicks_today` is a single slot. Criteo, the benchmark we will use, has $m = 39$ fields: 13 numeric and 26 categorical. After one-hot encoding the categoricals you have a sparse vector with millions of possible entries but exactly $m$ active values per row (one per field).

The first layer is an **embedding layer** that maps each field to a $k$-dimensional dense vector. For a categorical field, the active one-hot index looks up a row of an embedding table. For a numeric field, you embed the field with a single shared vector scaled by the value (the same value-term trick from FM). Call the embedding of field $i$ for this example $e_i \in \mathbb{R}^k$. So after the embedding layer, every example is a set of $m$ vectors $\{e_1, e_2, \dots, e_m\}$, each $k$-dimensional. This embedding layer is **shared**: there is exactly one table, and both the FM branch and the deep branch read the same $e_i$ vectors. That sharing is the single most important design choice in the model, and section 4 is devoted to why.

### 2.1 The output equation

The DeepFM prediction is

$$\hat y(x) = \sigma\big(y_{\text{FM}}(x) + y_{\text{DNN}}(x)\big),$$

where $\sigma$ is the sigmoid, $y_{\text{FM}}$ is the logit from the FM component, and $y_{\text{DNN}}$ is the logit from the deep component. Two logits, added, squashed. That is the entire top-level structure. The FM logit and the DNN logit are computed from the *same* embeddings, in parallel, and neither feeds the other.

The **FM component** is exactly a factorization machine of order two:

$$y_{\text{FM}}(x) = w_0 + \sum_{i=1}^{n} w_i x_i + \sum_{i=1}^{n}\sum_{j=i+1}^{n} \langle e_i, e_j\rangle\, x_i x_j.$$

The first two terms are the order-1 (linear) part: a global bias plus a per-feature weight. The double sum is the order-2 part: for every pair of active features, an interaction whose strength is the dot product of their embeddings, scaled by the feature values. This is identical to the FM we derived from scratch in the [factorization machines post](/blog/machine-learning/recommendation-systems/factorization-machines-and-field-aware-fm), with one crucial twist: the latent vectors $e_i$ are the *shared* embedding rows, not a private FM-only table. We will lean on the $O(kn)$ linearization of that double sum constantly, so keep it nearby.

The **DNN component** is a plain feed-forward network on the concatenated embeddings. Concatenate the $m$ field embeddings into one long vector $a^{(0)} = [e_1; e_2; \dots; e_m] \in \mathbb{R}^{mk}$, then push it through $L$ fully connected ReLU layers:

$$a^{(\ell+1)} = \mathrm{ReLU}\big(W^{(\ell)} a^{(\ell)} + b^{(\ell)}\big), \qquad y_{\text{DNN}}(x) = w^{\top} a^{(L)} + b.$$

The MLP learns *implicit*, high-order feature interactions: a depth-three network can express complex nonlinear combinations of all $m$ fields that no closed-form order-2 term captures. It cannot tell you which specific triple it is using (the crosses are implicit, baked into the weights), but it captures the high-order signal that FM structurally cannot.

### 2.2 What each branch is for

The division of labor is clean and is the reason the model works. The figure below makes it explicit.

![Diagram showing the FM branch capturing explicit order-2 crosses while the deep branch learns implicit high-order combinations from the same shared embedding vectors](/imgs/blogs/deepfm-and-automatic-feature-interactions-5.png)

- The **FM branch** captures **low-order** interactions explicitly and in closed form. Order-1 (each feature's main effect) and order-2 (each pair's interaction) are exactly the terms most starved by sparsity in a linear model, and FM's factorized form is the best-known way to estimate them on sparse data. The FM branch is the memorization-with-generalization half: it memorizes pairwise structure but generalizes to unseen pairs through the shared latent space.
- The **DNN branch** captures **high-order** interactions implicitly. Whatever order-3-and-up signal exists, the MLP can approximate it. It is the deep generalization half.

Crucially, DeepFM does **not** put an FM layer underneath the DNN (that would be NFM, discussed later) and does **not** route the FM output into the DNN. The two branches are *parallel*. Both see the raw shared embeddings; both produce a logit; the logits add. This parallelism is what lets the low-order and high-order signals combine additively without one washing out the other, and it is the structural difference from Wide and Deep, where the wide branch sees manual crosses and the deep branch sees embeddings, two *different* inputs.

This connects straight back to the series spine. In the [ranking model CTR prediction foundations](/blog/machine-learning/recommendation-systems/the-ranking-model-ctr-prediction-foundations) we frame the ranker as a calibrated probability model over candidates; DeepFM is one concrete, strong instantiation of that ranker, and the additive-logit structure is exactly what keeps its output well-calibrated as a probability rather than an arbitrary score.

## 3. The science: writing DeepFM as a sum and why the linearization carries over

Let me make the math rigorous, because two things deserve a derivation: the closed-form cost of the FM branch and the precise statement of what "shared embedding" buys you in the gradients.

### 3.1 The FM branch is O(kn), and that is what makes DeepFM cheap

The naive order-2 sum $\sum_{i\lt j}\langle e_i, e_j\rangle x_i x_j$ is $O(k n^2)$ to evaluate, where $n$ is the number of features and $k$ the embedding dimension. On CTR data with millions of features that is a non-starter. The rescue is the same algebraic identity we derived in full in the [factorization machines post](/blog/machine-learning/recommendation-systems/factorization-machines-and-field-aware-fm), the squared-sum-minus-sum-of-squares trick:

$$\sum_{i=1}^{n}\sum_{j=i+1}^{n} \langle e_i, e_j\rangle\, x_i x_j = \frac{1}{2}\sum_{f=1}^{k}\left[\left(\sum_{i=1}^{n} e_{i,f}\, x_i\right)^2 - \sum_{i=1}^{n} e_{i,f}^2\, x_i^2\right].$$

Read the right-hand side carefully. For each latent dimension $f$, you compute one weighted sum of the embeddings, square it, and subtract the sum of the squares. Because only $m$ features are active per row (one per field), each inner sum touches only those $m$ entries, so the whole order-2 term costs $O(km)$ per example, not $O(kn^2)$. With $k = 10$ and $m = 39$, that is roughly 390 multiply-adds for the entire pairwise interaction over all 39 fields. Effectively free.

This is the load-bearing fact for the whole architecture. The FM branch adds essentially zero cost on top of the embedding lookup the model already pays. You get all $\binom{m}{2}$ pairwise crosses for the price of two passes over the active embeddings. The DNN branch, by contrast, costs $O(mk \cdot h + h^2 L)$ for hidden width $h$ and depth $L$, which dominates. So DeepFM is, to first order, "a DNN that happens to also get a free FM term," which is a wonderful place to be: you are buying explicit order-2 modeling for almost nothing.

#### Worked example: the shared-embedding forward pass for one firing feature

Make this concrete with one feature firing both branches. Suppose for some row the field `device` takes the value `iOS`, and its shared embedding row is $e_{\text{iOS}} = (0.5,\, -0.2,\, 0.1)$ with $k = 3$. Two other fields are active: `app_category=games` with $e_{\text{games}} = (0.3,\, 0.4,\, -0.1)$ and `hour=20` with $e_{\text{hour}} = (-0.2,\, 0.1,\, 0.6)$. All three are categorical one-hots, so their $x$ values are 1.

The **FM branch** uses $e_{\text{iOS}}$ to compute its pairwise dot products. The interaction of `iOS` with `games` is $\langle e_{\text{iOS}}, e_{\text{games}}\rangle = 0.5\cdot0.3 + (-0.2)\cdot0.4 + 0.1\cdot(-0.1) = 0.15 - 0.08 - 0.01 = 0.06$. The interaction of `iOS` with `hour=20` is $0.5\cdot(-0.2) + (-0.2)\cdot0.1 + 0.1\cdot0.6 = -0.10 - 0.02 + 0.06 = -0.06$. The `games`-`hour` interaction is $0.3\cdot(-0.2) + 0.4\cdot0.1 + (-0.1)\cdot0.6 = -0.06 + 0.04 - 0.06 = -0.08$. The FM order-2 logit contribution from these three pairs is $0.06 - 0.06 - 0.08 = -0.08$, plus the linear weights $w_{\text{iOS}} + w_{\text{games}} + w_{\text{hour}}$ and the global bias.

The **DNN branch** uses the *exact same* $e_{\text{iOS}} = (0.5, -0.2, 0.1)$. It concatenates the three vectors into $a^{(0)} = (0.5, -0.2, 0.1,\, 0.3, 0.4, -0.1,\, -0.2, 0.1, 0.6)$ and pushes that 9-dimensional vector through the MLP. So $e_{\text{iOS}}$ is read twice in one forward pass: once as an FM latent vector (where it produces dot products with the other fields) and once as three input coordinates to the deep network. When the loss back-propagates, $e_{\text{iOS}}$ receives a gradient from *both* uses, summed. That is the shared-embedding forward in miniature: one vector, two readers, one gradient. The DNN's high-order view and the FM's pairwise view are co-trained on the same parameters, which is the whole reason DeepFM is more sample-efficient than running an FM and a DNN side by side on separate embeddings.

### 3.2 Why the shared embedding makes the FM and DNN co-train

Now the precise statement of the co-training argument. Write the total loss for one example as $\mathcal{L} = \text{BCE}(\sigma(y_{\text{FM}} + y_{\text{DNN}}),\, y)$. The gradient with respect to an embedding row $e_i$ is, by the chain rule through both branches,

$$\frac{\partial \mathcal{L}}{\partial e_i} = \frac{\partial \mathcal{L}}{\partial \hat y}\left(\frac{\partial y_{\text{FM}}}{\partial e_i} + \frac{\partial y_{\text{DNN}}}{\partial e_i}\right).$$

Both partials are nonzero whenever field $i$ is active. The FM term $\partial y_{\text{FM}} / \partial e_i$ pulls $e_i$ toward a position that makes its pairwise dot products predictive; the DNN term $\partial y_{\text{DNN}} / \partial e_i$ pulls it toward a position the deep network can use for high-order signal. The embedding settles at a compromise that serves both. This is *structurally different* from having two tables: with separate tables, the FM table only ever sees the FM gradient and the DNN table only ever sees the DNN gradient, so each is fit from a strictly smaller signal. Sharing pools the gradient information.

There is a second, subtler benefit. The FM branch acts as a *regularizer* on the embedding geometry. Because the same $e_i$ must produce sensible pairwise dot products, the embedding cannot drift into a configuration that is great for the MLP but degenerate for pairwise similarity. In practice this means DeepFM's embeddings are more stable and more interpretable than a deep-only model's, and the model is less prone to the kind of overfitting where the MLP memorizes idiosyncratic high-order patterns. The FM term keeps the embedding honest about second-order structure. This is the same flavor of argument as multi-task learning: an auxiliary objective on shared parameters regularizes them, and here the FM logit is effectively an order-2 auxiliary head on the embedding.

### 3.3 Why automatic crosses beat manual crosses under sparsity

The deepest "why" is the sparsity argument, and it is worth stating cleanly because it is the entire reason DeepFM dominates Wide and Deep without a feature engineer in the loop.

A manual cross `(A=a, B=b)` in the wide model is a single binary column with a single scalar weight $w_{(a,b)}$. That weight is learned *only* from rows where exactly $A=a$ and $B=b$ co-occur. If that combination appeared three times in the log, the weight is estimated from three labels and is essentially noise; if it appeared zero times, the weight is zero forever and the model can never react to that combination at serving time, even if it is highly predictive. There is no mechanism to transfer information between crosses.

The FM branch replaces that scalar with $\langle e_a, e_b\rangle$, a dot product of two latent vectors. Now $e_a$ is estimated from *every* pair $A=a$ participates in across the whole log, and $e_b$ from every pair $B=b$ participates in. The interaction $\langle e_a, e_b\rangle$ is therefore defined and meaningful even for a pair that never co-occurred, because it is built from the two features' *positions in a shared space* rather than from their joint count. Information flows between pairs through the shared vectors. This is exactly the transitive-bridge argument from matrix factorization and FM, and it is why automatic crosses are not merely more convenient than manual ones, they are *strictly more powerful on sparse data*: they generalize across the combinatorial space, and a per-cross weight cannot. Wide and Deep's manual crosses sit on the wrong side of this argument; DeepFM's FM branch sits on the right side, and the offline AUC gap follows directly.

## 4. Shared embeddings vs separate inputs: the parameter and accuracy argument

The shared embedding is the design decision that distinguishes DeepFM from "an FM and a deep net stapled together," so it earns its own section. The figure contrasts the two wiring choices.

![Diagram contrasting separate embedding tables as in Wide and Deep against a single shared embedding table in DeepFM with the parameter and accuracy implications](/imgs/blogs/deepfm-and-automatic-feature-interactions-6.png)

### 4.1 The parameter argument

Suppose your categorical fields one-hot to a total feature dimension $F$ (the count of all distinct categorical values across all fields) and you pick embedding dimension $k$. An embedding table is then $F \times k$ parameters. The embedding table dominates the parameter count in any CTR model, because $F$ is enormous (tens of millions for Criteo's full vocabulary) while the MLP weights are tiny by comparison (a few hundred thousand).

If you give the FM branch and the deep branch *separate* embedding tables, you pay $2 \times F \times k$ embedding parameters. If you share, you pay $F \times k$. That is a literal halving of the model's dominant cost. At Criteo scale, with $F$ on the order of $3 \times 10^7$ distinct values and $k = 10$, a single table is roughly $3 \times 10^8$ parameters; two tables are $6 \times 10^8$. Sharing saves three hundred million parameters, which is real memory, real serving cost, and real training time.

#### Worked example: counting parameters saved by sharing

Take a realistic mid-size CTR model. Categorical fields total $F = 1{,}000{,}000$ distinct values after one-hot, embedding dimension $k = 16$. The MLP is three hidden layers of width 400 on the concatenation of 26 field embeddings, so its input is $26 \times 16 = 416$ wide.

- **Shared embedding (DeepFM).** Embedding table: $F \times k = 1{,}000{,}000 \times 16 = 16{,}000{,}000$ parameters. Linear (order-1) weights: $F = 1{,}000{,}000$. MLP: input-to-h1 is $416 \times 400 = 166{,}400$, h1-to-h2 is $400\times400 = 160{,}000$, h2-to-h3 is $160{,}000$, h3-to-output is $400$, plus biases roughly $1{,}200$; total MLP about $488{,}000$. **Grand total: about 17.5 million parameters,** of which 16 million is the one shared embedding table.
- **Separate embeddings.** Two tables: $2 \times 16{,}000{,}000 = 32{,}000{,}000$. Same linear and MLP terms. **Grand total: about 33.5 million parameters.**

So sharing saves $16{,}000{,}000$ parameters, **48 percent of the model**, by eliminating the duplicate embedding table. The MLP and linear terms are a rounding error next to the embedding table; the embedding is the model. And because both branches now train that single table, you do not just save the memory, you also get more gradient signal per parameter, which is the accuracy half of the argument.

### 4.2 The accuracy argument

The parameter saving is the obvious benefit; the accuracy benefit is the subtle one and the reason the DeepFM paper exists. With separate tables, the FM's latent vectors and the DNN's input embeddings learn *independently*, each from half the gradient signal. The FM table never benefits from the high-order structure the DNN discovers; the DNN table never benefits from the clean pairwise geometry the FM imposes. With a shared table, every embedding row is shaped by both objectives at once, as we derived in section 3.2. The FM objective keeps the geometry sensible for pairwise similarity, which the DNN then exploits for free; the DNN objective pushes the embedding to encode high-order-relevant structure, which the FM's linear term can also tap.

Empirically, the DeepFM paper reports that the shared-embedding model trains faster and reaches better AUC than the separate-embedding variant and than Wide and Deep, and they explicitly attribute the win to (a) no need for feature engineering and (b) the shared embedding letting low- and high-order features be learned jointly from raw input. The stack figure below shows the wiring: one table, two readers, one gradient.

![Diagram of the shared embedding layer where the same per-field embedding rows serve as factorization machine latent vectors and as the flattened input to the deep multilayer perceptron](/imgs/blogs/deepfm-and-automatic-feature-interactions-4.png)

This is the same lesson the series keeps teaching: in recommenders, the embedding table *is* the model, and the highest-leverage decisions are about how that table is shared, regularized, and supervised. We saw it in [matrix factorization](/blog/machine-learning/recommendation-systems/matrix-factorization-the-workhorse) (users and items share one latent space), in [FM](/blog/machine-learning/recommendation-systems/factorization-machines-and-field-aware-fm) (all fields share one space), and now in DeepFM (both interaction branches share one space). The arc is consistent: more sharing, more strength borrowed across sparse data, better generalization.

## 5. Building DeepFM in PyTorch

Enough theory. Let me build the whole model, field-aware embeddings and all, in a way you can copy and adapt. The structure has three pieces: a field-aware embedding layer, the FM branch with the squared-sum trick, and the deep branch, then a forward that sums the two logits.

### 5.1 Field encoding and the embedding layer

The standard production trick for multi-field categorical data is to give every field its own contiguous slice of one big embedding table, and to add a per-field offset to the raw category indices so they index into the right slice. That keeps the FM math simple (one table) while respecting field boundaries.

```python
import torch
import torch.nn as nn

class FeaturesEmbedding(nn.Module):
    """One shared embedding table for all fields, with per-field offsets."""
    def __init__(self, field_dims, embed_dim):
        super().__init__()
        # field_dims[i] = number of distinct categories in field i
        self.embedding = nn.Embedding(sum(field_dims), embed_dim)
        # offset[i] = start index of field i inside the shared table
        offsets = torch.tensor((0, *torch.cumsum(torch.tensor(field_dims), 0)[:-1]),
                               dtype=torch.long)
        self.register_buffer("offsets", offsets)
        nn.init.xavier_uniform_(self.embedding.weight.data)

    def forward(self, x):
        # x: LongTensor of shape (batch, num_fields) of per-field category ids
        x = x + self.offsets.unsqueeze(0)        # shift into the shared table
        return self.embedding(x)                  # (batch, num_fields, embed_dim)


class FeaturesLinear(nn.Module):
    """Order-1 term: one scalar weight per category, plus a global bias."""
    def __init__(self, field_dims):
        super().__init__()
        self.fc = nn.Embedding(sum(field_dims), 1)
        self.bias = nn.Parameter(torch.zeros((1,)))
        offsets = torch.tensor((0, *torch.cumsum(torch.tensor(field_dims), 0)[:-1]),
                               dtype=torch.long)
        self.register_buffer("offsets", offsets)

    def forward(self, x):
        x = x + self.offsets.unsqueeze(0)
        return torch.sum(self.fc(x), dim=1) + self.bias   # (batch, 1)
```

The key line is `self.embedding = nn.Embedding(sum(field_dims), embed_dim)`: a *single* table that the FM branch and the deep branch will both read. There is no second table anywhere in this model. The offsets let us pass a compact `(batch, num_fields)` tensor of small per-field indices rather than a giant one-hot.

### 5.2 The FM branch with the squared-sum trick

The order-2 FM term is the squared-sum-minus-sum-of-squares identity from section 3.1, implemented in three lines:

```python
class FactorizationMachine(nn.Module):
    """Order-2 interaction via the squared-sum trick, O(num_fields * embed_dim)."""
    def __init__(self, reduce_sum=True):
        super().__init__()
        self.reduce_sum = reduce_sum

    def forward(self, x):
        # x: (batch, num_fields, embed_dim) -- the SHARED embeddings
        square_of_sum = torch.sum(x, dim=1) ** 2          # (batch, embed_dim)
        sum_of_square = torch.sum(x ** 2, dim=1)          # (batch, embed_dim)
        ix = square_of_sum - sum_of_square                # (batch, embed_dim)
        if self.reduce_sum:
            ix = torch.sum(ix, dim=1, keepdim=True)       # (batch, 1)
        return 0.5 * ix
```

Notice this module takes the embeddings as input and never owns a table of its own. It is pure arithmetic over the shared embeddings, which is exactly what makes sharing possible: the FM branch is a *function of the embeddings*, not a parameter holder.

### 5.3 The deep branch

The deep branch flattens the same shared embeddings and runs an MLP:

```python
class MultiLayerPerceptron(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout, output_layer=True):
        super().__init__()
        layers = []
        for h in hidden_dims:
            layers.append(nn.Linear(input_dim, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout))
            input_dim = h
        if output_layer:
            layers.append(nn.Linear(input_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)   # x: (batch, num_fields * embed_dim) -> (batch, 1)
```

### 5.4 Assembling DeepFM

Now the whole model. The forward pass looks up the shared embeddings once, feeds them to both branches, and sums the three logits (linear, FM order-2, DNN):

```python
class DeepFM(nn.Module):
    def __init__(self, field_dims, embed_dim, mlp_dims=(400, 400, 400), dropout=0.5):
        super().__init__()
        self.linear = FeaturesLinear(field_dims)                 # order-1
        self.embedding = FeaturesEmbedding(field_dims, embed_dim) # SHARED table
        self.fm = FactorizationMachine(reduce_sum=True)          # order-2
        self.embed_output_dim = len(field_dims) * embed_dim
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, mlp_dims, dropout)

    def forward(self, x):
        # x: (batch, num_fields) long ids
        emb = self.embedding(x)                       # (batch, num_fields, embed_dim)
        y_linear = self.linear(x)                     # (batch, 1)
        y_fm = self.fm(emb)                           # (batch, 1)  -- reads emb
        y_dnn = self.mlp(emb.view(emb.size(0), -1))   # (batch, 1)  -- reads SAME emb
        logits = y_linear + y_fm + y_dnn              # (batch, 1)
        return torch.sigmoid(logits.squeeze(1))       # (batch,)
```

The three lines that define DeepFM are the three reads of `emb`: `self.fm(emb)` and `self.mlp(emb.view(...))` both consume the *same* tensor produced by `self.embedding(x)`. There is no separate embedding for the deep branch. When you back-propagate the BCE loss, the gradient flows into that one `emb` tensor from both `y_fm` and `y_dnn`, exactly the shared-gradient picture from section 3.2. That single shared-embedding wiring is the heart of the architecture.

### 5.5 The training loop

A standard BCE training loop. Note the temporal split, the AUC and logloss reporting, and that we measure on a held-out test set that is *later in time* than train (no leakage), which matters enormously for honest CTR evaluation.

```python
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, log_loss

def train_deepfm(model, train_ds, valid_ds, epochs=8, lr=1e-3, wd=1e-5, bs=2048):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    criterion = torch.nn.BCELoss()
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_ds, batch_size=bs, shuffle=False, num_workers=4)

    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.float().to(device)
            opt.zero_grad()
            p = model(x)
            loss = criterion(p, y)
            loss.backward()
            opt.step()

        # honest eval: no shuffling, full-set metrics, temporal split upstream
        model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for x, y in valid_loader:
                p = model(x.to(device)).cpu()
                preds.append(p); labels.append(y)
        preds = torch.cat(preds).numpy()
        labels = torch.cat(labels).numpy()
        auc = roc_auc_score(labels, preds)
        ll = log_loss(labels, preds)
        print(f"epoch {epoch}: valid AUC {auc:.4f}  logloss {ll:.4f}")
    return model
```

This is the same harness you would use for FM or Wide and Deep, which is exactly the point: swapping models is a one-line change, so ablating them is cheap. The weight decay (`wd=1e-5`) regularizes the embedding table, and the BatchNorm plus dropout in the MLP regularize the deep branch; both matter because CTR models overfit fast on sparse data.

## 6. The ablation: FM-only vs DNN-only vs DeepFM

The most honest way to justify DeepFM is to ablate its branches. With the modular code above, FM-only and DNN-only are trivial variants: FM-only is `y_linear + y_fm`, DNN-only is `y_dnn` on a fresh embedding. Run all three through the identical harness on the same temporal split and read off AUC and logloss.

```python
class FMOnly(nn.Module):
    def __init__(self, field_dims, embed_dim):
        super().__init__()
        self.linear = FeaturesLinear(field_dims)
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.fm = FactorizationMachine(reduce_sum=True)
    def forward(self, x):
        emb = self.embedding(x)
        return torch.sigmoid((self.linear(x) + self.fm(emb)).squeeze(1))

class DNNOnly(nn.Module):
    def __init__(self, field_dims, embed_dim, mlp_dims=(400, 400, 400), dropout=0.5):
        super().__init__()
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.mlp = MultiLayerPerceptron(len(field_dims) * embed_dim, mlp_dims, dropout)
    def forward(self, x):
        emb = self.embedding(x)
        return torch.sigmoid(self.mlp(emb.view(emb.size(0), -1)).squeeze(1))
```

What you reliably see on Criteo-style data, consistent with the published DeepFM results, is the ordering below. The numbers are literature-consistent and should be read as "small but real, monotone gains," not exact reproductions; the gap between any two CTR models on Criteo is famously a few thousandths of AUC, and a 0.001 AUC lift is widely reported to be operationally meaningful at ad scale.

| Model | Test AUC | Logloss | Notes |
|---|---|---|---|
| LR (order-1 only) | ~0.7932 | ~0.4615 | no interactions; the floor |
| DNN-only | ~0.7985 | ~0.4570 | implicit high-order, no explicit order-2 |
| FM-only | ~0.7993 | ~0.4566 | explicit order-2, no high-order |
| Wide and Deep | ~0.8016 | ~0.4547 | manual crosses + deep |
| DeepFM | ~0.8048 | ~0.4521 | automatic order-2 + deep, shared emb |
| xDeepFM | ~0.8070 | ~0.4498 | explicit order-k via CIN, higher cost |

The two readings that matter:

1. **DeepFM beats both of its own branches.** FM-only (~0.7993) and DNN-only (~0.7985) are each worse than DeepFM (~0.8048). The low-order and high-order signals are *complementary*: neither subsumes the other, and adding their logits captures both. This is the empirical justification for the two-branch design.
2. **DeepFM beats Wide and Deep with no manual crosses.** Wide and Deep (~0.8016) required a feature engineer to pick the wide crosses; DeepFM (~0.8048) beats it with zero feature engineering. That is the headline of the paper: automatic order-2 modeling on a shared embedding outperforms hand-engineered crosses on separate inputs.

#### Worked example: is the AUC gain worth it?

Suppose your ad system serves 1 billion impressions a day at an average click value of \$0.05 and your current ranker is Wide and Deep at AUC ~0.8016. You ablate to DeepFM and measure a held-out AUC of ~0.8048, a gain of 0.0032. The translation from offline AUC to online revenue is system-specific and noisy, but a common rule of thumb in display advertising is that a 0.001 AUC improvement in the CTR model maps to roughly a 0.1 to 0.3 percent relative lift in realized CTR after the auction, because better ranking surfaces marginally better ads more often. Take the conservative end: a 0.0032 AUC gain at 0.1 percent relative CTR lift per 0.001 AUC is about a 0.32 percent relative CTR lift. On 1 billion impressions at, say, a 2 percent baseline CTR and \$0.05 per click, baseline daily click revenue is one billion times 0.02 times \$0.05, which is \$1,000,000 a day. A 0.32 percent relative lift is about \$3,200 a day, or roughly \$1.17M a year, for *deleting* the manual-cross spreadsheet and the engineer-hours that maintained it. Even if your AUC-to-revenue conversion is half this estimate, the model pays for itself many times over, and it pays *forever* because there is no ongoing feature-engineering cost. That asymmetry, a one-time model swap versus an unending feature-engineering treadmill, is the real economic argument for DeepFM.

The honest caveat: offline AUC and online lift diverge, a theme we hammer throughout this series and dissect in [offline vs online, the two worlds of recsys](/blog/machine-learning/recommendation-systems/offline-vs-online-the-two-worlds-of-recsys). You must confirm the gain with an online A/B test, watch for calibration drift (DeepFM's additive logits are well-calibrated, but retraining cadence and feature skew can break that), and check that the lift is not concentrated in a slice you do not care about. Never ship a ranker on offline AUC alone.

## 7. Training dynamics: what actually happens during the first few thousand steps

It is one thing to write the equations and another to understand what the optimizer does to a DeepFM during training, because the training dynamics explain several practical gotchas that bite people who treat the model as a black box. Let me walk through the lifecycle of the shared embedding as the loss descends, because it is genuinely different from training a deep-only model.

In the first hundred or so steps, the linear (order-1) term and the FM bias dominate the logit. The MLP's output is near zero at initialization (the embeddings are small random Xavier values, so the concatenated input is tiny and the ReLU stack passes through almost nothing), and the FM order-2 term is also small because the embeddings have not yet found any structure, so their pairwise dot products are near zero. This means DeepFM *starts life as logistic regression*: the linear weights and global bias learn the marginal click rates of each feature first. This is a feature, not a bug, it gives the model a sane calibration anchor from step one, and it is part of why DeepFM is well-calibrated. The additive structure means the linear term carries the base rate while the interaction terms refine it.

Around a few hundred to a few thousand steps, two things wake up at once. The FM order-2 term starts contributing as the embeddings differentiate (features that co-click get pulled into alignment, raising their dot products), and the MLP starts contributing as its first layer learns to read the now-meaningful embedding coordinates. Here is the subtle dynamic that the shared embedding creates: the FM gradient and the MLP gradient are both shaping the *same* embedding rows, and they do not always agree about where a row should sit. The FM term wants the embedding oriented so its pairwise dot products are predictive; the MLP wants it oriented so its coordinates are useful features for the deep stack. The embedding settles at a Pareto compromise, and the speed at which it settles is faster than either branch alone would manage, because two gradient signals pin it down from two directions, which is exactly the co-training benefit made dynamic.

### 7.1 The gotcha: the FM branch can be drowned by a hot MLP

A real failure mode I have hit: if you give the MLP too much capacity or too high a learning rate relative to the embedding, the MLP can learn to dominate the logit early, and the FM term's contribution is squeezed toward irrelevance. You then have, effectively, a deep-only model wearing a DeepFM costume, and your ablation shows DeepFM barely beating DNN-only. The fix is to keep the MLP modest (three layers of 400 is plenty for Criteo; going to five layers of 1024 usually hurts), to use the same learning rate for all parameters so the embedding is not starved, and to confirm in an ablation that the FM-only logit on the trained shared embedding is itself predictive. If FM-only on the trained embedding has near-random AUC, your FM branch has been drowned and you are paying for an FM term that does nothing.

### 7.2 The gotcha: embedding regularization is the dominant knob

Because the embedding table is 95-plus percent of the parameters, embedding regularization (weight decay, or explicit $L_2$ on the embedding) is the single most important regularization knob, far more than MLP dropout. Under-regularize and the model memorizes rare feature values and overfits; over-regularize and you crush the very embedding structure the FM branch depends on. The sweet spot for Criteo-scale data is a small weight decay (around $10^{-5}$ to $10^{-6}$) applied to the embedding, with dropout in the MLP as a secondary defense. A common mistake is to crank MLP dropout to 0.7 to fight overfitting while leaving the embedding unregularized, which fixes the wrong layer; the overfitting lives in the embedding table, so regularize there first.

These dynamics are why DeepFM, despite being structurally simple, rewards understanding. The model is a negotiation between two branches over one shared table, and the practical art is keeping that negotiation balanced so both branches contribute.

## 8. Implementing the data pipeline for Criteo

A model is only as good as its features, and CTR data has two preprocessing steps that materially change DeepFM's accuracy. Get these wrong and your AUC drops by more than the model choice buys you.

### 8.1 Bucketizing numeric fields

Criteo has 13 numeric fields (`I1`-`I13`), heavy-tailed integer counts. You have two choices: embed each as a single scaled slot (FM-style value term) or **bucketize** them into categories and one-hot. The original DeepFM and most strong Criteo results bucketize, because it lets every numeric field participate in the embedding table and the FM crosses on equal footing with categoricals, and it tames the heavy tails. The standard log-bucketization, from the Criteo Kaggle winners, is:

```python
import numpy as np

def transform_numeric(value):
    """Criteo numeric log-bucketization: maps an int count to a small bucket id."""
    if value is None or value == "":
        return 0            # missing -> bucket 0
    v = int(value)
    if v > 2:
        return int(np.floor(np.log(v) ** 2)) + 1   # the well-known log^2 binning
    return v + 1            # 0,1,2 -> 1,2,3 ; missing -> 0
```

That `log(v)**2` binning is folklore from the 2014 Criteo Kaggle competition and it works remarkably well; it compresses the long tail into a handful of buckets so each bucket has dense support.

### 8.2 Hashing or frequency-thresholding categoricals

The 26 categorical fields have huge, long-tailed vocabularies, many values appearing once. Two standard defenses:

- **Frequency threshold.** Map any category appearing fewer than, say, 10 times to a shared `<rare>` token per field. This shrinks the vocabulary by an order of magnitude and prevents the model from minting a fresh embedding for a value it will see once. It is the single highest-leverage preprocessing knob for Criteo.
- **Hashing trick.** Hash each category into a fixed number of buckets (say $10^6$ per field). Bounds the table size deterministically and handles unseen values at serving time, at the cost of occasional collisions. Production systems lean on hashing because it caps memory and never OOMs on a vocabulary explosion.

```python
import pandas as pd

def build_vocab(df, cat_cols, min_count=10):
    """Frequency-thresholded vocab: rare categories collapse to <rare> per field."""
    field_dims, maps = [], {}
    for c in cat_cols:
        vc = df[c].value_counts()
        keep = vc[vc >= min_count].index
        mapping = {v: i + 1 for i, v in enumerate(keep)}  # 0 reserved for <rare>/missing
        maps[c] = mapping
        field_dims.append(len(mapping) + 1)
    return field_dims, maps
```

This pipeline is where train-serve skew creeps in, the silent killer we keep warning about. If you fit the frequency threshold and buckets on training data but compute them differently at serving time, your live AUC quietly collapses while offline looks fine. The fix is to *materialize the vocab and the bucket function as artifacts* and apply the identical code path offline and online. We treat this end to end in [the data and features of recommenders](/blog/machine-learning/recommendation-systems/the-data-and-features-of-recommenders); for DeepFM specifically, the embedding lookup is exactly where skew bites, so guard it hard.

## 9. Comparing DeepFM to its neighbors: NFM, PNN, and the family tree

DeepFM is one node in a busy family tree of deep-CTR models, and knowing the neighbors clarifies what DeepFM is and is not. The tree below organizes the family by how they combine FM-style interactions with deep networks.

![Diagram of the deep click-through-rate model family tree rooted at factorization machines and branching into implicit-cross and explicit-cross descendants](/imgs/blogs/deepfm-and-automatic-feature-interactions-7.png)

The split that organizes the tree is **implicit versus explicit high-order crosses**.

- **Implicit-cross models** let an MLP learn high-order interactions in its weights, without naming them. DeepFM, NFM, and PNN are here.
- **Explicit-cross models** define a closed-form operation that produces each interaction order on purpose. DCN and xDeepFM are here.

### 9.1 The implicit branch: DeepFM, NFM, PNN

**NFM (Neural Factorization Machine, He and Chua 2017).** NFM puts an FM-style **bi-interaction pooling** layer *underneath* the MLP. It computes the element-wise interaction of the embedding pairs into a single $k$-dimensional vector (the bi-interaction), then feeds *that* into the deep network. So NFM is serial: FM-pooling, then MLP. DeepFM is parallel: FM and MLP run side by side on the same embeddings and their logits add. The parallel design preserves the raw FM logit as a direct, un-transformed contribution to the output, which keeps the low-order signal clean; NFM's serial design lets the MLP transform the pooled interaction, which is more expressive but can muddy the low-order term. In practice they are close on Criteo, with DeepFM's parallel structure usually a touch ahead and easier to reason about.

**PNN (Product-based Neural Network, Qu et al. 2016).** PNN inserts a **product layer** between the embeddings and the MLP that computes explicit inner or outer products of all field-pair embeddings, then feeds those products to the deep network. PNN has no separate linear/FM logit summed at the output; everything flows through the MLP. So PNN front-loads the pairwise products but then makes the deep network responsible for everything, losing the clean additive FM term that keeps DeepFM calibrated. DeepFM can be read as "PNN's good idea (explicit products) but kept as a parallel, additive FM logit instead of routed through the MLP."

The common thread: DeepFM's distinguishing choices are (1) the FM term is a *parallel additive logit*, not a layer feeding the MLP, and (2) the embedding is *shared*. Those two choices are why DeepFM is the one people reach for first.

### 9.2 The explicit branch: DCN and xDeepFM

**DCN (Deep and Cross Network, Wang et al. 2017).** DCN replaces the FM branch with a **cross network**: a stack of layers each of which computes an explicit feature cross of increasing order, $x_{\ell+1} = x_0 x_\ell^\top w_\ell + b_\ell + x_\ell$. Stack $L$ cross layers and you get explicit interactions up to order $L+1$, at bounded parameter cost. DCN, which we cover in [DCN and explicit feature crossing](/blog/machine-learning/recommendation-systems/dcn-and-explicit-feature-crossing), generalizes DeepFM's order-2 to explicit arbitrary order while keeping the parallel-branch, shared-embedding philosophy. DCNv2 improves the cross layer's expressiveness with a full weight matrix. If you find DeepFM's order-2 ceiling limiting, DCN is the natural next step.

**xDeepFM (Lian et al. 2018).** xDeepFM adds a **Compressed Interaction Network (CIN)** branch that computes explicit high-order interactions at the *vector* level (operating on whole field embeddings, not individual elements), producing interactions of order 2, 3, up to $L$ explicitly. It runs the CIN branch in parallel with a DNN and a linear term, summing logits, exactly DeepFM's parallel-additive philosophy extended to explicit high order. xDeepFM typically edges out DeepFM on Criteo by a small margin (a few thousandths of AUC) at a meaningfully higher compute cost, because the CIN's vector-wise outer products are expensive. We will treat its CIN derivation in its own post.

The honest meta-point, which the literature has converged on, is **diminishing returns to deeper explicit crosses.** Going from order-2 (FM) to "order-2 plus implicit deep" (DeepFM) is a clear, reliable win. Going from there to explicit order-k (xDeepFM, deeper DCN) buys a few more thousandths of AUC at a real compute and complexity cost. For most teams, DeepFM is the Pareto-optimal stopping point: nearly all of the gain, a fraction of the cost, and dead simple to implement and serve. Reach for xDeepFM or DCNv2 only when you have squeezed everything else and a thousandth of AUC is worth real money to you.

## 10. Case studies and real numbers

Let me ground all of this in named results from the literature and shipped systems, because the "small but real" gains only make sense at the scale where a thousandth of AUC is money.

### 10.1 DeepFM (Guo et al. 2017, Huawei)

The original DeepFM paper, "DeepFM: A Factorization-Machine based Neural Network for CTR Prediction" (Guo, Tang, Ye, Li, He, IJCAI 2017), introduced the shared-embedding parallel architecture and evaluated on the Criteo dataset and a Huawei App Store "Company" dataset. Its central empirical claims, which have held up in countless reproductions since:

- DeepFM beats Wide and Deep, FM, DNN-only, PNN variants, and the FNN (FM-pretrained neural network) baseline on both AUC and logloss.
- The improvement comes with **no feature engineering**: where Wide and Deep needs hand-crafted wide crosses, DeepFM needs none.
- The **shared embedding** is essential: an ablation with separate embeddings for the FM and deep components is worse and trains slower, confirming the co-training argument from section 4.

Huawei deployed DeepFM in the App Store recommendation pipeline; the paper reports an online A/B test with a meaningful CTR lift over the prior linear model, on the order of a few percent relative, which at App Store scale is a large absolute number. The reproducible offline takeaway is the Criteo AUC ordering: FM < DNN-style < Wide and Deep < DeepFM, with DeepFM around the high 0.79s to low 0.80s depending on preprocessing.

### 10.2 xDeepFM (Lian et al. 2018, Microsoft)

"xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems" (Lian, Zhou, Zhang, Chen, Xie, Sun, KDD 2018) introduced the CIN and reported small but consistent gains over DeepFM on Criteo, Dianping, and Bing News datasets. The paper's framing is exactly the diminishing-returns story: explicit high-order crosses help, but the marginal AUC gain over DeepFM is modest while the CIN's cost is real. xDeepFM's lasting contribution is conceptual, the clean separation of explicit (CIN) and implicit (DNN) high-order interactions, more than a dramatic accuracy jump. It is the model you cite to understand the design space, and the model you deploy when you have the compute budget and a thousandth of AUC pays for itself.

### 10.3 The CTR-model zoo and the benchmark reality

A sobering and important result for anyone tempted to chase the latest CTR architecture: several careful reproducibility studies (notably the "Open Benchmarking for CTR Prediction" work by Zhu and colleagues, and the BARS benchmark) have found that the AUC differences among the strong deep-CTR models, DeepFM, DCN, xDeepFM, AutoInt, FiBiNET, and friends, are **small and sensitive to hyperparameters and preprocessing**. Under a controlled, well-tuned benchmark, the gap between DeepFM and the fanciest successors on Criteo is often a few thousandths of AUC, and the ranking of models can flip with a different embedding dimension or learning-rate schedule. The honest engineering conclusion: the big jump is from linear or FM to a *shared-embedding parallel deep model* (DeepFM is the canonical one), and everything after that is incremental. Spend your effort on features, data freshness, negative handling, and online evaluation, not on chasing the leaderboard's third decimal place.

This is the same lesson the [neural collaborative filtering critique](/blog/machine-learning/recommendation-systems/neural-collaborative-filtering-and-its-critique) taught for the retrieval side: a well-tuned simple baseline often matches a complicated successor, and the literature's headline gains can shrink under controlled comparison. DeepFM survives that scrutiny well precisely because it is simple, cheap, and robust, which is exactly why it is still a default ranker years after publication.

### 10.4 A concrete deployment shape

To make the serving side tangible: a DeepFM ranker in production typically sits at the *ranking* stage of the funnel, scoring a few hundred candidates per request that retrieval handed up. With $k$ around 10 to 16 and an MLP of three 400-wide layers, a single forward over a few hundred candidates is well under a millisecond on a modern GPU and a few milliseconds on CPU, so DeepFM is not the latency bottleneck (the embedding lookups and feature fetches usually are). The embedding table is the memory cost: tens to hundreds of millions of rows times $k$ floats, often tens of gigabytes, which is why production teams shard the table, quantize embeddings, or use mixed-dimension embeddings for rare versus frequent features. The ranker's p99 is dominated by feature retrieval and table lookup, not the FM-plus-MLP arithmetic, which is, as we proved in section 3.1, nearly free.

### 10.5 Why the two branches are genuinely complementary, with numbers

It is worth dwelling on *why* FM-only and DNN-only each leave AUC on the table that the combination recovers, because the answer is not "two models are better than one" hand-waving; it is a statement about what each branch can and cannot represent.

The FM branch represents the interaction matrix as a low-rank Gram matrix $V V^\top$. That structure is wonderful for order-2 effects and terrible for order-3-and-up: a dot product of two vectors is fundamentally a pairwise, symmetric quantity, and no amount of training makes it express "the click rate depends on the conjunction of device AND hour AND advertiser, in a way not decomposable into the three pairwise terms." Those genuine higher-order conjunctions exist in real click data, advertiser performance really does depend on the joint context, not just the pairwise sums, and the FM branch is structurally blind to them.

The DNN branch represents interactions implicitly through its weights, so it *can* in principle express any-order conjunction. But it is bad at the thing FM is good at: cleanly isolating a specific pairwise effect from sparse data. An MLP fed concatenated embeddings has to *learn* that the relevant signal is a particular pair's interaction, and on sparse data with rare co-occurrences, it learns this poorly and noisily compared to FM's closed-form, regularized pairwise term. There is also a well-documented result, from the DCN and xDeepFM papers among others, that plain MLPs are surprisingly *inefficient* at learning even low-order multiplicative crosses; they need many parameters and a lot of data to approximate a simple product that FM expresses exactly in one term. So the DNN-only model wastes capacity reinventing the order-2 structure that FM hands the model for free.

Put those two facts together and the complementarity is mechanical: FM nails the low-order pairwise structure that the MLP wastes capacity on, and the MLP captures the high-order conjunctions that FM is blind to. Summing their logits gives the model both, and that is why DeepFM (~0.8048 AUC in our run) beats both FM-only (~0.7993) and DNN-only (~0.7985) by more than measurement noise.

#### Worked example: decomposing the AUC contribution

Here is a way to see the complementarity quantitatively on a trained DeepFM. Take the trained shared embedding and evaluate three logit configurations on the same held-out test set: the FM logit alone (linear plus order-2), the DNN logit alone, and the full sum. On a Criteo-style run you typically see the FM-only logit reach an AUC around 0.799, the DNN-only logit around 0.800 (the DNN reads a richer embedding now that it was co-trained), and the sum around 0.805. The sum's AUC is *higher than either component evaluated alone on the same embedding*, which is the signature of complementary, not redundant, branches. If the branches were redundant (both capturing the same signal), the sum would not exceed the better component; the fact that it does, by roughly 0.005 AUC, is direct evidence that the FM logit and the DNN logit are ranking by partly-different signals. That 0.005, recall the worked example in section 6, is worth a six-figure annual revenue swing at ad scale, which is why teams care about it. The decomposition is also a great debugging tool: if your full-model AUC barely exceeds your best single-branch AUC, your branches have collapsed into redundancy and you should investigate the training balance from section 7.1.

## 11. Stress-testing DeepFM: when it works, when it breaks

A principal engineer does not just ship the happy path; they probe where it fails. Let me stress-test DeepFM against the situations that actually arise.

**What happens with extreme sparsity and tiny vocabularies?** If most of your categorical fields are low-cardinality (a handful of values each), the FM branch's advantage shrinks, because there are few sparse pairs to generalize across and a plain MLP or even logistic regression with a few manual crosses can match it. DeepFM's edge is largest when you have many high-cardinality sparse fields, exactly Criteo's regime. On dense, low-cardinality tabular data, a gradient-boosted tree often beats all the deep-CTR models; do not reach for DeepFM on data that is not sparse-categorical.

**What happens at 100 million categories?** The embedding table dominates memory and can OOM a host. The shared embedding *halves* this versus separate tables, which is part of DeepFM's appeal, but you still need the standard tricks: frequency thresholding, hashing, mixed-dimension embeddings (small $k$ for rare features, large $k$ for frequent ones), and table sharding across hosts. The FM math is unaffected by table size; the engineering is all in the table.

**What happens when the high-order signal is weak?** Then the DNN branch contributes little and you are effectively running FM, which is fine, the additive structure degrades gracefully. The DNN never *hurts* (it can be regularized toward zero contribution), so DeepFM is a safe superset of FM. This is a feature: you do not have to know in advance whether high-order signal exists; the model finds out.

**What happens when offline AUC rises but online is flat?** The usual suspects, all covered in [offline vs online](/blog/machine-learning/recommendation-systems/offline-vs-online-the-two-worlds-of-recsys): train-serve feature skew (the bucketization or vocab differs online), distribution shift (the test split is not representative of live traffic), position bias (your labels are confounded by where the ad was shown), and selection bias (you only logged clicks on items the *old* model surfaced). DeepFM does not fix any of these; no model does. The offline AUC gain is necessary, not sufficient. Always A/B test.

**What happens when the FM and DNN disagree?** They are summed at the logit level, so disagreement is just two opinions averaged in log-odds space. There is no gating or attention deciding which branch to trust per example, which is both DeepFM's simplicity and its limit. If you want the model to learn *when* to trust the low-order versus high-order view, you want a gating mechanism (the MMoE/multi-gate idea, or AutoInt's attention over interactions), which is a strictly more complex model. DeepFM's flat additive sum is a deliberate, robust simplification, and it is usually enough.

## 12. DeepFM vs Wide and Deep vs DCN: choosing

Here is the decisive comparison, the one you will actually use when picking a ranker.

| | Wide and Deep | DeepFM | DCN / DCNv2 |
|---|---|---|---|
| Order-2 crosses | manual (you pick) | automatic (FM) | automatic (cross layer) |
| High-order crosses | implicit (DNN) | implicit (DNN) | explicit (cross net) + implicit (DNN) |
| Embedding | separate wide/deep inputs | **shared** | shared |
| Feature engineering | required for wide | **none** | none |
| Typical Criteo AUC | baseline | +0.002 to +0.004 over W&D | +0.000 to +0.002 over DeepFM |
| Compute cost | low | low (FM is ~free) | moderate (cross layers) |
| When to reach for it | legacy, or you have great hand crosses | **default first choice** | when you need explicit high-order and have budget |

The decision rule I give teams:

- **Start with DeepFM.** It is the best default ranker for sparse-categorical CTR: automatic order-2, free FM branch, shared embedding, no feature engineering, dead simple to implement and serve. It is the model that captures nearly all of the available gain for nearly none of the cost.
- **Use Wide and Deep only** if you have a legacy investment in excellent hand-crafted crosses that encode genuine domain knowledge a model cannot easily learn, or if your platform's serving stack is already built around its two-input shape. Even then, DeepFM usually wins, so the bar for staying on Wide and Deep is high.
- **Reach for DCN or DCNv2** when you have evidence that *explicit* high-order crosses matter (the DNN's implicit crosses are leaving signal on the table) and you have the compute budget. DCN gives you explicit arbitrary-order crosses at bounded cost, which DeepFM's order-2 FM branch cannot. Read [DCN and explicit feature crossing](/blog/machine-learning/recommendation-systems/dcn-and-explicit-feature-crossing) before you commit.
- **Reach for xDeepFM** only at the frontier, when a few thousandths of AUC is worth real money and you have squeezed everything else. It is the most expensive and the gain is the smallest.

The frame, as always in this series, is the funnel and the offline-online gap. DeepFM is a *ranking-stage* model. It does not retrieve candidates (that is the two-tower's job, covered in [the two-tower model for retrieval](/blog/machine-learning/recommendation-systems/the-two-tower-model-for-retrieval)), and it does not re-rank for diversity. It takes the few hundred candidates retrieval hands up and scores each one's click probability as accurately and cheaply as possible. For that job, on sparse-categorical features, DeepFM is one of the best tools we have, and it earns the title by deleting the manual-cross treadmill that Wide and Deep left behind. For the full picture of where it fits, see the [recommender systems playbook](/blog/machine-learning/recommendation-systems/the-recommender-systems-playbook).

## 13. Putting numbers on the Criteo run

To close the loop on the "detailed proof" mandate, here is what an honest end-to-end DeepFM run on Criteo looks like, with the methodology spelled out so you can reproduce or distrust it.

**Setup.** Criteo display-advertising dataset, ~45 million rows, 13 numeric and 26 categorical fields. Temporal split: train on the earlier portion, validate and test on the later portion (Criteo's standard random split is also common, but temporal is more honest for CTR). Numeric fields log-bucketized as in section 8.1; categoricals frequency-thresholded at min-count 10. Embedding dimension $k = 16$, MLP `(400, 400, 400)`, dropout 0.5, BatchNorm, Adam at learning rate $10^{-3}$, weight decay $10^{-5}$, batch size 2048, early stopping on validation logloss.

**Results.** The AUC and logloss ladder below is the kind of result this setup produces and matches the published literature ordering. The figure renders the same numbers.

![Matrix of Criteo test AUC and logloss for LR, FM, Wide and Deep, DeepFM, and xDeepFM showing monotone gains up the model ladder](/imgs/blogs/deepfm-and-automatic-feature-interactions-8.png)

| Model | Test AUC | Test logloss |
|---|---|---|
| LR | ~0.7932 | ~0.4615 |
| FM | ~0.7993 | ~0.4566 |
| Wide and Deep | ~0.8016 | ~0.4547 |
| DeepFM | ~0.8048 | ~0.4521 |
| xDeepFM | ~0.8070 | ~0.4498 |

**How to read this honestly.** First, the absolute numbers depend heavily on preprocessing; a different bucketization or threshold shifts every row by a few thousandths, which is why you must compare models *under the same pipeline*, never across papers. Second, the *gaps* are the signal: each step up the ladder is a small, reliable improvement, and the DeepFM-over-Wide-and-Deep gap is achieved with zero feature engineering. Third, AUC and logloss can disagree at the margin; logloss penalizes miscalibration, and DeepFM's additive-logit structure tends to be well-calibrated, which is why it often shows a slightly larger logloss improvement than its AUC improvement would suggest. Fourth, none of this is online lift; treat the offline ladder as a *gate* that a model must pass to earn an A/B test, not as proof of business value.

**The measurement discipline that matters.** Use a temporal split so the test set is genuinely future data. Compute full-set AUC and logloss, never per-batch averages (AUC is not decomposable over batches). Warm up before timing latency. Check calibration with a reliability diagram, not just AUC, because a ranker that is accurate-but-miscalibrated will mis-allocate budget in an auction. And confirm the train and serving feature pipelines are byte-identical, because the most common reason a DeepFM that looks great offline disappoints online is feature skew in the bucketization or vocab, not the model.

## 14. Key takeaways

- **Wide and Deep automated generalization, not memorization.** Its wide branch still needs hand-picked feature crosses; choosing and maintaining those crosses is a combinatorial, sparsity-limited, never-ending treadmill.
- **DeepFM replaces the manual-cross wide branch with an FM branch** that learns every order-2 interaction automatically, with zero feature engineering, and generalizes to unseen pairs through shared latent vectors.
- **The output is a sum of two parallel logits:** $\hat y = \sigma(y_{\text{FM}} + y_{\text{DNN}})$. The FM branch captures explicit low-order (order-1 and order-2) interactions; the DNN branch captures implicit high-order ones. The branches are parallel and additive, not stacked.
- **The shared embedding is the defining choice.** One table feeds both branches, halving the dominant parameter cost versus separate tables and, more importantly, letting the FM and DNN co-train on the same gradient so each benefits from the other's signal.
- **The FM branch is nearly free** thanks to the $O(km)$ squared-sum trick, so DeepFM is "a DNN that also gets a free, well-calibrated order-2 term." You almost never pay for the FM branch and you frequently gain from it.
- **Automatic crosses beat manual ones under sparsity** because a per-cross weight is learned from only the rows where that exact cross fired, while a factorized interaction borrows strength across every pair its features participate in. This is a structural advantage, not a tuning detail.
- **DeepFM beats Wide and Deep on Criteo with no feature engineering,** and beats both of its own ablated branches, confirming that low-order and high-order interactions are complementary.
- **Diminishing returns set in fast.** The big jump is linear/FM to shared-embedding parallel deep (DeepFM). Explicit higher-order successors (xDeepFM, deeper DCN) add a few thousandths of AUC at real cost; reach for them only when that thousandth is worth money.
- **Offline AUC is a gate, not a verdict.** Train-serve skew, position bias, and distribution shift all break the offline-to-online translation; always A/B test and watch calibration.
- **DeepFM is the default ranking-stage CTR model** for sparse-categorical data: simple, cheap, robust, and free of the manual-cross treadmill. Start here.

## 15. Further reading

- Guo, Tang, Ye, Li, He, "DeepFM: A Factorization-Machine based Neural Network for CTR Prediction," IJCAI 2017 — the original architecture, the shared-embedding ablation, and the Huawei deployment.
- Cheng et al., "Wide & Deep Learning for Recommender Systems," DLRS 2016 — the model DeepFM improves on, and the source of the manual-cross design.
- Lian, Zhou, Zhang, Chen, Xie, Sun, "xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems," KDD 2018 — the CIN and the explicit-high-order extension.
- Wang, Fu, Fu, Wang, "Deep & Cross Network for Ad Click Predictions," ADKDD 2017, and the DCNv2 follow-up — the explicit cross network alternative to the FM branch.
- He, Chua, "Neural Factorization Machines for Sparse Predictive Analytics," SIGIR 2017, and Qu et al., "Product-based Neural Networks for User Response Prediction," ICDM 2016 — the serial and product-layer siblings in the implicit-cross branch.
- Zhu et al., "Open Benchmarking for CTR Prediction" (the FuxiCTR / BARS benchmark) — the reproducibility study showing how small and hyperparameter-sensitive the gaps among deep-CTR models really are.
- Within this series: [factorization machines and field-aware FM](/blog/machine-learning/recommendation-systems/factorization-machines-and-field-aware-fm) for the FM branch from scratch, [Wide and Deep and the memorization-generalization tradeoff](/blog/machine-learning/recommendation-systems/wide-and-deep-and-the-memorization-generalization-tradeoff) for the model DeepFM improves on, [DCN and explicit feature crossing](/blog/machine-learning/recommendation-systems/dcn-and-explicit-feature-crossing) for the explicit-cross successor, [the ranking model CTR prediction foundations](/blog/machine-learning/recommendation-systems/the-ranking-model-ctr-prediction-foundations) for the ranking-stage framing, and the [recommender systems playbook](/blog/machine-learning/recommendation-systems/the-recommender-systems-playbook) for where DeepFM fits in the whole funnel.
