---
title: "Neural Collaborative Filtering and Its Critique"
date: "2026-06-19"
publishDate: "2026-06-19"
description: "Build GMF, MLP, and NeuMF in PyTorch, then watch a properly tuned dot-product baseline match them on MovieLens — and learn why the dot product, not the deep scorer, is what survives to production."
tags:
  [
    "recommendation-systems",
    "recsys",
    "neural-collaborative-filtering",
    "neumf",
    "matrix-factorization",
    "collaborative-filtering",
    "reproducibility",
    "machine-learning",
    "movielens",
    "pytorch",
  ]
category: "machine-learning"
subcategory: "Recommendation Systems"
author: "Hiep Tran"
featured: true
readTime: 52
image: "/imgs/blogs/neural-collaborative-filtering-and-its-critique-1.png"
---

In 2017 a paper landed that seemed to end an argument before it started. Matrix factorization had been the workhorse of recommenders since the Netflix Prize, but it scored a user-item pair with a humble dot product, and the dot product is a fixed, linear thing. The paper's pitch was irresistible: replace that rigid dot product with a neural network, let a multi-layer perceptron learn whatever interaction the data demands, and you get a strictly more powerful model. The benchmarks agreed. Neural Collaborative Filtering, and its flagship NeuMF, posted higher HitRate and NDCG on MovieLens and Pinterest than the matrix-factorization baselines it was tested against, and for the next three years nearly every new deep recommender cited it as the reason to go deep. If you trained a recommender in 2018 and someone asked why you used embeddings plus an MLP instead of plain factorization, "NCF showed it's better" was a complete answer.

Then, in 2020, a second paper quietly pulled the rug. Rendle and colleagues at Google re-ran the comparison with one change: they actually tuned the matrix-factorization baseline. A well-tuned dot product matched or beat the MLP. Worse for the deep camp, when they set the MLP the seemingly trivial task of *learning the dot product itself*, the MLP struggled — it took far more parameters and data to approximate a function the dot product computes exactly and for free. The headline result of the original paper had rested less on the deep model being good than on the baseline being under-tuned. This is not a story about a wrong paper. It is one of the cleanest teaching moments in applied machine learning: a reminder that "deep" is not automatically better, that a baseline you did not tune is not a baseline, and that the dot product is a remarkably strong inductive bias — strong enough that everything downstream in this series, including the two-tower retrieval model, is built to keep it.

This post sits in the series **Recommendation Systems: From Click to Production**, right after [matrix factorization](/blog/machine-learning/recommendation-systems/matrix-factorization-the-workhorse) and right before the two-tower model. The frame for the whole series, set up in [what is a recommender system](/blog/machine-learning/recommendation-systems/what-is-a-recommender-system), is the retrieval-to-ranking-to-re-ranking funnel fed by the serve-log-train feedback loop, read off the offline-versus-online gap. NCF is the moment that frame gets tested. We will build GMF, MLP, and NeuMF from scratch in PyTorch, train them on MovieLens with negative sampling, and — the part the original paper skimped on — train a properly tuned factorization baseline alongside them. Then we measure HitRate@10 and NDCG@10 under the leave-one-out protocol the NCF paper introduced, and we will see for ourselves how close the numbers really are. By the end you will be able to implement all four models, reason about why the dot product is both accurate and cheap to serve, and decide on principle when a learned scorer earns its keep and when it is a liability. The figure below is the architecture the whole post turns on: the NeuMF model, with its two parallel branches fusing into a single score.

![Diagram of the NeuMF architecture showing user and item embeddings feeding a GMF branch and an MLP branch in parallel which fuse into a single sigmoid score](/imgs/blogs/neural-collaborative-filtering-and-its-critique-1.png)

## 1. Where matrix factorization stops and a question begins

Matrix factorization gives every user $u$ a vector $p_u \in \mathbb{R}^k$ and every item $i$ a vector $q_i \in \mathbb{R}^k$, and scores their affinity with a dot product:

$$\hat{y}_{ui} = p_u^\top q_i = \sum_{f=1}^{k} p_{uf}\, q_{if}.$$

That single inner product is the entire scoring engine. It is linear in each factor, symmetric, and it treats every latent dimension independently before summing. For a decade that was enough, because the *embeddings* carry the modelling power: the model can place a user near the items they like in the $k$-dimensional space, and the dot product just reads off how aligned they are. If you have read the [matrix factorization](/blog/machine-learning/recommendation-systems/matrix-factorization-the-workhorse) post you know how far that geometry stretches.

But the dot product does impose real constraints, and they are easy to state. Consider three users and the latent dimension that encodes "likes slow-burn thrillers." Suppose user A is strongly positive on this dimension, user B is strongly negative, and the dimension's contribution to a score is $p_{uf}\, q_{if}$ — a product, linear in $q_{if}$. The dot product cannot, on its own, express "user A likes thrillers only when they are also short" unless that conjunction happens to fall along some other learned dimension. Cross-effects between dimensions are not modelled; each factor pair contributes additively. A function like "score is high only when factor 1 and factor 2 are *both* large, otherwise low" — a logical AND between latent tastes — is exactly the kind of interaction a dot product handles poorly and a nonlinear function could, in principle, capture.

This is the gap the NCF paper aimed at. Its claim, stripped to one sentence: the dot product is a *hand-chosen, fixed* interaction function, and we should *learn* the interaction function instead. Replace $p_u^\top q_i$ with $f(p_u, q_i)$ where $f$ is a neural network trained end to end. Because a multi-layer perceptron with enough width is a universal function approximator, $f$ can represent the dot product as a special case and also represent interactions the dot product cannot. On paper this is unarguable: a strictly larger hypothesis class contains the smaller one. The whole drama of the next several sections is the distance between "can represent in principle" and "will learn from finite data at finite compute," because that distance is where the 2020 critique lives.

There is a deeper way to frame this that pays off later, so it is worth stating now. Every model is a bet about the shape of the answer before it sees the data — that bet is its *inductive bias*. The dot product bets that affinity is bilinear in a shared latent space: users and items live in the same geometry, and how much a user likes an item is how aligned their vectors are. That bet is wrong in some corner cases (the conjunctions above) but it is *cheap* — it has no scorer parameters to fit and it generalizes from very little data because the hypothesis class is small. The MLP bets almost nothing: it says affinity is *some* continuous function of the two embeddings, and it will figure out which one from data. That weak bet buys a huge hypothesis class, but a huge hypothesis class needs a lot of data to pin down and offers no free generalization. The entire NCF debate is a referendum on which bet pays off on real recommender data, and the answer — spoiler — is that the strong, "wrong" bet of the dot product wins, because recommender data is sparse and skewed, exactly the regime where a strong prior beats a flexible model that has to learn everything.

Before we build anything, fix the data setting so the numbers later mean something. We use **MovieLens-1M**: about 6,040 users, 3,706 movies, roughly 1,000,209 ratings. Following the NCF paper, we treat it as *implicit* feedback — every rating, whatever its star value, becomes a positive interaction (the user engaged with the movie at all), and everything unobserved is a candidate negative. This is the implicit-feedback regime covered in [implicit feedback models](/blog/machine-learning/recommendation-systems/implicit-feedback-models-als-and-bpr): we do not predict the star rating, we rank items by how likely the user is to interact. That reframing matters, because it is the setting in which all four models below are trained and compared. And note the sparsity: MovieLens-1M fills only about 4.5 percent of its 22.4 million possible user-item cells, so any model has to generalize aggressively from a thin slice of observed interactions — which is precisely the data regime where the dot product's strong prior earns its keep and the MLP's flexibility becomes a liability rather than an asset.

## 2. GMF: generalizing the dot product

The first building block, Generalized Matrix Factorization (GMF), is the gentlest possible neural generalization of the dot product, and it is the right place to start because it makes the "learned interaction" idea concrete without leaving the world of linear algebra.

Recall the dot product is $p_u^\top q_i = \sum_f p_{uf} q_{if}$. GMF keeps the element-wise product $p_u \odot q_i$ (a $k$-dimensional vector whose $f$-th entry is $p_{uf} q_{if}$) but instead of summing the entries with equal weight, it learns a weight for each:

$$\hat{y}_{ui} = \sigma\!\left( h^\top (p_u \odot q_i) \right),$$

where $h \in \mathbb{R}^k$ is a learned output-layer weight vector and $\sigma$ is the sigmoid (because we are now predicting an interaction probability, not a star rating). Read that carefully and you see exactly what GMF buys you. If you set $h = \mathbf{1}$ (the all-ones vector) and drop the sigmoid, you recover the plain dot product. So GMF *contains* matrix factorization as the special case $h = \mathbf{1}$. What it adds is the freedom to up-weight or down-weight individual latent dimensions in the final sum: maybe dimension 3 ("action intensity") deserves twice the influence of dimension 7 ("art-house slowness") in the click probability. That is a genuine, if modest, generalization — and crucially it is still a *linear* function of the element-wise product.

GMF in PyTorch is barely more than two embedding tables and a single linear layer:

```python
import torch
import torch.nn as nn

class GMF(nn.Module):
    def __init__(self, n_users, n_items, dim=32):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, dim)
        self.item_emb = nn.Embedding(n_items, dim)
        self.out = nn.Linear(dim, 1)  # this is h^T applied to (p . q)
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)

    def forward(self, u, i):
        p = self.user_emb(u)          # (B, dim)
        q = self.item_emb(i)          # (B, dim)
        elementwise = p * q           # (B, dim)  the Hadamard product
        logit = self.out(elementwise) # (B, 1)    h^T (p . q) + bias
        return logit.squeeze(-1)      # (B,)  raw logit; apply BCEWithLogits later
```

Two implementation notes that matter in practice. First, we return the raw logit and pair it with `BCEWithLogitsLoss`, which is numerically stable (it fuses the sigmoid into the loss) — never apply `sigmoid` and then `BCELoss` separately. Second, the embedding initialization standard deviation of `0.01` is not decorative; with implicit feedback and heavy negative sampling, large initial embeddings push early logits to saturate the sigmoid and stall the gradient. Small init keeps you in the linear regime of the sigmoid where gradients flow. This is the kind of detail that, when wrong, quietly costs you a couple of points of HitRate and gets blamed on the architecture.

The honest thing to say about GMF is that it is *almost* matrix factorization. The added weight vector $h$ is $k$ extra parameters on top of the embeddings, and it can only re-scale dimensions, not mix them. It cannot represent a cross-effect between dimension 1 and dimension 2, because there is no term involving $p_{u1} q_{i2}$ anywhere in $h^\top (p_u \odot q_i)$. So GMF will, at best, match a well-tuned MF and occasionally edge it by a hair. If you wanted to surpass the dot product's *expressive* limits, you needed something that mixes dimensions nonlinearly — and that is the MLP branch.

One more thing GMF quietly loses versus a *good* matrix factorization is the bias terms. A production-grade MF predicts $\hat{y}_{ui} = \mu + b_u + b_i + p_u^\top q_i$, where $\mu$ is the global average interaction propensity, $b_u$ captures how active a user is overall, and $b_i$ captures how popular an item is independent of taste. Those bias terms do a lot of work — item popularity alone explains a large share of which items get clicked — and they cost three scalars. The bare GMF as written has none of them; the only popularity signal it can express is whatever leaks into embedding norm. When the original NCF paper's MF baseline under-performed, part of the story is almost certainly that the *strong* version of MF, with bias terms and a tuned regularizer, was not the thing being compared. A baseline that drops the bias terms and the hyperparameter search is not matrix factorization at its best; it is matrix factorization with one hand tied behind its back, which is exactly the kind of under-specified baseline the 2020 paper called out. If you implement GMF for real, add the user and item bias terms; they are nearly free and they matter.

## 3. The MLP branch: learning an arbitrary interaction

The MLP model throws out the element-wise product entirely. Instead of combining $p_u$ and $q_i$ multiplicatively, it *concatenates* them into a single $2k$-dimensional vector and feeds that to a stack of fully connected layers with nonlinear activations:

$$z_0 = \begin{bmatrix} p_u \\ q_i \end{bmatrix}, \qquad z_\ell = \mathrm{ReLU}(W_\ell z_{\ell-1} + b_\ell), \qquad \hat{y}_{ui} = \sigma(h^\top z_L).$$

The logic is seductive. Concatenation is lossless — both vectors are fully present in $z_0$ — and a deep ReLU network is a universal approximator, so $f(p_u, q_i)$ can in principle learn *any* continuous interaction, including the dot product, the element-wise-then-weighted sum of GMF, and arbitrarily complicated conjunctions the dot product cannot touch. The tower of layers is where the model is supposed to discover that "user likes thrillers AND movie is short" cross-effect on its own. This is the architectural embodiment of the deep-learning thesis applied to recommenders: stop hand-designing the interaction, let gradient descent find it.

Here is the MLP in PyTorch, with the tower-narrowing layer sizes the paper used (each layer halves the width):

```python
class MLPRec(nn.Module):
    def __init__(self, n_users, n_items, dim=32, layers=(64, 32, 16, 8)):
        super().__init__()
        # NOTE: MLP uses its OWN embeddings, separate from any GMF embeddings.
        self.user_emb = nn.Embedding(n_users, dim)
        self.item_emb = nn.Embedding(n_items, dim)
        seq, in_dim = [], 2 * dim          # concat doubles the width
        for out_dim in layers:
            seq += [nn.Linear(in_dim, out_dim), nn.ReLU()]
            in_dim = out_dim
        self.mlp = nn.Sequential(*seq)
        self.out = nn.Linear(in_dim, 1)
        for emb in (self.user_emb, self.item_emb):
            nn.init.normal_(emb.weight, std=0.01)

    def forward(self, u, i):
        p, q = self.user_emb(u), self.item_emb(i)
        z = torch.cat([p, q], dim=-1)      # (B, 2*dim)  concatenation, not product
        z = self.mlp(z)
        return self.out(z).squeeze(-1)
```

Now look closely at what the MLP has to do that the dot product gets for free. The dot product needs the *product* $p_{uf} q_{if}$. But a ReLU MLP is built from additions and thresholds — it has no native multiply. To compute even a single product of two of its inputs, a ReLU network must approximate the multiplication function $g(a, b) = ab$ using a piecewise-linear surface, and approximating a smooth product to low error over a range takes a surprising number of hidden units. There is a known result here worth internalizing: a ReLU network approximating the product of two scalars to error $\epsilon$ needs on the order of $\log(1/\epsilon)$ layers or a hidden width that grows as the precision tightens. The dot product needs $k$ multiplications and one sum; the MLP must *learn* each of those multiplications as a sub-network, and there are $k$ of them, and they only appear if the data forces them to. The figure below makes the contrast concrete: the dot-product scorer computes the answer in one pass, while the MLP scorer pays for the same answer in parameters and in a learning problem it might not solve.

![Diagram contrasting dot-product matrix factorization scoring on the left with MLP-learned scoring on the right showing the dot product uses one inner product and no extra parameters while the MLP needs many parameters and cannot use approximate nearest neighbor search](/imgs/blogs/neural-collaborative-filtering-and-its-critique-2.png)

This is the crux of the entire critique, and it is worth sitting with. The MLP is *more expressive* — its hypothesis class strictly contains the dot product. But more expressive does not mean easier to optimize, and it does not mean it will recover the simple, correct answer from finite data. Universal approximation is a statement about existence ("there is a setting of weights that does this"), not about learnability ("gradient descent on this data will find it"). The dot product hard-codes the right inductive bias; the MLP has to discover it, and discovery is expensive and unreliable. Section 7 shows an MLP visibly failing to learn a two-dimensional dot product on a clean toy task. Hold that thought.

## 4. NeuMF: fusing both branches

NeuMF (Neural Matrix Factorization) is the paper's headline model, and the idea is to get the best of both worlds: keep GMF's linear, MF-like branch *and* the MLP's nonlinear branch, run them in parallel, and fuse their outputs before the final score. Critically, NeuMF gives the two branches *separate* embedding tables — a GMF set of user and item embeddings and an independent MLP set — because forcing both branches to share one embedding space would shackle the flexible MLP to whatever embedding the rigid GMF prefers. The fusion concatenates the GMF output vector with the MLP's last hidden vector, then a single output layer maps the concatenation to a score:

$$\phi^{\mathrm{GMF}} = p_u^{G} \odot q_i^{G}, \qquad \phi^{\mathrm{MLP}} = z_L, \qquad \hat{y}_{ui} = \sigma\!\left( h^\top \begin{bmatrix} \alpha\,\phi^{\mathrm{GMF}} \\ (1-\alpha)\,\phi^{\mathrm{MLP}} \end{bmatrix} \right).$$

The layered view of this fusion is the cleanest way to hold it in your head: two embedding sets at the bottom, two branches in the middle, a concatenation that joins them, and one neuron at the top.

![Stacked layer diagram of NeuMF showing two embedding sets at the bottom feeding a GMF element-wise branch and an MLP dense branch in the middle which concatenate into a single output neuron with a sigmoid at the top](/imgs/blogs/neural-collaborative-filtering-and-its-critique-5.png)

The paper also recommends a pretraining trick that tells you something about how hard the fused model is to optimize: train GMF alone to convergence, train MLP alone to convergence, then initialize NeuMF from both, and scale the two halves of the fusion with a trade-off $\alpha$ (they use $\alpha = 0.5$ at init). NeuMF trained from random init with Adam underperforms; it really wants the pretrained start. That should already raise an eyebrow — if a model needs careful pretraining of its sub-models to beat them, the marginal value of the fusion over its parts is suspect. Here is NeuMF assembled from the two branches:

```python
class NeuMF(nn.Module):
    def __init__(self, n_users, n_items, gmf_dim=32, mlp_dim=32,
                 layers=(64, 32, 16, 8)):
        super().__init__()
        # Separate embeddings per branch — this is the key NeuMF design choice.
        self.gmf_user = nn.Embedding(n_users, gmf_dim)
        self.gmf_item = nn.Embedding(n_items, gmf_dim)
        self.mlp_user = nn.Embedding(n_users, mlp_dim)
        self.mlp_item = nn.Embedding(n_items, mlp_dim)
        seq, in_dim = [], 2 * mlp_dim
        for out_dim in layers:
            seq += [nn.Linear(in_dim, out_dim), nn.ReLU()]
            in_dim = out_dim
        self.mlp = nn.Sequential(*seq)
        self.out = nn.Linear(gmf_dim + in_dim, 1)  # fuse: concat GMF + last MLP
        for emb in (self.gmf_user, self.gmf_item, self.mlp_user, self.mlp_item):
            nn.init.normal_(emb.weight, std=0.01)

    def forward(self, u, i):
        gmf = self.gmf_user(u) * self.gmf_item(i)             # (B, gmf_dim)
        mlp_in = torch.cat([self.mlp_user(u), self.mlp_item(i)], dim=-1)
        mlp = self.mlp(mlp_in)                                # (B, last_layer)
        fused = torch.cat([gmf, mlp], dim=-1)                # concat both branches
        return self.out(fused).squeeze(-1)
```

There is a real subtlety in *why* NeuMF wants pretraining, and it is worth unpacking because it is diagnostic. Train GMF alone with Adam and it converges fine; train MLP alone with Adam and it converges fine; train NeuMF from scratch with Adam and it lands *below* both. The standard explanation is that Adam's per-parameter adaptive learning rates do not transfer cleanly when you suddenly couple two sub-networks that were each happy on their own — the fused loss surface has a worse-conditioned basin near random init, and the two branches fight over the shared output layer before either has learned anything useful. The paper's fix is to pretrain GMF and MLP separately (with Adam), then assemble NeuMF from those weights and fine-tune the *whole* thing with plain SGD (not Adam), because the pretrained weights already carry Adam's accumulated momentum and re-applying it double-counts. If a model only beats its components when you hand it its components' trained weights, the fusion is contributing very little of its own — a quiet corroboration of the critique to come.

#### Worked example: embedding-table memory for NeuMF versus MF

The serving cost of a learned scorer is the headline, but the *memory* cost of NeuMF's design is its own quiet tax, and it is easy to compute. NeuMF gives each branch its own embeddings, so it stores *two* user tables and *two* item tables, where plain MF stores one of each. Take a production-scale catalog: $M = 50$ million users, $N = 5$ million items, embedding dimension $k = 64$, float32 (4 bytes).

Plain MF stores one user table and one item table: $(50{,}000{,}000 + 5{,}000{,}000) \times 64 \times 4 \text{ bytes} = 55{,}000{,}000 \times 256 \approx 14.1\text{ GB}$.

NeuMF stores a GMF pair *and* an MLP pair — double the embeddings: $2 \times 14.1 \approx 28.2\text{ GB}$, plus the MLP weights (a rounding error at about 11 KB). So NeuMF roughly *doubles* the embedding table, the single largest memory consumer in a recommender, for an accuracy gain within noise of the dot product. At this scale that is the difference between a model that fits on one host's RAM and one that needs sharding across two, with all the cross-host-lookup latency that implies. The dot-product baseline is not just cheaper to *score*; it is cheaper to *store*. Both axes point the same way.

NeuMF is the most expressive of the three and, in a fair fight, posts the best HitRate of the neural models by a small margin. It is also the most parameter-heavy (two full embedding tables plus the MLP), the slowest to train, and — the point that will matter most in production — it scores with an MLP, which means it *cannot* use the dot product at serving time. We will return to that serving consequence with force in section 8. The matrix below lines up all four scorers on the dimensions that actually decide which one you ship: the form of the interaction, the parameter cost of the scorer, whether the scorer is compatible with approximate nearest neighbor search, and the headline HitRate.

![Comparison matrix of tuned matrix factorization versus GMF versus MLP versus NeuMF across interaction form, scorer parameters, approximate-nearest-neighbor compatibility, and HitRate at 10 showing tuned MF keeps a free dot product and stays nearest-neighbor compatible while the deep scorers cannot](/imgs/blogs/neural-collaborative-filtering-and-its-critique-3.png)

## 5. The loss: binary cross-entropy with negative sampling

All four models are trained the same way, and the training setup is where implicit feedback forces a design choice that the original paper got right and that everyone copying it should understand. We have only *positive* interactions — the things users actually engaged with. We have no explicit negatives. So we *sample* negatives: for each observed positive $(u, i^+)$, draw some number of items $i^-$ the user has not interacted with and label them as negatives. The model then learns to assign high probability to positives and low to sampled negatives via binary cross-entropy.

Formally, with a sigmoid output $\hat{y}_{ui} = \sigma(s_{ui})$ giving the predicted interaction probability and label $y_{ui} \in \{0, 1\}$, the per-example loss is

$$\mathcal{L} = -\sum_{(u,i) \in \mathcal{D} \cup \mathcal{D}^-} \Big[ y_{ui} \log \hat{y}_{ui} + (1 - y_{ui}) \log(1 - \hat{y}_{ui}) \Big],$$

where $\mathcal{D}$ is the set of observed positives and $\mathcal{D}^-$ is the sampled negatives. The number of negatives per positive — call it the negative ratio — is a real hyperparameter. The NCF paper used 4 negatives per positive during training. Too few and the model sees almost no contrast and collapses toward predicting "everything is a positive"; too many and you waste compute and tilt the implied class balance. Four is a sane default for MovieLens-scale data; we will tune it.

Here is the negative sampler and one training step. The sampler is deliberately simple (uniform random over items, with a cheap collision check against the user's known positives), which is what the paper did and which is also, as we will see in the [implicit feedback](/blog/machine-learning/recommendation-systems/implicit-feedback-models-als-and-bpr) discussion, a meaningful limitation:

```python
import numpy as np

def sample_negatives(user_pos: dict, n_items: int, neg_ratio: int):
    """Yield (user, item, label) triples: each positive plus neg_ratio negatives."""
    users, items, labels = [], [], []
    for u, pos_items in user_pos.items():
        pos_set = set(pos_items)
        for i_pos in pos_items:
            users.append(u); items.append(i_pos); labels.append(1.0)
            drawn = 0
            while drawn < neg_ratio:
                j = np.random.randint(n_items)
                if j in pos_set:           # skip false negatives we know about
                    continue
                users.append(u); items.append(j); labels.append(0.0)
                drawn += 1
    return (np.array(users), np.array(items), np.array(labels, dtype=np.float32))

def train_epoch(model, opt, loader, device):
    model.train()
    bce = nn.BCEWithLogitsLoss()
    total = 0.0
    for u, i, y in loader:
        u, i, y = u.to(device), i.to(device), y.to(device)
        opt.zero_grad()
        logit = model(u, i)
        loss = bce(logit, y)
        loss.backward()
        opt.step()
        total += loss.item() * len(y)
    return total / len(loader.dataset)
```

One subtle but important honesty note on sampled negatives: uniform sampling draws mostly *easy* negatives — random items the user would obviously never click, which the model learns to reject after a few epochs, after which they stop teaching it anything. This is the same false-negative-and-easy-negative problem that motivates pairwise losses like BPR and, later, hard-negative mining in two-tower retrieval. The pointwise BCE used here is fine for a fair benchmark (it is what NCF used) but it is not the strongest training signal available. We keep it constant across all four models precisely so the comparison is apples to apples; if you let the deep model use a better loss than the baseline, you have re-introduced the very bias the 2020 critique was about. Tune both, or tune neither.

### 5.1 The gradient, and why it pushes embeddings into place

It is worth deriving the gradient once, because it shows precisely what the loss does to the embeddings and why the dot-product model and the MLP model are learning the *same* thing through different machinery. Take a single example with score $s_{ui} = p_u^\top q_i$ (the GMF/MF case, with $h = \mathbf{1}$ for clarity), predicted probability $\hat{y} = \sigma(s)$, and label $y$. The binary cross-entropy for that example is $\ell = -[y \log \hat{y} + (1-y)\log(1-\hat{y})]$. The well-known and beautiful fact about the sigmoid-plus-cross-entropy pair is that the gradient of the loss with respect to the *logit* collapses to a single term:

$$\frac{\partial \ell}{\partial s} = \hat{y} - y.$$

That is the entire error signal: predicted probability minus label. For a positive ($y = 1$) it is $\hat{y} - 1 \le 0$; for a negative ($y = 0$) it is $\hat{y} - 0 \ge 0$. Now chain through to the embeddings, since $s = p_u^\top q_i$ gives $\partial s / \partial p_u = q_i$ and $\partial s / \partial q_i = p_u$:

$$\frac{\partial \ell}{\partial p_u} = (\hat{y} - y)\, q_i, \qquad \frac{\partial \ell}{\partial q_i} = (\hat{y} - y)\, p_u.$$

Read what gradient descent does with this. For a positive pair, $(\hat{y} - y) < 0$, so $p_u$ is nudged *toward* $q_i$ and $q_i$ toward $p_u$ — the two vectors are pulled together, increasing their dot product next time. For a sampled negative, $(\hat{y} - y) > 0$, so the update *pushes them apart*. The whole training dynamic is "pull positives together, push sampled negatives apart in embedding space," modulated by how wrong the current prediction is ($\hat{y} - y$ shrinks as the model gets it right, so confident-correct examples stop contributing — this is why easy negatives stop teaching after a few epochs). This is the same geometric story the [matrix factorization](/blog/machine-learning/recommendation-systems/matrix-factorization-the-workhorse) post tells, and it is exactly what makes the dot-product model's embeddings *retrievable*: the learned geometry is the thing an ANN index reads off. The MLP, by contrast, scatters the same error signal across its weight matrices, and the geometry of its embeddings is not what determines the score — the tower in front of them is — which is the structural reason the MLP's embeddings are not directly indexable.

### 5.2 The full data pipeline, end to end

To make the benchmark reproducible, here is the data preparation that sits in front of the training loop: load MovieLens, build the user-positives map, do a temporal leave-one-out split, and build the 99-negative test candidates. This is the unglamorous part that, done wrong, leaks future information into training and silently inflates every metric.

```python
import pandas as pd
import numpy as np

def prepare_movielens(ratings_path):
    # ratings.dat: userId::movieId::rating::timestamp
    df = pd.read_csv(ratings_path, sep="::", engine="python",
                     names=["user", "item", "rating", "ts"])
    # implicit feedback: any rating counts as a positive interaction
    df = df.sort_values(["user", "ts"])
    # contiguous ids so embedding tables are dense
    u_ids = {u: k for k, u in enumerate(df.user.unique())}
    i_ids = {i: k for k, i in enumerate(df.item.unique())}
    df["user"] = df.user.map(u_ids)
    df["item"] = df.item.map(i_ids)
    n_users, n_items = len(u_ids), len(i_ids)

    test_pos, train_pos = {}, {}
    for u, grp in df.groupby("user"):
        items = grp.item.tolist()           # already sorted by timestamp
        test_pos[u] = items[-1]             # most recent = held out (no leakage)
        train_pos[u] = items[:-1]           # everything earlier = train

    # 99 sampled negatives per user for the test ranking set
    rng = np.random.default_rng(0)
    user_neg99 = {}
    for u in test_pos:
        seen = set(train_pos[u]) | {test_pos[u]}
        negs = []
        while len(negs) < 99:
            j = int(rng.integers(n_items))
            if j not in seen:
                negs.append(j); seen.add(j)
        user_neg99[u] = negs
    return train_pos, test_pos, user_neg99, n_users, n_items
```

The two non-obvious correctness points: ids are remapped to a dense $0..n$ range so `nn.Embedding` tables have no holes, and the split is *temporal* (hold out each user's last-by-timestamp interaction), not random. A random hold-out would let the model train on a user's future and test on their past, which leaks information and is one of the quiet ways offline numbers end up better than they have any right to be — a trap the [offline-online gap](/blog/machine-learning/recommendation-systems/the-offline-online-gap-and-why-your-metric-lied) post catalogs in detail.

## 6. The critique: a tuned baseline matches the deep model

Now the heart of the post. In 2020, Rendle, Krichene, Zhang, and Anderson published "Neural Collaborative Filtering vs. Matrix Factorization Revisited," and the argument was not that the NCF models were bad. It was that the *comparison* was unfair, and that when made fair, the conclusion flips. Three findings, stated plainly.

**First: a properly tuned dot-product MF matches or beats the MLP.** The original NCF paper compared its neural models against an MF baseline whose hyperparameters — embedding dimension, regularization strength, learning rate, negative ratio — were not searched with anything like the care lavished on the neural models. When Rendle et al. ran a real hyperparameter search over the dot-product model on the *same* MovieLens-1M and Pinterest datasets with the *same* leave-one-out evaluation, the dot-product MF reached HitRate@10 and NDCG@10 on par with or above NeuMF. The deep model's apparent edge largely evaporated. The before-and-after of that baseline is the single most important picture in this post.

![Before and after diagram contrasting an under-tuned matrix factorization baseline as reported in the original deep paper with a properly tuned baseline showing the tuned version closes nearly the whole reported gap and reverses the conclusion](/imgs/blogs/neural-collaborative-filtering-and-its-critique-4.png)

**Second: the MLP cannot easily learn the dot product.** This is the deepest finding and the one that should change how you think. Rendle et al. set up a clean experiment: generate synthetic data where the true affinity *is* a dot product of known embeddings, then ask an MLP — given the same embeddings as input — to learn that dot product. The MLP needed far more parameters and data to approximate $p^\top q$ than the dot product needs to compute it (zero, it is exact), and even then it generalized worse, especially as the embedding dimension grew. The reason is exactly the multiplication problem from section 3: a ReLU MLP approximates products with piecewise-linear patches and never represents them exactly, so a function that is *trivial* in the dot-product hypothesis class is *hard* in the MLP class. The thing the MLP was supposed to be strictly better at — learning interactions — it does worse at for the most fundamental interaction of all.

It is worth being precise about *why* the difficulty grows with dimension, because it is not arbitrary. The dot product over a $k$-dimensional embedding is a sum of $k$ scalar products $p_f q_f$. A ReLU network has no multiply primitive, so it must build each product, and a single scalar product $g(a,b) = ab$ over a bounded box requires a number of linear pieces that grows as you tighten the error. There is a classical construction (Yarotsky, 2017) showing a ReLU network can approximate $x^2$ — and via the polarization identity $ab = \tfrac{1}{4}[(a+b)^2 - (a-b)^2]$, therefore $ab$ — to error $\epsilon$ with on the order of $\log(1/\epsilon)$ layers. That sounds cheap until you remember you need this for *all $k$ dimensions at once* and the network has to *discover* the polarization trick from data rather than being handed it. The hypothesis class is large enough to contain the right answer, but the optimization landscape gives gradient descent no reason to settle on the exact bilinear form rather than one of the astronomically many nearby functions that fit the training points and fail to extrapolate. The dot product does not approximate multiplication; it *is* multiplication, exact and free, and it never has to be discovered. That is the whole asymmetry in one sentence.

**Third: the dot product is the right inductive bias for retrieval.** This is the practitioner's takeaway and the reason it matters for this series specifically. A dot-product scorer turns top-K recommendation into maximum-inner-product search (MIPS), which approximate nearest neighbor indexes solve in sublinear time. An MLP scorer is a black box: to find a user's top items you must run the MLP on every candidate item, a full scan. So even if the MLP were a touch more accurate, it would be vastly more expensive to serve at catalog scale. The dot product is not a limitation you tolerate; it is a feature you fight to keep. That is precisely why the [two-tower model](/blog/machine-learning/recommendation-systems/the-two-tower-model-for-retrieval) puts all the deep capacity *inside the towers* that produce the embeddings, and keeps a plain dot product as the scorer.

The wider context is what makes this more than a single-paper dispute. A year earlier, in 2019, Dacrema, Cremonesi, and Jannach published "Are We Really Making Progress? An Analysis of Neural Recommendation Approaches" at RecSys. They tried to reproduce a batch of recently published deep recommenders and found that most could be matched or beaten by simple, well-tuned baselines — nearest-neighbor methods, linear models, plain matrix factorization — that the deep papers had under-tuned or omitted. NCF was one thread in a broader reproducibility crisis: a literature where "we beat the baseline" too often meant "we out-tuned a baseline nobody tuned." The fix is unglamorous and non-negotiable: a baseline you did not tune is not a baseline, and a comparison where only one side got a real hyperparameter budget is not evidence.

Why does this failure mode happen so reliably, even among careful researchers acting in good faith? Three forces conspire. First, *asymmetric effort*: you spend weeks tuning the model you are proposing because that is your contribution, and you grab the baseline's hyperparameters from a prior paper or a default config because tuning it is not your contribution — so the comparison silently measures who got more attention. Second, *publication incentive*: a result of "our new model ties a well-tuned old one" does not get accepted, so there is pressure, conscious or not, toward configurations where the new model wins. Third, *evaluation convenience*: sampled metrics are fast and everyone uses them, so they propagate, even though they can reorder models. None of these require bad faith; they are structural, which is why the corrective has to be structural too — preregistered baselines, shared tuning budgets, full-catalog metrics as a reporting requirement. The reason this episode is taught is that the fix generalizes far beyond recommenders: any time you read "method A beats method B," your first question should be whether B got the same tuning budget as A. Usually it did not.

#### Worked example: counting parameters of an MLP scorer versus a dot product at serving

Make the cost concrete. Take a catalog of $N = 1{,}000{,}000$ items, embedding dimension $k = 64$, and an MLP scorer with the NCF-style tower over a concatenated $2k = 128$-dimensional input: layers $128 \to 64 \to 32 \to 16 \to 1$.

The dot-product scorer has **zero** scorer parameters — scoring is $s = p_u^\top q_i$, a sum of 64 products, no weights of its own. The MLP scorer's parameters are the weight matrices and biases:

- Layer 1: $128 \times 64 + 64 = 8{,}256$
- Layer 2: $64 \times 32 + 32 = 2{,}080$
- Layer 3: $32 \times 16 + 16 = 528$
- Output: $16 \times 1 + 1 = 17$

Total scorer parameters: $8{,}256 + 2{,}080 + 528 + 17 = 10{,}881$. That is about eleven thousand learned numbers whose only job is to approximate a multiply-and-sum the dot product does exactly with none.

Now the serving cost, which is the part that bites. To recommend the top 10 items for one user with the **dot product**, you issue one MIPS query to an ANN index — roughly $O(\log N)$ work, on the order of a few hundred distance computations regardless of catalog size, returning in single-digit milliseconds. With the **MLP scorer**, there is no index trick: you must run the 10,881-parameter network on all $N = 1{,}000{,}000$ candidate items, which is $10^6$ forward passes per user request. Each forward pass is roughly $2 \times 10{,}881 \approx 21{,}762$ floating-point operations, so about $2.2 \times 10^{10}$ FLOPs per user request just for scoring — and that is *per request*, before ranking. At a million requests, the dot-product path scales as the ANN index does; the MLP path scales linearly with the catalog and is, for practical purposes, un-servable at top-K without first narrowing candidates with a dot-product retrieval stage. The accuracy difference between the two, as we are about to measure, is a couple of points of HitRate. The serving difference is several orders of magnitude. That asymmetry is the whole argument.

## 7. A toy task: watch the MLP fail to learn a dot product

Claims about "the MLP struggles to learn multiplication" land harder when you watch it happen, so here is a small, self-contained experiment you can run in a minute. We generate random pairs of two-dimensional vectors $a, b \in \mathbb{R}^2$, compute their true dot product $a^\top b = a_1 b_1 + a_2 b_2$ as the target, and train an MLP to predict it from the concatenation $[a_1, a_2, b_1, b_2]$. The dot product is the simplest possible interaction; if the MLP cannot nail this, the deep-is-better story is in real trouble.

```python
import torch, torch.nn as nn

torch.manual_seed(0)
def make_data(n):
    ab = torch.randn(n, 4)                 # [a1, a2, b1, b2]
    y = ab[:, 0] * ab[:, 2] + ab[:, 1] * ab[:, 3]   # true dot product
    return ab, y

Xtr, ytr = make_data(50_000)
Xte, yte = make_data(10_000)

mlp = nn.Sequential(
    nn.Linear(4, 128), nn.ReLU(),
    nn.Linear(128, 128), nn.ReLU(),
    nn.Linear(128, 1),
)
opt = torch.optim.Adam(mlp.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()
for epoch in range(200):
    opt.zero_grad()
    pred = mlp(Xtr).squeeze(-1)
    loss = loss_fn(pred, ytr)
    loss.backward(); opt.step()

with torch.no_grad():
    test_mse = loss_fn(mlp(Xte).squeeze(-1), yte).item()
    var_y = yte.var().item()
print(f"test MSE {test_mse:.4f}   target variance {var_y:.4f}   "
      f"fraction unexplained {test_mse / var_y:.3f}")
```

When you run this, the MLP gets *close* but never exact, and the residual is structured. Inside the training region (vectors near the unit ball where most random Gaussian mass sits) it does a passable job because it has tiled the region with enough linear patches. But push the test vectors to larger magnitudes — exactly where a recommender's popular, large-norm item embeddings live — and the error explodes, because a piecewise-linear surface cannot extrapolate a quadratic-in-inputs function like $a_1 b_1$. The dot product would have zero error everywhere. This is the failure in miniature: the MLP is not learning multiplication, it is memorizing a local linear approximation of it, and that approximation degrades precisely in the high-norm region that drives top-K rankings.

The recommender consequence is direct. Item popularity shows up as embedding norm — popular items have large-magnitude vectors so they win many dot products. An MLP scorer that mis-approximates products at large norms will systematically mis-rank exactly the high-traffic items that dominate engagement metrics. The dot product, by hard-coding the correct multiply, is *exact* in that regime for free. The inductive bias is not a constraint that costs accuracy; on the interaction that matters most, it *buys* accuracy.

There is an instructive "fix" for the toy task that proves the point rather than refuting it. If you change the MLP's input from the concatenation $[a_1, a_2, b_1, b_2]$ to the *element-wise product* $[a_1 b_1, a_2 b_2]$ and then let the network just sum, the error drops to essentially zero immediately — because you handed it the multiplication it could not learn. But notice what you have done: you re-introduced the dot product. Feeding the element-wise product is GMF; summing it is matrix factorization. The only way to make the MLP reliably score a dot product is to stop making it learn the dot product and give it the dot product. That is not a workaround; it is the 2020 paper's recommendation stated as a code change. Capacity was never the problem — the missing primitive was, and the primitive is the dot product itself.

#### Worked example: how much data would the MLP need to match the dot product?

Suppose you insist on the MLP scorer and want its approximation error on the product term down to, say, 1 percent relative error across the range of embedding values your model produces. A useful rule of thumb from approximation theory: a ReLU network approximating a smooth function over a $d$-dimensional input to relative error $\epsilon$ needs a number of pieces (and therefore parameters and training samples to fit them without overfitting) that grows roughly like $\epsilon^{-d/2}$ in the worst case. For the bilinear interaction across a $k$-dimensional embedding, the effective input dimension that the products span is large, and tightening $\epsilon$ from 10 percent to 1 percent multiplies the required capacity by an order of magnitude or more.

Put a number on it. If matching the dot product at 10 percent error took the model a few thousand effective samples per region, matching at 1 percent might take tens of thousands, and at 0.1 percent hundreds of thousands — for a function the dot product computes at *zero* error with *zero* samples. Meanwhile your actual interaction data is fixed at one million MovieLens ratings, most of them on a few hundred popular movies, leaving the long tail badly under-sampled. So the MLP is asked to learn a hard function from data that is both finite and skewed, while the dot product simply *is* the function. This is why, empirically, the MLP plateaus a hair below a tuned dot product rather than soaring above it: the expressiveness it gains is spent re-learning multiplication it could have been given.

## 8. Serving: the dot product is what makes retrieval cheap

The reproducibility argument would matter even if both models were trivially cheap to serve. They are not, and the serving gap is the most consequential difference between a dot-product scorer and a learned scorer — important enough that it dictates the architecture of every retrieval model later in this series.

Recommendation at scale is a two-stage funnel: **retrieval** narrows a catalog of millions down to a few hundred candidates, then **ranking** scores those candidates with an expensive model. The retrieval stage has a hard latency budget — single-digit milliseconds — and it has to consider the *entire* catalog. The only way to do that within budget is to make scoring a geometric operation you can index. A dot product is exactly that: scoring $p_u^\top q_i$ over all items is maximum-inner-product search, and ANN structures like HNSW graphs and IVF inverted lists (covered in [approximate nearest neighbor serving](/blog/machine-learning/recommendation-systems/approximate-nearest-neighbor-serving-faiss-hnsw-scann)) return an approximate top-K in time that grows logarithmically, not linearly, with catalog size.

An MLP scorer breaks this completely. There is no index for "items that maximize a 10,000-parameter neural network of this user's embedding," because the score is not a distance or an inner product — it is an arbitrary nonlinear function with no geometric structure to exploit. To get the top-K under an MLP scorer you must evaluate the MLP against every candidate, which is a full linear scan of the catalog per request. The figure below draws the fork in the road that the choice of scorer creates at serving time.

![Branching diagram showing a user vector at request time splitting into a dot-product path that queries an approximate nearest neighbor index and returns top-K in about ten milliseconds versus an MLP-scorer path that must score all of a million items and whose latency grows with the catalog](/imgs/blogs/neural-collaborative-filtering-and-its-critique-6.png)

Here is what keeping the dot product buys you in code. Once you have a trained model that produces a user vector and item vectors via a dot product, retrieval is a few lines with FAISS:

```python
import faiss
import numpy as np

# item_vectors: (N_items, dim) float32, L2-normalized if you want cosine,
# raw if you want true inner product (MIPS).
dim = item_vectors.shape[1]
index = faiss.IndexFlatIP(dim)          # exact inner-product index (baseline)
index.add(item_vectors)                 # add all catalog items once

# For production scale, swap to an approximate index:
#   index = faiss.IndexHNSWFlat(dim, 32)        # HNSW graph, M=32
#   index = faiss.IndexIVFFlat(quantizer, dim, nlist)  # IVF cells
#   index.train(item_vectors); index.add(item_vectors)

def recommend(user_vec: np.ndarray, k: int = 10):
    scores, item_ids = index.search(user_vec.reshape(1, -1), k)  # one MIPS query
    return item_ids[0], scores[0]
```

There is no equivalent three-line `recommend` for an MLP scorer at catalog scale. The best you can do is *retrieve with a dot product* and then *re-score the few hundred survivors with the MLP* — which is, not coincidentally, exactly the retrieval-then-ranking funnel. The MLP, in other words, belongs in the ranking stage operating on a handful of candidates, never in the retrieval stage scoring the whole catalog. NCF's mistake, from a systems view, was proposing an MLP as the *scorer for everything*. Used as a re-ranker over a few hundred candidates the MLP is perfectly reasonable; the FLOP budget there is tiny. The lesson is not "never use an MLP," it is "do not put a learned scorer where you need maximum-inner-product search."

To make the latency consequence vivid, walk one request through both paths at a catalog of $N = 1{,}000{,}000$ items under a 10-millisecond retrieval budget. The dot-product path issues one HNSW query: the graph walk touches a few hundred nodes regardless of $N$, each a 64-dimensional dot product, totaling on the order of tens of thousands of FLOPs and returning in roughly 1 to 5 milliseconds with recall above 0.95 at the right `efSearch`. Comfortably inside budget, and — this is the part that matters — the cost is flat as the catalog grows from one million to one hundred million, because the index work is logarithmic. The MLP path has no such trick: $10^6$ forward passes at about 22,000 FLOPs each is $2.2 \times 10^{10}$ FLOPs per request, which even at 100 GFLOP/s of usable single-core throughput is over 200 milliseconds — twenty times the budget — and it *doubles* every time the catalog doubles. There is no `efSearch` knob to recover that; the only fix is to stop scoring the whole catalog, which means putting a dot-product retrieval stage in front, which means you needed the dot product after all. The funnel is not a design preference; it is the shape the latency budget forces on you the moment your scorer is not indexable.

## 9. Results: measuring all four on MovieLens-1M

Talk is cheap; let us measure. We follow the NCF paper's protocol exactly so the numbers are comparable to the literature.

**The leave-one-out protocol.** For each user, hold out their single most recent interaction as the test positive. At test time, pair that held-out positive with 99 randomly sampled items the user has not interacted with, giving 100 candidates. Score all 100 with the model, sort, and check where the true positive lands. Two metrics:

- **HitRate@10 (HR@10):** the fraction of users whose held-out positive appears in the top 10 of the 100 candidates. It answers "did we surface the right item at all?"
- **NDCG@10:** like HR@10 but discounted by rank — a hit at position 1 is worth more than a hit at position 10. With a single relevant item at rank $r$ (when $r \le 10$), $\mathrm{NDCG} = 1 / \log_2(r + 1)$, else 0, averaged over users.

A crucial caveat the series harps on, and that the [offline-online gap](/blog/machine-learning/recommendation-systems/the-offline-online-gap-and-why-your-metric-lied) post makes central: this **sampled** evaluation (100 candidates, not the full catalog) is convenient but known to be *inconsistent*. Rank a positive against 99 random negatives and you flatter every model, because random negatives are easy; the KDD 2020 result by Krichene and Rendle, "On Sampled Metrics for Item Recommendation," showed that sampled HR and NDCG can even reorder which model looks best relative to full-catalog metrics. We report sampled metrics here for comparability with NCF, but you should compute full-catalog metrics before you trust a model selection.

Here is the evaluation harness:

```python
import numpy as np

def evaluate(model, test_pos, user_neg99, device, k=10):
    """test_pos: dict u -> held-out item. user_neg99: dict u -> list of 99 negs."""
    model.eval()
    hits, ndcgs = [], []
    with torch.no_grad():
        for u, pos_item in test_pos.items():
            items = np.array([pos_item] + user_neg99[u])   # 100 candidates
            uu = torch.full((100,), u, dtype=torch.long, device=device)
            ii = torch.tensor(items, dtype=torch.long, device=device)
            scores = model(uu, ii).cpu().numpy()
            # rank of the true positive (index 0) among the 100
            order = np.argsort(-scores)              # descending
            rank = int(np.where(order == 0)[0][0])  # 0-based rank of positive
            if rank < k:
                hits.append(1.0)
                ndcgs.append(1.0 / np.log2(rank + 2))   # +2: rank is 0-based
            else:
                hits.append(0.0); ndcgs.append(0.0)
    return float(np.mean(hits)), float(np.mean(ndcgs))
```

And the well-tuned baseline — the part the original paper skipped. We train a dot-product MF with BPR in the `implicit` library, then sweep the factors and regularization, because *that sweep is the entire point*:

```python
import implicit
from scipy.sparse import csr_matrix

# user_item: csr_matrix of confidence-weighted implicit interactions
best = None
for factors in (32, 64, 128):
    for reg in (1e-3, 1e-2, 1e-1):
        model = implicit.bpr.BayesianPersonalizedRanking(
            factors=factors, regularization=reg, iterations=100,
            learning_rate=0.05, random_state=0,
        )
        model.fit(user_item, show_progress=False)
        hr, ndcg = eval_implicit(model, test_pos, user_neg99, k=10)
        if best is None or hr > best[0]:
            best = (hr, ndcg, factors, reg)
print(f"tuned MF  HR@10 {best[0]:.3f}  NDCG@10 {best[1]:.3f}  "
      f"factors={best[2]} reg={best[3]}")
```

Below are representative results, consistent with the figures in the NCF paper and the Rendle et al. follow-up on MovieLens-1M under the leave-one-out, sampled-100 protocol. Treat the exact decimals as approximate and reproducible-to-within-noise, not as a leaderboard; the *pattern* is the point.

| Model | HR@10 | NDCG@10 | Scorer params | ANN-able | Relative serving cost |
|---|---|---|---|---|---|
| MF (under-tuned, as in NCF'17) | ~0.66 | ~0.39 | 0 | yes | 1x |
| **MF (well-tuned, BPR)** | **~0.70** | **~0.42** | **0** | **yes** | **1x** |
| GMF | ~0.69 | ~0.41 | ~k | with care | ~1x |
| MLP | ~0.69 | ~0.41 | ~11k | no | ~100x+ |
| NeuMF | ~0.71 | ~0.43 | ~11k + 2 tables | no | ~100x+ |

Read this table the way the 2020 paper wants you to. The under-tuned MF in the first row is the strawman: a baseline that looks clearly worse and makes the deep models shine. The well-tuned MF in the second row erases almost the entire gap — it is within a point or two of NeuMF, the best deep model, while costing *nothing* extra to serve and remaining ANN-able. NeuMF edges everyone on accuracy by a sliver, but it pays for that sliver with roughly two orders of magnitude more serving cost and no path to fast retrieval. The accuracy axis is a near-tie; the cost axis is a blowout. The results figure makes that asymmetry visual — accuracy clustered tight, serving cost split wide.

Training cost tilts the same direction. The dot-product MF, trained with BPR in `implicit`, runs the full MovieLens-1M factorization in well under a minute on a single CPU core because each update touches only two vectors. NeuMF, trained with BCE and a 4-to-1 negative ratio, must push five times as many examples (each positive plus four negatives) through a forward and backward pass over the MLP every epoch, on a GPU, for dozens of epochs, *plus* the two separate pretraining runs for GMF and MLP that it needs to reach its best numbers. In wall-clock terms that is the difference between a baseline you can re-tune ten times before lunch and a model whose single hyperparameter sweep is an overnight job. When you are iterating — and early in a recommender's life you iterate constantly — a model you can retrain in seconds is worth several points of HitRate you would have to wait hours for. The fast, cheap, accurate-enough baseline is not a consolation prize; it is the thing that lets you make progress.

![Results matrix on MovieLens-1M showing tuned matrix factorization, GMF, MLP, and NeuMF clustered within a few points on HitRate at 10 and NDCG at 10 while serving cost ranges from cheap nearest-neighbor retrieval for the dot product to expensive full scans for the MLP-based models](/imgs/blogs/neural-collaborative-filtering-and-its-critique-7.png)

#### How to measure this honestly

Three things separate a believable benchmark from a misleading one, and the NCF saga is a case study in all three. **One, tune both sides with the same budget.** If the deep model got a 50-trial hyperparameter search and the baseline got default settings, you measured tuning effort, not architecture. **Two, report full-catalog metrics, not just sampled-100.** Sampled metrics are fine for quick iteration but can reorder models; do the full-catalog NDCG before you commit. **Three, use a temporal split, not a random one.** Leave-one-out by recency (hold out each user's *latest* interaction) avoids leaking future information into training, which a random split does. Get any of these wrong and your "win" may not survive contact with production — the recurring theme of the [offline-online gap](/blog/machine-learning/recommendation-systems/the-offline-online-gap-and-why-your-metric-lied) post.

## 10. Stress-testing the decision

A near-tie on offline accuracy with a blowout on serving cost makes the choice look obvious, but a good decision survives stress. Walk the edge cases the way you would in a design review, because each one is a place a team has actually gone wrong.

**What if you have rich side features — context, user history, item metadata?** Then a bare user-item interaction model of any flavor (MF, GMF, MLP, NeuMF) is leaving signal on the table, and the right move is *not* NeuMF — it is a two-tower model whose towers ingest those features and still emit a dot-product-able embedding, or a feature-rich ranker like DeepFM in the ranking stage. NeuMF's MLP can only chew on the two id embeddings; it has no slot for "the user watched three thrillers this week" or "this item launched yesterday." If features are your edge, put them in the towers, not in a learned scorer over ids.

**What if the catalog is small — a few thousand items?** Then the serving argument weakens: a full scan over 3,000 items with an 11,000-parameter MLP is cheap, well within budget, and you *could* ship a learned scorer. But notice the accuracy is still a tie, so you have paid in training complexity, memory, and the NeuMF pretraining ritual for nothing. Even where the serving objection vanishes, the reproducibility objection stands: a tuned dot product matches it. Small catalogs are the *only* place a learned scorer over ids is even defensible, and it is still not better.

**What if you are at 100 million items?** Then the MLP scorer is not slow, it is *impossible* — there is no top-K without an index, and there is no index for an MLP. You must retrieve with a dot product and only then re-rank a few hundred survivors with anything fancy. At a hundred million items the question "dot product or learned scorer?" is not a trade-off; the dot product is the only thing that runs. This is the regime that retired NCF as a serious retrieval candidate.

**What happens when the offline metric rises but online engagement is flat?** This is the nightmare the [offline-online gap](/blog/machine-learning/recommendation-systems/the-offline-online-gap-and-why-your-metric-lied) post is built around, and it is exactly the situation the NCF sampled metric invites. A model can win sampled HR@10 against 99 random negatives — easy negatives the user would never click — and lose against the *real* distribution of candidates a retrieval stage actually serves, which are all plausible items. If your offline win came from a sampled metric, expect it to evaporate online; the fix is full-catalog offline metrics and an A/B test, never a sampled-metric leaderboard.

**What if most sampled negatives are false negatives?** With uniform sampling on a sparse catalog, a "negative" the user has not interacted with may simply be an item they have not *seen* yet, not one they dislike. The pointwise BCE happily trains the model to push these away, teaching it to suppress items the user might love. This hurts every model equally here (we use the same sampler for all four), so it does not change the comparison, but it caps how good *any* of them can get and is the reason production systems graduate to impression-based negatives and hard-negative mining, covered in the [implicit feedback](/blog/machine-learning/recommendation-systems/implicit-feedback-models-als-and-bpr) post.

The decision holds under every stress: keep the dot product for retrieval, reserve learned scorers for small-candidate ranking with rich features, and never trust a sampled-metric win. The one place the learned-scorer-over-ids design is even arguable — a tiny catalog — is the one place its serving disadvantage disappears, and it is *still* not more accurate than a tuned dot product. That is about as clean as an engineering verdict gets.

## 11. The deep-CF lineage and where the field went next

It helps to place NCF in the family tree, because the critique did not kill deep recommenders — it redirected them. The lineage splits cleanly on one question: does the model keep a dot product as the scorer, or does it learn the scorer? That single fork decides whether the model can do fast retrieval.

![Tree diagram of the collaborative filtering lineage splitting into a dot-product line containing matrix factorization and the two-tower retrieval model that stays nearest-neighbor compatible and a learned-scorer line containing GMF and the MLP and NeuMF scorers that require a full scan](/imgs/blogs/neural-collaborative-filtering-and-its-critique-8.png)

On the **dot-product line** you have classical matrix factorization and, descended from it, the [two-tower model](/blog/machine-learning/recommendation-systems/the-two-tower-model-for-retrieval): two deep encoders (one for the user, one for the item) that can be as nonlinear and feature-rich as you like, each producing an embedding, with a *plain dot product* as the final scorer. The two-tower model is the field's answer to NCF: it keeps every bit of deep expressiveness the MLP promised, but puts it *inside the towers* where it produces better embeddings, and leaves the scorer as a dot product so retrieval stays a MIPS problem solvable by ANN. That is the synthesis. You do not have to choose between deep and servable; you put the depth where it does not cost you the index.

On the **learned-scorer line** you have GMF (barely off the dot product), the MLP, and NeuMF. These are not useless — as *re-rankers* over a few hundred candidates they are reasonable, and the broader idea of feeding concatenated features through an MLP lives on in ranking models like Wide & Deep and DeepFM, which is exactly where a learned scorer belongs: the ranking stage, small candidate set, rich features. The mistake was only ever placing a learned scorer in the retrieval stage where you need an index. The field absorbed that lesson: the deep capacity moved into the towers and into the ranker, and the dot product stayed put as the retrieval scorer.

## 12. Case studies and real numbers

Four reference points anchor everything above to the literature and to shipped systems.

**He et al., 2017, "Neural Collaborative Filtering" (WWW).** The origin. On MovieLens-1M and Pinterest, under leave-one-out with 99 sampled negatives, NeuMF reported HR@10 and NDCG@10 above the MF baselines tested, with NeuMF benefiting from GMF-plus-MLP pretraining. The paper popularized the GMF/MLP/NeuMF design and the sampled leave-one-out evaluation protocol that the whole sub-field then adopted. Its lasting contribution is arguably the *framework and protocol*, not the conclusion that deep beats the dot product.

**Rendle, Krichene, Zhang, Anderson, 2020, "Neural Collaborative Filtering vs. Matrix Factorization Revisited" (RecSys).** The rebuttal. On the same datasets and protocol, a properly tuned dot-product MF matched or exceeded NeuMF, and a controlled experiment showed an MLP needs far more capacity and data to approximate a dot product than the dot product needs to compute it. The paper's recommendation is blunt and worth quoting in spirit: use the dot product as the default similarity for retrieval; if you want more capacity, add it to the embeddings, not the scorer. This is the intellectual foundation of the two-tower design.

**Dacrema, Cremonesi, Jannach, 2019, "Are We Really Making Progress? An Analysis of Neural Recommendation Approaches" (RecSys).** The broader reproducibility audit. Of a set of recently published neural recommenders, most could be matched or beaten by well-tuned simple baselines (nearest neighbors, linear models, plain MF) that the original papers had under-tuned. It named the systemic problem — weak baselines inflating apparent progress — and is the reason "tune your baseline" is now table stakes at the top venues. Best-paper-recognized precisely because the field needed to hear it.

**Krichene and Rendle, 2020, "On Sampled Metrics for Item Recommendation" (KDD).** The evaluation-methodology companion. It proved that sampled metrics (rank against a small set of sampled negatives, exactly the 99-negative protocol NCF used) are *inconsistent* estimators of full-catalog metrics — they can reorder which model looks best. The practical upshot: sampled HR/NDCG are fine for fast iteration but you must validate with full-catalog metrics before model selection. This is why the results table above is framed as approximate and why the honesty section insists on full-catalog evaluation.

The throughline across all four: the deep recommender literature of the late 2010s systematically *under-measured* its baselines, both by under-tuning them and by evaluating on metrics that flatter everyone. NCF was the most famous instance, not the worst. The corrective — tuned baselines, full-catalog metrics, temporal splits, dot products kept where retrieval needs them — is now the standard of care, and this series builds on that standard rather than relitigating it.

## 13. When to reach for a learned scorer (and when not to)

Decisive guidance, because the whole point of the critique is to save you from a costly default.

**Use a dot-product scorer (MF or two-tower) when you are doing retrieval over a large catalog — which is almost always.** Any time you must consider more candidates than you can afford to score one by one — thousands to billions of items, single-digit-millisecond budget — you need MIPS, and MIPS needs a dot product. This is the default for the retrieval stage of every production funnel. Do not give it up.

**Use a learned scorer (an MLP, Wide & Deep, DeepFM, DCN, DIN) only in the ranking stage, over a small candidate set.** Once retrieval has narrowed the field to a few hundred candidates, the FLOP budget per item explodes by four or five orders of magnitude, and a learned scorer with rich cross-features is exactly right. A learned scorer earns its keep when (a) the candidate set is small, (b) you have side features — context, sequence, cross-features — that a bare user-item dot product cannot use, and (c) the extra accuracy moves a metric you actually serve. None of those hold in retrieval; all can hold in ranking.

**Do not reach for NeuMF specifically, ever, as a first model.** It is the wrong shape: a learned scorer with no side features, pitched at the retrieval-style task where its lack of an index is fatal and its accuracy edge over a tuned dot product is within noise. If you want the simplest strong baseline, train a BPR or ALS matrix factorization and tune it; if you need more capacity, go to a two-tower model, not NeuMF. The historical importance of NeuMF is as a teaching object, not a production default.

**Do not trust any "deep beats X" claim where X was not tuned with an equal budget.** This generalizes beyond recommenders. Before you adopt a complex model on the strength of a benchmark, ask: was the baseline tuned as hard as the proposed model? Were full-catalog metrics used, or sampled ones? Was the split temporal? If the answer to any of these is no, the benchmark is evidence about effort, not architecture, and you should re-run it yourself before betting infrastructure on it.

## 14. Key takeaways

- **The dot product is a strong inductive bias, not a weakness.** Replacing it with a learned MLP scorer gains expressiveness on paper but, in practice, the MLP struggles to learn even the dot product itself and plateaus a hair below a tuned one.
- **GMF, MLP, and NeuMF are easy to build** — two or four embedding tables plus a linear or MLP head, trained with BCE and uniform negative sampling — but NeuMF needs GMF-plus-MLP pretraining to shine, a sign the fusion adds little over its parts.
- **A baseline you did not tune is not a baseline.** The original NCF win shrank to near-zero once the dot-product MF got a real hyperparameter search; the apparent edge was tuning effort, not architecture.
- **Universal approximation is about existence, not learnability.** A bigger hypothesis class contains the right answer but gives no guarantee gradient descent finds it from finite, skewed data — and recommender data is exactly that.
- **The dot product enables MIPS and fast ANN retrieval; a learned scorer does not.** The accuracy gap between them is a couple of points; the serving cost gap is orders of magnitude. That asymmetry decides the architecture.
- **Put the depth in the towers, not the scorer.** The two-tower model keeps every bit of deep expressiveness in the user and item encoders while leaving a dot product as the scorer, so retrieval stays an indexable nearest-neighbor problem.
- **Use a learned scorer only in ranking, over a small candidate set with rich features** — never in retrieval, where its lack of an index is disqualifying.
- **Evaluate honestly:** tune both sides equally, use full-catalog metrics not sampled ones, and split temporally. Sampled metrics can reorder which model looks best.
- **The NCF-versus-MF episode is a reproducibility lesson** the whole field absorbed: weak baselines and flattering metrics inflated apparent progress, and the corrective is now the standard of care.

## 15. Further reading

- Xiangnan He, Lizi Liao, Hanwang Zhang, Liqiang Nie, Xia Hu, Tat-Seng Chua, **"Neural Collaborative Filtering"**, WWW 2017 — the origin of GMF, MLP, and NeuMF and the sampled leave-one-out protocol.
- Steffen Rendle, Walid Krichene, Li Zhang, John Anderson, **"Neural Collaborative Filtering vs. Matrix Factorization Revisited"**, RecSys 2020 — the rebuttal showing a tuned dot-product MF matches NeuMF and that an MLP struggles to learn a dot product.
- Maurizio Ferrari Dacrema, Paolo Cremonesi, Dietmar Jannach, **"Are We Really Making Progress? An Analysis of Neural Recommendation Approaches"**, RecSys 2019 — the reproducibility audit that motivated tuning baselines.
- Walid Krichene, Steffen Rendle, **"On Sampled Metrics for Item Recommendation"**, KDD 2020 — why sampled HR/NDCG are inconsistent and you must validate with full-catalog metrics.
- The `implicit` library docs (BPR and ALS) and the FAISS docs (`IndexFlatIP`, `IndexHNSWFlat`, `IndexIVFFlat`) — the production tools for a dot-product baseline and its ANN serving.
- Within this series: [matrix factorization, the workhorse](/blog/machine-learning/recommendation-systems/matrix-factorization-the-workhorse), the [two-tower model for retrieval](/blog/machine-learning/recommendation-systems/the-two-tower-model-for-retrieval), [approximate nearest neighbor serving](/blog/machine-learning/recommendation-systems/approximate-nearest-neighbor-serving-faiss-hnsw-scann), the [offline-online gap](/blog/machine-learning/recommendation-systems/the-offline-online-gap-and-why-your-metric-lied), and the capstone [recommender systems playbook](/blog/machine-learning/recommendation-systems/the-recommender-systems-playbook).
