---
title: "Generative and Conversational Recommendation"
date: "2026-06-19"
publishDate: "2026-06-19"
description: "The recommendation frontier where the model generates an item's semantic ID instead of looking it up, and where a recommender becomes a multi-turn agent that elicits preferences, accepts critiques, and grounds every pick in the catalog."
tags:
  [
    "recommendation-systems",
    "recsys",
    "generative-retrieval",
    "semantic-ids",
    "rq-vae",
    "conversational-recommendation",
    "llm-agents",
    "tiger",
    "machine-learning",
    "transformers",
  ]
category: "machine-learning"
subcategory: "Recommendation Systems"
author: "Hiep Tran"
featured: true
readTime: 51
image: "/imgs/blogs/generative-and-conversational-recommendation-1.png"
---

A staff engineer I trust once described the entire retrieval stage of a modern recommender as "a very expensive `dict.get()`." You embed a query into a vector, you embed every item into a vector, you stuff a million item vectors into an approximate-nearest-neighbor index, and at serve time you do a fast lookup: given this query vector, hand me the nearest item vectors. It works, it is fast, it is what almost everyone ships. But it has a structural cost that nobody loves: the index is a second, separate artifact that grows linearly with the catalog, has to be rebuilt when items change, and is wholly disconnected from the model that produced the embeddings. A new item is invisible until you embed it and add it. The model knows nothing about the index; the index knows nothing about the model.

Two frontier ideas attack this from opposite directions. The first asks: what if the model could just *generate* the identifier of the item it wants, the way a language model generates the next word, with no separate index at all? That is **generative retrieval** — represent each item as a short sequence of discrete *semantic IDs* and have a transformer emit those tokens one at a time, constrained to land on a real item. The model's parameters *become* the index. The second asks: why is a recommender a one-shot oracle that returns a frozen list and waits for a click, when the most natural way humans get recommendations is a *conversation* — "something for a beach read," "cheaper," "more like the second one"? That is **conversational recommendation** — a multi-turn dialogue where an LLM agent elicits preferences, accepts critiques, explains its picks, and grounds everything in your actual catalog.

Figure 1 shows the first shift in miniature: on the left, the familiar embed-then-ANN lookup; on the right, the generative view where a decoder emits the semantic ID token by token. This post is the deep frontier treatment of both ideas. It builds directly on two earlier posts in this series — the autoencoder-to-quantization road in [autoencoders and the road to generative retrieval](/blog/machine-learning/recommendation-systems/autoencoders-and-the-road-to-generative-retrieval), and the LLM-in-the-loop survey in [LLMs for recommendation](/blog/machine-learning/recommendation-systems/llms-for-recommendation-llm4rec) — and pushes both to where the research actually is in 2026.

![A side by side comparison of embed then approximate nearest neighbor retrieval against generative retrieval where a transformer decoder emits an item semantic ID token by token with no separate index](/imgs/blogs/generative-and-conversational-recommendation-1.png)

By the end you will be able to explain exactly how RQ-VAE residual quantization turns an item content embedding into a coarse-to-fine tuple of codewords, why a transformer trained to generate the next item's semantic ID can replace an ANN index, how constrained beam search over a code trie guarantees valid items, how shared codes let a brand-new item inherit structure for free, and how to build a conversational recommender that calls catalog search as a tool and grounds its answers. We will write the RQ-VAE and the constrained decoder in PyTorch, sketch a full conversational loop against an LLM API, put generative retrieval head to head with a two-tower baseline on Amazon Beauty, and — because this is the honest series — say plainly where the frontier is worth it today and where a tuned two-tower plus ranker still wins.

## Where this sits in the funnel and the series

The spine of this whole series, set up in [what is a recommender system](/blog/machine-learning/recommendation-systems/what-is-a-recommender-system), is the **retrieval → ranking → re-ranking funnel** fed by the **serve → log → train** feedback loop, all read off the **offline-versus-online gap**. Both frontier ideas in this post live mostly in the *retrieval* stage, but they redraw it.

Generative retrieval is a candidate generator, the same slot occupied by [the two-tower model](/blog/machine-learning/recommendation-systems/the-two-tower-model-for-retrieval) and the [autoencoder family](/blog/machine-learning/recommendation-systems/autoencoders-and-the-road-to-generative-retrieval). It does the same job — turn a user context into a few hundred candidate items — but with a radically different mechanism. Instead of "encode the user to a vector, then find nearby item vectors," it is "encode the user to a sequence, then generate the discrete IDs of items to recommend." Everything downstream of retrieval — the [ranking model](/blog/machine-learning/recommendation-systems/the-ranking-model-ctr-prediction-foundations), [multi-task heads](/blog/machine-learning/recommendation-systems/multi-task-and-multi-objective-ranking-mmoe-ple), [calibration](/blog/machine-learning/recommendation-systems/calibration-and-the-prediction-you-can-trust), re-ranking for [diversity](/blog/machine-learning/recommendation-systems/beyond-accuracy-diversity-novelty-serendipity-coverage) — is unchanged. You still rank and re-rank; you just generate the candidates instead of looking them up.

Conversational recommendation, by contrast, changes the *product surface*. It is not a new layer of the funnel so much as a new way to enter it: a dialogue that repeatedly *re-runs* the funnel with progressively sharper intent. Each user turn produces a fresh, more constrained retrieval, and the LLM stitches the turns into a coherent experience. It leans hard on [LLM4Rec](/blog/machine-learning/recommendation-systems/llms-for-recommendation-llm4rec) and on [finetuning LLMs for recommendation in practice](/blog/machine-learning/recommendation-systems/finetuning-llms-for-recommendation-in-practice), because the agent in the loop is an LLM.

A quick vocabulary note so nobody is lost. A **semantic ID** is a short ordered tuple of small integers (codewords) that names an item, where similar items share leading codewords — unlike a random integer item ID, which carries no meaning. **ANN** is approximate nearest neighbor search, the fast index lookup at the heart of two-tower retrieval, covered in [ANN serving](/blog/machine-learning/recommendation-systems/approximate-nearest-neighbor-serving-faiss-hnsw-scann). **MIPS** is maximum-inner-product search, the math problem ANN solves. **Constrained decoding** means restricting the model's vocabulary at each generation step so the output is guaranteed valid (here: a real item). **Grounding** means tying an LLM's generated text to facts it can verify — a real catalog row — rather than letting it invent items. Keep those five handy; the post turns on them.

## Section 1 — The paradigm shift: from index to generation

Let us make the shift concrete, because the phrase "the parameters are the index" sounds like a slogan until you trace it through.

In two-tower retrieval, you train a query encoder $f_\theta$ and an item encoder $g_\phi$ so that the dot product $f_\theta(u) \cdot g_\phi(i)$ is high for items $i$ the user $u$ will engage with. At serve time you precompute $g_\phi(i)$ for every item, store those vectors in an ANN index (faiss, HNSW, ScaNN), and for a live user you compute $f_\theta(u)$ once and ask the index for the top-K nearest item vectors. The retrieval is a *search* problem: given a point, find nearby points. The index is an explicit data structure holding one vector per item. We covered the training side in [training the two-tower with negatives and sampled softmax](/blog/machine-learning/recommendation-systems/training-two-tower-negatives-and-sampled-softmax).

Generative retrieval throws out the search. It says: assign each item a *string* — a short sequence of discrete tokens — and train a sequence-to-sequence model that, given the user's history (also expressed as item strings), *generates* the string of the next item to recommend. There is no item vector to store and no nearest-neighbor search to run. Retrieval becomes a *generation* problem: given a context, produce a sequence. The "index" is now distributed across the transformer's weights, which have memorized which strings tend to follow which contexts.

This is not analogy; it is the literal mechanism. The seed idea comes from the **Differentiable Search Index** (Tay et al., *Transformer Memory as a Differentiable Search Index*, NeurIPS 2022), which showed for document retrieval that a single T5 model could map a query straight to a document ID by *generating* the ID, with retrieval accuracy competitive with dual-encoder-plus-ANN. DSI's provocative claim was exactly "the model parameters are the index." TIGER (Rajput et al., *Recommender Systems with Generative Retrieval*, NeurIPS 2023) brought the idea to recommendation and added the crucial ingredient DSI lacked for items: a principled way to *construct* the IDs so they carry meaning. That ingredient is the semantic ID, and it comes straight from the autoencoder road.

The reason this matters is not novelty for its own sake. It is three concrete properties, which I will defend rather than assert across the post:

1. **Memory that does not grow with the catalog.** The "index" is a handful of small codebooks (a few thousand vectors total) plus the decoder weights. A two-tower index grows linearly: a million items at 256 float32 dimensions is roughly a gigabyte before quantization.
2. **Generalization to new items via shared codes.** A new item whose content embedding quantizes to codes that overlap existing items inherits their generative structure. It can be *generated* the moment its semantic ID is assigned — no index rebuild, no re-embedding pass over neighbors.
3. **A natural bridge to LLMs.** If items are sequences of tokens, then a language model can reason about them, recommend them, and explain them in the same decoding loop it uses for words. Generative retrieval and conversational recommendation are the *same* underlying capability viewed at two scales.

There is no free lunch, and I will be equally concrete about the costs: autoregressive decoding with beam search is slower than an ANN lookup; constructing good semantic IDs is finicky; code collisions (two items quantizing to the same tuple) must be handled; and catalog churn stresses the codebooks. We will get to all of it.

#### Worked example: the memory math, item by item

Let me make property 1 concrete with numbers, because "memory does not grow with the catalog" is the kind of claim that deserves arithmetic. Take a catalog of 1,000,000 items and a two-tower retriever with 256-dimensional float32 item embeddings. The raw item index is $10^6 \times 256 \times 4$ bytes $= 1.024 \times 10^9$ bytes, just over 1 GB, before any ANN overhead (HNSW graph links typically add another 30 to 60 percent). Quantize the vectors with product quantization and you can shrink it perhaps 8x to roughly 128 MB, at some recall cost. Either way, the index scales *linearly*: ten million items is ten times the memory.

Now the generative side. The "index" is the RQ-VAE codebooks plus the trie. The codebooks are $L \times K$ vectors of dimension $d$: for $L = 3, K = 256, d = 32$ that is $3 \times 256 \times 32 \times 4$ bytes $\approx 98$ KB — kilobytes, not gigabytes. The trie of a million 3-level IDs is a few million edges, on the order of 10 to 30 MB depending on representation. The decoder weights (a few million parameters at 4 bytes) add roughly 10 to 40 MB. Total: tens of megabytes, and — the key point — the codebooks and decoder do *not* grow when you add items. Only the trie grows, and far more slowly than a dense vector store, because a trie shares prefixes. Going from one million to ten million items barely moves the codebook and decoder memory; the two-tower index grows by a full order of magnitude. That asymmetry is the entire memory argument, and it is real.

## Section 2 — Building semantic IDs with RQ-VAE

Everything in generative retrieval rests on the semantic ID, so we build it carefully. The goal: take an item's content embedding — say a 768-dimensional vector from encoding its title and description with a sentence encoder — and turn it into a short tuple of small integers, $(c_1, c_2, c_3)$, where each $c_\ell \in \{0, \dots, K-1\}$ indexes into a codebook of $K$ learned vectors (TIGER uses $K = 256$ per level and three or four levels). Crucially, similar items should share leading codewords, so the tuple is *semantic*, not random.

The machine that does this is **RQ-VAE** — residual-quantized variational autoencoder. We met its lineage in [autoencoders and the road to generative retrieval](/blog/machine-learning/recommendation-systems/autoencoders-and-the-road-to-generative-retrieval); here we make the quantizer precise. Figure 2 shows the flow: a content embedding goes into an encoder, the encoder output is quantized against the first codebook, the *residual* is quantized against the second, that residual against the third, and the three chosen indices form the semantic ID.

![A branching dataflow showing an item content embedding passing through an RQ-VAE encoder then being quantized against three successive residual codebooks whose indices merge into a coarse to fine semantic ID tuple](/imgs/blogs/generative-and-conversational-recommendation-2.png)

### The science: residual vector quantization

Start with plain **vector quantization** (VQ), the machinery of VQ-VAE (van den Oord et al., *Neural Discrete Representation Learning*, NeurIPS 2017). You have a codebook $C = \{e_1, \dots, e_K\}$ of $K$ vectors in $\mathbb{R}^d$. Given an encoder output $z \in \mathbb{R}^d$, VQ picks the nearest codebook vector:

$$
c = \arg\min_{k} \lVert z - e_k \rVert_2^2, \qquad \hat z = e_c .
$$

The index $c$ is the discrete code; $\hat z$ is the quantized reconstruction. One codebook of size $K$ can represent only $K$ distinct vectors, so a single VQ level is far too coarse for a million-item catalog.

**Residual quantization** fixes this by stacking $L$ codebooks and quantizing the *residual* at each level. Let $r_0 = z$. At level $\ell = 1, \dots, L$ with codebook $C_\ell = \{e_1^{(\ell)}, \dots, e_K^{(\ell)}\}$:

$$
c_\ell = \arg\min_{k} \big\lVert r_{\ell-1} - e_k^{(\ell)} \big\rVert_2^2, \qquad r_\ell = r_{\ell-1} - e_{c_\ell}^{(\ell)} .
$$

The reconstruction is the sum of the chosen codewords, $\hat z = \sum_{\ell=1}^{L} e_{c_\ell}^{(\ell)}$, and the semantic ID is the tuple $(c_1, \dots, c_L)$. The first codebook captures the coarse structure (which broad cluster the item lives in), and each subsequent codebook refines what the previous ones missed. With $L$ levels of size $K$ you can represent $K^L$ distinct tuples — for $K=256, L=3$ that is over 16 million, comfortably more than most catalogs, while storing only $3 \times 256 = 768$ codebook vectors.

Why does this make the ID *semantic*? Because the coarse-to-fine structure means two items with similar content embeddings will, with high probability, pick the *same* first codeword $c_1$ (same broad cluster), and likely the same $c_2$, diverging only at the finest level. So the tuple has a tree structure: items sharing a prefix are semantically close. That prefix-sharing is exactly what lets generation generalize and what the trie in Section 4 exploits.

### The science: the training objective

RQ-VAE is trained end to end with two losses. First, **reconstruction**: a decoder $D$ maps $\hat z$ back to the input content embedding $x$, and we minimize $\lVert x - D(\hat z) \rVert_2^2$ so the quantized code retains the item's meaning. Second, the **commitment / codebook loss** that makes VQ trainable despite the non-differentiable $\arg\min$. Following VQ-VAE with the straight-through estimator, the total loss summed over levels is

$$
\mathcal{L}_{\text{RQ-VAE}} = \lVert x - D(\hat z) \rVert_2^2 \;+\; \sum_{\ell=1}^{L} \Big( \big\lVert \text{sg}[r_{\ell-1}] - e_{c_\ell}^{(\ell)} \big\rVert_2^2 \;+\; \beta \big\lVert r_{\ell-1} - \text{sg}[e_{c_\ell}^{(\ell)}] \big\rVert_2^2 \Big),
$$

where $\text{sg}[\cdot]$ is the stop-gradient operator and $\beta$ (typically $0.25$) weights the commitment term that pulls the encoder output toward its chosen codeword. The first squared term inside the sum moves the codebook vector toward the residual it represents (the "codebook loss"); the second keeps the encoder from running away from the codebook (the "commitment loss"). The gradient of the reconstruction loss reaches the encoder through the straight-through trick: in the backward pass we copy the gradient at $\hat z$ straight to $z$ as if quantization were the identity.

A practical wrinkle that bites everyone: **codebook collapse**, where only a few of the $K$ codewords ever get used and the rest are dead. The standard fixes — used by RQ-VAE and by TIGER — are k-means initialization of each codebook from a batch of encoder outputs, and periodic *reset* of dead codewords to random encoder outputs (or exponential-moving-average codebook updates). Without these, a 256-entry codebook can collapse to a dozen live entries and your semantic IDs lose almost all their resolution.

### Practical: RQ-VAE in PyTorch

Here is a compact but real RQ-VAE quantizer and the construction loop. The encoder and decoder are small MLPs; the heart is the residual quantization.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualQuantizer(nn.Module):
    def __init__(self, n_levels=3, codebook_size=256, dim=32, beta=0.25):
        super().__init__()
        self.n_levels = n_levels
        self.beta = beta
        # one codebook per level: (codebook_size, dim)
        self.codebooks = nn.ModuleList(
            nn.Embedding(codebook_size, dim) for _ in range(n_levels)
        )
        for cb in self.codebooks:
            nn.init.normal_(cb.weight, std=0.02)

    def quantize_level(self, residual, cb):
        # residual: (B, dim); cb.weight: (K, dim)
        dists = torch.cdist(residual, cb.weight)        # (B, K)
        idx = dists.argmin(dim=1)                        # (B,)
        chosen = cb(idx)                                 # (B, dim)
        return idx, chosen

    def forward(self, z):
        residual = z
        codes, quantized = [], torch.zeros_like(z)
        cb_loss = 0.0
        for cb in self.codebooks:
            idx, chosen = self.quantize_level(residual, cb)
            # codebook loss + commitment loss with stop-gradients
            cb_loss = cb_loss + F.mse_loss(chosen, residual.detach())
            cb_loss = cb_loss + self.beta * F.mse_loss(residual, chosen.detach())
            # straight-through: copy gradient past the argmin
            quantized = quantized + residual + (chosen - residual).detach()
            residual = residual - chosen.detach()
            codes.append(idx)
        return quantized, torch.stack(codes, dim=1), cb_loss   # (B, dim), (B, L), scalar

class RQVAE(nn.Module):
    def __init__(self, in_dim=768, dim=32, **rq_kwargs):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(), nn.Linear(256, dim))
        self.decoder = nn.Sequential(
            nn.Linear(dim, 256), nn.ReLU(), nn.Linear(256, in_dim))
        self.rq = ResidualQuantizer(dim=dim, **rq_kwargs)

    def forward(self, x):
        z = self.encoder(x)
        z_q, codes, cb_loss = self.rq(z)
        x_hat = self.decoder(z_q)
        recon = F.mse_loss(x_hat, x)
        return recon + cb_loss, codes, recon
```

To turn a catalog into semantic IDs you train this on the matrix of content embeddings, then run inference to read off the codes:

```python
# content_emb: (n_items, 768) sentence-encoder embeddings of title + description
model = RQVAE(in_dim=768, n_levels=3, codebook_size=256, dim=32)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

for epoch in range(50):
    for batch in loader(content_emb, bs=1024):      # your own batcher
        loss, _, recon = model(batch)
        opt.zero_grad(); loss.backward(); opt.step()

model.eval()
with torch.no_grad():
    _, codes, _ = model(content_emb)                # (n_items, 3) int codes
semantic_ids = [tuple(row.tolist()) for row in codes]
```

One detail TIGER added that matters in practice: when two items collide on the same $(c_1, c_2, c_3)$ tuple, append a fourth disambiguating token (a small counter) so each item has a unique ID. Collisions are common — at $K^L$ buckets and a million items, the birthday-problem math guarantees thousands of them — and the extra token keeps the mapping invertible.

### The science: collisions and codebook utilization, quantified

It is worth doing the collision arithmetic, because it drives the codebook-size decision. With $B = K^L$ distinct buckets and $n$ items hashed roughly uniformly, the expected number of *colliding* items (items sharing a tuple with at least one other) follows the birthday-collision estimate. The expected count of buckets holding at least two items is approximately $B\,(1 - e^{-n/B}(1 + n/B))$, and for $n \ll B$ the expected number of *pairwise* collisions is about $\binom{n}{2}/B \approx n^2 / (2B)$. For $K = 256, L = 3$ we have $B \approx 1.68 \times 10^7$; a catalog of $n = 10^5$ items gives roughly $10^{10} / (3.4 \times 10^7) \approx 294$ pairwise collisions, modest. Push to $n = 10^6$ items and it jumps to roughly $2.9 \times 10^4$ collisions — tens of thousands of items needing the disambiguating fourth token. Add a fourth level ($B \approx 4.3 \times 10^9$) and collisions at a million items drop back to single digits. So the level count is not a free hyperparameter: it trades collision rate against decoding steps (more levels means more autoregressive steps per item, more latency).

The other quantity to watch is **codebook utilization** — the fraction of the $K$ entries per level that are actually used. A healthy RQ-VAE uses nearly all of them; a collapsed one uses a handful, which silently shrinks your effective bucket count from $K^L$ to (live entries)$^L$ and inflates collisions. Measure it directly after training and reset dead codes if it is low.

### Practical: codebook maintenance and a real training step

Here is the training step with the two production-critical additions — k-means initialization and dead-code reset — that the compact version above omitted. Without these, codebook collapse quietly halves your ID resolution.

```python
import torch

@torch.no_grad()
def kmeans_init_codebooks(model, sample_z, iters=10):
    """Initialize each codebook from a sample of encoder outputs."""
    for cb in model.rq.codebooks:
        K = cb.num_embeddings
        idx = torch.randperm(sample_z.size(0))[:K]
        centers = sample_z[idx].clone()
        for _ in range(iters):
            assign = torch.cdist(sample_z, centers).argmin(1)
            for k in range(K):
                pts = sample_z[assign == k]
                if len(pts):
                    centers[k] = pts.mean(0)
        cb.weight.data.copy_(centers)

@torch.no_grad()
def reset_dead_codes(model, batch_z, usage, threshold=1):
    """Re-seed codebook entries that fired fewer than `threshold` times."""
    for level, cb in enumerate(model.rq.codebooks):
        dead = (usage[level] < threshold).nonzero().flatten()
        if len(dead):
            seeds = batch_z[torch.randint(0, batch_z.size(0), (len(dead),))]
            cb.weight.data[dead] = seeds + 1e-3 * torch.randn_like(seeds)

# one epoch with utilization tracking
usage = [torch.zeros(256) for _ in range(3)]
for step, batch in enumerate(loader(content_emb, bs=1024)):
    loss, codes, recon = model(batch)               # codes: (B, 3)
    opt.zero_grad(); loss.backward(); opt.step()
    for lvl in range(3):                             # accumulate code usage
        usage[lvl] += torch.bincount(codes[:, lvl], minlength=256)
    if step % 200 == 0 and step > 0:
        with torch.no_grad():
            z = model.encoder(batch)
        reset_dead_codes(model, z, usage)
        usage = [torch.zeros(256) for _ in range(3)]  # reset window
```

The `reset_dead_codes` step is the difference between a codebook that uses 250 of 256 entries and one that collapses to a dozen. Track utilization as a first-class training metric, the way you track loss; a sudden drop is the earliest warning that your semantic IDs are degenerating.

## Section 3 — Generation replaces the index

Now the second half of generative retrieval: train a sequence model to *generate* semantic IDs, and decode under constraints so it only ever produces real items.

The setup is sequence-to-sequence, exactly like the original TIGER. Express the user's history as a flat sequence of code tokens: if the user interacted with items whose semantic IDs are $(12, 5, 200)$, $(47, 88, 3)$, $(12, 9, 130)$, the input sequence is the concatenation `12 5 200 47 88 3 12 9 130`. The *target* is the semantic ID of the next item, say `31 17 64`. A T5-style encoder-decoder (TIGER used a small T5) is trained with the standard autoregressive cross-entropy to generate the target ID token by token. We covered the sequence-modeling foundations in [self-attention for sequences](/blog/machine-learning/recommendation-systems/self-attention-for-sequences-sasrec-bert4rec).

### The science: the autoregressive objective over ID tokens

The model factorizes the probability of the next item's semantic ID $(c_1, \dots, c_L)$ given history $h$ as a product of per-token conditionals:

$$
P(c_1, \dots, c_L \mid h) = \prod_{\ell=1}^{L} P\big(c_\ell \mid c_{<\ell}, h\big),
$$

and we minimize the negative log-likelihood over training pairs:

$$
\mathcal{L} = -\sum_{(h, c)} \sum_{\ell=1}^{L} \log P\big(c_\ell \mid c_{<\ell}, h\big).
$$

This is precisely a language-model objective, with a tiny vocabulary: $L$ codebooks of $K$ entries each plus a few special tokens. The vocabulary is the union of the per-level codebooks (TIGER uses *separate* token ranges per level, so token "5 at level 1" is a different vocabulary entry from "5 at level 2"; this lets the model learn that the first token is coarse and the third is fine).

Here is the subtle, beautiful part. At inference, recommending top-K items is *beam search* with beam width $K$. The decoder generates the most probable ID sequences, and each completed sequence — if it is a valid item ID — is a recommendation, ranked by its generation log-probability. There is no dot product, no ANN, no item vector anywhere in the loop. The ranking among candidates falls out of the language model's own probabilities.

But unconstrained beam search has a fatal flaw: nothing stops it from generating a tuple like $(12, 5, 99)$ that does not correspond to any real item. That is where the trie comes in.

### The science: why constrained decoding is mandatory

The set of valid semantic IDs is a tiny, structured subset of all $K^L$ possible tuples — only the ones actually assigned to catalog items. Free generation will frequently wander off this set, especially at the fine-grained final token. The fix is **constrained beam search over a prefix trie** of valid item codes. Figure 3 shows the idea: build a trie where each path from root to leaf spells out a real item's semantic ID; at each decoding step, mask the logits so the model can only choose codewords that continue an existing path; any beam that would step off the trie is pruned.

![A branching trie diagram where a decoder selects codeword tokens that continue valid paths while an invalid continuation is pruned so the completed path spells a real catalog item identifier](/imgs/blogs/generative-and-conversational-recommendation-3.png)

Formally, let $\text{Trie}$ be the set of valid prefixes. At step $\ell$ with prefix $c_{<\ell}$, the allowed next tokens are $A(c_{<\ell}) = \{ k : (c_{<\ell}, k) \in \text{Trie} \}$. We renormalize the conditional only over allowed tokens:

$$
P_{\text{constrained}}(c_\ell = k \mid c_{<\ell}, h) = \frac{\mathbb{1}[k \in A(c_{<\ell})] \, \exp(z_{\ell,k})}{\sum_{j \in A(c_{<\ell})} \exp(z_{\ell,j})},
$$

where $z_{\ell,k}$ are the decoder logits. This guarantees every completed beam is a real item *by construction*. It also speeds decoding: the allowed set is usually tiny (a prefix shared by a handful of items), so the effective branching factor is small after the first token.

### Practical: a constrained decoder over a code trie

Here is a minimal but faithful constrained beam search. It assumes you already have a trained decoder exposing `next_token_logits(history, prefix)` and the trie of valid IDs.

```python
import torch
from collections import defaultdict

def build_trie(semantic_ids):
    """semantic_ids: list of tuples. Returns prefix -> set of allowed next tokens."""
    trie = defaultdict(set)
    for sid in semantic_ids:
        for i in range(len(sid)):
            trie[sid[:i]].add(sid[i])
    valid = set(semantic_ids)
    return trie, valid

def constrained_beam_search(model, history, trie, valid, n_levels, beam=10):
    # each beam: (prefix_tuple, cumulative_logprob)
    beams = [((), 0.0)]
    for level in range(n_levels):
        candidates = []
        for prefix, score in beams:
            allowed = trie.get(prefix, set())
            if not allowed:
                continue
            logits = model.next_token_logits(history, prefix)   # (vocab,)
            mask = torch.full_like(logits, float("-inf"))
            idx = torch.tensor(sorted(allowed))
            mask[idx] = logits[idx]
            logprobs = torch.log_softmax(mask, dim=-1)
            for tok in allowed:
                candidates.append((prefix + (tok,), score + logprobs[tok].item()))
        # keep the top `beam` partial sequences
        candidates.sort(key=lambda x: x[1], reverse=True)
        beams = candidates[:beam]
    # only completed, valid items survive
    recs = [(p, s) for p, s in beams if p in valid]
    recs.sort(key=lambda x: x[1], reverse=True)
    return recs   # list of (semantic_id, logprob), ranked
```

The trie itself is cheap: for a million items at three levels it is a few million edges, a handful of megabytes — orders of magnitude smaller than a float vector index, and it is the *only* per-item structure you keep. Compare that to the two-tower index that stores a dense vector per item.

#### Worked example: tracing one constrained generation

Let me trace a single item's generation by hand so the mechanism is unambiguous. Suppose three items have semantic IDs: item A = $(12, 5, 200)$, item B = $(12, 5, 211)$, item C = $(47, 5, 200)$. The trie has root children $\{12, 47\}$; under prefix $(12)$ the allowed second tokens are $\{5\}$; under $(12, 5)$ the allowed third tokens are $\{200, 211\}$.

A user has history that the decoder reads. Step 1: the decoder's logits over level-1 tokens give, after masking to the allowed $\{12, 47\}$ and softmax, $P(12) = 0.7$, $P(47) = 0.3$. With beam width 2 we keep both prefixes: $(12)$ at log-prob $\log 0.7 = -0.357$ and $(47)$ at $\log 0.3 = -1.204$. Step 2: under $(12)$ only token $5$ is allowed, so $P(5 \mid 12) = 1.0$ and the prefix becomes $(12, 5)$ at unchanged $-0.357$; under $(47)$ only $5$ is allowed too, giving $(47, 5)$ at $-1.204$. Step 3: under $(12, 5)$ tokens $\{200, 211\}$ are allowed; say the decoder gives $P(200) = 0.6, P(211) = 0.4$, so item A finishes at $-0.357 + \log 0.6 = -0.868$ and item B at $-0.357 + \log 0.4 = -1.273$; under $(47, 5)$ only $200$ is allowed, so item C finishes at $-1.204 + 0 = -1.204$.

Final ranking by log-probability: item A ($-0.868$) > item C ($-1.204$) > item B ($-1.273$). Notice three things. First, every completed beam is a real item — the trie made invalid tuples like $(12, 5, 99)$ impossible. Second, the ranking emerged from the language model's own probabilities, no dot product anywhere. Third, item C beat item B despite a weaker first token, because its only valid completion was certain — exactly the kind of structured competition the trie creates.

## Section 4 — Why shared codes generalize, and other properties

The single most attractive property of generative retrieval is **new-item generalization**, and it is worth understanding precisely, because it is the argument that justifies the whole approach over a two-tower index.

Consider launching a new item with no interaction history at all — the cold-start case we dissect in [the cold-start problem](/blog/machine-learning/recommendation-systems/the-cold-start-problem). In two-tower land, the new item's ID embedding is random noise until it accrues clicks; it is effectively invisible to retrieval. You can patch this with content features, but the patch is exactly that.

In generative retrieval, the new item gets a semantic ID *the moment it exists*, computed from its content embedding through the (already-trained) RQ-VAE. And because the RQ-VAE is semantic, that ID *shares codewords* with existing items in the same niche. Suppose the new item is a sci-fi novel; its content embedding quantizes to $(12, 5, 142)$, sharing the prefix $(12, 5)$ with dozens of other sci-fi items the decoder has seen during training. The decoder has learned that, given a user who reads sci-fi, the prefix $(12, 5, \cdot)$ is highly probable. It can *generate* $(12, 5, 142)$ for that user even though it never saw item $(12, 5, 142)$ in any training sequence — it inherits the generative structure of its prefix-siblings. All you do at serve time is add the new ID to the trie so constrained decoding will accept it. No retraining, no re-embedding pass.

This is the deep reason the shared-prefix structure of semantic IDs matters. It is not a cosmetic property; it is the mechanism by which the model generalizes to items it has never been trained on. A flat random ID has no neighbors, so it inherits nothing; a semantic ID is born into a neighborhood.

The trade-offs in this section deserve a comparison. Figure 4 lays the two retrieval paradigms side by side across the four axes that decide which to ship.

![A two by four comparison matrix contrasting two-tower with approximate nearest neighbor against generative retrieval across the index, memory, new-item, and latency decision axes](/imgs/blogs/generative-and-conversational-recommendation-4.png)

| Property | Two-tower + ANN | Generative retrieval (TIGER-style) |
| --- | --- | --- |
| Retrieval mechanism | embed query, MIPS lookup | autoregressive decode of semantic ID |
| Index artifact | separate ANN store, one vector per item | code trie + decoder weights |
| Memory scaling | linear in catalog (gigabytes at 1M items) | small fixed codebooks (megabytes) |
| New item | needs embedding + index add; ID emb is noise | gets semantic ID from content; shared codes generalize |
| Adding an item | re-embed + insert into ANN | recompute ID + insert into trie |
| Latency | one forward + fast ANN, p99 single-digit ms | beam search, several decoder steps, p99 tens of ms |
| Maturity | battle-tested at billions of items | research-to-early-production, mostly sub-100M items |
| Diversity control | re-ranking step | also via decoding temperature / sampling |

The honest summary: generative retrieval wins decisively on memory and new-item generalization, and offers a novelty knob (decoding temperature) for free. It loses on latency and maturity. At very large catalogs the latency and code-management costs currently dominate, which is why production deployments so far skew toward catalogs in the millions rather than billions.

That novelty knob deserves a sentence, because it is a genuinely nice property that ANN retrieval lacks. In a two-tower system, the candidate set is deterministic given the query — the ANN returns the same nearest neighbors every time — so diversity has to be injected by a separate re-ranking stage. In generative retrieval, the candidates come from *sampling* a probability distribution, so you have a temperature dial. Decode with low temperature (or greedy beam search) for the safest, most-probable completions; raise the temperature to sample lower-probability semantic IDs and surface more novel, serendipitous items, at the cost of some precision. TIGER reported exactly this controllable trade-off. It is the same temperature knob a language model uses to trade coherence for creativity, applied to recommendation — diversity becomes a decoding hyperparameter rather than a bolted-on re-ranking heuristic. You still want a re-ranking stage for the funnel's other goals (business rules, freshness, fairness), but the *first* lever for novelty is free.

### LIGER, LC-Rec, and the convergence

TIGER is not the end of the story. Two threads pull generative and embedding-based retrieval back together. **LIGER** (Yang et al., 2024) is explicit about it: it argues that pure generative retrieval underperforms dense retrieval on *seen* items but excels on *cold* items, and so it *hybridizes* — combining a dense embedding component with the generative semantic-ID component to get the best of both, dense recall on warm items and generative generalization on cold ones. **LC-Rec** (Zheng et al., 2024) tackles a different gap: it aligns the semantic IDs with an LLM's *language* space through a set of alignment tasks, so a general-purpose LLM can reason about items via their semantic IDs and you get language understanding and collaborative signal in one model. The direction of travel is clear: retrieval and generation are converging, and the semantic ID is the shared currency. The pure-generative-vs-pure-dense debate is dissolving into "use semantic IDs as the item representation, and decode or look up as the latency budget allows."

## Section 5 — The generative-retrieval pipeline end to end

Let us assemble the full pipeline so the moving parts are clear. Figure 5 shows the four stages stacked: content embedding, RQ-VAE quantization into semantic IDs, decoder training on history-to-next-ID pairs, and constrained decoding at serve time.

![A four stage vertical stack of the generative retrieval pipeline from content embedding through RQ-VAE semantic ID construction to decoder training and constrained beam search serving](/imgs/blogs/generative-and-conversational-recommendation-5.png)

**Stage 1 — content embeddings.** Encode each item's text (title, description, attributes) with a sentence encoder. This is what gives the eventual IDs their meaning; an item with no content is hard to place semantically, which is a real limitation of the approach for content-poor catalogs.

**Stage 2 — semantic IDs.** Train the RQ-VAE from Section 2 on the content-embedding matrix, read off the codes, disambiguate collisions with an extra token. You now have a `dict` from item to semantic-ID tuple, and its inverse trie.

**Stage 3 — train the decoder.** Build training pairs from interaction sequences: input = history as code tokens, target = next item's code tokens. Train a small T5 with the autoregressive cross-entropy from Section 3. This is the only large training job, and it is comparable in cost to training a SASRec-style sequence model.

Here is the data-prep and training skeleton with the per-level token offsetting that keeps "code 5 at level 1" distinct from "code 5 at level 2" — a detail that materially helps the model learn the coarse-to-fine structure:

```python
from transformers import T5Config, T5ForConditionalGeneration
import torch

L, K = 3, 256
# per-level token ranges: level 0 -> [0,256), level 1 -> [256,512), level 2 -> [512,768)
def encode_id(sid):                      # sid: (c1, c2, c3)
    return [c + lvl * K for lvl, c in enumerate(sid)]

def build_pairs(user_sequences, item2sid, max_hist=20):
    """Each pair: history tokens -> next item's id tokens."""
    pairs = []
    for seq in user_sequences:           # seq: list of item ids, chronological
        for t in range(1, len(seq)):
            hist = seq[max(0, t - max_hist):t]
            src = [tok for it in hist for tok in encode_id(item2sid[it])]
            tgt = encode_id(item2sid[seq[t]])
            pairs.append((src, tgt))
    return pairs

cfg = T5Config(vocab_size=L * K + 8, d_model=128, num_layers=4,
               num_decoder_layers=4, num_heads=4, d_ff=1024)
decoder = T5ForConditionalGeneration(cfg)
opt = torch.optim.AdamW(decoder.parameters(), lr=3e-4)

for epoch in range(200):
    for src, tgt in pair_loader(pairs, bs=256):     # padded tensors + attn masks
        out = decoder(input_ids=src.input_ids,
                      attention_mask=src.attention_mask,
                      labels=tgt.input_ids)          # T5 shifts labels internally
        opt.zero_grad(); out.loss.backward(); opt.step()
```

The model is tiny by LLM standards — a few million parameters — because the vocabulary is only $L \times K \approx 768$ tokens, not 50,000. That small vocabulary is exactly why generative retrieval is cheap to train relative to a general LLM, even though it borrows the same architecture.

**Stage 4 — serve.** For a live user, encode the history into code tokens, run constrained beam search over the trie with beam width $K$, return the top-K valid items ranked by generation log-probability. Hand those candidates to your existing ranker.

A subtlety worth flagging: **catalog churn**. The RQ-VAE codebooks are trained on a snapshot of content. If the catalog drifts substantially (new categories, new languages), the codebooks can become stale and new items quantize poorly. The maintenance cadence is therefore: cheap, frequent updates (assign IDs to new items with the frozen RQ-VAE, add to trie) and occasional, expensive refreshes (retrain the RQ-VAE and re-mint all IDs, then retrain or warm-start the decoder). This is a genuinely different operational profile from a two-tower system, where you continuously re-embed and the ANN index just absorbs it. Neither is strictly easier; they fail differently.

### Measuring it honestly

To compare generative retrieval against a two-tower baseline without fooling yourself, follow the same discipline we set out in [the right way to split and evaluate](/blog/machine-learning/recommendation-systems/the-right-way-to-split-and-evaluate). Use a **temporal leave-one-out** split (predict each user's chronologically last item from their prior history), compute **full** Recall@K and NDCG@K over the *entire* catalog — not sampled negatives, because sampled metrics are inconsistent (Krichene and Rendle, KDD 2020) — and report **index memory** and **serving latency p99** measured after warm-up. Tune both systems equally hard; an undertuned baseline is the oldest way to make a frontier method look good.

Here is a compact eval harness that scores the generative retriever's top-K against the held-out next item, computing Recall@K and NDCG@K over the full catalog. The same `recall_ndcg` function scores a two-tower's ANN top-K, so the comparison is apples to apples:

```python
import numpy as np

def recall_ndcg_at_k(ranked_items, ground_truth, k=10):
    """ranked_items: top-k item ids for one user; ground_truth: the held-out item."""
    topk = ranked_items[:k]
    hit = ground_truth in topk
    recall = 1.0 if hit else 0.0           # leave-one-out: 1 relevant item
    if hit:
        rank = topk.index(ground_truth)    # 0-based position
        ndcg = 1.0 / np.log2(rank + 2)     # DCG with ideal DCG = 1
    else:
        ndcg = 0.0
    return recall, ndcg

def evaluate(model, eval_users, item2sid, sid2item, trie, valid, k=10):
    recalls, ndcgs = [], []
    for hist, target in eval_users:        # target = held-out next item id
        history_tokens = [t for it in hist for t in encode_id(item2sid[it])]
        beams = constrained_beam_search(model, history_tokens, trie, valid,
                                        n_levels=3, beam=k)
        ranked = [sid2item[sid] for sid, _ in beams]   # back to item ids
        r, n = recall_ndcg_at_k(ranked, target, k=k)
        recalls.append(r); ndcgs.append(n)
    return float(np.mean(recalls)), float(np.mean(ndcgs))
```

Two honesty checks this harness enforces. First, it ranks against the *whole* catalog (the trie holds every valid item), so the metric is not inflated by an easy sampled-negative set. Second, because retrieval is leave-one-out with a single relevant item, Recall@K is exactly the hit rate and NDCG@K reduces to the position-discounted hit — no hidden weighting that could flatter one model. Run the identical scoring on the two-tower's ANN output and you have a defensible table.

## Section 6 — Conversational and agentic recommendation

Now the second frontier. Everything above keeps the classic product surface: the user does something, the system returns a list. Conversational recommendation changes the surface itself. Instead of guessing intent from a click stream, the system *talks* to the user — and crucially, the user can talk back.

Figure 6 contrasts the two surfaces. On the left, a single-shot ranked list: one query in, twenty items out, click or leave. On the right, a multi-turn conversation: the system *elicits* preferences ("what's your budget?"), the user *critiques* ("cheaper," "more like the second one"), and the slate narrows turn by turn until it converges on something the user actually wants.

![A side by side comparison of a single-shot frozen ranked list against a multi-turn conversational flow that elicits preferences and accepts critiques to narrow the candidate slate](/imgs/blogs/generative-and-conversational-recommendation-6.png)

The academic field is **conversational recommender systems** (CRS), and surveys (Jannach et al., *A Survey on Conversational Recommender Systems*, ACM Computing Surveys 2021; Gao et al., *Advances and Challenges in CRS*, 2021) carve it into a few core capabilities:

1. **Preference elicitation** — the system asks. "Are you shopping for yourself or a gift?" "Do you prefer something light or substantial?" Each question is chosen to maximally reduce uncertainty about the user's intent. Classical CRS framed this as choosing the attribute that most splits the candidate set (an information-gain criterion); LLM-based CRS lets the model phrase natural, context-aware questions.
2. **Critiquing** — the user steers. Given a recommendation, the user says "cheaper" or "more colorful" or "more like the second one but not a sequel." The system updates the candidate set against the critique. This is the single most powerful affordance conversation adds: it turns a one-shot guess into a steerable search.
3. **Explanation** — the system justifies. "I picked this because it matches the cozy mystery vibe you liked and it's under your budget." Explanations build trust and, importantly, make critiques *actionable* — the user knows what to push against.

The reason LLMs changed this field overnight is that they are natively good at exactly these three things: asking sensible clarifying questions, interpreting fuzzy natural-language critiques, and producing fluent justifications. Before LLMs, CRS was a brittle pipeline of intent classifiers, slot fillers, and templated responses. With a capable model — Claude Opus 4.5 or Claude Sonnet 4.5 for the reasoning-and-dialogue layer — the dialogue management mostly *works* out of the box. The hard part shifts from "can it converse?" to "can we stop it from recommending items that do not exist?"

### The science: why a good question is worth more than a good guess

The classical CRS framing of preference elicitation is information-theoretic, and it explains *why* asking the right question beats guessing. Model the user's target item as a random variable over the candidate set, with a current belief distribution $p$ over candidates. The uncertainty is the entropy $H(p) = -\sum_i p_i \log p_i$. A question $q$ with possible answers $a$ partitions the candidates; the *expected* uncertainty after asking is the conditional entropy $H(p \mid q) = \sum_a P(a)\, H(p \mid a)$, and the value of the question is the **information gain** $\text{IG}(q) = H(p) - H(p \mid q)$. The best question to ask is the one that maximizes information gain — the one whose answer most evenly splits the remaining candidates by your current belief, because an even split removes the most entropy.

This is why "what's your budget?" early in a shopping conversation is so powerful: price typically splits a catalog into roughly balanced halves, so its information gain is high — one answer can eliminate half the candidates. An attribute that 95 percent of items share ("do you want something useful?") has almost zero information gain; the answer barely moves the belief. A single well-chosen question can do more to narrow the slate than re-ranking the entire candidate list, which is the formal reason a conversation can beat a one-shot guess: the guess works with the entropy it has, while the question *reduces* the entropy before committing. LLM-based CRS does not compute information gain explicitly, but a strong model has implicitly learned to ask the high-gain questions — budget, use-case, recipient — first, which is exactly what you want.

### The science: the LLM as an agent over tools

The right framing — and the one that makes conversational recommendation actually shippable — is the LLM as an **agent over tools**, not as a recommender in itself. The LLM is brilliant at language and reasoning and *terrible* at being a database. Ask a raw LLM for product recommendations and it will confidently invent SKUs, prices, and availability that do not exist — the hallucination problem, which in a shopping context means recommending items you cannot sell.

The fix is **grounding**: give the LLM a *tool* — your real catalog search, your existing two-tower retriever, a SQL filter over the product table — and require it to call the tool and recommend *only* from the returned rows. The LLM's job becomes orchestration: read the conversation, decide what to search for, call the tool, and present the real results with explanations. The recommender and the catalog are tools; the LLM is the conductor. This is the same constrained-decoding instinct as the semantic-ID trie, lifted to the dialogue level: the model may only output items that provably exist.

Figure 7 draws the agentic loop: a user turn flows into the LLM, the LLM calls catalog search as a tool, the tool returns real items, the LLM grounds its recommendation in those rows, presents them with explanations, and asks the next clarifying question — repeat.

![A directed chain showing a user turn flowing into an LLM agent that calls a catalog search tool whose grounded results feed an explained reply and the next elicitation turn](/imgs/blogs/generative-and-conversational-recommendation-7.png)

Two grounding mechanisms are worth naming explicitly. **Constrained decoding / tool-only recommendation**: the LLM is instructed (and ideally structurally forced via tool-use schemas) to emit item references only as IDs returned by a tool call, never as free text. **RAG over the catalog**: retrieve relevant catalog rows (via your two-tower retriever or a keyword search), inject them into the prompt, and have the LLM recommend from that retrieved context — retrieval-augmented generation applied to recommendation. In practice you combine them: RAG supplies the candidates, tool-use schemas enforce that the model only references those candidates, and a final validation step rejects any item the model names that is not in the returned set.

Notice the symmetry with the first half of the post. In generative retrieval, the trie is a *hard structural constraint* — the decoder physically cannot emit a token that leaves the set of valid items. In conversational recommendation, the analogous constraint is *softer*: the system prompt and tool schema strongly steer the model toward tool-returned items, but a language model can still, occasionally, paraphrase an item into existence. That is why the post-validation step is not optional. The two frontier ideas are the same discipline — never output an item that does not exist — implemented at two points on the hardness spectrum: a trie mask that is impossible to violate, and a tool-and-validate loop that is merely very hard to violate. When you can make the constraint structural, do; when you cannot, validate after the fact. Both are forms of grounding, and grounding is the one thing you cannot skip in either system without shipping a recommender that lies.

### Practical: a conversational-rec loop with tool use

Here is a faithful sketch of the agentic loop against the Anthropic API. The LLM is given a `search_catalog` tool; it decides when to call it; we execute the search against the real catalog and feed results back; the model grounds its reply in those rows. I am using Claude Sonnet 4.5 as the dialogue model — a strong, cost-effective choice for an interactive, latency-sensitive product surface.

```python
import anthropic
import json

client = anthropic.Anthropic()   # reads ANTHROPIC_API_KEY

TOOLS = [{
    "name": "search_catalog",
    "description": "Search the real product catalog. Returns only items that exist, "
                   "with id, title, price, and attributes. Recommend ONLY from these.",
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "free-text intent"},
            "max_price": {"type": "number"},
            "category": {"type": "string"},
            "k": {"type": "integer", "default": 10},
        },
        "required": ["query"],
    },
}]

SYSTEM = (
    "You are a shopping assistant. You may recommend ONLY items returned by "
    "search_catalog. Never invent products, prices, or availability. Ask one "
    "clarifying question when intent is ambiguous. Explain each pick in one line."
)

def search_catalog(query, max_price=None, category=None, k=10):
    # back this with your real retriever: two-tower + ANN, or SQL filter.
    return real_retriever.search(query, max_price=max_price, category=category, k=k)

def agent_turn(messages):
    while True:
        resp = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=1024,
            system=SYSTEM,
            tools=TOOLS,
            messages=messages,
        )
        messages.append({"role": "assistant", "content": resp.content})
        if resp.stop_reason != "tool_use":
            return resp, messages                     # final grounded reply
        # execute every tool call the model requested
        tool_results = []
        for block in resp.content:
            if block.type == "tool_use" and block.name == "search_catalog":
                rows = search_catalog(**block.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(rows),       # real catalog rows
                })
        messages.append({"role": "user", "content": tool_results})
```

A production hardening you should add but I have elided for clarity: after the model produces its final reply, parse out every item it referenced and assert each one appears in the union of `search_catalog` results from this conversation. Drop or flag any that do not. This belt-and-suspenders check catches the rare case where the model paraphrases an item into existence despite the instructions — the conversational analog of the trie constraint.

#### Worked example: a three-turn elicitation transcript with grounding

Here is a concrete three-turn trace showing elicitation, critiquing, and grounding in action. The bracketed lines are the system's internal tool calls, not shown to the user.

**Turn 1.** *User:* "I need a gift for my dad who likes cooking." The model recognizes the intent is underspecified — budget and skill level are unknown — and elicits before searching. *Assistant:* "Happy to help find something for a cooking enthusiast. Roughly what's your budget, and is he a casual home cook or pretty serious about it?"

**Turn 2.** *User:* "Under \$60, and he's serious — already has all the basics." Now the model has enough to ground a search. `[search_catalog(query="serious home cook gift, beyond basics", max_price=60, category="kitchen", k=10)]` returns ten real SKUs with ids and prices. *Assistant:* "Here are three that fit a serious cook under \$60: a Japanese whetstone set (\$48) for keeping knives razor sharp, a digital instant-read thermometer (\$39) for precise temps, and a cast-iron grill press (\$32). The whetstone is my top pick — it's the upgrade casual cooks skip but serious ones love. Want something more giftable, or more practical?"

**Turn 3.** *User:* "More giftable, and more like the thermometer — gadgety." This is a critique on two axes (giftability, gadget-ness) plus an anchor ("like the thermometer"). The model interprets it and re-searches. `[search_catalog(query="gadget kitchen gift, precision tools, giftable", max_price=60, category="kitchen-gadgets", k=10)]` returns a fresh set. *Assistant:* "Then I'd point you to a precision kitchen scale with a companion app (\$54) — same precision-gadget appeal as the thermometer, and it comes in nice gift packaging. A close second is a sous-vide immersion stick (\$59) if he'd enjoy a new technique to play with."

Every item named came from a tool call against the real catalog — no invented products, no fictional prices. The conversation narrowed from "gift for a cooking dad" to "a precision gadget under \$60, giftable" across three turns, with the model eliciting on turn 1, recommending grounded items on turn 2, and absorbing a multi-axis critique on turn 3. That is the full CRS loop running on an LLM agent.

### The wins: cold start and the natural-language interface

Two product wins justify the added complexity. First, **cold start**. A conversational agent grounded in content-based catalog search needs *zero* interaction history to be useful — the user simply tells it what they want, in words. This is the same content-and-language strength that makes [LLM4Rec](/blog/machine-learning/recommendation-systems/llms-for-recommendation-llm4rec) shine on day-one catalogs with no clicks. Second, the **natural-language interface** itself: users can express intent that no fixed filter UI can capture ("something cozy for a rainy weekend that isn't too sad"). The model maps that to a search; the filter UI never could. For long-tail, high-consideration purchases — travel, electronics, gifts — this is a genuinely better surface.

## Section 7 — Honest assessment: where the frontier wins today

Time for the section the series exists for. Both of these are exciting; neither is a default. Here is my read on where each wins *today* and where the boring, proven stack still rules.

**Generative retrieval is promising but young.** The wins are real — catalog-independent memory, cold-start generalization, a clean LLM bridge — and on academic sequential benchmarks (Amazon Beauty/Sports/Toys), TIGER-style models edge out strong baselines like SASRec on Recall@K and NDCG@K. But three things keep it out of the highest-scale production tiers for now: **latency** (autoregressive beam search is several decoder forward passes versus one ANN lookup; at p99 you are comparing tens of milliseconds to single digits), **scale** (most reported deployments are sub-100M items; the trie and decoding cost at billions of items is an open problem), and **catalog churn** (codebook staleness and re-minting IDs is operationally heavier than continuously re-embedding into an ANN index). The convergence work — LIGER, LC-Rec — suggests the near-term sweet spot is *hybrid*: dense retrieval for warm head items, generative for cold-start and long-tail generalization, semantic IDs as the shared representation.

**Conversational recommendation changes the product, and adds cost.** It is a genuinely better surface for high-consideration, exploratory, natural-language-friendly journeys, and an excellent cold-start tool. But every turn is an LLM call: latency in the hundreds of milliseconds to seconds, and a per-conversation cost that, even at a few cents, dwarfs the fraction-of-a-cent of a classic retrieve-and-rank pass. For a high-frequency feed where users scroll dozens of items per second, the economics do not work — you cannot run an LLM agent per impression. For a considered purchase where a user spends minutes and the basket is worth dollars, the economics work fine.

**Where two-tower + ranking still rules.** For high-throughput feeds and large catalogs where you need single-digit-millisecond retrieval at billions of items, the two-tower-plus-ANN-plus-ranker funnel is still the right answer, and it is not close. It is mature, cheap per request, scales to billions, and the operational playbook is well understood. Do not replace it with a frontier method to chase a benchmark point; replace it only where the frontier's *specific* property — cold-start generalization, a conversational surface, catalog-independent memory — solves a problem the two-tower stack genuinely cannot.

The decision is not "old versus new." It is "which property does my problem need, and what latency and cost budget do I have?" Figure 8 shows the trade-off as measured numbers rather than vibes.

![A three by three results matrix comparing two-tower with ANN, a SASRec baseline, and a TIGER-style generative model across Recall at ten, index memory, and latency](/imgs/blogs/generative-and-conversational-recommendation-8.png)

## Section 8 — Case studies and real numbers

A few named results to anchor the claims. Be aware these are literature figures and vary by exact protocol; I cite the source and flag approximations.

**DSI — the seed idea (Tay et al., NeurIPS 2022).** *Transformer Memory as a Differentiable Search Index* showed, for document retrieval, that a single T5 could map a query to a document ID by *generating* it, with retrieval quality competitive with a dual-encoder-plus-ANN baseline, and that larger models helped more. DSI established the slogan and the mechanism ("the parameters are the index") but used naive document IDs; it left open how to build *meaningful* IDs, which is exactly what semantic IDs solve.

**TIGER — generative retrieval for recommendation (Rajput et al., NeurIPS 2023).** *Recommender Systems with Generative Retrieval* introduced RQ-VAE semantic IDs and a T5 decoder generating the next item's ID. On Amazon Product Reviews (Beauty, Sports and Outdoors, Toys and Games), TIGER reported Recall@5/@10 and NDCG@5/@10 improvements over strong sequential baselines including SASRec and S3-Rec, with relative gains commonly in the roughly 10% to 20% range on several reported metrics (precise figures vary by dataset and metric). TIGER also demonstrated the two qualitative wins flat ANN cannot easily offer: cold-start generalization via shared codes, and a diversity knob via decoding temperature. It is the proof-of-concept that the semantic-ID road arrives somewhere useful.

**LIGER and LC-Rec — the convergence (2024).** LIGER showed pure generative retrieval lags dense retrieval on *seen* items but wins on *cold* items, and hybridized the two to capture both regimes. LC-Rec aligned semantic IDs with an LLM's language space so a general LLM can reason over items via their IDs. Together they argue the future is not generative-versus-dense but a unified item representation served either way per the latency budget.

**Conversational recommendation — the field (Jannach et al., CSUR 2021; Gao et al., 2021).** The CRS surveys formalize preference elicitation, critiquing, and explanation as the core capabilities, and document the classical pipeline (intent classification, slot filling, templated response) that LLMs largely subsume. Recent LLM-agent recommenders — systems that wrap an LLM around catalog-search and recommender tools with grounding — are the practical realization, and the open challenges they name are exactly the ones in Section 7: grounding to avoid hallucinated items, latency and cost per turn, and faithful explanations.

#### Worked example: generative retrieval versus a two-tower baseline

Let me make the trade-off concrete with a literature-consistent comparison on **Amazon Beauty** (a standard sequential-recommendation benchmark), under temporal leave-one-out with full-catalog metrics. These numbers are illustrative of the *pattern* the literature reports, not a single paper's exact table — I am combining typical orders of magnitude to show the shape of the decision.

A tuned two-tower retriever lands around Recall@10 $\approx 0.071$, with an ANN index of roughly 180 MB (one 64-dimensional float vector per item over a catalog in the tens of thousands, plus index overhead) and p99 retrieval latency around 8 ms. A strong SASRec sequential baseline lands around Recall@10 $\approx 0.079$ at similar memory and latency. A TIGER-style generative model lands around Recall@10 $\approx 0.091$ — the best ranking quality — while storing only the codebooks and trie (on the order of 24 MB, roughly an order of magnitude less), but pays for it in p99 latency around 31 ms because of three or four autoregressive decoding steps with beam search.

Read the Pareto frontier honestly. Generative retrieval bought you roughly a 15% relative Recall@10 lift over the two-tower and a roughly 7x memory reduction, at the cost of roughly 4x latency. If your bottleneck is index memory (a huge catalog straining your serving hosts) or new-item coverage, that is a great trade. If your bottleneck is p99 latency on a high-throughput feed, it is a bad trade. Same numbers, opposite decisions — which is the entire point of measuring instead of arguing.

## Section 9 — When the frontier is worth it (and when not)

The decisive recommendation, stated plainly.

**Reach for generative retrieval when** your catalog is content-rich (so semantic IDs are meaningful), cold start and long-tail new-item coverage are first-order problems, index *memory* is a real constraint, or you want a clean representational bridge to LLM-based reasoning over items. The sweet spot today is catalogs in the millions, where the latency cost is tolerable and the memory and cold-start wins are large. Strongly consider the *hybrid* (LIGER-style) form: dense retrieval for warm head items, generative for cold-start and tail.

**Do not reach for generative retrieval when** you need single-digit-millisecond p99 at billions of items (the two-tower-plus-ANN funnel is mature and far cheaper per request), when your catalog is content-poor (the IDs will not be meaningful), or when a tuned two-tower already hits your retrieval target — do not pay decoding latency and codebook-maintenance complexity to chase an offline benchmark point. And never ship it on offline Recall alone; validate online, because the offline-online gap is as real here as anywhere ([the offline-online gap](/blog/machine-learning/recommendation-systems/the-offline-online-gap-and-why-your-metric-lied)).

**Reach for conversational recommendation when** the journey is high-consideration and exploratory (travel, electronics, gifts, real estate), when natural-language intent matters and a fixed filter UI cannot capture it, when cold start is acute (no history, but the user can tell you what they want), or when explanation and trust are part of the product. The per-conversation cost is justified by a high basket value and a user willing to spend minutes.

**Do not reach for conversational recommendation when** you have a high-throughput feed with dozens of impressions per second per user (you cannot afford an LLM call per impression), when latency budgets are tight (an LLM turn is hundreds of milliseconds to seconds), or when the task is low-consideration and a ranked list already converts well. A conversation is a *better* surface only when the user actually wants to converse.

**The unifying rule:** both frontier ideas are about *grounding generation in a real catalog* — the semantic-ID trie grounds token generation to real items; the tool-use loop grounds dialogue to real catalog rows. That shared discipline — never let the model output something that does not exist — is the single most important thing to get right in either. Get grounding wrong and generative retrieval returns phantom IDs and conversational rec sells imaginary products.

## Section 10 — Stress-testing the design

Let me pressure-test both approaches against the questions a skeptical reviewer will ask, the way we do throughout the series in [debugging a recommender that won't improve](/blog/machine-learning/recommendation-systems/debugging-a-recommender-that-wont-improve).

*What if the catalog has 100M+ items?* Generative retrieval's *memory* story stays great (codebooks are fixed-size regardless of catalog), but the *trie* grows and the *decoding* cost per query is constant in catalog size only if beam search stays small — and at huge scale the per-token allowed-set can still be large near the root. This is the open scaling frontier; today, two-tower-plus-ANN is the safer choice past roughly 100M items. The hybrid form helps: use generative retrieval as a *second* candidate source for cold and tail items, layered on a two-tower for the head.

*What about code collisions?* At $K^L$ buckets and millions of items, thousands of items will quantize to the same tuple (birthday problem). TIGER's fix — an extra disambiguating token — works but slightly weakens the semantic purity of the ID. A larger codebook or more levels reduces collisions at the cost of a bigger vocabulary and more decoding steps. It is a genuine tuning knob, not a solved problem.

*What about cold codes — codebook entries that never fire?* Codebook collapse (Section 2) wastes capacity and degrades ID resolution. The fixes are k-means init and dead-code reset / EMA updates. If you skip them, your "256-entry" codebook may have a dozen live entries and your semantic IDs lose most of their discriminative power — a silent failure that shows up as mediocre Recall you cannot explain.

*What if the LLM in a conversational rec hallucinates an item despite grounding?* This is the conversational analog of generating an invalid semantic ID. Defense in depth: instruct the model to recommend only from tool results, force item references through tool-use schemas rather than free text, and — the belt-and-suspenders step — post-validate every item the model names against the actual tool results, dropping any that do not appear. Treat a hallucinated item as a hard bug, because in commerce it means recommending something you cannot sell.

*What if conversational latency tanks engagement?* An LLM turn is slow. Mitigations: stream the response token by token so the user sees progress, run the catalog-search tool call in parallel with composing the preamble, cache and reuse retrieval across turns within a session, and use a fast model (Claude Sonnet 4.5 over Claude Opus 4.5 for the latency-sensitive interactive layer; reserve the heavier model for the hard reasoning turns). And measure online: a slower-but-better conversation can still lose to a fast list if users will not wait.

*What if offline Recall rises but online engagement is flat?* The oldest recsys trap, and these frontier methods are not immune. A generative retriever can ace offline Recall@10 against historical next-items while recommending obvious continuations that do not expand engagement. The fix is not a better decoder; it is honest online evaluation, diversity in re-ranking, and counterfactual estimation ([counterfactual and off-policy evaluation](/blog/machine-learning/recommendation-systems/counterfactual-and-off-policy-evaluation)). Generate the candidates with the frontier; still rank and re-rank with the discipline of the rest of the funnel.

## Key takeaways

1. **Generative retrieval replaces embed-then-ANN with generate-the-ID.** A transformer emits an item's discrete semantic ID token by token under trie constraints, so the model's parameters are the index — no separate vector store to grow or rebuild.
2. **RQ-VAE turns a content embedding into a coarse-to-fine semantic ID.** Quantize the encoder output, then quantize each residual against successive codebooks; the tuple of indices is the ID, and shared leading codewords make it semantic.
3. **Constrained beam search over a code trie is mandatory, not optional.** It guarantees every generated ID is a real item by masking each step to valid continuations, and it speeds decoding because allowed sets are small.
4. **Shared codes are why new items generalize for free.** A cold item inherits the generative structure of its prefix-siblings and can be recommended the moment its ID is minted and added to the trie — the core advantage over a flat random ID.
5. **The frontier is converging, not competing.** LIGER and LC-Rec hybridize dense and generative retrieval with semantic IDs as the shared representation; the future is "decode or look up as the latency budget allows," not one or the other.
6. **Conversational recommendation changes the product surface.** Multi-turn elicitation, critiquing ("cheaper," "more like the second one"), and explanation turn a one-shot guess into a steerable search — a better surface for high-consideration, natural-language journeys and acute cold start.
7. **The LLM is the conductor, the catalog is the instrument.** Treat the model as an agent over tools (catalog search, the existing retriever) and ground every recommendation in real returned rows; never let it invent items. Grounding is the discipline both frontier ideas share.
8. **Name the latency and cost budget before you choose.** Generative retrieval costs decoding latency for memory and cold-start wins; conversational rec costs an LLM call per turn for a better surface. Two-tower-plus-ranking still rules high-throughput feeds at billions of items.
9. **Measure honestly or your table lies.** Temporal split, full-catalog Recall/NDCG (sampled metrics are inconsistent), index memory, p99 latency after warm-up, equally tuned baselines, and online validation — the offline-online gap does not spare the frontier.

## Further reading

- Tay et al., *Transformer Memory as a Differentiable Search Index*, NeurIPS 2022 — the seed idea that a model can generate identifiers and "the parameters are the index."
- Rajput et al., *Recommender Systems with Generative Retrieval* (TIGER), NeurIPS 2023 — RQ-VAE semantic IDs and autoregressive generative retrieval for recommendation.
- van den Oord, Vinyals, Kavukcuoglu, *Neural Discrete Representation Learning* (VQ-VAE), NeurIPS 2017 — the vector-quantization machinery RQ-VAE extends.
- Yang et al., *LIGER: Combining Generative and Dense Retrieval for Sequential Recommendation*, 2024 — hybridizing dense and generative retrieval, warm versus cold items.
- Zheng et al., *Adapting Large Language Models by Integrating Collaborative Semantics for Recommendation* (LC-Rec), 2024 — aligning semantic IDs with an LLM's language space.
- Jannach, Manzoor, Cai, Chen, *A Survey on Conversational Recommender Systems*, ACM Computing Surveys 2021 — the formal CRS capabilities: elicitation, critiquing, explanation.
- Krichene, Rendle, *On Sampled Metrics for Item Recommendation*, KDD 2020 — why you must use full metrics, not sampled negatives, to compare these systems.
- Anthropic, *Tool use (function calling) and the Messages API* — the official reference for the grounded agentic loop with catalog-search tools.
- Within this series: [what is a recommender system](/blog/machine-learning/recommendation-systems/what-is-a-recommender-system), [autoencoders and the road to generative retrieval](/blog/machine-learning/recommendation-systems/autoencoders-and-the-road-to-generative-retrieval), [LLMs for recommendation](/blog/machine-learning/recommendation-systems/llms-for-recommendation-llm4rec), [finetuning LLMs for recommendation in practice](/blog/machine-learning/recommendation-systems/finetuning-llms-for-recommendation-in-practice), [the two-tower model for retrieval](/blog/machine-learning/recommendation-systems/the-two-tower-model-for-retrieval), and the capstone [the recommender systems playbook](/blog/machine-learning/recommendation-systems/the-recommender-systems-playbook).
