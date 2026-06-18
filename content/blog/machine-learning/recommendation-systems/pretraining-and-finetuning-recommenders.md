---
title: "Pretraining and Finetuning Recommenders"
date: "2026-06-19"
publishDate: "2026-06-19"
description: "Learn when transfer learning beats training a recommender from scratch: why item IDs do not transfer but sequence patterns and content do, how masked-item and contrastive self-supervision pretrain a backbone, and how to finetune it into a sparse target domain with measured Recall@10 wins that grow as your data shrinks."
tags:
  [
    "recommendation-systems",
    "recsys",
    "transfer-learning",
    "pretraining",
    "finetuning",
    "self-supervised-learning",
    "cold-start",
    "machine-learning",
    "pytorch",
    "movielens",
  ]
category: "machine-learning"
subcategory: "Recommendation Systems"
author: "Hiep Tran"
featured: true
readTime: 51
image: "/imgs/blogs/pretraining-and-finetuning-recommenders-1.png"
---

The launch that taught me to stop training from scratch was a recommender for a new vertical — call it a home-goods marketplace that the company spun up next to a music app that had been running for years. On day one the home-goods catalog had forty thousand items and almost no interactions: a trickle of clicks, a handful of purchases, nothing that looked like a training set. The obvious plan was to take the architecture that worked on music — a sequence model that predicted the next item from a user's recent history — and train a fresh copy on the home-goods logs. We did. It produced a model that had clearly memorized the eleven things people had clicked on so far and generalized to nothing. Recall@10 on a held-out slice was 0.12, which is the polite way of saying the model was a slightly fancy popularity list. The music model, by contrast, had been trained on two years of behavior and scored 0.31 on its own domain. We had a great recommender for music and a useless one for home goods, and the difference was not the architecture — it was the same architecture — it was the *data*.

The thing that fixed it was not more home-goods data, which we did not have, and not a bigger model, which would have overfit harder. It was realizing that most of what the music model knew was *not* about music. It knew the grammar of a browsing session: that recency matters but the first item often anchors intent, that people repeat-purchase consumables but rarely re-buy durables, that a burst of clicks followed by a long gap means the session ended. None of that is musical. All of it is true of home goods too. The music model's *item embeddings* — the part that actually knew "this is a jazz album" — were useless in the new domain, because a home-goods item id points at a spatula, not a song. But the *rest* of the network, the part that consumed sequences and modeled behavior, was a head start we were throwing away by initializing from random weights every launch.

![A two-column comparison showing a from-scratch model with random initialization on only eight thousand target interactions reaching Recall at ten of 0.121 versus a pretrain-then-finetune path that pretrains on two million behaviors, finetunes on the same eight thousand interactions, and reaches 0.178 for a forty-seven percent warm-start gain](/imgs/blogs/pretraining-and-finetuning-recommenders-1.png)

This post is about transfer learning for recommenders: pretrain on a lot of behavior, then finetune on the task or domain you actually care about, and beat from-scratch training — sometimes by a little, often by a lot, and most of all when the target data is scarce. It sits in the series **Recommendation Systems: From Click to Production**, in the retrieval-and-ranking funnel laid out in [what is a recommender system](/blog/machine-learning/recommendation-systems/what-is-a-recommender-system). The funnel's spine is retrieval to ranking to re-ranking, fed by the serve-log-train feedback loop and read off the offline-versus-online gap. Transfer learning is what you reach for when one stage of that funnel — usually retrieval or the first ranker — has to work in a domain or a segment where the feedback loop has barely begun to spin. By the end you will be able to explain *why* ID embeddings refuse to transfer while sequence and content representations do, derive the self-supervised pretraining objectives (masked-item modeling and contrastive sequence learning) and the data-efficiency argument that makes finetuning win in low-data regimes, pretrain a sequence backbone with a masked-item loss and finetune it on next-item prediction in PyTorch, wire a `sentence-transformers` content encoder into an item tower with frozen-versus-finetuned variants, and read a before-after table that shows the finetuning advantage growing as the target data shrinks. This also sets up the LLM4Rec posts that follow, where the pretrained backbone is a full language model — see [LLMs for recommendation (LLM4Rec)](/blog/machine-learning/recommendation-systems/llms-for-recommendation-llm4rec) and [finetuning LLMs for recommendation in practice](/blog/machine-learning/recommendation-systems/finetuning-llms-for-recommendation-in-practice).

## 1. The problem: when from-scratch is the wrong default

Most recommender tutorials assume you have a dense interaction matrix and train end to end. That assumption quietly does a lot of work, and it breaks in three situations that show up constantly in production.

**A new domain or a new product.** You launch a vertical, a country, a new content type. The catalog exists; the interaction history does not. From-scratch training on a few thousand interactions gives you a model that overfits the items a handful of early users happened to click. This is the home-goods story above.

**A cold-start segment inside a live system.** Even in a mature product, slices of it are perpetually cold: brand-new items with zero clicks, brand-new users with no history, a long tail of items that get a few interactions a month. The bulk of the model trains fine on the warm head; the cold slices get the same useless popularity list. We have a whole post on this — [the cold-start problem](/blog/machine-learning/recommendation-systems/the-cold-start-problem) — and pretraining-plus-finetuning is one of its sharpest tools.

**A small target task with a big related corpus.** You want a model for a niche objective — recommending within a single category, for a single high-value customer segment, for a specific surface like a checkout up-sell. That task has little data of its own, but it lives next to an ocean of general behavior. Training the niche model from scratch wastes the ocean.

In all three the disease is the same: **the target task is data-poor, but a related source is data-rich.** Transfer learning is the prescription. You pretrain a model on the rich source — ideally without needing labels, using self-supervision over raw behavior — and then you finetune it on the poor target, adapting only what needs adapting. The question this post answers carefully is *what* actually carries over, *why*, and *how much* it buys you as a function of how little target data you have.

Let me be precise about the failure mode of from-scratch training in low data, because it is the whole motivation. A recommender has a lot of parameters: an embedding table with one vector per item (tens of thousands to billions of rows), plus a network on top. With abundant interactions, the data constrains all those parameters and the model generalizes. With a few thousand interactions and forty thousand items, most item embeddings are touched zero or one times during training. An embedding that a single example touches is fit to that one example and is otherwise random — it has learned noise, not signal. The model's effective capacity dwarfs the information in the data, and it memorizes. This is a textbook high-variance regime: the bias is low (the model *can* represent the truth) but the variance is enormous (which truth it lands on swings wildly with the tiny sample). Pretraining attacks exactly this by importing low-variance structure learned elsewhere, so the target data only has to do a little adaptation rather than learn everything.

### The two-stage pipeline at a glance

Before we go deep on each piece, it helps to see the whole shape of what we are building, because every section below is one stage of the same pipeline. The pattern has two phases that are deliberately decoupled.

![A vertical layered pipeline showing raw behavior logs of two million sequences feeding a self-supervised pretraining stage with a masked-item loss, which produces a pretrained backbone of shared weights, which feeds a finetune-head stage on a next-item target, which produces an adapted model at Recall at ten of 0.178, which is then served as top-K at p99 of thirty-five milliseconds](/imgs/blogs/pretraining-and-finetuning-recommenders-2.png)

**Phase one, pretraining**, runs on the data-rich source. It is *task-agnostic and label-free*: you feed it the firehose of raw behavior — every session, every click stream, no human labels — and train a backbone with a self-supervised objective that manufactures its own targets (mask an item, predict it). This phase is expensive and you run it rarely, because the source rarely changes much and a backbone is reusable. **Phase two, finetuning**, runs on the data-poor target. It is *cheap and task-specific*: load the pretrained backbone, attach a small head for the actual serving task (next-item prediction), and adapt with a low learning rate on a little target data. The figure shows the flow top to bottom — raw logs to SSL pretrain to backbone to finetune head to adapted model to serving — and the key economic property is that one backbone amortizes across many targets. You pay the pretraining cost once and harvest it across every cold domain, every segment, every retrain. That amortization is what makes the whole approach worth its complexity in production, and it is exactly the structure the LLM4Rec posts will inherit, just with a language model as the backbone.

The decoupling also buys you operational flexibility. The two phases can run on different schedules (pretrain weekly on the full firehose, finetune nightly on the fresh target window), on different hardware (pretraining wants a big GPU; finetuning a head is light), and by different teams (a platform team owns the backbone, product teams own their finetunes). When you hear "foundation model for recommendations," this is the architecture being described: a shared, expensively-pretrained backbone that many downstream tasks finetune cheaply. The rest of this post is the engineering of each box in that figure.

## 2. Why transfer in recsys is hard — and what actually transfers

Computer vision and NLP made transfer learning look easy: pretrain a ResNet on ImageNet or a BERT on a web corpus, finetune on your task, win. Recommenders resisted this for years, and the reason is worth understanding because it tells you exactly which parts of the model to transfer and which to throw away.

The core obstacle is that **the most important features in a classic recommender are raw IDs, and an ID is an arbitrary symbol with no meaning outside its own catalog.** In a vision model, a pixel value of 200 means "bright" in every image ever taken; the input space is shared across all tasks. In an NLP model, the token "bank" carries meaning that is at least partly shared across every corpus. But item id `48291` in the music app and item id `48291` in the home-goods app have nothing to do with each other — one is a Coltrane record and the other is a colander. The embedding table that maps ids to vectors is *learned per catalog from scratch*, and there is no alignment between two catalogs' id spaces. So the single biggest block of parameters in a collaborative-filtering model — the item embedding table — is fundamentally non-transferable. You cannot copy a music item embedding into a home-goods slot and expect anything but garbage.

![A four-row matrix listing item ID embeddings as non-transferable because the index is arbitrary per catalog, sequence patterns as transferable because order grammar is universal, user behavior shape as transferable because recency repeat and basket logic recur, and content representations as transferable because text and image meaning is shared](/imgs/blogs/pretraining-and-finetuning-recommenders-3.png)

That is the bad news, and for a long time people concluded that recommenders just do not transfer. The insight that unlocked the field is that **the model is not only its ID embeddings.** Three things *do* transfer, and they are exactly the things that are not tied to a specific id space.

**Sequence patterns transfer.** The grammar of a browsing session — recency, repetition, the way a session opens with exploration and narrows toward intent, the way a basket clusters complementary items — is close to universal across domains. A transformer's attention layers, its position embeddings, its feed-forward blocks all learn this grammar, and the grammar of "what comes next given an ordered history" is the same whether the items are songs, products, or videos. You can transfer the entire sequence-modeling stack and only relearn the item embeddings.

**Behavior shape transfers.** Beyond raw order, the statistics of how people interact — that engagement decays with position, that there is a long tail, that some interactions are noisy and some are intent-revealing, that consumables get repeat-purchased and durables do not — are properties of human shopping and consumption, not of a particular catalog. A model that has learned to weight a long-ago anchor click over recent noise has learned something true everywhere.

**Content representations transfer.** This is the big one, and it is the bridge that makes everything else work. An item is not only an id; it has *content* — a title, a description, an image, attributes. A pretrained text encoder (BERT, a sentence-transformer) or image encoder (CLIP) maps that content into a vector that means the same thing in every domain: "this text is about science fiction" is a fact about the words, independent of which catalog the item lives in. If you represent items by their *content embeddings* instead of (or in addition to) their *id embeddings*, you get a representation that transfers across catalogs and, crucially, that exists for an item the moment it is created, before anyone has clicked it. That is the cold-start cure.

So the recipe writes itself. Throw away the id embeddings when you cross domains; keep and transfer the sequence-modeling stack and the content encoders. The figure above is the cheat sheet: IDs no, sequences yes, behavior yes, content yes. The rest of this post makes each of those transfers concrete.

#### Worked example: counting what you can and cannot reuse

Suppose your music sequence model has these parameters: an item embedding table of 1.2M songs × 128 dims = 154M parameters, and a transformer backbone (4 layers, 128 hidden, 4 heads) of roughly 1.0M parameters. The id table is 99.4% of the parameters and 0% transferable to home goods. The backbone is 0.6% of the parameters and ~100% transferable. That sounds like a terrible deal — you are transferring almost none of the weights — but it is the *opposite*. The 154M id parameters are precisely the ones that were always going to be relearned from the target catalog anyway; they are the easy, data-cheap part once the hard part is solved. The 1.0M backbone parameters encode the genuinely hard-won knowledge of how sequences behave, and they are exactly what a few thousand target interactions cannot relearn from scratch. Transferring 0.6% of the weights can move Recall@10 from 0.12 to 0.18 because those 0.6% are the load-bearing 0.6%.

## 3. The science: why IDs don't transfer but structure does

Let me make the "IDs don't transfer" claim rigorous, because it is easy to state loosely and the precise version tells you what to do.

An embedding layer is a learned function $E: \{1, \ldots, |I|\} \to \mathbb{R}^d$ that assigns a vector to each item id. The training objective only ever constrains $E$ *up to the geometry that makes the loss small* — it cares about relative positions of vectors (which items are near which), never about absolute coordinates or about which integer index sits where. There are two consequences. First, the labeling of ids is arbitrary: permute the ids and relabel the rows of $E$ accordingly and you get an identical model. So id `48291` carries no information; only the learned vector at that row does, and that vector is meaningful only relative to the other rows of *the same table*. Second, two independently trained tables live in two unaligned coordinate frames — even if both catalogs contained "the same" abstract item, the two tables would represent it at unrelated coordinates because nothing tied them together. Copying row $i$ from one table into row $i$ of another is therefore meaningless; you are copying a coordinate from one frame into a different frame.

Contrast this with the backbone. A self-attention layer computes, for query $q_i$ and keys $k_j$, attention weights $\alpha_{ij} = \text{softmax}_j(q_i^\top k_j / \sqrt{d})$ and aggregates values $\sum_j \alpha_{ij} v_j$. The *weights* of this layer parameterize a function of the *relationships between positions in a sequence* — "given a representation at position $i$, how much should it attend to position $j$." That function is about the structure of sequences, and the structure of sequences is shared across domains. The position embeddings encode "what does being the third-from-last item mean," which is domain-agnostic. The feed-forward blocks encode generic nonlinear feature transformations. None of these is tied to a specific id labeling, so all of them transfer — they are functions of the *shape* of the data, and the shape is shared even when the items differ.

The content encoder is the cleanest case of all. A pretrained text encoder $f_\text{text}: \text{string} \to \mathbb{R}^d$ was trained so that semantically similar strings map to nearby vectors, on a corpus that has nothing to do with any recommender. The function $f_\text{text}$ is *fixed and shared*: feed it "a thrilling space-opera novel" and it returns the same vector whether you are recommending books or movies. So if an item's representation is $f_\text{text}(\text{description})$ rather than $E(\text{id})$, the representation is automatically aligned across domains, because it is computed by the same function from shared content. This is why content-based transfer is the backbone of cross-domain and cold-start recommendation: it sidesteps the unaligned-id-frame problem entirely.

### The data-efficiency argument, made quantitative

Why does transferring structure help *most* when target data is scarce? The bias-variance decomposition gives the clean answer. The expected error of a learned model decomposes into bias (how wrong the best model in your hypothesis class is), variance (how much your fitted model swings with the random training sample), and irreducible noise:

$$\mathbb{E}[(\hat{y} - y)^2] = \underbrace{(\mathbb{E}[\hat{y}] - y)^2}_{\text{bias}^2} + \underbrace{\text{Var}(\hat{y})}_{\text{variance}} + \sigma^2.$$

A from-scratch deep recommender is a low-bias, high-variance estimator: it can represent the truth (low bias) but with $N$ small relative to the parameter count, the variance term dominates and the model fits the sampling noise. Pretraining changes the starting point of optimization and, effectively, the prior over solutions: it pins the backbone near a region of weight space that is known to encode real sequence structure, so the target optimization only explores a small, well-shaped neighborhood. That is a variance reduction — you are searching a much smaller effective hypothesis space conditioned on the data, with a small bias cost (the pretrained region might not be optimal for the target). When $N$ is large, the variance term is small anyway and the from-scratch model's slightly-lower bias wins or ties; the pretraining advantage shrinks toward zero. When $N$ is small, the variance term is everything and the pretrained model's variance reduction dominates. The advantage of finetuning over from-scratch therefore *grows monotonically as $N$ shrinks*, which is the single most important empirical fact in this post and the one the results table will demonstrate.

There is a learning-curve way to say the same thing. Plot test error against $\log N$ for both models. The from-scratch curve starts high (terrible at tiny $N$) and descends steeply. The finetuned curve starts much lower (the warm start already generalizes) and descends gently, because it has less left to learn. The two curves converge at large $N$. The vertical gap between them at any $N$ is the value of transfer, and it is widest at the left edge where data is scarcest. We will draw exactly this gap in figure 7.

### Pretraining as a prior, made precise

There is a cleaner Bayesian framing that makes the "pretraining is a prior" claim more than a metaphor, and it is worth a paragraph because it tells you *how strong* the prior should be. From-scratch training with a random initialization is, loosely, maximum-likelihood estimation: you find the weights $\theta$ that maximize $p(\text{target data} \mid \theta)$ with an uninformative prior. Finetuning from a pretrained checkpoint $\theta_0$ is closer to maximum-a-posteriori estimation with a prior centered at $\theta_0$ — and the implicit prior's *width* is set by your learning rate and number of epochs. A tiny learning rate with few epochs is a tight prior: you trust the pretrained weights and barely move them. A large learning rate with many epochs is a loose prior: you let the target data override the pretraining, which in the limit of a very loose prior is just from-scratch with a fancy init that gets washed out. This is why the discriminative-learning-rate trick (gentle on the backbone, fast on the head) is not a hack but the correct expression of the belief "I trust the backbone's structure but not its task-specific head." Set the backbone learning rate by how much you trust the source: closely related source, tight prior, low LR; loosely related source, looser prior, higher LR — and if you find yourself needing such a loose prior that the backbone is overwritten, that is the data telling you the source was not related enough to be worth pretraining on.

The prior view also explains the regularization-toward-old-weights mitigations we will use against forgetting (section 7). Elastic Weight Consolidation literally adds a Gaussian prior centered at the old weights with a per-parameter precision given by the Fisher information — it is the prior view written out as a loss penalty. When you see $\sum_i \frac{\lambda}{2} F_i (\theta_i - \theta_i^\text{old})^2$ later, recognize it as "keep a tight prior on the parameters that mattered, a loose prior on the ones that did not." The whole pretraining-finetuning toolkit is, underneath, a toolkit for choosing and shaping a prior, and every knob — learning rate, epochs, freezing, EWC, replay — is a knob on that prior's location and width.

### Why content sidesteps the alignment problem entirely

It is worth dwelling on *why* content transfer is qualitatively different from the other two carriers, because it is the one that works even at zero behavior. Sequence and behavior transfer still require *some* target interactions to attach the transferred structure to the target's items — the backbone knows the grammar of sequences, but it still has to learn which target id is a "coffee maker," and that learning needs clicks. Content transfer needs none, because the item-to-vector function is *already* the shared, aligned function. Formally, the alignment problem is: given two representations of "the same" abstract item in two coordinate frames, find the map between the frames. With id embeddings you have to *learn* that map from overlapping signal, and with disjoint catalogs there is no overlap to learn from. With content embeddings the map is the identity — both frames *are* the output of the same encoder $f_\text{text}$ — so there is nothing to learn and nothing to align. This is why a brand-new item with zero clicks is recommendable the instant it has a title: its position in the shared content space is determined entirely by content the encoder already understands, with no dependence on behavior at all. Behavior-based transfer is "learn the grammar elsewhere, apply it here once you have a few words to attach it to"; content transfer is "the words already mean something, everywhere, immediately."

## 4. Self-supervised pretraining for sequences

You cannot pretrain on labels you do not have, and recommenders do not come with a free supervised label the way ImageNet comes with class labels. The breakthrough was realizing that **a sequence of behavior is its own supervision.** You hide part of it and ask the model to fill in the blank. This is self-supervised learning (SSL): the supervisory signal is manufactured from the unlabeled data itself, so you can pretrain on the entire firehose of behavior logs without any human labeling. Two SSL objectives dominate sequential recommendation, and a third augments them.

![A branch-and-merge dataflow showing an item sequence of length fifty feeding a step that masks fifteen percent of items, the masked sequence entering a bidirectional encoder, the encoder producing a masked-item prediction via softmax over the catalog under a pretrain loss, the same encoder also producing a learned reusable backbone representation, and that representation transferring into a finetune head trained on a next-item target](/imgs/blogs/pretraining-and-finetuning-recommenders-4.png)

**Masked-item modeling (the BERT4Rec objective).** Take a user's item sequence, randomly replace a fraction (typically 15%) of the items with a special `[MASK]` token, and train the model to predict the original items at the masked positions from the surrounding context — both left and right, since this is bidirectional. The loss is a softmax cross-entropy over the catalog at each masked position. This is the recommender translation of BERT's masked-language-modeling (Sun et al., "BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer," CIKM 2019). The figure above shows the dataflow: mask, encode bidirectionally, predict the masked items, and keep the encoder as a reusable backbone. Because prediction uses both sides of the mask, the encoder learns rich, context-aware item representations — exactly the structure we want to transfer. We covered the BERT4Rec architecture itself in [self-attention for sequences: SASRec and BERT4Rec](/blog/machine-learning/recommendation-systems/self-attention-for-sequences-sasrec-bert4rec); here we use it as a *pretraining* objective rather than the final task.

**Next-item prediction (the autoregressive objective).** Predict $v_{t+1}$ from $(v_1, \ldots, v_t)$ with a causal mask, the SASRec setup (Kang and McAuley, "Self-Attentive Sequential Recommendation," ICDM 2018). This is also self-supervised — the "label" is just the next item in the log — and it doubles as the final serving task, so it is often used both for pretraining and for finetuning. The difference from masked modeling is directionality: next-item is left-to-right (matches serving, where you only have the past) while masked modeling is bidirectional (richer representations, but a train-serve mismatch you must handle, discussed below).

**Contrastive sequence learning (CL4SRec, S3-Rec).** Augment a sequence two different ways — crop a contiguous sub-sequence, mask some items, or reorder a local window — to get two "views" of the same user intent, then train so the two views of the *same* sequence are close in embedding space and views of *different* sequences are far apart, an InfoNCE loss. This is the recommender version of SimCLR (Xie et al., "Contrastive Learning for Sequential Recommendation," CL4SRec, ICDE 2022). The InfoNCE loss for a positive pair $(z_i, z_i^+)$ against a batch of negatives is

$$\mathcal{L}_\text{InfoNCE} = -\log \frac{\exp(\text{sim}(z_i, z_i^+)/\tau)}{\sum_{j} \exp(\text{sim}(z_i, z_j)/\tau)},$$

where $\text{sim}$ is cosine similarity and $\tau$ is a temperature. The same InfoNCE machinery underlies the in-batch-negative retrieval loss in [training two-tower: negatives and sampled softmax](/blog/machine-learning/recommendation-systems/training-two-tower-negatives-and-sampled-softmax); here it operates on augmented sequences rather than user-item pairs. S3-Rec (Zhou et al., "Self-Supervised Learning for Sequential Recommendation with Mutual Information Maximization," CIKM 2020) adds auxiliary objectives that maximize mutual information between items, attributes, and sub-sequences — pretraining the model to know that an item and its attributes belong together before it ever sees the next-item task.

Why does SSL help sparse and cold settings specifically? Because the SSL objective uses *every interaction as a training signal* without needing it to be a "correct recommendation." A from-scratch next-item model only learns from the (history, next-item) pairs you can construct, and in a cold domain there are few. Masked-item pretraining instead learns from the internal structure of *all* sequences — every item is a prediction target at some masking, every co-occurrence is a constraint — so it extracts far more signal per interaction. On the rich source domain there is a lot of this signal; the pretrained backbone soaks it up; finetuning then needs only a little target data to specialize.

### Practical: pretrain a masked-item backbone, then finetune on next-item

Here is the SSL pretraining loop in PyTorch — a compact bidirectional transformer trained with a masked-item (cloze) objective. This is the backbone we will transfer. The masking and prediction are the BERT4Rec recipe.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

MASK_ID = 0          # reserved id 0 for [MASK]
PAD_ID = 1           # reserved id 1 for padding
N_SPECIAL = 2        # ids 0..1 are special; real items start at 2

class SeqBackbone(nn.Module):
    """Bidirectional transformer encoder over item sequences.
    Reused for both masked-item pretraining and next-item finetuning."""
    def __init__(self, n_items, d_model=128, n_heads=4, n_layers=4,
                 max_len=50, dropout=0.2):
        super().__init__()
        self.item_emb = nn.Embedding(n_items + N_SPECIAL, d_model,
                                     padding_idx=PAD_ID)
        self.pos_emb = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model,
            dropout=dropout, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.ln = nn.LayerNorm(d_model)
        self.max_len = max_len

    def forward(self, seq):                       # seq: (B, L) of item ids
        B, L = seq.shape
        pos = torch.arange(L, device=seq.device).unsqueeze(0).expand(B, L)
        x = self.item_emb(seq) + self.pos_emb(pos)
        pad_mask = seq.eq(PAD_ID)                  # (B, L) True where padded
        h = self.encoder(x, src_key_padding_mask=pad_mask)
        return self.ln(h)                          # (B, L, d_model)


def mask_sequence(seq, n_items, mask_prob=0.15):
    """Cloze masking: replace mask_prob of non-pad positions with [MASK].
    Returns masked input and per-position targets (-100 = ignore)."""
    labels = seq.clone()
    probs = torch.rand(seq.shape, device=seq.device)
    maskable = seq.ne(PAD_ID)
    chosen = (probs < mask_prob) & maskable
    masked_input = seq.clone()
    masked_input[chosen] = MASK_ID
    labels[~chosen] = -100                          # ignore unmasked in loss
    return masked_input, labels


class MaskedItemHead(nn.Module):
    """Pretraining head: predict the masked item over the catalog.
    Ties weights to the item embedding table to save parameters."""
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.bias = nn.Parameter(torch.zeros(backbone.item_emb.num_embeddings))

    def forward(self, seq):
        h = self.backbone(seq)                      # (B, L, d)
        logits = h @ self.backbone.item_emb.weight.T + self.bias
        return logits                               # (B, L, n_items + 2)


def pretrain_epoch(model, loader, opt, n_items):
    model.train()
    total = 0.0
    for seq in loader:                              # seq: (B, L)
        masked, labels = mask_sequence(seq, n_items)
        logits = model(masked)                      # (B, L, V)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)), labels.view(-1),
            ignore_index=-100)
        opt.zero_grad(); loss.backward(); opt.step()
        total += loss.item()
    return total / len(loader)
```

After pretraining the backbone on the data-rich source, we **finetune on the target's next-item task.** We reuse the same `SeqBackbone` (loading its pretrained weights), attach a causal next-item head, and train with a left-to-right objective that matches serving. The key transfer move is loading `backbone.load_state_dict(pretrained)` and using a *lower learning rate on the backbone* than on the fresh head, so we adapt without clobbering what was learned.

```python
class NextItemModel(nn.Module):
    """Finetuning model: causal next-item prediction for serving."""
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.bias = nn.Parameter(torch.zeros(backbone.item_emb.num_embeddings))

    def forward(self, seq):
        L = seq.size(1)
        # causal mask so position t only sees <= t (matches serving)
        causal = torch.triu(torch.ones(L, L, device=seq.device),
                            diagonal=1).bool()
        # re-run encoder with causal mask
        B = seq.size(0)
        pos = torch.arange(L, device=seq.device).unsqueeze(0).expand(B, L)
        x = self.backbone.item_emb(seq) + self.backbone.pos_emb(pos)
        pad = seq.eq(PAD_ID)
        h = self.backbone.encoder(x, mask=causal, src_key_padding_mask=pad)
        h = self.backbone.ln(h)
        return h @ self.backbone.item_emb.weight.T + self.bias


def finetune(model, loader, n_items, lr_head=1e-3, lr_backbone=1e-4,
             epochs=20):
    # discriminative learning rates: gentle on backbone, faster on head
    head_params = [model.bias]
    backbone_params = list(model.backbone.parameters())
    opt = torch.optim.AdamW([
        {"params": backbone_params, "lr": lr_backbone},
        {"params": head_params, "lr": lr_head},
    ], weight_decay=0.01)
    for _ in range(epochs):
        model.train()
        for hist, target in loader:                 # hist:(B,L) target:(B,L)
            logits = model(hist)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), target.view(-1),
                ignore_index=-100)
            opt.zero_grad(); loss.backward(); opt.step()
    return model


# wiring it together
backbone = SeqBackbone(n_items=40000)
# ... pretrain on the data-rich source with MaskedItemHead(backbone) ...
ft_model = NextItemModel(backbone)                  # reuses pretrained weights
# ... finetune(ft_model, target_loader, n_items=40000) ...
```

Two engineering notes that matter. First, the masked-item pretraining is bidirectional but serving is causal (you only have the past), so the backbone sees both directions during pretraining and only the past during finetuning — this train-serve directionality shift is fine because finetuning re-adapts the attention to the causal regime, but you must not skip the finetune step and serve the masked model directly. Second, **tie the prediction weights to the item embedding table** (the `h @ item_emb.weight.T` above): it halves the head parameters and, more importantly, keeps the input and output item representations in the same space so the learned structure is consistent.

## 5. Content-based transfer: encoders that beat the cold start

ID embeddings are born random and only become useful after interactions accumulate. Content embeddings are born meaningful. That single difference is why content-based transfer is the most reliable cold-start tool in the box, and it deserves its own treatment because the design choice — *freeze or finetune the content encoder* — is one you will make on every project.

The idea: instead of (or alongside) representing item $i$ by a learned id vector $E(\text{id}_i)$, represent it by passing its content through a pretrained encoder. For text, a `sentence-transformers` model turns a title and description into a 384- or 768-dimensional vector. For images, CLIP turns a product photo into a vector. These encoders were pretrained on enormous general corpora and they already "know" that a science-fiction synopsis is near a space-opera synopsis, with zero recommender training. So a brand-new item, the instant it has a title, has a sensible embedding — no clicks required. We covered the content-feature plumbing in [content-based and hybrid recommenders](/blog/machine-learning/recommendation-systems/content-based-and-hybrid-recommenders); here the focus is the transfer-learning decision around the encoder.

![A two-column comparison showing a frozen content encoder with weights held fixed where only a small head trains fast and cheap reaching Recall at ten of 0.152 with stable cold-start behavior, versus a finetuned content encoder with weights updated that needs more target data and a low learning rate reaching 0.171 but carrying forgetting risk](/imgs/blogs/pretraining-and-finetuning-recommenders-5.png)

**Frozen encoder.** You run every item's content through the pretrained encoder once, cache the vectors, and treat them as fixed inputs. You train only a small projection head and the rest of the recommender on top. This is cheap (the expensive encoder runs once, offline), stable (the content space cannot drift or collapse during training), and robust in low data (very few parameters to fit). It is the right default, especially when the target data is scarce — there is not enough signal to safely move a 100M-parameter encoder, and a frozen encoder plus a small head is exactly the variance-reduction story from section 3.

**Finetuned encoder.** You let gradients flow into the content encoder so it adapts its representations to your catalog and objective — learning, say, that for *your* users the brand in the title matters more than the genre. This can lift accuracy when you have enough target data, but it carries two real risks. The first is **catastrophic forgetting**: a high learning rate on the encoder will quickly destroy the general semantic structure that made it useful, collapsing all items toward whatever the small target set rewards, which is exactly counterproductive for cold start. The second is **cost**: every training step now backpropagates through a large encoder. The figure above is the trade-off in one frame: frozen is cheap, stable, and great in low data; finetuned is more accurate with enough data but needs a low learning rate and risks forgetting.

The practical middle ground, and what I reach for most often, is a staged approach: freeze the encoder, train the head to convergence, *then* unfreeze the top layer or two of the encoder at a tiny learning rate (1e-5 to 1e-6) for a few epochs. You get most of the adaptation benefit with most of the stability. Parameter-efficient finetuning — adapters or LoRA, which insert small trainable matrices and freeze the original encoder weights — is the cleaner version of this for large encoders: you adapt with a few million trainable parameters instead of hundreds of millions, sidestepping both the forgetting and the cost, and it is the same machinery that the LLM4Rec finetuning posts will use on full language models.

### Practical: a content tower with a frozen-vs-finetuned switch

```python
from sentence_transformers import SentenceTransformer
import torch
import torch.nn as nn
import torch.nn.functional as F

class ContentItemTower(nn.Module):
    """Item embedding from text content, with a freeze switch on the encoder."""
    def __init__(self, encoder_name="all-MiniLM-L6-v2", out_dim=128,
                 freeze_encoder=True):
        super().__init__()
        self.encoder = SentenceTransformer(encoder_name)
        enc_dim = self.encoder.get_sentence_embedding_dimension()  # 384
        self.proj = nn.Sequential(
            nn.Linear(enc_dim, 256), nn.GELU(),
            nn.Dropout(0.1), nn.Linear(256, out_dim))
        self.freeze_encoder = freeze_encoder
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

    def forward(self, texts):                      # list[str] of item content
        # frozen path: cache encode() outputs offline for speed
        feats = self.encoder.encode(
            texts, convert_to_tensor=True,
            normalize_embeddings=True,
            device=next(self.proj.parameters()).device)
        if self.freeze_encoder:
            feats = feats.detach()                 # no grad into encoder
        return F.normalize(self.proj(feats), dim=-1)


# FROZEN: encode once, cache, train head only — cheap and stable
frozen_tower = ContentItemTower(freeze_encoder=True)

# FINETUNED: gradients flow into encoder — needs low LR, more data
ft_tower = ContentItemTower(freeze_encoder=False)
opt = torch.optim.AdamW([
    {"params": ft_tower.proj.parameters(),    "lr": 1e-3},
    {"params": ft_tower.encoder.parameters(), "lr": 1e-5},  # gentle!
], weight_decay=0.01)
```

The discriminative learning rates in the finetuned optimizer are not optional polish — they are the difference between adaptation and forgetting. A `1e-5` learning rate on the encoder nudges it; a `1e-3` would obliterate it.

#### Worked example: a cold-start item gets a transferable embedding

A new product lands in the catalog at 9am: a stainless steel French press, title and a two-sentence description, zero clicks. The id-embedding model represents it as a random vector — it will be recommended to no one until it accidentally gets a few clicks, and it may never get them because it is never shown (the cold-start death spiral). Now route its content through the frozen sentence-transformer. The description "double-walled stainless steel French press coffee maker, 1 liter" maps to a vector that is, by cosine similarity, 0.71 to existing "coffee maker" items, 0.63 to "kitchen brewing" items, and 0.12 to "garden tools." So at 9:01am, before a single click, the model can place the French press next to other coffee gear and surface it to users who browse coffee. The content embedding turned a zero-data item into a warm-started one using a function trained entirely outside the recommender. That is the whole game of content transfer: the encoder did the work, for free, the moment the item had a description.

## 6. Cross-domain recommendation: sharing users across catalogs

So far the transfer has been *within* one user population, across tasks or time. Cross-domain recommendation transfers *across catalogs* by exploiting the one thing two domains can genuinely share: **users.** A person who reads science-fiction books on your reading app is the same person who, on your video app, would enjoy science-fiction movies. If you can build a shared representation of that user, the dense book-reading history warm-starts the sparse movie history — before they have watched anything.

![A two-column comparison showing the movie domain alone where a new movie user has zero clicks and a cold profile so the system can only recommend generic popular items, versus a with-book-domain path where the same user has sci-fi book taste from two hundred forty reads that maps through a shared user representation into the movie domain and warm-starts sci-fi movie recommendations](/imgs/blogs/pretraining-and-finetuning-recommenders-6.png)

There are two structural ways domains overlap, and they call for different transfer designs.

**Shared users, disjoint items.** The same people use both apps; the catalogs are entirely separate. This is the classic cross-domain setup and the one the figure shows. The transfer carrier is the *user* representation: learn a user embedding that is informed by the source domain's behavior and use it (directly or through a learned mapping) to seed the target domain. CoNet (Hu et al., "CoNet: Collaborative Cross Networks for Cross-Domain Recommendation," CIKM 2018) does this with a dual network — one tower per domain — connected by *cross-connections* that let the two domains' hidden representations share knowledge through learned transfer matrices, so the book network's signal flows into the movie network and back. The cross-connections are the controlled channel through which book taste informs movie taste.

**Shared content, disjoint users.** Different user bases, but the items have comparable content (both are catalogs of text-described products, say). Here the transfer carrier is *content*: a content encoder gives aligned item representations across catalogs (section 5), and a model trained on the source's content-to-engagement mapping transfers to the target's items because the content space is shared.

The cross-domain payoff in the figure is concrete: a user with 240 sci-fi book reads and zero movie clicks. The movie-only model can only offer the generic popular feed. The cross-domain model maps their book taste through a shared user representation and warm-starts sci-fi movies on day one. That is real lift on exactly the cold users who are otherwise un-recommendable.

### Negative transfer: the risk that makes cross-domain hard

Cross-domain is not free upside, and pretending it is will burn you. The failure mode is **negative transfer**: the source domain pulls the target model in a direction that *hurts*, because the domains are less related than you assumed. If your reading app's users are mostly students reading textbooks and your video app's users want comedies, forcing a shared representation injects irrelevant or actively misleading signal. The shared-user assumption also leaks the source's biases: if the book domain over-represents one genre, the movie model inherits that skew.

Mathematically, negative transfer is what happens when the source task's optimal representation and the target task's optimal representation point in different directions, and the shared parameters cannot serve both — the model lands at a compromise that is worse for the target than target-only training would have been. The defenses: (1) measure relatedness before committing — if a model pretrained on the source does *not* beat from-scratch on a target validation set, the domains are not related enough and you should stop; (2) use *gated* or *partial* sharing (CoNet's cross-connections are gated; you can also share only lower layers) so the model can learn how much to transfer rather than being forced to share everything; (3) keep a domain-specific head so the shared backbone provides general structure and the head specializes. Negative transfer is the cross-domain analogue of the catastrophic forgetting we saw with content encoders, and the cure is the same family of moves: control *how much* transfers rather than transferring blindly.

## 7. Warm-start and continual finetuning over time

The most common — and most underrated — form of transfer in production has nothing to do with new domains. It is **transfer across time**: initialize today's model from yesterday's. Recommenders are retrained constantly (daily, hourly) as the feedback loop turns, and there are two ways to do each retrain. From-scratch retraining throws away the previous model and trains fresh on the latest window. Warm-start retraining initializes from the previous checkpoint and finetunes on the new data. Warm-start is usually faster to converge, more stable run-to-run (the embeddings do not jump to a new random basin every day), and it lets the model accumulate knowledge rather than forgetting it each cycle. We tie this to the feedback loop in [offline vs online: the two worlds of recsys](/blog/machine-learning/recommendation-systems/offline-vs-online-the-two-worlds-of-recsys); the warm-start retrain is the daily turn of that loop done right.

Continual finetuning has the same shape but a different goal: take one general model trained on all data and finetune a *copy* per segment — per country, per product line, per high-value customer cohort. Each segment gets a model specialized to its behavior while inheriting the general backbone's structure. This is transfer from the general to the specific, and it is enormously practical: you maintain one expensive pretrained backbone and many cheap finetuned heads.

Both run into the same hazard we keep meeting: **catastrophic forgetting.** When you finetune on the new window (or the new segment), the model can overfit the recent data and forget patterns that were rare recently but still matter — seasonal items, long-tail users, behaviors that spike monthly. The new data dominates the gradient and erases the old knowledge. The standard mitigations:

- **Replay.** Mix a sample of older data into each finetuning batch so the old distribution is never fully absent from the gradient. Even a 10% replay buffer dramatically reduces forgetting.
- **Low learning rate and few epochs on the warm start.** You are adapting, not retraining; a gentle update preserves the prior knowledge. This is the same discriminative-LR move from sections 4 and 5.
- **Regularize toward the previous weights.** Elastic Weight Consolidation (EWC) adds a penalty $\sum_i \frac{\lambda}{2} F_i (\theta_i - \theta_i^\text{old})^2$ that anchors important parameters (high Fisher information $F_i$) near their previous values while letting unimportant ones move freely. The model adapts where it can afford to and stays put where forgetting would hurt.
- **Periodic from-scratch anchor.** Even with warm-starting daily, retrain from scratch occasionally (weekly, monthly) to clear accumulated drift and re-anchor to the full distribution. Warm-start can slowly accumulate biases that a fresh train resets.

The trade-off is exactly bias-variance again. Warm-start reduces variance (stable, fast, accumulates knowledge) at the risk of bias from accumulated drift; the periodic from-scratch anchor pays variance to reset bias. A mature pipeline does both: warm-start the daily retrains, anchor from scratch periodically.

### A stress test: reasoning to a transfer decision under pressure

Let me work a realistic decision end to end, with the stress tests, because the framework is only useful if it survives contact with messy reality. The setup: you are launching a grocery recommender next to a mature general-merchandise recommender on the same platform. You have the general-merchandise behavior firehose (rich), a few thousand grocery interactions (poor), product titles and images for every grocery SKU (complete), and substantial user overlap (most grocery shoppers also buy general merchandise). What do you do?

Step one, name the scarcity. There are actually three cold problems here at once: cold *domain* (grocery as a whole is sparse), cold *items* (new SKUs arrive daily), and cold-ish *users* (some grocery-only shoppers). The matrix in figure 8 says different carriers win each column, so you will need a *combination*, not one method.

Step two, pick carriers per problem. For the cold domain, SSL-pretrain a sequence backbone on the general-merchandise firehose and finetune on grocery — sequence grammar (basket logic, repeat-purchase) transfers especially well to grocery, which is the most repeat-purchase-heavy vertical there is. For cold items, content transfer with a frozen encoder — a new SKU's title and image give it a vector immediately. For cold users, exploit the user overlap with a cross-domain shared user representation so a general-merchandise shopper's taste warm-starts their grocery feed. Combine them: a backbone pretrained on general behavior, item representations that fuse a frozen content embedding with a learned id embedding, and a shared user tower.

Now stress-test each choice.

*What happens with only implicit feedback?* Grocery is almost all implicit (purchases, no ratings), so your SSL objective and your finetune loss must both be implicit-friendly — masked-item and next-item are, BPR-style pairwise is, pointwise regression on a missing rating is not. Fine, the plan holds, but it means your negatives are the usual implicit-feedback minefield (see [implicit feedback models: ALS and BPR](/blog/machine-learning/recommendation-systems/implicit-feedback-models-als-and-bpr)) — un-purchased is not the same as disliked, and your sampled negatives during finetuning will include items the user simply never saw.

*What happens at 100M items?* The content-encoder forward over 100M SKUs is a one-time offline batch job (frozen encoder, cache the vectors), so it is fine — but the next-item softmax over 100M items during pretraining is not, and you must switch to sampled softmax with the $\log Q$ correction (see [sampled softmax and contrastive losses for retrieval](/blog/machine-learning/recommendation-systems/sampled-softmax-and-contrastive-losses-for-retrieval)). The plan holds with a loss swap.

*What happens when the source is not as related as you hoped?* Grocery basket dynamics might differ enough from general merchandise (grocery is high-frequency, low-consideration; electronics is the opposite) that the backbone transfers negatively. The test is built in: before committing, check that the pretrained-then-finetuned model beats grocery-from-scratch on a grocery validation set. If it does not, you have your answer — pretrain on a more related source (maybe a sibling grocery market) or drop behavioral pretraining and lean on content and cross-domain only. Do not ship a negative-transfer model because the architecture was elegant.

*What happens when the offline metric rises but online is flat?* This is the series' recurring nightmare and it is likely here, because your offline grocery test is drawn from the few early adopters, who are not representative of the post-launch population. The mitigation is to A/B on the cold slice specifically and to trust the *online* cold-item CTR over the offline Recall, and to expect the offline number to overstate the win. Plan for the gap rather than being surprised by it.

The decision that survives all four stress tests: SSL-pretrain on the most related available source *contingent on passing the relatedness test*, fuse a frozen content encoder for cold items, add a cross-domain user tower for overlap, use implicit-friendly losses with sampled softmax at scale, and validate online on the cold slice. That is the transfer toolkit assembled for a real launch — and notice that no single technique from this post is sufficient alone; the win is in matching each carrier to the scarcity it cures.

#### Worked example: warm-start versus daily from-scratch over two weeks

A team retrains a ranker nightly. With **daily from-scratch**, each night's model trains for 6 hours on the trailing 30-day window, and offline NDCG@10 bounces between 0.241 and 0.258 night to night — a 0.017 run-to-run swing — because each fresh init lands in a slightly different basin. Worse, every Monday the model has "forgotten" weekend-specific patterns it had learned the previous weekend, because the fresh init starts from nothing. With **warm-start**, each night initializes from the previous checkpoint and finetunes for 1.5 hours (4× faster, less to learn). NDCG@10 stabilizes at 0.255–0.262 (a 0.007 swing, less than half the volatility) and the weekend patterns persist because they are never fully erased. The catch: after three weeks of pure warm-starting, a long-tail-recall metric had quietly drifted down 0.004 as recent-popular patterns crowded out rare ones — caught and fixed by a monthly from-scratch anchor plus a 10% replay buffer on the daily finetunes. Warm-start bought 4× faster retrains and half the volatility; the replay buffer and the monthly anchor paid the small bias cost it incurred.

## 8. Results: from-scratch versus pretrain-then-finetune

Now the proof. The whole thesis is that finetuning beats from-scratch and that the gap *grows as target data shrinks.* Here is how to measure it honestly and what the numbers look like on a named benchmark.

**The experimental setup.** Use a sequential-recommendation benchmark with a clean temporal split — MovieLens-1M and the Amazon Reviews categories (Beauty, Sports, Toys) are the standard SASRec/BERT4Rec testbeds. To study transfer, treat one large category (or a large slice of MovieLens) as the data-rich *source* for pretraining, and a smaller, disjoint category as the data-poor *target* for finetuning. Critically, use a **leave-last-out temporal split**: for each user, the last item is the test target, the second-to-last is validation, the rest is training. This prevents leakage (you never train on the future) and matches how the model is actually used. Report **Recall@10** and **NDCG@10** computed over the *full* catalog as candidates, not a sampled subset — the KDD'20 result (Krichene and Rendle, "On Sampled Metrics for Item Recommendation," KDD 2020) showed that sampling candidates can reorder methods, so full-catalog evaluation is the trustworthy choice when feasible. To produce the data-efficiency curve, subsample the target *training* data to 100%, 10%, and 1% and retrain both the from-scratch and the finetuned model at each level; the test set is held fixed.

![A two-column comparison of Recall at ten across data regimes showing from-scratch reaching 0.192 at full data, 0.143 at ten percent, and 0.081 at one percent cold data, versus pretrained-then-finetuned reaching a tied 0.201 at full data but a winning 0.176 at ten percent and 0.149 at one percent cold data, demonstrating the gap widening as data shrinks](/imgs/blogs/pretraining-and-finetuning-recommenders-7.png)

The figure above is the data-efficiency curve as a before-after: at full data the two are nearly tied (0.192 vs 0.201, a +5% edge — pretraining barely matters when you have everything), at 10% the gap opens (0.143 vs 0.176, +23%), and at 1% cold data it is decisive (0.081 vs 0.149, +84%). The from-scratch model collapses toward popularity as data vanishes; the pretrained model degrades gracefully because the backbone already generalizes. This is the bias-variance prediction from section 3 made visible: the advantage of transfer is the vertical gap, and it is widest at the data-scarce left edge.

Here is the consolidated results table across methods and regimes. (These are representative numbers in the range reported by the SASRec, BERT4Rec, S3-Rec, and CL4SRec papers and consistent with what I have measured on Amazon-Beauty-scale targets; treat them as order-of-magnitude illustrations of the *pattern*, not as exact reproductions of any single paper, and re-run on your own split before trusting a precise figure.)

| Method | Recall@10 (full) | Recall@10 (10% data) | Recall@10 (cold-start items) | When it wins |
| --- | --- | --- | --- | --- |
| From scratch (SASRec) | 0.192 | 0.143 | 0.061 | You have abundant target data |
| SSL pretrain + finetune (S3-Rec style) | 0.201 | 0.176 | 0.118 | Sparse target, related source behavior |
| Contrastive pretrain (CL4SRec) | 0.198 | 0.172 | 0.109 | Sparse target, robustness to noise |
| Content transfer (frozen encoder) | 0.188 | 0.168 | 0.152 | Brand-new items, true cold start |
| Cross-domain shared user (CoNet) | 0.196 | 0.171 | 0.139 | Overlapping users across catalogs |

![A four-row by three-column matrix of method by Recall at ten across full data low data and cold-start regimes showing from-scratch baseline strong at full data but failing cold, SSL pretrain plus finetune winning in low data, content transfer frozen best on cold-start items, and cross-domain shared user strong in low data but needing user overlap](/imgs/blogs/pretraining-and-finetuning-recommenders-8.png)

The matrix figure above reads the same table by *regime* and is the decision aid: in the full-data column everything is tied (transfer barely helps), in the low-data column the SSL and cross-domain rows light up green (+20–25% over from-scratch), and in the cold-start-items column the *content* row wins outright (0.152) because content is the only thing that exists for an item with zero clicks — behavior-based transfer cannot help an item nobody has interacted with, but its title and description are there from minute one. Read the columns, not the rows: the right transfer method depends entirely on *which* kind of scarcity you face.

The NDCG@10 numbers track Recall but with a smaller absolute spread (NDCG rewards ranking the hit higher, and the pretrained models tend to rank their hits a position or two higher than from-scratch in the low-data regime): roughly 0.118 → 0.124 at full data, 0.082 → 0.103 at 10%, and 0.031 → 0.078 at cold-start for from-scratch versus SSL-pretrained. The relative pattern is identical: the win grows as data shrinks.

**How I would measure online, not just offline.** Offline Recall@10 is necessary but not sufficient — this series hammers the offline-online gap for a reason. To validate a transfer win online, A/B test the pretrained-finetuned model against the from-scratch incumbent *on the cold slice specifically* (new items, new users, new segment), because that is where transfer pays and an aggregate A/B would dilute the effect into noise. Watch cold-item CTR and cold-item coverage (what fraction of new items get any impressions), not just global CTR. A real shipped result in this shape: a content-transfer cold-start model lifted new-item CTR meaningfully and roughly halved the time-to-first-impression for new catalog items — the death-spiral fix — even though global CTR moved only slightly, because new items are a small fraction of traffic. Measure where the mechanism acts.

### The measurement traps that fake a transfer win

Three traps will make pretraining look like it works when it does not, and you should know all three because they are the reason offline transfer claims so often fail to replicate online.

**Trap one: a leaky split that lets the source contaminate the target test.** If your source and target share users and you split randomly rather than temporally, a user's *future* target interaction can leak into pretraining through the shared-user representation, and the test set is no longer a fair future. The fix is a global temporal cutoff: every pretraining interaction, source and target, must predate every target test interaction. Pretraining on the future is the most insidious form of leakage because it is invisible in the loss curves — everything looks great until production, where the future is genuinely unavailable.

**Trap two: a frozen target test that the pretraining secretly saw.** When you pretrain on "all behavior" and then carve out a target test set, make sure the test items and the test users' test interactions were *excluded* from pretraining. It is easy to pretrain on the whole log, then evaluate on a slice of that same log, and report a number that is partly memorization. The clean protocol: hold out the target test *first*, then pretrain on everything else.

**Trap three: sampled metrics that reorder methods.** As the KDD'20 result showed, scoring the true item against a small random sample of negatives (instead of the full catalog) produces metrics that can rank methods *differently* than full-catalog evaluation. Transfer methods that concentrate probability mass differently from-scratch are especially vulnerable to this reordering. If you must sample for speed during development, validate your final comparison on the full catalog before believing a ranking.

A blunt rule that catches most of these: if your pretrained model's offline win does *not* shrink toward zero as you increase target data, be suspicious — the data-efficiency theory predicts the gap closes at full data, and a gap that stays large at full data usually means leakage, not a miracle.

#### Worked example: the data-efficiency curve, point by point

Walk the figure-7 numbers as a curve to feel the mechanism. At **100% target data** (say 80,000 training interactions on the target category): from-scratch hits Recall@10 0.192, finetuned 0.201 — a +4.7% relative edge, basically a tie, because 80k interactions is enough to constrain the backbone from scratch. At **10% data** (8,000 interactions): from-scratch drops to 0.143 (it has started overfitting the sparse set), finetuned holds at 0.176 — now a +23% edge, because the backbone's structure is doing work the 8k interactions cannot do alone. At **1% data** (800 interactions, genuine cold-start territory): from-scratch collapses to 0.081 (essentially a popularity list — 800 interactions cannot constrain 40,000 item embeddings), while finetuned degrades only to 0.149 — a +84% edge. Plot those three points for each model and you see the two learning curves from section 3: from-scratch steep and starting low, finetuned gentle and starting high, converging at the right. The area between the curves is the total value of your pretraining investment, and three-quarters of it sits in the leftmost decile of data. If your product is mostly cold slices, that left edge is your whole business.

## 9. Case studies and the literature

Four lines of work anchor this post; here is what each actually showed.

**S3-Rec — self-supervised pretraining with mutual information (Zhou et al., CIKM 2020).** S3-Rec pretrains a sequence model with four self-supervised objectives that maximize mutual information between items, attributes, sub-sequences, and the full sequence — teaching the model that an item belongs with its attributes and that a sub-sequence belongs to its parent — *before* the next-item finetuning. On Amazon and Yelp benchmarks it reported double-digit relative improvements in NDCG@10 over SASRec-from-scratch, with the gains concentrated where sequences are short and sparse. The lesson that generalizes: SSL pretraining extracts structure from interactions that a from-scratch next-item loss leaves on the table, and that extra structure is worth the most exactly when interactions are scarce.

**CL4SRec — contrastive learning for sequences (Xie et al., ICDE 2022).** CL4SRec augments each sequence (crop, mask, reorder) into two views and trains an InfoNCE loss so the two views agree, jointly with the next-item loss. The contrastive objective makes the representation robust to the noise and order-jitter of real logs, and the paper showed consistent Recall and NDCG gains over SASRec, again largest on sparse datasets. The lesson: contrastive augmentation is a cheap, architecture-agnostic regularizer that improves the transferable representation without any extra labels.

**CoNet — cross-domain via cross-connections (Hu et al., CIKM 2018).** CoNet ran a dual network — one per domain — wired together by gated cross-connections that let each domain's hidden representations inform the other's, transferring knowledge through shared users. On Amazon cross-domain pairs (e.g., books → movies) it beat single-domain baselines, with the benefit concentrated on users with sparse target-domain histories. The lesson: cross-domain transfer is real and the gain lands on the cold users, but the *gating* matters — uncontrolled sharing invites negative transfer.

**Content transfer for cold start (the general pattern).** This one is less a single paper than a pattern that recurs across YouTube, Pinterest, and many e-commerce systems: represent items by content embeddings (text, image) from a pretrained encoder so new items have sensible representations from creation, then learn the recommender on top. Pinterest's PinSage and successors lean heavily on content/graph features so that a freshly created pin is recommendable immediately; YouTube's retrieval and ranking stacks blend content features with behavioral ones for the same reason. The lesson, repeated across shipped systems: content is the only signal that exists before behavior, so content transfer is the load-bearing cold-start mechanism, and it composes with the behavioral SSL pretraining rather than replacing it.

A meta-point across all four: none of them transfer the id embeddings. S3-Rec and CL4SRec transfer the *backbone* via SSL; CoNet transfers the *user* representation via cross-connections; content transfer sidesteps ids with the *content encoder*. The thing that does not transfer is never transferred; the things that do are. The whole field converged on the section-2 cheat sheet.

## 10. When to pretrain/finetune vs train from scratch

A decisive recommendation section, because every choice is a cost and you should know when transfer is *not* worth it.

**Reach for pretrain-then-finetune when:**

- **Your target is data-poor and a related source is data-rich.** This is the canonical win and the bigger the source-target data ratio, the bigger the payoff. New domain, new segment, niche task next to a general corpus.
- **You face genuine cold start** — new items or new users with zero behavior. Use *content* transfer (frozen encoder) for new items; use *cross-domain* or *content* transfer for new users. Behavioral SSL pretraining helps warm-ish-but-sparse, not zero-behavior, items.
- **You retrain frequently.** Warm-start every retrain from the previous checkpoint; it is faster and more stable. This is almost always correct in production.
- **You maintain many specialized models.** Pretrain one general backbone, finetune per segment. Cheaper to maintain and each segment benefits from the shared structure.

**Train from scratch (or skip the pretraining) when:**

- **You already have abundant target data.** At full data the transfer edge is a few percent and may not justify the pipeline complexity. If your warm head has plenty of interactions, from-scratch is fine there — apply transfer only to the cold slices.
- **No related source exists, or the source is unrelated.** If a model pretrained on the source does not beat from-scratch on your target validation set, the domains are not related enough — pretraining will at best waste effort and at worst cause negative transfer. *Always test this before committing.*
- **The cost of the pretraining pipeline exceeds the gain.** Pretraining a backbone, maintaining two-stage training, and managing forgetting is real engineering. If the target is not actually cold and the lift is marginal, the simpler from-scratch pipeline is the better *system*, even if it loses a fraction of a point offline.

**Decide which transfer carrier by which scarcity you face.** New items with zero clicks: content transfer is the *only* thing that works (there is no behavior to transfer). Sparse-but-nonzero target behavior with a related source: SSL backbone pretraining. Overlapping users across catalogs: cross-domain shared-user transfer. Frequent retrains: warm-start. The matrix in figure 8 is this decision in one frame — match the column (your scarcity) to the winning row (your method).

A few sharp anti-patterns, stated plainly: don't finetune a 100M-parameter content encoder on a few thousand interactions (it will forget faster than it learns — freeze it or use LoRA); don't force cross-domain sharing without measuring relatedness first (negative transfer is real and silent); don't warm-start forever without a periodic from-scratch anchor (accumulated drift); don't serve a bidirectionally-pretrained masked model directly (finetune it to the causal serving regime first); and don't pretrain at all if your target is already data-rich and there is no cold slice to rescue (you are adding complexity for a rounding error).

## Key takeaways

- **IDs don't transfer; structure does.** Item id embeddings live in unaligned per-catalog coordinate frames and carry no cross-domain meaning. Sequence patterns, behavior shape, and content semantics are shared across domains — transfer those and relearn the ids.
- **The finetuning advantage grows as target data shrinks.** It is a bias-variance / data-efficiency effect: pretraining reduces variance, which dominates the error in low data and vanishes in high data. At full data, transfer barely helps; at 1% data it can be decisive.
- **Self-supervision turns behavior into free labels.** Masked-item modeling (BERT4Rec-style) and contrastive sequence learning (CL4SRec, S3-Rec) pretrain a reusable backbone from unlabeled logs, extracting far more signal per interaction than a from-scratch next-item loss.
- **Content encoders are the cold-start cure.** A pretrained text/image encoder gives every item a meaningful, transferable embedding the moment it has content — before any clicks. Freeze it by default; finetune only with enough data and a tiny learning rate, or use LoRA.
- **Cross-domain transfer rides shared users, and risks negative transfer.** Map a dense source-domain user representation into a sparse target. Gate the sharing and measure relatedness first — uncontrolled sharing can hurt the target.
- **Warm-start your retrains, but anchor periodically.** Initialize today's model from yesterday's for speed and stability; guard against catastrophic forgetting with replay, low learning rates, and an occasional from-scratch anchor.
- **Measure transfer where it acts.** Validate cold-start and low-data wins on the cold slice specifically (cold-item CTR, time-to-first-impression), not on a global A/B that dilutes the effect into noise.
- **Test before you transfer.** If a model pretrained on the source does not beat from-scratch on your target validation set, the domains are not related enough — and at full target data, the simpler from-scratch pipeline may be the better system.

## Further reading

- Kang and McAuley, "Self-Attentive Sequential Recommendation" (SASRec), ICDM 2018 — the causal next-item backbone used for both pretraining and finetuning here.
- Sun et al., "BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer," CIKM 2019 — the masked-item (cloze) SSL objective.
- Zhou et al., "S3-Rec: Self-Supervised Learning for Sequential Recommendation with Mutual Information Maximization," CIKM 2020 — multi-objective SSL pretraining; double-digit gains on sparse sequences.
- Xie et al., "Contrastive Learning for Sequential Recommendation" (CL4SRec), ICDE 2022 — sequence augmentation and the InfoNCE objective for robust representations.
- Hu et al., "CoNet: Collaborative Cross Networks for Cross-Domain Recommendation," CIKM 2018 — gated cross-connections for shared-user transfer.
- Krichene and Rendle, "On Sampled Metrics for Item Recommendation," KDD 2020 — why to evaluate Recall@K and NDCG@K over the full catalog, not a sampled subset.
- `sentence-transformers` documentation (sbert.net) — pretrained text encoders for content embeddings, frozen or finetuned.
- Within this series: [self-attention for sequences: SASRec and BERT4Rec](/blog/machine-learning/recommendation-systems/self-attention-for-sequences-sasrec-bert4rec) for the backbone architecture, [content-based and hybrid recommenders](/blog/machine-learning/recommendation-systems/content-based-and-hybrid-recommenders) and [the cold-start problem](/blog/machine-learning/recommendation-systems/the-cold-start-problem) for content features and cold start, [LLMs for recommendation (LLM4Rec)](/blog/machine-learning/recommendation-systems/llms-for-recommendation-llm4rec) and [finetuning LLMs for recommendation in practice](/blog/machine-learning/recommendation-systems/finetuning-llms-for-recommendation-in-practice) for where this leads, and the capstone [the recommender systems playbook](/blog/machine-learning/recommendation-systems/the-recommender-systems-playbook).
