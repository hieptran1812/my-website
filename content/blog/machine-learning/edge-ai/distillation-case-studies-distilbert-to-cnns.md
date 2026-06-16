---
title: "Distillation case studies: DistilBERT, TinyBERT, MobileBERT, and CNNs"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "Dissect the four landmark knowledge-distillation recipes that actually ship — DistilBERT's triple loss, TinyBERT's two-stage layer matching, MobileBERT's bottleneck transfer, and CNN feature distillation — with the math, runnable code, and reported before-after numbers."
tags:
  [
    "edge-ai",
    "model-optimization",
    "knowledge-distillation",
    "distilbert",
    "tinybert",
    "mobilebert",
    "transformers",
    "inference",
    "efficient-ml",
  ]
category: "machine-learning"
subcategory: "Edge AI"
author: "Hiep Tran"
featured: true
readTime: 54
image: "/imgs/blogs/distillation-case-studies-distilbert-to-cnns-1.png"
---

The first time I shipped a BERT model into a product, the model worked beautifully and the latency budget did not. A 110-million-parameter encoder, fp32, running on the modest CPU instances we could actually afford, gave us a p99 north of 200 ms per request for a classifier that needed to clear 40 ms. The data scientist who trained it shrugged and said "use a smaller model." But the smaller models we tried — BERT trained from scratch with fewer layers — lost three or four accuracy points, which was exactly the difference between "ship it" and "do not ship it." We were stuck on the wrong part of the trade-off curve: every step toward the latency target cost us accuracy we could not give up.

The thing that got us unstuck was not a new architecture or a bigger GPU. It was knowledge distillation, and specifically the realization that someone had already done the hard work and published the recipe. We swapped in DistilBERT — six layers instead of twelve, 40 percent smaller, 60 percent faster — and it kept about 97 percent of the original's accuracy on our task after a short fine-tune. The latency problem evaporated. The accuracy we had been fighting to preserve came along for free, because DistilBERT was not a small model trained from scratch; it was a small model trained to *imitate* a big one, and that imitation carries far more signal than the hard labels alone.

This post is about *how those recipes actually work* — not the one-paragraph theory of distillation, which we cover in [knowledge distillation fundamentals](/blog/machine-learning/edge-ai/knowledge-distillation-fundamentals), and not the taxonomy of *what* you can distill, covered in [what to distill: response, feature, relation](/blog/machine-learning/edge-ai/what-to-distill-response-feature-relation). It is about the four landmark distilled models that turned that theory into things you can `pip install` and deploy: **DistilBERT**, **TinyBERT**, **MobileBERT**, and the broad family of **CNN distillations** (ResNet or an ensemble compressed into a MobileNet-class student). For each, we will pull the recipe apart term by term, do the math on the loss, look at the reported numbers, and figure out *why* the design choices matter. Figure 1 is the whole DistilBERT recipe on one slide — keep it in mind, because the other three are variations on the same theme of "give the student more than just the right answer."

![A branching dataflow diagram showing a 12-layer BERT teacher and a 6-layer student feeding three loss terms, soft logits, cosine embedding, and masked language modeling, which merge into a result of 40 percent smaller and 60 percent faster at 97 percent of GLUE](/imgs/blogs/distillation-case-studies-distilbert-to-cnns-1.png)

By the end you will be able to: read any of these four papers and know exactly which signals the student is matching and why; write the multi-term distillation loss in PyTorch / Hugging Face yourself; reproduce DistilBERT's loss from scratch; pick the right recipe for *your* compression target; and compose distillation with quantization to land on the accuracy–latency Pareto frontier this whole series is organized around (see [a taxonomy of model compression](/blog/machine-learning/edge-ai/a-taxonomy-of-model-compression) for that frame). Distillation is the one lever of the four that *adds* accuracy back rather than spending it, which is exactly why it pairs so well with the others.

## 1. The one idea behind all four recipes

Strip away the architecture-specific cleverness and every one of these models rests on a single observation from Hinton, Vinyals, and Dean's 2015 paper *Distilling the Knowledge in a Neural Network*: a trained teacher's output distribution contains far more information than the one-hot label it was trained against. When a strong image classifier sees a photo of a dog, it does not just say "dog." It says "92 percent dog, 5 percent wolf, 2 percent fox, 0.01 percent truck." That ranking of the wrong answers — the *dark knowledge* — encodes the teacher's learned similarity structure: dogs are like wolves, not like trucks. A student trained to match that full distribution learns the structure, not just the answer.

Formally, both teacher and student emit logits $z$, which a softmax with temperature $T$ turns into a probability distribution:

$$p_i(T) = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$

At $T = 1$ this is the ordinary softmax. As $T$ grows, the distribution *softens* — the gap between the top class and the runners-up shrinks, so the relative probabilities of the wrong classes become visible and trainable instead of being crushed to near-zero. The distillation loss is the cross-entropy (equivalently, up to a constant, the KL divergence) between the softened teacher and softened student distributions:

$$\mathcal{L}_{\text{soft}} = T^2 \cdot \mathrm{KL}\!\left(p^{\text{teacher}}(T) \,\Vert\, p^{\text{student}}(T)\right)$$

The $T^2$ factor is not decoration. Differentiating the softened cross-entropy, the gradient magnitude scales like $1/T^2$; multiplying the loss by $T^2$ restores the gradient to roughly the same scale as the hard-label gradient, so you can mix the two losses with stable, comparable weights regardless of the temperature you chose. Forget the $T^2$ and your soft loss quietly vanishes at high temperature and the student just learns the hard labels.

It is worth deriving that scaling once so the $T^2$ stops being a magic constant. Write the student's softened cross-entropy loss against the teacher as $C = -\sum_i p_i^{T} \log p_i^{S}$, where $p_i^{S} = \mathrm{softmax}(z_i^{S}/T)$ and the superscripts mark student and teacher. The gradient with respect to a student logit $z_k^{S}$ is the familiar softmax-cross-entropy form, but evaluated on the *scaled* logits, so the chain rule pulls out a $1/T$ from the $z/T$ inside the softmax:

$$\frac{\partial C}{\partial z_k^{S}} = \frac{1}{T}\left(p_k^{S} - p_k^{T}\right)$$

Now expand both softmaxes for *large* $T$. A first-order Taylor expansion of $\exp(z/T) \approx 1 + z/T$ gives, for zero-mean logits, $p_k \approx \frac{1}{N}\left(1 + z_k/T\right)$ where $N$ is the number of classes. Substituting,

$$\frac{\partial C}{\partial z_k^{S}} \approx \frac{1}{T}\left(\frac{1 + z_k^{S}/T}{N} - \frac{1 + z_k^{T}/T}{N}\right) = \frac{1}{N T^2}\left(z_k^{S} - z_k^{T}\right)$$

So at high temperature the soft-target gradient is proportional to the *logit difference* — distillation is, in this regime, literally matching the teacher's logits — and it carries a $1/T^2$ out front. Multiply the loss by $T^2$ and that factor cancels, leaving a gradient on the same scale as the hard-label cross-entropy. That is why DistilBERT can use fixed weights $5.0$ and $2.0$ on its soft and hard terms regardless of the temperature it picked: the $T^2$ already normalized the scales. It also tells you what temperature *does* — at $T \to \infty$ you match raw logits (all the dark knowledge, very gentle gradients); at $T = 1$ you match the sharp output distribution (most signal on the top class). Practitioners land in between, $T \in [2, 10]$, because that is where the wrong-class structure is visible without the gradients going to mush.

That is the *response-based* distillation we will see in DistilBERT's logit term and in every CNN recipe. The richer recipes — TinyBERT, MobileBERT, and the feature term in DistilBERT — add *feature-based* distillation: they also force the student's *internal* representations (hidden states, attention matrices, embeddings) to resemble the teacher's. The intuition, which we make rigorous in section 7, is that a final-logit match leaves the student's interior unconstrained, and for deep transformers that interior is where most of the capacity lives. The four models differ mostly in *how deep into the network the supervision reaches*.

#### Worked example: why soft targets beat hard labels

Take a three-class sentiment task — negative, neutral, positive — and a single review: "It was fine, I guess." The hard label is `neutral`, a one-hot vector $[0, 1, 0]$. That is all a from-scratch student ever sees. The teacher, at $T = 2$, emits something like $[0.18, 0.62, 0.20]$: mostly neutral, but leaning very slightly positive and clearly not strongly negative. That single soft vector teaches the student three things the hard label cannot — that this example is ambiguous, that positive is a closer second guess than negative, and roughly *how* ambiguous. Multiply that across millions of training tokens and the student inherits a calibrated picture of the decision boundary instead of a stack of one-hot cliffs. This is why a distilled student routinely beats an identically-sized student trained from scratch by 2–4 points: same parameters, vastly richer supervision.

#### Worked example: choosing the temperature on a 5-class task

Take a 5-class news classifier and a teacher that, on one borderline article, emits logits $[2.0, 1.5, 0.0, -1.0, -2.0]$ over {politics, business, tech, sports, weather}. At $T = 1$ the softmax is roughly $[0.55, 0.33, 0.07, 0.03, 0.01]$ — almost all the signal is "politics, maybe business," and the distinction between sports and weather (both near zero) is invisible to the student. At $T = 4$ the same logits soften to about $[0.30, 0.27, 0.18, 0.14, 0.11]$ — now the *full ordering* is visible and trainable: politics > business > tech > sports > weather. The student learns that this article is closer to sports than to weather, a relationship the sharp distribution and certainly the one-hot label never expressed. Push to $T = 16$ and the distribution flattens toward uniform $[0.21, 0.20, 0.20, 0.20, 0.19]$ — now there is almost no signal *and* the gradients are tiny, so training stalls. This is why the practical sweet spot is a moderate temperature: enough to expose the wrong-class ranking, not so much that everything collapses to uniform. For most classification distillation, $T$ between 2 and 8 and an ablation over two or three values is all the tuning the temperature ever needs.

## 2. DistilBERT: the triple loss, dissected

DistilBERT (Sanh, Debut, Chaumond, and Wolf, 2019, from Hugging Face) is the cleanest place to start because its recipe is minimal and every choice is deliberate. The student is a BERT with **half the layers** — 6 transformer blocks instead of 12 — but the *same* hidden width (768) and the same vocabulary and embeddings. That gives 66M parameters versus the teacher's 110M, about 40 percent smaller. Critically, the student is **distilled during pre-training**, on the same masked-language-modeling objective and corpus BERT itself was trained on, so the result is a general-purpose encoder you then fine-tune like any other BERT — not a single-task model.

Three design decisions make it work. Let's take them in order.

### 2.1 The three loss terms

DistilBERT trains the student against a weighted sum of three losses, shown decomposed in Figure 2. Each supplies a different kind of signal.

![A vertical stack diagram decomposing the DistilBERT objective into a soft-logit KL term, a masked language modeling cross-entropy term, and a cosine embedding term, summing to a weighted total with initialization from teacher layers below](/imgs/blogs/distillation-case-studies-distilbert-to-cnns-3.png)

**Term 1 — the soft-target loss $\mathcal{L}_{ce}$.** This is the Hinton KL between the teacher's and student's softened output distributions over the vocabulary, exactly as in section 1. For a masked-LM teacher this is the distribution over the entire 30,522-token vocabulary at each masked position — a very rich target. Sanh et al. use a temperature $T$ during distillation and set it back to 1 at inference. This term is the heart of the recipe and carries the largest weight.

**Term 2 — the masked-LM loss $\mathcal{L}_{mlm}$.** This is the ordinary supervised BERT objective: mask 15 percent of tokens, predict them, cross-entropy against the *true* tokens. It anchors the student to the ground truth so it does not drift to merely mimicking the teacher's mistakes. Soft targets alone can amplify systematic teacher errors; the hard target keeps the student honest.

**Term 3 — the cosine-embedding loss $\mathcal{L}_{cos}$.** This is the feature-based piece. It pushes the *direction* of the student's hidden-state vectors to align with the teacher's:

$$\mathcal{L}_{cos} = 1 - \cos\!\left(h^{\text{student}}, h^{\text{teacher}}\right) = 1 - \frac{h^{\text{student}} \cdot h^{\text{teacher}}}{\lVert h^{\text{student}}\rVert\,\lVert h^{\text{teacher}}\rVert}$$

Because student and teacher share the hidden width (768), you can compare hidden states directly with no projection. Cosine, not MSE, is the deliberate choice: it constrains the *geometry* — the angle, the relative arrangement of the representation space — without forcing the student to match the teacher's exact magnitudes, which would over-constrain a smaller model.

The total objective is a fixed linear combination. The paper's weights are $5.0$ on the soft-logit term, $2.0$ on the masked-LM term, and $1.0$ on the cosine term:

$$\mathcal{L}_{\text{DistilBERT}} = 5.0\,\mathcal{L}_{ce} + 2.0\,\mathcal{L}_{mlm} + 1.0\,\mathcal{L}_{cos}$$

The heavy weight on the soft term is the statement of intent: *imitating the teacher is the primary job; matching ground truth and aligning geometry are correctives.*

### 2.2 Initialization from the teacher

This is the choice people skip and then wonder why their reimplementation underperforms. DistilBERT's six student layers are not initialized randomly. They are **copied from the teacher**, taking every other layer (layers 1, 3, 5, 7, 9, 11 of the 12, or an equivalent even subset). Because the architectures are identical block-for-block and the widths match, you can literally load the teacher's weights into the student's slots.

Why does it matter so much? A randomly-initialized 6-layer transformer has to discover from scratch a representation that the distillation loss is simultaneously trying to align with a *specific* teacher. That is two moving targets. Initializing from teacher layers starts the student inside the teacher's basin of attraction — its representations already point roughly the right direction, so the cosine loss starts small and the optimization is mostly *refinement* rather than *discovery*. We will quantify this in section 7, but the headline is that init-from-teacher is worth a meaningful chunk of the final accuracy and is nearly free to do.

### 2.3 The reported numbers

The numbers Sanh et al. report, and the reason DistilBERT became the default small encoder for years: the student retains **97 percent of BERT-base's performance on the GLUE benchmark** while being **40 percent smaller** (66M vs 110M parameters) and **60 percent faster** at inference. On a CPU, the per-query latency drop is the whole point — halving the layer count halves the dominant matrix-multiply cost. The distillation pre-training itself took about 90 hours on 8 16GB V100s, which is cheap compared to pre-training BERT from scratch, because the student starts from teacher weights and only needs to refine.

It is worth being precise about *where* the 40 percent and 60 percent come from, because the asymmetry between them is instructive. The parameter reduction is *only* 40 percent, not 50, even though the layer count is halved — and the reason is that the embedding matrix (vocabulary size 30,522 times hidden width 768, roughly 23M parameters) is shared between teacher and student and is *not* halved. Of BERT-base's 110M parameters, about 23M live in the embeddings and the rest in the 12 transformer blocks; halving the blocks removes roughly 44M, landing at 66M, which is 40 percent off, not 50. The speedup, meanwhile, is *better* than the parameter ratio — 60 percent faster, i.e. about 1.6×, despite only a 1.7× parameter cut — because the embedding lookup is nearly free at inference (a gather, not a matmul), so the wall-clock cost is dominated by the transformer blocks you *did* halve. This is the general edge lesson in miniature: parameters and latency are not the same currency, and which one moves depends on *where* the parameters are and whether they sit on the compute-bound path. DistilBERT also dropped BERT's token-type embeddings and the next-sentence-prediction objective, both small simplifications that the authors found cost nothing on downstream tasks.

One more deployment detail that matters in practice: because DistilBERT keeps BERT's hidden width, vocabulary, and tokenizer, it is a *drop-in* replacement in almost any BERT pipeline — same `AutoTokenizer`, same input format, same fine-tuning code, only the model name changes. That drop-in property, more than any single benchmark number, is why it spread so fast. A team facing my opening latency problem did not have to redesign anything; they changed one string and re-fine-tuned.

#### Worked example: DistilBERT's loss, computed on one masked position

Make this concrete. One masked token, true word "bank." The teacher, at $T = 2$, puts probability mass like `bank: 0.55, river: 0.20, money: 0.15, dog: 0.001, ...` across the vocabulary — note it has *learned* that "bank" is ambiguous between the financial and river senses, which the one-hot label "bank" cannot express. Suppose the student currently predicts `bank: 0.40, river: 0.10, money: 0.10, ...`.

- $\mathcal{L}_{ce}$: the temperature-scaled KL from teacher to student, times $T^2 = 4$. Because the student under-weights "river" and "money," this term pushes it to *spread* probability the way the teacher does, teaching it the two senses.
- $\mathcal{L}_{mlm}$: ordinary cross-entropy against the true token "bank" — $-\log(0.40)$ at $T=1$. This keeps the top prediction correct.
- $\mathcal{L}_{cos}$: one minus the cosine between the student's and teacher's 768-d hidden vector at this position. If they point at, say, 30 degrees apart, $\cos 30° \approx 0.87$, so this term contributes $\approx 0.13$, nudging the student's representation to rotate toward the teacher's.

The weighted sum $5.0\,\mathcal{L}_{ce} + 2.0\,\mathcal{L}_{mlm} + 1.0\,\mathcal{L}_{cos}$ is backpropagated through the 6-layer student only — the teacher is frozen. Repeat across the whole corpus and the student converges to a 66M-parameter encoder that, fine-tuned on a downstream task, lands within a few tenths of a point of the 110M teacher on most GLUE tasks.

## 3. TinyBERT: two stages and matching every layer

DistilBERT matches logits and one geometry term. TinyBERT (Jiao et al., 2020) asks: what if we match *everything, at every layer, twice*? The result is far more aggressive compression — a 4-layer student (TinyBERT-4) with a 312 hidden dimension is roughly **7.5× smaller and 9.4× faster** than BERT-base while keeping about 96 percent of its quality — but the recipe is correspondingly heavier. Figure 3 lays out the two-stage flow as a timeline.

![A left-to-right timeline showing TinyBERT's general distillation stage matching embeddings attention and hidden states on a corpus, then a data augmentation step multiplying examples, then a task distillation stage on augmented data, ending at 7.5 times smaller and 96 percent of BERT](/imgs/blogs/distillation-case-studies-distilbert-to-cnns-4.png)

### 3.1 The four things TinyBERT matches

Within each layer, TinyBERT defines a *transformer-layer distillation* objective with several components. Because the student is narrower than the teacher (312 vs 768 hidden), a learned linear projection $W$ maps student features up into the teacher's space before comparing — this is the key trick that lets a *narrower* student be supervised by a wide teacher.

**Embedding-layer distillation.** Match the student's input embeddings to the teacher's, through a projection: $\mathcal{L}_{embd} = \mathrm{MSE}(E^S W_e, E^T)$. The student's vocabulary table is forced to live in the teacher's embedding geometry.

**Attention-matrix distillation.** This is the signature TinyBERT idea. For each head, the *unnormalized attention scores* $A = QK^\top / \sqrt{d}$ (before softmax) carry the linguistic structure the model has learned — who attends to whom. TinyBERT matches them directly with MSE:

$$\mathcal{L}_{attn} = \frac{1}{h} \sum_{i=1}^{h} \mathrm{MSE}\!\left(A_i^{S}, A_i^{T}\right)$$

where $h$ is the number of heads. The authors found matching the pre-softmax scores converges faster than matching the post-softmax weights. Attention encodes syntactic and coreference structure; transplanting it directly into the student is enormously informative.

**Hidden-state distillation.** Match the per-layer hidden states, with the same up-projection to bridge the width gap: $\mathcal{L}_{hidn} = \mathrm{MSE}(H^S W_h, H^T)$. This is the deepest form of feature supervision — the student must reproduce the teacher's representation at every layer, not just the output.

**Prediction-layer distillation.** Finally, the Hinton soft-logit KL on the output, identical in spirit to DistilBERT's $\mathcal{L}_{ce}$.

A *layer mapping function* $g(\cdot)$ decides which teacher layer supervises which student layer: student layer $m$ learns from teacher layer $n = g(m)$. For a 4-layer student distilling from a 12-layer teacher, the natural uniform map is $g(m) = 3m$ — student layers 1, 2, 3, 4 learn from teacher layers 3, 6, 9, 12. The choice is not innocent. A *uniform* map spreads supervision across the teacher's depth, so the student's four layers each absorb a "chunk" of the teacher's twelve-layer computation. A *top-heavy* map (e.g. learn only from the last few teacher layers) gives the student the teacher's most task-specialized representations but starves it of the lower-level structure built in early layers; a *bottom-heavy* map does the reverse. TinyBERT uses the uniform map and adds the embedding layer at the bottom (student layer 0 learns teacher layer 0) and the prediction layer at the top. The total transformer-layer objective is then the sum over the mapped layers:

$$\mathcal{L}_{\text{TinyBERT}} = \sum_{m} \mathcal{L}_{\text{layer}}\!\left(S_m,\, T_{g(m)}\right)$$

where each $\mathcal{L}_{\text{layer}}$ contributes the embedding, attention, and hidden-state MSE terms appropriate to that layer. This is the structural decision that makes layer-by-layer matching well-defined across mismatched depths, and it is the part people most often get subtly wrong — an off-by-one in the mapping silently degrades the student because it is now imitating the wrong teacher layer.

There is a subtlety in the attention matching worth dwelling on, because it is the most TinyBERT-specific choice. The attention score matrix $A \in \mathbb{R}^{l \times l}$ (sequence length $l$) is matched *per head* with MSE on the *unnormalized* scores $QK^\top/\sqrt{d}$. Why unnormalized? After softmax, attention weights are a probability distribution per row, often sharply peaked, and the gradient of an MSE on near-saturated softmax outputs is tiny (the softmax Jacobian vanishes where one entry approaches 1). Matching the pre-softmax scores keeps the gradients healthy and, the authors found, converges faster and to a better student. It also means the student is forced to learn the *same relative preferences* between key positions, not just the same final weights — a richer constraint. The cost is that you must expose the attention scores from both models, which in `transformers` means `output_attentions=True` on both teacher and student forward passes and then matching the right heads under the layer map.

### 3.2 Two-stage distillation

The second big idea is that you distill *twice*:

1. **General distillation.** Run the transformer-layer objective (embedding + attention + hidden states) on a large general corpus, exactly as in pre-training. This produces a general TinyBERT whose layers already mirror the teacher's. There is *no* prediction loss here — the student has no task yet — just the structural feature matching.
2. **Task-specific distillation.** Fine-tune the teacher on the downstream task, then distill again on the task data, now including the prediction (logit) loss. The student specializes while keeping its teacher-aligned interior.

The reason for two stages is a division of labor: general distillation gives the student a strong, transferable representation; task distillation specializes it without having to relearn the representation from a small task dataset. Skipping the general stage and only doing task distillation costs several points — the student never builds the rich interior that the small task set cannot teach.

### 3.3 Data augmentation: the quiet hero

Task datasets are small — GLUE tasks range from a few thousand to a few hundred thousand examples — and the richest part of distillation, the dark knowledge in the teacher's soft outputs, is only sampled where you have inputs to feed the teacher. So TinyBERT *augments the task data roughly 20×* before task distillation. The procedure: take each training sentence, and for each word, either replace it with a BERT-predicted alternative (for single-piece words) or with a GloVe nearest neighbor (for multi-piece words), with some probability. This manufactures many plausible variants of each example.

The augmentation does not add new labels — it adds new *inputs on which to query the teacher*. Each synthetic sentence is a fresh place to extract the teacher's soft distribution and attention structure. It densely samples the input region around the real data, so the student learns the teacher's behavior on a much larger effective support. The authors report that augmentation is responsible for a large share of TinyBERT's final accuracy; remove it and the small student overfits the tiny real task set.

Think of it as turning the teacher into a *labeling oracle* you can query for free. The teacher is a fixed function; every input you feed it returns a soft label that is far richer than any human annotation. The bottleneck on distillation quality is therefore not labels — those are unlimited — but *inputs at which the teacher's behavior is worth learning*. Augmentation manufactures those inputs. The reason word-level replacement (BERT-predicted substitutes for single-piece tokens, GloVe neighbors for the rest) works better than naive noise is that it stays on the *natural-language manifold*: the synthetic sentences are plausible, so the teacher's behavior on them is meaningful, whereas random token swaps would push inputs off-distribution where the teacher's outputs are themselves unreliable. This is a general principle of distillation augmentation — augment toward inputs the model will actually see, not toward arbitrary noise.

#### Worked example: TinyBERT-4 versus a from-scratch 4-layer BERT

Suppose you need a sub-15M-parameter encoder for an on-device intent classifier on a mid-range Android phone, target p50 under 10 ms at batch 1. Your options:

- **Train a 4-layer, 312-hidden BERT from scratch on your task.** With a few thousand labeled examples it will badly underfit; expect to land maybe 6–10 points below BERT-base. The model is small and fast, but the accuracy is unusable.
- **Use TinyBERT-4.** General distillation already gave it a teacher-aligned interior; task distillation on your data, augmented ~20×, specializes it. Reported retention is about 96 percent of BERT-base on GLUE — a few points, not ten. Same 14.5M parameters, same ~9× speedup, but a model you can actually ship.

The parameter count and latency are *identical*. The only difference is the supervision, and that difference is the entire ballgame. This is the recurring lesson: at the edge, distillation is rarely about making the model smaller — you could always do that — it is about making a small model *good*.

## 4. MobileBERT: redesign the block, then transfer into it

DistilBERT and TinyBERT keep BERT's block shape and shrink depth or width. MobileBERT (Sun et al., 2020) takes a different swing: **redesign the transformer block itself** for efficiency, then use a specially-built wide teacher to transfer knowledge into the new shape. The result is a *task-agnostic* model — distilled once during pre-training, then fine-tuned normally, like the original BERT — that is **4.3× smaller and 5.5× faster** than BERT-base while retaining roughly 99 percent of its GLUE score, and runs at about 62 ms on a Pixel 4 phone. Figure 4 contrasts a standard BERT block with MobileBERT's bottleneck block.

![A two-column before-and-after diagram contrasting a wide BERT block running attention and a large feed-forward network on full 768-dimensional hidden states against a MobileBERT inverted-bottleneck block that down-projects to 128 dimensions, runs attention and stacked feed-forward on the thin tensor, then up-projects, reaching 25 million parameters](/imgs/blogs/distillation-case-studies-distilbert-to-cnns-5.png)

### 4.1 The bottleneck block

The compute in a transformer block is dominated by the width of the hidden state flowing through attention and the feed-forward network. MobileBERT's block keeps the *outer* width large (so it stays compatible with a wide teacher) but immediately **down-projects** the hidden state to a thin bottleneck — 128 dimensions — runs attention and the feed-forward stack on that thin tensor, then **up-projects** back. Because the expensive operations now act on a 128-wide tensor instead of a 512- or 768-wide one, each block is dramatically cheaper.

To keep the parameter ratio between attention and feed-forward balanced after shrinking the width, MobileBERT uses **stacked feed-forward networks** — several smaller FFNs in series within a block — rather than one wide one. It also replaces layer normalization and GELU with cheaper operations (a no-op "NoNorm" and ReLU) specifically to be friendly to mobile inference, where those ops are surprisingly costly. This is architecture co-designed with the deployment target, not just a generic shrink.

The arithmetic of why the bottleneck pays is worth seeing. A standard self-attention plus feed-forward block on hidden width $d$ costs on the order of $d^2$ per token for the projections (the $Q$, $K$, $V$, and output matrices are each $d \times d$) and $2 d \cdot d_{ff}$ for the feed-forward, where $d_{ff}$ is the intermediate width (typically $4d$). At $d = 768$ that is dominated by terms in $768^2 \approx 590\text{k}$ and $768 \times 3072 \approx 2.4\text{M}$. MobileBERT keeps the *inter-block* width at 512 but runs the expensive interior on a bottleneck of $d_b = 128$, so the attention and FFN cost scales with $d_b^2 \approx 16\text{k}$ and $d_b \times d_{ff,b}$ instead — roughly a $(512/128)^2 = 16\times$ reduction in the quadratic terms, before accounting for the down/up projections you add back. The net is a model with the *representational reach* of a wide network at each block boundary (so the wide IB-BERT teacher can supervise it) but the *compute* of a thin one inside. That is the whole trick: pay for width only at the seams, not in the interior where the matmuls live. The reason GELU and LayerNorm get swapped out is that on a mobile CPU or DSP, these elementwise/reduction ops are not fused into the matmul and become a surprisingly large fraction of latency once the matmuls are cheap — a classic case of Amdahl's law biting the part you did not optimize.

### 4.2 The IB-BERT teacher and progressive transfer

You cannot distill a wide teacher into a 128-bottleneck student directly — the shapes do not line up. MobileBERT's solution is to build a special teacher, **IB-BERT** (inverted-bottleneck BERT): a model that is as wide as BERT-large at its core but carries an inverted-bottleneck structure so that its *layer outputs have the same dimensionality as the MobileBERT student's*. Now teacher and student match shape-for-shape at every layer, and feature transfer is well-posed.

The transfer itself is **progressive**. Rather than training all student layers against all teacher layers at once, MobileBERT transfers knowledge layer by layer, bottom-up: train the student's layer 1 to match the teacher's layer 1 (freezing or down-weighting the rest), then layer 2, and so on, before a joint fine-tune. The objectives at each layer are **feature-map transfer** (MSE on hidden states) and **attention transfer** (KL on attention distributions), plus the usual pre-training losses. Progressive transfer stabilizes the optimization — each layer is supervised against a teacher layer that has *already* been matched below it, so errors do not compound up the stack.

### 4.3 Why task-agnostic matters for deployment

The deployment-relevant property is that MobileBERT, like the original BERT and like DistilBERT, is distilled **once, task-agnostically, during pre-training**. You download it, fine-tune on your task with the standard recipe, and ship. TinyBERT's strongest results require its task-specific second stage with augmentation *per task*, which is more work to operationalize across many downstream tasks. For an organization shipping a dozen NLP features on-device, "distill once, fine-tune many" is a real operational advantage. MobileBERT trades a heavier *pre-training* recipe (build IB-BERT, do progressive transfer) for a *lighter* per-task recipe.

## 5. CNN distillation: the original recipe, still the workhorse

Distillation started in vision, and for edge deployment it remains the bread-and-butter way to get a small convolutional network up to the accuracy of a big one. The setup: a heavy teacher — a deep ResNet, or an *ensemble* of models, or a vision transformer — and a small student like MobileNet, EfficientNet-Lite, or a slim ResNet. The recipe is response + feature distillation, the same two ingredients we have seen, adapted to convolutions.

### 5.1 The response (logit) term

Identical to Hinton 2015: softened KL between teacher and student logits over the class set, plus the ordinary cross-entropy against the true label. For a 1000-class ImageNet classifier this soft term is rich — the teacher's ranking of the 999 wrong classes encodes the full visual similarity structure (Persian cat vs Siamese cat vs tabby, all close; cat vs airliner, far). The combined objective is the familiar:

$$\mathcal{L} = \alpha \, T^2 \,\mathrm{KL}\!\left(p^{T}_{\tau}\,\Vert\,p^{S}_{\tau}\right) + (1-\alpha)\,\mathrm{CE}\!\left(y, p^{S}\right)$$

with $\alpha$ typically 0.5–0.9 and $T$ in the 3–10 range for ImageNet-scale problems.

### 5.2 The feature term

The feature term matches intermediate *feature maps*. The seminal version is **FitNets** (Romero et al., 2015): pick a teacher "hint" layer and a student "guided" layer, add a learned convolution or linear "regressor" to match the channel/spatial shapes, and minimize MSE between the (regressed) student feature map and the teacher's. Later variants match *attention maps* derived from feature activations (Zagoruyko and Komodakis), or pairwise *relations* between examples rather than the activations themselves. The common thread: give the convolutional student a target inside the network, not just at the logits, because — same as transformers — a small CNN's interior is where it most needs guidance.

A third family, **relation-based distillation**, matches not the activations themselves but the *relationships between examples* — for instance, the pairwise distances or angles between a batch of feature vectors (Park et al.'s Relational Knowledge Distillation, 2019). The idea is that even if the student cannot reproduce the teacher's exact features, it can reproduce the *structure* of its feature space: which examples the teacher considers similar. This is the most permissive form of supervision and is especially useful when teacher and student architectures are very different (a vision transformer teacher into a CNN student, say), because it does not require any spatial or channel correspondence — only that the relative geometry of a batch is preserved.

A modern, robust recipe also leans on **strong augmentation and long schedules**. The "function matching" / consistent-teaching line of work (Beyer et al., 2022, *Knowledge distillation: A good teacher is patient and consistent*) showed that if you apply the *same* augmentation to the image fed to teacher and student, and train for a very long schedule (hundreds to thousands of epochs), a plain logit distillation can push a ResNet-50 student to match much larger teachers on ImageNet — no fancy feature loss required. Patience is itself a technique. The mechanism: distillation is a *function-matching* problem, fitting the student to the teacher's input-output map everywhere, not just at the training points. Consistent augmentation (same crop, same mixup coefficient for teacher and student) ensures the two see *the same point* in the augmented input space, so the soft target is a valid label for the student's input; inconsistent augmentation feeds the student image A but distills toward the teacher's opinion of image B, which is noise. The long schedule then samples enough of the augmented input manifold that the student's function matches the teacher's across the whole region the deployment will see, not just the original training images.

### 5.3 The practical recipe

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# teacher: a frozen ResNet-152; student: a MobileNetV3-Small.
teacher.eval()
for p in teacher.parameters():
    p.requires_grad = False

T = 4.0          # distillation temperature
alpha = 0.7      # weight on the soft (KD) term
beta = 100.0     # weight on the feature (hint) term

# a 1x1 conv to align student channels to the teacher hint shape.
regressor = nn.Conv2d(student_feat_ch, teacher_feat_ch, kernel_size=1).cuda()

def distill_step(images, labels, optimizer):
    with torch.no_grad():
        t_logits, t_feat = teacher(images, return_feat=True)
    s_logits, s_feat = student(images, return_feat=True)

    # 1) response / logit distillation (Hinton).
    kd = F.kl_div(
        F.log_softmax(s_logits / T, dim=1),
        F.softmax(t_logits / T, dim=1),
        reduction="batchmean",
    ) * (T * T)

    # 2) hard-label cross-entropy keeps the student honest.
    ce = F.cross_entropy(s_logits, labels)

    # 3) FitNets-style feature hint (MSE on aligned feature maps).
    feat = F.mse_loss(regressor(s_feat), t_feat)

    loss = alpha * kd + (1 - alpha) * ce + beta * feat
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()
```

#### Worked example: ResNet-152 into MobileNetV3 for a phone camera

Say you have a ResNet-152 fine-tuned to 80.0 percent top-1 on a 200-class product-recognition task, but it is 60M parameters and ~11 GFLOPs — far too heavy for a phone's NPU at the frame rate you need. Target: MobileNetV3-Small, 2.5M parameters, ~60 MFLOPs, real-time on the device.

- **From scratch** the MobileNetV3 fine-tunes to, say, 73.5 percent top-1 — a 6.5-point gap from the teacher.
- **With distillation** (the recipe above: $\alpha = 0.7$, $T = 4$, a FitNets hint on the last spatial block, long schedule with the teacher's augmentation), the same MobileNetV3 reaches ~78.0 percent — closing most of the gap, about a 4.5-point gain over from-scratch, for *zero* change in the deployed model.

The student is identical in size, FLOPs, and latency either way — call it ~8 ms per frame on the phone's NPU at int8 versus ~70 ms for the ResNet on the same chip. Distillation bought ~4.5 points of accuracy at no inference cost. That is the entire pitch of the lever, and it is why every production vision pipeline that ships a tiny model distills it from a big one.

## 6. The Hugging Face distillation sketch: reproduce DistilBERT's loss

Here is a compact, runnable skeleton of a BERT-style distillation training loop with the multi-term loss — the thing the worked examples above are computing. It is deliberately close to what `transformers`' own `Trainer` does, so you can drop it into a real project. The full official implementation lives in the Hugging Face `transformers` repo under `examples/research_projects/distillation`; this is the core of it.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForMaskedLM, AutoTokenizer

teacher = AutoModelForMaskedLM.from_pretrained("bert-base-uncased").eval().cuda()
for p in teacher.parameters():
    p.requires_grad = False

# build a 6-layer student config from the teacher, then init from teacher weights.
student = AutoModelForMaskedLM.from_pretrained("distilbert-base-uncased").cuda()

# DistilBERT loss weights and temperature (from Sanh et al. 2019).
TEMP = 2.0
W_CE, W_MLM, W_COS = 5.0, 2.0, 1.0
kl = nn.KLDivLoss(reduction="batchmean")
cos = nn.CosineEmbeddingLoss(reduction="mean")

def distill_loss(input_ids, attention_mask, mlm_labels):
    with torch.no_grad():
        t_out = teacher(input_ids, attention_mask=attention_mask, output_hidden_states=True)
    s_out = student(input_ids, attention_mask=attention_mask,
                    labels=mlm_labels, output_hidden_states=True)

    # 1) soft-target KL on softened logits, scaled by T^2.
    #    only count masked positions to match the MLM target.
    mask = (mlm_labels != -100)
    s_logits = s_out.logits[mask]
    t_logits = t_out.logits[mask]
    loss_ce = kl(
        F.log_softmax(s_logits / TEMP, dim=-1),
        F.softmax(t_logits / TEMP, dim=-1),
    ) * (TEMP ** 2)

    # 2) hard masked-LM loss (the model already computed it from labels).
    loss_mlm = s_out.loss

    # 3) cosine embedding loss on the last hidden state, direction only.
    s_hidden = s_out.hidden_states[-1]            # (B, L, H)
    t_hidden = t_out.hidden_states[-1]            # same H = 768
    am = attention_mask.unsqueeze(-1).bool()
    s_vec = s_hidden.masked_select(am).view(-1, s_hidden.size(-1))
    t_vec = t_hidden.masked_select(am).view(-1, t_hidden.size(-1))
    target = torch.ones(s_vec.size(0), device=s_vec.device)  # maximize cos sim
    loss_cos = cos(s_vec, t_vec, target)

    return W_CE * loss_ce + W_MLM * loss_mlm + W_COS * loss_cos
```

A few things worth pointing out in this sketch, because they are exactly where reimplementations go wrong:

- **Mask the logits to the masked positions.** The MLM target only exists where you masked tokens (`mlm_labels != -100`), so the soft loss should be computed there too, matching what BERT predicts.
- **The cosine target is all ones.** `CosineEmbeddingLoss` with target $+1$ *maximizes* cosine similarity (minimizes $1 - \cos$), which is the direction-alignment we want. Target $-1$ would push them apart — a classic sign bug.
- **The teacher runs under `torch.no_grad()` and `eval()`.** Forgetting `eval()` leaves teacher dropout on, so your "teacher distribution" is noisy and changes every step.
- **You can extend this to attention matching** (the TinyBERT recipe) by adding `output_attentions=True` and an MSE term between teacher and student attention scores per mapped layer — that single change is the conceptual jump from DistilBERT to TinyBERT.

Here is that extension, the TinyBERT-style layer-matching loss, so the jump from DistilBERT is concrete rather than hand-waved. The key pieces are the layer map and the projection that bridges a narrower student to the wide teacher:

```python
import torch
import torch.nn.functional as F

# student layer m learns from teacher layer g(m). 4-layer student, 12-layer teacher.
LAYER_MAP = {1: 3, 2: 6, 3: 9, 4: 12}

# learned projection lifting student hidden (312) into teacher space (768).
proj = torch.nn.Linear(312, 768, bias=False).cuda()

def tinybert_layer_loss(s_out, t_out):
    """s_out / t_out are model outputs with hidden_states and attentions."""
    loss = 0.0
    for s_layer, t_layer in LAYER_MAP.items():
        # attention-matrix MSE on the pre-softmax-like attention tensors, per head.
        s_attn = s_out.attentions[s_layer - 1]   # (B, heads, L, L)
        t_attn = t_out.attentions[t_layer - 1]
        loss = loss + F.mse_loss(s_attn, t_attn)

        # hidden-state MSE, student projected up into the teacher's width.
        s_hid = proj(s_out.hidden_states[s_layer])   # (B, L, 768)
        t_hid = t_out.hidden_states[t_layer]
        loss = loss + F.mse_loss(s_hid, t_hid)
    return loss
```

This is the entire structural difference between the two recipes: DistilBERT supervises one geometry term at the top, TinyBERT supervises attention and hidden states at four mapped depths. The `proj` matrix is learned during training and discarded at inference, so the deployed student is exactly its 14.5M parameters with zero projection overhead.

To run distillation at pre-training scale you would wrap this in a standard training loop over a large corpus with a masked-LM data collator (`DataCollatorForLanguageModeling`), gradient accumulation, and a linear-warmup schedule — but the *loss* above is the entire scientific content of DistilBERT.

## 7. The science: why these choices are not arbitrary

We have stated the recipes. Now the *why* — because the difference between a distillation that works and one that quietly underperforms is almost always one of these four mechanisms.

### 7.1 Why initialization from the teacher matters

Distillation is a *non-convex* optimization where the loss surface is shaped by the teacher's representations. A randomly-initialized student lands in a random basin; the feature-matching losses then drag it toward the teacher's basin, which may be far away and across barriers. Initializing the student's layers *from* the teacher's weights places it already inside the teacher's basin — the feature losses start near zero and the optimization is local refinement.

There is a cleaner way to see it. The cosine and hidden-state losses are minimized when student and teacher representations coincide. If the student begins at the teacher's weights, those losses begin at their global minimum for those layers and the only pressure is from the *capacity reduction* (fewer layers must now do the same job). The optimizer spends its budget compensating for lost capacity, not first finding the teacher's manifold and then compensating. Empirically this is worth a meaningful fraction of the final score and is why DistilBERT, MobileBERT, and the Minitron-style prune-then-distill recipes all init the student from the teacher. (For the LLM scaling version of "prune the teacher into the student then distill," see the [distillation in LLM](/blog/machine-learning/large-language-model/distillation-in-llm) deep-dive.)

Why *every other* layer rather than the first six or the last six? Taking layers 1, 3, 5, 7, 9, 11 samples the teacher's full depth, so the student inherits a coarse version of the entire computation — early-layer syntax, mid-layer semantics, late-layer task structure. Copying only the *bottom* six would give the student a strong feature extractor but no high-level reasoning; copying only the *top* six would give the reverse and, worse, the top layers expect inputs the student's (now absent) lower layers used to provide, so they would be fed representations they were never trained on. The alternating choice is a principled compromise that keeps the student's six layers spread across the function the twelve-layer teacher computed. This is the same logic as TinyBERT's uniform layer map, applied to *initialization* rather than *supervision* — and the two reinforce each other: init from teacher layers $\{1,3,5,7,9,11\}$ and then supervise each student layer against the teacher layer it was copied from, and the student barely has to move.

### 7.2 Why logit-only distillation undershoots for transformers

A transformer's output logits are a thin summary of a very deep computation. Matching only the logits constrains a low-dimensional projection of the student's behavior and leaves the entire interior — twelve layers of attention patterns and hidden states — unsupervised. The student is free to reach the same outputs through wildly different internal computations, and for a *smaller* model those alternative computations are usually worse-generalizing. Figure 5 contrasts the two regimes.

![A two-column before-and-after diagram contrasting logit-only distillation, where inner layers are unsupervised and the student drifts two to four points short, against feature and attention matching, where attention matrices and hidden states are matched per layer to close the gap](/imgs/blogs/distillation-case-studies-distilbert-to-cnns-6.png)

Here is the information-theoretic intuition. The teacher's final distribution over $V$ classes carries at most $\log_2 V$ bits per token of supervision. The teacher's *attention matrices* and *hidden states* across $L$ layers carry orders of magnitude more — they are dense, continuous, per-layer signals. Matching them gives the student a target at every depth, so its interior is pinned to a known-good computation rather than discovered freely. This is exactly why TinyBERT (attention + hidden + embedding + logits) compresses 7.5× while DistilBERT (logits + one geometry term) compresses ~1.7×: TinyBERT supervises far more of the student, so a far smaller student can still be pinned to good behavior. The depth of supervision *buys* compression.

A caveat that keeps you honest: more matching is not free. Attention/hidden matching needs a layer mapping and (for narrower students) projection matrices, it is sensitive to the choice of mapping, and it can over-constrain if the student is *too* small to ever reproduce the teacher's representations. The art is matching enough to pin the interior without demanding the impossible.

### 7.3 Why two stages and why augmentation

The two-stage idea is a bias–variance argument. General distillation on a large corpus is *low variance* — abundant data, so the student reliably learns a transferable representation. Task data is *high variance* — small, so distilling only on it would let the student overfit. Doing general distillation first fixes the representation cheaply; task distillation then makes a small, low-risk adjustment. Augmentation attacks the same problem from the data side: it inflates the effective task-set size so the soft-target signal is sampled densely, lowering the variance of the task stage. Both are mechanisms for *getting more bits of teacher supervision into a small student* — the recurring theme.

### 7.4 Tying back to response vs feature distillation

Everything in this section is the response-vs-feature distinction from D2 made concrete. DistilBERT is mostly response (logits) plus a light feature term (cosine). TinyBERT and MobileBERT are heavily feature-based (attention, hidden states, feature maps) plus a response term. CNN distillation spans both — logit KD is response, FitNets/attention-transfer are feature. The single axis that organizes all four case studies is *how deep into the network the supervision reaches*, and the empirical law is: **deeper supervision permits a smaller student at the same accuracy.** That is the one sentence to remember.

## 8. The numbers, side by side

Figure 6 places the four recipes on the same grid of properties so you can see the trade at a glance; Figure 7 does the same for the size/speed/accuracy outcomes.

![A four-by-four matrix comparing DistilBERT, TinyBERT, MobileBERT, and CNN distillation across what is matched, number of stages, compression achieved, and recipe cost](/imgs/blogs/distillation-case-studies-distilbert-to-cnns-2.png)

The detailed comparison, with every headline figure marked as reported in its source paper (round numbers; exact values vary by task and configuration):

| Model | Teacher | Student size | Compression | Speedup | Accuracy retained | What it matches | Stages |
|---|---|---|---|---|---|---|---|
| **DistilBERT** | BERT-base 110M | 66M (6 layers) | ~1.7× (40% smaller) | ~1.6× (60% faster) | ~97% of GLUE (reported) | soft logits + MLM + cosine hidden | 1 (pre-training) |
| **TinyBERT-4** | BERT-base 110M | 14.5M (4 layers, 312 hid) | ~7.5× | ~9.4× | ~96% of BERT-base GLUE (reported) | embed + attention + hidden + logits | 2 (general + task) + aug |
| **MobileBERT** | IB-BERT (BERT-large-wide) | 25M | ~4.3× | ~5.5× (≈62 ms Pixel 4) | ~99% of GLUE (reported) | feature map + attention, progressive | 1 (pre-training, progressive) |
| **CNN KD (e.g. ResNet→MobileNet)** | ResNet-152 / ensemble | ~2.5–4M | ~10×+ params | ~8× | ~98% of teacher top-1 (recipe-dependent) | soft logits + feature hints | 1 (task) |

![A four-by-four matrix showing parameters versus teacher, speedup, accuracy kept, and inference latency for DistilBERT, TinyBERT-4, MobileBERT, and a MobileNet CNN distillation, all retaining roughly 96 to 99 percent of teacher quality](/imgs/blogs/distillation-case-studies-distilbert-to-cnns-7.png)

Read the table as a single curve. As you move down the *depth of supervision* axis — from DistilBERT's logits-plus-cosine to TinyBERT's match-everything-twice — compression climbs from 1.7× to 7.5× while accuracy retention stays in a tight 96–99 percent band. That near-flatness of the accuracy column is the whole reason distillation is the lever you reach for first: it buys size and speed at a cost measured in *fractions of a point*, not the multiple points you pay for an equivalent from-scratch shrink. The recipe cost rises with compression, but the accuracy cost barely moves.

A measurement honesty note, because these numbers are easy to misuse. "Speedup" here is the paper's reported figure, usually on the authors' hardware at a particular sequence length and batch. On *your* target the real number depends on whether you are compute-bound or memory-bound (see [the roofline model](/blog/machine-learning/edge-ai/the-roofline-model-where-your-bottleneck-lives)), whether your runtime fuses the smaller ops, and whether batch=1 latency or throughput is what you care about. Always re-measure on the device, warmed up, at the batch and sequence length you actually serve, reporting p50 and p99 — not the paper's number.

How to measure it honestly, concretely, because this is where most "it was supposed to be 2× faster" disappointments come from:

- **Warm up first.** The first few inferences pay for kernel JIT, memory allocation, and (on mobile) bringing the NPU out of a low-power state. Discard the first 20–50 iterations, then time the next few hundred. The cold-start number is real for a once-per-app-launch model but irrelevant for a server.
- **Pin the batch and sequence length to production.** A distilled encoder timed at batch 32, sequence 512 looks great and tells you nothing about your batch=1, sequence-32 chat classifier. Latency at batch 1 is frequently *memory-bound* — dominated by loading weights from DRAM, not by FLOPs — and a smaller model helps there mostly by being smaller (fewer bytes to load), which is exactly why distillation *and* quantization both help a memory-bound op.
- **Report a distribution, not a mean.** p50 tells you the typical case; p99 tells you what your tail users feel and is what your SLA is written against. Thermal throttling on a phone or a Jetson can make the p99 several times the p50 once the device heats up, so time long enough to reach steady-state temperature.
- **Measure accuracy on *your* eval set, not GLUE.** "97 percent of GLUE retained" is a property of the published student on the published benchmark. On your domain-shifted task the retention could be higher or lower. The only number that matters for your ship/no-ship decision is the delta on your held-out set after fine-tuning.
- **Account for the whole pipeline.** Tokenization, padding, host-to-device copies, and post-processing can dwarf the model on a tiny student. A 5 ms model behind 15 ms of Python tokenization is a 20 ms service; distilling the model again will not move that 20 ms. Profile the request end to end before you optimize the part that is already small.

A minimal but honest latency harness for a distilled encoder looks like this — warm up, then collect a distribution and report percentiles, not a mean:

```python
import time, numpy as np, torch

model.eval().to("cpu")          # measure on the target you will deploy to
inputs = tokenizer("a representative production sentence", return_tensors="pt")

# 1) warm up: discard kernel JIT / allocator / power-state cost.
with torch.no_grad():
    for _ in range(30):
        model(**inputs)

# 2) time batch=1, the real on-device regime.
lat = []
with torch.no_grad():
    for _ in range(500):
        t0 = time.perf_counter()
        model(**inputs)
        lat.append((time.perf_counter() - t0) * 1000.0)   # ms

lat = np.array(lat)
print(f"p50 {np.percentile(lat,50):.1f} ms | "
      f"p99 {np.percentile(lat,99):.1f} ms | n={len(lat)}")
```

Run that on the fp32 teacher, the distilled student, and the distilled-plus-int8 student, on the *same* machine, and you get the only speedup numbers that matter for your deployment — the ones measured on your hardware at your batch size, with the tail visible.

## 9. Composing with quantization: distill, then int8

Distillation and quantization are *orthogonal* levers, and they stack almost multiplicatively, which is why the strongest edge results combine them. DistilBERT cuts the layer count 2×; int8 dynamic or static quantization then cuts the per-weight bytes ~4× and speeds up the matmuls on hardware with int8 kernels. The composite — DistilBERT + int8 — is the standard high-throughput CPU NLP deployment.

The ordering that works: **distill first, quantize second.** Distillation is a training-time procedure that needs full-precision gradients; quantization is a post-training (or QAT) step applied to the finished student. Quantizing first and then distilling fights itself — you would be distilling into a model whose precision keeps changing. The clean pipeline is: distill the student to convergence, then apply post-training quantization (or a short QAT pass if PTQ drops too much). For the int8 mechanics, calibration, and the PTQ-vs-QAT decision, see [post-training quantization](/blog/machine-learning/edge-ai/post-training-quantization-ptq) and [quantization-aware training](/blog/machine-learning/edge-ai/quantization-aware-training-qat).

Distillation also composes beautifully with **pruning**, and the deepest connection is that distillation is the standard *recovery* step after you prune. Pruning removes weights or whole structures and always costs accuracy; the cure is to fine-tune the pruned model — and the best fine-tune is a *distillation* fine-tune with the original dense model as teacher. This is exactly the Minitron / prune-then-distill recipe: prune the teacher down to the student's size, then distill the dense teacher back into the pruned student to recover the lost accuracy. The student is initialized *from the teacher by construction* (it is the pruned teacher), which is why this recipe is so data-efficient. The general rule across all the levers: distillation is the one that *adds accuracy back*, so it belongs *after* the lever that took accuracy away. Quantize a distilled model; recover a pruned model by distilling into it. The order is not arbitrary — it follows from which lever is a giver and which is a taker.

One trap when stacking distillation and QAT specifically: if you run quantization-aware training on the distilled student, keep the *full-precision teacher* in the loop as the distillation target during QAT. The fake-quantization noise the student now carries makes the soft-target supervision *more* valuable, not less — the teacher's smooth distribution helps the student find weights that are robust to the rounding, often recovering most of the int8 gap. Distilling during QAT (sometimes called "quantization-aware distillation") is a known way to push int8 students to within a fraction of a point of their fp32 selves.

```python
import torch
from transformers import AutoModelForSequenceClassification

# start from a fine-tuned DistilBERT student.
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased-finetuned-sst-2-english"
).eval()

# dynamic int8 quantization of the Linear layers — the cheapest composition.
quantized = torch.ao.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

# result: ~2x smaller again on top of DistilBERT's 1.7x, and faster int8
# matmuls on CPUs with the right kernels (oneDNN / fbgemm).
torch.save(quantized.state_dict(), "distilbert_sst2_int8.pt")
```

#### Worked example: DistilBERT + int8 on a CPU server

Compose the levers and account for them. Start from BERT-base, fp32, on a CPU inference instance: call it ~110M params, ~440 MB, and a p99 around 200 ms at batch 1 for a 128-token classifier (illustrative, hardware-dependent).

- **Distill to DistilBERT** (6 layers): ~66M params, ~265 MB fp32, p99 roughly halved by the 2× layer cut — into the ~100 ms range, retaining ~97 percent of accuracy.
- **Quantize to int8** (dynamic, Linear layers): weights drop ~4× — to roughly ~70 MB — and the int8 matmuls cut latency further on a CPU with fbgemm/oneDNN, often another ~1.5–2× — into the ~50–65 ms range.

End state: a model ~6× smaller and ~3–4× faster than the fp32 teacher, having paid maybe a point of accuracy total (most from distillation, a few tenths from int8 if calibrated well). That is the composition this whole series is built around — two levers, multiplied, landing you on a Pareto point neither could reach alone. Re-measure on the real instance: int8 only speeds things up if your CPU has the kernels, and a tiny model can be *memory-bound*, in which case the quantization helps size more than latency.

## 10. When to reach for which recipe

A decisive recommendation, drawn as a decision tree in Figure 8 and spelled out below.

![A decision tree branching first on transformer versus CNN, then for transformers on modest versus aggressive compression budget, routing to DistilBERT for modest low-effort needs, TinyBERT or MobileBERT for aggressive compression, and a ResNet-to-MobileNet feature-plus-logit recipe for vision](/imgs/blogs/distillation-case-studies-distilbert-to-cnns-8.png)

- **You need a smaller transformer with the least effort, ~40 percent off, ~97 percent quality, and a model you fine-tune like normal BERT.** Reach for **DistilBERT** (or just download it). One stage, simple loss, task-agnostic. This is the default and you should justify *not* using it.
- **You need aggressive compression (4–10×) for a transformer and can afford a heavier recipe.** Reach for **TinyBERT** if you can run its two-stage, augmented, task-specific pipeline and want maximum compression per accuracy point. Reach for **MobileBERT** if you would rather pay the cost *once* at pre-training and then fine-tune many tasks cheaply, and if you want an architecture co-designed for mobile inference. The split is operational: TinyBERT optimizes per-task quality at the cost of per-task work; MobileBERT optimizes deploy-many-tasks at the cost of a complex pre-training.
- **You need a small CNN for vision on-device.** Use **logit + feature distillation** from a strong ResNet/ensemble teacher into a MobileNet/EfficientNet-Lite student, with strong consistent augmentation and a long schedule. Add a FitNets-style hint if logit-only leaves a gap. This is the workhorse and it composes directly with int8 for the NPU.
- **You already hit your target with a from-scratch small model.** Then *do not distill.* Distillation costs you a trained teacher, a longer training loop, and recipe tuning. If a small model trained normally already clears your accuracy and latency bars, the lever is not worth pulling. The point of distillation is to *recover accuracy a small model otherwise loses* — if it is not losing any, there is nothing to recover.

When *not* to use a given recipe is just as important. Do not use TinyBERT's full pipeline if you have one task and a tight engineering deadline — DistilBERT plus a fine-tune is a quarter of the work for most of the win. Do not match attention/hidden states if your student is so small it physically cannot reproduce them — you will over-constrain and stall. Do not distill into a CNN and then forget to quantize for the NPU — half your edge win is on the table.

## 11. Stress-testing the recipes

Pose the hard cases and reason through them, because production is where the clean recipe meets the messy device.

**What if the teacher and student vocabularies differ?** DistilBERT shares the teacher's tokenizer and embeddings, which is why direct hidden-state comparison works. If you change the tokenizer, the embedding and hidden-state losses become ill-defined (different token-to-position alignment), and you are back to logit-level or sequence-level distillation only. Keep the tokenizer if you want feature matching.

**What if the student is much narrower than the teacher?** That is the TinyBERT case — you need projection matrices ($W_e$, $W_h$) to lift the student's features into the teacher's space before MSE. Those projections are learned and thrown away at inference, so they cost nothing at deploy time but are essential during training. Skip them and the shapes do not even multiply.

**What happens with a tiny task dataset?** This is exactly what TinyBERT's augmentation and two-stage design address. With a few thousand examples, a one-stage task distillation overfits; you want the general-distillation interior plus heavy augmentation to densely sample the teacher. If you cannot augment, lean harder on the general (pre-training) distillation and do a *light* task fine-tune rather than full task distillation.

**What if attention matching makes training unstable?** Match the pre-softmax attention *scores* (TinyBERT's choice) rather than post-softmax weights — it converges faster and is less prone to the vanishing gradients you get when matching near-saturated softmax outputs. And ramp feature-loss weights with a warmup so the logit loss does not get swamped early.

**What if your student, post-distillation, is memory-bound on the device?** Then the further speedup from quantization comes mostly from *smaller weights* (less DRAM traffic), not faster math — and that is fine; on a memory-bound op, halving the bytes moved roughly halves the time. Profile to know which regime you are in (the [roofline](/blog/machine-learning/edge-ai/the-roofline-model-where-your-bottleneck-lives) post is the tool); distillation reduced your op *count*, quantization reduces your *bytes per op*, and which one helps more depends on where your bottleneck lives.

**What if the teacher is wrong on a class?** The hard-label term ($\mathcal{L}_{mlm}$ / cross-entropy) exists precisely to keep the student anchored to ground truth so it does not faithfully reproduce the teacher's systematic errors. Do not set its weight to zero; the soft term alone will happily distill the teacher's mistakes.

**What if the student outperforms a weak teacher?** It happens — a well-regularized small student trained with strong augmentation can occasionally beat a mediocre teacher on the eval set, which makes the soft-target loss actively harmful (you are pulling the student *toward* worse predictions). The signal that this is happening: the distillation loss keeps dropping while the validation accuracy plateaus or regresses. The fix is to use a *better* teacher (a stronger model or an ensemble), down-weight the soft term, or — the cleanest option — only distill from teacher predictions where the teacher is *confident and correct* on held-out data. The premise of distillation is that the teacher is genuinely better; verify that premise before you spend a week on the recipe.

**What if you need the student to be a *different shape* than any teacher block?** This is the MobileBERT lesson generalized: if your student architecture (a bottleneck, a depthwise-separable CNN, a linear-attention model) has no natural per-layer correspondence to a standard teacher, you have two roads. Build a *custom teacher* whose layer outputs match the student's dimensions (MobileBERT's IB-BERT), or fall back to *relation-based* or *logit-only* distillation, which need no shape correspondence. The trade is the usual one: custom-teacher feature matching gives the most compression-per-point but is a lot of engineering; logit-only is trivial but undershoots. Pick based on how aggressive your compression target is.

**What if you are distilling a generative LLM, not a classifier?** The classification recipes here transfer in spirit but not in detail — for autoregressive models the "soft target" is a per-token distribution over a huge vocabulary, and sequence-level effects (exposure bias, the gap between teacher-forced and free-running generation) dominate. That is a deep topic with its own recipes (sequence-level KD, on-policy distillation, distilling chain-of-thought). The [distilling LLMs and reasoning](/blog/machine-learning/edge-ai/distilling-llms-and-reasoning) post in this series covers it; do not naively port the encoder recipe onto a decoder and expect TinyBERT's numbers.

## 12. Case studies recap with real numbers

To consolidate the reported figures (all marked as paper-reported; treat as round, configuration-dependent):

- **DistilBERT (Sanh et al., 2019).** 6-layer student from 12-layer BERT-base, triple loss (soft logits weight 5.0, MLM weight 2.0, cosine weight 1.0), initialized from alternating teacher layers, distilled during pre-training. **40 percent smaller, 60 percent faster, ~97 percent of BERT-base's GLUE.** Became the default small encoder; over a hundred million downloads on the Hub.
- **TinyBERT (Jiao et al., 2020).** Two-stage (general + task) distillation matching embeddings, attention matrices (pre-softmax), and hidden states (with projection), plus prediction loss, with ~20× data augmentation in the task stage. **TinyBERT-4: ~7.5× smaller, ~9.4× faster, ~96 percent of BERT-base on GLUE.** The most compression-per-point of the four.
- **MobileBERT (Sun et al., 2020).** Bottleneck (128-dim) blocks with stacked FFNs and mobile-friendly ops, knowledge progressively transferred from a wide inverted-bottleneck IB-BERT teacher, task-agnostic. **25M params, ~4.3× smaller, ~5.5× faster, ~99 percent of GLUE, ≈62 ms on a Pixel 4.** Distill-once-fine-tune-many.
- **CNN distillation (Hinton 2015; FitNets, Romero 2015; Beyer 2022).** ResNet/ensemble teacher into MobileNet/EfficientNet-Lite student via logit KD plus feature hints, with consistent augmentation and long schedules. **~10×+ fewer params, ~8× faster, frequently within ~1–2 points of the teacher's top-1.** The original recipe, still the vision workhorse.

The thread through all four: the student's parameter count and inference cost are set by its architecture; its *accuracy* is set by how much teacher supervision you pour in. Same model, more bits of teacher, more accuracy. That is the lever.

## 13. Key takeaways

- **Distillation is the lever that adds accuracy back.** A distilled student routinely beats an identically-sized from-scratch student by 2–4 points — same cost, richer supervision. Reach for it whenever a small model is losing accuracy you need.
- **Soft targets carry dark knowledge.** The teacher's ranking of the *wrong* answers encodes its learned similarity structure; the $T^2$-scaled KL is how you transfer it. Never forget the $T^2$.
- **Depth of supervision buys compression.** Logit-only (DistilBERT, ~1.7×) → add cosine geometry → add hidden states and attention (TinyBERT, ~7.5×). More internal matching pins a smaller student to good behavior. This is the single axis organizing all four case studies.
- **Initialize the student from the teacher.** It starts the optimization inside the teacher's basin, makes feature losses begin near zero, and is nearly free. Skipping it is the most common reimplementation mistake.
- **Keep the hard-label term.** Soft-only distillation faithfully reproduces the teacher's systematic errors; the ground-truth loss keeps the student honest.
- **Two stages and augmentation are about variance.** General distillation gives a low-variance representation cheaply; task distillation and ~20× augmentation densely sample the teacher on a small task set. Use them when task data is scarce.
- **Match pre-softmax attention scores, not post-softmax weights.** Faster convergence, fewer vanishing-gradient pathologies.
- **Distill first, quantize second.** The levers are orthogonal and stack near-multiplicatively (DistilBERT + int8 ≈ 6× smaller, 3–4× faster); the training-time lever goes before the deploy-time one.
- **Pick the recipe by modality and budget.** DistilBERT for low-effort transformers, TinyBERT/MobileBERT for aggressive compression (per-task quality vs deploy-many), feature+logit KD for vision CNNs. And if a from-scratch small model already hits target, do not distill at all.
- **Re-measure on the device.** Paper speedups are on paper hardware; your real number depends on compute-bound vs memory-bound, kernel support, and batch=1 reality. Warm up, report p50/p99.

## 14. Further reading

- **Hinton, Vinyals, Dean (2015), *Distilling the Knowledge in a Neural Network*** — the original soft-target / temperature / dark-knowledge paper. Read it first.
- **Sanh, Debut, Chaumond, Wolf (2019), *DistilBERT, a distilled version of BERT*** — the triple loss, init-from-teacher, and the 40/60/97 numbers. Official code in the `transformers` repo under `examples/research_projects/distillation`.
- **Jiao et al. (2020), *TinyBERT: Distilling BERT for Natural Language Understanding*** — transformer-layer (attention + hidden + embedding) distillation, two stages, data augmentation.
- **Sun et al. (2020), *MobileBERT: a Compact Task-Agnostic BERT for Resource-Limited Devices*** — bottleneck blocks, IB-BERT teacher, progressive knowledge transfer.
- **Romero et al. (2015), *FitNets: Hints for Thin Deep Nets*** — the seminal feature-map distillation for CNNs. And **Beyer et al. (2022), *Knowledge distillation: A good teacher is patient and consistent*** — consistent augmentation and long schedules for vision.
- Within this series: [a taxonomy of model compression](/blog/machine-learning/edge-ai/a-taxonomy-of-model-compression) for the four-lever Pareto frame; [knowledge distillation fundamentals](/blog/machine-learning/edge-ai/knowledge-distillation-fundamentals) for the theory these recipes instantiate; [what to distill: response, feature, relation](/blog/machine-learning/edge-ai/what-to-distill-response-feature-relation) for the supervision-type axis; [distilling LLMs and reasoning](/blog/machine-learning/edge-ai/distilling-llms-and-reasoning) for the generative scale-up; and the capstone [the edge optimization playbook](/blog/machine-learning/edge-ai/the-edge-optimization-playbook) for how distillation composes with the other levers end to end.
