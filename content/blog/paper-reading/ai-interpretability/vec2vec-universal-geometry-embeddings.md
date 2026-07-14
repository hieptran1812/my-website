---
title: "Harnessing the Universal Geometry of Embeddings: How vec2vec Translates Between Vector Spaces With Zero Paired Data"
date: "2026-07-14"
description: "A deep read of vec2vec — the first method that translates text embeddings from one model's space into another's with no paired data, no encoder access, and a striking security payload for vector databases."
tags: ["paper-reading", "embeddings", "vec2vec", "representation-learning", "platonic-representation-hypothesis", "unsupervised-translation", "embedding-inversion", "vector-database", "generative-adversarial-network", "cycle-consistency", "information-retrieval", "security"]
category: "paper-reading"
subcategory: "AI Interpretability"
author: "Hiep Tran"
featured: true
readTime: 31
paper:
  title: "Harnessing the Universal Geometry of Embeddings"
  authors: "Rishi Jha, Collin Zhang, Vitaly Shmatikov, John X. Morris"
  venue: "NeurIPS 2025 (arXiv:2505.12540)"
  url: "https://arxiv.org/abs/2505.12540"
---

> [!tldr]
> - **What it claims.** Two text-embedding models trained on different data, with different architectures and even different output dimensions, encode the *same* underlying geometry. That geometry can be learned and used to translate a vector from one model's space into another's — with **no paired examples, no access to the source encoder, and no known set of candidate matches**.
> - **The mechanism, in one line.** Per-space input/output *adapters* wrap a single *shared backbone* `T`; the whole thing is trained like a CycleGAN — adversarial losses make translations indistinguishable from real target-space vectors, while cycle-consistency and geometry-preserving losses keep the mapping honest.
> - **Why it matters.** vec2vec reaches cosine similarity up to **0.92** to the true target vector and **perfect nearest-neighbour matching (top-1 = 1.00)** on thousands of shuffled embeddings across model families — where every prior baseline degrades to random guessing.
> - **The surprising result.** A stolen dump of *embeddings only* — no text, no encoder — is enough to recover an email's topic, a patient's disease codes, and even names and dates. Off-the-shelf inversion applied to vec2vec translations leaks information for **up to 80% of emails**.
> - **Where it's soft.** GAN training is unstable: only **3 of 15** cross-backbone seeds converged, and the headline numbers use best-of-many initialisations. The "Strong Platonic Representation Hypothesis" is *demonstrated by existence*, not proven.

![Figure 1 from Jha et al. (2025): embeddings of the same texts from two different models (left, red vs teal) are pulled apart; vec2vec's shared latent representation (right) collapses them onto one another.](/imgs/blogs/vec2vec-universal-geometry-embeddings-fig1.webp)

The diagram above is the whole paper at a glance. On the left, embeddings of the *same sentences* produced by two different models — a T5-based retriever and a BERT-based one — live in two disjoint blobs with faint threads connecting matched pairs. They are incompatible: cosine similarity between a text's two embeddings is near zero. On the right, after passing both through vec2vec's learned adapters, the two blobs fuse into a single cloud where matched pairs sit on top of each other. The rest of this post unpacks how that fusion is learned without ever telling the model which red point matches which teal point — and what that means for anyone storing embeddings in a database.

## The problem: embeddings that cannot talk to each other

Text embeddings are the substrate of modern NLP. Retrieval, RAG, clustering, classification, deduplication, recommendation — all of it runs on the premise that a good encoder maps semantically similar texts to nearby vectors. If you have read [how embedding models are trained](/blog/machine-learning/large-language-model/embedding-models-training-finetuning-case-studies), you know the recipe: contrastive learning pulls positives together and pushes negatives apart until the geometry of the vector space mirrors the geometry of meaning.

Here is the friction. *Semantics is a property of the text, not of the model.* The sentence "Global wool production is about 2 million tonnes per year" means the same thing whether GTR or GTE encodes it. So two good encoders *ought* to agree about which texts are similar to which. And yet, if you embed a corpus with GTR and again with GTE and compare the two vectors for the same document, they look unrelated — cosine similarity hovers around zero. Each model has carved meaning into its own private coordinate system: different basis, different scale, different rotation, sometimes even a different number of dimensions (512 for CLIP, 768 for most BERT-based encoders, 2560 for Qwen3).

That incompatibility has always been treated as a fact of life. If you trained your RAG index on `text-embedding-3-small` and want to migrate to a new model, you re-embed the whole corpus. There is no adapter you can bolt on to reuse the old vectors.

### Why "matching" doesn't solve it

There is a large literature on *matching* or *correspondence* between embedding sets — Gromov-Wasserstein alignment, optimal-transport solvers, graph-matching. All of them share a crippling assumption: they need the two sets of embeddings to come from **the same or heavily overlapping inputs**. For every unknown vector, there must already exist a set of candidate vectors — computed by the other encoder on the *same* texts — to match against. Cross-lingual word-translation methods lean on the same crutch: overlapping vocabularies give you anchor points.

In the setting that actually matters — you hold a pile of embeddings from an unknown model and want to make sense of them — you have no such anchors. You do not have the source texts. You cannot query the source encoder. There is no candidate set. Matching has nothing to match against.

The paper frames this precisely. You hold a collection $\{u_1, \dots, u_n\}$ where each $u_i = M_1(d_i)$ was produced by an **unknown** encoder $M_1: \mathcal{V}^s \to \mathbb{R}^{d_{M_1}}$ from an **unknown** document $d_i$. You cannot query $M_1$, you do not know its training data or architecture, and you have no other view of the documents beyond their vectors. Your only asset is a *different* encoder $M_2$ that you *can* query freely, plus one weak piece of prior knowledge: the documents are text, in (say) English.

![Figure 3 from Jha et al. (2025): the unsupervised-translation setup. From only $u_i = M_1(d_i)$ — with $d_i$, $v_i$, and $M_1$ all invisible — vec2vec must produce $F(u_i)$ that lands near the true target embedding $v_i = M_2(d_i)$.](/imgs/blogs/vec2vec-universal-geometry-embeddings-fig3.webp)

The goal, drawn above: learn a translation $F$ such that $F(u_i) \approx v_i$, where $v_i = M_2(d_i)$ is the embedding the *known* encoder would have produced for the same document — a document you never see. Solve that, and every tool built for $M_2$ (inversion models, zero-shot classifiers) suddenly works on the opaque $M_1$ vectors. This is genuinely harder than matching: no encoder access, no paired data, no candidate set. The only thing you can lean on is shared *geometric structure* between the two spaces — and whether that structure is rich enough to bridge the gap is exactly the empirical question the paper answers.

## The Strong Platonic Representation Hypothesis

Why would you even expect this to be possible? The bet rests on a conjecture.

The original **Platonic Representation Hypothesis** (Huh et al., 2024 — analyzed separately in [this post on the Platonic Representation Hypothesis](/blog/paper-reading/ai-interpretability/position-the-platonic-representation-hypothesis)) observes that as vision models grow in scale and capability, their internal representations *converge*: measured by kernel-alignment metrics, big models trained on different data end up representing the world in increasingly similar ways, as if reaching toward one shared "Platonic" structure of reality. It is an **observational** claim — a similarity you can measure after the fact.

Jha et al. push it into a **constructive, stronger** form. Their **Strong Platonic Representation Hypothesis** states:

> Neural networks trained with the same objective and modality, but with different data and model architectures, converge to a universal latent space such that a translation between their respective representations can be learned **without any pairwise correspondence.**

The difference is everything. The original hypothesis says "these two spaces are *similar*." The strong version says "these two spaces are *the same space in disguise*, and I can hand you the map between them — built from unpaired samples alone." Similarity is something you observe; a learnable, correspondence-free translation is something you *build and test*. vec2vec is that test.

Think of it as two cartographers who each surveyed the same continent independently. One drew the map rotated 30 degrees with north-east in the corner and distances in leagues; the other used a different projection, different units, a different origin. The maps look nothing alike pixel-for-pixel. But because they describe the *same continent*, there exists a single transformation — rotate, rescale, re-project — that turns one into the other. Crucially, you can *recover that transformation from the maps alone*, without anyone telling you "this city on map A is that city on map B," because the relational structure (which cities are close, which rivers meet) is preserved on both. vec2vec is the machine that recovers the transformation between two embedding "maps" of the same semantic continent.

The analogy has a load-bearing limit, and the paper respects it: the transformation is *learned and approximate*, not an exact isometry. Two encoders do not agree perfectly; they agree *enough* that a neural network with the right inductive biases can find the bridge. How much is "enough," and where it breaks, is what the experiments quantify.

## What vec2vec actually contributes

Stripping the paper to its load-bearing claims:

1. **The first unsupervised embedding translator.** A method that maps a vector from an unknown model's space into a known model's space with no paired data, no encoder access, and no candidate set — a regime where all prior matching/correspondence methods fail.
2. **A learned universal latent space.** A shared representation in which embeddings of the same text from *different* encoders become nearly identical, even when the original embeddings have near-zero cosine similarity.
3. **Out-of-distribution robustness.** Translators trained only on Natural Questions (Wikipedia-flavoured) still translate tweets and clinical records — domains full of emojis and disease jargon the translator never saw.
4. **A concrete security attack.** Translation plus off-the-shelf inversion and zero-shot classification turns a stolen embedding database into recovered document content, quantified on emails, tweets, and medical records.

The rest of the post climbs the intuition-to-math ladder on each mechanism, then interrogates how much of the headline story survives scrutiny.

## The method: one backbone, many adapters

vec2vec borrows its skeleton from unsupervised image-to-image translation — specifically the CycleGAN family, which learns to turn horses into zebras without a single paired horse/zebra photo. (If GANs are hazy, the [GAN explainer](/blog/machine-learning/image-generation/generative-adversarial-networks-and-why-they-lost) covers the adversarial game and why cycle-consistency became its crutch.) The twist here is that embeddings have no spatial structure — no pixels, no locality — so the convolutional machinery is replaced with plain MLPs, and the design leans hard on a *modularity* trick.

### The architecture: adapters around a shared core

The central design decision is to *not* learn a monolithic $M_1$-space-to-$M_2$-space function. Instead, every embedding space gets its own small **adapter** into and out of a **single shared backbone**.

![A redrawn view of vec2vec's modular architecture: each embedding space plugs into one universal backbone `T` through its own input adapter (encode) and output adapter (decode); translation and reconstruction reuse the same `T`.](/imgs/blogs/vec2vec-universal-geometry-embeddings-1.webp)

Concretely, define:

- **Input adapters** $A_1, A_2: \mathbb{R}^d \to \mathbb{R}^Z$. Each takes a vector from one encoder's native space (dimension $d$) into a shared latent space of dimension $Z$. $A_1$ handles $M_1$'s space, $A_2$ handles $M_2$'s.
- **Shared backbone** $T: \mathbb{R}^Z \to \mathbb{R}^Z$. The universal transform, *shared by every space*. This is the "Platonic" core — the thing that is supposed to capture the geometry common to all encoders.
- **Output adapters** $B_1, B_2: \mathbb{R}^Z \to \mathbb{R}^d$. Each maps the shared latent back out into a specific encoder's space.

From these five modules, both the translations and the "do-nothing" reconstructions fall out as compositions:

$$
F_1 = B_2 \circ T \circ A_1, \qquad F_2 = B_1 \circ T \circ A_2
$$
$$
R_1 = B_1 \circ T \circ A_1, \qquad R_2 = B_2 \circ T \circ A_2
$$

Read them carefully, because the composition *is* the idea. $F_1$ translates an $M_1$ vector to $M_2$'s space: adapt in with $A_1$, transform with the shared $T$, adapt out with $B_2$ (the *other* space's output adapter). $R_1$ reconstructs an $M_1$ vector back to $M_1$'s space: same $A_1$, same $T$, but out through $B_1$ (its *own* output adapter). The only difference between translating and reconstructing is which output adapter you exit through. All the "meaning" lives in the shared $T$; the adapters are just per-space plugs. The full parameter set is $\theta = \{A_1, A_2, T, B_1, B_2\}$.

Why this factorisation matters: it forces $A_1$ and $A_2$ to map into a *common* latent that a *single* $T$ can process. If the shared backbone is to work for both spaces, the adapters have no choice but to discover the shared geometry — there is nowhere else to put it. And it scales: to add a third encoder $M_3$, you train one new pair $(A_3, B_3)$ against the *existing frozen* backbone, rather than a fresh $O(k^2)$ set of pairwise translators for $k$ models. (The paper trains pairwise, but the architecture makes the many-model extension obvious.)

Each module is an MLP with residual connections, layer normalization, and SiLU nonlinearities — nothing exotic. Discriminators (introduced next) mirror the structure but drop the residual connections "to simplify adversarial learning." No convolutions, because embeddings have no spatial bias to exploit.

### Loss 1: adversarial — make the fakes indistinguishable

**The problem it solves.** Suppose you only asked that $F_1$ map $M_1$ vectors somewhere into $M_2$'s space. Nothing stops it from collapsing every input to a single popular point, or producing vectors that are *individually* plausible but *collectively* wrong — the wrong distribution. You need a force that says "the cloud of translated vectors must look like the cloud of *real* $M_2$ vectors."

**Intuition.** This is the forger-versus-detective game at the heart of every GAN. The generator ($F$) forges $M_2$-space vectors; a discriminator ($D$) is a detective trained to separate forgeries from genuine $M_2$ vectors. The generator wins when the detective can no longer tell them apart — which, at the Nash equilibrium of this game, means the forged distribution matches the real one.

**Mechanism.** vec2vec plays this game at *two levels*, which is the clever part:

- **Output level.** Discriminators $D_1, D_2$ look at final embeddings and ask "is this a real vector in $M_1$'s / $M_2$'s space, or a translation?"
- **Latent level.** Discriminators $D_1^\ell, D_2^\ell$ look at the *shared latent* codes $T(A_1(u))$ and $T(A_2(v))$ and ask "did this latent come from an $M_1$ vector or an $M_2$ vector?" If the discriminator *cannot* tell, the two encoders' latents are distributionally identical — which is exactly the "universal latent" claim, enforced as a training signal.

**The math.** Applying the standard GAN objective at both levels:

$$
\mathcal{L}_{\text{adv}} = \mathcal{L}_{\text{GAN}}(D_1, F_1) + \mathcal{L}_{\text{GAN}}(D_2, F_2) + \mathcal{L}_{\text{GAN}}(D_1^\ell,\, T \circ A_1) + \mathcal{L}_{\text{GAN}}(D_2^\ell,\, T \circ A_2)
$$

where each $\mathcal{L}_{\text{GAN}}(D, G)$ is the usual min-max cross-entropy: the discriminator $D$ maximizes its ability to classify real-versus-generated, and the generator $G$ minimizes it. The first two terms police the *output* distributions; the last two police the *latent* distributions.

**Why the latent-level GAN is load-bearing.** Look ahead to the ablations: removing just the latent-level discriminator drops cosine similarity from **0.75 to 0.49** and destroys nearest-neighbour matching (top-1 falls to 0.00). The latent GAN is what actively *fuses* the two clouds in Figure 1's right panel — without it, the adapters are free to map into two separate regions of the latent that $T$ processes independently, and the "universal" space never forms.

### Loss 2: reconstruction — don't lose information on the round trip

**The problem.** The adapters could satisfy the adversarial loss while quietly throwing away information — mapping distinct inputs to indistinguishable latents that still *look* like the right distribution. You need to guarantee the encode-then-decode round trip is lossless within a space.

**Intuition.** An autoencoder's contract: if I encode $x$ into the latent and decode it straight back to its own space, I should recover $x$. If I can't, the latent isn't preserving what makes $x$ itself.

**The math.**

$$
\mathcal{L}_{\text{rec}} = \mathbb{E}_{x \sim p}\big\| R_1(x) - x \big\|_2^2 \; + \; \mathbb{E}_{y \sim q}\big\| R_2(y) - y \big\|_2^2
$$

where $p$ and $q$ are the empirical distributions of embeddings from $M_1$ and $M_2$, and $R_1 = B_1 \circ T \circ A_1$, $R_2 = B_2 \circ T \circ A_2$ are the same-space reconstructions from before. $\|\cdot\|_2^2$ is squared Euclidean distance. This is a pure autoencoding penalty — it never crosses spaces — but because it reuses the *shared* $T$, it forces $T$ to retain enough information for *both* spaces to decode from the common latent.

### Loss 3: cycle-consistency — the unsupervised stand-in for paired data

**The problem.** With no paired data, you cannot directly penalize "$F_1(u_i)$ is far from the true $v_i$" — you never see $v_i$. You need a self-supervised proxy for correctness.

**Intuition.** Translate a sentence from English to French and back to English. If you get the original sentence, the translation is probably faithful; if you get gibberish, it wasn't. Cycle-consistency turns this round-trip test into a loss. It is CycleGAN's core trick, and it is what makes correspondence-free training possible at all.

**Mechanism.** Take an $M_1$ vector $x$. Translate to $M_2$'s space with $F_1$, then translate back with $F_2$. You should land where you started. Same for the other direction.

**The math.**

$$
\mathcal{L}_{\text{CC}} = \mathbb{E}_{x \sim p}\big\| F_2(F_1(x)) - x \big\|_2^2 \; + \; \mathbb{E}_{y \sim q}\big\| F_1(F_2(y)) - y \big\|_2^2
$$

**Worked micro-example.** Say $x = [1.0,\, 0.0]$ lives in $M_1$'s (toy 2-D) space. $F_1$ sends it into $M_2$'s space as some vector; $F_2$ brings it back as $F_2(F_1(x)) = [0.9,\, 0.1]$. The cycle loss for this sample is $\|[0.9, 0.1] - [1.0, 0.0]\|_2^2 = (0.1)^2 + (0.1)^2 = 0.02$. Drive that to zero across the batch and the two translators become approximate inverses — which, combined with the adversarial pressure to hit the right *distribution*, is a strong hint that they are the *correct* inverses. Ablating this loss drops cosine from **0.75 to 0.50**: cycle-consistency contributes roughly as much as the latent GAN.

### Loss 4: vector-space preservation — keep the relational geometry

**The problem.** Cycle-consistency constrains the round trip but not the *shape* of the translated cloud. A translator could shuffle the geometry — swap which vectors are near which — while still being cycle-consistent on average. Retrieval and semantics both depend on *relative* distances, so those must survive translation.

**Intuition.** Back to the cartographers: a faithful map transformation preserves which cities are close to which. If Paris and Lyon are near on map A, their translated positions on map B must stay near. Vector-space preservation (VSP) penalizes any translation that distorts pairwise similarities.

**Mechanism.** For a batch, compute all pairwise dot products (the Gram matrix) in the source space, and again among the *translated* vectors. Penalize the difference. In the paper's exact form, over a batch of size $B$:

$$
\mathcal{L}_{\text{VSP}} = \frac{1}{B^2}\sum_{i=1}^{B}\sum_{j=1}^{B} \Big( \big\| M_1(x_i)^\top M_1(x_j) - F_2(M_2(y_i))^\top F_2(M_2(y_j)) \big\|_2^2 + \big\| M_2(y_i)^\top M_2(y_j) - F_1(M_1(x_i))^\top F_1(M_1(x_j)) \big\|_2^2 \Big)
$$

Read the first term: the pairwise dot products among *native* $M_1$ vectors ($M_1(x_i)^\top M_1(x_j)$) should match the pairwise dot products among vectors *translated into* $M_1$'s space from $M_2$ ($F_2(M_2(y_i))^\top F_2(M_2(y_j))$). It is a distribution-level geometry match: translated vectors must reproduce the target space's *relational* structure, not just its marginal distribution.

**Worked micro-example.** Two source vectors with $x_i^\top x_j = 0.8$ (fairly similar). If after translation $F(x_i)^\top F(x_j) = 0.5$, this pair contributes $(0.8 - 0.5)^2 = 0.09$ to the loss — the translator compressed their similarity and gets punished for it. Ablating VSP drops cosine from **0.75 to 0.58**, the gentlest of the three but still decisive for nearest-neighbour matching (top-1 collapses to 0.00 without it).

### Putting the objective together

The full training problem is a min-max: minimize over the generator parameters $\theta$, maximize over all four discriminators.

$$
\theta^{*} = \arg\min_{\theta}\; \max_{D_1, D_2, D_1^\ell, D_2^\ell}\; \mathcal{L}_{\text{adv}} + \lambda_{\text{gen}}\,\mathcal{L}_{\text{gen}}, \qquad \mathcal{L}_{\text{gen}} = \lambda_{\text{rec}}\mathcal{L}_{\text{rec}} + \lambda_{\text{CC}}\mathcal{L}_{\text{CC}} + \lambda_{\text{VSP}}\mathcal{L}_{\text{VSP}}
$$

The hyperparameters $\lambda_{\text{gen}}, \lambda_{\text{rec}}, \lambda_{\text{CC}}, \lambda_{\text{VSP}}$ trade off the adversarial pressure (get the distribution right) against the generator constraints (get the *content* right). The four losses are not redundant — each defends against a different failure mode, and the ablations show removing any one collapses the method.

![The four losses and what breaks without each: the objective is $\mathcal{L}_{adv} + \lambda\,\mathcal{L}_{gen}$; removing any single term collapses cross-backbone cosine from 0.75 toward the naïve baseline's 0.04.](/imgs/blogs/vec2vec-universal-geometry-embeddings-2.webp)

Here is a compact PyTorch-shaped sketch of a single training step, to make the moving parts concrete:

```python
# u: batch of M1 embeddings  [B, d1]   (unknown space, source)
# v: batch of M2 embeddings  [B, d2]   (known space, target)
# NOTE: u and v are UNPAIRED — different texts, no correspondence.

# ---- forward through the modular generator ----
zu = T(A1(u))              # [B, Z]  M1 -> shared latent
zv = T(A2(v))              # [B, Z]  M2 -> shared latent

F1_u = B2(zu)              # [B, d2] translate M1 -> M2 space
F2_v = B1(zv)              # [B, d1] translate M2 -> M1 space
R1_u = B1(zu)              # [B, d1] reconstruct M1 -> M1
R2_v = B2(zv)              # [B, d2] reconstruct M2 -> M2

# ---- generator losses ----
L_rec = mse(R1_u, u) + mse(R2_v, v)                     # round trip in-space
L_cc  = mse(B1(T(A2(F1_u))), u) + mse(B2(T(A1(F2_v))), v)   # F2(F1(u)) ~ u
L_vsp = gram_mismatch(u, F2_v) + gram_mismatch(v, F1_u)  # pairwise dot products

# ---- adversarial losses (two levels) ----
L_adv  = gan_g(D2(F1_u)) + gan_g(D1(F2_v))              # output-level
L_adv += gan_g(Dl2(zu))  + gan_g(Dl1(zv))              # latent-level: fuse the clouds

L_gen = L_adv + lam_gen * (lam_rec*L_rec + lam_cc*L_cc + lam_vsp*L_vsp)
L_gen.backward(); opt_G.step()

# discriminators take the opposite gradient on real vs generated ...
```

The whole method is roughly "CycleGAN, but MLPs, but with a shared backbone and a second discriminator on the latent." None of the ingredients is new; the *combination*, and the demonstration that it works across model families with zero pairing, is the contribution.

## How they measure success

Because vec2vec is the first method of its kind, there is no established baseline — so the evaluation design deserves attention.

Let $u_i = M_1(d_i)$ and $v_i = M_2(d_i)$ be the source and target embeddings of the *same* document $d_i$ (used only for scoring, never for training). A translation $F(u_i)$ is judged by how it ranks the true target $v_i$ among all candidate targets, ordered by cosine similarity to $F(u_i)$. Three metrics:

- **Mean cosine similarity** — the average $\cos(F(u_i), v_i)$ between a translation and its true target. Ideal: 1.0.
- **Top-1 accuracy** — the fraction of translations whose *nearest* candidate (by cosine) is the correct target. This is the strict "did you find the needle" metric. Ideal: 1.0.
- **Mean rank** — the average position of the true target in the similarity-sorted candidate list. Ideal: 1.0. A *random* ordering over $n$ candidates gives mean rank $n/2$.

That last number is the sanity anchor. In the in-distribution table, candidates are drawn in batches of 8192, so **random guessing scores mean rank ≈ 4096**. A method landing at rank 1.2 is not "a bit better than chance" — it is finding the exact match almost every time out of thousands, with no candidate set given in advance.

Two baselines frame the results:

- **Naïve** — just use $F(x) = x$, i.e. pretend the two spaces are already aligned. This measures raw geometric overlap between the encoders. For *same-backbone* pairs (two BERT-based models) it is surprisingly non-trivial; for *cross-backbone* pairs it is essentially random.
- **Oracle-aided optimal transport** — solve the assignment problem $\pi^{*} = \arg\min_\pi \sum_i \cos(u_i, v_{\pi(i)})$ with Hungarian / EMD / Sinkhorn / Gromov-Wasserstein solvers, reporting the best. This baseline is *strictly easier* than vec2vec's task: it is *given* the candidate set and only has to match within it. It is a pseudo-oracle, and vec2vec still has to beat it while solving the harder problem.

## Results: the universal latent is real

The first result is a picture of the mechanism working. For every ordered pair of five encoders, the paper measures pairwise cosine similarity between the *input* embeddings, and between their *vec2vec latents*, on the same batch of 1024 Natural Questions texts.

![Figure 4 from Jha et al. (2025): pairwise cosine similarities between input embeddings (left, mostly blue/low) versus their vec2vec latents (middle, mostly red/high); the right panel is the difference. Cross-model input similarity near zero becomes near-one in the shared latent.](/imgs/blogs/vec2vec-universal-geometry-embeddings-fig4.webp)

The left heatmap is mostly cold: input embeddings from different encoders have cosine similarity near zero (e.g., granite-vs-gtr at **-0.02**), with warmth only among same-backbone BERT models (gte-vs-e5 at **0.68**). The middle heatmap is mostly hot: in the shared latent, the *same* pairs jump to **0.75–0.96**. The right panel — their difference — is where vec2vec did the most work: the largest jumps are exactly the cross-backbone pairs that started near zero. The universal latent is not a metaphor; it is a measured collapse of cross-model distance.

### Translation quality across model families

The headline table (in-distribution, evaluated on 65,536 held-out NQ texts in batches of 8192):

| Direction | Metric | vec2vec | Naïve | Oracle-OT |
|---|---|---|---|---|
| granite → gtr (cross-backbone) | cos ↑ | **0.80** | -0.03 | 0.70 |
| | top-1 ↑ | **0.99** | 0.00 | 0.00 |
| | rank ↓ (of 8192) | **1.19** | 4168 | 2774 |
| stella → gte (cross-backbone) | cos ↑ | **0.92** | 0.56 | 0.87 |
| | top-1 ↑ | **1.00** | 1.00 | 1.00 |
| | rank ↓ | **1.00** | 1.00 | 1.00 |
| e5 → gtr (cross-backbone) | cos ↑ | **0.74** | 0.03 | 0.70 |
| | top-1 ↑ | **0.82** | 0.00 | 0.00 |
| | rank ↓ | **2.56** | 4081 | 3877 |

Two patterns matter. For **cross-backbone** pairs (a BERT model to a T5 or RoBERTa model), the naïve baseline is pure noise — rank ≈ 4096, top-1 = 0 — and even the oracle-aided OT baseline, *given the candidate set*, cannot match: top-1 stays at 0.00. vec2vec, solving the harder correspondence-free problem, hits top-1 of **0.82–0.99** and rank near 1. For **same-backbone** pairs (e.g. stella/gte, both BERT), the embeddings are already close enough that even the naïve baseline matches perfectly — vec2vec still lifts cosine but the matching was never the hard part there.

The gap is the story: where the spaces are genuinely different, prior methods fall to chance and vec2vec does not.

### Out-of-distribution: trained on Wikipedia, tested on tweets and patient records

The translators are trained only on Natural Questions — Wikipedia-derived queries and answers. The paper then evaluates them, unchanged, on two very different domains:

- **TweetTopic** — 800 tweets, colloquial, emoji-laden, nothing like Wikipedia.
- **MIMIC** — a pseudo-re-identified subset of clinical records labelled with 2,673 disease descriptions, full of medical jargon.

On TweetTopic, cross-backbone pairs like granite → gte still reach cosine **0.85**, top-1 **0.95**, rank **1.26**. On MIMIC (8192 candidates, random rank 4096), stella → gte holds top-1 **1.00**, rank **1.00**, and even harder pairs like e5 → gtr land rank **1029** — far from perfect but wildly better than the 4096 of chance. The translator learned something *domain-agnostic* about the geometry relating the two encoders, not something specific to Wikipedia text. That generalisation is one of the paper's stronger pieces of evidence for the "universal" claim.

### Crossing modalities and model scales

Two stress tests extend the reach:

- **CLIP (multimodal, 512-dim).** Translating to and from CLIP — a model trained partly on *images*, with a different output dimension — is harder: cosine drops to **0.50–0.78** and top-1 is uneven. But vec2vec still consistently beats the Gromov-Wasserstein OT baseline (which scores top-1 = 0.00 across the board). The shared geometry apparently survives even a partial modality gap.
- **Qwen3 (4B params, 2560-dim).** For a single representative pair, vec2vec translates gte → qwen at cosine **0.50**, top-1 **0.92**, and qwen → gte at cosine **0.84**, top-1 **0.88** — bridging a model 14× larger than the next-biggest and 37× larger than GTE, with a different backbone and training recipe. One pair is thin evidence (Qwen is compute-heavy, so they ran it once), but the direction is striking.

### Ablations: every loss earns its place

The subtractive ablation (gte → gtr, a cross-backbone pair) is the cleanest argument that the design is not over-built:

| Configuration | cos ↑ | top-1 ↑ | rank ↓ (of 8192) |
|---|---|---|---|
| **Full vec2vec** | **0.75** | **0.91** | **2.64** |
| − VSP loss | 0.58 | 0.00 | 4197 |
| − cycle-consistency loss | 0.50 | 0.00 | 3941 |
| − latent GAN | 0.49 | 0.00 | 3897 |
| − VSP *and* cycle loss | 0.47 | 0.00 | 3365 |
| − hyperparameter tuning | 0.50 | 0.00 | 4012 |
| Naïve baseline | 0.04 | 0.00 | 4084 |
| Oracle-OT baseline | 0.70 | 0.00 | 3064 |

The pattern is stark: remove *any single component* and top-1 accuracy collapses from 0.91 to **0.00** and mean rank reverts toward random. Cosine similarity degrades more gracefully (0.75 → 0.49–0.58), which exposes something important — a translator can produce vectors that are *on average* moderately close in cosine (0.58 is not nothing) while completely failing to make the *right* vector the *nearest* one. High average cosine without nearest-neighbour matching means the geometry is smeared, not aligned. Only the full method achieves both.

### Data efficiency

You do not need millions of vectors. Training gte → gtr with a fixed 1M GTR embeddings but varying GTE counts:

| GTE embeddings | cos ↑ | top-1 ↑ | rank ↓ |
|---|---|---|---|
| 1,000,000 | 0.75 | 0.92 | 2.73 |
| 500,000 | 0.75 | 0.92 | 2.73 |
| 100,000 | 0.74 | 0.85 | 4.52 |
| 50,000 | 0.74 | 0.81 | 3.91 |
| 10,000 | 0.57 | 0.01 | 1462 |

At **50K** embeddings the translator is nearly as good as at 1M. At 10K it still "learns something" (rank 1462 beats random 4096) but nearest-neighbour matching breaks. For an attacker, the practical reading is chilling: a stolen database of 50,000 vectors is plenty.

## The security payload

Everything so far is a representation-learning result. This section is why the paper reads like a security disclosure.

![Figure 2 from Jha et al. (2025): given only a vector database from an unknown model, vec2vec translates it into a known model's space using latent structure alone; the recovered embeddings reveal the topic of a real Enron email. The inset shows the adapters/backbone and the cycle/GAN losses.](/imgs/blogs/vec2vec-universal-geometry-embeddings-fig2.webp)

The threat model is minimal and realistic. An adversary compromises a vector database and exfiltrates *only the embedding vectors* — no source text, no metadata, and critically **no knowledge of which model produced them**. Vector stores are exactly this: giant tables of floats. The classic assumption has been that a bare embedding is opaque without its encoder. vec2vec breaks that assumption.

![A redrawn end-to-end view of the attack: stolen vectors plus a freely queryable known encoder feed vec2vec, whose translations then power both attribute inference and zero-shot inversion, leaking sensitive content for up to 80% of documents.](/imgs/blogs/vec2vec-universal-geometry-embeddings-3.webp)

The attack chain, drawn above: (1) hold the stolen $\{u_i = M_1(d_i)\}$ with $M_1$ unknown; (2) pick any known encoder $M_2$ you can query; (3) train vec2vec on *unpaired* samples so that $F(u_i) \approx M_2(d_i)$; (4) apply tools built for $M_2$ — zero-shot attribute inference and zero-shot inversion — to the translated vectors; (5) read off sensitive content. Two extraction techniques run in parallel on the translated vectors.

### Zero-shot attribute inference

**The method.** To test whether a document $d_i$ has attribute $c \in C$ (a topic label, a disease description), embed all candidate attributes with $M_2$, then measure cosine similarity between the *translated* vector $F(M_1(d_i))$ and each attribute embedding. The top-$k$ closest attributes are the prediction; score with top-$k$ accuracy.

**The result.** On TweetTopic (topics) and MIMIC (disease codes), attribute inference on vec2vec translations consistently beats the naïve baseline and — remarkably — often matches or beats the *ideal* zero-shot classifier that has access to the true same-space embeddings ($M_1$ or $M_2$ of the document *and* the labels). That ideal baseline is imaginary in the attack setting (the attacker doesn't have same-space embeddings), which makes it a generous reference point that vec2vec sometimes still outperforms. Most telling: vec2vec correctly surfaces disease concepts like "alveolar periostitis" that essentially never appear in its Natural Questions training data. Preserving the semantics of a medical term it never trained on is hard to explain except by a genuinely universal latent.

### Zero-shot inversion

**The method.** Inversion reconstructs the *text* from an embedding. The paper uses an off-the-shelf, zero-shot inverter (Zhang et al., 2025) built for standard encoders, and simply feeds it vec2vec's translated vectors. Success is scored by a GPT-4o judge answering a blunt yes/no: does the reconstruction leak information about the original?

**The result.** For some model pairs, the inverter extracts information from **up to 80% of Enron emails** and **67% of tweets** — from translated embeddings alone, never touching the source encoder. The reconstructions are imperfect but leak plenty:

- Ground truth: *"Subject: Enron Bashing on Frontline / Body: Karen, please call me when you receive this email. Thank you. Rick"* → reconstruction: *"Some emails discussing NROn Employee/s Complaint To thePublic ..."* — garbled, but it surfaces "Enron," "employee," "complaint," "public."
- Ground truth: *"Subject: Trades for 3/1/02 ..."* → *"... future transactions may await John G..."* — recovers a name and the trading context.
- Ground truth: *"The following expense report is ready for approval..."* → *"The upcoming expense statement from ..."* — recovers the document type.

The paper is careful to call these "a lower bound" — they used a *generic* inverter, not one specialized for translated vectors. A determined attacker would train a purpose-built inverter and do better. The disclosure is not "embeddings leak a little"; it is "the model that produced your embeddings was never the security boundary you assumed it was."

### What it cost to run

The honesty of the compute footnote deserves a mention. Training the reported models consumed roughly **176 GPU-days**; the Qwen pair alone took **20 days on an A100**. GAN instability forced a best-of-multiple-initialisations protocol (more on this in the critique). This is not a cheap attack to reproduce end-to-end — but the *per-pair* cost, once you have a recipe, is modest, and the data requirement (50K vectors) is trivial.

## Critique

### What is genuinely strong

- **The task is new and the baselines are fair.** vec2vec defines correspondence-free translation and then beats an *oracle-aided* baseline that gets the candidate set for free. Winning while handicapped is convincing.
- **The two-level GAN is a real idea.** Discriminating on the *latent* as well as the output is what fuses the clouds, and the ablation proves it — remove it and matching dies. That is a transferable trick for any cross-representation alignment problem.
- **OOD generalisation is the best evidence.** A translator trained on Wikipedia that recovers disease codes from clinical records is hard to dismiss as overfitting to a dataset's quirks. It suggests the learned map really is about encoder-to-encoder geometry.
- **The security framing is responsible and concrete.** Real datasets (Enron, MIMIC), an explicit threat model, LLM-judged leakage, and repeated "this is a lower bound" caveats.

### Where it is soft

- **GAN instability undercuts the headline numbers.** Appendix E is the tell: for *cross-backbone* pairs — the hard, interesting case — only **3 of 15** random seeds converged to 80% top-1 within the epoch budget. Same-backbone pairs were stable (14/15). The main tables select the best of multiple initialisations. So the reported 0.92 cosine is closer to a *best-case* than an *expected-case*. An attacker who cannot afford to try many seeds, or who cannot tell a good run from a bad one without paired validation data (which the threat model says they lack), may not reach these numbers. The paper is upfront about this, but it materially softens "vec2vec translates" into "vec2vec *can* translate, sometimes, with luck and compute."
- **"Strong PRH holds in practice" is demonstrated by existence, not proven.** One working method on contrastively-trained *text* encoders shows the universal geometry is *harnessable here*. It does not show convergence is universal across objectives, or that a *single* fixed latent underlies all models. The evidence is "a bridge exists between these particular spaces," which is weaker than the hypothesis's universal quantifier.
- **The inversion "leakage" bar is lenient.** An 80% figure sounds alarming until you read the metric: a GPT-4o judge answering yes/no to "does the reconstruction leak *any* information?" A reconstruction that recovers only the word "Enron" from an Enron email counts as leakage. That is a real signal — but "80% leak" and "80% reconstruction fidelity" are very different claims, and a skimming reader will conflate them.
- **Cosine-similarity heatmaps are doing a lot of narrative work.** Figure 4's "the latents are 0.9 similar" is the emotional core of the universality claim, but cosine in a learned latent trained *specifically to make those numbers high* is somewhat circular as evidence. The nearest-neighbour matching results are the harder, less-gameable evidence and deserve to be the headline over the heatmaps.

### Missing ablations I wanted

- **Seed-robustness *in the main tables.*** The convergence statistics live in an appendix; I would want the main results reported as median-over-seeds with error bars, not best-of-$k$, precisely because the appendix shows the variance is enormous for the interesting pairs.
- **A truly held-out encoder family.** All five main encoders are contrastive text retrievers on similar-ish backbones. What happens translating to a decoder-only embedding, or a model trained with a genuinely different objective (masked LM vs contrastive)? The Qwen and CLIP results gesture at this but are single-pair anecdotes.
- **What the latent dimension $Z$ should be, and why.** The bottleneck dimension presumably controls how much geometry the shared backbone can hold, but there is no sweep.

### What would change my mind

If a *single fixed initialisation* — no best-of-many, no oracle to pick the good seed — reliably hit top-1 above 0.8 on cross-backbone pairs, I would upgrade "the Strong PRH is harnessable with luck" to "the Strong PRH is robustly demonstrated." And if the universal latent held across genuinely different training *objectives* (not just five variants of contrastive text retrieval), I would believe the "universal" in the title rather than reading it as "universal among similar encoders." As it stands, the paper convincingly demonstrates a bridge between a specific family of spaces — which is already enough to matter for security — while leaving the strongest form of its own hypothesis as a compelling conjecture.

## What I'd build with this

These are my extrapolations, not the paper's claims.

- **A defensive canary.** If a stolen embedding table can be translated and inverted, the defence is to make translation *fail*. I would explore training embedding encoders with an adversarial term that *maximises* cross-model latent distance — a "non-Platonic" encoder deliberately off the universal manifold — and measure the retrieval-quality cost of that privacy.
- **Embedding migration as a product.** The benign twin of the attack: reuse an old vector index under a new model without re-embedding the corpus. vec2vec's data efficiency (50K vectors) makes a "bring your own index" migration adapter plausible, if the seed-stability problem is solved.
- **Universal rerankers and multi-index search.** If many encoders share one latent, a reranker trained in that latent could score candidates from *heterogeneous* indices at once — useful for federated or multi-tenant [vector search](/blog/machine-learning/recommendation-systems/approximate-nearest-neighbor-serving-faiss-hnsw-scann) where different tenants use different encoders.
- **A stability fix worth a paper on its own.** The 3/15 convergence rate is the bottleneck for every application above. Swapping the vanilla GAN for a more stable adversarial objective (spectral-normalised, R1-regularised, or a non-adversarial distribution-matching loss) is the obvious next experiment, and the paper explicitly leaves it to future work.

## References

- Rishi Jha, Collin Zhang, Vitaly Shmatikov, John X. Morris. *Harnessing the Universal Geometry of Embeddings.* NeurIPS 2025. [arXiv:2505.12540](https://arxiv.org/abs/2505.12540). Code on [GitHub](https://github.com/rjha18/vec2vec).
- Minyoung Huh, Brian Cheung, Tongzhou Wang, Phillip Isola. *The Platonic Representation Hypothesis.* 2024 — analyzed on this blog: [The Platonic Representation Hypothesis](/blog/paper-reading/ai-interpretability/position-the-platonic-representation-hypothesis).
- Jun-Yan Zhu, Taesung Park, Phillip Isola, Alexei Efros. *Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks (CycleGAN).* 2017 — the adversarial + cycle-consistency template vec2vec adapts. Background on the family: [Generative adversarial networks and why they lost](/blog/machine-learning/image-generation/generative-adversarial-networks-and-why-they-lost).
- John X. Morris et al. *Text Embeddings Reveal (Almost) As Much As Text.* 2023, and Collin Zhang et al. *Universal Zero-shot Embedding Inversion.* 2025 — the inversion machinery vec2vec plugs into. For how the underlying encoders are built: [Embedding models: training and finetuning](/blog/machine-learning/large-language-model/embedding-models-training-finetuning-case-studies).
