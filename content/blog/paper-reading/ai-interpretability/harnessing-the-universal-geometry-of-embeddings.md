---
title: "Harnessing the Universal Geometry of Embeddings: Translating Vectors Between Models With No Paired Data"
publishDate: "2026-10-15"
date: "2026-10-15"
description: "How vec2vec learns to translate text embeddings from an unknown model into a known one using only unpaired samples, why that works if the Strong Platonic Representation Hypothesis holds, and what it means for anyone storing embeddings."
category: "paper-reading"
subcategory: "AI Interpretability"
tags: ["paper-reading", "vec2vec", "embeddings", "geometry", "representation-learning", "platonic-representation", "embedding-inversion", "cycle-consistency", "security", "neurips-2025"]
author: "Hiep Tran"
featured: true
readTime: 18
image: "/imgs/blogs/harnessing-the-universal-geometry-of-embeddings-fig1.webp"
excerpt: "Two embedding models with different architectures put the same sentence at nearly orthogonal coordinates. vec2vec learns, without a single paired example, a shared latent space where those vectors almost coincide, and then uses it to read secrets out of a leaked vector database."
paper:
  title: "Harnessing the Universal Geometry of Embeddings"
  authors: "Rishi Jha, Collin Zhang, Vitaly Shmatikov, John X. Morris"
  venue: "NeurIPS 2025"
  url: "https://arxiv.org/abs/2505.12540v4"
---

> [!tldr]
> **The claim.** Text embeddings from different models (different architectures, sizes, training data) can be translated into each other **without any paired data, encoder access, or candidate matches**. The method, vec2vec, maps every space into a learned universal latent and back out, trained only with adversarial, reconstruction, cycle-consistency and geometry-preservation losses.
>
> **Why it matters.** It is a constructive test of a stronger version of the Platonic Representation Hypothesis: the shared structure is not just measurable, it is learnable from unpaired samples. It also turns a leaked vector database from "opaque numbers" into "documents you can partially read".
>
> **The most surprising result.** Between models with different backbones, where the raw vectors of the same text have cosine similarity near 0, translations reach cosine 0.74 to 0.87 against the true target and top-1 matching of 0.82 to 1.00 among 8,192 shuffled candidates. Oracle optimal transport, which is told the candidates, gets top-1 of 0.00 on the same pairs.
>
> **Where it is thin.** Training is a GAN: only 3 of 15 seeds converged for a cross-backbone pair. Out-of-distribution medical records drop top-1 as low as 0.01 for some pairs, translations to and from CLIP are uneven, and with 10K training samples the method fails outright.

Your vector database leaks: millions of 768-dimensional float arrays, no text, no model name. Most teams treat that as low severity, because without the exact encoder a vector is "just numbers".

This paper asks whether you can convert embeddings from an **unknown** model into the space of a model you **do** control, using nothing but the embeddings. If you can, every tool built for known encoders, including inversion models that rebuild the text, works on the leak.

![Figure 2 from Jha et al. (2025): given only a vector database from an unknown model, vec2vec translates it into a known model's space, where inversion recovers the topic of a real Enron email](/imgs/blogs/harnessing-the-universal-geometry-of-embeddings-fig2.webp)

The figure above is the threat model: secret documents embedded by Encoder A, an attacker who owns Encoder B and public text, a translator learned from two piles of embeddings that never share a document, and an off-the-shelf inverter that reads the result.

## 1. The problem: translation, not matching

Formally, we hold $n$ vectors $\{u_1, \dots, u_n\}$ with $u_i = M_1(d_i)$. Here $M_1$ is an unknown encoder mapping text to $\mathbb{R}^{d_{M_1}}$ and $d_i$ is an unknown document. We can query a second encoder $M_2$ as much as we like. The goal is a function $F$ such that $F(u_i)$ lands close to the ideal $v_i = M_2(d_i)$, which we never see.

| Setting | What you are given | What you must produce |
| --- | --- | --- |
| Supervised alignment (Procrustes, CCA, model stitching) | Pairs $(M_1(d), M_2(d))$ for the same texts | A map between spaces |
| Matching / correspondence (optimal transport) | Two sets of vectors that embed the **same** texts | A permutation pairing them up |
| **Unsupervised translation (this paper)** | Unpaired vectors from each space, on **disjoint** texts | A new vector in $M_2$'s space for each $u_i$ |

Matching assumes each unknown vector already has a counterpart in the other set. A leaked database has none, so translation must **generate** the target vector, which is strictly harder.

Each vec2vec model trains on **disjoint** sets of Natural Questions (NQ): 1 million 64-token sequences embedded by $M_1$ and a different 1 million embedded by $M_2$.

## 2. The bet: the Strong Platonic Representation Hypothesis

Why should this be possible at all? The [Platonic Representation Hypothesis](/blog/paper-reading/ai-interpretability/position-the-platonic-representation-hypothesis) of Huh et al. conjectures that large models converge toward the same latent representation of the world. That claim is about **measurement**: two models agree on which inputs are near each other.

The authors propose a stronger, constructive version. In their words, networks trained with the same objective and modality, but different data and architectures, converge to a universal latent space **such that a translation between their representations can be learned without any pairwise correspondence**. If that is false, vec2vec cannot work. If vec2vec works, it is evidence the hypothesis holds.

An analogy: two cartographers map the same island, one rotated and mirrored relative to the other. Comparing grid coordinates is useless, but the **shape** (which towns are close) is identical, so the maps can be aligned by shape alone.

![Different coordinates, same shape: the raw vectors for "dog" in two spaces disagree, but the pairwise geometry of the constellation is preserved, which is what vec2vec learns to exploit](/imgs/blogs/harnessing-the-universal-geometry-of-embeddings-2.webp)

Across backbones, the cosine between $M_1(\text{dog})$ and $M_2(\text{dog})$ is near zero because the coordinate systems are unrelated. The relative angles among the points are what transfer, and the losses below lock onto them.

The paper's Figure 1 shows this on real data: GTE and GTR embeddings occupy separate regions (left), while their vec2vec latents overlap almost completely (right).

![Figure 1 from Jha et al. (2025): left, GTR and GTE embeddings are incomparable; right, unpaired samples from the two models land on nearly the same points in the learned latent](/imgs/blogs/harnessing-the-universal-geometry-of-embeddings-fig1.webp)

## 3. The architecture: adapters around a shared backbone

**Problem.** We need one latent space that both models can enter and leave. A single matrix from space 1 to space 2 has nothing to anchor it without pairs.

**Mechanism.** vec2vec is modular. Each space gets an **input adapter** that lifts its vectors into a latent of dimension $Z$, a **shared backbone** processes every latent, and each space gets an **output adapter** that maps latents back out.

$$
A_1, A_2 : \mathbb{R}^{d} \to \mathbb{R}^{Z}, \qquad T : \mathbb{R}^{Z} \to \mathbb{R}^{Z}, \qquad B_1, B_2 : \mathbb{R}^{Z} \to \mathbb{R}^{d}
$$

Here $d$ is the embedding width of each space (768 for most models in the paper, 512 for CLIP, 2560 for Qwen3), $Z$ is the latent width, and $T$ is the shared backbone. Composing these gives four functions, two translators $F$ and two reconstructors $R$:

$$
F_1 = B_2 \circ T \circ A_1, \quad F_2 = B_1 \circ T \circ A_2, \quad R_1 = B_1 \circ T \circ A_1, \quad R_2 = B_2 \circ T \circ A_2
$$

Read $F_1$ as "enter from space 1, leave through space 2". The parameters are $\theta = \{A_1, A_2, T, B_1, B_2\}$; translation and reconstruction differ only in the exit adapter.

![The vec2vec architecture: space-specific adapters wrap one shared backbone; translators and reconstructors differ only in their exit adapter, and four discriminators police outputs and latents](/imgs/blogs/harnessing-the-universal-geometry-of-embeddings-1.webp)

**Implementation.** Embeddings have no spatial structure, so every component is an MLP with residual connections, layer normalization and SiLU. Discriminators drop the residuals to simplify adversarial training.

**When it fails.** If two models do not share latent structure, no $T$ serves both adapters, and training settles on something that fools the discriminators without preserving meaning. The ablations in Section 8 show that failure.

## 4. The losses: four ways to say "keep the meaning"

The full objective is a min-max game between the generator parameters $\theta$ and four discriminators:

$$
\theta^{*} = \arg\min_{\theta} \max_{D_1, D_2, D_1^{\ell}, D_2^{\ell}} \; \mathcal{L}_{\text{adv}}(F_1, F_2, D_1, D_2, D_1^{\ell}, D_2^{\ell}) + \lambda_{\text{gen}} \, \mathcal{L}_{\text{gen}}(\theta)
$$

$D_1, D_2$ judge **output embeddings** and $D_1^{\ell}, D_2^{\ell}$ judge **latents**. $\lambda_{\text{gen}}$ trades off the adversarial term against the generator constraints. We take the pieces one at a time.

### 4.1 Adversarial loss: match the distribution

**Problem.** Without pairs, the only signal we have about space 2 is what its vectors look like as a cloud. **Analogy:** a forger who has never seen the original painting, only many paintings by the same artist, learns to produce work an expert cannot tell apart.

**Math.** The standard GAN loss is applied at two levels:

$$
\mathcal{L}_{\text{adv}} = \mathcal{L}_{\text{GAN}}(D_1, F_1) + \mathcal{L}_{\text{GAN}}(D_2, F_2) + \mathcal{L}_{\text{GAN}}(D_1^{\ell}, T \circ A_1) + \mathcal{L}_{\text{GAN}}(D_2^{\ell}, T \circ A_2)
$$

Each term is the usual Goodfellow et al. objective, $\mathcal{L}_{\text{GAN}}(D, G) = \mathbb{E}_{y}[\log D(y)] + \mathbb{E}_{x}[\log(1 - D(G(x)))]$, where $D(\cdot)$ is the discriminator's probability that its input is real, $y$ is a real sample from the target distribution and $G(x)$ is a generated one. The output-level terms push translations to look like real target embeddings. The latent-level terms push latents from either space to be indistinguishable from each other, which is the "universal" part.

**When it fails.** A distribution match says nothing about **which** point goes where. The generator can map every GTE vector to a plausible but wrong GTR vector and still fool $D_2$. That is why the next three terms exist.

### 4.2 Reconstruction and cycle consistency: do no harm on the round trip

Reconstruction says entering the latent and leaving through your own adapter should return what you started with:

$$
\mathcal{L}_{\text{rec}}(R_1, R_2) = \mathbb{E}_{x \sim p}\,\lVert R_1(x) - x \rVert_2^2 + \mathbb{E}_{y \sim q}\,\lVert R_2(y) - y \rVert_2^2
$$

Here $p$ and $q$ are the distributions of embeddings from $M_1$ and $M_2$. Cycle consistency, borrowed from CycleGAN, says translating to the other space and back should also return the start:

$$
\mathcal{L}_{\text{CC}}(F_1, F_2) = \mathbb{E}_{x \sim p}\,\lVert F_2(F_1(x)) - x \rVert_2^2 + \mathbb{E}_{y \sim q}\,\lVert F_1(F_2(y)) - y \rVert_2^2
$$

**Worked example.** Take $x = (1, 0)$ with $F_2(F_1(x)) = (0.9, 0.3)$. The cycle penalty is $(1 - 0.9)^2 + (0 - 0.3)^2 = 0.10$, pushing both translators toward an invertible pair. This is the paper's "unsupervised proxy for pair alignment": information thrown away on the way out cannot come back on the return leg.

### 4.3 Vector space preservation: keep the shape

**Problem.** Cycle consistency is satisfied by any invertible map, including ones that scramble neighborhoods. We want the translation to keep **which texts are close to which**.

**Mechanism.** Vector space preservation (VSP) compares, within a batch, the matrix of pairwise dot products before and after translation. The paper's printed equation sums over a batch of $B$ embeddings from each side; I read its intent (consistent with the cited prior work) as:

$$
\mathcal{L}_{\text{VSP}} = \frac{1}{B^{2}} \sum_{i=1}^{B} \sum_{j=1}^{B} \Big[ \big(x_i \cdot x_j - F_1(x_i) \cdot F_1(x_j)\big)^2 + \big(y_i \cdot y_j - F_2(y_i) \cdot F_2(y_j)\big)^2 \Big]
$$

$x_i$ are embeddings from $M_1$, $y_i$ from $M_2$, and the dot is the inner product. (The printed equation pairs the $M_1$ Gram matrix with translations of the $M_2$ batch, which looks like an index slip; the reading above is my interpretation, not the paper's literal text.)

**Worked example.** Two unit vectors in space 1, $x_1 = (1, 0)$ and $x_2 = (0.6, 0.8)$, have $x_1 \cdot x_2 = 0.6$. If their translations come out with dot product 0.2, that pair contributes $(0.6 - 0.2)^2 = 0.16$. Pull the translations toward a 0.6 dot product and the penalty vanishes. Multiply that by every pair in a batch and you are matching the shape of the cloud, which is exactly the cartographer's trick from Section 2.

The generator term combines all three, each with its own weight:

$$
\mathcal{L}_{\text{gen}}(\theta) = \lambda_{\text{rec}}\,\mathcal{L}_{\text{rec}} + \lambda_{\text{CC}}\,\mathcal{L}_{\text{CC}} + \lambda_{\text{VSP}}\,\mathcal{L}_{\text{VSP}}
$$

| Loss | What it forbids | What it alone cannot stop |
| --- | --- | --- |
| Output GAN | Translations that look nothing like target embeddings | Right distribution, wrong points |
| Latent GAN | Two spaces using separate regions of the latent | Shared latent with scrambled content |
| Reconstruction | Adapters that destroy information | Translation that is consistent but wrong |
| Cycle consistency | Lossy translators | Invertible maps that scramble neighborhoods |
| VSP | Neighborhood scrambling | Global rotation (harmless) |

## 5. How a translation is scored

The ideal translator puts $F(u_i)$ closer to $v_i$ than to any other target. With $r_i$ defined as the rank of the true $v_i$ when all candidate targets are sorted by cosine similarity to $F(u_i)$, the paper reports three numbers:

$$
\overline{\cos} = \frac{1}{n}\sum_{i=1}^{n} \cos\big(F(u_i), v_i\big), \qquad \text{Top-1} = \frac{1}{n}\sum_{i=1}^{n} \mathbb{1}\{r_i = 1\}, \qquad \text{Rank} = \frac{1}{n}\sum_{i=1}^{n} r_i
$$

$\mathbb{1}\{\cdot\}$ is 1 when the condition holds and 0 otherwise. A perfect translator scores 1.0 on all three. A random one has mean rank about $n/2$: with batches of 8,192, that is about 4,096.

**Worked example.** Four documents. Translations rank their true targets at positions 1, 1, 3 and 1. Then Top-1 is ${3/4} = 0.75$ and Rank is $(1+1+3+1)/4 = 1.5$. A translator can have mediocre cosine and still nail Top-1, because ranking only needs the right target to be **closer than the others**, not close in absolute terms.

There is no prior unsupervised translator, so the baselines are deliberately unusual:

| Baseline | What it does | Why it is there |
| --- | --- | --- |
| Naive | $F(x) = x$ | Measures how similar the raw spaces already are |
| Oracle-aided optimal transport | Solves an assignment between $u$'s and the **true candidate** $v$'s (Hungarian, EMD, Sinkhorn, or Gromov-Wasserstein; best solver per experiment) | Strictly easier than vec2vec, since it knows the targets exist |

## 6. Results: it translates, and the latent really is shared

### 6.1 In distribution

Table 2 is the headline. Five text encoders form the main grid (GTR on T5; E5, GTE and Stella on BERT; Granite on RoBERTa), with CLIP and Qwen3 tested separately. Every vec2vec is trained on NQ and evaluated on 65,536 held-out NQ texts in chunks of 8,192.

![Table 2 from Jha et al. (2025): in-distribution translation results for every model pair, vec2vec against the naive and oracle optimal transport baselines](/imgs/blogs/harnessing-the-universal-geometry-of-embeddings-fig4.webp)

The pattern splits cleanly by backbone:

| Pair type | Example (M1 to M2) | vec2vec cos / Top-1 / Rank | Naive cos / Top-1 / Rank | OT cos / Top-1 / Rank |
| --- | --- | --- | --- | --- |
| Cross-backbone | granite to gtr | 0.80 / 0.99 / 1.19 | -0.03 / 0.00 / 4168.73 | 0.70 / 0.00 / 2773.72 |
| Cross-backbone | gte to gtr | 0.75 / 0.91 / 2.64 | 0.04 / 0.00 / 4084.15 | 0.70 / 0.00 / 3064.16 |
| Cross-backbone | stella to gtr | 0.82 / 1.00 / 1.10 | 0.00 / 0.00 / 4192.31 | 0.70 / 0.00 / 3555.64 |
| Same backbone (BERT) | stella to gte | 0.92 / 1.00 / 1.00 | 0.56 / 1.00 / 1.00 | 0.87 / 1.00 / 1.00 |
| Same backbone (BERT) | gte to e5 | 0.87 / 0.99 / 5.19 | 0.68 / 1.00 / 1.00 | 0.84 / 1.00 / 1.00 |

Across backbones, naive and oracle OT are at chance on rank while vec2vec hits 82 to 100 percent top-1. Within the BERT family the raw spaces are already compatible (naive top-1 of 1.00), so vec2vec mostly improves cosine; gte to e5 even has a worse mean rank (5.19) than doing nothing (1.00).

The intro quotes cosine "as high as 0.96", but the best output-space cosine in Table 2 is 0.92; 0.96 is a latent similarity from Figure 4.

### 6.2 Inside the latent

Figure 4 makes the Strong Platonic claim visible. The left heatmap shows cosine between raw embeddings of the same 1,024 NQ texts across model pairs: between -0.02 and 0.03 for every cross-backbone pair and 0.38 to 0.68 within BERT models. The middle heatmap shows the same pairs after the adapters and backbone: 0.75 to 0.96 everywhere.

![Figure 4 from Jha et al. (2025): pairwise cosine of input embeddings (left), their vec2vec latents (middle), and the absolute difference (right), on the same 1,024 NQ texts](/imgs/blogs/harnessing-the-universal-geometry-of-embeddings-fig3.webp)

That is the strongest single piece of evidence in the paper.

### 6.3 Out of distribution, other modalities, bigger models

| Test | Best case | Worst case | Read |
| --- | --- | --- | --- |
| TweetTopic (800 tweets, trained on Wikipedia NQ) | stella to gte: cos 0.90, Top-1 1.00 | gtr to e5: Top-1 0.53, rank 13.38 of 800 | Holds up on emoji-heavy colloquial text |
| MIMIC (8,192 medical records) | stella to gte: Top-1 1.00 | gtr to e5: Top-1 0.01, rank 1413.80 | Cosine stays 0.66 to 0.91, but exact matching degrades badly for many pairs |
| CLIP (multimodal, 512-d) | clip to granite: Top-1 0.72, rank 4.46 | gte to clip: Top-1 0.00, rank 3233.41 | Uneven; still beats Gromov-Wasserstein OT on rank in every row |
| Qwen3 embedding (4B params, 2560-d) | gte to qwen: Top-1 0.92, rank 2.28 of 1024 | qwen to gte: cos 0.84 vs OT 0.85 | Works across a 37x size gap, one pair only |

MIMIC is the case I would weigh most. High cosine with top-1 of 0.08 to 0.27 for several pairs means translations land in the right neighborhood without pinning the specific record: often enough for an attacker, not enough for a precise interoperability layer.

## 7. Reading secrets out of translated vectors

Translation is only interesting if meaning survives. The paper tests this two ways, both **zero-shot**: no model is trained on the target task.

**Attribute inference.** Embed every candidate label with $M_2$, compare against $F(M_1(d_i))$, and check whether one of the top-$k$ labels is correct. On TweetTopic (19 topics, $k = 1$), vec2vec reaches 0.24 to 0.38 and often matches or beats the "ideal" baseline that embeds documents and labels with the same model (for instance gte to granite: 0.37 versus 0.34 and 0.30). On MIMIC (2,673 MedCAT disease labels, $k = 10$), cross-backbone vec2vec beats naive (for example granite to gte, 0.36 versus 0.13) but sits well below the same-model ideal of 0.76 to 1.00.

Concepts like "alveolar periostitis" survive translation despite never appearing in NQ, the best qualitative argument that the latent is not a memorized lookup of training text.

**Inversion.** The translated vectors are fed to an off-the-shelf zero-shot inverter (Zhang et al. 2025), and GPT-4o judges whether the reconstruction leaks information about the original. On a 50-email Enron subset, leakage ranges from 20 percent to 80 percent depending on the pair.

![Figure 5 from Jha et al. (2025): percentage of Enron emails whose inversion from translated embeddings leaks information, by source and target model, judged by GPT-4o](/imgs/blogs/harnessing-the-universal-geometry-of-embeddings-fig5.webp)

"Subject: Enron Bashing on Frontline" comes back as "Some emails discussing NROn Employee/s Complaint To thePublic". On tweets, leakage reaches 67 percent. The reconstructions are crude, but names, dates and topics survive.

## 8. Ablations: every piece is load-bearing

The ablation in Table 6 removes one component at a time from the gte to gtr translator. Every removal drops top-1 from 0.91 to exactly 0.00 and rank back to chance.

![Ablation of the gte to gtr translator (Table 6): removing any single loss, the latent GAN, or the hyperparameter tuning collapses top-1 to zero, even where cosine stays near the OT baseline](/imgs/blogs/harnessing-the-universal-geometry-of-embeddings-3.webp)

The gap between cosine and top-1 is the interesting part. Without VSP, cosine is still 0.58 (naive: 0.04), so translations point at the right region, but they lose the fine structure that separates one text from its 8,191 neighbors.

Data requirements show a sharp threshold. Holding GTR fixed at 1M samples and shrinking the GTE side:

![Top-1 accuracy of the gte to gtr translator versus the number of GTE training embeddings (Table 7): a cliff between 10K and 50K, then a plateau](/imgs/blogs/harnessing-the-universal-geometry-of-embeddings-4.webp)

At 10K embeddings, top-1 is 0.01 and rank 1462.21: better than random, useless in practice. At 50K, top-1 jumps to 0.81 and rank to 3.91. From 500K on, it plateaus at 0.92. A practical reading (mine, not the paper's): a database with tens of thousands of leaked vectors is already enough raw material.

**Seed instability** is the elephant in the room. On fifteen seeds trained for a fixed 10 epochs, the same-backbone pair e5 to gte reached at least 80 percent top-1 on 14 runs (the last reached 72 percent). The cross-backbone pair e5 to gtr converged on only 3 of 15. The main results report the best of several initializations, which is a selection step the reader should keep in mind.

Cost is not trivial either: full models took 1 to 7 days each, the Qwen pair took 20 days on an A100, and the paper totals about 176 GPU days of training.

## 9. Critique

**What is strong.** The formulation is clean and new: generate the target vector on disjoint texts, not just match it. Figure 4 is direct evidence of a shared latent, and an oracle that knows the candidates still fails across backbones, which makes vec2vec's result hard to dismiss.

**What is weak.** The training procedure is fragile and the paper reports best-of-seeds. Model coverage is narrow: five main text encoders, four of them sharing BERT-family tokenization habits, plus T5-based GTR as the lone true outsider in the main grid. "Universal" is a big word for that sample. The VSP equation as printed does not quite match its description, which matters for anyone reimplementing it.

**What is missing.** There is no ablation on the latent dimension $Z$, the loss weights, or the depth of the adapters, and no study of how translation quality tracks model quality or size beyond the single Qwen pair. The MIMIC gap also suggests domain match carries more of the load than the headline numbers imply.

**Extrapolation flagged as mine.** The defensive implication is sturdier than the scientific one. Even if universality is partial, a 50K-sample threshold and 80 percent email leakage make embeddings **sensitive data**, on par with their text.

**What would change my mind.** I would stop believing the Strong Platonic claim if (a) vec2vec failed on a pair of genuinely unrelated modern encoders trained with different objectives (say a decoder-only embedding model versus a retrieval-tuned bidirectional one) after a fair hyperparameter search, or (b) a multi-seed report showed that cross-backbone success depends on lucky initializations more than on data or capacity. I would also want to see the latent similarity in Figure 4 drop sharply on MIMIC-style text; if it held at 0.75 or above there, the universality story gets much stronger.

## 10. What to do with this

If you run a vector database, classify embeddings at the same sensitivity as the source documents: not knowing the encoder is no longer protection. If you work on interpretability, shared structure across models is now learnable, not just measurable, which is the same premise behind [universal sparse autoencoders](/blog/paper-reading/ai-interpretability/universal-sparse-autoencoders-interpretable-cross-model-concept-alignment) and close kin to the [semantic hub hypothesis](/blog/paper-reading/ai-interpretability/the-semantic-hub-hypothesis-language-models-share-semantic-representations-across-languages-and-modalities). An embedding is not a private code. It is a coordinate system on a map everyone is drawing, and given enough points, anyone can line their map up with yours.

## References

1. Rishi Jha, Collin Zhang, Vitaly Shmatikov, John X. Morris. "Harnessing the Universal Geometry of Embeddings." NeurIPS 2025. [arXiv:2505.12540v4](https://arxiv.org/abs/2505.12540v4)
2. Minyoung Huh, Brian Cheung, Tongzhou Wang, Phillip Isola. "The Platonic Representation Hypothesis." 2024. [arXiv:2405.07987](https://arxiv.org/abs/2405.07987)
3. John X. Morris, Volodymyr Kuleshov, Vitaly Shmatikov, Alexander M. Rush. "Text Embeddings Reveal (Almost) As Much As Text." 2023. [arXiv:2310.06816](https://arxiv.org/abs/2310.06816)
4. Collin Zhang, John X. Morris, Vitaly Shmatikov. "Universal Zero-shot Embedding Inversion." 2025. [arXiv:2504.00147](https://arxiv.org/abs/2504.00147)
5. Jun-Yan Zhu, Taesung Park, Phillip Isola, Alexei A. Efros. "Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks." [arXiv:1703.10593](https://arxiv.org/abs/1703.10593)
6. Related posts on this site: [The Platonic Representation Hypothesis](/blog/paper-reading/ai-interpretability/position-the-platonic-representation-hypothesis), [Universal Sparse Autoencoders](/blog/paper-reading/ai-interpretability/universal-sparse-autoencoders-interpretable-cross-model-concept-alignment), [The Semantic Hub Hypothesis](/blog/paper-reading/ai-interpretability/the-semantic-hub-hypothesis-language-models-share-semantic-representations-across-languages-and-modalities).
