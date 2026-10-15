---
title: "Universal Sparse Autoencoders: One Concept Dictionary for Many Vision Models"
date: "2026-10-15"
publishDate: "2026-10-15"
description: "How a single sparse autoencoder with one encoder and one decoder per model learns a shared concept space across SigLIP, DinoV2 and ViT, and what its universality metrics actually measure."
tags: ["paper-reading", "universal-sae", "sparse-autoencoders", "cross-model-alignment", "concept-alignment", "mechanistic-interpretability", "dictionary-learning", "vision-transformers", "activation-maximization", "icml-2025-poster"]
category: "paper-reading"
subcategory: "AI Interpretability"
author: "Hiep Tran"
featured: true
readTime: 18
paper:
  title: "Universal Sparse Autoencoders: Interpretable Cross-Model Concept Alignment"
  authors: "Harrish Thasarathan, Julian Forsyth, Thomas Fel, Matthew Kowal, Konstantinos G. Derpanis"
  venue: "ICML 2025 (PMLR 267)"
  url: "https://arxiv.org/abs/2502.03714"
---

> [!tldr]
> - **What it proposes.** A Universal Sparse Autoencoder (USAE): one sparse concept space shared by several pretrained vision models, with a small encoder and decoder per model.
> - **Key mechanism.** Encode one model's activations into a sparse code, then force that same code to reconstruct *every* model's activations. Concept $k$ then means the same thing for all models by construction.
> - **Why it matters.** Earlier "universal feature" methods mined each model separately and matched concepts after the fact. USAE gets alignment from gradient descent in one training run.
> - **Most surprising result.** Universality and importance go together: concepts that co-fire across models are the ones that carry the most reconstruction energy ($r = 0.89$ among frequently co-firing concepts).
> - **Where it is thin.** Three models, last layer only, ImageNet only for training, and most evidence of "shared meaning" is qualitative. Only 23–38% of USAE concepts have a close match in an independently trained SAE.

Every interpretability paper you have read probably dissected one model. You train a sparse autoencoder (SAE) on its activations, you look at the features, you name them.

This paper asks a sharper question. If SigLIP, DinoV2 and a supervised ViT all learned "yellow" and "dog face", can we find those concepts **once**, in one dictionary, with a shared index?

## The problem: concepts that do not line up

An SAE takes a model's activation vector and rewrites it as a sparse combination of learned directions, which we hope are human-readable concepts. If you want background on why neurons are not already readable, the short version is [superposition](/blog/machine-learning/ai-interpretability/what-is-superposition): networks pack more features than dimensions, and SAEs try to unpack them.

The trouble starts when you compare models. Prior "universal feature" methods (Rosetta Neurons, Rosetta Concepts) extract concepts from each model separately, then match them by correlation or optimization. That is compute-heavy and gives no way to *translate* between models.

There is also a theoretical pull. The [Platonic representation hypothesis](/blog/paper-reading/ai-interpretability/position-the-platonic-representation-hypothesis) argues that large models converge toward a shared picture of the world. If that is even partly true, a shared concept space should exist, and we should be able to learn it directly.

![One SAE per model vs one shared concept space](/imgs/blogs/universal-sparse-autoencoders-interpretable-cross-model-concept-alignment-1.webp)

The figure above is the whole idea in one picture. On the left, three independent SAEs give three unaligned dictionaries. On the right, every model passes through one sparse code $Z$, so concept index $k$ is shared before anyone looks at a single heatmap.

## Contributions

1. **USAE architecture and objective.** Per-model encoders and decoders around one shared sparse code, trained so any model's code reconstructs every model.
2. **A cheap training schedule.** One encoder per step, all decoders in the loss, one optimizer step.
3. **Two universality metrics.** Firing entropy and co-fire proportion, which score how evenly a concept fires across models.
4. **An analysis of shared versus unique concepts** across SigLIP, DinoV2 and ViT, including a link between universality and importance.
5. **Coordinated activation maximization.** Synthesize one image per model that lights up the same universal concept, so you can compare how each model "sees" it.

## Method

### Technique 1: the TopK sparse autoencoder (the building block)

**The problem.** A model's activation is a dense vector where single dimensions mix many unrelated ideas. We want a rewrite where each active unit means one thing.

**Analogy.** Think of a paint mixer with 6,144 tubes of pigment. Any color you see on the wall (an activation) must be recreated using at most $K$ tubes. Forced to use few tubes, the mixer learns pigments that are pure and reusable, which is exactly what we want a concept to be.

**Mechanism.** Subtract a learned bias from the activation, project it up to a much wider space, keep only the $K$ largest entries and zero the rest. Then multiply that sparse vector by a dictionary to rebuild the activation. Training minimizes the rebuild error.

**Math.** Let $A \in \mathbb{R}^{n \times d}$ hold $n$ activation vectors of width $d$ (one row per image patch). The encoder is

$$
Z = \Psi_\theta(A) = \mathrm{TopK}\big(W_{\text{enc}}(A - b_{\text{pre}})\big), \qquad \hat{A} = Z D
$$

where $W_{\text{enc}}$ projects to $m$ dimensions, $b_{\text{pre}} \in \mathbb{R}^{d}$ is a learned centering bias, $Z \in \mathbb{R}^{n \times m}$ is the sparse code, and $D \in \mathbb{R}^{m \times d}$ is the dictionary whose row $D_k$ is concept $k$'s direction. $\mathrm{TopK}$ keeps the $K$ largest entries of each row of $Z$, so every patch uses at most $K$ concepts. When $m > d$ the dictionary is **overcomplete**; here $m = 6144$. The single-model loss is the Frobenius norm of the residual, $\mathcal{L}_{\text{SAE}} = \lVert A - ZD \rVert_F$.

**Tiny example.** Take $d = 2$, $m = 4$, $K = 1$. Suppose a patch has activation $a = [2.0, 0.1]$ and the pre-activations come out as $[0.2, 0.0, 1.5, 0.4]$. TopK keeps only index 3, giving $z = [0, 0, 1.5, 0]$. If $D_3 = [1.3, 0.0]$, the rebuild is $[1.95, 0.0]$, close to $a$ with one concept.

**When it fails.** Too small a $K$ and reconstruction suffers; too large and units go polysemantic again. SAEs also split and absorb features in ways that make a single unit's meaning slippery, a failure covered in the post on [scaling and evaluating sparse autoencoders](/blog/paper-reading/ai-interpretability/scaling-and-evaluating-sparse-autoencoders), which introduced this TopK variant.

### Technique 2: the universal objective

**The problem.** Train three SAEs separately and you get three codes with no shared indexing. We need a training signal that *forces* index $k$ to mean the same thing everywhere.

**Analogy.** Picture a translation bureau with one shared shorthand. A French clerk writes notes in shorthand, and the notes must be readable by the German, Japanese and French typists, who each expand them back into their own language. A note that only the French typist can expand is a bad note. Over time the shorthand converges to symbols that mean the same thing to everyone.

**Mechanism.** Each model $i$ gets its own encoder $\Psi^{(i)}_\theta$ and its own decoder $D^{(j)}$, because widths differ and each model's geometry is its own. All encoders write into the same code space $Z$. During training, a code produced from model $i$ is decoded by **every** model's decoder, and every reconstruction error counts.

![Figure 2 from Thasarathan et al. (2025): USAE training process](/imgs/blogs/universal-sparse-autoencoders-interpretable-cross-model-concept-alignment-fig2.webp)

In the authors' Figure 2 above, the encoder of model $i$ produces $Z$, and the fan of arrows on the right is the decoders $D^{(j)}$ each rebuilding their own model's activations $\hat{A}^{(j)}$. That fan is the alignment pressure.

**Math.** For $M$ models, let $A^{(i)} \in \mathbb{R}^{n \times d_i}$ be model $i$'s activations on the same $n$ inputs. The model-specific encoder is

$$
Z = \Psi^{(i)}_\theta\big(A^{(i)}\big) = \mathrm{TopK}\big(W^{(i)}_{\text{enc}}(A^{(i)} - b^{(i)}_{\text{pre}})\big)
$$

and any model $j$ is reconstructed from that same code as $\hat{A}^{(j)} = Z D^{(j)}$, with $D^{(j)}$ of shape $m \times d_j$. (The paper writes $D^{(j)} \in \mathbb{R}^{d_j \times m}$ in Eq. 4, but $ZD^{(j)}$ only type-checks as $m \times d_j$; I use the version that multiplies.) The loss for a step that encodes model $i$ is

$$
\mathcal{L}_{\text{Universal}} = \sum_{j=1}^{M} \big\lVert A^{(j)} - \Psi^{(i)}_\theta(A^{(i)})\, D^{(j)} \big\rVert_F
$$

Each term asks: can model $i$'s concepts rebuild model $j$'s view of the same patch? The diagonal term $j = i$ is ordinary SAE reconstruction. The off-diagonal terms are what make the code universal.

**Tiny example.** Extend the paint mixer. Model 1 sees a yellow patch as $a^{(1)} = [2.0, 0.1]$, model 2 sees it as $a^{(2)} = [0.0, 3.0]$. Encoder 1 produces $z = [0, 0, 1.5, 0]$ as before. If $D^{(1)}_3 = [1.3, 0.0]$ and $D^{(2)}_3 = [0.0, 2.0]$, then $\hat{a}^{(1)} = [1.95, 0.0]$ and $\hat{a}^{(2)} = [0.0, 3.0]$. One concept index, two models, both rebuilt: concept 3 is "yellow" for both.

**When it fails.** The objective rewards concepts *every* model can rebuild. A concept only DinoV2 has will cost reconstruction on the other decoders, so the objective can push model-specific detail out, or leave it in units that fire for one model only. The paper turns that second outcome into a feature (Technique 5), but it also explains why only part of each independent SAE's dictionary reappears.

### Technique 3: the one-encoder, all-decoders training schedule

**The problem.** The naive version encodes and decodes all $M$ models every step, which is $M^2$ reconstructions and a lot of memory. The cheap version samples one random pair $(i, j)$ per step, which the authors report converges slowly.

**Mechanism.** The paper picks the middle: sample one model $i$, encode its batch once, decode through all $M$ decoders, sum the losses, backprop, and step only optimizer $i$. Each encoder-decoder pair has its own Adam optimizer, so encoders and decoders receive an equal number of updates over time.

![One USAE training step](/imgs/blogs/universal-sparse-autoencoders-interpretable-cross-model-concept-alignment-2.webp)

The paper's own pseudocode (its Figure 3) reduces to a few lines. Here it is in PyTorch shape:

```python
def train_usae(encoders, decoders, acts, steps, opts):
    M = len(encoders)
    for t in range(steps):
        i = random.randrange(M)               # pick one model
        Z = encoders[i](acts[i].next_batch()) # shared sparse code, n x m
        loss = 0.0
        for j in range(M):                    # decode with EVERY model
            A_hat = Z @ decoders[j]           # n x d_j
            loss = loss + (acts[j].same_batch() - A_hat).norm(p="fro")
        loss.backward()
        opts[i].step()                        # only pair i moves
```

**Why it works.** The gradient still flows from every decoder's error back into encoder $i$, so the code learns to serve all models. Only pair $i$ steps, which keeps per-step cost at one encoder pass plus $M$ cheap matrix products. Note that `acts[j].same_batch()` must be activations of model $j$ on the *same images*: paired data is the whole trick.

**Implementation details that matter** (from Appendix A.1):

| Setting | Value |
| --- | --- |
| Models | SigLIP, DinoV2, ViT (ImageNet-supervised), all via timm |
| Layer | Final layer, patch tokens; class tokens dropped |
| Dictionary | 6,144 = 8 × 768 (expansion 8 over the widest model) |
| Encoder | Linear, then BatchNorm, then ReLU (plus TopK) |
| Loss | ℓ1 instead of ℓ2, which the authors found trained and read better |
| Optimizer | Adam per pair, lr 3e-4 decaying to 1e-6, cosine with warmup |
| Scaling | Per-model standardization from 1,000 random training samples |
| Compute | One NVIDIA RTX 6000, about three days |

Two practical notes. DinoV2's token grid is interpolated to a 16 × 16 pixel patch size so every model has the same patches. And per-model standardization matters: without it, the model with the largest activation scale would dominate the summed loss.

### Technique 4: measuring universality and importance

Once trained, how do we know a concept is actually shared rather than just reconstructable? The paper uses four measurements, and each is simple once the symbols are pinned down.

**Cross-model R².** The coefficient of determination compares reconstruction error to the variance of the data:

$$
R^2 = 1 - \frac{\lVert A - \hat{A} \rVert_F^2}{\lVert A - \bar{A} \rVert_F^2}
$$

where $\bar{A}$ is the mean activation. $R^2 = 1$ is perfect; $R^2 = 0$ means you did no better than predicting the mean. Encoding model $i$ and decoding with $D^{(j)}$ fills one cell $(i, j)$ of a 3 × 3 matrix. Tiny example: if total variance is 10 and residual error is 4, $R^2 = 1 - 4/10 = 0.6$.

**Energy (importance).** A concept matters if it contributes a lot to reconstruction on average:

$$
\mathrm{Energy}(k) = \big\lVert \mathbb{E}_x[Z_k(x)]\, D_k \big\rVert_2^2
$$

where $Z_k(x)$ is concept $k$'s activation on input $x$ and $D_k$ is its dictionary row. If concept $k$ averages 0.5 and its dictionary row has norm 2, its energy is $(0.5 \times 2)^2 = 1.0$.

**Firing entropy (FE).** Pick a threshold $\tau$. Concept $k$ "fires" for model $i$ on patch $x$ when $Z^{(i)}_k(x) > \tau$, giving the fire set $F^{(i)}_k$. Let $p^{(i)}_k = |F^{(i)}_k| / \sum_{j} |F^{(j)}_k|$ be model $i$'s share of all fires. Then

$$
\mathrm{FE}_k = -\frac{1}{\log M}\sum_{i=1}^{M} p^{(i)}_k \log p^{(i)}_k
$$

Dividing by $\log M$ maps it to $[0, 1]$. FE near 1 means the concept fires evenly across models; FE near 0 means one model owns it.

**Co-fire proportion (CFP).** A co-fire is a patch where the concept fires in *all* models at once: $C_k = \{x : \min_i Z^{(i)}_k(x) > \tau\}$. For each model, $\mathrm{CFP}^{(i)}_k = |C_k| / |F^{(i)}_k|$: of the times model $i$ detects the concept, how often do the others agree on the same patch?

![Scoring universality: firing entropy and co-fire proportion](/imgs/blogs/universal-sparse-autoencoders-interpretable-cross-model-concept-alignment-3.webp)

**Worked example** (my toy numbers, same formulas). Concept A fires 6 times in SigLIP, 4 in DinoV2, 5 in ViT, 15 total, and 3 of those patches are co-fires. Shares are 0.40, 0.27, 0.33. The entropy sum is ${0.40} \cdot 0.92 + 0.27 \cdot 1.32 + 0.33 \cdot 1.10 \approx 1.085$; divided by $\log 3 \approx 1.099$ that gives FE ≈ 0.99. CFP is 3/6 = 0.50 for SigLIP, 3/4 = 0.75 for DinoV2 and 3/5 = 0.60 for ViT. Concept B fires 8 times in DinoV2 and never elsewhere: FE = 0, CFP = 0.

**When it fails.** FE only looks at fire *counts*, so a concept could fire equally often in all three models but on different patches; that is why CFP is needed alongside it. Both depend on $\tau$, which the paper does not report in the main text. Treat the absolute numbers as threshold-relative.

### Technique 5: unique concepts as a by-product

The universal objective has a useful side effect. Since every concept index is shared, you can search for the ones with **low** firing entropy and get concepts that live in one model only. No extra training is needed; you sort by FE and read the bottom of the list.

This is the paper's tool for asking "what does DinoV2 know that the others do not?"

### Technique 6: coordinated activation maximization

**The problem.** Classic activation maximization synthesizes an input that excites one neuron of one model. With a shared concept index, we can do it for all models at once and compare the results side by side.

**Mechanism.** For concept $k$ and each model $i$, define the concept activation $Z^{(i)}_k(x) = [\Psi^{(i)}_\theta(f^{(i)}(x))]_k$, where $f^{(i)}$ is model $i$'s feature extractor. Optimize one image per model:

$$
x^{(i)}_\star = \arg\max_{x} \; Z^{(i)}_k(x) - \lambda R(x)
$$

where $R(x)$ is a naturalness regularizer and $\lambda$ its weight. The paper follows MACO (Fel et al., 2023), which optimizes the image's Fourier phase while keeping its magnitude spectrum fixed. The $M$ optimizations are independent; what ties them together is the shared index $k$.

**Why it is interesting.** Differences between the $M$ images are differences in how each model encodes the *same* concept. The paper reports, for example, that DinoV2's low and mid-level concepts appear at a much larger scale in its optimized images.

![Figure 8 from Thasarathan et al. (2025): coordinated activation maximization](/imgs/blogs/universal-sparse-autoencoders-interpretable-cross-model-concept-alignment-fig8.webp)

Figure 8 above shows six concepts (curves, crosses, objects from above, brown grass, round objects, keypads) with one optimized image per model and dataset exemplars. In the "brown grass" column, some models' images contain birds, which the authors read as animals also driving that concept.

**When it fails.** Activation maximization images are notoriously regularizer-dependent. A difference between two models' images could come from the concept or from how each model responds to MACO's phase-only parameterization; the paper does not separate the two.

## Experiments and results

### Qualitative: what universal concepts look like

![Figure 4 from Thasarathan et al. (2025): universal concepts across ViT, SigLIP and DinoV2](/imgs/blogs/universal-sparse-autoencoders-interpretable-cross-model-concept-alignment-fig4.webp)

Figure 4 shows eight concepts picked from the top 100 by energy, with heatmaps per model. They span levels of abstraction: colors (yellow, blue), foreground-background (bird against sky), thin wiry objects, parts (dog face excluding the eyes, animal jaw, bolts across metal and rubber), and groupings (faces of animals in a group).

### Cross-model reconstruction

![Figure 5 from Thasarathan et al. (2025): cross-model R² matrix](/imgs/blogs/universal-sparse-autoencoders-interpretable-cross-model-concept-alignment-fig5.webp)

Reading Figure 5 as a table (rows encode, columns decode):

| Encode \ Decode | SigLIP | DinoV2 | ViT |
| --- | --- | --- | --- |
| SigLIP | **0.61** | 0.36 | 0.25 |
| DinoV2 | 0.31 | **0.77** | 0.31 |
| ViT | 0.30 | 0.45 | **0.59** |

The diagonal is self-reconstruction and is highest. Independent SAEs reach about 0.8 R² on DinoV2 and 0.7 on SigLIP and ViT, so universality costs 0.03 to 0.11 points (my subtraction). Every off-diagonal cell is positive, meaning a code built from one model explains 25–45% of another model's variance.

### Universality and importance

![Figure 6 from Thasarathan et al. (2025): firing entropy, co-firing and energy](/imgs/blogs/universal-sparse-autoencoders-interpretable-cross-model-concept-alignment-fig6.webp)

The three panels of Figure 6 carry the paper's main quantitative claims:

- **(a) Firing entropy is bimodal.** Over 1,000 concepts sit at the maximum entropy of 1, and a second peak sits near 0.6 (concepts favoring two of three models). Few concepts land in 0.0–0.2, so truly model-specific concepts are rare.
- **(b) Co-firing decays smoothly.** Among the top 1,000 concepts by energy, roughly the first 100 co-fire above 0.5. Mean CFP is 0.344 for SigLIP, 0.326 for ViT and 0.266 for DinoV2 (overall 0.312).
- **(c) Universality predicts importance.** Across all concepts, co-fire frequency and energy correlate at $r = 0.63$ (slope 0.23). Restricted to concepts with at least 1,000 co-fires, $r = 0.89$ (slope 0.70).

DinoV2 is the odd one out in both the CFP numbers and its higher self-reconstruction. The authors attribute this to its self-distillation and masked-token objectives, which is plausible but not tested directly.

### Consistency with independent SAEs

Do USAE concepts exist in a normal SAE? The authors match decoder rows between each model's independent SAE and the USAE using cosine similarity and the Hungarian algorithm, against a baseline of random vectors drawn with each dictionary's mean and variance.

| Model | Share of USAE concepts with cosine > 0.5 | AUC over thresholds | AUC, top 1,000 co-firing concepts |
| --- | --- | --- | --- |
| ViT | 38% | 0.41 | 0.52 |
| DinoV2 | 26% | 0.36 | 0.65 |
| SigLIP | 23% | 0.30 | 0.40 |
| Random baseline | 0% | 0.13 | 0.13 |

Two readings. First, matches are far above chance, so universal training is not inventing an unrelated basis. Second, most USAE concepts have **no** close twin in the independent SAE, which the authors read as universal training surfacing shared concepts single-model training misses. The top-1,000 column (Appendix A.5.1) shows the most universal concepts match better, with DinoV2 jumping to the top.

### Unique concepts and out-of-distribution checks

Sorting by low firing entropy, the paper finds DinoV2-only concepts tied to geometry: converging perspective lines, image tilt left and right, depth cues, the top face of rectangular prisms, corners and dome brims. SigLIP-only concepts mix image and text: a "star" concept that fires on star shapes and on the written word, a capital "C" concept, and a music concept covering instruments and notes.

Appendix Table 1 applies the ImageNet-trained USAE to two unseen datasets:

| Model | R² ImageNet | R² DTD | R² CelebA |
| --- | --- | --- | --- |
| SigLIP | 0.61 | 0.54 | 0.52 |
| DinoV2 | 0.77 | 0.69 | 0.75 |
| ViT | 0.59 | 0.46 | 0.45 |

Reconstruction degrades but holds, and the co-firing trends repeat on both datasets.

**What is load-bearing and may not transfer.** All three backbones are ViT-style models at width 768 with 16-pixel patches after interpolation; matching patches across models is easy here and would not be for a CNN versus a transformer. Training is on one dataset, at one layer, with three models. The authors also note hyperparameter sensitivity grows as models are added.

## Critique

**What is strong.** The idea is clean and cheap: a few extra decoders and one summed loss give you a shared index that post-hoc methods needed expensive matching for. The training schedule is a sensible engineering compromise, and the FE plus CFP pair is a simple, reusable way to score universality.

**What is weak.**

- **Alignment is partly circular.** The objective forces concept $k$ to be reconstructable from every model, then high co-firing is presented as evidence of universality. Some co-firing is guaranteed by the training signal; the paper never measures how much co-firing a USAE produces on models that share *nothing*, for example a randomly initialized network.
- **The energy correlation may be partly mechanical.** A concept that fires often in all models will tend to have high average activation, and Energy uses the average activation. The jump from $r = 0.63$ to $r = 0.89$ after filtering to frequent co-firers is consistent with that.
- **Missing numbers.** The firing threshold $\tau$ and the TopK $K$ are not in the main text, and both shift every FE and CFP value.
- **Qualitative selection.** Concept visualizations are chosen from the top 100 by energy, and the limitations admit some concepts are uninterpretable. There is no human-rated interpretability score comparing USAE concepts to independent SAE concepts.
- **Small notation slips.** Beyond the decoder shape, the TopK constraint is written "for all $i \in [m]$" where it should range over the $n$ rows, and the appendix swaps the ℓ2 loss for ℓ1 while the main text equations show a Frobenius norm. None changes the method, but they matter if you reimplement it.

**Missing ablations.** I would want: a random-network control as a fourth model; USAE versus independent SAEs plus Hungarian matching on the same R² and CFP metrics; and the effect of $K$ and dictionary size on the share of universal concepts.

**What would change my mind.** If a USAE trained with a random-weight network as one of the models still showed a strong bimodal firing-entropy histogram and high co-firing with it, I would conclude the metrics mostly measure the objective, not shared structure in the models. Conversely, if a USAE trained on a CNN plus a ViT, with patch alignment handled carefully, reproduced the 0.6 to 0.9 universality-importance correlation, I would take universality as a real property of trained vision models.

## What I'd build with this

These are my extrapolations, not the paper's claims.

1. **Concept transfer for steering.** Find a concept in a well-studied model, then use the shared index to edit the same concept in another model through its decoder $D^{(j)}$. The translation property is the most under-explored part of the paper.
2. **Layer-wise universality.** Train USAEs at several depths and plot where shared concepts appear first. The limitations section names this as future work.
3. **Model diffing.** Use low-entropy concepts as a diff between two checkpoints or two vendors' models, which is close in spirit to the crosscoder work the paper cites.

## References

- Thasarathan, Forsyth, Fel, Kowal, Derpanis. *Universal Sparse Autoencoders: Interpretable Cross-Model Concept Alignment.* ICML 2025. [arXiv:2502.03714](https://arxiv.org/abs/2502.03714). Project page: [yorkucvil.github.io/UniversalSAE](https://yorkucvil.github.io/UniversalSAE).
- Gao et al. *Scaling and Evaluating Sparse Autoencoders* (the TopK SAE). Analysis on this site: [Scaling and evaluating sparse autoencoders](/blog/paper-reading/ai-interpretability/scaling-and-evaluating-sparse-autoencoders).
- Huh et al. *The Platonic Representation Hypothesis.* Analysis on this site: [Position: the Platonic representation hypothesis](/blog/paper-reading/ai-interpretability/position-the-platonic-representation-hypothesis).
- Background on why SAEs are needed at all: [What is superposition](/blog/machine-learning/ai-interpretability/what-is-superposition).
