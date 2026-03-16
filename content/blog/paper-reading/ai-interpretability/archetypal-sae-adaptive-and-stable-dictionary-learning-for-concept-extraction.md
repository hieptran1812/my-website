---
title: >-
  Archetypal SAE: Adaptive and Stable Dictionary Learning for Concept Extraction
  in Large Vision Models
publishDate: "2026-03-16"
category: paper-reading
subcategory: AI Interpretability
tags:
  - archetypal-sae
  - sparse-autoencoders
  - dictionary-learning
  - concept-extraction
  - vision-models
  - mechanistic-interpretability
  - icml-2025
date: "2026-03-16"
author: Hiep Tran
featured: false
aiGenerated: true
image: "/imgs/blogs/archetypal-sae-adaptive-and-stable-dictionary-learning-for-concept-extraction-20260316104615.png"
excerpt: >-
  Standard SAEs are unstable — identical training runs produce wildly different
  concept dictionaries. Archetypal SAEs fix this by anchoring dictionary atoms
  to the convex hull of training data, achieving 93% stability while matching
  reconstruction quality.
---

## TLDR

Sparse Autoencoders (SAEs) have become a popular tool for extracting interpretable concepts from neural network activations. But there is a serious, underappreciated problem: **they are unstable**. Train the same SAE twice on the same data, and you get two very different sets of "concepts." This makes it hard to trust SAEs as a reliable interpretability method.

This paper proposes **Archetypal SAEs (A-SAEs)** and their relaxed variant **RA-SAEs**, which constrain dictionary atoms to lie within the **convex hull** of the training data. The key insight is geometric: by anchoring learned concepts to actual data points, the solution space shrinks dramatically, forcing different runs to converge to similar dictionaries. The result is a jump from ~50% to ~93% stability across runs, while maintaining comparable reconstruction quality to standard SAEs. The authors also introduce two new benchmarks — **plausibility** and **identifiability** — for evaluating whether learned dictionaries actually correspond to meaningful concepts.

## Motivation: Why SAE Instability Matters

### The promise of SAEs for interpretability

The core idea behind SAEs is straightforward. Neural networks learn dense, distributed representations where individual neurons don't correspond to individual concepts. SAEs decompose these dense activations into a **sparse, overcomplete dictionary** — a larger set of directions where each activation is explained by only a few active "concepts."

This has been successful in language models (finding features like "the concept of deception" or "French text") and is now being applied to vision models. The dictionary learning objective is:

$$
(Z^*, D^*) = \arg\min \|A - ZD^T\|_F^2
$$

where $A \in \mathbb{R}^{n \times d}$ is the activation matrix ($n$ tokens, $d$ dimensions), $D \in \mathbb{R}^{k \times d}$ is the dictionary of $k$ learned atoms (concepts), and $Z \in \mathbb{R}^{n \times k}$ is the sparse code matrix.

What makes this a powerful framework is that **many concept extraction methods are special cases** of this same objective, differing only in their constraints:

| Method            | Constraint                                                     | What it finds                  |
| ----------------- | -------------------------------------------------------------- | ------------------------------ |
| **K-Means** (ACE) | $Z_i \in \{e_1, \ldots, e_k\}$ (one-hot)                       | Hard cluster assignments       |
| **PCA** (ICE)     | $D^TD = I$ (orthonormality)                                    | Orthogonal variance directions |
| **NMF** (CRAFT)   | $Z \geq 0, D \geq 0$ (non-negativity)                          | Additive parts-based features  |
| **SAE**           | $Z = \Psi_\theta(A), \|Z\|_0 \leq K$ (learned sparse encoding) | Sparse overcomplete concepts   |

This unified view is important because it reveals that the fundamental choice isn't "which algorithm to use" but rather "which constraints to impose." Each constraint makes a different assumption about the structure of concepts in the representation space.

For SAEs specifically, the encoder is a learned neural network:

$$
\Psi_\theta(A) = \sigma(AW_\theta + b)
$$

where $\sigma$ is a sparsity-inducing activation function. The paper experiments with three variants:

- **TopK**: keeps only the $K$ largest activations, zeroing the rest
- **JumpReLU**: applies a learned threshold per feature
- **ReLU**: standard rectified linear unit (the original SAE formulation)

The key advantage of SAEs over classical methods is that the learned encoder $\Psi_\theta$ can capture **nonlinear** relationships between activations and codes, while classical methods are limited to linear projections. This is why SAEs dominate on the reconstruction-sparsity Pareto frontier (Figure 2 in the paper).

### The instability problem

Here's the catch. The authors ran standard TopK SAEs **4 independent times** on **identical data** from DINOv2, varying only the random seed. They measured how consistent the learned dictionaries were using a stability metric based on optimal permutation matching:

$$
\text{Stability}(D, D') = \max_{\Pi \in \mathcal{P}(n)} \frac{1}{n} \text{Tr}(D^T \Pi D')
$$

where $\Pi$ ranges over signed permutation matrices (solved via the Hungarian algorithm) and atoms are normalized to unit $\ell_2$ norm. The signed permutation allows for matching atoms that point in opposite directions (which encode the same concept with flipped polarity).

The result is striking: **TopK SAEs achieve only ~0.50 stability** on DINOv2. This means roughly half of the "concepts" discovered in one run don't have a clear counterpart in another run. For comparison, classical methods like Semi-NMF achieve ~0.93 stability — but at the cost of much worse reconstruction.

Figure 1 of the paper makes this visceral. Consider the concept of "rabbit": in Run 1, a TopK SAE finds a dictionary atom that activates on rabbit images. In Run 2, the closest matching atom has only 0.58 cosine similarity — meaning the "rabbit concept" is substantially different between the two runs. RA-SAE, by contrast, finds nearly identical rabbit concepts across runs.

The authors also verified this isn't just a DINOv2 artifact — similar instability appears across all five tested architectures (ConvNeXt, ResNet50, SigLIP, ViT). They even tested with 5-10% dataset perturbation and found similar trends, ruling out data-specific effects.

This is a fundamental problem. If the concepts SAEs find are largely arbitrary — dependent on random initialization rather than genuine structure in the data — then we cannot rely on them for scientific understanding of neural networks. Two researchers analyzing the same model could reach contradictory conclusions about what concepts it uses.

### Why does this happen?

The instability arises because the SAE optimization landscape has many near-equivalent local minima. The dictionary $D$ is learned as **free parameters** in a high-dimensional space ($k \times d$ values with no constraints), and there are many different sets of directions that can reconstruct the data equally well. Without additional constraints, gradient descent settles into whichever minimum it finds first, which depends heavily on random initialization.

There's also a deeper geometric reason. Standard SAE dictionary atoms are unconstrained — they can point in **any direction** in $\mathbb{R}^d$, including directions that extrapolate far beyond the actual data distribution. The paper measures this with an **OOD (Out-of-Distribution) score** and finds that TopK SAE atoms score 0.551, meaning they frequently land in regions of representation space where no real activations exist. As Makelov et al. (2023) argued, probing such extrapolated directions "may fail to activate any meaningful mechanisms within the underlying model." These ghost directions are free to vary between runs because they aren't anchored to anything real.

Classical dictionary learning methods avoid this through stronger structural constraints — non-negativity (NMF), orthogonality (PCA), or convexity (Archetypal Analysis) — which reduce the solution space and force convergence to more consistent solutions. But these methods sacrifice reconstruction quality because their constraints are too restrictive for complex neural activations.

### The sparsity-stability relationship

An additional finding (Figure 3) reveals an **inverse relationship between sparsity and stability** for standard SAEs. As sparsity increases (fewer active concepts per activation), stability decreases — from ~0.60 at low sparsity to ~0.54 at 90% sparsity. This is counterintuitive: you might expect that using fewer concepts would make the dictionary more constrained and thus more stable. Instead, higher sparsity gives each individual atom more freedom to encode different things across runs.

Classical methods maintain ~0.92 stability regardless of sparsity level. RA-SAE also maintains ~0.93 stability across all tested sparsity levels, breaking this problematic trend.

## Background: Archetypal Analysis

Before diving into the method, it's worth understanding Archetypal Analysis (AA), introduced by Cutler & Breiman in 1994. It's a lesser-known but elegant alternative to PCA and NMF for decomposing data.

**Core idea**: In Archetypal Analysis, dictionary atoms must be **convex combinations of actual data points**:

$$
D = WA, \quad \text{where } W \in \Omega_{k,n}
$$

Here, $\Omega_{k,n}$ is the set of **row-stochastic matrices**: $W \in \mathbb{R}^{k \times n}, W \geq 0, W \cdot \mathbf{1}_n = \mathbf{1}_k$. Each row of $W$ sums to 1, so each dictionary atom is a weighted average of data points — geometrically, it lies within the **convex hull** of the data.

**Intuition**: Think of the convex hull as the "boundary" of the data cloud. Archetypes are extreme, prototypical points that define the shape of the data. Every data point can then be represented as a mixture of these archetypes. Unlike PCA (which finds orthogonal directions of maximum variance) or K-Means (which finds cluster centers), AA finds the "corners" of the data distribution.

This constraint dramatically reduces the space of possible solutions. Instead of searching over all possible directions in $\mathbb{R}^d$, we're restricted to the convex hull of actual activations. This geometric anchoring is what gives Archetypal Analysis its stability.

## Method: Archetypal SAE and Its Relaxed Variant

![](/imgs/blogs/archetypal-sae-adaptive-and-stable-dictionary-learning-for-concept-extraction-20260316104615.png)

A critical design decision in this paper: **only the decoder changes**. The encoder architecture $\Psi_\theta$ remains exactly the same as in standard SAEs. All the innovation is in how the dictionary $D$ is parameterized. This is elegant because it means RA-SAE is a **plug-and-play replacement** for the decoder in any existing SAE framework — TopK, JumpReLU, or vanilla ReLU.

### A-SAE: Pure Archetypal Constraint

The most direct approach constrains the SAE dictionary to lie within the convex hull of the data:

$$
D = WA, \quad W \in \Omega_{k,n}
$$

The encoder remains a standard SAE encoder:

$$
\Psi_\theta(A) = \sigma(AW_\theta + b)
$$

where $\sigma$ is a sparsity-inducing activation (TopK in their experiments).

**Geometric guarantees**:

- Dictionary atoms satisfy $D \in \text{conv}(A)$ — every learned concept is a convex combination of real activations
- Reconstructions satisfy $ZD \in \text{cone}(A)$ — they lie in the conic hull, a natural superset

The distinction between convex hull and conic hull matters here. The convex hull $\text{conv}(A)$ is the set of all weighted averages of data points (weights sum to 1). The conic hull $\text{cone}(A)$ additionally allows scaling — it's the set of all non-negative linear combinations. Since sparse codes $Z \geq 0$ can have entries greater than 1, the reconstruction $ZD$ lives in the conic hull rather than the convex hull. This is geometrically natural: a reconstruction can "amplify" a concept direction, but it can't point in a direction that no data point points toward.

This is elegant but has a severe **scalability problem**: the weight matrix $W \in \mathbb{R}^{k \times n}$ has dimensions proportional to the number of training tokens $n$. For DINOv2 with 14×14 patches on ImageNet, that's ~250 million tokens per epoch. A weight matrix with $k \times 250M$ entries is simply intractable.

### RA-SAE: The Practical Relaxation

The key innovation is a two-step relaxation that makes the archetypal constraint practical:

**Step 1: Compress the data via K-Means.** Instead of using all $n$ data points, compute $n'$ K-Means centroids $C \subset A$ where $n' \ll n$ (they use $n' = 32{,}000$ centroids). This reduces $W$ from $k \times 250M$ to $k \times 32K$ — a >7,800× reduction in parameters. Since K-Means centroids approximate the data distribution's geometry, they preserve the essential structure of the convex hull.

**Step 2: Add a bounded relaxation term.** Allow the dictionary to deviate slightly from the convex hull of centroids:

$$
D = WC + \Lambda, \quad \text{s.t. } W \in \Omega_{k,n'} \text{ and } \|\Lambda\|_2^2 \leq \delta
$$

where:

- $W \in \mathbb{R}^{k \times n'}$ is a trainable row-stochastic matrix (enforced by ReLU + row normalization at each step)
- $\Lambda \in \mathbb{R}^{k \times d}$ is a small perturbation bounded by $\delta$
- $C \in \mathbb{R}^{n' \times d}$ contains the 32,000 K-Means centroids (fixed after preprocessing)

**Why two relaxations are needed**: Using centroids instead of all data points introduces an approximation error — the convex hull of 32K centroids is strictly smaller than the convex hull of 250M tokens. The relaxation term $\Lambda$ compensates for this gap, allowing dictionary atoms to "reach" slightly beyond the centroid hull to better positions. Without $\Lambda$, reconstruction quality drops noticeably; without the centroid compression, the method doesn't scale. Together, they achieve both goals.

**Theoretical justification (Proposition F.1)**: If $C$ contains the extreme points of $A$, then $\text{cone}(C) = \text{cone}(A)$, meaning the reduced set preserves the full representational power of the original data. In practice, K-Means centroids don't perfectly capture all extreme points, which is why $\Lambda$ is needed.

### Training procedure

The training loss is the standard SAE reconstruction objective:

$$
\mathcal{L} = \|A - Z(WC + \Lambda)^T\|_F^2
$$

The row-stochastic constraint on $W$ is enforced at each training step by:

1. Computing raw weight updates via gradient descent
2. Applying **ReLU** to ensure non-negativity: $W \leftarrow \max(W, 0)$
3. **Row-normalizing**: dividing each row by its sum so rows sum to 1

This is a **projection-based approach** — after each gradient step, $W$ is projected back onto the feasible set $\Omega_{k,n'}$. It's simple and stable in practice, though it means gradient updates can be partially "wasted" when they push weights negative (and get clipped by ReLU).

The bound on $\Lambda$ is enforced by projecting onto the $\ell_2$ ball of radius $\sqrt{\delta}$ after each gradient step: if $\|\Lambda\|_2 > \sqrt{\delta}$, rescale $\Lambda \leftarrow \Lambda \cdot \sqrt{\delta} / \|\Lambda\|_2$.

### Training details

- **Dataset**: ImageNet (1.28M images), trained for 50 epochs
- **Token counts**: 60M tokens/epoch for ConvNeXt (7×7 patches), up to 250M tokens/epoch for DINOv2 (14×14 patches)
- **Dictionary size**: $k = 5d$ where $d$ is the feature dimension (e.g., 3,840 atoms for DINOv2's 768-dim features, 10,240 for ConvNeXt's 2,048-dim features)
- **Centroids**: 32,000 K-Means centroids computed as a preprocessing step on the full activation dataset
- **Encoder**: TopK activation with ~90% sparsity (only ~10% of dictionary atoms active per token)
- The authors release an open-source library called **Overcomplete** for large-scale SAE training on vision models

## New Evaluation Benchmarks

A major contribution of this paper is recognizing that **reconstruction quality alone is insufficient** for evaluating concept dictionaries. Two dictionaries can reconstruct equally well but contain very different — and differently meaningful — concepts. The authors introduce two benchmarks that directly test semantic quality.

### Plausibility Benchmark

**Idea**: If a vision model has learned to classify objects, its classification head weights $\{v_1, \ldots, v_c\}$ define directions in representation space that correspond to known semantic categories. A good dictionary should recover these directions.

**Metric**:

$$
\text{Plausibility} = \frac{1}{c} \sum_{i=1}^{c} \max_j \langle v_i, D_j \rangle
$$

For each class direction $v_i$, find the dictionary atom $D_j$ most aligned with it. Average over all classes.

**Why this matters**: This tests whether the dictionary captures concepts the model actually uses for its task, not just arbitrary directions that happen to reconstruct well. A plausibility score of 1.0 would mean every class has a perfectly aligned dictionary atom.

**Results (Table 2)**:

| Model    | TopK SAE | RA-SAE ($\delta$=0.01) |
| -------- | -------- | ---------------------- |
| ConvNeXt | 0.168    | **0.390**              |
| ResNet   | 0.230    | **0.601**              |
| ViT      | 0.294    | **0.362**              |

RA-SAE recovers 2-3x more "true" classification directions than standard SAEs. This is a dramatic improvement — it means RA-SAE dictionaries contain concepts that are far more aligned with what the model actually computes.

### Soft Identifiability Benchmark

**Idea**: Create synthetic datasets where we **know** the ground-truth concepts, then test whether the SAE can recover them. This is the gold standard for evaluation — if you know the answer, you can measure exactly how well each method recovers it.

**Setup**: The construction is clever. 12 synthetic datasets are created by **collaging 4 distinct objects** (e.g., different colored gems, animals, or common objects from ImageNet) into composite images. Each image contains exactly 4 objects, and the SAE is asked to discover concepts that correspond to each object class.

The images are processed through the pretrained vision model to get pooled activations, then the SAE is trained on these activations. For each object class $y_j$, the benchmark checks whether the SAE discovers a concept $z_i$ that activates above some threshold $\lambda$ precisely when $y_j$ is present:

$$
\text{Accuracy}_j = \max_{\lambda \in \mathbb{R}, i \in [k]} P_{(z,y)}((z_i > \lambda) = y_j)
$$

The optimization over $\lambda$ means we're searching for the best possible threshold — giving the SAE every chance to succeed. If no feature cleanly separates a concept, even the optimal threshold will yield low accuracy.

This benchmark tests **concept disentanglement**: can the SAE find features that independently detect each object, without confounding them? A perfect score (1.0) means every ground-truth concept maps one-to-one to an SAE feature.

**Results (Table 3)**:

| Method   | DINOv2    | ResNet    | SigLIP    | ViT       |
| -------- | --------- | --------- | --------- | --------- |
| TopK SAE | 0.814     | 0.815     | 0.829     | 0.833     |
| A-SAE    | **0.948** | **0.963** | **0.960** | **0.962** |
| RA-SAE   | 0.945     | 0.960     | 0.959     | 0.959     |

Archetypal variants achieve ~12% absolute improvement, nearly reaching the theoretical identifiability ceiling. This is a striking result: it means TopK SAEs fail to cleanly recover ~18% of known concepts, while Archetypal SAEs miss only ~4-5%. The gap is consistent across all four architectures, suggesting this is a fundamental advantage of the geometric constraint, not a model-specific quirk.

**Why the improvement?** When dictionary atoms are free to go out-of-distribution, they can "waste" capacity encoding directions that don't correspond to any real concept. By anchoring atoms to the convex hull, RA-SAE ensures every atom represents a blend of real activations — making it much more likely that natural concept boundaries are captured.

## Experiments and Results

### Experimental Setup

- **Vision Models**: DINOv2 (self-supervised), SigLIP (contrastive), ViT, ConvNeXt, ResNet50
- **Dataset**: ImageNet (1.28M images)
- **Tokens**: 60M (ConvNeXt, 7×7 patches) to 250M (DINOv2, 14×14 patches) per epoch
- **Dictionary size**: 5× the feature dimension (e.g., 3,840 atoms for DINOv2's 768-dim features)
- **Sparsity**: ~90% (controlled via TopK)
- **Centroids for C**: 32,000 via K-Means

### The Stability-Reconstruction Trade-off

This is the central experimental result. The authors compare across 5 methods at 5 sparsity levels on 4 models:

| Method                 | R² (Reconstruction) | Stability |
| ---------------------- | ------------------- | --------- |
| TopK SAE               | 89.52               | 0.542     |
| Semi-NMF               | ~67                 | 0.93+     |
| RA-SAE ($\delta$=0.01) | **89.34**           | **0.927** |

**The key finding**: RA-SAE breaks the stability-reconstruction trade-off. Classical methods (Semi-NMF, Convex-NMF) are stable but reconstruct poorly. Standard SAEs reconstruct well but are unstable. RA-SAE achieves **both** — near-identical reconstruction to TopK SAE (89.34 vs 89.52 R²) while matching classical methods in stability (0.927 vs 0.93+).

### Comprehensive Dictionary Quality Metrics (DINOv2)

Beyond stability and reconstruction, the authors evaluate several structural properties:

| Metric                    | TopK SAE | RA-SAE    | Better? |
| ------------------------- | -------- | --------- | ------- |
| R²                        | 89.52    | 89.34     | ≈       |
| Dead codes                | 0.00     | 0.02      | ≈       |
| **Stability**             | 0.542    | **0.927** | RA-SAE  |
| **OOD Score**             | 0.551    | **0.060** | RA-SAE  |
| **Stable Rank**           | 141.6    | **5.89**  | RA-SAE  |
| **Coherence**             | 0.728    | **0.973** | RA-SAE  |
| **Connectivity**          | 0.002    | **0.159** | RA-SAE  |
| **Negative Interference** | 135.7    | **0.012** | RA-SAE  |

Let me unpack these metrics — they tell a comprehensive story about dictionary quality beyond just "does reconstruction work":

**OOD Score** (lower is better): Quantifies how far dictionary atoms stray from the convex hull of actual data. TopK SAE atoms score 0.551, meaning they frequently land in regions of representation space where **no real activations exist**. RA-SAE stays close at 0.060. This is perhaps the most damning metric for standard SAEs: over half their "concepts" point toward phantom directions that the model never actually produces. As the authors cite, probing such extrapolated directions "may fail to activate any meaningful mechanisms within the underlying model."

**Stable Rank**: Defined as $\text{sr}(D) = \|D\|_F^2 / \|D\|_2^2$, this measures the effective dimensionality of the dictionary. TopK's 141.6 means the dictionary spans many near-equal dimensions — a sign of high redundancy where many atoms encode similar things. RA-SAE's 5.89 means the dictionary is much more compact: a few dominant concept directions with the rest providing fine-grained refinement. This is exactly what you'd want from an interpretable dictionary — a small number of major concepts rather than hundreds of near-duplicates.

**Coherence**: The maximum pairwise cosine similarity among atoms. Lower coherence means less redundancy. RA-SAE's higher coherence (0.973 vs 0.728) might seem surprising at first — doesn't higher coherence mean _more_ similar atoms? The key is that RA-SAE atoms are **structured**: they form organized clusters of related concepts (e.g., different hand positions, different animal body parts) rather than randomly scattered directions. The atoms are coherent because they belong to meaningful semantic neighborhoods, not because they're redundant.

**Connectivity**: Measured as $\ell_0(\mathbf{Z}\mathbf{Z}^T)$, this captures the combinatorial diversity of code activation patterns — how often different concepts co-activate. RA-SAE's higher connectivity (0.159 vs 0.002) means its concepts form **structured co-activation patterns**. For example, "rabbit ears" and "rabbit face" often co-activate but "rabbit ears" and "car wheel" rarely do. TopK's near-zero connectivity means its concepts activate nearly independently — which sounds good but actually suggests they're not capturing compositional structure.

**Negative Interference**: This is the most dramatic metric. It counts instances where two simultaneously active concepts push the reconstruction in **opposite directions** — they literally cancel each other out. TopK has catastrophic interference (135.7!) while RA-SAE has near-zero (0.012). Think about what 135.7 means: on average, each reconstruction involves over 100 pairs of concepts fighting each other. This strongly suggests TopK SAE "concepts" are more like **compression artifacts** (positive and negative components that combine to approximate a direction) than genuine semantic features. RA-SAE's near-zero interference is what we'd expect from a dictionary of real, independent concepts.

### Ablation: The Relaxation Parameter $\delta$

The relaxation parameter $\delta$ controls the trade-off between archetypal purity and reconstruction flexibility:

| $\delta$       | Reconstruction  | Stability        |
| -------------- | --------------- | ---------------- |
| 0 (pure A-SAE) | Lower           | Highest          |
| **0.01**       | **TopK-level**  | **Near-maximal** |
| 0.1            | Intermediate    | Good             |
| 1.0            | Approaches TopK | Degraded         |

**$\delta = 0.01$ is the sweet spot**: just enough flexibility to match TopK SAE reconstruction while preserving 93%+ stability. As $\delta$ increases, the constraint weakens and behavior converges toward unconstrained SAEs.

### Ablation: Centroid Selection Method

The authors compare different strategies for computing the centroid set $C$:

- **K-Means**: Most reliable and robust — the recommended choice
- **Isolation Forests**: High variance across runs
- **Convex hull computation**: Intractable in high dimensions
- **Outlier detection (LOF, One-class SVM)**: Inconsistent results

K-Means works well because it naturally captures the data distribution's geometry while being computationally efficient and deterministic (given the same initialization).

### Qualitative Findings: What Concepts Does RA-SAE Find?

The qualitative analysis is where this paper really shines. The authors applied RA-SAE to DINOv2-B (with 4 registers) and visualized the resulting 16,000 dictionary atoms using UMAP (Figure 7). The visualization reveals **clear semantic clustering** — atoms group into coherent semantic neighborhoods rather than being randomly distributed. Three particularly interesting clusters emerged:

#### UMAP visualization reveals structured concept space

The UMAP projection of RA-SAE's 16,000 dictionary atoms shows distinct clusters corresponding to:

1. **Hand positions and body poses** — a cluster of concepts encoding different ways humans interact with objects
2. **Spatial "under" relationships** — concepts capturing spatial arrangements across species
3. **Fine-grained animal facial features** — ear, eye, nose, and mouth concepts for different animals

This kind of structured organization doesn't appear in TopK SAE dictionaries, which look much more uniformly scattered in UMAP space. The clustering isn't imposed — it emerges naturally from the archetypal constraint, because atoms anchored to real data naturally group by the semantic structure already present in the model's representations.

#### Fine-grained body part decomposition (Figure 8)

For the "rabbit" class, RA-SAE's top-5 concepts cleanly separate: ears, body, face, paws, and fur texture as distinct dictionary atoms. Each concept activates on the corresponding spatial region of rabbit images. TopK SAE, by contrast, produces less organized concepts — multiple atoms fire on overlapping regions without clear semantic boundaries.

This is exactly the kind of **compositional structure** we want from an interpretability tool. Understanding that a model represents "rabbit" as a composition of ears + body + face + paws tells us much more about the model's internal representation than knowing it has a single "rabbit-ish" direction.

#### Complex pose concepts

RA-SAE discovers **hand position clusters** that represent compositional human poses: hands in pockets, arms resting on shoulders, specific grip positions, fingers interlaced. These are remarkable because they're not ImageNet classes — they're **sub-object compositional concepts** that the model has learned to represent but that standard analysis methods don't typically surface.

#### Spatial relationship features

A "spatial under" concept activates for birds under branches, zebras under trees, and cats under furniture. This is significant because it demonstrates the model has learned a **generalizable spatial relationship** that transfers across species and object categories. The concept isn't "bird" or "tree" — it's specifically "something underneath something else."

#### Three surprising emergent concepts (Figure 9)

The paper highlights three particularly surprising discoveries:

1. **Shadow-based depth reasoning**: Features that specifically respond to shadow patterns cast by dogs and other animals. This suggests DINOv2 uses shadow information for depth estimation or 3D understanding — a concept that would be easy to miss with less stable dictionaries and that reveals something non-obvious about how the model processes images.

2. **Role-specific "barber" concept**: A feature that activates specifically for the **person cutting hair**, not the client sitting in the chair. Both people are in the image, but the concept distinguishes them by their social role and action (holding scissors, standing behind). This shows the model encodes agent-specific information, not just visual appearance.

3. **Petal edge detection**: Dedicated features for the precise edges of flower petals — the boundary between petal and background. This shows extremely fine-grained visual concept extraction at the sub-object level, distinguishing "petal interior" from "petal edge" as separate concepts.

#### Why RA-SAE finds better concepts

These qualitative results aren't just cherry-picked examples — they reflect the quantitative metrics. Because RA-SAE atoms stay close to real data (low OOD score), they naturally correspond to patterns the model actually computes. Because they don't interfere with each other (low negative interference), each concept can be interpreted independently. And because they're stable across runs, researchers can trust that these concepts are genuine features of the model rather than artifacts of a particular random seed.

### Dead Codes: Atom Utilization

One concern with constrained dictionaries is whether all atoms get used. Dead codes — dictionary atoms that never activate — waste capacity and can indicate training issues.

| Method       | Dead Codes |
| ------------ | ---------- |
| Vanilla SAE  | 0.00       |
| TopK SAE     | 0.00       |
| JumpReLU SAE | 0.00       |
| Semi-NMF     | 0.064      |
| Convex-NMF   | 0.031      |
| RA-SAE       | 0.02       |

All SAE variants (including RA-SAE) achieve near-perfect atom utilization. Classical NMF methods have slightly more dead codes because their stricter constraints can leave some atoms stranded in regions of the dictionary space that no data point needs. RA-SAE's 0.02 dead code rate is negligible — the relaxation term $\Lambda$ likely helps atoms find useful positions even when the centroid-based initialization isn't perfect.

## Why Archetypal Constraints Work: Geometric Intuition

The paper provides several theoretical results that explain why the approach works. Let me walk through both the formal propositions and the intuition behind them.

### Formal theoretical results

**Proposition F.1 (Representational Preservation)**: If the centroid set $C$ contains the extreme points of $A$, then $\text{cone}(C) = \text{cone}(A)$. This means we lose no representational power by using centroids instead of the full dataset — any direction the full dataset can represent, the centroids can too. In practice, K-Means doesn't perfectly capture all extreme points, but the relaxation term $\Lambda$ compensates for this gap.

**Proposition F.2 (Geometric Stability)**: The archetypal constraint bounds dictionary atoms to the convex hull, which is a **compact set**. This means the space of valid dictionaries is bounded, reducing the number of distinct local minima the optimizer can reach. Different random initializations are more likely to converge to the same (or similar) solutions. Formally, the feasible set for an unconstrained SAE is all of $\mathbb{R}^{k \times d}$ (infinite), while for A-SAE it's $\text{conv}(A)^k$ (bounded). This massive reduction in search space is what drives stability.

**Proposition F.3 (Rank Bound)**: The rank of an archetypal dictionary $D = WA$ is bounded by $\text{rank}(A)$. Since $W$ is a linear map and rank can only decrease through linear transformation, the dictionary can't have higher rank than the data itself. This prevents the dictionary from "hallucinating" directions that don't exist in the data — it can only recombine existing directions.

**Proposition F.4 (OOD Bound)**: Under the assumption of non-interfering archetypes (atoms that don't cancel each other during reconstruction), the OOD score is inherently bounded. This explains why RA-SAE atoms stay close to the data distribution and connects the geometric constraint directly to the dramatically improved OOD scores in Table 1.

### Intuitive understanding

**Analogy 1 — Describing a room**: Think of unconstrained SAEs as trying to describe a room by pointing in any direction, including through walls and into imaginary spaces. Archetypal SAEs can only point toward actual furniture in the room. Both might describe the space equally well (similar reconstruction), but the constrained version gives you descriptions that correspond to real things.

**Analogy 2 — Coordinate systems**: Imagine you need to create a coordinate system for a country's geography. An unconstrained approach might place axes anywhere — some pointing into the ocean, some into neighboring countries. These axes "work" for reconstruction but don't correspond to any real location. An archetypal approach requires every axis to point toward an actual place in the country. The resulting coordinate system is less flexible but far more interpretable — every axis direction means something real.

**Analogy 3 — Why stability follows from anchoring**: Consider two people independently trying to describe the shape of a cloud. If they can use any vocabulary (unconstrained), they'll likely use very different words. But if they can only use words that point to actual features of the cloud (anchored), they'll converge on similar descriptions — because the cloud's actual shape constrains what can be said about it.

## Connection to the Broader SAE Literature

This paper arrives at an important moment in the SAE interpretability literature. Several recent works have raised concerns about SAE reliability:

### Historical lineage of dictionary learning

The paper traces a rich intellectual history from sparse coding (Olshausen & Field, 1996-97) through compressed sensing (Candès et al., 2006; Donoho, 2006) to modern SAEs. The key insight is that SAEs are not a fundamentally new idea — they're the latest entry in a long line of dictionary learning methods, each adding different constraints. Understanding this lineage helps appreciate what RA-SAE contributes: it imports a well-understood classical constraint (convex hull membership from Archetypal Analysis) into the modern SAE framework.

### Concept-based vs. attribution-based explainability

The paper positions SAEs within the broader explainability landscape. Attribution methods (saliency maps, Grad-CAM) answer "where does the model look?" while concept-based methods (ACE, ICE, CRAFT, SAEs) answer "what concepts does the model use?" These are complementary but different questions. SAEs are positioned as scalable successors to earlier concept methods, but the instability finding shows they're not yet mature enough to be fully trusted.

### Recent SAE concerns this paper addresses

- **Feature splitting** (studied in "A is for Absorption"): Single concepts split into multiple redundant features across the dictionary. RA-SAE's low stable rank (5.89 vs 141.6) suggests it naturally mitigates this — there are fewer redundant atoms to begin with.
- **Compositionality issues** (Wattenberg & Viegas, 2024): SAE features may not compose in meaningful ways. RA-SAE's low negative interference (0.012 vs 135.7) suggests much better compositionality — atoms add constructively rather than destructively.
- **Limited intervention effects** (Bhalla et al., 2024): Activating/deactivating SAE features doesn't reliably change model behavior. RA-SAE's higher plausibility scores suggest its features are more aligned with the model's actual computation, potentially making interventions more effective.
- **Overly specific features** (Bricken et al., 2023): Some SAE features are so specific they only activate on a handful of examples. RA-SAE's anchoring to data centroids may prevent this by ensuring atoms represent blend of real activation patterns rather than extreme outlier directions.

The instability finding is particularly concerning because it undermines the entire premise of using SAEs as a scientific instrument. If two researchers get different "concepts" from the same model, neither can claim to have found the model's true representations. RA-SAE's 93% stability brings SAEs much closer to being a reliable tool.

## My Thoughts

### What I found most compelling

The **instability measurement** is the paper's most important contribution, even more than the fix. Before this work, the community largely assumed SAEs find "real" features — the instability result shows this assumption needs serious scrutiny. The fact that only ~50% of TopK SAE features are reproducible means we should be skeptical of any individual SAE analysis that hasn't been validated across multiple runs. This has implications for every paper that uses SAEs to make claims about model internals without running multiple seeds.

The **negative interference metric** is also eye-opening. A value of 135.7 for TopK SAEs means learned concepts actively fight each other during reconstruction. This isn't just an aesthetic problem — it suggests TopK SAE "concepts" are more like compression artifacts than genuine semantic features. When concept A and concept B simultaneously activate but push the reconstruction in opposite directions, what does it mean to say "the model uses concept A"? The concept is not independent — it only makes sense in combination with its canceling partner. RA-SAE's near-zero interference (0.012) is what we'd expect from a dictionary that captures real, independent concepts.

The **unified dictionary learning framework** (Table of constraints) is an underappreciated contribution. By showing that K-Means, PCA, NMF, and SAEs are all special cases of the same objective, the paper reframes the question from "which method is best?" to "which constraints are appropriate?" This is a more productive framing because it reveals the design space and makes it clear that RA-SAE's innovation is purely about the constraint, not the architecture.

### The encoder-decoder asymmetry insight

A subtle but important design choice: **only the decoder is constrained**. The encoder remains a free neural network. This is smart because:

1. The encoder needs flexibility to learn complex nonlinear mappings from activations to codes
2. The decoder (dictionary) is what defines the "concepts" — constraining it is where stability matters
3. It makes RA-SAE a plug-and-play replacement that works with any encoder architecture (TopK, JumpReLU, etc.)

This suggests a general principle for constrained dictionary learning: constrain the representational basis (decoder) for interpretability, but let the encoding process (encoder) remain flexible for performance.

### Limitations to consider

**Vision-only evaluation**: All experiments are on vision models (DINOv2, SigLIP, ViT, ConvNeXt, ResNet50). The transfer to language model SAEs — where most of the mechanistic interpretability community works — is not demonstrated. Language model activations may have very different geometry (e.g., the "linear representation hypothesis" suggests LLM features are linear, which might mean convex hull constraints behave differently). The paper acknowledges the method generalizes in principle but provides no language model experiments.

**Computational overhead**: K-Means preprocessing on 250M tokens, maintaining the row-stochastic constraint via ReLU + normalization at every step, and projecting $\Lambda$ onto the norm ball all add overhead. The paper doesn't report wall-clock training times or memory comparisons, which is a notable omission. For practitioners already struggling with SAE training costs at scale, knowing the concrete overhead matters.

**The $\delta$ sensitivity**: While $\delta = 0.01$ works well empirically across all tested models, the optimal value likely depends on the model, layer, and dataset. The paper doesn't provide a principled way to select $\delta$ without experimentation. A validation-based or adaptive approach would make the method more practically useful.

**Centroid quality dependency**: The entire method hinges on K-Means centroids being a good summary of the data geometry. If the activation distribution has complex structure (long tails, multiple modes, high-dimensional clusters), 32,000 centroids may not suffice. The paper tests only on ImageNet, which is a relatively well-structured dataset.

**Stability metric limitations**: The stability metric uses the Hungarian algorithm, which has $O(k^3)$ complexity. For very large dictionaries (e.g., 65K+ atoms used in some language model SAEs), this becomes computationally expensive. Additionally, the metric only considers pairwise atom matching — it doesn't capture higher-order structural similarities between dictionaries.

### Ideas for future directions

- **Archetypal SAEs for language models**: The most obvious and impactful next step. Apply RA-SAEs to transformer language models and compare with standard SAEs from Anthropic's and OpenAI's work. If instability is equally bad in language models (which I suspect it is, given the similar overcomplete optimization landscape), this could fundamentally change how the field does mechanistic interpretability. The key question is whether the convex hull constraint is equally meaningful for language model activations, which may have different geometric structure than vision model activations.

- **Layer-wise $\delta$ scheduling**: Different layers likely have different activation geometries. Early vision layers represent simple features (edges, textures) in a relatively low-dimensional manifold, while later layers represent complex semantic concepts in a higher-dimensional space. Early layers might benefit from tighter constraints (more stability), while later layers might need more flexibility. A learned or scheduled $\delta$ per layer could improve both metrics.

- **Combining with other SAE improvements**: RA-SAE's constraint is purely on the decoder parameterization, making it orthogonal to improvements in encoder architecture (JumpReLU, Gated SAEs), training procedures (progressive training, curriculum learning), or sparsity penalties. Combining archetypal constraints with these could yield compounding gains.

- **Causal validation via activation patching**: The plausibility and identifiability benchmarks test correlation between features and concepts. A stronger test would be causal: do RA-SAE features, when activated or ablated via activation patching, reliably change model behavior? Given the higher plausibility scores, I'd expect RA-SAE to also perform better on causal benchmarks, but this needs verification. If RA-SAE features are more causally relevant, it would provide the strongest evidence yet for preferring constrained dictionaries.

- **Adaptive centroid selection**: Instead of fixed K-Means centroids computed once before training, use an online scheme that updates centroids periodically during training. As the dictionary evolves, the relevant regions of activation space may shift, and stale centroids could become suboptimal anchors.

- **Theoretical lower bounds on instability**: The paper shows standard SAEs have ~50% stability empirically but doesn't prove this is a fundamental limit. Is there a theoretical lower bound on stability for unconstrained overcomplete dictionaries? Understanding this could tell us whether RA-SAE's 93% is close to optimal or whether further improvements are possible.

- **Cross-model concept transfer**: Since RA-SAE finds stable concepts, it opens the possibility of comparing concept dictionaries across different models. Do DINOv2 and CLIP discover the same concepts? If RA-SAE dictionaries are stable enough to be compared across architectures, this could enable a new kind of model comparison that goes beyond performance metrics.

## References

1. [Archetypal SAE: Adaptive and Stable Dictionary Learning for Concept Extraction in Large Vision Models (arXiv:2502.12892v2)](https://arxiv.org/abs/2502.12892v2)
2. [Cutler, A. & Breiman, L. (1994). Archetypal Analysis. Technometrics, 36(4), 338-347.](https://doi.org/10.1080/00401706.1994.10485840)
3. [Bricken, T., et al. (2023). Towards Monosemanticity: Decomposing Language Models With Dictionary Learning.](https://transformer-circuits.pub/2023/monosemantic-features)
4. [Cunningham, H., et al. (2023). Sparse Autoencoders Find Highly Interpretable Features in Language Models.](https://arxiv.org/abs/2309.08600)
5. [Olshausen, B.A. & Field, D.J. (1996). Emergence of simple-cell receptive field properties by learning a sparse code for natural images. Nature, 381(6583), 607-609.](https://doi.org/10.1038/381607a0)
6. [Lee, D.D. & Seung, H.S. (2001). Algorithms for Non-negative Matrix Factorization. NeurIPS.](https://proceedings.neurips.cc/paper/2000/hash/f9d1152547c0bde01830b7e8bd60024c-Abstract.html)
7. [Bhalla, U., et al. (2024). Interpreting and Steering Features in Images.](https://arxiv.org/abs/2401.09095)
8. [Wattenberg, M. & Viegas, F. (2024). Relational Composition in Neural Networks: A Survey and Call to Action.](https://arxiv.org/abs/2407.14662)
