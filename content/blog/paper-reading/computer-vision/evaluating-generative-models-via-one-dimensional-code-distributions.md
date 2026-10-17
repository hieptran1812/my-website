---
title: "Evaluating Generative Models via One-Dimensional Code Distributions"
publishDate: "2026-10-17"
date: "2026-10-17"
description: "Why FID is blind to texture and artifacts, and how counting discrete 1D image tokens (CHD) and scoring corrupted token sequences (CMMS) tracks human judgment more closely, with every equation worked on tiny numbers."
category: "paper-reading"
subcategory: "Computer Vision"
tags: ["interpretability-analysis", "cvpr-2026", "cvpr-2026-highlight", "paper-reading", "generative-model-evaluation", "fid", "image-tokenizers", "titok", "image-quality-assessment", "hellinger-distance"]
author: "Hiep Tran"
featured: true
readTime: 18
image: "/imgs/blogs/evaluating-generative-models-via-one-dimensional-code-distributions-1.webp"
excerpt: "FID compares Gaussians fitted to features trained to ignore texture. This CVPR 2026 highlight evaluates generators by counting discrete tokens from a 1D tokenizer instead, and its two metrics, CHD and CMMS, line up with human ratings better than FID, CMMD and learned IQA models."
paper:
  title: "Evaluating Generative Models via One-Dimensional Code Distributions"
  authors: "Zexi Jia, Pengcheng Luo, Yijia Zhong, Jinchao Zhang, Jie Zhou"
  venue: "CVPR 2026 (Highlight); arXiv 2026"
  url: "https://arxiv.org/abs/2603.08064"
---

> [!tldr]
> **The claim.** FID and its variants judge generators in a feature space trained for recognition, which is built to ignore texture, sharpness and local coherence. Jia et al. move evaluation into the space of discrete tokens from a 1D image tokenizer (TiTok, 128 tokens per image, 4,096-entry codebook) and compare token statistics directly.
>
> **Why it matters.** Two metrics fall out. **CHD** (Codebook Histogram Distance) is training-free: a Hellinger distance between token histograms and token-pair histograms of real and generated sets. **CMMS** (Code Mixture Model Score) is a no-reference quality scorer trained only on synthetically corrupted token sequences from ImageNet, with no human labels.
>
> **The surprising result.** A small Transformer over 128 token ids, never trained on human ratings, beats DeQA-Score and QualiCLIP: Spearman 0.943 on AGIQA and pairwise preference accuracy 74.9% on HPDv2. CHD stabilizes with about 1,000 samples where FID needs over 10,000.
>
> **Where it fails.** Everything rides on one retrained tokenizer. A generator that copies its training set gets a near-perfect CHD. Several headline correlations are computed over only 6 to 10 models, and I could not reproduce the AGIQA CHD and FID correlations from the per-model numbers printed in Table 1.

## The metric that cannot see texture

Most practitioners have seen FID disagree with their own eyes: a model with smeared hands can post a better FID than one with crisp detail. The paper's diagnosis is blunt. The problem is not the Gaussian or the sample size, it is the space FID lives in.

Inception-V3 was trained to say "dog" whether the dog is sharp, blurry, or painted in oil. That invariance is what a classifier needs and what a quality metric must not have.

![Figure 1 from Jia et al. (2026): FID on continuous recognition features (left) versus comparing token distributions from a 1D tokenizer with a Hellinger distance (right)](/imgs/blogs/evaluating-generative-models-via-one-dimensional-code-distributions-fig1.webp)

FID fits one Gaussian to the real features and one to the generated features, then measures the Fréchet distance between them:

$$
\text{FID} = \|\mu_r - \mu_g\|_2^2 + \mathrm{Tr}\left(\Sigma_r + \Sigma_g - 2(\Sigma_r\Sigma_g)^{1/2}\right)
$$

Here $\mu_r, \Sigma_r$ are the mean vector and covariance matrix of real-image features, and $\mu_g, \Sigma_g$ the same for generated images. With 2,048-dimensional Inception features, $\Sigma$ has about 4.2 million entries, which is why covariance estimates are noisy and the matrix square root is numerically touchy.

The Gaussian assumption also breaks in a way that is easy to picture: real feature clouds are multi-modal.

![A two-cluster feature cloud with one fitted Gaussian whose mean falls in the empty gap, next to a token histogram that keeps both modes](/imgs/blogs/evaluating-generative-models-via-one-dimensional-code-distributions-1.webp)

The fitted ellipse wraps both clusters, and its mean $\mu$ lands where no image lives. The histogram on the right assumes no shape: two bumps stay two bumps.

## Why recognition features throw quality away

The paper gives an information-theoretic version of the "invariance" argument. Split an image $x$ into semantic content $x_s$ and appearance $x_a$, and let $\phi$ be an encoder. The chain rule of mutual information gives

$$
I(x_s, x_a; \phi(x)) = I(x_s; \phi(x)) + I(x_a; \phi(x) \mid x_s)
$$

where $I(\cdot\,;\cdot)$ is mutual information, the number of bits one variable tells you about another. Recognition training pushes up the first term (know the class) and pushes down the second (ignore appearance once the class is known). The second term is where blur, noise and texture glitches live.

The data processing inequality closes the door. If latent quality $q$ affects the image, which is then encoded, so $q \to x \to \phi(x)$, then

$$
I(q; x) \ge I(q; \phi(x))
$$

No post-processing of $\phi(x)$ recovers quality information the encoder discarded. CLIP-FID and DINO-FID change the encoder but not the objective.

A second, more mechanical loss is **global pooling**. Most encoders average a spatial feature map, $\phi(x) = \frac{1}{HW}\sum_{i,j} f_{i,j}(x)$, where $f_{i,j}(x)$ is the feature vector at grid cell $(i,j)$ of an $H \times W$ map. A six-fingered hand and a five-fingered hand can average to nearly the same vector. CMMD swaps the Gaussian for a kernel MMD but keeps the pooled features.

| Metric family | Feature space | Distribution model | What it can miss |
| --- | --- | --- | --- |
| FID, KID | Inception-V3, pooled | Gaussian (FID), kernel (KID) | Texture, sharpness, local artifacts |
| CLIP-FID, DINO-FID | CLIP or DINO, pooled | Gaussian | Same: invariance plus pooling |
| CMMD | CLIP, pooled | Kernel MMD | Pooling; kernel bandwidth choice |
| HPS, PickScore, DeQA | Learned on human labels | None (per image) | Domain shift, label cost |
| CHD, CMMS (this paper) | Discrete 1D tokens | Histograms (CHD), learned (CMMS) | Whatever the tokenizer cannot reconstruct |

## An image as a 128-word sentence

The replacement space is the output of a **1D image tokenizer**. TiTok encodes a 256 by 256 image into a short sequence of codebook indices, and a decoder reconstructs the image from them. The codebook is a vocabulary; the image is a sentence written in it.

The key difference from a classifier is the training objective. A tokenizer is trained to *reconstruct*, so it cannot afford to discard appearance: if blur were not encoded somewhere, the decoder could not draw it. The paper cites prior analysis showing that individual TiTok token positions encode disentangled attributes such as blur, lighting and sharpness.

In this paper each image becomes $\mathbf{c} = [c_1, \dots, c_N]$ with $N = 128$ tokens drawn from a codebook $\mathcal{V}$ with $|\mathcal{V}| = 4096$. "1D" means the tokens are not tied to image patches the way a VQGAN's 16 by 16 grid is; each token can describe global or local properties.

The central hypothesis is that quality shows up in token statistics. Natural images produce structured, low-entropy token patterns; degraded images produce more random, high-entropy ones:

$$
H(\mathbf{c} \mid q_{\text{high}}) \lt H(\mathbf{c} \mid q_{\text{low}})
$$

with $H$ the Shannon entropy. Spatial coherence is likewise measurable as the mutual information between adjacent tokens, $I(c_i; c_{i+1}) = H(c_i) + H(c_{i+1}) - H(c_i, c_{i+1})$, which falls when artifacts break natural co-occurrence.

Figure 2 is the sanity check behind the hypothesis. The authors apply 10 levels of five distortions (block shuffling, random masking, Gaussian noise, sharpness reduction, color distortion) to 1,000 images.

![Figure 2 from Jia et al. (2026): five distortion types at ten severity levels, the shift in token distributions, and the monotonic rise in CHD](/imgs/blogs/evaluating-generative-models-via-one-dimensional-code-distributions-fig2.webp)

Only a small subset of "perceptually sensitive" tokens moves, but consistently, and CHD climbs monotonically with severity for every distortion. That is necessary, not sufficient: FID also rises under synthetic noise. The real test is agreement with humans.

## CHD-1D: compare the vocabularies

**The problem.** Given a set of real images $\mathcal{R}$ and a set of generated images $\mathcal{G}$, produce one number for how different the distributions are, without fitting any parametric model.

**The analogy.** You can tell two authors apart by word counts alone: one never writes "however", the other overuses "very".

**The mechanism.** Tokenize every image, pool all tokens of a set into one bag, and normalize the counts into a histogram over the 4,096 codebook entries:

$$
h_{\mathcal{S}}^{(1)}(v) = \frac{1}{|\mathcal{S}| \cdot N} \sum_{I \in \mathcal{S}} \sum_{i=1}^{N} \mathbb{I}[c_i(I) = v], \quad v \in \mathcal{V}
$$

Here $\mathcal{S}$ is either $\mathcal{R}$ or $\mathcal{G}$, $|\mathcal{S}|$ the number of images in it, $c_i(I)$ the $i$-th token of image $I$, and $\mathbb{I}[\cdot]$ the indicator (1 if true, else 0). The denominator is the total token count, so the histogram sums to 1. Then compare the two histograms with the Hellinger distance:

$$
\text{CHD-1D}(\mathcal{R}, \mathcal{G}) = \frac{1}{\sqrt{2}} \left\| \sqrt{h_{\mathcal{R}}^{(1)}} - \sqrt{h_{\mathcal{G}}^{(1)}} \right\|_2 \in [0, 1]
$$

where the square root is taken element-wise. Zero means identical histograms; one means the two sets use completely disjoint tokens.

**A worked example.** Shrink the codebook to four tokens A, B, C, D. Real images use them with frequencies 0.5, 0.3, 0.2 and 0.0. Model A is close but sometimes emits D, a token real images never use. Model B has collapsed: it uses only A and B.

![CHD-1D worked on a 4-token codebook: real versus a model that uses an unseen token (Hellinger 0.23) and a mode-collapsed model (Hellinger 0.35)](/imgs/blogs/evaluating-generative-models-via-one-dimensional-code-distributions-2.webp)

For Model A, only two bins differ. The A bin contributes the squared root gap between 0.5 and 0.4, which is about 0.0056, and the D bin contributes the full 0.1. The sum is 0.1056, its square root 0.325, divided by the square root of 2 gives **0.23**. Model B loses tokens C and D entirely and overuses A, which totals to **0.35**. The collapsed model ranks worse, as a human would judge it.

**Why Hellinger, and when it fails.** The figure's note is the practical reason. KL divergence of Model A from the real histogram is infinite, because the real set puts zero mass on D. With 4,096 bins, empty bins are common; Hellinger is symmetric, bounded, and finite on them. The ablation agrees: Hellinger beats KL, cosine and Wasserstein on every benchmark (N-MSE 0.112 versus 0.124 for KL on AGIQA).

The failure mode is inherent to bag-of-words statistics: CHD-1D cannot see *arrangement*. Shuffle the 128 tokens of every image and the histogram is unchanged. That motivates the second-order statistic.

## CHD-2D: compare the grammar

**The problem.** Two generators can use the right words in the wrong order. Grass tokens and sky tokens in the right proportions, but grass in the sky.

**The mechanism.** Treat the quantized image as a grid of tokens $\{c(\mathbf{p})\}$ with positions $\mathbf{p} = (x, y)$, and pick a small set of displacements $\mathcal{D} = \{(1,0), (0,1)\}$, meaning "my right neighbour" and "my down neighbour". For each displacement $\Delta$, count how often token $u$ sits next to token $v$:

$$
h_{\mathcal{S}}^{(2)}(u, v; \Delta) = \frac{1}{Z_{\mathcal{S},\Delta}} \sum_{I \in \mathcal{S}} \sum_{\mathbf{p},\, \mathbf{p}+\Delta \in \Omega_I} \mathbb{I}[c(\mathbf{p}) = u,\, c(\mathbf{p}+\Delta) = v]
$$

Here $\Omega_I$ is the set of valid grid positions of image $I$ and $Z_{\mathcal{S},\Delta}$ the total number of valid pairs, so each table sums to 1. The paper then symmetrizes each table, $\tilde{h}(u,v) = \tfrac{1}{2}(h(u,v) + h(v,u))$, averages over the displacements to get $\bar{h}^{(2)}_{\mathcal{S}}$, and applies the same Hellinger distance to the flattened tables:

$$
\text{CHD-2D}(\mathcal{R}, \mathcal{G}) = \frac{1}{\sqrt{2}} \left\| \sqrt{\mathrm{vec}(\bar{h}^{(2)}_{\mathcal{R}})} - \sqrt{\mathrm{vec}(\bar{h}^{(2)}_{\mathcal{G}})} \right\|_2
$$

A full table would have 4,096 squared, about 16.8 million cells, so only pairs that actually occur are stored. The final metric is the plain average, $\text{CHD} = \tfrac{1}{2}(\text{CHD-1D} + \text{CHD-2D})$.

**A worked example.** Take a 4 by 4 token grid: two rows of sky (S), one of horizon (H), one of grass (G). The generated grid is identical except for one grass token in the top row.

![Real and generated 4 by 4 token grids differing in one cell, with right and down neighbour pairs counted into symmetric S, H, G pair-count tables; the generated table gains an S-G pair never seen in real data](/imgs/blogs/evaluating-generative-models-via-one-dimensional-code-distributions-3.webp)

A 4 by 4 grid has 24 neighbour pairs (12 rightward, 12 downward). In the real grid, S-S pairs number 10 and S-G pairs number 0. The single misplaced token breaks two S-S right pairs and one S-S down pair, and creates three S-G pairs. On the unigram side, only 1 of 16 tokens changed: S drops from 8 to 7 and G rises from 4 to 5. In the pair table a cell that was exactly zero becomes 3 of 24. That is the asymmetry the authors want: CHD-1D barely moves, CHD-2D lights up.

**When it fails, and an ambiguity.** TiTok's 128 tokens are a 1D sequence, not a patch grid, which is the whole point of the "1D" in the title. The paper itself notes that adjacency along the sequence "imposes an artificial order", then defines CHD-2D on positions $\mathbf{p} = (x, y)$ without saying how 128 sequence positions map to a grid. I read it as reshaping the sequence (for example to 8 by 16), but that is my guess. Whatever "neighbour" means here, it is a learned-slot neighbour, not a spatial one. The ablation still shows the combination helps: N-MSE on AGIQA is 0.128 for CHD-1D alone, 0.135 for CHD-2D alone, and 0.112 for both.

## CMMS: a quality judge trained on fake damage

**The problem.** CHD compares two *sets*. Practitioners also need to score a *single* image, with no reference, for filtering, ranking or reward modeling. Learned IQA models do this but need large human-labeled datasets and drift on new styles.

**The analogy.** A proofreader who has never seen your manuscript still flags typos, because they know what well-formed text looks like. CMMS learns that from clean and deliberately damaged token sequences.

**The mechanism.** Start from tokenized ImageNet-1K images (1.28M of them), and corrupt each token independently:

$$
\tilde{c}_i \sim \begin{cases} c_i & \text{with probability } 1 - p, \\ \mathcal{U}(\mathcal{V}) & \text{with probability } p, \end{cases}
$$

where $\mathcal{U}(\mathcal{V})$ is the uniform distribution over the 4,096 codes and $p$ the corruption rate. Uniform injection mimics local glitches. Two more degradations add structure: **semantic fragment swapping** exchanges contiguous token blocks between images or distant regions (misplaced parts, broken limbs, repeated fragments), and **pixel-space augmentation** applies blur, JPEG, noise, occlusion and photometric changes before tokenization.

![Figure 3 from Jia et al. (2026): natural images progressively corrupted from 5% to 30%, the self-supervised training signal for CMMS](/imgs/blogs/evaluating-generative-models-via-one-dimensional-code-distributions-fig3.webp)

Each corrupted sequence gets a target quality computed from its severity:

$$
q(p) = \exp(-20p), \qquad p \in [0, 0.3]
$$

The regressor is small: 512-dimensional token embeddings, a 2-layer Transformer encoder with 8 heads, average pooling to $\mathbf{g} \in \mathbb{R}^{512}$, and a 2-layer MLP to $\hat{q} \in [0, 1]$. Training takes under 24 hours; inference exceeds 1,000 images per second.

**A worked example of the target.** The constant 20 makes the target steep. At $p = 0.05$, about six of 128 tokens replaced, the target is already 0.37. At $p = 0.10$ it is 0.14, and at the top of the range, $p = 0.3$, it is 0.002.

![The CMMS target exp(-20p) against linear and cubic alternatives over p from 0 to 0.3, with the HPDv2 preference accuracy of each choice](/imgs/blogs/evaluating-generative-models-via-one-dimensional-code-distributions-4.webp)

The authors justify the shape by human vision: small damage to a clean image is very noticeable, more damage to a broken one matters less. The bar chart on the right shows the ablation: on HPDv2 the linear target ${1 - p}$ scores 71.2% preference accuracy, $(1-p)^3$ scores 72.8%, and $\exp(-20p)$ peaks at 74.9% before $\exp(-30p)$ drops back to 74.1%.

**When it fails.** CMMS has seen only synthetic damage. Its labels encode "how far from ImageNet-clean", not "how much a person likes it". A generator that produces clean, natural token sequences with a wrong composition or an off-prompt subject has no corruption for CMMS to detect. Also note that choosing the constant against human ratings means CMMS is not fully label-free: a little human signal enters through hyperparameter selection.

## VisForm and the evaluation setup

VisForm is the authors' stress test: 210,000 images over 62 visual forms (portraits, product photos, watercolor, anime, 3D renders, medical imagery, scientific diagrams, UI) from 12 generators spanning diffusion, consistency and autoregressive models. Each image is rated on 14 perceptual dimensions by at least three expert annotators, with Kendall's W above 0.75. CMMS is never trained or tuned on it.

The other benchmarks are AGIQA (2,982 images), HPDv2 (430,060 images) and HPDv3 (1.08M text-image pairs). Agreement is Spearman, Kendall, normalized MSE, and pairwise accuracy (how often the metric picks the human's winner).


## Results: what moved and by how much

Model-level correlation with human mean scores, from Tables 1 and 2:

| Metric | AGIQA Spearman | AGIQA N-MSE | HPDv3 Spearman | HPDv3 Kendall |
| --- | --- | --- | --- | --- |
| FID | 0.771 | 0.119 | 0.648 | 0.467 |
| CLIP-FID | 0.714 | 0.170 | 0.491 | 0.378 |
| DINO-FID | 0.657 | 0.135 | 0.782 | 0.556 |
| CMMD | 0.657 | 0.142 | 0.527 | 0.467 |
| **CHD** | **0.829** | **0.112** | **0.867** | **0.778** |
| QualiCLIP | 0.771 | 0.122 | 0.503 | 0.422 |
| DeQA-Score | 0.886 | 0.118 | 0.836 | 0.689 |
| **CMMS** | **0.943** | **0.050** | **0.872** | **0.778** |

Pairwise preference accuracy, from Table 3:

| Model | AGIQA | HPDv2 | HPDv3 | VisForm |
| --- | --- | --- | --- | --- |
| CLIPScore | 63.4 | 65.1 | 48.6 | 58.2 |
| QualiCLIP | 66.7 | 68.6 | 51.2 | 61.3 |
| MDIQA | 66.3 | 70.1 | 51.1 | 64.5 |
| DeQA-Score | 68.7 | 70.6 | 52.7 | 63.1 |
| **CMMS** | **71.5** | **74.9** | **61.3** | **66.7** |

The HPDv3 gap is the largest: 61.3% against 52.7% for DeQA-Score, which on that benchmark is barely above a coin flip. On VisForm, Figure 4 shows the cross-domain picture.

![Figure 4 from Jia et al. (2026): metric-human correlation on VisForm across 12 generators (left) and 21 visual domains (right), all metrics normalized to 0 to 1](/imgs/blogs/evaluating-generative-models-via-one-dimensional-code-distributions-fig4.webp)

CHD averages Spearman/Kendall of 0.93/0.89 across models and 0.87/0.73 across domains, while FID drops on non-photographic domains such as sketches and collages. The practical surprise is sample efficiency.

![Figure 5 from Jia et al. (2026): mean FID and CHD versus number of samples; CHD flattens near 1,000 images while FID keeps drifting past 10,000](/imgs/blogs/evaluating-generative-models-via-one-dimensional-code-distributions-fig5.webp)

FID is biased by sample count, so its curve keeps sliding past 10,000 images; CHD flattens around 1,000. For anyone evaluating an expensive model, a 10 times smaller sample budget is the most immediately useful result here.

The ablations that matter most:

| Choice | Setting | AGIQA | VisForm |
| --- | --- | --- | --- |
| Tokenizer (CHD N-MSE, lower is better) | VQ-VAE, 2D | 0.268 | 0.147 |
| | VQGAN, 2D | 0.245 | 0.139 |
| | TiTok, 1D | 0.112 | 0.024 |
| Tokens per image | 32 / 64 / 128 | 0.157 / 0.140 / 0.112 | 0.050 / 0.034 / 0.024 |
| CMMS input (accuracy, higher is better) | Pixels | 67.8 | 62.2 |
| | Tokens | 71.5 | 66.7 |
| CMMS training | Token corruption only | 69.7 | 61.4 |
| | Both degradation families | 71.5 | 66.7 |

The 1D-versus-2D tokenizer gap is large: more than halving N-MSE on AGIQA and cutting it about sixfold on VisForm. Codebook size saturates at 4,096 (8,192 barely helps), and 512 by 512 input is slightly worse than 256 by 256.

**What is load-bearing and may not transfer.** The tokenizer is not the public TiTok: the authors retrained it on 100M DataComp images (214 hours on 8 A100s), and that tokenizer is the metric. Its 100M-image training set defines which images look "natural". Change the tokenizer and every CHD value changes, so cross-paper comparisons are only as stable as agreement on one checkpoint.

## Critique

**What is strong.** The framing is clean and testable: reconstruction-trained codes keep appearance, recognition features discard it. CHD is cheap and sample-efficient, and CMMS reaching state-of-the-art preference accuracy without human labels is genuinely surprising.

**Small model counts.** The distribution-metric correlations rank 6 models (AGIQA) or 10 (HPDv3). With 6 items, Kendall's tau moves in steps of 2/15, so CHD's 0.733 against FID's 0.600 is a difference of one discordant pair. No confidence intervals are reported. The HPDv3 numbers are firmer: recomputing Spearman from the Table 2 columns gives 0.867 for CHD, matching the paper.

**A reproduction gap.** For AGIQA, recomputing Spearman from the six per-model values printed in Table 1 gives about 0.54 for CHD and 0.26 for FID, not the reported 0.829 and 0.771. CMMS reproduces exactly (0.943). Most likely the distribution-metric correlations were computed at a finer granularity than the per-model table shows, but the paper does not say. My recomputation is a flag for the reader, not a refutation; it is the first thing I would ask the authors about.

**Memorization is invisible.** A generator that returns its training images verbatim achieves near-zero CHD against held-in data. This is not unique to CHD (FID has the same blind spot), but it matters as generators grow, and it is exactly the failure that [the weight-generation memorization study](/blog/paper-reading/computer-vision/generative-modeling-of-weights-generalization-or-memorization) found hiding behind good-looking metrics in another domain.

**Shared-tokenizer bias.** Many modern generators are trained on discrete tokens themselves. The paper does not test whether CHD favors generators whose tokenizer resembles the evaluator's.

**What would change my mind.** If an independent group, using the released tokenizer, re-ran AGIQA and HPDv3 with bootstrap confidence intervals and CHD's advantage over DINO-FID and FID survived, I would treat CHD as a default second metric next to FID. If CHD instead ranked token-based generators systematically above diffusion models of equal human preference, I would treat it as a tokenizer-dependent artifact.

## Where this goes next

The authors name video and 3D as future work. Token statistics are cheap to compute and easy to attribute, which opens several paper-sized questions.

1. **Token-statistics metrics for video.** *Question:* does CHD with temporal co-occurrence (token $u$ at frame $t$, token $v$ at frame $t+1$) catch flicker and identity drift that FVD misses? *Why now:* 1D video tokenizers exist, and CHD-2D's pair-counting machinery generalizes by adding a time displacement to $\mathcal{D}$. *First experiment:* apply frame-shuffling and temporal jitter at 10 severities to real clips, and check that a temporal CHD rises monotonically while per-frame CHD stays flat, mirroring Figure 2. ([What FVD already gets wrong](/blog/machine-learning/video-generation/the-metrics-of-video-generation).)

2. **CMMS as a training reward.** *Question:* can CMMS replace human-preference reward models in RL fine-tuning of diffusion models, and how fast does it get hacked? *Why now:* it runs at over 1,000 images per second and needs no labels, so it is a cheap reward. *First experiment:* fine-tune a small diffusion model with CMMS as the only reward, track it against an independent judge (DeQA-Score plus a small human study), and record when they diverge.

3. **3D generation through rendered-view tokens.** *Question:* is the CHD of multi-view renders a better proxy for 3D asset quality than CLIP-based scores? *Why now:* the metric needs only images, and the paper shows it holds up on non-photographic domains in VisForm. *First experiment:* render 3D assets from fixed camera rings, compute per-view CHD and a cross-view co-occurrence term, and correlate with existing human 3D quality ratings.

4. **Interpretable failure diagnosis.** *Question:* which specific tokens drive a high CHD, and do they decode to recognizable artifacts? *Why now:* Figure 2 shows only a small subset of tokens shifts under degradation, and a histogram is directly attributable per bin. *First experiment:* rank tokens by their Hellinger contribution for a weak generator, decode the top 20, and test whether people can name the artifact.

5. **Tokenizer invariance.** *Question:* how much do CHD rankings change across tokenizers (TiTok, Instella-T2I, a video tokenizer)? *Why now:* the ablation shows Instella-T2I and TiTok give nearly the same N-MSE, which hints at a tokenizer-independent core. *First experiment:* rank HPDv3 models under three tokenizers and report their Kendall agreement. High means the metric is about the images; low means it is about the tokenizer.

## References

- Jia, Luo, Zhong, Zhang, Zhou. "Evaluating Generative Models via One-Dimensional Code Distributions." CVPR 2026 (Highlight). [arXiv:2603.08064](https://arxiv.org/abs/2603.08064). CVPR page: [cvpr.thecvf.com/virtual/2026/poster/39614](https://cvpr.thecvf.com/virtual/2026/poster/39614).
- Code: [github.com/zexiJia/1d-Distance](https://github.com/zexiJia/1d-Distance) (announced in the paper).
- Yu et al. "An Image is Worth 32 Tokens for Reconstruction and Generation" (TiTok). NeurIPS 2024.
- Heusel et al. "GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium" (FID). NeurIPS 2017.
- Jayasumana et al. "Rethinking FID: Towards a Better Evaluation Metric for Image Generation" (CMMD). CVPR 2024.
- On this site: [Evaluating image generation honestly](/blog/machine-learning/image-generation/evaluating-image-generation-honestly), [The metrics of video generation](/blog/machine-learning/video-generation/the-metrics-of-video-generation), [Generative modeling of weights: generalization or memorization?](/blog/paper-reading/computer-vision/generative-modeling-of-weights-generalization-or-memorization).
