---
title: "Consensus vs. Controversy: Mapping the Decision Space Where Architectures Diverge"
publishDate: "2026-10-16"
date: "2026-10-16"
description: "How a CVPR 2026 highlight splits ImageNet into the images 12 pretrained CNNs, ViTs and MLP-Mixers agree on and the 10% they fight over, what its Controversy Score and ADER metrics really measure, and where the evidence is thinner than the headline."
category: "paper-reading"
subcategory: "Computer Vision"
tags: ["interpretability-analysis", "cvpr-2026", "cvpr-2026-highlight", "model-disagreement", "inductive-bias", "vision-transformer", "mlp-mixer", "ensembles", "calibration"]
author: "Hiep Tran"
featured: true
readTime: 18
image: "/imgs/blogs/consensus-vs-controversy-mapping-the-decision-space-where-architectures-diverge-fig1.webp"
excerpt: "Twelve ImageNet classifiers from three architecture families land within a few points of each other on accuracy. This paper asks on which images they actually differ, and finds the divergence packed into a thin tail."
paper:
  title: "Consensus vs. Controversy: Mapping the Decision Space Where Architectures Diverge"
  authors: "Minhyeok Lee"
  venue: "CVPR 2026 (highlight)"
  url: "https://openaccess.thecvf.com/content/CVPR2026/html/Lee_Consensus_vs._Controversy_Mapping_the_Decision_Space_Where_Architectures_Diverge_CVPR_2026_paper.html"
---

> [!tldr]
> **Claim.** CNNs, Vision Transformers and MLP-Mixers reach similar ImageNet accuracy, but they do not make the same decisions. Score every validation image by how much 12 pretrained models disagree in confidence, and the top 10% shows about 4.5 times the disagreement of the bottom 10% (Controversy Score 4.46).
>
> **Why it matters.** The disagreement has structure. Models agree more inside their own family (83.5%) than across families (80.2%), and an unsupervised clustering of models by agreement recovers the families without being told. That tail is where architecture choice, and ensemble choice, actually matters.
>
> **Most surprising result.** Isotonic calibration makes the separation bigger, not smaller: the Controversy Score jumps from 4.46 to 18.65. The split is not just a miscalibration artifact.
>
> **Where it is thin.** The "disagreement" score is the spread of each model's top confidence, not a vote over labels. Its correlation with a label-based alternative is 0.012. And the headline cascade (83.7% at 1.18 calls) is still below the best single model alone (84.0% at one call).

Two models at 82% accuracy can be wrong on almost the same 18% of images or on almost disjoint sets. Aggregate accuracy cannot tell you which.

This single-author CVPR 2026 highlight from Minhyeok Lee (Chung-Ang University) goes looking for the images that separate architectures. It trains nothing. It runs 12 off-the-shelf `timm` models over the 50,000 ImageNet validation images, scores each image by how much the models disagree, and studies the two ends of that ranking. It is a measurement protocol, so the interesting part is reading the measurements carefully.

## The setup: twelve models, three families, no training

The model zoo is deliberately ordinary. Everything is pretrained, loaded from `timm`, and run at 224×224 with standard ImageNet normalization, batch 64, FP16.

| Family | Models | Count |
| --- | --- | --- |
| CNN | ResNet-50, ResNet-101, Wide-ResNet-50-2, EfficientNet-B0, EfficientNet-B2, MobileNetV3-Large, ConvNeXt-Tiny | 7 |
| Vision Transformer | ViT-Base/16, ViT-Small/16, DeiT-Base | 3 |
| MLP-based | MLP-Mixer-B/16, gMLP-S16 | 2 |

Top-1 accuracy spans 72.6% to 84.0% with a mean of 79.9%. ConvNeXt-Tiny is the best single model at 84.0%. The families are unbalanced, seven CNNs against two MLPs, which matters later when we read family averages.

Meding et al. showed ImageNet is "dichotomous": about 46% of images are trivial (every model right) and 11.5% impossible (every model wrong). Conwell et al. found across 224 models that training diet matters more than architecture for matching brain data. Neither says *which images* separate architectures.

![Figure 1 from Lee (CVPR 2026): (a) per-model accuracy, (b) disagreement on the consensus vs controversial subsets, (c) the 12×12 pairwise agreement matrix](/imgs/blogs/consensus-vs-controversy-mapping-the-decision-space-where-architectures-diverge-fig1.webp)

## Technique 1: the disagreement score D

**The problem.** You need one number per image that says "these models do not see this the same way." The obvious candidate is a vote: how many of the 12 models pick the same top-1 class. The paper defines that, then uses something else.

**The analogy.** Picture twelve judges scoring a dive. A vote asks whether they all held up the same card. The paper's score asks how spread out their certainty is: did they all sound equally sure, or was one shouting and another mumbling?

**The mechanism.** Each model $f_m$ maps an image $x_i$ to a probability vector $p_i^{(m)}$ over $K$ classes. The paper first defines an agreement score from top-1 votes:

$$
A_i = \max_k \frac{1}{M} \sum_{m=1}^{M} \mathbb{1}\left[\hat{y}_i^{(m)} = k\right], \qquad \hat{y}_i^{(m)} = \arg\max_k p_{i,k}^{(m)}
$$

Here $M = 12$ is the number of models, $K = 1000$ the number of classes, $\hat{y}_i^{(m)}$ model $m$'s predicted class on image $i$, and $\mathbb{1}[\cdot]$ the indicator. $A_i$ lies in $[1/M, 1]$; higher means more models voted for the plurality class.

The splits, though, use a different quantity, the standard deviation of each model's maximum probability:

$$
D_i = \operatorname{std}\left(\left\{\max_k p_{i,k}^{(m)}\right\}_{m=1}^{M}\right)
$$

Every model contributes one scalar, its top confidence, and $D_i$ is the spread of those twelve scalars.

**Worked example.** Suppose on a dog photo all 12 models say "golden retriever" with confidences between 0.90 and 0.97. The spread is tiny, $D$ is around 0.02. On a palace photo, six models say "palace" at 0.9, three say "monastery" at 0.4, three say "church" at 0.3. The confidences now range from 0.3 to 0.9 and $D$ lands around 0.26. The vote score for the palace is $A = 6/12 = 0.5$.

![Two bar charts of the twelve models' top confidence on a consensus image and a controversial image](/imgs/blogs/consensus-vs-controversy-mapping-the-decision-space-where-architectures-diverge-1.webp)

The figure makes the key subtlety visible. $D$ is a property of the confidence profile, not of the labels.

**When it fails.** $D$ can be high when every model picks the same class, if some are sure and some are hesitant. The paper says so directly: high $D$ marks images where models "differ substantially in confidence, even when some models may still share the same top-1 label." It can also be low when models pick different classes at similar confidence. Keep this in mind: every later claim about "architectures diverging" is filtered through a confidence-spread lens.

## Technique 2: controversial and consensus sets, and the Controversy Score

**The problem.** A single score per image is a distribution over 50,000 numbers. You want a compact summary of how concentrated the disagreement is.

**The mechanism.** Sort images by $D_i$. The top $\alpha\%$ is the controversial set $C_{\text{contro}}$, the bottom $\alpha\%$ the consensus set $C_{\text{cons}}$, with $\alpha = 10$ throughout. The Controversy Score is the ratio of their mean disagreements:

$$
\mathrm{CS} = \frac{\operatorname{mean}\left(D_i : i \in C_{\text{contro}}\right)}{\operatorname{mean}\left(D_i : i \in C_{\text{cons}}\right)}
$$

**The numbers.** Over the whole set, $D$ has mean 0.106 and standard deviation 0.056. The controversial set has mean 0.255, the consensus set 0.057, so CS = 0.255 / 0.057 = 4.46. A two-sample t-test gives $p \lt 0.001$. Within each set, the spread of $D$ is 0.024 for controversial and 0.005 for consensus images.

![A right-skewed density curve of D with the consensus band at the left and the controversial tail at the right](/imgs/blogs/consensus-vs-controversy-mapping-the-decision-space-where-architectures-diverge-2.webp)

The curve above is my illustrative fit to the paper's three summary numbers, not their histogram. It shows why a ratio of decile means is the right shape of summary for a right-skewed distribution: most mass sits in a band around 0.05 to 0.10, and a thin tail stretches past 0.30.

**When it fails.** CS is a ratio of the extremes of one sorted list. Any continuous score with a right skew produces a ratio well above 1 between its top and bottom deciles, including pure noise with a skewed distribution. So the 4.46 alone does not prove structure. The p-value has the same problem: the two sets are selected to be different, so a t-test between them is close to guaranteed to fire. The evidence for structure comes from later sections, not from CS itself. To its credit, the paper does supply that later evidence.

The paper's sensitivity check (Figure 6b) varies $\alpha$ over 5, 10, 15 and 20%. CS stays between 4.4 and 4.8, peaking at 10% (4.46). One oddity: for a skewed distribution you would expect the ratio to rise as $\alpha$ shrinks toward the extremes, and the paper does not explain why it peaks at 10.

## Technique 3: family structure in the agreement matrix

**The problem.** Disagreement could be noise spread evenly over model pairs. If inductive bias matters, models with the same bias should disagree less with each other.

**The mechanism.** Pairwise top-1 agreement between models $m_1$ and $m_2$ is the fraction of images where they predict the same class:

$$
\mathrm{Agr}(m_1, m_2) = \frac{1}{N} \sum_{i=1}^{N} \mathbb{1}\left[\hat{y}_i^{(m_1)} = \hat{y}_i^{(m_2)}\right]
$$

with $N = 50{,}000$. Average this over pairs within a family (intra) and across families (inter).

**What they find.**

| Pair or group | Agreement |
| --- | --- |
| ResNet variants with each other | above 0.88 |
| EfficientNets and MobileNet with ResNets | 84–87% |
| ViT-Base with DeiT-Base | 93.4% |
| ViT-Small with other ViTs | 81.6% average |
| CNN to ViT, cross-family | 84.2% average |
| MLP-Mixer with all others | 78.8% average |
| gMLP with all others | 83.5% average |
| ConvNeXt-Tiny with all others | 85.3% average, the highest |
| **All intra-family pairs** | **83.5%** |
| **All inter-family pairs** | **80.2%** |

The 3.3-point gap is the paper's central evidence for "architectural signatures." Two details make it more convincing than a raw average. MLP-Mixer's low agreement is not explained by low accuracy, since gMLP at 77.2% is comparable to several CNNs. And ConvNeXt, a CNN redesigned with transformer-era choices, agrees most with everyone, which is what you would predict if family signatures come from design.

![Figure 3 from Lee (CVPR 2026): (a) the disagreement distribution, (b) diversity vs disagreement, (c) intra-family vs inter-family agreement](/imgs/blogs/consensus-vs-controversy-mapping-the-decision-space-where-architectures-diverge-fig3.webp)

The strongest single piece of evidence is in Figure 6a. Cluster the 12 models hierarchically using only their pairwise agreement, with no architecture labels. ResNets merge first at distance below 0.10, ViTs form their own branch around 0.15, and MLPs join last above 0.20. The families fall out of behavior alone.

**When it fails.** Two confounds sit inside the family labels. First, training recipe: the ViTs and DeiT come with heavy augmentation and distillation recipes that the 2016-era ResNets do not share, so "family" partly means "training era." The paper's own discussion leans on Conwell et al., who found training diet dominates. Second, scale: with only 3 ViTs and 2 MLPs, one small model (ViT-Small) moves a family average a lot.

## Technique 4: diversity and the r = 0.82 correlation

**The problem.** $D$ measures confidence spread. You also want to know whether models predict *different classes*, which is what an ensemble needs.

**The mechanism.** Prediction diversity for image $i$ is the fraction of unique top-1 classes among the 12 models. With 12 models, if all agree diversity is 1/12; if all differ it is 1.

**The numbers.** Diversity has mean 0.117 over the dataset. The controversial set averages 0.185, the consensus set 0.045, a ratio above 4 that mirrors the CS. Across all images, diversity and $D$ correlate at $r = 0.82$.

**Worked example.** On the palace image above, three distinct classes among twelve models gives diversity 3/12 = 0.25. On the retriever, one class gives 1/12, about 0.083.

**Read carefully.** A diversity of 0.045 is below 1/12 = 0.083, the floor for "all 12 agree." That cannot happen with the definition as stated, so the reported diversity must be normalized differently than the text says, perhaps as (unique − 1)/(M − 1) or similar. This is my inference from the arithmetic; the main paper does not specify. It does not change the ranking, but it means the absolute numbers should not be decoded with the stated formula.

The paper also reports a sharp caveat in its robustness section. A label-aware alternative, top-5 Jaccard disagreement (mean pairwise dissimilarity of top-5 sets), gives a CS of 3.05 but correlates with standard $D$ at only $r = 0.012$. The paper reads this as "complementary aspects." I read it as a warning: two reasonable definitions of "the models disagree" pick out nearly unrelated images. Which images count as controversial depends heavily on the metric.

## Technique 5: ADER, decoding the 24.6

**The problem.** Is inter-family disagreement concentrated in the controversial set, or spread evenly?

**The mechanism.** The supplement defines a family-pair ratio. For families $F_a$ and $F_b$:

$$
\mathrm{ADER}(F_a, F_b) = \frac{d_{\text{cont}}(F_a, F_b) \,/\, d_{\text{all}}(F_a, F_b)}{\text{share}_{\text{cont}}}
$$

$d_{\text{cont}}$ is the disagreement rate between the two families on controversial samples, $d_{\text{all}}$ the rate on all samples, and $\text{share}_{\text{cont}} = |C_{\text{cont}}|/N = 0.10$.

**Worked decode.** The main text gives ADER = 24.6 and describes it as inter-family disagreements accounting for 24.6 times more disagreement share than expected from the 10% data share. If $d$ were a count, the numerator would be a share of disagreements, at most 1, and ADER could not exceed 10. So $d$ must be a rate, and 24.6 means the inter-family disagreement rate on controversial images is 24.6 × 0.10 = 2.46 times the overall rate.

![A 10x10 grid of images with the controversial tenth highlighted, and bars comparing the disagreement rate on all images to the rate on the controversial set](/imgs/blogs/consensus-vs-controversy-mapping-the-decision-space-where-architectures-diverge-3.webp)

That is still a real effect: on the controversial tenth, families disagree about two and a half times as often as on average. But dividing a rate ratio by 0.10 inflates it tenfold, and "24.6×" reads much stronger than "2.46×." This decode is mine, from the supplement's equation; check it against the code if it is released.

![Figure 5 from Lee (CVPR 2026): (a) mean diversity by disagreement quantile, (b) per-model expected calibration error, (c) Jaccard overlap of controversy sets found by model subsets](/imgs/blogs/consensus-vs-controversy-mapping-the-decision-space-where-architectures-diverge-fig5.webp)

A small note on the figure above. The paper's caption says panel (a) shows ADER, but the panel actually plots mean diversity across disagreement quantiles. The ECE panel and the Jaccard panel match their captions.

## Technique 6: calibration does not dissolve the controversy

**The problem.** If some models are simply overconfident, their max probabilities sit near 1 regardless of the image, and $D$ would measure who is miscalibrated rather than where architectures differ.

**The mechanism.** Fit isotonic regression per model to map raw confidence to empirical accuracy, then recompute $D$ and the splits. Expected Calibration Error (ECE) bins predictions by confidence and averages the gap between confidence and accuracy per bin.

| Metric | Before calibration | After calibration |
| --- | --- | --- |
| Controversy Score | 4.46 | 18.65 |
| Mean ECE | 0.089 | 0.044 |
| Corr. with top-5 Jaccard | 0.012 | −0.004 |

Calibration halves the mean ECE and quadruples the CS. Per model, ViT-Small has the lowest ECE (0.020) and ResNet-101 the highest (0.155), even though ResNet-101 is more accurate (81.9% vs 74.6%). The paper reads this as transformers being generally better calibrated in this zoo.

**Why it could go up.** Isotonic maps are monotone step functions. On consensus images, raw confidences cluster near the top, and calibration maps them to nearly the same high value, so consensus $D$ shrinks toward zero. Since the bottom-decile mean is the denominator, CS can balloon from a shrinking denominator alone. This explanation is mine, not the paper's. It still supports their point that miscalibration does not create the split, but 18.65 is not "four times more divergence."

The supplement adds two useful separations. Controversy is not difficulty: only 9.7% of consensus-by-D images are trivial (all correct) and only 1.0% of controversial images are impossible (all wrong). And it is not label noise: of 4.54% flagged label-error candidates, only 1.0% overlap the controversial set. The 9.7% figure is striking given that Meding et al. put 46% of ImageNet in the trivial bucket, which suggests (my inference) that low $D$ often means "everyone is equally unsure" rather than "everyone is sure."

## Technique 7: how many models do you need, and sequential gating

**Model count.** If a few models were enough to find the controversial set, the 12-model zoo would be overkill. Figure 5c compares controversy sets found by model subsets to the full 12-model set using Jaccard overlap: 3 models give 31.7%, 4 give 35.7%, 6 give 47.8%, 8 give 57.1%. Half the zoo finds less than half of the set. The paper reads this as diversity contributing unique views. It also means the controversial set is a property of *this* zoo: swap three models and you get a substantially different set.

**Sequential gating.** The practical demo is a three-stage cascade: ConvNeXt-Tiny first, escalate to ResNet-50 when confidence is below a threshold $\tau$, then to ViT-Base.

| Threshold τ | Avg. calls | Accuracy | ECE |
| --- | --- | --- | --- |
| 0.50 | 1.18 | 83.7% | 0.028 |
| 0.60 | 1.27 | 83.5% | 0.032 |
| 0.70 | 1.38 | 83.2% | 0.038 |
| 0.80 | 1.53 | 82.9% | 0.037 |
| 0.90 | 1.96 | 82.5% | 0.033 |

The "90% compute reduction" compares 1.18 calls against running all 12 models. The comparison that matters for deployment is against ConvNeXt-Tiny alone: one call, 84.0%.

![Accuracy vs average model calls for the five cascade thresholds, with a reference line for ConvNeXt-Tiny alone](/imgs/blogs/consensus-vs-controversy-mapping-the-decision-space-where-architectures-diverge-4.webp)

Every point of the cascade sits below and to the right of the single best model: more compute, lower accuracy. The paper notes the 0.3-point loss "vs. best single model," so the number is honest, but the framing as a cost-accuracy win does not hold. The cascade also gets *worse* as it escalates more, which says the fallback models are overriding correct ConvNeXt answers. Its real value is calibration: ECE 0.028 at τ = 0.50, which is good if you need trustworthy confidences. Calls are also not equal cost: ViT-Base/16 needs roughly four times the FLOPs of ConvNeXt-Tiny (my figure from the public model cards, not the paper).

## Results across datasets and classes

![Figure 6 from Lee (CVPR 2026): (a) clustering of models by agreement, (b) Controversy Score across thresholds, (c) the 20 most controversial ImageNet classes](/imgs/blogs/consensus-vs-controversy-mapping-the-decision-space-where-architectures-diverge-fig6.webp)

**Class concentration.** The top-20 controversial classes put 30–42% of their images in the controversial set, against a 10% base rate. Palace (740), steel drum (830), hook (600) and lipstick (626) lead. The paper groups them as fine-grained, unusual-viewpoint or ambiguous categories.

**Out-of-distribution.** The full pipeline reruns on ImageNet-A and ImageNet-R:

| Dataset | CS at 10% | ADER at 10% | Mean D, consensus → controversial |
| --- | --- | --- | --- |
| ImageNet-val | 4.46 | 24.6 | 0.057 → 0.255 |
| ImageNet-A | 2.89 | 9.82 | 0.098 → 0.283 |
| ImageNet-R | 3.34 | 10.14 | 0.088 → 0.294 |

The tail still exists under shift, but both summaries shrink. The mechanism is visible in the last column: the *consensus* floor rises (0.057 to about 0.09), because under shift nobody is calmly confident. Decoding ADER as before, the inter-family disagreement rate on the controversial set is about 1.0 times the overall rate on both OOD sets. In plain terms, under distribution shift the families disagree everywhere, and the controversial tenth no longer concentrates it. That reading is mine, and it runs against the paper's "phenomenon persists" framing.

**What is load-bearing and may not transfer.** The zoo is ImageNet-1k supervised `timm` models only: no CLIP, DINOv2 or SigLIP, no dense tasks. Large-scale pretraining may erase family signatures, as Conwell et al.'s training-diet result predicts.

## Critique

**What is strong.**

- The question is the right one, and the protocol is cheap enough that anyone with a GPU can rerun it.
- The clustering result is genuinely convincing: families emerge from behavior alone.
- The paper checks the obvious confounds it can: calibration, label noise, difficulty, threshold choice, OOD.

**What is weak.**

- **The metric does not match the story.** The prose talks about models predicting different classes; $D$ measures confidence spread. The label-aware alternative correlates at 0.012. Every downstream claim about "where architectures diverge" inherits this ambiguity.
- **Ratios of extremes.** CS and ADER are ratios over a selected tail, which flatters any skewed score. A null baseline is missing: shuffle the model-to-family assignment, or replace each model's confidence with a permuted copy, and report the CS and ADER that noise gives.
- **No look at the images.** For a paper about where inductive bias matters, it never characterizes the controversial images (object size, texture/shape cue conflict, crop, clutter) or which family is *right* on them. The limitations section names this.
- **Unbalanced families.** Seven CNNs, three ViTs, two MLPs. Family averages over two models are fragile, and family is confounded with training recipe and era.
- **The deployment demo underperforms the trivial baseline**, as shown above.

**Missing ablation.** Retrain two architectures with an identical recipe (for example, ConvNeXt and DeiT, both from the DeiT recipe) and check whether the intra/inter gap survives. That is the experiment that would separate architecture from training diet.

**What would change my mind.** If a family-shuffled null gave a CS near 1 and an intra/inter gap near zero, while the real assignment kept 4.46 and 3.3 points, I would accept that $D$ is tracking architecture rather than skew. If the controversial set under a label-based metric (vote entropy) overlapped heavily with the $D$-based set, I would drop my main objection to the metric.

## Where this goes next

The orchestrator's note for this paper was: use controversial-image sets to guide architecture ensembling and targeted data curation. The paper makes both tractable because it shows the signal is cheap to compute and concentrated. Here are five concrete directions.

1. **Controversy-selected ensembles.** *Question:* does choosing ensemble members by complementarity on the controversial set beat choosing by accuracy? *Why now:* the paper shows family complementarity concentrates in a small, cheap-to-identify subset. *First experiment:* split ImageNet-val in half; on half A, greedily add models from a larger `timm` zoo to maximize ensemble accuracy on A's controversial set; compare against top-k-by-accuracy on half B, at matched FLOPs.

2. **A label-based controversy metric, and a test of which one predicts ensemble gain.** *Question:* which definition (confidence spread $D$, vote entropy, mean pairwise Jensen-Shannon divergence) best predicts where a two-model ensemble beats its best member? *Why now:* the 0.012 correlation between $D$ and top-5 Jaccard shows the choice is not cosmetic. *First experiment:* compute all three per image; for each pair of models, regress per-image ensemble gain on each metric and compare explained variance.

3. **Targeted data curation from the controversial set.** *Question:* does fine-tuning on images near the controversial set close more of the family gap than an equal-size random set? *Why now:* controversy is concentrated in classes (palace, steel drum, hook) and not explained by label noise, so it points at under-covered regions of the data. *First experiment:* embed ImageNet-train with a frozen backbone, retrieve nearest neighbors of controversial val images, fine-tune ResNet-50 and DeiT-Base on 50k retrieved vs 50k random images, and measure intra/inter agreement and accuracy on the controversial set.

4. **Attribute the controversy to visual properties.** *Question:* on controversial images, which family is right, and what property predicts it? *Why now:* the paper names this as future work and the subset is small enough to annotate. *First experiment:* bucket controversial images by "CNNs right, ViTs wrong" and the reverse; measure object box size, Stylized-ImageNet-style texture/shape conflict, and occlusion per bucket; test whether CNN-wins skew to texture cues as Geirhos et al. would predict.

5. **Does the controversy space survive foundation-model pretraining?** *Question:* do family signatures persist when every model sees the same billions of images? *Why now:* the protocol is training-free, so adding CLIP ViT, CLIP ConvNeXt, DINOv2 and SigLIP linear probes costs only inference. *First experiment:* rerun the pipeline with matched-data pairs (OpenCLIP ConvNeXt vs OpenCLIP ViT trained on the same LAION subset) and compare the intra/inter gap to the ImageNet-1k zoo.

## References

- Minhyeok Lee. *Consensus vs. Controversy: Mapping the Decision Space Where Architectures Diverge.* CVPR 2026 (highlight). [Paper (CVF open access)](https://openaccess.thecvf.com/content/CVPR2026/html/Lee_Consensus_vs._Controversy_Mapping_the_Decision_Space_Where_Architectures_Diverge_CVPR_2026_paper.html), [supplementary material](https://openaccess.thecvf.com/content/CVPR2026/supplemental/Lee_Consensus_vs._Controversy_CVPR_2026_supplemental.pdf), [poster page](https://cvpr.thecvf.com/virtual/2026/poster/37126). No arXiv version or code release was found at the time of writing.
- Kristof Meding, Luca M. Schulze Buschoff, Robert Geirhos, Felix Wichmann. *Trivial or Impossible: Dichotomous Data Difficulty Masks Model Differences (on ImageNet and Beyond).* [arXiv:2110.05922](https://arxiv.org/abs/2110.05922).
- Colin Conwell et al. *A large-scale examination of inductive biases shaping high-level visual representation in brains and machines.* Nature Communications 15, 2024.
- Robert Geirhos et al. *ImageNet-trained CNNs are biased towards texture; increasing shape bias improves accuracy and robustness.* [arXiv:1811.12231](https://arxiv.org/abs/1811.12231).
- Related on this site: [generative modeling of weights](/blog/paper-reading/computer-vision/generative-modeling-of-weights-generalization-or-memorization), [ViT, SigLIP and DINO explained](/blog/machine-learning/computer-vision/vit-siglip-dino-explained), [calibration and the prediction you can trust](/blog/machine-learning/recommendation-systems/calibration-and-the-prediction-you-can-trust).
