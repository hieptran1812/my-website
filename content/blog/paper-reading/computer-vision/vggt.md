---
title: "VGGT-Ω: Feed-Forward 3D Reconstruction Gets a Scaling Law"
publishDate: "2026-10-15"
date: "2026-10-15"
description: "How VGGT-Ω cuts VGGT's training memory by 70%, trains on 15x more data including dynamic video, and shows that feed-forward 3D reconstruction error falls predictably with model and data size."
category: "paper-reading"
subcategory: "Computer Vision"
tags: ["3d-reconstruction", "cvpr-2026", "cvpr-2026-award-candidate", "vggt", "vggt-omega", "register-attention", "scaling-laws", "dynamic-scenes", "self-supervised-learning", "feed-forward-reconstruction"]
author: "Hiep Tran"
featured: true
readTime: 22
image: "/imgs/blogs/vggt-fig1.webp"
excerpt: "VGGT-Ω is VGGT rebuilt to scale: register attention, one dense head, a dynamic-video data engine and teacher-student training. Point error falls like a power law from 0.2B to 10B parameters and from 2K to 2M training sequences."
paper:
  title: "VGGT-Ω"
  authors: "Jianyuan Wang, Minghao Chen, Shangzhan Zhang, Nikita Karaev, Johannes Schönberger, Patrick Labatut, Piotr Bojanowski, David Novotny, Andrea Vedaldi, Christian Rupprecht"
  venue: "CVPR 2026 (Oral, Award Candidate)"
  url: "https://arxiv.org/abs/2605.15195"
---

> [!tldr]
> **Claim.** Feed-forward 3D reconstruction behaves like other foundation models: error drops predictably as you add parameters (0.2B to 10B) and training sequences (2K to 2M).
>
> **How.** VGGT-Ω keeps the VGGT recipe but makes it cheap enough to scale. Register attention replaces 25% of the global attention layers. A single dense depth head replaces VGGT's several dense heads. A data engine turns 40M Internet videos into 0.8M clean sequences, a third of them dynamic. Training memory drops to about 30% of VGGT's.
>
> **Most surprising.** On Sintel, camera AUC@3° jumps from 22.5 (MegaSaM, an optimization method) to 40.0. That is a 77% relative gain, and the model runs about 50x faster.
>
> **Where it fails.** Strong motion blur, abrupt field-of-view changes, and heavily distorted lenses. Self-supervised training barely moves the benchmarks. The paper also does not output motion masks, scene flow or a temporally coherent 4D model.

## Why this paper matters

The original [VGGT](/blog/paper-reading/computer-vision/vggt-visual-geometry-grounded-transformer) showed that one transformer, run once, could predict cameras, depth, point maps and tracks for a set of images. It beat pipelines that end with bundle adjustment. VGGT-Ω asks the question every foundation-model field eventually asks: **what happens if we make it ten times bigger and feed it ten times more data?**

In language models, loss falls as a power law in parameters and tokens (see [Chinchilla's compute-optimal scaling](/blog/machine-learning/scaling-laws/chinchilla-compute-optimal-scaling)). In 3D vision nobody had shown this, mostly because VGGT's training memory and its hunger for clean 3D labels made the experiment unaffordable. So the paper is two papers: engineering that makes VGGT cheap to scale, then the scaling curve itself.

## What changed from VGGT, at a glance

If you know VGGT, this table is the whole diff. Each row gets its own section below.

| Component | VGGT (CVPR 2025) | VGGT-Ω (CVPR 2026) |
| --- | --- | --- |
| Backbone init | DINOv2, patch 14 | DINOv3, patch 16 (about 25% fewer tokens) |
| Cross-frame mixing | Global attention in every block | Global attention, with 25% of layers swapped for register attention |
| Registers | Per-frame registers, auxiliary | 16 "scene tokens" per frame, used downstream |
| Dense heads | Depth, point map, track heads (DPT) | One depth head; DPT convs above 1/4 resolution replaced by MLP + pixel shuffle |
| Point maps, tracks | Predicted | Still supervised as losses, not predicted |
| Camera head | Iterative refinement | Single pass |
| Camera loss | Huber | ℓ1 |
| Data | Roughly 15x fewer sequences, static | About 4M sequences, static and dynamic |
| Extra training | None | Teacher-student self-supervision on 18M unlabeled videos |
| Model sizes | 1.2B | 200M, 500M, 1B, 10B |

![Figure 2 from Wang et al. (2026): the VGGT-Ω architecture. Camera and scene tokens are appended to DINO patch tokens, then global (or register) attention alternates with frame attention. Matching and point losses exist only at training time.](/imgs/blogs/vggt-fig1.webp)

Each frame gets one camera token and sixteen scene tokens, then L blocks alternate a cross-frame layer with a within-frame layer. Only two heads survive at inference: cameras and depth.

## The problem setup

The model is a function $f$ from $N$ images to $N$ cameras and $N$ depth maps:

$$
\big((g_1, D_1), \ldots, (g_N, D_N)\big) = f(I_1, \ldots, I_N)
$$

Here $I_i \in \mathbb{R}^{3 \times H \times W}$ is image $i$, and $D_i \in \mathbb{R}^{H \times W}$ is its depth map. The camera $g_i = (q_i, t_i, f_i) \in \mathbb{R}^9$ stacks a rotation quaternion $q_i \in \mathbb{R}^4$, a translation $t_i \in \mathbb{R}^3$ and a field of view $f_i \in \mathbb{R}^2$. The principal point is assumed to sit at the image center.

Each image becomes $H'W'$ patch tokens with $H' = H/r$ and $W' = W/r$ for patch size $r$. One camera token and sixteen registers are appended, so the full token tensor is:

$$
z \in \mathbb{R}^{N \times (H'W' + 17) \times C}
$$

with $C$ the hidden width. For a 512×512 image and $r = 16$, that is ${1024}$ patch tokens plus 17 extras, so ${1041}$ tokens per frame. The first frame gets its own learnable camera and register embeddings, which marks it as the reference. No other token knows which frame it came from, so the model is permutation equivariant in the remaining frames.

## Technique 1: register attention

**The problem.** Global attention lets every token in every frame attend to every other token. Its cost is quadratic in the total token count, $(N \cdot 1041)^2$ pairs per layer at 512 px. It is the main compute bottleneck in VGGT.

**The observation.** Those attention maps are mostly empty. The paper visualizes layer 13 of VGGT and finds that nearly all the attention mass sits on a handful of entries.

![Figure 3 from Wang et al. (2026): global attention in VGGT, layer 13. The attention matrix, its value histogram and the spatial heatmaps all show that only a few entries carry weight.](/imgs/blogs/vggt-fig2.webp)

**The analogy.** Think of a conference with 24 rooms of 1,041 people each. Global attention is everyone calling everyone. Register attention elects 16 delegates per room. The delegates meet in one hall. Then each delegate goes home and briefs their room. Information still crosses rooms, but through a narrow channel.

**The mechanism.** In a register-attention layer, only the registers take part in self-attention across frames:

$$
(z_1^{\text{scene}\prime}, \ldots, z_N^{\text{scene}\prime}) = \text{attn}(z_1^{\text{scene}}, \ldots, z_N^{\text{scene}})
$$

where $z_i^{\text{scene}} \in \mathbb{R}^{16 \times C}$ are frame $i$'s registers and the prime marks the updated values. Patch tokens and camera tokens pass through untouched. In the next frame-attention layer, $\text{attn}_f(z) = (\text{attn}(z_1), \ldots, \text{attn}(z_N))$, each frame's patch tokens read their own updated registers. That is how the scene summary flows back to the pixels.

![Token layout for one frame and the attention pattern of global versus register attention: 24,984 tokens and about 624M pairs against 384 register tokens and about 147K pairs.](/imgs/blogs/vggt-1.webp)

**Worked example.** Take $N = 24$ frames at 512×512. Global attention runs over ${24 \times 1041 = 24{,}984}$ tokens, so about 624M query-key pairs. Register attention runs over ${16 \times 24 = 384}$ tokens, so about 147K pairs. That is roughly 4,200x fewer pairs in that layer.

**What it costs in accuracy.** Swapping 25% of the global layers gives the same accuracy within noise (point error 0.073 versus 0.071 for all-global) and saves about 23% of backbone FLOPs and 16% of backbone memory in training. Swapping all of them cuts FLOPs to 6% of the original. But then accuracy falls back to the level of the original VGGT.

**When it fails.** In the all-register variant, a patch token in one frame can never directly see a patch token in another. Fine-grained correspondence has to squeeze through 16 vectors per frame. The 25% ratio is a measured compromise, not a derived optimum, and the paper does not sweep it.

There is a second payoff. Because the registers are forced to carry the whole scene, they become a compact scene descriptor. The paper reuses them for robot policies and language alignment, covered later.

## Technique 2: one dense head and a memory-light decoder

**The problem.** VGGT predicted depth, point maps and tracks, each through its own Dense Prediction Transformer (DPT) head. The last DPT conv blocks hold several feature maps at high resolution. Their forward activations dominate GPU memory, even though their parameters are a small fraction of the model. FSDP and gradient checkpointing shard weights or recompute layers. They do not remove the need to hold these big activation tensors at some point.

**The fix, part one.** Keep only a depth head, plus a light camera head. Point maps and matching still appear, but only as training losses. A point map is just depth unprojected through the camera, so it can be supervised without being predicted.

**The fix, part two.** Inside the depth head, replace the conv blocks above 1/4 resolution with one MLP and a pixel-shuffle reshape. The MLP outputs ${2u^2}$ channels with $u = 4$, so 32 channels. Pixel shuffle rearranges a $(H' \times W', 2u^2)$ tensor into $(uH') \times (uW') \times 2$. The two output channels are depth and confidence.

![The memory-light depth head: a 128×128×32 feature map is reshaped by pixel shuffle into a 512×512×2 depth and confidence output, with no high-resolution convolutions.](/imgs/blogs/vggt-3.webp)

**Worked example.** One low-resolution position carries 32 numbers. Pixel shuffle reads them as a 4×4 patch of depth values (16 numbers) and a 4×4 patch of confidences (16 numbers). Do that at every position of a 128×128 map and you get a 512×512 depth map and a 512×512 confidence map. The reshape has no weights and stores no extra activations.

The paper writes the input as $H' \times W'$, its patch-grid symbol. Since only blocks above 1/4 resolution were replaced, I read the input as the 1/4-resolution DPT feature map, as drawn; that reading is mine.

**When it fails.** The authors tried going all the way: no convolutions, MLP only. It scored well on benchmarks and trained with more stable gradients. But it produced visible blocky artifacts, worst outdoors with distant structures like sky and mountains, where depth is effectively unbounded. They keep the early low-resolution DPT convs (at 16×16 or 32×32 feature resolution), which cost little and remove most of the blockiness. Together with the single head and register attention, the paper reports that these changes save **70% of GPU memory** during training.

### The multi-task loss

Even with one head, training is multi-task:

$$
\mathcal{L} = \lambda_{\text{cam}}\mathcal{L}_{\text{cam}} + \lambda_{\text{depth}}\mathcal{L}_{\text{depth}} + \lambda_{\text{point}}\mathcal{L}_{\text{point}} + \lambda_{\text{match}}\mathcal{L}_{\text{match}}
$$

The supplement sets $\lambda_{\text{cam}} = 5.0$, $\lambda_{\text{depth}} = 1.0$, $\lambda_{\text{point}} = 0.5$ and $\lambda_{\text{match}} = 0.1$.

- **Camera.** $\mathcal{L}_{\text{cam}} = \sum_i |\hat g_i - g_i|$, a plain ℓ1. The paper found it more stable than VGGT's Huber loss.
- **Depth.** An uncertainty-weighted residual plus a gradient term:

$$
\mathcal{L}_{\text{depth}} = \sum_{i=1}^{N}\Big[\big\|c_i^D \odot (1 + D_i^{-1}) \odot e_i\big\| + \big\|c_i^D \odot \nabla e_i\big\|\Big] - \alpha \sum_{i=1}^{N} \log c_i^D
$$

Here $e_i = \hat D_i - D_i$ is the depth residual, $c_i^D$ the predicted confidence map, $\odot$ the element-wise product and $\alpha$ a weight. The $(1 + D_i^{-1})$ factor is new versus VGGT. It weights errors on near pixels more, which accounts for relative scale. The $-\alpha \log c$ term stops the model from setting confidence to zero everywhere.

- **Point.** The same form, with $e_i = \pi^{-1}(\hat D_i, \hat g_i) - P_i$. The function $\pi^{-1}$ unprojects predicted depth through the predicted camera, and $P_i$ is the ground-truth point map. This couples the camera and depth predictions.
- **Matching.** A binary cross-entropy on last-layer tokens, $\mathcal{L}_{\text{match}} = \mathbb{E}_{\text{pos}}[-\log\sigma(s)] + \mathbb{E}_{\text{neg}}[-\log(1-\sigma(s))]$. Here $s$ is the cosine similarity of two ℓ2-normalized tokens. Positives are patches that see the same 3D point. Negatives must be far from the epipolar line and different in mean color, because in a dynamic scene "not matched" does not mean "different place".

**Tiny check on the depth weight.** In the normalized units the ground truth is scaled to, an error of 0.1 at depth 1 gets weight ${1 + 1/1 = 2}$. The same error at depth 10 gets ${1 + 0.1 = 1.1}$. Near errors cost about twice as much, which is what you want when the far field is often sky.

Does dropping the extra heads hurt? Barely. Removing the point and matching losses raises point error from 0.073 to 0.078. VGGT's original multi-head setup reaches 0.070. So the single head keeps most of the multi-task benefit at a fraction of the memory.

## Technique 3: dynamic scenes without dynamic outputs

**The problem.** Most video has moving things in it. A model that only handles rigid scenes throws away most of the Internet. But how you represent the output decides whether camera motion and scene motion get tangled.

**Why not point maps or ray maps.** DUSt3R-style models predict point maps and recover cameras from them. That works for rigid scenes. With motion, you either segment out moving pixels (MonST3R) or invent dynamic point maps. Depth plus ray maps is another option. The paper argues that ray maps add an expensive dense output and can entangle camera information with pixel appearance.

![A fixed camera watching a dancer: pixels that see the dancer change across frames, but the camera parameters stay identical and only the dancer region of the depth map moves.](/imgs/blogs/vggt-4.webp)

**The analogy, from the paper.** A stationary camera watching a dancer sees large image motion but has fixed camera parameters. Any output that is computed from pixels is tempted to explain the dancer's motion as camera motion. Predicting $g_i$ directly from tokens sidesteps the trap.

**The mechanism.** So VGGT-Ω predicts only depth maps and cameras, the same outputs as in the static case. There is no motion mask, no flow, no 4D track. The statistical prior that separates "the camera moved" from "the world moved" is learned from data rather than hand-written. Classical non-rigid structure from motion used low-rank or local-rigidity constraints instead, which the paper calls brittle.

**When it fails.** If you need to know *which* pixels moved, or where a point went, you must derive it yourself. The representation seems to know anyway: k-means on PCA-reduced layer-4 tokens isolates a moving dancer from a static crowd, with no motion labels. By layer 23 the same procedure highlights all people, moving or not.

## Technique 4: a data engine for dynamic video

The second half of the scaling story is data. The paper combines about 3M sequences from public and internal datasets with a new annotation pipeline run on roughly 40M Internet-style videos. It drops Kubric and PointOdyssey, which VGGT used, because their backgrounds are fake geometry with invalid depth.

![The VGGT-Ω data funnel on a log scale: 40M videos, about 4M kept after VLM filtering, 0.8M annotated sequences (600K static, 200K dynamic), plus about 3M existing sequences for about 4M supervised sequences in total.](/imgs/blogs/vggt-5.webp)

The pipeline, stage by stage:

1. **VLM pre-filter.** A vision-language model rejects 50% of clips as too hard (cuts, heavy blur, overlays) and flags 40% as reconstructible but likely inaccurate. The remaining 10% go on. It also tags each clip as static or dynamic.
2. **Dynamic masks.** Grounding DINO boxes movable categories like people and cars. Those regions are excluded from matching and verification.
3. **Matching ensemble.** SIFT, SuperPoint with SuperGlue, ALIKED with LightGlue, and the VGGSfM tracker.
4. **Reconstruction.** The original VGGT initializes cameras when essential-matrix RANSAC has too few inliers. Then COLMAP runs bundle adjustment. Sequences fail if under 99.5% of images register, if field of view falls outside 30° to 120°, or if the distortion ratio exceeds 0.1. Patch-based multi-view stereo then produces dense depth.
5. **Multi-view consistency.** Each depth map is unprojected, reprojected into other views and compared. Sequences with under 5% valid pixels are dropped.
6. **Learned filter.** An XGBoost, random forest and CatBoost ensemble, trained on 500 hand-labeled static and 500 dynamic sequences, rejects bad reconstructions using features like trajectory smoothness, parallax angle and point-cloud linearity.

The output is about 600K static and 200K dynamic sequences. Note that depth on dynamic pixels is discarded. The model learns depth for moving things from synthetic data, where it is exact.

**Is the pseudo ground truth good?** On Sintel, restricted to sequences and pixels both pipelines accept, theirs scores 96.4% AUC@30° for camera and 99.3% $\delta_{1.25}$ for depth. MegaSaM scores 62.1% and 77.2%. The authors are explicit that they trade yield for precision. In their experience a smaller, very clean label set beats a larger noisy one.

The supplement's data-quality section is worth reading: sensor depth leaking onto chair backs, bundle-adjustment "doming", and MegaDepth labels that teach models to paint pedestrians onto walls. Each became a specific inference-time failure that no benchmark number caught.

## Technique 5: teacher-student training on unlabeled video

**The problem.** Labeled 3D data tops out in the millions. Unlabeled video is effectively unlimited. Can it help?

**The mechanism.** A DINO-style momentum teacher. Teacher and student both start from the supervised VGGT-Ω checkpoint. Each gets the same frames under different augmentations: color jitter, blur, random 90° rotations, patch masking, and random frame reordering (which changes the reference frame). After restoring a common order, the student must match the teacher's camera and depth outputs and its tokens at several layers (ℓ2 feature matching). The camera and depth heads are frozen to prevent collapse. The teacher never sees gradients. It tracks the student by an exponential moving average:

$$
\theta_T \leftarrow m\,\theta_T + (1 - m)\,\theta_S
$$

where $\theta_T$ and $\theta_S$ are teacher and student weights and $m = 0.999$. With that decay, the teacher's weights are an average over roughly the last ${1/(1-m) = 1000}$ student steps, which makes it a slower, smoother target.

**What it buys.** Not much on benchmarks. Swapping 10% of supervised steps for self-supervised ones lowers point error from 0.073 to 0.070, with better out-of-distribution generalization. The training schedule is 240K AdamW iterations: 160K supervised, 50K self-supervised, then 30K supervised.

**When it fails.** The authors tried novel-view synthesis (RayZer and E-RayZer variants), NeRF and Gaussian-splat rendering, token masking and temporal-order prediction. None helped, and the teacher-student approach needs a pretrained model. They call self-supervised reconstruction an open problem.

## The scaling result

Here is the headline. The paper measures **point error**: unproject predicted depth with predicted cameras and take the ℓ2 distance to ground-truth points, averaged over six datasets. They prefer it to Chamfer distance because nearest-neighbor matching can be dominated by big walls and floors. All runs see about the same number of training tokens.

![Point error against training sequences and model size on log-log axes, with the paper's points and power-law fits (fits are mine): slope about -0.19 for data and about -0.21 for parameters.](/imgs/blogs/vggt-2.webp)

The raw numbers from Figure 1:

| Training sequences (1B model) | Point error | | Parameters | Point error |
| --- | --- | --- | --- | --- |
| 2K | 0.275 | | 0.2B | 0.107 |
| 10K | 0.210 | | 1B | 0.073 |
| 50K | 0.160 | | 5B | 0.057 |
| 200K | 0.129 | | 10B | 0.046 |
| 2M | 0.073 | | | |

The paper says only that the curves "suggest" power laws. It does not fit one. I did, by least squares in log-log space on the labeled points:

$$
E(D) \approx 1.20\, D^{-0.19}, \qquad E(N) \approx 0.076\, N^{-0.21}
$$

where $E$ is point error, $D$ is training sequences and $N$ is parameters in billions. Read the data exponent this way: every 10x more sequences multiplies error by ${10^{-0.19} \approx 0.65}$, a 35% cut. Every 10x more parameters cuts error by about 38%. These fits are my extrapolation, not the paper's claim, and with four or five points each they are fragile.

Two things in the data curve deserve a second look. First, the last point (2M, 0.073) sits below the fit (0.078), so the curve may still be steepening, not flattening. Second, the axis is "sequences", which bundles together data quantity, data quality and the static/dynamic mix. The paper never separates them. That gap is exactly where follow-up work lives.

## Benchmarks: static and dynamic

Ten frames are sampled per scene. Camera AUC is the area under the accuracy curve of relative rotation and translation error below a threshold. Higher is better.

| Camera AUC@3° | 7 Scenes | NRGBD | ETH3D | DyCheck | Sintel | TUM-Dyn |
| --- | --- | --- | --- | --- | --- | --- |
| MegaSaM | 10.6 | 17.2 | 5.9 | 26.8 | 22.5 | 15.4 |
| VGGT | 10.9 | 81.7 | 18.8 | 21.0 | 15.0 | 16.6 |
| PI3 | 13.3 | 83.8 | 35.3 | 23.3 | 14.8 | 16.1 |
| DA3 (Giant) | 18.7 | 86.4 | 46.1 | 32.1 | 16.2 | 20.8 |
| VGGT-Ω 1B | 29.6 | 89.7 | 49.8 | 38.4 | 35.3 | 30.2 |
| VGGT-Ω 10B | **36.4** | **92.5** | **56.3** | **43.7** | **40.0** | **36.4** |

The pattern in prior work is a trade. Feed-forward models (VGGT, PI3, DA3) are strong on static, wide-baseline scenes. MegaSaM wins at strict thresholds on dynamic Sintel but collapses on wide-baseline ETH3D (AUC@30° of 38.1 versus DA3's 87.0). VGGT-Ω wins every column at both thresholds.

Depth tells the same story. On Sintel, $\delta_{1.25}$ rises from 86.1 (DA3) to 93.5 and AbsRel falls from 0.118 to 0.081 for the 10B model. Against MegaSaM, the $\delta_{1.25}$ jump is 74.1 to 93.5, at roughly 50x the speed.

![Figure 4 from Wang et al. (2026): qualitative results on static and dynamic scenes, including traffic flow, a tennis player's trajectory and an underwater coral reef, with 4 to 64 input frames.](/imgs/blogs/vggt-fig3.webp)

**Speed and memory at inference.** After fixing the released VGGT (it cached tensors from all 24 layers; the heads need four), VGGT and VGGT-Ω both handle about 1,250 frames on one 80 GB A100, while DA3 runs out of memory near 750. VGGT-Ω is faster, mostly from the patch-16 DINOv3 (about 25% fewer tokens) and the register layers (a 20% to 25% speedup). The all-register variant cuts 1,000-frame runtime from 240.2 s to 11.7 s, at lower accuracy.

![Figure 7 from Wang et al. (2026): inference memory and time versus number of input frames for DA3-Giant, VGGT, VGGT-Ω and the register-attention-only variant on an 80 GB A100. Crosses mark out-of-memory.](/imgs/blogs/vggt-fig4.webp)

Note the subtle point here. Removing global attention buys **speed, not memory**, at inference. Flash attention never materializes the attention matrix. Peak memory is set by activations that grow linearly with frames and tokens, which register attention does not shrink.

### Ablations (1B model, 2M sequences)

| Variant | Point error |
| --- | --- |
| All global attention | 0.071 |
| Default (25% register attention) | 0.073 |
| No point or matching loss | 0.078 |
| VGGT-style multi-head, multi-task | 0.070 |
| 10% of steps self-supervised | 0.070 |

## Registers as a scene representation

Because register attention forces the 16 scene tokens to carry the cross-frame summary, they work as a frozen spatial feature.

**Robotics.** Concatenating frozen VGGT-Ω registers into OpenVLA-OFT raises the LIBERO average success rate from 97.1% to 98.5%, with gains on all four suites (Spatial 97.6 to 99.3, Long 94.5 to 96.7). The baseline is already near ceiling, so the absolute gains are small.

**Language.** A small attention stack reads only the registers plus a learnable language token. It is aligned to mean-pooled VLM caption embeddings with a symmetric InfoNCE loss. After 10K iterations, retrieval on 100 curated videos reaches 76.8% top-1 and 97.0% top-3. Swapping in a text-only Qwen3 embedding with no extra training still gives 47.5% top-1. Geometry accuracy does not degrade after alignment. The language token never sees patch tokens, so the semantics must live in the registers.

## Critique

**What is strong.** The scaling design is clean: matched token budgets, one metric, six datasets. Each engineering change is ablated and cheap to adopt. And the paper is honest: self-supervision barely helps, MLP-only heads were rejected, and iterative camera refinement would add 4% to 6% AUC@3° but was dropped for simplicity.

**What is weak.**

- **Two short curves, no fit.** Four model points and five labeled data points over limited ranges. "Power law" is suggested, not tested against alternatives like a saturating or broken power law (see [broken scaling laws](/blog/machine-learning/scaling-laws/broken-neural-scaling-laws)).
- **The data axis is confounded.** Sequence count, source mix, static/dynamic ratio and label quality all change together. A practitioner cannot tell which knob is paying.
- **No compute-optimal analysis.** Every run has about the same token count, so we cannot say how to split a fixed budget between parameters and data.
- **Data you cannot get.** The 40M videos and several datasets are internal. The recipe is documented in detail, but the scaling curve cannot be independently reproduced at full scale.
- **Benchmark provenance.** On 2026-09-08 the project's code repository added a 1B checkpoint "from an additional training run to further validate reproducibility and address a potential concern with the original checkpoint", and asks that it serve as the reference for benchmark comparisons. The repository does not say what the concern was. Until that is clear, I would treat the 1B rows in the tables above with some caution and prefer numbers re-measured on the reproduction checkpoint.
- **Dynamic means "cameras and depth survive motion".** There is no evaluation of motion segmentation, scene flow or 4D consistency, since the model does not output them.

**What would change my mind.** If an independent group trained the 0.2B to 1B models on an open data mix and the point-error curve flattened well before 2M sequences, I would read the paper's curve as a data-quality effect of their internal pipeline, not a scaling law of the architecture. Likewise, if the reproduction checkpoint lands far below the published 1B numbers on Sintel, the headline 77% gain needs restating.

## Where this goes next

The orchestrating question is simple: now that feed-forward geometry has a curve, can we fit it and steer it? Here are four concrete projects, each tractable now because VGGT-Ω exists with released code, checkpoints and a documented data pipeline.

**1. Fit a real scaling law for geometric foundation models.**
*Question:* does point error follow $E(N, D) = E_\infty + A N^{-\alpha} + B D^{-\beta}$, as in language models, and what are $\alpha$, $\beta$ and the irreducible floor $E_\infty$? *Why now:* the 200M to 10B family and its training recipe make a grid feasible at small scale. *First experiment:* train 5 model sizes (50M to 1B) times 5 data sizes on public data only, fit the joint form, and test whether the fit predicts a held-out larger run within 10%.

**2. Data-mixture scaling for dynamic accuracy.**
*Question:* how does Sintel and DyCheck accuracy scale with the *fraction* of dynamic sequences, separately from total count? The paper recommends about 80% synthetic and 20% real but never sweeps the dynamic share. *Why now:* the annotation pipeline already tags static versus dynamic. *First experiment:* hold total sequences fixed at 200K and sweep the dynamic share across 0%, 10%, 30%, 50%, 70%, measuring static and dynamic benchmarks to find the crossover where static accuracy starts to drop.

**3. Quality-adjusted data scaling.**
*Question:* how many noisy sequences equal one clean sequence? *Why now:* the pipeline exposes explicit filter thresholds (5% valid depth, 99.5% registration, the classifier score), so quality can be dialed. *First experiment:* loosen one filter at a time, train at two data scales, and fit an "effective data" multiplier per filter, in the spirit of [data-quality scaling laws](/blog/machine-learning/scaling-laws/data-quality-scaling-laws).

**4. The register budget as a scaling axis.**
*Question:* is 16 registers and 25% register layers optimal, and does the best ratio shift with model size or frame count? *Why now:* the all-register model already matches VGGT at 6% of the FLOPs, so a cheap, accurate point probably exists in between. *First experiment:* sweep registers (4, 16, 64) against register-layer share (0%, 25%, 50%, 75%, 100%) at 200M parameters and plot accuracy versus FLOPs as a Pareto front.

## References

- Wang, J., Chen, M., Zhang, S., Karaev, N., Schönberger, J., Labatut, P., Bojanowski, P., Novotny, D., Vedaldi, A., Rupprecht, C. *VGGT-Ω*. CVPR 2026. [arXiv:2605.15195](https://arxiv.org/abs/2605.15195), [CVPR page](https://cvpr.thecvf.com/virtual/2026/poster/39730), [project page](http://vggt-omega.github.io/), [code](https://github.com/facebookresearch/vggt-omega).
- Wang, J. et al. *VGGT: Visual Geometry Grounded Transformer*. CVPR 2025. Our walkthrough: [VGGT on this site](/blog/paper-reading/computer-vision/vggt-visual-geometry-grounded-transformer).
- Background on the DINO backbones VGGT-Ω starts from: [ViT, SigLIP and DINO explained](/blog/machine-learning/computer-vision/vit-siglip-dino-explained).
- Scaling-law background: [Chinchilla compute-optimal scaling](/blog/machine-learning/scaling-laws/chinchilla-compute-optimal-scaling), [data-quality scaling laws](/blog/machine-learning/scaling-laws/data-quality-scaling-laws).
