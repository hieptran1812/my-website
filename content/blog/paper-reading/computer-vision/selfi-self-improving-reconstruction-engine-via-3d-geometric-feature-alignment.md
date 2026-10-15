---
title: "Selfi: Teaching VGGT to Agree With Itself Across Views"
publishDate: "2026-10-15"
date: "2026-10-15"
description: "How Selfi turns a frozen VGGT into a state-of-the-art unposed novel view synthesis and pose engine by training a small feature adapter on VGGT's own depth and cameras, with no 3D labels."
excerpt: "VGGT predicts good cameras and depth, but its features do not agree across views. Selfi reprojects pixels with VGGT's own outputs, trains a 24-channel adapter to match them, and uses the result for feed-forward Gaussians, bundle adjustment, and a depth shift that keeps the splats aligned after BA."
category: "paper-reading"
subcategory: "Computer Vision"
tags: ["3d-reconstruction", "cvpr-2026", "cvpr-2026-oral", "paper-reading", "selfi", "vggt", "novel-view-synthesis", "gaussian-splatting", "self-supervised-learning", "bundle-adjustment", "feature-matching", "3d-foundation-model"]
author: "Hiep Tran"
featured: true
image: "/imgs/blogs/selfi-self-improving-reconstruction-engine-via-3d-geometric-feature-alignment-fig1.webp"
readTime: 19
paper:
  title: "Selfi: Self Improving Reconstruction Engine via 3D Geometric Feature Alignment"
  authors: "Youming Deng, Songyou Peng, Junyi Zhang, Kathryn Heal, Tiancheng Sun, John Flynn, Steve Marschner, Lucy Chai"
  venue: "CVPR 2026 (oral); arXiv:2512.08930"
  url: "https://arxiv.org/abs/2512.08930"
---

> [!tldr]
> **Claim.** A frozen VGGT already knows a scene's cameras and depth, but its internal features are not consistent across views. Selfi uses VGGT's *own* predictions as pseudo-ground-truth to train a small adapter whose features match wherever two pixels see the same 3D point.
>
> **Why it matters.** Those aligned features feed a Gaussian-splat head that beats AnySplat and WorldMirror by large margins on DL3DV and RealEstate10K from unposed images, and they double as a matcher that refines VGGT's poses with bundle adjustment on 100 frames, where VGGT's own CoTracker-based BA runs out of memory.
>
> **Most surprising.** Training on pseudo-labels that have zero reprojection error under VGGT's cameras still yields correspondences that *improve* those cameras. On Tanks and Temples, AUC@3 climbs from 0.818 to 0.955.
>
> **Where it breaks.** Sky and distant background (VGGT's normalized scale), exposure differences between views, dynamic scenes, and long sequences: at 36 frames on DL3DV it falls 8.5 dB behind per-scene 3DGS.

Most "foundation model for 3D" papers ask what a model like VGGT can predict. Selfi asks a quieter question: **are the features inside VGGT geometrically consistent?** If a chair corner is visible in frame 3 and frame 7, do the two pixels carry similar feature vectors? The answer, per the paper, is "not enough", and fixing it pays off twice: once for rendering, once for pose.

You do not need labels to fix it. VGGT's own depth and cameras tell you which pixel in frame 7 matches which pixel in frame 3, so a new feature head can learn to agree with VGGT's geometry using nothing but VGGT. That is the "self-improving" loop.

If you have not read the backbone yet, start with our [VGGT walkthrough](/blog/paper-reading/computer-vision/vggt-visual-geometry-grounded-transformer). Selfi treats it as a frozen teacher.

![Figure 1 from Deng et al. (2025): the Selfi pipeline, from geometric feature alignment to Gaussian prediction, pose refinement and novel view rendering](/imgs/blogs/selfi-self-improving-reconstruction-engine-via-3d-geometric-feature-alignment-fig1.webp)

## The problem: good predictions, inconsistent features

Classical novel view synthesis (NVS) runs Structure-from-Motion (SfM) first, then optimizes each scene. That is slow and fragile: bad poses mean bad renders. Feed-forward methods like pixelSplat and MVSplat drop the per-scene optimization but still assume known cameras.

3D vision foundation models (VFMs) such as DUSt3R and VGGT remove the camera requirement. Feed in uncalibrated images, get back cameras, depth and point maps in one pass. The natural next step, taken by AnySplat and WorldMirror, is to decode VFM features straight into 3D Gaussians.

The paper's diagnosis is that this step underperforms because VFM features are trained to *predict* geometry, not to be *consistent* across views. A Gaussian head built on top has to rediscover cross-view agreement on its own. Selfi's bet is that you should bake that agreement into the features first.

| Approach | Needs poses? | Per-scene optimization? | Cross-view consistency comes from |
| --- | --- | --- | --- |
| 3DGS / NeRF | Yes (SfM) | Yes | The optimizer |
| pixelSplat, MVSplat | Yes | No | Cost volumes, epipolar priors |
| AnySplat, WorldMirror | No (VFM) | No | Whatever the VFM features happen to hold |
| **Selfi** | No (VFM) | No | An explicit alignment loss on the features |

## Technique 1: a feature adapter on a frozen VGGT

**Problem.** We want a per-pixel feature map where similarity means "same 3D point", but we do not want to retrain a 1B-parameter backbone.

**Analogy.** Think of VGGT as a fluent speaker of several dialects who describes the same object slightly differently depending on which window they look through. The adapter is a translator that maps every description into one shared vocabulary.

**Mechanism.** Selfi freezes the VGGT backbone and aggregator. It takes image tokens from four intermediate layers and runs them through a DPT head (the dense prediction transformer decoder) to get a full-resolution feature map:

$$
F_i = \mathrm{DPT}_{\text{adapter}}(T_i), \qquad T_i \in \mathbb{R}^{4 \times (H_p \times W_p) \times D}, \quad F_i \in \mathbb{R}^{H \times W \times C}
$$

Here $i$ indexes the frame, $H_p \times W_p$ is the patch grid, $D$ is VGGT's token width, $H \times W$ is the image resolution, and $C = 24$ is the feature dimension. That is tiny. A 24-dimensional vector per pixel is closer to a learned descriptor than to a semantic embedding.

**Why so small.** The paper does not ablate $C$. My reading is that a low dimension forces the adapter to spend its capacity on geometry, since there is no room to also carry rich semantics. The supplement's channel visualizations show each channel encoding a slightly different spatial pattern mixed with image structure.

## Technique 2: pseudo-ground-truth from reprojection

**Problem.** To train "same 3D point means same feature", we need to know which pixels correspond. We have no 3D labels.

**Analogy.** If you know where you stood and how far away the wall was, you can work out where that patch of wall would appear from a friend's position. VGGT gives you both the standing position (pose) and the distance (depth).

**Mechanism.** Run VGGT on the image set to get per-frame depth maps $D_i$, shared intrinsics $K$, and relative poses $[R_{t \leftarrow s} \mid t_{t \leftarrow s}]$. For a query pixel $p_s^n$ in source frame $s$, unproject it with its depth, move it into the target frame's coordinates, and project it back:

$$
P_t^n = R_{t \leftarrow s}\, D_s^n\, \pi_K^{-1}(p_s^n) + t_{t \leftarrow s}, \qquad p_t^n = \pi_K(P_t^n)
$$

Symbols: $\pi_K$ projects a 3D camera-space point to pixels with intrinsics $K$; $\pi_K^{-1}$ lifts a pixel to a unit-depth ray; $D_s^n$ is the source depth at pixel $n$; $P_t^n$ is the 3D point in target coordinates; $p_t^n$ is the pseudo-GT match.

Reprojection alone ignores occlusion. A point visible in the source may be hidden behind something in the target. Selfi keeps a hard visibility mask that compares the reprojected point's depth to what the target's own depth map says is there:

$$
V_t^n = \big[\, \lvert P_t^n \cdot [0\ 0\ 1]^\top - D_t^n \rvert \lt \alpha \,\big]
$$

$[\cdot]$ is the Iverson bracket (1 if true, 0 otherwise), $D_t^n$ is the target depth map sampled at $p_t^n$, and $\alpha = 0.05$. Note that VGGT works in a normalized scene scale, so $\alpha$ is a fraction of that normalized depth, not 5 cm.

![Where the pseudo-ground-truth match comes from: VGGT's depth and pose carry a source pixel to its target pixel, and a depth test drops what the target cannot see](/imgs/blogs/selfi-self-improving-reconstruction-engine-via-3d-geometric-feature-alignment-1.webp)

**Worked example.** Take $K$ with focal length 500 and principal point $(259, 147)$, matching a 518×294 image. Source pixel $(309, 147)$ with depth 2 lifts to camera point $(0.2,\ 0,\ 2)$, since $(309 - 259)/500 \times 2 = 0.2$. Let the target camera sit 0.1 units to the right with no rotation, so $t_{t \leftarrow s} = (-0.1, 0, 0)$. Then $P_t = (0.1,\ 0,\ 2)$ and $u = 500 \times 0.1 / 2 + 259 = 284$. The match is pixel $(284, 147)$.

Now check visibility. If the target depth map reads 2.01 at $(284, 147)$, the gap is 0.01, below $\alpha$, so $V = 1$. If a lamp sits in front and the target depth reads 1.2, the gap is 0.8 and $V = 0$: this pair is dropped from the loss.

**What the supplement adds.** At training time, the middle frame of an 11-frame clip is the source and the other 10 are targets, with 4,096 random query points per frame. Matches are also filtered when the point lands outside the target's field of view, lands behind the camera, or either pixel has VGGT confidence below 1.2.

**When it fails.** The pseudo-GT is only as good as VGGT. Wrong depth on the sky or on distant background produces confident but wrong matches, which is exactly the first limitation the authors list. The paper also tried a nearest-neighbor variant (match the target pixel whose 3D point is closest) and found it jittered from frame to frame even with $k = 10$, which hurt training.

## Technique 3: soft-argmax alignment instead of contrastive learning

**Problem.** Given a pseudo-GT match, how do you push the features toward it? The obvious answer, a CLIP-style contrastive loss as used by MASt3R, collapsed in the authors' hands: every feature converged to the same value and every query matched the same point.

**Analogy.** Contrastive learning grades only the one right answer against a few wrong ones. Selfi's loss grades the student's *pointing finger*: where on the whole target image does the student think the match is? Every pixel contributes to where the finger lands, so every pixel gets feedback.

![Figure 2 from Deng et al. (2025): features from a DPT adapter on frozen VGGT tokens are aligned using query points reprojected with VGGT's own depth and cameras](/imgs/blogs/selfi-self-improving-reconstruction-engine-via-3d-geometric-feature-alignment-fig2.webp)

**Mechanism.** Take the source feature $F_s^n \in \mathbb{R}^{C}$ at the query pixel. Compute cosine similarity against every target pixel, turn it into a probability map with a temperature softmax, and read off the expected location:

$$
S^n(u,v) = \frac{F_s^n \cdot F_t(u,v)}{\lVert F_s^n \rVert_2\, \lVert F_t(u,v) \rVert_2}, \qquad
w^n(u,v) = \frac{\exp\!\big(S^n(u,v)/\tau\big)}{\sum_{u',v'} \exp\!\big(S^n(u',v')/\tau\big)}
$$

$$
\hat{p}_t^n = \sum_{u,v} w^n(u,v)\,[u, v], \qquad
\mathcal{L}_{\text{align}} = \frac{1}{TN} \sum_{t=1}^{T} \sum_{n=1}^{N} V_t^n \,\big\lVert \hat{p}_t^n - p_t^n \big\rVert_2^2
$$

$(u, v)$ ranges over target pixels, $T$ is the number of target frames, $N$ the number of query points, and $\hat{p}_t^n$ is a soft-argmax: the weighted centroid of the heatmap. The loss is a plain squared pixel distance, masked by visibility.

**A note on $\tau$.** The paper reports $\tau = 100$. Read literally as $\exp(S/100)$ with cosine similarity in $[-1, 1]$, the softmax would be almost uniform over hundreds of thousands of pixels, which cannot localize anything. I read it as a scale of 100 (equivalently a temperature of 0.01) applied to the cosine. That is my interpretation, not a statement in the paper.

![Soft-argmax turns a heatmap into one point: every target cell's weight enters p_hat, so every pixel gets gradient, not just the matched one](/imgs/blogs/selfi-self-improving-reconstruction-engine-via-3d-geometric-feature-alignment-2.webp)

**Worked example.** Shrink the target to three pixels in a row at $u = 10, 11, 12$ with cosine similarities 0.99, 0.97 and 0.90. Scaled by 100 these become logits 99, 97 and 90. Relative to the max, the exponentials are $e^{0} = 1$, $e^{-2} \approx 0.135$, $e^{-9} \approx 0.0001$, so the weights are about 0.881, 0.119 and 0.0001.

The predicted location is ${10 \times 0.881 + 11 \times 0.119 + 12 \times 0.0001 \approx 10.12}$. If the pseudo-GT is $u = 10$, the loss is ${0.12}^2 \approx 0.014$. The gradient pushes pixel 11's similarity down and pixel 10's up, and also nudges pixel 12, which a contrastive loss with sampled negatives might never touch.

**Why it works.** The authors' hypothesis for the contrastive collapse is insufficient supervision: contrastive only grades matched points, while the soft-argmax supervises all target pixels. The cost is one $H \times W$ similarity map per query, which is why the feature dimension being 24 matters.

**When it fails.** A soft-argmax over a bimodal heatmap lands *between* the two modes. Repeated textures (tiled walls, window grids) are exactly where that happens. The paper does not analyze this case.

## Technique 4: a Gaussian head with view-dependent density

**Problem.** Turn the aligned features into something renderable, without poses and without per-scene optimization.

**Mechanism.** Freeze the adapter, then train a U-Net decoder that sees the aligned feature map concatenated with the RGB image. It predicts, per pixel, one 3D Gaussian: a quaternion $q_s$, scale $s_s$, color $c_s$, opacity $\sigma_s$, and a depth residual $\Delta D_s$ that corrects VGGT's depth:

$$
F^{\text{dec}}_s = \text{U-Net}\big(\mathrm{cat}(F_s, I_s)\big), \qquad
\mu_s = (D_s + \Delta D_s)\, \pi_K^{-1}(p_s)
$$

$\mu_s$ is the Gaussian center, placed along each pixel's ray at the corrected depth. All Gaussians from all source frames are rasterized into each target view with VGGT's predicted relative cameras, and the head is trained only with an RGB reconstruction loss $\mathcal{L}_{\text{RGB}} = \frac{1}{T}\sum_t \lVert \hat{I}_t - I_t \rVert$.

**The key modification.** Color uses spherical harmonics (SH), as in standard 3DGS. Selfi also puts SH on **density**. Each Gaussian's opacity becomes a function of viewing direction rather than one scalar. Since VGGT's geometry is imperfect in low-confidence regions, a Gaussian that looks right from its own camera may smear when seen from far away. View-dependent density lets it fade out for those views.

![Density that depends on who is looking: each Gaussian's opacity is an SH function of view direction, so far-view Gaussians fade and the faintest 30% are pruned](/imgs/blogs/selfi-self-improving-reconstruction-engine-via-3d-geometric-feature-alignment-4.webp)

The paper's own visualization makes the effect concrete. Six input views, target camera at the midpoint of the two center views, each input's Gaussians rendered alone. Without density SH, every input contributes a blurry, misaligned copy. With it, only the nearest inputs contribute, and far inputs go nearly transparent.

![Figure 6 from Deng et al. (2025): renders from each of six inputs into a mid-point target, without (middle row) and with (bottom row) view-dependent density](/imgs/blogs/selfi-self-improving-reconstruction-engine-via-3d-geometric-feature-alignment-fig6.webp)

The authors interpret density SH as a learned confidence score. It also gives a free pruning rule: per render camera, sort Gaussians by opacity and drop the lowest 30% (supplement), which saves rasterization memory.

**When it fails.** Density SH can hide geometry errors only when some nearby input sees the region well. A region seen only by far-away inputs has no good Gaussian to fall back on.

## Technique 5: bundle adjustment with the aligned features, and the depth shift

**Problem.** VGGT's poses are good but not perfect. Classical bundle adjustment (BA) can refine them, but it needs correspondences. VGGT's own BA uses CoTracker, which is heavy and runs out of memory on long sequences.

**Mechanism.** Selfi matches with its own features: dot products between aligned feature maps. At inference, it picks every fifth frame as a source, samples 2,048 points per source, and matches into targets within a radius of five frames, with the same confidence and field-of-view filters as training. BA then refines cameras and sparse 3D points.

The surprise is that this works at all. The adapter was trained to reproduce VGGT's geometry, whose pseudo-GT has zero reprojection error by construction. Yet the learned features generalize past their teacher and yield matches that *fix* the teacher's poses. My interpretation: the adapter averages VGGT's per-scene errors across a whole training distribution, so its features encode the consensus geometry rather than any one prediction's mistakes.

**The catch.** BA moves the sparse points and may also change the intrinsics. The dense Gaussians were placed using the old cameras. Swap in the new poses without moving the Gaussians, and rendering gets *worse*: in the ablation, BA alone drops PSNR from 24.67 to 24.61.

The authors plot sparse point depths before and after BA and find the change is close to linear in depth. So they fit an affine map $\phi(d) = a d + b$ on the sparse BA points and apply it to every dense depth map:

$$
\mu_s' = \phi(D_s + \Delta D_s)\, \pi_{K'}^{-1}(p_s), \qquad
s_s' = \frac{\phi(D_s + \Delta D_s)}{D_s + \Delta D_s}\, s_s
$$

$K'$ is the BA-refined intrinsics, $\mu_s'$ the moved center, and $s_s'$ the rescaled Gaussian size, so a splat that moves farther away also grows and keeps its pixel footprint.

![After bundle adjustment, one line re-places every Gaussian: BA shifts sparse-point depths almost affinely, so one fit phi moves each Gaussian along its own ray](/imgs/blogs/selfi-self-improving-reconstruction-engine-via-3d-geometric-feature-alignment-3.webp)

**Worked example.** Three sparse points sit at depths 1.0, 2.0 and 3.0 before BA and 1.05, 2.15 and 3.25 after. A least-squares line through them is $\phi(d) = 1.1 d - 0.05$. A dense Gaussian at depth 2.4 moves to ${1.1 \times 2.4 - 0.05 = 2.59}$, and its scale grows by ${2.59 / 2.4 \approx 1.08}$. One line, fit on a few hundred points, moves a few hundred thousand Gaussians.

**When it fails.** The affine fit is per frame and global. If BA corrects a local error (one wall bent, the rest fine), a single line cannot express it. The paper shows the linear fit suffices on its examples; it does not report residuals.

## Training recipe and compute

Everything is implemented in JAX. Both stages run on 128 H100 GPUs.

| Stage | Data | Iterations | LR (peak) | Wall clock |
| --- | --- | --- | --- | --- |
| Feature alignment (DPT adapter) | DL3DV only, 11-frame clips | 150K | 1e-4, cosine, 1K warm-up | ~2 days |
| Gaussian U-Net head | DL3DV + RealEstate10K, 6 inputs + 5 targets | 150K | 2e-4 | ~1.5 days |

Resolution is fixed at 294×518. "Lightweight adapter" refers to the trainable parameters, not to the compute: this is roughly 450 H100-days of training on top of a pretrained VGGT.

## Results

The headline comparisons are against pose-free feed-forward Gaussian methods (AnySplat, WorldMirror, Flare, NoPoSplat), with per-scene 3DGS on ground-truth poses as an upper bound.

![Figure 3 from Deng et al. (2025): novel views on DL3DV from AnySplat, WorldMirror, Selfi and ground truth; Selfi recovers thin guardrails and the text on the sign](/imgs/blogs/selfi-self-improving-reconstruction-engine-via-3d-geometric-feature-alignment-fig3.webp)

**Varying sequence length (Table 1, PSNR).**

| Dataset | Frames | AnySplat | WorldMirror | Selfi | 3DGS (GT poses) |
| --- | --- | --- | --- | --- | --- |
| DL3DV | 6 | 18.84 | 21.76 | **24.94** | 25.63 |
| DL3DV | 36 | 17.25 | 19.24 | **19.97** | 28.47 |
| RealEstate10K | 6 | 23.88 | 25.54 | **28.34** | 28.46 |
| RealEstate10K | 36 | 21.38 | 24.48 | **25.37** | 28.47 |

At six frames Selfi is within about 0.1 to 0.7 dB of per-scene 3DGS with ground-truth poses. As frames grow, every feed-forward method degrades and 3DGS improves. Pixel-aligned Gaussians from 36 views must agree with each other or produce duplicates, and nothing optimizes that agreement per scene.

**Varying overlap (Table 2).** On DL3DV's sparsest setting, Selfi reaches 19.46 PSNR against 3DGS's 19.58, with higher SSIM (0.7288 vs 0.6628) and lower LPIPS (0.2589 vs 0.3517). Averaged over overlaps on DL3DV, Selfi scores 22.30 against WorldMirror's 17.92.

**Two-view RealEstate10K (Table 3).** Selfi gets 29.01 PSNR, 0.942 SSIM, 0.053 LPIPS. Pose-required ReSplat and LVSM decoder-only have higher PSNR (29.72, 29.67) but clearly worse SSIM and LPIPS. Among pose-free methods, NoPoSplat and Flare sit near 26.9 PSNR.

**RayZer split (Table 4).** 23.75 PSNR versus RayZer's 23.72, with much better SSIM (0.850 vs 0.733) and LPIPS (0.129 vs 0.222). RayZer's external re-implementation gets a higher 25.47 PSNR. Note Selfi uses VGGT as extra supervision, which RayZer does not.

**Pose estimation (Table 6, AUC@3).**

| Frames | VGGT | VGGT + BA (CoTracker) | Selfi + BA |
| --- | --- | --- | --- |
| 10 | 0.790 | 0.835 | **0.867** |
| 40 | 0.858 | 0.872 | **0.882** |
| 100 | 0.845 | out of memory | **0.876** |

**BA feeding back into NVS (Table 5).** On DL3DV, BA plus depth shift lifts PSNR from 24.94 to 25.13 and AUC@3 from 0.813 to 0.870. Zero-shot on Tanks and Temples, AUC@3 goes from 0.818 to 0.955, while PSNR moves only 19.41 to 19.54. The pose gains are large; the rendering gains are small.

**Ablation (Table 7, DL3DV, smaller batch).**

| Configuration | PSNR | SSIM | LPIPS |
| --- | --- | --- | --- |
| VGGT tracking-head features, no alignment | 22.53 | 0.759 | 0.240 |
| + feature alignment | 23.29 | 0.792 | 0.210 |
| + density SH + RGB SH | 24.67 | 0.835 | 0.169 |
| + BA, no depth shift | 24.61 | 0.833 | 0.164 |
| + BA + depth shift | **24.88** | **0.844** | **0.157** |

Alignment alone is worth 0.76 dB. The two SH terms together add 1.38 dB, and either alone gives much less (23.70 RGB-only, 23.55 density-only), which suggests they interact. BA only helps once the depth shift moves the Gaussians.

**What is load-bearing that may not transfer.** Selfi inherits VGGT's normalized scale and static-scene training, and it is evaluated on the same two datasets it trained on (plus zero-shot MipNeRF360 and Tanks and Temples for BA). The Flare baseline was evaluated at 256×256 with a sliding three-frame window, which the authors note favors Flare. Exposure differences between inputs and targets penalize every method's PSNR; Selfi's strong SSIM/LPIPS against modest PSNR is consistent with that.

## Critique

**What is strong.** The core idea is clean and cheap to state: use the teacher's geometry to align the student's features, with a loss that supervises every target pixel. The ablation isolates alignment from the Gaussian head by keeping the head schedule fixed. And the negative result on contrastive learning is reported, not hidden in a footnote.

**What is weak.** "Self-improving" oversells the loop that was actually run. The pipeline runs once: VGGT labels, adapter trains, BA refines. The refined poses never go back to retrain the adapter or VGGT. Whether the loop *keeps* improving is the most interesting open question and is untested.

**Missing ablations.**
- Feature dimension $C$ and the four-layer token choice.
- Temperature $\tau$, especially given the ambiguity above.
- Alignment features fed to a stronger head, to separate "features are better" from "head is better tuned for these features".
- A comparison against simply fine-tuning VGGT's own tracking head with the same reprojection loss.

**What would change my mind.** If a version of WorldMirror's head trained on the same schedule over raw VGGT tokens closed most of the 0.76 dB alignment gap, the gain would be about training setup, not geometry. Conversely, if a second round of self-labeling with BA-refined poses gave a further gain, the "self-improving" framing would be fully earned.

## Where this goes next

The follow-up the paper invites is **iterated self-distillation of geometry foundation models across unlabeled video at scale.** Four concrete directions:

1. **Close the loop for real.** *Question:* does retraining the adapter on BA-refined poses and depth-shifted geometry keep improving pose AUC, or does it plateau after one round? *Why now:* Selfi shows the refined poses beat the teacher and provides the depth-shift trick that keeps dense geometry consistent with them. *First experiment:* run two more rounds on DL3DV, track AUC@3 and PSNR per round, and check for confirmation bias by measuring error on a held-out set with COLMAP poses.

2. **Distill the adapter back into the backbone.** *Question:* can VGGT itself be fine-tuned so its tokens are natively aligned, removing the adapter? *Why now:* the alignment loss needs only VGGT outputs, so it can be added as an auxiliary loss to VGGT training on any unlabeled video. *First experiment:* fine-tune VGGT's last aggregator blocks with $\mathcal{L}_{\text{align}}$ on a DPT probe, then compare pose AUC with and without the auxiliary loss. Compare to the scaled-up [VGGT-Ω](/blog/paper-reading/computer-vision/vggt) direction.

3. **Unlabeled internet video, with motion masks.** *Question:* does alignment on in-the-wild video help, once dynamic regions are excluded? *Why now:* the authors name dynamic scenes as a failure, and the visibility mask is already the place to filter. *First experiment:* add a mask from flow-vs-reprojection disagreement, train on a dynamic-heavy video set, and evaluate on DL3DV to see if static performance holds.

4. **Streaming alignment.** *Question:* can a recurrent reconstructor maintain aligned features over thousands of frames, where Selfi's per-pixel Gaussians already degrade at 36? *Why now:* stateful models like [CUT3R](/blog/paper-reading/computer-vision/continuous-3d-perception-model-with-persistent-state) give a persistent scene state to align against. *First experiment:* apply $\mathcal{L}_{\text{align}}$ between CUT3R's per-frame outputs and its state readout, and measure drift on 500-frame sequences.

All four are my extrapolations from the paper's results and limitations, not claims the authors make.

## References

- Deng, Peng, Zhang, Heal, Sun, Flynn, Marschner, Chai. *Selfi: Self Improving Reconstruction Engine via 3D Geometric Feature Alignment.* CVPR 2026. [arXiv:2512.08930](https://arxiv.org/abs/2512.08930). Project page: [denghilbert.github.io/selfi](https://denghilbert.github.io/selfi). CVPR oral page: [cvpr.thecvf.com/virtual/2026/oral/40355](https://cvpr.thecvf.com/virtual/2026/oral/40355).
- Wang et al. *VGGT: Visual Geometry Grounded Transformer.* CVPR 2025. Our walkthrough: [VGGT, or how one forward pass replaced SfM](/blog/paper-reading/computer-vision/vggt-visual-geometry-grounded-transformer).
- [VGGT-Ω](/blog/paper-reading/computer-vision/vggt), the scaled follow-up to VGGT.
- [CUT3R: continuous 3D perception with a persistent state](/blog/paper-reading/computer-vision/continuous-3d-perception-model-with-persistent-state).
