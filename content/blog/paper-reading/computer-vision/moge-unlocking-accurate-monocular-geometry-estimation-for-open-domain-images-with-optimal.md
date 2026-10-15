---
title: "MoGe: Unlocking Accurate Monocular Geometry Estimation for Open-Domain Images with Optimal Training Supervision"
publishDate: "2026-10-15"
date: "2026-10-15"
description: "How MoGe turns one photo into a 3D point map by predicting geometry only up to scale and a Z-shift, then supervising it with an exact L1 alignment solver and a multi-scale local loss."
category: "paper-reading"
subcategory: "Computer Vision"
tags: ["3d-reconstruction", "cvpr-2025", "cvpr-2025-oral", "moge", "monocular-geometry", "point-map", "depth-estimation", "camera-calibration"]
author: "Hiep Tran"
featured: true
readTime: 18
image: "/imgs/blogs/moge-unlocking-accurate-monocular-geometry-estimation-for-open-domain-images-with-optimal-fig1.webp"
excerpt: "MoGe predicts a 3D point map from a single image, but only up to an unknown scale and Z-shift. That one choice removes the focal-distance ambiguity from training, and an exact L1 alignment solver plus a local sphere loss do the rest."
paper:
  title: "MoGe: Unlocking Accurate Monocular Geometry Estimation for Open-Domain Images with Optimal Training Supervision"
  authors: "Ruicheng Wang, Sicheng Xu, Cassie Dai, Jianfeng Xiang, Yu Deng, Xin Tong, Jiaolong Yang"
  venue: "CVPR 2025 (oral)"
  url: "https://arxiv.org/abs/2410.19115"
---

> [!tldr]
> - **What it does.** MoGe takes one photo and predicts a 3D point for every pixel. The point map is *affine-invariant*: correct only up to a global scale and a shift along the camera's Z axis. Depth, focal length and field of view are derived from it afterwards.
> - **Key mechanism.** Supervision, not architecture. A plain DINOv2 ViT plus a conv decoder is trained with (1) an exact solver for the L1-optimal scale and shift (the ROE solver), and (2) a multi-scale local loss that aligns small spheres of points independently.
> - **Why it matters.** Across eight unseen datasets it cuts scale-invariant point-map error by about 35% versus the best prior method (average Rel 12.3 to 7.91), and it beats dedicated calibration networks at field-of-view estimation.
> - **Most surprising result.** Adding a single Z-shift degree of freedom matters more than any loss trick. An affine-invariant *depth* representation trained the same way collapses (point error 29.9 versus 9.84).
> - **Where it fails.** Thin structures, and anything temporal: each frame is solved alone, so video comes out flickering in scale.

![Figure 1 from Wang et al. (2025): open-domain images, the reconstructed point maps, depth and normals](/imgs/blogs/moge-unlocking-accurate-monocular-geometry-estimation-for-open-domain-images-with-optimal-fig1.webp)

The grid above is the whole promise: paintings, game screenshots, indoor rooms and street views, each turned into a clean 3D shape from one image. The rest of this post explains why the shapes come out clean, and it is almost entirely a story about how the loss is computed.

## The problem: depth is not geometry

Monocular depth estimation got very good between 2020 and 2024. MiDaS, Depth Anything and Marigold produce sharp depth maps for almost any photo. But a depth map is not a 3D shape. To lift depth into points you also need the camera intrinsics, mainly the focal length, and a wrong focal length bends the scene even if every depth value is perfect.

Guessing focal length from one image is hard. A photo of a car taken close with a wide lens and one taken far away with a telephoto lens can look nearly identical. This is the **focal-distance ambiguity**, and it sits at the center of the paper.

DUSt3R (and its multi-view descendants such as [VGGT](/blog/paper-reading/computer-vision/vggt-visual-geometry-grounded-transformer)) sidestepped intrinsics by predicting a point map directly: pixel in, 3D point out. But DUSt3R's point map is *scale-invariant*, it is correct up to one global scale. MoGe argues that this is still the wrong invariance for a single image, and that the training signal is where the real gains hide.

## Contributions

1. **Affine-invariant point maps** for single images: geometry up to scale and a Z-shift, which removes the focal-distance ambiguity from the supervision.
2. **The ROE solver**: a Robust, Optimal, Efficient way to compute the exact L1-optimal scale and shift between prediction and ground truth, fast enough to run inside every training step.
3. **A multi-scale local geometry loss**: many small spheres, each aligned on its own, so local shape is supervised even when the relative distance between objects is ambiguous.
4. **Derived camera and depth**: focal length and the Z-shift are recovered from the predicted points by a tiny least-squares fit, giving state-of-the-art field-of-view estimates for free.
5. **A 21-dataset, roughly 9-million-frame training mix**, with loss terms matched to label quality.

## Method

![Figure 3 from Wang et al. (2025): ViT encoder, conv decoder, affine-invariant point map plus mask, and the global, local, normal and mask losses](/imgs/blogs/moge-unlocking-accurate-monocular-geometry-estimation-for-open-domain-images-with-optimal-fig3.webp)

The network is deliberately boring: a DINOv2-pretrained ViT-Large encoder and a lightweight convolutional upsampler. It outputs two things per pixel, a 3D point and a validity mask (for sky and other regions at infinity). Everything interesting happens in the box labelled "Training" and in the small post-processing step that recovers the camera.

### Technique 1: the affine-invariant point map

**The problem.** If the network must output points in true camera space, it must also commit to a focal length and a distance. For ambiguous photos it cannot know them, so the targets it sees for near-identical images disagree. That is a noisy label, and noisy labels make blurry geometry.

**The analogy.** Think of a sculptor asked to model a statue from a photo. They can get the shape right. They cannot tell whether the statue was 2 meters away or 20. A fair judge should grade the shape and forgive the distance.

**The mechanism.** For an image $I \in \mathbb{R}^{H \times W \times 3}$ the model $F_\theta$ predicts a point map $P \in \mathbb{R}^{H \times W \times 3}$, with the X and Y axes aligned to the image's u and v axes. The prediction is defined only up to an affine family:

$$
P \;\cong\; sP + t, \qquad s \in \mathbb{R},\; t \in \mathbb{R}^3 .
$$

Here $s$ is a global scale and $t$ a global translation. With the usual assumptions (centered principal point, square pixels), the shift only needs a Z component, so $t = (0, 0, t_z)$.

![Why a Z-shift fixes focal-distance ambiguity: two cameras, one prediction, and scale-only versus scale-plus-shift alignment](/imgs/blogs/moge-unlocking-accurate-monocular-geometry-estimation-for-open-domain-images-with-optimal-1.webp)

**Why the shift is the key.** Scaling about the camera origin $O$ slides every point along its own ray. That preserves depth *ratios*: the back of an object stays the same multiple of the front. But a camera far away sees an object as relatively flat (back and front at almost the same depth), while a close camera sees it as deep. No scale can turn one into the other. A shift along $z$ can, because adding a constant to every depth changes the ratios.

**Tiny worked example.** A box spans depths 2 m to 3 m for a close camera (ratio 1.5). The same box seen by a telephoto camera spans 8 m to 9 m (ratio 1.125). To match the far ground truth with scale only, you need $s \cdot 2 = 8$, so $s = 4$, which sends the back to 12 m instead of 9 m. With scale and shift, $s = 1$ and $t_z = 6$ maps 2 to 8 and 3 to 9 exactly. The prediction was right; only the scale-only judge called it wrong.

![Figure 4 from Wang et al. (2025): similar images from different focal lengths, inconsistent scale-only supervision, consistent affine supervision](/imgs/blogs/moge-unlocking-accurate-monocular-geometry-estimation-for-open-domain-images-with-optimal-fig4.webp)

**When it fails.** The representation throws away metric scale and the absolute camera distance on purpose. Anyone who needs meters (robotics, AR placement) must recover them some other way. And the invariance is per image, which is exactly what breaks video.

### Technique 2: the global loss and the ROE solver

**The problem.** To compare an affine-invariant prediction with ground truth, you first have to pick $s$ and $t$. Earlier depth models used shortcuts: align medians, then normalize the spread. Those shortcuts are not the alignment that minimizes the loss, so the network is graded against a slightly wrong target on every step.

**The mechanism.** Let $\hat{p}_i$ be the predicted point for pixel $i$ and $p_i$ the ground truth, with $z_i$ its depth and $\mathcal{M}$ the set of labelled pixels. The global loss is a depth-weighted L1:

$$
\mathcal{L}_G = \sum_{i \in \mathcal{M}} \frac{1}{z_i} \left\| s\hat{p}_i + t - p_i \right\|_1 ,
$$

and MoGe uses the *optimal* alignment for that exact objective:

$$
(s^*, t^*) = \arg\min_{s,\,t} \sum_{i \in \mathcal{M}} \frac{1}{z_i} \left\| s\hat{p}_i + t - p_i \right\|_1 .
$$

The ${1/z_i}$ weight keeps distant points (large absolute errors) from drowning out near ones. Solving this as a linear program with simplex or interior-point methods costs more than $O(N^3)$, seconds per image, far too slow for training.

**The insight.** The optimum must make at least one residual exactly zero in Z: $s\hat{z}_k + t_z = z_k$ for some anchor point $k$. So for each candidate $k$, substitute $t_z = z_k - s\hat{z}_k$. Every residual then becomes a function of $s$ alone, of the form $|s a_i - b_i|$. That leaves $N$ independent one-dimensional problems, all solvable in parallel on a GPU.

**The 1D subproblem.** Drop indices for the moment and write it as

$$
\min_s \; l(s) = \sum_{i=1}^{N} w_i \left| s\hat{x}_i - x_i \right| .
$$

Each term is a V shape with its kink at $s = x_i / \hat{x}_i$. A sum of V shapes is convex and piecewise linear, so the minimum sits at one of those kinks. Sort the breakpoints, keep prefix sums of $w_i \hat{x}_i$, and the slope at each breakpoint costs $O(1)$. Walk until the slope changes sign. One subproblem is $O(N \log N)$ for the sort, and $N$ of them make $O(N^2 \log N)$ overall.

![ROE solver: the L1 alignment objective is piecewise linear in s, with the optimum on a breakpoint and a truncated variant that caps outliers](/imgs/blogs/moge-unlocking-accurate-monocular-geometry-estimation-for-open-domain-images-with-optimal-2.webp)

**Tiny worked example.** Three points with $\hat{x} = (1, 2, 4)$, $x = (2, 3, 9)$ and unit weights. The breakpoints are 2, 1.5 and 2.25. Sorted, they are 1.5, 2, 2.25. The slope of $l$ is $\sum_i w_i \hat{x}_i \,\mathrm{sign}(s\hat{x}_i - x_i)$:

| Interval of $s$ | Slope | Sign |
| --- | --- | --- |
| below 1.5 | $-2 - 1 - 4 = -7$ | falling |
| 1.5 to 2 | $+2 - 1 - 4 = -3$ | falling |
| 2 to 2.25 | $+2 + 1 - 4 = -1$ | falling |
| above 2.25 | $+7$ | rising |

So $s^* = 2.25$ with $l = 0.25 + 1.5 + 0 = 1.75$, lower than $l(2) = 2$. It is just a weighted median of the ratios, with weights $w_i\hat{x}_i$.

```python
import numpy as np

def solve_subproblem(xh, x, w):
    # min_s sum w_i |s*xh_i - x_i|, assumes xh_i > 0
    r = x / xh                      # breakpoints
    order = np.argsort(r)
    r, a = r[order], (w * xh)[order]
    q = np.cumsum(a)                # prefix sums of w_i * xh_i
    slope_right = 2 * q - q[-1]     # slope just right of each breakpoint
    k = np.argmax(slope_right >= 0) # first breakpoint where slope turns non-negative
    s = r[k]
    return s, np.sum(w * np.abs(s * xh - x))
```

**Robustness.** L1 is already gentler than L2, but the authors found it still gets dragged by gross errors, for example a foreground edge pixel predicted as distant background. So each residual is truncated: $\min(\tau, w_i|s\hat{x}_i - x_i|)$ with $\tau = 1$. The objective is no longer convex, yet its kinks are still enumerable, so the same breakpoint search applies with small changes.

**When it fails.** The "one residual is zero" argument enumerates every pixel as an anchor, so cost grows quadratically. MoGe keeps it practical by running on GPU in parallel, and the local loss below runs it on small sets.

### Technique 3: the multi-scale local geometry loss

**The problem.** From one image, the distance between two separate objects is often a guess. A single global alignment must compromise across all objects, so if the network puts the table a bit too close, the global fit punishes the chair's shape too, even though the chair was perfect.

**The analogy.** Grading an essay only by its overall argument misses that individual paragraphs are excellent. You also grade paragraph by paragraph.

**The mechanism.** Pick a ground-truth anchor $p_j$ and take the points within a sphere around it:

$$
\mathcal{S}_j = \{\, i \mid \|p_i - p_j\| \le r_j,\; i \in \mathcal{M} \,\}.
$$

Run the ROE solver on just that sphere to get its own $(s_j^*, t_j^*)$, then compute the same weighted L1. Summing over a set $\mathcal{H}_\alpha$ of sampled anchors gives

$$
\mathcal{L}_{S(\alpha)} = \sum_{j \in \mathcal{H}_\alpha} \sum_{i \in \mathcal{S}_j} \frac{1}{z_i} \left\| s_j^*\hat{p}_i + t_j^* - p_i \right\|_1 .
$$

![Global alignment vs multi-scale local alignment: one (s, t) leaves residuals on both objects, per-sphere alignment scores each local shape on its own](/imgs/blogs/moge-unlocking-accurate-monocular-geometry-estimation-for-open-domain-images-with-optimal-3.webp)

**Sizing the spheres.** A fixed radius in meters would be huge for a nearby cup and tiny for a distant building. MoGe instead ties the radius to depth:

$$
r_j = \alpha \cdot z_j \cdot \frac{\sqrt{W^2 + H^2}}{2f} ,
$$

where $z_j$ is the anchor depth, $f$ the ground-truth focal length, and $W, H$ the image size. A sphere of radius $r_j$ at depth $z_j$ projects to a disc of diameter ${2r_j f / z_j} = \alpha\sqrt{W^2+H^2}$. So $\alpha$ is simply the fraction of the image diagonal the sphere covers. Three scales are used: $\alpha = 1/4, 1/16, 1/64$, giving $\mathcal{L}_{S_1}, \mathcal{L}_{S_2}, \mathcal{L}_{S_3}$.

![Sizing the local spheres: radius tracks depth, so every sphere covers the same fraction alpha of the image diagonal](/imgs/blogs/moge-unlocking-accurate-monocular-geometry-estimation-for-open-domain-images-with-optimal-4.webp)

**Tiny worked example.** Image 640 by 480 (diagonal 800 px), $f = 500$ px, $\alpha = 1/4$, anchor at 4 m. Then $r_j = {0.25} \cdot 4 \cdot 800 / 1000 = 0.8$ m. Its projected diameter is ${1.6} \cdot 500 / 4 = 200$ px, a quarter of the diagonal, as designed.

**When it fails.** Spheres need dense, accurate ground truth inside them. That is why the authors apply finer local losses only to higher-quality datasets. On noisy LiDAR or sparse SfM, a sphere at $\alpha = 1/64$ may hold a handful of unreliable points.

### Technique 4: normals, the infinity mask, and the full objective

Two smaller terms finish the loss. A **normal loss** compares surface normals computed from the predicted point map (cross product of neighbouring grid edges) against ground truth:

$$
\mathcal{L}_N = \sum_{i \in \mathcal{M}} \angle(\hat{n}_i, n_i).
$$

A **mask loss** trains a single-channel head $\hat{M}$ to flag points with undefined geometry such as sky, with $M_{\text{inf}}$ the infinity label (from synthetic data, or SegFormer sky masks for real photos):

$$
\mathcal{L}_M = \left\| \hat{M} - (1 - M_{\text{inf}}) \right\|_2^2 .
$$

The total is $\mathcal{L} = \lambda_G\mathcal{L}_G + \lambda_{S_1}\mathcal{L}_{S_1} + \lambda_{S_2}\mathcal{L}_{S_2} + \lambda_{S_3}\mathcal{L}_{S_3} + \lambda_N\mathcal{L}_N + \lambda_M\mathcal{L}_M$. The mask matters more than it looks. The paper's alternative, labelling the sky at 1,000 times the average distance, damaged foreground accuracy because the network struggled with such large targets.

### Technique 5: recovering focal length and shift at inference

**The problem.** Users want a real camera-space point cloud, depth, and a field of view, not an affine-invariant blob.

**The mechanism.** Each predicted point $(x_i, y_i, z_i)$ belongs to a known pixel $(u_i, v_i)$, measured from the image center. A pinhole camera with focal $f$ must reproject the shifted point onto that pixel. So MoGe solves

$$
\min_{f,\,t'_z} \sum_{i=1}^{N} \left( \frac{f x_i}{z_i + t'_z} - u_i \right)^2 + \left( \frac{f y_i}{z_i + t'_z} - v_i \right)^2 ,
$$

where $t'_z = t_z / s$. For a fixed $t'_z$ the optimal $f$ has a closed form (a ratio of dot products), so the problem collapses to one variable. They solve it with Levenberg-Marquardt in SciPy on a 64 by 64 downsampled point map; it typically converges in under 10 iterations, around 3 ms.

**Tiny worked example.** One point at $(x, z) = (1, 4)$ that lands at $u = 50$ px. With $t'_z = 0$, $f = 50 \cdot 4 / 1 = 200$. With $t'_z = 1$, $f = 50 \cdot 5 = 250$. Focal length and shift are coupled, and only many points at different depths pin both down. Flat, fronto-parallel scenes give little leverage, which is where focal estimates should be trusted least (my inference, not tested in the paper).

## Experiments and results

**Setup.** ViT-L encoder, batch size 256, learning rates of 5e-6 (encoder) and 5e-5 (decoder) decaying by 5 times every 100K iterations. Training images vary in aspect ratio from 1:2 to 2:1 and from 250K to 500K pixels. On real data, the top 5% of per-pixel losses are discarded as label noise. Evaluation is zero-shot on eight datasets the model never saw: NYUv2, KITTI, ETH3D, iBims-1, GSO, Sintel, DDAD and DIODE. All baselines use ViT-L except LeReS (ResNeXt101) and the diffusion methods.

Metrics: point error $\mathrm{Rel}^p = \|\hat{p} - p\|_2 / \|p\|_2$, and $\delta_1^p$, the share of points with relative error below 0.25. For depth, $\mathrm{Rel}^d = |z - \hat{z}|/z$ and the usual 1.25 threshold.

**Point maps (Table 1, average over eight datasets).**

| Method | Scale-inv. Rel | Scale-inv. delta | Affine-inv. Rel | Affine-inv. delta | Local Rel |
| --- | --- | --- | --- | --- | --- |
| LeReS | 24.6 | 55.6 | 18.1 | 73.3 | 10.7 |
| DUSt3R | 13.8 | 84.7 | 11.4 | 86.0 | 7.97 |
| UniDepth | 12.3 | 87.7 | 9.43 | 90.5 | 9.21 |
| **MoGe** | **7.91** | **93.3** | **6.07** | **94.7** | **5.50** |

The 12.3 to 7.91 drop is the "over 35%" error reduction in the abstract. Local error inside object masks falls from 7.97 to 5.50, about 30%. MoGe is not best everywhere: on KITTI and DDAD, UniDepth (trained with lots of driving data) is level or slightly ahead.

**Depth and field of view.** Average depth error is lowest in every representation: 6.17 scale-invariant (Metric3D V2 6.33, UniDepth 8.43, Depth Anything V2 9.09), 4.72 affine-invariant depth (Marigold 8.41, GeoWizard 8.44), and 5.76 affine-invariant disparity (Depth Anything V2 7.37). For field of view on NYUv2, ETH3D and iBims-1, MoGe's mean error is 2.91 degrees, versus 4.06 for DUSt3R and 7.00 for the dedicated calibrator Wild Camera. DUSt3R still wins on NYUv2 alone (2.57 versus 3.41).

![Figure 7 from Wang et al. (2025): disparity and point maps from MoGe versus LeReS, UniDepth, DUSt3R, Metric3D V2 and Depth Anything V2](/imgs/blogs/moge-unlocking-accurate-monocular-geometry-estimation-for-open-domain-images-with-optimal-fig7.webp)

The qualitative grid shows what the averages hide: Depth Anything V2's disparity looks sharpest, but lifted into 3D (even with ground-truth intrinsics) its car stretches. MoGe's car stays boxy.

**The ablation that carries the argument (Table 4, ViT-Base).**

| Variant | Scale-inv. point Rel | Affine-inv. point Rel | Local Rel | Affine-inv. disparity Rel |
| --- | --- | --- | --- | --- |
| Scale-inv. depth + ray map (SI-log) | 11.2 | 9.09 | 9.19 | 8.23 |
| Affine-inv. depth + shift + ray map | 29.9 | 29.0 | 12.2 | 15.9 |
| Scale-inv. point map, ROE | 10.3 | 8.34 | 8.59 | 7.90 |
| Affine-inv. point map, L2 alignment | 13.5 | 10.3 | 9.48 | 9.37 |
| Affine-inv. point map, median alignment | 10.9 | 8.97 | 9.44 | 8.74 |
| Affine-inv. point map, ROE | 9.84 | 7.88 | 7.62 | 7.43 |
| Full without truncation | 9.81 | 7.91 | 7.12 | 7.45 |
| Full without local loss | 9.98 | 7.94 | 7.47 | 7.47 |
| Full | 9.78 | 7.83 | 7.16 | 7.30 |

Rows 1 to 6 use the global loss only. Read top to bottom, the story is clear: the representation gives the first big step (row 3 to row 6), and the alignment choice gives the second (L2 13.5, median 10.9, ROE 9.84 in scale-invariant point error). The full objective's gains over row 6 are small in these averages, and "without truncation" actually edges out "full" on local error (7.12 versus 7.16). The local loss shows more clearly in the paper's qualitative surfaces than in this table.

**What is load-bearing and may not transfer.** The 9-million-frame mix, with dataset weights balanced by DINOv2-style image retrieval, is doing a lot of quiet work. So is the decision to apply finer losses only to cleaner data. A lab reproducing MoGe on one or two datasets should expect the representation and ROE gains, not the headline averages.

## Critique

**What is strong.** The paper isolates a real flaw in how monocular geometry was supervised, fixes it with a representation change that costs nothing at inference, and proves the point with a clean ablation. The ROE solver is a nice piece of algorithm engineering: an exact optimum for a non-smooth objective at training speed. And deriving focal length from the point map, rather than regressing it, beats specialized calibration networks.

**What is weaker.**
- Gains from the local loss, normal loss and truncation are small in Table 4 and sometimes reversed on single metrics. Most of the qualitative evidence for them is a few surfaces in Figure 6.
- There is no seeds or variance analysis, so differences under about 0.2 in Rel are hard to call.
- The comparison mixes training sets. MoGe sees 21 datasets; baselines see their own. The ViT-B ablation controls this, the headline table does not.
- The centered-principal-point assumption is fine for phone photos but breaks on crops, and the paper evaluates with undistorted, centered images.

**Missing ablation.** A version of DUSt3R or UniDepth retrained on MoGe's data with only the representation swapped would separate "better data" from "better supervision" directly.

**What would change my mind.** If a scale-invariant point-map model trained on the same 9 million frames with an exact L1 scale solver matched MoGe within noise, I would conclude the Z-shift matters less than the paper claims and that alignment quality is the real story. Row 3 of the ablation (10.3 versus 9.84) suggests it would not, but that is one backbone and one data mix.

## Where this goes next

The authors themselves flag video and metric scale as open (their later MoGe-2 targets metric scale). Here are concrete directions that MoGe makes tractable. All are my extrapolation.

1. **Clip-level affine invariance for video.** *Question:* can one shared $(s, t_z)$ per short clip remove the frame-to-frame scale flicker? *Why now:* the ROE solver already handles any point set, so pooling points from 8 frames is a drop-in change. *First experiment:* fine-tune MoGe on synthetic video (Sintel-style sequences with known poses) with a loss that aligns all frames of a clip using one solved scale and shift, then measure the variance of per-frame scale against ground truth.
2. **Metric scale as a separate, small head.** *Question:* can metric scale be learned without re-introducing the focal-distance ambiguity into the shape? *Why now:* MoGe cleanly separates shape (affine-invariant) from camera (derived). *First experiment:* freeze MoGe, train an MLP on pooled encoder features to predict $\log s$ and $t_z$ on metric datasets only, and compare metric depth error with Metric3D V2 and UniDepth.
3. **ROE as a test-time anchor fitter.** *Question:* how few metric points (LiDAR, ARKit, SfM) are needed to lift a MoGe prediction to accurate meters? *Why now:* the truncated ROE solver is robust to outliers and exact for small sets. *First experiment:* on KITTI, sample 10, 100 and 1,000 LiDAR points, solve scale and shift with truncated ROE, and plot metric Rel versus anchor count against least squares.
4. **Local spheres across time for dynamic scenes.** *Question:* can per-sphere alignment over neighbouring frames supervise moving objects whose relative position is ambiguous? *Why now:* the local loss already decouples objects from the global frame, which is what dynamic reconstruction pipelines such as [MegaSaM](/blog/paper-reading/computer-vision/megasam-accurate-fast-and-robust-structure-and-motion-from-casual-dynamic-videos) struggle with. *First experiment:* use MoGe point maps as the depth prior inside a MegaSaM-style optimizer and compare against a depth-only prior on dynamic sequences.

## References

- Wang et al., "MoGe: Unlocking Accurate Monocular Geometry Estimation for Open-Domain Images with Optimal Training Supervision", CVPR 2025. [arXiv 2410.19115](https://arxiv.org/abs/2410.19115), [CVF open access](https://openaccess.thecvf.com/content/CVPR2025/html/Wang_MoGe_Unlocking_Accurate_Monocular_Geometry_Estimation_for_Open-Domain_Images_with_CVPR_2025_paper.html)
- Code and models: [github.com/microsoft/MoGe](https://github.com/microsoft/MoGe)
- Related on this blog: [VGGT: Visual Geometry Grounded Transformer](/blog/paper-reading/computer-vision/vggt-visual-geometry-grounded-transformer), [MegaSaM: accurate, fast and robust structure and motion from casual dynamic videos](/blog/paper-reading/computer-vision/megasam-accurate-fast-and-robust-structure-and-motion-from-casual-dynamic-videos)
