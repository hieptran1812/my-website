---
title: "MegaSaM: Accurate, Fast and Robust Structure and Motion from Casual Dynamic Videos"
publishDate: "2026-10-15"
date: "2026-10-15"
description: "How MegaSaM turns DROID-SLAM into a camera and depth estimator for shaky, dynamic phone videos by adding mono-depth priors, a learned movement map, and a Hessian test that decides when to trust each prior."
category: "paper-reading"
subcategory: "Computer Vision"
tags: ["3d-reconstruction", "cvpr-2025", "cvpr-2025-award-candidate", "paper-reading", "megasam", "visual-slam", "bundle-adjustment", "camera-pose-estimation", "video-depth", "dynamic-scenes"]
author: "Hiep Tran"
featured: true
image: "/imgs/blogs/megasam-accurate-fast-and-robust-structure-and-motion-from-casual-dynamic-videos-fig1.webp"
excerpt: "MegaSaM keeps DROID-SLAM's differentiable bundle adjustment and teaches it to survive moving objects, rotation-only cameras and unknown focal length. The trick is knowing, per video, which priors to trust."
readTime: 19
paper:
  title: "MegaSaM: Accurate, Fast, and Robust Structure and Motion from Casual Dynamic Videos"
  authors: "Zhengqi Li, Richard Tucker, Forrester Cole, Qianqian Wang, Linyi Jin, Vickie Ye, Angjoo Kanazawa, Aleksander Holynski, Noah Snavely"
  venue: "CVPR 2025 (oral, award candidate)"
  url: "https://arxiv.org/abs/2412.04463"
---

> [!tldr]
> **Claim.** A deep visual SLAM system (DROID-SLAM) can recover cameras, focal length and consistent depth from casual phone videos full of moving people and near-zero parallax, if you change how it is trained and initialized and let it decide per video when to lean on a monocular depth prior.
> **Why it matters.** Almost every 4D reconstruction and dynamic novel-view method assumes someone already handed it good cameras. MegaSaM is that someone, at about one second per frame or less.
> **Most surprising result.** On Sintel it cuts calibrated rotation error from 0.17 (the best baseline) to 0.04, and on 12 in-the-wild DynIBaR videos absolute trajectory error drops from 0.016 to 0.004.
> **Where it fails.** Videos where a moving object fills the frame, camera and object moving along the same line (selfies), zoom, and strong lens distortion.

Hand COLMAP a phone clip of a friend walking across a park and you usually get a crash, or a confident reconstruction with your friend smeared into a wall. Classic structure from motion assumes a still world and enough camera motion to triangulate. Casual video breaks both.

MegaSaM (Google DeepMind, UC Berkeley, Michigan) invents no new representation. It takes DROID-SLAM's learned bundle adjustment and adds a movement map that tells the solver which pixels to ignore, a monocular depth prior to break degeneracies, and a check on the solver's own Hessian that decides, per video, how much to trust that prior.

![Figure 1 from Li et al. (2025): input video frames (top) and MegaSaM's recovered cameras and point cloud, unprojected from its predicted video depth with no postprocessing (bottom).](/imgs/blogs/megasam-accurate-fast-and-robust-structure-and-motion-from-casual-dynamic-videos-fig1.webp)

## The problem: three things casual video does to SfM

Given a video $V = \{I_i\}_{i=1}^{N}$, MegaSaM wants per-frame camera poses $\hat G_i \in SE(3)$, one focal length $\hat f$ if it is unknown, and dense depth maps $\hat D_i$. The paper puts no constraint on how the camera or the objects move.

Casual footage attacks that goal three ways:

| What casual video does | Why classic SfM breaks | MegaSaM's answer |
| --- | --- | --- |
| People, cars, dogs move | Moving pixels look like wrong camera motion | Learned movement map $m_i$ downweights them |
| Camera mostly rotates | Depth is unobservable without baseline | Mono-depth prior, switched on by a Hessian test |
| Focal length unknown | Focal and depth trade off against each other | UniDepth focal init, optimized only when observable |

The pipeline has two stages: camera tracking runs differentiable bundle adjustment (BA) on low-resolution disparity, then consistent video depth freezes the cameras and polishes full-resolution depth.

![Figure 9 from Li et al. (2025): system overview. Left, camera tracking through differentiable BA with predicted flow, confidence and movement maps. Right, consistent video depth optimization with cameras and focal fixed.](/imgs/blogs/megasam-accurate-fast-and-robust-structure-and-motion-from-casual-dynamic-videos-fig2.webp)

## The base layer: DROID-SLAM's learned bundle adjustment

**The problem.** Bundle adjustment is the workhorse of SfM: pick cameras and depths so that every observed correspondence reprojects correctly. Hand-crafted correspondences are brittle on blur, texture-less walls and dynamics. DROID-SLAM's idea was to let a network predict the correspondences and their confidence, and to put a BA solver inside the network so the whole thing trains end to end.

**The analogy.** An assistant (a recurrent network) proposes where each pixel of frame $i$ lands in frame $j$ and how sure it is; a surveyor (BA) adjusts cameras and depths to agree with the confident proposals. They iterate until they agree.

**The mechanism.** The state is a low-resolution disparity map $\hat d_i \in \mathbb{R}^{\frac{H}{8}\times\frac{W}{8}}$ per frame plus a pose per frame. A frame graph $\mathcal{P}$ connects frames with overlapping views. For each pair $(I_i, I_j) \in \mathcal{P}$ a convolutional GRU $F$ predicts a 2D correspondence field $\hat u_{ij}$ and a confidence $\hat w_{ij}$, refined over iterations $k$.

The geometry gives a second correspondence field, the one the current cameras and depths imply:

$$
u_{ij} = \pi\!\left(\hat G_{ij} \circ \pi^{-1}(p_i, \hat d_i, K^{-1}),\; K\right)
$$

Here $p_i$ is the grid of pixel coordinates, $\pi^{-1}$ lifts each pixel to 3D using its disparity and the inverse intrinsics, $\hat G_{ij} = \hat G_j \circ \hat G_i^{-1}$ moves it into frame $j$'s coordinates, and $\pi$ projects it back with intrinsics $K \in \mathbb{R}^{3\times 3}$.

BA then minimizes the weighted disagreement between what the network sees and what the geometry implies. MegaSaM adds the focal length $\hat f$ to the unknowns, since DROID-SLAM assumed it known:

$$
\mathcal{C}(\hat G, \hat d, \hat f) = \sum_{(i,j)\in\mathcal{P}} \left\| \hat u_{ij} - u_{ij} \right\|^2_{\Sigma_{ij}}, \qquad \Sigma_{ij} = \operatorname{diag}(\hat w_{ij})^{-1}
$$

Confident correspondences (large $\hat w_{ij}$) get the larger say.

### Why the solver is fast: the Schur complement

**The problem.** One Levenberg–Marquardt step solves $(J^\top W J + \lambda \operatorname{diag}(J^\top W J))\,\Delta\xi = J^\top W r$, where $J$ is the Jacobian of residuals, $W$ the diagonal weight matrix, $r$ the residuals, and $\lambda$ a damping factor that the network itself predicts. The update $\Delta\xi = (\Delta G, \Delta d, \Delta f)$ covers every pose, the focal length, and every disparity pixel. That is a big system.

**The mechanism.** Split the approximate Hessian into a camera block and a disparity block:

$$
\begin{bmatrix} H_{G,f} & E_{G,f} \\ E_{G,f}^\top & H_d \end{bmatrix}
\begin{bmatrix} \Delta\xi_{G,f} \\ \Delta d \end{bmatrix}
=
\begin{bmatrix} \tilde r_{G,f} \\ \tilde r_d \end{bmatrix}
$$

Each reprojection term involves exactly one disparity variable, the one at pixel $p_i$ in frame $i$. So $H_d$ is diagonal, and inverting it is just taking reciprocals. Eliminate $\Delta d$ first:

$$
\Delta\xi_{G,f} = \left[H_{G,f} - E_{G,f} H_d^{-1} E_{G,f}^\top\right]^{-1}\left(\tilde r_{G,f} - E_{G,f} H_d^{-1} \tilde r_d\right), \qquad
\Delta d = H_d^{-1}\left(\tilde r_d - E_{G,f}^\top \Delta\xi_{G,f}\right)
$$

![Redrawn: the approximate Hessian is tiny and dense for poses plus focal, enormous but diagonal for disparity. The Schur complement folds the disparity block away so only the camera block is solved densely.](/imgs/blogs/megasam-accurate-fast-and-robust-structure-and-motion-from-casual-dynamic-videos-2.webp)

**Worked example (my sizes, for illustration).** A 50-frame Sintel clip at 1024 by 436 has a disparity grid of 128 by 54, or 6,912 unknowns per frame and 345,600 in total. The camera side has ${6} \times 50 + 1 = 301$ unknowns. Instead of factoring a 345,901-square system, you invert 345,600 scalars and factor a 301-square matrix, all differentiably.

**When it fails.** A regularizer coupling neighbouring pixels would make $H_d$ non-diagonal. MegaSaM's mono-depth term pulls each pixel toward its own prior, so the structure survives.

Training of this base model is plain supervised learning on synthetic static scenes, $\mathcal{L}_{\text{static}} = \mathcal{L}_{\text{cam}} + w_{\text{flow}}\mathcal{L}_{\text{flow}}$, where the camera loss compares poses and the flow loss compares ego-motion induced flow with ground truth. The paper sets $w_{\text{flow}} = 0.02$.

## Technique 1: a movement map, trained in a separate stage

**The problem.** BA treats every confident correspondence as evidence about the camera. A person walking left produces confident, consistent flow that says "the camera moved right". Plain DROID-SLAM believes it.

**The obvious fix fails.** Training the existing confidence $\hat w_{ij}$ on dynamic videos, hoping it learns to go low on movers, trains unstably through the BA layer, the authors report.

**The analogy.** "Do these two patches match?" is a question about one pair of frames. "Is this patch moving?" is a question about many frames. MegaSaM gives the second question its own network.

**The mechanism.** A second network $F_m$ predicts a per-frame movement probability map $m_i \in \mathbb{R}^{\frac{H}{8}\times\frac{W}{8}}$, conditioned on frame $i$ and its keyframe neighbours $\mathcal{N}(i) = \{I_j \mid (i,j)\in\mathcal{P}\}$. Per the supplement, it spatially average-pools for global context, then average-pools along time to fuse $I_i$ with all its neighbours. The BA weight becomes a product:

$$
\tilde w_{ij} = \hat w_{ij}\, m_i
$$

In the product $m_i$ acts as the weight a pixel keeps, so pixels on movers end up near zero, as drawn below.

![Redrawn: each BA residual is weighted by flow confidence times the movement map, here drawn as a static probability that is low on movers. A moving blob and an occluded edge both drop out of the final weight.](/imgs/blogs/megasam-accurate-fast-and-robust-structure-and-motion-from-casual-dynamic-videos-3.webp)

**Training in two stages.**

1. **Ego-motion pretraining.** Train $F$ with $\mathcal{L}_{\text{static}}$ on static data only: 163 TartanAir scenes and 5K static Kubric videos.
2. **Dynamic fine-tuning.** Freeze $F$. Train $F_m$ on 11K dynamic Kubric videos, fed with $F$'s features, using $\mathcal{L}_{\text{dynamic}} = \mathcal{L}_{\text{cam}} + w_{\text{motion}}\mathcal{L}_{\text{CE}}$ with $w_{\text{motion}} = 0.1$. The cross-entropy term supervises $m_i$ against ground-truth dynamic masks.

Each training example is a 7-frame clip, sampled so the average ego-motion flow between frames lies between 0.5 and 64 pixels. Training the tracker takes about 4 days on 8 A100 80GB GPUs.

**Worked example.** A pixel on a static wall with flow confidence 0.9 and movement weight 1.0 keeps weight 0.9. A pixel on a walking person with confidence 0.9 (the flow is sharp, the person is textured) but movement weight 0.05 drops to 0.045. Confidence alone could not do this: the person's flow really is reliable, it is just not about the camera.

**When it fails.** If movers cover most of the frame, there is little static evidence left to weight. The supplement's failure figure shows exactly this (a dirt-bike clip where the rider dominates). Movement also has to be distinguishable from parallax, which gets hard when the object moves along the camera's own direction.

## Technique 2: start from mono-depth instead of a constant

**The problem.** DROID-SLAM initializes every disparity to 1. With a healthy baseline the solver climbs out of that guess. With a rotating camera and moving objects, it does not, and the cameras that come out are wrong.

**The mechanism at inference.** Initialize each frame's disparity from a metric-aligned monocular prediction:

$$
D_i^{\text{align}} = \hat\alpha\, D_i^{\text{rel}} + \hat\beta
$$

$D_i^{\text{rel}}$ is affine-invariant disparity from DepthAnything, which the authors find more accurate and temporally consistent. The scale $\hat\alpha$ and shift $\hat\beta$ come from median-aligning it with metric depth $D_i^{\text{abs}}$ from UniDepth, which knows absolute scale. UniDepth also predicts a focal length per frame; the median across the video initializes $\hat f$, which stays fixed during the frontend.

**Worked example.** Take three pixels with relative disparity $[0.2, 0.5, 0.8]$ and UniDepth values $[0.15, 0.30, 0.45]$. Medians are 0.5 and 0.30. Centered ratios are $(0.15-0.30)/(0.2-0.5) = 0.5$ and $(0.45-0.30)/(0.8-0.5) = 0.5$, so $\hat\alpha = 0.5$. The shift is the median of $D^{\text{abs}} - \hat\alpha D^{\text{rel}} = [0.05, 0.05, 0.05]$, so $\hat\beta = 0.05$. The paper writes the formulas per pixel; I read $\hat\alpha$ as the median of those centered ratios, which is my interpretation.

**During training.** DepthAnything disparity is aligned with the ground truth's scale and shift, the first two poses are set to ground truth to fix the gauge, and focal length starts randomly perturbed by 25%.

**Frontend tracking.** Keyframes accumulate until there are $N_{\text{init}} = 8$ with enough motion. Their poses are solved by a motion-only BA with disparity frozen to the aligned prior. Then a sliding window adds and drops keyframes, and each local BA carries a prior term:

$$
\mathcal{C} = \sum_{(i,j)\in\mathcal{P}} \left\| \hat u_{ij} - u_{ij} \right\|^2_{\Sigma_{ij}} + w_d \sum_i \left\| \hat d_i - D_i^{\text{align}} \right\|^2
$$

with $w_d = 0.05$ in the frontend. Each term touches one pixel's disparity, so the Schur trick survives.

![Figure 2 from Li et al. (2025): on near-rotational sequences, (a) without mono-depth initialization and (b) without uncertainty-aware BA the cameras and geometry collapse; (c) the full system recovers them.](/imgs/blogs/megasam-accurate-fast-and-robust-structure-and-motion-from-casual-dynamic-videos-fig3.webp)

**When it fails.** With enough parallax, a constant pull toward a biased prior makes cameras worse. The next technique resolves that tension.

## Technique 3: ask the Hessian whether the geometry is observable

**The problem.** The backend runs a global BA over keyframes, registers the remaining frames with a pose graph, then runs a global BA over every frame. Should that global BA include the mono-depth term? With a big baseline, no: the authors observe it hurts accuracy. With a rotating camera, yes: without it BA finds degenerate solutions. The system has to decide without ground truth.

**The analogy.** If wiggling a variable barely changes the cost, the data has nothing to say about it, and the prior should speak instead. BA can measure that wiggle.

![Redrawn: under pure rotation, near, mid and far points on one ray all reproject to the same pixel and the reprojection error is flat in depth. With a baseline they land on different pixels and the error curve has a clear minimum.](/imgs/blogs/megasam-accurate-fast-and-robust-structure-and-motion-from-casual-dynamic-videos-1.webp)

**The math.** Under a Laplace approximation of the posterior $p(\theta \mid \mathcal{I})$, the covariance of the parameters at the MAP estimate $\theta^*$ is the inverse Hessian, $\Sigma_\theta = -H(\theta^*)^{-1}$. This is epistemic uncertainty: what the data fails to pin down. Inverting the full Hessian is too expensive for long videos, so following Ritter et al. the paper keeps only the diagonal:

$$
\Sigma_\theta \approx \operatorname{diag}\left(-H(\theta^*)\right)^{-1}
$$

The Jacobian says how much reprojection error moves when you nudge a variable. A static camera leaves reprojection unchanged as disparity changes, so the disparity entries of the Hessian are near zero and their variance blows up. That is the formal version of the left panel above.

**Worked example (mine, one dimension).** Suppose one pixel's residual is $r(d) = s\,(d - d^*)$, where the sensitivity $s$ grows with baseline. The Hessian entry is $s^2$ and the variance is ${1}/s^2$. A near-rotational clip with $s = 0.01$ gives variance 10,000; a walking clip with $s = 1$ gives variance 1. Same pixel, same scene, four orders of magnitude apart.

![Figure 4 from Li et al. (2025): camera path, reference image and disparity uncertainty. The rotation-dominant camel clip (top) has a far larger uncertainty range than the forward-moving clip (bottom), whose uncertainty peaks at the epipole.](/imgs/blogs/megasam-accurate-fast-and-robust-structure-and-motion-from-casual-dynamic-videos-fig4.webp)

**The switch.** After the frontend, MegaSaM collects the disparity diagonal from all keyframes and the focal entry $H_f$, and sets two knobs:

$$
w_d = \gamma_d \exp\!\left(-\beta_d \operatorname{med}\left(\operatorname{diag}(H_d)\right)\right), \qquad \text{optimize } f \text{ only if } H_f \ge \tau_f
$$

with $\gamma_d = 10^{-4}$, $\beta_d = 0.05$, $\tau_f = 50$. Small curvature means weak constraint, which means a strong prior. If $H_f \lt \tau_f$, the focal length stays at the UniDepth estimate.

**Worked example.** With a median disparity curvature of 10, $w_d = 10^{-4} \cdot e^{-0.5} \approx 6.1\times 10^{-5}$. At 60 it is ${10}^{-4} \cdot e^{-3} \approx 5.0\times 10^{-6}$, about twelve times weaker.

![Redrawn: left, the mono-depth weight decays exponentially as the median disparity curvature grows, so rotation-heavy clips keep the prior and large-baseline clips shed it; right, focal length is frozen below the threshold and optimized above it.](/imgs/blogs/megasam-accurate-fast-and-robust-structure-and-motion-from-casual-dynamic-videos-4.webp)

**When it fails.** One median per video is blunt: a clip that rotates and then walks gets one compromise weight. Diagonal Laplace also ignores correlations such as the focal-depth trade-off, so a confident diagonal can hide a sloppy direction. The paper reports no failure analysis of the rule itself.

## Technique 4: consistent video depth without fine-tuning a network

**The problem.** BA disparity lives at one-eighth resolution. Users want sharp, temporally stable depth at full resolution. CasualSAM got there by fine-tuning a depth network per video, which is slow.

**The mechanism.** With cameras fixed, optimize a stack of per-frame disparity maps $\hat D_i$ and aleatoric uncertainty maps $\hat M_i$ directly, by first-order optimization:

$$
\mathcal{C}_{\text{cvd}} = w_{\text{flow}}\,\mathcal{C}_{\text{flow}} + w_{\text{temp}}\,\mathcal{C}_{\text{temp}} + w_{\text{prior}}\,\mathcal{C}_{\text{prior}}
$$

- $\mathcal{C}_{\text{flow}}$: L1 gap between RAFT flow and the flow implied by cameras and $\hat D_i$, scaled by $\hat M_i$ plus $\log(1/\hat M_i)$. A Laplacian likelihood: a pixel may be declared noisy at a log cost, and movers end up noisy.
- $\mathcal{C}_{\text{temp}}$: disparity carried by flow into the next frame should match, same weighting.
- $\mathcal{C}_{\text{prior}} = \mathcal{C}_{\text{si}} + w_{\text{grad}}\mathcal{C}_{\text{grad}} + w_{\text{normal}}\mathcal{C}_{\text{normal}}$: scale-invariant log-disparity loss to the aligned mono-depth, gradient matching where the estimate drifts, and surface-normal agreement.

Weights: $w_{\text{flow}} = w_{\text{prior}} = 1$, $w_{\text{temp}} = 0.2$, $w_{\text{grad}} = 1$, $w_{\text{normal}} = 4$; frame gaps 1, 2, 4, 8, 15. Uncertainty starts from the movement maps; 100 warm-up steps with disparity frozen, then 400 joint steps, at 1.3 FPS on Sintel at 336 by 144.

The differences from CasualSAM: optimize variables rather than network weights, keep cameras fixed, and swap in the normal and gradient prior terms.

**When it fails.** It trusts the cameras completely; a wrong camera bakes its error into every depth map.

## Results

Trajectories are normalized to unit length and Sim(3)-aligned; time is per frame on one A100.

**Sintel** (18 synthetic sequences of 20 to 50 frames):

| Method | ATE (cal.) | RRE (cal.) | ATE (uncal.) | RRE (uncal.) | Time |
| --- | --- | --- | --- | --- | --- |
| CasualSAM | 0.036 | 0.20 | 0.067 | 0.47 | 1.6 min |
| LEAP-VO | 0.041 | 0.17 | n/a | n/a | 1.3 s |
| ACE-Zero | 0.053 | 0.30 | 0.065 | 1.92 | 10 s |
| Particle-SfM | 0.062 | 1.26 | 0.057 | 1.64 | 21 s |
| MonST3R | n/a | n/a | 0.078 | 0.49 | 1.0 s |
| **MegaSaM** | **0.018** | **0.04** | **0.023** | **0.06** | 1.0 s |

**DyCheck** (real handheld videos, 180 to 500 frames) and **in-the-wild** (12 DynIBaR videos, 100 to 600 frames, ground truth from COLMAP with hand-specified movers masked):

| Method | DyCheck ATE (uncal.) | DyCheck time | Wild ATE (uncal.) | Wild RRE (uncal.) | Wild time |
| --- | --- | --- | --- | --- | --- |
| CasualSAM | 0.209 | 2.8 min | 0.035 | 0.30 | 1.1 min |
| ACE-Zero | 0.056 | 1.6 s | 0.091 | 0.09 | 4.0 s |
| Particle-SfM | 0.087 | 35 s | 0.054 | 0.14 | 49 s |
| MonST3R | 0.690 | 1.0 s | 0.073 | 0.18 | 1.7 s |
| **MegaSaM** | **0.020** | 0.8 s | **0.004** | **0.02** | 0.7 s |

**Video depth** (abs-rel, lower is better; $\delta_{1.25}$, higher is better):

| Method | Sintel abs-rel | Sintel $\delta_{1.25}$ | DyCheck abs-rel | DyCheck $\delta_{1.25}$ |
| --- | --- | --- | --- | --- |
| DepthAnything-V2 | 0.37 | 58.6 | 0.20 | 84.7 |
| DepthCrafter | 0.27 | 68.2 | 0.22 | 83.7 |
| CasualSAM | 0.31 | 64.2 | 0.21 | 78.4 |
| MonST3R | 0.31 | 62.5 | 0.26 | 66.5 |
| **MegaSaM** | **0.21** | **73.1** | **0.11** | **94.1** |

The DAVIS comparison makes the robustness point better than the tables: narrow field of view, rotation-dominant motion, exactly where the Hessian switch matters.

![Figure 7 from Li et al. (2025): two DAVIS clips with rotation-dominant motion and narrow field of view. CasualSAM distorts the geometry, MonST3R turns rotation into translation, MegaSaM keeps both cameras and structure coherent.](/imgs/blogs/megasam-accurate-fast-and-robust-structure-and-motion-from-casual-dynamic-videos-fig5.webp)

**Ablation on Sintel** (Table 5):

| Configuration | ATE | RTE | RRE |
| --- | --- | --- | --- |
| DROID-SLAM | 0.030 | 0.022 | 0.50 |
| w/o mono-depth init | 0.038 | 0.026 | 0.49 |
| w/o movement map | 0.032 | 0.127 | 0.14 |
| w/o two-stage training | 0.035 | 0.136 | 0.17 |
| w/o uncertainty-aware BA (prior always on) | 0.033 | 0.013 | 0.11 |
| Full | 0.019 | 0.008 | 0.04 |

Rotation error tells the story: 0.50 for DROID-SLAM, 0.04 for the full system. Two oddities: the RTE values for the movement-map and two-stage rows (0.127, 0.136) exceed their own ATE, possibly a typo, and the full model's ATE is 0.019 here against 0.018 in Table 1.

**What might not transfer.** The tracker trains only on synthetic data yet generalizes to real video, which suggests the depth priors carry much of the domain gap; a weaker mono-depth model would degrade the rotation case with it. The Hessian constants are tied to the paper's normalization (disparity scaled so its 98th percentile is 2, focal divided by image resolution) and may need retuning on new footage.

## Critique

**What is strong.**

- A clean split: the network proposes correspondences and motion, a real solver enforces multi-view consistency. That is why it runs under a second per frame and still generalizes.
- The observability test is a cheap, ground-truth-free signal for when a prior should speak, reusable far beyond this paper.
- The gains are not marginal. MegaSaM wins every camera metric on all three benchmarks, and on Sintel its rotation error is about a quarter of the best baseline's.

**What is weak.**

- The rule that uses the Hessian is a single median per video with hand-tuned constants. There is no sensitivity study of $\beta_d$, $\gamma_d$ or $\tau_f$.
- In-the-wild ground truth is COLMAP with hand-masked movers, which is weakest on exactly the hardest videos.
- Missing ablation: an older depth model would separate "better SLAM" from "better priors".
- Failure modes are shown but not counted, so we do not know how common they are.

**What would change my mind.** If a sweep showed that the median-curvature switch helps only on Sintel-like synthetic rotation and is neutral or harmful on a large, diverse set of real phone videos, I would treat the uncertainty-aware BA as a benchmark-specific patch rather than a general principle. Equally, if a feed-forward model like [VGGT](/blog/paper-reading/computer-vision/vggt-visual-geometry-grounded-transformer) matched MegaSaM's rotation error on DAVIS-style clips without any optimization, the case for keeping a solver in the loop would weaken.

## Where this goes next

The series note flagged two directions: distilling MegaSaM into a feed-forward model, and using it as a data engine for 4D training. Here they are as concrete research questions. These are my proposals, not the paper's.

1. **Distill into a feed-forward 4D model, with the Hessian as a label filter.** *Question:* can a single-pass network learn MegaSaM's cameras and depth from its pseudo-labels on unlabeled casual video? *Why now:* at about a second per frame, labeling a huge corpus is affordable, and the Hessian diagnostics give a per-video confidence most pseudo-labelers lack. *First experiment:* fine-tune a VGGT-style backbone on MegaSaM labels twice, once on everything and once dropping weak-curvature videos, and compare ATE and RRE against the teacher on Sintel, DyCheck and DynIBaR.

2. **Validate the diagnostic as a quality score.** *Question:* does median disparity curvature (with $H_f$) predict MegaSaM's actual error? *Why now:* it is already computed in the backend and nobody has checked it. *First experiment:* log both diagnostics and the final ATE and RRE on every benchmark video and report rank correlation. Strong correlation makes MegaSaM a self-grading data engine.

3. **Per-pixel, per-window priors.** *Question:* does $w_d$ derived from each pixel's own Hessian entry beat one global median? *Why now:* the diagonal is already per pixel and the sliding window gives time segments. *First experiment:* concatenate rotation and walking segments from Kubric, where a global weight must compromise, and compare the two weightings.

4. **Attack the declared failures.** *Question:* can a per-frame focal length gated by a windowed $H_f$ handle zoom, and can a 3D tracking prior break the selfie degeneracy? *Why now:* the gating machinery only needs local statistics, and dense 4D tracking such as [D4RT](/blog/paper-reading/computer-vision/efficiently-reconstructing-dynamic-scenes-one-d4rt-at-a-time) offers motion cues that are not purely 2D. *First experiment:* Kubric clips with a zoom ramp and with an object moving along the optical axis; measure focal error and ATE.

5. **Self-train the movement head on real video.** *Question:* do masks from MegaSaM's own CVD uncertainty $\hat M_i$ improve $F_m$ beyond synthetic training? *Why now:* $F$ stays frozen, so only the movement head is at risk. *First experiment:* one self-training round on DAVIS-like clips, evaluated on held-out DynIBaR, watching whether the failure-mode videos get worse.

## References

- Li, Tucker, Cole, Wang, Jin, Ye, Kanazawa, Holynski, Snavely. *MegaSaM: Accurate, Fast, and Robust Structure and Motion from Casual Dynamic Videos.* CVPR 2025. [arXiv:2412.04463](https://arxiv.org/abs/2412.04463). [CVF open access](https://openaccess.thecvf.com/content/CVPR2025/html/Li_MegaSaM_Accurate_Fast_and_Robust_Structure_and_Motion_from_Casual_CVPR_2025_paper.html). Project page: [mega-sam.github.io](https://mega-sam.github.io).
- Teed and Deng. *DROID-SLAM.* NeurIPS 2021. Zhang et al. *CasualSAM.* ECCV 2022. Zhang et al. *MonST3R.* 2024.
- Related on this site: [VGGT: Visual Geometry Grounded Transformer](/blog/paper-reading/computer-vision/vggt-visual-geometry-grounded-transformer) and [D4RT: efficiently reconstructing dynamic scenes](/blog/paper-reading/computer-vision/efficiently-reconstructing-dynamic-scenes-one-d4rt-at-a-time).
