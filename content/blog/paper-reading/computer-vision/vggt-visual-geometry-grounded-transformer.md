---
title: "VGGT: Visual Geometry Grounded Transformer, or How One Forward Pass Replaced Structure-from-Motion"
publishDate: "2026-10-15"
date: "2026-10-15"
description: "A close read of VGGT, the CVPR 2025 best paper: a plain 1.2B-parameter transformer that maps one to hundreds of images to cameras, depth, point maps and tracks in a single pass, and beats optimization pipelines while doing it."
excerpt: "VGGT feeds every image of a scene into one large transformer that alternates frame-wise and global attention, and reads out cameras, depth maps, point maps and point tracks in under a second. Here is how each piece works, why the over-complete outputs help, and where the idea goes next."
category: "paper-reading"
subcategory: "Computer Vision"
tags: ["3d-reconstruction", "cvpr-2025", "cvpr-2025-award-candidate", "paper-reading", "vggt", "structure-from-motion", "multi-view-geometry", "point-maps", "camera-pose-estimation", "point-tracking", "transformer", "3d-foundation-model"]
author: "Hiep Tran"
featured: true
image: "/imgs/blogs/vggt-visual-geometry-grounded-transformer-fig1.webp"
readTime: 18
paper:
  title: "VGGT: Visual Geometry Grounded Transformer"
  authors: "Jianyuan Wang, Minghao Chen, Nikita Karaev, Andrea Vedaldi, Christian Rupprecht, David Novotny"
  venue: "CVPR 2025 (oral, best paper award); arXiv:2503.11651"
  url: "https://arxiv.org/abs/2503.11651"
---

> [!tldr]
> - **What it claims.** A fairly standard large transformer, with almost no 3D inductive bias, can take one, a few, or hundreds of images of a scene and output every camera, depth map, point map and 2D point track in one forward pass. No bundle adjustment, no global alignment, no triangulation.
> - **Why it matters.** Feed-forward VGGT reaches 85.3 AUC@30 on RealEstate10K camera pose in about 0.2 s, against 78.9 for VGGSfM v2, which needs about 10 s of optimization. It is the first convincing "3D foundation model" in the sense the field uses for CLIP or DINO.
> - **The surprising result.** The outputs are deliberately redundant (point maps can be computed from depth plus cameras), yet training on all of them helps, and at test time the *derived* point cloud beats the directly predicted one.
> - **Where it is soft.** Static scenes only (it fails under substantial non-rigid motion), no fisheye or panoramas, trouble with extreme rotations, and global attention cost grows quadratically with frame count: 200 frames takes 40.63 GB on an H100.

![Figure 1 from Wang et al. (2025): VGGT takes up to hundreds of images and predicts cameras, point maps, depth maps and point tracks for all of them at once.](/imgs/blogs/vggt-visual-geometry-grounded-transformer-fig1.webp)

## The problem: 3D reconstruction was still an optimization problem

For thirty years, turning photos into 3D meant Structure-from-Motion (SfM): match keypoints, triangulate, then run bundle adjustment (BA), a large nonlinear least-squares solve over every camera and point. Deep learning improved every stage, up to differentiable BA in VGGSfM, but the solver stayed at the center. That keeps systems slow and brittle.

DUSt3R and MASt3R regress aligned point clouds directly, but only from a *pair* of images. Larger scenes need many pairwise predictions plus a "global alignment" optimization, which costs seconds and runs out of memory past about 32 frames. VGGT asks: what if the network sees *all* the images at once and outputs everything directly?

## What goes in and what comes out

The input is a sequence of $N$ RGB images $(I_i)_{i=1}^N$, each $I_i \in \mathbb{R}^{3 \times H \times W}$, all observing the same static scene. The network $f$ maps them to one bundle of 3D annotations per frame:

$$
f\big((I_i)_{i=1}^{N}\big) = \big(g_i,\ D_i,\ P_i,\ T_i\big)_{i=1}^{N}
$$

Every symbol here is a per-frame output:

| Symbol | Shape | Meaning |
| --- | --- | --- |
| $g_i$ | $\mathbb{R}^9$ | camera: rotation quaternion $q \in \mathbb{R}^4$, translation $t \in \mathbb{R}^3$, field of view $f \in \mathbb{R}^2$ |
| $D_i$ | $\mathbb{R}^{H \times W}$ | depth of every pixel as seen from camera $i$ |
| $P_i$ | $\mathbb{R}^{3 \times H \times W}$ | 3D point of every pixel, in the **first camera's** coordinate frame |
| $T_i$ | $\mathbb{R}^{C \times H \times W}$ | dense $C$-dimensional features fed to a separate tracking head |

Two conventions matter. The principal point is assumed to sit at the image center, so focal length reduces to the field-of-view pair. And the first image is special: it defines the world frame, so its rotation is fixed to $q_1 = [0,0,0,1]$ and its translation to $t_1 = [0,0,0]$. The rest of the frames can arrive in any order; the network is permutation equivariant for all frames but the first.

Notice the redundancy: cameras plus depth give $P_i$ in closed form, and $P_i$ gives cameras via Perspective-n-Point. Predicting all of them anyway turns out to be one of the paper's best decisions.

## The backbone: DINO patches plus Alternating-Attention

![Figure 2 from Wang et al. (2025): images are patchified by DINO, a camera token is appended per frame, then the network alternates frame-wise and global self-attention L times before a camera head and DPT heads read out the predictions.](/imgs/blogs/vggt-visual-geometry-grounded-transformer-fig2.webp)

The architecture in Figure 2 is short to describe. Each image is cut into $K$ patch tokens $t^I_i \in \mathbb{R}^{K \times C}$ by a pretrained DINOv2 encoder (14 by 14 pixel patches). Each frame gets one extra camera token and four register tokens. All tokens from all frames then go through $L = 24$ blocks, and each block contains one frame-wise self-attention layer and one global self-attention layer.

The [ViT, SigLIP and DINO explainer](/blog/machine-learning/computer-vision/vit-siglip-dino-explained) covers why DINO features transfer so well. A plain 14 by 14 convolution patchifier was also tried; DINOv2 was more accurate and far more stable early in training.

### Alternating-Attention, ladder style

**The problem.** A transformer over all frames needs two things that pull against each other. It must mix information *across* images (that is what makes triangulation possible), and it must keep each image's own statistics intact (exposure, scale, which tokens belong together spatially).

**The analogy.** Think of a team of surveyors, one per photo. Frame-wise attention is each surveyor studying their own photo in private. Global attention is the meeting where everyone compares notes. VGGT runs 24 rounds of "study, then meet".

**The mechanism.** In a frame-wise layer, token $j$ of frame $i$ attends only to tokens of frame $i$. In a global layer, it attends to every token of every frame. There is no cross-attention anywhere, only self-attention with two different masks.

![Alternating-Attention as two attention masks: frame-wise self-attention is block-diagonal and keeps per-image statistics, global self-attention is fully dense and fuses across all images; each of the 24 blocks applies one of each.](/imgs/blogs/vggt-visual-geometry-grounded-transformer-1.webp)

**The math.** Stack all tokens into one sequence $X \in \mathbb{R}^{N K' \times C}$, where $K' = K + 5$ counts patches plus the camera token and 4 registers. Both layers compute masked attention with the same formula and differ only in the mask $M$:

$$
\mathrm{Attn}(X) = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d}} + M\right)V,
\qquad
M_{ab} =
\begin{cases}
0 & \text{global layer, or } \mathrm{frame}(a) = \mathrm{frame}(b) \\
-\infty & \text{frame-wise layer and } \mathrm{frame}(a) \neq \mathrm{frame}(b)
\end{cases}
$$

Here $Q, K, V$ are the query, key and value projections of $X$, $d$ is the per-head dimension, and $a, b$ index tokens. The frame-wise mask is block-diagonal; the global mask is all zeros. (This is my notation; the paper describes the masks in words, and in practice a frame-wise layer is simply implemented by reshaping frames into the batch dimension.)

**A worked example (my arithmetic).** At the paper's timing resolution of 336 by 518 pixels, a frame gives ${24 \times 37 = 888}$ patches, so $K' = 893$ tokens. With 200 frames, a global layer attends over ${178{,}600}$ tokens, about ${3.2 \times 10^{10}}$ attention scores per head. A frame-wise layer computes 200 separate ${893 \times 893}$ blocks, about ${1.6 \times 10^{8}}$. The ratio is exactly $N = 200$, which is why the global layers dominate cost at scale.

**When it breaks.** The ablation in Table 5 holds parameter count fixed at ${2L}$ attention layers and compares three backbones on ETH3D point maps (Overall Chamfer, lower is better):

| Backbone | Acc. | Comp. | Overall |
| --- | --- | --- | --- |
| Cross-attention | 1.287 | 0.835 | 1.061 |
| Global self-attention only | 1.032 | 0.621 | 0.827 |
| Alternating-Attention | 0.901 | 0.518 | 0.709 |

Global-only is clearly worse, so the frame-wise layers pull their weight. Cross-attention is worst and slower as frame count grows.

### The first-frame tokens: how the network knows where "world" is

**The problem.** Outputs are defined in camera 1's frame, but self-attention is permutation equivariant. Without a marker, the network has no way to tell which frame is the reference.

**The mechanism.** Frame 1 gets its own learnable camera and register tokens $\bar{t}^g, \bar{t}^R$; every other frame shares a second set $t^g, t^R$. Frame-wise layers make them frame-specific. Output registers are discarded; output camera tokens feed the camera head.

Register tokens give attention a place to park global information; the [StreamingLLM attention-sinks post](/blog/paper-reading/large-language-model/streaming-llm-attention-sinks) covers the LLM version of that phenomenon.

**When it breaks.** Everything is relative to frame 1. If frame 1 is a blurry outlier or shares little with the rest, every output inherits that weakness. The paper does not study the choice of reference frame.

## The heads: four readouts from one trunk

Two head types sit on the trunk. A **camera head** takes the 24th-layer camera tokens, runs four more self-attention layers and a linear layer, and outputs $\hat{g}_i$. A **DPT head** takes image tokens from blocks 4, 11, 17 and 23, upsamples them to a dense feature map $F_i \in \mathbb{R}^{C'' \times H \times W}$, and a ${3 \times 3}$ convolution reads out depth $D_i$, point map $P_i$, and tracking features $T_i$.

The DPT head also predicts aleatoric uncertainty maps $\Sigma^D_i$ and $\Sigma^P_i$ in $\mathbb{R}_+^{H \times W}$, one positive number per pixel. Those maps show up in the loss below, and after training they act as per-pixel confidence.

![Two routes to the same 3D point: route A reads the point map P_i(y) directly in camera-1 coordinates, route B takes depth D_i(y) along the ray from camera i and unprojects it with the predicted camera g_i; on ETH3D route B is more accurate.](/imgs/blogs/vggt-visual-geometry-grounded-transformer-2.webp)

### Over-complete predictions, and why the derived answer wins

Here is the counterintuitive part. At test time you have two ways to get a 3D point for pixel $y$ of frame $i$, drawn above. Route A: read $P_i(y)$ from the point head. Route B: take depth $D_i(y)$ and the predicted camera $\hat{g}_i$ and unproject:

$$
\hat{P}_i(y) = R_1^{\,i}\!\left(D_i(y)\, K_i^{-1}\begin{bmatrix} y \\ 1 \end{bmatrix}\right) + t_1^{\,i}
$$

where $K_i$ is the intrinsics matrix built from the predicted field of view (principal point at the center), $K_i^{-1}[y;1]$ is the viewing ray through pixel $y$, and $(R_1^{\,i}, t_1^{\,i})$ is the rigid transform from camera $i$ to camera 1, derived from $\hat{g}_i$. (The equation is my standard pinhole write-up of "unproject with the predicted camera"; the paper states it in words.)

On ETH3D, route B scores 0.677 Overall Chamfer against 0.709 for route A. The authors' explanation: depth and camera are simpler sub-problems than a full viewpoint-invariant point map, so composing two easier predictions beats one hard one.

But the heads still help each other during training. Table 6 drops one loss at a time:

| Trained with | Overall Chamfer (ETH3D) |
| --- | --- |
| no camera loss | 0.834 |
| no depth loss | 0.727 |
| no track loss | 0.790 |
| all three | 0.709 |

Camera supervision matters most. Tracking matters more than depth: a 2D correspondence objective improves 3D accuracy, presumably by forcing the trunk to agree on which pixels are the same physical point.

## The tracking head: correspondence on unordered images

**The problem.** Point tracking normally assumes a video: frames are ordered and motion is small between neighbors. VGGT's inputs are an unordered set, so the tracker cannot lean on temporal smoothness.

**The mechanism.** VGGT reuses the CoTracker2 architecture on top of its dense features $T_i$. Given query point $y_j$ in query image $I_q$, bilinearly sample $T_q$ at $y_j$ to get a $C$-dimensional vector. Correlate that vector with every other frame's feature map to get one $H \times W$ correlation map per frame. A stack of self-attention layers turns those maps into a predicted 2D location $\hat{y}_{j,i}$ in every frame, plus a visibility flag.

![Tracking head as tensors: the query feature is sampled from the C x H x W map T_q, correlated against every other frame's map to give an H x W correlation map, and refined by self-attention into a predicted 2D point and visibility.](/imgs/blogs/vggt-visual-geometry-grounded-transformer-4.webp)

**The math.** The tracking loss is a plain sum of distances between predicted and true correspondences:

$$
\mathcal{L}_{\text{track}} = \sum_{j=1}^{M} \sum_{i=1}^{N} \big\lVert y_{j,i} - \hat{y}_{j,i} \big\rVert
$$

where $M$ is the number of query points, $y_{j,i}$ is the ground-truth location of query $j$ in image $i$, and $\hat{y}_{j,i}$ is the prediction. A binary cross-entropy visibility loss is added, following CoTracker2. Ground-truth tracks come from unprojecting depth maps to 3D, reprojecting into target frames, and keeping only points whose reprojected depth matches the target depth map (so occluded points are excluded).

**A worked example.** Suppose query $y_1 = (100, 50)$ and the true match in frame 2 is $(140, 52)$. If the head predicts $(137, 56)$, the error is $\sqrt{3^2 + 4^2} = 5$ pixels, and that 5 is this pair's contribution to the loss before the weight $\lambda$.

**When it breaks.** On two-view matching (ScanNet-1500), VGGT's tracker beats the specialist matcher RoMa: AUC@5/10/20 of 33.9 / 55.2 / 73.4 against 31.8 / 53.4 / 70.9. For dynamic video the paper instead fine-tunes the backbone inside CoTracker (results below).

## Training: losses, normalization, scale

### The multi-task loss

$$
\mathcal{L} = \mathcal{L}_{\text{camera}} + \mathcal{L}_{\text{depth}} + \mathcal{L}_{\text{pmap}} + \lambda\, \mathcal{L}_{\text{track}}, \qquad \lambda = 0.05
$$

The first three terms have similar ranges, so they are unweighted. The camera loss is a Huber loss $\sum_i \lVert \hat{g}_i - g_i \rVert_\epsilon$ on the 9-vector: squared for small residuals, absolute for large ones, so bad training cameras do not dominate.

### The uncertainty-weighted depth loss

**The problem.** Some pixels are unknowable (sky, glass, regions seen once), and plain regression wastes capacity chasing them.

**The analogy.** Let the network bet. Each pixel gets a confidence; a confident wrong answer costs a lot, a hedged wrong answer costs less, but hedging everywhere is taxed.

**The math.** Following DUSt3R, plus a gradient term common in monocular depth:

$$
\mathcal{L}_{\text{depth}} = \sum_{i=1}^{N} \big\lVert \Sigma^D_i \odot (\hat{D}_i - D_i) \big\rVert + \big\lVert \Sigma^D_i \odot (\nabla \hat{D}_i - \nabla D_i) \big\rVert - \alpha \log \Sigma^D_i
$$

Here $\hat{D}_i$ is predicted depth, $D_i$ the ground truth, $\Sigma^D_i$ the predicted per-pixel confidence, $\odot$ the element-wise product, $\nabla$ the spatial image gradient, and $\alpha$ a hyperparameter whose value the paper does not report. The point-map loss is identical with $P$ and $\Sigma^P$.

**A worked example (my derivation, ignoring the gradient term).** For a single pixel with absolute error $e$, the loss is $\Sigma e - \alpha \log \Sigma$. Setting the derivative $e - \alpha / \Sigma$ to zero gives $\Sigma^\star = \alpha / e$. With an illustrative $\alpha = 1$, a pixel with error 0.1 settles at confidence 10, while a pixel with error 2 settles at 0.5. Confidence ends up inversely proportional to the error the network expects to make, which is exactly why the paper can use $\Sigma$ as a confidence map after training.

**When it breaks.** Confidence is learned on the training distribution; on fisheye or extreme rotations there is no reason to expect calibration.

### Ground-truth normalization, not prediction normalization

Scale and global frame are unobservable from images alone. VGGT removes the ambiguity by expressing everything in camera 1's frame, then dividing translations, point maps and depths by the average Euclidean distance of all 3D points to the origin. If a scene's points average 4 m from camera 1, every target is divided by 4.

The twist versus DUSt3R: the *predictions* are not normalized. The network must learn to output the canonical scale itself. The authors report that normalizing predictions was neither needed for convergence nor helpful, and added training instability.

### Scale of the run

| Setting | Value |
| --- | --- |
| Parameters | about 1.2B |
| Attention | 24 frame-wise + 24 global layers, dim 1024, 16 heads, QKNorm, LayerScale (init 0.01) |
| Optimizer | AdamW, 160K iterations, peak LR 0.0002, cosine schedule, 8K warmup |
| Batch | 2 to 24 frames per scene, 48 frames total per batch |
| Resolution | max side 518 px, aspect ratio randomized between 0.33 and 1.0 |
| Compute | 64 A100 GPUs for nine days, bfloat16, gradient checkpointing, grad clip 1.0 |
| Data | 17 datasets incl. Co3Dv2, BlendMVS, DL3DV, MegaDepth, Kubric, ScanNet, Replica, Virtual KITTI, Aria, plus a synthetic artist-asset set |

The data mix is described as broadly comparable to MASt3R's in size and diversity. That is the honest headline: the architecture is simple, and the dataset is not.

## Results

### Camera pose

![Table 1 from Wang et al. (2025): camera pose AUC@30 on RealEstate10K (unseen by every method) and CO3Dv2 with 10 random frames, with runtime on one H100.](/imgs/blogs/vggt-visual-geometry-grounded-transformer-fig4.webp)

| Method | Re10K AUC@30 | CO3Dv2 AUC@30 | Time |
| --- | --- | --- | --- |
| DUSt3R | 67.7 | 76.7 | about 7 s |
| MASt3R | 76.4 | 81.8 | about 9 s |
| VGGSfM v2 | 78.9 | 83.4 | about 10 s |
| Fast3R (concurrent) | 72.7 | 82.5 | about 0.2 s |
| VGGT feed-forward | 85.3 | 88.2 | about 0.2 s |
| VGGT + BA | 93.5 | 91.8 | about 1.8 s |

The gap is larger on RealEstate10K, which no method trained on, suggesting real generalization. And BA still helps: VGGT's maps initialize BA well enough to skip triangulation, finishing in about 2 s. On the phototourism IMC benchmark the pattern repeats: feed-forward VGGT gets AUC@10 of 71.26 (below VGGSfM v2's 76.82), and VGGT + BA reaches 84.91.

### Dense geometry

On DTU multi-view depth, VGGT scores 0.382 Overall without ground-truth cameras, versus 1.741 for DUSt3R; methods *given* true cameras range from 0.295 to 0.578. On ETH3D point maps, VGGT's 0.677 beats MASt3R's 0.826 and DUSt3R's 1.005, in 0.2 s against their 7 to 9 s.

![Figure 3 from Wang et al. (2025): VGGT versus DUSt3R on in-the-wild inputs, a single oil painting, two non-overlapping views, and 32 views with repeated texture.](/imgs/blogs/vggt-visual-geometry-grounded-transformer-fig3.webp)

In Figure 3, DUSt3R predicts a distorted plane for the painting and fails on non-overlapping views. Single-view reconstruction comes free: with one frame, global attention degenerates to frame-wise attention, and the model does well although never trained for it.

### As a backbone

| Downstream task | Baseline | With VGGT features |
| --- | --- | --- |
| TAP-Vid RGB-S, $\delta^{vis}_{avg}$ | CoTracker 78.9 | 84.0 |
| TAP-Vid Kinetics, AJ | CoTracker 49.6 | 57.2 |
| TAP-Vid DAVIS, AJ | CoTracker 61.8 | 64.7 |
| GSO novel view synthesis, PSNR | LVSM 31.71 (known cameras) | 30.41 (no input cameras, 20% data) |

For tracking, the VGGT backbone replaces CoTracker2's feature extractor and everything is fine-tuned on Kubric. For NVS, target views enter as Plücker-ray images and a DPT head regresses RGB; that row is "competitive", not a win.

### What it costs

![Runtime and peak memory of the VGGT backbone versus number of input frames on one H100 at 336 by 518: 0.04 s and 1.88 GB for one frame, 8.75 s and 40.63 GB for 200 frames in a single pass.](/imgs/blogs/vggt-visual-geometry-grounded-transformer-3.webp)

Table 9 (backbone only, FlashAttention v3) is sobering: 100 frames take 3.12 s and 21.15 GB, 200 frames 8.75 s and 40.63 GB. The camera head adds about 5% runtime; each DPT head costs about 0.03 s and 0.2 GB per frame and can run frame by frame. The [FlashAttention post](/blog/machine-learning/high-performance-computing/kernel-fusion-and-flashattention-beating-the-memory-wall) explains why exact attention fits at this scale at all.

### What is load-bearing that might not transfer

- **The data.** 17 datasets with 3D labels; a new domain (underwater, endoscopy) has none.
- **Static, pinhole world.** Rigid targets, centered principal point; non-rigid motion, fisheye and panoramas fail.
- **Frame budget.** Training used at most 24 frames per scene; "hundreds" at test time works, but no accuracy curve versus frame count is given.

## Critique

**What is strong.** The ablations isolate the two claims that matter at fixed parameter count. The evaluation includes a benchmark nobody trained on, flags MegaDepth overlap on IMC, and reports runtime and memory honestly. Code and weights are public.

**What is weak.**

- "Hundreds of views" is shown qualitatively; pose numbers use 10 frames.
- No data-scale ablation, so "transformers can do SfM" is entangled with "this data mix is very good".
- The reference-frame choice is never stress-tested, though every output depends on it.

**What would change my mind.** If a version of VGGT trained on substantially less 3D data lost its edge over VGGSfM v2 on RealEstate10K, I would read the result as "data beats solvers", not "a feed-forward transformer is the right architecture for SfM". Conversely, a curve showing pose accuracy holding flat from 10 to 200 frames would make the scaling claim much stronger than the paper currently makes it.

## Where this goes next

The open question is how to make a VGGT-style trunk handle time, stream, and serve as a 3D-aware backbone. Five directions (my proposals, not the paper's):

1. **Dynamic scenes via per-frame surfaces.** *Question:* can the same trunk predict time-varying point maps plus per-pixel scene flow? *Why now:* the appendix already defines the scene as a per-image surface $S_i$ "as the scene can change over time", so the formalism is ready; only the targets assume rigidity. *First experiment:* fine-tune on PointOdyssey and Kubric (already in the mix) with a flow head and dynamic mask, and evaluate on casual dynamic videos against [MegaSaM](/blog/paper-reading/computer-vision/megasam-accurate-fast-and-robust-structure-and-motion-from-casual-dynamic-videos), which attacks the same footage with an optimization back end.

2. **Streaming with a causal global layer.** *Question:* can global attention be made causal with a cached key/value memory so frames arrive one at a time? *Why now:* frame-wise layers are already per-frame, and heads never look across frames, so only the 24 global layers need a cache. *First experiment:* mask global attention causally, fine-tune briefly, and plot RealEstate10K AUC@30 against sequence length for causal versus bidirectional, with memory per new frame as the second axis.

3. **VGGT tokens as the 3D sense of a VLM.** *Question:* do VGGT's output image tokens improve spatial reasoning in a vision-language model compared with DINO tokens alone? *Why now:* the paper shows those tokens transfer to tracking and NVS with light fine-tuning. *First experiment:* concatenate pooled VGGT tokens with an open VLM's vision tokens, train only the projector, and evaluate multi-image spatial question answering.

4. **Self-supervision through differentiable BA.** *Question:* can unlabeled video supply the training signal that labeled 3D data currently provides? *Why now:* the authors found differentiable BA promising in small experiments but about 4 times slower per step with Theseus, and flag it as a direction for unsupervised training. *First experiment:* fine-tune on unlabeled video with a reprojection loss from a BA layer applied every few steps, and measure pose AUC on a held-out domain.

5. **Cheaper global attention.** *Question:* how much of global attention is actually needed? *Why now:* Table 9 shows memory near 41 GB at 200 frames, and the frame-wise/global split makes it natural to sparsify only the global layers. *First experiment:* restrict global layers to the camera and register tokens plus a top-k subset of patches per frame, and track ETH3D Chamfer versus memory. The [VGGT-Ω analysis](/blog/paper-reading/computer-vision/vggt) covers one CVPR 2026 descendant of this line.

## References

- Jianyuan Wang, Minghao Chen, Nikita Karaev, Andrea Vedaldi, Christian Rupprecht, David Novotny. *VGGT: Visual Geometry Grounded Transformer.* CVPR 2025. [arXiv:2503.11651](https://arxiv.org/abs/2503.11651) and [CVF open access](https://openaccess.thecvf.com/content/CVPR2025/html/Wang_VGGT_Visual_Geometry_Grounded_Transformer_CVPR_2025_paper.html).
- Code and weights: [github.com/facebookresearch/vggt](https://github.com/facebookresearch/vggt).
- On this site: [VGGT-Ω (CVPR 2026)](/blog/paper-reading/computer-vision/vggt), [MegaSaM](/blog/paper-reading/computer-vision/megasam-accurate-fast-and-robust-structure-and-motion-from-casual-dynamic-videos), [ViT, SigLIP and DINO explained](/blog/machine-learning/computer-vision/vit-siglip-dino-explained), [StreamingLLM attention sinks](/blog/paper-reading/large-language-model/streaming-llm-attention-sinks).
