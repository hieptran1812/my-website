---
title: "D4RT: Reconstructing Dynamic Scenes One Query at a Time"
publishDate: "2026-10-15"
date: "2026-10-15"
description: "How D4RT replaces dense per-frame decoders with a single point query, (u, v, t_src, t_tgt, t_cam), and gets depth, 3D tracks, point clouds and cameras out of one feedforward model, faster than anything before it."
category: "paper-reading"
subcategory: "Computer Vision"
tags: ["3d-reconstruction", "cvpr-2026", "cvpr-2026-award-candidate", "paper-reading", "d4rt", "4d-reconstruction", "3d-point-tracking", "camera-pose-estimation", "feedforward-reconstruction", "transformer"]
author: "Hiep Tran"
featured: true
readTime: 18
image: "/imgs/blogs/efficiently-reconstructing-dynamic-scenes-one-d4rt-at-a-time-fig1.webp"
excerpt: "D4RT encodes a video once, then answers independent point queries for depth, 3D correspondence and camera parameters through one tiny decoder."
paper:
  title: "Efficiently Reconstructing Dynamic Scenes One D4RT at a Time"
  authors: "Chuhan Zhang, Guillaume Le Moing, Skanda Koppula, Ignacio Rocco, et al. (Google DeepMind, UCL, Oxford)"
  venue: "CVPR 2026 (oral, award candidate)"
  url: "https://arxiv.org/abs/2512.08924"
---

> [!tldr]
> **What it claims.** One feedforward transformer, D4RT, reconstructs a dynamic scene from a video: depth, point clouds, 3D point tracks, camera intrinsics and extrinsics. It does this through a single interface: ask the decoder "where is the surface point seen at pixel $(u, v)$ in frame $t_{src}$, at time $t_{tgt}$, in the coordinates of camera $t_{cam}$?" and it returns a 3D point.
>
> **Why it matters.** Every task becomes a choice of which queries to ask. No per-task heads, no dense per-frame decoding, no test-time optimization.
>
> **Most surprising result.** The decoder is so cheap that tracking *every pixel* of a video becomes practical: D4RT produces 18 to 300 times more full-video 3D tracks than prior methods at a fixed frame rate on one A100, and estimates camera pose at 200+ FPS while being more accurate than VGGT and MegaSaM.
>
> **Where it fails.** It is trained at 256×256 on 48-frame clips, longer videos need chunking plus Sim(3) stitching, and it does not win every column: SpatialTrackerV2 is still better on ADT tracking without ground-truth intrinsics.

Most 3D reconstruction systems decode a full depth or point map for every frame, then glue frames together with camera estimates. That works for static scenes and breaks the moment something moves. D4RT (Dynamic 4D Reconstruction and Tracking) instead asks a much smaller question, many times.

## The problem: dynamic scenes break the per-frame recipe

Take a video of a swan gliding across a pond. A static method lifts every frame into 3D and merges them: the water and trees line up, but the swan is copied once per frame into a row of ghosts.

The paper groups prior work into three families, each with a gap:

| Family | Example | What it gives | What it misses |
| --- | --- | --- | --- |
| Optimization pipelines | MegaSaM | Accurate depth and cameras | Many separate models, expensive test-time optimization, no dynamic correspondence |
| Feedforward point maps | VGGT, π³, DUSt3R | Fast static reconstruction | Separate heads per task, no correspondence for moving parts |
| 3D trackers | SpatialTrackerV2, St4RTrack | Tracks moving points | Multi-stage, iterative refinement, or pairwise only |

D4RT claims one design covers all three. The teaser shows the shape: encode the video once, then probe a decoder with independent queries.

![Figure 1 from Zhang et al. (2026): D4RT encodes a video into a global scene representation, then a lightweight decoder answers independent queries to produce point clouds, 3D tracks and cameras](/imgs/blogs/efficiently-reconstructing-dynamic-scenes-one-d4rt-at-a-time-fig1.webp)

The encoder will look familiar from [VGGT](/blog/paper-reading/computer-vision/vggt-visual-geometry-grounded-transformer); the decoder is where D4RT departs.

## Technique 1: the encoder and the Global Scene Representation

**The problem.** The decoder will answer questions about any pixel at any time. Something has to have looked at the whole video first and worked out which pixels correspond, how the camera moved, and what moved on its own.

**The analogy.** A person who watches the clip once, takes thorough notes, and answers every later question from the notes alone.

**The mechanism.** The encoder $E$ is a Vision Transformer with interleaved local (frame-wise) and global self-attention layers, the same pattern VGGT uses. Frames are resized to a fixed square resolution and cut into spatio-temporal patches of size 2×16×16 (2 frames, 16×16 pixels). Because the resize throws away the aspect ratio, the original aspect ratio is embedded as one extra token.

**The math.** Given a video

$$
V \in \mathbb{R}^{T \times H \times W \times 3}, \qquad F = E(V) \in \mathbb{R}^{N \times C},
$$

where $T$ is the number of frames, $H \times W$ the frame size, $N$ the number of tokens and $C$ the channel width. $F$ is the **Global Scene Representation**. It is computed once and then frozen for all decoding.

**Worked example.** With the training setup (48 frames at 256×256) and 2×16×16 patches, the encoder sees ${48/2 = 24}$ temporal slots times ${16 \times 16 = 256}$ spatial patches, so $N = 6{,}144$ video tokens plus the aspect-ratio token. That is the size of the "notes" every query reads.

**When it fails.** Everything downstream depends on $F$. The ablations show this is the expensive, data-hungry part: initializing the encoder from VideoMAE instead of random weights cuts Sintel AbsRel (scale-only) from 0.738 to 0.302 and ATE from 0.334 to 0.091. Scaling the backbone from ViT-B (90M) to ViT-g (1B) lowers scale-only AbsRel from 0.319 to 0.191.

## Technique 2: the query, the whole idea in five numbers

**The problem.** Prior models emit outputs per frame in a fixed reference (often the first frame), so "where is this moving point later, seen from another camera" is either impossible or needs another model.

**The analogy.** A library request slip: you do not photocopy the library, you ask for exactly one item.

**The mechanism.** A query has two source parameters and two target parameters:

- $(u, v) \in [0, 1]^2$: normalized pixel coordinates of a point of interest.
- $t_{src}$: the frame in which that pixel is observed.
- $t_{tgt}$: the moment in time at which you want the point's position (the *temporal state*).
- $t_{cam}$: the frame whose camera defines the coordinate system of the answer.

The decoder returns a single 3D point:

$$
q = (u, v, t_{src}, t_{tgt}, t_{cam}), \qquad P = D(q, F) \in \mathbb{R}^{3}.
$$

The key property is that the three time indices need not coincide. Time ($t_{tgt}$) and viewpoint ($t_{cam}$) are fully disentangled.

![One D4RT query: the pixel seen in frame 1, located at time 3, expressed in camera 5 coordinates](/imgs/blogs/efficiently-reconstructing-dynamic-scenes-one-d4rt-at-a-time-1.webp)

**Worked example.** A ball rolls right across five frames. The query $q = (0.4, 0.6, 1, 3, 5)$ takes the surface point under pixel $(0.4, 0.6)$ in frame 1 (on the ball), finds it at time 3 (after the ball has moved), and reports it in camera 5's axes. Nothing asks the model to reconstruct the other frames or the background.

**How the query becomes a token.** The query token is a sum of four pieces:

1. A Fourier feature embedding of $(u, v)$.
2. Learned discrete embeddings for $t_{src}$, $t_{tgt}$ and $t_{cam}$.
3. An embedding of the local 9×9 RGB patch centred at $(u, v)$ in frame $t_{src}$ (more on this below).

The decoder $D$ is an 8-layer cross-attention transformer with 144M parameters. Each query token cross-attends into $F$ and a linear projection maps the output to $P$.

![Figure 2 from Zhang et al. (2026): the encoder produces the Global Scene Representation; a query embedding built from the local patch, Fourier features and discrete time embeddings cross-attends into it](/imgs/blogs/efficiently-reconstructing-dynamic-scenes-one-d4rt-at-a-time-fig2.webp)

**Why independence matters.** Queries never attend to each other. The paper gives three reasons this is deliberate:

- **Cheap training.** Only 2,048 random queries per clip are decoded to get a supervision signal, not a full dense output.
- **No distribution shift at inference.** You can ask any mix of queries and the model has never learned to expect particular neighbours. The authors report "major performance drops" in early experiments when they enabled self-attention between queries.
- **Trivial parallelism.** A batch of queries is embarrassingly parallel, and cost scales linearly with the number of points you ask for.

**When it fails.** No query can borrow evidence from a neighbour, so fine detail must come from $F$ or the query itself. That is why the local patch exists.

## Technique 3: one interface, four tasks

**The problem.** Depth, point clouds, tracks and cameras usually need four heads, each trained and tuned separately.

**The mechanism.** In D4RT every task is a set of queries, the Cartesian product of choices per field (the paper's Table 1):

| Task | $(u, v)$ | $t_{src}$ | $t_{tgt}$ | $t_{cam}$ | Keep |
| --- | --- | --- | --- | --- | --- |
| Point track | fixed | fixed | ${1 \ldots T}$ | $= t_{tgt}$ | full $P$ per frame |
| Point cloud | all pixels | ${1 \ldots T}$ | $= t_{src}$ | fixed | full $P$ |
| Depth map | all pixels | ${1 \ldots T}$ | $= t_{src}$ | $= t_{src}$ | only $z$ |
| Cameras | coarse $h \times w$ grid | $i$ | $i$ | $i$, then $j$ | feed to solver |

![Table 1 of the paper as pictures: tracks, depth, point clouds and cameras are four slices of the same query space](/imgs/blogs/efficiently-reconstructing-dynamic-scenes-one-d4rt-at-a-time-2.webp)

**Worked example.** A depth map for frame 10 is 65,536 queries of the form $(u, v, 10, 10, 10)$, keeping $z$. A track for one pixel in frame 10 is 48 queries $(u, v, 10, k, k)$ for $k = 1 \ldots 48$. Same decoder, same weights, different slip.

In the point-cloud row every pixel is predicted directly in one shared camera, so there is no per-frame transform through estimated cameras for camera error to leak into.

**When it fails.** The interface returns positions (plus auxiliary outputs) only. Semantics are outside it today; see the follow-up ideas.

## Technique 4: cameras without a camera head

**The problem.** Extrinsics and intrinsics are not 3D points, yet D4RT has no camera head. They must be derived from point queries.

**The analogy.** Two surveyors map the same fence posts from different spots; lining up their maps tells you where each one stood.

### Extrinsics via Umeyama

For frames $i$ and $j$, sample a grid of source points $\{(u_k, v_k)\}_k$ in frame $i$ and decode each twice:

$$
q_{i,k} = (u_k, v_k, i, i, i), \qquad q_{j,k} = (u_k, v_k, i, i, j).
$$

Both sets describe the *same* 3D points at the *same* moment (time $i$), only expressed in different camera frames. Writing $A_k = D(q_{i,k}, F)$ and $B_k = D(q_{j,k}, F)$, the relative pose is the rigid transform that best aligns them:

$$
(R^{*}, \mathbf{t}^{*}) = \arg\min_{R \in SO(3),\ \mathbf{t} \in \mathbb{R}^3} \sum_k \left\lVert B_k - (R A_k + \mathbf{t}) \right\rVert^2 .
$$

Here $R$ is a 3×3 rotation and $\mathbf{t}$ a translation. Umeyama's algorithm solves this in closed form with an SVD of a 3×3 cross-covariance matrix, so the cost is negligible.

### Intrinsics via similar triangles

Assume a pinhole camera with principal point at $(0.5, 0.5)$. A decoded point $P = (p_x, p_y, p_z)$ in its own camera frame, seen at normalized pixel $(u, v)$, satisfies the pinhole projection $u - 0.5 = f_x\, p_x / p_z$. Rearranging:

$$
f_x = \frac{p_z\,(u - 0.5)}{p_x}, \qquad f_y = \frac{p_z\,(v - 0.5)}{p_y}.
$$

Each grid point gives one estimate; D4RT takes the median over the $k$ estimates for robustness.

![Cameras fall out of the point predictions: focal length from similar triangles, relative pose from rigidly aligning the same points decoded in two reference frames](/imgs/blogs/efficiently-reconstructing-dynamic-scenes-one-d4rt-at-a-time-3.webp)

**Worked example.** A pixel at $u = 0.75$ decodes to $p_x = 0.5$ m, $p_z = 2.0$ m. Then $f_x = 2.0 \times 0.25 / 0.5 = 1.0$ in normalized units, a horizontal field of view of about 53°.

**When it fails.** Points near the principal point make $(u - 0.5)$ and $p_x$ both tiny, so their ratio is noisy; the median handles it but a grid concentrated near the centre would not. Lens distortion breaks the pinhole assumption; the paper notes a non-linear refinement step can be added on top for fisheye, but does not evaluate it.

## Technique 5: the local RGB patch

**The problem.** The encoder works on 16×16 patches at 256×256. A query that only knows $(u, v)$ and time has to recover edge-level detail from tokens that are much coarser than a pixel.

**The mechanism.** Add an embedding of the 9×9 RGB patch around $(u, v)$ in frame $t_{src}$ to the query token. The paper credits two effects: the appearance helps the query find reliable correspondences in $F$, and the low-level cues help separate an object from its surroundings, giving sharper boundaries.

**The numbers.** On a ViT-L ablation model, evaluated on Sintel:

| Local patch | AbsRel (S) | AbsRel (SS) | ATE | RPE-T | RPE-R |
| --- | --- | --- | --- | --- | --- |
| without | 0.366 | 0.306 | 0.173 | 0.031 | 0.262 |
| with | 0.302 | 0.257 | 0.091 | 0.028 | 0.245 |

ATE nearly halves. A patch-size sweep in the appendix finds 9 to 12 pixels best for both depth and pose.

**Subpixel decoding.** Because $(u, v)$ is continuous, the decoder can be queried at any resolution, independent of the encoder's 256×256. The appendix shows that querying at the original video resolution and taking the RGB patch from the *original-resolution* frame lowers the depth-boundary error $\epsilon^{acc}_{PDBE}$ from 3.323 (no patch, native resolution) to 2.193, with no change to the encoder. That is far simpler than the DPT-style skip connections most baselines rely on.

**When it fails.** The patch comes from the source frame only, so motion blur or an imminent occlusion feeds it misleading appearance.

## Technique 6: the training objective

**The mechanism.** The main loss is L1 on normalized 3D positions. Both predicted and ground-truth point sets are divided by their own mean depth, then passed through $\text{sign}(x) \cdot \log(1 + |x|)$ so far-away points do not dominate. Auxiliary linear heads on the decoder output predict 2D position, surface normal, visibility, motion (displacement) and a confidence $c$. The full objective over a batch of $N$ queries is

$$
\mathcal{L} = \frac{1}{N} \sum_{i=1}^{N} \Big( c_i \lambda_{3D} \mathcal{L}_{3D} - \lambda_{conf} \log c_i + \lambda_{2D} \mathcal{L}_{2D} + \lambda_{vis} \mathcal{L}_{vis} + \lambda_{disp} \mathcal{L}_{disp} + \lambda_{normal} \mathcal{L}_{normal} \Big),
$$

with weights $\lambda_{3D} = 1.0$, $\lambda_{2D} = \lambda_{vis} = \lambda_{disp} = 0.1$, $\lambda_{conf} = 0.2$ and $\lambda_{normal} = 0.5$. Each term applies only where ground truth exists. (The appendix equation also lists a separate $\lambda_{conf}\mathcal{L}_{conf}$ term with the same weight; I have folded it into the $-\log c$ penalty, which is what the main text describes.)

**Worked example.** The confidence term is a learned trade. If a query has 3D error $\mathcal{L}_{3D} = 0.5$, setting $c = 1$ costs ${0.5}$. Setting $c = 0.4$ costs ${0.4 \times 0.5 - 0.2 \log 0.4 \approx 0.20 + 0.18 = 0.38}$, cheaper. For an easy query with error 0.05, high confidence wins. The optimum is $c^{*} = \lambda_{conf} / (\lambda_{3D}\mathcal{L}_{3D})$ (capped by whatever range $c$ is constrained to), so the model learns to down-weight its own hard cases.

**Ablation.** Removing confidence hurts pose the most (ATE +0.126, RPE-R +0.115); removing 2D position or normals hurts depth most (AbsRel S +0.071 and +0.043). There is a small depth versus pose tension: dropping visibility slightly helps depth and hurts pose, while dropping 2D position slightly helps rotation error (RPE-R -0.028).

**Query sampling.** 30% of training queries are drawn near depth or motion discontinuities (found with a Sobel filter on depth), and $t_{tgt} = t_{cam}$ is forced with probability 0.4. Training is 500k AdamW steps on 64 TPU chips (just over 2 days) over public and internal datasets.

## Technique 7: tracking every pixel with an occupancy grid

**The problem.** Complete 4D reconstruction needs a track for every pixel of every frame, including pixels revealed later. Naively that is one $T$-query track per pixel per frame:

$$
\text{queries}_{naive} = (T \cdot H \cdot W) \times T = O(T^2 H W).
$$

For 48 frames at 256×256, that is about ${1.5 \times 10^{8}}$ queries, most of them redundant: a static wall pixel visible in all 48 frames would be tracked 48 times.

**The mechanism (Algorithm 1).** Keep a boolean occupancy grid $G \in \{0, 1\}^{T \times H \times W}$, all false. Repeatedly:

1. Sample a batch of *unvisited* source points $(u, v, t_{src})$ from $G$.
2. For each, decode the full track $\{D((u, v, t_{src}, k, k), F)\}_{k=1}^{T}$.
3. Mark every spatio-temporal pixel the track visibly passes through as visited.
4. Stop when no cell of $G$ is false.

![Dense tracking of all pixels: each new track marks every cell it visibly passes through, so the next track seeds only from pixels nobody has covered yet](/imgs/blogs/efficiently-reconstructing-dynamic-scenes-one-d4rt-at-a-time-4.webp)

**Worked example.** In the toy video in the figure (4 frames of 6×6, a 2×2 object moving across a static background), the naive scheme starts 144 tracks. The occupancy grid needs 40: 36 seeded from frame 1, plus 4 for pixels revealed behind the moving object. That is 3.6x fewer. The toy is mine; the paper's measured number on real videos is a **5 to 15x** speedup, depending on how much motion there is.

**Why prior methods cannot.** Most give no dynamic correspondence, frame-level decoders pay the naive cost, and heavy per-query decoders cost too much per track. Throughput on one A100 (Table 3):

| Method | Max tracks @ 60 FPS | @ 24 FPS | @ 10 FPS | @ 1 FPS |
| --- | --- | --- | --- | --- |
| DELTA | 0 | 5 | 408 | 5,770 |
| SpatialTrackerV2 | 29 | 84 | 219 | 2,290 |
| D4RT | **550** | **1,570** | **3,890** | **40,180** |

That is where the "18 to 300x faster" claim comes from.

**When it fails.** A track wrongly predicted visible claims cells it does not cover, and those pixels never get their own track.

## Results: what the numbers say

The qualitative comparison summarizes the dynamic-scene gap: MegaSaM duplicates the swan, π³ loses the flower, and SpatialTrackerV2 tracks from one frame only, leaving holes behind the swan and train.

![Figure 4 from Zhang et al. (2026): reconstructions from MegaSaM, pi3, SpatialTrackerV2 and D4RT on swan, flower and toy-train videos](/imgs/blogs/efficiently-reconstructing-dynamic-scenes-one-d4rt-at-a-time-fig4.webp)

**3D tracking (TAPVid-3D, camera coordinates, 3D Average Jaccard, higher is better):**

| GT intrinsics | Method | DriveTrack | ADT | PStudio |
| --- | --- | --- | --- | --- |
| no | CoTracker3 + VGGT | 0.129 | 0.132 | 0.045 |
| no | SpatialTrackerV2 | 0.064 | **0.260** | 0.097 |
| no | D4RT | **0.257** | 0.240 | **0.138** |
| yes | CoTracker3 + VGGT | 0.245 | 0.175 | 0.215 |
| yes | SpatialTrackerV2 | 0.195 | 0.303 | 0.175 |
| yes | D4RT | **0.304** | **0.307** | **0.372** |

In world coordinates on DriveTrack (without GT intrinsics), D4RT's L1 track error is 0.020 against 0.139 for SpatialTrackerV2 and 0.499 for St4RTrack.

**Reconstruction, depth and pose** (lower is better except AUC):

| Method | Sintel cloud L1 | Sintel depth AbsRel (S) | Sintel ATE | ScanNet ATE | Re10K Pose AUC@30 |
| --- | --- | --- | --- | --- | --- |
| MegaSaM | 1.531 | 0.342 | 0.074 | 0.029 | 71.0 |
| VGGT | 1.582 | 0.318 | 0.168 | 0.016 | 70.2 |
| SpatialTrackerV2 | 1.375 | 0.209 | 0.126 | 0.018 | 75.7 |
| π³ | 1.139 | 0.241 | 0.086 | 0.015 | 78.7 |
| D4RT | **0.768** | **0.171** | **0.065** | **0.014** | **83.5** |

Depth is "top-tier" rather than a clean sweep: on KITTI (AbsRel S) D4RT ties π³ at 0.055, and on Bonn π³ is slightly better (0.033 versus 0.036).

**Speed versus accuracy.** Figure 3 plots pose accuracy (one minus the error averaged over ATE, RTE and RPE on Sintel and ScanNet) against throughput, with baselines stripped of non-camera heads. D4RT reaches 200+ FPS, 9x faster than VGGT and 100x faster than MegaSaM, and is the most accurate.

![Figure 3 from Zhang et al. (2026): pose accuracy against throughput in FPS on an A100; D4RT sits in the fastest and most accurate corner](/imgs/blogs/efficiently-reconstructing-dynamic-scenes-one-d4rt-at-a-time-fig3.webp)

**What is load-bearing and might not transfer.** The 1B VideoMAE-initialized encoder (random init is dramatically worse) and internal training data others cannot reproduce. Everything runs at 256×256 and 48 frames; long videos are overlapping chunks aligned with Umeyama on the top 85% most confident points, with no loop closure. The dense-tracking speedup leans on visibility predictions that are only indirectly evaluated. If you compare against [MegaSaM](/blog/paper-reading/computer-vision/megasam-accurate-fast-and-robust-structure-and-motion-from-casual-dynamic-videos), remember MegaSaM's accuracy comes from optimization you can run on any video length.

## Critique

**What is strong.**

- The interface is genuinely minimal. One query type covers five outputs, and the decoder never needs to know which task it is serving.
- Independent queries are backed by an efficiency argument and a reported negative result (query self-attention hurt), and they enable an all-pixel tracker that was not feasible before.

**What is weak or under-tested.**

- **No breakdown of where the gains come from.** D4RT changes the decoder, the encoder size (1B), the pretraining and the data mixture at once. There is no "VGGT-sized encoder plus D4RT decoder, same data" control, so how much of the win is the query interface versus scale is open.
- **Intrinsics are not evaluated on their own.** No focal-length error table, and the fisheye refinement is mentioned, not shown.
- **Mixed tracking results get a uniform headline.** SpatialTrackerV2 wins ADT tracking without GT intrinsics (AJ 0.260 vs 0.240) and in world coordinates on ADT (APD 0.336 vs 0.319).

**What would change my mind.** If a matched-scale control (same encoder, same data, a dense DPT-style head instead of the query decoder) closed most of the accuracy gap, I would read D4RT as a speed and flexibility paper rather than an accuracy paper. Conversely, if the query decoder kept its lead at matched scale, the interface itself is the contribution and every reconstruction model should adopt it.

## Where this goes next

The orchestrating note for this series asks how queryable 4D decoding extends to semantics and to long, streaming video. These four directions are my proposals, not the paper's.

1. **Semantic queries in the same interface.** *Question:* can the query return a feature vector (for example a distilled DINO or CLIP embedding) in addition to $P$, so "what is this point and where does it go" is one call? *Why now:* D4RT already has auxiliary linear heads on the decoder output (normals, visibility, motion); a feature head is the same shape of change, and the occupancy-grid tracker would then give temporally consistent semantics for every pixel for free. *First experiment:* freeze D4RT, add a linear head regressing frozen DINOv2 features at $t_{src}$, and measure video object segmentation (DAVIS J&F) by propagating labels along the dense tracks.

2. **Streaming encoder with a persistent scene state.** *Question:* can $F$ be updated incrementally as frames arrive, rather than recomputed per 48-frame chunk? *Why now:* the decoder only reads $F$ through cross-attention, so it does not care how $F$ was built; the query's $t_{cam}$ and $t_{tgt}$ embeddings would need to become relative or continuous. *First experiment:* replace global attention with causal attention plus a fixed-size memory of tokens, and compare KITTI 1000-frame ATE against the paper's chunk-and-Umeyama baseline (Appendix B).

3. **Continuous time queries.** *Question:* can $t_{tgt}$ be a real number, giving positions between frames? *Why now:* $(u, v)$ is already continuous and the appendix shows subpixel decoding works; time is the only discrete field. *First experiment:* swap discrete time embeddings for Fourier features of $t$, train with frame dropping, and evaluate interpolated 3D tracks on PointOdyssey against held-out frames.

4. **Uncertainty-aware dense tracking.** *Question:* does using the confidence $c$ (not only visibility) to decide what counts as "visited" improve completeness on disocclusions? *Why now:* Algorithm 1 is a few lines on top of the decoder and the confidence head already exists. *First experiment:* mark cells visited only if $c$ exceeds a threshold, sweep the threshold, and plot track count against point-cloud L1 on Sintel.

## References

- Zhang, Le Moing, Koppula, Rocco, Momeni, Xie, Sun, Sukthankar, Barral, Hadsell, Ghahramani, Zisserman, Zhang, Sajjadi. *Efficiently Reconstructing Dynamic Scenes One D4RT at a Time.* CVPR 2026. [arXiv:2512.08924](https://arxiv.org/abs/2512.08924), [project page](https://d4rt-paper.github.io/), [CVPR page](https://cvpr.thecvf.com/virtual/2026/poster/38780).
- Wang et al. *VGGT: Visual Geometry Grounded Transformer.* CVPR 2025. Our walkthrough: [VGGT](/blog/paper-reading/computer-vision/vggt-visual-geometry-grounded-transformer).
- Li et al. *MegaSaM: Accurate, Fast and Robust Structure and Motion from Casual Dynamic Videos.* CVPR 2025. Our walkthrough: [MegaSaM](/blog/paper-reading/computer-vision/megasam-accurate-fast-and-robust-structure-and-motion-from-casual-dynamic-videos).
- Umeyama. *Least-squares estimation of transformation parameters between two point patterns.* IEEE TPAMI, 1991.
- Koppula et al. *TAPVid-3D: A Benchmark for Tracking Any Point in 3D.* NeurIPS 2024.
