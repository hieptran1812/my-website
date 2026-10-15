---
title: "CUT3R: Continuous 3D Perception with a Persistent State, or 3D Reconstruction as a Recurrent Network"
publishDate: "2026-10-15"
date: "2026-10-15"
description: "A close read of CUT3R (CVPR 2025 oral): a recurrent transformer whose fixed-size state is updated and read by every incoming image, producing metric pointmaps and cameras online, and able to predict geometry for views nobody captured."
excerpt: "CUT3R keeps 768 state tokens that every new frame both writes into and reads from. Out come metric pointmaps and camera poses for each frame as it arrives, at about 17 FPS, plus the ability to query the state with a virtual camera and get back geometry for unseen regions. Here is how each piece works, what the numbers say, and where the idea should go next."
category: "paper-reading"
subcategory: "Computer Vision"
tags: ["3d-reconstruction", "cvpr-2025", "cvpr-2025-oral", "paper-reading", "cut3r", "pointmaps", "online-reconstruction", "recurrent-models", "dust3r", "camera-pose-estimation", "dynamic-scenes"]
author: "Hiep Tran"
featured: true
image: "/imgs/blogs/continuous-3d-perception-model-with-persistent-state-fig1.webp"
readTime: 18
paper:
  title: "Continuous 3D Perception Model with Persistent State"
  authors: "Qianqian Wang, Yifei Zhang, Aleksander Holynski, Alexei A. Efros, Angjoo Kanazawa"
  venue: "CVPR 2025 (oral); arXiv:2501.12387"
  url: "https://arxiv.org/abs/2501.12387"
---

> [!tldr]
> - **What it claims.** A single recurrent model, CUT3R (Continuous Updating Transformer for 3D Reconstruction), turns a stream of uncalibrated images into metric-scale pointmaps and camera poses, one frame at a time, by keeping a persistent state of 768 tokens that each frame both updates and reads.
> - **Why it matters.** It is online and fast: 16.58 FPS on KITTI against 0.35 FPS for MonST3R with global alignment, and it handles static scenes, dynamic scenes and sparse photo collections with the same weights.
> - **The surprising result.** The state is not just a cache of what was seen. Query it with a virtual camera (a "raymap") and it returns plausible metric geometry for regions no input image covered.
> - **Where it is soft.** It drifts on long sequences because nothing corrects old mistakes, the unseen-region predictions are blurry because the model is deterministic, and it still trails optimization-based methods on camera pose.

Structure-from-Motion and SLAM start every scene from a blank slate, which is why they break on two photos, on a camera that only rotates, or on a person walking through the frame. The DUSt3R family fixed part of this by learning a prior from data, but it reasons about images in pairs and needs an offline "global alignment" step to stitch the pairs together.

CUT3R asks a different question: what if a 3D model worked the way a person walking into a new restaurant does? You get a rough layout from one glance, refine it with every step, and can guess what is behind the counter before you see it. The paper builds exactly that loop and shows it competes with offline methods while running online.

![Figure 1 from Wang et al. (2025): CUT3R reconstructs static scenes, dynamic scenes and sparse photo collections from a stream of images, estimating cameras and dense geometry for each new frame](/imgs/blogs/continuous-3d-perception-model-with-persistent-state-fig1.webp)

## The problem: pairs do not make a stream

To see why a state helps, look at what the pairwise approach costs. DUSt3R predicts two pointmaps in one shared frame for an image pair. For $N$ images you run it on many pairs and then solve an optimization that makes all the pairwise predictions agree.

That has three consequences the paper calls out:

- **Offline.** The alignment needs all images up front. A new frame means re-solving.
- **Slow.** On KITTI video depth the paper measures DUSt3R-GA at 0.76 FPS and MASt3R-GA at 0.31 FPS.
- **Static by assumption.** Global alignment enforces multi-view consistency, which helps static geometry but can damage moving objects.

Recent alternatives each fix one of these, as the table shows. [VGGT](/blog/paper-reading/computer-vision/vggt-visual-geometry-grounded-transformer) takes a third route: all frames jointly in one pass, fast but needing every frame at once.

| Method | Input | Online? | Dynamic scenes? | Extra alignment? |
| --- | --- | --- | --- | --- |
| DUSt3R / MASt3R | image pairs | no | not trained for it | global alignment |
| MonST3R | image pairs | no | yes | global alignment plus optical flow |
| Spann3R | stream | yes | no (static training) | no |
| CUT3R | stream | yes | yes | no |

## What goes in and what comes out

The input is a sequence of RGB images $I_1, I_2, \dots, I_t$ with no intrinsics and no poses. They may be consecutive video frames or an unordered photo collection, possibly with little overlap.

For every frame $t$, CUT3R outputs four things:

- $\hat{X}_t^{\text{self}}$: a pointmap in the frame's own camera coordinates, one 3D point per pixel, plus a confidence map $C_t^{\text{self}}$.
- $\hat{X}_t^{\text{world}}$: a pointmap in the world frame, defined as the coordinate frame of the first image, plus $C_t^{\text{world}}$.
- $\hat{P}_t$: the 6-DoF rigid transform from frame $t$ to the world.
- Everything is metric, in meters.

Accumulating the world pointmaps gives a dense reconstruction that grows as the camera moves.

The outputs are redundant on purpose. You could compute the world pointmap from the self pointmap and the pose. The paper says the redundancy simplifies training, and more importantly lets each output be supervised directly on datasets that only label part of it, such as pose-only RealEstate10K or single-view depth sets.

## The persistent state

**The problem it solves.** A stream model needs somewhere to keep what it learned from frames ${1}$ through $t-1$, in a form the next frame can use, without that memory growing with $t$.

**Analogy.** Think of a whiteboard of fixed size in a meeting room. Every person who walks in reads what is on the board to understand the context, and also rewrites part of it with what they know. The board never gets bigger, so the writing has to get more compact as more people arrive.

**Mechanism.** The state $s$ is a set of 768 tokens, each 768-dimensional. Before any image arrives, it is initialized as $s_0$, a set of learnable tokens shared across all scenes. So even the empty state carries a learned prior about what scenes tend to look like.

![Figure 3 from Wang et al. (2025): each image is encoded, interacts with the state through simultaneous state update and state readout, and produces pointmaps and cameras; on the right, a raymap queries the state for unseen structure](/imgs/blogs/continuous-3d-perception-model-with-persistent-state-fig2.webp)

The figure above is the whole method: image and state tokens meet in a pair of decoders, and heads decode geometry. The right side is a virtual camera that only reads.

## One step of the recurrence

Each new frame first goes through a ViT encoder that cuts it into 16×16 pixel patches:

$$
F_t = \text{Encoder}_i(I_t)
$$

Here $I_t$ is the image at step $t$ and $F_t$ is its set of patch tokens. The encoder is a ViT-Large initialized from DUSt3R's pretrained encoder weights. For a 512×384 frame (my example, not the paper's), that is a 32×24 grid, so 768 image tokens.

Then the core exchange, the paper's Equation 2:

$$
[z_t', F_t'],\; s_t = \text{Decoders}\big([z, F_t],\; s_{t-1}\big)
$$

Every symbol:

- $s_{t-1}$ and $s_t$: the state before and after seeing frame $t$. Same shape, 768 tokens of dimension 768.
- $z$: a learnable "pose token" prepended to the image tokens. Its output $z_t'$ carries image-level information such as ego-motion.
- $F_t'$: the image tokens after they have read from the state.
- $\text{Decoders}$: two interconnected ViT-Base decoders, one for the image side and one for the state side. At every decoder block, each side cross-attends to the other.

![One CUT3R step drawn with tensor shapes: image tokens and the 768 by 768 state pass through two cross-attending decoders, and the state comes out the same size it went in](/imgs/blogs/continuous-3d-perception-model-with-persistent-state-1.webp)

The key design choice is that **update and readout happen in the same pass**. The state side writes new information from the frame (state update), and the image side reads past context (state readout). There is no separate "first write, then read" step, which keeps one forward pass per frame.

### The heads

After the exchange, three heads decode explicit 3D outputs:

$$
\hat{X}_t^{\text{self}}, C_t^{\text{self}} = \text{Head}_{\text{self}}(F_t'), \quad
\hat{X}_t^{\text{world}}, C_t^{\text{world}} = \text{Head}_{\text{world}}(F_t', z_t'), \quad
\hat{P}_t = \text{Head}_{\text{pose}}(z_t')
$$

The self and world heads are DPT dense-prediction heads in the final training stages (linear heads in the early low-resolution stages). The pose head is a 2-layer MLP with hidden size 768.

Per the supplement, the world head first runs two self-attention blocks whose LayerNorms are modulated by the pose token $z_t'$ (an idea from LRM). The pose token effectively tells the head "rotate and shift by this much", an implicit rigid transform.

**Worked micro-example.** Suppose frame 1 sees a table edge 2 m in front of the camera, so its world point equals its self point, $(0, 0, 2)$. At frame 2 the camera has moved 0.5 m to the right with no rotation. The same edge now has self coordinates $(-0.5, 0, 2)$, while $\hat{P}_2$ should be a translation of $(0.5, 0, 0)$ and the world head should still output $(0, 0, 2)$. The state is what lets the network know where frame 1 put the world origin.

**When it fails.** The state has a fixed budget, so information from early frames must be compressed or overwritten as more arrive. Nothing ever revisits frame 1's prediction, so an early pose error is baked in. The paper names this drift as its first limitation.

## The confidence-weighted loss

**The problem.** Some pixels cannot be predicted well: sky, reflections, moving objects, regions at the edge of view. A plain regression loss lets those pixels dominate training.

**Analogy.** A student allowed to say "I am not sure" on each exam question, but who pays a small fee every time they say it. They will only claim uncertainty where they are truly likely to be wrong.

**The math.** CUT3R follows MASt3R and uses a confidence-aware regression loss, the paper's Equation 7:

$$
\mathcal{L}_{\text{conf}} = \sum_{(\hat{x}, c) \in (\hat{\mathcal{X}}, \mathcal{C})} \left( c \cdot \left\lVert \frac{\hat{x}}{\hat{s}} - \frac{x}{s} \right\rVert_2 - \alpha \log c \right)
$$

- $\hat{x}$: a predicted 3D point (one pixel of any self or world pointmap); $x$: its ground truth.
- $c$: the predicted confidence for that pixel.
- $\hat{s}$, $s$: scale normalization factors for the prediction and the ground truth.
- $\alpha$: a weight on the confidence regularizer.

When the ground truth is metric, the paper sets $\hat{s} := s$. That one line is what forces the network to learn absolute scale, because it can no longer cancel a scale error by rescaling its own prediction.

Why does this produce meaningful confidence? Write the per-pixel error as $e$. The per-pixel loss is $c e - \alpha \log c$, and its derivative with respect to $c$ is $e - \alpha / c$. Setting that to zero:

$$
c^\star = \frac{\alpha}{e}
$$

![Per-pixel loss as a function of the confidence the network reports, for three error levels: the minimum sits at c equals alpha over e, so a large error pulls the optimal confidence down](/imgs/blogs/continuous-3d-perception-model-with-persistent-state-3.webp)

**Worked example.** With $\alpha = 1$: a pixel with error 0.25 m wants confidence 4, a pixel with error 0.5 m wants 2, a pixel with error 1 m wants 1. Confidence ends up roughly inversely proportional to error, which is exactly what you want when you later filter a point cloud by confidence. (In the DUSt3R lineage, confidence is parameterized to stay above 1, so very large errors simply bottom out.)

**When it fails.** The network can report low confidence on a whole class of hard pixels, such as thin structures, and stop trying to get them right.

### Pose and color losses

The pose $\hat{P}_t$ is parameterized as a quaternion $\hat{q}_t$ and a translation $\hat{\tau}_t$, with an L2 loss (Equation 8):

$$
\mathcal{L}_{\text{pose}} = \sum_{t=1}^{N} \left( \lVert \hat{q}_t - q_t \rVert_2 + \left\lVert \frac{\hat{\tau}_t}{\hat{s}} - \frac{\tau_t}{s} \right\rVert_2 \right)
$$

Translation uses the same scale normalization as the points. For raymap queries, there is an extra MSE color loss $\mathcal{L}_{\text{rgb}} = \lVert \hat{I}_r - I_r \rVert_2^2$ between predicted and true pixel colors.

## Querying the state with a raymap

**The problem.** If the state really understands the scene, it should be able to answer questions about parts nobody photographed. How do you ask?

**Analogy.** It is like Masked Autoencoders one level up, which is the paper's own comparison. MAE fills in a missing patch from the rest of the image. CUT3R fills in a missing image from the rest of the scene.

**Mechanism.** A virtual camera is written as a raymap $R$: a 6-channel image whose pixel holds the origin and the direction of the ray through that pixel. A separate lightweight encoder, only 2 transformer blocks, turns it into tokens:

$$
F_r = \text{Encoder}_r(R)
$$

Then $F_r$ goes through the same decoders as an image would, with shared weights, to read from the state into $F_r'$. The same heads decode a pointmap, and a new head decodes color: $\hat{I}_r = \text{Head}_{\text{color}}(F_r')$.

The crucial difference: **the state is not updated** by a raymap query. A raymap carries no new scene content, only a question, so it is allowed to read and not write.

![A raymap query from above: the observed camera saw one wall, the virtual camera points at a region nobody captured, and the state returns predicted points there without being updated](/imgs/blogs/continuous-3d-perception-model-with-persistent-state-2.webp)

**Worked micro-example (mine).** Take a pinhole camera at the world origin with focal length 500 px and principal point $(256, 192)$. The pixel at $(356, 192)$ is 100 px right of center, so its ray direction is $(100/500, 0, 1) = (0.2, 0, 1)$, about $(0.196, 0, 0.981)$ after normalizing. Its six raymap channels are $(0, 0, 0, 0.196, 0, 0.981)$. Move the camera and every pixel's origin changes; rotate it and every direction changes. The raymap is the camera, written in pixels.

**How it is trained.** During training, each frame after the first is replaced with its raymap with 20% probability. This mode is enabled only for datasets with metric 3D labels, because a raymap is specified in meters and the state stores a metric scene. A raymap from a dataset with unknown scale would describe a world of the wrong size.

![Figure 6 from Wang et al. (2025): one input image, a held-out ground-truth view used only for its camera, the predicted depth for that view, and the pointmap before and after the query](/imgs/blogs/continuous-3d-perception-model-with-persistent-state-fig5.webp)

The paper tests this on MapFree and ARKitScenes validation scenes not seen in training. One input image goes in; the ground-truth image of a second view is used only for its camera parameters. The predicted geometry follows large viewpoint changes and adds structure the input never showed: bushes, ground, an oven, a stool.

**When it fails.** The predictions lose high-frequency detail. The model is a deterministic regressor, so for an ambiguous region it predicts something like the average of the possibilities, which looks blurry. The paper says this gets worse the further the query camera is from the input views.

## Training: 32 datasets and a curriculum

A recurrent 3D model is only as good as the variety of scenes its state has seen. CUT3R trains on 32 datasets, mixing synthetic and real, static and dynamic, indoor, outdoor and object-centric. Examples include CO3Dv2, ARKitScenes, ScanNet++, TartanAir, Waymo, MegaDepth, MapFree, DL3DV and DynamicStereo.

Partial labels are handled per dataset:

| Data type | Example | What is supervised |
| --- | --- | --- |
| Full metric 3D | ARKitScenes, ScanNet++ | pointmaps, pose, raymap queries |
| Camera only | RealEstate10K, CoP3D, MVImgNet | pose only |
| Single views | Synscapes, EDEN, IRS | per-view geometry, state reset to $s_0$ after each view |
| No dense depth originally | MapFree, DL3DV | depth from multi-view stereo run by the authors |

The curriculum has four stages:

1. 4-view sequences, mainly static data, 224×224 resolution.
2. Add dynamic scenes and partially annotated data, still 224×224.
3. Higher resolution, varied aspect ratios, longest side 512.
4. Freeze the encoder and train only the decoders and heads on sequences of 4 to 64 views, for long-context reasoning.

Training uses AdamW at learning rate ${10}^{-4}$ on eight 80 GB A100 GPUs. Color jitter is applied per sequence rather than per frame, so the model cannot use color shifts to tell frames apart.

## Results

The paper evaluates on depth, camera pose and reconstruction, against DUSt3R, MASt3R, MonST3R and Spann3R. None of the evaluation datasets are used in training.

### Video depth

![Table 2 from Wang et al. (2025): video depth on Sintel, Bonn and KITTI under per-sequence scale and metric scale, with FPS on KITTI](/imgs/blogs/continuous-3d-perception-model-with-persistent-state-fig3.webp)

| Method (per-sequence scale) | Online | Sintel Abs Rel ↓ | Bonn Abs Rel ↓ | KITTI Abs Rel ↓ | FPS |
| --- | --- | --- | --- | --- | --- |
| DUSt3R-GA | no | 0.656 | 0.155 | 0.144 | 0.76 |
| MonST3R-GA | no | **0.378** | **0.067** | 0.168 | 0.35 |
| Spann3R | yes | 0.622 | 0.144 | 0.198 | 13.55 |
| CUT3R | yes | 0.421 | 0.078 | **0.118** | **16.58** |

CUT3R is best on KITTI, close to MonST3R on Bonn and Sintel, and about 47 times faster than MonST3R (16.58 against 0.35 FPS; the paper rounds this to "nearly 50×"). Without any alignment, in metric scale, CUT3R reaches 0.103 Abs Rel on Bonn against 0.272 for MASt3R-GA, and 0.122 on KITTI against 0.467. On Sintel metric depth both are poor (1.029 vs 1.022).

### Camera pose

Pose is measured by Absolute Trajectory Error (ATE) after Sim(3) alignment. Among online methods CUT3R is best overall:

| Online method | Sintel ATE ↓ | TUM-dynamics ATE ↓ | ScanNet ATE ↓ |
| --- | --- | --- | --- |
| DUSt3R (align to frame 1) | 0.290 | 0.140 | 0.246 |
| Spann3R | 0.329 | 0.056 | **0.096** |
| CUT3R | **0.213** | **0.046** | 0.099 |

The honest caveat is the paper's own: a gap remains to optimization methods. MonST3R-GA reaches 0.111 ATE on Sintel, about half of CUT3R's 0.213.

### 3D reconstruction from sparse views

On 7-Scenes (3 to 5 frames per scene) and NRGBD (2 to 4 frames), CUT3R's mean accuracy is 0.126 on 7-Scenes against 0.146 for DUSt3R-GA and 0.298 for Spann3R, at 17.00 FPS against DUSt3R-GA's 0.68. DUSt3R-GA keeps the better normal consistency (0.736 vs 0.727).

![Figure 4 from Wang et al. (2025): reconstructions of in-the-wild internet videos with moving people, comparing Spann3R, MonST3R and CUT3R](/imgs/blogs/continuous-3d-perception-model-with-persistent-state-fig4.webp)

The paper's argument: Spann3R, trained only on static data, handles moving people poorly, while MonST3R, fine-tuned on dynamic data, may lose some static quality.

### Does the state actually learn?

The cleanest experiment in the paper is "revisiting". Run the sequence online to get the final state $s_N$, freeze it, and then process the same images again with that state. The second pass gets the benefit of the whole scene for every frame, including the first.

![Table 5 of the paper as bars on 7-Scenes: revisiting with the final state improves both accuracy and completion over the online pass, which already beats DUSt3R-GA](/imgs/blogs/continuous-3d-perception-model-with-persistent-state-4.webp)

Completion on 7-Scenes improves from 0.154 to 0.107 and accuracy from 0.126 to 0.113. On NRGBD the gains are smaller (accuracy 0.099 to 0.094, completion unchanged at 0.076). This is evidence that the state accumulates useful scene information rather than only remembering the last frame.

**What is load-bearing in their setup and might not transfer.** The evaluations are short: sparse 2 to 5 views for reconstruction, and standard video clips for depth and pose. Training tops out at 64 views. The paper does not report what happens at 1,000 frames, which is exactly the regime the drift limitation predicts trouble for.

## Critique

**What is strong.**

- One model, one set of weights, covers video and photo collections, static and dynamic scenes, depth, pose and reconstruction. That generality is the real contribution.
- The revisit experiment is a clean test of a specific claim about the state, not just a benchmark number.

**What is weak.**

- No length study. We never see accuracy as a function of sequence length, so we cannot tell how fast the 768-token state saturates.
- No ablation on state size, on the two-decoder design, or on the redundant outputs. The paper asserts redundancy simplifies training but does not measure it.
- The unseen-region results are qualitative only. There is no metric for how accurate the predicted geometry of unobserved regions is.
- Pose still trails optimization by roughly 2× ATE on Sintel.

**What would change my mind.** If a length sweep showed CUT3R's pose error at 500 frames staying close to its error at 50 frames, I would stop calling drift its main weakness. Conversely, if a state-size ablation showed 128 tokens performing as well as 768, I would doubt that the state is doing much more than tracking the last few frames.

For a contrasting design, [D4RT](/blog/paper-reading/computer-vision/efficiently-reconstructing-dynamic-scenes-one-d4rt-at-a-time) encodes a whole video once and answers independent point queries through one small decoder, instead of carrying a recurrent state. And [Selfi](/blog/paper-reading/computer-vision/selfi-self-improving-reconstruction-engine-via-3d-geometric-feature-alignment) takes a frozen VGGT and trains a small feature adapter on its own depth and cameras, a different route to making a feed-forward reconstructor more consistent.

## Where this goes next

The paper names the problem plainly: online models drift, and the state has a fixed size. For anyone who wants to reconstruct a city block or a full building in a single stream, those are the two walls. These follow-ups are my proposals, not the paper's.

1. **A capacity law for the state.**
   - *Question:* how does reconstruction quality degrade with sequence length, as a function of state size?
   - *Why now:* CUT3R gives a clean knob (number of state tokens) and an architecture where everything else stays fixed.
   - *First experiment:* run the released model on long KITTI or Waymo drives, plot ATE against frame count, and then retrain with 256, 768 and 2048 tokens on the same curriculum to find where the curves bend.

2. **Hierarchical state for city scale.**
   - *Question:* can a set of local states, one per submap, plus a small global state replace one state for everything?
   - *Why now:* the world frame is already defined by the first image, and the state can be reset to $s_0$ (the paper does this for single views), so starting a new submap is a well-defined operation.
   - *First experiment:* split long drives into fixed-length chunks, reset the state at each chunk, chain chunks by the predicted pose of the overlap frame, and compare drift with a single unbroken state.

3. **Gated writes to resist forgetting.**
   - *Question:* does a learned write gate on the state (in the spirit of gated linear attention or state-space models) keep early-frame information longer?
   - *Why now:* the raymap query gives a direct probe of memory. After $N$ more frames, query the state with frame 1's camera and measure how well it reconstructs frame 1.
   - *First experiment:* plot that "recall error" against $N$ for the baseline and for a gated update, trained identically.

4. **Revisit as cheap loop closure.**
   - *Question:* can periodic revisits replace global alignment for drift correction?
   - *Why now:* Table 5 already shows revisiting helps, and it costs only a second forward pass, no optimization.
   - *First experiment:* every $K$ frames, re-run the last window of frames through the current state and replace their poses; sweep $K$ and measure the accuracy against compute trade-off on long sequences.

## References

- Qianqian Wang, Yifei Zhang, Aleksander Holynski, Alexei A. Efros, Angjoo Kanazawa. *Continuous 3D Perception Model with Persistent State.* CVPR 2025. [arXiv:2501.12387](https://arxiv.org/abs/2501.12387). [CVF open access](https://openaccess.thecvf.com/content/CVPR2025/html/Wang_Continuous_3D_Perception_Model_with_Persistent_State_CVPR_2025_paper.html).
- Project page: [cut3r.github.io](https://cut3r.github.io/).
- Related on this site: [VGGT](/blog/paper-reading/computer-vision/vggt-visual-geometry-grounded-transformer), [D4RT for dynamic scenes](/blog/paper-reading/computer-vision/efficiently-reconstructing-dynamic-scenes-one-d4rt-at-a-time), [Selfi](/blog/paper-reading/computer-vision/selfi-self-improving-reconstruction-engine-via-3d-geometric-feature-alignment).
