---
title: "FUSER: Feed-Forward Multiview 3D Registration Transformer and SE(3)^N Diffusion Refinement"
publishDate: "2026-10-15"
date: "2026-10-15"
description: "How FUSER registers dozens of point-cloud scans in one transformer pass with zero pairwise matching, and how FUSER-DF polishes the result with diffusion over all poses at once."
category: "paper-reading"
subcategory: "Computer Vision"
tags: ["3d-reconstruction", "cvpr-2026", "cvpr-2026-award-candidate", "paper-reading", "point-cloud-registration", "fuser", "se3-diffusion", "transformer", "multiview-registration"]
author: "Hiep Tran"
featured: true
readTime: 18
image: "/imgs/blogs/fuser-feed-forward-multiview-3d-registration-transformer-and-se-3-n-diffusion-refinement-fig1.webp"
excerpt: "FUSER replaces the pairwise-then-global registration pipeline with a single feed-forward transformer over all scans, then refines its poses with diffusion on SE(3)^N. Minutes become fractions of a second."
paper:
  title: "FUSER: Feed-Forward MUltiview 3D Registration Transformer and SE(3)^N Diffusion Refinement"
  authors: "Haobo Jiang, Jin Xie, Jian Yang, Liang Yu, Jianmin Zheng"
  venue: "CVPR 2026 (oral, award candidate)"
  url: "https://arxiv.org/abs/2512.09373"
---

> [!tldr]
> **What it claims.** Multiview point-cloud registration does not need a pose graph. FUSER feeds every scan into one transformer and regresses every scan's global pose directly, with zero pairwise matches.
> **Why it matters.** The classic pipeline runs a pairwise matcher on every scan pair and then synchronizes. On 60-scan 3DMatch sequences that takes 155 to 763 seconds. FUSER takes 0.31 seconds.
> **Most surprising result.** Initializing the 3D transformer with attention weights from a 2D image model ($\pi^3$) nearly triples the fraction of scans within 3° of ground-truth rotation on ScanNet (12.9% to 36.6%), despite the modality gap.
> **Where it fails.** The diffusion refinement (FUSER-DF) costs about 10x the runtime and does not improve mean rotation error. Evaluation is indoor RGB-D only, at most 200 scans.

Registering many scans has always been two stages: align pairs, collect the relative poses into a graph, then solve for absolute poses that agree with it. Errors enter at every stage, and nothing downstream sees the raw geometry that caused them.

FUSER asks a blunt question, and the answer builds on [VGGT](/blog/paper-reading/computer-vision/vggt-visual-geometry-grounded-transformer) and on synchronization work like [QuadSync](/blog/paper-reading/computer-vision/quadsync-quadrifocal-tensor-synchronization-via-tucker-decomposition): what if one network looks at all the scans at once and just outputs the poses? It then refines those poses with diffusion over the joint space of all poses, $SE(3)^N$.

## The problem, stated precisely

We have $N$ point clouds $\mathcal{S} = \{\mathbf{S}_i\}$, where $\mathbf{S}_i \in \mathbb{R}^{M_i \times 3}$ holds the $M_i$ points of scan $i$. Each scan overlaps some others partially. We want a rigid pose for each one:

$$
\mathbf{T}_i = (\mathbf{R}_i, \mathbf{t}_i) \in SE(3), \qquad \mathbf{R}_i \in SO(3),\ \mathbf{t}_i \in \mathbb{R}^3,
$$

so that applying $\mathbf{T}_i$ to every scan puts them all in one common frame. $SO(3)$ is the group of 3D rotations and $SE(3)$ adds translation. Note that the answer is only defined up to one global rigid motion: rotate and shift the whole assembled room and nothing about the alignment changes. Keep that in mind; it drives the loss design later.

### How everyone did it before

The standard pipeline builds a graph $\mathcal{G} = (\mathcal{S}, \mathcal{E})$. Each edge $(i,j)$ stores a relative pose $\mathbf{T}_{i \leftarrow j} = \mathbf{T}_i^{-1}\mathbf{T}_j$ estimated by a pairwise registration method (FCGF, Predator, YOHO, GeoTransformer, and so on). A synchronization step then finds absolute poses that best explain all edges, typically with robust averaging or iteratively reweighted least squares.

The paper names four problems: pairs are solved without context, a bad pair contaminates synchronization, pair count grows quadratically, and hand-tuned choices hold it all together.

![Pairwise-then-global builds and trusts a quadratic pose graph; FUSER predicts every pose jointly.](/imgs/blogs/fuser-feed-forward-multiview-3d-registration-transformer-and-se-3-n-diffusion-refinement-1.webp)

**Worked example: how fast the graph grows.** A full graph over $N$ scans has $N(N-1)/2$ edges. For the paper's ScanNet setup of 30 scans that is 435 pairs per sequence. Over the 32 test sequences that gives ${32} \times 435 = 13{,}920$, which is exactly the "#Pair" column for the full-graph baselines in Table 1 (the arithmetic check is mine). At 200 scans, as on ArkitScenes, a single sequence has 19,900 pairs. FUSER's "#Pair" column is 0 everywhere.

## The architecture at a glance

FUSER has three parts: an **absolute geometric encoder** that turns each scan into a few superpoint tokens, a **geometric alternating attention** trunk that mixes those tokens within and across scans, and a **global pose head** that reads out one pose per scan.

![Figure 2 from Jiang et al. (2026): the FUSER architecture, from unordered scans through absolute geometric encoding and alternating attention to per-scan pose regression.](/imgs/blogs/fuser-feed-forward-multiview-3d-registration-transformer-and-se-3-n-diffusion-refinement-fig1.webp)

The analogy is assembling a torn map. The old way compares every pair of fragments on its own, then argues about a consensus. FUSER puts all fragments on one table and looks at everything before committing.

| Stage | Input | Output | Key choice |
| --- | --- | --- | --- |
| Absolute geometric encoder | $\mathbf{S}_i$, $M_i \times 3$ | $M_i'$ superpoints with $d$-dim features | Sparse 3D CNN on absolute coordinates, 5 stride-2 levels |
| Alternating attention | all superpoint tokens | refined tokens | 32 layers: 16 intra-scan, 16 cross-scan, init from $\pi^3$ |
| Pose head | tokens of scan $i$ | $\hat{\mathbf{T}}_i$ | self-attention, average pool, two MLPs, SVD projection |

## Technique 1: absolute geometric encoding

**The problem.** Pairwise descriptors such as KPConv-based ones are deliberately translation-invariant: they normalize coordinates so the same corner looks the same wherever it sits. That is perfect for matching. It is useless for regressing an absolute translation, because the network has thrown away where the scan sits in its own camera frame.

**The mechanism.** FUSER uses a sparse 3D CNN built with MinkowskiEngine that consumes the raw coordinates. Each scan is voxelized hierarchically. Five sparse convolution layers with kernel 3 and stride 2 shrink the scan to a small set of superpoints $\mathbf{S}'_i \in \mathbb{R}^{M_i' \times 3}$ with features $\mathbf{F}_i \in \mathbb{R}^{M_i' \times d}$, where $M_i' \ll M_i$. The appendix lists the channel widths growing from 32 to 1024 along the hierarchy.

The depth is a budget choice: all-scan attention is quadratic in total tokens, so resolution is pushed as low as possible. Even very coarse superpoints keep enough geometry to regress pose.

**Worked example.** Each stride-2 level halves the grid along each axis, so five levels coarsen it by ${2}^5 = 32$ per axis. If you start at a 2.5 cm voxel (my illustrative choice; the paper does not state the base voxel size), superpoints sit roughly 0.8 m apart. A 4 m by 4 m room slice then holds on the order of a few dozen occupied surface cells, not tens of thousands of points.

**When it fails.** The encoder sees each scan in its own sensor frame, which suits depth cameras. An 80 m LiDAR sweep at the same voxel budget loses detail a room-scale model never had to give up (see the follow-up section).

![Each scan becomes a handful of superpoint tokens; layers alternate block-diagonal and full attention.](/imgs/blogs/fuser-feed-forward-multiview-3d-registration-transformer-and-se-3-n-diffusion-refinement-2.webp)

## Technique 2: geometric alternating attention

**The problem.** Global attention over every token of every scan is expensive and mixes two different jobs: understanding one scan's local shape and relating scans to each other.

**The mechanism.** FUSER borrows VGGT's alternating design. The trunk has $L = 32$ layers that alternate: 16 intra-scan blocks where each scan's tokens attend only to tokens from the same scan, and 16 cross-scan blocks where every token attends to every token from every scan. The figure above draws this as two attention masks. Intra-scan is block-diagonal; cross-scan is dense.

Two changes from VGGT matter.

1. **No reference token.** VGGT marks the first frame with a learnable token so it can define the world frame. FUSER removes it because it made training unstable under changing scan order. The trunk is then permutation-equivariant:

$$
\mathrm{AA}\big(P_\pi(\mathcal{S}'),\, P_\pi(\mathcal{F})\big) = P_\pi\big(\mathrm{AA}(\mathcal{S}', \mathcal{F})\big),
$$

where $P_\pi$ reorders the scans by permutation $\pi$ and $\mathrm{AA}$ is the alternating attention stack. Shuffle the scans in, get the same features shuffled out.

2. **Absolute sinusoidal position encoding instead of RoPE.** VGGT uses 2D rotary embeddings on patch coordinates. FUSER puts sinusoidal encodings of superpoint coordinates into every attention layer. The authors report that a relative 3D RoPE did worse, and offer a plausible reason: in cross-scan attention, two tokens live in different coordinate frames, so their "relative position" is meaningless and actively misleading.

**Worked example on cost.** Suppose $N$ scans each produce $m$ tokens. A cross-scan layer computes $(Nm)^2$ attention scores; an intra-scan layer computes only $N m^2$. With $N = 200$ and an illustrative $m = 64$, that is about 164 million scores versus 0.8 million. With FlashAttention, the paper reports 2.83 GB of GPU memory on its longest 200-scan sequences.

### The 2D-to-3D attention prior

The most surprising piece of the paper is how the trunk is initialized. Instead of training the 32 alternating layers from scratch, FUSER copies the weights of $\pi^3$, a VGGT-style model trained on large-scale image reconstruction. No architectural change is needed because the trunks are structurally the same; only the input encoder and position embedding differ.

The ablation is not subtle. Trained on ScanNet alone:

| FUSER, ScanNet-only training | Rot within 3° | Mean / median RE | TE within 0.05 m | Mean / median TE (m) |
| --- | --- | --- | --- | --- |
| Without 2D attention prior | 12.9% | 34.8° / 11.3° | 5.4% | 0.74 / 0.40 |
| With $\pi^3$ initialization | 36.6% | 22.6° / 4.1° | 15.4% | 0.45 / 0.16 |

The authors credit transferable attention habits (view grouping, alignment consistency, sparsity). That is a hypothesis; the paper does not probe the transferred heads.

## Technique 3: reference-free pose supervision

**The problem.** The world frame is arbitrary. Different datasets and sequences put the origin in different places, and choosing scan 1 as the reference breaks permutation equivariance.

**The mechanism.** FUSER predicts a pose $\hat{\mathbf{T}}_i$ per scan, but supervises only relative poses. For every ordered pair $i \neq j$ it forms $\hat{\mathbf{T}}_{i \leftarrow j} = \hat{\mathbf{T}}_i^{-1}\hat{\mathbf{T}}_j$ and compares it to ground truth with three losses:

$$
\mathcal{L}_r(i,j) = \arccos\!\left(\frac{\mathrm{Tr}\big(\mathbf{R}_{i\leftarrow j}^{\top}\hat{\mathbf{R}}_{i\leftarrow j}\big) - 1}{2}\right), \qquad
\mathcal{L}_t(i,j) = \ell_\beta\big(\hat{\mathbf{t}}_{i\leftarrow j} - \mathbf{t}_{i\leftarrow j}\big),
$$

$$
\mathcal{L}_p(i,j) = \frac{1}{N_j}\sum_{l=1}^{N_j} \big\| (\hat{\mathbf{R}}_{i\leftarrow j}\mathbf{p}_l + \hat{\mathbf{t}}_{i\leftarrow j}) - (\mathbf{R}_{i\leftarrow j}\mathbf{p}_l + \mathbf{t}_{i\leftarrow j}) \big\|_1 ,
$$

$$
\mathcal{L} = \frac{1}{N(N-1)} \sum_{i \neq j} \mathcal{L}_r(i,j) + \gamma_t \mathcal{L}_t(i,j) + \gamma_p \mathcal{L}_p(i,j).
$$

Here $\mathcal{L}_r$ is the geodesic angle between true and predicted relative rotations, $\ell_\beta$ is a Huber loss with threshold $\beta = 0.06$, $\mathbf{p}_l$ are the $N_j$ points of scan $j$, and $\gamma_t = \gamma_p = 0.1$. The rotation head outputs a 9D proxy that SVD projects onto a valid rotation matrix.

**Why it removes the frame.** Multiply every predicted pose on the left by any rigid motion $\mathbf{G}$. The relative pose is unchanged: $(\mathbf{G}\hat{\mathbf{T}}_i)^{-1}\mathbf{G}\hat{\mathbf{T}}_j = \hat{\mathbf{T}}_i^{-1}\mathbf{G}^{-1}\mathbf{G}\hat{\mathbf{T}}_j = \hat{\mathbf{T}}_i^{-1}\hat{\mathbf{T}}_j$. So the loss cannot reward or punish the network for where it puts the world origin, which is exactly the ambiguity the problem has.

**Worked example.** Say the true relative rotation is the identity and the prediction is off by 10° about the z axis. Then $\mathrm{Tr}(\hat{\mathbf{R}}) = 1 + 2\cos 10^\circ \approx 2.970$, so the argument of arccos is $(2.970 - 1)/2 \approx 0.985$, and $\mathcal{L}_r = \arccos(0.985) \approx 10^\circ$. The loss reads off the angle directly, which is why the paper's metrics report errors in degrees.

**When it fails.** The loss weights all pairs equally, overlapping or not. For scans on opposite sides of a building the relative pose is nearly unobservable, yet still supervised. Mild for a room; dominant at city scale (my extrapolation).

## Technique 4: FUSER-DF, diffusion over all poses jointly

FUSER is fast but coarse at the strictest thresholds. FUSER-DF refines its output with a denoising diffusion process whose state is the whole pose tuple $\mathbf{T}_{1:N} \in SE(3)^N$.

### Background: SE(3) diffusion for one pair

An earlier method (reference [40] in the paper) treats pairwise registration as diffusion on $SE(3)$. The forward process pushes the true transform $\mathbf{H}_0$ toward the identity $\mathbb{H}$ with noise; the reverse process starts at identity and denoises back, using a pairwise registration network $f_\theta$ as the denoiser. Two limits follow: it is pairwise only, and it starts every chain from an uninformed guess.

### The prior-aware forward process

FUSER-DF changes the endpoint. Instead of diffusing toward identity, each scan's pose diffuses from ground truth $\mathbf{T}_i^0$ toward FUSER's own prediction $\hat{\mathbf{T}}_i$:

$$
\mathbf{T}_i^t = \mathrm{Exp}\!\big(\gamma\sqrt{1-\bar\alpha_t}\,\boldsymbol\varepsilon\big)\; \mathcal{F}\big(\sqrt{\bar\alpha_t};\, \mathbf{T}_i^0, \hat{\mathbf{T}}_i\big), \qquad
\mathcal{F} = \mathrm{Exp}\!\Big((1-\sqrt{\bar\alpha_t})\,\mathrm{Log}\big(\hat{\mathbf{T}}_i (\mathbf{T}_i^0)^{-1}\big)\Big)\mathbf{T}_i^0 .
$$

$\mathrm{Exp}$ and $\mathrm{Log}$ map between the 6D tangent space (rotation vector plus translation) and the group. $\bar\alpha_t = \prod_{s \le t}(1-\beta_s)$ with a cosine schedule, $\boldsymbol\varepsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$, and $\gamma = 0.1$ scales the noise. Read $\mathcal{F}$ as "walk a fraction ${1}-\sqrt{\bar\alpha_t}$ of the geodesic from the truth toward the prior". At $t = 0$ you are at the truth; at $t = T$ you are at FUSER's guess, plus a small random kick.

![FUSER-DF starts the reverse chain at FUSER's estimate, so only a short correction remains.](/imgs/blogs/fuser-feed-forward-multiview-3d-registration-transformer-and-se-3-n-diffusion-refinement-3.webp)

The noise term adds training diversity; without it each scan would have one deterministic trajectory.

### The prior-aware reverse process

The reverse chain starts at FUSER's estimate and steps toward the truth. The derivation (a Lie-group cousin of [the DDPM posterior](/blog/machine-learning/image-generation/the-math-of-ddpm)) goes through a variational lower bound with a "prior-aware denoising matching" term. Its posterior mean has a clean three-way form:

$$
\mu^{\text{post}}_i = \mathrm{Exp}\!\Big(\lambda_0 \,\mathrm{Log}\big(\mathbf{T}_i^{t\to 0}\mathbf{T}_i^t\big) + \lambda_1 \,\mathrm{Log}(\mathbf{T}_i^t) + \lambda_2 \,\mathrm{Log}(\hat{\mathbf{T}}_i)\Big),
$$

where $\mathbf{T}_i^{t\to 0} = \mathbf{T}_i^0(\mathbf{T}_i^t)^{-1}$ is the residual from the noisy pose to the truth, and

$$
\lambda_0 = \frac{\sqrt{\bar\alpha_{t-1}}\,\beta_t}{1-\bar\alpha_t}, \quad
\lambda_1 = \frac{\sqrt{\alpha_t}\,(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}, \quad
\lambda_2 = 1 + \frac{(\sqrt{\bar\alpha_t}-1)(\sqrt{\alpha_t}+\sqrt{\bar\alpha_{t-1}})}{1-\bar\alpha_t}.
$$

$\lambda_0$ weighs the truth, $\lambda_1$ the current noisy pose, $\lambda_2$ the FUSER prior. The learned model replaces the unknown residual with a prediction, and here is the elegant part: that predictor is FUSER itself. Feed FUSER the scans transformed by their current noisy poses, $\mathcal{S}^t$, and it outputs all residuals at once, $(\hat{\mathbf{T}}_1^{t\to 0}, \dots, \hat{\mathbf{T}}_N^{t\to 0}) = \mathrm{FUSER}(\mathcal{S}^t)$. Matching the means then reduces to ordinary registration training, so the losses from Technique 3 apply unchanged.

![Figure 3 from Jiang et al. (2026): the prior-aware SE(3)^N denoising step, with FUSER as the surrogate that predicts residual poses for every scan.](/imgs/blogs/fuser-feed-forward-multiview-3d-registration-transformer-and-se-3-n-diffusion-refinement-fig2.webp)

**Worked example: how fast the prior fades.** I computed the three weights with the standard cosine schedule ($s = 0.008$, my assumption for the schedule's offset) and $T = 200$, the paper's training horizon. They always sum to 1, so the update is a weighted average in log space:

| Step $t$ | $\bar\alpha_t$ | $\lambda_0$ (truth) | $\lambda_1$ (noisy) | $\lambda_2$ (prior) |
| --- | --- | --- | --- | --- |
| 200 | 0.000 | 0.008 | 0.000 | 0.992 |
| 150 | 0.144 | 0.017 | 0.975 | 0.008 |
| 100 | 0.494 | 0.022 | 0.977 | 0.001 |
| 10 | 0.992 | 0.168 | 0.832 | 0.000 |
| 2 | 0.999 | 0.596 | 0.404 | 0.000 |

The paper says $\lambda_2$ "dominates at the beginning and gradually decays". Under this schedule it is closer to a switch: negligible by $t = 150$. The prior's real job is choosing the starting point, not steering the chain.

**When it fails.** Diffusion can only correct what the surrogate can see. If FUSER's first estimate has a scan in the wrong room (a symmetric layout flipped), the reverse chain starts in the wrong basin and ten small steps will not cross it. The paper does not report a failure breakdown that would show how often this happens.

## Results

Training mixes 3DMatch, ScanNet, ScanNet++ and ArkitScenes, with sequence length sampled from 2 to 50 scans, for 2 epochs with Adam (learning rate ${5} \times 10^{-5}$, weight decay 0.05) on 8 NVIDIA L20 GPUs. The model has about 0.6B parameters.

### ScanNet, 30 scans per sequence

![Table 1 from Jiang et al. (2026): multiview registration on ScanNet with full, pruned and sparse pose graphs against FUSER and FUSER-DF.](/imgs/blogs/fuser-feed-forward-multiview-3d-registration-transformer-and-se-3-n-diffusion-refinement-fig5.webp)

All baselines use YOHO descriptors for the pairwise step, except LMVR, which is end-to-end. The headline comparison against the strongest prior method, MDGD on a sparse graph:

| Method | #Pair | Rot within 3° | Mean / median RE | Mean / median TE (m) |
| --- | --- | --- | --- | --- |
| MDGD (sparse) | 6004 | 56.1% | 17.4° / 19.0° | 0.37 / 0.31 |
| SGHR (pruned) | 13920 | 59.4% | 21.7° / 19.1° | 0.64 / 0.47 |
| FUSER | 0 | 69.4% | 6.7° / 2.1° | 0.15 / 0.07 |
| FUSER-DF | 0 | 72.0% | 7.1° / 2.0° | 0.15 / 0.06 |

The median rotation error drops by roughly 9x. FUSER-DF helps where the thresholds are strict (translation within 0.05 m goes 36.2% to 43.2%) and slightly worsens the mean rotation error, which suggests it sharpens the good cases without rescuing the bad ones.

![Figure 4 from Jiang et al. (2026): FUSER versus FUSER-DF versus ground truth; the diffusion refinement yields visibly smoother merged surfaces.](/imgs/blogs/fuser-feed-forward-multiview-3d-registration-transformer-and-se-3-n-diffusion-refinement-fig3.webp)

### 3DMatch (60 scans) and ArkitScenes (200 scans)

The 3DMatch protocol here is new and harder than the usual one: sequences are built from raw single frames, 60 per sequence, rather than TSDF-fused fragments. Registration recall (RR) counts relative poses within 15° and 0.3 m.

| Dataset | Best baseline (RR / RE / TE) | FUSER | FUSER-DF |
| --- | --- | --- | --- |
| 3DMatch, 60 scans | Full+PARENet: 61.9% / 31.2° / 0.68 m | 90.3% / 3.2° / 0.14 m | 92.0% / 3.1° / 0.14 m |
| ArkitScenes, 200 scans | SGHR+GeoTrans: 26.7% / 91.2° / 1.48 m | 92.1% / 5.4° / 0.13 m | 95.0% / 5.6° / 0.12 m |

The ArkitScenes result stands out: trained on at most 50 scans, tested on 200, FUSER holds while pairwise pipelines collapse to RR around 25%.

![Figure 5 from Jiang et al. (2026): qualitative comparison of PARENet+SGHR, GeoTrans+SGHR, FUSER and ground truth, with registration recall and runtime per scene.](/imgs/blogs/fuser-feed-forward-multiview-3d-registration-transformer-and-se-3-n-diffusion-refinement-fig4.webp)

### Runtime

| Sequence | Fastest two-stage pipeline | FUSER | FUSER-DF |
| --- | --- | --- | --- |
| 3DMatch, 60 scans | SGHR+PARENet: 155.4 s | 0.31 s | 2.91 s |
| ArkitScenes, 200 scans | SGHR+PARENet: 583 s | 0.61 s | 6.50 s |

That is a factor of about 500 on 3DMatch and about 950 on ArkitScenes for the feed-forward model (my division of the paper's numbers). An appendix shows in-the-wild iPhone 14 Pro Max scans of a kitchen and a study registered in 0.547 s and 0.471 s with about 3 GB of GPU memory.

### Scaling with data

![Table 4 from Jiang et al. (2026): ablations on ScanNet, removing the 2D attention prior and adding training datasets one at a time.](/imgs/blogs/fuser-feed-forward-multiview-3d-registration-transformer-and-se-3-n-diffusion-refinement-fig6.webp)

Adding datasets helps on the headline metric, with one flat step: rotation within 3° goes 36.6% (ScanNet only) to 35.8% (+ArkitScenes) to 56.9% (+ScanNet++) to 69.4% (+3DMatch). Mean rotation error falls from 22.6° to 6.7°. Four indoor datasets is a small corpus and the curve has not flattened, the clearest sign this recipe will keep improving with data.

### What is load-bearing in the setup

- **The $\pi^3$ initialization.** Without it, ScanNet-only FUSER is worse than several classical baselines at the 3° threshold. Anyone reproducing this without the 2D weights should expect a much weaker model.
- **Indoor, in-domain data.** ScanNet and 3DMatch training splits are in the mix, and every benchmark is a room. Absolute coordinate encoding is tuned for that range.

## Critique

**What is strong.** No pose graph, large gains on three datasets, two to three orders of magnitude faster, and generalization from 50-scan training to 200-scan testing. The reference-free loss cleanly handles gauge freedom, and reusing FUSER as the surrogate means refinement needs no new network.

**What is weak.**

1. **FUSER-DF's value is narrow.** It costs about 9 to 11x the runtime and improves the strict-threshold tails, but mean rotation error gets slightly worse on ScanNet (6.7° to 7.1°) and ArkitScenes (5.4° to 5.6°). There is no comparison against a cheap alternative such as one pass of ICP or a second FUSER pass on the re-posed scans.
2. **The 3DMatch protocol is the authors' own.** Building sequences from single frames is more realistic, but all baselines were re-run under it, and it is not the protocol their numbers were tuned for.
3. **No failure analysis.** Median rotation error of 2.1° next to a mean of 6.7° says a tail of scans fails badly. Which ones? Symmetric rooms, low overlap, sparse scans? The paper does not say.
4. **The 2D prior explanation is untested.** "View grouping, alignment consistency, attention sparsity" are named, not measured. A layer-wise transplant ablation would settle it.

**What would change my mind.** If a matched-compute baseline (a pairwise-free, non-transformer method, or FUSER without $\pi^3$ trained on all four datasets) closed most of the gap, the credit would belong to data scale rather than to the 2D attention prior. Conversely, if FUSER held its RR on outdoor LiDAR with no retraining, I would upgrade it from "best indoor registration model" to "general registration backbone".

## Where this goes next

The orchestrating question for follow-up work is the one the paper stops short of: scale feed-forward registration to LiDAR maps with hundreds of scans and loop closures. Here are concrete projects that FUSER makes tractable.

1. **Chunked FUSER with learned loop closure.**
   *Question:* can a feed-forward registrar handle a 2,000-scan drive whose start and end revisit the same street?
   *Why now:* FUSER already runs 200 scans in 0.61 s and 2.83 GB, and its losses are reference-free, so overlapping chunks can be registered independently and stitched with one rigid transform per chunk.
   *First experiment:* KITTI odometry sequences split into 150-scan windows with 50% overlap; stitch windows by their shared scans; measure drift with and without a cross-window attention pass over only the scans flagged as revisits.

2. **Range-adaptive absolute encoding.**
   *Question:* does absolute coordinate encoding survive a 100 m LiDAR sweep?
   *Why now:* the paper shows relative RoPE hurts and absolute encoding helps, but only at room scale. That makes the encoding choice a clear, isolated variable.
   *First experiment:* swap the five-level voxel hierarchy for a log-polar or range-binned sparse grid on nuScenes or KITTI; compare against plain FUSER fine-tuned on the same data.

3. **FUSER as a pose-graph initializer, not a replacement.**
   *Question:* is the best large-scale system FUSER on local windows plus classical sparse optimization on top?
   *Why now:* FUSER gives sub-second, globally consistent local poses. That is exactly the initialization pose-graph optimizers crave, and it removes the quadratic pairwise stage.
   *First experiment:* feed FUSER window poses as odometry factors into GTSAM with a standard place-recognition loop detector; compare ATE against LiDAR SLAM baselines at equal runtime.

4. **Probe the 2D-to-3D attention transfer.**
   *Question:* which $\pi^3$ layers carry the gain, and is it about multiview structure or just good initialization statistics?
   *Why now:* the structural compatibility means layer subsets can be swapped one at a time with no architecture change.
   *First experiment:* initialize only the cross-scan layers, only the intra-scan layers, and a variance-matched random init; report rotation within 3° on ScanNet-only training.

5. **Uncertainty-aware diffusion refinement.**
   *Question:* can FUSER-DF's stochasticity flag the tail of failed scans?
   *Why now:* the reverse process already samples noise; run it several times and the per-scan spread is a free uncertainty estimate.
   *First experiment:* run 8 seeds of FUSER-DF, rank scans by pose variance, and check whether high-variance scans account for most of the gap between median and mean error.

## References

- Haobo Jiang, Jin Xie, Jian Yang, Liang Yu, Jianmin Zheng. *FUSER: Feed-Forward MUltiview 3D Registration Transformer and SE(3)^N Diffusion Refinement.* CVPR 2026. [arXiv:2512.09373](https://arxiv.org/abs/2512.09373)
- Code: [github.com/Jiang-HB/FUSER](https://github.com/Jiang-HB/FUSER)
- CVPR 2026 page: [cvpr.thecvf.com/virtual/2026/poster/38237](https://cvpr.thecvf.com/virtual/2026/poster/38237)
- [VGGT: Visual Geometry Grounded Transformer](/blog/paper-reading/computer-vision/vggt-visual-geometry-grounded-transformer)
- [QuadSync: Quadrifocal Tensor Synchronization via Tucker Decomposition](/blog/paper-reading/computer-vision/quadsync-quadrifocal-tensor-synchronization-via-tucker-decomposition)
- [The math of DDPM](/blog/machine-learning/image-generation/the-math-of-ddpm)
