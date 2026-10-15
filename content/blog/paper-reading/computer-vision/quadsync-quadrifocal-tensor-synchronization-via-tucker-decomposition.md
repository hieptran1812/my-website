---
title: "QuadSync: Making Four-View Geometry Practical with One Tucker Decomposition"
publishDate: "2026-10-15"
date: "2026-10-15"
description: "A close read of QuadSync, a CVPR 2026 award candidate: stack every quadrifocal tensor of a scene into one block tensor, notice it always has Tucker rank (4,4,4,4) with the cameras as factors, and synchronize all poses at once."
excerpt: "Quadrifocal tensors were long filed under 'theoretical interest only'. QuadSync shows that the block tensor built from all of them is a fixed sparse core multiplied by the stacked camera matrix on four sides, which turns four-view geometry into a low-rank recovery problem. Here is the geometry, the algebra, the ADMM-IRLS solver, and where it still breaks."
category: "paper-reading"
subcategory: "Computer Vision"
tags: ["3d-reconstruction", "cvpr-2026", "cvpr-2026-award-candidate", "paper-reading", "quadsync", "structure-from-motion", "multi-view-geometry", "tensor-decomposition", "camera-pose-estimation", "synchronization"]
author: "Hiep Tran"
featured: true
image: "/imgs/blogs/quadsync-quadrifocal-tensor-synchronization-via-tucker-decomposition-fig1.webp"
readTime: 20
paper:
  title: "QuadSync: Quadrifocal Tensor Synchronization via Tucker Decomposition"
  authors: "Daniel Miao, Gilad Lerman, Joe Kileel"
  venue: "CVPR 2026 (oral, award candidate); arXiv:2602.22639"
  url: "https://arxiv.org/abs/2602.22639"
---

> [!tldr]
> - **What it claims.** Stack the quadrifocal tensor of every camera quadruple into one big block tensor. With the right per-block scales, that tensor equals a fixed sparse ${4 \times 4 \times 4 \times 4}$ core multiplied on all four sides by the stacked camera matrix. Its multilinear rank is (4,4,4,4) no matter how many cameras there are.
> - **Why it matters.** It is the first global synchronization method for four-view measurements: cameras fall out of a higher-order SVD, then a robust ADMM-IRLS solver refines them.
> - **The surprising result.** The rank does not drop when cameras are collinear, the exact case where pairwise location averaging fails and pairwise and trifocal block methods lose rank.
> - **Where it is soft.** It needs dense viewing graphs. On datasets where fewer than about 30% of quadrifocal blocks are estimated, it loses to two-view pipelines, its rotation errors are often worse than MPLS, and runtime grows fast (4,226 s on the 46-camera ETH3D facade scene for Joint Opt.).

![Figure 1 from Miao, Lerman and Kileel (2026): mean location error on 11 ETH3D scenes for QuadSync, Joint Opt. and five baselines (log scale).](/imgs/blogs/quadsync-quadrifocal-tensor-synchronization-via-tucker-decomposition-fig1.webp)

On most ETH3D scenes the two new methods sit at or near the bottom (lower is better); on a few they lose badly. This post explains both halves.

## The problem: synchronization only ever listened to pairs

Structure from motion (SfM) runs feature matching, relative pose estimation, **synchronization**, then reconstruction. Synchronization is the step that takes many noisy *relative* measurements ("camera 3 is rotated 12 degrees from camera 7") and finds one set of *absolute* camera poses that agrees with as many of them as possible.

Incremental systems like COLMAP add cameras one at a time and accumulate error. Global systems like GLOMAP and Theia solve for everyone at once, but they listen almost exclusively to **pairwise** reports: essential or fundamental matrices between two views.

Pairs lose information: a pairwise measurement gives the *direction* between two cameras, not the distance. If every camera lies on one line, as on a car driving down a road, all those directions are identical and the spacing becomes unknowable. Higher-order measurements that tie three or four views to the same 3D point carry the missing scale information. The paper quotes earlier authors calling quadrifocal tensors "useful only from a theoretical stand-point." QuadSync is the rebuttal.

![Four cameras look at one world point X; the fundamental matrix ties two rays, the trifocal tensor three, and the quadrifocal tensor all four.](/imgs/blogs/quadsync-quadrifocal-tensor-synchronization-via-tucker-decomposition-1.webp)

The figure shows the three levels: a ${3 \times 3}$ fundamental matrix, a 27-number trifocal tensor, and an 81-number quadrifocal tensor that treats all four views symmetrically.

The same group's [TrifocalSync (NeurIPS 2024)](https://arxiv.org/abs/2409.09313) showed the block trifocal tensor has Tucker rank (6,4,4). QuadSync goes one order up. If you want the opposite philosophy, where a network predicts all cameras in one forward pass, see our read of [VGGT](/blog/paper-reading/computer-vision/vggt-visual-geometry-grounded-transformer).

## Contributions

1. **A new object and its theory.** The block quadrifocal tensor $\mathcal{Q}^n$, with an exact Tucker factorization whose factors are the cameras, multilinear rank (4,4,4,4) and projection rank (2,2,2,2,2,2), both independent of $n$.
2. **A scale-recovery guarantee.** For ${n \ge 5}$ generic cameras, the rank constraint alone pins down the unknown per-block scales up to a rank-one pattern, so cameras are recoverable.
3. **QuadSync.** The first global synchronization algorithm for quadrifocal tensors: HOSVD initialization, then ADMM inside an IRLS loop.
4. **Joint Opt.** A new factorization of the block essential matrix that shares factors with the block trifocal tensor, so one solver synchronizes two-, three- and four-view measurements together.

## Method

### Step 0: what a quadrifocal tensor actually is

**The problem.** We need a single algebraic object that says "these four image points are views of the same 3D point." The fundamental matrix does this for two views via ${x_i^\top F_{ij} x_j = 0}$.

**The intuition.** A 3D point has 3 unknowns and four image points give 8 constraints. The quadrifocal tensor is the bookkeeping of that over-determination.

**The mechanism.** Each camera is a ${3 \times 4}$ matrix, so it has three rows, each a vector in $\mathbb{R}^4$. Pick one row from each of four cameras, stack them into a ${4 \times 4}$ matrix, and take its determinant. Do this for all ${3^4 = 81}$ row choices and you have the tensor.

**The math.** A pinhole camera is

$$
P_i = K_i R_i \,[\, I_{3\times 3} \mid -t_i \,] \in \mathbb{R}^{3\times 4},
$$

where $K_i \in \mathbb{R}^{3\times 3}$ is the calibration, $R_i \in SO(3)$ the orientation and $t_i \in \mathbb{R}^3$ the camera location. A world point $X \in \mathbb{R}^4$ (homogeneous) projects to $x_i = P_i X$. The quadrifocal tensor of cameras $i,j,k,l$ is

$$
(Q_{ijkl})_{pqrs} = \det \begin{bmatrix} P_i^{p} \\ P_j^{q} \\ P_k^{r} \\ P_l^{s} \end{bmatrix}, \qquad p,q,r,s \in \{1,2,3\},
$$

where $P_i^{p}$ is row $p$ of $P_i$. Like every multiview tensor it is defined only up to a nonzero scale, because each $P_i$ is.

**Tiny example.** If you pick all four rows from the same camera, you are taking the determinant of four vectors chosen from only three rows, so two rows repeat and the determinant is zero. That is exactly the paper's Proposition 3.3: the "super-diagonal" blocks $Q_{iiii}$ vanish.

**When it fails.** Estimating a quadrifocal tensor from raw matches is little-studied, so the paper builds them from trifocal tensors (Step 6), and the theory inherits that step's error.

### Step 1: the block quadrifocal tensor

**The problem.** Our four-view measurements each have an arbitrary scale and cover only some quadruples. We need one container where global structure shows.

**The mechanism.** Given $n$ cameras, build $\mathcal{Q}^n \in \mathbb{R}^{3n \times 3n \times 3n \times 3n}$ and put $Q_{ijkl}$ into block position $(i,j,k,l)$. Every quadruple of camera indices, including repeated ones, gets a ${3 \times 3 \times 3 \times 3}$ block.

![The block quadrifocal tensor for n = 4 cameras: 256 blocks, color-coded by which index pattern they hold.](/imgs/blogs/quadsync-quadrifocal-tensor-synchronization-via-tucker-decomposition-2.webp)

A 4D array cannot be drawn directly, so the figure nests two grids: the outer grid indexes $(i,j)$ and the inner grid $(k,l)$. Only the 24 blocks with four distinct indices are "true" four-view tensors. The rest hold lower-order geometry (Proposition 3.3):

| Index pattern | Count for n = 4 | What the block contains |
| --- | --- | --- |
| all four distinct | 24 | the quadrifocal tensor proper |
| one pair equal | 144 | entries of the trifocal tensor $T_{ijk}$ |
| two pairs equal | 36 | entries of the fundamental matrix $F_{ij}$ |
| three equal | 48 | the epipole: camera center of view $i$ seen in view $j$ |
| all four equal | 4 | exactly zero |

So the block tensor already contains the pairwise and triplewise information. Also, permuting indices only flips a sign, ${\mathcal{Q}^n_{\pi(ijkl)} = \text{sgn}(\pi)\,\mathcal{Q}^n_{ijkl}}$, because swapping rows of a determinant flips its sign. That lets the unknown scale tensor $\Lambda$ be restricted to symmetric tensors, cutting the number of free scales by roughly ${24\times}$.

Missing measurements are set to zero. The paper writes the observed index set as $\Omega$ and the measured tensor as

$$
\tilde{\mathcal{Q}}^n = \mathcal{P}_\Omega\!\left(\Lambda \odot_b \mathcal{Q}^n\right),
$$

where $\Lambda \in \mathbb{R}^{n\times n\times n\times n}$ holds the unknown per-block scales $\lambda_{ijkl}$, $\odot_b$ multiplies each block by its scale, and $\mathcal{P}_\Omega$ zeroes every block outside $\Omega$.

### Step 2: the Tucker decomposition, the heart of the paper

**The problem.** A tensor with $(3n)^4$ entries is huge: for ${n = 10}$ that is 810,000 numbers. We need a constraint that says "these numbers came from real cameras" and that does not get weaker as $n$ grows.

**The intuition.** A Tucker decomposition writes a big tensor as a small core tensor multiplied by one matrix along each axis, like a matrix SVD $U \Sigma V^\top$ generalized to more than two axes. If the factor matrices are tall and thin, the tensor is "low rank" in every direction.

**The mechanism.** The determinant is multilinear in its rows. Expanding $\det$ of four row vectors gives a sum over the 24 permutations of four coordinates, each term a product of one coordinate from each row, with sign $\pm 1$. The sign pattern is a fixed ${4\times4\times4\times4}$ array, and the row vectors all come from one matrix: the stacked cameras.

**The math.** Theorem 3.1 states that for suitable blockwise scales,

$$
\mathcal{Q}^n = \mathcal{G}_Q \times_1 C \times_2 C \times_3 C \times_4 C, \qquad C = \begin{bmatrix} P_1 \\ \vdots \\ P_n \end{bmatrix} \in \mathbb{R}^{3n \times 4},
$$

where $\mathcal{G}_Q \in \mathbb{R}^{4\times4\times4\times4}$ is a constant sparse core with entries in $\{-1, 0, 1\}$, and $\times_m$ is the mode-$m$ product (multiply every mode-$m$ fiber by the matrix). When the cameras do not all share one center, ${\text{mlrank}(\mathcal{Q}^n) = (4,4,4,4)}$. The multilinear rank is the tuple of matrix ranks of the four flattenings $\mathcal{Q}^n_{(m)} \in \mathbb{R}^{3n \times 27n^3}$.

Reading the determinant expansion, the core is the 4D Levi-Civita symbol: 24 nonzero entries out of 256. That identification is my reading of the proof sketch; the paper only states the entry set and sparsity.

![The Tucker factorization: a 3n x 3n x 3n x 3n tensor equals a fixed 4x4x4x4 core multiplied by the same 3n x 4 camera stack on all four modes.](/imgs/blogs/quadsync-quadrifocal-tensor-synchronization-via-tucker-decomposition-3.webp)

The left cube grows like $n^4$; the right side is a fixed core plus one ${3n \times 4}$ matrix. Corollary 3.1.1 follows immediately: unfold $\mathcal{Q}^n$ along any mode, take the top 4 left singular vectors, and they span $CH$ for some unknown invertible ${4\times4}$ matrix $H$. Rows ${3i-2}$ to ${3i}$ of that basis are camera $i$, up to one global projective transform.

**Worked example (mine, not the paper's).** I built 5 random calibrated cameras, filled $\mathcal{Q}^5$ (${15^4}$ entries) by the determinant formula with unit scales, and took the singular values of the mode-1 flattening:

```python
import numpy as np, itertools
rng = np.random.default_rng(0)
Ps = []
for i in range(5):
    R, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    t = rng.normal(size=3)
    Ps.append(R @ np.hstack([np.eye(3), -t[:, None]]))
C = np.vstack(Ps)                      # (15, 4) stacked cameras
N = C.shape[0]
Q = np.zeros((N, N, N, N))
for a, b, c, d in itertools.product(range(N), repeat=4):
    Q[a, b, c, d] = np.linalg.det(np.stack([C[a], C[b], C[c], C[d]]))
s = np.linalg.svd(Q.reshape(N, -1), compute_uv=False)
print(np.round(s[:6] / s[0], 6))       # [1. 1. 1. 1. 0. 0.]
U = np.linalg.svd(Q.reshape(N, -1))[0][:, :4]
print(np.linalg.norm(C - U @ (U.T @ C)) / np.linalg.norm(C))  # ~2.6e-15
```

Exactly four nonzero singular values, and the cameras lie in their span to machine precision: the whole HOSVD initialization.

**When it fails.** The factorization holds only for the *right* scales. Measured blocks come with arbitrary scales, and the next step is about that.

### Step 3: why the rank pins down the unknown scales

**The problem.** Each $Q_{ijkl}$ is estimated up to its own scale $\lambda_{ijkl}$. With wrong scales the block tensor is no longer low rank, and the cameras cannot be read off.

**The intuition.** Low rank is a rigid property, so very few scale patterns can preserve it.

**The math.** Theorem 3.4: for ${n \ge 5}$ generic cameras, if $\Lambda \odot_b \mathcal{Q}^n$ has multilinear rank (4,4,4,4), then there exist $\alpha, \beta, \gamma, \delta \in \mathbb{R}^n$ with

$$
\lambda_{ijkl} = \alpha_i \beta_j \gamma_k \delta_l \quad \text{whenever } i,j,k,l \text{ are not all equal}.
$$

A rank-one scale pattern just rescales each camera, which is a harmless ambiguity. So the rank constraint is enough to recover the scales.

**Worked example (mine).** I multiplied each of the 625 blocks of my 5-camera tensor by a random scale in $[0.5, 2]$. The normalized singular values went from (1, 1, 1, 1, 0, 0, ...) to (1, 0.976, 0.967, 0.960, 0.213, 0.193, 0.184, 0.162). The fifth singular value jumped from 0 to 0.21. That gap is precisely the signal the solver drives back to zero by choosing $\Lambda$.

The paper also proves a projection rank of (2,2,2,2,2,2) (Theorem 3.2), a constraint not implied by the multilinear rank. QuadSync enforces it implicitly by keeping the core fixed rather than estimating it.

Why four views over two? Remark 2 counts codimensions: the set of valid block tensors has codimension $\Omega(n^4)$ in its ambient space, against $\Omega(n^3)$ for block trifocal tensors and $\Omega(n^2)$ for block essential matrices. More constraints per unknown means noise has less room to hide.

### Step 4: the collinear case

**The problem.** Pairwise methods estimate locations from directions $t_{ij}$. On a straight trajectory every direction is the same unit vector, and the spacing is undetermined.

![Collinear cameras: direction-only averaging cannot tell two spacings apart, while four-view blocks keep their rank and fix the spacing.](/imgs/blogs/quadsync-quadrifocal-tensor-synchronization-via-tucker-decomposition-4.webp)

In the left panel a ghost configuration with different spacing produces identical pairwise directions. Algebraically (Remark 1), when cameras are collinear, the n-view fundamental matrix drops from rank 6 to rank 4 and the block trifocal tensor drops from (6,4,4) to (5,4,4). Methods built on them need patches such as virtual cameras.

The block quadrifocal tensor keeps (4,4,4,4) unless all cameras share the *same center*. My demo above confirms it: placing the 5 camera centers on the x-axis still gave exactly four nonzero singular values.

![Figure 3 from Miao, Lerman and Kileel (2026): QuadSync camera poses (red) against ground truth (black) on a near-collinear subsequence of ETH3D SLAM plant scene 1.](/imgs/blogs/quadsync-quadrifocal-tensor-synchronization-via-tucker-decomposition-fig3.webp)

On a real near-collinear sequence the recovered poses sit on the ground truth. The paper states this configuration cannot be reconstructed from fundamental matrices. In synthetic collinear tests (appendix Table 1) with all quadruples observed and 1% camera noise, mean location error is 0.04 and mean rotation error 0.23 degrees; with 60% of quadruples and 5% noise it degrades to 0.64 and 4.52 degrees.

### Step 5: the QuadSync solver (ADMM inside IRLS)

**The problem.** Fit cameras $C$ and scales $\Lambda$ to a noisy, incomplete $\tilde{\mathcal{Q}}^n$; the fit is quartic in $C$ and nonconvex in the scales.

**The objective.** The paper minimizes a sum of *unsquared* block residuals, which is robust to outliers:

$$
\min_{\Lambda, C} \sum_{(i,j,k,l)\in\Omega} \left\| \Lambda_{ijkl} (\tilde{\mathcal{Q}}^n)_{ijkl} - [\![ \mathcal{G}_Q; C, C, C, C ]\!]_{ijkl} \right\|_F \quad \text{s.t. } \Lambda \in S^4(\mathbb{R}^n),\ \|\Lambda\|_F^2 = 1.
$$

Here $[\![ \mathcal{G}_Q; C, C, C, C ]\!]$ is shorthand for the Tucker product, $S^4(\mathbb{R}^n)$ is the set of symmetric order-4 tensors, and the norm constraint rules out the trivial answer $\Lambda = 0$.

**Trick 1: split the factors (ADMM).** Four copies of the same $C$ make the problem quartic. Replace them with four independent copies $C_1, \dots, C_4$ plus a consensus variable $B$, and require ${C_1 = C_2 = C_3 = C_4 = B}$. With three of them fixed, each $C_m$ enters linearly, so its update is a least-squares solve, and each of its ${3n}$ rows is an independent ${4\times 4}$ system. The consensus is enforced by the usual ADMM penalty $\frac{\rho}{2}\sum_m \|C_m - B + \Gamma_m\|_F^2$, where $\Gamma_m$ are scaled dual variables. Then $B$ is the average of $C_m + \Gamma_m$ and each $\Gamma_m$ accumulates the disagreement $C_m - B$.

**Trick 2: reweight (IRLS).** An unsquared norm is awkward to minimize, so each outer iteration replaces it with a weighted squared norm. The weight of block $(i,j,k,l)$ is $w_{ijkl} = 1/m_{ijkl}$ with $m_{ijkl} = \max(\delta, \sqrt{r_{ijkl}})$, where $r_{ijkl}$ is that block's residual norm from the previous iteration and $\delta$ caps the weight.

**Worked example.** Squared weights times squared residuals give $w^2 r^2 = r^2 / r = r$ whenever ${r > \delta^2}$, so the weighted least-squares problem reproduces the robust loss. Take three blocks with residuals 0.1, 0.1 and 2.0, the last one an outlier. Under plain least squares the outlier's gradient is ${2 \times 2.0 = 4.0}$ against 0.2 for each good block, a 20-to-1 pull. Under IRLS each block contributes a gradient of unit size, so the outlier pulls no harder than any honest block.

**Scales in closed form.** With cameras fixed, each $\Lambda_{ijkl}$ is a one-dimensional least-squares fit of the measured block to the model block, then $\Lambda$ is symmetrized and renormalized.

```python
# One QuadSync outer iteration, schematically (shapes for n cameras)
# Qt: measured block tensor (3n,3n,3n,3n), blocks normalized to norm 1
# C1..C4, B: (3n, 4); Gam1..Gam4: (3n, 4); Lam: (n, n, n, n) symmetric
W = 1.0 / np.maximum(delta, np.sqrt(block_residual_norms(Lam, Qt, G, C1, C2, C3, C4)))
for _ in range(10):                      # inner alternation, as in the paper
    for m in range(4):
        Cs[m] = solve_rows(m, W, Lam, Qt, G, Cs, B, Gams[m], rho)   # 3n tiny 4x4 solves
    Lam = symmetrize(blockwise_scale_fit(W, Qt, G, Cs))
    Lam /= np.linalg.norm(Lam)
B = sum(Cs[m] + Gams[m] for m in range(4)) / 4
for m in range(4):
    Gams[m] += Cs[m] - B
```

**Settings.** HOSVD initialization, $\Gamma_m = 0$, ${\rho = 0.01}$, 4 IRLS rounds with 1 ADMM loop each. HOSVD is enough to start this nonconvex problem, unlike two-view methods that need specialized initializers.

**When it fails, and a fix.** The row solve multiplies a ${4 \times 27n^3}$ matrix by a ${27n^3 \times 27n^3}$ diagonal weight. For the 13-image relief scene that is ${27 \times 13^3 = 59{,}319}$ columns. The appendix samples random columns instead and reports similar accuracy with only 30 of them. Because the rank does not grow with $n$, the sample size can stay $O(1)$.

### Step 6: where the quadrifocal tensors come from

Nobody ships a robust quadrifocal estimator, so the authors build one from trifocal tensors:

1. Estimate trifocal tensors from point matches (robust outlier rejection, linear fit, bundle adjustment).
2. For every 4-cycle $(i,j,k,l)$, chain $T_{ijk}, T_{jkl}, T_{kli}, T_{lij}$ through ${4\times4}$ projective transforms on shared cameras and measure how far cameras $i$ and $j$ drift around the loop.
3. Drop cycles with rotation drift above 3 degrees or translation drift above 0.2; for the rest, average the camera estimates and compute $Q_{ijkl}$ by the determinant formula.

Trifocal blocks are filled in too, and every block is normalized to unit norm. The cycle test doubles as an outlier filter.

### Step 7: Joint Opt., all three orders at once

The block trifocal tensor factors as $[\![ \mathcal{G}_T; P, C, C ]\!]$, with $P \in \mathbb{R}^{3n\times 6}$ the stacked line projection matrices. Theorem 4.1 adds a new factorization of the block essential matrix, $E^n = P\, G_E\, P^\top$ with a fixed ${6\times 6}$ core $G_E$. All three objects share $C$ or $P$, so one objective sums the three weighted residuals, each divided by its block count ($n_Q$, $n_T$, $n_E$) so no order dominates, with ADMM consensus on both $C$ and $P$. It runs calibrated only.

## Experiments and results

The benchmarks are 11 ETH3D and 6 EPFL scenes, against TrifocalSync, NRFM, LUD, and MPLS rotations paired with BATA or Cycle-Sync locations. Graphs are pruned first: cameras in less than 5% of observed quadruples are dropped (relaxed to 2%, then 1%, if too few remain).

![Figure 2 from Miao, Lerman and Kileel (2026): mean location error on 6 EPFL scenes (log scale).](/imgs/blogs/quadsync-quadrifocal-tensor-synchronization-via-tucker-decomposition-fig2.webp)

The paper reports QuadSync or Joint Opt. as best or near-best on location in 7 of 11 ETH3D scenes and 4 of 6 EPFL scenes. The appendix tables show the pattern behind that count:

| Scene (cameras kept / total) | Quad blocks estimated | QuadSync | Joint Opt. | TrifocalSync | MPLS + Cycle-Sync |
| --- | --- | --- | --- | --- | --- |
| relief (13/13) | 95.21% | 0.0020 | 0.0020 | 0.0020 | 0.0052 |
| terrace (11/23) | 92.53% | 0.0096 | 0.0096 | 0.0103 | 0.0147 |
| HerzP8 (8/8) | 80.76% | 0.0010 | 0.0010 | 0.0180 | 0.0375 |
| FountainP11 (11/11) | 76.55% | 0.0002 | 0.0002 | 0.0098 | 0.0095 |
| EntryP10 (10/10) | 26.50% | 0.0007 | 0.0007 | 0.0654 | 0.0897 |
| facade (46/76) | 61.98% | 3.0294 | 0.0248 | 0.0222 | 0.0158 |
| relief 2 (16/20) | 14.31% | 0.5210 | 0.5360 | 0.5554 | 0.0158 |
| terrains (23/42) | 8.29% | 0.4631 | 0.4335 | 0.0797 | 0.0075 |

Mean location error in each dataset's units (appendix Tables 3 and 4). On dense scenes the gap is up to two orders of magnitude (0.0002 against 0.0095 on FountainP11); on sparse ones the two-view baseline wins by a similar margin. EntryP10 wins at 26.50% completion, so density is a strong predictor, not a law.

![Table 4 from Miao, Lerman and Kileel (2026): mean location error for all 17 scenes and 7 methods.](/imgs/blogs/quadsync-quadrifocal-tensor-synchronization-via-tucker-decomposition-fig5.webp)

Three more details from the appendix matter:

- **Rotation is the weak side.** Mean rotation error for QuadSync is 32.67 degrees on relief 2 and 30.49 on CastleP19, where MPLS gets 0.10 and 0.28. Even on scenes QuadSync wins for location, MPLS rotations are often lower (terrace 0.0693 against 0.0796).
- **Joint Opt. rescues failures.** On facade, QuadSync alone hits 3.03 while Joint Opt. gets 0.0248; the pairwise and triplewise terms anchor what the sparse quadrifocal blocks cannot.
- **Runtime grows fast.** Joint Opt. takes 36.3 s on relief but 4,226.2 s on the 46-camera facade, against 1,633.0 s for TrifocalSync.

![Figure 6 from Miao, Lerman and Kileel (2026): CastleP30 poses recovered per handpicked cluster (colored) against ground truth (black).](/imgs/blogs/quadsync-quadrifocal-tensor-synchronization-via-tucker-decomposition-fig4.webp)

On a synthetic 30-camera scene split into three overlapping clusters, noise-free runtime falls from 1,666.3 s for the full problem to 150.2 s for clusters plus alignment, with 2 CPU cores. The price is accuracy: at 1% noise, mean location error rises from 0.11 (full) to 0.53 (aligned). CastleP30, which QuadSync cannot solve whole because cameras face outward and see little in common, reconstructs well from handpicked clusters.

**What may not transfer.** The win needs a dense, cleanly filtered quadruple graph. Turntable-like captures (EPFL, small ETH3D scenes) are ideal; a drone sweep where each camera sees only its neighbors is the opposite regime.

## Critique

**What is strong.** "The cameras are the Tucker factors" is a clean, new statement, backed by a scale-identifiability theorem rather than a heuristic. The collinear result answers a real failure mode (driving, corridors) structurally, not with a patch. Fixing the core enforces the projection-rank constraint for free.

**What is weak.**
- The quadrifocal tensors are synthesized from chained trifocal estimates, so some "four-view information" is three-view information re-expressed. The paper cannot separate the gain of the cycle filter from that of fourth-order geometry.
- Rotation accuracy lags strong rotation averagers, and the paper's headline count focuses on location.
- Datasets are small (largest kept scene: 46 cameras) and pruned for density, which favors the method.
- No end-to-end comparison with GLOMAP or COLMAP after bundle adjustment.

**Missing ablation.** Run TrifocalSync on the *same* cycle-filtered triplets that QuadSync uses. If it closes most of the gap, the filter is the contribution; if not, the fourth order is.

**What would change my mind.** If a version using directly estimated quadrifocal tensors (from 6-point or line correspondences) matched or beat the trifocal-derived version on sparse scenes, I would believe four-view synchronization is broadly practical, not just a dense-graph specialist. Conversely, if the filtered-TrifocalSync ablation matched QuadSync, I would read the paper as a theory contribution with a clever preprocessing win.

## Where this goes next

These are my extrapolations, not the paper's claims.

1. **Learned initialization, classical certification.** *Question:* can a feed-forward model like VGGT, or a registration network like [FUSER](/blog/paper-reading/computer-vision/fuser-feed-forward-multiview-3d-registration-transformer-and-se-3-n-diffusion-refinement), supply $C$ and let the rank-(4,4,4,4) residual act as a consistency check? *Why now:* the Tucker form is a cheap, explicit test of whether predicted cameras agree with measured four-view geometry. *First experiment:* initialize QuadSync from VGGT poses on ETH3D, compare against the HOSVD start, and check whether the fifth singular value flags bad predictions.

2. **Robustness to adversarial block corruption.** *Question:* how many quadruples can be arbitrary outliers before the IRLS solution breaks? *Why now:* the codimension count promises more redundancy than pairwise methods. *First experiment:* replace a growing fraction of blocks with random tensors and plot location error against corruption rate for QuadSync, TrifocalSync and NRFM.

3. **Adaptive mixing of orders on sparse graphs.** *Question:* can the solver weight pairwise, trifocal and quadrifocal terms per region, using four-view blocks only where the graph is dense? *Why now:* Joint Opt. already shares factors across all three orders, and facade shows the mix rescuing a QuadSync failure. *First experiment:* replace the global ${1/n_Q, 1/n_T, 1/n_E}$ normalization with weights based on local completion rate and rerun the 8 scenes where QuadSync loses.

4. **Automatic clustering for scale.** *Question:* can graph partitioning pick the clusters the distributed experiment handpicked? *Why now:* the 11x runtime drop on 30 cameras shows the payoff, and the alignment step is just a projective transform on shared cameras. *First experiment:* partition by quadruple density with overlaps of 5 cameras, then compare accuracy and runtime with the full solve on EPFL and a 200-camera 1DSfM scene.

5. **Native quadrifocal estimation for collinear video.** *Question:* can quadrifocal tensors estimated from 6 point or line correspondences in dashcam video beat pairwise SfM on straight trajectories? *Why now:* the collinear rank result is the clearest win, and dynamic-video pipelines like [MegaSaM](/blog/paper-reading/computer-vision/megasam-accurate-fast-and-robust-structure-and-motion-from-casual-dynamic-videos) struggle exactly when baselines are small and parallel. *First experiment:* KITTI straight segments, sliding windows of 4 frames, compare scale drift against a pairwise global SfM.

## References

- Daniel Miao, Gilad Lerman, Joe Kileel. *QuadSync: Quadrifocal Tensor Synchronization via Tucker Decomposition.* CVPR 2026. [arXiv:2602.22639](https://arxiv.org/abs/2602.22639), [CVF open access](https://openaccess.thecvf.com/content/CVPR2026/html/Miao_QuadSync_Quadrifocal_Tensor_Synchronization_via_Tucker_Decomposition_CVPR_2026_paper.html).
- Code: [github.com/dmiao153/QuadSync](https://github.com/dmiao153/QuadSync).
- Daniel Miao, Gilad Lerman, Joe Kileel. *Tensor-based synchronization and the low-rankness of the block trifocal tensor.* NeurIPS 2024. [arXiv:2409.09313](https://arxiv.org/abs/2409.09313).
- Richard Hartley, Andrew Zisserman. *Multiple View Geometry in Computer Vision.* Cambridge University Press, 2003.
- On this site: [VGGT](/blog/paper-reading/computer-vision/vggt-visual-geometry-grounded-transformer), [FUSER](/blog/paper-reading/computer-vision/fuser-feed-forward-multiview-3d-registration-transformer-and-se-3-n-diffusion-refinement), [MegaSaM](/blog/paper-reading/computer-vision/megasam-accurate-fast-and-robust-structure-and-motion-from-casual-dynamic-videos).
