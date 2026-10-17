---
title: "Understanding and Enforcing Weight Disentanglement in Task Arithmetic"
publishDate: "2026-10-17"
date: "2026-10-17"
description: "Why adding fine-tuned weight deltas together works at all, how task-feature specialization explains it, and what the OrthoReg regularizer really does to task vectors during fine-tuning."
category: "paper-reading"
subcategory: "Computer Vision"
tags: ["interpretability-analysis", "cvpr-2026", "cvpr-2026-oral", "task-arithmetic", "model-merging", "orthoreg", "weight-disentanglement", "task-vectors", "clip", "neural-tangent-kernel"]
author: "Hiep Tran"
featured: true
readTime: 18
image: "/imgs/blogs/understanding-and-enforcing-weight-disentanglement-in-task-arithmetic-fig1.webp"
excerpt: "Task arithmetic merges fine-tuned models by adding their weight deltas. This CVPR 2026 oral explains when that works (disjoint task features), why weight orthogonality is the visible fingerprint, and how a one-line regularizer, OrthoReg, makes task vectors merge better."
paper:
  title: "Understanding and Enforcing Weight Disentanglement in Task Arithmetic"
  authors: "Shangge Liu, Yuehan Yin, Lei Wang, Qi Fan, Yinghuan Shi, Wenbin Li, Yang Gao, Dacheng Tao"
  venue: "CVPR 2026 (oral)"
  url: "https://arxiv.org/abs/2604.17078"
---

> [!tldr]
> **What it claims.** Task arithmetic (adding fine-tuned weight deltas to a pretrained model) works when the model routes different tasks through *different internal features*. The authors call this Task-Feature Specialization (TFS), prove it is sufficient for weight disentanglement, and show it leaves a measurable fingerprint: weight columns that are nearly orthogonal.
>
> **What it builds.** OrthoReg, a regularizer that pushes each layer's weight update $\Delta W$ toward $\Delta W^\top \Delta W = I$ during fine-tuning. It plugs into plain fine-tuning, tangent-space fine-tuning, attention-only fine-tuning and LoRA.
>
> **Most surprising number.** On CLIP ViT-L-14, merging eight OrthoReg task vectors reaches 100.08% normalized accuracy, up from 89.19%, at about 12% extra training time versus 78% extra for tangent-space fine-tuning.
>
> **Where it is thin.** The "angle control" proof leans on a random-matrix argument that, by my own check, gives no more orthogonality than any two random vectors of the same size. And the normalized-accuracy headline partly reflects weaker single-task specialists, not only a better merge.

Task arithmetic is the strangest trick in model editing that actually works. Fine-tune a CLIP image encoder on cars, subtract the pretrained weights, and call the difference a "task vector". Do the same for seven more datasets, add all eight vectors back, and one model handles all eight. Subtract a vector instead, and the model forgets that task.

Nobody fully agreed on *why*. A change made for cars should leak into digits. This CVPR 2026 oral gives a causal story and turns it into a regularizer. It sits in the same wave as our read of [task transfer in vision-language models](/blog/paper-reading/computer-vision/understanding-task-transfer-in-vision-language-models), which asks the complementary question of which tasks help each other.

![Figure 1 from Liu et al. (2026): task-feature specialization as the common cause of weight vector orthogonality and weight disentanglement](/imgs/blogs/understanding-and-enforcing-weight-disentanglement-in-task-arithmetic-fig1.webp)

The figure is the whole thesis. Features colored by task (top) produce mutually orthogonal weight columns (left) and a merged model that matches each task's own fine-tune on that task's data (right). TFS is the common cause; orthogonality and disentanglement are its two effects.

## The setup: task vectors and the wish list

Start with notation, because everything later is a single inner product. A network $f(x;\theta)$ has pretrained weights $\theta_0 \in \mathbb{R}^P$. Fine-tuning on task $t$ gives $\theta_t^*$, and the task vector is

$$
\tau_t = \theta_t^* - \theta_0, \qquad \theta_{\text{MT}} = \theta_0 + \sum_{t=1}^{T} \alpha_t \tau_t .
$$

Here $\alpha_t$ is a scalar weight per task and $\theta_{\text{MT}}$ is the merged multi-task model. In every experiment the authors use a single shared $\alpha$ picked by grid search over $\{0, 0.05, \dots, 1.0\}$, so the comparison measures the vectors themselves rather than a clever merge rule.

The wish list is called **weight disentanglement**, a definition inherited from tangent task arithmetic (TTA, Ortiz-Jimenez et al., 2023). A model is disentangled if, for any coefficients, the merged model behaves like task $i$'s own fine-tune on task $i$'s data domain $\mathcal{D}_i$, and like the pretrained model outside every domain. It is a statement about outcomes. It says nothing about which property of $\theta_0$ or of the $\tau_t$ makes the outcome happen, which is the gap this paper targets.

### Making disentanglement a single equation

The paper reduces the wish list to two tasks, $t$ and $j$, and asks for $f(x;\theta_0+\tau_t+\tau_j) = f(x;\theta_0+\tau_t)$ on $x \in \mathcal{D}_t$. Adding task $j$'s vector should not change task $t$'s answers.

Now linearize. Under the neural tangent kernel (NTK) approximation, a small parameter change acts on the output through the Jacobian $J(x) = \nabla_\theta f(x;\theta_0)$:

$$
f(x;\theta_0+\tau) \approx f(x;\theta_0) + \tau^\top J(x).
$$

Plug both sides of the two-task condition into this, and everything cancels except one term. Lemma 1 states that disentanglement between $t$ and $j$ is equivalent to

$$
\tau_j^\top J(x) = 0 \quad \text{for all } x \in \mathcal{D}_t .
$$

Call $\tau_j^\top J(x)$ the **interference**. It is the dot product between task $j$'s weight change and the direction in weight space that task $t$'s inputs are sensitive to. Every argument in the paper is about forcing that number toward zero.

## Technique 1: task-feature specialization

**The problem.** We need a property of the network under which interference vanishes, ideally one that sounds like something a real model could have.

**The analogy.** Think of a kitchen with eight cooks sharing one long row of burners. If the pastry chef only ever uses burners 1 to 3 and the grill cook only ever uses burners 5 to 7, then turning up burner 6 cannot burn the croissants. Interference is zero not because anyone is careful, but because nobody touches the same knob.

**The mechanism.** Take one linear layer with weight matrix $W = [w_1, \dots, w_d]$. Each column $w_k$ extracts a feature with activation $z_k$. Task $t$'s **specialized feature set** $I_t$ is the set of columns the output is sensitive to on task $t$'s data: for $k \notin I_t$, the paper requires $\mathbb{E}_{x\sim\mathcal{D}_t}\left[\,|\partial f / \partial z_k|\,\right] = 0$. TFS is the assumption that two tasks' sets do not overlap, $I_t \cap I_j = \varnothing$.

**The math.** Split the interference column by column:

$$
\tau_j^\top J(x) = \sum_{k=1}^{d} \left\langle (\tau_j)_k,\; \nabla_{w_k} f(x;\theta_0) \right\rangle ,
$$

where $(\tau_j)_k$ is the update task $j$ made to column $k$. Two facts kill every term. If $k \notin I_t$, the chain rule through $z_k$ makes the gradient $\nabla_{w_k} f$ on task $t$'s inputs approximately zero. If $k \notin I_j$, task $j$'s loss never pushed on column $k$, so $(\tau_j)_k \approx 0$. Under TFS, every column falls in at least one of those cases, so the sum is zero. That is Theorem 1: TFS plus the NTK approximation is sufficient for weight disentanglement.

![Worked example of the Theorem 1 case split: task t's gradient and task j's update never land on the same column, so their product is zero everywhere](/imgs/blogs/understanding-and-enforcing-weight-disentanglement-in-task-arithmetic-1.webp)

**A tiny example.** The figure uses eight columns. Task $t$ owns columns 1 to 3, so its gradient row reads (0.8, -0.5, 0.6, 0, 0, 0, 0, 0). Task $j$ owns columns 5 to 7, so its update row reads (0, 0, 0, 0, 0.4, -0.7, 0.3, 0). Multiply column by column and every entry is zero. Now let column 4 be shared, with gradient 0.6 and update 0.5: the product is 0.30, and the merged model leaks task $j$ into task $t$.

**When it fails.** That last case is the realistic one: MNIST and SVHN digits share almost every feature. The paper says plainly that TFS "rarely holds" in practice, so the rest of it is about shared burners.

## Technique 2: orthogonality as the visible fingerprint

TFS is defined through gradients of the output, which you cannot read off a checkpoint. The paper's second move is to find something you *can* measure.

**The mechanism.** Corollary 1 says TFS forces **block orthogonality**: any column used by task $t$ is orthogonal to any column used by task $j$. The appendix argument goes through feature covariance. With layer or batch normalization, a layer's input has roughly zero mean, so the covariance of $z = W^\top \mathrm{In}(x)$ is $\Sigma_z = W^\top\, \mathbb{E}[\mathrm{In}(x)\mathrm{In}(x)^\top]\, W$. TFS makes $\Sigma_z$ block diagonal, since features owned by different tasks are decorrelated, and that structure carries back to the Gram matrix $W^\top W$ under the paper's assumptions about the input covariance.

**The evidence.** In a pretrained CLIP ViT-B/16, angles between *all* column pairs, not only cross-task pairs, pile up at 90 degrees.

![Figure 2 from Liu et al. (2026): angles between weight column pairs in a pretrained CLIP ViT-B/16 cluster tightly at 90 degrees](/imgs/blogs/understanding-and-enforcing-weight-disentanglement-in-task-arithmetic-fig2.webp)

The left panel is one MLP weight matrix with a mean of about 90 degrees and a standard deviation under 2 degrees. The right panel summarizes deviations from 90 degrees across all transformer linear layers: 95.27% of column pairs sit within 5 degrees of orthogonal.

**How to read it.** Orthogonality is not claimed to *cause* disentanglement. It shares a cause with it, which makes it a *clue*; an appendix argues in Bayesian terms that seeing it raises our belief in TFS-like structure.

**When it fails.** High-dimensional columns are close to orthogonal for boring reasons too. Two random vectors in 768 dimensions have a cosine with standard deviation about ${1}/\sqrt{768} \approx 0.036$, which is about 2 degrees. So a tight peak at 90 degrees is partly what any wide random matrix looks like. Whether pretrained CLIP is *more* orthogonal than a random baseline of the same shape is the comparison I most wanted and did not find in the paper.

## Technique 3: OrthoReg, the regularizer

**The problem.** You cannot enforce TFS directly, because you do not know the feature sets in advance. You also cannot enforce block orthogonality, because you do not know where the block boundaries are.

**The move.** Enforce a stronger condition with no blocks at all: make each layer's *update* $\Delta W$ column-wise orthonormal. The fine-tuning loss becomes

$$
\mathcal{L} = \mathcal{L}_{\text{task}}(\theta_0+\Delta\theta) + \lambda\, \mathcal{L}_{\text{ortho}}(\Delta\theta),
\qquad
\mathcal{L}_{\text{ortho}}(\Delta\theta) = \sum_{l} \left\| (\Delta W^{(l)})^\top \Delta W^{(l)} - I \right\|_F^2 .
$$

Here $l$ runs over the tuned linear layers, $\Delta W^{(l)} = W^{(l)} - W_0^{(l)}$ is that layer's update, $I$ is the identity, $\|\cdot\|_F$ is the Frobenius norm, and $\lambda$ sets the strength (validated in [0.1, 100]). The Gram matrix $\Delta W^\top \Delta W$ holds every pairwise dot product of update columns. The penalty pushes its off-diagonal entries to zero and its diagonal to one.

![Figure 3 from Liu et al. (2026): OrthoReg adds an orthogonality penalty on each layer's weight update inside the transformer block](/imgs/blogs/understanding-and-enforcing-weight-disentanglement-in-task-arithmetic-fig3.webp)

**A worked example.** Take a 2 by 2 update with columns $c_1 = (1, 0)$ and $c_2 = (0.5, 1)$. The Gram matrix is [[1, 0.5], [0.5, 1.25]], so $G - I$ is [[0, 0.5], [0.5, 0.25]] and the penalty is ${0.25} + 0.25 + 0.0625 = 0.5625$. The gradient asks $c_2$ to drop its component along $c_1$ and shrink toward unit length. That is all the regularizer does, layer by layer.

**What the code does.** The released code ([RL-MIND/OrthoReg](https://github.com/RL-MIND/OrthoReg)) sums the *unsquared* Frobenius norm, builds the Gram on the smaller side of non-square matrices, and penalizes every trainable 2D parameter. Its README example uses $\lambda = 10$ with AdamW at a learning rate of 1e-5.

### The part the paper does not discuss: it starts as anti-weight-decay

Expand the penalty and something odd falls out. With $G = \Delta W^\top \Delta W$,

$$
\left\| G - I \right\|_F^2 = \operatorname{tr}(G^2) - 2\,\operatorname{tr}(G) + d,
\qquad
\nabla_{\Delta W}\, \mathcal{L}_{\text{ortho}} = 4\,\Delta W\,(G - I).
$$

Fine-tuning starts at $\Delta W = 0$, where this gradient is exactly zero. For small updates, $\operatorname{tr}(G) = \|\Delta W\|_F^2$ is second order while the off-diagonal terms inside $\operatorname{tr}(G^2)$ are fourth order. So the leading effect of the penalty is a gradient of about $-4\,\Delta W$: descent *grows* the update along whatever direction it already points. This is my derivation, not the paper's.

![The per-column OrthoReg penalty plotted against column norm: flat at zero, pushing small updates outward toward a unit-norm minimum, the opposite of weight decay](/imgs/blogs/understanding-and-enforcing-weight-disentanglement-in-task-arithmetic-3.webp)

The plot shows one column scaled by $s$: the penalty is $(s^2-1)^2$, flat at $s=0$, minimized at $s=1$. Weight decay (the dashed curve) is minimized at zero. At a learning rate of 1e-5 with Adam, which moves each weight by at most roughly the learning rate per step, update columns probably never get near unit norm in a short fine-tune. If that estimate is right, OrthoReg in practice is a mix of "grow the update" and "decorrelate its columns", and the decorrelation only gets traction as the update grows. The shared merge coefficient $\alpha$ then rescales whatever size comes out.

## Why it should work: norm control and angle control

Theorem 2 claims OrthoReg promotes disentanglement even when features overlap. The proof sketch has one approximation and two controls.

**Step 1, alignment.** For a typical input from task $t$, the Jacobian $J(x)$ points roughly along $\tau_t$, because $\tau_t$ was built by accumulating those very gradients. This turns the interference into

$$
\left| \tau_j^\top J(x) \right| \approx \|\tau_j\|_2 \cdot \|J(x)\|_2 \cdot \left|\cos\angle(\tau_j,\tau_t)\right| .
$$

![Interference as a projection: when task j's vector is at 50 degrees to task t's direction its shadow is large; at 87 degrees it nearly vanishes](/imgs/blogs/understanding-and-enforcing-weight-disentanglement-in-task-arithmetic-2.webp)

The figure reads this geometrically. Interference is the shadow task $j$'s vector casts on task $t$'s Jacobian. At 50 degrees, $|\cos|$ is 0.64 and the shadow is long. At 87 degrees it is about 0.05.

**Step 2, norm control.** If $\|\Delta W^\top \Delta W - I\|_F^2 \le \xi$, then $\|\Delta W\|_F^2 \le d + \sqrt{d\xi}$, so the task vector's norm is bounded. Note this is an *upper* bound near the orthonormal target, consistent with the anti-weight-decay reading above: the regularizer caps the size at roughly unit-norm columns, it does not keep updates small.

**Step 3, angle control.** Write each near-orthonormal update by polar decomposition as $\Delta W = QP$, with $Q$ having orthonormal columns and $P$ close to the identity. Then the inner product of two task vectors is dominated by $\sum_l \langle \operatorname{vec}(Q_t^{(l)}), \operatorname{vec}(Q_j^{(l)})\rangle$. Lemma 3 shows that two matrices drawn independently and uniformly from the set of orthonormal-column matrices (the Stiefel manifold) have an inner product with mean zero, and argues that its distribution is "sharply concentrated", more so than for unconstrained random vectors.

**My check on Step 3.** I computed the second moment. For $A, B$ uniform $m \times d$ orthonormal-column matrices, $\mathbb{E}[\operatorname{tr}(A^\top B)^2] = d/m$, so the cosine between the vectorized matrices has standard deviation ${1}/\sqrt{md}$. That is *exactly* the spread for two random unit vectors in $\mathbb{R}^{md}$. A quick simulation with $m = 256$, $d = 64$ and 2,000 draws gave 0.0077 for Stiefel pairs, 0.0080 for Gaussian pairs, and ${1}/\sqrt{md} = 0.0078$. The concentration is just high dimension at work, not an extra effect of the orthogonality constraint.

That matters because real fine-tuned task vectors are not independent random draws. The baseline heatmaps in the paper show cosines well above random scale between distinct tasks, because tasks share low-rank directions. So the real question is not "are random orthonormal matrices orthogonal" but "why does OrthoReg remove the *shared* directions". My guess, which is yours to test: forcing every update toward a full-rank, isotropic spectrum stops each task from concentrating its change in the few top directions that all tasks agree on.

## The bridge to tangent task arithmetic

TTA fine-tunes the linearized model directly, solving a ridge-like problem in $\tau$. By the representer theorem, its solution is a combination of training Jacobians, $\tau_t^* = \sum_i \alpha_i J(x_i)$. The inner product of two TTA task vectors is then a double sum of NTK values $\langle J(x_i), J(x_k)\rangle$ across tasks. TTA's own finding is that CLIP's NTK is *localized*: kernel values between inputs from different task domains are near zero. So TTA vectors come out nearly orthogonal for free.

So both methods win through the same door, inter-task orthogonality: TTA implicitly and expensively, OrthoReg explicitly. The appendix cost table (Cars, ViT-L-14):

| Method | Total params (M) | Training time (min) | Peak GPU memory (MB) | Merged Abs. Acc. |
| --- | --- | --- | --- | --- |
| Non-lin. FT | 342.56 | 158.21 | 42,589 | 84.07 |
| TTA (linearized) | 685.12 | 280.86 | 68,031 | 86.19 |
| Non-lin. FT + OrthoReg | 342.56 | 177.04 | 44,500 | 88.23 |
| ATT-FT | 342.56 | 126.28 | 36,591 | 87.81 |
| ATT-FT + OrthoReg | 342.56 | 132.96 | 36,977 | 90.41 |

OrthoReg adds 11.9% training time to plain fine-tuning (${177.04} / 158.21 = 1.119$); TTA adds 77.5% (${280.86} / 158.21 = 1.775$) and about 60% more memory. And OrthoReg on plain fine-tuning beats TTA's merged accuracy by two points.

## Results

The benchmark is standard: eight classification datasets (Cars, DTD, EuroSAT, GTSRB, MNIST, RESISC45, SUN397, SVHN) on CLIP ViT-B-32, B-16 and L-14 image encoders, text encoder frozen.

![Table 1 from Liu et al. (2026): task addition of eight task vectors on three CLIP backbones, every baseline with and without OrthoReg](/imgs/blogs/understanding-and-enforcing-weight-disentanglement-in-task-arithmetic-fig4.webp)

The pattern is uniform: all twelve baseline and backbone pairs improve in absolute accuracy. The biggest lifts land on the weakest baseline, plain fine-tuning (+3.09, +2.29 and +4.16 points on the three backbones). TTA gains least (+1.67, +0.88, +1.33), which fits the story that TTA vectors were already near-orthogonal. ATT-FT plus OrthoReg is the best row on every backbone, reaching 90.41% on ViT-L-14.

**Task negation** (Table 2) asks the reverse: subtract a task vector so the model forgets that task while keeping at least 95% of its zero-shot ImageNet accuracy. Lower target accuracy is better. On ViT-L-14, ATT-FT's target accuracy drops from 24.85% to 14.67% with OrthoReg (-10.18) while ImageNet moves from 76.42% to 75.40%. TTA on ViT-L-14 does not move at all (8.36% both ways), again the method that was already orthogonal.

**The mechanism check.** Figure 5 plots the pairwise cosine between the eight task vectors for plain fine-tuning on ViT-B-16.

![Figure 5 from Liu et al. (2026): cosine similarity between task vectors, without and with OrthoReg, on ViT-B-16](/imgs/blogs/understanding-and-enforcing-weight-disentanglement-in-task-arithmetic-fig5.webp)

Without OrthoReg, several off-diagonal pairs light up; with it, the off-diagonal is nearly uniformly dark. This is the paper's strongest evidence, because it measures the quantity the theory says matters.

**LoRA ablations** (Table 6) show gains on every attention subset, up to +4.17 on ViT-B-16, and a slight loss (-0.27) for MLP-only LoRA on ViT-B-32.

### What the normalized accuracy hides

Normalized accuracy is merged accuracy divided by each task's own single-task fine-tune, averaged over tasks. The paper reads 100.08% on ViT-L-14 as "the functional realization of ideal weight disentanglement". There is a second way to get there: weaker specialists.

Dividing absolute by normalized accuracy gives a rough implied specialist accuracy. It is approximate, since an average of ratios is not a ratio of averages, and the arithmetic is mine. On ViT-L-14, plain fine-tuning's implied specialists drop from about 94.3% (${84.07} / 0.8919$) to 88.2% (${88.23} / 1.0008$) with OrthoReg.

![Merged accuracy rises while implied specialist accuracy falls with OrthoReg on ViT-L-14, closing the gap between them](/imgs/blogs/understanding-and-enforcing-weight-disentanglement-in-task-arithmetic-4.webp)

The chart shows the same pattern for ATT-FT (93.8% to 90.4%) and a smaller one for TTA (92.5% to 90.8%), while LoRA-ATT's specialists hold steady (93.2% to 93.4%). So for full fine-tuning, OrthoReg buys mergeability partly by giving up a few points of single-task fit. The merged model still wins on absolute accuracy, which is what you deploy, so the trade is good. But "near-total absence of interference" overstates it. One more oddity: the ViT-B-16 plain fine-tuning baseline lists 75.39 for both absolute and normalized accuracy, which looks like a transcription error in Table 1.

**What may not transfer.** Everything is CLIP ViT image encoders on eight classification datasets, with a frozen text head that already gives each task a fixed classifier. Generative models, LLMs and tasks with heavy feature overlap are untested.

## Critique

**What is strong.**
- The causal framing is clean, and the Theorem 1 case split is short enough to check by hand.
- One line of loss, four fine-tuning schemes, twelve out of twelve absolute-accuracy gains.
- The cosine heatmaps test the mechanism directly, and the TTA cost comparison is concrete.

**What is weak.**
- **The angle-control lemma proves too little.** As computed above, uniformly random orthonormal matrices are exactly as orthogonal as uniformly random vectors. The proof needs task updates to be *independent* draws, which is precisely what fails when tasks share features.
- **The regularizer's target is odd.** Pulling $\Delta W^\top\Delta W$ to $I$ means unit-norm update columns, a large change relative to typical fine-tuning, and at small updates the penalty acts as negative weight decay. No ablation separates the "decorrelate" part from the "grow" part, for instance an off-diagonal-only penalty.
- **LoRA is underspecified.** A rank-$r$ update cannot satisfy $\Delta W^\top\Delta W = I$ when $r$ is smaller than the matrix side. The paper reports LoRA results but does not say how the penalty is adapted, and the public code has no LoRA path.
- **Theorem numbering drifts** (main-text Theorems 1 and 2 are 3 and 4 in the appendix).
- **Random-matrix baseline for Figure 2 is missing**, so we cannot tell how much of CLIP's 90-degree peak is learned structure.

**What would change my mind.** If an off-diagonal-only version of the penalty (no pull toward unit norm) kept most of the Table 1 gains, I would accept that orthogonality, not update size, is the active ingredient. If it lost them, the story is mostly about scale and $\alpha$ tuning. Separately, if task-feature overlap measured from gradients failed to predict which pairs interfere on the eight-task benchmark, I would treat TFS as a nice sufficient condition rather than the explanation.

## Where this goes next

These are my proposals, not the paper's.

1. **TFS as a merge-compatibility predictor.** *Question:* can you tell, before merging, which task pairs will interfere? *Why now:* the paper gives an operational definition, the set of columns with non-negligible $\mathbb{E}|\partial f/\partial z_k|$ on each task's data. *First experiment:* for the 28 pairs among the eight CLIP tasks, compute the Jaccard overlap of each pair's top-sensitivity column sets per layer, then correlate it with the pairwise accuracy drop when merging two vectors. A strong rank correlation turns TFS from a theorem into a tool for picking what to merge.

2. **OrthoReg for LLM model merging.** *Question:* does explicit update orthogonality help merge instruction, math and code fine-tunes of a language model, where task overlap is far heavier than between Cars and MNIST? *Why now:* the regularizer is model-agnostic and cheap. *First experiment:* fine-tune a 1B to 3B model on three skill datasets with and without OrthoReg, merge with task arithmetic and TIES, and measure each skill plus task-vector cosines. Watch whether the anti-weight-decay effect destabilizes training.

3. **A rank-aware OrthoReg for LoRA.** *Question:* what is the right target for a rank-$r$ update $BA$? *Why now:* the paper shows LoRA gains but leaves the formulation open. *First experiment:* penalize $\|B^\top B - I_r\|_F^2 + \|AA^\top - I_r\|_F^2$, which costs only $r \times r$ Gram matrices, against a direct cross-task penalty $\langle B_tA_t, B_jA_j\rangle$ computed with traces of small matrices. Compare merged accuracy and cosine heatmaps on the same eight tasks.

4. **Separate angle from norm.** *Question:* which half of the "dual control" does the work? *Why now:* the paper's own expansion makes the two pieces explicit. *First experiment:* three variants, full OrthoReg, off-diagonal-only penalty, and a target of $cI$ for a few values of $c$, on ViT-B-32 with every other setting held fixed. This directly answers the "what would change my mind" test above.

5. **Pretraining orthogonality as a foundation-model selection score.** *Question:* does the column-angle distribution of a pretrained backbone predict how well it supports task arithmetic? *Why now:* the paper argues orthogonality is evidence of TFS in $\theta_0$. *First experiment:* compute the deviation-from-90-degrees histogram, relative to a shape-matched random matrix, for CLIP, SigLIP and DINOv2 backbones of similar size, and correlate it with eight-task merge accuracy under one fixed fine-tuning recipe.

If you tune adapters you plan to merge, see our guide to [debugging LoRA and PEFT runs](/blog/machine-learning/debugging-training/debugging-lora-and-peft).

## References

- Shangge Liu, Yuehan Yin, Lei Wang, Qi Fan, Yinghuan Shi, Wenbin Li, Yang Gao, Dacheng Tao. *Understanding and Enforcing Weight Disentanglement in Task Arithmetic.* CVPR 2026 (oral). [arXiv:2604.17078](https://arxiv.org/abs/2604.17078). CVPR page: [cvpr.thecvf.com/virtual/2026/oral/40387](https://cvpr.thecvf.com/virtual/2026/oral/40387).
- Code: [github.com/RL-MIND/OrthoReg](https://github.com/RL-MIND/OrthoReg).
- Gabriel Ilharco et al. *Editing Models with Task Arithmetic.* ICLR 2023.
- Guillermo Ortiz-Jimenez, Alessandro Favero, Pascal Frossard. *Task Arithmetic in the Tangent Space: Improved Editing of Pre-trained Models.* NeurIPS 2023.
- Ruochen Jin et al. *Fine-tuning Attention Modules Only: Enhancing Weight Disentanglement in Task Arithmetic.* ICLR 2025.
- Related on this site: [Understanding task transfer in vision-language models](/blog/paper-reading/computer-vision/understanding-task-transfer-in-vision-language-models), [Debugging LoRA and PEFT](/blog/machine-learning/debugging-training/debugging-lora-and-peft).
