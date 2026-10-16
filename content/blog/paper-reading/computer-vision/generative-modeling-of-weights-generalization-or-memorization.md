---
title: "Generative Modeling of Weights: Generalization or Memorization?"
publishDate: "2026-10-16"
date: "2026-10-16"
description: "Four well-known weight generators mostly hand back copies of their training checkpoints, and a pinch of Gaussian noise does as well. Here is how the paper proves it, and what a generator that truly generalizes would need."
category: "paper-reading"
subcategory: "Computer Vision"
tags: ["interpretability-analysis", "cvpr-2026", "cvpr-2026-highlight", "paper-reading", "weight-generation", "memorization", "diffusion-models", "hypernetworks", "weight-space-symmetry"]
author: "Hiep Tran"
featured: true
readTime: 18
image: "/imgs/blogs/generative-modeling-of-weights-generalization-or-memorization-1.webp"
excerpt: "Diffusion models that 'generate' neural network weights mostly reproduce the checkpoints they were trained on. A nearest-neighbor test, a behavioral overlap metric, and two embarrassingly simple baselines show why, and point at what a real weight generator needs."
paper:
  title: "Generative Modeling of Weights: Generalization or Memorization?"
  authors: "Boya Zeng, Yida Yin, Zhiqiu Xu, Zhuang Liu"
  venue: "CVPR 2026 (Highlight); arXiv 2025"
  url: "https://arxiv.org/abs/2506.07998"
---

> [!tldr]
> **The claim.** Generative models trained on collections of network checkpoints (Hyper-Representations, G.pt, HyperDiffusion, p-diff) are said to synthesize new, high-performing weights. Zeng, Yin, Xu and Liu test whether those weights are actually *new*.
>
> **Why it matters.** If a weight generator only copies its training set, it is an expensive lookup table. Every downstream claim (fast adaptation, model zoos on demand, weight-space priors) rests on generalization.
>
> **The surprising result.** For three of the four methods, generated checkpoints sit far closer to some training checkpoint than any two training checkpoints sit to each other. For HyperDiffusion, 94.4% of generated MLPs are within L2 distance 10 of a training MLP, while every training pair is more than 50 apart. Adding Gaussian noise to training weights matches or beats them on the accuracy-novelty trade-off. The fourth, p-diff, beats noise only because it averages checkpoints from one training run.
>
> **Where it fails.** The study covers four methods in their primary setups, mostly small networks. It shows memorization convincingly; it does not prove no weight generator can generalize.

## The question nobody had asked

Image diffusion models were audited for memorization early. Weight generation borrowed the same machinery (autoencoders, latent diffusion, transformers over flattened parameters) but kept a weaker yardstick: *does the generated network score well on its task?*

That yardstick cannot tell a generator from a photocopier. A copy of a good checkpoint also scores well. So the paper asks the question that should have come first: are generated weights meaningfully different from the training checkpoints, while still performing well?

The authors pick four influential methods and evaluate each under its own primary setup.

| Method | Generator | Training data (checkpoints) | Original task |
| --- | --- | --- | --- |
| Hyper-Representations | autoencoder + KDE in latent space | 2,896 checkpoints, one architecture | SVHN classification |
| G.pt | conditional diffusion (input weights + target loss) | 2.1M checkpoints from 10,728 runs | MNIST classification |
| HyperDiffusion | unconditional diffusion, 1.4B parameters | 2,749 neural-field MLPs, one per shape | ShapeNet airplanes (3D) |
| p-diff | unconditional latent diffusion | 300 checkpoints from one extra epoch of a converged model | CIFAR-100 classification |

Two details in that table will matter later. HyperDiffusion has 1.4B parameters to model 2,749 training examples. And p-diff's 300 checkpoints all come from consecutive steps of a single run.

## Test 1: find the nearest training checkpoint

**The problem.** A generated checkpoint is a vector $\mathbf{w}_g \in \mathbb{R}^d$, where $d$ is the number of parameters in the target network. We want to know whether it is a near-copy of something in the training set.

**The analogy.** Plagiarism detection. You do not grade the essay; you search the library for the closest existing essay and measure how close it is. Then you calibrate that number against how close two honestly written essays usually are.

**The mechanism.** For each generated checkpoint, find the training checkpoint at minimum L2 distance. Do the same for each training checkpoint against the rest of the training set, excluding itself. Compare the two distributions.

**The math.** Let $\mathcal{T} = \{\mathbf{w}_1, \dots, \mathbf{w}_n\}$ be the $n$ training checkpoints, each in $\mathbb{R}^d$. The novelty score of a generated checkpoint is

$$
d_{\min}(\mathbf{w}_g) = \min_{\mathbf{w} \in \mathcal{T}} \; \min_{\pi \in \Pi} \; \lVert \mathbf{w}_g - \pi(\mathbf{w}) \rVert_2 ,
$$

where $\Pi$ is the set of hidden-neuron permutations the method was trained to respect, and $\pi(\mathbf{w})$ is $\mathbf{w}$ with its neurons reordered. For methods that do not augment with permutations, $\Pi$ contains only the identity. The baseline is the same quantity for training checkpoints, $d_{\min}(\mathbf{w}_i)$ computed over $\mathcal{T} \setminus \{\mathbf{w}_i\}$.

The permutation term is there for fairness. G.pt trains with permutation augmentation, so it could legitimately output a reordered copy; the authors enumerate all permutations to catch that. Hyper-Representations gets a different correction: its autoencoder cannot reconstruct training weights faithfully (reconstruction degrades SVHN accuracy), so generated weights are compared against *reconstructed* training weights instead.

**A tiny example.** Take three training checkpoints in a 2-parameter world: $(0, 0)$, $(6, 0)$, $(0, 8)$. Nearest-neighbor distances among them are 6, 6 and 8. Now a generator outputs $(0.3, 0.4)$. Its nearest training point is $(0, 0)$ at distance 0.5. A typical training point has a neighbor 6 to 8 away; the generated point has one 0.5 away. That ratio, not the absolute number, is the signal.

**When it fails.** L2 distance in raw weight space ignores symmetries the method did not train on (scaling, sign flips), so a generator could in principle produce a functionally new network that *looks* close, or a functionally identical one that looks far. That is why the paper never stops at weight distance: Test 2 checks behavior.

### What the heatmaps show

The first evidence is visual. For three random generated checkpoints per method, the paper plots 64 random parameter values as a row, then stacks the three nearest training checkpoints underneath.

![Figure 2 from Zeng et al. (2025): weight heatmaps for one generated checkpoint (top row, outlined) and its three nearest training checkpoints, for each of the four methods](/imgs/blogs/generative-modeling-of-weights-generalization-or-memorization-fig1.webp)

In every panel, at least one training row is nearly indistinguishable from the generated row. For p-diff, *all* rows look alike, which is expected because its training checkpoints differ by only a few gradient steps.

The geometry is easiest to see as a cloud of points. Training checkpoints from independent runs are spread far apart. Generated checkpoints do not fill the space between them; they huddle on top of individual training points.

![Schematic of weight space: blue training checkpoints are spread far apart (any pair more than 50 apart in L2), red generated checkpoints cluster tightly on top of individual training checkpoints (94.4% within L2 10), and a green from-scratch model sits alone in open space](/imgs/blogs/generative-modeling-of-weights-generalization-or-memorization-1.webp)

The histograms make it quantitative. For Hyper-Representations, G.pt and HyperDiffusion, the red distribution (generated to nearest training) sits well left of the blue one (training to nearest other training).

![Figure 4 from Zeng et al. (2025): histograms of minimum L2 distance to the training set, generated checkpoints versus training checkpoints, for each method](/imgs/blogs/generative-modeling-of-weights-generalization-or-memorization-fig2.webp)

The HyperDiffusion numbers are the starkest: 94.4% of generated checkpoints are within L2 distance 10 of some training checkpoint, while every pair of training checkpoints is more than 50 apart. A new model trained from scratch would land in open space. A generated one lands on a neighbor.

p-diff is the exception in this plot, and the exception is suspicious rather than reassuring. Its training checkpoints are extremely close to one another (they come from one run), so even an interpolated output looks "far" by comparison. We come back to it below.

## Test 2: do they behave the same?

**The problem.** Small weight distance does not guarantee identical function. Networks are nonlinear; a nudge in the wrong direction can flip predictions. So the authors check behavior directly.

**Mechanism, qualitative.** For classifiers, project the test set onto two principal components and draw each model's decision regions. For HyperDiffusion's neural fields, reconstruct the 3D shape each MLP encodes. Then put each generated model next to its nearest training model.

![Figure 5 from Zeng et al. (2025): decision boundaries of generated models next to their nearest training models, and HyperDiffusion airplane shapes generated versus nearest training shape](/imgs/blogs/generative-modeling-of-weights-generalization-or-memorization-fig3.webp)

The decision regions are nearly identical across all classes, and the airplanes differ only in minor details such as wing edges.

### The novelty metric: IoU of mistakes

**The analogy.** Two students who got the same score are not necessarily the same student. Two students who got the *same questions wrong* probably are, or one copied the other.

**The math.** Let $\{(X_i, y_i)\}_{i=1}^{N}$ be the test set. For a model $M$, collect the indices it gets wrong. For two models $M_1, M_2$:

$$
I_k = \{\, i \in \{1,\dots,N\} \mid M_k(X_i) \neq y_i \,\}, \qquad
\mathrm{IoU}(M_1, M_2) = \frac{\lvert I_1 \cap I_2 \rvert}{\lvert I_1 \cup I_2 \rvert}.
$$

Here $I_k$ is the error set of model $k$, $\lvert \cdot \rvert$ counts elements, and IoU ranges from 0 (completely different mistakes) to 1 (identical mistakes). The metric comes from the p-diff paper itself, which makes using it against p-diff fair. A generated model's novelty is judged by its **maximum** similarity over all training models:

$$
s_{\max}(M_g) = \max_{M \in \mathcal{T}} \mathrm{IoU}(M_g, M).
$$

Lower $s_{\max}$ means more novel.

**A tiny example.** On 20 test images, model A is wrong on images 2, 5, 7, 11, 14, 18 and model B on 2, 5, 7, 11, 16, 18. They share 5 mistakes; together they make 7 distinct mistakes. IoU is ${5/7 \approx 0.71}$.

![Three 4x5 grids of test images: model A's mistakes in red, model B's mistakes in orange, and the overlay showing 5 shared mistakes and 2 unshared, giving IoU = 5/7 = 0.71](/imgs/blogs/generative-modeling-of-weights-generalization-or-memorization-2.webp)

**When it fails.** For HyperDiffusion there is no error set, so the paper swaps in Chamfer Distance (CD) between reconstructed shapes. Minimum CD to *test* shapes measures quality; minimum CD to *training* shapes measures novelty. The appendix also reports raw prediction overlap, with a useful caveat: two models with accuracy $x$ must agree on at least $\max(2x - 1, 0)$ of the test set, so overlap is inflated for accurate models. IoU of errors avoids that floor.

## Test 3: can a generator beat a pinch of noise?

This is the paper's sharpest move. Being close to training data is not damning by itself if the generator's outputs are *better* than anything cheaper. So the authors build the cheapest possible generator.

**The mechanism.** Take a training checkpoint, add isotropic Gaussian noise:

$$
\mathbf{w}_{\text{noise}} = \mathbf{w} + \sigma \, \boldsymbol{\varepsilon}, \qquad \boldsymbol{\varepsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}_d),
$$

where $\mathbf{w}$ is a training checkpoint sampled uniformly (for Hyper-Representations, a reconstruction of one of its top-30% checkpoints, matching how its KDE sampler works), $\sigma$ is a scalar amplitude, and $\boldsymbol{\varepsilon}$ is a standard normal vector in $d$ dimensions. The authors pick $\sigma$ so the noise-added models' maximum similarity roughly matches the generated models'. Then they compare accuracy at matched novelty.

**A tiny example.** With $\mathbf{w} = (0.50, -0.20, 1.10)$, $\sigma = 0.02$ and $\boldsymbol{\varepsilon} = (1.0, -0.5, 0.3)$, the result is $(0.52, -0.21, 1.106)$: novel only in the weak sense a memorizing generator is.

![Figure 6 from Zeng et al. (2025): accuracy versus maximum similarity for training, generated, and noise-added models; HyperDiffusion panel uses minimum Chamfer distance to training and test shapes](/imgs/blogs/generative-modeling-of-weights-generalization-or-memorization-fig4.webp)

**The result.** For G.pt and Hyper-Representations, noise-added models reach comparable or even higher accuracy than generated models at the same maximum similarity. For HyperDiffusion, the CD distributions to training and test shapes show no significant difference between generated and noise-added MLPs. A diffusion model with 1.4B parameters, trained for this purpose, is matched by `w + 0.02 * randn_like(w)`.

**When the test is unfair.** Noise is a weak baseline in one direction: it can only explore a small ball around each checkpoint. A generator that produced genuinely novel, high-accuracy models far from every training point would win this comparison easily. None did.

## The p-diff exception: interpolation in a tiny region

p-diff is the one method that beats the noise baseline, and its generated models even exceed the training models' accuracy. The authors do not take that as generalization; they ask where it comes from.

**The observation.** Plot the distribution of individual parameter values across checkpoints. The generated values concentrate around the *mean* of the training values. Averaging fine-tuned weights from a shared base is known to raise accuracy (the model-soups effect from Wortsman et al., 2022). So p-diff might be learning to average.

**Two interpolation baselines.**

$$
\mathbf{w}_{\text{avg}} = \frac{1}{16}\sum_{j=1}^{16} \mathbf{w}_{t_j},
\qquad
w_{\text{gauss}}^{(k)} \sim \mathcal{N}\!\big(\mu_k, \sigma_k^2\big), \;\; k = 1,\dots,d,
$$

where $\mathbf{w}_{t_j}$ are 16 randomly selected training checkpoints, $\mu_k$ and $\sigma_k^2$ are the mean and variance of parameter $k$ across the training set, and each coordinate of the Gaussian baseline is drawn independently. Both approximate "somewhere in the middle of the training cloud".

**The result.** Both baselines match p-diff's accuracy and maximum similarity closely, and t-SNE places p-diff's weights next to theirs. p-diff appears to interpolate.

The appendix explains why that interpolation is not generalization: the region it interpolates within is tiny.

| Distance (mean L2) | Value |
| --- | --- |
| between all pairs of p-diff training checkpoints | 6.9 |
| between training and generated checkpoints | 6.3 |
| between all pairs of from-scratch models | 46.1 |
| training to nearest training | 0.3 |
| generated to nearest training | 5.4 |
| from-scratch to nearest from-scratch | 44.1 |

![p-diff's training checkpoints form a short trajectory from one run with mean pairwise L2 6.9; generated checkpoints and the 16-checkpoint average sit in its middle; independently trained models are spread across a region with mean pairwise L2 46.1, drawn to the same scale](/imgs/blogs/generative-modeling-of-weights-generalization-or-memorization-3.webp)

That 5.4 versus 0.3 gap is what made p-diff look "novel" in Test 1. Against the 44.1 between independent models, its whole output distribution fits inside one basin of one run: a useful accuracy trick, like checkpoint averaging, but not a model of the space of good networks.

## Why do they memorize? Three suspects

Section 4 of the paper moves from "it memorizes" to "why". Three factors get experiments.

### Suspect 1: too little data

In image diffusion, larger models on smaller datasets memorize more. The authors scale G.pt's training set from 2.1M to 20.4M checkpoints and retrain.

| G.pt | mean L2, generated to nearest training | mean L2, training to nearest training | mean accuracy of generated |
| --- | --- | --- | --- |
| original (2.1M) | 2.25 | 3.64 | 94.0% |
| 10x data (20.4M) | 2.78 | 2.70 | 94.1% |

After scaling, a generated checkpoint is as far from its nearest training neighbor as training checkpoints are from each other, with no loss in accuracy. That is what an independently trained model looks like.

Data is not a universal fix, though. Scaling Hyper-Representations from 2,896 to 24,136 checkpoints leaves generated weights far closer to training weights (8.11 and 9.24) than training weights are to each other (49.97 and 48.17), and the ZU memorization score stays at $-13.6$.

### Suspect 2: too much capacity

**The test.** If a model memorizes, it should be able to memorize *noise* as well as structure (the Zhang et al. 2017 argument). The authors reinitialize one or all layers of HyperDiffusion's training MLPs with Kaiming uniform random weights, so part or all of each "checkpoint" carries no shape information, then retrain HyperDiffusion.

| Layers reinitialized | mean dist, training to nearest training | mean dist, generated to nearest training |
| --- | --- | --- |
| none | 109.5 | 7.0 |
| 1st | 114.0 | 9.0 |
| 2nd | 90.3 | 1.5 |
| 3rd | 106.0 | 2.6 |
| 4th | 126.4 | 3.6 |
| all | 16.0 | 0.8 |

Even when *every* layer is random, the generator reproduces training weights (0.8 versus 16.0). There is no pattern to learn in random weights, so this is pure memorization capacity. The authors also tried shrinking the model, changing training length, and adding regularization; none produced generalization, and most just degraded the generated models.

### Suspect 3: symmetry used only as augmentation

**The problem.** Hidden neurons have no natural order. Swap two of them, swap the matching rows and columns of the adjacent weight matrices, and the network computes exactly the same function.

**The math.** For a two-layer block $f(\mathbf{x}) = W_2\, \phi(W_1 \mathbf{x} + \mathbf{b}_1)$ with $h$ hidden units and elementwise activation $\phi$, any $h \times h$ permutation matrix $P$ gives

$$
W_2 P^{\top} \, \phi\big(P W_1 \mathbf{x} + P \mathbf{b}_1\big) = W_2\, \phi(W_1 \mathbf{x} + \mathbf{b}_1),
$$

because $\phi$ acts elementwise, so it commutes with reordering, and $P^{\top} P = I$. Each hidden layer of width $h$ therefore has $h!$ equivalent weight vectors. With $\phi = \tanh$, flipping the sign of a neuron's incoming and outgoing weights is a second symmetry, since $\tanh(-z) = -\tanh(z)$.

**A tiny example.** With $h = 3$ there are ${3! = 6}$ orderings. Swap neurons 1 and 3: the network is unchanged, but the flattened weight vector has its first and last blocks exchanged, so its L2 distance from the original is generally large.

![A 3-neuron MLP and the same MLP with neurons 1 and 3 swapped compute the same function, but their flattened weight vectors list the neuron blocks in different orders and are different points in weight space](/imgs/blogs/generative-modeling-of-weights-generalization-or-memorization-4.webp)

A generator that sees flattened vectors has to learn this equivalence from data. The paper checks whether augmentation teaches it.

**Invariance test on Hyper-Representations.** Its autoencoder is trained with a contrastive loss built from permuted pairs. If it had learned the symmetry, reconstructing a model and its permuted twin should give identical accuracy and an error IoU of 1. Instead the reconstructions differ in accuracy by 6.01% (permuted) and 5.82% (sign-flipped), against a 12.72% average difference between unrelated original models. Their error similarity is 0.62 for permuted inputs and only 0.19 for sign-flipped ones.

**Augmentation test on HyperDiffusion.** Adding 1, 3, or 7 random permutations of each training MLP enlarges the dataset by 2x, 4x and 8x (training epochs doubled to allow convergence).

![Figure 12 from Zeng et al. (2025): HyperDiffusion samples with default training and with 1, 3 and 7 added permutations per training MLP; meaningful shapes disappear as soon as permutations are added](/imgs/blogs/generative-modeling-of-weights-generalization-or-memorization-fig5.webp)

With a single extra permutation, HyperDiffusion stops producing meaningful shapes. With three, it fails to converge. Augmentation did not inject the symmetry; it turned one tight mode per shape into several far-apart modes that the model could no longer cover. The authors' reading: symmetry should be built into the architecture (graph networks over neurons, as in recent weight-space classifiers), not sprinkled on the data.

## Results in one table

| Method | Weight-space memorization (Test 1) | Behavioral copy (Test 2) | Beats noise baseline? | Notes |
| --- | --- | --- | --- | --- |
| Hyper-Representations | yes, vs reconstructed training weights | prediction overlap with nearest training 98.5% (training pairs: 75.6%) | no | data scaling to 24,136 does not help |
| G.pt | yes (2.25 vs 3.64) | overlap 98.2% (training pairs: 97.9%) | no | 10x data closes the gap |
| HyperDiffusion | yes, 94.4% within L2 10 | nearly identical airplanes | no difference | memorizes even random weights |
| p-diff | looks novel, because training set is one run | overlap 93.5% (training pairs: 91.4%) | yes | matched by averaging 16 checkpoints |

**What is load-bearing and might not transfer.** Every target network is small (MNIST/SVHN/CIFAR classifiers, neural-field MLPs) and only G.pt has a large training set. Testing each method in its primary setup is right for checking published claims, but leaves larger scales open; the G.pt data-scaling result hints the picture can change.

## Critique

**What is strong.**

- The paper evaluates methods on their own terms: their primary setups, their own similarity metric (from p-diff), and fair corrections (permutation search for G.pt, reconstruction for Hyper-Representations).
- The noise baseline reframes the question from "is it close to training data?" to "does it beat a free alternative?", which is the right test for any generator.
- The random-weights experiment is a clean capacity argument: a model that reproduces random numbers has not learned structure.

**What is weaker.**

- Four methods is a sample. Follow-ups that cite these methods (several are listed in the paper) are not tested, so "the field memorizes" is my extrapolation, not the paper's claim.
- L2 nearest-neighbor ignores scaling and sign symmetries for the methods that did not train on them, and only partially accounts for permutation. A symmetry-canonicalized distance (for example after weight matching) would be a stronger novelty test.
- The data-scaling success is shown on one method only; the failure on Hyper-Representations suggests data size is not the whole story, and the paper does not isolate what differs.
- There is no positive control showing the tests can detect real generalization; the G.pt 10x run partly plays this role.

**What would change my mind.** A weight generator trained on checkpoints from independent runs whose samples are, under a permutation-canonicalized metric, as far from their nearest training checkpoint as training checkpoints are from each other, *and* that beat the noise, averaging and Gaussian baselines on accuracy at matched novelty. That would show the memorization here is a property of these four designs, not of the task.

## Where this goes next

A negative result with a clean protocol is a good launch pad. Each idea below is my own extension.

**1. A generator that is permutation-equivariant by construction.**
- *Question:* does a diffusion model whose denoiser is a graph network over neurons (equivariant to hidden-unit permutations) generalize where flattened-vector diffusion memorizes?
- *Why now:* the paper shows augmentation fails and gives a ready-made test suite (nearest-neighbor L2, error IoU, noise baseline) to score the alternative.
- *First experiment:* replace HyperDiffusion's transformer with an equivariant neural-functional denoiser, train on the same 2,749 airplane MLPs, and rerun Test 1 and the 1/3/7-permutation experiment. Success means shapes survive augmentation.

**2. Provable generalization through canonicalization.**
- *Question:* if every training checkpoint is first mapped to a canonical neuron order (weight matching to a reference), does the generator still memorize?
- *Why now:* Table 1 shows memorization even with random layers, so you can test whether canonicalization removes spurious modes without changing the data.
- *First experiment:* canonicalize G.pt's 2.1M checkpoints, retrain at the original scale, and check whether generated-to-nearest L2 rises to the training-to-nearest level, as it did only with 10x data.

**3. A memorization audit for hypernetwork claims.**
- *Question:* do task-supervised hypernetworks (which never see a checkpoint collection) also collapse onto a few solutions?
- *Why now:* the paper separates them from checkpoint-trained generators but leaves them untested; the IoU and noise-baseline protocol applies unchanged, with independently trained models as the reference set.
- *First experiment:* sample 100 networks from a published hypernetwork, train 100 from scratch, and compare max-IoU distributions and accuracy at matched novelty.

**4. Novelty metrics that are symmetry-aware and scale-free.**
- *Question:* can a single metric replace raw L2 and the per-task IoU/CD swap?
- *Why now:* the paper needs three novelty measures for four methods, and the overlap floor of $\max(2x-1,0)$ shows how easily metrics get confounded.
- *First experiment:* combine permutation-aligned L2 with activation CKA, and check it ranks generated, noise-added and from-scratch models as IoU does.

**5. Scaling curves for memorization.**
- *Question:* at what ratio of training checkpoints to generator parameters does memorization switch off?
- *Why now:* the G.pt 2.1M to 20.4M result gives one crossing point, and Table 1 gives an extreme memorizing point.
- *First experiment:* train G.pt-style generators at three model sizes and four data sizes, and plot generated-to-nearest over training-to-nearest L2. A ratio near 1 is the target.

## References

- Boya Zeng, Yida Yin, Zhiqiu Xu, Zhuang Liu. *Generative Modeling of Weights: Generalization or Memorization?* arXiv:2506.07998, 2025; CVPR 2026 Highlight. [https://arxiv.org/abs/2506.07998](https://arxiv.org/abs/2506.07998)
- Project page and code: [https://boyazeng.github.io/weight_memorization](https://boyazeng.github.io/weight_memorization)
- CVPR 2026 poster page: [https://cvpr.thecvf.com/virtual/2026/poster/40002](https://cvpr.thecvf.com/virtual/2026/poster/40002)
- On this site: [where architectures diverge](/blog/paper-reading/computer-vision/consensus-vs-controversy-mapping-the-decision-space-where-architectures-diverge), [HyperDAS hypernetworks](/blog/paper-reading/ai-interpretability/hyperdas-towards-automating-mechanistic-interpretability-with-hypernetworks), [the memorization-generalization tradeoff in Wide & Deep](/blog/machine-learning/recommendation-systems/wide-and-deep-and-the-memorization-generalization-tradeoff).
- Methods analyzed: Schürholt et al. (Hyper-Representations, 2021; 2022), Peebles et al. (G.pt, 2022), Erkoç et al. (HyperDiffusion, 2023), Wang et al. (p-diff, 2024).
