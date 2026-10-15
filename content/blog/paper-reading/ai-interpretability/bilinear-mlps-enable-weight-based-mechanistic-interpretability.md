---
title: "Bilinear MLPs Enable Weight-based Mechanistic Interpretability: Reading an MLP Straight From Its Weights"
date: "2026-10-15"
publishDate: "2026-10-15"
description: "How dropping the nonlinearity from a gated MLP turns the layer into a third-order tensor you can eigendecompose, and what that reveals about MNIST digits, overfitting, adversarial masks and a sentiment negation circuit in a small language model."
tags: ["paper-reading", "bilinear-mlp", "weight-analysis", "mechanistic-interpretability", "eigendecomposition", "sparse-autoencoders", "neural-networks", "iclr-2025-spotlight"]
category: "paper-reading"
subcategory: "AI Interpretability"
author: "Hiep Tran"
featured: true
readTime: 21
paper:
  title: "Bilinear MLPs Enable Weight-based Mechanistic Interpretability"
  authors: "Michael T. Pearce, Thomas Dooms, Alice Rigg, Jose Oramas, Lee Sharkey"
  venue: "ICLR 2025 (spotlight)"
  url: "https://arxiv.org/abs/2410.08417"
---

> [!tldr]
> **Claim.** If you remove the activation function from a gated MLP, so the layer computes $(Wx) \odot (Vx)$, every output becomes a quadratic form in the input. The whole layer is then a single third-order tensor, and you can study it with plain linear algebra, directly from the weights, without running a single input through it.
> **Why it matters.** Almost all modern interpretability is activation-based: you collect a dataset, record activations, and fit something (an SAE, a probe) to them. That tells you which features are present, not how the weights build them. Bilinear MLPs make weight-based analysis practical again.
> **Most surprising result.** In language models, SAE output features turn out to be strikingly low-rank: two eigenvectors of one interaction matrix already give most features a correlation above 0.75 with the real activation, and one hand-picked circuit (a "not" plus "bad" means "boost good" gate) is readable straight from the spectrum.
> **Where it fails.** You still need meaningful output directions, which in deep models come from SAEs and therefore from data. Eigenvectors are orthogonal by construction, with no guarantee of being monosemantic. And bilinear layers cost about 6% in data efficiency compared to SwiGLU.

Most of what we know about MLP layers in transformers comes from watching them run. That is a strange state of affairs: the weights are right there, fully known, and yet we treat the layer as a black box we have to probe with data. This paper, by Pearce, Dooms, Rigg, Oramas and Sharkey, argues the black box is mostly self-inflicted. It comes from one elementwise nonlinearity, and a well-known MLP variant already lives without it.

I will walk through five ideas: the bilinear layer and its tensor form, the symmetry trick, the eigendecomposition that is the heart of the paper, the three analysis recipes, and then the experiments on images and language that test whether the eigenvectors mean anything.

## The problem: nonlinearities hide the computation

An ordinary MLP computes $P\,\sigma(Wx)$, where $\sigma$ is ReLU or GELU applied to each hidden unit separately. The elementwise $\sigma$ is what makes the network expressive. It is also what makes it opaque: once a hidden unit passes through a kink, the contribution of each input direction depends on every other input direction in a way you cannot write down as a clean sum.

That is why interpretability work has drifted toward activations. Sparse dictionary learning, the SAE family, finds an overcomplete set of directions that the model seems to use. The paper's complaint is precise: SAEs "only describe which features are present, not how they are formed or what their downstream effect is." Tools that connect features across layers, like [sparse feature circuits](/blog/paper-reading/ai-interpretability/sparse-feature-circuits-discovering-and-editing-interpretable-causal-graphs-in-language-models) or [transcoders](/blog/paper-reading/ai-interpretability/transcoders-find-interpretable-llm-feature-circuits), either use gradient-based linear approximations or train a replacement MLP on a dataset. Both inherit the data distribution, so neither gives strong guarantees outside it.

The alternative is an architecture whose computation is readable by construction, without giving up accuracy. That is what the bilinear MLP offers.

## Technique 1: the bilinear layer is a gated MLP minus the gate's kink

**The problem.** We want an MLP that performs well in a transformer but whose input to output map can be written with linear algebra.

**The analogy.** Picture two committees reading the same memo. Each committee member writes one score. The final verdict for seat $a$ is the product of committee W's score and committee V's score for that seat. Nobody applies a threshold. A product of two linear judgments is the simplest nonlinear thing you can build, and it is still fully described by who weighted which word.

**The mechanism.** Modern transformers use Gated Linear Units. A GLU projects the input up twice with two matrices, passes one copy through an activation, and multiplies them elementwise. SwiGLU uses Swish as that activation. The bilinear layer simply deletes the activation, then projects back down as usual.

**The math.** A GLU hidden layer is

$$
g(x) = (Wx) \odot \sigma(Vx), \qquad \text{GLU}(x) = P\,g(x)
$$

where $x \in \mathbb{R}^{d_\text{input}}$ is the input, $W, V \in \mathbb{R}^{d_\text{hidden} \times d_\text{input}}$ are the two up-projections, $\odot$ is the elementwise product, $\sigma$ is the activation, and $P \in \mathbb{R}^{d_\text{output} \times d_\text{hidden}}$ is the down-projection. The bilinear layer sets $\sigma$ to the identity: $g(x) = (Wx) \odot (Vx)$.

![Redrawn: a SwiGLU MLP next to a bilinear MLP. The only difference is the removed activation on the gate branch, which turns each hidden unit into a quadratic form in the input.](/imgs/blogs/bilinear-mlps-enable-weight-based-mechanistic-interpretability-1.webp)

Look at a single hidden unit $a$. Write $w_{a:}$ and $v_{a:}$ for row $a$ of $W$ and $V$. Then

$$
g(x)_a = (w_{a:}^\top x)(v_{a:}^\top x) = x^\top \left(w_{a:} v_{a:}^\top\right) x
$$

The $d_\text{input} \times d_\text{input}$ matrix $B_{a::} = w_{a:} v_{a:}^\top$ is what the paper calls an **interaction matrix**. Entry $(i, j)$ says how strongly input coordinate $i$ multiplies with input coordinate $j$ to drive unit $a$. Stack all units and you get the **bilinear tensor** $B$ with entries $b_{aij} = w_{ai} v_{aj}$, of shape $(d_\text{hidden}, d_\text{input}, d_\text{input})$. Folding $P$ in gives shape $(d_\text{output}, d_\text{input}, d_\text{input})$.

**Worked micro-example.** Take ${2}$ input dimensions and one unit with $w = (1, 1)$ and $v = (1, -1)$. Then $g(x) = (x_1 + x_2)(x_1 - x_2) = x_1^2 - x_2^2$. At $x = (3, 1)$ that is ${4 \times 2 = 8}$, and also ${9 - 1 = 8}$. The interaction matrix is $w v^\top$, with rows $(1, -1)$ and $(1, -1)$.

**When it fails.** A bilinear unit can only model pairs of features. A purely linear dependence on one feature has to be faked as a quadratic. Appendix L of the paper fixes this with biases, $(Wx + b) \odot (Vx + c)$, which expands to an interaction term, a linear term $cWx + bVx$, and a constant $cb$. The authors note that in practice training often fails to use this flexibility. Performance is the other cost, covered in the results section: close to SwiGLU, not equal.

## Technique 2: only the symmetric part matters

**The problem.** The interaction matrix in the example above, rows $(1, -1)$ and $(1, -1)$, does not look anything like $x_1^2 - x_2^2$. It carries information that never reaches the output.

**The analogy.** A handshake between person $i$ and person $j$ is the same handshake whichever of them you list first. Booking it as "$i$ shook $j$" with weight 3 and "$j$ shook $i$" with weight $-1$ is just a noisy way of saying the pair shook with total weight 2.

**The mechanism.** Any square matrix splits uniquely into a symmetric part and an antisymmetric part. Because the interaction matrix is always sandwiched between two copies of the same $x$, the antisymmetric part contributes exactly zero.

**The math.**

$$
B_{a::} = \underbrace{\tfrac{1}{2}\left(B_{a::} + B_{a::}^\top\right)}_{\text{symmetric}} + \underbrace{\tfrac{1}{2}\left(B_{a::} - B_{a::}^\top\right)}_{\text{antisymmetric}}, \qquad x^\top A x = 0 \text{ for antisymmetric } A
$$

The second identity holds because $x^\top A x$ is a scalar, so it equals its own transpose $x^\top A^\top x = -x^\top A x$. From here on the paper replaces every interaction matrix by its symmetric part, $b_{aij} = \tfrac{1}{2}(w_{ai} v_{aj} + w_{aj} v_{ai})$.

**Worked micro-example.** The symmetric part of rows $(1, -1), (1, -1)$ is rows $(1, 0), (0, -1)$. That is exactly $x_1^2 - x_2^2$ read off the diagonal. The discarded antisymmetric part is rows $(0, -1), (1, 0)$, and it gives $-x_1 x_2 + x_2 x_1 = 0$ for any $x$.

**Why it matters.** Symmetric real matrices have real eigenvalues and orthogonal eigenvectors, by the spectral theorem. Without this step the eigendecomposition below could produce complex numbers and non-orthogonal directions, and the "sum of independent contributions" story would fall apart.

## Technique 3: the eigendecomposition of an output direction

This is the paper's main tool, and Figure 1 shows the whole idea in two panels.

![Figure 1 from Pearce et al. (2025): A) a bilinear layer as an elementwise product and as a third-order tensor. B) multiplying the tensor by an output direction u gives an interaction matrix Q, which decomposes into eigenvectors and eigenvalues.](/imgs/blogs/bilinear-mlps-enable-weight-based-mechanistic-interpretability-fig1.webp)

**The problem.** The tensor $B$ is huge: for MNIST it is ${10 \times 512 \times 512}$ after folding in the head. We want to ask one question at a time, such as "what in the input makes the layer push toward the digit 5?", and get a short, ranked answer.

**The analogy.** A recipe for one dish is a list of ingredient pairs and how well they combine. Eigendecomposition rewrites that list as a handful of "flavor profiles". Each profile is a fixed blend of ingredients. The dish's score is the sum, over profiles, of how strongly the input matches that profile, squared, times how much the dish likes it. Because the profiles are orthogonal, they do not interact with each other.

**The mechanism.** Pick an output direction $u$: a class unembedding, an SAE feature's encoder vector, or anything downstream you care about. Contract the tensor's output axis with $u$ to get one symmetric matrix $Q$. Eigendecompose $Q$. Sort the eigenvectors by $|\lambda|$. The top few are your answer.

**The math.** With $Q = u \cdot_\text{out} B = \sum_a u_a B_{a::}$, the spectral theorem gives

$$
Q = \sum_{i=1}^{d} \lambda_i\, v_i v_i^\top
$$

where $d$ is the rank of $W$ and $V$, the $v_i \in \mathbb{R}^{d_\text{input}}$ are orthonormal eigenvectors and the $\lambda_i$ are real eigenvalues. The layer's output along $u$ is then

$$
u^\top \text{layer}(x) = x^\top Q x = \sum_{i=1}^{d} \lambda_i \left(v_i^\top x\right)^2
$$

Each term is the "activation" of eigenvector $v_i$: square how much the input lines up with $v_i$, scale by $\lambda_i$. Positive $\lambda_i$ pushes the output up, negative pushes it down. This decomposition is exact, not an approximation. Truncating to the top $k$ terms gives the best rank-$k$ approximation of $Q$.

![Redrawn: the weight-only pipeline. Weights become a tensor, an output direction slices out a symmetric interaction matrix, and its eigenvectors explain that output as a sum of squared projections.](/imgs/blogs/bilinear-mlps-enable-weight-based-mechanistic-interpretability-2.webp)

**Worked micro-example.** Our symmetric matrix with rows $(1, 0), (0, -1)$ has eigenvectors $e_1$ with $\lambda = 1$ and $e_2$ with $\lambda = -1$. So $x^\top Q x = (e_1^\top x)^2 - (e_2^\top x)^2$, which is $x_1^2 - x_2^2$ again. Now take a single eigenvector $v = (1, -1)/\sqrt{2}$ over two pixels with $\lambda = 2$. Pixel 1 on, pixel 2 off: ${2 \cdot (1/\sqrt{2})^2 = 1}$. Pixel 2 alone: also 1. Both on: $(v^\top x)^2 = 0$, output 0. The squared projection behaves like XOR.

That XOR behavior is why the paper says the sign of an eigenvector is arbitrary and why its plots show blue (positive) and red (negative) regions: a large activation needs the input to overlap the blue region or the red region, but not both.

**When it fails.** You need a good $u$. For a classifier the unembedding is natural. Deep inside a language model you need features from somewhere, usually SAEs trained on data, which reintroduces the dependency the method was meant to remove. And orthogonality is a constraint, not a discovery: if the true features overlap, the eigenvectors will be forced to mix them.

## Technique 4: three recipes depending on what you already know

The paper frames analysis as a function of which features you already have. The figure below is my summary of section 3.

![Redrawn: choosing the analysis method. With input and output features, transform B into the feature basis. With output features only, eigendecompose. With nothing, run a higher-order SVD first.](/imgs/blogs/bilinear-mlps-enable-weight-based-mechanistic-interpretability-3.webp)

- **Input and output features → direct interactions.** Given dictionaries $F^\text{in}$ and $F^\text{out}$, rotate the tensor into feature space: $\tilde b_{abc} = \sum_{ijk} f^\text{out}_{ai}\, b_{ijk}\, f^\text{in}_{bj}\, f^\text{in}_{ck}$. For SAEs the output side uses encoder rows $e^\text{out}_{ai}$ instead, so that an output feature's activation is $z^\text{out}_c = \text{ReLU}\left(\sum_{ab} \tilde b_{abc}\, z^\text{in}_a z^\text{in}_b\right)$ in terms of input feature activations $z^\text{in}$. With sparse inputs, only the few pairs where both features fire contribute.
- **Output features only → eigendecomposition.** Technique 3, applied per output direction.
- **No features → HOSVD.** Flatten the two input axes so $B$ becomes a $(d_\text{output}, d_\text{input}^2)$ matrix, run an ordinary SVD, and treat each right singular vector as an interaction matrix $q$ to eigendecompose further: $B_{\text{out}, \text{in} \times \text{in}} = \sum_i \sigma_i\, u^{(i)}_\text{out} \otimes q^{(i)}_{\text{in} \times \text{in}}$. Appendix D shows that on MNIST the top output direction from this roughly splits the digits into two groups.

The middle recipe carries nearly all the experiments, so the rest of the post focuses on it.

## Technique 5: adversarial masks built from the pseudoinverse

**The problem.** If eigenvectors really drive the output, we should be able to steer the model by writing them into the input. The catch: eigenvectors for different digits are not orthogonal to each other. An input along "digit 3, eigenvector 1" may also light up "digit 8, eigenvector 2".

**The analogy.** In a key-value store, a good key matches its own slot and no other. The eigenvectors are the slots. We want keys.

**The mechanism and math.** Stack a set of eigenvectors (the paper uses the top 10 positive) as the rows of a matrix $V$. Take its pseudoinverse $V^+$. The mask for eigenvector $i$ is the matching slice $m_i$ of $V^+$, which satisfies

$$
v_j \cdot m_i = \delta_{ij}
$$

where $\delta_{ij}$ is 1 when $i = j$ and 0 otherwise. So $m_i$ activates eigenvector $i$ with unit projection and every other eigenvector in the set with zero projection.

**Worked micro-example.** Let $v_1 = (1, 0.6)$ and $v_2 = (0.6, 1)$. Using $v_1$ itself as the mask hits $v_2$ with projection ${0.6 + 0.6 = 1.2}$, larger than its hit on $v_1$. Inverting the ${2 \times 2}$ matrix (determinant ${1 - 0.36 = 0.64}$) gives $m_1 = (1.5625, -0.9375)$. Check: $v_1 \cdot m_1 = 1.5625 - 0.5625 = 1$ and $v_2 \cdot m_1 = 0.9375 - 0.9375 = 0$.

**What the paper found.** A mask built from a digit-3 eigenvector does not look like a 3, yet adding it drops accuracy, averaged over masks from the top three eigenvectors, significantly more than a randomly permuted mask. This held even for a model regularized with Gaussian input noise of standard deviation 0.15. For an unregularized model, the authors built masks that touch only the outer ring of pixels that are active on less than 1% of samples, exploiting the edge overfitting that the eigenvectors exposed. No gradient descent, no forward passes: the attack is computed from weights.

**When it fails.** This is a toy, single-layer setting on MNIST. Nothing in the paper shows the construction scaling to deep networks, where the input to the bilinear layer is itself a learned representation.

## Image experiments: do the eigenvectors mean anything?

The image models are deliberately tiny: a ${784 \to 512}$ embedding, one bilinear layer of width 512, and a 10-way head (Appendix G). Training uses AdamW, weight decay 1.0, batch 2048, and Gaussian input noise of norm 0.5 unless stated. Eigenvectors are projected back to pixel space through the embedding so they can be drawn.

### Qualitative: strokes, edges and Gabor-like pairs

For MNIST, the top positive eigenvector of each digit looks like a curve segment of that digit or a prototype of it. For Fashion-MNIST they act as localized edge detectors, such as the gap between trouser legs. That edge detection depends on the XOR behavior from Technique 3.

![Figure 3 from Pearce et al. (2025): top four positive (top row) and negative (bottom row) eigenvectors for the digit 5, with the eigenvalue spectra on the left. Only 20 positive and 20 negative of 512 eigenvalues are shown.](/imgs/blogs/bilinear-mlps-enable-weight-based-mechanistic-interpretability-fig2.webp)

The digit-5 spectrum is the clearest example. Only a small fraction of the 512 eigenvalues are non-negligible. The first two positive eigenvectors both detect the 5's horizontal top stroke at slightly different positions, which the authors compare to Gabor filters. The next two catch the bottom curve. The negative ones are harder to read, but they tend to encode evidence against a 5, such as an upward curve in the top right where a 5 would have a flat stroke.

### Spotting overfitting from weights alone

This is the most practical result in the image section. Without regularization, the top eigenvectors put weight on a few outlying pixels near the border. Adding dense Gaussian noise to the inputs removes those artifacts and makes the eigenvectors look like digits, with test accuracy staying between 97.2% and 98.1% across the noise levels in Figure 4. Appendix B compares other regularizers:

| Technique | Effect on eigenvectors (Appendix B) |
| --- | --- |
| Dense Gaussian input noise | Best trade-off: sparser, more interpretable, small accuracy cost |
| Salt-and-pepper noise | Lower accuracy and lower interpretability |
| Perlin noise | Worst; did not fix overfitting |
| Weight decay | Zeroes the long tail of eigenvalues, eigenvectors mostly unchanged |
| Dropout, activation noise | No visible effect in these shallow models |
| Translation augmentation | Smoother features, higher rank, overfitting not removed |

The weight decay row is the interesting one for scaling: it does not change what the top eigenvectors look like, but it cuts how many you need to inspect.

### Quantitative: consistency and truncation

![Figure 5 from Pearce et al. (2025): A) cosine similarity between ordered eigenvectors of independently trained models of the same size. B) classification error after keeping only the top n eigenvalues per digit, across model sizes.](/imgs/blogs/bilinear-mlps-enable-weight-based-mechanistic-interpretability-fig3.webp)

Two checks matter here. First, independent training runs learn the same top eigenvectors: cosine similarity of the top eigenvector sits between 0.8 and 0.9 depending on model size, with larger models more consistent. Second, truncation is cheap: keeping only a handful of eigenvectors per digit costs very little accuracy (Appendix F reports drops as small as 0.01%), and the curves for widths from 30 to 1000 nearly overlap. Small and large models reach similar accuracy from their top few eigenvectors.

### Ground truth: the similarity challenge

The strongest image test uses a model whose answer key is known. In a mechanistic interpretability challenge set by Casper (2023), a binary MNIST classifier outputs True when an input has high cosine similarity to a specific target image of a "1", or to its complement (one minus the grayscale). Earlier solvers needed the dataset and a hint.

The authors train a bilinear model on the same task (not the original architecture, a point I return to in the critique) and decompose the single meaningful direction, True minus False. One large positive eigenvalue (0.19) appears and the rest are near zero. That top eigenvector computes the target similarity: the blue region matches the target, the red region matches the complement, and both at once cancel. This task needed biases, because the model must compare its similarity score to a learned threshold.

## Language experiments: circuits between SAE features

Section 5 moves to a 6-layer bilinear transformer (model width 512, MLP width 2048, about 33M parameters) trained on a cleaned TinyStories with a 4096-token lowercase tokenizer, plus two FineWeb models shaped like GPT2-small and GPT2-medium with bilinear MLPs (162M and 335M parameters). SAEs use TopK with $k = 30$, following [Gao et al.](/blog/paper-reading/ai-interpretability/scaling-and-evaluating-sparse-autoencoders).

### The sentiment negation circuit

The authors say plainly that they cherry-picked this circuit to show what the method can do. It lives in layer 4 of the TinyStories model, between two output SAE features (indices 1882 and 1179) whose decoder vectors have cosine similarity $-0.975$. They fire on negation tokens such as "not", "never" and "wasn't". One boosts positive-sentiment tokens ("good", "safe", "nice") and the other boosts negative ones ("bad", "hurt", "sad"). Call them not-good and not-bad.

![Figure 8 from Pearce et al. (2025): the sentiment negation circuit. A) the top 15 interactions for the not-good feature. B) input features projected onto the top positive and negative eigenvectors. C) the not-good activation against its two-eigenvector approximation.](/imgs/blogs/bilinear-mlps-enable-weight-based-mechanistic-interpretability-fig4.webp)

Read with the direct-interaction recipe, the not-good feature is an AND gate. Input features that boost negative sentiment have strong positive interactions with negation features. Both kinds have negligible self-interactions, so neither fires the output alone. The interactions shown are far above the typical cross-interaction, whose standard deviation is 0.004.

![Redrawn: the not-good feature as an AND gate. A negation feature and a negative-sentiment feature only drive the output together, because the useful weight sits in the cross term and the self terms are near zero.](/imgs/blogs/bilinear-mlps-enable-weight-based-mechanistic-interpretability-4.webp)

The eigendecomposition gives the same picture from a different angle. The spectrum has two outliers, $+0.62$ and $-0.66$. A sentiment feature alone projects about equally onto both eigenvectors, so the two squared terms nearly cancel. Add a negation feature and the input swings onto the positive eigenvector.

**Worked micro-example (my toy numbers, not the paper's).** Suppose a sentiment feature projects $(0.5, 0.5)$ onto the (positive, negative) eigenvectors and a negation feature projects $(0.5, -0.5)$. Sentiment alone gives ${0.62 \times 0.25 - 0.66 \times 0.25 = -0.01}$. Negation alone gives the same $-0.01$. Together the projections add to $(1, 0)$ and the output is ${0.62}$. That is an AND gate made of two squares.

Quantitatively, the two-eigenvector approximation correlates 0.66 with the true not-good activation when the feature is active, and 0.76 at large activations.

### Low-rank approximations at scale

Is the negation circuit a lucky exception? Section 5.2 checks every output feature at roughly two-thirds depth in three models: "ts-tiny" (6 layers, TinyStories), "fw-small" (12 layers) and "fw-medium" (16 layers). For each feature, approximate its activation with the top $k$ eigenvectors of its interaction matrix and compute the correlation over inputs where the feature is active.

![Figure 9 from Pearce et al. (2025): A) average active-only correlation versus number of top eigenvectors for three model sizes. B) distribution of correlations using the top two eigenvectors. C) scatter plots for nine random fw-medium features.](/imgs/blogs/bilinear-mlps-enable-weight-based-mechanistic-interpretability-fig5.webp)

The average correlation starts around 0.65 with a single eigenvector and climbs steadily as more are added. Most features exceed 0.75 with just two. The scatter plots show the approximation tracking the tail of large activations, and the low correlations come mostly from low-activation features.

A side finding in Appendix H deserves attention. The correlation depends heavily on how long the SAE was trained, while the usual SAE metrics barely move:

| Relative SAE training steps | 1 | 2 | 4 | 8 | 16 |
| --- | --- | --- | --- | --- | --- |
| Normalized MSE | 0.17 | 0.16 | 0.16 | 0.15 | 0.15 |
| Loss recovered | 0.60 | 0.61 | 0.65 | 0.65 | 0.66 |
| Mean two-eigenvector correlation | 0.17 | 0.28 | 0.42 | 0.52 | 0.59 |

Reconstruction error changes by about 10% while the correlation more than triples. The authors call it a hidden transition near convergence. I read it as a possible new SAE quality metric that reconstruction loss is blind to; that is my interpretation, not a claim the paper tests.

## Results: what does bilinearity cost?

Interpretability is worthless if nobody will train the architecture. Appendix I compares MLP variants in the 6-layer TinyStories model ($d_\text{model} = 512$, $d_\text{hidden} = 2048$, about 30M parameters):

| Final loss (TinyStories, 6 layers) | Bilinear | ReGLU | SwiGLU |
| --- | --- | --- | --- |
| Constant epochs (data-matched) | 1.337 | 1.332 | 1.321 |
| Constant wall-clock time (compute-matched) | 1.337 | 1.337 | 1.336 |

SwiGLU reaches the bilinear model's final loss with 6% fewer epochs, so bilinear is less data efficient. At matched compute the gap vanishes, because the bilinear layer skips the activation. The authors add a footnote that fused kernels could change this.

Appendix J tries the obvious migration path. Swish is $x \cdot \text{sigmoid}(\beta x)$, and $\beta = 0$ makes the gate linear. Fine-tuning TinyLlama-1.1B on FineWeb while annealing $\beta$ to zero over the first 120M tokens, then training 280M more tokens, raises loss by about 0.05 against a continued-training baseline, with the gap shrinking as training continues. That is one schedule on one GPU, so treat it as a proof of concept.

**What might not transfer.** Every image result uses a single bilinear layer between a linear embedding and a linear head, which is the setting where the unembedding gives a perfect $u$. Every language result uses models up to 335M parameters with narrow (4x) SAEs for the circuit work. The headline "weight-based interpretability is viable for large language models" is an extrapolation from these sizes.

## Critique

**What is strong.** The core insight is simple and exact. The decomposition is not an attribution approximation: $x^\top Q x = \sum_i \lambda_i (v_i^\top x)^2$ holds for every input, so whatever you read off the spectrum really is the layer's computation along $u$. The ground-truth similarity task is a good test because the answer was fixed in advance. The overfitting and adversarial-mask results show the weights telling you something you can act on without data. And the cost table is honest about the 6% data-efficiency gap.

**What is weak.**

- **The output directions still come from data.** Beyond the last layer, $u$ is an SAE feature, and SAEs are trained on activations. The paper says this in its limitations, but it undercuts the "weights only" framing for anything deep.
- **The circuit is cherry-picked.** The authors say so. Figure 9 is the systematic evidence, and it measures correlation, not causal faithfulness. A correlation of 0.66 on the showcase circuit leaves a lot of the not-good feature unexplained.
- **The challenge model is not the original.** The Casper task was solved on a retrained bilinear model with biases, so it shows bilinear models are readable, not that the method reverse-engineers an arbitrary trained network.
- **Orthogonality is imposed.** For high-rank spectra, eigenvectors may be forced to blend unrelated features. The authors suggest sparse dictionary learning on the tensor as a fix, which is not tested here.
- **Single-layer scope.** Everything is one layer at a time. Composing two bilinear layers gives a degree-4 polynomial, and the paper does not show how the analysis carries across layers.

**Missing ablation.** I would want a causal test: ablate everything except the top two eigenvectors of the not-good interaction matrix inside the running model and measure the change in next-token loss on negation contexts. That would turn "correlates 0.66" into "accounts for X% of the behavior".

**What would change my mind.** If the low-rank correlations in Figure 9 dropped sharply in a 1B-plus bilinear model, or if the top eigenvectors of SAE features stopped being interpretable once SAE width grew past 4x, I would read this as a property of small models rather than of bilinear layers. Conversely, if a fine-tuned bilinear Llama-scale model yielded readable eigenvector circuits matching causal ablations, I would treat weight-based analysis as a real alternative to activation-based tooling.

## References

1. Michael T. Pearce, Thomas Dooms, Alice Rigg, Jose Oramas, Lee Sharkey. "Bilinear MLPs Enable Weight-based Mechanistic Interpretability." ICLR 2025. [arXiv:2410.08417](https://arxiv.org/abs/2410.08417)
2. Code: [github.com/tdooms/bilinear-decomposition](https://github.com/tdooms/bilinear-decomposition)
3. Noam Shazeer. "GLU Variants Improve Transformer." 2020.
4. Lee Sharkey. "A technical note on bilinear layers for interpretability." 2023.
5. Related on this site: [Sparse Feature Circuits](/blog/paper-reading/ai-interpretability/sparse-feature-circuits-discovering-and-editing-interpretable-causal-graphs-in-language-models), [Transcoders Find Interpretable LLM Feature Circuits](/blog/paper-reading/ai-interpretability/transcoders-find-interpretable-llm-feature-circuits), [Scaling and Evaluating Sparse Autoencoders](/blog/paper-reading/ai-interpretability/scaling-and-evaluating-sparse-autoencoders)
