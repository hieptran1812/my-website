---
title: "Position: The Platonic Representation Hypothesis, and Why Different Models May Be Learning the Same World"
publishDate: "2026-10-15"
date: "2026-10-15"
description: "How Huh, Cheung, Wang and Isola measure representational convergence with mutual nearest neighbors, why they think scale drives it, and the contrastive-learning math that says every modality should land on the same PMI kernel."
category: "paper-reading"
subcategory: "AI Interpretability"
tags: ["paper-reading", "platonic-representation", "representation-learning", "representational-alignment", "mutual-nearest-neighbors", "contrastive-learning", "pointwise-mutual-information", "convergence", "interpretability", "icml-2024-oral"]
author: "Hiep Tran"
featured: true
readTime: 20
image: "/imgs/blogs/position-the-platonic-representation-hypothesis-fig1.webp"
excerpt: "Vision models and language models are trained on different data with different objectives, yet as they grow, they start to agree on which things are similar. This paper argues the agreement has an endpoint: a shared statistical model of reality."
paper:
  title: "Position: The Platonic Representation Hypothesis"
  authors: "Minyoung Huh, Brian Cheung, Tongzhou Wang, Phillip Isola"
  venue: "ICML 2024"
  url: "https://arxiv.org/abs/2405.07987"
---

> [!tldr]
> **The claim.** Neural networks trained with different objectives, on different data and even different modalities, are converging toward a shared statistical model of reality in their representation spaces. The authors call the endpoint the *platonic representation*.
>
> **Why it matters.** If true, a large language model and a large vision model are two windows onto the same geometry. That would explain why a single linear layer can stitch a vision encoder into an LLM, and it implies image data should help train text models (and the reverse).
>
> **The most surprising result.** Language models that never saw a pixel align more with a vision model (DINOv2) the better they are at predicting text. Alignment to vision even tracks downstream scores on Hellaswag and GSM8K.
>
> **Where it is thin.** The absolute alignment numbers are small (the cross-modal score tops out around 0.16 on a scale of 0 to 1), the math only holds for lossless, bijective observations of a discrete world, and the evidence is correlational. It is a position paper: a well-argued bet, not a theorem.

A position paper has no new architecture to dissect. Instead there are three things to take apart: a **measurement** (how do you compare two unrelated models?), a set of **arguments** (why would they converge?), and a **derivation** (what would they converge to?).

## The hypothesis in one picture

The paper opens with a cartoon. There is a world $Z$: here, a red sphere next to a blue cone. A camera captures it as an image $X$. Someone describes it in words as $Y$. An image model $f_{\text{img}}$ embeds $X$; a language model $f_{\text{text}}$ embeds $Y$.

![Figure 1 from Huh et al. (2024): the world Z is projected into images X and text Y, and the image and text encoders both end up modeling Z](/imgs/blogs/position-the-platonic-representation-hypothesis-fig1.webp)

The hypothesis is that both embeddings are, underneath, modeling $Z$. A model that learns the structure of the shadows well enough ends up learning the structure of the thing casting them. Hence the name: Plato's prisoners see only shadows on the cave wall, and the authors bet our models are reconstructing the world outside. The driver, they claim, is **scale**: more parameters, data and tasks push representations together. That is the testable part.

## What "the same representation" means

Before measuring convergence, we need a definition that works across models with different widths, architectures and input types. A 768-dimensional ViT feature and a 4096-dimensional LLaMA feature cannot be compared coordinate by coordinate.

**The problem.** Two models can encode the same knowledge in totally different coordinates. Rotate a feature space and nothing about what it knows has changed, yet every number is different.

**The analogy.** Think of two city maps drawn by different cartographers, one in metric with north up, one in miles with east up. You cannot overlay them. But you can ask each map the same question: "which three cafes are closest to the station?" If the answers match for every landmark, the maps describe the same city.

**The mechanism.** So instead of comparing vectors, compare *similarity structures*. Each model induces a table of how similar every pair of inputs is. Two models are aligned when their tables agree.

**The math.** The paper makes this precise with three definitions:

- A **representation** is a function $f: \mathcal{X} \to \mathbb{R}^n$ that maps each input in a data domain $\mathcal{X}$ to an $n$-dimensional feature vector.
- A **kernel** $K: \mathcal{X} \times \mathcal{X} \to \mathbb{R}$ records how the representation measures similarity between two inputs.
- A **kernel-alignment metric** $m: \mathcal{K} \times \mathcal{K} \to \mathbb{R}$ scores how similar two kernels are, where $\mathcal{K}$ is the space of kernels.

The kernel used throughout is the plain inner product:

$$
K(x_i, x_j) = \langle f(x_i), f(x_j) \rangle
$$

Here $x_i, x_j \in \mathcal{X}$ are two inputs and $\langle \cdot, \cdot \rangle$ is the dot product. Over $b$ items this gives a $b \times b$ matrix, and two models are compared through their two matrices. Alignment is "a similarity of similarities," and because inner products ignore rotations, it has exactly the invariance the map analogy asked for.

## Measuring alignment: the mutual k-NN metric

**The problem.** The standard tool for comparing kernels is CKA (Centered Kernel Alignment). The authors found it showed only a very weak trend, even within one modality. CKA is global: it weighs every pair equally, yet, as the appendix puts it, the similarity between an orange and Bill Gates may not be critical.

**The analogy.** Back to the two maps. CKA asks both maps to agree on *every* pairwise distance. Mutual nearest neighbors only asks them to agree on *who your neighbors are*: weaker, but more robust.

**The mechanism.** Take $b$ paired items (an image and its caption, or the same item twice). Embed one side with model $f$, the other with model $g$. For each item, find its $k$ nearest neighbors in each space, count the shared neighbors, divide by $k$, and average.

![Mutual k-NN alignment: the same paired items are embedded by two models, each item's k nearest neighbors are found in each space, and the overlap fraction is averaged](/imgs/blogs/position-the-platonic-representation-hypothesis-1.webp)

**The math.** Let $\phi_i = f(x_i)$ and $\psi_i = g(y_i)$ be the features of pair $i$, with $\Phi = \{\phi_1, \dots, \phi_b\}$ and $\Psi = \{\psi_1, \dots, \psi_b\}$. The neighbor sets are

$$
S(\phi_i) = d_{\text{knn}}(\phi_i, \Phi \setminus \phi_i), \qquad S(\psi_i) = d_{\text{knn}}(\psi_i, \Psi \setminus \psi_i)
$$

where $d_{\text{knn}}$ returns the *indices* of the $k$ nearest neighbors. Returning indices rather than vectors is the trick: indices are shared across models, vectors are not. The per-item score is

$$
m_{\text{NN}}(\phi_i, \psi_i) = \frac{1}{k} \, \big| S(\phi_i) \cap S(\psi_i) \big|
$$

and the alignment between the two models is its mean over all $i$. It lives in $[0, 1]$: 1 means identical neighborhoods everywhere, and a random pair of models scores roughly $k / (b-1)$, the chance overlap (that baseline is my arithmetic, not a number the paper reports).

**A worked example.** Take $k = 4$ and a photo of a dog, item $a$. In the vision model, its four nearest images are $\{b, c, d, e\}$. In the language model, the four nearest captions to "a dog on grass" are $\{b, d, e, f\}$. The shared set is $\{b, d, e\}$, so $m_{\text{NN}} = 3/4 = 0.75$ for this item. Repeat for all 1,024 items and average.

In the actual experiments $k = 10$. Vision-vision alignment uses 1,000 images from the Places-365 validation set. Cross-modal alignment uses 1,024 image-caption pairs from WIT (Wikipedia-based Image Text), with features $l_2$-normalized before computing distances.

**Why it works, and when it fails.** The metric is rotation invariant, cheap and batchable. It is *not ordinal* (it ignores neighbor rank), and its absolute value depends on $k$ and $b$, which matters when judging whether 0.16 is "a lot". Across 78 vision models most alternative metrics rank-correlate highly with it; cross-modally, sensitivity varies by metric.

## Evidence 1: strong vision models are all alike

The paper first surveys prior evidence. Model stitching (Lenc and Vedaldi, 2015) plugs the early layers of one network into the later layers of another through a learned affine layer $h$: if $F = f_1 \circ \dots \circ f_k \circ h \circ g_{k+1} \circ \dots \circ g_m$ still works, the two networks are compatible at layer $k$.

The authors then test **78 vision models**: 17 ViTs (ImageNet-21k classification, MAE, DINO, CLIP), a random ResNet-50, and 60 contrastive ResNets on real and synthetic data. Competence is linear-probe accuracy on the **19 VTAB tasks**; a model "solves" a task at 80% or more of the best model's score. They bin models by tasks solved and measure alignment *within* each bin.

![Figure 2 from Huh et al. (2024): left, intra-bucket alignment rises sharply for models solving more VTAB tasks; right, a UMAP of models where competent ones cluster together](/imgs/blogs/position-the-platonic-representation-hypothesis-fig2.webp)

Read the left panel from left to right. Models that solve few tasks barely agree with each other, and the bars stay below 0.1. The two most competent buckets jump to roughly 0.3 and 0.4 (my reading of the plot). The right panel embeds models in 2D using distance $-\log(\text{alignment})$: the competent models (blue) sit in one tight cluster, while weak ones scatter.

The authors borrow Tolstoy for the takeaway: **all strong models are alike, each weak model is weak in its own way.** The platonic hypothesis adds that the shared "happy" representation reflects reality.

## Evidence 2: language models align with vision models

This is the headline experiment. Same-modality convergence could be explained by shared architectures or shared datasets. Cross-modal convergence cannot: an LLM trained on text has never seen a pixel.

**The setup.** Take WIT, where each item is an image $x_i$ from Wikipedia and its caption $y_i$. Build one kernel from a vision model and one from a language model:

$$
K_{\text{img}}(i, j) = \langle f_{\text{img}}(x_i), f_{\text{img}}(x_j) \rangle, \qquad K_{\text{text}}(i, j) = \langle f_{\text{text}}(y_i), f_{\text{text}}(y_j) \rangle
$$

The pairing makes the neighbor indices line up: index $i$ is "this image" on one side and "its caption" on the other. Language models (BLOOM, OpenLLaMA, LLaMA) are scored by ${1} - \text{bits-per-byte}$ on 4M OpenWebText tokens, which is comparable across tokenizers. Vision models are ViTs trained with ImageNet-21K classification, MAE, DINOv2, CLIP, and CLIP fine-tuned on ImageNet-12K.

![Figure 3 from Huh et al. (2024): alignment to DINOv2 grows roughly linearly with language-modeling score, for every DINOv2 size; smaller panels repeat the trend for MAE, ImageNet-21K, CLIP and CLIP fine-tuned](/imgs/blogs/position-the-platonic-representation-hypothesis-fig3.webp)

Three things to read off this figure:

| Observation | What it shows |
| --- | --- |
| Lines slope upward in every panel | Better language models align more with vision models |
| Larger vision models sit on higher lines | Better vision models align more with LLMs too |
| CLIP panels score higher, the fine-tuned CLIP lower | Language supervision helps; narrowing to ImageNet labels hurts |

The CLIP result is a nice control: fine-tuning CLIP on ImageNet makes it a better classifier but less aligned, which fits the multitask argument below (a narrower task tolerates more idiosyncratic solutions).

**One detail that matters.** Vision features are each layer's class token; LLM features are each layer's average-pooled tokens. Alignment is computed for every layer pair and the **maximum** is kept, with values above the 95th percentile clipped. The LLM's last token showed no strong signal. Keep this in mind for the critique.

## Evidence 3: alignment predicts downstream skill

If convergence is toward a more accurate model of reality, alignment should go with being good at things. Figure 4 tests that on a broader set of LLMs, adding OLMo, LLaMA3, Gemma, Mistral and Mixtral.

![Figure 4 from Huh et al. (2024): language models that align more with DINOv2 score higher on Hellaswag (roughly linear) and GSM8K (an emergence-like jump)](/imgs/blogs/position-the-platonic-representation-hypothesis-fig4.webp)

On Hellaswag (commonsense completion) the relationship looks linear, from about 0.3 for bloom-560m to about 0.7 for llama3-70b. On GSM8K (grade-school math) it looks like emergence: scores sit near zero until alignment reaches roughly 0.2, then rise steeply. Bubble size is model size.

This plot is the easiest to over-read. Both axes are driven by scale and data quality, so it shows co-variation, not that alignment *causes* skill.

A smaller experiment (the paper's Figure 9) varies caption density: LLaMA3-8B-Instruct summarizes long Densely Captioned Images captions, and denser captions align better with vision. A richer caption is closer to a *bijective* description of the image, the condition the theory below needs.

## Why would representations converge?

Section 3 starts from the ordinary training objective and asks which of its parts could push different models toward the same answer:

$$
f^* = \arg\min_{f \in \mathcal{F}} \; \mathbb{E}_{x \sim \text{dataset}} \big[ \mathcal{L}(f, x) \big] + \mathcal{R}(f)
$$

Here $f^*$ is the trained model, $\mathcal{F}$ is the function class (the architecture and its size), $\mathcal{L}(f, x)$ is the training objective on example $x$, and $\mathcal{R}(f)$ is regularization, explicit or implicit. The paper attaches one hypothesis to each colored piece.

![Three pressures toward one representation: task generality acts on the objective and data, capacity on the function class, and simplicity bias on the regularizer; each narrows the set of good solutions](/imgs/blogs/position-the-platonic-representation-hypothesis-2.webp)

### Pressure 1: task generality

**Intuition.** Every task is a constraint. Many representations can tell cats from dogs. Far fewer can tell cats from dogs, count objects, estimate depth, and caption the scene all at once. Stack enough constraints and the feasible set shrinks.

The paper calls this the **Multitask Scaling Hypothesis**: fewer representations are competent for $N$ tasks than for $M$ tasks when $M \lt N$. Modern objectives are secretly multitask: contrastive learning serves many classification tasks at once, masked autoencoders solve random reconstruction tasks, and next-token prediction mixes countless tasks. This echoes Cao and Yamins' "Contravariance principle": an easy goal has many solutions, a hard goal has few.

**Micro-example.** Suppose each task independently rules out half of all candidate representations (a toy assumption of mine, not the paper's). After one task, 50% survive; after 10 tasks, ${0.5^{10}} \approx 0.1\%$ survive. Two models trained on the same 10 tasks are drawing from a pool a thousand times smaller than two models trained on one task.

### Pressure 2: model capacity

**The Capacity Hypothesis**: bigger models are more likely to converge to a shared representation than smaller models. Suppose a globally optimal representation exists for a family of objectives. A small model's hypothesis space may not contain it, so two small models settle on two different compromises. Larger hypothesis spaces are more likely to cover the optimum, so two large models, even of different architectures, both find it.

**When it fails.** Coverage is necessary, not sufficient. A bigger space also contains *more* ways to fit the training data. That gap is what the third pressure is meant to close.

### Pressure 3: simplicity bias

**The Simplicity Bias Hypothesis**: deep networks are biased toward simple fits to the data, and the bigger the model, the stronger the bias. So as models grow, they should converge to a smaller solution space.

A billion-parameter model *could* detect a dog with a baroque, idiosyncratic feature set. It does not, the argument goes, because weight decay, dropout and the implicit bias of gradient descent favor simple solutions.

**When they fail.** None of the three is proven in the paper. They come from a literature survey with schematic figures, not experiments that isolate each one, and they interact: capacity widens the space, simplicity narrows it.

## The math: contrastive learners recover PMI

Section 4 asks what the endpoint is, and gives what the authors call "just one concrete candidate" in a toy world.

### An idealized world

**The setup.** The world is a sequence of $T$ discrete events $\mathbf{Z} = [z_1, \dots, z_T]$ drawn from an unknown distribution $\mathbb{P}(\mathbf{Z})$. An observation function $\text{obs}$ maps each event into a measurement space (pixels, sounds, words), and the critical assumption is that it is **bijective and deterministic**: no information lost, no noise.

**Co-occurrence.** Two observations $x_a$ and $x_b$ co-occur if they appear within a time window $T_{\text{window}}$ of each other:

$$
P_{\text{coor}}(x_a, x_b) \;\propto \sum_{(t, t') \,:\, |t - t'| \le T_{\text{window}}} \mathbb{P}(X_t = x_a, X_{t'} = x_b)
$$

Here $X_t$ is the observation at time $t$. It counts how often two things show up close together, like "sand" and "beach" in a paragraph. It is symmetric.

### What a contrastive learner converges to

**The problem the learner solves.** Positive pairs are sampled from $P_{\text{coor}}$; negative pairs are drawn independently from the marginal. An encoder $f_X: \mathcal{X} \to \mathbb{R}^d$ is trained so the dot product classifies a pair as positive or negative: the NCE family, including SimCLR and SimCSE.

**The mechanism.** A good binary classifier outputs the log-odds that a pair is positive. So at the optimum, the dot product approximates that log-odds, up to an offset:

$$
\begin{aligned}
\langle f_X(x_a), f_X(x_b) \rangle &\approx \log \frac{\mathbb{P}(\text{pos} \mid x_a, x_b)}{\mathbb{P}(\text{neg} \mid x_a, x_b)} + \tilde{c}_X(x_a) \\
&= \log \frac{P_{\text{coor}}(x_a \mid x_b)}{P_{\text{coor}}(x_a)} + c_X(x_a) \\
&= K_{\text{PMI}}(x_a, x_b) + c_X(x_a)
\end{aligned}
$$

where $K_{\text{PMI}}$ is the **pointwise mutual information** kernel, $\log \frac{P(x_a, x_b)}{P(x_a) P(x_b)}$, and $c_X(x_a)$ is a term that does not depend on $x_b$. Because the left-hand side is symmetric in $a$ and $b$, the offset $c_X$ must actually be a constant. So the learner's kernel *is* PMI, up to a constant.

PMI measures how much more often two things co-occur than chance would predict. Positive PMI means "these go together," zero means "unrelated," negative means "these avoid each other."

**Worked example.** Here is the binary NCE case from the appendix, with numbers. Let $p_{\text{pos}}$ be the probability of sampling a positive pair. The loss is minimized when the score $g$ equals

$$
g^*(x_a, x_b) = \log \frac{p_{\text{pos}} \, P_{\text{coor}}(x_a, x_b)}{(1 - p_{\text{pos}}) \, P(x_a) P(x_b)} = K_{\text{PMI}}(x_a, x_b) + \log \frac{p_{\text{pos}}}{1 - p_{\text{pos}}}
$$

Take $P_{\text{coor}}(\text{sand}, \text{beach}) = 0.10$, $P(\text{sand}) = 0.20$, $P(\text{beach}) = 0.25$. Chance co-occurrence would be ${0.20 \times 0.25} = 0.05$, so PMI $= \log(0.10/0.05) = \log 2 \approx 0.69$. With balanced sampling, $p_{\text{pos}} = 0.5$, the offset is $\log 1 = 0$ and the optimal dot product is ${0.69}$. With $p_{\text{pos}} = 0.2$, every score shifts down by $\log(0.25) \approx -1.39$, yet every *difference* between pairs is unchanged. The offset is the "$c_X$" above.

### From one modality to all of them

Now the payoff. A bijective observation of a discrete variable preserves probabilities. So co-occurrence in observation space equals co-occurrence in event space, and the same holds for the PMI kernel:

$$
P_{\text{coor}}(x_a, x_b) = P_{\text{coor}}(z_a, z_b), \qquad K_{\text{PMI}}(x_a, x_b) = K_{\text{PMI}}(z_a, z_b)
$$

Run the same argument for a second modality $Y$ and you get

$$
K_{\text{PMI}}(z_a, z_b) = \langle f_X(x_a), f_X(x_b) \rangle - c_X = \langle f_Y(y_a), f_Y(y_b) \rangle - c_Y
$$

Two encoders, trained on different modalities, end up with the same kernel up to a constant: a kernel that encodes pairwise statistics of the world itself.

![Same world, same co-occurrence, same kernel: two bijective observations of the events Z give identical co-occurrence statistics, so contrastive learners on either modality recover the same PMI kernel](/imgs/blogs/position-the-platonic-representation-hypothesis-3.webp)

The paper boils this down to a slogan: certain representation learners reduce to "find an embedding in which similarity equals PMI."

**Can a network represent PMI exactly?** Only if $K_{\text{PMI}} + C$ is positive semi-definite for some constant $C$. Appendix Proposition F.1 shows this holds for a smooth enough world, where off-diagonal PMI values sit in a narrow band and a suitable constant makes the matrix diagonally dominant. The authors call this "somewhat strict" and note exactness is not needed for convergence.

**When it fails.** The whole chain rests on bijective, deterministic observations of a discrete world. A caption is not a bijection of an image. Once observations are lossy or stochastic, $P_{\text{coor}}(x_a, x_b) = P_{\text{coor}}(z_a, z_b)$ no longer holds, and the argument says nothing about how far apart the kernels can drift.

### A study in color

The authors test the idea on colors, building four 3D color layouts.

![Figure 8 from Huh et al. (2024): color embeddings from human perception (CIELAB), from pixel co-occurrence PMI in CIFAR-10, and from two language models (SimCSE and RoBERTa) form similar layouts](/imgs/blogs/position-the-platonic-representation-hypothesis-fig8.webp)

| Panel | Source of the color similarity |
| --- | --- |
| Perception | CIELAB coordinates, designed so distances match human perception |
| Vision | PMI of pixel colors co-occurring within 4 pixels, 300,000 CIFAR-10 samples |
| Language (contrastive) | SimCSE embeddings of "The color of this thing is ⟨color⟩" |
| Language (predictive) | RoBERTa-L, last four hidden layers concatenated |

Each dissimilarity matrix becomes a 3D layout via multidimensional scaling, aligned onto CIELAB for display. Pixel statistics and the statistics of 20 color words both recover roughly the perceptual color wheel: the PMI story in miniature.

## Implications the authors draw

The paper's consequences, with my read on each:

| Implication | My read |
| --- | --- |
| Scaling is sufficient, not efficient | Reasonable; restates scaling-law practice |
| Data can be shared across modalities (a pixel is worth some number of words) | The most actionable idea |
| Translation across modalities gets easier | Consistent with linear stitching of vision into LLMs |
| Hallucination and bias may drop with scale | Conditional on lossless, diverse data, as the authors flag |

The modality-sharing claim is where this paper has aged best. Later work like [vec2vec](/blog/paper-reading/ai-interpretability/harnessing-the-universal-geometry-of-embeddings) pushed the idea further by translating between text embedding spaces with *no* paired data at all, which only works if the spaces share geometry.

## Counterexamples and limitations, in the authors' own words

The paper is unusually candid about where its hypothesis could break.

- **Modalities hold different information.** Text cannot fully convey a solar eclipse; an image cannot convey "I believe in the freedom of speech." Their refinement: convergence is capped by the mutual information between signals and by model capacity.
- **Not every domain is converging.** Robotics has no standard state representation, partly because hardware makes data slow and expensive.
- **Sociological bias.** We build systems to mimic humans, and the "hardware lottery" favors designs that fit available chips.
- **Special-purpose systems may not converge.** A protein-folding model and a lane follower may share little.
- **Lots left to explain.** In Figure 3, alignment reaches only about 0.16 out of a maximum of 1. Noise, or real disagreement? The authors leave it open.

## Results at a glance

| Experiment | Models | Data | Finding |
| --- | --- | --- | --- |
| Vision-vision (Fig. 2) | 78 vision models | 1,000 Places-365 images, 19 VTAB tasks | Competent models cluster |
| Language-vision (Fig. 3) | BLOOM, OpenLLaMA, LLaMA vs. 5 ViT families | 1,024 WIT pairs | Linear in LM score, peaks near 0.16 |
| Alignment vs. skill (Fig. 4) | adds OLMo, LLaMA3, Gemma, Mistral | Hellaswag, GSM8K | Linear vs. emergence-like |
| Caption density (Fig. 9) | all of the above | Densely Captioned Images | Denser captions align better |
| Color (Fig. 8) | SimCSE, RoBERTa, pixel PMI | CIFAR-10, 20 color words | Same perceptual layout |

**What is load-bearing and might not transfer.** First, the max over layer pairs: a search over many noisy scores biases the result upward. Second, average-pooled LLM features: the last token gave no strong signal, so the effect depends on the readout. Third, WIT captions are loosely descriptive, which per the caption-density result probably *lowers* alignment, cutting in the paper's favor.

## Critique

**What is strong.** The framing is clean and testable. The cross-modal experiment is the right test, because shared architecture and data cannot explain it. The PMI derivation turns a slogan into a precise statement, and the color study is a reproducible sanity check.

**What is weak.**

- **Scale does most of the lifting.** Alignment, LM score, Hellaswag and GSM8K all rise with size and data quality. "Bigger models have richer features" explains the curves as well as "models converge to one target." A missing control: alignment between large models and a strong but *unrelated* representation.
- **Small absolute numbers, no anchors.** The paper reports neither a chance baseline nor a ceiling (say, two seeds of one vision model on WIT). My chance estimate, $k/(b-1) \approx 10/1023 \approx 0.01$, says 0.16 is real signal, but without the ceiling we cannot say how converged it is.
- **The theory covers a world that is not ours.** Bijective observations of discrete events exclude exactly what the experiments use: captions and lossy sensors.
- **The three pressures are a lens, not a decomposition.** No experiment isolates one.

**Missing ablation.** Alignment per layer pair, without the max, to see whether the selection step is doing the work.

**What would change my mind.** I would drop the hypothesis's strong form if, with chance and ceiling baselines in place, the alignment of frontier vision and language models plateaued well below the seed-to-seed ceiling while their benchmark scores kept climbing. I would also count against it a demonstration that two equally capable models trained on *disjoint* broad datasets reach low mutual alignment. Conversely, if a clean ceiling showed that 0.16 already sits near the noise limit of WIT captions, I would move toward the strong form.

For a practitioner the takeaway is narrower than the title: strong models' spaces are similar enough that cheap maps between them often work. That is why stitching, linear adapters and [contrastive objectives that learn PMI-like structure](/blog/machine-learning/recommendation-systems/sampled-softmax-and-contrastive-losses-for-retrieval) transfer so well, and why building diffusion models on pretrained encoders, as in representation autoencoders for diffusion transformers, pays off (that connection is mine).

## References

- Huh, M., Cheung, B., Wang, T., and Isola, P. (2024). *Position: The Platonic Representation Hypothesis*. ICML 2024. [arXiv:2405.07987](https://arxiv.org/abs/2405.07987)
- Project page: [phillipi.github.io/prh](https://phillipi.github.io/prh)
- Code: [github.com/minyoungg/platonic-rep](https://github.com/minyoungg/platonic-rep)
- Related on this site: [vec2vec: translating between embedding spaces with no paired data](/blog/paper-reading/ai-interpretability/harnessing-the-universal-geometry-of-embeddings)
- Related on this site: [Sampled softmax and contrastive losses for retrieval](/blog/machine-learning/recommendation-systems/sampled-softmax-and-contrastive-losses-for-retrieval)
- Related on this site: Diffusion transformers with representation autoencoders
