---
title: "Your Large Vision-Language Model Only Needs A Few Attention Heads For Visual Grounding"
publishDate: "2026-10-16"
date: "2026-10-16"
description: "How three attention heads inside a frozen LLaVA already know where the referred object is, how the paper finds them with two cheap criteria, and how far training-free grounding gets on RefCOCO."
category: "paper-reading"
subcategory: "Computer Vision"
tags: ["interpretability-analysis", "cvpr-2025", "cvpr-2025-highlight", "paper-reading", "visual-grounding", "localization-heads", "vision-language-models", "attention-maps", "training-free", "llava"]
author: "Hiep Tran"
featured: true
readTime: 19
image: "/imgs/blogs/your-large-vision-language-model-only-needs-a-few-attention-heads-for-visual-grounding-fig1.webp"
excerpt: "A few attention heads in frozen LVLMs already localize the object a sentence refers to. Find them once, sum three maps, draw a box, and you get grounding that rivals fine-tuned models with zero training."
paper:
  title: "Your Large Vision-Language Model Only Needs A Few Attention Heads For Visual Grounding"
  authors: "Seil Kang, Jinyeong Kim, Junhyeok Kim, Seong Jae Hwang"
  venue: "CVPR 2025 (highlight)"
  url: "https://arxiv.org/abs/2503.06287"
---

> [!tldr]
> **Claim.** Inside a frozen Large Vision-Language Model (LVLM) such as LLaVA-1.5, a handful of attention heads already point at the image region a sentence refers to. The authors call them **localization heads**.
>
> **Why it matters.** Find those heads once with two cheap statistics (how much a head looks at the image, and how compact its attention blob is), then sum the maps of the top three. You get a bounding box, and with SAM a mask, without training a single weight.
>
> **Most surprising result.** On RefCOCO, LLaVA-1.5-13B with three heads reaches 87.2 Acc@0.5 on REC val, in the same range as Shikra-13B (87.8) and Ferret-13B (89.5), which were fine-tuned for exactly this task. Asking the same frozen model to print box coordinates gets 2.92 on RefCOCOg.
>
> **Where it fails.** Counting and ordinal phrases ("third from right") smear across neighbours, small LVLMs trail trained decoders like F-LMM by up to 19 cIoU, and models that pool visual tokens lose the spatial grid the method depends on.

Most work that turns an LVLM into a grounding model bolts something on. LISA adds a `[SEG]` token and a mask decoder. Shikra and Ferret fine-tune the language model to emit coordinates. All of them need grounding data and a training run.

This paper asks a narrower question first: if the model can *describe* "the left white horse next to the black one", it must be attending to that horse somewhere. Can we just read the attention out? The average attention map says no. A few specific heads say yes, loudly.

## The puzzle: average attention is noise

Vision Transformers and diffusion models are famous for attention maps you can almost frame on a wall. The natural first move with an LVLM is the same: take the text-to-image attention, average it over layers and heads, and look.

![Figure 1 from Kang et al. (2025): ground truth, the averaged text-to-image attention, and two individual heads (L14 H24, L14 H13) of LLaVA-1.5-7B for three referring expressions](/imgs/blogs/your-large-vision-language-model-only-needs-a-few-attention-heads-for-visual-grounding-fig1.webp)

The second column of Figure 1 is what the average gives you: sparse dots that have nothing to do with the pizza mouth or the guy on the right. The third and fourth columns are two single heads in layer 14 of LLaVA-1.5-7B. They light up the referred object across all three samples, including the tricky "girl on outer seat" case.

So the signal exists, but averaging over roughly a thousand heads drowns it. The paper is a recipe for finding the needles and a demonstration that they are enough.

## Where the attention row comes from

**The problem.** We need one attention vector per head that answers "where in the image does this whole sentence look?" A decoder-only LLM produces one attention row per token, so we have to pick a query.

**The analogy.** In a causal decoder, information flows left to right, so the last token of the prompt has seen everything. It is like the last person in a relay who holds the full baton. The paper uses that last token's query as the sentence summary.

**The mechanism.** An LVLM turns the image $X_v$ into a sequence of visual embeddings with a vision encoder and a projector, and the text $X_t$ into token embeddings. The two sequences are concatenated and fed to the LLM:

$$
Z_v \in \mathbb{R}^{P^2 \times d}, \qquad Z_t \in \mathbb{R}^{L \times d}, \qquad Z^0 = [Z_v ; Z_t] \in \mathbb{R}^{(P^2 + L) \times d}
$$

Here $P^2$ is the number of flattened image tokens (a $P \times P$ patch grid), $L$ is the number of text tokens and $d$ is the hidden size. For LLaVA-1.5, $P = 24$, so there are 576 image tokens.

At layer $\ell$ and head $h$, the hidden state is projected to queries $Q$, keys $K$ and values $V$, each of shape $(P^2 + L) \times d_h$, where $d_h$ is the per-head dimension. The paper takes only the query row of the last text token, $q_{\text{txt}} \in \mathbb{R}^{1 \times d_h}$:

$$
a^{\ell,h} = \operatorname{softmax}\!\left(\frac{q_{\text{txt}} K^\top}{\sqrt{d_h}}\right) \in \mathbb{R}^{P^2 + L}
$$

Only the first $P^2$ entries, $a^{\ell,h}[1 : P^2]$, are attention to image tokens. Reshape them row by row into a $P \times P$ map $A^{\ell,h}$ and you have a heatmap over the patch grid. The paper names a head by position: L14 H24 is head 24 of layer 14.

![A 576-token image strip and the text tokens, the last token's attention row, and the first 576 weights folded into a 24 by 24 map](/imgs/blogs/your-large-vision-language-model-only-needs-a-few-attention-heads-for-visual-grounding-1.webp)

**Tiny example.** Shrink the image to $P = 2$ (4 image tokens) and the text to $L = 3$. Suppose one head gives $a = [0.05, 0.30, 0.05, 0.10, 0.20, 0.10, 0.20]$. The image part is the first four numbers, which reshape to the ${2} \times 2$ map with rows $[0.05, 0.30]$ and $[0.05, 0.10]$: this head looks mostly at the top-right patch.

**When it fails.** The trick assumes the image tokens sit in a known raster order. Models that pool, resample or compress visual tokens (the appendix names Q-Former style and pooling designs) break the reshape, and the authors leave those for future work. The first two LLM layers are also excluded throughout, because early layers are known to behave differently.

## Criterion 1: does the head even look at the image?

**The problem.** Softmax forces each row to sum to one, so every head attends *somewhere*. Many heads spend almost all of that budget on text tokens or on a few sink tokens. A head that barely looks at the image cannot localize anything in it.

**The analogy.** Before asking where someone is looking in a photo, check whether they are looking at the photo at all or at the caption underneath.

**The math.** The attention sum is the total probability mass a head puts on image tokens:

$$
S^{\ell,h}_{\text{img}} = \sum_{i=1}^{P^2} a^{\ell,h}[i], \qquad 0 \le S^{\ell,h}_{\text{img}} \le 1
$$

In the tiny example above, $S_{\text{img}} = 0.05 + 0.30 + 0.05 + 0.10 = 0.50$: half of that head's attention goes to the image.

The authors average $S_{\text{img}}$ for every head over 1,000 random RefCOCO *training* samples, sort heads in ascending order, and plot the curve. Most heads sit near zero. The threshold $\tau$ is placed at the point of maximum curvature of that sorted curve, which is $\tau = 0.24$ for LLaVA-1.5-7B. Heads with $S_{\text{img}} \ge \tau$ survive.

![Schematic hockey-stick curve of average attention sum over 1,024 heads sorted ascending, with the dashed threshold at 0.24](/imgs/blogs/your-large-vision-language-model-only-needs-a-few-attention-heads-for-visual-grounding-2.webp)

The curve in the figure is a schematic of the shape the paper reports in its Figure 3, not a replot of their data; the 0.24 threshold is theirs. The point is the shape: a long flat tail of text-attending heads and a short steep tail of image-attending ones. The appendix shows the same selected heads come out across a range of $\tau$ values, so the elbow choice is a convenience rather than a knob.

**When it fails.** A high attention sum says "this head looks at the image", not "this head looks at the right part". A head that spreads its mass evenly across all 576 patches passes Criterion 1 and is useless for grounding. That is what Criterion 2 is for.

## Criterion 2: is the attention a compact blob?

**The problem.** Among image-attending heads, we want those whose attention forms one tight cluster, because objects occupy contiguous patches.

**The analogy.** Two flashlights put out the same total light. One makes a single bright disc, the other a scatter of dots. Spatial entropy measures which one you are holding.

**The mechanism.** Binarize the map $A^{\ell,h}$: cells above the map's mean become 1, the rest 0. Find the connected components $\{C_i\}_{i=1}^{N}$ of the 1-cells using 8-neighbour connectivity (diagonals count). Treat each component's share of the lit area as a probability and take its Shannon entropy:

$$
H(A^{\ell,h}) = -\sum_{i=1}^{N} P(C_i) \log P(C_i), \qquad P(C_i) = \frac{|C_i|}{\sum_{j=1}^{N} |C_j|}
$$

$|C_i|$ is the number of cells in component $i$. One single blob gives $N = 1$, $P(C_1) = 1$ and $H = 0$. Many similar-sized fragments push $H$ up.

![Two binarized 8 by 8 grids with 12 lit cells each: one 3 by 4 blob with entropy 0, and five fragments of 4, 3, 2, 2, 1 cells with entropy about 1.52](/imgs/blogs/your-large-vision-language-model-only-needs-a-few-attention-heads-for-visual-grounding-3.webp)

**Worked example.** Both grids above light up exactly 12 cells, so Criterion 1 cannot tell them apart. The left one is a single ${3} \times 4$ blob, so $H = 0$. The right one splits into components of 4, 3, 2, 2 and 1 cells, giving probabilities $\tfrac{4}{12}, \tfrac{3}{12}, \tfrac{2}{12}, \tfrac{2}{12}, \tfrac{1}{12}$:

$$
H = -\left(0.333 \ln 0.333 + 0.25 \ln 0.25 + 2 \cdot 0.167 \ln 0.167 + 0.083 \ln 0.083\right) \approx 0.366 + 0.347 + 0.597 + 0.207 \approx 1.52
$$

Same light, very different entropy. Note that the measure ignores *where* the blob is and how big it is; it only rewards being in one piece.

A small detail worth flagging: the appendix says $H$ "ranges from 0 to 1". With natural logs and five components, the raw formula above already exceeds 1, so either they normalize (for example by $\log N$) or the statement is loose. For ranking heads the scale does not matter, only the order.

**When it fails.** A head can form a crisp single blob on the *wrong* thing: the most salient object, an attention sink in a corner, or a fixed image location. Compactness is necessary, not sufficient. The paper's own ablation (below) shows that picking the lowest-entropy heads per sample, without the frequency step, gets this exact failure.

## Selection frequency: which heads are good *consistently*

The two criteria score a head on one sample. The paper's key move is to turn per-sample scores into a population statistic.

For each of 1,000 RefCOCO training pairs, keep only heads with $S_{\text{img}} \ge \tau$, then record the 10 with the lowest spatial entropy. A head's **selection frequency** is the fraction of samples in which it made that top-10 list. Heads are ranked by frequency, and the top $k$ become the localization heads for that model, fixed forever after.

![Figure 5 from Kang et al. (2025): collect all attention maps, discard heads below the attention-sum threshold, select the 10 lowest-entropy heads per sample, and count selection frequency across samples](/imgs/blogs/your-large-vision-language-model-only-needs-a-few-attention-heads-for-visual-grounding-fig5.webp)

As an illustration (my numbers, not the paper's): if L14 H24 is among the 10 lowest-entropy image-attending heads in 620 of 1,000 samples, its frequency is 62%. A head that forms a crisp blob only on images with one big central object might score 5%. Frequency rewards heads that are compact *across* scene types, which is a proxy for "compact because it is following the text".

Is that proxy valid? The authors check it directly. They binarize each head's map into a pseudo-mask, compute IoU against the ground-truth mask, and correlate a head's frequency rank with its average IoU.

![Figure 6 from Kang et al. (2025): (a) selection frequency per head for LLaVA-1.5-7B, LLaVA-1.5-13B, DeepSeek-VL-1.3B and DeepSeek-VL-7B, and (b) frequency rank against average IoU](/imgs/blogs/your-large-vision-language-model-only-needs-a-few-attention-heads-for-visual-grounding-fig6.webp)

Panel (a) is the headline: in every model only a few heads have high selection frequency, and the tail drops off fast. Panel (b) shows that rank predicts IoU, with Spearman correlation above 0.7 for all four LVLMs ($p \lt 0.001$). The relationship is sharpest among the top-ranked heads.

For LLaVA-1.5-7B, the winner is L14 H24 and the runner-up is L14 H13, the same two heads from Figure 1. The appendix shows the same top-3 heads survive any choice of "top-N lowest entropy" from N = 1 to 14, and the pattern repeats across InternVL, LLaVA, Mini-Gemini, ShareGPT4V and Yi-VL. That consistency is what licenses the paper's claim that localization heads are an innate property of LVLMs.

## From three maps to a box and a mask

Once the heads are known, grounding a new image-text pair is a single forward pass plus some classic image processing.

1. Run the frozen LVLM on the image and the referring expression.
2. Pull the $P \times P$ maps from the top $k = 3$ localization heads.
3. Smooth each with a Gaussian (kernel $\kappa = 7$, $\sigma = 1.0$) to suppress isolated hot cells.
4. Sum them element-wise into one combined map.
5. Binarize the combined map at its mean.
6. Run a convex hull over the lit regions, keep the largest hull, and take the tightest box that encloses it.
7. For segmentation, feed that box to SAM as a prompt.

In symbols, with $G_\sigma$ the Gaussian filter and $\mathcal{T}$ the set of top-$k$ heads:

$$
\bar{A} = \sum_{(\ell,h) \in \mathcal{T}} G_\sigma * A^{\ell,h}, \qquad M[u,v] = \mathbb{1}\!\left[\bar{A}[u,v] > \operatorname{mean}(\bar{A})\right]
$$

$M$ is the binary pseudo-mask on the patch grid, and $u, v$ index rows and columns. The box is the bounding rectangle of the largest convex hull of $M$, scaled from the ${24} \times 24$ grid back to pixel coordinates.

![Three 6 by 6 head maps with stray hot cells, smoothed and summed so the stray cells fade, binarized, outlined with a convex hull and tight box, then passed to SAM](/imgs/blogs/your-large-vision-language-model-only-needs-a-few-attention-heads-for-visual-grounding-4.webp)

**Why summing works.** Each head carries the object plus its own idiosyncratic noise. The object cells line up across heads and add constructively; a stray hot cell in one head gets only one vote. This is ensemble averaging with three members, and the Gaussian blur makes the votes overlap even when the heads disagree by a patch.

**How much the blur matters.** Less than you might think. Without smoothing ($\sigma = 0$), LLaVA-1.5-13B scores 85.5 REC and 74.3 RES on RefCOCO val, against 87.2 and 76.1 with the default. The paper calls smoothing an auxiliary post-processing step, and the ablation backs that up.

**When it fails.** The box is the largest connected hull, so two separated instances of the referred class are a problem: the method returns one of them. And the ${24} \times 24$ grid means a patch is about 14 pixels at 336 px input, so thin or tiny objects get coarse boxes that SAM has to rescue.

## Results: training-free, close to fine-tuned

The evaluation uses RefCOCO, RefCOCO+ and RefCOCOg for referring expression comprehension (REC, Acc@0.5) and segmentation (RES, cIoU), plus ReasonSeg for queries that need world knowledge. Ten LVLMs from 1.3B to 13B are tested, all with $k = 3$, all frozen, on a single RTX A6000.

![Figure 8 from Kang et al. (2025): REC boxes and RES masks from Shikra-13B, ReCLIP, LISA-13B, ZS-RS and the localization-head method on three LVLMs, for expressions with similar objects and spatial relations](/imgs/blogs/your-large-vision-language-model-only-needs-a-few-attention-heads-for-visual-grounding-fig8.webp)

The qualitative grid shows the cases that trip CLIP-based training-free methods: "fourth remote from left", "doughnut on left second from front". CLIP scores whole crops against the sentence and is weak at orientation words, so earlier training-free work had to split the expression or add spatial post-processing. The LVLM reads the whole sentence natively.

### Headline numbers (RefCOCO val)

| Method | Training | REC Acc@0.5 | RES cIoU |
| --- | --- | --- | --- |
| GroundVLP (best prior training-free, REC) | none | 65.0 | n/a |
| Ref-Diff (best prior training-free, RES) | none | n/a | 35.2 |
| Shikra-13B | fine-tuned LVLM | 87.8 | n/a |
| Ferret-13B | fine-tuned LVLM | 89.5 | n/a |
| LISA-13B | fine-tuned LVLM + decoder | n/a | 73.4 |
| GSVA-13B | fine-tuned LVLM + decoder | n/a | 77.7 |
| DeepSeek-VL-1.3B + 3 heads | none | 73.2 | 56.3 |
| LLaVA-1.5-7B + 3 heads | none | 86.5 | 74.2 |
| LLaVA-1.5-13B + 3 heads | none | 87.2 | 76.1 |

Three things stand out. Against other training-free methods the jump is enormous: 65.0 to 87.2 on REC and 35.2 to 76.1 on RES. Against fine-tuned LVLMs that share LLaVA-1.5's base LLM, the frozen model is within a point or two. And the gap widens on RefCOCO+ testB (74.0 vs Ferret-13B's 75.2 is close, but UNINEXT's 79.8 is not), so dedicated grounders still win on the hardest split.

On ReasonSeg, LLaVA-1.5-13B with localization heads reaches 60.5 overall on val against LISA-13B's 60.3, and 49.9 on test against 50.8. That is parity with a model trained on reasoning segmentation, from a model that never saw it.

The model-scaling trend is clean: 1.3B to 13B improves monotonically within a family, and LLaVA to LLaVA-1.5 improves at fixed size. The authors read this as grounding ability growing with scale and data, which is plausible but rests on one family's comparison.

### The ablation that justifies the design

| Criteria used | Selection | REC | RES |
| --- | --- | --- | --- |
| attention sum only | greedy per sample | 23.7 | 18.8 |
| entropy only | greedy per sample | 29.8 | 21.5 |
| both | greedy per sample | 67.4 | 63.8 |
| attention sum only | fixed by frequency | 23.9 | 19.3 |
| entropy only | fixed by frequency | 31.3 | 25.7 |
| both | fixed by frequency | **87.2** | **76.1** |

(LLaVA-1.5-13B, RefCOCO val.) Either criterion alone is near useless. Both criteria with per-sample greedy picking gets 67.4; the same criteria with the fixed, frequency-ranked heads gets 87.2. The 20-point gap is the evidence that per-sample compactness finds blobs, while frequency finds heads that follow the *text*.

### How many heads?

![Average RES cIoU over 10 LVLMs against the number of combined localization heads k, peaking at k = 3, with LLaVA-1.5-13B as a second line](/imgs/blogs/your-large-vision-language-model-only-needs-a-few-attention-heads-for-visual-grounding-5.webp)

Averaged over all ten LVLMs on RefCOCO val RES, cIoU goes 64.5, 66.0, 67.1, 65.4, 58.9 for $k = 1$ to 5. Even a single head gets 64.5. Three is the sweet spot for almost every model, despite very different architectures and total head counts. Going from three heads to five costs about 8 points on average, which says the fourth- and fifth-ranked heads are not just weaker localizers but add noise.

### What is load-bearing and might not transfer

- **The raster grid.** Everything assumes a fixed $P \times P$ token layout. Dynamic-resolution models and token-merging designs need a mapping back to image coordinates first.
- **RefCOCO-tuned head selection.** The heads are chosen on 1,000 RefCOCO training images. Transfer to ReasonSeg is encouraging, but domain shifts (documents, medical, aerial) are untested.
- **SAM for masks.** RES quality is partly SAM's. The heads give a box; SAM turns a decent box into a good mask.
- **Single-object assumption.** The largest-hull rule returns one region. The appendix shows a multi-object variant using spaCy noun extraction (66.7 vs F-LMM's 65.7 on PNG with DeepSeek-VL-7B), but it is not a formal pipeline.

## Where it breaks: ordinals and small models

The paper shows its own failure case. For "third from right" on a bunch of bananas, the LLaVA-1.5-13B localization head (L15 H39) lights up both the third and fourth bananas, and the box lands on the wrong one. The attention map makes the failure legible: the model knows roughly where to look but cannot count precisely.

Small models are the other weak spot. Against F-LMM, which also reads frozen LVLM attention but trains a mask decoder on top of all heads, DeepSeek-VL-1.3B gets 56.3 vs 75.0 cIoU on RefCOCO val. At 7B the gap nearly closes: LLaVA-1.5-7B is at 74.2 vs 75.2. My reading: in small models the localization signal is spread across more heads, and a trained decoder can integrate it where three fixed heads cannot.

The most revealing number in the appendix is the direct-output baseline. Ask LLaVA-1.5-7B to *write* box coordinates for RefCOCOg and it scores 2.92. Read three of its heads and it scores 82.3. The knowledge is inside; the text interface simply never learned to report it.

## Critique

**What is strong.** The method is almost embarrassingly simple, which is the point: no learned parameters, one forward pass, a fixed list of three head indices per model. The selection-frequency statistic is a genuinely good idea, and the greedy-vs-fixed ablation proves it matters. Validating the heads against ground-truth IoU with a rank correlation, rather than eyeballing heatmaps, is the right standard of evidence for an interpretability claim.

**What is weak.**

- **No causal test.** The paper shows the heads *correlate* with the referred region. It never ablates or patches them to see if the model's text answers degrade. "Localization head" is a readout claim, not yet a mechanism claim.
- **Selection uses ground-truth-shaped data indirectly.** Selection itself is unsupervised (attention sum and entropy, no labels), which is fair. But the choice of $k = 3$ and the comparison of settings are reported on validation splits, so some tuning happened in sight of the benchmark.
- **Comparisons mix training regimes.** Some RES baselines (LISA, GSVA) are not trained on the corresponding RefCOCO splits, and the training-free baselines are CLIP or diffusion based rather than other LVLM readouts. A stronger baseline is "all heads, averaged, then same post-processing", which is Figure 1's second column and is never scored in a table.
- **Spatial entropy is a blunt instrument.** It ignores blob size and position, and the 0 to 1 range stated in the appendix does not match the formula as written.

**What would change my mind.** If zero-ablating the three localization heads left LLaVA-1.5's answers on spatial questions (GQA relations, "what is to the left of") essentially unchanged, I would read these heads as passive correlates rather than the model's grounding machinery. Conversely, a clear drop on spatial QA with no drop on non-spatial QA would make the paper's "innate property" framing much stronger than its current evidence supports.

## Where this goes next

The orchestrating question for follow-up work: now that we know *which* three heads carry the "where" signal, what else can they steer or score? Each idea below is my extrapolation, not something the paper does.

**1. Hallucination detection from head emptiness.** *Question:* when an LVLM names an object that is not in the image, do the localization heads fail to form a compact blob for that noun? *Why now:* the paper gives fixed head indices and a cheap compactness score per token, so you can score every generated noun for free. *First experiment:* on POPE or CHAIR, compute spatial entropy and attention sum of the top-3 heads for each object noun the model emits; measure AUROC for separating hallucinated from grounded nouns, then use it to trigger regeneration.

**2. Grounding-guided visual token pruning.** *Question:* can the combined localization map decide which of the 576 visual tokens to keep in later layers? *Why now:* token pruning work (see the [sibling post on visual-token sparsity](/blog/paper-reading/computer-vision/what-do-visual-tokens-really-encode-uncovering-sparsity-and-redundancy-in-multimodal)) usually ranks tokens by generic attention, which [attention sinks](/blog/paper-reading/large-language-model/streaming-llm-attention-sinks) contaminate; localization heads are text-conditioned and blob-shaped by construction. *First experiment:* at layer 15 of LLaVA-1.5-7B, keep only tokens inside the dilated pseudo-mask plus a small random budget, and compare accuracy-vs-FLOPs against FastV-style pruning on TextVQA and RefCOCO.

**3. Grounding-aware RL rewards.** *Question:* can the localization map serve as a dense reward that keeps a model's answers spatially honest during RL fine-tuning? *Why now:* it is a training-free, differentiable-free readout that already rivals fine-tuned grounders, so it can score rollouts without a separate reward model. *First experiment:* during GRPO on a visual reasoning set, add a reward term equal to the IoU between the box the model *states* and the box its own localization heads produce; check whether this reduces the 2.92-style gap between what the model knows and what it says.

**4. Causal ablation and patching of localization heads.** *Question:* are these heads necessary and sufficient for the model's spatial answers? *Why now:* the paper hands us exact head indices for ten models, which makes ablation cheap. *First experiment:* zero-ablate or mean-ablate the top-3 heads in LLaVA-1.5-7B and measure spatial-relation QA vs color/attribute QA; then patch head outputs from a "left horse" prompt into a "right horse" prompt and see if the answer flips.

**5. Localization heads in dynamic-resolution models.** *Question:* do models with variable patch grids, like [Qwen2-VL](/blog/machine-learning/frontier-models/qwen2-vl-enhancing-vision-language-models-perception-of-the-world-at-any-resolution), still concentrate grounding in a few heads, and are they the same heads across resolutions? *Why now:* the selection procedure needs only the token-to-position mapping, which these models expose through their 2D position encodings. *First experiment:* rerun selection frequency at three input resolutions and measure the overlap of the top-10 head sets.

## References

- Seil Kang, Jinyeong Kim, Junhyeok Kim, Seong Jae Hwang. "Your Large Vision-Language Model Only Needs A Few Attention Heads For Visual Grounding." CVPR 2025. [arXiv:2503.06287](https://arxiv.org/abs/2503.06287) · [CVF open access](https://openaccess.thecvf.com/content/CVPR2025/html/Kang_Your_Large_Vision-Language_Model_Only_Needs_A_Few_Attention_Heads_CVPR_2025_paper.html)
- Related on this site: [What do visual tokens really encode?](/blog/paper-reading/computer-vision/what-do-visual-tokens-really-encode-uncovering-sparsity-and-redundancy-in-multimodal) (sparsity and redundancy in multimodal visual tokens)
- Related on this site: [StreamingLLM and attention sinks](/blog/paper-reading/large-language-model/streaming-llm-attention-sinks) (why some heads dump attention on a few tokens)
- Related on this site: [Qwen2-VL](/blog/machine-learning/frontier-models/qwen2-vl-enhancing-vision-language-models-perception-of-the-world-at-any-resolution) (dynamic-resolution visual tokens)
