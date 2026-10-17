---
title: "Mechanisms of Object Localization in Vision–Language Models"
publishDate: "2026-10-17"
date: "2026-10-17"
description: "How LLaVA and InternVL turn a flat sequence of visual tokens into a bounding box: object tokens act as a container, the LLM rebuilds the 2D grid from corner anchors, and a handful of attention heads do the work."
category: "paper-reading"
subcategory: "Computer Vision"
tags: ["interpretability-analysis", "cvpr-2026", "cvpr-2026-highlight", "vision-language-models", "mechanistic-interpretability", "object-localization", "causal-mediation-analysis", "llava", "internvl", "attention-heads"]
author: "Hiep Tran"
featured: true
image: "/imgs/blogs/mechanisms-of-object-localization-in-vision-language-models-fig1.webp"
excerpt: "A mechanistic study of how VLMs draw boxes: the box follows a container of object tokens, not their arrangement, and a few attention heads carry the causal effect."
readTime: 18
paper:
  title: "Mechanisms of Object Localization in Vision-Language Models"
  authors: "Timothy Schaumlöffel, Martina G. Vilas, Gemma Roig"
  venue: "CVPR 2026 (Highlight)"
  url: "https://arxiv.org/abs/2605.19792"
---

> [!tldr]
> - **What it asks.** When you ask LLaVA-1.5 or InternVL-3.5 "give me the bounding box of the dog", which visual tokens, which layers and which attention heads actually produce the answer?
> - **The key mechanism: containerization.** The predicted box is set by *where* object-like tokens sit, not by how they are arranged. Pad an object with copies of its own tokens and the box grows to match; scramble the tokens inside the object and the box barely moves.
> - **Spatial layout is rebuilt inside the LLM.** The visual backbone loses most positional information by its last layer. The LLM reconstructs a grid from a flat sequence, leaning on strong corner signals that survive the projector.
> - **Sparse heads.** Causal mediation analysis finds that a very small set of attention heads carries the effect, in layers 11 to 16 for LLaVA and 16 to 22 for InternVL. Localization heads and classification heads barely overlap, yet ablating classification heads still breaks localization: identify first, then locate.
> - **Where it stops.** One object per category, COCO only (plus a Pascal VOC check), two model families, and attention heads only. The MLPs, training dynamics and multi-object scenes are untouched.

![Figure 9 from Schaumlöffel et al. (2026): object extension examples. Yellow is the original object mask, green is the padding filled with copies of the object's own tokens; the predicted box grows with the padded region](/imgs/blogs/mechanisms-of-object-localization-in-vision-language-models-fig1.webp)

The picture above is the paper's most striking result. Surround an object with a ring of tokens copied at random from inside it, and the predicted box swells to cover the ring, even though the copies sit in the wrong places. The model is drawing a box around a *container* of object stuff. The rest of this post unpacks how they showed that and where in the network the box gets computed.

## The problem

VLMs of the ViT → MLP → LLM family caption well and still misclassify single objects and misplace boxes. Most inherit a CLIP encoder, trained with one global image-text match per image, which never asks for pixel-level precision.

Yet these models *can* answer "where is the dog" with coordinates, so somewhere before the tokens that spell out `[x1, y1, x2, y2]` they build spatial structure. Earlier mechanistic work on VLMs studied classification, VQA and hallucination; this paper offers a first layer-level and head-level account of localization.

It sits next to two posts on this site that ask related questions. [Your large VLM only needs a few attention heads for visual grounding](/blog/paper-reading/computer-vision/your-large-vision-language-model-only-needs-a-few-attention-heads-for-visual-grounding) finds a few heads whose attention maps already point at the referred object. [What do visual tokens really encode](/blog/paper-reading/computer-vision/what-do-visual-tokens-really-encode-uncovering-sparsity-and-redundancy-in-multimodal) asks how much of the visual token sequence is actually used. This paper adds causal interventions: it deletes, shuffles and patches tokens and heads and watches the box change.

## Setup: two models, one curated dataset, two prompts

The two models are chosen to bracket the design space.

| | LLaVA-1.5 (7B, 13B) | InternVL-3.5 8B |
|---|---|---|
| Vision encoder | CLIP ViT-L/14, 336 px input | InternViT-300M, 448 px tiles |
| Tokens per view | 24 × 24 = 576, one token per patch | 16 × 16 = 256 after pixel shuffle (each 2 × 2 block merged) |
| Views | one padded, resized image | one global thumbnail plus up to 6 local high-res tiles (capped in the paper) |
| LLM | Vicuna 7B or 13B | Qwen3-8B |

The data is label-corrected COCO validation, filtered to objects between 0.4% and 60% of the image and one instance per queried category: 6,403 objects in 3,560 images. Each object is then inpainted away with LaMa, and only pairs where the model finds the object in the original and misses it in the inpainted copy survive. Intersected across models, that leaves 2,248 objects in 1,720 images, with context-only "detections" filtered out.

Two prompts probe each image:

- **Localization.** "Please provide the bounding box coordinates of the {class}." Scored as the fraction of predictions with IoU above 0.5, 0.7 and 0.9, averaged over the three thresholds.
- **Classification.** "List all objects in the image. Choose only from {class1}, {class2}, ..." Correct if the class name appears in the answer.

The list prompt beats a yes/no prompt on hallucination: false positive rates of 0.27 to 0.45 against 0.56 to 0.70.

Baselines on this subset are LLaVA-7B at 35.34% localization and 58.10% classification, LLaVA-13B at 46.98% and 65.30%, InternVL3.5-8B at 72.64% and 83.30%.

## Technique 1: token ablation, or which tokens carry the object

**The problem.** Before asking how the model localizes, you need to know what evidence it uses. Is the box computed from the tokens on the object, from global "register" tokens, or from the tokens a gradient method flags as important?

**The analogy.** A jury placing a crime scene. Remove the witnesses who stood there, random bystanders, or the ones the lawyer rates highest. If losing the on-scene witnesses is uniquely devastating, the verdict was built from them.

**The mechanism.** Ablation happens at the LLM input, after the projector. A removed token is not deleted (that would shift every later position); its embedding is overwritten with one average visual embedding from ImageNet validation. Four selection strategies compete at matched budgets:

1. **Object tokens.** Project the object mask onto the token grid and take every token that overlaps it by at least one pixel. Shrink or dilate this set by one or two rings of tokens (padding −2, −1, +1, +2).
2. **Register tokens.** Tokens whose embedding norm exceeds the mean by two standard deviations, the ViT's global scratch space.
3. **Integrated Gradients.** The top-k tokens by attribution to the correct class logit or the box coordinates.
4. **Random tokens.** k random tokens, three seeds.

**The math.** Let $V = \{v_1, \dots, v_N\}$ be the visual token embeddings at the LLM input, with $N = 576$ for LLaVA, and let $S \subseteq \{1, \dots, N\}$ be the ablated set. The ablated input is

$$
\tilde v_i = \begin{cases} \bar v & i \in S \\ v_i & i \notin S \end{cases}, \qquad \bar v = \frac{1}{|D|} \sum_{x \in D} \frac{1}{N} \sum_{j=1}^{N} v_j(x),
$$

where $D$ is the ImageNet validation set and $v_j(x)$ is token $j$ of image $x$. The quantity reported is accuracy after ablation and the drop from baseline, at a matched fraction $|S| / N$.

![Schematic token grid: which tokens each ablation removes, and what LLaVA-7B localization accuracy is left (baseline 35.3%)](/imgs/blogs/mechanisms-of-object-localization-in-vision-language-models-1.webp)

**The worked example.** On LLaVA-7B the object mask covers about 8% of tokens, roughly 46 of 576. Replacing those 46 drops localization from 35.34% to 5.92%, an 83% relative loss. Replacing a random 8% leaves 35.09%. Replacing the top 8% by Integrated Gradients leaves 15.84%, which hurts, but much less than the object set. The register tokens do nothing for localization (35.07%).

| LLaVA-7B ablation | Tokens removed | Loc. (%) | Cls. (%) |
|---|---|---|---|
| None | 0% | 35.34 | 58.10 |
| Object, −1 padding | 3% | 31.73 | 55.29 |
| Object | 8% | 5.92 | 19.44 |
| Object, +2 padding | 21% | 0.34 | 10.59 |
| Integrated Gradients | 8% | 15.84 | 43.95 |
| Random | 8% | 35.09 | 56.58 |

Localization collapses harder than classification: with +1 padding all three models are under 10% localization while classification still succeeds 11% to 24% of the time. Growing the mask hurts much more than shrinking it, so boundary tokens hold real information. Pascal VOC agrees: LLaVA-7B localization falls from 52.41% to 9.52%.

**When it fails.** Mean replacement is a choice: the LLM may read "average token here" as weak evidence of background. No other counterfactual (say, patches from another image) is tested.

## Technique 2: containerization

**The problem.** Two stories fit the ablation. Either the model reads the semantic layout (head at top, legs at bottom) and infers extent, or it finds the region "full of dog" and boxes it.

**The analogy.** Coloring a dog drawing versus outlining a blob of dog-colored paint: one cares which part goes where, the other only where the paint stops.

**The mechanism.** The authors run two interventions that pull those stories apart.

- **Object extension.** Grow the object's token region by $p$ rings. Fill each new token with a copy of a random token from *inside* the original object. Ten seeds per setting. Score the predicted box against the original ground truth box and against boxes scaled outward by $s$ rings.
- **Shuffling.** Permute the tokens inside the object mask at the LLM input, keeping the set of tokens and the container shape but destroying the internal arrangement. As a contrast, permute *all* image tokens.

**The math.** Write the predicted box as $\hat b = f(\tilde V)$ and let $b^{(s)}$ be the ground truth box expanded by $s$ token rings. The extension experiment fills a matrix

$$
A_{p,s} = \frac{1}{|\mathcal{X}|} \sum_{x \in \mathcal{X}} \frac{1}{3} \sum_{\tau \in \{0.5,\, 0.7,\, 0.9\}} \mathbb{1}\!\left[\mathrm{IoU}\big(\hat b_p(x),\, b^{(s)}(x)\big) \gt \tau\right],
$$

where $p$ is the padding applied to the input, $s$ the scaling applied to the reference box, $\mathcal{X}$ the probe set and $\tau$ the IoU threshold. If the model is a container reader, the mass of $A$ sits on the diagonal $p = s$.

![Figure 1 from Schaumlöffel et al. (2026): accuracy of predictions at each padding level against ground truth boxes scaled by each amount, for LLaVA-7B and InternVL3.5-8B. The bright diagonal means the box grows with the padded container](/imgs/blogs/mechanisms-of-object-localization-in-vision-language-models-fig2.webp)

**The worked example.** For LLaVA-7B, inputs padded by one ring score 44.0% against the +1-scaled box and only 24.6% against the original box. Padded by two rings, they score 45.9% against the +2 box and 11.3% against the original. InternVL shows the same diagonal (45.0% at +1, 45.9% at +2). LLaVA-13B is even cleaner: 55.7% at +1 and 60.3% at +2.

The shuffle experiment closes the case for LLaVA.

| Shuffle mode | LLaVA-7B Loc. | LLaVA-13B Loc. | InternVL Loc. | LLaVA-7B Cls. |
|---|---|---|---|---|
| None | 35.34 | 46.98 | 72.64 | 58.10 |
| All image tokens | 1.90 | 0.30 | 4.70 | 62.42 |
| Inside object mask | 35.30 | 45.20 | 37.70 | 58.44 |

Shuffling inside the mask leaves LLaVA-7B at 35.30%, indistinguishable from baseline; shuffling the whole image destroys it. Classification ignores both shuffles. The box depends on *where object tokens are*, the label on *which tokens exist*.

![Containerization: pad the object with its own scrambled tokens and the box grows; scramble the inside and the box stays put (LLaVA-7B numbers)](/imgs/blogs/mechanisms-of-object-localization-in-vision-language-models-2.webp)

**When it fails.** Look at the InternVL column: the inside-mask shuffle cuts its localization from 72.64% to 37.70%, a 34.9 point drop. The "largely independent of internal structure" summary fits LLaVA and InternVL only partly. My untested guess: pixel shuffle packs a 2 × 2 patch block into each token, so tokens carry local geometry that a shuffle breaks.

## Technique 3: the global view carries space, the local view carries detail

**The problem.** InternVL sees each image twice: a 448 px thumbnail (the global view) and several 448 px tiles cut from the full-resolution image (local views). Does the box come from both, or one?

**The mechanism and result.** Ablate object tokens in only one view. Removing them from the global view drops localization by 36.4 points (72.64% to 36.20%). Removing them from the local views drops it by 9.7 points. Classification follows the same direction but with smaller gaps, 9.5 against 6.6 points. Ablating both together is far worse than either alone (localization 11.27%), so the two views are complementary rather than redundant.

By size, small objects (under 32² pixels) lose 64.8% of localization accuracy without the global view and 54.3% without the local one. For large objects, removing the local view can slightly *help* (+6.5%), which suggests choosing the crop count by task.

## Technique 4: positional decoding, or how a 1D sequence becomes a grid

**The problem.** The LLM receives the 576 visual tokens as a flat sequence with 1D positions. To output a box it needs to know that token 25 is directly below token 1. Where does that 2D knowledge come from?

**The mechanism.** Text typed on a fixed-width page with the line breaks removed can be recovered if row ends are marked. To see whether the LLM does this, train a linear probe per layer and axis to predict each token's grid position (10 epochs on 50,000 ImageNet images, tested on 10,000).

**The math.** For layer $\ell$ with hidden state $h_i^{(\ell)} \in \mathbb{R}^d$ for visual token $i$, and true column $c_i \in \{0, \dots, 23\}$, the column probe is

$$
\hat c_i = \arg\max_{k} \left( W_c^{(\ell)} h_i^{(\ell)} + u_c^{(\ell)} \right)_k, \qquad W_c^{(\ell)} \in \mathbb{R}^{24 \times d},
$$

with a matching row probe $W_r^{(\ell)}$. Here $d$ is the hidden width at that layer and $u$ is a bias. The score at layer $\ell$ is the fraction of tokens whose position is predicted exactly.

![The projector hands the LLM a flat sequence with strong corner signals; by mid layers the LLM has rebuilt a grid, with row starts decoded best](/imgs/blogs/mechanisms-of-object-localization-in-vision-language-models-3.webp)

**The result.** In both vision backbones, position is decodable early and largely gone by the last layer, as contrastive ViTs trade spatial precision for semantics. At the projector output, position is weak overall but strong at the four image corners. Inside the LLM, decodability climbs fast and peaks around layer 12 for LLaVA-7B, 13 for LLaVA-13B and 7 for InternVL (the appendix caption swaps the two LLaVA layer numbers; the trend is the same).

![Figure 2 from Schaumlöffel et al. (2026): left, positional decoding accuracy per layer through the ViT backbone, the projector P and the LLM for three models; right, per-position accuracy at the best LLM layer, highest near the image edges](/imgs/blogs/mechanisms-of-object-localization-in-vision-language-models-fig3.webp)

On the right, tokens at inferred row boundaries, the "line breaks", are decoded best: the LLM appears to use corner anchors to infer row length and rebuild a grid.

**When it fails.** A probe shows information is readable, not that the model uses it. The corner tokens are never ablated; I return to that below.

## Technique 5: attention knockout, finding the layers

**The problem.** Now that we know *which* tokens matter, find *where* in the LLM their information gets read out.

**The mechanism.** Block attention from every token after the image (the prompt and answer tokens) to the object tokens, in a group of four consecutive layers at a time. After the knockout, those layers cannot pull anything from the object. If accuracy drops, those layers are where the extraction happens.

**The math.** For a head with queries $q_t$ and keys $k_j$, the knocked-out attention weight is

$$
\alpha_{t,j} = \frac{\exp\!\left(q_t^\top k_j / \sqrt{d_h} + m_{t,j}\right)}{\sum_{j'} \exp\!\left(q_t^\top k_{j'} / \sqrt{d_h} + m_{t,j'}\right)}, \qquad m_{t,j} = \begin{cases} -\infty & t \gt t_{\text{img}},\ j \in S_{\text{obj}} \\ 0 & \text{otherwise} \end{cases}
$$

where $d_h$ is the head dimension, $t_{\text{img}}$ the index of the last image token and $S_{\text{obj}}$ the object token set. The mask is applied in every head of the chosen layer group.

**The result.** For LLaVA, blocking in the early to middle layers hurts badly while late-layer blocking barely matters. For InternVL the damage concentrates in middle to late layers. Both tasks share early layers, then localization depends on a further band: identify, then locate. For LLaVA, the worst localization knockouts line up with the positional-decoding peak. Six-layer groups hurt more than single layers, so evidence accumulates across layers.

## Technique 6: causal mediation down to single heads

**The problem.** A band of layers is still dozens of heads. Which individual heads carry the effect?

**The analogy.** Copy one wire at a time from a working house into a broken one and see which wire turns the lights back on.

**The mechanism.** Activation patching on 50 image pairs. For each pair there is a **source** run (object present, model correct) and a **base** run (object inpainted away, model fails). For one head at a time, copy that head's output from the source run into the base run, at all prompt token positions, giving a **patched** run. Answers are scored by teacher-forced perplexity with template tokens like brackets masked out; classification uses the yes/no prompt for one decisive token.

**The math.** With $P_{\text{base}}$, $P_{\text{src}}$ and $P_{\text{patched}}$ the perplexities of the reference answer under the three runs, the Mediation Fraction of a head is

$$
\mathrm{MF} = \frac{P_{\text{base}} - P_{\text{patched}}}{P_{\text{base}} - P_{\text{src}}}.
$$

An MF near 1 means the head alone carries the information that separates the two images. Near 0 means the head is irrelevant. Negative means patching it in makes the base run *worse*.

![Mediation Fraction on a toy example: the share of the source-to-base perplexity gap that patching one head closes. Illustrative numbers](/imgs/blogs/mechanisms-of-object-localization-in-vision-language-models-4.webp)

**The worked example (illustrative, not from the paper).** Say the correct box has perplexity 2 with the object present and 10 with it removed, so the gap is 8. Patching head A brings it to 4: MF = 6/8 = 0.75. Head B brings it to 9.6: MF = 0.05. Head C pushes it to 11: MF = −0.125, an interfering head.

```python
def mediation_fraction(model, src, base, head, answer):
    cache = model.run_with_cache(src)                    # source activations
    p_src  = model.perplexity(src,  answer)
    p_base = model.perplexity(base, answer)
    with model.patch(head, cache[head]):                 # one head, all prompt positions
        p_patch = model.perplexity(base, answer)
    return (p_base - p_patch) / (p_base - p_src)
```

![Figure 4 from Schaumlöffel et al. (2026): Mediation Fraction for every attention head by layer, for localization and classification, in LLaVA-7B and InternVL3.5-8B. Almost every head is near zero](/imgs/blogs/mechanisms-of-object-localization-in-vision-language-models-fig4.webp)

**The result.** The heatmaps are mostly blank. For LLaVA, the nonzero heads cluster in layers 11 to 16 for both tasks. For InternVL they sit in layers 16 to 22. Localization uses a tighter set of heads than classification, which spreads across earlier and middle layers.

The overlap is small. Among the top 10 heads per task, only two are shared in LLaVA and one in InternVL. Even among the top 50, the shared count is 20 for LLaVA-7B, 15 for LLaVA-13B and 18 for InternVL. InternVL also has many negative MF heads, read as carrying conflicting context.

## Technique 7: cumulative head ablation, necessity not just correlation

**The problem.** A high MF says a head is *sufficient* to carry some effect. It does not say the model *needs* it.

**The mechanism.** Rank heads by mean MF. Zero out the output of an increasing fraction of heads, three ways: localization-critical heads first, classification-critical heads first, or near-zero heads first as a control. Track localization accuracy and summarize each curve by a normalized area under it.

![Figure 5 from Schaumlöffel et al. (2026): localization accuracy as attention heads are cumulatively zeroed, ranked by Mediation Fraction. Removing localization-critical heads collapses accuracy; removing unimportant heads barely matters](/imgs/blogs/mechanisms-of-object-localization-in-vision-language-models-fig5.webp)

**The result.** Removing localization-critical heads collapses accuracy quickly (normalized AUC 0.02 for LLaVA-7B, 0.04 for InternVL). Removing unimportant heads leaves a much flatter curve (AUC 0.25 and 0.41). The surprise is the middle curve. Classification-critical heads barely overlap with localization heads, yet removing them also destroys localization (AUC 0.09 and 0.15). The paper's conclusion: localization is built on top of the object-identification representation. First the model works out *what*, then a smaller set of heads works out *where*.

## Critique

**What is load-bearing and might not transfer.** The probe set is one object per category on curated COCO, which hides any mechanism for picking *which* of two dogs to box. LLaVA's one-token-per-patch layout is what makes the container picture so clean; models with heavier token compression (InternVL already shows weaker shuffle robustness) or learned resamplers may not have a clean container to read. And the mediation runs use only 50 image pairs and the binary prompt for classification, a different prompt from the one used in every other experiment.

**What is strong.** The experimental logic is careful. The object-removed control set rules out context hallucination before any mechanism claim is made. The token ablations are budget matched against random and gradient baselines, so "object tokens matter" is not just "removing tokens hurts". The extension experiment cleanly separates a layout reader from a container reader. And the head analysis moves from sufficiency (MF) to necessity (ablation), which many circuit papers skip.

**What is weak or underdetermined.**

- The containerization claim is stated more strongly than the InternVL shuffle data supports. A 34.9 point drop is not "largely irrelevant".
- The 1D to 2D reconstruction story rests on probes alone. The corner-anchor mechanism is plausible and never causally tested.
- Mediation covers attention heads only; MLPs may carry much of the spatial computation.
- "Sequential mechanism" could also be a shared object representation, written by classification heads and read by both tasks in parallel. The ablation cannot tell those apart.
- Two families, three checkpoints. Grounding-trained models such as Qwen-VL, the most interesting comparison, are absent.

**What would change my mind.** If ablating the four corner tokens at the projector output left boxes and mid-layer position decoding intact, the corner-anchor story would be wrong. And if a grounding-trained VLM such as Qwen2.5-VL showed a strong shuffle sensitivity *inside* the mask (box shape tracking internal layout), I would read containerization as a symptom of weak grounding training, not a general VLM mechanism.

## Where this goes next

The authors point at head fine-tuning, grounding-aware attention supervision and video. Concrete versions follow; they are my extrapolation, not the paper's claims.

1. **Steer boxes through the localization heads.** *Question:* can you move or resize a predicted box by editing the outputs of the top-MF localization heads alone? *Why now:* the paper ranks a few heads per model, and box extent follows the token container. *First experiment:* patch localization-head outputs from a +1-padded run into the unpadded run and check whether the box grows one ring with the input untouched.
2. **Causally test the corner anchors.** *Question:* is the 2D grid really rebuilt from the corners? *Why now:* the probes say which tokens carry position at the projector and which layer peaks. *First experiment:* ablate the four corner tokens (control: four random edge tokens), re-run probes and localization, then plant fake corners at the wrong row length and see whether boxes shear.
3. **Compare across VLM families, especially grounding-trained ones.** *Question:* does containerization survive in models trained with box supervision (Qwen2.5-VL, Molmo, LLaVA-Grounding), and do their localization heads sit at the same relative depth? *Why now:* the toolkit is defined and the code is public. *First experiment:* run extension and inside-mask shuffle on Qwen2.5-VL and plot its diagonal and shuffle drop next to LLaVA's.
4. **Extend to video and tracking.** *Question:* in video VLMs that output a box per frame, is there a container per frame, and are the localization heads shared across time? *Why now:* inpainting the object out of every frame yields the same source/base pairs. *First experiment:* pad the container in one frame only and see whether the box changes there alone or drags its neighbors.
5. **Use localization heads as a training signal.** *Question:* does supervising the top-MF heads' attention to fall on the ground truth mask improve grounding more cheaply than full fine-tuning? *Why now:* the heads are few, so the loss is cheap. *First experiment:* fine-tune only those heads' query/key projections on RefCOCO with an attention-to-mask loss, against LoRA on the full LLM at equal parameter count.

## References

- Schaumlöffel, Vilas, Roig. *Mechanisms of Object Localization in Vision-Language Models.* CVPR 2026. [arXiv:2605.19792](https://arxiv.org/abs/2605.19792), [CVF open access](https://openaccess.thecvf.com/content/CVPR2026/html/Schaumloffel_Mechanisms_of_Object_Localization_in_Vision-Language_Models_CVPR_2026_paper.html)
- Code: [github.com/t9s9/vlm-loc-mechanisms](https://github.com/t9s9/vlm-loc-mechanisms)
- On this site: [Your large VLM only needs a few attention heads for visual grounding](/blog/paper-reading/computer-vision/your-large-vision-language-model-only-needs-a-few-attention-heads-for-visual-grounding)
- On this site: [What do visual tokens really encode](/blog/paper-reading/computer-vision/what-do-visual-tokens-really-encode-uncovering-sparsity-and-redundancy-in-multimodal)
- On this site: [Circuit mechanisms for spatial relation generation in diffusion transformers](/blog/paper-reading/computer-vision/circuit-mechanisms-for-spatial-relation-generation-in-diffusion-transformers)
