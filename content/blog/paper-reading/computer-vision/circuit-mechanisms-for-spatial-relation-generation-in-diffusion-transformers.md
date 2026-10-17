---
title: "Circuit Mechanisms for Spatial Relation Generation in Diffusion Transformers"
publishDate: "2026-10-17"
date: "2026-10-17"
description: "How a diffusion transformer learns to put the red square above the blue circle: two cross-attention heads that paint a positional gradient and then fill it with a shape when the text encoder is random, and a single fused token decoded by one head when the encoder is T5. The circuits behind the near-perfect accuracy, and why one of them falls apart when you add the word 'the'."
category: "paper-reading"
subcategory: "Computer Vision"
tags: ["interpretability-analysis", "cvpr-2026", "cvpr-2026-highlight", "mechanistic-interpretability", "diffusion-transformer", "pixart", "cross-attention", "spatial-relations", "text-to-image", "t5", "circuit-analysis"]
author: "Hiep Tran"
featured: true
readTime: 16
image: "/imgs/blogs/circuit-mechanisms-for-spatial-relation-generation-in-diffusion-transformers-fig2.webp"
excerpt: "Train the same diffusion transformer twice on the same toy task, once with random token embeddings and once with T5. Both learn to place objects correctly. Open them up and you find two completely different machines, and only one survives a filler word."
paper:
  title: "Circuit Mechanisms for Spatial Relation Generation in Diffusion Transformers"
  authors: "Binxu Wang, Jingxuan Fan, Xu Pan"
  venue: "CVPR 2026 (highlight)"
  url: "https://arxiv.org/abs/2601.06338"
---

> [!tldr]
> **Question.** Text-to-image models still misplace objects. How does a diffusion transformer (DiT) turn a relation word into a layout at all?
>
> **Setup.** Small PixArt-style DiTs trained from scratch on two colored shapes and eight spatial relations, with either random token embeddings (RTE) or pretrained T5 as the text encoder.
>
> **RTE finding.** A two-stage circuit: a "relation head" (L2H8) turns "above" into a gradient over the 8 by 8 image grid at sampling step 0, then an "object head" (L4H3) reads that tag and writes the shape there.
>
> **T5 finding.** No head reads the relation word. T5 folds the relation into the embedding of the *second shape word*, and one head (L3H7) decodes the layout from it.
>
> **The twist.** Both score about 0.81 to 0.84 on relations in-domain. Insert one "the" before the second object and T5-DiT falls from 0.808 to 0.498, while RTE-DiT holds at 0.862.

![Figure 1 from Wang, Fan and Pan (2026): the task and the PixArt-style model. The prompt is embedded by T5, by a random token encoder with positions (RTE), or by RTE without positions, and enters every DiT block through cross attention.](/imgs/blogs/circuit-mechanisms-for-spatial-relation-generation-in-diffusion-transformers-fig1.webp)

The diagram above is the whole apparatus. The rest of this post opens the two trained models and walks through what each one computes.

## The problem: relations are where text-to-image still fails

Ask a modern model for "a red square" and you get one. Ask for "a red square to the upper left of a blue circle" and success falls sharply. The paper's benchmark scatter (appendix Figure 8) shows single-object attribute scores have improved far faster than spatial-relation scores.

The literature has two explanations. One camp blames cross attention, so methods like Attend-and-Excite push attention around at inference. The other blames the text encoder: CLIP and T5 may not preserve "left of" versus "right of" well enough to recover.

This paper looks instead of arguing. Train a DiT that *does* solve relations, find its circuit, and you learn which component carries the relation. The authors do it twice, once per encoder, and get two different answers. For background on the backbone, see the [Diffusion Transformers post](/blog/machine-learning/image-generation/diffusion-transformers-dit).

## Contributions

1. A minimal, fully scoreable task on which small DiTs reach near-perfect accuracy, so circuits are studied in a model that works.
2. **Attention Synopsis**, which compresses about 10 million cross-attention maps into a layer by head table per (image region, word group) pair.
3. **Weight-space head screening**, which tests every head for "turns a relation feature into a spatial gradient" from weights alone.
4. A two-stage relation-then-object circuit in RTE-DiT, and a fused-token circuit in T5-DiT, each confirmed causally.
5. Evidence that the T5 circuit breaks under small prompt changes that the RTE circuit shrugs off.

## The setup, briefly

Prompts take the form *[color A] [shape A] [relation] [color B] [shape B]*, such as "red square above and to the left of blue circle", paired with a 128 by 128 image. Shapes are circle, triangle and square; colors are red and blue; relations are left, right, above, below and the four diagonals. Color words are randomly dropped and relations paraphrased.

The Stable Diffusion VAE gives a 4 by 16 by 16 latent, and patch size 2 gives an **8 by 8 grid of 64 image tokens**. Every attention map below lives on that grid. Sizes run from DiT-B (12 layers, 12 heads, 768 dimensions) down to DiT-nano (3 layers, 3 heads, 192 dimensions).

The RTE encoder maps each token ID to a fixed Gaussian vector with T5's dimension (4096) and norm, plus sinusoidal positions: no semantics, no context. Training runs 160,000 steps at batch size 256. Sampling uses DPM-Solver++ with 14 steps and guidance 4.5, and an OpenCV pipeline scores color, shape, unique binding and spatial relation.

| DiT-B variant (trained template) | shape | color | binding | spatial relation |
| --- | --- | --- | --- | --- |
| RTE (with positions) | 0.877 | 0.928 | 0.855 | 0.843 |
| RTE without positions | 0.859 | 0.899 | 0.415 | 0.207 |
| T5 | 0.857 | 0.892 | 0.820 | 0.808 |
| CLIP | 0.806 | 0.900 | 0.772 | 0.759 |

*Table 2 of the paper: template "O1 is Rel O2", 264 prompts by 50 samples each.*

The no-positions row is the cleanest sanity check. Without positions, cross attention sees a bag of words, so "red A above blue B" and "blue B above red A" are the same input. Binding halves and relation falls to 0.207, which tells you pretrained semantics are not required, only word order.

Training dynamics hint at different internals. Color converges first, then shape, then binding, and relation last. T5-DiT converges faster with tighter gaps between stages.

## Technique 1: Attention Synopsis, finding the heads that matter

**The problem.** DiT-B has 12 layers, 12 heads, 14 steps, two guidance passes, 64 image tokens and 20 text tokens, times many samples and 96 prompts. Nobody can inspect that, and averaging blurs the specific interactions you care about.

**The analogy.** A phone company asked "who talks to whom about the weather" does not read every call. It tags callers by neighborhood and topics by keyword, then sums minutes per pair. The synopsis tags image tokens by which object they belong to and text tokens by word type.

**The mechanism.** Segment each generated image into masks for each shape and the background, downsample to 8 by 8, and build a multi-hot text mask over a word group (all relation words, or object 1's shape word). Their outer product is a template that selects only the attention entries linking that region to that word group.

**The math.** Let $A^{\text{cross}}_{\ell,t,h}(i,j)$ be the cross-attention weight from image token $i$ to text token $j$ at layer $\ell$, step $t$, head $h$. With masks $M_{\text{img}}$ and $M_{\text{text}}$:

$$
T = M_{\text{img}} \otimes M_{\text{text}}, \qquad \text{Score}_{\ell,t,h} = \sum_{i,j} A^{\text{cross}}_{\ell,t,h}(i,j)\, T(i,j)
$$

Attention changes smoothly over steps, so averaging over $t$ leaves an $L \times H$ table, the synopsis. Each cell is "how much energy flows from this object's pixels to this word group through this head".

**A tiny example.** Suppose the square covers 6 of 64 tokens and "above" is text token 4. The template is 1 on those 6 rows in column 4. If L2H8 puts weight 0.4 on "above" from each square token, it scores 2.4, while a head that ignores relation words scores near 0.

**When it fails.** It needs clean masks and clean word groups. Shapes on gray with a 20-token vocabulary make both easy; overlapping real objects and multi-token phrases do not.

![Figure 4 from Wang, Fan and Pan (2026): the relation head in RTE-DiT-B. A: synopses single out L2H8. B: its activation across steps; the map for "below and right" is the sum of "below" and "right". C: the same gradients from the QK product of positions and relation embeddings. D: ablating L2H8 breaks relation accuracy.](/imgs/blogs/circuit-mechanisms-for-spatial-relation-generation-in-diffusion-transformers-fig2.webp)

## Technique 2: the relation head paints a gradient

For object 1, one head dominates the synopsis: layer 2, head 8. The pattern recurs at DiT-mini and DiT-micro, while DiT-nano has no such head and gets about 5% relation accuracy.

The striking fact is **when** L2H8 fires: fully at step 0, before there is any image. Its attention is a smooth slope across the canvas, oriented by the relation word, that slowly concentrates where the object lands. At step 0 the only spatial signal in an image token is its positional embedding, so the query must essentially be position and the key the relation word.

**The QK circuit.** Stack the 2D sinusoidal positional embeddings $p_s$ of the grid cells into $P \in \mathbb{R}^{S^2 \times d}$, where $S = 8$ and $d$ is the model width. Stack the relation-word embeddings, after PixArt's caption MLP, into $V \in \mathbb{R}^{M \times d}$. For head $h$ in layer $\ell$ with slices $W_{Q,h}, W_{K,h} \in \mathbb{R}^{d \times d_h}$:

$$
\Phi^{(\ell)}_h = \left(P\, W^{(\ell)}_{Q,h}\right)\left(V\, W^{(\ell)}_{K,h}\right)^{\top} \in \mathbb{R}^{S^2 \times M}
$$

Each column is a pre-softmax logit for one relation word at every cell. Reshaped to 8 by 8, L2H8's columns are clean ramps: "above" highest along the top rows, "right" highest on the right.

![Redrawn: L2H8's QK circuit on the 64-token grid. Each cell is the logit between that cell's position and the relation word. "Below" plus "right" sums to the diagonal map for "below and right".](/imgs/blogs/circuit-mechanisms-for-spatial-relation-generation-in-diffusion-transformers-1.webp)

**Scoring "is this a ramp".** Each relation $m$ gets a direction such as $d_{\text{above}} = (0, -1)$ or $d_{\text{right}} = (1, 0)$, with $y$ growing downward, and a template on coordinates $x_j, y_k \in [-1, 1]$:

$$
T_m(j,k) = d_{m,x}\, x_j + d_{m,y}\, y_k, \qquad \rho_h = \frac{\langle \bar\Phi_m, \bar T_m \rangle}{\lVert \bar\Phi_m \rVert\, \lVert \bar T_m \rVert}
$$

A bar means mean-centered, and $\rho_h$ is the cosine between the head's map and the ideal ramp (the paper adds a signed projection and an energy score). Running this over every head is the weight-space screen: matrix products, no sampling.

**Worked example on a 3 by 3 grid.** Take rows $y \in \{-1, 0, 1\}$ from top to bottom. "Above" is $T = -y$: top row $+1$, middle 0, bottom $-1$. "Below" is $T = y$ and "right" is $T = x$, so "below and right" is $x + y$, which is $+2$ at bottom-right and $-2$ at top-left. That is the additive decomposition Figure 4B shows in the trained model.

The authors call these maps **positional tags**. Through its value-output (VO) path, L2H8 writes onto each image token a vector whose strength follows the slope, marking "object 1 goes here" before pixels exist. They compare it to the morphogen gradients that tell embryonic cells which organ to become.

**When it fails.** It needs explicit positions and an explicit relation token. The "query is roughly position" approximation also weakens in later layers, where image content dominates the residual stream.

## Technique 3: the object head reads the tag

A tagged canvas is not a picture; something must put a square where the tag says "object 1". The synopsis for "object tokens attending to their own shape word" again lights up one head: **L4H3**.

L4H3 is the mirror of L2H8. It ignores which relation was asked for and whether the shape came first or second. It switches on later, around steps 4 to 8 of 14, and its attention sharpens from a blob to the object's outline.

![Figure 5 from Wang, Fan and Pan (2026): the object head L4H3. A: synopses for object tokens attending to their own shape word. B: tokens at the eventual square attend to "square" over sampling. C: feeding L2H8's VO output into L4H3 makes tagged cells attend to "square"; without it the pattern is weak. D: ablating L4H3 lowers shape accuracy.](/imgs/blogs/circuit-mechanisms-for-spatial-relation-generation-in-diffusion-transformers-fig3.webp)

**The causal test.** Feed L4H3 a synthetic input of positional embeddings plus L2H8's VO output, nothing else. With the tag, tagged cells attend selectively to "square"; without it, the pattern is weak. The downstream head reads the upstream head's tag, which is what makes this a circuit rather than two co-active heads.

**The ablation.** Removing L2H8's attention to relation words cuts relation accuracy from 67% to 33%, with other heads negligible. Removing L4H3's attention to object words cuts shape accuracy from 90% to 76%. The smaller effect suggests other paths also carry shape.

As schematic code, RTE-DiT runs a two-step program:

```python
# RTE-DiT, per sampling step (schematic of the circuit, not the model code)
def relation_then_object(img_tokens, pos_emb, text_emb, rel_idx, shape_idx):
    # Stage 1 (layer 2, head 8): query = position, key = relation word.
    logits = (pos_emb @ Wq_L2H8) @ (text_emb[rel_idx] @ Wk_L2H8).T   # [64, 1]
    tag = softmax_over_text(logits) * (text_emb[rel_idx] @ Wvo_L2H8)  # [64, d]
    img_tokens = img_tokens + tag            # "object 1 belongs on this slope"

    # Stage 2 (layer 4, head 3): tagged tokens look up the matching shape word.
    q = img_tokens @ Wq_L4H3                 # carries the tag
    k = text_emb[shape_idx] @ Wk_L4H3        # "square", "circle"
    attn = softmax(q @ k.T)                  # tagged cells pick shape 1
    return img_tokens + attn @ (text_emb[shape_idx] @ Wvo_L4H3)
```

Relation first, identity second, each in its own head. If every model worked like this, layout editing would mean swapping one key vector.

## Technique 4: in T5-DiT, the relation hides in the second shape word

Run the synopsis on T5-DiT and no relation head appears, yet objects land correctly. The relation must enter through some other token.

**Token masking finds it.** Masking relation, filler or color words barely matters. Masking end-of-sentence collapses every metric. Masking shape1 lowers relation accuracy by 15%, and masking **shape2** lowers shape, binding and relation by about 50%.

That fits T5. Its self-attention lets the embedding at "circle" in "red square above blue circle" know it is a blue circle below a square. The DiT does not need to read "above"; it reads one fused token.

![Figure 7 from Wang, Fan and Pan (2026): T5-DiT. A: masking the relation word barely matters, masking shape2 or end-of-sentence does. B: screening with relation vectors from the shape2 embedding finds L3H7. C: vector arithmetic on shape2 moves the generated objects.](/imgs/blogs/circuit-mechanisms-for-spatial-relation-generation-in-diffusion-transformers-fig4.webp)

**Variance partitioning.** Model the contextual shape2 vector of sample $i$ as a sum of factor effects:

$$
x_i = \mu + \sum_{f} \beta^{(f)}_{y^{(f)}_i} + \varepsilon_i, \qquad R^2_{f,\text{part}} = \frac{\operatorname{tr}\!\left(A\,(P_{\text{all}} - P_{-f})\right)}{\operatorname{tr}(A)}
$$

Here $f$ ranges over shape2, color2, shape1 and relation; $x_i$ is the embedding, $\mu$ the mean, $\beta^{(f)}_{\ell}$ the effect vector for level $\ell$ of factor $f$, and $\varepsilon_i$ the residual. $A = X_c X_c^{\top}$ is the centered Gram matrix, $P_{\text{all}}$ projects onto all one-hot factor columns and $P_{-f}$ onto all but $f$. Partial $R^2$ is the variance only $f$ explains: a multivariate ANOVA on 4096-dimensional vectors.

![Redrawn from Table 1: partial R^2 of each factor in the shape2 token. In raw T5, shape2's identity dominates (37.5%) and relation explains 12.1%. After PixArt's caption MLP, relation is the largest factor (21.3%).](/imgs/blogs/circuit-mechanisms-for-spatial-relation-generation-in-diffusion-transformers-2.webp)

The four factors explain 73.2% of the T5 embedding's variance and 53.4% after the MLP. The trainable caption MLP rotates the space so the relation becomes the loudest component of that token.

**A relation head after all, keyed on shape2.** The effect vectors $\beta^{(\text{rel})}_\ell$ give eight relation directions $V_{\text{rel}}$ inside shape2. In the QK screen, one or two heads turn them into ramps, notably **L3H7**. T5-DiT still makes a positional gradient; it reads the relation from an object word.

**Moving objects by editing one vector.** Take "blue circle is to the lower left of red square". In the T5 embedding of "square", subtract the lower-left factor and add a scaled lower-right one:

$$
\tilde V^{*}_{\text{square}} = V^{*}_{\text{square}} - V_{\text{lower left}} + 3\, V_{\text{lower right}}
$$

Generation then consistently places the objects in the new relation while keeping shapes and colors. No word changed, which is about as direct as causal evidence gets.

![Redrawn: the vector edit and the filler-word failure. Top: editing only the shape2 embedding flips the layout from lower left to lower right. Bottom: mean object displacement (Dx, Dy, pixels of 128). Adding "the" moves T5-DiT from (-0.8, -0.5) to (-33.7, -24.9); RTE-DiT moves under a pixel.](/imgs/blogs/circuit-mechanisms-for-spatial-relation-generation-in-diffusion-transformers-4.webp)

## Experiments: the same score, a different failure surface

The difference shows up when the prompt drifts by one token. The authors test five templates, only the first seen in training.

| Template | RTE DiT-B relation | T5 DiT-B relation | T5 mean (Dx, Dy) px |
| --- | --- | --- | --- |
| O1 is Rel O2 (trained) | 0.843 | 0.808 | (-0.8, -0.5) |
| O1 is Rel the O2 | 0.862 | 0.498 | (-33.7, -24.9) |
| O1 Rel O2 | 0.823 | 0.608 | (-4.5, -16.6) |
| O1 Rel the O2 | 0.842 | 0.400 | (-35.1, -37.0) |
| the O1 is Rel the O2 | 0.833 | 0.529 | (-18.4, -20.7) |

*Table 2 of the paper. Nonzero mean displacement means a systematic bias, not random errors.*

![Redrawn from Table 2: relation accuracy across templates. RTE-DiT stays between 0.82 and 0.87; T5-DiT matches it only on the trained template and loses 0.20 to 0.41 on each variant.](/imgs/blogs/circuit-mechanisms-for-spatial-relation-generation-in-diffusion-transformers-3.webp)

T5-DiT's shape and color barely move (0.915 and 0.931 on "O1 is Rel the O2"); only the relation breaks. One "the" pushes object 1 toward the lower left of object 2 on average, whatever relation was asked for.

The circuit explains why. Inserting "the" shifts the shape2 embedding by an average vector $\Delta V_{\text{the}}$ that overlaps several relation directions. L3H7 reads the shifted token and paints a wrong gradient: the cosine between its maps and the reference ramps goes from -0.98 to 0.45. Locative inversion and color synonyms do similar damage.

![Figure 27 from Wang, Fan and Pan (2026): why T5-DiT breaks. A: the shift from adding "the" overlaps relation directions. B: L3H7's maps lose alignment. C: the generated displacement drifts systematically. D: other perturbations also hit spatial accuracy only.](/imgs/blogs/circuit-mechanisms-for-spatial-relation-generation-in-diffusion-transformers-fig5.webp)

Two extensions round this out. A CLIP-encoded DiT-mini has a T5-like circuit: shape2 matters most, and screening finds two layer-0 heads (L0H2, L0H5) whose joint ablation drives relation accuracy toward chance (1 in 8). Unlike T5-DiT, CLIP-DiT-B is fairly robust across templates (0.759 to 0.794).

On pretrained PixArt-Sigma, relations are weak: only 8 of 30 object pairs show nontrivial accuracy once seed and image priors are controlled. Masking on "A dog is on the left of a human" finds spatial information mostly in the relation token but spread across nouns, and the screen flags three salient heads. Promising for the tools, but thin evidence next to the toy models.

**What may not transfer.** An 8 by 8 grid with fixed sinusoidal positions makes "query is roughly position" a good approximation; at 64 by 64 with RoPE it may not be. A 20-token context makes word groups trivial. And T5's brittleness is partly data: the model saw one template, so any shift in the fused token is out of distribution.

## Critique

**What is strong.** Each circuit is found by one method, confirmed by ablation, and then driven by an intervention (injecting L2H8's output, editing the shape2 vector). The weight-space screen tests a precise hypothesis across every head in seconds. Holding DiT, data and training fixed while swapping only the encoder isolates the cause, and the answer unites the "blame attention" and "blame the encoder" camps.

**What is weak.** The robustness comparison favors RTE: to it, "the" is one more unrelated token, while T5 is designed to let every word color every other. Training T5-DiT on a few templates might close the gap, and that control is missing. The ablation numbers (67% to 33%) sit below Table 2's 0.843 and come from a different evaluation set the paper does not reconcile. The step-0 argument leans on PixArt's fixed positions without testing other schemes.

**Missing experiments.** Template augmentation for T5-DiT; an MMDiT (SD3 or Flux style) where text tokens are updated by image tokens; a dose-response sweep of the vector edit's scale.

**What would change my mind.** If a T5-DiT trained on varied templates still had no relation-word head and still collapsed under fillers, I would accept the strong reading that T5-style encoders steer DiTs into a brittle fused solution. If light augmentation produced an RTE-like two-stage circuit, the brittleness is a data artifact and the encoder-as-bottleneck claim weakens.

## Where this goes next

The natural follow-up is circuit analysis of attribute binding and counting in large pretrained DiTs. The paper makes it tractable because its tools need only a hypothesis map and a feature vector, not a hand-traced path. These are my extrapolations, not the paper's claims.

1. **A binding circuit in PixArt-Sigma.** *Question:* does "red cube and blue sphere" use tag-then-fill, with one head tagging the cube region and another reading "red" there? *Why now:* synopses need only masks and word groups, and open-vocabulary segmenters supply masks on real images. *First experiment:* synopses for (cube pixels, own color word versus other color word) over 200 two-object prompts, then ablate the top head and count color swaps.

2. **Counting as a multi-peak tag.** *Question:* for "three apples", does a head produce a three-peak positional tag keyed by the count word? *Why now:* the screen generalizes by swapping the ramp $T_m$ for a k-peak template. *First experiment:* train a toy DiT on 1 to 5 identical shapes, screen heads against k-peak templates using the count word's embedding, and check whether peaks track k.

3. **Does augmentation buy back the modular circuit?** *Question:* trained on many templates, does T5-DiT grow a relation-word head? *Why now:* the toy setup is cheap and variance partitioning gives a scalar to track. *First experiment:* train with 1, 5 and 20 templates; plot relation partial $R^2$ in shape2, the relation-word synopsis peak and filler-word accuracy.

4. **Training-free layout steering.** *Question:* can relation vectors fix PixArt-Sigma's spatial errors without attention guidance? *Why now:* the edit works in toy T5-DiT and the factorization applies to any contextual embedding. *First experiment:* on PixArt-Sigma's 8 workable pairs, extract $V_{\text{rel}}$ from noun tokens, apply the edit, and compare accuracy and quality against Attend-and-Excite.

5. **Relation circuits in joint-attention DiTs.** *Question:* in MMDiT, does the relation stay in the noun or migrate through the text stream? *Why now:* the QK screen runs per layer at no sampling cost. *First experiment:* screen each layer of an MMDiT checkpoint with that layer's noun embedding as the key and find where ramp alignment first appears.

The sibling post on [how vision-language models localize objects](/blog/paper-reading/computer-vision/mechanisms-of-object-localization-in-vision-language-models) runs the reverse direction, reading location out of an image. For why a found circuit needs more than one test, see [hypothesis-testing the circuit hypothesis in LLMs](/blog/paper-reading/ai-interpretability/hypothesis-testing-the-circuit-hypothesis-in-llms).

## References

- Binxu Wang, Jingxuan Fan, Xu Pan. *Circuit Mechanisms for Spatial Relation Generation in Diffusion Transformers.* CVPR 2026 (highlight). [arXiv:2601.06338](https://arxiv.org/abs/2601.06338). [CVPR page](https://cvpr.thecvf.com/virtual/2026/poster/37559).
- Junsong Chen et al. *PixArt-alpha: Fast Training of Diffusion Transformer for Photorealistic Text-to-Image Synthesis*, 2023.
- Nelson Elhage et al. *A Mathematical Framework for Transformer Circuits*, 2021 (the QK and VO view).
- Hila Chefer et al. *Attend-and-Excite*, 2023; Gaoyang Zhang et al. *CoMPaSS*, 2024 (the attention and encoder camps).
- On this site: [Diffusion Transformers (DiT)](/blog/machine-learning/image-generation/diffusion-transformers-dit).
