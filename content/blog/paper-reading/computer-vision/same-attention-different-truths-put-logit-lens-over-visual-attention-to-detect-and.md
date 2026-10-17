---
title: "Same Attention, Different Truths: Reading What an LVLM Looks At to Catch Hallucinated Objects"
publishDate: "2026-10-17"
date: "2026-10-17"
description: "Why hallucinated objects get just as much visual attention as real ones, how a logit lens on the attended patches tells them apart, and how mask-and-regenerate plus a visual-logit fusion cut CHAIR hallucination on LLaVA-1.5 roughly in half without training."
category: "paper-reading"
subcategory: "Computer Vision"
tags: ["interpretability-analysis", "cvpr-2026", "cvpr-2026-highlight", "paper-reading", "object-hallucination", "logit-lens", "vision-language-models", "llava", "training-free", "attention-analysis"]
author: "Hiep Tran"
featured: true
readTime: 18
image: "/imgs/blogs/same-attention-different-truths-put-logit-lens-over-visual-attention-to-detect-and-fig1.webp"
excerpt: "Real and hallucinated objects receive the same visual attention. What differs is what the attended patches mean, and a logit lens can read it."
paper:
  title: "Same Attention, Different Truths: Put Logit-Lens over Visual Attention to Detect and Mitigate LVLM Object Hallucination"
  authors: "Zichuan Wang, Songlin Yang, Bo Peng, Zhenchen Tang, Yang Li, Beibei Dong, Jing Dong"
  venue: "CVPR 2026 (Highlight)"
  url: "https://arxiv.org/abs/2608.07302"
---

> [!tldr]
> - **Claim.** Large vision-language models (LVLMs) do not hallucinate objects because they stop looking at the image. In LLaVA-1.5-7B, real and hallucinated object tokens get the same image-attention magnitude in layers 20–27.
> - **Mechanism.** Project the hidden states of the top-3 attended image patches through the model's own output head (a logit lens). Patches behind a real "chair" decode to "chair". Patches behind a hallucinated "bowl" decode to something else.
> - **Two kinds of hallucination.** Mask those patches and regenerate. If the object vanishes, it was *visual uncertainty* (a confusable blob). If it survives and attention drifts elsewhere, it was a *contextual prior* (kitchens have microwaves).
> - **Results.** A training-free Detect-Mitigate pipeline takes LLaVA-1.5-7B from CHAIR_S 49.8 to 26.8 and CHAIR_I 20.4 to 10.0, beats PAI and Devils on four models, and keeps AMBER object coverage at 51.2 (greedy is 51.0).
> - **Where it is thin.** Every fix costs at least one full regeneration, latency is never reported, the layer band is tuned on one model, and synonym matching runs through WordNet on subword fragments.

![Figure 1 from Wang et al. (2026): LLaVA describes a garden table, mentions a real "chair" and a hallucinated "bowl"; the attention heatmaps (a) look equally focused and the mean attention magnitudes (b) on a COCO subset are statistically indistinguishable.](/imgs/blogs/same-attention-different-truths-put-logit-lens-over-visual-attention-to-detect-and-fig1.webp)

The figure above is the whole paper in one panel. Both words drew focused attention, and the bars in (b) overlap. If weak attention caused hallucination, the "bowl" bar would be shorter.

## The problem: "it isn't looking" is an incomplete diagnosis

Object hallucination is an LVLM describing something that is not in the image. Ask LLaVA to "Describe this image" for a garden table and it may add "a vase, a bowl, and a cup", where only the vase exists. For captioning, retrieval labels or any downstream agent that trusts the description, this is the failure that matters most.

The dominant explanation for two years has been **insufficient visual attention**. Either textual priors swamp the multimodal interaction, or the visual pathway processes the image badly and attention goes to the wrong places. The fixes follow from the story: PAI amplifies attention on image tokens, Devils exploits mid-layer visual signals, ClearSight boosts the visual pathway, and VCD contrasts the real image against a perturbed one to subtract the language prior.

This paper asks a narrower question first: **do hallucinated tokens actually attend less?** If not, turning up the attention knob treats a symptom that is not there, and we need to know what the attended region *contains*. The logit lens answers that, and splits the problem into two mechanisms with two fixes.

Our walkthrough of [the few attention heads that do visual grounding](/blog/paper-reading/computer-vision/your-large-vision-language-model-only-needs-a-few-attention-heads-for-visual-grounding) showed attention can localize objects well. This paper shows well-localized attention can still ground the wrong word.

## Contributions

1. **Same attention:** no systematic attention-magnitude gap between real and hallucinated object tokens (LLaVA-1.5-7B, 500 COCO images).
2. **Different truths:** a logit lens over the top-attended patches decodes to the generated word only for real objects.
3. **Two mechanisms**, split by a masking intervention: visual uncertainty and contextual prior, about 2:1.
4. **A training-free Detect-Mitigate framework:** LLCC detects, HARM fixes type 1, VEED fixes type 2, with best CHAIR and AMBER numbers on four models.

## Method

The analysis asks how much the model attends, what it attends to, and why. Each answer becomes one component of the framework.

### Setup and notation

An LVLM is a vision encoder, a projector $M$ and a language model. Image tokens $v_i$ are projected to $\tilde v_i = M(v_i)$ and concatenated with system and text tokens for a causal decoder. LLaVA-1.5 uses a 24 by 24 patch grid, so $N = 576$ image tokens. At step $t$ the output head $W_U$ (the unembedding matrix, shape vocabulary size by hidden size) maps the last-layer hidden state $h_t^{(L)}$ to the next-token distribution:

$$
p(x_t \mid I, X_{\lt t}) = \mathrm{softmax}\!\left(W_U\, h_t^{(L)}\right)
$$

Attention is the usual causal softmax. With query $q_t$, key $k_j$, key dimension $d_k$, and mask $m_{tj}$ equal to zero for $j \le t$ and $-\infty$ otherwise:

$$
\alpha_{tj} = \mathrm{softmax}\!\left(\frac{q_t^\top k_j}{\sqrt{d_k}} + m_{tj}\right)
$$

We write $\alpha^{(l)}_{tj}$ for the weight that the token generated at step $t$ places on image token $j$ at layer $l$, averaged over heads. That single tensor, of shape layers by image tokens per generated token, is all the analysis needs.

### Step 1: how much does the model attend? The four-stage profile

**The problem.** Before claiming attention is not the cause, you need a fair place to measure it. Attention to the image varies wildly by layer, so averaging over all 32 layers mixes signal with noise.

**The intuition.** A writer describing a photo glances at everything, thinks about grammar, looks hard at the photo to check the noun, then polishes the sentence. Measure during the "look hard" phase.

**The mechanism.** Log each generated token's attention share on system, image and text tokens per layer. Non-object tokens barely look at the image after layer 2. Object tokens show four stages:

| Stage | Layers (LLaVA-1.5-7B) | What happens |
| --- | --- | --- |
| Initialization | 1–2 | High attention on both image and text while context is aggregated |
| Text-attention | 3–19 | Text dominates, image attention stays low |
| **Image-attention** | **20–27** | Image attention rises and dominates; evidence for the candidate object is gathered |
| Language-organization | 28–32 | Image attention falls, text rises as the sentence is organized |

![Figure 2 from Wang et al. (2026): attention ratio by layer for object tokens (a) and non-object tokens (b). Only object tokens show the mid-to-late surge onto the image, shaded as the Image-Attention Stage (layers 20–27).](/imgs/blogs/same-attention-different-truths-put-logit-lens-over-visual-attention-to-detect-and-fig2.webp)

**The measurement.** Call the image-attention stage $S_{IA}$ (layers 20–27). Use CHAIR, which checks generated object words against COCO annotations, to label each object token real or hallucinated. Then compare their mean image attention inside $S_{IA}$. Figure 1(b) is the result: no systematic difference. Layer-by-layer heatmaps (their Figure 3) show both "chair" and the hallucinated "bowl" converging on compact regions in the same layers with comparable magnitude.

**When it fails.** The stage boundaries are read off one model's curve. The main paper never prints the bands it used for the other three models, which matters for reproduction.

### Step 2: what does the model attend to? A logit lens on image patches

**The problem.** If magnitude does not separate real from hallucinated, something about the *content* of the attended patches must.

**The intuition.** A logit lens is like asking the model to read its own notes early. Instead of waiting for the last layer, take an intermediate hidden state and pass it straight through the final output head, as if the network ended there. In text LLMs this shows the answer forming layer by layer. Here, the authors point it at **image tokens**, which asks: "if this patch's hidden state had to be a word, which word would it be?"

**The math.** For hidden state $h_t^{(l)}$ at layer $l$:

$$
p_{\text{lens}}^{(l)}\!\left(\cdot \mid h_t^{(l)}\right) = \mathrm{softmax}\!\left(W_U\, h_t^{(l)}\right)
$$

No new parameters: $W_U$ is the same unembedding used for generation. For each generated object token $o_t$, average its attention over the stage,

$$
\bar\alpha_{tj} = \frac{1}{|S_{IA}|} \sum_{l \in S_{IA}} \alpha^{(l)}_{tj},
$$

take the $k$ most attended image tokens $\Omega_t = \mathrm{TopK}(\bar\alpha_t, k)$, and decode each one's hidden state through the lens.

![Figure 4 from Wang et al. (2026): logit-lens readouts of the most attended patches. For real object tokens (a) the attended suitcase decodes to fragments like "suit" and "lug"; for hallucinated tokens (b) such as "cell phone" the attended blurry patch decodes to unrelated words.](/imgs/blogs/same-attention-different-truths-put-logit-lens-over-visual-attention-to-detect-and-fig4.webp)

**What comes out.** Real objects are **semantically consistent**: when LLaVA writes "suitcase", the attended suitcase patches decode to "suit" or "lug". Hallucinated objects are **inconsistent**: when it writes "cell phone", the attended "blurry patch on the ground" does not decode to anything phone-like. The attention went somewhere specific; the thing it found there was not the word that came out.

![Redrawn: the same attention mass, two different readouts. Left, the top-3 patches behind a real "chair" decode to chair-like words. Right, the top-3 patches behind a hallucinated "bowl" carry the same attention but decode elsewhere (readouts illustrative).](/imgs/blogs/same-attention-different-truths-put-logit-lens-over-visual-attention-to-detect-and-1.webp)

**When it fails.** The logit lens is noisy at intermediate layers because hidden states are not yet in the output basis. It works here partly because $S_{IA}$ is late (20–27 of 32). Readouts are also subword pieces ("lug", "suit"), which complicates matching.

### Step 3: why does it attend there? The masking intervention

**The problem.** A correlation between inconsistent readouts and hallucination is not yet a mechanism. The authors intervene.

**The mechanism.** Take the top-$k$ attended patches $\Omega_t$ for a hallucinated token, black them out (or fill them with the mean color) in the input image, and regenerate the caption with the same prompt. Two behaviors appear:

- **Visual uncertainty.** The hallucinated word disappears. The attended region was ambiguous (a round, ceramic-looking area becomes "bowl"; a dark blurry rectangle becomes "cell phone"). The model forced a label out of weak evidence. Remove the evidence and the label goes with it.
- **Contextual prior.** The word persists and attention moves to other patches. The logit lens shows those new patches decode to objects that *are* present, yet the model still writes "microwave" in a kitchen. Attention here acts like a procedure: the model has learned it must look somewhere before naming an object, so it looks, and then the co-occurrence prior decides.

![Figure 5 from Wang et al. (2026): (a) a visual-uncertainty case where masking the attended region removes the hallucinated "bowl"; (b) a contextual-prior case where "microwave" survives masking and attention shifts to new regions. HOT is the hallucinated object token.](/imgs/blogs/same-attention-different-truths-put-logit-lens-over-visual-attention-to-detect-and-fig5.webp)

Over 500 images, LLaVA-1.5-7B's ratio of visual-uncertainty to contextual-prior hallucinations is about **2:1**. Hence two remedies: boosting attention does nothing for the third of cases where the model saw real evidence and overruled it.

### The Detect-Mitigate framework

![Figure 6 from Wang et al. (2026): the full pipeline. LLCC flags hallucinated object tokens via logit-lens readouts; HARM masks their high-attention regions and regenerates, which both fixes type 1 and sorts what is left into type 2; VEED then rewrites type 2 tokens by fusing visual-patch logits into decoding.](/imgs/blogs/same-attention-different-truths-put-logit-lens-over-visual-attention-to-detect-and-fig6.webp)

Step 1 picks the layers and the object tokens, Step 2 is the detector, Step 3 is the classifier and type-1 fix, and a new decoding rule handles type 2.

### Technique 1: Logit-Lens Consistency Check (LLCC)

**The problem.** Detect hallucinated objects during generation, without a ground-truth annotation and without running the lens on every token.

**The intuition.** A fact-checker who, for every noun you write, looks at the part of the photo you were staring at and asks "does that region actually say this?"

**The mechanism, in three steps.**

1. **Object token identification.** Object tokens are the ones that look hard at the image in $S_{IA}$. Compute the average image attention

$$
A_{\text{img}}(o_t) = \frac{1}{|S_{IA}|} \sum_{l \in S_{IA}} \sum_{j=1}^{N_{\text{img}}} \alpha^{(l)}_{tj},
$$

where $N_{\text{img}}$ is the number of image tokens. Tokens with $A_{\text{img}}(o_t) > \tau_{\text{attn}}$ are treated as objects, with $\tau_{\text{attn}} = 0.15$. This is a cheap filter: no lexicon, no parser, just the attention profile from Step 1.

2. **Semantic consistency check.** For each object token, take the top-$k$ attended patches ($k = 3$), decode each through the lens, and keep the argmax word $v^i_t = \arg\max_{v \in \mathcal V} \mathrm{softmax}(W_U h(\omega^i_t))_v$, where $\mathcal V$ is the vocabulary and $\omega^i_t$ is the $i$-th selected patch.

3. **Label.** Compare $o_t$ against the decoded words with a semantic similarity $\mathrm{Sim}$ (WordNet, following CHAIR and AMBER's synonym handling) and threshold $\tau_{\text{sim}} = 0.8$. The token is real if a decoded word clears the threshold, and a hallucination otherwise.

The paper writes the rule per decoded word; I read it as "real if any readout matches".

**Worked example.** The caption fragment "The table has a vase, a bowl" runs through both checks in the figure below. Only "table", "vase" and "bowl" clear the 0.15 attention filter. "Vase" has two readouts at similarity 1.0, so it is real. "Bowl" decodes to "white", "plate" and "blur", best similarity 0.62, so it is flagged.

![Redrawn: LLCC on a toy caption. Top, per-token image attention against the 0.15 object threshold. Bottom, WordNet similarity of each object's three logit-lens readouts against the 0.8 consistency threshold (values illustrative, thresholds from the paper).](/imgs/blogs/same-attention-different-truths-put-logit-lens-over-visual-attention-to-detect-and-2.webp)

Here is the check as a PyTorch-shaped sketch against a Hugging Face LLaVA model. It is my reconstruction from the equations, not the authors' code; `image_slice`, the layer band and the WordNet helper are the parts you would adapt.

```python
import torch
from nltk.corpus import wordnet as wn

S_IA = range(19, 27)            # layers 20-27, zero-indexed
TAU_ATTN, TAU_SIM, K = 0.15, 0.8, 3

def wn_sim(a: str, b: str) -> float:
    sa, sb = wn.synsets(a, pos="n"), wn.synsets(b, pos="n")
    if not sa or not sb:
        return 1.0 if a == b else 0.0
    return max((x.wup_similarity(y) or 0.0) for x in sa for y in sb)

@torch.no_grad()
def llcc_step(out, step_pos, image_slice, lm_head, final_norm, tokenizer, word):
    # out: model output with output_attentions=True, output_hidden_states=True
    # attn[l]: [batch, heads, q_len, k_len]; hidden[l]: [batch, seq, d]
    attn = torch.stack([out.attentions[l][0, :, step_pos, image_slice].mean(0)
                        for l in S_IA])                 # [|S_IA|, N_img]
    a_img = attn.sum(-1).mean().item()                  # eq. 5
    if a_img <= TAU_ATTN:
        return "not-object"
    alpha_bar = attn.mean(0)                            # eq. 4, [N_img]
    top = alpha_bar.topk(K).indices + image_slice.start
    readouts = []
    for l in S_IA:                                      # lens on each stage layer
        h = final_norm(out.hidden_states[l + 1][0, top])  # [K, d]
        ids = lm_head(h).argmax(-1)                     # eq. 7
        readouts += tokenizer.convert_ids_to_tokens(ids.tolist())
    words = {r.lstrip("▁").lower() for r in readouts}
    best = max(wn_sim(word, w) for w in words)
    return "real" if best > TAU_SIM else "hallucination"  # eq. 8
```

The paper does not pin down which stage layers the lens reads (I decode all eight) or whether the final norm precedes $W_U$ (standard logit-lens practice).

**When it fails.** WordNet does not know "lug" is a fragment of "luggage", so a strict implementation flags real objects with subword readouts. The paper does not give its matching rule for fragments.

### Technique 2: High-Attention Regions Masking (HARM)

**The problem.** Remove type-1 hallucinations, and at the same time find out which flagged tokens are type 2.

**The intuition.** If a witness keeps pointing at a smudge on the photo and calling it a bowl, cover the smudge and ask again.

**The math.** Let $\Omega$ be the union of top-$k$ patches over all flagged tokens. Build a binary mask $\mathbf{M}$ that is 1 on pixels of patches in $\Omega$ and 0 elsewhere, pick a fill value $\mu$ (mean image color or zero), and form

$$
I^{\text{mask}} = (1 - \mathbf{M}) \odot I + \mathbf{M} \odot \mu,
$$

where $\odot$ is elementwise product. Regenerate with the same prompt to get $O_{\text{new}}$. Then classify: type 1 (visual uncertainty) if $o_t \notin O_{\text{new}}$, type 2 (contextual prior) if $o_t \in O_{\text{new}}$.

![Redrawn: HARM as a before and after. The attended patches behind the hallucinated token are filled with the mean color and the caption regenerated. A type-1 token disappears; a type-2 token survives while its attention drifts to new patches.](/imgs/blogs/same-attention-different-truths-put-logit-lens-over-visual-attention-to-detect-and-3.webp)

**Worked example.** With LLaVA-1.5's 336-pixel input, each of the 576 patches is 14 by 14 pixels, so masking $k = 3$ patches per flagged object hides about 0.5% of the image per object. Two flagged objects is roughly 1%. That is why the authors can claim the mask "limits impact on original content": the rest of the caption is regenerated from a nearly identical image.

**When it fails.** A small real object under the mask vanishes too, trading a hallucination for an omission. And classification is string membership: if the new caption says "dish" instead of "bowl", the token counts as gone though the hallucination survived.

### Technique 3: Visual Evidence Enhanced Decoding (VEED)

**The problem.** Type-2 tokens survive masking because the language prior wins even when the attended patches hold correct evidence. Masking more does not help; the decision itself has to be rebalanced.

**The intuition.** A second opinion from the patch. The decoder says "microwave" because kitchens have microwaves. The patch it is staring at says "counter". Blend the two votes so the patch gets a real say.

**The math.** Let $\omega^{\max}_t$ be the single most attended patch at step $t$. Its visual logits are $z^{\text{vis}}_t = W_U h(\omega^{\max}_t)$, computed by the lens already. Let $z^{\text{mask}}_t$ be the decoder's logits at step $t$ given the masked image $I^{\text{mask}}$ and the prefix so far. The fused next-token distribution is

$$
p_t = \mathrm{softmax}\!\Big((1-\alpha)\, z^{\text{vis}}_t + \alpha\, z^{\text{mask}}_t\Big),
$$

where $\alpha \in [0, 1]$ sets the mix. Note the direction: a **lower** $\alpha$ gives the visual patch **more** weight.

**Worked example.** Four candidate tokens: microwave, counter, sink, oven. Suppose $z^{\text{vis}} = [1, 4, 2, 1]$ and $z^{\text{mask}} = [5, 3, 2, 2]$. On their own, the patch puts 0.81 on "counter" and the decoder puts 0.81 on "microwave". With $\alpha = 0.5$ the fused logits are $[3.0, 3.5, 2.0, 1.5]$, a softmax of $[0.31, 0.51, 0.11, 0.07]$, and "counter" wins.

![Redrawn: VEED on four candidate tokens. The patch's logit-lens distribution favors "counter", the masked-image decoder favors "microwave", and the alpha = 0.5 fusion flips the argmax to "counter" (logits illustrative).](/imgs/blogs/same-attention-different-truths-put-logit-lens-over-visual-attention-to-detect-and-4.webp)

**When it fails.** Lens logits from an intermediate layer are not calibrated like final logits, so a fixed $\alpha$ means different things for different patches. And a patch only votes for what it depicts: where the right continuation is not an object word, it can inject a noun the sentence did not need.

## Experiments and results

**Detection.** 500 COCO 2014 images, captions from LLaVA-1.5-7B, CHAIR labels as ground truth, $\tau_{\text{attn}} = 0.15$, $k = 3$, $\tau_{\text{sim}} = 0.8$.

| Detector | Signal | Precision | Recall | F1 |
| --- | --- | --- | --- | --- |
| Uncertainty Score | output probability | 0.597 | 0.642 | 0.618 |
| InterConf | decoded probabilities, all layers | 0.672 | 0.691 | 0.681 |
| SVAR | summed attention, layers 5–18 | 0.650 | 0.722 | 0.684 |
| **LLCC** | lens on attended patches, layers 20–27 | **0.787** | **0.796** | **0.793** |

The SVAR row is the paper's argument in miniature. SVAR measures attention *quantity* in middle layers and lands at 0.68 F1. LLCC measures attention *content* in the image-attention band and gains 11 points.

**Mitigation on CHAIR** (500 COCO val images, "Describe this image", lower is better):

| Method | LLaVA-1.5-7B S / I | LLaVA-1.5-13B S / I | Shikra-7B S / I | Qwen2-VL-7B S / I |
| --- | --- | --- | --- | --- |
| Greedy | 49.8 / 20.4 | 47.8 / 19.8 | 58.4 / 22.2 | 31.4 / 12.7 |
| OPERA | 42.9 / 18.7 | 42.1 / 16.4 | 38.1 / 16.7 | 28.0 / 10.3 |
| Devils | 32.1 / 13.7 | 35.4 / 14.1 | 32.8 / 13.1 | n/a |
| PAI | 29.8 / 13.2 | 33.2 / 13.5 | 37.9 / 15.0 | 24.7 / 8.6 |
| **Ours** | **26.8 / 10.0** | **31.3 / 12.4** | **31.4 / 12.7** | **24.0 / 8.3** |

**AMBER** (1,004 images; CHAIR, Hal and Cog lower is better, Cover higher is better), LLaVA-1.5-7B:

| Method | CHAIR | Cover | Hal | Cog |
| --- | --- | --- | --- | --- |
| Greedy | 6.9 | 51.0 | 32.0 | 3.3 |
| Devils | 3.5 | 50.2 | 19.9 | 1.3 |
| PAI | 4.2 | 50.7 | 18.4 | 1.6 |
| **Ours** | **2.8** | **51.2** | **14.7** | **1.2** |

Two derived numbers carry the story. On LLaVA-1.5-7B, sentence-level CHAIR drops 46% relative to greedy (49.8 to 26.8). On AMBER, the share of responses with any hallucination falls from 32.0% to 14.7% while coverage stays flat, where most baselines buy their gains by mentioning fewer objects. The margin over PAI shrinks on the stronger Qwen2-VL (24.7 to 24.0 on CHAIR_S), which is what you would expect if a better model has fewer type-1 confusions left to fix.

**What is load-bearing and might not transfer.** The layer band was read off LLaVA-1.5-7B. Thresholds were set on CHAIR-labelled COCO captions and mitigation is also scored on COCO. Everything is greedy with one prompt, and the 2:1 type ratio is reported for one model only.

## Critique

**What is strong.** The central measurement is simple and falsifiable. The masking intervention turns a correlation into a causal split. And the detector reuses the model's own unembedding: no training, no extra model.

**What is weak.**

- **Cost is invisible.** HARM needs a second full generation; VEED decodes against $I^{\text{mask}}$ again. That is at least 2× decode work per flagged caption, and latency is never reported.
- **"No systematic difference" is means with standard deviations**, not a distributional test, and head-averaged. A per-head breakdown might separate the populations.
- **The taxonomy is defined by the intervention.** A token that becomes a synonym, or drops out because the new caption is shorter, counts as type 1.
- **VEED's scale mismatch** is not discussed in the main text, and the $\alpha$ sweep is left to the appendix.

**Missing ablation.** HARM alone versus HARM plus VEED per model, with the type-2 share alongside, plus a POPE or MME check that general ability is untouched.

**What would change my mind.** If a per-head analysis showed that hallucinated tokens draw measurably less attention from the specific grounding heads that localization work has identified, then "same attention" would be an averaging artifact, and the content signal would be a proxy for which heads fired. Conversely, if LLCC kept its F1 when the layer band is set blind on a new model family, I would treat the four-stage profile as a general property of LVLMs rather than a LLaVA trait.

## Where this goes next

The authors' own framing suggests the general move: read the *content* of what attention selects, then use that readout to steer decoding. Four directions where the paper makes that tractable today, flagged as my extrapolation.

1. **Single-pass content-gated attention.** *Question:* can we skip regeneration by down-weighting attention, inside layers 20–27, to patches whose readout disagrees with the candidate token? *Why now:* the disagreement is detectable at the step it happens, for one $W_U$ matmul on three hidden states. *First experiment:* gate $\alpha^{(l)}_{tj}$ by readout similarity, renormalize, and compare CHAIR and wall-clock against HARM on the same 500 images.

2. **Attribute and count hallucination.** *Question:* does the inconsistency appear for "red car" over a blue car, or "three chairs" over two? *Why now:* the lens decodes colors and numerals as easily as nouns. *First experiment:* run LLCC on adjective and numeral tokens on AMBER's attribute split, checking first whether those tokens show the four-stage profile at all.

3. **Relation hallucination.** *Question:* for "a cup on the table", do the patches attended by "on" decode to both objects, and do false relations break the pair? *Why now:* top-$k$ gives regions per token; a relation is a constraint between two such sets. *First experiment:* on Visual Genome relation captions, test whether false relations have lower overlap between relation and object regions.

4. **A learned readout instead of WordNet.** *Question:* does a tuned lens (a per-layer affine map trained to match final logits) fix the subword and calibration problems? *Why now:* the paper isolates exactly those two weak links. *First experiment:* train a tuned lens for layers 20–27 on 10k COCO captions, then rerun detection and VEED to see whether one $\alpha$ transfers across models.

Why $k = 3$ suffices connects to [what visual tokens really encode](/blog/paper-reading/computer-vision/what-do-visual-tokens-really-encode-uncovering-sparsity-and-redundancy-in-multimodal): object information concentrates in few tokens. [Mechanisms of object localization](/blog/paper-reading/computer-vision/mechanisms-of-object-localization-in-vision-language-models) traces how the model finds the region at all, and the text-only analogue is [knowledge awareness and hallucination in language models](/blog/paper-reading/ai-interpretability/do-i-know-this-entity-knowledge-awareness-and-hallucinations-in-language-models).

## References

- Wang, Z., Yang, S., Peng, B., Tang, Z., Li, Y., Dong, B., Dong, J. *Same Attention, Different Truths: Put Logit-Lens over Visual Attention to Detect and Mitigate LVLM Object Hallucination.* CVPR 2026. [arXiv:2608.07302](https://arxiv.org/abs/2608.07302) · [CVF open access](https://openaccess.thecvf.com/content/CVPR2026/html/Wang_Same_Attention_Different_Truths_Put_Logit-Lens_over_Visual_Attention_to_CVPR_2026_paper.html) · [CVPR poster page](https://cvpr.thecvf.com/virtual/2026/poster/36231)
- Code (announced): [github.com/wzczc/SADT](https://github.com/wzczc/SADT)
- nostalgebraist. *Interpreting GPT: the logit lens.* LessWrong, 2020.
- Rohrbach et al. *Object Hallucination in Image Captioning* (CHAIR). 2018.
- Liu, Zheng, Chen. *Paying More Attention to Image* (PAI). ECCV 2024.
- Related on this site: [few attention heads for visual grounding](/blog/paper-reading/computer-vision/your-large-vision-language-model-only-needs-a-few-attention-heads-for-visual-grounding), [what visual tokens really encode](/blog/paper-reading/computer-vision/what-do-visual-tokens-really-encode-uncovering-sparsity-and-redundancy-in-multimodal), [mechanisms of object localization](/blog/paper-reading/computer-vision/mechanisms-of-object-localization-in-vision-language-models).
