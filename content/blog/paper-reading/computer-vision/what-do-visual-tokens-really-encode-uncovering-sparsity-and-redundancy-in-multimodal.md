---
title: "What Do Visual Tokens Really Encode? Sink, Dead and Alive Tokens Inside an MLLM"
publishDate: "2026-10-16"
date: "2026-10-16"
description: "How a simple input-embedding probe shows that 40% of an MLLM's visual tokens carry no image content, that the rest arrive already readable, and that the LLM's own visual computation is mostly redundant."
category: "paper-reading"
subcategory: "Computer Vision"
tags: ["paper-reading", "interpretability-analysis", "cvpr-2026", "cvpr-2026-highlight", "multimodal-llm", "visual-tokens", "token-pruning", "attention-sinks", "embedlens", "llava"]
author: "Hiep Tran"
featured: true
readTime: 19
image: "/imgs/blogs/what-do-visual-tokens-really-encode-uncovering-sparsity-and-redundancy-in-multimodal-1.webp"
excerpt: "Visual tokens entering an MLLM split into sink, dead and alive groups. Only the alive 60% carry image content, they arrive already readable, and most of the LLM's own visual computation can be skipped."
paper:
  title: "What Do Visual Tokens Really Encode? Uncovering Sparsity and Redundancy in Multimodal Large Language Models"
  authors: "Yingqi Fan, Junlong Tong, Anhao Zhao, Xiaoyu Shen"
  venue: "CVPR 2026 (highlight)"
  url: "https://arxiv.org/abs/2603.00510"
---

> [!tldr]
> **Claim.** In a LLaVA-style MLLM, the visual tokens that the projector hands to the language model are not a uniform stream. They split into three groups: **sink** tokens (image-agnostic attention placeholders), **dead** tokens (a big repetitive cluster that nobody reads) and **alive** tokens (about 60% of the input, the only ones carrying image-specific meaning).
>
> **Why it matters.** Dropping sinks and dead tokens costs nothing on LLaVA-1.5-7B; dropping the dead cluster even lifts the average score from 52.7 to 53.7. The alive tokens arrive already "pre-linguistic": a cosine lookup against the LLM's own input vocabulary recovers object labels before any LLM layer runs.
>
> **Most surprising.** The LLM's internal visual computation is largely redundant. Visual tokens enter with norms so large that the first few layers barely move them, and feeding them in at layer 4 or 5 instead of layer 0 slightly *improves* the average.
>
> **Where it breaks.** The headline pruning and skipping numbers are all on LLaVA-1.5-7B with a CLIP encoder. The paper's own appendix says Qwen2.5-VL, InternVL3 and Qwen3-VL do not show the same repetitive dead cluster.

The standard multimodal LLM recipe is almost embarrassingly simple: run a CLIP-style vision encoder, push its patch features through a small projector, and paste the result into the language model as if it were a sequence of words. LLaVA-1.5 does exactly this with 576 tokens per image. It works well enough that nobody has had to ask what those 576 tokens actually contain.

This paper asks. Its answer is that the tokens are wildly unequal, that a large share of them are inert, and that the language model does far less "visual thinking" than its compute bill suggests. 

## The tension the paper starts from

CLIP is trained to align a *whole image* with a *whole caption*. Nothing in its contrastive loss says each 14 by 14 patch must carry a crisp, word-like meaning. Yet the LLM consumes those patches one token at a time, the same way it consumes words.

So the authors ask four questions:

- Do all patches carry meaningful semantics?
- Are patch embeddings already "pre-linguistic," readable by the LLM as they arrive, or does the LLM have to translate them first?
- How much does the LLM's own processing of visual tokens (visual self-attention and visual FFN) actually contribute?
- If some of it matters, at which depth?

The figure below is the answer in one picture. Read it as a schematic of the 24 by 24 grid, not as measured positions.

![Schematic of LLaVA's 24 by 24 visual token grid: alive tokens on the objects, dead tokens across the background, a handful of sink tokens, and a bar showing that about 40% can be dropped](/imgs/blogs/what-do-visual-tokens-really-encode-uncovering-sparsity-and-redundancy-in-multimodal-1.webp)

## Technique 1: EmbedLens, reading a token against the input vocabulary

**The problem.** To say what a visual token "means," you need a decoder. The usual tool is the logit lens: push a hidden state through the LLM's unembedding matrix at the output and read the top words. The trouble is that by the time a state reaches the unembedding, the LLM has already processed it. You cannot tell whether the meaning was in the visual token from the start or was written in by the language backbone.

**The analogy.** Instead of asking "what word would the model *say* here," ask "which word in the dictionary does this vector *look like* right now." It is a nearest-neighbor lookup in the space the token was projected into.

**The mechanism.** Take the LLM's input embedding matrix, the table that turns token IDs into vectors. Compare the visual token against every row by cosine similarity and return the top-k words.

$$
\text{EmbedLens}(\mathbf{h}) = \operatorname{TopK}_{i \in \mathcal{V}} \; \frac{\mathbf{h}^{\top}\mathbf{e}_i}{\lVert \mathbf{h}\rVert_2 \, \lVert \mathbf{e}_i\rVert_2}
$$

Here $\mathbf{W}_E \in \mathbb{R}^{T \times d}$ is the input embedding matrix with $T$ vocabulary rows of width $d$, $\mathbf{e}_i$ is the row for token $i$, $\mathcal{V}$ is the vocabulary, and $\mathbf{h} \in \mathbb{R}^{d}$ is the target representation (a projected visual token, or any hidden state). Because it is cosine, the token's norm does not matter, only its direction.

**Tiny example.** For LLaMA-2 7B, $d = 4096$ and the vocabulary has 32,000 rows, so one EmbedLens call is a single matrix-vector product of shape (32,000 by 4,096) times (4,096). A visual patch sitting on a man's hat might come back with top-2 words "hat, black." That one patch is carrying both an object and a color.

![EmbedLens geometry: a visual token h compared by cosine angle to vocabulary embeddings, with alive tokens near the text centroid and a dead cluster far away](/imgs/blogs/what-do-visual-tokens-really-encode-uncovering-sparsity-and-redundancy-in-multimodal-2.webp)

**Does it find anything?** The authors score it on MS COCO 2017. For each annotated object label $t^{*}_j$, they check whether it appears in the top-5 EmbedLens words of any visual token at layer $\ell$:

$$
A^{(\ell)} = \frac{1}{N} \sum_{j=1}^{N} \mathbb{1}\!\left[\, t^{*}_j \in \text{EmbedLens}_{\text{top-}k}\!\big(\mathbf{h}^{(\ell)}_j\big) \right]
$$

where $N$ is the number of evaluated labels, $\mathbf{h}^{(\ell)}_j$ is the $j$-th visual token's representation at layer $\ell$, and $\mathbb{1}[\cdot]$ is 1 when the label is retrieved. Across LLaVA-1.5-7B and 13B, InternVL3-8B and Qwen2.5-VL, EmbedLens beats LogitLens at shallow and middle layers.

![Figure 3 from Fan et al. (2026): label-token matching accuracy of EmbedLens versus LogitLens across layers for four MLLMs](/imgs/blogs/what-do-visual-tokens-really-encode-uncovering-sparsity-and-redundancy-in-multimodal-fig2.webp)

**When it fails.** EmbedLens can only report concepts that exist as single vocabulary rows. A color or a shape that the vocabulary spells as two subword pieces is invisible to it, and so is any non-verbal geometric information (position, depth, texture). A miss from EmbedLens means "not word-like," not "empty."

## Technique 2: finding the three clusters

**The problem.** Before labelling individual tokens, the authors want the macro structure: do visual tokens group, and are the groups the same from image to image?

**The mechanism.** Anchor-based clustering on normalized post-projection embeddings. Pick an anchor $\mathbf{v}_a$, and every token whose cosine similarity with it clears a threshold $\tau$ joins its cluster:

$$
\mathcal{C}_a = \left\{\, j \;\middle|\; \frac{\langle \mathbf{v}_a, \mathbf{v}_j\rangle}{\lVert \mathbf{v}_a\rVert_2 \, \lVert \mathbf{v}_j\rVert_2} \ge \tau \right\}
$$

with $\tau = 0.9$ in the main analysis, and checked at 0.95 and 0.8 in the appendix. To ask whether a cluster is *image-agnostic*, they compare normalized cluster centroids across two images. With $S_{ij} = \mathbf{C}^{(1)}_i \cdot \mathbf{C}^{(2)}_j$ as the cosine between centroid $i$ of image 1 and centroid $j$ of image 2, the stability of cluster $i$ is its best match:

$$
s_i = \max_{j} \, S_{ij}
$$

An $s_i$ near 1, averaged over many image pairs, means the cluster shows up in the same place no matter what the image shows. All statistics come from 10K COCO samples.

**What falls out.** Three archetypes, visible in the t-SNE below:

| Cluster | Where it sits | Cross-image stability | Becomes |
| --- | --- | --- | --- |
| High-homogeneity | Its own tight spot | Approaches 100% | **Sink** tokens |
| Text-distancing | Far from text embeddings, the largest cluster (about 30% of tokens) | High | **Dead** tokens |
| Text-proximity | Several small clusters around the text centroid | Image-specific | **Alive** tokens |

![Figure 2 from Fan et al. (2026): t-SNE of visual and textual clusters in the word embedding space, with an example image whose patches are labelled by their EmbedLens words](/imgs/blogs/what-do-visual-tokens-really-encode-uncovering-sparsity-and-redundancy-in-multimodal-fig1.webp)

Notice the labels on the right panel. Patches on people read "people, four," a patch on a hat reads "hat, black," and the sink patches read nonsense subwords like "igh" and "jav." That last detail becomes a handy tracking tool: once a sink cluster's EmbedLens label is known, you can follow it through layers by its reference token ID.

## Technique 3: two kinds of sink, and why both are disposable

**The problem.** "Attention sink" already has a literature. In LLMs, a few tokens soak up attention without carrying content, as in [StreamingLLM's attention sinks](/blog/paper-reading/large-language-model/streaming-llm-attention-sinks). In ViTs, a few high-norm patches act as "registers" that people often believe store global image information. Are the image-agnostic clusters one of these?

**ViT sinks.** Following prior work, a patch is a ViT sink if its final-layer L2 norm in CLIP exceeds $\tau = 75$. Across 1,000 random images, the centroids of these sinks reach cross-image cosine similarity above 0.99 with variance below ${10}^{-5}$. They also land in exactly the most stable cluster from the clustering step. So the "global information" story is hard to square with the data: a vector that is the same for every image cannot encode anything about a particular image.

**LLM sinks.** These are found inside the language backbone with Kang et al.'s test: the ratio between the magnitude on a few fixed "sink channels" and the RMS norm of the hidden state,

$$
\phi(\mathbf{x}) = \max_{d \in \mathcal{D}_{\text{sink}}} \frac{\lvert \mathbf{x}[d] \rvert}{\sqrt{\tfrac{1}{D}\sum_{d'=1}^{D} \mathbf{x}[d']^{2}}}
$$

where $\mathbf{x}$ is a token's hidden state of width $D$, and $\mathcal{D}_{\text{sink}}$ is the set of sink channels (for a LLaMA-2 7B backbone, channels 1415 and 2533). A token with $\phi$ above 20 counts as a sink. The surprising part: these tokens only start acting like sinks after layer 2, yet they already have cross-image similarity above 0.95 at the embedding stage. The projector has learned to pick a consistent handful of patches as future sinks from the very start.

**Where the sink forms.** Tracking cosine similarity with the `<bos>` token sublayer by sublayer, the LLM sinks snap into the `<bos>` direction right after the second MLP block. The authors call MLP-2 the "sink aligner." Skipping MLP-2 for sink tokens stops that alignment.

**The test that matters.** Prune the sinks at the input and measure. On LLaVA-1.5-7B (average of General, OCR, CV-centric and Hallucination groups):

| Setting | General | OCR | CV-centric | Hallu. | Avg. |
| --- | --- | --- | --- | --- | --- |
| Full model | 58.7 | 36.9 | 54.0 | 61.1 | 52.7 |
| Remove LLM sinks | 58.6 | 37.0 | 54.4 | 61.3 | 52.8 |
| Remove ViT sinks | 58.7 | 37.2 | 53.8 | 61.1 | 52.7 |
| Remove both | 58.6 | 37.2 | 54.1 | 61.3 | 52.8 |

Nothing moves. The appendix explains why: the attention that used to land on visual sinks is reallocated to textual sinks in the system prompt. The model needs *somewhere* to dump attention, and it does not care whether the dump is an image patch or a prompt token.

## Technique 4: dead tokens, the 30% nobody reads

**The problem.** The largest cluster in every image is also the strangest. It is highly repetitive across images, sits far from every text embedding, and under EmbedLens returns punctuation and partial syllables.

**The mechanism of "dead."** The authors check two things across LLM layers. First, representation drift: dead tokens keep almost the same representation through the whole network while other tokens change substantially. Second, attention flow: the share of text-to-visual attention landing on dead tokens stays below 8% at every layer, and the per-token average is far lower than for the rest.

![Figure 5 from Fan et al. (2026): layer-wise cosine similarity within each cluster (left) and text-to-visual attention share per cluster (right)](/imgs/blogs/what-do-visual-tokens-really-encode-uncovering-sparsity-and-redundancy-in-multimodal-fig3.webp)

**Where they come from.** Inside CLIP's ViT, dead-token norms stay low through shallow and middle layers, then rise sharply around layer 21 and lock onto a fixed direction. Their semantic entropy at the LLM's last layer is high: 6.45 for dead tokens versus 3.36 for the remaining tokens (sinks sit at 6.83). High entropy here means the output distribution has no preferred concept.

**The pruning test.** Removing the dead cluster at the input, against removing the same number of tokens at random from the rest:

| Setting | General | OCR | CV-centric | Hallu. | Avg. |
| --- | --- | --- | --- | --- | --- |
| Full model | 58.7 | 36.9 | 54.0 | 61.1 | 52.7 |
| Remove dead tokens | 58.7 | 37.1 | **57.7** | 61.2 | **53.7** |
| Remove same count of others | 57.2 | 34.3 | 48.5 | 60.2 | 50.1 |

**Worked example.** At about 30% of 576, the dead cluster is roughly 170 tokens per image (my arithmetic from the paper's percentage). Deleting them buys 3.7 points on the CV-centric (RefCOCO grounding) average. Deleting 170 tokens chosen from the alive remainder costs 5.5 points there. Same token budget, opposite sign: the group label matters far more than the count.

**When it fails.** Dead tokens are a property of this encoder. The appendix states that Qwen2.5-VL, InternVL3 and Qwen3-VL do not show the same repetitive clustering pattern, and suggests newer encoders produce less redundant tokens. So "drop 30% for free" is a LLaVA-1.5 and CLIP finding until someone shows otherwise.

## Technique 5: alive tokens, sparse but dense

Alive tokens are what is left after removing sinks and dead tokens: about 60% of the input, clustered near the text centroid. Two findings about them pull in different directions, and both are true.

**Sparse in coverage.** Using COCO boxes, call any token inside an object's bounding box an object token. At the LLM input, only about 2% of object tokens retrieve their own object label through EmbedLens. That rises through a "semantic-alignment stage" and plateaus around 10% after layer 7. Over all visual tokens, the share with any grounded label never exceeds 7%. That is a neat explanation for why random token pruning and image downsampling work so well on object-level VQA: only a small fraction of tokens carry the object's name.

**Dense in content.** To test how much a single alive token can hold, the authors build a 70-image diagnostic benchmark where one object or character is scaled to fit inside exactly one patch. Each image gets three questions: what is it, what color, how many. Three groups: object (30 images), OCR on white padding (20) and OCR on a colored background (20, by subtraction).

| Group | Recognition correct | Color correct | Counting correct |
| --- | --- | --- | --- |
| Object (30) | 6 | 19 | 7 |
| OCR (20) | 8 | 14 | 6 |
| OCR with background (20) | 7 | 3 | 6 |

Three lessons, all from the paper:

- **Compositionality.** Of the 6 correctly recognized objects, 5 also got color and count right. A single patch carries several attributes, and the LLM can use them together.
- **Tokenization limits OCR.** Rendered strings that the tokenizer splits into odd subwords ("tT," "gb") fail, while contiguous strings like "Fu," "Ty" or even "Jack" are read correctly.
- **Contextual color bias.** On colored backgrounds, color accuracy collapses from 14 to 3, and wrong answers follow the background. The model ties color to the dominant regional statistics, not to the object.

## Technique 6: high norms make shallow layers a no-op

**The problem.** If alive tokens arrive already readable, what are the LLM's early layers doing to them?

**The mechanism.** A transformer layer updates a token through the residual stream: $\mathbf{h}^{(\ell+1)} = \mathbf{h}^{(\ell)} + \Delta^{(\ell)}$, where $\Delta^{(\ell)}$ is the sum of the attention and FFN outputs. How much the token's *direction* changes depends on the update size relative to the token's size. As a rough gauge (my framing, not the paper's equation):

$$
\rho^{(\ell)} = \frac{\lVert \Delta^{(\ell)} \rVert_2}{\lVert \mathbf{h}^{(\ell)} \rVert_2}
$$

The paper measures that visual tokens enter the LLM with much larger L2 norms than text tokens. Text norms grow with depth, so text tokens are being actively rewritten early on. Visual tokens are not: with a big $\lVert \mathbf{h} \rVert$, a normal-sized $\Delta$ gives a small $\rho$, and layer-to-layer cosine similarity stays near 1.

**Tiny example.** Give a text token norm 10 and a visual token norm 100 (illustrative values, not measured). The same update of norm 5 changes the text token by $\rho = 0.5$ and the visual token by $\rho = 0.05$. The text token's direction can swing a lot; the visual token barely turns.

![Schematic of L2 norm against LLM depth: visual tokens enter with a large norm, text tokens start small and catch up in the middle layers, with the early-layer band where updates are small relative to the visual norm](/imgs/blogs/what-do-visual-tokens-really-encode-uncovering-sparsity-and-redundancy-in-multimodal-3.webp)

**The causal test.** Scale every projected visual embedding by 0.01 before it enters the LLM. Now the visual tokens are small, the early layers start rewriting them like text, and visual understanding gets *worse*. The authors read this as the projector deliberately amplifying norms so that visual tokens skip shallow processing and land already aligned with middle-layer representations.

![Figure 8 from Fan et al. (2026): L2 norm and layer-wise similarity of visual and textual tokens, before (top) and after (bottom) scaling the visual embeddings down by 0.01](/imgs/blogs/what-do-visual-tokens-really-encode-uncovering-sparsity-and-redundancy-in-multimodal-fig4.webp)

**When it fails.** The norm argument explains why early layers do little. It does not prove that the middle layers are where the useful visual work happens; that needs the injection experiment below.

## Technique 7: decoupling and late injection

Two interventions, both marked as training-based in the paper (the main text does not spell out the training recipe).

**Layer decoupling.** Zero the visual-only outputs of a sublayer so it cannot change the visual residual stream, while text tokens still attend to visual tokens normally. `vMHA` removes visual-to-visual self-attention; `vFFN` removes the FFN update on visual tokens. The figure below shows which blocks of the attention pattern are switched off.

![Before and after attention masks: the visual-to-visual block and the visual FFN are zeroed while the question-to-visual block stays active](/imgs/blogs/what-do-visual-tokens-really-encode-uncovering-sparsity-and-redundancy-in-multimodal-4.webp)

**Shallow-layer skipping.** Feed visual tokens in at layer $L_{in}$ instead of layer 0, so they skip every layer before it.

![Table 3 from Fan et al. (2026): effect of layer decoupling and shallow-layer skipping on LLaVA-1.5-7B](/imgs/blogs/what-do-visual-tokens-really-encode-uncovering-sparsity-and-redundancy-in-multimodal-fig5.webp)

The numbers, from that table:

| Setting | General | OCR | CV-centric | Hallu. | Avg. |
| --- | --- | --- | --- | --- | --- |
| Full model | 58.7 | 36.9 | 54.0 | 61.1 | 52.7 |
| minus vMHA | 58.7 | 35.4 | 37.2 | 62.0 | 48.3 |
| minus vFFN | 57.8 | 35.4 | 50.6 | 61.7 | 51.4 |
| L_in = 4 | 58.7 | 36.6 | 56.4 | 61.2 | 53.2 |
| L_in = 5 | 58.4 | 36.7 | 56.8 | 61.1 | **53.3** |
| L_in = 6 | 58.0 | 36.6 | 56.0 | 61.1 | 52.9 |
| L_in = 8 | 57.9 | 35.9 | 52.7 | 61.1 | 51.9 |
| L_in = 10 | 57.0 | 36.0 | 52.0 | 60.1 | 51.3 |

How to read it:

- **Hallucination improves without visual processing.** Removing either sublayer raises the hallucination score (61.1 to 62.0 and 61.7). Visual self-updates seem to amplify noise.
- **General VQA barely cares.** vMHA removal leaves it at 58.7; vFFN costs 0.9 points, which the authors read as the FFN injecting some task knowledge.
- **Grounding needs visual attention.** CV-centric collapses from 54.0 to 37.2 without vMHA. Spatial reasoning still relies on token-to-token attention inside the visual stream.
- **Skipping 4 to 6 layers is free or better.** CV-centric even rises to 56.8 at $L_{in} = 5$. Past layer 6, the cost starts to show.

One honest caveat on my side: the paper says skipping the first 6 layers "preserves or slightly improves" general performance, but General goes from 58.7 to 58.0 at $L_{in} = 6$. The average is up; the General column is slightly down.

## What is load-bearing and might not transfer

- **One model for every intervention.** Tables 1, 2 and 3 are LLaVA-1.5-7B. EmbedLens accuracy is shown across four model families, but the pruning and skipping results are not.
- **CLIP specifically.** The dead cluster forms around layer 21 of CLIP's ViT. Newer encoders reportedly lack the repetitive pattern, so the 30% dead share should not be assumed elsewhere.
- **Grouped benchmark averages.** Each column averages several benchmarks with normalization (MME divided by 20, for example). A 1-point average move can hide a larger move on one benchmark; the appendix tables have the per-benchmark scores.
- **Small diagnostic set.** The multi-trajectory benchmark is 70 images, and the object group has 6 correct recognitions. The compositionality claim ("5 of 6") rests on very few samples.

## Critique

**What is strong.** The paper turns a vague belief ("many visual tokens are redundant") into a typed taxonomy with a mechanism for each type: where sinks get aligned (MLP-2), where dead tokens are born (CLIP layer 21), and why early layers do nothing (norm). EmbedLens is simple, model-agnostic and cheap, and the label-matching comparison against LogitLens is a fair test of it. The dead-versus-random pruning control is exactly the right control.

**What is weak.** The dead and sink definitions come from clustering thresholds on one encoder, and the evidence that they generalize is mostly a negative result in the appendix. The training-based rows in Table 3 mix two effects: the architecture change and whatever the retraining itself does. A "retrain with no change" control row would separate them. The alive-token story also leans on EmbedLens, which by construction cannot see non-verbal information, so "only 7% of tokens carry grounded semantics" is a lower bound on content, not a measurement of it.

**Missing ablation.** Late injection and dead-token pruning are each tested alone. Their combination, the configuration a practitioner would actually ship, is not. Neither is any measurement of wall-clock or FLOP savings.

**What would change my mind.** If dropping the same "dead" cluster on a SigLIP-based LLaVA variant hurt accuracy, or if a no-change retraining control matched the $L_{in} = 5$ gains, I would treat both headline claims as artifacts of LLaVA-1.5's CLIP pipeline rather than facts about MLLMs.

## Where this goes next

The orchestrating question is how to design MLLM connectors that never emit dead or sink tokens and skip redundant visual FFN work. Here are four concrete follow-ups. These are my proposals, not the paper's.

**1. A connector that learns to emit only alive tokens.**
- *Question:* can a projector be trained to output a variable number of tokens with no dead or sink members, instead of pruning after the fact?
- *Why now:* the paper gives a cheap, label-free detector for each type (cross-image centroid similarity for sinks and dead tokens, EmbedLens distance to the text centroid for alive).
- *First experiment:* add a gating head to LLaVA-1.5's MLP projector, supervise its keep/drop decision with the paper's cluster labels, and compare against post-hoc dead-token pruning at matched token counts.

**2. A text-side sink instead of visual sinks.**
- *Question:* if attention just reroutes to system-prompt sinks after visual sinks are pruned, does a single learned register token in the connector do the job better?
- *Why now:* the paper localizes sink formation to MLP-2 and shows the reallocation directly.
- *First experiment:* drop all visual sinks, add one learnable token after the image, and measure attention mass and hallucination scores across layers.

**3. Visual tokens that bypass the FFN, not just early layers.**
- *Question:* if vFFN mostly adds task knowledge and hurts hallucination, can visual tokens skip FFNs everywhere except a few middle layers?
- *Why now:* Table 3 isolates vMHA and vFFN, and the norm analysis says which depths matter.
- *First experiment:* sweep "vFFN on only in layers k to k+3" on LLaVA-1.5-7B, report the same four benchmark groups plus measured FLOPs, and pair it with late injection at $L_{in} = 5$.

**4. Does the taxonomy survive a modern encoder?**
- *Question:* do SigLIP or native-resolution encoders produce dead tokens of a different shape (spread out instead of one cluster), or truly none?
- *Why now:* EmbedLens and the centroid stability score are model-agnostic and need no training.
- *First experiment:* run the clustering and EmbedLens pipeline on Qwen2.5-VL and InternVL3, then repeat the dead-versus-random pruning control using a distance-to-text-centroid threshold instead of cluster membership.

## References

- Yingqi Fan, Junlong Tong, Anhao Zhao, Xiaoyu Shen. *What Do Visual Tokens Really Encode? Uncovering Sparsity and Redundancy in Multimodal Large Language Models.* CVPR 2026 (highlight). [arXiv:2603.00510](https://arxiv.org/abs/2603.00510) · [CVPR page](https://cvpr.thecvf.com/virtual/2026/poster/39001)
- Code: [github.com/EIT-NLP/EmbedLens](https://github.com/EIT-NLP/EmbedLens)
- Related on this site: [StreamingLLM and attention sinks](/blog/paper-reading/large-language-model/streaming-llm-attention-sinks), [Your LVLM only needs a few attention heads for visual grounding](/blog/paper-reading/computer-vision/your-large-vision-language-model-only-needs-a-few-attention-heads-for-visual-grounding)
