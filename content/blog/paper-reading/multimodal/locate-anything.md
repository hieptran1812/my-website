---
title: "LocateAnything: Fast, High-Quality Visual Grounding with Parallel Box Decoding"
date: "2026-07-16"
description: "A detailed walk through NVIDIA's LocateAnything: why serializing a bounding box into coordinate tokens is both slow and inaccurate, and how Parallel Box Decoding predicts a whole box in a single forward pass to fix both at once."
tags: ["paper-reading", "parallel-box-decoding", "visual-grounding", "object-detection", "vision-language-model", "multi-token-prediction", "vlm", "gui-grounding", "nvidia"]
category: "paper-reading"
subcategory: "Multimodal"
author: "Hiep Tran"
featured: true
readTime: 30
paper:
  title: "LocateAnything: Fast and High-Quality Vision-Language Grounding with Parallel Box Decoding"
  authors: "Shihao Wang, Shilong Liu, et al. (NVIDIA)"
  venue: "arXiv / NVIDIA LPR, 2026"
  url: "https://research.nvidia.com/labs/lpr/locate-anything/LocateAnything.pdf"
---

> [!tldr]
> - **What it is.** A unified vision-language model for detection and grounding that stops treating a bounding box as a string of coordinate tokens. Instead, **Parallel Box Decoding (PBD)** emits a whole box `(x1, y1, x2, y2)` as one atomic block in a *single* forward pass.
> - **The mechanism.** Each output "block" is a fixed-length unit of ${6}$ tokens. During training the model is shown the same ground truth in two aligned formats — a serial next-token stream and a masked block-parallel stream — and trained on both with a joint loss.
> - **Why it matters.** Serializing a box into ${8}$–${21}$ tokens is a *double* mistake: it is slow (strictly sequential decode) *and* inaccurate (the four coordinates are learned as if independent). PBD fixes both. LocateAnything-3B runs at **12.7 boxes/second** — ${10}\times$ faster than textual-coordinate Qwen3-VL and ${2.5}\times$ faster than Rex-Omni — while *improving* localization.
> - **The surprising result.** Parallel decoding usually trades accuracy for speed. Here it *raises* high-IoU accuracy: LVIS F1 at IoU ${0.95}$ jumps from Rex-Omni's ${20.7}$ to **31.1**. Structure, not just serialization, was the bottleneck.
> - **Where it's soft.** It is supervised-fine-tuning only (no RL yet), the hybrid fallback thresholds are hand-tuned, and the headline throughput is measured on COCO with batch size ${1}$.

Object detection used to be a solved-shape problem: a specialized head like DETR or Faster R-CNN regresses four numbers per box and you are done. The moment we asked a *single* vision-language model to also read UI screenshots, ground free-form phrases, detect text, and point at object parts — all from natural language — we gave up those regression heads and started spelling coordinates out as text tokens. That decision is quietly expensive, and LocateAnything is a paper about paying it back.

![Figure 1 from Wang et al. (2026): LocateAnything handles multi-object referring, pointing, GUI grounding, detection, OCR, and layout grounding under one model; the bottom panel shows the decoding-paradigm evolution — from 21 textual-digit steps, to 10 quantized-token steps, to just 2 parallel-box steps.](/imgs/blogs/locate-anything-fig1.webp)

The diagram above is the whole thesis at a glance. The top half is the *breadth*: one model, many localization tasks. The bottom half is the *idea*: to emit the box for a ship, textual-digit decoding spends ${21}$ sequential steps (`1`, `3`, `0`, `,`, `6`, `4`, `7`, …), quantized-token decoding spends ${10}$, and Parallel Box Decoding spends **2**. The rest of this post unpacks how you get from ${21}$ to ${2}$ without the accuracy falling off a cliff — and why, counterintuitively, it goes *up*.

## The problem: a box is not a sentence

Modern VLMs — the [Qwen-VL series](/blog/paper-reading/multimodal/qwen2-vl-enhancing-vision-language-models-perception-of-the-world-at-any-resolution), InternVL, Shikra, [SEED1.5-VL](/blog/paper-reading/multimodal/seed1-5-vl-native-resolution-vision-language) — formulate detection as **next-token prediction (NTP)**. The model answers "where is the ship?" by generating a token sequence that encodes the coordinates. There are two dominant encodings:

- **Textual digits.** The number ${1024}$ becomes the four characters `1`, `0`, `2`, `4`. A single box (four numbers plus box delimiters) can run to ${20}$-plus tokens.
- **Quantized tokens.** Coordinates are normalized to a fixed grid (LocateAnything uses `[0, 1000]`) and each value becomes one special token: `<130>`, `<647>`, `<911>`, `<832>`. Now a box is roughly ${x_1 \to y_1 \to x_2 \to y_2}$ plus `<box>`/`</box>` — around ${6}$–${10}$ tokens.

Both share a fatal shape: they **serialize a 2D geometric object into a 1D stream** and then decode it strictly left to right. This is a double failure, and it is worth being precise about *why*, because the whole paper is an answer to it.

**Failure one — latency.** Autoregressive decode is sequential by construction: token ${i+1}$ cannot start until token ${i}$ is committed, because it conditions on it. A scene with ${300}$ objects and ${8}$ tokens per box is ${2400}$ sequential forward passes. On a single GPU with batch size ${1}$ that is the entire wall-clock cost, and it grows linearly with the number of boxes. Detection throughput collapses in exactly the crowded scenes — traffic, retail shelves, drone footage — where you most want it.

**Failure two — a structure mismatch.** The four coordinates of a box are not independent draws; they are tightly coupled (${x_2 > x_1}$, the aspect ratio is constrained, the center is one point). But token-by-token cross-entropy trains each coordinate token *conditioned on the previous ones as if they were arbitrary language*. The supervision never tells the model "these four numbers are one geometric object." It learns the correlations only implicitly, through data, and imperfectly.

You might think **Multi-Token Prediction (MTP)** — predict several future tokens in parallel — rescues the latency half for free. It does reduce steps, but the naive version makes the *accuracy* half worse. Generic MTP (à la Medusa, block diffusion, SDLM) chops the token stream into arbitrary fixed-size chunks and asks the model to fill them in parallel. Those chunk boundaries do not respect box boundaries.

![Figure 2 from Wang et al. (2026): next-token prediction generates coordinates one at a time; generic multi-token prediction cuts the stream into arbitrary chunks that straddle box and category boundaries ("unstructured"); Parallel Box Decoding cuts on box boundaries ("box-aligned").](/imgs/blogs/locate-anything-fig2.webp)

Look at the middle row. A generic-MTP block might contain `</box> <box> <122>` — the *end* of one box, the *start* of the next, and the first coordinate of a third object. The model is now being trained to predict, in parallel, a combination of tokens that spans two boxes and possibly two categories. As the authors put it, this forces the model "to fit many unreliable patterns, inducing spurious correlations." You have kept the accuracy problem and added a new one.

The paper's move is the third row: **cut the stream exactly on box boundaries.** A parallel block is a whole box or nothing. That single alignment decision is what lets parallelism *help* accuracy instead of hurting it — because now the thing you predict in parallel is precisely the thing that is internally coupled.

### Contributions, in one map

Stripped to essentials, the paper claims three things:

1. **Parallel Box Decoding.** Treat each bounding box (or point) as an atomic, fixed-length block and predict the whole block in one parallel step — box-aligned MTP, trained jointly with NTP so the model keeps its causal reasoning.
2. **A hybrid decoding policy.** Decode in parallel by default; detect when a parallel block is unreliable (malformed or spatially ambiguous) and fall back to NTP re-decoding *for that block only*, then resume. This buys most of the speed while capping worst-case errors.
3. **LocateAnything-Data.** A data engine and a ${138}$M-query, ${785}$M-box corpus spanning detection, GUI, referring, OCR, layout, and pointing — the fuel that makes a single model precise across all of them.

The rest of the post takes each load-bearing technique from intuition to math.

## Method

LocateAnything is built on a native-resolution VLM: a **Moon-ViT** vision encoder (from the [Kimi-VL](/blog/paper-reading/multimodal/kimi-vl) line), a **Qwen2.5** language decoder, and a two-layer MLP projector between them. "Native resolution" matters here for a specific reason: high-precision localization needs fine spatial detail, and downsampling every image to a fixed ${448 \times 448}$ grid throws that away before the model ever sees it.

![Figure 3 from Wang et al. (2026): the architecture. An image plus a text query enters a Moon-ViT encoder, is projected by a two-layer MLP into a Qwen2.5 decoder, and the decoder emits a sequence of fixed-length blocks — Semantic, Box, Negative, and End.](/imgs/blogs/locate-anything-fig3.webp)

Given an image ${\mathcal{I}}$, the encoder produces visual tokens ${Z = \text{Encoder}(\mathcal{I})}$ at native resolution. Those tokens, plus the text query ${\mathcal{E}}$, condition the decoder — which, and this is the departure from a standard VLM, emits not a free-form token stream but a sequence of **box-aligned blocks**.

### Technique 1 — The block-based output formulation

**The problem it solves.** Parallel decoding needs *uniform tensor shapes*: to predict a block in one forward pass, every block must have the same length so it packs into a rectangular tensor. But real outputs are ragged — a category name is a few tokens, a box is four numbers, a "no object" answer is one word. You need a container that makes all of them the same shape without losing information.

**Intuition.** Think of a label printer that only prints fixed-width labels. Whatever you feed it — a short word, a long phrase, four numbers — comes out on a ${6}$-slot strip, with unused slots left blank. Downstream machinery can now handle every strip identically because they are all the same size. LocateAnything's blocks are those fixed-width strips; the "blanks" are a special `<null>` padding token.

**The mechanism.** Continuous coordinates are normalized to `[0, 1000]` and discretized into coordinate tokens (the Pix2Seq trick). The output is then reorganized into a sequence of blocks ${\mathcal{B} = (b_1, b_2, \dots, b_N)}$. Conditioned on the visual features ${Z}$ and the query ${\mathcal{E}}$, the joint probability factorizes across blocks:

$$P(\mathcal{B} \mid Z, \mathcal{E}) = \prod_{i=1}^{N} P(b_i \mid b_{<i}, Z, \mathcal{E})$$

Read this carefully against the NTP baseline. Standard NTP factorizes over *tokens*: ${\prod_t P(t \mid t_{<t}, \dots)}$. Here the product runs over *blocks*, and each factor ${P(b_i \mid b_{<i}, \dots)}$ is the probability of an *entire* block conditioned on all previous blocks. The unit of autoregression has been promoted from token to box.

Each block ${b_i}$ is an atomic unit of constant length ${L = 6}$: enough to hold a bounding box (four coordinates) plus two structural tokens (`<box>` and `</box>`). Any unused slot is filled with `<null>`. The paper defines four functional block types:

| Block type | Contents | Purpose |
|---|---|---|
| **Semantic** | `<ref> ship </ref>` + `<null>` padding | Encodes the linguistic identity of what follows; long phrases span consecutive blocks |
| **Box** | `<box> <x1> <y1> <x2> <y2> </box>` | Four quantized coordinates — the actual localization |
| **Negative** | `<box> None <box>` + padding | Explicitly says "the queried object is absent" |
| **End** | `<im_end>` + padding | Terminates generation |

**Worked micro-example.** Suppose the query is "detect person, cat" and the image has one person box and no cat. The output block sequence is:

```python
b1 (Semantic): <ref> Person <ref> <null> <null> <null>
b2 (Box):      <box> <100> <675> <218> <800> </box>
b3 (Semantic): <ref> Cat <ref> <null> <null> <null>
b4 (Negative): <box> None  <box> <null> <null> <null>
b5 (End):      <im_end> <null> <null> <null> <null> <null>
```

Five blocks, each exactly ${6}$ slots wide. `b2` carries a real box; `b4` says "no cat here" — which, and we will come back to this, is what teaches the model to *abstain* instead of hallucinating a box.

**Why it works / when it fails.** The fixed width is the enabling trick: uniform shape is what makes a block predictable in a single parallel tensor operation. The cost is that ${L = 6}$ caps how much a single Semantic block can hold; a long referring phrase ("the third person from the left wearing a red backpack") must be split across several consecutive Semantic blocks, spending steps you would rather spend on boxes. For detection and short phrases this is nearly free; for long compositional queries it erodes the parallelism advantage.

### Technique 2 — Parallel Box Decoding

**The problem it solves.** Given the block formulation, how do you actually predict a whole ${6}$-token block *at once*, when autoregression is defined to produce one token at a time?

**Intuition.** A relay race versus a starting gun. NTP is a relay: runner ${i+1}$ cannot start until runner ${i}$ hands off the baton, so four coordinates take four legs. PBD is a starting gun: all runners in a block leave together on a single shot. The catch — the reason you cannot fire the gun for the *whole race* — is that later blocks still depend on earlier ones (you should not detect the same object twice, or leave gaps). So PBD fires the gun *within* a block and runs a relay *between* blocks. This is exactly the "semi-autoregressive" pattern from block-diffusion language models, specialized to geometry.

**The mechanism.** Within a block, the model is given the block's *first* token as an anchor (the prediction context) and the rest as `[mask]` placeholders; it predicts all masked positions simultaneously in one forward pass. Between blocks it stays strictly causal — block ${i}$ sees all committed blocks ${b_{<i}}$ but no future block. So the block-level factorization from Technique 1 stays autoregressive, while *inside* each factor the ${L}$ tokens resolve in parallel. Writing the anchor of block ${i}$ as ${a_i}$ and its ${L}$ token slots as ${t_{i,1}, \dots, t_{i,L}}$, the intra-block prediction is (my notation, making the paper's "predict all masked tokens in one step" explicit):

$$P(b_i \mid b_{<i}, Z, \mathcal{E}) = \prod_{j=1}^{L} P\!\left(t_{i,j} \mid a_i,\, b_{<i},\, Z,\, \mathcal{E}\right)$$

The thing to notice is what is *absent* from the conditioning of ${t_{i,j}}$: the other tokens ${t_{i,k}}$ of the same block. Each token in the block is predicted from the anchor and the past blocks, **not** from its intra-block siblings. That is what makes them computable in parallel — none waits on another. The bet PBD makes is that a single bidirectional forward pass over the block (see the attention mask below) lets the ${L}$ predictions *coordinate* through the shared hidden states even though they do not condition on each other's sampled tokens. For a tightly coupled unit like a box, that bet pays off; for an arbitrary chunk of language, it would not.

**Worked micro-example.** To emit `<box> <130> <647> <911> <887> </box>`:

- *NTP (Slow Mode):* ${6}$ forward passes. Pass 1 emits `<box>`, pass 2 conditions on it to emit `<130>`, …, pass 6 emits `</box>`. Six sequential steps.
- *PBD (Fast Mode):* ${1}$ forward pass. The anchor `<box>` is given; `<130>`, `<647>`, `<911>`, `<887>`, `</box>` are all masked and predicted together. One step.

That is the ${6 \to 1}$ collapse per box. Across a ${300}$-box scene it is the difference between ${\sim 1800}$ and ${\sim 300}$ decode steps.

**Why it works / when it fails.** It works because box coordinates are *jointly determined by the visual evidence*: once the model has localized the ship in its hidden states, the four corners are a deterministic read-out, not a sequential negotiation. Parallelizing a read-out is safe. It fails precisely when the read-out is *ambiguous* — two objects packed so tightly that the "correct" corner sits between them, or a category boundary the model has not resolved — and then predicting all four at once with no sibling conditioning produces a blurred, low-IoU box. Technique 4 is the safety net for exactly this.

### Technique 3 — Dual-formulation training

**The problem it solves.** If you *only* train the model to predict masked blocks in parallel, you risk destroying the causal reasoning that made the base VLM good in the first place — its ability to generate coherent, ordered output. But if you only train NTP, you never learn the parallel skill. You need both, from the *same* ground truth, without one corrupting the other.

**Intuition.** Teaching someone to both read aloud (left to right, one word at a time) and to fill in a cloze passage (several blanks at once) using the *same* text. If you drilled only cloze, their reading would get jumpy; only reading, and they cannot do the blanks. So you show the identical passage in both formats and grade both — with a wall between the two exercises so the reading answers do not leak into the cloze blanks.

![One ground-truth box is expanded into two aligned targets — a serial NTP stream decoded left-to-right, and a parallel block-MTP stream that keeps one anchor per block and unmasks the rest in a single pass — and both are trained jointly under L = Lntp + Lmtp.](/imgs/blogs/locate-anything-1.webp)

**The mechanism.** A single concatenated input sequence is built:

$$x_{\text{all}} = x_{\text{vis}} \oplus x_{q} \oplus x_{\text{ntp}} \oplus x_{\text{blk}}$$

where ${\oplus}$ is concatenation. Here ${x_{\text{vis}}}$ (visual tokens) and ${x_q}$ (the text query) are the **shared context**; ${x_{\text{ntp}}}$ is the standard next-token sequence; and ${x_{\text{blk}}}$ is the block-wise MTP sequence. Critically, ${x_{\text{ntp}}}$ and ${x_{\text{blk}}}$ are the *same ground truth in two formats* — one token-level, one block-level. The block stream is built by traversing the NTP stream left to right, splitting and padding into blocks by the rules above, then, within each block, **keeping the first token as context and replacing the rest with `[mask]`**. (Set the block size to ${1}$ and this MTP formulation degenerates exactly into NTP — a nice consistency check.)

The redrawn figure above shows the split concretely: the ground-truth box on top forks into a blue serial strip (six cards, decoded in six causal passes) and a mixed strip (one blue anchor, five amber `[mask]` slots, resolved in one pass). Two cross-entropy losses come out the bottom and sum.

**The attention mask.** The heart of the dual formulation is a single attention mask that lets both streams read the shared context while staying isolated from each other. It has three behaviors:

![Figure 4 from Wang et al. (2026): the joint NTP–MTP attention mask. The shared context and NTP stream use causal attention; MTP blocks are block-causal across blocks and bidirectional within a block; the two streams are isolated so the NTP labels never leak into the masked block predictions.](/imgs/blogs/locate-anything-fig6.webp)

- **Causal attention for NTP.** The shared context and the NTP stream attend only to preceding tokens — standard causal masking. Crucially, they are *forbidden* from attending to ${x_{\text{blk}}}$, so the ground-truth tokens sitting in the NTP stream cannot leak into the masked-block predictions. This is what keeps the MTP task honest, and it makes the NTP stream's behavior identical to inference-time KV-cache decoding.
- **Causal flow across blocks.** Within ${x_{\text{blk}}}$, block ${i}$ attends to the shared context and all *committed* blocks ${b_{<i}}$, but not to future blocks. This is where the model learns cross-box dependencies — "I already emitted a box here, do not duplicate it" — which is what suppresses the duplicate and missing boxes that plague naive parallel decoders.
- **Bidirectional intra-block attention.** The ${L}$ tokens *inside* a block attend to each other fully. This is the channel through which the four coordinates coordinate: even though (from Technique 2) no token conditions on another's sampled value, they share hidden states bidirectionally within the forward pass, so the corners are computed *jointly*.

**The objective.** Both streams are trained with cross-entropy, and the losses simply add:

$$\mathcal{L} = \mathcal{L}_{\text{ntp}} + \mathcal{L}_{\text{mtp}}$$

**Why it works / when it fails.** The joint loss is not just a hedge — it is *load-bearing for accuracy*. The ablation (Table 6c) shows that training with the isolated losses caps the model's ceiling, and adding the joint formulation pushes the Slow-Mode F1 from ${50.1}$ to **52.1** on COCO. That is a striking result: the NTP stream, which you decode at *inference* time in Slow Mode, gets *better* because it was co-trained with the block stream. The shared representation learned to be geometrically coherent, and both readouts benefit. The failure mode is cost: you are training on two copies of every target, which inflates sequence length and memory (the paper needs custom infrastructure — stream packing and MagiAttention — to make the heterogeneous mask tractable, covered later).

### Technique 4 — On-demand inference and the hybrid fallback

**The problem it solves.** Fast Mode (pure MTP) is fast but, as Technique 2 warned, it degrades in exactly two situations. You want its speed almost always, and its failure only where it actually fails — not a blanket accuracy tax.

**Intuition.** A fast checkout lane that flips to a careful manual scan only when an item won't scan. Most items fly through; the occasional problem item gets the slow, reliable treatment; the line does not slow down for everyone.

The two failure modes are worth naming precisely, because the trigger is built from them:

- **Format irregularity.** In busy multi-category scenes, the model hesitates at category boundaries — should the next block continue the current class or switch? — and emits a malformed block that mixes structural and coordinate tokens, e.g. `<box><211></ref><911><887></box>`. The `</ref>` has no business inside a box.
- **Spatial ambiguity.** When objects sit in a dense regular grid (rows of parked cars, abacus beads, stacked logs), the parallel prediction blurs the boundary and returns a coordinate *between* two objects — a low-IoU box.

![Figure 5 from Wang et al. (2026): corrected NTP re-decoding. When a parallel block is malformed (format irregularity) or blurred (spatial ambiguity), the block is discarded, generation reverts to the last verified prefix, and NTP re-decodes just that block before parallel decoding resumes.](/imgs/blogs/locate-anything-fig4.webp)

**The mechanism.** During MTP inference the model continuously validates each block's syntactic integrity and monitors spatial confidence. The ambiguity trigger fires when **both** conditions hold at once:

$$\underbrace{p_{\text{top-1}}(\text{coord}) < 0.7}_{\text{model is unsure}} \quad \text{and} \quad \underbrace{\big(\max_{k \le 5} c_k - \min_{k \le 5} c_k\big) > 80}_{\text{top-5 candidates disagree}}$$

Here ${p_{\text{top-1}}}$ is the probability mass on the most likely coordinate token, and ${c_1, \dots, c_5}$ are the top-5 candidate coordinate values in the normalized `[0, 1000]` space. The first clause says the model is not confident; the second says its confusion is *spatial* — its top guesses are spread more than ${80}$ units apart (${8\%}$ of the frame), not just split between two adjacent bins. Requiring *both* is what keeps the fallback rare: a merely close call (low ${p}$ but tight cluster) does not trigger, only a genuinely ambiguous one.

**Worked micro-example.** Consider the left edge of one car in a tightly packed row. The model's top-5 candidate values for ${x_1}$ come out as `{412, 415, 488, 491, 410}` with a top-1 probability of ${0.55}$. Both clauses fire: ${0.55 < 0.7}$ (unsure) *and* ${491 - 410 = 81 > 80}$ (the guesses split into two clusters, ~${410}$ and ~${490}$ — the boundary between this car and its neighbor). The parallel block is thrown out and NTP re-decodes it, letting the sequential conditioning pick one cluster cleanly. Contrast a confident box where the top-5 are `{413, 414, 412, 415, 413}` at ${p = 0.62}$: the probability is low, but the spread is ${3 \ll 80}$, so no fallback — the model is just choosing between adjacent bins of the *same* edge, which parallel decoding handles fine.

On a format violation or a fired ambiguity trigger, the compromised block is **discarded**, generation reverts to the last verified prefix, and NTP re-decodes the tokens of *that block only*, autoregressively. Once the block is complete, the model switches back to MTP. The KV cache is truncated after each MTP step to keep only committed tokens — evicting the mask tokens and the duplicated anchor — so the cache state matches the causal prefix the model saw in training.

![At inference, each parallel block is committed only if it passes a format and spatial-confidence gate; a malformed or spatially ambiguous block is discarded and NTP re-decodes just that block before MTP resumes, so the fallback cost is localized.](/imgs/blogs/locate-anything-2.webp)

The redrawn flowchart above is the control loop: MTP predicts a block, the gate checks it, a passing block commits and advances, a failing block routes through a localized NTP re-decode and then rejoins the fast path. That gives the three **on-demand modes**:

| Mode | Decoder | Throughput (COCO) | Use case |
|---|---|---|---|
| **Slow** | Pure NTP | ${4.3}$ BPS | High-precision labeling, offline curation — the accuracy ceiling |
| **Fast** | Pure MTP | ${15.3}$ BPS | On-device robotics, embodied agents — latency-bound |
| **Hybrid** | MTP + NTP fallback | ${12.7}$ BPS | Production default — most of the speed, guarded accuracy |

One honest wrinkle the supplementary reveals: Hybrid is not always close to Slow. On tasks where *every* token is spatially load-bearing — dense scene-text OCR — the gap widens sharply. Slow Mode scores ${64.4}$ mean F1 on SROIE versus Hybrid's ${39.3}$, and ${43.2}$ versus ${29.1}$ on HierText. There the trigger does not fire often enough, because the failures are pervasive rather than isolated to a few ambiguous blocks, so the localized fallback cannot rescue them. For OCR-heavy pipelines you would actually want Slow Mode, not the Hybrid default — a reminder that "on-demand" means *you* pick the mode per workload, not that Hybrid dominates everywhere.

**Why it works / when it fails.** The fallback is cheap because it is *localized*: only the problem block pays the sequential cost, not the whole sequence. And it works because NTP genuinely is more precise on the hard cases — the sequential conditioning that PBD drops is exactly what disambiguates dense grids. It fails if the trigger is miscalibrated: too eager and you fall back constantly (losing the speed win), too lax and blurred boxes slip through. The thresholds (${0.7}$, ${80}$) are hand-set constants, and the paper's own limitations section flags reducing fallback frequency via RL as future work — an implicit admission that the current trigger is a reasonable heuristic, not an optimum.

### Technique 5 — LocateAnything-Data and the data engine

**The problem it solves.** A single model that is precise across detection, GUI, referring, OCR, layout, and pointing needs training data across all of them — at a scale and diversity the open-source grounding datasets simply do not have. Referring datasets in particular are small, and almost all detection data contains *only positive examples*, which teaches a model to always emit a box.

**Intuition.** A factory line that turns cheap raw materials (unlabeled images, plain detection boxes) into expensive finished goods (rich, multi-target grounding annotations) by chaining specialist machines, each doing the one thing it is best at.

![Figure 9 from Wang et al. (2026): the multi-target grounding data engine. Top: from labeled detection boxes, Qwen3-VL writes object-centric queries, Molmo points, and a "point-in-box" rule keeps only points inside the ground-truth boxes. Bottom: from unlabeled images, Qwen3-VL writes queries, Molmo points, and SAM 3 turns points into boxes (or Rex-Omni predicts boxes directly); Qwen3-VL post-verifies everything.](/imgs/blogs/locate-anything-fig5.webp)

**The mechanism** runs two pipelines:

- **From detection datasets** (Open Images, Objects365): for each ground-truth box, its category label prompts **Qwen3-VL** to synthesize detailed object-centric queries — attributes, spatial relations, reasoning cues. Those queries prompt **Molmo** to predict candidate points. Because the ground-truth boxes are known, only points that fall *inside* the corresponding box are kept as reliable supervision. This is a clean self-verification: the box you already trust filters the points.
- **From unlabeled images** (Unsplash, SA-1B): **Qwen3-VL** writes diverse queries directly from the image; **Molmo** points; **SAM 3** converts points into boxes; or, alternatively, **Rex-Omni** predicts boxes directly. Every generated box is finally **post-verified by Qwen3-VL** to filter inconsistent predictions.

The engine also *enriches* thin labels. GUI data like GroundCUA ships only short element descriptions, so the paper renders the target box on the screenshot, crops a local region, and asks Qwen3-VL to describe the element from three complementary angles: **appearance** (color, shape, iconography, text), **spatial** (position relative to other UI components), and **functional** (the user intent behind the element). One terse label ("crop tool") becomes several interpretable grounding queries — which is exactly the compositional variety a UI-grounding model needs to generalize beyond memorized button names.

Negatives are constructed deliberately: queries referring to objects that do *not* exist in the image, assigned the Negative block. This is what lets the model learn to abstain — the second pipeline's post-verification and the negative blocks together attack hallucination from both sides.

**The scale.** The result is ${12}$M unique images, ${138}$M natural-language queries (${139}$M with over ${22}$M negatives), and **785M bounding boxes**. The mixture is detection-heavy — that is where dense coordinate supervision lives:

| Domain | Share of queries | Role |
|---|---|---|
| General object detection | ${66.9\%}$ | Dense coordinate supervision (${83.1\%}$ of all boxes) |
| GUI element grounding | ${16.5\%}$ | Embodied agents, UI navigation |
| Natural-language referring | ${7.3\%}$ | Linking compositional intents to regions |
| Text localization (OCR) | ${3.6\%}$ | Tight grounding of text |
| Document / layout grounding | ${3.5\%}$ | Structural reasoning |
| Point-based localization | ${2.2\%}$ | Fine-grained precision |

**Why it works / when it fails.** The engine works because it composes *reliable* signals: known boxes filter noisy points, and a strong VLM verifies the rest — no single fallible model is trusted end to end. Its weakness is inherited bias: the queries, points, and boxes are all generated by existing models (Qwen3-VL, Molmo, SAM 3, Rex-Omni), so LocateAnything's ceiling on any axis is bounded by what those teachers already know, and their blind spots become its blind spots. This is the standard risk of model-generated supervision, and the paper does not measure it directly.

### The training and infrastructure that make it run

Training proceeds in four progressive stages. The first two build a base VLM's world knowledge with **no detection data at all** — Stage 1 on captioning to align the native-resolution encoder, Stage 2 on a broad multimodal mixture (math, science, charts, OCR, VQA). Only then do the LocateAnything stages begin: Stage 3 folds in the full ${138}$M-query grounding mixture with all components unfrozen (LR ${4 \times 10^{-5}}$, max sequence length ${25{,}600}$, ${256}$ GPUs); Stage 4 is a dense-detection finetune that drops general data to ${20\%}$ and up-weights many-object images like MOT20Det and SKU110K (LR decayed to ${1 \times 10^{-5}}$).

Two systems tricks make the dual-formulation tractable. **Stream packing** assembles variable-length samples into densely packed sequences (a best-fit-decreasing heuristic hitting ${>95\%}$ packing efficiency at a ${36{,}864}$-token budget) so the ragged block-expanded samples do not waste GPU memory on padding. **MagiAttention** natively supports the heterogeneous mask — causal for NTP, block-causal-plus-bidirectional for MTP, all within one packed multi-sample sequence — which conventional Flash-Attention kernels, assuming a uniform pattern, cannot express. Neither is glamorous, but the joint formulation does not train efficiently without them.

## Experiments & results

Everything below is the default **Hybrid Mode**, on a single NVIDIA H100 at batch size ${1}$, BF16, with throughput in **boxes per second (BPS)**. The headline table is LVIS (long-tailed) and COCO (common objects):

| Method | Throughput | LVIS F1@0.5 | LVIS F1@0.95 | LVIS mean | COCO F1@0.5 | COCO F1@0.95 | COCO mean |
|---|---|---|---|---|---|---|---|
| Qwen3-VL-8B (textual) | ${1.0}$ | ${61.5}$ | ${20.2}$ | ${44.8}$ | ${62.8}$ | ${14.0}$ | ${45.7}$ |
| SEED1.5-VL | – | ${65.6}$ | ${19.5}$ | ${46.7}$ | ${71.3}$ | ${14.3}$ | ${51.4}$ |
| Rex-Omni-3B (quantized) | ${5.0}$ | ${64.3}$ | ${20.7}$ | ${46.9}$ | ${72.0}$ | ${15.9}$ | ${52.9}$ |
| **LocateAnything-3B** | **12.7** | ${62.3}$ | **31.1** | **50.7** | ${70.1}$ | **19.3** | **54.7** |

Two numbers carry the paper. First, **12.7 BPS** — ${10}\times$ faster than the textual-coordinate Qwen3-VL and ${2.5}\times$ faster than the quantized Rex-Omni, at the same ${3}$B model size. Second, and more important, **LVIS F1 at IoU ${0.95}$ is 31.1 versus Rex-Omni's ${20.7}$** — a ${+10.4}$ jump at the *high-IoU* end. This is the crux: the accuracy gain is concentrated where boxes must be *tight*, which is precisely what the box-aligned, jointly-coordinated prediction should improve. LocateAnything trades a little at the loose IoU ${0.5}$ threshold (${62.3}$ vs ${65.6}$) for a large gain where precision counts.

The story holds across tasks. On dense benchmarks it reaches ${39.9}$ mean F1 on VisDrone (Rex-Omni: ${35.8}$) and ${58.7}$ on Dense200. On **GUI grounding** (ScreenSpot-Pro) it posts a SOTA ${60.3}$ average, beating both the generalist Qwen3-VL-30B-A3B (${53.7}$) and the UI-specialist GUI-Owl-32B (${58.0}$) — with a ${3}$B model. On document layout it hits ${76.8}$ (DocLayNet) and ${70.1}$ (M6Doc); on referring, ${78.7}$ on HumanRef.

The box-aligned training also carries over to **pointing** — predicting a single coordinate that must land inside the target — which the paper handles with the same block machinery (a point is just a shorter geometric unit). Here LocateAnything-3B is state of the art across the board: ${83.9}$ F1@Point on COCO, ${87.6}$ on Dense200, ${91.0}$ on RefCOCOg-test, beating both larger generalists like OVIS2.5-9B and the point-specialist Rex-Omni-3B (${80.5}$/${82.5}$/${85.1}$ on the same three). That the *same* atomic-block formulation tops both box and point benchmarks is the strongest evidence that PBD learns a genuine geometric read-out, not a box-specific trick.

### The ablation that carries the argument

The main table mixes the PBD idea with ${138}$M-sample data scaling, so it cannot isolate the decoding contribution. The COCO-only ablation (Table 6, everything trained on COCO alone) does, and it is the most convincing evidence in the paper.

![On COCO-only training, the three PBD modes push the throughput-accuracy frontier past every NTP and generic-MTP baseline: PBD-Slow beats both NTP encodings on accuracy at equal speed, while PBD-Fast and PBD-Hybrid multiply throughput without collapsing F1.](/imgs/blogs/locate-anything-3.webp)

Read the frontier scatter above point by point:

| Method | Throughput | Mean F1 | Reading |
|---|---|---|---|
| Textual (NTP) | ${1.3}$ | ${49.1}$ | Slowest, weakest — per-digit serialization |
| Quantized (NTP) | ${3.9}$ | ${50.1}$ | Faster tokens, still 1D |
| Block-Diff-B6 (generic MTP) | ${4.7}$ | ${44.8}$ | Parallel *and worse* — unaligned chunks |
| SDLM-B6 (generic MTP) | ${5.5}$ | ${46.1}$ | Same story |
| **PBD (Slow)** | ${3.9}$ | **52.1** | Best accuracy, same speed as quantized |
| **PBD (Hybrid)** | ${13.2}$ | ${51.6}$ | Near-Slow accuracy, ${3.4}\times$ the speed |
| **PBD (Fast)** | ${16.9}$ | ${49.6}$ | Fastest, still beats both NTP encodings |

Three conclusions fall out. **(1)** Box-aligned supervision alone helps: PBD-Slow at ${52.1}$ beats Quantized-NTP at ${50.1}$ *at identical throughput* — the box-alignment improved accuracy with no speed cost, confirming the structure-mismatch thesis. **(2)** Generic MTP is a trap: SDLM and Block-Diffusion sit *below* the NTP baselines on accuracy while barely moving speed (${44.8}$–${46.1}$ F1 at ${4.7}$–${5.5}$ BPS), and increasing their block size only trades a sliver of speed for more accuracy loss. Alignment, not parallelism per se, is what matters. **(3)** PBD-Fast at ${16.9}$ BPS still clears both NTP encodings on F1 — the fast path is not a desperate accuracy sacrifice. The dashed frontier connects the three PBD points; every baseline is strictly inside it.

The decoding design also transfers backbones: instantiated on Qwen3-VL-4B (COCO-only), PBD lifts COCO F1 from ${50.8}$ to ${52.0}$ while raising throughput from ${2.8}$ to ${9.4}$ BPS — so the gains are not tied to the Moon-ViT/Qwen2.5 pairing. One more ablation worth noting: box output *order* matters, and sorting boxes by the top-left corner (x then y) beats center-distance, area, and random — a reminder that autoregressive-over-blocks still imposes a sequence, and its order is a real hyperparameter.

**What might not transfer.** The main-table SOTA leans on two things a re-implementer may not have: the ${138}$M-sample data engine (built from four other strong models) and native-resolution inference (COCO/LVIS evaluated at short-side ${840}$px). Strip those away — as the COCO-only ablation does — and the *decoding* gain is real but more modest (${+2}$ F1 in Slow Mode). The ${+3.8}$/${+1.8}$ headline improvements over Rex-Omni are PBD *and* data *and* resolution together.

## Critique

**What's genuinely strong.** The central insight is clean and, in hindsight, obviously right: the unit of parallel prediction should match the unit of geometric coupling. The paper does not just assert this — the generic-MTP baselines in Table 6 are a real, controlled refutation of the naive alternative, and the high-IoU LVIS jump (${20.7 \to 31.1}$) is the kind of result that is hard to get by accident. The dual-formulation ablation (Slow-Mode F1 ${50.1 \to 52.1}$ from joint training) is subtle and convincing: it shows the two streams are *complementary*, not redundant. And the hybrid fallback is an honest engineering answer to a real failure mode rather than a paper-over.

**What's weak or unfalsifiable.** Three things. First, the fallback thresholds (${p < 0.7}$, spread ${> 80}$) are magic constants with no sensitivity analysis — how much does F1 or throughput move if they are ${0.6}$ or ${0.8}$? Without that sweep, I cannot tell how robust Hybrid Mode is, and the numbers could be lightly tuned to COCO. Second, the throughput comparison is measured at batch size ${1}$; the sequential-decode disadvantage of NTP baselines partly disappears under batching, which is how these models actually serve — the ${10}\times$ claim is a best case for PBD's setup. Third, the entire data engine is bootstrapped from Qwen3-VL, Molmo, SAM 3, and Rex-Omni, so "improves over Rex-Omni" is partly "distills Rex-Omni plus three other models at scale"; the paper never disentangles how much of the gain is the decoder versus the teachers.

**What ablation is missing.** A per-mode accuracy-vs-fallback-rate curve on a dense benchmark would show how often Hybrid actually falls back and what it costs — the single most useful number for anyone deploying this, and it is absent. A batched-throughput comparison. And a measurement of the format-irregularity / spatial-ambiguity *frequency* — the two failure modes are asserted with qualitative figures, not counted.

**What would change my mind.** If a sensitivity sweep showed the hybrid thresholds are knife-edge — that F1 swings several points across a reasonable range of ${p}$ and spread — I would downgrade this from "a clean idea with a robust guard" to "a clean idea with a fragile production story." Conversely, a batched-throughput table where PBD *keeps* a ${2}\times$-plus edge over quantized NTP would make the speed claim much stronger than the batch-size-${1}$ number currently supports.

## What I'd build with this

These are my extrapolations, not the paper's claims.

- **RL on the fallback policy.** The paper flags this itself. The hybrid trigger is a hand-tuned classifier; a small policy trained to minimize fallback rate subject to an IoU floor could push the frontier further and remove the magic constants — exactly the kind of thing GRPO-style post-training is good at.
- **PBD for non-box geometry.** The block idea is not box-specific. Polygons, rotated boxes, keypoint skeletons, and even segmentation-mask control points are all "sets of coordinates that are jointly determined." A ${2L}$-slot block for an ${L}$-vertex polygon, decoded in one pass, is a natural next target — and would test whether the coordination-through-hidden-states bet holds for larger geometric units.
- **Port the ambiguity trigger to general speculative decoding.** The "${p_{\text{top-1}} < 0.7}$ *and* top-5 spread large" test is a cheap, general signal for "this parallel guess is unreliable." A speculative-decoding verifier for arbitrary structured output (JSON, code) could reuse the same two-clause gate to decide when to fall back to careful sequential decoding.
- **A batched serving benchmark.** Before trusting the ${10}\times$ claim in production, I would build the missing batched-throughput comparison and measure the real fallback rate under load — the two numbers that decide whether Hybrid Mode is a genuine win or a batch-size-${1}$ artifact.

## References

- **Paper:** Wang, Liu, et al. *LocateAnything: Fast and High-Quality Vision-Language Grounding with Parallel Box Decoding.* NVIDIA LPR, 2026. [PDF](https://research.nvidia.com/labs/lpr/locate-anything/LocateAnything.pdf) · [Project page](https://research.nvidia.com/labs/lpr/locate-anything/)
- **Prior art it builds on / compares to:** Pix2Seq (coordinate tokenization), Rex-Omni (the most-related quantized-token grounding VLM), Medusa / Block Diffusion / SDLM (the generic-MTP baselines it refutes), SAM 3 and Molmo (data-engine components).
- **Sibling posts on this blog:** [SEED1.5-VL](/blog/paper-reading/multimodal/seed1-5-vl-native-resolution-vision-language) (a native-resolution grounding VLM baseline here), [Qwen2-VL](/blog/paper-reading/multimodal/qwen2-vl-enhancing-vision-language-models-perception-of-the-world-at-any-resolution) (the any-resolution, coordinate-token lineage), [Grounding Everything in Tokens](/blog/paper-reading/multimodal/grounding-everything-in-tokens-for-multimodal-large-language-models) (the token-as-grounding formulation), and [Kimi-VL](/blog/paper-reading/multimodal/kimi-vl) (the Moon-ViT encoder LocateAnything reuses).
