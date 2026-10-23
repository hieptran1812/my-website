---
title: "DFlash: Block Diffusion for Flash Speculative Decoding, and What Changed from v1 to v2"
date: "2026-10-23"
description: "How DFlash drafts a whole block of tokens in one forward pass by conditioning a five-layer diffusion drafter on the target model's own hidden states, and exactly what the ICML 2026 revision added to the February preprint."
tags: ["paper-reading", "dflash", "speculative-decoding", "block-diffusion", "diffusion-language-models", "llm-inference", "kv-injection", "eagle-3"]
category: "paper-reading"
subcategory: "Large Language Model"
author: "Hiep Tran"
featured: true
readTime: 14
paper:
  title: "DFlash: Block Diffusion for Flash Speculative Decoding"
  authors: "Jian Chen, Yesheng Liang, Zhijian Liu"
  venue: "ICML 2026 (arXiv 2602.06036v2)"
  url: "https://arxiv.org/abs/2602.06036"
---

> [!tldr]
> - **What it proposes.** Replace the autoregressive drafter with a small block diffusion model that fills 16 masked positions in one forward pass; the target still verifies, so output is lossless.
> - **The key mechanism.** The drafter reads fused hidden states from five target layers, injected as extra keys and values into each of its five layers.
> - **What it beats.** On Qwen3-8B with greedy decoding it averages a 4.86x speedup and an acceptance length of 6.49 tokens, against 1.76x and 2.96 for EAGLE-3 at the same 16-token budget.
> - **Most surprising result.** A five-layer drafter emitting 16 tokens is *cheaper* per cycle than a one-layer EAGLE-3 drafter emitting 8, because one wide pass beats eight narrow ones on a GPU.
> - **Where it is thin.** Gains shrink with batch size (5.1x at concurrency 1 falls to 2.8x at 32 on SGLang), chat tasks gain far less than math, and there is no head-to-head against other diffusion drafters.
> - **v1 to v2.** The ICML version keeps every v1 number and adds four pieces of evidence: long-context adaptation, a KV injection versus input fusion ablation, the injection equations with a memory budget, and results on more models plus vLLM.

![Figure 1 from Chen, Liang and Liu (2026): speedup of DFlash and EAGLE-3 over autoregressive decoding on Qwen3-8B across math, code and chat benchmarks](/imgs/blogs/dflash-block-diffusion-drafter-v1-v2-fig1.webp)

On every benchmark the DFlash bar towers over EAGLE-3. The rest of this post explains where that gap comes from.

## The problem: the drafter was still serial

[Speculative decoding](/blog/machine-learning/large-language-model/speculative-decoding) lets a cheap drafter guess the next $\gamma$ tokens and the target check them in one parallel pass, keeping the matching prefix plus one "bonus" token of its own. The output distribution is exactly the target's.

The best prior drafters, the [EAGLE family](/blog/machine-learning/speculative-decoding/eagle-speculative-decoding-feature-alignment), still guess one token per forward pass. To keep 8 passes affordable, EAGLE-3 shrinks the drafter to one layer, whose errors compound, capping speedups at roughly 2 to 3x.

Diffusion language models fill many masked positions at once, but as standalone generators they lag in quality, and prior diffusion drafters (DiffuSpec, SpecDiff-2) used 7B-parameter models that stall around 3 to 4x. DFlash wants a diffusion drafter that is tiny and accurate. Its answer, in the paper's phrase: "the target knows best."

## Contributions

1. A drafter that is a lightweight block diffusion model, emitting a whole draft block in one forward pass.
2. **KV injection**: target hidden states become persistent keys and values in every draft layer, so acceptance grows with drafter depth.
3. A training recipe matched to inference: clean-token anchored blocks, a position-decaying loss, a frozen shared embedding and LM head.

## Method

![Figure 2 from Chen, Liang and Liu (2026): the DFlash inference design, where fused target context features are injected into each draft layer's KV cache](/imgs/blogs/dflash-block-diffusion-drafter-v1-v2-fig2.webp)

In Figure 2, the target decodes a token ("for"); hidden states from several of its layers are fused into a **target context feature** written into every draft layer's KV cache. The drafter reads the clean token plus mask tokens with bidirectional attention, and the target's frozen LM head turns its outputs into draft tokens for verification.

### Technique 1: parallel drafting changes the cost model

**The problem.** If drafting cost grows with the number of guesses, you cannot afford many guesses or a deep drafter.

**Intuition.** A scribe writes one line at a time; a photocopier prints the whole page in one press. A GPU decode step is a mostly idle photocopier: one pass over one token is memory-bound, so 16 positions instead of 1 barely changes wall-clock time.

**Mechanism.** A cycle costs drafting plus verifying, divided by the tokens it yields. An autoregressive drafter pays one pass per guess; a block diffusion drafter pays one pass per block.

**The math.** Following the paper's Eq. 1, the average per-token latency is

$$
L = \frac{T_{\text{draft}} + T_{\text{verify}}}{\tau}, \qquad \eta = \frac{L_{\text{target}}}{L},
$$

where $T_{\text{draft}}$ is the time to produce the draft, $T_{\text{verify}}$ is the time for the target to check it, $\tau \in [1, \gamma + 1]$ is the expected number of tokens accepted per cycle (including the bonus token), $L_{\text{target}}$ is the plain autoregressive per-token latency of the target, and $\eta$ is the speedup. The two drafting regimes differ only in $T_{\text{draft}}$:

$$
T_{\text{draft}}^{\text{AR}} = \gamma \cdot t_{\text{step}}, \qquad T_{\text{draft}}^{\text{diffusion}} = t_{\text{parallel}} \ll \gamma \cdot t_{\text{step}},
$$

with $t_{\text{step}}$ the latency of one drafter forward pass and $t_{\text{parallel}}$ the latency of one pass over the whole block. For moderate block sizes, $t_{\text{parallel}}$ is nearly flat in $\gamma$.

**Worked example (my illustrative numbers, not the paper's).** Say one target forward costs 20 ms and verification costs about one target forward. EAGLE-3 with $t_{\text{step}} = 1$ ms drafting 7 tokens gives $T_{\text{draft}} = 7$ ms and, with $\tau = 3.0$, $L = 27/3 = 9$ ms, so $\eta \approx 2.2$x. A five-layer DFlash pass of, say, 3 ms with $\tau = 6.5$ gives $L = 23/6.5 \approx 3.5$ ms, so $\eta \approx 5.7$x. Note what moved: the drafter got deeper and *cheaper per cycle*, and acceptance doubled.

![Figure 3 from Chen, Liang and Liu (2026): draft cost of 1, 3 and 5-layer DFlash against 1-layer EAGLE-3 as the number of draft tokens grows](/imgs/blogs/dflash-block-diffusion-drafter-v1-v2-fig3.webp)

Figure 3 measures it: EAGLE-3's cost climbs with draft tokens, the DFlash lines stay nearly flat, and five layers at 16 tokens sit below EAGLE-3 at 8. The animation shows the rhythm: eight beats versus one.

<figure class="blog-anim">
<svg viewBox="0 0 900 380" role="img" aria-label="Top lane: an autoregressive drafter fills 8 draft slots one at a time, one forward pass each. Bottom lane: DFlash turns a bonus token and 15 mask slots into tokens on one beat, then verification marks a green accepted prefix and a red first mismatch." style="width:100%;height:auto;max-width:880px">
<style>
.d1-h{font:600 16px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937)}
.d1-s{font:500 13px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280)}
.d1-n{font:700 16px ui-sans-serif,system-ui;fill:var(--accent,#6366f1)}
.d1-slot{fill:var(--surface,#f3f4f6);stroke:var(--border,#d1d5db);stroke-width:1.5}
.d1-tok{fill:var(--accent,#6366f1)}
.d1-m{font:600 13px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280);text-anchor:middle}
.d1-ok{fill:none;stroke:#16a34a;stroke-width:3}
.d1-bad{fill:none;stroke:#dc2626;stroke-width:3}
.d1-dim{fill:var(--background,#fff);opacity:.7}
@keyframes d1-t0{0%,5.0%{opacity:0}9.2%,92%{opacity:1}97%,100%{opacity:0}}
.d1-t0{animation:d1-t0 6.0s ease-in-out infinite}
@keyframes d1-t1{0%,14.2%{opacity:0}18.3%,92%{opacity:1}97%,100%{opacity:0}}
.d1-t1{animation:d1-t1 6.0s ease-in-out infinite}
@keyframes d1-t2{0%,23.3%{opacity:0}27.5%,92%{opacity:1}97%,100%{opacity:0}}
.d1-t2{animation:d1-t2 6.0s ease-in-out infinite}
@keyframes d1-t3{0%,32.5%{opacity:0}36.7%,92%{opacity:1}97%,100%{opacity:0}}
.d1-t3{animation:d1-t3 6.0s ease-in-out infinite}
@keyframes d1-t4{0%,41.7%{opacity:0}45.8%,92%{opacity:1}97%,100%{opacity:0}}
.d1-t4{animation:d1-t4 6.0s ease-in-out infinite}
@keyframes d1-t5{0%,50.8%{opacity:0}55.0%,92%{opacity:1}97%,100%{opacity:0}}
.d1-t5{animation:d1-t5 6.0s ease-in-out infinite}
@keyframes d1-t6{0%,60.0%{opacity:0}64.2%,92%{opacity:1}97%,100%{opacity:0}}
.d1-t6{animation:d1-t6 6.0s ease-in-out infinite}
@keyframes d1-t7{0%,69.2%{opacity:0}73.3%,92%{opacity:1}97%,100%{opacity:0}}
.d1-t7{animation:d1-t7 6.0s ease-in-out infinite}
@keyframes d1-flip{0%,16.7%{opacity:0}20.8%,92%{opacity:1}97%,100%{opacity:0}}
@keyframes d1-mask{0%,16.7%{opacity:1}20.8%,92%{opacity:0}97%,100%{opacity:1}}
.d1-flip{animation:d1-flip 6.0s ease-in-out infinite}
.d1-mask{opacity:0;animation:d1-mask 6.0s ease-in-out infinite}
@keyframes d1-v0{0%,33.3%{opacity:0}35.8%,92%{opacity:1}97%,100%{opacity:0}}
.d1-v0{animation:d1-v0 6.0s ease-in-out infinite}
@keyframes d1-v1{0%,36.7%{opacity:0}39.2%,92%{opacity:1}97%,100%{opacity:0}}
.d1-v1{animation:d1-v1 6.0s ease-in-out infinite}
@keyframes d1-v2{0%,40.0%{opacity:0}42.5%,92%{opacity:1}97%,100%{opacity:0}}
.d1-v2{animation:d1-v2 6.0s ease-in-out infinite}
@keyframes d1-v3{0%,43.3%{opacity:0}45.8%,92%{opacity:1}97%,100%{opacity:0}}
.d1-v3{animation:d1-v3 6.0s ease-in-out infinite}
@keyframes d1-v4{0%,46.7%{opacity:0}49.2%,92%{opacity:1}97%,100%{opacity:0}}
.d1-v4{animation:d1-v4 6.0s ease-in-out infinite}
@keyframes d1-v5{0%,50.0%{opacity:0}52.5%,92%{opacity:1}97%,100%{opacity:0}}
.d1-v5{animation:d1-v5 6.0s ease-in-out infinite}
@keyframes d1-v6{0%,53.3%{opacity:0}55.8%,92%{opacity:1}97%,100%{opacity:0}}
.d1-v6{animation:d1-v6 6.0s ease-in-out infinite}
@keyframes d1-rest{0%,56.7%{opacity:0}60.0%,92%{opacity:1}97%,100%{opacity:0}}
.d1-rest{animation:d1-rest 6.0s ease-in-out infinite}
@media (prefers-reduced-motion:reduce){.d1-t0,.d1-t1,.d1-t2,.d1-t3,.d1-t4,.d1-t5,.d1-t6,.d1-t7,.d1-flip,.d1-mask,.d1-v0,.d1-v1,.d1-v2,.d1-v3,.d1-v4,.d1-v5,.d1-v6,.d1-rest{animation:none}}
</style>
<text class="d1-h" x="20" y="34">Autoregressive drafter (EAGLE-3)</text>
<text class="d1-n" x="880" y="34" text-anchor="end">8 passes</text>
<text class="d1-s" x="20" y="54">one forward pass per draft token</text>
<rect class="d1-slot" x="20" y="66" width="44" height="44" rx="6"/>
<rect class="d1-tok d1-t0" x="20" y="66" width="44" height="44" rx="6"/>
<text class="d1-m d1-t0" x="42" y="128">pass 1</text>
<rect class="d1-slot" x="72" y="66" width="44" height="44" rx="6"/>
<rect class="d1-tok d1-t1" x="72" y="66" width="44" height="44" rx="6"/>
<text class="d1-m d1-t1" x="94" y="128">pass 2</text>
<rect class="d1-slot" x="124" y="66" width="44" height="44" rx="6"/>
<rect class="d1-tok d1-t2" x="124" y="66" width="44" height="44" rx="6"/>
<text class="d1-m d1-t2" x="146" y="128">pass 3</text>
<rect class="d1-slot" x="176" y="66" width="44" height="44" rx="6"/>
<rect class="d1-tok d1-t3" x="176" y="66" width="44" height="44" rx="6"/>
<text class="d1-m d1-t3" x="198" y="128">pass 4</text>
<rect class="d1-slot" x="228" y="66" width="44" height="44" rx="6"/>
<rect class="d1-tok d1-t4" x="228" y="66" width="44" height="44" rx="6"/>
<text class="d1-m d1-t4" x="250" y="128">pass 5</text>
<rect class="d1-slot" x="280" y="66" width="44" height="44" rx="6"/>
<rect class="d1-tok d1-t5" x="280" y="66" width="44" height="44" rx="6"/>
<text class="d1-m d1-t5" x="302" y="128">pass 6</text>
<rect class="d1-slot" x="332" y="66" width="44" height="44" rx="6"/>
<rect class="d1-tok d1-t6" x="332" y="66" width="44" height="44" rx="6"/>
<text class="d1-m d1-t6" x="354" y="128">pass 7</text>
<rect class="d1-slot" x="384" y="66" width="44" height="44" rx="6"/>
<rect class="d1-tok d1-t7" x="384" y="66" width="44" height="44" rx="6"/>
<text class="d1-m d1-t7" x="406" y="128">pass 8</text>
<line x1="20" y1="160" x2="880" y2="160" stroke="var(--border,#d1d5db)" stroke-width="1" stroke-dasharray="4 4"/>
<text class="d1-h" x="20" y="196">DFlash block diffusion (block 16)</text>
<text class="d1-n" x="880" y="196" text-anchor="end">1 pass</text>
<text class="d1-s" x="20" y="216">bonus token + 15 masks become tokens together, then the target verifies</text>
<rect class="d1-slot" x="20" y="230" width="46" height="46" rx="6"/>
<rect class="d1-tok" x="20" y="230" width="46" height="46" rx="6" opacity=".55"/>
<rect class="d1-slot" x="74" y="230" width="46" height="46" rx="6"/>
<text class="d1-m d1-mask" x="97" y="258">m</text>
<rect class="d1-tok d1-flip" x="74" y="230" width="46" height="46" rx="6"/>
<rect class="d1-slot" x="128" y="230" width="46" height="46" rx="6"/>
<text class="d1-m d1-mask" x="151" y="258">m</text>
<rect class="d1-tok d1-flip" x="128" y="230" width="46" height="46" rx="6"/>
<rect class="d1-slot" x="182" y="230" width="46" height="46" rx="6"/>
<text class="d1-m d1-mask" x="205" y="258">m</text>
<rect class="d1-tok d1-flip" x="182" y="230" width="46" height="46" rx="6"/>
<rect class="d1-slot" x="236" y="230" width="46" height="46" rx="6"/>
<text class="d1-m d1-mask" x="259" y="258">m</text>
<rect class="d1-tok d1-flip" x="236" y="230" width="46" height="46" rx="6"/>
<rect class="d1-slot" x="290" y="230" width="46" height="46" rx="6"/>
<text class="d1-m d1-mask" x="313" y="258">m</text>
<rect class="d1-tok d1-flip" x="290" y="230" width="46" height="46" rx="6"/>
<rect class="d1-slot" x="344" y="230" width="46" height="46" rx="6"/>
<text class="d1-m d1-mask" x="367" y="258">m</text>
<rect class="d1-tok d1-flip" x="344" y="230" width="46" height="46" rx="6"/>
<rect class="d1-slot" x="398" y="230" width="46" height="46" rx="6"/>
<text class="d1-m d1-mask" x="421" y="258">m</text>
<rect class="d1-tok d1-flip" x="398" y="230" width="46" height="46" rx="6"/>
<rect class="d1-slot" x="452" y="230" width="46" height="46" rx="6"/>
<text class="d1-m d1-mask" x="475" y="258">m</text>
<rect class="d1-tok d1-flip" x="452" y="230" width="46" height="46" rx="6"/>
<rect class="d1-slot" x="506" y="230" width="46" height="46" rx="6"/>
<text class="d1-m d1-mask" x="529" y="258">m</text>
<rect class="d1-tok d1-flip" x="506" y="230" width="46" height="46" rx="6"/>
<rect class="d1-slot" x="560" y="230" width="46" height="46" rx="6"/>
<text class="d1-m d1-mask" x="583" y="258">m</text>
<rect class="d1-tok d1-flip" x="560" y="230" width="46" height="46" rx="6"/>
<rect class="d1-slot" x="614" y="230" width="46" height="46" rx="6"/>
<text class="d1-m d1-mask" x="637" y="258">m</text>
<rect class="d1-tok d1-flip" x="614" y="230" width="46" height="46" rx="6"/>
<rect class="d1-slot" x="668" y="230" width="46" height="46" rx="6"/>
<text class="d1-m d1-mask" x="691" y="258">m</text>
<rect class="d1-tok d1-flip" x="668" y="230" width="46" height="46" rx="6"/>
<rect class="d1-slot" x="722" y="230" width="46" height="46" rx="6"/>
<text class="d1-m d1-mask" x="745" y="258">m</text>
<rect class="d1-tok d1-flip" x="722" y="230" width="46" height="46" rx="6"/>
<rect class="d1-slot" x="776" y="230" width="46" height="46" rx="6"/>
<text class="d1-m d1-mask" x="799" y="258">m</text>
<rect class="d1-tok d1-flip" x="776" y="230" width="46" height="46" rx="6"/>
<rect class="d1-slot" x="830" y="230" width="46" height="46" rx="6"/>
<text class="d1-m d1-mask" x="853" y="258">m</text>
<rect class="d1-tok d1-flip" x="830" y="230" width="46" height="46" rx="6"/>
<rect class="d1-ok d1-v0" x="17" y="227" width="52" height="52" rx="8"/>
<rect class="d1-ok d1-v1" x="71" y="227" width="52" height="52" rx="8"/>
<rect class="d1-ok d1-v2" x="125" y="227" width="52" height="52" rx="8"/>
<rect class="d1-ok d1-v3" x="179" y="227" width="52" height="52" rx="8"/>
<rect class="d1-ok d1-v4" x="233" y="227" width="52" height="52" rx="8"/>
<rect class="d1-ok d1-v5" x="287" y="227" width="52" height="52" rx="8"/>
<rect class="d1-bad d1-v6" x="341" y="227" width="52" height="52" rx="8"/>
<rect class="d1-dim d1-rest" x="396" y="228" width="490" height="50" rx="6"/>
<text class="d1-s d1-rest" x="20" y="306" style="fill:#16a34a">green: accepted prefix (matches the target)</text>
<text class="d1-s d1-rest" x="344" y="326" style="fill:#dc2626">red: first mismatch, the rest is discarded</text>
<text class="d1-s" x="20" y="366">Both lanes run on the same 6 s clock: by the time DFlash has drafted all 16 slots, EAGLE-3 has 2.</text>
</svg>
<figcaption>Same clock, two drafters. The autoregressive drafter needs 8 sequential forward passes for 8 tokens; DFlash fills a 16-slot block in a single pass, and the target then keeps the matching prefix up to the first mismatch.</figcaption>
</figure>

**When it fails.** Flat cost holds only while the GPU is underused. At large batch sizes the target is compute-bound, a 16-token block makes verification expensive, and the SGLang speedup falls with concurrency. The authors suggest shrinking the block under load and leave that to future work.

### Technique 2: the target knows best, via KV injection

**The problem.** A small diffusion model guessing 16 future tokens on its own is bad at it. The paper tests exactly this: a five-layer block diffusion drafter with no target conditioning gets only 2.83x on GSM8K and 3.73x on MATH-500 at temperature 0 (Table 10), with acceptance lengths of 3.38 and 4.61.

**Intuition.** The target model has already read the whole context and, as Samragh et al. (2025) observed, its hidden states quietly encode information about several future tokens. The drafter should be a translator of that knowledge, not an independent thinker. The paper calls it a "diffusion adapter."

**Mechanism.** DFlash takes hidden states from 5 target layers, spread uniformly from the second to the third-to-last, concatenates them, projects them once to the drafter's width, and stores the result in each draft layer's KV cache across drafting iterations. EAGLE-3 fuses target features with token embeddings at the *input* only, so in a deep drafter the signal dilutes layer by layer.

**The math.** This is spelled out only in v2 (Appendix A.3). Let $H^{(l)} \in \mathbb{R}^{n \times D}$ be the target's hidden states at layer $l$ for the $n$ context tokens, with $D$ the hidden width. The fused feature is

$$
H_t = \text{RMSNorm}\!\left( W_c \left[ H^{(l_1)} ; \dots ; H^{(l_5)} \right] \right), \qquad W_c \in \mathbb{R}^{D \times 5D},
$$

where $[\,\cdot\,;\,\cdot\,]$ concatenates along the feature axis, so the input is $n \times 5D$ and $H_t$ is $n \times D$. At draft layer $i$, with $H_d \in \mathbb{R}^{b \times D}$ the hidden states of the $b$ block positions:

$$
Q_i = W_i^Q H_d, \qquad K_i = \left[ W_i^K H_t ; W_i^K H_d \right]_{\text{seq}}, \qquad V_i = \left[ W_i^V H_t ; W_i^V H_d \right]_{\text{seq}}.
$$

The subscript "seq" means concatenation along the sequence axis: the keys and values have $n + b$ rows, but only the $b$ block positions issue queries. Target features never pass through the drafter's query projection, output projection or FFN. They are pure memory that every layer can look up.

![KV injection drawn as one draft layer's attention grid: four target-feature columns and four block columns, queried only by the bonus token and three mask positions](/imgs/blogs/dflash-block-diffusion-drafter-v1-v2-2.webp)

In this toy grid, every query row (bonus token $x_0$, masks $m_1$ to $m_3$) sees all four blue target-feature columns and all four green block columns, in all five layers.

**Worked example.** The memory cost is small. For Qwen3.5-35B-A3B, $D = 2048$ in BF16 (2 bytes), so $W_c$ costs $5 \times 2048 \times 2048 \times 2 \approx 42$ MB next to a roughly 70 GB target. The paper puts decode-time activation for a 16-token block below 400 KB.

**Why it works, and the evidence.** The v2 ablation (Table 9, Qwen3-4B, five-layer drafters, block size 8) isolates the injection choice:

| Variant | Injection | GSM8K τ / speedup | HumanEval | MT-Bench |
| --- | --- | --- | --- | --- |
| EAGLE-3-5L (autoregressive) | input | 4.2 / 2.1x | 4.3 / 2.2x | 3.1 / 1.4x |
| DFlash-AR (autoregressive) | KV | 4.8 / 2.4x | 4.6 / 2.3x | 3.4 / 1.5x |
| DFlash (block diffusion) | input | 3.5 / 2.9x | 3.5 / 2.9x | 2.6 / 2.0x |
| DFlash (block diffusion) | KV | 4.2 / 3.3x | 4.0 / 3.2x | 3.0 / 2.2x |

KV injection raises acceptance in both drafting styles, and block diffusion wins on speed even with *lower* acceptance than autoregressive drafting. In the depth ablation (Table 6), MATH-500 acceptance climbs 5.64, 5.99, 6.33 for 3, 5 and 8 layers, but speedup peaks at 5 (4.71x).

### Technique 3: training that matches inference

**The problem.** Standard block diffusion training chops a response into fixed blocks and masks random positions. At inference, DFlash's block always starts with one clean token (the target's bonus token) followed by masks. Training on a different pattern wastes capacity.

**Mechanism.** DFlash samples random **anchor** tokens in each response (512 per sequence in the paper's setup), makes each anchor the first position of a block, and masks the next block-size minus one positions. All blocks are packed into one sequence and trained in a single forward and backward pass with a sparse mask built in Flex Attention: each block sees itself bidirectionally plus the injected target features, and never another block.

![Figure 4 from Chen, Liang and Liu (2026): the training attention mask, with clean prompt and response tokens, anchor tokens starting each masked block, and invisible cells blocking cross-block leakage](/imgs/blogs/dflash-block-diffusion-drafter-v1-v2-fig4.webp)

Random anchors double as data augmentation. In Table 13 (three-layer drafters, 100K samples) MATH-500 goes from 4.13x / 4.94 with standard blocks to 4.69x / 5.64 with sampled anchors, and the fixed block count keeps long-context training cost bounded.

**The loss.** Errors early in a block are worse than errors late, because verification accepts only a prefix. DFlash weights the cross-entropy at block position $k$ by

$$
w_k = \exp\!\left( -\frac{k - 1}{\gamma} \right),
$$

where $\gamma$ is a decay constant (7 for block size 16, 5 for 10, 4 for 8). Note the paper reuses $\gamma$ here; it is not the speculation budget from Eq. 1.

![Loss weights for a 16-token block with decay constant 7, falling from 1.00 to 0.12, above a draft row where a mismatch at position 6 discards positions 7 to 16](/imgs/blogs/dflash-block-diffusion-drafter-v1-v2-3.webp)

**Worked example.** With $\gamma = 7$: $w_1 = 1$, $w_8 = e^{-1} \approx 0.37$, $w_{16} = e^{-15/7} \approx 0.12$. Position 1 counts about eight times as much as position 16. If tokens 1 to 5 are right and token 6 wrong, the target accepts 5 plus its bonus, and tokens 7 to 16 are discarded even if correct. The paper's Figure 5 shows the decayed loss converging faster and higher than uniform weighting.

The drafter also shares the target's frozen embedding and LM head, so only its transformer layers train, and its 800K training samples (Nemotron Post-Training v2 plus CodeAlpaca) are regenerated by the target. The [SpecForge deep-dive](/blog/machine-learning/open-source-library/specforge-speculative-decoding-training-deep-dive) walks the training code.

## Experiments and results

Transformers runs use H200s, SGLang runs one B200. The drafter has 5 layers and block size 16 (10 for LLaMA-3.1); EAGLE-3 uses AngelSlim checkpoints for Qwen3.

![Average acceptance length and speedup on Qwen3-8B at temperature 0: EAGLE-3 with tree sizes 16 and 60 versus DFlash with block size 16](/imgs/blogs/dflash-block-diffusion-drafter-v1-v2-1.webp)

| Setting (source) | Baseline | EAGLE-3 | DFlash |
| --- | --- | --- | --- |
| Qwen3-8B, temp 0, avg of 7 tasks (Table 1) | 1.00x | 1.76x, τ 2.96 (tree 16); 2.02x, τ 3.40 (tree 60) | 4.86x, τ 6.49 |
| Qwen3-8B, temp 1, avg (Table 1) | 1.00x | 1.68x (tree 16); 1.88x (tree 60) | 4.03x, τ 5.48 |
| Qwen3-8B thinking, MATH-500, temp 0 (Table 2) | 1.00x | not reported | 4.64x, τ 5.82 |
| Qwen3-8B SGLang MATH-500, concurrency 1 / 32 (Table 3) | 230 / 5694 tok/s | not reported | 5.1x / 2.8x |
| LLaMA-3.1-8B SGLang GSM8K, concurrency 1 (Table 5) | 249 tok/s | 1.6x (10), 1.9x (60) | 2.4x |

**What's load-bearing.** The LLaMA-3.1 row is the fairest comparison: the same training data as EAGLE-3 and its official checkpoint. There the margin shrinks to roughly 1.3 to 1.5x. Math carries the headline; MT-Bench speedups are about half as large.

Block size has a useful asymmetry (Table 8): a drafter trained at 16 and run at 8 reaches τ 5.09 on MATH-500, close to the 5.21 of one trained at 8, but the reverse gains nothing (5.02). Train large, shrink at serving time.

## What changed from v1 to v2

I diffed the text of v1 (5 Feb 2026, 11 pages, "Preprint") against v2 (28 May 2026, 13 pages, ICML 2026, PMLR 306). Every number in v1's tables reappears unchanged in v2; the method section, Eqs. 1 to 4 and the training hyperparameters are identical. The revision is additive.

![Content map of v1 versus v2: method and Tables 1 to 3 unchanged, four new evidence sections, one renamed section, renumbered ablations, and a new venue](/imgs/blogs/dflash-block-diffusion-drafter-v1-v2-5.webp)

What v2 **adds**:

1. **Long-context adaptation (new Section 5.4, Table 4).** A Qwen3.5-27B drafter trained at 4K context, fine-tuned on 1.6K LongAlign-10K samples for 3 epochs and tested on LongBench. On HotpotQA the base drafter falls from 4.91 at 4K to 3.61 at 16K, while the fine-tuned one rises to 6.05. On gov report at 16K: 3.57 versus 6.00; on qasper at 32K: 2.09 versus 3.56.
2. **KV injection versus input fusion (new Section 5.5.5, Table 9).** The four-row table above. In v1, the claim that injection beats input fusion was argued but not isolated.
3. **The injection equations and memory budget (new Appendix A.3).** The $H_t$, $Q_i$, $K_i$, $V_i$ equations and the 42 MB / 400 KB overhead figures. v1 described injection in words only.
4. **More models and vLLM (new Appendix A.4, Tables 11 and 12).** On SGLang at concurrency 8, DFlash beats Qwen3.5's native multi-token prediction head everywhere (Qwen3.5-9B MATH-500: τ 7.3 at 3.5x versus 6.7 at 1.7x), and runs on Qwen3.5-27B, Qwen3-Coder-Next and GPT-OSS-20B/120B at 1.2x to 3.9x. On vLLM, Qwen3.5-9B MATH-500 gets 4.0x at concurrency 1 and 1.9x at 32.

What v2 **renames or moves**: "Performance on SGLang" becomes "Performance on Serving Frameworks"; ablations move from Section 5.4 to 5.5, renumbering v1's Tables 4 to 9 as 5 to 13 and appendix A.3 as A.5. It adds a conflict-of-interest disclosure, an impact statement and new acknowledgements. Nothing is removed.

![Acceptance length versus context length on HotpotQA: the base drafter declines past its 4K training length while the long-context fine-tune keeps rising to 6.05 at 16K](/imgs/blogs/dflash-block-diffusion-drafter-v1-v2-4.webp)

The long-context result matters most in practice: the drafter's long-context weakness is a training-distribution gap that a tiny fine-tune closes, which is what reasoning models with long outputs need.

## Critique

**What's strong.** The cost argument is measured (Figure 3), the ablations isolate each choice, v2's injection ablation separates "deeper drafter" from "better conditioning," and the MTP comparison is the baseline a serving engineer would actually weigh.

**What's weak.** The text says DFlash gives "a 2.4x improvement over EAGLE-3 (16)" at temperature 0, but averaging Table 1's 4B and 8B rows gives about 4.9x over 1.79x, which is 2.7x. The quoted 2.4x matches the tree-60 baseline instead (4.9x over 2.05x). It is a conservative error, yet it shows the headline ratios deserve recomputation. No other diffusion drafter appears in any table (the authors cite missing code), the batch-size decline gets no cost breakdown, and v2's GPT-OSS-120B results (1.3x to 1.7x) show the method is far from uniformly 6x.

**Missing ablation.** Which five target layers to tap is fixed by a uniform rule; there is no sweep over layer choice, and none over the fused feature's dimensionality.

**What would change my mind:** an EAGLE-3 drafter trained on the same target-regenerated data, at matched parameter count and with KV injection, landing within 20% of DFlash's speedup. That would say the win is mostly conditioning and data, and the diffusion part is a convenient way to parallelize rather than the core idea.

## What I'd build with this

My extrapolations, not the paper's: an adaptive block-size scheduler driven by batch occupancy, using the train-at-16, serve-at-8 asymmetry; a per-position confidence signal for variable-length blocks, which is what the follow-up [DSpark](/blog/paper-reading/large-language-model/dspark-confidence-scheduled-speculative-decoding) builds on top of DFlash; and KV injection retrofitted into EAGLE-3, since DFlash-AR shows part of the gain transfers. For an engineering tour of v1, see the earlier [DFlash deep-dive](/blog/machine-learning/large-language-model/dflash-block-diffusion-speculative-decoding).

## References

- Chen, J., Liang, Y., and Liu, Z. *DFlash: Block Diffusion for Flash Speculative Decoding.* ICML 2026. [arXiv 2602.06036](https://arxiv.org/abs/2602.06036) (v1: [2602.06036v1](https://arxiv.org/abs/2602.06036v1), v2: [2602.06036v2](https://arxiv.org/abs/2602.06036v2)). Project page: [dflash.z-lab.ai](https://dflash.z-lab.ai).
- Li, Y. et al. *EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test.* [arXiv 2503.01840](https://arxiv.org/abs/2503.01840).
- Samragh, M. et al. *Your LLM Knows the Future: Uncovering Its Multi-Token Prediction Potential.* [arXiv 2507.11851](https://arxiv.org/abs/2507.11851).
- Arriola, M. et al. *Block Diffusion: Interpolating Between Autoregressive and Diffusion Language Models.* [arXiv 2503.09573](https://arxiv.org/abs/2503.09573).
- On this blog: [SpecForge training deep-dive](/blog/machine-learning/open-source-library/specforge-speculative-decoding-training-deep-dive), [DSpark: confidence-scheduled speculative decoding](/blog/paper-reading/large-language-model/dspark-confidence-scheduled-speculative-decoding), [EAGLE feature alignment](/blog/machine-learning/speculative-decoding/eagle-speculative-decoding-feature-alignment).
