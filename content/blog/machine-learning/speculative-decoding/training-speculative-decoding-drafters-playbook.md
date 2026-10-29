---
title: "Training a Speculative Decoding Drafter: Data, Algorithms, Evaluation, and the Frameworks That Do It"
date: "2026-10-29"
publishDate: "2026-10-29"
description: "A practitioner's playbook for training and fine-tuning draft models: how to think about acceptance length, how to build the data, which algorithm to pick from Medusa to EAGLE-3 to DFlash, how to evaluate honestly, the mistakes that silently cost you speed, and where SpecForge and DeepSpec fit."
tags: ["speculative-decoding", "draft-model", "eagle-3", "specforge", "deepspec", "dflash", "llm-inference", "distillation", "training"]
category: "machine-learning"
subcategory: "Speculative Decoding"
author: "Hiep Tran"
featured: true
readTime: 13
---

Most teams meet speculative decoding as a serving flag. They download a public EAGLE checkpoint, turn it on, and get a 2x speedup on the benchmark. Then they run it on their own traffic and get 1.3x. The checkpoint was fine. The trouble is that a drafter is trained to imitate one particular model on one particular kind of text, and theirs was trained on somebody else's.

This post is about closing that gap by training or fine-tuning the drafter yourself. We will cover how to think about the objective, how to build the data, which algorithm to choose, how to evaluate it, the mistakes that quietly cost you speed, and the frameworks (SpecForge, DeepSpec and a few others) that do the heavy lifting.

<figure class="blog-anim">
<svg viewBox="0 0 780 340" role="img" aria-label="Two lanes on one clock: the cheap drafter proposes five tokens one at a time, the expensive target verifies all five in one forward pass, the first three are accepted, the fourth is replaced by the target's token upon, the fifth is discarded, giving tau equal to four tokens for one target pass" style="width:100%;height:auto;max-width:820px">
<style>
.ts1-lane{fill:var(--surface,#f3f4f6);stroke:var(--border,#d1d5db);stroke-width:1.2}
.ts1-lbl{font:600 15px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937)}
.ts1-sub{font:500 12px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280)}
.ts1-tok{font:600 14px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937);text-anchor:middle}
.ts1-pulse{fill:var(--text-secondary,#6b7280);fill-opacity:.18;stroke:var(--text-secondary,#6b7280);stroke-width:1.4}
.ts1-blk{fill:none;stroke:var(--text-primary,#1f2937);stroke-width:1.6}
.ts1-fill{fill:var(--text-secondary,#6b7280);fill-opacity:.22;transform-box:fill-box;transform-origin:left center}
.ts1-cell{fill:var(--background,#fff);stroke:var(--border,#d1d5db);stroke-width:1.4}
.ts1-g{fill:#16a34a;fill-opacity:.28;stroke:#16a34a;stroke-width:2}
.ts1-r{fill:#dc2626;fill-opacity:.28;stroke:#dc2626;stroke-width:2;opacity:0}
.ts1-a{fill:#d97706;fill-opacity:.3;stroke:#d97706;stroke-width:2}
.ts1-x{fill:var(--text-secondary,#6b7280);fill-opacity:.25;stroke:var(--text-secondary,#6b7280);stroke-width:1.4}
.ts1-strike{stroke:var(--text-secondary,#6b7280);stroke-width:2}
.ts1-tag{font:600 12px ui-sans-serif,system-ui;text-anchor:middle}
.ts1-cnt{font:700 17px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937);text-anchor:middle}
.ts1-axis{stroke:var(--text-secondary,#6b7280);stroke-width:1.2}
.ts1-on{opacity:0}
@keyframes ts1-d1{0%,2%{opacity:0}5%,93%{opacity:1}98%,100%{opacity:0}}
@keyframes ts1-d2{0%,6%{opacity:0}9%,93%{opacity:1}98%,100%{opacity:0}}
@keyframes ts1-d3{0%,10%{opacity:0}13%,93%{opacity:1}98%,100%{opacity:0}}
@keyframes ts1-d4{0%,14%{opacity:0}17%,93%{opacity:1}98%,100%{opacity:0}}
@keyframes ts1-d5{0%,18%{opacity:0}21%,93%{opacity:1}98%,100%{opacity:0}}
@keyframes ts1-ver{0%,24%{transform:scaleX(0)}42%,93%{transform:scaleX(1)}98%,100%{transform:scaleX(0)}}
@keyframes ts1-row{0%,42%{opacity:0}46%,93%{opacity:1}98%,100%{opacity:0}}
@keyframes ts1-onk{0%,42%{opacity:0}46%,64%{opacity:1}68%,100%{opacity:0}}
@keyframes ts1-g1{0%,48%{opacity:0}51%,93%{opacity:1}98%,100%{opacity:0}}
@keyframes ts1-g2{0%,51%{opacity:0}54%,93%{opacity:1}98%,100%{opacity:0}}
@keyframes ts1-g3{0%,54%{opacity:0}57%,93%{opacity:1}98%,100%{opacity:0}}
@keyframes ts1-rk{0%,57%{opacity:0}60%,64%{opacity:1}68%,100%{opacity:0}}
@keyframes ts1-xk{0%,60%{opacity:0}63%,93%{opacity:1}98%,100%{opacity:0}}
@keyframes ts1-ak{0%,64%{opacity:0}68%,93%{opacity:1}98%,100%{opacity:0}}
@keyframes ts1-ck{0%,72%{opacity:0}76%,93%{opacity:1}98%,100%{opacity:0}}
.ts1-p1{animation:ts1-d1 12s ease-in-out infinite}
.ts1-p2{animation:ts1-d2 12s ease-in-out infinite}
.ts1-p3{animation:ts1-d3 12s ease-in-out infinite}
.ts1-p4{animation:ts1-d4 12s ease-in-out infinite}
.ts1-p5{animation:ts1-d5 12s ease-in-out infinite}
.ts1-v{animation:ts1-ver 12s ease-in-out infinite}
.ts1-rw{animation:ts1-row 12s ease-in-out infinite}
.ts1-on{animation:ts1-onk 12s ease-in-out infinite}
.ts1-gg1{animation:ts1-g1 12s ease-in-out infinite}
.ts1-gg2{animation:ts1-g2 12s ease-in-out infinite}
.ts1-gg3{animation:ts1-g3 12s ease-in-out infinite}
.ts1-r{animation:ts1-rk 12s ease-in-out infinite}
.ts1-xx{animation:ts1-xk 12s ease-in-out infinite}
.ts1-aa{animation:ts1-ak 12s ease-in-out infinite}
.ts1-cc{animation:ts1-ck 12s ease-in-out infinite}
@media (prefers-reduced-motion:reduce){.ts1-p1,.ts1-p2,.ts1-p3,.ts1-p4,.ts1-p5,.ts1-v,.ts1-rw,.ts1-on,.ts1-gg1,.ts1-gg2,.ts1-gg3,.ts1-r,.ts1-xx,.ts1-aa,.ts1-cc{animation:none}}
</style>
<text class="ts1-lbl" x="20" y="82">drafter</text>
<text class="ts1-sub" x="20" y="100">(cheap)</text>
<text class="ts1-lbl" x="20" y="164">target</text>
<text class="ts1-sub" x="20" y="182">(expensive)</text>
<rect class="ts1-lane" x="160" y="55" width="600" height="60" rx="8"/>
<rect class="ts1-lane" x="160" y="137" width="600" height="60" rx="8"/>
<g class="ts1-p1"><rect class="ts1-pulse" x="172" y="67" width="60" height="36" rx="6"/><text class="ts1-tok" x="202" y="90">the</text></g>
<g class="ts1-p2"><rect class="ts1-pulse" x="242" y="67" width="60" height="36" rx="6"/><text class="ts1-tok" x="272" y="90">cat</text></g>
<g class="ts1-p3"><rect class="ts1-pulse" x="312" y="67" width="60" height="36" rx="6"/><text class="ts1-tok" x="342" y="90">sat</text></g>
<g class="ts1-p4"><rect class="ts1-pulse" x="382" y="67" width="60" height="36" rx="6"/><text class="ts1-tok" x="412" y="90">on</text></g>
<g class="ts1-p5"><rect class="ts1-pulse" x="452" y="67" width="60" height="36" rx="6"/><text class="ts1-tok" x="482" y="90">a</text></g>
<rect class="ts1-fill ts1-v" x="530" y="145" width="220" height="44" rx="6"/>
<rect class="ts1-blk" x="530" y="145" width="220" height="44" rx="6"/>
<text class="ts1-tok" x="640" y="172" style="font-size:12.5px">verify all 5 in one forward pass</text>
<line class="ts1-axis" x1="160" y1="214" x2="752" y2="214"/>
<path d="M752 209 L762 214 L752 219 Z" fill="var(--text-secondary,#6b7280)"/>
<text class="ts1-sub" x="160" y="230">one clock, time runs left to right</text>
<text class="ts1-lbl" x="20" y="266">output</text>
<g class="ts1-rw">
<rect class="ts1-cell" x="172" y="243" width="78" height="38" rx="6"/>
<rect class="ts1-cell" x="262" y="243" width="78" height="38" rx="6"/>
<rect class="ts1-cell" x="352" y="243" width="78" height="38" rx="6"/>
<rect class="ts1-cell" x="442" y="243" width="78" height="38" rx="6"/>
<rect class="ts1-cell" x="532" y="243" width="78" height="38" rx="6"/>
</g>
<rect class="ts1-g ts1-gg1" x="172" y="243" width="78" height="38" rx="6"/>
<rect class="ts1-g ts1-gg2" x="262" y="243" width="78" height="38" rx="6"/>
<rect class="ts1-g ts1-gg3" x="352" y="243" width="78" height="38" rx="6"/>
<rect class="ts1-r" x="442" y="243" width="78" height="38" rx="6"/>
<rect class="ts1-a ts1-aa" x="442" y="243" width="78" height="38" rx="6"/>
<rect class="ts1-x ts1-xx" x="532" y="243" width="78" height="38" rx="6"/>
<g class="ts1-rw">
<text class="ts1-tok" x="211" y="267">the</text>
<text class="ts1-tok" x="301" y="267">cat</text>
<text class="ts1-tok" x="391" y="267">sat</text>
<text class="ts1-tok" x="571" y="267">a</text>
</g>
<text class="ts1-tok ts1-on" x="481" y="267">on</text>
<text class="ts1-tok ts1-aa" x="481" y="267">upon</text>
<line class="ts1-strike ts1-xx" x1="556" y1="262" x2="586" y2="262"/>
<text class="ts1-tag ts1-gg3" x="301" y="300" fill="#16a34a">accepted</text>
<text class="ts1-tag ts1-aa" x="481" y="300" fill="#d97706">correction</text>
<text class="ts1-tag ts1-xx" x="571" y="300" fill="var(--text-secondary,#6b7280)">discarded</text>
<text class="ts1-cnt ts1-cc" x="390" y="330">tau = 4 tokens for 1 target pass</text>
</svg>
<figcaption>One round of speculative decoding: the drafter guesses five tokens, the target checks them in a single pass, keeps the matching prefix and adds one token of its own. Training a drafter means making that green run longer.</figcaption>
</figure>

The animation above is the mental model for everything that follows. The drafter guesses a few tokens, the target checks all of them in one forward pass, keeps the matching prefix and adds one token of its own. **Training a drafter has one goal: make the green run longer.** Every decision about data, loss and architecture is judged by whether it does that.

## 1. The drafter imitates one model, not language

Speculative decoding is lossless. Rejected tokens are resampled from a corrected distribution, so the output is provably the target's own distribution (the proof is in [token acceptance and rejection sampling](/blog/machine-learning/speculative-decoding/speculative-decoding-token-acceptance-rejection-sampling)). That changes how you approach training:

| Usual model training | Drafter training |
| --- | --- |
| Goal: be correct | Goal: agree with the target, including its mistakes |
| A bad model ships bad answers | A bad drafter ships correct answers, just slowly |
| Data quality means human-quality text | Data quality means *the target's* text |
| Measured by benchmark accuracy | Measured by tokens accepted per target pass |

The first row matters most. If the target always opens with "Certainly!", a drafter that has learned better manners loses a token on every response. You are not teaching the drafter to write well. You are teaching it to predict one specific model, quirks included.

## 2. The one number you are optimizing: acceptance length

Let $\tau$ be the expected number of tokens produced per target forward pass, bonus token included. If the drafter proposes $k$ tokens and each draft forward costs a fraction $c$ of a target forward, the speedup is roughly

$$
S \approx \frac{\tau}{1 + k\,c}.
$$

$\tau$ is not one accuracy number. Let $a_j$ be the probability that position $j$ is accepted given that positions $1$ to $j-1$ were. Then

$$
\tau = 1 + \sum_{k=1}^{K} \prod_{j=1}^{k} a_j .
$$

![Accepted length as a shrinking staircase of survival probabilities plus one bonus token](/imgs/blogs/training-speculative-decoding-drafters-playbook-2.webp)

The staircase shows why. With per-position acceptance of 0.80 falling to 0.68, the chance of surviving to slot 7 is only 0.12. Add up the bars and the bonus and you get $\tau = 3.80$. Two consequences follow:

- **Early slots dominate.** Raising slot 1 from 0.80 to 0.85 lifts every bar after it. Raising slot 7 by the same amount barely changes anything.
- **Top-1 accuracy on teacher-forced data is misleading.** It measures slot 1 with perfect context. At inference, slots 2 and later see the drafter's own guesses. A drafter can score 80% teacher-forced and still reach $\tau \approx 2$.

So the training metric to watch is accuracy *per draft depth*, and the evaluation metric is $\tau$ on your own traffic.

## 3. Data: regenerate with the target

The most common mistake costs more than any architecture choice: training on the dataset's original answers. ShareGPT and UltraChat answers were written by other models with other habits. A drafter trained on them learns a voice your target doesn't use.

![Two drafters on the same prompt: one trained on dataset answers is rejected at the first token, one trained on regenerated answers survives four tokens](/imgs/blogs/training-speculative-decoding-drafters-playbook-3.webp)

The fix is **self-distillation by regeneration**. Keep the prompts, throw away the answers, and have the target write new ones. Medusa proposed this for exactly this reason. EAGLE-3, SpecForge and DeepSpec all use it. DeepSpec trains its released drafters on open-perfectblend prompts with answers regenerated by each target in non-thinking mode. Its data stage requires an inference engine serving the target for this step.

The regeneration has to match production exactly:

- **Same chat template and system prompt.** A drafter that never saw your system prompt's style during training will mis-predict the opening of every response.
- **Same thinking mode.** A reasoning model in thinking mode writes long, repetitive chains that are easy to predict. Non-thinking answers are a different distribution. Train on the mode you serve, or on a mix in production proportions.
- **Same sampling settings.** Greedy regeneration gives the drafter a sharper target than temperature 0.7 does.
- **Same domain mix.** If 60% of your traffic is code, 60% of your prompts should be code.

Hidden-state drafters (EAGLE, DFlash, DSpark) need more than text. They also need the target's internal features for every training token.

### Offline versus online features

There are two ways to provide those features, and the tradeoff is storage against GPUs.

![Per-token feature cost by model size and total cache size for two corpora](/imgs/blogs/training-speculative-decoding-drafters-playbook-4.webp)

**Offline** means running the target once over the corpus, saving the hidden states to disk, and training against them. EAGLE-3 keeps three intermediate layers plus the final hidden state, which in bf16 is $8H$ bytes per token: 32 KB for Qwen3-8B, so about 128 MiB per 4,096-token sample. LMSYS quotes roughly 12 TB for UltraChat plus ShareGPT. DeepSpec warns its default Qwen3-4B target cache reaches about 38 TB. Offline is reproducible and needs few training GPUs. You pay in disk.

**Online** keeps the target running in a separate inference server that streams features to the trainer as it goes. SpecForge does this with a patched SGLang server, and the vLLM team's EAGLE-3 pipeline does the same with vLLM. Nothing touches the disk. You pay in GPUs held for the whole run. A practical rule: go offline for a single fine-tuning run on under a million samples, and online once the cache would outgrow your storage.

## 4. Algorithms as a ladder

Each generation of drafter fixed a specific training weakness in the one before it. Read the lineage as a ladder, not a menu.

![Matrix of drafter algorithms by what they read from the target, how they draft, and their key training idea](/imgs/blogs/training-speculative-decoding-drafters-playbook-5.webp)

- **Separate small LM.** A 0.5B model from the same family, distilled on target outputs. Easy to reason about, but it reads nothing from the target, and $c$ is large. Covered in [draft models](/blog/machine-learning/speculative-decoding/draft-models-for-speculative-decoding).
- **[Medusa](/blog/machine-learning/speculative-decoding/medusa-multi-head-speculative-decoding).** $K$ heads on the target's last hidden state, each guessing a different future offset. Cheap to train with a frozen backbone, but the heads guess independently, so slot 3 cannot see what slot 2 guessed.
- **[EAGLE-1 and EAGLE-2](/blog/machine-learning/speculative-decoding/eagle-speculative-decoding-feature-alignment).** One autoregressive decoder layer that reads the top hidden state plus the next token, trained with feature regression and a token loss. EAGLE-2 changed inference (dynamic trees), not training.
- **HASS.** Keeps the EAGLE architecture but trains the drafter on its *own* earlier features (context alignment) and distills the target's top-K distribution. This is the first serious fix for exposure bias.
- **EAGLE-3.** Fuses low, middle and high layers, drops the feature loss for a pure token loss, and trains with training-time test. Its key result is about scaling: earlier EAGLEs stopped improving with more data, while EAGLE-3 keeps improving.
- **MTP heads.** DeepSeek-V3 trains multi-token-prediction modules during pretraining and reuses them as a drafter. It is free if your target has them and unavailable if it doesn't.
- **[DFlash](/blog/paper-reading/large-language-model/dflash-block-diffusion-drafter-v1-v2) and [DSpark](/blog/paper-reading/large-language-model/dspark-confidence-scheduled-speculative-decoding).** Parallel drafters that fill a whole block of masked slots in one pass, so they can afford five layers instead of one. In DFlash's Qwen3-8B greedy setup at a 16-token budget, $\tau$ reaches 6.49 against EAGLE-3's 2.96. DSpark adds a tiny sequential head to stop later slots from drifting, and a 2-layer DSpark beats a 5-layer DFlash.

The default choice in late 2026 is EAGLE-3 for broad engine support. Pick DFlash or DSpark when your engine supports them and you run at low batch sizes, where the parallel block pays off most.

## 5. Training-time test: teaching the drafter to survive its own mistakes

At inference, an autoregressive drafter's second step reads the feature its first step produced, not the target's. If training only ever shows it ground-truth features, it has never practiced recovering from its own drift. This is exposure bias, and it shows up as the cliff after slot 1.

<figure class="blog-anim">
<svg viewBox="0 0 880 300" role="img" aria-label="Training-time test unrolled over four steps: at step 0 every feature input is the target's hidden state, shown blue; at each later step the token window shifts left one slot and more inputs at the drafting frontier become the drafter's own output, shown orange, until most are orange at step 3; a loss-weight chart highlights the current step's weight, 1.0, 0.8, 0.64 and 0.51" style="width:100%;height:auto;max-width:860px">
<style>
.ts6-cell{fill:var(--surface,#f3f4f6);stroke:var(--border,#d1d5db);stroke-width:1.4}
.ts6-tok{font:600 14px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937);text-anchor:middle}
.ts6-h{font:600 15px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937)}
.ts6-sub{font:500 13px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280)}
.ts6-val{font:600 13px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937)}
.ts6-link{stroke:var(--border,#d1d5db);stroke-width:1.4}
.ts6-b{fill:#3b82f6}
.ts6-o{fill:#f97316}
.ts6-bar{fill:var(--text-secondary,#6b7280);fill-opacity:.45}
.ts6-hl{fill:none;stroke:var(--accent,#6366f1);stroke-width:2.2;transform:translateY(150px)}
.ts6-s0,.ts6-s1,.ts6-s2{opacity:0}
@keyframes ts6-k0{0%,22%{opacity:1}25%,97%{opacity:0}100%{opacity:1}}
@keyframes ts6-k1{0%,22%{opacity:0;transform:translateX(80px)}25%,47%{opacity:1;transform:translateX(0)}50%,100%{opacity:0;transform:translateX(0)}}
@keyframes ts6-k2{0%,47%{opacity:0;transform:translateX(80px)}50%,72%{opacity:1;transform:translateX(0)}75%,100%{opacity:0;transform:translateX(0)}}
@keyframes ts6-k3{0%,72%{opacity:0;transform:translateX(80px)}75%,97%{opacity:1;transform:translateX(0)}100%{opacity:0;transform:translateX(0)}}
@keyframes ts6-mv{0%,23%{transform:translateY(0)}25%,48%{transform:translateY(50px)}50%,73%{transform:translateY(100px)}75%,98%{transform:translateY(150px)}100%{transform:translateY(0)}}
.ts6-s0{animation:ts6-k0 12s ease-in-out infinite}
.ts6-s1{animation:ts6-k1 12s ease-in-out infinite}
.ts6-s2{animation:ts6-k2 12s ease-in-out infinite}
.ts6-s3{animation:ts6-k3 12s ease-in-out infinite}
.ts6-hl{animation:ts6-mv 12s ease-in-out infinite}
@media (prefers-reduced-motion:reduce){.ts6-s0,.ts6-s1,.ts6-s2,.ts6-s3,.ts6-hl{animation:none}}
</style>
<text class="ts6-sub" x="20" y="103">tokens</text>
<text class="ts6-sub" x="20" y="168">features</text>
<g class="ts6-s ts6-s0">
<text class="ts6-h" x="110" y="58">step 0: every input is the target's hidden state</text>
<rect class="ts6-cell" x="110" y="78" width="70" height="40" rx="6"/><text class="ts6-tok" x="145" y="103">t1</text>
<line class="ts6-link" x1="145" y1="118" x2="145" y2="142"/><rect class="ts6-b" x="124" y="142" width="42" height="42" rx="5"/>
<rect class="ts6-cell" x="190" y="78" width="70" height="40" rx="6"/><text class="ts6-tok" x="225" y="103">t2</text>
<line class="ts6-link" x1="225" y1="118" x2="225" y2="142"/><rect class="ts6-b" x="204" y="142" width="42" height="42" rx="5"/>
<rect class="ts6-cell" x="270" y="78" width="70" height="40" rx="6"/><text class="ts6-tok" x="305" y="103">t3</text>
<line class="ts6-link" x1="305" y1="118" x2="305" y2="142"/><rect class="ts6-b" x="284" y="142" width="42" height="42" rx="5"/>
<rect class="ts6-cell" x="350" y="78" width="70" height="40" rx="6"/><text class="ts6-tok" x="385" y="103">t4</text>
<line class="ts6-link" x1="385" y1="118" x2="385" y2="142"/><rect class="ts6-b" x="364" y="142" width="42" height="42" rx="5"/>
<rect class="ts6-cell" x="430" y="78" width="70" height="40" rx="6"/><text class="ts6-tok" x="465" y="103">t5</text>
<line class="ts6-link" x1="465" y1="118" x2="465" y2="142"/><rect class="ts6-b" x="444" y="142" width="42" height="42" rx="5"/>
<rect class="ts6-cell" x="510" y="78" width="70" height="40" rx="6"/><text class="ts6-tok" x="545" y="103">t6</text>
<line class="ts6-link" x1="545" y1="118" x2="545" y2="142"/><rect class="ts6-b" x="524" y="142" width="42" height="42" rx="5"/>
</g>
<g class="ts6-s ts6-s1">
<text class="ts6-h" x="110" y="58">step 1: the frontier reads the drafter's output</text>
<rect class="ts6-cell" x="110" y="78" width="70" height="40" rx="6"/><text class="ts6-tok" x="145" y="103">t2</text>
<line class="ts6-link" x1="145" y1="118" x2="145" y2="142"/><rect class="ts6-b" x="124" y="142" width="42" height="42" rx="5"/>
<rect class="ts6-cell" x="190" y="78" width="70" height="40" rx="6"/><text class="ts6-tok" x="225" y="103">t3</text>
<line class="ts6-link" x1="225" y1="118" x2="225" y2="142"/><rect class="ts6-b" x="204" y="142" width="42" height="42" rx="5"/>
<rect class="ts6-cell" x="270" y="78" width="70" height="40" rx="6"/><text class="ts6-tok" x="305" y="103">t4</text>
<line class="ts6-link" x1="305" y1="118" x2="305" y2="142"/><rect class="ts6-b" x="284" y="142" width="42" height="42" rx="5"/>
<rect class="ts6-cell" x="350" y="78" width="70" height="40" rx="6"/><text class="ts6-tok" x="385" y="103">t5</text>
<line class="ts6-link" x1="385" y1="118" x2="385" y2="142"/><rect class="ts6-b" x="364" y="142" width="42" height="42" rx="5"/>
<rect class="ts6-cell" x="430" y="78" width="70" height="40" rx="6"/><text class="ts6-tok" x="465" y="103">t6</text>
<line class="ts6-link" x1="465" y1="118" x2="465" y2="142"/><rect class="ts6-o" x="444" y="142" width="42" height="42" rx="5"/>
<rect class="ts6-cell" x="510" y="78" width="70" height="40" rx="6"/><text class="ts6-tok" x="545" y="103">t7</text>
<line class="ts6-link" x1="545" y1="118" x2="545" y2="142"/><rect class="ts6-o" x="524" y="142" width="42" height="42" rx="5"/>
</g>
<g class="ts6-s ts6-s2">
<text class="ts6-h" x="110" y="58">step 2: drafter features spread back</text>
<rect class="ts6-cell" x="110" y="78" width="70" height="40" rx="6"/><text class="ts6-tok" x="145" y="103">t3</text>
<line class="ts6-link" x1="145" y1="118" x2="145" y2="142"/><rect class="ts6-b" x="124" y="142" width="42" height="42" rx="5"/>
<rect class="ts6-cell" x="190" y="78" width="70" height="40" rx="6"/><text class="ts6-tok" x="225" y="103">t4</text>
<line class="ts6-link" x1="225" y1="118" x2="225" y2="142"/><rect class="ts6-b" x="204" y="142" width="42" height="42" rx="5"/>
<rect class="ts6-cell" x="270" y="78" width="70" height="40" rx="6"/><text class="ts6-tok" x="305" y="103">t5</text>
<line class="ts6-link" x1="305" y1="118" x2="305" y2="142"/><rect class="ts6-o" x="284" y="142" width="42" height="42" rx="5"/>
<rect class="ts6-cell" x="350" y="78" width="70" height="40" rx="6"/><text class="ts6-tok" x="385" y="103">t6</text>
<line class="ts6-link" x1="385" y1="118" x2="385" y2="142"/><rect class="ts6-o" x="364" y="142" width="42" height="42" rx="5"/>
<rect class="ts6-cell" x="430" y="78" width="70" height="40" rx="6"/><text class="ts6-tok" x="465" y="103">t7</text>
<line class="ts6-link" x1="465" y1="118" x2="465" y2="142"/><rect class="ts6-o" x="444" y="142" width="42" height="42" rx="5"/>
<rect class="ts6-cell" x="510" y="78" width="70" height="40" rx="6"/><text class="ts6-tok" x="545" y="103">t8</text>
<line class="ts6-link" x1="545" y1="118" x2="545" y2="142"/><rect class="ts6-o" x="524" y="142" width="42" height="42" rx="5"/>
</g>
<g class="ts6-s ts6-s3">
<text class="ts6-h" x="110" y="58">step 3: almost every input is drafter-made</text>
<rect class="ts6-cell" x="110" y="78" width="70" height="40" rx="6"/><text class="ts6-tok" x="145" y="103">t4</text>
<line class="ts6-link" x1="145" y1="118" x2="145" y2="142"/><rect class="ts6-b" x="124" y="142" width="42" height="42" rx="5"/>
<rect class="ts6-cell" x="190" y="78" width="70" height="40" rx="6"/><text class="ts6-tok" x="225" y="103">t5</text>
<line class="ts6-link" x1="225" y1="118" x2="225" y2="142"/><rect class="ts6-o" x="204" y="142" width="42" height="42" rx="5"/>
<rect class="ts6-cell" x="270" y="78" width="70" height="40" rx="6"/><text class="ts6-tok" x="305" y="103">t6</text>
<line class="ts6-link" x1="305" y1="118" x2="305" y2="142"/><rect class="ts6-o" x="284" y="142" width="42" height="42" rx="5"/>
<rect class="ts6-cell" x="350" y="78" width="70" height="40" rx="6"/><text class="ts6-tok" x="385" y="103">t7</text>
<line class="ts6-link" x1="385" y1="118" x2="385" y2="142"/><rect class="ts6-o" x="364" y="142" width="42" height="42" rx="5"/>
<rect class="ts6-cell" x="430" y="78" width="70" height="40" rx="6"/><text class="ts6-tok" x="465" y="103">t8</text>
<line class="ts6-link" x1="465" y1="118" x2="465" y2="142"/><rect class="ts6-o" x="444" y="142" width="42" height="42" rx="5"/>
<rect class="ts6-cell" x="510" y="78" width="70" height="40" rx="6"/><text class="ts6-tok" x="545" y="103">t9</text>
<line class="ts6-link" x1="545" y1="118" x2="545" y2="142"/><rect class="ts6-o" x="524" y="142" width="42" height="42" rx="5"/>
</g>
<rect class="ts6-b" x="110" y="214" width="18" height="18" rx="3"/><text class="ts6-sub" x="136" y="228">target hidden state (ground truth)</text>
<rect class="ts6-o" x="370" y="214" width="18" height="18" rx="3"/><text class="ts6-sub" x="396" y="228">drafter's own output</text>
<text class="ts6-sub" x="110" y="266">each step shifts the window left one slot</text>
<line x1="590" y1="40" x2="590" y2="290" stroke="var(--border,#d1d5db)" stroke-width="1"/>
<text class="ts6-h" x="620" y="72">loss weight 0.8^k</text>
<text class="ts6-sub" x="620" y="115">step 0</text><rect class="ts6-bar" x="680" y="96" width="140.0" height="28" rx="4"/><text class="ts6-val" x="828.0" y="115">1.00</text>
<text class="ts6-sub" x="620" y="165">step 1</text><rect class="ts6-bar" x="680" y="146" width="112.0" height="28" rx="4"/><text class="ts6-val" x="800.0" y="165">0.80</text>
<text class="ts6-sub" x="620" y="215">step 2</text><rect class="ts6-bar" x="680" y="196" width="89.6" height="28" rx="4"/><text class="ts6-val" x="777.6" y="215">0.64</text>
<text class="ts6-sub" x="620" y="265">step 3</text><rect class="ts6-bar" x="680" y="246" width="71.4" height="28" rx="4"/><text class="ts6-val" x="759.4" y="265">0.51</text>
<rect class="ts6-hl" x="612" y="90" width="256" height="40" rx="6"/>
</svg>
<figcaption>Training-time test unrolls the drafter during training: from step 1 on it reads its own features instead of the target's, exactly as it will at inference, and each deeper step enters the loss with weight 0.8^k.</figcaption>
</figure>

Training-time test (TTT) unrolls the drafter during training. SpecForge's default is `ttt_length = 7`. Step 0 reads real target features. From step 1 on, the drafter's own output becomes the next input, and each deeper step's loss is weighted by $0.8^k$. Deeper slots get less weight because, as the staircase showed, they contribute less to $\tau$.

Two loss choices matter here:

| Choice | Option A | Option B | Recommendation |
| --- | --- | --- | --- |
| Target signal | Hard label (target's sampled token) | Soft KL to the target distribution | KL: it carries the target's uncertainty, and acceptance under sampling depends on the whole distribution |
| Vocabulary | Full target vocab (151,936 for Qwen) | Trimmed draft vocab (32k) | Trim, but build the 32k from *your* domain's token frequencies |

The vocab trim has a catch. A position whose true token is not in the draft vocabulary is dropped from the loss, because the drafter cannot emit it. A vocab built from chat data and deployed on code silently drops more supervision and accepts fewer tokens.

## 6. Evaluation that predicts production

A drafter evaluation has two layers. The first is model-level: $\tau$ and per-position acceptance, measured per domain at your deployment temperature. DeepSpec's evaluation runs nine benchmarks (GSM8K, MATH500, AIME25, HumanEval, MBPP, LiveCodeBench, MT-Bench, Alpaca and Arena-Hard-v2). That spread is the honest approach, because $\tau$ on math is routinely about twice $\tau$ on open chat. The second layer is system-level: wall-clock tokens per second at the batch sizes you actually run, because speculation's gain shrinks as batches grow (see [speculative decoding in production](/blog/machine-learning/model-serving/speculative-decoding-in-production)).

This script turns per-round verification logs into both numbers:

```python
# spec_eval.py: tau and per-position conditional acceptance from verify logs.
# Each JSONL line: {"domain": "code", "proposed": 7, "accepted": 3}
# ("accepted" counts draft tokens kept, excluding the bonus token.)
import json
import sys
from collections import defaultdict

def summarize(path: str, max_k: int = 8) -> None:
    rounds = defaultdict(list)
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            rounds[r["domain"]].append((r["proposed"], r["accepted"]))

    for domain, rs in sorted(rounds.items()):
        tau = sum(a + 1 for _, a in rs) / len(rs)
        print(f"{domain:>8}  rounds={len(rs):6d}  tau={tau:.2f}")
        # conditional acceptance at slot k: among rounds that reached slot k
        for k in range(1, max_k + 1):
            reached = [(p, a) for p, a in rs if p >= k and a >= k - 1]
            if not reached:
                break
            acc = sum(1 for _, a in reached if a >= k) / len(reached)
            bar = "#" * int(acc * 40)
            print(f"          slot {k}: {acc:.2f} {bar}")

if __name__ == "__main__":
    summarize(sys.argv[1])
```

Run it on logs from your real traffic, not just the benchmark suite, and plot the per-slot column. The shape of that curve is the best diagnostic you have.

## 7. Best practices and the failure modes that cost you tau

Every failure below leaves the output correct, so none of them raises an error. They just make you slower than you should be. The per-position curve is where they show up.

![Per-position acceptance curves for a healthy drafter and three common failure modes](/imgs/blogs/training-speculative-decoding-drafters-playbook-7.webp)

Read the shape, not the average. A cliff after slot 1 means exposure bias. A low, flat line from slot 1 means something is structurally broken. A healthy shape shifted down means the data is wrong.

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| $\tau$ near 1.2, low from slot 1 | Tokenizer or chat-template mismatch; wrong auxiliary layer ids in the exported config | Diff tokenizer files byte for byte; regenerate with the serving template; check the export config against training |
| Slot 1 fine, cliff after | No TTT, or `ttt_length` too short | Enable TTT; check per-depth training accuracy rises at every depth |
| Great on eval, poor in prod | Eval greedy, prod sampled; or domain shift | Evaluate at deployment temperature on logged traffic |
| High $\tau$, no wall-clock gain | Batch too large, or drafter cost $c$ too high | Measure at production batch; shorten $k$ under load |
| $\tau$ dropped after a target update | Drafter trained on the old target's features | Version-lock drafter to target; re-fine-tune on every target release |
| Low loss, low acceptance | Loss on prompt tokens, or trained on dataset answers | Mask prompt tokens; regenerate answers |

Best practices that prevent most of them:

1. **Start from a public drafter and fine-tune.** A few tens of thousands of regenerated in-domain samples usually beats training from scratch on a million generic ones.
2. **Copy the target's embeddings and LM head and freeze them.** The drafter should spend its capacity on prediction, not on relearning the vocabulary.
3. **Train on production sequence lengths.** A drafter trained on 2k-token samples has never seen the position range of a 32k-token RAG prompt.
4. **Watch accuracy per TTT depth during training.** If depth 0 keeps climbing while depth 3 stays flat, the unroll is broken.
5. **Treat export as a test boundary.** Load the exported checkpoint into the serving engine and measure $\tau$ there before calling the run done. A wrong field in the config fails silently.
6. **Re-measure after quantizing the target.** FP8 or FP4 targets produce slightly different hidden states than the BF16 model the drafter was trained on.

## 8. Frameworks: SpecForge, DeepSpec and friends

| Framework | From | Algorithms | Serving engine | Pick it when |
| --- | --- | --- | --- | --- |
| [SpecForge](/blog/machine-learning/open-source-library/specforge-speculative-decoding-training-deep-dive) | SGLang team | EAGLE-3, DFlash, P-EAGLE and more | SGLang | You serve on SGLang and want online or offline training at scale |
| DeepSpec | DeepSeek | DSpark, DFlash, EAGLE-3 | Engine of your choice for regeneration | You want the full three-stage recipe (data, train, eval) with nine-benchmark eval; built on SpecForge |
| vLLM EAGLE-3 pipeline | vLLM team | EAGLE-3 | vLLM | You serve on vLLM; target generates on-policy data and evaluates acceptance during training |
| NeMo RL / AutoModel | NVIDIA | EAGLE-3 (Llama, Phi-3, Qwen3 dense) | vLLM, Megatron | You already train in the NeMo stack; supports online draft training alongside RL |
| SafeAILab EAGLE | EAGLE authors | EAGLE-1/2/3 | Reference code | You need the reference implementation to reproduce the paper |

DeepSpec's layout is the clearest template for any of them: **data preparation** (download prompts, regenerate answers with the target, build the target cache), **training** (one worker per GPU, configs that assume one 8-GPU node), and **evaluation** (acceptance across benchmarks). It ships drafters for Qwen3-4B, 8B and 14B and Gemma-4-12B for every algorithm, so you can compare DSpark, DFlash and EAGLE-3 on the same target before committing.

## Case studies

**1. EAGLE-3 and the data ceiling.** The EAGLE-3 authors found that EAGLE-1 and EAGLE-2 stopped improving when given more training data. Forcing the drafter to regress the target's top-layer features was the bottleneck. With the feature loss gone, fused multi-layer inputs and TTT, the drafter kept improving with more data and reported speedups of up to 6.5x. That one result is what made industrial training frameworks worth building: a drafter that rewards more data justifies a pipeline.

**2. DeepSeek-V3's free drafter.** DeepSeek-V3 trained a multi-token-prediction module as an auxiliary pretraining objective. At inference it reused that module as the drafter, reporting 85–90% acceptance on the second token and about 1.8x tokens per second. The lesson: the cheapest drafter is the one trained alongside the target. If you control pretraining, MTP is the highest-leverage option.

**3. A community EAGLE-3 on 20K samples.** A public EAGLE-3 drafter for MiniMax-M2.5 (a single decoder layer reading three auxiliary layers, trained with the TTT loss) was fine-tuned on just 20K samples regenerated by the target. That is the realistic scale for a team adapting a drafter to its own model. The data had to be regenerated, but it did not have to be large.

## When to train your own drafter, and when not to

Train or fine-tune when a public drafter's $\tau$ on your logged traffic falls well below its published number, when you serve a fine-tuned target nobody else has a drafter for, or when your traffic is concentrated in one domain such as code, SQL or a fixed agent format. In those cases a few days of regeneration and training routinely lifts $\tau$ by a token or more per round.

Skip it when you run at large batch sizes where speculation barely helps, when your target changes every week and you cannot re-fine-tune on that schedule, or when a public drafter already gets within 10% of its published $\tau$ on your traffic. In every case, measure first with the script above. The per-position curve tells you whether the problem is your data, your training loop or your deployment, and each needs a different fix.
