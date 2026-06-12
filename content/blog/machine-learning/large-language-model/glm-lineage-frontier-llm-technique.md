---
title: "The GLM Lineage: Five Years of Frontier-LLM Technique, From Blank Infilling to Agentic RL"
date: "2026-06-10"
publishDate: "2026-06-10"
description: "A senior-engineer read of every GLM paper and technical report, from the 2021 blank-infilling objective through GLM-130B's stability tricks to GLM-4.5's deep-narrow MoE and agentic RL — extracting the architecture, finetuning, engineering, and data techniques worth stealing."
tags:
  [
    "glm",
    "glm-130b",
    "glm-4",
    "glm-4.5",
    "large-language-model",
    "mixture-of-experts",
    "reinforcement-learning",
    "agentic",
    "pretraining",
    "fine-tuning",
    "quantization",
    "training-stability",
  ]
category: "machine-learning"
subcategory: "Large Language Model"
author: "Hiep Tran"
featured: true
readTime: 51
---

Most people meet GLM the way I did: as "the Chinese model that's suddenly near the top of the agentic-coding leaderboards." That framing throws away the most interesting thing about it. GLM is not a 2025 model. It is a five-year, publicly documented research program — one of the very few frontier-scale lineages where almost every decision is written down, from a 2021 ACL paper about a weird pretraining objective to a 2025 technical report that casually describes training a 355-billion-parameter Mixture-of-Experts with the Muon optimizer and an asynchronous RL stack. Read end to end, the GLM reports are something rarer than a good model: they are a *curriculum* in how frontier LLMs are actually built.

This post is the opening article of a series that reads the entire GLM body of work and extracts the reusable techniques — the things you could lift into your own training run, serving stack, or alignment pipeline tomorrow. This first article is the map: the whole lineage, the four axes the techniques sort into, and the through-lines that connect a 2021 blank-infilling trick to a 2025 agentic-RL recipe. Later articles drill into each stage.

This is a long read because the material earns it. If you want the single most important idea up front, here it is: **GLM is an accreting stack of techniques, not a sequence of rewrites.** Each release keeps the load-bearing parts of the last one and adds exactly one decisive new layer. Understanding *which* layer got added *when*, and *why the old layers survived*, is most of what there is to learn.

## Why "just another GPT clone" is the wrong mental model

Before the map, let's kill the lazy framing. The instinct is to slot GLM into the "open Chinese GPT" bucket and move on. The reports tell a different story on every axis.

| Assumption | The naive view | The reality in the reports |
| --- | --- | --- |
| "It's a decoder-only GPT" | Same architecture as everyone else | The 2021 core is **autoregressive blank infilling** — a hybrid of BERT-style bidirectional encoding and GPT-style autoregressive decoding in one transformer, with a custom 2D positional scheme |
| "Scale was the whole game" | Bigger model, more data | GLM-130B's headline contribution is **training stability on FP16**, not scale — they explicitly call stability "the decisive factor" for 100B-class success |
| "Quantization is a deployment afterthought" | Quantize at the end if you must | GLM-130B was the **first 100B model to land INT4 with ~0 quality loss**, designed so a 130B model runs inference on 4× RTX 3090 |
| "Alignment = SFT then RLHF, same as Llama" | Standard post-training | GLM-4.5 trains **three separate expert models** (reasoning, agent, chat) and **distills them into one hybrid model** that toggles thinking on and off |
| "MoE means copy DeepSeek-V3" | More experts, wider model | GLM-4.5 went **deeper and narrower** with *fewer* experts but **2.5× more attention heads per width**, because deeper-narrower models reason better at equal loss |

Every row is a technique we'll unpack. The point of the table is that the dismissive framing fails immediately: GLM made deliberate, non-obvious, and frequently contrarian calls at the architecture, engineering, alignment, and data layers — and wrote down the reasoning. That is exactly the material a staff-level engineer should be mining.

> If a lab tells you *what* they built, you get a model. If they tell you *why* each piece survived three generations of rewrites, you get a methodology. GLM is unusually generous with the second kind of information.

## The mental model: a lineage that accretes

![The GLM lineage from 2021 to 2025, where each release keeps the blank-infilling backbone and adds one new technique layer](/imgs/blogs/glm-lineage-frontier-llm-technique-1.png)

The diagram above is the mental model for the whole series: a timeline where each node is a release and each label is the *one decisive technique* it added. Read it left to right and the structure jumps out — this is not six independent models, it is one backbone with six layers of accretion.

- **2021 — GLM** ([arXiv:2103.10360](https://arxiv.org/abs/2103.10360)) introduces *autoregressive blank infilling* and 2D positional encoding. This is the genome. Everything downstream is still, technically, a blank-infilling model.
- **2022 — GLM-130B** ([arXiv:2210.02414](https://arxiv.org/abs/2210.02414)) scales the genome to 130B bilingual parameters and contributes the *engineering* layer: DeepNorm, embedding-gradient-shrink, FP32 softmax, and INT4 inference on commodity GPUs.
- **2023 — ChatGLM-6B → ChatGLM2 → ChatGLM3** adds the *product* layer: multi-query then grouped-query attention, 32K context, and the function-calling prompt format that makes tool use possible.
- **2024 — GLM-4 (All Tools)** ([arXiv:2406.12793](https://arxiv.org/abs/2406.12793)) adds the *alignment + agentic* layer: a real SFT→RLHF pipeline, 128K–1M context, and autonomous multi-tool use.
- **2025 H1 — GLM-4.5** ([arXiv:2508.06471](https://arxiv.org/abs/2508.06471)) adds the *frontier* layer: a 355B deep-narrow MoE, expert-model distillation, and large-scale agentic RL on verifiable tasks.
- **2025 H2 — GLM-4.6** extends context to 200K, adds FP8 RL rollouts, and squeezes ~30% more output efficiency.

The accretion is the lesson. When GLM-4.5 needs a stable optimizer it does not re-derive stability from scratch — it inherits the hard-won FP16 lessons from 2022 and bolts QK-Norm and Muon on top. When it needs cheap RL rollouts, it inherits the "narrow weight distributions quantize beautifully" insight from GLM-130B's INT4 work and applies it to FP8. A frontier model is, in practice, a *pile of surviving decisions*. GLM is one of the clearest case studies of which decisions survive.

It's worth being precise about what "accretion" buys you, because it's not just a tidy narrative — it's a risk-management strategy. Every frontier run is a bet placed months before you see the result, and the most expensive way to lose is to change three things at once and not know which one broke. By holding the objective and the data pipeline nearly constant and changing *one* major axis per generation — engineering in 2022, alignment in 2024, MoE architecture in 2025 — the GLM team kept each release's failures *attributable*. When GLM-130B's loss spiked, they knew it wasn't the objective (that was three years old); it had to be the new FP16 engineering. That attributability is why the reports can be so specific about cause and effect, and it's a discipline you can copy at any scale: change one load-bearing thing per run, and keep the rest boring on purpose.

A useful comparison point throughout this series is the [Kimi K2 report](/blog/paper-reading/large-language-model/kimi-k2-open-agentic-intelligence): both are ultra-sparse MoE lineages obsessed with stable optimization and agentic data, and the two teams made revealingly *different* calls (MuonClip + QK-Clip vs. plain Muon + QK-Norm; 384 experts vs. 160). Reading them side by side is the fastest way to separate "load-bearing technique" from "one team's taste."

## The four axes the techniques sort into

![A taxonomy tree sorting every GLM technique into architecture, alignment, engineering, and data and evaluation](/imgs/blogs/glm-lineage-frontier-llm-technique-2.png)

To extract techniques systematically rather than as a grab-bag, the rest of this article organizes them on four axes, shown in the tree above. Every method in the GLM corpus lands cleanly in one of them:

1. **Architecture** — the objective, attention, normalization, positional encoding, and MoE design. *What the network is.*
2. **Finetuning & alignment** — SFT data philosophy, RLHF, expert distillation, agentic RL. *How it learns to be useful.*
3. **Engineering & infra** — training stability, mixed precision, parallelism, quantization, RL infrastructure. *How it survives contact with real hardware.*
4. **Data & evaluation** — corpus construction, deduplication, quality filtering, curriculum, and the benchmarks the team built to measure itself. *What it learns from and how they know it worked.*

These are not academic categories. They are the four places a frontier training run can fail, and the GLM reports have a distinct, quotable opinion in each. The four sections below take them in turn; a final synthesis section lays all four across all four generations so you can see what persisted and what got reinvented.

## 1. Architecture: from blank infilling to deep-narrow MoE

> **Senior rule of thumb:** the objective you pretrain on is the most expensive decision you will ever make, because every downstream technique inherits its assumptions. GLM's entire lineage is shaped by a 2021 bet that you should not have to choose between understanding and generation.

### Autoregressive blank infilling: one transformer, two reading modes

In 2021 the field had three pretraining religions: autoencoding (BERT — great at understanding, can't generate), autoregressive (GPT — great at generation, weak bidirectional context), and encoder-decoder (T5 — powerful but doubles your parameters). GLM's founding move was to refuse the choice. The objective is **autoregressive blank infilling**: take an input, mask out several spans, and train the model to reconstruct the masked spans autoregressively — but let the model read the *unmasked* context bidirectionally.

Mechanically, the sequence is split into two parts. **Part A** is the corrupted text with each masked span collapsed to a single `[MASK]` token. **Part B** is the masked spans themselves, in *shuffled* order, each wrapped in `[START]`/`[END]` sentinels. The trick that makes it one model instead of two is the attention mask.

![The blank-infilling hybrid attention mask, where Part A is fully bidirectional and Part B decodes autoregressively](/imgs/blogs/glm-lineage-frontier-llm-technique-3.png)

The matrix above is the whole mechanism on a six-token example. Read a row as "what this query token is allowed to attend to":

- **Top-left 3×3 block (all green):** Part A tokens attend to *all* of Part A. This is BERT-style bidirectional encoding.
- **Top-right 3×3 block (all masked):** Part A cannot see Part B. The context never peeks at the answer.
- **Bottom-left 3×3 block (all green):** every Part B token attends to *all* of Part A. The decoder gets the full bidirectional context for free.
- **Bottom-right triangle (lower-triangular green):** within Part B, token *i* attends only to tokens ≤ *i*. This is GPT-style autoregressive decoding.

One transformer, one forward pass, and the upper-left quadrant is an encoder while the lower-right quadrant is a decoder. That is the genome, and it is genuinely elegant. Here is the span-sampling and Part-A/Part-B construction in runnable form:

```python
import numpy as np

def sample_glm_spans(seq_len, mask_ratio=0.15, lam=3, rng=None):
    """GLM blank-infilling corruption: sample Poisson(lam) spans until
    mask_ratio of tokens are covered. Returns span (start, length) pairs."""
    rng = rng or np.random.default_rng(0)
    target = int(round(mask_ratio * seq_len))
    masked, spans, taken = 0, [], np.zeros(seq_len, dtype=bool)
    while masked < target:
        length = max(1, rng.poisson(lam))           # span length ~ Poisson(3)
        start = int(rng.integers(0, seq_len))        # random start
        if taken[start:start + length].any():        # no overlap
            continue
        spans.append((start, min(length, seq_len - start)))
        taken[start:start + length] = True
        masked = int(taken.sum())
    return sorted(spans)

def build_parts(tokens, spans, rng=None):
    """Part A = corrupted text with one [MASK] per span.
    Part B = the spans themselves, SHUFFLED, each as [S] ...span... [E]."""
    rng = rng or np.random.default_rng(0)
    span_set = {s for (s, l) in spans for s in range(s, s + l)}
    part_a, b_spans = [], []
    i = 0
    while i < len(tokens):
        hit = next((sp for sp in spans if sp[0] == i), None)
        if hit:
            part_a.append("[MASK]")
            b_spans.append(["[S]"] + tokens[i:i + hit[1]] + ["[E]"])
            i += hit[1]
        elif i in span_set:
            i += 1
        else:
            part_a.append(tokens[i]); i += 1
    rng.shuffle(b_spans)                              # span order is randomized
    return part_a, [t for span in b_spans for t in span]
```

The attention mask is the other half of the implementation, and it is the part worth getting exactly right — a single wrong cell silently turns your encoder into a leaky decoder that cheats by reading the answer:

```python
import numpy as np

def blank_infilling_mask(len_a, b_span_lens):
    """len_a = number of Part-A tokens (including the [MASK]s).
    b_span_lens = list of Part-B span lengths (each incl [S] and [E]).
    Returns a boolean (q, k) matrix: True = query may attend to key."""
    n = len_a + sum(b_span_lens)
    m = np.zeros((n, n), dtype=bool)
    m[:len_a, :len_a] = True                  # Part A: fully bidirectional
    off = len_a
    for L in b_span_lens:                      # each Part-B span...
        m[off:off + L, :len_a] = True          #   sees ALL of Part A
        for i in range(L):                     #   but is autoregressive in itself
            m[off + i, off:off + i + 1] = True
        off += L
    return m                                    # top-right A->B block stays False
```

That function produces precisely the figure-3 pattern: the top-left `len_a × len_a` block is all-True (Part A reads itself bidirectionally), the top-right block stays False (Part A never peeks at the answer), every Part-B span sees all of Part A, and within each span the mask is lower-triangular. An encoder and a decoder, in one boolean matrix — and the whole reason GLM never had to choose between BERT and GPT.

The **span shuffling** matters more than it looks: because Part B's spans appear in random order, the model is forced to learn dependencies between blanks regardless of position, which is what gives GLM its surprisingly strong infilling behavior. And because the model must *generate* the `[END]` token to terminate a span, it never knows the span length in advance — making this a true generative objective, not a fixed-length cloze.

The payoff was immediate and measurable: GLM-Large scored **77.0 on SuperGLUE versus BERT-Large's 72.0** at the same parameter count and data, while *also* being able to do abstractive summarization (a thing BERT structurally cannot do). One objective, both capability classes.

The other half of the objective's power is that GLM doesn't run *one* blank-infilling regime — it mixes several by varying the span statistics, and that mixture is a tunable knob. The 2021 paper trains a **document-level** regime (sample a single long span covering 50–100% of the text, which teaches free-form generation) alongside a **sentence-level** regime (mask complete sentences to ~15% coverage, which teaches seq2seq tasks like summarization), on top of the default short-span regime. GLM-130B compressed this into two mask tokens you'll see all over the codebase: `[MASK]` for short Poisson spans, applied to 30% of training sequences, and `[gMASK]` for a long generative suffix, applied to the other 70%. The split is itself a design choice — tilt toward `[gMASK]` and the model leans generative, tilt toward `[MASK]` and it leans toward understanding. This is the earliest clean instance of a theme the lineage repeats at every scale: the pretraining objective is a *mixture*, and the mixture weights are a lever you set deliberately for the capability profile you want, not a fixed recipe you inherit.

### 2D positional encoding: how the model stays blind to span length

The construction above raises a question: if a `[MASK]` in Part A expands into a variable-length span in Part B, how do positions line up? GLM's answer is **2D positional encoding** — every token carries *two* position ids:

- **Position 1 (inter-position):** the token's index in the *corrupted* Part A. Every token of a reconstructed span shares the position id of the `[MASK]` it fills.
- **Position 2 (intra-position):** 0 for all of Part A; for Part B, it runs 1…span-length within each span.

The model literally cannot infer how long a span will be from the positional signal, which is exactly the property you want for open-ended generation. This is the part of the genome that got *replaced* rather than preserved — by GLM-4 the team had moved to 2D-RoPE and standard rotary schemes — but the conceptual descendant is alive and well: the `[gMASK]` "generate a long suffix" objective that GLM-130B uses for 70% of its training is a direct generalization of the document-level blank-infilling regime.

### Scaling the genome: GLM-130B's architecture choices

When the objective scaled to 130B, the architecture around it had to harden. The GLM-130B configuration is worth memorizing as a reference point for "what a stable dense 100B looks like":

| Spec | GLM-130B (2022) |
| --- | --- |
| Parameters | 130B dense, bilingual EN+ZH |
| Layers / hidden / heads | 70 (a "9×8−2" pipeline-balanced split) / 12,288 / 96 |
| Sequence length / vocab | 2,048 / 150,000 |
| FFN | GLU with GeLU (GeGLU) |
| Normalization | Post-LN + DeepNorm, α = (2N)^½ |
| Positional encoding | RoPE |
| Objective | `[MASK]` short spans (30% of data) + `[gMASK]` long suffix (70%) |

Two of these are load-bearing and underappreciated. **GeGLU** (a gated GLU variant of the FFN) and **RoPE** both became near-universal afterward, but GLM-130B was an early large-scale adopter. And the **70 = 9×8−2 layer count** is a tell: the depth was chosen to balance an 8-stage pipeline (9 layers per stage, minus 2 to account for the word-embedding ends). That is an engineering constraint leaking into an architecture number — a recurring theme we'll see again with GLM-4.5's deep-narrow choice.

### The deep-narrow MoE turn: more heads, fewer experts

By GLM-4.5 the architecture had crossed into sparse Mixture-of-Experts, and this is where the team made its most quotable contrarian call. The obvious move in 2025 was to copy DeepSeek-V3: very wide, 256 routed experts, 128 attention heads. GLM-4.5 did almost the opposite.

![Before-after comparison of dense GLM-130B versus the deep-narrow MoE design of GLM-4.5](/imgs/blogs/glm-lineage-frontier-llm-technique-4.png)

The before/after above contrasts the dense 2022 block with the 2025 sparse design. The headline numbers: GLM-4.5 is **355B total / 32B activated**, with **89 MoE layers**, hidden dim **5,120**, **160 routed experts** (top-8) plus 1 shared expert, and — the key choice — **96 attention heads on a 5,120-wide model**. That is roughly **2.5× the heads-per-width** of comparable models. The report is blunt about why, and the finding is the kind of thing you only learn by running the ablation:

> Increasing the head count did *not* improve training loss, but it *consistently* improved performance on reasoning benchmarks such as MMLU and BBH.

Read that twice. Training loss — the thing you stare at all day — was flat. The benefit showed up only downstream, on reasoning. If you had been optimizing for loss curves you would have removed the extra heads. The same philosophy drives the depth: "we reduce the width and number of routed experts and increase the number of layers, as we found deeper models exhibit better reasoning capacity." Fewer experts (160 vs. DeepSeek-V3's 256), narrower hidden dim, but deeper and head-richer. This is a *reasoning-first* architecture, not a loss-first one.

The supporting cast of architecture techniques is worth a checklist, because each is independently stealable:

- **Loss-free balance routing.** Instead of an auxiliary load-balancing loss that fights the main objective, GLM-4.5 uses a per-expert bias that is nudged to equalize load (bias update rate 0.001 for the first 15T tokens, then frozen to 0), with a tiny sequence-level balance loss (weight 1e-4) as a safety net. This is the aux-loss-free approach popularized by DeepSeek, and it matters because the old auxiliary loss measurably degraded quality.
- **QK-Norm** normalizes the query and key projections to keep attention logits in a sane range — the modern, cleaner descendant of GLM-130B's FP32-softmax hack (more on that below). Notably, GLM-4.5 uses it but GLM-4.5-Air does not, a reminder that these are *configuration* choices, not laws.
- **MTP (Multi-Token Prediction) layer.** One extra MoE layer is repurposed to predict multiple future tokens, which enables speculative decoding at inference time — an architecture decision made for *serving*, baked in at pretraining.
- **Partial RoPE + GQA** with only 8 KV heads, keeping the KV cache small for long context — the same [KV-cache](/blog/machine-learning/large-language-model/kv-cache) economics every serving engineer fights.

The loss-free routing is the one most teams get wrong, so it's worth seeing in code. The trick is that the balancing signal is a *controller*, not a loss term:

```python
def update_expert_bias(load_counts, bias, target_load, lr=1e-3):
    """Aux-loss-free load balancing (GLM-4.5): nudge each expert's routing
    bias toward equal load instead of adding a balance term to the objective.
    The bias shifts gating scores BEFORE top-k selection, but is NOT used to
    weight the expert's output — so it steers routing without distorting the
    computation or fighting the language-modeling gradient."""
    err = target_load - load_counts        # under-loaded experts get a positive nudge
    bias = bias + lr * np.sign(err)        # sign-based: lr=1e-3 first 15T tokens, then 0
    return bias
```

Because the update uses only `sign(err)` and is applied to the gating *score* rather than the output weight, it never appears in the backward pass of the main loss. That is the entire difference between aux-loss-free balancing and the older auxiliary-loss approach that measurably degraded quality: one is a thermostat sitting next to the network, the other is a second objective pulling against the first.

It's worth dwelling on how *different* this makes GLM-4.5 from the DeepSeek-V3 design it's so often compared to, because the deltas are deliberate. DeepSeek-V3 runs 256 routed experts on a 7,168-wide model with 128 heads; GLM-4.5 runs 160 experts on a 5,120-wide model with 96 heads but proportionally *more* of them, and 89 MoE layers to DeepSeek's 58. Both keep a single shared expert that every token uses (a cheap way to absorb the "common" computation so the routed experts can specialize). Both activate 8 experts per token. So the macro-recipe is shared — sparse MoE, shared expert, top-8 — but GLM-4.5 spent its parameter budget on *depth and heads* rather than *width and expert count*, on the explicit hypothesis that depth buys reasoning. When you read two frontier MoE reports side by side, the shared structure tells you what the field agrees on, and the deltas tell you where each team placed its bet.

If you want the broader landscape these choices live in — how GLM-4.5's deep-narrow MoE compares to Qwen, Llama, Gemma, and DeepSeek — the [modern LLM architectures](/blog/machine-learning/large-language-model/modern-llm-architectures-qwen-llama-gemma-deepseek) deep-dive is the companion map.

### The middle generations, in one table

Between the 2022 dense giant and the 2025 sparse flagship sit the models that did the unglamorous hardening, and each step is a clean, dated architecture lesson:

| Model | Year | Architecture move | Why it mattered |
| --- | --- | --- | --- |
| ChatGLM-6B | 2023 | small dense, ~1T tokens, 2K ctx | proved the recipe shrinks to a laptop-deployable size |
| ChatGLM2-6B | 2023 | Multi-Query Attention + FlashAttention | context 2K → 32K, +42% inference speed from a smaller KV cache |
| ChatGLM3-6B | 2023 | function-calling prompt format | the substrate every later agentic capability is built on |
| GLM-4 | 2024 | GQA, 2D-RoPE, RMSNorm + SwiGLU, unified 150K vocab | grouped-query attention recovers MQA's quality while keeping its KV savings |

Two details are worth lifting. GLM-4's tokenizer **merges its Chinese/multilingual byte-level BPE with OpenAI's `cl100k_base`** — a pragmatic way to inherit strong English tokenization instead of re-deriving it. And the FFN is sized at **10/3 × hidden** so that switching to SwiGLU (which adds a third projection) keeps the parameter count constant. These are the bookkeeping choices that never make the abstract but quietly decide whether your model fits in memory. The MQA→GQA arc is also a tidy worked example of a recurring serving lesson: MQA slashes the KV cache by collapsing to one KV head but costs quality; GQA recovers most of that quality with a handful of KV-head groups. For the tokenizer rabbit hole specifically, see [designing and choosing a tokenizer](/blog/machine-learning/large-language-model/designing-choosing-tokenizer-llm).

## 2. Engineering: doing more with less hardware

> **Senior rule of thumb:** at 100B+ parameters, your model does not fail because the math is wrong. It fails because a number overflowed a float, a gradient spiked, or the run cost more GPUs than you have. The GLM engineering story is almost entirely about *not* failing for those reasons.

### The FP16 stability stack: GLM-130B's real contribution

The single most useful engineering artifact in the entire GLM corpus is GLM-130B's account of how they kept a 130B FP16 run from collapsing. Most labs that hit this wall switch to BF16 and move on. GLM-130B deliberately stayed on FP16 — because BF16 used ~15% more runtime memory and was unsupported on the V100-class GPUs they wanted the model to be *accessible* on — and then had to solve every problem BF16 would have papered over.

![A graph showing how DeepNorm, embedding-gradient-shrink, and FP32 softmax jointly keep GLM-130B's FP16 training stable](/imgs/blogs/glm-lineage-frontier-llm-technique-5.png)

The graph above traces the two failure modes and the specific fix for each. Work through it:

**Failure mode 1 — embedding gradient spikes.** Early in training, the embedding layer's gradient norm ran orders of magnitude larger than every other layer, and those spikes *preceded* loss collapses by several steps. The fix is one line, and it's beautiful:

```python
## Embedding Gradient Shrink (EGS): shrink the embedding's effective gradient
## by routing most of the forward signal through a detached (no-grad) copy.
## At alpha=0.1, only 10% of the gradient flows back into the embedding.
alpha = 0.1
word_embedding = word_embedding * alpha + word_embedding.detach() * (1 - alpha)
```

Because `.detach()` blocks gradient flow, this expression is an identity in the forward pass but scales the embedding's gradient by `alpha` in the backward pass. At α = 0.1 it "wipes out most spikes" with negligible cost. Just as important is the *diagnostic* GLM-130B extracted from the same observation: **gradient norm is an early-warning signal for divergence.** Watch it, and you can intervene before the loss curve tells you you're already dead.

**Failure mode 2 — attention logit overflow.** As the model scales, attention scores grow until they overflow the FP16 range. GLM-130B's fix is to compute the **attention softmax in FP32** while keeping everything else FP16, combined with **Post-LN + DeepNorm** (`DeepNorm(x) = LayerNorm(α·x + Network(x))`, α = (2N)^½) to bound the value scale of deep layers. The DeepNorm choice was itself the survivor of a bake-off: the report states plainly that Pre-LN, Post-LN, and Sandwich-LN *all diverged* in their test runs, and only DeepNorm was stable.

> Pre-LN, Post-LN, and Sandwich-LN all diverged; DeepNorm is the most stable one, with a small gradient norm that does not spike in early training.

This is the kind of negative result that saves other teams months. Three reasonable normalization choices, three divergent runs, one survivor — and they told you. The lineage then carried the *intent* forward while modernizing the *mechanism*: GLM-4.5's QK-Norm is the spiritual successor to FP32-softmax, solving the same attention-logit-range problem with a cleaner primitive. (Compare Kimi K2's QK-Clip, which clips rather than normalizes — same disease, different prescription.)

### Quantization and precision: the through-line that pays for the RL

The second engineering through-line is "low precision, everywhere, as a first-class design constraint." GLM has done this in every era, and the *reason* it transfers is one observation made back in 2022.

![Before-after comparison of INT4 weight-only quantization in GLM-130B versus FP8 RL rollouts in GLM-4.5](/imgs/blogs/glm-lineage-frontier-llm-technique-6.png)

The before/after above shows the two ends of the through-line. On the left, GLM-130B became the **first 100B-scale model to ship INT4** with essentially no quality loss (LAMBADA −0.74%, MMLU +0.05%), using nothing fancier than round-to-nearest (RTN) weight-only quantization with *no* calibration data. The reason it worked is the load-bearing insight:

> GLMs tend to have much narrower weight distributions than similar-sized GPTs, so INT4 suffices where GPT-style models are limited to INT8.

Narrow weight distributions quantize cleanly. That single property let a 130B model run inference on **4× RTX 3090 (24 GB)** or **8× RTX 2080 Ti (11 GB)** — a model you could, in principle, host in a closet.

On the right, the *same instinct* shows up in GLM-4.5's RL stack, three years later, in a completely different context. Reinforcement learning is rollout-dominated: most of the wall-clock is the model generating trajectories, not the gradient step. So GLM-4.5's slime infrastructure runs **BF16 for training but FP8 for the inference rollout**, with online block-wise FP8 quantization of the policy parameters at each iteration. Cheap rollouts make RL affordable. The philosophy — *spend your precision only where the gradient needs it* — is identical to the INT4 work; only the venue changed. If you want the full landscape of where INT4, INT8, and FP8 each win, the [quantization in LLM](/blog/machine-learning/large-language-model/quantization-in-llm) and [past the 4-bit wall](/blog/machine-learning/large-language-model/past-4-bit-wall-frontier-llm-quantization) deep-dives are the companions.

The quality receipts are what make the precision philosophy more than bravado. GLM-130B's INT4 quantization moved LAMBADA by −0.74% and MMLU by *+0.05%* — within noise, on a 130B model, with zero calibration data. That's the empirical permission slip to deploy a frontier model on consumer GPUs. On the FP8 side, the slime stack's online block-wise quantization keeps the *training* path in BF16 (where the gradient lives and precision actually matters) while running only the *rollout* forward passes in FP8, so a numerical error in a rollout costs you a slightly noisier reward estimate, not a corrupted gradient. The asymmetry is the insight: you can be aggressive about precision exactly where the result is a sampled trajectory you're going to grade anyway, and conservative where it feeds the optimizer. Most teams pick one precision for the whole pipeline; GLM picks per *role*.

### Parallelism and the cost ledger

The unglamorous engineering numbers are worth keeping as a calibration reference, because they tell you what "frontier on a budget" actually looked like:

| Engineering metric | GLM-130B value |
| --- | --- |
| Hardware | 96× DGX-A100 (8×40 GB) = 768 A100-40G GPUs |
| Parallelism | 4-way tensor × 8-way pipeline × data parallel |
| Global batch | 4,224 |
| Throughput | HFU 43.3% / MFU 32.5% |
| Wall-clock | ~60 days (May 6 – Jul 3, 2022) for 400B tokens |
| Optimizer | AdamW, LR 1e-7 → 8e-5, cosine 10× decay |

A 43.3% hardware-FLOP utilization on 768 A100s for two months is a sobering reminder that the *systems* problem dominates the *science* problem at this scale. For the modern version of these concerns — multi-node training recipes and the failure modes you hit — see [multi-node LLM training](/blog/machine-learning/mlops/multi-node-llm-training-recipe-troubleshooting).

#### Why the topology looks the way it does

The 4×8 tensor-pipeline split isn't arbitrary; it falls out of memory arithmetic. A 130B model in FP16 is ~260 GB of weights alone, and the AdamW optimizer states (FP32 momentum + variance, plus an FP32 master copy of the weights) add roughly another terabyte — vastly past a single 40 GB A100. **Tensor parallelism** (4-way) shards each layer's matmuls across 4 GPUs so the weights fit; **pipeline parallelism** (8-way) splits the 70 layers into 8 stages so each device holds activations for only a slice; **data parallelism** replicates the whole assembly to consume the 4,224 global batch. The `70 = 9×8 − 2` layer count is the tell that the *pipeline* dictated an *architecture* number — 9 layers per stage across 8 stages, minus 2 to balance the embedding-heavy first and last stages. And the 32.5% model-FLOP utilization is the toll for all that cross-device communication: at 100B scale, the bytes you move between GPUs, not the FLOPs you compute, set the clock. It's the same lesson the [serving-at-scale](/blog/machine-learning/large-language-model/serving-llms-at-scale-production-systems) story teaches from the inference side — past a certain size, every problem is a bandwidth-budget problem wearing a different hat.

## 3. Finetuning & alignment: SFT → RLHF → agentic RL

> **Senior rule of thumb:** the base model decides what's *possible*; post-training decides what's *delivered*. GLM's alignment story is the part that changed the most across the lineage, because "useful" is a moving target — it went from "answer politely" in 2023 to "autonomously resolve a GitHub issue" in 2025.

### The SFT data stance: authentic humans over synthetic templates

GLM-4's alignment chapter contains one sentence that is worth more than most alignment papers, because it's a falsifiable design stance stated plainly:

> In SFT, we find that authentic human prompts and interactions, instead of template-based or model-generated responses, are vital to the alignment quality.

In a moment when the field was drunk on synthetic instruction data, the GLM team's measured position was that *real* human prompts and interactions carry the alignment signal that templates and model-generated responses do not. They scored responses on a rubric of safety, factuality, relevance, helpfulness, and human preference. This is a cheap technique to steal: before you generate ten million synthetic SFT examples, ask whether the marginal authentic example is worth more.

### RLHF that GLM actually shipped: ChatGLM-RLHF

The RLHF was not a sketch. The companion paper [ChatGLM-RLHF](https://arxiv.org/abs/2404.00934) documents a production system with three components — preference-data collection, reward-model training, and policy optimization — using **both PPO and DPO**. The engineering details are the interesting part: **reward-variance reduction**, **model parallelism with fused gradient descent**, and **regularization constraints to avoid catastrophic forgetting** of the SFT capabilities. The payoff was a ~15% average win-rate gain over the SFT model on Chinese alignment tasks. The reason RLHF earned its complexity is specific: it fixes the things SFT structurally can't — response rejection, safety, bilingual token mixing, and multi-turn coherence. If you want the conceptual grounding on when to reach for PPO vs. DPO vs. GRPO, the [GRPO vs DPO vs PPO decision guide](/blog/machine-learning/large-language-model/grpo-vs-dpo-vs-ppo-decision-guide) is the prerequisite read.

### Long-context alignment: the LongAlign recipe

Stretching a model to 128K is two problems, not one: a *positional* problem (does attention still work at that length) and an *alignment* problem (does the model actually *use* the long context to follow instructions). GLM solved the second with [LongAlign](https://arxiv.org/abs/2401.18058), and the techniques are sharper than the usual "train on long documents" hand-wave:

- **Self-Instruct long instruction data.** Long-context instruction pairs are scarce, so they're synthesized — long source documents plus generated tasks over them — rather than scraped.
- **Packing with sorted batching.** Long sequences are wildly variable in length; naive batching wastes compute on padding. LongAlign packs multiple sequences per batch and sorts by length to minimize padding — a throughput win that's easy to underestimate.
- **The loss-weighting fix.** Here's the subtle bug: when you pack *k* sequences into one batch and take a normal mean loss, a packed batch containing one long sequence and many short ones under-weights the short sequences relative to an unpacked run. LongAlign introduces a loss weighting that **restores each sequence's correct contribution under packing**, recovering up to ~30% on long tasks while leaving short-task quality intact.

They also shipped **LongBench-Chat**, a 10K–100K-token evaluation, because — as in the data section below — the capability they were optimizing didn't have a good public ruler. The open GLM-4-9B-Chat-1M variant pushes this to a 1M-token context (≈2M Chinese characters), which is best read as an existence proof rather than a production guarantee, but the recipe that gets there is the reusable artifact.

### Agentic groundwork: AgentTuning and All Tools

GLM-4's autonomous tool use did not appear in 2024 — it was assembled over three years, and [AgentTuning](https://arxiv.org/abs/2310.12823) is the keystone. The method is a study in *not destroying general ability while adding a narrow skill*:

- **AgentInstruct**, a dataset of **1,866 GPT-4-generated, reward-filtered agent trajectories** across six environments (ALFWorld, WebShop, Mind2Web, knowledge graph, OS, database).
- **A 1:4 mixing ratio.** The agent trajectories are mixed with general ShareGPT-style instructions at roughly 1 part agent to 4 parts general. Train on agent data alone and the model becomes a brittle agent that forgot how to chat; the 1:4 ratio is what lets AgentLM-70B reach GPT-3.5-turbo's agent performance on *unseen* tasks **without** regressing on general benchmarks.

That hard-won "don't let the narrow skill cannibalize the general one" lesson is exactly why GLM-4 All Tools could be a single model that *autonomously decides when and which tool(s) to use* — web browser, Python interpreter, text-to-image, or user-defined functions — rather than a tool-router bolted onto a chat model. The capability compounds: WebGLM gave it retrieval-augmented browsing, ChatGLM3 gave it the function-call format, AgentTuning gave it trajectories, and All Tools cashed all three in at once.

### The GLM-4.5 training pipeline: 23 trillion tokens to one hybrid model

By GLM-4.5 the post-training had become a multi-stage pipeline elaborate enough to need its own diagram. This is the centerpiece technique of the modern era, so it's worth walking carefully.

![The GLM-4.5 training pipeline, from 23 trillion tokens of pretraining through three expert models to one distilled hybrid](/imgs/blogs/glm-lineage-frontier-llm-technique-7.png)

The pipeline above is the full journey. Trace it stage by stage:

1. **General pretraining — 15T tokens.** Web, multilingual, code, and math/science, with quality-bucketed up-sampling (the highest-quality web bucket is seen 3+ times; aggressive MinHash + SemDedup deduplication).
2. **Code & reasoning continual pretraining — 7T tokens.** The corpus is reweighted to up-sample GitHub code and coding/math/science web. This two-phase structure — general first, then domain-heavy — is itself a technique: don't dilute your whole run with code, concentrate it in a second phase.
3. **Mid-training, three stages.** Context grows from 4K → 32K (repo-level code, with same-repo files concatenated) → injects ~500B tokens of *synthetic reasoning traces* → extends to 128K with long-context and agent-trajectory data (~100B tokens). Mid-training is where GLM-4.5 puts the capabilities that pretraining can't cheaply provide.
4. **Post-training — three expert models.** This is the novel move. Rather than one model that must be good at everything, GLM-4.5 trains **three separate experts** — a Reasoning expert, an Agent (agentic/coding) expert, and a General-chat expert — each with its own cold-start SFT and its own expert-specific RL.
5. **Unified distillation.** The three experts are distilled back into **one** model via an "Overall SFT" stage that deliberately *mixes* full-reasoning data with no-reasoning data, so the single output model learns to operate in both thinking and direct-response modes.
6. **General RL.** A final RL pass (RLHF + RLAIF + rule-based) polishes the unified model.

The expert-distillation idea is the one most worth stealing. Training specialists is easier than training a generalist — each expert's RL can use domain-specific rewards without cross-contamination — and distillation recovers a single deployable model. It's the alignment analogue of "divide and conquer," and it's how GLM-4.5 gets a *hybrid reasoning* model (one set of weights that toggles thinking on/off) instead of shipping two models. For the general theory of why distillation transfers capability so efficiently, see [distillation in LLM](/blog/machine-learning/large-language-model/distillation-in-llm).

To make the expert-distillation concrete, here is what the three specialists actually are and how they fold into one model:

| Expert | Cold-start SFT focus | Expert-specific RL signal | Contribution to the unified model |
| --- | --- | --- | --- |
| Reasoning | extended chain-of-thought | verifiable math/science/code rewards + difficulty curriculum | the `thinking` mode |
| Agent | tool-use and coding trajectories | verifiable agentic rewards (search answers, SWE unit tests) | function-calling and multi-turn tool use |
| General chat | broad instruction following | RLHF + RLAIF preference rewards | direct-response mode and helpfulness |

The unification step — the "Overall SFT" — samples millions of examples at up to 128K context and **deliberately mixes data that contains full reasoning traces with data that has none**, so the single output model learns both to deliberate and to answer immediately, and to choose between them. That is the entire mechanism behind a *hybrid reasoning* model: it is not a router sitting in front of two checkpoints, it is one set of weights that saw both behaviors during the distillation SFT and internalized a switch. Training three narrow specialists and merging them costs more total compute than a single generalist run, but each specialist's RL is cleaner — domain-specific rewards, no cross-objective interference — and you still ship one deployable model. That trade (more training compute for cleaner reward signals and a single artifact) is the calculation every team running multi-capability RL should be making explicitly.

### Agentic RL with verifiable rewards

The most modern technique in the corpus is GLM-4.5's agentic RL, and its defining constraint is discipline about *which* tasks get RL at all.

![A graph of GLM-4.5's agentic RL, where web-search and SWE agents produce verifiable rewards that drive a GRPO update](/imgs/blogs/glm-lineage-frontier-llm-technique-8.png)

The graph above shows the loop. The deliberate choice is to focus agentic RL on **web-search and code-generation agents — domains where every action or answer can be automatically checked.** A web-search answer is graded against a known target; a code patch is graded by running unit tests in a hardened sandbox. That gives *dense, verifiable rewards* with no reward model to hack. The mechanics carry several stealable details:

- **GRPO without the KL term.** Sample K trajectories per prompt, set each trajectory's advantage to its reward minus the group mean, and drop the KL regularizer entirely. (See the [GRPO fine-tuning](/blog/machine-learning/large-language-model/fine-tuning-llm-with-grpo) walkthrough for the base algorithm.)
- **Mask the environment tokens.** Only model-generated tokens contribute to the loss; tool outputs and environment feedback are masked out. This is a subtle but critical correctness detail — you don't want to train the model to "predict" the search results.
- **Format penalty as a hard gate.** If the model emits a malformed tool call, the trajectory is halted with *zero* reward. Reliability is trained, not hoped for.

Here is the loss-masking idea in runnable form — the part that's easy to get wrong:

```python
import torch

def grpo_advantages(rewards):
    """GRPO: advantage = reward - group mean (no value network, no KL)."""
    return rewards - rewards.mean()

def masked_policy_loss(logp, ref_logp, advantages, model_token_mask):
    """Only model-generated tokens contribute. Environment/tool-feedback
    tokens (model_token_mask == 0) are excluded from the objective."""
    ratio = torch.exp(logp - ref_logp)                  # importance ratio
    per_token = ratio * advantages.unsqueeze(-1)        # broadcast traj advantage
    per_token = per_token * model_token_mask            # <-- the load-bearing line
    # token-weighted mean (GLM-4.5's choice for code RL: avoids length bias)
    return -(per_token.sum() / model_token_mask.sum())
```

Two more findings from the RL chapter are the kind that only show up after expensive mistakes, and both became case studies below: **single-stage RL at the full 64K output length beats a progressive length curriculum** (lengthening gradually causes the model to *irreversibly* unlearn long-context skill), and a **dynamic sampling temperature** raises exploration as reward plateaus, guarded by a validation check that forbids any temperature that drops performance more than 1%. For the broader debate on what RL actually does to reasoning models, the [tricks or traps RL deep-dive](/blog/paper-reading/large-language-model/part-i-tricks-or-traps-a-deep-dive-into-rl-for-llm-reasoning) and [GLM-4.5's own paper-reading](/blog/paper-reading/large-language-model/glm-4-5-agentic-reasoning-and-coding-arc-foundation-models) go deeper than this survey can.

The loop has one more turn that's easy to miss and very much worth stealing: **iterative self-distillation inside the RL itself.** Once the policy plateaus under RL, GLM-4.5 replaces the original cold-start SFT data with the *current RL model's own best outputs*, retrains a fresh SFT model on that stronger data, and resumes RL from there at higher difficulty. Each lap ratchets the floor up — the SFT starting point for round *n+1* is better than where round *n* finished — which is why the diagram routes "stronger policy" into "new SFT data" rather than looping a reward back. There's also a clean test-time-compute story hiding in the agentic results: on BrowseComp, accuracy scales smoothly as you allow the model more browsing turns (from 8 up to 128), which means the RL didn't just teach a fixed policy, it taught a policy that *productively uses more interaction budget when given it* — the agentic analogue of a reasoning model thinking longer. Both behaviors are emergent properties of training on verifiable, multi-turn tasks, and neither shows up if you only RL on single-shot, checkable answers.

### The slime RL infrastructure

The agentic RL is only affordable because of the infrastructure underneath it. GLM-4.5's open-source **slime** stack is three modules — Training (Megatron) reading from a Data Buffer, Rollout (SGLang + a router) writing to it — running in one of two modes. Synchronous *colocated* mode (training and inference on the same workers) suits reasoning/math/code RL where rollouts are fast. Asynchronous *disaggregated* mode (separate rollout and training GPUs, orchestrated with Ray) suits slow agentic/SWE rollouts, where a single long trajectory would otherwise stall the whole pipeline. The FP8-rollout / BF16-train split discussed earlier lives here. This is a whole article in the series on its own; the technique to file away now is *disaggregate rollout from training when your trajectories have high variance in length*, which is exactly the agentic case.

### The other half of RL: when the reward isn't verifiable

Verifiable rewards are the glamorous part, but most of "being a good assistant" is not automatically checkable. GLM-4.5's General RL stage handles that with **multi-source feedback** — rule-based signals where possible, a human-preference reward model (RLHF), and a model-based judge (RLAIF) — split into targeted sub-stages that read like a punch-list of everything that annoys users:

- **Holistic RL** over ~5,000 prompts spanning a 7 / 33 / 139-tier category hierarchy, to lift broad helpfulness.
- **Instruction-Following RL** against 7 major and 151 minor constraint types — the "did it actually do all five things I asked, in the format I specified" axis.
- **Function-Calling RL**, itself split into a *step-wise* rule-based reward (exact match on tool name and arguments) and an *end-to-end* multi-turn reward over a whole trajectory, with an LLM-simulated user driving the conversation.
- **Pathology RL**, a dataset built specifically to kill language-mixing, repetition, and format errors — the failure modes that don't show up in benchmarks but instantly break user trust.

The lesson worth stealing is structural: don't run one undifferentiated RLHF pass and hope. Decompose "good behavior" into named failure modes and target each with the cheapest sufficient reward — exact-match rules where you can, preference models where you must.

### The reasoning branch: GLM-Z1 and finding the hard problems

Before GLM-4.5 unified everything, the reasoning capability lived in its own line: **GLM-Z1-32B-0414**, built on the open GLM-4-32B base via cold-start SFT plus extended RL on math, code, and logic — a 32B model that the team reports rivaling the 671B DeepSeek-R1 on several math tasks. Two of its techniques carried into GLM-4.5's reasoning RL:

- **Pairwise-ranking (critic) RL for the non-verifiable parts.** Math has checkable answers; "is this explanation good" does not. GLM-Z1 runs verifiable-reward RL for STEM *and* a separate pairwise-preference RL for general capability, as two distinct signal streams rather than one blended reward.
- **A difficulty-based curriculum that hunts the frontier.** GLM-4.5's reasoning RL runs two stages from a verified-answer pool: stage one on moderate problems (16 samples per prompt), stage two on *extremely* difficult problems — specifically those where `pass@8 = 0` but `pass@512 > 0`, sampled 512 times each. That filter is clever: it isolates problems the model *can* eventually solve but *usually* fails, which is exactly where the learning signal is densest. The ablations show this lifting AIME'24 to ~83% before the rest of the stack.

There's also the **Rumination** variant (GLM-Z1-Rumination-32B), which integrates search *into* the thinking process and is trained end-to-end with multiple rule-based rewards for open-ended deep-research tasks — the same "tools inside the reasoning trace" idea that GLM-4.5 generalizes. For the broader picture of how RL reshapes reasoning, the blog's [pretraining large reasoning models](/blog/machine-learning/large-language-model/pretraining-large-reasoning-models) and [DeepSeek-R1 read](/blog/paper-reading/large-language-model/deepseek-r1-incentivizing-reasoning-capability-in-llms-via-reinforcement-learning) are the neighbors.

## 4. Data & evaluation: curate, decontaminate, and measure yourself

> **Senior rule of thumb:** a model is a function of its data and a fiction of its benchmarks. The GLM team treats both with unusual rigor — and, tellingly, builds its *own* benchmarks when the existing ones don't measure what they care about.

### The data pipeline, generation by generation

The data techniques compound across the lineage. GLM-130B established the canonical stages — **exact + fuzzy deduplication → quality filtering → tokenization** — on a 400B-token bilingual corpus (≈200B English from the Pile, ≈200B Chinese from WudaoCorpora plus crawl). It also slipped in **Multi-task Instruction Pretraining (MIP)**: 5% of tokens drawn from 74 prompted datasets, mixed *into* pretraining to boost zero-shot ability. That "season the pretraining with a little instruction data" trick predates the instruction-tuning craze and still works.

GLM-4 scaled this to ~10T tokens with classifier-based quality filtering, tuned domain-mixing ratios, synthetic data for code and math, long-context construction by concatenation-and-filtering, and explicit benchmark decontamination. GLM-4.5 took it to 23T with the two-phase up-sampling structure described earlier. The reusable shape across all three:

- **Dedup is stage one, always** — both exact and fuzzy/semantic.
- **Up-sample quality, don't just filter.** The best web bucket is seen multiple times; low-quality data is down-weighted, not merely dropped.
- **Concentrate domain data in a second phase** rather than diluting the whole run.
- **Mid-training is the home for the expensive data** — repo-level code, synthetic reasoning, agent trajectories — that you can't afford across the full corpus.

### Building your own ruler

The most distinctive data-axis technique is that the GLM group *builds the benchmarks it needs.* When you read the GLM-4 evals, a striking fraction of them are the team's own instruments:

| Benchmark | What it measures | Origin |
| --- | --- | --- |
| [AgentBench](https://arxiv.org/abs/2308.03688) | LLM-as-agent across 8 interactive environments (OS, DB, KG, web, games) | THUDM |
| [AlignBench](https://arxiv.org/abs/2311.18743) | Chinese alignment quality, 683 queries, 8 categories, LLM-judge + CoT | THUDM |
| [LongBench](https://arxiv.org/abs/2308.14508) | Bilingual long-context, 21 datasets / 6 task types | THUDM |
| [NaturalCodeBench](https://arxiv.org/abs/2405.04520) | Real-world Python + Java from natural prompts, Docker-executed | THUDM |
| HumanEval-X | Multilingual code generation (from CodeGeeX) | THUDM |

Why build your own benchmarks? Because the existing ones measured the wrong thing. NaturalCodeBench exists precisely because HumanEval-style toy problems don't predict real-world coding performance. AgentBench exists because no benchmark measured multi-turn decision-making across realistic environments. The technique here is meta and important: **if you're optimizing a capability no public benchmark captures, the benchmark is part of the deliverable.** GLM-4.5's evals lean heavily on *verifiable* and *execution-based* measures — SWE-bench Verified, TAU-bench, LiveCodeBench, BrowseComp — for the same reason the RL does: a number you can check is worth ten numbers you have to trust.

The headline GLM-4.5 results are the receipts for all four axes working together:

| Axis | Benchmark | GLM-4.5 |
| --- | --- | --- |
| Agentic | TAU-bench | 70.1 |
| Agentic | BFCL v3 | 77.8 (best in its comparison set) |
| Reasoning | AIME 24 | 91.0 |
| Reasoning | MMLU-Pro | 84.6 |
| Reasoning | MATH 500 | 98.2 |
| Coding | SWE-bench Verified | 64.2 |
| Coding | Terminal-Bench | 37.5 |

Third overall and second on agentic across all evaluated models, at roughly half DeepSeek-R1's parameters and a third of Kimi K2's. The architecture, engineering, alignment, and data techniques are not independent wins — they're the same accreting stack, measured.

### The vision branch: the same instincts, a different modality

The lineage's multimodal arm — GLM-4.1V-Thinking and GLM-4.5V ([arXiv:2507.01006](https://arxiv.org/abs/2507.01006)) — is the clearest proof that what GLM has is a *methodology*, not a bag of text tricks, because the same instincts cross cleanly into pixels. The architecture is a ViT encoder (initialized from AIMv2-Huge, with its 2D convolutions swapped for 3D to downsample video) feeding a GLM language model through an MLP projector, with 2D-RoPE in the vision tower (so it survives extreme >200:1 aspect ratios) and 3D-RoPE in the decoder. The post-training reuses the text playbook almost verbatim: cold-start SFT enforcing a `<think>…</think><answer>…</answer>` format, then **RL with verifiable rewards across every domain at once** — STEM, OCR, charts, grounding, GUI agents, video understanding. The one genuinely new technique is **RLCS (Reinforcement Learning with Curriculum Sampling)**: offline difficulty scoring plus online adaptive reweighting that keeps each sample's difficulty matched to the model's *evolving* ability — the multimodal cousin of GLM-4.5's `pass@8=0 / pass@512>0` filter. The result, a 9B model surpassing the 72B Qwen2.5-VL on 29 of 42 benchmarks, is the *same* "verifiable reward + curriculum" recipe that powers the text models. A dedicated article in this series takes the vision branch apart; for now it's a reminder that the four axes are modality-agnostic.

## The through-lines: what persisted, what got reinvented

![A matrix laying the four technique axes across four GLM generations, showing which techniques persisted and which were reinvented](/imgs/blogs/glm-lineage-frontier-llm-technique-9.png)

Lay all four axes across all four generations — the matrix above — and the lineage's logic becomes legible. Three patterns:

**Architecture and data persist and scale.** The blank-infilling objective's descendants run through the whole lineage; the bilingual corpus and dedup-first pipeline only grew (400B → 10T → 23T). These are the *foundation* techniques: you set them early and they compound. The cost of changing them is enormous, so GLM rarely did.

**Engineering and alignment get reinvented every generation.** Stability went from "reorder LayerNorm" (2021) → "DeepNorm + EGS + FP32 softmax" (2022) → "128K context infra" (2024) → "Muon + FP8 + slime" (2025). Alignment went from nothing → "5% instruction pretraining" → "SFT + RLHF + All Tools" → "expert distillation + agentic RL." These axes track the *frontier*, which moves, so the techniques churn. If you're picking which GLM papers to read first, read the engineering and alignment chapters of the *latest* report and the architecture/data chapters of the *earliest* — that's where each axis's signal is densest.

**The instincts are constant even when the mechanisms aren't.** "Spend precision only where the gradient needs it" produced INT4 in 2022 and FP8 in 2025. "Keep attention logits in range" produced FP32-softmax in 2022 and QK-Norm in 2025. "Don't make the model choose between two capabilities" produced blank infilling in 2021 and hybrid reasoning in 2025. The *mechanisms* are disposable; the *instincts* are the actual intellectual property. That's the deepest reason to read a lineage instead of a model: you can only see an instinct by watching it survive three different implementations.

## Technique deep-cuts: ten stories from the reports

Surveys flatten the texture. Here are ten specific moments from the GLM reports, each a self-contained lesson, each the kind of detail that only shows up when a team is honest about what actually happened.

### 1. The gradient spike that predicted the future

The GLM-130B team noticed that the embedding layer's gradient norm would spike *several steps before* the loss collapsed. That temporal gap is the gift: it turns gradient norm from a postmortem statistic into a leading indicator. The embedding-gradient-shrink fix (α = 0.1) suppressed the spikes, but the more transferable lesson is the *monitoring* discipline — instrument the embedding gradient norm and you get a smoke alarm, not an autopsy. I have personally watched runs die from exactly this failure mode while staring at a loss curve that looked fine until it didn't; the GLM-130B report is the document I wish I'd read first.

### 2. The FP16-over-BF16 bet

Choosing FP16 over BF16 in 2022 looks, in hindsight, like fighting the tide — BF16 makes most of these overflow problems disappear. But the GLM-130B team made the call *on accessibility grounds*: BF16 cost ~15% more runtime memory and didn't run on V100-class hardware, which would have locked out exactly the researchers who most needed an open 130B. They paid for that choice with DeepNorm, EGS, and FP32-softmax — three techniques the field then got to keep. The lesson isn't "use FP16." It's that a constraint you accept for the *right* reason (here, who gets to use the model) can force you to invent techniques that outlive the constraint.

### 3. INT4 in a closet

The narrow-weight-distribution insight is the unsung hero of the GLM-130B report. It's a one-line empirical observation — GLMs have narrower weight distributions than GPTs — but it's *why* INT4 RTN worked with no calibration when everyone else was stuck at INT8. The downstream consequence was a 130B model that runs on 4× RTX 3090. When you're designing a model you intend to quantize, the weight distribution is a property you can influence at training time, not just measure at deployment. GLM treated quantization-friendliness as an architecture goal, and got single-server inference for free.

### 4. "Authentic human prompts are vital"

In the middle of the synthetic-data gold rush, GLM-4's report quietly states that authentic human prompts and interactions beat template-based or model-generated responses for alignment quality. It's a single sentence with enormous practical weight: it says the marginal *real* example is worth more than the marginal *synthetic* one, which inverts the economics most teams were running on. The technique is a discipline, not a trick — but disciplines are cheaper to adopt than tricks, and this one is free.

### 5. The browser that beat GPT-4

GLM-4 All Tools, on browser-based information-seeking, scored **78.08 versus GPT-4's 67.12** — a clear win on a capability the GLM team had been building toward since WebGLM in 2023. The lesson is about *focus compounding*: a team that has shipped retrieval-augmented browsing (WebGLM), agent tuning (AgentTuning), and a function-calling prompt format (ChatGLM3) has accumulated exactly the pieces needed for autonomous tool use. The All-Tools win wasn't a 2024 breakthrough; it was three years of agentic groundwork cashing out at once.

### 6. The attention heads that didn't lower loss

GLM-4.5's finding that 2.5× more attention heads improved reasoning benchmarks *without* improving training loss is the report's most quietly subversive result. It's a direct warning against optimizing the proxy (loss) instead of the goal (reasoning). If your architecture search is ranked by validation loss, you will systematically discard changes like this one. The technique to internalize: when you can afford it, evaluate architecture variants on *downstream* tasks, because the loss curve is blind to some of the things you actually care about.

### 7. Cosine over WSD, because WSD underfit

GLM-4.5 explicitly rejected the fashionable warmup-stable-decay (WSD) learning-rate schedule in favor of plain cosine decay, after finding that WSD-trained models "perform worse on general benchmarks (SimpleQA, MMLU), indicating underfitting in the stable stage." This is a useful counter-data-point to a popular technique: WSD is great for some things, but the GLM team measured it underfitting at their scale and config, and said so. Schedules are not universal; measure yours.

### 8. The single-stage 64K RL and the unlearning cliff

The intuitive way to RL a long-context reasoner is to start short and lengthen gradually. GLM-4.5 tried it and found that progressive length curricula caused the model to **"unlearn" its long-context capabilities — a significant and irreversible drop.** Their fix was to run single-stage RL directly at the full 64K output length, which "continually pushes the model's limits." The transferable warning: a curriculum that *removes* a capability between stages can cause irreversible forgetting. Sometimes the harder, full-difficulty regime from step one is safer than the gentle ramp.

### 9. The temperature that climbs as reward plateaus

GLM-4.5's RL raises the sampling temperature when average reward stabilizes — restoring exploration just as the policy starts to converge and stagnate — but guards it with a validation check: the next phase's temperature is "the maximum value that does not cause a performance drop of more than 1% from the current optimum." It's a small control-loop, but it encodes a real principle: exploration and exploitation aren't a one-time tradeoff, they're a schedule, and you can manage that schedule with a cheap held-out guardrail.

### 10. The XML tool-call template that dodges JSON escaping

A tiny, intensely practical detail: GLM-4.5 uses an **XML-like function-call template** (`<tool_call>`, `<arg_key>`/`<arg_value>` tags) instead of JSON, specifically to **reduce character-escaping headaches when function arguments contain code.** Anyone who has watched a model mangle a JSON payload because the `code` field had quotes and backslashes in it will recognize this immediately. For an agentic foundation model whose arguments are *frequently* code, the serialization format is not a cosmetic choice — it's a reliability choice. It's the most copy-pasteable idea in the whole report.

### 11. The 1:4 ratio that saved the generalist

AgentTuning's most transferable number is the **1:4 agent-to-general data ratio**. The naive instinct, when you want a model to be a good agent, is to train it on agent trajectories. Do that exclusively and you get a model that's a competent agent and a worse everything-else — the narrow skill cannibalizes the broad one. AgentTuning's measured fix was to dilute the agent data four-to-one with general instructions, which preserved general benchmarks while *still* generalizing the agent skill to unseen environments. The lesson generalizes far beyond agents: whenever you graft a narrow capability onto a generalist, the mixing ratio with general data is a first-class hyperparameter, not an afterthought — and the right answer is usually "less narrow data than you'd think."

### 12. The packing bug that quietly mis-weights your loss

LongAlign's loss-weighting fix addresses a bug almost everyone hits and almost no one notices. To train efficiently on variable-length long-context data you pack several sequences into one batch. But a standard token-mean or sequence-mean loss over a packed batch silently changes how much each sequence contributes relative to an unpacked run — a packed batch dominated by one long sequence under-counts the short ones. The model trains, the loss goes down, and you never see the problem; you just get slightly worse instruction-following at length. LongAlign's correction restores each sequence's intended weight under packing. The meta-lesson: any time you change the *batching* for throughput, re-derive what your loss is actually averaging over, because efficiency hacks love to smuggle in a silent reweighting.

### 13. The science score that jumped just from cleaning the data

A small, almost embarrassing GLM-4.5 result: restricting the science RL data to **expert-verified multiple-choice questions** moved GPQA-Diamond from 62.9% to 65.8% — a near-3-point gain with no algorithmic change, purely from removing noisy or wrong items from the reward pool. It's a useful counterweight to the instinct that every gain must come from a cleverer method. When your reward is only as trustworthy as the data behind it, *cleaning the data is a learning algorithm.* For verifiable-reward RL especially, a contaminated answer key doesn't just add noise — it actively teaches the model to be confidently wrong.

### 14. Why slow rollouts forced two RL architectures

slime ships *two* execution modes for a reason that's pure systems engineering. Reasoning and math RL have fast, uniform-length rollouts, so colocating training and inference on the same GPUs (synchronous mode) keeps everything busy. Agentic and SWE rollouts are the opposite: a single trajectory might involve dozens of tool calls and minutes of sandboxed test execution, with wild variance in length. Run those synchronously and the slowest trajectory in a batch stalls every training GPU behind it. slime's disaggregated asynchronous mode partitions GPUs into dedicated rollout and training pools so a long agentic trajectory never blocks a gradient step. The principle — *match your training topology to the variance of your rollout, not just its average* — is the kind of thing you only learn after a fleet of expensive GPUs spends a week mostly idle.

## What to steal from the GLM playbook

The GLM corpus is a buffet, and not every dish belongs on your plate. Here's the senior-engineer triage.

**Reach for these techniques when:**

- **You're training at a scale where stability is the binding constraint.** DeepNorm, embedding-gradient-shrink, and gradient-norm monitoring are battle-tested 100B-scale survival gear. Steal the *monitoring* even if you use BF16 — the early-warning discipline is precision-agnostic.
- **You're going to quantize, and you know it now.** Design for narrow weight distributions and you may get INT4/FP8 nearly free. Treat quantization-friendliness as a training-time goal.
- **You need one model that does two contradictory things** (thinking and direct, or several domains). Expert-model distillation — train specialists, distill into one — is the cleanest path to a hybrid model the lineage offers.
- **Your task has checkable outcomes.** Verifiable-reward agentic RL (with environment-token masking and format-penalty gating) is the highest-leverage post-training technique in the corpus, *but only* where you can automatically grade the result.
- **Your agent's tool arguments are code.** Use the XML-style tool-call template. This one is free and immediate.

**Skip or adapt when:**

- **You're below ~10B parameters.** Most of the engineering stack (DeepNorm bake-offs, EGS, FP32-softmax, slime disaggregation) is solving problems you don't have yet. The architecture and data instincts still apply; the stability machinery doesn't.
- **Your tasks aren't verifiable.** Agentic RL on un-checkable tasks reintroduces exactly the reward-hacking GLM-4.5 designed it to avoid. Use RLHF/RLAIF there, not verifiable-reward RL.
- **You can't afford three expert training runs.** Expert distillation assumes you can train specialists *and* distill. If you have one budget, a single well-mixed SFT may be the pragmatic call.
- **You're chasing a public-benchmark number.** GLM builds its own benchmarks *because* it's optimizing capabilities the public ones miss. If your goal is a leaderboard, the "build your own ruler" technique is the wrong tool.

The unifying meta-technique — the one thing that survives every scale and every axis — is the GLM team's habit of *writing down the negative results.* The diverged LayerNorm runs, the WSD underfitting, the irreversible unlearning from length curricula: these are the expensive lessons, and the reports give them to you free. That's why this series exists. Read on for the per-stage deep-dives, where we open up each of these techniques one report at a time.

## What the rest of this series covers

This was the map. Each remaining article opens one stage of the lineage and goes to the primary source, extracting the techniques in depth rather than in survey:

- **Autoregressive Blank Infilling (GLM, 2021)** — the objective, the 2D positional scheme, and why the hybrid mask still echoes in today's hybrid-reasoning models.
- **Engineering GLM-130B** — the full normalization bake-off, the FP16-over-BF16 decision, the parallelism topology, and INT4 quantization end to end.
- **From ChatGLM to GLM-4 All Tools** — the authentic-data SFT philosophy, ChatGLM-RLHF, the LongAlign recipe, and how All Tools learned autonomous tool use.
- **GLM-4.5 Architecture** — deep-narrow MoE, loss-free routing, QK-Norm, the MTP layer, and the Muon optimizer, ablation by ablation.
- **Training GLM-4.5** — expert-model distillation, the GRPO-without-KL recipe, the difficulty curriculum, dynamic-temperature exploration, and the slime infrastructure.
- **How Zhipu Measures Its Models** — AgentBench, AlignBench, LongBench, and NaturalCodeBench as a benchmark-*design* discipline, not just a scoreboard.
- **The Vision Branch** — GLM-4.1V-Thinking, GLM-4.5V, and RLCS for multimodal reasoning with verifiable rewards.

Read in order, they are a build-log for a frontier model. Read individually, each is a technique you can put to work on Monday.

## Further reading

- **GLM (2021)** — [General Language Model Pretraining with Autoregressive Blank Infilling](https://arxiv.org/abs/2103.10360), the founding objective.
- **GLM-130B (2022)** — [An Open Bilingual Pre-trained Model](https://arxiv.org/abs/2210.02414), the engineering and stability bible.
- **GLM-4 (2024)** — [ChatGLM: A Family of LLMs from GLM-130B to GLM-4 (All Tools)](https://arxiv.org/abs/2406.12793), alignment and agentic capability.
- **GLM-4.5 (2025)** — [Agentic, Reasoning, and Coding (ARC) Foundation Models](https://arxiv.org/abs/2508.06471), the modern MoE + RL recipe; and its dedicated [paper-reading on this blog](/blog/paper-reading/large-language-model/glm-4-5-agentic-reasoning-and-coding-arc-foundation-models).
- **Sibling lineage** — the [Kimi K2 report](/blog/paper-reading/large-language-model/kimi-k2-open-agentic-intelligence) for a same-era contrast on Muon, MoE sparsity, and agentic data.
