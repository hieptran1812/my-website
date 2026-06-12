---
title: "ChatTS: Teaching LLMs to Actually Read Time Series"
date: "2026-06-10"
publishDate: "2026-06-10"
description: "A deep dive into ChatTS, ByteDance's time-series multimodal LLM — why serialized digits and rendered plots both fail, how a value-preserving patch encoder makes numbers a native modality, and how a fully synthetic attribute-grounded data pipeline trains it to understand and reason over multivariate time series."
tags: ["time-series", "multimodal-llm", "chatts", "synthetic-data", "evol-instruct", "bytedance", "qwen2.5", "anomaly-detection", "alignment", "reasoning"]
category: "machine-learning"
subcategory: "Signal Processing"
author: "Hiep Tran"
featured: true
readTime: 51
---

Here is a small experiment you can run in five minutes that quietly embarrasses the entire field of "LLMs for time series." Take a CPU-utilization curve from a production host — a thousand points, a clear daily seasonality, and a single sharp spike at 3:14 AM that is the entire reason anyone is looking at this series. Paste the raw numbers into your favorite frontier chat model and ask: *"At what index does the anomaly occur, and what is its magnitude relative to baseline?"* Then render the same series as a PNG, hand it to the vision endpoint of the same model, and ask the same question. The text version will burn through a few thousand tokens, confidently miscount the index by a few hundred, and round the magnitude to something that "feels" right. The vision version will see a spike, locate it roughly, and have no idea what the actual value is because it is reading pixels, not numbers. Neither one can tell you the value at index 314 to two significant figures, which is the one thing the series unambiguously contains.

This is not a prompt-engineering problem. It is a *modality* problem. Vision-language models work because images are fed to the LLM as a native modality through a dedicated encoder, not described in words or transcribed into hex. Audio models work the same way. Time series — arguably the most numerically precise modality of all — got neither treatment. The community's two default moves were to serialize the numbers into the text channel (lossy, expensive, and it confuses the tokenizer) or to plot the numbers and feed the picture (lossy in a different, worse way: the actual magnitudes are gone). ChatTS is the paper that finally says: stop. Treat time series the way vision models treat images. Build an encoder, preserve the numbers, train on data where the ground truth is exact, and let one model both *understand* (what is the trend, where is the anomaly, what is the period) and *reason* (why are these two metrics correlated, what would explain this pattern, what should an operator do).

![ChatTS inference pipeline: multivariate time series patched and encoded into value-preserving tokens that interleave with text and feed one LLM](/imgs/blogs/chatts-aligning-time-series-llms-1.webp)

The diagram above is the mental model for the whole system: a multivariate time series enters, gets split into fixed-size patches, each patch is encoded by a small MLP into a vector that lives in the LLM's text-embedding space, those vectors are spliced *inline* with the text tokens at the exact positions the prompt refers to them, and one decoder — Qwen2.5-14B-Instruct — reads the whole interleaved stream and emits an answer. There is no plot. There are no serialized digits crowding the context. There is a Time-Series Multimodal LLM (TS-MLLM), the first one to take raw multivariate time series as a native input for both understanding and reasoning, and it is trained *exclusively* on synthetic data. This post is a tour of how each of those pieces works and, more importantly, why each is necessary.

> [!tldr] The five-bullet version
> - **The problem:** LLMs cannot read time series. Serializing numbers into text is lossy and ~16× more expensive in tokens; rendering to a plot throws away the actual values. Both proxies cap how well a model can analyze a series.
> - **The fix:** ChatTS is a TS-MLLM — time series is a *native modality*. A context-aware encoder patches each series, encodes patches with a 5-layer MLP into the text-embedding space, and **interleaves** the resulting tokens with text at `<ts>` markers, preserving both the numbers and their textual context.
> - **The data:** Trained *only* on synthetic data via two methods — **attribute-based generation** (assemble each series from sampled attributes so the label is exact by construction) and **Time Series Evol-Instruct** (evolve seed QA along reasoning and situation axes, with an attribute-based eliminator keeping every QA grounded).
> - **The result:** Built on `Qwen2.5-14B-Instruct`, ChatTS beats GPT-4o (both text-serialized and vision-of-plots) by roughly **+46% on alignment** and **+25.8% on reasoning**, at **~1/16 the token cost** ($0.02 vs $3.25 on a representative task batch).
> - **Where it's weak:** The whole thing rests on the *synthetic* attribute taxonomy being a faithful model of real series; the encoder normalizes magnitude into prompt tokens (a workaround, not a principle); and "value-preserving" is relative, not exact.

This is one entry in a longer reading of [ByteDance Research's open model releases](/blog/machine-learning/bytedance-research-model-atlas); its closest sibling is [Timer-S1](/blog/machine-learning/signal-processing/timer-s1-time-series-foundation-model), which attacks the *forecasting* side of time series while ChatTS attacks *understanding and reasoning*. If you have read the speech-modality posts on [Orpheus TTS](/blog/machine-learning/signal-processing/orpheus-tts-llm-speech-snac) and [distilling non-verbal TTS](/blog/machine-learning/signal-processing/distilling-fast-tts-non-verbal), the structural rhyme will be obvious: every one of these is the same trick — take a non-text modality, give it an encoder that lands in token space, and let a language model do the reasoning. ChatTS is that trick applied to the one modality where precision is non-negotiable.

## 1. The real problem: an LLM cannot see a number it was never given

Let us be precise about what fails, because the failure modes are different for the two prior approaches and the differences matter for understanding why ChatTS is built the way it is.

### Approach one: serialize the numbers into text

The most obvious thing to do with a time series and a language model is to print the series as a string and put it in the prompt: `"The series is: 0.13, 0.14, 0.12, 0.88, 0.13, ..."`. This is what most "LLM + time series" demos do, and it has three concrete problems.

The first is **token economy**. A floating-point number rendered to a few significant figures is multiple BPE tokens. A series of length 1024 with, say, four characters of precision each, plus separators, is on the order of three to five thousand tokens — for *one* univariate series. The ChatTS paper measures this directly: on their real-world evaluation set (Dataset A), the text-serialized GPT-4o approach consumed **1.3M tokens at a cost of $3.25**, versus ChatTS's **0.08M tokens at $0.02**. That is a ~16× token blowup and a ~160× cost blowup, and it gets multiplicatively worse the moment you have multivariate input, because now you are pasting `N` series into one context window.

The second is **tokenizer mangling**. BPE tokenizers were trained on web text, not on number streams. The number `0.88` might be one token in one position and split as `0` / `.` / `88` in another, depending on surrounding bytes. The model has to learn to reassemble a magnitude from a non-deterministic token decomposition, and it has to do this *while also* tracking position (is this the 314th number or the 315th?). Anyone who has watched a frontier model miscount items in a long list knows how this ends: the model loses count, and the index it reports for the anomaly is off by an amount proportional to the series length.

The third is **no inductive bias for shape**. The text channel has no notion that adjacent numbers are adjacent in *time*. "Local shape" — a spike, a dip, a level shift — is exactly the kind of contiguous-window pattern that a convolution or a patch encoder captures trivially and that a flat token stream captures only by brute-force memorization. The ChatTS ablation makes this concrete: models trained on naively LLM-generated data "performed significantly worse, particularly for local fluctuation detection and numerical analysis." Local structure is where the text proxy bleeds the most.

### Approach two: render the numbers as a plot

The second move is to render the series as a line chart and feed the PNG to a vision-language model. This is what "GPT-4o (Vision)" means in the ChatTS tables. It fixes the token-economy problem (an image is a fixed token budget regardless of series length) and it gives the model a real spatial inductive bias for shape. But it introduces a worse problem: **the numbers are gone**.

A rendered plot is a quantization of the series into pixels. A 1024-point series squeezed into an 800-pixel-wide axis means each pixel column averages over more than one sample; a spike narrower than a pixel column can vanish entirely. And critically, the model can read the *shape* but not the *value* — it can tell you "there is a spike around the middle" but not "the spike reaches 0.88 against a baseline of 0.13," because that information was never encoded. On the alignment benchmark, GPT-4o-vision scores **0.609 overall categorical F1 and 0.436 numerical** — respectable on shape, weak on numbers. The numerical column is where vision collapses, exactly as the mechanism predicts.

![Before/after: serialized-text and rendered-plot proxies both destroy information that ChatTS's native value-preserving encoder keeps](/imgs/blogs/chatts-aligning-time-series-llms-2.webp)

The figure above is the comparison that motivates the architecture. On the left, the two proxies: serialize-to-digits (1.3M tokens, $3.25, local shape mangled by the tokenizer) and render-to-plot (values lost to pixel quantization, vision MLLM stuck at 0.609 alignment F1). On the right, the native path: a value-preserving patch encoder, 0.08M tokens at $0.02, and 0.889 alignment F1 — a relative +46% over GPT-4o. The two left boxes are red because they *lose information*; the right column is green because it does not. The whole rest of this post is an explanation of how the right column is built.

### Why this is a modality problem, not a scaling problem

The tempting counterargument is "this is just a small-model problem; GPT-5-class models will read serialized numbers fine." The ChatTS ablation closes that door. They test `Qwen2.5-32B-Instruct` — more than twice the parameters of ChatTS's 14B backbone — as a *text-only* baseline, and it "still does not outperform ChatTS (14B)." Scaling the language model does not fix a modality that was never given to it properly. The 32B model is bigger; it is not better at reading numbers it received through a lossy channel. This is the same lesson the vision community learned years ago: you do not get a good image model by describing images in words to a larger LLM; you get one by building an image encoder. ChatTS is that argument, transposed to time series.

| Approach | Token cost (Dataset A) | Numbers preserved? | Local shape? | Align F1 (overall) | Numerical acc |
|---|---|---|---|---|---|
| Text-serialized (GPT-4o) | 1.3M / $3.25 | Mangled by BPE | Weak | 0.542 | 0.371 |
| Vision-of-plot (GPT-4o) | 0.13M / $0.32 | **Lost** | Good | 0.609 | 0.436 |
| Agent + classic TS tools (ReAct) | — | Partial | Tool-dependent | ~0.498 | ~0.412 |
| Text-only Qwen2.5-32B | high | Mangled | Weak | < ChatTS | < ChatTS |
| **ChatTS-14B (native TS modality)** | **0.08M / $0.02** | **Encoded** | **Encoded** | **0.889** | **0.788** |

The numbers in that table are the spine of the whole paper. Everything else — the encoder design, the synthetic data pipeline, the interleaving scheme — exists to make the last row possible.

### Why AIOps is the forcing function

It is no accident that ChatTS comes out of ByteDance's NetManAIOps group rather than a general-purpose ML lab. AIOps — the practice of operating large infrastructure with the help of ML — is the domain where every one of the failures above is a daily, expensive problem. A company running tens of thousands of hosts emits *millions* of metric series: CPU, memory, disk, network, per-service latencies, queue depths, error rates. When something breaks, an on-call engineer has minutes to figure out *which* of those series is the cause, and the current state of the art is "stare at dashboards." The dream is a system that reads every series, in natural language, and surfaces the handful that matter with an explanation a human can act on.

That dream has three constraints that map exactly onto ChatTS's design choices. First, **the data cannot leave the building** — production telemetry is sensitive, so an API-only model like GPT-4o is often a non-starter, which is why a self-hostable open model matters. Second, **the volume is enormous**, so per-series cost has to be tiny — which is why the 160× token-cost reduction is not a nice-to-have but a feasibility threshold. Third, **the questions are multivariate and contextual** — "did these three metrics move together, and what does that imply" — which is why the interleaving design that binds each series to its name in the prompt is the whole game. ChatTS reads as a general contribution, but it is shaped end to end by the operational problem it was built to solve, and that shaping is exactly why its synthetic taxonomy is so well-matched to server metrics and so untested on, say, financial series.

## 2. The architecture: a thin encoder under a standard LLM

The first thing to internalize about ChatTS's architecture is how *little* of it is new. The backbone is `Qwen2.5-14B-Instruct`, unmodified. There is no exotic attention variant, no time-aware positional scheme baked into the transformer, no new training objective. The entire contribution on the model side is two thin trainable layers below the LLM — a patch encoder and a projector — plus a normalization scheme that happens partly in the prompt text. This is deliberate: by keeping the LLM standard, ChatTS inherits all of Qwen2.5's instruction-following and reasoning ability for free, and spends its novelty budget entirely on getting time series *into* that ability.

![ChatTS model stack from bottom to top: value-preserving normalization, patching, 5-layer MLP encoder, projector, Qwen2.5-14B backbone, LM head](/imgs/blogs/chatts-aligning-time-series-llms-6.webp)

The stack above reads bottom-to-top. At the base, **min-max normalization** scales each series into `[0, 1]` and emits its scaling factor and offset as **text tokens** in the prompt. Above that, **patching** splits each series into fixed-size patches. The **5-layer MLP encoder** maps each patch to an embedding. The **projector** aligns those embeddings to the LLM's text-token space. The **Qwen2.5-14B-Instruct backbone** processes the interleaved sequence with full-parameter SFT (no LoRA — the whole model is fine-tuned). The **LM head** emits a text answer. Only the two blue layers (encoder, projector) are architecturally new; the green head is the LLM's own. Let us take each piece in turn.

### 2.1 Patching: borrowing PatchTST's one good idea

ChatTS divides each input series into fixed-size, non-overlapping patches. This is the same move PatchTST made and that every modern TS model — including its sibling [Timer-S1](/blog/machine-learning/signal-processing/timer-s1-time-series-foundation-model) — now uses: a patch is the time-series analogue of a token. Patching does two things at once. First, it gives the model a *local receptive field* for free — a patch of, say, 16 consecutive samples captures a small window of shape (the rising edge of a spike, a half-period of seasonality) as a single unit, so the encoder is reasoning about windows rather than isolated points. Second, it slashes sequence length: a 1024-point series at patch size 16 becomes 64 patch tokens instead of 1024 point tokens, which is what keeps the interleaved sequence short enough to be cheap.

The paper does not pin a single global patch size in the text (it defers exact constants to the released code, where sequence length and patch parameters are config knobs — the default generation length is 256), but the operative range is clear: ChatTS trains on series with **lengths from 64 to 1024**, and supports up to **30 series per prompt**. The patch count per series is therefore `ceil(L / patch_size)`, and the *total* TS-token budget for a prompt is the sum over all series — which is why the token economy stays favorable even with multivariate input.

The patch size is also the central tuning tension, and it is worth understanding why. A *smaller* patch means more patches per series, finer temporal resolution (a single-sample spike is more likely to dominate its patch and stay visible), but a longer token sequence and more cost. A *larger* patch means fewer tokens and cheaper inference, but a narrow spike can get averaged into a wider window and lose prominence — the same quantization failure the rendered-plot proxy suffers, only milder. The whole reason patching works at all is that this tension lands in a sweet spot for operational metrics: events that matter (a level shift, a sustained dip, a periodic pattern) are usually several samples wide, so a moderate patch size captures them as coherent windows without blowing up the token budget. The pathological case — a true single-sample outlier — is exactly the regime where you would want a smaller patch, and it is one of the knobs a deployment would tune to its own anomaly profile.

### 2.2 The 5-layer MLP encoder: simpler than you expect

Here is the part that surprises people: the per-patch encoder is **a 5-layer MLP**. Not a transformer, not a TCN, not a state-space model — a multilayer perceptron. The paper is explicit that "a simple structure can map the patch features to a space aligned with the text embedding." Each patch (a vector of `patch_size` real values) goes through the MLP and comes out as a single embedding vector of the LLM's hidden width.

Why does something this simple work? Because the patch is already a good representation. Patching has done the hard part — turning a long, position-sensitive stream into a stack of fixed-width windows — and the MLP only has to learn a per-window feature map into embedding space. The transformer that *does* the heavy lifting of relating patches across time is the LLM itself; the encoder does not need to re-derive temporal attention because the patch embeddings are about to be processed by 14B parameters of attention anyway. This is the same division of labor that lets vision-language models use a relatively shallow projector on top of a frozen-ish ViT: don't duplicate the reasoning machinery in the adapter.

Here is a faithful sketch of the encoder in PyTorch. This is not the exact released code, but it is the right shape — patch, normalize, MLP, and the value-preserving bookkeeping that turns into prompt tokens:

```python
import torch
import torch.nn as nn


class PatchTSEncoder(nn.Module):
    """Context-aware, value-preserving time-series patch encoder.

    Maps a single 1-D series of arbitrary length L into a stack of
    patch embeddings that live in the LLM's hidden space, plus the
    (scale, offset) pair that the prompt carries as text so the
    original magnitude is recoverable.
    """

    def __init__(self, patch_size: int = 16, hidden: int = 5120, depth: int = 5):
        super().__init__()
        self.patch_size = patch_size
        # 5-layer MLP: patch_size -> hidden, with GELU between layers.
        dims = [patch_size] + [hidden] * (depth - 1) + [hidden]
        layers = []
        for i in range(depth):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < depth - 1:
                layers.append(nn.GELU())
        self.mlp = nn.Sequential(*layers)

    def normalize(self, x: torch.Tensor):
        """Min-max into [0, 1]; return values + recoverable (scale, offset)."""
        lo = x.min()
        hi = x.max()
        scale = (hi - lo).clamp_min(1e-8)          # value scaling factor
        offset = lo                                 # value offset
        x_norm = (x - offset) / scale
        return x_norm, scale, offset

    def patchify(self, x_norm: torch.Tensor) -> torch.Tensor:
        """[L] -> [num_patches, patch_size] with right-padding."""
        L = x_norm.shape[0]
        pad = (-L) % self.patch_size
        if pad:
            x_norm = torch.cat([x_norm, x_norm.new_zeros(pad)])
        return x_norm.view(-1, self.patch_size)     # [num_patches, patch_size]

    def forward(self, series: torch.Tensor):
        x_norm, scale, offset = self.normalize(series)
        patches = self.patchify(x_norm)             # [P, patch_size]
        embeds = self.mlp(patches)                  # [P, hidden]
        # The (scale, offset) are NOT fed through the MLP. They are emitted
        # as text in the prompt, e.g. "scale=0.75, offset=0.13", so the LLM
        # can read the patch shape AND recover the true magnitude.
        return embeds, float(scale), float(offset)
```

The two return paths in `forward` are the entire value-preservation story, and they deserve their own section.

### 2.3 Value preservation: min-max normalization plus text-channel magnitude

This is the cleverest and, in my opinion, the most quietly debatable design choice in the paper. A naive encoder would standardize each series (subtract mean, divide by standard deviation) and feed only the shape, because neural nets train better on normalized inputs. But standardization *destroys the magnitude* — and magnitude is half the question. "Is CPU at 88% or 8.8%?" is not answerable from shape alone.

ChatTS's answer is to **min-max normalize into `[0, 1]` for the encoder** (so the MLP sees well-conditioned inputs) but to **carry the scaling factor and offset as text tokens in the prompt**. The prompt literally contains the value-scaling and value-offset numbers, and the LLM — which is genuinely good at arithmetic over small numbers of digits — is expected to recombine "the patch shape says this point is at 0.92 of the normalized range" with "scale is 0.75, offset is 0.13" to recover "the true value is `0.13 + 0.92 × 0.75 ≈ 0.82`." The paper frames this as "leveraging the numerical understanding capabilities of LLMs."

It works — the numerical accuracy column (0.788 for ChatTS vs 0.436 for vision GPT-4o) is the proof — but notice what it *is*: a hybrid. The shape lives in the encoded modality; the absolute scale lives in the text channel. It is value-*preserving* in the sense that the information is recoverable, not in the sense that the encoder itself carries calibrated magnitudes. That distinction will matter in the Critique section.

### 2.4 Interleaving: keeping the context, not just the curve

The last architectural piece is how the TS tokens get into the sequence, and it is the part that earns the "context-aware" in "context-aware encoder." Naively, you could prepend all the TS embeddings, then all the text. ChatTS does not. It uses **token-level concatenation based on the position of each series in the original input**: the encoded patches for series `k` are inserted *between the surrounding text tokens*, at the exact spot in the prompt where the text refers to that series.

Concretely, the prompt uses a `<ts><ts/>` placeholder: `"I have time series 1: <ts><ts/> and time series 2: <ts><ts/>. Do both stay steady?"`. The processor replaces each placeholder with that series' patch embeddings in place. The result is a single sequence where "time series 1" is *immediately followed by* the actual encoded series 1, then "and time series 2" is followed by encoded series 2, then the question.

![Variable variates and lengths fan into per-series patch tokens that splice at their text markers into one interleaved sequence](/imgs/blogs/chatts-aligning-time-series-llms-8.webp)

The graph above shows why this is the right call for *multivariate* input, where it matters most. Series 1 (length 100), series 2 (length 150), through series N (up to 30 of them, any lengths) are each patched and encoded independently into `k1`, `k2`, ... `kn` tokens. The prompt text, with its `<ts>` markers, is spliced together with those token groups so each series lands at its own marker. The output is one sequence the LLM reads left to right. The dashed control arrow from "prompt text" is the splicing operation that decides *where* each group goes. Because the binding between "the words *time series 1*" and "the numbers of series 1" is positional and immediate, the model never has to guess which curve a sentence is talking about — the paper notes this is "particularly critical for multivariate scenarios where referencing the corresponding time series in textual form is often necessary." Try doing that with serialized numbers and you get a wall of digits with no clean handle for the prompt to point at.

This also explains the "arbitrary length and quantity" claim. Nothing in the encoder or the interleaving cares how many series there are or how long each is — each series is independently patched and encoded, and the LLM's own context window is the only real limit. In practice the release caps it at 30 series per prompt and 64–1024 points each, but that is a practical budget, not an architectural ceiling.

### 2.5 The training curriculum: alignment first, then reasoning

The architecture is only half the story; the *order* in which you teach the model matters as much as what you teach it. ChatTS uses a two-stage curriculum that mirrors how the two synthetic data methods divide the labor, and the ordering is not arbitrary — it encodes a dependency.

**Stage one is alignment.** The model is fine-tuned on the three attribute-grounded datasets (UTS, MTS-Shape, MTS-Local, 35k each) whose sole job is to teach *perception*: given a series, name its trend, its seasonality, its local fluctuations, and their magnitudes. This is where the encoder and projector learn to land patch embeddings in a part of the LLM's representation space that the LLM can actually read as "a series with these properties." A small amount of instruction-following (IF) data is mixed in even at this stage, because a model trained on nothing but "describe this series" templates rapidly forgets how to follow general instructions — the IF data is a regularizer against template collapse.

**Stage two is SFT** on the evolved reasoning data (TSEvol, 24,270 examples) plus a larger slice of explicit instruction-following data (5,050 examples). Critically, the paper mixes roughly **30% of the alignment dataset back into the SFT stage** to "reduce overfitting" — that is, to keep the perception sharp while the model learns to reason. Without that mix-back, a known failure pattern emerges: the model gets better at *talking about* series in the abstract but quietly loses the precise numerical perception it learned in stage one, because the reasoning data does not reinforce exact attribute reading. The 30% mix is a continual-learning trick — rehearse the old task while learning the new one.

The whole curriculum trains with **full-parameter SFT** (not LoRA or any parameter-efficient method) on `8×A800` GPUs via DeepSpeed and LLaMA-Factory. The choice of full-parameter fine-tuning is worth noting: because the encoder is producing a genuinely new modality of input, the LLM's lower layers need to adapt to read it, and freezing them (as LoRA effectively does for most of the weight mass) leaves too little capacity to integrate the new modality. This is the same reason vision-language models that aim for strong perception eventually unfreeze the LLM rather than relying on a frozen backbone plus a trained projector — the projector alone cannot teach a frozen LLM to read a modality it was never pretrained on.

| Curriculum stage | Data | Objective | Why this order |
|---|---|---|---|
| 1. Alignment | UTS + MTS-Shape + MTS-Local (105k) + IF | Learn to perceive attributes | Reasoning is impossible without perception |
| 2. SFT | TSEvol (24.3k) + IF (5k) + 30% alignment rehearsal | Learn to reason; retain perception | Mix-back prevents perception drift |

The dependency is the point: you cannot reason about a series you cannot read, so perception must come first, and it must be *retained* through the reasoning stage. The two-stage curriculum with rehearsal is how ChatTS enforces that ordering.

## 3. Why synthetic data, and why it is not a cop-out

Here is the bind that every "LLM understands time series" project hits and that kills most of them: **there is no labeled data**. To teach a model to say "this series has an upward trend with weekly seasonality and a spike at index 314," you need a corpus of series each annotated with its trend, seasonality, anomalies, and so on. That corpus does not exist at scale, and you cannot crowdsource it — ask ten annotators to label the "trend strength" of a noisy series and you get ten different numbers. Real-world time series come with *outcomes* (the server crashed) but almost never with *attribute labels* (the series had a level shift of magnitude 0.4 at t=880).

ChatTS's response is the move that makes the whole paper work: if you cannot find labeled data, **generate it from the labels**. Instead of generating series and then labeling them (hard, noisy, expensive), generate the *labels* first — sample a set of attributes — and then synthesize a series that provably has exactly those attributes. The label is correct by construction, because the label *is* the generation recipe. This inverts the usual data problem and it is why ChatTS trains *exclusively* on synthetic data and still beats GPT-4o on real-world evaluation series.

There are two synthetic methods, and they target the two halves of the task. **Attribute-based generation** produces the *alignment* data — series paired with descriptions of their attributes, which teaches the model to perceive trend/seasonality/anomaly accurately. **Time Series Evol-Instruct (TSEvol)** produces the *reasoning* data — diverse, evolved question-answer pairs that teach the model to do something *with* what it perceives. The training set is small by LLM standards — roughly 134k examples total, all synthetic — and is split across alignment and SFT stages.

| Stage | Dataset | # Samples | What it teaches |
|---|---|---|---|
| Alignment | UTS (univariate) | 35,000 | Perceive single-series attributes |
| Alignment | MTS-Shape | 35,000 | Perceive multivariate shape |
| Alignment | MTS-Local | 35,000 | Perceive local fluctuations / anomalies |
| SFT | TSEvol | 24,270 | Reason over series (diverse QA) |
| SFT | Instruct-Follow | 5,050 | Keep instruction-following sharp |

The split is the design. Alignment first, on attribute-grounded data, to nail perception; then SFT on evolved QA, with a slice of alignment data mixed back in (the paper mixes ~30% of the alignment set into SFT "to reduce overfitting"). The whole thing trains with full-parameter SFT on `8×A800` GPUs using DeepSpeed and LLaMA-Factory. Let us take the two generation methods one at a time.

## 4. Attribute-based generation: the label space is the taxonomy

The attribute-based generator is best understood as a small, well-typed DSL for time series. There is a fixed **attribute set** — the paper's "All Attribute Set" — that fully describes any series the generator can produce. You sample a point in that attribute space, and a rule-based generator deterministically renders the series. Because the attributes *are* the label, every generated series ships with a perfect, machine-checkable ground truth.

![The synthetic attribute taxonomy: four families — trend, seasonality, noise, local fluctuation — whose sub-attributes form the exact label space](/imgs/blogs/chatts-aligning-time-series-llms-3.webp)

The tree above is the taxonomy, and it doubles as the label space. There are four families. **Trend** (4 types: increase, decrease, flat, and their parameterizations) sets the global drift. **Seasonality** (7 types) sets periodic structure, parameterized by period and amplitude. **Noise** (3 types) sets the stochastic floor — Gaussian and its levels. And **local fluctuation** (19 types) — the family in amber because it is the hardest and the most valuable — covers the transient, localized events: spikes, dips, level shifts, sudden increases and decreases, wide fluctuations, and so on. That is 33 attribute types total. The paper itself defers the full enumeration of the 19 local-fluctuation types to the released source code; the representative ones (spike, dip, level shift, sudden up/down, wide fluctuation) are the ones operators actually care about, because those are the anomalies.

### 4.1 The three-stage generation pipeline

The generator is not "sample random attributes and render," which would produce physically nonsensical series. It is a three-stage pipeline that grounds the synthesis in *real metric semantics*.

![Attribute-based generation: a metric name conditions a plausible attribute subset, the sampler draws values under constraints, and a rule-based generator emits a labeled series](/imgs/blogs/chatts-aligning-time-series-llms-7.webp)

The pipeline above is the data factory. First, an **attribute selector**: there is a pool of **567 real-world metric names** (`cpu.usage`, `request.latency_p99`, `disk.io_wait`, and the like), and an LLM is asked to choose a *physically plausible* attribute subset for that metric. This is the step that keeps the data realistic — a CPU-usage metric gets bounded-in-`[0,100]` attributes and plausible daily seasonality, not a metric that goes negative or oscillates at 1 Hz. Second, an **attribute sampler** draws the concrete numbers: where the spike occurs, its amplitude, the seasonal period and amplitude, the trend slope — "based on rules and constraints," so the values stay physically sane. Third, a **rule-based generator** renders an array that exactly matches the sampled attributes. The selector is lavender (external — it calls an LLM), the constraints box is amber (it is the guardrail), and the output series is green because it arrives with an exact label attached.

The payoff line from the paper: this can "theoretically generate an infinite number of different time series." There is no labeling bottleneck because there is no labeling step.

The reason the metric-conditioning step matters so much is subtle and worth dwelling on. A purely random attribute sample produces series that are *valid* (the label is still exact) but *implausible* — a series that simultaneously has a steep linear trend, three different seasonal periods, and a level shift every ten points is a perfectly well-defined synthetic object that looks like nothing in production. Train on enough of those and the model learns a distribution of series that does not match reality, and the synthetic-to-real transfer collapses. The LLM-driven selector is the guard against this: by asking "what attributes are physically plausible for `cpu.usage`?" before sampling, the generator stays inside the manifold of *realistic* series. This is why the 567 real metric names are not decoration — they are the anchor that keeps the infinite synthetic space tethered to the finite real one. The constraints box in the pipeline figure is the second half of the same guard: even within a plausible attribute subset, the *values* (slope magnitude, spike amplitude, seasonal period) are sampled under physical bounds so a CPU metric never exceeds 100% and a latency never goes negative. Plausibility is enforced twice — once at attribute selection, once at value sampling — and that double guard is what makes "train on synthetic, deploy on real" survive contact with reality.

### 4.2 A runnable attribute-based generator

Here is a compact but faithful implementation of the core idea — sample attributes, render a series, and get the label for free. This is the skeleton the real generator elaborates with 33 attribute types and metric-conditioned selection:

```python
import numpy as np


def generate_series(L=512, rng=None, attrs=None):
    """Render a series from sampled attributes; return (series, label).

    The label is exact by construction: it IS the recipe.
    """
    rng = rng or np.random.default_rng()
    t = np.arange(L)
    attrs = attrs or {}

    # --- TREND (4 types) ---
    trend_type = attrs.get("trend", rng.choice(["increase", "decrease", "flat"]))
    slope = {"increase": 1.0, "decrease": -1.0, "flat": 0.0}[trend_type]
    slope *= attrs.get("trend_strength", rng.uniform(0.0005, 0.003))
    series = slope * t

    # --- SEASONALITY (7 types; here: sinusoidal) ---
    if attrs.get("seasonal", rng.random() < 0.7):
        period = attrs.get("period", int(rng.choice([24, 48, 96, 168])))
        amp = attrs.get("amplitude", rng.uniform(0.2, 1.0))
        series = series + amp * np.sin(2 * np.pi * t / period)
        season_label = {"period": period, "amplitude": round(amp, 3)}
    else:
        season_label = None

    # --- NOISE (3 types; here: gaussian) ---
    noise_level = attrs.get("noise", rng.uniform(0.01, 0.15))
    series = series + rng.normal(0, noise_level, size=L)

    # --- LOCAL FLUCTUATION (19 types; here: spike / dip / level shift) ---
    local_label = None
    if attrs.get("local", rng.random() < 0.6):
        kind = attrs.get("local_type", rng.choice(["spike", "dip", "level_shift"]))
        pos = attrs.get("local_pos", int(rng.integers(L // 5, 4 * L // 5)))
        mag = attrs.get("local_mag", rng.uniform(1.0, 3.0))
        if kind == "spike":
            series[pos] += mag
        elif kind == "dip":
            series[pos] -= mag
        elif kind == "level_shift":
            series[pos:] += mag           # permanent shift, not transient
        local_label = {"type": kind, "position": pos, "magnitude": round(mag, 3)}

    # The label is exact because we built the series FROM it.
    label = {
        "trend": trend_type,
        "seasonality": season_label,
        "noise_level": round(noise_level, 3),
        "local_fluctuation": local_label,
    }
    return series.astype(np.float32), label


series, label = generate_series(L=512)
print(label)
#: {'trend': 'increase',
#:  'seasonality': {'period': 96, 'amplitude': 0.71},
#:  'noise_level': 0.083,
#:  'local_fluctuation': {'type': 'level_shift',
#:                        'position': 311, 'magnitude': 2.04}}
```

Stare at that for a second, because it is the whole insight. The `label` dict is not produced by *looking at* `series` — it is the *input* to producing `series`. There is no annotation error possible, because there is no annotation. When you then write a training example "Describe this series: `<ts><ts/>`" with the target answer "It has an upward trend, daily seasonality (period 96, amplitude 0.71), and a level shift of magnitude 2.04 at index 311," the supervision is perfect. Train enough of these and the model learns to perceive each attribute, because every attribute was independently varied with a known answer.

### 4.3 The trend sub-attribute and why it matters

To make this concrete, consider just the trend family. The four trend types are not "up / down" — they are parameterized: an increasing trend has a slope, a decreasing trend has a slope, a flat trend has slope zero, and the strength is sampled. When the model sees ten thousand series with sampled trend slopes and ten thousand correct answers about those slopes, it learns to *estimate* slope from shape, not just classify direction. That is why ChatTS's trend categorical F1 hits **0.927** (Dataset A) — it is not guessing "up or down," it has learned a regressor over slope that it expresses categorically. The same logic applies family by family: seasonality F1 **0.973**, local fluctuation **0.895**. Each is high precisely because each attribute was an independently varied, perfectly labeled axis in training.

| Attribute family | # types | Sampled parameters | ChatTS Cate. F1 (Dataset A) |
|---|---|---|---|
| Trend | 4 | direction, slope/strength | 0.927 |
| Seasonality | 7 | period, amplitude | 0.973 |
| Local fluctuation | 19 | type, position, magnitude | 0.895 |
| Overall | 33 | — | **0.889** |

## 5. Time Series Evol-Instruct: from perception to reasoning

Alignment data teaches the model to *perceive*. It does not teach the model to *reason*, because the alignment QA is templated — "what is the trend," "where is the anomaly" — and a model trained only on templates answers only templated questions. Real users ask open-ended, compositional, situational questions: "Two of these three metrics spiked together at the same time — what does that suggest?" or "Given this latency pattern, what would you check first?" To get there, ChatTS borrows the **Evol-Instruct** idea from WizardLM and adapts it to time series.

The base Evol-Instruct recipe is: start with seed instructions and have an LLM *evolve* each one into something harder or broader — deepen it (add reasoning steps, add constraints, increase specificity) or broaden it (jump to a related but distinct topic). Iterate, and the instruction set grows in diversity and difficulty without a human writing each example. ChatTS's twist — **TSEvol** — is to make the evolution *attribute-aware*: every evolution step can pull a fresh attribute subset from the pool as additional context, so the evolved questions cover a *broader* set of time-series attributes than the seeds did, and it adds two new evolution *types* tailored to the domain — **reasoning** questions and **situation** questions — on top of the standard deepen/broaden operators.

![Time Series Evol-Instruct loop: seed QA is evolved in depth and breadth, conditioned on sampled attributes, then filtered by an attribute-based eliminator](/imgs/blogs/chatts-aligning-time-series-llms-4.webp)

The graph above is the TSEvol loop. The **seed QA** comes from the alignment dataset plus some initial LLM-generated QAs. Each seed is evolved two ways: **in-depth** (make it harder — more reasoning, tighter constraints) and **in-breadth** (jump to new topics). Both feed into the domain-specific **reasoning + situation** evolution types. A **sampled attribute subset** is injected as context (the dashed control arrow), so the rewrite is grounded in real attributes rather than hallucinated ones. An LLM **rewrites** the QA. And then — this is the part that keeps the whole thing honest — an **attribute-based eliminator** checks every generated QA against the actual generating attributes and **drops any QA that contradicts them**. The survivors form the **evolved QA pool of 24,270** examples.

That eliminator is the load-bearing wall. The failure mode of any LLM-generated training data is that the generator hallucinates — it writes a plausible-sounding answer that is wrong about the series. Because ChatTS *knows the true attributes* of every series (they were sampled, not annotated), it can mechanically verify that the QA is consistent: if the answer claims "there is a downward trend" but the generating attribute set says the trend is increasing, that QA is eliminated. This is the same correctness-by-construction guarantee from the attribute generator, now applied as a *filter* on reasoning data. It is why TSEvol can use an LLM to write reasoning QA without the usual "synthetic data poisons the model with its own errors" problem.

The ablation confirms the division of labor: removing TSEvol causes a large drop in *reasoning* and only a modest drop in *alignment*, exactly as you would predict — TSEvol's whole job is reasoning diversity. Removing the attribute-based generation, conversely, tanks *alignment* (especially local-fluctuation detection and numerical analysis). The two methods are not redundant; each owns one half of the capability.

### 5.1 What "reasoning" and "situation" questions look like

A **reasoning** question is one that requires combining multiple perceived attributes or inferring a cause: "Series A trends up while series B trends down with the same period — are they anti-correlated, and over what window?" A **situation** question grounds the series in a scenario: "This is `disk.io_wait` on a database host; the value held flat then stepped up by 0.4 and stayed there. What kind of event does a permanent level shift in IO wait usually indicate?" These are the questions that make the difference between a model that can *label* a series and a model an SRE would actually consult. The reasoning benchmark — inductive, deductive, causal, and multiple-choice — is built to test exactly this, and it is where ChatTS's +25.8% average gain over baselines lives.

### 5.2 The correlation pool: how TSEvol learns multivariate relationships

The single hardest reasoning skill for a time-series model is *cross-series* reasoning — noticing that two metrics moved together, or that one leads another. Templated alignment data cannot teach this, because alignment is per-series perception. TSEvol's mechanism for it is a **correlation pool**: a structure that records sets of series with *related* attributes (e.g., two series that share a spike at the same position, or a leading/lagging seasonal pair). When TSEvol evolves a multivariate QA, it draws from the correlation pool so the question can be *about* the relationship, and — because the relationship was constructed, not observed — the correct answer is again known by construction.

This is the piece that makes the multivariate-correlation case study (§7.2) work at all. Without a correlation pool, the model would see many individual series during training but very few *examples of relationships between series*, and it would learn to describe each curve in isolation while being unable to say "these two are correlated." The correlation pool is the deliberate injection of relational training signal, and it is why ChatTS can answer "which of these thirty metrics moved together" rather than just thirty separate descriptions.

### 5.3 A sketch of the evolution operators

Here is the shape of the TSEvol loop in pseudocode-flavored Python, faithful to the described mechanism — note that every evolved QA is gated by the attribute-consistency eliminator before it can enter the pool:

```python
def tsevol_step(seed_qa, attribute_pool, correlation_pool, llm, checker):
    """One evolution step: deepen or broaden a seed QA, grounded in
    sampled attributes, then keep it only if it stays consistent."""
    # Pull fresh attribute context so the rewrite covers broader attributes.
    attrs = attribute_pool.sample_subset()
    if seed_qa.is_multivariate:
        attrs = correlation_pool.augment(attrs)   # relational grounding

    # Choose an evolution operator (the WizardLM-style menu, plus
    # the two TS-specific types added by ChatTS).
    op = random.choice([
        "in_depth",     # add reasoning steps / constraints; make it harder
        "in_breadth",   # jump to a related but distinct question
        "reasoning",    # TS-specific: require inference over attributes
        "situation",    # TS-specific: ground in a real scenario
    ])

    evolved = llm.rewrite(seed_qa, operator=op, attribute_context=attrs)

    # The load-bearing filter: drop any QA whose answer contradicts the
    # KNOWN generating attributes. No hallucinated supervision survives.
    if checker.is_consistent(evolved, attrs):
        return evolved
    return None          # eliminated; try again
```

The `checker.is_consistent` call is what separates TSEvol from "ask GPT-4 to write training data," which is the standard recipe that quietly poisons models with the generator's own errors. Because the true attributes are known for every series, consistency is a *mechanical* check, not a judgment call — and that mechanical check is the entire reason a fully-synthetic, LLM-evolved reasoning corpus does not degrade the model.

## 6. Experiments: the modality, not the backbone, is doing the work

Now the results, in detail, because the headline numbers are only convincing if you know what they are measured against.

![Results matrix: ChatTS-14B leads every task group over text GPT-4o, vision GPT-4o, agent, and a same-size Qwen2.5 baseline](/imgs/blogs/chatts-aligning-time-series-llms-5.webp)

The matrix above is the result in one frame: rows are methods, columns are task groups (alignment categorical F1, alignment numerical accuracy, reasoning), and ChatTS's row is uniformly green while the baselines are amber-to-red. Read it top to bottom and the story is "the native-modality model wins every column," which is exactly the claim the architecture predicts.

### 6.1 The two benchmarks

ChatTS evaluates on two datasets. **Dataset A** is *real-world* — actual production series with human-validated attribute labels — and it is the one that matters for the "trained on synthetic, works on real" claim. **Dataset B** is *synthetic* evaluation data held out from training, which stress-tests perception on cleanly labeled series. Tasks split into **alignment** (perceive attributes; scored as categorical F1 via rule-based matching and numerical accuracy as `max(1 - |answer - label| / |label|, 0)`) and **reasoning** (inductive, deductive, causal, and multiple-choice, scored with RAGAS-style LLM fuzzy matching and direct accuracy for MC/T&F).

### 6.2 Alignment results

| Task group | GPT-4o (Text) | GPT-4o (Vision) | **ChatTS-14B** |
|---|---|---|---|
| Trend (Cate. / Num.) | 0.585 / 0.882 | 0.659 / 0.613 | **0.927 / 0.874** |
| Seasonality (Cate. / Num.) | 0.811 / 0.768 | 0.811 / 0.559 | **0.973 / 0.849** |
| Local fluctuation (Cate. / Num.) | 0.379 / 0.256 | 0.537 / 0.414 | **0.895 / 0.805** |
| **Overall (Cate. / Num.)** | 0.542 / 0.371 | 0.609 / 0.436 | **0.889 / 0.788** |

The headline is the **local fluctuation** row. Text GPT-4o scores 0.379 categorical / 0.256 numerical; vision does a bit better on shape (0.537) but worse where it counts. ChatTS scores 0.895 / 0.805 — more than double on the numerical side. Local fluctuations are anomalies, and anomalies are the entire commercial reason for time-series understanding. The paper summarizes the alignment gains as "46.0%–75.9% improvement in categorical metrics and 80.7%–112.7% in numerical metrics compared to industry-leading models like GPT-4o." The ~+46% headline is the *low* end of the categorical range; the numerical improvements are nearly double.

### 6.3 Reasoning results

| Reasoning task | GPT-4o (Vision) | GPT-4o (Text) | **ChatTS-14B** |
|---|---|---|---|
| Inductive | 0.322 | 0.336 | **0.518** |
| Deductive | 0.605 | 0.628 | **0.744** |
| Causal | 0.652 | 0.685 | **0.804** |
| MCQ2 | 0.490 | 0.470 | **0.600** |
| **Average** | 0.517 | 0.530 | **0.667** |

The reasoning average goes from ~0.53 (best baseline) to 0.667 — a +25.8% relative improvement, with the largest single jump on **inductive** reasoning (+34.5%), which is the hardest: inferring a general rule from observed series behavior. Note that this is *reasoning over perception* — the model first has to read the series correctly (which the alignment results show it can) and then reason about it. A model that misperceives the series cannot reason about it correctly, so the reasoning gain is partly downstream of the alignment gain. This is the deepest argument for the native modality: better perception is a prerequisite for better reasoning, and you do not get better perception from a lossy proxy.

### 6.4 The cost dimension

The efficiency numbers are not a footnote; they change the deployment calculus entirely. On Dataset A, ChatTS used **0.08M tokens at $0.02**, against GPT-4o-vision's **0.13M / $0.32** and GPT-4o-text's **1.3M / $3.25**. ChatTS is a 14B open model you can run on your own A800s; the baselines are API calls. So ChatTS is simultaneously *more accurate*, *~16× cheaper in tokens than text-serialization*, and *self-hostable* — which for the kind of internal monitoring data that motivates this work (it is from ByteDance's NetManAIOps group) is often a hard requirement, because you cannot ship production telemetry to a third-party API.

### 6.5 The ablations that close the argument

Three ablations are worth stating because each rules out a "but maybe it's just X" objection:

- **Bigger LLM doesn't fix it.** A text-only `Qwen2.5-32B-Instruct` — 2.3× the parameters — does not match ChatTS-14B. Rules out "scale the backbone."
- **Drop the TS modality entirely (text-only ChatTS).** Overall alignment F1 collapses from **0.889 to 0.464**. Rules out "the synthetic SFT alone did it" — the modality is doing roughly half the work.
- **Drop attribute-based generation (use only LLM-generated data).** Alignment, especially local-fluctuation and numerical, drops sharply. Rules out "any synthetic data would do" — the *attribute grounding* matters.

Each ablation removes one ingredient and watches the relevant metric fall. Together they form the argument that all three pieces — native modality, attribute-grounded alignment data, evolved reasoning data — are load-bearing.

### 6.6 Reading the per-task breakdown: where the gap is widest

The aggregate +46%/+25.8% numbers hide the more interesting story, which is *where* the gap concentrates. Walk down the alignment table by task and the pattern is consistent: the gap between ChatTS and the baselines is *smallest* on the easy, global attributes and *largest* on the hard, local, numerical ones.

On **trend**, both GPT-4o variants are already decent (text 0.585 categorical, vision 0.659) — direction of drift is a coarse, global property that survives even a lossy proxy, so ChatTS's 0.927 is a real but not dramatic improvement. On **seasonality**, the baselines are surprisingly strong on the categorical side (both at 0.811) because periodicity is a salient visual pattern, but they fall apart on the *numerical* side (vision 0.559) because reading the exact period and amplitude off a plot is precisely the thing pixels cannot do — and ChatTS's 0.849 numerical there is the encoder paying off. On **local fluctuation**, the gap is a chasm: text GPT-4o manages 0.379 categorical / 0.256 numerical, vision 0.537 / 0.414, and ChatTS 0.895 / 0.805. Local fluctuations are short, contiguous, numerically-defined events — a spike of magnitude 2.04 at index 311 — which is the worst case for both proxies (the tokenizer mangles the local window; the plot quantizes the spike) and the best case for a patch encoder that sees the window as a unit and carries its value.

The lesson for a practitioner is that you should not read the headline number and assume uniform improvement. If your use case is "is this metric generally trending up," a frontier vision model is *already adequate* and the gain from a TS-MLLM is modest. If your use case is "exactly when and how big was the anomaly" — which is the use case that pays the bills in monitoring — the gain is the difference between unusable (0.256) and production-grade (0.805). The value of ChatTS is concentrated precisely in the questions that matter most operationally.

### 6.7 The synthetic-vs-real generalization gap

One number deserves singling out because it is the crux of the whole "trained on synthetic" bet: ChatTS's overall alignment is **0.862 on the synthetic Dataset B** and **0.889 on the real Dataset A**. The model does roughly as well on real series as on the synthetic series it was trained to perceive — if anything slightly *better* on real data, likely because the real evaluation series are drawn from the same operational-metric distribution the 567-name metric pool was sampled to mimic, and that distribution is in some ways gentler than the deliberately-hard synthetic stress set. The fact that there is no large synthetic-to-real degradation is the empirical heart of the paper: it says the attribute taxonomy is a *faithful enough* model of real operational series that a model trained entirely on the synthetic version transfers. That is a strong result, and it is also exactly the result a skeptic should want to see stress-tested on series *outside* the operational-metric distribution, which the paper does not do.

## 7. Case studies: where this lands in practice

Theory is cheap; here are the concrete situations where a TS-MLLM like ChatTS changes how the work gets done, and the specific places it would bite. These are framed from the operational world ChatTS was built for — AIOps, observability, and the monitoring of high-cardinality production metrics.

### 7.1 The 3 AM anomaly explanation

An on-call SRE gets paged: `request.latency_p99` on the checkout service crossed threshold. Today, the alert is a single number and a dashboard the human has to read. With a TS-MLLM in the loop, the alert can carry a *generated explanation*: "p99 latency held flat at ~120ms until 03:14, then stepped up by ~210ms and stayed there — a level shift, not a transient spike, consistent with a config change or a downstream dependency degrading rather than a load burst." That is the exact distinction ChatTS's local-fluctuation accuracy (0.895/0.805) is good at. The value is not that the model is smarter than the SRE; it is that it produces the first-pass triage description in natural language, at the moment of the page, for every metric, without a human pre-writing a runbook per metric. The failure mode to watch: the model is only as calibrated as its synthetic level-shift distribution, so a level shift far outside the trained magnitude range may be mis-described.

### 7.2 Multivariate correlation triage

A capacity-planning engineer is staring at thirty metrics from one host and wants to know which ones moved *together*. The interleaving design is built for exactly this: paste all thirty as `<ts>` markers, ask "which of these metrics are correlated, and over what window?" Because each series is bound positionally to its name in the prompt, the model can answer "metrics 4, 9, and 22 (`gc.pause`, `heap.used`, `request.latency`) all spiked in the same 5-minute window around 14:30, suggesting a GC-pressure cascade." A serialized-text approach would drown in digits; a single plot would overlay thirty lines into spaghetti. This is the multivariate-native case the paper repeatedly emphasizes, and it is the one classic tooling handles worst.

### 7.3 Replacing a brittle rule-based anomaly stack

Many production monitoring systems are a pile of hand-tuned rules: "alert if value > 3σ for 5 consecutive points," "alert if week-over-week change > 20%." Each rule is a per-metric labor cost and a source of false positives. A TS-MLLM offers a different shape: describe what you care about in natural language ("alert me on permanent level shifts but not transient spikes") and let the model classify the fluctuation type. ChatTS's 19-way local-fluctuation taxonomy maps almost directly onto the distinctions operators draw by hand. The honest caveat: this does not replace the *fast path* (you still want a cheap statistical detector firing in milliseconds); it replaces the *explanation and classification* path that currently eats human time.

### 7.4 The agent baseline and why native beats orchestration

ChatTS compares against an **agent baseline**: a ReAct loop that calls classic TS tools — STL decomposition for trend/seasonality, ADTK for anomaly detection, Rocket for classification — and reasons over their outputs. This is the "obvious" engineering answer: don't make the LLM read numbers, give it tools that read numbers. And it is *worse* than ChatTS (~0.498 alignment vs 0.889). The reason is instructive: the tools are each good at one attribute, but stitching their outputs back into a coherent multivariate story is itself hard, and every tool boundary is a place to lose context. A native model that perceives all attributes jointly does not have the stitching problem. If you are building a monitoring agent today, this is the result to internalize: tool-calling is not a free substitute for a model that can read the modality. (For the broader argument about when to orchestrate tools versus build a native capability, the [vector-database deep dive](/blog/machine-learning/ai-agent/vector-database) makes the analogous point for retrieval.)

### 7.5 The "trained on synthetic, deployed on real" leap of faith

The single most load-bearing empirical claim in ChatTS is that a model trained *only* on synthetic series generalizes to *real* production series (Dataset A is real-world). This works because the attribute taxonomy is a reasonable basis for the space of real series — most production metrics genuinely *are* a trend plus seasonality plus noise plus occasional local events. Where it would break is a metric whose structure is not in the taxonomy: regime-switching series, series with multiplicative (not additive) seasonality, heavy-tailed bursty traffic that does not look like any sampled "spike." The case study here is a cautionary one: before deploying ChatTS on a new metric class, sanity-check that the metric's real structure is something the attribute generator can produce, because the model has never seen anything it cannot.

### 7.6 Cost-bound batch analysis

A platform team wants to auto-summarize *every* metric in a fleet nightly — tens of thousands of series. At GPT-4o-text prices ($3.25 per ~comparable batch) this is a five-figure monthly bill and a data-egress compliance headache. At ChatTS's $0.02 per batch on self-hosted A800s, it is a background job. This is the unglamorous case study that actually drives adoption inside a company like ByteDance: the accuracy win gets it in the door, but the 160× cost win and the self-hosting are what make "summarize everything, every night" a decision someone will actually approve.

### 7.7 The Chinese-language and reasoning-tuned variants

The release evolved past the paper: there is a `ChatTS-14B-0801` with "enhanced reasoning compatibility and Chinese support," an 8B model for cheaper inference, and GPTQ-Int4 quantized variants of both. The case study here is about productionization realism — the research artifact (14B, English, full precision) is not the same as the deployed artifact (8B-Int4, bilingual, vLLM-served with `limit_mm_per_prompt={"timeseries": 50}`). When you read the paper's numbers, remember they are the 14B full-precision ceiling; the thing you would actually serve trades some of that for throughput, and the quantization interacts with the value-preservation scheme in ways the paper does not measure.

### 7.8 Running it: the inference surface

Concretely, here is what calling ChatTS looks like, which makes the whole interface tangible. Time series go in as a flat list of NumPy arrays; the `<ts><ts/>` placeholder marks where each lands; the processor handles patching and interleaving:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
import numpy as np

MODEL_PATH = "bytedance-research/ChatTS-14B"
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, trust_remote_code=True, device_map=0, torch_dtype="float16")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
processor = AutoProcessor.from_pretrained(
    MODEL_PATH, trust_remote_code=True, tokenizer=tokenizer)

cpu = np.random.normal(size=(100,))   # arbitrary count and length is fine
mem = np.random.normal(size=(150,))

prompt = ("I have CPU usage: <ts><ts/> and memory usage: <ts><ts/>. "
          "Do both stay steady, and is either anomalous?")
chat = (f"<|im_start|>system\nYou are helpful.<|im_end|>\n"
        f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n")

inputs = processor(text=[chat], timeseries=[cpu, mem],
                   padding=True, return_tensors="pt")
inputs = {k: v.to(0) for k, v in inputs.items()}
out = model.generate(**inputs, max_new_tokens=300)
print(tokenizer.decode(out[0], skip_special_tokens=True))
```

The only thing that looks unusual is `timeseries=[cpu, mem]` riding alongside `text=` — that is the whole multimodal interface. Everything else is a standard `transformers` generate call, which is the point: ChatTS is a normal LLM that happens to also accept arrays.

## 8. Critique: where I am skeptical, and what would change my mind

I think ChatTS is the correct framing of the problem and a genuinely strong result. I also think there are four places where the design is a workaround dressed as a principle, and an honest reader should hold these in mind.

**The value-preservation is a hybrid, not a property of the encoder.** Magnitude is preserved by writing the min-max scale and offset into the *prompt text* and trusting the LLM's arithmetic to recombine them. That works at the precision the benchmarks demand, but it means absolute-value accuracy is bounded by the LLM's small-number arithmetic, not by the encoder. A series with a huge dynamic range (a spike 1000× the baseline) compresses the baseline into a tiny slice of `[0,1]` after min-max, and the encoder's resolution there is whatever the MLP learned — the fine structure near the floor can be lost even though the headline magnitude is "preserved." A truly value-native encoder would carry calibrated magnitude in the embedding itself. ChatTS does not; it offloads the hard part to the text channel.

**The whole thing is downstream of one taxonomy.** Trend + seasonality + noise + 19 local fluctuations is a *model* of what time series are. It is a good model for the AIOps metrics ChatTS targets — most server metrics really are that. But it is not a universal basis. Multiplicative seasonality, regime switches, fractal/self-similar traffic, series whose "anomaly" is a *change in the noise process* rather than a localized event — none of these are obviously in the 33-type set, and the model has provably never seen anything outside it. The paper's "trained on synthetic, works on real" result is real, but it is evidence that *this taxonomy covers these metrics*, not that it covers time series in general.

**"First TS-MLLM" is a claim about a moving frontier.** The paper positions ChatTS as the first multimodal LLM to take multivariate time series as native input for understanding *and* reasoning. That framing is fair as of its publication, but the field is moving fast and the contribution is better stated as "the first to make the modality + synthetic-attribute-data combination work at this quality," because the individual ideas (patch encoders, Evol-Instruct, multimodal projectors) are borrowed. The novelty is the *integration and the data pipeline*, which is genuine but less headline-grabbing than "first."

**The baselines are strong but not exhaustive.** GPT-4o (text and vision), a same-size and a larger Qwen, and a ReAct agent are good baselines. What is missing is a comparison against a *purpose-built supervised model per task* — a dedicated anomaly detector trained on the same synthetic anomalies, a dedicated seasonality estimator. ChatTS would likely lose to a specialist on its own narrow task (a tuned spike detector will beat a generalist at spike detection), and the paper's value proposition is "one model, many tasks, natural-language interface," not "best at any single task." That is a fine value proposition, but the tables compare against generalists, which flatters the generalist.

**What would change my mind, concretely:** (1) An evaluation on a metric class *outside* the synthetic taxonomy — say, regime-switching financial series or multiplicative-seasonality retail demand — showing ChatTS degrades gracefully rather than confabulating. If it holds up there, the taxonomy is more universal than I credit. (2) A stress test of the value-preservation under extreme dynamic range (spikes 100–1000× baseline) measuring numerical accuracy near the floor; if it stays high, the hybrid normalization is more robust than I fear. (3) An ablation that replaces the text-channel scale/offset with a magnitude-carrying embedding and shows it does *not* help — that would prove the hybrid is not just a shortcut but the right design. Absent those, I read ChatTS as a strong, correctly-framed, AIOps-calibrated result whose generalization claims should be scoped to "the kinds of series the generator can produce."

## 9. When to reach for a TS-MLLM, and when not to

**Reach for ChatTS (or a TS-MLLM) when:**

- You need a *natural-language interface* over time series — explanations, triage descriptions, ad-hoc questions — rather than a single scalar prediction. This is the sweet spot: the LLM's job is to *talk about* the series.
- Your data is *multivariate and contextual* — many metrics per host, where the relationship between series and the meaning of each (its name, its unit) matters. The interleaving design is built for this and classic tooling handles it worst.
- You are *cost- and privacy-bound* — you want to analyze huge volumes of internal telemetry without shipping it to an API. A self-hosted 14B (or 8B-Int4) at $0.02/batch is the enabling fact.
- Your series genuinely *look like* trend + seasonality + noise + local events — i.e., standard operational metrics. This is where the synthetic taxonomy is a faithful model and the perception accuracy (0.889 alignment) actually transfers.

**Do not reach for it when:**

- You need a *point forecast* with calibrated uncertainty. That is a forecasting problem, and a forecasting foundation model like [Timer-S1](/blog/machine-learning/signal-processing/timer-s1-time-series-foundation-model) (state-of-the-art zero-shot MASE/CRPS on GIFT-Eval) is the right tool. ChatTS understands and reasons; it does not forecast.
- You need a *millisecond-latency detector* in the hot path. A 14B LLM forward pass is not your fast anomaly trigger; a cheap statistical detector is. Use the TS-MLLM for the explanation/classification *after* the fast path fires.
- You need *best-in-class accuracy on one narrow task*. A specialist supervised model trained on that exact task will beat a generalist; reach for the TS-MLLM when you value breadth and the language interface over peak single-task accuracy.
- Your series structure is *outside the synthetic taxonomy* — regime-switching, multiplicative seasonality, exotic noise processes. Until someone shows graceful degradation there, treat ChatTS as scoped to the operational-metric world it was built and validated for.

The deeper takeaway transcends time series. ChatTS is the latest instance of a recipe that keeps working across modalities: do not describe a non-text modality to an LLM in words, and do not render it to an image — build an encoder that lands the modality in token space, preserve the information that matters (here, the numbers), interleave it with text so context is kept, and generate training data where the ground truth is exact by construction. The same shape produced strong [speech models](/blog/machine-learning/signal-processing/orpheus-tts-llm-speech-snac) and [non-verbal TTS](/blog/machine-learning/signal-processing/distilling-fast-tts-non-verbal). Time series was the modality that got skipped, treated as something you serialize or plot rather than something you encode. ChatTS's contribution is to stop skipping it — and the numbers say that once you treat numbers as a first-class modality, an LLM can finally read them.

## References

- **Paper:** Xie et al., "ChatTS: Aligning Time Series with LLMs via Synthetic Data for Enhanced Understanding and Reasoning," VLDB 2025 (ByteDance). arXiv: [2412.03104](https://arxiv.org/abs/2412.03104) · [HTML v3](https://arxiv.org/html/2412.03104v3)
- **Code:** [NetManAIOps/ChatTS](https://github.com/NetManAIOps/ChatTS) — generators (`generate_align_datasets.sh`, `generate_tsevol_dataset`), Transformers + vLLM inference, the attribute-set config.
- **Models:** [bytedance-research/ChatTS-14B](https://huggingface.co/bytedance-research/ChatTS-14B) and the later `ChatTS-8B`, `ChatTS-14B-0801`, and GPTQ-Int4 variants on Hugging Face.
- **Backbone:** [Qwen2.5-14B-Instruct](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct).

Cross-links worth following:
- [ByteDance Research model atlas](/blog/machine-learning/bytedance-research-model-atlas) — the hub for this reading series.
- [Timer-S1: time-series foundation model](/blog/machine-learning/signal-processing/timer-s1-time-series-foundation-model) — the forecasting sibling to ChatTS's understanding-and-reasoning focus.
- [Orpheus TTS over SNAC](/blog/machine-learning/signal-processing/orpheus-tts-llm-speech-snac) and [distilling fast non-verbal TTS](/blog/machine-learning/signal-processing/distilling-fast-tts-non-verbal) — the same encoder-into-token-space recipe applied to speech.
- [Vector databases](/blog/machine-learning/ai-agent/vector-database) — for the broader native-capability-vs-tool-orchestration argument.
