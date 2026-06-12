---
title: "Canary and Parakeet: How NVIDIA's FastConformer Makes Speech Recognition Fast"
date: "2026-06-11"
publishDate: "2026-06-11"
description: "A deep dive into NVIDIA's FastConformer ASR encoder and the Canary and Parakeet model families — 8x downsampling, depthwise-separable convolutions, the CTC/RNN-T/TDT/attention decoder zoo, cache-aware streaming, and 11-hour long-audio transcription."
tags: ["asr", "speech-recognition", "fastconformer", "conformer", "nvidia", "canary", "parakeet", "rnn-t", "ctc", "tdt", "streaming-asr", "deep-learning"]
category: "machine-learning"
subcategory: "Deep Learning"
author: "Hiep Tran"
featured: true
readTime: 49
---

## The frame-rate problem nobody talks about

Speech recognition has a quiet scaling problem, and it is not the one people expect. Everyone worries about model size and dataset size; the real bottleneck for an ASR encoder is the **frame rate** — how many feature frames per second the encoder has to chew through. Audio is sampled at 16,000 samples per second, turned into a spectrogram at, say, 100 frames per second, and every one of those frames flows through every layer of the encoder. Self-attention over a long audio clip is quadratic in the number of frames, so a ten-minute clip at 100 fps is 60,000 frames, and the attention matrix is 60,000 × 60,000 — billions of entries, per layer, per head. The frame rate, not the parameter count, is what makes long-audio ASR expensive.

NVIDIA's **FastConformer** is a deceptively simple answer: cut the frame rate. Where the standard Conformer encoder downsamples the audio 4× before the transformer layers, FastConformer downsamples **8×**, using cheap depthwise-separable convolutions, so the encoder sees a *quarter* as many frames. That single change makes the encoder about **2.4× faster** with no meaningful loss in accuracy, and it unlocks transcribing **up to 11 hours of audio in a single pass** on one A100. On top of that one encoder, NVIDIA built two model families: **Parakeet** (FastConformer + a transducer or CTC decoder, tuned for fast, accurate, long-audio transcription) and **Canary** (FastConformer + an attention decoder, tuned for multilingual transcription *and* translation across 25 languages).

This is the fifth post in a series reading NVIDIA's model reports for their reusable techniques — the first four covered language models ([Minitron](/blog/machine-learning/large-language-model/nvidia-minitron-pruning-distillation), [Nemotron-4 340B](/blog/machine-learning/large-language-model/nemotron-4-340b-synthetic-data-alignment), [Llama-Nemotron](/blog/machine-learning/large-language-model/llama-nemotron-efficient-reasoning-models), [Nemotron-H](/blog/machine-learning/large-language-model/nemotron-h-hybrid-mamba-transformer)) — and this one turns to speech. The reusable techniques are the **8× downsampling trick**, the **depthwise-separable convolutions** that make it cheap, the **decoder zoo** (CTC, RNN-T, TDT, attention) and how to choose among them, and **cache-aware streaming**. If you have read the [training ASR models guide](/blog/machine-learning/deep-learning/training-asr-models), this is the architecture deep dive that complements it.

The mismatch the whole design resolves:

| Question | The naive assumption | What FastConformer shows |
|---|---|---|
| What bounds ASR encoder cost? | Model size | The frame rate — frames per second through the encoder |
| Is 4× downsampling already aggressive? | Yes, don't lose resolution | No — 8× works with no accuracy loss |
| Are convolutions expensive? | Standard convs, yes | Depthwise-separable convs are cheap |
| One decoder for everything? | Pick CTC or attention | A zoo — CTC, RNN-T, TDT, attention — matched to the task |
| Must transducers waste steps on blanks? | Yes, one step per frame | No — TDT predicts a duration and skips them |
| Can one model do offline and streaming? | Train two models | No — cache-aware streaming does both |

![Pipeline diagram of the FastConformer ASR stack: an audio waveform at 16 kHz becomes a log-mel spectrogram, is downsampled 8x by depthwise convolution, encoded by the FastConformer encoder, decoded by one of CTC, RNN-T, TDT, or an attention decoder, and emitted as a transcript with optional translation](/imgs/blogs/canary-parakeet-fastconformer-asr-1.webp)

Before the details, it is worth situating this in the broader arc of speech recognition. For most of the deep-learning era, ASR progress was measured almost entirely in word error rate — the field competed to be *more accurate*, and the models got bigger and slower in pursuit of it. FastConformer represents a maturation: accuracy on common benchmarks has saturated to the point where the differences between top models are small, and the competition has shifted to *efficiency* — the same model quality, but faster, cheaper, longer-context, streaming, multilingual. This mirrors what happened in language models, where the frontier moved from "can it do the task" to "can it do the task affordably at scale." FastConformer is speech recognition's efficiency turn, and the techniques it uses — aggressive downsampling, cheap convolutions, decoder specialization, cache-aware streaming — are the speech analogues of the quantization, pruning, and architecture-search moves that the rest of this NVIDIA series applies to language. Reading it alongside the LLM posts, you see one engineering culture applying the same efficiency playbook across modalities.

The diagram above is the mental model: audio becomes features, the features are **aggressively downsampled**, the FastConformer encoder turns them into a rich representation, and a **decoder head** converts that representation into text. The whole article is a tour of that pipeline — what the encoder block does, why 8× downsampling is the key trick, how depthwise-separable convolutions make it cheap, which decoder to pick for which job, and how the same model streams. The organizing idea:

> The expensive part of an ASR encoder is the number of frames it processes. FastConformer's entire philosophy is to spend a little cheap convolution up front to drastically cut the frame rate, so every expensive attention layer downstream has far less to do.

## 1. The Conformer block: global meaning and local sound

Before FastConformer, there was the **Conformer**, and understanding its block is the foundation. The insight behind the Conformer is that speech needs *two kinds* of context, and a single mechanism is bad at both.

![Hand-authored diagram of a Conformer block as a vertical stack: a half-weighted feed-forward network, then multi-head self-attention with relative positional encoding providing global context, then a depthwise convolution module with kernel size nine providing local context, then a second half-weighted feed-forward network, then layer normalization, with a side note explaining why speech needs both global and local context](/imgs/blogs/canary-parakeet-fastconformer-asr-2.webp)

A Conformer block, shown above, is a **macaron**: two half-weighted feed-forward networks (each contributing half its output to the residual, hence the ×0.5) sandwich two sequence-mixing modules — multi-head self-attention and a convolution module. The reason for both mixers is that speech has structure at two scales:

- **Global context (attention).** What a word means depends on the whole utterance — the topic, the speaker's earlier words, the grammatical structure spanning seconds. Self-attention, with relative positional encoding, lets any frame attend to any other frame, capturing these long-range dependencies. This is the same all-pairs mixing that powers language models, applied to audio frames.
- **Local context (convolution).** What a *phoneme* is depends on its immediate acoustic neighborhood — the formant transitions, the onset and offset, the fine-grained spectral shape over a few tens of milliseconds. A convolution with a small kernel is exactly the right tool for this local pattern-matching, the same reason [convolutions work](/blog/machine-learning/computer-vision/why-convolution-works) for local structure in images.

Attention alone is bad at local detail (it has no inductive bias for locality — it must learn it), and convolution alone is bad at global context (its receptive field is limited by kernel size and depth). The Conformer's bet, validated across years of ASR results, is that you want *both in every block*: the convolution handles the fine phonetic structure, the attention handles the long-range meaning, and the FFNs do the per-frame processing. FastConformer keeps this block structure intact — it does not change *what* the block computes, only how many frames the block sees and how cheap its convolutions are.

### A short history of how we got here

It helps to see the Conformer in context, because its design was hard-won. The first neural ASR encoders were recurrent (LSTMs), processing audio frame by frame — accurate but slow and hard to parallelize. Then came pure-attention (Transformer) encoders, which parallelized well and captured global context but struggled with the fine local acoustic structure that convolutions handle naturally, and they were data-hungry. Convolutional encoders (like the early QuartzNet and Citrinet lines) were fast and good at local structure but limited in global context. The **Conformer** (2020) was the synthesis: take the Transformer's attention for global context, add a convolution module for local structure, wrap them in the macaron FFNs, and you get a block that beats both pure-attention and pure-convolution encoders. It became the dominant ASR encoder almost immediately, and FastConformer is its efficiency-focused evolution. The lineage matters because it shows the Conformer's two-mixer design was not arbitrary — it was the resolution of a genuine tension (global vs local, the strengths of attention vs convolution) that the field had been circling for years. FastConformer inherits that resolution and makes it cheaper.

### Why relative positional encoding

A specific design choice in the Conformer's attention deserves emphasis because it is load-bearing for FastConformer's long-audio capability: the attention uses **relative** positional encoding rather than absolute. Absolute positional encoding assigns each frame a position index (frame 0, frame 1, ...) and embeds it; relative encoding instead encodes the *distance* between a query frame and a key frame. For audio, where clips range from a one-second command to an eleven-hour recording, absolute positions are a disaster — the model trained on short clips never sees position 100,000 and generalizes poorly to it. Relative positions, by contrast, are bounded by the attention window and identical regardless of where in the clip you are, so a model trained on short clips transfers cleanly to long ones. This is one of the quiet reasons the architecture scales to 11-hour audio: the positional scheme does not break at lengths it never trained on. The general lesson is that for inputs of wildly varying length, encode *relationships* (distances) not *absolute coordinates*, because relationships generalize across scales that absolute coordinates cannot.

### Second-order optimization: match the mixer to the structure scale

The reusable principle is that **different scales of structure want different mixing primitives**. Local structure (phonemes, edges, n-grams) is best captured by convolution, whose locality bias is a feature; global structure (meaning, long-range dependency) is best captured by attention, whose all-pairs reach is a feature. Forcing one primitive to do both jobs wastes capacity — attention has to learn locality it could have gotten for free, convolution has to stack deep to reach context it could have gotten directly. The Conformer's lasting lesson, which echoes the [hybrid-architecture thinking in Nemotron-H](/blog/machine-learning/large-language-model/nemotron-h-hybrid-mamba-transformer), is to use multiple primitives matched to the scales your data actually has, rather than one primitive everywhere out of habit.

## 2. The FastConformer trick: downsample 8×, not 4×

Here is the single most important idea in the whole architecture, and it is almost embarrassingly simple: **process fewer frames**.

![Before-and-after comparison: on the left, the Conformer uses 4x convolutional subsampling, so more frames reach the encoder, making it slower and more attention-memory-hungry; on the right, FastConformer uses 8x depthwise-separable subsampling with 256 channels, sending a quarter as many frames to the encoder, running 2.4x faster at the same accuracy](/imgs/blogs/canary-parakeet-fastconformer-asr-3.webp)

A spectrogram at 100 frames per second is *more temporal resolution than the encoder needs*. Phonemes last on the order of 50–150 milliseconds; you do not need a frame every 10 milliseconds to recognize them. The original Conformer already exploited this with **4× subsampling** — three convolutional layers that each halve... no, that reduce the frame rate by 4× total, bringing 100 fps down to 25 fps before the transformer layers. FastConformer doubles down: **8× subsampling**, bringing 100 fps down to **12.5 fps**, so the encoder processes *half* as many frames as the 4× Conformer and a *quarter* as many as the raw spectrogram.

The consequences cascade through the whole encoder:

- **Attention is quadratic in frame count.** Halving the frames quarters the attention compute and the attention memory. This is where most of the 2.4× speedup comes from.
- **Every layer does less.** Fewer frames means every FFN, every convolution, every layer norm processes fewer tokens.
- **Longer audio fits.** Because the frame count is halved, you can process twice the audio duration in the same memory — a key enabler of the 11-hour long-audio capability.

The design specifics that make 8× work without losing accuracy:

- **Depthwise-separable convolutions** for the subsampling (§3), so the extra downsampling layer is cheap.
- **A reduced channel count (256)** in the downsampling module, keeping the parameter footprint small.
- **A smaller convolution kernel (9)** in the Conformer blocks, since after 8× downsampling each frame already covers more time, so a smaller kernel reaches the same effective temporal context.

The headline result: FastConformer is **~2.4× faster than Conformer** with no significant quality degradation. You lose almost nothing in word error rate and gain a large speedup, which is the definition of a free lunch in architecture design.

```python
import torch.nn as nn

class FastConformerSubsampling(nn.Module):
    """8x time downsampling via depthwise-separable convs, 256 channels.
    Brings 100 fps spectrogram frames down to ~12.5 fps for the encoder."""
    def __init__(self, feat_in, d_model, channels=256):
        super().__init__()
        # three stride-2 stages -> 2*2*2 = 8x downsampling
        self.stages = nn.ModuleList([
            DepthwiseSeparableConv2d(in_ch, channels, kernel=3, stride=2)
            for in_ch in (1, channels, channels)
        ])
        self.out = nn.Linear(channels * (feat_in // 8), d_model)

    def forward(self, spec):                 # spec: [B, 1, T, F] at ~100 fps
        x = spec
        for stage in self.stages:
            x = stage(x).relu()              # each stage halves T (and F)
        b, c, t, f = x.shape                 # T is now ~12.5 fps (8x fewer)
        return self.out(x.transpose(1, 2).reshape(b, t, c * f))
```

### Why 8× does not hurt accuracy

The natural worry about downsampling 8× is that you are throwing away temporal resolution and must be losing information. Why does word error rate barely move? Two reasons. First, the **information is still there** — downsampling does not delete the audio, it pools it. The depthwise-separable convolutions in the subsampling module *learn* how to summarize each group of frames into one, preserving the discriminative information (the formant transitions, the spectral cues) while discarding redundancy. A 100 fps spectrogram is highly redundant — adjacent frames are nearly identical — so pooling them 8× loses little that the model needed. Second, the **downstream layers can recover detail** — the Conformer's convolution module, operating on the downsampled frames, still captures fine local structure, just at a coarser frame grid that, after 8×, still has enough resolution for phoneme-scale patterns (phonemes span many 12.5 fps frames). The empirical result — no significant WER degradation at 8× — confirms that 100 fps was over-sampled for the task. The lesson is that "throwing away resolution" and "throwing away information" are not the same: redundant resolution can be pooled away losslessly-enough, and the only way to know how much is redundant is to try cutting it and measure. FastConformer measured, and the answer was "a lot."

### Word error rate, the metric that governs everything

A note on the metric, since it governs the trade-offs. ASR quality is measured by **word error rate (WER)** — the edit distance between the predicted transcript and the reference, normalized by reference length: $\text{WER} = (S + D + I) / N$, where $S$, $D$, $I$ are substitutions, deletions, and insertions and $N$ is the number of reference words. Every architecture decision in FastConformer is justified by "WER does not degrade significantly" — the 8× downsampling, the smaller kernel, the depthwise-separable convs all hold WER roughly fixed while cutting cost. This is the right framing for an efficiency paper: hold the quality metric fixed and show the speedup, rather than trading quality for speed. When you read that FastConformer is "2.4× faster with no significant quality degradation," the "no significant degradation" is a WER claim, and it is the part that makes the speedup matter — a speedup that cost three WER points would be a different and far less interesting result. The discipline of holding WER fixed is what separates a genuine efficiency win from merely picking a worse point on the quality curve.

### Second-order optimization: resolution is a knob, and the default is too high

The deep lesson is that **input resolution is a tunable hyperparameter, and the conventional default is often higher than necessary**. ASR inherited 100 fps spectrograms from an era before the encoder cost mattered; FastConformer simply asks "how few frames can we get away with?" and finds the answer is far fewer than tradition assumed. This generalizes everywhere: image models process more pixels than they need, video models more frames, point-cloud models more points. Before optimizing the model, ask whether you can *reduce the input resolution* — it is often the highest-leverage, lowest-effort speedup available, because cost scales with input size and the model rarely needs the full resolution it is fed.

## 3. Depthwise-separable convolutions: the cheap way to downsample

The 8× downsampling would not be free if it used standard convolutions — it is free because it uses **depthwise-separable** convolutions, which factor a convolution into two much cheaper steps.

![Hand-authored diagram contrasting a standard convolution, which mixes all input channels across kernel positions in one expensive operation, with a depthwise-separable convolution that splits into a depthwise step (a per-channel spatial filter) followed by a pointwise 1x1 step (channel mixing), reducing the cost from C-in times C-out times K to C-in times the sum of K and C-out](/imgs/blogs/canary-parakeet-fastconformer-asr-4.webp)

A standard 2D convolution does everything at once: for each output channel, it sums over all input channels and all kernel positions. Its cost (in multiply-accumulates per output position) is

$$\text{cost}_{\text{standard}} = C_{\text{in}} \times C_{\text{out}} \times K$$

where $K$ is the kernel size (positions). The depthwise-separable convolution **factors this into two steps**:

1. **Depthwise convolution** — apply one $K$-sized kernel *per input channel*, doing spatial filtering but *no channel mixing*. Cost: $C_{\text{in}} \times K$.
2. **Pointwise convolution** — a $1 \times 1$ convolution that mixes channels (maps $C_{\text{in}}$ to $C_{\text{out}}$) but does *no spatial filtering*. Cost: $C_{\text{in}} \times C_{\text{out}}$.

Total cost: $C_{\text{in}} \times (K + C_{\text{out}})$ instead of $C_{\text{in}} \times C_{\text{out}} \times K$. For typical values ($C_{\text{out}} = 256$, $K = 9$), that is a reduction from $\sim 2300 \times C_{\text{in}}$ to $\sim 265 \times C_{\text{in}}$ — roughly **9× cheaper**. The insight, the same one that powered MobileNets in computer vision, is that the standard convolution *couples* two independent operations (spatial filtering and channel mixing), and decoupling them removes a huge amount of redundant computation with little loss of expressivity, because most useful convolutions do not actually need every input channel to interact with every kernel position.

```python
import torch.nn as nn

class DepthwiseSeparableConv2d(nn.Module):
    """Factor a conv into depthwise (spatial, per-channel) + pointwise (1x1, mix)."""
    def __init__(self, in_ch, out_ch, kernel, stride):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel, stride=stride,
                                   padding=kernel // 2, groups=in_ch)   # per-channel
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1)        # mix channels

    def forward(self, x):
        return self.pointwise(self.depthwise(x))   # cost ~ C_in*(K + C_out)
```

FastConformer uses depthwise-separable convolutions in two places: the **subsampling module** (where they make 8× downsampling cheap) and the **convolution module inside each Conformer block** (where they make the local mixing cheap). Both are the same trick — factor the convolution, pay far less.

### Where depthwise-separable convolutions came from

The technique is borrowed, and the borrowing is itself instructive. Depthwise-separable convolutions were popularized by the **MobileNet** family in computer vision, where the goal was to run convolutional networks on phones — devices with tight compute and memory budgets. The vision community discovered that the standard convolution's coupling of spatial and channel mixing was the main cost, and that factoring it gave a roughly 8-9× compute reduction with minimal accuracy loss, which is exactly what made on-device vision feasible. FastConformer transplants this vision technique into speech, where it does the same job: makes the convolutions cheap enough that you can afford the extra 8× downsampling and the in-block convolution module. The cross-pollination is the lesson — a technique invented to fit vision models on phones turns out to be exactly what speech models need to cut their frame-rate cost. Staying fluent across subfields lets you spot these transplants; the engineer who knew MobileNets saw immediately that depthwise-separable convs would help an ASR encoder, because the cost structure (expensive convolutions dominating) was the same even though the domain was different.

### The parameter and compute math, worked out

To make the savings concrete, take the in-block convolution module: input and output channels both equal the model dimension $d$ (say 512), kernel size $K = 9$. A standard convolution costs $d \times d \times K = 512 \times 512 \times 9 \approx 2.36$ million multiply-accumulates per output position. The depthwise-separable version costs $d \times K$ (depthwise) plus $d \times d$ (pointwise) $= 512 \times 9 + 512 \times 512 \approx 4{,}600 + 262{,}000 \approx 267{,}000$ — about **9× cheaper**. Notice where the cost now lives: the pointwise $1\times1$ (channel mixing) dominates, and the depthwise (spatial) is nearly free. This tells you something useful — after factoring, the channel mixing is the remaining cost, which is why grouped pointwise convolutions and channel-reduction tricks (like FastConformer's 256-channel downsampling module) target exactly that term. The general lesson is that factoring an operation does not just make it cheaper, it *reveals which sub-operation now dominates*, pointing you at the next optimization. After depthwise-separable factoring, the answer is "channel mixing," and that is where the next round of efficiency work goes.

### Second-order optimization: factor coupled operations into independent ones

The general principle, which recurs across efficient deep learning, is that **a single operation that couples two independent transformations is usually wasteful, and factoring it into the two transformations is cheaper with little loss**. Standard convolution couples spatial and channel mixing; depthwise-separable splits them. (The same pattern appears in low-rank factorizations of weight matrices, in grouped convolutions, and in the [FFN Fusion / attention separation of Llama-Nemotron](/blog/machine-learning/large-language-model/llama-nemotron-efficient-reasoning-models).) When you see an expensive operation, ask whether it is secretly doing two independent jobs at once, and whether separating them lets you do each cheaply. It usually does.

## 4. The decoder zoo: one encoder, four heads

The FastConformer encoder produces a sequence of rich frame representations; turning those into text is the decoder's job, and there is no single best decoder — there is a **zoo**, each member trading off speed, accuracy, streaming ability, and capability differently.

![Tree diagram showing a shared FastConformer encoder branching to four decoder heads: CTC (non-autoregressive, fastest, no language model), RNN-T (streaming, accurate, frame-by-frame), TDT (RNN-T plus duration, skips blanks), and an attention encoder-decoder Transformer used by Canary for ASR plus translation](/imgs/blogs/canary-parakeet-fastconformer-asr-5.webp)

The four decoder heads, and what each is for:

- **CTC (Connectionist Temporal Classification).** The simplest and fastest. CTC is **non-autoregressive** — it predicts a label (or a blank) for every frame independently, then collapses repeats and blanks to get the transcript. No recurrence, no per-token dependency, so it decodes in a single forward pass — extremely fast. The cost is accuracy: because frames are predicted independently, CTC has no internal language model, so it can produce locally-plausible but globally-inconsistent transcripts. Great when speed dominates and you can add an external language model.
- **RNN-T (RNN Transducer).** The accuracy workhorse and the default for streaming. RNN-T has a **prediction network** (an internal language model over the emitted tokens) and a **joint network** that combines encoder frames with the prediction state, emitting a token or a blank at each step. It is autoregressive (each token depends on the previous), so it models the output distribution properly and is more accurate than CTC, and crucially it is **naturally streaming** — it consumes frames left to right and emits as it goes. The cost is decoding speed: it steps frame by frame, emitting many blanks.
- **TDT (Token-and-Duration Transducer).** RNN-T's faster successor (§5) — same accuracy benefits, but it predicts a *duration* alongside each token so it can skip the blank frames RNN-T would decode one by one.
- **AED (Attention Encoder-Decoder).** A full Transformer decoder that cross-attends to the encoder output and generates the transcript autoregressively, like a translation model. This is what **Canary** uses, because it is the most flexible — it can do **multilingual ASR and translation** (the decoder generates text in any target language conditioned on a task token), at the cost of being non-streaming and slower.

The genius of the design is that the **encoder is shared**. You train one strong FastConformer encoder and pair it with whichever decoder your task needs: CTC for raw speed, RNN-T/TDT for accurate streaming, AED for multilingual translation. Parakeet ships the transducer/CTC variants; Canary ships the AED variant; the encoder underneath is the same family.

```python
enc_out = fastconformer_encoder(features)        # shared rep: [B, T_sub, d_model]

ctc_logits = ctc_head(enc_out)                   # non-autoregressive, fastest
rnnt_tokens = rnnt_decode(enc_out, pred_net)     # streaming, accurate
tdt_tokens = tdt_decode(enc_out, pred_net)       # RNN-T + duration, skips blanks
canary_text = attn_decoder(enc_out, task="translate_to_de")  # multilingual AED
```

### CTC and RNN-T, mechanically

It is worth a closer look at *why* CTC and RNN-T differ in accuracy, because it explains the whole decoder zoo. The core problem in ASR is **alignment**: you have $T$ audio frames and $U$ output tokens with $U \ll T$, and you do not know which frames correspond to which tokens. Both CTC and RNN-T solve this with a **blank** symbol and a sum over all valid alignments, but they differ in one crucial way. CTC assumes the output tokens are **conditionally independent given the audio** — it predicts each frame's label without conditioning on the previously emitted tokens. This is what makes it fast (every frame can be predicted in parallel) and what makes it less accurate (it has no model of which token sequences are plausible, so it can emit "their" where "there" was meant, because nothing tells it the surrounding words). RNN-T removes the independence assumption: its **prediction network** conditions each output on the previously emitted tokens, acting as an internal language model, so it knows "there" is more likely than "their" in context. That conditioning is the accuracy gain, and the sequential dependency it introduces is the speed cost. The whole decoder zoo is variations on this theme — how much output-side conditioning to keep (accuracy) versus how much parallelism to allow (speed).

$$\mathcal{L}_{\text{CTC}} = -\log \sum_{\pi \in \mathcal{B}^{-1}(y)} \prod_t P(\pi_t \mid x) \qquad \text{(tokens independent given } x\text{)}$$

The CTC loss above sums over all frame-label alignments $\pi$ that collapse (via $\mathcal{B}$, which removes blanks and repeats) to the target $y$ — and the product over $t$ with each term depending only on $x$ is exactly the conditional-independence assumption that RNN-T's prediction network removes.

### Second-order optimization: share the expensive part, vary the cheap part

The architectural lesson is to **share the expensive, general-purpose component and vary the cheap, task-specific one**. The encoder is the expensive part — it learns the acoustic representation, takes the most compute, and is the hardest to train. The decoder is comparatively cheap and is where the task-specific behavior lives (speed vs accuracy vs multilingual). By sharing one encoder across decoder heads, NVIDIA amortizes the expensive training across the whole product line and lets each deployment pick its speed/capability trade-off without retraining the backbone. This is the same "train once, derive many" philosophy that runs through the whole NVIDIA series — here applied to the encoder/decoder split.

## 5. TDT: stop wasting steps on blanks

The Token-and-Duration Transducer deserves its own section because it is a clean, reusable idea that makes transducer decoding much faster.

![Before-and-after comparison: on the left, a standard RNN-T emits a token or a blank at every frame, most frames emit blank, and many decoding steps are wasted; on the right, TDT predicts a token and its duration, jumps ahead past the blank frames, and uses far fewer steps for faster inference](/imgs/blogs/canary-parakeet-fastconformer-asr-6.webp)

The inefficiency TDT fixes is specific to transducers. An RNN-T decodes by stepping through encoder frames and, at each frame, emitting either a token or a **blank** (meaning "no output here, advance to the next frame"). In practice, **most frames emit blank** — speech is slower than the frame rate, so a single phoneme spans many frames, and the transducer spends most of its decoding steps emitting blanks just to advance through them. Each blank is a full forward pass of the joint network that produces no output. It is pure overhead.

**TDT** generalizes the transducer by **decoupling token prediction from duration prediction**. At each step, instead of just emitting a token-or-blank, it emits a **token** *and* a **duration** — how many frames to skip before the next decision. So when the model emits a phoneme that spans, say, 8 frames, it emits the token *and a duration of 8*, and jumps 8 frames forward in one step, rather than emitting the token and then 7 blanks. The result is **far fewer decoding steps** — TDT skips the majority of blank predictions — and therefore significantly faster inference, with no loss in accuracy (the duration prediction is learned jointly and is accurate).

```python
def tdt_decode(enc_out, model):
    """TDT: emit (token, duration) and jump ahead, instead of one step per frame."""
    t, tokens = 0, []
    while t < len(enc_out):
        token, duration = model.step(enc_out[t], tokens)   # predict BOTH
        if token != BLANK:
            tokens.append(token)
        t += max(duration, 1)        # skip `duration` frames in one step
    return tokens                    # far fewer iterations than frame-by-frame RNN-T
```

### How the duration is learned and bounded

A practical detail on TDT: the duration is a *discrete* prediction over a small set of allowed values (say, 0 to 4 frames, or up to some maximum), trained jointly with the token prediction. The model learns, from data, the typical duration distribution — most tokens span a few frames, and the model predicts how many to skip. The maximum allowed duration is a hyperparameter that trades speed for safety: a larger maximum lets the model skip more frames per step (faster) but risks over-skipping past a token boundary (an error); a smaller maximum is safer but slower. In practice a modest maximum captures most of the speedup, because the duration distribution is concentrated — you do not need to skip 50 frames at once, you need to skip the 4-8 blanks between tokens, and even a maximum of 4-8 collapses most of the wasted steps. The lesson is that the "predict the stride" pattern needs a *bounded* stride to stay safe: you let the model jump ahead, but cap how far, so a confident-but-wrong duration prediction cannot skip an entire word. Bounded skipping captures most of the benefit at a fraction of the risk.

### Second-order optimization: predict the skip, do not iterate it

The reusable idea is to **make the model predict how far to advance, rather than advancing one unit at a time and checking**. RNN-T iterates frame by frame and emits blanks to advance; TDT predicts the advance directly. This "predict the stride" pattern shows up across sequence modeling — in non-autoregressive generation, in adaptive computation, in any setting where a model spends steps just moving forward through input it has already understood. If your decoder is emitting a lot of "nothing happened here, continue" signals, that is a sign it should be predicting durations or strides instead, collapsing many cheap-but-not-free steps into one.

## 6. Cache-aware streaming: one model, offline and live

ASR has two deployment modes with opposite requirements: **offline** (transcribe a recording — accuracy matters, latency does not) and **streaming** (transcribe live — latency is everything). The naive approach trains two separate models. FastConformer's **cache-aware streaming** does both with one.

![Pipeline diagram of cache-aware streaming: an audio chunk arrives, the cached left context is prepended, the FastConformer encodes the chunk with limited lookahead, a partial transcript is emitted, and the cache is updated for the next chunk](/imgs/blogs/canary-parakeet-fastconformer-asr-7.webp)

The problem with streaming a Conformer is that both its mixers want context the future has not provided yet. Attention, by default, attends to the *whole* utterance, including frames that have not arrived. The convolution module looks at neighboring frames on both sides, again including future ones. To stream, you must limit how far into the future the model looks (the **lookahead**), because every bit of lookahead is latency — the model has to wait for those future frames before it can emit.

**Cache-aware streaming** handles this by processing audio in **chunks** with a **cache of past context**:

1. An audio **chunk** arrives (e.g., 80 ms of audio).
2. The model **prepends the cached left context** — the encoder states from previous chunks — so it has the full past available without recomputing it.
3. The FastConformer encodes the chunk with **limited lookahead** (a small, bounded right context), so latency is bounded.
4. It **emits a partial transcript** for the chunk.
5. It **updates the cache** with this chunk's states for the next chunk.

The "cache-aware" part is the key efficiency: the left context is *cached*, not recomputed, so each chunk only does new work proportional to the chunk size, not the whole history so far. And critically, the model is **trained with this chunked, limited-lookahead regime**, so it learns to produce accurate transcripts under streaming constraints — the same weights then work offline (full context) or streaming (cached chunks), tuned by the lookahead setting. One model, two modes, with the latency/accuracy trade-off set at inference time by how much lookahead you allow.

```python
def stream(model, audio_chunks, lookahead):
    """Cache-aware streaming: each chunk reuses cached left context, bounded latency."""
    cache = model.init_cache()
    for chunk in audio_chunks:                # arrives live, e.g. every 80 ms
        out, cache = model.encode_chunk(
            chunk, cache=cache, right_context=lookahead)  # bounded future = bounded latency
        yield model.decode(out)               # emit partial transcript now
```

### The latency arithmetic of streaming

Streaming latency is governed by a simple, unforgiving equation worth making explicit. The latency before the model can emit a token is, roughly, the **chunk size plus the lookahead** — the model must wait for the current chunk of audio to arrive, plus however many future frames its limited right-context needs, before it can produce output for the current frame. So if your chunk is 80 ms and your lookahead is 160 ms, your algorithmic latency floor is ~240 ms regardless of how fast the hardware is. This is why lookahead is the central streaming knob: every millisecond of future context the model needs is a millisecond of latency the user feels. The accuracy-latency trade-off is direct — more lookahead means the model sees more future context and transcribes more accurately (it can use upcoming sounds to disambiguate the current one), but it also means more latency. Cache-aware streaming lets you pick a point on this curve at inference time by setting the lookahead, and the *same weights* serve a low-latency captioning use case (small lookahead) and a higher-accuracy near-real-time use case (larger lookahead). The lesson, reusable in any streaming system, is that latency is set by how much future you wait for, not by how fast you compute — so to cut latency, cut the lookahead, and to control the accuracy cost of doing so, train the model under that constraint so it learns to do well with limited future.

### Second-order optimization: cache the past, bound the future

The two ideas that make streaming work are reusable far beyond ASR. **Cache the past**: anything you have already computed and will need again should be stored, not recomputed — the same principle as the [KV cache in LLMs](/blog/machine-learning/large-language-model/kv-cache-optimization-and-management), here applied to encoder states. **Bound the future**: latency is determined entirely by how far ahead the model must see before it can emit, so to control latency you cap the lookahead, accepting a small accuracy cost for a large latency win. Any streaming model — ASR, live translation, online detection — lives or dies on these two: reuse past computation, and never wait for more future than the latency budget allows.

## 7. Canary vs Parakeet: same backbone, different missions

With the encoder and decoders understood, the two model families fall into place — they are the same FastConformer encoder pointed at different missions.

![Matrix comparing Canary and Parakeet: both use the FastConformer encoder; Canary pairs it with a Transformer attention decoder for ASR plus translation across 25 languages, while Parakeet pairs it with CTC, RNN-T, or TDT decoders for fast ASR on audio up to 11 hours long](/imgs/blogs/canary-parakeet-fastconformer-asr-8.webp)

- **Parakeet** = FastConformer + **CTC / RNN-T / TDT**. The transcription specialist: fast, accurate, English-and-multilingual ASR, with the transducer variants for streaming and the long-audio capability (up to **11 hours** in one pass on an A100 80GB via local/limited-context attention). When you need to transcribe audio quickly and accurately, especially long recordings, Parakeet is the model.
- **Canary** = FastConformer + **attention encoder-decoder**. The multilingual generalist: ASR in 25 languages *and* speech translation between English and the other 24, all in one model, selected by a task token. The attention decoder's flexibility is what enables translation (the output language is decoupled from the input), at the cost of being non-streaming and slower than the transducers.

The **"less is more" data strategy** is worth highlighting: Canary achieves strong multilingual results trained on *less* data than some competitors, by emphasizing **data quality and synthetic data** over raw scale — the same philosophy as the [Nemotron-4 340B synthetic-data approach](/blog/machine-learning/large-language-model/nemotron-4-340b-synthetic-data-alignment). High-quality, well-curated, partly-synthetic training data beats a larger pile of noisy data, in speech as in language.

### Why translation wants an attention decoder

It is worth understanding why Canary's translation capability *requires* the attention decoder and could not be done with a transducer. Transcription is **monotonic** — the output tokens follow the input audio in order, left to right, which is exactly what transducers (RNN-T, TDT) and CTC assume and exploit (they consume frames in order and emit tokens in order). Translation is **non-monotonic** — the word order of the target language differs from the source, so the output cannot simply track the input left to right (German puts verbs at the end; the translation of a sentence's first word might be its last). A transducer's monotonic alignment assumption breaks for translation. An attention decoder, by contrast, cross-attends *freely* to any part of the encoder output at each output step, so it can reorder — attend to the end of the audio to produce the start of the German sentence. This is why Canary uses the attention decoder and accepts its non-streaming cost: the flexibility to reorder is *necessary* for translation, not optional. The lesson is that the output structure of your task (monotonic vs reordering) dictates the decoder: monotonic tasks can use the fast, streaming, order-preserving decoders, while reordering tasks need the slower, more flexible attention decoder. The decoder zoo is not arbitrary — each member fits a class of output structure.

### Second-order optimization: the decoder choice is a product decision

The lesson from the two families is that, once the encoder is fixed, **the decoder choice is a product decision, not a research one**. Do you need streaming? Use a transducer. Do you need maximum speed? Use CTC. Do you need translation or many languages in one model? Use attention. Do you need long-audio transcription? Use Parakeet's transducer with local attention. The encoder gives you the representation; the decoder selects the product. This decoupling is why NVIDIA can serve such a wide range of speech products from one architectural foundation — and it is the reusable organizational insight: invest in one excellent backbone, then ship products by varying the cheap, task-specific head.

## 8. Failure modes the design guards against

As with the rest of the series, the architecture is a set of safeguards against specific failure modes.

- **Encoder cost exploding on long audio.** Symptom: quadratic attention blows up on long clips. Cause: too many frames. Safeguard: 8× downsampling cuts the frame rate (§2).
- **Downsampling being too expensive to do aggressively.** Symptom: more downsampling layers cost too much. Cause: standard convolutions. Safeguard: depthwise-separable convolutions make extra downsampling cheap (§3).
- **CTC producing inconsistent transcripts.** Symptom: locally-plausible but globally-wrong output. Cause: no internal language model. Safeguard: use a transducer (RNN-T/TDT) when accuracy matters, or add an external LM to CTC (§4).
- **Transducers wasting decode steps on blanks.** Symptom: slow decoding despite a good model. Cause: one step per frame, mostly blanks. Safeguard: TDT predicts durations and skips blanks (§5).
- **Streaming latency from unbounded lookahead.** Symptom: live transcription lags. Cause: the model waits for too much future context. Safeguard: cache-aware streaming with bounded lookahead (§6).
- **Needing separate offline and streaming models.** Symptom: two models to train and maintain. Cause: offline and streaming have opposite context needs. Safeguard: one cache-aware model handles both via the lookahead setting (§6).
- **Multilingual ASR needing per-language models.** Symptom: 25 models for 25 languages. Cause: a transcription-only decoder. Safeguard: Canary's attention decoder does all languages and translation in one model (§7).

The meta-lesson, again, is that each design choice is a "don't" — don't process too many frames, don't use expensive convolutions, don't decode blanks one by one, don't wait for unbounded future — and the architecture is the disciplined accumulation of those don'ts.

## 9. Case studies from the FastConformer line

### 1. The 2.4× speedup from one number

The headline FastConformer result — 2.4× faster than Conformer at the same accuracy — traces almost entirely to one change: 8× subsampling instead of 4×. It is a case study in how a single, well-chosen hyperparameter can dominate an architecture's efficiency. The frame rate is the lever, and halving it (relative to the 4× Conformer) roughly halves the encoder's work. The lesson is to find the *one number* that dominates your cost — here, frames per second — and attack it directly, rather than chasing diffuse micro-optimizations across the whole model. The biggest wins usually come from one structural change, not a hundred small ones.

### 2. The 11-hour transcription

Parakeet transcribing up to 11 hours of audio in a single pass on one A100 is a case study in how efficiency compounds into capability. It is enabled by the stack: 8× downsampling (fewer frames), local/limited-context attention (so attention is not quadratic over the whole 11 hours), and the memory headroom those create. No single trick gets you to 11 hours — it is the combination of a low frame rate and bounded-context attention. The lesson is that a qualitatively new capability (transcribe a whole audiobook in one pass) often emerges from stacking quantitative efficiencies past a threshold, not from one breakthrough. Efficiency is not just cost reduction; past a point it unlocks things that were simply impossible before.

### 3. CTC vs RNN-T as a speed/accuracy dial

The coexistence of CTC and RNN-T heads on the same encoder is a clean case study in the speed/accuracy trade-off. CTC is non-autoregressive and decodes in one pass — fastest, but no internal language model, so less accurate on hard audio. RNN-T is autoregressive with an internal LM — more accurate, but slower. The same encoder feeds both, so the choice is purely a deployment decision: real-time captioning with a downstream LM might use CTC; high-accuracy offline transcription uses RNN-T. The lesson is that decoder choice lets you slide along the speed/accuracy curve *without retraining the encoder*, which is exactly the flexibility a shared backbone is supposed to provide.

### 4. TDT's blank-skipping

TDT's speedup is a case study in eliminating structural waste. A transducer's frame-by-frame decoding spends most of its steps on blanks — a known inefficiency that people tolerated because it seemed intrinsic to the transducer formulation. TDT showed it is not intrinsic: by predicting durations, you skip the blanks. The lesson is to question "intrinsic" inefficiencies — the frame-by-frame blank emission *felt* like an unavoidable part of how transducers work, but it was really just one way to formulate them, and a better formulation (token + duration) removes the waste. When something is slow "because that's how the algorithm works," ask whether a different formulation of the same algorithm avoids the slow part.

### 5. The reduced kernel size after downsampling

A subtle case study: FastConformer reduces the convolution kernel size to 9 (from larger values in the Conformer) precisely *because* of the 8× downsampling. After 8× subsampling, each frame spans 8× more real time, so a kernel of size 9 reaches the same effective temporal context that a larger kernel reached at the higher frame rate. The lesson is that architecture hyperparameters are *coupled* — changing the frame rate changes what kernel size you need, because kernel size measures context in frames and the frames now mean more time. When you change one structural parameter (downsampling), re-derive the others (kernel size) that were calibrated to the old setting, rather than carrying them over unchanged.

### 6. "Less is more" data for Canary

Canary's strong multilingual results from less training data than competitors is a case study in data quality over quantity, echoing the [Nemotron-4 340B synthetic-data thesis](/blog/machine-learning/large-language-model/nemotron-4-340b-synthetic-data-alignment). High-quality, well-curated, partly-synthetic speech data — carefully filtered, with synthetic augmentation for rare languages and conditions — outperforms a larger but noisier corpus. The lesson, identical across speech and language, is that the marginal value of clean data far exceeds the marginal value of more noisy data, and that for multilingual and low-resource settings, synthetic generation of high-quality examples is a powerful lever. Scale matters, but quality-adjusted scale matters more.

### 7. One encoder, a whole product line

The shared FastConformer encoder underpinning both Parakeet and Canary is a case study in amortization. Training a strong speech encoder is expensive; NVIDIA does it once and derives a transcription specialist (Parakeet) and a multilingual translator (Canary) by varying the decoder. This is the same "train once, derive many" pattern as [Minitron's family](/blog/machine-learning/large-language-model/nvidia-minitron-pruning-distillation) and [Nemotron-H's sizes](/blog/machine-learning/large-language-model/nemotron-h-hybrid-mamba-transformer). The lesson is that the encoder/decoder split is an *amortization boundary*: put your expensive, general training investment in the encoder, and let cheap, swappable decoders fan it out into products. The boundary you choose for sharing determines how much you can derive from one investment.

### 8. Cache-aware streaming unifying two products

The cache-aware streaming design unifying offline and streaming ASR into one model is a case study in collapsing product complexity. Maintaining two models — one offline, one streaming — doubles training, evaluation, and serving burden, and risks the two drifting apart in quality. Training one model under the streaming regime, controllable by the lookahead at inference, gives you both behaviors from one set of weights. The lesson echoes the [Llama-Nemotron reasoning toggle](/blog/machine-learning/large-language-model/llama-nemotron-efficient-reasoning-models): build the mode-switching into the model rather than maintaining separate models, and control the behavior at inference time. One model with a knob beats two models with a router.

### 9. Relative positional encoding for variable-length audio

A detail worth a case study: the Conformer's attention uses *relative* positional encoding, not absolute, and this matters for audio. Audio clips vary enormously in length (a one-second command, an eleven-hour audiobook), and absolute position embeddings would generalize poorly across that range — position 50,000 is unseen during training on short clips. Relative positional encoding, which encodes the *distance* between frames rather than their absolute index, generalizes to any length, which is part of what makes the 11-hour transcription possible. The lesson is that for variable-length inputs, relative positioning is far more robust than absolute, because it never sees an out-of-range position — a principle that applies to any model processing inputs whose length varies by orders of magnitude.

### 10. The macaron FFN structure

The Conformer block's two *half*-weighted FFNs (the macaron structure) rather than one full FFN is a small but real case study. Splitting the FFN into two halves, one before and one after the attention-and-convolution mixers, and weighting each at 0.5, was found to improve over a single FFN — the model benefits from per-frame processing both before and after the sequence mixing. The lesson is that *where* you place the per-position processing relative to the sequence mixing matters, and sandwiching the mixers between two lighter FFNs can beat a single heavy one. Architecture is not just which components you include but how you order and weight them, and the macaron arrangement is a learned-from-experiments improvement that FastConformer kept.

### 11. Local attention for long audio

Parakeet's use of *local* (limited-context) attention for long-audio transcription is a case study in trading global context for tractability where the trade is safe. Full global attention over 11 hours is quadratic and impossible; local attention, where each frame attends only to a window of nearby frames, is linear and feasible. For ASR this trade is largely safe because speech recognition is mostly a *local* task — recognizing a word depends far more on its acoustic neighborhood than on something said an hour ago — so bounding the attention window costs little accuracy. The lesson is to match the attention span to the actual dependency range of the task: if the task is local, local attention captures it at a fraction of the cost, and the apparent loss of global context is mostly illusory because the task did not need it.

### 12. The encoder as the transferable asset

A strategic case study: the FastConformer encoder is the asset that transfers across NVIDIA's speech products and across versions. As Canary and Parakeet evolve (v1, v2, v3), the encoder improvements carry forward and benefit every decoder head. The encoder is where the accumulated investment lives. The lesson, for anyone building a model family, is to identify the *transferable asset* — the component whose improvements benefit everything downstream — and concentrate investment there. For speech, it is the acoustic encoder; for the NVIDIA LLM line, it is the base model; in both, the derived products inherit the backbone's gains automatically, which is the whole point of the architecture.

### 13. TDT and CTC as complementary speed plays

A nuanced case study: TDT and CTC both make decoding fast, but in opposite ways, and the contrast is instructive. CTC is fast because it is *non-autoregressive* — it gives up the internal language model to decode in one parallel pass. TDT is fast because it *skips blanks* — it keeps the autoregressive internal LM (so it stays accurate) but reduces the number of steps. So CTC trades accuracy for speed, while TDT gets speed *without* trading accuracy, by attacking a different source of slowness (wasted steps rather than sequential dependency). The lesson is that "make it faster" has multiple independent levers — parallelism and step-reduction are different — and the best one depends on what you are willing to give up. TDT is often the better choice precisely because its speedup is nearly free, where CTC's costs accuracy.

### 14. Synthetic data for rare languages and conditions

Extending the "less is more" theme, a case study in *targeted* synthetic data: for the long tail of languages and acoustic conditions where real labeled audio is scarce, synthetic generation (text-to-speech to create audio for under-represented text, or augmentation to simulate noise, reverberation, and accents) fills the gaps that real data cannot. This is how a multilingual model reaches 25 languages without 25 equally-large real corpora. The lesson is that synthetic data is most valuable exactly where real data is scarcest — the head of the distribution has plenty of real data, the tail does not, and synthetic generation is a tail-filling tool. Aim it at the gaps, not the abundance.

### 15. The product-line breadth from a research encoder

Stepping back, the sheer breadth of products derived from the FastConformer encoder — streaming and offline, transcription and translation, English and 25-language, short-command and 11-hour, fast-CTC and accurate-transducer — is itself a case study in research-to-product leverage. One encoder architecture, plus a menu of decoders and inference modes, becomes a whole catalog. The lesson is that the return on a strong, general backbone is measured in how many distinct products it can spawn, and FastConformer's return is unusually high because the encoder/decoder split and the cache-aware streaming design were built for fan-out from the start. Designing for derivability — making it easy to spin off variants — multiplies the value of the core investment.

### 16. Why speech borrowed from vision and language

A final reflective case study: FastConformer is a synthesis of ideas from neighboring fields. Depthwise-separable convolutions came from efficient *vision* models (MobileNets); the attention and the encoder/decoder structure came from *language* models (Transformers); the transducer and CTC came from *speech*'s own lineage. The Conformer and FastConformer are what you get when you take the best mixing primitive from each field and combine them for audio. The lesson is that the strongest architectures are often *syntheses* — borrowing the right primitive from each domain rather than inventing everything fresh — and that staying fluent across vision, language, and speech lets you transplant a technique (like depthwise-separable convs) from where it was invented to where it is newly useful.

### 20. The encoder/decoder split versus an end-to-end model

A comparative case study: an alternative design is a single end-to-end model that maps audio straight to text with no clean encoder/decoder boundary — and some architectures take that route. FastConformer's deliberate split, with a reusable encoder and swappable decoders, is a design choice with real consequences. The cost is a slightly less integrated model (the encoder is trained to produce a representation that *several* decoders can use, which is marginally less optimal than co-training one encoder with one decoder end to end). The benefit is enormous flexibility — one encoder, many products — and the ability to improve the encoder once and have every decoder benefit. NVIDIA judged the flexibility worth the small integration cost, and the breadth of the resulting product line vindicates that judgment. The lesson is that modularity has a price (slightly less end-to-end optimality) and a payoff (reuse and flexibility), and for a product line serving many use cases, the payoff usually dominates. A perfectly co-trained single-purpose model wins one benchmark; a modular backbone wins a catalog.

### 21. Versioned improvements that compound

A final case study on the model line's evolution: Canary and Parakeet ship in versions (v1, v2, v3), and each version's improvements — better training data, refined architectures, new languages — accrue to the shared foundation. Because the encoder is shared and the decoders are modular, an improvement to the encoder in v3 lifts every decoder head, and a new decoder capability can be added without disturbing the encoder. This versioning-on-a-shared-foundation is how a model line stays current without constant from-scratch rebuilds. The lesson, familiar from software engineering, is that a well-factored foundation lets you ship incremental improvements that compound, whereas a monolithic model forces a rebuild for each significant change. The architecture's modularity is not just a training-time convenience; it is what lets the product line *evolve* over years, each version building on the last rather than replacing it.

## The bigger picture: efficiency unlocks ubiquity

Step back and the FastConformer line makes a point that is easy to miss amid the architecture details: **efficiency is what makes a capability ubiquitous**. Accurate speech recognition existed before FastConformer — large attention encoders could transcribe well. What FastConformer changed is the *cost*, and cost is what determines whether a capability is a research demo or a feature in everything. At 2.4× the speed and a quarter of the frames, transcription becomes cheap enough to run on every meeting, every video, every voice command, every call — not as a premium feature but as a default. The 11-hour single-pass capability turns "transcribe this audiobook" from an overnight batch job into an interactive operation. The streaming design turns "transcribe my speech" into something that happens live with imperceptible latency. None of these are new *capabilities* in the sense of doing something previously impossible; they are the same capability made *cheap and fast enough to be everywhere*.

This is the thread that ties the speech work to the rest of the NVIDIA series. Every report in this series is, at bottom, about making a capability affordable enough to deploy at scale: [Minitron](/blog/machine-learning/large-language-model/nvidia-minitron-pruning-distillation) makes model families affordable, [Nemotron-4](/blog/machine-learning/large-language-model/nemotron-4-340b-synthetic-data-alignment) makes alignment data affordable, [Llama-Nemotron](/blog/machine-learning/large-language-model/llama-nemotron-efficient-reasoning-models) makes reasoning affordable, [Nemotron-H](/blog/machine-learning/large-language-model/nemotron-h-hybrid-mamba-transformer) makes long context affordable, and FastConformer makes speech recognition affordable. The recurring move is to find the one cost that dominates — token count, human labels, KV cache, frame rate — and attack it structurally. The capability was always possible; the engineering makes it ubiquitous.

For practitioners, the reusable mindset is to **ask what dominates your cost and whether the field's default is wasteful there**. ASR defaulted to 100 fps because nobody had questioned it; FastConformer questioned it and found 12.5 fps suffices. Your domain has its own unquestioned default — an input resolution, a sampling rate, a context length, a precision — that tradition set higher than necessary. Finding and cutting it is often the single highest-leverage thing you can do, because cost scales with that default and quality usually does not depend on all of it. FastConformer is a master class in that move: one number (the frame rate), cut in half, for a 2.4× win and a cascade of new capabilities.

### 17. The task token: one decoder, many behaviors

Canary's attention decoder does ASR *and* translation *and* 25 languages from one set of weights, and the mechanism is a **task token** — a special token prepended to the decoder's input that tells it what to do ("transcribe English," "translate to German"). This is the same conditional-behavior-via-a-control-token pattern as the [Llama-Nemotron reasoning toggle](/blog/machine-learning/large-language-model/llama-nemotron-efficient-reasoning-models): rather than building separate models for separate tasks, you train one model on a mix of tasks, each labeled with its task token, and the model learns to condition its behavior on the token. At inference you select the behavior by setting the token. The case study is a reminder of how powerful this pattern is — it turns N models into one model with N modes, and the modes can be combined (transcribe-then-translate) in ways separate models cannot. The lesson is that multi-task and multilingual capability is often best achieved not by N specialized models but by one model conditioned on a control token, which amortizes training and enables composition.

### 18. Why the encoder/decoder boundary is the right amortization point

A meta case study on architecture boundaries: NVIDIA chose to share the *encoder* and vary the *decoder*, and that specific boundary is what makes the product line work. Why not share the decoder and vary the encoder? Because in ASR the encoder is the expensive, general, hard-to-train part (it learns the acoustic representation from raw audio), and the decoder is the cheap, task-specific part (it converts a good representation into the desired output format). The expensive-and-general component is the right thing to share; the cheap-and-specific component is the right thing to vary. Get the boundary backwards — share the decoder, retrain the encoder per task — and you would pay the expensive cost repeatedly while sharing the cheap part. The lesson generalizes to any modular system: the sharing boundary should put the expensive, general work on the shared side and the cheap, specific work on the varied side, so you amortize the cost that is worth amortizing. FastConformer's encoder/decoder split is a textbook example of choosing that boundary correctly.

### 19. Multilingual as a capability multiplier

A strategic case study: Canary's 25-language support is not 25 separate achievements but one capability that multiplies the model's reach. A single multilingual model is far more valuable than 25 monolingual ones — it shares parameters and learning across languages (so low-resource languages benefit from high-resource ones), it is one model to deploy and maintain instead of 25, and it can do cross-lingual tasks (translation) that monolingual models cannot. The "less is more" data strategy is what makes it feasible: rather than needing a massive corpus per language, the model leverages cross-lingual transfer and synthetic data to reach languages where real data is thin. The lesson is that multilingual capability, done well, is a multiplier rather than a sum — the languages reinforce each other, and one model covering many is worth more than the sum of many models covering one each.

### 22. The downsampling module as the highest-leverage component

A focusing case study: of all the components in FastConformer, the **subsampling module** — the few depthwise-separable convolution layers at the very front that do the 8× downsampling — is wildly disproportionate in its impact. It is a small fraction of the model's parameters, yet it determines the frame rate that every downstream layer inherits, and the frame rate is the dominant cost. Get the subsampling right (8×, cheap convs, 256 channels) and the whole encoder is fast; get it wrong (too little downsampling, expensive convs) and no amount of downstream optimization recovers the lost efficiency. The lesson is that in many architectures there is one early component whose decisions propagate through everything after it — here, the thing that sets the sequence length — and that component deserves disproportionate design attention. Optimizing the big downstream layers while leaving an over-resolution front end in place is optimizing the wrong thing; the leverage is at the point where the sequence length is set.

### 23. Speech as the proving ground for cross-modal techniques

A reflective closing case study: speech sits between vision and language — it is a sequence (like language) of perceptual features (like vision) — and FastConformer shows how speech becomes a *proving ground* where techniques from both neighbors combine. Depthwise-separable convolutions (from vision) cut the convolution cost; attention and the encoder/decoder structure (from language) provide global context and flexible decoding; the transducer and CTC (from speech's own lineage) handle the alignment problem. The result is stronger than any single field's toolkit would produce alone. The lesson, and a fitting one to end on, is that the most fertile architecture work often happens at the *intersections* of modalities, where you can transplant the best primitive from each neighbor — and that an engineer fluent across vision, language, and speech has a structural advantage, because they can see which technique from one domain solves a problem in another. FastConformer is what that cross-modal fluency produces: a speech model built from the best ideas of vision and language, fitted to audio's particular structure.

## When to reach for FastConformer / Canary / Parakeet — and when not to

**Reach for them when:**

- **You transcribe long audio.** The 8× downsampling and local attention make long-form transcription (meetings, lectures, audiobooks) tractable where standard models choke.
- **You need streaming and offline from one model.** Cache-aware streaming gives you both, controlled by the lookahead.
- **You need multilingual ASR or translation.** Canary's attention decoder does 25 languages and translation in one model.
- **You need a speed/accuracy dial.** The CTC/RNN-T/TDT decoder zoo lets you pick the trade-off per deployment on a shared encoder.
- **You serve at scale.** FastConformer's 2.4× encoder speedup directly cuts serving cost, and TDT cuts decoding cost on top.

**Skip them (or look elsewhere) when:**

- **Your audio is always short.** For one-second commands, the frame-rate advantage is small, and a simpler model may suffice.
- **You need maximal accuracy regardless of cost.** A larger, slower attention model with full global context might edge out the efficient variants on the hardest benchmarks.
- **You lack the NeMo/Riva tooling.** The streaming, long-audio, and TDT features are realized through NVIDIA's speech tooling; without it you reimplement a lot.
- **Your task is not transcription/translation.** For speech *generation* (TTS) or *understanding* (intent, emotion), this encoder is a starting point but not the whole solution — see the [CosyVoice training](/blog/machine-learning/deep-learning/training-cosyvoice) and [speech tokenizer](/blog/machine-learning/deep-learning/speech-tokenizer) discussions.

The one-sentence version:

> The cost of an ASR encoder is the number of frames it processes, so cut the frame rate hard with cheap depthwise-separable downsampling, share one strong FastConformer encoder across a zoo of decoders matched to each task, and you get fast, accurate, streaming, long-audio, multilingual speech recognition from a single architectural foundation.

## Further reading

- [FastConformer / Canary / Parakeet model docs (NVIDIA NeMo)](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/models.html) — the architecture and model-card details.
- [Training ASR models](/blog/machine-learning/deep-learning/training-asr-models) — the broader ASR training picture this architecture slots into.
- [Speech tokenizer](/blog/machine-learning/deep-learning/speech-tokenizer) — how audio becomes discrete units, a complementary view of speech representation.
- [Training CosyVoice](/blog/machine-learning/deep-learning/training-cosyvoice) — the speech-generation counterpart to recognition.
- [Why convolution works](/blog/machine-learning/computer-vision/why-convolution-works) — the locality intuition behind the convolution module and depthwise-separable convs.
- [Choosing the right LLM architecture for a task](/blog/machine-learning/large-language-model/choosing-right-llm-architecture-task) — the same "match the architecture to the workload" thinking, on the language side, and a useful companion when deciding between encoder, decoder, and streaming trade-offs for a given speech deployment.
- [Nemotron-4 340B synthetic data](/blog/machine-learning/large-language-model/nemotron-4-340b-synthetic-data-alignment) — the "less is more" data philosophy applied to language.

One last reflection connects this post to the series and to what follows. The thread running through the NVIDIA reports is that *the dominant cost is rarely where you first look, and attacking it structurally beats optimizing around it*. For language models the cost was tokens, human labels, and the KV cache; for speech it is the frame rate. In each case the team found the one quantity that dominates and cut it at the source — fewer tokens via distillation, fewer labels via a reward model, no cache via state-space layers, fewer frames via downsampling. The next post pushes this further into a new modality entirely: Cosmos, where the dominant cost is the sheer size of video data, and the structural answer is a tokenizer that compresses video by up to 2048× before any world model touches it. The pattern is the same — find the cost, compress it at the source — applied to pixels and time instead of tokens and frames. If FastConformer is the speech chapter of NVIDIA's efficiency playbook, Cosmos is the chapter on the physical world, and the underlying move is identical: make the expensive thing small before you do the expensive computation on it — whether that expensive thing is a pile of audio frames, a context window of tokens, or hours of high-resolution video, the discipline of compressing at the source before computing is what turns an impressive demo into a deployable system.

*Next in the series: Cosmos, NVIDIA's world foundation models — the tokenizer that compresses video up to 2048x, and the diffusion and autoregressive models that learn to predict the physical world.*
