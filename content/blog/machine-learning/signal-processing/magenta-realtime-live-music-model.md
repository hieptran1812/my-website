---
title: "Magenta RealTime: inside an open-weights live music model"
date: "2026-06-10"
publishDate: "2026-06-10"
description: "A principal-engineer deep dive into Magenta RealTime — how a ~800M block-autoregressive model streams 48 kHz stereo music faster than real time, steered live by text and audio, from the SpectroStream codec up to classifier-free guidance."
tags:
  [
    "music-generation",
    "audio-codec",
    "residual-vector-quantization",
    "real-time-inference",
    "autoregressive-models",
    "spectrostream",
    "musiccoca",
    "classifier-free-guidance",
    "generative-audio",
    "streaming",
    "magenta",
  ]
category: "machine-learning"
subcategory: "Signal Processing"
author: "Hiep Tran"
featured: true
readTime: 52
---

Most of what we call "AI music" is a batch job wearing a creative costume. You type a prompt, a model thinks for ten or thirty seconds, and a finished clip falls out the bottom. You listen, you don't like the bridge, you change three words, you wait again. It is generation as a vending machine: insert prompt, receive artifact. The interaction loop is measured in coffee breaks, and the model has no idea whether you are still in the room.

Magenta RealTime — released by Google's Magenta and Lyria teams alongside the paper [*Live Music Models*](https://arxiv.org/abs/2508.04651) (arXiv:2508.04651) — throws that loop out. It is an **open-weights live music model**: a system that produces a *continuous, never-ending stream* of 48 kHz stereo music in real time, while you steer its style on the fly with text prompts, audio prompts, or weighted blends of both. There is no "generate" button to wait on. The music is already playing; your job is to play *it*, like an instrument that improvises with you.

That single design constraint — **the audio must come out faster than it is consumed, forever** — rewrites every layer of the stack. You cannot run a 30-second diffusion pass per clip. You cannot attend over a three-minute context. You cannot wait for the full-resolution codec to decode 64 quantizer levels before the speaker needs the next sample. Every component of Magenta RealTime is shaped by the real-time tax, and that is what makes it worth dissecting.

![The Magenta RealTime live-music loop: prompts feed MusicCoCa, the block-autoregressive LM generates SpectroStream tokens from a rolling 10-second context, and each decoded 2-second chunk is appended back into that context.](/imgs/blogs/magenta-realtime-live-music-model-1.webp)

The diagram above is the mental model, and the rest of this article is a tour of it. Read it as a loop, not a pipeline. A text or audio prompt is embedded by **MusicCoCa** into a small bundle of style tokens. A ~800M-parameter **block-autoregressive language model** consumes those style tokens plus a rolling **10-second context** of previously generated audio, and emits **SpectroStream** codec tokens for the next **2-second chunk**. SpectroStream decodes those tokens into 48 kHz stereo audio, which is played — and immediately appended back into the 10-second context so it conditions the *next* chunk. The loop turns once every two seconds of audio, and on a free Colab TPU it turns in about 1.25 seconds of wall-clock time. That gap between 2.0 and 1.25 is the entire reason this works.

## Why live generation is a different animal

Before we touch a single component, internalize the mismatch between how people assume music models work and what "live" actually forces. Almost every architectural decision below is downstream of this table.

| Question | The offline-model assumption | The live-model reality |
| --- | --- | --- |
| When does audio appear? | After the whole clip is generated. | Continuously, in 2-second chunks, indefinitely. |
| How much context can it see? | The whole song so far (tens of seconds to minutes). | A rolling **10-second window** — five 2-second chunks. |
| How is the codec used? | Decode all quantizer levels for max fidelity. | Generate only the **coarse first-16** of 64 RVQ levels; fill the rest at decode. |
| What is the latency budget? | Seconds per clip is fine. | Generate 2 s of audio in **< 2 s**, every time, or you underrun. |
| How does the user steer? | Re-prompt and regenerate from scratch. | Change the prompt **mid-stream**; it takes effect at the next chunk boundary. |
| What is the failure mode? | A bad clip. | An **audible glitch or silence** — the stream stalls in someone's ears. |

Offline models optimize a single number: quality per clip. Live models optimize a *distribution over wall-clock time*: every chunk must be good enough **and** ready on time, or the illusion of a live instrument collapses. A model that produces gorgeous music at a real-time factor of 0.9 is, for this purpose, broken — it falls behind and the audio buffer drains to silence.

This is the same shift that separates a batch renderer from a game engine. A film renderer can spend an hour on a frame; a game engine has 16 milliseconds, no exceptions, or it drops the frame and you see it. Magenta RealTime is the game engine of music generation, and like a game engine, its cleverness is mostly about *budgeting* — of tokens, of context, of quantizer depth — to hit a hard deadline. If you have read about [first-audio-byte latency in real-time TTS](/blog/machine-learning/signal-processing/real-time-tts-first-audio-byte-latency), the instincts transfer directly: the enemy is the deadline, and every design choice buys or spends time against it.

> A live model is not a faster offline model. It is a model that has accepted a deadline as a first-class loss term.

We will go layer by layer, bottom-up: the codec that defines what a "token" even is, the token budget that one chunk costs, the embedding model that turns words into steering, the block-autoregressive decoder that fills the budget on time, the real-time accounting that makes it stream, and the control surface you actually touch. Then seven war stories from building with it, and a closing call on when a live model is the right tool and when it is a trap.

## 1. SpectroStream: 48 kHz stereo at 4 kbps

**Senior rule of thumb: in a codec-LM stack, the codec decides the model's vocabulary and frame rate before the model designer makes a single choice.** Everything the language model can say is constrained by what the codec can represent and how many tokens per second it costs. So start there.

SpectroStream is Magenta's high-fidelity neural audio codec — the successor to SoundStream, and a cousin of the residual-vector-quantization (RVQ) codecs you have met if you read about [EnCodec, SoundStream, and Mimi](/blog/machine-learning/signal-processing/speech-tokenizers-encodec-soundstream-mimi). Like all of them, it is an encoder–quantizer–decoder sandwich: an encoder maps audio to a sequence of latent frames, a stack of vector quantizers turns each frame into discrete codes, and a decoder reconstructs the waveform from those codes. The discrete codes are what the language model predicts. No codec, no tokens; no tokens, no LM.

What makes SpectroStream worth its own section is three numbers and one structural choice.

- **48 kHz stereo.** Not 16 kHz mono speech, not 24 kHz — full-bandwidth, two-channel music. This is a much harder reconstruction target than speech codecs face, because music has wide-band transients (cymbals, plucks), stereo imaging, and harmonic structure that the ear is merciless about.
- **25 Hz frame rate.** The encoder downsamples hard: one latent frame every 40 milliseconds. That low frame rate is a deliberate gift to the language model — fewer frames per second means fewer autoregressive steps per second of audio.
- **64 RVQ levels, codebook size 1024.** Each frame is quantized by a stack of **64** residual quantizers, each with a 1024-entry codebook. At full depth this is roughly **16 kbps**. But — and this is the live trick — generation only uses the **first 16** levels, about **4 kbps**.

The structural choice: SpectroStream operates in the **time–frequency domain** (a spectrogram-like representation produced by a delayed-fusion encoder over the time–frequency plane), rather than SoundStream's purely time-domain convolutions. For 48 kHz music that pays off — the model gets a representation where harmonic and transient structure are easier to compress without smearing.

![SpectroStream stacks 64 residual quantizers; each level codes the leftover error from the one before. Live generation emits only the coarse first-16 levels (4 kbps); decode reconstructs all 64 (16 kbps).](/imgs/blogs/magenta-realtime-live-music-model-2.webp)

### How residual quantization actually buys fidelity

The figure above is the whole idea of RVQ on one page. Start with a latent frame $z_t$ — a real-valued vector the encoder produced for time step $t$. A single vector quantizer (VQ) replaces $z_t$ with the nearest entry in a learned codebook of 1024 vectors. That nearest-neighbor replacement has an error: the **residual** $r_1 = z_t - \hat{c}_1$, where $\hat{c}_1$ is the chosen codebook vector. One codebook of 1024 entries can only say so much; the residual is whatever it could not capture.

Residual quantization then quantizes the residual. A second codebook approximates $r_1$, leaving $r_2 = r_1 - \hat{c}_2$; a third approximates $r_2$; and so on. After $L$ levels, the reconstruction is the sum of all chosen codebook vectors:

$$
\hat{z}_t = \sum_{i=1}^{L} \hat{c}_i, \qquad r_i = r_{i-1} - \hat{c}_i, \quad r_0 = z_t.
$$

Each additional level chips away at the leftover error, so reconstruction quality climbs with depth while the bitrate climbs linearly. With $L = 16$ levels at 1024 entries each, every frame costs $16 \times \log_2 1024 = 16 \times 10 = 160$ bits; at 25 frames per second that is $160 \times 25 = 4000$ bits/s — the **4 kbps** live band. Push to the full $L = 64$ and you get $64 \times 10 \times 25 = 16{,}000$ bits/s — the **16 kbps** full-fidelity band.

The genius of the coarse-to-fine ordering is that the levels are *prioritized*: the first few codebooks carry the gross structure (which note, which instrument, roughly where in the stereo field), and the deep levels carry the fine timbral detail that makes a hi-hat sizzle instead of tick. So you can **truncate** the stack and still get coherent music — just less crisp. That truncation is exactly what the live model exploits.

### How the deep levels come back for free

If generation only emits 16 of 64 levels, where do the other 48 come from? They are *predicted at decode time* by a small model that never enters the real-time autoregressive loop. The decoder conditions on the coarse 16 codes and infers a plausible set of fine codes (or directly a higher-fidelity latent), then reconstructs the waveform. The reason this is "free" is purely about where the cost lands: the language model's budget is measured in *sequential autoregressive steps against a 1.25-second clock*, while the decode-side refinement is a single feed-forward pass that runs comfortably inside the per-chunk window. You move the expensive-looking part (48 extra levels) out of the loop that has the deadline and into the part that does not.

This split also explains why the coarse-to-fine ordering of RVQ is non-negotiable here. Because levels are prioritized — gross structure first, fine timbre last — the coarse 16 carry enough information for a learned decoder to *hallucinate believable* fine detail. If RVQ levels were unordered or equally important, you could not truncate to 16 and reconstruct from them; the missing levels would carry structure the decoder cannot guess. The whole live formulation leans on the fact that the first 16 codes are a faithful low-resolution summary, not a random subset.

Neural codecs like SpectroStream are trained with the usual machinery that makes 48 kHz reconstruction sound good: a reconstruction loss in the time–frequency domain, adversarial discriminators that punish unnatural artifacts, and feature-matching losses — the same family of techniques that give a [HiFi-GAN vocoder](/blog/machine-learning/signal-processing/hifi-gan) its sharpness. The discriminators matter for music specifically, because L2 spectral losses alone produce a dull, over-smoothed reconstruction; the adversarial term is what keeps transients crisp and cymbals from turning to mush. That perceptual quality is what makes the 4 kbps live band listenable rather than merely intelligible.

### Second-order: bitrate, token throughput, and the deadline are the same knob

Here is the non-obvious coupling that ties the codec to the deadline. The language model must emit one token per RVQ level per frame that it generates. So the number of tokens per second of audio is:

$$
\text{tokens/s} = (\text{frame rate}) \times (\text{RVQ levels generated}) = 25 \times 16 = 400.
$$

That **400 tokens/second** is the model's actual workload. If SpectroStream had used a 50 Hz frame rate, the same 16 levels would cost 800 tokens/s and the model would have to be twice as fast to keep up. If generation used all 64 levels, that is 1600 tokens/s — four times the work, almost certainly too slow for real time on a free TPU. The codec's low frame rate and the model's shallow generation depth are not independent decisions; they are two halves of one latency budget. SpectroStream was designed *so that* a tractable LM could hit real time, and the LM only generates 16 of 64 levels *because* that is what the deadline affords. The remaining 48 levels are reconstructed at decode time by a small model conditioned on the coarse codes — fidelity you get "for free" because it never enters the autoregressive loop.

This is the codec-LM contract in one sentence: **the codec sets the price of a second of audio in tokens, and the model spends exactly as many as the clock allows.**

## 2. The token budget: what two seconds costs

**Senior rule of thumb: when a system has a hard deadline, draw the unit of work as a fixed budget and account for every token in it.** For Magenta RealTime, the unit of work is one 2-second chunk, and it has a precisely countable cost.

![A 2-second chunk is 50 frames x 16 levels = 800 tokens; live uses 16 of 64 RVQ levels so the budget fits a 1.25-second generation window.](/imgs/blogs/magenta-realtime-live-music-model-3.webp)

Walk the multiplication in the figure. A chunk is 2.0 seconds. At a 25 Hz frame rate, that is $2.0 \times 25 = 50$ frames. Each frame the model generates carries 16 RVQ codes (the live band). So one chunk is:

$$
50 \ \text{frames} \times 16 \ \text{levels} = 800 \ \text{tokens}.
$$

Eight hundred tokens is the entire generative task per chunk. Compare it to the full-fidelity alternative: $50 \times 64 = 3200$ tokens. By generating only the coarse 16 levels, the model does **4× less autoregressive work** per chunk — and that 4× is the difference between landing inside the ~1.25-second generation window and missing it.

### The rolling-context trick

Now look at the other half of the budget: context. The model conditions on the previous **10 seconds** of audio — five prior 2-second chunks. If those chunks entered the context at full live depth (16 levels), the context alone would be $5 \times 800 = 4000$ tokens, and the model would be attending over 4000 + 800 = 4800 tokens per step. That is a lot of attention to do 400 times a second.

So the prior chunks do **not** enter at 16 levels. They enter at the **coarse first-4** levels only. Five chunks of context at 4 levels is $5 \times 50 \times 4 = 1000$ tokens instead of 4000 — a 4× reduction on the conditioning side, mirroring the 4× reduction on the generation side. The history that conditions the next chunk is a low-resolution sketch of what just played: enough to keep the groove, the key, and the instrumentation coherent, cheap enough to carry ten seconds of it without blowing the attention budget. MusicCoCa-style conditioning, separately, uses the first 6 RVQ levels at inference.

This is a recurring pattern in real-time generative systems and worth naming: **resolution is a currency, and you spend it where the ear notices.** Full 64-level detail at decode, where it is cheap and perceptually huge. Sixteen levels for generation, where each level is an autoregressive tax. Four levels for context, where you only need the gist. If you have tuned [streaming ASR pipelines](/blog/machine-learning/signal-processing/streaming-asr-production-pipeline), this is the same chunk-and-context bookkeeping, pointed at synthesis instead of recognition.

### A worked number: the slack

Let's make the deadline concrete. The model must produce 800 tokens per chunk, and it has roughly 1.25 seconds of wall-clock to do it (on the free TPU), for 2.0 seconds of audio. That is:

$$
\frac{800 \ \text{tokens}}{1.25 \ \text{s}} = 640 \ \text{tokens/s of throughput},
$$

against an audio consumption rate of $400$ tokens/s. The model generates tokens about **1.6× faster than the audio plays them**, which is exactly the real-time factor (RTF) we will return to in §5. The 0.75 seconds of slack per chunk (2.0 − 1.25) is the buffer that absorbs jitter — a slow step here, a thermal hiccup there — without the stream stalling. Shrink the slack and you live dangerously; that is precisely what happens under load, and it shows up in one of our case studies.

## 3. MusicCoCa: steering with text and audio

**Senior rule of thumb: a controllable generative model is only as good as its conditioning space — if text and audio prompts do not live in the same geometry, you cannot blend them.** Magenta RealTime's control surface is a small joint embedding model called **MusicCoCa**, and it is the reason you can say "warm lo-fi" *and* hum a reference clip and have the model understand both as points in one space.

![MusicCoCa: a text tower and an audio tower are contrastively aligned into one 768-d space, then quantized to 12 style tokens that condition the language model.](/imgs/blogs/magenta-realtime-live-music-model-4.webp)

The name is a portmanteau of its two ancestors: **MuLan**, Google's music–language contrastive model, and **CoCa** (Contrastive Captioner), which combines contrastive alignment with a generative captioning objective. MusicCoCa inherits the two-tower structure you see in the figure:

- A **text tower**: a 12-layer Transformer that reads up to 128 tokens of natural-language description ("driving techno with a detuned bassline").
- An **audio tower**: a 12-layer Vision-Transformer-style encoder that reads log-mel spectrograms (128 mel channels, 16×16 patches) of a reference clip.

Both towers project into a **shared 768-dimensional space**, and the model is trained so that a piece of audio and a text description of that audio land near each other — the contrastive objective — while a generative captioning loss (weighted roughly equally) keeps the text side grounded in real musical language. Training ran for about 16,000 steps at batch size 1024. The payoff of the joint space is the thing that makes live steering feel magical: because "disco funk" the phrase and a disco-funk *clip* map to nearby points, you can mix them, interpolate between them, and do arithmetic on them, and the language model downstream cannot tell which modality a style token came from.

### From a 768-d vector to 12 tokens

The language model does not consume a raw 768-d float vector — it consumes discrete tokens, like everything else in its world. So the MusicCoCa embedding is **quantized into 12 discrete style tokens** (codebook size 1024). Twelve tokens is a tiny, fixed-size style "preamble" that gets prepended to the generation context. It is deliberately small: style is a slowly-varying global property, not something that needs per-frame resolution. Twelve tokens of style, 1000 tokens of coarse audio context, 800 tokens to generate — that is the shape of one decoding step.

Quantizing the style has a subtle benefit beyond token-count discipline: it makes style a *categorical, composable* object. Because the conditioning is discrete and bounded, the model sees style during training as one of a finite vocabulary of "vibes," which regularizes how it generalizes to blends and interpolations at inference. It is the same reason discrete codecs generalize better than continuous ones for autoregressive generation — bounded vocabularies are friendlier to sample from.

### How the towers are actually aligned

The contrastive objective is the heart of why the joint space works, and it is worth being precise about. During training, the model sees batches of (audio clip, text caption) pairs. It pushes the embedding of each audio clip toward the embedding of its *own* caption and away from every *other* caption in the batch — and symmetrically for text toward audio. Formally this is the InfoNCE loss you have met in CLIP-style models: for a batch of $N$ pairs, the audio-to-text term maximizes the similarity of the matched pair against the $N-1$ mismatched ones, scaled by a learned temperature. The CoCa half adds a generative captioning decoder on top of the audio tower, trained to actually *write* the caption, which forces the audio representation to retain fine-grained, language-grounded musical attributes rather than collapsing to a few coarse genre clusters.

The practical consequence is the property we will exploit for steering: the space is **metric and smooth**. Distances mean something — two clips that sound alike are close — and straight-line paths between points pass through musically sensible intermediate styles. A space trained only to *classify* genre would give you sharp cluster boundaries and ugly interpolations; a space trained with contrastive-plus-generative alignment gives you a continuum you can walk. When you blend "lo-fi" and "jazz" by averaging their embeddings, the result lands in a region the model has implicitly seen as "music that is somewhat both," because the training pushed real lo-fi-jazz clips into exactly that neighborhood.

One more detail with downstream teeth: the audio tower reads **log-mel spectrograms**, not raw waveforms. That choice trades some fidelity of representation for a compact, perceptually-weighted input that a 12-layer ViT can digest cheaply — appropriate for a *conditioning* model whose job is to capture style, not to reconstruct audio. It is a different design point from SpectroStream, which must represent audio faithfully enough to decode it. Two encoders, two jobs, two input representations: the codec preserves everything; the style model keeps only what distinguishes one vibe from another.

### Second-order: the embedding arithmetic preview

Because text and audio share the 768-d geometry, *control becomes vector arithmetic*. Want something 70% lo-fi and 30% jazz? Embed both, take a weighted average, quantize, condition. Want to morph from one style to another over thirty seconds? Walk a straight line between their embeddings, re-quantizing as you go, and the music drifts continuously. We will formalize this in §6, but the key structural fact lives here in the embedding model: **steering is possible because the conditioning space is continuous, joint, and arithmetic-friendly, and only quantized at the very end.** Get the embedding space wrong and no amount of clever decoding gives you smooth live control.

## 4. Block-autoregressive generation

**Senior rule of thumb: the shape of your autoregressive factorization decides your latency, not just your loss.** Magenta RealTime's language model is a ~800M-parameter encoder–decoder Transformer built in T5X, and the interesting part is not its size — it is *how* it factorizes the 800-token-per-chunk problem so each step is short enough to hit the clock.

The naive approach would be a single flat autoregressive sequence: 800 tokens per chunk, generated one at a time, each conditioned on all previous tokens in the chunk plus context. That is 800 sequential forward passes per chunk. Even at a few milliseconds per pass, 800 passes blows past the 1.25-second budget. Flat AR over codec tokens does not stream.

Instead, Magenta RealTime uses a **block-autoregressive** structure with two factorized modules — the same idea as the RQ-Transformer and the depth-transformer trick you may recognize from [Orpheus TTS over the SNAC codec](/blog/machine-learning/signal-processing/orpheus-tts-llm-speech-snac). The factorization splits the problem along its two natural axes: **time** (50 frames) and **depth** (16 RVQ levels per frame).

![Block-autoregressive decode factorizes the 800-token chunk into an outer temporal loop over 50 frames and an inner depth loop over 16 levels, keeping each step's sequence short enough for real time.](/imgs/blogs/magenta-realtime-live-music-model-5.webp)

Read the figure as two nested loops:

- The **temporal transformer** is the outer loop. It advances one frame at a time across the 50 frames of the chunk, carrying the autoregressive state of "the music so far." At each frame it produces a context vector summarizing everything up to that moment.
- The **depth transformer** is the inner loop. Given the temporal context for frame $t$, it predicts the 16 RVQ levels of that frame **coarse-to-fine** — level 1 first (the gross structure), then level 2 conditioned on level 1, up to level 16. Sixteen small steps, each over a short sequence, instead of one giant step.

The win is that no single autoregressive step ever attends over 800 tokens. The temporal loop runs 50 steps; the depth loop runs 16 steps per frame over a tiny depth sequence. The sequences are short, the steps are cheap, and the total work fits the window. This is the canonical move for making codec-LMs real time: **factorize the joint distribution over (frame, level) so the inner products stay small.**

### The sampling recipe

The released configuration samples with concrete, reproducible knobs, and they matter more than they look:

- **Classifier-free guidance (CFG) weight 5.0** — a strong push toward the conditioning. We dissect CFG in §6; for now, 5.0 is aggressive, which is what you want for *responsive* steering at the cost of some diversity.
- **Temperature 1.3** — slightly above 1.0, biasing toward exploration so the stream stays alive and surprising rather than collapsing into a loop.
- **Top-K 40** — truncate the sampling distribution to the 40 most likely codes per step, a cheap guard against the long tail of low-probability codes that produce artifacts.

These are not arbitrary. A live model that samples too conservatively (low temperature, no top-K headroom) gets stuck in repetitive grooves — a real failure mode when there is no human re-prompting every clip. One that samples too loosely produces noise and instability. The recipe is tuned for *continuous* generation, where the model must stay interesting for minutes without supervision.

### Inside the T5X encoder–decoder

Why an encoder–decoder Transformer, rather than a decoder-only stack like most modern LLMs? Because the live-music problem has a natural split between *what conditions the generation* and *what is being generated*. The encoder ingests the conditioning — the 12 style tokens and the coarse rolling context — and produces a set of key/value representations. The decoder then generates the new chunk's tokens, attending to that fixed encoded conditioning via cross-attention while self-attending over the tokens it has produced so far. This is the classic T5 shape, and it maps cleanly onto "condition on context, emit the next two seconds."

The split has a real efficiency payoff in the streaming regime. The conditioning — style plus the coarse history — is encoded *once* per chunk and reused across all of that chunk's decoding steps. You are not re-encoding ten seconds of context on every one of the inner depth steps; you encode it at the top of the chunk and the decoder cross-attends to the cached result. For a system that runs the decoder hundreds of times per second, amortizing the context encoding across the chunk is exactly the kind of bookkeeping that buys back milliseconds against the deadline. It is the same instinct as a [KV cache](/blog/machine-learning/large-language-model/kv-cache) in text generation — compute the expensive shared thing once, reuse it on every step — applied at the granularity of a chunk.

The temporal and depth modules sit inside this decoder. The temporal module carries the frame-to-frame autoregressive state (the outer loop), and the depth module expands each frame into its 16 coarse-to-fine codes (the inner loop). Both are comparatively small; the ~800M parameter budget is split to keep each module's per-step compute low enough that 50 outer steps and 800 inner predictions fit the window. Size here is in service of the deadline, not raw capacity — a 7B model with the same factorization would model music better and miss the clock, which for a live model is the same as being wrong.

### Second-order: why block-AR beats frame-AR and flat-AR

There is a spectrum of factorizations, and the choice is a latency–quality tradeoff:

| Factorization | Steps per chunk | Per-step cost | Real-time? |
| --- | --- | --- | --- |
| Flat AR (one sequence of 800) | 800 | grows with position | No — too many sequential steps |
| Frame-parallel, depth-flat | 50 | predict 16 levels jointly | Risky — joint 16-way head is lossy |
| **Block-AR (temporal × depth)** | 50 outer × 16 inner | short sequences both | **Yes** |

Block-AR is the sweet spot: it keeps the autoregressive dependencies that matter (coarse-to-fine within a frame, frame-to-frame over time) while keeping every sequence short. Predicting all 16 levels of a frame in one shot would be faster but throws away the coarse-to-fine dependency that makes RVQ work — level 16 genuinely depends on what levels 1–15 chose. Flat AR keeps every dependency but pays 800 sequential steps. The block factorization is how you get *most* of the modeling power at a *fraction* of the step count. The figure's bottom line says it plainly: a flat 800-token run per chunk would blow the latency budget; factorization keeps each step short.

### Anatomy of one decoding step

Let's make the loop concrete by walking a single chunk end to end with numbers, because the abstract description hides how tightly everything is packed.

The clock ticks at $t$. Chunk $N$ is playing through the speaker; the model is generating chunk $N{+}1$. First, the encoder runs once: it ingests the 12 style tokens (current steering) plus the coarse rolling context — the first-4 RVQ levels of the last five chunks, about $5 \times 50 \times 4 = 1000$ tokens — and produces the cross-attention memory. That encode happens a single time for the whole chunk.

Then the decoder runs the nested loops. The temporal module steps across the 50 frames of the chunk. At frame $f$, it updates its autoregressive state and hands a context vector to the depth module. The depth module predicts 16 codes for that frame, coarse-to-fine: it samples code 1 (with CFG weight 5.0, temperature 1.3, top-K 40), conditions on it to sample code 2, and so on through code 16. That is $50 \times 16 = 800$ sampled codes for the chunk, each a draw from a 1024-way categorical distribution, every one of them cross-attending to the once-encoded conditioning.

Those 800 coarse codes go to SpectroStream's decoder, which expands them to the full reconstruction and emits 2.0 seconds of 48 kHz stereo PCM — about $2.0 \times 48{,}000 = 96{,}000$ samples per channel. The waveform is crossfaded against the tail of chunk $N$ and pushed into the playback buffer. Finally, the first-4 levels of these 800 codes are appended to the rolling context and the oldest chunk's coarse codes are dropped, keeping the window at ten seconds. The whole sequence — encode once, 50 temporal steps, 800 depth samples, one codec decode, one crossfade, one context update — completes in roughly 1.25 seconds, leaving 0.75 seconds before chunk $N$ finishes. Then $t$ advances and it happens again, forever.

That is the entire system in one paragraph, and every number in it traces back to a deadline decision: 1000 context tokens (coarse-4, not full), 800 generated tokens (16 levels, not 64), 50 temporal steps (25 Hz, not 50), one shared encode (encoder–decoder, not decoder-only). Pull any one of them the wrong way and the 1.25 seconds becomes 2.1 and the music stops.

## 5. Real time, actually

**Senior rule of thumb: "real time" is not a property of a model, it is an inequality between a generation clock and a playback clock — write the inequality down.** For Magenta RealTime the inequality is: time to generate 2 seconds of audio < 2 seconds. On a free-tier Colab **TPU v2-8**, it generates those 2 seconds in about **1.25 seconds**, for a **real-time factor (RTF) of 1.6**. The paper also reports the larger configuration hitting RTF ~1.8 on an H100. Anything above 1.0 means the model outruns playback; the headroom is what keeps the stream alive.

![One 2-second control cycle: rendering a chunk in 1.25 s of wall-clock builds a lead buffer, but a prompt change is only audible when the next chunk plays — so control latency is one chunk.](/imgs/blogs/magenta-realtime-live-music-model-6.webp)

The timeline figure traces a single control cycle and shows where the slack and the latency both come from. At $t = 0$, chunk $N$ starts playing and the user changes the prompt. By $t = 0.25$ s the LM has encoded the new style and the 10-second context. By $t = 1.25$ s — well before chunk $N$ finishes playing at $t = 2.0$ s — chunk $N{+}1$ is fully rendered and sitting in the buffer, with 0.75 seconds of slack to spare. At $t = 2.0$ s the player crossfades from $N$ to $N{+}1$, and only *now* does the new style become audible. By $t = 4.0$ s the stream is in steady state, always one chunk ahead of the speaker.

### The lead buffer is the product

RTF > 1.0 does not just mean "fast." It means the system can build and hold a **lead buffer**: at steady state, the next chunk is always already generated before the current one finishes. That buffer is what makes the audio *gapless*. Without it, any single slow step would drain the speaker to silence — an underrun, the audio equivalent of a dropped frame. The 0.75 s of slack per chunk is the depth of that buffer, and it is the single most important number for perceived reliability. A model at RTF 1.05 technically "keeps up" but has almost no buffer, so the first thermal throttle or scheduling hiccup produces an audible stall. RTF 1.6 means you can lose a third of your throughput and still not glitch.

### Crossfade: hiding the seam

Chunks are generated independently-ish (conditioned on shared context, but decoded as separate 2-second segments), and naive concatenation at the boundary produces a click — a discontinuity in the waveform where chunk $N$'s last sample meets chunk $N{+}1$'s first. The fix is a short **crossfade**: overlap the tail of $N$ with the head of $N{+}1$ and equal-power blend them, so the transition is a smooth ramp instead of a step. This is bread-and-butter audio engineering, but it is load-bearing here: without it, you would hear the chunk structure as a rhythmic tick every two seconds. The crossfade is why the 2-second chunking is *inaudible* even though it is the central architectural fact.

### Buffer management in practice

The lead buffer is simple to state and easy to get wrong. The rule is: never let the playback head catch the generation head. Concretely, you keep a queue of decoded chunks; the player pulls from the front at exactly real time (one chunk's worth of audio every 2.0 seconds), and the generator pushes to the back as fast as it can (one chunk every ~1.25 seconds at RTF 1.6). At steady state the queue hovers around one chunk deep — chunk $N{+}1$ ready while $N$ plays — and the 0.75-second slack accumulates into that depth.

The naive implementation generates exactly one chunk ahead and assumes RTF stays at 1.6. That works until it doesn't. A more robust design pre-rolls a *deeper* buffer before audio starts — generate three or four chunks while the user is still reading the loading spinner — so that several seconds of slack are banked. Then, when a single chunk takes 2.3 seconds instead of 1.25 (a thermal blip, a scheduler hiccup), the queue absorbs it: the player keeps pulling from the banked chunks and never hears the stall. The cost is a slightly longer startup latency — you wait an extra couple of seconds before the first note — which for most live applications is a fine trade against a mid-performance dropout.

The instrumentation that matters is per-chunk generation time, tracked as a rolling distribution rather than an average. If the 95th-percentile chunk time creeps toward 2.0 seconds, you are one bad afternoon from glitching, regardless of what the mean says. The right response is to deepen the buffer (more startup latency) or move to hardware with more headroom — not to hope the average holds. This is the same discipline as tail-latency engineering in request serving: the mean is a comfort, the tail is the truth.

### Second-order: the latency you cannot remove

Here is the hard limit of the whole approach. Because control is applied at chunk boundaries, **the minimum control latency equals the chunk size: 2 seconds.** When you change the prompt mid-chunk, the chunk currently rendering (or already in the buffer) cannot be un-generated; your change lands on the *next* one. This is "sample-and-hold" control: the model samples your intent at each boundary and holds it for the chunk. You feel it as a slight lag between turning a knob and hearing the result — about two seconds, worst case.

You cannot engineer this away without shrinking the chunk, and shrinking the chunk has its own costs: smaller chunks mean more frequent (and proportionally more expensive) context updates, more crossfade seams, and less coherent per-chunk structure. Two seconds is the chosen tradeoff between responsiveness and stability. The 10-second context has a symmetric cost: it is also the **maximum** musical memory, so the model cannot model long-term song structure — a verse that returns four minutes later is beyond its horizon. These two numbers, 2 s of latency and 10 s of memory, are the defining limitations of the live formulation, and no amount of cleverness inside the loop removes them.

## 6. Steering it live

**Senior rule of thumb: if conditioning lives in a continuous joint space, expose it as vector arithmetic and you get blending and morphing for free.** This is where MusicCoCa's geometry pays off as a *control surface*.

![Steering live: style is a weighted average of prompt embeddings, and classifier-free guidance amplifies the conditional direction at decode time with a guidance weight of 5.0.](/imgs/blogs/magenta-realtime-live-music-model-7.webp)

There are two composable operations, shown in the two rows of the figure.

**(1) Weighted blend.** Each prompt $c_i$ — text or audio — is embedded by MusicCoCa into $M(c_i)$. A blended style is just a normalized weighted average:

$$
c = \frac{\sum_i w_i \, M(c_i)}{\sum_i w_i}.
$$

Set $w_{\text{lo-fi}} = 0.7$ and $w_{\text{jazz}} = 0.3$ and you get a style vector 70% of the way to lo-fi and 30% toward jazz. Because the embedding space is joint, the prompts can be a mix of modalities: blend a text prompt with an audio reference clip and a second text prompt, all in one weighted sum. To *morph* live, animate the weights over time — ramp $w_{\text{lo-fi}}$ from 1.0 down to 0.0 while ramping $w_{\text{jazz}}$ up, re-quantizing the blend each chunk, and the music drifts continuously from one vibe to the other. This is the interpolation the demo page shows off, and it is nothing more exotic than walking a line in 768-d space.

**(2) Classifier-free guidance.** A weighted blend tells the model *which way* to go; CFG controls *how hard*. At each decoding step the model computes two predictions: a **conditional** one given the style $c$, and an **unconditional** one given a null style. CFG extrapolates along the line from unconditional to conditional:

$$
\text{logits}_{\text{guided}} = \text{logits}_{\text{uncond}} + w \cdot \big(\text{logits}_{\text{cond}} - \text{logits}_{\text{uncond}}\big),
$$

with guidance weight $w = 5.0$. A weight of 1.0 would be plain conditional generation; 5.0 pushes the distribution well past the conditional, sharpening adherence to the requested style at the cost of diversity. For a live instrument you *want* this aggressiveness — when the user says "more aggressive," they want to hear it now, not a subtle nudge two chunks later. The figure's bottom row spells the mechanics out: uncond is the anchor, cond is the direction, and the guided sample is pushed 5× along that direction.

### The two-second contract

Combine §5 and §6 and you get the defining UX property of the system, which is worth stating as a rule because it governs everything you build on top: **steering is sample-and-hold with a one-chunk delay.** Every control you expose — a prompt change, a weight ramp, a guidance bump — is sampled at the chunk boundary and held for two seconds. Good live interfaces for Magenta RealTime lean into this: they quantize the user's gestures to the beat, they show a "pending" state between gesture and effect, and they never promise instant response. Fighting the two-second contract produces an interface that feels broken; embracing it produces one that feels like playing an instrument with a slightly long action — a pipe organ, not a piano.

### Audio prompts versus text prompts in practice

Because both modalities land in the same space, you can steer with either — but they behave differently as controls, and knowing the difference matters when you build an interface. Text prompts are *low-bandwidth and interpretable*: "warm lo-fi, vinyl crackle" specifies a vibe at the level of words, and the model fills in everything below that with its priors. They are great for coarse, nameable styles and for letting users type their intent. Their weakness is precision — you cannot type the exact swing feel or the specific synth patch you hear in your head; language under-determines the audio.

Audio prompts are *high-bandwidth and specific*: a reference clip pins down timbre, groove, and instrumentation far more tightly than any caption, because it carries the actual acoustic fingerprint that MusicCoCa's audio tower reads. The cost is interpretability and availability — you need a clip that already sounds like what you want, and the embedding captures whatever incidental properties that clip happens to have (its room, its mix) along with the ones you intended. A reference clip recorded in a boomy room can drag the style toward "boomy" even when you only meant "this bassline."

The strong move is to combine them: anchor with an audio prompt for timbre and groove, then nudge with text for the attributes you can name. The weighted-blend control makes this trivial — give the audio reference $w = 0.6$ and a text prompt $w = 0.4$ and you get specificity plus steerability in one style vector. This is also where the two-second contract reasserts itself: switching the audio reference is just as sample-and-hold as switching the text, so a "drop in a new groove" gesture lands on the next chunk like everything else.

### Second-order: guidance and the diversity collapse

CFG at 5.0 is a loaded gun. Crank guidance too high and the model collapses onto a narrow mode of the requested style — every bar starts to sound the same, the timbre goes metallic, and the stream loses the surprise that makes it feel alive. This is the audio analogue of the over-saturated, over-sharpened look you get from too much guidance in image diffusion. The released 5.0 is a tuned compromise: high enough that steering feels immediate, low enough that the music keeps breathing. If you build on the weights and expose a guidance slider, clamp its top end — users will turn it to eleven, and eleven sounds bad. We will see exactly this in the case studies.

## Running it: the inference loop

Enough theory — here is what driving the model actually looks like. The original `magenta-rt` library runs inference on a free Colab TPU (the v2-8 tier) with JAX. Setup is a couple of lines:

```bash
pip install --no-deps git+https://github.com/magenta/magenta-realtime.git  # original library, Colab / TPU v2-8
pip install "jax[tpu]" librosa soundfile

python -c "from magenta_rt import system; system.MagentaRT()"  # pulls codec + style model + LM on first call
```

The central object is a `MagentaRT` system that owns the codec, the style model, and the language model. You embed a style once, then call `generate_chunk` in a loop, feeding each chunk's state forward. The shape of the loop is the §1 diagram in code:

```python
import numpy as np
import soundfile as sf
from magenta_rt import system

mrt = system.MagentaRT()                        # full stack: SpectroStream + MusicCoCa + ~800M block-AR LM

style = mrt.embed_style(                         # text prompt -> 12 discrete style tokens
    "upbeat disco funk, slap bass, four-on-the-floor"
)

state = None                                     # rolling 10s context, threaded across calls
chunks = []
for step in range(30):                           # 30 chunks = 60 seconds of continuous audio
    chunk, state = mrt.generate_chunk(state=state, style=style)  # this call must beat the 2s clock
    chunks.append(chunk.samples)                 # float32 stereo, 48 kHz

audio = np.concatenate(chunks, axis=0)           # the library crossfades the seams for you
sf.write("jam.wav", audio, samplerate=48_000)
print(f"generated {len(audio) / 48_000:.1f}s of 48kHz stereo")
```

Two things to notice. First, `state` is the rolling context — the coarse-level tokens of the last five chunks — threaded from call to call; that is the feedback edge in the mental-model diagram. Second, `generate_chunk` is the unit that must beat the 2-second clock; everything in §2–§4 is about making this one call fast enough.

Re-steering mid-stream is just swapping the `style` you pass in. Because of the two-second contract, the change takes effect on the next chunk:

```python
styles = {
    "lofi": mrt.embed_style("warm lo-fi hip hop, vinyl crackle, mellow"),
    "jazz": mrt.embed_style("upright bass jazz trio, brushed drums, swing"),
}

def blend(weights):
    # c = sum(w_i * M(c_i)) / sum(w_i)  -- the §6 weighted-average control.
    num = sum(w * styles[k].embedding for k, w in weights.items())
    den = sum(weights.values())
    return mrt.style_from_embedding(num / den)

state = None
out = []
for step in range(30):
    # Ramp from 100% lo-fi to 100% jazz over the 60-second jam.
    t = step / 29
    style = blend({"lofi": 1.0 - t, "jazz": t})
    chunk, state = mrt.generate_chunk(
        state=state,
        style=style,
        guidance_weight=5.0,   # strong steering; clamp this in any UI you build
        temperature=1.3,
        topk=40,
    )
    out.append(chunk.samples)
```

That loop is a 60-second style morph: it starts as lo-fi, ends as a jazz trio, and slides continuously between because the blend is linear in MusicCoCa space and re-quantized every chunk. The whole "live instrument" experience is this loop wrapped in a UI that maps knobs and pads to `weights` and `guidance_weight`.

## A short history: how we got to live models

Magenta RealTime did not appear from nowhere; it is the latest move in a five-year arc of codec-plus-language-model music generation, and seeing the lineage clarifies what is genuinely new.

The first wave was **Jukebox** (2020): a VQ-VAE codec plus autoregressive transformers that could generate raw-audio music with vocals, but at enormous cost — minutes to hours per clip, far from any notion of interactivity. It proved codec-LM music was possible and demonstrated exactly why naive scaling cannot stream.

The second wave was **MusicLM** (2023), which is the direct conceptual ancestor of Magenta RealTime's conditioning. MusicLM introduced the idea of conditioning music generation on a joint text-audio embedding (MuLan) and generating hierarchically through semantic and acoustic tokens. It made text-to-music coherent and high-quality, but it was offline: you described a clip and waited. Its embedding model, MuLan, is one of the two parents of MusicCoCa.

In parallel, **MusicGen** (2023) showed a cleaner single-stage approach — a transformer over EnCodec tokens with a clever codebook-interleaving pattern to handle RVQ depth, conditioned on text and optional melody. It is the workhorse open baseline, strong and simple, and squarely offline. **Stable Audio** brought latent diffusion to the same problem, optimizing hard for prompt adherence on fixed-length clips.

What every one of these shares is the *batch* assumption: prompt in, clip out, no interaction during generation, full-song (or fixed-length) context. Magenta RealTime's contribution is to ask what changes when you delete that assumption and demand a continuous, steerable stream. The answer turns out to be: almost everything. The codec gets a coarse live band; the context shrinks to a rolling window; the autoregressive factorization splits into temporal and depth loops; conditioning becomes a live, blendable control surface; and "quality" gains a hard deadline term. It inherits MusicLM's joint-embedding conditioning and the codec-LM template, and reorganizes the rest of the stack around the clock. That reorganization — not a new loss or a bigger model — is the paper's idea, and it defines a new category the authors name *live music models*.

## Where it sits in the landscape

**Senior rule of thumb: judge a model against the constraint it was built for, not a leaderboard it never targeted.** Magenta RealTime will not beat a 3.3B offline model on every audio-quality metric in every regime — but that is the wrong comparison. The right one is: among models you can run, which can *stream live and be steered*, and how much fidelity does that cost?

![Live music models vs offline generators: Magenta RealTime is the only open-weights model that streams live, and still posts the lowest FD_openl3 despite the smallest size.](/imgs/blogs/magenta-realtime-live-music-model-8.webp)

The surprising result from the paper's evaluation is that the live constraint does **not** cost fidelity here — it arguably helps. On the Song Describer Dataset (47-second fixed-length clips), measuring Fréchet distance in OpenL3 space (FD_openl3, lower is better):

| Model | Live? | Params | FD_openl3 ↓ | KL_passt ↓ | CLAP ↑ |
| --- | --- | --- | --- | --- | --- |
| **Magenta RT** | ✅ | ~760–800M | **72.14** | **0.47** | 0.35 |
| Stable Audio Open | ❌ | 1.1B | 96.51 | 0.55 | **0.41** |
| MusicGen-stereo-large | ❌ | 3.3B | 190.47 | 0.52 | 0.31 |

Magenta RealTime posts the **lowest FD_openl3 and the lowest KL** of the three, despite being the smallest and the only one bound by a real-time constraint. It trails Stable Audio Open on CLAP score (text–audio alignment), which makes sense — Stable Audio is a text-to-audio model optimized for exactly that prompt-adherence metric, while Magenta RT splits its capacity between fidelity, steerability, and speed. The headline is not "Magenta RT wins"; it is "**the live constraint is nearly free here**," which is a genuinely non-obvious result and the paper's strongest claim.

The matrix figure adds the dimension the table hides: **availability and control.** Magenta RT is the only model in the live quadrant with open weights. Its API sibling **Lyria RealTime** offers longer context and extended controls but is closed and cloud-only. Stable Audio Open and MusicGen are open but offline — they cannot stream or be steered mid-generation. If your application needs a continuous, controllable stream you can run yourself, the field of one is Magenta RealTime.

### Training, briefly

For completeness: the model was trained on roughly **190,000 hours** of music, predominantly instrumental stock music from multiple sources. Training examples were **12-second clips** structured exactly like inference — 10 seconds of context plus a 2-second target — so the model learns precisely the conditional distribution it will be asked to sample at run time: "given ten seconds of coarse history and a style, produce the next two seconds." That train/inference symmetry is why the model behaves predictably in the streaming loop; it was never trained on a regime it does not see live.

Optimization used **Adafactor** on an inverse-square-root schedule with 10,000 warmup steps, running for roughly **1.86 million steps** at batch size 512 on **TPU-v6e** pods (the paper reports 128 chips for the Base configuration, 256 for the Large). Adafactor rather than Adam is a memory-pragmatic choice for large models — it factorizes the second-moment estimate and avoids storing a full optimizer state per parameter — and it is a common default in the T5X ecosystem the model is built in.

The paper presents two sizes: a **Base** configuration around 220M parameters and a **Large** around 760–770M, with the open-weights Magenta RealTime release corresponding to the larger ~800M-class model. The "predominantly instrumental" detail in the data matters downstream more than any hyperparameter: the model is much stronger at instrumental textures — grooves, pads, beds, basslines — than at vocals, which is exactly what you would expect from its diet and is worth knowing before you prompt it for a pop chorus. It also shapes what "good steering" feels like: the model responds crisply to instrumentation and genre cues and weakly to anything that implies lead vocals or lyrics, because that is the distribution it learned.

## Case studies from building with it

Theory survives contact with a speaker poorly. These are nine concrete situations — some are documented limitations, some are the kind of thing you discover the first afternoon you wire the model to live audio. Each follows the same arc: the symptom, the wrong first guess, the actual cause, the fix, and the lesson.

### 1. The ten-second amnesia

**Symptom.** You start a jam with a strong, memorable four-bar hook. Ninety seconds in, the hook is gone — not varied, *gone* — and the music has wandered into an unrelated groove that shares only the broad style. Ask it to "bring back the intro" and it has no idea what you mean.

**Wrong first guess.** The model is "forgetting" because of a sampling-temperature drift, or the style embedding is decaying. People reach for the temperature knob first.

**Actual cause.** The context window is **10 seconds**, full stop. Anything that played more than five chunks ago is not in the model's input at all — it is not faded, it is absent. The model is not forgetting your hook; it never sees it. Long-term musical structure (a chorus that returns, a motif that develops over a minute) is architecturally outside the live formulation, because carrying minutes of context would blow both the attention budget and the latency budget.

**Fix.** Structure lives *above* the model, not inside it. If you need a returning chorus, you re-introduce its style (or its audio) as a prompt at the right moment — you, the human or the orchestration layer, are the long-term memory. Some builders keep a library of "section" style embeddings and schedule them.

**Lesson.** The 10-second context is the price of real time. Treat the model as a stateful *texture generator* with a 10-second horizon, and put song-level structure in your application logic. Asking the model for structure it cannot see is the single most common misuse.

### 2. The crossfade seam

**Symptom.** You bypass the library's playback helper and concatenate raw chunks yourself for a custom pipeline. The result has a faint but unmistakable **tick** every two seconds — a metronome you did not ask for, locked to the chunk rate.

**Wrong first guess.** The model is generating a transient at chunk starts, or there is a DC offset in the decoder. You start hunting for a bug in SpectroStream.

**Actual cause.** Chunks are decoded as separate 2-second segments. Their waveforms do not line up sample-for-sample at the boundary, so naive concatenation creates a discontinuity — a step in the signal — and a step is a click. At 0.5 Hz (every 2 seconds) it reads as a rhythmic tick. The codec is fine; the *seam* is the problem.

**Fix.** Equal-power crossfade across a short overlap region at every boundary — exactly what the library does for you when you let it assemble the stream. Generate with a small overlap, ramp the outgoing chunk down and the incoming chunk up over a few milliseconds, and the discontinuity disappears into a smooth blend.

**Lesson.** The chunk structure is supposed to be *inaudible*, and crossfade is what makes it so. Any time you take over stream assembly, you inherit the responsibility for the seams. This is generic real-time audio hygiene, but it bites every team that reimplements the playback path.

### 3. The two-second lag complaint

**Symptom.** A musician testing your interface says it "feels laggy" and "doesn't respond." They hit the pad for "drop the drums" and the drums keep going for what feels like forever before cutting.

**Wrong first guess.** Network latency, UI event lag, or a slow inference call. You profile the request path looking for the missing milliseconds.

**Actual cause.** There is nothing to fix in the request path — this is the **two-second contract**. The chunk currently in the buffer was generated *before* the user's gesture, so the gesture cannot affect it; it lands on the next chunk, up to ~2 seconds later. The system is working exactly as designed; the design has an irreducible one-chunk control latency.

**Fix.** Design the interface around sample-and-hold instead of pretending it is instant. Quantize gestures to the bar so the delay feels musical rather than laggy. Show a "pending" indicator between the gesture and the audible change. Where possible, let users *schedule* changes a beat ahead so the latency is hidden inside the groove.

**Lesson.** You cannot engineer away a latency that comes from the chunk size; you can only make it feel intentional. The best live interfaces for this model treat the two-second delay like the throw of a real instrument's action and design *with* it.

### 4. Classifier-free guidance turned to eleven

**Symptom.** You expose a "style strength" slider mapped to CFG weight, and a user drags it to maximum. The output turns harsh, metallic, and weirdly repetitive — every bar nearly identical, the timbre brittle.

**Wrong first guess.** The model is broken at high strength, or the codec is clipping. You suspect a numerical issue in the decoder.

**Actual cause.** **CFG diversity collapse.** Guidance extrapolates past the conditional distribution; at the tuned 5.0 it sharpens style adherence, but crank it to, say, 15 and the distribution collapses onto a narrow mode. The model stops exploring, locks onto one over-saturated rendering of the style, and the artifacts are the audio version of an over-sharpened, over-contrasted image.

**Fix.** Clamp the slider's top end. Map the user-facing "strength" control to a safe range (roughly 3–7), not the raw guidance scalar's full domain. If you want a "more intense" feeling beyond that, change the *prompt*, not the guidance weight.

**Lesson.** Guidance weight is a tuned hyperparameter, not a user knob with a wide safe range. Exposing it raw invites users to find the cliff. The released 5.0 sits where steering is immediate but the music still breathes — respect that as a center, not a floor.

### 5. The cold-start mush

**Symptom.** The very first second or two of a fresh stream sounds like undifferentiated wash — no clear groove, vague texture — before it "snaps in" to something coherent.

**Wrong first guess.** The style embedding is wrong, or the first chunk is corrupted. You re-check the prompt.

**Actual cause.** At $t = 0$ the rolling context is **empty** — there are no prior chunks to condition on. The model conditions on style plus silence/nothing, so the first chunk has only the style to go on and none of the self-reinforcing musical context that makes later chunks coherent. It takes a chunk or two for the context buffer to fill with real audio and the groove to lock.

**Fix.** Prime the context. Seed the stream with a short audio prompt in the desired style (an "intro" clip) so the first real chunk already has musical context to extend, or accept a brief warm-up and fade the stream in over the first chunk so the cold start is not front-and-center.

**Lesson.** A feedback loop needs something to feed back. The 10-second context that powers steady-state coherence is exactly what is missing at the start, so the start is the weakest moment by construction. Design the first two seconds deliberately.

### 6. Audio-prompt feedback drift

**Symptom.** Clever idea: feed the model's own recent output back in as an *audio* style prompt to "keep it consistent." Instead of stabilizing, the timbre slowly runs away — over a minute it drifts into an increasingly extreme, self-caricaturing version of itself.

**Wrong first guess.** The style embedding is too weak and needs reinforcing — so you turn the audio-prompt weight *up*, which makes it worse.

**Actual cause.** A positive feedback loop. Conditioning on your own output amplifies whatever idiosyncrasies that output already had; those amplified traits show up in the next output, which you feed back again, and the system walks itself off the manifold of natural music. It is the audio equivalent of repeatedly re-encoding a JPEG, or a microphone pointed at its own speaker.

**Fix.** Do not close the style loop with the model's own raw output. If you want consistency, anchor on a *fixed* reference (a human-chosen clip or a frozen embedding), not a moving one. If you must adapt, blend the fixed anchor heavily against any feedback term so the loop gain stays below 1.

**Lesson.** Every feedback system has a stability condition. The model is already a feedback loop through its 10-second context; adding a second, user-controlled feedback path through the style input can push loop gain over the edge. Keep your steering anchored to something that does not move.

### 7. Free-tier thermal throttle

**Symptom.** A demo runs flawlessly for five minutes on the free Colab TPU, then starts to **stutter** — occasional gaps, the stream catching its breath — even though nothing in the code changed.

**Wrong first guess.** A memory leak, or the context state growing unbounded. You start profiling allocations.

**Actual cause.** The RTF 1.6 headroom is a *typical* figure, not a guaranteed floor. Under sustained load — shared free-tier hardware, thermal limits, scheduling contention — effective throughput sags toward 1.0×. When it does, the 0.75-second-per-chunk lead buffer drains. The moment a chunk takes longer than 2.0 seconds to generate, the speaker underruns and you hear a gap.

**Fix.** Build a deeper buffer before playback starts — generate several chunks ahead so transient slowdowns are absorbed by the buffer instead of the ears. Monitor per-chunk generation time and widen the buffer adaptively when it creeps up. For anything beyond a demo, run on hardware with real headroom rather than the shared free tier.

**Lesson.** Real-time factor is a distribution, not a constant, and the tail is what glitches. The lead buffer exists precisely to convert throughput variance into latency instead of dropouts — so size it for the worst chunk you will tolerate, not the average one. RTF 1.6 is comfortable; RTF 1.05 is a stutter waiting for a hot afternoon.

### 8. The vocal prompt that never lands

**Symptom.** A user prompts "powerful female vocal belting over the chorus," and the model produces a perfectly good instrumental — drums, synths, a chord progression that wants a vocal — but no voice. They prompt harder, add "with lyrics," and still get an instrumental bed.

**Wrong first guess.** The prompt is being ignored, or the style embedding is broken for vocal terms. People escalate the guidance weight to force it.

**Actual cause.** The training data is *predominantly instrumental* stock music. The model has weak priors for lead vocals because it saw few of them, so vocal-implying prompts steer it toward the *instrumental neighborhood* of that style rather than producing a voice. Cranking guidance only amplifies the instrumental interpretation it does have — it cannot guide toward a region the training data barely populated.

**Fix.** Match the tool to its strengths. Use Magenta RealTime for the instrumental bed and layer vocals from a dedicated vocal/song-generation system if you need them. If you only have this model, prompt for what it knows — instrumentation, genre, mood, groove — and stop asking for vocals it cannot reliably make.

**Lesson.** A model's capabilities are bounded by its data distribution, not its prompt parser. No prompt engineering conjures a capability the training set did not teach. Knowing the diet ("mostly instrumental stock music") tells you the failure mode before you hit it — and saves the afternoon you would otherwise spend fighting the guidance weight.

### 9. The stereo image that collapses to mono

**Symptom.** On good monitors the output sounds fine, but a user listening on a stereo system complains it sounds "flat" or "centered" compared to a reference track — the wide stereo image they expected is narrower.

**Wrong first guess.** A bug in the WAV writer is summing channels, or the playback path is mono. You check the file's channel count and find two channels, which deepens the confusion.

**Actual cause.** This is the perceptual cost of the **4 kbps live band**. Stereo width and fine spatial imaging are exactly the kind of detail that lives in the *deep* RVQ levels — the ones generation skips and the decoder reconstructs by inference. The decoder produces a plausible stereo field, but "plausible from 16 coarse codes" is narrower and less precise than what full 64-level encoding preserves. The audio is genuinely stereo; its imaging is just lower-resolution than a 16 kbps master.

**Fix.** Accept it as the live tradeoff, or post-process. A light stereo-widening or mastering pass downstream can restore some perceived width for a polished deliverable. For interactive use, it rarely matters; for a final mixed product, treat the model's output as a stem to be mastered, not a master.

**Lesson.** Truncating to the coarse band is not free — it is *cheap*, which is different. The fidelity you trade away is the fine detail (spatial imaging, the airiest high-frequency content), precisely because RVQ's coarse-to-fine ordering puts gross structure first and that detail last. The trade is usually invisible and occasionally audible; know which of your use cases is which.

## When to reach for a live music model — and when not to

Magenta RealTime is a sharp tool with a narrow, deep edge. The whole point of this article is that its design is a set of tradeoffs in service of one constraint, so the right question is always: *is your problem actually shaped like that constraint?*

**Reach for a live music model when:**

- You need a **continuous, indefinite stream** of music, not a finished clip — interactive installations, generative radio, game soundtracks that react, live performance backing.
- **Interactivity is the product.** The value is in steering the music *as it plays* — jamming, improvising, morphing styles on a controller — and a two-second control lag is acceptable.
- You want **open weights you can run and embed**, on your own hardware or even on-device, without a cloud API in the loop. That is the one thing only Magenta RT offers in the live quadrant.
- **Instrumental texture** is the goal. The model's stock-music, mostly-instrumental training makes it excellent at grooves, pads, and beds, and steerable in real time.
- You are building **research or tooling on top of a live model** — controllability experiments, new interfaces, embedding-arithmetic studies — and need a strong, documented, open baseline.

**Skip it (or reach for an offline model) when:**

- You need **long-term song structure** — verses, choruses that return, a four-minute arc with development. The 10-second context cannot model it; you would be fighting the architecture. Use an offline model that sees the whole song.
- You want **one polished, final clip** and do not care about interactivity. An offline text-to-audio model (Stable Audio Open, MusicGen) will give you higher prompt-adherence (CLAP) and lets you cherry-pick takes.
- **Vocals or lyrics** are central. The instrumental-heavy training is a poor fit; specialized vocal or song-generation systems will serve you better.
- You need **instant, sub-100ms control response**. The two-second contract is irreducible; if your interaction model cannot tolerate a one-chunk delay, this is the wrong tool.
- You lack **real-time-capable hardware** for your deployment. The free TPU is great for demos but throttles under sustained load; production needs genuine headroom, and if you cannot provide it, an offline batch pipeline is more honest.

The deeper lesson generalizes past music. Magenta RealTime is a clean case study in **designing an entire generative stack around a latency deadline** — co-designing the codec frame rate, the quantizer depth, the context resolution, and the autoregressive factorization so that a tractable model hits a hard real-time clock. Every layer spends or saves time against the deadline, and the surprising payoff is that, measured fairly, the live constraint costs almost no fidelity. That co-design — not any single component — is the contribution worth stealing for your own real-time generative system, whether it makes music, speech, or something nobody has streamed yet.

## Further reading

- [*Live Music Models*](https://arxiv.org/abs/2508.04651) — the Lyria Team paper (arXiv:2508.04651) that introduces the live-music-model formulation and Magenta RealTime.
- [magenta/magenta-realtime](https://github.com/magenta/magenta-realtime) — the open-source repository, model weights, and Colab demo. Note the project has since shipped **MRT2**, a successor with `small` (230M) and `base` (2.4B) variants, JAX and MLX backends, and on-device Apple Silicon inference.
- [Speech tokenizers: EnCodec, SoundStream, and Mimi](/blog/machine-learning/signal-processing/speech-tokenizers-encodec-soundstream-mimi) — the RVQ codec lineage SpectroStream extends, with the residual-quantization math in depth.
- [Orpheus TTS over the SNAC codec](/blog/machine-learning/signal-processing/orpheus-tts-llm-speech-snac) — the same LLM-over-codec, temporal-versus-depth decoding pattern applied to speech.
- [Real-time TTS: chasing first-audio-byte latency](/blog/machine-learning/signal-processing/real-time-tts-first-audio-byte-latency) — the streaming-latency mindset that governs every live generative audio system.
