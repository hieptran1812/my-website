---
title: "Multimodal and speech at the edge: on-device VLMs and streaming ASR"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "Text LLMs on a phone are basically solved — the frontier is a device that can see, hear, and respond in real time, so this post derives the real-time factor, the vision-token prefill blowup, and the latency budget, then shows runnable whisper.cpp and small-VLM flows with measured numbers."
tags:
  [
    "edge-ai",
    "model-optimization",
    "multimodal",
    "speech-recognition",
    "vision-language-models",
    "whisper",
    "streaming-asr",
    "on-device-llm",
    "inference",
    "efficient-ml",
  ]
category: "machine-learning"
subcategory: "Edge AI"
author: "Hiep Tran"
featured: true
readTime: 56
image: "/imgs/blogs/multimodal-and-speech-at-the-edge-1.png"
---

Hold up your phone, point the camera at the back of a router, and ask out loud: "which of these ports is the WAN port?" A useful answer comes back as speech a beat later — and the whole exchange never touched a server. The microphone fed an always-on keyword spotter, the spotter woke a streaming speech recognizer, the recognizer's transcript and the camera frame both flowed into a small language model, and the model talked back. Camera in, voice in, voice out, all on a battery-powered slab of silicon in your hand.

That loop is where on-device AI is actually going, and it is the natural close to the on-device-LLM track of this series. The text-only part of the problem is, frankly, solved enough. We have spent whole posts on it: you can [run an LLM locally with llama.cpp and GGUF](/blog/machine-learning/edge-ai/running-llms-locally-llama-cpp-and-gguf), you can [make on-device LLMs genuinely fast](/blog/machine-learning/edge-ai/making-on-device-llms-fast) with KV-cache tricks and speculative decoding, and you can [design a small language model from scratch](/blog/machine-learning/edge-ai/small-language-models-by-design) that beats a quantized big one at the same memory budget. A 3B model answering text questions on a flagship phone at twenty tokens a second is no longer a research demo. It is a shipped feature.

The frontier moved. The interesting, unsolved, *paged-at-2am* part of edge AI now is multimodal: a device that **sees** (vision-language models), **hears** (speech recognition), and **responds** — and does it in real time, where "real time" is a hard physical constraint, not a vibe. This post is about that frontier. We will derive the **real-time factor** that decides whether speech recognition can even keep up with a talker, we will derive how a single image **blows up the prefill** of a language model by hundreds of tokens, and we will pin down the **latency budget** that separates "feels live" from "feels broken." Then we make it concrete: runnable `whisper.cpp` commands with measured real-time factors, a quantized small vision-language model answering a question about an image with measured time-to-first-token, and honest before→after tables on named hardware.

Figure 1 is the whole loop in one picture — the wake-word gate up front, the speech and vision branches fanning in, the shared small LLM, the speech-and-text response — and it is the map for everything below. By the end you will be able to look at a multimodal feature spec and say, with numbers, whether it can run on the device or whether part of it has to be offloaded, and exactly which part.

![The on-device multimodal loop where a wake-word spotter gates a streaming speech branch and a vision branch that both feed one shared small language model producing speech and text output](/imgs/blogs/multimodal-and-speech-at-the-edge-1.png)

Throughout, we lean on the same four-lever frame the rest of the series uses — quantization, pruning, distillation, efficient architecture, sitting on compilers and runtimes, read off the [accuracy-efficiency Pareto frontier](/blog/machine-learning/edge-ai/a-taxonomy-of-model-compression). Multimodal does not need new levers. It needs you to apply the same levers to *two more model families* — an audio encoder-decoder and a vision encoder — and to respect two new hard constraints: the clock (speech is streaming) and the token budget (an image is expensive to read).

## Why multimodal is a different animal than text

A text LLM on device has one tensor coming in — a sequence of token IDs — and one constraint that dominates: memory bandwidth during decode. We covered that to death. Multimodal breaks both assumptions.

First, the **inputs are continuous and time-bound**. Audio arrives as a stream of samples at 16 kHz; you do not get to wait for the user to finish a paragraph before you start, because by then the conversation has moved on. A frame from a camera is 150,000 pixels that have to be compressed into something a language model can read. Neither of these is a tidy list of integers. There is a heavy *encoder* in front of the language model whose job is to turn the raw modality into tokens, and that encoder has its own latency, its own memory, and its own quantization story.

Second, the **constraints have a clock attached**. For text, slow is annoying. For speech, slow is *fatal* in a precise sense: if your recognizer takes 1.2 seconds of compute to process 1.0 second of audio, you fall behind by 0.2 seconds every second, forever. The backlog grows without bound and the transcript drifts further and further behind the speaker until it is useless. There is a hard threshold — the real-time factor must stay below one — and crossing it is not a gradual degradation, it is a cliff. We derive that cliff in a moment.

Third, **the token economics flip**. In text decode, you pay for one new token at a time and the cost is memory-bandwidth-bound (you stream the whole model to produce one token). In a vision-language model, a single image lands as *hundreds to thousands of tokens all at once*, and they all go through prefill — the compute-bound, attention-quadratic phase — before the model says a single word. The image does not cost you one token. It can cost you more tokens than the entire rest of the conversation. That is the central tension of edge VLMs and we will count it exactly.

There is a fourth difference that does not get enough airtime, and it is worth naming before we go deep: **the encoder is a model you also have to optimize**, and it does not obey the same bottleneck rules as the LLM you are used to tuning. The audio encoder and the vision encoder are both *encoders* — non-autoregressive, single-forward-pass, compute-bound. That is the opposite of the LLM decode loop, which is one-token-at-a-time and memory-bandwidth-bound. So the profiling instinct you built up tuning a text LLM — "decode is bandwidth-bound, so quantize the weights and stream them faster" — is exactly wrong for the encoder. The encoder is a dense matmul wall; it is bound by raw arithmetic throughput, and the lever that helps it is the *efficient-architecture* lever (fewer FLOPs, smaller patches' worth of work) and operator fusion in the runtime, not weight-streaming. You end up tracking two different bottleneck regimes at once — encoder is compute-bound, decoder is memory-bound — and a profiler that does not separate the two will lie to you. We will keep them separate throughout.

There is also a subtler point about *where the work lands in time*. For a text turn, all the compute is in the decode loop, spread evenly across the answer. For a multimodal turn, the work is front-loaded: the audio encoder runs, the vision encoder runs, the LLM prefills the whole image-plus-text prompt, and only *then* does the first answer token appear. Everything the user perceives as "the lag before it starts talking" is concentrated in that front-loaded burst. That is why time-to-first-token, not steady-state throughput, is the metric that governs whether a multimodal feature feels alive. A model that decodes at 40 tokens a second but takes two seconds to start is worse, for a camera-Q&A feature, than one that decodes at 15 tokens a second but starts in 400 ms. The series has hammered tokens-per-second for text; multimodal forces TTFT to the front.

So the plan: speech first (the clock constraint), then vision (the token constraint), then the unified science of the latency budget and multimodal memory, then runnable code and measured results, then a stress test of what breaks under load, then a decision tree for when on-device multimodal is ready and when to offload.

## Speech at the edge: Whisper.cpp and the encoder-decoder

Speech recognition — automatic speech recognition, ASR — is the modality that has come furthest on device, largely because of one model family and one port of it. The model is **Whisper**, the encoder-decoder transformer trained by Radford et al. (2022) on 680,000 hours of weakly-supervised multilingual audio. The port is **whisper.cpp**, Georgi Gerganov's ggml/GGUF reimplementation — the same author and the same numerical machinery as llama.cpp.

### The architecture, and why the ports compose with everything we know

Whisper is a textbook sequence-to-sequence transformer. Audio comes in, gets turned into a **log-mel spectrogram** — 80 mel-frequency bins computed over short overlapping windows, giving a 2D image of "how much energy at each pitch over time." That spectrogram is fed to a **ViT-style audio encoder** (convolutional stem, then transformer blocks) that produces a sequence of audio embeddings. Then a **text decoder** autoregressively generates tokens — the transcript — attending to the encoder's output through cross-attention. It is an image model bolted to a text decoder, which is exactly why every trick from this series transfers.

![Whisper.cpp as a layered stack from log-mel front end through audio encoder and text decoder, with GGUF k-quantization shrinking the weights to a phone-sized footprint](/imgs/blogs/multimodal-and-speech-at-the-edge-2.png)

The encoder runs **once** per 30-second audio window — it is not autoregressive, so it is a single forward pass, compute-bound, that you pay up front. The decoder runs autoregressively, one token per transcript word-piece, and like any decoder it is memory-bound when the model is large and compute-bound when it is small. For the tiny and base sizes that actually run on a phone, the decoder is small enough that the encoder pass often dominates wall-clock time for short utterances.

It is worth pinning down *why* the encoder dominates for short clips, because it shapes every streaming decision below. The log-mel front end always produces a fixed-shape input — Whisper pads or trims every clip to exactly 30 seconds, so the encoder always does the same amount of work regardless of whether the user said "yes" or recited a paragraph. The encoder processes a $80 \times 3000$ mel matrix (80 bins, 3000 frames at the 10 ms hop) through its convolutional stem and transformer blocks every single time. The decoder, by contrast, does work proportional to the *number of words actually spoken*. For a one-second "set a timer" utterance, the decoder emits maybe five tokens while the encoder still chewed through the full 30-second-shaped input. So for short commands the encoder is essentially the entire cost, and for long dictation the decoder catches up and can overtake it. This is the first thing the streaming design has to fight: re-running that fixed 30-second encoder pass on every hop is the dominant waste, and it is why a growing-buffer approach balloons RTF — you keep paying the full encoder bill over and over.

Here is the second consequence, and it is the one that surprises people who came from text LLMs. Because the encoder is a fixed-shape compute-bound matmul, **quantizing the encoder buys you almost nothing in latency on a CPU that is already arithmetic-bound** — int8 matmul is faster than fp32 matmul only if the kernel and the hardware actually exploit the narrower type with vectorized integer dot-products (SDOT/UDOT on ARM, VNNI on x86). If the runtime falls back to a dequantize-then-fp32-matmul path, you pay the dequant cost and get fp32 speed, i.e. a net loss. So on the encoder side, quantization is mostly a *memory* win (smaller file, smaller working set) and only a speed win when the integer kernels are real. On the decoder side, which is bandwidth-bound for the larger sizes, quantization is a straight speed win because you are streaming fewer bytes per token. Two halves of the same model, two different reasons to quantize. Keep that distinction; it is the whole reason whisper.cpp's k-quants help tiny/base mostly on RAM and large mostly on decode throughput.

Because it is ggml underneath, the *same GGUF quantization that shrinks an LLM shrinks Whisper.* The model ships in five sizes — tiny (39M params), base (74M), small (244M), medium (769M), large (1.55B) — and each can be quantized with the k-quant schemes from the LLM world. A quick footprint table, in float16 versus int5 GGUF (`Q5_0`-class):

| Whisper size | Params | f16 file | Q5 GGUF file | Approx peak RAM (Q5) |
| --- | --- | --- | --- | --- |
| tiny | 39 M | ~75 MB | ~31 MB | ~75 MB |
| base | 74 M | ~142 MB | ~57 MB | ~140 MB |
| small | 244 M | ~466 MB | ~182 MB | ~340 MB |
| medium | 769 M | ~1.5 GB | ~539 MB | ~900 MB |
| large-v3 | 1.55 B | ~3.1 GB | ~1.1 GB | ~1.6 GB |

(File sizes are the GGUF weights; peak RAM adds the activation working set and the KV-cache for the decoder, so it is a bit larger than the weights. Treat these as representative order-of-magnitude figures, consistent with whisper.cpp's published model cards — measure on your target.)

The thing to internalize: a phone can comfortably hold **tiny, base, and small**. Medium is a stretch and large is a no for most handsets. So the practical edge-ASR conversation is almost entirely about the bottom three sizes, and the question becomes: how accurate is small-but-fast, and can it keep up with a live talker? That second question is the real one, and it has a name.

### The real-time factor: deriving the cliff

The single most important number in edge speech is the **real-time factor**, RTF. It is dead simple to define and its consequences are profound. Let $T_{\text{compute}}$ be the wall-clock time your device spends recognizing a clip, and $T_{\text{audio}}$ be the duration of the clip itself. Then

$$\text{RTF} = \frac{T_{\text{compute}}}{T_{\text{audio}}}.$$

If you transcribe a 10-second clip in 3 seconds, your RTF is 0.3. If it takes you 14 seconds, your RTF is 1.4. The interpretation:

- $\text{RTF} < 1$: you process audio *faster than it arrives*. You can run live, with headroom equal to $1 - \text{RTF}$.
- $\text{RTF} = 1$: you exactly keep pace, with zero margin. Any hiccup — a thermal throttle, a background app — pushes you over.
- $\text{RTF} > 1$: you fall behind, permanently.

Here is why $\text{RTF} \geq 1$ is a true cliff and not a gradient. Suppose audio arrives continuously and you process it in chunks. In each wall-clock second, $\frac{1}{\text{RTF}}$ seconds of audio gets processed but 1 second of new audio arrives. The backlog $B(t)$ — unprocessed audio waiting in the buffer — grows at rate

$$\frac{dB}{dt} = 1 - \frac{1}{\text{RTF}} \quad\text{seconds of audio per second of wall-clock.}$$

When $\text{RTF} < 1$, that rate is negative: the buffer drains, latency stabilizes at a small value. When $\text{RTF} = 1$, the rate is zero: the buffer holds whatever it had, latency is constant but never shrinks. When $\text{RTF} > 1$, the rate is positive and *constant*, so the backlog — and therefore the lag between speaker and transcript — grows **linearly without bound**. After a minute of talking at RTF 1.4, you are roughly $60 \times 0.286 \approx 17$ seconds behind. That is the cliff. There is no "a little slow." You either keep up or you diverge.

![A matrix contrasting real-time factor values of 0.3, 0.9, and 1.4 showing which can stream live and how the latency backlog behaves below versus above one](/imgs/blogs/multimodal-and-speech-at-the-edge-4.png)

So the engineering target for live ASR on device is not "fast." It is **RTF comfortably below one** — I want at least 30% headroom, RTF around 0.7 or lower, because the device will thermal-throttle, other apps will steal cycles, and a transient spike at RTF near 1 means a stutter the user feels. Headroom is the budget for the real world.

There is one more layer to the cliff that the simple backlog argument hides, and it matters for how you size the window. The RTF you measure on a whole 10-second file is an *average*. Streaming does not process the file; it processes a window every hop, and the window RTF is what has to stay below one. Worse, the window RTF is not constant — the encoder cost is roughly fixed per window (the 30-second-shaped pass), but the decoder cost varies with how many words landed in that window. A window that catches a fast burst of speech makes the decoder do more work, spiking the instantaneous RTF above the average. So the real safety condition is not "average RTF < 1" but "**worst-case window RTF < 1**," and the gap between them is your true headroom. If the average is 0.6 but a dense window peaks at 0.95, you have almost no margin, and the first thermal throttle pushes that window over one. This is why I budget against the worst window I can provoke, not the comfortable average — and it is why a "fast on the benchmark clip" model can still stutter on real, variable-rate human speech.

Make this quantitative. Suppose the per-window compute splits into a fixed encoder term $T_{\text{enc}}$ and a per-word decoder term $t_{\text{dec}} \cdot w$, where $w$ is the words in the window and $W$ is the window length in seconds. Then the window RTF is

$$\text{RTF}_{\text{window}} = \frac{T_{\text{enc}} + t_{\text{dec}} \cdot w}{W}.$$

The encoder term sets the *floor* RTF (what you pay even in silence), and the decoder term sets the *slope* (how fast RTF climbs with speech density). A tiny model has a small $T_{\text{enc}}$ and a small $t_{\text{dec}}$, so both the floor and the slope are low — that is why tiny survives dense speech bursts that drown a small model. When you choose a window length $W$, you are trading: a longer $W$ amortizes the fixed encoder cost over more audio (lower floor RTF) but increases the worst-case word count $w$ in a window and delays the first partial. This little equation is the whole tuning surface behind `--length` and `--step`.

### Why offline Whisper is not streaming, and how to fix it

Here is the trap. Whisper, as published, is an **offline** model. It is trained on 30-second windows and it is happiest when it has the *whole* utterance, because the encoder attends bidirectionally across the entire 30 seconds — every audio frame sees every other frame, past and future. That bidirectional context is a big chunk of why Whisper is so accurate. It is also why naive Whisper is not streaming: to get its best transcript, it wants to wait until the speaker is done.

If you just call `whisper.cpp` on a growing buffer every second, you re-run the encoder over the entire buffer each time, which is both wasteful (RTF balloons as the buffer grows) and produces flickering transcripts (the model revises earlier words as new audio arrives). That is the naive approach, and it is bad.

The real fix is **chunked streaming**: process the audio in overlapping windows with a fixed hop, commit the stable middle of each window, and carry a little overlap for context.

![A before-after figure contrasting offline batch Whisper that waits for the full thirty-second window against chunked streaming that emits partial text every second with slightly higher word error rate](/imgs/blogs/multimodal-and-speech-at-the-edge-3.png)

The trade is exactly what you would expect from the bidirectional-context argument: streaming gives up some *right context* (the model has not heard the rest of the sentence yet when it commits a word), so streaming word error rate is typically 2-4 absolute points worse than the same model run offline. In exchange you get **partial transcripts in hundreds of milliseconds** instead of after the whole utterance. The whisper.cpp repo ships a `stream` example that does exactly this with a sliding window over a microphone capture, and there is active work (e.g. the `whisper-streaming` family of wrappers) that adds smarter commit policies — only finalize a word once two successive windows agree on it.

A clean way to think about the streaming latency: with a window length $W$ and a hop $H$, you emit a fresh partial roughly every $H$ seconds, and each partial reflects audio up to "now minus a little." If you process a window in $H \cdot \text{RTF}_{\text{window}}$ seconds and you want to stay live, you need the per-window RTF below one, and you want the hop small enough that the user perceives the captions as keeping pace. Typical edge settings: $W \approx 5$ s window, $H \approx 1$ s hop, with the model run on the most recent $W$ seconds each hop.

The commit policy is where the engineering subtlety lives. Every hop re-decodes the last $W$ seconds, which means the most recent words are *unstable* — they will likely be revised when the next hop hears more context. The fix the `whisper-streaming` family uses is **local agreement**: a word is only finalized (printed in stable black, never to be revised) once two consecutive hops produce the same word at the same position. Words past that agreement point stay in a tentative "gray" state and are allowed to flicker. The user sees stable text trailing slightly behind a live, twitchy edge — which is exactly how good live captions look. The cost of agreement is one extra hop of latency on every committed word (you wait for the confirmation), so the committed-text lag is roughly $H$ plus the window's processing time. With $H = 1$ s and a window RTF of 0.5, committed captions trail the speaker by something like 1.5 seconds — inside the "laggy but usable" band for captioning, and the price you pay for not flickering.

#### Worked example: tuning the streaming window on a mid-range phone

Put numbers on it. Say you are targeting a mid-range Snapdragon phone, CPU-only via whisper.cpp, with `base.en` Q5 measured at an *average* RTF of 0.45 on clean speech but a worst-case dense-window RTF of 0.8. You want live captions. Walk the window choices:

- **$W = 3$ s, $H = 0.5$ s.** Snappy — a fresh partial twice a second, committed text trailing ~1 s. But the fixed encoder cost is amortized over only 3 seconds, so the floor RTF is higher, and the worst-case window now peaks near 0.95 under a fast talker. Almost no thermal headroom. Verdict: feels great for 30 seconds, then the phone warms up and it starts stuttering. Too aggressive.
- **$W = 5$ s, $H = 1$ s.** The default. Encoder cost amortized over 5 seconds drops the floor; worst-case window ~0.8, leaving ~20% headroom. Committed text trails ~1.5 s. Verdict: the safe live-captioning setting on this chip. Survives a thermal throttle that lifts every RTF by ~15% (0.8 → 0.92, still under one).
- **$W = 8$ s, $H = 2$ s.** Maximum efficiency — fewest encoder re-runs, lowest average RTF (~0.35), most battery headroom. But partials arrive only every 2 seconds and committed text trails ~3 s, which crosses out of the responsive band for live conversation. Verdict: good for background "log everything" transcription, too laggy for face-to-face captions.

The decision: ship $W = 5$, $H = 1$, with local-agreement commit. It is the only setting whose *worst-case* window survives thermal throttle while keeping committed lag inside the usable band. Notice the reasoning never touched WER — for live captioning the latency budget picks the window, and you accept whatever WER that window happens to give. Accuracy is the constraint you optimize *after* the clock constraint is satisfied, never before.

### The always-on front end: wake-word and keyword spotting

Running Whisper continuously is a battery and thermal disaster — even tiny at RTF 0.12 means the SoC's vector units are busy 12% of the time, all day, which drains the battery and heats the phone. No shipped product does that. Instead, a **tiny always-on keyword spotter (KWS)** sits in front of the heavy stack and only wakes it when it hears a trigger.

A wake-word model — "Hey Siri," "OK Google," "Alexa" — is a tiny neural net, often well under 50 KB, frequently running on a dedicated low-power DSP or NPU island that stays on while the main CPU sleeps. It is precisely the kind of model we cover in the [TinyML on microcontrollers](/blog/machine-learning/edge-ai/tinyml-on-microcontrollers) post: a small convolutional or recurrent net over mel features, int8-quantized, with a tensor arena measured in tens of kilobytes. Its job is not to transcribe; its job is to answer one binary question — "did the trigger phrase happen?" — at vanishingly small power.

This gating is the architectural key to making always-on multimodal feasible. The expensive ASR and VLM stack runs for seconds at a time, a few times an hour, gated by a model that costs microwatts. That is why figure 1 puts the wake-word spotter up front as the gate: it converts an "always running a 75 MB model" problem into a "running a 50 KB model always, and a 75 MB model occasionally" problem. The power math only closes because of the gate.

Let me actually run the power math, because "the gate makes it feasible" is easy to assert and worth proving. Suppose the active ASR stack draws an extra $P_{\text{asr}} \approx 1.5$ W above idle when it runs (a fair figure for a phone CPU/NPU busy at RTF 0.12), and the wake-word spotter on its low-power DSP island draws $P_{\text{kws}} \approx 2$ mW continuously. A typical phone battery holds on the order of $15{,}000$ mWh. Now compare two designs over an hour, with the user actually invoking the assistant for, say, 30 seconds of cumulative real interaction:

$$E_{\text{always-on ASR}} = P_{\text{asr}} \cdot 3600\text{ s} = 1.5\text{ W} \cdot 1\text{ h} = 1500 \text{ mWh}.$$

That is **10% of the battery per hour** just listening — the device is dead in a workday and warm in your pocket the whole time. Now the gated design:

$$E_{\text{gated}} = \underbrace{P_{\text{kws}} \cdot 1\text{ h}}_{\text{always listening}} + \underbrace{P_{\text{asr}} \cdot \tfrac{30}{3600}\text{ h}}_{\text{30 s of real use}} = 2\text{ mWh} + 12.5\text{ mWh} \approx 14.5 \text{ mWh}.$$

That is about **0.1% of the battery per hour** — a hundredfold improvement, and it is dominated by the brief bursts of real use, not the listening. The wake-word island's 2 mW is rounding error. *This* is what "the power math only closes with the gate" means quantitatively: the gate is not a 20% optimization, it is the difference between a feature that ships and one that drains the battery before lunch. Note also the thermal corollary — the always-on design keeps the main SoC busy 12% of *every* second forever, so it never cools; the gated design lets the SoC sleep between bursts, which is what keeps the phone cool enough to sustain RTF below one when it does wake up. Power and thermal are the same coin.

#### Worked example: Whisper tiny vs base vs small on an M2 laptop

Let me make the speech trade-offs concrete with representative numbers on a named target — an Apple M2 MacBook Air, CPU-only via whisper.cpp (no Core ML acceleration, to keep it portable), batch=1, transcribing a clean English clip. These are the kind of figures the whisper.cpp benchmarks and community reports land on; treat them as representative and measure your own.

- **Whisper tiny, Q5 GGUF.** ~31 MB on disk, ~75 MB peak RAM. RTF around **0.12** — it chews through a 10-second clip in ~1.2 s. Word error rate on clean English: roughly **11-13%**. Verdict: blazing fast, huge real-time headroom, but the accuracy is "gist, not transcript." Great for command-and-control ("set a timer for ten minutes"), shaky for dictation.
- **Whisper base, Q5 GGUF.** ~57 MB on disk, ~140 MB peak RAM. RTF around **0.30** — still 3x faster than real time. WER roughly **8-9%**. Verdict: the sweet spot for most on-device live captioning. Plenty of headroom and usable accuracy.
- **Whisper small, Q5 GGUF.** ~182 MB on disk, ~340 MB peak RAM. RTF around **0.6-0.8** on the same CPU. WER roughly **5-6%**. Verdict: noticeably better transcripts, but the RTF headroom is thin on a phone-class chip; fine on a laptop, risky for sustained live streaming on a handset under thermal load.

The lesson is the classic accuracy-latency Pareto front, now with the RTF cliff drawn on it: **base is the default**, you reach down to tiny when you need the headroom (or you are on a weaker chip), and you reach up to small only when accuracy matters more than guaranteed live performance and you have RTF budget to spare. On a flagship phone with NPU/Core ML acceleration, every one of these RTFs drops by roughly 2-4x, which moves small into live-streaming range — but only on the good phones, which is exactly the kind of fragmentation that makes you target base as the safe floor.

### Running it: whisper.cpp commands and measuring RTF

Enough theory. Here is the actual flow — build whisper.cpp, grab a quantized model, transcribe a file, stream from the mic, and measure the real-time factor yourself.

```bash
# Build whisper.cpp (CPU; add -DGGML_METAL=ON on macOS for GPU/ANE-ish speedups)
git clone https://github.com/ggerganov/whisper.cpp
cd whisper.cpp
cmake -B build -DGGML_NATIVE=ON
cmake --build build -j --config Release

# Download a model and quantize it to 5-bit GGUF (k-quant style)
bash ./models/download-ggml-model.sh base.en
./build/bin/quantize models/ggml-base.en.bin models/ggml-base.en-q5_0.bin q5_0

# Offline transcribe a 16 kHz mono WAV, print timestamps
./build/bin/whisper-cli \
  -m models/ggml-base.en-q5_0.bin \
  -f samples/jfk.wav \
  --output-txt --print-progress
```

To *measure* the real-time factor honestly, do not eyeball it — whisper.cpp prints timing, but the clean way is to divide compute time by audio duration, after a warm-up run so you are not measuring first-call overhead and page faults:

```bash
# Audio duration in seconds (ffprobe), and a timed transcription
AUDIO=samples/gb1.wav
DUR=$(ffprobe -v error -show_entries format=duration \
      -of default=noprint_wrappers=1:nokey=1 "$AUDIO")

# Warm-up (discard), then the measured run
./build/bin/whisper-cli -m models/ggml-base.en-q5_0.bin -f "$AUDIO" >/dev/null 2>&1
START=$(python3 -c "import time;print(time.time())")
./build/bin/whisper-cli -m models/ggml-base.en-q5_0.bin -f "$AUDIO" >/dev/null 2>&1
END=$(python3 -c "import time;print(time.time())")

python3 - "$START" "$END" "$DUR" <<'PY'
import sys
start, end, dur = map(float, sys.argv[1:4])
compute = end - start
print(f"compute={compute:.2f}s  audio={dur:.2f}s  RTF={compute/dur:.3f}")
PY
```

For live streaming from the microphone, whisper.cpp ships the `stream` example, which does the chunked sliding-window thing for you:

```bash
# Live mic captioning: 5s context window, commit step ~1s, base.en quantized
./build/bin/whisper-stream \
  -m models/ggml-base.en-q5_0.bin \
  --step 1000 --length 5000 --keep 200 \
  --threads 4
```

`--length` is the analysis window in milliseconds, `--step` is the hop (how often it re-decodes and emits a partial), and `--keep` carries a little audio across steps for context continuity. Tighten `--step` for snappier captions at the cost of more compute (higher effective RTF); loosen it to save battery. This is the streaming-vs-accuracy knob from the before-after figure, exposed as two CLI flags.

If you want to *measure the per-window RTF* — the worst-case number that actually decides whether streaming survives, not the file-average — instrument the window loop directly. Here is a small Python harness that uses the `pywhispercpp` bindings (or you can shell out to `whisper-cli` per window) to time each hop independently and report the distribution, which is the honest way to see the worst-case spike:

```python
import time, numpy as np
from collections import deque
# pip install pywhispercpp  (Python bindings over whisper.cpp)
from pywhispercpp.model import Model

SR = 16000                 # Whisper expects 16 kHz mono
WINDOW_S, HOP_S = 5.0, 1.0 # the W and H we tuned above
model = Model("base.en-q5_0", n_threads=4)

def stream_rtf(audio: np.ndarray):
    """Replay a clip as if it were live; report per-hop window RTF."""
    win, hop = int(WINDOW_S * SR), int(HOP_S * SR)
    buf = deque(maxlen=win)
    rtfs, pos = [], 0
    while pos < len(audio):
        chunk = audio[pos:pos + hop]
        buf.extend(chunk)
        pos += hop
        window = np.array(buf, dtype=np.float32)
        t0 = time.perf_counter()
        model.transcribe(window)          # decode the current window
        compute = time.perf_counter() - t0
        # this window covered HOP_S seconds of *new* audio
        rtfs.append(compute / HOP_S)
    rtfs = np.array(rtfs)
    print(f"hops={len(rtfs)}  mean RTF={rtfs.mean():.3f}  "
          f"p95={np.percentile(rtfs,95):.3f}  max={rtfs.max():.3f}")
    return rtfs

# Warm up (page in weights, JIT the kernels), THEN measure.
audio = np.fromfile("speech_f32_16k.raw", dtype=np.float32)
stream_rtf(audio)          # discard — warm-up
stream_rtf(audio)          # the real measurement
```

The number you care about is `max`, not `mean`. A model with `mean RTF=0.45` and `max=0.95` is living dangerously; one with `mean=0.45` and `max=0.6` has real headroom. Reporting only the mean is the single most common way edge-ASR benchmarks lie — they hide the dense-window spike that is exactly what stutters in production. Run this on your *weakest* target device, warm, ideally after a few minutes of sustained load so the chip has throttled, and read the `max` column against the cliff at 1.0.

## Vision-language at the edge: the architecture and the token blowup

Now the harder modality. A **vision-language model (VLM)** takes an image *and* text and produces text — "what's in this photo," "read the sign," "is the stove on." The canonical open architecture, from Liu et al.'s LLaVA (2023), is beautifully simple and is what almost every edge VLM follows.

### Three parts: encoder, projector, LLM

A VLM is three components in a row:

1. A **vision encoder** — typically a Vision Transformer (a CLIP-pretrained ViT-L/14 in classic LLaVA; a smaller ViT or a convolutional encoder in edge variants). It chops the image into fixed patches and produces one embedding per patch. This is exactly the family of models we optimize in [efficient attention and vision transformers for the edge](/blog/machine-learning/edge-ai/efficient-attention-and-vision-transformers-for-edge) — the same patch-embedding, the same quadratic-attention concerns.
2. A **projector** — a tiny MLP (sometimes a single linear layer in the original LLaVA, later a 2-layer MLP) that maps each patch embedding from the vision encoder's dimension into the language model's token-embedding dimension. This is the cheapest part by far: a couple of matrix multiplies. Its job is just to make image embeddings *look like* word embeddings to the LLM.
3. A **small LLM** — the language model that consumes the projected image tokens followed by the text prompt tokens and generates the answer. On the edge this is a 1-3B model, and it is exactly the kind of [small language model by design](/blog/machine-learning/edge-ai/small-language-models-by-design) we covered, quantized to 4 bits with the [llama.cpp / GGUF stack](/blog/machine-learning/edge-ai/running-llms-locally-llama-cpp-and-gguf).

![A layered stack showing a VLM as an input image through a ViT vision encoder and a small projector producing hundreds of image tokens that feed a small quantized language model](/imgs/blogs/multimodal-and-speech-at-the-edge-5.png)

The elegance is that you reuse a frozen vision encoder and a (mostly frozen) small LLM, and you mainly *train the projector* plus a light fine-tune. The trouble is hiding in step 1's output, and it is the defining problem of edge VLMs.

### Counting vision tokens: the prefill blowup, derived

Here is the part everyone underestimates. A ViT does not produce one embedding for the image. It produces **one embedding per patch**, and there are a lot of patches.

Take a square input image of side $S$ pixels and a patch size of $P$ pixels. The number of patches — and therefore the number of vision tokens the LLM has to ingest — is

$$N_{\text{vis}} = \left(\frac{S}{P}\right)^2.$$

Plug in the LLaVA-1.5 defaults: $S = 336$, $P = 14$. Then $S/P = 24$, and

$$N_{\text{vis}} = 24^2 = 576 \text{ tokens.}$$

**One image is 576 tokens.** Your text prompt — "what is in this image?" — is maybe 8 tokens. The image outweighs the prompt by 70 to 1. And it gets worse with higher resolution: bump the input to $672 \times 672$ (which LLaVA-1.5's "AnyRes" tiling effectively does to read fine detail) and you are at $48^2 = 2304$ tokens for a single image. High-resolution document understanding can push past 4,000 vision tokens per image.

Now connect that to the prefill cost we derived in [making on-device LLMs fast](/blog/machine-learning/edge-ai/making-on-device-llms-fast). Prefill is the phase where the LLM ingests the whole prompt before generating the first output token, and its self-attention cost is *quadratic in sequence length*. For a prompt of length $L$ tokens through a transformer with $d$ model dimension and $L_{\text{layers}}$ layers, the attention compute scales like

$$C_{\text{prefill}} \sim L_{\text{layers}} \cdot \left( \underbrace{L \cdot d^2}_{\text{projections}} + \underbrace{L^2 \cdot d}_{\text{attention}} \right).$$

The first term (the QKV and output projections, the MLP) is linear in $L$; the second (the attention score matrix) is quadratic. When $L$ jumps from 8 (text only) to 584 (text + one image), the linear term grows 73x and the quadratic term grows over 5,000x. Time-to-first-token (TTFT) is dominated by prefill, so **adding one image can multiply your TTFT by an order of magnitude or more.** This is why a text chatbot on your phone replies instantly but the same phone takes a noticeable beat to start answering a question about a photo.

A fair objection: on a small model and a short-ish prompt, isn't prefill still dominated by the *linear* term, since $L^2 \cdot d$ only overtakes $L \cdot d^2$ once $L > d$? Yes — and that is exactly the regime edge VLMs live in, which is why the picture is subtle. For a 2.7B model, $d$ is roughly 2560, so at $L = 584$ the attention term ($584^2 \cdot 2560$) is still smaller than the projection term ($584 \cdot 2560^2$); prefill is mostly the *linear* matmuls. The practical takeaway flips the usual "quadratic attention is the enemy" story: at edge VLM scales, **TTFT is dominated by the linear-in-$L$ matmuls, and the number that drives it is simply the token count $L$.** Halve the tokens and you roughly halve the prefill matmul work, full stop. The quadratic term is a tax that grows faster, so it bites harder at high resolution (4,000-token documents push you into the regime where $L > d$ and attention dominates), but for ordinary single-image VLM prompts the lesson is the clean one: **cut the token count, cut the TTFT, almost linearly.** That is why token reduction is so effective — it attacks the dominant linear term head-on, and as a bonus knocks the quadratic term down by the square.

It is also worth being precise about the encoder's own cost, because people forget the ViT does quadratic-in-patches attention too. The vision encoder runs self-attention over its $N_{\text{vis}}$ patches, so its attention cost scales as $N_{\text{vis}}^2 \cdot d_{\text{vit}}$ per layer. At 576 patches this is real but bounded (the ViT is small, a few hundred million params, and it runs once); at 2304 patches for a high-res tile, the encoder's own attention becomes a meaningful slice of TTFT *before the LLM even starts*. So high resolution hurts you twice — once in the encoder's quadratic patch-attention, once in the LLM's prefill over the resulting tokens. Both fixes are the same: produce fewer tokens.

![A before-after figure showing a naive VLM with 576 vision tokens producing a 1.8 second TTFT against a pooled VLM with 144 tokens cutting TTFT to half a second and shrinking the KV-cache](/imgs/blogs/multimodal-and-speech-at-the-edge-6.png)

And it is not just TTFT. Every one of those 576 image tokens occupies a slot in the **KV-cache** for the rest of the generation, because the decoder attends back to them at every step. KV-cache memory scales linearly with sequence length:

$$M_{\text{KV}} = 2 \cdot L_{\text{layers}} \cdot L \cdot d_{\text{kv}} \cdot b_{\text{bytes}},$$

where the factor of 2 is for keys and values, $d_{\text{kv}}$ is the per-token key/value dimension (heads times head-dim), and $b_{\text{bytes}}$ is the precision of the cache. A 576-token image inflates this cache by 576-token's worth at every layer — on a memory-tight phone, that can be the difference between fitting and OOM. (For the full KV-cache treatment, including quantizing it, see [LLM quantization of activations and the KV-cache](/blog/machine-learning/edge-ai/llm-quantization-activations-smoothquant-kv-cache) — the same techniques apply directly to those image-token KV entries.)

Let me put a real number on that KV inflation, because "576 tokens' worth at every layer" is abstract until you multiply it out. Take a 2.7B model: roughly $L_{\text{layers}} = 32$ layers, $d_{\text{kv}} = 2560$ (let's assume no grouped-query reduction for the worst case), and an fp16 cache so $b_{\text{bytes}} = 2$. The KV cost of the 576 image tokens alone is

$$M_{\text{KV,img}} = 2 \cdot 32 \cdot 576 \cdot 2560 \cdot 2 \approx 189 \text{ MB}.$$

That is 189 MB of cache that exists *only because there is an image in the context*, and it sits resident for the entire generation because the decoder attends back to every image token at every step. On a phone where you budgeted maybe 2 GB for the whole feature, a 189 MB surprise is a meaningful fraction — and it is per image, so a two-image prompt doubles it. Now pool the image to 144 tokens and the same formula gives ~47 MB, a 4x reduction, recovered for free along with the TTFT win. And if you additionally quantize the KV-cache to int8 ($b_{\text{bytes}} = 1$), you halve it again to ~24 MB. The image-token KV is one of the highest-leverage places to spend a quantized-cache budget precisely because image tokens are numerous, redundant, and never need to be re-read with full precision — they were a lossy compression of pixels to begin with.

#### Worked example: vision tokens to TTFT on a Jetson Orin Nano

Let me trace one image through a small VLM on a named target — a Jetson Orin Nano (the kind of edge box covered in [TensorRT and GPU edge inference on Jetson](/blog/machine-learning/edge-ai/tensorrt-and-gpu-edge-inference-on-jetson)) — running MobileVLM 2.7B quantized to 4-bit via llama.cpp's multimodal path.

- **Input:** a $336 \times 336$ photo, ViT with $P=14$. Vision tokens: $N_{\text{vis}} = 24^2 = 576$.
- **Prompt:** "What objects are on the desk?" — 8 text tokens.
- **Prefill length:** $L = 576 + 8 = 584$ tokens.
- **Vision encoder pass:** one forward pass of the ViT, ~50-90 ms on the Orin's GPU. This happens *before* the LLM sees anything.
- **LLM prefill of 584 tokens:** dominated by the quadratic attention term, ~1.2-1.7 s on the Orin for a 4-bit 2.7B model. Together with the encoder, **TTFT lands around 1.3-1.8 s**.
- **Decode:** after the first token, generation runs at the model's normal ~12-18 tokens/s for a 4-bit 2.7B on the Orin — fast, because decode is one-token-at-a-time and only the 584-token context sits in the KV-cache.

Now apply **token reduction**. MobileVLM's "Lightweight Downsample Projector" pools the 576 patch tokens down by 4x to **144 tokens** before they ever reach the LLM. Re-run the prefill length: $L = 144 + 8 = 152$. The linear term shrinks ~3.8x; the quadratic attention term shrinks ~14.7x. **TTFT drops to roughly 0.5-0.6 s** and the KV-cache contribution from the image shrinks 4x. The answer quality on standard visual-question-answering benchmarks barely moves (a point or two), because most of those 576 patches were redundant — adjacent patches of a flat wall carry almost no new information. This is the single highest-leverage optimization in edge VLMs, and it is exactly the before→after in the figure above.

### Small on-device VLMs worth knowing

The edge-VLM landscape has matured fast. The ones to know:

- **MobileVLM / MobileVLM V2** (Chu et al., 2023). Purpose-built for the edge: a smaller LLM (MobileLLaMA 1.4B or 2.7B), a lightweight downsample projector that cuts vision tokens 4x, and tuned for mobile/Jetson inference. The reference design for "VLM that respects the token budget."
- **LLaVA-class small variants.** The original LLaVA recipe (ViT + projector + Vicuna) scaled down by swapping in a small LLM. LLaVA-Phi and similar pairings put the LLaVA training recipe on top of a Phi-class 2.7B model. Same architecture, edge-sized backbone.
- **Moondream.** A tiny (~1.6-1.8B) VLM explicitly designed to run locally, with a focus on practical visual-question-answering and captioning at a footprint that fits a laptop or a beefy phone. Popular precisely because it is small and the inference story is clean.
- **The Gemma / Phi vision direction.** Google's PaliGemma and the Gemma-vision line, and Microsoft's Phi-3.5-vision / Phi-vision models, are the "designed small, vision-capable" frontier — small LLMs with native vision heads, quantizable to run on-device. This is the same "small by design beats shrunk-big" thesis from the SLM post, extended to multimodal.

All of them quantize with the llama.cpp / GGUF stack: the LLM goes to `Q4_K_M`-class 4-bit, and the vision encoder is typically kept at a higher precision (fp16 or int8) because it is small relative to the LLM and more sensitive to quantization — a recurring pattern we will flag in the checklist.

### Running a small VLM: llama.cpp multimodal and measuring TTFT

llama.cpp's multimodal support (the `mtmd` / `llava` path, using a separate multimodal projector file) is the most portable way to run a small VLM on the edge. The flow: get the quantized LLM GGUF plus the vision projector (`mmproj`) file, then run with both.

```bash
# Build llama.cpp with the multimodal CLI
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
cmake -B build -DGGML_NATIVE=ON
cmake --build build -j --config Release

# You need TWO files for a VLM:
#   1) the quantized LLM weights (GGUF), and
#   2) the multimodal projector (mmproj GGUF) for the vision encoder.
# These ship together for MobileVLM / LLaVA / Moondream on Hugging Face.

# Ask a question about an image
./build/bin/llama-mtmd-cli \
  -m models/mobilevlm-2.7b-q4_k_m.gguf \
  --mmproj models/mobilevlm-2.7b-mmproj-f16.gguf \
  --image samples/desk.jpg \
  -p "What objects are on the desk?" \
  -ngl 99 --temp 0.2
```

`-ngl 99` offloads all layers to the GPU (on a Jetson or a Metal Mac); drop it for CPU-only. The `--mmproj` file is the vision encoder plus projector — it is what turns `--image` into the 576 (or pooled 144) image tokens that get prepended to your prompt.

To measure TTFT with vision tokens — the metric that actually matters for a camera-Q&A feature — you want the wall-clock from "image submitted" to "first output token." A clean way is to instrument it in Python against a server that reports timing, or to parse llama.cpp's own prompt-eval timing (which is the prefill time, the dominant chunk of TTFT):

```python
import subprocess, time, re

def measure_ttft(model, mmproj, image, prompt, ngl=99):
    """Run llama-mtmd-cli once and pull prefill (prompt-eval) timing,
    which dominates time-to-first-token for a VLM with image tokens."""
    cmd = [
        "./build/bin/llama-mtmd-cli",
        "-m", model, "--mmproj", mmproj,
        "--image", image, "-p", prompt,
        "-ngl", str(ngl), "--temp", "0.2", "-n", "32",
    ]
    t0 = time.time()
    out = subprocess.run(cmd, capture_output=True, text=True)
    wall = time.time() - t0
    # llama.cpp prints "prompt eval time = ... ms / N tokens"
    m = re.search(r"prompt eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*tokens", out.stderr)
    if m:
        prefill_ms, n_tok = float(m.group(1)), int(m.group(2))
        print(f"prefill={prefill_ms:.0f} ms over {n_tok} tokens "
              f"({prefill_ms/n_tok:.1f} ms/token) | total wall={wall:.2f}s")
    return out.stdout

# Warm up once (page in the model), then measure
measure_ttft("models/mobilevlm-2.7b-q4_k_m.gguf",
             "models/mobilevlm-2.7b-mmproj-f16.gguf",
             "samples/desk.jpg", "What objects are on the desk?")
measure_ttft("models/mobilevlm-2.7b-q4_k_m.gguf",
             "models/mobilevlm-2.7b-mmproj-f16.gguf",
             "samples/desk.jpg", "What objects are on the desk?")
```

The `N tokens` in that prompt-eval line is your real prefill length — and you will see it sitting up at ~584 (text + 576 image tokens) for naive LLaVA-style, or ~152 for a token-pooled MobileVLM. Watching that number is the most direct way to confirm the vision-token blowup is real on your own hardware, and to confirm token reduction actually shrank it.

### The token-reduction lever, in code

Since token reduction is the highest-leverage VLM optimization, it is worth seeing what it actually does to the patch embeddings — it is not magic, it is a pooling op between the vision encoder and the projector. The simplest form, the one MobileVLM's downsample projector uses, is an average-pool over a $2 \times 2$ neighborhood of patches on the 2D patch grid, which turns a $24 \times 24$ grid of 576 tokens into a $12 \times 12$ grid of 144 tokens before the LLM ever sees them:

```python
import torch
import torch.nn.functional as F

def downsample_vision_tokens(patch_embeds, grid=24, factor=2):
    """patch_embeds: [B, N, D] from the ViT, N = grid*grid (e.g. 576).
    Pool 2x2 neighborhoods on the patch grid -> N/factor^2 tokens."""
    B, N, D = patch_embeds.shape
    assert N == grid * grid, f"{N} != {grid}x{grid}"
    # back to a 2D feature map: [B, D, grid, grid]
    x = patch_embeds.transpose(1, 2).reshape(B, D, grid, grid)
    # average-pool the spatial grid by `factor`
    x = F.avg_pool2d(x, kernel_size=factor, stride=factor)  # [B, D, grid/f, grid/f]
    # flatten back to a token sequence
    new_grid = grid // factor
    x = x.reshape(B, D, new_grid * new_grid).transpose(1, 2)  # [B, N/f^2, D]
    return x  # 576 -> 144 tokens when factor=2

vit_out = torch.randn(1, 576, 1024)        # pretend ViT output
pooled  = downsample_vision_tokens(vit_out)  # [1, 144, 1024]
print(vit_out.shape, "->", pooled.shape)     # confirms 576 -> 144
```

Two things to notice. First, the pooling preserves the 2D spatial structure — adjacent patches get averaged, which is why it loses so little accuracy: neighboring patches of a wall, the sky, or a uniform surface are nearly identical, so averaging them throws away redundancy, not signal. Where it *does* hurt is fine-grained tasks (reading small text, counting dense objects), which is the honest limit of pooling and the reason high-resolution document VLMs keep their tokens. Second, the projector (the MLP that maps ViT-dim to LLM-dim) runs on the *pooled* 144 tokens, so it gets cheaper too — but the projector was never the bottleneck. The win is entirely downstream: the LLM prefills 144+8 instead of 576+8 tokens, and the KV-cache holds 144 image-token entries instead of 576. One pooling op, applied once, pays out across the entire generation.

## The unified science: latency budget and multimodal memory

Speech gave us the RTF cliff. Vision gave us the prefill blowup. Step back and there is one shared physical constraint behind both — the **latency budget for "feels live"** — and one shared resource constraint — **multimodal memory**.

### The latency budget for "feels live"

Human perception has well-studied thresholds, and they set hard targets that no amount of accuracy can buy back. For interactive response — the gap between a user's action and the system's visible reaction:

- **Under ~100 ms** feels *instantaneous* — the system seems to react as fast as a direct physical manipulation.
- **~100-300 ms** feels *responsive* — there is a perceptible delay, but the flow of interaction holds. This is the target band for conversational turn-taking and live captions.
- **~300 ms-1 s** feels *laggy but usable* — the user notices the wait and starts to disengage; acceptable for a one-shot query, painful for back-and-forth.
- **Over ~1 s** the user's attention wanders; you need a progress indicator or streaming partials to hold them.

For live captioning, the relevant latency is "how far behind the speaker is the displayed text," and the target is comfortably inside the responsive band — you want captions trailing speech by a few hundred milliseconds, not seconds. That is why streaming ASR with a small hop matters more than offline accuracy for a captioning product: a slightly-wrong word *now* beats a perfect word *two seconds late*.

For camera-Q&A, the relevant latency is TTFT plus the time to speak/render the first chunk of answer. Here the budget is more forgiving — a one-shot question tolerates the laggy-but-usable band — but it is exactly the band the vision-token blowup threatens, which is why token reduction is not optional for a snappy camera-Q&A feel. You are trying to drag TTFT from "over a second" down toward the responsive band, and pooling 576 image tokens to 144 is what gets you there.

A useful way to budget an end-to-end multimodal turn, additively, for a spoken question about an image:

$$T_{\text{turn}} \approx \underbrace{T_{\text{KWS}}}_{\sim 0} + \underbrace{T_{\text{ASR}}}_{\text{RTF} \cdot T_{\text{audio}}} + \underbrace{T_{\text{enc}}}_{\text{vision pass}} + \underbrace{T_{\text{prefill}}}_{\text{TTFT}} + \underbrace{T_{\text{first speak}}}_{\text{TTS start}}.$$

Each term is a lever. The wake-word gate is effectively free (it ran in the background). ASR latency is the RTF times the utterance length, so keep RTF low. Vision encoding is a fixed cost you pay per image. Prefill is the term that blows up with vision tokens — and the term token reduction attacks. The last term, starting to speak, is the TTS first-chunk latency, which streaming TTS keeps small. Sum them and compare to the budget: if it busts, the equation tells you *which term* to cut.

There is a scheduling subtlety hiding in that sum: not every term is strictly sequential. The ASR and the vision encoding can overlap, because the camera frame is captured the moment the user starts speaking — you do not have to wait for the transcript to finish before running the ViT. A good implementation kicks off the vision encoder on the captured frame *in parallel* with the streaming ASR, so by the time the user stops talking and the transcript finalizes, the image tokens are already computed and waiting. That overlap turns the additive budget into something closer to $T_{\text{turn}} \approx \max(T_{\text{ASR}}, T_{\text{enc}}) + T_{\text{prefill}} + T_{\text{first speak}}$, shaving the encoder term off the critical path entirely when ASR is the longer of the two (which it usually is for a multi-word question). On a phone with both a CPU running ASR and a GPU/NPU running the ViT, this is free parallelism you would be foolish not to take. The only constraint is memory — running both encoders at once means both activation working sets are resident simultaneously, which is exactly the peak the next section warns about.

#### Worked example: budgeting a spoken camera question end to end

Trace a full turn on a flagship phone: the user holds up the camera, says "what kind of plant is this?" (about 1.5 seconds of speech), and wants a spoken answer that *feels* prompt. Lay out the budget term by term, target the responsive-to-laggy band (under ~1 s to first audible word is the goal):

- **$T_{\text{KWS}}$**: the wake word ("hey assistant") already fired and woke the stack; by the time the question starts the heavy models are warm. Effective cost on the critical path: **~0 ms**.
- **$T_{\text{ASR}}$**: streaming ASR at the committed-lag of ~1 hop behind the speaker. The transcript finalizes roughly $H \approx 1$ s after the last word, but partials were flowing the whole time, so the *usable* transcript is ready about **300 ms** after speech ends if you accept the last partial. Call it **300 ms** to a confident transcript.
- **$T_{\text{enc}}$**: ViT forward pass on the $336 \times 336$ frame, ~60 ms on the phone's NPU — **but it ran in parallel with the ASR**, so it is already done. Critical-path cost: **~0 ms**.
- **$T_{\text{prefill}}$**: with a token-reduced VLM at 144 image tokens plus ~10 text tokens, prefill of ~154 tokens on a flagship NPU for a 4-bit 2-3B model lands around **350-450 ms**. This is the dominant term, and it is dominant *only because* we pooled the tokens — at 576 tokens it would be ~1.4 s and bust the budget by itself.
- **$T_{\text{first speak}}$**: streaming TTS starts vocalizing as soon as the first few words of the answer are decoded, ~150 ms to the first audible phoneme.

Sum the critical path: $0 + 300 + 0 + 400 + 150 \approx 850$ ms from "user stops talking" to "answer starts speaking." That is inside the laggy-but-usable band and feels like a quick, thoughtful pause — acceptable for a one-shot question. Now do the same arithmetic *without* token reduction: the prefill term jumps to ~1.4 s, the total to ~1.85 s, and the interaction crosses from "thoughtful pause" into "is it broken?" The single decision that moved this feature from shippable to sluggish was pooling 576 tokens to 144. The budget equation made that legible: every other term was already small or overlapped away, so the prefill term was the only one worth attacking.

### Multimodal memory: model plus KV plus encoder activations

Memory is the constraint that quietly kills edge multimodal, and the full series treatment is in [memory is the real constraint](/blog/machine-learning/edge-ai/memory-is-the-real-constraint). The new wrinkle for multimodal is that you are now holding *three* things in RAM at peak, not one:

$$M_{\text{peak}} \approx \underbrace{M_{\text{LLM weights}}}_{\text{static, streamable}} + \underbrace{M_{\text{encoder weights}}}_{\text{static}} + \underbrace{M_{\text{KV-cache}}}_{\text{grows with context + image tokens}} + \underbrace{M_{\text{encoder activations}}}_{\text{transient, peaks during the vision pass}}.$$

The first two are static weights — for a 4-bit 2.7B LLM that is ~1.4 GB, plus a few hundred MB for an fp16 ViT encoder. The KV-cache, as derived above, is inflated by the image tokens. And the often-forgotten term: the **vision encoder's activations** during its forward pass. A ViT processing a $336 \times 336$ image at fp16 holds a non-trivial activation working set — the patch embeddings, the attention maps across 576 patches at each layer. That working set peaks *during the encoder pass*, before the LLM prefill, so if you schedule naively you can have the encoder's activations and the LLM weights resident simultaneously and blow your budget.

The fix is the same lifetime-aware scheduling from the memory post: run the encoder, extract the (small) image tokens, *free the encoder activations*, then run the LLM. The image tokens themselves are tiny (576 vectors); it is the encoder's *intermediate* activations that are large and transient. Treat them as a transient that must be freed before the LLM's peak, exactly like any other activation working set.

Put a number on the transient so the danger is concrete. A ViT-L processing a $336 \times 336$ image at fp16 holds, at each of its ~24 layers, the patch hidden states ($576 \times 1024$) plus the attention score matrix ($576 \times 576$ per head, times ~16 heads). The hidden states alone are about $576 \times 1024 \times 2 \approx 1.2$ MB per layer, and the attention scores add another few MB per layer at peak; with intermediate MLP activations the per-layer working set is several MB, and the runtime typically keeps a couple of layers live at once during the forward pass. Round it: the encoder's transient activation peak is on the order of **a few hundred MB** for a high-resolution pass. That is the same order as a quarter of your whole feature budget — appearing for tens of milliseconds and then, if you free it correctly, vanishing before the LLM's prefill peak. If you *don't* free it (a naive implementation that holds the whole graph live), that few-hundred-MB transient coexists with the 1.4 GB of LLM weights and the inflating KV-cache, and you OOM on a device that would have fit comfortably with proper lifetime scheduling. The bug is invisible on a desktop with 32 GB of RAM and fatal on a phone with 6 GB shared between the OS, the camera pipeline, and your model.

There is a related trap with the camera pipeline itself. The raw frame, the decoded RGB tensor, the resized-and-normalized model input, and the preprocessing intermediates are all activations too, and they live *before* the ViT even starts. On a phone the camera stack is already holding a couple of full-resolution frame buffers; if your preprocessing copies rather than views, you can have three copies of a 12-megapixel frame resident at once. The discipline is the same as inside the model: each preprocessing buffer has a lifetime, and you free the big raw frame the instant you have the small $336 \times 336$ normalized tensor. Multimodal memory is not just the model's three terms — it is the whole pipeline from sensor to token, and the sensor end can be the surprise.

#### Worked example: does the feature fit in 6 GB?

A flagship phone advertises 8 GB of RAM, but the OS, the launcher, the camera service, and whatever else is open typically leave you ~6 GB *and the OS will kill you well before zero*. Budget a token-reduced camera-Q&A feature and check:

- **LLM weights**, 4-bit 2.7B: ~1.4 GB, static, memory-mappable (so it can be paged, but assume resident for latency).
- **Vision encoder weights**, fp16 ViT-L (~300M params): ~0.6 GB, static.
- **KV-cache**, 144 image tokens + ~256-token conversation at fp16 on a 2.7B: from the formula above, ~130 MB. (At int8 KV: ~65 MB.)
- **Encoder transient activations**, freed before LLM prefill if scheduled right: peaks at ~0.3 GB *during the ViT pass only*.
- **Camera/preprocessing buffers**: ~0.1 GB if you free the raw frame promptly.

The *static* floor is $1.4 + 0.6 = 2.0$ GB. The *peak* — if encoder activations and camera buffers are freed before LLM prefill, which they should be — is the static floor plus the larger of {encoder transient, KV-cache during decode}, i.e. roughly $2.0 + 0.3 \approx 2.3$ GB at the encoder moment, dropping to $2.0 + 0.13 \approx 2.13$ GB during decode. Comfortably inside 6 GB. **But** the naive schedule that holds the encoder graph live through LLM prefill stacks everything: $1.4 + 0.6 + 0.3 + 0.13 + 0.1 \approx 2.53$ GB — still fits here, but on a 4 GB mid-ranger with ~2.5 GB usable, that naive 2.53 GB peak is the difference between shipping and a low-memory kill. The lesson generalizes: on the flagship you can be sloppy; on the device that actually represents your install base, lifetime-aware scheduling of the encoder transient is what keeps you under the OS killer. Always budget against the peak, against your *weakest* supported device, with the OS's real headroom — not the spec sheet's RAM number.

## Results: measured tables on named targets

Time for the detailed-proof part of the series mandate — before→after numbers on named hardware. Two tables: ASR (Whisper size x quant) and VLM (model x vision tokens). All figures are representative of published whisper.cpp / llama.cpp benchmarks and the source papers; measure your own target, because RTF and TTFT swing 2-4x between a laptop CPU and a phone NPU.

![A matrix of on-device results comparing Whisper tiny and base against MobileVLM showing footprint, speed metric, quality, and named target hardware](/imgs/blogs/multimodal-and-speech-at-the-edge-7.png)

### ASR results: Whisper size x quantization on an M2 MacBook (CPU, batch=1)

| Model | Quant | File size | Peak RAM | RTF (lower = faster) | WER (clean EN) | Verdict |
| --- | --- | --- | --- | --- | --- | --- |
| tiny.en | f16 | ~75 MB | ~110 MB | ~0.10 | ~12% | fastest, gist-only |
| tiny.en | Q5_0 | ~31 MB | ~75 MB | ~0.12 | ~12-13% | command/control |
| base.en | f16 | ~142 MB | ~210 MB | ~0.25 | ~8% | strong default |
| base.en | Q5_0 | ~57 MB | ~140 MB | ~0.30 | ~8-9% | **edge sweet spot** |
| small.en | Q5_0 | ~182 MB | ~340 MB | ~0.6-0.8 | ~5-6% | accuracy, thin headroom |

Read the RTF column against the cliff: every row here is below one, so all of these *can* stream live on this laptop. On a phone-class chip, multiply RTF by roughly 2-4x for CPU and the small row creeps toward or past one — which is why base is the safe floor on handsets. Quantization to Q5 costs you essentially nothing in WER (a fraction of a point) while halving the file and shaving the RAM; it is the free lunch. The one honest caveat: quantization can occasionally hurt *rare-word and accented* recognition more than the clean-English WER suggests, so validate on your actual domain audio, not just on the standard test set.

### VLM results: small models, vision-token reduction, on a Jetson Orin Nano

| Model | LLM size | Quant | Vision tokens | TTFT | Decode tok/s | Peak RAM | Note |
| --- | --- | --- | --- | --- | --- | --- | --- |
| LLaVA-1.5-class | 7B | Q4_K_M | 576 | ~3.0-3.5 s | ~6-8 | ~5.5 GB | too heavy for handsets |
| MobileVLM 2.7B | 2.7B | Q4_K_M | 576 (no pool) | ~1.3-1.8 s | ~12-18 | ~1.8 GB | image-token bound |
| MobileVLM V2 2.7B | 2.7B | Q4_K_M | 144 (pooled 4x) | ~0.5-0.6 s | ~14-20 | ~1.6 GB | **token-reduced** |
| Moondream2 | ~1.8B | Q4_K_M | ~144-256 | ~0.5-0.8 s | ~16-22 | ~1.4 GB | tiny, laptop/phone |

The story the table tells: **vision tokens, not parameters, gate TTFT on a small VLM.** Dropping from a 7B to a 2.7B halves TTFT; pooling the *same* 2.7B's vision tokens from 576 to 144 halves it again — and the second halving is nearly free in accuracy. Two models with the same LLM backbone differ in TTFT mostly by their vision-token count. That is the lever. If your camera-Q&A feels sluggish, look at the prompt-eval token count first, not the model size.

How to measure these honestly: warm up (page in the model, JIT the kernels), pin the clock if you can, watch thermals (a Jetson under sustained load throttles and your TTFT drifts up over a few minutes), and report batch=1 because that is the on-device reality — no request batching to hide latency behind. Report TTFT and decode-tok/s separately, because they have different bottlenecks (prefill is compute-bound, decode is memory-bandwidth-bound) and a single "tokens per second" number averages away the thing the user actually feels.

## Stress test: what breaks under load

Benchmarks run on a cold device with a clean clip and a single image, and they all look great. Production is none of those things. The honest engineering question is not "how fast is it cold and idle?" but "what is the first thing that breaks when reality leans on it?" Three failure modes dominate, and each maps to one of the constraints we derived.

**Failure 1: RTF crosses one under sustained streaming.** This is the speech cliff arriving in slow motion. You ship `small.en` because it hit RTF 0.7 in testing, and live captioning works beautifully — for the first ninety seconds. Then the SoC, busy at 70% utilization continuously, heats past its thermal limit and the governor throttles the clock by 20-30%. Every RTF scales inversely with clock speed, so your comfortable 0.7 drifts to 0.9, then a dense window peaks at 1.05, and the backlog starts growing. The captions, which were trailing speech by 1.5 seconds, begin trailing by 3, then 5, then they are useless. The user experience is not "captions got a bit slower" — it is "captions worked and then silently fell apart five minutes into the meeting," which is worse, because the failure is delayed and correlated with exactly the long sessions where captions matter most. The fix is not a faster model; it is *headroom*. You must validate at the throttled clock, budget RTF against the worst-case window at the throttled clock, and if `small` cannot hold it, step down to `base` even though `small` "passed" the cold benchmark. The cold benchmark was measuring the wrong machine.

**Failure 2: vision tokens explode TTFT on a high-resolution image.** Your camera-Q&A feature is tuned and snappy on the $336 \times 336$ test images — TTFT ~500 ms, everyone is happy. Then a user points the camera at a dense restaurant menu and asks "what's the cheapest vegetarian dish?" To read small text, the VLM's AnyRes path tiles the image at high resolution, and the token count quietly jumps from 144 to 2,000+. Prefill, which was linear-ish in token count and comfortable at 154 tokens, now has to ingest 2,000+ tokens — and at this scale the *quadratic* attention term has woken up because the sequence length finally exceeds the model dimension. TTFT does not double; it goes from 500 ms to 6-8 seconds, and the KV-cache for those 2,000 tokens blows past your memory budget and either OOMs or evicts. The feature that felt instant on a houseplant feels broken on a menu — and menus, signs, and documents are exactly the high-value visual-question-answering cases. The honest fix is a *resolution policy*: cap the input resolution (accepting that you cannot read the smallest text), or detect the document case and explicitly offload it to the cloud, or expose a "read fine detail" mode that warns the user it will be slower. What you cannot do is let an unbounded token count silently determine your latency. Count $N_{\text{vis}} = (S/P)^2$ for your *maximum* allowed resolution and make sure even that worst case fits the budget, or gate it.

**Failure 3: thermal throttling under sustained multimodal streaming.** Combine the two. A live "narrate what the camera sees" feature runs the ASR, the vision encoder, and the LLM prefill *continuously*, which is the most thermally brutal workload a phone can sustain — there is no idle gap for the SoC to cool. Within a few minutes the chip is at its thermal ceiling, the governor has throttled every clock, and *both* failure modes fire at once: ASR RTF crosses one, and VLM TTFT drifts up as the GPU clock drops. Battery temperature climbs, and on some phones the OS will start killing background processes or even your foreground app to shed heat. This is the workload that exposes whether your headroom was real. The mitigations are architectural, not parametric: duty-cycle the heavy stack (process a frame every N seconds, not every frame), drop the vision resolution under thermal pressure (a feedback loop that reads the thermal state and reduces tokens), and — the honest one — accept that *continuous* multimodal narration is at the edge of what a passively-cooled phone can sustain, and design the feature to degrade gracefully (lower frame rate, shorter answers, "tap to analyze" instead of continuous) rather than melt down. The wake-word gate that saved the always-on case has no equivalent here, because the whole point of the feature is to run continuously; the only lever left is to do *less work per second*.

The thread through all three: the cold, idle, single-input benchmark is the best case, and the device spends almost none of its real life there. Budget against the worst case — throttled clock, maximum-resolution image, sustained load — and the failures above become design parameters you chose, not surprises that page you at 2am.

## Case studies: real numbers from the literature

A few grounded results, cited, to anchor the claims above.

**Whisper (Radford et al., 2022).** The original paper's headline is robustness from scale of weak supervision — Whisper approaches human-level robustness on out-of-distribution English without dataset-specific fine-tuning. The size ladder (tiny 39M → large 1.55B) is from the paper; the WER falls monotonically as you go up the ladder, which is exactly the accuracy-latency Pareto you trade against on device. whisper.cpp then makes the bottom of that ladder run in real time on commodity CPUs — the project's own benchmarks show tiny and base at RTF well under one on laptops and even on a Raspberry Pi for tiny.

**LLaVA (Liu et al., 2023).** Established the visual-instruction-tuning recipe — frozen CLIP ViT-L/14 vision encoder, a projection layer, and a Vicuna LLM — and showed you can build a capable VLM cheaply by mostly training the projector and lightly tuning the LLM. The 336px / 14px-patch / 576-token configuration we counted above is straight from LLaVA-1.5. It is the architecture everyone scaled down for the edge.

**MobileVLM (Chu et al., 2023).** The explicit edge-VLM paper: a mobile-tuned LLM (MobileLLaMA), a CLIP-style vision encoder, and the Lightweight Downsample Projector that cuts vision tokens roughly 4x. The paper reports running at interactive speeds on a Snapdragon mobile SoC and a Jetson Orin — the token-reduction-drives-TTFT result is theirs, and it is the single most important edge-VLM finding to internalize.

**Moondream.** A community-driven tiny VLM (~1.6-1.8B) built specifically for local inference, demonstrating that a sub-2B VLM can do genuinely useful captioning and visual-question-answering on a laptop, and increasingly on phones. It is the existence proof that "useful VLM" and "fits on device" are no longer mutually exclusive.

The thread through all four: the *vision-language capability* is established and the *edge cost* is dominated by the vision-token count, which is exactly where the engineering leverage lives.

## When on-device multimodal is ready (and when to offload)

Decisive recommendation time. On-device multimodal is genuinely ready for some things and genuinely not ready for others, and the boundary is predictable. The decision splits cleanly by modality, and within each modality there is one constraint that gates it.

![A decision tree splitting on-device multimodal readiness into a speech branch gated by real-time factor and a vision branch gated by vision-token budget, each ending in a ship or offload rule](/imgs/blogs/multimodal-and-speech-at-the-edge-8.png)

**Speech: ship on device when RTF stays below ~0.7 with a wake-word gate.** Command-and-control, live captions, and dictation with tiny/base/small Whisper are ready *today* on a mid-range phone, provided you gate with a wake-word spotter so you are not running the recognizer all day. Offload to cloud ASR when you need large-Whisper-class accuracy (medical transcription, rare languages, heavy accents in noise) that the on-device sizes cannot hit, or when the device is too weak to keep RTF below one even at tiny — then stream the audio up and accept the network round-trip.

**Vision: ship on device when you can keep vision tokens at or below ~256 and TTFT under ~1 s.** Camera-Q&A, on-device captioning, and visual command ("what's this plant," "read this label") with a token-reduced 2-3B VLM (MobileVLM V2, Moondream) are ready on a flagship phone or a Jetson. Offload when you need high-resolution document understanding (4,000+ vision tokens — the prefill simply will not fit the latency budget on a handset), or when you need a large VLM's reasoning that a 2-3B backbone cannot match.

The cross-cutting gates that override both branches:

- **Thermal.** Sustained multimodal load throttles the SoC, and a model that hit RTF 0.6 cold drifts to RTF 0.9 hot. Always validate under sustained load, not on a cold first run. Budget headroom for the throttled state, not the cold state.
- **Power.** The wake-word gate is non-negotiable for always-on features; without it the battery math does not close.
- **Memory.** Add up LLM weights + encoder weights + image-inflated KV-cache + transient encoder activations and check it against the device's real available RAM (not its spec sheet — the OS and other apps take a big bite). If the peak busts, you offload or you shrink (smaller LLM, fewer vision tokens, quantized KV-cache).

### The real-time multimodal-on-edge checklist

Run down this list before you commit a multimodal feature to on-device:

1. **RTF < 1 with headroom.** Measure RTF on your *weakest* target device, under thermal load, and demand RTF ≤ 0.7. If you cannot, the speech path offloads.
2. **Vision-token budget.** Count $N_{\text{vis}} = (S/P)^2$ for your encoder and resolution. If it is over ~256, apply token pooling/downsampling before you blame the LLM. TTFT lives here.
3. **Streaming, not batch.** For live speech, chunk with a small hop and a stable-commit policy; do not re-run offline Whisper on a growing buffer.
4. **Wake-word gating.** An always-on <50 KB KWS up front. The heavy stack runs only when triggered. This is what makes always-on power-feasible.
5. **Quantize the LLM hard, the encoder gently.** 4-bit GGUF for the language model; keep the vision encoder at fp16 or int8 — it is small and quantization-sensitive, and the savings are not worth the accuracy risk.
6. **Free encoder activations before LLM prefill.** Lifetime-aware scheduling so the transient vision activations do not coexist with the LLM's peak.
7. **Thermal validation.** Test under sustained load; budget for the throttled RTF and TTFT, not the cold-start numbers.
8. **Measure TTFT and decode separately.** They have different bottlenecks; one averaged tok/s number hides the latency the user actually feels.

If every box is checked, ship it on device. If a box fails and you cannot fix it with a lever, that is your offload boundary — and now you know precisely *which part* to offload, not "the whole thing to the cloud."

## Key takeaways

- **The frontier of on-device AI is multimodal, not text.** Text LLMs on a phone are solved enough; the open problem is seeing, hearing, and responding in real time, locally — and it composes from the same four levers applied to an audio encoder-decoder and a vision encoder.
- **Speech lives or dies on the real-time factor.** $\text{RTF} = T_{\text{compute}} / T_{\text{audio}}$; below one you can stream live, at or above one your latency backlog grows without bound. It is a cliff, not a slope. Target RTF ≤ 0.7 for headroom.
- **Offline Whisper is not streaming.** Its accuracy comes from bidirectional 30-second context; streaming trades 2-4 WER points of that context for partial transcripts in hundreds of milliseconds. Use chunked sliding windows, not a growing buffer.
- **A wake-word gate makes always-on feasible.** A <50 KB keyword spotter on a low-power island runs continuously; the 75 MB recognizer runs only when triggered. The power math only closes with the gate.
- **One image is hundreds of tokens.** $N_{\text{vis}} = (S/P)^2 = 576$ for LLaVA-1.5 defaults. Those tokens flow through the quadratic prefill, multiplying TTFT and inflating the KV-cache. The image, not the prompt, dominates.
- **Vision-token reduction is the single highest-leverage edge-VLM optimization.** Pooling 576 → 144 tokens roughly quarters the linear prefill cost and cuts the quadratic term ~15x, dropping TTFT from seconds to sub-second, at a cost of a point or two of accuracy.
- **Budget the latency additively.** $T_{\text{turn}} \approx T_{\text{ASR}} + T_{\text{enc}} + T_{\text{prefill}} + T_{\text{first speak}}$. "Feels live" is the 100-300 ms responsive band; the equation tells you which term to cut.
- **Multimodal memory is three things, not one.** LLM weights + encoder weights + image-inflated KV-cache + transient encoder activations. Free the encoder activations before LLM prefill or you double-count the peak.
- **Quantize the LLM hard, the encoder gently.** 4-bit for the language model; fp16/int8 for the small, sensitive vision encoder.

## Further reading

- **Radford et al., "Robust Speech Recognition via Large-Scale Weak Supervision" (Whisper, 2022)** — the model, the size ladder, the robustness-from-scale thesis.
- **whisper.cpp** (Gerganov) — the ggml/GGUF port, the `stream` example, and the on-device benchmarks; the quantization and CLI flags used above.
- **Liu et al., "Visual Instruction Tuning" (LLaVA, 2023)** and the LLaVA-1.5 follow-up — the encoder-projector-LLM recipe and the 336px/576-token configuration.
- **Chu et al., "MobileVLM: A Fast, Strong and Open Vision Language Assistant for Mobile Devices" (2023)** — the edge-VLM design and the lightweight downsample projector that drives the token-reduction result.
- **Moondream** — a tiny VLM built for local inference; the existence proof for sub-2B on-device VLMs.
- **Within this series:** the [taxonomy of model compression](/blog/machine-learning/edge-ai/a-taxonomy-of-model-compression) for the four-lever Pareto frame; [running LLMs locally with llama.cpp and GGUF](/blog/machine-learning/edge-ai/running-llms-locally-llama-cpp-and-gguf) and [making on-device LLMs fast](/blog/machine-learning/edge-ai/making-on-device-llms-fast) for the LLM backbone; [efficient attention and vision transformers for the edge](/blog/machine-learning/edge-ai/efficient-attention-and-vision-transformers-for-edge) for the vision encoder; [small language models by design](/blog/machine-learning/edge-ai/small-language-models-by-design) for the LLM choice; [TinyML on microcontrollers](/blog/machine-learning/edge-ai/tinyml-on-microcontrollers) for the wake-word front end; and the capstone [edge optimization playbook](/blog/machine-learning/edge-ai/the-edge-optimization-playbook) that ties the whole series together.
