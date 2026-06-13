---
title: "Kokoro-82M: How an 82-Million-Parameter Model Reaches Near Human-Level TTS"
date: "2026-06-13"
publishDate: "2026-06-13"
description: "A full engineering teardown of Kokoro-82M — the cut-down StyleTTS 2 architecture, the iSTFTNet vocoder, the length-indexed voicepacks, the misaki G2P, and the loss stack that trained it — and why a tiny non-autoregressive model beats billion-parameter LLM-TTS on cost and latency."
tags:
  [
    "text-to-speech",
    "kokoro",
    "styletts2",
    "istftnet",
    "vocoder",
    "speech-synthesis",
    "g2p",
    "signal-processing",
    "neural-audio",
    "adain",
  ]
category: "machine-learning"
subcategory: "Signal Processing"
author: "Hiep Tran"
featured: true
readTime: 51
---

Every few months a model shows up that quietly violates the working assumptions of its field. In late 2024, Kokoro-82M did that for text-to-speech. It is **82 million parameters** — smaller than a single transformer layer of a frontier LLM — yet on blind listening tests it traded blows with systems twenty to forty times its size, ran faster than real time on a laptop CPU, and shipped under an Apache-2.0 license you can put in a commercial product without a lawyer. The entire model is a 327 MB file. The author, who goes by `hexgrad`, trained it for roughly the price of a nice dinner.

If your mental model of "good TTS" is a multi-billion-parameter autoregressive model that emits audio tokens one at a time — the architecture behind most of the impressive demos from 2023–2024 — then Kokoro is a useful shock. It is not autoregressive. It runs no diffusion at inference. It has no language-model backbone. It generates an entire utterance in **one forward pass**, and the "voice" you select is not a learned speaker embedding sampled at runtime but a frozen lookup table baked into the weights.

This post is a full teardown. We will reconstruct the architecture from its two ancestors — **StyleTTS 2** ([arXiv:2306.07691](https://arxiv.org/abs/2306.07691)) for the acoustic model and **iSTFTNet** ([arXiv:2203.02395](https://arxiv.org/abs/2203.02395)) for the vocoder — walk the inference path component by component, dissect the voicepack format, work through the loss stack that produced the weights, and close with production war-stories and a decision framework for when Kokoro is the right tool and when it is a trap.

A note on honesty up front: Kokoro's model card is famously terse. The training data is described only as "a few hundred hours" of permissive audio, the exact recipe is not published, and there is no Kokoro-specific paper. Where I describe losses and training stages I am describing **StyleTTS 2's** recipe, because that is the architecture Kokoro is built on, and I will flag clearly which parts Kokoro confirms it changed (it dropped the diffusion sampler; it froze the styles into voicepacks) versus which parts are inherited and inferred. I will not invent numbers the model card does not give.

## Why a tiny model has no business sounding this good

The reason Kokoro is surprising is that the dominant 2023–2024 paradigm pushed TTS toward LLM-shaped architectures: tokenize audio with a neural codec, then train a big autoregressive transformer to predict audio tokens from text tokens, exactly the way a language model predicts the next word. That paradigm gave us zero-shot voice cloning and expressive prosody, but it inherited the LLM tax: billions of parameters, autoregressive latency that scales with output length, and a sampling temperature that occasionally produces a hallucinated word or a garbled tail.

Kokoro descends from the **other** branch of the TTS family tree — the non-autoregressive, parametric branch that goes FastSpeech → VITS → StyleTTS → StyleTTS 2. This branch never tokenizes audio. It predicts a duration for each phoneme, expands the phoneme sequence to frame resolution in a single deterministic step, and feeds those frames to a vocoder that emits a waveform. No token-by-token loop, no codec round-trip, no sampling lottery.

| Assumption | The naive view | The reality with Kokoro |
| --- | --- | --- |
| Good TTS needs billions of parameters | Quality scales with model size, like LLMs | 82M reaches near-human MOS on clean read speech |
| Naturalness requires autoregression | You must condition each frame on the last | One non-autoregressive pass; prosody comes from a predictor, not a sampler |
| You need diffusion at inference | Diffusion is what makes it sound real | Diffusion was used in the *parent* model; Kokoro ships the decoder only |
| A "voice" is a learned speaker embedding | Sampled or encoded from reference audio at runtime | A frozen 256-d vector looked up from a table |
| Multilingual means one big model | Scale the model to add languages | Multilingual comes mostly from the **G2P front-end**, not the acoustic model |
| Real-time TTS needs a GPU | Neural vocoders are heavy | Runs faster than real time on a CPU; iSTFT does the heavy lifting |

The right way to read that table: almost every cost in modern TTS comes from choices Kokoro simply declines to make. It declines autoregression, so latency is flat. It declines runtime diffusion, so there is no iterative sampler. It declines runtime voice encoding, so there is no reference-audio path to get wrong. What is left is a lean feed-forward pipeline whose only job is to turn phonemes plus a style vector into a spectral representation that an inverse FFT can finish.

That leanness is also Kokoro's ceiling, and we will be honest about that throughout. It cannot clone a voice from three seconds of reference audio. It cannot invent a speaking style it never saw. Its multilingual support is only as good as its phonemizer. These are not bugs; they are the direct consequences of the same decisions that make it tiny and fast.

## The mental model: one forward pass, no autoregression

![Kokoro inference: one non-autoregressive forward pass from text through G2P, BERT, the prosody predictor, length regulation, and the decoder to 24 kHz audio.](/imgs/blogs/kokoro-82m-1.webp)

The diagram above is the mental model, and the rest of this article is a tour of it. Read it left to right. Raw text enters the **misaki** grapheme-to-phoneme front-end, which produces a phoneme sequence capped at 510 tokens. Those phonemes fan out to two encoders in parallel: a phoneme-level **ALBERT** model (a compact BERT that supplies prosody-aware contextual features) and a convolutional **text encoder** that produces the linguistic content stream. A **prosody predictor** consumes the BERT features plus a style vector and emits three things — a duration for each phoneme, an F0 (pitch) contour, and an energy curve. The durations drive a **length-regulation** step that expands the per-phoneme features into per-frame features. Finally the **decoder** — AdaIN residual blocks feeding an iSTFTNet head — turns those frames, the pitch, the energy, and the style into a 24 kHz waveform.

The amber box at the bottom-left is the part that makes Kokoro Kokoro: the **voicepack**, a frozen 256-dimensional style vector that is injected into both the prosody predictor and the decoder. There is no encoder producing that vector at runtime and no diffusion model sampling it. It is read from a table. Change the table entry and you change the voice; everything else in the graph is identical across all 54 voices.

Notice what is *not* in the diagram. There is no loop back from the output to the input — the defining feature of an autoregressive model. There is no diffusion sampler iterating over noise levels. There is no audio codec, no discrete acoustic tokens, no vector-quantized bottleneck. The entire computation is a directed acyclic graph evaluated exactly once per utterance, which is why latency is a function of *text length* and nothing else: a 10-word sentence and a 10-word sentence both take one pass, and a 100-word paragraph takes one (larger) pass. Compare that to an autoregressive model, where a 100-word paragraph means hundreds of sequential decoder steps, each one waiting on the last.

Everything below is a zoom into one of those boxes.

## Running Kokoro end to end

Before we dissect the parts, it helps to see the whole thing run, because the API is almost insultingly small — and that smallness is itself a consequence of the architecture. There is no streaming token loop to manage, no KV cache to size, no sampler temperature to tune. You hand it text and a voice name; it hands back a 24 kHz waveform.

```bash
pip install kokoro soundfile        # inference package + a wav writer
pip install "misaki[en]"            # English G2P with its embedded lexicon
```

The one external dependency worth calling out is `espeak-ng`, which misaki uses as its out-of-dictionary fallback; install it from your OS package manager (`apt install espeak-ng`, `brew install espeak-ng`). With that in place, a complete synthesis is six lines:

```python
import soundfile as sf
from kokoro import KPipeline

pipeline = KPipeline(lang_code='a')                 # 'a' = American English
text = "Kokoro generates this entire sentence in a single forward pass."

for i, (graphemes, phonemes, audio) in enumerate(pipeline(text, voice='af_heart')):
    print(phonemes)                                 # the exact G2P output
    sf.write(f'out_{i}.wav', audio, 24000)          # 24 kHz, always
```

Walk the control flow against the mental-model figure. `KPipeline` runs misaki to phonemize the text, splits it into chunks below the 510-phoneme cap on sentence boundaries, indexes the `af_heart` voicepack by each chunk's phoneme count, and runs one forward pass per chunk — yielding a `(graphemes, phonemes, audio)` triple as each chunk finishes. That the generator hands you the `phonemes` string for free is the exact instrumentation hook section 2 argued for: you can log what the model was actually asked to pronounce on every single request, at zero extra cost.

Two deployment facts fall out of the static, non-autoregressive graph. First, there is an **official ONNX export** (and a community `Kokoro-82M-ONNX`), and because the computation has no autoregressive loop and no dynamic control flow beyond the fixed length cap, it exports to a clean static graph that runs under `onnxruntime` on CPU, in the browser via WebAssembly, or on mobile — environments where serving an autoregressive token loop is genuinely painful. Second, the latency is flat and predictable, which makes capacity planning trivial:

| Output length | Autoregressive LLM-TTS | Kokoro (non-autoregressive) |
| --- | --- | --- |
| 1 sentence | N decoder steps, N waits | 1 forward pass |
| 1 paragraph | 5–10× the steps, serial | 1 pass per sub-chunk, batchable |
| Hardware floor | GPU for interactive use | CPU is enough for real time |
| Latency shape | grows with output length | flat in output length |

With the whole thing running in your head, the rest of the post is just opening each box and asking why it is built the way it is.

## 1. From StyleTTS 2 to Kokoro: what got cut

**Senior rule of thumb: when a model is suspiciously small, the interesting question is not what it contains but what its parent contained that it threw away.**

![From StyleTTS 2 to Kokoro: the style-diffusion sampler, the style encoders, and the SLM discriminator are training-only; Kokoro ships the predictors and decoder plus frozen voicepacks.](/imgs/blogs/kokoro-82m-2.webp)

StyleTTS 2 is the architecture Kokoro inherits, and it is worth understanding the full thing before we understand the subtraction. StyleTTS 2's headline contribution was **style diffusion**: instead of encoding a speaking style from reference audio, it trained a small diffusion model to *sample* a style vector conditioned on the text. At inference you draw a latent, run a few diffusion steps, and get a style appropriate to what is being said — different draws give different but plausible deliveries. That diffusion sampler, plus a pair of style encoders (one acoustic, one prosodic) that extract styles from reference audio during training, plus a **WavLM-based speech-language-model discriminator** used for adversarial training, are what give StyleTTS 2 its naturalness and its diversity.

Kokoro keeps the spine and drops the rest. The model card's own description is blunt: *"Decoder only: no diffusion, no encoder release."* Concretely:

- **The style diffusion sampler is gone.** Kokoro does not sample styles at runtime. Instead, the styles for its specific set of voices were computed once and frozen into voicepacks. This is the single most important architectural decision and the subject of section 3.
- **The two style encoders are gone from the shipped artifact.** They were needed during training to extract style targets from audio; at inference, with frozen styles, they serve no purpose.
- **The WavLM SLM discriminator is gone.** Discriminators are a training-time device. No GAN ever runs its discriminator at inference; that is true of every adversarial model. It shaped the weights and then retired.
- **What survives** is the part that actually generates sound: the ALBERT/BERT prosody encoder, the convolutional text encoder, the duration and prosody predictors, and the iSTFTNet decoder.

The before/after figure makes the correspondence explicit. The style diffusion sampler is replaced by frozen voicepacks. The style encoders are slimmed to just the text encoder that still runs. The discriminator is dropped entirely. The predictors and the decoder carry over unchanged. The net effect is a model that, at inference, is a strict subgraph of its parent — which is exactly why it fits in 82M parameters and 327 MB on disk.

There is a real cost to this subtraction, and it is the honest counterweight to all of Kokoro's wins: **Kokoro cannot produce a voice it was not given.** StyleTTS 2 can sample a fresh style for every utterance; Kokoro can only look one up. You can interpolate between two stored voicepacks (we will do exactly that later) but you cannot conjure a speaker that was never frozen into the table. If your application needs zero-shot cloning, Kokoro is the wrong branch of the family tree, and no amount of clever prompting will change that — the machinery that would sample a new style is simply not in the file.

Why make this trade? Because for the overwhelmingly common case — a fixed catalog of high-quality narrator voices — you do not *need* runtime style sampling. You need one good style per voice, computed carefully once. Freezing it removes the diffusion sampler from the latency budget, removes a source of run-to-run variance (frozen styles are deterministic), and removes a whole class of "the diffusion picked a weird delivery" failures. For an audiobook reader or a screen reader or a voice agent with a brand voice, determinism is a feature, not a regression.

## 2. Grapheme-to-phoneme: misaki, not espeak

**Senior rule of thumb: in a parametric TTS system the acoustic model is rarely the thing that mispronounces your text. The G2P front-end is.**

Kokoro does not consume text. It consumes **phonemes** — the atomic sound units of speech, written in the International Phonetic Alphabet. Converting "read" into the right phonemes (it depends on tense), expanding "Dr." into "doctor" or "drive" depending on context, and handling "2024" or "$3.50" is the job of a grapheme-to-phoneme (G2P) module that runs entirely *before* the neural network sees anything. Kokoro's G2P is a library called **misaki**, written by the same author.

This matters more than it sounds. Because the acoustic model only ever sees phonemes, every pronunciation decision is made by misaki, and every pronunciation *error* is misaki's, not the network's. If Kokoro says "lead" (the metal) when you meant "lead" (to guide), the weights did exactly what they were asked; the phoneme string was wrong. This is the central reason a parametric model's multilingual support lives in the front-end: adding a language is mostly a matter of adding a G2P for it, not retraining the acoustic model. Kokoro v1.0 jumped from 1 language to 8 largely by extending the phonemizer.

misaki is a deliberate upgrade over the older default in this lineage, **espeak-ng**. espeak-ng is a rule-based phonemizer: fast, tiny, multilingual, and frequently wrong on English, because English orthography is a war crime against rules. misaki's English path instead uses:

- **An embedded lexicon.** A large dictionary of words to their phoneme strings, self-contained — no external data files to ship or version. Known words get correct, hand-curated pronunciations.
- **A part-of-speech tagger for heteronyms.** Words like "read", "lead", "wind", "bass", "tear", "live", and "record" have different pronunciations depending on grammatical role. misaki runs an averaged-perceptron POS tagger and uses the tag to disambiguate. "I `read` (present) books" and "I `read` (past) a book" get different phonemes.
- **espeak-ng as a fallback.** For out-of-dictionary words — proper nouns, neologisms, technical jargon, code identifiers — misaki falls back to espeak's rule engine to produce *some* phonemization rather than failing.

That last point is the source of a whole category of production surprises, which we will get to in the case studies. The short version: words that fall through to espeak get espeak-quality pronunciation, which on unusual names or domain jargon can be noticeably off, and there is no signal in the audio telling you it happened. The fix is always to teach misaki the word, never to fight the acoustic model.

```python
from misaki import en                      # misaki English G2P, Kokoro's front-end

g2p = en.G2P(trf=False, british=False)     # American English, lightweight tagger

for text in ["I read books daily.", "I read that book yesterday.",
             "The bass guitar.",     "The bass was huge."]:
    phonemes, tokens = g2p(text)
    print(f"{text:32s} -> {phonemes}")      # the two 'read' lines differ; so do 'bass'
```

The two `read` lines come out as different phoneme strings, and so do the two `bass` lines — present-tense verb versus past-tense verb, the instrument versus the fish. That disambiguation is the part-of-speech tagger doing its job *before* a single tensor is allocated; the neural network never has the chance to get it wrong because it never sees the ambiguity. The same machinery handles "live" (verb vs adjective), "tear" (rip vs cry), "wind" (breeze vs coil), "record" (noun vs verb), and the dozens of other English heteronyms that a rule-based phonemizer flips a coin on.

The design lesson generalizes well beyond Kokoro. If you are building any parametric TTS product, **instrument your G2P**. Log the phoneme string for every request, not just the text. When a user reports "it mispronounced my name," you want to see the phonemes that were generated, because nine times out of ten the bug is one dictionary entry away from fixed, and zero times out of ten is it worth retraining a vocoder over.

## 3. The voicepack: a "voice" is a length-indexed lookup table

**Senior rule of thumb: when a model claims to have 54 voices in 327 MB, find out where the voices are actually stored. In Kokoro, they are not in the weights you think they are.**

![A voice is a length-indexed style table: ref_s = pack[len(phonemes)], whose first 128 dims drive the decoder and last 128 dims drive the prosody predictor.](/imgs/blogs/kokoro-82m-3.webp)

This is the part of Kokoro that most surprises engineers reading the inference code for the first time. A Kokoro "voice" is not a 256-dimensional vector. It is a **tensor of shape `[510, 256]`** — a table with one 256-dimensional style vector for *every possible phoneme-sequence length* from 1 up to 510. At inference, the pipeline counts the phonemes in your utterance and indexes the table by that count:

```python
ref_s = pack[len(ps) - 1]            # pack is [510, 256]; index by phoneme count
audio = model(ps, ref_s, speed)      # one forward pass for this chunk
```

In that snippet `pack` is the `[510, 256]` voicepack tensor for the chosen voice and `ps` is the phoneme string for the current chunk; the index `len(ps) - 1` selects the style row that was frozen for utterances of exactly this length.

Why would a style vector depend on how long the sentence is? Because prosody does. The natural pitch range, pacing, and energy of a two-word interjection are not the natural prosody of a forty-word subordinate clause. By storing a different style vector per length bucket, the voicepack bakes in a coarse length-conditioned prosody prior: short utterances get the style that was statistically right for short utterances, long ones get the long-utterance style. It is a cheap, frozen substitute for the length-awareness that an autoregressive model gets for free by attending over its own history.

The second surprise is how that 256-d vector is *used*, and it is a clean split. From the actual forward pass:

```python
s = ref_s[:, 128:]                                  # last 128 dims -> prosody
duration, F0_pred, N_pred = predict_prosody(s, ...) # predictor uses s
frames = length_regulate(asr, duration)             # expand to frame resolution
audio  = decoder(frames, F0_pred, N_pred, ref_s[:, :128])  # first 128 -> decoder
```

The 256-dimensional style vector is two concatenated 128-dimensional vectors with different jobs:

- **`ref_s[:, :128]` — the acoustic style — goes to the decoder.** It controls timbre, the grain of the voice, the AdaIN modulation in the vocoder. This is the part that makes a voice *sound like that person*.
- **`ref_s[:, 128:]` — the prosodic style — goes to the prosody predictor.** It controls how that voice paces itself: durations, pitch dynamics, energy. This is the part that makes a voice *deliver* a certain way.

This decoupling is inherited straight from StyleTTS 2, which deliberately trained separate acoustic and prosodic style encoders precisely so the two could be controlled independently. In Kokoro the two encoders are gone but their *output spaces* survive, glued together as the two halves of every voicepack row. The figure shows the whole flow: index the table by phoneme count to get a row, slice the row in half, route the acoustic half to the decoder and the prosodic half to the predictor.

### Blending voices is just averaging vectors

Because a voice is a vector, you can do arithmetic on voices. A 50/50 blend of two speakers is a 50/50 average of their voicepack rows. This is not a documented feature so much as an inevitable consequence of the representation:

```python
import torch
from kokoro import KModel, KPipeline

pipeline = KPipeline(lang_code='a')                 # American English
af = pipeline.load_voice('af_heart')                # shape [510, 256]
am = pipeline.load_voice('am_michael')

blend = 0.5 * af + 0.5 * am                          # a new, never-trained voice
for _, ps, audio in pipeline('Hello.', voice=blend): # use it like any voicepack
    pass
```

The blend is a genuinely new timbre and a new prosody, produced with zero training, because the style space the voicepacks live in is smooth enough that midpoints are valid voices. This is the *only* form of novel-voice creation Kokoro supports — convex combinations of existing voicepacks. You can interpolate; you cannot extrapolate to a speaker outside the convex hull of what was frozen. There is no inverse path from "here is 5 seconds of my voice" to a voicepack row, because the encoder that would compute it was never shipped.

### Second-order optimization: the length cliff and the 510-token wall

Two non-obvious consequences fall out of this design, and both bite in production.

First, the **length cliff**. Because the style vector changes with phoneme count, the *same voice* reading the *same words* can sound subtly different depending on how the text was chunked. Split a paragraph into sentences and each sentence indexes a different voicepack row; concatenate the audio and you may hear a faint discontinuity in pacing at the seams. The fix is to chunk on natural boundaries (sentences, clauses) where a small prosodic reset is acceptable anyway, not mid-clause.

Second, the **510-token wall**. The phoneme sequence is hard-capped because the ALBERT encoder has a fixed maximum position count. The actual assertion in the code is `len(input_ids) + 2 <= context_length`, where `context_length` equals the BERT's `max_position_embeddings` of 512, leaving 510 phoneme tokens after two special tokens. Phonemes, not characters — and English averages well under one phoneme per character, so 510 phonemes is roughly a long paragraph, but a dense technical sentence with lots of long words can hit it sooner than you expect. Past the wall you do not get a graceful truncation warning by default; you get a hard assertion or, in some wrappers, silently dropped audio. Production pipelines must split long text into chunks *below* the phoneme cap, which is exactly what `KPipeline` does when you let it.

## 4. Duration, alignment, and differentiable upsampling

**Senior rule of thumb: the hardest problem in non-autoregressive TTS is not making sound — it is deciding how many frames of sound each phoneme gets. Everything downstream depends on getting the alignment right.**

![Durations expand phonemes into frames: each phoneme owns a contiguous run of frames, so the soft training-time alignment collapses to a hard block-diagonal at inference.](/imgs/blogs/kokoro-82m-4.webp)

An autoregressive model never faces this problem. It emits frames one at a time and stops when it feels like stopping, so duration is implicit in the generation loop. A non-autoregressive model has no loop; it must decide, *before generating any audio*, exactly how long each phoneme lasts, because it will expand the phoneme sequence to frame resolution in a single shot. That decision is the **duration predictor**, and the expansion is **length regulation**.

The figure is the whole idea on a toy example: the word "hello" as five phonemes with predicted durations 2, 1, 3, 2, 2 frames. Each phoneme claims a contiguous run of frames — phoneme `/HH/` owns frames 1–2, `/AH/` owns frame 3, `/L/` owns frames 4–6, and so on. Stack those ownership rows and you get a **block-diagonal alignment matrix**: a hard, monotonic staircase where every phoneme maps to a solid horizontal band of frames and the bands never overlap or reorder. Monotonic, because speech does not skip back and forth in the phoneme sequence; block, because each phoneme is held for a stretch.

At inference the alignment is exactly that hard staircase: round each predicted duration to an integer, repeat each phoneme's encoded feature that many times, and you have a frame-resolution feature sequence ready for the decoder. Here is the actual mechanism, paraphrased from the forward pass:

```python
duration = torch.sigmoid(pred_durations).sum(dim=-1) / speed  # frames per phoneme
pred_dur = torch.round(duration).clamp(min=1).long()          # integer, >= 1

align = torch.zeros(input_len, int(pred_dur.sum()))           # phonemes x frames
c = 0
for i in range(input_len):
    align[i, c : c + pred_dur[i]] = 1   # phoneme i owns pred_dur[i] contiguous frames
    c += pred_dur[i]
frames = encoded_text @ align           # expand: [feat, phon] x [phon, frames]
```

Note the `/ speed` term: dividing every duration by a speed factor is exactly how Kokoro implements its speed control. `speed=1.3` shortens every phoneme by 30% and you get faster, higher-pitched-feeling speech with no resampling artifacts, because the change happens in *frame allocation*, not in post-hoc time-stretching. This is one of the quiet advantages of parametric TTS: speed is a first-class knob, not a DSP hack.

It is worth grounding the numbers. Kokoro's frames are mel frames at a hop of ~300 samples against the 24 kHz output, so each frame is roughly 11–12 milliseconds of audio. A phoneme with a predicted duration of 5 frames lasts about 55–60 ms; a stressed vowel that the predictor stretches to 12 frames lasts ~140 ms; a clipped consonant at the 1-frame floor lasts ~12 ms. The total number of output frames is simply `sum(pred_dur)`, and the output audio length in samples is `sum(pred_dur) * hop`. So when you set `speed=1.3`, you are dividing every per-phoneme frame count by 1.3 before rounding, the frame total drops by ~23%, and the waveform comes out ~23% shorter — with the pitch contour and timbre re-rendered at the new pacing rather than pitch-shifted by a resampler. That is why Kokoro's fast speech sounds like a person talking faster and not like a tape played at the wrong speed: the F0 predictor and decoder generate fresh audio for the compressed timeline instead of stretching existing samples.

The `clamp(min=1)` floor is load-bearing too. Without it, a phoneme the predictor judged near-zero duration would claim zero frames and vanish from the alignment matrix — a dropped sound, which is far more perceptually jarring than a slightly-too-long one. Forcing every phoneme to own at least one frame guarantees the monotonic staircase has no gaps and every phoneme is at least minimally pronounced, trading a rare over-articulation for never silently deleting a sound.

### Why "differentiable" duration is the whole trick

The hard staircase is fine at inference, but it is a disaster at *training time*, and this is where StyleTTS 2 made its key contribution. Rounding durations to integers and building a 0/1 alignment matrix is non-differentiable — `torch.round` has zero gradient almost everywhere, and the indexing that places the 1s is not differentiable at all. If you train through a hard alignment, no gradient flows from the audio loss back into the duration predictor. The predictor would have to be trained with a separate, weaker supervised duration loss and could never be tuned end-to-end against how the *audio* actually sounds.

StyleTTS 2's **differentiable duration modeling** fixes this by replacing the hard staircase with a soft one during training. Instead of a 0/1 matrix, it builds a continuous alignment by:

- Having the duration predictor output, for each phoneme $i$ and each integer step $k$, the probability that phoneme $i$ lasts at least $k$ frames — call it $q[k, i]$. Summing these gives an expected duration and, crucially, a *differentiable* one.
- Smearing each phoneme's frame ownership with a **Gaussian kernel** ($\sigma \approx 1.5$ frames) instead of a hard indicator, so the alignment matrix has soft edges that pass gradient.
- Normalizing across phonemes with a softmax so each frame's attention over phonemes sums to one: $a_{\text{pred}}[n, i] = e^{\tilde{f}[n,i]} / \sum_j e^{\tilde{f}[n,j]}$.

With a soft, differentiable alignment, gradients from the waveform reconstruction loss and the adversarial losses flow *all the way back into the duration predictor*. The predictor learns durations that make the final audio sound right, not just durations that match a forced-alignment label. That end-to-end signal is a large part of why StyleTTS 2 — and therefore Kokoro — has such natural rhythm. The staircase you see at inference is the hard limit of the soft alignment the model was trained with.

### Second-order optimization: durations are where "robotic" comes from

When a parametric TTS system sounds robotic, the duration predictor is the usual culprit, not the vocoder. Flat, mechanical pacing is under-varied durations; clipped or rushed word endings are under-predicted durations on final phonemes; the "TTS voice" cadence is durations regressing too hard to the mean. Because Kokoro's durations are partly conditioned on the prosodic half of the voicepack (`ref_s[:, 128:]`), a voice that sounds rhythmically off is often a voicepack issue rather than a fundamental model limit — which is why some community-fine-tuned voicepacks sound markedly more natural than others on the same weights.

## 5. The decoder: AdaIN, a harmonic source, and the iSTFT head

**Senior rule of thumb: a modern neural vocoder is a source-filter model wearing a GAN costume. Find the source and the filter and the rest is plumbing.**

![Inside the decoder: an F0-driven harmonic-plus-noise source feeds AdaIN ResBlocks carrying the 128-d acoustic style, and a conv head emits magnitude and phase for one iSTFT.](/imgs/blogs/kokoro-82m-6.webp)

The decoder is where frame-resolution features become a waveform, and it is the most signal-processing-heavy part of the model. The figure traces the dataflow. There are two inputs that matter beyond the aligned content features: the **F0 contour** (pitch) and the **acoustic style** `ref_s[:, :128]`. Let us follow each.

**The harmonic-plus-noise source.** Speech physics is a source-filter system: the vocal folds produce a periodic excitation (the source) at the fundamental frequency F0, and the vocal tract shapes it into vowels and consonants (the filter). StyleTTS 2's decoder makes this explicit with a **neural source filter (NSF)** module. The predicted F0 contour drives a bank of harmonic sine generators — sinusoids at F0, 2·F0, 3·F0, and so on — which are summed and mixed with a noise component for the unvoiced parts of speech (the `s` and `f` sounds that have no pitch). The result is a rich excitation signal that already has the right pitch and harmonic structure *before* the network adds timbre. This is why Kokoro's pitch is so stable: the periodicity is built from an explicit oscillator, not hallucinated by a convolution stack that has to learn to count.

**AdaIN ResBlocks carry the style.** The content features and the harmonic source flow into a stack of residual blocks, but these are not ordinary ResBlocks — they are **adaptive instance normalization (AdaIN)** blocks. AdaIN is the mechanism that injects the acoustic style. For each block, the 128-dimensional acoustic style vector is projected into a per-channel scale $\gamma$ and shift $\beta$, and the block's normalized activations are modulated as $\text{AdaIN}(x, s) = \gamma(s) \cdot \frac{x - \mu(x)}{\sigma(x)} + \beta(s)$. In plain terms: normalize each feature channel to zero mean and unit variance, then re-color it according to the style. Swap the style vector and the same content gets re-colored into a different voice. This is the exact same mechanism style-transfer networks use to repaint a photo in the texture of a painting; here it repaints phonetic content in the timbre of a speaker. It is also why timbre control is so clean in this architecture: the voice lives in the modulation parameters, fully separated from the content in the activations.

The reason to build the excitation from an explicit oscillator rather than learning it is a concrete failure mode of pure-ConvNet vocoders: pitch jitter and the "buzzy" artifact. A convolution stack asked to generate a 200 Hz periodic signal has to learn to emit a consistent period across thousands of samples, and small errors accumulate into wavering pitch and rough voicing — most audible on sustained vowels and held notes. By handing the network a clean harmonic source that is *already* at the right F0, the NSF module relieves it of inventing periodicity; the convolutions only have to shape spectral envelope and add the noise-like detail of fricatives, which they are good at. This is the classic source-filter separation from speech science, and bolting it onto a GAN vocoder is much of why this lineage has such steady pitch. The AdaIN modulation and the source operate on different aspects, too: the source sets *what frequencies are present* (driven by F0), while AdaIN sets *how they are colored* (driven by the acoustic style) — pitch and timbre controlled by separate, interpretable inputs rather than entangled in one black box.

**The iSTFT head.** After the ResBlocks, a final convolution produces not a waveform but a **spectral representation** — a magnitude spectrogram and a phase spectrogram — which an inverse short-time Fourier transform converts to audio. This is the iSTFTNet idea and it gets its own section next, because it is the single biggest reason Kokoro is fast. The activations are specific: magnitude is produced through an **exponential** (so it is always positive, as a magnitude must be), and phase is recovered from a sine/cosine pair via `atan2` (so it wraps correctly in $[-\pi, \pi]$). Then `torch.istft` does an overlap-add reconstruction back to 24 kHz samples.

```python
half   = n_fft // 2 + 1                                   # x: [2*half, frames]
spec   = torch.exp(x[:, :half])                          # magnitude > 0
phase  = x[:, half:]
real   = spec * torch.cos(phase)
imag   = spec * torch.sin(phase)
audio  = torch.istft(torch.complex(real, imag),
                    n_fft=n_fft, hop_length=hop, win_length=n_fft,
                    window=torch.hann_window(n_fft))     # overlap-add -> waveform
```

Here `x` is the final convolution output, with `n_fft//2 + 1` channels carrying the log-magnitude and another `n_fft//2 + 1` carrying the phase, so the exponential and the `atan2`-style sin/cos recovery turn that single tensor into a complex spectrum that `torch.istft` reconstructs.

### Second-order optimization: the float32 promotion

A subtle but real production detail: the iSTFT path is numerically sensitive. Magnitude-times-phase reconstruction followed by overlap-add accumulates error, and in half precision the phase term in particular can drift enough to produce audible artifacts. Practical Kokoro deployments **promote the signal to float32 at the vocoder output** even when the rest of the model runs in float16 — the precision matters exactly at the FFT boundary and almost nowhere else. If you are quantizing or half-precision-ing Kokoro for speed, keep the iSTFT in float32; the savings elsewhere are real and the savings there are a false economy that you will hear as a faint metallic ring.

## 6. iSTFTNet: stop upsampling, let the FFT finish the job

**Senior rule of thumb: do not ask a neural network to learn something a deterministic algorithm already does perfectly. The last octave of upsampling in a vocoder is an inverse FFT pretending to be a ConvNet.**

![iSTFTNet stops upsampling early: HiFi-GAN upsamples all the way to samples with transposed convs, while iSTFTNet predicts magnitude and phase and lets one iSTFT finish the reconstruction.](/imgs/blogs/kokoro-82m-5.webp)

To see why iSTFTNet is fast you have to know what it replaces. **HiFi-GAN**, the workhorse vocoder of the late 2010s and the direct ancestor here, takes an 80-bin mel spectrogram at frame resolution and upsamples it all the way to waveform resolution using a stack of **transposed convolutions**. To go from a frame every ~256 samples to one value per sample, it upsamples by factors that multiply to 256 (for example 8 × 8 × 2 × 2), with residual blocks at each stage. Those late upsampling stages are enormous: they operate at or near full audio resolution, on tens of thousands of time steps per second, and they dominate the vocoder's compute. (We have a full teardown of this in [HiFi-GAN](/blog/machine-learning/signal-processing/hifi-gan).)

iSTFTNet's insight, from the 2022 paper, is that a mel vocoder is secretly solving three inverse problems at once: recover the full-resolution magnitude spectrum, recover the phase, and convert frequency back to time. A pure-ConvNet vocoder solves all three implicitly and jointly with its upsampling stack — which means it is spending a large fraction of its compute learning to approximate an *inverse Fourier transform*, a transform we have a closed-form, exact, blazingly fast algorithm for.

So iSTFTNet stops upsampling early. It runs a **reduced** transposed-conv stack — enough to get from mel resolution to a coarse STFT frame resolution, but not all the way to samples — and then changes the final layer to output `(n_fft/2 + 1) × 2` channels: a magnitude spectrogram and a phase spectrogram. An exponential activation makes the magnitudes positive; a sine activation shapes the phase. Then a single **inverse short-time Fourier transform** does the frequency-to-time conversion and the final upsampling in one deterministic, non-learned step. The before/after figure lines the two up: HiFi-GAN's tall stack of transposed-conv upsamplers versus iSTFTNet's short stack plus a spectral head plus one iSTFT.

The payoff is exactly the layers it deletes. The most expensive upsampling stages — the ones running at full sample resolution — are replaced by `torch.istft`, which is a couple of FFTs and an overlap-add. The original paper reported large speedups across HiFi-GAN variants with comparable quality, and in Kokoro this is the structural reason an 82M model vocodes faster than real time on a CPU. The FFT is doing the work the transposed convolutions used to fake.

| Vocoder stage | HiFi-GAN | iSTFTNet (Kokoro) |
| --- | --- | --- |
| Input | 80-bin mel | 80-bin mel |
| Upsampling | full stack, ~256× to samples | reduced stack to STFT frames |
| Most expensive layers | transposed convs at sample rate | none — replaced |
| Final stage | 1-channel waveform conv | `(n_fft/2+1)×2` mag+phase, then iSTFT |
| Frequency→time | learned implicitly | exact, via inverse FFT |
| Relative cost | baseline | substantially lower |

It helps to see where the compute actually goes. A transposed convolution that upsamples by a factor of $r$ at a layer with $C$ channels and $T$ time steps costs on the order of $C^2 \cdot k \cdot rT$ multiply-accumulates, where $k$ is the kernel size — and crucially, the $T$ at the *last* upsampling layer is already at or near the full sample rate, so that layer alone runs over tens of thousands of time steps per second of audio. iSTFTNet deletes exactly those layers. By stopping at a coarse STFT frame rate and handing off to the inverse FFT — which is $O(N \log N)$ per frame for an `n_fft` of size $N$, a few hundred operations rather than a convolution over the full waveform — it removes the most expensive part of the compute graph and replaces it with one of the most optimized routines in all of numerical computing. The FFT in `torch.istft` is the same battle-hardened code that powers every spectrogram, every audio codec, and every software-defined radio on the planet; asking a transposed convolution to approximate it was always the inefficiency.

The exact STFT configuration matters for both correctness and quality, and it is small:

| STFT parameter | Typical Kokoro / iSTFTNet value | Why it is set there |
| --- | --- | --- |
| `n_fft` | small (e.g. 20–40 for the post-conv head) | keeps the phase prediction problem low-dimensional |
| `hop_length` | matches the residual upsample factor | so frames tile the waveform without gaps |
| `win_length` | equals `n_fft` | clean Hann overlap-add reconstruction |
| window | Hann | the partition-of-unity property the iSTFT assumes |
| output rate | 24 kHz | Kokoro's fixed sample rate |

There is a tradeoff, and it is the standard one for spectral vocoders: phase. Predicting phase directly is harder than predicting magnitude, and a model that gets the phase slightly wrong produces the characteristic "phasey" or metallic vocoder artifacts. iSTFTNet keeps phase errors in check by only asking the network for a *coarse* STFT (the reduced stack means a smaller `n_fft` and fewer frequency bins to get right) and letting the iSTFT's overlap-add average out residual phase error across hops. This is the same reason the float32 promotion at the head matters: phase is the fragile quantity, and it is fragile precisely because it is the part the deterministic FFT cannot fix for you.

## 7. How these weights were trained: the loss stack

**Senior rule of thumb: you cannot understand what a generative audio model can and cannot do until you know which losses pulled on which parameters. The capability map is the loss map.**

![The losses that shaped these weights: seven StyleTTS 2 objectives across two stages, each supervising a specific module from mel reconstruction to the WavLM adversarial signal.](/imgs/blogs/kokoro-82m-7.webp)

Here is the honesty caveat in full force: there is no published Kokoro training recipe, no Kokoro loss table, no Kokoro paper. What follows is **StyleTTS 2's** training procedure, which is the procedure that produced this architecture and the closest faithful account we have of how weights of this shape are made. Kokoro confirms two deviations — it dropped the diffusion sampler from the shipped model and it froze styles into voicepacks — but the reconstruction, adversarial, prosody, and alignment machinery is inherited. Read this section as "how a StyleTTS 2 decoder is trained," because that is what Kokoro's decoder is.

StyleTTS 2 trains in **two stages**, and the loss figure organizes the seven objectives by what they supervise and when they run.

**Stage 1 — acoustic pre-training.** The text encoder, the style encoders, the duration/alignment machinery, and the decoder are trained together on reconstruction and adversarial objectives, with the pitch extractor providing ground-truth F0 labels. The losses active here:

- **$\mathcal{L}_{\text{mel}}$ — mel reconstruction.** An L1 loss between the mel spectrogram of the generated waveform and the mel of the ground-truth audio, using the predicted pitch and energy. This is the workhorse that teaches the decoder to make sound that matches the target. It runs in both stages.
- **$\mathcal{L}_{\text{adv}} + \mathcal{L}_{\text{fm}}$ — decoder adversarial + feature matching.** A **least-squares GAN (LSGAN)** loss against two discriminators — a **multi-period discriminator (MPD)** and a **multi-resolution discriminator (MRD)** — that judge the waveform at multiple periodicities and STFT resolutions, plus a feature-matching loss that aligns the generator's discriminator-internal features to real audio's. This is the HiFi-GAN adversarial recipe and it is what removes the buzzy, over-smoothed quality that pure-L1 vocoders have. Both stages.
- **$\mathcal{L}_{\text{mono}}$ — monotonic alignment (TMA).** The transferable monotonic aligner objective that learns the text-to-speech alignment without external forced-alignment labels, enforcing the monotonic block structure we saw in section 4. Stage 1.

**Stage 2 — joint training.** Everything except the pitch extractor is optimized jointly, now end-to-end through the differentiable duration model:

- **$\mathcal{L}_{\text{dur}}$ — duration.** L1 between predicted and ground-truth phoneme durations. With differentiable upsampling, this is augmented by the gradient that now flows from the audio losses into the duration predictor. Stage 2.
- **$\mathcal{L}_{f0} + \mathcal{L}_{n}$ — pitch and energy.** L1 losses matching the predicted F0 contour and the predicted energy curve to values extracted from ground-truth audio. These train the prosody predictor. Stage 2.
- **$\mathcal{L}_{\text{slm}}$ — SLM adversarial.** The signature StyleTTS 2 loss: a **speech-language-model discriminator** built on a frozen, pre-trained **WavLM** (12 layers, trained on ~94k hours of speech) judges whether generated speech is human, with a minimax objective $\min_G \max_{D} \, \mathbb{E}_x[\log D(x)] + \mathbb{E}_t[\log(1 - D(G(t)))]$. Because WavLM has heard an enormous amount of real speech, it is a far more discerning critic of naturalness than a discriminator trained from scratch, and training the generator to fool it is much of what pushes StyleTTS 2 to human-level MOS. This is the loss that *requires* differentiable duration — you cannot backprop the SLM signal through a hard alignment. Stage 2.
- **$\mathcal{L}_{\text{diff}}$ — style diffusion (cut at inference).** A denoising score-matching / EDM-style objective $\mathcal{L}_{\text{edm}} = \lambda(\sigma)\,\lVert K(E(x)+\sigma\xi;\,t,\sigma) - E(x)\rVert_2^2$ that trains the 3-layer style-diffusion transformer to sample style vectors from text. This is the part Kokoro keeps for *training* (to learn a good style space) but does **not** ship — the styles it produces were frozen into voicepacks instead of being sampled at runtime. Stage 2.

![Two-stage training: the acoustic stack, mel reconstruction, MPD/MRD adversarial, and monotonic alignment come first; differentiable duration, pitch/energy, the WavLM SLM adversarial, and style diffusion come second.](/imgs/blogs/kokoro-82m-8.webp)

The timeline figure orders these by stage. The thing to take away is the *dependency*: the SLM adversarial loss (the naturalness engine) and the differentiable duration model (the rhythm engine) are both stage-2, both end-to-end, and both depend on stage-1 having already produced a competent acoustic model and alignment. You cannot start with the fancy losses; you have to earn them with reconstruction first.

Collected in one place, the loss stack reads as a map of which gradient pulls on which parameters:

| Loss | Type | Trains | Stage | Ships in Kokoro? |
| --- | --- | --- | --- | --- |
| $\mathcal{L}_{\text{mel}}$ | L1 reconstruction | decoder | 1 + 2 | yes (weights) |
| $\mathcal{L}_{\text{adv}} + \mathcal{L}_{\text{fm}}$ | LSGAN + feature match (MPD/MRD) | decoder | 1 + 2 | yes (weights) |
| $\mathcal{L}_{\text{mono}}$ | monotonic alignment (TMA) | aligner | 1 | yes (weights) |
| $\mathcal{L}_{\text{dur}}$ | L1 + end-to-end gradient | duration predictor | 2 | yes (weights) |
| $\mathcal{L}_{f0} + \mathcal{L}_{n}$ | L1 | prosody predictor | 2 | yes (weights) |
| $\mathcal{L}_{\text{slm}}$ | WavLM adversarial (minimax) | generator | 2 | yes (weights) |
| $\mathcal{L}_{\text{diff}}$ | EDM score-matching | style diffusion | 2 | trained, **not** shipped |

"Ships in Kokoro" means the parameters that loss trained are present in the released file. Every loss except the diffusion objective shaped weights you actually run. The diffusion loss is the odd one out: it trained the style space well enough that the frozen voicepacks sampled from it are good, but the sampler itself stayed behind. That single row is the whole "decoder only, no diffusion" decision expressed as a loss-table footnote.

Why does the WavLM discriminator matter so much that it is worth singling out? Because a from-scratch discriminator only knows what "real" sounds like from the training set it shares with the generator, so the two can collude into a comfortable local optimum that sounds plausible but synthetic. WavLM was pre-trained self-supervised on ~94k hours of diverse real speech — far more than the TTS model's few hundred hours — so it carries a model of human speech the generator never had access to. Training the generator to fool that external critic injects a naturalness prior that no reconstruction loss can supply, because reconstruction only ever asks "does this match *this* recording," while the SLM asks "does this sound like *a human at all*." That distinction is much of the gap between "good TTS" and "I cannot tell it is TTS," and it is the reason the differentiable-duration plumbing exists: the SLM gradient has to reach the duration predictor for the rhythm to become human, and it cannot reach through a hard alignment.

### The data story, and what little we know of it

Kokoro's training data is the most-asked and least-answered question about the model, so here is exactly what the card says and nothing more:

- **Volume:** "A few hundred hours" of audio. For context, that is *tiny* by modern TTS standards — XTTS and the LLM-TTS systems train on tens of thousands of hours. The card elsewhere says Kokoro v1.0 was trained on under 100 hours per the public discussion, in fewer than 20 epochs.
- **Provenance:** "Permissive/non-copyrighted audio data" — public-domain audio, Apache/MIT-licensed audio, and **synthetic audio generated by closed commercial TTS models**. The named CC-BY datasets are Koniwa (<1 hour) and SIWIS (<11 hours).
- **Explicitly excluded:** Synthetic audio from *open* TTS models, and any custom voice clones. This is a deliberate, defensible data-provenance stance: the author leaned on closed commercial TTS as a teacher (a form of distillation) and permissively-licensed real audio, and avoided the licensing minefield of scraped or cloned voices.
- **Compute:** "About \$1000 for 1000 hours of A100 80GB vRAM," with v1.0 specifically around 500 A100-hours at roughly \$1/hour. That is the entire training budget. Not a typo — a few hundred dollars of GPU time produced a model that competes with systems that cost orders of magnitude more.
- **License:** Apache-2.0 weights. Commercial use, no strings.

The data story is the real headline if you care about reproducibility and ethics. Kokoro is a strong argument that, for clean read-speech TTS, **data quality and architecture matter more than data quantity**. A few hundred well-chosen, well-licensed hours through a well-designed parametric architecture beat tens of thousands of scraped hours through a brute-force autoregressive one — at least on the axis of "clean narrator voice," which is the axis most products actually need.

| Property | Kokoro v0.19 | Kokoro v1.0 |
| --- | --- | --- |
| Languages | 1 | 8 |
| Voices | 10 | 54 |
| G2P | espeak-heavy | misaki (espeak fallback) |
| Architecture | StyleTTS 2 decoder + iSTFTNet | same |
| Sample rate | 24 kHz | 24 kHz |
| License | Apache-2.0 | Apache-2.0 |

## 8. Where Kokoro sits in the TTS landscape

**Senior rule of thumb: there is no best TTS model, only a best model for a latency budget, a controllability requirement, and a licensing constraint. Pick the axis you actually care about.**

![Where Kokoro sits: against autoregressive LLM-TTS it wins on parameters, latency, and training cost, but it cannot sample unseen voices the way zero-shot cloning systems can.](/imgs/blogs/kokoro-82m-9.webp)

The positioning matrix lays out the tradeoff against the systems Kokoro is most often compared to. Read it as four axes, not one score.

- **Parameters.** Kokoro is 82M. XTTS-v2 is ~470M. Orpheus is built on a 3B LLM. Tortoise is roughly a billion across its stack. On pure footprint, Kokoro is in a different weight class — it fits comfortably on a phone, in a browser via ONNX/WASM, or on a CPU-only server.
- **Latency.** Kokoro is non-autoregressive: one forward pass, latency flat in output length, faster than real time on CPU. The autoregressive systems (Orpheus, Tortoise) generate audio tokens sequentially, so latency grows with output length and effectively requires a GPU for interactive use. XTTS sits in between. (Our deep-dive on [Orpheus TTS](/blog/machine-learning/signal-processing/orpheus-tts-llm-speech-snac) walks through exactly why the LLM-codec approach pays that latency tax.)
- **Voice control.** This is the axis where Kokoro *loses*. Its voices are fixed voicepacks; it cannot clone a voice from reference audio. XTTS, Orpheus, and Tortoise all do zero-shot cloning from a few seconds of audio. If "make it sound like *this specific person*" is the requirement, Kokoro is disqualified.
- **Training cost.** Kokoro cost ~\$1k. The others cost orders of magnitude more in compute and data. For anyone who wants to *fine-tune* or *retrain* rather than just run, this is decisive: you can fine-tune Kokoro on a single GPU in an afternoon.

The honest summary is that Kokoro is not a general-purpose TTS model that happens to be small. It is a **specialist**: a fixed-catalog, non-autoregressive, low-latency, permissively-licensed read-speech engine. On its specialty it is close to unbeatable on cost and latency. Off its specialty — voice cloning, wildly expressive or emotional delivery, conversational backchannels — it is the wrong tool, and the systems that beat it there pay for the privilege in size, latency, and licensing.

There is also a strategic point hiding in the four axes. Most production TTS workloads are not the demo workloads that drive model hype. The demo that gets attention is "clone any voice from three seconds," but the workload that pays the bills is "read this fixed catalog of strings in our brand voice, cheaply, at scale, on commodity hardware, with predictable latency and clean licensing." Kokoro is built for the second list, and the second list is enormous: screen readers, audiobook production, IVR and voice agents, e-learning narration, accessibility tooling, on-device assistants. For that majority of real work, the question is not "is Kokoro as capable as a 3B LLM-TTS" — it plainly is not — but "do I need any of the capability I would be paying for," and the answer is usually no. Choosing the specialist when you only need the specialty is not settling; it is the whole game.

## Case studies and engineering notes from running it

These are the situations where Kokoro's design decisions show up as real behavior in production. Each is a concrete consequence of something in the sections above.

### 1. The voicepack length cliff at chunk seams

**Symptom.** A team building an audiobook pipeline split chapters into sentences, synthesized each independently, and concatenated. Listeners reported a faint "unevenness" — the pacing seemed to micro-reset between some sentences. **Wrong first hypothesis.** They assumed the decoder was non-deterministic and chased a seed bug. **Actual root cause.** Each sentence had a different phoneme count and therefore indexed a different row of the `[510, 256]` voicepack, drawing a slightly different prosodic style (the length-conditioned prior from section 3). Short sentences got the short-utterance style; long ones got the long-utterance style; at the seam the two styles abut. **Fix.** Chunk on sentence boundaries (where a small prosodic reset is natural anyway) and add a few hundred milliseconds of silence at seams so the ear reads the reset as a normal pause. **Lesson.** In Kokoro, *how you chunk text is a prosody decision*, not just an engineering one, because the voice vector is length-indexed.

### 2. espeak fallback drift on proper nouns

**Symptom.** A voice agent for a fintech product mispronounced several company and product names, and the errors were inconsistent across similar words. **Wrong first hypothesis.** "The model is bad at names; we need a bigger model." **Actual root cause.** The names were out-of-dictionary for misaki and fell through to the **espeak-ng fallback**, whose rule-based English phonemization mangles unusual orthography. The acoustic model rendered the espeak phonemes faithfully — garbage in, garbage out — and there was no error signal because the fallback succeeds silently. **Fix.** Add the names to misaki's lexicon (or pass explicit phonemes for known-hard words) so they never reach espeak. **Lesson.** When a parametric TTS mispronounces, log and inspect the *phoneme string*, not the text. The bug is almost always one G2P entry away, and retraining the acoustic model would not have fixed a single one of these names.

### 3. Real-time on a CPU, and the float32 boundary

**Symptom.** A team shipped Kokoro on CPU-only edge servers for a privacy-sensitive on-device use case and saw it run faster than real time — but a half-precision experiment to squeeze more throughput introduced a faint metallic ring. **Root cause.** The iSTFT head is numerically sensitive (section 5). The magnitude/phase reconstruction and overlap-add accumulate error that float16 cannot hold cleanly, and the artifact lands in the phase term. **Fix.** Keep the model in the precision that is fast for the platform but **promote to float32 at the vocoder output** before the iSTFT. **Lesson.** Quantization in audio models is not uniform — the FFT boundary is a precision cliff. The non-autoregressive design is what makes CPU real-time possible in the first place; there is no sequential decode loop to serialize, so the whole utterance vocodes in one vectorized shot.

### 4. Garbled or truncated tails on long input

**Symptom.** Long paragraphs occasionally produced audio that degraded or cut off near the end. **Wrong first hypothesis.** "The model loses coherence on long text, like an LLM running out of context." **Actual root cause.** The **510-phoneme wall** (section 3). A dense paragraph exceeded the phoneme cap; depending on the wrapper, the model either hit the `len(input_ids)+2 <= 512` assertion or silently dropped the overflow. **Fix.** Use `KPipeline`'s built-in splitting, which chunks text below the phoneme cap on natural boundaries and stitches the audio. Never feed raw arbitrary-length text to `KModel` directly. **Lesson.** The cap is a hard architectural limit from the ALBERT position embeddings, not a soft quality degradation. Treat it like a buffer size: respect it explicitly.

### 5. Fine-tuning to a new language — the decoder transfers, the G2P does not

**Symptom.** A community effort to bring Kokoro to German (and later Marathi) found that the acoustic quality transferred quickly with modest fine-tuning, but getting *correct pronunciation* was the long pole. **Root cause.** The decoder and prosody machinery are largely language-agnostic — they operate on phonemes and styles, not orthography — so a few hours of target-language audio adapts them. But the **G2P is entirely language-specific**: German compound nouns, Marathi's Devanagari script, and the phoneme inventories differ completely from English, and misaki's English lexicon and POS tagger are useless for them. **Fix.** The bulk of the work in these recipes is building or wiring a correct target-language phonemizer; the neural fine-tune is comparatively cheap. **Lesson.** This confirms the section-2 thesis from the other direction: in a parametric system, *the language lives in the front-end*. Want a new language? Budget your time for the phonemizer, not the GPU.

### 6. "No diffusion" means no novel voices, only interpolation

**Symptom.** A product wanted a "custom brand voice" and assumed they could prompt or fine-tune Kokoro into a specific new persona from a short clip. **Wrong first hypothesis.** "It's a neural TTS, surely it can clone." **Actual root cause.** Kokoro ships **no style encoder and no diffusion sampler** (section 1), so there is no path from reference audio to a voicepack. The only ways to get a "new" voice are: (a) interpolate existing voicepacks, which stays inside the convex hull of what was frozen, or (b) actually fine-tune the weights on new audio and compute fresh voicepacks. **Fix.** They blended two existing voicepacks to land on an acceptable brand-adjacent timbre, then later did a proper fine-tune for the real custom voice. **Lesson.** The voicepack representation is a *lookup*, not an *encoder*. Interpolation is free; extrapolation requires training.

### 7. iSTFT window/hop mismatches and clicks

**Symptom.** A reimplementation of the Kokoro inference path in another runtime produced periodic clicks. **Root cause.** The iSTFT (section 6) is only artifact-free if its `n_fft`, `hop_length`, `win_length`, and window function exactly match what the model was trained with. A mismatched hop or a rectangular-instead-of-Hann window breaks the overlap-add reconstruction, and the constructive/destructive interference at frame boundaries shows up as a click at the hop period. **Fix.** Mirror the training STFT parameters exactly in the inference iSTFT; do not "tidy up" the window or round the hop. **Lesson.** The deterministic FFT half of the vocoder is only deterministic if you feed it the parameters it expects. Porting a spectral vocoder is mostly about getting the STFT config byte-for-byte right.

### 8. Synthetic-data distillation and the provenance ceiling

**Symptom.** A team evaluating Kokoro for a regulated domain asked whether its outputs were "safe" to use given the training data. **Root cause / nuance.** Kokoro's training set includes **synthetic audio from closed commercial TTS models** (section 7). That is a deliberate distillation choice and it is part of why a few hundred hours sufficed — the teacher models supplied clean, consistent targets. But it also means Kokoro's voice characteristics partly reflect those teachers, and the relevant terms-of-service and provenance questions are worth a real legal read for high-stakes use, even though the *weights* are Apache-2.0. **Fix / stance.** For most uses the Apache-2.0 weights and permissive real-audio components are clean; for regulated or high-liability deployments, document the data provenance the card describes and get sign-off. **Lesson.** "Apache-2.0 weights" answers the *licensing of the model*; it does not by itself answer every *training-data provenance* question. Read the card, which is unusually candid about exactly this.

### 9. Throughput from bucketed batching

**Symptom.** A TTS API backed by Kokoro served thousands of short prompts per minute (UI strings, notifications, chat replies) and was leaving GPU utilization on the floor — each request ran one at a time at a fraction of the card's capacity. **Wrong first hypothesis.** "We need more replicas." **Actual root cause.** The deployment treated Kokoro like an autoregressive model and serialized requests, but Kokoro has no decode loop, so there is nothing forcing one-at-a-time execution. **Fix.** Bucket pending requests by phoneme length, pad each bucket to its max length, and run the bucket as a single batched forward pass. Because every utterance in a bucket takes the same one pass and latency is flat in length, batching is nearly free — the only waste is the padding of the shorter utterances, which bucketing minimizes. Throughput jumped several-fold on the same hardware. **Lesson.** Non-autoregressive TTS batches like a vision model, not like an LLM. If you are serving Kokoro one request at a time, you are leaving most of your compute unused; the flat-latency property that makes single requests fast is the same property that makes batching trivial.

### 10. The 24 kHz sample-rate trap

**Symptom.** Kokoro audio piped into a telephony stack came out sounding chipmunk-fast; piped into a 48 kHz video pipeline it came out slow and dull. **Wrong first hypothesis.** "The model's pitch is wrong." **Actual root cause.** Kokoro **always** outputs 24 kHz, because the sample rate is baked into the iSTFT configuration (section 6) — it is not a per-request option. The downstream stacks assumed 8 kHz and 48 kHz respectively and reinterpreted the same samples at the wrong rate, which is a pure pitch/tempo shift. **Fix.** Resample explicitly at the boundary — `24000 -> 8000` for telephony, `24000 -> 48000` for video — with a real resampler (`soxr`, `librosa.resample`, or `torchaudio`), never by relabeling the rate. **Lesson.** A vocoder's output rate is a property of its STFT config, not a knob. Every Kokoro integration needs an explicit, audited resample step at the point where its 24 kHz audio meets a system that expects something else; "it sounded fine in my test" usually means your test player happened to assume 24 kHz too.

## When to reach for Kokoro, and when not to

Kokoro is a sharp tool with a sharp edge. The decision is rarely about quality in the abstract; it is about whether your requirements line up with the four axes from section 8.

**Reach for Kokoro when:**

- You need a **fixed catalog of high-quality narrator voices** — audiobooks, screen readers, IVR, e-learning, voice agents with a brand voice — and not arbitrary cloning.
- **Latency and throughput matter** and you want flat, predictable latency. One forward pass, faster than real time on CPU, trivially batchable.
- You are **CPU-bound, edge-bound, or browser-bound.** 82M parameters and an iSTFT vocoder run where a 3B autoregressive model cannot.
- You need **permissive licensing** for a commercial product. Apache-2.0 weights, no usage strings.
- You want a model you can **actually fine-tune** on a single GPU for a new voice or language without a data-center budget.
- **Determinism is a feature.** Frozen styles mean the same text gives the same audio every time — important for testing, caching, and regulated workflows.

**Skip Kokoro when:**

- You need **zero-shot voice cloning** from a few seconds of reference audio. The encoder that would do this is not in the model. Use XTTS, Orpheus, or a dedicated cloning system.
- You need **highly expressive or emotional delivery** that varies per utterance — laughter, shouting, crying, dramatic range. Frozen styles and a deterministic predictor give consistent, natural *read* speech, not theatrical range.
- You need **conversational paralinguistics** — backchannels, disfluencies, "um"s and "uh"s placed naturally. That is the home turf of LLM-TTS systems (see [Orpheus](/blog/machine-learning/signal-processing/orpheus-tts-llm-speech-snac)).
- Your target **language has no good phonemizer.** Without a correct G2P, Kokoro mispronounces fluently and confidently. Budget for the front-end first.
- You need **singing, or precise musical pitch control.** The F0 predictor is tuned for speech prosody, not melody.

The meta-lesson is the one worth keeping after the implementation details fade: Kokoro is a master class in **subtraction**. It took a human-level architecture, identified every component that existed only to serve runtime flexibility it did not need — the diffusion sampler, the style encoders, the discriminator — and deleted them, freezing the one thing it did need (a good style per voice) into a lookup table. What remained was small enough to run anywhere and fast enough to run in real time, and it gave up exactly the capabilities (cloning, open-ended expressivity) that the deleted components provided. That is not a smaller version of a big model. It is a different, sharper answer to the question "what does a TTS system for a fixed set of voices actually need?" — and for an enormous fraction of real products, the answer is "a lot less than we have been shipping."

## Further reading

- **StyleTTS 2** — the parent architecture, with the style diffusion, differentiable duration, and SLM adversarial training Kokoro inherits: [arXiv:2306.07691](https://arxiv.org/abs/2306.07691).
- **iSTFTNet** — the fast spectral vocoder that does the heavy lifting: [arXiv:2203.02395](https://arxiv.org/abs/2203.02395).
- **HiFi-GAN, in depth** — the vocoder iSTFTNet modifies, and the source of the MPD/MRD adversarial recipe: [HiFi-GAN](/blog/machine-learning/signal-processing/hifi-gan).
- **Orpheus TTS** — the autoregressive LLM-plus-codec approach Kokoro contrasts with, and why it pays a latency tax: [Orpheus TTS](/blog/machine-learning/signal-processing/orpheus-tts-llm-speech-snac).
- **Training CosyVoice** — another modern TTS training pipeline, for comparison on data and objectives: [Training CosyVoice](/blog/machine-learning/deep-learning/training-cosyvoice).
- **misaki** and **kokoro** on GitHub (`hexgrad/misaki`, `hexgrad/kokoro`) — the actual G2P and inference code referenced throughout.
