---
title: "Speech Data for ASR: Turning Long Audio Into Aligned Training Pairs"
date: "2026-06-30"
publishDate: "2026-06-30"
description: "How ASR training data is really made — audio-transcript pairs, the axes that decide label cost, where the audio comes from, and the segmentation-plus-forced-alignment machinery that manufactures utterance-level pairs from a long recording and a book."
tags:
  - training-data
  - speech-recognition
  - asr
  - forced-alignment
  - voice-activity-detection
  - montreal-forced-aligner
  - librispeech
  - common-voice
  - audio-preprocessing
  - ctc-segmentation
category: "machine-learning"
subcategory: "Training Data"
author: "Hiep Tran"
featured: true
readTime: 33
---

The first time you try to train an acoustic model on "some audiobooks I downloaded," you discover that speech data does not arrive in the shape your model wants. Your model wants short clips — a few seconds each — paired with the exact words spoken in that clip. What you have is a 38-minute MP3 of a narrator reading a Victorian novel, and separately, the full text of that novel from Project Gutenberg. Nowhere is there a file that says "seconds 41.2 to 45.6 contain the words *it was the best of times*." That mapping does not exist yet. You have to manufacture it.

That manufacturing step — taking long recordings plus a known transcript and cutting them into aligned, utterance-level `(audio, text)` pairs — is the single most important, least discussed part of building ASR data. Everyone talks about model architectures and decoding; almost nobody explains where the training pairs come from, and the ones who do usually wave at "forced alignment" as if it were a solved black box. It is not a black box. It is a pipeline with specific failure modes, specific tools, and specific numbers you should expect, and getting it wrong quietly poisons your training set with mislabeled clips that no loss curve will ever flag.

![The end-to-end pipeline that turns a long recording and its book into thousands of short aligned training pairs](/imgs/blogs/speech-data-for-asr-2.webp)

The diagram above is the mental model for this entire post, and the rest of the article is a tour of it. You start with a long recording and a known transcript. You standardize the audio to 16 kHz mono. You run voice-activity detection to find the speech and throw away the silence, music, and applause. You run forced alignment to snap the known words onto the audio timeline. You cut on the resulting word boundaries. You filter out the segments the aligner was not confident about. What falls out the right side is a pile of clean `(audio, text)` pairs — the thing your model actually trains on. Miss any stage and the pile is contaminated.

This post is written for engineers who will build or debug an ASR data pipeline: the person who has to explain why the model transcribes the LibriVox intro boilerplate onto every clip, or why word error rate on their in-house data is triple what the leaderboard promised. We build intuition first, then the math of alignment, then real code with `torchaudio`, Silero VAD, the Montreal Forced Aligner, and `webdataset`, then a worked scenario with concrete segment counts, and finally a troubleshooting section and a set of named case studies. If you want the model side of the story, the companion post on [how Whisper handles audio](/blog/machine-learning/signal-processing/whisper-under-the-hood) covers the encoder; here we stay resolutely on the data.

## Why speech data is a different problem

Before the pipeline, the mismatch — because "just point the trainer at the audiobooks" is exactly the assumption that produces a broken model, and a staff engineer needs the mechanism, not the slogan.

| Assumption (from the text-data world) | Naive view | Reality for ASR |
| --- | --- | --- |
| Data is already `(input, label)` pairs | Download the corpus and train | You get long audio and a separate transcript; the pairing is something *you* build |
| The transcript is the label | The book text is ground truth | The book text describes what *should* be said; the aligner must find *where* and drop what is not there |
| Any sample rate is fine | Audio is audio | A 44.1 kHz clip fed to a 16 kHz model silently halves every frequency — a hard, invisible bug |
| Silence and music are harmless | The model learns to ignore them | Un-transcribed non-speech breaks alignment and teaches the model to hallucinate words over noise |
| More hours is always better | Scale fixes everything | A smaller, tightly aligned, confidence-filtered set beats a larger sloppily-cut one at the same compute |

The through-line is that ASR supervision is *manufactured*, not *found*. In text pretraining you scrape documents and the document *is* the training example — the [sourcing and collecting](/blog/machine-learning/training-data/sourcing-and-collecting-training-data) post is about finding good documents, full stop. In ASR you scrape recordings, and each recording is raw material that must be machined into thousands of examples. The machining tools — VAD and forced alignment — are where most of the quality is won or lost, and they have no equivalent in the text world. That single fact reorganizes the whole data pipeline around segmentation and alignment rather than mere collection.

There is a second difference that bites later. A text model's loss will spike on garbage because the garbage is *visible* in the token stream. An ASR model trained on a misaligned clip sees a plausible waveform and a plausible transcript; the loss is finite and the gradient points somewhere. The model dutifully learns the wrong mapping, and you find out only when evaluation word error rate refuses to drop below some floor you cannot explain. Misalignment is a *silent* corruption, which is why the confidence-filter stage at the end of the pipeline is not optional polish — it is the immune system.

## 1. What ASR training data actually is

**Senior rule of thumb: an ASR example is a short audio clip plus the verbatim words in it — and "short," "verbatim," and "in it" are each a decision you have to enforce, not a property the data gives you for free.**

A training pair is a waveform of typically 1 to 20 seconds and the exact orthographic (or phonetic) transcription of the speech in that window. "Typically 1 to 20 seconds" is not arbitrary: too short and the model never sees enough context to disambiguate coarticulation; too long and batches become ragged, memory blows up on the attention matrix, and a single alignment error contaminates a big clip. Most modern pipelines target a mean around 4 to 8 seconds with a hard cap near 20 to 30.

The corpora you can build or buy fall on two axes that, together, decide almost everything about label cost and difficulty.

![ASR corpora placed on speaking-style and acoustic-condition axes, with example datasets in each cell](/imgs/blogs/speech-data-for-asr-1.webp)

The matrix above is the map. The vertical axis is *speaking style*: **read** speech (someone reading prepared text — audiobooks, prompts), **conversational** speech (two people talking naturally — phone calls, interviews), and **in-the-wild** speech (whatever the internet contains — lectures, vlogs, streams). The horizontal axis is *acoustic condition*: **clean/studio** versus **noisy/real-world** (background noise, reverberation, overlapping speakers, far-field microphones). The top-left cell — read, clean — is where LibriSpeech lives, and it is *cheap* to label because the words are known in advance and the acoustics are forgiving. The bottom-right cell — in-the-wild, noisy — is where YouTube-scale corpora live, and it is *expensive*, because you cannot trust the transcript, the acoustics fight you, and human verification is the only ground truth.

A third distinction cuts across the grid: **scripted versus spontaneous**. Scripted speech has a known target text, which is what makes forced alignment possible at all. Spontaneous speech has disfluencies — "um," restarts, half-words, crosstalk — that no book contains, so you cannot align it to anything and must transcribe it from scratch (expensive) or accept weak, approximate labels (cheap but noisy). The entire forced-alignment machinery in this post applies to the *scripted* world; the spontaneous world is the subject of the follow-up on [weak supervision and low-resource speech](/blog/machine-learning/training-data/weak-supervision-and-low-resource-speech).

Where you sit on this map determines your strategy. Read-clean means "align a book to a recording," which is a solved, automatable problem. Conversational-noisy means "pay humans to transcribe and verify," which is a budgeting problem covered by [data scaling laws and budgets](/blog/machine-learning/training-data/data-scaling-laws-and-budgets). Everything in between is a negotiation between the two.

## 2. Where the audio comes from: the sources

**Senior rule of thumb: pick your source by the label tier you need, not by the raw hour count — a thousand gold hours will teach a model more reliable behavior than a hundred thousand weakly-labeled ones, and the two serve completely different jobs.**

There is now a well-worn ladder of open ASR corpora, and they differ far more in *how the labels were made* than in domain. Understanding the construction tells you exactly what each one is good for.

![Open ASR corpora compared across hours, style, label tier, and license](/imgs/blogs/speech-data-for-asr-5.webp)

The comparison figure above lays the four workhorses side by side; here they are with the construction detail that matters.

| Corpus | Hours (approx) | Style | How labels were made | Best used as |
| --- | --- | --- | --- | --- |
| **LibriSpeech** | 960 (English) | read audiobooks | LibriVox audio force-aligned to Gutenberg book text, then filtered by alignment quality | clean training + the default eval set |
| **Common Voice** | ~30k across 100+ languages | read prompts | volunteers read sentences aloud; other volunteers vote clips up/down | multilingual training, accent coverage |
| **GigaSpeech** | 10k transcribed (of a ~40k pool) | audiobooks, podcasts, YouTube | weak labels from existing captions, then a forced-alignment + segmentation pipeline, graded XS→XL | scaling English training with mixed domains |
| **People's Speech** | ~30k | mostly read / lectures | forced alignment against existing transcripts of CC-licensed audio | permissively-licensed large-scale training |
| **YODAS** | ~370k | in-the-wild YouTube | uploaded/auto captions used directly as weak labels | massive pretraining, self/weak supervision |

Read down the "how labels were made" column and the whole taxonomy resolves into one variable: *how much do you trust the transcript?* LibriSpeech trusts it a lot — the book is verbatim and the alignment was quality-filtered, so it is the closest thing to gold in the open world. Common Voice trusts human votes rather than alignment; the prompts are known, but the trust comes from crowd verification that the reader actually said the prompt cleanly. GigaSpeech starts from *weak* labels (whatever captions existed) and *earns* trust through an alignment-and-segmentation pipeline plus graded confidence buckets. YODAS barely trusts the labels at all — auto-generated YouTube captions are frequently wrong, mistimed, or auto-translated — which is exactly why it is enormous and cheap, and why you use it for pretraining rather than evaluation.

The practical mixing rule follows directly, and it echoes the general principle from [data mixing, domain weighting, and curriculum](/blog/machine-learning/training-data/data-mixing-domain-weighting-and-curriculum): pretrain on the weakly-labeled ocean for breadth, then fine-tune and *evaluate* on the gold puddle for reliability. Never let a weakly-labeled corpus into your evaluation set — its label noise makes your word error rate uninterpretable, and you will spend weeks chasing a "regression" that is really just a mistimed caption.

## 3. The segmentation problem, and why VAD comes first

**Senior rule of thumb: forced alignment does not want your whole two-hour recording — it wants speech, in chunks, with the silence and non-speech already carved off, or it will happily align words onto a saxophone solo.**

You have a 38-minute chapter. You cannot force-align 38 minutes in one shot: the alignment search is a dynamic program over a trellis whose size grows with `audio_frames × transcript_tokens`, and at chapter scale that is both slow and numerically fragile — a single hard region early on can throw off the entire downstream path. The fix is to break the recording into speech regions *first*, so alignment runs on tractable, mostly-speech windows.

Voice-activity detection (VAD) is the tool. A VAD model reads the audio in short frames — 10 to 30 milliseconds each — and emits, per frame, the probability that the frame contains speech. You threshold that probability, smooth it with hysteresis (a higher threshold to *enter* a speech region than to *leave* it, so a brief dip in energy mid-word does not split a word in two), and merge the surviving frames into contiguous regions.

![VAD as a keep/discard fork: frames become a speech probability, get thresholded, then merged into utterances or dropped as non-speech](/imgs/blogs/speech-data-for-asr-4.webp)

The figure above shows the fork. Frames flow into the VAD model, which produces a per-frame speech probability. The threshold-plus-hysteresis stage splits the stream: frames above the bar become speech regions that get merged into utterance-sized windows (roughly 0.3 to 20 seconds); frames below it are silence, music, or applause and get discarded. That discard arm is doing real work — every second of non-speech it removes is a second forced alignment cannot misinterpret.

Here is a production-grade VAD pass using Silero VAD, which is small, fast on CPU, and robust across languages:

```python
import torch, torchaudio

# Silero ships as a torch.hub model; utils gives you the timestamp helper.
model, utils = torch.hub.load('snakers4/silero-vad', 'silero_vad', trust_repo=True)
get_speech_timestamps, _, read_audio, _, _ = utils

wav = read_audio('chapter.wav', sampling_rate=16_000)   # 1-D float tensor, 16 kHz
regions = get_speech_timestamps(
    wav, model,
    sampling_rate=16_000,
    threshold=0.5,                # p(speech) to enter a region; hysteresis is internal
    min_speech_duration_ms=250,   # drop blips shorter than a syllable
    min_silence_duration_ms=300,  # bridge micro-pauses so a word is not split
    speech_pad_ms=120,            # keep a little context on each side of a region
    return_seconds=False,         # sample indices, not seconds
)
# regions -> [{'start': 6624, 'end': 71360}, {'start': 88192, 'end': 140544}, ...]
```

Two of those parameters carry most of the risk. `min_silence_duration_ms` too low and natural inter-word pauses shatter every sentence into single words; too high and two sentences with a real pause between them get glued into one 30-second monster. `speech_pad_ms` compensates for the VAD clipping the quiet onsets and offsets of speech (fricatives, trailing vowels) — set it to zero and you will chop the leading "s" off half your clips, which the model then learns to hallucinate. After VAD you merge and cap:

```python
def merge_regions(regions, sr=16_000, bridge_gap=0.2, max_len=20.0, min_len=0.6):
    out = []
    for r in regions:
        dur = (r['end'] - r['start']) / sr
        if dur < min_len:
            continue                                   # too short to be a useful clip
        if out and (r['start'] - out[-1]['end']) / sr < bridge_gap:
            out[-1]['end'] = r['end']                  # glue across a tiny gap
        else:
            out.append(dict(r))
    # A region longer than max_len will be re-split later on word boundaries
    # by the aligner; VAD only produces coarse windows.
    return out
```

VAD gives you *coarse* windows — "speech is roughly here." It does not know *which words* are where. That is the aligner's job, and VAD's real contribution is having deleted the non-speech before the aligner ever sees it.

## 4. Forced alignment: snapping known words onto the timeline

**Senior rule of thumb: forced alignment is the reverse of recognition — recognition asks "what words are in this audio?"; alignment already knows the words and asks only "where is each one?" — and that constraint makes it far more accurate than recognition, which is exactly why you exploit it.**

This is the heart of the pipeline. You have a speech region and you have the words that should be in it (from the book). Forced alignment finds the time boundaries of each word (and often each phone) by finding the single most probable alignment path between the acoustic model's frame-level outputs and the *known* token sequence.

The intuition first. A CTC acoustic model reads the audio and, for each frame, outputs a probability distribution over the vocabulary (characters or phones) plus a special *blank* symbol. If you were doing recognition you would search over *all* possible token sequences. In alignment you already know the sequence, so you only search over the ways that fixed sequence can be *stretched* across the frames — each token can occupy one or more frames, blanks fill the gaps, and repeated tokens are separated by blanks. The Viterbi algorithm walks a trellis of `frames × tokens` and picks the highest-probability monotonic path. That path *is* the alignment: it tells you which frames belong to which token, and frame indices convert to seconds.

<figure class="blog-anim">
<svg viewBox="0 0 720 320" role="img" aria-label="Forced alignment: a frontier sweeps the audio timeline left to right and each known transcript word snaps onto its time span in turn" style="width:100%;height:auto;max-width:820px">
<style>
.fa-band{fill:var(--surface,#f3f4f6);stroke:var(--border,#d1d5db);stroke-width:1.5}
.fa-cellline{stroke:var(--border,#d1d5db);stroke-width:1.5}
.fa-src{font:600 20px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280);text-anchor:middle}
.fa-lbl{font:600 13px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280)}
.fa-word{font:700 20px ui-sans-serif,system-ui;fill:var(--accent,#6366f1);text-anchor:middle}
.fa-frontierline{stroke:var(--accent,#6366f1);stroke-width:3}
.fa-tick{stroke:var(--text-secondary,#6b7280);stroke-width:1.5}
@keyframes fa-sweep{0%{transform:translateX(0)}100%{transform:translateX(640px)}}
@keyframes fa-w1{0%,6%{opacity:0}12%,100%{opacity:1}}
@keyframes fa-w2{0%,26%{opacity:0}32%,100%{opacity:1}}
@keyframes fa-w3{0%,46%{opacity:0}52%,100%{opacity:1}}
@keyframes fa-w4{0%,66%{opacity:0}72%,100%{opacity:1}}
@keyframes fa-w5{0%,87%{opacity:0}93%,100%{opacity:1}}
.fa-frontier{animation:fa-sweep 8s linear infinite}
.fa-c1{animation:fa-w1 8s ease-in-out infinite}
.fa-c2{animation:fa-w2 8s ease-in-out infinite}
.fa-c3{animation:fa-w3 8s ease-in-out infinite}
.fa-c4{animation:fa-w4 8s ease-in-out infinite}
.fa-c5{animation:fa-w5 8s ease-in-out infinite}
@media (prefers-reduced-motion:reduce){.fa-frontier{animation:none;transform:translateX(640px)}.fa-c1,.fa-c2,.fa-c3,.fa-c4,.fa-c5{animation:none;opacity:1}}
</style>
<text class="fa-lbl" x="40" y="60">known transcript (from the book)</text>
<text class="fa-src" x="110" y="100">the</text>
<text class="fa-src" x="240" y="100">sun</text>
<text class="fa-src" x="365" y="100">did</text>
<text class="fa-src" x="495" y="100">not</text>
<text class="fa-src" x="620" y="100">shine</text>
<text class="fa-lbl" x="40" y="180">audio timeline (16 kHz waveform)</text>
<rect class="fa-band" x="40" y="200" width="640" height="80" rx="8"/>
<line class="fa-cellline" x1="180" y1="200" x2="180" y2="280"/>
<line class="fa-cellline" x1="300" y1="200" x2="300" y2="280"/>
<line class="fa-cellline" x1="430" y1="200" x2="430" y2="280"/>
<line class="fa-cellline" x1="560" y1="200" x2="560" y2="280"/>
<text class="fa-word fa-c1" x="110" y="248">the</text>
<text class="fa-word fa-c2" x="240" y="248">sun</text>
<text class="fa-word fa-c3" x="365" y="248">did</text>
<text class="fa-word fa-c4" x="495" y="248">not</text>
<text class="fa-word fa-c5" x="620" y="248">shine</text>
<line class="fa-tick" x1="40" y1="290" x2="40" y2="300"/>
<line class="fa-tick" x1="360" y1="290" x2="360" y2="300"/>
<line class="fa-tick" x1="680" y1="290" x2="680" y2="300"/>
<text class="fa-lbl" x="40" y="316">0.0 s</text>
<text class="fa-lbl" x="650" y="316">3.4 s</text>
<g class="fa-frontier"><line class="fa-frontierline" x1="40" y1="196" x2="40" y2="284"/></g>
</svg>
<figcaption>Forced alignment sweeps the audio left to right; as the frontier passes each region, the next known transcript word snaps onto its time span, turning one long recording into word-level (audio, text) boundaries.</figcaption>
</figure>

The animation above is the mechanism in motion: a frontier moves along the audio, and each known word locks onto its span the moment the alignment path reaches it. That "locking" is the Viterbi backtrace committing to a boundary. Because the word sequence is fixed, the model is never confused about *what* to look for — only *where* — which is why alignment timestamps are far more trustworthy than raw recognition output.

`torchaudio` ships a first-class forced-alignment API built on exactly this. The `MMS_FA` bundle is a multilingual CTC model designed for alignment; `F.forced_align` runs the constrained Viterbi and returns the per-frame token assignment plus per-frame scores, and `F.merge_tokens` collapses the repeated frames into token spans:

```python
import torch
import torchaudio.functional as F
from torchaudio.pipelines import MMS_FA as bundle

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = bundle.get_model().to(device)      # CTC emission model for alignment
LABELS = bundle.get_dict()                 # {char: index}; index 0 is <blank>

def tokenize(words):
    # MMS_FA is character-based; keep only in-dictionary characters per word.
    return [[LABELS[c] for c in w if c in LABELS] for w in words]

@torch.inference_mode()
def align(waveform, words):
    emission, _ = model(waveform.to(device))          # (1, T, V) log-probs, ~20 ms/frame
    per_word = tokenize(words)
    flat = torch.tensor([sum(per_word, [])], dtype=torch.int32, device=device)
    aligned, scores = F.forced_align(emission, flat, blank=0)
    scores = scores[0].exp()                          # frame prob at the aligned token
    spans = F.merge_tokens(aligned[0], scores)        # list of TokenSpan(token, start, end, score)
    # regroup flat token spans back into per-word groups
    words_spans, i = [], 0
    for toks in per_word:
        words_spans.append(spans[i:i + len(toks)])
        i += len(toks)
    ratio = waveform.size(1) / emission.size(1)       # samples per emission frame
    return words_spans, ratio
```

Frame indices become seconds through `ratio` (samples per emission frame) divided by the sample rate. A word's start is its first token span's start; its end is its last token span's end; its confidence is the length-weighted mean of its token scores:

```python
def word_timings(words_spans, ratio, sr=16_000):
    rows = []
    for spans in words_spans:
        if not spans:                                 # empty = out-of-vocabulary word
            rows.append(None); continue
        start = spans[0].start * ratio / sr
        end   = spans[-1].end   * ratio / sr
        num = sum(s.score * (s.end - s.start) for s in spans)
        den = sum((s.end - s.start) for s in spans)
        rows.append((start, end, num / den))          # (start_s, end_s, confidence)
    return rows
```

That `confidence` column — the aligner's own probability that the token was really there — is the signal the whole quality-filter stage runs on. A confidently-aligned word sat right on top of a matching acoustic frame; a low-confidence one was forced onto a frame that did not match, which is precisely what happens at drift, at out-of-vocabulary words, and over non-speech.

### The three tools you will actually reach for

There are three mainstream ways to do this in 2026, and they trade accuracy against setup cost:

- **`torchaudio` CTC alignment** (the code above). Pure Python, GPU-friendly, no external install, great for character-level word timings. This is the default for building pipelines inside a PyTorch training stack.
- **Montreal Forced Aligner (MFA)**. A Kaldi-based tool that aligns at the *phone* level using a pronunciation dictionary and a triphone acoustic model. More accurate phone boundaries, mature multilingual models, but a heavier install and a batch-CLI workflow. This is what you use when you need phoneme boundaries (TTS data, linguistics) or when the language has a good MFA model but no good CTC model. Cross-reference the debugging companion on [CTC and alignment failures](/blog/machine-learning/debugging-training/debugging-ctc-and-alignment) before you fight the beam settings.
- **CTC-segmentation**. An algorithm purpose-built for the *long-audio* case: it aligns a long transcript to a long recording in one pass and returns per-utterance confidence, which is ideal for the audiobook-chapter problem. It underpins several open corpora's construction.

Here is the MFA workflow for the same audiobook chapter, because the CLI shape is not obvious the first time:

```bash
# 1. Lay out a one-speaker corpus: each recording's .wav beside its .txt transcript.
#    corpus/chapter_0001.wav   corpus/chapter_0001.txt   (16 kHz mono WAV)

# 2. Download an English acoustic model and its matching pronunciation dictionary.
mfa model download acoustic   english_us_arpa
mfa model download dictionary english_us_arpa

# 3. Align. MFA runs its own internal speech detection + Kaldi triphone alignment.
mfa align ./corpus english_us_arpa english_us_arpa ./aligned \
    --beam 100 --retry_beam 400 \
    --output_format long_textgrid --clean

# 4. Output: ./aligned/chapter_0001.TextGrid — word + phone tiers with time intervals.
```

The `--beam` / `--retry_beam` pair is the knob that matters: MFA aligns with the narrow beam first for speed, and any utterance it cannot align retries with the wide beam. When a whole recording fails to align, widening `retry_beam` is the first thing to try before you assume the transcript is wrong.

## 5. Audio preprocessing: the boring stage that silently breaks everything

**Senior rule of thumb: standardize sample rate, channels, and amplitude before anything else — a mismatch here does not throw an error, it just degrades every clip in a way no downstream stage can detect.**

Three transforms, in order, and each has a failure mode:

**Resample to 16 kHz.** Almost every ASR model expects 16 kHz. Speech energy lives below 8 kHz, and by the Nyquist relation a 16 kHz sample rate captures everything up to 8 kHz — enough for intelligibility, cheap to compute. The trap is directional: feeding a 44.1 kHz clip to a 16 kHz model without resampling means the model interprets your samples as if they arrived 2.76× slower, so every formant lands at the wrong frequency and every clip is quietly garbled. Downsampling must also low-pass filter first to avoid aliasing, which is why you use a real resampler and never naive decimation.

**Downmix to mono.** ASR models take one channel. Averaging stereo to mono is the safe default; taking only the left channel silently drops half your signal if the recording is panned.

**Normalize amplitude.** Peak- or RMS-normalize so clips have consistent loudness. Wildly varying input levels make the acoustic front-end's implicit gain assumptions wrong and hurt convergence, exactly the class of subtle input scaling bug catalogued in [audio input bugs](/blog/machine-learning/debugging-training/audio-input-bugs).

```python
import torch
import torchaudio
import torchaudio.functional as AF

def load_16k_mono(path: str) -> torch.Tensor:
    wav, sr = torchaudio.load(path)                # (channels, samples) at native sr
    if wav.size(0) > 1:                            # downmix any layout to mono
        wav = wav.mean(dim=0, keepdim=True)
    if sr != 16_000:                               # resample WITH anti-alias low-pass
        wav = AF.resample(wav, sr, 16_000)
    peak = wav.abs().max()
    if peak > 0:
        wav = wav / peak * 0.891                   # peak-normalize to about -1 dBFS
    return wav                                      # (1, samples), float32 in [-1, 1]
```

The `0.891` is a deliberate 1 dB of headroom below full scale so that any later processing (dithering, format conversion) does not clip. It is a small thing that prevents a class of "why does 3% of my corpus sound crunchy" bugs.

## 6. The label-quality spectrum

**Senior rule of thumb: every ASR corpus lives somewhere on a quality ladder, and the higher the rung the smaller the dataset — so a real training set is a blend of a lot of cheap weak labels and a little expensive gold, used for different jobs.**

![The label-quality funnel from weakly-labeled audio down to human-verified gold](/imgs/blogs/speech-data-for-asr-6.webp)

The funnel above is the shape of a production corpus. At the wide top is **weakly-labeled** audio — hundreds of thousands of hours whose "transcript" is a YouTube auto-caption, an uploaded subtitle, or a pseudo-label from running an existing model (often Whisper) over unlabeled audio. It is enormous and nearly free, and it is *approximate*: timings drift, words are wrong, non-speech is mislabeled. In the middle is **forced-aligned** audio — tens of thousands of hours where a known transcript existed and you machined it into tight pairs with the pipeline in this post. It is much cleaner, bounded by how much text-matched audio you can find. At the narrow bottom is **human-verified** audio — a thousand or so hours where a person listened and confirmed every word. It is the most trustworthy and by far the most expensive, and its job is calibration and evaluation, not bulk training.

The reason the funnel narrows is pure economics, and it maps onto the general [measuring data quality](/blog/machine-learning/training-data/measuring-data-quality) tradeoff: quality costs human minutes, and human minutes do not scale. A rough field figure is that careful human transcription-and-verification runs on the order of five to ten times real time — an hour of audio can take five to ten hours of human effort to transcribe cleanly — which is why nobody hand-labels a hundred thousand hours. The engineering move is to let the tiers do different jobs: pretrain on the weak ocean, fine-tune on the aligned middle, and reserve the gold puddle for the evaluation set that tells you the truth. The one rule you never break is keeping the tiers *labeled* in your metadata, so you can weight, filter, and — critically — never accidentally evaluate on a weak-labeled clip.

## Worked scenario: aligning one audiobook chapter

Let us machine one real chapter end to end and count what falls out, because the numbers are the intuition. The input is a single LibriVox chapter: 38 minutes 24 seconds (2,304 seconds), originally a 44.1 kHz stereo MP3, narrated by one reader. The reference text is the matching Project Gutenberg chapter: about 6,900 words.

**Step 0 — text normalization.** Before alignment, the transcript must match how the acoustic model spells things: lowercase, expand numbers to words ("1849" → "eighteen forty nine"), strip most punctuation, and — the step everyone forgets — *remove the LibriVox boilerplate*. Every LibriVox recording opens with a spoken disclaimer ("This is a LibriVox recording. All LibriVox recordings are in the public domain...") and often a chapter announcement that is **not in the book text**. Leave it in the audio with no matching words and the aligner will smear the first real sentence backward across 15 seconds of un-transcribed speech. We trim the leading ~14 seconds and normalize the text to about 7,050 alignment tokens.

**Step 1 — preprocess.** `load_16k_mono` turns the 44.1 kHz stereo MP3 into 16 kHz mono, peak-normalized. 2,304 seconds in, 2,290 seconds out after trimming the intro.

**Step 2 — VAD.** Silero VAD with the parameters above finds 547 raw speech regions; merging across sub-0.2-second gaps and dropping sub-0.6-second blips leaves 512 coarse windows and removes about 41 seconds of inter-chapter silence and page-turn pauses.

**Step 3 — forced alignment.** Running `MMS_FA` over the windows against the normalized text yields word-level timestamps. Cutting into utterances at pauses longer than 0.5 seconds (with an 18-second hard cap) produces **523 segments**, mean duration **4.3 seconds**, min 0.9 s, max 16.8 s, totaling 37.4 minutes of speech.

**Step 4 — confidence filter.** Each segment carries a mean per-word alignment confidence. The distribution is bimodal: a fat peak near 0.85 (clean speech) and a thin tail below 0.4 (trouble). We drop segments below 0.40.

| Confidence threshold | Segments kept | Hours kept | What the dropped tail contains |
| --- | --- | --- | --- |
| none (0.00) | 523 | 0.623 | everything, including 3 boilerplate + noise clips |
| 0.40 (chosen) | 489 | 0.582 | drops mistimed sentence edges, a cough region, a mis-normalized numeral |
| 0.60 (aggressive) | 441 | 0.522 | also drops fast-speech and quiet-onset clips that are actually fine |

At the chosen threshold we drop **34 segments (6.5%)**, retaining **489 clean pairs** totaling **34.9 minutes**, mean 4.28 seconds. The dropped tail is exactly what you want gone: three clips where residual boilerplate leaked in, a stretch where the narrator coughed and laughed (non-speech the VAD `speech_pad_ms` had over-extended into), and one segment where "1849" was mis-normalized and the digits could not align. Push to 0.60 and you start throwing away *good* fast-speech clips — the confidence dips not because the alignment is wrong but because rapid speech has genuinely lower per-frame acoustic probability. The 0.40 knee is where you drop noise without eating signal.

Finally, package the survivors into shards for training. `webdataset` writes them as tar files streamable straight into a `DataLoader`:

```python
import io, webdataset as wds, soundfile as sf

with wds.TarWriter('librichapter-000000.tar') as sink:
    for i, seg in enumerate(kept):                      # kept = the 489 survivors
        s, e = int(seg['start'] * 16_000), int(seg['end'] * 16_000)
        clip = wav[:, s:e].squeeze().numpy()
        buf = io.BytesIO(); sf.write(buf, clip, 16_000, format='WAV')
        sink.write({
            '__key__': f'{i:06d}',
            'wav': buf.getvalue(),
            'txt': seg['text'],
            'json': {'conf': round(seg['conf'], 3),
                     'dur': round(seg['end'] - seg['start'], 2)},
        })
```

One 38-minute file became 489 training examples in a few minutes of compute and zero human labeling. Scale that across a few thousand LibriVox chapters and you have reconstructed something very close to how LibriSpeech itself was built.

## Troubleshooting: when alignment goes wrong

Alignment fails in a handful of recognizable ways, and each has a distinct signature. The failures share a root cause — the aligner is *forced* to place every word somewhere, so when reality and transcript disagree, it does not error, it *drifts*, dragging every subsequent boundary off the audio.

![Misalignment failures on the left with their concrete fixes on the right](/imgs/blogs/speech-data-for-asr-7.webp)

The before/after figure pairs each symptom with its fix; here is each one as symptom → detection → fix.

### Misalignment and transcript drift

- **Symptom.** Word boundaries are all shifted by a growing offset; late-chapter clips contain the *wrong* words entirely; per-segment confidence declines monotonically through the file.
- **Root cause.** A missing or extra chunk of transcript early on — dropped boilerplate, a skipped paragraph, a duplicated sentence — knocks the token stream out of sync with the audio, and every word after inherits the error.
- **Detection.** Plot mean confidence against timestamp. A healthy alignment is flat and high; a drift shows a cliff at the point of divergence, then sustained low confidence. A cheap automated check: after aligning, transcribe a few random clips with a real recognizer and compare — drift shows up as high word error on late clips only.
- **Fix.** Use a long-audio aligner (CTC-segmentation) that anchors on high-confidence islands and re-syncs, rather than a naive left-to-right pass. Fix the text: verify the transcript is complete and boilerplate is removed. Segment on strong anchors so one bad region cannot poison the whole file.

### Out-of-vocabulary and proper-noun words

- **Symptom.** Alignment is fine until a rare name or foreign word, then boundaries around it are garbage; in MFA the utterance fails outright with an OOV error.
- **Root cause.** The pronunciation dictionary has no entry for the word, or the CTC model never saw those characters, so the aligner cannot match it acoustically and *guesses* a span — usually too short — which shoves its neighbors.
- **Detection.** MFA emits an `oovs_found.txt` listing every out-of-vocabulary token; for CTC, the `word_timings` helper returns `None` for words whose tokens fell out of the dictionary.
- **Fix.** Add the word to the lexicon with a hand- or G2P-generated pronunciation (`mfa g2p` produces pronunciations for OOVs automatically). For CTC, ensure the character set covers the transcript. If a word genuinely cannot be aligned, drop the *segment* containing it rather than trusting its neighbors.

### Non-speech slipping into segments

- **Symptom.** Some clips contain music, applause, laughter, or long breaths; the model trained on them starts hallucinating words during silence at inference.
- **Root cause.** VAD was too permissive (threshold too low, `speech_pad_ms` too high) so non-speech regions were passed to the aligner, which then stretched real words across the non-speech to cover it.
- **Detection.** Non-speech segments align with low confidence and anomalous duration-per-character (very long clips with few words). Flag clips where audio duration divided by character count is a large outlier.
- **Fix.** Tighten VAD; run a dedicated audio-tagging model to detect music/applause and exclude those regions before alignment; and — the backstop — the confidence filter, which drops most of these because words forced over noise score low.

### Speaker and domain imbalance

- **Symptom.** Word error rate is great on audiobook-style read speech and terrible on your actual target (phone calls, a specific accent, a technical domain).
- **Root cause.** Your manufactured corpus inherited the demographics of its source. LibriVox skews toward certain accents, older public-domain literary vocabulary, and clean read prose; a model trained only on it has never heard your users.
- **Detection.** Break evaluation word error rate down by speaker, accent, and domain rather than reporting one number. A large gap between the aggregate and the worst slice is the tell.
- **Fix.** This is a data *mixing* problem, not an alignment one: deliberately source and up-weight the under-represented slices, exactly the rebalancing logic in [data selection and pruning](/blog/machine-learning/training-data/data-selection-and-pruning). Add targeted human-verified data for the slices that matter most.

### Sample-rate and channel mismatch

- **Symptom.** Alignment confidence is uniformly mediocre across an otherwise-clean recording, or a subset of files aligns far worse than the rest with no obvious content difference.
- **Root cause.** Those files were a different sample rate or channel layout and were fed to a 16 kHz mono model without conversion — the classic silent audio bug.
- **Detection.** Log the native sample rate and channel count of every file *before* preprocessing and assert they were converted. A histogram of native sample rates almost always reveals a contaminating subset (8 kHz telephone audio mixed into a 16 kHz corpus, or 48 kHz video rips).
- **Fix.** Force every file through the `load_16k_mono` path; never trust that "the corpus is 16 kHz." For genuinely telephone-band (8 kHz) audio, upsampling to 16 kHz recovers the format but not the lost high frequencies — treat it as its own domain rather than pretending it is wideband.

## Case studies: how the big corpora were built

These are not hypotheticals — they are the datasets you will actually train on, and each one is a case study in a different point on the label-quality funnel.

### 1. LibriSpeech — automation as the whole point

LibriSpeech is the canonical demonstration that the pipeline in this post *works at scale*. Its authors took LibriVox — volunteer-read public-domain audiobooks — and the matching Project Gutenberg book texts, and force-aligned the two. The clever part was quality control: they aligned each chapter, kept only the segments where alignment was confident, and split the result by *alignment quality* into "clean" and "other" subsets, with the noisier readers and harder alignments quarantined into "other." That is why `test-clean` and `test-other` exist and why models always report both — the split is literally a confidence filter made into a benchmark. The lesson: 960 hours of gold English training data were produced with essentially zero human transcription, because the transcript already existed and only the *alignment* had to be manufactured. LibriSpeech is what your audiobook pipeline is trying to reproduce.

### 2. Common Voice — trust from crowds, not alignment

Mozilla's Common Voice took the opposite tack. There is no long audio to segment: volunteers are shown a short sentence and record themselves reading it, so each clip is *born* as an aligned pair. The hard problem is not *where* the words are — the clip is one sentence — but *whether the reader said it correctly and cleanly*. Trust comes from a second crowd: other volunteers listen and vote each clip up or down, and a clip needs a threshold of up-votes to enter the "validated" set. The result is a corpus that is unusually broad — 100-plus languages, many accents, real consumer microphones — precisely because the barrier to contributing is a phone and a minute. Its weakness is the flip side: read-prompt speech is not conversational, and the acoustic diversity is uneven across languages. The lesson: when you cannot align (no long audio, no book), human verification is the trust mechanism, and crowd-sourcing makes it scale across languages that no company would fund individually.

### 3. GigaSpeech — earning trust from weak labels

GigaSpeech is the most instructive because it sits in the *middle* of the funnel and shows the machining explicitly. It pools audiobooks, podcasts, and YouTube — sources whose "transcripts" are existing captions of wildly varying quality. The construction pipeline is essentially this post: normalize the text, run forced alignment against the weak captions, segment on the alignment, and — the key step — *grade* every segment by alignment confidence, then expose graded subsets from XS (a tiny high-confidence slice) up to XL (10,000 hours, looser). It also standardizes punctuation and normalizes the text so that transcripts are consistent across the three source types. The lesson: you do not need clean labels to start; you need a pipeline that *measures* label quality and lets the trainer choose its own quality/quantity tradeoff by picking a subset. This is [classifier- and confidence-based filtering](/blog/machine-learning/training-data/classifier-and-perplexity-based-quality-filtering) applied to speech.

### 4. People's Speech — scale under a permissive license

The People's Speech corpus set out to prove you could assemble roughly 30,000 hours of supervised English *and* keep it under a permissive (Creative-Commons-family) license — the licensing constraint being the actual hard part, since most large audio is copyrighted. Construction relied on forced alignment against existing transcripts of CC-licensed audio, with confidence-based filtering to control label noise across a very heterogeneous source pool. The lesson for a practitioner: at scale, *license provenance* becomes a first-class data-quality axis alongside acoustic and label quality — a corpus you cannot legally ship is worth zero hours regardless of its word error rate, a point the [sourcing and collecting](/blog/machine-learning/training-data/sourcing-and-collecting-training-data) post makes for text and that applies double to audio.

### 5. YODAS — the weak-label ocean

YODAS pushed to the far top of the funnel: roughly 370,000 hours of YouTube audio with labels taken directly from uploaded and auto-generated captions, across 100-plus languages. The captions are frequently wrong, mistimed, auto-translated, or machine-generated — so YODAS is essentially *unusable as gold* and *invaluable as pretraining*. Its size makes it a testbed for weak and self-supervised methods that can tolerate label noise, and for the pseudo-labeling loop where you run a strong model over the audio to *replace* the weak captions with better ones. The lesson: at the scale where human anything is impossible, the data strategy shifts from "clean the labels" to "build methods robust to dirty labels" — which is the entire subject of the [weak supervision and low-resource speech](/blog/machine-learning/training-data/weak-supervision-and-low-resource-speech) follow-up.

### 6. TED-LIUM — a domain in a box

TED-LIUM built its corpus from TED talks aligned to their published transcripts — a tidy example of the audiobook pattern applied to a single, coherent domain (prepared-but-spoken lectures, one speaker, good microphones, a distinctive vocabulary). Because the domain is narrow and consistent, forced alignment is easy and the resulting data is clean, but a model trained only on it inherits TED's register — confident, rehearsed, monologue speech — and stumbles on dialogue. The lesson ties back to the imbalance troubleshooting above: a clean single-domain corpus is a feature for studying that domain and a trap if you mistake it for general coverage.

## When to reach for forced-alignment data, and when not to

Reach for the manufacture-from-long-audio pipeline in this post when:

- You have **long recordings and a matching known transcript** (audiobooks, lectures with slides/scripts, dubbed media with subtitles, parliamentary proceedings with hansards). This is the pipeline's home turf and it scales to thousands of hours with no human labeling.
- You need **word- or phone-level timestamps** for downstream work — TTS training data, karaoke/subtitle timing, or phonetic analysis. Only alignment gives you these.
- You are building **read/clean or lecture-style** training data where the transcript is reliable and the acoustics are forgiving.
- You want a **quality-graded** corpus — the confidence score gives you a free knob to trade quantity for cleanliness, exactly as GigaSpeech does.

Skip it, or handle with extra care, when:

- The speech is **spontaneous and unscripted** (conversations, meetings, interviews). There is no ground-truth text to align to; you must transcribe from scratch or accept weak labels, and forced alignment has nothing to bite on.
- The transcript is **weak or approximate** (auto-captions). You *can* still align, but treat the output as weakly-labeled — confidence-filter hard and never let it near evaluation.
- You need a **trustworthy evaluation set**. Eval data should be human-verified, full stop; a mis-timed clip in your test set makes every number a lie.
- The domain is **acoustically hostile** (far-field, heavy overlap, extreme noise). Alignment degrades badly here; budget for human transcription of at least a seed set, and lean on the noisy-conversational strategies rather than the audiobook pipeline.

The one-sentence version: forced alignment is a machine that turns *known text plus long audio* into training pairs almost for free, it is the reason the open ASR world exists, and its output is only ever as trustworthy as the transcript you fed it and the confidence threshold you enforced on the way out. Everything downstream — the [scaling budget](/blog/machine-learning/training-data/data-scaling-laws-and-budgets), the [mixing weights](/blog/machine-learning/training-data/data-mixing-domain-weighting-and-curriculum), the [quality measurement](/blog/machine-learning/training-data/measuring-data-quality) — assumes the pairs are real. Make them real here, and the rest of the stack has a chance.

## Further reading

- [How Whisper handles audio](/blog/machine-learning/signal-processing/whisper-under-the-hood) — the encoder side of ASR, and why 16 kHz log-mel is the standard front-end.
- [Debugging CTC and alignment](/blog/machine-learning/debugging-training/debugging-ctc-and-alignment) — when the trellis, beam, or blank handling goes wrong.
- [Audio input bugs](/blog/machine-learning/debugging-training/audio-input-bugs) — the silent sample-rate, channel, and scaling failures that survive every unit test.
- [Weak supervision and low-resource speech](/blog/machine-learning/training-data/weak-supervision-and-low-resource-speech) — pseudo-labeling, self-training, and building ASR where no clean transcript exists.
- [Why data decides the model](/blog/machine-learning/training-data/why-data-decides-the-model) — the series thesis that the corpus, not the architecture, sets the ceiling.
