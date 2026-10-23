---
title: "The Whisper Family, Release by Release: From the 2022 Checkpoints to large-v3-turbo"
date: "2026-10-23"
publishDate: "2026-10-23"
description: "A chronological reference catalog of OpenAI's open-weight Whisper speech models through October 2026: the nine original checkpoints, large-v2, large-v3 and large-v3-turbo, plus the open ecosystem built on the weights and the hosted models that replaced them, with dates, specs, licenses and primary sources."
tags: ["whisper", "openai", "speech-recognition", "asr", "whisper-large-v3", "whisper-turbo", "distil-whisper", "open-weights", "model-family", "timeline"]
category: "machine-learning"
subcategory: "Frontier Models"
author: "Hiep Tran"
featured: false
readTime: 16
---

Whisper is the odd member of every open-weight family list. It has had exactly four releases, the newest open checkpoint is two years old, and it is still the speech model most people reach for when they want transcription on their own hardware. OpenAI published nine checkpoints in September 2022, an improved large model in December 2022, a retrained large-v3 in November 2023 and a decoder-pruned large-v3-turbo at the end of September 2024. Then the open line stopped. Every speech-to-text model OpenAI shipped after that, from gpt-4o-transcribe in March 2025 to gpt-transcribe in July 2026, is hosted only.

That shape is unusual, and it is the main thing worth knowing about the family. The architecture barely moved across the four releases. What moved was the training recipe, the amount of audio (680k hours to 5M hours), the input features (80 to 128 mel bins) and, last, the decoder depth (32 layers to 4). Meanwhile the community did more with the weights than OpenAI did: distillation, CTranslate2 and ggml ports, batched long-form pipelines. This post is the map. It walks every open release in date order, says what changed from the previous one, gives a spec table, names the license, and links a primary source for each claim. The timeline below is the mental model: a dense two years of open checkpoints, then an empty open lane while the hosted lane fills up.

![Whisper release timeline from September 2022 to October 2026 with three lanes: OpenAI open weights, community models built on the weights, and hosted-only API models, showing the open lane empty after large-v3-turbo](/imgs/blogs/whisper-model-family-timeline-1.webp)

Look at the top lane after October 2024. **As of October 23, 2026, the newest open-weight OpenAI speech model is still Whisper large-v3-turbo** (`openai/whisper-large-v3-turbo`, shipped in `openai-whisper` 20240930). The newest OpenAI speech models of any kind are `gpt-transcribe` and `gpt-live-transcribe` (July 28, 2026), which have no published weights.

A note on method. I pulled the GitHub release list and CHANGELOG of `openai/whisper`, the announcement threads in its Discussions tab (the closest thing Whisper has to release notes), the `model-card.md` in the repo, the Hugging Face Hub API for every repo under `openai` and `distil-whisper` (creation dates and each `config.json`), the arXiv papers, and the OpenAI API changelog for the hosted models. The OpenAI blog itself returned 403 to my fetcher, so where a date only appears there I fall back to the model card or the Hub and say so. "Hub created" is a lower bound on the public date. If you want the mechanism rather than the catalog (log-mel front end, special-token task format, timestamp decoding), read [Whisper Under the Hood](/blog/machine-learning/signal-processing/whisper-under-the-hood) first.

## The timeline at a glance

| Date | Release | Weights | One-line significance |
| --- | --- | --- | --- |
| 2022-09 (Hub created 09-26) | Whisper tiny, base, small, medium, large + four `.en` | Open, MIT | 680k hours of weakly supervised audio; 39M to 1550M params |
| 2022-12-05 (Hub), announced 12-08 | large-v2 | Open, MIT | Same architecture, 2.5x more epochs plus heavy regularization |
| 2022-12-06 | arXiv 2212.04356 | n/a | The paper, already reporting large-v2 numbers |
| 2023-03 | `whisper-1` API | Hosted only | large-v2 behind an endpoint |
| 2023-11-06 | large-v3 | Open, MIT | 128 mel bins, Cantonese token, 5M hours (4M pseudo-labeled by v2) |
| 2024-09-30 | large-v3-turbo (`turbo`) | Open, MIT | large-v3 with the decoder pruned from 32 to 4 layers and fine-tuned |
| 2025-03-20 | gpt-4o-transcribe, gpt-4o-mini-transcribe | Hosted only | First non-Whisper OpenAI transcription models |
| 2025-10-16 (SDK) | gpt-4o-transcribe-diarize | Hosted only | Speaker labels in the transcription endpoint |
| 2026-05-07 | gpt-realtime-whisper | Hosted only | Streaming speech-to-text on the Realtime API |
| 2026-07-28 | gpt-transcribe, gpt-live-transcribe | Hosted only | Current file and streaming transcription models |
| 2026-08-26 | Deprecation notice | n/a | `whisper-1` and the gpt-4o transcribe models shut down 2027-02-26 |

Community releases built on the weights (Distil-Whisper, faster-whisper, whisper.cpp, WhisperX) get their own section below instead of rows here, because none of them is an OpenAI model.

## Whisper (September 2022): nine checkpoints and a recipe

The first release was a family from day one. The `openai/whisper` GitHub repo was created on September 16, 2022, the model card gives the release as September 2022, and all nine Hub repos (`whisper-tiny` through `whisper-large`, plus `tiny.en`, `base.en`, `small.en`, `medium.en`) were created on September 26, 2022. The paper, *Robust Speech Recognition via Large-Scale Weak Supervision* by Radford, Kim, Xu, Brockman, McLeavey and Sutskever, went up as [arXiv 2212.04356](https://arxiv.org/abs/2212.04356) on December 6, 2022.

### What was new

The architecture was deliberately boring: a plain encoder-decoder Transformer. Audio is resampled to 16 kHz, cut into 30-second windows, and turned into an 80-channel log-magnitude mel spectrogram on 25 ms windows with a 10 ms stride. Two conv layers halve that to 1,500 encoder positions. The decoder is a text model over a 51,865-token byte-level BPE vocabulary, and the task (language, transcribe or translate, timestamps or not) is selected by special tokens in the decoder prompt.

The novelty was the data. The model card describes 680,000 hours of audio with transcripts collected from the internet: 438k hours of English audio with English transcripts (65%), 126k hours of non-English audio with English transcripts (about 18%, which is what teaches translation), and 117k hours of non-English audio with matching transcripts (about 17%) covering 98 languages. No human labeling pass; the paper filtered and de-duplicated machine-generated transcripts instead.

The headline result was robustness, not a leaderboard number. The best zero-shot Whisper model scores 2.5 WER on LibriSpeech test-clean, roughly mid-2019 supervised state of the art. But against a wav2vec 2.0 Large model with the same 2.7 WER on test-clean, zero-shot Whisper made **55.2% fewer errors on average** across other English test sets (average WER 12.8 versus 29.3 in the paper's Table 2). Even `tiny` (39M parameters, 6.7 WER on test-clean) was roughly competitive with the best supervised LibriSpeech model once you leave LibriSpeech.

| Size | Params | Enc / dec layers | d_model | English-only | README VRAM | README speed vs large |
| --- | --- | --- | --- | --- | --- | --- |
| tiny | 39M | 4 / 4 | 384 | tiny.en | ~1 GB | ~10x |
| base | 74M | 6 / 6 | 512 | base.en | ~1 GB | ~7x |
| small | 244M | 12 / 12 | 768 | small.en | ~2 GB | ~4x |
| medium | 769M | 24 / 24 | 1024 | medium.en | ~5 GB | ~2x |
| large | 1550M | 32 / 32 | 1280 | none | ~10 GB | 1x |

Layer counts and widths come from each repo's `config.json`; the rest from the [GitHub README](https://github.com/openai/whisper). The README notes that the `.en` models help most at `tiny` and `base` and that the gap is small by `small.en` and `medium.en`. **License:** code and weights MIT, per the repo `LICENSE`.

## large-v2 (December 2022): same model, trained harder

The `openai/whisper-large-v2` Hub repo was created on December 5, 2022, and the [announcement thread](https://github.com/openai/whisper/discussions/661) followed on December 8. Nothing about the network changed.

### What changed versus large

- **2.5x more training epochs** over the same data.
- **Regularization to survive those epochs:** SpecAugment, stochastic depth and BPE dropout.
- **Naming:** the original `large` was renamed `large-v1`, and `load_model("large")` started returning v2.

OpenAI reported about a 5% relative error reduction in English and about 10% in other languages, with an explicit warning that some audio gets worse. The thread also notes that v2 sits more "on trend" with the smaller sizes in the scaling curves, which reads as an admission that v1 large was undertrained. The arXiv paper posted two days earlier already reports large-v2 numbers, so the 55.2% robustness figure above is a v2 result.

| Spec | large-v2 |
| --- | --- |
| Params | 1550M |
| Encoder / decoder layers | 32 / 32 |
| Mel bins | 80 |
| Vocabulary | 51,865 |
| Data | 680k hours, 2.5x the v1 epochs |
| License | MIT |

large-v2 matters beyond its own scores for two reasons. It is the model OpenAI put behind the `whisper-1` API in March 2023 (at $0.006 per minute, per the launch post), and it is the teacher that labeled most of large-v3's training data.

## large-v3 (November 6, 2023): more audio, a new front end

The `large-v3` [release thread](https://github.com/openai/whisper/discussions/1762) and the `openai-whisper==20231106` package both landed on November 6, 2023; the Hub repo was created the next day.

### What changed versus large-v2

- **Input features:** 128 mel frequency bins instead of 80. This is the one change that breaks drop-in compatibility: v3 checkpoints need a different feature extractor, and every downstream port had to learn `n_mels=128`.
- **Vocabulary:** one new language token for Cantonese, taking the vocabulary from 51,865 to 51,866.
- **Data:** 1 million hours of weakly labeled audio plus 4 million hours of audio **pseudo-labeled by large-v2**, trained for 2.0 epochs over the mixture. That is about 7x the hours of v1 and v2.
- **Alias:** `"large"` now points at v3.

![Bar chart of Whisper training data in hours drawn to scale: 680k for v1 and v2, 5M for large-v3 split into 1M weakly labeled and 4M pseudo-labeled by large-v2, with Distil-Whisper's 22k and 98k hour distillation sets for comparison](/imgs/blogs/whisper-model-family-timeline-3.webp)

Notice how much of the v3 bar is v2 talking to itself. Four of the five million hours carry transcripts that large-v2 produced, which makes v3 a self-training result as much as a scaling result. OpenAI reported 10% to 20% error reduction versus large-v2 across languages where v3 is below 60% error on Common Voice 15 and FLEURS. Its evaluation notes are worth reading for anyone benchmarking: Korean switched to CER because of inconsistent spacing in the labels, Serbian was scored after converting both Latin and Cyrillic to Cyrillic, and all Chinese was normalized to Simplified.

| Spec | large-v3 |
| --- | --- |
| Params | 1550M |
| Encoder / decoder layers | 32 / 32 |
| d_model | 1280 |
| Mel bins | 128 |
| Vocabulary | 51,866 |
| Data | 1M h weakly labeled + 4M h pseudo-labeled, 2.0 epochs |
| License | MIT (repo); see the license note below |

## large-v3-turbo (September 30, 2024): cut the decoder

`turbo` shipped in `openai-whisper` [v20240930](https://github.com/openai/whisper/releases/tag/v20240930) on September 30, 2024. The Hub repo and the [release thread](https://github.com/openai/whisper/discussions/2363) are dated October 1, and `model-card.md` says September 2024, so expect both dates in the wild.

### What changed versus large-v3

- **Decoder: 32 layers down to 4**, the same depth as `tiny`. The encoder keeps all 32 layers.
- **Training:** prune, then fine-tune for two more epochs on the same multilingual **transcription** data used for v3, with translation data excluded. OpenAI says plainly not to expect turbo to translate well.
- **Inspiration:** the thread credits Distil-Whisper's finding that a small decoder buys most of the speed for little accuracy, but turbo is fine-tuned, not distilled.
- **Default:** the `whisper` CLI now uses `--model turbo` unless told otherwise.

![Scatter of Whisper parameter count against README relative speed: tiny, base, small, medium and large fall on one declining curve while turbo sits far above it at about half of large's parameters and roughly 8x its speed](/imgs/blogs/whisper-model-family-timeline-4.webp)

The point to take from the scatter is that turbo breaks the old trade-off line. The original five sizes trade parameters for speed smoothly. Turbo keeps 809M parameters, most of them in the encoder, yet runs at about 8x large's speed per the README, close to `tiny`'s ~10x. That works because autoregressive decoding runs the decoder once per token while the encoder runs once per 30-second window. Combined with a same-week patch to use `F.scaled_dot_product_attention`, OpenAI's thread says turbo transcribes faster than `tiny` used to.

Accuracy: OpenAI says turbo performs similarly to large-v2 across languages, with larger degradation on some, naming Thai and Cantonese, and better relative results on the cleaner FLEURS recordings than on Common Voice.

| Spec | large-v3-turbo |
| --- | --- |
| Params | 809M (README, Hub card); 798M (`model-card.md`) |
| Encoder / decoder layers | 32 / 4 |
| Mel bins | 128 |
| Vocabulary | 51,866 |
| README VRAM / speed | ~6 GB / ~8x |
| Tasks | Transcription; translation not trained |
| License | MIT |

## The open ecosystem built on the weights

OpenAI's last code release was `openai-whisper` [v20250625](https://github.com/openai/whisper/releases/tag/v20250625) in June 2025: safer `torch.load`, Triton fixes, docs that state the turbo translation limitation. No new weights. Most of the work on Whisper since 2023 happened outside OpenAI, and four projects define how people actually run it.

![Lineage diagram: large-v1 to large-v2 to large-v3 to large-v3-turbo, with Distil-Whisper models branching from v2 and v3, the whisper-1 API serving large-v2, and faster-whisper, whisper.cpp and WhisperX as runtimes that consume any checkpoint](/imgs/blogs/whisper-model-family-timeline-5.webp)

Read the diagram as two layers: models above, runtimes below. Every runtime accepts any checkpoint, which is why a new OpenAI release propagated to all of them within weeks.

**Distil-Whisper (Hugging Face).** [arXiv 2311.00430](https://arxiv.org/abs/2311.00430) (Gandhi, von Platen, Rush, November 1, 2023) keeps the full encoder, shrinks the decoder to 2 layers, and trains it on pseudo-labels from the teacher. The paper reports 5.8x faster, 51% fewer parameters and within 1% WER on out-of-distribution test data, plus about 2x speed as a speculative-decoding draft with identical outputs to the teacher. Releases, by Hub creation date:

| Model | Hub created | Enc / dec | Mel | Notes from the card |
| --- | --- | --- | --- | --- |
| distil-large-v2, distil-medium.en | 2023-10-24 | 32/2, 24/2 | 80 | First release |
| distil-small.en | 2023-12-06 | 12/4 | 80 | Smallest variant |
| distil-large-v3 | 2024-03-21 | 32/2 | 128 | 6.3x faster than large-v3, within 1% WER long-form; 22k h |
| distil-large-v3.5 | 2024-12-05 (variants 2025-03-14) | 32/2 | 128 | 98k h, 80 epochs vs 11; 756M params |

The v3.5 card compares itself directly to turbo: 1.46x turbo's real-time factor, 7.08 versus 7.30 short-form OOD WER (better), 11.39 versus 10.25 long-form OOD WER (worse). Its 98k hours came from 196k hours of public data pseudo-labeled by large-v3, keeping only segments where the pseudo-label and the ground truth disagree by at most 10% WER. All Distil-Whisper models are MIT, inherited from Whisper.

**faster-whisper (SYSTRAN, repo created February 11, 2023, MIT).** A reimplementation on CTranslate2 that claims up to 4x faster than `openai/whisper` at the same accuracy, with int8 on CPU and GPU. Its README benchmark transcribes 13 minutes of audio with large-v2 on an RTX 3070 Ti: 2m23s for `openai/whisper` in fp16, 1m03s for faster-whisper in fp16, 16s with int8 and batch size 8. Latest release as of this writing: v1.2.1, October 31, 2025.

**whisper.cpp (ggml, repo created September 25, 2022, MIT).** A C/C++ port that started within days of the release and became the default for CPU, Apple Silicon and mobile. Releases run from v1.0.4 (December 2022) to v1.9.5 (October 6, 2026), so it is the most actively shipped Whisper artifact today.

**WhisperX (Bain, Huh, Han, Zisserman; [arXiv 2303.00747](https://arxiv.org/abs/2303.00747), Interspeech 2023, BSD-2-Clause).** Fixes long-form problems (drift, hallucination, repetition, no batching) with VAD-based "cut and merge" pre-segmentation and forced phoneme alignment for word timestamps. The abstract reports a twelve-fold transcription speedup from batched inference.

## Hosted-only OpenAI speech models

One line each, from the [OpenAI API changelog](https://developers.openai.com/api/docs/changelog) unless noted. None has published weights.

- `whisper-1`, March 2023: large-v2 served through the API (launch post). Shutdown scheduled February 26, 2027.
- `gpt-4o-transcribe` and `gpt-4o-mini-transcribe`, March 20, 2025: hosted only.
- `gpt-4o-transcribe-diarize`, October 16, 2025 per the SDK release notes: hosted only.
- `gpt-4o-mini-transcribe-2025-12-15`, December 15, 2025: hosted snapshot, made the default slug January 13, 2026.
- `gpt-realtime-whisper`, May 7, 2026: hosted only, streaming speech-to-text. The name says Whisper; nothing published says it shares weights with the open models.
- `gpt-transcribe` and `gpt-live-transcribe`, July 28, 2026: hosted only, the current migration targets.
- August 26, 2026: deprecation of `whisper-1`, `gpt-4o-transcribe`, `gpt-4o-mini-transcribe` and `gpt-4o-transcribe-diarize` announced.

## What is latest as of October 2026

Open weights: **large-v3-turbo** remains the newest OpenAI checkpoint, and nothing speech-related has appeared under `openai` on the Hub since October 2024. The organization's 2025 and 2026 uploads are `gpt-oss`, `gpt-oss-safeguard`, `circuit-sparsity` and `privacy-filter`. The newest open model in the extended family is **distil-large-v3.5**. Hosted: `gpt-transcribe` for files and `gpt-live-transcribe` for streaming. The only hosted model that serves Whisper weights, `whisper-1`, has a shutdown date.

## How the family evolved

![Layer stacks drawn one block per layer: large, large-v2, large-v3, large-v3-turbo and distil-large-v3 all keep a 32-layer encoder while the decoder goes from 32 to 4 to 2 layers, and the input changes from 80 to 128 mel bins at v3](/imgs/blogs/whisper-model-family-timeline-2.webp)

The stacks make the whole history visible at once: the encoder column never changes height, the decoder column collapses, with `tiny` (4 + 4 layers) drawn alongside for scale and the cut layers shown dashed.

### 1. The encoder was frozen in design for four years

Every large checkpoint has a 32-layer, 1280-wide encoder over 1,500 positions. The only front-end change in four releases is 80 to 128 mel bins. That stability is why the ecosystem could keep up: a port written for large-v1 needed one config change to load v3.

### 2. Data grew 7x, and the model became its own labeler

680k hours to 5M hours, but four-fifths of the increase was large-v2 output. Distil-Whisper used the same move one level down, pseudo-labeling with v2 and then v3. Whisper's real contribution to the field may be this recipe: weak supervision at scale, then self-labeling to scale further.

### 3. Speed came from the decoder, never the encoder

Distil-Whisper cut the decoder to 2 layers in 2023; OpenAI cut it to 4 in 2024 and credited Distil-Whisper. Neither touched the encoder, because the encoder runs once per window while the decoder runs once per token.

### 4. Capabilities narrowed as speed rose

large-v3 does transcription and translation. Turbo drops translation by design. Distil-large-v3 cards target English. The fastest open variants are specialists.

### 5. The license stayed MIT while the frontier went hosted

The open line never changed license, which is rare in this catalog. What changed is that every new OpenAI speech model since March 2025 is API-only, and in August 2026 even the API version of the open weights got a shutdown date.

## Which variant to pick today

| Need | Pick | Why |
| --- | --- | --- |
| Multilingual transcription, one GPU, best speed for accuracy | large-v3-turbo | ~8x large's speed, near-v2 accuracy, MIT |
| Speech translation into English | large-v3 (or large-v2) | Turbo was not trained on translation |
| Hardest languages, accuracy first | large-v3, then evaluate v2 | v3 averages 10% to 20% better, but OpenAI warns per-language results vary |
| English only, fastest | distil-large-v3.5 | 1.46x turbo's RTF, better short-form OOD WER |
| CPU, Mac or phone | whisper.cpp with turbo or a quantized smaller size | ggml runtime, actively released |
| Batch transcription on GPU | faster-whisper with int8 and batching | 13 minutes of audio in 16 s in its own benchmark |
| Word timestamps, long recordings | WhisperX | VAD segmentation plus forced alignment |
| Tiny footprint, English | tiny.en or base.en | `.en` helps most at these sizes |

If you are fine-tuning rather than picking, [Debugging ASR Finetuning](/blog/machine-learning/debugging-training/debugging-asr-finetuning) covers the Whisper-specific failure modes. If you are open to non-Whisper models, compare against [Qwen3-ASR](/blog/machine-learning/frontier-models/qwen3-asr-technical-report) and [NVIDIA's Canary and Parakeet](/blog/machine-learning/deep-learning/canary-parakeet-fastconformer-asr) before defaulting to turbo.

## What this catalog could not verify

- **The exact September 2022 announcement date.** The OpenAI blog was not fetchable; I use the model card ("September 2022"), the repo creation date (September 16) and the Hub creation date (September 26).
- **`whisper-1`'s exact launch day.** Search results quote the launch post's large-v2 and $0.006 per minute details, but I did not fetch a page that shows the date, so I give March 2023.
- **Turbo's parameter count.** The README and Hub card say 809M; `model-card.md` says 798M. I did not count the weights.
- **The Hub license tags.** The `tiny` through `large-v3` Hub repos are tagged `apache-2.0`, while `large-v3-turbo` and the GitHub repo say MIT. A Hub discussion on the turbo repo raises the inconsistency; I found no resolution from OpenAI.
- **gpt-4o-transcribe-diarize's date** comes from SDK release notes, not an OpenAI announcement.
- **Whether `gpt-realtime-whisper` shares anything with the open weights** is not stated anywhere I could find.

## References

1. Radford et al., *Robust Speech Recognition via Large-Scale Weak Supervision*, [arXiv 2212.04356](https://arxiv.org/abs/2212.04356), December 6, 2022.
2. [openai/whisper README](https://github.com/openai/whisper), [model-card.md](https://github.com/openai/whisper/blob/main/model-card.md), [CHANGELOG](https://github.com/openai/whisper/blob/main/CHANGELOG.md) and [releases](https://github.com/openai/whisper/releases).
3. Announcement threads: [large-v2](https://github.com/openai/whisper/discussions/661), [large-v3](https://github.com/openai/whisper/discussions/1762), [turbo](https://github.com/openai/whisper/discussions/2363).
4. Hugging Face model cards and configs: [whisper-large-v2](https://huggingface.co/openai/whisper-large-v2), [whisper-large-v3](https://huggingface.co/openai/whisper-large-v3), [whisper-large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo).
5. Gandhi, von Platen, Rush, *Distil-Whisper*, [arXiv 2311.00430](https://arxiv.org/abs/2311.00430); [distil-large-v3](https://huggingface.co/distil-whisper/distil-large-v3) and [distil-large-v3.5](https://huggingface.co/distil-whisper/distil-large-v3.5) cards.
6. [SYSTRAN/faster-whisper](https://github.com/SYSTRAN/faster-whisper) README and releases.
7. [ggml-org/whisper.cpp](https://github.com/ggml-org/whisper.cpp) releases.
8. Bain et al., *WhisperX*, [arXiv 2303.00747](https://arxiv.org/abs/2303.00747); [m-bain/whisperX](https://github.com/m-bain/whisperX).
9. [OpenAI API changelog](https://developers.openai.com/api/docs/changelog), entries March 20, 2025 through August 26, 2026.
10. [Introducing ChatGPT and Whisper APIs](https://openai.com/index/introducing-chatgpt-and-whisper-apis/), OpenAI, March 2023.
